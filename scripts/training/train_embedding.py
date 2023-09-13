"""
This script can be used to train the context embedding network of a
given experiment to produce a point estimate for the parameters.
Idea: Study if the embedding network is able to learn a representation
that contains all the information needed to predict the parameters.
"""

import argparse
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm

from fm4ar.utils.config import load_config
from fm4ar.nn.resnets import DenseResidualNet
from fm4ar.nn.embedding_nets import create_embedding_net
from fm4ar.datasets import load_dataset
from fm4ar.utils.torchutils import build_train_and_test_loaders


if __name__ == "__main__":

    script_start = time.time()
    print("\nTRAIN CONTEXT EMBEDDING NET\n")

    # Set random seed and float precision
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")  # type: ignore

    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment-dir", type=Path)
    args = parser.parse_args()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4 if device.type == "cuda" else 0

    # Load the experiment config
    config = load_config(experiment_dir=args.experiment_dir)

    # Load the dataset and define data loaders
    dataset = load_dataset(config=config)
    train_loader, test_loader = build_train_and_test_loaders(
        dataset=dataset,
        train_fraction=0.95,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )
    parameter_names = (
        dataset.names if dataset.names is not None
        else [f"Parameter {i}" for i in range(dataset.theta_dim)]
    )

    # Define the model: context embedding net + some linear layers
    context_embedding_net, output_dim = create_embedding_net(
        input_dim=dataset.context_dim,
        embedding_net_kwargs=config["model"]["context_embedding_kwargs"],
    )
    model = torch.nn.Sequential(
        OrderedDict(
            context_embedding_net=context_embedding_net,
            activation=torch.nn.ELU(),
            dense_resnet=DenseResidualNet(
                input_dim=output_dim,
                output_dim=dataset.theta_dim,
                hidden_dims=(64, 32, 16, 8),
                activation="elu",
                dropout=0.0,
                batch_norm=False,
            ),
        )
    ).to(device)

    # Define an optimizer and a learning rate scheduler
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=5.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.epochs,
        eta_min=5.0e-7,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))  # type: ignore

    # Keep track of the best test loss
    best_test_loss = float("inf")

    # Run the training loop
    for epoch in range(1, args.epochs + 1):

        # Train for one epoch
        model.train()
        print(f"Training epoch {epoch}:")
        train_losses = []
        with tqdm(train_loader, unit=" batches", ncols=80) as tq:
            for theta_true, x in tq:

                # Compute the loss for this batch
                optimizer.zero_grad()
                theta_true = theta_true.to(device, non_blocking=True)
                theta_pred = model(x.to(device, non_blocking=True))
                with autocast(enabled=(device.type == "cuda")):
                    loss = torch.nn.functional.mse_loss(theta_pred, theta_true)

                # Take a gradient step (with AMP)
                scaler.scale(loss).backward()  # type: ignore
                scaler.unscale_(optimizer)  # type: ignore
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)  # type: ignore
                scaler.update()  # type: ignore

                # Save the loss
                train_losses.append(loss.item())
                tq.set_postfix(loss=loss.item())

        avg_loss = sum(train_losses) / len(train_losses)
        print(f"Done! Average training loss: {avg_loss:.4f}\n")

        # Evaluate on the validation set
        model.eval()
        print(f"Test epoch {epoch}:")
        test_losses = []
        list_of_rel_errors = []
        with (
            torch.no_grad(),
            tqdm(test_loader, unit=" batches", ncols=80) as tq,
        ):
            for theta_true, x in tq:

                # Compute the loss for this batch
                theta_true = theta_true.to(device, non_blocking=True)
                theta_pred = model(x.to(device, non_blocking=True))
                loss = torch.nn.functional.mse_loss(theta_pred, theta_true)
                test_losses.append(loss.item())
                tq.set_postfix(loss=loss.item())

                # Compute the relative errors
                # We are computing this in the normalized parameter space for
                # theta, which includes 0, hence we add 1 to the denominator.
                # Even in the unnormed parameter space, however, there are
                # parameters that can take on a true value of 0...
                # TODO: Is there a better metric that we could use here?
                rel_error = (
                    torch.abs(theta_pred - theta_true)
                    / (1 + torch.abs(theta_true))
                ).cpu().numpy()
                list_of_rel_errors.append(rel_error)

        avg_loss = sum(test_losses) / len(test_losses)
        print(f"Done! Average test loss: {avg_loss:.4f}\n")

        # Compute the average relative error
        rel_errors = np.concatenate(list_of_rel_errors, axis=0)
        statistics = {
            "Parameter": parameter_names,
            "median": np.median(rel_errors, axis=0),
            "mean": np.mean(rel_errors, axis=0),
            "std": np.std(rel_errors, axis=0),
            "min": np.min(rel_errors, axis=0),
            "max": np.max(rel_errors, axis=0),
        }
        print("Average relative errors:\n")
        print(tabulate(statistics, headers="keys", floatfmt=".3f"))
        print()

        # Take a learning rate step
        scheduler.step()

        # Save the model if it is the best so far
        if avg_loss < best_test_loss:
            print("Saving the trained model...", end=" ")
            file_path = args.experiment_dir / "dummy_model__best.pt"
            torch.save(model.state_dict(), file_path)
            print("Done!\n\n")

    print(f"This took {time.time() - script_start:.2f} seconds!\n", flush=True)
