"""
This script can be used to pre-train a transformer-based context
embedding model on the training set of the dataset. The idea is to
randomly mask out parts of the spectra and ask a dummy decoder to
predict them from the context embedding.
"""


import argparse
import time
from itertools import chain
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm

from fm4ar.utils.config import load_config
from fm4ar.nn.resnets import DenseResidualNet
from fm4ar.nn.embedding_nets import create_embedding_net
from fm4ar.datasets import load_dataset
from fm4ar.utils.torchutils import build_train_and_test_loaders


if __name__ == "__main__":

    script_start = time.time()
    print("\nPRE-TRAIN TRANSFORMER-BASED CONTEXT EMBEDDING NET\n")

    # Set random seed and float precision
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")  # type: ignore

    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
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
        collate_fn="collate_and_corrupt",
    )
    parameter_names = (
        dataset.names if dataset.names is not None
        else [f"Parameter {i}" for i in range(dataset.theta_dim)]
    )

    # Construct the embedding network
    context_embedding_net, output_dim = create_embedding_net(
        input_dim=dataset.context_dim,
        embedding_net_kwargs=config["model"]["context_embedding_kwargs"],
    )
    context_embedding_net = context_embedding_net.to(device)

    # Construct the decoder (which we won't really need after training)
    decoder = DenseResidualNet(
        input_dim=1,  # wavelength
        output_dim=1,  # flux
        hidden_dims=(2, 8, 32, 128, 512, 2048, 512, 128, 32, 8, 2),
        context_features=output_dim,
        activation="elu",
        dropout=0.1,
        batch_norm=False,
    ).to(device)

    # Define an optimizer and a learning rate scheduler
    optimizer = torch.optim.Adam(
        params=chain(context_embedding_net.parameters(), decoder.parameters()),
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
        context_embedding_net.train()
        decoder.train()
        print(f"Training epoch {epoch}:")
        train_losses = []
        with tqdm(train_loader, unit=" batches", ncols=80) as tq:

            # Note: For pre-training, we do not need theta --- we only try to
            # get the transformer to understand the structure of a spectrum.
            for _, x in tq:

                optimizer.zero_grad()

                # Move data to the device
                batch_size, n_bins, _ = x.shape
                x = x.to(device, non_blocking=True)

                # Create a random mask to select a subset of the wavelengths.
                mask: torch.Tensor = torch.Tensor(torch.rand(n_bins) > 0.9)
                n_pred = (~mask).sum()

                with autocast(enabled=(device.type == "cuda")):

                    # Get the context embedding = representations of spectra
                    # We reshape this to have one context for each wavelength
                    # of each spectrum in the batch (see below).
                    context = x[:, mask, :]
                    z = (
                        context_embedding_net(context)
                        .unsqueeze(1)
                        .repeat(1, n_pred, 1)
                        .reshape(batch_size * n_pred, -1)
                    )

                    # Predict the flux at the masked-out wavelengths.
                    # Note: We need to reshape the wavelength dimension into
                    # the batch dimension for the DenseResidualNet to work.
                    true_flux = x[:, ~mask, 0].reshape(batch_size, n_pred)
                    true_wlen = x[:, ~mask, 1].reshape(batch_size * n_pred, 1)
                    pred_flux = decoder(x=true_wlen, context=z)
                    pred_flux = pred_flux.reshape(batch_size, n_pred)

                    # Compute the loss
                    loss = torch.nn.functional.mse_loss(pred_flux, true_flux)

                # Take a gradient step (with AMP)
                scaler.scale(loss).backward()  # type: ignore
                scaler.unscale_(optimizer)  # type: ignore
                torch.nn.utils.clip_grad_norm_(
                    context_embedding_net.parameters(), 1.0
                )
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                scaler.step(optimizer)  # type: ignore
                scaler.update()  # type: ignore

                # Save the loss
                train_losses.append(loss.item())
                tq.set_postfix(loss=loss.item())

        avg_loss = sum(train_losses) / len(train_losses)
        print(f"Done! Average training loss: {avg_loss:.4f}\n")

        # Evaluate on the validation set
        context_embedding_net.eval()
        decoder.eval()
        print(f"Test epoch {epoch}:")
        test_losses = []
        with (
            torch.no_grad(),
            tqdm(test_loader, unit=" batches", ncols=80) as tq,
        ):
            for theta_true, x in tq:

                batch_size, n_bins, _ = x.shape
                x = x.to(device, non_blocking=True)

                mask = torch.Tensor(torch.rand(n_bins) > 0.9)
                n_pred = (~mask).sum()

                context = x[:, mask, :]
                z = (
                    context_embedding_net(context)
                    .unsqueeze(1)
                    .repeat(1, n_pred, 1)
                    .reshape(batch_size * n_pred, -1)
                )

                true_flux = x[:, ~mask, 0].reshape(batch_size, n_pred)
                true_wlen = x[:, ~mask, 1].reshape(batch_size * n_pred, 1)
                pred_flux = decoder(x=true_wlen, context=z)
                pred_flux = pred_flux.reshape(batch_size, n_pred)

                loss = torch.nn.functional.mse_loss(pred_flux, true_flux)
                test_losses.append(loss.item())

        avg_loss = sum(test_losses) / len(test_losses)
        print(f"Done! Average test loss: {avg_loss:.4f}\n")

        # Take a learning rate step
        scheduler.step()

        # Save the model(s) if the test loss improved
        if avg_loss < best_test_loss:
            print("Saving the trained model...", end=" ")
            file_path = args.experiment_dir / "context_embedding_net__best.pt"
            torch.save(context_embedding_net.state_dict(), file_path)
            file_path = args.experiment_dir / "decoder__best.pt"
            torch.save(decoder.state_dict(), file_path)
            print("Done!\n\n")

    print(f"This took {time.time() - script_start:.2f} seconds!\n", flush=True)
