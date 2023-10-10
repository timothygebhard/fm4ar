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

import numpy as np
import torch
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torcheval.metrics.functional import r2_score

from tqdm import tqdm

from fm4ar.utils.config import load_config
from fm4ar.nn.resnets import DenseResidualNet
from fm4ar.nn.embedding_nets import create_embedding_net
from fm4ar.datasets import load_dataset
from fm4ar.utils.torchutils import (
    build_train_and_test_loaders,
    get_lr,
    perform_scheduler_step,
)


class ModelSaver:
    """
    Simple wrapper to save the model based on the test loss.
    """

    def __init__(self, pretrain_dir: Path):
        super().__init__()
        self.pretrain_dir = pretrain_dir
        self.best_test_loss = float("inf")

    def __call__(
        self,
        test_loss: float,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
    ) -> None:
        """
        Save the model if the test loss is better than the best so far.
        """

        if test_loss >= self.best_test_loss:
            print()
            return

        print("Saving the trained model...", end=" ")
        self.best_test_loss = test_loss
        file_path = self.pretrain_dir / "encoder__best.pt"
        torch.save(encoder.state_dict(), file_path)
        file_path = self.pretrain_dir / "decoder__best.pt"
        torch.save(decoder.state_dict(), file_path)
        print("Done!\n\n")


def get_optimizer_and_scheduler(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.OneCycleLR]:
    """
    Define an optimizer and a learning rate scheduler.
    """

    # Combine the parameters of the two models
    params = chain(encoder.parameters(), decoder.parameters())

    # Define the optimizer and the scheduler
    optimizer = torch.optim.Adam(
        params=params,
        lr=5.0e-5,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=5.0e-5,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    return optimizer, scheduler


def train_batch(
    x_vanilla: torch.Tensor,
    x_corrupt: torch.Tensor,
    device: torch.device,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    scaler: torch.cuda.amp.GradScaler,
    tq: tqdm,
) -> float:
    """
    Run a training step on a given batch `x` of data.
    """

    optimizer.zero_grad()

    x_vanilla = torch.Tensor(x_vanilla.to(device, non_blocking=True))
    x_corrupt = torch.Tensor(x_corrupt.to(device, non_blocking=True))

    with autocast(enabled=(device.type == "cuda")):
        z = encoder(x_corrupt)
        true_flux = x_vanilla[:, :, 0]
        pred_flux = decoder(z, context=z)
        loss = torch.nn.functional.mse_loss(pred_flux, true_flux)

    # Take a gradient step (with AMP)
    scaler.scale(loss).backward()  # type: ignore
    scaler.unscale_(optimizer)  # type: ignore
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    scaler.step(optimizer)  # type: ignore
    scale = scaler.get_scale()  # type: ignore
    scaler.update()  # type: ignore

    # Take a learning rate step (unless optimizer.step() was skipped; this
    # basically only seems to happen on the first batch of the first epoch?)
    # See: https://discuss.pytorch.org/t/optimizer-step-before-lr-
    #   scheduler-step-error-using-gradscaler/92930
    if not scale > scaler.get_scale():  # type: ignore
        perform_scheduler_step(
            scheduler=scheduler,
            loss=loss,
            end_of="batch",
        )

    tq.set_postfix(loss=loss.item())

    return loss.item()


def train_epoch(
    epoch: int,
    device: torch.device,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    train_loader: DataLoader,
) -> tuple[float, float]:
    """
    Train the model for one epoch.
    """

    encoder.train()
    decoder.train()

    train_start = time.time()
    train_losses = []

    print(f"Training epoch {epoch}:")
    with tqdm(train_loader, unit=" batches", ncols=80) as tq:
        for x_vanilla, x_corrupt in tq:
            loss = train_batch(
                x_vanilla=x_vanilla,
                x_corrupt=x_corrupt,
                device=device,
                encoder=encoder,
                decoder=decoder,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                tq=tq,
            )
            train_losses.append(loss)

    train_time = time.time() - train_start
    avg_train_loss = float(np.mean(train_losses))

    print(f"Mean train loss: {avg_train_loss:.9f}")
    print(f"Training time:   {train_time:.2f} seconds\n")

    return avg_train_loss, train_time


def test_batch(
    x_vanilla: torch.Tensor,
    x_corrupt: torch.Tensor,
    device: torch.device,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    tq: tqdm,
) -> tuple[float, float]:
    """
    Evaluate the model on a batch of data.
    """

    x_vanilla = torch.Tensor(x_vanilla.to(device, non_blocking=True))
    x_corrupt = torch.Tensor(x_corrupt.to(device, non_blocking=True))

    z = encoder(x_corrupt)
    true_flux = x_vanilla[:, :, 0]
    pred_flux = decoder(z, context=z)

    loss = torch.nn.functional.mse_loss(pred_flux, true_flux).item()

    r2_scores = [
        r2_score(input=pred_flux[i], target=true_flux[i]).item()
        for i in range(len(true_flux))
    ]
    r2_scores = [r2 for r2 in r2_scores if np.isfinite(r2)]
    r2 = float(np.median(r2_scores))

    tq.set_postfix(loss=loss, r2=r2)

    return loss, r2


def test_epoch(
    epoch: int,
    device: torch.device,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    test_loader: DataLoader,
) -> tuple[float, float, float]:
    """
    Evaluate the model on the test (= validation) set.
    """

    encoder.eval()
    decoder.eval()

    test_losses = []
    r2_scores = []
    test_start = time.time()

    print(f"Test epoch {epoch}:")
    with (
        torch.no_grad(),
        tqdm(test_loader, unit=" batches", ncols=80) as tq,
    ):
        for x_vanilla, x_corrupt in tq:
            loss, r2 = test_batch(
                x_vanilla=x_vanilla,
                x_corrupt=x_corrupt,
                device=device,
                encoder=encoder,
                decoder=decoder,
                tq=tq,
            )
            test_losses.append(loss)
            r2_scores.append(r2)

    test_time = time.time() - test_start
    avg_test_loss = float(np.mean(test_losses))
    avg_test_r2_score = float(np.mean(r2_scores))

    print(f"Mean test loss:  {avg_test_loss:.9f}")
    print(f"Mean R2 score:   {avg_test_r2_score:.3f}")
    print(f"Test time:       {test_time:.2f} seconds\n")

    return avg_test_loss, avg_test_r2_score, test_time


def log_metrics(
    epoch: int,
    train_loss: float,
    test_loss: float,
    learning_rate: float,
    train_time: float,
    test_time: float,
    test_r2_score: float,
    # pretrain_dir: Path,
    use_wandb: bool = True,
) -> None:
    """
    Log metrics to wandb.
    """

    # Log to local file
    # write_history(
    #     experiment_dir=pretrain_dir,
    #     epoch=epoch,
    #     train_loss=train_loss,
    #     test_loss=test_loss,
    #     learning_rates=[learning_rate],
    #     extra_info={
    #         "train_time": train_time,
    #         "test_time": test_time,
    #         "test_r2_score": test_r2_score,
    #     },
    #     overwrite=True,
    # )

    # Log to wandb
    if use_wandb:
        wandb.log(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_time": train_time,
                "test_time": test_time,
                "test_r2_score": test_r2_score,
            }
        )


if __name__ == "__main__":

    script_start = time.time()
    print("\nPRE-TRAIN TRANSFORMER-BASED CONTEXT EMBEDDING NET\n")

    # Set random seed and float precision
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")  # type: ignore

    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment-dir", type=Path)
    args = parser.parse_args()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define some shortcuts
    num_workers = 4 if device.type == "cuda" else 0
    use_wandb = not args.disable_wandb

    # Load the experiment config
    # Note: We set `add_noise_to_flux` to False because the `pretrain_collate`
    # function will take care of adding noise to the corrupted flux.
    config = load_config(experiment_dir=args.experiment_dir)
    config["data"]["add_noise_to_flux"] = False

    # Create extra directory for pretraining
    pretrain_dir = args.experiment_dir / "pretrain"
    pretrain_dir.mkdir(exist_ok=True)

    # Load the dataset and define data loaders
    dataset = load_dataset(config=config)
    train_loader, test_loader = build_train_and_test_loaders(
        dataset=dataset,
        train_fraction=config["data"]["train_fraction"],
        batch_size=args.batch_size,
        num_workers=num_workers,
        train_collate_fn="collate_pretrain",
        test_collate_fn="collate_pretrain",
    )
    parameter_names = (
        dataset.names if dataset.names is not None
        else [f"Parameter {i}" for i in range(dataset.theta_dim)]
    )

    # Construct the embedding network
    encoder_kwargs = config["model"]["context_embedding_kwargs"]
    encoder, output_dim = create_embedding_net(
        input_dim=dataset.context_dim,
        embedding_net_kwargs=encoder_kwargs,
    )
    encoder = encoder.to(device)

    # Construct the decoder (which we won't really need after training)
    decoder_kwargs = config["model"]["decoder_kwargs"]
    decoder = DenseResidualNet(**decoder_kwargs).to(device)

    # Define an optimizer and a learning rate scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(encoder, decoder)

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="fm4ar",
            dir=pretrain_dir,
            group="pretrain_transformer",
            config={
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "experiment_dir": args.experiment_dir,
                "dataset": config["data"],
                "embedding_net_kwargs": encoder_kwargs,
                "decoder_kwargs": decoder_kwargs,
            }
        )
        print("\n")

    # Create a model saver and a scaler for AMP
    model_saver = ModelSaver(pretrain_dir=pretrain_dir)
    scaler = GradScaler(enabled=(device.type == "cuda"))  # type: ignore

    # Run the training loop
    for epoch in range(1, args.epochs + 1):

        # Train for one epoch
        train_loss, train_time = train_epoch(
            epoch=epoch,
            device=device,
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
        )

        # Evaluate on the test (= validation) set
        test_loss, test_r2_score, test_time = test_epoch(
            epoch=epoch,
            device=device,
            encoder=encoder,
            decoder=decoder,
            test_loader=test_loader,
        )

        # Save the model if the test loss is better than the best so far
        model_saver(test_loss=test_loss, encoder=encoder, decoder=decoder)

        # Log metrics to wandb
        log_metrics(
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            learning_rate=get_lr(optimizer)[0],
            train_time=train_time,
            test_time=test_time,
            test_r2_score=test_r2_score,
            # pretrain_dir=pretrain_dir,
            use_wandb=use_wandb,
        )

        # Take a learning rate step
        perform_scheduler_step(
            scheduler=scheduler,
            loss=test_loss,
            end_of="epoch",
        )

    if use_wandb:
        wandb.finish()

    print(f"This took {time.time() - script_start:.2f} seconds!\n")
