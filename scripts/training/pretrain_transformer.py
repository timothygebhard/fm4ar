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

from fm4ar.utils.htcondor import (
    CondorSettings,
    create_submission_file,
    condor_submit_bid,
)
from fm4ar.utils.config import load_config
from fm4ar.nn.resnets import DenseResidualNet
from fm4ar.nn.embedding_nets import create_embedding_net
from fm4ar.datasets import load_dataset
from fm4ar.utils.torchutils import (
    build_train_and_test_loaders,
    get_lr,
    get_number_of_model_parameters,
    # load_and_or_freeze_model_weights,
    perform_scheduler_step,
)


def soft_clip(flux: torch.Tensor, bound: float = 100.0) -> torch.Tensor:
    return torch.Tensor(flux / (1 + torch.abs(flux / bound)))


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
    n_batches: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.OneCycleLR]:
    """
    Define an optimizer and a learning rate scheduler.
    """

    # Combine the parameters of the two models
    params = chain(encoder.parameters(), decoder.parameters())

    # Define the optimizer and the scheduler
    optimizer = torch.optim.AdamW(
        params=params,
        lr=5.0e-5,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=5.0e-5,
        epochs=args.epochs,
        steps_per_epoch=n_batches,
        pct_start=0.2,
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
    scaler: GradScaler,
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
        true_flux = soft_clip(x_vanilla[:, :, 0])
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
    scaler: torch.cuda.amp.GradScaler,
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

    print(f"Mean train loss: {avg_train_loss:.3e}")
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
    true_flux = soft_clip(x_vanilla[:, :, 0])
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
        autocast(enabled=(device.type == "cuda")),
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

    print(f"Mean test loss:  {avg_test_loss:.3e}")
    print(f"Mean R2 score:   {avg_test_r2_score:.3f}")
    print(f"Test time:       {test_time:.2f} seconds\n")

    return avg_test_loss, avg_test_r2_score, test_time


def create_submission_file_and_launch_job(args: argparse.Namespace) -> None:
    """
    Create a submission file and launch the job.
    """

    # Ensure the `pretrain` subdirectory exists
    pretrain_dir = args.experiment_dir / "pretrain"
    pretrain_dir.mkdir(exist_ok=True)

    # Collect arguments
    arguments = [
        Path(__file__).resolve().as_posix(),
        f"--experiment-dir {args.experiment_dir}",
        f"--batch-size {args.batch_size}",
        f"--epochs {args.epochs}",
        "--disable-wandb" if args.disable_wandb else "",
    ]

    # Create a submission file
    condor_settings = CondorSettings(
        num_cpus=10,
        num_gpus=1,
        memory_cpus=65_000,
        memory_gpus=85_000,
        arguments=arguments,
        log_file_name="pretrain",
        bid=35,
    )
    file_path = create_submission_file(
        condor_settings=condor_settings,
        experiment_dir=pretrain_dir,
    )

    # Launch the job
    condor_submit_bid(file_path=file_path, bid=condor_settings.bid)


def run_pretraining(args: argparse.Namespace) -> None:
    """
    Run the pre-training.
    """

    # Set random seed and float precision
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")  # type: ignore

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define some shortcuts
    num_workers = 8 if device.type == "cuda" else 0
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

    # Load weights of decoder and freeze them
    # TODO: This needs to be generalized
    # load_and_or_freeze_model_weights(
    #     model=decoder,
    #     freeze_weights=False,
    #     load_weights={
    #         "file_path": (
    #             "/home/tgebhard/projects/fm4ar/experiments/fm/"
    #             "vasist-2023/pretrain-autoencoder-1/pretrain/"
    #             "decoder__best.pt"
    #         ),
    #         "prefix": "",
    #     }
    # )

    # Print the number of trainable parameters
    print("Number of parameters (trainable / total):")
    for name, model in [("Encoder", encoder), ("Decoder", decoder)]:
        total = get_number_of_model_parameters(model)
        trainable = get_number_of_model_parameters(model, (True,))
        print(f"{name}: {trainable:,} / {total:,}")
    print("\n")

    # Define an optimizer and a learning rate scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(
        encoder=encoder,
        decoder=decoder,
        n_batches=len(train_loader),
    )

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
            scaler=scaler,
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
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "learning_rate": get_lr(optimizer)[0],
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_time": train_time,
                    "test_time": test_time,
                    "test_r2_score": test_r2_score,
                }
            )

        # Take a learning rate step
        perform_scheduler_step(
            scheduler=scheduler,
            loss=test_loss,
            end_of="epoch",
        )


if __name__ == "__main__":

    script_start = time.time()
    print("\nPRE-TRAIN CONTEXT EMBEDDING NET\n")

    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--start-submission", action="store_true")
    args = parser.parse_args()

    # Either create a submission file or run the pre-training
    if args.start_submission:
        create_submission_file_and_launch_job(args)
    else:
        run_pretraining(args)

    print(f"This took {time.time() - script_start:.2f} seconds!\n")
