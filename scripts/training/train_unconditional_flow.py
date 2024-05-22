"""
Train an unconditional normalizing flow on the combined samples from
different methods (nested sampling, FMPE, NPE). This can be used to
obtain a reference posterior.
For the time being, this script provides a rather simple training loop,
without advanced features like automatic restarts to limit the runtime.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import wandb
from dynesty.utils import resample_equal
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fm4ar.datasets.theta_scalers import get_theta_scaler
from fm4ar.nested_sampling.posteriors import load_posterior
from fm4ar.nn.flows import create_unconditional_flow_wrapper
from fm4ar.utils.htcondor import (
    create_submission_file,
    condor_submit_bid,
)
from fm4ar.utils.multiproc import get_number_of_available_cores
from fm4ar.torchutils.schedulers import (
    Scheduler,
    get_scheduler_from_config,
    perform_scheduler_step,
)
from fm4ar.torchutils.optimizers import get_lr, get_optimizer_from_config
from fm4ar.unconditional_flow.config import (
    InputFileConfig,
    UnconditionalFlowConfig,
    load_config,
)


def get_cli_arguments() -> argparse.Namespace:
    """
    Get command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--start-submission",
        action="store_true",
        help="Create submit file and launch training as an HTCondor job.",
    )
    args = parser.parse_args()
    return args


def load_samples(input_files: list[InputFileConfig]) -> torch.Tensor:
    """
    Load and combine samples from all methods.
    """

    all_samples = []

    for input_file in input_files:

        print(f"Loading samples from {input_file.file_path.name}...", end=" ")

        # We need to distinguish between samples from nested sampling (which
        # we can load using the `load_posterior()` convenience function) and
        # samples from FMPE or NPE (which we can load using `h5py`).
        if input_file.file_type == "ns":
            experiment_dir = input_file.file_path.parent
            samples, weights = load_posterior(experiment_dir=experiment_dir)
            weights = weights / weights.sum()
            samples = resample_equal(samples, weights)
            samples = samples[: input_file.n_samples]
        elif input_file.file_type == "ml":
            with h5py.File(input_file.file_path, "r") as f:
                samples = np.array(f["theta"][: input_file.n_samples])
        else:
            raise ValueError(f"Unknown file type: {input_file.file_type}!")
        all_samples.append(samples)

        print(f"Done! ({len(samples):,} samples loaded)")

    # Combine all samples into a torch tensor
    samples = np.concatenate(all_samples, axis=0)

    return torch.from_numpy(samples).float()


def prepare_and_launch_job(
    args: argparse.Namespace,
    config: UnconditionalFlowConfig,
) -> None:
    """
    Prepare and launch the job as an HTCondor job.
    """

    # Create a directory for the logs
    log_dir = args.experiment_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Collect the arguments for the job
    htcondor_config = config.htcondor.copy()
    htcondor_config.arguments = [
        Path(__file__).as_posix(),
        f"--experiment-dir {args.experiment_dir}",
    ]

    # Create the submit file
    file_path = create_submission_file(
        htcondor_config=htcondor_config,
        experiment_dir=args.experiment_dir,
    )

    # Submit the job
    condor_submit_bid(
        file_path=file_path,
        bid=htcondor_config.bid,
    )


def prepare_data(
    config: UnconditionalFlowConfig,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Load samples and create dataloaders for training and validation.
    """

    # Load combined samples and shuffle them
    print("Loading samples:", flush=True)
    samples = load_samples(input_files=config.input_files)
    samples = samples[torch.randperm(len(samples))]
    print(f"Loaded {len(samples):,} in total.\n")

    # Get the theta scaler and scale the samples
    print("Rescaling the samples...", end=" ", flush=True)
    scaler = get_theta_scaler(config=config.theta_scaler)
    samples = scaler.forward_tensor(samples)
    print("Done!")

    # Shuffle the samples
    print("Shuffling the samples...", end=" ", flush=True)
    samples = samples[torch.randperm(len(samples))]
    print("Done!")

    # Split into training and validation
    print("Splitting into training and validation...", end=" ", flush=True)
    n = int(len(samples) * config.training.train_fraction)
    theta_train = samples[:n]
    theta_valid = samples[n:]
    print("Done!")

    # Create dataloaders
    print("Creating dataloaders...", end=" ", flush=True)
    train_loader = DataLoader(
        dataset=TensorDataset(theta_train),
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=get_number_of_available_cores(),
        pin_memory=True,
    )
    valid_loader = DataLoader(
        dataset=TensorDataset(theta_valid),
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=get_number_of_available_cores(),
        pin_memory=True,
    )
    print("Done!\n")

    return train_loader, valid_loader, samples


def prepare_model_optimizer_scheduler(
    config: UnconditionalFlowConfig,
    dim_theta: int,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, Scheduler]:
    """
    Prepare the model, optimizer, and scheduler.
    """

    # Create the flow (with default settings)
    print("Creating the unconditional flow...", end=" ", flush=True)
    model = create_unconditional_flow_wrapper(
        dim_theta=dim_theta,
        flow_wrapper_config=config.model.flow_wrapper,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Done!")
    print(f"Number of trainable parameters: {n_params:,}\n")

    # Create an optimizer and a scheduler
    print("Creating optimizer and LR scheduler...", end=" ", flush=True)
    optimizer = get_optimizer_from_config(
        model_parameters=model.parameters(),
        optimizer_config=config.training.optimizer,
    )
    scheduler = get_scheduler_from_config(
        optimizer=optimizer,
        scheduler_config=config.training.scheduler,
    )
    print("Done!\n\n")

    return model, optimizer, scheduler


def prepare_wandb(
    args: argparse.Namespace,
    config: UnconditionalFlowConfig,
) -> bool:
    """
    Initialize Weights & Biases and define metrics, if desired.
    """

    # Pop the `enable` key from the configuration
    use_wandb = bool(config.wandb.pop("enable", True))

    # Initialize Weights & Biases and define metrics, if desired
    if use_wandb:
        wandb.init(
            config=config.model.dict(),
            dir=args.experiment_dir,
            **config.wandb,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    return use_wandb


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Scheduler,
    train_loader: torch.utils.data.DataLoader,
    dim_theta: int,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train the model for one epoch.
    """

    model.train()
    training_losses = []

    # Prepare sigma for adding noise to the samples
    if config.training.add_noise is None:
        sigma = torch.zeros(dim_theta)
    elif isinstance(config.training.add_noise, float):
        sigma = config.training.add_noise * torch.ones(dim_theta)
    else:
        sigma = torch.tensor(config.training.add_noise)
    sigma = sigma.float().to(device, non_blocking=True)

    with tqdm(
        iterable=train_loader,
        ncols=80,
        desc=f"[Train]    Epoch {epoch:4d}",
    ) as progressbar:

        for batch in progressbar:

            theta = batch[0].to(device, non_blocking=True)
            theta = theta + sigma * torch.randn_like(theta)

            optimizer.zero_grad()
            loss = -model.log_prob(theta=theta).mean()

            if ~(torch.isnan(loss) | torch.isinf(loss)):

                loss.backward()
                if config.training.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=model.parameters(),
                        max_norm=config.training.gradient_clipping,
                    )
                optimizer.step()

                training_losses.append(loss.item())
                progressbar.set_postfix(loss=loss.item())

                perform_scheduler_step(
                    scheduler=scheduler,
                    end_of="batch",
                    loss=loss.item(),
                )

        avg_train_loss = float(np.mean(training_losses))
        progressbar.set_postfix(loss=avg_train_loss)

    return avg_train_loss


def valid_epoch(
    model: torch.nn.Module,
    scheduler: Scheduler,
    valid_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Perform a single validation epoch.
    """

    model.eval()
    valid_losses = []

    with tqdm(
        iterable=valid_loader,
        ncols=80,
        desc=f"[Validate] Epoch {epoch:4d}",
    ) as progressbar:

        for batch in progressbar:

            theta = batch[0].to(device, non_blocking=True)
            with torch.no_grad():
                loss = -model.log_prob(theta=theta).mean()

            if ~(torch.isnan(loss) | torch.isinf(loss)):
                valid_losses.append(loss.item())
                progressbar.set_postfix(loss=loss.item())
            else:
                print("NaN or Inf loss encountered!", flush=True)

        avg_valid_loss = float(np.mean(valid_losses))
        progressbar.set_postfix(loss=avg_valid_loss)

    # Perform a scheduler step (for ReduceLROnPlateau scheduler)
    perform_scheduler_step(
        scheduler=scheduler,
        end_of="epoch",
        loss=avg_valid_loss,
    )

    return avg_valid_loss


def run_training_loop(config: UnconditionalFlowConfig) -> None:
    """
    Run the training loop.
    """

    # Set up Weights & Biases, if desired
    use_wandb = prepare_wandb(args=args, config=config)

    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load the data and prepare the dataloaders
    train_loader, valid_loader, samples = prepare_data(config=config)
    dim_theta = samples.shape[1]

    # Prepare the model, optimizer, and scheduler
    model, optimizer, scheduler = prepare_model_optimizer_scheduler(
        config=config,
        dim_theta=dim_theta,
        device=device,
    )

    # Keep track of the best model
    early_stopping_counter = 0
    best_loss = np.inf

    # Train the model for the specified number of epochs
    print("Running training:\n")
    for epoch in range(config.training.epochs):

        # Train and validate the model
        avg_train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            dim_theta=dim_theta,
            device=device,
            epoch=epoch,
        )
        avg_valid_loss = valid_epoch(
            model=model,
            scheduler=scheduler,
            valid_loader=valid_loader,
            device=device,
            epoch=epoch,
        )

        # Log everything to Weights & Biases, if desired
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "test_loss": avg_valid_loss,
                    "learning_rate": float(get_lr(optimizer)[0]),
                }
            )

        # Save a checkpoint of the best model, if applicable
        if avg_valid_loss < best_loss:
            print("Saving best model...", end=" ", flush=True)
            early_stopping_counter = 0
            best_loss = avg_valid_loss
            file_path = args.experiment_dir / "model__best.pt"
            torch.save(
                {
                    "dim_theta": samples.shape[1],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "epoch": epoch,
                },
                file_path,
            )
            print("Done!")

        # Handle early stopping
        early_stopping_counter += 1
        if early_stopping_counter >= config.training.early_stopping:
            print("Early stopping criterion reached!")
            break

        print()


if __name__ == "__main__":

    script_start = time.time()
    print("\nTRAIN UNCONDITIONAL NORMALIZING FLOW\n")

    # Get command line arguments and load the configuration file
    args = get_cli_arguments()
    config = load_config(experiment_dir=args.experiment_dir)

    # Set global random seed
    torch.manual_seed(config.random_seed)

    # Either prepare and launch the job as an HTCondor job, or train model
    if args.start_submission:
        prepare_and_launch_job(args=args, config=config)
    else:
        run_training_loop(config=config)

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
