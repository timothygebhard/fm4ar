"""
Train an unconditional normalizing flow on the combined samples from
different methods (nested sampling, FMPE, NPE). This can be used to
obtain a reference posterior.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fm4ar.datasets.theta_scalers import get_theta_scaler
from fm4ar.nested_sampling.posteriors import load_posterior
from fm4ar.nn.flows import create_unconditional_flow_wrapper
from fm4ar.utils.multiproc import get_number_of_available_cores
from fm4ar.torchutils.schedulers import (
    get_scheduler_from_config,
    perform_scheduler_step,
)
from fm4ar.torchutils.optimizers import get_optimizer_from_config
from fm4ar.unconditional_flow.config import InputFileConfig, load_config


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
            samples, _ = load_posterior(experiment_dir=experiment_dir)
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


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nTRAIN UNCONDITIONAL NORMALIZING FLOW\n")

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory.",
    )
    args = parser.parse_args()

    # Load (and validate) the configuration file
    config = load_config(experiment_dir=args.experiment_dir)

    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # -------------------------------------------------------------------------
    # Prepare training
    # -------------------------------------------------------------------------

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

    # Split into training and validation
    print("Splitting into training and validation...", end=" ", flush=True)
    n = int(len(samples) * config.training["train_fraction"])
    theta_train = samples[:n]
    theta_valid = samples[n:]
    print("Done!")

    # Create dataloaders
    print("Creating dataloaders...", end=" ", flush=True)
    train_loader = DataLoader(
        dataset=TensorDataset(theta_train),
        batch_size=config.training["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=get_number_of_available_cores(),
        pin_memory=True,
    )
    valid_loader = DataLoader(
        dataset=TensorDataset(theta_valid),
        batch_size=config.training["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=get_number_of_available_cores(),
        pin_memory=True,
    )
    print("Done!\n")

    # Create the flow (with default settings)
    print("Creating the unconditional flow...", end=" ", flush=True)
    model = create_unconditional_flow_wrapper(
        dim_theta=samples.shape[1],
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
        optimizer_config=config.training["optimizer"],
    )
    scheduler = get_scheduler_from_config(
        optimizer=optimizer,
        scheduler_config=config.training["scheduler"],
    )
    print("Done!\n\n")

    # Keep track of the best model
    best_loss = np.inf

    # -------------------------------------------------------------------------
    # Train the flow
    # -------------------------------------------------------------------------

    print("Running training:\n")
    for epoch in range(config.training["epochs"]):

        # ---------------------------------------------------------------------
        # Train the model
        # ---------------------------------------------------------------------

        model.train()
        training_losses = []

        with tqdm(
            iterable=train_loader,
            ncols=80,
            desc=f"[Train]    Epoch {epoch:4d}",
        ) as progressbar:

            for batch in progressbar:

                theta = batch[0].to(device, non_blocking=True)
                optimizer.zero_grad()
                loss = -model.log_prob(theta=theta).mean()

                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()  # type: ignore
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

        # ---------------------------------------------------------------------
        # Validate the model
        # ---------------------------------------------------------------------

        model.eval()
        test_losses = []

        with tqdm(
            iterable=valid_loader,
            ncols=80,
            desc=f"[Validate] Epoch {epoch:4d}",
        ) as progressbar:

            for batch in progressbar:

                theta = batch[0].to(device, non_blocking=True)
                loss = -model.log_prob(theta=theta).mean()

                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    test_losses.append(loss.item())
                    progressbar.set_postfix(loss=loss.item())

            avg_test_loss = float(np.mean(test_losses))
            progressbar.set_postfix(loss=avg_test_loss)

        perform_scheduler_step(
            scheduler=scheduler,
            end_of="epoch",
            loss=avg_test_loss,
        )
        print()

        # ---------------------------------------------------------------------
        # Save the model
        # ---------------------------------------------------------------------

        if avg_test_loss < best_loss:
            print("Saving best model...", end=" ", flush=True)
            best_loss = avg_test_loss
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

        print()

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
