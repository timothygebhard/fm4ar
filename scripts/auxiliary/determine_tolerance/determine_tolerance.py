"""
Determine required `tolerance` (of the ODE solver) for `sample_batch()`

This script will draw samples with two different tolerances (where one
is usually determined by the tolerance required to get good results for
importance sampling) from an FMPE model and then train a discriminator
to distinguish the two sets of samples. If they discriminator fails to
distinguish between them, the "candidate tolerance" is small enough.
"""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange
from yaml import safe_load

from fm4ar.models.build_model import FMPEModel, build_model
from fm4ar.nn.mlp import MLP
from fm4ar.target_spectrum import load_target_spectrum
from fm4ar.torchutils.dataloaders import get_number_of_workers
from fm4ar.utils.hdf import load_from_hdf, save_to_hdf
from fm4ar.utils.paths import expand_env_variables_in_path as expand_path


def draw_samples_from_fmpe_model(config: dict) -> dict[float, torch.Tensor]:
    """
    Load the FMPE model and draw samples from it (unless there already
    exists an HDF file with the samples, in which case we can just load
    and return those). Returns a mapping from tolerance to the samples.
    """

    tolerances = config["tolerances"]

    # Initialize the model and target spectrum
    model: FMPEModel | None = None
    target_spectrum: dict | None = None

    # Draw samples with different tolerances
    samples: dict[float, torch.Tensor] = {}
    for tolerance in tolerances:

        # Define path where we would expect to find the samples
        samples_file_path = Path(f"tolerance-{tolerance:.0e}.hdf")

        # If the samples already exist, load them
        if samples_file_path.exists():

            print(f"Loading samples with `{tolerance=:.5f}`...", end=" ")
            as_numpy = load_from_hdf(
                file_path=samples_file_path,
                keys=["samples"],
            )
            samples[tolerance] = torch.from_numpy(as_numpy["samples"]).float()
            print("Done!", flush=True)

        # Otherwise, draw the samples from the FMPE model
        else:

            # Load the model and target spectrum if not already loaded
            if model is None or target_spectrum is None:

                # Load the reference spectrum to be used as the context
                print("Loading spectrum...", end=" ", flush=True)
                file_path = Path(config["target_spectrum"]["file_path"])
                target_spectrum = load_target_spectrum(
                    file_path=expand_path(file_path),
                    index=0,
                )
                print("Done!")

                # Load the model onto the GPU
                print("Loading model...", end=" ", flush=True)
                file_path = Path(config["fmpe_model"]["file_path"])
                model = build_model(  # type: ignore
                    experiment_dir=None,
                    file_path=expand_path(file_path),
                    device="cuda",
                )
                assert model is not None  # needed for mypy
                model.network.eval()
                print("Done!\n\n")

            # Construct the basic context (with batch size = 1)
            context = {
                k: torch.from_numpy(target_spectrum[k]).float()
                for k in ["wlen", "flux", "error_bars"]
            }

            # Repeat the context to create a batch; move to GPU
            chunk_size = config["dataset"]["chunk_size"]
            chunk_context = {
                k: v.repeat(chunk_size, 1).to("cuda", non_blocking=True)
                for k, v in context.items()
            }

            # Determine number of chunks
            n_chunks = (
                config["dataset"]["n_samples"]
                // config["dataset"]["chunk_size"]
            )

            # Draw samples (with a progress bar)
            print(f"Drawing samples with `{tolerance=:.5f}`:\n", flush=True)
            list_of_chunks = []
            with torch.no_grad():
                for _ in trange(n_chunks):
                    with autocast(enabled=config["fmpe_model"]["use_amp"]):
                        chunk = model.sample_batch(
                            context=chunk_context,
                            tolerance=tolerance,
                        )
                    list_of_chunks.append(chunk.cpu())

            # Combine samples and save them to an HDF file
            print("\nSaving samples to HDF...", end=" ", flush=True)
            samples[tolerance] = torch.cat(list_of_chunks, dim=0)
            save_to_hdf(
                file_path=samples_file_path,
                samples=samples[tolerance].numpy(),
            )
            print("Done!", flush=True)

    return samples


def create_dataloaders(
    samples: dict[float, torch.Tensor],
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for the discriminator.
    """

    # Define shortcuts
    tolerances = config["tolerances"]
    n = len(samples[tolerances[0]])
    train_test_split = config["dataset"]["train_test_split"]
    num_workers = get_number_of_workers(config["training"]["n_workers"])

    # Add class labels (0 and 1) to the samples and do train/valid split
    dataset = TensorDataset(
        torch.cat([samples[tolerances[0]], samples[tolerances[1]]], dim=0),
        torch.cat([torch.zeros(n, 1), torch.ones(n, 1)], dim=0),
    )
    train_set, valid_set = random_split(dataset, train_test_split)

    # Construct data loaders for training and validation
    train_loader = DataLoader(
        train_set,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader, valid_loader


def train_discriminator_model(
    config: dict[str, Any],
    discriminator: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
) -> tuple[float, float]:
    """
    Train the discriminator model to distinguish between the two sets
    of samples drawn with different tolerances and return the maximum
    accuracy on the validation set.
    """

    # Create optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        params=discriminator.parameters(),
        lr=config["training"]["lr"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config["training"]["n_epochs"],
    )

    # Define loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Keep track of the maximum accuracies
    max_train_accuracy = 0.0
    max_valid_accuracy = 0.0

    # Run training for the given number of epochs
    print("\n\nTraining the discriminator model:\n")
    for epoch in range(config["training"]["n_epochs"]):

        epoch_start = time()
        print(f"[{epoch + 1:3d}] | ", end="")

        # Train for one epoch
        discriminator.train()
        losses = 0.0
        matches = 0.0
        for x, y_true in train_loader:
            x = x.to("cuda", non_blocking=True)
            y_true = y_true.to("cuda", non_blocking=True)
            optimizer.zero_grad()
            y_pred = discriminator(x)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            scheduler.step()
            y_pred = (torch.sigmoid(y_pred) > 0.5).float()
            matches += (y_pred == y_true).float().mean().item()
        train_loss = losses / len(train_loader)
        accuracy = matches / len(train_loader)
        if accuracy > max_train_accuracy:
            max_train_accuracy = accuracy
        print(f"train_loss: {train_loss:.4f} | ", end="")
        print(f"train_acc: {100 * accuracy:.2f}% | ", end="")

        # Evaluate on the test set
        discriminator.eval()
        losses = 0.0
        matches = 0.0
        with torch.no_grad():
            for x, y_true in valid_loader:
                x = x.to("cuda", non_blocking=True)
                y_true = y_true.to("cuda", non_blocking=True)
                y_pred = discriminator(x)
                losses += criterion(y_pred, y_true).item()
                y_pred = (torch.sigmoid(y_pred) > 0.5).float()
                matches += (y_pred == y_true).float().mean().item()
        valid_loss = losses / len(valid_loader)
        accuracy = matches / len(valid_loader)
        if accuracy > max_valid_accuracy:
            max_valid_accuracy = accuracy
        print(f"valid_loss: {valid_loss:.4f} | ", end="")
        print(f"valid_acc: {100 * accuracy:.2f}% | ", end="")
        print(f"∆t = {time() - epoch_start:.2f}s", flush=True)

    return max_train_accuracy, max_valid_accuracy


def store_max_accuracies(
    config: dict[str, Any],
    max_train_accuracy: float,
    max_valid_accuracy: float,
) -> None:
    """
    Store the maximum accuracies in a CSV file.
    """

    print("\nStoring maximum accuracies in CSV...", end=" ", flush=True)

    # Initialize the CSV file if it does not exist
    file_path = Path(__file__).parent / "accuracies.csv"
    if not file_path.exists():
        with open(file_path, "w") as f:
            f.write("tol_1,tol_2,max_train_acc,max_valid_acc,timestamp\n")

    # Load the existing data, append the new data, and sort by tolerance
    df = pd.read_csv(file_path)
    df.loc[-1] = [
        config["tolerances"][0],
        config["tolerances"][1],
        max_train_accuracy,
        max_valid_accuracy,
        datetime.utcnow().isoformat(),
    ]
    df = df.sort_values(by=["tol_1", "tol_2"]).reset_index(drop=True)

    # Save the updated data to the CSV file
    df.to_csv(file_path, index=False)

    print("Done!", flush=True)


if __name__ == "__main__":

    script_start = time()
    print("\nDETERMINE REQUIRED TOLERANCE FOR SAMPLE()\n")

    # Get command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default="config.yaml",
        required=True,
        help="Path to file containing the settings for this script.",
    )
    args = parser.parse_args()

    # Load the configuration
    with open(args.config, "r") as f:
        config = safe_load(f)

    # Draw samples from the FMPE model (or load them if they exist)
    samples = draw_samples_from_fmpe_model(config=config)

    # Create dataloaders for the discriminator
    train_loader, valid_loader = create_dataloaders(
        samples=samples,
        config=config,
    )

    # Create new model to discriminate between samples
    discriminator = MLP(**config["discriminator_model"]).to("cuda")

    # Train the discriminator model
    # Somehow, the explicit `del` statement is necessary to terminate properly;
    # without it, the script seems to get stuck in a `_clean_up_worker` call?
    max_train_accuracy, max_valid_accuracy = train_discriminator_model(
        config=config,
        discriminator=discriminator,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
    del train_loader, valid_loader

    # Store the maximum accuracies in a CSV file
    store_max_accuracies(
        config=config,
        max_train_accuracy=max_train_accuracy,
        max_valid_accuracy=max_valid_accuracy,
    )

    # Print the total runtime
    print(f"\nThis took {time() - script_start:.1f} seconds!\n")
