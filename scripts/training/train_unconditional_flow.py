"""
Train an unconditional normalizing flow on the combined samples from
all methods (nested sampling, FMPE, NPE).

TODO: This should be generalized to work with a configuration file!
"""

import argparse
import time

import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fm4ar.nn.flows import create_unconditional_nsf
from fm4ar.datasets.scaling import get_theta_scaler
from fm4ar.nested_sampling.posteriors import load_posterior
from fm4ar.utils.paths import get_experiments_dir, get_root_dir


def load_samples() -> torch.Tensor:
    """
    Load and combine samples from all methods.
    """

    all_samples = []

    # Load samples from nautilus posterior
    experiment_dir = (
        get_root_dir()
        / "scripts"
        / "nested_sampling"
        / "results"
        / "nautilus"
        / "15_high-res"
    )
    samples, _ = load_posterior(experiment_dir)
    samples = samples[:2_333_333]
    all_samples.append(samples)
    print(f"Loaded {len(samples):,} samples from nautilus posterior.")

    # Load samples from FMPE
    file_path = (
        get_experiments_dir()
        / "aaai-workshop"
        / "fmpe"
        / "importance_sampling"
        / "importance_sampling_results_minimized.hdf"
    )
    with h5py.File(file_path, "r") as f:
        samples = np.array(f["theta"])[:2_333_333]
    all_samples.append(samples)
    print(f"Loaded {len(samples):,} samples from FMPE.")

    # Load samples from NPE
    file_path = (
        get_experiments_dir()
        / "aaai-workshop"
        / "npe"
        / "importance_sampling"
        / "importance_sampling_results_minimized.hdf"
    )
    with h5py.File(file_path, "r") as f:
        samples = np.array(f["theta"])[:2_333_333]
    all_samples.append(samples)
    print(f"Loaded {len(samples):,} samples from NPE.")

    # Combine samples
    samples = np.concatenate(all_samples, axis=0)
    print(f"Loaded {len(samples):,} in total.\n")

    return torch.from_numpy(samples).float()


if __name__ == "__main__":

    script_start = time.time()
    print("\nTRAIN UNCONDITIONAL FLOW\n")

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32_768,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    # Fix random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load combined samples and shuffle them
    samples = load_samples()
    samples = samples[torch.randperm(len(samples))]

    # Get the theta scaler and scale the samples
    print("Scaling the samples...", end=" ", flush=True)
    config = dict(data=dict(name="vasist-2023", theta_scaler="standardizer"))
    scaler = get_theta_scaler(config=config)
    samples = scaler.forward(samples)
    print("Done!")

    # Split into training and validation
    print("Splitting into training and validation...", end=" ", flush=True)
    n = int(len(samples) * 0.95)
    x_train = samples[:n]
    x_test = samples[n:]
    print("Done!\n")

    # Create dataloaders
    train_dataset = TensorDataset(x_train)
    test_dataset = TensorDataset(x_test)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create the flow (with default settings)
    print("Creating the unconditional flow...", end=" ", flush=True)
    model = create_unconditional_nsf()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Done!")
    print(f"Number of trainable parameters: {n_params:,}\n\n")

    # Create an optimizer and a scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True,
    )

    # Keep track of the best model
    best_loss = np.inf

    # Create results directory
    results_dir = (
        get_experiments_dir()
        / "aaai-workshop"
        / "unconditional-flow"
    )
    results_dir.mkdir(exist_ok=True)

    # Train the flow
    print("Running training:\n")
    for i in range(args.epochs):

        # Train the model
        model.train()
        training_losses = []
        with tqdm(train_loader, ncols=80, desc=f"[Train] Epoch {i:4d}") as tq:
            for batch in tq:
                x = batch[0].to(device, non_blocking=True)
                optimizer.zero_grad()
                loss = model.forward_kld(x)
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                    training_losses.append(loss.item())
                    tq.set_postfix(loss=loss.item())
            avg_train_loss = float(np.mean(training_losses))
            tq.set_postfix(loss=avg_train_loss)

        # Test the model
        model.eval()
        test_losses = []
        with tqdm(test_loader, ncols=80, desc=f"[Test]  Epoch {i:4d}") as tq:
            for batch in tq:
                x = batch[0].to(device, non_blocking=True)
                loss = model.forward_kld(x)
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    test_losses.append(loss.item())
                    tq.set_postfix(loss=loss.item())
            avg_test_loss = float(np.mean(test_losses))
            tq.set_postfix(loss=avg_test_loss)

        # Update the learning rate
        print()
        scheduler.step(avg_test_loss)

        # Save the model
        if avg_test_loss < best_loss:
            print("Saving best model...", end=" ", flush=True)
            best_loss = avg_test_loss
            file_path = results_dir / "model__best.pt"
            torch.save(model.state_dict(), file_path)
            print("Done!")

        print()

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
