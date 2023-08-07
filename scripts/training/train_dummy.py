"""
Minimal, self-contained training script to learn a dummy model.
"""

import argparse
import time
from pathlib import Path

import torch

from tqdm import tqdm

# from fm4ar.dingo.nn.embedding_nets import ConvNet
from fm4ar.nn import DenseResidualNet
from fm4ar.datasets import load_dataset
from fm4ar.utils.torchutils import build_train_and_test_loaders


if __name__ == "__main__":
    script_start = time.time()
    print("\nTRAIN DUMMY MODEL\n", flush=True)

    # Set random seed
    torch.manual_seed(0)

    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=".")
    args = parser.parse_args()

    # Load the dataset
    config = {
        "data": {
            "name": "ardevol-martinez-2022",
            "which": "training",
            "type": "type-1",
            "instrument": "NIRSPEC",
            "standardize_theta": True,
            "standardize_x": True,
            "add_noise_to_x": True,
        }
    }
    dataset = load_dataset(config=config)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4 if device.type == "cuda" else 0

    # Create data loaders
    train_loader, test_loader = build_train_and_test_loaders(
        dataset=dataset,
        train_fraction=0.95,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    # Define a minimal network
    # model = ConvNet(
    #     input_channels=1,
    #     input_dim=dataset.context_dim[0],
    #     output_dim=dataset.theta_dim,
    # )
    model = DenseResidualNet(
        input_dim=dataset.context_dim[0],
        output_dim=dataset.theta_dim,
        hidden_dims=(
            1024,
            512,
            256,
            128,
            64,
            32,
            32,
            16,
            16,
            16,
            16,
            16,
        ),
        dropout=0.1,
        batch_norm=False,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Define an optimizer and a learning rate scheduler
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=3.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    # Print some general information
    print(f"Training samples:   {len(train_loader.dataset):,}")  # type: ignore
    print(f"Test samples:       {len(test_loader.dataset):,}")  # type: ignore
    print(f"Number of epochs:   {args.epochs:,}")
    print(f"Batch size:         {args.batch_size:,}")
    print(f"Device:             {device}")
    print(f"Model parameters:   {n_params:,}\n")

    # Run the training loop
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        model.train()
        print(f"Training epoch {epoch}:")
        train_losses = []
        with tqdm(train_loader, unit=" batches", ncols=80) as tq:
            for theta_true, spectrum in tq:
                optimizer.zero_grad()
                # spectrum = spectrum.unsqueeze(1)
                theta_true = theta_true.to(device)
                theta_pred = model(spectrum.to(device))
                loss = torch.nn.functional.mse_loss(theta_pred, theta_true)
                loss.backward()  # type: ignore
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())
                tq.set_postfix(loss=loss.item())
        avg_loss = sum(train_losses) / len(train_losses)
        print(f"Done! Average training loss: {avg_loss:.4f}\n")

        # Evaluate on the test set
        model.eval()
        print(f"Test epoch {epoch}:")
        test_losses = []
        with tqdm(test_loader, unit=" batches", ncols=80) as tq:
            for theta_true, spectrum in tq:
                with torch.no_grad():
                    # spectrum = spectrum.unsqueeze(1)
                    theta_true = theta_true.to(device)
                    theta_pred = model(spectrum.to(device))
                    loss = torch.nn.functional.mse_loss(theta_pred, theta_true)
                test_losses.append(loss.item())
                tq.set_postfix(loss=loss.item())
        avg_loss = sum(test_losses) / len(test_losses)
        print(f"Done! Average test loss: {avg_loss:.4f}\n")
        scheduler.step(avg_loss)

        # Save a checkpoint every 5 epochs
        if epoch % 5 == 0:
            print("Saving a checkpoint...", end=" ", flush=True)
            file_path = args.output_dir / f"dummy_model__{epoch:03d}.pt"
            torch.save(model.state_dict(), file_path)
            print("Done!\n")

    # Save the trained model
    print("Saving the trained model...", end=" ", flush=True)
    file_path = args.output_dir / "dummy_model__final.pt"
    torch.save(model.state_dict(), file_path)
    print("Done!\n")

    print(f"This took {time.time() - script_start:.2f} seconds!\n", flush=True)
