from pathlib import Path

from fm4ar.utils.config import load_config
from fm4ar.datasets.dataset import load_dataset
from fm4ar.utils.torchutils import build_train_and_test_loaders

if __name__ == "__main__":

    # Load the configuration
    config = load_config(
        Path("/experiments/_other_/fm-demo")
    )

    # Load dataset
    dataset = load_dataset(config)

    # Create dataloaders
    train_loader, test_loader = build_train_and_test_loaders(
        dataset=dataset,
        train_fraction=0.95,
        batch_size=17,
        num_workers=0,
        drop_last=True,
    )

    # Iterate over batches
    print("Train loader:")
    for _, batch in enumerate(train_loader):
        theta, spectra = batch
        print("theta:", theta.shape)
        print("spectra:", spectra.shape)
        break

    print("Test loader:")
    for _, batch in enumerate(test_loader):
        theta, spectra = batch
        print("theta:", theta.shape)
        print("spectra:", spectra.shape)
        break
