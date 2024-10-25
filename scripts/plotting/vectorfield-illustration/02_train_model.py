"""
Train a simple flow matching model for the vectorfield illustration.
"""

import time
import warnings
from pathlib import Path

from threadpoolctl import threadpool_limits

from fm4ar.training.preparation import prepare_new
from fm4ar.training.stages import train_stages
from fm4ar.utils.config import load_config

if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE TRAIN SET FOR VECTOR FIELD ILLUSTRATION\n")

    # Define the experiment dir and load the model config
    experiment_dir = Path(".")
    config = load_config(experiment_dir=experiment_dir)

    # Prepare the model and the dataset
    model, dataset = prepare_new(
        experiment_dir=experiment_dir,
        config=config,
    )

    # Run training
    warnings.filterwarnings("ignore")
    with threadpool_limits(limits=1, user_api="blas"):
        complete = train_stages(model=model, dataset=dataset)

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
