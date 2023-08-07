from pathlib import Path

import torch
from torchinfo import summary

from fm4ar.utils.config import load_config
from fm4ar.models import create_cf_model
# from fm4ar.dingo.models.df.model import create_df_model

if __name__ == "__main__":

    experiment_dir = Path(
        "/Users/timothy/Desktop/projects/fm4ar/experiments/fm-demo"
    )

    config = load_config(experiment_dir=experiment_dir)

    batch_size = 16
    context_dim = (403, 2)
    theta_dim = 7

    config["model"]["theta_dim"] = theta_dim
    config["model"]["context_dim"] = context_dim

    model = create_cf_model(model_kwargs=config["model"])

    summary(
        model=model,
        input_data=[
            torch.randn(batch_size, ),  # t
            torch.randn(batch_size, theta_dim),  # theta
            torch.randn(batch_size, *context_dim),  # context
        ],
        depth=5,
    )
