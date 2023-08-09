"""
Test that we can construct a flow matching model from a configuration.
"""

from fm4ar.models.build_model import build_model


def test__build_fm_model() -> None:

    # Define a dummy configuration
    config = {
        "model": {
            "model_type": "fm",
            "t_theta_with_glu": True,
            "context_with_glu": False,
            "sigma_min": 0.0001,
            "time_prior_exponent": 0.5,
            "theta_dim": 7,
            "context_dim": 403,
            "context_embedding_kwargs": {
                "01_SoftClip": {
                    "model_type": "SoftClip",
                    "kwargs": {"bound": 10},
                },
                "02_DenseResidualNet": {
                    "model_type": "DenseResidualNet",
                    "kwargs": {
                        "hidden_dims": [256, 128],
                        "activation": "elu",
                        "output_dim": 128,
                        "dropout": 0,
                        "batch_norm": False,
                    },
                },
            },
            "t_theta_embedding_kwargs": {
                "01_PositionalEncoding": {
                    "model_type": "PositionalEncoding",
                    "kwargs": {"n_freqs": 5, "encode_theta": False},
                },
                "02_DenseResidualNet": {
                    "model_type": "DenseResidualNet",
                    "kwargs": {
                        "hidden_dims": [256, 128, 64, 32],
                        "activation": "gelu",
                        "output_dim": 32,
                        "dropout": 0,
                        "batch_norm": False,
                    },
                },
            },
            "posterior_kwargs": {
                "model_type": "DenseResidualNet",
                "kwargs": {
                    "hidden_dims": [128, 64, 32, 16],
                    "activation": "gelu",
                    "dropout": 0.3,
                    "batch_norm": False,
                },
            },
        }
    }

    # Build the model
    model = build_model(config=config)
