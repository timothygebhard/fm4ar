"""
Test that we can construct a flow matching model from a configuration.
"""

import torch

from fm4ar.models.build_model import build_model


def test__npe_model__glasflow() -> None:

    # Define sizes
    batch_size = 5
    theta_dim = 16
    n_bins = 947
    n_features = 3

    # Define a dummy configuration
    config = {
        "model": {
            "model_type": "npe",
            "theta_dim": theta_dim,
            "context_dim": (n_bins, n_features),
            "context_embedding_kwargs": {
                "00_DropFeatures": {
                    "model_type": "DropFeatures",
                },
                "01_DenseResidualNet": {
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
            "posterior_kwargs": {
                "flow_library": "glasflow",
                "num_flow_steps": 14,
                "base_transform_type": "rq-coupling",
                "base_transform_kwargs": {
                    "hidden_dim": 512,
                    "num_transform_blocks": 3,
                    "activation": "elu",
                    "dropout_probability": 0.1,
                    "batch_norm": False,
                    "num_bins": 10
                },
            },
        }
    }

    # Build the model
    model = build_model(config=config)

    # Define dummy theta and context
    theta = torch.randn(batch_size, theta_dim)
    context = torch.randn(batch_size, n_bins, n_features)

    # Make sure we sample from it
    samples = model.sample_batch(context=context)
    assert samples.shape == (batch_size, theta_dim)

    # Make sure we can compute the log probability
    log_prob = model.log_prob_batch(theta=theta, context=context)
    assert log_prob.shape == (batch_size,)

    # Make sure we can sample and compute the log probability
    samples, log_prob = model.sample_and_log_prob_batch(context=context)
    assert samples.shape == (batch_size, theta_dim)
    assert log_prob.shape == (batch_size,)


def test__npe_model__normflows() -> None:

    # Define sizes
    batch_size = 5
    theta_dim = 16
    n_bins = 947
    n_features = 3

    # Define a dummy configuration
    config = {
        "model": {
            "model_type": "npe",
            "theta_dim": theta_dim,
            "context_dim": (n_bins, n_features),
            "context_embedding_kwargs": {
                "00_DropFeatures": {
                    "model_type": "DropFeatures",
                },
                "01_DenseResidualNet": {
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
            "posterior_kwargs": {
                "flow_library": "normflows",
                "num_flow_steps": 14,
                "base_transform_type": "rq-coupling",
                "base_transform_kwargs": {
                    "num_blocks": 3,
                    "num_hidden_channels": 512,
                    "num_bins": 10,
                    "tail_bound": 10,
                    "activation": "elu",
                    "dropout_probability": 0.1
                }
            }
        }
    }

    # Build the model
    model = build_model(config=config)

    # Define dummy theta and context
    theta = torch.randn(batch_size, theta_dim)
    context = torch.randn(batch_size, n_bins, n_features)

    # Make sure we sample from it
    samples = model.sample_batch(context=context)
    assert samples.shape == (batch_size, theta_dim)

    # Make sure we can compute the log probability
    log_prob = model.log_prob_batch(theta=theta, context=context)
    assert log_prob.shape == (batch_size,)

    # Make sure we can sample and compute the log probability
    samples, log_prob = model.sample_and_log_prob_batch(context=context)
    assert samples.shape == (batch_size, theta_dim)
    assert log_prob.shape == (batch_size,)
