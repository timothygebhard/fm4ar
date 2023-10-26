"""
Test that we can construct a flow matching model from a configuration.
"""

import torch

from fm4ar.models.build_model import build_model


def test__npe_model__glasflow() -> None:

    # Define a dummy configuration
    config = {
        "model": {
            "model_type": "npe",
            "theta_dim": 7,
            "context_dim": 403,
            "context_embedding_kwargs": {
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
    theta = torch.randn(10, 7)
    context = torch.randn(10, 403)

    # Make sure we sample from it
    samples = model.sample_batch(context=context)
    assert samples.shape == (10, 7)

    # Make sure we can compute the log probability
    log_prob = model.log_prob_batch(theta=theta, context=context)
    assert log_prob.shape == (10,)

    # Make sure we can sample and compute the log probability
    samples, log_prob = model.sample_and_log_prob_batch(context=context)
    assert samples.shape == (10, 7)
    assert log_prob.shape == (10,)


def test__npe_model__normflows() -> None:

    # Define a dummy configuration
    config = {
        "model": {
            "model_type": "npe",
            "theta_dim": 7,
            "context_dim": 403,
            "context_embedding_kwargs": {
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
    theta = torch.randn(10, 7)
    context = torch.randn(10, 403)

    # Make sure we sample from it
    samples = model.sample_batch(context=context)
    assert samples.shape == (10, 7)

    # Make sure we can compute the log probability
    log_prob = model.log_prob_batch(theta=theta, context=context)
    assert log_prob.shape == (10,)

    # Make sure we can sample and compute the log probability
    samples, log_prob = model.sample_and_log_prob_batch(context=context)
    assert samples.shape == (10, 7)
    assert log_prob.shape == (10,)
