"""
Utilities for configuring log-probability evaluation during training.
"""

from pydantic import BaseModel, Field


class ODESolverSettings(BaseModel):
    """
    Configuration for the ODE solver used for log-probability evaluation.
    """

    method: str = Field(
        default="dopri5",
        description="The ODE solver method to use."
    )
    tolerance: float = Field(
        default=1e-3,
        description="The tolerance for the ODE solver.",
    )


class LogProbEvaluationConfig(BaseModel):
    """
    Control how the log-prob is evaluated during validation.
    """

    interval: int | None = Field(
        default=None,
        description=(
            "Number of epochs between log-probability calculation. "
            "None means no log-probability calculation."
        ),
    )
    n_samples: int = Field(
        default=1024,
        description=(
            "Number of samples to draw for the log-probability calculation. "
            "Using too large of a value will produce an out-of-memory error."
        ),
    )
    ode_solver: ODESolverSettings = Field(
        default_factory=ODESolverSettings,
        description=(
            "Additional keyword arguments for the `model.log_prob_batch()` "
            "call to specify the solver method and tolerance. This is only "
            "used for FMPE models; NPE models will ignore this setting."
        ),
    )
    use_amp: bool = Field(
        default=False,
        description=(
            "Whether to use automatic mixed precision for the log-probability "
            "calculation. Combining this with the ODE solver is still highly "
            "experimental, therefore this setting is independent from the "
            "AMP setting for the general training loop."
        ),
    )
