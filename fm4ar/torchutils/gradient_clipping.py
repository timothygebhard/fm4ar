"""
Utilities for configuring gradient clipping.
"""

from pydantic import BaseModel, Field


class GradientClippingConfig(BaseModel):
    """
    Configuration for gradient clipping.
    """

    enabled: bool = Field(
        default=True,
        description="Whether gradient clipping should be enabled.",
    )
    max_norm: float = Field(
        default=1.0,
        description="Maximum norm for the gradients.",
    )
    norm_type: float = Field(
        default=2.0,
        description="Type of the used norm.",
    )
