"""
Define a parser for likelihood configurations.
"""

from pydantic import BaseModel, Field


class LikelihoodConfig(BaseModel):
    """
    Configuration for the likelihood function.
    """

    # TODO: We might want to figure out a way to specify generic distributions
    #  for the likelihood function in the configuration file. For now, we just
    #  assume a multivariate normal distribution.

    # TODO: We need to figure out a way to specify generic covariance matrices
    #  in the configuration file. For now, we just assume that the covariance
    #  matrix is given as `sigma ** 2 * np.eye(len(x_obs))`.

    sigma: float = Field(
        ...,
        description="Standard deviation of the likelihood function",
    )
