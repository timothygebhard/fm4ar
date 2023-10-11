"""
The utilities in this module can be used to convert a PyTorch tensor of
floats to an IEEE 754 binary representation.
This may be useful when the input data cover a large range of values,
making it difficult to normalize them to a range that is suitable for
a neural network.

The code here was originally inspired by the following GitHub repostory:
https://github.com/KarenUllrich/pytorch-binary-converter

However, most of the code has been re-written at this point to fix
various issues in the original implementation (e.g., handle 0 inputs,
or numbers larger than 512), and to improve the readability and
performance of the code.
"""

from typing import Literal

import torch


def check_bounds(
    f: torch.Tensor,
    precision: Literal["half", "single", "double"] = "single",
) -> None:
    """
    Check if the input tensor is within the bounds of the given
    `precision`, and does not contain subnormal numbers.
    """

    # Get the data type for the given precision
    if precision == "half":
        dtype = torch.float16
    elif precision == "single":
        dtype = torch.float32
    elif precision == "double":
        dtype = torch.float64
    else:
        raise ValueError(f"Unknown precision: {precision}")

    # Check of NaNs and Infs
    if torch.any(torch.isnan(f)):
        raise ValueError("Input tensor contains NaNs!")
    if torch.any(torch.isinf(f)):
        raise ValueError("Input tensor contains Infs!")

    # Ensure that the input tensor does not contain values below the mininum
    if torch.any(torch.Tensor(f < torch.finfo(dtype).min)):
        raise ValueError("Input tensor contains values smaller than min!")

    # Ensure that the input tensor does not contain values above the maximum
    if torch.any(torch.Tensor(f > torch.finfo(dtype).max)):
        raise ValueError("Input tensor contains values larger than max!")

    # Ensure that the input tensor does not contain subnormal numbers
    if torch.any(
        torch.Tensor(torch.abs(f) < torch.finfo(dtype).tiny)
        & torch.Tensor(f != 0.0)
    ):
        raise ValueError(
            f"Input tensor contains subnormal numbers, which is currenly "
            "not supported by this code."
        )


def float2bits(
    f: torch.Tensor,
    precision: Literal["half", "single", "double"] = "single",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Turn a float tensor into the corresponding bit pattern tensor.
    """

    # Make sure the input is valid
    check_bounds(f, precision=precision)

    # Get the number of bits for the given precision
    if precision == "half":
        num_e_bits = 5
        num_m_bits = 10
        bias = 15
    elif precision == "single":
        num_e_bits = 8
        num_m_bits = 23
        bias = 127
    elif precision == "double":
        num_e_bits = 11
        num_m_bits = 52
        bias = 1023
    else:
        raise ValueError(f"Unknown precision: {precision}")

    # Compute the sign bit(s): 0 for positive, 1 for negative numbers
    s = get_sign_bit(f)

    # Compute (raw) exponent and mantissa
    # Note: `torch.frexp()` uses the convention `f = mantissa * 2 ** exponent`
    # with the mantissa in the open interval (-1, 1).
    mantissa, exponent = torch.frexp(torch.abs(f))

    # The IEEE 754 standard assumes the mantissa to be in the interval [1, 2).
    # Therefore, we need to apply some corrections to the results above to get
    # the mantissa in the correct range and apply the bias to the exponent.
    # The `- 1` in the exponent compensates for the `* 2` in the mantissa.
    mantissa = 2 * torch.abs(mantissa)
    exponent = exponent - 1 + bias

    # Special case: Handle zeros
    # Zeros in IEEE 754 are represented with an all-zero exponent and mantissa,
    # which does not work with the correct we applied to the exponent above.
    # Instead of an if/else statement, we simply overwrite the exponent in
    # case `f` is zero. (See the `exponent2bits()` call.)
    is_not_zero = f != 0

    # TODO: Special case: Handle subnormal numbers
    # Subnormal numbers have an all-zero exponent, but a non-zero mantissa.
    # is_subnormal = is_not_zero & (torch.abs(f) < torch.finfo(f.dtype).tiny)

    # Convert adjusted exponent and mantissa to bits
    e = exponent2bits(is_not_zero * exponent, num_bits=num_e_bits)
    m = mantissa2bits(mantissa, num_bits=num_m_bits)

    # Combine sign, exponent, and mantissa bits to get the final result
    b = torch.cat([s, e, m], dim=-1).to(dtype)

    return torch.Tensor(b)


def get_sign_bit(
    f: torch.Tensor,
) -> torch.Tensor:
    """
    Get the sign bit of a float tensor: 0 for positive, 1 for negative.
    """

    return torch.Tensor(f < 0.0).unsqueeze(-1).int()


def mantissa2bits(
    mantissa: torch.Tensor,
    num_bits: int = 23,
) -> torch.Tensor:
    """
    Turn mantissa tensor (which will be in the interval [1, 2)) to bits.
    """

    exponent_bits = torch.arange(num_bits)
    exponent_bits = exponent_bits.repeat(mantissa.shape + (1,))
    out = (mantissa.unsqueeze(-1) * 2**exponent_bits) % 1
    return torch.floor(2 * out).int()


def exponent2bits(
    exponent: torch.Tensor,
    num_bits: int = 8,
) -> torch.Tensor:
    """
    Turn exponent tensor (which should be all integers) to bits.
    """

    # Create a tensor [num_bits - 1, ..., 2, 1, 0] for bitwise right shifts
    shifts = torch.arange(num_bits - 1, -1, -1).unsqueeze(0)

    # Expand the input tensor shape to include an additional dimension.
    # Perform bitwise right shift using broadcasting along this new dimension.
    shifted = exponent.unsqueeze(-1).int() >> shifts

    # Perform bitwise AND operation with 1 to isolate each bit.
    return (shifted & 1).int()
