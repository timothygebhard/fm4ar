"""
Unit tests for `fm4ar.utils.ieee754`.
"""

import torch

from fm4ar.utils.ieee754 import float2bits


def tensor2str(tensor: torch.Tensor) -> str:
    """
    Auxiliary function to convert a bit tensor to a string.
    """

    s = "".join([str(int(x)) for x in tensor[0].reshape(-1,)])
    e = "".join([str(int(x)) for x in tensor[1:9].reshape(-1,)])
    m = "".join([str(int(x)) for x in tensor[9:].reshape(-1,)])

    return f"{s} {e} {m}"


def test__float2bits() -> None:
    """
    Test `fm4ar.utils.ieee754.integer2bit`.
    """

    test_input = torch.tensor(
        [
            -8,
            -1.23456789,
            0.0,
            1.175494350822287508e-38,  # smallest "normal" number
            1.24e-24,
            0.125,
            0.3333333,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            255.0,
            512.0,
            6.791e32,
            3.40282346639e+38,  # largest "normal" float
        ],
        dtype=torch.float32,
    )

    output = float2bits(test_input, precision="single")
    assert output.ndim == 2
    assert output.shape[0] == len(test_input)
    assert output.shape[1] == 32

    expected = [
        "1 10000010 00000000000000000000000",  # -8.0
        "1 01111111 00111100000011001010010",  # -1.23456789
        "0 00000000 00000000000000000000000",  # 0.0
        "0 00000001 00000000000000000000000",  # 1.175494350822287508e-38
        "0 00101111 01111111110000101110110",  # 1.24e-24
        "0 01111100 00000000000000000000000",  # 0.125
        "0 01111101 01010101010101010101010",  # 0.3333333
        "0 01111110 00000000000000000000000",  # 0.5
        "0 01111111 00000000000000000000000",  # 1.0
        "0 10000000 00000000000000000000000",  # 2.0
        "0 10000000 10000000000000000000000",  # 3.0
        "0 10000001 00000000000000000000000",  # 4.0
        "0 10000110 11111110000000000000000",  # 255.0
        "0 10001000 00000000000000000000000",  # 512.0
        "0 11101100 00001011110110111001010",  # 6.791e32
        "0 11111110 11111111111111111111111",  # 3.40282346639e+38
    ]

    for f, g in zip(output, expected, strict=True):
        assert tensor2str(f) == g
