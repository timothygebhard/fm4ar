"""
Utility functions for plotting.
"""

import colour
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np

# Define colorblind-friendly color palette
# Source: https://github.com/mpetroff/accessible-color-cycles
CBF_COLORS = [
    "#5790fc",  # Blue
    "#f89c20",  # Orange
    "#e42536",  # Red
    "#964a8b",  # Purple
    "#9c9ca1",  # Gray
    "#7a21dd",  # Violet
]


def set_font_family(font_family: str | None = None) -> None:
    """
    Globally set the font family for matplotlib.

    Args:
        font_family: The font family to use (e.g., "Gillius ADF").
            If None, the default font family is used ("Dejavu Sans").
    """

    if font_family is not None:
        plt.rcParams['font.sans-serif'] = font_family
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.it'] = f'{font_family}:italic'
        plt.rcParams['mathtext.bf'] = f'{font_family}:bold'
        plt.rcParams['mathtext.cal'] = f'{font_family}:italic'


def adjust_lightness(
    color: str | tuple[float, float, float],
    amount: float,
    colorspace: str = "JzAzBz",
) -> tuple[float, float, float]:
    """
    Adjust the lightness of a `color` by the given `amount`.

    Args:
        color: The color to adjust.
        amount: The amount by which to adjust the lightness. Must be
            in [-1, 1]. Negative values darken the color, positive
            values lighten the color.
        colorspace: The colorspace in which to perform the adjustment.
            Default: "JzAzBz".

    Returns:
        The adjusted color as an RGB tuple.
    """

    # Make sure the input color is an RGB tuple
    # `cnames` can resolve color strings such as "C0" or "Gold"
    try:
        src = mc.to_rgb(mc.cnames[color])  # type: ignore
    except KeyError:
        src = mc.to_rgb(color)

    # Define the "destination": black for darkening, white for lightening
    dst = (0, 0, 0) if amount < 0 else (1, 1, 1)

    # Perform the lightness adjustment in the specified colorspace by lerping
    # between the source and destination colors and converting back to RGB
    LATENT = colour.convert([src, dst], 'Output-Referred RGB', colorspace)
    gradient = colour.algebra.lerp(
        abs(amount),
        LATENT[0],
        LATENT[1],
    )
    RGB = colour.convert(gradient, colorspace, 'Output-Referred RGB')

    # Clipping is needed to avoid occasional out-of-range values
    R, G, B = tuple(np.clip(RGB, 0, 1))

    return R, G, B
