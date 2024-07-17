"""
Unit tests for `fm4ar.utils.plotting`.
"""

from fm4ar.utils.plotting import (
    CBF_COLORS,
    adjust_lightness,
    set_font_family,
)


def test__cbf_colors() -> None:
    """
    Test `CBF_COLORS`.
    """

    assert len(CBF_COLORS) == 6


def test__set_font_family() -> None:
    """
    Test `set_font_family()`.
    """

    set_font_family(font_family=None)
    set_font_family(font_family="Gillius ADF")


def test__adjust_lightness() -> None:
    """
    Test `adjust_lightness()`.
    """

    # Case 1: Lighten color to white
    assert adjust_lightness("#ff0000", 1.0) == (1.0, 1.0, 1.0)

    # Case 2: Darken color to black
    assert adjust_lightness((0, 0, 1), -1.0) == (0.0, 0.0, 0.0)

    # Case 3: Lighten color "halfway"
    assert adjust_lightness("green", 0.5) == (
        0.63195573962033091,
        0.76973352049109811,
        0.62495852322427337,
    )

    # Case 4: Darken color "a bit"
    assert adjust_lightness("HotPink", -0.1) == (
        0.92640900349093425,
        0.38729293836676354,
        0.65546788838070991,
    )
