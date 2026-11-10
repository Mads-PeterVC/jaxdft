"""The grid rule (theory post 02a, 'Net force and the egg-box effect'; decision 8): the
egg-box effect is a test of the no-wraparound rule, with an undersized grid as the
positive control. Thresholds and measured values are in support/tolerances.py."""

import numpy as np
import pytest

from support.h2 import L_TEST, basis_for, h2, run, translated
from support.tolerances import (
    EGGBOX_RULE_MAX,
    EGGBOX_UNDERSIZED_MIN,
    GRID_ENERGY_RULE_MAX,
    GRID_ENERGY_UNDERSIZED_MIN,
)

pytestmark = pytest.mark.slow
TOL = 1e-10


def eggbox_peak_to_peak(n):
    """Translate the molecule along x across one grid spacing in 8 steps."""
    s = h2()
    b = basis_for(s, fft_shape=(n, n, n), allow_aliasing=n < 30)
    spacing = L_TEST / n
    e = [float(run(translated(s, [k * spacing / 8, 0, 0]), b, tol=TOL).total_energy) for k in range(8)]
    return np.ptp(e)


def test_eggbox_is_noise_with_the_rule_grid():
    assert basis_for(h2()).fft_shape == (30, 30, 30)
    assert eggbox_peak_to_peak(30) < EGGBOX_RULE_MAX


def test_eggbox_is_large_on_an_undersized_grid():
    """Positive control: review measured 8.4e-6 Ha at N = 24."""
    assert eggbox_peak_to_peak(24) > EGGBOX_UNDERSIZED_MIN


def test_energy_insensitive_to_grid_above_the_rule_only():
    s = h2()
    e = {n: float(run(s, basis_for(s, fft_shape=(n,) * 3, allow_aliasing=n < 30), tol=TOL).total_energy) for n in (24, 30, 36)}
    assert abs(e[36] - e[30]) < GRID_ENERGY_RULE_MAX
    assert abs(e[24] - e[30]) > GRID_ENERGY_UNDERSIZED_MIN
