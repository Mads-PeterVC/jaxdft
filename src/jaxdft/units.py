"""Unit conversions, used only at boundaries (input parsing, published data, plots).

Inside the package everything is in atomic units: Hartree and Bohr. Constants are
CODATA 2022 (theory post 1b, 'Units at the boundary').
"""

from __future__ import annotations

import jax.numpy as jnp

BOHR_IN_ANGSTROM = 0.529177210544  # CODATA 2022
HARTREE_IN_EV = 27.211386245981  # CODATA 2022
RYDBERG_IN_HARTREE = 0.5  # exact in atomic units
HARTREE_IN_INVERSE_CM = 219474.63136314  # CODATA 2022, E_h / hc
PROTON_MASS_IN_ELECTRON_MASS = 1836.152673426  # CODATA 2022
HARTREE_IN_JOULE = 4.3597447222060e-18  # CODATA 2022
# Atomic unit of pressure (energy density), E_h / a_0^3, in GPa.
HARTREE_PER_BOHR3_IN_GPA = HARTREE_IN_JOULE / (BOHR_IN_ANGSTROM * 1e-10) ** 3 * 1e-9

# Size of one unit in the atomic unit of its dimension.
_DIMENSIONS = {
    "length": {"Bohr": 1.0, "Angstrom": 1.0 / BOHR_IN_ANGSTROM},
    "energy": {
        "Hartree": 1.0,
        "Rydberg": RYDBERG_IN_HARTREE,
        "eV": 1.0 / HARTREE_IN_EV,
        "cm-1": 1.0 / HARTREE_IN_INVERSE_CM,
    },
    "pressure": {"Hartree/Bohr^3": 1.0, "GPa": 1.0 / HARTREE_PER_BOHR3_IN_GPA},
}


def convert(value, from_unit: str, to_unit: str):
    """Convert ``value`` from ``from_unit`` to ``to_unit`` (case-sensitive names:
    "Bohr", "Angstrom", "Hartree", "Rydberg", "eV", "cm-1", "Hartree/Bohr^3",
    "GPa"). Returns a float64 JAX array."""
    for table in _DIMENSIONS.values():
        if from_unit in table and to_unit in table:
            x = jnp.asarray(value, dtype=jnp.float64)
            if from_unit == to_unit:
                return x
            # Express the factor through the published constant directly where possible,
            # so e.g. 1 Bohr -> Angstrom is exactly BOHR_IN_ANGSTROM.
            return x * (table[from_unit] / table[to_unit])
    known = sorted(u for t in _DIMENSIONS.values() for u in t)
    raise ValueError(f"cannot convert {from_unit!r} to {to_unit!r}; known units: {known}")
