"""Goedecker-Teter-Hutter / Hartwigsen-Goedecker-Hutter pseudopotentials, local part
(theory post 2a, 'The GTH local pseudopotential').

    v(r) = -Z/r erf(r / (sqrt2 r_loc)) + exp(-x^2/2) (C1 + C2 x^2 + C3 x^4 + C4 x^6),  x = r/r_loc

Its atomic Fourier transform is analytic. Nonlocal projectors arrive with Milestone 2b;
until then files that have them are refused. Only this module reads the file format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import jax.numpy as jnp

_DATA = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class GTH:
    """Local GTH pseudopotential: ionic charge ``zion``, ``rloc`` (Bohr) and the four
    coefficients ``c`` = (C1, C2, C3, C4) in Ha (missing ones are zero). ``source`` is
    the file it was parsed from, if any: provenance for logs, ignored by equality."""

    zion: float
    rloc: float
    c: tuple = field(default=(0.0, 0.0, 0.0, 0.0))
    source: str | None = field(default=None, compare=False)
    local_support = None  # unbounded: the Coulomb tail

    def __post_init__(self):
        c = tuple(float(x) for x in self.c)
        if len(c) > 4:
            raise ValueError(f"at most four local coefficients, got {len(c)}")
        object.__setattr__(self, "c", c + (0.0,) * (4 - len(c)))
        if not self.rloc > 0:
            raise ValueError(f"rloc must be positive, got {self.rloc}")

    @property
    def ionic_charge(self):
        return self.zion

    @property
    def alpha(self) -> float:
        """lim_{q->0} [v_hat(q) + 4 pi Z / q^2] (Ha Bohr^3)."""
        c1, c2, c3, c4 = self.c
        r = self.rloc
        return 2 * math.pi * self.zion * r**2 + (2 * math.pi) ** 1.5 * r**3 * (c1 + 3 * c2 + 15 * c3 + 105 * c4)

    def local(self, q):
        """v_hat(q) = int v(r) e^{-iq.r} d^3r in Ha Bohr^3, Coulomb tail included for q > 0.
        At q = 0 it returns ``alpha``, the finite non-Coulomb limit, with finite gradients."""
        q = jnp.asarray(q, dtype=jnp.float64)
        c1, c2, c3, c4 = self.c
        r = self.rloc
        y = (q * r) ** 2
        poly = c1 + c2 * (3 - y) + c3 * (15 - 10 * y + y**2) + c4 * (105 - 105 * y + 21 * y**2 - y**3)
        short = (2 * math.pi) ** 1.5 * r**3 * jnp.exp(-y / 2) * poly
        positive = q > 0
        q2 = jnp.where(positive, q**2, 1.0)
        coulomb = jnp.where(positive, -4 * math.pi * self.zion / q2 * jnp.exp(-y / 2), 2 * math.pi * self.zion * r**2)
        return coulomb + short


def parse_hgh(path) -> GTH:
    """Parse a GTH/HGH file in the CP2K-style text format used by PseudoLibrary:

        <element> <names...>
        <electrons per angular channel...>
        <r_loc> <n_exp> <C1> ... <C_nexp>
        <n_projector_channels>
        [projector blocks]
    """
    text = Path(path).read_text()
    lines = [ln.split() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        raise ValueError(f"{path}: expected at least four non-empty lines")
    try:
        zion = sum(int(x) for x in lines[1])
        rloc, nexp = float(lines[2][0]), int(lines[2][1])
        coeffs = tuple(float(x) for x in lines[2][2:])
        nprj = int(lines[3][0])
    except (ValueError, IndexError) as err:
        raise ValueError(f"{path}: malformed GTH file ({err})") from None
    if len(coeffs) != nexp:
        raise ValueError(f"{path}: n_exp = {nexp} but {len(coeffs)} local coefficients given")
    if nprj != 0:
        raise NotImplementedError(f"{path}: nonlocal projectors ({nprj} channels) arrive with Milestone 2b")
    return GTH(zion, rloc, coeffs, source=str(path))


def hgh_lda_path(element: str) -> Path:
    """Path of the vendored HGH LDA file for ``element`` (see data/README.md)."""
    matches = sorted((_DATA / "hgh_lda").glob(f"{element.lower()}-q*.hgh"))
    if len(matches) != 1:
        available = sorted(p.name.split("-")[0].capitalize() for p in (_DATA / "hgh_lda").glob("*.hgh"))
        raise ValueError(f"no vendored HGH LDA pseudopotential for {element!r}; available: {available}")
    return matches[0]


def hgh_lda(element: str) -> GTH:
    """The vendored HGH LDA pseudopotential for ``element``."""
    return parse_hgh(hgh_lda_path(element))
