"""Goedecker-Teter-Hutter / Hartwigsen-Goedecker-Hutter pseudopotentials (theory post
2a, 'The GTH local pseudopotential'; theory post 2b, 'The separable nonlocal
pseudopotential').

    v(r) = -Z/r erf(r / (sqrt2 r_loc)) + exp(-x^2/2) (C1 + C2 x^2 + C3 x^4 + C4 x^6),  x = r/r_loc

plus separable projectors per angular momentum l (``channels``). All transforms are
analytic. Only this module reads the file format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import jax.numpy as jnp

_DATA = Path(__file__).resolve().parent / "data"


# Normalisation constants of the reduced projectors p_hat_i^l(q) / q^l, divided by
# pi^(5/4) r_l^(l + 3/2) exp(-y/2), y = (q r_l)^2, and the polynomial in y that
# multiplies them (theory post 2b, the projector table; HGH 1998).
_PROJECTOR_TABLE = {
    (0, 1): (4 * math.sqrt(2), (1.0,)),
    (0, 2): (8 * math.sqrt(2 / 15), (3.0, -1.0)),
    (0, 3): (16 / 3 * math.sqrt(2 / 105), (15.0, -10.0, 1.0)),
    (1, 1): (8 / math.sqrt(3), (1.0,)),
    (1, 2): (16 / math.sqrt(105), (5.0, -1.0)),
    (1, 3): (32 / (3 * math.sqrt(1155)), (35.0, -14.0, 1.0)),
    (2, 1): (8 * math.sqrt(2 / 15), (1.0,)),
    (2, 2): (16 / 3 * math.sqrt(2 / 105), (7.0, -1.0)),
}
_MAX_PROJECTORS = {0: 3, 1: 3, 2: 2}


@dataclass(frozen=True)
class HGHChannel:
    """Separable projectors of one angular momentum ``l``: Gaussian radius ``r`` (Bohr)
    and the symmetric coupling matrix ``h`` (Ha, a tuple of tuples, n_i x n_i).

    ``projectors(q2)`` returns the reduced projectors p_hat_i^l(q) / q^l for
    i = 1..n_i, shape (n_i, *q2.shape), in Bohr^(3/2 + l), as functions of q^2:
    polynomials in q^2 times a Gaussian, with no square root, so every derivative is
    finite at q = 0. The cell factor Omega^(-1/2) and the angular part are not
    included (they belong to the nonlocal term)."""

    l: int
    r: float
    h: tuple

    def __post_init__(self):
        l = int(self.l)
        h = tuple(tuple(float(x) for x in row) for row in self.h)
        n = len(h)
        if l not in _MAX_PROJECTORS:
            raise NotImplementedError(f"projectors with l = {l} are not supported (l <= 2)")
        if n > _MAX_PROJECTORS[l]:
            raise NotImplementedError(f"{n} projectors for l = {l}: at most {_MAX_PROJECTORS[l]} are tabulated")
        if any(len(row) != n for row in h):
            raise ValueError(f"coupling matrix for l = {l} is not square: {h}")
        if any(abs(h[i][j] - h[j][i]) > 1e-14 * max(1.0, abs(h[i][j])) for i in range(n) for j in range(n)):
            raise ValueError(f"coupling matrix for l = {l} is not symmetric: {h}")
        if not self.r > 0:
            raise ValueError(f"projector radius must be positive, got {self.r}")
        object.__setattr__(self, "l", l)
        object.__setattr__(self, "r", float(self.r))
        object.__setattr__(self, "h", h)

    @property
    def n_projectors(self) -> int:
        return len(self.h)

    def projectors(self, q2):
        q2 = jnp.asarray(q2, dtype=jnp.float64)
        y = q2 * self.r**2
        gauss = math.pi**1.25 * self.r ** (self.l + 1.5) * jnp.exp(-y / 2)
        rows = []
        for i in range(1, self.n_projectors + 1):
            norm, poly = _PROJECTOR_TABLE[(self.l, i)]
            p = sum(c * y**k for k, c in enumerate(poly))
            rows.append(norm * p * gauss)
        return jnp.stack(rows)


@dataclass(frozen=True)
class GTH:
    """GTH pseudopotential: ionic charge ``zion``, ``rloc`` (Bohr), the four local
    coefficients ``c`` = (C1, C2, C3, C4) in Ha (missing ones are zero) and the nonlocal
    ``channels`` (a tuple of ``HGHChannel``, empty for a local-only potential).
    ``source`` is the file it was parsed from, if any: provenance for logs, ignored by
    equality."""

    zion: float
    rloc: float
    c: tuple = field(default=(0.0, 0.0, 0.0, 0.0))
    channels: tuple = field(default=())
    source: str | None = field(default=None, compare=False)
    local_support = None  # unbounded: the Coulomb tail

    def __post_init__(self):
        c = tuple(float(x) for x in self.c)
        if len(c) > 4:
            raise ValueError(f"at most four local coefficients, got {len(c)}")
        object.__setattr__(self, "c", c + (0.0,) * (4 - len(c)))
        channels = tuple(self.channels)
        if not all(isinstance(ch, HGHChannel) for ch in channels):
            raise TypeError("channels must be HGHChannel instances")
        if len({ch.l for ch in channels}) != len(channels):
            raise ValueError("at most one channel per angular momentum")
        object.__setattr__(self, "channels", channels)
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
        <r_l> <n_l> <h_11> ... <h_1n>        for l = 0, 1, ...
                    <h_22> ... <h_2n>        (upper triangle, n_l lines)
                    ...

    Channels with n_l = 0 are skipped. Lines that remain after the last block are an
    error, so another dialect (e.g. with spin-orbit coefficients) is never misread.
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

    channels = []
    row = 4
    for l in range(nprj):
        try:
            r, n = float(lines[row][0]), int(lines[row][1])
            upper = [[float(x) for x in lines[row][2:]]]
            for i in range(1, n):
                upper.append([float(x) for x in lines[row + i]])
        except (ValueError, IndexError) as err:
            raise ValueError(f"{path}: malformed projector block for l = {l} ({err})") from None
        if any(len(u) != n - i for i, u in enumerate(upper[:n])) or (n == 0 and upper[0]):
            raise ValueError(f"{path}: projector block for l = {l} is not an upper triangle of size {n}")
        row += max(n, 1)
        if n == 0:
            continue
        h = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                h[i][j] = h[j][i] = upper[i][j - i]
        channels.append(HGHChannel(l, r, tuple(tuple(x) for x in h)))
    if row != len(lines):
        raise ValueError(f"{path}: {len(lines) - row} unexpected line(s) after the projector blocks")
    return GTH(zion, rloc, coeffs, tuple(channels), source=str(path))


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
