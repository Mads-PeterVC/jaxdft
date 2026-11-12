"""HGH nonlocal data: the vendored silicon file, the parser's projector blocks and the
reduced projectors (theory post 02b, 'The separable nonlocal pseudopotential',
'Silicon parameters and the protocol'; engineering log, 2b API).

The reference formulas are transcribed independently in ``support/hgh.py``.
"""

import hashlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jaxdft.pseudo.gth import GTH, HGHChannel, hgh_lda, hgh_lda_path, parse_hgh
from support.budget import examples
from support.hgh import TABLE, quadrature_projector, reduced_projector
from support.silicon import A_SI
from support.tolerances import PROJECTOR_RTOL

SI_SHA256 = "9d7cca22d6a7873e001af2c31998cdeb570a7468ae5dacab37f8ae5db2220d4b"


# --- the vendored file --------------------------------------------------------------


def test_vendored_silicon_file_is_pinned():
    """Byte-identical to PseudoLibrary hgh_lda_hgh/si-q4.hgh; the hash is the one in
    references.bib and the data README."""
    assert hashlib.sha256(hgh_lda_path("Si").read_bytes()).hexdigest() == SI_SHA256


def test_silicon_parameters():
    """The table of the theory post, verbatim."""
    si = hgh_lda("Si")
    assert isinstance(si, GTH)
    assert si.zion == 4 and si.ionic_charge == 4
    assert si.rloc == 0.44
    assert tuple(si.c) == (-7.33610297, 0.0, 0.0, 0.0)
    s, p = si.channels
    assert isinstance(s, HGHChannel) and isinstance(p, HGHChannel)
    assert (s.l, s.r) == (0, 0.42273813)
    assert s.h == ((5.90692831, -1.26189397), (-1.26189397, 3.25819622))
    assert (p.l, p.r) == (1, 0.48427842)
    assert p.h == ((2.72701346,),)


def test_off_diagonal_coupling_follows_from_the_diagonal():
    """HGH 1998 fix h12 by h22: -1/2 sqrt(3/5) h22 reproduces the stored value to all
    eight digits, a check of the reading of the upper triangle."""
    s = hgh_lda("Si").channels[0]
    np.testing.assert_allclose(s.h[0][1], -0.5 * math.sqrt(3 / 5) * s.h[1][1], atol=5e-9)


def test_alpha_of_silicon_and_the_g0_constants():
    """alpha = 2 pi Z r^2 + (2 pi)^(3/2) r^3 C1 = -4.976525423 Ha Bohr^3; at
    a = 10.263 Bohr the eigenvalue shift sum alpha / Omega = -0.036829280 Ha and the
    G = 0 energy N_el sum alpha / Omega = -0.294634239 Ha (theory post; review)."""
    si = hgh_lda("Si")
    np.testing.assert_allclose(si.alpha, -4.976525423, atol=1e-9)
    omega = A_SI**3 / 4
    np.testing.assert_allclose(omega, 270.248315, atol=1e-6)
    np.testing.assert_allclose(2 * si.alpha / omega, -0.036829280, atol=1e-9)
    np.testing.assert_allclose(8 * 2 * si.alpha / omega, -0.294634239, atol=1e-9)


def test_hydrogen_has_no_channels():
    assert hgh_lda("H").channels == ()


def test_gth_without_channels_by_default():
    assert GTH(4, 0.44, (-7.0,)).channels == ()


# --- the parser ---------------------------------------------------------------------


def _write(tmp_path, text):
    p = tmp_path / "x.hgh"
    p.write_text(text)
    return p


def test_parser_reads_a_three_projector_block(tmp_path):
    """Upper triangle over three lines, symmetrised."""
    text = (
        "X test\n    2    2\n     0.40000000    1    -5.0\n    1\n"
        "     0.30000000    3     1.0    0.2    0.3\n"
        "                                 2.0    0.4\n"
        "                                        3.0\n"
    )
    (ch,) = parse_hgh(_write(tmp_path, text)).channels
    assert ch.l == 0 and ch.r == 0.3
    assert ch.h == ((1.0, 0.2, 0.3), (0.2, 2.0, 0.4), (0.3, 0.4, 3.0))


def test_parser_reads_a_channel_with_no_projectors(tmp_path):
    """A channel count above the populated channels: n_i = 0 contributes nothing."""
    text = "X test\n    2\n     0.40000000    1    -5.0\n    2\n     0.30000000    1     1.0\n     0.50000000    0\n"
    channels = parse_hgh(_write(tmp_path, text)).channels
    assert [(ch.l, len(ch.h)) for ch in channels] == [(0, 1)]


def test_parser_refuses_l_above_two(tmp_path):
    """Higher l are refused until an element needs them (theory post)."""
    blocks = "".join("     0.30000000    1     1.0\n" for _ in range(4))
    text = "X test\n    2\n     0.40000000    1    -5.0\n    4\n" + blocks
    with pytest.raises(NotImplementedError):
        parse_hgh(_write(tmp_path, text))


def test_parser_refuses_projectors_outside_the_table(tmp_path):
    """(l = 2, i = 3) is not in the theory post's table."""
    text = (
        "X test\n    2\n     0.40000000    1    -5.0\n    3\n"
        "     0.30000000    0\n     0.30000000    0\n"
        "     0.30000000    3     1.0    0.0    0.0\n"
        "                                 1.0    0.0\n"
        "                                        1.0\n"
    )
    with pytest.raises(NotImplementedError):
        parse_hgh(_write(tmp_path, text))


def test_parser_refuses_trailing_lines(tmp_path):
    """Lines it does not understand (e.g. spin-orbit coefficients in another dialect)
    are an error, never silently ignored."""
    text = "X test\n    2\n     0.40000000    1    -5.0\n    1\n     0.30000000    1     1.0\n     0.1   0.2\n"
    with pytest.raises(ValueError):
        parse_hgh(_write(tmp_path, text))


def test_parser_refuses_a_short_upper_triangle(tmp_path):
    text = "X test\n    2\n     0.40000000    1    -5.0\n    1\n     0.30000000    2     1.0    0.2\n"
    with pytest.raises(ValueError):
        parse_hgh(_write(tmp_path, text))


# --- channel validation -------------------------------------------------------------


@pytest.mark.parametrize(
    ("l", "r", "h", "error"),
    [
        (0, 0.3, ((1.0, 0.2), (0.3, 1.0)), ValueError),  # not symmetric
        (0, 0.3, ((1.0, 0.2),), ValueError),  # not square
        (0, -0.3, ((1.0,),), ValueError),  # r <= 0
        (3, 0.3, ((1.0,),), NotImplementedError),  # l > 2
        (2, 0.3, ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)), NotImplementedError),  # (2, 3)
    ],
)
def test_channel_validation(l, r, h, error):
    with pytest.raises(error):
        HGHChannel(l, r, h)


# --- the reduced projectors ---------------------------------------------------------


def _channel(l, n, r):
    return HGHChannel(l, r, tuple(tuple(float(i == j) for j in range(n)) for i in range(n)))


@pytest.mark.parametrize(("l", "i"), TABLE)
@pytest.mark.parametrize("q", [0.3, 1.0, 3.0, 7.0])
def test_projectors_match_quadrature(l, i, q):
    """projectors(q^2) q^l against 4 pi int p j_l r^2 dr, for every tabulated (l, i),
    at the q of the throwaway check (worst relative difference 5e-14)."""
    r = 0.37 + 0.1 * l
    n = 3 if l < 2 else 2
    got = float(_channel(l, n, r).projectors(jnp.asarray(q**2))[i - 1]) * q**l
    ref = quadrature_projector(l, i, r, q)
    np.testing.assert_allclose(got, ref, rtol=PROJECTOR_RTOL, atol=1e-14)


@settings(max_examples=examples(10))
@given(st.sampled_from(TABLE), st.floats(0.2, 0.8), st.floats(0.0, 60.0))
def test_projectors_match_the_theory_table(li, r, q2):
    """The package against an independent transcription of the table."""
    l, i = li
    n = 3 if l < 2 else 2
    got = float(_channel(l, n, r).projectors(jnp.asarray(q2))[i - 1])
    np.testing.assert_allclose(got, reduced_projector(l, i, r, q2), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(("l", "i"), TABLE)
def test_projectors_are_normalised_in_reciprocal_space(l, i):
    """Parseval for p(r) Y_lm, whose transform is (-i)^l p_hat(q) Y_lm:
    (2 pi)^-3 int_0^inf p_hat(q)^2 q^2 dq = int_0^inf p(r)^2 r^2 dr = 1."""
    r = 0.42
    n = 3 if l < 2 else 2
    ch = _channel(l, n, r)
    x, w = np.polynomial.legendre.leggauss(400)
    Q = 40.0 / r
    q = 0.5 * Q * (x + 1)
    w = 0.5 * Q * w
    p_hat = np.asarray(ch.projectors(jnp.asarray(q**2)))[i - 1] * q**l
    np.testing.assert_allclose(np.sum(w * p_hat**2 * q**2) / (2 * np.pi) ** 3, 1.0, rtol=1e-12)


def test_projectors_shape_and_dtype():
    s = hgh_lda("Si").channels[0]
    q2 = jnp.linspace(0.0, 10.0, 7).reshape(7, 1)
    out = s.projectors(q2)
    assert out.shape == (2, 7, 1)
    assert out.dtype == jnp.float64


@pytest.mark.parametrize(("l", "i"), TABLE)
def test_projectors_have_finite_derivatives_of_every_order_at_q_zero(l, i):
    """Functions of q^2, polynomial times a Gaussian: no square root, so first and
    second derivatives at q = 0 are finite (theory post, decision 7)."""
    n = 3 if l < 2 else 2
    ch = _channel(l, n, 0.45)

    def f(q):  # through the Cartesian vector, as the nonlocal term uses it
        return ch.projectors(jnp.sum(q**2))[i - 1]

    zero = jnp.zeros(3)
    assert np.isfinite(float(f(zero)))
    assert np.all(np.isfinite(np.asarray(jax.grad(f)(zero))))
    assert np.all(np.isfinite(np.asarray(jax.hessian(f)(zero))))
