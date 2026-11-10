"""GTH/HGH local pseudopotential: parser, vendored file, form factor and alpha
(theory post 02a, 'The GTH local pseudopotential'; engineering log, 2a API)."""

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jaxdft.pseudo.cohen_bergstresser import silicon
from jaxdft.pseudo.gth import GTH, hgh_lda, hgh_lda_path, parse_hgh
from support.budget import examples

H_SHA256 = "2f4069ba8ea992435a2c614bed3b2a1ed18b808fc1ffc5d7a9dcaa519f1f5969"

SI_Q4 = """Si GTH-PADE-q4 GTH-LDA-q4
    2    2
     0.44000000    1    -7.33610297
    2
     0.42273813    2     5.90692831    -1.26189397
                                        3.25819622
     0.48427842    1     2.72701346
"""


def numerical_form_factor(zion, rloc, c, q):
    """v_hat(q) = int v(r) e^{-iq.r} d^3r by Gauss-Legendre quadrature of the
    short-range part, written from the real-space potential (not the closed form)."""
    from math import erf, sqrt

    x, w = np.polynomial.legendre.leggauss(400)
    R = 14.0 * rloc
    r = 0.5 * R * (x + 1)
    w = 0.5 * R * w
    erf_v = np.vectorize(erf)
    t = r / rloc
    v = -zion / r * erf_v(r / (sqrt(2) * rloc)) + np.exp(-(t**2) / 2) * (
        c[0] + c[1] * t**2 + c[2] * t**4 + c[3] * t**6
    )
    short = v + zion / r  # the Coulomb tail -Z/r is transformed analytically
    return np.sum(w * short * 4 * np.pi * r**2 * np.sinc(q * r / np.pi)) - 4 * np.pi * zion / q**2


# --- the vendored file and the parser -----------------------------------------------


def test_vendored_hydrogen_file_is_pinned():
    """Provenance: byte-identical to PseudoLibrary hgh_lda_hgh/h-q1.hgh (data README)."""
    assert hashlib.sha256(hgh_lda_path("H").read_bytes()).hexdigest() == H_SHA256


def test_hydrogen_parameters():
    """HGH 1998 LDA H, as in the file: Z = 1, r_loc = 0.2, C1, C2, no projectors."""
    h = hgh_lda("H")
    assert isinstance(h, GTH)
    assert h.zion == 1
    assert h.ionic_charge == 1
    assert h.rloc == 0.2
    assert tuple(h.c) == (-4.18023680, 0.72507482, 0.0, 0.0)
    assert h.local_support is None


def test_parse_hgh_reads_the_file_path(tmp_path):
    p = tmp_path / "h.hgh"
    p.write_bytes(hgh_lda_path("H").read_bytes())
    h = parse_hgh(p)
    assert (h.zion, h.rloc, tuple(h.c)) == (1, 0.2, (-4.18023680, 0.72507482, 0.0, 0.0))


def test_parse_hgh_four_local_coefficients(tmp_path):
    p = tmp_path / "x.hgh"
    p.write_text("X test\n    3\n     0.35000000    4    -1.5    0.25    0.125    -0.0625\n    0\n")
    x = parse_hgh(p)
    assert x.zion == 3
    assert tuple(x.c) == (-1.5, 0.25, 0.125, -0.0625)


def test_parse_hgh_with_projectors_is_not_implemented_yet(tmp_path):
    """Nonlocal projectors arrive with 2b; until then a file with them is refused,
    never silently truncated to its local part."""
    p = tmp_path / "si-q4.hgh"
    p.write_text(SI_Q4)
    with pytest.raises(NotImplementedError):
        parse_hgh(p)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "H x\n    1\n",
        "H x\n    1\n     0.2    3    -4.0    0.7\n    0\n",  # nexp says 3, gives 2
        "H x\n    1\n     abc    2    -4.0    0.7\n    0\n",
    ],
)
def test_parse_hgh_malformed_raises_value_error(tmp_path, text):
    p = tmp_path / "bad.hgh"
    p.write_text(text)
    with pytest.raises(ValueError):
        parse_hgh(p)


def test_hgh_lda_unknown_element_raises():
    with pytest.raises(ValueError):
        hgh_lda("Xx")


# --- form factor and alpha ----------------------------------------------------------


@settings(max_examples=examples(10))
@given(
    st.floats(0.15, 0.6),
    st.lists(st.floats(-8.0, 8.0), min_size=4, max_size=4),
    st.sampled_from([1, 2, 4]),
    st.sampled_from([0.3, 1.0, 3.0, 7.0]),
)
def test_form_factor_matches_quadrature(rloc, c, zion, q):
    """Closed form against a numerical radial transform (checked to 1e-15 relative
    while preparing the post)."""
    got = float(GTH(zion, rloc, c).local(jnp.asarray(q)))
    ref = numerical_form_factor(zion, rloc, c, q)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


@settings(max_examples=examples(10))
@given(st.floats(0.15, 0.6), st.lists(st.floats(-8.0, 8.0), min_size=4, max_size=4), st.sampled_from([1, 2, 4]))
def test_alpha_is_the_non_coulomb_q_to_zero_limit(rloc, c, zion):
    g = GTH(zion, rloc, c)
    expected = 2 * np.pi * zion * rloc**2 + (2 * np.pi) ** 1.5 * rloc**3 * (
        c[0] + 3 * c[1] + 15 * c[2] + 105 * c[3]
    )
    np.testing.assert_allclose(g.alpha, expected, rtol=1e-14, atol=1e-14)
    q = 1e-4
    np.testing.assert_allclose(float(g.local(q)) + 4 * np.pi * zion / q**2, g.alpha, atol=1e-6 * max(1, abs(g.alpha)) + 1e-6)


def test_alpha_of_hydrogen_is_a_near_cancellation():
    """0.251327 - 0.252625 = -0.001297886 Ha Bohr^3 (review 2.4): a sharp check of the
    last digits of C1 and C2."""
    np.testing.assert_allclose(hgh_lda("H").alpha, -0.001297886, atol=5e-10)


def test_local_at_zero_is_alpha_and_differentiable():
    """local(0) returns the finite non-Coulomb limit, with finite gradients, so the
    G = 0 entry never produces NaN under autodiff."""
    h = hgh_lda("H")
    np.testing.assert_allclose(float(h.local(0.0)), h.alpha, atol=1e-15)
    g = jax.grad(lambda q: h.local(q))(0.0)
    assert np.isfinite(float(g))
    qs = jnp.linspace(0.0, 10.0, 101)
    assert np.all(np.isfinite(np.asarray(jax.vmap(jax.grad(h.local))(qs))))


def test_form_factor_is_float64():
    assert hgh_lda("H").local(jnp.linspace(0.1, 5.0, 7)).dtype == jnp.float64


def test_cohen_bergstresser_protocol_additions():
    """Empirical models have no point ions; alpha = 0 is CB's V0 = 0 convention."""
    cb = silicon()
    assert cb.ionic_charge is None
    assert cb.alpha == 0.0
