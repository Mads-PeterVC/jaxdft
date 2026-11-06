"""The FFT grid on PlaneWaveBasis (theory post 02a, 'The FFT grid'; engineering log:
the grid lives on the basis, an explicit fft_shape overrides the rule)."""

import itertools
import math

import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import Lattice, PlaneWaveBasis
from jaxdft.basis.planewave import min_fft_shape, rule_fft_shape
from jaxdft.scf.density import electron_density

GAMMA = np.zeros((1, 3))


def smooth_235(n):
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def rule_shape(A, ecut):
    """N_i >= 2 floor(2 Gmax |a_i| / 2 pi) + 1, rounded up to 2-3-5 smooth."""
    gmax = math.sqrt(2 * ecut)
    return tuple(smooth_235(2 * math.floor(2 * gmax * np.linalg.norm(a) / (2 * math.pi)) + 1) for a in A)


def random_coefficients(basis_k, n, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(basis_k.n_pad, n)) + 1j * rng.normal(size=(basis_k.n_pad, n))
    c = np.where(np.asarray(basis_k.mask)[:, None], c, 0)
    q, _ = np.linalg.qr(c[np.asarray(basis_k.mask)])
    out = np.zeros_like(c)
    out[np.asarray(basis_k.mask)] = q
    return jnp.asarray(out)


def test_rule_shape_at_the_test_size():
    """L = 10 Bohr, Ecut = 10 Ha: rule minimum 29, rounded to 30 (review table 1, #4)."""
    b = PlaneWaveBasis.build(Lattice(10.0 * np.eye(3)), GAMMA, 10.0)
    assert b.fft_shape == (30, 30, 30)


@pytest.mark.parametrize(
    "A",
    [10.0 * np.eye(3), np.diag([7.0, 9.5, 12.0]), 5.13 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])],
    ids=["cube", "orthorhombic", "fcc"],
)
@pytest.mark.parametrize("ecut", [3.3, 6.0, 10.0])
def test_rule_shape(A, ecut):
    b = PlaneWaveBasis.build(Lattice(A), GAMMA, ecut)
    assert b.fft_shape == rule_shape(A, ecut)


def test_explicit_shape_overrides_rule():
    b = PlaneWaveBasis.build(Lattice(10.0 * np.eye(3)), GAMMA, 10.0, fft_shape=(36, 40, 45))
    assert b.fft_shape == (36, 40, 45)


def test_undersized_shape_needs_explicit_opt_in():
    """Below the no-wraparound rule the density aliases; that is allowed only as the
    deliberate positive control of the grid-rule test."""
    lat = Lattice(10.0 * np.eye(3))
    with pytest.raises(ValueError):
        PlaneWaveBasis.build(lat, GAMMA, 10.0, fft_shape=(24, 24, 24))
    b = PlaneWaveBasis.build(lat, GAMMA, 10.0, fft_shape=(24, 24, 24), allow_aliasing=True)
    assert b.fft_shape == (24, 24, 24)


def test_the_bound_itself_is_accepted_without_opt_in():
    """The check is against the bound (29 at L = 10, Ecut = 10, theory post), not its
    2-3-5 rounding (30): a grid of exactly the bound does not alias."""
    lat = Lattice(10.0 * np.eye(3))
    assert min_fft_shape(lat, 10.0) == (29, 29, 29)
    assert rule_fft_shape(lat, 10.0) == (30, 30, 30)
    assert PlaneWaveBasis.build(lat, GAMMA, 10.0, fft_shape=(29, 29, 29)).fft_shape == (29, 29, 29)
    with pytest.raises(ValueError):
        PlaneWaveBasis.build(lat, GAMMA, 10.0, fft_shape=(28, 28, 28))


def test_shape_too_small_for_the_basis_itself_always_raises():
    """A grid that cannot hold the basis Miller indices at all is never valid."""
    with pytest.raises(ValueError):
        PlaneWaveBasis.build(Lattice(10.0 * np.eye(3)), GAMMA, 10.0, fft_shape=(12, 12, 12), allow_aliasing=True)


def test_to_grid_is_the_plane_wave_sum():
    """to_grid(c)[j] = sum_G c(G) exp(2 pi i m . j / N): no 1/sqrt(Omega), no 1/N."""
    lat = Lattice(np.diag([5.0, 6.0, 7.0]))
    b = PlaneWaveBasis.build(lat, GAMMA, 2.0)[0]
    c = random_coefficients(b, 1)[:, 0]
    got = np.asarray(b.to_grid(c))
    N = b.fft_shape
    m = np.asarray(b.miller)[np.asarray(b.mask)]
    cc = np.asarray(c)[np.asarray(b.mask)]
    for j in itertools.islice(itertools.product(*[range(n) for n in N]), 0, None, 37):
        phase = np.exp(2j * np.pi * (m @ (np.asarray(j) / np.asarray(N))))
        np.testing.assert_allclose(got[j], np.sum(cc * phase), atol=1e-12)


def test_round_trip_and_parseval():
    lat = Lattice(np.diag([5.0, 6.0, 7.0]))
    b = PlaneWaveBasis.build(lat, np.array([[0.1, -0.2, 0.3]]), 4.0)[0]
    c = random_coefficients(b, 3)
    f = b.to_grid(c)
    assert f.shape == (*b.fft_shape, 3)
    np.testing.assert_allclose(b.from_grid(f), c, atol=1e-13)
    n_tot = np.prod(b.fft_shape)
    np.testing.assert_allclose(np.sum(np.abs(f) ** 2, axis=(0, 1, 2)) / n_tot, 1.0, atol=1e-13)


def test_from_grid_is_zero_on_padding():
    lat = Lattice(np.diag([5.0, 6.0, 7.0]))
    b = PlaneWaveBasis.build(lat, GAMMA, 3.0, n_pad=None)
    bp = PlaneWaveBasis.build(lat, GAMMA, 3.0, n_pad=int(b.n_pw[0]) + 7)[0]
    f = jnp.asarray(np.random.default_rng(1).normal(size=bp.fft_shape))
    out = np.asarray(bp.from_grid(f))
    assert np.all(out[~np.asarray(bp.mask)] == 0)


def direct_density_coefficients(basis_k, c, f, volume):
    """rho~(K) = (1/Omega) sum_n f_n sum_{G - G' = K} c_n(G) conj(c_n(G'))."""
    m = np.asarray(basis_k.miller)[np.asarray(basis_k.mask)]
    cc = np.asarray(c)[np.asarray(basis_k.mask)]
    out = {}
    for i in range(len(m)):
        for j in range(len(m)):
            K = tuple(m[i] - m[j])
            out[K] = out.get(K, 0) + np.sum(f * cc[i] * np.conj(cc[j])) / volume
    return out


def test_density_has_no_aliasing_and_integrates_to_n():
    """Exactness claim 1: the grid density has exactly the Fourier coefficients of the
    continuous |psi|^2 (review: 3e-20)."""
    A = np.diag([5.0, 5.5, 6.0])
    lat = Lattice(A)
    b = PlaneWaveBasis.build(lat, GAMMA, 2.0)
    c = random_coefficients(b[0], 2)
    f = np.array([2.0, 0.7])
    rho = electron_density(lat, b, c[None], jnp.asarray(f)[None], jnp.ones(1))
    assert rho.shape == b.fft_shape and rho.dtype == jnp.float64
    assert float(rho.min()) >= 0.0
    volume = float(np.linalg.det(A))
    np.testing.assert_allclose(volume / rho.size * float(rho.sum()), f.sum(), rtol=1e-13)
    rho_g = np.fft.fftn(np.asarray(rho)) / rho.size
    ref = direct_density_coefficients(b[0], c, f, volume)
    N = np.asarray(b.fft_shape)
    for K, val in ref.items():
        np.testing.assert_allclose(rho_g[tuple(np.mod(K, N))], val, atol=1e-14)
    # and nothing outside the difference set
    mask = np.ones(b.fft_shape, dtype=bool)
    for K in ref:
        mask[tuple(np.mod(K, N))] = False
    assert np.max(np.abs(rho_g[mask])) < 1e-15
