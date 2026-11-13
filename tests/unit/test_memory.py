"""Memory of dense diagonalisation and of batched k-points (theory post 02c, 'Memory
of batched k-points', acceptance criterion 8; engineering log, '2026-11-12:
Milestone 2c public API').

Memory is XLA's own estimate of the peak temporaries of a compiled program
(``memory_analysis()``), so it is measurable on the CPU; the V100 failed on exactly
the 9.44 GiB this estimate gives for the band sampling below. The band-sampling test
lowers band_structure's jitted core, ``_solve``, because only a lowered program can
be measured; its signature is fixed here (k_batch_size as a static argument).
"""

import jax
import jax.numpy as jnp
import numpy as np

from jaxdft import (
    Kinetic,
    LocalPotential,
    PlaneWaveBasis,
    apply_H,
    band_structure,
    dense_eigensolver,
    diamond,
    hamiltonian,
)
from jaxdft.calculations.bands import _solve, bind_terms
from jaxdft.pseudo.cohen_bergstresser import silicon as cb_silicon
from jaxdft.structure.kpoints import special_points_fcc
from support.silicon import basis_for, si, si_terms
from support.tolerances import (
    BAND_SAMPLING_TEMP_MAX_GIB,
    CHUNKED_ATOL,
    DENSE_20HA_TEMP_MAX_GIB,
)

GIB = 2**30


def test_dense_solver_memory_does_not_grow_with_the_basis():
    """One silicon k-point at 20 Ha (1139 plane waves, 30^3 grid): 1.375 GiB of
    temporaries when H is applied to the whole identity at once."""
    s = si()
    b = basis_for(s, ecut=20.0, kshape=(1, 1, 1))
    ham = hamiltonian(s, b[0], bind_terms(s, si_terms()), density=jnp.full(b.fft_shape, 0.03))
    n = b.n_pad

    def solve(h):
        return dense_eigensolver(lambda p: apply_H(h, p), n, 5)

    temp = jax.jit(solve).lower(ham).compile().memory_analysis().temp_size_in_bytes
    assert temp / GIB < DENSE_20HA_TEMP_MAX_GIB, temp / GIB


def test_band_sampling_of_the_failing_v100_case_fits():
    """band_minimum_on_segment's sampling: 41 k-points from Gamma to X, LDA silicon at
    12 Ha, dense solver, chunks of 8 k-points (its default)."""
    s = si()
    P = special_points_fcc()
    ts = jnp.linspace(0.0, 1.0, 41)[:, None]
    k = jnp.asarray(P["G"]) + ts * (jnp.asarray(P["X"]) - jnp.asarray(P["G"]))
    b = PlaneWaveBasis.build(s.lattice, k, 12.0)
    terms = bind_terms(s, si_terms())
    rho = jnp.full(b.fft_shape, 0.03)
    temp = _solve.lower(s, b, terms, 5, dense_eigensolver, rho, 8).compile().memory_analysis().temp_size_in_bytes
    assert temp / GIB < BAND_SAMPLING_TEMP_MAX_GIB, temp / GIB


def test_dense_column_chunks_do_not_change_the_result():
    rng = np.random.default_rng(11)
    n = 150  # not a multiple of the default chunk of 64
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = jnp.asarray(np.diag(np.linspace(0, 20, n)) + 0.1 * (X + X.conj().T))
    whole, _ = dense_eigensolver(lambda x: H @ x, n, 6, column_batch_size=None)
    chunked, vecs = dense_eigensolver(lambda x: H @ x, n, 6, column_batch_size=64)
    np.testing.assert_allclose(chunked, whole, atol=CHUNKED_ATOL)
    res = np.linalg.norm(np.asarray(H @ vecs) - np.asarray(vecs) * np.asarray(chunked), axis=0)
    assert np.all(res < 1e-11)


def test_band_structure_k_chunks_do_not_change_the_result():
    """Cohen-Bergstresser silicon at 7 k-points, chunks of 3 (not a divisor of 7)."""
    s = diamond(5.43, "Si", unit="Angstrom")
    terms = (Kinetic(), LocalPotential({"Si": cb_silicon()}))
    k = jnp.linspace(0.0, 0.5, 7)[:, None] * jnp.ones(3)
    b = PlaneWaveBasis.build(s.lattice, k, 4.0)
    whole = band_structure(s, b, n_bands=8, terms=terms)
    chunked = band_structure(s, b, n_bands=8, terms=terms, k_batch_size=3)
    np.testing.assert_allclose(chunked.eigenvalues, whole.eigenvalues, atol=CHUNKED_ATOL)
