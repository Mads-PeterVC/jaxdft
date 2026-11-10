"""Extension points of the SCF (engineering log, 'SCF extension points'): term roles,
the preconditioner and the initial density are swappable without editing the SCF,
and the defaults reproduce the built-in behaviour exactly."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdft
from jaxdft import AtomicLocal, Ewald, Hartree, Kinetic, SCFOptions, Xc, dense_matrix, energy, hamiltonian, scf
from jaxdft.hamiltonian.preconditioners import diagonal_preconditioner
from jaxdft.scf.density import AtomicGaussians
from jaxdft.xc.lda import lda_pw92
from support.h2 import converged_small, h2_terms, hydrogen
from support.tolerances import SCF_DENSITY_ATOL, SCF_ENERGY_ATOL, SCF_TOL

# --- term roles ----------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PointIons:
    """An ionic energy term that is not AtomicLocal and has no ``pseudos``: it reaches
    the SCF only through the declared role."""

    inner: AtomicLocal = field(metadata=dict(static=True))

    is_ionic = True

    def ionic_charges(self, structure):
        return self.inner.ionic_charges(structure)

    def static_ionic_charges(self, structure):
        return self.inner.static_ionic_charges(structure)

    def bind(self, structure):
        return PointIons(self.inner.bind(structure))

    def energy(self, structure, basis, density):
        return self.inner.energy(structure, basis, density)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IonIon:
    """An ion-ion term not named Ewald."""

    inner: Ewald = field(metadata=dict(static=True))

    density_dependent = False
    ion_ion = True

    def energy(self, structure, basis, density):
        return self.inner.energy(structure, basis, density)


def renamed_terms():
    pseudos = {"H": hydrogen()}
    return (Kinetic(), PointIons(AtomicLocal(pseudos)), Hartree(), Xc(lda_pw92), IonIon(Ewald(pseudos)))


def test_terms_are_found_by_role_not_by_class_name():
    s, b, res = converged_small()
    other = scf(s, b, renamed_terms(), SCFOptions(tol=SCF_TOL))
    assert float(other.total_energy) == float(res.total_energy)
    np.testing.assert_array_equal(other.density, res.density)
    assert set(other.energies) == {"Kinetic", "PointIons", "Hartree", "Xc", "IonIon"}


def test_energy_and_forces_use_the_static_charges_role():
    """energy() sizes n_bands from static_ionic_charges, not from a ``pseudos`` field."""
    s, b, _ = converged_small()
    opts = SCFOptions(tol=SCF_TOL)
    e_ref, g_ref = jax.value_and_grad(lambda st: energy(st, b, h2_terms(), opts))(s)
    e, g = jax.value_and_grad(lambda st: energy(st, b, renamed_terms(), opts))(s)
    assert float(e) == float(e_ref)
    np.testing.assert_array_equal(g.positions, g_ref.positions)


def test_without_an_ion_ion_term_the_scf_refuses():
    s, b, _ = converged_small()
    terms = renamed_terms()[:-1]
    with pytest.raises(ValueError, match="ion-ion"):
        scf(s, b, terms, SCFOptions(tol=SCF_TOL))


# --- the Hamiltonian diagonal and the preconditioner -----------------------------------


def test_hamiltonian_diagonal_is_the_kinetic_diagonal():
    """With only the kinetic term the summed diagonal is the diagonal of the dense
    matrix on the physical block; energy terms (a grid potential) add nothing to it."""
    s, b, _ = converged_small()
    bk = b[0]
    mask = np.asarray(bk.mask)
    kin = hamiltonian(s, bk, (Kinetic(),))
    dense = np.real(np.diag(np.asarray(dense_matrix(kin))))
    np.testing.assert_array_equal(np.asarray(kin.diagonal())[mask], dense[mask])
    full = hamiltonian(s, bk, jaxdft.calculations.bands.bind_terms(s, h2_terms()))
    np.testing.assert_array_equal(full.diagonal(), kin.diagonal())


def test_preconditioner_is_an_option():
    """The SCF builds its preconditioner from the option, once per k-point, with the
    prepared Hamiltonian; without one it reaches the same ground state."""
    s, b, res = converged_small()
    seen = []

    def recording(ham):
        seen.append(ham)
        return diagonal_preconditioner(ham)

    rec = scf(s, b, h2_terms(), SCFOptions(tol=SCF_TOL, preconditioner=recording))
    assert seen and all(isinstance(h, jaxdft.Hamiltonian) for h in seen)
    assert float(rec.total_energy) == float(res.total_energy)

    plain = scf(s, b, h2_terms(), SCFOptions(tol=SCF_TOL, preconditioner=None))
    np.testing.assert_allclose(float(plain.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)
    np.testing.assert_allclose(plain.density, res.density, atol=SCF_DENSITY_ATOL)


def test_every_eigensolver_accepts_tol():
    """The SCF always passes ``tol``: it is part of the eigensolver interface."""
    import inspect

    for solver in (jaxdft.dense_eigensolver, jaxdft.lobpcg):
        assert "tol" in inspect.signature(solver).parameters


# --- the initial density ---------------------------------------------------------------


def test_initial_density_is_an_option():
    """A callable initial density is used exactly as an explicit density0 would be."""
    s, b, _ = converged_small()
    uniform = jnp.full(b.fft_shape, 2.0 / float(s.lattice.volume))
    calls = []

    def flat(structure, basis, charges):
        calls.append(charges)
        return jnp.full(basis.fft_shape, jnp.sum(charges) / structure.lattice.volume)

    via_option = scf(s, b, h2_terms(), SCFOptions(tol=SCF_TOL, initial_density=flat))
    via_density0 = scf(s, b, h2_terms(), SCFOptions(tol=SCF_TOL), density0=uniform)
    assert len(calls) == 1
    assert float(via_option.total_energy) == float(via_density0.total_energy)
    np.testing.assert_array_equal(via_option.density, via_density0.density)


def test_atomic_gaussians_carry_the_electrons():
    s, b, _ = converged_small()
    charges = jnp.asarray([1.0, 1.0])
    for width in (0.5, 0.8, 1.2):
        rho = AtomicGaussians(width)(s, b, charges)
        assert rho.shape == b.fft_shape and rho.dtype == jnp.float64
        assert float(rho.min()) >= 0.0
        np.testing.assert_allclose(float(s.lattice.volume) / rho.size * float(rho.sum()), 2.0, rtol=1e-14)
