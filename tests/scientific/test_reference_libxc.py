"""jaxdft's LDA against libxc lda_x and lda_c_pw, pointwise (theory post 02a,
'Exchange and correlation'; decisions 2 and 11). Energy per particle and potential
on a logarithmic density grid from 1e-10 to 1e3 Bohr^-3."""

import jax.numpy as jnp
import numpy as np

from jaxdft.xc.lda import lda_pw92, lda_x
from support.reference import libxc_lda
from support.tolerances import LIBXC_RTOL

REF = libxc_lda()
RHO = jnp.asarray(REF["rho"])


def test_reference_is_unpolarised_lda_c_pw_not_mod():
    assert set(REF["functionals"]) == {"lda_x", "lda_c_pw", "lda_c_pw_mod"}
    assert len(REF["rho"]) > 100


def test_comparison_tells_lda_c_pw_mod_apart():
    """Negative control: the variant with more parameter digits differs from ours by
    3e-7 to 6e-6 relative at valence densities, far outside LIBXC_RTOL."""
    rho = np.asarray(REF["rho"])
    eps_c = np.asarray(lda_pw92.eps(RHO) - lda_x.eps(RHO))
    rel = np.abs(eps_c / np.asarray(REF["functionals"]["lda_c_pw_mod"]["zk"]) - 1)
    assert rel[rho >= 1e-4].min() > 1000 * LIBXC_RTOL


def test_exchange():
    ref = REF["functionals"]["lda_x"]
    _, v = lda_x(RHO)
    np.testing.assert_allclose(lda_x.eps(RHO), ref["zk"], rtol=LIBXC_RTOL, atol=0)
    np.testing.assert_allclose(v, ref["vrho"], rtol=LIBXC_RTOL, atol=0)


def test_correlation():
    """jaxdft exposes exchange and exchange + correlation; correlation is their
    difference, which loses nothing here (|eps_c| is a sizeable fraction of |eps_x|)."""
    ref = REF["functionals"]["lda_c_pw"]
    _, vx = lda_x(RHO)
    _, vxc = lda_pw92(RHO)
    np.testing.assert_allclose(lda_pw92.eps(RHO) - lda_x.eps(RHO), ref["zk"], rtol=LIBXC_RTOL, atol=0)
    np.testing.assert_allclose(vxc - vx, ref["vrho"], rtol=LIBXC_RTOL, atol=0)
