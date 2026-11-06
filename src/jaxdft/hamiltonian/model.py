"""Standard sets of terms (engineering log, 2a API: 'lda_terms')."""

from __future__ import annotations

from jaxdft.hamiltonian.atomic_local import AtomicLocal
from jaxdft.hamiltonian.ewald import Ewald
from jaxdft.hamiltonian.hartree import Hartree
from jaxdft.hamiltonian.kinetic import Kinetic
from jaxdft.hamiltonian.xc import Xc
from jaxdft.xc.lda import lda_pw92


def lda_terms(pseudos: dict) -> tuple:
    """Kinetic, local pseudopotential, Hartree, LDA (Slater + PW92) and Ewald."""
    return (Kinetic(), AtomicLocal(pseudos), Hartree(), Xc(lda_pw92), Ewald(pseudos))
