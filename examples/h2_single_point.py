"""Single-point LDA calculation of H2 in a cubic box (Milestone 2a).

Usage: uv run python examples/h2_single_point.py [log file]
"""

import sys

import numpy as np

import jaxdft
from jaxdft import PlaneWaveBasis, SCFOptions, dimer_in_box, lda_terms, scf
from jaxdft.pseudo.gth import hgh_lda

log_file = sys.argv[1] if len(sys.argv) > 1 else "h2_single_point.log"
jaxdft.log_to_file(log_file)

# Two H atoms 1.45 Bohr apart along z, centred in a 10 Bohr cube.
structure = dimer_in_box("H", bond_length=1.45, box=10.0)

# Plane waves up to 10 Ha at the Gamma point only.
gamma = np.zeros((1, 3))
basis = PlaneWaveBasis.build(structure.lattice, kpoints=gamma, ecut=10.0)

# Kinetic, HGH local pseudopotential, Hartree, LDA (Slater + PW92), Ewald.
terms = lda_terms({"H": hgh_lda("H")})

options = SCFOptions(tol=1e-9)
result = scf(structure, basis, terms, options)

print(f"E = {float(result.total_energy):.10f} Ha in {int(result.n_iter)} iterations")
