"""Single-point LDA calculation of linear, symmetric H3 in a cubic box.

H3 is not a stable molecule: the linear symmetric geometry is the saddle point of
the H + H2 -> H2 + H exchange reaction (bonds of about 0.93 Angstrom). It has three
electrons. The SCF is spin-unpolarised, so the third electron half-fills the
second level (occupations 2, 1, 0): a restricted, not a spin-polarised, result.
The linear geometry keeps that level non-degenerate; in an equilateral triangle
it would be a degenerate pair, which the SCF's frontier-gap check rejects.

The log is printed to standard output as the SCF runs.

Usage: uv run python examples/h3_single_point.py
"""

import numpy as np

import jaxdft
from jaxdft import PlaneWaveBasis, SCFOptions, Structure, lda_terms, scf
from jaxdft.pseudo.gth import hgh_lda

jaxdft.log_to_stream()  # live log on stdout

# Three H atoms along z, 1.76 Bohr apart, centred in a 12 Bohr cube.
box, bond = 12.0, 1.76
cell = box * np.eye(3)
positions = [[box / 2, box / 2, box / 2 + dz] for dz in (-bond, 0.0, bond)]
structure = Structure.from_cartesian(cell, positions, ["H", "H", "H"], unit="Bohr")

# Plane waves up to 10 Ha at the Gamma point only.
gamma = np.zeros((1, 3))
basis = PlaneWaveBasis.build(structure.lattice, kpoints=gamma, ecut=10.0)

# Kinetic, HGH local pseudopotential, Hartree, LDA (Slater + PW92), Ewald.
terms = lda_terms({"H": hgh_lda("H")})

options = SCFOptions(tol=1e-9)
result = scf(structure, basis, terms, options)

print(f"E = {float(result.total_energy):.10f} Ha in {int(result.n_iter)} iterations")
print(f"occupations = {np.asarray(result.occupations[0])}")
