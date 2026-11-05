"""Tolerances for Milestone 1a, with justification (see theory post, 'Tolerances').

EIGENVALUE_ATOL: eigh is backward stable, so the error is about eps_mach * ||H_phys||.
    ||H_phys|| <= ecut, at most a few hundred Ha in the tests, gives ~1e-13 Ha.
    1e-10 Ha leaves three orders of margin. Any physical error (a wrong lattice
    vector, a missing plane wave, leaking padding) is O(E0) ~ 0.1 Ha.
RESIDUAL_RTOL: ||Hc - eps c|| <= tau * ||H||, relative so it does not depend on the cutoff.
DEGENERACY_ATOL: for grouping eigenvalues into levels. Far above EIGENVALUE_ATOL and
    far below the smallest level spacing at the high-symmetry points
    (E0 = 2 pi^2 / a^2 ~ 0.19 Ha for a = 10.26 Bohr).
SHELL_MARGIN_RTOL: test cutoffs sit at least this far (relative) from any shell
    1/2|k+G|^2, so ties at the cutoff cannot flip on round-off (~4e-16 relative).
"""

EIGENVALUE_ATOL = 1e-10
RESIDUAL_RTOL = 1e-12
DEGENERACY_ATOL = 1e-6
SHELL_MARGIN_RTOL = 1e-8

# --- Milestone 1b (theory post 01b, 'Reference values' and 'Against references') ---
# Physics tolerances against Cohen & Bergstresser (1966) Table I, in eV.
# Table I rounds to 0.1 eV (+-0.05) and the paper states convergence 'within 0.1 eV' for
# each energy measured from Gamma25'.
#   from_vbm:        splitting from the VBM:          0.05 + 0.1     = 0.15 eV
#   between_levels:  difference of two such energies: 0.05 + 2 * 0.1 = 0.25 eV
CB_TABLE_TOL_EV = {"from_vbm": 0.15, "between_levels": 0.25}

# Grouping eigenvalues into degenerate levels in 1b. Symmetry-forced degeneracies come
# out at ~1e-13 Ha; level spacings and the physics tolerances are >= ~5e-3 Ha.
LEVEL_ATOL = 1e-8

# A level pair counts as split (negative control for band sticking) above this, in Ha.
# Six orders above LEVEL_ATOL; a generic face point splits by several mHa.
SPLIT_MIN = 1e-4

# The Delta-line minimum must lie below the conduction level at X by more than this (Ha):
# far above numerical error (1e-10 Ha), far below the ~0.13 eV (~5e-3 Ha) model difference.
INTERIOR_MARGIN = 1e-4
