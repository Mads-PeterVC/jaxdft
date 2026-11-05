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
