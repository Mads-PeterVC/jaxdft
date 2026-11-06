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

# --- Milestone 2a (theory post 02a, 'Tolerances' and 'Decisions from review') ---
# Exact identities (transforms, Madelung, Ewald eta-independence, the Gaussian
# Hartree identity, gather vs FFT): lattice sums converge to ~1e-12, so 1e-10
# leaves a margin of 100.
EXACT_ATOL = 1e-10

# SCF tolerance on the L2 density residual (e Bohr^-3/2). Energy errors are then
# O(1e-12) Ha, force errors O(1e-8) Ha/Bohr.
SCF_TOL = 1e-9
# For finite-difference weight derivatives: SCF noise / delta must stay far below
# the 1e-5 Ha tolerance at delta = 2.5e-4.
SCF_TOL_TIGHT = 1e-12

# Energies of two converged runs that must agree exactly (grid-vector translation,
# axis permutation, atom exchange, initial density, mixing, dense vs LOBPCG):
# O(SCF_TOL^2) errors, with margin.
SCF_ENERGY_ATOL = 1e-10
# Densities of such runs, pointwise: first order in the residual. Peak density ~0.3.
SCF_DENSITY_ATOL = 1e-7

# Direct vs double-counting energy: first order in the residual of the final
# iterate (run at tol 1e-10). The review measured 5e-13.
DOUBLE_COUNTING_ATOL = 1e-8

# Finite-difference forces: central difference, single-atom step h = 1e-3 Bohr.
# Truncation h^2 E'''/6 with E''' ~ 1.06 Ha/Bohr^3 (review) is 1.8e-7 Ha/Bohr.
FD_STEP = 1e-3
FORCE_ATOL = 1e-6

# One-sided weight derivatives, second-order one-sided stencil (review 2.2):
# left (HOMO) side measured 2.9e-7 at delta = 1e-3; right (LUMO) side 4.1e-6 at
# delta = 2.5e-4 (observed order ~1.6). Fallback if fragile: Richardson, not a
# looser tolerance.
WEIGHT_STEP = {-1: 1e-3, +1: 2.5e-4}
WEIGHT_ATOL = 1e-5
# Left minus right derivative equals Z * gap (review: 1e-9).
WEIGHT_GAP_ATOL = 1e-8

# Egg-box (theory post 'Net force and the egg-box effect'): with the grid rule
# satisfied the peak-to-peak variation under translation is SCF noise (review:
# 2e-11 at the test size); an undersized grid (N = 24 at the test size) gives
# 8.4e-6. The thresholds sit between, with a factor of >= 8 to each measurement.
EGGBOX_RULE_MAX = 1e-9
EGGBOX_UNDERSIZED_MIN = 1e-6
# Energy change from the rule grid (N = 30) to a larger one: < 1e-9; to the
# undersized N = 24 grid: 3.3e-7 measured.
GRID_ENERGY_RULE_MAX = 1e-9
GRID_ENERGY_UNDERSIZED_MIN = 1e-7

# Net force of an off-grid, non-symmetric placement (review: 2.5e-11).
NET_FORCE_MAX = 1e-8

# Final eigenpair residuals must be below this fraction of the SCF tolerance.
EIGEN_RESIDUAL_FRACTION = 0.1

# --- Milestone 2a reference codes (theory post 02a, 'Tolerances', decision 11) ---
# DFTK solves the same discrete problem (same pseudopotential file, functional,
# box, cutoff and FFT grid), so agreement is limited by the SCF tolerances of the
# two runs. Fixed in the theory post before either code was run.
DFTK_ENERGY_ATOL = 1e-7  # Ha, per term and total
DFTK_EIGENVALUE_ATOL = 1e-6  # Ha, after the explicit alpha shift
DFTK_FORCE_ATOL = 1e-6  # Ha/Bohr
# DFTK's PspCorrection against N_el sum_j alpha_j / Omega: a closed form, so only
# rounding separates them.
PSP_CORRECTION_ATOL = 1e-12

# libxc lda_x and lda_c_pw evaluate the same closed forms as jaxdft in double
# precision. Not fixed in the theory post; set when the reference was generated.
# Measured worst relative difference 6e-12 (lda_c_pw at rho = 1e-10, r_s ~ 1300,
# rounding in libxc's evaluation), ~1e-15 for rho >= 1e-4. 1e-10 relative leaves
# a margin of 15 at the worst point and still fails any parameter transcription
# error (lda_c_pw_mod differs by ~1e-6 relative).
LIBXC_RTOL = 1e-10
