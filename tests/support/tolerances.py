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

# --- Milestone 2b (theory post 02b, 'Tolerances' and 'Decisions from review') ---
# Measured values are from the throwaway check and/or the review (both independent of
# jaxdft); "derived" marks a tolerance with no measurement behind it.

# Projector transforms against quadrature: the throwaway check found 5e-14 relative,
# the review 2.7e-15. The quadrature itself is converged to ~1e-15.
PROJECTOR_RTOL = 1e-10

# Even FFT grid: forces at the ideal geometry and the space-group invariance of the
# density are exact identities of the discrete problem, up to SCF error (theory post:
# 1e-10). Measured 8e-16 (N = 24), review 3.8e-15 (N = 24) and 2.4e-15 (N = 26).
SYMMETRY_FORCE_ATOL = 1e-10
# Odd grid (N = 25, 12 Ha): 6e-7 Ha/Bohr (both). A window, so that a silently
# symmetric odd grid also fails.
ODD_GRID_FORCE_WINDOW = (1e-8, 5e-6)
# Half-shifted 2^3 grid at the ideal geometry: 1.0e-3 Ha/Bohr (both). Positive control.
SHIFTED_GRID_FORCE_MIN = 1e-4
# Rigid translation by half a grid spacing at 12 Ha: 2.0e-7 Ha (both). A whole grid
# spacing is exact (SCF_ENERGY_ATOL).
EGGBOX_HALF_SPACING_MIN = 2e-8

# Net force at the displaced geometry (theory post, 'Tolerances'): with XC measured
# 5.7e-7 Ha/Bohr on 2^3 and ~5e-7 on 3^3, where the tests now run (2^3 closes the
# gap over the grid; decided with the project lead); without XC 4e-15 on 2^3 (review
# 2.5e-15), first order in the SCF tolerance since it vanishes only at
# self-consistency.
NET_FORCE_XC_MAX = 5e-6
NET_FORCE_NO_XC_MAX = 1e-8

# Finite-difference forces in fractional coordinates: step 1e-3, tolerance 1e-6 Ha
# per unit fractional coordinate (measured 2.6e-7 on the 3^3 grid, clean h^2).
FD_FRACTIONAL_STEP = 1e-3
FD_FRACTIONAL_ATOL = 1e-6

# Weight derivatives of silicon. DERIVED, NOT MEASURED: no fractional-charge silicon
# run exists. The second-order one-sided stencil of 2a at delta = 1e-3 is kept, with
# the 2a tolerance. On the 3^3 grid (w_k = 1/27) the largest step, 2 delta, moves
# Z * 2 delta = 0.008 electrons: 0.22 in the single VBM state at Gamma, 0.11 in each
# state of the CBM pair, inside the capacity 2 of a state. A failure is raised with
# the project lead, not met by loosening (engineering log, 2b API).
SI_WEIGHT_STEP = 1e-3
SI_WEIGHT_ATOL = 1e-5
# The degenerate-frontier variant is looser (review, open question 1); same caveat.
SI_WEIGHT_ATOL_DEGENERATE = 5e-5
# Left minus right derivative equals Z times the gap over the grid: the occupations
# are exact functions of N, so only rounding separates them (2a: 1e-9 measured).
SI_WEIGHT_GAP_ATOL = 1e-8

# Monotonicity in the cutoff at a fixed FFT grid: no tolerance beyond SCF accuracy.
MONOTONE_SLACK = 1e-10

# Band-structure eigenvalues at grid k-points against the SCF eigenvalues: an exact
# identity (theory post: 1e-10), checked at SCF_TOL_TIGHT so that the difference
# between the last input density and the returned output density is negligible.
BANDS_VS_SCF_ATOL = 1e-10

# Clip-and-renormalise at convergence: rescaling by 1 + O(1e-16) only.
PROJECTION_AT_CONVERGENCE_MAX = 1e-12

# --- Milestone 2b cross-checks (not references) ------------------------------------
# From independent NumPy implementations written from the theory post: the throwaway
# check while preparing it and/or the review. They test the implementation against
# the same specification, not the physics.
# Point-charge Ewald energy at a = 10.263 Bohr: -8.3980 (post), -8.398009228 (review).
CROSS_CHECK_SI_EWALD = -8.398009228
CROSS_CHECK_SI_EWALD_ATOL = 5e-9
# Test-size total energy, N = 24: review only.
CROSS_CHECK_SI_ENERGY = -7.836151895
CROSS_CHECK_SI_ENERGY_ATOL = 1e-7
# Gap over the 2^3 grid at the test size: 0.44 eV (post), 0.443 eV (review).
CROSS_CHECK_SI_GRID_GAP_EV = 0.443
CROSS_CHECK_SI_GRID_GAP_ATOL_EV = 1e-3
# dE/df_0x at the displaced geometry, 3^3 grid, 12 Ha: 0.102076953 (post) and
# 0.102076953324 (review). At SCF_TOL the derivative carries ~1e-9 of SCF error.
CROSS_CHECK_SI_DEDF = 0.102076953
CROSS_CHECK_SI_DEDF_ATOL = 5e-8
