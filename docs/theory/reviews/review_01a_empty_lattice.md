# Review — `docs/theory/01a_empty_lattice.qmd` (Milestone 1a, empty lattice)

Reviewer pass over the theory post before implementation. Everything quantitative in
the post was re-derived independently (enumeration of the bcc reciprocal lattice, basis
counts for fcc with a = 10.26 bohr). **No physics error found.** The items below are
(A) statements that are wrong or imprecise as written, (B) details the post hides that
it should not, (C) engineering issues that will cost rework if settled the way the post
proposes, and (D) answers to the five open questions.

## 0. What was verified

| Claim in the post | Result |
|---|---|
| `A` rows for fcc, `Omega = det A = a^3/4` | correct |
| `B = 2*pi*(A^-1)^T` = bcc, `(2*pi/a)[[-1,1,1],[1,-1,1],[1,1,-1]]`, `A B^T = 2*pi*I` | correct |
| bcc points = integer triples all-even or all-odd | correct (every component has the parity of `m1+m2+m3`) |
| Fractional coordinates of Gamma, X, L, W, K, U under `k = f B` | all six correct |
| Degeneracy tables at Gamma, X, L, W, K, U (six levels each) | all 36 levels and degeneracies correct |
| K and U have identical spectra | correct |
| `N_PW ~ Omega (2 Ecut)^{3/2} / (6 pi^2)` | correct (sphere volume / BZ volume) |
| k-independent sphere exact only to `(Gmax - |k|)^2 / 2` | correct |
| Padding superset `|G| <= Gmax + max|k|` | correct and sufficient |
| `B_{k+G0} = B_k - G0` | correct |
| `||H|| <= Ecut`, so eigenvalue error ~ 1e-13 Ha | correct, but see A5 |

The U|K equivalence is worth showing rather than asserting:
`U - (2*pi/a)(1,1,1) = -(2*pi/a)(0, 3/4, 3/4)`, and `(1,1,1)` is a valid (all-odd) bcc
vector; inversion plus a cyclic permutation of the axes maps that to
`(2*pi/a)(3/4, 3/4, 0) = K`. One line, and the reader can check it.

## A. Corrections

**A1. "a fixed |G| sphere ... loses exactness and nesting at fixed k" (open question 1).**
Nesting is *not* lost. `B(Gmax)` is a subset of `B(Gmax')` at fixed k for
`Gmax < Gmax'` just as it is for the k-dependent sphere, so Courant-Fischer
monotonicity holds either way. What the fixed sphere actually loses is (i) the
completeness statement (exact only below `(Gmax - |k|)^2 / 2`) and (ii) a k-independent
basis quality: states at large `|k|` sit in an off-centre sphere and are described
worse than states near Gamma.

The post should also state the cost of the convention it *chooses*, which it currently
omits: because `N_PW(k)` changes as k moves along a path, the basis changes
discontinuously along the path, so in 1b band energies acquire small jumps of the order
of the variational error wherever a plane wave enters or leaves the sphere. This is a
known artifact of the standard convention, invisible in 1a (every eigenvalue below the
cutoff is exact) and visible in 1b. Say so now so it is not debugged as a bug later.

**A2. "invariant under the 48 operations of O_h ... and under k -> -k (time reversal)".**
Inversion is an element of O_h (m-3m), so for fcc the time-reversal statement is implied
by the point-group statement, not an independent check. Either drop it or mark it as
"independent only for point groups without inversion".

**A3. Scaling invariant `a^2 * eps` independent of `a`.**
True for the exact folded bands. As a *test on computed eigenvalues* it holds only if
`Ecut` is scaled as `a^-2` together with the lattice, or if the comparison is restricted
to bands below the smaller of the two cutoffs. At fixed `Ecut` a change of `a` changes
which plane waves are in the basis and how many bands are exact. State the condition.

**A4. "`N_PW` approaches the asymptote, with relative error shrinking as `Ecut` grows".**
The relative error is bounded by `O(Ecut^{-1/2})` but is not monotone and changes sign.
Measured at Gamma, a = 10.26 bohr:

| Ecut (Ha) | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|
| `N_PW(Gamma)` | 137 | 411 | 1139 | 3287 | 9257 |
| asymptote | 144.2 | 407.8 | 1153.5 | 3262.6 | 9228.1 |
| relative error | -5.0% | +0.8% | -1.3% | +0.7% | +0.3% |

A property test asserting monotone shrinkage will be flaky. Assert a bound,
`|N/N_asym - 1| <= C * Ecut^{-1/2}` with `C` measured once and a stated margin.

**A5. Tolerance argument.**
`||H|| <= Ecut` holds *because of* the k-dependent cutoff convention (every diagonal
entry is at most `Ecut` by construction); with the fixed-|G| sphere the bound would be
`(Gmax + |k|)^2 / 2`. Worth one clause, since it is a second argument for the chosen
convention. With sentinel padding, `||H||` becomes the sentinel value, which the
`eps_mach * ||H||` argument would say destroys the tolerance. Empirically it does not:
for an exactly decoupled block-diagonal matrix (300 physical, 100 padding rows, physical
spectrum spanning 20 Ha) LAPACK returned the physical eigenvalues to 1.9e-13 Ha for
sentinels of 4e1, 1e4, 1e8 and 1e12 alike. So the dense path is safe, but the argument
in the post as written is not the reason. Either sharpen it (the realized error for a
decoupled block is governed by the norm of that block) or keep the sentinel modest.
The conditioning does matter for the iterative solvers in 1c: a 1e12 sentinel gives
LOBPCG a spectrum spread of twelve orders of magnitude. Recommend a sentinel a small
multiple of `Ecut`.

**A6. "Rayleigh-Ritz principle" for *every* eigenvalue** is the Courant-Fischer min-max
theorem. Name it, since the claim is about the n-th eigenvalue, not the lowest.

**A7. "the basis-level form of the variational invariant in `CLAUDE.md`".**
`CLAUDE.md` states monotonicity of the *total energy*. Per-eigenvalue monotonicity is a
stronger and different statement; the total-energy version follows from minimising over
a nested space, not from the eigenvalues. Half a sentence to distinguish them.

## B. Details the post hides

**B1. Bloch's theorem to the plane-wave sum.** The post asserts the expansion. The
honest half-line: Bloch's theorem gives `psi = exp(i k.r) u(r)` with `u` lattice
periodic; expanding `u` in its Fourier series over the reciprocal lattice *is* the sum
over G. Then the restriction to G (rather than arbitrary wavevectors) is a consequence,
not a convention. One extra sentence for why different k do not mix: `H` commutes with
the lattice translations, and plane waves belonging to different k in the first BZ are
orthogonal.

**B2. Which space the cutoff sphere lives in.** "a sphere of radius `Gmax` centred on
`-k`" is correct but is the sentence readers trip over. Add: equivalently, the set of
`q = k + G` inside a sphere of radius `Gmax` centred at the origin. Both pictures are
used in the code (the first for enumeration bounds, the second for the energy).

**B3. The integer bound for enumeration.** "checked against brute enumeration" hides the
one place a plane wave can silently go missing, and the brute-force reference needs the
same bound. Since `G.a_i = 2*pi*m_i`, the bound `|m_i| <= (Gmax + |k|) * |a_i| / (2*pi)`
is tight and provable; a guessed symmetric cube is not safe for a skewed cell. This is a
correctness condition and belongs in the theory post, with the one-line derivation.
Matching test: enumerate over a box twice as large and assert no additional G satisfies
the cutoff.

**B4. Boundary ties at the cutoff.** The `<=` in the basis definition is an exact float
comparison on `|k + mB|^2 / 2`. Computed by two algebraically equivalent routes (via `B`
versus in the cubic frame) the shell energies differ by about 4e-16 relative. At Gamma,
X and L the shells are 6- to 24-fold degenerate, so an `Ecut` placed on a shell flips
that whole shell in or out at once and the "count matches brute enumeration" test fails
spuriously. Two cheap fixes, both worth stating in the post: choose test cutoffs
strictly between shells (with the margin stated), and have the count comparison use a
tolerance window (count below `Ecut(1-delta)`, assert nothing missing below
`Ecut(1+delta)`). The reference enumeration must use the same arithmetic route as the
code, or the test is measuring round-off.

**B5. Fourier convention for a local potential.** The post defers `V(G - G')` to 1b with
no normalisation. The factor that goes with the `Omega^{-1/2}` plane-wave normalisation
chosen here (`V_{GG'} = (1/Omega) * integral over the cell of V(r) exp(-i(G-G').r)`) is
the single most common factor-of-Omega bug in plane-wave codes, and it is free to fix
now, in the post that defines the normalisation. One equation.

**B6. What "n bands" means when `N_PW(k)` varies.** Requesting `n_bands` under `vmap`
needs `n_bands <= min_k N_PW(k)`, and the exactness claim covers only eigenvalues below
`Ecut`, which is a k-dependent count. "The first n bands are exact" is not implied by
what the post proves. Make the restriction explicit where the test is stated.

**B7. Forward pointer on cutoffs.** One sentence that the density/FFT grid in 1c needs a
cutoff of `4 * Ecut` (QE's `ecutrho`), because `rho = |psi|^2` doubles the largest G.
Costs nothing here and prevents confusion later.

## C. Engineering

**C1. The proposed padding is about 2.5x more expensive than necessary, because `vmap`
needs a common *shape*, not a common *set*.** The post proposes one candidate set of G
large enough for every k, plus a per-k mask. But the arrays handed to `vmap` can hold a
*different* G list per k, as long as they have the same length: shape
`(n_k, N_pad, 3)`, padded to `N_pad = max_k N_PW(k)` plus a margin. `N_pad` is a host-side
Python int computed before tracing, exactly as in the shared-set scheme, so nothing
about jit changes.

Measured for a = 10.26 bohr over the six high-symmetry points:

| Ecut (Ha) | `N_PW` range over k | shared candidate set | ratio | dense `eigh` cost |
|---|---|---|---|---|
| 5 | 137-150 | 259 | 1.73 | 5.2x |
| 10 | 409-412 | 609 | 1.48 | 3.2x |
| 20 | 1139-1162 | 1591 | 1.37 | 2.6x |
| 40 | 3258-3287 | 4015 | 1.22 | 1.8x |
| 80 | 9204-9272 | 10849 | 1.17 | 1.6x |

The true spread across k is about 2%; the shared candidate set costs 17-73% in basis
size and 1.6-5.2x in dense diagonalisation, worst at the small cutoffs used in tests.
The masking machinery is identical in both schemes. The only thing a shared set buys is
a single shared index map onto the FFT grid in 1c, and a per-k index array vmaps just as
well. Recommend per-k G lists padded to a common length; at minimum, put both numbers in
the post so the choice is made with them in view.

Note also that `max_k |k|` is only bounded if k-points are folded into the first BZ; for
fcc the bound is the BZ circumradius `|W| = sqrt(5)/2 * (2*pi/a)`.

**C2. "Exactly decoupled" is a physics requirement, not an implementation detail.**
Spell out the concrete requirement: masked rows *and columns* must be zeroed, not merely
given a large diagonal. In 1a this is invisible (H is diagonal anyway); in 1b
`V(G - G')` will couple padding into the physical block and produce eigenvalues that are
wrong but entirely plausible. The strongest test is solver-independent and cheap: run
the same k with two different `N_pad` and assert the requested bands agree to the stated
tolerance. That catches a leak in the dense and the iterative path alike, and is better
than checking that padding eigenvalues come out high.

**C3. 1a does not actually test the eigensolver or the operator/dense agreement.**
`H` is diagonal, so `eigh` only sorts and `apply_H` versus dense is a diagonal multiply.
The stated goal (fix the interface before 1b) is met, but the test list should say that
these tests do not yet exercise the non-trivial path, and add one unit test that pushes a
random Hermitian matrix through the same dense-build / apply / diagonalise interface.
A non-zero `V_0` does not help here (a constant potential is still diagonal), though it
is a one-line test worth having for the sign and the units.

**C4. Residual tolerance should be relative.** `||Hc - eps*c|| ~ 0` with an absolute
1e-10 stops being meaningful as `Ecut` grows. Use `<= tol * ||H||` or normalise by
`|eps|`.

**C5. Two cheap invariants missing from the list.** `N_PW(S k) = N_PW(k)` for
`S` in O_h, and `N_PW(k + G0) = N_PW(k)`: both are sharp tests of the enumeration bound
in B3, and they fail loudly when the integer box is too small in a skewed cell.

## D. The five open questions

**1. Cutoff convention — keep the k-dependent sphere.** It is what DFTK, QE
(`ecutwfc`, Ry) and ABINIT (`ecut`, Ha) do, it is what makes "every eigenvalue below the
cutoff is exact" true, and it is what makes `||H|| <= Ecut` true. Fix the wording in A1
and add the band-path discontinuity caveat.

**2. Row convention — keep rows.** It matches the whole Python ecosystem (ASE `cell`,
pymatgen `Lattice.matrix`, spglib) and it is what makes `G = m @ B` and `k = f @ B`
read naturally with numpy broadcasting. DFTK is the outlier; handle it with one
transpose at the comparison boundary, with a round-trip test, and say so in the post
rather than in a comment.

**3. k-path — use the Si path for the plot, but test W explicitly.**
L-Gamma-X-U|K-Gamma never visits W, yet W is in the degeneracy table, is the only point
with a 4-fold lowest level, and is the point that sets `max |k|`. Recommend: the
scientific tests check all six high-symmetry points as isolated k-points (cheap and
exact, and it validates the table), and the plot keeps the Si path for continuity with
1b. The longer path is optional; the isolated points give the same coverage for less.

**4. Plot units — agree, with an addition.** `E_0 = (2*pi/a)^2 / 2` makes the analytic
levels the small rationals in the table and makes the axis ticks land on them, which is
the pedagogical win. Also print the Hartree values somewhere in the validation document,
since the reference codes report Ha or eV and 1b needs the conversion anyway.

**5. Padding — per-k G lists padded to a common length, not a shared candidate set.**
See C1 for the measured cost, and C2 for the decoupling requirement and the
`N_pad`-invariance test that should accompany whichever scheme is chosen.

## E. Smaller points

- The YAML `status: "In review"` is not a Quarto key; harmless, but if the site renders a
  status badge it will need a filter or a listing field.
- "the conventional path jumps from U to K without a gap in the bands": true for the
  values; the slopes are discontinuous there. Worth the half-clause, since the kink is
  visible in the plot and readers will ask.
- The degeneracy table is described as "only for orientation and review" but it is in
  fact the most falsifiable content in the post. It is worth promoting it to a scientific
  test (it is, under invariant "the degeneracies at Gamma, X, L, W, K and U match the
  table") and stating the tolerance used to group degenerate levels, which is a separate
  number from the 1e-10 eigenvalue tolerance and should be larger.
