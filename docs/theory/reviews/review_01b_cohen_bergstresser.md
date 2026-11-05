# Review — `docs/theory/01b_cohen_bergstresser.qmd` (Milestone 1b, Cohen–Bergstresser silicon)

I rebuilt the model independently in NumPy from the post's own equations (fcc rows,
tau = +-(a/8)(1,1,1), `V(G) = cos(G.tau) * V_S(|G|^2)` with `V_S` in Ry halved to Ha,
k-dependent cutoff) and tested every quantitative claim against it. **The conventions
and the normalisation are correct** — the model reproduces the paper's Table I to
0.089 eV or better. The findings below are two sharp ones (a test that cannot
distinguish the quantity it is testing, and a stronger symmetry test available for
free), several engineering points, and answers to the five open questions.

## 0. Verified

| Claim | Result |
|---|---|
| `tau = +-(a/8)(1,1,1)` <-> `f = +-(1/8,1/8,1/8)` under the 1a row convention | correct |
| `G.tau = 2*pi*m.f` | correct (`B A^T = 2*pi*I`) |
| `S(G) = cos(pi(h+k+l)/4)`, vanishes identically on `|G|^2` = 4 and 12 | correct |
| `S` not constant on a shell: `-sqrt2/2` for (1,1,1), `+sqrt2/2` for (-1,1,1) | correct |
| Normalisation `V_tilde = S * V_S`, `V_S` = `v_hat`/`Omega_at`, Ry -> Ha | **reproduces Table I to <= 0.089 eV** |
| Degeneracies 1,3,3,1 at Gamma; all levels even-degenerate at X; 1,1,2,1,2 at L | correct |
| Gamma15 below Gamma2' in Si | correct (3.424 vs 3.889 eV) |
| L labels (L2', L1, L3', L1, L3) | **correct** — see finding 3; this one looks wrong and is not |
| CBM on Gamma–X, indirect gap | 0.855 X, 0.820 eV (post/@epnet2026: 0.85 X, 0.82 eV) |
| `E1 = 7 E0`, `E2 = 21 E0` = 1.3 and 3.9 Ha at a = 5.43 Å | correct (1.3123, 3.9369 Ha) |
| `(1,1,0)2pi/a` equivalent to X, so CB Fig. 1's path is L–Gamma–X–U\|K–Gamma | correct (differs by the all-odd (1,1,-1)) |
| "low bands converged well below 0.01 eV by about 10 Ha" | correct, and conservative — 6 Ha suffices (see finding 5) |
| `V_tilde(0) = 0` so `tr H = tr T`; `V_tilde(-G) = conj(V_tilde(G))`; translation is a unitary rephasing | all correct |
| Phase-sign argument (band energies cannot see it; check in real space) | correct |
| `@epnet2026` = arXiv:2610.14992 | exists, title matches exactly |

Converged values (20 Ha, 1139 PW) against Table I:

| | G25'-G15 | G25'-G2' | gap (Delta) | X4-X1 | L3'-L1 | L3'-L3 | G25'-L1 |
|---|---|---|---|---|---|---|---|
| ours | 3.424 | 3.889 | 0.820 | 3.954 | 3.129 | 5.235 | 1.876 |
| paper | 3.4 | 3.8 | 0.8 | 4.0 | 3.1 | 5.2 | 1.9 |
| deviation | 0.024 | **0.089** | 0.020 | 0.046 | 0.029 | 0.035 | 0.024 |

The 0.15 eV tolerance passes, with the least margin on Gamma25'–Gamma2'.

## 1. The indirect-gap acceptance test cannot distinguish the quantity it tests

The post is careful that "Gamma25' – X1 (Delta1 minimum)" means the Delta-line minimum,
not the energy at X. But the numbers are 0.820 eV at the Delta minimum and **0.949 eV at
X itself**, and the paper's value is 0.8. An implementation that evaluated the level at X
instead would deviate by 0.149 eV — **inside the proposed 0.15 eV tolerance**. The
acceptance test would pass on the wrong quantity, and it would pass for a reason
(a 0.13 eV difference between two different physical quantities) that has nothing to do
with the 0.15 eV error budget.

Fixes, all cheap:

1. Assert both: the Delta minimum against 0.8 eV, and `E(X1) - VBM` as a separate
   recorded value with no published counterpart.
2. Assert the **position** of the minimum, `0.855 +- 0.01` in units of X. That is sharp,
   has nothing to do with the 0.15 eV budget, and is exactly the quantity an
   implementation that confused the two would get wrong.
3. Name them differently in the test code (`gap_indirect_delta` vs `e_x1`), because the
   paper's label "X1" is used for both in the same table.

## 2. The X-point degeneracy holds on the whole square face, and the mechanism needs time reversal

The post asserts double degeneracy at X and says "it is checked at X". Measured with
the CB potential at `Ecut = 8 Ha`, along `k = (0, 1, z) 2pi/a` — the square (200) face,
from X at z = 0 to W at z = 0.5:

| z | 0.00 | 0.10 | 0.25 | 0.40 | 0.50 |
|---|---|---|---|---|---|
| lowest 8 levels pair up | yes | yes | yes | yes | yes |
| residual splitting within a pair | <= 7e-13 Ha | <= 6e-13 | <= 3e-13 | <= 4e-13 | <= 3e-13 |

At a generic interior k, (0.3, 0.7, 0.2), they do not pair. So the sticking is a property
of the entire square face, not of the X point. Testing it on the face is strictly
stronger: an implementation can get X right from the O_h invariance alone, and the face
test is the one that actually probes the non-symmorphic structure. Recommend testing at
three or four points on the face including a generic one (not X, not W).

The mechanism also deserves a clause rather than a name. The usual argument combines the
glide/screw operation — which squares to a non-lattice translation and so contributes a
phase of -1 on the zone face — with **time reversal**; the non-symmorphic group alone is
not the whole story. Worth saying explicitly, because the degeneracy lifts as soon as a
later milestone breaks time reversal (spin–orbit, magnetic field), and a reader who
learned it as "forced by the space group" will be surprised.

## 3. The L labels look wrong and are right — say why

The post's ordering at L, (L2', L1, L3', L1, L3), contradicts the ordering a reader is
likely to half-remember (L1 lowest). It is correct. I settled it by computing inversion
parity rather than by recall, and recommend the post do the same, because the BSW primes
mean different things at Gamma and at L and that is exactly what makes the ordering
look wrong.

Since the origin is an inversion centre and `-k` is equivalent to `k` at Gamma, X and L,
every level there carries a definite parity, computable as the permutation
`c(G) -> c(-G - 2k)` acting on the eigenvectors. Measured:

- Gamma: Gamma1 (+), Gamma25' (+), Gamma15 (-), Gamma2' (-)
- L: lowest (-), second (+), L3' pair (-), conduction singlet (+), conduction pair (+)

so the lowest level at L is odd and the second is even, which is (L2', L1, ...) under the
convention where the prime is odd at L — while at Gamma the prime is *even*
(Gamma25' is even, Gamma15 is odd). Stating the parities next to the labels makes the
post convention-free and resolves the apparent contradiction in one line.

This is also an invariant worth adding to the list. It is the only proposed check that
tests **eigenvectors**, it is well defined inside a degenerate level (it is a trace over
the level's projector, so it survives the eigenvector-rotation caveat from 1a), and it is
three lines of code. Note it does *not* detect the structure-factor phase sign — the
post's real-space test is still the right instrument for that.

## 4. The paper's own basis is far worse than "within 0.1 eV" suggests

The post plans to show how far the paper's raw `E1` basis is from convergence. The
numbers, so the validation document has an expectation (exact diagonalisation, no
Löwdin correction):

| basis | N_PW(Gamma) | G25'-G15 | G25'-G2' | gap | X4-X1 | L3'-L1 | L3'-L3 | G25'-L1 |
|---|---|---|---|---|---|---|---|---|
| CB `E1` = 7 E0 | **15** | 3.259 | 3.431 | **0.412** | 4.083 | 2.916 | 5.334 | 1.348 |
| CB `E2` = 21 E0 | 113 | 3.423 | 3.884 | 0.814 | 3.948 | 3.124 | 5.229 | 1.875 |
| 6 Ha | 181 | 3.424 | 3.889 | 0.820 | 3.954 | 3.129 | 5.235 | 1.876 |
| 20 Ha | 1139 | 3.424 | 3.889 | 0.820 | 3.954 | 3.129 | 5.235 | 1.876 |

The `E1` basis is 15 plane waves and errs by up to 0.41 eV (the indirect gap is off by a
factor of two); `E2` is already converged to ~5 meV except on Gamma2'. So the Löwdin
correction is doing real work, and the agreement we obtain (<= 0.089 eV) is evidence that
CB's second-order treatment was good, not that their basis was adequate. Worth one
sentence, since the post currently leaves the impression that `E1` plus a small
correction is nearly converged.

A caveat on the tolerance derivation: "the paper states its own basis converged within
0.1 eV" is a statement about **levels**, while every entry in Table I is a **difference**
of two levels, which could carry up to 0.2 eV if the two errors are uncorrelated. The
0.15 eV figure is fine empirically (worst observed deviation 0.089 eV), but the
derivation should either note that the errors are correlated (both levels are pushed up
by truncation) or round the budget up. Also worth a second look at the Gamma25'–Gamma2'
transcription: our converged 3.889 rounds to 3.9, the table says 3.8, and it is the
tightest entry in the acceptance set.

## 5. Convergence: 6 Ha is enough, which makes the test suite cheaper

Measured on all seven Table I splittings: 4 Ha (113 PW) is within 18 meV, **6 Ha (181 PW)
within 1 meV**, and nothing moves after that. The post's "well below 0.01 eV by about
10 Ha" is true but costs 411 plane waves. Since `CLAUDE.md` requires the suite to run on
a laptop CPU, state 6 Ha as the production test cutoff and keep 10-12 Ha for a single
convergence figure in the validation document.

## 6. Engineering

**6a. A dense `V` contradicts the matrix-free mandate, and the memory figure to quote is
per-k.** The post says the dense build costs `O(N_pad^2)`, "fine at these cutoffs". Under
`vmap` it is `O(n_k N^2)`:

| Ecut | N | dense V, 100 k-points (complex128) | 44-neighbour gather indices |
|---|---|---|---|
| 8 Ha | 283 | 0.13 GB | 5.0 MB |
| 12 Ha | 537 | 0.46 GB | 9.5 MB |
| 20 Ha | 1139 | 2.08 GB | 20.0 MB |

Measured sparsity at `Ecut` = 12 Ha, N = 537: **20 to 44 non-zeros per row, median 33** —
about 94% sparse. The bound is the shell multiplicity 8 + 12 + 24 = 44, **independent of
cutoff**, so the sparsity only improves as the basis grows. A gather over <= 44 neighbours
is `O(44 N)` per apply instead of `O(N^2)`, is what `CLAUDE.md` means by a matrix-free
operator, and is the structure that the FFT application replaces later. Build the dense
matrix too — from the operator, as in 1a — for the agreement test, but do not make it the
operator. At 20 Ha the difference is 2 GB versus 20 MB on a 24 GB laptop.

**6b. Use the post's own integer identity in the lookup.** The post derives
`G.tau = 2*pi*m.f` and then describes building `V_tilde` from Cartesian differences. For
this crystal both the shell index and the structure factor are exactly integer-valued:
with `G = mB` for fcc, the cubic indices are
`(h,k,l) = (-m1+m2+m3, m1-m2+m3, m1+m2-m3)`, so `|G|^2 (a/2pi)^2 = h^2+k^2+l^2` is an
exact integer and `S = cos(pi(h+k+l)/4)` takes eight exact values. Keying the shell
lookup on integer arithmetic removes the entire class of cutoff-boundary round-off that
the 1a post worried about, and it makes the "`V` differs within a shell" test exact. Keep
a float assertion that `|G|^2` matches the integer, for the general-cell path.

**6c. Keep complex128.** The post notes `V_tilde` is real for the symmetric origin. That
is a test, not an optimisation: a general origin, or unequal weights, makes it complex,
and the translation-invariance invariant the post proposes *requires* the complex path to
work. Say so, or someone will specialise to real symmetric and break the origin test.

**6d. The real-space phase check needs no FFT grid.** FFT grids are out of scope for 1b,
but the inversion-symmetry test synthesises `V(r)`. Direct summation over the <= 44
non-zero `G` at a handful of points is enough and keeps the scope boundary clean. Worth
one clause so the test does not pull FFT into 1b.

**6e. Degeneracy grouping needs its own tolerance.** Symmetry-forced degeneracies come
out at ~1e-13 Ha (~3e-12 eV) in my runs, so 1e-8 Ha is safe and is nine orders away from
the 0.15 eV physics tolerance. State both numbers so they are not confused.

**6f. "Every level at X is even"** in the invariant list reads as a parity statement next
to a discussion of parity. Say "has even degeneracy".

**6g. "A gap between bands 4 and 5 over the whole path"** tests the path, not the zone.
True as written; worth adding that the path covers Delta, where the minimum actually is,
so the test is meaningful rather than lucky.

## 7. The five open questions

**1. Published reference values — keep Table I as acceptance; do not promote
@epnet2026.** I confirmed arXiv:2610.14992 exists with exactly that title, but two
reasons not to make it a test: it is an unrefereed preprint, and more importantly it is
not independent evidence — we reproduce the same numbers ourselves (0.855 X, 0.820 eV),
so citing it as a reference value is circular. Quote it in the validation document as
corroboration that an independent group gets the same numbers from the same model, and
keep acceptance on the primary source plus the CBM-position assertion from finding 1.

**2. Reference code — skip DFTK for 1b.** Table I already discriminates the normalisation
overwhelmingly. Scaling `V` by 2 (a Ry/Ha slip) moves Gamma25'–Gamma15 from 3.42 to
8.29 eV and the indirect gap from 0.82 to 7.69 eV; scaling by 1/2 gives 1.20 eV and closes
the gap entirely (-2.78 eV, a metal). Both are tens of tolerances away. A factor of
`Omega` would be worse still. Rung 3 would cross-validate the *solver* at meV level, and
1a already did that analytically. Installing Julia also imports a problem the post
identifies honestly — DFTK's own form-factor normalisation is undocumented and would have
to be read — so it is not a free reference. Defer Julia to the SCF milestone, where
there is no analytic or published reference to lean on.

**3. Structure representation.** Agree with raising it separately as an API question. One
constraint from this post: `G.tau = 2*pi*m.f` only holds if positions are stored
fractional, so "fractional at every public interface" should be an invariant of the
`Crystal` pytree, not a convention — a Cartesian-positions constructor should convert at
construction and never store Cartesian.

**4. Where the CB parameters live.** Agree with `jaxdft/pseudo/`. Add that the element
should expose the tabulated data in publication units *and* the physical shell radii it
was tabulated at (see question 5), since the guard needs the latter.

**5. Off-shell behaviour — raise, but the stated reason is subtly wrong and the guard
must be in physical units.** "Strain the cell and the `|G|` move off the tabulated
values" is not what happens under isotropic scaling: the dimensionless shell indices stay
exactly 3, 8, 11 in units of `(2pi/a)^2` for any `a`. An implementation keyed on the
integer index would therefore accept **any** lattice constant silently and return a
plausible but meaningless model — which is a worse failure than the one the post is
guarding against. There are two independent reasons the model is fixed-cell, and the post
states only the first:

1. `v_hat(q)` is sampled only at three physical `q`, namely
   `sqrt(3), sqrt(8), sqrt(11) * (2pi/a_CB)` with `a_CB` = 5.43 Å;
2. `V_S = v_hat / Omega_at` carries an explicit `1/Omega`, so even at fixed `q` the
   tabulated number is tied to the cell volume.

So the guard should compare **physical** `|G|^2` in Bohr^-2 against
`{3, 8, 11} * (2pi/a_CB)^2`, not the dimensionless index, and the relative 1e-8 tolerance
is the right order. Under anisotropic strain the shells split and the index stops being
an integer, which the integer-keyed version would also miss. Worth stating both reasons:
the second is the one that would bite if someone later tried to "fix" the first by
interpolating `v_hat`.
