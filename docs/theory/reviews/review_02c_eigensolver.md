# Review: `docs/theory/02c_eigensolver.qmd` — Milestone 2c, eigensolver cost, accuracy and memory

**Bottom line.** Every number the post quotes reproduces, several of them exactly: all
eight rows of the stopping-test table (370 / 203 / 330 / 159 / 344 / 226 / 390 / 160),
both memory figures (0.2323 GiB for one k-point, 9.4451 GiB for 41), `n_pad = 548` and
the $24^3$ grid, and the cluster-scan sums. The interlacing bound on $\sigma$ is
correct as stated, and the adaptive-tolerance rule is essentially the one Quantum
ESPRESSO has used for twenty years — including the refinement branch (a), which is
QE's "threshold on eigenvalues was too large: diagonalizing with lowered threshold".

Three things should change before implementation.

1. **The diagnosis of the residual floor is not established by the evidence given.** The
   penalty is applied *only* when a direction is dropped, and on this laptop drops occur
   at iteration 1 of every call and never again — so lowering $\sigma$ changes the
   trajectory without ever changing a selected eigenvector. In a forced 200-iteration
   run the two $\sigma$ end a factor $2\times10^{6}$ apart in final residual
   ($1.13\times10^{-7}$ against $5.04\times10^{-14}$) with the penalty numerically
   inactive in both: exactly the false positive the single GPU run per variant is
   vulnerable to. Keep the fix (it is derived and free), but
   describe it as a bound worth having, not as the cause, and instrument the drop
   count to settle it.
2. **The real failure mode is not a floor but a blow-up.** Asked for a residual it
   cannot reach, the solver does not sit at $10^{-13}$: it climbs back to $10^{-7}$ and
   wanders there for the remaining 140 iterations, with the eigenvalues still correct to
   $3\times10^{-14}$ (the $r^2/\Delta$ law) and the *vectors* wrong at $2\times10^{-6}$.
   There is no stagnation stop and no best-iterate return. That, not $\sigma$, is what
   turns a tight tolerance into a 300-iteration run on either device.
3. **`EIGEN_FLOOR` = $2\times10^{-14}$ is below what the solver attains.** At the test
   size the best achievable max-over-bands residual is $2\!-\!5\times10^{-14}$, so any
   `tol` below $4\times10^{-13}$ asks `lobpcg` for the unattainable and gets the
   blow-up of point 2. Acceptance criterion "the penalty fix should make `EIGEN_FLOOR`
   the floor of the solver as well" will fail as written.

The mechanism paragraph for stopping on the needed bands is right in outcome but too
general in its reasoning; §2.4 gives the k-resolved measurement. Nothing in the
milestone is physics in disguise: all eight configurations I ran agree on the total
energy to $10^{-13}$ Ha.

**Method.** Unlike earlier reviews in this series I did not rebuild the physics in
NumPy — this milestone is about the code's own numerics, so the checks run jaxdft
itself out of the repository's `.venv` (JAX 0.12.0.1, CPU, float64, commit `3383ca3`,
Apple-silicon LAPACK) plus an instrumented copy of `lobpcg`'s loop body with switchable
penalty, switchable stopping set and per-iteration tracking, driven through the real
`SCFOptions(eigensolver=…)` extension point. Caveats: one machine, one LAPACK, no GPU,
and the laptop's `eigh` is not the cluster's. The adaptive-tolerance table (Table 2) was
**not** reproduced — it needs the unwritten rule inside `F` — so §2.7 argues from the
reference codes and from a direct measurement of refinement cost instead.

---

## 1. Verification table

| # | Claim in the post | Check | Result |
|---:|:--|:--|:--|
| 1 | Stopping table, $\sum\max_k I_k$: 370, 203, 330, 159, 344, 226, 390, 160 | eight SCF runs, real `scf`, instrumented solver | **370, 203, 330, 159, 344, 226, 390, 160** — all eight exact |
| 2 | "The total energy and the number of SCF steps are the same in every row" | same runs | 8 steps on $3^3$, 9 on $4^3$; energies $-7.908\,821\,857\,778\,4$ and $-7.925\,085\,649\,325\,3$, spread $<3\times10^{-14}$ Ha ✓ |
| 3 | "stopping on the needed bands alone gives 1.5–1.8 times less wall time" | same runs | 1.76, 1.82, 1.49, 2.23 (this laptop; absolute times 8.6/4.9/10.6/5.8/17.4/11.7/27.6/12.4 s vs the post's 7.3/4.1/9.5/4.7/16.5/11.0/31.6/14.7) ✓ |
| 4 | Three buffer bands cut iterations but cost more per iteration than they save | same runs | $3^3$: 203→159 iterations (−22%), 4.9→5.8 s (+18%); $4^3$: 226→160 (−29%), 11.7→12.4 s (+6%). Break-even at $4^3$ on CPU ✓ |
| 5 | `eig_ok` checks only the first `n_bands` columns; `lobpcg` stops on all `n_solve` | source | ✓ `scf.py` slices `eig_res[:, :n_bands]`; `lobpcg`'s `cond` uses `jnp.max(rnorm)` |
| 6 | $\tau = \max(0.05\,\varepsilon_\text{SCF}, 2\times10^{-14})$, "about $5\times10^{-11}$ by default" | source | ✓ `EIGEN_TOL_FRACTION=0.05`, `EIGEN_FLOOR=2e-14`, `tol=1e-9` |
| 7 | The SCF asks for $5\times10^{-14}$ at its tightest tolerance | `SCF_TOL_TIGHT=1e-12` | ✓ $\tau=5\times10^{-14}$; `eig_ok` then demands $<10^{-13}$ |
| 8 | A dropped column is exactly zero, so $T$ is block diagonal and the dropped part is $\sigma\mathbb{1}$ | traced through `_gram_orthonormalise` twice | ✓ — and the zero columns are the *leading* ones in both passes (eigh sorts ascending), so `keep & keep2` indexes them consistently |
| 9 | Cauchy interlacing: $\lambda_j(\text{kept}) \le \varepsilon_j$, so any $\sigma>\varepsilon_k$ suffices | $X^\dagger HX=\operatorname{diag}(\varepsilon)$ is a principal submatrix of the kept part | ✓ correct as stated (see §2.3 for the three caveats) |
| 10 | $\sigma=10^3(\max|T|+1)$ makes $\lVert T\rVert$ a thousand times that of the problem | measured $\lVert T\rVert_2$ at iteration 1 of each call | ✓ 3503–7575 with the current rule vs 3.1–9.9 with the bound — a factor 350–1000, at **every** call |
| 11 | $\sigma=2\max|T|$ or moving the RR `eigh` to the CPU removes the GPU floor | `reference/gpu_lobpcg_floor` | ✓ data say what the post says (`gpu/rr` 49 it/4.3e-14, `gpu/penalty` 52 it/2.9e-14, vs `gpu/jaxdft` 300 it/2.9e-12) — but see §2.1 |
| 12 | "at residual targets near $10^{-13}$ the CPU stalls too, and the bounded $\sigma$ removes the stall" | 6 bands, Si $2^3$ k-point 1, $\tau=10^{-13}$ and $5\times10^{-14}$ | **not reproduced as a $\sigma$ effect**: 49–50 iterations and $3\!-\!8\times10^{-14}$ for $\sigma\in\{10^3(\max|T|+1),\,2\max|T|,\,\text{bound}\}$, identical to 2 digits. §2.1 |
| 13 | `EIGEN_FLOOR` is "the floor of the operator", $\lVert H\rVert\epsilon$ | $\lVert H\rVert_2=11.69$ Ha, sentinel 24 Ha | $\lVert H\rVert\epsilon_\text{mach}=2.6\times10^{-15}$ (physical) to $5.3\times10^{-15}$ (padded) — the constant is 4–8× larger than the estimate, and still below what is attainable. §2.2 |
| 14 | 0.23 GiB of temporaries for one k-point, 9.45 GiB for 41, $n_\text{pad}=548$, grid $24^3$ | `memory_analysis()` on the lowered `_solve` | **0.2323 GiB, 9.4451 GiB, 548, $24^3$** ✓ exact; and 9.4451 GiB matches the V100's "9.44GiB" allocation |
| 15 | Peak memory of the SCF grows linearly with $n_k$ | biocloud/aicloud scans | ✓ host 11.9 MB/k ($n=4$) and 12.5 MB/k ($n=8$); GPU device memory 7.4 MB/k, 0.089→3.776 GB from 1 to 512 k-points |
| 16 | $\sum\max_k I_k$ "rises from 343 at $4^3$ to about 600–700 at $8^3$, with the same number of SCF steps" | both scans | ✓ 343→697 (biocloud), 345→604 (aicloud), 9 steps throughout. Not monotone: $3^3$ gives 347/409, above $4^3$ |
| 17 | "the time per k-point and SCF step grows with the number of k-points" on CPU and V100 alike | both scans, per k per step | CPU 21.7→121.4 ms (5.6×) ✓. **V100: 6.17→7.84 ms from $n=2$ to 8 (1.27×), non-monotone (5.08 at $n=4$).** §2.6 |
| 18 | 634 of 642 tests pass on the V100 | `reference/aicloud_pytest` | ✓ 634 passed, 3 failed, 5 errors; the OOM is `test_conduction_minimum_is_interior_to_gamma_x` |
| 19 | `jax.lax.map` with `batch_size` is available (JAX 0.12) | `inspect.signature(lax.map)` | ✓ `map(f, xs, *, batch_size=None)`; remainders handled by `_batch_and_remainder` — see §2.9 |
| 20 | "the eigenvalue converges quadratically in the residual" | forced run: $r=1.13\times10^{-7}$, gap 0.05 Ha | eigenvalue error $2.9\times10^{-14}$, bound $r^2/\Delta=2.5\times10^{-13}$ ✓ |
| 21 | `knyazev2001`, `parlett1998` in the bibliography | `docs/references.bib` | ✓ both present |

---

## 2. Findings

### 2.1 The penalty is active once per call, not once per stall — and the floor has not been pinned on it

The post's causal chain is: drops → $\sigma=10^3(\max|T|+1)$ → $\lVert T\rVert$ inflated
→ `eigh` error $\epsilon_\text{mach}\lVert T\rVert$ → residual floor. Two measurements
complicate it.

**Where drops actually happen.** In every CPU run I traced — cold start, warm start,
tight tolerance, forced over-iteration — directions are dropped at **iteration 1 and
nowhere else**. That is structural, not numerical: at the first iteration $P=0$, so half
of $[W,P]$ is identically zero and all $k$ of those columns are dropped. Afterwards the
Gram test `lam > 1e-12 * max(lam)` never fires again, in 200 iterations, including the
140 spent grinding past the attainable residual. So on this machine the penalty acts
exactly once per `lobpcg` call — which, because the SCF warm-starts with $P=0$ every
step, means once per SCF step per k-point. Measured there, it is dramatic:
$\lVert T\rVert_2 = 3503\ldots7575$ at that first iteration against $3.1\ldots9.9$ with
the proposed bound.

**But it costs nothing measurable on this CPU.** Refining a converged block
(§2.8), the residual after that first, penalised iteration agrees to seven digits
between the two rules ($4.490\,027\,669\times10^{-6}$ vs $4.490\,027\,462\times10^{-6}$),
and the refinement then lands at $2\!-\!5\times10^{-14}$ either way. At $\tau=10^{-13}$
and $\tau=5\times10^{-14}$ the three penalties give 49–50 iterations and the same final
residual to two digits (table row 12). The post's sentence "at residual targets near
$10^{-13}$ the CPU stalls too, and the bounded $\sigma$ removes the stall" does not
reproduce here as a $\sigma$ effect.

**Why the GPU evidence is weaker than it looks.** Forced to 200 iterations at an
impossible tolerance, my two runs end at $1.126\times10^{-7}$ ($\sigma=10^3$) and
$5.045\times10^{-14}$ (bound) — a factor $2.2\times10^{6}$ apart (over the last 20
iterations, medians $7.6\times10^{-9}$ and $2.6\times10^{-14}$, a factor
$2.9\times10^{5}$) — with the penalty numerically inactive
after iteration 1 in both and the same 6 drops in both. That is a *trajectory*
difference, not a penalty effect, and it is precisely the artefact a single run per
variant cannot exclude. The reference README itself notes the stalled residual moved
between $2.9\times10^{-12}$ and $4.2\times10^{-12}$ across runs. Note also that
`gpu/gram` (the Gram `eigh` on the CPU) did *not* help while `gpu/rr` did: if noisy
drop detection were the channel, moving the Gram `eigh` should have mattered most.

**Recommendation.** Keep the bounded $\sigma$ — it is derived, it costs one subtraction,
and it removes a 1000× inflation of $\lVert T\rVert$ that nothing justifies. But
(i) rewrite the paragraph to claim only that, not the cause of the floor; (ii) add the
number of dropped directions per iteration to the solver's diagnostics and record
whether late drops ever occur on the V100 — that single number decides the mechanism;
(iii) before accepting the fix on the GPU, repeat the diagnosis **three times per
variant**, since the quantity being measured varies between runs by more than the
effect at stake.

### 2.2 Past its attainable residual the solver blows up, not stalls

Forced to 200 iterations with an unreachable tolerance, the tracked residual goes
$1.6\times10^{-5} \to 1.3\times10^{-10} \to 6.2\times10^{-13}$ (iteration 60) and then
**back up to $6\times10^{-8}$**, after which it oscillates over five orders of magnitude
— sampled every 20 iterations: $1.7\times10^{-14}$, $5.4\times10^{-10}$,
$3.0\times10^{-9}$, $4.1\times10^{-9}$, $1.9\times10^{-8}$, $1.1\times10^{-7}$. Both
penalty rules do this (the bounded $\sigma$ run rises to $1.9\times10^{-8}$ at iteration
80 and happens to be recovering at iteration 200); where the run is when it stops is
luck. The eigenvalues remain accurate to $2.9\times10^{-14}$ throughout (the
$r^2/\Delta$ law),
so the damage is entirely in the vectors — the orbitals are off by $r/\Delta\sim
2\times10^{-6}$, and the density is built from the vectors.

Consequences for the milestone:

- The "cap of 300 iterations in three SCF steps (1040 iterations, 21 s)" in the
  post's own scratch check is this, not a floor.
- `lobpcg` returns its *last* iterate, not its best. A run that hits `max_iter` can
  return something five orders worse than what it had at iteration 60. `eig_ok` catches
  it (the SCF fails loudly) — but the diagnosis in the log then shows a residual that
  has nothing to do with the solver's actual accuracy, which is probably why the GPU
  failures looked like a floor at 1.4–1.7e-12.
- Two cheap, standard additions belong in this milestone, and they are the ones that
  actually make the tight tolerance safe on an unfamiliar device:
  **(a) a stagnation stop** — leave the loop when $\max_j r_j$ has not improved by, say,
  a factor 0.8 over the last 5 iterations (carry the running best in the `while_loop`
  state; it is one extra scalar); **(b) return the best iterate seen**, not the last
  (one extra block in the carry — $O(n_\text{pad}\,n_\text{solve})$, negligible against
  the FFT workspace).
  Duersch, Shao, Yang & Gu (*SIAM J. Sci. Comput.* **40**(5) C655–C676, 2018,
  `10.1137/17M1129830`) is the reference for both this and the basis-selection
  problem of §2.1; it is worth a line in the post and an entry in `references.bib`.

### 2.3 `EIGEN_FLOOR` is below the attainable residual

Measured at the test size, the smallest max-over-bands residual LOBPCG reaches is
$1.9\!-\!5\times10^{-14}$, against $\lVert H\rVert\epsilon_\text{mach}=2.6\times10^{-15}$.
The gap is the accumulation over the Rayleigh–Ritz, QR and inverse steps, and it means:

- `tol` $= 10^{-12}$ asks for $\tau=5\times10^{-14}$ and is met with a margin of only
  1–2.5× on this CPU. The V100 with the penalty fixed reached $2.9\times10^{-14}$ (Si)
  and $3.7\times10^{-14}$ (H2) — also inside, also barely.
- **Any `tol` below $4\times10^{-13}$ clamps $\tau$ to `EIGEN_FLOOR` = $2\times10^{-14}$,
  which is not attainable**, and the SCF then runs 300 iterations per step and fails
  with a misleading residual (§2.2). Nothing warns the caller.
- Acceptance criterion in the post — "`EIGEN_FLOOR` … The penalty fix should make it the
  floor of the solver as well, on both devices. The tests check that it does" — asserts
  something that is false on the CPU today.

Recommendation: state the attainable floor as a measured multiple of
$\lVert H\rVert\epsilon_\text{mach}$ (at the test size, ~10×) rather than as
$\lVert H\rVert\epsilon$ itself; raise `EIGEN_FLOOR` to the measured value with margin
(e.g. $10^{-13}$) or make the stopping test relative —
$r_j \le \texttt{tol}\,(\lVert H\rVert + |\theta_j|)$, the backward-stable criterion of
Duersch et al. — which is cutoff-independent and is what `RESIDUAL_RTOL` already does
in the 1a tests. Either way, add a guard that refuses (or warns about) an `SCFOptions.tol`
whose implied $\tau$ is below the floor.

### 2.4 Which band is slowest: right outcome, over-general reasoning

The post argues from the Knyazev bound that "the last band of the block is the slowest",
hence stopping on the needed bands helps. Measured per k-point on the $3^3$ grid at the
converged density ($n_\text{solve}=6$, $n_\text{bands}=5$, $\tau=5\times10^{-11}$, cold
start), the picture is sharper and partly different:

| $\lambda_7-\lambda_6$ (Ha) | k-points | slowest band | $I_k$ for 5 bands | $I_k$ for 6 bands |
|:--|:--:|:--:|:--:|:--:|
| $\approx 0$ (degenerate across the edge) | 9 | **band 5** (needed) | 40–49 | 40–49 (no gain) |
| 0.026 | 12 | band 6 (buffer) | 32–35 | 62–72 (**2×**) |
| 0.196 | 6 | band 6 (buffer) | 29–32 | 33–36 (10%) |

So: the buffer band is the slowest at 18 of 27 k-points, the last *needed* band at 9;
over all k, $\max_k I_k$ falls from 72 to 49 and $\overline{I}$ from 51.8 to 36.0.
The gain is entirely controlled by $\lambda_{k+1}-\lambda_k$, exactly as the post's
appeal to the Knyazev bound predicts — but with one inversion worth stating, because it
is the opposite of the natural worry:

**An exact degeneracy at the block edge is harmless; a small non-zero gap is what
hurts.** When $\lambda_6=\lambda_7$ the block converges to *some* basis of the
degenerate eigenspace and the residual goes to zero without obstruction; when the gap
is 0.026 Ha the buffer band must resolve two nearly-equal eigenvalues and takes twice
as long. The post's "Degenerate levels at the edge" paragraph therefore worries about
the benign case. $\Gamma$ is the example in both directions: $n_\text{bands}=5$ cuts
the $\Gamma_{15}$ triplet, and $\Gamma$ is the slowest k-point under *either* criterion
(49 iterations), so it alone sets $\max_k I_k$ after the fix and caps what the fix can
buy at this test size.

Suggested rewrite: keep the Knyazev argument, drop "the last band of the block is the
slowest" in favour of "the slowest band is whichever one sits closest to
$\lambda_{k+1}$, which is usually but not always the buffer band", and replace the
degeneracy paragraph with the distinction above.

### 2.5 Acceptance criterion 5 cannot see the fix it is testing

The cost budget is to be set "on the $2^3$ test grid". Measured there, the stopping fix
moves $\sum\max_k I_k$ from 193 to 175 — 9%, within the noise of a mixer change. On
$3^3$ it moves 370→203 and on $4^3$ 344→226. With only 8 k-points the maximum over k is
close to the mean, which is the whole point of the fix. Set the budget on the $3^3$
grid (the displaced-geometry grid the force tests already use), or assert on
$\overline{I}$ rather than $\sum\max_k I_k$, or both.

### 2.6 Lockstep efficiency, measured

The post says the efficiency "has not been measured"; it is cheap to measure by putting
a `jax.debug.callback` on the iteration count inside the vmapped solver (under `vmap`
the callback fires once per k-point, so the full $I_k$ vector reaches the host without
changing the traced computation). Silicon, `tol` $=10^{-9}$, buffer 1, summed over the
SCF:

| grid | $n_k$ | criterion | $\sum\max_k I_k$ | $\sum\overline{I}$ | efficiency |
|:--:|:--:|:--|:--:|:--:|:--:|
| $2^3$ | 8 | all bands | 193 | 153.6 | 0.796 |
| $2^3$ | 8 | needed | 175 | 148.5 | 0.849 |
| $3^3$ | 27 | all bands | 370 | 236.3 | 0.639 |
| $3^3$ | 27 | needed | 203 | 158.9 | 0.783 |
| $4^3$ | 64 | all bands | 344 | 219.3 | 0.637 |
| $4^3$ | 64 | needed | 226 | 168.1 | 0.744 |

Three things follow, all of them answers the post wants:

- The efficiency does fall with $n_k$ (0.80 → 0.64) and the fix raises it back
  (0.64 → 0.78), confirming the post's framing.
- **The headroom left for chunking is small.** After the stopping fix, perfect per-k
  stopping would take 203 → 159 on $3^3$ (−22%) and 226 → 168 on $4^3$ (−26%). Against
  the 45% the stopping fix itself buys, lockstep is the second-order effect. Chunk size
  should therefore be chosen for memory and kernel occupancy, not for lockstep — which
  settles open question 5 in favour of "one batch until measured" for the SCF.
- The fastest k-point is untouched by the fix (`min_per_step` identical in both runs);
  only the slow tail moves, as the mechanism in §2.4 requires.

Also worth correcting in the post's framing of problem 1: on the **V100** the time per
k-point and SCF step is nearly flat ($n=2$: 6.17 ms, $n=8$: 7.84 ms, with 5.08 ms at
$n=4$) while $\sum\max_k I_k$ triples, i.e. the time *per iteration* per k-point falls
2.4× as the batch fills the device. On the CPU it is flat (1.5–2.0 ms), so there the
k-point cost growth is entirely the iteration count. The GPU gain from the stopping fix
will therefore be smaller than the CPU's 1.5–2.2×, and the post should predict that
rather than being surprised by it.

### 2.7 The adaptive tolerance: what DFTK and QE actually do (open question 1)

Both references the post asks about implement essentially this rule, and both add
something it lacks.

**DFTK** (`AdaptiveDiagtol`, `src/scf/scf_callbacks.jl`): multiplies
$\lVert \rho_\text{next}-\rho_\text{in}\rVert$ by `ratio_ρdiff` = 0.2, clamps into
[`diagtol_min`, `diagtol_max` = 0.005], **takes the minimum over the whole history so
the tolerance never increases**, and uses `min(diagtol_first, 5·diagtol_max)` for the
first two iterations. The docstring recommends *lowering* the ratio for hard cases. A
comment records that size-dependent ratios were tried —
`ratio_ρdiff = 10/N_occ` and `2Ω/(\sqrt{\prod N_\text{fft}}\,N_\text{occ})` — which is
the unit-carrying scaling the post says it cannot derive; worth citing as prior art
rather than re-deriving.

**Quantum ESPRESSO** (`PW/src/electrons.f90`): `IF (iter == 2) ethr = 1.D-2`, then
`ethr = MIN(ethr, 0.1D0*dr2/MAX(1.D0,nelec))`, then `ethr = MAX(ethr, 1.D-13)` with the
comment that below that "iterative diagonalization may become unstable". `dr2` is the
estimated SCF energy error (Ry), so QE's ratio is per electron and relates two
energy-like quantities; `ethr` is on eigenvalue accuracy, not on a residual norm. And
QE already does the post's option (a): when `dr2 < tr2_min` it prints
*"Threshold (ethr) on eigenvalues was too large: Diagonalizing with lowered threshold"*
and repeats the diagonalisation inside the same iteration.

Concrete suggestions:

1. **Make $\tau_i$ non-increasing** — $\tau_i = \min(\tau_{i-1}, c\,r_{i-1})$, as both
   codes do. The post's rule as written lets $\tau$ rise again whenever Anderson
   overshoots and $r$ jumps, which is the configuration most likely to produce the
   repeated refinements of §2.8.
2. **Divide by $N_\text{el}$.** $r$ is an extensive quantity
   ($\lVert\cdot\rVert_{L^2}$ over the cell); the eigensolver residual is per orbital.
   Without a $1/N_\text{el}$ (QE) or $1/N_\text{occ}$ (DFTK's experiment) factor, a
   default $c$ tuned on 8-electron silicon will be wrong by the system size on
   aluminium supercells — which is the next milestone.
3. **$\tau_\text{max}$**: DFTK uses 0.005 Ha and QE 0.01 Ry $\approx$ 0.005 Ha. The
   post's $10^{-2}$ Ha is 2× looser than both; 5×10$^{-3}$ Ha is the safer default and
   costs nothing, since the first step is thrown away anyway.
4. **$c = 0.1$ collides with `EIGEN_CHECK_FRACTION` = 0.1.** The step that achieves
   $r_i < \varepsilon_\text{SCF}$ was computed with $\tau_i = c\,r_{i-1}$; if
   convergence is smooth, $r_{i-1}\gtrsim\varepsilon$ and $\tau_i \approx 0.1\varepsilon$
   — exactly the `eig_ok` threshold, with no margin. DFTK has no such check and so no
   such collision. Either take $c \le$ `EIGEN_TOL_FRACTION` $=0.05$, or (better) keep
   refinement (a) and let it carry the final accuracy, as QE does.
5. The statement that $c$ "carries units and is not universal" is right and worth
   keeping, but the post should say *which* units: with $\lVert\cdot\rVert =
   (\Omega/N\sum r^2)^{1/2}$ in e·Bohr$^{-3/2}$ and residuals in Ha, $c$ is in
   Ha·Bohr$^{3/2}$/e. A reader comparing with DFTK's 0.2 needs to know that DFTK's
   $\Delta\rho$ norm is not the same norm (check whether DFTK's `history_Δρ` carries
   the $\sqrt{\mathrm{d}V}$ factor before quoting the two ratios side by side; for this
   cell the two conventions differ by $\sqrt{\Omega/N}=0.14$).

### 2.8 Refinement cost: it is not the lost search directions (open question 2)

Measured directly — converge a block to a loose $\tau$, then warm-start a tight
refinement from it with $P=0$, exactly as option (a) would — at three k-points of the
test size:

| loose $\tau$ | iterations to reach it | iterations to refine to $5\times10^{-14}$ |
|:--:|:--:|:--:|
| $10^{-5}$ | 15–27 | 19–36 |
| $10^{-8}$ | 22–39 | 13–21 |
| $5\times10^{-11}$ | 28–48 | 7–13 |

(cold start to $5\times10^{-14}$ at the same three k-points, for comparison: 35, 51, 59.)
So a refinement costs about
3–4 iterations per decade of residual, and the two penalty rules give the same count to
±2. Losing $P$ costs roughly one iteration — the first step after a restart searches a
$k$-dimensional instead of a $2k$-dimensional space — so **"LOBPCG should keep its
search directions across calls" does not explain 64 iterations**. Two explanations fit
the measurement: the refinement fired at more than one SCF step (each costing 10–20),
or it fired from a much looser block than $\tau_\text{min}$-minus-a-decade.

Recommendation: keep design (a) — it is QE's, it keeps `fixed_point` generic, and it
guarantees that the converged $F$ is the tight map that implicit differentiation will
need — and add two things: **log how many refinements happened** (a counter in the
history, not just the summed `eig_iters`), and **latch**: once a step has refined, set
$\tau = \tau_\text{min}$ for all later steps. Latching is free, it follows from the
monotonicity of §2.7.1, and it removes the only real failure mode of (a), which is a
loose evaluation that dips below $\varepsilon_\text{SCF}$, triggers a refinement whose
residual is *above* $\varepsilon_\text{SCF}$, and then does it again next step.

### 2.9 Memory: chunking over k does not fix the band scan

The 0.23 GiB per k-point is not a constant — it is $\sim n_\text{grid}\times
n_\text{pad}\times 16$ bytes, both factors growing with the cutoff. Measured for one
k-point of silicon with the dense solver:

| $E_\text{cut}$ (Ha) | $n_\text{pad}$ | FFT grid | temporaries |
|:--:|:--:|:--:|:--:|
| 12 | 528 | $24^3$ | 0.224 GiB |
| 20 | 1174 | $30^3$ | 0.970 GiB |
| 30 | 2085 | $36^3$ | 2.974 GiB |
| 40 | 3263 | $45^3$ | **9.042 GiB** |

So `k_batch_size` bounds the band scan only at the test cutoff: the acceptance criterion
"below 2 GiB" means chunks of at most 8 today, and at 40 Ha — an ordinary production
cutoff for silicon, and the cutoff the 2b convergence table itself goes up to — *one*
k-point exceeds it. The missing fix is in `dense_eigensolver` itself: it applies the
operator to the whole $n_\text{pad}\times n_\text{pad}$ identity in one call, so the
FFT workspace holds $n_\text{pad}$ wavefunctions at once. Applying it to the identity in
column blocks (the same `lax.map(..., batch_size=…)`, one level down) bounds the
temporaries by the *column* chunk, independently of $n_k$ and of the cutoff. That is
three lines and it is the change that generalises. I would do both and state the
criterion as "temporaries per k-point below X, independent of $n_\text{pad}$".

Smaller points on the chunking fix:

- `lax.map` with a `batch_size` that does not divide $n_k$ compiles a *second* program
  for the remainder (`_batch_and_remainder`). Inside the SCF's `while_loop` that is a
  second copy of the whole per-k pipeline per step. Prefer a chunk size dividing $n_k$,
  or round the batch up and mask.
- `per_k(basis_k, x0)` takes two arguments; `lax.map` maps a single pytree, so the
  callers need a wrapper. Worth a sentence, since this is where a reader will reach for
  `vmap`'s signature by analogy.
- The claim that `lax.cond` outside the k-point `vmap` "makes this a real branch" is
  correct and worth keeping — under `vmap` with a batched predicate it would degrade to
  `select` and evaluate both branches. The predicate here (the density residual) is a
  scalar outside the batch, so the branch is real.
- The SCF's own k-memory is not as harmless as the post suggests at scale: the V100 run
  already holds 7.4 MB per k-point (3.78 GB at $8^3$). A $12^3$ grid on the same cell
  would be ~13 GB before anything else.

### 2.10 Band scans: dense versus LOBPCG (open question 4)

The post's arithmetic ("548 operator applications per k-point for the dense matrix,
against a few hundred band-applications for LOBPCG") understates the iterative side.
LOBPCG applies $H$ to $2k$ columns per iteration ($W$ and $P$), so with $k=6$ and the
40–50 iterations a cold start needs at these k-points it is $480$–$600$ band
applications — a wash with the dense solver's 548 at this cutoff, and the dense solver
additionally returns every eigenvalue, needs no stopping criterion, and cannot
mis-converge. The balance tips to LOBPCG only as $n_\text{pad}$ grows (dense is
$O(n_\text{pad})$ applications plus an $O(n_\text{pad}^3)$ `eigh`, LOBPCG is
$O(k\,I)$), i.e. at 30–40 Ha rather than 12.

There is also a trap: `band_structure` calls `solver(apply, dim=…, n_bands=…)` and
passes **no** `x0`, **no** preconditioner and **no** `tol`. Swapping in `lobpcg` today
would silently use a random unmasked start, no preconditioning and `tol=1e-10`, which on
a band path is both slow and unchecked (`band_structure` has no `eig_ok`). So: keep the
dense solver for `band_minimum_on_segment` and add chunking; if a later milestone wants
LOBPCG on band paths, it first needs a preconditioner, a residual check and a
k-sequential warm start — which conflicts with the k-batch and is its own design.

### 2.11 The interlacing bound is right; three caveats to state (open question 3)

$X^\dagger H X = \operatorname{diag}(\varepsilon)$ is a principal submatrix of the kept
part of $T$, so Cauchy interlacing gives $\lambda_j(\text{kept}) \le \varepsilon_j$ and
any $\sigma > \varepsilon_k$ excludes the dropped directions from the lowest $k$. The
proposal $\sigma = \varepsilon_k + (\varepsilon_k-\varepsilon_1) + 1$ Ha is sound, and
empirically it is far *below* $\max|T|$ (1.82 Ha against 5.5 Ha at the test k-point)
while returning eigenvalues that match the dense solver to $2.9\times10^{-14}$. On the
post's specific worry — that $\varepsilon$ comes from the previous iteration's `eigh`:

- They are not stale. The loop recomputes `eps_new` as the eigenvalues of
  $X_\text{new}^\dagger A X_\text{new}$ at the *end* of each body, after the QR
  re-orthonormalisation, so at the top of the next iteration $\varepsilon$ is exactly
  the current block's Ritz values.
- Their rounding is $O(\epsilon_\text{mach}\lVert T_2\rVert)\approx10^{-14}$ Ha, and the
  interlacing needs only $\sigma>\varepsilon_k$ — the +1 Ha margin covers it by fourteen
  orders.
- The caveat that *is* worth a sentence is a different one: interlacing applies to the
  Rayleigh quotient on an orthonormal basis, and the code solves a standard (not
  generalised) eigenproblem for $T$, i.e. assumes $S^\dagger S = I$. The
  non-orthonormality of $[X,Q]$ enters as $O(\lVert H\rVert\,\delta_\text{orth})\approx
  10^{-12}$ Ha. Also covered by the margin, but it is the assumption a reader will
  question, not the Ritz values.
- Finally: "+1 Ha" is dimensionful. Fine under the atomic-units invariant, but say so,
  and note it degenerates gracefully at $k=1$ ($\sigma=\varepsilon_1+1$).

### 2.12 Scope (open question 6)

No. Across the eight configurations of Table 1 the converged energy varies by
$<3\times10^{-14}$ Ha, and the SCF step counts are identical. Structurally: `n_converge`
only loosens the *buffer* columns, which feed nothing but the next warm start —
`grid_occupations`, `frontier_gap` and the returned orbitals all slice `[:, :n_bands]`.
The one invariant worth stating explicitly in the post is that `n_bands` (the needed
set) already includes the first conduction band, so the gap guard and both one-sided
weight derivatives of 2a keep seeing converged states.

---

## 3. Smaller comments

1. "On the CPU and the V100 alike, the time per k-point and SCF step grows" — true on
   the CPU (5.6× from 1 to 512 k-points), only 1.27× and non-monotone on the V100 (§2.6).
2. "It falls as $n_k$ grows, because the maximum of more samples is larger" — measured
   0.80 → 0.64 → 0.64, so it saturates rather than keeps falling; the cluster scan's
   $3^3$ point (347 iterations, above $4^3$'s 343) shows the same saturation plus noise.
   Worth one hedging clause.
3. The post says LOBPCG converging the buffer bands is "mostly a bug, not a property of
   lockstep". Measured: of the 370 → 159 total reduction on $3^3$, 167 comes from the
   stopping set and 44 from the buffer width. "Mostly" is fair.
4. `EIGEN_CHECK_FRACTION` (0.1) and `EIGEN_TOL_FRACTION` (0.05) never appear in the post
   although the whole adaptive-tolerance section depends on their ratio. One sentence.
5. The energy-gradient path (`energy`, used by the force tests) runs `_scf_core` with
   `history=False`, so none of the new observability reaches a force run. If the
   adaptive tolerance is to be diagnosed in the derivative tests, that needs saying.
6. The claim "A poor start column costs iterations; it cannot cost correctness" is right
   for the *returned* bands but assumes the solver converges at all; with §2.2 in mind,
   "cannot cost correctness, provided the run does not hit `max_iter`" is the honest form.
7. Observability: `eig_iters_mean` needs the per-k vector, which `diagonalise` already
   has (`its` is returned from the vmapped `per_k`) — `jnp.mean(its)` next to
   `jnp.max(its)` is a one-line change, and the full histogram costs nothing more if the
   history can hold an $n_k$-vector per step.
8. "the same 9.45 GiB of temporaries on the laptop" — worth adding that this is
   `memory_analysis` of the *compiled* program, i.e. XLA's own estimate of peak
   temporaries, which is why it can be quoted without a GPU.
9. Table 2's row $\varepsilon_\text{SCF}=10^{-12}$, $c=0.01$ (229 iterations, worse than
   $c=0.1$'s 89) is the one result in the post that the stated mechanism does not
   explain — a tighter eigensolver should not cost 2.5× more iterations unless something
   else is happening. It is flagged as the §2.2 blow-up, but since it is also the
   regression case of acceptance test 4, the post should say what the expected iteration
   count is after the fix, not only that it must not hit the cap.
