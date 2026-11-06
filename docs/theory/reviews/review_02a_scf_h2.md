# Review: `docs/theory/02a_scf_h2.qmd` — Milestone 2a, self-consistent LDA for H₂ in a box

Reviewed against an independent NumPy/SciPy re-implementation written from the post
alone (plane-wave basis, FFT grid, GTH local pseudopotential, Hartree, Slater+PW92,
Ewald, Anderson-mixed SCF, dense `eigh` and LOBPCG, Hellmann–Feynman forces,
one-sided weight derivatives). Every formula and every quoted number in the post was
recomputed. Unless stated otherwise, agreement figures below are from that
re-implementation.

**Bottom line.** I found no physics error. All eleven analytic identities check, and
all four numerical tables in the post reproduce to the digits printed. The findings
below are about *tests that will not do what the post expects them to do*, about
quantities the post leaves as adjectives where a number is available, and about two
engineering traps. Findings 1, 2 and 3 should change the acceptance-test design.

---

## 1. Verification table

| # | Claim in the post | Result |
|---|---|---|
| 1 | GTH $\hat v(q)$ closed form, $y=(qr_\text{loc})^2$ | ✓ vs. numerical radial transform, rel. 7e-14 … 1e-12 at $q=0.3,1,3,7$, random $C_1\dots C_4$ |
| 2 | $\alpha = 2\pi Zr_\text{loc}^2+(2\pi)^{3/2}r_\text{loc}^3(C_1+3C_2+15C_3+105C_4)$ | ✓ equals $\int(v+Z/r)4\pi r^2dr$ to 1e-16; $\alpha(\mathrm H)=-0.001\,297\,886$ Ha·Bohr³ |
| 3 | HGH H: $Z=1$, $r_\text{loc}=0.2$, $C_1=-4.18023680$, $C_2=0.72507482$, no projectors | ✓ byte-identical to `PseudoLibrary/pseudos/hgh_lda_hgh/h-q1.hgh` and to CP2K `GTH_POTENTIALS` `GTH-PADE-q1`; ABINIT's HGH file (header: "from PRB58, 3641 (1998)") has the same to 6 dp, so the `@hgh1998` citation is right |
| 4 | Grid rule $N_i\ge2\lfloor 2G_\text{max}\lvert a_i\rvert/2\pi\rfloor+1$, then 2/3/5-smooth | ✓ $L{=}10,E_\text{cut}{=}10$: $N_\text{min}=29\to N=30$, $N_\text{PW}=1503$ ("about 1500, a 30³ grid") |
| 5 | Density from FFT is exact, no aliasing | ✓ FFT vs. direct coefficient convolution, 3e-20 |
| 6 | FFT local operator, built densely, equals $\tilde V(\mathbf G-\mathbf G')$ and is Hermitian | ✓ 1.6e-15 and 1.1e-17 |
| 7 | FFT normalisation set (round trip, $\Omega N_\text{tot}^{-1}\sum\lvert\psi\rvert^2=1$) | ✓ 4e-17, 1−9e-16 |
| 8 | $\mathbf G=0$ bookkeeping: dropped divergences $\propto(N_\text{el}-Q)^2$ | ✓ re-derived term by term: $+2\pi N_\text{el}^2$, $-4\pi N_\text{el}Q$, $+2\pi Q^2$, all over $\Omega G^2$ |
| 9 | Ewald formula incl. background term | ✓ η-independent to 1e-14 for the H₂ cell ($E_\text{Ewald}=0.126\,737\,138\,563$ Ha at $L{=}10$, $d{=}1.45$) |
| 10 | Madelung sc −0.8800594421, bcc −0.8959292557, fcc −0.8958736152, NaCl −1.7475645946 | ✓ all four, max deviation 3.3e-11, at two η |
| 11 | Gaussian Hartree identity $E_\text{H}=E_\text{Ewald}+Z^2/2\sqrt\pi\sigma+2\pi Z^2\sigma^2/\Omega$ | ✓ 2.2e-16 at $L{=}8,\sigma{=}0.6$ (post: 1e-15) |
| 12 | Slater $\varepsilon_\text{x}=-0.4581652933/r_s$ | ✓ $-0.458\,165\,293\,283\,14$ |
| 13 | PW92 unpolarised form and parameters | ✓ standard `lda_c_pw`; DFTK's `LDA()` does expand to `[:lda_x, :lda_c_pw]` |
| 14 | Jellium rung: $E_\text{H}=0$, $E_\text{xc}=N_\text{el}\varepsilon_\text{xc}(r_s)$ | ✓ 3.7e-36 and 1e-12 through the full density→energy path |
| 15 | Direct energy = double-counting energy at convergence | ✓ 5.1e-13 Ha |
| 16 | Iterative and dense eigensolvers give the same SCF result | ✓ ΔE = 7e-16 Ha, Δε = 2e-13 Ha |
| 17 | $E$ at $d{=}1.44$, $L{=}12$: −1.1030/−1.1266/−1.1330/−1.1353 | ✓ −1.102958 / −1.126626 / −1.132952 / −1.135284 |
| 18 | $E_\text{cut}$ error at $L{=}10$: 33 / 9 / 3.5 / 1.2 mHa above 50 Ha | ✓ 33.27 / 9.18 / 3.45 / 1.16 |
| 19 | Box table $E-E(14)$: −12.6 / −2.2 / 0.03 / 0 mHa | ✓ −12.608 / −2.207 / +0.034 / 0.000 |
| 20 | $r_e$: 1.510 / 1.475 / 1.454 / 1.453 Bohr | ✓ 1.5104 / 1.4749 / 1.4537 / 1.4526 (5-point quartic fit) |
| 21 | $\omega_e$ "came out at 4100–4200 cm⁻¹" | ✓ 4171 / 4155 / 4109 cm⁻¹ at 20/30/40 Ha (3728 at 10 Ha) |
| 22 | HOMO–LUMO gap "about 0.35 Ha" at $L{=}10$, $E_\text{cut}{=}10$, $d{=}1.45$ | ✓ 0.347192 Ha |
| 23 | SCF converges in 8 Anderson steps to 1e-10 | ≈ 9–11 steps here (mixing-parameter dependent); 3e-13 reached at step 11 |
| 24 | Forces: Hellmann–Feynman = FD of the converged discrete energy | ✓ error 7.1e-5 → 7.1e-7 for $h=10^{-2}\to10^{-3}$, clean $h^2$; net force 1.1e-16 |
| 25 | $\partial E/\partial\lambda_j\rvert_\pm$ = explicit part + $Z_j\varepsilon_\pm$ | ✓ formula correct (verified at $\lambda=1.01$ by a central difference converging as $\delta^2$); difference of the two sides = $Z\times$gap to 1e-9. **But see finding 2** |
| 26 | $L{\approx}12$, $E_\text{cut}{\ge}30$ needs ~1.3e4 PW, 2.7 GB dense | ✓ 13 517 PW, 2.72 GiB |
| 27 | "about 1 s per SCF in NumPy" at the test size | ✓ 0.39 s for a full 11-iteration SCF with LOBPCG |
| 28 | Quadrupole–quadrupole $O(L^{-5})$, small boxes lower | ✓ consistent with the box table; H₂ is neutral and dipole-free |

---

## 2. Findings

### 2.1 The egg-box effect is absent at these settings, and the tests built on it will measure noise

**Severity: changes the test design.**

The post devotes a paragraph of theory, two invariants and a validation-document
deliverable to the egg-box, and blames it for the crude frequency fit. Measured
peak-to-peak energy variation under rigid translation of the molecule along $x$
(8 shifts over one grid spacing, SCF to 1e-10):

| $L$, $E_\text{cut}$ | $N$ | egg-box $E$ p-p (Ha) | max $\lvert\sum_j\mathbf F_j\rvert$ (Ha/Bohr) |
|---|---|---|---|
| 10, 10 | 30 | 2.1e-11 | 2.5e-11 |
| 10, 20 | 45 | 2.8e-13 | 5.1e-13 |
| 12, 30 | 60 | 2.3e-13 | 4.7e-13 |
| 10, 40 | 60 | 4.0e-15 | 2.0e-14 |

A shift along the cube diagonal gives 1.1e-10 Ha. These are at or below
$\tau_\text{SCF}$-induced noise. The reason is in the post's own argument: with the
no-wraparound grid the kinetic, Hartree, local and Ewald terms are *exactly*
translation-covariant, so XC quadrature is the only source — and for a smooth
pseudo-density at supersampling 2 it is negligible.

What *is* dramatic is the behaviour at the threshold ($L{=}10$, $E_\text{cut}{=}10$,
rule minimum $N_\text{min}=29$):

| $N$ | 20 | 24 | 29 | 30 | 36 | 45 |
|---|---|---|---|---|---|---|
| egg-box p-p (Ha) | 1.09e-4 | 8.40e-6 | 5.86e-11 | 2.07e-11 | 1.78e-12 | 4.25e-12 |
| $E$ (Ha) | −1.104864236 | −1.104881062 | −1.104880727 | −1.104880728 | −1.104880728 | −1.104880728 |

Seven orders of magnitude across the no-wraparound boundary. Recommendations:

- Recast the egg-box test as a **test of the grid rule**, with an undersized grid as
  the positive control. As written ("reports the size of the egg-box variation and
  how it decreases with the grid") it will report a monotonically shrinking sequence
  of numbers that are all SCF noise.
- The invariant "the net force vanishes for the centred molecule by symmetry" is
  satisfied to 1.1e-16 *by construction* of the structure factor and tests nothing.
  Replace it with the net force for an off-grid, non-symmetric placement, where it is
  a real (and here, tiny) quantity.
- Drop the claim that "four energy points on a cubic is egg-box sensitive". The crude
  frequency came from the fit, not the grid: a 5-point quartic over
  $d\in[1.38,1.54]$ on the same discrete energy gives 4155 cm⁻¹ at $E_\text{cut}=30$
  with no grid contamination.
- The "variational monotonicity needs a fixed grid" precaution is correct in
  principle and worth keeping, but the measured grid sensitivity above threshold is
  below 1e-9 Ha ($E$ row above), so the default rule gives a monotone sequence anyway
  (table 17 is monotone). Say so, otherwise the reader over-weights the caveat.

### 2.2 The proposed weight-derivative test fails on the LUMO side

**Severity: the test as specified will not pass.**

The post specifies "a second-order one-sided stencil with $\delta=10^{-3}$, at
$10^{-5}$ Ha". At $L{=}10$, $E_\text{cut}{=}10$, $d{=}1.45$, with separate SCF runs at
$\lambda_1 = 1\mp\delta,\ 1\mp2\delta$ and $\tau_\text{SCF}=10^{-12}$:

| $\delta$ | left (HOMO) side error | right (LUMO) side error |
|---|---|---|
| 2e-3 | −1.16e-6 | +1.05e-4 |
| 1e-3 | −2.90e-7 | +3.80e-5 |
| 5e-4 | −7.26e-8 | +1.28e-5 |
| 2.5e-4 | −1.82e-8 | +4.13e-6 |

The left side is clean second order and passes with a factor of 34 to spare. The
right side misses the stated tolerance by 4×, and its observed order is ≈1.6, not 2.

The **formula is right** — this is not a physics error. Verified independently at
$\lambda_1=1.01$, where everything is smooth: a central difference agrees with
$\partial E_\text{direct}/\partial\lambda_1 + Z\varepsilon_\text{LUMO}$ to
9.3e-6 / 2.3e-6 / 5.8e-7 at $\delta = 2$e-3 / 1e-3 / 5e-4, i.e. exactly $\delta^2$.
And the two one-sided derivatives differ by $0.347\,192\,47 = Z(\varepsilon_\text{LUMO}-\varepsilon_\text{HOMO})$
to 1e-9.

The cause is that $E(N_\text{el})$ is strongly and *non-analytically* curved just
above $N_\text{el}=2$: $\varepsilon_\text{LUMO}$ moves at $-0.80$ Ha per unit
$\lambda$ at $\delta\to0^+$ but only $-0.57$ by $\delta=5\times10^{-3}$, so the
third derivative of $E$ is ~140 Ha at the integer point and growing as
$\delta\to0$. (Not a degeneracy problem: $\varepsilon_\text{LUMO+1}-\varepsilon_\text{LUMO}=0.070$ Ha,
so the post's gap guard would not fire.) Recommendations: quote the two sides'
tolerances separately; use $\delta=2.5\times10^{-4}$ for the LUMO side (4e-6) or a
Richardson-extrapolated four-point stencil; and state in the post that the FD check
of the right derivative is intrinsically the harder of the two.

### 2.3 $\varepsilon_\text{LUMO}$ is a box state, so $\partial E/\partial\lambda\rvert_+$ is not converged in $L$

**Severity: physics the post should state.**

At $E_\text{cut}{=}20$, $d{=}1.45$:

| $L$ (Bohr) | 8 | 10 | 12 | 14 |
|---|---|---|---|---|
| $\varepsilon_\text{HOMO}$ (Ha) | −0.37450 | −0.36599 | −0.36578 | −0.36704 |
| $\varepsilon_\text{LUMO}$ (Ha) | −0.00336 | −0.01390 | −0.01228 | −0.00950 |
| gap (Ha) | 0.3711 | 0.3521 | 0.3535 | 0.3575 |

The total energy is converged to 0.03 mHa by $L=12$, and the HOMO to ~1 mHa, but the
LUMO wanders over 10 mHa non-monotonically. H₂'s $\sigma_u^*$ is unbound in free
space; in the box it is a particle-in-a-box state whose energy is set by $L$, not by
the molecule. Consequences worth one paragraph in the post:

- The right weight derivative $\partial E/\partial\lambda_j\rvert_+ = \text{explicit} + Z_j\varepsilon_\text{LUMO}$
  is a box artefact at the 10 mHa level, while the left one is physical. The
  structural check (difference of sides = $Z\times$gap) is unaffected, because both
  sides inherit the same $\varepsilon_\text{LUMO}$.
- The validation document promises "the bond length's dependence on $L$"; add
  $\varepsilon_\text{LUMO}$ and the gap, which are far more $L$-sensitive.
- The gap guard ("raises if the gap is below a threshold") is testing a
  box-dependent quantity. Fine for H₂ at 0.35 Ha, but say that it is not a
  physical-gap criterion.

### 2.4 $\alpha(\mathrm H)$ is anomalously small, so the $\mathbf G=0$ convention is weakly tested

**Severity: tolerance claim is overstated.**

$\alpha(\mathrm H) = 2\pi Zr_\text{loc}^2 + (2\pi)^{3/2}r_\text{loc}^3(C_1+3C_2)
= +0.251\,327 - 0.252\,625 = -0.001\,297\,886$ Ha·Bohr³ — a 0.5 % residue of a
cancellation. At $L=10$ that gives

- eigenvalue shift $\sum_j\lambda_j\alpha_j/\Omega = -2.60\times10^{-6}$ Ha,
- `PspCorrection` energy $N_\text{el}\sum_j\lambda_j\alpha_j/\Omega = -5.19\times10^{-6}$ Ha,

against the post's proposed tolerances of 1e-6 Ha (eigenvalues) and 1e-7 Ha (per
energy term): factors of 2.6 and 52. The post's "a wrong $\mathbf G=0$ convention …
fails by orders of magnitude" is true of the Hartree and Ewald background terms but
*not* of the $\alpha$ term for hydrogen. For comparison, HGH silicon has
$\alpha = -4.98$ Ha·Bohr³, 3800× larger. Recommendations: put the expected
−2.60e-6 Ha and −5.19e-6 Ha in the post so the DFTK comparison asserts a value rather
than an agreement; and note that the near-cancellation makes $\alpha$ an unusually
sharp provenance test of the last digits of $C_1$ and $C_2$ in the parameter file.

### 2.5 DFTK's `PspCorrection` is energy-only, so the eigenvalue offset is real (open question 3)

DFTK's `TermPspCorrection` contributes an energy and a no-op operator; it is not in
the Hamiltonian. So if 2a includes $\alpha$ in $\tilde V_\text{loc}(\mathbf 0)$ — which
it should, for Janak — DFTK's eigenvalues will be *higher* by $2.60\times10^{-6}$ Ha
at the test size. Because that is only 2.6× the proposed eigenvalue tolerance, a
comparison script that forgets the shift will come within a factor of three of
passing. Make the shift an explicit, asserted constant in the comparison, and confirm
it by reading DFTK's reported `PspCorrection` and checking it equals
$N_\text{el}\sum_j\lambda_j\alpha_j/\Omega$. (I did not run DFTK; this is from the
term's definition, and should be confirmed when the reference data is generated.)

### 2.6 Two things to pin in the DFTK run besides `fft_size`

- **Symmetries.** DFTK detects the structure's point group and symmetrises the
  density. For the centred, symmetric H₂ this is harmless; for the distorted-bond and
  off-centre configurations the two codes would be solving slightly different
  discrete problems. Pass `symmetries=false`.
- **Version pinning.** Record the DFTK *and* libxc versions, and the pseudopotential
  by family + file hash, not by name. The `lda_c_pw` vs `lda_c_pw_mod` issue the post
  raises is a libxc-version-sensitive detail.

### 2.7 Nyquist planes: the analytic $\tilde V_\text{loc}$ is not Hermitian there

**Severity: engineering trap, no physics consequence.**

With even $N$, `fftfreq` holds $m=-N/2$ but not $+N/2$, so the Nyquist planes have no
conjugate partner. Evaluating $\hat v$ analytically on them and inverse-transforming
gives a **complex** $V_\text{loc}(\mathbf r)$ whenever the structure is not
inversion-symmetric about a grid point. Measured at $L{=}10$, $E_\text{cut}{=}10$,
$N{=}30$:

- centred, symmetric H₂: $\max\lvert\operatorname{Im}V\rvert = 5.6\times10^{-16}$ (the structure factor is real by symmetry);
- **atom 1 displaced by $10^{-3}$ Bohr** (exactly the finite-difference force geometry):
  $\max\lvert\operatorname{Im}V\rvert = 3.9\times10^{-4}$ Ha, against $\max\lvert\operatorname{Re}V\rvert=6.9$ Ha.

It is harmless to the physics — $\tilde\rho$ vanishes there, and the basis has
$\max\lvert m_i\rvert=7$, so the Hamiltonian only needs $\lvert m_i\rvert\le14$ while
the Nyquist index is 15 — but `apply_H` becomes non-Hermitian unless the Nyquist
planes are zeroed (or the real part taken). Make "zero the Nyquist planes of any
analytically-evaluated form factor" an explicit rule in the post, and run the
Hermiticity assertion on a **displaced** geometry, where it can actually fail.

### 2.8 Claim 2 deserves the sharp statement, not just "no difference wraps"

Two steps are implicit in the exactness argument and are worth a sentence each,
because they are exactly what a reader would get wrong:

1. Truncating the analytic $\tilde V_\text{loc}$ to the grid is harmless: every basis
   pair has $\lvert\mathbf G-\mathbf G'\rvert\le2G_\text{max}$, which is inside the
   grid, and $\tilde\rho$ vanishes outside the $2G_\text{max}$ sphere, so neither
   $H$ nor $E_\text{loc}$ sees the discarded components. (Verified: the reciprocal
   and real-space forms of $E_\text{loc}$ agree to 1e-16.)
2. The restriction of $V\psi$ to the basis is unaliased because
   $N_i\ge2\lfloor2G_\text{max}\lvert a_i\rvert/2\pi\rfloor+1 \ge 4\max_i\lvert m_i^\psi\rvert+1$,
   using $\lfloor2x\rfloor\ge2\lfloor x\rfloor$. *This* is why supersampling 2 is the
   right factor and not merely a convention — a product of a grid-bandlimited
   potential with a basis function reaches $3\max\lvert m^\psi\rvert$, and the
   inequality is satisfied with exactly one index to spare (30 > 28 at the test
   size). Stating only "no difference $\mathbf G-\mathbf G'$ wraps" proves the
   density claim, not the operator claim.

### 2.9 Clipping negative densities must be followed by renormalisation

The post says "negative grid densities from mixing are clipped to zero before XC".
Clipping changes $\int\rho$, and the whole $\mathbf G=0$ cancellation argument assumes
$N_\text{el}=\sum_j\lambda_jZ_j$ exactly; a non-neutral iterate reinstates the
divergence that the three dropped terms were cancelling. In my implementation the SCF
only behaved once the clipped density was rescaled back to $N_\text{el}$. The post
already wants "$\int\rho=N_\text{el}$ checkable at every iterate" (open question 4) —
it is only checkable if it is enforced. Make clip-then-renormalise the stated rule.

### 2.10 Tie the eigensolver tolerance to $\tau_\text{SCF}$

$\tau_\text{SCF}=10^{-9}$ on the $L^2$ density residual can only be reached if the
eigenvectors are converged well below it; an iterative solver with a fixed tolerance
will silently floor the SCF residual. In my runs SciPy's LOBPCG at `tol=1e-11`
reached density residuals of 3e-13 at the acceptance size, but at
$L{=}12$, $E_\text{cut}{=}30$ it exited early with eigenpair accuracies ~6e-12 and
still only just made it. Two additions: assert that the final eigenpair residuals
$\lVert H\psi-\varepsilon\psi\rVert$ are below a fixed fraction of $\tau_\text{SCF}$,
and extend "unconverged results are never returned silently" to cover the
eigensolver, not only the SCF loop.

### 2.11 Quantify the functional and parameterisation differences

The post calls these "small" and "not obviously below the tolerance". Measured:

- **Teter Padé vs. Slater+PW92.** Pointwise $\varepsilon_\text{xc}$ differs by up to
  4.3e-4 Ha over $r_s\in[0.5,20]$. End to end at $L{=}12$, $E_\text{cut}{=}30$:
  $\Delta E = +0.49$ mHa, $\Delta r_e = +0.0004$ Bohr (0.0002 Å),
  $\Delta\omega_e = -3$ cm⁻¹. So the post's claim holds for geometry and frequency,
  but the **total energy moves by 0.5 mHa — the same order as the 1.2 mHa basis-set
  error at 40 Ha**. Worth stating so nobody compares total energies across the two.
  It is also worth saying *why* the mismatch is this small: the Teter Padé
  (ABINIT `ixc=1`) is itself a fit to PW92, and its $r_s\to0$ exchange coefficient is
  literally the 0.4581652933 quoted in the post.
- **`lda_c_pw` vs `lda_c_pw_mod`.** Evaluated on the converged H₂ density at the test
  size, the correlation energy shifts by $+2.1\times10^{-7}$ Ha — 2× the proposed
  1e-7 per-term DFTK tolerance. The post's instinct was right; the number makes the
  test's purpose concrete.

### 2.12 Smaller points

- **FD force tolerance has a factor of ~6, not more.** Measured $E''' \approx 1.06$
  Ha/Bohr³ along the bond at $d{=}1.45$, so a central difference with a single-atom
  step of $10^{-3}$ Bohr has a truncation error of $1.8\times10^{-7}$ Ha/Bohr. The
  post's $\sim10^{-7}$ estimate is right to within a factor of two and the $10^{-6}$
  tolerance holds — but $E'''$ grows at the distorted bond length that the post also
  wants tested. Quote the measured $E'''$ so the tolerance has a derivation.
- **Vacuum density numbers.** I reproduce the $L{=}14$ figure (5.8e-14 vs. 5e-14) but
  get 4.1e-10 at $L{=}10$ (post: 2e-8), and the stated maximum of 0.23 does not match
  any cutoff I ran (0.254 / 0.273 / 0.323 at $E_\text{cut}=10/20/40$). The grid
  maximum is a *sampling* artefact — the true peak is at a nucleus, which is not a
  grid point — so it is a fragile number to quote. Either drop the maximum or say it
  is grid-dependent. The argument about libxc's density thresholds at large $L$ is
  unaffected.
- **Initial density has no width.** "A sum of normalised Gaussians" needs a $\sigma$
  (I used 0.8 Bohr). With $r_\text{loc}=0.2$ the pseudo-density is sharp and a poor
  width costs iterations; since the post also wants "independence of the initial
  density" tested, the default should be written down.
- **LDA at stretched bonds.** If the validation document fits a Morse curve over a
  wide $d$ range, say that spin-unpolarised LDA is qualitatively wrong beyond
  ~2.5 Bohr. Not an issue for a harmonic fit near $r_e$.
- **Variational-in-orbitals test.** Correct, but it holds because the aufbau,
  gapped SCF solution is a minimum of $E[\{\psi\}]$ on the orthonormality manifold,
  not merely a stationary point. One clause would make the test's premise explicit.
- **"$E_\text{xc}$ is a quadrature" is the only inexactness** — worth saying so
  outright, since it makes finding 2.1 obvious in advance: every other term in the
  discrete functional is exactly translation-covariant.

---

## 3. Answers to the open questions

**1. Iterative eigensolver in 2a — yes, keep it in 2a.** The cost argument is weaker
than the post implies (a full 11-iteration SCF at the acceptance size takes 0.39 s
with LOBPCG and 9 s with dense `eigh`; at $L{=}12$, $E_\text{cut}{=}30$, 13 517 PW,
LOBPCG took 3.8 s where dense is not feasible), but the benefit is large and the
agreement is exact: LOBPCG matched dense `eigh` to 7e-16 Ha in the total energy and
2e-13 Ha in the eigenvalues. Three concrete points for the post:

- Writing your own is the right call. `lobpcg_standard` targets the *largest*
  eigenvalues; the usual workaround, applying it to $\sigma I - H$, destroys the
  relative gap (here $\sigma\approx2E_\text{cut}=20$ Ha against a 0.35 Ha gap) and
  has no place for the kinetic preconditioner.
- **Because forces use `stop_gradient` on the orbitals, the eigensolver never needs
  to be differentiable.** That removes the main reason one would want a JAX-native
  routine and lets you choose purely on numerics. Worth stating in the post — it is a
  non-obvious consequence of the design.
- The fragile part of LOBPCG is conditioning of the $[X, R, P]$ block, not the
  algorithm: SciPy's implementation emitted "not reaching the requested tolerance"
  warnings at the larger size even with warm starts. Plan for explicit
  re-orthonormalisation with a dropped-direction fallback, warm-start from the
  previous SCF step (it is what makes the cost low), and the tolerance coupling of
  finding 2.10. This interacts with the 1a padding finding: keep the sentinel on
  masked rows a small multiple of $E_\text{cut}$.

**2. Functional — PW92, as proposed.** The numbers are in finding 2.11: the
Padé/PW92 mismatch costs 0.0002 Å in $r_e$ and 3 cm⁻¹ in $\omega_e$, both far below
the pseudopotential error, but 0.49 mHa in the total energy. Reference-code agreement
is the binding constraint and it points to PW92. Add the two sentences quantifying the
mismatch, and the observation that the Padé is a fit to PW92 in the first place.

**3. $\alpha$ in the Hamiltonian — include it, as proposed.** Janak then holds
exactly: with $\alpha$ in $\tilde V_\text{loc}(\mathbf 0)$ my left weight derivative
matched the FD to 2.9e-7 with $\delta^2$ scaling. DFTK will differ by exactly
$\sum_j\lambda_j\alpha_j/\Omega = -2.60\times10^{-6}$ Ha at the test size (finding
2.5). Assert that number rather than "any difference is a documented constant".

**4. Mixing variable — density, as proposed, with one caveat.** Density keeps $F$
textbook and $\int\rho=N_\text{el}$ checkable. The caveat is finding 2.9: density
mixing is the variant that needs clipping, and clipping is what makes $F$
non-smooth and breaks neutrality mid-iteration. Potential mixing has no such step.
Since the post wants the mixed variable to live in the solver rather than in $F$, make
sure the interface lets 2b/2c swap it without touching $F$ — that is the real
decision, and it defers the choice cheaply.

**5. Julia and DFTK — proceed, with finding 2.6's pins.** `symmetries=false`,
explicit `fft_size`, pinned DFTK and libxc versions, pseudopotential by family and
hash. Read DFTK's `PspCorrection` out of its energy breakdown and assert it equals
$N_\text{el}\sum_j\lambda_j\alpha_j/\Omega$: that pins the convention by measurement
instead of by reading DFTK's source.

**6. All-electron LDA reference — prefer a fully numerical diatomic solver over
CCCBDB.** NIST CCCBDB's "LSDA" is Slater+VWN5, not Slater+PW92, so it introduces
exactly the functional ambiguity you are trying to avoid; if you cite it, say which
correlation functional it used. The cleaner option is HelFEM, a finite-element
diatomic solver with a libxc interface, which can be run with `lda_x`+`lda_c_pw`
itself — basis-set-free and functional-matched, so the residual is purely the
pseudopotential error. Cite S. Lehtola, *Int. J. Quantum Chem.* **119**, e25944
(2019), doi:10.1002/qua.25944 (and e25945 for the atomic companion). My
pseudopotential result is $r_e = 1.4526$ Bohr = 0.7686 Å, consistent with the post's
"within about 0.005 Å of 0.765 Å" expectation.

**7. Test sizes — $L=10$ Bohr, $E_\text{cut}=10$ Ha is right.** 1503 plane waves, a
30³ grid, 0.39 s per SCF. Three additions to the acceptance set, each motivated
above and each cheap:

- a configuration that is **not** inversion-symmetric about a grid point (one atom
  displaced), since that is the only one that exercises the Nyquist/Hermiticity path
  (2.7) and gives a non-trivial net force (2.1);
- an **undersized grid** ($N=24$ at this cutoff) as the positive control for the grid
  rule — it moves the energy by 3.3e-7 Ha and the egg-box by five orders of magnitude;
- a run with $\lambda\ne1$ on one atom, so the weight path is covered by an
  acceptance test and not only by the derivative check.
