# Review: `docs/theory/02b_scf_silicon.qmd` — Milestone 2b, self-consistent LDA silicon

**Bottom line.** The physics is sound and every number in the post reproduces from an
independent NumPy plane-wave LDA built from the post alone — including all nine band
energies, both convergence tables, all six energy terms, the force derivative to nine
digits, and the lattice constant and bulk modulus. One claim is wrong: the FFT-grid
condition for the diamond space group is **N even**, not $4\mid N$, and the post's only
negative control ($N=25$) is odd, so its table cannot tell the two apart. Two further
claims are right but argued in a way that will read as wrong to anyone who knows
standard practice (shifted grids) or that the measurement does not support (buffer
bands). Everything else is wording, hidden detail, or an engineering choice.

Method: an independent plane-wave LDA (FFT grid by the 2a rule, GTH local part,
separable HGH projectors built from the post's own table, FFT-Poisson Hartree,
Slater+PW92 with a complex-step potential, point-charge Ewald, Pulay-mixed density SCF,
dense `eigh` cross-checked against LOBPCG, analytic Hellmann–Feynman forces in
fractional coordinates). Nothing was taken from the jaxdft sources except the parsed
pseudopotential parameters, which were re-fetched from PseudoLibrary.

---

## 1. Verification table

Every quantity the post states, checked independently. "Agreement" is the deviation
from the value printed in the post.

| # | Claim in the post | Check | Agreement |
|---:|:--|:--|:--|
| 1 | $\int_0^\infty p_i^l(r)^2r^2dr = 1$, all eight tabulated $(l,i)$ | Gauss quadrature | $4\times10^{-16}$ |
| 2 | The eight-row $\hat p_i^l(q)$ table, $q=0.3,1,3,7$ Bohr⁻¹ | quadrature of $4\pi\int p\,j_l\,r^2dr$ | worst rel. $2.7\times10^{-15}$ (post: $5\times10^{-14}$) |
| 3 | $\alpha(\mathrm{Si}) = -4.976\,525$ Ha·Bohr³ | $2\pi Zr_\text{loc}^2+(2\pi)^{3/2}r_\text{loc}^3C_1$ | $-4.976\,525\,423$ ✓ |
| 4 | $\sum_j\alpha_j/\Omega = -0.036\,829$ Ha at $a=10.263$ | $\Omega = 270.248\,315$ Bohr³ | $-0.036\,829\,280$ ✓ |
| 5 | $N_\text{el}\sum_j\alpha_j/\Omega = -0.294\,634$ Ha (DFTK `PspCorrection`) | same | $-0.294\,634\,239$ ✓ |
| 6 | $h^s_{12} = -\tfrac12\sqrt{3/5}\,h^s_{22}$ reproduces the stored value | arithmetic | $-1.261\,893\,970$ vs stored $-1.26189397$ ✓ |
| 7 | Si parameters as distributed in PseudoLibrary | file re-fetched; SHA-256 `9d7cca22…20d4b` | matches the hash recorded in `references.bib` ✓ |
| 8 | $N_\text{PW}$ (largest k) $=150,415,754,1174,2148,3287$ at $E_\text{cut}=5\ldots40$, $4^3$ grid | basis enumeration | exact, all six |
| 9 | Test size: 544 plane waves, $24^3$ grid at $E_\text{cut}=12$ | basis + 2a grid rule | exact |
| 10 | $V_\text{nl}=BDB^\dagger$ equals the Legendre formula; Hermitian; rank $\le N_\text{proj}$ | dense build at a general k, $\lambda=(1,0.7)$, no symmetry | $6\times10^{-14}$ vs matrix-free apply; Hermitian to $0$; exactly 10 nonzero eigenvalues |
| 11 | $E_\text{Ewald}=-8.3980$ Ha | point-charge Ewald, $\eta$-independent to $4\times10^{-15}$ | $-8.398\,009\,228$ ✓ |
| 12 | $E$ vs $E_\text{cut}$, $4^3$ grid: $-7.86914,-7.92143,-7.92685,-7.92747,-7.92770,-7.92772$ | six SCF runs | $-7.869137,-7.921426,-7.926852,-7.927470,-7.927697,-7.927724$ — all six exact to the digits printed |
| 13 | $E$ vs $N^3$ at $E_\text{cut}=15$: $-7.30090,-7.83809,-7.91059,-7.92685,-7.93157,-7.93312$ | six SCF runs | $-7.300904,-7.838092,-7.910590,-7.926852,-7.931573,-7.933125$ — all six exact |
| 14 | Terms at $E_\text{cut}=15$, $4^3$: $3.1728/1.5855/-2.4435/0.5590/-2.4027/-8.3980$ | same run | identical to four decimals, all six |
| 15 | Test size gap over the grid $=0.44$ eV | $2^3$ grid, $E_\text{cut}=12$ | $0.443$ eV ✓ |
| 16 | Grid gap on the $4^3$ grid $= 0.61$ eV | band 5 min − band 4 max | $0.610$ eV ✓ |
| 17 | Band table: $-11.98$, $2.54$, $3.12$, $1.41$, $0.47$ at $0.85\,\Gamma X$, $0.61$, $2.61$, $4.51$, $3.47$ eV | bands from the converged $4^3$ density | $-11.976$, $2.537$, $3.124$, $1.408$, $0.473$ at $0.843$, $0.610$, $2.607$, $4.510$, $3.471$ — all nine |
| 18 | Degeneracies 1,3,3,1 at Γ; pairs at X; 1,1,2,1,2 at L | eigenvalues | ✓ exactly |
| 19 | $a_0=10.193$ Bohr $=5.394$ Å, $B_0=97$ GPa | cubic fit, 7 points, $E_\text{cut}=20$, $4^3$ | $10.1927$ Bohr, $5.3937$ Å, $97.0$ GPa ✓ |
| 20 | $\partial E/\partial f_{1x}=0.102\,076\,953$ Ha at the displaced geometry | analytic Hellmann–Feynman | $0.102\,076\,953\,324$ — all nine digits |
| 21 | FD difference $1.02\times10^{-6}$ at $h=2\times10^{-3}$, $2.55\times10^{-7}$ at $h=10^{-3}$ | central differences of the SCF energy | $1.03\times10^{-6}$, $2.63\times10^{-7}$; clean $h^2$ (ratio 3.9) |
| 22 | Atom derivatives sum to $\approx4\times10^{-6}$, net force $\approx5\times10^{-7}$ Ha/Bohr | same run | $(3.97,3.45,4.55)\times10^{-6}$; $4.94\times10^{-7}$ Ha/Bohr ✓ |
| 23 | Net force at the displaced geometry, $2^3$ grid: $5.7\times10^{-7}$ with XC, $4\times10^{-15}$ without | two SCF runs | $5.68\times10^{-7}$ and $2.5\times10^{-15}$ ✓ |
| 24 | Grid-symmetry table: $8\times10^{-16}$ / $6\times10^{-7}$ / $1\times10^{-7}$ / $1\times10^{-15}$ | forces at the ideal geometry on $N=24,25,25,28$ | $3.8\times10^{-15}$ / $6.2\times10^{-7}$ / $1.4\times10^{-7}$ / $3.3\times10^{-15}$ ✓ |
| 25 | Net force vanishes on every grid (inversion) | same runs | $(3\ldots9)\times10^{-16}$ on all grids ✓ |
| 26 | Egg-box: $2.0\times10^{-7}$ Ha for half a grid spacing, $2\times10^{-15}$ for a whole one | rigid translation, $E_\text{cut}=12$, $N=24$ | $2.014\times10^{-7}$ and $-5.3\times10^{-15}$ ✓ |
| 27 | Half-shifted $2^3$ grid: largest force $1.0\times10^{-3}$ Ha/Bohr | SCF with $s=\tfrac12$ | $1.01\times10^{-3}$ ✓ |
| 28 | The 2a per-k gap guard accepts a state with a negative gap over the grid | local-only Si, $E_\text{cut}=12$, $2^3$ | gap over grid $-0.063$ Ha, smallest per-k gap $+0.0001$ Ha — qualitative claim confirmed; the post's $-0.03$ Ha presumably comes from a different setting |
| 29 | Weight derivative: the two sides differ by $Z\times0.112$ Ha while the grid gap is $0.010$ Ha | displaced geometry, $3^3$ grid | BZ-average of $\varepsilon_5-\varepsilon_4 = 0.1122$ Ha, grid gap $0.0103$ Ha ✓ |
| 30 | $\varepsilon_{n,-\mathbf{k}}=\varepsilon_{n\mathbf{k}}$; shifting $\mathbf{k}$ by $\mathbf{G}_0$ changes nothing | explicit, $2^3$ grid | $1.1\times10^{-14}$ and $1.2\times10^{-14}$; k/−k folding reproduces the energy bit-for-bit |
| 31 | Diamond with origin at the bond centre: inversion symmorphic, tetrahedral operations with fractional translations | brute-force space group from the 48 cubic operations | 48 operations, 12 symmorphic, inversion among them ✓ — but see finding 1 for the translations |
| 32 | Mixing: converged in 9–10 Anderson steps at every size | Pulay, $\tau=10^{-9}$ | 10–12 steps everywhere, no stalls; clip-and-renormalise inactive near convergence ✓ |

Not checked here: the Quarto rendering; the DFTK comparison itself (not run); the claim
that H₂ needs 80 Ha for comparable convergence; the 6 s wall time; and the specific
$-0.03$ Ha in the 2a-guard anecdote.

---

## 2. Findings

### 1. The grid condition is $N$ even, not $4\mid N$ — **major, physics error**

> "the operations of the tetrahedral site group carry fractional translations that are
> multiples of $\tfrac14$ of the lattice vectors. They map the grid onto itself only if
> $4\mid N$."

Computing the space group of diamond by brute force (all 48 cubic operations, atoms at
$\pm(\tfrac18,\tfrac18,\tfrac18)$ in primitive fractional coordinates, origin at the bond
centre) gives 48 operations whose required fractional translations are exactly

$$(0,0,0),\quad (\tfrac12,0,0),\quad (0,\tfrac12,0),\quad (0,0,\tfrac12)$$

in the **primitive** fractional basis — 12 operations each. They are *halves* of
primitive lattice vectors. The $\tfrac14$ in the post is the same translation expressed
in **cubic** fractional coordinates, where these three vectors read
$(\tfrac14,\tfrac14,0)$, $(\tfrac14,0,\tfrac14)$, $(0,\tfrac14,\tfrac14)$. But the FFT grid
is laid out along the primitive vectors, $r_j = \sum_i (j_i/N_i)\mathbf{a}_i$, so the
condition is $N_i/2\in\mathbb{Z}$: **$N$ even in every direction**.

Direct confirmation, both by grid mapping and by forces at the ideal geometry:

| $N$ | operations mapping the grid onto itself | largest force, $E_\text{cut}=12$ | largest force, $E_\text{cut}=15$ |
|:--:|:--:|:--:|:--:|
| 24 ($4\mid N$) | 48/48 | $3.8\times10^{-15}$ | — |
| 25 (odd) | 12/48 | $6.2\times10^{-7}$ | $1.4\times10^{-7}$ |
| **26 (even, $4\nmid N$)** | **48/48** | $\mathbf{2.4\times10^{-15}}$ | $\mathbf{5.3\times10^{-15}}$ |
| 27 (odd) | 12/48 | — | — |
| 28 ($4\mid N$) | 48/48 | — | $3.3\times10^{-15}$ |
| 30 (even, $4\nmid N$) | 48/48 | — | — |

The post's table cannot distinguish the two rules because its only symmetry-breaking
entry, $N=25$, is odd and therefore fails both. $N=26$ is the discriminating case and
gives machine-precision forces.

**Change.** Replace $4\mid N$ with "$N$ even in every direction" throughout (the grid
discussion, the invariant list, the tolerances section), and fix the test-size
annotation "a $24^3$ grid, so $4\mid N$" → "so $N$ is even". Add the $N=26$ row, or
replace the $N=25$ rows with a $26$ and a $25$ at the same cutoff so the table shows
the discriminating pair. If the $\tfrac14$ is kept as intuition, say explicitly that it
is in cubic coordinates and that the primitive-basis translation is $\tfrac12\mathbf{a}_i$.

This materially changes **open question 3** (see below): the default rule already
produces an even $N$ most of the time.

### 2. Why shifted grids break the symmetry — **medium, the argument as given is misleading**

> "A half-shifted grid in the fcc primitive basis is not invariant under the cubic group."

As a statement about the k-point *set* this is true: for the half-shifted $2^3$ grid,
288 of the $48\times8$ images fall outside the grid modulo reciprocal lattice vectors;
e.g. $(\tfrac14,\tfrac14,\tfrac14)\tfrac{2\pi}{a}$ rotates to $(\tfrac14,-\tfrac14,\tfrac14)\tfrac{2\pi}{a}$,
which is not in the grid. The measured $1.01\times10^{-3}$ Ha/Bohr force confirms the
consequence.

But the statement as written will look wrong to any reader who knows that shifted
Monkhorst–Pack grids are the standard choice for fcc, and the reason is worth a
sentence, because it is the interesting part. Decomposing the grid into **full** cubic
stars:

| grid | stars touched | members present / full star size |
|:--|:--:|:--|
| unshifted $2^3$ | 3 | 1/1, 4/4, 3/3 — **all** members |
| unshifted $4^3$ | 8 | 1/1, 8/8, 4/4, 6/6, 24/24, 12/12, 3/3, 6/6 — **all** |
| half-shifted $2^3$ | 2 | 2/8, 6/24 — **exactly one quarter of each** |
| half-shifted $4^3$ | 10 | 2/8, 6/24 (×5), 12/48 (×2), 2/8, … — **exactly one quarter of each** |

Because the shifted grid contains the *same fraction* of every star, a uniform-weight
sum over it of any point-group-invariant scalar equals the fully symmetric
star-weighted average exactly. That is why Monkhorst and Pack's $q=2$ fcc prescription
reduces to the Chadi–Cohen pair with weights $\tfrac14$ and $\tfrac34$, and why a
symmetry-reducing code can still report a small set of inequivalent points for a shifted
fcc grid — my orbit count gives 2 and 10 for the shifted $2^3$ and $4^3$ grids, the
familiar numbers. Band energies, eigenvalue sums and the density of states are therefore
symmetry-correct on a shifted grid.

What is *not* correct is the **density**: $\rho_{R\mathbf{k}}$ is a rotated copy of
$\rho_\mathbf{k}$, not equal to it, so $\sum_\mathbf{k}w_\mathbf{k}\rho_\mathbf{k}$ needs
the grid to contain every star member, not a fixed fraction of each. A code that does
not symmetrise the density therefore converges to a slightly asymmetric $\rho$, and the
force at the ideal geometry is the symptom. That is exactly the right justification for
the Γ-centred choice, and it also explains why DFTK is free to use shifted grids: it
symmetrises.

**Change.** State it as: the shifted grid samples the same fraction of every star, so BZ
integrals of invariant scalars stay symmetric — what breaks is the per-k density, which
is why a code without density symmetrisation must use the Γ-centred grid.

### 3. The price of the Γ-centred grid is never measured — **medium, missing evidence**

> "The price is slower convergence with $N$ than shifted grids give (table below)."

The table below contains no shifted grids. Measured, all at $E_\text{cut}=15$ with the
unshifted $6^3$ value as reference:

| grid | k-points | $E$ (Ha) | error vs unshifted $6^3$ |
|:--|:--:|:--:|:--:|
| unshifted $2^3$ | 8 | $-7.838\,092$ | 95.0 mHa |
| **half-shifted $2^3$** | **8** | $-7.927\,201$ | **5.9 mHa** |
| unshifted $4^3$ | 64 | $-7.926\,852$ | 6.3 mHa |
| **half-shifted $3^3$** | **27** | $-7.933\,259$ | **−0.1 mHa** |
| unshifted $6^3$ | 216 | $-7.933\,125$ | 0 |

The half-shifted $2^3$ grid matches the unshifted $4^3$ grid at an eighth of the
k-points, and the half-shifted $3^3$ grid is converged to the sub-0.1 mHa level with an
eighth of the k-points of the unshifted $6^3$. That is the honest statement of what the
symmetry choice costs, and it is large enough that a reader may reasonably want the
shifted grid plus density symmetrisation at a later milestone.

A corollary: the post guesses that "$6^3$ is itself perhaps 0.5 mHa from the limit". The
shifted $3^3$ result sits $0.13$ mHa *below* the unshifted $6^3$ (which is allowed —
convergence in $N$ is not variational, as the post says), so the limit is within roughly
$0.2$ mHa of the $6^3$ value, not $0.5$. Quoting the shifted-grid number would let the
post state that instead of guessing it.

### 4. Buffer bands: iteration count is the wrong metric — **medium, the measurement does not support the rule**

> "asking for 5 bands took 535 eigensolver iterations in total over the SCF, and 8 bands
> took 424 … The alternative is to keep requiring all bands to converge, at the measured
> cost of about 25 % more eigensolver iterations."

Iterations are not the cost; iterations × block width is, because each column costs a
pair of FFTs. Counting band-applications (one FFT pair each) over a full SCF at
$E_\text{cut}=15$ on the $4^3$ grid, LOBPCG with the same tolerance:

| bands requested | total band-applications | SCF steps | total energy |
|:--:|:--:|:--:|:--|
| 5 | 37 036 | 11 | $-7.926\,851\,643$ |
| 6 | 43 467 | 11 | $-7.926\,851\,643$ |
| 8 | 64 571 | 11 | $-7.926\,851\,643$ |
| 10 | 67 026 | 11 | $-7.926\,851\,644$ |

Per-band iteration counts do fall with buffer bands, consistent with the post, but the
total work rises by 74 % from 5 to 8 bands while the energy is unchanged to $10^{-9}$.
On a GPU a wider block is cheaper per column, so the balance can flip — which is the
point: this is a wall-clock question on the target device, not an iteration-count
question. The post's two numbers do not establish that buffer bands are cheaper.

**Change.** Keep the rule (converge only the needed bands; the buffer is a setting), but
state the cost in band-applications or seconds, and say the default is to be fixed by
measurement on the GPU. Weaken "It matters more for the next point" accordingly.

### 5. The nonlocal term needs no square root and no angular guard at all — **medium, simplification**

The post's rule is to evaluate $\big[q^lY_{lm}(\hat{\mathbf{q}})\big]\cdot\big[\hat p_i^l(q)/q^l\big]$
and never form $\hat{\mathbf{q}}$. That is right, but it stops one step short. Both
factors are polynomials in the Cartesian components of $\mathbf{q}$ times
$e^{-q^2r_l^2/2}$:

- $q^lY_{lm}$ is a solid harmonic — a homogeneous polynomial (the post says so);
- $\hat p_i^l(q)/q^l$ is a polynomial in $y=q^2r_l^2$ times $e^{-y/2}$ — look at the
  table: every entry's $q$-dependence is through $y$ and the explicit factor $q^l$.

So the entire matrix element is a polynomial in $q_x,q_y,q_z$ times a Gaussian, with
**no $\sqrt{\cdot}$ anywhere**. This matters because the existing `GTH.local(q)` takes
the *magnitude*, and `atomic_local.py` protects it with the double-`where`
`q = where(nz, sqrt(where(nz, g2, 1)), 0)`. Mirroring that signature for the projectors
(the post says "the reduced projectors $\hat p_i^l(q)/q^l$ as traceable functions of $q$")
pulls a square root back into a term that does not need one, and then the derivative at
$\mathbf{k}+\mathbf{G}=0$ is correct only because $\nabla q^2 = 0$ wherever $q=0$ —
true, but a non-obvious fact for the post to be relying on silently, and false for
second derivatives (phonons, out of scope today).

**Change.** Make the protocol take $q^2$ (or the vector $\mathbf{q}$) rather than $q$,
state the polynomial-times-Gaussian form as the reason the $\Gamma$ point needs no
special case, and keep the invariant "finite value and finite derivatives when the basis
contains $\mathbf{k}+\mathbf{G}=0$" as the regression test.

### 6. Ghost states are not mentioned — **medium, hidden detail**

This is the post that introduces separable projectors, and it does not say that the
Kleinman–Bylander form is *not* equivalent to the semilocal pseudopotential away from
the atomic reference state, and can support spurious deeply bound states. For HGH the
local part is analytic rather than a chosen channel, so the usual "which channel is
local" failure mode does not arise, but the general warning belongs here — a reader
told only "the projectors are repulsive" has no reason to expect the failure mode
exists.

It also supplies a free invariant: **no eigenvalue below the 3s-derived band**. In my
run the lowest eigenvalue at Γ is $\Gamma_1$ at $-11.976$ eV with nothing below it, at
every k-point of the grid. One assert, and it would catch a mis-signed $h^l$ or a
mis-parsed projector block immediately.

### 7. "Smooth, periodic function of $\mathbf{k}$" needs one qualification — **minor, pedagogy**

> "For an insulator the integrand is a smooth, periodic function of $\mathbf{k}$, and the
> average over a uniform grid converges exponentially with the grid density
> [@monkhorst_pack1976]"

Individual band energies are *not* smooth: $\varepsilon_{n\mathbf{k}}$ is degenerate at
symmetry points and bands cross inside the valence manifold, so each $\varepsilon_n$ has
kinks. What is analytic is the projector onto the occupied manifold, hence
$\sum_{n\le4}\varepsilon_{n\mathbf{k}}$ and $\rho_\mathbf{k}$ — which is what the energy
and the density are built from, and it is analytic precisely because the gap does not
close. That is also the only place the insulating condition enters, and it is worth one
sentence because the post's tests elsewhere do use individual band energies.

Relatedly, @monkhorst_pack1976 constructs special points; it does not prove exponential
convergence. The exponential rate is the periodic-trapezoid (Poisson summation)
argument: the error is the set of Fourier coefficients of the integrand at the aliasing
vectors, which decay exponentially for an analytic integrand. One clause would make the
citation carry the right weight.

### 8. "This is exact, not an approximation" overstates the band structure — **minor**

The diagonalisation at an off-grid $\mathbf{k}$ is exact, as the post says. But
$\rho^\ast$ is the density of the *discrete* k-grid problem, so the band energies
inherit its k-grid error. Measured, same cutoff, $4^3$ vs $6^3$ density:

| quantity (eV) | from $4^3$ density | from $6^3$ density |
|:--|:--:|:--:|
| valence-band width | $-11.976$ | $-11.966$ |
| $\Gamma_{25'}-\Gamma_{15}$ | $2.537$ | $2.553$ |
| $\Gamma_{25'}-\Gamma_{2'}$ | $3.124$ | $3.124$ |
| $\Gamma_{25'}-\mathrm{L}_1$ | $1.408$ | $1.419$ |
| conduction level at X | $0.610$ | $0.634$ |
| indirect gap | $0.473$ at $0.843\,\Gamma X$ | $0.496$ at $0.842\,\Gamma X$ |

So the band table carries roughly $0.02$ eV of k-grid error — negligible against the
$0.3$–$0.9$ eV difference from Cohen–Bergstresser, and it moves the indirect gap from
$0.47$ to $0.50$ eV, *closer* to the $0.52$ eV of @hybertsen_louie1986. Worth quoting:
it strengthens the comparison the post is drawing, and it tells the reader what "exact"
does and does not cover.

### 9. Monotonicity in $E_\text{cut}$ in the table is not the variational statement — **minor**

The invariant is correctly qualified ("monotone in $E_\text{cut}$ **at a fixed grid and
fixed k-grid**"), but the prose about the table — "The energy decreases monotonically
with the cutoff, as it must" — is about a table whose FFT grid changes at every row
(15, 24, 25, 30, 36, 45). Changing the grid at fixed cutoff is not variational:
measured at $E_\text{cut}=12$, $E = -7.836\,151\,895$, $-7.836\,151\,704$,
$-7.836\,151\,623$ on $N = 24, 25, 26$ — a $2.7\times10^{-7}$ Ha spread, and the larger
grid gives the *higher* energy. Here it is harmless (the smallest cutoff step,
$30\to40$ Ha, is $2.7\times10^{-5}$ Ha, a hundred times larger), but the assert should
either hold the grid fixed or carry a tolerance of that order. One clause: "as it must
at a fixed grid; the table varies the grid, and that part is empirical".

### 10. The grid rule needs one extra inequality once $\mathbf{k}\ne0$ — **minor, worth stating**

The 2a aliasing argument is k-independent — two basis vectors at the same $\mathbf{k}$
have $|\mathbf{G}-\mathbf{G}'|\le2G_\text{max}$ whatever $\mathbf{k}$ is — so the rule
$N_i \ge 2\lfloor 2G_\text{max}|\mathbf{a}_i|/2\pi\rfloor+1$ carries over unchanged, which
the post is right to assume silently. What does *not* carry over is the placement of
$\psi$ on the grid: the k-dependent sphere reaches $|\mathbf{G}|\le G_\text{max}+|\mathbf{k}|$,
so $\max|m_i|$ grows with the k-grid and must stay below $N/2$ or two basis functions
share a grid frequency. It holds with room everywhere here ($\max|m_i| = 6$ vs $N/2=12$
at $E_\text{cut}=12$; $11$ vs $22.5$ at $40$ Ha), and `basis/planewave.py` already
computes its enumeration bound with `kmax` included — so this is one sentence and one
assert, not a problem.

### 11. When is the gap guard evaluated? — **minor, specify**

The post fixes *what* the guard tests but not *when*. Mid-SCF the fixed occupations can
be non-aufbau for several iterations even when the converged state is gapped (the
initial Gaussian density has a much smaller gap), so a guard evaluated every iteration
would reject healthy runs, and one evaluated only at convergence reports the failure
after the user has paid for the whole SCF. Recommend: evaluate at convergence as the
acceptance criterion, and additionally log the gap over the grid each iteration so a
closing gap is visible in the history. Say which in the post.

### 12. Supercell equivalence has extra variational freedom — **minor, affects open question 4**

The $\Gamma$-only supercell problem admits densities with only the supercell period, so
the primitive-cell solution is *a* fixed point of the supercell SCF but not obviously
*the* one it finds. The test should start the supercell from the symmetric atomic
initial density and assert both the energy per cell and that the converged supercell
density is invariant under the primitive translation. Worth one clause, since otherwise
a failure of this test is ambiguous between "the weights are wrong" and "the supercell
SCF found a different state".

### 13. Attribution of the structural-properties claim — **minor, check before publishing**

> "Total-energy calculations in the LDA have reproduced the lattice constant and bulk
> modulus of silicon to about a per cent and a few per cent since [@yin_cohen1982], with
> the LDA tending to overbind slightly."

The accuracy claim is fine. The "overbind slightly" clause sits awkwardly next to the
citation: Yin and Cohen used the Wigner correlation functional and their own
pseudopotentials, and as I recall their silicon lattice constant came out slightly
*above* experiment, not below. I could not verify their table from here, so treat this
as "check the paper", not as an error. The safe formulation is to state the modern-LDA
result the post actually measured ($0.7\,\%$ below experiment with PW92 and HGH) and
note that the sign depends on the correlation parametrisation and the pseudopotential.

### 14. Provenance of the Cohen–Bergstresser column — **minor**

The column is headed "CB calculated [@cb1966]" and the values match the published
Table I. Worth one word ("published") to distinguish them from the converged
reproduction of 1b, which differs from Table I by up to $0.089$ eV — and from
Cohen–Bergstresser's own 15-plane-wave basis, which differs by up to $0.53$ eV. At the
0.1 eV rounding of the table it makes no difference to the conclusion, but the
distinction has already caused one trap in the 1b acceptance test.

### 15. Smaller wording points

- "Silicon has neither a local-only pseudopotential nor a converged energy at Γ" — the
  Γ-only energy is 0.63 Ha off, so "converged" understates it; it is not even
  qualitatively right (the Γ-only gap over the grid is 2.13 eV against 0.47 eV).
- "$\hat p_i^l(q)$ … are cell-independent, in Bohr$^{3/2}$" — worth adding that the
  reduced form $\hat p_i^l/q^l$ carries Bohr$^{3/2+l}$, since that is the quantity the
  protocol actually exposes.
- "the factors $i^l$ cancel between bra and ket" — only because $h^l$ is diagonal in
  $l$; one clause avoids the reader checking whether cross-$l$ terms exist.
- The sum over $m$ is basis-independent within the $l$ shell: any real orthonormal basis
  of the $(2l+1)$-dimensional space gives the same $V_\text{nl}$, which is what the
  addition theorem says. Worth stating, because it means the real-harmonic convention
  and the $m$ ordering need not match DFTK's for the comparison to agree — one less
  convention to pin.
- The post never says how many bands the *path* calculation needs; the quoted
  $\mathrm{L}_3$ and $\mathrm{X}_1$ levels require eight to ten, not five.
- Egg-box explanation: "The silicon density has more weight at large $|\mathbf{G}|$
  relative to the grid" is plausible but asserted. The mechanism chain is worth one
  sentence, since it is not obvious that a term with no explicit position dependence can
  produce a force: the XC quadrature is the only term that is not exactly
  translation-covariant, so it is the only term that lets the self-consistent density
  break the symmetry, and that asymmetric density then produces a nonzero force through
  the terms that *do* depend explicitly on position ($E_\text{loc}$, $E_\text{nl}$,
  $E_\text{Ewald}$). The XC-off control in the post is exactly the right test of that
  chain.

---

## 3. Answers to the open questions

**1. Fractional electron counts at several k-points.** Option (a), and your reason for
it is right: an untested derivative is an untested claim. Two observations make (a)
cheaper than it looks. First, the structural identity is strong on its own — I reproduce
your $Z\times0.112$ Ha versus a $0.0103$ Ha grid gap, so the broken 2a behaviour is
detected by the identity alone, with no fractional-charge SCF. Second, the fragility you
fear is controllable: the frontier is at least two-fold degenerate whenever
$\mathbf{k}$ is not time-reversal invariant, so choose the test geometry and grid so the
frontier sits at a time-reversal-invariant point of the grid, where the level is
non-degenerate and the "share equally within a tolerance" branch is not exercised at
all. Then run one fractional case with a degenerate frontier as a separate, looser test.
Also note that the tolerance-based frontier set makes $E(\lambda)$ only piecewise
smooth, so the finite difference must be one-sided with $\delta$ small enough that the
frontier set does not change between $\lambda$ and $\lambda\mp\delta$ — assert that set
equality inside the test rather than discovering it as noise.

**2. Buffer bands.** Keep "converge $N_\text{el}/2+1$, carry a few more", but see
finding 4: your two iteration counts do not show that buffer bands are cheaper, and in
band-applications they are 74 % *more* expensive at nb = 8 than at nb = 5 for an
identical energy. Pick the default from a wall-clock measurement on the GPU, where the
block-width effect runs the other way. My measurement suggests the useful range is
small: nb = 6 costs 17 % more than nb = 5, and nb = 10 costs barely more than nb = 8. A
default of one or two buffer bands, with the setting exposed, looks right; "3" is
defensible but is not supported by the evidence currently in the post.

**3. Grid symmetry.** This changes with finding 1. The real condition is $N$ even, and
the 2a rule already returns an even $N$ whenever $2\lfloor\cdot\rfloor+1$ rounds up to an
even FFT-friendly size — $23\to24$ at $E_\text{cut}=12$, $29\to30$ at 20 Ha. It returns
an odd $N$ only when it lands exactly on an odd smooth number, as at $E_\text{cut}=15$
($N=25$) and $27$. So: offer the rounding-up option (`even_fft_size=True` or similar),
make the symmetry tests use it, and leave the default alone. Two notes. (i) The cost is
the step to the next even FFT-friendly size and can be appreciable if only radices 2, 3
and 5 are allowed ($25\to30$ is $1.7\times$ the grid points); allowing 7 makes it
$25\to28$, $1.4\times$. (ii) DFTK has exactly this notion — `symmetries_respect_rgrid`
and `next_compatible_fft_size`, which round the grid up so that symmetries with
fractional translations of denominator 2, 3, 4 or 6 are representable, and which default
to *off* when an explicit `fft_size` is given. Since your DFTK protocol passes an
explicit `fft_size` *and* `symmetries=false`, that is consistent — but it is worth one
line in the protocol so the next reader does not wonder, and it is independent
corroboration that "compatible grid size" is the standard name for this concept.

**4. Supercell equivalence test.** Worth the care. It is the only test in the list that
checks the k-point weights against something other than DFTK, and it also exercises the
nonlocal term's per-atom sum (four atoms, $N_\text{proj}=20$) and the $\mathbf{G}=0$
bookkeeping at a different $\Omega$ and $N_\text{el}$. The FFT grids are not a real
obstacle: pass $(2N_1,N_2,N_3)$ for the supercell against $(N_1,N_2,N_3)$ for the
primitive cell and the two discrete problems are identical, because the supercell
$\Gamma$ basis is precisely the union of the primitive per-k bases at the same cutoff.
Add the caveat of finding 12. If you want it cheaper, run it with XC switched off first:
then the agreement should be at solver tolerance and a failure is unambiguous.

**5. Time-reversal reduction.** Leave it out, and go further than the post: say in the
text *why* it is exactly free when you do add it, because that is the pedagogical
content. $\rho_{-\mathbf{k}} = \overline{\rho_\mathbf{k}} = \rho_\mathbf{k}$, so folding
$\mathbf{k}$ with $-\mathbf{k}$ changes nothing at all in the density — I folded the
$2^3$ grid to representatives with weights $1/8$ and $2/8$ and reproduced the unfolded
energy bit-for-bit ($-7.836\,151\,895$ both), with
$\varepsilon_{n,-\mathbf{k}}=\varepsilon_{n\mathbf{k}}$ to $1.1\times10^{-14}$. The only
thing needing care is that the gap guard and the frontier search run over the unfolded
grid, or equivalently over representatives with their weights, which is the same set of
eigenvalues. So the reduction is a pure factor of two with no physics attached — which
is a good reason to defer it to a milestone where cost matters, and to say that is why.
