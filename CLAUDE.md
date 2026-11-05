# JAX DFT

A GPU-first plane-wave density functional theory code written in JAX.

## Purpose

This project has two goals, in this order:

1. Explore building a substantial scientific code through agentic development.
2. Produce real, verifiable physics along the way.

It is not trying to be a state-of-the-art production DFT code. Correctness,
clarity and sound architecture matter more than feature count or peak
performance. Work proceeds milestone by milestone; there is no complete
up-front plan, but early choices must not block the long-term ambitions below.

## Stack

- Python, managed with `uv` (`uv sync`, `uv run ...`)
- JAX for all physics: arrays, `jit`, `vmap`, `grad`, sharding
- `pytest` for all tests, `hypothesis` for property-based tests
- Quarto (system install) for documentation, theory, validation and logs
- Possible later: Rust via Maturin/PyO3 for non-differentiable components

### Development environment

- Local development runs JAX on **CPU** (Apple Silicon MacBook). Do not use
  Metal/MPS backends: they are experimental and do not support float64.
- Production and scaling runs use CUDA GPUs on HPC (Slurm).
- All code must run correctly on CPU. Nothing may depend on a GPU being present.
- Keep test systems small enough that the full test suite runs on a laptop CPU.

## Project layout

```
jaxdft/
├── src/jaxdft/        # the package: all physics lives here
├── tests/
│   ├── unit/          # programmatic correctness
│   ├── property/      # hypothesis tests of mathematical invariants
│   └── scientific/    # physical correctness against references
├── reference/         # scripts and stored outputs from reference codes
├── docs/              # Quarto website
│   ├── theory/        # one theory post per milestone
│   ├── validation/    # comparisons against references, with interpretation
│   ├── engineering/   # engineering_log.qmd, performance.qmd
│   └── research/      # research_log.qmd
└── CLAUDE.md
```

## Definition of done

A change is done only when this passes:

```
uv run pytest && quarto render docs/
```

A broken validation document is a failing test.

## Conventions (non-negotiable)

### Units
- Atomic units throughout: energy in Hartree, length in Bohr.
- Convert at the boundary only (input parsing, output/plotting). Never mix units
  inside the package.
- Many published parameters are in Rydberg (e.g. Cohen–Bergstresser form
  factors). Every unit conversion needs a test.

### Precision
- Double precision always. Enable `jax_enable_x64` in one package-level place
  that every entry point goes through.
- Never silently downcast to float32. A test must fail if any physical array
  is float32.

### Arrays and state
- All physical fields are JAX arrays, never NumPy arrays, inside the package.
- No global mutable state.
- Physics functions are pure functions of their inputs.

## Physics invariants

Code must never break these. Check them when refactoring or extending.

- The total energy is variational with respect to the wavefunctions (or density)
  within the chosen functional and basis.
- For nested plane-wave bases at fixed k-point sampling, the total energy
  decreases monotonically with increasing cutoff.
- The electron density integrates to the number of electrons.
- Forces equal −∂E/∂R; verify against finite differences.
- The SCF solution preserves the symmetry of the input structure.

## Architecture

### Design behind abstractions
Physical approximations must be interchangeable without rewriting core
algorithms. The SCF loop must not know or care which XC functional,
pseudopotential, or basis set it uses.

- XC functional: a callable `(density) -> (energy, potential)`
- Pseudopotential: an object exposing reciprocal-space radial data, behind a
  parser interface; nothing outside the parser reads file formats. The local
  part is a form factor `local(q) -> v̂(q)` (atomic Fourier transform,
  Ha·Bohr³, cell-independent). Nonlocal projectors and their coefficients are
  added as further methods in the milestone that first needs them; local and
  nonlocal parts are used by separate Hamiltonian terms.
- Basis set: an object defining representation and transforms between real
  and reciprocal space (plane waves now; keep LCAO possible later)
- Diagonaliser and k-point generation: likewise swappable

If the SCF loop imports anything from the XC, pseudopotential, or basis
modules by name, that is a design smell.

### Hamiltonian as an operator
- Implement H as a matrix-free operator `apply_H(psi)` from the start.
- Dense diagonalisation (`jnp.linalg.eigh`) is fine for small systems, but
  build the dense matrix from the operator and test them against each other.
  Iterative solvers (LOBPCG, Davidson) must slot in without changing H.

## Long-term ambitions

Not current requirements. When making architectural choices, prefer options
that keep these achievable. Where an ambition lists concrete implications,
follow them from day one.

### Multi-node, multi-GPU
The code should eventually run across multiple nodes and GPUs on HPC using
JAX distributed primitives (`jax.distributed.initialize()`, sharding,
collectives). No mpi4py.
- K-points are a batch dimension always: use `vmap`, not Python loops.
- Design data layouts that can be sharded (k-points first, later bands or
  G-vectors).
- Keep communication implicit in array operations and sharding, not manual.

### Automatic differentiation throughout
- The total energy is a differentiable pure function of atomic positions,
  cell parameters and per-atom weights from the first milestone that has a
  total energy.
- Forces and stress come from `jax.grad(energy)`, never implemented separately.
- Do not optimise away intermediates that autodiff must traverse.
- Later targets: Hessians (phonons, via Hessian-vector products), elastic
  constants, field responses, band derivatives (group velocities, effective
  masses), sensitivity to XC or pseudopotential parameters.
- Eigenvector derivatives are ill-defined at degeneracies. Prefer quantities
  invariant within degenerate subspaces (density, energy, band sums).

### Differentiating through SCF
- The SCF is a fixed-point problem. Write it as `solve(F, initial_state)`
  where `F` is one SCF update, so implicit differentiation can wrap it later
  (`custom_vjp` around the fixed-point solve).
- Never rely on unrolling SCF iterations for gradients.
- First derivatives may `stop_gradient` the converged density
  (Hellmann–Feynman). Second derivatives may not.

### Per-atom weights
- The energy function takes per-atom weights λ ∈ [0, 1] (default 1) that
  scale each atom's local pseudopotential, nonlocal projectors, ionic charge
  in the Ewald sum, and its contribution to the electron count (keeping the
  cell neutral).
- This makes atom existence differentiable. Endpoint derivatives (λ = 0, 1)
  follow from a single SCF via Hellmann–Feynman.
- In gapped systems dE/dλ is one-sided (valence vs conduction band); report
  both sides rather than a smeared value.

### Performance
- Prefer operations XLA can fuse; no Python loops over array elements.
- FFTs dominate cost: never move data off-device between FFT steps.
- Keep JIT boundaries coarse (e.g. one `jit` over a full SCF iteration),
  not many small jitted functions.
- Profile before optimising. Track wall time per milestone in
  `docs/engineering/performance.qmd`.

### Rust interop (aspirational only)
- Candidates: pseudopotential file parsing, k-point generation, symmetry
  analysis, IO, SCF mixing bookkeeping.
- Keep these as pure functions with plain array interfaces so they could move
  to Rust later. Do not design around Rust now.
- A Rust function inside a differentiable path needs a `custom_vjp` with an
  explicitly supplied gradient. Prefer keeping anything physics touches in JAX.

## Testing

### Programmatic correctness
- Unit tests for every module.
- Hypothesis for invariants that must hold for any valid input: FFT round
  trips, normalisation, hermiticity of H, operator vs dense-matrix agreement,
  unit conversion round trips.
- Finite-difference force tests are mandatory once forces exist. They validate
  the energy, the forces, and the autodiff path at once.

### Scientific correctness
A separate concern from "the code does what it was written to do". Validate
against a ladder of references, in this order:

1. Analytic limits (empty lattice / free electron dispersion, exactly
   solvable cases)
2. Published numerical results (e.g. empirical pseudopotential silicon bands)
3. Reference codes run on the same system (DFTK, GPAW, Quantum ESPRESSO)

Scientific tests encode a physical expectation and a tolerance with a stated
justification. Reference data lives in `reference/` with the script that
produced it.

## Milestone workflow

Each milestone follows the same loop:

1. **Theory post** in `docs/theory/`, scoped to this milestone's physics only.
2. **Review** of the theory post before any implementation.
3. **Acceptance criteria and reference data**, written as failing scientific
   tests derived from the theory post.
4. **Implementation** until unit, property and scientific tests pass.
5. **Validation document** in `docs/validation/`: plots, deviations, and an
   interpretation of whether results are physically sensible.
6. **Log entries**: engineering log for design decisions, research log for
   what results mean.

Do not write or change acceptance tests to fit an implementation. If a test
looks wrong, stop and raise it.

### Milestones so far
- **Step 0:** package, `uv` environment, pytest + hypothesis, Quarto site,
  x64 setup and test, this file.
- **Milestone 1a:** empty lattice band structure for an fcc lattice. Plane-wave
  basis, reciprocal lattice, cutoff, k-point paths (vmapped), kinetic operator,
  diagonalisation. Validated against analytic folded free-electron bands.
- **Milestone 1b:** Cohen–Bergstresser empirical pseudopotential silicon.
  Fixed local potential from published form factors and the structure factor.
  Validated against published band structures. No SCF, XC, or total energy.
- **Next (outline only):** self-consistency with Hartree (FFT Poisson) + LDA
  and a real norm-conserving pseudopotential, total energy, Ewald, forces.

Later milestones are defined when we get there.

## Documentation

- Analysis, theory and validation go in `docs/` as `.qmd` files.
- Quarto documents are for understanding and analysis, not computation. They
  call `jaxdft` functions; they never reimplement physics.
- Never hardcode numerical results in documents. Compute them from the library
  so docs stay in sync with code.
- Plots use a shared style module. Prefer interactive charts when they allow for deeper analysis, using e.g. altair.

### Logs
- `docs/engineering/engineering_log.qmd`: design decisions, refactors,
  numerical stability issues, data layout and performance choices. Supplements
  git history with the reasoning.
- `docs/research/research_log.qmd`: physical results, unexpected behaviour and
  its interpretation, comparisons with literature, choices of approximation.
- Add entries as new dated H2 sections. Never edit existing entries.

## Using reference implementations

Reference codes (DFTK.jl, GPAW, Quantum ESPRESSO, ABINIT) are sources of
understanding and reference data, not code.

- Consult them with a specific question ("how is the G = 0 term handled?"),
  not by browsing.
- Literature and documentation first, source code second. DFTK is the most
  readable modern reference.
- Never copy or line-by-line translate source. Write what you learned in your
  own words (theory post or engineering log), then implement from that note.
- Record which reference informed a design choice in the engineering log,
  with file/function or paper citation.
- When conventions differ between codes (FFT normalisation, G-vector ordering,
  structure factor signs, cutoff definitions), document the difference and
  our choice explicitly.
- Prefer running reference codes to generate validation data over inferring
  expected results from their source.
- Licences: GPAW, Quantum ESPRESSO and ABINIT are GPL; DFTK is MIT.
