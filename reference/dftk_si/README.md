# DFTK reference: silicon, LDA, HGH, k-grids (Milestone 2b)

`dftk_si.json` holds DFTK results for the discrete problem jaxdft solves at the
Milestone 2b acceptance size. It is produced by `run_dftk.jl`, from the
repository root:

```
~/.juliaup/bin/julia --project=reference/dftk_si -e 'using Pkg; Pkg.instantiate()'
~/.juliaup/bin/julia --project=reference/dftk_si reference/dftk_si/run_dftk.jl
```

It takes about ten seconds. `Project.toml` and `Manifest.toml` pin the
environment (the same versions as `reference/dftk_h2`). The JSON records the
Julia, DFTK and libxc versions it was made with.

**The calculation** (theory post 02b, 'Against references', 'Test size'):

- **Cell and basis:** diamond silicon, a = 10.263 Bohr, fcc primitive vectors
  as in `jaxdft.fcc_lattice`, atoms at ±(⅛, ⅛, ⅛) as in `jaxdft.diamond`,
  E_cut = 12 Ha. The FFT grid is set explicitly to jaxdft's 24³, so both codes
  solve the same discrete problem.
- **k-points:** Γ-centred Monkhorst–Pack grids, `MonkhorstPack(n; kshift=0)`.
  DFTK lists the points in [−½, ½), jaxdft in [0, 1); the tests match them
  modulo 1.
- **Model:** `model_DFT(...; functionals=LDA(), symmetries=false)`, zero
  temperature. `LDA()` is libxc `lda_x` + `lda_c_pw`. With symmetries off DFTK
  neither reduces the k-grid nor symmetrises the density; `n_symmetries` = 1
  in the JSON confirms it. With an explicit `fft_size`, DFTK's search for a
  symmetry-compatible grid is inactive; the script asserts the grid it used.
- **Pseudopotential:** the same HGH file jaxdft vendors,
  `src/jaxdft/pseudo/data/hgh_lda/si-q4.hgh`, identified by its SHA-256 in the
  JSON, read with DFTK's HGH parser `PspHgh` (DFTK 0.8.2's `load_psp`
  recognises only `.gth`).
- **SCF:** tolerance 1e-11 on the L2 density change. Five bands are converged
  (`FixedBands`): the four occupied and the lowest conduction band.
- **Geometries:**
  - `ideal`: the diamond structure on the 2³ test grid;
  - `displaced`: atom 0 moved by (0.012, −0.005, 0.008) fractional
    (`tests/support/silicon.py::DISPLACEMENT`), on the 3³ grid, where the gap
    over the grid stays open.

**Conventions to know when comparing:**

- DFTK reports the G = 0 part of the local pseudopotential, N_el Σⱼ αⱼ / Ω, as
  the separate term `PspCorrection` (−0.294634 Ha here). jaxdft includes it in
  `AtomicLocal`.
- DFTK keeps that term out of the Hamiltonian. Its eigenvalues are therefore
  higher than jaxdft's by −Σⱼ αⱼ / Ω = 0.036829 Ha.
- Forces are Cartesian, in Ha/Bohr, and equal −∂E/∂R.

**Use.** These are code-to-code acceptance data for
`tests/scientific/test_reference_dftk_si.py`. The tolerances are fixed in the
theory post: 1e-7 Ha per energy term and in total, 1e-6 Ha for eigenvalues and
1e-6 Ha/Bohr for forces.
