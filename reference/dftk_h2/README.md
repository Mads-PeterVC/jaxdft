# DFTK reference: H₂, LDA, HGH, Γ (Milestone 2a)

`dftk_h2.json` holds DFTK results for the discrete problem jaxdft solves at the
Milestone 2a acceptance size. It is produced by `run_dftk.jl`, from the
repository root:

```
~/.juliaup/bin/julia --project=reference/dftk_h2 -e 'using Pkg; Pkg.instantiate()'
~/.juliaup/bin/julia --project=reference/dftk_h2 reference/dftk_h2/run_dftk.jl
```

It takes a few seconds. `Project.toml` and `Manifest.toml` pin the environment.
The JSON records the Julia, DFTK and libxc versions it was made with.

**The calculation** (theory post 02a, 'Against references' and decision 5):

- **Cell and basis:** a cubic box, L = 10 Bohr, Γ only, E_cut = 10 Ha. The FFT
  grid is set explicitly to jaxdft's 30³, so both codes solve the same discrete
  problem.
- **Model:** `model_DFT(...; functionals=LDA(), symmetries=false)`. `LDA()` is
  libxc `lda_x` + `lda_c_pw`. Symmetries are off, so DFTK does not symmetrise
  the density at the distorted geometry.
- **Pseudopotential:** the same HGH file jaxdft vendors,
  `src/jaxdft/pseudo/data/hgh_lda/h-q1.hgh`, identified by its SHA-256 in the
  JSON. DFTK 0.8.2's `load_psp` recognises only the `.gth` extension, so the
  script calls its HGH parser `PspHgh` directly.
- **SCF:** tolerance 1e-11 on the L2 density change. Two bands are converged
  (`FixedBands`), so the LUMO is converged too.
- **Geometries:**
  - `symmetric`: d = 1.45 Bohr along z, centred in the box (`jaxdft.dimer_in_box`);
  - `distorted`: d = 1.2 Bohr, tilted off every axis, centre off every grid point.
    This is the same construction as `tests/support/h2.py::distorted_off_centre`.
  The JSON stores the fractional positions actually used, and the tests build
  their structures from those.

**Conventions to know when comparing:**

- DFTK reports the G = 0 part of the local pseudopotential, N_el Σⱼ αⱼ / Ω, as
  the separate term `PspCorrection`. jaxdft includes it in `AtomicLocal`.
- DFTK keeps that term out of the Hamiltonian. Its eigenvalues are therefore
  higher than jaxdft's by −Σⱼ αⱼ / Ω = 2.596e-6 Ha.
- Forces are Cartesian, in Ha/Bohr, and equal −∂E/∂R.

**Use.** These are code-to-code acceptance data for
`tests/scientific/test_reference_dftk_h2.py`. The tolerances are fixed in the
theory post: 1e-7 Ha per energy term, 1e-6 Ha for eigenvalues and 1e-6 Ha/Bohr
for forces.
