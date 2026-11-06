# libxc reference: lda_x and lda_c_pw (Milestone 2a)

`libxc_lda.json` holds libxc's spin-unpolarised `lda_x`, `lda_c_pw` and, as a
negative control, `lda_c_pw_mod`. For each it stores the energy per particle (`zk`, Ha) and the potential
(`vrho` = ∂(ρ ε)/∂ρ, Ha) on 131 densities, logarithmically spaced from 1e-10 to
1e3 Bohr⁻³. It is produced by `run_libxc.jl`, from the repository root:

```
~/.juliaup/bin/julia --project=reference/dftk_h2 reference/libxc_lda/run_libxc.jl
```

It uses the pinned Julia project of the DFTK reference, which includes
Libxc.jl. The JSON records the libxc version: 7.0.0 when this was written.

**Use.** `tests/scientific/test_reference_libxc.py` compares these values
pointwise with jaxdft's `lda_x` and `lda_pw92`, to 1e-10 relative. The tolerance
was set when the data was generated; the theory post did not fix it. The
`lda_c_pw_mod` control must fail that comparison by orders of magnitude, which
shows that the test can tell the parameter sets apart.
`tests/support/tolerances.py` gives the reasoning.
