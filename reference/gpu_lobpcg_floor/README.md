# V100: why LOBPCG stalls near 1e-12

On an aicloud V100 every test that runs the SCF at `SCF_TOL_TIGHT` = 1e-12
fails (see `reference/aicloud_pytest/`): the eigensolver must reach residuals
of 1e-13 and stops near 1.5e-12, while the CPU reaches 3e-14. Two diagnosis
runs locate the cause. Both ran in one process with `JAX_PLATFORMS=cuda,cpu`,
comparing the GPU with the CPU of the same node, on the H2 test size (Γ,
10 Ha) and silicon at one non-Γ k-point of the 2³ grid (12 Ha), each with
the Hamiltonian at a density converged on the CPU and the SCF's tight
eigensolver tolerance, 5e-14.

`gpu_lobpcg_floor.json` holds both results (the shephex contexts), made by
`extract.py` from the pulled experiments:

- **`building_blocks`** (`shephex/scripts/aicloud/gpu_residual_floor.py`):
  FFT round trip, `apply_H` on the GPU against the CPU, `eigh` accuracy at
  Rayleigh–Ritz sizes, and jaxdft's `lobpcg` with growing iteration caps,
  with the GPU orbitals re-checked on the CPU.
- **`dense_steps`** (`shephex/scripts/aicloud/gpu_lobpcg_dense_steps.py`):
  a diagnostic copy of `lobpcg` on the GPU with one change at a time: the
  Rayleigh–Ritz `eigh`, the Gram `eigh`, or the QR and inverse moved to the
  CPU by host callback, all three moved, or the penalty on dropped
  directions lowered from 1e3 to 2 times max|T|. The unmodified copy
  reproduces jaxdft's `lobpcg` on both devices.

Both ran on jaxdft commit `acd8ff8` (the aicloud clone); the scripts did not
record it themselves.

**Limits.** One run each, one GPU model, two small systems, the default
preconditioner and a fixed random start. GPU results vary slightly between
runs (the stalled residual for silicon was 4.2e-12 in one run and 2.9e-12 in
the other).

## Third run: per-iteration trajectories (2026-11-12)

The two runs above compared variants by their end state, one run each. Three
explanations of the failures then stood side by side (theory post 2c), and
the end state could not separate them. `lobpcg_trajectory.json` holds a
per-iteration record from an instrumented copy of `lobpcg`
(`shephex/scripts/aicloud/lobpcg_trajectory.py`, job `1167888`, jaxdft
`3383ca3` on the aicloud clone): the largest residual, the directions dropped
in each orthonormalisation pass, max|T| with the penalty, the backward error
of the Rayleigh–Ritz `eigh` for the selected pairs, and the penalty when one
is applied.

Same two problems and tolerance as above. Variants: today's penalty
(`jaxdft`), the interlacing bound of the 2c post (`bound`), and today's
penalty with the Rayleigh–Ritz `eigh` on the host (`rr_host`, GPU only).
Each from three random starts (`seed` 0, 1, 2), stopping at the tolerance
(`stop`) and forced to 300 iterations (`forced`). `runs` is the V100 node
(GPU and its CPU); `laptop_runs` is the same script on the laptop, CPU only,
run with `--local`, which reproduced its result exactly when repeated.

**Limits.** One job, one GPU model, two small systems. The script's
`gram_min_rel` column is not stored: it pairs Gram eigenvalues with column
norms by index, which does not identify the dropped columns.
