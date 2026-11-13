# Milestone 2c: the eigensolver on the clusters

Records of jaxdft itself, not of a reference code: the cluster runs the 2c
theory post asks for ('On the clusters'). They are read by the 2c validation
document. The k-grid reruns are in `reference/si_kgrid_scaling/` (`*_2c.json`)
and the V100 test suite in `reference/aicloud_pytest/`.

| file | runs | script |
|---|---|---|
| `lobpcg_check.json` | jaxdft's `lobpcg` as the SCF calls it, on the H₂ (Γ, 10 Ha) and silicon (k-point 1 of 2³, 12 Ha) problems of `reference/gpu_lobpcg_floor/`, on a V100 and the CPU of its node, three random starts, at the per-k floor and at an unreachable tolerance | `shephex/scripts/aicloud/lobpcg_check_2c.py` |
| `settings_v100.json` | silicon 8³ at 12 Ha on a V100: `buffer_bands` 0–3 at one batch, and k-chunks of 64, 128, 256 at one buffer band | `shephex/scripts/aicloud/si_eigensolver_settings_scan.py` |
| `k_chunks.json` | silicon 8³ at 12 Ha with three buffer bands: one batch and k-chunks of 64, 128, 256, on a V100 and on 8 biocloud cores | `shephex/scripts/<cluster>/si_k_chunk_scan.py` |

**How it was made.** The scripts were submitted with shephex from the laptop,
into clones of jaxdft at commit `fe43ee5` on branch `milestone-2c` (each
experiment records the commit). `uv run hex remote pull
shephex/data/<cluster>/<directory>` brings them back, and `uv run python
reference/eigensolver_2c/extract.py` writes the JSON files from the shephex
contexts and options. A `k_batch_size` of 0 means one batch (shephex options
cannot be `None`).

**Fields.** Each experiment carries its options and the shephex context:
`commit`, `devices`/`backend`, timings (`first_s` with compilation,
`steady_s` a repeat in the same process), `n_iter`, `energy` (Ha), the sums
over the SCF history of the first run (`sum_eig_iters`: per step the largest
LOBPCG count over k-points; `sum_eig_iters_mean`: the mean), `refined_steps`,
`eig_drops_max`, and peak memory (`peak_device_memory_gb` on the GPU,
`peak_memory_gb` as process resident memory in the chunk scans).
`lobpcg_check.json` holds one experiment whose `runs` map
`system/device/mode/seed` to iterations, the stagnation flag, late drops, the
true residual of the needed bands, the floor, and whether that residual is
within the SCF's `eig_ok` check (2 × floor).

**Limits.** One measurement per setting, one GPU model, 8 CPU cores of a
shared node; settings of a scan ran at the same time on different nodes.
