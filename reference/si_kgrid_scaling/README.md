# Silicon SCF cost against the k-grid, on Slurm clusters

Timings of jaxdft itself, not of a reference code: self-consistent LDA silicon
on Γ-centred n³ k-grids, n = 1 … 8, one JSON per cluster. They are read by
`docs/engineering/performance.qmd`.

| file | cluster | hardware per job (shephex profile) |
|---|---|---|
| `biocloud.json` | biocloud | 8 CPU cores, 16 GB (`biocloud-jaxdft-cpu`, `JAX_PLATFORMS=cpu`) |
| `aicloud.json` | aicloud | 1 V100 32 GB, 8 CPU cores (`aicloud-jaxdft-v100`, `JAX_PLATFORMS=cuda,cpu`) |

**How it was made.**

1. `shephex/scripts/<cluster>/si_kgrid_scaling.py` submits one experiment per
   grid with the cluster's shephex profile, into the venv of a jaxdft clone
   on the cluster (synced with the `cuda` extra on aicloud).
2. `uv run hex remote pull shephex/data/<cluster>/si-kgrid-scaling` brings
   the experiments back; that directory is not tracked.
3. `uv run python reference/si_kgrid_scaling/extract.py <cluster>` (from the
   repository root) writes `<cluster>.json` from each experiment's shephex
   context and jaxdft log.

**The calculation** is the one of the Milestone 2b performance table: diamond
silicon, a = 10.263 Bohr, HGH LDA with nonlocal projectors, E_cut = 12 Ha,
default `SCFOptions`, no symmetry reduction of the grid. Each experiment runs
the SCF twice in one process: `first_s` includes compilation, `steady_s` is
the repeat.

**Fields per experiment.** The grid (`n`, `n_kpoints`), basis size
(`n_pw_max`, `fft_shape`), result (`energy` in Ha, `n_iter`), timings in
seconds, `peak_memory_gb` (peak resident memory of the process), the machine
(`node`, `cpu_model`, `cpus_visible` = CPUs Slurm allows the process,
`cpus_node`), the jaxdft `commit` on the cluster and the Slurm `job`.
`eig_iterations_first` and `eig_iterations_steady` are the log's `eig it`
column: per SCF step, the largest number of LOBPCG iterations over the
k-points. GPU runs add `devices`, `backend` and `peak_device_memory_gb`
(peak bytes in use on the GPU).

**Limits.** One measurement per grid and cluster. The jobs of a cluster ran
at the same time: on biocloud on three nodes (the three largest grids shared
one, so they may have competed for memory bandwidth), on aicloud on two nodes
with one GPU each. Timings are comparable within a file and between the two
files as "this GPU against these 8 cores", not with the laptop numbers
elsewhere in `performance.qmd`.
