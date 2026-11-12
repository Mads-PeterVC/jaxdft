# biocloud: silicon SCF cost against the k-grid

`si_kgrid_scaling.json` holds timings of jaxdft itself, not of a reference
code: self-consistent LDA silicon on Γ-centred n³ k-grids, n = 1 … 8, run on
the biocloud Slurm cluster. It is read by `docs/engineering/performance.qmd`.

**How it was made.**

1. `shephex/scripts/biocloud/si_kgrid_scaling.py` submits one experiment per
   grid with the shephex profile `biocloud-jaxdft-cpu` (8 CPUs, 16 GB, one
   task, `JAX_PLATFORMS=cpu`), into the venv of a jaxdft clone on the
   cluster.
2. `uv run hex remote pull shephex/data/biocloud/si-kgrid-scaling` brings the
   experiments back; that directory is not tracked.
3. `uv run python reference/biocloud_si_kgrid/extract.py` (from the
   repository root) writes the JSON from each experiment's shephex context
   and jaxdft log.

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
k-points.

**Limits.** One measurement per grid. The eight jobs ran at the same time,
on three nodes (the three largest grids shared one), so they may have
competed for memory bandwidth. Timings are comparable within this file, not
with the laptop numbers elsewhere in `performance.qmd`.
