# aicloud: the test suite on a V100

`pytest_gpu.json` records a run of the full jaxdft test suite on one NVIDIA
V100 (aicloud), with JAX's default device on the GPU. Nothing in the suite is
GPU-specific, so the run checks that what passes on CPU also passes there.

**How it was made.**

1. The aicloud clone of jaxdft is synced with every default group and the
   `cuda` extra (`uv sync --extra cuda`), because the tests need the `dev`
   and `docs` groups.
2. `shephex/scripts/aicloud/pytest_on_gpu.py` submits one job with the
   shephex profile `aicloud-jaxdft-v100` (`gpu:v100:1`, 8 CPUs, 32 GB,
   `JAX_PLATFORMS=cuda,cpu`; the CPU platform is needed by the
   `jax.debug.callback` of the SCF log). The job runs `pytest -ra` in the
   clone with GPU memory preallocation off and stores the full output.
3. `uv run hex remote pull shephex/data/aicloud/pytest-gpu` brings it back;
   `uv run python reference/aicloud_pytest/extract.py` writes the JSON.

**Fields per run.** The jaxdft `commit`, the `devices`, the pytest
`summary` line and its `counts`, `elapsed_s`, and `problems`: every failed or
erroring test with the last exception line of its traceback.

**Limits.** One run, on one GPU model. The pytest output itself stays in the
untracked experiment directory; the JSON keeps only the outcome and the
error of each failing test.
