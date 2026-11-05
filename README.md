# jaxdft

A GPU-first plane-wave density functional theory code written in JAX.
See `CLAUDE.md` for goals, conventions and the milestone workflow.

```sh
uv sync                                   # environment (dev + docs groups)
uv run pytest                             # unit, property and scientific tests
uv run quarto render docs/                # documentation site -> docs/_site
```
