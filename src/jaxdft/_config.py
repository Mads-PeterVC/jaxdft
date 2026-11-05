"""Process-wide JAX configuration.

This is the single place where JAX numerical configuration is set. It is
imported first by ``jaxdft/__init__.py``, so every entry point that imports
anything from ``jaxdft`` runs it before any array is created.
"""

import jax

# Double precision is a project-wide convention: physical arrays are float64
# or complex128. JAX defaults to 32-bit, so this must run before any array is
# created.
jax.config.update("jax_enable_x64", True)
