"""Precision checks for physical arrays.

The project convention is that every physical array is double precision
(float64 or complex128). These helpers let tests and debug code assert that
nothing has been silently downcast.
"""

from typing import Any

import jax
import jax.numpy as jnp

DOUBLE_DTYPES = (jnp.dtype(jnp.float64), jnp.dtype(jnp.complex128))


def is_double_precision(x: Any) -> bool:
    """Return whether every inexact array leaf of the pytree ``x`` is 64-bit.

    Integer and boolean leaves (e.g. Miller indices, masks) are ignored, as
    they carry no floating-point precision.
    """
    for leaf in jax.tree_util.tree_leaves(x):
        dtype = jnp.result_type(leaf)
        if jnp.issubdtype(dtype, jnp.inexact) and dtype not in DOUBLE_DTYPES:
            return False
    return True


def assert_double_precision(x: Any) -> None:
    """Raise ``TypeError`` if any inexact array leaf of ``x`` is not 64-bit."""
    for path, leaf in jax.tree_util.tree_leaves_with_path(x):
        dtype = jnp.result_type(leaf)
        if jnp.issubdtype(dtype, jnp.inexact) and dtype not in DOUBLE_DTYPES:
            name = jax.tree_util.keystr(path) or "<root>"
            raise TypeError(f"array {name} has dtype {dtype}; expected float64/complex128")
