"""jaxdft: a GPU-first plane-wave density functional theory code in JAX."""

# Must be imported before anything else creates a JAX array.
from jaxdft import _config  # noqa: F401
from jaxdft.precision import assert_double_precision, is_double_precision

__all__ = ["assert_double_precision", "is_double_precision"]
