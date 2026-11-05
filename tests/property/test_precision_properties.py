import jax.numpy as jnp
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from jaxdft.precision import is_double_precision

finite = st.floats(allow_nan=False, allow_infinity=False, width=64)


@given(hnp.arrays(dtype=float, shape=hnp.array_shapes(max_dims=3), elements=finite))
def test_float64_input_round_trips_exactly(x):
    """Converting float64 data to a JAX array must not lose any bits."""
    y = jnp.asarray(x)
    assert y.dtype == jnp.float64
    assert (y == x).all()


@given(st.floats(min_value=1.0, max_value=2.0))
def test_arithmetic_keeps_double_resolution(x):
    """An increment of 1e-12 is invisible in float32 but must survive here."""
    y = jnp.asarray(x)
    assert is_double_precision(y + 1e-12)
    assert (y + 1e-12) - y > 0
