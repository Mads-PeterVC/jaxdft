import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdft
from jaxdft.precision import assert_double_precision, is_double_precision


def test_x64_enabled_after_import():
    assert jax.config.jax_enable_x64


def test_default_dtypes_are_double():
    assert jnp.zeros(3).dtype == jnp.float64
    assert jnp.array(1.0).dtype == jnp.float64
    assert (jnp.array(1.0) * 1j).dtype == jnp.complex128
    assert jnp.fft.fft(jnp.ones(4)).dtype == jnp.complex128
    assert jnp.linalg.eigh(jnp.eye(2))[0].dtype == jnp.float64


def test_double_precision_accepted():
    tree = {"a": jnp.ones(3), "b": [jnp.ones(2, dtype=jnp.complex128)], "n": jnp.arange(3)}
    assert is_double_precision(tree)
    assert_double_precision(tree)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16, jnp.complex64])
def test_single_precision_rejected(dtype):
    tree = {"ok": jnp.ones(3), "bad": jnp.ones(3, dtype=dtype)}
    assert not is_double_precision(tree)
    with pytest.raises(TypeError, match="bad"):
        assert_double_precision(tree)


def test_numpy_float32_rejected():
    assert not is_double_precision(np.ones(2, dtype=np.float32))


def test_public_api():
    assert jaxdft.assert_double_precision is assert_double_precision


def test_x64_enabled_in_fresh_process():
    """conftest imports jaxdft for every test; check the package alone suffices."""
    import subprocess
    import sys

    code = "import jaxdft, jax.numpy as jnp; print(jnp.zeros(1).dtype)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "float64"
