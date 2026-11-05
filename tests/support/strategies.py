"""Hypothesis strategies for lattices, k-points and cutoffs."""

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from support.free_electron import A_SI, fcc_vectors


@st.composite
def cells(draw, scale_min=6.0, scale_max=9.0, max_condition=4.0):
    """Skewed, well-conditioned cells (rows are lattice vectors), in Bohr."""
    scale = draw(st.floats(scale_min, scale_max))
    noise = draw(hnp.arrays(float, (3, 3), elements=st.floats(-0.45, 0.45)))
    A = scale * (np.eye(3) + noise)
    assume(np.linalg.det(A) > 0)
    assume(np.linalg.cond(A) < max_condition)
    return A


def kfracs(lo=-1.0, hi=1.0):
    """Fractional k-points, deliberately also outside the first zone."""
    return hnp.arrays(float, (3,), elements=st.floats(lo, hi))


def fcc():
    return fcc_vectors(A_SI)
