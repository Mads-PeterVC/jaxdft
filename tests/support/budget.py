"""Hypothesis example budgets, scaled by the active profile.

Each property test states a *base* example count that reflects its cost (most
of which is JAX compiling for each new array shape). The profile selected by
the ``HYPOTHESIS_PROFILE`` environment variable multiplies every base count, so
a deeper run tests proportionally more examples everywhere:

    @settings(max_examples=examples(10))

Tests without an explicit count get ``UNDECORATED_BASE`` times the factor.
A literal ``max_examples=<int>`` would bypass the profile and is rejected by
``tests/unit/test_hypothesis_budget.py``.
"""

import os

PROFILE_SCALES = {"default": 1, "ci": 5, "deep": 20}
UNDECORATED_BASE = 50
PROFILE_ENV = "HYPOTHESIS_PROFILE"


def profile_scale(name: str) -> int:
    """Scale factor for a named profile; an unknown name is an error."""
    try:
        return PROFILE_SCALES[name]
    except KeyError:
        raise ValueError(
            f"unknown {PROFILE_ENV}={name!r}; expected one of {sorted(PROFILE_SCALES)}"
        ) from None


ACTIVE_PROFILE = os.environ.get(PROFILE_ENV, "default")
SCALE = profile_scale(ACTIVE_PROFILE)


def examples(base: int) -> int:
    """Example count for a test whose base budget is ``base``."""
    return base * SCALE
