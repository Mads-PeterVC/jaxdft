import hypothesis

# Import the package first so the x64 configuration is active for every test.
import jaxdft  # noqa: F401
from support.budget import ACTIVE_PROFILE, PROFILE_SCALES, UNDECORATED_BASE

# JAX traces and compiles on first call, so per-example timing is meaningless.
# Per-test budgets scale through support.budget.examples(); the profile's own
# max_examples covers tests that state no budget.
for _name, _scale in PROFILE_SCALES.items():
    hypothesis.settings.register_profile(
        _name, deadline=None, max_examples=UNDECORATED_BASE * _scale
    )
hypothesis.settings.load_profile(ACTIVE_PROFILE)
