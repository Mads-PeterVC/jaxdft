import os

import hypothesis

# Import the package first so the x64 configuration is active for every test.
import jaxdft  # noqa: F401

# JAX traces and compiles on first call, so per-example timing is meaningless.
hypothesis.settings.register_profile("default", deadline=None, max_examples=50)
hypothesis.settings.register_profile("ci", deadline=None, max_examples=200)
hypothesis.settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
