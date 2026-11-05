"""The Hypothesis profile must control every property test's example count."""

import ast
from pathlib import Path

import hypothesis
import pytest

from support.budget import (
    ACTIVE_PROFILE,
    PROFILE_SCALES,
    UNDECORATED_BASE,
    examples,
    profile_scale,
)

TESTS_DIR = Path(__file__).resolve().parents[1]


def _literal_max_examples():
    for path in sorted(TESTS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "max_examples":
                if isinstance(node.value, ast.Constant):
                    yield f"{path.relative_to(TESTS_DIR)}:{node.value.lineno}"


def test_no_literal_max_examples():
    # A literal count overrides the profile, so a deeper run would test nothing more.
    assert list(_literal_max_examples()) == []


def test_examples_scale_with_active_profile():
    scale = PROFILE_SCALES[ACTIVE_PROFILE]
    assert examples(10) == 10 * scale
    assert hypothesis.settings().max_examples == UNDECORATED_BASE * scale


def test_profile_scales_are_ordered():
    assert PROFILE_SCALES["default"] == 1
    assert PROFILE_SCALES["default"] < PROFILE_SCALES["ci"] < PROFILE_SCALES["deep"]


def test_unknown_profile_is_an_error():
    with pytest.raises(ValueError, match="unknown HYPOTHESIS_PROFILE"):
        profile_scale("cl")
