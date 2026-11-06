"""Mixing schemes for the fixed-point solver (theory post 2a, 'Self-consistency as a
fixed point'): mixing belongs to the solver, not to the SCF map."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Anderson:
    """Anderson acceleration (Pulay/DIIS mixing) over the last ``depth`` residual
    differences, with damping ``beta``. ``depth=0`` is simple linear mixing,
    x <- x + beta (F(x) - x)."""

    depth: int = 6
    beta: float = 0.7

    def __post_init__(self):
        if int(self.depth) != self.depth or self.depth < 0:
            raise ValueError(f"depth must be a non-negative integer, got {self.depth}")
        if not 0 < self.beta <= 1:
            raise ValueError(f"beta must lie in (0, 1], got {self.beta}")
