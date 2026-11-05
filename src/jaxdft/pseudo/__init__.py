"""Pseudopotentials.

Protocol (CLAUDE.md, theory post 1b). A pseudopotential exposes reciprocal-space radial
data; nothing outside its parser/constructor reads file formats.

- ``local(q) -> v_hat(q)``: atomic Fourier transform of the local part,
  v_hat(q) = int v(r) exp(-i q.r) d^3r, in Ha Bohr^3, independent of any cell.
  Must be traceable (jit/vmap/grad).
- ``local_support: float | None``: q (Bohr^-1) beyond which v_hat is exactly zero;
  None if unbounded (needs the FFT-based term of a later milestone).
- optional ``validate(q_values)``: host-side; raise ValueError if the model is not
  defined at some q where it will actually be used.

Nonlocal projectors are added as further methods in the milestone that needs them.
"""
