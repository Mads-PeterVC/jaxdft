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
- ``ionic_charge``: the valence charge Z of the point ion (None for empirical models
  without one), used by the Ewald term and for the electron count.
- ``alpha``: lim_{q->0} [v_hat(q) + 4 pi Z / q^2], the finite G = 0 limit (Ha Bohr^3).

- optional ``channels``: a tuple of nonlocal projector channels (theory post 2b), one
  per angular momentum, each with ``l``, the symmetric coupling matrix ``h`` (Ha) and
  ``projectors(q2) -> (n_i, *q2.shape)``, the reduced projectors p_hat_i^l(q) / q^l in
  Bohr^(3/2 + l), cell-independent and traceable, as functions of q^2 (never of q).
  Absent or empty: no projectors.
"""
