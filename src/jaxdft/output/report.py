"""Plain-text log blocks: pure functions from data to lines of text (engineering log,
'SCF observability'). Units follow the package: Hartree and Bohr."""

from __future__ import annotations

import dataclasses
import functools
import math
from pathlib import Path

import numpy as np

from jaxdft.basis.planewave import min_fft_shape, rule_fft_shape

RULE = "=" * 88


def section(title: str) -> list[str]:
    return ["", f"== {title} " + "=" * max(0, 85 - len(title))]


def key_values(pairs) -> list[str]:
    """Aligned ``key : value`` lines."""
    pairs = list(pairs)
    width = max((len(k) for k, _ in pairs), default=0)
    return [f"  {k:<{width}} : {v}" for k, v in pairs]


def describe(obj, depth: int = 0) -> str:
    """A short, stable, human-readable description of a setting: dataclasses by their
    fields, terms with ``pseudos`` by species, functions by qualified name, arrays by
    shape and dtype. Never a memory address."""
    if depth > 4:
        return type(obj).__name__
    if obj is None or isinstance(obj, (bool, int, str)):
        if isinstance(obj, str) and ("/" in obj or "\\" in obj):
            return repr(Path(obj).name)
        return repr(obj)
    if isinstance(obj, float):
        return f"{obj:.10g}"
    if isinstance(obj, functools.partial):
        kw = ", ".join(f"{k}={describe(v, depth + 1)}" for k, v in obj.keywords.items())
        return f"{describe(obj.func, depth + 1)}({kw})" if kw else describe(obj.func, depth + 1)
    if isinstance(obj, (tuple, list)):
        inner = ", ".join(describe(v, depth + 1) for v in obj)
        return f"({inner})" if isinstance(obj, tuple) else f"[{inner}]"
    if isinstance(obj, dict):
        return "{" + ", ".join(f"{k}: {describe(v, depth + 1)}" for k, v in obj.items()) + "}"
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        if getattr(obj, "size", 2) == 1:
            return describe(np.asarray(obj).item(), depth + 1)
        return f"array{tuple(obj.shape)} {obj.dtype}"
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = [f for f in dataclasses.fields(obj) if f.repr and not f.name.startswith("_")]
        inner = ", ".join(f"{f.name}={describe(getattr(obj, f.name), depth + 1)}" for f in fields)
        return f"{type(obj).__name__}({inner})"
    if isinstance(getattr(obj, "pseudos", None), dict):
        return f"{type(obj).__name__}({describe(obj.pseudos, depth + 1)})"
    if callable(obj) and hasattr(obj, "__qualname__"):
        return f"{getattr(obj, '__module__', '?')}.{obj.__qualname__}"
    if type(obj).__repr__ is object.__repr__:
        return type(obj).__name__
    return repr(obj)


# --- setup blocks ----------------------------------------------------------------------


def machine_lines(info: dict) -> list[str]:
    title = f"jaxdft {info['jaxdft']}" + (f" (git {info['git']})" if info.get("git") else "")
    lines = [RULE, f"{title:<60}{info['time']:>28}", RULE]
    lines += section("Machine")
    devices = ", ".join(f"{n} x {k}" for k, n in info["devices"].items())
    lines += key_values(
        [
            ("host", info["host"]),
            ("os", info["os"]),
            ("python", info["python"]),
            ("jax / jaxlib", f"{info['jax']} / {info['jaxlib']}"),
            ("backend", info["backend"]),
            ("devices", devices),
            ("process", info["processes"]),
            ("cpu count", info["cpu_count"]),
            ("float64 enabled", info["x64"]),
        ]
        + [(f"${k}", v) for k, v in info["environment"].items()]
    )
    return lines


def structure_lines(structure) -> list[str]:
    A = np.asarray(structure.lattice.vectors)
    lines = section("Structure")
    lines += ["  lattice vectors (Bohr, rows)"]
    lines += [f"    a{i + 1} = [{a[0]:12.6f} {a[1]:12.6f} {a[2]:12.6f}]" for i, a in enumerate(A)]
    lines += [f"  volume = {float(structure.lattice.volume):.6f} Bohr^3", ""]
    lines += [f"  {'#':>3} {'species':<8}{'fractional':^30}  {'Cartesian (Bohr)':^39} {'weight':>7}"]
    frac = np.asarray(structure.positions)
    cart = np.asarray(structure.cartesian_positions)
    w = np.asarray(structure.weights)
    for j, s in enumerate(structure.species):
        f = " ".join(f"{x:9.6f}" for x in frac[j])
        c = " ".join(f"{x:12.6f}" for x in cart[j])
        lines.append(f"  {j:>3} {s:<8}{f}   {c} {w[j]:7.4f}")
    return lines


def model_lines(terms) -> list[str]:
    lines = section("Model (energy terms)")
    return lines + [f"  {i}. {describe(t)}" for i, t in enumerate(terms, 1)]


def basis_lines(structure, basis) -> list[str]:
    n_pw = np.asarray(basis.n_pw).reshape(-1)
    k = np.asarray(basis.kpoints).reshape(-1, 3)
    rule = rule_fft_shape(structure.lattice, basis.ecut)
    bound = min_fft_shape(structure.lattice, basis.ecut)
    shape = tuple(basis.fft_shape)
    grid_note = "rule" if shape == rule else ("ABOVE rule" if all(map(int.__ge__, shape, rule)) else "BELOW rule")
    lines = section("Basis")
    lines += key_values(
        [
            ("Ecut", f"{basis.ecut:g} Ha"),
            ("k-points", f"{len(k)} (fractional, uniform weights)"),
            ("plane waves per k", f"min {n_pw.min()}, max {n_pw.max()} (padded to {basis.n_pad})"),
            ("FFT grid", f"{'x'.join(map(str, shape))} ({grid_note}; rule {rule}, no-wraparound bound {bound})"),
        ]
    )
    lines += [f"    k{i:<3} [{x[0]:9.6f} {x[1]:9.6f} {x[2]:9.6f}]  N_PW = {n}" for i, (x, n) in enumerate(zip(k, n_pw))]
    return lines


def settings_lines(options, n_el: float, n_bands: int, eig_tol: float, initial: str, n_el_initial: float) -> list[str]:
    lines = section("Electrons and SCF settings")
    lines += key_values(
        [
            ("electrons", f"{n_el:g} (spin-unpolarised, aufbau)"),
            ("bands", n_bands),
            ("converged quantity", "|F(rho) - rho| = (int (F(rho) - rho)^2 dr)^(1/2)"),
            ("tolerance", f"{options.tol:.1e}"),
            ("max iterations", options.max_iter),
            ("mixer", describe(options.mixer)),
            ("eigensolver", describe(options.eigensolver)),
            ("eigensolver tolerance", f"{eig_tol:.1e} Ha (max |H psi - eps psi|)"),
            ("preconditioner", describe(options.preconditioner)),
            ("min frontier gap", f"{options.min_gap:g} Ha"),
            ("initial density", f"{initial} (integrates to {n_el_initial:.12f})"),
        ]
    )
    return lines


# --- iterations ----------------------------------------------------------------------------

ITERATION_HEADER = (
    f"{'iter':>4}  {'E (Ha)':>16}  {'dE (Ha)':>10}  {'|F(rho)-rho|':>12}  {'proj':>8}  "
    f"{'HOMO (Ha)':>10}  {'gap (Ha)':>9}  {'eig res':>8}  {'eig it':>6}  {'t (s)':>7}"
)


def iteration_lines_header(tol: float) -> list[str]:
    lines = section("SCF iterations")
    lines += [
        f"  converged when |F(rho)-rho| < {tol:.1e}. proj: change made by clipping/renormalising",
        "  the mixed density. eig res/it: worst eigensolver residual and iterations over k.",
        "  E is the Kohn-Sham energy of the output orbitals; t is wall time since the start.",
        "",
        ITERATION_HEADER,
    ]
    return lines


def _f(x, fmt, width):
    return f"{'-':>{width}}" if x is None or not math.isfinite(x) else f"{x:{width}{fmt}}"


def iteration_line(i: int, values: dict, elapsed=None) -> str:
    """One row: ``values`` holds energy, delta_energy, residual, projection, homo, gap,
    eig_residual_max and eig_iters (floats; NaN prints as '-')."""
    e = float(values["energy"])
    de = float(values["delta_energy"])
    return (
        f"{int(i):>4}  {_f(e, '.10f', 16)}  {_f(de, '.2e', 10)}  {_f(float(values['residual']), '.3e', 12)}  "
        f"{_f(float(values['projection']), '.1e', 8)}  {_f(float(values['homo']), '.6f', 10)}  "
        f"{_f(float(values['gap']), '.6f', 9)}  {_f(float(values['eig_residual_max']), '.1e', 8)}  "
        f"{_f(float(values['eig_iters']), '.0f', 6)}  {_f(elapsed, '.2f', 7)}"
    )


# --- result -------------------------------------------------------------------------------


def result_lines(
    status: str,
    energies: dict | None,
    total: float | None,
    eigenvalues,
    occupations,
    n_el_density: float | None,
    timings: dict,
    peak_memory: int | None,
) -> list[str]:
    lines = section("Result")
    lines += [f"  {status}"]
    if energies is not None:
        lines += ["", "  energy terms (Ha)"]
        width = max(len(k) for k in energies)
        lines += [f"    {k:<{width}}  {float(v):+.12f}" for k, v in energies.items()]
        lines += [f"    {'total':<{width}}  {float(total):+.12f}"]
    if eigenvalues is not None:
        eps, occ = np.asarray(eigenvalues), np.asarray(occupations)
        lines += ["", "  eigenvalues (Ha) [occupation]"]
        for ik in range(eps.shape[0]):
            row = "  ".join(f"{e:+.6f} [{f:.3g}]" for e, f in zip(eps[ik], occ[ik]))
            lines.append(f"    k{ik:<3} {row}")
    if n_el_density is not None:
        lines += ["", f"  int rho dr = {n_el_density:.12f}"]
    lines += section("Timing")
    lines += key_values([(k, f"{v:.3f} s") for k, v in timings.items()])
    if peak_memory is not None:
        lines += key_values([("peak device memory", f"{peak_memory / 2**20:.1f} MiB")])
    lines += ["", RULE]
    return lines
