# WP1 — Components and cascade

Role: Implementer · Wave: 1 · Branch: `wave-1/WP1-cascade` · Depends on: WP0 (merged)
Verified by: WP7 (global-ODE reference for lines) · Time box: 5 days (then status review)

## Goal

Turn the GSM core into a usable line simulator: a component protocol, the straight and smooth-bend components,
explicit frame changes between H- and E-plane components, a `Line` object, and frequency sweeps. At the end, the
§7 "sequence of bends" study of the JINST paper is a ten-line script.

## Read first

`CONVENTIONS.md` (all), `src/wrmodes/gsm.py`, `src/wrmodes/modes.py`, `src/wrmodes/transfer_matrix.py`,
`benchmarks/paper_figures.py` (subcommand `sequence`), `tests/test_gsm.py`.

## Specification

1. **`wrmodes/components/`.** A base class `Component` with `gsm(f, basis1=None, basis2=None) -> GSM` and
   `default_basis(f)`, plus a `name`. Components are immutable and hold no frequency-dependent state.
2. **`Guide(w, h, sigma=None)`** holds the global cross-section and, for WP2, the wall conductivity.
3. **`Straight(length)`** wraps `gsm.straight`. Loss is added by WP2; keep a hook for it.
4. **`SmoothBend(length, curvature, plane='H'|'E', breakpoints=None, atol=1e-6)`**:
   * Per family $n$, call `solve_transfer_matrix` with `error_control='full'`; convert with
     `from_transfer_matrix`; combine with `block_diag`. Families with a single propagating mode and no coupling
     are a straight section.
   * **Evanescent modes in the basis** (needed next to miters and steps): propagate them as uncoupled straight
     modes, $e^{-i\beta L}$, and document this approximation. Add a test showing that it is exact in the
     straight limit, and that a bend's result for the propagating modes does not depend on whether evanescent
     modes are included.
5. **`FrameChange(from_plane, to_plane)`**:
   * Lossless, reflectionless; maps the local basis of one plane to the other (`CONVENTIONS.md` §2).
   * Its matrix is the overlap of `wrmodes.modes.transverse_field` on a quadrature grid, rounded to the exact
     signed permutation after checking it is one to 1e-12.
6. **`Line(guide, components, plane='H')`**:
   * Inserts the frame changes automatically between components of different planes, and records them in the
     line's description, so they are never hidden.
   * `line.gsm(f, cutoff_factor=1.0)` returns the cascade in the global frame.
   * `line.sweep(freqs, input_mode, processes=None)` runs frequencies in parallel processes, keeping one BLAS
     thread per process (`blas_threads=1`, as in 0.1). It returns an object with `freqs`, `loss_db` (input to the
     same mode), `converted_power[mode]`, `reflected_power` and the GSMs when requested.
7. **Bases.** A line chooses one global basis per frequency (all families of the propagating set, plus evanescent
   modes up to `cutoff_factor`). Components return GSMs in exactly that basis, mapped to their local frame.

## Acceptance tests

| Check | Target |
| --- | --- |
| Same-plane cascade of two bends and a straight vs product of transfer matrices (0.1 method) | 1e-12 on S |
| Paper §7 sequence: the per-bend losses and the cascade of `benchmarks/paper_figures.py sequence` at 10 frequencies | 1e-10 vs the 0.1 computation |
| H → E → H frame changes compose to identity; `FrameChange` is a signed permutation | exact |
| An E-plane bend in a line equals the 0.1 E-plane result (a = h, b = w) after the frame change | 1e-10 |
| Every component: reciprocity and unitarity (lossless) | 1e-10 |
| `sweep` with 1 and 4 processes | identical results |
| WP7's global-ODE reference for three lines (same plane, mixed planes, with straights of 10 m) | 1e-6 of input power for every output mode |

## Deliverables

* `src/wrmodes/components/` and `src/wrmodes/line.py`, with tests in `tests/test_components.py` and
  `tests/test_line.py`.
* `examples/cascade_sequence.py`.
* `docs/reports/WP1.md`.

## Out of scope

Ohmic loss (WP2), miters (WP3/WP4), tapers and twists (wave 2), plotting.
