# WP1 — Components and cascade: report

Branch `wave-1/WP1-cascade`, from `main` at 223aeeb.

## What was done

| Item | Files |
| --- | --- |
| Component protocol (the shared stub, plus `plane` and the `default_basis(f, cutoff_factor=None)` signature; see Review fixes) | `src/wrmodes/components/base.py`, `src/wrmodes/components/__init__.py` (exports added) |
| `Guide(w, h, sigma=None)`, `map_basis` | `src/wrmodes/components/guide.py` |
| `Straight(length)` | `src/wrmodes/components/straight.py` |
| `SmoothBend(length, curvature, plane, breakpoints, atol)`, picklable `SechCurvature`, `ConstantCurvature` | `src/wrmodes/components/bends.py` |
| `FrameChange(from_plane, to_plane)`, `frame_overlap` | `src/wrmodes/components/frames.py` |
| `Line(guide, components, plane='H')`, `Line.sweep`, `SweepResult` | `src/wrmodes/line.py` |
| Tests | `tests/test_components.py` (20), `tests/test_line.py` (15) |
| Acceptance numbers with the full basis | `benchmarks/wp1_cascade_checks.py` |
| Example: the JINST §7 sequence | `examples/cascade_sequence.py` |

`wrmodes/__init__.py` got one line at the end: `from .line import Line, SweepResult, Guide, Straight, SmoothBend,
FrameChange, SechCurvature, ConstantCurvature`.

### Design

* **Protocol** (as revised after review; see Review fixes). `Component.plane` is required: `'H'` or `'E'` for
  components whose bases are local bases of that frame, `None` for components that the H↔E frame change leaves
  unchanged (straights, frame changes). The base class sets it to `UNDECLARED`, and `Line` raises `TypeError` for
  it. Optional attributes: `guides = (g_in, g_out)` for components that change the cross-section; `guide` and
  `with_guide(guide)` for components that use the line's guide in their `default_basis`.
* **Bases.** `Guide.basis(f, cutoff_factor, plane, evanescent_families='live')` builds one basis per frequency
  in the global (H) frame, then maps it to the frame of `plane` (`map_basis`: a ↔ b, (m, n) → (n, m), tag
  `'E'`). So every component of a line sees the same physical modes. The 'live' rule is described under Review fixes.
* **SmoothBend.** For each local family n, the propagating modes go through
  `solve_transfer_matrix(error_control='full', atol, breakpoints)` and `from_transfer_matrix`. A family with zero
  coupling matrix becomes `straight`; in practice this is TE1,0 alone, below the TE2,0 cutoff. Evanescent modes are
  propagated as uncoupled straight modes, exp(−|β|L), with no reflection. This approximation is documented in the
  docstring: it is exact for κ = 0, and the propagating block does not depend on it. The result is put together
  with `block_diag` in the exact order of the given basis. Each GSM carries `info[n]` (n_slices, error_estimate,
  success).
* **FrameChange.** The matrix is the quadrature overlap ∬ e_to·e_from of `transverse_field`, on a Gauss–Legendre
  grid of 2·max(m, n) + 32 points per side. It is checked to be a signed permutation to 1e-12, rounded, and then
  checked against `modes.eplane_frame_signs` (H→E) or its transpose (E→H). S11 = S22 = 0, S21 = F, S12 = Fᵀ. Port
  bases carry the frame tags `'H'` / `'E'`.
* **Line.** `plane` is the frame of the line's two ports; `'H'` (the default) is the global frame. A straight takes
  the frame of the element before it, so a straight between two E-plane bends needs no frame change. Explicit
  `FrameChange` elements are accepted and checked. Inserted frame changes appear in `line.elements`, are flagged in
  `line.inserted`, and are marked "(inserted by Line)" in `line.describe()`. A `Line` is itself a component;
  nested lines are flattened into the outer one. Components that change the cross-section are supported through
  `guides` (see Review fixes).
* **Sweep.** `processes=None` means `os.cpu_count()`, capped at the number of frequencies; `1` runs in-process.
  Every process limits BLAS to one thread (`_blas_limit(1)`, and `blas_threads=1` in the solver), so the results
  do not depend on the number of processes. Workers are started with `spawn`, and the line is pickled once per
  worker (initializer). If the line cannot be pickled (e.g. a lambda curvature) and `fork` exists, `fork` is used.
  Otherwise a `TypeError` says what to do. `SweepResult` has `freqs`, `loss_db` (input → same mode),
  `converted_power[mode]` (every other propagating output mode, NaN where it does not propagate; keys accept
  `Mode`, names or tuples), `reflected_power` (total over the propagating modes of port 1), and `gsms` when
  `keep_gsm=True`.
* **Wall-loss hook (WP2).** `Guide(w, h, sigma)` accepts a conductivity. `Straight._loss_matrix(basis, f)` and
  `SmoothBend._loss_matrix(basis, f)` call `Guide.require_lossless`, which raises `NotImplementedError` when
  `sigma` is not None, so `Line.gsm` raises too. `FrameChange` stays valid in a lossy guide (zero length).

## Results

All numbers come from `benchmarks/wp1_cascade_checks.py` (full run) and from the tests. WR-90, w = 22.86 mm,
h = 10.16 mm.

* Same-plane cascade (bend 0.15 m with κ = 1/0.25 m⁻¹, 0.7 m straight, bend 0.12 m with κ = −1/0.3 m⁻¹) at
  40 GHz, all 26 propagating modes, against the product of the full transfer matrices: max |ΔS| = **5.0e-16**.
* JINST §7 sequence (sech 60°/0.6 m, 10 m, sech 90°/0.4 m, 1 m, sech −90°/0.4 m) at 10 frequencies, equally
  spaced from 18 to 90 GHz, against the product of the full 2N×2N Φ of bends and straights:

  | Plane | Basis | Modes | max \|ΔS\| vs Φ product | max \|ΔS21\| vs T-only product | max \|S11\| of the line | max \|R\| of a single bend (input family) | Reciprocity / unitarity |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | H | input family (n = 0) | 2–13 | 8.7e-16 | 4.6e-7 | 2.7e-4 | 6.8e-4 | 1.1e-13 / 2.3e-13 |
  | H | all families | 5–129 | 2.7e-12 | 4.6e-7 | 6.5e-2 | — | 2.3e-13 / 6.2e-13 |
  | E | input family (global m = 1) | 3–13 | 7.4e-16 | 6.2e-8 | 4.0e-4 | 2.4e-4 | 9.0e-14 / 1.7e-13 |
  | E | all families | 5–129 | 1.0e-12 | 6.2e-8 | 1.8e-2 | — | 1.2e-13 / 3.1e-13 |

  The T-only product of `benchmarks/paper_figures.py sequence` ignores the reflections between the bends. It
  differs from the full cascade by at most 4.6e-7 (H) and 6.2e-8 (E) on S21 within the input family. That is of
  the order of the products of reflection amplitudes, |R|² ≈ (2.4–6.8e-4)² ≈ 1e-7–5e-7. The reviewer's 2.8e-7 was
  measured on another set of frequencies. In the full basis, the large |S11| (6.5e-2) comes from modes just above
  cutoff in other families: for the 90° bend alone, at 66 GHz, TM9,2 has f_c/f = 0.9997 and |S11| = 0.033 (H plane);
  TM2,9 gives 0.0096 (E plane). These modes do not reach the input family in a coplanar line.
* E-plane bend (sech 90°/0.4 m) in a line, mapped back to the local frame, against
  `transmission_matrix(f, h, w, n, …)` of wrmodes 0.1, every family: max |ΔS| = 2.0e-14 (25 GHz, 10 modes),
  1.5e-13 (40 GHz, 26 modes), 4.3e-15 (60 GHz, 57 modes).
* Mixed-plane line (H 60°, 0.5 m, E 90°, 2 m, E −45°, H −90°; 2 frame changes inserted) at 33 GHz, against the
  Φ product in the global frame (Φ_global = diag(Fᵀ, Fᵀ) Φ_local diag(F, F)): < 1e-10 (test). TE1,0 → TE1,1
  coupling is non-zero (> 1e-3), so the families really mix.
* FrameChange: equal to `eplane_frame_signs` exactly, for bases with evanescent modes too (cutoff_factor 1.7).
  The quadrature overlap deviates from the rounded permutation by < 1e-12. H→E followed by E→H gives exactly the
  identity (`np.array_equal`), also inside a `Line`.
* Evanescent modes: a bend with κ = 0 and cutoff_factor = 1.6 equals `straight` to 1e-12. With cutoff_factor = 1.6,
  the propagating blocks of a bend are bit-identical to those of the propagating-only bend, in both planes. A mixed
  line with cutoff_factor 1.5 has the same propagating block as with 1.0, to 1e-13.
* Sweep with 1 and 4 processes (5 frequencies, a mixed H/E line): bit-identical loss, converted and reflected
  powers, and GSMs. Power balance transmitted + converted + reflected = 1 to 1e-10.
* Test suite after the review fixes: 141 tests pass (93 from WP0, 48 new: 27 in `test_components.py`, 21 in
  `test_line.py`). Under the load of the other agents, the full suite took 354 s.
* Runtime: a line with all families is costly at high frequency, because `error_control='full'` needs ≥ 1024
  slices per family. One sech 90° bend with all 129 modes at 90 GHz takes 6.2 s (7 families × 2048 slices,
  error estimates ≈ 5e-11). The full-basis sequence took 194 s (H) and 133 s (E) for 10 frequencies with the
  reference included (serial).

## Acceptance tests

| Check | Target | Result | Value |
| --- | --- | --- | --- |
| Same-plane cascade of two bends and a straight vs product of transfer matrices | 1e-12 on S | pass | 5.0e-16 (40 GHz, 26 modes) |
| §7 sequence at 10 frequencies vs product of full 2N×2N Φ | 1e-10 | pass | ≤ 8.7e-16 (input family), ≤ 2.7e-12 (all families, up to 129 modes) |
| Same vs T-only product of `paper_figures.py sequence` | about 1e-6, with the size of the reflections | pass | 4.6e-7 (H), 6.2e-8 (E); single-bend \|R\| ≤ 6.8e-4 |
| H→E→H compose to identity; FrameChange is a signed permutation | exact | pass | exact (`array_equal`); quadrature within 1e-12 |
| E-plane bend in a line = 0.1 E-plane result after the frame change | 1e-10 | pass | ≤ 1.5e-13 |
| Every component: reciprocity, unitarity | 1e-10 | pass | ≤ 2.3e-13 / ≤ 6.2e-13 |
| `sweep` with 1 and 4 processes | identical | pass | bit-identical |
| WP7's global-ODE reference for three lines | 1e-6 of input power | not done | WP7's reference is not available yet; the coordinator will run it |

## Open issues

* **WP7 comparison** (last acceptance row) is still to be run by the coordinator. `Line.gsm(f, basis1=...)` takes
  a sub-basis (e.g. one global family), which keeps the global-ODE comparison cheap.
* **Cost of full-basis lines** at high frequency (see Runtime). A future option: restrict a line to the families
  reachable from the input mode. In a coplanar line that is one family; with H- and E-plane bends, the closure of
  (m, n) under both. Not done, because the card asks for all families.
* **Evanescent modes in bends** are uncoupled (documented approximation). Its error next to a miter or step has
  not been quantified. The 2D solver (WP3) could measure it by comparing a short bend in both models.
* On macOS, the `fork` fallback for unpicklable lines may be unsafe with the system BLAS. The default path is
  `spawn` with a picklable line. The fork fallback test runs on Linux only.
* Near-cutoff modes (f_c/f > 0.999) give reflections of a few percent in their own family in the coupled-mode
  model (TM9,2 at 66 GHz). The model is not expected to be accurate there. Users looking at `reflected_power`
  with a full basis should know this.

## Conventions issues

* **§3 evanescent truncation vs the E frame.** "Evanescent modes are only added to families that have a
  propagating mode" depends on the frame. Resolved in the review fixes with the 'live' rule (union of the live
  families of both frames). The coordinator will update §3.
* **Line `plane` argument.** The card gives `Line(guide, components, plane='H')` without defining `plane`. It is
  implemented as the frame of the line's ports ('H' = the global frame of §2). Frame-independent components
  inherit the current frame.
* §2 lists "a twist label" as a possible frame tag. `map_basis` and `FrameChange` handle only 'H' and 'E'
  (the coordinator will drop twist tags from CONVENTIONS).

## Notes for other cards

* Components declare `plane` ('H', 'E' or None) and accept local bases with that frame tag (or no tag). They
  must return GSMs whose `basis1`/`basis2` are exactly the bases they were given. `Line` never calls a
  component's `default_basis`; it always calls `gsm(f, basis1, basis2)` with the line's bases.
* **WP4 Step (PR #4) in a Line.** I read its interface. It needs two additions: `plane` (its `frame`), and
  `guides = (Guide in, Guide out)` in global terms. For `plane='H'` these are `(Guide(a1, b1), Guide(a2, b2))`;
  for `plane='E'` they are `(Guide(b1, a1), Guide(b2, a2))`. The line then hands it `basis2` from its rule in the
  output guide. The line's default `cutoff_factor=1` gives a Step only propagating modes on its ports; as agreed,
  Step solves internally on the union with its own default basis and returns the requested sub-block.
* WP2: the hook is `_loss_matrix(basis, f)`, the full N×N matrix Λ with dA⁺/dz = −(i diag β + Λ)A⁺.
  `Straight.gsm` already uses it: S21 = expm(−(i diag β + Λ)L), S12 = S21ᵀ. A test double checks this. A
  non-zero Λ in `SmoothBend` raises `NotImplementedError`; WP2 adds Λ to the coupled-mode matrix there. The
  "K = 0 → straight" shortcut is only taken when Λ = 0.
* Merge note: `components/base.py` now differs from the stub (the `UNDECLARED` sentinel, `plane = UNDECLARED`, and
  the `default_basis(f, cutoff_factor=None)` signature and docstring). `components/__init__.py` starts with the
  stub line and adds 4 import lines.

## Review fixes

From the independent review of PR #3 and the coordinator's follow-up. No physics errors were found.

1. **Evanescent families: the 'live' rule.** `Guide.basis`, `Line.gsm` and `Line.sweep` take
   `evanescent_families='live'` by default. An evanescent global mode (M, N) is kept if a propagating mode has
   the same N (a live H-frame family) or the same M (a live E-frame family). `'propagating'` (the old H-frame
   rule) and `'all'` remain available, and `Line.sweep` forwards the choice. Tests: at 10 GHz with cutoff_factor 3,
   local E family n = 1 now holds TE0,1, TE1,1 and TM1,1 (only TE0,1 before). In a square guide the H- and
   E-frame bases now hold the same modes (12 and 20 GHz). Local families without a propagating mode are uncoupled
   straight modes in `SmoothBend`. With cutoff_factor = 1 nothing changes: the quick run of
   `benchmarks/wp1_cascade_checks.py` reproduced every number of the first version exactly.
2. **`plane` in the protocol.** `Component.plane = UNDECLARED` in `base.py`. `Line` raises `TypeError` for a
   component that does not declare it, and for any value other than 'H', 'E' or None. `Straight.plane = None`
   and `FrameChange.plane = None`. `default_basis(f, cutoff_factor=None)` now means "the component's own default";
   for WP1's components that default is 1. Line passes its own bases and does not impose a truncation through
   `default_basis`.
3. **Empty bases** below the first cutoff: `SmoothBend.gsm` and `FrameChange` return 0×0 GSMs. New tests: the
   components at 5 GHz, and a mixed-plane sweep from 4 GHz (NaN loss and reflection until TE1,0 propagates, finite
   after).
4. **`Line.gsm(basis1=...)`** checks (a, b) against `guide.dims(plane)` and raises on a mismatch.
5. **Documentation.**
   * The `Line.gsm` docstring says that a sub-basis is exact only for coplanar lines of constant cross-section.
   * The `SmoothBend` docstring states the conditions for treating evanescent modes as uncoupled: |β_e| ≫ the
     curvature coupling κ|K|, and |β_e|L ≫ 1. Both fail near cutoff, and the error next to a miter or step is
     unquantified.
   * `benchmarks/wp1_cascade_checks.py` no longer imports from `tests/`; it carries its own reference helpers.
6. **Interfaces for the next cards.**
   * **Loss hook:** renamed to `_loss_matrix(basis, f)` and implemented in `Straight` (see Notes).
   * **Changes of cross-section:** supported through `guides = (g_in, g_out)`, so steps are no longer refused.
     `Line` tracks the current guide, attaches the following components to the new guide, and always calls
     `e.gsm(f, current_basis, next_basis)`. `next_basis` is the line's rule in the output guide, in the
     component's frame. The output port is in the last guide, `line.guides = (input, output)`, and
     `describe()` marks each change of guide. Nested lines are flattened, so a nested line may change the guide.
   * **Tests:** a test double (`IdealJunction`) is compared bit for bit with the manual cascade, at cutoff_factor
     1 and 1.5, together with frame changes and an E-plane bend in the new guide. A line that starts in the wrong
     guide is refused. WP4's `Step` was not run here because it is not on this branch.
7. **Memoization.** `Line.gsm(..., cache=dict)` memoizes component GSMs keyed by (original component, guide, f,
   port bases). Straights are not cached. Within one call a memo is always used, so a component that appears twice
   is computed once. Test: an ensemble of three lines that share two bends and differ only in the straight length
   needs 2 bend computations instead of 6, with bit-identical S. Another guide is a separate entry. The cache
   relies on the components being immutable, as the protocol requires.

Not touched: `wrmodes/gsm.py` (GSM.star with two junctions on one plane is handled by the coordinator), twist
frame tags, CONVENTIONS.md.
