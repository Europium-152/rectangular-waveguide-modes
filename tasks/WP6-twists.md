# WP6 — Twists

Role: Implementer · Wave: 2 · Branch: `wave-2/WP6-twists` · Depends on: T3, T3b (merged), WP1, WP2 (merged) ·
Verified by: WP7b (independent method, COMSOL campaign 2) · Time box: 6 days (then status review)

## Goal

A `Twist` component: a straight rectangular guide whose cross-section rotates about the axis by a total angle
θ over a length L with a given rate profile, coupling all families, with the line's global frame at both ports.

## Read first

* `CONVENTIONS.md`: §2 (a twist rotates the line frame; no `FrameChange` follows it), §3 (the 'live' rule and
  `evanescent_families`), §6, §8.
* `docs/theory/T3-twist-coupling.md` (summary, §3 closed forms, §5 end frames, §6 profiles, §8 notes for WP6) and
  `docs/theory/T3b-twist-coupling.md` (§0, box 2b `K_matrix_fast`, box 4, §5, §6, §8, §9 notes for WP6). The two
  agree to 8e-18; use either set of closed forms, not a mixture.
* Doane & Anderson (1989) eqs. 17–18 and Fig. 9 (20-inch 90° twists in WR187 tall guide; constant, cosine and
  triangular rate), digitized in `benchmarks/literature/doane1989_fig9.csv`; Unger (1957) eq. 20;
  Katsenelenbaum pp. 139–158 (eq. 4.29 has a typo, see T3 §6).

## Specification

1. `Twist(angle_deg, length, profile='constant', guide=None)`. `profile` is `'constant'`, `'cosine'`
   (θ′ ∝ sin(πz/L)), `'triangular'` (θ′ rising linearly to L/2 and falling back), or a callable rate shape
   $r(t)$, $t = z/L$, normalized internally so that $\int_0^L\theta'\,dz = \theta$. Define the named profiles
   exactly as Doane & Anderson do, and state it in the docstring. Positive angle: the sense of T3's θ (state it in
   terms of the global $(u, v)$ axes).
2. Frames (CONVENTIONS §2): the global guide frame rotates with the cross-section, so the twist's GSM has the
   line's global-frame basis at **both** ports, for any angle, and the guide after it is the same W × H guide.
   The angle changes only the orientation in space of what follows (a 90° twist turns the next H-plane bend into
   a bend in the other spatial plane). `plane` is `'H'` (local frame = global frame); `Line` needs no new rule.
   Check in a test that a `Line` [Twist(+90), SmoothBend(plane='H'), Twist(−90)] runs and conserves power, and
   state in the report what it represents physically.
3. Method: T3-1/box 4 system, linear in θ′, coefficients frequency-independent in the V–I form. Constant rate:
   `expm` of the section; variable rate: midpoint slicing with `expm` per slice (T3b §9) or the 4th-order
   commutator-free Magnus of `transfer_matrix`, with automatic slice refinement. Build the GSM directly when the
   basis has evanescent modes (CONVENTIONS §6).
4. Basis: all families coupled by the selection rule (m + m′ and n + n′ odd). Use `evanescent_families='all'`
   when `cutoff_factor > 1` (T3b §9: about 6 % relative on Doane's twist).
5. Loss: with `guide.sigma`, add the T1 loss matrix of the (constant) cross-section in the co-rotating frame.
6. `convergence_study(f)` in mode count and slices.

## Acceptance tests

| Check | Target |
| --- | --- |
| Exact two-mode law (T3 box 4 / T3b square-guide limit) | 1e-12 |
| $Z(-\theta)Z(\theta) = I$ at zero length; +90° then −90° twist of finite length vs the identity in the long-twist limit | 1e-12 (algebraic); the finite case reported |
| θ → 0, with and without loss | `Straight` to 1e-12 |
| Reciprocity, unitarity (lossless), passivity (lossy) | CONVENTIONS §6 |
| T3/T3b full-basis tables for WR187 at 15, 16.5, 18 GHz (constant, cosine, triangular) | reproduced to 1e-3 relative (regression values) |
| Doane & Anderson Fig. 9, constant and cosine | within the digitizing uncertainty |
| Fig. 9 triangular | reported against the T3 finding (0.11 / 0.25 / 0.60 %); settled by COMSOL campaign 2 |
| Galerkin propagation constants of the uniformly twisted guide (T3 §7, T3b §5) | within the Galerkin's own convergence (state it) |

## Deliverables

* `src/wrmodes/components/twist.py`, exported in `wrmodes.components`; the `Line` rule of item 2.
* `tests/test_twist.py`; `docs/reports/WP6.md`; `examples/twist.py` (Doane's three twists).

## Out of scope

Twists of non-rectangular guides; twists combined with tapers or bends in one component (cascade them instead).
