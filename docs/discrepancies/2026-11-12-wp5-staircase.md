# Discrepancy: WP5 taper vs a staircase of WP4 Steps — acceptance row met for the b-taper, not for the H-plane taper

*Opened by WP5 on 2026-11-12. To be settled by the coordinator, a theory agent and Daniel, not by WP5.*

## The row

Card WP5: "Staircase of WP4 `Step`s (H-plane taper and a taper of b, both): converges to the WP5 result as the step
count grows; extrapolated difference ≤ 1e-4 in every propagating entry (report the convergence)."

## What was done

`benchmarks/taper/staircase.py` (data `benchmarks/taper/data/staircase.npz`, `staircase_b_long.npz`). A taper of
length L is replaced by n uniform sections of length L/n with the sides of the taper at the section midpoints,
joined by WP4 mode-matching `Step`s (default truncation, parity-reduced), plus a step from each end cross-section.
The sections carry the modes of the coupled class with cutoff below cf·f, cf = max(6, 4n/(kL)) (every mode with
|β|·L/n ≲ 4, so that neighbouring steps interact through their evanescent fields); the effect of cf is checked by
doubling it at the largest n. The reference is `Taper(..., mode_factor=8, atol=1e-9)` (B5′ on).

* 'H-plane': linear taper of a, 22.86 → 45 mm, b = 10.16 mm, L = 0.1 m, 15 GHz (TE3,0 crosses cutoff), family 0.
* 'b-taper': linear taper of b, 22.86 × 10.16 → 22.86 × 5.08 mm, L = 0.1 m, 35 GHz (TE1,2/TM1,2 cross cutoff),
  modes with m = 1.

## Result (max |S_stair − S_taper| over all propagating entries)

| n | H-plane | b-taper |
| --- | --- | --- |
| 20 | 2.1e-1 | 2.7e-1 |
| 40 | 1.1e-1 | 6.6e-2 |
| 80 | 3.1e-2 | 2.3e-2 |
| 160 | 5.9e-3 | 7.4e-3 |
| 320 | 2.0e-3 | 2.1e-3 |
| 640 | 9.7e-4 | 3.0e-4 |
| 1280 | not run (about 1 h) | **8.6e-5** |
| Richardson (last two n, observed order) | 6.2e-4 (n = 320, 640; order 2.0) | **5.6e-5** (n = 640, 1280; order 3.0) |
| section truncation ×2 at the given n | 4.7e-3 (n = 320) | 1.1e-5 (n = 320), 5.8e-4 (n = 640) |

Data: `staircase.npz` (n = 20–320, both cases), `staircase_b_long.npz` (b-taper, 160–640 with the truncation check),
`staircase_b_1280.npz` (b-taper, 320–1280), `staircase_h_640.npz` (H-plane, 160–640). Times per staircase: H-plane
29 s (n = 320) and 275 s (n = 640); b-taper 20 s, 100 s and 760 s (n = 320, 640, 1280).

**The b-taper meets the row** (raw and extrapolated difference below 1e-4 at n = 1280). **The H-plane taper does
not**: at n = 640 the difference is 9.7e-4 (extrapolated 6.2e-4), and the staircase's own truncation uncertainty is
larger than that.

## Why we think the taper is right and the staircase is not yet converged

1. The staircase has three slowly converging parameters at once: the step count (each step of size δ adds O(δ²)
   near-field and reflection effects, so the sum converges as O(1/n) or slower; the observed orders, 1.5–2.7, are not
   asymptotic), the section truncation (must grow with n; at n = 320 doubling it still changes the H-plane result
   by 4.7e-3, more than the difference to the taper), and each step's own truncation (edge singularities; the Step's
   default resolves features down to a quarter of the side, not the step height δ). T2 §13 anticipated this: "the
   comparison as step → 0 must be made at converged mode counts on both sides, or with extrapolation".
2. The taper itself is checked by other references to far better than 1e-4: the T2b §9 table (independent
   prototype, 5e-6), the T2 prototype solver (8e-8), exact single-mode solutions through cutoff (1e-11), the
   uniform limits (5e-13), and its own convergence in mode count, B5′ and slices (3e-5 between mode_factor 4 and 8).
3. The staircase–taper difference falls monotonically with n in both cases (by 3000× from n = 20 to 1280 for the
   b-taper), and the case with the cheaper staircase (b-taper: fewer modes per section) reaches 1e-4.

## What would settle it

For the H-plane case, a staircase with n = 1280–2560 and section mode counts growing with n (700–1500 modes per
section): hours of compute, best done by WP7b (whose independent references
this is meant to be) or a later card. Alternatively, the coordinator may accept the T2b, T2-prototype and exact-
solution comparisons as the independent checks of the taper and record the staircase as "converging, reported".
