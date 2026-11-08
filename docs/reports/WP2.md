# WP2 report: ohmic wall loss and mode filters

Card: `tasks/WP2-loss-and-filters.md`. Branch `wave-1/WP2-loss-and-filters`, started from `wave-1/WP4-steps` (WP1
components and Line, WP4 Step and `junction_gsm`, conventions update 1) with the theory branches T1 and T1b merged.
Specification: `docs/theory/T1-wall-impedance.md` (boxes B1-B10); references in the tests: `docs/theory/T1/t1lib.py`
and `docs/theory/T1b/t1b_coefficients.py` (imported by the tests only, nothing copied into `src/`).

## What was done

* **`wrmodes.loss`** (new): `surface_resistance(f, sigma)`, `surface_impedance` (B1, no default material);
  `wall_integrals` (B2a, closed forms per wall, vectorized, straight part and kappa coefficient); `loss_matrices`
  (B3, Lambda^+ and Lambda^-, any wall impedances); `loss_matrix_full`, `bend_loss_matrices` (B4, B4b);
  `coupling_mask` (T1 section 5 rule: only propagating modes are coupled through Lambda, evanescent modes get their
  own Lambda^+_pp only); **`attenuation_constant(mode, a, b, f, sigma)`** (Np/m, broadcasts over f; the first name
  WP7's `ref_interfaces.attenuation_function` tries) and `attenuation_constants(basis, f, sigma)`;
  `lossy_straight_gsm`, `straight_gsm_from_loss_matrix` (uniform section without growing exponentials: GSM of a
  short slice from the matrix exponential, doubled by star products).
* **Straight** (WP1 hook): `_loss_matrix(basis, f)` now returns Lambda_full (2N x 2N); a guide with `sigma` makes the
  section lossy with the full propagating system (degenerate TE/TM coupling, cross-family broad-wall coupling,
  Lambda^-), evanescent modes with their own attenuation. `plane` stays None (tested: E-frame result mapped back
  equals the H-frame result to 1.2e-16, degenerate and cross-family coupling included).
* **SmoothBend** (WP1 hook): `_loss_matrix` returns (Lambda0_full, Lambda1_full); each family is solved with
  `solve_transfer_matrix(..., loss_matrix=(L0, L1))`, kappa Lambda1 included; the "K = 0 -> straight" shortcut now
  requires K = 0 and Lambda1 = 0 and then uses the lossy straight; evanescent modes get Lambda^+_pp. New option
  `families='coupled'` (one system with all propagating modes).
* **`solve_transfer_matrix(..., loss_matrix=None)`**: Lambda0_full or (Lambda0_full, Lambda1_full); the system is
  [(D - L0) + kappa(z)(K - L1)] y, the Magnus integrator, error estimate and minimum-slice rule are unchanged (the
  rule sees K - L1). None is the 0.1 solver bit for bit.
* **`wrmodes.filters`** (new): `LossyWallFilter(length, Zw, walls=('u-','u+'), plane='H', modes_per_family=None)`
  and `ResistiveSheetFilter(length, R_sheet, v0=0.0, n_sheets=1, positions=None, plane='H',
  modes_per_family=None)`, both Components with `plane`, usable in a `Line`. Exact methods: LS modes of the loaded
  guide (B6 impedance walls; B7/B8 sheets, any number and position) from Chebyshev collocation of the 1D problem
  (all roots at once, no continuation in Z) polished by Newton on the exact dispersion function (a transfer of
  (X, X') across the guide, entire in k_t^2), fields normalized with the unconjugated reaction, junctions in the
  reciprocal B9 form through `wrmodes.modematch.junction_gsm` (complex overlaps), Redheffer cascade
  junction * section * reversed junction. Walls on v are solved in the frame rotated by 90 degrees. Also public:
  `impedance_wall_modes`, `sheet_modes` (transverse wavenumbers).
* **Guide**: `wall_impedance(f)`; `require_lossless` message now says "not supported by this component" (for steps
  and miters). **Line**: a component with `guides` (Step) in a lossy line raises `NotImplementedError` ("perfectly
  conducting walls only ...") instead of a guide-mismatch `ValueError`.
* `wrmodes/__init__.py`: one line, `from .filters import LossyWallFilter, ResistiveSheetFilter`.
* Tests: `tests/test_loss.py` (41 tests, about 70 s); WP1's hook tests in `tests/test_components.py` updated (the
  hooks now carry loss). `examples/lossy_line.py`. Discrepancy report
  `docs/discrepancies/2026-11-08-wp2-wp7-straight-attenuation.md`.

Files of other cards touched (hook implementation only): `components/straight.py`, `components/bends.py`,
`components/guide.py`, `line.py` (one branch), `transfer_matrix.py` (new optional argument), `tests/test_components.py`.

## Results

**Attenuation, aluminium WR-90.** TE1,0: 0.1227 dB/m (18 GHz), 0.2303 dB/m (90 GHz); equal to the card's formula
and to Marcuvitz eqs. (4), (10) for every propagating mode to 1e-12 (T1 and T1b references). Loss matrices equal
t1lib and T1b to 1e-12 relative (random sides, frequencies and complex unequal walls, orders 0 and 1); wall
integrals equal the quadrature of `transverse_field` to 1e-11.

**Coupled vs diagonal attenuation.** With one pair of lossy walls (exact Leontovich modes known, B6), the eigenvalue
of the coupled propagating system is within 4e-4 of the exact attenuation for TE1,0 and TE0,1 at 18/54/90 GHz; the
diagonal (Marcuvitz) value is off by up to 3.8e-3 (TE0,1, side walls, 90 GHz) and 1.7e-3 (TE1,0, broad walls).
In a 20 m aluminium straight at 90 GHz TE1,0 therefore decays 1.4e-3 faster than alpha L (4.613 vs 4.607 dB).

**Long straights** (20 m aluminium): finite, reciprocal to 4.6e-13 / 3.1e-12 / 7.8e-12 at 18 / 54 / 90 GHz, passive
(margin 0.22-0.25); TE1,1 -> TM1,1 fraction 2.7 % / 4.2 % / 6.1 %, TE2,1 -> TM2,1 9.5 % (54 GHz), 14 % (90 GHz);
largest reflection 2.5e-3 in amplitude (near-cutoff modes, Lambda^-), 0.16 s at 90 GHz (129 modes).

**Lossy SPARC bends** (in-mode loss, dB; T1 section 6 table in brackets): 40 cm 90 deg sech H 90 GHz 2.4193
(2.4193), E 0.4272 (0.4272); 60 cm 60 deg constant H 8.0718 (8.0718), E 0.1749 (0.1749). kappa Lambda1 changes
converted powers by up to 1.33e-4 (H, 90 GHz, 40 cm sech; T1: 1.3e-4). Families treated separately vs `families='coupled'` (largest
change of any |S_ij|^2 of the propagating block / of the TE1,0 column): 4e-6 / 1e-10-7e-8 at 30 GHz; 6.3e-5 /
1.2e-5 (H), 7e-8 (E) at 60 GHz; 1.3e-4 / 2.4e-5 (H), 4e-8 (E) at 90 GHz. The TE1,0 part is the broad-wall
correction above (about 1e-4 dB), the largest entries are near-cutoff modes. Time per bend at 90 GHz: separate
2-7 s, coupled 96-105 s.

**20 m aluminium line with three bends** (`examples/lossy_line.py`: 6.2 m straight, 40 cm 90 deg sech H, 6.2 m,
40 cm 90 deg sech E, 6.2 m, 60 cm 60 deg constant H; one process, BLAS 1 thread):

| f | modes | TE1,0 loss, perfect walls | aluminium | alpha L alone | time perfect / aluminium |
| --- | --- | --- | --- | --- | --- |
| 18 GHz | 5 | 0.0013 dB | 2.4571 dB | 2.4550 dB | 0.0 s / 0.1 s |
| 54 GHz | 45 | 1.1231 dB | 4.7615 dB | 3.6153 dB | 2.4 s / 2.3 s |
| 90 GHz | 129 | 3.9835 dB | 8.7086 dB | 4.6067 dB | 14.4 s / 15.8 s |

**Butterweck lossy-wall filter** (22.86 x 10.16 mm, 75 GHz, 250 ohm on both narrow walls, 15 cm), exact method:

| modes per family M | H10 | H20 | H30 | H10 -> H30 | TE0,1 | TE1,1 | TM1,1 | time (94 port modes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40 | 0.5693 | 2.3169 | 5.3412 | -33.535 | 2.8350 | 12.237 | 2.6623 | 0.9 s |
| 80 (default) | 0.5691 | 2.3157 | 5.3386 | -33.542 | 2.8355 | 12.240 | 2.6623 | 4.4 s |
| 160 | 0.5688 | 2.3147 | 5.3366 | -33.552 | 2.8356 | 12.241 | 2.6624 | 25 s |

(dB). T1: 0.569 / 2.315 / 5.337 / -33.55 at 160 modes. Reciprocal to 2.5e-16, passive; the junction reflections of
H10-H30 are below 3.2e-3. LS^u roots equal `t1lib.lsu_roots_all` to 1e-12. Z = 0 gives the empty guide to 2e-13;
at Z = 2.5 ohm the exact attenuation of TE1,0-TE3,0 equals the first-order lossy straight on the same walls to 2e-3.

**Resistive sheet.** b = 10 mm, centred sheet: EH11^(2) (LSM^v) maximum attenuation 1.553 dB/cm at 55.71 ohm
(75 GHz) and 1.554 dB/cm at 55.75 ohm (lambda = 4 mm, T1: 55.75 ohm, 1.554); EH11^(1) 5.25 dB/cm at 418.6 ohm
(his 420 ohm, about 5 dB/cm). Filter, 10 cm, 56 ohm, b = 10.16 mm, 75 GHz, convergence in the modes per type Nn:

| Nn | 6 | 10 | 14 | 20 | 40 (default) | 80 |
| --- | --- | --- | --- | --- | --- | --- |
| TE1,1 (dB) | 12.721 | 12.887 | 12.972 | 12.882 | 12.884 | 12.888 |
| TM1,1 (dB) | 16.511 | 16.532 | 16.574 | 16.572 | 16.563 | 16.563 |
| TE0,1 (dB) | 10.906 | 11.064 | 11.133 | 11.037 | 11.045 | 11.050 |
| time | 0.2 s | 0.2 s | 0.4 s | 0.6 s | 2.3 s | 11.7 s |

The convergence is slow and not monotone (the coordinator's -12.72 / -12.89 / -12.97 dB are reproduced); from
Nn = 20 on, TE1,1 stays within 0.006 dB. TE1,0 and (centred sheet) TE1,2 pass with |S21| = 1 to 1e-12; R = 1e15 ohm
gives the empty guide to 1e-9; reciprocity 2e-16. Three equidistant 70 ohm sheets (Butterweck's experiment) work in
a lossy Line. The k_t = 0 point of b = 10 mm at lambda = 4 mm needs no special handling: the filter matches TE/TM
modes directly to the sheet modes (no `ls_rotation`), and the sheet moves the n = 5 modes away from k_v = k. A sheet
mode left exactly at k_v = k (nodes at every sheet) is self-orthogonal and is refused with a clear ValueError
(tested).

## Acceptance tests

| Check | Target | Result |
| --- | --- | --- |
| TE1,0 attenuation, WR-90 aluminium, 18-90 GHz | formula to 1e-6; 0.12 / 0.23 dB/m | **pass**: 1e-12; 0.1227 / 0.2303 dB/m |
| Lossless limit | identical to 0.1 and WP1 | **pass**: bitwise (`np.array_equal`) for Straight, SmoothBend, Line, solver |
| Passivity of every lossy component | sv <= 1 | **pass** (filters with an untouched TE1,0: 1 + 9e-16) |
| Reciprocity | 1e-10 | **pass**: straights <= 7.8e-12 (20 m, 90 GHz), bends <= 8e-13, filters <= 3e-16 |
| Lossy straight vs WP7 surface-integral reference | 1e-6 on alpha, every mode | **pass** for the per-mode attenuation (WP7 `compare_wp2_loss.py`: <= 2.0e-12, all cases); **discrepancy** for the decay of S21 of long straights (3.9e-3 WR-90, 2.6e-2 square guide at fc/f = 0.9993): WP2 couples all propagating modes as T1 specifies, see `docs/discrepancies/2026-11-08-wp2-wp7-straight-attenuation.md` |
| Lossy-wall filter, Butterweck (75 GHz, 250 ohm, 15 cm) | 0.6 / 2.5 / 5.4 dB within 10 % | **pass**: 0.569 (-5.2 %), 2.316 (-7.4 %), 5.339 (-1.1 %) |
| Lossy-wall H10 -> H30 conversion, 15 cm | -36 ... -24.5 dB | **pass**: -33.54 dB (-33.55 with 160 modes) |
| Resistive sheet, b = 10 mm, 75 GHz | 56 ohm, 1.5 dB/cm (5 %) | **pass**: 55.71 ohm (-0.5 %), 1.553 dB/cm (+3.5 %) |
| Bend with loss: power balance with the wall integral | 1e-6 | **pass**: 1.9e-14 (H, family 0) and 1.2e-14 (E, family 1), 30 GHz 90 deg sech, dissipation 1.0e-2 |

WP7 adversarial tests (`tests/adversarial/test_wp2_adversarial.py`, run against this branch): 6 pass (alpha of all
129 modes at 90 GHz, aluminium and copper; alpha just above cutoff at rel = 1e-2 and 1e-5; lossy bend passive,
reciprocal and exact lossless limit; bend power balance), 2 fail (`test_square_guide_degenerate_groups_in_a_lossy_
straight` 2.6e-2, `test_20m_lossy_straight_at_90GHz` 3.6e-3): the discrepancy above.

## Open issues

* **WP7 straight-attenuation tests** (discrepancy report): to be settled by T1 and Daniel.
* **Families in lossy bends**: the default treats local families separately (lossless-bend speed); the neglected
  broad-wall coupling changes |S|^2 entries by up to 1.3e-4 at 90 GHz (near-cutoff modes) and the TE1,0 column of an
  H-plane bend by 2.4e-5 (1e-4 dB). `families='coupled'` removes it at 15-40 times the cost. Straights always keep
  it, so a line mixes the two treatments.
* **Filters and metal loss**: the filters treat their other walls as perfect even in a lossy guide (a 15 cm section
  of aluminium WR-90 would add about 0.03 dB). Dielectric carriers of sheets are not modelled.
* **Lambda^- reflections**: lossy straights now reflect (first order, up to 2.5e-3 in amplitude for near-cutoff
  modes over 20 m at 90 GHz; 2.4e-4 at 18 GHz).
* Filter cost: the lossy-wall default (80 modes per family) takes 4.4 s at 75 GHz for 94 port modes; the sheet
  default (Nn = 40) 2.3 s.

## Conventions issues

* Section 3, "For metal walls the effect [of broad-wall family coupling] is at most about 6e-6 of the power (T1b)":
  true for converted power in straights, but the same coupling shifts the attenuation itself by up to 1.4e-3
  relative (TE1,0, 90 GHz) and changes |S|^2 entries of a bend by up to 1.3e-4 when neglected. The sentence could
  say that it bounds conversion only.
* Section 6, passivity "largest singular value of S_p <= 1": components that leave a mode untouched reach
  1 + 1e-15; a stated round-off tolerance (e.g. 1e-12) would help.
EOF
echo ok