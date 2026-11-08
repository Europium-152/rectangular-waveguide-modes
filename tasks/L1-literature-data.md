# L1 — Literature benchmark data

Role: Literature (two agents, L1a and L1b, digitizing independently) · Wave: 1 ·
Branch: `wave-1/L1-literature-data` · Depends on: — · Verified by: comparison of L1a and L1b ·
Time box: 3 days (then status review)

## Goal

Turn the published results that wrmodes 1.0 will be validated against into machine-readable data, each value
with an uncertainty, so that verifiers and tests can use them without opening the papers.

## Read first

* `CONVENTIONS.md` §9 (data files) and §8 (tolerances: digitized data never set a tolerance tighter than their
  uncertainty).
* The papers below, from Daniel's `references/` folder (see `tasks/README.md`, *References*).

## Items

| ID | Source | What | Output |
| --- | --- | --- | --- |
| D1 | Alessandri, Mongiardo & Sorrentino, IEEE MGWL 4 (1994), Figs. 4–5 | return loss of 90° mitered E-plane bends vs normalized frequency, for every d/b shown, plus the crosses of their ref. [7] | `benchmarks/literature/alessandri1994_fig4.csv`, `_fig5.csv` |
| D2 | Doane & Anderson, Microwave J. 32(3) (1989), Fig. 9 (p. 159, PDF p. 7) | total spurious mode power (%) vs frequency, 20-inch 90° twists in WR187 tall guide (1.872 × 0.872 in), constant, cosine and triangular rates | `doane1989_fig9.csv` |
| D3 | Ma et al., Electron. Commun. Jpn. II 80(5) (1997) | every figure of mitered H-plane bend results (return loss, transmission, optimum miter vs frequency), guide dimensions | `ma1997_fig<k>.csv` |
| D4 | Reiter & Arndt, IEEE MTT 43, 796–801 (1995) | results for E- and H-plane bends or miters, if any, and the geometry of every example | `reiter1995_fig<k>.csv` |
| D5 | Butterweck, IEEE MTT-16 (1968); Butterweck & de Ronde, Philips Tech. Rev. 29 (1968) | all quoted numbers: filter attenuations (0.6/2.5/5.4 dB over 15 cm; 4 and 36 dB/m; 1.5 dB/cm at 56 Ω; 0.1 dB/cm mica sheet), taper H10→H30 conversion (−40 dB bottle, −18 dB linear, 140 mm), strangled-bend dimensions (22.86 × 10.13 mm, r = 420 mm, b₁ = 2 mm, λ ≈ 4 mm), H-plane miter claim (< 0.4 dB above 8 f_c) | `butterweck1968_numbers.csv` with the exact sentence and page of each |
| D6 | Marcuvitz, *Waveguide Handbook* (1951), pp. 296–335 (PDF pp. 310–349) | transcription of the equivalent-circuit formulas for E-plane and H-plane steps (symmetric and asymmetric), mitered and curved bends; the parameter ranges and stated accuracies | `docs/literature/marcuvitz-steps-and-bends.md` |
| D7 | Doane & Anderson (1989), eqs. 16–18 | design rules and twist estimate, transcribed | in `docs/literature/doane1989.md` |
| D8 | Kotik & Sivov, Radio Eng. Electron. Phys. 10, 140–142 (1965) | H-plane miter loss vs frequency, **when Daniel's scan arrives** | `kotik1965_fig<k>.csv` |

## Method

* Digitize with a reproducible tool (WebPlotDigitizer or a script). Record the axis calibration points in the CSV
  header.
* L1a and L1b digitize each curve independently. The final file holds the mean, and its `uncertainty` column is
  the larger of half their difference and the axis resolution (half the minor tick spacing divided by the number
  of pixels between ticks).
* CSV header (comment lines starting with `#`): full reference, figure, curve label, axes and units, guide
  dimensions, calibration points, digitizers.

## Acceptance tests

| Check | Target |
| --- | --- |
| Every curve has two independent digitizations | yes, with their difference recorded |
| Spot check by the coordinator: 3 points per figure re-read by eye | within the stated uncertainty |
| Transcribed formulas (D6, D7) | checked symbol by symbol by the other L1 agent |

## Deliverables

`benchmarks/literature/*.csv`, `docs/literature/*.md`, and a `benchmarks/literature/README.md` index (source,
figure, file, uncertainty range).

## Out of scope

Interpreting or simulating the data (WP3, WP7, wave 2 cards).
