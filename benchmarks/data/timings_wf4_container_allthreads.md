# wrmodes timings (2026-11-24T13:11:06)

Machine: Intel(R) Xeon(R) Processor @ 2.30GHz, 2 logical cores (2 physical), 7.8 GB RAM, Linux-6.18.44-fc-v819-x86_64-with-glibc2.39; Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1; BLAS openblas 0.3.31.dev (1 threads), openblas 0.3.34.106.0 (1 threads); wrmodes 1.0.0b1 (f05927225b)

| item | description | modes (port 1, port 2) | BLAS threads | best (s) |
|---|---|---|---|---|
| lossy_straight_90GHz | Straight 1 m, WR-90 aluminium (3.5e7 S/m), all propagating modes (all families coupled by the wall loss), 90 GHz | 129, 129 | all | 0.104 |
| smoothbend_propagating_40GHz | SmoothBend H-plane sech 90 deg, 20 cm, WR-90 aluminium, propagating modes (transfer-matrix route), 40 GHz | 26, 26 | all (self-limited to 1) | 0.217 |
| smoothbend_evanescent_40GHz | the same bend, basis to 1.5 f (evanescent route, VariableGuide slice GSMs, internal list to 3 f), 40 GHz | 57, 57 | all | 2.356 |
| taper_raised_cosine_90GHz | Taper raised cosine WR-90 -> 22.86 x 1.6 mm over 5 cm (E3's taper), perfect walls, default bases, 90 GHz | 129, 13 | all | 3.937 |
| strangled_bend_40GHz | StrangledBend E3 geometry (90 deg, R = 50 mm, 1.6 mm flat, 5 cm raised-cosine tapers), WR-90, perfect walls, 40 GHz | 26, 26 | all | 1.319 |
| strangled_bend_90GHz | StrangledBend E3 geometry (90 deg, R = 50 mm, 1.6 mm flat, 5 cm raised-cosine tapers), WR-90, perfect walls, 90 GHz | 129, 129 | all | 18.357 |
| twist_constant_18GHz | Twist 90 deg constant, 20 in, tall WR187 (Doane), defaults (mode_factor 3, tail), 18 GHz | 23, 23 | all (self-limited to 1) | 0.357 |
| twist_triangular_18GHz | Twist 90 deg triangular, 20 in, tall WR187 (Doane), defaults (mode_factor 2, tail), 18 GHz | 23, 23 | all (self-limited to 1) | 4.595 |
| miter_mm_H_30GHz | MiterMM H-plane 90 deg optical mirror, WR-90, default truncation, 30 GHz | 12, 12 | all | 0.055 |
| miter_mm_H_60GHz | MiterMM H-plane 90 deg optical mirror, WR-90, default truncation, 60 GHz | 57, 57 | all | 0.230 |
| miter_mm_H_90GHz | MiterMM H-plane 90 deg optical mirror, WR-90, default truncation, 90 GHz | 129, 129 | all | 0.541 |
| miter_mm_E_30GHz | MiterMM E-plane 90 deg optical mirror, WR-90, default truncation, 30 GHz | 12, 12 | all | 0.084 |
| miter_mm_E_60GHz | MiterMM E-plane 90 deg optical mirror, WR-90, default truncation, 60 GHz | 57, 57 | all | 0.264 |
| miter_mm_E_90GHz | MiterMM E-plane 90 deg optical mirror, WR-90, default truncation, 90 GHz | 129, 129 | all | 0.523 |
| fem2d_miter_H_one_family_90GHz | fem2d.Miter H-plane 90 deg, WR-90, family n = 0 only (13 modes), default mesh, 90 GHz | 13, 13 | all | 0.590 |
| fem2d_miter_H_full_90GHz | fem2d.Miter H-plane 90 deg, WR-90, all propagating modes (129), default mesh, 90 GHz | 129, 129 | all | 4.150 |
| step_parity_on_90GHz | Step WR-90 -> 22.86 x 1.6 mm, default truncation (3415, 526 modes), parity reduction on, 90 GHz | 3415, 526 | all | 0.289 |
| step_parity_off_90GHz | Step WR-90 -> 22.86 x 1.6 mm, default truncation (3415, 526 modes), parity reduction off, 90 GHz | 3415, 526 | all | 2.525 |
| lossy_wall_filter_M80_75GHz | LossyWallFilter 15 cm, 250 ohm narrow walls, WR-90, M = 80 modes per family, propagating basis, 75 GHz | 94, 94 | all | 6.162 |
| demo_line_pec | straight 1 m -> H-plane sech bend 90 deg, 30 cm -> straight 0.5 m -> constant-rate 90 deg twist, 15 cm -> straight 0.3 m -> E-plane 90 deg chamfered miter (d/h = 0.87, COMSOL E1's validated chamfer) -> straight 0.2 m -> linear H-plane taper to 22.86 x 10.16 mm over 10 cm -> straight 2 m -> two 45 deg optical H-plane miters with spacer 3a -> straight 0.5 m. Perfect walls. | – | all | 0.652 per frequency (16 f, total 10.4) |
| demo_line_copper | straight 1 m -> H-plane sech bend 90 deg, 30 cm -> straight 0.5 m -> constant-rate 90 deg twist, 15 cm -> straight 0.3 m -> E-plane 90 deg chamfered miter (d/h = 0.87, COMSOL E1's validated chamfer) -> straight 0.2 m -> linear H-plane taper to 22.86 x 10.16 mm over 10 cm -> straight 2 m -> two 45 deg optical H-plane miters with spacer 3a -> straight 0.5 m. Copper (5.8e7 S/m). | – | all | 0.860 per frequency (16 f, total 13.8) |

