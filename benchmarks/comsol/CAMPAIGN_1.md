# COMSOL campaign 1 (Gate A): instructions for Daniel

This replaces the generated Java scripts of `README_WP7.md`, which did not run in COMSOL. The instructions follow
your own validated setup, the one used for the paper's Fig. 6 (`hyperbolic_secant_E_bend_60_deg_lowfreq_parametric.java`):

* RF module, Electromagnetic Waves, Frequency Domain, 3D, vacuum inside the guide;
* rectangular TE10 ports with the slit (domain-backed) condition and a 2 cm PML behind each port;
* free tetrahedral mesh, h_max = λ_min/5, where λ_min is the free-space wavelength at the highest frequency of the
  sweep;
* export of S-parameters.

Only the TE10 port mode is needed. Every other mode is absorbed by the PML and is not exported. The wrmodes side
computes TE10 → TE10 transmission and reflection for comparison.

Please send back, for each case:
1. the exported tables, named as below, in COMSOL's text table format;
2. the model as `.java` (File ▸ Compact History, then Save As ▸ Java), for the record;
3. the COMSOL version, the number of degrees of freedom, and the solver tolerance, written in a line or two.

Everything goes in `benchmarks/comsol/campaign_1/`.

## Common settings

| Item | Value |
| --- | --- |
| Guide | WR-90 inside: 22.86 × 10.16 mm (0.9 × 0.4 in) |
| Walls | PEC, except in C2 |
| Ports | Rectangular, TE10, slit + domain-backed, PML 2 cm behind; port 1 excited, port 2 passive |
| Mesh | h_max = c/f_max/5 (as in Fig. 6); see the mesh check below |
| Solver | Iterative as before, relative tolerance 1e-3 (Fig. 6 used 1e-2, which may limit the comparison at the 0.01 dB level) |
| Export per frequency | `emw.S11dB`, `emw.S21dB`, and the complex `emw.S11`, `emw.S21` (real and imaginary parts) |

**Mesh check.** For C1 and C3, repeat three frequencies (C1: 30, 60, 90 GHz; C3: all) with h_max halved. Export the
same quantities to a file with the suffix `_fine`. The difference between the two meshes is the COMSOL error bar
used in the comparison.

**Phase reference.** COMSOL references S-parameter phases to the port boundaries, and the comparison accounts for
that. Please do not use port de-embedding, and keep the port positions exactly as given, because the phase
comparison depends on them.

## C1: H-plane plain 90° miter in WR-90 (the main Gate A benchmark)

Electric field along y (perpendicular to the bend plane), side a = 22.86 mm in the bend plane (x–z), side
b = 10.16 mm along y. Draw the outline below in a work plane (x, z), in mm, and extrude it by 10.16 mm in y.

```
( 0.0000, -71.4300)   input port edge, outer wall
( 0.0000, -11.4300)   outer wall ends; the mirror starts here
(22.8600,  11.4300)   mirror ends, output arm outer wall starts
(82.8600,  11.4300)   output port edge, outer wall
(82.8600, -11.4300)   output port edge, inner wall
(22.8600, -11.4300)   inner corner
(22.8600, -71.4300)   input port edge, inner wall
```

* The input port is the face z = −71.43 mm and the output port the face x = 82.86 mm. Each is 60 mm from its
  reference plane: z = −11.43 mm and x = 22.86 mm, the planes through the inner corner (CONVENTIONS.md §6).
* Add the 2 cm PML beyond each port, extruding the arms by 20 mm.
* The mirror is the 45° face from (0, −11.43) to (22.86, 11.43). It passes through the intersection of the arm
  axes, at (11.43, 0).

Frequencies (41 points): 18 to 90 GHz in 2 GHz steps, plus 26.3, 52.5, 53.0 and 53.5 GHz (near the predicted loss
maxima just above the TE40 and TE80 cutoffs).

Files: `C1_miter90.txt`, `C1_miter90_fine.txt`.

### C1b (optional, if C1 goes smoothly): optical miters at 60° and 120°

These test the new definition of the plain miter (mirror through the intersection of the arm axes, perpendicular
to the bisector). Same construction, ports again 60 mm from the reference planes.

60° outline (mm):
```
( 0.0000, -66.5991)  ( 0.0000, -19.7973)  (22.8600, 19.7973)  (63.3915, 43.1982)
(74.8215,  23.4009)  (22.8600,  -6.5991)  (22.8600, -66.5991)
```
120° outline (mm):
```
( 0.0000, -79.7973)  ( 0.0000,  -6.5991)  (22.8600,  6.5991)  (86.2515, -30.0000)
(74.8215, -49.7973)  (22.8600, -19.7973)  (22.8600, -79.7973)
```

In both, the output port is the edge between the 4th and 5th vertices. Frequencies: 30, 60 and 90 GHz. Files:
`C1b_miter60.txt`, `C1b_miter120.txt`.

## C2: wall loss on your existing 15 cm 60° bends

Take your two Fig. 6 models (hyperbolic secant and constant curvature, 15 cm, 60°, E-plane, TE10) unchanged, except
for the walls:
* Replace PEC by **Impedance boundary condition** on all waveguide walls, but not on the PML outer boundaries.
* Material: σ = 3.5×10⁷ S/m (aluminium baseline), εr = 1, μr = 1.

Use the same sweep as Fig. 6: 18–60 GHz in 2 GHz steps.

Files: `C2_sech_aluminium.txt`, `C2_constant_aluminium.txt`. If convenient, also rerun the PEC models with the
1e-3 solver tolerance (`C2_sech_pec.txt`, `C2_constant_pec.txt`). That would separate the effect of the tolerance
from that of the walls.

## C3: strangling steps (E-plane steps, height reduced)

Both steps are centred: the reduced guide has the same width 22.86 mm and its height h is centred on the WR-90
height.

* **C3a, single step.** WR-90 (straight, 40 mm from port 1 to the step) to 22.86 × h (40 mm from the step to
  port 2), for h = 2.0 mm and h = 1.6 mm.
  * Port 2 is a rectangular TE10 port on the reduced guide; give it the reduced guide's dimensions.
  * Frequencies 8, 10 and 12 GHz. Only TE10 propagates on both sides there.
  * Files: `C3a_step_h2.0.txt`, `C3a_step_h1.6.txt`.
* **C3b, strangled section.** WR-90 (40 mm) to 22.86 × 1.6 mm (length exactly 100 mm) and back to WR-90 (40 mm).
  * Frequencies 10, 30 and 60 GHz. Above 13 GHz the WR-90 sides carry several modes, which the PML absorbs.
  * File: `C3b_strangled_100mm.txt`.

## After you send the files

The coordinator converts the tables, de-embeds the phases with the port distances given here, and compares them with:
* C1: wrmodes' 2D FEM miter (WP3);
* C2: lossy SmoothBend (WP2);
* C3: mode matching (WP4) and Line.

Results go into `benchmarks/validation_v1/REPORT.md`. Any disagreement beyond the COMSOL error bar plus 0.02 dB
(CONVENTIONS.md §8) gets a discrepancy report.
