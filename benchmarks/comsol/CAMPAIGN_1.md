# COMSOL campaign 1 (Gate A): instructions for Daniel

This replaces the generated Java scripts of `README_WP7.md`, which did not run in COMSOL. The instructions follow
your own validated setup, the one used for the paper's Fig. 6 (`hyperbolic_secant_E_bend_60_deg_lowfreq_parametric.java`):

* RF module, Electromagnetic Waves, Frequency Domain, vacuum inside the guide;
* rectangular TE10 ports with the slit (domain-backed) condition and a PML behind each port;
* free mesh, h_max = λ_min/5, where λ_min is the free-space wavelength at the highest frequency of the sweep;
* export of S-parameters.

Only TE10 → TE10 transmission and reflection are needed. The port computes them by projecting onto the TE10 mode,
and every other mode is orthogonal to it. The PML absorbs whatever passes the port. So the ports can sit close to
the discontinuity: **15 mm from the nearest non-uniform point is enough**. The distance only shifts phases, which the
comparison de-embeds.

The models are kept small on purpose:
* **C1 is 2D.** An H-plane miter does not vary along y, so E_y is exactly a 2D problem. The application-library
  model "H-Bend Waveguide 2D" is the template.
* **C3 is a quarter model**, using symmetry.
* **C2 reuses your existing models.**

Please send back, for each case:
1. the exported tables, named as below, in COMSOL's text table format;
2. the model as `.java` (File ▸ Compact History, then Save As ▸ Java), for the record;
3. the COMSOL version, the number of degrees of freedom, and the solver tolerance, written in a line or two.

Everything goes in `benchmarks/comsol/campaign_1/`. Anything that turns out too heavy for your machine can be
skipped or reduced. The order below is the order of priority.

## Common settings

| Item | Value |
| --- | --- |
| Guide | WR-90 inside: 22.86 × 10.16 mm (0.9 × 0.4 in) |
| Walls | PEC, except in C2 |
| Ports | Rectangular, TE10, slit + domain-backed, PML behind (about 1 cm is enough; 2 cm as in Fig. 6 if cheap); port 1 excited, port 2 passive |
| Mesh | h_max = c/f_max/5, as in Fig. 6 |
| Solver | relative tolerance 1e-3 if affordable (Fig. 6 used 1e-2) |
| Export per frequency | `emw.S11dB`, `emw.S21dB`, and the complex `emw.S11`, `emw.S21` (real and imaginary parts) |

**Mesh check** (cheap in C1 and C3): repeat two frequencies with h_max halved and export them to a file with the
suffix `_fine`. The difference between the two meshes is the COMSOL error bar used in the comparison.

**Phase reference.** COMSOL references S-parameter phases to the port boundaries. Please do not use port
de-embedding, and keep the port positions as given.

## C1: H-plane plain 90° miter in WR-90, in 2D (priority 1)

The 2D plane is the bend plane (x, z), with side a = 22.86 mm across the guide. E is out of plane, the TE10 field
of an H-plane bend. Draw this outline, in mm (in COMSOL's 2D plane, use x horizontal and z as the vertical
coordinate):

```
( 0.0000, -26.4300)   input port edge, outer wall
( 0.0000, -11.4300)   outer wall ends; the mirror starts here
(22.8600,  11.4300)   mirror ends, output arm outer wall starts
(37.8600,  11.4300)   output port edge, outer wall
(37.8600, -11.4300)   output port edge, inner wall
(22.8600, -11.4300)   inner corner
(22.8600, -26.4300)   input port edge, inner wall
```

* **Ports:** the input port is the edge z = −26.43 mm and the output port the edge x = 37.86 mm, each 15 mm from
  its reference plane. The reference planes are z = −11.43 mm and x = 22.86 mm, the planes through the inner corner
  (CONVENTIONS.md §6). Add the PML beyond each port.
* **Mirror:** the 45° edge from (0, −11.43) to (22.86, 11.43). It passes through the intersection of the arm axes,
  at (11.43, 0).
* **Frequencies** (23 points): 18 to 90 GHz in 4 GHz steps, plus 26.3, 52.5, 53.0 and 53.5 GHz, near the predicted
  loss maxima just above the TE40 and TE80 cutoffs. If 2D is as cheap as expected, 2 GHz steps are welcome.
* **Mesh check:** 30 and 90 GHz.
* **Files:** `C1_miter90.txt`, `C1_miter90_fine.txt`.

### C1b: optical miters at 60° and 120°, in 2D (priority 2)

These test the plain-miter definition: the mirror passes through the intersection of the arm axes, perpendicular to
the bisector. In both outlines the output port is the edge between the 4th and 5th vertices.

60° outline, in mm:
```
( 0.0000, -34.7973)  ( 0.0000, -19.7973)  (22.8600, 19.7973)  (35.8504, 27.2973)
(47.2804,   7.5000)  (22.8600,  -6.5991)  (22.8600, -34.7973)
```
Here the mirror cuts into each arm before its reference plane, so each port is 28.1982 mm from its reference plane
(15 mm from the mirror's end).

120° outline, in mm:
```
( 0.0000, -34.7973)  ( 0.0000,  -6.5991)  (22.8600,  6.5991)  (47.2804, -7.5000)
(35.8504, -27.2973)  (22.8600, -19.7973)  (22.8600, -34.7973)
```
Here each port is 15 mm from its reference plane.

* **Frequencies:** 30, 60 and 90 GHz.
* **Files:** `C1b_miter60.txt`, `C1b_miter120.txt`.

## C3: strangling steps, quarter model (priority 3)

* **Geometry.** Both steps are centred: the reduced guide keeps the width 22.86 mm, and its height h is centred on
  the WR-90 height.
* **Quarter model.** Model one quarter, with perfect magnetic conductor (PMC) on the two symmetry planes: the plane
  through the middle of the broad side (x = a/2) and the plane through the middle of the height. Use the
  quarter-port settings COMSOL offers for symmetric rectangular ports, or check the port normalization on a straight
  guide first.
* **Legs:** 15 mm from each port to the step.
* **C3a, single step:** WR-90 → 22.86 × h, for h = 2.0 mm and h = 1.6 mm.
  * Port 2 is a rectangular TE10 port on the reduced guide; give it the reduced guide's dimensions.
  * Frequencies 8, 10 and 12 GHz, where only TE10 propagates on both sides.
  * Mesh check: 10 GHz.
  * Files: `C3a_step_h2.0.txt`, `C3a_step_h1.6.txt`.
* **C3b, strangled section:** WR-90 (15 mm) → 22.86 × 1.6 mm (length exactly 100 mm) → WR-90 (15 mm).
  * Frequencies 10 and 30 GHz, and 60 GHz if affordable. Above 13 GHz the WR-90 sides carry several modes, which the
    PML absorbs; the quarter symmetry keeps only the modes TE10 can couple to.
  * File: `C3b_strangled_100mm.txt`.

## C2: wall loss on your existing 15 cm 60° bends (priority 4)

Take your two Fig. 6 models (hyperbolic secant and constant curvature, 15 cm, 60°, E-plane, TE10) unchanged, except
for the walls:
* Replace PEC by **Impedance boundary condition** on all waveguide walls, but not on the PML outer boundaries.
* Material: σ = 3.5×10⁷ S/m (aluminium baseline), εr = 1, μr = 1.

The cost is the same as for Fig. 6. If that is too long, 18–40 GHz in 2 GHz steps is enough.

Files: `C2_sech_aluminium.txt`, `C2_constant_aluminium.txt`.

## After you send the files

The coordinator converts the tables, de-embeds the phases with the port distances given here, and compares them with:
* C1: wrmodes' 2D FEM miter (WP3);
* C3: mode matching (WP4) and Line;
* C2: lossy SmoothBend (WP2).

Results go into `benchmarks/validation_v1/REPORT.md`. Any disagreement beyond the COMSOL error bar plus 0.02 dB
(CONVENTIONS.md §8) gets a discrepancy report.
