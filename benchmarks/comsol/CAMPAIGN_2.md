# COMSOL campaign 2 (Gate B): instructions for Daniel

Five cases for the wave 2 components, in the same spirit as campaign 1: GUI instructions, small models, results
returned as text tables plus the `.java` of each model. It applies what Gate A taught us:

* **Mesh:** user-defined. Mesh 1 ▸ Free Tetrahedral (3D) or Free Triangular (2D) on all domains, with a Size node
  set to "Custom", maximum element size as given per case. Do not use the physics-controlled mesh; it moved a
  15 cm bend by 0.02–0.03 dB.
* **Ports:** Rectangular, slit + domain-backed, with a PML behind each port (2 cm). Set "Type of port" and "Mode
  number" explicitly on every port; wave excitation **on for port 1 only**. Where a case asks for several ports
  on one boundary, add one Port feature per mode on the same boundary, and check each by plotting its port mode
  field (the electric field on the port boundary) before solving: it must look as described.
* **Near a cutoff, ports instead of PML.** The frequencies below were chosen at least 4 % away from the cutoff of
  any mode the component excites, so a PML is safe. Please don't move them.
* **Export:** Results ▸ Derived Values ▸ Global Evaluation of the complex `emw.S11`, `emw.S21` (and `emw.S31`, …
  where there are more ports) for all frequencies, then Table ▸ Export. Not a Data export at a point.
* **Mesh check:** the case says which frequency to repeat with the finer mesh (suffix `_fine`).
* **Return:** `results_files/` and `java_model_files/` as before, plus for each case the number of degrees of
  freedom and the solution time (one line each). Aluminium means σ = 3.774×10⁷ S/m (COMSOL's library value, as in
  campaign 1), εr = μr = 1, as an Impedance boundary condition on every waveguide wall between the ports (not on
  the PML outer walls).

Port distances are measured from the reference planes given per case. The comparison de-embeds them, so please
keep the ports where they are given and do not use COMSOL's port de-embedding.

Order of priority: E3, E4, E1, then the two cheap 2D cases E2 and E5, which take minutes and can be done whenever.

## E3: strangled E-plane bend (priority 1, 3D)

WR-90 tapered in height to a 1.6 mm flat guide, bent 90° in the E plane, tapered back: the SPARC candidate.

**Geometry.** Draw the outline in a work plane (local coordinates p, q, in mm), then Extrude it by 22.86 mm (the
broad side). The work plane is the bend plane and contains the 10.16 mm side, as in your C2 models, so the TE10
electric field lies in the work plane.

| Piece | Feature in the work plane | Definition (mm) |
| --- | --- | --- |
| Input leg | Polygon or lines | p from −5.08 to 5.08, q from −65 to −50 |
| Input taper, two walls | Parametric Curve, parameter s from 0 to 50 | p = ±(10.16 − 8.56·0.5·(1 − cos(π·s/50)))/2, q = −50 + s |
| Bend, two walls | Circular Arc, centre (50, 0), from 90° to 180° | radius 49.2 (inner wall) and 50.8 (outer wall) |
| Output taper, two walls | Parametric Curve, parameter s from 0 to 50 | p = 50 + s, q = 50 ± (1.6 + 8.56·0.5·(1 − cos(π·s/50)))/2 |
| Output leg | Polygon or lines | p from 100 to 115, q from 44.92 to 55.08 |

The input taper closes at q = 0 with height 1.6 mm (p = ±0.8), which is where the arcs start, at angle 180°. The
arcs end at angle 90°, at (50, 49.2) and (50, 50.8), where the output taper starts. Convert to Solid, then
Extrude 22.86 mm. Add the 2 cm PML blocks beyond each port.

* **Ports:** port 1 on the face q = −65, port 2 on the face p = 115; TE10 (E across the 10.16 mm side).
  Reference planes q = −50 and p = 100: **15 mm** each.
* **Walls:** PEC, then the same model again with aluminium on every wall.
* **Frequencies:** 18 and 40 GHz.
* **Mesh:** maximum element size c_const/40[GHz]/5 = 1.5 mm, and in the Size node "Resolution of narrow regions" = 3,
  so that the 1.6 mm gap gets at least three elements across. Mesh check: 18 GHz with c_const/40[GHz]/7.
* **Files:** `E3_strangled_PEC.txt`, `E3_strangled_aluminium.txt`, `E3_strangled_PEC_fine.txt`.

## E4: 20-inch triangular-rate 90° twist in WR187 (priority 2, 3D)

Doane & Anderson's Fig. 9 twist. Our theory predicts 0.11 % of spurious power at 15 GHz and 0.60 % at 18 GHz, mostly
in modes reached only at second order; Fig. 9 reads about 0.05 % and 0.08 %. This case decides it.

**Guide.** WR187 inside 22.15 × 47.55 mm (0.872 × 1.872 in), used "tall": the incident mode has its electric
field along the **long** side (47.55 mm) and one half-sine across the short side. Its cutoff is 6.77 GHz; it is not
the fundamental mode (that one, with E along the short side, has cutoff 3.15 GHz).

**Geometry.** Axis along z, twist length L = 508 mm, total angle Θ = 90° (π/2), triangular rate: the twist rate rises
linearly to the middle and falls back. The rotation angle at height s along the axis is

θ(s) = if(s < L/2, 2·Θ·(s/L)², Θ·(1 − 2·(1 − s/L)²))

which goes from 0 to 90° with zero rate at both ends. Build the four walls as Parametric Surface features (parameters
s from 0 to 508 mm and t across the wall), with a = 22.15 mm, b = 47.55 mm:

| Wall | u | v | t range |
| --- | --- | --- | --- |
| 1 | a/2 | t | −b/2 … b/2 |
| 2 | −a/2 | t | −b/2 … b/2 |
| 3 | t | b/2 | −a/2 … a/2 |
| 4 | t | −b/2 | −a/2 … a/2 |

with x = u·cos θ(s) − v·sin θ(s), y = u·sin θ(s) + v·cos θ(s), z = s. Set the surfaces' relative tolerance to
1e-6 (raise the maximum number of knots if COMSOL asks). Close the solid with a 22.15 × 47.55 mm rectangle at z = 0
(x across 22.15) and a 47.55 × 22.15 mm rectangle at z = 508 mm (x across 47.55), both centred on the axis, and
Convert to Solid. Add straight legs: a block x ±11.075, y ±23.775 for z from −15 to 0, and a block x ±23.775,
y ±11.075 for z from 508 to 523, then the 2 cm PML blocks beyond each. Form Union. (If your COMSOL version's Sweep
feature accepts a twist angle that varies along the spine as an expression, that is an easier route to the same
solid.)

* **Port 1** (z = −15): the incident mode, E along y (the 47.55 mm side), half-sine across x. Wave excitation on.
* **Output ports** (all on the face z = 523, where the long side now lies along x): one port per mode below. Names
  are those of our theory (T3), with m half-sines across the short side and n across the long side.

| Port | Mode (T3 name) | Field at the output face | Cutoff (GHz) |
| --- | --- | --- | --- |
| 2 | TE10, the twisted incident mode | E along x (long side), one half-sine across y | 6.77 |
| 3 | TE01 | E along y (short side), one half-sine across x | 3.15 |
| 4 | TE03 | E along y, three half-sines across x | 9.46 |
| 5 | TE12 | one half-sine across y, two across x, TE | 9.25 |
| 6 | TM12 | same indices, TM | 9.25 |
| 7 | TE21 | two half-sines across y, one across x, TE | 13.90 |
| 8 | TM21 | same indices, TM | 13.90 |

  If COMSOL numbers the indices the other way round on that face, swap them so that the plotted field matches the
  description; please note in your summary which COMSOL mode numbers you used.
* **Frequencies:** 15 and 18 GHz. **Not 16.5 GHz:** there TE23/TM23 (cutoff 16.51 GHz, coupled at first order) sit
  0.07 % below cutoff, which no PML handles.
* **Mesh:** maximum element size c_const/18[GHz]/5 = 3.3 mm. Mesh check, if affordable: 18 GHz with c_const/18[GHz]/7.
* **Files:** `E4_twist_triangular.txt` (S11, S21 … S81), `E4_twist_triangular_fine.txt`.
* **What decides it** (WP6 predictions, `benchmarks/twist/data/e4_predictions.npz`): the power per output mode, not
  the TE10 transmission in dB, which differs from Fig. 9 by only 0.003 dB (15 GHz) and 0.02 dB (18 GHz). At 18 GHz
  the theory puts 0.38 % of the input in TE12 alone, against about 0.15 % in total from Fig. 9, so COMSOL needs a
  per-mode accuracy of about 2e-4 of the input: the mesh check matters here.
* **Optional control:** the same twist with the cosine rate of Fig. 9, θ(s) = Θ·(1 − cos(π·s/L))/2, at 18 GHz,
  where theory and Fig. 9 agree (0.94 % vs about 1.0 %). File `E4_twist_cosine.txt`.

## E1: E-plane 90° miters in WR-90 (priority 3, 3D)

The 2D outline of campaign 1's C1, but in the E plane: the in-plane side is the narrow side h = 10.16 mm, and the
outline is extruded by 22.86 mm. Draw it in a work plane (p, q in mm) as in E3, so the TE10 electric field lies in
the work plane.

The chamfer d is the distance of the miter plane from the inner corner, normalized to h (CONVENTIONS §6, as
Alessandri et al.). Two miters:

Plain (optical mirror), d/h = 0.7071:
```
( 0.0000, -20.0800)  ( 0.0000,  -5.0800)  (10.1600,  5.0800)  (25.1600,  5.0800)
(25.1600,  -5.0800)  (10.1600,  -5.0800)  (10.1600, -20.0800)
```
Alessandri's d/h = 0.87:
```
( 0.0000, -20.0800)  ( 0.0000,  -2.7393)  ( 7.8193,  5.0800)  (25.1600,  5.0800)
(25.1600,  -5.0800)  (10.1600,  -5.0800)  (10.1600, -20.0800)
```
* **Ports:** port 1 on the edge q = −20.08 (extruded: the face), port 2 on p = 25.16; TE10. Reference planes
  q = −5.08 and p = 10.16 (through the inner corner): **15 mm** each.
* **Frequencies:** 8 to 12.4 GHz in 0.4 GHz steps (Alessandri's range), plus 24 and 36 GHz (overmoded).
* **Mesh:** maximum element size c_const/36[GHz]/5. Mesh check: 10 and 36 GHz with c_const/36[GHz]/8.
* **Files:** `E1_miter_E_plain.txt`, `E1_miter_E_d087.txt`, and the `_fine` files.

## E2: linear H-plane taper, 22.86 → 45 mm (priority 4, 2D)

The 2D H-plane setup of C1. A symmetric linear taper from WR-90 width 22.86 mm to 45 mm, over L = 100 mm and
L = 200 mm. Inside it, TE30 passes through cutoff (at a width of 29.98 mm). Outline in mm, centred on the axis:
```
(-11.4300, -15)  (11.4300, -15)  (11.4300, 0)  (22.5000, L)  (22.5000, L+15)
(-22.5000, L+15)  (-22.5000, L)  (-11.4300, 0)
```
* **Ports:** port 1 at z = −15 (WR-90 side), TE10, excited. On the wide end z = L + 15: port 2 TE10 (mode number 1)
  and port 3 TE30 (mode number 3). Reference planes z = 0 and z = L: **15 mm** each.
* **Frequency:** 15 GHz only.
* **Mesh:** maximum element size c_const/15[GHz]/10; mesh check with /20.
* **Files:** `E2_taper_L100.txt`, `E2_taper_L200.txt` (S11, S21, S31), and `_fine` for L = 100.

## E5: two chained 45° H-plane miters (priority 5, 2D)

Two plain (optical-mirror) 45° miters make a 90° turn, with a straight spacer between them. For a 45° deflection the
mirror is long (it makes 22.5° with the guide axis) and reaches 27.59 mm back along each arm from the apex, so the
ports sit 15 mm beyond the mirror ends. The spacer s is the centre-line length between the two miters' reference
planes (each 4.7345 mm from its apex). At s = 2a the two mirrors meet exactly; that is the shortest spacer for which
they do not overlap.

s = 2a = 45.72 mm:
```
( 0.0000, -42.5945)  ( 0.0000, -27.5945)  (22.8600, 27.5945)  (78.0489, 50.4545)  (93.0489, 50.4545)
(93.0489,  27.5945)  (55.1889, 27.5945)  (22.8600, -4.7345)  (22.8600, -42.5945)
```
s = 3a = 68.58 mm:
```
(  0.0000, -42.5945)  (  0.0000, -27.5945)  (22.8600, 27.5945)  (39.0245, 43.7589)  ( 94.2134, 66.6189)
(109.2134,  66.6189)  (109.2134, 43.7589)  (71.3534, 43.7589)  (22.8600, -4.7345)  ( 22.8600, -42.5945)
```
* **Ports:** port 1 on the bottom edge (z = −42.5945), port 2 on the right edge; TE10. Reference planes: z = −4.7345
  and x = 55.1889 (s = 2a) or x = 71.3534 (s = 3a): **37.86 mm** each.
* **Frequencies:** 30 and 56 GHz (60 GHz would sit 1.7 % above the TE90 cutoff).
* **Mesh:** maximum element size c_const/56[GHz]/10; mesh check /20 at 56 GHz.
* **Files:** `E5_two45_s2a.txt`, `E5_two45_s3a.txt`, `_fine` files.

## After you send the files

WP7b's `benchmarks/comsol/campaign_2/compare_campaign_2.py` compares them with the wave 2 components: E1 with the FEM
and mode-matching miters (WP3b, WP4b), E2 and E3 with the tapers and the strangled bend (WP5), E4 with the twist (WP6),
E5 with the chained-miter outline and the miter cascade (WP3b). The criterion is that of Gate A: the COMSOL error
shown by the mesh check plus 0.02 dB (CONVENTIONS §8), and for E4 the spurious power per mode.
