# COMSOL campaign for Gate A (WP7): C1–C3

> **Superseded by `CAMPAIGN_1.md`.** The generated Java scripts below did not run in COMSOL (Daniel, 9 Nov 2026);
> COMSOL is driven through its GUI from the written instructions in `CAMPAIGN_1.md` instead. This file and the
> scripts are kept for the record.

Three COMSOL models that check the wave-1 components of wrmodes 1.0, plus a probe. They were written by the WP7
verifier **without access to COMSOL**. They compile against stand-ins for the COMSOL API, and a dry run with those
stand-ins produces output files that the post-processing reads. They have **not been run in COMSOL**. The calls
follow the existing export in this folder (COMSOL 6.2.0.415) and the COMSOL 6.2 API reference. The few property
names that could not be checked are looked up at run time (see "If something fails").

| File | Case | Size (estimate) |
| --- | --- | --- |
| `WP7_C0_probe.java` | Writes the property names of a rectangular port and an impedance boundary to `WP7_C0_probe.txt`. Nothing is solved. | seconds |
| `WP7_C1_h_miter.java` | WR-90 H-plane plain 90° miter (mirror through the crossing point of the arm axes, d/a = 0.7071), 3D, 43 frequencies: 18–30 GHz in 1 GHz steps, 32–90 GHz in 2 GHz steps | small model; 4 extra solves at 18 GHz pick the port orientation |
| `WP7_C2_sech_bend_aluminium.java` | 60 cm, 60° hyperbolic-secant H-plane bend in WR-90 with impedance walls (aluminium, σ = 3.5×10⁷ S/m), 10 frequencies: 18, 21, 23, 25, 28, 30, 31.5, 34, 36, 38 GHz | large: about 2–5 million unknowns at 38 GHz with 8 elements per wavelength |
| `WP7_C3_strangling_step.java` | Centred step WR-90 (22.86 × 10.16 mm) ↔ 22.86 × 1.6 mm, 5 frequencies (20, 25, 40, 55, 75 GHz), TE10 excited from each side | small |

The frequencies of C2 stop at 38 GHz. A 60 cm guide meshed for 90 GHz would need tens of millions of unknowns. The
list avoids frequencies within 1 % of a TE_m0 cutoff.

## What to run

1. **Check the version.** These scripts target COMSOL **6.2** with the RF Module, the version of the existing export.
   If your version is different, say so when you send the results.
2. Put the four `.java` files in an empty folder, open a terminal there (the COMSOL `bin` folder on the path), and
   run:

   ```
   comsolcompile WP7_C0_probe.java
   comsolbatch -inputfile WP7_C0_probe.class
   comsolcompile WP7_C3_strangling_step.java
   comsolbatch -inputfile WP7_C3_strangling_step.class
   comsolcompile WP7_C1_h_miter.java
   comsolbatch -inputfile WP7_C1_h_miter.class
   comsolcompile WP7_C2_sech_bend_aluminium.java
   comsolbatch -inputfile WP7_C2_sech_bend_aluminium.class
   ```

   Run them in this order. C3 is the quickest real check. Outputs go to the working folder. To send them elsewhere,
   add `-Dwp7.out=<folder>` to the Java options. On Windows, `comsolbatch.exe` and `comsolcompile.exe` are in
   `C:\Program Files\COMSOL\COMSOL62\Multiphysics\bin\win64`.
3. You can also open each compiled `.class` file in the COMSOL Desktop (File > Open), which calls `run()`.

## What to send back

From the working folder, for each case that ran (`<k>` = 1, 2, 3):

* `WP7_C<k>_case.json`: the geometry, sampling planes and frames;
* `WP7_C<k>_fields.csv`: E and H at the sampling points (C1 about 25 MB, C2 about 2 MB, C3 about 4 MB; zip them);
* `WP7_C<k>_sparams.csv`: COMSOL's own TE10 S-parameters;
* `WP7_C<k>.log`;
* `WP7_C0_probe.txt` and `WP7_C0.log`, always;
* the batch log if a run stopped with an error.

The `.mph` files are saved too (`WP7_C<k>.mph`, last frequency only). They are not needed, but keep them for
inspection.

The coordinator then runs:

```
python benchmarks/comsol/wp7_postprocess.py <folder>           # -> benchmarks/comsol/data/WP7_C<k>.npz
python benchmarks/validation_v1/compare_comsol.py               # -> results/comsol_comparison.json
```

## How the S-parameters are obtained

The GSM entries are **not** taken from COMSOL's port mode fields, whose sign and phase conventions differ from
`CONVENTIONS.md` and could not be checked here. Instead, each model writes E and H at Gauss–Legendre points of one
sampling plane per port, at a known position in the port's local frame (CONVENTIONS.md §2). The post-processing
then projects them on the normalized fields of `wrmodes.modes.transverse_field` (§4–5):

* V_p = ∫E_t·e_p ds = (a⁺+a⁻)√(2/Y_p);
* I_p = ∫H_t·(ẑ×e_p) ds = (a⁺−a⁻)√(2Y_p).

This gives the forward and backward amplitude of every propagating mode at each port. The amplitudes are moved to
the reference planes with exp(∓iβ·offset): the leads are straight and lossless where the offset is not zero. The
column of the GSM for TE10 incidence then follows in the phase and sign conventions of `CONVENTIONS.md`:
S11[:, TE10] = a⁻(port 1)/a⁺_TE10(port 1) and S21[:, TE10] = a⁺(port 2)/a⁺_TE10(port 1). For C3, TE10 incident
from port 2 also gives S12 and S22.

COMSOL's own TE10 S-parameters are exported as a check. They must agree in magnitude.

Each model's sampling planes and reference planes:

| Case | Reference planes (CONVENTIONS.md §6) | Sampling planes | Local frames |
| --- | --- | --- | --- |
| C1 | y = −a/2 and x = −a/2, through the inner corner (−a/2, −a/2) | 15 mm into each lead (offset −15 mm at port 1, +15 mm at port 2), away from the inner-corner field singularity | port 1: u = +x, v = −z, z = +y; port 2: u = +y, v = −z, z = −x |
| C2 | the two end planes of the bend | the end planes themselves (offset 0, so the lossy leads are not de-embedded) | port 1: u = −y, v = −z, z = +x; port 2 rotated by 60° about v |
| C3 | the junction plane x = 0 | 15 mm into the large guide, 10 mm into the small one | u = +y, v = +z, z = +x at both ports |

## Terminations and their check

* **C1: domain-backed slit ports (TE10) backed by PML.** This is the arrangement of the existing export. The PML
  absorbs the higher-order modes the miter makes. Up to 13 TE_m0 modes propagate at 90 GHz, too many for the
  single-digit mode numbers of rectangular ports. A slit port launches its wave in the direction set by
  `PortOrientation`, which depends on how COMSOL orients the interior face. The script therefore solves at 18 GHz
  once for every allowed combination of orientations of the two ports, keeps the combination with the largest
  |S11|² + |S21|² ≤ 1, and logs all of them.
* **C2 and C3: exterior ports, one per propagating mode that the component couples to TE10, no PML.** C2 has TE_m0
  for m ≤ 5. C3 has TE10 on both sides, plus TE12, TM12, TE14 and TM14 in WR-90. Each is activated only at the
  frequencies where its mode propagates.
* **Check:** the post-processing reports the largest amplitude travelling *towards* the component other than the
  excitation (`spurious`). It comes from imperfect absorption by the PML or the ports. Values below about 1e-3 of the
  incident amplitude mean the termination is good enough for the 0.02 dB target.

## If something fails

* **"WP7: feature … has none of the properties …"** The names of the rectangular-port mode settings (mode type and
  mode number) could not be checked without COMSOL. The script looks them up among the port's properties, and the
  message lists the properties it found. Send `WP7_C0_probe.txt`: the fix is a one-word change in
  `benchmarks/comsol/wp7/java_common.txt`, followed by `python benchmarks/comsol/wp7/generate_java.py`.
* **The C1 orientation step stops** ("no port orientation launches the TE10 wave"). Send `WP7_C1.log`.
* **A geometry error in C2.** The bend is drawn with the parametric-curve construction of the existing E-bend
  export, extended by straight leads (the heading is clamped outside the bend). Send the batch log.
* **Memory or run time in C2.** Lower `ppw` (elements per wavelength) from 8 to 6 in the model parameters, or drop
  the frequencies above 31.5 GHz, and say which settings you used.

## Mesh

The models use free tetrahedral meshes with h_max = c/(f_mesh·ppw). They are re-meshed when the frequency moves to
the next band (C1: 30/45/60/75/90 GHz; C2: 25/31.5/38 GHz; C3: 25/40/55/75 GHz). The defaults are ppw = 8 (C1, C2)
and ppw = 10 (C3). One convergence check costs a single extra run: C3 with `ppw` = 14 (change the `param` line in
the Java file, or the generator). If you can, run it and send its outputs in a separate folder.
