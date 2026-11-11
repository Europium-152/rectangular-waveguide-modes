# COMSOL campaign 1 — data

Run by Daniel in COMSOL 6.2.0.415 from the GUI, following `../CAMPAIGN_1.md`.

* `raw/`: COMSOL global-value exports (emw.S11, emw.S21 of the TE10 ports). `C2_constant_aluminum_firstrow.txt` is
  the header and first row of a 243 MB export in which every mesh node carried the same global value.
  `*_tol1eminus4` files: re-solved with stationary relative tolerance 1e-4 (no change, see the report).
  `C2_constant_PEC_tol1eminus4`: the constant-curvature model with the impedance boundary disabled.
* `java/`: the models, saved as Java from the GUI, for the record (not meant to be rerun as scripts).
* `compare_campaign_1.py`: Gate A comparison; `RESULTS.md`: its output.

Gate A summary: `benchmarks/validation_v1/REPORT.md`, section "COMSOL campaign 1 — Gate A".
