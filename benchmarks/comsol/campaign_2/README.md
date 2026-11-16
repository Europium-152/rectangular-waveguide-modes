# COMSOL campaign 2 — data (Gate B)

Run by Daniel in COMSOL 6.2.0.415 from the GUI, following `../CAMPAIGN_2.md` (11–12 Nov 2026).

* `raw/`: Global Evaluation tables (S-parameters of the rectangular ports). E2 and E5 are parameter sweeps
  (`mesh_factor` 10 / 20 = coarse / fine mesh, and `L` for E2). E4 has ports for every propagating mode at both ends.
* `java/`: the models, saved as Java (record only). `E1_taper_parameter_sweep.java` is the E2 model.
* `comsol_logs.txt`: degrees of freedom and solution times.
* `compare_campaign_2.py` and `RESULTS.md`: the comparison with wrmodes (WP7b's script, adapted by the coordinator to
  the actual export formats).

Status (16 Nov 2026): all five cases pass (122 comparisons); E3 and E4 were rerun on 16 Nov after the first pass
showed a port-orientation problem (E3) and a missing PML block and port orientation (E4); the first-pass E3/E4 files
were replaced. Gate B summary: the "Gate B" section of `benchmarks/validation_v2/REPORT.md`.
