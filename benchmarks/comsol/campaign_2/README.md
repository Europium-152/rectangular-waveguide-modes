# COMSOL campaign 2 — data (Gate B)

Run by Daniel in COMSOL 6.2.0.415 from the GUI, following `../CAMPAIGN_2.md` (11–12 Nov 2026).

* `raw/`: Global Evaluation tables (S-parameters of the rectangular ports). E2 and E5 are parameter sweeps
  (`mesh_factor` 10 / 20 = coarse / fine mesh, and `L` for E2). E4 has ports for every propagating mode at both ends.
* `java/`: the models, saved as Java (record only). `E1_taper_parameter_sweep.java` is the E2 model.
* `comsol_logs.txt`: degrees of freedom and solution times.
* `compare_campaign_2.py` and `RESULTS.md`: the comparison with wrmodes (WP7b's script, adapted by the coordinator to
  the actual export formats).

Status (15 Nov 2026): E1, E2, E5 pass. **E3 and E4 must be rerun**: in E3 the launched wave never enters the guide
(|S11| and |S21| about 1e-4, halving with the mesh: no port orientation set, unlike the other models); in E4 the
input face reports 100 % reflection and the main output mode 17–168 % (only one of the two PML blocks is in the PML
feature, and the port orientation on the input face is suspect). See the Gate B section of
`benchmarks/validation_v2/REPORT.md` when complete.
