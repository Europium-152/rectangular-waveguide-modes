"""Tests of the COMSOL campaign 2 post-processing (benchmarks/comsol/campaign_2/compare_campaign_2.py), card WP7b
item 6, on small synthetic exports in the campaign 1 format written here (no fake data is committed). A stub model
replaces the wrmodes components, so that parsing, multi-port columns, de-embedding, the Gate A criterion, the E4
per-mode judgement and the handling of missing files are tested without running the solvers.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("compare_campaign_2",
                                              ROOT / "benchmarks" / "comsol" / "campaign_2" / "compare_campaign_2.py")
cc2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc2)


def _fmt(z):
    return f"{z.real:.17g}{z.imag:+.17g}i"


def write_export(path, data, table=False):
    """data: {f_GHz: {'S11': complex, ...}} -> a COMSOL-style export (global row, or table if table=True)."""
    lines = ["% Model:              synthetic.mph", "% Version:            COMSOL 6.2.0.415",
             "% Date:               Nov 15 2026, 10:00", "% Dimension:          3", "% Nodes:              1",
             "% Description:        S-parameters", "% Length unit:        m"]
    keys = sorted(next(iter(data.values())))
    if table:
        lines.append("% freq (GHz)              " + "  ".join(f"emw.{k} (1)" for k in keys))
        for f, row in data.items():
            lines.append(f"{f:<25g}" + "  ".join(_fmt(row[k]) for k in keys))
    else:
        cols = [f"emw.{k} (1) @ freq={f:g}" for f in data for k in keys]
        lines.append("% X                       Y                        " + "   ".join(cols))
        vals = [_fmt(data[f][k]) for f in data for k in keys]
        lines.append("0                         -0.02                    " + "   ".join(vals))
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def write_table(path, rows, params=()):
    """Daniel's Global Evaluation table format (campaign 2): optional parameter columns, 'freq (GHz)', then
    'S11 (1)' ... columns. rows: [(param values..., f_GHz, {'S11': complex, ...}), ...]."""
    keys = list(rows[0][-1])
    head = "% " + "".join(f"{c:<48}" for c in (*params, "freq (GHz)", *(f"{k} (1)" for k in keys)))
    lines = ["% Model:              synthetic.mph", "% Version:            COMSOL 6.2.0.415",
             "% Date:               Nov 15 2026, 10:00", "% Table:              Table 1 - Global Evaluation 1", head]
    for r in rows:
        lines.append("".join(f"{x!s:<50}" for x in r[:-1]) + "".join(f"{_fmt(r[-1][k]):<50}" for k in keys))
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def stub(case, variant, f):
    """Smooth made-up reference-plane values; unitary enough to look like data."""
    ph = np.exp(-1j * 0.3 * f)
    if case == "E4":
        return {"out": {"TE1,0": 0.99, "TE1,2": 0.006, "TE0,3": 0.002}, "in": {"TE1,0": 1e-8}}
    if case == "E2":
        return {"S11": 0.002 * ph, "S21": 0.99 * ph, "S31": 0.05 * ph}
    return {"S11": 0.05 * ph, "S21": 0.98 * ph}


def comsol_from(case, variant, freqs, err_db=0.0, err_mode=None, err_power=0.0):
    """Port-plane values that COMSOL would export if it agreed with the stub (plus planted errors)."""
    out = {}
    for f in freqs:
        m = stub(case, variant, f)
        if case in ("E1", "E3", "E5"):
            b = cc2.beta("TE1,0", cc2.W, cc2.H, f)
            d = 37.86e-3 if case == "E5" else 15e-3
            row = {k: cc2.deembed(v, b, d, b, d) for k, v in m.items()}
        elif case == "E2":
            b1, b2, b3 = (cc2.beta("TE1,0", cc2.W, cc2.H, f), cc2.beta("TE1,0", 45e-3, cc2.H, f),
                          cc2.beta("TE3,0", 45e-3, cc2.H, f))
            row = {"S11": cc2.deembed(m["S11"], b1, 15e-3, b1, 15e-3),
                   "S21": cc2.deembed(m["S21"], b2, 15e-3, b1, 15e-3),
                   "S31": cc2.deembed(m["S31"], b3, 15e-3, b1, 15e-3)}
        else:
            row = dict(m)
        row["S21"] = row["S21"] * 10 ** (err_db / 20)
        if err_mode:
            v = row[err_mode]
            row[err_mode] = v * np.sqrt((abs(v) ** 2 + err_power) / abs(v) ** 2)
        out[f] = row
    return out


def test_read_multiport_global_export_and_table(tmp_path):
    data = {15.0: {f"S{k}1": complex(0.1 * k, -0.01 * k) for k in range(1, 9)},
            18.0: {f"S{k}1": complex(-0.05 * k, 0.02 * k) for k in range(1, 9)}}
    write_export(tmp_path / "a.txt", data)
    write_export(tmp_path / "b.txt", data, table=True)
    for name in ("a.txt", "b.txt"):
        r = cc2.read(tmp_path / name)
        assert sorted(r) == [15.0, 18.0]
        assert sorted(r[18.0]) == sorted(data[18.0])
        for f in data:
            for k in data[f]:
                assert abs(r[f][k] - data[f][k]) < 1e-15


def test_missing_files_are_skipped_with_a_message(tmp_path, capsys):
    out, log, verdicts = cc2.run(["E1", "E2", "E3", "E4", "E5"], raw=tmp_path, model=stub)
    assert verdicts == []
    assert any("E3_strangled_PEC.txt" in m and m.startswith("[skip]") for m in log)
    assert "[skip] E4_twist_triangular.txt" in capsys.readouterr().out


def test_deembedding_and_gate_a_criterion(tmp_path):
    """Exact agreement passes; 0.03 dB fails without a mesh check; with a fine run 0.015 dB away the allowance is
    0.035 dB and the same 0.03 dB passes."""
    freqs = [8.0, 10.0]
    write_export(tmp_path / "E1_miter_E_plain.txt", comsol_from("E1", "plain_FEM", freqs))
    _, _, v = cc2.run(["E1"], raw=tmp_path, model=stub)
    assert v and all(x[-1] for x in v) and max(abs(x[4]) for x in v) < 1e-9
    write_export(tmp_path / "E1_miter_E_plain.txt", comsol_from("E1", "plain_FEM", freqs, err_db=0.03))
    _, _, v = cc2.run(["E1"], raw=tmp_path, model=stub)
    assert not any(x[-1] for x in v)
    write_export(tmp_path / "E1_miter_E_plain_fine.txt", comsol_from("E1", "plain_FEM", freqs, err_db=0.03 - 0.015))
    _, _, v = cc2.run(["E1"], raw=tmp_path, model=stub)
    assert all(x[-1] for x in v)                       # |delta| = 0.015 vs fine, allowed 0.015 + 0.02


def _e2_rows(err_power=0.0):
    """The parameter sweep of campaign 2: mesh_factor 10 and 20, L = 0.1 and 0.2 m; ports 1-2 narrow end (TE10,
    TE30), 3-4 wide end (TE10, TE30), so wrmodes' S21/S31 are COMSOL's S31/S41."""
    rows = []
    for mf in (10, 20):
        for L, var in ((0.1, "L100"), (0.2, "L200")):
            r = comsol_from("E2", var, [15.0], err_mode="S31", err_power=err_power)[15.0]
            rows.append((mf, L, 15.0, {"S11": r["S11"], "S21": 0j, "S31": r["S21"], "S41": r["S31"]}))
    return rows


def test_e2_taper_three_ports_and_te30_power(tmp_path):
    write_table(tmp_path / "E2_taper_parameter_sweep.txt", _e2_rows(), params=("mesh_factor", "L (m)"))
    _, _, v = cc2.run(["E2"], raw=tmp_path, model=stub)
    assert len(v) == 4 and all(x[-1] for x in v)
    write_table(tmp_path / "E2_taper_parameter_sweep.txt", _e2_rows(3e-4), params=("mesh_factor", "L (m)"))
    _, _, v = cc2.run(["E2"], raw=tmp_path, model=stub)
    te30 = [x for x in v if x[2] == "TE30"][0]
    assert not te30[-1] and abs(te30[4] + 3e-4) < 1e-12


E4_JAVA = """
    model.component("comp1").physics("emw").feature("port1").selection().set(18);
    model.component("comp1").physics("emw").feature("port2").selection().set(18);
    model.component("comp1").physics("emw").feature("port3").selection().set(8);
    model.component("comp1").physics("emw").feature("port4").selection().set(8);
    model.component("comp1").physics("emw").feature("port5").selection().set(8);
    model.component("comp1").physics("emw").feature("port1").set("PortExcitation", "off");
    model.component("comp1").physics("emw").feature("port2").set("PortExcitation", "on");
    model.component("comp1").physics("emw").feature("port2").set("PortModeNumber", "01");
    model.component("comp1").physics("emw").feature("port3").set("PortModeNumber", "01");
    model.component("comp1").physics("emw").feature("port4").set("PortModeNumber", "21");
    model.component("comp1").physics("emw").feature("port5").set("PortModeNumber", "30");
"""


def _e4_export(path, err=0.0):
    """Ports 1-2 on the input face (2 excited, COMSOL '01' = TE1,0), 3-5 on the output face: COMSOL (m, n) is
    wrmodes n,m, so port 4 'TE21' is TE1,2 and port 5 'TE30' is TE0,3."""
    rows = []
    for f in (15.0, 18.0):
        p = {"S12": 1e-4, "S22": 1e-4, "S32": 0.99, "S42": 0.006 + err, "S52": 0.002}
        rows.append((f, {k: complex(np.sqrt(v), 0.0) for k, v in p.items()}))
    write_table(path, rows)


def test_e4_judged_on_power_per_output_mode(tmp_path):
    """A 3e-4 error planted in TE1,2 fails only that mode; a fine run 2.5e-4 away in that power widens the allowance
    to 3.5e-4 and it passes. The port layout comes from the model's .java."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (tmp_path / "java").mkdir()
    (tmp_path / "java" / "E4_twist_triangular.java").write_text(E4_JAVA)
    _e4_export(raw / "E4_twist_triangular.txt", err=3e-4)
    out, _, v = cc2.run(["E4"], raw=raw, model=stub)
    assert sorted({x[2] for x in v}) == ["TE0,3", "TE1,0", "TE1,2"]
    assert [x[2] for x in v if not x[-1]] == ["TE1,2", "TE1,2"]
    _e4_export(raw / "E4_twist_triangular_fine.txt", err=0.5e-4)
    _, _, v = cc2.run(["E4"], raw=raw, model=stub)
    assert all(x[-1] for x in v)


def test_e3_aluminium_uses_the_pec_mesh_check(tmp_path):
    write_export(tmp_path / "E3_strangled_PEC.txt", comsol_from("E3", "PEC", [18.0, 40.0], err_db=0.01))
    write_export(tmp_path / "E3_strangled_PEC_fine.txt", comsol_from("E3", "PEC", [18.0]))
    write_export(tmp_path / "E3_strangled_aluminum.txt", comsol_from("E3", "aluminium", [18.0, 40.0], err_db=0.025))
    _, _, v = cc2.run(["E3"], raw=tmp_path, model=stub)
    al = {x[3]: x for x in v if x[1] == "aluminium"}
    assert al[18.0][-1] and abs(al[18.0][5] - 0.03) < 1e-9    # 0.01 mesh change + 0.02
    assert not al[40.0][-1]                                  # no mesh check at 40 GHz: 0.02 only


def test_wrmodes_model_runs_for_e2_and_e4():
    """The real model (merged WP5 and WP6 components) returns finite values with the expected keys."""
    m = cc2.wrmodes_model("E2", "L100", 15.0)
    assert set(m) == {"S11", "S21", "S31"} and all(np.isfinite(abs(v)) for v in m.values())
    m = cc2.wrmodes_model("E4", "triangular", 18.0)
    assert set(m) == {"in", "out"} and abs(sum(m["out"].values()) + sum(m["in"].values()) - 1) < 1e-9
