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


def stub(case, variant, f):
    """Smooth made-up reference-plane values; unitary enough to look like data."""
    ph = np.exp(-1j * 0.3 * f)
    if case == "E4":
        p = {"S21": 0.995, "S31": 0.04, "S41": 0.01, "S51": 0.06, "S61": 0.03, "S71": 0.02, "S81": 0.015}
        return {"S11": 0.01 * ph, **{k: v * ph for k, v in p.items()}}
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


def test_e2_taper_three_ports_and_te30_power(tmp_path):
    write_export(tmp_path / "E2_taper_L100.txt", comsol_from("E2", "L100", [15.0]))
    _, _, v = cc2.run(["E2"], raw=tmp_path, model=stub)
    assert len(v) == 2 and all(x[-1] for x in v)
    write_export(tmp_path / "E2_taper_L100.txt", comsol_from("E2", "L100", [15.0], err_mode="S31", err_power=3e-4))
    _, _, v = cc2.run(["E2"], raw=tmp_path, model=stub)
    te30 = [x for x in v if x[2] == "TE30"][0]
    assert not te30[-1] and abs(te30[4] + 3e-4) < 1e-12


def test_e4_judged_on_power_per_output_mode(tmp_path):
    """Eight ports (S11 ... S81). A 3e-4 error planted in TE1,2 (port 5) fails only that mode; with a fine run that
    differs by 2.5e-4 in that power the allowance becomes 3.5e-4 and it passes."""
    write_export(tmp_path / "E4_twist_triangular.txt",
                 comsol_from("E4", "triangular", [15.0, 18.0], err_mode="S51", err_power=3e-4))
    _, _, v = cc2.run(["E4"], raw=tmp_path, model=stub)
    assert len(v) == 2 * 7
    bad = [x for x in v if not x[-1]]
    assert [x[2] for x in bad] == ["TE1,2", "TE1,2"]
    write_export(tmp_path / "E4_twist_triangular_fine.txt",
                 comsol_from("E4", "triangular", [15.0, 18.0], err_mode="S51", err_power=0.5e-4))
    _, _, v = cc2.run(["E4"], raw=tmp_path, model=stub)
    assert all(x[-1] for x in v)


def test_e3_aluminium_uses_the_pec_mesh_check(tmp_path):
    write_export(tmp_path / "E3_strangled_PEC.txt", comsol_from("E3", "PEC", [18.0, 40.0], err_db=0.01))
    write_export(tmp_path / "E3_strangled_PEC_fine.txt", comsol_from("E3", "PEC", [18.0]))
    write_export(tmp_path / "E3_strangled_aluminium.txt", comsol_from("E3", "aluminium", [18.0, 40.0], err_db=0.025))
    _, _, v = cc2.run(["E3"], raw=tmp_path, model=stub)
    al = {x[3]: x for x in v if x[1] == "aluminium"}
    assert al[18.0][-1] and abs(al[18.0][5] - 0.03) < 1e-9    # 0.01 mesh change + 0.02
    assert not al[40.0][-1]                                  # no mesh check at 40 GHz: 0.02 only


def test_wrmodes_model_runs_for_e2_and_e4():
    """The real model (merged WP5 and WP6 components) returns finite values with the expected keys."""
    m = cc2.wrmodes_model("E2", "L100", 15.0)
    assert set(m) == {"S11", "S21", "S31"} and all(np.isfinite(abs(v)) for v in m.values())
    m = cc2.wrmodes_model("E4", "triangular", 18.0)
    assert set(m) == {"S11", *cc2.E4_PORTS} and all(np.isfinite(abs(v)) for v in m.values())
