"""WF4 items 4-6: the registry check and the timing harness (benchmarks/check_registry.py, benchmarks/timings.py)."""
import importlib.util
import json
import os
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_registry_entries_are_well_formed():
    """Every status is in the vocabulary, every data file and script exists (the pytest ids are checked in CI by
    `python benchmarks/check_registry.py`, which collects the suite)."""
    cr = _load("check_registry", "benchmarks/check_registry.py")
    problems, n = cr.check(collect=False)
    assert problems == [] and n >= 80
    reg = json.loads((REPO / "benchmarks" / "registry.json").read_text(encoding="utf-8"))
    ids = {e["id"] for e in reg["entries"]}
    for i in ("gatea-comsol-c1", "gatea-comsol-c1b", "gatea-comsol-c2", "gatea-comsol-c3a", "gatea-comsol-c3b",
              "fem2d-marcuvitz-529b-corners", "wp4-review-fd-hplane-step", "wf4-doane-fig9-five-modes"):
        assert i in ids
    st = {e["id"]: e["status"] for e in reg["entries"]}
    assert st["wp4-marcuvitz-hplane-step"] == "superseded" and st["wp6-e4-twist"] == "superseded"
    assert not any("waiting for data" in e["status"] or "predictions ready" in e["status"] for e in reg["entries"])


def test_registry_check_finds_problems(tmp_path):
    cr = _load("check_registry", "benchmarks/check_registry.py")
    reg = json.loads((REPO / "benchmarks" / "registry.json").read_text(encoding="utf-8"))
    e = dict(reg["entries"][0])
    e.update(id="bad", status="waiting for data", data_file=["benchmarks/no_such_file.npz"], script=["nope.py"])
    reg["entries"] = [reg["entries"][0], e, dict(reg["entries"][0])]
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(reg))
    problems, _ = cr.check(p, collect=False)
    text = "\n".join(problems)
    assert "duplicate id" in text and "status 'waiting for data'" in text and "does not exist" in text


def test_comsol_log_parser():
    tm = _load("timings", "benchmarks/timings.py")
    text = ("E2_taper_parameter_sweep:\n[Nov 12, 2026, 4:28 AM] Number of degrees of freedom solved for: 27294.\n"
            "[Nov 12, 2026, 4:28 AM] Number of degrees of freedom solved for: 42,460.\n"
            "[Nov 12, 2026, 4:28 AM] Solution time (Study 1): 9 s.\n\nC1_miter90:\n"
            "[Nov 10, 2026] Complete mesh consists of 5000 domain elements, 10 boundary elements.\n"
            "[Nov 10, 2026] Number of degrees of freedom solved for: 1234.\n"
            "[Nov 10, 2026] Solution time (Study 1): 61 s. (1 minute, 1 second)\n")
    runs = tm.parse_comsol_log(text)
    assert runs["E2_taper_parameter_sweep"]["dofs"] == [27294, 42460]
    assert runs["E2_taper_parameter_sweep"]["solution_time_s"] == 9.0
    assert runs["C1_miter90"]["mesh_elements"] == 5000 and runs["C1_miter90"]["solution_time_s"] == 61.0
    assert tm.case_of("C1b_60miter") == "C1b" and tm.case_of("C3b_strangled_100mm") == "C3b"
    assert tm.case_of("C3a_step_h1_6") == "C3a" and tm.case_of("E4_twist_triangle") == "E4"
    rows = tm.comsol_table(tm.COMSOL_LOGS)
    by = {r["run"]: r for r in rows}
    assert by["E4_twist_triangle"]["export"] == "E4_twist_triangular"            # matched by name
    assert by["E2_taper_parameter_sweep"]["solves"] == 4 and by["E1_miter_E_plain"]["solves"] == 14
    assert all(r["case"] for r in rows) and len(rows) == 13


def test_timing_harness_machine_and_one_item(tmp_path):
    tm = _load("timings", "benchmarks/timings.py")
    m = tm.machine_info()
    for k in ("cpu", "logical_cores", "ram_gb", "os", "python", "numpy", "scipy", "blas", "wrmodes", "git_commit"):
        assert k in m
    out = tmp_path / "t.json"
    res = tm.main(["--quick", "--only", "lossy_straight_30GHz", "miter_mm_H_30GHz", "--out", str(out)])
    keys = [r["key"] for r in res["components"]]
    assert keys == ["lossy_straight_30GHz", "miter_mm_H_30GHz"]
    assert json.loads(out.read_text())["machine"]["wrmodes"] == m["wrmodes"]
    assert out.with_suffix(".md").exists() and res["components"][0]["blas_threads"] == 1
