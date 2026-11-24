"""WF4 (tasks/WF4-release-fixes.md): the common convergence record, SmoothBend's convergence study, internal
amplitudes of a line, absorbed power of a sweep, the LossyWallFilter quadrature. Report: docs/reports/WF4.md.
(Item 1 is tested in test_twist.py, item 3 in test_line.py, the timing harness and the registry in
test_wf4_tools.py.)"""
import os
import warnings

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

import wrmodes as wr
from wrmodes import Line, Guide, Straight, SmoothBend, SechCurvature
from wrmodes.components import Twist, Taper
from wrmodes.convergence import ConvergenceRecord, RECORD_KEYS
from wrmodes.modematch import Step, MiterMM, convergence_study as mm_convergence_study

WR90 = Guide(22.86e-3, 10.16e-3)
INCH = 0.0254


def _check_record(r, parameters):
    assert isinstance(r, ConvergenceRecord)
    for k in RECORD_KEYS:
        assert k in r, k
    n = len(r["values"])
    for k in ("parameter", "variants", "n_modes", "powers", "max_change", "wall_time"):
        assert len(r[k]) == n, k
    assert set(r["parameter"]) <= set(parameters)
    for p in r["powers"]:
        assert set(p) == {"transmitted", "reflected"} and p["transmitted"]
        total = sum(p["transmitted"].values()) + sum(p["reflected"].values())
        assert total <= 1 + 1e-9
    # the finest row of each (parameter, variant) has max_change 0
    last = {}
    for i, (p, v) in enumerate(zip(r["parameter"], r["variants"])):
        last[(p, tuple(sorted(v.items())))] = i
    assert all(r["max_change"][i] == 0.0 for i in last.values())
    assert all(t >= 0 for t in r["wall_time"])
    assert isinstance(r.table(), str) and len(r.rows()) == n


def test_twist_record_and_aliases():
    tw = Twist(90, 20 * INCH, "constant", guide=Guide(0.872 * INCH, 1.872 * INCH))
    r = tw.convergence_study(15e9, mode_factors=(1.0, 2.0), tails=(True, False))
    _check_record(r, ("mode_factor", "slices"))
    assert r["parameter"] == ["mode_factor"] * 4 + ["slices"]
    assert r["variants"][:4] == [{"tail": True}, {"tail": True}, {"tail": False}, {"tail": False}]
    assert r["richardson"]["order"] == 1 and r["richardson"]["variant"] == {"tail": False}
    # aliases of the earlier return value (one release)
    assert len(r["modes"]) == 4 and r["extrapolated"] is not None and r["change"] is not None and r["slices"]
    assert r["powers"][1]["transmitted"]["TE2,1"] == pytest.approx(r["modes"][1]["powers"]["TE2,1"], abs=0)


def test_taper_record():
    tp = Taper(0.1, 45e-3, 10.16e-3, "linear", guide=WR90)
    r = tp.convergence_study(15e9)
    _check_record(r, ("mode_factor", "slices"))
    assert r["input_mode"] == "TE1,0" and "rows" in r and True in r["richardson"]


def test_miter_step_and_fem_records():
    mm = MiterMM(plane="H").with_guide(WR90)
    r = mm.convergence_study(30e9, [8, 12, 16])
    _check_record(r, ("n_modes",))
    assert list(r["n_modes"]) == [8, 12, 16] and r["S"].shape[0] == 3 and r["max_change"][-1] == 0.0
    st = Step(22.86e-3, 10.16e-3, 22.86e-3, 5e-3, plane="H")
    r = st.convergence_study(12e9, [2.0, 4.0])
    _check_record(r, ("cutoff_factor",))
    assert r["n_modes"].shape == (2, 2) and list(r["cutoff_factor"]) == [2.0, 4.0]
    r2 = mm_convergence_study(st, 12e9, [2.0, 4.0], input_mode="TE1,0")
    assert r2["powers"] == r["powers"]


def test_fem2d_record():
    pytest.importorskip("skfem")
    pytest.importorskip("gmsh")
    from wrmodes.fem2d import Miter, convergence_study
    m = Miter(90, plane="H", guide=WR90)
    r = convergence_study(m, 10e9, mesh_densities=(4, 6), n_evanescent=(2, 4))
    _check_record(r, ("mesh_density", "n_evanescent"))
    assert r["parameter"] == ["mesh_density"] * 2 + ["n_evanescent"] * 2 and len(r["ndof"]) == 4
    assert "mesh" in r and "modes_complex" in r


def test_smoothbend_evanescent_convergence_study():
    """WF4 item 6: SmoothBend's study on an evanescent basis (internal mode factor 1.5-8, the same record)."""
    b = SmoothBend(0.05, SechCurvature(90, 0.05), plane="H", guide=WR90)
    r = b.convergence_study(20e9, mode_factors=(1.5, 2.0, 3.0))
    _check_record(r, ("mode_factor",))
    assert r["variants"][0] == {"route": "propagating"} and r["values"] == [1.0, 1.5, 2.0, 3.0]
    assert r["n_modes"][1] < r["n_modes"][2] < r["n_modes"][3]
    assert r["max_change"][2] < r["max_change"][1] < 2e-3
    with pytest.raises(ValueError):
        b.convergence_study(20e9, mode_factors=(1.1,), cutoff_factor=1.2)


def _step_line():
    return Line(WR90, [Straight(0.01), Step(22.86e-3, 10.16e-3, 22.86e-3, 1.6e-3), Straight(0.1),
                       Step(22.86e-3, 1.6e-3, 22.86e-3, 10.16e-3), SmoothBend(0.1, SechCurvature(60, 0.1)),
                       Straight(0.02)])


def test_internal_amplitudes_are_consistent_with_the_cascade():
    """WF4 item 6: at the ports the amplitudes are the GSM column; at every junction the net forward power over the
    propagating modes equals the transmitted power (lossless line)."""
    ln, f = _step_line(), 12e9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wr.line.EvanescentJunctionWarning)
        A = ln.internal_amplitudes(f, "TE1,0", cutoff_factor=2.0)
        G = ln.gsm(f, cutoff_factor=2.0)
    j = G.basis1.index("TE1,0")
    assert len(A.junctions) == len(ln.elements) + 1
    assert np.max(np.abs(A.junctions[-1]["forward"] - G.S21[:, j])) < 1e-13
    assert np.max(np.abs(A.junctions[0]["backward"] - G.S11[:, j])) < 1e-13
    T = np.sum(np.abs(G.S21[G.basis2.propagating_mask(f), j]) ** 2)
    for J in A.junctions:
        p = J["basis"].propagating_mask(f)
        assert abs(np.sum(np.abs(J["forward"][p]) ** 2) - np.sum(np.abs(J["backward"][p]) ** 2) - T) < 1e-12
    top = A.largest(3)
    assert top[0][0] >= top[1][0] >= top[2][0] and all(0 < t[1] < len(ln.elements) for t in top)


def test_internal_amplitudes_name_a_trapped_mode():
    """The recipe of the docstring on a resonator: the 1.6 mm x 100 mm strangled section of COMSOL C3b, at a
    transmission resonance of TE1,0 the amplitude inside the section exceeds the incident one."""
    ln = Line(WR90, [Step(22.86e-3, 10.16e-3, 22.86e-3, 1.6e-3), Straight(0.1), Step(22.86e-3, 1.6e-3, 22.86e-3, 10.16e-3)])
    fs = np.linspace(8.0e9, 12.0e9, 201)
    t = [abs(ln.gsm(f).S21[0, 0]) for f in fs]
    f0 = fs[int(np.argmax(t))]
    A = ln.internal_amplitudes(f0, "TE1,0")
    amp, k, left, right, mode, direction, prop = A.largest(1)[0]
    assert mode == "TE1,0" and prop and amp > 1.0 and "straight" in (left + right)


def test_sweep_absorbed_power():
    """WF4 item 6: SweepResult.absorbed_power = 1 - sum |S|^2 over the input column (wall loss)."""
    f = np.array([12e9, 15e9])
    lossless = Line(WR90, [SmoothBend(0.2, SechCurvature(90, 0.2))]).sweep(f, "TE1,0", processes=1)
    assert np.all(np.abs(lossless.absorbed_power) < 1e-12)
    g = Guide(22.86e-3, 10.16e-3, sigma=3.5e7)
    lossy = Line(g, [Straight(1.0)]).sweep(f, "TE1,0", processes=1)
    from wrmodes.loss import attenuation_constant
    alpha = np.array([attenuation_constant(("TE", 1, 0), g.w, g.h, x, 3.5e7) for x in f])
    assert np.allclose(lossy.absorbed_power, 1 - np.exp(-2 * alpha * 1.0), rtol=1e-4, atol=0)


def test_lossy_wall_filter_quadrature_resolves_320_modes():
    """WF4 item 2 (WP7b open issue 5): the u quadrature integrates products of profiles with up to M half-periods
    each, for M = 320 (the case that failed with 1.5 M + 48 points)."""
    from wrmodes.filters import lsu_quadrature_points
    for M in (80, 160, 320, 640):
        n = lsu_quadrature_points(M)
        x, w = leggauss(n)
        u = 0.5 * (x + 1)
        m = np.arange(M - 3, M + 1)
        S = np.sin(np.pi * m[:, None] * u[None, :])
        G = (S * 0.5 * w) @ S.T
        assert np.max(np.abs(G - 0.5 * np.eye(len(m)))) < 1e-12, M
    x, w = leggauss(int(1.5 * 320) + 48)                     # the rule before WF4
    u = 0.5 * (x + 1)
    s = np.sin(np.pi * 320 * u)
    assert abs((s * 0.5 * w) @ s - 0.5) > 1e-9


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("WRMODES_LONG"), reason="about 9 min and 3 GB: set WRMODES_LONG=1 to run")
def test_lossy_wall_filter_at_320_modes_per_family():
    """WF4 item 2: Butterweck's filter (WR-90, 75 GHz, 250 ohm, 15 cm) at M = 320 (RootFindingError before WF4);
    within 0.33 % of M = 160 in the attenuation of every port mode. About 9 minutes and 3 GB."""
    from wrmodes.filters import LossyWallFilter
    f = 75e9
    basis = WR90.basis(f, 1.0, "H")
    att = {}
    for M in (160, 320):
        g = LossyWallFilter(0.15, 250.0, plane="H", modes_per_family=M, guide=WR90).gsm(f, basis, basis)
        att[M] = -20 * np.log10(np.abs(np.diag(g.S21)))
        assert g.reciprocity_defect() < 1e-10 and g.passivity_margin() >= -1e-12
    assert np.max(np.abs(att[320] - att[160]) / att[160]) < 5e-3
