"""Self-checks of the WP7 references (benchmarks/validation_v1): they must be right before they judge anyone.

* frame relabelling: analytic derivation = quadrature of transverse_field = wrmodes.modes.eplane_frame_signs;
* global-ODE line reference = wrmodes 0.1 product of transfer matrices (solver at atol = 1e-10) to 1e-9 in power
  (WP7 acceptance), on cheap cases; the stored reference data are checked the same way when present;
* surface-integral loss = textbook closed forms; WP2 card numbers (0.12 / 0.23 dB/m);
* Marcuvitz 5.26a transcription: eq. (2a) against its asymptotic forms (2c) and (2d);
* the L1 CSV reader and comparison on a synthetic file.
"""
import os

import numpy as np
import pytest

import ref_common
import ref_frames
import ref_line
import ref_literature as L
import ref_loss
from wrmodes.modes import Mode, ModeBasis, eplane_frame_signs

WR90 = ref_line.WR90


# ----------------------------------------------------------------------------------------- frames
@pytest.mark.parametrize("w,h,f", [(*WR90, 70e9), (0.02, 0.02, 40e9), (0.04, 0.01, 60e9)])
def test_eplane_relabelling_derivation(w, h, f):
    gb = ModeBasis.propagating(f, w, h)
    lb = ref_frames.eplane_local_basis(gb)
    F = ref_frames.eplane_matrix(gb, lb)
    assert np.allclose(ref_frames.overlap_matrix(gb, lb, nq=64), F, atol=1e-12)
    assert np.array_equal(F, eplane_frame_signs(gb, lb))       # CONVENTIONS.md 2 implementation agrees
    assert np.allclose(F @ F.T, np.eye(len(gb)))


def test_eplane_examples_of_conventions():
    """TE10 -> -TE01, TE20 -> +TE02, TM11 -> +TM11, TM21 -> -TM12 (CONVENTIONS.md 2)."""
    for g, loc, s in [(("TE", 1, 0), ("TE", 0, 1), -1), (("TE", 2, 0), ("TE", 0, 2), 1),
                      (("TM", 1, 1), ("TM", 1, 1), 1), (("TM", 2, 1), ("TM", 1, 2), -1)]:
        assert ref_frames.local_mode(g) == Mode(*loc)
        assert ref_frames.eplane_sign(g) == s


# ----------------------------------------------------------------------------------------- lines
def _dP(S, S01):
    return max(np.max(np.abs(np.abs(S[k]) ** 2 - np.abs(S01[k]) ** 2)) for k in S)


@pytest.mark.parametrize("name", ["same-plane", "mixed-planes", "long-straights"])
def test_line_reference_matches_v01_at_20GHz(name):
    segs = ref_line.reference_lines()[name]
    _, _, S, _ = ref_line.line_transfer(20e9, *WR90, segs)
    _, _, S01 = ref_line.line_transfer_v01(20e9, *WR90, segs, atol=1e-10)
    assert _dP(S, S01) < 1e-9


def test_line_reference_single_bend_equals_v01_solver():
    """One E-plane sech bend in a line = the 0.1 solver run on the local h x w guide, relabelled."""
    seg = [ref_line.bend(0.15, ref_line.sech_curvature(60, 0.15), "E")]
    f = 35e9
    gb, _, S, _ = ref_line.line_transfer(f, *WR90, seg)
    _, _, S01 = ref_line.line_transfer_v01(f, *WR90, seg, atol=1e-10)
    assert _dP(S, S01) < 1e-9
    # unitarity and reciprocity of the reference itself
    Sf = np.block([[S["S11"], S["S12"]], [S["S21"], S["S22"]]])
    assert np.max(np.abs(Sf.conj().T @ Sf - np.eye(len(Sf)))) < 1e-9
    assert np.max(np.abs(Sf - Sf.T)) < 1e-9


def test_straight_line_is_exact_phase():
    f, L_ = 90e9, 20.0
    gb, _, S, _ = ref_line.line_transfer(f, *WR90, [ref_line.straight(L_)])
    # beta L = 3.8e4 rad: one ulp of beta (2e-16 relative) is 1e-11 rad of phase, so 1e-10 is the attainable level
    assert np.allclose(np.diag(S["S21"]), np.exp(-1j * gb.beta(f) * L_), rtol=0, atol=1e-10)
    assert np.max(np.abs(S["S11"])) == 0


def test_stored_line_reference_agrees_with_v01():
    path = os.path.join(ref_common.DATA, "line_reference.npz")
    if not os.path.exists(path):
        pytest.skip("run benchmarks/validation_v1/make_references.py")
    data = ref_common.load_npz(path)
    worst = max(float(v) for k, v in data.items() if k.endswith("dP_vs_v01"))
    assert worst < 1e-9


# ----------------------------------------------------------------------------------------- loss
@pytest.mark.parametrize("g", list(ref_loss.GUIDES))
def test_loss_reference_equals_textbook(g):
    a, b = ref_loss.GUIDES[g]
    f = ref_loss.FREQUENCIES[g][2]
    basis = ModeBasis.propagating(f, a, b)
    A, _ = ref_loss.loss_matrix(basis, f, ref_loss.SIGMA_AL)
    tb = np.array([ref_loss.textbook_alpha(m, a, b, f, ref_loss.SIGMA_AL) for m in basis.modes])
    assert np.max(np.abs(np.diag(A) / tb - 1)) < 1e-9
    assert np.allclose(A, A.T)
    assert np.all(np.linalg.eigvalsh(A) > 0)                # wall loss is positive definite


def test_wp2_card_numbers_for_te10_in_aluminium():
    a, b = WR90
    for f, val in ((18e9, 0.12), (90e9, 0.23)):
        assert round(ref_loss.alpha(("TE", 1, 0), a, b, f, ref_loss.SIGMA_AL) * ref_loss.NP_TO_DB, 2) == val


def test_power_normalization_of_reference_fields():
    a, b = WR90
    for m in [("TE", 1, 0), ("TE", 0, 2), ("TE", 3, 1), ("TM", 2, 1)]:
        assert abs(ref_loss.power(m, a, b, 70e9) - 1) < 1e-12


# ----------------------------------------------------------------------------------------- Marcuvitz
def test_marcuvitz_2a_reduces_to_its_asymptotic_forms():
    a, b = 22.86e-3, 10.16e-3
    f = 12e9                                               # b / lambda_g = 0.34
    B, _, _ = L.marcuvitz_eplane_step(f, a, b, 0.05 * b)
    assert abs(B / L.marcuvitz_eplane_step_2c(f, a, b, 0.05 * b) - 1) < 0.01       # (2c), alpha << 1
    d = 0.05
    B, _, S = L.marcuvitz_eplane_step(f, a, b, (1 - d) * b)
    k = 2 * np.pi * f / 299792458.0
    x = b / (2 * np.pi / np.sqrt(k * k - (np.pi / a) ** 2))
    B2d = 2 * x * (d / 2) ** 2 * (2 * np.log(2 / d) / (1 - d) + 1 + 17 / 16 * x * x)
    assert abs(B / B2d - 1) < 0.03                                                    # (2d), delta << 1
    assert abs(S) < 0.05


# ----------------------------------------------------------------------------------------- literature CSV
SYNTH = """# reference: synthetic test file (WP7), not literature
# figure: 4
# axes: x = normalized frequency f/fc, y = |S11| (dB)
# b/a: 0.5
normalized_frequency,S11_dB,uncertainty,x_uncertainty,curve
1.2,{y1},0.3,0.0,d/b = 0.87
1.5,{y2},0.3,0.005,d/b = 0.87
1.2,{y3},0.5,0.0,d/b = 1.41
"""


def _fake_miter(f, chamfer, plane, a, b):
    s11 = 0.1 * chamfer * (f / 1e10)                      # any smooth model
    return s11, np.sqrt(1 - s11 ** 2)


def test_csv_reader_and_comparison(tmp_path):
    a = 22.86e-3
    fc = L.cutoff_te10(a)
    y = lambda x, d: 20 * np.log10(0.1 * d * x * fc / 1e10)          # noqa: E731
    p = tmp_path / "alessandri1994_fig4.csv"
    p.write_text(SYNTH.format(y1=y(1.2, 0.87) + 0.2, y2=y(1.5, 0.87) - 0.25, y3=y(1.2, 1.41) + 0.4))
    curves = L.read_csv(str(p))
    assert [c.label for c in curves] == ["d/b = 0.87", "d/b = 1.41"]
    assert curves[0].parameter() == 0.87 and curves[0].header["b/a"] == "0.5"
    rows = L.compare_miter_file(str(p), _fake_miter)
    assert all(r["ok"] for r in rows)
    assert rows[0]["n"] == 2 and rows[0]["worst_ratio"] < 1
    # a point outside its uncertainty must fail
    p.write_text(SYNTH.format(y1=y(1.2, 0.87) + 0.4, y2=y(1.5, 0.87), y3=y(1.2, 1.41)))
    rows = L.compare_miter_file(str(p), _fake_miter)
    assert not rows[0]["ok"] and rows[1]["ok"]


def test_csv_without_uncertainty_is_refused(tmp_path):
    p = tmp_path / "ma1997_fig11.csv"
    p.write_text("# curve: c/a = 0.6\nf_over_fc,S11_dB\n1.2,-20\n")
    with pytest.raises(ValueError):
        L.read_csv(str(p))


def test_ma_chamfer_conversion():
    assert np.isclose(L.ma_chamfer(1.0), 1 / np.sqrt(2))      # diagonal miter, Alessandri d/b = 0.707
    assert np.isclose(L.ma_chamfer(0.0), np.sqrt(2))          # right-angle corner, d/b = 1.41


# ----------------------------------------------------------------------------------------- comparison plumbing
def _mock_wp1(monkeypatch):
    """Card-shaped stand-ins for WP1 built on the WP7 reference itself, to test the comparison code paths."""
    import sys
    import types
    from wrmodes.gsm import GSM

    class Guide:
        def __init__(self, w, h, sigma=None):
            self.w, self.h, self.sigma = w, h, sigma

    class Straight:
        def __init__(self, length):
            self.length = length

    class SmoothBend:
        def __init__(self, length, curvature, plane="H", breakpoints=None, atol=1e-6):
            self.length, self.curvature, self.plane, self.breakpoints = length, curvature, plane, breakpoints or ()

    class Line:
        def __init__(self, guide, components, plane="H"):
            self.guide, self.components = guide, components

        def gsm(self, f, cutoff_factor=1.0):
            segs = [ref_line.straight(c.length) if isinstance(c, Straight) else
                    ref_line.bend(c.length, c.curvature, c.plane, c.breakpoints) for c in self.components]
            gb, _, S, _ = ref_line.line_transfer(f, self.guide.w, self.guide.h, segs)
            # scramble the mode order to check that comparisons index by mode, not by position
            perm = np.arange(len(gb))[::-1]
            b = ModeBasis(gb.a, gb.b, [gb.modes[i] for i in perm], sort=False)
            P = np.ix_(perm, perm)
            return GSM(S["S11"][P], S["S12"][P], S["S21"][P], S["S22"][P], b, b, f)

    mod = types.ModuleType("wrmodes.components")
    mod.Guide, mod.Straight, mod.SmoothBend = Guide, Straight, SmoothBend
    lmod = types.ModuleType("wrmodes.line")
    lmod.Line = Line
    monkeypatch.setitem(sys.modules, "wrmodes.components", mod)
    monkeypatch.setitem(sys.modules, "wrmodes.line", lmod)


def test_wp1_comparison_plumbing_with_reference_as_mock(monkeypatch):
    if not os.path.exists(os.path.join(ref_common.DATA, "line_reference.npz")):
        pytest.skip("run make_references.py")
    _mock_wp1(monkeypatch)
    import compare_wp1_lines as C
    rows = C.compare_lines(freqs=[20e9])
    assert len(rows) == 3 and all(r["ok"] and r["worst"] < 1e-12 for r in rows)


def test_marcuvitz_comparison_plumbing(monkeypatch):
    import compare_literature as C
    rows = C.compare_marcuvitz(lambda f, a, b, b2: L.marcuvitz_eplane_step(f, a, b, b2)[2])
    assert all(r["ok"] for r in rows)
    rows = C.compare_marcuvitz(lambda f, a, b, b2: 1.03 * L.marcuvitz_eplane_step(f, a, b, b2)[2])
    assert not all(r["ok"] for r in rows)


# ----------------------------------------------------------------------------------------- coupled wall loss
def test_coupled_first_order_straight_reference():
    """Eigenmode solution = matrix exponential (1 m); diag L+ = (1+i) alpha; reciprocal (S21 symmetric); passive."""
    from scipy.linalg import expm
    a, b, f, L_ = 0.02, 0.02, 30e9, 1.0
    basis = ModeBasis.propagating(f, a, b)
    Zs = (1 + 1j) * ref_loss.surface_resistance(f, ref_loss.SIGMA_AL)
    Lp, Lm = ref_loss.first_order_matrices(basis, f, Zs)
    A, _ = ref_loss.loss_matrix(basis, f, ref_loss.SIGMA_AL)
    assert np.allclose(np.diag(Lp), (1 + 1j) * np.diag(A), rtol=1e-12, atol=0)
    assert np.max(np.abs(Lp - Lp.T)) < 1e-15 and np.max(np.abs(Lm - Lm.T)) < 1e-15
    R, T = ref_loss.straight_gsm_blocks(basis, f, Lp, Lm, L_)
    N, beta = len(basis), basis.beta(f)
    Phi = expm(np.block([[np.diag(-1j * beta) - Lp, -Lm], [Lm, np.diag(1j * beta) + Lp]]) * L_)
    Rx = -np.linalg.solve(Phi[N:, N:], Phi[N:, :N])
    assert np.max(np.abs(T - (Phi[:N, :N] + Phi[:N, N:] @ Rx))) < 1e-10 and np.max(np.abs(R - Rx)) < 1e-12
    assert np.max(np.abs(T - T.T)) < 1e-11
    assert np.linalg.norm(np.block([[R, T], [T, R]]), 2) < 1


def test_exact_leontovich_root_reduces_to_ideal_mode():
    """Z -> 0: the exact root tends to m pi / a, and the first-order LSE^u attenuation 2 m^2 pi^2 Re Z /
    (k a^3 eta0 beta) (Butterweck eq. 21 for TE_m0, independent of the wall-integral code) is reached."""
    a, b, f = 0.02286, 0.01016, 40e9
    k = 2 * np.pi * f / 299792458.0
    for m in (1, 3):
        Z = 1e-3 * (1 + 1j)
        g, ku = ref_loss.exact_gamma(f, a, b, m, 0, "LSE", Z, Z)
        beta = np.sqrt(k * k - (m * np.pi / a) ** 2)
        assert abs(ku - m * np.pi / a) < 1e-4 * m * np.pi / a
        first = 2 * m * m * np.pi ** 2 * Z.real / (k * a ** 3 * ref_loss.ETA0 * beta)
        assert abs(-g.imag / first - 1) < 1e-4
        Lp, _ = ref_loss.first_order_matrices(ModeBasis(a, b, [("TE", m, 0)]), f, {"u-": Z, "u+": Z})
        assert abs(Lp[0, 0].real / first - 1) < 1e-10


@pytest.mark.parametrize("walls", ["u", "v"])
def test_coupled_first_order_is_closer_to_exact_leontovich_than_group_blocks(walls):
    """Aluminium on one pair of WR-90 walls at 90 GHz (72 groups): the full first-order coupled system is within
    1e-3 of the exact attenuation for every group with fc/f < 0.9 (median about 4e-4); the group-block-only reference
    is off by up to 3.8e-3 (TE0,1, side walls) and 1.7e-3 (TE1,0, broad walls)."""
    rows = ref_loss.one_pair_comparison(*WR90, 90e9, ref_loss.SIGMA_AL, walls)
    far = [r for r in rows if r["fc_over_f"] < 0.9]
    assert max(r["coupled"] for r in far) < 1e-3
    assert np.median([r["coupled"] for r in far]) < 5e-4
    assert max(r["group"] for r in far) > 1.5e-3


def test_backward_coupling_matters_near_cutoff():
    """20 mm square guide, side walls only, 30 GHz: TE0,4/TE4,0 at fc/f = 0.9993. Without the forward-backward block
    the decay is off by 1.8 %, with it (alone or in the full system) by < 1e-3."""
    rows = ref_loss.one_pair_comparison(0.02, 0.02, 30e9, ref_loss.SIGMA_AL, "u")
    r = next(r for r in rows if "TE0,4" in r["modes"])
    assert r["fc_over_f"] > 0.999
    assert r["group"] > 1e-2 and r["group_back"] < 1e-3 and r["coupled"] < 1e-3
