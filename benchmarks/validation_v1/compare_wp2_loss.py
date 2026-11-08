"""Compare WP2's wall loss with the WP7 surface-integral reference (ref_loss).

    python benchmarks/validation_v1/compare_wp2_loss.py

Acceptance (tasks/WP2-loss-and-filters.md): "Lossy straight vs WP7 surface-integral reference: 1e-6 relative on
alpha for every mode". Two comparisons, for WR-90 and a 20 mm square guide, aluminium and copper:

1. WP2's per-mode attenuation function (located by ref_interfaces.attenuation_function) against diag(A).
2. A lossy straight, Line(Guide(w, h, sigma), [Straight(L)]).gsm(f), L = 1 m, against the WP7 first-order coupled
   system (ref_loss.first_order_matrices, straight_gsm_blocks): every propagating pair coupled through the walls,
   forward and backward blocks, Z_s = (1+i) R_s. Same theory, independent derivation and implementation:
   tolerance 1e-8 absolute on S11 and S21. (Until 2026-11-08 this compared decay rates with each degenerate group's
   own block; see docs/discrepancies/2026-11-08-wp2-wp7-straight-attenuation.md for why that was wrong.)

Exits with status 0 and a message when WP2 is not merged yet. Writes results/wp2_comparison.json.
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ref_common  # noqa: E402
import ref_interfaces as I  # noqa: E402
import ref_loss  # noqa: E402
from wrmodes.modes import Mode, ModeBasis  # noqa: E402

TOL_REL = 1e-6
TOL_S = 1e-8


def degenerate_groups(basis, f, rtol=1e-9):
    beta = basis.beta(f).real
    groups, used = [], set()
    for i in range(len(basis)):
        if i in used:
            continue
        g = [j for j in range(len(basis)) if j not in used and abs(beta[j] - beta[i]) <= rtol * beta[i]]
        used.update(g)
        groups.append(g)
    return groups


def reference_group_alphas(A, groups):
    """alpha of every group: the eigenvalues of A restricted to the group (= diag for single modes)."""
    return [np.sort(np.linalg.eigvalsh(A[np.ix_(g, g)])) for g in groups]


def second_order_size(A, basis, f, L=1.0):
    """Largest relative change of alpha_eff from the off-group couplings: compare |diag expm((D - A) L)| with the
    group-only result. Shows how much of a straight's |S21| is not first-order alpha."""
    beta = basis.beta(f)
    D = np.diag(-1j * beta)
    full = expm((D - A) * L)
    groups = degenerate_groups(basis, f)
    Ab = np.zeros_like(A)
    for g in groups:
        Ab[np.ix_(g, g)] = A[np.ix_(g, g)]
    blk = expm((D - Ab) * L)
    a_full = -np.log(np.abs(np.diag(full))) / L
    a_blk = -np.log(np.abs(np.diag(blk))) / L
    return float(np.max(np.abs(a_full / a_blk - 1)))


def compare_function(data):
    name, fn = I.attenuation_function()
    rows = []
    for g, (a, b) in ref_loss.GUIDES.items():
        for mat, sigma in ref_loss.MATERIALS.items():
            for f in ref_loss.FREQUENCIES[g]:
                key = f"{g}/{mat}/{f / 1e9:g}"
                A = data[f"{key}/A"]
                modes = [Mode.parse(str(m)) for m in data[f"{key}/modes"]]
                got = np.array([float(np.real(fn(m, a, b, f, sigma))) for m in modes])
                rel = np.abs(got / np.diag(A) - 1)
                rows.append(dict(case=key, max_rel=float(rel.max()), worst_mode=modes[int(rel.argmax())].name,
                                 ok=bool(rel.max() <= TOL_REL)))
                print(f"{name}: {key:28s} max rel. dev. {rel.max():.2e}")
    return rows


def compare_straight(data, L=1.0):
    """WP2's lossy straight against the WP7 first-order coupled reference (ref_loss.straight_gsm_blocks): every
    propagating pair coupled through the walls, forward and backward blocks (resolution of
    docs/discrepancies/2026-11-08-wp2-wp7-straight-attenuation.md). Tolerance 1e-8 absolute on S11 and S21."""
    Guide, Straight, Line = I.get("Guide"), I.get("Straight"), I.get("Line")
    rows = []
    for g, (a, b) in ref_loss.GUIDES.items():
        for mat, sigma in ref_loss.MATERIALS.items():
            line = Line(Guide(a, b, sigma=sigma), [Straight(L)])
            for f in ref_loss.FREQUENCIES[g]:
                key = f"{g}/{mat}/{f / 1e9:g}"
                basis = ModeBasis.propagating(f, a, b)
                Zs = (1 + 1j) * ref_loss.surface_resistance(f, sigma)
                Lp, Lm = ref_loss.first_order_matrices(basis, f, Zs)
                R, T = ref_loss.straight_gsm_blocks(basis, f, Lp, Lm, L)
                gsm = line.gsm(f, cutoff_factor=1.0)
                i1 = np.array([gsm.basis1.index(m) for m in basis.modes])
                i2 = np.array([gsm.basis2.index(m) for m in basis.modes])
                dev = max(np.max(np.abs(gsm.S21[np.ix_(i2, i1)] - T)), np.max(np.abs(gsm.S11[np.ix_(i1, i1)] - R)))
                rows.append(dict(case=key, max_abs_dS=float(dev), ok=bool(dev <= TOL_S)))
                print(f"straight {key:28s} max |dS| vs coupled first-order reference {dev:.2e}")
    return rows


def main():
    data = ref_common.load_npz(os.path.join(ref_common.DATA, "loss_reference.npz"))
    out = {}
    for label, fn in (("function", compare_function), ("straight", compare_straight)):
        try:
            out[label] = fn(data)
        except I.InterfaceMissing as exc:
            print(f"WP2 {label} comparison skipped: {exc}")
            out[label] = {"skipped": str(exc)}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "wp2_comparison.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    bad = [r for v in out.values() if isinstance(v, list) for r in v if not r["ok"]]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
