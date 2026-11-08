"""T3 check 5: end frames of a twist (card item 4, CONVENTIONS.md section 2 as updated by PR #7).

Rule: the global guide frame is attached to the guide and rotates with the cross-section. The co-rotating frame
(u, v) of the derivation IS the guide frame at every z, so a twist's GSM has the same basis at both ports and no
FrameChange follows it. Checks:

* zero length (propagation dropped, Z(theta) = exp(theta M1)): a +90 deg twist followed by a -90 deg twist is the
  identity, and so is -90 then +90 (exact, any truncation);
* covariance under the H <-> E frame change of CONVENTIONS.md section 2 (wrmodes.modes.eplane_frame_signs, F):
  the twist matrix of the same guide described in the other frame is F C F^T, and F M F^T for the system matrix,
  so a twist can be computed in either frame (its `plane` attribute only fixes which a, b it is given in);
* turning a twist around (CONVENTIONS.md section 2, P S_swapped P) gives the twist with the reversed profile
  theta'(L - z) and the SAME sign of theta (a helix keeps its handedness under the rotation by pi about u);
* information only: the physical zero-length limit of a 90 deg twist of a square guide is the 90 deg relabeling,
  which exp((pi/2) M1) approaches only to about 2 %: the limits L -> 0 and N -> infinity do not commute.
"""
import os
import sys

import numpy as np
import scipy.constants as cst

sys.path.insert(0, os.path.dirname(__file__))
from twist_formulas import (coupling_matrix, frame_change_matrix, system_matrix, transfer_matrix,  # noqa: E402
                            zero_length_twist)
from wrmodes.gsm import from_transfer_matrix  # noqa: E402
from wrmodes.modes import ModeBasis, eplane_frame_signs  # noqa: E402


def main():
    worst = 0.0
    for a, b in [(0.02286, 0.01016), (0.02, 0.02), (0.03, 0.017)]:
        basis = ModeBasis.below(3 * cst.c / (2 * min(a, b)), a, b)
        I = np.eye(2 * len(basis))
        Zp, Zm = zero_length_twist(basis, np.pi / 2), zero_length_twist(basis, -np.pi / 2)
        e1, e2 = np.max(np.abs(Zm @ Zp - I)), np.max(np.abs(Zp @ Zm - I))
        # H <-> E covariance: the same guide described with u along the other side
        Eb = ModeBasis(b, a, [m.__class__(m.kind, m.n, m.m) for m in basis])
        F = eplane_frame_signs(basis, Eb)
        cov = np.max(np.abs(coupling_matrix(Eb) - F @ coupling_matrix(basis) @ F.T)) / np.max(np.abs(coupling_matrix(basis)))
        f = 0.9 * 3 * cst.c / (2 * min(a, b))
        FF = np.kron(np.eye(2), F)
        covM = np.max(np.abs(system_matrix(Eb, f, 37.0) - FF @ system_matrix(basis, f, 37.0) @ FF.T))
        covM /= np.max(np.abs(system_matrix(basis, f, 37.0)))
        print(f"a={a*1e3:6.2f} b={b*1e3:6.2f} mm, {len(basis):3d} modes: Z(-90)Z(+90) - I = {e1:.1e}, "
              f"Z(+90)Z(-90) - I = {e2:.1e}; H/E covariance of C {cov:.1e}, of M {covM:.1e}")
        worst = max(worst, e1, e2, cov, covM)
    assert worst < 1e-10

    # reversal: P S_swapped P of the twist = twist with the reversed profile, same sign of theta
    a, b, f, L, th = 0.03, 0.017, 16e9, 0.25, np.pi / 2
    basis = ModeBasis.propagating(f, a, b)
    prof = lambda z: (th / L) * 2 * (z / L)                      # noqa: E731  ramp: theta' = 2 theta z / L^2
    rev = lambda z: (th / L) * 2 * ((L - z) / L)                 # noqa: E731
    g = from_transfer_matrix(transfer_matrix(basis, f, prof, L, slices=200), basis, f)
    gr = from_transfer_matrix(transfer_matrix(basis, f, rev, L, slices=200), basis, f)
    gneg = from_transfer_matrix(transfer_matrix(basis, f, lambda z: -rev(z), L, slices=200), basis, f)
    St = g.reversed().S
    e_rev = np.max(np.abs(St - gr.S))
    e_neg = np.max(np.abs(St - gneg.S))
    print(f"\nturning a ramp twist around ({len(basis)} modes): |reversed - twist(theta'(L-z), +theta)| = {e_rev:.1e}; "
          f"against -theta: {e_neg:.1e}")
    assert e_rev < 1e-10 and e_neg > 1e-3
    print("PASS")

    print("\nPhysical zero-length limit, square guide: exp((pi/2) M1) on the TE10/TE01 block (rows/cols TE10, TE01) "
          "vs the 90 deg relabeling [[0, 1], [-1, 0]] (two-mode truncation: rotation by c pi/2 = 72.9 deg):")
    for fac in (2, 4, 8, 16):
        sq = ModeBasis.below(fac * cst.c / 2, 1.0, 1.0)
        Z = zero_length_twist(sq, np.pi / 2)
        i, j = sq.index("TE1,0"), sq.index("TE0,1")
        blk = Z[np.ix_([i, j], [i, j])]
        print(f"  {len(sq):5d} modes: [[{blk[0,0].real:+.4f}, {blk[0,1].real:+.4f}], [{blk[1,0].real:+.4f}, "
              f"{blk[1,1].real:+.4f}]]")
    Fq, to = frame_change_matrix(ModeBasis(1.0, 1.0, ["TE1,0", "TE0,1"]), 1)
    print("  90 deg relabeling (rows/cols in basis order TE0,1, TE1,0):", Fq.tolist())


if __name__ == "__main__":
    main()
