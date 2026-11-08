"""Square-guide limit (section 4 of the note) and end-frame bookkeeping (section 5).

Run:  cd <worktree> && PYTHONPATH=$PWD/src:$PWD/docs/theory/T3b python3 docs/theory/T3b/square_and_frames.py

 1. Two-mode truncation {TE0,1, TE1,0}, a = b: A = [[0, -c], [c, 0]] (order TE0,1, TE1,0), B = 0, and the transfer
    matrix for ANY profile theta(z) is exp(-i beta L) R(-c theta) on the pair, no reflection. The polarization,
    read from the field at the centre, turns by -c theta in the local frame, (1 - c) theta in the fixed frame.
 2. Full basis (all propagating modes): rotation angle and leakage for 90-degree twists of several lengths, and the
    eigenvalue splitting of a uniform twist, c_eff = (gamma_1 - gamma_2) / (2 tau), against c = 8/pi^2.
 3. Zero-length limit of a 90-degree twist in a square guide, exp((pi/2) X), X = [[A, B], [B, A]], against the
    frame change of CONVENTIONS.md section 2 (eplane_frame_signs), as the basis grows.
 4. End frames (CONVENTIONS.md section 2 as updated by PR #7): +90 then -90 twist = identity at zero length.
"""
import numpy as np
import scipy.constants as cst
from scipy.linalg import expm

from wrmodes.modes import Mode, ModeBasis, transverse_field, eplane_frame_signs
from wrmodes.gsm import from_transfer_matrix
from twist_coupling import A_B_matrices, M_matrix, K_matrix_fast, C_TE10_TE01 as c

TE10, TE01 = Mode("TE", 1, 0), Mode("TE", 0, 1)


def pol_angle(a10, a01, a):
    """Local polarization angle (rad, counter-clockwise from v_hat) of a10 e_1,0 + a01 e_0,1 at the centre."""
    eu = a10 * transverse_field(TE10, a, a, 0.0, 0.0)[0] + a01 * transverse_field(TE01, a, a, 0.0, 0.0)[0]
    ev = a10 * transverse_field(TE10, a, a, 0.0, 0.0)[1] + a01 * transverse_field(TE01, a, a, 0.0, 0.0)[1]
    return np.arctan2(-eu, ev)


def transfer(basis, f, profile, L, Th, nslice):
    M0 = M_matrix(basis, f, 0.0)
    M1 = M_matrix(basis, f, 1.0) - M0
    dz = L / nslice
    Phi = np.eye(2 * len(basis), dtype=complex)
    for s in range(nslice):
        zm = (s + 0.5) * dz
        Phi = expm((M0 + profile(zm, L, Th) * M1) * dz) @ Phi
    return Phi


def const(z, L, Th):
    return Th / L


def tri(z, L, Th):
    return 2 * Th / L * 2 * min(z, L - z) / L


def wobble(z, L, Th):
    """A deliberately irregular profile for the 'any profile' claim (its integral is computed, not Th)."""
    return Th / L * (1 + 0.8 * np.sin(7 * np.pi * z / L) + 0.3 * np.cos(3 * np.pi * z / L))


def two_mode():
    a = 30e-3
    f = 1.4 * cst.c / (2 * a)
    basis = ModeBasis(a, a, [TE10, TE01])
    A, B = A_B_matrices(basis, f)
    print(f"1. two-mode truncation, square guide: order {basis.names}")
    print(f"   A = {np.array2string(A.real, precision=12)}, max|B| = {np.max(np.abs(B)):.1e}")
    assert np.allclose(A, [[0, -c], [c, 0]], atol=1e-14) and np.max(np.abs(B)) < 1e-14
    beta = basis.beta(f)[0].real
    worst = 0.0
    for prof in (const, tri, wobble):
        for Th, L in ((np.pi / 2, 0.3), (2.0, 1.1), (np.pi ** 3 / 16, 0.7)):
            # total angle of the sliced profile (the claim is exact for any profile, the sliced one included)
            theta_L = sum(prof((s + 0.5) * L / 200, L, Th) for s in range(200)) * L / 200
            Phi = transfer(basis, f, prof, L, Th, 200)
            i10, i01 = basis.index(TE10), basis.index(TE01)
            R = np.array([[np.cos(c * theta_L), -np.sin(c * theta_L)], [np.sin(c * theta_L), np.cos(c * theta_L)]])
            # order (TE0,1, TE1,0), A = [[0, -c], [c, 0]]: a01 = -sin(c theta), a10 = cos(c theta) for TE1,0 in
            expect = np.exp(-1j * beta * L) * R
            worst = max(worst, np.max(np.abs(Phi[:2, :2] - expect)), np.max(np.abs(Phi[:2, 2:])))
            out = Phi[:2, :2] @ np.eye(2)[:, [i10]]                                # incident TE1,0
            o = (out[:, 0] * np.exp(1j * beta * L)).real                           # common phase removed
            phi = pol_angle(o[i10], o[i01], a)
            if prof is const:
                print(f"   theta = {np.degrees(theta_L):7.2f} deg: local polarization {np.degrees(phi):8.3f} deg "
                      f"(-c theta = {np.degrees(-c * theta_L):8.3f}), fixed frame {np.degrees(phi + theta_L):7.3f} deg "
                      f"((1-c) theta = {np.degrees((1 - c) * theta_L):7.3f}); |TE1,0|^2 = {abs(out[i10, 0]) ** 2:.6f}")
                assert abs(np.angle(np.exp(1j * (phi + c * theta_L)))) < 1e-10
    print(f"   transfer matrix vs exp(-i beta L) R(-c theta) for constant, triangular and irregular profiles: "
          f"max error {worst:.1e}")
    print(f"   full TE1,0 -> TE0,1 conversion at c theta = pi/2: theta = pi^3/16 = {np.degrees(np.pi ** 3 / 16):.2f} deg")
    assert worst < 1e-10


def full_basis():
    a = 30e-3
    print("2. full basis (all propagating modes), square guide a = b = 30 mm, incident TE1,0, 90-degree twists")
    print("   2a/lambda  L/lambda  profile     local angle (deg)  [-c theta = -72.951]  ellipticity  P(other modes)  refl.")
    assert abs(np.degrees(pol_angle(np.cos(c * np.pi / 2), -np.sin(c * np.pi / 2), a)) + 72.951) < 1e-3
    for x in (2.5, 4.5):
        f = x * cst.c / (2 * a)
        lam = cst.c / f
        basis = ModeBasis.propagating(f, a, a)
        i10, i01 = basis.index(TE10), basis.index(TE01)
        for Ll in (5, 20, 80):
            for name, prof in (("constant", const), ("triangular", tri)):
                Phi = transfer(basis, f, prof, Ll * lam, np.pi / 2, 120 if Ll < 80 else 300)
                S = from_transfer_matrix(Phi, basis, f)
                t = S.S21[:, i10]
                r = S.S11[:, i10]
                # Stokes parameters of the pair in the (e_1,0 ~ v_hat, e_0,1 ~ -u_hat) basis: orientation of the
                # polarization ellipse (counter-clockwise from v_hat) and ellipticity V/I
                s0 = abs(t[i10]) ** 2 + abs(t[i01]) ** 2
                phi = 0.5 * np.arctan2(2 * (np.conj(t[i10]) * t[i01]).real, abs(t[i10]) ** 2 - abs(t[i01]) ** 2)
                ell = 2 * abs((np.conj(t[i10]) * t[i01]).imag) / s0
                other = 1 - abs(t[i10]) ** 2 - abs(t[i01]) ** 2 - np.sum(np.abs(r) ** 2)
                print(f"   {x:9.1f}  {Ll:8d}  {name:10s}  {np.degrees(phi):17.3f}  {ell:21.1e}  "
                      f"{other:14.2e}  {np.sum(np.abs(r) ** 2):.1e}")
    print("   uniform twist, eigenvalue splitting of the TE1,0/TE0,1 pair (basis: modes with chi < 3 k):")
    for x in (1.4, 2.5, 4.5):
        f = x * cst.c / (2 * a)
        k = 2 * np.pi * f / cst.c
        basis = ModeBasis.below(3 * f, a, a)
        beta0 = basis.beta(f)[basis.index(TE10)].real
        line = []
        for ta in (0.01, 0.1, 0.3):
            tau = ta / a
            lam = np.linalg.eigvals(M_matrix(basis, f, tau))
            g = (1j * lam).real
            fw = g[np.argsort(np.abs(g - beta0))][:2]                  # the split TE1,0 / TE0,1 pair
            ceff = abs(fw[0] - fw[1]) / (2 * tau)
            line.append(f"tau a = {ta}: c_eff = {ceff:.6f}")
        print(f"   2a/lambda = {x}: " + ", ".join(line) + f"   (c = {c:.6f})")


def zero_length():
    """At zero length the twist terms decouple in the V-I form: V -> exp(theta K) V, I -> exp(-theta K^T) I,
    independent of frequency. For a square guide and theta = 90 deg both must tend to the frame change F."""
    a = 30e-3
    print("3. zero-length 90-degree twist, square guide: exp((pi/2) K) (E) and exp(-(pi/2) K^T) (H) vs F")
    for chi_a in (10, 20, 40, 80, 120):
        basis = ModeBasis.below(chi_a / a * cst.c / (2 * np.pi), a, a)
        K = K_matrix_fast(basis)
        F = eplane_frame_signs(basis, basis)
        TV = expm(np.pi / 2 * K)
        TI = expm(-np.pi / 2 * K.T)
        i10, i01 = basis.index(TE10), basis.index(TE01)
        low = [i for i, m in enumerate(basis.modes) if max(m.m, m.n) <= 2]
        eV = np.max(np.abs((TV - F)[np.ix_(low, low)]))
        eI = np.max(np.abs((TI - F)[np.ix_(low, low)]))
        print(f"   chi a < {chi_a:3d} ({len(basis):4d} modes): E: TE1,0 -> TE0,1 {TV[i01, i10]:+.5f}, max|T-F| (m,n <= 2) "
              f"{eV:.4f};  H: TE1,0 -> TE0,1 {TI[i01, i10]:+.5f}, max|T-F| {eI:.4f}   (F = {F[i01, i10]:+.0f})")


def frames():
    """CONVENTIONS.md section 2 (updated, PR #7): the global guide frame rotates with the cross-section, so a twist's
    GSM has the same (global) frame at both ports and no FrameChange follows it. Check: a +90-degree twist followed
    by a -90-degree twist is the identity at zero length (GSM star product), and how far it is from it at finite
    length."""
    print("4. end frames: +90 twist followed by -90 twist (GSM cascade, propagating modes)")
    for name, (a, b) in (("square 30 mm", (30e-3, 30e-3)), ("tall WR187", (22.149e-3, 47.549e-3))):
        for f in (9e9, 16.5e9):
            basis = ModeBasis.propagating(f, a, b)
            N = len(basis)
            A, B = A_B_matrices(basis, f)
            X = np.block([[A, B], [B, A]])
            plus = from_transfer_matrix(expm(np.pi / 2 * X), basis, f)        # zero length: only the tau terms
            minus = from_transfer_matrix(expm(-np.pi / 2 * X), basis, f)
            G = plus @ minus
            err = max(np.max(np.abs(G.S21 - np.eye(N))), np.max(np.abs(G.S12 - np.eye(N))),
                      np.max(np.abs(G.S11)), np.max(np.abs(G.S22)))
            # finite length: two 20-wavelength constant-rate twists, +90 then -90
            L = 20 * cst.c / f
            Pp = from_transfer_matrix(transfer(basis, f, const, L, np.pi / 2, 60), basis, f)
            Pm = from_transfer_matrix(transfer(basis, f, const, L, -np.pi / 2, 60), basis, f)
            Gf = Pp @ Pm
            i10 = basis.index(TE10)
            print(f"   {name:12s} {f / 1e9:5.1f} GHz ({N:2d} modes): zero length max|S - I_GSM| = {err:.1e};  "
                  f"two 20-lambda constant twists: |S21[TE1,0,TE1,0]|^2 = {abs(Gf.S21[i10, i10]) ** 2:.6f}")
            assert err < 1e-10


if __name__ == "__main__":
    two_mode()
    full_basis()
    zero_length()
    frames()
