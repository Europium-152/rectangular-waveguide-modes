"""Generalized scattering matrix of a junction between two guides by mode matching.

Formulation (CONVENTIONS.md sections 4-6). At the junction plane, write the transverse fields of each guide with
the normalized voltages v = a+ + a- and currents i = a+ - a- of every mode,

    E_t = sum_p v_p sqrt(2 / Y_p) e_p,     H_t = sum_p i_p sqrt(2 Y_p) z_hat x e_p .

The large guide (L) sees the aperture of the small guide (S); its tangential electric field equals that of the small
guide on the aperture and vanishes on the metal of the step, and the tangential magnetic fields agree on the
aperture. Projecting the first condition on the large guide's modes and the second on the small guide's modes
(Galerkin) gives, with X[p, q] = int_S e^S_p . e^L_q ds,

    v_L = F v_S,     i_S = F^T i_L,     F = diag(sqrt(Y_L)) X^T diag(1 / sqrt(Y_S))      (N_L x N_S).

With the large guide at port 1 (incident a1 = A+_L, the small guide at port 2 behind the junction), eliminating
the outgoing waves gives, with R = (I + F^T F)^-1 (size N_S),

    S11 = 2 F R F^T - I,   S12 = 2 F R,   S21 = 2 R F^T = S12^T,   S22 = 2 R - I .

S is symmetric by construction (R is symmetric); it is enforced exactly by symmetrizing R. Because X is real, the
complex power is conserved exactly by the truncated equations (int E_L x H_L* = int E_S x H_S*), so the
propagating block of S is unitary at every truncation, up to round-off.
"""
import numpy as np

from ..gsm import GSM

__all__ = ["junction_gsm"]


def junction_gsm(basis_large, basis_small, X, f, large_port=1, name=""):
    """GSM of the junction of a large guide and a small guide whose cross-section lies inside it.

    Parameters
    ----------
    basis_large, basis_small : ModeBasis
        Mode bases of the two guides at the junction plane.
    X : ndarray (len(basis_small), len(basis_large))
        Overlap matrix of the normalized fields, X[p, q] = int_small e^small_p . e^large_q ds
        (``wrmodes.modematch.overlap_matrix``).
    f : float
        Frequency (Hz). No mode of either basis may be exactly at cutoff (CONVENTIONS.md section 3).
    large_port : 1 or 2
        Port of the large guide. With large_port=2 the small guide is at port 1 and the ports of the
        large_port=1 result are exchanged (same physical junction seen from the other side; no reversal signs,
        since the frame is not turned around).
    name : str
        Name of the GSM.

    Returns
    -------
    GSM
        Zero-length junction; both reference planes are the junction plane. Amplitudes normalized with the
        unconjugated reaction of CONVENTIONS.md section 5 (principal square roots of the wave admittances).
    """
    X = np.asarray(X, dtype=float)
    NS, NL = len(basis_small), len(basis_large)
    if X.shape != (NS, NL):
        raise ValueError(f"X has shape {X.shape}, expected {(NS, NL)}")
    sYL = np.sqrt(basis_large.admittance(f))
    sYS = np.sqrt(basis_small.admittance(f))
    F = sYL[:, None] * X.T / sYS[None, :]                    # N_L x N_S
    G = F.T @ F
    R = np.linalg.solve(np.eye(NS) + G, np.eye(NS))
    R = 0.5 * (R + R.T)
    FR = F @ R
    S_LL = 2.0 * FR @ F.T
    S_LL = 0.5 * (S_LL + S_LL.T) - np.eye(NL)
    S_LS = 2.0 * FR                                           # (N_L x N_S): from small port to large port
    S_SS = 2.0 * R - np.eye(NS)
    if large_port == 1:
        return GSM(S_LL, S_LS, S_LS.T, S_SS, basis_large, basis_small, f, name)
    if large_port == 2:
        return GSM(S_SS, S_LS.T, S_LS, S_LL, basis_small, basis_large, f, name)
    raise ValueError("large_port must be 1 or 2")
