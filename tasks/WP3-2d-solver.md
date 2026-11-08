# WP3 — 2D modal-port solver for coplanar components (wave 1: H-plane miter)

Role: Implementer · Wave: 1 · Branch: `wave-1/WP3-2d-solver` · Depends on: WP0 (merged); WP1 `Component`
interface · Verified by: WP7 (COMSOL, Ma 1997 data) at Gate A; WP4 mode matching (wave 2) ·
Time box: 8 days (then status review)

## Goal

A finite-element solver for any component whose geometry does not vary along $v$ (side b). Each family $n$ and
LS polarization is solved as a scalar 2D Helmholtz problem in the $u$–$z$ plane, with exact modal port
conditions, and the result is returned as a GSM in the TE/TM basis of `CONVENTIONS.md`. Wave 1 delivers the
general solver and the **H-plane miter** (family $n = 0$, LSM, Dirichlet), which answers the first SPARC design
question. All other miters follow in wave 2.

## Read first

* `CONVENTIONS.md`, §7 in particular (LS modes, rotation to TE/TM).
* `src/wrmodes/modes.py` (`ls_rotation`, `transverse_field`).
* Butterweck & de Ronde (1968) p. 97, on the optical H-plane miter and Bouwkamp's < 0.4 dB above 8 f_c.
* Ma et al. (1997), mitered H-plane bends (geometry and results).
* Alessandri et al. (1994), Fig. 1: the miter geometry and the d/b parameter.

## Specification

1. **Equations.** For family $n$: $\nabla^2\psi + k_t^2\psi = 0$ in the 2D outline, with
   $k_t^2 = k^2 - (n\pi/b)^2$.
   * LSM: $\psi$ is the $E_v$ profile, $\psi = 0$ on walls.
   * LSE: $\psi$ is the $H_v$ profile, $\partial_n\psi = 0$ on walls.
   * The port modes are $\sin(m\pi u'/a)$ (LSM) and $\cos(m\pi u'/a)$ (LSE), with the same $\beta$ as the 3D mode
     $(m, n)$.
2. **Ports.** Modal (Dirichlet-to-Neumann) port conditions with all propagating port modes plus $M$ evanescent
   ones; show convergence in $M$. Use a formulation that keeps the discrete system complex-symmetric, so that
   reciprocity holds to rounding.
3. **Normalization.**
   * Derive the factor relating the 2D modal amplitude to the LS amplitude $a_{LS}$ of `CONVENTIONS.md`,
     including its sign; then convert with `ls_rotation` per port.
   * Write the derivation in `docs/theory/WP3-normalization.md`.
   * This is the step most likely to go wrong. The acceptance tests below check it twice: against a straight
     section, and in phase against wrmodes on smooth bends.
4. **Geometry.**
   * `Miter(angle, chamfer=None, plane='H')`, any bend angle, with the mirror perpendicular to the bisector.
     `chamfer` is Alessandri's $d$ (distance from the inner corner to the mirror), normalized to the in-plane
     side; `None` means the plain miter whose mirror joins the two outer corners.
   * `Coplanar2D(outline, ports)` for a general polygon or arc outline: used for smooth bends (validity map) and,
     in wave 2, double miters and chamfers.
   * Port planes sit at a stated distance from the discontinuity (default one guide width). The component's GSM
     is de-embedded to the reference planes it documents.
5. **Discretization.**
   * scikit-fem, P2 elements; mesh with gmsh, both through the optional extra `wrmodes[fem]`.
   * Default mesh size $h \le \lambda_t/10$, refined at re-entrant corners.
   * Provide `convergence_study()`, which reports the change of every |S| entry under mesh refinement and $M$.
6. **Performance.** WR-90 H-plane miter at 90 GHz: under 2 s per frequency on a laptop for the default settings.
   Frequencies are independent (WP1 `sweep`).

## Acceptance tests

| Check | Target |
| --- | --- |
| Straight channel (no discontinuity) | S equals `gsm.straight` to 1e-6, signs included |
| Smooth H- and E-plane bends with κ/β ≤ 0.03, 3 frequencies | complex S21 of every propagating mode equals wrmodes to 1e-3 absolute (gauge and sign check) |
| Reciprocity | 1e-10 |
| Unitarity | ≤ 1e-5 at default settings, decreasing under refinement |
| Symmetric miter | S11 = S22 to 1e-10 |
| H-plane miter vs Ma et al. 1997 (D3 of L1) | within the digitizing uncertainty |
| WR-90 H-plane plain miter, 18–90 GHz | loss curve with convergence evidence; comparison with Bouwkamp's < 0.4 dB above 52 GHz, reported (not pass/fail) |

## Deliverables

* `src/wrmodes/fem2d/` (importing it without the extra raises a clear message).
* `tests/test_fem2d.py`, skipped when the extra is missing.
* `docs/theory/WP3-normalization.md`, `docs/reports/WP3.md`, and `examples/h_plane_miter.py`.

## Out of scope (wave 2)

E-plane and chamfered miters beyond a smoke test, double miters, LSE families other than for the smooth-bend
check, impedance walls, the validity map.
