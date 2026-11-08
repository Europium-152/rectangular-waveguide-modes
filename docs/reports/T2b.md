# T2b report — taper coupling, independent derivation (for comparison with T2)

Branch `wave-1/T2b-taper-coupling`. Card `tasks/T2-taper-coupling.md`, second theory agent. I did not look at
the T2 worktree or branch.

## What was done

* **Derivation by a different route** (`docs/theory/T2b-taper-coupling.md`, §2): a coordinate transformation
  $u = u_0 + a\xi$, $v = v_0 + b\eta$ maps the tapered, offset, curved guide onto a uniform guide; Maxwell's
  equations in covariant components keep their Cartesian form, the walls become coordinate surfaces (PEC: $E_\zeta = 0$
  and one transverse covariant component), and the covariant transverse fields on $\zeta$ = const are the physical
  ones rescaled, so the amplitudes are exactly the local-mode amplitudes of CONVENTIONS.md. Projection with the
  metric-free reaction integral gives the V–I equations with no tilted-wall or moving-domain term.
* **Results, all boxed, each with a Python version in `docs/theory/T2b/t2b_formulas.py`:**
  * B1 $dV/dz = -iZI + TV$, $dI/dz = -iYV - T^{\mathsf T}I$ (exact $T^I = -T^{\mathsf T}$);
  * B2 $Z$, $Y$ = T0 curvature integrals over the local rectangle (local $a, b$; $u$ from the reference line);
  * B3a $T_{pq} = \iint_{R(z)} e_q\cdot\partial_ze_p$ (fixed domain) and B3 closed forms for all TE/TM pairs, in terms
    of the wall slopes $\Omega_u = a'$ or $-2u_0'$, $\Omega_v = b'$ or $-2v_0'$;
  * B4 power-wave form ($A = \tfrac12(P - P^{\mathsf T})$, $B = \tfrac12(P + P^{\mathsf T}) + \operatorname{diag}(Y'/2Y)$,
    $P = Y^{1/2}TY^{-1/2}$);
  * B5 a solution method regular and stable through cutoff (Magnus slices of the V–I system, constant real
    reference admittance, Redheffer cascade, junctions to modal waves at the ports), and B5' a **quasi-static tail
    correction** $Y\to Y - T_{\rm out}^{\mathsf T}Z_{\rm out}^{-1}T_{\rm out}$;
  * B6 small-slope formulas, and B6' a regular first-order formula (reciprocity) valid when a mode crosses cutoff.
* Selection rules, the curvature × taper question (no cross terms, exactly), the tilted-wall term in the
  cross-section language, TE10 single-mode limit, validity limits.
* Check scripts in `docs/theory/T2b/` with their outputs (`output_*.txt`), including an exact-Maxwell-solution
  generator for wedge guides (`exact_wedge.py`: $H^{(2)}_\nu$, $J_\nu$ cylindrical waves, LSM and LSE).

## Results

* $T$ closed form = transformation-route integrals = complex-step derivative of the fixed-domain overlap:
  6.5e-15 and 3.8e-15 relative (31 modes, 3 geometries with all four slopes, one square guide); central difference
  with Richardson 1.3e-10. Sympy (symbolic $a, b, a', b', u_0', v_0'$), 12 pairs: exact (e.g. $T_{\rm TE10,TE30} = 3a'/4a$,
  $T_{\rm TE10,TE20} = -8u_0'/3a$, $T_{\rm TM11,TE11} = 2(ba' - ab')/(a^2 + b^2)$, $T_{\rm TE11,TM11} = 0$).
* Moving domain $R(z')$ gives $-T^{\mathsf T}$ (to 3e-15), different from $T$ by 0.55 max|T|.
* B4 vs numerical change of variables of B1: 3.7e-10 (finite-difference limited).
* Exact Maxwell fields (wedge guides) satisfy B1: voltage residual $\propto M^{-3}$ (4.1e-7 / 5.1e-8 / 6.4e-9 at
  $M$ = 32/64/128), current residual $\propto M^{-1}$ (4.6e-5 / 2.3e-5 / 1.2e-5), in four cases: symmetric and
  asymmetric H-plane wedges, an asymmetric E-plane wedge (families coupled), and a curved reference line
  (κ = 4 m⁻¹) with $a'$ and $u_0'$ ≈ 0.29. With $-T^{\mathsf T}$ in the voltage equation the residual stays at 1e-2 to
  8e-2. The tilted-wall condition $hE_z + w_nE_n = 0$ holds on the exact fields to 4e-14.
* Solver B5 vs exact wedge fields through cutoff crossings (TE30/TE50, LSE/LSM modes reflected at their cutoff
  plane inside the section): error on the first 5 modes of each port, plain truncation $M$ = 6/12:
  3.5e-2/1.7e-2, 8.9e-3/3.7e-3, 7.3e-3/3.9e-3; curved, $M$ = 4/8: 1.9e-2/5.0e-3. **With B5'**: 6.3e-3/3.7e-4,
  6.8e-4/3.3e-5, 2.0e-4/1.3e-5; curved 1.1e-2/3.2e-4. Reciprocity ≤ 3e-14, unitarity of the propagating block ≤ 5e-14
  in all runs.
* Small slope (linear H-plane taper 22.86 → 45 mm, 15 GHz, TE30 cut off at 29.98 mm inside): B6' vs full,
  relative error 5.7e-2, 3.1e-3, 5.4e-4, 3.0e-4 for slopes 0.22, 0.11, 0.055, 0.028; TE10 reflection from B6:
  0.23, 8.1e-2, 4.2e-2, 2.1e-2 (∝ slope). The WKB "direct + secondary wave" split is off by 50–100 % for a linear
  taper (the Airy region dominates); it is kept only as the qualitative explanation of the bottle taper.

## Acceptance tests

| Check | Target | Result |
| --- | --- | --- |
| Closed forms vs numerical derivative of the overlap, domain stated | 1e-8 | **pass**: 3.8e-15 (complex step), 1.3e-10 (Richardson); domain $R(z)$ fixed; moving domain gives $-T^{\mathsf T}$ |
| Tilted-wall term, derived and checked separately | analytic + numerical | **pass**: on this route it is contained in $E_\zeta = 0$; in the cross-section route it cancels the Leibniz term exactly (§2.1); wall condition 4e-14 on exact fields; wrong-domain variant fails |
| TE₁₀ in a linear H-plane taper (published single-mode theory) | exact in its limit | **pass**: $B_{11} = -\frac12(\ln b/\beta)'$ = Katsenelenbaum et al. 1998 eq. 4.47 (W = Cb/β) to 4e-16; first-order reflection vs full solution, error ∝ slope (2.1e-2 at slope 0.028) |
| Full equations vs WP4 staircase | convergence | **not done** (wave 2). In its place: exact Maxwell solutions (residuals → 0; GSM errors → 0 with $M$) |
| Uniform-guide limit | reduces to T0 exactly | **pass**: $T \equiv 0$ (0.0), B1–B2 are then the T0 equations term by term |
| Lossless: power-wave coupling structure | shown analytically | **pass**: $A$ real antisymmetric, $B$ real symmetric above cutoff (0.0 numerically); V–I form: $Z, Y$ symmetric, $T^I = -T^{\mathsf T}$; GSM unitarity ≤ 5e-14 |
| T2 and T2b identical | identical or discrepancy report | **pending** (coordinator): compare with `t2b_formulas.T_closed`, `power_wave_matrices` |

## Open issues

* **Convergence in the number of modes.** The plain truncation converges as $1/M$ for any tilted wall (normal $H$
  at the wall, TE modes with $Z_{qq} = \omega\mu_0$ not growing with $q$). B5' fixes it in all tests; WP5 should use
  it (or extrapolate in $M$). I have not proven its convergence order; observed factor 15–30 per doubling.
* The reference solver `docs/theory/T2b/t2b_solver.py` is a check tool (slow: Python loops, quadrature of the
  curvature integrals); WP5 should use closed forms for the curvature terms (T0/wrmodes 0.1) and vectorize.
* No comparison with the WP4 staircase yet (wave 2). Katsenelenbaum's general eqs. 4.45 were not compared term by
  term (different normalization and sign conventions); only the TE10 case 4.47 was.

## Conventions issues

* CONVENTIONS.md §6 measures lengths along the centre line. For a taper with an offset $u_0(z)\ne0$ in a bend, the
  curvature is defined for a reference line that is then not the guide axis; the GSM length is along the
  reference line. Not a problem when the reference line is the guide axis ($u_0 = 0$), which WP5 should prefer.
* The card's sentence "symmetric height taper couples $\Delta n$ even, across families" is right; note that
  Butterweck & de Ronde's list (H₃₂, H₃₄, …) contains modes reached only at second order.

## Notes for other cards

* T2 comparison: my $T_{pq}$ is the coefficient of $V_q$ in $dV_p/dz$; the current equation has $-T^{\mathsf T}$.
  Field and amplitude conventions are those of CONVENTIONS.md §4–5 with the local $a, b$ and the offset applied as
  $e_p(u - u_0, v - v_0)$.
* WP5: per-family GSMs are valid only for $a$, $u_0$, κ tapers; any $b'$ or $v_0'$ couples families.
