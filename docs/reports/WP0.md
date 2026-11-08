# WP0 — Foundations (wave 0) report

Branch `wave-0/foundations`, from `main` at 441431e (wrmodes 0.1.0).

## What was done

| Item | Files | Result |
| --- | --- | --- |
| Field convention behind the 0.1 coupling coefficients | `docs/theory/T0-field-convention.md`, `tests/test_conventions.py` | $e_{TM} = \nabla\Pi$, $e_{TE} = \nabla\Gamma\times\hat z$. The curvature coupling matrix re-derived from these fields by quadrature matches `coupled_mode_matrices` to 1e-10 relative in 4 cases with TE and TM modes; the opposite TE/TM sign fails. |
| Conventions | `CONVENTIONS.md` | units and phasors, local and global frames, modes and bases, fields, amplitude normalization (unconjugated reaction, symmetric GSMs), GSM definition, LS modes and the TE/TM ↔ LS rotation, tolerances, data files, code rules |
| Mode basis | `src/wrmodes/modes.py` | `Mode`, `ModeBasis` (ordering compatible with `select_coupled_modes`), $\beta$ branch with $\operatorname{Im}\beta \le 0$, admittances, fields, `ls_rotation` |
| GSM core | `src/wrmodes/gsm.py`, `tests/test_gsm.py` | `GSM`, Redheffer star product, `cascade`, `straight` (evanescent modes decay), `from_transfer_matrix`, `block_diag`, reciprocity/unitarity/passivity checks |
| Regression lock of 0.1 | `tests/data/make_regression_v0_1.py`, `tests/data/regression_v0_1.npz`, `tests/test_regression_v0_1.py` | 32 SPARC cases (WR-90 H/E plane, 60°/60 cm and 90°/40 cm, sech and constant, 20/45/70/90 GHz), full T and R |
| CI | `.github/workflows/tests.yml` | pytest on Ubuntu (Python 3.9, 3.13), macOS and Windows (3.12) |
| Task cards | `tasks/` | process, roles, template; T1–T3, L1, WP1–WP4, WP7 for wave 1 |

## Numbers

* Test suite: 87 tests, all passing (20 s): the 27 tests of 0.1 plus 60 new ones.
* The 0.1 bend matrices are exactly reciprocal and unitary in the new GSM form (defects about 3e-14),
  confirming the amplitude normalization of `CONVENTIONS.md` §5.
* GSM cascade of two bends and a straight equals the 0.1 transfer-matrix product to 1e-12.

## Findings worth knowing

* **Constants.** `scipy.constants` gives $\mu_0\varepsilon_0c^2 = 1$ only to about 1e-10 (CODATA values).
  wrmodes now defines $\eta_0 = \mu_0 c$ and never uses `epsilon_0` together with `mu_0` and `c`.
* **Branch of β.** wrmodes 0.1 uses `sqrt(k² − χ² + 0j)`, which gives $+i|\beta|$ below cutoff: a growing
  forward wave. That never mattered, because 0.1 only keeps propagating modes. The new `modes.propagation_constant`
  uses $-i|\beta|$; the 0.1 functions are unchanged.
* **Label correction to the plan.** In LS terms, the H-plane TE₁₀ problem (family $n = 0$) is LSM (TM to $v$:
  $E_v$, Dirichlet). The plan's physics section had the LSE/LSM labels swapped; `CONVENTIONS.md` §7 is
  authoritative.

## Open items

* Version string set to `1.0.0.dev0`. The 0.1 API is unchanged.
* The `fem` optional extra is declared (`scikit-fem`, `gmsh`) but not used until WP3.
