# T3 — Coupled-mode equations of twisted rectangular guides

Role: Theory · Wave: 1 · Branch: `wave-1/T3-twist-coupling` · Depends on: T0
Verified by: T3b (independent derivation), then Doane & Anderson Fig. 9 and COMSOL in wave 2 ·
Time box: 4 days (then status review)

## Goal

Derive the coupled-mode equations of a rectangular guide whose cross-section rotates about the axis at a rate
$\theta'(z)$ (twist), in the full $(m, n)$ basis of `CONVENTIONS.md`, and the frame bookkeeping at its ends.
WP6 (wave 2) implements it.

## Read first

* `CONVENTIONS.md` (in particular §2, frame changes), `docs/theory/T0-field-convention.md`.
* Katsenelenbaum (1998), pp. 139–158 (PDF pp. 156–175): twisted waveguides.
* Doane & Anderson, Microwave J. 32(3), 153–160 (1989), p. 159 (PDF p. 7):
  * eq. 17, the triangular twist rate;
  * eq. 18, Unger's estimate $P_c \simeq (8c\theta)^2/(\Delta\beta L_t)^4$ with $c\approx 8/\pi^2$ for TE₁₀–TE₀₁;
  * Fig. 9, 20-inch 90° twists in WR187 for constant, cosine and triangular rates.
* Unger, BSTJ 36, 1292–1307 (1957): conversion for linearly tapered coupling (the source of eq. 18).

## Specification

1. **Twisted frame.** Use local coordinates $(u, v)$ co-rotating with the cross-section, with $\theta(z)$ the
   rotation from the port-1 frame. Derive the coupled equations from the T0 expansion. The coupling comes from
   the rotation generator acting on the fields:
   $\partial_\theta e = \hat z\times e - (u\,\partial_v - v\,\partial_u)e$ for an active rotation
   $e_\theta(r) = R(\theta)\,e(R(-\theta)r)$. Check: the rotation-invariant field $e = (u, v)$ gives 0. Coefficients are
   proportional to $\theta'(z)$.
2. **Closed forms.** Give the coupling coefficients for all TE/TM pairs. State the selection rules: which $(m, n)$
   couple to which, and that a twist mixes families. Check that TE₁₀–TE₀₁ gives
   $c = \iint e_{01}\cdot\partial_\theta e_{10}\,ds = 8/\pi^2$, independent of the aspect ratio (coordinator's
   reviewer found 0.8106 by quadrature for a square guide and for WR187), as in Doane.
3. **Square-guide limit** ($a = b$). TE₁₀ and TE₀₁ are degenerate, so their coupling $c\,\theta'$ is resonant. In the
   two-mode truncation the polarization rotates by $-c\theta$ in the local (co-rotating) frame, i.e. by
   $(1 - 8/\pi^2)\theta$ in the fixed frame: the field does **not** follow the guide, unlike the circular-guide case
   where $c = 1$ and it stays fixed. Derive this two-mode result exactly, then quantify how the other modes of the
   full basis change it. Both are analytic or quasi-analytic tests for WP6.
4. **End frames.** A twist by $\theta_{tot}$ ends in a frame rotated by $\theta_{tot}$. For 90° it equals the
   E/H-plane frame change of `CONVENTIONS.md` §2. Specify the `FrameChange` that follows it, and check that a
   +90° twist, a frame change and a −90° twist compose to identity at zero length.
5. **Rate profiles.** Constant, cosine and triangular (Doane eq. 17). Give the first-order conversion integral
   for each, to compare with Fig. 9. Eq. 18 behaves as an envelope: the coordinator's reviewer found 0.30 / 0.47 /
   0.68 % at 15 / 16.5 / 18 GHz from eq. 18 for the 20-inch WR187 twist, against 0.0054 / 0.0001 / 0.045 % from
   the first-order integral. Check both, including the transcription of eq. 18 against the scan.

## Acceptance tests

| Check | Target |
| --- | --- |
| Closed forms vs quadrature of $\iint e_p\cdot\partial_\theta e_q$ | 1e-10 relative |
| Square guide, two-mode truncation | rotation by $-c\theta$ in the local frame, exact |
| First-order integral for the WR187 20-inch triangular twist | same order as Fig. 9 (below 0.1 % at 15–18 GHz) |
| T3 and T3b, derived independently | identical formulas (or a discrepancy report) |

## Deliverables

* `docs/theory/T3-twist-coupling.md`, with boxed formulas and the frame bookkeeping.
* `docs/theory/T3/` check scripts.

## Out of scope

Implementation (WP6), twists combined with bends in the same section, non-rectangular guides.
