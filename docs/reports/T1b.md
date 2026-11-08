# T1b report: ohmic loss and impedance walls (independent second derivation of card T1)

Branch `wave-1/T1b-wall-impedance`. Note: `docs/theory/T1b-wall-impedance.md`. Scripts: `docs/theory/T1b/`.
I did not look at T1's worktree or branch.

## What was done

* Derived the loss terms by **Lorentz reciprocity** with perfect-conductor test modes. This gives an exact
  identity, $da^\pm_p/dz=\mp i\beta_pa^\pm_p\pm\frac14\oint Z_wH_{\tan}\cdot h_p^\mp\,dl$. It is first order in
  $Z_w$ once $H_{\tan}$ is replaced by the PEC modal sum. I then checked the result term by term against the T0
  projection with its wall terms kept: the $dV/dz$ and $dI/dz$ form of the note, eq. (VI).
* Final formulas (all boxed in the note):
  * $\Lambda^{++}=\Lambda^{--}=\tfrac12[\sqrt{Y_p}\sqrt{Y_q}A+B/((k\eta_0)^2\sqrt{Y_p}\sqrt{Y_q})]$;
  * $\Lambda^{+-}=\Lambda^{-+}$, the same with $-A$;
  * $A=\sum_wZ_w\int(e_p\cdot\hat n)(e_q\cdot\hat n)$ and $B=\sum_wZ_w\int g_pg_q$;
  * closed forms per wall, for TE–TE, TM–TM and TE–TM pairs, for the same and different $(m,n)$;
  * $dy/dz=(D+\kappa(K-\Lambda^{(1)}_{\text{full}})-\Lambda_{\text{full}})y$ with
    $\Lambda_{\text{full}}=[[\Lambda^{++},\Lambda^{+-}],[-\Lambda^{+-},-\Lambda^{++}]]$.
* Bends: the factor $h=1+\kappa u$ on the wall integrals gives first moments $\Lambda^{(1)}$, in closed form. The
  result is exact in κ to first order in $Z$, so no κZ cross term is dropped.
* Item 5: the exact LSE$^u$/LSM$^u$ modes of a guide with two impedance walls (dispersion relations, fields,
  normalization), mode matching to the ideal guide, the limit that gives Butterweck's eq. 21, and where first order
  fails.
* Item 6: the exact LSE$^v$/LSM$^v$ modes with resistive sheets (closed forms for one off-centre sheet, and a
  transverse transfer matrix for any number of sheets), with the junction overlaps.
* `t1b_coefficients.py` has one function per boxed formula, for the numerical comparison with T1. Inputs: mode tuples
  `('TE'|'TM', m, n)`, `a` and `b` in m, `f` in Hz, `Z` in ohm, either a scalar or a dict over `'u-','u+','v-','v+'`.

## Acceptance tests

| Check | Target | Result | |
| --- | --- | --- | --- |
| Diagonal vs textbook TE/TM (Marcuvitz §2.2 eqs. 4, 10), WR-90 and 20 mm square, 18–90 GHz | 1e-6 rel. | 7.4e-14. Sympy shows the identity holds exactly for TE$_{mn}$, TE$_{m0}$, TE$_{0n}$, TM$_{mn}$ and the card's TE$_{10}$ form | pass |
| Coefficients vs direct surface integrals (all 4 blocks, 31×31 pairs, 3 guides, a different complex $Z$ per wall; bend weights too) | 1e-8 rel. | 7.5e-15; power loss $\frac12R_s\oint\lvert H\rvert^2=2\operatorname{Re}\Lambda_{pp}$ to 5e-15 | pass |
| Butterweck lossy wall, first order vs his theory (4 and 36 dB/m) | 5 % | 3.869 dB/m (−3.3 %), 35.950 dB/m (−0.1 %); equal to his eq. 21 to 1e-15 | pass |
| Butterweck lossy wall, exact method vs measurements (0.6, 2.5, 5.4 dB, 15 cm) | 10 % | 0.569 / 2.317 / 5.341 dB (−5.1 / −7.3 / −1.1 %); H$_{10}$→H$_{30}$ 0.0211 (−33.5 dB), inside his 0.016–0.06 | pass |
| Resistive sheet, b = 10 mm, λ = 4 mm (56 Ω, 1.5 dB/cm) | 5 % | 55.75 Ω (−0.4 %), 1.554 dB/cm (+3.6 %); EH$_{11}^{(1)}$: 418 Ω, 5.25 dB/cm (his ≈420 Ω, ≈5 dB/cm) | pass |
| T1 and T1b identical | — | for the coordinator; the functions above are ready for it | not done here |

Further independent checks:
* First order against exact LS modes, for impedance on $u$-walls or on $v$-walls with two different complex $Z$:
  the error is $O(Z^2)$, two decades per decade of $Z$.
* The degenerate-pair eigenvectors are exactly the LS combinations, which fixes the sign of the TE–TM terms.
* For $R_\square\to\infty$ the sheet junction is the `ls_rotation` matrix $Q$ to 1e-14.
* Lossy-straight GSM: reciprocal to 6e-13 and passive (WR-90 aluminium, 126 modes, 90 GHz).
* Full test suite: 93 passed.

## Results (WR-90, aluminium σ = 3.5e7 S/m)

* TE$_{10}$: 0.1227 dB/m at 18 GHz, 0.2303 dB/m at 90 GHz.
* Degenerate TE$_{mn}$/TM$_{mn}$ pairs couple strongly:
  * $|\Lambda_{\text{TE,TM}}|$ up to 0.18 dB/m;
  * loss eigenmodes rotated 1–45° from pure TE/TM;
  * TE→TM conversion up to 3.9e-4 of the power after 1 m, and 3.4e-2 after 20 m.
  * With all four walls equal, the coupling is $\propto(a-b)$, so it vanishes in a square guide.
* Non-matched pairs: forward conversion ≤ 5.8e-6, backward ≤ 3.6e-7 (all families, 18–90 GHz).
* SPARC bends: dropping $\kappa\Lambda^{(1)}$ changes the main transmission by ≤ 3.1e-3 dB, but a converted power by
  up to 6.4e-4 (above the 1e-4 target). Keep it.

## Open issues

1. **The PEC-basis coupled system is correct only to $O(Z)$, and its $O(Z^2)$ error grows with the number of modes.**
   At 250 Ω, the H$_{10}$ eigen-attenuation is off by −13 % with the 11 propagating modes and by −67 % with 40
   modes. For aluminium it is ≤ 1.6e-4 of $\Lambda_{pp}$, so negligible. Large impedances must use the exact
   modes, as WP2 item 4 already plans.
2. **Validity limits of first order.**
   * LSM-type modes need $ka|Z|/\eta_0\ll1$. At 250 Ω first order overestimates the attenuation of the LSM$^u$ modes ($n\ge1$)
     of Butterweck's filter by a factor 14–29.
   * Near cutoff it needs $|\Lambda_{pp}|\ll\beta_p$. For aluminium this fails within about 0.8 % above cutoff.
3. Mode matching to the sheet guide converges slowly (TE$_{11}$: −12.72 / −12.89 / −12.97 dB with 6 / 10 / 14 modes
   per type). WP2 should show convergence.
4. The $\kappa\Lambda^{(1)}$ terms have no independent exact check yet. Both the closed forms and the quadrature
   come from the same derivation. The 2D solver with impedance walls (wave 2) can provide one.
5. The roots are found by continuation from the PEC values. This is robust in every case I ran, but I have not
   proved that it finds every root for arbitrary complex $Z$.

## Conventions issues

* CONVENTIONS.md §3 says that components that do not vary along $v$ couple only modes of the same $n$. Loss on the
  $v$-walls couples $n$ with $n'$:
  * when $Z_{v-}=Z_{v+}$, only $n+n'$ even (e.g. TE$_{10}$–TE$_{12}$);
  * otherwise, all $n'$.

  The effect is ≤ 6e-6 in power for metal walls. Still, a component with lossy broad walls is not strictly
  family-conserving, and WP2 should say so if it works per family.
* Nothing else. The card's sign structure for $\Lambda_{\text{full}}$ is consistent with my derivation, and
  $Z_s=(1+i)R_s$ is consistent with the $e^{+i\omega t}$ Leontovich condition (checked with a plane wave).
