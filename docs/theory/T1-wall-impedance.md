# T1 — Ohmic loss and impedance walls in the coupled-mode equations

*Theory note, wave 1 (card `tasks/T1-wall-impedance.md`). Status: derived; every boxed formula is checked by the
scripts in `docs/theory/T1/` (section 10). Conventions: `CONVENTIONS.md` throughout ($e^{+i\omega t}$, forward
modes $e^{-i\beta z}$, fields of `wrmodes.modes`, amplitudes normalized with the unconjugated reaction, principal
square roots). WP2 implements the boxed results B1–B10 as written.*

Notation. Local frame $(u, v, z)$ with $u\in[-a/2,a/2]$, $v\in[-b/2,b/2]$, $u' = u + a/2$, $v' = v + b/2$.
$k_m = m\pi/a$, $k_n = n\pi/b$, $\chi_p^2 = k_m^2 + k_n^2$, $\eta_0 = \mu_0c$ (so $\omega\mu_0 = k\eta_0$,
$\omega\varepsilon_0 = k/\eta_0$). The four walls are labelled by their outward normal $\hat n$ (pointing from the
guide into the wall):

| wall | position | $\hat n$ | tangent $\hat t = \hat z\times\hat n$ |
| --- | --- | --- | --- |
| `u+` | $u = +a/2$ (outer wall of a bend with κ > 0) | $+\hat u$ | $+\hat v$ |
| `u-` | $u = -a/2$ | $-\hat u$ | $-\hat v$ |
| `v+` | $v = +b/2$ | $+\hat v$ | $-\hat u$ |
| `v-` | $v = -b/2$ | $-\hat v$ | $+\hat u$ |

Each wall $w$ has its own impedance $Z_w$ (Ω); a perfect wall has $Z_w = 0$.

## 1. Leontovich condition (card item 1)

$$\boxed{\;E_{\tan} = Z_s\,H_{\tan}\times\hat n,\qquad \hat n \text{ from the guide into the wall},\qquad
Z_s = \sqrt{\frac{i\omega\mu_0}{\sigma}} = (1+i)R_s,\quad R_s=\sqrt{\frac{\omega\mu_0}{2\sigma}}\;}\tag{B1}$$

Sign check for $e^{+i\omega t}$ (sympy S1): inside the conductor the field varies as $e^{-\gamma x}$ along $\hat n$
with $\gamma = \sqrt{i\omega\mu_0\sigma} = (1+i)/\delta$; Faraday's law gives $H = (\gamma/i\omega\mu_0)\,\hat n\times E$,
hence $E_{\tan} = (i\omega\mu_0/\gamma)\,H\times\hat n$ and $Z_s = i\omega\mu_0/\gamma = (1+i)R_s$. The power flowing
into the wall is $\tfrac12\mathrm{Re}(E\times H^*)\cdot\hat n = \tfrac12\mathrm{Re}(Z_s)|H_{\tan}|^2\ge 0$. This is
Butterweck's eq. (20) with his outward normal (his $e^{j\omega t}$ is the same convention). In the wall frame
$(\hat n,\hat t,\hat z)$, B1 reads

$$E_t = Z\,H_z,\qquad E_z = -Z\,H_t \qquad(E_t = E\cdot\hat t,\ H_t = H\cdot\hat t).$$

## 2. Projection with boundary terms

Start from T0: $E_t = \sum_q V_q e_q$, $H_t = \sum_q I_q\,\hat z\times e_q$ (transverse fields), i.e.
$V_p = \iint E_t\cdot e_p$, $I_p = \iint H_t\cdot(\hat z\times e_p)$, with the scale factor $h = 1 + \kappa u$ and
$d_p = \nabla_t\cdot e_p = -\chi_p^2\Pi_p$ (TM; 0 for TE), $g_p = \nabla_t\cdot(\hat z\times e_p) = -\chi_p^2\Gamma_p$
(TE; 0 for TM). Project the two transverse Maxwell equations of T0 and integrate by parts, now keeping every
boundary term. On *any* wall the ideal modes satisfy $e_p\cdot\hat t = 0$ and $d_p = 0$, so the boundary terms
that contain $e_p\cdot\hat t$, $e_q\cdot\hat t$ or $d_p$ still vanish. The two that survive contain the actual wall
values of $E_z$ and $E_t$, which vanish only on a perfect wall:

$$\frac{dV_p}{dz} = [\text{T0 terms}] + \oint h\,E_z\,(e_p\cdot\hat n)\,dl,\qquad
\frac{dI_p}{dz} = [\text{T0 terms}] - \frac{i}{\omega\mu_0}\oint h\,g_p\,E_t\,dl .\tag{1}$$

The T0 terms are unchanged: $-i\omega\mu_0\sum_q I_q\iint h\,e_p\cdot e_q + \frac{i}{\omega\varepsilon_0}\sum_q I_q\iint h\,d_pd_q$
and $-i\omega\varepsilon_0\sum_qV_q\iint h\,e_p\cdot e_q + \frac{i}{\omega\mu_0}\sum_q V_q\iint h\,g_pg_q$. Equations (1)
are exact for any Maxwell field, whatever its wall condition (sympy S2 checks them for generic LSE$^u$ and LSM$^u$
fields with arbitrary transverse wavenumber, projected on TE and TM modes).

**Leontovich substitution.** With B1, $E_z = -Z H_t$ and $E_t = Z H_z$ on the wall. To first order in $Z$ the wall
fields are those of the mode series: $H_t = \sum_q I_q\,(\hat z\times e_q)\cdot\hat t = \sum_q I_q\,(e_q\cdot\hat n)$ and
$H_z = \tfrac{i}{\omega\mu_0}\nabla_t\cdot(E_t\times\hat z) = -\tfrac{i}{k\eta_0}\sum_q V_q g_q$. Hence

$$\boxed{\;\frac{dV_p}{dz} = [\text{T0}] - \sum_q N_{pq} I_q,\qquad
\frac{dI_p}{dz} = [\text{T0}] - \sum_q G_{pq} V_q,\qquad
N_{pq} = \sum_w Z_w\!\oint_w\! h\,(e_p\cdot\hat n)(e_q\cdot\hat n)\,dl,\quad
G_{pq} = \frac{1}{(k\eta_0)^2}\sum_w Z_w\!\oint_w\! h\,g_p\,g_q\,dl\;}\tag{B2}$$

$N$ is a series impedance per unit length (transverse wall currents $\propto H_t$), $G$ a shunt admittance per unit
length (longitudinal wall currents $\propto H_z$, TE modes only). Both are symmetric.

### Closed forms of the wall integrals

Write $c_p = 2/(\sqrt{ab}\,\chi_p)$ (TM) or $\sqrt{\epsilon_m\epsilon_n}/(\sqrt{ab}\,\chi_p)$ (TE) for the prefactor of
eq. 5.4 ($\epsilon_0 = 1$, $\epsilon_{l>0} = 2$). The normal fields on the walls are

| | on `u±`: $e\cdot\hat n = \pm(\pm1)^{m}A_p\sin k_nv'$ | on `v±`: $e\cdot\hat n = \pm(\pm1)^{n}B_p\sin k_mu'$ |
| --- | --- | --- |
| TM$_{mn}$ | $A_p = c_pk_m$ | $B_p = c_pk_n$ |
| TE$_{mn}$ | $A_p = -c_pk_n$ | $B_p = c_pk_m$ |

and $g_p = -\chi_p^2c_p\cos k_mu'\cos k_nv'$ (TE). With the wall signs $s^{u+}_{pq} = (-1)^{m_p+m_q}$,
$s^{u-}_{pq} = 1$, $t^{v+}_{pq} = (-1)^{n_p+n_q}$, $t^{v-}_{pq} = 1$, write the wall integrals of B2 as
$\oint_w h\,(\dots) = W^{w,0} + \kappa\,W^{w,1}$ (straight part and coefficient of κ; $u$ is the centred coordinate):

$$\boxed{\begin{aligned}
N^{u\pm,0}_{pq} &= A_pA_q\,s^{u\pm}_{pq}\,\tfrac b2\,\delta_{n_pn_q}[n\ge1], &
\Gamma^{u\pm,0}_{pq} &= \chi_p^2\chi_q^2c_pc_q\,s^{u\pm}_{pq}\,\tfrac{b}{\epsilon_n}\,\delta_{n_pn_q}, &
W^{u\pm,1} &= \pm\tfrac a2\,W^{u\pm,0},\\
N^{v\pm,0}_{pq} &= B_pB_q\,t^{v\pm}_{pq}\,\tfrac a2\,\delta_{m_pm_q}[m\ge1], &
\Gamma^{v\pm,0}_{pq} &= \chi_p^2\chi_q^2c_pc_q\,t^{v\pm}_{pq}\,\tfrac{a}{\epsilon_m}\,\delta_{m_pm_q}, \\
N^{v\pm,1}_{pq} &= B_pB_q\,t^{v\pm}_{pq}\,S^1_{m_pm_q}, &
\Gamma^{v\pm,1}_{pq} &= \chi_p^2\chi_q^2c_pc_q\,t^{v\pm}_{pq}\,C^1_{m_pm_q},
\end{aligned}}\tag{B2a}$$

with $\Gamma$ (TE–TE only) the integral of $g_pg_q$, so that $N_{pq} = \sum_w Z_w(N^{w,0}+\kappa N^{w,1})$ and
$G_{pq} = (k\eta_0)^{-2}\sum_wZ_w(\Gamma^{w,0}+\kappa\Gamma^{w,1})$, and, for $m+m'$ odd (zero otherwise, also for $m = m'$),

$$S^1_{mm'} = \int_0^a\!\big(u'-\tfrac a2\big)\sin k_mu'\sin k_{m'}u'\,du' = -\frac{a^2}{\pi^2}\frac{4mm'}{(m^2-m'^2)^2},\qquad
C^1_{mm'} = \int_0^a\!\big(u'-\tfrac a2\big)\cos k_mu'\cos k_{m'}u'\,du' = -\frac{a^2}{\pi^2}\frac{2(m^2+m'^2)}{(m^2-m'^2)^2}.$$

Pairs explicitly (products $A_pA_q$ and $B_pB_q$): side walls TE–TE $c_pc_qk_n^2$, TM–TM $c_pc_qk_{m_p}k_{m_q}$,
TM–TE $-c_pc_qk_{m_p}k_n$; broad walls TE–TE $c_pc_qk_m^2$, TM–TM $c_pc_qk_{n_p}k_{n_q}$, TM–TE $c_pc_qk_mk_{n_q}$.
Consequences:

* side walls couple only modes with the **same $n$** (any $m$, $m'$; with equal $Z$ on both, only $m+m'$ even);
* broad walls couple only modes with the **same $m$** (any $n$, $n'$; with equal $Z$, only $n+n'$ even), i.e. they
  couple *different families*. This breaks the family-block structure of `CONVENTIONS.md` §3 at $O(Z)$; the
  effect is non-synchronous and negligible for metal walls (section 5);
* in a bend, the broad walls add a κ-coupling between $m$ and $m'$ of opposite parity.

The degenerate pair TE$_{mn}$/TM$_{mn}$ ($m,n\ge1$) has $c_{TE}c_{TM} = 4/(ab\chi^2)$ and, straight guide,
$N_{EM} = \frac{2k_mk_n}{ab\chi^2}\big[a(Z_{v+}+Z_{v-}) - b(Z_{u+}+Z_{u-})\big]$, $G_{EM} = 0$.

## 3. Power waves (card item 2)

With $V_p = (a^+_p+a^-_p)\sqrt{2/Y_p}$, $I_p = (a^+_p-a^-_p)\sqrt{2Y_p}$ (sympy S3):

$$\boxed{\;\Lambda^{\pm}_{pq} = \tfrac12\Big[\pm\sqrt{Y_p}\sqrt{Y_q}\,N_{pq} + \frac{G_{pq}}{\sqrt{Y_p}\sqrt{Y_q}}\Big]\quad
\text{(principal roots)}\;}\tag{B3}$$

$$\boxed{\;\begin{aligned}
\frac{da^+_p}{dz} &= -i\beta_pa^+_p - \textstyle\sum_q\big(\Lambda^+_{pq}a^+_q + \Lambda^-_{pq}a^-_q\big),\\
\frac{da^-_p}{dz} &= +i\beta_pa^-_p + \textstyle\sum_q\big(\Lambda^-_{pq}a^+_q + \Lambda^+_{pq}a^-_q\big),
\end{aligned}\qquad
\frac{dy}{dz} = \big(D + \kappa K - \Lambda_{\rm full}\big)y,\quad
\Lambda_{\rm full} = \begin{bmatrix}\Lambda^+ & \Lambda^-\\ -\Lambda^- & -\Lambda^+\end{bmatrix}\;}\tag{B4}$$

with $y = [a^+; a^-]$, $D = \mathrm{diag}(-i\beta, +i\beta)$ and $K$ the wrmodes curvature matrix. In the card's
notation $\Lambda^{++} = \Lambda^{--} = \Lambda^+$ and $\Lambda^{+-} = \Lambda^{-+} = \Lambda^-$. $\Lambda^\pm$ are symmetric
(reciprocity) and linear in the $Z_w$. The structure is that of T0's $C^\pm$ (with $\Lambda\leftrightarrow i\kappa C$).
For the degenerate pair, $\sqrt{Y_{TE}}\sqrt{Y_{TM}} = 1/\eta_0$ on both sides of cutoff, so
$\Lambda^\pm_{EM} = \pm N_{EM}/(2\eta_0)$.

**Diagonal = textbook attenuation.** For a propagating mode with $Z = (1+i)R_s$ on all walls,
$\Lambda^+_{pp} = (1+i)\alpha_p$: $\mathrm{Re}\,\Lambda^+_{pp}$ is the attenuation constant and the phase constant
grows by the same amount, $\beta\to\beta + \alpha$. $\mathrm{Re}\,\Lambda^+_{pp}$ equals Marcuvitz (1951) §2.2
eq. (4) for E$_{mn}$ = TM$_{mn}$ and eq. (10) for H$_{mn}$ = TE$_{mn}$ symbolically (sympy S5), and the card's TE₁₀
formula $\alpha = R_s/(\eta_0b\sqrt{1-(f_c/f)^2})\,[1 + (2b/a)(f_c/f)^2]$ (S5d). For one mode, B4 is the transmission
line with series impedance $i\beta/Y + N$ and shunt admittance $i\beta Y + G$ (S3e).

**Power balance.** For propagating modes, $-\frac{d}{dz}\sum_p(|a^+_p|^2 - |a^-_p|^2)$ computed from $\Lambda_{\rm full}$
equals $\tfrac12\oint h\,\mathrm{Re}(Z)\,(|H_t|^2 + |H_z|^2)\,dl$ of the reconstructed wall fields, for complex,
unequal $Z_w$ and in a bend (check Q7, 1.5e-14). This is the dissipation WP2's power-balance test needs.

## 4. Exactly solvable cases: one pair of impedance walls

### 4.1 First-order LS eigenvalues

With impedance on one pair of opposite walls only, the exact modes are longitudinal-section modes with respect to
the **wall normal**. For the side walls ($u = \pm a/2$) these are LS modes with respect to $u$, not the LS$^v$ modes of
`CONVENTIONS.md` §7:

* LSE$^u$ ($E_u = 0$): $\Phi = X(u')\cos(k_nv')e^{-i\gamma z}$, $E = -\nabla\times(\hat u\Phi)$; contains TE$_{m0}$;
* LSM$^u$ ($H_u = 0$): $\Psi = X(u')\sin(k_nv')e^{-i\gamma z}$, $H = \nabla\times(\hat u\Psi)$; contains TE$_{0n}$.

The 2×2 blocks $\Lambda^+$ of the pairs (TE$_{mn}$, TM$_{mn}$) have exactly these eigenvalues (check Q5, 1e-15), and
for the broad walls alone $\Lambda^+$ is diagonal in the LS$^v$ basis of `ls_rotation` (Q5, 2e-16):

$$\boxed{\;\Lambda_{\rm LSE^u} = \frac{k_m^2(Z_{u+}+Z_{u-})}{\eta_0ka\beta},\quad
\Lambda_{\rm LSM^u} = \frac{k(Z_{u+}+Z_{u-})}{\eta_0a\beta}\,\big(\tfrac12\text{ if }m=0\big),\quad
\Lambda_{\rm LSE^v} = \frac{k_n^2(Z_{v+}+Z_{v-})}{\eta_0kb\beta},\quad
\Lambda_{\rm LSM^v} = \frac{k(Z_{v+}+Z_{v-})}{\eta_0b\beta}\,\big(\tfrac12\text{ if }n=0\big)\;}\tag{B5}$$

For TE$_{m0}$ with lossy side walls this is Butterweck's eq. (21), $\alpha_m = 2\pi^2m^2\,\mathrm{Re}Z_w/(\beta_ma^3kZ_0)$.

### 4.2 Exact dispersion relation (card item 5)

With $X(u')$ satisfying B1 at both walls (sympy S6a,b derive it from the field ansatz):

$$\boxed{\;F(k_u) = \big(k_u^2 + k^2p_+p_-\big)\sin k_ua - i\,k\,k_u\,(p_++p_-)\cos k_ua = 0,\qquad
p_\pm = \frac{\eta_0}{Z_{u\pm}}\ (\text{LSE}^u),\quad p_\pm = \frac{Z_{u\pm}}{\eta_0}\ (\text{LSM}^u),\qquad
\gamma^2 = k^2 - k_n^2 - k_u^2\;}\tag{B6}$$

($k_u = 0$ is a trivial root; Im γ ≤ 0.) Profiles: LSE$^u$ $X = \sin k_uu' - i\frac{k_uZ_{u-}}{k\eta_0}\cos k_uu'$,
LSM$^u$ $X = \cos k_uu' + i\frac{kZ_{u-}}{\eta_0k_u}\sin k_uu'$. Equal walls factorize into even and odd modes,
$(k_u\sin\frac{k_ua}2 - ikp\cos\frac{k_ua}2)(k_u\cos\frac{k_ua}2 + ikp\sin\frac{k_ua}2) = 0$ (S6c). The roots tend to
$m\pi/a$ as $Z\to0$; to first order $k_u = \frac{m\pi}a\big(1 + i\frac{Z_{u+}+Z_{u-}}{\eta_0ka}\big)$ for LSE$^u$ (S6d), which
gives B5 and, for $m = 1$ and real symmetric $Z_w$, **Butterweck's eq. (29)**
$\gamma_1^2 \approx \pi^2/a^2 - k^2 - (\pi^2/a^2)\,4Z_w/(jZ_0ka)$ (his $e^{-\gamma z}$), and from it eq. (21) (S6e,f).
A robust root finder (Chebyshev collocation of $-X'' = k_u^2X$ with the Robin conditions, then Newton on
$F/k_u$) is `t1lib.lsu_roots_all`; continuation in $Z$ from 0 jumps branches for high orders at $|Z|\sim Z_0$.

**Junction ideal guide → impedance-walled guide.** Let the impedance-walled modes $i$ have transverse fields
$(\mathcal E_i, \mathcal H_i)$ normalized by $\tfrac12\iint(\mathcal E_i\times\mathcal H_i)\cdot\hat z = 1$ (unconjugated; they are
mutually orthogonal in this reaction, check 1e-12). Test E continuity with $e_p$ and H continuity with
$\mathcal E_i$ (reciprocity), with $a, b$ the incident/reflected amplitudes on the ideal side and $c, d$ the
forward/backward amplitudes on the other:

$$\boxed{\;a + b = P\,(c + d),\quad c - d = P^{T}(a - b),\quad P_{pi} = \sqrt{Y_p/2}\iint\mathcal E_i\cdot e_p\,ds;\qquad
S_{11} = 2PWP^T - I,\ S_{12} = 2PW,\ S_{21} = 2WP^T,\ S_{22} = 2W - I,\ W = (I + P^TP)^{-1}\;}\tag{B9}$$

The second junction of a filter is the same GSM with its ports swapped. $S$ is symmetric by construction. For
family $n = 0$ (TE$_{m0}$ ↔ LSE$^u$) the fields are $\mathcal E = \sqrt{2/Y_i}\,f_i\hat v$, $\mathcal H = -\sqrt{2Y_i}f_i\hat u$ with
$f_i = X_i/\sqrt{b\int X_i^2}$, $Y_i = \gamma_i/(k\eta_0)$, so $P_{pi} = \sqrt{Y_p/Y_i}\;b\!\int f_pf_i\,du'$.

## 5. Which off-diagonal terms matter (card item 3), WR-90 aluminium 18–90 GHz

Numbers from `wr90_aluminium.py` (σ = 3.5×10⁷ S/m, all four walls).

**Attenuation constants** (dB/m): TE₁₀ 0.1227 (18 GHz), 0.1410 (30), 0.1665 (45), 0.1898 (60), 0.2109 (75),
0.2303 (90); TE₂₀ 0.17–0.24; TE₀₁ 0.319 (18 GHz) → 0.109 (60) → 0.116 (90); TE₁₁/TM₁₁ 0.586/0.420 at 18 GHz,
0.263/0.422 at 90 GHz. Modes just above cutoff reach up to 13 dB/m. The phase constant grows by α (rad/m): 0.027 rad/m
for TE₁₀ at 90 GHz, 0.5 rad over 20 m.

**Degenerate pairs (resonant).** TE$_{mn}$ and TM$_{mn}$ have equal β, so $\Lambda^+_{EM}$ acts resonantly: the loss
eigenmodes are fixed rotations of (TE, TM). $|\Lambda_{EM}|/|\Lambda_{MM}-\Lambda_{EE}|$ is 0.25–20, i.e. the mixing is
strong. The lower-loss eigenmode is close to LSE$^v$ ($E_v = 0$, the field parallel to the broad walls): for
(1,1) its angle is −25.3° vs −24.5° for LSE$^v$ at 75 GHz, −24.9° vs −24.3° at 90 GHz; for (2,1) −43.5° vs −42.3°
at 90 GHz; at 18 GHz it is in between LS$^v$ and LS$^u$. A pure TE$_{mn}$ launched into a straight line becomes
this LS-like mixture: for the pairs listed by the script the TM$_{mn}$ fraction of the surviving power is
5e-5–4e-4 after 1 m and 3–17 % after 20 m (up to 34 % for TE₉,₃ at 90 GHz, over all pairs with β/k > 0.5). **WP2 must keep the TE/TM degenerate coupling.**

**Non-matched pairs (average out).** Forward: $\max|\Lambda^+_{pq}|/|\beta_p-\beta_q|\le 1.2\times10^{-3}$ (amplitude)
over all coupled non-degenerate pairs at all frequencies, $\le5.1\times10^{-4}$ across families (broad walls).
TE₁₀ over a 20 m straight line converts at most 1.5e-7 of its power in total (mostly TM₁₂, 90 GHz). Backward:
$|\Lambda^-_{pq}|/(\beta_p+\beta_q)\le 8\times10^{-5}$ for every pair involving TE₁₀; values above 1e-3 (up to
1.5e-2) occur only for modes close to cutoff ($f_c/f\ge0.98$). These
couplings are harmless and may be kept or dropped.

**Validity of the first-order system (important for WP2).** The ideal-mode series cannot represent the
non-zero tangential E on an impedance wall (its term-by-term derivative diverges there). Consequently the
*first-order* coupled system B4 contains spurious $O(Z^2)$ terms through the far-off-diagonal $\Lambda_{pq}$ that grow
with the number of modes kept. TE₁₀ eigenvalue of B4 vs the exact B6 root, family $n = 0$ with 3, 11, 40, 80
modes: at 250 Ω (Butterweck) $|\Delta\Lambda|/|\Lambda|$ = 0.14, 0.41, 0.85, 0.96 and the attenuation is off by
−0.5 %, −13 %, −67 %, −89 %; at 2.5 Ω $|\Delta\Lambda|/|\Lambda|$ = 1.4e-3, 4.4e-3, 1.5e-2, 3.0e-2 (mostly phase; the
attenuation is off by −5e-7, −1.5e-5, −2e-4, −8e-4). For aluminium WR-90
(two lossy walls, exact LS$^u$/LS$^v$ result as reference): the diagonal (Marcuvitz) value is within 4e-3 of the
exact Leontovich value (largest for TE₀₁ on the side walls at 90 GHz, a genuine $O(ka\,Z/\eta_0)$ correction of
the overmoded guide); B4 restricted to the propagating modes is within 3.5e-4; adding evanescent modes with
full coupling makes TE₁₀ worse (1e-3 at cutoff_factor 3). Rule for WP2: **couple only propagating modes through
Λ (cutoff_factor = 1, as the transfer-matrix solver does); evanescent modes get only their diagonal Λ_pp.** For
metal walls all of this is far below the accuracy of σ itself.

## 6. Bends (card item 4)

All steps of sections 2–3 keep $h = 1+\kappa u$ exactly (the wall element is $h\,dl\,dz$; Leontovich is local).
Hence, to first order in $Z$ and exactly in κ,

$$\boxed{\;\Lambda(\kappa) = \Lambda_0 + \kappa\,\Lambda_1,\qquad
\frac{dy}{dz} = \big[(D - \Lambda_{0,\rm full}) + \kappa(z)\,(K - \Lambda_{1,\rm full})\big]\,y\;}\tag{B4b}$$

with $\Lambda_0$ from $W^{w,0}$ and $\Lambda_1$ from $W^{w,1}$ (B2a), assembled as in B3–B4. The form is still
"constant + κ(z) × constant", so the Magnus solver of `transfer_matrix.py` applies unchanged with D and K
replaced (its commutator term is written for general matrices). The curvature matrix $K$ itself is unchanged
at $O(Z)$ (it depends on the ideal-guide admittances). Neglected: $O(Z^2)$ and $O(\kappa Z^2)$ terms, and the
spurious far-mode terms of section 5. The κZ cross term $\kappa\Lambda_1$: for equal walls its diagonal vanishes
(the side-wall weights ±a/2 cancel, the broad-wall $S^1_{mm}$, $C^1_{mm}$ vanish), so it only couples $m$ to $m\pm$odd.
Size for the SPARC bends (one family, Magnus with 1024/2048 slices):

| bend | orient. | f (GHz) | $\kappa_{\max}a/2$ | $\lVert\kappa\Lambda_1\rVert/\lVert\Lambda_0\rVert$ | in-mode loss: lossless / $+\Lambda_0$ / $+\kappa\Lambda_1$ (dB) | max change of a converted power |
| --- | --- | --- | --- | --- | --- | --- |
| 40 cm 90° sech | H | 18 | 0.117 | 4.0e-2 | 0.0000 / 0.0493 / 0.0495 | 1.9e-10 |
| 40 cm 90° sech | H | 90 | 0.117 | 6.0e-2 | 2.3206 / 2.4163 / 2.4193 | 1.3e-4 |
| 40 cm 90° sech | E | 90 | 0.052 | 2.0e-2 | 0.2877 / 0.4242 / 0.4272 | 3.0e-5 |
| 60 cm 60° const | H | 90 | 0.020 | 1.0e-2 | 7.9297 / 8.0695 / 8.0718 | 8.5e-5 |
| 60 cm 60° const | E | 90 | 0.009 | 3.4e-3 | 0.0015 / 0.1739 / 0.1749 | 1.6e-6 |

The cross term changes the in-mode loss by ≤ 3.1e-3 dB and converted powers by ≤ 1.3e-4 — at the level of the 1e-4
tolerance of `CONVENTIONS.md` §8. It costs nothing to include: **WP2 should include $\kappa\Lambda_1$**.

## 7. Large wall impedance: where first order fails (card item 5)

The expansion parameter is set by how strongly the wall perturbs the transverse wavenumber (exact B6 vs B5,
`check_coefficients.py` Q6 and `butterweck_lossy_wall.py`):

* LSE$^u$ (TE$_{m0}$-like; the wall field is the weak $H_z$): $\epsilon = |Z_{u+}+Z_{u-}|/(\eta_0ka)$. The relative
  error of the first-order α is of order ε (1.5ε for the complex test impedances of Q6). Butterweck's filter
  (250 Ω real, 75 GHz): ε = 0.037; first-order α within 0.16 % (m = 1), 0.7 % (m = 3), 2.5 % (m = 5) of the exact value.
* LSM$^u$ with $m\ge1$ (modes with $n\ge1$, strong $H_v$ on the wall): $\epsilon = ka|Z_{u+}+Z_{u-}|/(\eta_0\pi^2m^2)$
  (error ≈ 0.5ε in Q6). At 250 Ω, ε = 4.8 for m = 1: first order gives 516 dB/m for LSM$^u_{11}$, exact 36.6 dB/m.
* LSM$^u$ with $m = 0$ (TE$_{0n}$): $\epsilon = ka|Z_{u+}+Z_{u-}|/\eta_0$ (error ≈ 0.1ε in Q6); at 250 Ω, ε = 48:
  257 vs 8.9 dB/m.

So the Leontovich-perturbation treatment holds for TE$_{m0}$ in Butterweck's filter but **fails completely for every
mode with $n\ge1$**: their exact attenuation saturates and even decreases as $Z$ grows (the wall approaches a
magnetic wall for them). The exact method (B6 + B9) must be used for the lossy-wall filter. Leontovich itself
requires the field to vary slowly along the wall compared with the penetration depth (Butterweck's footnote 5);
for a dielectric absorber with $|Z|\sim Z_0$ this is an assumption of the model, not checked here.

### Butterweck's lossy-wall filter: results

X-band guide 22.86 × 10.16 mm, 75 GHz, $Z_w$ = 250 Ω real on both narrow walls, 15 cm (`butterweck_lossy_wall.py`):

| | H₁₀ | H₂₀ | H₃₀ | H₁₀→H₃₀ |
| --- | --- | --- | --- | --- |
| first order B5 = eq. (21) | 3.869 dB/m, 0.580 dB | 15.66 dB/m, 2.349 dB | 35.95 dB/m, 5.393 dB | |
| Butterweck theory | 4 dB/m, 0.6 dB | — | 36 dB/m, 5.4 dB | −36 … −24.5 dB |
| first-order B4 (3 modes TE₁₀–TE₃₀) | 0.573 dB | 2.349 dB | 5.394 dB | −32.9 dB |
| exact B6 + B9 (160 modes) | 0.569 dB | 2.315 dB | 5.337 dB | −33.55 dB |
| Butterweck measurement (Philips Tech. Rev. 29, p. 94–95) | 0.6 dB | 2.5 dB | 5.4 dB | about −30 dB |

The exact method converges (80 → 160 modes: 2e-3 dB, 1e-2 dB on the conversion), is reciprocal to 1e-15 and its
modes are orthogonal to 1e-12; the junction reflections are ≤ 1.6e-3. The H₁₀→H₃₀ conversion oscillates with the
beat length 12.8 cm (the filter is about one beat length long, as Butterweck notes); its envelope peaks at about
−26 dB.

**Discrepancy in Butterweck's eqs. (28)–(31).** Projecting $E_y(a/2)$ through his eqs. (24)–(27) gives a coupling
term with a factor ν/μ (sympy S8; it is also what makes the coupling symmetric, as in B3, where
$\Lambda_{\nu\mu}\propto\nu\mu/\sqrt{\beta_\nu\beta_\mu}$). The printed eq. (28), and hence (30)–(31), omit it. As printed, eq. (30)
gives −42.5 dB at 15 cm and eq. (31) gives −48.7 … −37.3 dB, not the quoted −36 … −24.5 dB; with the factor ν = 3,
eq. (30) gives −33.0 dB at 15 cm, 0.55 dB from the exact −33.55 dB (within 1.3 dB over 1–30 cm), and eq. (31)
gives −39.2 … −27.8 dB. The quoted band cannot be reproduced from the printed formulas; the exact value lies inside it.

## 8. Resistive sheet parallel to the broad walls (card item 6)

A sheet of resistance $R_\square$ at $v' = v_0$ ($\xi = R_\square/\eta_0$) carries $J = E_{\tan}/R_\square$:
$E_{\tan}$ continuous and $H_{\tan}(v_0^-) - H_{\tan}(v_0^+) = R_\square^{-1}\,\hat v\times E_{\tan}$ (Butterweck eq. 2). It
does not vary along $u$, so the modes are LS$^v$ modes (`CONVENTIONS.md` §7) with $u$-dependence $\sin k_mu'$ or
$\cos k_mu'$ and a piecewise $Y(v')$, $\gamma^2 = k^2 - k_m^2 - k_v^2$:

* LSM$^v$ ($H_v = 0$, Butterweck's EH$^{(2)}$): $\Psi = \sin k_mu'\,Y(v')$, $H = \nabla\times(\hat v\Psi)$; $Y'(0) = Y'(b) = 0$,
  $Y'$ continuous, $Y(v_0^+) = Y(v_0^-) - iY'(v_0)/(k\xi)$;
* LSE$^v$ ($E_v = 0$, EH$^{(1)}$): $\Phi = \cos k_mu'\,Y(v')$, $E = -\nabla\times(\hat v\Phi)$; $Y(0) = Y(b) = 0$, $Y$
  continuous, $Y'(v_0^+) = Y'(v_0^-) + ikY(v_0)/\xi$.

With $d_1 = v_0$, $d_2 = b - v_0$ (sympy S7 derives both from the field ansatz):

$$\boxed{\;\text{LSM}^v:\ \xi k\sin k_vb + i\,k_v\sin k_vd_1\sin k_vd_2 = 0,\qquad
\text{LSE}^v:\ \xi k_v\sin k_vb + i\,k\sin k_vd_1\sin k_vd_2 = 0\;}\tag{B7}$$

For a centred sheet these factor into $\sin\frac{k_vb}2 = 0$ (even $n$, node at the sheet, unaffected) and
Butterweck's eqs. (9) $2\xi + i\frac{k_v}k\tan\frac{k_vb}2 = 0$ and (8) $2\xi + i\frac k{k_v}\tan\frac{k_vb}2 = 0$
(S7c,d). Several sheets: propagate $(Y, Y')$ with $\cos/\sin$ between sheets and apply the jump at each (B8,
`t1lib.sheet_dispersion`); the boundary value $Y'(b)$ (LSM) or $Y(b)$ (LSE) must vanish. LSM$^v$ with $n = 0$
($Y$ = const) is TE$_{m0}$, an exact mode for any sheets: the sheet does not touch TE₁₀.

**Butterweck's example** ($b$ = 10 mm, λ = 4 mm, centred sheet, `butterweck_sheet.py`): EH₁₁⁽²⁾ (LSM$^v$, root
from $n = 1$) has its maximum attenuation **1.554 dB/cm at $R_\square$ = 55.75 Ω** with the exact γ (+3.6 % and −0.4 %
from his 1.5 dB/cm and 56 Ω); with his approximation α ≈ Im$k_v^2/2k$: 1.532 dB/cm at 55.30 Ω; at the optimum
$u = k_vb/2 = 1.2562 + 0.5515i$, $xy = 0.693$ (his 0.69), $2\xi v = 2.33$ (his 2.3). EH₁₁⁽¹⁾ (LSE$^v$): 5.25 dB/cm at
418 Ω (his ≈5 dB/cm, 420 Ω). Off-centre: the EH₁₁⁽²⁾-like mode reaches 1.76 dB/cm at $v_0'/b = 0.3$ (56 Ω) and
falls to 0.22 dB/cm at 0.1.

**Junction empty guide → sheet guide (B10).** Same formulas as B9, with the sheet modes of one $u$-index $m$
(LSM$^v$ $n\ge0$ and LSE$^v$ $n\ge1$, the same count as TE$_{m0}$, TE$_{mn}$, TM$_{mn}$) and fields
(LSM$^v$) $\mathcal E_u = \frac{k_m}{i\omega\varepsilon_0}\cos k_mu'\,Y'$, $\mathcal E_v = \frac{\gamma^2+k_m^2}{i\omega\varepsilon_0}\sin k_mu'\,Y$,
$\mathcal H_u = i\gamma\sin k_mu'\,Y$, $\mathcal H_v = 0$; (LSE$^v$) $\mathcal E_u = -i\gamma\cos k_mu'\,Y$, $\mathcal E_v = 0$,
$\mathcal H_u = -\frac{k_m}{i\omega\mu_0}\sin k_mu'\,Y'$, $\mathcal H_v = \frac{\gamma^2+k_m^2}{i\omega\mu_0}\cos k_mu'\,Y$; then normalize by
$\tfrac12\iint(\mathcal E_u\mathcal H_v - \mathcal E_v\mathcal H_u) = 1$. The overlaps factor into $\int_0^a$ (giving $a/2$, same $m$)
times $\int_0^b Y(v')\{\sin,\cos\}(k_nv')\,dv'$, elementary on each sub-interval between sheets. Checks: for
$R_\square\to\infty$, $|P| = |Q^T|$ of `ls_rotation` (4.6e-8); sheet modes orthogonal (2e-14); junction reciprocal
(4e-16); TE₁₀ passes with $|S_{11}|$ = 3e-15; a 10 cm filter (56 Ω, $b$ = 10.16 mm, $n\le12$, 25 modes per side) is passive; it attenuates TE₁₁ by 12.9 dB
and TM₁₁ by 16.5 dB and leaves TE₁₀, TE₁₂, TM₁₂ untouched.
**Thickness.** The model is an infinitely thin film. A dielectric carrier perturbs TE₁₀ (Butterweck: 0.1 mm mica,
ε_r = 6, gives 0.1 dB/cm for H₁₀ with a 70 Ω film; a 12 µm Mylar sheet gives no measurable loss): the normal $E$ of
TE₁₀ jumps in the dielectric, which creates tangential E at the film. Not modelled here.

**Note for WP2.** With $b$ = 10 mm and λ = 4 mm exactly, $k = 5\pi/b$: family $n = 5$ has $k_t = 0$, where
`ls_rotation` is singular. The sheet junction checks therefore use $b$ = 10.16 mm.

## 9. Final formulas in one place

| id | content |
| --- | --- |
| B1 | Leontovich $E_{\tan} = Z_sH_{\tan}\times\hat n$, $\hat n$ into the wall, $Z_s = (1+i)R_s$ |
| B2, B2a | $N_{pq}$, $G_{pq}$ per wall, closed forms, straight and κ parts |
| B3 | $\Lambda^\pm_{pq} = \frac12[\pm\sqrt{Y_p}\sqrt{Y_q}N_{pq} + G_{pq}/(\sqrt{Y_p}\sqrt{Y_q})]$ |
| B4, B4b | $dy/dz = (D - \Lambda_{0,\rm full}) y+ \kappa(K - \Lambda_{1,\rm full})y$, $\Lambda_{\rm full} = [[\Lambda^+,\Lambda^-],[-\Lambda^-,-\Lambda^+]]$ |
| B5 | first-order LS eigenvalues for one pair of walls |
| B6 | exact LS$^u$ dispersion with two impedance walls |
| B7, B8 | resistive sheet(s), LS$^v$ dispersion |
| B9, B10 | junctions by overlap integrals (impedance walls, sheets) |

Machine-checkable versions: `docs/theory/T1/t1lib.py` — `wall_integrals` (B2a, one pair, one wall),
`wall_matrices`, `NG_matrices` (B2), `loss_matrices` (B3; `order=1` gives $\Lambda_1$), `loss_matrix_full` (B4),
`attenuation_marcuvitz`, `ls_first_order` (B5), `lsu_dispersion`, `lsu_roots_all` (B6), `sheet_dispersion_closed`
(B7), `sheet_dispersion` (B8); the junctions B9/B10 are `junction` in `butterweck_lossy_wall.py` and
`junction_sheet` in `butterweck_sheet.py`. Every function takes SI units and the conventions above.

## 10. Checks (all run; outputs quoted above)

| script | what | result |
| --- | --- | --- |
| `check_sympy.py` | S1 Leontovich sign; S2 exact projected equations with boundary terms; S3 power waves; S4 wall integrals; S5 diagonal = Marcuvitz; S6 LS$^u$ dispersion, Butterweck (29), (21); S7 sheet dispersion, Butterweck (8), (9); S8 factor ν/μ in Butterweck (28) | all pass |
| `check_coefficients.py` | Q1 closed forms vs quadrature (≤ 1.5e-14); Q2 Λ (≤ 1.2e-13); Q3 diagonal vs Marcuvitz/card (≤ 7.4e-14); Q4 symmetry; Q5 LS eigenvalues/eigenvectors (≤ 1e-15); Q6 exact vs first order is O(Z²); Q7 power balance (1.5e-14) | all pass |
| `wr90_aluminium.py` | section 5 and 6 numbers | — |
| `butterweck_lossy_wall.py` | section 7 | all pass |
| `butterweck_sheet.py` | section 8 | all pass |

Run each with `PYTHONPATH=<worktree>/src python3 docs/theory/T1/<script>.py` (they import `t1lib` from their own
folder). `check_sympy.py` takes about 10 min (S2), `butterweck_lossy_wall.py` about 1 min.
