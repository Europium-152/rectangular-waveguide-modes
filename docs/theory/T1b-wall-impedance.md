# T1b — Ohmic loss and impedance walls in the coupled-mode equations (independent derivation)

*Theory note, wave 1, card T1 (second, independent derivation T1b, for comparison with T1). Method: Lorentz
reciprocity with perfect-conductor test fields, then a cross-check against the T0 projection. Every boxed formula
has a Python function in `docs/theory/T1b/t1b_coefficients.py`; the scripts in `docs/theory/T1b/` check them.*

Run all checks from the repository root:

```
PYTHONPATH=src python3 docs/theory/T1b/check_coefficients.py      # items 2-4: quadrature, sympy, textbook, GSM
PYTHONPATH=src python3 docs/theory/T1b/check_exact_modes.py       # first order vs exact LS modes, O(Z^2)
PYTHONPATH=src python3 docs/theory/T1b/wr90_aluminium.py          # item 3 numbers, item 4 SPARC bends
PYTHONPATH=src python3 docs/theory/T1b/butterweck_lossy_wall.py   # item 5
PYTHONPATH=src python3 docs/theory/T1b/butterweck_sheet.py        # item 6
```

All conventions are those of `CONVENTIONS.md`: $e^{+i\omega t}$, forward modes $e^{-i\beta z}$ with
$\operatorname{Im}\beta\le0$, $e_{TM}=\nabla_t\Pi$, $e_{TE}=\nabla_t\Gamma\times\hat z$,
$E_t=\sum(a^++a^-)\sqrt{2/Y}\,e$, $H_t=\sum(a^+-a^-)\sqrt{2Y}\,\hat z\times e$, $\omega\mu_0=k\eta_0$,
$\eta_0=\mu_0c$. Walls are labelled `u-` ($u=-a/2$), `u+` ($u=+a/2$), `v-` ($v=-b/2$), `v+` ($v=+b/2$), each with its own
impedance $Z_w$; $u'=u+a/2$, $v'=v+b/2$, $k_m=m\pi/a$, $k_n=n\pi/b$, $\chi^2=k_m^2+k_n^2$.

## 1. Leontovich condition

With $\hat n$ the unit normal pointing **from the guide into the wall**,

```latex
\boxed{E_{\tan} = Z_w\, H_{\tan}\times\hat n \quad\Longleftrightarrow\quad \hat n\times E = Z_w\,H_{\tan},\qquad
Z_s=(1+i)R_s,\; R_s=\sqrt{\omega\mu_0/(2\sigma)} .}
```

Sign check for $e^{+i\omega t}$: in a conductor filling $x>0$ ($\hat n=\hat x$) the transmitted wave is
$E=\hat y E_0e^{-\gamma x}$, $\gamma=(1+i)/\delta$. Faraday's law $\nabla\times E=-i\omega\mu_0H$ gives
$H=\hat z\,\gamma E_0/(i\omega\mu_0)$, so $E_y/H_z=i\omega\mu_0/\gamma=\omega\mu_0\delta(1+i)/2=(1+i)R_s$. And
$H\times\hat n=H_z\hat z\times\hat x=H_z\hat y$, i.e. $E_y=Z_sH_z$: the boxed form with $Z_s=(1+i)R_s$. Butterweck's
eq. (20) is the same statement ("$n$ the outward directed normal", $e^{j\omega t}$).

On the wall $u=+a/2$ this reads $E_v=Z\,H_z$, $E_z=-Z\,H_v$; on $u=-a/2$: $E_v=-Z H_z$, $E_z=Z H_v$.

## 2. Straight guide: derivation by reciprocity

**Reciprocity identity.** Let $(E,H)$ be the field in the guide with impedance walls and $(E',H')$ any field of
the same guide with perfectly conducting walls; both satisfy the source-free Maxwell equations in the same medium,
so $\nabla\cdot(E\times H'-E'\times H)=0$. Integrated over a slice $dz$ of the cross-section $S$:

```latex
\frac{d}{dz}\iint_S (E\times H'-E'\times H)\cdot\hat z\,ds = -\oint_{\partial S}(E\times H'-E'\times H)\cdot\hat n\,dl
 = -\oint Z_w\, H_{\tan}\cdot H'_{\tan}\,dl ,
```

because $\hat n\times E'=0$ on the walls and $(E\times H')\cdot\hat n=(\hat n\times E)\cdot H'=Z_wH_{\tan}\cdot H'$.
This is exact; no expansion has been used yet.

**Test fields.** Take $(E',H')$ = the forward ($e^{-i\beta_pz}$) or the backward ($e^{+i\beta_pz}$) PEC mode $p$
with the amplitude normalization of the conventions: transverse $E'_t=\sqrt{2/Y_p}\,e_p$ for both,
$H'_t=+\sqrt{2Y_p}\,\hat z\times e_p$ (forward) or $-\sqrt{2Y_p}\,\hat z\times e_p$ (backward), and the
longitudinal field from $-i\omega\mu_0H_z=\nabla_t\cdot(E_t\times\hat z)$, the same for both. On a wall, the
tangential magnetic field of the forward (+) and backward (−) mode is

```latex
h_p^{\pm} = \pm\sqrt{2Y_p}\;(e_p\cdot\hat n)\,\hat t + H_{zp}\,\hat z,\qquad
H_{zp}=-\,\frac{i\sqrt{2/Y_p}}{k\eta_0}\,g_p,\qquad g_p=\nabla_t\cdot(\hat z\times e_p)=\begin{cases}-\chi_p^2\Gamma_p&\text{TE}\\0&\text{TM}\end{cases}
```

on the wall, with $\hat t=\hat z\times\hat n$, using $(\hat z\times e_p)\cdot\hat t=e_p\cdot\hat n$ (the tangential
transverse $H$ is carried by the *normal* $E$ of the mode). The cross-section integrals follow from the
orthonormality of the $e_p$ and the amplitude expansion of $(E,H)$, using $(e_q\times(\hat z\times e_p))\cdot\hat z=e_q\cdot e_p$:
with the backward test mode, $\iint(E\times H'-E'\times H)\cdot\hat z=-4a_p^+e^{i\beta_pz}$; with the forward test
mode, $+4a_p^-e^{-i\beta_pz}$. Hence, exactly,

```latex
\frac{da_p^+}{dz}=-i\beta_pa_p^+ +\tfrac14\oint Z_w\,H_{\tan}\cdot h_p^-\,dl,\qquad
\frac{da_p^-}{dz}=+i\beta_pa_p^- -\tfrac14\oint Z_w\,H_{\tan}\cdot h_p^+\,dl .
```

**First order in $Z_w$.** The wall integral is already $O(Z_w)$, so $H_{\tan}$ may be replaced by its value in the
PEC modal expansion, $H=\sum_q(a_q^+h_q^++a_q^-h_q^-)$; the error is $O(Z_w^2)$. With
$h_q^s\cdot h_p^{s'}=2ss'\sqrt{Y_p}\sqrt{Y_q}(e_p\cdot\hat n)(e_q\cdot\hat n)-2g_pg_q/\big((k\eta_0)^2\sqrt{Y_p}\sqrt{Y_q}\big)$
this gives the card's form
$da^\pm_p/dz=\mp i\beta_pa^\pm_p\mp\sum_q(\Lambda^{\pm\pm}_{pq}a^\pm_q+\Lambda^{\pm\mp}_{pq}a^\mp_q)$ with

```latex
\boxed{\;
\Lambda^{++}_{pq}=\Lambda^{--}_{pq}=\tfrac12\Big[\sqrt{Y_p}\sqrt{Y_q}\,A_{pq}+\frac{B_{pq}}{(k\eta_0)^2\sqrt{Y_p}\sqrt{Y_q}}\Big],\qquad
\Lambda^{+-}_{pq}=\Lambda^{-+}_{pq}=\tfrac12\Big[-\sqrt{Y_p}\sqrt{Y_q}\,A_{pq}+\frac{B_{pq}}{(k\eta_0)^2\sqrt{Y_p}\sqrt{Y_q}}\Big]\;}
```

```latex
\boxed{\;A_{pq}=\sum_{w}Z_w\!\int_w (e_p\cdot\hat n)(e_q\cdot\hat n)\,dl,\qquad
B_{pq}=\sum_{w}Z_w\!\int_w g_p\,g_q\,dl\;}
```

($\sqrt{Y_p}\sqrt{Y_q}$ with principal roots, as in T0; this form holds for evanescent modes too, since the
reciprocity argument never conjugates). $A$ comes from the wall currents along $z$ (the normal $E$ of the mode), $B$
from the transverse wall currents (the $H_z$ of TE modes). The final system, **as WP2 should implement it**, is

```latex
\boxed{\;\frac{dy}{dz}=\big(D+\kappa K-\Lambda_{\text{full}}\big)\,y,\qquad y=\begin{bmatrix}A^+\\A^-\end{bmatrix},\qquad
\Lambda_{\text{full}}=\begin{bmatrix}\Lambda^{++}&\Lambda^{+-}\\-\Lambda^{-+}&-\Lambda^{--}\end{bmatrix}
=\begin{bmatrix}\Lambda^{++}&\Lambda^{+-}\\-\Lambda^{+-}&-\Lambda^{++}\end{bmatrix}\;}
```

with $D=\operatorname{diag}(-i\beta,+i\beta)$ and $K$ of `coupled_mode_matrices`. Since
$K=[[-C^+,-C^-],[C^-,C^+]]$ (with the $i$ inside the code's $C$), the loss enters exactly like the curvature:
$\kappa C^+\to\kappa C^++\Lambda^{++}$, $\kappa C^-\to\kappa C^-+\Lambda^{+-}$. $\Lambda^{++}$ and $\Lambda^{+-}$ are
symmetric, so every lossy section has a symmetric GSM (checked: $\max|S-S^T|/\max|S|=6\times10^{-13}$ for 1 m of
WR-90 aluminium at 90 GHz with 126 modes; largest singular value 0.98687).

**Same result in the T0 voltage/current form.** With $V_p=(a^+_p+a^-_p)\sqrt{2/Y_p}$, $I_p=(a^+_p-a^-_p)\sqrt{2Y_p}$:

```latex
\boxed{\;\frac{dV_p}{dz}=-i\frac{\beta_p}{Y_p}I_p-\sum_qA_{pq}I_q,\qquad
\frac{dI_p}{dz}=-i\beta_pY_pV_p-\frac{1}{(k\eta_0)^2}\sum_qB_{pq}V_q\;}
```

These are exactly the boundary terms that the T0 projection drops for perfect walls: in
$\iint e_p\cdot\nabla_t E_z=\oint E_z\,e_p\cdot\hat n-\iint d_pE_z$ the wall term is no longer zero, since
$E_z=-Z_wH_t\cdot\hat t$, giving $-\sum_qA_{pq}I_q$; and in $-\iint g_pH_z$, with
$-i\omega\mu_0H_z=\nabla_t\cdot(E_t\times\hat z)$, the wall term $\oint g_p(E_t\times\hat z)\cdot\hat n=\oint g_pZ_wH_z$
(the hidden boundary term of T0) gives $-\sum_qB_{pq}V_q/(k\eta_0)^2$. So the projection and the reciprocity
derivations agree term by term. (In the projection, the $dV/dz$ boundary term uses the exact $E_z$ on the wall; the
$dI/dz$ term uses the term-by-term $H_z$, which is wrong by $O(Z_w)$, i.e. $O(Z_w^2)$ in the equation.)

### 2.1 Closed forms, per wall

Field values on the walls (from $e_{TM}=\nabla\Pi$, $e_{TE}=\nabla\Gamma\times\hat z$; $N_p$ the normalization
constant of $\Pi$ or $\Gamma$ in CONVENTIONS.md §4):

| | $e\cdot\hat n$ on $u'=0$ / $u'=a$ (up to the sign of $\hat n$) | $e\cdot\hat n$ on $v'=0$ / $v'=b$ | $g$ |
| --- | --- | --- | --- |
| TM$_{mn}$ | $c^u\sin k_nv'$, $c^u=N k_m$; factor $(-1)^m$ on $u'=a$ | $c^v\sin k_mu'$, $c^v=N k_n$; factor $(-1)^n$ on $v'=b$ | 0 |
| TE$_{mn}$ | $c^u\sin k_nv'$, $c^u=-N k_n$ | $c^v\sin k_mu'$, $c^v=N k_m$ | $-\chi^2N\cos k_mu'\cos k_nv'$ |

The signs of $\hat n$ cancel in the products. With the wall combinations

```latex
Z^u_{mm'}=Z_{u-}+(-1)^{m+m'}Z_{u+},\qquad Z^v_{nn'}=Z_{v-}+(-1)^{n+n'}Z_{v+},
```

```latex
\boxed{\;A_{pq}=\delta_{n_pn_q}\,\frac b2\,c^u_pc^u_q\,Z^u_{m_pm_q}+\delta_{m_pm_q}\,\frac a2\,c^v_pc^v_q\,Z^v_{n_pn_q},\qquad
B_{pq}=\chi_p^2\chi_q^2N_pN_q\Big[\delta_{n_pn_q}\frac{b}{\epsilon_{n}}Z^u_{m_pm_q}+\delta_{m_pm_q}\frac{a}{\epsilon_{m}}Z^v_{n_pn_q}\Big]\;(\text{TE–TE only})}
```

Written out for the three kinds of pair ($p=(m,n)$, $q=(m',n')$):

```latex
\begin{aligned}
\text{TM–TM:}\quad &A=N_pN_q\Big[\delta_{nn'}\tfrac b2k_mk_{m'}Z^u_{mm'}+\delta_{mm'}\tfrac a2k_nk_{n'}Z^v_{nn'}\Big],\quad B=0\\
\text{TE–TE:}\quad &A=N_pN_q\Big[\delta_{nn'}\tfrac b2k_n^2Z^u_{mm'}+\delta_{mm'}\tfrac a2k_m^2Z^v_{nn'}\Big],\quad
B=\chi_p^2\chi_q^2N_pN_q\Big[\delta_{nn'}\tfrac b{\epsilon_n}Z^u_{mm'}+\delta_{mm'}\tfrac a{\epsilon_m}Z^v_{nn'}\Big]\\
\text{TM}_p\text{–TE}_q:\quad &A=N_pN_q\Big[-\delta_{nn'}\tfrac b2k_mk_nZ^u_{mm'}+\delta_{mm'}\tfrac a2k_nk_mZ^v_{nn'}\Big],\quad B=0
\end{aligned}
```

(TE terms with $k_n=0$ or $k_m=0$ vanish automatically, which takes care of $n=0$ and $m=0$.) Code:
`wall_A`, `wall_B`, `lambda_pp`, `lambda_pm`, `lambda_full`, `dV_dI_boundary`.

**Selection rules.** The $u$-walls couple modes of the same $n$ (any $m,m'$; only $m+m'$ even if $Z_{u-}=Z_{u+}$);
the $v$-walls couple modes of the same $m$ and **different families** $n\ne n'$ (only $n+n'$ even if
$Z_{v-}=Z_{v+}$). So broad-wall loss breaks the family rule of CONVENTIONS.md §3 (e.g. TE$_{10}$–TE$_{12}$/TM$_{12}$);
the effect is negligible for metal walls (§3 below), but a lossy component is not, strictly, family-conserving.

**Degenerate pair with all four walls equal.** For TE$_{mn}$/TM$_{mn}$ ($m,n\ge1$),
$A_{\text{TM,TE}}=N_{\text{TE}}N_{\text{TM}}k_mk_nZ\,(a-b)$: the TE–TM loss coupling vanishes in a square guide and
has opposite signs from the $u$- and the $v$-walls.

## 3. Checks of item 2, and WR-90 aluminium (18–90 GHz)

**Textbook attenuation.** For equal walls ($Z=R_s$), $\operatorname{Re}\Lambda^{++}_{pp}$ equals Marcuvitz's
attenuation constants identically (sympy, `check_coefficients.py`): TE$_{mn}$ (§2.2 eq. (10), p. 60), TE$_{m0}$,
TE$_{0n}$, TM$_{mn}$ (§2.2 eq. (4), p. 57), and the card's TE$_{10}$ form
$\alpha=R_s/(\eta_0b\sqrt{1-(f_c/f)^2})\,[1+(2b/a)(f_c/f)^2]$. Numerically, WR-90 and a 20 mm square guide, every
propagating mode with $m\le10$, $n\le6$, 18–90 GHz: max relative deviation $7.4\times10^{-14}$. With
$Z_s=(1+i)R_s$, $\operatorname{Im}\Lambda_{pp}=\operatorname{Re}\Lambda_{pp}$ (the familiar equal phase shift).
WR-90 TE$_{10}$, aluminium: 0.1227 dB/m at 18 GHz, 0.1410 (30), 0.1665 (45), 0.1898 (60), 0.2109 (75), 0.2303 dB/m at 90 GHz.

**Direct surface integrals.** All four blocks $\Lambda^{\pm\pm}_{pq}$ computed from
$-\tfrac14\oint Z_w\,h^s_q\cdot h^{s'}_p\,dl$ with the fields of `wrmodes.modes` (Gauss–Legendre, each wall a
different complex $Z$, all 31×31 pairs with $m\le4$, $n\le3$, three guides): max relative deviation from the closed
forms $7.5\times10^{-15}$; the power loss $\tfrac12R_s\oint|H_{\tan}|^2$ equals $2\operatorname{Re}\Lambda_{pp}$ to
$5\times10^{-15}$.

**Exact modes (independent of the derivation).** With impedance on one pair of opposite walls the guide separates
and its exact modes are LS modes with respect to the wall normal (§5). The eigenvalues of the first-order system
match the exact propagation constants with an error falling as $Z^2$ (two decades per decade of $Z$; the ratio
error/shift is $1\times10^{-5}$ to $1.3\times10^{-4}$ at $|Z|\approx0.01\ \Omega$ and $1\times10^{-3}$ to
$1.3\times10^{-2}$ at $|Z|\approx1\ \Omega$, WR-90 at 60 GHz, 8 modes), for lossy $u$-walls (families $n=0,1$) and
lossy $v$-walls (families $m=1,2$), with *different* complex impedances on the two walls ($0.7+0.9i$ and $1.3+0.6i$,
times the scale). For the degenerate pairs, the
eigenvectors of the 2×2 block of $\Lambda^{++}$ are exactly (to $10^{-16}$) the LSE$^u$/LSM$^u$ combinations
(lossy $u$-walls) or the LSE$^v$/LSM$^v$ combinations of CONVENTIONS.md §7 (lossy $v$-walls). This fixes the sign of
the TE–TM terms: the opposite sign gives the wrong combinations.

**Which off-diagonal terms matter (WR-90, aluminium, all four walls, `wr90_aluminium.py`).**

* *Degenerate TE$_{mn}$/TM$_{mn}$ pairs* couple resonantly. $|\Lambda_{\text{TE,TM}}|$ is comparable to the
  diagonal terms: 0.04–0.18 dB/m (largest 0.180 dB/m, TE/TM$_{9,4}$ at 90 GHz) against 0.2–0.6 dB/m. The loss
  eigenmodes are mixtures rotated by 1°–45° from pure TE/TM (e.g. TE/TM$_{11}$: 13.5° at 18 GHz, 25° at 90 GHz;
  TE/TM$_{21}$ at 60 GHz: 44°), with attenuations split well beyond the textbook values (TE/TM$_{11}$ at 90 GHz:
  0.263 and 0.422 dB/m → eigenvalues 0.219 and 0.466 dB/m). A TE$_{mn}$ input converts into TM$_{mn}$: up to
  $3.9\times10^{-4}$ of the input power after 1 m and $3.4\times10^{-2}$ after 20 m (pairs with $\beta>0.2k$). These
  terms must be kept (WP2 spec item 2 already does); with only $u$-walls or only $v$-walls lossy the eigenmodes would
  be exactly LS$^u$ or LS$^v$; with all four walls lossy they lie in between ("LS-like").
* *Non-matched pairs* average out. Over all pairs of propagating modes of all families ($\beta>0.05k$), the
  forward conversion bound $4|\Lambda^{++}_{pq}|^2/(\beta_p-\beta_q)^2$ is at most $1.4\times10^{-7}$ (30 GHz) …
  $5.8\times10^{-6}$ (90 GHz, TE$_{0,6}$–TE$_{2,6}$), the backward bound
  $4|\Lambda^{+-}_{pq}|^2/(\beta_p+\beta_q)^2\le3.6\times10^{-7}$; the forward-to-backward self term
  $|\Lambda^{+-}_{pp}|/2\beta_p\le5\times10^{-3}$ (a change of the wave impedance, i.e. a reflection of order
  $10^{-5}$ in power at a junction to a lossless guide). The inter-family ($v$-wall) terms are among these
  and negligible.
* *Near cutoff* the first-order theory needs $|\Lambda_{pp}|\ll|\beta_p|$. $|\Lambda_{pp}|=0.01\beta_p$ is reached at
  $f/f_c-1\approx0.7$–$0.9\%$ (TE$_{10}$, TE$_{20}$, TM$_{11}$, aluminium); closer to cutoff the mode needs the exact
  treatment (or can be dropped, its contribution being negligible).

## 4. Bends (item 4)

The T0 projection holds for any scale factor $h=1+\kappa u$; with impedance walls the same steps as in §2 (now
$\partial_zE_t=\nabla_t(hE_z)+i\omega\mu_0h\,\hat z\times H_t$, etc.) give the boundary terms with the factor $h$
under the wall integrals, and no other new term (the extra terms $\oint h\,d_p(\dots)$ and
$\oint h\,g_p\,e_q\cdot\hat t$ vanish on the walls as before). The result is first order in $Z_w$ but **exact in
$\kappa$**:

```latex
\boxed{\;\frac{dy}{dz}=\Big(D+\kappa\big(K-\Lambda^{(1)}_{\text{full}}\big)-\Lambda_{\text{full}}\Big)y,\qquad
\Lambda^{(1)}:\;\text{same formulas as }\Lambda\text{ with }A\to A^{(1)},\,B\to B^{(1)}\;}
```

```latex
\boxed{\;\begin{aligned}
A^{(1)}_{pq}&=\sum_wZ_w\!\int_wu\,(e_p\cdot\hat n)(e_q\cdot\hat n)\,dl
=\delta_{n_pn_q}\tfrac b2c^u_pc^u_q\tfrac a2\big[(-1)^{m_p+m_q}Z_{u+}-Z_{u-}\big]+c^v_pc^v_q\,Z^v_{n_pn_q}\,S_{m_pm_q},\\
B^{(1)}_{pq}&=\chi_p^2\chi_q^2N_pN_q\Big[\delta_{n_pn_q}\tfrac b{\epsilon_n}\tfrac a2\big[(-1)^{m_p+m_q}Z_{u+}-Z_{u-}\big]+Z^v_{n_pn_q}C_{m_pm_q}\Big]\;(\text{TE–TE}),\\
S_{mm'}&=\int_0^a(u'-\tfrac a2)\sin k_mu'\sin k_{m'}u'\,du'=-\Big(\frac a\pi\Big)^2\frac{4mm'}{(m^2-m'^2)^2},\quad
C_{mm'}=-\Big(\frac a\pi\Big)^2\frac{2(m^2+m'^2)}{(m^2-m'^2)^2}\quad(m+m'\text{ odd; else }0).
\end{aligned}}
```

Code: `wall_A1`, `wall_B1`, `lambda1_pp`, `lambda1_pm`, `lambda1_full`; checked against quadrature with the
weight $1+\kappa u$ to $7.5\times10^{-15}$. The $\kappa Z_w$ "cross terms" are thus not neglected here; they are the
first moments $\kappa\Lambda^{(1)}$. For equal opposite walls they have no diagonal part (the longer outer wall and
the shorter inner wall compensate to first order in $\kappa$) and couple $m+m'$ odd, i.e. the same pairs as $K$.
What remains neglected is $O(Z_w^2)$ and the change of the surface impedance of a curved wall, $O(\delta/R)$
($\delta\approx0.3\ \mu$m in aluminium at 90 GHz, $R\ge0.1$ m: $<10^{-5}$).

**Size for the SPARC bends** (sech profile, $T=4$, aluminium; H plane $a=22.86$ mm family $n=0$, E plane
$a=10.16$ mm local family $n=1$; 20, 45, 70, 90 GHz; peak $\kappa a/2=0.023$–0.117): dropping $\kappa\Lambda^{(1)}$
changes the main transmission by up to $3.1\times10^{-3}$ dB (E plane, 90° over 40 cm, 90 GHz; total ohmic+conversion
loss there 0.42 dB) and the power of a converted mode by up to $6.4\times10^{-4}$ of the input. The second exceeds
the $10^{-4}$ target of CONVENTIONS.md §8, so **WP2 should keep $\kappa\Lambda^{(1)}$** (it costs nothing: it is
added to $K$). Loss itself adds 0.05–0.17 dB to these bends (e.g. H plane 60°/60 cm: 0.075 dB at 20 GHz, 0.141 dB at
90 GHz on top of the conversion loss).

## 5. Large wall impedance: Butterweck's lossy-wall filter (item 5)

### 5.1 Exact modes with impedance walls at $u=\mp a/2$

With $Z_{u-}$, $Z_{u+}$ on the walls normal to $u$ and PEC walls at $v=\pm b/2$, the exact modes are LS modes with
respect to $u$, built from potentials along $\hat u$:

* LSE$^u$ ($E_u=0$): $E=-\nabla\times(\hat u\psi)$, $\psi=X(u')\cos k_nv'$, $n\ge0$;
* LSM$^u$ ($H_u=0$): $H=\nabla\times(\hat u\psi)$, $\psi=Y(u')\sin k_nv'$, $n\ge1$.

The Leontovich condition on both $u$-walls reduces, for each type separately (the two tangential components give
the same condition), to a Robin condition $\partial_n\psi=-iq\psi$ ($\hat n$ outward), with $q=k\eta_0/Z$ (LSE$^u$) or
$q=kZ/\eta_0$ (LSM$^u$). With $\gamma^2=k^2-k_x^2-k_n^2$:

```latex
\boxed{\;\begin{aligned}
\text{LSE}^u:&\;(1+p_-p_+k_x^2)\sin k_xa-i(p_-+p_+)\,k_x\cos k_xa=0,\quad p_\pm=\frac{Z_{u\pm}}{k\eta_0},\quad X=p_-k_x\cos k_xu'+i\sin k_xu',\\
\text{LSM}^u:&\;(k_x^2+q_-q_+)\sin k_xa-i(q_-+q_+)\,k_x\cos k_xa=0,\quad q_\pm=\frac{kZ_{u\pm}}{\eta_0},\quad Y=k_x\cos k_xu'+iq_-\sin k_xu'.
\end{aligned}}
```

$Z\to0$ gives $k_x=m\pi/a$ (LSE$^u$: $m\ge1$, Dirichlet; LSM$^u$: $m\ge0$, Neumann); LSE$^u_{m0}$ = TE$_{m0}$,
LSM$^u_{0n}$ = TE$_{0n}$. Roots are continued from these values (`lsu_kx_roots`). Transverse fields, normalized
with $\tfrac12\iint(e\times h)\cdot\hat z=1$ (`impedance_guide_modes`):
LSE$^u$: $e=c\,(0,\;i\gamma X\cos k_nv')$, $h=\frac{c}{i\omega\mu_0}\big((k^2-k_x^2)X\cos k_nv',\;-k_nX'\sin k_nv'\big)$;
LSM$^u$: $e=\frac{c}{i\omega\varepsilon_0}\big((k^2-k_x^2)Y\sin k_nv',\;k_nY'\cos k_nv'\big)$, $h=c\,(0,\;-i\gamma Y\sin k_nv')$.
LSE and LSM modes are orthogonal in the unconjugated reaction (one has $E_u=0$, the other $H_u=0$), and modes of one
type are orthogonal because the Robin problem is complex-symmetric.

**Low-impedance limit (Butterweck eq. 21).** For TE$_{m0}$ and $Z_{u\pm}=Z_w$, $k_x\tan(k_xa/2)=ik\eta_0/Z_w$ (even
modes) gives $k_x\approx k_m(1+2iZ_w/(k\eta_0a))$, $\gamma\approx\beta_m-2ik_m^2Z_w/(k\eta_0a\beta_m)$, i.e.
$\alpha_m=2\pi^2m^2\operatorname{Re}Z_w/(\beta_ma^3k\eta_0)$, Butterweck's eq. (21). The first-order $\Lambda_{pp}$ of
§2 gives the same expression identically (only $B^u$ contributes): $3.869$, $15.660$, $35.950$ dB/m for H$_{10}$,
H$_{20}$, H$_{30}$ at 75 GHz, 250 Ω (difference from eq. 21: $<10^{-15}$), i.e. 0.580, 2.349, 5.393 dB over 15 cm.
Against Butterweck's quoted theory values 4 and 36 dB/m: −3.3 % and −0.1 % (he rounds).

### 5.2 Mode matching to the ideal guide

At a junction PEC guide | impedance guide (same cross-section), test the continuity of $E_t$ with the PEC $h_p$ and
the continuity of $H_t$ with the impedance-guide $e_i$:

```latex
\boxed{\;a^++a^-=R\,(c^++c^-),\quad R^T(a^+-a^-)=c^+-c^-,\qquad
R_{pi}=\tfrac12\iint(e_i\times h_p)\cdot\hat z\,ds=\tfrac12\sqrt{2Y_p}\iint e_i\cdot e_p\,ds}
```

```latex
\boxed{\;S_{11}=(I+RR^T)^{-1}(RR^T-I),\quad S_{12}=2(I+RR^T)^{-1}R,\quad S_{21}=S_{12}^T,\quad S_{22}=I-R^TS_{12}\;}
```

(reciprocal by construction; `junction_gsm`, `lossy_wall_junction`). The overlaps reduce to one-dimensional
integrals of trigonometric functions with complex $k_x$ (elementary; evaluated with 200-point Gauss–Legendre, exact
to rounding because the integrands are entire). The filter is junction ⋆ propagation $e^{-i\gamma_iL}$ ⋆ reversed
junction.

**Butterweck's example** (22.86 × 10.16 mm, 75 GHz, $Z_w$ = 250 Ω, $L$ = 15 cm, family $n=0$):

| | H$_{10}$ | H$_{20}$ | H$_{30}$ | H$_{10}$→H$_{30}$ |
| --- | --- | --- | --- | --- |
| first order (= Butterweck eq. 21), 15 cm | 0.580 dB | 2.349 dB | 5.393 dB | |
| exact mode alone, 15 cm | 0.579 dB | 2.353 dB | 5.431 dB | |
| exact filter with junctions (M = 40, converged to 0.001 dB) | 0.569 dB | 2.317 dB | 5.341 dB | 0.0211 (−33.5 dB) |
| Butterweck: theory / measured (Philips Tech. Rev. 29, p. 95) | 0.6 / 0.6 dB | — / 2.5 dB | 5.4 / 5.4 dB | 0.016–0.06 (−36 to −24.5 dB) |
| exact vs measured | −5.1 % | −7.3 % | −1.1 % | inside the range |

The junction reflection is $|S_{11}|\approx2\times10^{-4}$; the conversion over 10–20 cm varies between 0.0099 and
0.038 (beat between H$_{10}$ and H$_{30}$).

### 5.3 Where the first-order treatment fails

Three distinct mechanisms, quantified in `butterweck_lossy_wall.py` (75 GHz, WR-90, $Z$ real, both side walls):

1. **LSM$^u$ modes (families $n\ge1$)**: the perturbation parameter is $ka|Z|/\eta_0$, not $|Z|/\eta_0$. The first-order
   attenuation of LSM$^u_{0,1}$ (= TE$_{01}$) is off by $2\times10^{-4}$ at 1 Ω, 2 % at 10 Ω, 72 % at 50 Ω and by a
   factor 29 at 250 Ω, i.e. relative error 27.9 ($ka|Z|/\eta_0$ = 0.095, 0.95, 4.8, 24). LSE$^u$ modes (parameter
   $2|Z|/(k\eta_0a)$ = 0.037 at 250 Ω) are fine: $1.6\times10^{-3}$ (H$_{10}$), $7\times10^{-3}$ (H$_{30}$) at 250 Ω,
   3–6 % at 1000 Ω. So Butterweck's filter is within first-order theory for the H$_{m0}$ modes but not for the modes
   with $n\ge1$ ("very high attenuation figures", his p. 278): first order overestimates the attenuation of
   LSM$^u$ modes by more than an order of magnitude at 250 Ω, because a large wall impedance pushes their field away
   from the wall. The exact $n=1$ filter (15 cm, 250 Ω, mode matching with 25 modes, reciprocal to $10^{-16}$) gives
   TE$_{01}$→TE$_{01}$ −2.8 dB and TE$_{11}$→TE$_{11}$ −12.2 dB. The junction code for $n\ge1$ reduces to a lossless
   identity for $Z\to0$ ($|RR^T-I|=3\times10^{-13}$).
2. **Near cutoff**: first order needs $|\Lambda_{pp}|\ll|\beta_p|$; at 250 Ω, H$_{11,0}$ has $|\Lambda|/\beta=0.46$.
3. **Truncated coupled system in the PEC basis**: the system of §2 is right to $O(Z)$, but its $O(Z^2)$ terms grow
   with the number of modes $N$ (the PEC modes cannot represent the non-zero tangential $E$ at the wall; the
   second-order sums do not converge). At 250 Ω the eigen-attenuation of the H$_{10}$-like mode is off by −0.4 %
   ($N=3$), −13 % ($N=11$, all propagating modes), −67 % ($N=40$); the 15 cm transmissions with the 11 propagating
   modes are 0.501/2.087/4.731 dB instead of 0.569/2.317/5.341 dB. For aluminium walls the same artifact is
   $\le9\times10^{-6}$ (TE$_{10}$) and $\le1.6\times10^{-4}$ (TE$_{01}$, $N=40$) of $\Lambda_{pp}$ at 90 GHz:
   negligible. Large impedances therefore need the exact modes and mode matching (WP2 spec item 4), and the
   coupled system should be used only with $ka|Z|/\eta_0\ll1$ and $|Z|N/\eta_0\ll1$.

Butterweck also warns (footnote 5) that a local wall impedance is itself questionable for large $Z_w$; that is a
limit of the Leontovich model, not of this note.

## 6. Resistive sheet parallel to the broad walls (item 6)

A sheet at $v=v_0$ ($v'=s=v_0+b/2$) with resistance $R_\square$: $E_{\tan}$ continuous,
$\hat v\times(H^+-H^-)=E_{\tan}/R_\square$ (Butterweck eq. 2). The loaded guide is layered in $v$ and uniform in $u$,
$z$, so its exact modes are LS modes with respect to $v$ (the LS modes of CONVENTIONS.md §7), with the same
$u$-dependence as the empty-guide modes of index $m$ (the sheets conserve $m$, not $n$):

* LSM$^v$ (Butterweck EH$^{(2)}$, $H_v=0$): $\psi_e=\sin(k_mu')P(v')$, $P'(0)=P'(b)=0$; at the sheet $P'$ continuous,
  $P(s^+)-P(s^-)=-i\frac{\eta_0}{kR_\square}P'(s)$;
* LSE$^v$ (EH$^{(1)}$, $E_v=0$): $\psi_h=\cos(k_mu')P(v')$, $P(0)=P(b)=0$; $P$ continuous,
  $P'(s^+)-P'(s^-)=+i\frac{k\eta_0}{R_\square}P(s)$;

with $P''=-k_y^2P$ between sheets and $\gamma^2=k^2-k_m^2-k_y^2$. For one sheet:

```latex
\boxed{\;\text{LSM}^v:\;\sin k_yb+i\frac{\eta_0k_y}{kR_\square}\sin k_ys\,\sin k_y(b-s)=0,\qquad
\text{LSE}^v:\;k_y\sin k_yb+i\frac{k\eta_0}{R_\square}\sin k_ys\,\sin k_y(b-s)=0\;}
```

For $s=b/2$ these factor into the modes that do not see the sheet ($\sin(k_yb/2)=0$) and Butterweck's eqs. (9) and
(8): $2\xi+i(k_y/k)\tan(k_yb/2)=0$ and $2\xi+i(k/k_y)\tan(k_yb/2)=0$, $\xi=R_\square/\eta_0$. LSM$^v$ always has the
root $k_y=0$: TE$_{m0}$, untouched by the sheet. Several sheets: the code propagates $(P,P')$ through the layers
with the jump matrices above (`sheet_dispersion`, any number of sheets; residual of the closed forms at the roots
$5\times10^{-15}$).

Fields (normalized $\tfrac12\iint(e\times h)\cdot\hat z=1$, `sheet_modes`):
LSM$^v$: $e=\frac{c}{i\omega\varepsilon_0}\big(k_m\cos k_mu'\,P',\;(k^2-k_y^2)\sin k_mu'\,P\big)$,
$h=c\,(i\gamma\sin k_mu'\,P,\;0)$;
LSE$^v$: $e=c\,(-i\gamma\cos k_mu'\,P,\;0)$, $h=\frac{c}{i\omega\mu_0}\big(-k_m\sin k_mu'\,P',\;(k^2-k_y^2)\cos k_mu'\,P\big)$.

**Junction** empty guide | sheet guide: the same boxed mode-matching formulas as §5.2, with

```latex
\boxed{\;R_{pi}=\tfrac12\sqrt{2Y_p}\iint e_i\cdot e_p\,ds=\tfrac12\sqrt{2Y_p}\times
\begin{cases}\dfrac{c}{i\omega\varepsilon_0}\dfrac a2\Big[k_m\,\mathcal E^u_p\!\displaystyle\int_0^b\!\sin k_nv'\,P'\,dv'+(k^2-k_y^2)\,\mathcal E^v_p\!\int_0^b\!\cos k_nv'\,P\,dv'\Big]&\text{LSM}^v\\[2ex]
-i\gamma c\,\dfrac a{\epsilon_m}\,\mathcal E^u_p\displaystyle\int_0^b\sin k_nv'\,P\,dv'&\text{LSE}^v\end{cases}}
```

for PEC modes $p=(m,n)$ of the same $m$, with $e_p=(\mathcal E^u_p\cos k_mu'\sin k_nv',\;\mathcal E^v_p\sin k_mu'\cos k_nv')$:
$\mathcal E^u=Nk_m$, $\mathcal E^v=Nk_n$ (TM); $\mathcal E^u=-Nk_n$, $\mathcal E^v=Nk_m$ (TE). The $v$-integrals are
taken piecewise between sheets (`sheet_junction`, `sheet_filter_gsm`). Check: for $R_\square\to\infty$ the junction
GSM is $S_{11}=0$ and $S_{21}$ = the LS rotation $Q$ of `wrmodes.modes.ls_rotation` up to row signs, to
$10^{-14}$ (centred and off-centre sheets, $m=0,1,2$), which ties the sheet modes to the conventions.

**Butterweck's example** ($b$ = 10 mm, λ = 4 mm, $a$ = 22.86 mm, one centred sheet, `butterweck_sheet.py`):

| | optimum $R_\square$ | maximum attenuation |
| --- | --- | --- |
| EH$_{11}^{(2)}$ (LSM$^v$), exact roots, $\alpha=-\operatorname{Im}\gamma$ | 55.75 Ω | 1.554 dB/cm |
| same with Butterweck's $\beta\approx k$, $\alpha\approx\operatorname{Im}k_y^2/2k$ | | 1.532 dB/cm |
| Butterweck (eqs. 15–16; text) | 56 Ω (0.372 η₀λ/b = 56.1) | 1.5 dB/cm (2.76/kb² = 1.526) |
| EH$_{11}^{(1)}$ (LSE$^v$), exact | 418.3 Ω | 5.25 dB/cm (Butterweck: ≈420 Ω, ≈5 dB/cm) |

Deviations from Butterweck's 56 Ω and 1.5 dB/cm: −0.4 % and +3.6 % (the +3.6 % is mostly his $\beta\approx k$). With
$b$ = 10.16 mm: 54.9 Ω, 1.505 dB/cm. A 10 cm filter (56 Ω, WR-90, 75 GHz) passes TE$_{10}$ with $|S_{21}|=1$ exactly
and is reciprocal to $10^{-16}$; the TE$_{11}$ transmission converges slowly with the number of modes (−12.72,
−12.89, −12.97 dB for 6, 10, 14 modes per type), because the sheet modes have a kink or jump at the sheet. Three
equidistant 70 Ω sheets: the modes with a node at every sheet (n = 4) are not attenuated (Butterweck's footnote 1).

**Thickness.** The sheet is infinitely thin. A dielectric carrier perturbs H$_{10}$ (Butterweck p. 278: 0.1 mm mica,
$\varepsilon_r$ = 6, 70 Ω film: 0.1 dB/cm on H$_{10}$; 12 μm Mylar: not measurable). A carrier of finite thickness
is one more layer in the transverse resonance (out of scope).

## 7. Summary for WP2 and for the comparison with T1

* Use $dy/dz=(D+\kappa(K-\Lambda^{(1)}_{\text{full}})-\Lambda_{\text{full}})y$ with the boxed $\Lambda$, $\Lambda^{(1)}$; keep
  the degenerate TE/TM coupling and $\kappa\Lambda^{(1)}$; the $v$-wall coupling between families is negligible for
  metal walls ($<6\times10^{-6}$ in power for WR-90) and may be dropped if a component works per family.
* Lossy-wall filter: exact LS$^u$ modes + mode matching (§5), not the coupled system.
* Resistive sheets: exact LS$^v$ modes (any number of sheets) + mode matching (§6).
* Comparison interface (`docs/theory/T1b/t1b_coefficients.py`, inputs: mode tuples `('TE'|'TM', m, n)`, `a`, `b`
  in m, `f` in Hz, `Z` scalar or dict over `'u-','u+','v-','v+'` in ohm): `wall_A`, `wall_B`, `lambda_pp`
  ($\Lambda^{++}=\Lambda^{--}$), `lambda_pm` ($\Lambda^{+-}=\Lambda^{-+}$), `lambda_full`, `wall_A1`, `wall_B1`,
  `lambda1_pp`, `lambda1_pm`, `lambda1_full`, `alpha_te_marcuvitz`, `alpha_tm_marcuvitz`, `butterweck_eq21`,
  `lsu_dispersion`, `sheet_dispersion`.
