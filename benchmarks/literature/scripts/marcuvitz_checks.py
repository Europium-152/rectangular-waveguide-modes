"""Numerical checks of the Marcuvitz (1951) transcription, docs/literature/marcuvitz-steps-and-bends.md (card L1).

Evaluates the transcribed formulas and compares them with values read by L1 on the book's own figures (readings
hard-coded below; read on 400 dpi renders by pixel cuts at fixed abscissa, about +-1 % of full scale):

* Sec. 5.24a/b, H-plane step, eq. (2a) vs Figs. 5.24-3 and 5.24-6, with the suspected misprints
  (lambda_g' instead of lambda_g in the denominator of (2a); 3 + alpha^2 instead of 1 + 3 alpha^2 in N_22 of 5.24b);
  location of the pole of X_11, X_22.
* Sec. 5.26a, E-plane step, eq. (2a)/(2b) with the power factor inside or outside the logarithm, vs Fig. 5.26-3.
* Sec. 5.28b, arbitrary-angle E-plane corner, eq. (1) with Psi(x) = psi(x + 1) or psi(x), vs Fig. 5.28-6.
* Sec. 5.28a/5.29a, which two-port (pi or T) reproduces eqs. (1)-(2) from the alternative circuit.

    python3 benchmarks/literature/scripts/marcuvitz_checks.py
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma, ellipe, ellipk

def params_sym(al):
    r = (1-al)/(1+al)
    R1 = -r**al
    R2 = r**(1/al)
    T = 4*al/(1-al**2)*r**(0.5*(al+1/al))
    return R1, R2, T

def params_asym(al):
    r = (1-al)/(1+al)
    R1 = -((1+3*al**2)/(1-al**2))*r**(2*al)
    R2 = ((3+al**2)/(1-al**2))*r**(2/al)
    T = 16*al**2/(1-al**2)**2*r**(al+1/al)
    return R1, R2, T

def N_sym(al, aL):
    """aL = a/lambda"""
    ap = np.sqrt(1-al**2)
    R1, R2, T = params_sym(al)
    r = (1-al)/(1+al)
    F = lambda k: ellipk(k**2); E = lambda k: ellipe(k**2)
    apL = al*aL
    Q = 1-np.sqrt(1-(2*aL/3)**2+0j)
    Qp = 1-np.sqrt(1-(2*apL/3)**2+0j)
    Q, Qp = Q.real if abs(Q.imag) < 1e-15 else Q, Qp.real if abs(Qp.imag) < 1e-15 else Qp
    N11 = (2*aL**2*(1+16*R1/(np.pi*(1-al**2))*(E(al)-ap**2*F(al))*(E(ap)-al**2*F(ap)) - R1**2 - al**2*T**2)
           + 12*al**4/(1-al**2)**2*r**(4*al)*(Q-0.5*(2*aL/3)**2)
           + 48*al**2/(1-al**2)**2*r**(al+3/al)*(Qp-0.5*(2*apL/3)**2))
    N22 = (2*aL**2*(al**2+16*al**2*R2*E(al)/(np.pi*(1-al**2))*(F(ap)-E(ap)) - al**2*R2**2 - T**2)
           + 48*al**2/(1-al**2)**2*r**(3*al+1/al)*(Q-0.5*(2*aL/3)**2)
           + 12/(1-al**2)**2*r**(4/al)*(Qp-0.5*(2*apL/3)**2))
    N12 = (2*aL**2*(ap**2-4*E(al)/np.pi*(E(ap)-al**2*F(ap)) - R1 + al**2*R2)*T
           + 24*al**3/(1-al**2)**2*r**(0.5*(7*al+1/al))*(Q-0.5*(2*aL/3)**2)
           + 24*al/(1-al**2)**2*r**(0.5*(al+7/al))*(Qp-0.5*(2*apL/3)**2))
    return (R1, R2, T), (N11, N22, N12)

def N_asym(al, aL, n22_den='1+3a2'):
    R1, R2, T = params_asym(al)
    r = (1-al)/(1+al)
    apL = al*aL
    Q = 1-np.sqrt(1-aL**2+0j); Qp = 1-np.sqrt(1-apL**2+0j)
    Q = Q.real if abs(Q.imag) < 1e-15 else Q; Qp = Qp.real if abs(Qp.imag) < 1e-15 else Qp
    lnr = np.log(r)
    den = (1+3*al**2) if n22_den == '1+3a2' else (3+al**2)
    N11 = (2*aL**2*(1+8*al*(1-al**2)/(1+3*al**2)*R1*lnr + 16*al**2/(1+3*al**2)*R1 - R1**2 - al**2*T**2)
           + 2*(32/3*al**4/(1-al**2)**2*r**(3*al))**2*(Q-0.5*aL**2)
           + 2*(32*al**2/(1-al**2)**2*r**(al+2/al))**2*(Qp-0.5*apL**2))
    N22 = (2*aL**2*(al**2-8*al*(1-al**2)/(3+al**2)*R2*lnr + 16*al**2/den*R2 - al**2*R2**2 - T**2)
           + 2*(32*al**2/(1-al**2)**2*r**(2*al+1/al))**2*(Q-0.5*aL**2)
           + 2*(32/3/(1-al**2)**2*r**(3/al))**2*(Qp-0.5*apL**2))
    N12 = (2*aL**2*((1-al**2)**2/(2*al)*lnr - R1 - al**2*R2)*T
           + 2/3*r**(5*al+1/al)*(32*al**3/(1-al**2)**2)**2*(Q-0.5*aL**2)
           + 2/3*r**(al+5/al)*(32*al/(1-al**2)**2)**2*(Qp-0.5*apL**2))
    return (R1, R2, T), (N11, N22, N12)

def circuit(al, aL, asym=False, den_prime=False, n22_den='1+3a2'):
    """Returns X lam_g/(Z0 2a), Z0'/Z0 * lam_g a/(lam_g' a') [sym] , l/a ; den_prime: use lam_g' in (2a) denominator."""
    (R1, R2, T), (N11, N22, N12) = (N_asym(al, aL, n22_den) if asym else N_sym(al, aL))
    D = (1-R1)*(1-R2)-T**2
    A = ((1+R1)*(1-R2)+T**2)/D
    Ap = ((1-R1)*(1+R2)+T**2)/D
    C = 2*T/D
    X11 = A/(1-1/(4*A)*((A+1)**2*N11+2*(A+1)*C*N12+C**2*N22))
    X22 = Ap/(1-1/(4*Ap)*((Ap+1)**2*N22+2*(Ap+1)*C*N12+C**2*N11))
    X0 = Ap - C**2/A          # the approximate form printed with X0 = X22 - X12^2/X11 ~ A' - C^2/A
    apL = al*aL
    two_ap_over_lgp_sq = (2*apL)**2 - 1     # (2a'/lam_g')^2
    two_a_over_lg_sq = (2*aL)**2 - 1        # (2a/lam_g)^2
    two_ap_over_lg_sq = al**2*two_a_over_lg_sq   # (2a'/lam_g)^2
    num = 1 + two_ap_over_lgp_sq*X0**2
    den = 1 + (two_ap_over_lgp_sq if den_prime else two_ap_over_lg_sq)*X0*X22
    Xn = X11*num/den                        # X lam_g/(Z0 2a)
    Zr = X11/(al**2*(X22-X0))*num           # Z0'/Z0 * lam_g a/(lam_g' a')
    la = al*X0/np.pi
    return Xn, Zr, la, dict(X11=X11, X22=X22, X0=X0, A=A, Ap=Ap, C=C, N=(N11, N22, N12))
def B526(al, bl, inside=True, full=True):
    """B lam_g/(Y0 b) from Marcuvitz 5.26a (2a) (full) or (2b); bl = b/lam_g; inside: power term inside ln."""
    p = ((1+al)/(1-al))**(0.5*(al+1/al))
    L = np.log((1-al**2)/(4*al)*p) if inside else np.log((1-al**2)/(4*al))*p
    if bl == 0:
        return 2*L
    s = np.sqrt(1-bl**2); sp = np.sqrt(1-(al*bl)**2)
    A = ((1+al)/(1-al))**(2*al)*(1+s)/(1-s) - (1+3*al**2)/(1-al**2)
    Ap = ((1+al)/(1-al))**(2/al)*(1+sp)/(1-sp) + (3+al**2)/(1-al**2)
    C = (4*al/(1-al**2))**2
    if full:
        br = L + 2*(A+Ap+2*C)/(A*Ap-C**2) + (bl/4)**2*((1-al)/(1+al))**(4*al)*((5*al**2-1)/(1-al**2)+4/3*al**2*C/A)**2
    else:
        br = L + 2/A
    return 2*br


def network_check(Ba=0.9, Bb=1.3):
    """Shift the reference planes of a pi (shunt jBa, series jBb*s, shunt jBa) or T network until it is a pure shunt."""
    def line(phi):
        return np.array([[np.cos(phi), 1j * np.sin(phi)], [1j * np.sin(phi), np.cos(phi)]])

    def sh(Y):
        return np.array([[1, 0], [Y, 1]])

    def se(Y):
        return np.array([[1, 1 / Y], [0, 1]])

    out = []
    for top in ('pi', 'T'):
        for s1 in (1, -1):
            for s2 in (1, -1):
                N = (sh(s1 * 1j * Ba) @ se(s2 * 1j * Bb) @ sh(s1 * 1j * Ba) if top == 'pi'
                     else se(s1 * 1j * Ba) @ sh(s2 * 1j * Bb) @ se(s1 * 1j * Ba))
                f = lambda p: (line(p) @ N @ line(p))[0, 1].imag
                ps = np.linspace(-np.pi / 2, np.pi / 2, 2001)
                v = [f(p) for p in ps]
                for i in range(len(ps) - 1):
                    if np.sign(v[i]) != np.sign(v[i + 1]):
                        p = brentq(f, ps[i], ps[i + 1])
                        M = line(p) @ N @ line(p)
                        out.append((top, s1, s2, p, M[1, 0].imag))
    return out


def main():
    print('Sec. 5.24a, X lambda_g/(Z0 2a): book Fig. 5.24-3 vs eq. (2a)')
    fig_s = {0.5: [0.254, 0.524, 0.972, 1.787, 2.48], 1.4: [0.377, 0.888, 1.816, 3.303, 4.37]}
    als = [0.304, 0.404, 0.504, 0.604, 0.654]
    for aL in (0.5, 1.4):
        for al, v in zip(als, fig_s[aL]):
            x = circuit(al, aL)[0]
            xp = circuit(al, aL, den_prime=True)[0]
            print(f'  a/lam={aL} alpha={al}: book {v:.3f}  printed {x:.3f} ({100 * (x / v - 1):+.0f} %)  '
                  f'lambda_g\' variant {xp:.3f} ({100 * (xp / v - 1):+.0f} %)')
    print('  Fig. 5.24-2 at a/lam = 0.5 (book 1.52 at alpha 0.5, 1.33 at 0.7; l/a peak 2.53e-3 near 0.48):',
          round(circuit(0.5, 0.5)[1], 3), round(circuit(0.7, 0.5)[1], 3), round(circuit(0.48, 0.5)[2], 5))
    print('  Fig. 5.24-2 at a/lam = 1.4 (book about 1.49 at alpha 0.5):', round(circuit(0.5, 1.4)[1], 3))
    print('Sec. 5.24b, X lambda_g/(Z0 2a): book Fig. 5.24-6 vs eq. (5a), N22 with 1 + 3 alpha^2 (printed) or 3 + alpha^2')
    fig_a = {1 / 1.05: [0.207, 0.613, 2.283, 3.254], 0.5: [0.138, 0.395, 1.13, 1.707]}
    for aL in (1 / 1.05, 0.5):
        for al, v in zip([0.404, 0.504, 0.654, 0.704], fig_a[aL]):
            x = circuit(al, aL, asym=True)[0]
            x2 = circuit(al, aL, asym=True, n22_den='3+a2')[0]
            print(f'  lam/a={1 / aL:.2f} alpha={al}: book {v:.3f}  printed {x:.3f} ({100 * (x / v - 1):+.0f} %)  '
                  f'3+alpha^2 variant {x2:.3f} ({100 * (x2 / v - 1):+.0f} %)')
    print('Pole of X11 or X22 (first alpha where one turns negative):')
    for asym in (False, True):
        for aL in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4):
            first = None
            for al in np.arange(0.05, 0.99, 0.005):
                d = circuit(al, aL, asym=asym)[3]
                if d['X11'] < 0 or d['X22'] < 0:
                    first = round(al, 3)
                    break
            print(f'  {"asym" if asym else "sym "} a/lam={aL}: {first}')
    print('Sec. 5.26a, B lambda_g/(Y0 b), power factor inside / outside the ln (book Fig. 5.26-3, b/lam_g = 0: '
          'about 3.8, 2.45, 1.7, 0.78 at b\'/b = 0.1, 0.2, 0.3, 0.5)')
    for bl in (0, 0.4, 0.8):
        print('  b/lam_g', bl, ' '.join(f'{al}: {B526(al, bl):.3f} / {B526(al, bl, inside=False):.3f}'
                                     for al in (0.1, 0.2, 0.3, 0.5, 0.7)))
    print('Sec. 5.28b, B_a lambda_g/(Y0 2b) with Psi(x) = psi(x+1) / psi(x) (book Fig. 5.28-6: 0.645, 0.88, 1.07 at '
          '60, 90, 120 deg)')
    for th in (30, 60, 75, 90, 120):
        t = np.radians(th)
        x = -0.5 * (1 - t / np.pi)
        print(f'  {th} deg: {digamma(x + 1) - digamma(0.5):.4f} / {digamma(x) - digamma(-0.5):.4f};  '
              f'B_b b/(Y0 lam_g) = {1 / np.tan(t / 2) / (2 * np.pi):.4f}')
    Ba, Bb = 0.9, 1.3
    print('Network topology (Ba = 0.9, Bb = 1.3): 5.28a eq. (1) gives B =', round((Ba ** 2 + 1) / Bb - 2 * Ba, 4),
          'and eq. (2) phi =', round(np.arctan2(1, 2 * Bb - Ba), 4), '; 5.29a eq. (1) gives B =',
          round((1 + Ba ** 2) / Bb + 2 * Ba, 4), 'and |phi| =', round(np.arctan2(1, Ba + 2 * Bb), 4))
    for top, s1, s2, p, B in network_check(Ba, Bb):
        print(f'  {top}: sign(B_a) {s1:+d}, sign(B_b) {s2:+d}: pure shunt {B:+.4f} at phi = {p:+.4f}')


if __name__ == '__main__':
    main()
