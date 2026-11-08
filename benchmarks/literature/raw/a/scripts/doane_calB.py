# Calibration B: y scale from the side ticks 0, 0.5, 1, 1.5 only (top frame treated as a plain border)
def side_param(yk):   # yk: pixel rows at Y = 0, 0.5, 1, 1.5
    s = CubicSpline(YK[:4], yk, bc_type='natural')
    per = (yk[3] - yk[0]) / 1.5
    return lambda Y: np.where(Y <= 1.5, s(np.minimum(Y, 1.5)), yk[3] + (Y - 1.5) * per)
SLB = side_param(ly[:4]); SRB = side_param(ry[:4])
yL2 = float(SLB(2.0)); yR2 = float(SRB(2.0))
dL = yL2 - C01[1]; dR = yR2 - C11[1]
C01B = np.array([P(pL, yL2), yL2]); C11B = np.array([P(pR, yR2), yR2])
def TcB(X):
    u = (X - 15) / 3; x = ST(X); return np.array([x, P(pT, x) + (1 - u) * dL + u * dR])
def LcB(Y): y = SLB(Y); return np.array([P(pL, y), y])
def RcB(Y): y = SRB(Y); return np.array([P(pR, y), y])
def coonsB(X, Y):
    u = (X - 15) / 3; v = Y / 2
    return ((1 - v) * Bc(X) + v * TcB(X) + (1 - u) * LcB(Y) + u * RcB(Y)
            - ((1 - u) * (1 - v) * C00 + u * (1 - v) * C10 + (1 - u) * v * C01B + u * v * C11B))
def invB(px, py, guess=(16.5, 1.0)):
    r = least_squares(lambda q: coonsB(q[0], q[1]) - np.array([px, py]), guess, xtol=1e-12, ftol=1e-12)
    return r.x
# where is the top frame in calibration B?
topY = [invB(*Tc(X), guess=(X, 1.95))[1] for X in (15.0, 16.5, 18.0)]
