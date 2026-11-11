"""Axis calibration from tick marks (pass a): linear least squares with a small shear for the axis tilt."""
import numpy as np
from l2common import ticks, line_samples_v, line_samples_h, robust_linfit


class TickCal:
    def __init__(self, im, X, Y, xt, yt, xvals, yvals, xr_axis, yr_axis, extra_x=(), extra_y=(), label=''):
        """X, Y: approximate column of the y axis and row of the x axis. xt = (band, range) for x ticks, yt for y
        ticks. extra_x / extra_y: (value, 'axis') to use the axis line itself as a tick."""
        sv = line_samples_v(im, X, range(yr_axis[0], yr_axis[1], 20), half=10, win=20, thr=100, minmass=1500)
        cv, rv, _, _ = robust_linfit(sv[:, 0], sv[:, 1])
        sh = line_samples_h(im, Y, range(xr_axis[0], xr_axis[1], 20), half=10, win=20, thr=100, minmass=1500)
        ch, rh, _, _ = robust_linfit(sh[:, 0], sh[:, 1])
        self.kx, self.ky = cv[0], ch[0]            # dx/dy of the y axis, dy/dx of the x axis
        self.Yax = np.polyval(ch, X)                 # x-axis row at the y axis
        self.Xax = np.polyval(cv, Y)
        tx = ticks(im, 'x', int(round(Y)), xt[0], xt[1], 0.7)
        ty = ticks(im, 'y', int(round(X)), yt[0], yt[1], 0.7)
        assert len(tx) == len(xvals), (label, tx, xvals)
        assert len(ty) == len(yvals), (label, ty, yvals)
        px, vx = list(tx), list(xvals)
        py, vy = list(ty), list(yvals)
        for v in extra_x:
            px.append(self.Xax); vx.append(v)
        for v in extra_y:
            py.append(self.Yax); vy.append(v)
        self.px, self.vx, self.py, self.vy = map(np.array, (px, vx, py, vy))
        self.cx = np.polyfit(self.px, self.vx, 1)
        self.cy = np.polyfit(self.py, self.vy, 1)
        self.rx = self.vx - np.polyval(self.cx, self.px)
        self.ry = self.vy - np.polyval(self.cy, self.py)
        self.sx = float(np.sqrt(np.mean(self.rx ** 2)))
        self.sy = float(np.sqrt(np.mean(self.ry ** 2)))
        self.xpp = abs(self.cx[0]); self.ypp = abs(self.cy[0])

    def __call__(self, px, py):
        # shear: a point's x tick position is where its column meets the x axis following the y-axis tilt
        pxa = px - self.kx * (py - self.Yax)
        pya = py - self.ky * (px - self.Xax)
        return np.polyval(self.cx, pxa), np.polyval(self.cy, pya)

    def describe(self, fmt='{:.4g}'):
        s = ('x ticks px ' + ', '.join(f'{p:.1f}->{fmt.format(v)}' for p, v in zip(self.px, self.vx)) +
             '; y ticks px ' + ', '.join(f'{p:.1f}->{fmt.format(v)}' for p, v in zip(self.py, self.vy)) +
             f'; fit x = {self.cx[0]:.7g}*px + {self.cx[1]:.7g} (rms residual {self.sx:.3g}, max '
             f'{np.abs(self.rx).max():.3g}), y = {self.cy[0]:.7g}*py + {self.cy[1]:.7g} (rms residual {self.sy:.3g}, '
             f'max {np.abs(self.ry).max():.3g}); axis tilt dx/dy {self.kx:+.5f}, dy/dx {self.ky:+.5f} corrected as a '
             'shear')
        return s
