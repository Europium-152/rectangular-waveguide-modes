exec(open('/home/claude/wt/L1a-scratch/rt_out.py').read())
FIG = sys.argv[1]
if FIG == '3a':
    box = (560, 560, 1340, 1100); cal = grid_calib(560, 580, 1340, 1080, [13, 14, 15, 16, 17], [-5, -10, -15, -20, -25, -30]); yr = (-35, 0)
else:
    box = (560, 1340, 1340, 1920); cal = grid_calib(560, 1360, 1340, 1900, [13, 14, 15, 16, 17], [-5, -10, -15, -20, -25, -30]); yr = (-35, 0)
cX, cY, P, res, slopes = cal
print('resid', abs(res).max(0), 'slopes', min(slopes), max(slopes))
J, comps, mx, my = panel(*box, cX, cY, (12, 18), yr)
def cy_(c): return todata(cX, cY, (c['x0'] + c['x1']) / 2, (c['y0'] + c['y1']) / 2)
left = sorted([c for c in comps if c['x1'] - c['x0'] > 200], key=lambda c: c['y0'])
right = sorted([c for c in comps if c['x1'] - c['x0'] <= 200], key=lambda c: c['y0'])
print(len(left), len(right))
gam = [90, 105, 120, 135, 150]
rows = []; tr = []
for g, cl, cr in zip(gam, left, right):
    for c in (cl, cr):
        t = comp_trace(c); tr.append(t)
        rows += grid_rows('gamma=%d' % g, t, cX, cY, mx, my)
overlay(img, [t[:, :2] for t in tr], ['red', 'blue'] * 5, 'rt%s_ov.png' % FIG, crop=box, r=1)
import collections; print(collections.Counter(r[0] for r in rows))
plane = 'H' if FIG == '3a' else 'E'
side = 'a' if FIG == '3a' else 'b'
H = header(FIG[0] + '(' + FIG[1] + ')', "Return loss curves of arbitrary angle H-plane or E-plane bends (waveguide Ku-band, WR62 housing: 15.799 x 7.899 mm). (%s) %s-plane bend." % (FIG[1], plane),
    "gamma = 90, 105, 120, 135, 150 deg (solid lines, top to bottom); each line is interrupted by its label between about 16.1 and 17.2 GHz, so there is no data there",
    "unmitered (sharp-corner) %s-plane bend in WR62 (a = 15.799 mm, b = 7.899 mm), side %s in the bend plane; gamma is the angle between the two arms as drawn in the inset (gamma = 180 deg would be a straight guide, gamma = 90 deg a right-angle bend; deflection = 180 deg - gamma)" % (plane, side),
    "frequency in GHz (axis 12.0..18.0, dotted grid every 1 GHz)", "|S11| in dB (axis label |S11|/dB, axis 0..-35 dB, dotted grid every 5 dB)", P, res, cX, cY, slopes,
    extra=["uncertainty: reading uncertainty of y (dB) at the given x: half the printed line width (perpendicular, converted to vertical with the local slope) and 1 px calibration error in both axes",
           "x_uncertainty: 0 (regular grid)", "sampling: x on a 0.05 GHz grid where ink exists (no interpolation across gaps > 0.25 GHz)",
           "axis resolution: no ticks; grid every 1 GHz (%.1f px) and 5 dB (%.1f px): 0.5 px = %.4f GHz and %.3f dB" % (1 / mx, 5 / my, mx / 2, my / 2),
           "the gamma=90 curve reaches the 0 dB frame near 17.7 GHz; points within 4 px of the frame are left out"])
write_csv('/home/claude/wt/L1a-scratch/out/reiter1995_fig%s.csv' % FIG, H, rows)
