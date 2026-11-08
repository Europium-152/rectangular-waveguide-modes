import sys, numpy as np
sys.argv = ['x']
exec(open('rei.py').read())
from writer import write_csv
REF = ['Source: J. M. Reiter, F. Arndt, "Rigorous analysis of arbitrarily shaped H- and E-plane discontinuities in rectangular',
       '  waveguides by a full-wave boundary contour mode-matching method", IEEE Trans. MTT 43(4), 796-801 (1995).']
F3 = ['Figure 3, p. 799 (PDF p. 4): "Return loss curves of arbitrary angle H-plane or E-plane bends (waveguide Ku-band, WR62',
      '  housing 15.799 x 7.899 mm). (a) H-plane bend. (b) Mitered H-plane bend. (c) E-plane bend. (d) Mitered E-plane bend."',
      'Text p. 798 (PDF p. 3): "For these calculations, 8 modes in the waveguide ports and 15 modes in the cavity subregions (formed by',
      '  the port planes and the corner structure) have turned out to yield stable results."',
      'Geometry (insets, read by L1b): gamma = interior angle between the inner walls at the inner corner (gamma = 90 deg is a',
      '  right-angle bend, bend angle = 180 deg - gamma); a = 15.799 mm, b = 7.899 mm. In (b) a_opt and in (d) b_2opt is the',
      '  distance from the inner corner to the miter wall (arrow drawn from the inner corner to the chamfer).']
METH = ['Method (L1b): page rendered with pdftoppm -r 300; calibration = affine map fitted to the intersections of the dotted',
        '  gridlines (each gridline fitted to its dots), so that the slight rotation of the printed figure is corrected;',
        '  dots (components < 10 px) removed; labels and leader lines masked; pixels assigned to guide polylines (Euclidean,',
        '  5-6 px), centre line every 4 px (column centres; row centres where |slope| > 1.5 px/px); points within 5-6 px of',
        '  another curve dropped. "#r" parts of a curve were traced separately and carry the same label.',
        'uncertainty: y = hypot(2 px, 2 px * slope) in data units + calibration residual (px) in data units, capped at 10 dB;',
        '  x_uncertainty = (2 px + calibration residual) in GHz.',
        'Axes: x = frequency in GHz, linear 12..18; y = |S11| in dB, linear.']
COLS = ('curve', 'x', 'y', 'uncertainty', 'x_uncertainty')
FMT = ['%s', '%.4f', '%.3f', '%.3f', '%.4f']

def rows(pts, lab=lambda k: k.split('#')[0]):
    r = []
    for k, v in pts.items():
        for X, Y, uY, uX in v:
            r.append((lab(k), X, Y, uY, uX))
    r.sort(key=lambda q: (q[0], q[1]))
    return r

def calline(c, vl, hl):
    return ('Calibration (L1b): gridlines x = %s at approx. px %s; y = %s at approx. px %s; affine map px = M (f, S) + t,' % (
        [v for v, _ in vl], [p for _, p in vl], [v for v, _ in hl], [p for _, p in hl]),
        '  M = %s, t = %s; max residual of the gridline intersections %.2f px (x), %.2f px (y).' % (
        np.round(c['M'], 4).tolist(), np.round(c['t'], 2).tolist(), c['res'][0], c['res'][1]))

for key, spec, kw, title, extra in [
        ('3a', A, dict(tol=6), '(a) H-plane bend (no miter), gamma = 90, 105, 120, 135, 150 deg; y range -35..0 dB.',
         ['  Curves are interrupted by their labels between 16.0 and about 16.95 GHz; gamma=90 above 16.9 GHz runs into the top frame.']),
        ('3b', B, dict(tol=5, refine=(12,)), '(b) mitered H-plane bend, gamma=105 deg (a_opt = 0.949 a), 120 deg (a_opt = 0.955 a),',
         ['  135 deg (a_opt = 0.956 a); y range -55..-25 dB. Nulls below -55 dB (outside the axes). Branch labels: the',
          '  branches ending at -26/-25.5 dB belong to gamma=105, those ending at -33/-34.5 dB to gamma=120 (leader lines on',
          '  the left; on the right, by L1b judgment from the ordering of the curves). The left branches of gamma=105 and 120',
          '  cross near 14.5 GHz, -48 dB; points within 5 px of the crossing are dropped.']),
        ('3c', C, dict(tol=6, refine=(12,)), '(c) E-plane bend (no miter), gamma = 90..150 deg; y range -35..0 dB.',
         ['  Curves are interrupted by their labels between 16.0 and about 16.95 GHz.']),
        ('3d', Dd, dict(tol=5, cross_r=6), '(d) mitered E-plane bend, gamma=105 deg (b_2opt = 0.873 b), 120 deg (b_2opt = 0.898 b),',
         ['  135 deg (b_2opt = 0.918 b); y range -65..-35 dB. "right branch": the rising branch from the null near 17.2 GHz',
          '  to -50 dB at 18 GHz; it is not identifiable as gamma=105 or 135 from the figure. gamma=105 and 135 both reach',
          '  -65 dB near 16.6 GHz; points within 6 px of each other are dropped there.'])]:
    c, parts, pts = panel(**spec, **kw)
    vl, hl = spec['cal_args'][0], spec['cal_args'][1]
    H = REF + F3 + ['Panel: ' + title] + extra + list(calline(c, vl, hl)) + METH + [
        'Digitizer: L1b (independent of L1a). Script: benchmarks/literature/raw/b/scripts/rei.py.']
    print(write_csv('reiter1995_fig%s.csv' % key, H, rows(pts), cols=COLS, fmt=FMT), sum(len(v) for v in pts.values()))

c, parts, pts = fig4()
H = REF + ['Figure 4, p. 799 (PDF p. 4): "Calculated and measured return loss of a cascaded structure of mitered 90 H-plane and',
           '  E-plane bends (waveguide X-band, WR90-housing: 22.86 x 10.16 mm). Optimum miter dimensions in (mm): H-plane',
           '  a_opt = 0.976a, E-plane b_opt = 0.874b." Text p. 798 (PDF p. 3): "Fig. 4 shows the overall return loss for a structure with an',
           '  intermediate length of zero l = 0 and of a length identical to the waveguide height l = b between the two bends."',
           'Geometry: mitered 90 deg H-plane bend followed by a mitered 90 deg E-plane bend, straight of length l between them',
           '  (inset), l = 0 or l = b = 10.16 mm; a = 22.86 mm.',
           'Curves: l=0 and l=b. Theory (dashed) and measurement (solid) coincide within the line width over most of the band;',
           '  where they separate (columns with two distinct runs: near the nulls at 9.4 and 11.7 GHz and above about 11.9 GHz)',
           '  no point is given. Label assignment of the 11.7 GHz null to l=b by continuity of the curves through the crossings',
           '  near 10.35 and 11.1 GHz (L1b judgment). Printed minima (not traced): l=b near 9.4 GHz, measured about -40 dB,',
           '  theory about -43 dB; near 11.72 GHz, theory about -55 dB, measured about -60 dB (rough readings, +-1 dB).',
           'Axes: x = frequency in GHz, linear 8.2..12.4; y = |S11| in dB, linear -70..-10.',
           'Calibration (L1b): the figure has no ticks or gridlines; the frame is the calibration: left x = 506.5 px (8.2 GHz),',
           '  right 1229.0 px (12.4), top y = 1966.5 px (-10 dB), bottom 2401.0 px (-70). Check: label centres of 9, 10, 11 GHz',
           '  and of -30, -50 dB agree with this map within 1 px. Calibration uncertainty taken as 1.5 px.',
           'Method (L1b): as for Fig. 3 but with the frame calibration; labels, leader lines, legend and inset masked.',
           'uncertainty: y = hypot(2 px, 2 px * slope) + 1.5 px in dB (cap 10 dB); x_uncertainty = 3.5 px in GHz.',
           'Digitizer: L1b (independent of L1a). Script: benchmarks/literature/raw/b/scripts/rei.py.']
print(write_csv('reiter1995_fig4.csv', H, rows(pts), cols=COLS, fmt=FMT), sum(len(v) for v in pts.values()))
