import numpy as np, sys
from writer import write_csv
UCAP = 10.0
COLS = ('curve', 'x', 'y', 'uncertainty', 'x_uncertainty')
REF = ['Source: Z. Ma, T. Yamane, E. Yamashita, "Analysis and design of mitered H-plane bends in rectangular waveguides',
       '  by combining the mode-matching and port reflection coefficient methods", Electronics and Communications in',
       '  Japan, Part 2, 80(5), 1-7 (1997) (transl. from Denshi Joho Tsushin Gakkai Ronbunshi J80-C-I(2), 78-85).',
       'Guide (p. 4): a = 15.799 mm, b = 7.899 mm; single-mode TE10 band 9.49 GHz to 19.0 GHz (f_c = 9.49 GHz).']
UTXT = ['uncertainty: L1b reading uncertainty of y = hypot(2 px * |slope|, 2 px) * (y units per px) + calibration residual,',
        '  2 px = half the printed line width; capped at %.0f where the curve is near vertical (then use x_uncertainty).' % UCAP,
        'x_uncertainty: 2 px in x units + calibration residual.']
METHOD = ['Method (L1b): page rendered with pdftoppm -r 300; pixel tracing in Python (numpy, Pillow, scipy); axis calibration',
          '  from tick marks (least squares); labels, ticks and reference lines masked; pixels assigned to hand-placed guide',
          '  polylines (Euclidean distance); column centre line sampled every 0.005 (normalized) or every 0.02 GHz where ink',
          '  exists, row centre line where |slope| > 1.5 px/px; points within 5 px of another curve dropped (crossings).',
          '  Overlays of the points on the page checked by eye.']

def rows_from(pts, rename, ux, ucap=None):
    ucap = UCAP if ucap is None else ucap
    rows = []
    for k, v in pts.items():
        for x, y, u in v:
            rows.append((rename(k), x, y, min(u, ucap), ux))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows

def fig11():
    g = {}
    exec(open('ma11.py').read().replace("if __name__ == '__main__':", 'if False:'), g)
    rows = rows_from(g['pts'], lambda k: 'c/a=' + k.rstrip('rst'), 2 * g['dbx'] + g['rx'])
    H = REF + [
     'Figure: Fig. 11, p. 6 (PDF p. 6). "Variation of the return loss of a mitered bend with different size of the mitered',
     '  part of the bend." Text p. 6: "It is found that when 0.6 <= c/a <= 0.8, the return loss of the bend can be made',
     '  small. It is also seen that when c/a = 0.631, the low return loss bandwidth (|S11| < -30 dB) is the widest,',
     '  reaching about 39% of the single mode frequency band."',
     'Geometry: 90 deg mitered H-plane bend, guide above; c = size of the mitered part (defined in Fig. 1/3 of the paper;',
     '  c/a = 0 is the right-angle corner bend). Curves are computed with 60 division steps, division method (a).',
     'Curves: c/a=0 (long dash), c/a=0.60 (solid), c/a=0.631 (long-short dash; nulls near 1.68 and 1.94),',
     '  c/a=0.70 (short dash; null near 1.38), c/a=1.0 (sparse dots), c/a=1.30 (dashes near the top).',
     '  The horizontal dotted line at -30 dB is a reference line, not data. Line styles were matched to the labels by',
     '  L1b; the c/a=0.631 assignment of the double-null curve follows the 39% bandwidth statement (|S11| < -30 dB',
     '  from about 1.55 to 1.96). c/a=0 above 1.9 (near 0 dB) and c/a=1.30 below 1.55 (hidden in the top frame) are',
     '  uncertain or missing. Parts below -50 dB are outside the axes.',
     'Axes: x = f/f_c (dimensionless), linear 1..2; y = |S11| in dB, linear -50..0.',
     'Calibration (L1b): ' + g['CAL']] + METHOD + UTXT + [
     'Digitizer: L1b (independent of L1a). Script: benchmarks/literature/raw/b/scripts/ma11.py.']
    print(write_csv('ma1997_fig11.csv', H, rows, cols=COLS, fmt=['%s', '%.4f', '%.3f', '%.3f', '%.4f']), len(rows))


def _write(name, o, fig_lines, rename, extra_rows=(), xu_extra=0.0, fmt_y='%.3f', ucap=None):
    ux = 2 * o['dbx'] + o['rx'] + o['hx'] * o['dbx']
    rows = rows_from(o['pts'], rename, ux, ucap) + list(extra_rows)
    rows.sort(key=lambda r: (r[0], r[1]))
    cal = ('x: ticks (px, two opposite axes) %s / %s -> %s; y: ticks %s / %s -> %s (PDF page rendered at 300 dpi;' % (
        o['xcal'][0], o['xcal'][1], o['xcal'][2], o['ycal'][0], o['ycal'][1], o['ycal'][2]),
        '  map = least-squares line through the mean of the two axes; half their largest difference, x %.2f px, y %.2f px,' % (o['hx'], o['hy']),
        '  is added to the uncertainties; fit residual x %.4f, y %.4f).' % (o['rx'], o['ry']))
    H = REF + fig_lines + ['Calibration (L1b): ' + cal[0], cal[1], cal[2]] + METHOD + UTXT + [
        '  BEM circles: centre from a Hough transform (radius 5-8 px), uncertainty 1.5 px.',
        'Digitizer: L1b (independent of L1a). Scripts: benchmarks/literature/raw/b/scripts/ma*.py, magen.py.']
    print(write_csv(name, H, rows, cols=COLS, fmt=['%s', '%.4f', fmt_y, fmt_y, '%.4f']), len(rows))

def _circ_rows(o, label):
    out = []
    for x, y, r in o['circ']:
        out.append((label, float(o['fx'](x)), float(o['fy'](y)), 1.5 * o['dby'] + o['ry'] + o['hy'] * o['dby'],
                    1.5 * o['dbx'] + o['rx'] + o['hx'] * o['dbx']))
    return out

def fig5():
    g = {}
    exec(open('ma5.py').read().replace("if __name__ == '__main__':", 'if False:'), g)
    o = g['o']; o.update(xcal=g['XCAL'], ycal=g['YCAL'])
    L = ['Figure: Fig. 5, p. 4 (PDF p. 4). "Variation of the power transmission coefficient of a waveguide right angle corner',
         '  bend with different numbers of modal terms in the electromagnetic field expressions."',
         'Geometry: 90 deg right-angle corner H-plane bend (no miter, c/a = 0), guide above.',
         'Curves: "3/6/12 terms": the three computed curves coincide within the line width except near 17-18.6 GHz, where',
         '  the 3-term dotted curve lies slightly to the right; the traced line is the solid (12 terms) there.',
         '  "BEM": circles, boundary-element results (legend "BEM [9]"; the text p. 4 cites the BEM as ref. [6]).',
         'Axes: x = frequency in GHz, linear (frame 9.5..19); y = |S21|^2 (power transmission), linear 0..0.9.']
    L.append('Note: in this file the y uncertainty cap is 0.1 (not 10 dB): near-vertical parts are read through x_uncertainty.')
    _write('ma1997_fig5.csv', o, L, lambda k: '3/6/12 terms', _circ_rows(o, 'BEM'), fmt_y='%.4f', ucap=0.1)

def fig6():
    g = {}
    exec(open('ma6.py').read().replace("if __name__ == '__main__':", 'if False:'), g)
    o = g['o']; o.update(xcal=g['XCAL'], ycal=g['YCAL'])
    L = ['Figure: Fig. 6, p. 4 (PDF p. 4). "Variation of the power transmission coefficient of a mitered bend with different',
         '  numbers of steps used in the staircase approximation." The y label reads |S21| (dB).',
         'Geometry: mitered H-plane bend, c/a = 1.0 (inset), guide above; staircase with 10/20/40/60 steps.',
         'Curves: "40/60 steps" (solid and long-short dash coincide within the line width), "20 steps" (dotted),',
         '  "10 steps" (long-short-short dash), "BEM" circles (legend "BEM [9]", text cites ref. [6]).',
         '  The curves differ by 3-8 px; tracing used 3 px guide tolerance, so the 20- and 10-step curves are sparse.',
         '  The x calibration uses the bottom ticks only (the top 16 GHz tick is merged with a circle); top and',
         '  bottom ticks differ by 1.5-2 px elsewhere.',
         'Axes: x = frequency in GHz, linear (frame 9.5..19); y = |S21| in dB, linear -16..0.']
    _write('ma1997_fig6.csv', o, L, lambda k: k, _circ_rows(o, 'BEM'))

def fig8():
    g = {}
    exec(open('ma89.py').read().replace("if __name__ == '__main__':", 'if False:'), g)
    o = g['o8']; o.update(xcal=g['F8']['xcal'], ycal=g['F8']['ycal'])
    L = ['Figure: Fig. 8, p. 5 (PDF p. 5). "Difference between the return loss of a mitered bend calculated with three',
         '  different division methods (the number of the division steps is 20)."',
         'Geometry: mitered H-plane bend, c/a = 1.0, 20 steps; division methods (a), (b), (c) of Fig. 7 (staircase on,',
         '  above, below the slanted line).',
         'Curves: (a) solid, (b) long dash, (c) dotted. Above about 16.6 GHz the three coincide and the points labelled (a)',
         '  are the common line.',
         'Axes: x = frequency in GHz, linear (frame 9.5..19); y = |S11| in dB, linear -35..0.',
         '  Note: the left y-axis ticks lie 7 px lower than the right ones (half difference 3.75 px = 0.26 dB, included).']
    _write('ma1997_fig8.csv', o, L, lambda k: k)

def fig9():
    g = {}
    exec(open('ma89.py').read().replace("if __name__ == '__main__':", 'if False:'), g)
    o = g['o9']; o.update(xcal=g['F9']['xcal'], ycal=g['F9']['ycal'])
    L = ['Figure: Fig. 9, p. 5 (PDF p. 5). "Difference between the return loss of a mitered bend calculated with three',
         '  different division methods (the number of the division steps is 60)."',
         'Geometry: mitered H-plane bend, c/a = 1.0, 60 steps; division methods (a), (b), (c).',
         'Curves: (a)=(b)=(c): the three coincide within the line width; one common line traced.',
         'Axes: x = frequency in GHz, linear (frame 9.5..19); y = |S11| in dB, linear -35..0.']
    _write('ma1997_fig9.csv', o, L, lambda k: k)

if __name__ == '__main__':
    for name in sys.argv[1:]:
        globals()[name]()
