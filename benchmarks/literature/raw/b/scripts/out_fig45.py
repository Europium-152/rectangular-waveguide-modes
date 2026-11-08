import numpy as np
from writer import write_csv
UCAP = 10.0
COLS = ('curve', 'x', 'y', 'uncertainty', 'x_uncertainty')
FMT = ['%s', '%.4f', '%.3f', '%.3f', '%.4f']
COMMON_U = [
 'uncertainty: L1b reading uncertainty of y (dB) = hypot(2 px * |slope|, 2 px) * (dB/px) + calibration residual,',
 '  2 px = half the printed line width; capped at %.0f dB where the curve is near vertical (then use x_uncertainty).' % UCAP,
 'x_uncertainty: L1b reading uncertainty of x = 2 px * (x/px) + calibration residual.',
]
REF = ['Source: F. Alessandri, M. Mongiardo, R. Sorrentino, "Rigorous mode matching analysis of mitered E-plane bends in',
       '  rectangular waveguide", IEEE Microwave and Guided Wave Letters 4(12), 408-410 (1994), doi:10.1109/75.336229.']
GEOM = ['Geometry: 90 deg E-plane miter (inset): narrow wall b in the bend plane, d = distance from the inner corner to the',
        '  miter wall (d/b = 1.41: no mitering; d/b = 0.707: diagonal). The paper gives no absolute dimensions.',
        'Axes: x = frequency normalized to the cut-off frequency (dimensionless), linear; y = |S11| in dB, linear.']
METHOD = ['Method (L1b): PDF page 2 rendered with pdftoppm -r 300; pixel tracing in Python (numpy, Pillow, scipy). Axis',
          '  calibration from the tick marks (least-squares line through all major ticks). Labels, inset and gridlines',
          '  masked; dark pixels assigned to hand-drawn guide polylines (one per curve or branch); column centre line',
          '  sampled every 0.005 in x where ink exists (dashed: only on dashes), row centre line every 6 px where',
          '  |slope| > 1.5 px/px; points within 6 px of another curve (crossings) removed.',
          '  Overlays of the points on the page were checked by eye.']

exec(open('fig4.py').read().replace("if __name__ == '__main__':", 'if False:'))
rows = []
for k, v in pts.items():
    for x, y, u in v:
        rows.append((k, x, y, min(u, UCAP), 2 * dbx + rx))
H = REF + [
 'Figure: Fig. 4, p. 409 (PDF p. 2). "Return loss of a 90 deg mitered E-plane bend in rectangular waveguide. The',
 '  frequency is normalized with respect to the cut-off frequency. Crosses represent results from [7]."',
 '  [7] = J. M. Reiter, F. Arndt, "A full-wave boundary contour mode-matching method (BCMM) ...", 1994 IEEE MTT-S',
 '  Int. Microwave Symp. Dig., San Diego, pp. 1021-1024.',
 'Curves: d/b=1.41 (solid), d/b=0.707 (dashed), d/b=0.87 (dotted: one point per dot);',
 '  ref7_d/b=<v>: crosses of ref. [7] (centre = midpoint of the vertical stroke), assigned to the nearest curve.',
 '  The dashed d/b=0.707 curve is hidden under its crosses for x > 1.45, hence sparse there.',
 '  The text (p. 409-410) quotes for d/b = 0.87 "a matching better than -26 dB in the whole band".',
] + GEOM + ['  x range 1..2, y range -40..0 dB.', 'Calibration (L1b): ' + CAL,
 'Scale: %.4f dB/px, %.6f per px in x.' % (dby, dbx)] + METHOD + [
 '  Crosses erased (21 x 21 px box) before tracing the curves.'] + COMMON_U + [
 'Digitizer: L1b (independent of L1a). Script: benchmarks/literature/raw/b/scripts/fig4.py.']
print(write_csv('alessandri1994_fig4.csv', H, rows, cols=COLS, fmt=FMT), len(rows))

exec(open('fig5.py').read().replace("if __name__ == '__main__':", 'if False:'))
rows = []
for k, v in pts.items():
    for x, y, u in v:
        rows.append(('d/b=' + k.rstrip('r'), x, y, min(u, UCAP), 2 * dbx + rx))
rows.sort(key=lambda r: (r[0], r[1]))
H = REF + [
 'Figure: Fig. 5, p. 409 (PDF p. 2). "Return loss of a mitered E-plane bend in rectangular waveguide. The frequency is',
 '  normalized with respect to the cut-off frequency. The dept[h] of the mitering is described by the parameter d/b."',
 'Curves: d/b=0.9 (thin solid, upper), d/b=0.87 (thin solid, lower; null near x = 1.83-1.85, below the -50 dB axis),',
 '  d/b=0.85 (thick dashed; null near x = 1.62-1.65), d/b=0.83 (thin dashed; null near x = 1.30-1.38).',
 '  The parts below -50 dB are outside the axes; the two branches either side of a null share the curve label.',
] + GEOM + ['  x range 1..2, y range -50..0 dB.', 'Calibration (L1b): ' + CAL,
 'Scale: %.4f dB/px, %.6f per px in x.' % (dby, dbx)] + METHOD + COMMON_U + [
 'Digitizer: L1b (independent of L1a). Script: benchmarks/literature/raw/b/scripts/fig5.py.']
print(write_csv('alessandri1994_fig5.csv', H, rows, cols=COLS, fmt=FMT), len(rows))
