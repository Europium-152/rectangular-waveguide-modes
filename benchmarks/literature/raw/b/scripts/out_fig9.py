import numpy as np
from writer import write_csv
exec(open('d9pts.py').read().replace("if __name__ == '__main__':", 'if False:'))
from d9map import BL, BR, TL, TR, BT, TT, LT, RT
FLOOR = (0.01, 0.01)
rows = []
for k, v in pts.items():
    for fi, pi, uy, ux, dp, df in v:
        rows.append((k, fi, pi, max(uy, FLOOR[0]), max(ux, FLOOR[1])))
H = [
 'Source: J. L. Doane, T. N. Anderson, "Oversized rectangular waveguides with mode-free bends and twists for broadband',
 '  applications", Microwave Journal 32(3), 153-160 (1989).',
 'Figure: Fig. 9, p. 159 (PDF p. 7). "Mode conversion in 20" 90 deg twists in WR187 tall guide (1.872" x 0.872" ID)',
 '  for various rates of twist variations." Text p. 159: "In Figure 9 we compare the theoretical predictions for 20"',
 '  twists in WR187." (theory, ref. 2 of the paper).',
 'Geometry: 90 deg twist, length 20 in (508 mm); WR187 used as tall guide: inside 1.872 in x 0.872 in',
 '  (47.55 mm x 22.15 mm); rate of twist constant, cosine or triangular (eq. 17 for triangular).',
 'Curves: constant (dashed), cosine (solid, upper), triangular (solid, lower).',
 'Axes: x = frequency in GHz, linear 15..18; y = total spurious mode power in % of input power, linear 0..2.',
 'Calibration (L1b): the PDF page is a photograph of a curved magazine page (frame not rectangular: the bottom frame',
 '  drops 60 px from left to right, the left and right frames lean). Page 7 rendered with pdftoppm -r 300, cropped',
 '  at (3200, 8350, 9200, 12400) and resized by 1/2. Frame lines traced pixel by pixel; tick marks found as dark',
 '  runs perpendicular to the frame. Map = Coons patch through the four traced frame lines, each parameterized',
 '  by its ticks (PCHIP). Calibration points (half-res px): corners (15,0) %s, (18,0) %s, (15,2) %s, (18,2) %s;' % (
     tuple(round(c, 1) for c in BL), tuple(round(c, 1) for c in BR), tuple(round(c, 1) for c in TL), tuple(round(c, 1) for c in TR)),
 '  bottom ticks f=15.5..17.5 at x = %s; top ticks at x = %s;' % ([round(v, 1) for v in BT], [round(v, 1) for v in TT]),
 '  left ticks P=0.5,1,1.5 at y = %s; right ticks at y = %s.' % ([round(v, 1) for v in LT], [round(v, 1) for v in RT]),
 '  Check: a single homography fitted to the same 20 points has max residual 0.018 GHz and 0.028 %.',
 'Method (L1b): curve pixels (grey < 100) assigned to guide polylines (Euclidean distance, 30 px), labels and pointer',
 '  arrows masked; centre line every 8 px (column centres, row centres where |slope| > 1.5); points within 16 px of',
 '  another curve (crossings) dropped; points mapped to data with the Coons patch.',
 'uncertainty (y, %%): hypot(2.5 px, 2.5 px * slope) in data units + |difference between the Coons and the homography',
 '  calibrations| at that point, floor %.2f %%. x_uncertainty (GHz): 2.5 px + calibration difference, floor %.2f GHz.' % FLOOR,
 '  2.5 px (half-res) = half the printed line width. Show-through of the reverse page is lighter than the curves.',
 'Note: the triangular curve rises from 0.047 % at 15 GHz to 0.150 % at 18 GHz in this calibration (it is straight in',
 '  the photo while the bottom frame bends); both calibrations agree on this within 0.012 %.',
 'Digitizer: L1b (independent of L1a). Scripts: benchmarks/literature/raw/b/scripts/d9*.py.',
]
print(write_csv('doane1989_fig9.csv', H, rows, cols=('curve', 'x', 'y', 'uncertainty', 'x_uncertainty'),
                fmt=['%s', '%.4f', '%.4f', '%.4f', '%.4f']), len(rows))
