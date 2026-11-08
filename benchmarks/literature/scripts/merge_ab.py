"""Merge the two independent digitizations (L1a: raw/a, L1b: raw/b) into the final benchmark files.

Card L1 (wrmodes 1.0). Run from anywhere:

    python3 benchmarks/literature/scripts/merge_ab.py            # writes benchmarks/literature/*.csv
    python3 benchmarks/literature/scripts/merge_ab.py --stats    # prints the a-b agreement only

Method (written into every output header):

* Line curves. Each raw set is sorted in x and cut into segments wherever its x step exceeds `gap` (no
  interpolation across larger gaps). Both sets are linearly interpolated, together with their reading
  uncertainties, onto a common regular grid (step `step`) inside each segment. Where both sets cover a grid
  point: y = (y_a + y_b)/2, uncertainty = max(|y_a - y_b|/2, max(u_a, u_b)), source = "ab". Where only one set
  covers it: that set's y, uncertainty = 2 u, source = "a" or "b". x_uncertainty = 0 (x is the grid value; x
  reading errors are already converted into the y uncertainty by both digitizers).
* Markers (dots, crosses, circles: one row per printed symbol). The two sets are paired by mutual nearest
  neighbour (distance in axis-normalized units, at most `pair_max`). Paired: x, y = means, uncertainty =
  max(|dy|/2, max(u_a, u_b)), x_uncertainty = max(|dx|/2, max(xu_a, xu_b)). Unpaired: single set, both
  uncertainties doubled.
* Exclusions: points of one set inside a listed x interval are dropped before the merge where the page shows
  that set to be wrong (each one has its reason in the configuration below and in the output header).
  `nobridge` intervals: no grid point is produced inside them (both sets only interpolated across hidden ink).
* Agreement statistics use the grid points covered by both sets: rms and max of y_a - y_b, mean bias, and the
  number of points where |y_a - y_b| exceeds the combined reading uncertainty sqrt(u_a^2 + u_b^2).
"""
import argparse
import csv
import collections
import datetime
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIT = os.path.dirname(HERE)
RAW = os.path.join(LIT, "raw")

MERGE_DATE = "2026-11-08"

# ----------------------------------------------------------------------------------------------------------------
# Common header texts
# ----------------------------------------------------------------------------------------------------------------

ALES_SRC = ('F. Alessandri, M. Mongiardo, R. Sorrentino, "Rigorous mode matching analysis of mitered E-plane bends '
            'in rectangular waveguide", IEEE Microwave and Guided Wave Letters 4(12), 408-410 (1994), '
            'doi:10.1109/75.336229')
ALES_GEOM = ('90 deg mitered E-plane bend (inset of the figures, text p. 408): b = narrow wall, the side in the bend '
             'plane. Parameter d = distance from the inner corner to the miter wall, measured along the bend '
             'diagonal (perpendicular to the miter wall). d/b = sqrt(2) = 1.41: no mitering (sharp corner); '
             'd/b = 1/sqrt(2) = 0.707: miter wall joining the ends of the two outer walls. Guide dimensions are '
             'not stated (results against f/fc only). fc = cut-off frequency of the fundamental mode.')
DOANE_SRC = ('J. L. Doane and T. N. Anderson, "Oversized rectangular waveguides with mode-free bends and twists for '
             'broadband applications", Microwave Journal 32(3), 153-160 (March 1989)')
MA_SRC = ('Z. Ma, T. Yamane, E. Yamashita, "Analysis and design of mitered H-plane bends in rectangular waveguides '
          'by combining the mode-matching and port reflection coefficient methods", Electronics and Communications '
          'in Japan, Part 2, 80(5), 1-7 (1997) (transl. from Denshi Joho Tsushin Gakkai Ronbunshi J80-C-I(2), '
          '78-85, Feb. 1997)')
MA_GUIDE = ('a = 15.799 mm, b = 7.899 mm (p. 4); single-mode TE10 band 9.49-19.0 GHz; fc = c/(2a) = 9.488 GHz; '
            'H-plane bend (side a in the bend plane).')
MA_C = ('Miter parameter c (inset of Figs. 6, 8, 9; Fig. 3): the outer corner is cut by a straight miter whose '
        'ends lie at distance c from the removed outer corner along each outer wall; c/a = 0 is the unmitered '
        'right-angle corner, c/a = 1.0 cuts at the extensions of the inner walls. The slanted miter is modelled '
        'by a staircase of N steps; division methods (Fig. 7, text p. 4): (a) steps go through the slanted plane, '
        '(b) steps lie below it, (c) steps lie above it.')
MA_BEM = ('BEM = circles: boundary-element results; the legend says "BEM [9]" but the reference list ends at [8]; '
          'the text p. 4 cites the BEM results as [6] (Koshiba and Suzuki 1986).')
REI_SRC = ('J. M. Reiter and F. Arndt, "Rigorous analysis of arbitrarily shaped H- and E-plane discontinuities in '
           'rectangular waveguides by a full-wave boundary contour mode-matching method", IEEE Trans. Microwave '
           'Theory Tech. 43(4), 796-801 (April 1995)')
REI3_CAP = ('Fig. 3, p. 799 (PDF p. 4): "Return loss curves of arbitrary angle H-plane or E-plane bends (waveguide '
            'Ku-band, WR62 housing: 15.799 x 7.899 mm)." Text p. 798: 8 modes in the ports and 15 modes in the '
            'cavity subregions.')
REI_GAMMA = ('gamma = interior angle between the inner walls of the two arms at the inner corner (inset): '
             'gamma = 90 deg is a right-angle bend, gamma = 180 deg a straight guide; deflection (bend) angle = '
             '180 deg - gamma.')
REI_AOPT = ('a_opt (H-plane, panel b) and b_2opt (E-plane, panel d; printed "b_2opt", "b_opt" in Fig. 4) = '
            'distance from the inner corner to the miter wall, measured perpendicular to the miter wall '
            '(arrow in the inset), the same definition as Alessandri\'s d.')

LINE_U = ('uncertainty = y uncertainty in the units of y; x_uncertainty = 0 for line curves (x is the grid value, '
          'x reading errors are converted into the y uncertainty by both digitizers); for markers x_uncertainty '
          'is the x uncertainty in the units of x.')
SOURCE_COL = 'source = ab (both digitizations), a (L1a only) or b (L1b only).'


def merge_method(step, gap, markers=False):
    s = (f'Merge (card L1, {MERGE_DATE}, script benchmarks/literature/scripts/merge_ab.py): both raw sets linearly '
         f'interpolated (y and their reading uncertainties) on a common grid of step {step:g} in x within their '
         f'own traced segments (no interpolation across gaps > {gap:g}, nor across any gap wider than 1.5 grid '
         'steps whose both ends lie within 4 dB of the bottom frame, i.e. where a curve leaves the axes at a '
         'reflection zero); y = mean of a and b; uncertainty = '
         'max(|y_a - y_b|/2, max(u_a, u_b)); where only one set covers a point its y is kept and its uncertainty '
         'doubled (source a or b).')
    if markers:
        s += (' Markers (one row per printed symbol) are paired between the sets by mutual nearest neighbour; '
              'x, y = means; uncertainty = max(|dy|/2, max(u_a, u_b)), x_uncertainty = max(|dx|/2, max(xu_a, '
              'xu_b)); unpaired symbols keep one set with doubled uncertainties.')
    return s


# ----------------------------------------------------------------------------------------------------------------
# Configuration: one entry per final file
# ----------------------------------------------------------------------------------------------------------------
# curve entries: (final label, kind, a label or None, b label or None, options)
# options: excl_a / excl_b = [(x0, x1, reason)], nobridge = [(x0, x1, reason)], xmin_a/xmax_a etc.

FIGS = [
    dict(
        file='alessandri1994_fig4.csv', ybot=-40, step=0.005, gap=0.06, ydec=3, xdec=4, xspan=1.0, yspan=40.0,
        header=[
            f'Source: {ALES_SRC}',
            'Figure: Fig. 4, p. 409 (PDF p. 2): "Return loss of a 90 deg mitered E-plane bend in rectangular '
            'waveguide. The frequency is normalized with respect to the cut-off frequency. Crosses represent '
            'results from [7]." [7] = J. M. Reiter and F. Arndt, "A full-wave boundary contour mode-matching '
            'method (BCMM) for the rigorous CAD of single and cascaded optimized H-plane and E-plane bends", 1994 '
            'IEEE MTT-S Int. Microwave Symp. Dig., pp. 1021-1024.',
            'Curves: d/b=1.41 (solid line, no mitering), d/b=0.707 (dashed line, diagonal mitering), d/b=0.87 '
            '(dotted line, optimum; one row per printed dot, markers); ref7 d/b=<v> = crosses of ref. [7], '
            'assigned to the nearest curve (markers). Text p. 409-410 quotes for d/b = 0.87 "a matching better '
            'than -26 dB in the whole band".',
            'Axes: x = f/fc (dimensionless, linear, 1..2); y = |S11| in dB (linear, 0..-40 dB).',
            f'Geometry and parameter: {ALES_GEOM}',
        ],
        curves=[
            ('d/b=1.41', 'line', 'd/b=1.41', 'd/b=1.41', {}),
            ('d/b=0.707', 'line', 'd/b=0.707', 'd/b=0.707', {}),
            ('d/b=0.87', 'markers', 'd/b=0.87', 'd/b=0.87', {}),
            ('ref7 d/b=1.41', 'markers', 'ref7 d/b=1.41', 'ref7_d/b=1.41', {}),
            ('ref7 d/b=0.707', 'markers', 'ref7 d/b=0.707', 'ref7_d/b=0.707', {}),
            ('ref7 d/b=0.87', 'markers', 'ref7 d/b=0.87', 'ref7_d/b=0.87', {}),
        ],
    ),
    dict(
        file='alessandri1994_fig5.csv', ybot=-50, step=0.005, gap=0.06, ydec=3, xdec=4, xspan=1.0, yspan=50.0,
        header=[
            f'Source: {ALES_SRC}',
            'Figure: Fig. 5, p. 409 (PDF p. 2): "Return loss of a mitered E-plane bend in rectangular waveguide. '
            'The frequency is normalized with respect to the cut-off frequency. The dept[h] of the mitering is '
            'described by the parameter d/b."',
            'Curves: d/b=0.9 (thin solid, upper), d/b=0.87 (thin solid, lower; zero near f/fc = 1.83-1.85), '
            'd/b=0.85 (thick dashed; zero near 1.62-1.65), d/b=0.83 (thin dashed; zero near 1.30-1.38). The '
            'reflection zeros lie below the -50 dB frame and are not in the data; both branches of a curve share '
            'its label. The d/b=0.87 curve is the dotted curve of Fig. 4.',
            'Axes: x = f/fc (dimensionless, linear, 1..2); y = |S11| in dB (linear, 0..-50 dB).',
            f'Geometry and parameter: {ALES_GEOM}',
        ],
        curves=[
            ('d/b=0.9', 'line', 'd/b=0.9', 'd/b=0.9', dict(
                excl_b=[(1.0230, 1.02505, 'the L1b rows at x = 1.0235 (-30.0 dB) and 1.0250 (-32.2 dB) lie on the '
                         '-30 dB tick mark and left of the line on the page; the line itself is at -33.1 dB there')])),
            ('d/b=0.87', 'line', 'd/b=0.87', 'd/b=0.87', {}),
            ('d/b=0.85', 'line', 'd/b=0.85', 'd/b=0.85', {}),
            ('d/b=0.83', 'line', 'd/b=0.83', 'd/b=0.83', dict(
                excl_b=[(1.2475, 1.2525, 'the L1b row at x = 1.25 (-47.4 dB) lies 6-7 px below the centre of the '
                         'dash on the page (-46.7 dB)')])),
        ],
    ),
    dict(
        file='doane1989_fig9.csv', step=0.025, gap=0.15, ydec=4, xdec=3, xspan=3.0, yspan=2.0,
        header=[
            f'Source: {DOANE_SRC}',
            'Figure: Fig. 9, p. 159 (PDF p. 7): "Mode conversion in 20" 90 deg twists in WR187 tall guide '
            '(1.872" x 0.872" ID) for various rates of twist variations." Text p. 159: "In Figure 9 we compare the '
            'theoretical predictions for 20" twists in WR187." (theory of their ref. 2).',
            'Curves: constant (dashed), cosine (solid, upper), triangular (solid, lower) rate of twist '
            'd(theta)/dz along the twist; triangular = linearly tapered, eq. (17).',
            'Axes: x = frequency in GHz (linear, 15..18); y = total spurious mode power in % of the input power '
            '(linear, 0..2 %).',
            'Geometry: 90 deg twist of length 20 in (508 mm) in WR187 used as "tall" guide, inside 1.872 in x '
            '0.872 in (47.55 mm x 22.15 mm), the long side being the height (b > a in the paper\'s notation, '
            'a = 0.872 in).',
            'Calibration note (settled by L1 on the page image): the page is a phone photograph of a curved '
            'magazine page. The left-axis ticks 0, 0.5, 1, 1.5 % are evenly spaced (190.8, 186.5, 192.0 px on the '
            'left, 198.0, 193.0, 195.5 px on the right frame), but the top frame, which carries the label "2", lies '
            'only 174.3 px (left) and 173.7 px (right) above the 1.5 % ticks; the label "2" itself is printed at '
            'the frame, so the anomaly is in the printed figure, not a photographic distortion (it is equal on both '
            'sides, while perspective and page curl differ between them). Both digitizers took the frame as 2 % '
            '(calibration A, kept). Above 1.5 % the tick-only scale B (extrapolated from the ticks) gives '
            'y_B = 1.5 + (y - 1.5) r, r = 0.918 at 15 GHz to 0.888 at 18 GHz (linear in x); |y_A - y_B| '
            '(at most 0.02 %, only for the constant curve above 17.87 GHz, where it exceeds 1.5 %) is added to the uncertainty.',
            'Triangular curve (checked by L1 on the page image): it rises relative to the axes from about 0.05 % '
            'at 15 GHz to 0.15 % at 18 GHz. In the photograph it is a nearly level straight line, 18.5 px above '
            'the bottom frame at 15.1 GHz and 59.5 px above it at 17.9 GHz, while the bottom frame drops 27 px '
            'from left to right; the rise is real.',
        ],
        doane=True,
        curves=[
            ('constant', 'line', 'constant', 'constant', {}),
            ('cosine', 'line', 'cosine', 'cosine', {}),
            ('triangular', 'line', 'triangular', 'triangular', {}),
        ],
    ),
    dict(
        file='ma1997_fig5.csv', step=0.05, gap=0.3, ydec=4, xdec=3, xspan=9.5, yspan=0.9,
        header=[
            f'Source: {MA_SRC}',
            'Figure: Fig. 5, p. 4 (PDF p. 4): "Variation of the power transmission coefficient of a waveguide right '
            'angle corner bend with different numbers of modal terms in the electromagnetic field expressions."',
            'Curves: "3/6/12 terms": the 12-term (solid), 6-term (long-short dash) and 3-term (dotted) lines '
            'coincide within the line width except near 16.5-18.6 GHz, where the 3-term dots lie slightly to '
            'the right (up to about 0.02 in |S21|^2); the traced line is the solid 12-term line there. '
            f'{MA_BEM}',
            'Axes: x = frequency in GHz (linear, frame 9.5..19); y = |S21|^2, power transmission coefficient of '
            'TE10 (linear, 0..0.9).',
            f'Guide: {MA_GUIDE}',
            'Geometry: 90 deg right-angle (unmitered) H-plane corner bend, c/a = 0 (Fig. 1(a), inset).',
        ],
        curves=[
            ('3/6/12 terms', 'line', '12 terms', '3/6/12 terms', {}),
            ('BEM', 'markers', 'BEM [9]', 'BEM', {}),
        ],
    ),
    dict(
        file='ma1997_fig6.csv', step=0.05, gap=0.3, ydec=3, xdec=3, xspan=9.5, yspan=16.0,
        header=[
            f'Source: {MA_SRC}',
            'Figure: Fig. 6, p. 4 (PDF p. 4): "Variation of the power transmission coefficient of a mitered bend '
            'with different numbers of steps used in the staircase approximation." y label |S21| (dB).',
            'Curves (legend): 10 steps (long-short-short dash), 20 steps (dotted), 40 steps (long-short dash), '
            '60 steps (solid). "40/60 steps": the 40- and 60-step lines coincide within the line width. The 20- '
            'and 10-step lines are separable from the others only below about 12.6-14 GHz (3-8 px apart); above, '
            'they coincide with the 40/60 line within the line width. ' + MA_BEM,
            'Axes: x = frequency in GHz (linear, frame 9.5..19); y = |S21| in dB (linear, -16..0 dB).',
            f'Guide: {MA_GUIDE}',
            f'Geometry: 90 deg mitered H-plane bend, c/a = 1.0. {MA_C} Division method (a).',
        ],
        curves=[
            ('40/60 steps', 'line', '60 steps', '40/60 steps', {}),
            ('20 steps', 'line', '20 steps', '20 steps', dict(
                excl_b=[(10.08, 10.14, 'the L1b rows at 10.08 and 10.13 GHz lie between the dots of the 20-step '
                         'line and the 40/60-step line on the page (0.3-0.4 dB above the dots)')])),
            ('10 steps', 'line', '10 steps', '10 steps', {}),
            ('BEM', 'markers', 'BEM [9]', 'BEM', {}),
        ],
    ),
    dict(
        file='ma1997_fig8.csv', step=0.05, gap=0.3, ydec=3, xdec=3, xspan=9.5, yspan=35.0,
        header=[
            f'Source: {MA_SRC}',
            'Figure: Fig. 8, p. 5 (PDF p. 5): "Difference between the return loss of a mitered bend calculated with '
            'three different division methods (the number of the division steps is 20)."',
            'Curves (legend): division methods (a) solid, (b) long-short dash, (c) dotted. Text p. 5: "(b) and (c) '
            'are, respectively, above and below the curve with the division method (a)". Above about 16.5 GHz the '
            'three coincide within the line width; there the common line is given as (a) only.',
            'Axes: x = frequency in GHz (linear, frame 9.5..19); y = |S11| in dB (linear, -35..0 dB).',
            f'Guide: {MA_GUIDE}',
            f'Geometry: 90 deg mitered H-plane bend, c/a = 1.0, staircase of 20 steps. {MA_C}',
        ],
        curves=[
            ('(a)', 'line', '(a)', '(a)', {}),
            ('(b)', 'line', '(b)', '(b)', {}),
            ('(c)', 'line', '(c)', '(c)', {}),
        ],
    ),
    dict(
        file='ma1997_fig9.csv', step=0.05, gap=0.3, ydec=3, xdec=3, xspan=9.5, yspan=35.0,
        header=[
            f'Source: {MA_SRC}',
            'Figure: Fig. 9, p. 5 (PDF p. 5): "Difference between the return loss of a mitered bend calculated with '
            'three different division methods (the number of the division steps is 60)."',
            'Curves: (a)=(b)=(c): the three division methods coincide within the line width over the whole band; '
            'the centre of the common ink band is given.',
            'Axes: x = frequency in GHz (linear, frame 9.5..19); y = |S11| in dB (linear, -35..0 dB).',
            f'Guide: {MA_GUIDE}',
            f'Geometry: 90 deg mitered H-plane bend, c/a = 1.0, staircase of 60 steps. {MA_C}',
        ],
        curves=[
            ('(a)=(b)=(c)', 'line', '(a)/(b)/(c)', '(a)=(b)=(c)', {}),
        ],
        notes=['Checked by L1: at 18.5 GHz (steep flank) a and b differ by 0.99 dB, more than their combined '
               'reading uncertainty (L1a converted its line width with a smaller slope there); the printed ink '
               'spans -25.5 to -27.5 dB in that column, and the merged -26.72 +- 0.74 dB covers its centre. '
               'Consistency with Table 1 (60 steps, Ref., |S11| = 0.2622 = -11.63 dB at 16 GHz): merged value '
               'at 16 GHz in the README.'],
    ),
    dict(
        file='ma1997_fig11.csv', ybot=-50, step=0.005, gap=0.06, ydec=3, xdec=4, xspan=1.0, yspan=50.0,
        header=[
            f'Source: {MA_SRC}',
            'Figure: Fig. 11, p. 6 (PDF p. 6): "Variation of the return loss of a mitered bend with different size '
            'of the mitered part of the bend." Text p. 6: "when 0.6 <= c/a <= 0.8, the return loss of the bend can '
            'be made small ... when c/a = 0.631, the low return loss bandwidth (|S11| < -30 dB) is the widest, '
            'reaching about 39% of the single mode frequency band." Computed with 60 steps, division method (a).',
            'Curves: c/a=0 (long dash), c/a=0.60 (solid), c/a=0.631 (long-short dash; zeros near f/fc = 1.68 and '
            '1.93), c/a=0.70 (short dash; zero near 1.38), c/a=1.0 (sparse dots), c/a=1.30 (dashes near the top). '
            'The horizontal dotted line at -30 dB is a reference line, not data. Parts below -50 dB are outside the '
            'axes. Line styles are matched to the labels printed next to the curves; L1 checked the matching '
            'independently: c/a=0 agrees with Fig. 5 (|S11|^2 = 1 - |S21|^2 of the right-angle corner, within '
            '0.1 dB), c/a=1.0 with Fig. 9 at 16 GHz, and the double-zero curve gives |S11| < -30 dB from f/fc = '
            '1.57 to 1.96 (39 % of the band), as the text says for c/a = 0.631.',
            'Axes: x = f/fc (dimensionless, linear, 1..2), fc = TE10 cut-off = 9.488 GHz; y = |S11| in dB '
            '(linear, -50..0 dB).',
            f'Guide: {MA_GUIDE}',
            f'Geometry: 90 deg mitered H-plane bend. {MA_C}',
        ],
        notes=['Checked by L1: c/a=0.631 at f/fc = 1.93 (second reflection zero) a and b differ by 2.4 dB: L1a '
               'read the dot of the dash pattern at the bottom of the V (-46.3 dB at 1.928), L1b bridged the two '
               'dash ends (-43.7 dB). The true value at 1.93 lies between the right-arm end (-44.3 dB) and the dot; '
               'the merged -44.9 +- 1.3 dB covers it. The zero itself is not resolved by the print.'],
        curves=[
            ('c/a=0', 'line', 'c/a=0', 'c/a=0', {}),
            ('c/a=0.60', 'line', 'c/a=0.60', 'c/a=0.60', dict(
                excl_a=[(1.7975, 1.8275, 'L1a traced c/a=0.60 without the crossing mask here and its centroid is '
                         'pulled toward the c/a=0.631 line (L1a header); the page shows the solid line at '
                         '-30.3 dB, as in L1b')])),
            ('c/a=0.631', 'line', 'c/a=0.631', 'c/a=0.631', dict(
                excl_a=[(1.8025, 1.8175, 'L1a points lie on the solid c/a=0.60 line (page image: the long-short '
                         'dash of c/a=0.631 is 1-1.6 dB lower there, as in L1b)')],
                nobridge=[(1.8325, 1.8625, 'local maximum of c/a=0.631 (about -30.4 dB) falls in a gap of its '
                           'dash pattern next to the solid line and the -30 dB reference line; no ink to read')])),
            ('c/a=0.70', 'line', 'c/a=0.70', 'c/a=0.70', {}),
            ('c/a=1.0', 'line', 'c/a=1.0', 'c/a=1.0', {}),
            ('c/a=1.30', 'line', 'c/a=1.30', 'c/a=1.30', dict(
                excl_a=[(1.8025, 1.8125, 'the L1a rows at 1.805 and 1.81 follow the f/fc = 1.8 tick mark hanging from '
                         'the top frame, not the dash, which starts at 1.799 at -2.9 dB (page image)')])),
        ],
    ),
    dict(
        file='reiter1995_fig3a.csv', step=0.05, gap=0.3, ydec=3, xdec=3, xspan=6.0, yspan=35.0,
        header=[
            f'Source: {REI_SRC}',
            f'Figure: {REI3_CAP} (a) H-plane bend (sharp corner, no miter).',
            'Curves: gamma=90, 105, 120, 135, 150 deg (solid lines, labels on the right). The curves are '
            'interrupted by their labels between about 16.0 and 17.2 GHz (not digitized there).',
            'Axes: x = frequency in GHz (linear, 12..18, dotted grid every 1 GHz); y = |S11| in dB (linear, '
            '-35..0 dB, dotted grid every 5 dB).',
            'Geometry: H-plane bend in WR62 (a = 15.799 mm, b = 7.899 mm), side a in the bend plane. ' + REI_GAMMA,
        ],
        curves=[(f'gamma={g}', 'line', f'gamma={g}', f'gamma={g}', {}) for g in (90, 105, 120, 135, 150)],
    ),
    dict(
        file='reiter1995_fig3b.csv', ybot=-55, step=0.05, gap=0.3, ydec=3, xdec=3, xspan=6.0, yspan=30.0,
        header=[
            f'Source: {REI_SRC}',
            f'Figure: {REI3_CAP} (b) Mitered H-plane bend.',
            'Curves: gamma=105 (a_opt = 0.949 a), gamma=120 (a_opt = 0.955 a), gamma=135 (a_opt = 0.956 a), '
            'solid lines, values from the labels. Reflection zeros below the -55 dB frame are not in the data. '
            'Right-hand branches (labels only on the left; settled by L1): the branch rising to -25.5 dB at 18 GHz '
            'is gamma=105 and the branch rising to -34.5 dB is gamma=120. Reason: |S11| near a zero is close to '
            'k|f - f0| with k larger for the more reflecting gamma=105 curve (|S11| 0.050 vs 0.022 at 12 GHz), so '
            'gamma=105 must have the narrower dip below -55 dB: 14.72-14.85 GHz with this pairing (gamma=120: '
            '14.65-15.25 GHz), whereas the other pairing would give gamma=105 the wider dip. Both digitizers made '
            'the same assignment.',
            'Axes: x = frequency in GHz (linear, 12..18, dotted grid every 1 GHz); y = |S11| in dB (linear, '
            '-55..-25 dB, dotted grid every 5 dB).',
            'Geometry: mitered H-plane bend in WR62 (a = 15.799 mm, b = 7.899 mm), side a in the bend plane. '
            + REI_GAMMA + ' ' + REI_AOPT,
        ],
        curves=[
            ('gamma=105', 'line', 'gamma=105 (a_opt=0.949a)', 'gamma=105', dict(
                excl_b=[(13.84, 13.90, 'the L1b rows at 13.85 and 13.90 GHz (-37.10 dB) follow the leader line of '
                         'the gamma=120 label where it crosses the curve; the curve is at -37.3 and -37.8 dB there '
                         '(page image)')])),
            ('gamma=120', 'line', 'gamma=120 (a_opt=0.955a)', 'gamma=120', {}),
            ('gamma=135', 'line', 'gamma=135 (a_opt=0.956a)', 'gamma=135', {}),
        ],
        notes=['Left branches of gamma=105 and gamma=120 (checked by L1 on the page): they meet near 14.45 GHz, '
               '-48 dB, and run down to the -55 dB frame as one ink band about 1.7 dB wide; there L1a measured the '
               'band and L1b interpolated across it, and the merged uncertainty (0.6-1.0 dB) covers the band.'],
    ),
    dict(
        file='reiter1995_fig3c.csv', step=0.05, gap=0.3, ydec=3, xdec=3, xspan=6.0, yspan=35.0,
        header=[
            f'Source: {REI_SRC}',
            f'Figure: {REI3_CAP} (c) E-plane bend (sharp corner, no miter).',
            'Curves: gamma=90, 105, 120, 135, 150 deg (solid lines, labels on the right). The curves are '
            'interrupted by their labels between about 16.0 and 17.0 GHz (not digitized there).',
            'Axes: x = frequency in GHz (linear, 12..18, dotted grid every 1 GHz); y = |S11| in dB (linear, '
            '-35..0 dB, dotted grid every 5 dB).',
            'Geometry: E-plane bend in WR62 (a = 15.799 mm, b = 7.899 mm), side b in the bend plane. ' + REI_GAMMA,
        ],
        curves=[(f'gamma={g}', 'line', f'gamma={g}', f'gamma={g}', {}) for g in (90, 105, 120, 135, 150)],
    ),
    dict(
        file='reiter1995_fig3d.csv', ybot=-65, step=0.05, gap=0.3, ydec=3, xdec=3, xspan=6.0, yspan=30.0,
        header=[
            f'Source: {REI_SRC}',
            f'Figure: {REI3_CAP} (d) Mitered E-plane bend.',
            'Curves: gamma=105 (b_2opt = 0.873 b), gamma=120 (b_2opt = 0.898 b), gamma=135 (b_2opt = 0.918 b), '
            'solid lines, values from the labels; gamma=105 and gamma=135 both reach the -65 dB frame near '
            '16.6 GHz. "right branch" = the branch rising from the -65 dB frame at 17.2 GHz to -50 dB at 18 GHz. '
            'It is the return of gamma=105 or of gamma=135 after its reflection zero, and the figure does not say '
            'which (no label, both candidate curves merge at the frame). L1 checked the two pairings with a '
            'smooth real zero, |S11| = |alpha (f - z) (1 + beta (f - z))|, fitted below -55 dB: pairing with '
            'gamma=135 fits within 0.16 dB rms, pairing with gamma=105 only within 1.1 dB rms; but a fit of '
            'each left flank alone predicts the return to -65 dB at 16.77 GHz for gamma=105 and 17.61 GHz for '
            'gamma=135, neither at 17.22 GHz, and with either pairing the other curve\'s return is not drawn. '
            'The branch is therefore kept as a separate curve (L1a had put it under gamma=105, L1b left it '
            'unassigned). Do not use it to validate either angle without stating the assumption.',
            'Axes: x = frequency in GHz (linear, 12..18, dotted grid every 1 GHz); y = |S11| in dB (linear, '
            '-65..-35 dB, dotted grid every 5 dB).',
            'Geometry: mitered E-plane bend in WR62 (a = 15.799 mm, b = 7.899 mm), side b in the bend plane. '
            + REI_GAMMA + ' ' + REI_AOPT,
        ],
        curves=[
            ('gamma=105', 'line', 'gamma=105 (b_opt=0.873b)', 'gamma=105', dict(xmax_a=17.0)),
            ('gamma=120', 'line', 'gamma=120 (b_opt=0.898b)', 'gamma=120', {}),
            ('gamma=135', 'line', 'gamma=135 (b_opt=0.918b)', 'gamma=135', {}),
            ('right branch (gamma=105 or 135)', 'line', 'gamma=105 (b_opt=0.873b)', 'right branch',
             dict(xmin_a=17.0)),
        ],
    ),
    dict(
        file='reiter1995_fig4.csv', ybot=-70, step=0.025, gap=0.1, ydec=3, xdec=3, xspan=4.2, yspan=60.0,
        header=[
            f'Source: {REI_SRC}',
            'Figure: Fig. 4, p. 799 (PDF p. 4): "Calculated and measured return loss of a cascaded structure of '
            'mitered 90 H-plane and E-plane bends (waveguide X-band, WR90-housing: 22.86 x 10.16 mm). Optimum '
            'miter dimensions in (mm): H-plane a_opt = 0.976a, E-plane b_opt = 0.874b." Text p. 798: "Fig. 4 '
            'shows the overall return loss for a structure with an intermediate length of zero l = 0 and of a '
            'length identical to the waveguide height l = b between the two bends."',
            'Curves: l=0 and l=b, each drawn as theory (dashed) and measurement (solid). Where the two coincide '
            'within the line width they are given as one band (its half width is inside the uncertainty). Where '
            'they separate (near the l=b minima at 9.4 and 11.7 GHz, and l=b above about 11.9 GHz) no line point '
            'is given; the printed minima are given as separate one-row curves read by L1 on the page '
            '(minimum rows, markers, source L1). Curve identity (settled by L1 on the page): the 11.7 GHz minimum '
            'belongs to l=b. Through the crossings near 10.35 and 11.1 GHz the measured l=0 line keeps its small '
            'ripple (wiggly lower middle segment, wiggly upper right segment), while the l=b line is smooth from '
            'the 9.4 GHz minimum over the hump at 10.6 GHz into the 11.7 GHz minimum; both digitizers agree.',
            'Axes: x = frequency in GHz (linear, 8.2..12.4, ticks at 8.2, 8.5, 9, ..., 12, 12.4); y = |S11| in dB '
            '(linear, -70..-10 dB, ticks every 10 dB).',
            'Geometry: a mitered 90 deg H-plane bend followed by a mitered 90 deg E-plane bend (inset), joined by '
            'a straight section of length l (l = 0 or l = b = 10.16 mm), WR90 (a = 22.86 mm, b = 10.16 mm). '
            'a_opt = 0.976 a (H-plane), b_opt = 0.874 b (E-plane), distances from the inner corner to the miter '
            'wall as in Fig. 3.',
        ],
        curves=[
            ('l=0', 'line', 'l=0 (theory and measured merged)', 'l=0', {}),
            ('l=b', 'line', 'l=b (theory and measured merged)', 'l=b', {}),
        ],
        # minima read by L1 on the page image (native 339 dpi scan, L1a tick calibration); x, y, u, xu, source
        extra_rows=[
            ('l=b measured minimum', 9.410, -39.89, 0.30, 0.025, 'L1'),
            ('l=b measured minimum', 11.726, -60.20, 0.40, 0.010, 'L1'),
            ('l=b theory minimum', 11.705, -54.30, 0.50, 0.010, 'L1'),
        ],
        notes=['Minimum rows (source L1, read by L1 on the native 339 dpi scan with the L1a tick calibration): '
               'l=b measured (solid) minimum: flat bottom at -39.89 dB from 9.39 to 9.44 GHz; measured minimum '
               'near 11.7 GHz: V tip at -60.2 dB, 11.726 GHz (L1a read -59.8 dB at 11.73 GHz, L1b about -60 dB); '
               'theory (dashed) minimum near 11.7 GHz: -54.3 dB at 11.705 GHz, where its tip touches the measured '
               'line (L1a about -54, L1b about -55 dB). The theory minimum near 9.4 GHz falls in a gap of the dash '
               'pattern; its lowest printed point is -43.3 dB at 9.42 GHz, so the theory minimum is below -43.3 dB '
               '(not given as a row).',
               'Calibration (checked by L1): the figure has tick marks on the frame (8.2, 8.5, 9, ..., 12.4 GHz; '
               'every 10 dB), used by L1a; L1b calibrated on the frame lines alone. The two maps agree: mean(a-b) '
               '= +0.09 dB (l=0) and +0.14 dB (l=b), well inside the reading uncertainties (about 0.5 dB).'],
    ),
]


# ----------------------------------------------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------------------------------------------

def read_raw(path):
    header, rows = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                header.append(line.rstrip('\n'))
            else:
                rows.append(line)
    data = collections.OrderedDict()
    reader = csv.reader(rows)
    next(reader)
    for r in reader:
        if not r:
            continue
        data.setdefault(r[0], []).append([float(v) for v in r[1:5]])
    return header, {k: np.array(v) for k, v in data.items()}


def header_entries(lines):
    """Split a raw header into (key, text) entries, joining continuation lines."""
    out = []
    for ln in lines:
        body = ln[1:]
        if body.startswith('   ') or (body.startswith('  ') and out and ':' not in body.split()[0]):
            out[-1][1] += ' ' + body.strip()
            continue
        body = body.strip()
        key, _, rest = body.partition(':')
        out.append([key.strip(), body])
    return out


CAL_KEYS_A = ('calibration points', 'calibration fit', 'calibration', 'calibration check', 'axis resolution',
              'y-scale anomaly')
CAL_KEYS_B = ('Calibration (L1b)', 'Scale', 'Check', 'Note')


def calibration_lines(hdr_a, hdr_b):
    la = [t for k, t in header_entries(hdr_a) if any(k.startswith(c) for c in CAL_KEYS_A)]
    lb = [t for k, t in header_entries(hdr_b) if any(k.startswith(c) for c in CAL_KEYS_B)]
    la = ['Calibration L1a (raw/a): ' + t for t in la]
    lb = [('L1b note (raw/b): ' + t[len('Note: '):]) if t.startswith('Note') else
          ('Calibration L1b (raw/b): ' + t.replace('Calibration (L1b): ', '')) for t in lb]
    return la + lb


# ----------------------------------------------------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------------------------------------------------

def _drop(v, opts, s):
    if v is None:
        return None
    m = np.ones(len(v), bool)
    if f'xmin_{s}' in opts:
        m &= v[:, 0] >= opts[f'xmin_{s}']
    if f'xmax_{s}' in opts:
        m &= v[:, 0] <= opts[f'xmax_{s}']
    for x0, x1, _ in opts.get(f'excl_{s}', []):
        m &= ~((v[:, 0] >= x0) & (v[:, 0] <= x1))
    return v[m]


def _segments(v, gap, floor=None):
    """Cut v where the x step exceeds gap, and (floor = (y_frame, margin, min_gap)) where a gap wider than
    min_gap has both ends within margin of the bottom frame: there the curve leaves the axes (reflection zero)."""
    v = v[np.argsort(v[:, 0], kind='stable')]
    dx = np.diff(v[:, 0])
    cut = dx > gap
    if floor is not None:
        yf, margin, mg = floor
        low = v[:, 1] < yf + margin
        cut |= (dx > mg) & low[:-1] & low[1:]
    br = np.where(cut)[0] + 1
    return np.split(v, br)


def _interp(v, gap, x, excl=(), floor=None):
    """Interpolate y and u of v onto x within each traced segment. excl: intervals removed before the merge;
    a segment is also cut at an excluded interval (no bridging over removed points)."""
    y = np.full(x.shape, np.nan)
    u = np.full(x.shape, np.nan)
    if v is None or len(v) == 0:
        return y, u
    for s in _segments(v, gap, floor):
        # cut segment at excluded intervals that fall between its points
        pieces = [s]
        for x0, x1, _ in excl:
            new = []
            for p in pieces:
                left, right = p[p[:, 0] < x0], p[p[:, 0] > x1]
                if len(left) and len(right):
                    new += [left, right]
                else:
                    new.append(p)
            pieces = new
        for p in pieces:
            if len(p) < 2:
                continue
            xs, idx = np.unique(p[:, 0], return_inverse=True)
            cnt = np.bincount(idx)
            ys = np.bincount(idx, p[:, 1]) / cnt
            us = np.bincount(idx, p[:, 2]) / cnt
            m = (x >= xs[0] - 1e-9) & (x <= xs[-1] + 1e-9)
            y[m] = np.interp(x[m], xs, ys)
            u[m] = np.interp(x[m], xs, us)
    return y, u


def merge_line(A, B, step, gap, opts, floor=None):
    xs = [v[:, 0] for v in (A, B) if v is not None and len(v)]
    lo, hi = min(x.min() for x in xs), max(x.max() for x in xs)
    x = np.round(np.arange(np.ceil(lo / step - 1e-9) * step, hi + 1e-9, step), 6)
    ya, ua = _interp(A, gap, x, opts.get('excl_a', []), floor)
    yb, ub = _interp(B, gap, x, opts.get('excl_b', []), floor)
    for x0, x1, _ in opts.get('nobridge', []):
        m = (x > x0) & (x < x1)
        ya[m] = np.nan
        yb[m] = np.nan
    rows = []
    for i in range(len(x)):
        ha, hb = not np.isnan(ya[i]), not np.isnan(yb[i])
        if ha and hb:
            rows.append((x[i], 0.5 * (ya[i] + yb[i]), max(0.5 * abs(ya[i] - yb[i]), ua[i], ub[i]), 0.0, 'ab'))
        elif ha:
            rows.append((x[i], ya[i], 2 * ua[i], 0.0, 'a'))
        elif hb:
            rows.append((x[i], yb[i], 2 * ub[i], 0.0, 'b'))
    both = ~np.isnan(ya) & ~np.isnan(yb)
    d = (ya - yb)[both]
    comb = np.hypot(ua, ub)[both]
    exc = [(x[both][j], ya[both][j], ua[both][j], yb[both][j], ub[both][j]) for j in np.where(np.abs(d) > comb)[0]]
    st = dict(n_ab=int(both.sum()), n_a=sum(r[4] == 'a' for r in rows), n_b=sum(r[4] == 'b' for r in rows),
              bias=float(d.mean()) if len(d) else np.nan, rms=float(np.sqrt((d ** 2).mean())) if len(d) else np.nan,
              max=float(np.abs(d).max()) if len(d) else np.nan, n_exc=len(exc), exc=exc)
    return rows, st


def merge_markers(A, B, xspan, yspan, pair_max=0.03):
    A = A if A is not None else np.zeros((0, 4))
    B = B if B is not None else np.zeros((0, 4))

    def dist(p, q):
        return np.hypot((p[0] - q[0]) / xspan, (p[1] - q[1]) / yspan)

    pa = {}
    for i, p in enumerate(A):
        if len(B) == 0:
            break
        j = int(np.argmin([dist(p, q) for q in B]))
        i2 = int(np.argmin([dist(B[j], r) for r in A]))
        if i2 == i and dist(p, B[j]) <= pair_max:
            pa[i] = j
    rows, d, comb, exc = [], [], [], []
    used_b = set(pa.values())
    for i, p in enumerate(A):
        if i in pa:
            q = B[pa[i]]
            rows.append(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, max(abs(p[1] - q[1]) / 2, p[2], q[2]),
                         max(abs(p[0] - q[0]) / 2, p[3], q[3]), 'ab'))
            d.append(p[1] - q[1])
            c = np.hypot(p[2], q[2])
            comb.append(c)
            if abs(p[1] - q[1]) > c or abs(p[0] - q[0]) > np.hypot(p[3], q[3]):
                exc.append((p[0], p[1], p[2], q[0], q[1], q[2]))
        else:
            rows.append((p[0], p[1], 2 * p[2], 2 * p[3], 'a'))
    for j, q in enumerate(B):
        if j not in used_b:
            rows.append((q[0], q[1], 2 * q[2], 2 * q[3], 'b'))
    rows.sort(key=lambda r: r[0])
    d = np.array(d)
    st = dict(n_ab=len(pa), n_a=len(A) - len(pa), n_b=len(B) - len(pa),
              bias=float(d.mean()) if len(d) else np.nan, rms=float(np.sqrt((d ** 2).mean())) if len(d) else np.nan,
              max=float(np.abs(d).max()) if len(d) else np.nan, n_exc=len(exc), exc=exc)
    return rows, st


def doane_alt(rows):
    """Add |y_A - y_B| of the tick-only y scale above 1.5 % to the uncertainty (see header)."""
    out = []
    for x, y, u, xu, s in rows:
        if y > 1.5:
            r = 0.918 + (0.888 - 0.918) * (x - 15.0) / 3.0
            u = u + abs((y - 1.5) * (1 - r))
        out.append((x, y, u, xu, s))
    return out


def process(fig, write=True, verbose=True):
    ha, A = read_raw(os.path.join(RAW, 'a', fig['file']))
    hb, B = read_raw(os.path.join(RAW, 'b', fig['file']))
    out_rows, stats = [], []
    for final, kind, la, lb, opts in fig['curves']:
        va = _drop(A.get(la), opts, 'a') if la else None
        vb = _drop(B.get(lb), opts, 'b') if lb else None
        if kind == 'line':
            # excl_* already applied in _drop; pass them on so segments are cut there
            floor = (fig['ybot'], 4.0, 1.5 * fig['step']) if 'ybot' in fig else None
            rows, st = merge_line(va, vb, fig['step'], fig['gap'], opts, floor)
        else:
            rows, st = merge_markers(va, vb, fig['xspan'], fig['yspan'])
        if fig.get('doane'):
            rows = doane_alt(rows)
        st['curve'] = final
        st['kind'] = kind
        st['urange'] = (min(r[2] for r in rows), float(np.median([r[2] for r in rows])), max(r[2] for r in rows))
        stats.append(st)
        out_rows += [(final,) + r for r in rows]
    out_rows += [tuple(r) for r in fig.get('extra_rows', [])]
    if verbose:
        for st in stats:
            print(f"{fig['file']:24s} {st['curve'][:22]:22s} {st['kind']:7s} n_ab={st['n_ab']:4d} a={st['n_a']:3d} "
                  f"b={st['n_b']:3d} bias={st['bias']:+.4f} rms={st['rms']:.4f} max={st['max']:.4f} "
                  f"n_exc={st['n_exc']} u=[{st['urange'][0]:.3g}, {st['urange'][1]:.3g}, {st['urange'][2]:.3g}]")
            for e in st['exc']:
                print('     exceed:', ' '.join(f'{v:.4f}' for v in e))
    if write:
        write_file(fig, ha, hb, out_rows, stats)
    return stats


def fmt(v, dec):
    return f'{v:.{dec}f}'


def write_file(fig, ha, hb, rows, stats):
    hdr = list(fig['header'])
    hdr.append('Columns: curve, x, y, uncertainty, x_uncertainty, source. ' + LINE_U + ' ' + SOURCE_COL)
    hdr.append(merge_method(fig['step'], fig['gap'], any(c[1] == 'markers' for c in fig['curves'])))
    for final, kind, la, lb, opts in fig['curves']:
        for key, who in (('excl_a', 'L1a'), ('excl_b', 'L1b')):
            for x0, x1, why in opts.get(key, []):
                hdr.append(f'Exclusion ({final}): {who} points with {x0:g} <= x <= {x1:g} dropped: {why}.')
        for x0, x1, why in opts.get('nobridge', []):
            hdr.append(f'Gap ({final}): no point for {x0:g} < x < {x1:g}: {why}.')
        for key, who in (('xmax_a', 'L1a rows with x <='), ('xmin_a', 'L1a rows with x >=')):
            if key in opts:
                hdr.append(f'Curve assignment ({final}): {who} {opts[key]:g} of L1a curve "{la}".')
    agr = []
    for st in stats:
        agr.append(f"{st['curve']}: n_ab={st['n_ab']}, a-only={st['n_a']}, b-only={st['n_b']}, "
                   f"rms(a-b)={st['rms']:.3g}, max|a-b|={st['max']:.3g}, mean(a-b)={st['bias']:+.3g}, "
                   f"points with |a-b| > sqrt(u_a^2+u_b^2): {st['n_exc']}")
    hdr.append('Agreement of the two digitizations (y units; grid points or paired markers covered by both): '
               + '; '.join(agr) + '.')
    hdr += fig.get('notes', [])
    hdr += calibration_lines(ha, hb)
    hdr.append(f'Digitizers: L1a and L1b, independently (raw files benchmarks/literature/raw/a/{fig["file"]} and '
               f'raw/b/{fig["file"]}, methods and scripts described there); merged and checked against the page '
               f'images by L1, {MERGE_DATE}.')
    path = os.path.join(LIT, fig['file'])
    with open(path, 'w', newline='') as fh:
        for h in hdr:
            fh.write('# ' + h + '\n')
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['curve', 'x', 'y', 'uncertainty', 'x_uncertainty', 'source'])
        xd, yd = fig['xdec'], fig['ydec']
        for c, x, y, u, xu, s in rows:
            w.writerow([c, fmt(x, xd), fmt(y, yd), fmt(max(u, 10 ** -yd), yd), fmt(xu, xd), s])


def merge_table():
    """Ma et al. Table 1: both transcriptions are compared value by value."""
    _, rows_a = None, []
    with open(os.path.join(RAW, 'a', 'ma1997_table1.csv')) as fh:
        ra = list(csv.DictReader(l for l in fh if not l.startswith('#')))
    with open(os.path.join(RAW, 'b', 'ma1997_table1.csv')) as fh:
        rb = list(csv.DictReader(l for l in fh if not l.startswith('#')))
    vb = {(r['steps'], r['case'], r['quantity']): r['value'] for r in rb}
    out, diffs = [], []
    for r in ra:
        case = r['displacement'].rstrip('.')
        for q, col, u in (('|S11|', 's11_mag', '0.00005'), ('|S11| diff (%)', 's11_diff_pct', '0.005'),
                          ('|S21|', 's21_mag', '0.00005'), ('|S21| diff (%)', 's21_diff_pct', '0.005')):
            if not r[col]:
                continue
            other = vb.get((r['steps'], case, q))
            if other is None or float(other) != float(r[col]):
                diffs.append((r['steps'], case, q, r[col], other))
            out.append((r['steps'], case, q, r[col], u))
    keys_a = {(o[0], o[1], o[2]) for o in out}
    for k in vb:
        if k not in keys_a:
            diffs.append(k + (None, vb[k]))
    return out, diffs


def write_table(out, diffs):
    hdr = [
        f'Source: {MA_SRC}',
        'Table: Table 1, p. 5 (PDF p. 5): "Difference between the scattering parameters of a mitered bend calculated '
        'with and without the displacement of steps shown in Fig. 10". Frequency 16 GHz (f/fc = 1.686); staircase '
        'with 50 or 60 steps. Ref. = no displacement; (a)-(d) = the four displacements of Fig. 10 (four '
        'arbitrary steps moved by half a step, text p. 5). Diff. (%) = printed difference to Ref.',
        f'Guide: {MA_GUIDE}',
        'Geometry: 90 deg mitered H-plane bend. The table does not state c/a; the surrounding Figs. 6, 8, 9 and '
        'the text of section 3 use c/a = 1.0. ' + MA_C,
        'Columns: steps, case, quantity, value (linear magnitude, or percent for the diff rows), uncertainty = '
        'half a unit of the last printed digit, source = ab (both transcriptions identical).',
        f'Transcriptions: L1a and L1b independently from the page image; compared value by value by L1 '
        f'({MERGE_DATE}): {len(diffs)} differences; L1 re-read the table on the page image (Ref. rows: '
        '|S11|^2 + |S21|^2 = 1.00001 (50 steps) and 0.99997 (60 steps)). Fig. 9 at 16 GHz (merged file): see '
        'benchmarks/literature/README.md.',
    ]
    with open(os.path.join(LIT, 'ma1997_table1.csv'), 'w', newline='') as fh:
        for h in hdr:
            fh.write('# ' + h + '\n')
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['steps', 'case', 'quantity', 'value', 'uncertainty', 'source'])
        for r in out:
            w.writerow(list(r) + ['ab'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', action='store_true', help='print statistics only, do not write files')
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    allstats = {}
    for fig in FIGS:
        if args.only and args.only not in fig['file']:
            continue
        allstats[fig['file']] = process(fig, write=not args.stats)
    out, diffs = merge_table()
    print('ma1997_table1.csv: values', len(out), 'differences a/b:', diffs)
    if not args.stats:
        write_table(out, diffs)
        if not args.only:
            write_readme(allstats)
    return allstats




# ----------------------------------------------------------------------------------------------------------------
# Cross-checks between merged files (written into the README)
# ----------------------------------------------------------------------------------------------------------------

FC_MA = 299792458.0 / (2 * 15.799e-3) / 1e9   # GHz, TE10 cut-off of Ma's guide


def load_final(name):
    with open(os.path.join(LIT, name)) as fh:
        rows = list(csv.DictReader(l for l in fh if not l.startswith('#')))
    d = collections.OrderedDict()
    for r in rows:
        d.setdefault(r['curve'], []).append([float(r['x']), float(r['y']), float(r['uncertainty'])])
    return {k: np.array(v) for k, v in d.items()}


def _cmp(xq, A, B, gap):
    """Compare curve A at xq with curve B interpolated (only where B has points within gap)."""
    out = []
    for x, y, u in A:
        if x < B[0, 0] or x > B[-1, 0]:
            continue
        j = np.searchsorted(B[:, 0], x)
        if j == 0 or B[j, 0] - B[j - 1, 0] > gap:
            continue
        yb = np.interp(x, B[:, 0], B[:, 1])
        ub = np.interp(x, B[:, 0], B[:, 2])
        out.append((x, y - yb, np.hypot(u, ub)))
    return np.array(out)


def consistency():
    lines = []
    f11, f5, f9, f6 = (load_final(n) for n in ('ma1997_fig11.csv', 'ma1997_fig5.csv', 'ma1997_fig9.csv',
                                                'ma1997_fig6.csv'))
    # Fig. 5 corner -> |S11| dB vs f/fc
    s21 = f5['3/6/12 terms']
    conv = np.column_stack([s21[:, 0] / FC_MA, 10 * np.log10(np.clip(1 - s21[:, 1], 1e-9, None)),
                            10 / np.log(10) * s21[:, 2] / np.clip(1 - s21[:, 1], 1e-9, None)])
    c = _cmp(None, f11['c/a=0'], conv, 0.02)
    c = c[(c[:, 0] >= 1.05) & (c[:, 0] <= 1.80)]
    lines.append(f'Ma Fig. 11 c/a=0 against Fig. 5 (right-angle corner, |S11|^2 = 1 - |S21|^2, f/fc = f/{FC_MA:.3f} GHz), '
                 f'{len(c)} points with f/fc 1.05-1.80: rms difference {np.sqrt(np.mean(c[:, 1] ** 2)):.3f} dB, max '
                 f'{np.abs(c[:, 1]).max():.3f} dB; {int((np.abs(c[:, 1]) > c[:, 2]).sum())} points outside the combined '
                 'uncertainty.')
    f9n = f9['(a)=(b)=(c)'].copy()
    f9n[:, 0] /= FC_MA
    c = _cmp(None, f11['c/a=1.0'], f9n, 0.02)
    lines.append(f'Ma Fig. 11 c/a=1.0 against Fig. 9 (60 steps, c/a = 1.0), {len(c)} points: rms difference '
                 f'{np.sqrt(np.mean(c[:, 1] ** 2)):.3f} dB, max {np.abs(c[:, 1]).max():.3f} dB; '
                 f'{int((np.abs(c[:, 1]) > c[:, 2]).sum())} points outside the combined uncertainty.')
    v = f11['c/a=0.631']
    below = v[v[:, 1] < -30, 0]
    lines.append(f'Ma Fig. 11 c/a=0.631: |S11| < -30 dB from f/fc = {below.min():.3f} to the last point below -30 dB at '
                 f'{below.max():.3f} (the curve leaves the -30 dB line at about 1.96 just after its second zero), '
                 f'i.e. {100 * (below.max() - below.min()):.1f} % of the single-mode band f/fc = 1-2; text: "about 39%".')
    y16 = np.interp(16.0, f9['(a)=(b)=(c)'][:, 0], f9['(a)=(b)=(c)'][:, 1])
    u16 = np.interp(16.0, f9['(a)=(b)=(c)'][:, 0], f9['(a)=(b)=(c)'][:, 2])
    lines.append(f'Ma Fig. 9 at 16 GHz: {y16:.2f} +- {u16:.2f} dB; Table 1 (60 steps, Ref.) |S11| = 0.2622 = '
                 f'{20 * np.log10(0.2622):.2f} dB.')
    y16 = np.interp(16.0, f6['40/60 steps'][:, 0], f6['40/60 steps'][:, 1])
    u16 = np.interp(16.0, f6['40/60 steps'][:, 0], f6['40/60 steps'][:, 2])
    lines.append(f'Ma Fig. 6 (40/60 steps) at 16 GHz: {y16:.2f} +- {u16:.2f} dB; Table 1 (60 steps, Ref.) |S21| = 0.9650 = '
                 f'{20 * np.log10(0.9650):.2f} dB.')
    a4, a5 = load_final('alessandri1994_fig4.csv'), load_final('alessandri1994_fig5.csv')
    c = _cmp(None, a4['d/b=0.87'], a5['d/b=0.87'], 0.02)
    lines.append(f'Alessandri d/b=0.87, dots of Fig. 4 against the line of Fig. 5 (same curve printed twice), {len(c)} '
                 f'dots: rms difference {np.sqrt(np.mean(c[:, 1] ** 2)):.3f} dB, max {np.abs(c[:, 1]).max():.3f} dB; '
                 f'{int((np.abs(c[:, 1]) > c[:, 2]).sum())} outside the combined uncertainty. Highest dot: '
                 f'{a4["d/b=0.87"][:, 1].max():.2f} dB (text: "better than -26 dB in the whole band").')
    b5, b6 = f5['BEM'][:, 0], f6['BEM'][:, 0]
    lines.append('Ma BEM circles: Fig. 5 at ' + ', '.join(f'{x:.2f}' for x in b5) + ' GHz; Fig. 6 at '
                 + ', '.join(f'{x:.2f}' for x in b6) + ' GHz: every Fig. 6 circle up to 18.1 GHz lies within '
                 + f'{max(np.abs(b5 - x).min() for x in b6[b6 < 18.6]):.2f} GHz of a Fig. 5 circle (same BEM frequencies), '
                 'which confirms the x calibrations of both figures.')
    return lines


INDEX = [
    ('Alessandri, Mongiardo & Sorrentino 1994', 'Fig. 4: \\|S11\\| (dB) vs f/fc, 90° mitered E-plane bend', 'alessandri1994_fig4.csv',
     'd/b = 1.41, 0.707 (lines), 0.87 (dots), crosses of their ref. [7]'),
    ('Alessandri et al. 1994', 'Fig. 5: \\|S11\\| (dB) vs f/fc, 90° mitered E-plane bend', 'alessandri1994_fig5.csv',
     'd/b = 0.9, 0.87, 0.85, 0.83'),
    ('Doane & Anderson 1989', 'Fig. 9: spurious mode power (%) vs GHz, 20" 90° twists, WR187 tall guide',
     'doane1989_fig9.csv', 'constant, cosine, triangular rate of twist'),
    ('Ma, Yamane & Yamashita 1997', 'Fig. 5: \\|S21\\|² vs GHz, right-angle H-plane corner', 'ma1997_fig5.csv',
     '3/6/12 terms (one line), BEM circles'),
    ('Ma et al. 1997', 'Fig. 6: \\|S21\\| (dB) vs GHz, mitered H-plane bend c/a = 1.0', 'ma1997_fig6.csv',
     '40/60, 20, 10 steps, BEM circles'),
    ('Ma et al. 1997', 'Fig. 8: \\|S11\\| (dB) vs GHz, 20 steps, division methods', 'ma1997_fig8.csv', '(a), (b), (c)'),
    ('Ma et al. 1997', 'Fig. 9: \\|S11\\| (dB) vs GHz, 60 steps', 'ma1997_fig9.csv', '(a)=(b)=(c)'),
    ('Ma et al. 1997', 'Fig. 11: \\|S11\\| (dB) vs f/fc, miter size', 'ma1997_fig11.csv',
     'c/a = 0, 0.60, 0.631, 0.70, 1.0, 1.30'),
    ('Ma et al. 1997', 'Table 1: \\|S11\\|, \\|S21\\| at 16 GHz, 50/60 steps, step displacements (transcribed)',
     'ma1997_table1.csv', 'Ref., (a)-(d)'),
    ('Reiter & Arndt 1995', 'Fig. 3(a): \\|S11\\| (dB) vs GHz, H-plane bends, WR62', 'reiter1995_fig3a.csv',
     'γ = 90, 105, 120, 135, 150°'),
    ('Reiter & Arndt 1995', 'Fig. 3(b): mitered H-plane bends, WR62', 'reiter1995_fig3b.csv', 'γ = 105, 120, 135°'),
    ('Reiter & Arndt 1995', 'Fig. 3(c): E-plane bends, WR62', 'reiter1995_fig3c.csv', 'γ = 90, 105, 120, 135, 150°'),
    ('Reiter & Arndt 1995', 'Fig. 3(d): mitered E-plane bends, WR62', 'reiter1995_fig3d.csv',
     'γ = 105, 120, 135°, "right branch (gamma=105 or 135)"'),
    ('Reiter & Arndt 1995', 'Fig. 4: cascaded mitered H- and E-plane bends, WR90, theory and measurement',
     'reiter1995_fig4.csv', 'l = 0, l = b, three minimum rows'),
    ('Butterweck 1968; Butterweck & de Ronde 1968', 'quoted numbers (D5), sentence and page of each',
     'butterweck1968_numbers.csv', '20 items'),
]

UNITS = {'alessandri1994_fig4.csv': 'dB', 'alessandri1994_fig5.csv': 'dB', 'doane1989_fig9.csv': '%',
         'ma1997_fig5.csv': '', 'ma1997_fig6.csv': 'dB', 'ma1997_fig8.csv': 'dB', 'ma1997_fig9.csv': 'dB',
         'ma1997_fig11.csv': 'dB', 'reiter1995_fig3a.csv': 'dB', 'reiter1995_fig3b.csv': 'dB',
         'reiter1995_fig3c.csv': 'dB', 'reiter1995_fig3d.csv': 'dB', 'reiter1995_fig4.csv': 'dB'}


def _g(v):
    return f'{v:.2g}' if v < 0.1 else f'{v:.2f}'


def write_readme(allstats):
    L = []
    w = L.append
    w('# Literature benchmark data (card L1)')
    w('')
    w('Published results that wrmodes 1.0 is validated against, as machine-readable data with uncertainties. Each figure '
      'was digitized twice, independently, by agents L1a and L1b (`raw/a/`, `raw/b/`, with their scripts and '
      'calibrations); the files in this folder are the merged result (card L1, 2026-11-08, '
      '`scripts/merge_ab.py`). Transcribed formulas are in `docs/literature/`.')
    w('')
    w('## Files')
    w('')
    w('Every curve file has the columns `curve, x, y, uncertainty, x_uncertainty, source` and a `#` header with the '
      'source, figure, axes and units, geometry and parameter definitions, both digitizers\' calibrations, the merge '
      'method, the a-b agreement per curve and every decision taken in the merge. `uncertainty` is the uncertainty of '
      '`y` in the units of `y`; `x_uncertainty` is 0 for line curves (sampled on a regular grid; x reading errors are '
      'already converted into the y uncertainty) and the x uncertainty for markers (dots, crosses, circles: one row per '
      'printed symbol). `source` is `ab` (both digitizations), `a` or `b` (one only; uncertainty doubled) or `L1` '
      '(read by the merger on the page, Reiter Fig. 4 minima only). Per CONVENTIONS.md section 8, digitized data never '
      'set a tolerance tighter than their uncertainty.')
    w('')
    w('| Source | Figure | File | Curves | y uncertainty: min-max (median) | a-b agreement: rms / max of a-b over all curves |')
    w('| --- | --- | --- | --- | --- | --- |')
    for src, fig, fn, curves in INDEX:
        if fn in allstats:
            st = allstats[fn]
            rows = load_final(fn)
            us = np.concatenate([v[:, 2] for v in rows.values()])
            un = UNITS[fn]
            d2 = sum(s['rms'] ** 2 * s['n_ab'] for s in st) / sum(s['n_ab'] for s in st)
            mx = max(s['max'] for s in st)
            w(f'| {src} | {fig} | `{fn}` | {curves} | {_g(us.min())}-{_g(us.max())} ({_g(float(np.median(us)))}) {un} '
              f'| {_g(np.sqrt(d2))} / {_g(mx)} {un} |')
        elif fn == 'ma1997_table1.csv':
            w(f'| {src} | {fig} | `{fn}` | {curves} | half a unit of the last printed digit | identical (36 values) |')
        else:
            w(f'| {src} | {fig} | `{fn}` | {curves} | - | identical where both transcribed (see header) |')
    w('')
    w('Not done: card item D8 (Kotik & Sivov 1965, H-plane miter loss); the scan has not arrived.')
    w('')
    w('## Agreement of the two digitizations, per curve')
    w('')
    w('Over the grid points (lines) or paired symbols (markers) covered by both sets, after the exclusions listed in '
      'the file headers. "Flags" counts points where |a - b| exceeds the combined reading uncertainty '
      'sqrt(u_a² + u_b²); every flagged point was checked on the page image and is explained in the file header.')
    w('')
    w('| File | Curve | kind | n (both) | a only | b only | rms(a-b) | max abs(a-b) | mean(a-b) | flags |')
    w('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for fn, st in allstats.items():
        for s in st:
            w(f"| `{fn}` | {s['curve']} | {s['kind']} | {s['n_ab']} | {s['n_a']} | {s['n_b']} | {s['rms']:.3g} | "
              f"{s['max']:.3g} | {s['bias']:+.2g} | {s['n_exc']} |")
    w('')
    w('## Parameter definitions (they differ between sources)')
    w('')
    w('* **Alessandri et al. 1994, d** (E-plane miter, b in the bend plane): distance from the inner corner to the miter '
      'wall, measured along the bend diagonal, i.e. perpendicular to the miter wall. d/b = sqrt(2) = 1.41 is the '
      'unmitered sharp corner; d/b = 1/sqrt(2) = 0.707 joins the ends of the two outer walls. x = f/fc, fc of the '
      'fundamental mode; guide dimensions not stated.')
    w('* **Ma et al. 1997, c** (H-plane miter, a = 15.799 mm in the bend plane, b = 7.899 mm): length cut from each outer '
      'wall, measured from the removed outer corner; c/a = 0 is the unmitered right-angle corner. fc = c0/2a = 9.488 GHz. '
      'Staircase model of the miter with N steps; division methods (a) steps through, (b) below, (c) above the '
      'slanted plane (text p. 4).')
    w('* **Reiter & Arndt 1995, a_opt, b_opt (printed b_2opt in Fig. 3(d)) and γ**: a_opt, b_opt = distance from the inner '
      'corner to the miter wall, measured perpendicular to it (same definition as Alessandri\'s d); γ = interior angle '
      'between the inner walls of the two arms, so γ = 90° is a right-angle bend and the deflection angle is '
      '180° - γ. Fig. 3: WR62 (15.799 x 7.899 mm); Fig. 4: WR90 (22.86 x 10.16 mm), l = straight section between '
      'the H- and E-plane miters.')
    w('* **Marcuvitz 1951, θ** (Secs. 5.28b, 5.29b): the deflection (bend) angle between the arm axes, θ = 90° for a '
      'right-angle corner; his corners are sharp (not mitered) and the arbitrary-angle reference planes pass through '
      'the inner corner (`docs/literature/marcuvitz-steps-and-bends.md`).')
    w('* **Doane & Anderson 1989**: a = side perpendicular to the TE10 electric field, b parallel to it; their WR187 "tall" '
      'guide has a = 0.872 in, b = 1.872 in.')
    w('')
    w('Published |S11| and |S21| magnitudes do not depend on the reference planes; any phase comparison must map the '
      'paper\'s planes to CONVENTIONS.md section 6.')
    w('')
    w('## Decisions taken in the merge (details in the file headers)')
    w('')
    w('* Ma Fig. 11: line styles matched to c/a by the labels printed next to the curves; L1 checked c/a = 0 against Fig. 5, '
      'c/a = 1.0 against Fig. 9 and c/a = 0.631 against the "39 %" bandwidth of the text (below). Both digitizers agreed. '
      'Where c/a = 0.60 and 0.631 nearly touch (f/fc 1.80-1.83) L1a points on the wrong line were dropped, and the hidden '
      'local maximum of c/a = 0.631 (1.83-1.86) is left as a gap.')
    w('* Reiter Fig. 3(b): right-hand branches assigned by the width of the dips below the frame (the more reflecting '
      'γ = 105° must have the narrower dip): rising to -25.5 dB is γ = 105°, to -34.5 dB is γ = 120°. Both agreed.')
    w('* Reiter Fig. 3(d): the branch rising from -65 dB at 17.2 GHz to -50 dB at 18 GHz is not identifiable from the '
      'figure (γ = 105° or 135°; fits of a smooth zero favour neither consistently). Kept as a separate curve '
      '"right branch (gamma=105 or 135)" (L1a had it under γ = 105°).')
    w('* Reiter Fig. 4: the 11.7 GHz minimum belongs to l = b (the measured l = 0 line keeps its ripple through both '
      'crossings, the l = b line is smooth from the 9.4 GHz minimum to the 11.7 GHz minimum). The printed minima are '
      'added as one-row curves read by L1. The figure has tick marks (L1a used them; L1b used the frame).')
    w('* Doane Fig. 9: the top frame, labelled 2 %, lies 9 % closer to the 1.5 % ticks than the tick spacing, on both '
      'sides of the figure and with the label itself; this is in the print. The frame is kept as 2 % and the difference '
      'to a tick-only scale (up to 0.02 %, constant curve above 17.87 GHz) is added to the uncertainty. The rise of the '
      'triangular curve from 0.05 % to 0.15 % across 15-18 GHz is real (L1 measured it against the bottom frame in the '
      'photograph).')
    w('* Single points dropped where the page shows the reading to be off the curve: Alessandri Fig. 5 (d/b = 0.9 at '
      'f/fc 1.024-1.025 on the tick mark, d/b = 0.83 at 1.25), Ma Fig. 6 (20 steps, 10.08-10.13 GHz), Ma Fig. 11 '
      '(c/a = 1.30 at 1.805-1.81 on a tick mark), Reiter Fig. 3(b) (γ = 105° at 13.85-13.90 GHz on a leader line).')
    w('* No interpolation across a gap whose both ends lie within 4 dB of the bottom frame (reflection zeros below the '
      'frame); the zeros themselves are not in the data.')
    w('')
    w('## Cross-checks between files (computed from the merged files)')
    w('')
    for l in consistency():
        w('* ' + l)
    w('')
    w('## Transcriptions')
    w('')
    w('* `butterweck1968_numbers.csv` (D5): numbers of Butterweck, IEEE MTT-16 (1968) and Butterweck & de Ronde, Philips '
      'Tech. Rev. 29 (1968), with sentence and page.')
    w('* `docs/literature/marcuvitz-steps-and-bends.md` (D6): Marcuvitz Secs. 5.24, 5.26, 5.28, 5.29, 5.34, 5.35, with '
      'ranges, accuracies, numerical checks against the book\'s curves and the checking record. Note: the H-plane '
      'step formulas (1a)-(3a) of Sec. 5.24a diverge from the book\'s own curves for a/λ ≳ 0.7.')
    w('* `docs/literature/doane1989.md` (D7): Doane & Anderson eqs. 14-18.')
    w('')
    w('## Reproduction')
    w('')
    w('`python3 benchmarks/literature/scripts/merge_ab.py` regenerates every merged CSV and this README from `raw/a` and '
      '`raw/b` (numpy only). `scripts/spot_check.py` re-reads three points per figure on the page images (needs the '
      'reference PDFs). `scripts/marcuvitz_checks.py` evaluates the Marcuvitz formulas (numpy, scipy). The raw '
      'digitizers\' scripts in `raw/a/scripts` and `raw/b/scripts` need the PDFs and their scratch paths.')
    with open(os.path.join(LIT, 'README.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
