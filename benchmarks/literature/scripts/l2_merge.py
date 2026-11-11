"""Merge the two L2 digitizations (raw/a: pass a, raw/b: pass b) into the final benchmark files (card L2).

    python3 benchmarks/literature/scripts/l2_merge.py           # writes the L2 CSVs and the README section
    python3 benchmarks/literature/scripts/l2_merge.py --stats   # prints the a-b agreement only

Same merge rules as card L1 (merge_ab.merge_line / merge_markers, imported): lines on a regular x grid inside the
traced segments of each pass, y = mean, uncertainty = max(|a - b|/2, u_a, u_b); one pass only: its y, uncertainty
doubled; no interpolation across gaps longer than `gap`, nor across a gap whose both ends lie within 6 dB of the
bottom frame (a reflection zero below the axes). Markers paired by mutual nearest neighbour. numpy only.

Kotik & Sivov Fig. 3, rigorous curve: a drafting term is added in quadrature to the reading uncertainty (see
DRAFT below): the dashed curve of the same figure, a known formula, deviates from that formula by up to 0.011 in
|B1|^2 (rms 0.005), which shows how accurately the figure was drawn.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIT = os.path.dirname(HERE)
RAW = os.path.join(LIT, 'raw')
sys.path.insert(0, HERE)
from merge_ab import read_raw, merge_line, merge_markers  # noqa: E402

MERGE_DATE = '2026-11-11'
DRAFT = 0.005          # Kotik Fig. 3: rms deviation of the printed dashed curve from its own formula
README_BEGIN = '<!-- L2 begin (written by scripts/l2_merge.py) -->'
README_END = '<!-- L2 end -->'

KOTIK_SRC = ('I. P. Kotik and A. N. Sivov, "Diffraction of electromagnetic waves on a mirror in a break of a plane '
             'waveguide", Radio Engineering and Electronic Physics 10(1), 140-142 (1965) (English translation of '
             'Radiotekhnika i Elektronika; scan, paper on PDF pages 3-5)')
KOTIK_GEOM = ('right-angle bend of a parallel-plate guide of width d (Figs. 1-2, p. 140): field E = E_y along the '
              'mirror edge, i.e. the H-plane TE_n0 problem of a rectangular guide of width a = d; the plane mirror '
              'lies on the diagonal of the square where the two guides cross (Fig. 1: from the end of one outer wall '
              'to the end of the other), so it passes through the intersection of the arm axes: the optical mirror, '
              'CONVENTIONS.md section 6 plain miter, d/a = 1/sqrt(2) measured from the inner corner. kd = 2 pi a / '
              'lambda; TE_n0 cut-off at kd = n pi (the solid curve is drawn for 3 < kd/pi < 10, 3 to 9 propagating '
              'TE_n0 modes).')
KOTIK_AX = ('x = kd (dimensionless), linear, gridlines at kd = 10 (left frame), 15, 20, 25, 30; the tick labels '
            '4 pi ... 10 pi are printed between them without tick marks (checked against the gridline scale, see '
            'docs/reports/L2.md); y = |B1|^2 = TE10 power transmission, linear, gridlines 0.5, 0.75, 1.0')

FILES = [
    dict(file='kotik1965_fig3_rigorous.csv', step=0.05, gap=0.65, xdec=3, ydec=5, draft=DRAFT,
         src=KOTIK_SRC, axes=KOTIK_AX, geom=KOTIK_GEOM,
         figure='Fig. 3, p. 142 (PDF p. 5): |B1|^2 vs kd, solid line = rigorous solution (truncated system of '
                'eqs. 9-10, up to N = 18 modes per guide, text p. 141)',
         curves=[('rigorous', 'line', 'rigorous', 'rigorous')],
         notes=['Text p. 142: at kd ~ 30 the wave loses only 3 % of its power in the bend; no drastic change of the '
                'solution around the critical values kd = n pi. Shape on the page: flat at 0.85 for kd 10-12.1, sharp '
                'dip to 0.746 at kd = 12.9 (just above the TE40 cut-off 4 pi = 12.57), smaller dips just above 6 pi '
                '(0.874 at kd = 19.0) and 8 pi (0.915 at 25.4); 0.971 at kd = 30 (the 3 % of the text).',
                'Beyond kd = 28.5 the printed solid line is thin and faint, has a gap near kd = 29.3-29.7 and runs into '
                'the dashed curve at kd = 31.4 (= 10 pi, end of the computed range); points there carry the larger '
                'reading uncertainty of that stretch, the last 0.5 kd before the end is not digitized (merged with '
                'the dashed line).',
                'Flags (|a-b| > combined reading uncertainty, both checked on the page): kd 24.65-24.90, the steep '
                'descent into the dip after 8 pi that starts on the kd = 25 gridline (the ridge follower of pass a and '
                'the column runs of pass b split the stroke differently there; |a-b| <= 0.005), and kd 28.30-28.40, '
                'where the stroke becomes thin and faint (|a-b| <= 0.0025). The merged uncertainty covers |a-b|/2.',
                f'Drafting term: uncertainty = sqrt(u_merge^2 + {DRAFT}^2); {DRAFT} = rms deviation of the printed '
                'dashed curve from its formula 1 - 7.4/(kd)^(3/2) (max 0.011 at kd = 12, see '
                'kotik1965_fig3_asymptotic.csv), i.e. the accuracy with which the figure was drawn.']),
    dict(file='kotik1965_fig3_asymptotic.csv', step=0.05, gap=1.6, xdec=3, ydec=5,
         src=KOTIK_SRC, axes=KOTIK_AX, geom=KOTIK_GEOM,
         figure='Fig. 3, p. 142 (PDF p. 5): dashed line = asymptotic formula of Katsenelenbaum (1963), their ref. [1]',
         curves=[('asymptotic', 'line', 'asymptotic', 'asymptotic')],
         notes=['Formula (text p. 142), as printed in the translation: |B~1|^2 = 1 - [7.4/(kd)^(2/3)]; the exponent '
                'must be 3/2: with 2/3 the value at kd = 10 is negative (1 - 7.4/4.64), with 3/2 it is 0.766, where '
                'the dashed curve starts. Use the formula |B~1|^2 = 1 - 7.4/(kd)^(3/2) itself, not these points: '
                'the points document the drawing.',
                'Printed curve minus formula: -0.010 at kd = 12, +0.004 for kd > 18 (rms 0.005, max 0.011) in both '
                'passes; a kd shift cannot explain it (it would need -0.45 at kd = 12 and +1 at kd = 25): it is the '
                'drafting accuracy of the figure. Text: for kd > 6 pi the formula deviates from the rigorous '
                'solution by at most 3 % (relative).',
                'The dashes are interrupted where the solid curve runs on them (kd 16.9-17.2, 22.5-23.3, 23.9-25.3, '
                '30.7-31.6); the merge interpolates across gaps up to 1.6 in kd (the curve is smooth).']),
]

ALIM_COMMON = dict(step=0.005, gap=0.031, xdec=4, ydec=3)
FLAG_NOTES = {
    'alimenti1994_fig2_mm.csv': [
        'Flags: the dots at f/fc 1.33 and 1.49 sit where the MM and MMFD strokes run side by side, which widens the '
        'ink; the two passes place the dot centres 0.005 apart in x (5 px). x_uncertainty covers half of it.'],
    'alimenti1994_fig3_mm.csv': [
        'Reflection zero (both lines): the strokes plunge at f/fc = 1.978 +- 0.005 (pass a scale) to the -70 dB '
        'frame; one of the two lines turns back at about -47 dB and rises to -37 dB at f/fc 1.99, the other runs '
        'along the frame; not digitized.',
        'Cross-check: this curve lies 0.5-1.0 dB above Ma et al. 1997 Fig. 11, c/a = 1.0 (ma1997_fig11.csv; the '
        'same geometry, d/a = 0.707) between f/fc 1.22 and 1.83 (140 points, mean +0.68 dB; at f/fc = 1.686: '
        '-10.80 dB here, -11.60 dB in Ma, Ma Table 1 -11.63 dB at 16 GHz = 1.686 fc). The page reading is '
        'confirmed by both passes (rms 0.13 dB) and by the overlay check; the difference is between the papers.'],
    'alimenti1994_fig3_mmfd.csv': [
        'Flag: the last diamond (f/fc 1.93) lies where the two lines start to separate; the passes differ by '
        '0.63 dB in y.'],
    'alimenti1994_fig4_t038.csv': [
        'Flag: f/fc 1.955, the near-vertical descent into the second zero (dashes), |a-b| = 1.8 dB, covered by the '
        'merged uncertainty. The second zero lies at f/fc = 1.967 +- 0.004 (between the lowest dashes of the two '
        'flanks, -42 dB); the dashed line rises to -16 dB at f/fc = 1.985.',
        'The open squares (symbols of the line style) are left out by both passes; the line is interpolated across '
        'them where the gap is <= 0.031.'],
    'alimenti1994_fig6_bcmm.csv': [
        'Flags: f/fc 1.90, the end of the BCMM line at its last diamond, and the diamond at f/fc 1.79 (0.004 apart '
        'in x); covered by the merged uncertainties.'],
}
for k, files in [(2, [('alimenti1994_fig2_mmfd.csv', [('MMFD', 'line')], -10),
                      ('alimenti1994_fig2_mm.csv', [('MM markers', 'markers')], -10)]),
                 (3, [('alimenti1994_fig3_mm.csv', [('MM', 'line')], -70),
                      ('alimenti1994_fig3_mmfd.csv', [('MMFD markers', 'markers')], -70)]),
                 (4, [('alimenti1994_fig4_t030.csv', [('t/a=0.30', 'line')], -50),
                      ('alimenti1994_fig4_t034.csv', [('t/a=0.34', 'line')], -50),
                      ('alimenti1994_fig4_t038.csv', [('t/a=0.38', 'line')], -50)]),
                 (5, [('alimenti1994_fig5_matching_frequency.csv', [('matching frequency', 'line')], None),
                      ('alimenti1994_fig5_bandwidth.csv', [('bandwidth one zero', 'line'),
                                                            ('bandwidth two zeros', 'line')], None)]),
                 (6, [('alimenti1994_fig6_mmfd.csv', [('MMFD', 'line')], -40),
                      ('alimenti1994_fig6_bcmm.csv', [('BCMM', 'line'), ('BCMM markers', 'markers')], -40)])]:
    for fn, curves, ybot in files:
        d = dict(ALIM_COMMON, file=fn, fig=k, curves=[(c, kind, c, c) for c, kind in curves])
        d['notes'] = FLAG_NOTES.get(fn, [])
        if ybot is not None:
            d['ybot'] = ybot
        if k == 5:
            d.update(step=0.0005, gap=0.004, xdec=5, ydec=4)
        d['xspan'], d['yspan'] = ((1.0, 10.0) if k == 2 else (1.0, 70.0) if k == 3 else (1.0, 40.0))
        FILES.append(d)

UNITS = {'kotik': '', 'alimenti1994_fig5': ''}


def unit(fn):
    return '' if fn.startswith('kotik') or 'fig5' in fn else ' dB'


def header_lines(path):
    with open(path) as fh:
        return [ln[2:].rstrip('\n') for ln in fh if ln.startswith('#')]


def pick(lines, key):
    return [ln[len(key) + 1:].strip() for ln in lines if ln.startswith(key + ':')]


def process(spec, write=True, verbose=True):
    pa, pb = os.path.join(RAW, 'a', spec['file']), os.path.join(RAW, 'b', spec['file'])
    ha, A = read_raw(pa)
    hb, B = read_raw(pb)
    ha, hb = header_lines(pa), header_lines(pb)
    rows, stats = [], []
    for final, kind, la, lb in spec['curves']:
        va, vb = A.get(la), B.get(lb)
        if kind == 'line':
            floor = (spec['ybot'], 6.0, 1.5 * spec['step']) if 'ybot' in spec else None
            r, st = merge_line(va, vb, spec['step'], spec['gap'], {}, floor)
        else:
            r, st = merge_markers(va, vb, spec['xspan'], spec['yspan'])
        if spec.get('draft'):
            r = [(x, y, float(np.hypot(u, spec['draft'])), xu, s) for x, y, u, xu, s in r]
        st.update(curve=final, kind=kind, urange=(min(v[2] for v in r), float(np.median([v[2] for v in r])),
                                                  max(v[2] for v in r)), n=len(r))
        stats.append(st)
        rows += [(final,) + tuple(v) for v in r]
    if verbose:
        for st in stats:
            print(f"{spec['file']:42s} {st['curve'][:20]:20s} {st['kind']:7s} n={st['n']:4d} n_ab={st['n_ab']:4d} "
                  f"a={st['n_a']:3d} b={st['n_b']:3d} rms={st['rms']:.3g} max={st['max']:.3g} bias={st['bias']:+.2g} "
                  f"flags={st['n_exc']} u=[{st['urange'][0]:.3g}, {st['urange'][1]:.3g}, {st['urange'][2]:.3g}]")
            for e in st['exc']:
                print('     exceed:', ' '.join(f'{v:.4f}' for v in e))
    if write:
        write_file(spec, ha, hb, rows, stats)
    return stats


def write_file(spec, ha, hb, rows, stats):
    if spec['file'].startswith('kotik'):
        hdr = [f"Source: {spec['src']}", f"Figure: {spec['figure']}", f"Axes: {spec['axes']}",
               f"Geometry: {spec['geom']}"]
    else:
        hdr = ['Source: ' + pick(ha, 'source')[0], 'Figure: ' + pick(ha, 'figure')[0], 'Axes: ' + pick(ha, 'axes')[0],
               'Geometry: ' + pick(ha, 'geometry')[0], 'Curves: ' + pick(ha, 'curves')[0]]
    hdr.append('Columns: curve, x, y, uncertainty, x_uncertainty, source. uncertainty = y uncertainty in the units '
               'of y; x_uncertainty = 0 for lines (regular grid; x reading errors are converted into the y '
               'uncertainty), x uncertainty for markers (one row per printed symbol). source = ab (both passes), a '
               'or b (one pass only, uncertainty doubled).')
    hdr.append(f"Merge (card L2, {MERGE_DATE}, scripts/l2_merge.py, rules of card L1): lines interpolated on a grid "
               f"of step {spec['step']:g} in x inside each pass's traced segments (no interpolation across gaps > "
               f"{spec['gap']:g}" + (', nor across a gap wider than 1.5 grid steps whose both ends lie within 6 dB of the bottom frame, i.e. a reflection zero below the frame' if
                                     'ybot' in spec else '') +
               "); y = mean of a and b; uncertainty = max(|y_a - y_b|/2, u_a, u_b); one pass only: its y, "
               "uncertainty doubled. Markers paired by mutual nearest neighbour (axis-normalized distance <= 0.03); "
               "x, y = means, uncertainties = max(half difference, u_a, u_b); unpaired: one pass, doubled.")
    agr = [f"{st['curve']}: n_ab={st['n_ab']}, a-only={st['n_a']}, b-only={st['n_b']}, rms(a-b)={st['rms']:.3g}, "
           f"max|a-b|={st['max']:.3g}, mean(a-b)={st['bias']:+.3g}, points with |a-b| > sqrt(u_a^2+u_b^2): "
           f"{st['n_exc']}" for st in stats]
    hdr.append('Agreement of the two digitizations (y units; grid points or paired markers covered by both): '
               + '; '.join(agr) + '.')
    hdr += spec.get('notes', [])
    for tag, h in (('a', ha), ('b', hb)):
        for key in ('calibration', 'calibration (L2%s)' % tag):
            for t in pick(h, key):
                hdr.append(f'Calibration L2{tag} (raw/{tag}): {t}')
        for t in pick(h, 'method (L2%s)' % tag):
            hdr.append(f'Method L2{tag}: {t}')
        for t in pick(h, 'jump (L2%s)' % tag):
            hdr.append(f'Jump L2{tag}: {t}')
    hdr.append(f"Digitizers: L2 passes a and b (raw files benchmarks/literature/raw/a/{spec['file']} and "
               f"raw/b/{spec['file']}; scripts in benchmarks/literature/scripts/l2/, they need the reference PDFs); "
               f"merged by scripts/l2_merge.py, {MERGE_DATE}; overlay check scripts/l2_check.py.")
    with open(os.path.join(LIT, spec['file']), 'w', newline='') as fh:
        for h in hdr:
            fh.write('# ' + h + '\n')
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['curve', 'x', 'y', 'uncertainty', 'x_uncertainty', 'source'])
        xd, yd = spec['xdec'], spec['ydec']
        for c, x, y, u, xu, s in rows:
            w.writerow([c, f'{x:.{xd}f}', f'{y:.{yd}f}', f'{max(u, 10 ** -yd):.{yd}f}', f'{xu:.{xd}f}', s])


INDEX = [
    ('Alimenti, Mongiardo & Sorrentino 1994', 'Fig. 2: \\|S11\\| (dB) vs f/fc, unmitered H-plane corner (d/a = 1.414)',
     'alimenti1994_fig2_mmfd.csv', 'MMFD line (MM line coincides)'),
    ('Alimenti et al. 1994', 'Fig. 2', 'alimenti1994_fig2_mm.csv', 'MM dots (10 markers)'),
    ('Alimenti et al. 1994', 'Fig. 3: \\|S11\\| (dB), fully mitered H-plane bend (t/a = 0, d/a = 0.707)',
     'alimenti1994_fig3_mm.csv', 'MM line (MMFD coincides up to f/fc = 1.93)'),
    ('Alimenti et al. 1994', 'Fig. 3', 'alimenti1994_fig3_mmfd.csv', 'MMFD diamonds (10 markers)'),
    ('Alimenti et al. 1994', 'Fig. 4: \\|S11\\| (dB) vs f/fc, mitered H-plane bends', 'alimenti1994_fig4_t030.csv',
     't/a = 0.30 (d/a = 0.919)'),
    ('Alimenti et al. 1994', 'Fig. 4', 'alimenti1994_fig4_t034.csv', 't/a = 0.34 (d/a = 0.948)'),
    ('Alimenti et al. 1994', 'Fig. 4', 'alimenti1994_fig4_t038.csv', 't/a = 0.38 (d/a = 0.976)'),
    ('Alimenti et al. 1994', 'Fig. 5: design chart vs t/a', 'alimenti1994_fig5_matching_frequency.csv',
     'matching frequency f/fc (left axis)'),
    ('Alimenti et al. 1994', 'Fig. 5', 'alimenti1994_fig5_bandwidth.csv',
     '30 dB bandwidth / fc (right axis), one-zero and two-zero branches, jump at t/a = 0.3867'),
    ('Alimenti et al. 1994', 'Fig. 6: \\|S11\\| (dB), t/a = 0.386 (d/a = 0.980)', 'alimenti1994_fig6_mmfd.csv',
     'MMFD'),
    ('Alimenti et al. 1994', 'Fig. 6', 'alimenti1994_fig6_bcmm.csv', 'BCMM line + 10 diamonds'),
    ('Kotik & Sivov 1965', 'Fig. 3: \\|B1\\|² vs kd, 90° H-plane optical-mirror bend (d/a = 0.707)',
     'kotik1965_fig3_rigorous.csv', 'rigorous solution (solid)'),
    ('Kotik & Sivov 1965', 'Fig. 3', 'kotik1965_fig3_asymptotic.csv',
     'asymptotic formula 1 - 7.4/(kd)^(3/2) as drawn (dashed)'),
]


def readme_section(allstats):
    L = [README_BEGIN, '', '## Card L2: Alimenti et al. 1994 and Kotik & Sivov 1965', '',
         'Digitized twice by agent L2 with different rasters, calibrations and extraction methods (pass a: 400 dpi '
         'render, ridge follower, tick or gridline calibration with tilt correction; pass b: the embedded 300 ppi '
         'scan, column-run scan, major + minor ticks or straight gridline fits; raw sets in `raw/a/`, `raw/b/`, '
         'scripts in `scripts/l2/`), merged with the L1 rules by `scripts/l2_merge.py`; `scripts/l2_check.py` '
         'overlays the merged points on the rasterized figures and reports the pixel residuals. Same columns as '
         'above. Report: `docs/reports/L2.md`.', '',
         '| Source | Figure | File | Curves | rows | y uncertainty: min-max (median) | a-b agreement: rms / max |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    for src, fig, fn, curves in INDEX:
        st = allstats.get(fn)
        if not st:
            continue
        n = sum(s['n'] for s in st)
        lo, hi = min(s['urange'][0] for s in st), max(s['urange'][2] for s in st)
        med = float(np.median([s['urange'][1] for s in st]))
        nab = sum(s['n_ab'] for s in st)
        rms = np.sqrt(sum(s['rms'] ** 2 * s['n_ab'] for s in st if s['n_ab']) / max(nab, 1))
        mx = max(s['max'] for s in st)
        u = unit(fn)
        L.append(f'| {src} | {fig} | `{fn}` | {curves} | {n} | {lo:.3g}-{hi:.3g} ({med:.3g}){u} | '
                 f'{rms:.3g} / {mx:.3g}{u} |')
    L += ['', '**Geometry conversions (CONVENTIONS.md section 6, d = distance of the miter plane from the inner '
          'corner):**', '',
          '* **Alimenti et al. 1994, t**: the dimension arrows of the inset of Fig. 4 run from the end of the 45° '
          'miter on the outer wall to the thin line that continues the other arm\'s inner wall, so t is the part of '
          'the outer wall left standing and the cut measured from the removed outer corner is c = a - t (Ma et al. '
          '1997\'s c). Hence **d/a = (1 + t/a)/√2** (not (2 - t/a)/√2): t/a = 0 is the fully mitered bend of Fig. 3 '
          '(optical mirror, d/a = 0.707), t/a = 0.386 gives d/a = 0.980 (Ma\'s c/a = 0.614). Checked against Ma '
          'Fig. 11: the zero frequency rises with t/a here and falls with c/a there, and t/a = 0.30 (c/a = 0.70) '
          'puts the zero at f/fc ≈ 1.29 here, 1.38 in Ma.',
          '* **Kotik & Sivov 1965**: d is the guide width (our a), kd = 2πa/λ; the mirror is the diagonal of the '
          'crossing square (Figs. 1-2), i.e. the optical mirror, d/a = 1/√2.',
          '* Fig. 2 (unmitered corner) is d/a = √2.', '',
          '**Agreement of the two passes, per curve** (same definitions as above):', '',
          '| File | Curve | kind | n (both) | a only | b only | rms(a-b) | max abs(a-b) | mean(a-b) | flags |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for fn, st in allstats.items():
        for s in st:
            L.append(f"| `{fn}` | {s['curve']} | {s['kind']} | {s['n_ab']} | {s['n_a']} | {s['n_b']} | "
                     f"{s['rms']:.3g} | {s['max']:.3g} | {s['bias']:+.2g} | {s['n_exc']} |")
    L += ['', README_END]
    return '\n'.join(L) + '\n'


def write_readme(allstats):
    path = os.path.join(LIT, 'README.md')
    text = open(path).read()
    sec = readme_section(allstats)
    if README_BEGIN in text:
        pre = text[:text.index(README_BEGIN)]
        post = text[text.index(README_END) + len(README_END):].lstrip('\n')
        text = pre + sec + ('\n' + post if post else '')
    else:
        text = text.rstrip('\n') + '\n\n' + sec
    with open(path, 'w') as fh:
        fh.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', action='store_true')
    args = ap.parse_args()
    allstats = {}
    for spec in FILES:
        allstats[spec['file']] = process(spec, write=not args.stats)
    if not args.stats:
        write_readme(allstats)
    return allstats


if __name__ == '__main__':
    main()
