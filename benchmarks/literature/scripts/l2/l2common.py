"""Shared helpers of the L2 digitizations (card L2). numpy/scipy/Pillow only."""
import os
import subprocess
import numpy as np
from PIL import Image

ALIMENTI_PDF = 'Design_of_Mitered_H-plane_Bends_in_Rectangular_Waveguides_by_Combined_Mode_Matching_and_Finite_Differences.pdf'
KOTIK_PDF = ('Kotik and Sivov - 1965 - Diffraction of electromagnetic waves on a mirror in a break of a plane '
             'waveguide(Electromagnetic wav.pdf')


def render(pdf, page, dpi, out_dir, tag):
    """Rasterize one PDF page with pdftoppm into out_dir (outside the repository); return the gray array."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f'{tag}-p{page}-{dpi}')
    png = stem + '.png'
    if not os.path.exists(png):
        subprocess.run(['pdftoppm', '-r', str(dpi), '-f', str(page), '-l', str(page), '-png', '-singlefile', pdf, stem],
                       check=True)
    return np.array(Image.open(png).convert('L')).astype(float), png


def native(pdf, page, out_dir, tag):
    """Extract the embedded page scan (pdfimages) of one page; return the gray array."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f'{tag}-p{page}-native')
    png = stem + '-000.png'
    if not os.path.exists(png):
        subprocess.run(['pdfimages', '-png', '-f', str(page), '-l', str(page), pdf, stem], check=True)
    return np.array(Image.open(png).convert('L')).astype(float), png


def line_samples_v(im, xc, ys, half=12, win=20, thr=100, minmass=2000):
    """Ink centroids of a (nearly) vertical line near column xc, one per window starting at each y in ys."""
    pts = []
    x0 = int(round(xc)) - half
    for y in ys:
        sub = 255 - im[y:y + win, x0:x0 + 2 * half + 1]
        sub = np.where(sub < thr, 0, sub)
        w = sub.sum(0)
        if w.sum() > minmass:
            pts.append((y + win / 2, (w * np.arange(x0, x0 + 2 * half + 1)).sum() / w.sum()))
    return np.array(pts)


def line_samples_h(im, yc, xs, half=12, win=20, thr=100, minmass=1500):
    pts = []
    y0 = int(round(yc)) - half
    for x in xs:
        sub = 255 - im[y0:y0 + 2 * half + 1, x:x + win]
        sub = np.where(sub < thr, 0, sub)
        w = sub.sum(1)
        if w.sum() > minmass:
            pts.append((x + win / 2, (w * np.arange(y0, y0 + 2 * half + 1)).sum() / w.sum()))
    return np.array(pts)


def robust_linfit(t, v, k=2.5, it=3):
    m = np.ones(len(t), bool)
    for _ in range(it):
        c = np.polyfit(t[m], v[m], 1)
        r = v - np.polyval(c, t)
        s = max(np.std(r[m]), 0.3)
        m = np.abs(r) < k * s
    r = v - np.polyval(c, t)
    return c, float(np.std(r[m])), float(np.abs(r[m]).max()), m


def bin_curve(x, y, uy, step, x0=None):
    """Average points in x bins of width step (bin centre = mean x of the bin). Returns x, y, u arrays; u is the
    rms of the point uncertainties combined with the scatter inside the bin."""
    order = np.argsort(x)
    x, y, uy = x[order], y[order], uy[order]
    x0 = x[0] if x0 is None else x0
    k = np.floor((x - x0) / step).astype(int)
    out = []
    for b in np.unique(k):
        m = k == b
        if m.sum() == 0:
            continue
        sc = np.std(y[m]) if m.sum() > 1 else 0.0
        out.append((x[m].mean(), y[m].mean(), np.sqrt(np.mean(uy[m] ** 2) + sc ** 2)))
    return np.array(out)


def write_raw(path, header_lines, rows, ydec=5, xdec=4):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        for h in header_lines:
            fh.write('# ' + h + '\n')
        fh.write('curve,x,y,uncertainty,x_uncertainty\n')
        for c, x, y, u, xu in rows:
            fh.write(f'{c},{x:.{xdec}f},{y:.{ydec}f},{u:.{ydec}f},{xu:.{xdec}f}\n')


def ticks(im, axis, pos, band, rng, thr_frac=0.6, thr=128):
    """Tick marks along an axis. axis='x': the band (rows pos+band[0] .. pos+band[1]) is scanned over columns
    rng; axis='y': columns pos+band[0] .. pos+band[1] scanned over rows rng. Returns the ink-weighted centres of
    the runs where the ink count exceeds thr_frac of the band width."""
    a, b = band
    if axis == 'x':
        sub = im[pos + a:pos + b, rng[0]:rng[1]] < thr
        prof = sub.sum(0)
    else:
        sub = im[rng[0]:rng[1], pos + a:pos + b] < thr
        prof = sub.sum(1)
    on = prof >= thr_frac * (b - a)
    out = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            w = prof[i:j].astype(float)
            out.append(rng[0] + (w * np.arange(i, j)).sum() / w.sum())
            i = j
        else:
            i += 1
    return np.array(out)
