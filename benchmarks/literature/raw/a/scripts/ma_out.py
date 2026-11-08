import numpy as np
def ma_header(fig, page, caption, curves, x_desc, y_desc, cal, extra=(), method_extra=''):
    mx, cx, my, cy, rx, ry, XT, XV, YT, YV = cal
    H = [
     "source: Z. Ma, T. Yamane, E. Yamashita, \"Analysis and design of mitered H-plane bends in rectangular waveguides by combining the mode-matching and port reflection coefficient methods\", Electronics and Communications in Japan, Part 2, 80(5), 1-7 (1997) (translated from Denshi Joho Tsushin Gakkai Ronbunshi J80-C-I(2), 78-85, Feb. 1997)",
     "figure: Fig. %s, p. %d (PDF p. %d): \"%s\"" % (fig, page, page, caption),
     "curves: " + curves,
     "guide dimensions (p. 4): a = 15.799 mm, b = 7.899 mm; single-mode (TE10) band 9.49-19.0 GHz; H-plane bends (a in the bend plane)",
     "x: " + x_desc,
     "y: " + y_desc,
    ] + list(extra) + [
     "method: native 1-bit scan (300 dpi) extracted with pdfimages (no measurable skew), curves tracked by a curvature-aware follower and measured by ink centroids in 3 px bins (numpy/scipy); circles located by ring template matching and a least-squares circle fit; legend and inset masked" + method_extra + "; scripts in benchmarks/literature/raw/a/scripts/ (L1a; they read page images extracted from the reference PDFs, which are not in the repository)",
     "calibration points (page pixels -> value): x ticks px " + ", ".join("%.1f->%g" % (a, b) for a, b in zip(XT, XV)) + "; y ticks px " + ", ".join("%.1f->%g" % (a, b) for a, b in zip(YT, YV)),
     "calibration fit: x = %.7g*px + %.6g (max residual %.4g), y = %.6g*py + %.6g (max residual %.4g)" % (mx, cx, np.abs(rx).max(), my, cy, np.abs(ry).max()),
     "digitizer: L1a (independent of L1b), 2026-11-08",
    ]
    return H
