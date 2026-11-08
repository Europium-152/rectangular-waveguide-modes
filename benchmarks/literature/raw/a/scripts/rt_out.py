exec(open('/home/claude/wt/L1a-scratch/rt_common.py').read())
REF = "J. M. Reiter and F. Arndt, \"Rigorous analysis of arbitrarily shaped H- and E-plane discontinuities in rectangular waveguides by a full-wave boundary contour mode-matching method\", IEEE Trans. Microwave Theory Tech. 43(4), 796-801 (April 1995)"
def grid_rows(label, t, cX, cY, mx, my, step=0.05, maxgap=0.25, cal_px=1.0, x_lo=-1e9, x_hi=1e9):
    D = np.array([todata(cX, cY, p[0], p[1]) for p in t[:, :2]])
    o = np.argsort(D[:, 0]); D = D[o]; hw = t[o, 3]
    grid = np.round(np.arange(np.ceil(D[:, 0].min() / step) * step, D[:, 0].max() + 1e-9, step), 4)
    grid = grid[(grid >= max(D[:, 0].min(), x_lo)) & (grid <= min(D[:, 0].max(), x_hi))]
    return [(label, g, y, u, 0.0) for g, y, u in to_grid(D[:, 0], D[:, 1], hw, grid, mx, -my, maxgap=maxgap, cal_px=cal_px)]
def header(fig, caption, curves, geom, x_desc, y_desc, P, res, cX, cY, slopes, extra=()):
    return [
     "source: " + REF,
     "figure: Fig. %s, p. 799 (PDF p. 4): \"%s\"" % (fig, caption),
     "curves: " + curves,
     "geometry: " + geom,
     "x: " + x_desc, "y: " + y_desc,
     ] + list(extra) + [
     "method: native 1-bit scan (339 dpi) extracted with pdfimages; dotted grid lines located from their dots, each fitted with a line, the grid intersections give an affine calibration (scan rotation %.2f to %.2f deg); curves are the long connected components inside the frame, measured by ink centroids in 3 px bins along each component (numpy/scipy); scripts in benchmarks/literature/raw/a/scripts/ (L1a; they read page images extracted from the reference PDFs, which are not in the repository)" % (min(slopes), max(slopes)),
     "calibration points (grid intersections, page pixels (x,y) -> (X,Y)): " + "; ".join("(%.1f,%.1f)->(%g,%g)" % tuple(p) for p in P),
     "calibration fit: X = %.6g*px + %.6g*py + %.6g, Y = %.6g*px + %.6g*py + %.6g; max residual %.4f in X, %.3f in Y" % (cX[0], cX[1], cX[2], cY[0], cY[1], cY[2], abs(res[:, 0]).max(), abs(res[:, 1]).max()),
     "digitizer: L1a (independent of L1b), 2026-11-08",
    ]
