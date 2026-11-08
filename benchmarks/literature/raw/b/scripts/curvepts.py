import numpy as np
from trace import segments
from writer import unc

def curve_points(mk, xgrid_px, dbpx_x, dbpx_y, fx, fy, steep=1.5, row_step=6, s_px=2.0, cal=0.0, min_cols=4):
    """Points of one continuous (solid or dashed) curve given its pixel mask.
    Shallow parts: column-centre line, sampled at xgrid_px (local linear fit over +-2 px; needs min_cols columns).
    Steep parts (|dy/dx| > steep in px): row-centre line, one point every row_step rows.
    Returns list of (x_value, y_value, uncertainty_y)."""
    cols = np.flatnonzero(mk.any(0))
    yc = np.array([np.nonzero(mk[:, c])[0].mean() for c in cols])
    pts = []
    for xg in xgrid_px:
        sel = np.abs(cols - xg) <= 2
        if sel.sum() < min_cols:
            continue
        p = np.polyfit(cols[sel], yc[sel], 1)
        wide = np.abs(cols - xg) <= 6
        s = np.polyfit(cols[wide], yc[wide], 1)[0] if wide.sum() >= 4 else p[0]
        if abs(s) > steep:
            continue
        pts.append((float(fx(xg)), float(fy(np.polyval(p, xg))), unc(abs(s) * dbpx_y, dbpx_y, s_px, s_px, cal)))
    # row centres, one per run of pixels in a row (a row can cut the same curve twice)
    cand = []
    for r in np.flatnonzero(mk.any(1)):
        xs_r = np.flatnonzero(mk[r])
        br = np.flatnonzero(np.diff(xs_r) > 3)
        for a, b in zip(np.r_[0, br + 1], np.r_[br, xs_r.size - 1]):
            cand.append((r, xs_r[a:b + 1].mean()))
    cand = np.array(cand) if cand else np.zeros((0, 2))
    for i in range(len(cand)):
        r, xv0 = cand[i]
        if int(r) % row_step:
            continue
        sel = (np.abs(cand[:, 0] - r) <= 3) & (np.abs(cand[:, 1] - xv0) <= 8)
        if sel.sum() < 5:
            continue
        if np.ptp(cand[sel, 0]) < 4:
            continue
        q = np.polyfit(cand[sel, 0], cand[sel, 1], 1)  # dx/dy
        if abs(q[0]) * steep >= 1.0:  # shallow here, handled by columns
            continue
        s = 1.0 / q[0] if abs(q[0]) > 1e-3 else 1e3
        xv = np.polyval(q, r)
        wide = np.abs(cols - xv) <= 6
        if wide.sum() >= 6 and abs(np.polyfit(cols[wide], yc[wide], 1)[0]) <= steep / 2:
            continue  # the column centre line says this part is shallow: stray pixels
        pts.append((float(fx(xv)), float(fy(r)), unc(abs(s) * dbpx_y, dbpx_y, s_px, s_px, cal)))
    pts.sort()
    return pts

def dot_points(mk, fx, fy, dbpx_y, s_px=2.0, cal=0.0, minw=2):
    """One point per dot/dash (run of consecutive columns): centroid of its column-centre line."""
    cols = np.flatnonzero(mk.any(0))
    yc = np.array([np.nonzero(mk[:, c])[0].mean() for c in cols])
    pts = []
    for s0, e0 in segments(cols):
        if e0 - s0 + 1 < minw:
            continue
        xm, ym = cols[s0:e0 + 1].mean(), yc[s0:e0 + 1].mean()
        sl = (yc[e0] - yc[s0]) / max(cols[e0] - cols[s0], 1)
        pts.append((float(fx(xm)), float(fy(ym)), unc(abs(sl) * dbpx_y, dbpx_y, s_px, s_px, cal)))
    return pts

def drop_near_others(pts_by, masks_by, px, py, r=6):
    """Remove points of a curve that lie within r px of pixels assigned to another curve (crossings),
    where the centre line of either curve is contaminated. Returns new dict and the number removed."""
    from scipy.spatial import cKDTree
    trees = {k: cKDTree(np.c_[np.nonzero(mk)[1], np.nonzero(mk)[0]]) for k, mk in masks_by.items() if mk.any()}
    out, nrem = {}, 0
    for k, pp in pts_by.items():
        keep = []
        for p in pp:
            q = (float(px(p[0])), float(py(p[1])))
            near = any(t.query_ball_point(q, r) for kk, t in trees.items() if kk != k and kk.rstrip('r') != k.rstrip('r'))
            if near:
                nrem += 1
            else:
                keep.append(p)
        out[k] = keep
    return out, nrem
