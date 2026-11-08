import numpy as np, os
OUT = '/home/claude/wt/L1b/benchmarks/literature/raw/b'

def write_csv(fname, header_lines, rows, cols=('curve', 'x', 'y', 'uncertainty'), fmt=None):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, fname)
    with open(path, 'w') as f:
        for h in header_lines:
            f.write('# ' + h + '\n' if h else '#\n')
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(fmt[i] % v if fmt else str(v) for i, v in enumerate(r)) + '\n')
    return path

def unc(slope_val_per_px, dy_per_px, sx_px=1.0, sy_px=1.0, cal=0.0):
    """reading uncertainty in y units: sx px along x converted through the local slope, sy px along y, plus the
    calibration residual (added linearly)."""
    return np.hypot(sx_px * slope_val_per_px, sy_px * dy_per_px) + cal
