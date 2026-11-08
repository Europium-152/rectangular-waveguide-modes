"""Helpers for digitizing raster figures (L1b)."""
import numpy as np
from PIL import Image

def load(path, thr=140):
    im = np.array(Image.open(path).convert('L'))
    return im, im < thr

def vruns(col):
    """runs of True in 1D array: list of (start, end_inclusive)"""
    idx = np.flatnonzero(col)
    if idx.size == 0:
        return []
    br = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]]
    return list(zip(starts, ends))

def lin(p0, v0, p1, v1):
    """pixel->value linear map from two calibration points"""
    s = (v1 - v0) / (p1 - p0)
    return lambda p: v0 + (np.asarray(p, float) - p0) * s

def fitcal(pix, val):
    """least-squares linear map from several calibration points; returns map, residual (in value units)"""
    pix = np.asarray(pix, float); val = np.asarray(val, float)
    A = np.c_[pix, np.ones_like(pix)]
    c, *_ = np.linalg.lstsq(A, val, rcond=None)
    res = val - A @ c
    return (lambda p: c[0] * np.asarray(p, float) + c[1]), np.max(np.abs(res)), c
