"""Check of L1's Reiter & Arndt (1995) Fig. 3 digitization, independent of L1's calibrations (WP3b discrepancy file
docs/discrepancies/2026-11-12-reiter-and-ma-many-step-miters.md).

Renders PDF page 4 at 400 dpi (pdftoppm), finds the four plot frames as the outermost long dark lines, calibrates
each panel from its frame alone (x: 12 .. 18 GHz; y: the printed frame values), and draws L1's merged points (red
circles) and the FEM values of data/reiter1995.npz (green dots) on the page. It prints, per panel, the fraction
of digitized points with ink within 3 px of their position (3 px is about 0.2 dB in y and 0.03 GHz in x).
Images go to the output folder only (never into the repository).

Usage: python benchmarks/fem2d/reiter_overlay_check.py <Reiter PDF> <scratch folder>
"""
import os
import subprocess
import sys

import numpy as np

from _lit import load

FRAME_DB = {"a": (0, -35), "b": (-25, -55), "c": (0, -35), "d": (-35, -65)}
BOXES_150DPI = {"a": (300, 255, 560, 465), "b": (790, 255, 1040, 465), "c": (300, 640, 560, 850),
                "d": (790, 640, 1040, 850)}


def main(pdf, out):
    from PIL import Image, ImageDraw
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, "reiter400")
    subprocess.check_call(["pdftoppm", "-r", "400", "-f", "4", "-l", "4", "-png", pdf, stem])
    page = Image.open(stem + "-4.png").convert("RGB")
    ink = np.array(page.convert("L")) < 128
    s = 400 / 150
    frames = {}
    for p, box in BOXES_150DPI.items():
        x0, y0, x1, y1 = [int(v * s) for v in box]
        sub = ink[y0:y1, x0:x1]
        vc = np.where(sub.sum(0) > 0.48 * sub.shape[0])[0] + x0
        hr = np.where(sub.sum(1) > 0.48 * sub.shape[1])[0] + y0
        frames[p] = (vc[0], vc[-1], hr[0], hr[-1]) + FRAME_DB[p]
    d, m = load("reiter1995.npz")
    dr = ImageDraw.Draw(page)
    hits = {p: [] for p in frames}
    for k, c in enumerate(m["parameters"]["curves"]):
        p = c["file"][-5]
        if c["status"] != "pass/fail" or p not in frames:
            continue
        L, R, T, B, yt, yb = frames[p]
        X = lambda x: L + (x - 12) / 6 * (R - L)
        Y = lambda y: T + (yt - y) / (yt - yb) * (B - T)
        key = "c%02d" % k
        for x, y, o in zip(d[key + "_x"], d[key + "_lit"], d[key + "_ours"]):
            px, py = int(round(X(x))), int(round(Y(y)))
            hits[p].append(bool(ink[py - 3:py + 4, px - 1:px + 2].any()))
            dr.ellipse([px - 3, py - 3, px + 3, py + 3], outline=(255, 0, 0))
            if yb < o < yt:
                dr.ellipse([X(x) - 3, Y(o) - 3, X(x) + 3, Y(o) + 3], fill=(0, 160, 0))
    for p, (L, R, T, B, yt, yb) in frames.items():
        page.crop((L - 60, T - 40, R + 40, B + 40)).save(os.path.join(out, f"reiter_fig3{p}_overlay.png"))
        print("Fig. 3(%s): frame px x %d-%d, y %d-%d; digitized points on ink (within 3 px): %.1f %%" % (
            p, L, R, T, B, 100 * np.mean(hits[p])))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
