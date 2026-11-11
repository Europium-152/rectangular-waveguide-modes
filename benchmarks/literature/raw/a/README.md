# L1a digitizations (set a)

Card L1, agent L1a. The digitizations here are independent of set b (L1b). The coordinator merges the two sets
into `benchmarks/literature/*.csv`. Every CSV starts with `#` comment lines that give the source, figure, curves,
axes and units, geometry, calibration points, method and digitizer. Its columns are `curve, x, y, uncertainty,
x_uncertainty`. `uncertainty` is L1a's reading uncertainty of y at the given x (half the printed line width
converted with the local slope, plus calibration error and interpolation error). `x_uncertainty` is non-zero
only for markers and dots, which have their own x.

| Source | Figure | File | Curves | y uncertainty (min to max, median) |
| --- | --- | --- | --- | --- |
| Alessandri, Mongiardo & Sorrentino 1994 | Fig. 4, \|S11\| dB vs f/fc, 90° mitered E-plane bend | `alessandri1994_fig4.csv` | d/b = 1.41, 0.707, 0.87 and the crosses of ref. [7] | 0.10–0.54 dB, 0.17 dB |
| Alessandri et al. 1994 | Fig. 5, \|S11\| dB vs f/fc | `alessandri1994_fig5.csv` | d/b = 0.9, 0.87, 0.85, 0.83 | 0.12–1.5 dB, 0.20 dB |
| Doane & Anderson 1989 | Fig. 9, spurious mode power (%) vs GHz, 20" 90° twists in WR187 tall guide | `doane1989_fig9.csv` | constant, cosine, triangular | 0.004–0.022 %, 0.008 % |
| Ma, Yamane & Yamashita 1997 | Fig. 5, \|S21\|² vs GHz, right-angle H-plane corner | `ma1997_fig5.csv` | 12 terms, BEM circles | 0.003–0.086, 0.008 |
| Ma et al. 1997 | Fig. 6, \|S21\| dB vs GHz, mitered bend c/a = 1.0 | `ma1997_fig6.csv` | 60, 20, 10 steps, BEM circles | 0.05–4.8 dB, 0.13 dB |
| Ma et al. 1997 | Fig. 8, \|S11\| dB vs GHz, 20 steps, division methods | `ma1997_fig8.csv` | (a), (b), (c) | 0.18–1.1 dB, 0.23 dB |
| Ma et al. 1997 | Fig. 9, \|S11\| dB vs GHz, 60 steps | `ma1997_fig9.csv` | (a)/(b)/(c) merged | 0.21–1.1 dB, 0.32 dB |
| Ma et al. 1997 | Fig. 11, \|S11\| dB vs f/fc, miter size c/a | `ma1997_fig11.csv` | c/a = 0, 0.60, 0.631, 0.70, 1.0, 1.30 | 0.11–3.8 dB, 0.19 dB |
| Ma et al. 1997 | Table 1, \|S11\|, \|S21\| at 16 GHz (transcribed) | `ma1997_table1.csv` | 50 and 60 steps, step displacements | 0.00005 (last digit) |
| Reiter & Arndt 1995 | Fig. 3(a), H-plane bends, WR62 | `reiter1995_fig3a.csv` | γ = 90 ... 150° | 0.10–0.27 dB, 0.16 dB |
| Reiter & Arndt 1995 | Fig. 3(b), mitered H-plane bends | `reiter1995_fig3b.csv` | γ = 105, 120, 135° | 0.15–1.9 dB, 0.22 dB |
| Reiter & Arndt 1995 | Fig. 3(c), E-plane bends | `reiter1995_fig3c.csv` | γ = 90 ... 150° | 0.17–0.27 dB, 0.19 dB |
| Reiter & Arndt 1995 | Fig. 3(d), mitered E-plane bends | `reiter1995_fig3d.csv` | γ = 105, 120, 135° | 0.13–3.2 dB, 0.17 dB |
| Reiter & Arndt 1995 | Fig. 4, cascaded mitered H- and E-plane bends, WR90 | `reiter1995_fig4.csv` | l = 0, l = b (theory and measurement merged) | 0.22–0.74 dB, 0.33 dB |

Transcriptions (D5, D6, D7) are in `docs/literature/raw/a/`. Kotik & Sivov 1965 (D8) is not done: the scan has
not arrived yet.

## Method and reproduction

`scripts/` holds the scripts that produced the files. `dig.py` is the toolkit: thresholding, axis and tick
location, deskew, a curvature-aware curve follower, ink-centroid tracing, outlier rejection, gridding with
uncertainties, and ring and cross marker detection. One script per figure family: `fig4.py` and
`fig5*.py` (Alessandri), `doane*.py`, `ma*.py`, `rt*.py` (Reiter). The scripts were run in a scratch folder.
They expect these page images there, extracted from Daniel's reference PDFs with poppler:

```
pdfimages -f 2 -l 2 -png "Alessandri ... 1994 ....pdf" al_native        # 339 dpi 1-bit scan
pdfimages -f 7 -l 7 -png "Doane and Anderson - 1989 ....pdf" dn7        # phone photograph
pdfimages -png "Analysis and design of mitered H-plane bends ....pdf" ma/p   # 300 dpi 1-bit scans
pdfimages -f 4 -l 4 -png "Reiter and Arndt - 1995 ....pdf" rt4          # 339 dpi 1-bit scan
```

Then `scripts/prepare.py` makes the deskewed Alessandri pages and the Doane crop. The scripts contain absolute paths of that scratch folder (`/home/claude/wt/L1a-scratch`). Change the paths to rerun
them. Calibration always uses the printed tick marks or grid lines located in pixels, never the tick labels. The
calibration points are listed in each CSV header.

## Card L2 files

`alimenti1994_*.csv` and `kotik1965_*.csv` in this folder are not by L1a: they are pass a of card L2 (agent L2, 2026-11-11),
written by `benchmarks/literature/scripts/l2/` (`kotik_a.py`, `alimenti_a.py`). Method and calibration are in each
file header and in `docs/reports/L2.md`.
