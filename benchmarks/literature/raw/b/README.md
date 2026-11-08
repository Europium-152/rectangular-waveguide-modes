# L1b raw digitizations (card L1)

Independent digitization by agent L1b; the coordinator merges these with L1a's set (`raw/a/`) into
`benchmarks/literature/*.csv`. Every CSV has a `#` header with source, figure, curve labels, axes and units, guide
dimensions, calibration points, method and digitizer, then the columns `curve,x,y,uncertainty,x_uncertainty`
(`uncertainty` = L1b reading uncertainty of `y`, in the units of `y`; `x_uncertainty` = same for `x`).
Points are unevenly spaced; branches of one curve on either side of a null share the label.

| File | Source | Curves | Points | y uncertainty (median; range) |
| --- | --- | --- | --- | --- |
| `alessandri1994_fig4.csv` | Alessandri et al. 1994, Fig. 4 | d/b = 1.41, 0.707, 0.87 and the crosses of ref. [7] | 230 | 0.21 dB; 0.19-0.87 |
| `alessandri1994_fig5.csv` | Alessandri et al. 1994, Fig. 5 | d/b = 0.9, 0.87, 0.85, 0.83 | 564 | 0.28 dB; 0.27-10 (cap, steep flanks) |
| `doane1989_fig9.csv` | Doane & Anderson 1989, Fig. 9 | constant, cosine, triangular twist rate | 786 | 0.010 %; 0.010-0.042 |
| `ma1997_fig5.csv` | Ma et al. 1997, Fig. 5 | 3/6/12 terms (coincident), BEM circles | 358 | 0.008; 0.006-0.1 (cap) |
| `ma1997_fig6.csv` | Ma et al. 1997, Fig. 6 | 40/60, 20, 10 steps, BEM circles | 503 | 0.16 dB; 0.14-10 (cap) |
| `ma1997_fig8.csv` | Ma et al. 1997, Fig. 8 | division methods (a), (b), (c), 20 steps | 760 | 0.51 dB; 0.51-3.0 |
| `ma1997_fig9.csv` | Ma et al. 1997, Fig. 9 | (a)=(b)=(c), 60 steps | 449 | 0.38 dB; 0.37-10 (cap) |
| `ma1997_fig11.csv` | Ma et al. 1997, Fig. 11 | c/a = 0, 0.60, 0.631, 0.70, 1.0, 1.30 | 559 | 0.28 dB; 0.26-10 (cap) |
| `ma1997_table1.csv` | Ma et al. 1997, Table 1 | transcribed table (16 GHz, 50/60 steps) | 36 values | half a unit of the last digit |
| `reiter1995_fig3a.csv` | Reiter & Arndt 1995, Fig. 3(a) H-plane bend | γ = 90-150° | 462 | 0.30 dB; 0.30-1.4 |
| `reiter1995_fig3b.csv` | Fig. 3(b) mitered H-plane bend | γ = 105, 120, 135° | 361 | 0.34 dB; 0.26-10 (cap) |
| `reiter1995_fig3c.csv` | Fig. 3(c) E-plane bend | γ = 90-150° | 489 | 0.28 dB; 0.28-1.7 |
| `reiter1995_fig3d.csv` | Fig. 3(d) mitered E-plane bend | γ = 105, 120, 135°, right branch | 351 | 0.25 dB; 0.24-4.6 |
| `reiter1995_fig4.csv` | Reiter & Arndt 1995, Fig. 4 cascaded mitered H+E bends | l = 0, l = b | 334 | 0.50 dB; 0.48-10 (cap) |

Transcriptions (D5-D7) are in `docs/literature/raw/b/`. D8 (Kotik & Sivov 1965) is not available yet.

`scripts/` holds the Python used (numpy, scipy, Pillow, scikit-image). They read page renders
(`pdftoppm -r 300`) from the L1b scratch directory, which are not in the repository (copyright); to rerun, render the
pages named in each CSV header into the parent directory of `scripts/` under the names used in the scripts
(`ales-2.png`, `doane-7.png` cropped as stated, `ma-4.png`, `ma-5.png`, `ma-6.png`, `rei-4.png`).
