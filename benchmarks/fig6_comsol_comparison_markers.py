"""Figure 6 of the paper, publication version: coupled-mode theory (wrmodes) vs COMSOL Multiphysics for the two
60 degree WR-90 E-plane bends, with wrmodes evaluated ONLY at the 22 frequencies simulated in COMSOL
(18-60 GHz, 2 GHz steps) and drawn as triangles, so that the figure cannot be read as a densely sampled FEM sweep.

Geometries are exactly those of the COMSOL models (see comsol/):
  * hyperbolic secant: r0 = 0.055 m, S = 4, theta = 60 deg -> L = 2 S / gamma = 15.02 cm
                       (hyperbolic_secant_E_bend_60_deg_lowfreq_parametric.mph)
  * constant curvature: L = 15 cm, theta = 60 deg -> R = 14.32 cm  (constant_E_bend_60_deg_lowfreq_15cm.mph)
  * a = 0.4 in (in the bend plane), b = 0.9 in, TE10 input (E-plane bend).

Output: fig6_comsol_comparison_markers.pdf (and .png) next to this script; difference statistics are printed.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import PAPER_SETTINGS, constant_curvature, mode_conversion_loss_db

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ style (close to the paper figures)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'TeX Gyre Termes', 'Nimbus Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'axes.grid': True, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
})

# ------------------------------------------------------------------ geometry of the COMSOL models
A_SIDE = 0.4 * 0.0254          # m, in the bend plane
B_SIDE = 0.9 * 0.0254          # m
MODE_IN = ('TE', 0, 1)         # TE10 in the usual convention (E-plane bend)
THETA = np.radians(60)

R0_SECH, S = 0.055, 4.0
GAMMA = 4 / THETA / R0_SECH * np.arctan(np.tanh(S / 2))
L_SECH = 2 * S / GAMMA


def kappa_sech(z):
    return 1 / (R0_SECH * np.cosh(GAMMA * (z - L_SECH / 2)))


L_CONST = 0.15
kappa_const = constant_curvature(60, L_CONST)

CASES = [  # (label, COMSOL file, length, curvature, colour)
    ('Constant curvature', 'comsol/15cm_60deg_constant_curve_side_a_short_e_field_horizontal', L_CONST, kappa_const, 'C0'),
    ('Hyperbolic secant curvature', 'comsol/15cm_60deg_hyperbolic_curve_side_a_short_e_field_horizontal', L_SECH, kappa_sech, 'C1'),
]


def read_comsol(fname):
    lines = open(os.path.join(HERE, fname)).read().splitlines()
    hdr = [l for l in lines if l.startswith('% X')][0]
    f = np.array([float(s.split('freq=')[1].split()[0]) for s in hdr.split('emw.S21dB')[1:]]) * 1e9
    vals = np.array([l for l in lines if not l.startswith('%')][0].split()[3:], dtype=float)
    return f, -vals  # mode-conversion loss in dB (positive)


# ------------------------------------------------------------------ compute
results = []
for label, fname, L, kappa, color in CASES:
    f, comsol = read_comsol(fname)
    cmt = np.array([mode_conversion_loss_db(fi, A_SIDE, B_SIDE, MODE_IN, L, kappa) for fi in f])
    d = comsol - cmt
    print(f"{label:28s} L = {L*100:.3f} cm, {len(f)} frequencies: max|COMSOL-CMT| = {np.max(np.abs(d)):.4f} dB "
          f"(rms {np.sqrt(np.mean(d**2)):.4f}, mean {np.mean(d):+.4f}); max loss {comsol.max():.2f} dB")
    results.append((label, f, comsol, cmt, color))

# ------------------------------------------------------------------ plot
fig, (ax, axd) = plt.subplots(2, 1, figsize=(3.4, 3.3), sharex=True,
                              gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.08))
for label, f, comsol, cmt, color in results:
    ax.plot(f / 1e9, cmt, ls='none', marker='^', ms=4.5, mfc='none', mec=color, mew=0.9)
    ax.plot(f / 1e9, comsol, ls='none', marker='x', ms=4.5, color=color, mew=0.9)
    axd.plot(f / 1e9, comsol - cmt, ls='none', marker='+', ms=5, color=color, mew=0.9)

ax.set_ylabel('Mode conversion loss\n[dB]')
axd.set_ylabel('Difference [dB]')
axd.set_xlabel('Frequency (GHz)')
axd.set_ylim(-0.02, 0.02)
axd.set_yticks([-0.02, -0.01, 0, 0.01, 0.02])
ax.set_xlim(16, 62)

# legend: colour = curvature profile, marker = method
handles = [
    Line2D([], [], color='C0', lw=2, label='Constant curvature'),
    Line2D([], [], color='C1', lw=2, label='Hyperbolic secant curvature'),
    Line2D([], [], ls='none', marker='^', mfc='none', mec='0.25', ms=4.5, mew=0.9, label='Our method (wrmodes)'),
    Line2D([], [], ls='none', marker='x', color='0.25', ms=4.5, mew=0.9, label='COMSOL Multiphysics$^{\\circledR}$'),
    Line2D([], [], ls='none', marker='+', color='0.25', ms=5, mew=0.9, label='Difference (COMSOL $-$ our method)'),
]
ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.02), frameon=False, ncol=1,
          handlelength=1.6, borderaxespad=0)

out = os.path.join(HERE, 'fig6_comsol_comparison_markers')
fig.savefig(out + '.pdf', bbox_inches='tight')
fig.savefig(out + '.png', bbox_inches='tight', dpi=200)
print(f"saved {out}.pdf and .png")
