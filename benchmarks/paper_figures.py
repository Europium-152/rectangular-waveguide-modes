"""Figures and table of the JINST paper (revised version) computed with wrmodes >= 0.1.0 (transfer-matrix solver).

Usage:  python paper_figures.py [all | power | convergence | sparc | truncation | length | sequence] [--out DIR]

Figures are written to ./figures (or --out), numerical data to ./results; the numbers quoted in the paper are
printed and saved in results/paper_numbers.json. The convergence figure uses the reference solver of
validation/vcommon.py and, for its panel (c), the results of validation/run_accuracy_suite.py if present.
"""
import os, sys, json, pickle, argparse, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, FormatStrFormatter
import wrmodes as wr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'validation'))
INCH = 0.0254
WR90_BROAD, WR90_NARROW = 0.9 * INCH, 0.4 * INCH
ORIENT = {'TE10 H-plane': (WR90_BROAD, WR90_NARROW, ('TE', 1, 0)), 'TE10 E-plane': (WR90_NARROW, WR90_BROAD, ('TE', 0, 1)),
          'TE01 H-plane': (WR90_NARROW, WR90_BROAD, ('TE', 1, 0)), 'TE01 E-plane': (WR90_BROAD, WR90_NARROW, ('TE', 0, 1))}
COLORS = {'TE01 E-plane': 'C0', 'TE01 H-plane': 'C1', 'TE10 E-plane': 'C2', 'TE10 H-plane': 'C3'}
LABELS = {'TE01 E-plane': r'tall-guide/TE$_{0,1}$ E-plane', 'TE01 H-plane': r'tall-guide/TE$_{0,1}$ H-plane',
          'TE10 E-plane': r'TE$_{1,0}$ E-plane', 'TE10 H-plane': r'TE$_{1,0}$ H-plane'}
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'TeX Gyre Termes', 'Nimbus Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.minor.visible': True, 'ytick.minor.visible': True, 'axes.grid': True, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 300})


def sech(theta, L, T=4.0):
    th = np.radians(theta); g = 2 * T / L; R0 = 4 / th / g * np.arctan(np.tanh(T / 2))
    k = lambda z: 1 / (R0 * np.cosh(g * (np.asarray(z) - L / 2)))
    k.R0 = R0
    return k


def constant(theta, L):
    k0 = np.radians(theta) / L
    return lambda z: np.full(np.shape(z), k0)


def loss_db(x):
    return -20 * np.log10(np.abs(x))


def loss(f, orient, L, kappa, **kw):
    a, b, mi = ORIENT[orient]
    _, r = wr.solve_transfer_matrix(f, a, b, mi, L, kappa, profile=False, **kw)
    return float(loss_db(r.T[r.input_index, r.input_index]))


NUM = {}


def save_numbers(out_res):
    path = os.path.join(out_res, 'paper_numbers.json')
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(NUM)
    json.dump(old, open(path, 'w'), indent=1)


# ------------------------------------------------------------------------------------------- power balance
def fig_power(out):
    a, b, mi = WR90_NARROW, WR90_BROAD, ('TE', 0, 1)
    th = np.radians(60); r0 = 0.055; S = 4; g = 4 / th / r0 * np.arctan(np.tanh(S / 2)); L = 2 * S / g
    k = lambda z: 1 / (r0 * np.cosh(g * (np.asarray(z) - L / 2)))
    z = np.linspace(0, L, 401)
    modes, r = wr.solve_transfer_matrix(60e9, a, b, mi, L, k, z_eval=z, error_control='full')
    N = len(modes)
    Pf, Pb = np.abs(r.y[:N]) ** 2, np.abs(r.y[N:]) ** 2
    tot = Pf.sum(0) - Pb.sum(0)
    NUM['power_balance'] = dict(N=N, max_net_flow_change_along_z=float(np.max(np.abs(tot - tot[0]))),
                                net_flow=float(tot[0]), reflected_power=float(Pb[:, 0].sum()),
                                max_backward_power_inside=float(Pb.sum(0).max()),
                                z_max_backward_cm=float(z[np.argmax(Pb.sum(0))] * 100),
                                loss_dB=float(loss_db(r.T[0, 0])),
                                peak_TM11=float(Pf[[m[0] for m in modes].index('TM1,1')].max()),
                                slices=r.n_slices)
    fig, ax = plt.subplots(2, 1, figsize=(4.2, 3.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]))
    ax[0].plot(z * 100, Pf.sum(0), 'k-', lw=1.5, label=r'$\sum_p |A^+_p|^2$')
    for i, m in enumerate(modes):
        if Pf[i].max() > 1e-4:
            ax[0].plot(z * 100, Pf[i], lw=1, label=m[0].replace('TE', r'TE$_{').replace('TM', r'TM$_{') + r'}$')
    ax[0].set_ylabel('Normalized power'); ax[0].set_ylim(-0.02, 1.02)
    ax[0].legend(frameon=False, fontsize=7, loc='center left', bbox_to_anchor=(1.01, 0.5))
    ax[1].semilogy(z * 100, Pb.sum(0), 'k-', lw=1, label=r'$\sum_p |A^-_p|^2$')
    ax[1].semilogy(z * 100, np.abs(tot - tot[0]) + 1e-17, color='C3', lw=0.8, label=r'$|\Delta \sum_p(|A^+_p|^2-|A^-_p|^2)|$')
    ax[1].set_ylim(1e-17, 1e-3); ax[1].set_ylabel('Power')
    ax[1].yaxis.set_major_locator(FixedLocator([1e-15, 1e-11, 1e-7, 1e-3]))
    ax[1].legend(frameon=False, fontsize=7, loc='center left', bbox_to_anchor=(1.01, 0.5))
    ax[1].set_xlabel('Position along the bend $z$ [cm]')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'power_balance.pdf'), bbox_inches='tight')
    print('power balance', NUM['power_balance'])


# ------------------------------------------------------------------------------------------- convergence
def fig_convergence(out, out_res):
    from vcommon import reference
    from wrmodes.transfer_matrix import coupled_mode_matrices, _slice_exponentials, _product, _scattering_from_transfer
    L, k = 0.6, sech(60, 0.6)
    cases = [('TE10 H-plane', 'o'), ('TE10 E-plane', 's'), ('TE01 E-plane', '^')]
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.35))
    trunc = {}
    # (a) modal truncation
    for c, mk in cases:
        a, b, mi = ORIENT[c]
        for f, ls in [(60e9, '--'), (89.5e9, '-')]:
            modes, rfull = wr.solve_transfer_matrix(f, a, b, mi, L, k, profile=False, atol=1e-10)
            full = loss_db(rfull.T[rfull.input_index, rfull.input_index])
            Ns, err = [], []
            for N in range(1, len(modes) + 1):
                _, r = wr.solve_transfer_matrix(f, a, b, mi, L, k, profile=False, atol=1e-10, maximum_modes=N)
                Ns.append(N); err.append(abs(loss_db(r.T[r.input_index, r.input_index]) - full))
            err = np.array(err); err[err == 0] = np.nan
            trunc[(c, f)] = (Ns, err)
            axs[0].semilogy(Ns, err, marker=mk, ms=3, ls=ls, lw=0.8, color=COLORS[c], label=f"{LABELS[c]}, {f/1e9:g} GHz")
    axs[0].set_xlabel('Number of retained modes $N$'); axs[0].set_ylabel(r'$|\Delta$ loss$|$ w.r.t. all modes [dB]')
    axs[0].set_ylim(1e-8, 10)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7,
               handlelength=2.5, columnspacing=1.2)
    axs[0].set_title('a) modal truncation', loc='left')
    # numbers: modes needed for 0.01 dB, effect of the last (near-cutoff) modes
    need = {}
    for (c, f), (Ns, err) in trunc.items():
        e = np.nan_to_num(err, nan=0.0)
        ok = [N for N, x in zip(Ns, e) if np.all(e[Ns.index(N):] <= 0.01)]
        need[f"{c} {f/1e9:g} GHz"] = dict(N_for_0p01dB=int(ok[0]) if ok else None, N_full=int(Ns[-1]),
                                          last_three_modes_max_dB=float(np.max(e[-4:-1])) if len(e) > 3 else 0.0)
    NUM['modal_truncation'] = need
    # (b) error vs number of slices (fixed slicing, no adaptivity)
    conv = {}
    for c, mk in cases:
        a, b, mi = ORIENT[c]
        for f, ls in [(60e9, '--'), (89.5e9, '-')]:
            modes, beta, T_ref, R_ref, _ = reference(f, a, b, mi[2], L, k, (), False)
            j = [m[0] for m in modes].index(f"TE{mi[1]},{mi[2]}")
            D, K, _ = coupled_mode_matrices(f, a, b, modes)
            ns, ep = [8, 16, 32, 64, 128, 256, 512, 1024], []
            for n in ns:
                T, R = _scattering_from_transfer(_product(_slice_exponentials(D, K, k, np.linspace(0, L, n + 1))), len(modes))
                ep.append(np.max(np.abs(np.abs(T[:, j]) ** 2 - np.abs(T_ref[:, j]) ** 2)))
            _, rad = wr.solve_transfer_matrix(f, a, b, mi, L, k, profile=False)
            conv[(c, f)] = dict(n=ns, err=ep, adaptive_slices=rad.n_slices)
            axs[1].loglog(ns, ep, marker=mk, ms=3, ls=ls, lw=0.8, color=COLORS[c])
            axs[1].plot([rad.n_slices], [ep[ns.index(rad.n_slices)]], marker=mk, ms=7, mfc='none', mec='k', mew=0.8,
                        ls='none')
    nn = np.array([16, 1024]); axs[1].loglog(nn, 2e-1 * (nn / 16.) ** -4, 'k:', lw=0.8)
    axs[1].text(60, 2.5e-4, r'$\propto n^{-4}$', fontsize=7)
    axs[1].axhline(1e-6, color='0.4', lw=0.6, ls='--')
    axs[1].set_xlabel('Number of slices $n$'); axs[1].set_ylabel('Max. error of output power')
    axs[1].set_title('b) slicing', loc='left'); axs[1].set_ylim(1e-13, 3)
    axs[1].xaxis.set_major_locator(FixedLocator([8, 32, 128, 512])); axs[1].xaxis.set_major_formatter(FormatStrFormatter('%g'))
    axs[1].xaxis.set_minor_formatter(NullFormatter())
    NUM['slicing'] = {f"{c} {f/1e9:g} GHz": dict(adaptive_slices=v['adaptive_slices'],
                                                 err_at_adaptive=float(v['err'][v['n'].index(v['adaptive_slices'])]))
                      for (c, f), v in conv.items()}
    # (c) estimated vs actual error over the validation test matrix
    path = os.path.join(HERE, 'validation', 'results', 'accuracy_suite.pkl')
    if os.path.exists(path):
        res = [r for r in pickle.load(open(path, 'rb')) if r['profile'] != 'step (no breakpoints)']
        est = np.array([max(r['est'], 1e-16) for r in res]); act = np.array([max(r['err_P'], 1e-16) for r in res])
        conv_ok = np.array([r['status'] == 0 for r in res])
        axs[2].axhspan(1e-16, 1e-9, color='0.88', lw=0, zorder=0)
        axs[2].text(5e-5, 3e-16, 'below the accuracy\nof the reference', fontsize=6.5, color='0.35', ha='right', va='bottom')
        axs[2].loglog(est[conv_ok], act[conv_ok], '.', ms=2, color='C0', alpha=0.6, rasterized=True)
        xx = np.array([1e-16, 1e-4]); axs[2].loglog(xx, xx, 'k-', lw=0.6)
        axs[2].axhline(1e-6, color='0.4', lw=0.6, ls='--')
        axs[2].axvline(1e-6, color='0.4', lw=0.6, ls='--')
        axs[2].set_xlim(1e-16, 1e-4); axs[2].set_ylim(1e-16, 1e-4)
        axs[2].set_xlabel('Estimated error'); axs[2].set_ylabel('Actual error of output power')
        axs[2].set_title(f'c) {len(res)} test cases', loc='left')
        for ax in [axs[2]]:
            ax.xaxis.set_major_locator(FixedLocator([1e-16, 1e-12, 1e-8, 1e-4])); ax.yaxis.set_major_locator(FixedLocator([1e-16, 1e-12, 1e-8, 1e-4]))
    fig.tight_layout(w_pad=0.6); fig.savefig(os.path.join(out, 'convergence.pdf'), bbox_inches='tight')
    print('modal truncation', json.dumps(need, indent=1)); print('slicing', NUM['slicing'])


# ------------------------------------------------------------------------------------------- SPARC bends + table
def fig_sparc(out, out_res):
    freqs = np.arange(18e9, 90.01e9, 0.5e9)
    data = {}
    t0 = time.perf_counter()
    for bend, (th, L) in {'60cm_60deg': (60, 0.6), '40cm_90deg': (90, 0.4)}.items():
        for p in ['constant', 'sech']:
            k = constant(th, L) if p == 'constant' else sech(th, L)
            for c in ORIENT:
                data[(bend, p, c)] = np.array([loss(f, c, L, k) for f in freqs])
    NUM['sparc_sweep_time_s'] = time.perf_counter() - t0
    np.savez(os.path.join(out_res, 'sparc_bends.npz'), freqs=freqs, **{'/'.join(k): v for k, v in data.items()})

    def intervals(l, thr):
        above = l > thr; segs = []; i = 0
        while i < len(freqs):
            if above[i]:
                j = i
                while j + 1 < len(freqs) and above[j + 1]: j += 1
                segs.append((freqs[i] / 1e9, freqs[j] / 1e9)); i = j + 1
            else: i += 1
        return segs

    def fmt(segs):
        if not segs: return '--'
        return ', '.join((f"{a:.0f}--{b:.0f}" if b < 90 else f"$>{a:.0f}$") if round(a) != round(b) else f"{a:.0f}" for a, b in segs)

    rows = [r'\begin{table}[t]', r'\centering',
            r"\caption{Peak mode-conversion loss between 18 and 90~GHz for the sixteen curves of figure~\ref{fig:four-bends-losses}, and the frequency intervals (GHz) over which the loss exceeds 0.5, 1 and 3~dB. Intervals are given with a 0.5~GHz resolution; ``$>f$'' means from $f$ up to 90~GHz.}",
            r'\label{tab:peak-losses}', r'\small', r'\begin{tabular}{llrrlll}', r'\hline',
            r'Bend & Orientation & Peak [dB] & at [GHz] & $>0.5$~dB & $>1$~dB & $>3$~dB \\', r'\hline']
    peaks = {}
    for bnd, bname in [('60cm_60deg', r'60~cm, $60\degree$'), ('40cm_90deg', r'40~cm, $90\degree$')]:
        for p, pname in [('constant', 'constant'), ('sech', 'hyp. secant')]:
            for i, c in enumerate(['TE10 H-plane', 'TE10 E-plane', 'TE01 H-plane', 'TE01 E-plane']):
                l = data[(bnd, p, c)]
                peaks[f"{bnd} {p} {c}"] = dict(peak=float(l.max()), at=float(freqs[np.argmax(l)] / 1e9),
                                                above1=intervals(l, 1.0))
                pk = f"{l.max():.2f}" if l.max() >= 0.01 else f"{l.max():.3f}"
                rows.append(f"{bname + ', ' + pname if i == 0 else ''} & {LABELS[c]} & {pk} & {freqs[np.argmax(l)]/1e9:.1f} & "
                            f"{fmt(intervals(l, 0.5))} & {fmt(intervals(l, 1.0))} & {fmt(intervals(l, 3.0))} \\\\")
            rows.append(r'\hline')
    rows += [r'\end{tabular}', r'\end{table}']
    open(os.path.join(out, 'table_peak_losses.tex'), 'w').write('\n'.join(rows) + '\n')
    NUM['sparc_peaks'] = peaks
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 4.6), sharex=True, sharey=True)
    for ax, (bnd, p, title) in zip(axs.ravel(), [('60cm_60deg', 'constant', 'a) 60 cm 60° constant'),
                                                ('40cm_90deg', 'constant', 'b) 40 cm 90° constant'),
                                                ('60cm_60deg', 'sech', 'c) 60 cm 60° hyperbolic secant'),
                                                ('40cm_90deg', 'sech', 'd) 40 cm 90° hyperbolic secant')]):
        for c in ['TE01 E-plane', 'TE01 H-plane', 'TE10 E-plane', 'TE10 H-plane']:
            ax.plot(freqs / 1e9, data[(bnd, p, c)], color=COLORS[c], lw=1.2, label=LABELS[c])
        ax.axhline(1.0, color='k', ls='--', lw=0.7); ax.set_title(title, loc='left'); ax.set_ylim(0, 3.5); ax.set_xlim(18, 90)
    for ax in axs[1]: ax.set_xlabel('Frequency [GHz]')
    for ax in axs[:, 0]: ax.set_ylabel('Mode conversion loss [dB]')
    axs[0, 0].legend(frameon=False, loc='upper left', fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, '60_and_90_bends_mode_loss_v3.pdf'), bbox_inches='tight')
    print('SPARC sweep (1160 points) in %.1f s' % NUM['sparc_sweep_time_s'])


# ------------------------------------------------------------------------------------------- truncation T
def fig_truncation(out):
    freqs = np.arange(18e9, 90.01e9, 1e9)
    configs = [('T=3, L=60 cm', 3, 0.60, dict(color='C0', ls='-')), ('T=4, L=60 cm', 4, 0.60, dict(color='k', ls='-', lw=1.6)),
               ('T=5, L=60 cm', 5, 0.60, dict(color='C3', ls='-')), ('T=4, L=66 cm', 4, 0.66, dict(color='k', ls='--')),
               ('T=4, L=54 cm', 4, 0.54, dict(color='k', ls=':'))]
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.1), sharex=True)
    res = {}
    for c in ORIENT:
        for label, T, L, st in configs:
            k = sech(60, L, T)
            res[(c, label)] = (np.array([loss(f, c, L, k) for f in freqs]), k.R0)
    for ax, c in zip(axs, ['TE10 H-plane', 'TE10 E-plane', 'TE01 E-plane']):
        for label, T, L, st in configs:
            l, R0 = res[(c, label)]
            ax.plot(freqs / 1e9, l, lw=st.get('lw', 1), color=st['color'], ls=st['ls'],
                    label=f"$T={T}$, $L={L*100:.0f}$ cm ($R_0={R0*100:.0f}$ cm)")
        ax.set_title(LABELS[c], loc='left'); ax.set_xlabel('Frequency [GHz]'); ax.set_xlim(18, 90)
    axs[0].set_ylabel('Mode conversion loss [dB]')
    axs[0].set_ylim(0, 1.0); axs[1].set_ylim(0, 0.3); axs[2].set_ylim(0, 8)
    axs[1].legend(frameon=False, fontsize=6.5, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'truncation_sensitivity.pdf'), bbox_inches='tight')
    NUM['truncation'] = {f"{c} | {lab}": float(v[0].max()) for (c, lab), v in res.items()}
    print('truncation peaks', json.dumps(NUM['truncation'], indent=1))


# ------------------------------------------------------------------------------------------- 90 deg bend length
def fig_length(out):
    freqs = np.arange(18e9, 90.01e9, 1e9)
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.3), sharex=True, sharey=True)
    res = {}
    for ax, c in zip(axs, ['TE10 H-plane', 'TE10 E-plane']):
        for L, col in zip([0.40, 0.50, 0.60, 0.70, 0.80], ['C3', 'C1', 'k', 'C2', 'C0']):
            l = np.array([loss(f, c, L, sech(90, L)) for f in freqs])
            res[f"{c} L={L*100:.0f} cm"] = dict(peak=float(l.max()), at=float(freqs[np.argmax(l)] / 1e9),
                                                first_above_1dB=float(freqs[l > 1][0] / 1e9) if np.any(l > 1) else None)
            ax.plot(freqs / 1e9, l, color=col, lw=1, label=f'$L={L*100:.0f}$ cm')
        ax.axhline(1, color='k', ls='--', lw=0.7)
        ax.set_title(LABELS[c] + r', $90\degree$ hyperbolic secant', loc='left'); ax.set_xlabel('Frequency [GHz]'); ax.set_xlim(18, 90)
    axs[0].set_ylabel('Mode conversion loss [dB]'); axs[0].set_ylim(0, 2.5)
    axs[0].legend(frameon=False, fontsize=7, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'bend_length_90deg.pdf'), bbox_inches='tight')
    NUM['bend_length'] = res
    print('bend length', json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------- sequence of bends
def fig_sequence(out):
    freqs = np.arange(18e9, 90.01e9, 0.5e9)
    rng = np.random.default_rng(1)
    NS, L1, L2, dL = 96, 10.0, 1.0, 0.5
    res = {}
    for c in ['TE10 H-plane', 'TE10 E-plane']:
        a, b, mi = ORIENT[c]
        rec = {k: [] for k in ['l60', 'l90', 'sum', 'mean', 'lo', 'hi']}
        for f in freqs:
            modes, T60, _, beta = wr.transmission_matrix(f, a, b, mi[2], 0.6, sech(60, 0.6))
            _, T90, _, _ = wr.transmission_matrix(f, a, b, mi[2], 0.4, sech(90, 0.4))
            k90 = sech(90, 0.4)
            _, T90m, _, _ = wr.transmission_matrix(f, a, b, mi[2], 0.4, lambda z: -k90(z))
            j = [m[0] for m in modes].index(f"TE{mi[1]},{mi[2]}")
            e = np.zeros(len(modes), complex); e[j] = 1
            ens = []
            for _ in range(NS):
                A = T90m @ wr.straight_section(beta, L2 + rng.uniform(-dL, dL)) @ T90 @ \
                    wr.straight_section(beta, L1 + rng.uniform(-dL, dL)) @ T60 @ e
                ens.append(loss_db(A[j]))
            ens = np.array(ens)
            l60, l90, l90m = loss_db(T60[j, j]), loss_db(T90[j, j]), loss_db(T90m[j, j])
            for kk, v in [('l60', l60), ('l90', l90), ('sum', l60 + l90 + l90m), ('mean', ens.mean()),
                          ('lo', ens.min()), ('hi', ens.max())]:
                rec[kk].append(v)
        res[c] = {k: np.array(v) for k, v in rec.items()}
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.3), sharex=True)
    for ax, c in zip(axs, ['TE10 H-plane', 'TE10 E-plane']):
        r = res[c]; f = freqs / 1e9
        ax.fill_between(f, r['lo'], r['hi'], color=COLORS[c], alpha=0.25, lw=0, label='coherent cascade, envelope over straight-section lengths')
        ax.plot(f, r['mean'], color=COLORS[c], lw=1.2, label='coherent cascade, average')
        ax.plot(f, r['sum'], 'k--', lw=1, label='sum of individual bend losses (dB)')
        ax.plot(f, r['l60'], color='0.5', lw=0.7, ls=':', label=r'single $60\degree$ bend')
        ax.plot(f, r['l90'], color='0.5', lw=0.7, ls='-.', label=r'single $90\degree$ bend')
        ax.set_title(LABELS[c], loc='left'); ax.set_xlabel('Frequency [GHz]'); ax.set_xlim(18, 90); ax.set_ylim(0, 2.0)
    axs[0].set_ylabel('Mode conversion loss [dB]')
    axs[1].legend(frameon=False, fontsize=6, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'bend_sequence.pdf'), bbox_inches='tight')
    nums = {}
    for c, r in res.items():
        d = {}
        for thr in [0.5, 1.0]:
            sel = r['sum'] < thr
            d[f"sum<{thr}"] = dict(up_to_GHz=float(freqs[sel].max() / 1e9), max_mean_minus_sum=float(np.max(np.abs(r['mean'] - r['sum'])[sel])),
                                   max_hi_over_sum=float(np.max((r['hi'] / r['sum'])[sel & (r['sum'] > 0.05)])),
                                   max_hi=float(np.max(r['hi'][sel])))
        i90 = np.argmin(abs(freqs - 90e9))
        d['at_90GHz'] = dict(sum=float(r['sum'][i90]), lo=float(r['lo'][i90]), hi=float(r['hi'][i90]), l90=float(r['l90'][i90]))
        d['max_hi_above_80GHz'] = float(r['hi'][freqs >= 80e9].max())
        nums[c] = d
    NUM['sequence'] = nums
    print('sequence', json.dumps(nums, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', nargs='*', default=['all'])
    ap.add_argument('--out', default=os.path.join(HERE, 'figures'))
    args = ap.parse_args()
    out_res = os.path.join(HERE, 'results')
    os.makedirs(args.out, exist_ok=True); os.makedirs(out_res, exist_ok=True)
    w = set(args.what)
    if w & {'all', 'power'}: fig_power(args.out)
    if w & {'all', 'convergence'}: fig_convergence(args.out, out_res)
    if w & {'all', 'sparc'}: fig_sparc(args.out, out_res)
    if w & {'all', 'truncation'}: fig_truncation(args.out)
    if w & {'all', 'length'}: fig_length(args.out)
    if w & {'all', 'sequence'}: fig_sequence(args.out)
    save_numbers(out_res)
