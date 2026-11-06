"""Runtime benchmark: reproduces, with wrmodes, the simulation of the COMSOL model
`comsol/hyperbolic_secant_E_bend_60_deg_lowfreq_parametric.java` (Figure 6 of the paper), and times it.

Case (identical to the COMSOL parameters):
  * WR-90 waveguide: short side 0.4 in = 0.01016 m parallel to the bend plane, long side 0.9 in = 0.02286 m
    perpendicular to it; TE10 input mode (electric field parallel to the short side and to the bend plane).
  * 60 degree hyperbolic secant bend, minimum curvature radius r0 = 0.055 m, truncation S = 4, giving
    gamma = 4 atan(tanh(S/2)) / (theta r0) = 53.27 1/m and a bend length L = 2 S / gamma = 0.15017 m
    (the COMSOL geometry runs from z = -S/gamma to +S/gamma).
  * Frequencies 18 to 60 GHz in 2 GHz steps (22 points).
  * Solver settings of the paper: all propagating modes of the coupled family, 10 grid points per guided
    wavelength, tol = 0.5 (no adaptive mesh refinement).

Run:   python runtime_comsol_case.py            (one timed sweep)
       python runtime_comsol_case.py --repeat 5 (best / mean of 5 sweeps)
Compare the printed wall-clock time with the COMSOL run time on the same machine.
"""
import sys, os, time, platform, argparse
import numpy as np
import scipy
import wrmodes as wr

# ----------------------------------------------------------------------------- parameters (COMSOL model)
SIDE_A = 0.4 * 0.0254          # m, parallel to the bend plane  (COMSOL: side_a = wr90_short)
SIDE_B = 0.9 * 0.0254          # m, perpendicular to the bend plane (COMSOL: side_b = wr90_long)
MODE_IN = ('TE', 0, 1)         # TE10 in the usual convention; (m, n) = (0, 1) in the a/b convention of the paper
THETA = np.radians(60.0)       # COMSOL: theta = 60[deg]
R0 = 0.055                     # m, COMSOL: r0 = 0.055[m]
S = 4.0                        # COMSOL: S = 4
GAMMA = 4 / THETA / R0 * np.arctan(np.tanh(S / 2))   # COMSOL: beta = 4/theta/r0*atan(tanh(S/2))
LENGTH = 2 * S / GAMMA                                # bend runs from -S/beta to +S/beta
FREQS = np.arange(18e9, 60e9 + 1, 2e9)                # COMSOL: range(18,2,60) GHz
SETTINGS = dict(maximum_modes=50, max_nodes=1_000_000, resolution='auto', tol=5e-1)


def curvature(z):
    """kappa(z) = sech(gamma (z - L/2)) / r0, z measured from the bend entrance."""
    return 1 / (R0 * np.cosh(GAMMA * (z - LENGTH / 2)))


def sweep(verbose=True):
    """One frequency sweep. Returns (losses in dB, per-point times, number of modes)."""
    losses, times, nmodes = [], [], []
    name_in = f"{MODE_IN[0]}{MODE_IN[1]},{MODE_IN[2]}"
    for f in FREQS:
        t0 = time.perf_counter()
        modes, res = wr.solve_mode_coupling(f, SIDE_A, SIDE_B, MODE_IN, length=LENGTH, curvature=curvature, **SETTINGS)
        dt = time.perf_counter() - t0
        j = [m[0] for m in modes].index(name_in)
        loss = -20 * np.log10(np.abs(res.y[j, -1]))
        losses.append(loss); times.append(dt); nmodes.append(len(modes))
        if verbose:
            print(f"  {f/1e9:5.1f} GHz  {len(modes):2d} modes  {len(res.x):5d} grid points  loss = {loss:8.4f} dB  "
                  f"({dt*1e3:7.1f} ms, solver status {res.status})", flush=True)
    return np.array(losses), np.array(times), np.array(nmodes)


def read_comsol(path):
    lines = open(path).read().splitlines()
    hdr = [l for l in lines if l.startswith('% X')][0]
    f = np.array([float(s.split('freq=')[1].split()[0]) for s in hdr.split('emw.S21dB')[1:]]) * 1e9
    vals = np.array([l for l in lines if not l.startswith('%')][0].split()[3:], dtype=float)
    return f, -vals


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat', type=int, default=1, help='number of timed sweeps (best and mean are reported)')
    args = ap.parse_args()

    print(f"wrmodes runtime benchmark -- COMSOL case of Figure 6")
    print(f"  machine : {platform.node()} / {platform.machine()} / {platform.processor() or platform.platform()}")
    print(f"  python  : {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}")
    print(f"  bend    : 60 deg hyperbolic secant, r0 = {R0*100:.2f} cm, S = {S:g}, gamma = {GAMMA:.3f} 1/m, "
          f"L = {LENGTH*100:.3f} cm")
    print(f"  guide   : a = {SIDE_A*1e3:.2f} mm (in bend plane), b = {SIDE_B*1e3:.2f} mm, input {MODE_IN}")
    print(f"  sweep   : {len(FREQS)} points, {FREQS[0]/1e9:.0f}-{FREQS[-1]/1e9:.0f} GHz, step {(FREQS[1]-FREQS[0])/1e9:.0f} GHz")

    # The library caches the coupling coefficients (lru_cache); the first sweep includes that cost, as COMSOL's
    # run time includes its own setup. Clear the cache before every repetition so each sweep is a cold run.
    from wrmodes.mode_coupling import coupling_coefficient
    totals = []
    for k in range(args.repeat):
        coupling_coefficient.cache_clear()
        print(f"\nsweep {k+1}/{args.repeat}:")
        t0 = time.perf_counter()
        losses, times, nmodes = sweep(verbose=(k == 0))
        total = time.perf_counter() - t0
        totals.append(total)
        print(f"  total wall-clock time: {total:.2f} s  (slowest point {times.max()*1e3:.0f} ms at "
              f"{FREQS[np.argmax(times)]/1e9:.0f} GHz with {nmodes[np.argmax(times)]} modes)")

    print(f"\nRESULT: {len(FREQS)} frequencies in {min(totals):.2f} s (best of {args.repeat})"
          + (f", mean {np.mean(totals):.2f} s" if args.repeat > 1 else ""))
    print(f"        peak loss {losses.max():.4f} dB at {FREQS[np.argmax(losses)]/1e9:.0f} GHz")

    comsol_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'comsol',
                               '15cm_60deg_hyperbolic_curve_side_a_short_e_field_horizontal')
    if os.path.exists(comsol_file):
        fc, lc = read_comsol(comsol_file)
        if len(fc) == len(FREQS) and np.allclose(fc, FREQS):
            d = losses - lc
            print(f"        vs COMSOL S21: max |diff| = {np.max(np.abs(d)):.4f} dB, rms = {np.sqrt(np.mean(d**2)):.4f} dB")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime_comsol_case.txt')
    np.savetxt(out, np.c_[FREQS / 1e9, losses, nmodes, times], header='f [GHz]   loss [dB]   N modes   time [s]',
               fmt=['%6.1f', '%10.5f', '%4d', '%9.4f'])
    print(f"        per-point results written to {out}")
