"""Gate B: compare wrmodes with COMSOL campaign 2 (benchmarks/comsol/CAMPAIGN_2.md).

    python benchmarks/comsol/campaign_2/compare_campaign_2.py [E1 E2 E3 E4 E5] [--raw DIR] [--out results.md]

COMSOL exports (default folder raw/ next to this script, Daniel's `results_files/`; the models in java/) are global
S-parameters of rectangular ports, exported as in campaign 1 (header '% X Y emw.S11 (1) @ freq=...', one data row) or
as a Global Evaluation table (campaign 2: optional parameter columns such as mesh_factor and L, then 'freq (GHz)',
then 'S11 (1)', ..., 'S10_3 (1)' ...). Parameter sweeps come back keyed by their parameter values. E2 is one sweep
file (mesh_factor 10/20 = coarse/fine, L = 0.1/0.2 m; ports 1-2 narrow end TE10/TE30, 3-4 wide end TE10/TE30); E5
the mesh sweeps; E4 has ports for every propagating mode at both ends, laid out as in its .java (`e4_ports`). A
missing file is skipped with a message.

Conventions: COMSOL and wrmodes both use exp(+j omega t), so complex values compare directly once the port
distances are de-embedded: an entry S_ij at the reference planes becomes S_ij exp(-i beta_i d_i) exp(-i beta_j d_j)
at the ports (d the distance from reference plane to port, beta of the port mode). Phases are compared modulo
180 degrees (COMSOL chooses the sign of each port mode).

Criterion (Gate A, CONVENTIONS.md section 8): |wrmodes - COMSOL finest| <= (COMSOL mesh-check change) + 0.02 dB on
the transmission of the incident mode, where the mesh-check change is |coarse - fine| of the case (if no fine run
exists, the 0.02 dB alone). E4 is judged on the power of every output mode (fraction of the input):
|P_wrmodes - P_COMSOL| <= (mesh-check change of that power) + 1e-4.

wrmodes values come from the merged components (that is the comparison, not a reference): E1 `fem2d.Miter` and
`MiterMM` (E plane), E2 `Taper`, E3 `StrangledBend`, E4 `Twist`, E5 the single-domain chained outline
(`fem2d.chained_miter_outline`) and the `Line` cascade [Miter(45), Straight(s), Miter(45)]. For E4 the WP7b
helical-FEM reference (benchmarks/validation_v2/data/twist_doane_<profile>.npz) is shown as well when present.
E1 and E5 need the 'fem' extra (scikit-fem, gmsh).

Each case function takes `model(case, variant, f_GHz) -> {'S11': ..., 'S21': ..., ...}` (values at the reference
planes, in the COMSOL port order); the default `wrmodes_model` computes them. Tests pass a stub.
"""
import argparse
import pathlib
import re

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
REPO = HERE.parents[2]
C0 = 299792458.0
W, H = 22.86e-3, 10.16e-3
A187, B187 = 22.15e-3, 47.55e-3
SIGMA_AL = 3.774e7          # COMSOL's library aluminium (as campaign 1)
DB_TOL = 0.02               # CONVENTIONS.md section 8, against COMSOL
P_TOL = 1e-4                # CONVENTIONS.md section 8, power per converted mode


# ------------------------------------------------------------------------------------------------ reading
def read(path):
    """{f_GHz: {'S11': complex, ...}} from a COMSOL export (global-evaluation row or table)."""
    lines = pathlib.Path(path).read_text().splitlines()
    hdr = [l for l in lines if l.startswith("%")]
    data = [l for l in lines if l.strip() and not l.startswith("%")]
    out = {}
    head = next((l for l in hdr if "@ freq=" in l), None)
    if head is not None:                     # one row, columns 'emw.Sij (1) @ freq=F'
        cols = re.findall(r"emw\.(S\d\d) \(1\) @ freq=([\d.eE+-]+)", head)
        row = data[0].split()
        vals = [complex(s.replace("i", "j")) for s in row[len(row) - len(cols):]]
        for (q, f), v in zip(cols, vals):
            out.setdefault(float(f), {})[q] = v
        return dict(sorted(out.items()))
    head = hdr[-1]                           # table: optional parameter columns, frequency, one column per Sij
    names = re.findall(r"(?:emw\.)?(S[\d_]+) \(1\)", head)
    lead = re.split(r"\s{2,}", head.lstrip("% ").strip())[: -len(names)]   # e.g. ['mesh_factor', 'L (m)', 'freq (GHz)']
    fcol = next(i for i, c in enumerate(lead) if c.startswith("freq"))
    unit = 1.0 if "GHz" in lead[fcol] else 1e-9
    for l in data:
        row = l.split()
        f = round(float(row[fcol]) * unit, 9)
        vals = dict(zip(names, (complex(x.replace("i", "j")) for x in row[len(lead):])))
        params = tuple(float(row[i]) for i in range(len(lead)) if i != fcol)
        if params:
            out.setdefault(params, {})[f] = vals
        else:
            out[f] = vals
    return {k: dict(sorted(v.items())) if isinstance(v, dict) and k and isinstance(k, tuple) else v
            for k, v in sorted(out.items())}


def sweep(data, **fixed):
    """Select one parameter combination of a parameter-sweep export: sweep(d, p0=10, p1=0.1) by position."""
    for params, rows in data.items():
        if all(abs(params[int(k[1:])] - v) < 1e-12 for k, v in fixed.items()):
            return rows
    return None


def load(raw, name, log):
    p = pathlib.Path(raw) / name
    if not p.exists():
        msg = f"[skip] {name}: not found in {raw}"
        print(msg, flush=True)
        log.append(msg)
        return None
    return read(p)


# ------------------------------------------------------------------------------------------------ helpers
def db(x):
    return 20 * np.log10(np.maximum(abs(x), 1e-300))


def beta(mode, a, b, f_ghz):
    """Propagation constant (1/m) of mode 'TEm,n' / 'TMm,n' in an a x b guide (a along u)."""
    m, n = (int(x) for x in mode[2:].split(","))
    k = 2 * np.pi * f_ghz * 1e9 / C0
    return np.sqrt(complex(k ** 2 - (np.pi * m / a) ** 2 - (np.pi * n / b) ** 2)).real


def deembed(s, beta_i, d_i, beta_j, d_j):
    """Shift an entry S_ij from the reference planes to ports d_i, d_j (m) beyond them."""
    return s * np.exp(-1j * beta_i * d_i) * np.exp(-1j * beta_j * d_j)


def phase_mod180(z):
    return (np.degrees(np.angle(z)) + 90) % 180 - 90


def judge_db(w, c, c_coarse):
    """(delta dB vs finest, allowed, pass)."""
    d = db(w) - db(c)
    mesh = abs(db(c_coarse) - db(c)) if c_coarse is not None else 0.0
    return d, mesh + DB_TOL, abs(d) <= mesh + DB_TOL


def table(out, title, head):
    out.append(f"\n### {title}\n")
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))


# ------------------------------------------------------------------------------------------------ wrmodes model
def wrmodes_model(case, variant, f_ghz):
    """wrmodes values at the reference planes, keyed like the COMSOL export (port order of CAMPAIGN_2.md)."""
    import wrmodes as wr
    from wrmodes.components import Guide
    from wrmodes.modes import Mode
    f = f_ghz * 1e9
    te10 = Mode("TE", 1, 0)
    if case == "E1":
        chamfer = None if variant.startswith("plain") else 0.87
        if variant.endswith("MM"):
            comp = wr.MiterMM(chamfer, plane="E")
        else:
            from wrmodes.fem2d import Miter
            comp = Miter(90, chamfer=chamfer, plane="E")
        g = wr.Line(Guide(W, H), [comp]).gsm(f)
        i = g.basis1.index(te10)
        return {"S11": g.S11[i, i], "S21": g.S21[i, i]}
    if case == "E2":
        L = 0.1 if variant == "L100" else 0.2
        g = wr.Taper(L, 45e-3, H, profile="linear", guide=Guide(W, H)).gsm(f)
        i = g.basis1.index(te10)
        return {"S11": g.S11[i, i], "S21": g.S21[g.basis2.index(te10), i],
                "S31": g.S21[g.basis2.index(Mode("TE", 3, 0)), i]}
    if case == "E3":
        guide = Guide(W, H, sigma=SIGMA_AL) if variant == "aluminium" else Guide(W, H)
        g = wr.Line(guide, [wr.StrangledBend(90, 0.05, 1.6e-3, 0.05)]).gsm(f)
        i = g.basis1.index(te10)
        return {"S11": g.S11[i, i], "S21": g.S21[i, i]}
    if case == "E4":
        g = wr.Twist(90, 0.508, profile=variant, guide=Guide(A187, B187)).gsm(f)
        i = g.basis1.index(te10)
        return {"out": {n: float(abs(g.S21[j, i]) ** 2) for j, n in enumerate(g.basis2.names)},
                "in": {n: float(abs(g.S11[j, i]) ** 2) for j, n in enumerate(g.basis1.names)}}
    if case == "E5":
        s = (2 if variant.startswith("s2a") else 3) * W
        from wrmodes.fem2d import Coplanar2D, Miter, chained_miter_outline
        guide = Guide(W, H)
        if variant.endswith("cascade"):
            # propagating modes only: with evanescent modes the 45 deg optical miter's entries at its reference
            # planes (22.9 mm inside the mirror) make the cascade ill-conditioned (Gate B finding, see REPORT)
            g = wr.Line(guide, [Miter(45, plane="H"), wr.Straight(s), Miter(45, plane="H")]).gsm(f, cutoff_factor=1.0)
        else:
            out_, mirror, length = chained_miter_outline([45, 45], [s], a=W)
            g = Coplanar2D(out_, b=H, plane="H", guide=guide, mirror=mirror, length=length).gsm(f)
        i = g.basis1.index(te10)
        return {"S11": g.S11[i, i], "S21": g.S21[i, i]}
    raise ValueError(case)


# ------------------------------------------------------------------------------------------------ cases
def e1(raw, out, model=wrmodes_model, log=None):
    log = [] if log is None else log
    table(out, "E1: E-plane 90° miters, WR-90, ports 15 mm from the reference planes",
          ["miter", "f (GHz)", "COMSOL \\|S21\\| (dB)", "COMSOL fine", "FEM (dB)", "Δ FEM", "MM (dB)", "Δ MM",
           "allowed", "pass", "\\|S11\\| COMSOL / FEM / MM", "Δ∠S21 FEM (°, mod 180)"])
    verdicts = []
    for variant, fn in (("plain", "E1_miter_E_plain"), ("d087", "E1_miter_E_d087")):
        c, cf = load(raw, fn + ".txt", log), load(raw, fn + "_fine.txt", log)
        if c is None:
            continue
        for f in c:
            b = beta("TE1,0", W, H, f)
            ref = cf[f] if cf and f in cf else c[f]
            coarse = c[f] if cf and f in cf else None
            res = {}
            for tag in ("FEM", "MM"):
                try:
                    m = model("E1", f"{variant}_{tag}", f)
                    res[tag] = {k: deembed(v, b, 15e-3, b, 15e-3) for k, v in m.items()}
                except ImportError as e:
                    res[tag] = None
                    log.append(f"[skip] E1 {tag}: {e}")
            cells, ok = [], True
            for tag in ("FEM", "MM"):
                if res[tag] is None:
                    cells += ["–", "–"]
                    continue
                d, allowed, p = judge_db(res[tag]["S21"], ref["S21"], coarse["S21"] if coarse else None)
                ok &= p
                cells += [f"{db(res[tag]['S21']):.4f}", f"{d:+.4f}"]
                verdicts.append(("E1", variant, tag, f, d, allowed, p))
            fine = f"{db(cf[f]['S21']):.4f}" if cf and f in cf else "–"
            s11 = " / ".join(f"{abs(x['S11']):.4f}" for x in (ref, res["FEM"] or {"S11": np.nan}, res["MM"] or {"S11": np.nan}))
            ph = f"{phase_mod180(res['FEM']['S21'] / ref['S21']):+.2f}" if res["FEM"] else "–"
            out.append(f"| {variant} | {f:g} | {db(c[f]['S21']):.4f} | {fine} | " + " | ".join(cells)
                       + f" | {(abs(db(coarse['S21']) - db(ref['S21'])) if coarse else 0) + DB_TOL:.4f} | {'yes' if ok else 'NO'} | {s11} | {ph} |")
    return verdicts


def e2(raw, out, model=wrmodes_model, log=None):
    log = [] if log is None else log
    table(out, "E2: linear H-plane taper 22.86 → 45 mm, 15 GHz, ports 15 mm from the reference planes",
          ["L", "COMSOL \\|S21\\| (dB)", "COMSOL fine", "wrmodes", "Δ (dB)", "allowed", "pass",
           "P(TE30) COMSOL", "wrmodes", "ΔP", "\\|S11\\| COMSOL / wrmodes"])
    verdicts = []
    swp = load(raw, "E2_taper_parameter_sweep.txt", log)   # columns mesh_factor, L (m); ports 1-2 narrow, 3-4 wide
    for variant in ("L100", "L200"):
        if swp is None:
            break
        L = 0.1 if variant == "L100" else 0.2
        remap = lambda rows: None if rows is None else {f: {"S11": r["S11"], "S21": r["S31"], "S31": r["S41"]}
                                                        for f, r in rows.items()}
        c, cf = remap(sweep(swp, p0=10, p1=L)), remap(sweep(swp, p0=20, p1=L))
        if c is None:
            continue
        for f in c:
            m = model("E2", variant, f)
            b1, b2, b3 = beta("TE1,0", W, H, f), beta("TE1,0", 45e-3, H, f), beta("TE3,0", 45e-3, H, f)
            w = {"S11": deembed(m["S11"], b1, 15e-3, b1, 15e-3), "S21": deembed(m["S21"], b2, 15e-3, b1, 15e-3),
                 "S31": deembed(m["S31"], b3, 15e-3, b1, 15e-3)}
            ref = cf[f] if cf and f in cf else c[f]
            coarse = c[f] if cf and f in cf else None
            d, allowed, p = judge_db(w["S21"], ref["S21"], coarse["S21"] if coarse else None)
            dp = abs(w["S31"]) ** 2 - abs(ref["S31"]) ** 2
            mesh_p = abs(abs(coarse["S31"]) ** 2 - abs(ref["S31"]) ** 2) if coarse else 0.0
            p3 = abs(dp) <= mesh_p + P_TOL
            verdicts.append(("E2", variant, "TE10", f, d, allowed, p))
            verdicts.append(("E2", variant, "TE30", f, dp, mesh_p + P_TOL, p3))
            fine = f"{db(cf[f]['S21']):.4f}" if cf and f in cf else "–"
            out.append(f"| {variant} | {db(c[f]['S21']):.4f} | {fine} | {db(w['S21']):.4f} | {d:+.4f} | {allowed:.4f} "
                       f"| {'yes' if p and p3 else 'NO'} | {abs(ref['S31']) ** 2:.5f} | {abs(w['S31']) ** 2:.5f} "
                       f"| {dp:+.1e} | {abs(ref['S11']):.4f} / {abs(w['S11']):.4f} |")
    return verdicts


def e3(raw, out, model=wrmodes_model, log=None):
    log = [] if log is None else log
    table(out, "E3: strangled E-plane bend (WR-90 → 1.6 mm, 90°, R = 50 mm), ports 15 mm from the reference planes",
          ["walls", "f (GHz)", "COMSOL \\|S21\\| (dB)", "COMSOL fine", "wrmodes", "Δ (dB)", "allowed", "pass",
           "\\|S11\\| COMSOL / wrmodes", "Δ∠S21 (°, mod 180)"])
    verdicts = []
    pec, pec_fine = load(raw, "E3_strangled_PEC.txt", log), load(raw, "E3_strangled_PEC_fine.txt", log)
    for variant in ("PEC", "aluminium"):
        c = pec if variant == "PEC" else load(raw, "E3_strangled_aluminum.txt", log)
        if c is None:
            continue
        for f in c:
            b = beta("TE1,0", W, H, f)
            m = model("E3", variant, f)
            w = {k: deembed(v, b, 15e-3, b, 15e-3) for k, v in m.items()}
            checked = pec is not None and pec_fine is not None and f in pec and f in pec_fine
            # the PEC mesh check bounds both models (same mesh); the finest PEC run is the PEC reference
            mesh = abs(db(pec[f]["S21"]) - db(pec_fine[f]["S21"])) if checked else 0.0
            ref = pec_fine[f] if (variant == "PEC" and checked) else c[f]
            d = db(w["S21"]) - db(ref["S21"])
            p = abs(d) <= mesh + DB_TOL
            verdicts.append(("E3", variant, "TE10", f, d, mesh + DB_TOL, p))
            fine = f"{db(pec_fine[f]['S21']):.4f}" if (variant == "PEC" and checked) else "–"
            out.append(f"| {variant} | {f:g} | {db(c[f]['S21']):.4f} | {fine} | {db(w['S21']):.4f} | {d:+.4f} "
                       f"| {mesh + DB_TOL:.4f} | {'yes' if p else 'NO'} | {abs(ref['S11']):.4f} / {abs(w['S11']):.4f} "
                       f"| {phase_mod180(w['S21'] / ref['S21']):+.2f} |")
    return verdicts


def _wp7b_twist_powers(profile, f_ghz):
    """WP7b helical-FEM reference powers {mode: P} for incident TE1,0, or None."""
    p = REPO / "benchmarks" / "validation_v2" / "data" / f"twist_doane_{profile}.npz"
    if not p.exists():
        return None
    import json
    with np.load(p, allow_pickle=False) as z:
        key = f"doane_{profile}_f{f_ghz:g}_main"
        if f"{key}_S21" not in z.files:
            return None
        names = [str(s) for s in z[f"{key}_names"]]
        S21 = z[f"{key}_S21"]
        meta = json.loads(str(z["meta"]))
    i = names.index("TE1,0")
    return {n: float(abs(S21[j, i]) ** 2) for j, n in enumerate(names)}, meta


def e4_ports(java):
    """Port layout of an E4 model from its .java: {port: (face, 'TE'/'TM', m_comsol, n_comsol, excited)}.

    COMSOL's rectangular-port indices (m, n) count half-periods along the port's long and short side; wrmodes (T3)
    uses a = 22.15 mm along u, so COMSOL (m, n) is wrmodes TE/TM n,m (the incident mode is COMSOL '01' = TE1,0)."""
    j = pathlib.Path(java).read_text()
    ports = {}
    for m in re.finditer(r'feature\("port(\d+)"\)\.selection\(\)\.set\((\d+)\)', j):
        ports[int(m.group(1))] = {"face": int(m.group(2)), "type": "TE", "num": "10", "exc": int(m.group(1)) == 1}
    for m in re.finditer(r'feature\("port(\d+)"\)\.set\("(PortModeNumber|PortModeType|PortExcitation)", "([^"]+)"\)', j):
        k, key, v = int(m.group(1)), m.group(2), m.group(3)
        if key == "PortModeNumber":
            ports[k]["num"] = v
        elif key == "PortModeType":
            ports[k]["type"] = v
        else:
            ports[k]["exc"] = v == "on"
    return {k: (p["face"], p["type"], int(p["num"][0]), int(p["num"][1]), p["exc"]) for k, p in ports.items()}


def _sname(i, j):
    return f"S{i}{j}" if i < 10 and j < 10 else f"S{i}_{j}"


def e4(raw, out, model=wrmodes_model, log=None):
    log = [] if log is None else log
    table(out, "E4: 20-inch 90° twists in WR187 (tall), power per output mode (% of the input), ports for every "
               "propagating mode at both ends",
          ["profile", "f (GHz)", "mode (wrmodes; COMSOL port)", "COMSOL", "COMSOL fine", "wrmodes Twist",
           "Δ (wrmodes − COMSOL finest)", "allowed", "pass", "WP7b helical FEM"])
    verdicts = []
    jdir = pathlib.Path(raw).parent / "java"
    for profile in ("triangular", "cosine"):
        c = load(raw, f"E4_twist_{profile}.txt", log)
        cf = load(raw, f"E4_twist_{profile}_fine.txt", log) if profile == "triangular" else None
        jp = jdir / f"E4_twist_{profile}.java"
        if c is None or not jp.exists():
            continue
        ports = e4_ports(jp)
        exc = next(k for k, p in ports.items() if p[4])
        in_face = ports[exc][0]
        for f in c:
            m = model("E4", profile, f)
            ref = cf[f] if cf and f in cf else c[f]
            coarse = c[f] if cf and f in cf else None
            wp7b = _wp7b_twist_powers(profile, f)
            sums = {"in": 0.0, "out": 0.0}
            for k, (face, typ, mc, nc, _) in sorted(ports.items()):
                key = _sname(k, exc)
                if key in ref:
                    sums["in" if face == in_face else "out"] += abs(ref[key]) ** 2
            # the model is lossless: COMSOL's own power-balance error |sum - 1| bounds its error from below, and is
            # added to the allowance (it matters for the main mode of runs without a mesh check)
            balance = abs(sums["in"] + sums["out"] - 1.0)
            for k, (face, typ, mc, nc, _) in sorted(ports.items()):
                key = _sname(k, exc)
                if key not in ref:
                    continue
                side = "in" if face == in_face else "out"
                name = f"{typ}{nc},{mc}"
                pw = m["out" if side == "out" else "in"].get(name)
                if pw is None or side == "in":
                    continue
                pc = abs(ref[key]) ** 2
                mesh = (abs(abs(coarse[key]) ** 2 - pc) if coarse else 0.0) + balance
                if max(pc, pw) < 1e-6 and mesh < 1e-6 + balance:
                    continue                                    # neither side puts power there
                ok = abs(pw - pc) <= mesh + P_TOL
                verdicts.append(("E4", profile, name, f, pw - pc, mesh + P_TOL, ok))
                ref7 = f"{100 * wp7b[0][name]:.4f}" if wp7b and name in wp7b[0] else "–"
                fine = f"{100 * abs(cf[f][key]) ** 2:.4f}" if cf and f in cf else "–"
                out.append(f"| {profile} | {f:g} | {name}; {k} ({typ}{mc}{nc}) | {100 * abs(c[f][key]) ** 2:.4f} | {fine} "
                           f"| {100 * pw:.4f} | {100 * (pw - pc):+.4f} | {100 * (mesh + P_TOL):.4f} "
                           f"| {'yes' if ok else 'NO'} | {ref7} |")
            out.append(f"| {profile} | {f:g} | COMSOL power out / reflected | {sums['out']:.5f} / {sums['in']:.5f} "
                       f"| | wrmodes {sum(m['out'].values()):.5f} / {sum(m['in'].values()):.5f} | | | | |")
    return verdicts


def e5(raw, out, model=wrmodes_model, log=None):
    log = [] if log is None else log
    table(out, "E5: two optical 45° H-plane miters, ports 37.86 mm from the reference planes",
          ["spacer", "f (GHz)", "COMSOL \\|S21\\| (dB)", "COMSOL fine", "single domain", "Δ", "cascade (cf 1)", "Δ",
           "allowed", "pass", "\\|S11\\| COMSOL / domain / cascade"])
    verdicts = []
    d_port = 37.86e-3
    for variant in ("s2a", "s3a"):
        swp = load(raw, f"E5_two45_{variant}_mesh_sweep.txt", log)
        if swp is None:
            continue
        c, cf = sweep(swp, p0=10), sweep(swp, p0=20)
        for f in c:
            b = beta("TE1,0", W, H, f)
            ref = cf[f] if cf and f in cf else c[f]
            coarse = c[f] if cf and f in cf else None
            res = {}
            for tag in ("domain", "cascade"):
                try:
                    m = model("E5", f"{variant}_{tag}", f)
                    res[tag] = {k: deembed(v, b, d_port, b, d_port) for k, v in m.items()}
                except ImportError as e:
                    res[tag] = None
                    log.append(f"[skip] E5 {tag}: {e}")
                except np.linalg.LinAlgError as e:   # the cascade of two 45 deg optical miters at s = 2a (WP3b)
                    res[tag] = None
                    log.append(f"[n/a] E5 {variant} {tag} {f:g} GHz: {str(e)[:60]}...")
            cells, ok, allowed = [], True, DB_TOL
            for tag in ("domain", "cascade"):
                if res[tag] is None:
                    cells += ["–", "–"]
                    continue
                d, allowed, p = judge_db(res[tag]["S21"], ref["S21"], coarse["S21"] if coarse else None)
                ok &= p
                verdicts.append(("E5", variant, tag, f, d, allowed, p))
                cells += [f"{db(res[tag]['S21']):.4f}", f"{d:+.4f}"]
            fine = f"{db(cf[f]['S21']):.4f}" if cf and f in cf else "–"
            s11 = " / ".join(f"{abs(x['S11']):.4f}" if x else "–" for x in (ref, res["domain"], res["cascade"]))
            out.append(f"| {variant} | {f:g} | {db(c[f]['S21']):.4f} | {fine} | " + " | ".join(cells)
                       + f" | {allowed:.4f} | {'yes' if ok else 'NO'} | {s11} |")
    return verdicts


CASES = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5}


def run(cases, raw=RAW, model=wrmodes_model):
    out, log, verdicts = [], [], []
    for k in cases:
        verdicts += CASES[k](raw, out, model=model, log=log)
    return out, log, verdicts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cases", nargs="*", default=list(CASES))
    p.add_argument("--raw", default=str(RAW), help="folder with the COMSOL exports (results_files)")
    p.add_argument("--out", default=None, help="write the markdown tables here")
    a = p.parse_args()
    out, log, verdicts = run(a.cases, a.raw)
    n_fail = sum(1 for v in verdicts if not v[-1])
    out.append(f"\n{len(verdicts)} comparisons, {n_fail} outside the criterion; {len(log)} skipped item(s).")
    print("\n".join(out))
    if a.out:
        pathlib.Path(a.out).write_text("\n".join(out) + "\n")
