"""Timing harness of wrmodes 1.0 (WF4 items 5 and 6): wall time of one GSM per component at stated sizes, a
demonstration line per frequency, and wrmodes against COMSOL on the cases of campaigns 1 and 2.

Usage (from the repository root; Windows, macOS and Linux):

    python benchmarks/timings.py                       # components + demonstration line, one BLAS thread
    python benchmarks/timings.py --threads all         # the same with all BLAS threads
    python benchmarks/timings.py --quick               # smaller sizes (a smoke test of the harness, ~1 min)
    python benchmarks/timings.py --only twist_triangular_18GHz demo_line_pec
    python benchmarks/timings.py --comsol-cases [--comsol-log FILE ...] [--cases C1 C2 E1 ...]
    python benchmarks/timings.py --parse-logs FILE [FILE ...]   # the COMSOL log table only

Options: --repeat N (best of N runs per item, default 1), --out FILE.json (default
benchmarks/data/timings[_comsol]_<date>_<time>.json, no host name; a markdown copy of the tables goes next to it).

Output: JSON with the machine (CPU model, cores, RAM, OS, Python, NumPy, SciPy, BLAS vendor and threads, wrmodes
version and git commit) and one record per item: its description, frequency, mode count, the BLAS threads used and
the wall times. One BLAS thread by default (`--threads 1`, as `Line.sweep` uses per process); `--threads all` lets
BLAS use every core. Components that limit BLAS themselves while they solve (`Twist.gsm` and the propagating route
of `SmoothBend`, one thread each, so that results do not depend on the thread count) are not sped up by
`--threads all`; their records say so (`self_limited`).

--comsol-cases times wrmodes on every case of COMSOL campaigns 1 (C1, C1b, C2, C3a, C3b) and 2 (E1-E5) at the
frequencies and with the settings of their comparison scripts (benchmarks/comsol/campaign_*/compare_campaign_*.py),
with one BLAS thread and with all threads, and computes the accuracy each achieved against the other (the largest
|S21| difference in dB against the finest COMSOL run, and COMSOL's own mesh-check change). The COMSOL logs
(benchmarks/comsol/campaign_2/comsol_logs.txt, and any --comsol-log, e.g. campaign 1's when Daniel supplies them;
same format: a line '<run name>:' followed by COMSOL's '[date] Number of degrees of freedom solved for: N.' and
'[date] Solution time (Study 1): T s.' lines) are parsed into degrees of freedom, solution time and the
frequencies (and parameter combinations) of each run, read from the run's export in raw/. A case without a COMSOL
log entry is shown with '–'. The output has one table per case: the COMSOL runs (DOF, total time, solves, time
per solve) and the wrmodes variants (frequencies, time per frequency with 1 and all threads, max |dB| difference).
The fem2d cases (C1, C1b, E1, E5) need wrmodes[fem].

No POSIX-only calls; no multiprocessing (each item runs in this process, one after the other), so the harness runs
the same under the Windows 'spawn' start method. Card: tasks/WF4-release-fixes.md.
"""
import argparse
import contextlib
import datetime
import difflib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

import wrmodes as wr                                                     # noqa: E402
from wrmodes.components import Guide, SmoothBend, SechCurvature, Straight, Twist   # noqa: E402

W90, H90 = 22.86e-3, 10.16e-3
W62, H62 = 15.799e-3, 7.899e-3
INCH = 0.0254
SIGMA_AL, SIGMA_CU = 3.5e7, 5.8e7
COMSOL_LOGS = [REPO / "benchmarks" / "comsol" / "campaign_2" / "comsol_logs.txt"]
RAW_DIRS = [REPO / "benchmarks" / "comsol" / "campaign_1" / "raw", REPO / "benchmarks" / "comsol" / "campaign_2" / "raw"]


# ============================================================================================ machine
def _cpu_model():
    system = platform.system()
    try:
        if system == "Linux":
            for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        elif system == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_bytes():
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    system = platform.system()
    try:
        if system == "Linux":
            for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) * 1024
        elif system == "Darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        elif system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullTotalPhys)
    except Exception:
        pass
    return None


def _physical_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except Exception:
        return None


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def blas_info():
    """BLAS libraries loaded by NumPy/SciPy (threadpoolctl): vendor, version, threads."""
    try:
        from threadpoolctl import threadpool_info
        import scipy.linalg  # noqa: F401  (loads SciPy's BLAS too)
        return [dict(api=i.get("internal_api"), vendor=i.get("user_api"), version=i.get("version"),
                     threads=i.get("num_threads"), threading_layer=i.get("threading_layer"),
                     architecture=i.get("architecture"), library=os.path.basename(i.get("filepath", "")))
                for i in threadpool_info() if i.get("user_api") == "blas"]
    except Exception as exc:                                    # pragma: no cover
        return [dict(error=str(exc))]


def machine_info():
    import scipy
    ram = _ram_bytes()
    return dict(cpu=_cpu_model(), logical_cores=os.cpu_count(), physical_cores=_physical_cores(),
                ram_gb=None if ram is None else round(ram / 2 ** 30, 1), os=platform.platform(),
                machine=platform.machine(), python=sys.version.split()[0], numpy=np.__version__,
                scipy=scipy.__version__, blas=blas_info(), wrmodes=wr.__version__, git_commit=_git_commit(),
                date=datetime.datetime.now().isoformat(timespec="seconds"))


@contextlib.contextmanager
def blas_threads(n):
    """n = 1, an integer, or None (all threads: no limit)."""
    if n is None:
        yield
        return
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=int(n), user_api="blas"):
        yield


_WARNED = []


def best_of(fn, repeat):
    """(result, wall times) of `repeat` runs; the categories of the warnings issued go to _WARNED."""
    times, out = [], None
    for _ in range(max(1, repeat)):
        t = time.perf_counter()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = fn()
        times.append(time.perf_counter() - t)
        _WARNED[:] = sorted({x.category.__name__ for x in w})
    return out, times


# ============================================================================================ components
def demo_line(lossy):
    """The demonstration line (WF4, from the coordinator; generic, WR-62, 15-30 GHz). lossy: copper walls.

    straight 1 m -> H-plane sech bend 90 deg, 30 cm -> straight 0.5 m -> constant-rate 90 deg twist, 15 cm ->
    straight 0.3 m -> E-plane 90 deg chamfered miter (d/h = 0.87, COMSOL E1's validated chamfer) -> straight 0.2 m ->
    linear H-plane taper to 22.86 x 10.16 mm over 10 cm -> straight 2 m -> two 45 deg optical H-plane miters with
    spacer 3a -> straight 0.5 m.

    Perfect walls: the E-plane miter is `MiterMM` (mode matching) and the 45 degree miters `fem2d.Miter` (MiterMM
    is 90 degrees only). Copper: every miter is `fem2d.Miter` with the guide's wall impedance (MiterMM has perfect
    walls only, and a Line refuses it in a lossy guide)."""
    from wrmodes.modematch import MiterMM
    from wrmodes.fem2d import Miter
    g = Guide(W62, H62, sigma=SIGMA_CU if lossy else None)
    zw = "guide" if lossy else None
    emiter = Miter(90, chamfer=0.87, plane="E", wall_impedance=zw) if lossy else MiterMM(0.87, plane="E")
    return wr.Line(g, [Straight(1.0), SmoothBend(0.3, SechCurvature(90, 0.3), plane="H"), Straight(0.5),
                       Twist(90, 0.15, "constant"), Straight(0.3), emiter, Straight(0.2),
                       wr.Taper(0.10, W90, H90, "linear"), Straight(2.0),
                       Miter(45, plane="H", wall_impedance=zw), Straight(3 * W90),
                       Miter(45, plane="H", wall_impedance=zw), Straight(0.5)],
                   name="demonstration line, " + ("copper" if lossy else "perfect walls"))


def _gsm_item(comp, f, basis=None, basis2=None):
    def run():
        if basis is None:
            return comp.gsm(f)
        return comp.gsm(f, basis, basis2)
    return run


def component_items(quick=False):
    """(key, description, f, callable -> GSM, self_limited) for the components of P0's list (WF4 item 6)."""
    from wrmodes.modematch import MiterMM, Step
    from wrmodes.filters import LossyWallFilter
    al = Guide(W90, H90, sigma=SIGMA_AL)
    pec = Guide(W90, H90)
    tall = Guide(0.872 * INCH, 1.872 * INCH)
    items = []
    f = 30e9 if quick else 90e9
    items.append((f"lossy_straight_{f / 1e9:g}GHz", f"Straight 1 m, WR-90 aluminium (3.5e7 S/m), all propagating modes "
                  f"(all families coupled by the wall loss), {f / 1e9:g} GHz", f,
                  _gsm_item(Straight(1.0, guide=al), f, al.basis(f, 1.0)), False))
    fb, Lb = (20e9, 0.1) if quick else (40e9, 0.2)
    bend = SmoothBend(Lb, SechCurvature(90, Lb), plane="H", guide=al)
    items.append((f"smoothbend_propagating_{fb / 1e9:g}GHz", f"SmoothBend H-plane sech 90 deg, {Lb * 100:g} cm, WR-90 "
                  f"aluminium, propagating modes (transfer-matrix route), {fb / 1e9:g} GHz", fb,
                  _gsm_item(bend, fb, al.basis(fb, 1.0, plane="H")), True))
    items.append((f"smoothbend_evanescent_{fb / 1e9:g}GHz", f"the same bend, basis to 1.5 f (evanescent route, "
                  f"VariableGuide slice GSMs, internal list to 3 f), {fb / 1e9:g} GHz", fb,
                  _gsm_item(bend, fb, al.basis(fb, 1.5, plane="H")), False))
    ft = 30e9 if quick else 90e9
    items.append((f"taper_raised_cosine_{ft / 1e9:g}GHz", f"Taper raised cosine WR-90 -> 22.86 x 1.6 mm over 5 cm (E3's "
                  f"taper), perfect walls, default bases, {ft / 1e9:g} GHz", ft,
                  _gsm_item(wr.Taper(0.05, W90, 1.6e-3, "raised_cosine", guide=pec), ft), False))
    for fs in ((40e9,) if quick else (40e9, 90e9)):
        items.append((f"strangled_bend_{fs / 1e9:g}GHz", f"StrangledBend E3 geometry (90 deg, R = 50 mm, 1.6 mm flat, "
                      f"5 cm raised-cosine tapers), WR-90, perfect walls, {fs / 1e9:g} GHz", fs,
                      _gsm_item(wr.StrangledBend(90, 0.05, 1.6e-3, 0.05, guide=pec), fs), False))
    ftw = 15e9 if quick else 18e9
    for prof in ("constant", "triangular"):
        items.append((f"twist_{prof}_{ftw / 1e9:g}GHz", f"Twist 90 deg {prof}, 20 in, tall WR187 (Doane), defaults "
                      f"(mode_factor {3 if prof == 'constant' else 2}, tail), {ftw / 1e9:g} GHz", ftw,
                      _gsm_item(Twist(90, 20 * INCH, prof, guide=tall), ftw, tall.basis(ftw, 1.0)), True))
    for plane in ("H", "E"):
        for fm in ((30e9,) if quick else (30e9, 60e9, 90e9)):
            items.append((f"miter_mm_{plane}_{fm / 1e9:g}GHz", f"MiterMM {plane}-plane 90 deg optical mirror, WR-90, "
                          f"default truncation, {fm / 1e9:g} GHz", fm,
                          _gsm_item(MiterMM(plane=plane, guide=pec), fm), False))
    try:
        from wrmodes.fem2d import Miter
        ff = 30e9 if quick else 90e9
        m = Miter(90, plane="H", guide=pec)
        full = pec.basis(ff, 1.0, plane="H")
        one = full.subset([i for i, md in enumerate(full.modes) if md.n == 0])
        items.append((f"fem2d_miter_H_one_family_{ff / 1e9:g}GHz", f"fem2d.Miter H-plane 90 deg, WR-90, family n = 0 "
                      f"only ({len(one)} modes), default mesh, {ff / 1e9:g} GHz", ff, _gsm_item(m, ff, one, one), False))
        items.append((f"fem2d_miter_H_full_{ff / 1e9:g}GHz", f"fem2d.Miter H-plane 90 deg, WR-90, all propagating "
                      f"modes ({len(full)}), default mesh, {ff / 1e9:g} GHz", ff, _gsm_item(m, ff, full, full), False))
    except ImportError:
        pass
    fst = 30e9 if quick else 90e9
    st = Step(W90, H90, W90, 1.6e-3)
    b1, b2 = st.default_basis(fst)
    for reduce in (True, False):
        items.append((f"step_parity_{'on' if reduce else 'off'}_{fst / 1e9:g}GHz", f"Step WR-90 -> 22.86 x 1.6 mm, "
                      f"default truncation ({len(b1)}, {len(b2)} modes), parity reduction {'on' if reduce else 'off'}, "
                      f"{fst / 1e9:g} GHz", fst, (lambda r=reduce: st.solve(fst, b1, b2, reduce=r)), False))
    M = 40 if quick else 80
    items.append((f"lossy_wall_filter_M{M}_75GHz", f"LossyWallFilter 15 cm, 250 ohm narrow walls, WR-90, M = {M} modes "
                  f"per family, propagating basis, 75 GHz", 75e9,
                  _gsm_item(LossyWallFilter(0.15, 250.0, modes_per_family=M, guide=pec), 75e9), False))
    return items


def run_components(args):
    rows = []
    threads = None if args.threads == "all" else int(args.threads)
    for key, desc, f, fn, limited in component_items(args.quick):
        if args.only and key not in args.only:
            continue
        with blas_threads(threads):
            g, times = best_of(fn, args.repeat)
        rows.append(dict(key=key, description=desc, f=f, n_modes=[len(g.basis1), len(g.basis2)],
                         blas_threads="all" if threads is None else threads, self_limited=limited,
                         times_s=times, best_s=min(times), warnings=list(_WARNED)))
        print(f"{key:34s} {min(times):9.3f} s   ({len(g.basis1)}, {len(g.basis2)} modes)", flush=True)
    freqs = np.linspace(15e9, 30e9, 4 if args.quick else 16)
    for lossy in (False, True):
        key = "demo_line_copper" if lossy else "demo_line_pec"
        if args.only and key not in args.only:
            continue
        try:
            line = demo_line(lossy)
        except ImportError as exc:
            print(f"{key}: skipped ({exc})")
            continue
        per_f = []
        for f in freqs:
            with blas_threads(threads):
                g, times = best_of(lambda: line.gsm(f), args.repeat)
            i = g.basis1.index("TE1,0")
            per_f.append(dict(f=f, best_s=min(times), times_s=times, n_modes=[len(g.basis1), len(g.basis2)],
                              warnings=list(_WARNED),
                              te10_loss_db=float(-20 * np.log10(abs(g.S21[g.basis2.index("TE1,0"), i])))
                              if "TE1,0" in g.basis2 else None))
            print(f"{key:34s} {f / 1e9:5.1f} GHz {min(times):9.3f} s", flush=True)
        rows.append(dict(key=key, description=demo_line.__doc__.split("\n\n")[1].strip().replace("\n", " ")
                         + (" Copper (5.8e7 S/m)." if lossy else " Perfect walls."),
                         elements=[e.name or type(e).__name__ for e in line.elements],
                         blas_threads="all" if threads is None else threads, cutoff_factor=1.0, per_frequency=per_f,
                         total_s=sum(p["best_s"] for p in per_f), mean_s=float(np.mean([p["best_s"] for p in per_f]))))
    return rows


# ============================================================================================ COMSOL logs
_RUN = re.compile(r"^\s*([A-Za-z0-9_.+-]+)\s*:\s*$")
_DOF = re.compile(r"Number of degrees of freedom solved for:\s*([\d,]+)")
_TIME = re.compile(r"Solution time(?: \(([^)]*)\))?:\s*([\d.]+)\s*s")
_MESH = re.compile(r"Complete mesh consists of\s*([\d,]+) domain elements")


def parse_comsol_log(text):
    """{run name: dict(dofs=[...], solution_time_s=float or None, studies={study: s}, mesh_elements=int or None)}.

    A run starts with a line '<name>:'; COMSOL's own lines follow ('[date] Number of degrees of freedom solved
    for: N.', '[date] Solution time (Study 1): T s. (...)', '[date] Complete mesh consists of ...'). Several DOF
    lines in one run are the solves of a parameter sweep; several 'Solution time' lines are summed."""
    runs, cur = {}, None
    for line in text.splitlines():
        m = _RUN.match(line)
        if m and "[" not in line:
            cur = runs.setdefault(m.group(1), dict(dofs=[], solution_time_s=None, studies={}, mesh_elements=None))
            continue
        if cur is None:
            continue
        m = _DOF.search(line)
        if m:
            cur["dofs"].append(int(m.group(1).replace(",", "")))
        m = _TIME.search(line)
        if m:
            study = m.group(1) or "Study"
            cur["studies"][study] = cur["studies"].get(study, 0.0) + float(m.group(2))
            cur["solution_time_s"] = sum(cur["studies"].values())
        m = _MESH.search(line)
        if m:
            cur["mesh_elements"] = int(m.group(1).replace(",", ""))
    return runs


def _raw_files():
    return {p.stem: p for d in RAW_DIRS if d.exists() for p in sorted(d.glob("*.txt"))}


def _export_solves(path):
    """(frequencies in GHz, number of parameter combinations) of a COMSOL export."""
    sys.path.insert(0, str(REPO / "benchmarks" / "comsol" / "campaign_2"))
    from compare_campaign_2 import read
    data = read(path)
    if data and isinstance(next(iter(data)), tuple):
        freqs = sorted({f for rows in data.values() for f in rows})
        return freqs, len(data)
    return sorted(data), 1


def case_of(name):
    m = re.match(r"(C\d+b?|E\d+)", name)
    if not m:
        return None
    c = m.group(1)
    if c == "C1" and name.startswith("C1b"):
        return "C1b"
    if c == "C3":
        return "C3b" if "strangled" in name or name.startswith("C3b") else "C3a"
    return c


def comsol_table(log_files):
    """Rows per COMSOL run: case, run, raw export matched, DOFs, solution time, frequencies, solves, time per solve."""
    raws = _raw_files()
    rows = []
    for lf in log_files:
        lf = pathlib.Path(lf)
        if not lf.exists():
            print(f"[comsol] log {lf} not found")
            continue
        for name, r in parse_comsol_log(lf.read_text(encoding="utf-8", errors="replace")).items():
            stem = name if name in raws else (difflib.get_close_matches(name, list(raws), n=1, cutoff=0.85) or [None])[0]
            freqs, combos = (None, None)
            if stem is not None:
                try:
                    freqs, combos = _export_solves(raws[stem])
                except Exception as exc:                        # pragma: no cover
                    print(f"[comsol] {stem}: export not readable ({exc})")
            solves = None if freqs is None else len(freqs) * combos
            t = r["solution_time_s"]
            rows.append(dict(log=str(lf.name), run=name, case=case_of(name), export=stem, dofs=r["dofs"],
                             max_dofs=max(r["dofs"]) if r["dofs"] else None, mesh_elements=r["mesh_elements"],
                             solution_time_s=t, freqs_ghz=freqs, parameter_combinations=combos, solves=solves,
                             time_per_solve_s=(t / solves) if (t is not None and solves) else None))
    return rows


# ============================================================================================ wrmodes on the cases
def _db(x):
    return 20 * np.log10(max(abs(x), 1e-300))


def _campaign1_cases():
    """{case: [(variant, model(f_GHz) -> (S11, S21) at the reference planes, COMSOL coarse file, fine files)]} of
    benchmarks/comsol/campaign_1/compare_campaign_1.py (same components and settings)."""
    from wrmodes.components import ConstantCurvature
    from wrmodes.modematch import Step
    W, H = W90, H90
    te10 = wr.Mode("TE", 1, 0)

    def te(g):
        i1, i2 = g.basis1.index(te10), g.basis2.index(te10)
        return g.S11[i1, i1], g.S21[i2, i1]

    def miter(ang):
        def model(f):
            from wrmodes.fem2d import Miter
            return te(wr.Line(Guide(W, H), [Miter(ang, plane="H")]).gsm(f * 1e9))
        return model

    th, r0 = np.pi / 3, 0.055
    L_sech = 2 * th * r0 / np.arctan(np.tanh(2.0))

    def bend(L, kap):
        line = wr.Line(Guide(W, H, sigma=3.774e7), [SmoothBend(L, kap, plane="E")])
        return lambda f: te(line.gsm(f * 1e9))

    def step(h):
        st = Step(W, H, W, h, plane="H")
        return lambda f: te(st.gsm(f * 1e9))

    c3b = wr.Line(Guide(W, H), [Step(W, H, W, 1.6e-3), Straight(0.1), Step(W, 1.6e-3, W, H)])
    return {"C1": [("90", miter(90), "C1_miter90", ["C1_miter90_fine", "C1_miter90_fine_long_pml"])],
            "C1b": [("60", miter(60), "C1b_60miter", ["C1b_60miter_fine"]),
                    ("120", miter(120), "C1b_120miter", ["C1b_120miter_fine"])],
            "C2": [("sech", bend(L_sech, SechCurvature(60, L_sech)), "C2_sech_aluminum", []),
                   ("constant, user mesh", bend(0.15, ConstantCurvature.from_angle(60, 0.15)),
                    "C2_constant_aluminum_user_mesh", [])],
            "C3a": [("h = 2.0 mm", step(2.0e-3), "C3a_step_h2_0", ["C3a_step_h2_0_fine_tol1eminus4"]),
                    ("h = 1.6 mm", step(1.6e-3), "C3a_step_h1_6", ["C3a_step_h1_6_fine"])],
            "C3b": [("1.6 mm x 100 mm", lambda f: te(c3b.gsm(f * 1e9)), "C3b_strangled_100mm", [])]}


def _time_campaign1(case, threads):
    sys.path.insert(0, str(REPO / "benchmarks" / "comsol" / "campaign_1"))
    from compare_campaign_1 import read, RAW
    out = []
    for variant, model, coarse, fines in _campaign1_cases()[case]:
        c = read(coarse + ".txt")
        fine = {}
        for fn in fines:
            if (RAW / (fn + ".txt")).exists():
                fine.update(read(fn + ".txt"))
        fine_mesh = dict(fine)                                 # the mesh check: coarse vs fine mesh only
        if (RAW / (coarse + "_more_ports.txt")).exists():
            # as compare_campaign_1.c1: ports for TE10-TE40 on each arm, complete below the TE50 cutoff only
            fine.update({f: v for f, v in read(coarse + "_more_ports.txt").items() if f * 1e9 < 5 * 299792458 / (2 * W90)})
        times, deltas, deltas11, mesh = [], [], [], []
        for f in c:
            with blas_threads(threads), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = time.perf_counter()
                s11, s21 = model(f)
                times.append(time.perf_counter() - t)
            ref = fine.get(f, c[f])
            deltas.append(_db(s21) - _db(ref["S21"]))
            deltas11.append(_db(s11) - _db(ref["S11"]))
            if f in fine_mesh:
                mesh.append(abs(_db(c[f]["S21"]) - _db(fine_mesh[f]["S21"])))
        out.append(dict(variant=variant, freqs_ghz=list(c), times_s=times, delta_db=deltas,
                        delta_s11_db=deltas11 if case in ("C3a", "C3b") else None,
                        comsol_mesh_check_db=max(mesh) if mesh else None))
    return out


def _time_campaign2(case, threads):
    sys.path.insert(0, str(REPO / "benchmarks" / "comsol" / "campaign_2"))
    import compare_campaign_2 as c2
    times = {}

    def model(cs, variant, f):
        t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = c2.wrmodes_model(cs, variant, f)
        times.setdefault(variant, {})[f] = time.perf_counter() - t
        return r

    with blas_threads(threads):
        verdicts = c2.CASES[case](c2.RAW, [], model=model, log=[])
    out = []
    for variant, tf in times.items():
        vs = [v for v in verdicts if (v[1] == variant or f"{v[1]}_{v[2]}" == variant)]
        out.append(dict(variant=variant, freqs_ghz=sorted(tf), times_s=[tf[f] for f in sorted(tf)],
                        delta=[float(v[4]) for v in vs], allowed=[float(v[5]) for v in vs],
                        quantity="dB" if case != "E4" else "power fraction (per mode)", passed=all(v[6] for v in vs)))
    return out


CAMPAIGN1 = ("C1", "C1b", "C2", "C3a", "C3b")
CAMPAIGN2 = ("E1", "E2", "E3", "E4", "E5")


def run_comsol_cases(args):
    logs = [p for p in COMSOL_LOGS] + [pathlib.Path(p) for p in (args.comsol_log or [])]
    table = comsol_table(logs)
    cases = args.cases or list(CAMPAIGN1 + CAMPAIGN2)
    results = []
    for case in cases:
        rec = dict(case=case, comsol=[r for r in table if r["case"] == case], wrmodes={})
        for label, threads in (("1 thread", 1), ("all threads", None)):
            try:
                rec["wrmodes"][label] = (_time_campaign1 if case in CAMPAIGN1 else _time_campaign2)(case, threads)
            except ImportError as exc:
                rec["wrmodes"][label] = []
                rec["skipped"] = str(exc)
                break
            for v in rec["wrmodes"][label]:
                print(f"{case:4s} {label:11s} {v['variant']:24s} {np.mean(v['times_s']):8.3f} s per frequency "
                      f"({len(v['times_s'])} f)", flush=True)
        results.append(rec)
    return table, results


# ============================================================================================ output
def markdown(machine, components=None, comsol=None):
    out = [f"# wrmodes timings ({machine['date']})", "",
           f"Machine: {machine['cpu']}, {machine['logical_cores']} logical cores"
           + (f" ({machine['physical_cores']} physical)" if machine.get("physical_cores") else "")
           + f", {machine['ram_gb']} GB RAM, {machine['os']}; Python {machine['python']}, NumPy {machine['numpy']}, "
           f"SciPy {machine['scipy']}; BLAS "
           + ", ".join(f"{b.get('api')} {b.get('version')} ({b.get('threads')} threads)" for b in machine["blas"])
           + f"; wrmodes {machine['wrmodes']} ({machine['git_commit'][:10]})", ""]
    if components:
        out += ["| item | description | modes (port 1, port 2) | BLAS threads | best (s) |", "|---|---|---|---|---|"]
        for r in components:
            if "per_frequency" in r:
                out.append(f"| {r['key']} | {r['description']} | – | {r['blas_threads']} | "
                           f"{r['mean_s']:.3f} per frequency ({len(r['per_frequency'])} f, total {r['total_s']:.1f}) |")
            else:
                lim = " (self-limited to 1)" if r["self_limited"] and r["blas_threads"] != 1 else ""
                out.append(f"| {r['key']} | {r['description']} | {r['n_modes'][0]}, {r['n_modes'][1]} | "
                           f"{r['blas_threads']}{lim} | {r['best_s']:.3f} |")
        out.append("")
    if comsol:
        table, results = comsol
        for rec in results:
            out += [f"## {rec['case']}", ""]
            if rec["comsol"]:
                out += ["| COMSOL run | DOF | solution time (s) | solves (f x parameters) | time per solve (s) |",
                        "|---|---|---|---|---|"]
                for r in rec["comsol"]:
                    dof = "/".join(str(d) for d in r["dofs"]) or "–"
                    out.append(f"| {r['run']} | {dof} | {r['solution_time_s'] if r['solution_time_s'] is not None else '–'} "
                               f"| {r['solves'] if r['solves'] else '–'} | "
                               + (f"{r['time_per_solve_s']:.2f}" if r["time_per_solve_s"] else "–") + " |")
            else:
                out.append("COMSOL: – (no log entry)")
            out += ["", "| wrmodes variant | frequencies | s per frequency, 1 thread | all threads | max \\|Δ\\| vs "
                    "COMSOL finest (\\|S21\\|; steps also \\|S11\\|) | COMSOL mesh check / allowed |", "|---|---|---|---|---|---|"]
            one, allt = rec["wrmodes"].get("1 thread", []), {v["variant"]: v for v in rec["wrmodes"].get("all threads", [])}
            for v in one:
                a = allt.get(v["variant"])
                d = v.get("delta_db", v.get("delta"))
                dmax = f"{max(abs(x) for x in d):.4f}" if d else "–"
                unit = " dB" if v.get("quantity", "dB") == "dB" else " (power)"
                chk = v.get("comsol_mesh_check_db")
                chk = f"{chk:.4f} dB" if chk is not None else (f"{max(v['allowed']):.4f} allowed" if v.get("allowed") else "–")
                if v.get("delta_s11_db"):
                    dmax += f" (\\|S11\\| {max(abs(x) for x in v['delta_s11_db']):.4f})"
                out.append(f"| {v['variant']} | {len(v['freqs_ghz'])} ({min(v['freqs_ghz']):g}–{max(v['freqs_ghz']):g} GHz) "
                           f"| {np.mean(v['times_s']):.3f} | " + (f"{np.mean(a['times_s']):.3f}" if a else "–")
                           + f" | {dmax}{unit if d else ''} | {chk} |")
            if rec.get("skipped"):
                out.append(f"\n(skipped: {rec['skipped']})")
            out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--threads", default="1", help="BLAS threads: 1 (default), an integer, or 'all'")
    p.add_argument("--repeat", type=int, default=1, help="best of N runs per item")
    p.add_argument("--quick", action="store_true", help="smaller sizes (smoke test)")
    p.add_argument("--only", nargs="*", help="item keys to run")
    p.add_argument("--comsol-cases", action="store_true", help="time wrmodes on the COMSOL campaign 1 and 2 cases")
    p.add_argument("--comsol-log", nargs="*", help="further COMSOL log files (same format)")
    p.add_argument("--cases", nargs="*", help="COMSOL cases to run (C1 C1b C2 C3a C3b E1 E2 E3 E4 E5)")
    p.add_argument("--parse-logs", nargs="*", help="print the table of these COMSOL logs and exit")
    p.add_argument("--out", default=None, help="JSON output file")
    a = p.parse_args(argv)
    machine = machine_info()
    if a.parse_logs is not None:
        rows = comsol_table([pathlib.Path(x) for x in (a.parse_logs or COMSOL_LOGS)])
        for r in rows:
            print(f"{r['run']:32s} case {r['case'] or '?':4s} export {r['export'] or '–':32s} DOF {r['dofs']} "
                  f"time {r['solution_time_s']} s, solves {r['solves']}")
        return rows
    result = dict(machine=machine, settings=dict(threads=a.threads, repeat=a.repeat, quick=a.quick))
    if a.comsol_cases:
        table, results = run_comsol_cases(a)
        result.update(comsol_logs=table, comsol_cases=results)
        md = markdown(machine, comsol=(table, results))
    else:
        result["components"] = run_components(a)
        md = markdown(machine, components=result["components"])
    out = pathlib.Path(a.out) if a.out else (REPO / "benchmarks" / "data" / (
        f"timings_{'comsol_' if a.comsol_cases else ''}{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float))
    out.with_suffix(".md").write_text(md + "\n")
    print(md)
    print(f"\nwritten: {out} and {out.with_suffix('.md')}")
    return result


if __name__ == "__main__":
    main()
