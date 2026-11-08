"""Literature comparisons for WP3 (miters) and WP4 (steps): L1's CSV data (D1, D3) and Marcuvitz (D6).

CSV format (tasks/L1-literature-data.md, CONVENTIONS.md section 9): comment lines starting with '#' (reference,
figure, curve label, axes and units, guide dimensions, calibration points, digitizers), then a header row and data
rows. The reader below accepts:

* the x column: the first column that is not one of the others (a column whose name contains 'freq' is preferred);
* the y column: the next one (or a column named y / value);
* ``uncertainty`` (required, same units as y; CONVENTIONS.md section 8: the tolerance of the comparison) and,
  optionally, ``x_uncertainty``;
* optionally a ``curve`` (or ``label``) column when one file holds several curves; otherwise the curve label is the
  ``# curve:`` header line, or the file name.

Header lines of the form ``# key: value`` are kept in ``header``.

A point passes when the model lies within the y uncertainty of the data for some abscissa inside x +- x_uncertainty.
The comparison reports the number of points within tolerance and the largest residual in units of the uncertainty.

What each figure means (the WP7 reading of the papers; see FIGURES):

* Alessandri, Mongiardo & Sorrentino, IEEE MGWL 4(12), 408-410 (1994), Figs. 4-5: 90 deg mitered E-plane bend,
  |S11| in dB vs frequency normalized to the TE10 cutoff; d is the distance from the inner corner to the mirror,
  normalized to the narrow side b (the side in the bend plane); d/b = 0.707 is the diagonal miter (mirror through
  the axes' crossing point) and d/b = 1.41 the unmitered corner (Fig. 1 and p. 408). The paper does not state the
  broad side a, on which the E-plane result depends through k_t b = (pi b / a) sqrt((f/f_c)^2 - 1): the ratio b/a
  must be supplied (header key ``b/a`` or argument).
* Ma, Yamane & Yamashita, Electron. Commun. Jpn. 2, 80(5), 1-7 (1997): mitered H-plane bend in a 15.799 x 7.899 mm
  guide (p. 4). The miter is given by c, the length cut from each outer wall at the outer corner (Figs. 6, 8, 9,
  11), so d = (2a - c)/sqrt(2), d/a = (2 - c/a)/sqrt(2): c/a = 1 is the diagonal miter (d/a = 0.707), c/a = 0 the
  right-angle corner (Fig. 5, d/a = 1.414). Table 1 (p. 5) gives, at 16 GHz for c/a = 1, |S11| = 0.2622 and
  |S21| = 0.9650 with 60 staircase steps; displacing the steps changed them by up to 0.57 % and 0.04 %.
* Marcuvitz, Waveguide Handbook (1951), sec. 5.26a, eqs. (1), (2a) (pp. 307-308, PDF pp. 321-322): symmetric change
  in height b -> b' of a rectangular guide (E-plane step), shunt susceptance B at the junction plane,
  Y0/Y0' = b'/b = alpha. Stated accuracy: within 1 % for b / lambda_g < 1. ``marcuvitz_eplane_step`` implements it;
  the H-plane step (sec. 5.24) is left to L1's transcription (D6).
"""
import csv
import io
import os
import re

import numpy as np
import scipy.constants as cst

SQ2 = np.sqrt(2.0)


# ----------------------------------------------------------------------------------------- CSV
class Curve:
    def __init__(self, label, x, y, unc, x_unc, header, xname, yname, path):
        self.label, self.x, self.y, self.unc, self.x_unc = label, x, y, unc, x_unc
        self.header, self.xname, self.yname, self.path = header, xname, yname, path

    def parameter(self, name=None):
        """The numerical parameter in the curve label, e.g. 0.87 from 'd/b = 0.87' (None if there is none)."""
        text = self.label if name is None else self.header.get(name, self.label)
        m = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", str(text).strip()) or \
            re.search(r"=\s*([-+]?\d*\.?\d+)", str(text))
        return float(m.group(1)) if m else None

    def __repr__(self):
        return f"Curve({self.label!r}, {len(self.x)} points, {self.xname} -> {self.yname})"


def read_csv(path):
    """Curves of an L1 CSV file (see the module docstring)."""
    with open(path, newline="", encoding="utf-8") as fh:
        text = fh.read()
    header, rows = {}, []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            body = s.lstrip("#").strip()
            if ":" in body:
                k, v = body.split(":", 1)
                header.setdefault(k.strip().lower(), v.strip())
        elif s:
            rows.append(line)
    reader = csv.DictReader(io.StringIO("\n".join(rows)))
    cols = [c.strip() for c in reader.fieldnames]
    low = {c.lower(): c for c in cols}
    if "uncertainty" not in low:
        raise ValueError(f"{path}: no 'uncertainty' column (L1 card and CONVENTIONS.md 9 require one)")
    special = {"uncertainty", "x_uncertainty", "curve", "label"}
    data_cols = [c for c in cols if c.lower() not in special]
    xcol = next((c for c in data_cols if "freq" in c.lower()), data_cols[0])
    ycol = low.get("y") or low.get("value") or next(c for c in data_cols if c != xcol)
    ccol = low.get("curve") or low.get("label")
    groups = {}
    for r in reader:
        r = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
        lab = r[ccol] if ccol else header.get("curve", os.path.splitext(os.path.basename(path))[0])
        g = groups.setdefault(lab, [[], [], [], []])
        g[0].append(float(r[xcol]))
        g[1].append(float(r[ycol]))
        g[2].append(float(r[low["uncertainty"]]))
        g[3].append(float(r[low["x_uncertainty"]]) if "x_uncertainty" in low and r[low["x_uncertainty"]] else 0.0)
    return [Curve(lab, *(np.array(v) for v in g), header=header, xname=xcol, yname=ycol, path=path)
            for lab, g in groups.items()]


def compare_curve(curve, model, n_sub=7):
    """Compare model(x) -> y with a curve. Returns dict(n, n_within, worst_ratio, worst_x, residuals)."""
    ratios, res = [], []
    for x, y, u, dx in zip(curve.x, curve.y, curve.unc, curve.x_unc):
        xs = np.linspace(x - dx, x + dx, n_sub) if dx > 0 else np.array([x])
        vals = np.array([model(xx) for xx in xs])
        d = np.abs(vals - y)
        k = int(np.argmin(d))
        res.append(vals[k] - y)
        ratios.append(d[k] / u if u > 0 else (0.0 if d[k] == 0 else np.inf))
    ratios = np.array(ratios)
    i = int(np.argmax(ratios))
    return dict(n=len(ratios), n_within=int(np.sum(ratios <= 1.0)), worst_ratio=float(ratios[i]),
                worst_x=float(curve.x[i]), residuals=np.array(res), ok=bool(np.all(ratios <= 1.0)))


# ----------------------------------------------------------------------------------------- quantities
def quantity(S, kind):
    """Value plotted in a figure from a complex S entry: 'S11_dB', 'S21_dB' (20 log10 |S|), 'S21_power' (|S|^2),
    'S_mag' (|S|)."""
    if kind.endswith("_dB"):
        return 20 * np.log10(np.abs(S))
    if kind.endswith("_power"):
        return np.abs(S) ** 2
    return np.abs(S)


def ma_chamfer(c_over_a):
    """Ma et al. (1997) cut length c/a -> chamfer d/a (distance inner corner to mirror over a)."""
    return (2.0 - c_over_a) / SQ2


def cutoff_te10(a):
    return cst.c / (2 * a)


# Figure table: what the axes and curve labels of each L1 file mean (from the papers, see module docstring).
FIGURES = {
    "alessandri1994_fig4": dict(plane="E", angle=90, x="f/fc", quantity="S11_dB", param="d/b", a=None, b=None),
    "alessandri1994_fig5": dict(plane="E", angle=90, x="f/fc", quantity="S11_dB", param="d/b", a=None, b=None),
    "ma1997_fig5": dict(plane="H", angle=90, x="GHz", quantity="S21_power", param="c/a", fixed_param=0.0,
                        a=15.799e-3, b=7.899e-3),
    "ma1997_fig6": dict(plane="H", angle=90, x="GHz", quantity="S21_dB", param="c/a", fixed_param=1.0,
                        a=15.799e-3, b=7.899e-3),
    "ma1997_fig8": dict(plane="H", angle=90, x="GHz", quantity="S11_dB", param="c/a", fixed_param=1.0,
                        a=15.799e-3, b=7.899e-3),
    "ma1997_fig9": dict(plane="H", angle=90, x="GHz", quantity="S11_dB", param="c/a", fixed_param=1.0,
                        a=15.799e-3, b=7.899e-3),
    "ma1997_fig11": dict(plane="H", angle=90, x="f/fc", quantity="S11_dB", param="c/a", a=15.799e-3, b=7.899e-3),
}

# Ma et al. (1997) Table 1, p. 5: 16 GHz, c/a = 1, 60 steps; spread under step displacement 0.57 % and 0.04 %.
MA1997_TABLE1 = dict(f=16e9, c_over_a=1.0, S11=0.2622, S21=0.9650, rel_unc_S11=0.0057, rel_unc_S21=0.0004,
                     a=15.799e-3, b=7.899e-3)


def figure_spec(path_or_name):
    base = os.path.splitext(os.path.basename(path_or_name))[0]
    for key, spec in FIGURES.items():
        if base.startswith(key):
            return key, dict(spec)
    raise KeyError(f"no figure specification for {base}; add it to ref_literature.FIGURES")


def curve_chamfer(curve, spec):
    """Chamfer d (normalized to the in-plane side) of a curve, from its label and the figure's parameter."""
    p = spec.get("fixed_param")
    if p is None:
        p = curve.parameter()
    if p is None:
        raise ValueError(f"cannot read the {spec['param']} value of curve {curve.label!r}")
    return ma_chamfer(p) if spec["param"] == "c/a" else p


def miter_model(miter_s, spec, chamfer):
    """model(x) -> plotted value, with miter_s(f, chamfer, plane, a, b) -> (S11, S21) for TE10 (WP3 adapter).

    For E-plane figures the in-plane side is b (Alessandri's narrow side) and f_c the TE10 cutoff of side a.
    """
    a, b = spec["a"], spec["b"]
    fc = cutoff_te10(a)

    def model(x):
        f = x * fc if spec["x"] == "f/fc" else x * 1e9
        S11, S21 = miter_s(f, chamfer, spec["plane"], a, b)
        return quantity(S11 if spec["quantity"].startswith("S11") else S21, spec["quantity"])
    return model


def compare_miter_file(path, miter_s, a=None, b_over_a=None):
    """All curves of an L1 miter file against miter_s. For Alessandri's figures, b/a must be given (or in the
    header as 'b/a'); a defaults to 22.86 mm (only the ratio matters for normalized frequency)."""
    key, spec = figure_spec(path)
    curves = read_csv(path)
    if spec["a"] is None:
        ratio = b_over_a if b_over_a is not None else curves[0].header.get("b/a")
        if ratio is None:
            raise ValueError(f"{key}: the paper does not state the guide; pass b_over_a (b/a)")
        spec["a"] = 22.86e-3 if a is None else a
        spec["b"] = float(ratio) * spec["a"]
    out = []
    for c in curves:
        d = curve_chamfer(c, spec)
        r = compare_curve(c, miter_model(miter_s, spec, d))
        r.update(file=os.path.basename(path), curve=c.label, chamfer=d)
        out.append(r)
    return out


# ----------------------------------------------------------------------------------------- Marcuvitz
def marcuvitz_eplane_step(f, a, b, b_small):
    """Symmetric E-plane step (change in height b -> b_small, equal widths a), Marcuvitz sec. 5.26a eq. (2a).

    Returns (B/Y0, Y0'/Y0, S11) with S11 the TE10 reflection seen from the larger guide at the junction plane
    T: load admittance y = Y0'/Y0 + jB/Y0 = 1/alpha + jB/Y0, S11 = (1 - y)/(1 + y). |S11| does not depend on the
    sign convention of j or on the reference-plane phase. Valid for b/lambda_g < 1 (within 1 %).
    """
    alpha = b_small / b
    k = 2 * np.pi * f / cst.c
    beta = np.sqrt(k * k - (np.pi / a) ** 2)
    lg = 2 * np.pi / beta
    x, xp = b / lg, b_small / lg
    if not (0 < alpha < 1) or x >= 1:
        raise ValueError("Marcuvitz 5.26a: needs b' < b and b / lambda_g < 1")
    r = (1 + alpha) / (1 - alpha)
    sq, sqp = np.sqrt(1 - x * x), np.sqrt(1 - xp * xp)
    A = r ** (2 * alpha) * (1 + sq) / (1 - sq) - (1 + 3 * alpha ** 2) / (1 - alpha ** 2)
    Ap = r ** (2 / alpha) * (1 + sqp) / (1 - sqp) + (3 + alpha ** 2) / (1 - alpha ** 2)
    C = (4 * alpha / (1 - alpha ** 2)) ** 2
    B = 2 * x * (np.log((1 - alpha ** 2) / (4 * alpha) * r ** (0.5 * (alpha + 1 / alpha)))
                 + 2 * (A + Ap + 2 * C) / (A * Ap - C * C)
                 + (b / (4 * lg)) ** 2 * ((1 - alpha) / (1 + alpha)) ** (4 * alpha)
                 * ((5 * alpha ** 2 - 1) / (1 - alpha ** 2) + 4 / 3 * alpha ** 2 * C / A) ** 2)
    y = 1 / alpha + 1j * B
    return B, 1 / alpha, (1 - y) / (1 + y)


def marcuvitz_eplane_step_2c(f, a, b, b_small):
    """Small-alpha asymptotic form, eq. (2c), for a self-check of the transcription (5 % for alpha < 0.6 and
    b/lambda_g < 0.5)."""
    alpha = b_small / b
    k = 2 * np.pi * f / cst.c
    lg = 2 * np.pi / np.sqrt(k * k - (np.pi / a) ** 2)
    x = b / lg
    return 2 * x * (np.log(np.e / (4 * alpha)) + alpha ** 2 / 3 + 0.5 * x * x * (1 - alpha ** 2) ** 4)
