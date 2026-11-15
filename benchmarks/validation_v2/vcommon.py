"""Shared helpers of the WP7b references (wave 2 verification): npz files with the metadata of CONVENTIONS.md
section 9, GSM serialization, and small field utilities.

Every reference script in this folder writes `data/<name>.npz` with a JSON `meta` entry (wrmodes_version, git_commit,
created, generator and the parameters of the run). GSMs are stored as their four blocks with the mode names of both
port bases, so that comparisons can pick entries by name.
"""
import datetime
import json
import os
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
W90, H90 = 22.86e-3, 10.16e-3


def git_commit():
    """Commit of the working tree, with '+dirty' when src/ has uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", "src"]) != 0
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def meta(generator, **params):
    import wrmodes
    m = {"wrmodes_version": wrmodes.__version__, "git_commit": git_commit(),
         "created": datetime.date.today().isoformat(),
         "generator": os.path.relpath(os.path.abspath(generator), REPO).replace(os.sep, "/")}
    m.update(params)
    return json.dumps(m, sort_keys=True, default=str)


def save_npz(name, generator, arrays, **params):
    path = name if os.path.isabs(name) else os.path.join(DATA, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, meta=np.array(meta(generator, **params)), **arrays)
    return path


def load_npz(name):
    path = name if os.path.isabs(name) else os.path.join(DATA, name)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["meta"] = json.loads(str(data["meta"]))
    return data


def gsm_arrays(prefix, g):
    """Arrays describing a GSM: '<prefix>_S' (2, 2, n, n) padded blocks would be wasteful; store blocks."""
    return {f"{prefix}_S11": np.asarray(g.S11), f"{prefix}_S21": np.asarray(g.S21),
            f"{prefix}_S12": np.asarray(g.S12), f"{prefix}_S22": np.asarray(g.S22),
            f"{prefix}_names1": np.array(g.basis1.names), f"{prefix}_names2": np.array(g.basis2.names)}


class StoredGSM:
    """Blocks of a GSM read back from an npz (entries by mode name)."""

    def __init__(self, d, prefix):
        self.S11, self.S21, self.S12, self.S22 = (d[f"{prefix}_{k}"] for k in ("S11", "S21", "S12", "S22"))
        self.names1 = [str(s) for s in d[f"{prefix}_names1"]]
        self.names2 = [str(s) for s in d[f"{prefix}_names2"]]

    def s21(self, out, inp):
        return self.S21[self.names2.index(out), self.names1.index(inp)]

    def s11(self, out, inp):
        return self.S11[self.names1.index(out), self.names1.index(inp)]

    def s22(self, out, inp):
        return self.S22[self.names2.index(out), self.names2.index(inp)]

    def s12(self, out, inp):
        return self.S12[self.names1.index(out), self.names2.index(inp)]


def sub_block(S, names_row, names_col, rows, cols):
    """Sub-block of S with the given row/column mode names."""
    ir = [list(names_row).index(r) for r in rows]
    ic = [list(names_col).index(c) for c in cols]
    return np.asarray(S)[np.ix_(ir, ic)]


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-300))


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0
