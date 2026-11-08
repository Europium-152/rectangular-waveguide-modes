"""Shared helpers of the WP7 references: npz files with the metadata of CONVENTIONS.md section 9."""
import datetime
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def git_commit():
    """Commit of the working tree, with '+dirty' when it has uncommitted changes (as in tests/data)."""
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


def save_npz(path, generator, arrays, **params):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, meta=np.array(meta(generator, **params)), **arrays)
    return path


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["meta"] = json.loads(str(data["meta"]))
    return data


def power_db(x):
    return 10 * np.log10(np.maximum(np.abs(x) ** 2, 1e-300))
