"""Shared helpers of the WP4b benchmarks: the `meta` entry of CONVENTIONS.md section 9 and file paths."""
import datetime
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:          # pragma: no cover - outside a git checkout
        return "unknown"


def meta(generator, description, **parameters):
    import wrmodes
    return {"wrmodes_version": wrmodes.__version__, "git_commit": git_commit(),
            "created": datetime.date.today().isoformat(), "generator": generator, "description": description,
            "parameters": parameters}


def save(name, meta_dict, **arrays):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    np.savez_compressed(path, meta=np.array(json.dumps(meta_dict)), **arrays)
    return path


def load(name):
    d = np.load(os.path.join(DATA, name), allow_pickle=False)
    return {k: d[k] for k in d.files}, json.loads(str(d["meta"]))
