"""Check benchmarks/registry.json (WF4 item 4; CONVENTIONS.md section 9). Run in CI (the fem job, which has every
optional dependency, so that every test module is collected).

    python benchmarks/check_registry.py [--no-collect]

Checks, for every entry:

* the fields id, component, card, reference_type, reference, data_file, test, script, tolerance, status, note are
  present; ids are unique; reference_type is one of the registry's reference_types;
* status is one of the documented vocabulary (`status_vocabulary` of the registry: pass, not met, qualitative,
  superseded, reference) and note is a non-empty free text; a superseded entry names, in its note, an entry that
  exists;
* every data_file exists (a path ending in '/' is a non-empty folder; '<...>' placeholders match by glob, at least
  one file);
* every script exists;
* every test id is collected by pytest (`pytest --collect-only -q`; a parametrized test matches its base id, and a
  test file matches when any of its tests is collected). --no-collect skips this (faster, for local edits).

Exit status 1 with one line per problem, 0 otherwise.
"""
import argparse
import glob
import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
REGISTRY = REPO / "benchmarks" / "registry.json"
VOCABULARY = ("pass", "not met", "qualitative", "superseded", "reference")
FIELDS = ("id", "component", "card", "reference_type", "reference", "data_file", "test", "script", "tolerance",
          "status", "note")


def collected_tests():
    """Node ids collected by pytest in the repository (run from the repository root)."""
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                         cwd=str(REPO), capture_output=True, text=True)
    ids = [line.strip() for line in out.stdout.splitlines() if "::" in line]
    if not ids:
        raise RuntimeError("pytest collected nothing:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    return ids


def test_is_collected(tid, ids, bases, files):
    tid = tid.replace("\\", "/")
    if "::" not in tid:
        return tid in files
    return tid in ids or tid in bases


def path_exists(p):
    if p.endswith("/"):
        d = REPO / p
        return d.is_dir() and any(d.iterdir())
    if "<" in p:
        return bool(glob.glob(str(REPO / re.sub(r"<[^>]*>", "*", p))))
    return (REPO / p).exists()


def check(registry=REGISTRY, collect=True):
    reg = json.loads(pathlib.Path(registry).read_text(encoding="utf-8"))
    problems = []
    vocab = tuple(reg.get("status_vocabulary", {}))
    if vocab != VOCABULARY:
        problems.append(f"status_vocabulary is {vocab}, expected {VOCABULARY}")
    entries = reg["entries"]
    ids = [e.get("id") for e in entries]
    seen = set()
    for i in ids:
        if i in seen:
            problems.append(f"duplicate id {i!r}")
        seen.add(i)
    if collect:
        tests = collected_tests()
        bases = {t.split("[")[0] for t in tests}
        files = {t.split("::")[0] for t in tests}
    for e in entries:
        eid = e.get("id", "?")
        for k in FIELDS:
            if k not in e:
                problems.append(f"{eid}: field {k!r} missing")
        if e.get("reference_type") not in reg.get("reference_types", []):
            problems.append(f"{eid}: reference_type {e.get('reference_type')!r} not in {reg.get('reference_types')}")
        st = e.get("status")
        if st not in VOCABULARY:
            problems.append(f"{eid}: status {st!r} is not one of {VOCABULARY}")
        note = e.get("note")
        if not isinstance(note, str) or not note.strip():
            problems.append(f"{eid}: note must be a non-empty text")
        elif st == "superseded" and not any(other in note for other in ids if other != eid):
            problems.append(f"{eid}: superseded, but its note names no other entry")
        for key in ("data_file", "test", "script"):
            if not isinstance(e.get(key, []), list):
                problems.append(f"{eid}: {key} must be a list")
        for p in e.get("data_file", []):
            if not path_exists(p):
                problems.append(f"{eid}: data_file {p!r} does not exist")
        for p in e.get("script", []):
            if not (REPO / p).is_file():
                problems.append(f"{eid}: script {p!r} does not exist")
        if collect:
            for t in e.get("test", []):
                if not test_is_collected(t, set(tests), bases, files):
                    problems.append(f"{eid}: test {t!r} is not collected by pytest")
    return problems, len(entries)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--no-collect", action="store_true", help="do not check the test ids against pytest")
    p.add_argument("--registry", default=str(REGISTRY))
    a = p.parse_args(argv)
    problems, n = check(a.registry, collect=not a.no_collect)
    for line in problems:
        print(line)
    print(f"{n} entries, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
