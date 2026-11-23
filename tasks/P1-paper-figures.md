# P1 — Article figures, tables and numbers from the release

Role: Implementer · Wave: 3 · Branch: `wave-3/P1-paper-figures` · Depends on: P0 (approved), R1 (wrmodes 1.0.0
tagged) · Checked by: P3 (reproduces two figures from a clean install) · Time box: 5 days (then status review)

## Goal

Every figure, table and number of the article comes from a script run against the tagged release, so that a reader
can regenerate it, and so that no number is typed by hand.

## Read first

`FIGURES.md` and `CLAIMS.md` (P0, approved; manuscript repository); `CONVENTIONS.md` §9 (data files); `benchmarks/README.md`
and `benchmarks/paper_figures.py` (the JINST paper's figure scripts, as a pattern); the journal's figure rules in
`JOURNAL.md` (manuscript repository).

## Specification

1. One script per item, `paper/figures/<NN>_<slug>.py`, run against `wrmodes==1.0.0` installed from PyPI or the
   tag (the script checks `wrmodes.__version__` and refuses another version unless `--allow-version` is given).
2. Each script writes its results as npz with `meta` (CONVENTIONS §9), the figure as PDF in the journal's size and
   fonts (`paper/figures/style.py`, one style for all), and its quoted numbers into `paper/numbers/<NN>.json`.
3. `paper/tools/make_numbers.py` merges the JSON files into `paper/generated/numbers.tex`, one LaTeX macro per
   number with its rounding stated in the JSON. The manuscript uses only these macros for computed numbers.
4. Stored reference data (COMSOL exports, literature CSV, WP7b references) are read from the repository with their
   registry ids; every wrmodes curve is recomputed.
5. `make -C paper figures` (or a Python runner) regenerates everything; total run time and the slow items stated
   in `paper/figures/README.md`. Expensive results are cached, and `--force` recomputes them.
6. Run-time table: from `benchmarks/timings.py` (WF4) as run by Daniel, stored with its machine description.

## Acceptance

* Every item of `FIGURES.md` exists, or is dropped with Daniel's agreement.
* From a clean environment with `wrmodes==1.0.0`, the runner reproduces every npz to 1e-10 (or the stated
  tolerance of a non-deterministic item) and every macro exactly.
* Every number in `numbers.tex` is traced, in its JSON, to a script and a registry id or data file.

## Deliverables

In the public repository: `paper/figures/`, `paper/numbers/`, `paper/tools/`, `paper/generated/` (generated, committed), `docs/reports/P1.md`. In the manuscript
repository: the submodule pinned at the release tag, and a build that reads the figures and `paper/generated/` from it.

## Out of scope

Prose; changing the library (a bug found here goes to a fix card and a 1.0.x release, and the figures are rerun).
