# R1 — Release wrmodes 1.0.0 (Gate C)

Role: Implementer (documentation and packaging) · Wave: 3 · Branch: `wave-3/R1-release` · Depends on: WF4
(merged) · Reviewed by: Reviewer (fresh context, follows the guide from a clean install) + Daniel · Time box: 5 days
(then status review)

## Goal

A release a new user can install, learn and cite: documentation, examples, packaging and an archived DOI. The
article's figures are made from this release (P1), so it has to be the version readers will install.

## Read first

`README.md`; `CONVENTIONS.md`; `pyproject.toml`; `examples/`; `.github/workflows/tests.yml`; the module and class
docstrings of `wrmodes`; `paper/NEEDS.md` (documentation items); `benchmarks/registry.json`.

## Specification

1. **README** for 1.0: what the library does (components, cascade, loss, validation in one paragraph each), install
   (`pip install wrmodes`, `wrmodes[fem]`), a quick start with a `Line`, links to the guide, citation. The 0.1
   quick start stays as "smooth bends only". The swarm and task-card text moves to `CONTRIBUTING.md`.
2. **User guide** (`docs/guide/`, Markdown): concepts (modes and bases, families, frames, GSMs, reference planes);
   one page per component with a worked example, its method, its accuracy and its cost; lines and sweeps; choosing
   `cutoff_factor` and `mode_factor` and reading the warnings (EvanescentJunction, BuriedPlane, NonPassiveCascade);
   loss; validation summary generated from the registry; known limits (from the reports' open issues).
3. **API reference** built from the docstrings (MkDocs with mkdocstrings, NumPy docstring style), built in CI with
   warnings as errors. Hosting (GitHub Pages or Read the Docs) is Daniel's choice; until he enables it the build is
   only checked.
4. **Examples**: one runnable script per component plus a line, each under a minute; remove or rename
   `examples/test1.py`, `test2.py`; the notebook either updated or removed. A CI job runs every example.
5. **Packaging**: `CHANGELOG.md` (0.1 → 1.0.0); `CITATION.cff`; `.zenodo.json`; the sdist excludes bulky benchmark
   data that the tests do not need (state the sizes); the wheel holds `src/wrmodes` only; `python -m build` and
   `twine check` in CI; a clean-install job on Linux, macOS and Windows that installs the built wheel (with and
   without `[fem]`) in a fresh environment and runs the quick start and the examples.
6. **Version**: 1.0.0rc1 in this PR. After Daniel's check, a one-line PR sets 1.0.0; Daniel enables the Zenodo
   GitHub integration, tags `v1.0.0`, creates the GitHub release (Zenodo archives it) and uploads to PyPI.
7. **AI assistance**: a short statement in `CONTRIBUTING.md` and the README on how 1.0 was developed (agent-written
   code and tests, independent verification, maintainer review), in wording Daniel approves.

## Acceptance

* A reviewer with no context installs the wheel on a clean machine, follows the README and two guide pages, and
  runs every example without help; any stumble is fixed.
* Docs build without warnings; every public class and function has a docstring rendered in the reference.
* Clean-install jobs pass on three OSes; `twine check` passes.

## Deliverables

README, CONTRIBUTING, CHANGELOG, CITATION.cff, .zenodo.json, `docs/guide/`, MkDocs config, examples, CI jobs,
`docs/reports/R1.md`.

## Out of scope

Code changes beyond docstrings (they go back to WF4 or a fix card); the article.
