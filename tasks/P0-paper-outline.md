# P0 — Article outline, claims and figure list

Role: Writer · Wave: 3 · Branch: `P0-outline` (manuscript repository) · Depends on: wave 2 (merged, 1.0.0b1) and access to the
private manuscript repository `Europium-152/wrmodes_paper` (tasks/README.md, "Wave 3 decisions") · Reviewed by: Reviewer +
Daniel · Time box: 3 days (then status review)

## Goal

Plan the article about the tool, before any prose is written: what it claims, what evidence backs each claim, which
figures and tables carry that evidence, and what the release must provide for them. Daniel approves the outline; it
then fixes the scope of WF4, R1 and P1.

The article presents wrmodes 1.0 as a general tool for overmoded rectangular-waveguide lines. SPARC appears in at
most one sentence of motivation and the acknowledgements. **No result of the SPARC design studies** (plan studies
1–6, WP8a) goes into it: those belong to the SPARC article, after the bends are built and measured. Examples are
generic and, where possible, from the literature.

## Read first

`CONVENTIONS.md`; `tasks/README.md`; the plan's "Release plan and article" section (Daniel's project doc "wrmodes
expansion plan"); `benchmarks/registry.json`; `benchmarks/validation_v1/REPORT.md` and
`benchmarks/validation_v2/REPORT.md` (Gate A and Gate B sections); `benchmarks/comsol/campaign_1/RESULTS.md` and
`campaign_2/RESULTS.md`; every file in `docs/discrepancies/`; the reports in `docs/reports/`, at least WP1–WP7b,
WF2, WF3, L1, L2, T2b and T3b; the theory notes `docs/theory/T0`–`T3b`. The JINST smooth-bend paper (Daniel's
`JINST___Couple_mode_paper_V3` folder) for what is already published and must only be cited.

## Specification

1. **Target journal: IEEE Transactions on Microwave Theory and Techniques**, IEEEtran in journal mode (template from
   Daniel's `IEEE-Transactions-LaTeX2e-templates-and-instructions` folder; the coordinator puts it in the manuscript
   repository). The readers are microwave engineers: the article is a methods paper with validation, not a software
   announcement. Read the current TMTT author information and IEEE's author policies: length limit, article types (full paper or
   software paper), template, figure rules, data and code availability, and its policy on AI-assisted writing and
   code. Summarize them in `JOURNAL.md` with links and the date read.
2. **Candidate contributions.** Evaluate each one against the evidence and the literature (web search for prior
   work; say what is new and what is not). Keep the ones that hold; propose a title and an abstract of at most 200
   words.
   * One framework: a GSM cascade with a common basis and frames that joins coupled-mode components (bends with
     loss, tapers through cutoff, the strangled bend, twists) with mode-matching and 2D FEM components (miters,
     steps) and LS-mode filters; cascading through reference planes buried inside a discontinuity (WF3); the
     passivity check.
   * Convergence of coupled-mode series: fast when the walls are coordinate surfaces of an orthogonal system
     (straights, coplanar bends), slow like 1/χ_max when they are not (tapers, twists: H_n = w_n H_z on the wall),
     and the quasi-static tail correction that restores it (T2b §7, B5′; `twist_tail_matrix`, WF2).
   * The validation method: two independent methods per component, the independence rule, literature data with
     digitizing uncertainties, two COMSOL campaigns, and what they taught (user-defined meshes, multi-mode ports
     instead of PMLs near cutoff, power-balance allowances).
   * Findings about published results, stated with care and with the evidence that supports them: Doane & Anderson
     (1989) Fig. 9 triangular-rate curve; the Marcuvitz §5.24a N12 misprint; the curves of Alimenti (1994), Ma et
     al. (1997) and Reiter & Arndt (1995) on which three independent methods agree with each other and not with the
     paper. Daniel decides how each one is worded and whether it goes in.
   * Run time: what each component costs and why (the convergence item above explains most of it).
3. **Examples.** Generic and reproducible: Doane & Anderson's three 90° twists in WR187; Kotik & Sivov's optical
   H-plane miter (overmoded, up to kd ≈ 32); a Butterweck strangled E-plane bend and mode filters; Marcuvitz steps;
   two 45° miters with a spacer (COMSOL E5); one demonstration line of several components that shows trapped-mode
   resonances and the effect of wall loss, built from generic dimensions (not the SPARC line).
4. **Outline.** Sections with a one-paragraph content note and a page or word budget each. The JINST paper is
   submitted, not yet published: cite it as submitted, and do not repeat it: give a list of what is summarized from it (at most a paragraph) and what is new.
5. **Claims table** (`CLAIMS.md`): one row per claim the article will make, with its evidence (registry
   id, report section, data file, test) and its status: ready, needs a new computation (say which), or needs a new
   COMSOL run (avoid unless a claim cannot stand without one; Daniel runs it).
6. **Figure and table list** (`FIGURES.md`): for each item, its content, data source (stored reference data
   such as COMSOL exports, literature CSV and WP7b references may be read; every wrmodes curve is recomputed from
   the release), estimated run time, and the script name `paper/figures/<NN>_<slug>.py` that P1 will write in the public repository.
7. **What the release must provide** (`NEEDS.md`): API, documentation or performance items without which a
   figure or claim cannot be made (e.g. a uniform `convergence_study` output, a timing harness), for WF4 and R1. The coordinator copies these items into the WF4 and R1 cards,
   which live in the public repository.
8. **References** (`references.bib` seed): every reference the outline needs, with DOIs checked by a web
   lookup (title, authors, year and pages must match). No reference from memory alone.

## Acceptance

* Daniel approves `OUTLINE.md`, `CLAIMS.md` and `FIGURES.md` (comments in the PR).
* Every claim has evidence that exists in the repository or a named computation that will produce it.
* No SPARC design-study result in any file; the reviewer checks this.
* Every bibliography entry resolved by a lookup, with the lookup date.

## Deliverables

In the private manuscript repository `Europium-152/wrmodes_paper`, branch `P0-outline`, PR to its `main`:
`JOURNAL.md`, `OUTLINE.md`, `CLAIMS.md`, `FIGURES.md`, `NEEDS.md`, `references.bib`, `reports/P0.md`. Nothing about the
article's content goes into the public repository at this stage.

## Out of scope

Prose beyond the abstract and section notes; figure scripts (P1); the SPARC studies.
