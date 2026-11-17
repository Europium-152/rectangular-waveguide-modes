# P3 — Independent referee review

Role: Reviewer (fresh context; never saw P0–P2 being written) · Wave: 3 · Branch: `wave-3/P3-referee` · Depends on:
P2 (all checkpoints merged) · Time box: 3 days

## Goal

Read the manuscript as the journal's referee would, before a real one does, and test it against the repository.

## Read first

Only the built PDF of the manuscript and the journal's author guide (`paper/JOURNAL.md`) at first. Write the first
part of the report (below) before opening anything else. Then the repository at the tag `v1.0.0`.

## Specification

1. **Referee report, before opening the repository**: summary of the paper; significance and novelty; major
   points (unclear method, unsupported claim, missing comparison or prior work); minor points; a recommendation.
   Search the literature for prior work on each claimed contribution and cite what you find.
2. **Checks against the repository**, after part 1 is committed:
   * install `wrmodes==1.0.0` in a clean environment and reproduce two figures chosen by you (not the cheapest)
     with P1's scripts;
   * trace at least ten numbers of the text to their JSON, script and registry id;
   * for every statement about another author's result, read the evidence (discrepancy file, report, data) and say
     whether it supports the wording;
   * check every reference: it exists, its bibliographic data are right, and it says what the text cites it for
     (PDFs in Daniel's reference folder, read-only, or a web source);
   * check that equations agree with the theory notes they name.
3. Each point gets a severity (major, minor, typo) and a location (section, line or equation).

## Acceptance

Part 1 committed before part 2 (commit order checked by the coordinator); every check of part 2 done or its
blocker stated.

## Deliverables

`paper/review/P3-referee-report.md`, `docs/reports/P3.md`.

## Out of scope

Editing the manuscript (P4 answers the report).
