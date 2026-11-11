# CC2 — COMSOL campaign 2 (Gate B)

Role: Coordinator (instructions) + Daniel (runs COMSOL) · Wave: 2 · Deliverable: `benchmarks/comsol/CAMPAIGN_2.md` ·
Used by: WP7b (post-processing), Gate B · Written: after this card set is approved, before WP5/WP6 finish

## Goal

One COMSOL campaign for the wave 2 components, written as GUI instructions (as campaign 1), small enough for the
COMSOL machine, and built on the lessons of Gate A.

## Lessons from campaign 1 (apply to every case)

* User-defined mesh (Free Tetrahedral / Free Triangular on all domains, Size node with an explicit maximum
  element size), never the physics-controlled mesh: it shifted a 15 cm bend by 0.02–0.03 dB.
* A mesh check (two sizes) at two frequencies per case.
* Within about 2 % above the cutoff of any mode the component excites, terminate with ports for **every**
  propagating mode instead of a PML: the PML reflects near-cutoff modes (26.3 GHz miter: 0.12 dB with PMLs,
  0.009 dB with ports).
* Export global quantities with a Global Evaluation (or a point inside the domain), not at (0, 0, 0).
* Ports of each kind explicitly set (Type of port = Rectangular, mode number), all excitation off except port 1.
* Symmetry planes only where the instructions say, with the boundary condition named (PEC or PMC).

## Cases (priority order; geometry and frequencies fixed in CAMPAIGN_2.md)

1. **E1, E-plane 90° miters in WR-90**, plain (d/b = 0.707) and Alessandri's d/b = 0.87: 3D half model (PMC at
   x = w/2), 8–12 GHz (Alessandri's range) plus 30 and 60 GHz. Verifies WP3b and WP4b in the E plane.
2. **E2, H-plane linear taper** 22.86 → 45 mm, b = 10.16 mm, L = 0.1 and 0.2 m, 15 GHz: 2D, ports for TE10–TE40 at
   the wide end and TE10–TE20 at the narrow end. Gives S21(TE30←TE10) of the T2b §9 table.
3. **E3, strangled E-plane bend**: WR-90 → 22.86 × 1.6 mm raised-cosine tapers of 50 mm, 90° E-plane arc of
   R = 50 mm (centre line) in the flat guide, back up; PEC and aluminium (σ = 3.774e7 S/m as in campaign 1);
   3D half model (PMC at x = w/2); 18 and 30 GHz.
4. **E4, triangular twist**: WR187 (47.55 × 22.15 mm), 20-inch 90° triangular-rate twist (Doane & Anderson),
   16.5 and 18 GHz, 3D; ports at the output for TE10 and the spurious modes of T3 §6, mapped from T3's frame to
   the model's (mode names depend on which side is called a). Check the cutoffs of the second-order modes:
   one of them lies within 1 % of 16.5 GHz in WR187. Settles T3 open issue 1 (0.25 / 0.60 % predicted vs ≈ 0.05–0.08 % read from Fig. 9).
5. **E5, two 45° H-plane miters** with a spacer of 0, a/2 and 2a in WR-90, 2D, 30 and 60 GHz (chained-miter study
   and the evanescent cascade).

## Acceptance

Daniel can run every case from the text alone on the COMSOL machine; each case states its expected run size
(degrees of freedom) and time. Results return as in campaign 1 (`results_files/`, `java_model_files/`).
