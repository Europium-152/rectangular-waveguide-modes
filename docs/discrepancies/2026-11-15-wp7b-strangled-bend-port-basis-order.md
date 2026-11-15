# Discrepancy: StrangledBend port 2 basis order differs from port 1 for some flat heights (WP5)

*Found by WP7b (adversarial test `tests/adversarial/test_wp7b_adversarial.py::test_strangled_bend_port_bases_are_equal`,
strict xfail until fixed). Owner: WP5. Severity: minor (no wrong physics; standalone cascades fail).*

## Observation

Both ports of a `StrangledBend` are the same full-height guide, so CONVENTIONS.md §3 requires `basis2 == basis1`
(same sides to 1e-12 and the same ordered modes). For WR-90 at 60 GHz, 25 of 200 flat heights between 1.5 and 2.6 mm
give `basis2 != basis1` in `default_basis(f)`, and `gsm(f)` returns a GSM whose port 2 basis has two modes swapped:

```
b1 = c / (2 * 60e9) * (1 - 1e-3) ** -1        # 2.5008 mm
sb = wrmodes.StrangledBend(90, 0.05, b1, 0.03, guide=Guide(22.86e-3, 10.16e-3))
g = sb.gsm(60e9)
g.basis1.a, g.basis2.a        # 0.01016, 0.010159999999999999
# order differs at positions 51, 52: basis1 [..., 'TE4,0', 'TE0,9', ...], basis2 [..., 'TE0,9', 'TE4,0', ...]
g @ g                         # ValueError: cannot join ... port bases differ
```

In the E frame of WR-90 (a = 10.16 mm, b = 22.86 mm = 2.25 a) local TE4,0 and TE0,9 have exactly the same cutoff
(59.02 GHz). The port 2 side a is computed from the profile (a(L) = 0.010159999999999999, one rounding error below
h), which breaks the tie the other way. The physics is right: comparing S11 and S22 by mode name, the strangled
bend is symmetric to 1e-14, and the WP7b FEM of the same geometry gives the same entries (to its 3e-3 accuracy at
this near-cutoff point). Inside a `Line` the component works, because the line imposes its own basis.

## Consequences

* `sb.gsm(f) @ other.gsm(f)` and `g.reversed()` comparisons by index fail or mislead whenever the guide has exactly
  degenerate modes of different families (WR-90 in the E frame: (4, 0)/(0, 9), (8, 0)/(0, 18), ... and all modes
  m/a = n/b). A user comparing S11 with S22 index by index sees a spurious asymmetry of order 1.

## Suggested fix (for WP5, not applied by WP7b)

Build both port bases from the end guides as given (`guide` at both ends of a symmetric strangled bend, and the
exact `a2`, `b2` of a `Taper`), not from the profile evaluated at z = L, or snap the end values of a(z), b(z) to the
given end dimensions. Then remove the strict xfail marker of the test.
