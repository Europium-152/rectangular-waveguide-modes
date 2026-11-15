"""WP7b adversarial tests of WP3b (impedance walls and chained miters in the 2D FEM), card WP7b item 7.

They need the optional FEM extra (scikit-fem, gmsh) and run in CI's fem job. Weak points: a vanishing and a strongly
reactive wall impedance, chained miters with zero spacer (a zero-length cascade inside one domain, and the
documented refusal of overlapping opposite turns), and two 45 degree optical mirrors so close that they intersect.
"""
import numpy as np
import pytest

pytest.importorskip("skfem")
pytest.importorskip("gmsh")

import wrmodes as wr  # noqa: E402
from wrmodes.components import Guide, Straight  # noqa: E402
from wrmodes.fem2d import Coplanar2D, Miter, chained_miter_outline, straight_outline  # noqa: E402
from wrmodes.modes import Mode  # noqa: E402

W90, H90 = 22.86e-3, 10.16e-3
TE10 = Mode("TE", 1, 0)


def test_vanishing_wall_impedance_is_the_perfect_miter():
    """Z_s -> 0 on every in-plane wall of a miter: linear in Z_s (1e-9 and 1e-12 ohm agree to 1e-10) and equal to
    the perfectly conducting miter within the FEM discretization (default mesh, 5e-4)."""
    f = 30e9
    g = Guide(W90, H90)
    pec = Miter(90, plane="H", guide=g).gsm(f)
    z9 = Miter(90, plane="H", guide=g, wall_impedance=1e-9).gsm(f)
    z12 = Miter(90, plane="H", guide=g, wall_impedance=1e-12).gsm(f)
    assert np.max(np.abs(z9.S - z12.S)) < 1e-10
    assert np.max(np.abs(z12.S - pec.S)) < 5e-4


@pytest.mark.parametrize("Zs", [250 + 250j, 5 - 400j, 0.05 + 0.05j])
def test_reactive_and_lossy_walls_are_passive_and_reciprocal(Zs):
    """Strongly reactive impedance walls (Re Z_s >= 0, Im of either sign, including a surface-wave-supporting
    inductive wall) on a 5 cm straight with families n = 0 and 1 propagating: passive, reciprocal."""
    f = 30e9
    c = Coplanar2D(straight_outline(W90, 0.05), b=H90, plane="H", guide=Guide(W90, H90), wall_impedance=Zs)
    s = c.gsm(f)
    assert np.all(np.isfinite(s.S))
    assert s.reciprocity_defect() <= 1e-9 * max(1.0, np.max(np.abs(s.S)))
    sv = np.linalg.svd(np.block([[s.S11, s.S12], [s.S21, s.S22]]), compute_uv=False)
    assert sv[0] <= 1 + 1e-9


def test_chained_u_turn_with_zero_spacer_refused():
    """Two 90 deg miters turning the same way with no spacer would fold the guide onto itself: documented
    ValueError, not a mesh of an overlapping domain."""
    with pytest.raises(ValueError):
        chained_miter_outline([90, 90], [0.0], a=W90)


def test_periscope_zero_spacer_against_cascade():
    """Two optical 90 deg miters turning opposite ways with zero spacer (a periscope offset by a): the single FEM
    domain against the zero-length Line cascade [Miter(90), Miter(-90)] with evanescent modes (cutoff_factor 4).
    The difference measures what the cascade misses at zero distance; |dS21(TE1,0)| < 1e-2, both lossless."""
    f = 20e9
    g = Guide(W90, H90)
    out, mirror, length = chained_miter_outline([90, -90], [0.0], a=W90)
    one = Coplanar2D(out, b=H90, plane="H", guide=g, mirror=mirror, length=length).gsm(f)
    line = wr.Line(g, [Miter(90, plane="H"), Miter(-90, plane="H")]).gsm(f, cutoff_factor=4.0)
    assert one.unitarity_defect() < 1e-8
    i1, i2 = one.basis1.index(TE10), line.basis1.index(TE10)
    assert abs(one.S21[i1, i1] - line.S21[i2, i2]) < 1e-2


def test_two_45_degree_optical_mirrors_that_intersect():
    """Two optical 45 deg miters with a spacer of 1.5a, shorter than the 2a at which the mirrors meet (CAMPAIGN_2
    E5): the outline follows both mirrors to their intersection; the result is lossless and symmetric."""
    f = 30e9
    out, mirror, length = chained_miter_outline([45, 45], [1.5 * W90], a=W90)
    s = Coplanar2D(out, b=H90, plane="H", guide=Guide(W90, H90), mirror=mirror, length=length).gsm(f)
    assert s.unitarity_defect() < 1e-8
    assert np.max(np.abs(s.S11 - s.S22)) < 1e-8
