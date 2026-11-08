"""Geometry and meshing of 2D coplanar components (u-z plane of CONVENTIONS.md section 2).

A component is described by its **core**: a closed outline in the plane of side a, made of straight segments and
circular arcs, two of whose edges are the **port planes**. The solver attaches a straight arm of length
`port_distance` to each port plane and puts the modal (Dirichlet-to-Neumann) ports at the far ends of the arms.

Coordinates are 2D Cartesian (x, y) in metres. Each port plane is given by its two end points: `inner` (u = -a/2)
and `outer` (u = +a/2), so that u increases from `inner` to `outer`. The longitudinal axis z of the component goes
into the core at port 1 and out of the core at port 2; the frames of the two ports must share the same v axis
(v = z x u), which `PortPlane` checks (CONVENTIONS.md section 2: port 2's frame is port 1's frame transported
along the centre line, rotated about v).
"""
from dataclasses import dataclass, field
import numpy as np

__all__ = ["PortPlane", "Outline", "MeshData", "build_mesh"]


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


@dataclass(frozen=True)
class PortPlane:
    """A port plane of the core: segment from `inner` (u = -a/2) to `outer` (u = +a/2).

    `deembed` (m, >= 0) is the distance from this plane to the reference plane of the component, measured into the
    core: the GSM is computed at this plane and shifted to the reference plane analytically (CONVENTIONS.md
    section 6). It is non-zero only when the core is not a uniform guide right up to the reference plane (e.g. a
    deeply chamfered miter, whose mirror cuts the outer wall before the reference plane).
    """
    inner: tuple
    outer: tuple
    deembed: float = 0.0

    @property
    def a(self):
        return float(np.hypot(*(np.subtract(self.outer, self.inner))))

    @property
    def u_hat(self):
        d = np.subtract(self.outer, self.inner)
        return d / np.hypot(*d)


@dataclass(frozen=True)
class Outline:
    """Closed outline of the core.

    vertices : sequence of (x, y)
        Vertices in order (either orientation).
    arcs : dict {i: (cx, cy)}
        The edge from vertex i to vertex i+1 is a circular arc (shorter than a half turn) with this centre;
        all other edges are straight.
    ports : (PortPlane, PortPlane)
        Port 1 and port 2. Each must be an edge (two consecutive vertices) of the outline.
    """
    vertices: tuple
    ports: tuple
    arcs: dict = field(default_factory=dict)

    def __post_init__(self):
        v = tuple(tuple(float(c) for c in p) for p in self.vertices)
        object.__setattr__(self, "vertices", v)
        object.__setattr__(self, "arcs", {int(k): tuple(map(float, c)) for k, c in dict(self.arcs).items()})
        if len(self.ports) != 2:
            raise ValueError("a two-port component needs exactly two port planes")
        n = len(v)
        scale = max(np.ptp(np.array(v), axis=0))
        object.__setattr__(self, "_scale", scale)
        for p in self.ports:
            i, j = self._find(p.inner), self._find(p.outer)
            if (i - j) % n not in (1, n - 1):
                raise ValueError(f"port plane {p} is not an edge of the outline")
            e = min(i, j) if abs(i - j) == 1 else max(i, j)
            if e in self.arcs:
                raise ValueError("a port plane must be a straight edge")
            if p.deembed < 0:
                raise ValueError("deembed must be >= 0")
        for k, c in self.arcs.items():
            p0, p1 = np.array(v[k]), np.array(v[(k + 1) % n])
            r0, r1 = np.hypot(*(p0 - c)), np.hypot(*(p1 - c))
            if abs(r0 - r1) > 1e-9 * scale:
                raise ValueError(f"arc {k}: end points are not on one circle about its centre")
        # frames: z into the core at port 1, out of the core at port 2; v = z x u must agree
        z1 = self.inward_normal(0)
        z2 = -self.inward_normal(1)
        s1, s2 = _cross(z1, self.ports[0].u_hat), _cross(z2, self.ports[1].u_hat)
        if not np.isclose(s1, s2):
            raise ValueError("the port frames do not share the v axis (v = z x u): swap inner and outer of one "
                             "port (CONVENTIONS.md section 2)")

    def _find(self, pt):
        d = np.hypot(*(np.array(self.vertices) - np.asarray(pt, float)).T)
        i = int(np.argmin(d))
        if d[i] > 1e-9 * self._scale:
            raise ValueError(f"point {pt} is not a vertex of the outline")
        return i

    def contains(self, pt):
        """Point-in-polygon test on the polygon of the vertices (arcs replaced by many chords)."""
        poly = self.polyline(32)
        x, y = pt
        inside = False
        for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
            if (y0 > y) != (y1 > y):
                if x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                    inside = not inside
        return inside

    def polyline(self, n_arc=16):
        pts = []
        n = len(self.vertices)
        for i in range(n):
            p0 = np.array(self.vertices[i])
            pts.append(p0)
            if i in self.arcs:
                c = np.array(self.arcs[i])
                p1 = np.array(self.vertices[(i + 1) % n])
                a0, a1 = np.arctan2(*(p0 - c)[::-1]), np.arctan2(*(p1 - c)[::-1])
                da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
                r = np.hypot(*(p0 - c))
                for t in np.linspace(0, 1, n_arc + 1)[1:-1]:
                    pts.append(c + r * np.array([np.cos(a0 + t * da), np.sin(a0 + t * da)]))
        return np.array(pts)

    def inward_normal(self, k):
        """Unit normal of port plane k (0 or 1) pointing into the core."""
        p = self.ports[k]
        u = p.u_hat
        nrm = np.array([-u[1], u[0]])
        mid = 0.5 * (np.array(p.inner) + np.array(p.outer))
        eps = 1e-6 * p.a
        if self.contains(mid + eps * nrm):
            return nrm
        if self.contains(mid - eps * nrm):
            return -nrm
        raise ValueError("cannot determine the inner side of a port plane")

    def area(self):
        x, y = self.polyline(64).T
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def full_boundary(self, port_distance):
        """Outline of the whole domain (core plus arms of length port_distance): (vertices, arcs)."""
        V = [np.array(v) for v in self.vertices]
        n = len(V)
        edges = {}
        for k, p in enumerate(self.ports):
            edges[frozenset((self._find(p.inner), self._find(p.outer)))] = self.inward_normal(k)
        out, arcs = [], {}
        for i in range(n):
            j = (i + 1) % n
            out.append(V[i])
            key = frozenset((i, j))
            if key in edges:
                nrm = edges[key]
                out += [V[i] - port_distance * nrm, V[j] - port_distance * nrm]
            elif i in self.arcs:
                arcs[len(out) - 1] = self.arcs[i]
        return out, arcs

    def reentrant_corners(self, port_distance):
        """Vertices of the domain (core plus arms) with an interior angle above pi (field singularities)."""
        V, arcs = self.full_boundary(port_distance)
        V = np.array(V)
        n = len(V)
        orient = np.sign(sum(_cross(V[i], V[(i + 1) % n]) for i in range(n)))     # +1 counter-clockwise
        out = []
        for i in range(n):
            t_in = _edge_tangent(V, arcs, i - 1, end=True)
            t_out = _edge_tangent(V, arcs, i, end=False)
            if _cross(t_in, t_out) * orient < -1e-9:     # turning against the orientation: reflex vertex
                out.append(V[i])
        return np.array(out).reshape(-1, 2)

def _edge_tangent(V, arcs, i, end):
    """Unit tangent of edge i (from V[i] to V[i+1]) at its start (end=False) or end (end=True)."""
    n = len(V)
    i %= n
    p0, p1 = np.asarray(V[i]), np.asarray(V[(i + 1) % n])
    if i not in arcs:
        t = p1 - p0
        return t / np.hypot(*t)
    c = np.asarray(arcs[i])
    r = (p1 if end else p0) - c
    t = np.array([-r[1], r[0]])
    if _cross(p0 - c, p1 - c) < 0:
        t = -t
    return t / np.hypot(*t)


@dataclass
class MeshData:
    """A triangle mesh of the domain (core plus arms), with the data the solver needs."""
    p: np.ndarray                 # (2, Np) vertex coordinates
    t: np.ndarray                 # (3, Nt) triangles
    region: np.ndarray            # (Nt,) 0 = arm of port 1, 1 = core, 2 = arm of port 2
    arcs: list                    # [(cx, cy, r)] circles of curved boundary edges
    ports: tuple                  # the PortPlanes of the outline
    normals: tuple                # inward unit normals of the port planes
    port_distance: float          # length of the arms (m)
    h: float                      # nominal element size (m)


# ------------------------------------------------------------------------------------------------ gmsh
def _arm(port, nrm, length):
    """Rectangle of the arm attached to a port plane (outside the core)."""
    pi, po = np.array(port.inner), np.array(port.outer)
    return [pi, po, po - length * nrm, pi - length * nrm]


class _Builder:
    """Builds gmsh built-in geometry with shared points and curves."""

    def __init__(self, gmsh, scale):
        self.g = gmsh.model.geo
        self.points, self.curves = {}, {}
        self.tol = 1e-9 * scale

    def point(self, x):
        key = (round(x[0] / self.tol), round(x[1] / self.tol))
        if key not in self.points:
            self.points[key] = self.g.addPoint(float(x[0]), float(x[1]), 0.0)
        return self.points[key]

    def curve(self, a, b, centre=None):
        ia, ib = self.point(a), self.point(b)
        key = (min(ia, ib), max(ia, ib), None if centre is None else self.point(centre))
        if key in self.curves:
            tag, start = self.curves[key]
            return tag if start == ia else -tag
        if centre is None:
            tag = self.g.addLine(ia, ib)
        else:
            tag = self.g.addCircleArc(ia, self.point(centre), ib)
        self.curves[key] = (tag, ia)
        return tag

    def surface(self, pts, arcs=None):
        arcs = arcs or {}
        n = len(pts)
        loop = [self.curve(pts[i], pts[(i + 1) % n], arcs.get(i)) for i in range(n)]
        return self.g.addPlaneSurface([self.g.addCurveLoop(loop)])


def _gmsh_mesh(regions, h, refine_points, refine_factor, refine_radius, scale):
    """Mesh a list of (region_id, points, arcs) with gmsh; returns p (2, N), t (3, M), region (M,)."""
    import gmsh
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize(["-noenv"], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("wrmodes_fem2d")
        b = _Builder(gmsh, scale)
        surf = [(rid, b.surface(pts, arcs)) for rid, pts, arcs in regions]
        b.g.synchronize()
        fields = gmsh.model.mesh.field
        const = fields.add("MathEval")
        fields.setString(const, "F", repr(float(h)))
        used = [const]
        if len(refine_points):
            ptags = [b.g.addPoint(float(x), float(y), 0.0) for x, y in refine_points]
            b.g.synchronize()
            dist = fields.add("Distance")
            fields.setNumbers(dist, "PointsList", ptags)
            thr = fields.add("Threshold")
            fields.setNumber(thr, "InField", dist)
            fields.setNumber(thr, "SizeMin", refine_factor * h)
            fields.setNumber(thr, "SizeMax", h)
            fields.setNumber(thr, "DistMin", 0.0)
            fields.setNumber(thr, "DistMax", refine_radius)
            used.append(thr)
        mn = fields.add("Min")
        fields.setNumbers(mn, "FieldsList", used)
        fields.setAsBackgroundMesh(mn)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.RandomSeed", 1)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        index = {int(t): i for i, t in enumerate(tags)}
        p = coords.reshape(-1, 3)[:, :2].T.copy()
        tris, reg = [], []
        for rid, s in surf:
            et, _, en = gmsh.model.mesh.getElements(2, s)
            k = list(et).index(2)
            tri = np.array([index[int(x)] for x in en[k]]).reshape(-1, 3).T
            tris.append(tri)
            reg.append(np.full(tri.shape[1], rid))
        t = np.hstack(tris)
        region = np.concatenate(reg)
        gmsh.model.remove()
    finally:
        if own:
            gmsh.finalize()
    # drop unused nodes (e.g. the refinement points)
    used_nodes = np.unique(t)
    remap = -np.ones(p.shape[1], dtype=int)
    remap[used_nodes] = np.arange(len(used_nodes))
    return p[:, used_nodes], remap[t], region


def _reflect(x, origin, direction):
    """Mirror points x (2, N) about the line through origin along direction."""
    d = np.asarray(direction, float) / np.hypot(*direction)
    o = np.asarray(origin, float)[:, None]
    r = x - o
    along = d[:, None] * (d @ r)[None, :]
    return o + 2 * along - r


def build_mesh(outline, h, port_distance, refine_factor=0.2, refine_radius=None, mirror=None):
    """Mesh the core of `outline` plus the two arms.

    Parameters
    ----------
    outline : Outline
        The core and its port planes.
    h : float
        Element size away from re-entrant corners (m).
    port_distance : float
        Length of the arms (m): distance from each port plane to the modal port; 0 puts the modal ports on the
        port planes of the core (no arms).
    refine_factor, refine_radius : float
        Element size at re-entrant corners (as a fraction of h) and the distance (m) over which it grows back to h
        (default: 2 h).
    mirror : (origin, direction), optional
        Mesh only the half of the domain on the side of port 1 of this symmetry line, and mirror it, so that the
        mesh is exactly symmetric (used by symmetric miters: S11 = S22 then holds to rounding). The outline must
        be symmetric about the line, with port 2 the mirror image of port 1.
    """
    refine_radius = 2 * h if refine_radius is None else refine_radius
    n1, n2 = outline.inward_normal(0), outline.inward_normal(1)
    arm1 = _arm(outline.ports[0], n1, port_distance)
    arm2 = _arm(outline.ports[1], n2, port_distance)
    scale = max(np.ptp(np.array(outline.vertices), axis=0)) + port_distance
    # the field singularities are those of the physical guide, whose arms continue beyond the port planes
    corners = outline.reentrant_corners(max(port_distance, min(p.a for p in outline.ports)))
    arms = port_distance > 0
    if mirror is None:
        regions = [(1, list(map(np.array, outline.vertices)), outline.arcs)]
        if arms:
            regions = [(0, arm1, None)] + regions + [(2, arm2, None)]
        p, t, region = _gmsh_mesh(regions, h, corners, refine_factor, refine_radius, scale)
    else:
        origin, direction = mirror
        half = _half_outline(outline, origin, direction)
        regions = ([(0, arm1, None)] if arms else []) + [(1, half, None)]
        p, t, region = _gmsh_mesh(regions, h, corners, refine_factor, refine_radius, scale)
        p, t, region = _mirror_mesh(p, t, region, origin, direction, scale)
    arcs = [(c[0], c[1], float(np.hypot(*(np.array(outline.vertices[k]) - c)))) for k, c in outline.arcs.items()]
    if mirror is not None and arcs:
        raise NotImplementedError("mirrored meshes of outlines with arcs are not supported")
    p, t = np.ascontiguousarray(p, dtype=float), np.ascontiguousarray(t, dtype=np.int64)
    return MeshData(p=p, t=t, region=region, arcs=arcs, ports=outline.ports, normals=(n1, n2),
                    port_distance=port_distance, h=h)


def _half_outline(outline, origin, direction):
    """Vertices of the part of the (polygonal) core on the side of port 1 of the symmetry line."""
    if outline.arcs:
        raise NotImplementedError("mirrored meshes of outlines with arcs are not supported")
    d = np.asarray(direction, float) / np.hypot(*direction)
    nrm = np.array([-d[1], d[0]])
    o = np.asarray(origin, float)
    side = lambda x: float(nrm @ (np.asarray(x) - o))
    s1 = side(outline.ports[0].inner) + side(outline.ports[0].outer)
    sgn = np.sign(s1)
    V = [np.array(v) for v in outline.vertices]
    n = len(V)
    out = []
    tol = 1e-9 * max(np.ptp(np.array(outline.vertices), axis=0))
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        sa, sb = sgn * side(a), sgn * side(b)
        if sa >= -tol:
            out.append(a)
        if (sa > tol and sb < -tol) or (sa < -tol and sb > tol):
            out.append(a + (b - a) * sa / (sa - sb))
    # snap points lying on the line exactly onto it
    res = []
    for x in out:
        if abs(side(x)) <= tol:
            x = o + d * (d @ (x - o))
        res.append(x)
    return res


def _mirror_mesh(p, t, region, origin, direction, scale):
    d = np.asarray(direction, float) / np.hypot(*direction)
    nrm = np.array([-d[1], d[0]])
    dist = nrm @ (p - np.asarray(origin, float)[:, None])
    on_line = np.abs(dist) <= 1e-9 * scale
    pm = _reflect(p, origin, direction)
    N = p.shape[1]
    new_index = np.arange(N) + N
    new_index[on_line] = np.flatnonzero(on_line)       # shared nodes of the symmetry line
    keep = ~on_line
    remap = -np.ones(N, dtype=int)
    remap[keep] = N + np.arange(keep.sum())
    remap[on_line] = np.flatnonzero(on_line)
    p_all = np.hstack([p, pm[:, keep]])
    t_m = remap[t][[0, 2, 1]]                          # mirrored triangles: keep a consistent orientation
    region_m = np.where(region == 0, 2, region)
    return p_all, np.hstack([t, t_m]), np.concatenate([region, region_m])
