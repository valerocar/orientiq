import numpy as np
from scipy.special import expit
from scipy.spatial import ConvexHull, QhullError

CHUNK_SIZE = 2000


def overhang(
    normals: np.ndarray,
    areas: np.ndarray,
    candidates: np.ndarray,
    angle: float = 45.0,
) -> np.ndarray:
    """Hard-threshold overhang area for each candidate direction.

    Args:
        normals: (F, 3) face normals.
        areas: (F,) face areas.
        candidates: (N, 3) gravity direction candidates on S².
        angle: overhang angle in degrees.

    Returns:
        (N,) overhang area per candidate.
    """
    threshold = np.float32(-np.cos(np.radians(angle)))
    n = np.asarray(normals, dtype=np.float32)
    a = np.asarray(areas, dtype=np.float32)
    c = np.asarray(candidates, dtype=np.float32)
    N = len(c)
    result = np.empty(N, dtype=np.float32)
    for i in range(0, N, CHUNK_SIZE):
        dots = c[i:i+CHUNK_SIZE] @ n.T
        result[i:i+CHUNK_SIZE] = (dots < threshold) @ a
    return result


def overhang_smooth(
    normals: np.ndarray,
    areas: np.ndarray,
    candidates: np.ndarray,
    angle: float = 45.0,
    beta: float = 50.0,
) -> np.ndarray:
    """Sigmoid-smoothed overhang area for each candidate direction.

    Returns:
        (N,) smoothed overhang area per candidate.
    """
    threshold = -np.cos(np.radians(angle))
    dots = candidates @ normals.T  # (N, F)
    u = beta * (-dots - threshold)
    sig = expit(u)  # numerically safe sigmoid
    return sig @ areas  # (N,)


def build_height(
    vertices: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Build height (max - min vertex projection) for each candidate.

    Args:
        vertices: (V, 3) vertex positions.
        candidates: (N, 3) gravity direction candidates.

    Returns:
        (N,) build height per candidate.
    """
    v = np.asarray(vertices, dtype=np.float32)
    c = np.asarray(candidates, dtype=np.float32)
    N = len(c)
    result = np.empty(N, dtype=np.float32)
    for i in range(0, N, CHUNK_SIZE):
        proj = c[i:i+CHUNK_SIZE] @ v.T
        result[i:i+CHUNK_SIZE] = proj.max(axis=1) - proj.min(axis=1)
    return result


def violation(
    normals: np.ndarray,
    areas: np.ndarray,
    candidates: np.ndarray,
    no_support_faces: np.ndarray,
    angle: float = 45.0,
) -> np.ndarray:
    """Functional surface violation (hard constraint).

    Returns:
        (N,) violation area per candidate. Any value > 0 means infeasible.
    """
    threshold = -np.cos(np.radians(angle))
    ns_normals = normals[no_support_faces]
    ns_areas = areas[no_support_faces]
    dots = candidates @ ns_normals.T  # (N, S)
    mask = dots < threshold
    return mask @ ns_areas


def support_volume(
    normals: np.ndarray,
    areas: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    candidates: np.ndarray,
    angle: float = 45.0,
) -> np.ndarray:
    """Support volume proxy: overhang area weighted by height above build plate.

    For each candidate, sums A_f * h_f for overhanging faces, where h_f is the
    height of the face centroid above the lowest point of the mesh.

    Args:
        normals: (F, 3) face normals.
        areas: (F,) face areas.
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face vertex indices.
        candidates: (N, 3) gravity direction candidates.
        angle: overhang angle in degrees.

    Returns:
        (N,) support volume proxy per candidate.
    """
    threshold = np.float32(-np.cos(np.radians(angle)))
    n = np.asarray(normals, dtype=np.float32)
    a = np.asarray(areas, dtype=np.float32)
    v = np.asarray(vertices, dtype=np.float32)
    c = np.asarray(candidates, dtype=np.float32)
    centroids = v[faces].mean(axis=1)  # (F, 3) — precompute once

    N = len(c)
    result = np.empty(N, dtype=np.float32)
    for i in range(0, N, CHUNK_SIZE):
        chunk = c[i:i+CHUNK_SIZE]
        dots = chunk @ n.T                          # (chunk, F)
        mask = dots < threshold
        c_proj = chunk @ centroids.T                # (chunk, F)
        min_proj = (chunk @ v.T).min(axis=1)        # (chunk,)
        heights = c_proj - min_proj[:, None]         # (chunk, F)
        result[i:i+CHUNK_SIZE] = (mask * heights * a).sum(axis=1)
    return result


def surface_quality(
    normals: np.ndarray,
    areas: np.ndarray,
    candidates: np.ndarray,
    critical_faces: np.ndarray,
) -> np.ndarray:
    """Surface quality penalty for critical faces.

    Penalizes faces that are neither parallel nor perpendicular to gravity.

    Returns:
        (N,) quality penalty per candidate.
    """
    cn = np.asarray(normals[critical_faces], dtype=np.float32)
    ca = np.asarray(areas[critical_faces], dtype=np.float32)
    c = np.asarray(candidates, dtype=np.float32)
    N = len(c)
    result = np.empty(N, dtype=np.float32)
    for i in range(0, N, CHUNK_SIZE):
        dots = c[i:i+CHUNK_SIZE] @ cn.T
        penalty = (1 - np.abs(dots)) ** 2
        result[i:i+CHUNK_SIZE] = penalty @ ca
    return result


def _point_to_hull_distance(point_2d, hull_points):
    """Signed distance from a 2D point to the nearest edge of a convex hull.

    Positive = inside the hull, negative = outside.
    """
    try:
        hull = ConvexHull(hull_points)
    except (QhullError, ValueError):
        return 0.0

    # Hull equations: each row is [a, b, c] where ax + by + c <= 0 for interior
    # Distance to each halfplane: ax + by + c (negative = inside)
    eqs = hull.equations  # (n_edges, 3): [a, b, offset]
    dists = eqs[:, 0] * point_2d[0] + eqs[:, 1] * point_2d[1] + eqs[:, 2]
    # All dists should be <= 0 for interior points
    # The "most positive" distance is the closest to violating → margin
    # Flip sign so positive = inside
    return float(-dists.max())


def _build_basis(g):
    """Build an orthonormal basis (u, v) perpendicular to g."""
    g = g / np.linalg.norm(g)
    if abs(g[0]) < 0.9:
        u = np.cross(g, np.array([1, 0, 0], dtype=g.dtype))
    else:
        u = np.cross(g, np.array([0, 1, 0], dtype=g.dtype))
    u = u / np.linalg.norm(u)
    v = np.cross(g, u)
    return u, v


def _volumetric_com(vertices, faces):
    """Compute volumetric center of mass assuming a closed watertight mesh.

    Uses the signed tetrahedron method: each face forms a tetrahedron with the
    origin. The signed volume and weighted centroid are accumulated.

    Falls back to vertex centroid if the mesh volume is near zero (open mesh).
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    # Signed volume of each tetrahedron (origin, v0, v1, v2)
    cross = np.cross(v1, v2)
    tet_vol = np.einsum('ij,ij->i', v0, cross)  # (v0 × v1) · v2 per face
    total_vol = tet_vol.sum() / 6.0
    if abs(total_vol) < 1e-10:
        # Open mesh or degenerate — fall back to vertex centroid
        return v.mean(axis=0).astype(np.float32)
    # Weighted centroid: each tet contributes (v0+v1+v2) * tet_vol
    centroid = ((v0 + v1 + v2) * tet_vol[:, None]).sum(axis=0) / (24.0 * total_vol)
    return centroid.astype(np.float32)


def stability(
    vertices: np.ndarray,
    faces: np.ndarray,
    candidates: np.ndarray,
    base_tol: float = 0.02,
) -> np.ndarray:
    """Tip-over stability: critical tilt angle before the part topples.

    For each candidate gravity direction, computes the critical tilt angle
    (in radians). Larger = more stable. We return negative stability so it
    can be minimized like other objectives.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face vertex indices.
        candidates: (N, 3) gravity direction candidates.
        base_tol: fraction of build height to consider as "base" layer.

    Returns:
        (N,) negative stability (radians) per candidate. More negative = more stable.
    """
    v = np.asarray(vertices, dtype=np.float32)
    c = np.asarray(candidates, dtype=np.float32)
    com = _volumetric_com(vertices, faces)
    N = len(c)
    result = np.empty(N, dtype=np.float32)

    for i in range(N):
        g = c[i]

        # Project vertices along gravity
        proj = v @ g  # (V,)
        z_min = proj.min()
        z_max = proj.max()
        h_range = z_max - z_min
        if h_range < 1e-8:
            result[i] = 0.0
            continue

        # Base vertices: within base_tol of the bottom
        base_mask = proj < (z_min + base_tol * h_range)
        base_verts = v[base_mask]
        if len(base_verts) < 3:
            result[i] = 0.0  # not enough base → unstable
            continue

        # Build 2D basis perpendicular to g
        u, w = _build_basis(g)

        # Project base vertices to 2D
        base_2d = np.column_stack([base_verts @ u, base_verts @ w])

        # Project center of mass to 2D
        com_2d = np.array([com @ u, com @ w])

        # Distance from CoM to hull edge
        d = _point_to_hull_distance(com_2d, base_2d)

        # Height of center of mass above base
        h_cog = (com @ g) - z_min

        if h_cog < 1e-8:
            result[i] = -np.pi / 2  # CoM at base → maximally stable
            continue

        # Critical tilt angle (negative so minimizing = maximizing stability)
        result[i] = -np.arctan2(np.maximum(d, 0), h_cog)

    return result
