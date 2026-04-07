"""orient_opt — Optimal 3D printing orientation via S² search."""

from .critical_points import CriticalPoint, find_critical_points
from .optimizer import coarse_then_refine
from .pareto import pareto_front, pairwise_pareto
from .types import OrientationResult

__all__ = [
    "optimize_orientation",
    "optimize_from_file",
    "pareto_front",
    "pairwise_pareto",
    "OrientationResult",
    "find_critical_points",
    "CriticalPoint",
]


def optimize_orientation(
    normals,
    areas,
    vertices,
    critical_faces=None,
    no_support_faces=None,
    lam: float = 0.7,
    overhang_angle: float = 45.0,
    n_samples: int = 500,
    beta: float = 50.0,
    n_refine: int = 5,
) -> OrientationResult:
    """Find the optimal 3D printing orientation for a triangle mesh.

    Args:
        normals: (F, 3) face normals, unit vectors.
        areas: (F,) face areas.
        vertices: (V, 3) vertex positions.
        critical_faces: (C,) indices of faces needing good surface quality.
        no_support_faces: (S,) indices of faces that must not overhang.
        lam: trade-off parameter, 0=height only, 1=overhang only.
        overhang_angle: threshold angle in degrees.
        n_samples: number of Fibonacci sphere sample points.
        beta: sigmoid sharpness for gradient-based refinement.
        n_refine: number of top candidates to refine via gradient descent.

    Returns:
        OrientationResult with optimal quaternion, rotation matrix, and diagnostics.
    """
    return coarse_then_refine(
        normals=normals,
        areas=areas,
        vertices=vertices,
        critical_faces=critical_faces,
        no_support_faces=no_support_faces,
        lam=lam,
        overhang_angle=overhang_angle,
        n_samples=n_samples,
        beta=beta,
        n_refine=n_refine,
    )


def optimize_from_file(
    path: str,
    critical_face_colors: list[tuple] | None = None,
    no_support_face_colors: list[tuple] | None = None,
    **kwargs,
) -> OrientationResult:
    """Load mesh from file and optimize orientation.

    Args:
        path: path to STL/OBJ file.
        critical_face_colors: RGB tuples identifying critical surface regions.
        no_support_face_colors: RGB tuples identifying no-support regions.
        **kwargs: passed to optimize_orientation.

    Returns:
        OrientationResult.
    """
    from .preprocess import extract_arrays, load_mesh, region_labels_from_colors

    mesh = load_mesh(path)
    normals, areas, vertices = extract_arrays(mesh)

    critical_faces = None
    if critical_face_colors:
        critical_faces = region_labels_from_colors(mesh, critical_face_colors)

    no_support_faces = None
    if no_support_face_colors:
        no_support_faces = region_labels_from_colors(mesh, no_support_face_colors)

    return optimize_orientation(
        normals, areas, vertices,
        critical_faces=critical_faces,
        no_support_faces=no_support_faces,
        **kwargs,
    )
