"""Multi-objective optimization: Pareto fronts and pairwise intersection."""

from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from .objectives import build_height, overhang, overhang_smooth, surface_quality, violation
from .optimizer import coarse_then_refine
from .types import OrientationResult


def non_dominated_front_2obj(objectives: np.ndarray) -> np.ndarray:
    """Fast O(N log N) non-dominated front for exactly 2 objectives.

    Args:
        objectives: (N, 2) array of objective values (lower is better).

    Returns:
        Array of indices of the Pareto-optimal points.
    """
    order = np.argsort(objectives[:, 0])
    sorted_obj = objectives[order]
    pareto = []
    min_b = np.inf
    for i in range(len(order)):
        if sorted_obj[i, 1] < min_b:
            pareto.append(order[i])
            min_b = sorted_obj[i, 1]
    return np.array(pareto, dtype=np.int32)


def non_dominated_sort(objectives: np.ndarray) -> list[np.ndarray]:
    """Non-dominated sorting of objective vectors.

    For 2 objectives, uses the fast O(N log N) algorithm.

    Args:
        objectives: (N, k) array of objective values (lower is better).

    Returns:
        List of arrays, each containing indices of one Pareto front.
    """
    try:
        from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
        fronts = fast_non_dominated_sort(objectives)
        return [np.array(f) for f in fronts]
    except ImportError:
        return _non_dominated_sort_naive(objectives)


def _non_dominated_sort_naive(objectives: np.ndarray) -> list[np.ndarray]:
    """Fallback non-dominated sorting without pymoo."""
    n = len(objectives)
    remaining = set(range(n))
    fronts = []
    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        fronts.append(np.array(front))
        remaining -= set(front)
    return fronts


_OBJECTIVE_REGISTRY = {
    "overhang": lambda normals, areas, vertices, candidates, **kw: overhang(
        normals, areas, candidates, angle=kw.get("overhang_angle", 45.0)
    ),
    "build_height": lambda normals, areas, vertices, candidates, **kw: build_height(
        vertices, candidates
    ),
    "surface_quality": lambda normals, areas, vertices, candidates, **kw: surface_quality(
        normals, areas, candidates, kw["critical_faces"]
    ),
}


def pareto_front(
    normals: np.ndarray,
    areas: np.ndarray,
    vertices: np.ndarray,
    critical_faces: np.ndarray | None = None,
    no_support_faces: np.ndarray | None = None,
    objectives: list[str] | None = None,
    n_samples: int = 500,
    n_pareto_steps: int = 50,
    overhang_angle: float = 45.0,
) -> list[OrientationResult]:
    """Compute a Pareto front by sweeping lambda for 2 objectives.

    Args:
        objectives: list of 2 objective names from {"overhang", "build_height", "surface_quality"}.
        n_pareto_steps: number of lambda values to sweep.

    Returns:
        List of non-dominated OrientationResult objects.
    """
    if objectives is None:
        objectives = ["overhang", "build_height"]
    if len(objectives) != 2:
        raise ValueError("pareto_front requires exactly 2 objectives")

    results = []
    lambdas = np.linspace(0, 1, n_pareto_steps)
    for lam in lambdas:
        result = coarse_then_refine(
            normals=normals,
            areas=areas,
            vertices=vertices,
            critical_faces=critical_faces,
            no_support_faces=no_support_faces,
            lam=lam,
            overhang_angle=overhang_angle,
            n_samples=n_samples,
            n_refine=3,
        )
        results.append(result)

    # Extract objective values and filter non-dominated
    obj_values = np.array([
        [r.overhang_area, r.build_height] for r in results
    ])

    # Map objective names to column indices
    obj_map = {"overhang": 0, "build_height": 1}
    cols = [obj_map[o] for o in objectives]
    obj_subset = obj_values[:, cols]

    fronts = non_dominated_sort(obj_subset)
    if len(fronts) == 0:
        return results
    pareto_indices = fronts[0]
    return [results[i] for i in pareto_indices]


def pairwise_pareto(
    normals: np.ndarray,
    areas: np.ndarray,
    vertices: np.ndarray,
    critical_faces: np.ndarray | None = None,
    no_support_faces: np.ndarray | None = None,
    objectives: list[str] | None = None,
    n_samples: int = 500,
    n_pareto_steps: int = 50,
    angular_tolerance: float = 3.0,
    overhang_angle: float = 45.0,
) -> list[OrientationResult]:
    """Pairwise Pareto intersection across k objectives.

    For each pair of objectives, computes the pairwise Pareto front.
    Returns results that appear on every pairwise front (within angular tolerance).

    Args:
        objectives: list of k objective names.
        angular_tolerance: degrees within which two gravity directions are considered equal.

    Returns:
        List of OrientationResult objects in the pairwise intersection.
    """
    if objectives is None:
        objectives = ["overhang", "build_height", "surface_quality"]
    if len(objectives) < 2:
        raise ValueError("pairwise_pareto requires at least 2 objectives")

    # Compute pairwise Pareto curves
    pair_curves: list[list[OrientationResult]] = []
    for obj_a, obj_b in combinations(objectives, 2):
        curve = pareto_front(
            normals=normals,
            areas=areas,
            vertices=vertices,
            critical_faces=critical_faces,
            no_support_faces=no_support_faces,
            objectives=[obj_a, obj_b],
            n_samples=n_samples,
            n_pareto_steps=n_pareto_steps,
            overhang_angle=overhang_angle,
        )
        pair_curves.append(curve)

    if len(pair_curves) < 2:
        return pair_curves[0] if pair_curves else []

    # Intersection: a point is in the intersection if it's within angular_tolerance
    # of a point on every pairwise curve
    cos_tol = np.cos(np.radians(angular_tolerance))

    # Use first curve as candidates, check against all others
    candidates = pair_curves[0]
    result = []

    for candidate in candidates:
        g = candidate.gravity_direction
        on_all_curves = True
        for curve in pair_curves[1:]:
            curve_gs = np.array([r.gravity_direction for r in curve])
            tree = cKDTree(curve_gs)
            # Use Euclidean distance as proxy; for small angles on S²,
            # ||a-b||² ≈ 2(1 - cos(angle))
            dist_threshold = np.sqrt(2 * (1 - cos_tol))
            dist, _ = tree.query(g)
            if dist > dist_threshold:
                on_all_curves = False
                break
        if on_all_curves:
            result.append(candidate)

    return result
