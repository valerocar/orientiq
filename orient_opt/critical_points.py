"""Critical point detection on S² for orientation objective functions.

A critical point g* satisfies ‖∇_{S²}f(g*)‖ = 0.  The algorithm:

1. Sample 2000 Fibonacci points and compute gradient norms.
2. Find local minima of ‖∇f‖ on the sphere k-NN graph.
3. Refine each candidate via Riemannian GD on ‖∇f‖².
4. Cluster nearby refined points (angular distance < cluster_angle_deg).
5. Classify each cluster: min / max / saddle / degenerate (extended set).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sampling import fibonacci_sphere


@dataclass
class CriticalPoint:
    """A critical point of the objective on S²."""

    g: np.ndarray       # (3,) unit vector
    cp_type: str        # "min" | "max" | "saddle" | "degenerate"
    f_value: float
    grad_norm: float    # residual ‖∇f‖ after refinement


def _refine_gradient_norm(
    g_init: np.ndarray,
    gradient_fn,
    eta: float = 0.05,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Riemannian GD on ‖∇f‖² to find a zero of the gradient.

    Returns:
        (g, grad_norm) — refined point and residual gradient norm.
    """
    g = g_init.copy()
    for _ in range(max_iter):
        grad = gradient_fn(g)
        gn = float(np.linalg.norm(grad))
        if gn < tol:
            return g, gn
        # Step in direction that reduces ‖grad‖ (gradient of ‖∇f‖² on S²)
        g = g - eta * grad
        g = g / np.linalg.norm(g)
    gn = float(np.linalg.norm(gradient_fn(g)))
    return g, gn


def _build_knn(points: np.ndarray, k: int) -> np.ndarray:
    """Return (N, k) index array of k nearest neighbours on S² (by dot product).

    Excludes the point itself.
    """
    dots = points @ points.T  # (N, N)
    # Highest dot product = closest on sphere.  Exclude self (set diagonal to -2).
    np.fill_diagonal(dots, -2.0)
    return np.argsort(dots, axis=1)[:, -k:]  # (N, k), descending


def _cluster(points: np.ndarray, cluster_angle_deg: float) -> list[list[int]]:
    """Greedy clustering by angular distance.  Uses dot(p, r) > cos(tol) — NOT abs.

    Returns list of clusters, each a list of indices into `points`.
    """
    cos_tol = np.cos(np.radians(cluster_angle_deg))
    assigned = np.full(len(points), -1, dtype=int)
    cluster_id = 0
    for i in range(len(points)):
        if assigned[i] >= 0:
            continue
        assigned[i] = cluster_id
        for j in range(i + 1, len(points)):
            if assigned[j] >= 0:
                continue
            # Same direction only — NOT antipodal (no abs here)
            if points[i] @ points[j] > cos_tol:
                assigned[j] = cluster_id
        cluster_id += 1

    clusters: list[list[int]] = [[] for _ in range(cluster_id)]
    for idx, cid in enumerate(assigned):
        clusters[cid].append(idx)
    return clusters


def _classify(
    g: np.ndarray,
    objective_fn,
    neighbor_gs: np.ndarray,
) -> str:
    """Classify a critical point as min / max / saddle / degenerate.

    Args:
        g: the critical point (3,).
        objective_fn: callable(g) -> float.
        neighbor_gs: (k, 3) nearest neighbours from the full sample grid.

    Returns:
        "min" | "max" | "saddle" | "degenerate"
    """
    f0 = objective_fn(g)
    f_neighbors = np.array([objective_fn(nb) for nb in neighbor_gs])

    n_higher = int((f_neighbors > f0).sum())
    n_lower = int((f_neighbors < f0).sum())

    if n_lower == 0:
        return "min"
    if n_higher == 0:
        return "max"
    return "saddle"


def find_critical_points(
    objective_fn,
    gradient_fn,
    n_samples: int = 2000,
    k_neighbors: int = 8,
    cluster_angle_deg: float = 5.0,
    grad_norm_tol: float = 0.1,
    refine_eta: float = 0.05,
    refine_max_iter: int = 300,
    degenerate_spread_deg: float = 20.0,
    degenerate_min_cluster_size: int = 5,
) -> list[CriticalPoint]:
    """Find all critical points of f on S².

    Args:
        objective_fn: callable(g: np.ndarray shape (3,)) -> float.
        gradient_fn: callable(g: np.ndarray shape (3,)) -> np.ndarray shape (3,),
            returning the S² (tangent-projected) gradient.
        n_samples: number of Fibonacci sample points for coarse search.
        k_neighbors: neighbourhood size for local-minimum detection.
        cluster_angle_deg: points within this angular distance are merged.
        grad_norm_tol: gradient norm threshold to accept a point as critical.
        refine_eta: step size for gradient-norm refinement.
        refine_max_iter: max iterations for gradient-norm refinement.
        degenerate_spread_deg: clusters spanning more than this angle with
            more than `degenerate_min_cluster_size` members are flagged as
            "degenerate" (extended critical set, e.g. equatorial line).
        degenerate_min_cluster_size: minimum cluster size for degenerate flag.

    Returns:
        List of CriticalPoint, one per distinct critical point / degenerate set.
    """
    samples = fibonacci_sphere(n_samples)  # (N, 3)

    # --- Step 1: compute gradient norms at all sample points ---
    grad_norms = np.array([np.linalg.norm(gradient_fn(g)) for g in samples])

    # --- Step 2: find local minima of gradient norm on k-NN graph ---
    knn = _build_knn(samples, k_neighbors)  # (N, k)
    neighbor_norms = grad_norms[knn]         # (N, k)
    is_local_min = np.all(grad_norms[:, None] <= neighbor_norms, axis=1)
    seed_indices = np.where(is_local_min)[0]

    if len(seed_indices) == 0:
        return []

    # --- Step 3: refine each seed ---
    refined_gs = []
    refined_norms = []
    for idx in seed_indices:
        g_ref, gn = _refine_gradient_norm(
            samples[idx], gradient_fn, eta=refine_eta, max_iter=refine_max_iter
        )
        refined_gs.append(g_ref)
        refined_norms.append(gn)

    refined_gs = np.array(refined_gs)       # (M, 3)
    refined_norms = np.array(refined_norms) # (M,)

    # --- Step 4: cluster (no abs — same direction only) ---
    clusters = _cluster(refined_gs, cluster_angle_deg)

    cos_spread = np.cos(np.radians(degenerate_spread_deg))

    critical_points: list[CriticalPoint] = []
    for cluster in clusters:
        cluster_gs = refined_gs[cluster]       # (C, 3)
        cluster_gns = refined_norms[cluster]   # (C,)

        # Representative: member with smallest gradient norm
        best = int(np.argmin(cluster_gns))
        g_rep = cluster_gs[best]
        gn_rep = cluster_gns[best]

        # Skip if still far from critical
        if gn_rep > grad_norm_tol:
            continue

        f_val = float(objective_fn(g_rep))

        # Check for degenerate (extended) critical set
        if len(cluster) >= degenerate_min_cluster_size:
            # Compute pairwise angular spread within cluster
            dots = cluster_gs @ cluster_gs.T
            min_dot = float(dots.min())
            if min_dot < cos_spread:
                critical_points.append(CriticalPoint(
                    g=g_rep, cp_type="degenerate", f_value=f_val, grad_norm=gn_rep
                ))
                continue

        # Classify using neighbourhood from full sample grid
        # Find k nearest sample points to g_rep
        dot_to_samples = samples @ g_rep
        top_k_idx = np.argsort(dot_to_samples)[-k_neighbors:]
        neighbor_gs = samples[top_k_idx]

        cp_type = _classify(g_rep, objective_fn, neighbor_gs)
        critical_points.append(CriticalPoint(
            g=g_rep, cp_type=cp_type, f_value=f_val, grad_norm=gn_rep
        ))

    # --- Step 5: second-pass degeneracy detection across clusters ---
    # If >= degenerate_min_cluster_size critical points of the same type share
    # nearly the same f-value AND collectively span > degenerate_spread_deg,
    # they form an extended (degenerate) critical set.
    if len(critical_points) >= degenerate_min_cluster_size:
        f_values = np.array([cp.f_value for cp in critical_points])
        cp_gs = np.array([cp.g for cp in critical_points])
        f_tol = 0.01 * (f_values.max() - f_values.min() + 1e-12)

        for i, cp in enumerate(critical_points):
            if cp.cp_type == "degenerate":
                continue
            same_f = np.where(np.abs(f_values - cp.f_value) <= f_tol)[0]
            if len(same_f) < degenerate_min_cluster_size:
                continue
            group_gs = cp_gs[same_f]
            dots = group_gs @ group_gs.T
            min_dot = float(dots.min())
            if min_dot < cos_spread:
                for idx in same_f:
                    critical_points[idx] = CriticalPoint(
                        g=critical_points[idx].g,
                        cp_type="degenerate",
                        f_value=critical_points[idx].f_value,
                        grad_norm=critical_points[idx].grad_norm,
                    )

    return critical_points
