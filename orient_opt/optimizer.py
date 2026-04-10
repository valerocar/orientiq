import numpy as np

from .gradients import build_height_gradient, overhang_smooth_gradient
from .hopf import quaternion_to_rotation_matrix, sphere_to_quaternion
from .objectives import build_height, overhang, overhang_smooth, violation
from .sampling import fibonacci_sphere
from .types import OrientationResult


def riemannian_gd(
    g_init: np.ndarray,
    objective_fn,
    gradient_fn,
    eta: float = 0.05,
    tol: float = 1e-6,
    max_iter: int = 300,
) -> np.ndarray:
    """Riemannian gradient descent on S².

    Args:
        g_init: (3,) starting point on S².
        objective_fn: callable(g) -> scalar.
        gradient_fn: callable(g) -> (3,) S² gradient.
        eta: step size.
        tol: convergence tolerance on gradient norm.
        max_iter: maximum iterations.

    Returns:
        (3,) optimized point on S².
    """
    g = g_init.copy()
    for _ in range(max_iter):
        grad = gradient_fn(g)
        if np.linalg.norm(grad) < tol:
            break
        g = g - eta * grad
        g = g / np.linalg.norm(g)
    return g


def coarse_then_refine(
    normals: np.ndarray,
    areas: np.ndarray,
    vertices: np.ndarray,
    critical_faces: np.ndarray | None = None,
    no_support_faces: np.ndarray | None = None,
    lam: float = 0.7,
    overhang_angle: float = 45.0,
    n_samples: int = 500,
    beta: float = 50.0,
    eta: float = 0.05,
    tol: float = 1e-6,
    max_iter: int = 300,
) -> OrientationResult:
    """Full optimization pipeline: sample, evaluate, filter, refine, select best.

    Returns:
        OrientationResult with optimal quaternion and diagnostics.

    Raises:
        ValueError: if all candidates are infeasible.
    """
    candidates = fibonacci_sphere(n_samples)

    # Evaluate objectives on all candidates
    oh = overhang(normals, areas, candidates, angle=overhang_angle)
    bh = build_height(vertices, candidates)

    # Filter infeasible
    if no_support_faces is not None and len(no_support_faces) > 0:
        viol = violation(normals, areas, candidates, no_support_faces, angle=overhang_angle)
        feasible_mask = viol == 0
    else:
        feasible_mask = np.ones(len(candidates), dtype=bool)

    feasible_count = feasible_mask.sum()
    if feasible_count == 0:
        raise ValueError(
            "All candidates are infeasible. The no_support_faces constraint "
            "may be too aggressive."
        )

    feas_candidates = candidates[feasible_mask]
    feas_oh = oh[feasible_mask]
    feas_bh = bh[feasible_mask]

    # Normalize objectives to [0, 1]
    oh_min, oh_max = feas_oh.min(), feas_oh.max()
    bh_min, bh_max = feas_bh.min(), feas_bh.max()
    oh_range = oh_max - oh_min if oh_max > oh_min else 1.0
    bh_range = bh_max - bh_min if bh_max > bh_min else 1.0
    oh_hat = (feas_oh - oh_min) / oh_range
    bh_hat = (feas_bh - bh_min) / bh_range

    # Scalarized objective
    scores = lam * oh_hat + (1 - lam) * bh_hat

    # Collect all objectives for debugging
    all_objectives = np.column_stack([feas_oh, feas_bh])

    def _objective(g):
        g2 = g.reshape(1, 3)
        o = overhang_smooth(normals, areas, g2, angle=overhang_angle, beta=beta)[0]
        h = build_height(vertices, g2)[0]
        o_hat = (o - oh_min) / oh_range
        h_hat = (h - bh_min) / bh_range
        return lam * o_hat + (1 - lam) * h_hat

    def _gradient(g):
        grad_o = overhang_smooth_gradient(g, normals, areas, angle=overhang_angle, beta=beta)
        grad_h = build_height_gradient(g, vertices)
        return lam * grad_o / oh_range + (1 - lam) * grad_h / bh_range

    # Refine all feasible candidates via gradient descent
    best_score = float("inf")
    best_g = feas_candidates[np.argmin(scores)].copy()
    for i in range(len(feas_candidates)):
        g_refined = riemannian_gd(feas_candidates[i], _objective, _gradient, eta=eta, tol=tol, max_iter=max_iter)
        score = _objective(g_refined)
        if score < best_score:
            best_score = score
            best_g = g_refined

    # Compute final objective values at best_g
    g2 = best_g.reshape(1, 3)
    final_oh = overhang(normals, areas, g2, angle=overhang_angle)[0]
    final_bh = build_height(vertices, g2)[0]
    final_sq = float("nan")
    if critical_faces is not None and len(critical_faces) > 0:
        from .objectives import surface_quality
        final_sq = surface_quality(normals, areas, g2, critical_faces)[0]

    q = sphere_to_quaternion(best_g)
    R = quaternion_to_rotation_matrix(q)

    return OrientationResult(
        quaternion=q,
        rotation_matrix=R,
        gravity_direction=best_g,
        overhang_area=final_oh,
        build_height=final_bh,
        surface_quality=final_sq,
        feasible_count=int(feasible_count),
        all_objectives=all_objectives,
    )
