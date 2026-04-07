"""Tests for critical point detection on S².

Ground-truth function: f(g) = g[2]  (height of the gravity vector's z-component)
  - Global minimum at g = [0, 0, -1] (south pole), f = -1
  - Global maximum at g = [0, 0,  1] (north pole), f = +1
  - Every point on the equator (g[2] = 0) is a saddle — a degenerate circle.

Gradient: ∇_{S²}f(g) = e_z - (e_z·g)g = [0, 0, 1] - g[2]*g  (projected to tangent plane)
  - At north pole g=[0,0,1]: grad = [0,0,1] - 1*[0,0,1] = 0  ✓
  - At south pole g=[0,0,-1]: grad = [0,0,1] - (-1)*[0,0,-1] = [0,0,1]-[0,0,1] = 0  ✓
  - On equator (g[2]=0): grad = [0,0,1] - 0 = [0,0,1], projected = [0,0,1] - (0)*g = [0,0,1]
    Wait — on equator g[2]=0, so grad_eucl = [0,0,1], tangent = [0,0,1] - ([0,0,1]·g)g
    = [0,0,1] - 0*g = [0,0,1], ‖grad‖ = 1 ≠ 0.  Equator is NOT critical for f = g[2].

So for f(g) = g[2]: only two isolated critical points (poles), both non-degenerate.

For testing the degenerate-detection logic we use f(g) = g[2]² which has:
  - North and south poles as maxima (f=1)
  - Entire equator as a degenerate minimum (f=0)
"""

import numpy as np
import pytest

from orient_opt.critical_points import find_critical_points


# ── helpers ──────────────────────────────────────────────────────────────────

def _project_tangent(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project v onto the tangent plane of S² at g."""
    return v - (v @ g) * g


def f_linear(g: np.ndarray) -> float:
    """f(g) = g[2].  Critical points: north pole (max) and south pole (min)."""
    return float(g[2])


def grad_linear(g: np.ndarray) -> np.ndarray:
    """S² gradient of f(g) = g[2]."""
    ez = np.array([0.0, 0.0, 1.0])
    return _project_tangent(g, ez)


def f_quadratic(g: np.ndarray) -> float:
    """f(g) = g[2]².  Critical: poles (max, f=1), equatorial circle (degenerate min, f=0)."""
    return float(g[2] ** 2)


def grad_quadratic(g: np.ndarray) -> np.ndarray:
    """S² gradient of f(g) = g[2]²."""
    grad_eucl = np.array([0.0, 0.0, 2.0 * g[2]])
    return _project_tangent(g, grad_eucl)


# ── tests for f(g) = g[2] ────────────────────────────────────────────────────

def test_both_poles_found():
    """Both north [0,0,1] and south [0,0,-1] poles must be detected."""
    cps = find_critical_points(f_linear, grad_linear, n_samples=2000)
    gs = np.array([cp.g for cp in cps])

    has_north = any(np.dot(g, [0, 0, 1]) > 0.99 for g in gs)
    has_south = any(np.dot(g, [0, 0, -1]) > 0.99 for g in gs)

    assert has_north, "North pole not detected"
    assert has_south, "South pole not detected"


def test_no_antipodal_collapse():
    """Poles must appear as two separate critical points, not merged into one."""
    cps = find_critical_points(f_linear, grad_linear, n_samples=2000)
    gs = np.array([cp.g for cp in cps])

    north_pts = [g for g in gs if np.dot(g, [0, 0, 1]) > 0.99]
    south_pts = [g for g in gs if np.dot(g, [0, 0, -1]) > 0.99]

    assert len(north_pts) >= 1, "North pole missing"
    assert len(south_pts) >= 1, "South pole missing"


def test_pole_classification():
    """North pole should be classified as max, south pole as min."""
    cps = find_critical_points(f_linear, grad_linear, n_samples=2000)

    for cp in cps:
        if np.dot(cp.g, [0, 0, 1]) > 0.99:
            assert cp.cp_type == "max", f"North pole classified as {cp.cp_type!r}, expected 'max'"
        if np.dot(cp.g, [0, 0, -1]) > 0.99:
            assert cp.cp_type == "min", f"South pole classified as {cp.cp_type!r}, expected 'min'"


def test_f_values_at_poles():
    """f-values at poles must be close to ±1."""
    cps = find_critical_points(f_linear, grad_linear, n_samples=2000)

    for cp in cps:
        if np.dot(cp.g, [0, 0, 1]) > 0.99:
            assert abs(cp.f_value - 1.0) < 0.05, f"North pole f={cp.f_value:.4f}, expected ~1"
        if np.dot(cp.g, [0, 0, -1]) > 0.99:
            assert abs(cp.f_value + 1.0) < 0.05, f"South pole f={cp.f_value:.4f}, expected ~-1"


def test_grad_norms_small():
    """All returned critical points should have small residual gradient norm."""
    cps = find_critical_points(f_linear, grad_linear, n_samples=2000, grad_norm_tol=0.1)
    for cp in cps:
        assert cp.grad_norm < 0.15, f"grad_norm={cp.grad_norm:.4f} too large"


# ── tests for f(g) = g[2]² (degenerate equatorial minimum) ──────────────────

def test_degenerate_equator_detected():
    """f(g)=g[2]² has a degenerate minimum on the equator; at least one 'degenerate' point."""
    cps = find_critical_points(
        f_quadratic,
        grad_quadratic,
        n_samples=2000,
        grad_norm_tol=0.15,
        degenerate_min_cluster_size=3,
        degenerate_spread_deg=15.0,
    )
    types = [cp.cp_type for cp in cps]
    assert "degenerate" in types, (
        f"No degenerate critical set found for f=g[2]². Types: {types}"
    )


def test_quadratic_poles_are_maxima():
    """f(g)=g[2]² has maxima at both poles (f=1)."""
    cps = find_critical_points(
        f_quadratic,
        grad_quadratic,
        n_samples=2000,
        grad_norm_tol=0.15,
    )
    pole_cps = [cp for cp in cps if abs(abs(cp.g[2]) - 1.0) < 0.05]
    assert len(pole_cps) >= 1, "No pole critical points found for f=g[2]²"
    for cp in pole_cps:
        assert cp.cp_type in ("max", "degenerate"), (
            f"Pole classified as {cp.cp_type!r} for f=g[2]²"
        )
