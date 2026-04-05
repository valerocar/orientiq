import numpy as np
from scipy.special import expit


def overhang_smooth_gradient(
    g: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    angle: float = 45.0,
    beta: float = 50.0,
) -> np.ndarray:
    """Analytical gradient of smoothed overhang on S².

    Computes the Euclidean gradient and projects onto the tangent plane of S² at g.

    Args:
        g: (3,) current gravity direction (unit vector).
        normals: (F, 3) face normals.
        areas: (F,) face areas.
        angle: overhang angle in degrees.
        beta: sigmoid sharpness.

    Returns:
        (3,) Riemannian gradient on S².
    """
    threshold = -np.cos(np.radians(angle))
    dots = normals @ g  # (F,)
    u = beta * (-dots - threshold)
    sig = expit(u)
    sig_deriv = beta * sig * (1 - sig)  # sigmoid'(u) * beta
    # Euclidean gradient: -sum(A_f * sigmoid'(u_f) * n_f)
    weights = areas * sig_deriv  # (F,)
    grad_e = -(weights[:, None] * normals).sum(axis=0)  # (3,)
    # Project to tangent plane of S² at g
    grad_s2 = grad_e - np.dot(grad_e, g) * g
    return grad_s2


def build_height_gradient(
    g: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """Subgradient of build height on S².

    Build height H(g) = max(v·g) - min(v·g).
    Gradient is v_max - v_min projected onto S² tangent plane.

    Returns:
        (3,) Riemannian gradient on S².
    """
    proj = vertices @ g  # (V,)
    v_max = vertices[np.argmax(proj)]
    v_min = vertices[np.argmin(proj)]
    grad_e = v_max - v_min
    grad_s2 = grad_e - np.dot(grad_e, g) * g
    return grad_s2
