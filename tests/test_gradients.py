import numpy as np

from orient_opt.gradients import overhang_smooth_gradient
from orient_opt.objectives import overhang_smooth


def test_gradient_finite_difference():
    """Analytical gradient should match finite-difference approximation."""
    rng = np.random.default_rng(42)
    normals = rng.standard_normal((20, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    areas = rng.uniform(0.1, 1.0, 20)

    g = rng.standard_normal(3)
    g /= np.linalg.norm(g)

    grad = overhang_smooth_gradient(g, normals, areas, angle=45.0, beta=50.0)

    # Finite difference in tangent plane
    eps = 1e-5
    # Two orthogonal tangent vectors at g
    if abs(g[0]) < 0.9:
        v1 = np.cross(g, [1, 0, 0])
    else:
        v1 = np.cross(g, [0, 1, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(g, v1)
    v2 /= np.linalg.norm(v2)

    def _eval(g_):
        g_ = g_ / np.linalg.norm(g_)
        return overhang_smooth(normals, areas, g_.reshape(1, 3), angle=45.0, beta=50.0)[0]

    # Directional derivatives
    for v in [v1, v2]:
        fd = (_eval(g + eps * v) - _eval(g - eps * v)) / (2 * eps)
        analytical = np.dot(grad, v)
        np.testing.assert_allclose(analytical, fd, atol=1e-3,
            err_msg=f"Gradient mismatch in direction {v}")
