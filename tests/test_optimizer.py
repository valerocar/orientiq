import numpy as np

from orient_opt.optimizer import riemannian_gd


def test_riemannian_gd_convergence():
    """GD should converge to the known minimum of a simple quadratic on S²."""
    target = np.array([0.0, 0.0, 1.0])

    def objective(g):
        return np.sum((g - target) ** 2)

    def gradient(g):
        grad_e = 2 * (g - target)
        return grad_e - np.dot(grad_e, g) * g

    g0 = np.array([0.5, 0.5, np.sqrt(0.5)])
    g0 /= np.linalg.norm(g0)

    result = riemannian_gd(g0, objective, gradient, eta=0.1, tol=1e-8, max_iter=200)
    np.testing.assert_allclose(result, target, atol=1e-4)
