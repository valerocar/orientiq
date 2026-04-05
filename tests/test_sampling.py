import numpy as np

from orient_opt.sampling import fibonacci_sphere


def test_unit_norm():
    pts = fibonacci_sphere(500)
    norms = np.linalg.norm(pts, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_shape():
    pts = fibonacci_sphere(100)
    assert pts.shape == (100, 3)


def test_covers_hemisphere():
    """Points should cover both hemispheres."""
    pts = fibonacci_sphere(500)
    assert pts[:, 2].min() < -0.99
    assert pts[:, 2].max() > 0.99


def test_single_point():
    """fibonacci_sphere(1) should return a single valid unit vector."""
    pts = fibonacci_sphere(1)
    assert pts.shape == (1, 3)
    np.testing.assert_allclose(np.linalg.norm(pts[0]), 1.0, atol=1e-12)


def test_two_points():
    """fibonacci_sphere(2) should return two antipodal points."""
    pts = fibonacci_sphere(2)
    assert pts.shape == (2, 3)
    norms = np.linalg.norm(pts, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_no_nan():
    """No NaN values in output for various sizes."""
    for n in [1, 2, 3, 10, 500]:
        pts = fibonacci_sphere(n)
        assert np.isfinite(pts).all(), f"NaN/inf found for n={n}"


def test_equidistribution():
    """Mean pairwise angular distance should be close to theoretical sqrt(4pi/N)."""
    n = 200
    pts = fibonacci_sphere(n)
    # Check that no two points are unreasonably close
    # For N=200, min angular distance should be > ~5 degrees
    dots = pts @ pts.T
    np.fill_diagonal(dots, -2)  # exclude self
    max_cos = dots.max()
    min_angle_deg = np.degrees(np.arccos(np.clip(max_cos, -1, 1)))
    assert min_angle_deg > 3.0, f"Min angular separation too small: {min_angle_deg}°"
