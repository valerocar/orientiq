import numpy as np

from orient_opt.objectives import build_height, overhang, stability, violation


def test_overhang_cube_z_up(cube_mesh):
    """With gravity = (0,0,1), only the -Z face overhangs (normals face down)."""
    normals, areas, vertices = cube_mesh
    candidates = np.array([[0.0, 0.0, 1.0]])
    oh = overhang(normals, areas, candidates)[0]
    # The -Z face has 2 triangles, each area 0.5 → total 1.0
    np.testing.assert_allclose(oh, 1.0, atol=1e-10)


def test_overhang_cube_z_down(cube_mesh):
    """With gravity = (0,0,-1), only the +Z face overhangs."""
    normals, areas, vertices = cube_mesh
    candidates = np.array([[0.0, 0.0, -1.0]])
    oh = overhang(normals, areas, candidates)[0]
    np.testing.assert_allclose(oh, 1.0, atol=1e-10)


def test_build_height_cube(cube_mesh):
    """Cube build height along Z should be 1.0 (from -0.5 to 0.5)."""
    normals, areas, vertices = cube_mesh
    candidates = np.array([[0.0, 0.0, 1.0]])
    bh = build_height(vertices, candidates)[0]
    np.testing.assert_allclose(bh, 1.0, atol=1e-10)


def test_build_height_cube_diagonal(cube_mesh):
    """Build height along space diagonal should be sqrt(3)."""
    normals, areas, vertices = cube_mesh
    g = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    bh = build_height(vertices, g.reshape(1, 3))[0]
    np.testing.assert_allclose(bh, np.sqrt(3), atol=1e-10)


def test_violation_no_constraint(cube_mesh):
    """With no no_support_faces, violation should be zero."""
    normals, areas, vertices = cube_mesh
    candidates = np.array([[0.0, 0.0, 1.0]])
    viol = violation(normals, areas, candidates, no_support_faces=np.array([], dtype=int))
    np.testing.assert_allclose(viol[0], 0.0, atol=1e-10)


def test_violation_with_constraint(cube_mesh):
    """Mark top face (+Z, indices 2,3) as no-support. Gravity (0,0,-1) overhangs them."""
    normals, areas, vertices = cube_mesh
    candidates = np.array([[0.0, 0.0, -1.0]])
    no_support = np.array([2, 3])  # +Z face triangles
    viol = violation(normals, areas, candidates, no_support, angle=45.0)[0]
    np.testing.assert_allclose(viol, 1.0, atol=1e-10)


def test_overhang_single_candidate(cube_mesh):
    """Overhang should work with a single candidate."""
    normals, areas, vertices = cube_mesh
    candidates = np.array([[0.0, 0.0, 1.0]])
    result = overhang(normals, areas, candidates)
    assert result.shape == (1,)
    assert np.isfinite(result).all()


def test_stability_cube(cube_mesh):
    """Stability should return finite values for a cube mesh."""
    normals, areas, vertices = cube_mesh
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ])
    candidates = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    stab = stability(vertices, faces, candidates)
    assert stab.shape == (2,)
    assert np.isfinite(stab).all()
