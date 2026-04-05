import numpy as np

from orient_opt import optimize_orientation


def test_cube_build_height_face_aligned(cube_mesh):
    """For a cube, height-only optimization should yield face-aligned (height=1.0)."""
    normals, areas, vertices = cube_mesh
    # With lam=0, only build height matters. Axis-aligned gives height=1.0 (minimum).
    result = optimize_orientation(
        normals, areas, vertices,
        lam=0.0,
        n_samples=500,
        n_refine=5,
    )
    g = result.gravity_direction

    axis_directions = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=float)
    dots = axis_directions @ g
    best_alignment = np.abs(dots).max()
    assert best_alignment > 0.99, (
        f"Optimal gravity {g} not face-aligned. Best alignment: {best_alignment}"
    )


def test_cube_build_height_optimal(cube_mesh):
    """For a cube, height-minimizing orientation should also be face-aligned (height=1)."""
    normals, areas, vertices = cube_mesh
    result = optimize_orientation(
        normals, areas, vertices,
        lam=0.0,  # height only
        n_samples=500,
        n_refine=5,
    )
    # Face-aligned gives height=1.0, diagonal gives sqrt(3) ≈ 1.73
    assert result.build_height < 1.05


def test_result_fields(cube_mesh):
    """Check that OrientationResult has all expected fields with correct shapes."""
    normals, areas, vertices = cube_mesh
    result = optimize_orientation(normals, areas, vertices)
    assert result.quaternion.shape == (4,)
    assert result.rotation_matrix.shape == (4, 4)
    assert result.gravity_direction.shape == (3,)
    assert np.isscalar(result.overhang_area)
    assert np.isscalar(result.build_height)
    assert result.feasible_count > 0
    assert result.all_objectives.ndim == 2


def test_no_support_constraint(cube_mesh):
    """Marking bottom face as no-support should prevent gravity pointing up."""
    normals, areas, vertices = cube_mesh
    # Bottom face (-Z) = indices 0, 1. These face downward (normal = [0,0,-1]).
    # With gravity (0,0,1), dot = -1 < threshold → overhang. Should be blocked.
    result = optimize_orientation(
        normals, areas, vertices,
        no_support_faces=np.array([0, 1]),
        lam=1.0,
        n_samples=500,
    )
    g = result.gravity_direction
    # Gravity should NOT be close to (0,0,1) since that overhangs the no-support face
    assert g[2] < 0.8, f"Expected gravity away from +Z, got {g}"
