import numpy as np

from orient_opt.hopf import quaternion_to_rotation_matrix, sphere_to_quaternion


def test_round_trip_z_axis():
    """Quaternion for (0,0,1) should be identity-like."""
    g = np.array([0.0, 0.0, 1.0])
    q = sphere_to_quaternion(g)
    R = quaternion_to_rotation_matrix(q)
    result = R[:3, :3] @ np.array([0, 0, 1.0])
    np.testing.assert_allclose(result, g, atol=1e-10)


def test_round_trip_random():
    """For random g, applying the quaternion to (0,0,1) should recover g."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        g = rng.standard_normal(3)
        g = g / np.linalg.norm(g)
        q = sphere_to_quaternion(g)
        R = quaternion_to_rotation_matrix(q)
        result = R[:3, :3] @ np.array([0, 0, 1.0])
        np.testing.assert_allclose(result, g, atol=1e-10)


def test_unit_quaternion():
    rng = np.random.default_rng(123)
    for _ in range(20):
        g = rng.standard_normal(3)
        g = g / np.linalg.norm(g)
        q = sphere_to_quaternion(g)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)


def test_z_component_zero():
    """The Hopf section gauge choice: z-component of quaternion should be 0."""
    g = np.array([0.6, 0.8, 0.0])
    g = g / np.linalg.norm(g)
    q = sphere_to_quaternion(g)
    # For non-south-pole points, z should be 0
    assert abs(q[3]) < 1e-10


def test_near_south_pole():
    """Near-south-pole directions should produce valid quaternions."""
    for nz in [-0.99, -0.999, -0.9999, -0.99999]:
        r = np.sqrt(max(1 - nz**2, 0))
        g = np.array([r, 0.0, nz])
        g = g / np.linalg.norm(g)
        q = sphere_to_quaternion(g)
        assert np.isfinite(q).all(), f"Non-finite quaternion for nz={nz}: {q}"
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)
        R = quaternion_to_rotation_matrix(q)
        result = R[:3, :3] @ np.array([0, 0, 1.0])
        np.testing.assert_allclose(result, g, atol=1e-6,
                                   err_msg=f"Round-trip failed for nz={nz}")


def test_south_pole():
    """South pole should not crash and should produce a valid quaternion."""
    g = np.array([0.0, 0.0, -1.0])
    q = sphere_to_quaternion(g)
    assert np.isfinite(q).all()
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)
    R = quaternion_to_rotation_matrix(q)
    result = R[:3, :3] @ np.array([0, 0, 1.0])
    np.testing.assert_allclose(result, g, atol=1e-10)
