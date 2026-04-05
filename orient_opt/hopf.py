import numpy as np


def sphere_to_quaternion(g: np.ndarray) -> np.ndarray:
    """Hopf section: gravity direction on S² to rotation quaternion.

    Returns the quaternion (w, x, y, z) that rotates (0, 0, 1) to g.
    Uses the z=0 gauge choice (Hopf section).

    Args:
        g: (3,) unit vector.

    Returns:
        (4,) quaternion (w, x, y, z).
    """
    nx, ny, nz = g

    s = np.sqrt(2 * (1 + nz))
    if s < 1e-6:
        # South pole singularity: rotate 180° about x-axis first,
        # then apply formula to the flipped direction.
        g_flipped = np.array([nx, -ny, -nz])
        q_flip = sphere_to_quaternion(g_flipped)
        # 180° about x: q_x = (0, 1, 0, 0)
        q_x = np.array([0.0, 1.0, 0.0, 0.0])
        return _quat_multiply(q_flip, q_x)
    w = s / 2
    x = -ny / s
    y = nx / s
    z = 0.0
    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 4x4 homogeneous rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),     0],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),     0],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y), 0],
        [0,                 0,                 0,                 1],
    ])


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
