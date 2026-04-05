import numpy as np


def fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n approximately equidistributed points on S².

    Uses the Fibonacci spiral method with the golden ratio.

    Returns:
        (n, 3) array of unit vectors.
    """
    if n < 2:
        return np.array([[0.0, 0.0, 1.0]])
    golden_ratio = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    nz = 1 - 2 * i / (n - 1)
    phi = 2 * np.pi * i / (golden_ratio ** 2)
    r = np.sqrt(np.maximum(1 - nz ** 2, 0))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), nz])
    return points
