from dataclasses import dataclass

import numpy as np


@dataclass
class OrientationResult:
    quaternion: np.ndarray          # (4,) — (w, x, y, z)
    rotation_matrix: np.ndarray     # (4, 4) — homogeneous transform
    gravity_direction: np.ndarray   # (3,) — optimal g on S²
    overhang_area: float
    build_height: float
    surface_quality: float          # NaN if no critical faces
    feasible_count: int             # how many of n_samples were feasible
    all_objectives: np.ndarray      # (feasible_count, k) for debugging
