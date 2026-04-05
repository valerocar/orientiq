"""Mesh loading and preprocessing via trimesh."""

import numpy as np


def load_mesh(path: str):
    """Load a mesh from file (STL, OBJ, etc.) via trimesh.

    Returns:
        trimesh.Trimesh object.
    """
    import trimesh
    mesh = trimesh.load(path, force="mesh")
    return mesh


def extract_arrays(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract face normals, face areas, and vertex positions from a trimesh mesh.

    Returns:
        (normals, areas, vertices) — (F,3), (F,), (V,3) arrays.
    """
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return normals, areas, vertices


def region_labels_from_colors(
    mesh,
    colors: list[tuple],
    tolerance: float = 10.0,
) -> np.ndarray:
    """Identify faces matching given RGB colors.

    Args:
        mesh: trimesh.Trimesh with face colors.
        colors: list of (R, G, B) tuples (0-255 scale).
        tolerance: max Euclidean distance in RGB space for a match.

    Returns:
        (N,) array of face indices that match any of the given colors.
    """
    face_colors = np.asarray(mesh.visual.face_colors[:, :3], dtype=np.float64)
    mask = np.zeros(len(face_colors), dtype=bool)
    for color in colors:
        color = np.array(color, dtype=np.float64)
        dist = np.linalg.norm(face_colors - color, axis=1)
        mask |= dist < tolerance
    return np.where(mask)[0]
