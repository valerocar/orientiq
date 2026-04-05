"""Shared test fixtures: simple meshes for testing."""

import numpy as np
import pytest


@pytest.fixture
def cube_mesh():
    """Unit cube centered at origin. 12 triangular faces, 8 vertices."""
    vertices = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ])
    # 6 faces, each split into 2 triangles
    faces = np.array([
        # -Z face
        [0, 2, 1], [0, 3, 2],
        # +Z face
        [4, 5, 6], [4, 6, 7],
        # -Y face
        [0, 1, 5], [0, 5, 4],
        # +Y face
        [2, 3, 7], [2, 7, 6],
        # -X face
        [0, 4, 7], [0, 7, 3],
        # +X face
        [1, 2, 6], [1, 6, 5],
    ])
    normals = np.array([
        [0, 0, -1], [0, 0, -1],
        [0, 0,  1], [0, 0,  1],
        [0, -1, 0], [0, -1, 0],
        [0,  1, 0], [0,  1, 0],
        [-1, 0, 0], [-1, 0, 0],
        [ 1, 0, 0], [ 1, 0, 0],
    ], dtype=float)
    areas = np.full(12, 0.5)  # each triangle is half of a unit square
    return normals, areas, vertices
