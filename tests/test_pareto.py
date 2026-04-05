import numpy as np

from orient_opt.pareto import non_dominated_sort, pareto_front


def test_non_dominated_sort_simple():
    """Basic non-dominated sorting correctness."""
    objectives = np.array([
        [1.0, 3.0],  # dominated by [1, 2]
        [1.0, 2.0],  # non-dominated
        [2.0, 1.0],  # non-dominated
        [3.0, 3.0],  # dominated
    ])
    fronts = non_dominated_sort(objectives)
    front0 = set(fronts[0].tolist())
    assert front0 == {1, 2}


def test_non_dominated_sort_single():
    objectives = np.array([[1.0, 2.0]])
    fronts = non_dominated_sort(objectives)
    assert len(fronts[0]) == 1


def test_pareto_front_returns_results(cube_mesh):
    """pareto_front should return a non-empty list of OrientationResult."""
    normals, areas, vertices = cube_mesh
    results = pareto_front(
        normals, areas, vertices,
        objectives=["overhang", "build_height"],
        n_samples=100,
        n_pareto_steps=5,
    )
    assert len(results) > 0
    for r in results:
        assert r.quaternion.shape == (4,)


def test_pareto_front_non_dominated(cube_mesh):
    """All returned results should be non-dominated."""
    normals, areas, vertices = cube_mesh
    results = pareto_front(
        normals, areas, vertices,
        objectives=["overhang", "build_height"],
        n_samples=100,
        n_pareto_steps=10,
    )
    obj = np.array([[r.overhang_area, r.build_height] for r in results])
    # Check no point is dominated by another
    for i in range(len(obj)):
        for j in range(len(obj)):
            if i == j:
                continue
            assert not (np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i])), (
                f"Point {i} is dominated by point {j}"
            )
