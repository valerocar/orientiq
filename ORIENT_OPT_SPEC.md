# Orientation Optimizer — Implementation Specification

## 1. What This Is

A Python library that finds the optimal 3D printing orientation for a triangle mesh. Given a mesh (face normals, face areas, vertex positions) and optional region labels, it returns a rotation quaternion that minimizes support material, build height, and surface quality degradation — subject to hard constraints on functional surfaces.

The solver exploits a symmetry reduction: rotation about the gravity axis is irrelevant to all print objectives, so the search space is S² (the unit sphere — 2 degrees of freedom) rather than SO(3) (3 degrees of freedom). This is formalized via the Hopf fibration S³ → S².

## 2. Mathematical Foundation

### 2.1 Configuration Space

The search variable is a unit vector **g** ∈ S² ⊂ ℝ³, representing the gravity direction in the part's local frame. All print objectives depend only on the angles between each face normal and **g**.

### 2.2 Quaternion Reconstruction (Hopf Section)

Given the optimal **g** = (nx, ny, nz), the rotation quaternion that aligns **g** with the build direction (0,0,1) is:

```
q = (w, x, y, z) where:
    w = sqrt((1 + nz) / 2)
    x = -ny / sqrt(2 * (1 + nz))
    y =  nx / sqrt(2 * (1 + nz))
    z = 0
```

This is valid for nz > -1. For nz ≈ -1 (south pole), rotate the mesh 180° about any horizontal axis first, then apply the formula. The z=0 condition is the gauge choice that eliminates the irrelevant z-rotation.

### 2.3 Objective Functions

All objectives take **g** ∈ S² and return a scalar. All are computed from face normals, face areas, and vertex positions.

**Overhang area (hard threshold):**
```
O(g) = sum(A_f  for f where  n_f · g < -cos(alpha_0))
```
Default alpha_0 = 45° = π/4, so threshold = -cos(45°) ≈ -0.7071.

**Overhang area (smoothed, for gradient descent):**
```
O_smooth(g) = sum(A_f * sigmoid(beta * (-n_f · g - cos(alpha_0)))  for all f)
```
where sigmoid(t) = 1 / (1 + exp(-t)), beta = 50 (sharpness parameter).

**Gradient of smoothed overhang on S²:**
```
# Euclidean gradient:
grad_E = -sum(A_f * sigmoid'(u_f) * n_f  for all f)
    where u_f = -n_f · g - cos(alpha_0)
    and sigmoid'(u) = beta * sigmoid(u) * (1 - sigmoid(u))

# Project to tangent plane of S² at g:
grad_S2 = grad_E - (grad_E · g) * g
```

**Build height:**
```
H(g) = max(v · g for all vertices v) - min(v · g for all vertices v)
```

**Surface quality penalty:**
```
Q(g) = sum(A_f * (1 - |n_f · g|)²  for f in critical_faces)
```

**Functional surface violation (hard constraint):**
```
V(g) = sum(A_f  for f in no_support_faces  where  n_f · g < -cos(alpha_0))
```
Any g with V(g) > 0 is infeasible.

### 2.4 Scalarized Objective

After filtering infeasible points:
```
J(g) = lambda * O_hat(g) + (1 - lambda) * H_hat(g)
```
where O_hat and H_hat are normalized to [0, 1] over the feasible set. Default lambda = 0.7.

### 2.5 Fibonacci Spiral Sampling

Generate N approximately equidistributed points on S²:
```
for i in range(N):
    nz = 1 - 2*i / (N - 1)
    phi = 2 * pi * i / golden_ratio²
    r = sqrt(1 - nz²)
    g[i] = (r * cos(phi), r * sin(phi), nz)
```
where golden_ratio = (1 + sqrt(5)) / 2. Default N = 500.

### 2.6 Riemannian Gradient Descent on S²

```
for k in range(max_iter):
    grad_E = euclidean_gradient(J, g)
    grad_S = grad_E - (grad_E @ g) * g        # tangent plane projection
    if norm(grad_S) < tol:
        break
    g = g - eta * grad_S
    g = g / norm(g)                            # retraction (re-normalize)
```
Default: eta = 0.01, tol = 1e-6, max_iter = 100.

### 2.7 Non-Dominated Sorting (for Pareto front)

Given a set of points with objective vectors, non-dominated sorting identifies which points are Pareto optimal. Use `pymoo.util.nds.fast_non_dominated_sort` or implement directly:

A point **a** dominates **b** if a_i ≤ b_i for all i and a_j < b_j for at least one j.

### 2.8 Pairwise Pareto Intersection

For k objectives, compute C(k,2) pairwise Pareto curves. Each curve is obtained by:
1. For a pair (F_i, F_j), sweep lambda from 0 to 1 in ~50 steps.
2. For each lambda, run the full coarse-then-refine pipeline with J = lambda * F_i_hat + (1-lambda) * F_j_hat.
3. Collect the resulting (g, objective_vector) pairs.
4. Run non-dominated sorting on the (F_i, F_j) values to get the pairwise Pareto front.

Intersection: a point g is in the pairwise intersection if it lies within angular tolerance epsilon (default 3°) of a point on every pairwise Pareto curve. Use scipy.spatial.cKDTree for efficient nearest-neighbour queries in ℝ³.


## 3. Architecture

### 3.1 Module Structure

```
orient_opt/
├── __init__.py          # Public API: optimize_orientation, pareto_front, pairwise_pareto
├── sampling.py          # fibonacci_sphere(n) -> (N, 3) array
├── objectives.py        # overhang, overhang_smooth, build_height, surface_quality, violation
├── gradients.py         # overhang_smooth_gradient, surface_quality_gradient
├── optimizer.py         # riemannian_gd, coarse_then_refine, optimize_single
├── pareto.py            # non_dominated_sort, pareto_front_2obj, pairwise_intersection
├── hopf.py              # sphere_to_quaternion, quaternion_to_matrix
├── preprocess.py        # load_mesh (trimesh wrapper), extract normals/areas/vertices
└── types.py             # OrientationResult dataclass
```

### 3.2 Data Flow

```
STL/OBJ file
    ↓  (preprocess.py — trimesh)
normals: (F, 3), areas: (F,), vertices: (V, 3), labels: (F,)
    ↓  (sampling.py)
candidates: (N, 3) points on S²
    ↓  (objectives.py — vectorized)
objective_matrix: (N, k) — all objectives at all candidates
    ↓  (filter infeasible)
feasible_candidates: (M, 3), feasible_objectives: (M, k)
    ↓  (rank + shortlist top K)
shortlist: (K, 3)
    ↓  (optimizer.py — gradient descent per candidate)
refined: (K, 3)
    ↓  (select best)
g_star: (3,)
    ↓  (hopf.py)
quaternion: (4,)
```

### 3.3 Key Design Decisions

- **All objectives are batch-evaluated.** The function signature is `overhang(normals, areas, candidates) -> (N,)` where candidates is (N, 3). This is a single matrix multiply: `dots = candidates @ normals.T` gives an (N, F) matrix. All thresholding and summing happens on this matrix. No loops over candidates.

- **Gradient is also batched** over faces but sequential over gradient descent iterations (each iteration depends on the previous).

- **Normalization of objectives for scalarization** is computed once from the coarse grid values: `O_hat = (O - O_min) / (O_max - O_min)` over the feasible set. This avoids recomputing normalization at each gradient step.

- **The solver never modifies the mesh.** Input arrays are read-only. The output is a quaternion, not a transformed mesh.


## 4. Public API

### 4.1 Main Function

```python
def optimize_orientation(
    normals: np.ndarray,          # (F, 3) face normals, unit vectors
    areas: np.ndarray,            # (F,) face areas
    vertices: np.ndarray,         # (V, 3) vertex positions
    critical_faces: np.ndarray = None,      # (C,) indices into faces
    no_support_faces: np.ndarray = None,    # (S,) indices into faces
    lam: float = 0.7,            # trade-off parameter, 0=height, 1=overhang
    overhang_angle: float = 45.0, # degrees
    n_samples: int = 500,
    beta: float = 50.0,          # sigmoid sharpness
    quality_threshold: float = None,  # auto-computed if None
    n_refine: int = 5,           # number of candidates for gradient refinement
) -> OrientationResult:
    ...
```

### 4.2 Result Dataclass

```python
@dataclass
class OrientationResult:
    quaternion: np.ndarray        # (4,) — (w, x, y, z)
    rotation_matrix: np.ndarray   # (4, 4) — homogeneous transform
    gravity_direction: np.ndarray # (3,) — optimal g on S²
    overhang_area: float
    build_height: float
    surface_quality: float        # NaN if no critical faces
    feasible_count: int           # how many of n_samples were feasible
    all_objectives: np.ndarray    # (feasible_count, k) for debugging
```

### 4.3 Pareto Front Function

```python
def pareto_front(
    normals: np.ndarray,
    areas: np.ndarray,
    vertices: np.ndarray,
    critical_faces: np.ndarray = None,
    no_support_faces: np.ndarray = None,
    objectives: list[str] = ["overhang", "build_height"],
    n_samples: int = 500,
    n_pareto_steps: int = 50,     # lambda sweep resolution
    overhang_angle: float = 45.0,
) -> list[OrientationResult]:
    ...
```

### 4.4 Pairwise Intersection Function

```python
def pairwise_pareto(
    normals: np.ndarray,
    areas: np.ndarray,
    vertices: np.ndarray,
    critical_faces: np.ndarray = None,
    no_support_faces: np.ndarray = None,
    objectives: list[str] = ["overhang", "build_height", "surface_quality"],
    n_samples: int = 500,
    n_pareto_steps: int = 50,
    angular_tolerance: float = 3.0,  # degrees
    overhang_angle: float = 45.0,
) -> list[OrientationResult]:
    ...
```

### 4.5 Convenience: From Mesh File

```python
def optimize_from_file(
    path: str,
    critical_face_colors: list[tuple] = None,  # RGB tuples identifying critical regions
    **kwargs,
) -> OrientationResult:
    """Load mesh via trimesh, extract arrays, call optimize_orientation."""
    ...
```


## 5. Dependencies

```
# Core (required)
numpy>=1.24

# Mesh I/O (required for optimize_from_file, not for optimize_orientation)
trimesh>=4.0

# Pareto front computation
pymoo>=0.6    # only pymoo.util.nds.fast_non_dominated_sort is used

# Pairwise intersection (nearest-neighbor queries)
scipy>=1.10   # scipy.spatial.cKDTree
```

No PyTorch, no JAX, no autograd. The gradient is closed-form.


## 6. Testing Strategy

### 6.1 Unit Tests

- **sampling.py**: Fibonacci spiral points all have unit norm; mean pairwise angular distance matches theoretical prediction sqrt(4π/N).
- **hopf.py**: For any g ∈ S², verify that applying quaternion σ(g) to (0,0,1) recovers g. Verify quaternion has unit norm. Verify z-component is 0.
- **objectives.py**: For a known simple mesh (cube, tetrahedron), verify overhang area by hand for axis-aligned gravity directions.
- **gradients.py**: Finite-difference check of the analytical gradient against numerical gradient for random g and random normals/areas.
- **optimizer.py**: Starting from a point near a known optimum (e.g., a flat-bottomed box where the optimal orientation is trivially the identity), verify convergence within tolerance.

### 6.2 Integration Tests

- **Cube**: optimal orientation for overhang minimization should be face-aligned (one of the 6 axis directions).
- **Sphere**: all orientations should give approximately equal overhang — verify near-constant objective landscape.
- **Dental crown** (if available): verify that the occlusal surface does not face downward when no_support_faces is set correctly.
- **Round-trip**: for a random mesh, run optimizer, apply the output quaternion to the mesh, verify that the overhang area of the rotated mesh matches the reported value.

### 6.3 Benchmarks

- **Speed**: time optimize_orientation for 10K, 50K, 100K, 500K face meshes. Target: <1s for 100K faces.
- **Quality**: compare objective value against brute-force 10K-point grid search. The refined result should be equal or better.
- **Pareto quality**: compare pairwise Pareto intersection points against NSGA-II on the same problem. Intersection points should all be non-dominated in the NSGA-II population.


## 7. Implementation Order

### Phase 1: Core solver (single-objective scalarized)
1. `sampling.py` — fibonacci_sphere
2. `objectives.py` — overhang (hard), overhang_smooth, build_height, violation
3. `gradients.py` — overhang_smooth_gradient
4. `hopf.py` — sphere_to_quaternion, quaternion_to_rotation_matrix
5. `optimizer.py` — coarse_then_refine, optimize_single → wraps into optimize_orientation
6. `types.py` — OrientationResult dataclass
7. Unit tests for all of the above
8. Integration test on a cube and a tetrahedron

### Phase 2: Multi-objective
9. `objectives.py` — add surface_quality
10. `pareto.py` — non_dominated_sort (or import from pymoo), pareto_front_2obj
11. `pareto.py` — pairwise_intersection
12. Public API: pareto_front(), pairwise_pareto()
13. Tests for Pareto computation

### Phase 3: Mesh I/O and convenience
14. `preprocess.py` — load_mesh, extract_arrays, region_labels_from_colors
15. Public API: optimize_from_file()
16. Integration test on real dental mesh

### Phase 4: Polish
17. CLI entry point: `python -m orient_opt mesh.stl --lambda 0.7`
18. Logging and progress output
19. Benchmarks
20. README with usage examples


## 8. Numerical Pitfalls to Watch

- **South pole singularity**: the section formula divides by sqrt(2*(1+nz)), which blows up at nz = -1. Add a guard: if nz < -0.99, rotate the problem by 180° about the x-axis, solve, then compose the rotations.

- **Sigmoid overflow**: for large beta * |u|, exp(-beta*u) can overflow. Use `np.clip` on the argument or use `scipy.special.expit` which handles this.

- **Normalization by zero range**: if all feasible candidates have the same overhang value (e.g., O_max == O_min), the normalization (O - O_min) / (O_max - O_min) divides by zero. Set O_hat = 0 in this case (the objective is constant and doesn't contribute to the trade-off).

- **Empty feasible set**: if all 500 candidates are infeasible (V(g) > 0 for all g), the solver should report this clearly rather than crash. This likely means the no_support_faces constraint is too aggressive.

- **Gradient step overshooting**: if eta is too large, the retraction (normalize after step) still lands on S² but may oscillate. Use backtracking line search or a conservative fixed eta = 0.01.


## 9. Reference Implementations for Validation

- **trimesh.poses.compute_stable_poses**: computes stable resting poses from the convex hull. Not an optimizer, but the stable poses can be used as additional candidate starting points for gradient descent.

- **Brute-force grid search**: generate 10,000 Fibonacci points, evaluate all objectives, take the best. This is the ground truth for validation — the refined solver should match or beat it.


## 10. Future Extensions (Not in Scope Now)

- Joint orientation + packing solver (couples S² optimization with SE(2) bin packing).
- GPU acceleration via CuPy or JAX (drop-in replacement for NumPy arrays).
- FastAPI wrapper for SaaS deployment.
- Interactive S² visualization of the objective landscape and Pareto front (PyVista or Three.js).
