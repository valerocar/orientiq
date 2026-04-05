# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python library that finds the optimal 3D printing orientation for triangle meshes. It minimizes support material, build height, and surface quality degradation while respecting hard constraints on functional surfaces.

**Key insight:** Rotation about the gravity axis doesn't affect print objectives, so the search space reduces from SO(3) (3 DOF) to S² (2 DOF) via the Hopf fibration. The search variable is a unit vector **g** ∈ S² representing the gravity direction in the part's local frame.

## Build and Test Commands

```bash
pip install -e ".[dev]"          # install with all dev dependencies
python -m pytest tests/ -v       # run all tests
python -m pytest tests/test_hopf.py::test_south_pole -v  # run a single test
python -m orient_opt mesh.stl    # CLI usage
python -m orient_opt mesh.stl --lambda 0.5 --n-samples 1000  # CLI with options
```

## Architecture

### Data Flow
```
STL/OBJ → trimesh loading → normals/areas/vertices extraction
    → Fibonacci sampling (~500 points on S²)
    → Batch objective evaluation (dots = candidates @ normals.T)
    → Filter infeasible (no_support_faces constraint)
    → Coarse-then-refine: Riemannian gradient descent on top K candidates
    → Hopf reconstruction → quaternion output
```

### Module Layout (`orient_opt/`)
| Module | Responsibility |
|---|---|
| `sampling.py` | Fibonacci sphere sampling (equidistributed points on S²) |
| `objectives.py` | Overhang area, build height, surface quality, constraint violations |
| `gradients.py` | Analytical S² gradients for smoothed overhang (no autodiff) |
| `optimizer.py` | Riemannian gradient descent + coarse-then-refine pipeline |
| `pareto.py` | Multi-objective: non-dominated sorting, Pareto fronts, pairwise intersection |
| `hopf.py` | Quaternion ↔ sphere conversions (Hopf section) |
| `preprocess.py` | Mesh loading via trimesh, region labeling by color |
| `types.py` | `OrientationResult` dataclass |

### Web App (`webapp/`)
| File | Responsibility |
|---|---|
| `server.py` | FastAPI backend: mesh serving, optimization, Pareto front, rotation endpoints |
| `static/index.html` | Three.js frontend: 3D viewer, objective sphere overlay, critical point analysis |

**Modes:**
- **Single mode:** Original mesh (left) + optimized mesh (right). Objective sphere overlay (upper-left of right panel) shows level sets and critical points of the selected objective on S².
- **Pair mode:** Dual-colored original mesh (left) + interactive Pareto sphere (right). Click Pareto points to inspect orientations.

**Objective sphere overlay (single mode):**
- Icosahedral mesh (`IcosahedronGeometry` + `mergeVertices`) for uniform sampling without pole artifacts
- Client-side objective evaluation at each sphere vertex (overhang, build height, support volume, surface quality)
- 15 contour lines (level sets) via marching triangles
- Critical point detection: vertex-level (discrete Morse theory on 1-ring) + face-level (dual mesh) for extrema between vertices
- Critical point types: minimum (green), maximum (blue), saddle (yellow) — hover for tooltip
- Double-click snaps to nearest critical point and rotates the optimized model to that orientation
- Red marker shows current selected orientation

**API endpoints:**
| Endpoint | Method | Description |
|---|---|---|
| `/api/models` | GET | List available 3D models |
| `/api/mesh/{name}` | GET | Get mesh data (binary format) |
| `/api/optimize` | POST | Run single-objective optimization |
| `/api/pareto` | POST | Compute 2D Pareto front |
| `/api/rotate` | POST | Rotate mesh to arbitrary gravity direction |

**Running the webapp:**
```bash
pip install fastapi uvicorn fast_simplification
uvicorn webapp.server:app --reload
```

### Public API
- `optimize_orientation()` — single-objective optimizer → `OrientationResult`
- `pareto_front()` — sweep λ ∈ [0,1] for multi-objective Pareto front
- `pairwise_pareto()` — all C(k,2) pairwise Pareto curves with intersection
- `optimize_from_file()` — convenience wrapper that loads mesh from file

## Dependencies
- `numpy >= 1.24` — core numerics
- `trimesh >= 4.0` — mesh I/O
- `scipy >= 1.10` — `cKDTree` for Pareto intersection queries
- `pymoo >= 0.6` — `fast_non_dominated_sort` only
- `fastapi` + `uvicorn` — webapp server
- `fast_simplification` — mesh decimation for rendering

No PyTorch/JAX/autograd — all gradients are closed-form analytical.

## Numerical Guardrails
- **Hopf south pole:** Guard `nz < -0.99` (division by `sqrt(2*(1+nz))`)
- **Sigmoid overflow:** Use `scipy.special.expit` or clip argument
- **Normalization:** Fallback to 0 when `O_max == O_min`
- **Empty feasible set:** Report clear error if all candidates violate constraints
- **Gradient step:** Conservative η=0.01 with normalize-as-retraction on S²

## Specification

The full mathematical specification is in `ORIENT_OPT_SPEC.md` — objective function definitions, gradient derivations, algorithm parameters, and testing strategy.
