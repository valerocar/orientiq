# orient-opt

Find the optimal 3D-printing orientation for triangle meshes.

## Overview

`orient-opt` minimizes support material, build height, and surface quality degradation for FDM/SLA 3D printing. It exploits the **Hopf fibration** to reduce the orientation search from SO(3) (3 DOF) to S\u00b2 (2 DOF), since rotation about the gravity axis doesn't affect print objectives. All gradients are closed-form analytical -- no autograd required.

## Why This Exists

Choosing how to orient a part on the build plate is one of the most consequential decisions in 3D printing. A poor orientation can double support material usage, add hours of print time, and ruin surface finish on functional faces -- yet most users pick orientations by eye or trial and error.

Automating this choice is hard because the search space is the full rotation group SO(3), objectives conflict with each other (less support often means taller builds), and real parts have hard constraints like "this mating surface must face up." Existing slicers offer limited heuristics, and general-purpose optimizers ignore the geometry of the problem.

`orient-opt` solves this by exploiting a key symmetry: rotating a part around the gravity axis changes nothing about print quality, so the effective search space is the 2-sphere S² rather than all 3D rotations. This Hopf-fibration reduction, combined with closed-form analytical gradients on S² and Fibonacci-lattice sampling, makes the search both fast and mathematically rigorous -- no autograd frameworks needed. The library also supports functional surface constraints and multi-objective Pareto analysis, so users can explore trade-offs rather than committing to a single weighting upfront.

## Features

- **Single-objective optimization** with configurable overhang/height trade-off
- **Multi-objective Pareto fronts** between any pair of objectives
- **Functional surface constraints** -- mark faces as no-support or critical quality via mesh colors
- **Coarse-then-refine** search: Fibonacci sampling + Riemannian gradient descent on S\u00b2
- **CLI, Python API, and interactive web viewer**

## Quick Run

The easiest way to get started — just run:

```bash
./run.sh
```

This will create the conda environment (if needed), install dependencies, and start the web app.

## Installation

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate optorient3d
pip install -e ".[dev]"
```

### Option 2: pip only

```bash
pip install -e "."            # core (numpy only)
pip install -e ".[all]"       # + mesh I/O (trimesh) and Pareto (pymoo, scipy)
pip install -e ".[webapp]"    # + FastAPI, uvicorn, fast_simplification
pip install -e ".[dev]"       # all of the above + pytest
```

Requires Python >= 3.10.

## Quick Start

### CLI

```bash
python -m orient_opt model.stl
python -m orient_opt model.stl --lambda 0.5 --n-samples 1000
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--lambda` | 0.7 | Trade-off: 0 = height only, 1 = overhang only |
| `--overhang-angle` | 45.0 | Overhang threshold in degrees |
| `--n-samples` | 500 | Fibonacci sample points on S\u00b2 |
| `--n-refine` | 5 | Candidates refined via gradient descent |
| `--beta` | 50.0 | Sigmoid sharpness |

### Python API

```python
from orient_opt import optimize_from_file

result = optimize_from_file("model.stl", lam=0.7)

print(result.gravity_direction)   # optimal gravity vector on S\u00b2
print(result.quaternion)          # (w, x, y, z)
print(result.overhang_area)
print(result.build_height)
```

With functional surface constraints:

```python
result = optimize_from_file(
    "model.stl",
    no_support_face_colors=[(255, 0, 0)],    # red faces must not overhang
    critical_face_colors=[(0, 0, 255)],       # blue faces need good surface quality
)
```

Low-level usage with raw arrays:

```python
from orient_opt import optimize_orientation

result = optimize_orientation(
    normals,           # (F, 3) face normals
    areas,             # (F,) face areas
    vertices,          # (V, 3) vertex positions
    lam=0.7,
    n_samples=500,
)
```

### Multi-objective

```python
from orient_opt import pareto_front, pairwise_pareto

# Pareto front between two objectives
front = pareto_front(normals, areas, vertices, n_points=50)

# All pairwise Pareto curves with intersection points
curves = pairwise_pareto(normals, areas, vertices)
```

## Web App

An interactive 3D viewer built with FastAPI and Three.js.

```bash
./run.sh
```

Or manually:

```bash
conda activate optorient3d
uvicorn webapp.server:app --reload
```

Place `.stl` files in the `3dmodels/` directory. The viewer supports:
- Overhang visualization with color-coded faces
- Support pillar preview
- Objective-based optimization (overhang, support volume, build height, surface quality, stability)
- Interactive Pareto front exploration

## Project Structure

```
orient_opt/
  __init__.py        # Public API: optimize_orientation, optimize_from_file
  __main__.py        # CLI entry point
  sampling.py        # Fibonacci sphere sampling
  objectives.py      # Overhang area, build height, surface quality
  gradients.py       # Analytical S\u00b2 gradients (no autodiff)
  optimizer.py       # Riemannian gradient descent + coarse-then-refine
  pareto.py          # Non-dominated sorting, Pareto fronts
  hopf.py            # Quaternion <-> sphere (Hopf section)
  preprocess.py      # Mesh loading, color-based region labeling
  types.py           # OrientationResult dataclass
webapp/
  server.py          # FastAPI backend
  static/index.html  # Three.js frontend
scripts/
  benchmark_report.py  # HTML report with Plotly visualizations
tests/               # pytest suite
```

## How It Works

```
STL/OBJ -> trimesh -> normals, areas, vertices
  -> Fibonacci sampling (~500 points on S\u00b2)
  -> Batch objective evaluation
  -> Filter infeasible candidates (no-support constraints)
  -> Coarse top-K selection
  -> Riemannian gradient descent refinement
  -> Hopf reconstruction -> quaternion output
```

See [ORIENT_OPT_SPEC.md](ORIENT_OPT_SPEC.md) for the full mathematical specification.

## Testing

```bash
python -m pytest tests/ -v
```

## License

TBD
