"""FastAPI backend for 3D print orientation optimizer."""

import io
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orient_opt.critical_points import find_critical_points
from orient_opt.gradients import build_height_gradient, overhang_smooth_gradient
from orient_opt.objectives import build_height, overhang, overhang_smooth, support_volume, surface_quality, stability
from orient_opt.optimizer import coarse_then_refine
from orient_opt.sampling import fibonacci_sphere
from orient_opt.hopf import sphere_to_quaternion, quaternion_to_rotation_matrix
from orient_opt.pareto import non_dominated_front_2obj

app = FastAPI()

MODELS_DIR = Path(__file__).resolve().parent.parent / "3dmodels"
MAX_FACES = 10_000
N_SAMPLES_SINGLE = 2_000
N_SAMPLES_PARETO = 100_000

# Cache
model_cache: dict = {}       # name -> original trimesh
render_cache: dict = {}      # name -> (verts, faces, normals) prepared
model_list: list = []


@app.on_event("startup")
def load_models():
    global model_list
    print("Loading models...")
    for stl_path in sorted(MODELS_DIR.glob("*.stl")):
        name = stl_path.stem
        mesh = trimesh.load(str(stl_path), force="mesh")
        model_cache[name] = mesh
        # Pre-decimate and cache render mesh
        render_cache[name] = _prepare_mesh(mesh, MAX_FACES)
        print(f"  {name}: {len(mesh.faces):,} faces -> {len(render_cache[name][1]):,} render")
    model_list = sorted(model_cache.keys(), key=lambda n: len(model_cache[n].faces))
    print(f"Loaded {len(model_list)} models")


def _prepare_mesh(mesh, max_faces):
    """Decimate, center, normalize a mesh. Returns verts, faces, normals."""
    if len(mesh.faces) > max_faces:
        import fast_simplification
        ratio = 1.0 - max_faces / len(mesh.faces)
        verts_s, faces_s = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32),
            target_reduction=ratio,
        )
        mesh = trimesh.Trimesh(vertices=verts_s, faces=faces_s, process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    verts = verts - center
    scale = np.abs(verts).max()
    if scale > 0:
        verts = verts / scale
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms
    return verts, faces, normals


def _overhang_analysis(verts, faces, normals, angle=45.0):
    """Compute overhang indices and support pillar centroids."""
    g = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    threshold = -np.cos(np.radians(angle))
    dots = normals @ g
    oh_mask = dots < threshold
    oh_indices = np.where(oh_mask)[0].astype(np.int32)
    centroids = verts[faces[oh_mask]].mean(axis=1).astype(np.float32)
    z_min = float(verts[:, 2].min())
    return oh_indices, centroids, z_min


def _pack_mesh_binary(verts, faces, normals, oh_indices, pillar_centroids, z_min,
                      n_faces_original=0, extra_floats=None):
    """Pack mesh data into a binary buffer.

    Layout:
      Header (7 uint32):  n_verts, n_faces, n_oh, n_pillars, n_faces_original, extra_count, reserved
      extra_floats:       extra_count float32 values (e.g. original_value, optimized_value)
      z_min:              1 float32
      vertices:           n_verts * 3 float32
      faces:              n_faces * 3 int32
      normals:            n_faces * 3 float32
      oh_indices:         n_oh int32
      pillar_centroids:   n_pillars * 3 float32
    """
    buf = io.BytesIO()
    n_verts = len(verts)
    n_faces = len(faces)
    n_oh = len(oh_indices)
    n_pillars = len(pillar_centroids)
    extra = extra_floats if extra_floats is not None else []
    extra_count = len(extra)

    # Header
    buf.write(struct.pack('<7I', n_verts, n_faces, n_oh, n_pillars,
                          n_faces_original, extra_count, 0))
    # Extra floats
    for v in extra:
        buf.write(struct.pack('<f', v))
    # z_min
    buf.write(struct.pack('<f', z_min))
    # Data arrays
    buf.write(verts.astype(np.float32).tobytes())
    buf.write(faces.astype(np.int32).tobytes())
    buf.write(normals.astype(np.float32).tobytes())
    buf.write(oh_indices.astype(np.int32).tobytes())
    if n_pillars > 0:
        buf.write(pillar_centroids.astype(np.float32).tobytes())

    return buf.getvalue()


# --- API ---

@app.get("/api/models")
def list_models():
    return [
        {"name": name, "n_faces": len(model_cache[name].faces)}
        for name in model_list
    ]


@app.get("/api/mesh/{model_name}")
def get_mesh(model_name: str):
    if model_name not in model_cache:
        raise HTTPException(404, f"Model '{model_name}' not found")

    verts, faces, normals = render_cache[model_name]
    oh_indices, pillar_centroids, z_min = _overhang_analysis(verts, faces, normals)

    data = _pack_mesh_binary(
        verts, faces, normals, oh_indices, pillar_centroids, z_min,
        n_faces_original=len(model_cache[model_name].faces),
    )
    return Response(content=data, media_type="application/octet-stream")


class OptimizeRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    objective: str


@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    if req.model_name not in model_cache:
        raise HTTPException(404, f"Model '{req.model_name}' not found")
    if req.objective not in OBJECTIVES:
        raise HTTPException(400, f"Unknown objective '{req.objective}'")

    mesh = model_cache[req.model_name]
    rv, rf, rn = render_cache[req.model_name]

    # Float64 copies for optimizer
    on = rn.astype(np.float64)
    ov = rv.astype(np.float64)
    of = rf
    # Compute face areas from geometry
    v0, v1, v2 = ov[of[:, 0]], ov[of[:, 1]], ov[of[:, 2]]
    opt_areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) / 2

    g_z = np.array([[0.0, 0.0, 1.0]])

    if req.objective == "overhang":
        result = coarse_then_refine(
            normals=on, areas=opt_areas, vertices=ov,
            lam=1.0, n_samples=N_SAMPLES_SINGLE, n_refine=5,
        )
        best_g = result.gravity_direction
        orig_val = float(overhang(on, opt_areas, g_z)[0])
        opt_val = float(overhang(on, opt_areas, best_g.reshape(1, 3))[0])

    elif req.objective == "support_volume":
        candidates = fibonacci_sphere(N_SAMPLES_SINGLE)
        sv = support_volume(on, opt_areas, ov, of, candidates)
        best_idx = np.argmin(sv)
        best_g = candidates[best_idx]
        orig_val = float(support_volume(on, opt_areas, ov, of, g_z)[0])
        opt_val = float(sv[best_idx])

    elif req.objective == "surface_quality":
        candidates = fibonacci_sphere(N_SAMPLES_SINGLE)
        critical_faces = np.arange(len(of))
        sq = surface_quality(on, opt_areas, candidates, critical_faces)
        best_idx = np.argmin(sq)
        best_g = candidates[best_idx]
        orig_val = float(surface_quality(on, opt_areas, g_z, critical_faces)[0])
        opt_val = float(sq[best_idx])

    elif req.objective == "stability":
        candidates = fibonacci_sphere(N_SAMPLES_SINGLE)
        stab = stability(ov, of, candidates)
        best_idx = np.argmin(stab)
        best_g = candidates[best_idx]
        orig_val = float(stability(ov, of, g_z)[0])
        opt_val = float(stab[best_idx])

    else:  # build_height
        result = coarse_then_refine(
            normals=on, areas=opt_areas, vertices=ov,
            lam=0.0, n_samples=N_SAMPLES_SINGLE, n_refine=5,
        )
        best_g = result.gravity_direction
        orig_val = float(build_height(ov, g_z)[0])
        opt_val = float(build_height(ov, best_g.reshape(1, 3))[0])

    # Rotate render mesh
    q = sphere_to_quaternion(best_g)
    R = quaternion_to_rotation_matrix(q)[:3, :3].astype(np.float32)

    rot_verts = (R @ rv.T).T
    rot_normals = (R @ rn.T).T

    # Re-center
    center = (rot_verts.max(axis=0) + rot_verts.min(axis=0)) / 2
    rot_verts = rot_verts - center
    scale = np.abs(rot_verts).max()
    if scale > 0:
        rot_verts = rot_verts / scale

    oh_indices, pillar_centroids, z_min = _overhang_analysis(
        rot_verts, rf, rot_normals
    )

    data = _pack_mesh_binary(
        rot_verts, rf, rot_normals, oh_indices, pillar_centroids, z_min,
        n_faces_original=len(mesh.faces),
        extra_floats=[orig_val, opt_val],
    )
    return Response(content=data, media_type="application/octet-stream")


OBJECTIVES = ("overhang", "support_volume", "build_height", "surface_quality", "stability")


def _eval_objective(name, on, opt_areas, ov, of, candidates):
    """Evaluate a named objective on all candidates. Returns (N,) array."""
    if name == "overhang":
        return overhang(on, opt_areas, candidates)
    elif name == "support_volume":
        return support_volume(on, opt_areas, ov, of, candidates)
    elif name == "build_height":
        return build_height(ov, candidates)
    elif name == "surface_quality":
        critical = np.arange(len(of))
        return surface_quality(on, opt_areas, candidates, critical)
    elif name == "stability":
        return stability(ov, of, candidates)
    raise ValueError(f"Unknown objective: {name}")


def _face_penalty_mask(name, normals, areas, verts, faces, angle=45.0):
    """Return boolean mask of penalized faces for a given objective at gravity=[0,0,1]."""
    g = np.array([0.0, 0.0, 1.0])
    if name in ("overhang", "support_volume"):
        threshold = -np.cos(np.radians(angle))
        return (normals @ g) < threshold
    elif name == "build_height":
        proj = verts[faces].mean(axis=1) @ g  # face centroid heights
        h_range = proj.max() - proj.min()
        if h_range == 0:
            return np.zeros(len(faces), dtype=bool)
        normalized = (proj - proj.min()) / h_range
        return (normalized > 0.9) | (normalized < 0.1)
    elif name == "surface_quality":
        dots = normals @ g
        penalty = (1 - np.abs(dots)) ** 2
        return penalty > 0.5
    elif name == "stability":
        # Highlight base faces (bottom 2% by height)
        proj = verts[faces].mean(axis=1) @ g
        h_range = proj.max() - proj.min()
        if h_range == 0:
            return np.zeros(len(faces), dtype=bool)
        return proj < (proj.min() + 0.02 * h_range)
    return np.zeros(len(faces), dtype=bool)


def _get_mesh_data(model_name):
    """Get cached mesh data + computed areas as float64."""
    rv, rf, rn = render_cache[model_name]
    on = rn.astype(np.float64)
    ov = rv.astype(np.float64)
    of = rf
    v0, v1, v2 = ov[of[:, 0]], ov[of[:, 1]], ov[of[:, 2]]
    opt_areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) / 2
    return rv, rf, rn, on, ov, of, opt_areas


class ParetoRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    objective_a: str
    objective_b: str


@app.post("/api/pareto")
def pareto(req: ParetoRequest):
    if req.model_name not in model_cache:
        raise HTTPException(404, f"Model '{req.model_name}' not found")
    if req.objective_a not in OBJECTIVES or req.objective_b not in OBJECTIVES:
        raise HTTPException(400, "Unknown objective")
    if req.objective_a == req.objective_b:
        raise HTTPException(400, "Objectives must be different")

    rv, rf, rn, on, ov, of, opt_areas = _get_mesh_data(req.model_name)
    candidates = fibonacci_sphere(N_SAMPLES_PARETO).astype(np.float64)

    # Evaluate both objectives on all candidates
    vals_a = _eval_objective(req.objective_a, on, opt_areas, ov, of, candidates)
    vals_b = _eval_objective(req.objective_b, on, opt_areas, ov, of, candidates)
    obj_matrix = np.column_stack([vals_a, vals_b])

    # Non-dominated front (fast O(N log N) for 2 objectives)
    pareto_idx = non_dominated_front_2obj(obj_matrix)

    # Face penalty masks for original orientation (gravity = +Z)
    mask_a = _face_penalty_mask(req.objective_a, rn, opt_areas.astype(np.float32), rv, rf)
    mask_b = _face_penalty_mask(req.objective_b, rn, opt_areas.astype(np.float32), rv, rf)
    # Encode: 0=none, 1=A only, 2=B only, 3=both
    face_flags = mask_a.astype(np.uint8) + mask_b.astype(np.uint8) * 2

    # Pack binary
    buf = io.BytesIO()
    n_cand = len(candidates)
    n_pareto = len(pareto_idx)
    n_faces = len(rf)

    # Header: n_candidates, n_pareto, n_faces, n_faces_original, reserved x3
    buf.write(struct.pack('<7I', n_cand, n_pareto, n_faces,
                          len(model_cache[req.model_name].faces), 0, 0, 0))
    # Candidate gravity directions (N, 3) float32
    buf.write(candidates.astype(np.float32).tobytes())
    # Objective values (N, 2) float32
    buf.write(obj_matrix.astype(np.float32).tobytes())
    # Pareto indices
    buf.write(pareto_idx.tobytes())
    # Face flags (F,) uint8
    buf.write(face_flags.tobytes())

    return Response(content=buf.getvalue(), media_type="application/octet-stream")


class RotateRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    gravity: list[float]


@app.post("/api/rotate")
def rotate(req: RotateRequest):
    if req.model_name not in model_cache:
        raise HTTPException(404, f"Model '{req.model_name}' not found")

    rv, rf, rn, on, ov, of, opt_areas = _get_mesh_data(req.model_name)
    g = np.array(req.gravity, dtype=np.float64)
    norm = np.linalg.norm(g)
    if norm < 1e-10:
        raise HTTPException(400, "Gravity vector must be non-zero")
    g = g / norm

    q = sphere_to_quaternion(g)
    R = quaternion_to_rotation_matrix(q)[:3, :3].astype(np.float32)

    rot_verts = (R @ rv.T).T
    rot_normals = (R @ rn.T).T

    # Re-center
    center = (rot_verts.max(axis=0) + rot_verts.min(axis=0)) / 2
    rot_verts = rot_verts - center
    scale = np.abs(rot_verts).max()
    if scale > 0:
        rot_verts = rot_verts / scale

    oh_indices, pillar_centroids, z_min = _overhang_analysis(rot_verts, rf, rot_normals)

    data = _pack_mesh_binary(
        rot_verts, rf, rot_normals, oh_indices, pillar_centroids, z_min,
        n_faces_original=len(model_cache[req.model_name].faces),
    )
    return Response(content=data, media_type="application/octet-stream")


class CriticalPointsRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    objective: str
    lam: float = 0.7
    overhang_angle: float = 45.0
    beta: float = 50.0
    n_samples: int = 2000


@app.post("/api/critical_points")
def critical_points(req: CriticalPointsRequest):
    if req.model_name not in model_cache:
        raise HTTPException(404, f"Model '{req.model_name}' not found")
    if req.objective not in OBJECTIVES:
        raise HTTPException(400, f"Unknown objective '{req.objective}'")

    _, _, _, on, ov, of, opt_areas = _get_mesh_data(req.model_name)
    lam = float(np.clip(req.lam, 0.0, 1.0))

    # Build objective/gradient callables consistent with coarse_then_refine
    oh_vals = overhang(on, opt_areas, fibonacci_sphere(200), angle=req.overhang_angle)
    bh_vals = build_height(ov, fibonacci_sphere(200))
    oh_min, oh_max = float(oh_vals.min()), float(oh_vals.max())
    bh_min, bh_max = float(bh_vals.min()), float(bh_vals.max())
    oh_range = (oh_max - oh_min) if oh_max > oh_min else 1.0
    bh_range = (bh_max - bh_min) if bh_max > bh_min else 1.0

    def objective_fn(g: np.ndarray) -> float:
        g2 = g.reshape(1, 3)
        o = float(overhang_smooth(on, opt_areas, g2, angle=req.overhang_angle, beta=req.beta)[0])
        h = float(build_height(ov, g2)[0])
        return lam * (o - oh_min) / oh_range + (1 - lam) * (h - bh_min) / bh_range

    def gradient_fn(g: np.ndarray) -> np.ndarray:
        grad_o = overhang_smooth_gradient(g, on, opt_areas, angle=req.overhang_angle, beta=req.beta)
        grad_h = build_height_gradient(g, ov)
        return lam * grad_o / oh_range + (1 - lam) * grad_h / bh_range

    cps = find_critical_points(
        objective_fn, gradient_fn, n_samples=req.n_samples
    )

    return [
        {
            "g": cp.g.tolist(),
            "type": cp.cp_type,
            "f_value": cp.f_value,
            "grad_norm": cp.grad_norm,
        }
        for cp in cps
    ]


# Static files & root
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
