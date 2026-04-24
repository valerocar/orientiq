"""FastAPI backend — image to 3D via TripoSR (local inference, MPS/CPU)."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import asyncio
import sys
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "TripoSR"))

app = FastAPI()

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"

jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

_model = None
_rembg_session = None


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        import rembg
        _rembg_session = rembg.new_session()
    return _rembg_session


def _get_model():
    global _model
    if _model is None:
        from tsr.system import TSR
        print("Loading TripoSR model…")
        _model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        _model.renderer.set_chunk_size(131072)
        _model.to(DEVICE)
        print("Model loaded.")
    return _model


def _boost_saturation(mesh, factor: float):
    if mesh.visual.vertex_colors is None or factor == 1.0:
        return
    vc = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
    lum = 0.2126 * vc[:, 0:1] + 0.7152 * vc[:, 1:2] + 0.0722 * vc[:, 2:3]
    vc = np.clip(lum + factor * (vc - lum), 0.0, 1.0)
    mesh.visual.vertex_colors[:, :3] = (vc * 255).astype(np.uint8)


def _run_inference(job_id: str, image_bytes: bytes,
                   foreground_ratio: float, resolution: int,
                   threshold: float, saturation: float):
    jobs[job_id]["status"] = "running"
    timings: dict[str, float] = {}
    tmp_path = None
    try:
        from tsr.utils import remove_background, resize_foreground
        model = _get_model()
        rembg_session = _get_rembg_session()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name

        img = Image.open(tmp_path).convert("RGBA")

        jobs[job_id]["stage"] = "Removing background"
        print(f"[{job_id}] removing background…")
        t0 = time.perf_counter()
        img = remove_background(img, rembg_session)
        timings["rembg"] = round(time.perf_counter() - t0, 3)

        jobs[job_id]["stage"] = "Preprocessing"
        print(f"[{job_id}] resizing foreground (ratio={foreground_ratio})…")
        t0 = time.perf_counter()
        img = resize_foreground(img, foreground_ratio)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = img_np[:, :, :3] * img_np[:, :, 3:4] + (1 - img_np[:, :, 3:4]) * 0.5
        img_rgb = Image.fromarray((img_np * 255).astype(np.uint8))
        timings["preprocess"] = round(time.perf_counter() - t0, 3)

        jobs[job_id]["stage"] = "Running TripoSR"
        print(f"[{job_id}] running TripoSR on {DEVICE}…")
        t0 = time.perf_counter()
        with torch.no_grad():
            scene_codes = model([img_rgb], device=DEVICE)
        timings["triplane"] = round(time.perf_counter() - t0, 3)

        jobs[job_id]["stage"] = "Extracting mesh"
        print(f"[{job_id}] extracting mesh (res={resolution}, threshold={threshold})…")
        t0 = time.perf_counter()
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True,
                                    resolution=resolution, threshold=threshold)
        timings["marching"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        dest = RESULTS_DIR / f"{job_id}.glb"
        mesh = meshes[0]
        _boost_saturation(mesh, saturation)
        mesh.export(str(dest))
        timings["export"] = round(time.perf_counter() - t0, 3)

        print(f"[{job_id}] saved {dest} | timings: {timings}")
        jobs[job_id]["status"] = "done"
        jobs[job_id]["stage"] = "Done"
        jobs[job_id]["result"] = f"/api/result/{job_id}"
        jobs[job_id]["timings"] = timings

    except Exception as e:
        print(f"[{job_id}] ERROR:\n{traceback.format_exc()}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["stage"] = ""
        jobs[job_id]["error"] = f"{type(e).__name__}: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    foreground_ratio: float = Form(0.85),
    resolution: int = Form(256),
    threshold: float = Form(25.0),
    saturation: float = Form(2.5),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    image_bytes = await image.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 20 MB)")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "stage": "Queued", "result": None, "error": None, "timings": {}}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_inference,
                         job_id, image_bytes, foreground_ratio,
                         resolution, threshold, saturation)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/result/{job_id}")
def result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    if jobs[job_id]["status"] != "done":
        raise HTTPException(400, "Job not complete")
    glb_path = RESULTS_DIR / f"{job_id}.glb"
    if not glb_path.exists():
        raise HTTPException(404, "Result file missing")
    return FileResponse(str(glb_path), media_type="model/gltf-binary", filename="model.glb")


@app.get("/api/device")
def device_info():
    return {"device": DEVICE}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
