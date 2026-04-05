"""Generate an HTML report: original vs support-volume-optimized vs overhang-optimized."""

import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orient_opt.objectives import overhang as compute_overhang
from orient_opt.objectives import support_volume as compute_support_volume
from orient_opt.optimizer import coarse_then_refine
from orient_opt.sampling import fibonacci_sphere
from orient_opt.hopf import sphere_to_quaternion, quaternion_to_rotation_matrix

MODELS_DIR = Path(__file__).resolve().parent.parent / "3dmodels"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "results" / "report.html"
MAX_FACES = 10_000
N_SAMPLES = 500
N_REFINE = 0  # coarse grid only, no gradient refinement


def _face_colors(verts, faces, normals, gravity, angle=45.0):
    """Return per-face color array: red for overhang faces, steelblue otherwise."""
    threshold = -np.cos(np.radians(angle))
    dots = normals @ gravity  # (F,)
    colors = np.full((len(faces), 3), [70, 130, 180], dtype=np.uint8)  # steelblue
    colors[dots < threshold] = [220, 50, 50]  # red for overhang
    return colors


def _support_pillars(verts, faces, normals, angle=45.0):
    """Compute support pillar geometry for overhanging faces.

    Returns (centroids_top, z_min) for overhanging faces — each pillar goes
    from (cx, cy, z_min) up to (cx, cy, cz).
    """
    threshold = -np.cos(np.radians(angle))
    dots = normals @ np.array([0.0, 0.0, 1.0])
    overhang_mask = dots < threshold

    if not overhang_mask.any():
        return None

    centroids = verts[faces].mean(axis=1)  # (F, 3)
    oh_centroids = centroids[overhang_mask]
    z_min = verts[:, 2].min()
    return oh_centroids, z_min


def make_mesh_figure(verts, faces, title, face_colors=None, show_supports=False, normals_for_supports=None):
    """Create a Plotly 3D mesh figure with optional per-face coloring and support pillars."""
    # Center and normalize
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    verts_c = verts - center
    scale = np.abs(verts_c).max()
    if scale > 0:
        verts_c = verts_c / scale

    if face_colors is not None:
        facecolor = [f"rgb({r},{g},{b})" for r, g, b in face_colors]
        mesh_trace = go.Mesh3d(
            x=verts_c[:, 0], y=verts_c[:, 1], z=verts_c[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            facecolor=facecolor,
            opacity=0.5,
            flatshading=True,
            lighting=dict(ambient=0.4, diffuse=0.6, specular=0.2, roughness=0.5),
            lightposition=dict(x=100, y=200, z=300),
        )
    else:
        mesh_trace = go.Mesh3d(
            x=verts_c[:, 0], y=verts_c[:, 1], z=verts_c[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="steelblue",
            opacity=0.5,
            flatshading=True,
            lighting=dict(ambient=0.4, diffuse=0.6, specular=0.2, roughness=0.5),
            lightposition=dict(x=100, y=200, z=300),
        )

    fig = go.Figure(data=[mesh_trace])

    # Add support pillars if requested
    if show_supports and normals_for_supports is not None:
        pillar_data = _support_pillars(verts_c, faces, normals_for_supports)
        if pillar_data is not None:
            oh_centroids, z_min = pillar_data
            # Build line segments: each pillar is 3 points (base, top, None for break)
            xs, ys, zs = [], [], []
            for cx, cy, cz in oh_centroids:
                xs.extend([cx, cx, None])
                ys.extend([cy, cy, None])
                zs.extend([z_min, cz, None])
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color="orange", width=3),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=title, font=dict(size=12)),
        height=350,
    )
    return fig


def _compute_normals_for_render(verts, faces):
    """Compute face normals from vertices and faces."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return normals / norms


def _process_one(stl_path):
    """Process a single model: original, support-volume-optimized, overhang-optimized."""
    mesh = trimesh.load(str(stl_path), force="mesh")
    n_faces_orig = len(mesh.faces)

    # Simplify for optimization if needed
    if n_faces_orig > MAX_FACES:
        mesh_opt = mesh.simplify_quadric_decimation(MAX_FACES)
    else:
        mesh_opt = mesh

    normals = np.asarray(mesh_opt.face_normals, dtype=np.float64)
    areas = np.asarray(mesh_opt.area_faces, dtype=np.float64)
    vertices = np.asarray(mesh_opt.vertices, dtype=np.float64)
    opt_faces = np.asarray(mesh_opt.faces)

    g_z = np.array([0.0, 0.0, 1.0])

    # Fibonacci candidates
    candidates = fibonacci_sphere(N_SAMPLES)

    # Original metrics
    orig_overhang = compute_overhang(normals, areas, g_z.reshape(1, 3))[0]
    orig_sv = compute_support_volume(normals, areas, vertices, opt_faces, g_z.reshape(1, 3))[0]

    # Column 2: Optimize for support volume (pick best from coarse grid)
    sv_values = compute_support_volume(normals, areas, vertices, opt_faces, candidates)
    sv_best_idx = np.argmin(sv_values)
    sv_best_g = candidates[sv_best_idx]
    sv_best_val = sv_values[sv_best_idx]
    sv_q = sphere_to_quaternion(sv_best_g)
    sv_R = quaternion_to_rotation_matrix(sv_q)[:3, :3]
    sv_overhang = compute_overhang(normals, areas, sv_best_g.reshape(1, 3))[0]

    # Column 3: Optimize for overhang area (pick best from coarse grid)
    overhang_result = coarse_then_refine(
        normals=normals, areas=areas, vertices=vertices,
        lam=1.0, n_samples=N_SAMPLES, n_refine=N_REFINE,
    )
    o_overhang = compute_overhang(
        normals, areas, overhang_result.gravity_direction.reshape(1, 3)
    )[0]
    o_sv = compute_support_volume(
        normals, areas, vertices, opt_faces,
        overhang_result.gravity_direction.reshape(1, 3),
    )[0]

    # --- Render meshes ---
    if n_faces_orig > 50_000:
        mesh_render = mesh.simplify_quadric_decimation(50_000)
    else:
        mesh_render = mesh

    render_verts = np.asarray(mesh_render.vertices, dtype=np.float64)
    render_faces = np.asarray(mesh_render.faces)
    render_normals = _compute_normals_for_render(render_verts, render_faces)

    # Column 1: Original with overhang faces in red + support pillars
    orig_colors = _face_colors(render_verts, render_faces, render_normals, g_z)
    orig_fig = make_mesh_figure(
        render_verts, render_faces,
        f"Original<br><sub>Overhang: {orig_overhang:.1f} | Sup.Vol: {orig_sv:.1f}</sub>",
        face_colors=orig_colors,
        show_supports=True,
        normals_for_supports=render_normals,
    )

    # Column 2: Support-volume-optimized with overhang faces in red + support pillars
    sv_verts = (sv_R @ render_verts.T).T
    sv_normals = (sv_R @ render_normals.T).T
    sv_colors = _face_colors(sv_verts, render_faces, sv_normals, g_z)
    sv_fig = make_mesh_figure(
        sv_verts, render_faces,
        f"Min Support Vol.<br><sub>Overhang: {sv_overhang:.1f} | Sup.Vol: {sv_best_val:.1f}</sub>",
        face_colors=sv_colors,
        show_supports=True,
        normals_for_supports=sv_normals,
    )

    # Column 3: Overhang-optimized with overhang faces in red (no pillars)
    R_o = overhang_result.rotation_matrix[:3, :3]
    o_verts = (R_o @ render_verts.T).T
    o_normals = (R_o @ render_normals.T).T
    o_colors = _face_colors(o_verts, render_faces, o_normals, g_z)
    o_fig = make_mesh_figure(
        o_verts, render_faces,
        f"Min Overhang<br><sub>Overhang: {o_overhang:.1f} | Sup.Vol: {o_sv:.1f}</sub>",
        face_colors=o_colors,
    )

    return {
        "orig_json": orig_fig.to_json(),
        "sv_json": sv_fig.to_json(),
        "overhang_json": o_fig.to_json(),
        "n_faces": n_faces_orig,
        "orig_overhang": orig_overhang,
        "orig_sv": orig_sv,
        "opt_sv": sv_best_val,
        "opt_overhang": o_overhang,
    }


def build_html(model_results):
    """Build HTML report with 3 columns: original, support-vol-opt, overhang-opt."""
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        '<meta charset="utf-8">',
        "<title>Orientation Optimization Report</title>",
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #f8f9fa; }",
        "h1 { color: #333; }",
        ".model-row { display: flex; gap: 12px; margin-bottom: 30px; background: white; border-radius: 8px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); align-items: flex-start; }",
        ".model-col { flex: 1; }",
        ".col-header { text-align: center; font-size: 11px; color: #666; margin-bottom: 5px; }",
        ".model-name { font-size: 15px; font-weight: 600; color: #333; text-align: center; margin-bottom: 8px; }",
        ".metric { text-align: center; font-size: 12px; font-weight: 600; margin-top: 4px; }",
        ".metric.good { color: #27ae60; }",
        ".metric.same { color: #999; }",
        ".skipped { color: #999; font-style: italic; padding: 40px; text-align: center; width: 100%; }",
        ".legend { font-size: 12px; color: #666; margin-bottom: 20px; }",
        ".legend span.red { color: #dc3232; font-weight: 600; }",
        "</style>",
        "</head><body>",
        "<h1>Orientation Optimization Report</h1>",
        '<p class="legend"><span class="red">Red faces</span> = overhangs needing support. Column 1: Original | Column 2: Min support volume (area x height) | Column 3: Min overhang area</p>',
    ]

    plot_calls = []

    for i, (name, data) in enumerate(model_results):
        html_parts.append('<div class="model-row">')

        if data is None:
            html_parts.append(f'<div class="skipped">{name} — skipped (error)</div>')
            html_parts.append("</div>")
            continue

        (orig_json, sv_json, overhang_json, elapsed, n_faces,
         orig_oh, orig_sv, opt_sv, opt_oh) = data

        sv_pct = (1 - opt_sv / orig_sv) * 100 if orig_sv > 0 else 0
        oh_pct = (1 - opt_oh / orig_oh) * 100 if orig_oh > 0 else 0

        # Column 1: Original
        orig_id = f"orig_{i}"
        html_parts.append('<div class="model-col">')
        html_parts.append(f'<div class="model-name">{name}</div>')
        html_parts.append(f'<div class="col-header">{n_faces:,} faces | {elapsed:.1f}s</div>')
        html_parts.append(f'<div id="{orig_id}"></div>')
        plot_calls.append((orig_id, orig_json))
        html_parts.append("</div>")

        # Column 2: Support-volume-optimized
        sv_id = f"sv_{i}"
        sv_css = "good" if sv_pct > 1 else "same"
        html_parts.append('<div class="model-col">')
        html_parts.append(f'<div class="metric {sv_css}">{sv_pct:+.1f}% support vol.</div>')
        html_parts.append(f'<div id="{sv_id}"></div>')
        plot_calls.append((sv_id, sv_json))
        html_parts.append("</div>")

        # Column 3: Overhang-optimized
        o_id = f"overhang_{i}"
        o_css = "good" if oh_pct > 1 else "same"
        html_parts.append('<div class="model-col">')
        html_parts.append(f'<div class="metric {o_css}">{oh_pct:+.1f}% overhang</div>')
        html_parts.append(f'<div id="{o_id}"></div>')
        plot_calls.append((o_id, overhang_json))
        html_parts.append("</div>")

        html_parts.append("</div>")

    # Lazy rendering via IntersectionObserver
    html_parts.append("<script>")
    html_parts.append("var plotQueue = [")
    for div_id, json_str in plot_calls:
        html_parts.append(f'  ["{div_id}", {json_str}],')
    html_parts.append("];")
    html_parts.append("""
var observer = new IntersectionObserver(function(entries) {
  entries.forEach(function(entry) {
    if (entry.isIntersecting) {
      var el = entry.target;
      var id = el.id;
      for (var i = 0; i < plotQueue.length; i++) {
        if (plotQueue[i][0] === id) {
          var spec = plotQueue[i][1];
          Plotly.newPlot(id, spec.data, spec.layout, {responsive: true});
          observer.unobserve(el);
          break;
        }
      }
    }
  });
}, {rootMargin: '200px'});
plotQueue.forEach(function(item) {
  var el = document.getElementById(item[0]);
  if (el) observer.observe(el);
});
""")
    html_parts.append("</script>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def main():
    models = ['ring', 'spiral', 'house', 'keyring', 'rocket', 'trebol',
              'barry', 'tool', 'tripod', 'tube']

    model_results = []
    for name in tqdm(models, desc="Processing", unit="model"):
        stl_path = MODELS_DIR / f"{name}.stl"
        t0 = time.time()
        try:
            data = _process_one(stl_path)
            elapsed = time.time() - t0
            model_results.append((name, (
                data["orig_json"], data["sv_json"], data["overhang_json"],
                elapsed, data["n_faces"],
                data["orig_overhang"], data["orig_sv"],
                data["opt_sv"], data["opt_overhang"],
            )))
            sv_pct = (1 - data["opt_sv"] / data["orig_sv"]) * 100 if data["orig_sv"] > 0 else 0
            o_pct = (1 - data["opt_overhang"] / data["orig_overhang"]) * 100 if data["orig_overhang"] > 0 else 0
            tqdm.write(f"  {name}: sup.vol {data['orig_sv']:.0f}->{data['opt_sv']:.0f} ({sv_pct:+.0f}%) | overhang {data['orig_overhang']:.0f}->{data['opt_overhang']:.0f} ({o_pct:+.0f}%) [{elapsed:.1f}s]")
        except Exception as e:
            model_results.append((name, None))
            tqdm.write(f"  {name}: ERROR ({e})")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(model_results)
    OUTPUT_FILE.write_text(html)
    print(f"\nReport saved to {OUTPUT_FILE}")
    print(f"Open: file://{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
