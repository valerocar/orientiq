#!/usr/bin/env python3
"""
Plane-intersection pipeline:
  1. Load STL → sample point cloud
  2. Iterative RANSAC plane segmentation
  3. For each plane: project inliers → 2D convex hull → clean flat face
  4. Show all faces coloured separately; save each as its own STL
"""

import os
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

STL_PATH    = "3dmodels/bracket_v2.stl"
OUT_DIR     = "3dmodels/bracket_faces"
N_POINTS    = 80_000
MAX_PLANES  = 25
MIN_POINTS  = 200
DIST_THRESH = 0.8
RANSAC_N    = 3
RANSAC_ITER = 2000

PALETTE = np.array([
    [0.9, 0.2, 0.2], [0.2, 0.75, 0.2], [0.2, 0.4,  0.9],
    [0.9, 0.75, 0.1], [0.7, 0.2, 0.8], [0.1, 0.8,  0.8],
    [0.9, 0.5,  0.1], [0.5, 0.9, 0.3], [0.4, 0.2,  0.8],
    [0.8, 0.8,  0.2], [0.2, 0.6, 0.5], [0.6, 0.3,  0.1],
    [0.1, 0.3,  0.6], [0.8, 0.1, 0.5], [0.3, 0.8,  0.6],
    [0.6, 0.6,  0.9], [0.9, 0.3, 0.6], [0.1, 0.5,  0.9],
    [0.5, 0.2,  0.5], [0.2, 0.9, 0.5], [0.8, 0.4,  0.0],
    [0.0, 0.5,  0.5], [0.5, 0.0, 0.5], [0.3, 0.3,  0.9],
    [0.9, 0.0,  0.3],
])


# ── geometry helpers ──────────────────────────────────────────────────────────

def local_2d_basis(normal):
    """Two orthonormal vectors spanning the plane with the given normal."""
    n = normal / np.linalg.norm(normal)
    ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(ref, n);  u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def make_face_mesh(pts, plane_model, colour):
    """
    Project 3D inlier points onto their plane, compute 2D convex hull,
    fan-triangulate, and return a coloured Open3D TriangleMesh.
    """
    normal = np.asarray(plane_model[:3], float)
    normal /= np.linalg.norm(normal)
    d = plane_model[3]
    plane_pt = -d * normal          # a point on the plane

    # Project points onto plane
    offsets   = np.dot(pts - plane_pt, normal)
    projected = pts - np.outer(offsets, normal)

    # Express in plane's 2D frame
    u, v   = local_2d_basis(normal)
    cent   = projected - plane_pt
    pts_2d = np.column_stack([cent @ u, cent @ v])

    # Convex hull in 2D
    try:
        hull = ConvexHull(pts_2d)
    except Exception:
        return None

    hv_2d = pts_2d[hull.vertices]          # hull vertices in 2D
    n_h   = len(hv_2d)
    hv_3d = plane_pt + np.outer(hv_2d[:, 0], u) + np.outer(hv_2d[:, 1], v)
    c_3d  = hv_3d.mean(axis=0)             # centroid for fan triangulation

    verts = np.vstack([hv_3d, c_3d])
    faces = np.array([[i, (i + 1) % n_h, n_h] for i in range(n_h)])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(colour.tolist())
    return mesh


# ── pipeline ──────────────────────────────────────────────────────────────────

def load_point_cloud(path, n):
    src = o3d.io.read_triangle_mesh(path)
    pcd = src.sample_points_poisson_disk(n)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
    return pcd


def iterative_ransac(pcd):
    remaining = pcd
    results   = []            # list of (plane_model, pts_array, colour)
    idx_map   = np.arange(len(np.asarray(pcd.points)))
    colours   = np.full((len(idx_map), 3), 0.55)   # default grey

    for i in range(MAX_PLANES):
        if len(np.asarray(remaining.points)) < MIN_POINTS:
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=DIST_THRESH,
            ransac_n=RANSAC_N,
            num_iterations=RANSAC_ITER)

        colour    = PALETTE[i % len(PALETTE)]
        orig_idx  = idx_map[inliers]
        colours[orig_idx] = colour

        inlier_pts = np.asarray(remaining.select_by_index(inliers).points)
        results.append((model, inlier_pts, colour))

        remaining = remaining.select_by_index(inliers, invert=True)
        idx_map   = np.delete(idx_map, inliers)

        n = model[:3];  d = model[3]
        print(f"  Plane {i+1:2d}: {len(inliers):5d} pts  "
              f"n=({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f})  d={d:+.1f}")

    coloured = o3d.geometry.PointCloud()
    coloured.points  = pcd.points
    coloured.normals = pcd.normals
    coloured.colors  = o3d.utility.Vector3dVector(colours)
    return coloured, results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Stage 1: raw point cloud ──────────────────────────────────────────────
    print(f"Loading {STL_PATH}…")
    pcd = load_point_cloud(STL_PATH, N_POINTS)
    print(f"  {len(np.asarray(pcd.points))} points sampled\n")
    print("Stage 1 – raw point cloud  (close window to continue)")
    o3d.visualization.draw_geometries([pcd], window_name="Stage 1 – Point cloud",
                                      width=1200, height=800)

    # ── Stage 2: RANSAC segmentation ──────────────────────────────────────────
    print("\nStage 2 – RANSAC plane segmentation")
    coloured, planes = iterative_ransac(pcd)
    print(f"\n  {len(planes)} planes found")
    print("Stage 2 – segmented cloud  (close window to continue)")
    o3d.visualization.draw_geometries([coloured],
                                      window_name="Stage 2 – Plane segmentation",
                                      width=1200, height=800)

    # ── Stage 3: flat face meshes from convex hulls ───────────────────────────
    print("\nStage 3 – building flat face meshes from convex hulls")
    face_meshes = []
    for i, (model, pts, colour) in enumerate(planes):
        mesh = make_face_mesh(pts, model, colour)
        if mesh is None:
            print(f"  Face {i+1}: skipped (degenerate)")
            continue
        face_meshes.append(mesh)
        path = os.path.join(OUT_DIR, f"face_{i+1:02d}.stl")
        o3d.io.write_triangle_mesh(path, mesh)
        print(f"  Face {i+1:2d}: {len(np.asarray(mesh.vertices))} verts  → {path}")

    print(f"\n  Showing {len(face_meshes)} faces  (close window to finish)")
    o3d.visualization.draw_geometries(face_meshes,
                                      window_name="Stage 3 – Flat faces (convex hull per plane)",
                                      width=1200, height=800)
    print("Done.")


if __name__ == "__main__":
    main()
