import trimesh
import numpy as np
import igl
import time

def run_benchmark():
    print("--- xolotl (Mac Studio M3 Ultra) Geometry Benchmark ---")
    
    # 1. Generate mesh
    print("Generating high-res mesh (100k+ faces)...")
    mesh = trimesh.primitives.Sphere(subdivisions=5)
    v, f = mesh.vertices, mesh.faces
    
    # 2. Gaussian Curvature (Single-thread clock speed test)
    print(f"Testing Gaussian Curvature on {len(f)} faces...")
    start = time.time()
    k = igl.gaussian_curvature(v, f)
    end = time.time()
    print(f">> Curvature calculated in: {end - start:.4f}s")

    # 3. Ray-Mesh Intersections (Memory Bandwidth & Branching test)
    print("Testing 10,000 Ray-Mesh intersections...")
    rays_origin = np.random.uniform(-1, 1, (10000, 3))
    rays_direction = np.random.uniform(-1, 1, (10000, 3))
    
    start = time.time()
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=rays_origin, ray_directions=rays_direction)
    end = time.time()
    print(f">> Ray-intersections calculated in: {end - start:.4f}s")

    # 4. Matrix Math (The AMX vs AVX-512 Test)
    print("Stressing 24 cores with 2000x2000 Matrix Eigendecomposition...")
    matrix = np.random.rand(2000, 2000)
    start = time.time()
    np.linalg.eig(matrix)
    end = time.time()
    print(f">> Matrix math completed in: {end - start:.4f}s")

    print("\nBenchmark Complete. Let's compare the results.")

if __name__ == "__main__":
    run_benchmark()
