"""CLI entry point: python -m orient_opt mesh.stl [options]"""

import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        prog="orient-opt",
        description="Find optimal 3D printing orientation for a triangle mesh.",
    )
    parser.add_argument("mesh", help="Path to mesh file (STL, OBJ, etc.)")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.7,
                        help="Trade-off: 0=height only, 1=overhang only (default: 0.7)")
    parser.add_argument("--overhang-angle", type=float, default=45.0,
                        help="Overhang threshold angle in degrees (default: 45)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of sample points on S² (default: 500)")
    parser.add_argument("--n-refine", type=int, default=5,
                        help="Number of candidates for gradient refinement (default: 5)")
    parser.add_argument("--beta", type=float, default=50.0,
                        help="Sigmoid sharpness (default: 50)")

    args = parser.parse_args()

    from . import optimize_from_file

    result = optimize_from_file(
        args.mesh,
        lam=args.lam,
        overhang_angle=args.overhang_angle,
        n_samples=args.n_samples,
        n_refine=args.n_refine,
        beta=args.beta,
    )

    q = result.quaternion
    g = result.gravity_direction
    print(f"Gravity direction: [{g[0]:.6f}, {g[1]:.6f}, {g[2]:.6f}]")
    print(f"Quaternion (w,x,y,z): [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
    print(f"Overhang area: {result.overhang_area:.4f}")
    print(f"Build height: {result.build_height:.4f}")
    if not np.isnan(result.surface_quality):
        print(f"Surface quality: {result.surface_quality:.4f}")
    print(f"Feasible candidates: {result.feasible_count}/{args.n_samples}")


if __name__ == "__main__":
    main()
