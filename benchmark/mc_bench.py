"""
Marching cubes benchmark: torchmcubes (CPU) vs pure-PyTorch (MPS/CPU).

Tests sphere and torus implicit surfaces at resolutions 128, 256, 384.
"""

import time
import numpy as np
import torch
from torchmcubes import marching_cubes as mc_cpu

# ── Lookup tables (standard Lorensen & Cline) ─────────────────────────────────

# fmt: off
EDGE_TABLE = [
    0x000,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,
    0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00,
    0x190,0x099,0x393,0x29a,0x596,0x49f,0x795,0x69c,
    0x99c,0x895,0xb9f,0xa96,0xd9a,0xc93,0xf99,0xe90,
    0x230,0x339,0x033,0x13a,0x636,0x73f,0x435,0x53c,
    0xa3c,0xb35,0x83f,0x936,0xe3a,0xf33,0xc39,0xd30,
    0x3a0,0x2a9,0x1a3,0x0aa,0x7a6,0x6af,0x5a5,0x4ac,
    0xbac,0xaa5,0x9af,0x8a6,0xfaa,0xea3,0xda9,0xca0,
    0x460,0x569,0x663,0x76a,0x066,0x16f,0x265,0x36c,
    0xc6c,0xd65,0xe6f,0xf66,0x86a,0x963,0xa69,0xb60,
    0x5f0,0x4f9,0x7f3,0x6fa,0x1f6,0x0ff,0x3f5,0x2fc,
    0xdfc,0xcf5,0xfff,0xef6,0x9fa,0x8f3,0xbf9,0xaf0,
    0x650,0x759,0x453,0x55a,0x256,0x35f,0x055,0x15c,
    0xe5c,0xf55,0xc5f,0xd56,0xa5a,0xb53,0x859,0x950,
    0x7c0,0x6c9,0x5c3,0x4ca,0x3c6,0x2cf,0x1c5,0x0cc,
    0xfcc,0xec5,0xdcf,0xcc6,0xbca,0xac3,0x9c9,0x8c0,
    0x8c0,0x9c9,0xac3,0xbca,0xcc6,0xdcf,0xec5,0xfcc,
    0x0cc,0x1c5,0x2cf,0x3c6,0x4ca,0x5c3,0x6c9,0x7c0,
    0x950,0x859,0xb53,0xa5a,0xd56,0xc5f,0xf55,0xe5c,
    0x15c,0x055,0x35f,0x256,0x55a,0x453,0x759,0x650,
    0xaf0,0xbf9,0x8f3,0x9fa,0xef6,0xfff,0xcf5,0xdfc,
    0x2fc,0x3f5,0x0ff,0x1f6,0x6fa,0x7f3,0x4f9,0x5f0,
    0xb60,0xa69,0x963,0x86a,0xf66,0xe6f,0xd65,0xc6c,
    0x36c,0x265,0x16f,0x066,0x76a,0x663,0x569,0x460,
    0xca0,0xda9,0xea3,0xfaa,0x8a6,0x9af,0xaa5,0xbac,
    0x4ac,0x5a5,0x6af,0x7a6,0x0aa,0x1a3,0x2a9,0x3a0,
    0xd30,0xc39,0xf33,0xe3a,0x936,0x83f,0xb35,0xa3c,
    0x53c,0x435,0x73f,0x636,0x13a,0x033,0x339,0x230,
    0xe90,0xf99,0xc93,0xd9a,0xa96,0xb9f,0x895,0x99c,
    0x69c,0x795,0x49f,0x596,0x29a,0x393,0x099,0x190,
    0xf00,0xe09,0xd03,0xc0a,0xb06,0xa0f,0x905,0x80c,
    0x70c,0x605,0x50f,0x406,0x30a,0x203,0x109,0x000,
]

TRI_TABLE = [
    [-1],
    [0,8,3,-1],
    [0,1,9,-1],
    [1,8,3,9,8,1,-1],
    [1,2,10,-1],
    [0,8,3,1,2,10,-1],
    [9,2,10,0,2,9,-1],
    [2,8,3,2,10,8,10,9,8,-1],
    [3,11,2,-1],
    [0,11,2,8,11,0,-1],
    [1,9,0,2,3,11,-1],
    [1,11,2,1,9,11,9,8,11,-1],
    [3,10,1,11,10,3,-1],
    [0,10,1,0,8,10,8,11,10,-1],
    [3,9,0,3,11,9,11,10,9,-1],
    [9,8,10,10,8,11,-1],
    [4,7,8,-1],
    [4,3,0,7,3,4,-1],
    [0,1,9,8,4,7,-1],
    [4,1,9,4,7,1,7,3,1,-1],
    [1,2,10,8,4,7,-1],
    [3,4,7,3,0,4,1,2,10,-1],
    [9,2,10,9,0,2,8,4,7,-1],
    [2,10,9,2,9,7,2,7,3,7,9,4,-1],
    [8,4,7,3,11,2,-1],
    [11,4,7,11,2,4,2,0,4,-1],
    [9,0,1,8,4,7,2,3,11,-1],
    [4,7,11,9,4,11,9,11,2,9,2,1,-1],
    [3,10,1,3,11,10,7,8,4,-1],
    [1,11,10,1,4,11,1,0,4,7,11,4,-1],
    [4,7,8,9,0,11,9,11,10,11,0,3,-1],
    [4,7,11,4,11,9,9,11,10,-1],
    [9,5,4,-1],
    [9,5,4,0,8,3,-1],
    [0,5,4,1,5,0,-1],
    [8,5,4,8,3,5,3,1,5,-1],
    [1,2,10,9,5,4,-1],
    [3,0,8,1,2,10,4,9,5,-1],
    [5,2,10,5,4,2,4,0,2,-1],
    [2,10,5,3,2,5,3,5,4,3,4,8,-1],
    [9,5,4,2,3,11,-1],
    [0,11,2,0,8,11,4,9,5,-1],
    [0,5,4,0,1,5,2,3,11,-1],
    [2,1,5,2,5,8,2,8,11,4,8,5,-1],
    [10,3,11,10,1,3,9,5,4,-1],
    [4,9,5,0,8,1,8,10,1,8,11,10,-1],
    [5,4,0,5,0,11,5,11,10,11,0,3,-1],
    [5,4,8,5,8,10,10,8,11,-1],
    [9,7,8,5,7,9,-1],
    [9,3,0,9,5,3,5,7,3,-1],
    [0,7,8,0,1,7,1,5,7,-1],
    [1,5,3,3,5,7,-1],
    [9,7,8,9,5,7,10,1,2,-1],
    [10,1,2,9,5,0,5,3,0,5,7,3,-1],
    [8,0,2,8,2,5,8,5,7,10,5,2,-1],
    [2,10,5,2,5,3,3,5,7,-1],
    [7,9,5,7,8,9,3,11,2,-1],
    [9,5,7,9,7,2,9,2,0,2,7,11,-1],
    [2,3,11,0,1,8,1,7,8,1,5,7,-1],
    [11,2,1,11,1,7,7,1,5,-1],
    [9,5,8,8,5,7,10,1,3,10,3,11,-1],
    [5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1],
    [11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1],
    [11,10,5,7,11,5,-1],
    [10,6,5,-1],
    [0,8,3,5,10,6,-1],
    [9,0,1,5,10,6,-1],
    [1,8,3,1,9,8,5,10,6,-1],
    [1,6,5,2,6,1,-1],
    [1,6,5,1,2,6,3,0,8,-1],
    [9,6,5,9,0,6,0,2,6,-1],
    [5,9,8,5,8,2,5,2,6,3,2,8,-1],
    [2,3,11,10,6,5,-1],
    [11,0,8,11,2,0,10,6,5,-1],
    [0,1,9,2,3,11,5,10,6,-1],
    [5,10,6,1,9,2,9,11,2,9,8,11,-1],
    [6,3,11,6,5,3,5,1,3,-1],
    [0,8,11,0,11,5,0,5,1,5,11,6,-1],
    [3,11,6,0,3,6,0,6,5,0,5,9,-1],
    [6,5,9,6,9,11,11,9,8,-1],
    [5,10,6,4,7,8,-1],
    [4,3,0,4,7,3,6,5,10,-1],
    [1,9,0,5,10,6,8,4,7,-1],
    [10,6,5,1,9,7,1,7,3,7,9,4,-1],
    [6,1,2,6,5,1,4,7,8,-1],
    [1,2,5,5,2,6,3,0,4,3,4,7,-1],
    [8,4,7,9,0,5,0,6,5,0,2,6,-1],
    [7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1],
    [3,11,2,7,8,4,10,6,5,-1],
    [5,10,6,4,7,2,4,2,0,2,7,11,-1],
    [0,1,9,4,7,8,2,3,11,5,10,6,-1],
    [9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1],
    [8,4,7,3,11,5,3,5,1,5,11,6,-1],
    [5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1],
    [0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1],
    [6,5,9,6,9,11,4,7,9,7,11,9,-1],
    [10,4,9,6,4,10,-1],
    [4,10,6,4,9,10,0,8,3,-1],
    [10,0,1,10,6,0,6,4,0,-1],
    [8,3,1,8,1,6,8,6,4,6,1,10,-1],
    [1,4,9,1,2,4,2,6,4,-1],
    [3,0,8,1,2,9,2,4,9,2,6,4,-1],
    [0,2,4,4,2,6,-1],
    [8,3,2,8,2,4,4,2,6,-1],
    [10,4,9,10,6,4,11,2,3,-1],
    [0,8,2,2,8,11,4,9,10,4,10,6,-1],
    [3,11,2,0,1,6,0,6,4,6,1,10,-1],
    [6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1],
    [9,6,4,9,3,6,9,1,3,11,6,3,-1],
    [8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1],
    [3,11,6,3,6,0,0,6,4,-1],
    [6,4,8,11,6,8,-1],
    [7,10,6,7,8,10,8,9,10,-1],
    [0,7,3,0,10,7,0,9,10,6,7,10,-1],
    [10,6,7,1,10,7,1,7,8,1,8,0,-1],
    [10,6,7,10,7,1,1,7,3,-1],
    [1,2,6,1,6,8,1,8,9,8,6,7,-1],
    [2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1],
    [7,8,0,7,0,6,6,0,2,-1],
    [7,3,2,6,7,2,-1],
    [2,3,11,10,6,8,10,8,9,8,6,7,-1],
    [2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1],
    [1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1],
    [11,2,1,11,1,7,10,6,1,6,7,1,-1],
    [8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1],
    [0,9,1,11,6,7,-1],
    [7,8,0,7,0,6,3,11,0,11,6,0,-1],
    [7,11,6,-1],
    [7,6,11,-1],
    [3,0,8,11,7,6,-1],
    [0,1,9,11,7,6,-1],
    [8,1,9,8,3,1,11,7,6,-1],
    [10,1,2,6,11,7,-1],
    [1,2,10,3,0,8,6,11,7,-1],
    [2,9,0,2,10,9,6,11,7,-1],
    [6,11,7,2,10,3,10,8,3,10,9,8,-1],
    [7,2,3,6,2,7,-1],
    [7,0,8,7,6,0,6,2,0,-1],
    [2,7,6,2,3,7,0,1,9,-1],
    [1,6,2,1,8,6,1,9,8,8,7,6,-1],
    [10,7,6,10,1,7,1,3,7,-1],
    [10,7,6,1,7,10,1,8,7,1,0,8,-1],
    [0,3,7,0,7,10,0,10,9,6,10,7,-1],
    [7,6,10,7,10,8,8,10,9,-1],
    [6,8,4,11,8,6,-1],
    [3,6,11,3,0,6,0,4,6,-1],
    [8,6,11,8,4,6,9,0,1,-1],
    [9,4,6,9,6,3,9,3,1,11,3,6,-1],
    [6,8,4,6,11,8,2,10,1,-1],
    [1,2,10,3,0,11,0,6,11,0,4,6,-1],
    [4,11,8,4,6,11,0,2,9,2,10,9,-1],
    [10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1],
    [8,2,3,8,4,2,4,6,2,-1],
    [0,4,2,4,6,2,-1],
    [1,9,0,2,3,4,2,4,6,4,3,8,-1],
    [1,9,4,1,4,2,2,4,6,-1],
    [8,1,3,8,6,1,8,4,6,6,10,1,-1],
    [10,1,0,10,0,6,6,0,4,-1],
    [4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1],
    [10,9,4,6,10,4,-1],
    [4,9,5,7,6,11,-1],
    [0,8,3,4,9,5,11,7,6,-1],
    [5,0,1,5,4,0,7,6,11,-1],
    [11,7,6,8,3,4,3,5,4,3,1,5,-1],
    [9,5,4,10,1,2,7,6,11,-1],
    [6,11,7,1,2,10,0,8,3,4,9,5,-1],
    [7,6,11,5,4,10,4,2,10,4,0,2,-1],
    [3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1],
    [7,2,3,7,6,2,5,4,9,-1],
    [9,5,4,0,8,6,0,6,2,6,8,7,-1],
    [3,6,2,3,7,6,1,5,0,5,4,0,-1],
    [6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1],
    [9,5,4,10,1,6,1,7,6,1,3,7,-1],
    [1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1],
    [4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1],
    [7,6,10,7,10,8,5,4,10,4,8,10,-1],
    [6,9,5,6,11,9,11,8,9,-1],
    [3,6,11,0,6,3,0,5,6,0,9,5,-1],
    [0,11,8,0,5,11,0,1,5,5,6,11,-1],
    [6,11,3,6,3,5,5,3,1,-1],
    [1,2,10,9,5,11,9,11,8,11,5,6,-1],
    [0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1],
    [11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1],
    [6,11,3,6,3,5,2,10,3,10,5,3,-1],
    [5,8,9,5,2,8,5,6,2,3,8,2,-1],
    [9,5,6,9,6,0,0,6,2,-1],
    [1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1],
    [1,5,6,2,1,6,-1],
    [1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1],
    [10,1,0,10,0,6,9,5,0,5,6,0,-1],
    [0,3,8,5,6,10,-1],
    [10,5,6,-1],
    [11,5,10,7,5,11,-1],
    [11,5,10,11,7,5,8,3,0,-1],
    [5,11,7,5,10,11,1,9,0,-1],
    [10,7,5,10,11,7,9,8,1,8,3,1,-1],
    [11,1,2,11,7,1,7,5,1,-1],
    [0,8,3,1,2,7,1,7,5,7,2,11,-1],
    [9,7,5,9,2,7,9,0,2,2,11,7,-1],
    [7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1],
    [2,5,10,2,3,5,3,7,5,-1],
    [8,2,0,8,5,2,8,7,5,10,2,5,-1],
    [9,0,1,2,3,5,2,5,10,5,3,7,-1],
    [8,2,9,8,9,7,2,10,9,5,9,3,10,3,9,-1],
    [1,3,5,3,7,5,-1],
    [0,8,7,0,7,1,1,7,5,-1],
    [9,0,3,9,3,5,5,3,7,-1],
    [9,8,7,5,9,7,-1],
    [5,8,4,5,10,8,10,11,8,-1],
    [5,0,4,5,11,0,5,10,11,11,3,0,-1],
    [0,1,9,8,4,10,8,10,11,10,4,5,-1],
    [10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1],
    [2,5,1,2,8,5,2,11,8,4,5,8,-1],
    [0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1],
    [0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1],
    [9,4,5,2,11,3,-1],
    [2,5,10,3,5,2,3,4,5,3,8,4,-1],
    [5,10,2,5,2,4,4,2,0,-1],
    [3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1],
    [5,10,2,5,2,4,1,9,2,9,4,2,-1],
    [8,4,5,8,5,3,3,5,1,-1],
    [0,4,5,1,0,5,-1],
    [8,4,5,8,5,3,9,0,5,0,3,5,-1],
    [9,4,5,-1],
    [4,11,7,4,9,11,9,10,11,-1],
    [0,8,3,4,9,7,9,11,7,9,10,11,-1],
    [1,10,11,1,11,4,1,4,0,7,4,11,-1],
    [3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1],
    [4,11,7,9,11,4,9,2,11,9,1,2,-1],
    [9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1],
    [11,7,4,11,4,2,2,4,0,-1],
    [11,7,4,11,4,2,8,3,4,3,2,4,-1],
    [2,9,10,2,7,9,2,3,7,7,4,9,-1],
    [9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1],
    [3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1],
    [1,10,2,8,7,4,-1],
    [4,9,1,4,1,7,7,1,3,-1],
    [4,9,1,4,1,7,0,8,1,8,7,1,-1],
    [4,0,3,7,4,3,-1],
    [4,8,7,-1],
    [9,10,8,10,11,8,-1],
    [3,0,9,3,9,11,11,9,10,-1],
    [0,1,10,0,10,8,8,10,11,-1],
    [3,1,10,11,3,10,-1],
    [1,2,11,1,11,9,9,11,8,-1],
    [3,0,9,3,9,11,1,2,9,2,11,9,-1],
    [0,2,11,8,0,11,-1],
    [3,2,11,-1],
    [2,3,8,2,8,10,10,8,9,-1],
    [9,10,2,0,9,2,-1],
    [2,3,8,2,8,10,0,1,8,1,10,8,-1],
    [1,10,2,-1],
    [1,3,8,9,1,8,-1],
    [0,9,1,-1],
    [0,3,8,-1],
    [-1],
]

# Edge endpoint corner indices
EDGE_CORNERS = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

# Corner offsets (dx, dy, dz)
CORNER_OFFSETS = [
    (0,0,0),(1,0,0),(1,1,0),(0,1,0),
    (0,0,1),(1,0,1),(1,1,1),(0,1,1),
]
# fmt: on


# ── Implicit surfaces ─────────────────────────────────────────────────────────

def sphere_field(res: int, device: torch.device) -> torch.Tensor:
    """Signed distance to unit sphere, values negative inside."""
    t = torch.linspace(-1.5, 1.5, res, device=device)
    x, y, z = torch.meshgrid(t, t, t, indexing="ij")
    return x**2 + y**2 + z**2 - 1.0


def torus_field(res: int, device: torch.device) -> torch.Tensor:
    """Signed distance to torus (R=0.8, r=0.3)."""
    t = torch.linspace(-1.5, 1.5, res, device=device)
    x, y, z = torch.meshgrid(t, t, t, indexing="ij")
    R, r = 0.8, 0.3
    q = (torch.sqrt(x**2 + z**2) - R)**2 + y**2 - r**2
    return q


# ── Pure-PyTorch marching cubes (MPS-compatible) ──────────────────────────────

def mc_torch(field: torch.Tensor, iso: float = 0.0):
    """
    Vectorised marching cubes. Works on any PyTorch device (CPU / MPS / CUDA).
    Returns (vertices [V,3], faces [F,3]) as CPU tensors.
    """
    device = field.device
    R = field.shape[0]
    N = R - 1  # cells per axis

    # ── 1. Gather corner values for every cell ────────────────────────────────
    # corners: list of 8 tensors each [N,N,N]
    corners = [
        field[ox:ox+N, oy:oy+N, oz:oz+N]
        for (ox, oy, oz) in CORNER_OFFSETS
    ]
    # signs: True means value >= iso  → inside (positive side)
    signs = torch.stack([(c >= iso) for c in corners], dim=-1)  # [N,N,N,8]

    # ── 2. Compute case index ─────────────────────────────────────────────────
    # bit i = sign of corner i
    bits = torch.tensor([1 << i for i in range(8)], dtype=torch.int32, device=device)
    case_idx = (signs.int() * bits).sum(dim=-1).int()  # [N,N,N]

    # ── 3. Filter active cells (case not 0 or 255) ───────────────────────────
    active_mask = (case_idx != 0) & (case_idx != 255)
    active_cases = case_idx[active_mask].cpu().tolist()
    if not active_cases:
        return torch.zeros(0, 3), torch.zeros(0, 3, dtype=torch.long)

    # Cell origin coordinates (ix, iy, iz) for active cells
    idx3 = active_mask.nonzero(as_tuple=False)  # [A, 3]  on device
    ix = idx3[:, 0].float()
    iy = idx3[:, 1].float()
    iz = idx3[:, 2].float()

    # Corner values for active cells: [A, 8]
    active_corners = torch.stack(
        [corners[k][active_mask] for k in range(8)], dim=1
    )  # [A, 8] — on device

    # ── 4. For each active cell, find triangles and interpolate ───────────────
    scale = 3.0 / (R - 1)  # voxel-to-world scale (field spans -1.5..1.5)

    all_verts = []
    all_faces = []
    vert_count = 0

    # Process in Python (variable-length output per cell)
    # Move needed tensors to CPU once for the loop
    ix_c  = ix.cpu()
    iy_c  = iy.cpu()
    iz_c  = iz_c = iz.cpu()
    ac_c  = active_corners.cpu()

    for cell_i, case in enumerate(active_cases):
        tris = TRI_TABLE[case]
        tri_edges = [e for e in tris if e != -1]
        if not tri_edges:
            continue

        ci_x = ix_c[cell_i].item()
        ci_y = iy_c[cell_i].item()
        ci_z = iz_c[cell_i].item()
        cv   = ac_c[cell_i]  # [8]

        # Interpolate vertex for each edge
        edge_verts = {}
        for e in set(tri_edges):
            c0, c1 = EDGE_CORNERS[e]
            v0 = cv[c0].item()
            v1 = cv[c1].item()
            denom = v1 - v0
            t = 0.5 if abs(denom) < 1e-10 else (iso - v0) / denom
            t = max(0.0, min(1.0, t))
            ox0, oy0, oz0 = CORNER_OFFSETS[c0]
            ox1, oy1, oz1 = CORNER_OFFSETS[c1]
            wx = (ci_x + ox0 + t * (ox1 - ox0)) * scale - 1.5
            wy = (ci_y + oy0 + t * (oy1 - oy0)) * scale - 1.5
            wz = (ci_z + oz0 + t * (oz1 - oz0)) * scale - 1.5
            edge_verts[e] = [wx, wy, wz]

        # Emit triangles
        for i in range(0, len(tri_edges), 3):
            e0, e1, e2 = tri_edges[i], tri_edges[i+1], tri_edges[i+2]
            all_verts.extend([edge_verts[e0], edge_verts[e1], edge_verts[e2]])
            all_faces.append([vert_count, vert_count+1, vert_count+2])
            vert_count += 3

    if not all_verts:
        return torch.zeros(0, 3), torch.zeros(0, 3, dtype=torch.long)

    verts = torch.tensor(all_verts, dtype=torch.float32)
    faces = torch.tensor(all_faces, dtype=torch.long)
    return verts, faces


# ── Pre-build padded TRI_TABLE tensor ─────────────────────────────────────────

def _build_tri_table_tensor() -> torch.Tensor:
    """[256, 15] int32 tensor; -1 for unused slots (max 5 tris × 3 edges = 15)."""
    t = torch.full((256, 15), -1, dtype=torch.int32)
    for i, row in enumerate(TRI_TABLE):
        vals = [v for v in row if v != -1]
        t[i, :len(vals)] = torch.tensor(vals, dtype=torch.int32)
    return t

_TRI_TABLE_T: torch.Tensor = _build_tri_table_tensor()  # built once on CPU

def _tri_table_on(device: torch.device) -> torch.Tensor:
    return _TRI_TABLE_T.to(device)


# ── Fully vectorised marching cubes (MPS/CUDA/CPU) ────────────────────────────

# Edge endpoint corner pairs for all 12 edges, as index tensors
_EC0 = torch.tensor([c[0] for c in EDGE_CORNERS], dtype=torch.long)
_EC1 = torch.tensor([c[1] for c in EDGE_CORNERS], dtype=torch.long)
_CO  = torch.tensor(CORNER_OFFSETS, dtype=torch.float32)  # [8,3]


def mc_torch_vec(field: torch.Tensor, iso: float = 0.0):
    """
    Fully vectorised marching cubes — no Python loop over cells.
    Works on any PyTorch device (CPU / MPS / CUDA).
    Returns (vertices [V,3], faces [F,3]) as CPU tensors.
    """
    device = field.device
    R = field.shape[0]
    N = R - 1

    # ── 1. Corner values and case index ──────────────────────────────────────
    corners = torch.stack(
        [field[ox:ox+N, oy:oy+N, oz:oz+N] for (ox, oy, oz) in CORNER_OFFSETS],
        dim=-1
    )  # [N,N,N,8]

    bits = torch.tensor([1 << i for i in range(8)], dtype=torch.int32, device=device)
    case_idx = ((corners >= iso).int() * bits).sum(-1).int()  # [N,N,N]

    # ── 2. Active cells ───────────────────────────────────────────────────────
    active_mask = (case_idx != 0) & (case_idx != 255)
    active_cells = active_mask.nonzero(as_tuple=False).float()  # [A,3]
    A = active_cells.shape[0]
    if A == 0:
        return torch.zeros(0, 3), torch.zeros(0, 3, dtype=torch.long)

    active_cases   = case_idx[active_mask].long()            # [A]
    active_cv      = corners[active_mask]                    # [A,8]

    # ── 3. Look up triangle edge indices: [A,15] ─────────────────────────────
    tri_tbl = _tri_table_on(device)                          # [256,15]
    tri_rows = tri_tbl[active_cases]                         # [A,15]

    # ── 4. Interpolate all 12 edge vertices for every active cell: [A,12,3] ──
    ec0 = _EC0.to(device)   # [12]
    ec1 = _EC1.to(device)   # [12]
    co  = _CO.to(device)    # [8,3]

    v0 = active_cv[:, ec0]   # [A,12]
    v1 = active_cv[:, ec1]   # [A,12]
    denom = v1 - v0
    t = torch.where(denom.abs() < 1e-10,
                    torch.full_like(denom, 0.5),
                    (iso - v0) / denom)
    t = t.clamp(0.0, 1.0)   # [A,12]

    off0 = co[ec0]           # [12,3]
    off1 = co[ec1]           # [12,3]
    # cell origins: [A,1,3] + edge interpolation [A,12,3]
    origins = active_cells.unsqueeze(1)                      # [A,1,3]
    # t: [A,12] → [A,12,1]
    edge_pts = origins + off0.unsqueeze(0) + t.unsqueeze(-1) * (off1 - off0).unsqueeze(0)
    # convert grid coords to world coords (field spans -1.5..1.5)
    edge_pts = edge_pts * (3.0 / (R - 1)) - 1.5             # [A,12,3]

    # ── 5. Gather triangle vertices ───────────────────────────────────────────
    # tri_rows: [A,15]  values 0-11 = edge index, -1 = no triangle
    valid  = tri_rows >= 0                                   # [A,15] bool
    # clamp -1 → 0 so gather doesn't OOB; mask will discard those
    ei     = tri_rows.clamp(min=0).long()                    # [A,15]

    # [A,15,3]: gather edge vertex positions for each of the 15 slots
    tri_verts = edge_pts.gather(
        1,
        ei.unsqueeze(-1).expand(-1, -1, 3)
    )                                                        # [A,15,3]

    # Keep only valid vertex slots and reshape to triangles
    valid_flat = valid.reshape(-1)                           # [A*15]
    verts_flat = tri_verts.reshape(-1, 3)                    # [A*15, 3]
    verts_out  = verts_flat[valid_flat].cpu()                # [V*3, 3]

    n_tris = verts_out.shape[0] // 3
    faces_out = torch.arange(n_tris * 3, dtype=torch.long).reshape(n_tris, 3)

    return verts_out, faces_out


# ── Benchmark runner ──────────────────────────────────────────────────────────

def bench(label: str, fn, warmup=1, runs=3):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize() if torch.backends.mps.is_available() else None
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize() if torch.backends.mps.is_available() else None
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times)//2]
    print(f"  {label:<45} {med*1000:7.1f} ms")
    return med


def main():
    cpu = torch.device("cpu")
    mps = torch.device("mps") if torch.backends.mps.is_available() else None

    print("\n=== Marching Cubes: torchmcubes(CPU) vs PyTorch-native ===\n")

    for surface_name, surface_fn in [("sphere", sphere_field), ("torus", torus_field)]:
        print(f"Surface: {surface_name}")
        print(f"  {'Method':<45} {'Median':>9}")
        print(f"  {'-'*55}")

        for res in [128, 256, 384]:
            field_cpu = surface_fn(res, cpu)
            field_np  = field_cpu.numpy()

            # torchmcubes baseline (CPU)
            def run_torchmcubes():
                level = -field_cpu.view(res, res, res)
                mc_cpu(level.detach().cpu(), 0.0)

            t_cpu = bench(f"res={res}  torchmcubes  (CPU)", run_torchmcubes)

            # PyTorch-native on CPU
            def run_torch_cpu():
                mc_torch(field_cpu, iso=0.0)

            t_torch_cpu = bench(f"res={res}  mc_torch     (CPU)", run_torch_cpu)

            # Vectorised on CPU
            def run_torch_vec_cpu():
                mc_torch_vec(field_cpu, iso=0.0)

            t_vec_cpu = bench(f"res={res}  mc_torch_vec (CPU)", run_torch_vec_cpu)

            if mps:
                field_mps = surface_fn(res, mps)

                def run_torch_mps():
                    mc_torch(field_mps, iso=0.0)
                t_torch_mps = bench(f"res={res}  mc_torch     (MPS)", run_torch_mps)

                def run_torch_vec_mps():
                    mc_torch_vec(field_mps, iso=0.0)
                t_vec_mps = bench(f"res={res}  mc_torch_vec (MPS)", run_torch_vec_mps)

                print(f"  {'':45}   torchmcubes baseline : {t_cpu*1000:.1f} ms")
                print(f"  {'':45}   mc_torch_vec MPS     : {t_vec_mps*1000:.1f} ms  "
                      f"({t_cpu/t_vec_mps:.2f}× vs torchmcubes)")

        print()


if __name__ == "__main__":
    main()
