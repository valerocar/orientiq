#!/usr/bin/env python3
"""Bracket v2 – reconstructed from 4-view images."""

import numpy as np
import trimesh
import trimesh.creation as tc
from shapely.geometry import Polygon

# ── dimensions (mm) ──────────────────────────────────────────────────────────
BASE_W   = 162   # base frame outer width  (X)
BASE_D   = 128   # base frame outer depth  (Y)
BASE_T   =   3   # base thickness
FLANGE_X =  24   # side flange width  (base extends this far beyond wall)
FLANGE_Y =  19   # front/back flange depth

WALL_T   =   4   # all wall thickness
INNER_W  = BASE_W - 2 * FLANGE_X   # 114  (between side wall inner faces)
INNER_D  = BASE_D - 2 * FLANGE_Y   # 90   (front-to-back cavity)

H_BACK   =  96   # side wall height at back
H_FRONT  =  52   # side wall height at front (above notch)

NOTCH_Y  =  28   # notch extent from front edge (Y direction)
NOTCH_Z  =  22   # notch height from base top   (Z direction)

DIV_T    =   4   # center divider thickness

TOP_PANEL_D = 40  # horizontal top panel depth (front-to-back span, at the back)
TOP_PANEL_T =  4  # top panel thickness

BIG_R    = 10.5  # large side-wall hole radius
SMALL_R  =  2.3  # small hole radius
MOUNT_R  =  3.5  # base corner mounting hole radius

SLOT_W   = 22    # rectangular slot width  (Y direction)
SLOT_H   =  6    # rectangular slot height (Z direction in wall, or X in top panel)


# ── helpers ───────────────────────────────────────────────────────────────────

def extrude_in_x(polygon: Polygon, thickness: float) -> trimesh.Trimesh:
    """Extrude a (Y,Z) Shapely polygon by `thickness` in the +X direction."""
    mesh = tc.extrude_polygon(polygon, thickness)
    # extrude_polygon puts extrusion in Z, polygon in X,Y.
    # Remap:  new_X = old_Z,  new_Y = old_X,  new_Z = old_Y
    T = np.array([[0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=float)
    mesh.apply_transform(T)
    return mesh


def extrude_in_y(polygon: Polygon, thickness: float) -> trimesh.Trimesh:
    """Extrude a (X,Z) Shapely polygon by `thickness` in the +Y direction."""
    mesh = tc.extrude_polygon(polygon, thickness)
    # Remap:  new_X = old_X,  new_Y = old_Z,  new_Z = old_Y
    T = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=float)
    mesh.apply_transform(T)
    return mesh


def cyl_x(r, L, x, y, z, n=32):
    """Cylinder along X axis centred at (x,y,z)."""
    c = tc.cylinder(r, L + 1, sections=n)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    c.apply_translation([x, y, z])
    return c


def cyl_z(r, L, x, y, z, n=24):
    """Cylinder along Z axis centred at (x,y,z)."""
    c = tc.cylinder(r, L + 1, sections=n)
    c.apply_translation([x, y, z])
    return c


def box_at(w, d, h, x, y, z):
    b = tc.box([w, d, h])
    b.apply_translation([x, y, z])
    return b


def subtract(base, *tools):
    try:
        all_tools = trimesh.util.concatenate(list(tools))
        return trimesh.boolean.difference([base, all_tools], engine='manifold')
    except Exception as e:
        print(f"  boolean failed: {e}")
        return base


# ── main ─────────────────────────────────────────────────────────────────────

def make_bracket():
    z0  = BASE_T
    cx, cy = 0.0, 0.0

    yf = cy - INNER_D / 2   # front inner Y
    yb = cy + INNER_D / 2   # back inner Y

    lx_out = cx - INNER_W / 2 - WALL_T   # left wall outer face
    lx_in  = cx - INNER_W / 2            # left wall inner face
    rx_in  = cx + INNER_W / 2            # right wall inner face
    rx_out = cx + INNER_W / 2 + WALL_T   # right wall outer face
    lx_mid = (lx_out + lx_in) / 2
    rx_mid = (rx_in + rx_out) / 2

    # ── 1. BASE FRAME ─────────────────────────────────────────────────────────
    base_outer = box_at(BASE_W, BASE_D, BASE_T, cx, cy, BASE_T / 2)
    base_void  = box_at(INNER_W, INNER_D, BASE_T + 2, cx, cy, BASE_T / 2)
    base_holes = [base_void]
    for sx, sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        base_holes.append(cyl_z(MOUNT_R, BASE_T,
                                cx + sx*(BASE_W/2 - 14),
                                cy + sy*(BASE_D/2 - 14),
                                BASE_T/2))
    base = subtract(base_outer, *base_holes)

    # ── 2. SIDE WALL PROFILE (angled top + front-bottom notch) ────────────────
    # Polygon in (Y_local, Z), Y_local = 0 at front, INNER_D at back
    D = INNER_D
    side_poly = Polygon([
        (0,       NOTCH_Z),    # front bottom (above notch)
        (0,       H_FRONT),    # front top
        (D,       H_BACK),     # back top
        (D,       0),          # back bottom
        (NOTCH_Y, 0),          # notch back corner
        (NOTCH_Y, NOTCH_Z),    # notch inner corner
    ])

    lw_solid = extrude_in_x(side_poly, WALL_T)
    lw_solid.apply_translation([lx_out, yf, z0])

    rw_solid = extrude_in_x(side_poly, WALL_T)
    rw_solid.apply_translation([rx_in, yf, z0])

    # Holes in each side wall: 1 large + 1 small above + 2 small below
    hole_z = z0 + H_BACK * 0.58
    hole_y = cy - INNER_D * 0.08   # slightly toward front

    wall_holes = []
    for wx in (lx_mid, rx_mid):
        wall_holes.append(cyl_x(BIG_R,   WALL_T, wx, hole_y, hole_z))
        wall_holes.append(cyl_x(SMALL_R, WALL_T, wx, hole_y, hole_z + 18))   # above
        wall_holes.append(cyl_x(SMALL_R, WALL_T, wx, hole_y, hole_z - 15))   # below 1
        wall_holes.append(cyl_x(SMALL_R, WALL_T, wx, hole_y, hole_z - 26))   # below 2

    lw = subtract(lw_solid, *wall_holes[:4])
    rw = subtract(rw_solid, *wall_holes[4:])

    # ── 3. BACK WALL (arched top) ──────────────────────────────────────────────
    back_W = INNER_W + 2 * WALL_T
    # Parabolic arch: H_BACK at edges, peak at centre
    arch_extra = 18
    n_arc = 10
    arc_xs = np.linspace(-back_W / 2, back_W / 2, n_arc)
    arc_zs = H_BACK + arch_extra * (1 - (2 * arc_xs / back_W) ** 2)
    arc_pts = [(x, z) for x, z in zip(arc_xs, arc_zs)]
    back_poly = Polygon(arc_pts + [(back_W/2, 0), (-back_W/2, 0)])
    bw = extrude_in_y(back_poly, WALL_T)
    bw.apply_translation([cx - back_W/2, yb, z0])

    # ── 4. HORIZONTAL TOP PANEL (at back, with slots + small holes) ────────────
    top_panel_z = z0 + H_BACK   # sits at the top of the walls at full height
    # Spans inner width, runs from back edge forward by TOP_PANEL_D
    top_solid = box_at(INNER_W, TOP_PANEL_D, TOP_PANEL_T,
                       cx, yb - TOP_PANEL_D/2, top_panel_z + TOP_PANEL_T/2)

    # 2 rectangular slots through the top panel (in Z direction)
    slot_cuts = []
    for sy_off in (-TOP_PANEL_D*0.2, TOP_PANEL_D*0.2):
        slot_cuts.append(box_at(SLOT_W, SLOT_H, TOP_PANEL_T + 2,
                                cx, yb - TOP_PANEL_D/2 + sy_off,
                                top_panel_z + TOP_PANEL_T/2))
    # 2 small holes in top panel
    for sx_off in (-INNER_W*0.2, INNER_W*0.2):
        slot_cuts.append(cyl_z(SMALL_R + 1, TOP_PANEL_T,
                               cx + sx_off, yb - TOP_PANEL_D*0.6,
                               top_panel_z + TOP_PANEL_T/2))

    top_panel = subtract(top_solid, *slot_cuts)

    # ── 5. CENTER DIVIDER (front-to-back, same angled top as side walls) ───────
    div_poly = Polygon([
        (0, 0), (D, 0), (D, H_BACK), (0, H_FRONT)
    ])
    div = extrude_in_x(div_poly, DIV_T)
    div.apply_translation([cx - DIV_T / 2, yf, z0])

    # ── 6. SMALL TOP TAB above back wall ─────────────────────────────────────
    tab = box_at(back_W, WALL_T, 14, cx, yb + WALL_T/2, z0 + H_BACK + arch_extra/2 + 7)

    # ── ASSEMBLE ──────────────────────────────────────────────────────────────
    result = trimesh.util.concatenate([base, lw, rw, bw, top_panel, div, tab])

    out = "3dmodels/bracket_v2.stl"
    result.export(out)
    print(f"Exported {out}: {len(result.vertices)} vertices, {len(result.faces)} faces")
    return result


if __name__ == "__main__":
    make_bracket()
