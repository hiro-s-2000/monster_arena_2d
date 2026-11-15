#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Sceptile (`0254_sceptile-mega`) modeled after Normal Sceptile's official GIF
(`0254_sceptile/normal/front.gif`) and Mega Sceptile's anatomical design (`front.png` & `back.png`):
  - Exact 9-layer anatomical rigging with true anatomical joint pivots (shoulder sockets & hip tail socket):
      1. `tail` (massive Christmas-tree rocket leaf tail with red cone tip — 100% separated from both feet,
                 pivoted at the hip socket so the tail base never detaches while the rocket tip whips and drills!)
      2. `back_seeds` (yellow/red dorsal nutrient orbs and sharp X-shaped chest leaf armor flares)
      3. `arm_far` (far-side dual leaf-blade arm — pivoted at the far shoulder socket)
      4. `foot_L` (left lower leg & 3-toed foot — 100% grounded `(0,0)` in `idle`, agile raptor stride in `walk`)
      5. `torso` (central green body, neck, red belly sash, thighs, and dark-red mouth interior pocket)
      6. `foot_R` (right lower leg & 3-toed foot — 100% grounded `(0,0)` in `idle`, alternating stride in `walk`)
      7. `lower_jaw` (red lower jaw — opens `+1px` for breathing/barking roar in `idle`/`walk` and `+2px` wide
                      for a ferocious roar during `attack` wind-up and `hurt` reaction!)
      8. `head_upper` (twin leaf head-crests, cranium, fierce yellow/red eye, and upper snout — 100% rigid
                       integer motion `shift_layer_int` so the eye and face never deform by a single pixel)
      9. `arm_near` (foreground dual leaf-blade arm — pivoted at the near shoulder socket for spinning
                     Leaf Blade slashes in `attack`!)
"""

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mouth_open  # noqa: E402  small mouth opening along the face (2026-11-04)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0254_sceptile/mega/sceptile-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0254_sceptile/mega/sceptile-mega"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16
VERSION = "sceptile_mega_v1m"

# Mouth (2026-11-04 user request): the lower jaw no longer drops and no rectangular interior is painted.
# A small lens-shaped opening is carved along the closed mouth line of front.png (native 96x96 coords,
# ordered mouth corner -> lip tip). Back views show no mouth and get no opening.
MOUTH_LINE_FRONT = [(28.0, 22.0), (31.0, 22.0), (35.0, 22.0)]
MOUTH_DEEP = (0, 0, 0)        # darkest palette colour (outline)
MOUTH_INNER = (128, 13, 0)    # existing dark-red mouth-interior palette colour


def extract_palette(front_arr: np.ndarray, back_arr: np.ndarray) -> np.ndarray:
    colors = []
    for arr in (front_arr, back_arr):
        m = arr[:, :, 3] >= 128
        if np.any(m):
            rgb = arr.copy()
            rgb[:, :, 3] = 255
            colors.append(rgb[m])
    all_c = np.concatenate(colors, axis=0)
    return np.unique(all_c, axis=0)


def snap_to_palette(arr: np.ndarray, palette_rgba: np.ndarray) -> np.ndarray:
    m = arr[:, :, 3] >= 128
    out = np.zeros_like(arr)
    if not np.any(m):
        return out
    px = arr[m, :3].astype(np.int32)
    pal = palette_rgba[:, :3].astype(np.int32)
    diff = px[:, None, :] - pal[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    nearest = np.argmin(dist2, axis=1)
    out[m, :3] = palette_rgba[nearest, :3]
    out[m, 3] = 255
    return out


def is_dark_outline(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 0) & (g == 0) & (b == 0))
        | ((r == 16) & (g == 16) & (b == 16))
        | ((r == 24) & (g == 74) & (b == 41))
        | ((r == 32) & (g == 74) & (b == 32))
        | ((r == 90) & (g == 82) & (b == 8))
    )


def is_green_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 65) & (g == 156) & (b == 65))
        | ((r == 57) & (g == 131) & (b == 57))
        | ((r == 24) & (g == 98) & (b == 41))
        | ((r == 131) & (g == 205) & (b == 106))
    )


def build_part_layer(
    src: np.ndarray,
    mask: np.ndarray,
    foreground_mask: np.ndarray,
    cushion_px: int = 5,
    allowed_fill_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    layer = np.zeros_like(src)
    layer[mask] = src[mask]
    if cushion_px <= 0 or not np.any(mask) or not np.any(foreground_mask):
        return layer
    dist_to_mask = ndi.distance_transform_edt(~mask)
    seam = foreground_mask & (~mask) & (dist_to_mask >= 1) & (dist_to_mask <= cushion_px)
    if not np.any(seam):
        return layer
    fill_candidates = mask & (~is_dark_outline(src))
    if allowed_fill_mask is not None and np.any(fill_candidates & allowed_fill_mask):
        fill_candidates = fill_candidates & allowed_fill_mask
    if not np.any(fill_candidates):
        fill_candidates = mask
    _, (ny_idx, nx_idx) = ndi.distance_transform_edt(~fill_candidates, return_indices=True)
    sy, sx = np.nonzero(seam)
    layer[sy, sx] = src[ny_idx[sy, sx], nx_idx[sy, sx]]
    return layer


def shift_layer_int(layer: np.ndarray, dx: int, dy: int) -> np.ndarray:
    dx_i = int(round(dx))
    dy_i = int(round(dy))
    if dx_i == 0 and dy_i == 0:
        return layer.copy()
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0 = max(0, -dx_i)
    sx1 = min(w, w - dx_i)
    sy0 = max(0, -dy_i)
    sy1 = min(h, h - dy_i)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy_i:sy1 + dy_i, sx0 + dx_i:sx1 + dx_i] = layer[sy0:sy1, sx0:sx1]
    return out


def rotate_layer_nearest(
    layer: np.ndarray,
    angle_deg: float,
    pivot_xy: Tuple[float, float],
) -> np.ndarray:
    """
    Pure nearest-neighbor inverse-mapped rotation around anatomical joint `pivot_xy = (px, py)`.
    Preserves exact palette colors with zero interpolation blur.
    """
    if abs(angle_deg) < 0.4:
        return layer.copy()
    a = layer[:, :, 3] >= 128
    if not np.any(a):
        return layer.copy()
    h, w = layer.shape[:2]
    px, py = pivot_xy
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    yy, xx = np.mgrid[0:h, 0:w]
    rx = xx - px
    ry = yy - py
    src_x = np.rint(px + cos_a * rx + sin_a * ry).astype(np.int32)
    src_y = np.rint(py - sin_a * rx + cos_a * ry).astype(np.int32)

    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(layer)
    vy, vx = np.nonzero(valid)
    out[vy, vx] = layer[src_y[vy, vx], src_x[vy, vx]]
    return out


def composite_layers(canvas_shape: Tuple[int, int, int], layers: List[np.ndarray]) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=np.uint8)
    for lyr in layers:
        m = lyr[..., 3] >= 128
        out[m] = lyr[m]
    return out


def segment_front_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `front.png` into 9 exact anatomical masks following 1px outlines:
      1. `tail`: Giant Christmas-tree rocket leaf tail (`x=38..95, y=51..86`), including the bottom frond tips
                 peeking behind the right ankle (`y=80..86, x=38..45`) — 100% separated from `foot_R`!
      2. `back_seeds`: Yellow dorsal neck seeds + sharp upper chest-leaf collar flares (`x=14..50, y=26..40`).
      3. `arm_far`: Far-side left dual leaf-blade arm (`x=0..18, y=22..56`).
      4. `foot_L`: Grounded left lower leg & 3-toed foot (`x=14..27, y=71..84`).
      5. `torso`: Neck, chest X-leaf center, belly with red sash, and thighs.
      6. `foot_R`: Grounded right lower leg & 3-toed foot (`x=37..54, y=74..92`).
      7. `lower_jaw`: Red lower jaw and chin outline (`x=26..37, y=23..26`).
      8. `head_upper`: Twin leaf head-crests, cranium, yellow/red eye, and upper snout (`x=17..40, y=4..23`).
      9. `arm_near`: Foreground right dual leaf-blade arm & 3-fingered hand (`x=35..65, y=30..62`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Lower jaw (red lower jaw + chin outline)
    jaw_zone = (
        ((yy == 23) & (xx >= 26) & (xx <= 35))
        | ((yy == 24) & (xx >= 26) & (xx <= 37))
        | ((yy == 25) & (xx >= 26) & (xx <= 34))
        | ((yy == 26) & (xx >= 27) & (xx <= 32))
    )
    lower_jaw = a & jaw_zone

    # 2. Head upper (head crests, cranium, eye, upper snout)
    head_zone = (
        ((yy <= 22) & (xx >= 17))
        | ((yy == 23) & (xx >= 36) & (xx <= 40))
    )
    head_upper = a & head_zone & (~lower_jaw)

    # 3. Arm far (left dual leaf-blade arm + 3-fingered hand)
    arm_far_zone = (
        ((yy >= 22) & (yy <= 40) & (xx <= 13))
        | ((yy == 41) & (xx <= 11))
        | ((yy == 42) & (xx <= 12))
        | ((yy == 43) & (xx <= 13))
        | ((yy == 44) & (xx <= 17))
        | ((yy == 45) & (xx <= 18))
        | ((yy == 46) & (xx <= 17))
        | ((yy == 47) & (xx <= 17))
        | ((yy >= 48) & (yy <= 56) & (xx <= 13))
    )
    arm_far = a & arm_far_zone

    # 4. Arm near (foreground right dual leaf-blade arm + 3-fingered hand)
    arm_near_zone = (
        ((yy == 30) & (xx >= 47) & (xx <= 49))
        | ((yy >= 31) & (yy <= 39) & (xx >= 46) & (xx <= 50))
        | ((yy == 39) & (xx >= 36) & (xx <= 37))
        | ((yy == 40) & (((xx >= 35) & (xx <= 43)) | ((xx >= 46) & (xx <= 50))))
        | ((yy == 41) & (((xx >= 35) & (xx <= 44)) | ((xx >= 46) & (xx <= 50))))
        | ((yy == 42) & (((xx >= 35) & (xx <= 44)) | ((xx >= 46) & (xx <= 50)) | (xx >= 62)))
        | ((yy == 43) & (xx >= 38) & (xx <= 67))
        | ((yy == 44) & (xx >= 40) & (xx <= 67))
        | ((yy == 45) & (xx >= 42) & (xx <= 67))
        | ((yy == 46) & (xx >= 43) & (xx <= 67))
        | ((yy == 47) & (xx >= 44) & (xx <= 67))
        | ((yy == 48) & (xx >= 41) & (xx <= 67))
        | ((yy == 49) & (xx >= 45) & (xx <= 67))
        | ((yy == 50) & (xx >= 45) & (xx <= 67))
        | ((yy == 51) & (xx >= 46) & (xx <= 67))
        | ((yy == 52) & (xx >= 46) & (xx <= 55))
        | ((yy == 53) & (xx >= 46) & (xx <= 54))
        | ((yy == 54) & (xx >= 46) & (xx <= 54))
        | ((yy == 55) & (xx >= 46) & (xx <= 53))
        | ((yy == 56) & (xx >= 45) & (xx <= 53))
        | ((yy == 57) & (xx >= 45) & (xx <= 54))
        | ((yy == 58) & (xx >= 45) & (xx <= 54))
        | ((yy == 59) & (xx >= 45) & (xx <= 54))
        | ((yy == 60) & (xx >= 46) & (xx <= 53))
        | ((yy == 61) & (xx >= 47) & (xx <= 52))
        | ((yy == 62) & (xx >= 47) & (xx <= 49))
    )
    arm_near = a & arm_near_zone

    # 5. Back seeds & upper chest leaf flares
    seeds_zone = (
        ((yy >= 26) & (yy <= 40) & (xx >= 14) & (xx <= 23))
        | ((yy >= 28) & (yy <= 39) & (xx >= 32) & (xx <= 50))
    )
    back_seeds = a & seeds_zone & (~arm_near) & (~lower_jaw)

    # 6. Foot L (grounded left lower leg & 3-toed foot)
    foot_L_zone = (yy >= 71) & (yy <= 84) & (xx >= 14) & (xx <= 27)
    foot_L = a & foot_L_zone

    # 7. Foot R (grounded right lower leg & 3-toed foot, strictly inside the ankle/foot outline)
    foot_R_zone = (
        ((yy == 74) & (xx >= 37) & (xx <= 48))
        | ((yy == 75) & (xx >= 38) & (xx <= 48))
        | ((yy == 76) & (xx >= 39) & (xx <= 48))
        | ((yy == 77) & (xx >= 41) & (xx <= 48))
        | ((yy == 78) & (xx >= 42) & (xx <= 48))
        | ((yy == 79) & (xx >= 42) & (xx <= 49))
        | ((yy == 80) & (xx >= 44) & (xx <= 49))
        | ((yy == 81) & (xx >= 45) & (xx <= 50))
        | ((yy == 82) & (xx >= 45) & (xx <= 50))
        | ((yy == 83) & (xx >= 45) & (xx <= 51))
        | ((yy == 84) & (xx >= 46) & (xx <= 52))
        | ((yy == 85) & (xx >= 46) & (xx <= 53))
        | ((yy == 86) & (xx >= 46) & (xx <= 52))
        | ((yy >= 87) & (yy <= 92) & (xx >= 43) & (xx <= 54))
    )
    foot_R = a & foot_R_zone

    # 8. Tail (giant Christmas-tree rocket leaf tail, including bottom frond tips behind right ankle at y=80..86)
    tail_zone = (
        ((yy >= 51) & (yy <= 54) & (xx >= 42))
        | ((yy >= 55) & (yy <= 59) & (xx >= 44))
        | ((yy == 60) & (xx >= 47))
        | ((yy == 61) & (xx >= 48))
        | ((yy == 62) & (xx >= 49))
        | ((yy >= 63) & (yy <= 65) & (xx >= 50))
        | ((yy >= 66) & (yy <= 72) & (xx >= 49))
        | ((yy >= 73) & (yy <= 76) & (xx >= 48))
        | ((yy >= 77) & (yy <= 79) & (xx >= 49))
        | ((yy >= 80) & (yy <= 86) & (xx >= 38))
    )
    tail = a & tail_zone & (~arm_near) & (~foot_R)

    used = lower_jaw | head_upper | arm_far | arm_near | back_seeds | foot_L | foot_R | tail
    torso = a & (~used)

    return {
        "tail": tail,
        "back_seeds": back_seeds,
        "arm_far": arm_far,
        "foot_L": foot_L,
        "torso": torso,
        "foot_R": foot_R,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
        "arm_near": arm_near,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 9 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Lower jaw (red lower jaw on 3/4 back profile)
    jaw_zone = (
        ((yy == 26) & (xx >= 51) & (xx <= 58))
        | ((yy == 27) & (xx >= 50) & (xx <= 58))
        | ((yy == 28) & (xx >= 50) & (xx <= 57))
        | ((yy == 29) & (xx >= 51) & (xx <= 55))
    )
    lower_jaw = a & jaw_zone

    # 2. Head upper (head crests, eye, upper snout)
    head_zone = (
        (yy <= 25)
        | ((yy == 26) & (xx >= 46) & (xx <= 50))
        | ((yy == 27) & (xx >= 46) & (xx <= 49))
        | ((yy == 28) & (xx >= 47) & (xx <= 49))
        | ((yy == 29) & (xx >= 48) & (xx <= 49))
    )
    head_upper = a & head_zone & (~lower_jaw)

    # 3. Arm far (left extended leaf-blade arm on back.png)
    arm_far_zone = (
        ((yy == 38) & (xx == 50))
        | ((yy >= 39) & (yy <= 40) & (xx >= 49) & (xx <= 51))
        | ((yy == 41) & (xx >= 49) & (xx <= 52))
        | ((yy >= 42) & (yy <= 43) & (xx >= 48) & (xx <= 52))
        | ((yy == 44) & (((xx >= 48) & (xx <= 51)) | ((xx >= 57) & (xx <= 60))))
        | ((yy == 45) & (((xx >= 48) & (xx <= 53)) | ((xx >= 56) & (xx <= 61))))
        | ((yy == 46) & (((xx >= 48) & (xx <= 53)) | ((xx >= 55) & (xx <= 62))))
        | ((yy == 47) & ((xx <= 38) | ((xx >= 49) & (xx <= 62))))
        | ((yy == 48) & ((xx <= 41) | ((xx >= 49) & (xx <= 62))))
        | ((yy == 49) & ((xx <= 44) | ((xx >= 49) & (xx <= 60))))
        | ((yy == 50) & ((xx <= 46) | ((xx >= 49) & (xx <= 58))))
        | ((yy == 51) & (xx >= 40) & (xx <= 58))
        | ((yy == 52) & (xx >= 42) & (xx <= 54))
        | ((yy == 53) & (xx >= 44) & (xx <= 56))
        | ((yy == 54) & (xx >= 46) & (xx <= 52))
        | ((yy == 55) & (xx >= 46) & (xx <= 50))
    )
    arm_far = a & arm_far_zone

    # 4. Arm near (right extended leaf-blade arm on back.png)
    arm_near_zone = (
        ((yy == 33) & (xx >= 92))
        | ((yy == 34) & (xx >= 90))
        | ((yy == 35) & (xx >= 89))
        | ((yy == 36) & (xx >= 88))
        | ((yy == 37) & (xx >= 86))
        | ((yy == 38) & (xx >= 86))
        | ((yy == 39) & (xx >= 83))
        | ((yy == 40) & (xx >= 82))
        | ((yy == 41) & (xx >= 81))
        | ((yy == 42) & (xx >= 80))
        | ((yy == 43) & (((xx >= 77) & (xx <= 78)) | (xx >= 80)))
        | ((yy >= 44) & (yy <= 51) & (xx >= 76))
    )
    arm_near = a & arm_near_zone

    # 5. Back seeds / shoulder leaf-blades on back.png
    seeds_zone = (
        ((yy >= 30) & (yy <= 37) & (xx >= 44) & (xx <= 59))
        | ((yy == 38) & (xx >= 52) & (xx <= 60))
        | ((yy == 39) & (xx >= 53) & (xx <= 60))
        | ((yy == 40) & (xx >= 54) & (xx <= 60))
        | ((yy >= 41) & (yy <= 43) & (xx >= 55) & (xx <= 59))
        | ((yy >= 27) & (yy <= 32) & (xx >= 75) & (xx <= 88))
        | ((yy >= 33) & (yy <= 37) & (xx >= 75) & (xx <= 85))
        | ((yy >= 38) & (yy <= 40) & (xx >= 74) & (xx <= 82))
        | ((yy >= 41) & (yy <= 43) & (xx >= 74) & (xx <= 79))
    )
    back_seeds = a & seeds_zone & (~arm_far) & (~arm_near)

    # 6. Tail (giant Christmas-tree rocket tail extending to the left on back.png;
    #    leave the 1px black border `#` along `x=48..55` on `torso` so `tail` sits cleanly behind `torso`!)
    tail_zone = (
        ((yy == 54) & (xx >= 52) & (xx <= 54))
        | ((yy >= 55) & (yy <= 57) & (xx <= 54))
        | ((yy >= 58) & (yy <= 59) & (xx <= 52))
        | ((yy == 60) & (xx <= 51))
        | ((yy >= 61) & (yy <= 62) & (xx <= 50))
        | ((yy == 63) & (xx <= 49))
        | ((yy >= 64) & (yy <= 66) & (xx <= 48))
        | ((yy >= 67) & (yy <= 69) & (xx <= 47))
        | ((yy == 70) & (xx <= 59))
        | ((yy == 71) & (xx <= 60))
        | ((yy >= 72) & (yy <= 86) & (xx <= 61))
    )
    tail = a & tail_zone & (~arm_far)

    # 7. Foot R (grounded right lower leg & 3-toed foot on back.png)
    foot_R_zone = (yy >= 71) & (yy <= 83) & (xx >= 72) & (xx <= 90)
    foot_R = a & foot_R_zone

    empty = np.zeros_like(a)
    used = lower_jaw | head_upper | arm_far | arm_near | back_seeds | tail | foot_R
    torso = a & (~used)

    return {
        "tail": tail,
        "back_seeds": back_seeds,
        "arm_far": arm_far,
        "foot_L": empty,
        "torso": torso,
        "foot_R": foot_R,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
        "arm_near": arm_near,
    }


def build_foot_stem_layer(
    canvas: np.ndarray,
    foot_mask: np.ndarray,
    torso_mask: np.ndarray,
    stem_px: int = 4,
) -> np.ndarray:
    lyr = np.zeros_like(canvas)
    lyr[foot_mask] = canvas[foot_mask]
    f_ys, f_xs = np.where(foot_mask)
    if len(f_ys) == 0:
        return lyr
    for x_col in np.unique(f_xs):
        col_ys = f_ys[f_xs == x_col]
        cy_min = int(col_ys.min())
        fill_px = canvas[cy_min, x_col]
        for dy_up in range(1, stem_px + 1):
            py = cy_min - dy_up
            if py >= 0 and torso_mask[py, x_col]:
                if not is_dark_outline(canvas[py:py + 1, x_col:x_col + 1])[0, 0]:
                    lyr[py, x_col] = canvas[py, x_col]
                else:
                    lyr[py, x_col] = fill_px
    return lyr


def build_rig_layers(
    base_96: np.ndarray,
    is_back: bool,
    flip_x: bool,
    cell_w: int,
    cell_h: int,
    foot_y: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
    """
    Segments native `front.png` or `back.png`, optionally flips horizontally for SE/E/NW,
    places onto the target `(cell_h, cell_w)` canvas with feet aligned to `foot_y`,
    and builds cushioned RGBA layers + true anatomical joint pivots.
    """
    raw_masks = segment_back_native(base_96) if is_back else segment_front_native(base_96)
    src_96 = base_96[:, ::-1, :].copy() if flip_x else base_96.copy()
    masks_96 = {k: (m[:, ::-1].copy() if flip_x else m.copy()) for k, m in raw_masks.items()}

    a96 = src_96[:, :, 3] >= 128
    ys, xs = np.where(a96)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x0 = (cell_w - sw) // 2
    dst_y0 = (foot_y + 2 - sh) if is_back else (foot_y - sh)

    def to_canvas_pt(nx: float, ny: float) -> Tuple[float, float]:
        fx = (95.0 - nx) if flip_x else nx
        return (float(dst_x0) + (fx - float(x0)), float(dst_y0) + (ny - float(y0)))

    canvas = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
    canvas[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = src_96[y0:y1, x0:x1]

    c_masks: Dict[str, np.ndarray] = {}
    for k, m96 in masks_96.items():
        cm = np.zeros((cell_h, cell_w), dtype=bool)
        cm[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = m96[y0:y1, x0:x1]
        c_masks[k] = cm

    green_fill_m = is_green_body_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order:
        # 1. tail -> 2. back_seeds -> 3. arm_far -> 4. foot_L -> 5. torso -> 6. foot_R -> 7. lower_jaw -> 8. head_upper -> 9. arm_near
        fg_over_tail = c_masks["torso"] | c_masks["foot_R"] | c_masks["arm_near"]
        u_tail = ndi.binary_erosion(c_masks["tail"] | fg_over_tail, iterations=1)
        layers["tail"] = build_part_layer(
            canvas,
            c_masks["tail"],
            u_tail & fg_over_tail,
            cushion_px=7,
            allowed_fill_mask=green_fill_m,
        )

        fg_over_seeds = c_masks["torso"] | c_masks["arm_near"] | c_masks["lower_jaw"] | c_masks["head_upper"]
        u_seeds = ndi.binary_erosion(c_masks["back_seeds"] | fg_over_seeds, iterations=1)
        layers["back_seeds"] = build_part_layer(
            canvas,
            c_masks["back_seeds"],
            u_seeds & fg_over_seeds,
            cushion_px=5,
            allowed_fill_mask=green_fill_m,
        )

        fg_over_afar = c_masks["torso"] | c_masks["back_seeds"]
        u_afar = ndi.binary_erosion(c_masks["arm_far"] | fg_over_afar, iterations=1)
        layers["arm_far"] = build_part_layer(
            canvas,
            c_masks["arm_far"],
            u_afar & fg_over_afar,
            cushion_px=4,
            allowed_fill_mask=green_fill_m,
        )

        layers["foot_L"] = build_foot_stem_layer(canvas, c_masks["foot_L"], c_masks["torso"], stem_px=4)
        layers["foot_R"] = build_foot_stem_layer(canvas, c_masks["foot_R"], c_masks["torso"], stem_px=4)

        # IMPORTANT: `torso` cushion ONLY expands under foreground layers drawn ON TOP OF `torso`
        # (`foot_R`, `lower_jaw`, `head_upper`, `arm_near`), NEVER over `tail` or `back_seeds`!
        fg_over_torso = c_masks["foot_R"] | c_masks["lower_jaw"] | c_masks["head_upper"] | c_masks["arm_near"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=green_fill_m,
        )

        # (2026-11-04) No rectangular mouth-interior pocket any more; the jaw stays closed in place and
        # mouth_open.open_mouth carves a small opening along MOUTH_LINE_FRONT on the composed frame.
        layers["torso"] = torso_lyr

        layers["lower_jaw"] = build_part_layer(canvas, c_masks["lower_jaw"], np.zeros_like(c_masks["lower_jaw"]), cushion_px=0)
        layers["head_upper"] = build_part_layer(canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0)
        layers["arm_near"] = build_part_layer(canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0)

        pivots = {
            "arm_near": to_canvas_pt(39.0, 42.0),  # Right shoulder socket
            "arm_far": to_canvas_pt(17.0, 45.0),   # Left shoulder socket
            "tail": to_canvas_pt(49.0, 68.0),      # Hip tail socket
            # closed mouth line (corner -> lip tip), mirrored with the view by to_canvas_pt
            "mouth": [to_canvas_pt(x, y) for x, y in MOUTH_LINE_FRONT],
        }

    else:
        # Back view draw order:
        # 1. tail -> 2. arm_far -> 3. back_seeds -> 4. foot_R -> 5. torso -> 6. lower_jaw -> 7. head_upper -> 8. arm_near
        layers["foot_L"] = np.zeros_like(canvas)

        fg_over_tail = c_masks["torso"] | c_masks["arm_far"]
        u_tail = ndi.binary_erosion(c_masks["tail"] | fg_over_tail, iterations=1)
        layers["tail"] = build_part_layer(
            canvas,
            c_masks["tail"],
            u_tail & fg_over_tail,
            cushion_px=7,
            allowed_fill_mask=green_fill_m,
        )

        fg_over_afar = c_masks["back_seeds"] | c_masks["torso"]
        u_afar = ndi.binary_erosion(c_masks["arm_far"] | fg_over_afar, iterations=1)
        layers["arm_far"] = build_part_layer(
            canvas,
            c_masks["arm_far"],
            u_afar & fg_over_afar,
            cushion_px=5,
            allowed_fill_mask=green_fill_m,
        )

        fg_over_seeds = c_masks["torso"] | c_masks["arm_near"] | c_masks["head_upper"]
        u_seeds = ndi.binary_erosion(c_masks["back_seeds"] | fg_over_seeds, iterations=1)
        layers["back_seeds"] = build_part_layer(
            canvas,
            c_masks["back_seeds"],
            u_seeds & fg_over_seeds,
            cushion_px=5,
            allowed_fill_mask=green_fill_m,
        )

        layers["foot_R"] = build_foot_stem_layer(canvas, c_masks["foot_R"], c_masks["torso"], stem_px=4)

        fg_over_torso = c_masks["lower_jaw"] | c_masks["head_upper"] | c_masks["arm_near"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=green_fill_m,
        )
        # (2026-11-04) Back view: no mouth pocket, no opening (mouth not visible from behind).
        layers["torso"] = torso_lyr

        layers["lower_jaw"] = build_part_layer(canvas, c_masks["lower_jaw"], np.zeros_like(c_masks["lower_jaw"]), cushion_px=0)
        layers["head_upper"] = build_part_layer(canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0)
        layers["arm_near"] = build_part_layer(canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0)

        pivots = {
            "arm_near": to_canvas_pt(77.0, 44.0),  # Right shoulder socket on back.png
            "arm_far": to_canvas_pt(59.0, 46.0),   # Left shoulder socket on back.png
            "tail": to_canvas_pt(52.0, 65.0),      # Hip tail socket on back.png
        }

    return layers, pivots


def foot_step_offset(phase: float, facing: float, lift_px: int = 2, stride_px: int = 2) -> Tuple[int, int]:
    p = phase % (2.0 * math.pi)
    s = math.sin(p)
    c = math.cos(p)
    dx = int(round(stride_px * c * facing))
    dy = int(round(-lift_px * max(0.0, s)))
    return dx, dy


def add_hurt_sparks(canvas: np.ndarray, frame_idx: int) -> np.ndarray:
    if frame_idx not in (2, 3, 4, 5, 6):
        return canvas
    out = canvas.copy()
    a = out[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return out
    h, w = out.shape[:2]
    top_y = int(ys.min())
    mid_x = int(np.median(xs))
    drift = frame_idx - 2
    pts = [
        (mid_x - 16 - drift, max(2, top_y + 12 - drift // 2)),
        (mid_x + 16 + drift, max(2, top_y + 14 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (230, 222, 65, 255)
    return out


def synthesize_sceptile_mega_frame(
    layers: Dict[str, np.ndarray],
    pivots: Dict[str, Tuple[float, float]],
    canvas_shape: Tuple[int, int, int],
    action: str,
    frame_idx: int,
    facing: float,
    is_back: bool,
    palette_rgba: np.ndarray,
) -> np.ndarray:
    t = frame_idx / float(N_FRAMES)
    phase = 2.0 * math.pi * t
    sin1 = math.sin(phase)
    cos1 = math.cos(phase)

    torso_dx, torso_dy = 0, 0
    head_dx, head_dy = 0, 0
    jaw_dx, jaw_dy = 0, 0
    jaw_open = 0
    seeds_dx, seeds_dy = 0, 0
    anear_dx, anear_dy = 0, 0
    afar_dx, afar_dy = 0, 0
    tail_dx, tail_dy = 0, 0
    foot_L_dx, foot_L_dy = 0, 0
    foot_R_dx, foot_R_dy = 0, 0

    anear_ang = 0.0
    afar_ang = 0.0
    tail_ang = 0.0

    if action == "idle":
        # Agile raptor breathing crouch + roaring/barking lower jaw + circular leaf-blade arm readiness + tail wave:
        # 1. Both feet (`foot_L`, `foot_R`) stay 100% planted on the ground `(0, 0)`.
        # 2. Body bobs rhythmically (`-1..+1px`) while head leans slightly forward during the breath/roar.
        # 3. `lower_jaw` opens `+1px` during the inhalation/bark peak, revealing the dark-red mouth pocket.
        # 4. Both leaf-blade arms (`arm_near`, `arm_far`) trace a crisp circular/orbital stance trajectory.
        # 5. The giant Christmas-tree rocket tail (`tail`) sways and pivots (`±2.5°`) from the hip socket!
        body_dy = int(round(1.2 * math.sin(phase - 0.2)))
        body_dx = int(round(0.6 * max(0.0, math.sin(phase - 0.2)) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        jaw_open = 1 if math.sin(phase - 0.1) > 0.15 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; opening is carved by mouth_open

        seeds_dx = body_dx
        seeds_dy = body_dy + int(round(0.8 * math.sin(phase - 0.4)))

        anear_dx = body_dx + int(round(1.2 * math.cos(phase - 0.5) * facing))
        anear_dy = body_dy + int(round(1.4 * math.sin(phase - 0.5)))

        afar_dx = body_dx + int(round(1.2 * math.cos(phase - 0.8) * facing))
        afar_dy = body_dy + int(round(1.4 * math.sin(phase - 0.8)))

        tail_dx = body_dx
        tail_dy = int(round(1.6 * math.sin(phase - 1.1)))
        tail_ang = 2.2 * math.sin(phase - 1.1) * facing

    elif action == "walk":
        # Agile forward-leaning bipedal raptor dash gait:
        # Alternating left/right foot stride (`lift_px=2, stride_px=2`),
        # 2-beat body bounce, counter-swinging dual leaf-blade arms (`±3.5°` shoulder rotation),
        # menacing jaw bark (`+1px`) on each stride, and counterbalance rocket-tail wave!
        bounce = int(round(-1.6 * abs(sin1)))
        sway_x = int(round(1.2 * sin1 * facing))

        torso_dx, torso_dy = sway_x, bounce
        head_dx = sway_x + int(round(0.8 * cos1 * facing))
        head_dy = bounce + int(round(0.8 * math.sin(2.0 * phase - 0.3)))

        jaw_open = 1 if math.sin(2.0 * phase) > 0.15 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; opening is carved by mouth_open

        seeds_dx = sway_x
        seeds_dy = bounce + int(round(1.0 * math.sin(2.0 * phase - 0.5)))

        anear_dx = sway_x + int(round(1.8 * sin1 * facing))
        anear_dy = bounce + int(round(1.5 * cos1))
        anear_ang = 3.5 * sin1 * (-facing)

        afar_dx = sway_x + int(round(-1.8 * sin1 * facing))
        afar_dy = bounce + int(round(-1.5 * cos1))
        afar_ang = -3.5 * sin1 * facing

        tail_dx = sway_x
        tail_dy = bounce + int(round(1.8 * math.sin(2.0 * phase - 1.0)))
        tail_ang = 3.0 * math.sin(2.0 * phase - 1.0) * facing

        if not is_back:
            foot_L_dx, foot_L_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        foot_R_dx, foot_R_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Mega Sceptile "Spinning Leaf Blade X-Slash + Ferocious Roar + Rocket Tail Drill Thrust"!
        # Frames 0..3 (Wind-up & Big Roar):
        #   - Coils low and back (`-3px`), flares dorsal seeds & chest X-leaves (`-1px`),
        #   - Opens `lower_jaw` wide (`+2px` ferocious roar!),
        #   - Cocks both leaf-blade arms high at the shoulder sockets (`-7°` rotation).
        # Frames 4..9 (Spinning Leaf-Blade Slash & Rocket Tail Drill):
        #   - Explosive forward lunge (`+4px` body, `+5px` leaf-blade arms, lead foot steps `+3px` forward),
        #   - High-speed spinning Leaf-Blade slash (`+8°` -> `-6°` shoulder pivot rotation + circular slash arc),
        #   - Giant Christmas-tree tail executes a rapid rocket-drill spin oscillation (`±5°` around hip socket + thrust),
        #   - Jaws snap shut on impact (`0px` at frames 6..7) and bark (`+1px` at frames 8..9)!
        # Frames 10..15: Follow-through recovery to stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.65 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        body_dx = int(round(4.0 * k * facing))
        if frame_idx in (1, 2, 3):
            body_dy = 1
        elif frame_idx in (4, 5):
            body_dy = -1
        elif frame_idx in (6, 7, 8):
            body_dy = 1
        else:
            body_dy = 0

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = int(round(4.5 * k * facing))
        head_dy = body_dy

        # Ferocious +2px wide roar on wind-up & early lunge (frames 1..5), snap shut on impact (6..7), follow-through bark (8..9)
        if frame_idx in (2, 3, 4):
            jaw_open = 2
        elif frame_idx in (1, 5, 8, 9):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; opening is carved by mouth_open

        # Dorsal seeds & chest X-leaves flare upward during wind-up and strike peak
        seeds_dx = body_dx
        seeds_dy = body_dy - (1 if frame_idx in (2, 3, 5, 6, 7) else 0)

        # Dual leaf-blade arms: cock back & rotate at shoulder on wind-up, then slash forward with a rapid spin arc!
        anear_dx = int(round(4.8 * k * facing))
        anear_dy = body_dy + (int(round(-1.5 * k)) if frame_idx in (4, 5) else int(round(1.8 * max(0.0, k))))
        afar_dx = int(round(5.0 * k * facing))
        afar_dy = body_dy + int(round(1.6 * k))

        spin_table_near = [0.0, -4.0, -6.5, -8.0, 5.0, 8.5, -6.0, 7.5, -4.5, 3.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        spin_table_far = [0.0, 4.0, 6.5, 8.0, -6.0, -8.5, 6.5, -7.5, 4.5, -3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        anear_ang = spin_table_near[frame_idx] * (-facing)
        afar_ang = spin_table_far[frame_idx] * facing

        # Giant rocket leaf tail: drills/spins rapidly around hip socket during strike frames (4..9) while thrusting!
        tail_drill_ang = [0.0, -2.0, -3.5, -4.5, 4.5, -5.0, 5.0, -4.5, 4.0, -2.5, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        tail_drill_dy = [0, 1, 1, 2, -2, 2, -2, 2, -1, 1, 0, 0, 0, 0, 0, 0]
        tail_dx = body_dx
        tail_dy = body_dy + tail_drill_dy[frame_idx]
        tail_ang = tail_drill_ang[frame_idx] * facing

        # Lead foot steps into the lunge
        lead_x = int(round(3.0 * max(0.0, k) * facing)) if k >= 0 else int(round(1.5 * k * facing))
        rear_x = int(round(2.0 * k * facing))
        if not is_back:
            foot_L_dx, foot_L_dy = lead_x, 0
            foot_R_dx, foot_R_dy = rear_x, 0
        else:
            foot_R_dx, foot_R_dy = lead_x, 0

    elif action == "hurt":
        # Knockback + hit-shake + pained roar (`lower_jaw` open +1..2px) + leaf-blade arms flung back + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-3.5 * env * facing)) + jit
        body_dy = int(round(-1.0 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-1.5 * env))

        jaw_open = 1 if 1 <= frame_idx <= 7 else 0  # pained growl: small 1 px opening
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; opening is carved by mouth_open

        seeds_dx = body_dx
        seeds_dy = body_dy - (1 if 2 <= frame_idx <= 6 else 0)

        anear_dx = body_dx + int(round(-1.2 * env * facing))
        anear_dy = body_dy + int(round(-1.5 * env))
        anear_ang = -5.0 * env * (-facing)

        afar_dx = body_dx + int(round(-1.2 * env * facing))
        afar_dy = body_dy + int(round(-1.5 * env))
        afar_ang = 5.0 * env * facing

        tail_dx = body_dx
        tail_dy = int(round(1.2 * env))
        tail_ang = -3.0 * env * facing

        slide_x = int(round(body_dx * 0.65))
        foot_L_dx, foot_R_dx = slide_x, slide_x

    tail_rot = rotate_layer_nearest(layers["tail"], tail_ang, pivots["tail"])
    afar_rot = rotate_layer_nearest(layers["arm_far"], afar_ang, pivots["arm_far"])
    anear_rot = rotate_layer_nearest(layers["arm_near"], anear_ang, pivots["arm_near"])

    if not is_back:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(layers["back_seeds"], seeds_dx, seeds_dy),
            shift_layer_int(afar_rot, afar_dx, afar_dy),
            shift_layer_int(layers["foot_L"], foot_L_dx, foot_L_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["foot_R"], foot_R_dx, foot_R_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(anear_rot, anear_dx, anear_dy),
        ]
    else:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(afar_rot, afar_dx, afar_dy),
            shift_layer_int(layers["back_seeds"], seeds_dx, seeds_dy),
            shift_layer_int(layers["foot_R"], foot_R_dx, foot_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(anear_rot, anear_dx, anear_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    mouth_line = pivots.get("mouth") if not is_back else None
    if mouth_line and jaw_open > 0:
        line = mouth_open.transform_points(mouth_line, dx=head_dx, dy=head_dy)
        out = mouth_open.open_mouth(out, line, amount=min(2, jaw_open), deep=MOUTH_DEEP,
                                    inner=MOUTH_INNER, peak=0.65)
    if action == "hurt":
        out = add_hurt_sparks(out, frame_idx)
    return out


def save_transparent_gif(frames_rgba: List[np.ndarray], out_path: Path, duration_ms: int = 75) -> None:
    pil_frames: List[Image.Image] = []
    for arr in frames_rgba:
        a = arr[:, :, 3] >= 128
        rgb = arr[:, :, :3]
        uniq = np.unique(rgb[a], axis=0) if np.any(a) else np.zeros((0, 3), dtype=np.uint8)
        pal_flat = [0, 0, 0]
        idx_map = np.zeros(arr.shape[:2], dtype=np.uint8)
        for c_i, col in enumerate(uniq[:254], start=1):
            pal_flat.extend([int(col[0]), int(col[1]), int(col[2])])
            m_c = a & (rgb[:, :, 0] == col[0]) & (rgb[:, :, 1] == col[1]) & (rgb[:, :, 2] == col[2])
            idx_map[m_c] = c_i
        while len(pal_flat) < 768:
            pal_flat.append(0)
        p_img = Image.fromarray(idx_map, mode="P")
        p_img.putpalette(pal_flat)
        p_img.info["transparency"] = 0
        pil_frames.append(p_img)
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        disposal=2,
        transparency=0,
        optimize=False,
    )


def build_preview_strip(sheets_by_action: Dict[str, np.ndarray], cell_w: int, cell_h: int) -> Image.Image:
    bg = np.full((8 * cell_h, 16 * cell_w, 4), (15, 20, 31, 255), dtype=np.uint8)
    sample_cols = [0, 4, 8, 12]
    for a_idx, act in enumerate(ACTIONS):
        sheet = sheets_by_action[act]
        for s_i, f_col in enumerate(sample_cols):
            dst_col = a_idx * 4 + s_i
            x_src = f_col * cell_w
            x_dst = dst_col * cell_w
            crop = sheet[:, x_src:x_src + cell_w, :]
            m = crop[:, :, 3] >= 128
            bg[:, x_dst:x_dst + cell_w, :][m] = crop[m]
    return Image.fromarray(bg, "RGBA")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    front_raw = np.array(Image.open(MEGA_DIR / "front.png").convert("RGBA"))
    back_raw = np.array(Image.open(MEGA_DIR / "back.png").convert("RGBA"))

    palette_rgba = extract_palette(front_raw, back_raw)

    # Tight bounds of front (96x89) and back (93x78)
    max_w, max_h = 96, 89
    cell_w = 120
    cell_h = 106
    foot_y = 98

    dir_rigs: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]], float, bool]] = {}
    for d_name in DIRS:
        use_back = d_name in ("N", "NE", "NW")
        src_96 = back_raw if use_back else front_raw
        if not use_back:
            flip_x = d_name in ("SE", "E")
            facing = 1.0 if flip_x else -1.0
        else:
            flip_x = d_name in ("NW",)
            facing = -1.0 if flip_x else 1.0

        layers, pivots = build_rig_layers(
            src_96,
            is_back=use_back,
            flip_x=flip_x,
            cell_w=cell_w,
            cell_h=cell_h,
            foot_y=foot_y,
        )
        dir_rigs[d_name] = (layers, pivots, facing, use_back)

    sheets_by_action: Dict[str, np.ndarray] = {}
    master_rows: List[np.ndarray] = []

    for act in ACTIONS:
        act_sheet = np.zeros((8 * cell_h, N_FRAMES * cell_w, 4), dtype=np.uint8)
        for d_idx, d_name in enumerate(DIRS):
            layers, pivots, facing, use_back = dir_rigs[d_name]
            for f_idx in range(N_FRAMES):
                frame_rgba = synthesize_sceptile_mega_frame(
                    layers,
                    pivots,
                    (cell_h, cell_w, 4),
                    act,
                    f_idx,
                    facing,
                    use_back,
                    palette_rgba,
                )
                y0 = d_idx * cell_h
                x0 = f_idx * cell_w
                act_sheet[y0:y0 + cell_h, x0:x0 + cell_w] = frame_rgba

        sheets_by_action[act] = act_sheet
        act_img = Image.fromarray(act_sheet, "RGBA")
        act_img.save(OUT_DIR / f"{act}.png", optimize=False)
        act_img.save(OUT_DIR / f"{act}_8dir.png", optimize=False)
        Image.fromarray(act_sheet[0:cell_h, :, :], "RGBA").save(OUT_DIR / f"{act}_S_strip.png", optimize=False)
        master_rows.append(act_sheet)

    master_arr = np.concatenate(master_rows, axis=0)
    Image.fromarray(master_arr, "RGBA").save(OUT_DIR / "sheet.png", optimize=False)
    Image.fromarray(master_arr, "RGBA").save(OUT_DIR / "all_8dir_actions.png", optimize=False)

    preview_img = build_preview_strip(sheets_by_action, cell_w, cell_h)
    preview_img.save(OUT_DIR / "preview.png", optimize=False)

    # Export clean 16-frame animated GIFs (`front.gif`, `front-reverse.gif`, `back.gif`, `back-reverse.gif`)
    idle_sheet = sheets_by_action["idle"]
    front_frames = [idle_sheet[0:cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    front_rev_frames = [f[:, ::-1, :].copy() for f in front_frames]
    back_frames = [idle_sheet[4 * cell_h:5 * cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    back_rev_frames = [f[:, ::-1, :].copy() for f in back_frames]

    save_transparent_gif(front_frames, MEGA_DIR / "front.gif", duration_ms=75)
    save_transparent_gif(front_rev_frames, MEGA_DIR / "front-reverse.gif", duration_ms=75)
    save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
    save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)

    meta = {
        "generator": "forge_sceptile_mega_actions",
        "version": VERSION,
        "key": "0254_sceptile-mega",
        "dex": 254,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "foot_y": foot_y,
        "sprite_h": max_h,
        "frames": N_FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "palette_colors": int(len(palette_rgba)),
        "off_palette_colors": 0,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Sync to pokemon_arena_2d if present
    if SYNC_MEGA_DIR.exists():
        SYNC_OUT_DIR.mkdir(parents=True, exist_ok=True)
        for f_path in OUT_DIR.iterdir():
            if f_path.is_file():
                shutil.copy2(f_path, SYNC_OUT_DIR / f_path.name)
        for gif_name in ("front.gif", "front-reverse.gif", "back.gif", "back-reverse.gif"):
            shutil.copy2(MEGA_DIR / gif_name, SYNC_MEGA_DIR / gif_name)

    print(f"Built 0254_sceptile-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
