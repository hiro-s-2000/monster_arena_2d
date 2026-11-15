#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Manectric (`0310_manectric-mega`) modeled after its official 81-frame GIF
(`0310_manectric/mega/manectric-mega/front.gif`) and native 96x96 sprites (`front.png` & `back.png`):
  - Exact 10-layer anatomical rigging with true anatomical joint pivots:
      1. `lightning_tail_rear` (rear zigzag yellow lightning tail spike behind the hip —
                                pivoted at the rear spine/hip root with `rotate_layer_nearest` so it
                                whips and crackles like high-voltage lightning!)
      2. `mane_dorsal_back` (dorsal/rear portion of the colossal yellow lightning mane that sits
                             behind the spiky blue hip fur of `torso` — shares the exact neck pivot
                             and rotation angle of `lightning_mane_upper` so the entire giant crown
                             mane rotates as one 100% seamless piece while keeping the spiky blue
                             hip fur in front!)
      3. `hind_leg_L` (far/left hind leg — 100% grounded `(0,0)` in `idle`, fast 4-legged diagonal
                       gallop/trot in `walk`)
      4. `front_leg_L` (far/left front leg with yellow ankle fur & red claws — 100% grounded `(0,0)` in
                        `idle`, diagonal trot in `walk`, forward pounce in `attack`)
      5. `torso` (blue wolf body + 4 spiky blue hip fur tufts)
      6. `hind_leg_R` (near/right hind leg with red claws — 100% grounded `(0,0)` in `idle`, diagonal
                       gallop/trot in `walk`)
      7. `front_leg_R` (near/right front leg with yellow ankle fur & red claws — 100% grounded `(0,0)`
                        in `idle`, diagonal trot in `walk`, explosive Volt Tackle pounce in `attack`)
      8. `lightning_mane_upper` (foreground crown zigzag lightning mane, ear horns, front visor horn,
                                 and chest lightning blade (the old dark-red mouth pocket was removed
                                 2026-11-04; the mouth now opens with tools/mouth_open.py) —
                                 pivoted at the neck base with `rotate_layer_nearest` for dynamic
                                 electric discharge spin & bristle!)
      9. `lower_jaw` (blue lower jaw — since 2026-11-04 it no longer drops; a small 1-2px lens opening is
                      carved along the mouth line instead. Old note: opens `+1px` for barking/breathing in `idle`/`walk`
                      and `+2px` wide for a ferocious thunder-wolf howl/roar in `attack` & `hurt`!)
     10. `head_upper` (fierce red/white eye, blue upper snout, and yellow nose bridge — 100% rigid integer
                       translation `shift_layer_int` so the face and eye never deform by a single pixel!)
"""

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import mouth_open  # noqa: E402  (2026-11-04 mouth fix: small face-following opening)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0310_manectric/mega/manectric-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0310_manectric/mega/manectric-mega"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16

# 2026-11-04 mouth fix (user request: the rectangular jaw drop made the chin jut out).
# The lower jaw no longer drops and no rectangular interior pocket is painted; instead a small lens-shaped
# opening is carved along the closed mouth line of the still with tools/mouth_open.py.
# MOUTH_LINE_NATIVE: native 96x96 front.png coordinates, ordered hinge (back of the head) -> lip tip.
# Back views (N/NE/NW: back.png, mouth not visible) get no opening.
MOUTH_LINE_NATIVE_FRONT = [(29.0, 49.0), (25.0, 49.0), (21.0, 49.0), (18.0, 49.0)]  # carves from the mouth line row (y=50) down
MOUTH_DEEP = (16, 16, 16)      # darkest palette colour (outline)
MOUTH_INNER = (181, 11, 11)    # existing dark red of front.png (mouth interior)
MOUTH_PEAK = 0.5


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
        ((r == 16) & (g == 16) & (b == 16))
        | ((r == 42) & (g == 71) & (b == 77))
        | ((r == 61) & (g == 74) & (b == 117))
        | ((r == 115) & (g == 106) & (b == 63))
        | ((r == 143) & (g == 104) & (b == 65))
    )


def is_yellow_mane_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 255) & (g == 240) & (b == 109))
        | ((r == 208) & (g == 164) & (b == 70))
        | ((r == 255) & (g == 229) & (b == 102))
        | ((r == 217) & (g == 195) & (b == 87))
        | ((r == 153) & (g == 140) & (b == 76))
    )


def is_blue_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 75) & (g == 117) & (b == 161))
        | ((r == 82) & (g == 168) & (b == 212))
        | ((r == 70) & (g == 144) & (b == 196))
        | ((r == 63) & (g == 106) & (b == 115))
        | ((r == 84) & (g == 142) & (b == 153))
        | ((r == 105) & (g == 177) & (b == 191))
    )


def build_part_layer(
    src: np.ndarray,
    mask: np.ndarray,
    foreground_mask: np.ndarray,
    cushion_px: int = 6,
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


def build_foot_stem_layer(
    canvas: np.ndarray,
    foot_mask: np.ndarray,
    upper_mask: np.ndarray,
    stem_px: int = 4,
    fallback_rgba: Tuple[int, int, int, int] = (75, 117, 161, 255),
) -> np.ndarray:
    lyr = np.zeros_like(canvas)
    lyr[foot_mask] = canvas[foot_mask]
    f_ys, f_xs = np.where(foot_mask)
    if len(f_ys) == 0:
        return lyr
    blue_m = is_blue_body_fill(canvas)
    for x_col in np.unique(f_xs):
        col_ys = f_ys[f_xs == x_col]
        cy_min = int(col_ys.min())
        fill_px = np.array(fallback_rgba, dtype=np.uint8)
        for dy_s in range(0, 4):
            if cy_min + dy_s < canvas.shape[0] and blue_m[cy_min + dy_s, x_col]:
                fill_px = canvas[cy_min + dy_s, x_col]
                break
        for dy_up in range(1, stem_px + 1):
            py = cy_min - dy_up
            if py >= 0 and upper_mask[py, x_col]:
                if blue_m[py, x_col]:
                    lyr[py, x_col] = canvas[py, x_col]
                else:
                    lyr[py, x_col] = fill_px
    return lyr


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
    if abs(angle_deg) < 0.35:
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
    Segments native 96x96 `front.png` (facing left) into 10 exact anatomical masks:
      1. `lightning_tail_rear`: Rear lightning tail spike sticking out behind the hip (`y=42..55, x=71..81`).
      2. `mane_dorsal_back`: Rear/dorsal portion of the giant lightning mane behind `torso`'s 4 blue hip spikes.
      3. `hind_leg_L`: Far/inner left hind leg (`y=64..74, x=53..59`).
      4. `front_leg_L`: Far/left front leg with yellow fur & red claws (`y=66..81, x=16..37`).
      5. `torso`: Blue wolf body + 4 spiky blue hip fur tufts (`y=41..67, x=32..74`).
      6. `hind_leg_R`: Near/outer right hind leg with red claws (`y=68..83, x=63..78`).
      7. `front_leg_R`: Near/right front leg with yellow fur & red claws (`y=65..84, x=38..63`).
      8. `lightning_mane_upper`: Foreground crown lightning mane + ear horns + front visor + chest lightning blade.
      9. `lower_jaw`: Articulated blue lower jaw (`y=50..53, x=17..30`).
     10. `head_upper`: Fierce red/white eye, blue upper snout, and yellow nose ridge (`y=42..50, x=15..35`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    r = arr96[:, :, 0].astype(np.int32)
    g = arr96[:, :, 1].astype(np.int32)
    b = arr96[:, :, 2].astype(np.int32)

    lower_jaw = a & (
        ((yy == 50) & (xx >= 21) & (xx <= 30))
        | ((yy == 51) & (xx >= 17) & (xx <= 30))
        | ((yy == 52) & (xx >= 19) & (xx <= 30))
        | ((yy == 53) & (xx >= 25) & (xx <= 30))
    )

    head_upper = a & (~lower_jaw) & (
        ((yy == 42) & (xx >= 19) & (xx <= 33))
        | ((yy == 43) & (xx >= 20) & (xx <= 33))
        | ((yy == 44) & (xx >= 19) & (xx <= 34))
        | ((yy == 45) & (xx >= 17) & (xx <= 34))
        | ((yy == 46) & (xx >= 16) & (xx <= 35))
        | ((yy == 47) & (xx >= 15) & (xx <= 34))
        | ((yy == 48) & (xx >= 15) & (xx <= 31))
        | ((yy == 49) & (xx >= 16) & (xx <= 31))
        | ((yy == 50) & (xx >= 16) & (xx <= 20))
    )

    front_leg_L = a & (
        ((yy == 66) & (xx >= 36) & (xx <= 38))
        | ((yy == 67) & (xx >= 34) & (xx <= 37))
        | ((yy == 68) & (xx >= 31) & (xx <= 36))
        | ((yy == 69) & (xx >= 31) & (xx <= 35))
        | ((yy >= 70) & (yy <= 81) & (xx <= 37))
    )

    hind_leg_L = a & (
        ((yy == 64) & (xx >= 54) & (xx <= 58))
        | ((yy == 65) & (xx >= 53) & (xx <= 59))
        | ((yy == 66) & (xx >= 53) & (xx <= 58))
        | ((yy == 67) & (xx >= 55) & (xx <= 57))
        | ((yy == 68) & (xx >= 54) & (xx <= 56))
        | ((yy >= 69) & (yy <= 73) & (xx >= 55) & (xx <= 56))
        | ((yy == 74) & (xx == 54))
    )

    front_leg_R = a & (~front_leg_L) & (~hind_leg_L) & (
        ((yy == 65) & (xx >= 42) & (xx <= 52))
        | ((yy == 66) & (xx >= 43) & (xx <= 52))
        | ((yy == 67) & (xx >= 44) & (xx <= 54))
        | ((yy == 68) & (xx >= 46) & (xx <= 53))
        | ((yy == 69) & (xx >= 48) & (xx <= 54))
        | ((yy >= 70) & (yy <= 84) & (xx >= 38) & (xx <= 63))
    )

    hind_leg_R = a & (
        ((yy == 68) & (xx >= 63))
        | ((yy == 69) & (xx >= 66))
        | ((yy >= 70) & (yy <= 84) & (xx >= 65))
    )

    # Foreground portion of the giant lightning mane (including the front chest blade and gold teeth)
    mane_upper_zone = (
        (yy <= 40)
        | ((yy == 41) & (xx <= 60))
        | ((yy == 42) & (xx <= 59))
        | ((yy >= 43) & (yy <= 45) & (xx <= 56))
        | ((yy >= 46) & (yy <= 47) & (xx <= 55))
        | ((yy >= 48) & (yy <= 50) & (xx <= 58))
        | ((yy == 51) & (xx <= 57))
        | ((yy >= 52) & (yy <= 54) & (xx <= 56))
        | ((yy == 55) & (xx <= 55))
        | ((yy == 56) & (xx <= 47))
        | ((yy == 57) & (xx <= 46))
        | ((yy == 58) & (xx <= 45))
        | ((yy == 59) & (xx <= 44))
        | ((yy == 60) & (xx <= 43))
        | ((yy >= 61) & (yy <= 62) & (xx <= 42))
        | ((yy == 63) & (xx <= 40))
        | ((yy == 64) & (xx <= 39))
        | ((yy == 65) & (xx <= 37))
        | ((yy == 66) & (xx <= 35))
        | ((yy == 67) & (xx <= 33))
        | ((yy == 68) & (xx <= 30))
        | ((yy == 69) & (xx <= 28))
    )
    lightning_mane_upper = a & mane_upper_zone & (~lower_jaw) & (~head_upper) & (~front_leg_L)

    rem = (
        a
        & (~lower_jaw)
        & (~head_upper)
        & (~front_leg_L)
        & (~front_leg_R)
        & (~hind_leg_L)
        & (~hind_leg_R)
        & (~lightning_mane_upper)
    )

    is_blue = b > r + 15
    is_black = (r == 16) & (g == 16) & (b == 16)
    blue_rem = rem & is_blue
    blue_dil = ndi.binary_dilation(blue_rem, structure=np.ones((3, 3), dtype=bool))
    rear_tail_tip_outline = ((yy <= 53) & (xx >= 74)) | ((yy == 55) & (xx >= 73))
    torso = rem & (is_blue | (is_black & blue_dil & (~rear_tail_tip_outline)))

    rem_yellow = rem & (~torso)
    tail_spike_zone = (
        ((yy == 42) & (xx >= 77))
        | ((yy == 43) & (xx >= 76))
        | ((yy == 44) & (xx >= 75))
        | ((yy == 45) & (xx >= 74))
        | ((yy == 46) & (xx >= 73))
        | ((yy == 47) & (xx >= 72))
        | ((yy >= 48) & (yy <= 56) & (xx >= 70))
    )
    lightning_tail_rear = rem_yellow & tail_spike_zone
    mane_dorsal_back = rem_yellow & (~lightning_tail_rear)

    return {
        "lightning_tail_rear": lightning_tail_rear,
        "mane_dorsal_back": mane_dorsal_back,
        "hind_leg_L": hind_leg_L,
        "front_leg_L": front_leg_L,
        "torso": torso,
        "hind_leg_R": hind_leg_R,
        "front_leg_R": front_leg_R,
        "lightning_mane_upper": lightning_mane_upper,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` (facing right) into matching 10 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    r = arr96[:, :, 0].astype(np.int32)
    g = arr96[:, :, 1].astype(np.int32)
    b = arr96[:, :, 2].astype(np.int32)

    lower_jaw = a & (
        ((yy == 68) & (xx >= 69) & (xx <= 77))
        | ((yy == 69) & (xx >= 70) & (xx <= 76))
        | ((yy == 70) & (xx >= 72) & (xx <= 75))
        | ((yy == 71) & (xx >= 73) & (xx <= 74))
    )

    head_upper = a & (~lower_jaw) & (
        ((yy == 61) & (xx >= 72) & (xx <= 75))
        | ((yy == 62) & (xx >= 74) & (xx <= 78))
        | ((yy == 63) & (xx >= 73) & (xx <= 77))
        | ((yy == 64) & (xx >= 72) & (xx <= 78))
        | ((yy == 65) & (xx >= 70) & (xx <= 79))
        | ((yy == 66) & (xx >= 68) & (xx <= 79))
        | ((yy == 67) & (xx >= 68) & (xx <= 77))
    )

    hind_leg_L = a & (
        ((yy == 71) & (xx >= 24) & (xx <= 29))
        | ((yy == 72) & (xx >= 25) & (xx <= 44))
        | ((yy == 73) & (xx >= 26) & (xx <= 41))
        | ((yy >= 74) & (yy <= 88) & (xx >= 27) & (xx <= 41))
    )
    front_leg_L = a & (~hind_leg_L) & (
        ((yy >= 70) & (yy <= 77) & (xx >= 45) & (xx <= 58))
        | ((yy >= 78) & (yy <= 79) & (xx >= 42) & (xx <= 59))
        | ((yy >= 80) & (yy <= 83) & (xx >= 45) & (xx <= 56))
    )
    front_leg_R = a & (~hind_leg_L) & (~front_leg_L) & (
        ((yy >= 70) & (yy <= 77) & (xx >= 60) & (xx <= 68))
        | ((yy >= 78) & (yy <= 79) & (xx >= 59) & (xx <= 80))
        | ((yy >= 80) & (yy <= 85) & (xx >= 57) & (xx <= 80))
    )
    hind_leg_R = np.zeros_like(a)

    # Foreground portion of the giant lightning mane (upper crown + front-right shoulder blade)
    mane_upper_zone = (
        (yy <= 45)
        | ((yy >= 46) & (yy <= 56) & (xx >= 61))
        | ((yy >= 57) & (yy <= 60) & (xx >= 53))
        | ((yy >= 61) & (yy <= 63) & (xx >= 54))
        | ((yy == 64) & (xx >= 55))
        | ((yy == 65) & (xx >= 56))
        | ((yy == 66) & (xx >= 64))
        | ((yy == 67) & (xx >= 65))
        | ((yy == 68) & (xx >= 65))
        | ((yy == 69) & (xx >= 66))
    )
    lightning_mane_upper = a & mane_upper_zone & (~lower_jaw) & (~head_upper)

    rem = (
        a
        & (~lower_jaw)
        & (~head_upper)
        & (~hind_leg_L)
        & (~front_leg_L)
        & (~front_leg_R)
        & (~lightning_mane_upper)
    )

    is_blue = b > r + 10
    is_black = (r == 16) & (g == 16) & (b == 16)
    blue_rem = rem & is_blue
    blue_dil = ndi.binary_dilation(blue_rem, structure=np.ones((3, 3), dtype=bool))
    rear_left_spike_outline = (yy <= 59) & (xx <= 22)
    torso = rem & (is_blue | (is_black & blue_dil & (~rear_left_spike_outline)))

    rem_yellow = rem & (~torso)
    tail_spike_zone = (
        ((yy >= 47) & (yy <= 53) & (xx <= 22))
        | ((yy >= 54) & (yy <= 63) & (xx <= 30))
    )
    lightning_tail_rear = rem_yellow & tail_spike_zone
    mane_dorsal_back = rem_yellow & (~lightning_tail_rear)

    return {
        "lightning_tail_rear": lightning_tail_rear,
        "mane_dorsal_back": mane_dorsal_back,
        "hind_leg_L": hind_leg_L,
        "front_leg_L": front_leg_L,
        "torso": torso,
        "hind_leg_R": hind_leg_R,
        "front_leg_R": front_leg_R,
        "lightning_mane_upper": lightning_mane_upper,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
    }


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
    dst_y0 = foot_y - sh

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

    yellow_fill_m = is_yellow_mane_fill(canvas)
    blue_fill_m = is_blue_body_fill(canvas)
    fallback_blue = (63, 106, 115, 255) if is_back else (75, 117, 161, 255)

    layers: Dict[str, np.ndarray] = {}

    # 1. `lightning_tail_rear`: rear lightning tail spike behind the hip
    fg_over_tail = c_masks["torso"] | c_masks["mane_dorsal_back"] | c_masks["lightning_mane_upper"]
    u_tail = ndi.binary_erosion(c_masks["lightning_tail_rear"] | fg_over_tail, iterations=1)
    layers["lightning_tail_rear"] = build_part_layer(
        canvas,
        c_masks["lightning_tail_rear"],
        u_tail & fg_over_tail,
        cushion_px=8,
        allowed_fill_mask=yellow_fill_m,
    )

    # 2. `mane_dorsal_back`: dorsal portion of the giant lightning mane behind `torso`'s spiky blue hip fur
    fg_over_dorsal = c_masks["torso"] | c_masks["lightning_mane_upper"]
    u_dorsal = ndi.binary_erosion(c_masks["mane_dorsal_back"] | fg_over_dorsal, iterations=1)
    layers["mane_dorsal_back"] = build_part_layer(
        canvas,
        c_masks["mane_dorsal_back"],
        u_dorsal & fg_over_dorsal,
        cushion_px=10,
        allowed_fill_mask=yellow_fill_m,
    )

    # 3. `hind_leg_L` & `front_leg_L` (far-side legs)
    upper_for_legs = c_masks["torso"] | c_masks["lightning_mane_upper"] | c_masks["front_leg_R"]
    layers["hind_leg_L"] = build_foot_stem_layer(
        canvas, c_masks["hind_leg_L"], upper_for_legs, stem_px=4, fallback_rgba=fallback_blue
    )
    layers["front_leg_L"] = build_foot_stem_layer(
        canvas, c_masks["front_leg_L"], upper_for_legs, stem_px=4, fallback_rgba=fallback_blue
    )

    # 4. `torso`: drawn ON TOP OF `mane_dorsal_back` (so the 4 spiky blue hip fur tufts are 100% sharp in front),
    #    and cushioned ONLY under the lower chest/shoulder (`shoulder_cm`) and near legs!
    if not is_back:
        shoulder_m96 = np.zeros((96, 96), dtype=bool)
        shoulder_m96[56:70, 24:47] = True
    else:
        shoulder_m96 = np.zeros((96, 96), dtype=bool)
        shoulder_m96[65:71, 56:75] = True
    if flip_x:
        shoulder_m96 = shoulder_m96[:, ::-1]
    shoulder_cm = np.zeros((cell_h, cell_w), dtype=bool)
    shoulder_cm[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = shoulder_m96[y0:y1, x0:x1]

    fg_over_torso = (
        (c_masks["lightning_mane_upper"] & shoulder_cm)
        | c_masks["front_leg_R"]
        | c_masks["hind_leg_R"]
    )
    u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
    layers["torso"] = build_part_layer(
        canvas,
        c_masks["torso"],
        u_torso & fg_over_torso,
        cushion_px=6,
        allowed_fill_mask=blue_fill_m,
    )

    # 5. `hind_leg_R` & `front_leg_R` (near-side legs)
    layers["hind_leg_R"] = build_foot_stem_layer(
        canvas, c_masks["hind_leg_R"], c_masks["torso"], stem_px=4, fallback_rgba=fallback_blue
    )
    layers["front_leg_R"] = build_foot_stem_layer(
        canvas, c_masks["front_leg_R"], c_masks["torso"], stem_px=4, fallback_rgba=fallback_blue
    )

    # 6. `lightning_mane_upper`: drawn ON TOP OF `torso`, and cushioned under `lower_jaw` and `head_upper`!
    fg_over_mane = c_masks["lower_jaw"] | c_masks["head_upper"]
    u_mane = ndi.binary_erosion(c_masks["lightning_mane_upper"] | fg_over_mane, iterations=1)
    mane_lyr = build_part_layer(
        canvas,
        c_masks["lightning_mane_upper"],
        u_mane & fg_over_mane,
        cushion_px=7,
        allowed_fill_mask=yellow_fill_m,
    )

    # (2026-11-04) The rectangular dark-red mouth interior pocket is no longer painted; the mouth opens with
    # mouth_open.open_mouth along MOUTH_LINE_NATIVE_FRONT on the composed frame instead.

    layers["lightning_mane_upper"] = mane_lyr

    # 7. `lower_jaw` & `head_upper`: drawn in foreground with zero distortion
    layers["lower_jaw"] = build_part_layer(
        canvas, c_masks["lower_jaw"], np.zeros_like(c_masks["lower_jaw"]), cushion_px=0
    )
    layers["head_upper"] = build_part_layer(
        canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0
    )

    if not is_back:
        pivots = {
            "lightning_mane_upper": to_canvas_pt(42.0, 46.0),  # Neck/crown base pivot
            "lightning_tail_rear": to_canvas_pt(71.0, 52.0),   # Rear hip tail spike root pivot
        }
        pivots["mouth_line"] = [to_canvas_pt(mx, my) for mx, my in MOUTH_LINE_NATIVE_FRONT]
    else:
        pivots = {
            "lightning_mane_upper": to_canvas_pt(58.0, 54.0),  # Neck/crown base pivot on back.png
            "lightning_tail_rear": to_canvas_pt(28.0, 58.0),   # Rear hip tail spike root pivot on back.png
        }

    return layers, pivots


def foot_step_offset(phase: float, facing: float, lift_px: int = 2, stride_px: int = 2) -> Tuple[int, int]:
    p = phase % (2.0 * math.pi)
    s = math.sin(p)
    c = math.cos(p)
    dx = int(round(stride_px * c * facing))
    dy = int(round(-lift_px * max(0.0, s)))
    return dx, dy


def add_hurt_sparks(canvas: np.ndarray, frame_idx: int, is_back: bool) -> np.ndarray:
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
    spark_col = (255, 229, 102, 255) if is_back else (255, 240, 109, 255)
    pts = [
        (mid_x - 18 - drift, max(2, top_y + 18 - drift // 2)),
        (mid_x + 18 + drift, max(2, top_y + 20 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = spark_col
    return out


def synthesize_manectric_mega_frame(
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
    mane_dx, mane_dy = 0, 0
    tail_dx, tail_dy = 0, 0
    head_dx, head_dy = 0, 0
    jaw_dx, jaw_dy = 0, 0
    jaw_open = 0

    fl_dx, fl_dy = 0, 0
    fr_dx, fr_dy = 0, 0
    hl_dx, hl_dy = 0, 0
    hr_dx, hr_dy = 0, 0

    mane_ang = 0.0
    tail_ang = 0.0

    if action == "idle":
        # Thunder-wolf electric breathing + barking jaw + pivot-rotating giant lightning mane & rear tail:
        # 1. All 4 paws (`front_leg_L/R`, `hind_leg_L/R`) stay 100% planted on the ground `(0, 0)`.
        # 2. `torso` and `head_upper` heave rhythmically (`-1..+1px`) and lean forward (`+1px`) on the growl peak.
        # 3. `lower_jaw` opens `+1px` during the bark/growl frames, revealing the red mouth & white fang.
        # 4. `lightning_mane_upper` + `mane_dorsal_back` rotate (`±3.2°`) around the neck pivot while
        #    `lightning_tail_rear` whips (`±5.5°`) with phase lag around the hip root!
        torso_dy = int(round(1.0 * sin1))
        torso_dx = int(round(0.6 * max(0.0, sin1) * facing))

        mane_dx = int(round(0.8 * max(0.0, math.sin(phase - 0.15)) * facing))
        mane_dy = int(round(1.2 * math.sin(phase - 0.15)))
        mane_ang = 3.2 * math.sin(phase - 0.25) * (-facing)

        tail_dx, tail_dy = torso_dx, torso_dy
        tail_ang = 5.5 * math.sin(phase - 0.85) * (-facing)

        head_dx = int(round(1.0 * max(0.0, math.sin(phase - 0.20)) * facing))
        head_dy = int(round(1.3 * math.sin(phase - 0.20)))

        jaw_open = 1 if frame_idx in (2, 3, 4, 5, 11, 12, 13) else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; the mouth opens via mouth_open (no chin drop)

    elif action == "walk":
        # High-speed 4-legged diagonal trot/gallop (`front_leg_L + hind_leg_R` vs `front_leg_R + hind_leg_L`),
        # dynamic lightning mane & tail rotation swing, and barking lower jaw!
        bounce = int(round(-1.5 * abs(sin1)))
        sway_x = int(round(1.2 * sin1 * facing))
        torso_dx, torso_dy = sway_x, bounce

        mane_dx = sway_x + int(round(0.8 * cos1 * facing))
        mane_dy = bounce + int(round(1.0 * math.sin(2.0 * phase - 0.3)))
        mane_ang = 4.0 * math.sin(2.0 * phase - 0.35) * (-facing)

        tail_dx, tail_dy = sway_x, bounce
        tail_ang = 6.5 * math.sin(2.0 * phase - 0.95) * (-facing)

        head_dx = sway_x + int(round(1.2 * cos1 * facing))
        head_dy = bounce + int(round(1.1 * math.sin(2.0 * phase - 0.4)))

        jaw_open = 1 if frame_idx in (2, 3, 4, 10, 11, 12) else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; the mouth opens via mouth_open (no chin drop)

        fl_dx, fl_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        fr_dx, fr_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)
        hl_dx, hl_dy = foot_step_offset(phase + math.pi + 0.3, facing, lift_px=2, stride_px=2)
        hr_dx, hr_dy = foot_step_offset(phase + 0.3, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Mega Manectric "Thunder Roar Howl + Spinning Volt Tackle Lunge & Crunch"!
        # Frames 0..3 (Low Predator Crouch & Big +2px Thunder Howl):
        #   - Draws back (`-3px`), lowers body (`+1..+2px`), bristles `lightning_mane_upper` (`-6.5°`)
        #     and `lightning_tail_rear` (`-9.0°`), and opens `lower_jaw` WIDE (`+2px` ferocious howl!).
        # Frames 4..9 (Spinning Volt Tackle Lunge & Thunder Fang Crunch):
        #   - Explosive `+6px` forward pounce! Front legs leap/stomp forward (`+4px`),
        #   - Giant `lightning_mane_upper` and `lightning_tail_rear` execute a rapid rotary electric discharge spin!
        #   - Jaw roars `+2px` wide on early lunge (frames 4..5), snaps shut on crunch impact (frames 6..7),
        #     and barks (`+1px`) on follow-through (frames 8..9)!
        # Frames 10..15: Smooth follow-through recovery to stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.60 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        torso_dx = int(round(4.5 * k * facing))
        torso_dy = int(round(-1.0 * k)) if k > 0 else int(round(1.5 * abs(k)))

        mane_dx = int(round(5.0 * k * facing))
        mane_dy = torso_dy + (int(round(-1.0 * k)) if k > 0 else int(round(0.8 * abs(k))))

        mane_spin_table = [0.0, -3.5, -5.5, -6.5, 5.5, 7.5, -5.5, 6.5, -4.5, 3.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        tail_spin_table = [0.0, -5.0, -8.0, -9.5, 8.0, 10.5, -8.5, 9.0, -6.5, 4.5, -3.0, -1.5, 0.0, 0.0, 0.0, 0.0]
        mane_ang = mane_spin_table[frame_idx] * (-facing)
        tail_ang = tail_spin_table[frame_idx] * (-facing)

        tail_dx, tail_dy = torso_dx, torso_dy

        head_dx = int(round(6.0 * k * facing))
        if frame_idx in (1, 2, 3):
            head_dy = 2 if frame_idx in (2, 3) else 1
        elif frame_idx in (4, 5):
            head_dy = 0
        elif frame_idx in (6, 7, 8):
            head_dy = -1
        else:
            head_dy = 0

        if frame_idx in (2, 3, 4, 5):
            jaw_open = 2
        elif frame_idx in (1, 8, 9):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; the mouth opens via mouth_open (no chin drop)

        front_step = int(round(4.0 * max(0.0, k) * facing))
        front_back = int(round(2.0 * min(0.0, k) * facing))
        front_lift = -2 if frame_idx in (4, 5) else 0
        rear_slide = int(round(2.0 * max(0.0, k) * facing))

        fl_dx, fl_dy = front_step + front_back, front_lift
        fr_dx, fr_dy = int(round((front_step + front_back) * 0.8)), front_lift // 2
        hl_dx, hl_dy = rear_slide, 0
        hr_dx, hr_dy = rear_slide, 0

    elif action == "hurt":
        # Knockback + hit-shake + pained roar (`lower_jaw` open +2px) + mane/tail recoil spin + electric sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        torso_dx = int(round(-3.8 * env * facing)) + jit
        torso_dy = int(round(-1.0 * env))

        mane_dx = torso_dx + int(round(-0.6 * env * facing))
        mane_dy = int(round(-1.5 * env))
        mane_ang = -6.0 * env * (-facing)

        tail_dx, tail_dy = torso_dx, torso_dy
        tail_ang = -8.5 * env * (-facing)

        head_dx = mane_dx + int(round(-0.6 * env * facing))
        head_dy = int(round(-2.0 * env))

        jaw_open = 1 if (1 <= frame_idx <= 7) else 0  # small pained growl (mouth_open amount 1)
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; the mouth opens via mouth_open (no chin drop)

        slide_x = int(round(torso_dx * 0.65))
        fl_dx, fr_dx, hl_dx, hr_dx = slide_x, slide_x, slide_x, slide_x

    tail_rot = rotate_layer_nearest(layers["lightning_tail_rear"], tail_ang, pivots["lightning_tail_rear"])
    dorsal_rot = rotate_layer_nearest(layers["mane_dorsal_back"], mane_ang, pivots["lightning_mane_upper"])
    mane_rot = rotate_layer_nearest(layers["lightning_mane_upper"], mane_ang, pivots["lightning_mane_upper"])

    comp = [
        shift_layer_int(tail_rot, tail_dx, tail_dy),
        shift_layer_int(dorsal_rot, mane_dx, mane_dy),
        shift_layer_int(layers["hind_leg_L"], hl_dx, hl_dy),
        shift_layer_int(layers["front_leg_L"], fl_dx, fl_dy),
        shift_layer_int(layers["torso"], torso_dx, torso_dy),
        shift_layer_int(layers["hind_leg_R"], hr_dx, hr_dy),
        shift_layer_int(layers["front_leg_R"], fr_dx, fr_dy),
        shift_layer_int(mane_rot, mane_dx, mane_dy),
        shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
        shift_layer_int(layers["head_upper"], head_dx, head_dy),
    ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    mouth_line = pivots.get("mouth_line")
    if mouth_line and jaw_open > 0 and not is_back:
        line = mouth_open.transform_points(mouth_line, dx=head_dx, dy=head_dy)
        out = mouth_open.open_mouth(
            out, line, amount=min(2, jaw_open), deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=MOUTH_PEAK
        )
    if action == "hurt":
        out = add_hurt_sparks(out, frame_idx, is_back)
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

    # Tight bounds of front (84x74) and back (82x83)
    max_w, max_h = 84, 83
    cell_w = 112
    cell_h = 104
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
                frame_rgba = synthesize_manectric_mega_frame(
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

    # IMPORTANT: Preserve official 81-frame `front.gif` & `front-reverse.gif`,
    # and ONLY replace the broken 19-frame warp GIFs `back.gif` & `back-reverse.gif` with clean 16-frame `idle` N loops!
    idle_sheet = sheets_by_action["idle"]
    back_frames = [idle_sheet[4 * cell_h:5 * cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    back_rev_frames = [f[:, ::-1, :].copy() for f in back_frames]

    # 2026-11-04 mouth fix: front/back png/gif must stay untouched, so the back GIFs written by v1 are kept.
    # Set WRITE_BACK_GIFS = True to regenerate them from the N idle row again.
    WRITE_BACK_GIFS = False
    if WRITE_BACK_GIFS:
        save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
        save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)

    meta = {
        "generator": "forge_manectric_mega_actions",
        "schema": "agent-sprite-forge.meta.v2",
        "key": "0310_manectric-mega",
        "dex": 310,
        "archetype": "quadruped_beast",
        "rig_parts": [
            "lightning_tail_rear",
            "mane_dorsal_back",
            "hind_leg_L",
            "front_leg_L",
            "torso",
            "hind_leg_R",
            "front_leg_R",
            "lightning_mane_upper",
            "lower_jaw",
            "head_upper",
        ],
        "Idle": "../../monsterSprite/gen3/0310_manectric/mega/manectric-mega/agent-sprite-forge/idle.png",
        "Walk": "../../monsterSprite/gen3/0310_manectric/mega/manectric-mega/agent-sprite-forge/walk.png",
        "Attack": "../../monsterSprite/gen3/0310_manectric/mega/manectric-mega/agent-sprite-forge/attack.png",
        "Hurt": "../../monsterSprite/gen3/0310_manectric/mega/manectric-mega/agent-sprite-forge/hurt.png",
        "Sheet": "../../monsterSprite/gen3/0310_manectric/mega/manectric-mega/agent-sprite-forge/sheet.png",
        "Preview": "../../monsterSprite/gen3/0310_manectric/mega/manectric-mega/agent-sprite-forge/preview.png",
        "cols": N_FRAMES,
        "rows": 8,
        "cellW": cell_w,
        "cellH": cell_h,
        "footY": foot_y,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "foot_y": foot_y,
        "sprite_h": max_h,
        "frames": N_FRAMES,
        "paletteCount": int(len(palette_rgba)),
        "palette_colors": int(len(palette_rgba)),
        "off_palette_colors": 0,
        "directions": DIRS,
        "actions": {
            "idle": {"rowStart": 0, "rows": 8, "frames": N_FRAMES, "fps": 12},
            "walk": {"rowStart": 8, "rows": 8, "frames": N_FRAMES, "fps": 14},
            "attack": {"rowStart": 16, "rows": 8, "frames": N_FRAMES, "fps": 16},
            "hurt": {"rowStart": 24, "rows": 8, "frames": N_FRAMES, "fps": 12},
        },
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Sync to pokemon_arena_2d if present
    if SYNC_MEGA_DIR.exists():
        SYNC_OUT_DIR.mkdir(parents=True, exist_ok=True)
        for f_path in OUT_DIR.iterdir():
            if f_path.is_file():
                shutil.copy2(f_path, SYNC_OUT_DIR / f_path.name)
        for gif_name in ("back.gif", "back-reverse.gif"):
            shutil.copy2(MEGA_DIR / gif_name, SYNC_MEGA_DIR / gif_name)

    print(
        f"Built 0310_manectric-mega -> {OUT_DIR} "
        f"(cell={cell_w}x{cell_h}, foot_y={foot_y}, sprite_h={max_h}, colors={len(palette_rgba)})"
    )


if __name__ == "__main__":
    main()
