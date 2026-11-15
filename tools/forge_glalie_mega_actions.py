#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Glalie (`0362_glalie-mega` / メガオニゴーリ) modeled after Mega Glalie's
official 59-frame animated GIF (`front.gif` & `back.gif`) and native 96x96 sprites (`front.png` & `back.png`):
  - Exact 7-layer anatomical rigging with true joint pivots (horn bases, cheek armor mounts & unhinged jaw hinge):
      1. `horn_L` (far-side dark-rock horn with cyan ice crystal edge — pivoted at its skull root `rotate_layer_nearest`
                   to flex and roll during roars and Ice Spinner charges)
      2. `horn_R` (near-side towering dark-rock horn with dual cyan ice stripes — pivoted at its skull root
                   `rotate_layer_nearest` for intimidating horn threat & Ice Spinner rotation)
      3. `side_ice_cheeks_L` (left armored ice cheek / lower-left ice spike — counter-rotates with body roll)
      4. `side_ice_cheeks_R` (right armored ice cheek plate & lower-right ice spike — rolls & spins with body)
      5. `mouth_cavity_core` (deep gaping mouth interior dome + dark rock cheek walls — padded with seamless
                              deep-blue/navy oral cavity cushion so even when `lower_jaw_beard` drops `+3px` wide
                              for a Freeze-Dry roar, zero transparent gaps ever appear)
      6. `lower_jaw_beard` (massive bifurcated rock-and-ice beard lower jaw + lower teeth row — pivoted at the
                            jaw hinge `rotate_layer_nearest` + vertical shift to open `+1..+2px` for blizzard roars
                            in `idle`/`walk`, open maximum `+3px` during `attack` Freeze-Dry wind-up roar, and
                            clamp shut `-2px` for a bone-shattering Ice Fang / Crunch (`こおりのキバ` / `かみくだく`) bite!)
      7. `head_upper` (central forehead ice horn, glowing blue eyes with red pupils, white ice skull mask, and
                       upper 3 teeth — 100% rigid integer translation `shift_layer_int` ONLY so the face, eyes,
                       forehead horn, and upper teeth never deform by even 1 pixel!)
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
import mouth_open  # noqa: E402  small face-following mouth opening (2026-11-04 mouth fix)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0362_glalie/mega/glalie-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0362_glalie/mega/glalie-mega"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16

# 2026-11-04 mouth fix (user: rectangular jaw drop looked like a jutting chin):
# the jaw/beard no longer drops or rotates; instead a small lens-shaped gap is carved along the closed mouth line
# (bottom edge of the upper teeth, native front.png coords, left corner -> right corner) with mouth_open.
MOUTH_LINE_FRONT = [(30, 57), (33, 56), (37, 56), (41, 55), (45, 56), (47, 55), (49, 52), (52, 50)]
MOUTH_PEAK = 0.45
MOUTH_DEEP = (1, 1, 1)        # darkest palette colour (outline)
MOUTH_INNER = (32, 49, 94)    # existing navy throat shade of the blue mouth interior


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
    return a & ((r == 1) & (g == 1) & (b == 1))


def is_glalie_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 57) & (g == 57) & (b == 65))
        | ((r == 43) & (g == 116) & (b == 168))
        | ((r == 32) & (g == 49) & (b == 94))
        | ((r == 123) & (g == 123) & (b == 148))
        | ((r == 172) & (g == 172) & (b == 205))
        | ((r == 197) & (g == 205) & (b == 230))
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
    # Keep only the primary connected component so nearest-neighbor rotation never leaves a 1px stray speck
    a = out[..., 3] >= 128
    if np.any(a):
        lbl, n_comp = ndi.label(a)
        if n_comp > 1:
            sizes = ndi.sum(a, lbl, range(1, n_comp + 1))
            main_id = int(np.argmax(sizes)) + 1
            stray = a & (lbl != main_id)
            out[stray] = 0
    return out


def segment_front_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `front.png` into 7 exact anatomical masks following 1px outlines:
      1. `horn_L`: Far-side left dark-rock horn with cyan ice stripe (`x=19..28, y=22..33`).
      2. `horn_R`: Near-side right towering dark-rock horn with dual cyan ice stripes (`x=53..71, y=15..34`).
      3. `side_ice_cheeks_L`: Left cheek ice tip (`y=46..55`) + lower-left ice armor spike (`x=26..34, y=69..76`).
      4. `side_ice_cheeks_R`: Right cheek ice armor plate (`y=34..44, x=53..67`) + lower-right ice armor spike (`x=60..73, y=67..77`).
      5. `mouth_cavity_core`: Huge gaping blue/navy mouth cavity dome (`(43,116,168)`, `(32,49,94)`) + dark-rock side walls.
      6. `lower_jaw_beard`: Bifurcated black/charcoal ice-rock lower jaw beard + curved lower white teeth row (`y=57..80`).
      7. `head_upper`: Central forehead ice horn, blue/red eyes, white skull mask, and upper 3 teeth (`y=18..57`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. horn_L (far-side left dark horn with cyan crystal edge)
    horn_L = a & (
        ((yy == 22) & (xx <= 22))
        | ((yy == 23) & (xx <= 26))
        | ((yy >= 24) & (yy <= 28) & (xx <= 28))
        | ((yy >= 29) & (yy <= 30) & (xx <= 25))
        | ((yy >= 31) & (yy <= 32) & (xx <= 24))
        | ((yy == 33) & (xx <= 23))
    )

    # 2. horn_R (near-side right towering dark horn with two cyan crystal stripes)
    horn_R = a & (
        ((yy >= 15) & (yy <= 25) & (xx >= 53))
        | ((yy == 26) & ((xx == 53) | (xx >= 57)))
        | ((yy >= 27) & (yy <= 33) & (xx >= 60))
        | ((yy == 34) & (xx >= 62) & (xx <= 66))
    )

    # 3. side_ice_cheeks_L (left cheek ice tip + lower-left ice spike)
    side_ice_cheeks_L = a & (
        ((yy == 46) & (xx <= 23))
        | ((yy == 47) & (xx <= 23))
        | ((yy == 48) & (xx <= 22))
        | ((yy == 49) & (xx <= 22))
        | ((yy == 50) & (xx <= 23))
        | ((yy == 51) & (xx <= 24))
        | ((yy == 52) & (xx <= 24))
        | ((yy == 53) & (xx <= 23))
        | ((yy == 54) & (xx <= 24))
        | ((yy == 55) & (xx == 25))
        | ((yy == 69) & (xx >= 26) & (xx <= 30))
        | ((yy == 70) & (xx >= 26) & (xx <= 31))
        | ((yy == 71) & (xx >= 27) & (xx <= 32))
        | ((yy == 72) & (xx >= 28) & (xx <= 33))
        | ((yy == 73) & (xx >= 27) & (xx <= 34))
        | ((yy == 74) & (xx >= 27) & (xx <= 34))
        | ((yy == 75) & (xx >= 27) & (xx <= 32))
        | ((yy == 76) & (xx >= 27) & (xx <= 30))
    )

    # 4. side_ice_cheeks_R (right upper ice cheek armor plate + lower-right ice spike)
    side_ice_cheeks_R = (
        a
        & (
            ((yy == 34) & (xx >= 55))
            | ((yy >= 35) & (yy <= 41) & (xx >= 53))
            | ((yy == 42) & (xx >= 61))
            | ((yy == 43) & (xx >= 63))
            | ((yy == 44) & (xx >= 66) & (xx <= 67))
            | ((yy == 67) & (xx >= 70))
            | ((yy == 68) & (xx >= 68))
            | ((yy == 69) & (xx >= 64))
            | ((yy == 70) & (xx >= 61))
            | ((yy >= 71) & (yy <= 73) & (xx >= 60))
            | ((yy == 74) & (xx >= 61))
            | ((yy == 75) & (xx >= 63))
            | ((yy == 76) & (xx >= 66))
            | ((yy == 77) & (xx >= 71))
        )
        & (~horn_R)
    )

    # 5. head_upper (central forehead ice horn, angry blue/red eyes, white skull mask, and upper 3 teeth)
    head_upper = (
        a
        & (
            ((yy >= 18) & (yy <= 49) & (xx <= 61))
            | ((yy == 50) & ((xx <= 49) | ((xx >= 53) & (xx <= 59))))
            | ((yy == 51) & ((xx <= 48) | ((xx >= 55) & (xx <= 59))))
            | ((yy == 52) & ((xx <= 47) | ((xx >= 55) & (xx <= 58))))
            | ((yy == 53) & ((xx <= 46) | ((xx >= 57) & (xx <= 59))))
            | ((yy == 54) & (xx <= 46))
            | ((yy == 55) & ((xx <= 29) | ((xx >= 32) & (xx <= 46))))
            | ((yy == 56) & ((xx <= 29) | ((xx >= 42) & (xx <= 45))))
            | ((yy == 57) & (xx >= 26) & (xx <= 28))
        )
        & (~horn_L)
        & (~horn_R)
        & (~side_ice_cheeks_L)
        & (~side_ice_cheeks_R)
    )

    # 6. lower_jaw_beard (split black/charcoal ice-rock lower jaw beard + curved lower white teeth row)
    r = arr96[..., 0].astype(np.int32)
    g = arr96[..., 1].astype(np.int32)
    b = arr96[..., 2].astype(np.int32)
    is_mouth_blue_or_navy = a & (
        ((r == 43) & (g == 116) & (b == 168))
        | ((r == 32) & (g == 49) & (b == 94))
        | ((r == 125) & (g == 187) & (b == 238) & (yy >= 55))
    )

    lower_jaw_beard = (
        a
        & (yy >= 57)
        & (~head_upper)
        & (~side_ice_cheeks_L)
        & (~side_ice_cheeks_R)
        & (~is_mouth_blue_or_navy)
    )

    used = (
        horn_L
        | horn_R
        | side_ice_cheeks_L
        | side_ice_cheeks_R
        | head_upper
        | lower_jaw_beard
    )
    mouth_cavity_core = a & (~used)

    return {
        "horn_L": horn_L,
        "horn_R": horn_R,
        "side_ice_cheeks_L": side_ice_cheeks_L,
        "side_ice_cheeks_R": side_ice_cheeks_R,
        "mouth_cavity_core": mouth_cavity_core,
        "lower_jaw_beard": lower_jaw_beard,
        "head_upper": head_upper,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 7 anatomical masks following the true 1px outlines.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. horn_L (left dark horn with cyan ice stripe: y=16..32, x<=35)
    horn_L = a & (
        ((yy >= 16) & (yy <= 20) & (xx <= 30))
        | ((yy >= 21) & (yy <= 23) & (xx <= 35))
        | ((yy >= 24) & (yy <= 26) & (xx <= 28))
        | ((yy >= 27) & (yy <= 29) & (xx <= 27))
        | ((yy >= 30) & (yy <= 31) & (xx <= 26))
        | ((yy == 32) & (xx <= 25))
    )

    # 2. horn_R (right dark horn with two cyan ice stripes: y=16..37, x>=56)
    horn_R = a & (
        ((yy >= 16) & (yy <= 33) & (xx >= 58))
        | ((yy >= 34) & (yy <= 36) & (xx >= 56) & (xx <= 72))
        | ((yy == 37) & (xx >= 58) & (xx <= 68))
    )

    # 3. side_ice_cheeks_L (lower-left white ice spike on back: y=65..73, x<=27)
    side_ice_cheeks_L = a & (
        ((yy >= 65) & (yy <= 69) & (xx <= 25))
        | ((yy >= 70) & (yy <= 71) & (xx <= 26))
        | ((yy >= 72) & (yy <= 73) & (xx <= 27))
    )

    # 4. side_ice_cheeks_R (upper-right white ice cheek y=34..45 x>=69 + lower-right ice spike y=67..79 x>=48)
    side_ice_cheeks_R = (
        a
        & (
            ((yy >= 34) & (yy <= 45) & (xx >= 69))
            | ((yy >= 67) & (yy <= 69) & (xx >= 48) & (xx <= 55))
            | ((yy >= 70) & (yy <= 71) & (xx >= 48) & (xx <= 57))
            | ((yy >= 72) & (yy <= 79) & (xx >= 49))
        )
        & (~horn_R)
    )

    # 5. head_upper (top white ice skull dome + back cyan diamond crystal + tip of forehead ice horn)
    head_upper = (
        a
        & (
            ((yy >= 18) & (yy <= 40))
            | ((yy == 41) & (xx >= 27) & (xx <= 74))
            | ((yy == 42) & (xx >= 28) & (xx <= 66))
            | ((yy == 43) & (xx >= 29) & (xx <= 64))
            | ((yy == 44) & (xx >= 30) & (xx <= 63))
            | ((yy == 45) & (xx >= 31) & (xx <= 62))
            | ((yy == 46) & (xx >= 33) & (xx <= 60))
            | ((yy == 47) & (xx >= 34) & (xx <= 40))
            | ((yy == 48) & (xx >= 35) & (xx <= 38))
        )
        & (~horn_L)
        & (~horn_R)
        & (~side_ice_cheeks_R)
    )

    # 6. lower_jaw_beard (lower rock beard mass & right-side lower jaw teeth at y=57..79)
    lower_jaw_beard = (
        a
        & (yy >= 57)
        & (~side_ice_cheeks_L)
        & (~side_ice_cheeks_R)
    )

    used = (
        horn_L
        | horn_R
        | side_ice_cheeks_L
        | side_ice_cheeks_R
        | head_upper
        | lower_jaw_beard
    )
    mouth_cavity_core = a & (~used)

    return {
        "horn_L": horn_L,
        "horn_R": horn_R,
        "side_ice_cheeks_L": side_ice_cheeks_L,
        "side_ice_cheeks_R": side_ice_cheeks_R,
        "mouth_cavity_core": mouth_cavity_core,
        "lower_jaw_beard": lower_jaw_beard,
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
    places onto the target `(cell_h, cell_w)` canvas with bottom aligned to `foot_y`,
    and builds cushioned RGBA layers + true anatomical joint pivots.
    CRITICAL RULE: Background layers are NEVER included in `foreground_mask`.
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

    body_fill_m = is_glalie_body_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order (back-to-front):
        # 1. horn_L -> 2. horn_R -> 3. side_ice_cheeks_L -> 4. mouth_cavity_core
        # -> 5. lower_jaw_beard -> 6. side_ice_cheeks_R -> 7. head_upper
        fg_over_horn_L = c_masks["head_upper"]
        u_hl = ndi.binary_erosion(c_masks["horn_L"] | fg_over_horn_L, iterations=1)
        layers["horn_L"] = build_part_layer(
            canvas,
            c_masks["horn_L"],
            u_hl & fg_over_horn_L,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_horn_R = c_masks["head_upper"] | c_masks["side_ice_cheeks_R"]
        u_hr = ndi.binary_erosion(c_masks["horn_R"] | fg_over_horn_R, iterations=1)
        layers["horn_R"] = build_part_layer(
            canvas,
            c_masks["horn_R"],
            u_hr & fg_over_horn_R,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_cheek_L = c_masks["head_upper"] | c_masks["lower_jaw_beard"] | c_masks["mouth_cavity_core"]
        u_cl = ndi.binary_erosion(c_masks["side_ice_cheeks_L"] | fg_over_cheek_L, iterations=1)
        layers["side_ice_cheeks_L"] = build_part_layer(
            canvas,
            c_masks["side_ice_cheeks_L"],
            u_cl & fg_over_cheek_L,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        # IMPORTANT: `mouth_cavity_core` cushion ONLY expands under foreground layers drawn ON TOP OF it
        # (`lower_jaw_beard`, `side_ice_cheeks_R`, `head_upper`), NEVER over background layers!
        fg_over_mouth = c_masks["lower_jaw_beard"] | c_masks["side_ice_cheeks_R"] | c_masks["head_upper"]
        u_mouth = ndi.binary_erosion(c_masks["mouth_cavity_core"] | fg_over_mouth, iterations=1)
        mouth_lyr = build_part_layer(
            canvas,
            c_masks["mouth_cavity_core"],
            u_mouth & fg_over_mouth,
            cushion_px=6,
            allowed_fill_mask=body_fill_m,
        )

        # (2026-11-04) The rectangular oral-cavity pocket painted behind the jaw rim was removed: the jaw no longer
        # drops, the mouth opens with mouth_open along the teeth line instead.
        jaw_ys, jaw_xs = np.where(c_masks["lower_jaw_beard"])
        layers["mouth_cavity_core"] = mouth_lyr

        # `lower_jaw_beard` is drawn over `mouth_cavity_core`, and underneath `side_ice_cheeks_R` and `head_upper`.
        # Add a 4px upper stem copying the row just above `jy_min` so downward shift + rotation never leaves a notch!
        fg_over_jaw = c_masks["side_ice_cheeks_R"] | c_masks["head_upper"]
        u_jaw = ndi.binary_erosion(c_masks["lower_jaw_beard"] | fg_over_jaw, iterations=1)
        jaw_lyr = build_part_layer(
            canvas,
            c_masks["lower_jaw_beard"],
            u_jaw & fg_over_jaw,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )
        if len(jaw_ys) > 0:
            jy_min = int(jaw_ys.min())
            top_xs = jaw_xs[jaw_ys == jy_min]
            if len(top_xs) > 0:
                tx_min, tx_max = int(top_xs.min()), int(top_xs.max())
                for dy_up in range(1, 5):
                    py = jy_min - dy_up
                    if py >= 0:
                        for px in range(tx_min, tx_max + 1):
                            if c_masks["mouth_cavity_core"][py, px]:
                                jaw_lyr[py, px] = canvas[py, px]
        layers["lower_jaw_beard"] = jaw_lyr

        fg_over_cheek_R = c_masks["head_upper"]
        u_cr = ndi.binary_erosion(c_masks["side_ice_cheeks_R"] | fg_over_cheek_R, iterations=1)
        layers["side_ice_cheeks_R"] = build_part_layer(
            canvas,
            c_masks["side_ice_cheeks_R"],
            u_cr & fg_over_cheek_R,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        # Topmost foreground layer `head_upper` has zero foreground_mask so its crisp 1px outline never expands
        layers["head_upper"] = build_part_layer(
            canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0
        )

        pivots = {
            "horn_L": to_canvas_pt(25.0, 29.0),
            "horn_R": to_canvas_pt(58.0, 29.0),
            "side_ice_cheeks_L": to_canvas_pt(30.0, 62.0),
            "side_ice_cheeks_R": to_canvas_pt(60.0, 52.0),
            "lower_jaw_beard": to_canvas_pt(52.0, 62.0),
        }

    else:
        # Back view draw order (back-to-front):
        # 1. horn_L -> 2. side_ice_cheeks_L -> 3. mouth_cavity_core -> 4. lower_jaw_beard
        # -> 5. side_ice_cheeks_R -> 6. head_upper -> 7. horn_R
        fg_over_horn_L = c_masks["head_upper"]
        u_hl = ndi.binary_erosion(c_masks["horn_L"] | fg_over_horn_L, iterations=1)
        layers["horn_L"] = build_part_layer(
            canvas,
            c_masks["horn_L"],
            u_hl & fg_over_horn_L,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_cheek_L = c_masks["mouth_cavity_core"] | c_masks["lower_jaw_beard"]
        u_cl = ndi.binary_erosion(c_masks["side_ice_cheeks_L"] | fg_over_cheek_L, iterations=1)
        layers["side_ice_cheeks_L"] = build_part_layer(
            canvas,
            c_masks["side_ice_cheeks_L"],
            u_cl & fg_over_cheek_L,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_mouth = c_masks["lower_jaw_beard"] | c_masks["side_ice_cheeks_R"] | c_masks["head_upper"]
        u_mouth = ndi.binary_erosion(c_masks["mouth_cavity_core"] | fg_over_mouth, iterations=1)
        mouth_back_lyr = build_part_layer(
            canvas,
            c_masks["mouth_cavity_core"],
            u_mouth & fg_over_mouth,
            cushion_px=6,
            allowed_fill_mask=body_fill_m,
        )
        jaw_ys, jaw_xs = np.where(c_masks["lower_jaw_beard"])
        if len(jaw_ys) > 0:
            jy_min = int(jaw_ys.min())
            for py in range(jy_min, min(cell_h, jy_min + 10)):
                row_xs = jaw_xs[jaw_ys == py]
                if len(row_xs) > 0:
                    jx_min, jx_max = int(row_xs.min()), int(row_xs.max())
                    for px in range(jx_min, jx_max + 1):
                        if c_masks["lower_jaw_beard"][py, px]:
                            mouth_back_lyr[py, px] = canvas[py, px]
        layers["mouth_cavity_core"] = mouth_back_lyr

        fg_over_jaw = c_masks["side_ice_cheeks_R"]
        u_jaw = ndi.binary_erosion(c_masks["lower_jaw_beard"] | fg_over_jaw, iterations=1)
        jaw_back_lyr = build_part_layer(
            canvas,
            c_masks["lower_jaw_beard"],
            u_jaw & fg_over_jaw,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )
        jaw_ys, jaw_xs = np.where(c_masks["lower_jaw_beard"])
        if len(jaw_ys) > 0:
            jy_min = int(jaw_ys.min())
            top_xs = jaw_xs[jaw_ys == jy_min]
            if len(top_xs) > 0:
                tx_min, tx_max = int(top_xs.min()), int(top_xs.max())
                for dy_up in range(1, 5):
                    py = jy_min - dy_up
                    if py >= 0:
                        for px in range(tx_min, tx_max + 1):
                            if c_masks["mouth_cavity_core"][py, px]:
                                jaw_back_lyr[py, px] = canvas[py, px]
        layers["lower_jaw_beard"] = jaw_back_lyr

        fg_over_cheek_R = c_masks["head_upper"] | c_masks["horn_R"]
        u_cr = ndi.binary_erosion(c_masks["side_ice_cheeks_R"] | fg_over_cheek_R, iterations=1)
        layers["side_ice_cheeks_R"] = build_part_layer(
            canvas,
            c_masks["side_ice_cheeks_R"],
            u_cr & fg_over_cheek_R,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_head = c_masks["horn_R"]
        u_hd = ndi.binary_erosion(c_masks["head_upper"] | fg_over_head, iterations=1)
        layers["head_upper"] = build_part_layer(
            canvas,
            c_masks["head_upper"],
            u_hd & fg_over_head,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        layers["horn_R"] = build_part_layer(
            canvas, c_masks["horn_R"], np.zeros_like(c_masks["horn_R"]), cushion_px=0
        )

        pivots = {
            "horn_L": to_canvas_pt(27.0, 26.0),
            "horn_R": to_canvas_pt(63.0, 31.0),
            "side_ice_cheeks_L": to_canvas_pt(25.0, 68.0),
            "side_ice_cheeks_R": to_canvas_pt(58.0, 64.0),
            "lower_jaw_beard": to_canvas_pt(45.0, 60.0),
        }

    return layers, pivots


def add_blizzard_roar_particles(
    canvas: np.ndarray,
    frame_idx: int,
    facing: float,
    is_back: bool,
    intensity: int = 1,
) -> np.ndarray:
    """
    Adds crisp palette-locked ice crystal particles (`(187, 238, 255)` & `(254, 254, 254)`)
    bursting from Mega Glalie's giant unhinged mouth during Freeze-Dry / Blizzard roars!
    """
    out = canvas.copy()
    a = out[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return out
    h, w = out.shape[:2]
    mid_x = int(np.median(xs)) + int(round(8 * facing))
    mid_y = int(np.median(ys)) + (6 if not is_back else -2)
    drift = (frame_idx % 5) + 1

    offsets = [
        (int(round((14 + drift * 2) * facing)), -4 - drift),
        (int(round((18 + drift * 2) * facing)), 3 + drift),
    ]
    if intensity >= 2:
        offsets.append((int(round((12 + drift * 3) * facing)), 8 + drift))
        offsets.append((int(round((-16 - drift) * facing)), -2 - drift))

    for idx, (ox, oy) in enumerate(offsets):
        px = mid_x + ox
        py = mid_y + oy
        if 2 <= px < w - 3 and 2 <= py < h - 3:
            col = (254, 254, 254, 255) if idx % 2 == 0 else (187, 238, 255, 255)
            out[py, px] = col
            out[py, px + 1] = (125, 187, 238, 255)
            out[py + 1, px] = (125, 187, 238, 255)
    return out


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
        (mid_x - 18 - drift, max(2, top_y + 12 - drift // 2)),
        (mid_x + 18 + drift, max(2, top_y + 14 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (187, 238, 255, 255)
            out[sy, sx] = (254, 254, 254, 255)
    return out


def synthesize_glalie_mega_frame(
    layers: Dict[str, np.ndarray],
    pivots: Dict[str, Tuple[float, float]],
    canvas_shape: Tuple[int, int, int],
    action: str,
    frame_idx: int,
    facing: float,
    is_back: bool,
    palette_rgba: np.ndarray,
    mouth_line: Optional[List[Tuple[float, float]]] = None,
    mouth_peak: float = MOUTH_PEAK,
) -> np.ndarray:
    t = frame_idx / float(N_FRAMES)
    phase = 2.0 * math.pi * t
    sin1 = math.sin(phase)
    cos1 = math.cos(phase)

    mouth_dx, mouth_dy = 0, 0
    head_dx, head_dy = 0, 0
    jaw_dx, jaw_dy = 0, 0
    hl_dx, hl_dy = 0, 0
    hr_dx, hr_dy = 0, 0
    cl_dx, cl_dy = 0, 0
    cr_dx, cr_dy = 0, 0

    hl_ang = 0.0
    hr_ang = 0.0
    cl_ang = 0.0
    cr_ang = 0.0
    jaw_ang = 0.0  # (2026-11-04) the jaw is never rotated any more
    mouth_amt = 0
    roar_FX = 0

    if action == "idle":
        # Heavy Floating Hover (inspired by official 59-frame GIF bobbing + blizzard roar vibration):
        # 1. Mega Glalie levitates in a deep, rhythmic figure-8 hover (`-2..+2px` y, `±1px` x).
        # 2. The giant bifurcated ice beard lower jaw (`lower_jaw_beard`) drops open `+1..+2px` and pivots
        #    (`±3.5°`) to unleash a menacing Blizzard Roar at frames `3..7` and a second bark at `11..13`!
        # 3. The twin dark horns (`horn_L`, `horn_R`) and side ice armor cheeks (`side_ice_cheeks_L/R`)
        #    pivot and roll (`±4.0°`) with atmospheric phase lag.
        body_dy = int(round(2.0 * math.sin(phase - 0.2)))
        body_dx = int(round(1.0 * math.cos(phase - 0.2) * facing))

        mouth_dx, mouth_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        if frame_idx in (4, 5, 6):
            mouth_amt = 1
            roar_FX = 1
        elif frame_idx in (3, 7, 11, 12):
            mouth_amt = 1
        else:
            mouth_amt = 0

        jaw_dx = head_dx
        jaw_dy = head_dy

        hl_dx, hl_dy = body_dx, body_dy
        hl_ang = 4.0 * math.sin(phase - 0.5) * facing

        hr_dx, hr_dy = body_dx, body_dy
        hr_ang = 4.5 * math.sin(phase - 0.5) * (-facing)

        cl_dx, cl_dy = body_dx, body_dy + int(round(0.8 * math.sin(phase - 0.7)))
        cl_ang = 3.5 * math.sin(phase - 0.7) * facing

        cr_dx, cr_dy = body_dx, body_dy + int(round(0.8 * math.sin(phase - 0.7)))
        cr_ang = 4.0 * math.sin(phase - 0.7) * (-facing)

    elif action == "walk":
        # Menacing Ice-Boulder Roll & Hover Advance + Rhythmic Jaw Roar:
        # 2-cycle hovering surge (`2.0 * phase`), rolling ice-armor swing (`horn_L/R ±6.5°`, `side_ice_cheeks ±6.0°`),
        # and intimidating jaw snapping (`+1..+2px` open -> `0px` snap) on each propulsion beat!
        surge_x = int(round(2.0 * sin1 * facing))
        surge_y = int(round(1.8 * math.sin(2.0 * phase - 0.3)))

        mouth_dx, mouth_dy = surge_x, surge_y
        head_dx = surge_x + int(round(0.8 * cos1 * facing))
        head_dy = surge_y

        s2 = math.sin(2.0 * phase)
        if s2 > 0.60:
            mouth_amt = 1
            roar_FX = 1
        elif s2 > 0.15:
            mouth_amt = 1
        else:
            mouth_amt = 0

        jaw_dx = head_dx
        jaw_dy = head_dy

        hl_dx, hl_dy = head_dx, head_dy
        hl_ang = 6.0 * math.sin(2.0 * phase - 0.4) * facing

        hr_dx, hr_dy = head_dx, head_dy
        hr_ang = 6.5 * math.sin(2.0 * phase - 0.4) * (-facing)

        cl_dx = surge_x + int(round(-0.8 * cos1 * facing))
        cl_dy = surge_y + int(round(1.0 * math.sin(2.0 * phase - 0.5)))
        cl_ang = 6.0 * math.sin(2.0 * phase - 0.5) * facing

        cr_dx = surge_x + int(round(0.8 * cos1 * facing))
        cr_dy = surge_y + int(round(-1.0 * math.sin(2.0 * phase - 0.5)))
        cr_ang = -6.0 * math.sin(2.0 * phase - 0.5) * (-facing)

    elif action == "attack":
        # Signature Mega Glalie "Freeze-Dry Mega Roar + High-Speed Ice Spinner Roll & Refrigerate Jaw Crunch (`かみくだく` / `こおりのキバ`)"!
        # Frames 0..3 (Wind-up & Maximum +3px Freeze-Dry Roar):
        #   - Pulls massive ice boulder body back (`-4px`), tilts horns back (`-9°`),
        #   - Unhinges `lower_jaw_beard` to MAXIMUM (`+3px` & `+6.5°` wide open!) blasting Freeze-Dry crystals!
        # Frames 4..10 (High-Speed Ice Spinner Barrel-Roll Lunge & Bone-Crushing Jaw Crunch!):
        #   - Explosive forward lunge (`+7px` at frames 6..7),
        #   - Vigorous high-speed Ice Spinner rotation across horns, cheek armor, and jaw (`±11°..±12°` alternating roll!),
        #   - Giant ice beard lower jaw slams shut (`-2px` upward crunch at frames 6..7) to crush the target,
        #     followed by a victorious follow-through roar (`+2px` at frames 9..10)!
        # Frames 11..15: Smooth airborne recovery to hover stance.
        lunge_x_table = [0, -2, -3, -4, 3, 6, 7, 7, 5, 3, 2, 0, -1, -1, 0, 0]
        lunge_y_table = [0, -1, -2, -2, 0, 1, 2, 1, 0, -1, 0, 1, 0, 0, 0, 0]

        body_dx = int(round(lunge_x_table[frame_idx] * facing))
        body_dy = lunge_y_table[frame_idx]

        mouth_dx, mouth_dy = body_dx, body_dy
        head_dx = body_dx + (int(round(1.0 * facing)) if frame_idx in (5, 6, 7) else 0)
        head_dy = body_dy

        # Jaw curve: wide +2..+3px Freeze-Dry roar on wind-up (1..4), clamp shut -2px on Ice Fang/Crunch impact (6..7),
        # follow-through roar +2px (9..10)
        # (2026-11-04) jaw stays on the head; roar = small opening along the teeth line (2 px at the peak only),
        # crunch frames 5..8 = mouth closed.
        mouth_amt_table = [0, 1, 2, 2, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        mouth_amt = mouth_amt_table[frame_idx]
        jaw_dx = head_dx
        jaw_dy = head_dy

        if frame_idx in (2, 3, 4, 9, 10):
            roar_FX = 2 if frame_idx in (2, 3) else 1

        # Ice Spinner barrel-roll oscillation across horns and cheek armor plates during strike frames (4..10)
        roll_horn_L = [0.0, -5.0, -7.5, -9.5, 10.0, -11.0, 11.0, -10.0, 8.0, -5.5, 3.0, -1.5, 0.0, 0.0, 0.0, 0.0]
        roll_horn_R = [0.0, 5.0, 8.0, 10.5, -11.0, 12.0, -12.0, 11.0, -8.5, 6.0, -3.5, 1.5, 0.0, 0.0, 0.0, 0.0]
        roll_cheek_L = [0.0, -4.5, -7.0, -8.5, 9.5, -10.5, 10.5, -9.5, 7.5, -5.0, 2.5, -1.0, 0.0, 0.0, 0.0, 0.0]
        roll_cheek_R = [0.0, 4.5, 7.5, 9.5, -10.5, 11.5, -11.5, 10.5, -8.0, 5.5, -3.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        spinner_dy = [0, 0, -1, -1, 1, -2, 2, -2, 1, -1, 0, 0, 0, 0, 0, 0]

        hl_dx = head_dx
        hl_dy = head_dy + (spinner_dy[frame_idx] // 2)
        hl_ang = roll_horn_L[frame_idx] * facing

        hr_dx = head_dx
        hr_dy = head_dy - (spinner_dy[frame_idx] // 2)
        hr_ang = roll_horn_R[frame_idx] * (-facing)

        cl_dx = body_dx
        cl_dy = body_dy + spinner_dy[frame_idx]
        cl_ang = roll_cheek_L[frame_idx] * facing

        cr_dx = body_dx + (int(round(1.0 * facing)) if frame_idx in (5, 6, 7) else 0)
        cr_dy = body_dy - spinner_dy[frame_idx]
        cr_ang = roll_cheek_R[frame_idx] * (-facing)

    elif action == "hurt":
        # Heavy recoil knockback + unhinged jaw shock (`+2..+3px`) + hit-shake + ice sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-4.0 * env * facing)) + jit
        body_dy = int(round(-1.2 * env))

        mouth_dx, mouth_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-1.5 * env))

        mouth_amt = 1 if 1 <= frame_idx <= 8 else 0

        jaw_dx = head_dx
        jaw_dy = head_dy

        hl_dx, hl_dy = head_dx, head_dy
        hl_ang = -6.5 * env * facing

        hr_dx, hr_dy = head_dx, head_dy
        hr_ang = -7.5 * env * (-facing)

        cl_dx, cl_dy = body_dx, body_dy + int(round(1.0 * env))
        cl_ang = -6.0 * env * facing

        cr_dx, cr_dy = body_dx, body_dy + int(round(1.0 * env))
        cr_ang = -6.5 * env * (-facing)

    hl_rot = rotate_layer_nearest(layers["horn_L"], hl_ang, pivots["horn_L"])
    hr_rot = rotate_layer_nearest(layers["horn_R"], hr_ang, pivots["horn_R"])
    cl_rot = rotate_layer_nearest(layers["side_ice_cheeks_L"], cl_ang, pivots["side_ice_cheeks_L"])
    cr_rot = rotate_layer_nearest(layers["side_ice_cheeks_R"], cr_ang, pivots["side_ice_cheeks_R"])
    jaw_rot = rotate_layer_nearest(layers["lower_jaw_beard"], jaw_ang, pivots["lower_jaw_beard"])

    if not is_back:
        comp = [
            shift_layer_int(hl_rot, hl_dx, hl_dy),
            shift_layer_int(hr_rot, hr_dx, hr_dy),
            shift_layer_int(cl_rot, cl_dx, cl_dy),
            shift_layer_int(layers["mouth_cavity_core"], head_dx, head_dy),
            shift_layer_int(jaw_rot, jaw_dx, jaw_dy),
            shift_layer_int(cr_rot, cr_dx, cr_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
        ]
    else:
        comp = [
            shift_layer_int(hl_rot, hl_dx, hl_dy),
            shift_layer_int(cl_rot, cl_dx, cl_dy),
            shift_layer_int(layers["mouth_cavity_core"], head_dx, head_dy),
            shift_layer_int(jaw_rot, jaw_dx, jaw_dy),
            shift_layer_int(cr_rot, cr_dx, cr_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(hr_rot, hr_dx, hr_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    if mouth_line is not None and mouth_amt > 0:
        line = mouth_open.transform_points(mouth_line, dx=head_dx, dy=head_dy)
        out = mouth_open.open_mouth(out, line, amount=mouth_amt, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=mouth_peak)
    out = snap_to_palette(out, palette_rgba)
    if roar_FX > 0:
        out = add_blizzard_roar_particles(out, frame_idx, facing, is_back, intensity=roar_FX)
    if action == "hurt":
        out = add_hurt_sparks(out, frame_idx)
    return out


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

    # Tight bounds of front (58x66) and back (61x64)
    max_w, max_h = 61, 66
    cell_w = 96
    cell_h = 92
    foot_y = 82

    dir_rigs: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]], float, bool]] = {}
    mouth_lines: Dict[str, Tuple[List[Tuple[float, float]], float]] = {}
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
        # Mouth line in cell coords (front views only; back views show no mouth). Native front bbox (19,15)
        # lands at (19,16) in the cell, so cell = native + (0,+1); flipped views mirror x as 95 - x.
        if not use_back:
            ml = mouth_open.transform_points(MOUTH_LINE_FRONT, dy=1, mirror_w=96 if flip_x else None)
            mouth_lines[d_name] = (ml, MOUTH_PEAK)

    sheets_by_action: Dict[str, np.ndarray] = {}
    master_rows: List[np.ndarray] = []

    for act in ACTIONS:
        act_sheet = np.zeros((8 * cell_h, N_FRAMES * cell_w, 4), dtype=np.uint8)
        for d_idx, d_name in enumerate(DIRS):
            layers, pivots, facing, use_back = dir_rigs[d_name]
            for f_idx in range(N_FRAMES):
                frame_rgba = synthesize_glalie_mega_frame(
                    layers,
                    pivots,
                    (cell_h, cell_w, 4),
                    act,
                    f_idx,
                    facing,
                    use_back,
                    palette_rgba,
                    mouth_line=mouth_lines.get(d_name, (None, MOUTH_PEAK))[0],
                    mouth_peak=mouth_lines.get(d_name, (None, MOUTH_PEAK))[1],
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

    # NOTE: Preserve existing official 59-frame `front.gif` and `back.gif` untouched!

    meta = {
        "generator": "forge_glalie_mega_actions",
        "key": "0362_glalie-mega",
        "dex": 362,
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

    # Sync `agent-sprite-forge/` to pokemon_arena_2d if present
    if SYNC_MEGA_DIR.exists():
        SYNC_OUT_DIR.mkdir(parents=True, exist_ok=True)
        for f_path in OUT_DIR.iterdir():
            if f_path.is_file():
                shutil.copy2(f_path, SYNC_OUT_DIR / f_path.name)

    print(f"Built 0362_glalie-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, fy={foot_y}, sh={max_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
