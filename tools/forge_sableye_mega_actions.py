#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Sableye (`0302_sableye-mega`) modeled after Normal Sableye's official GIF
(`0302_sableye/normal/front.gif`) and Mega Sableye's anatomical design (`front.png` & `back.png`):
  - Exact 9-layer anatomical rigging with true anatomical joint/crystal pivots:
      1. `giant_ruby` (massive red ruby crystal shield — reconstructed as a complete, unbroken 3D crystal
                       polygon behind/around occlusions so `rotate_layer_nearest` can spin, tilt, and slam
                       the giant ruby forward without ever leaving a gap or cutout hole!)
      2. `ear_L` (left pointed purple ear horn)
      3. `ear_R` (right pointed purple ear horn & side spikes)
      4. `arm_far` (far-side purple clawed arm gripping/supporting the ruby)
      5. `leg_L` (left purple leg & 2-clawed foot — 100% grounded `(0,0)` in `idle`, goblin stride in `walk`)
      6. `leg_R` (right purple leg & foot — 100% grounded `(0,0)` in `idle`, alternating stride in `walk`)
      7. `torso` (compact purple body, dorsal green/blue/red back gems, and dark crimson mouth interior pocket)
      8. `lower_jaw` (jagged golden-toothed lower jaw & chin — opens `+1px` for sinister cackling/barking in
                      `idle`/`walk` and `+2px` wide for a ferocious roar in `attack` and `hurt`!)
      9. `head_upper` (purple cranium, glowing red gem eye, brow, and upper lip — 100% rigid integer motion
                       `shift_layer_int` so the face and gem eye never deform by a single pixel!)
"""

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import sys

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mouth_open  # noqa: E402  small face-following mouth opening (2026-11-04)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0302_sableye/mega/sableye-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0302_sableye/mega/sableye-mega"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16
VERSION = "sableye_mega_v1m"  # v1m: mouth opens small along the grin (mouth_open.py), no jaw drop / no pocket

# Closed mouth line on native front.png (face looks left): upper-lip edge just above the golden teeth,
# ordered hinge/corner (right, x=66) -> lip tip (left, x=54). The opening is carved below it (into the teeth band),
# so the chin outline (y=55) and both mouth corners stay closed.
MOUTH_LINE_FRONT_NATIVE = [(66.0, 50.0), (64.0, 50.0), (62.0, 51.0), (60.0, 52.0), (54.0, 52.0)]
MOUTH_PEAK = 0.6
MOUTH_DEEP = (0, 0, 0)        # darkest palette colour (outline)
MOUTH_INNER = (155, 46, 64)   # existing dark crimson (the old mouth-interior colour)


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
        | ((r == 57) & (g == 57) & (b == 82))
        | ((r == 91) & (g == 75) & (b == 33))
        | ((r == 155) & (g == 46) & (b == 64))
    )


def is_purple_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 90) & (g == 74) & (b == 148))
        | ((r == 115) & (g == 90) & (b == 172))
        | ((r == 139) & (g == 115) & (b == 213))
        | ((r == 172) & (g == 164) & (b == 246))
    )


def is_ruby_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 204) & (g == 63) & (b == 124))
        | ((r == 255) & (g == 74) & (b == 90))
        | ((r == 255) & (g == 117) & (b == 135))
        | ((r == 255) & (g == 168) & (b == 181))
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
    Pure nearest-neighbor inverse-mapped rotation around anatomical/crystal pivot `pivot_xy = (px, py)`.
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


def segment_front_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `front.png` into 9 anatomical masks:
      1. `giant_ruby`: Massive red ruby shield (`x=18..55, y=15..79`), excluding the red gem eye (`x=56..64, y=43..50`).
      2. `ear_L`: Left pointed purple ear horn (`x=55..59, y=30..35`).
      3. `ear_R`: Right pointed purple ear horn (`x=69..76, y=29..41`).
      4. `arm_far`: Empty on front view (tucked behind the giant ruby).
      5. `leg_L`: Grounded left purple leg & 2-clawed foot (`x=49..57, y=66..78`).
      6. `leg_R`: Grounded right purple leg & foot (`x=58..66, y=65..74`).
      7. `torso`: Compact purple body & chest (`x=53..63, y=55..66`).
      8. `lower_jaw`: Jagged golden teeth & lower jaw outline (`x=54..67, y=51..55`).
      9. `head_upper`: Purple cranium, glowing red gem eye, brow, and side ear spikes (`x=50..76, y=36..52`).
      10. `arm_near`: Foreground 3-clawed purple hand & spiky forearm gripping the edge of the ruby (`x=50..67, y=57..64`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    r = arr96[:, :, 0].astype(np.int32)
    g = arr96[:, :, 1].astype(np.int32)
    b = arr96[:, :, 2].astype(np.int32)

    is_ruby_col = a & (
        ((r == 155) & (g == 46) & (b == 64))
        | ((r == 204) & (g == 63) & (b == 124))
        | ((r == 255) & (g == 74) & (b == 90))
        | ((r == 255) & (g == 117) & (b == 135))
        | ((r == 255) & (g == 168) & (b == 181))
    )
    eye_zone = (yy >= 43) & (yy <= 50) & (xx >= 56) & (xx <= 64)

    # Bottom-left and bottom-right dark outline pixels belonging to `giant_ruby`
    ruby_outline_zone = (
        ((yy == 65) & (xx >= 53) & (xx <= 55))
        | ((yy == 66) & (xx == 55))
        | ((yy == 67) & (xx == 54))
        | ((yy >= 68) & (yy <= 69) & (xx == 53))
        | ((yy == 70) & (xx == 52))
        | ((yy >= 71) & (yy <= 72) & (xx == 51))
        | ((yy == 73) & (xx == 50))
        | ((yy == 74) & (xx == 49))
        | ((yy == 75) & ((xx <= 37) | (xx == 49)))
        | ((yy == 76) & ((xx <= 39) | (xx == 49)))
        | ((yy == 77) & ((xx <= 41) | (xx == 48)))
        | ((yy == 78) & ((xx <= 43) | (xx == 47)))
        | ((yy == 79) & (xx <= 46))
    )
    giant_ruby = a & ((is_ruby_col & (~eye_zone)) | ruby_outline_zone)

    # 2. `arm_near`: foreground 3-clawed hand & spiky forearm gripping the ruby edge (`y=57..64, x=50..67`)
    arm_near_zone = (
        ((yy == 57) & (xx >= 64) & (xx <= 65))
        | ((yy == 58) & (xx >= 50) & (xx <= 67))
        | ((yy >= 59) & (yy <= 61) & (xx >= 52) & (xx <= 66))
        | ((yy == 62) & (xx >= 51) & (xx <= 64))
        | ((yy == 63) & (xx >= 51) & (xx <= 57))
        | ((yy == 64) & (xx >= 53) & (xx <= 56))
    )
    arm_near = a & arm_near_zone & (~giant_ruby)

    # 3. `ear_L` and `ear_R`
    ear_L_zone = (yy >= 30) & (yy <= 35) & (xx >= 55) & (xx <= 59)
    ear_L = a & ear_L_zone & (~giant_ruby)

    ear_R_zone = (yy >= 29) & (yy <= 40) & (xx >= 69) & (xx <= 76)
    ear_R = a & ear_R_zone & (~giant_ruby)

    # 4. `lower_jaw`: jagged golden teeth (`y`, `Y`) + chin outline (`y=51..55, x=54..67`)
    jaw_zone = (
        ((yy == 51) & (xx >= 64) & (xx <= 67))
        | ((yy == 52) & (xx >= 61) & (xx <= 66))
        | ((yy == 53) & (xx >= 54) & (xx <= 64))
        | ((yy == 54) & (xx >= 54) & (xx <= 62))
        | ((yy == 55) & (xx >= 55) & (xx <= 60))
    )
    lower_jaw = a & jaw_zone & (~giant_ruby)

    # 5. `head_upper`: purple cranium, red gem eye, brow, upper lip, and side ear spikes (`y=36..52`)
    head_zone = (
        ((yy >= 36) & (yy <= 50) & (xx >= 50) & (xx <= 76))
        | ((yy == 51) & (((xx >= 54) & (xx <= 63)) | ((xx >= 70) & (xx <= 71))))
        | ((yy == 52) & (xx >= 54) & (xx <= 60))
    )
    head_upper = a & head_zone & (~giant_ruby) & (~ear_L) & (~ear_R) & (~lower_jaw)

    # 6. `leg_L`: left leg & 2-clawed foot (`y=66..78, x=49..57`)
    leg_L_zone = (yy >= 66) & (yy <= 78) & (xx >= 49) & (xx <= 57)
    leg_L = a & leg_L_zone & (~giant_ruby)

    # 7. `leg_R`: right leg & foot (`y=65..74, x=58..66`)
    leg_R_zone = (yy >= 65) & (yy <= 74) & (xx >= 58) & (xx <= 66)
    leg_R = a & leg_R_zone & (~giant_ruby)

    empty = np.zeros_like(a)
    used = giant_ruby | arm_near | ear_L | ear_R | lower_jaw | head_upper | leg_L | leg_R
    torso = a & (~used)

    return {
        "giant_ruby": giant_ruby,
        "ear_L": ear_L,
        "ear_R": ear_R,
        "arm_far": empty,
        "leg_L": leg_L,
        "leg_R": leg_R,
        "torso": torso,
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
    r = arr96[:, :, 0].astype(np.int32)
    g = arr96[:, :, 1].astype(np.int32)
    b = arr96[:, :, 2].astype(np.int32)

    is_ruby_col = a & (
        ((r == 155) & (g == 46) & (b == 64))
        | ((r == 204) & (g == 63) & (b == 124))
        | ((r == 255) & (g == 74) & (b == 90))
        | ((r == 255) & (g == 117) & (b == 135))
        | ((r == 255) & (g == 168) & (b == 181))
    )
    # Small red gem embedded on the center of Sableye's back torso (`y=68..72, x=48..54`)
    back_gem_zone = (yy >= 67) & (yy <= 72) & (xx >= 48) & (xx <= 54)
    # Bottom tip outline of `giant_ruby` peeking between Sableye's legs (`y=74..77, x=49..53`)
    ruby_bottom_tip = (
        ((yy == 74) & (xx >= 50) & (xx <= 52))
        | ((yy == 75) & (xx >= 49) & (xx <= 53))
        | ((yy == 76) & (xx >= 50) & (xx <= 53))
        | ((yy == 77) & (xx == 52))
    )
    giant_ruby = a & ((is_ruby_col & (~back_gem_zone)) | ruby_bottom_tip)

    # 2. `ear_L` (left 3-spiked ear in back view: `x=36..46, y=41..59`)
    ear_L_zone = (
        ((yy >= 41) & (yy <= 49) & (xx >= 36) & (xx <= 45))
        | ((yy >= 50) & (yy <= 55) & (xx >= 36) & (xx <= 45))
        | ((yy >= 56) & (yy <= 59) & (xx >= 38) & (xx <= 46))
    )
    ear_L = a & ear_L_zone & (~giant_ruby)

    # 3. `ear_R` (right ear horn & spikes in back view: `x=58..68, y=40..58`)
    ear_R_zone = (
        ((yy >= 40) & (yy <= 44) & (xx >= 58) & (xx <= 68))
        | ((yy >= 45) & (yy <= 58) & (xx >= 62) & (xx <= 68))
    )
    ear_R = a & ear_R_zone & (~giant_ruby)

    # 4. `head_upper` (purple cranium in back view: `y=44..61, x=45..64`)
    head_zone = (yy >= 44) & (yy <= 61) & (xx >= 46) & (xx <= 65)
    head_upper = a & head_zone & (~giant_ruby) & (~ear_L) & (~ear_R)

    # 5. `arm_far` (left arm holding back of ruby: `y=61..68, x=39..47`)
    arm_far_zone = (
        ((yy == 61) & (xx >= 45) & (xx <= 47))
        | ((yy >= 62) & (yy <= 66) & (xx >= 39) & (xx <= 47))
        | ((yy == 67) & (xx >= 39) & (xx <= 44))
        | ((yy == 68) & (xx >= 41) & (xx <= 42))
    )
    arm_far = a & arm_far_zone & (~giant_ruby)

    # 6. `arm_near` (right clawed arm: `y=63..70, x=58..66`)
    arm_near_zone = (
        ((yy == 63) & (xx >= 60) & (xx <= 64))
        | ((yy >= 64) & (yy <= 67) & (xx >= 59) & (xx <= 66))
        | ((yy >= 68) & (yy <= 69) & (xx >= 61) & (xx <= 66))
        | ((yy == 70) & (xx >= 62) & (xx <= 65))
    )
    arm_near = a & arm_near_zone & (~giant_ruby)

    # 7. `leg_L` (left leg & foot: `y=73..79, x=44..49`)
    leg_L_zone = (yy >= 73) & (yy <= 79) & (xx >= 44) & (xx <= 49)
    leg_L = a & leg_L_zone & (~giant_ruby)

    # 8. `leg_R` (right leg & foot: `y=72..81, x=55..66`)
    leg_R_zone = (
        ((yy >= 72) & (yy <= 75) & (xx >= 55) & (xx <= 64))
        | ((yy >= 76) & (yy <= 81) & (xx >= 58) & (xx <= 66))
    )
    leg_R = a & leg_R_zone & (~giant_ruby)

    empty = np.zeros_like(a)
    used = giant_ruby | ear_L | ear_R | head_upper | arm_far | arm_near | leg_L | leg_R
    torso = a & (~used)

    return {
        "giant_ruby": giant_ruby,
        "ear_L": ear_L,
        "ear_R": ear_R,
        "arm_far": arm_far,
        "leg_L": leg_L,
        "leg_R": leg_R,
        "torso": torso,
        "lower_jaw": empty,
        "head_upper": head_upper,
        "arm_near": arm_near,
    }


def complete_front_ruby_under_hand(base_96: np.ndarray, ruby_mask_96: np.ndarray, hand_mask_96: np.ndarray) -> np.ndarray:
    """
    Completes the 7-row patch (`y=58..64, x=50..55`) of `giant_ruby` on `front.png` that is covered
    by `arm_near` (Sableye's 3-clawed hand), restoring the continuous 1px dark-ruby border `r` `(155, 46, 64)`
    and pink-red crystal facet fill `F` `(255, 117, 135)` / `H` `(255, 168, 181)`.
    When `arm_near` is drawn over `giant_ruby` at rest `(0,0)`, the result is 100% identical to `front.png`,
    and when `giant_ruby` rotates or slams, its right edge has zero claw-shaped holes!
    """
    ruby_96 = np.zeros_like(base_96)
    ruby_96[ruby_mask_96] = base_96[ruby_mask_96]
    # Exact right-edge border x-coordinate of the ruby across y=58..64:
    right_edge_x = {
        58: 54,
        59: 54,
        60: 54,
        61: 53,
        62: 53,
        63: 53,
        64: 54,
    }
    for y, rx in right_edge_x.items():
        for x in range(48, rx + 1):
            if not ruby_mask_96[y, x]:
                if x == rx:
                    ruby_96[y, x] = (155, 46, 64, 255)
                elif y >= 61 and x >= rx - 2:
                    ruby_96[y, x] = (255, 168, 181, 255)
                else:
                    ruby_96[y, x] = (255, 117, 135, 255)
    return ruby_96


def complete_back_ruby_behind_body(base_96: np.ndarray, ruby_mask_96: np.ndarray, body_mask_96: np.ndarray) -> np.ndarray:
    """
    Completes the occluded interior of `giant_ruby` behind Sableye's purple body on `back.png`
    so that rotating or slamming the giant ruby in `N`/`NE`/`NW` views never exposes a cutout hole!
    """
    ruby_96 = np.zeros_like(base_96)
    ruby_96[ruby_mask_96] = base_96[ruby_mask_96]
    for y in range(40, 77):
        # Right boundary of the ruby crystal polygon on back.png:
        if y <= 67:
            rx = 58
        else:
            rx = int(round(58 - (y - 67) * (6.0 / 10.0)))
        # Left boundary is already present in ruby_mask_96
        existing_xs = np.where(ruby_mask_96[y])[0]
        if len(existing_xs) == 0:
            continue
        lx = int(existing_xs.min())
        for x in range(lx, rx + 1):
            if not ruby_mask_96[y, x] and body_mask_96[y, x]:
                if x == rx:
                    ruby_96[y, x] = (155, 46, 64, 255)
                elif y <= 46 and x >= 50:
                    ruby_96[y, x] = (255, 74, 90, 255)
                else:
                    ruby_96[y, x] = (204, 63, 124, 255)
    return ruby_96


def build_rig_layers(
    base_96: np.ndarray,
    is_back: bool,
    flip_x: bool,
    cell_w: int,
    cell_h: int,
    foot_y: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
    """
    Segments native `front.png` or `back.png`, completes occluded crystal facets on `giant_ruby`,
    optionally flips horizontally for SE/E/NW, places onto `(cell_h, cell_w)` canvas with feet aligned
    to `foot_y`, and builds cushioned RGBA layers + crystal/joint pivots.
    """
    raw_masks = segment_back_native(base_96) if is_back else segment_front_native(base_96)

    if not is_back:
        ruby_native = complete_front_ruby_under_hand(base_96, raw_masks["giant_ruby"], raw_masks["arm_near"])
    else:
        body_union = (
            raw_masks["ear_L"]
            | raw_masks["ear_R"]
            | raw_masks["head_upper"]
            | raw_masks["arm_far"]
            | raw_masks["arm_near"]
            | raw_masks["torso"]
            | raw_masks["leg_L"]
            | raw_masks["leg_R"]
        )
        ruby_native = complete_back_ruby_behind_body(base_96, raw_masks["giant_ruby"], body_union)

    src_96 = base_96[:, ::-1, :].copy() if flip_x else base_96.copy()
    ruby_96 = ruby_native[:, ::-1, :].copy() if flip_x else ruby_native.copy()
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

    ruby_canvas = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
    ruby_canvas[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = ruby_96[y0:y1, x0:x1]

    c_masks: Dict[str, np.ndarray] = {}
    for k, m96 in masks_96.items():
        cm = np.zeros((cell_h, cell_w), dtype=bool)
        cm[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = m96[y0:y1, x0:x1]
        c_masks[k] = cm

    purple_fill_m = is_purple_body_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order (back to front):
        # 1. leg_R -> 2. leg_L -> 3. torso -> 4. ear_L -> 5. ear_R -> 6. lower_jaw -> 7. head_upper -> 8. giant_ruby -> 9. arm_near
        layers["arm_far"] = np.zeros_like(canvas)

        # Build leg_R and leg_L with vertical hip stems; give leg_L its complete 1px left black outline `#`
        # along the ruby seam (y=66..76) and zero stray cushion below y=70 so no purple pixel ever pokes under the ruby tip!
        leg_R_stem = build_foot_stem_layer(canvas, c_masks["leg_R"], c_masks["torso"], stem_px=4)
        layers["leg_R"] = leg_R_stem

        leg_L_stem = build_foot_stem_layer(canvas, c_masks["leg_L"], c_masks["torso"], stem_px=4)
        # Also include the 1px shared black outline `#` immediately to the left of leg_L (or right of leg_L when flipped)
        leg_L_dil = ndi.binary_dilation(c_masks["leg_L"], iterations=1)
        shared_leg_border = leg_L_dil & c_masks["giant_ruby"] & is_dark_outline(canvas)
        leg_L_stem[shared_leg_border] = canvas[shared_leg_border]
        layers["leg_L"] = leg_L_stem

        # `torso` cushioned under `giant_ruby`, `arm_near`, `lower_jaw`, `head_upper`
        fg_over_torso = c_masks["giant_ruby"] | c_masks["arm_near"] | c_masks["lower_jaw"] | c_masks["head_upper"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=purple_fill_m,
        )

        # v1m: no rectangular mouth-interior pocket any more; the jaw never drops and the mouth is carved
        # with mouth_open.open_mouth along MOUTH_LINE_FRONT_NATIVE instead.
        layers["torso"] = torso_lyr

        # `ear_L` and `ear_R` cushioned downward into `head_upper`
        fg_over_ears = c_masks["head_upper"]
        layers["ear_L"] = build_part_layer(
            canvas, c_masks["ear_L"], fg_over_ears, cushion_px=3, allowed_fill_mask=purple_fill_m
        )
        layers["ear_R"] = build_part_layer(
            canvas, c_masks["ear_R"], fg_over_ears, cushion_px=3, allowed_fill_mask=purple_fill_m
        )

        # `lower_jaw` cushioned under `giant_ruby` so leaning out or rotating ruby never gaps
        layers["lower_jaw"] = build_Part_with_ruby_cushion(
            canvas, c_masks["lower_jaw"], c_masks["giant_ruby"], cushion_px=4, allowed_fill_mask=None
        )

        # `head_upper` cushioned under `giant_ruby` with lavender/purple head colors
        layers["head_upper"] = build_Part_with_ruby_cushion(
            canvas, c_masks["head_upper"], c_masks["giant_ruby"], cushion_px=6, allowed_fill_mask=purple_fill_m
        )

        # `giant_ruby` (with completed facet under `arm_near`)
        layers["giant_ruby"] = ruby_canvas

        # `arm_near` (foreground 3-clawed hand gripping the edge of `giant_ruby`)
        layers["arm_near"] = build_part_layer(
            canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0
        )

        pivots = {
            "giant_ruby": to_canvas_pt(48.0, 56.0),  # Hand grip pivot on the ruby shield
            "arm_near": to_canvas_pt(61.0, 60.0),    # Near shoulder/elbow socket
            "arm_far": to_canvas_pt(54.0, 58.0),
            "mouth_line": [to_canvas_pt(mx, my) for mx, my in MOUTH_LINE_FRONT_NATIVE],
        }

    else:
        # Back view draw order (back to front):
        # 1. giant_ruby -> 2. arm_far -> 3. leg_L -> 4. leg_R -> 5. torso -> 6. arm_near -> 7. ear_L -> 8. ear_R -> 9. head_upper
        layers["lower_jaw"] = np.zeros_like(canvas)
        layers["giant_ruby"] = ruby_canvas

        fg_over_afar = c_masks["torso"] | c_masks["head_upper"]
        layers["arm_far"] = build_part_layer(
            canvas, c_masks["arm_far"], fg_over_afar, cushion_px=4, allowed_fill_mask=purple_fill_m
        )

        layers["leg_L"] = build_foot_stem_layer(canvas, c_masks["leg_L"], c_masks["torso"], stem_px=4)
        layers["leg_R"] = build_foot_stem_layer(canvas, c_masks["leg_R"], c_masks["torso"], stem_px=4)

        fg_over_torso = c_masks["head_upper"] | c_masks["arm_near"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        layers["torso"] = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=5,
            allowed_fill_mask=purple_fill_m,
        )

        fg_over_anear = c_masks["head_upper"]
        layers["arm_near"] = build_part_layer(
            canvas, c_masks["arm_near"], fg_over_anear, cushion_px=3, allowed_fill_mask=purple_fill_m
        )

        layers["ear_L"] = build_part_layer(
            canvas, c_masks["ear_L"], c_masks["head_upper"], cushion_px=3, allowed_fill_mask=purple_fill_m
        )
        layers["ear_R"] = build_part_layer(
            canvas, c_masks["ear_R"], c_masks["head_upper"], cushion_px=3, allowed_fill_mask=purple_fill_m
        )
        layers["head_upper"] = build_part_layer(
            canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0
        )

        pivots = {
            "giant_ruby": to_canvas_pt(47.0, 56.0),  # Hand grip pivot on back.png
            "arm_near": to_canvas_pt(58.0, 64.0),    # Right shoulder socket on back.png
            "arm_far": to_canvas_pt(47.0, 63.0),     # Left shoulder socket on back.png
        }

    return layers, pivots


def build_Part_with_ruby_cushion(
    canvas: np.ndarray,
    part_mask: np.ndarray,
    ruby_mask: np.ndarray,
    cushion_px: int = 5,
    allowed_fill_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Builds a body-part layer that lies partially behind `giant_ruby` on `front.png`,
    extending its interior fill `cushion_px` pixels into `ruby_mask` so that when `giant_ruby`
    rotates/tilts or Sableye leans out to roar, no transparent seam ever opens between them.
    """
    return build_part_layer(
        canvas,
        part_mask,
        ruby_mask,
        cushion_px=cushion_px,
        allowed_fill_mask=allowed_fill_mask,
    )


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
        (mid_x - 16 - drift, max(2, top_y + 10 - drift // 2)),
        (mid_x + 16 + drift, max(2, top_y + 12 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (229, 186, 66, 255)
    return out


def synthesize_sableye_mega_frame(
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

    ruby_dx, ruby_dy = 0, 0
    ruby_ang = 0.0

    torso_dx, torso_dy = 0, 0
    head_dx, head_dy = 0, 0
    ear_L_dx, ear_L_dy = 0, 0
    ear_R_dx, ear_R_dy = 0, 0
    jaw_dx, jaw_dy = 0, 0
    mouth_amt = 0
    anear_dx, anear_dy = 0, 0
    afar_dx, afar_dy = 0, 0
    leg_L_dx, leg_L_dy = 0, 0
    leg_R_dx, leg_R_dy = 0, 0

    anear_ang = 0.0
    afar_ang = 0.0

    if action == "idle":
        # Sinister goblin peek-out + cackling/roaring jagged jaw + mystic rotating giant ruby shield!
        # 1. Both feet (`leg_L`, `leg_R`) stay 100% grounded `(0, 0)`.
        # 2. `giant_ruby` slowly pivots/rotates (`±4.5°`) and hovers (`±1px`), gripped by `arm_near`.
        # 3. Sableye's head & upper torso peek out from behind the ruby and let out a menacing cackle/bark
        #    with `lower_jaw` opening `+1..+2px`!
        ruby_dy = int(round(1.0 * math.sin(phase - 0.5)))
        ruby_dx = int(round(0.6 * math.cos(phase - 0.5) * facing))
        ruby_ang = 4.5 * math.sin(phase - 0.3) * facing

        body_dy = int(round(1.2 * math.sin(phase)))
        # Peek slightly out from behind the ruby during the cackle peak (frames 2..7)
        peek_dx = int(round(-1.0 * max(0.0, math.sin(phase)) * facing))

        torso_dx, torso_dy = peek_dx, body_dy
        head_dx, head_dy = peek_dx, body_dy

        ear_perk = -1 if frame_idx in (3, 4, 5) else 0
        ear_L_dx, ear_L_dy = head_dx, head_dy + ear_perk
        ear_R_dx, ear_R_dy = head_dx, head_dy + ear_perk

        # Sinister double-cackle in idle: small 1px opening on frames 2..6, 11, 12 (jaw stays on the head)
        mouth_amt = 1 if frame_idx in (2, 3, 4, 5, 6, 11, 12) else 0
        jaw_dx = head_dx
        jaw_dy = head_dy

        # `arm_near` grips the edge of `giant_ruby` and tracks its rotation/hover
        anear_dx = ruby_dx
        anear_dy = ruby_dy + int(round(0.8 * math.sin(phase - 0.3)))
        anear_ang = 2.5 * math.sin(phase - 0.3) * facing

        afar_dx = peek_dx
        afar_dy = body_dy

    elif action == "walk":
        # Creepy goblin march while tilting & spinning the heavy giant ruby shield + barking jaw!
        bounce = int(round(-1.5 * abs(sin1)))
        sway_x = int(round(1.4 * sin1 * facing))

        ruby_dx = sway_x + int(round(0.8 * cos1 * facing))
        ruby_dy = bounce + int(round(1.0 * cos1))
        ruby_ang = 6.0 * sin1 * facing

        torso_dx, torso_dy = sway_x, bounce
        head_dx = sway_x + int(round(-0.8 * max(0.0, sin1) * facing))
        head_dy = bounce + int(round(0.8 * math.sin(2.0 * phase - 0.3)))

        ear_L_dx, ear_L_dy = head_dx, head_dy
        ear_R_dx, ear_R_dy = head_dx, head_dy

        # Barking / cackling jaw on each step beat
        mouth_amt = 1 if math.sin(2.0 * phase) > 0.1 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy

        anear_dx = ruby_dx
        anear_dy = ruby_dy
        anear_ang = 3.5 * sin1 * facing

        afar_dx = sway_x
        afar_dy = bounce + int(round(-1.0 * cos1))
        afar_ang = -3.0 * sin1 * facing

        leg_L_dx, leg_L_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        leg_R_dx, leg_R_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Mega Sableye "Ferocious Peek-Out Roar + Dynamic Spinning Giant Ruby Bash"!
        # Frames 0..3 (Wind-up & Big Roar):
        #   - Pulls `giant_ruby` back (`-3px`) and tilts it back (`-9°`),
        #   - Sableye leans UP and OUT from behind the ruby (`head_dy = -1`, `peek_dx`),
        #   - Opens `lower_jaw` WIDE (`+2px` ferocious roar!) and perks ears (`-1px`).
        # Frames 4..9 (Dynamic Spinning Giant Ruby Bash & Power Gem Spin):
        #   - Hurls `giant_ruby` forward (`+6px` lunge toward target!) while dynamically spinning/rotating
        #     `giant_ruby` (`+13°` -> `-11°` -> `+14°` -> `-9°` -> `+6°`) around its crystal center pivot!
        #   - `arm_near` and `arm_far` thrust forward with the spinning ruby shield,
        #   - `lower_jaw` stays wide open (`+2px` on frames 4..5, `+1px` on frames 6..8) in a battle roar!
        # Frames 10..15: Smooth follow-through recovery to guarded stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.55 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.18 * math.sin(u * math.pi)

        ruby_lunge_table = [0, -2, -3, -3, 3, 5, 5, 4, 3, 2, 0, -1, -1, 0, 0, 0]
        ruby_dy_table = [0, -1, -1, 0, -1, 1, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0]
        ruby_spin_table = [0.0, -5.0, -8.0, -10.0, 10.0, 13.0, -10.0, 12.0, -8.0, 5.0, -3.0, -1.5, 0.0, 0.0, 0.0, 0.0]

        ruby_dx = int(round(ruby_lunge_table[frame_idx] * facing))
        ruby_dy = ruby_dy_table[frame_idx]
        ruby_ang = ruby_spin_table[frame_idx] * facing

        body_dx = int(round(4.0 * k * facing))
        body_dy = -1 if frame_idx in (1, 2, 3, 4) else (1 if frame_idx in (5, 6, 7) else 0)

        torso_dx, torso_dy = body_dx, body_dy
        # On wind-up (frames 1..3), head peeks out slightly further behind ruby to roar clearly
        head_dx = body_dx + (int(round(-1.0 * facing)) if frame_idx in (1, 2, 3) else 0)
        head_dy = body_dy

        ear_perk = -1 if frame_idx in (1, 2, 3, 4, 5) else 0
        ear_L_dx, ear_L_dy = head_dx, head_dy + ear_perk
        ear_R_dx, ear_R_dy = head_dx, head_dy + ear_perk

        # Big +2px roar during wind-up & strike (frames 1..5), +1px bark during impact hold (frames 6..8)
        if frame_idx in (1, 2, 3, 4, 5):
            mouth_amt = 2
        elif frame_idx in (6, 7, 8):
            mouth_amt = 1
        else:
            mouth_amt = 0
        jaw_dx = head_dx
        jaw_dy = head_dy

        anear_dx = ruby_dx
        anear_dy = ruby_dy
        anear_ang = 5.0 * k * facing

        afar_dx = body_dx + int(round(1.5 * k * facing))
        afar_dy = body_dy
        afar_ang = -4.5 * k * facing

        lead_x = int(round(2.5 * max(0.0, k) * facing)) if k >= 0 else int(round(1.2 * k * facing))
        rear_x = int(round(1.5 * k * facing))
        leg_L_dx, leg_L_dy = lead_x, 0
        leg_R_dx, leg_R_dy = rear_x, 0

    elif action == "hurt":
        # Knockback recoil + hit-shake + ruby shield jolted upward/tilted + pained roar (`+2px` jaw) + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-3.5 * env * facing)) + jit
        body_dy = int(round(-1.0 * env))

        ruby_dx = body_dx + int(round(-0.8 * env * facing))
        ruby_dy = int(round(-2.0 * env))
        ruby_ang = -8.5 * env * facing

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-1.5 * env))

        ear_L_dx, ear_L_dy = head_dx, head_dy - (1 if 2 <= frame_idx <= 6 else 0)
        ear_R_dx, ear_R_dy = head_dx, head_dy - (1 if 2 <= frame_idx <= 6 else 0)

        mouth_amt = 1 if 1 <= frame_idx <= 7 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy

        anear_dx = ruby_dx
        anear_dy = ruby_dy
        anear_ang = -5.0 * env * facing

        afar_dx = body_dx
        afar_dy = body_dy
        afar_ang = 4.0 * env * facing

        slide_x = int(round(body_dx * 0.65))
        leg_L_dx, leg_R_dx = slide_x, slide_x

    ruby_rot = rotate_layer_nearest(layers["giant_ruby"], ruby_ang, pivots["giant_ruby"])
    anear_rot = rotate_layer_nearest(layers["arm_near"], anear_ang, pivots["arm_near"])
    afar_rot = rotate_layer_nearest(layers["arm_far"], afar_ang, pivots["arm_far"])

    if not is_back:
        comp = [
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["ear_L"], ear_L_dx, ear_L_dy),
            shift_layer_int(layers["ear_R"], ear_R_dx, ear_R_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
        ]
        # Carve the small mouth on the head before the ruby / near arm are laid over it,
        # so a ruby that swings over the face is never cut.
        head_comp = composite_layers(canvas_shape, comp)
        if mouth_amt > 0 and pivots.get("mouth_line"):
            line = mouth_open.transform_points(pivots["mouth_line"], dx=head_dx, dy=head_dy)
            head_comp = mouth_open.open_mouth(
                head_comp, line, amount=mouth_amt, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=MOUTH_PEAK
            )
        comp = [
            head_comp,
            shift_layer_int(ruby_rot, ruby_dx, ruby_dy),
            shift_layer_int(anear_rot, anear_dx, anear_dy),
        ]
    else:
        comp = [
            shift_layer_int(ruby_rot, ruby_dx, ruby_dy),
            shift_layer_int(afar_rot, afar_dx, afar_dy),
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(anear_rot, anear_dx, anear_dy),
            shift_layer_int(layers["ear_L"], ear_L_dx, ear_L_dy),
            shift_layer_int(layers["ear_R"], ear_R_dx, ear_R_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gifs", action="store_true", help="do not rewrite front/back(.reverse).gif")
    ap.add_argument("--no-sync", action="store_true", help="do not copy to pokemon_arena_2d")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    front_raw = np.array(Image.open(MEGA_DIR / "front.png").convert("RGBA"))
    back_raw = np.array(Image.open(MEGA_DIR / "back.png").convert("RGBA"))

    palette_rgba = extract_palette(front_raw, back_raw)

    # Tight bounds of front (59x65) and back (43x69)
    max_w, max_h = 59, 69
    cell_w = 96
    cell_h = 96
    foot_y = 86

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
                frame_rgba = synthesize_sableye_mega_frame(
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

    if not args.no_gifs:
        save_transparent_gif(front_frames, MEGA_DIR / "front.gif", duration_ms=75)
        save_transparent_gif(front_rev_frames, MEGA_DIR / "front-reverse.gif", duration_ms=75)
        save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
        save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)

    meta = {
        "generator": "forge_sableye_mega_actions",
        "key": "0302_sableye-mega",
        "version": VERSION,
        "dex": 302,
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
    if SYNC_MEGA_DIR.exists() and not args.no_sync:
        SYNC_OUT_DIR.mkdir(parents=True, exist_ok=True)
        for f_path in OUT_DIR.iterdir():
            if f_path.is_file():
                shutil.copy2(f_path, SYNC_OUT_DIR / f_path.name)
        if not args.no_gifs:
            for gif_name in ("front.gif", "front-reverse.gif", "back.gif", "back-reverse.gif"):
                shutil.copy2(MEGA_DIR / gif_name, SYNC_MEGA_DIR / gif_name)

    print(f"Built 0302_sableye-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
