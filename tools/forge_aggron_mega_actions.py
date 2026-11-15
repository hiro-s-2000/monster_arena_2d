#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Aggron (`0306_aggron-mega`) modeled after Normal Aggron's official GIF
(`0306_aggron/normal/front.gif`) and Mega Aggron's steel-fortress Kaiju anatomy (`front.png` & `back.png`):
  - Exact anatomical rigging with true joint pivots (shoulder/elbow sockets & tail socket):
      1. `tail` (massive segmented steel tail — 100% separated from both feet, pivoted at the tail socket
                 so the tail base never detaches while the segmented steel rings whip and spin!)
      2. `shoulder_armor_far` (far-side dark-iron curved shoulder horn & pauldron)
      3. `arm_far` (far-side white spiked steel gauntlet & 3 claws — pivoted at the far arm socket)
      4. `leg_L` (left armored shin & 3-clawed foot — 100% grounded `(0,0)` in `idle`, heavy tank stomp in `walk`)
      5. `torso` (central silver & dark-iron chest/belly armor plates, back carapace, thighs, and dark-red mouth pocket)
      6. `leg_R` (right armored shin & 3-clawed foot — 100% grounded `(0,0)` in `idle`, alternating stomp in `walk`)
      7. `lower_jaw` (steel lower jaw, red tongue, and long pointed chin-spike — opens `+1px` for deep metallic
                      roar/bark in `idle`/`walk` and `+2px` wide for a thunderous Kaiju roar in `attack` & `hurt`!)
      8. `head_upper` (white armored helmet, 3 steel horns, fierce blue eye, and upper jaw teeth — 100% rigid
                       integer motion `shift_layer_int` so the face, eye, and horns never deform by a single pixel)
      9. `shoulder_armor_near` / `arm_near` (foreground giant curved dark-iron shoulder pauldron + white spiked
                       steel claw gauntlet — pivoted at the shoulder/elbow sockets for spinning Heavy Slam /
                       Metal Claw lariat smashes in `attack`!)
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
import mouth_open  # noqa: E402  (small face-following mouth opening, 2026-11-04)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0306_aggron/mega/aggron-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0306_aggron/mega/aggron-mega"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16

# 2026-11-04 mouth fix: the jaw layer no longer drops and no rectangular crimson pocket is painted.
# The mouth opens as a small lens along the closed lower-lip line of the front art (mouth_open.py).
# Coordinates are in the native 96x96 `front.png` (unflipped), ordered hinge/corner (back of head) -> lip tip.
MOUTH_LINE_SRC = [(32, 37), (31, 38), (30, 39), (29, 40), (28, 40), (27, 39)]
MOUTH_INK = (16, 16, 16)        # darkest palette colour (outline)
MOUTH_INNER = (156, 49, 65)     # existing dark-red mouth interior colour


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
        | ((r == 32) & (g == 32) & (b == 41))
    )


def is_steel_armor_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 205) & (g == 205) & (b == 205))
        | ((r == 172) & (g == 172) & (b == 172))
        | ((r == 164) & (g == 164) & (b == 172))
        | ((r == 139) & (g == 139) & (b == 139))
        | ((r == 131) & (g == 131) & (b == 148))
        | ((r == 106) & (g == 106) & (b == 106))
        | ((r == 90) & (g == 90) & (b == 98))
        | ((r == 82) & (g == 82) & (b == 74))
        | ((r == 57) & (g == 57) & (b == 57))
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
    Segments native 96x96 `front.png` into exact anatomical masks following 1px outlines:
      1. `tail`: Segmented steel tail extending to the right (`x=78..92, y=39..61`).
      2. `shoulder_armor_far`: Far-side dark-iron shoulder horn rising behind the left side of the helmet (`x=16..24, y=13..50`).
      3. `arm_far`: Far-side left white spiked steel gauntlet & 3 claws (`x=2..23, y=41..75`).
      4. `leg_L`: Grounded left armored shin & 3-clawed foot (`x=16..42, y=68..82`).
      5. `torso`: Central silver & dark-iron chest/belly armor plates, upper thighs, and back carapace.
      6. `leg_R`: Grounded right armored shin & 3-clawed foot (`x=59..85, y=69..87`).
      7. `lower_jaw`: Steel lower jaw, red tongue, and long pointed chin-spike (`x=24..36, y=37..53`).
      8. `head_upper`: White armored helmet, 3 horns, fierce blue eye, and upper jaw teeth (`x=11..51, y=11..40`).
      9. `shoulder_armor_near`: Giant curved dark-iron shoulder pauldron & tall horn (`x=46..68, y=8..53`).
     10. `arm_near`: Foreground right white spiked steel gauntlet & 3 claws (`x=60..80, y=33..71`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Tail (segmented steel tail behind right arm)
    tail_zone = (
        ((yy >= 39) & (yy <= 45) & (xx >= 79))
        | ((yy >= 46) & (yy <= 52) & (xx >= 78))
        | ((yy >= 53) & (yy <= 60) & (xx >= 80))
        | ((yy == 61) & (xx >= 81))
    )
    tail = a & tail_zone

    # 2. Arm near (foreground right white spiked gauntlet + 3 claws)
    arm_near_zone = (
        ((yy >= 33) & (yy <= 38) & (xx >= 71) & (xx <= 77))
        | ((yy == 39) & (xx >= 70) & (xx <= 77))
        | ((yy == 40) & (xx >= 69) & (xx <= 77))
        | ((yy == 41) & (xx >= 68) & (xx <= 77))
        | ((yy == 42) & (xx >= 67) & (xx <= 77))
        | ((yy == 43) & (xx >= 66) & (xx <= 77))
        | ((yy == 44) & (xx >= 64) & (xx <= 78))
        | ((yy == 45) & (xx >= 63) & (xx <= 78))
        | ((yy >= 46) & (yy <= 49) & (xx >= 62) & (xx <= 78))
        | ((yy >= 50) & (yy <= 53) & (xx >= 61) & (xx <= 79))
        | ((yy >= 54) & (yy <= 55) & (xx >= 61) & (xx <= 80))
        | ((yy >= 56) & (yy <= 57) & (xx >= 60) & (xx <= 80))
        | ((yy == 58) & (xx >= 60) & (xx <= 80))
        | ((yy == 59) & (xx >= 60) & (xx <= 79))
        | ((yy == 60) & (xx >= 61) & (xx <= 79))
        | ((yy >= 61) & (yy <= 63) & (xx >= 61) & (xx <= 79))
        | ((yy == 64) & (xx >= 62) & (xx <= 79))
        | ((yy == 65) & (xx >= 63) & (xx <= 79))
        | ((yy == 66) & (xx >= 65) & (xx <= 79))
        | ((yy == 67) & (xx >= 67) & (xx <= 79))
        | ((yy == 68) & (xx >= 68) & (xx <= 78))
        | ((yy == 69) & (xx >= 69) & (xx <= 78))
        | ((yy == 70) & (xx >= 70) & (xx <= 77))
        | ((yy == 71) & (xx >= 71) & (xx <= 75))
    )
    arm_near = a & arm_near_zone & (~tail)

    # 3. Shoulder armor near (giant curved dark-iron shoulder pauldron + tall blade horn)
    sh_near_zone = (
        ((yy >= 8) & (yy <= 24) & (xx >= 58) & (xx <= 69))
        | ((yy >= 25) & (yy <= 29) & (xx >= 58) & (xx <= 67))
        | ((yy >= 30) & (yy <= 32) & (xx >= 57) & (xx <= 65))
        | ((yy == 33) & (xx >= 56) & (xx <= 64))
        | ((yy == 34) & (xx >= 49) & (xx <= 63))
        | ((yy == 35) & (xx >= 47) & (xx <= 63))
        | ((yy == 36) & (xx >= 47) & (xx <= 62))
        | ((yy == 37) & (xx >= 46) & (xx <= 62))
        | ((yy >= 38) & (yy <= 39) & (xx >= 47) & (xx <= 62))
        | ((yy == 40) & (xx >= 48) & (xx <= 63))
        | ((yy >= 41) & (yy <= 42) & (xx >= 47) & (xx <= 61))
        | ((yy >= 43) & (yy <= 46) & (xx >= 47) & (xx <= 60))
        | ((yy >= 47) & (yy <= 48) & (xx >= 48) & (xx <= 60))
        | ((yy == 49) & (xx >= 49) & (xx <= 60))
        | ((yy >= 50) & (yy <= 51) & (xx >= 50) & (xx <= 60))
        | ((yy == 52) & (xx >= 50) & (xx <= 58))
        | ((yy == 53) & (xx >= 53) & (xx <= 56))
    )
    shoulder_armor_near = a & sh_near_zone & (~arm_near)

    # 4. Lower jaw (steel lower jaw, red tongue, and long pointed chin spike)
    jaw_zone = (
        ((yy == 37) & (xx >= 27) & (xx <= 36))
        | ((yy == 38) & (xx >= 25) & (xx <= 35))
        | ((yy == 39) & (xx >= 25) & (xx <= 34))
        | ((yy == 40) & (xx >= 24) & (xx <= 33))
        | ((yy == 41) & (xx >= 24) & (xx <= 32))
        | ((yy == 42) & (xx >= 24) & (xx <= 30))
        | ((yy == 43) & (xx >= 25) & (xx <= 30))
        | ((yy >= 44) & (yy <= 45) & (xx >= 25) & (xx <= 29))
        | ((yy >= 46) & (yy <= 52) & (xx >= 26) & (xx <= 29))
        | ((yy == 53) & (xx >= 27) & (xx <= 28))
    )
    lower_jaw = a & jaw_zone

    # 5. Head upper (white armored helmet, 3 horns, fierce blue eye, upper jaw teeth & upper mouth roof)
    head_zone = (
        ((yy >= 11) & (yy <= 21) & (xx >= 19) & (xx <= 52))
        | ((yy >= 22) & (yy <= 25) & (xx >= 11) & (xx <= 51))
        | ((yy == 26) & (xx >= 16) & (xx <= 49))
        | ((yy >= 27) & (yy <= 29) & (xx >= 19) & (xx <= 48))
        | ((yy >= 30) & (yy <= 32) & (xx >= 20) & (xx <= 46))
        | ((yy >= 33) & (yy <= 34) & (xx >= 21) & (xx <= 44))
        | ((yy == 35) & (xx >= 21) & (xx <= 43))
        | ((yy == 36) & (xx >= 22) & (xx <= 42))
        | ((yy >= 37) & (yy <= 38) & (xx >= 22) & (xx <= 40))
        | ((yy == 39) & (xx >= 23) & (xx <= 38))
        | ((yy == 40) & (xx >= 24) & (xx <= 35))
    )
    head_upper = a & head_zone & (~lower_jaw) & (~shoulder_armor_near)

    # 6. Arm far (left white spiked steel gauntlet & 3 claws)
    arm_far_zone = (
        ((yy >= 41) & (yy <= 49) & (xx <= 11))
        | ((yy == 50) & (xx <= 16))
        | ((yy >= 51) & (yy <= 52) & (xx <= 19))
        | ((yy >= 53) & (yy <= 56) & (xx <= 21))
        | ((yy >= 57) & (yy <= 75) & (xx <= 20))
    )
    arm_far = a & arm_far_zone & (~head_upper) & (~lower_jaw)

    # 7. Shoulder armor far (far-side dark-iron shoulder horn + upper arm behind helmet)
    sh_far_zone = (
        ((yy >= 13) & (yy <= 42) & (xx <= 24))
        | ((yy >= 43) & (yy <= 52) & (xx >= 12) & (xx <= 24))
    )
    shoulder_armor_far = a & sh_far_zone & (~head_upper) & (~lower_jaw) & (~arm_far)

    # 8. Leg L (grounded left lower shin, ankle & 3-clawed foot sliced cleanly across uniform silver shin at y=71)
    leg_L_zone = (
        ((yy == 71) & (xx >= 26) & (xx <= 34))
        | ((yy == 72) & (xx >= 26) & (xx <= 35))
        | ((yy == 73) & (xx >= 26) & (xx <= 36))
        | ((yy >= 74) & (yy <= 82) & (xx >= 16) & (xx <= 42))
    )
    leg_L = a & leg_L_zone & (~arm_far)

    # 9. Leg R (grounded right lower shin, ankle & 3-clawed foot sliced cleanly across uniform silver shin at y=75)
    leg_R_zone = (
        ((yy >= 75) & (yy <= 76) & (xx >= 65) & (xx <= 85))
        | ((yy >= 77) & (yy <= 87) & (xx >= 59) & (xx <= 85))
    )
    leg_R = a & leg_R_zone & (~arm_near) & (~tail)

    used = (
        tail
        | arm_near
        | shoulder_armor_near
        | lower_jaw
        | head_upper
        | arm_far
        | shoulder_armor_far
        | leg_L
        | leg_R
    )
    torso = a & (~used)

    return {
        "tail": tail,
        "shoulder_armor_far": shoulder_armor_far,
        "arm_far": arm_far,
        "leg_L": leg_L,
        "torso": torso,
        "leg_R": leg_R,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
        "shoulder_armor_near": shoulder_armor_near,
        "arm_near": arm_near,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Head upper (helmet & horns at top y=4..26, x=32..66)
    head_zone = (
        ((yy >= 4) & (yy <= 12) & (xx >= 32) & (xx <= 66))
        | ((yy == 13) & (xx >= 33) & (xx <= 66) & ((xx <= 40) | (xx >= 48)))
        | ((yy == 14) & (xx >= 50) & (xx <= 66))
        | ((yy == 15) & (xx >= 53) & (xx <= 66))
        | ((yy == 16) & (xx >= 55) & (xx <= 66))
        | ((yy >= 17) & (yy <= 18) & (xx >= 57) & (xx <= 66))
        | ((yy == 19) & (xx >= 58) & (xx <= 66))
        | ((yy == 20) & (xx >= 59) & (xx <= 66))
        | ((yy == 21) & (xx >= 60) & (xx <= 66))
        | ((yy == 22) & (xx >= 61) & (xx <= 66))
        | ((yy >= 23) & (yy <= 26) & (xx >= 62) & (xx <= 66))
    )
    head_upper = a & head_zone

    # 2. Shoulder armor far (far-left dark-iron shoulder horn at y=3..24, x=23..30)
    sh_far_zone = (
        ((yy >= 3) & (yy <= 20) & (xx >= 23) & (xx <= 30))
        | ((yy == 21) & (xx >= 24) & (xx <= 26))
        | ((yy == 22) & (xx >= 24) & (xx <= 25))
        | ((yy >= 23) & (yy <= 24) & (xx == 24))
    )
    shoulder_armor_far = a & sh_far_zone

    # 3. Arm far (far-left white spiked gauntlet at y=23..49, strictly outside the carapace outline at x=19)
    arm_far_zone = (
        ((yy >= 23) & (yy <= 30) & (xx <= 12))
        | ((yy == 31) & (xx <= 18))
        | ((yy >= 32) & (yy <= 33) & (xx <= 17))
        | ((yy >= 34) & (yy <= 37) & (xx <= 16))
        | ((yy >= 38) & (yy <= 39) & (xx <= 17))
        | ((yy >= 40) & (yy <= 47) & (xx <= 18))
        | ((yy == 48) & (xx <= 17))
        | ((yy == 49) & (xx <= 16))
    )
    arm_far = a & arm_far_zone

    # 4. Arm near (near-right white spiked gauntlet + 3 claws at y=32..64, x=70..87)
    arm_near_zone = (
        ((yy >= 32) & (yy <= 43) & (xx >= 79))
        | ((yy >= 44) & (yy <= 46) & (xx >= 72))
        | ((yy == 47) & (xx >= 71))
        | ((yy >= 48) & (yy <= 50) & (xx >= 70))
        | ((yy >= 51) & (yy <= 52) & (xx >= 71))
        | ((yy >= 53) & (yy <= 54) & (xx >= 72))
        | ((yy >= 55) & (yy <= 56) & (xx >= 73))
        | ((yy >= 57) & (yy <= 58) & (xx >= 74))
        | ((yy == 59) & (xx >= 75))
        | ((yy >= 60) & (yy <= 64) & (xx >= 77))
    )
    arm_near = a & arm_near_zone

    # 5. Shoulder armor near (near-right giant dark-iron shoulder pauldron + tall horn)
    sh_near_zone = (
        ((yy >= 6) & (yy <= 25) & (xx >= 70) & (xx <= 81))
        | ((yy == 26) & (xx >= 64) & (xx <= 79))
        | ((yy == 27) & (xx >= 60) & (xx <= 78))
        | ((yy == 28) & (xx >= 58) & (xx <= 77))
        | ((yy >= 29) & (yy <= 30) & (xx >= 56) & (xx <= 76))
        | ((yy >= 31) & (yy <= 39) & (xx >= 55) & (xx <= 76))
        | ((yy == 40) & (xx >= 57) & (xx <= 75))
        | ((yy == 41) & (xx >= 58) & (xx <= 77))
        | ((yy == 42) & (xx >= 59) & (xx <= 78))
        | ((yy == 43) & (xx >= 61) & (xx <= 78))
        | ((yy >= 44) & (yy <= 49) & (xx >= 68) & (xx <= 76))
    )
    shoulder_armor_near = a & sh_near_zone & (~arm_near) & (~head_upper)

    # 6. Leg L (far-left ankle & foot at y=55..75, x=10..18)
    leg_L_zone = (
        ((yy >= 55) & (yy <= 59) & (xx >= 11) & (xx <= 18))
        | ((yy >= 60) & (yy <= 62) & (xx >= 11) & (xx <= 17))
        | ((yy >= 63) & (yy <= 67) & (xx >= 10) & (xx <= 16))
        | ((yy >= 68) & (yy <= 70) & (xx >= 11) & (xx <= 15))
        | ((yy >= 71) & (yy <= 72) & (xx >= 10) & (xx <= 16))
        | ((yy == 73) & (xx >= 11) & (xx <= 16))
        | ((yy == 74) & (xx >= 13) & (xx <= 16))
        | ((yy == 75) & (xx == 14))
    )
    leg_L = a & leg_L_zone & (~arm_far)

    # 7. Leg R (near-right lower ankle & foot sliced across uniform silver ankle plate at y=73..80, x=50..70)
    leg_R_zone = (yy >= 73) & (yy <= 80) & (xx >= 50) & (xx <= 70)
    leg_R = a & leg_R_zone & (~arm_near)

    # 8. Tail (massive segmented tail extending down-left from the lower back socket at y=50..91)
    tail_zone = (
        ((yy == 50) & (xx >= 27) & (xx <= 33))
        | ((yy == 51) & (xx >= 24) & (xx <= 36))
        | ((yy == 52) & (xx >= 22) & (xx <= 38))
        | ((yy == 53) & (xx >= 20) & (xx <= 40))
        | ((yy >= 54) & (yy <= 58) & (xx >= 19) & (xx <= 40))
        | ((yy >= 59) & (yy <= 63) & (xx >= 18) & (xx <= 40))
        | ((yy >= 64) & (yy <= 69) & (xx >= 16) & (xx <= 40))
        | ((yy == 70) & (xx >= 16) & (xx <= 41))
        | ((yy >= 71) & (yy <= 75) & (xx >= 16) & (xx <= 42))
        | ((yy >= 76) & (yy <= 80) & (xx >= 18) & (xx <= 46))
        | ((yy >= 81) & (yy <= 91) & (xx >= 22) & (xx <= 53))
    )
    tail = a & tail_zone & (~leg_L) & (~leg_R)

    empty = np.zeros_like(a)
    used = (
        head_upper
        | shoulder_armor_far
        | arm_far
        | arm_near
        | shoulder_armor_near
        | leg_L
        | leg_R
        | tail
    )
    torso = a & (~used)

    return {
        "tail": tail,
        "shoulder_armor_far": shoulder_armor_far,
        "arm_far": arm_far,
        "leg_L": leg_L,
        "torso": torso,
        "leg_R": leg_R,
        "lower_jaw": empty,
        "head_upper": head_upper,
        "shoulder_armor_near": shoulder_armor_near,
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
    dst_y0 = (foot_y + 3 - sh) if is_back else (foot_y - sh)

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

    steel_fill_m = is_steel_armor_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order:
        # 1. tail -> 2. shoulder_armor_far -> 3. arm_far -> 4. leg_L -> 5. leg_R
        # -> 6. torso -> 7. lower_jaw -> 8. head_upper -> 9. shoulder_armor_near -> 10. arm_near
        fg_over_tail = c_masks["torso"] | c_masks["arm_near"] | c_masks["leg_R"]
        u_tail = ndi.binary_erosion(c_masks["tail"] | fg_over_tail, iterations=1)
        layers["tail"] = build_part_layer(
            canvas,
            c_masks["tail"],
            u_tail & fg_over_tail,
            cushion_px=6,
            allowed_fill_mask=steel_fill_m,
        )

        fg_over_sh_far = c_masks["torso"] | c_masks["head_upper"] | c_masks["lower_jaw"] | c_masks["arm_far"]
        u_sh_far = ndi.binary_erosion(c_masks["shoulder_armor_far"] | fg_over_sh_far, iterations=1)
        layers["shoulder_armor_far"] = build_part_layer(
            canvas,
            c_masks["shoulder_armor_far"],
            u_sh_far & fg_over_sh_far,
            cushion_px=5,
            allowed_fill_mask=steel_fill_m,
        )

        fg_over_afar = c_masks["torso"] | c_masks["leg_L"]
        u_afar = ndi.binary_erosion(c_masks["arm_far"] | fg_over_afar, iterations=1)
        layers["arm_far"] = build_part_layer(
            canvas,
            c_masks["arm_far"],
            u_afar & fg_over_afar,
            cushion_px=5,
            allowed_fill_mask=steel_fill_m,
        )

        layers["leg_L"] = build_foot_stem_layer(canvas, c_masks["leg_L"], c_masks["torso"], stem_px=4)
        layers["leg_R"] = build_foot_stem_layer(canvas, c_masks["leg_R"], c_masks["torso"], stem_px=4)

        # IMPORTANT: `torso` cushion ONLY expands under foreground layers drawn ON TOP OF `torso`
        # (`lower_jaw`, `head_upper`, `shoulder_armor_near`, `arm_near`), NEVER over `tail` or `arm_far`!
        fg_over_torso = (
            c_masks["lower_jaw"]
            | c_masks["head_upper"]
            | c_masks["shoulder_armor_near"]
            | c_masks["arm_near"]
        )
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=steel_fill_m,
        )

        # (2026-11-04) No rectangular crimson pocket any more: the jaw stays put and the mouth opens as a
        # small lens along the lower-lip line (see `apply_aggron_mouth`).
        layers["torso"] = torso_lyr

        layers["lower_jaw"] = build_part_layer(canvas, c_masks["lower_jaw"], np.zeros_like(c_masks["lower_jaw"]), cushion_px=0)
        layers["head_upper"] = build_part_layer(canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0)

        fg_over_sh_near = c_masks["arm_near"]
        u_sh_near = ndi.binary_erosion(c_masks["shoulder_armor_near"] | fg_over_sh_near, iterations=1)
        layers["shoulder_armor_near"] = build_part_layer(
            canvas,
            c_masks["shoulder_armor_near"],
            u_sh_near & fg_over_sh_near,
            cushion_px=5,
            allowed_fill_mask=steel_fill_m,
        )
        layers["arm_near"] = build_part_layer(canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0)

        pivots = {
            "shoulder_armor_near": to_canvas_pt(55.0, 44.0),
            "arm_near": to_canvas_pt(65.0, 48.0),
            "arm_far": to_canvas_pt(18.0, 52.0),
            "tail": to_canvas_pt(78.0, 52.0),
            "mouth_line": [to_canvas_pt(float(mx), float(my)) for mx, my in MOUTH_LINE_SRC],
        }

    else:
        # Back view draw order:
        # 1. tail -> 2. shoulder_armor_far -> 3. arm_far -> 4. leg_L -> 5. leg_R
        # -> 6. torso -> 7. head_upper -> 8. shoulder_armor_near -> 9. arm_near
        layers["lower_jaw"] = np.zeros_like(canvas)

        fg_over_tail = c_masks["torso"] | c_masks["leg_L"] | c_masks["leg_R"]
        u_tail = ndi.binary_erosion(c_masks["tail"] | fg_over_tail, iterations=1)
        layers["tail"] = build_part_layer(
            canvas,
            c_masks["tail"],
            u_tail & fg_over_tail,
            cushion_px=6,
            allowed_fill_mask=steel_fill_m,
        )

        fg_over_sh_far = c_masks["torso"] | c_masks["head_upper"]
        u_sh_far = ndi.binary_erosion(c_masks["shoulder_armor_far"] | fg_over_sh_far, iterations=1)
        layers["shoulder_armor_far"] = build_part_layer(
            canvas,
            c_masks["shoulder_armor_far"],
            u_sh_far & fg_over_sh_far,
            cushion_px=5,
            allowed_fill_mask=steel_fill_m,
        )

        fg_over_afar = c_masks["torso"] | c_masks["leg_L"]
        u_afar = ndi.binary_erosion(c_masks["arm_far"] | fg_over_afar, iterations=1)
        layers["arm_far"] = build_part_layer(
            canvas,
            c_masks["arm_far"],
            u_afar & fg_over_afar,
            cushion_px=5,
            allowed_fill_mask=steel_fill_m,
        )

        fg_over_legL = c_masks["torso"]
        u_legL = ndi.binary_erosion(c_masks["leg_L"] | fg_over_legL, iterations=1)
        layers["leg_L"] = build_part_layer(
            canvas,
            c_masks["leg_L"],
            u_legL & fg_over_legL,
            cushion_px=4,
            allowed_fill_mask=steel_fill_m,
        )
        layers["leg_R"] = build_foot_stem_layer(canvas, c_masks["leg_R"], c_masks["torso"], stem_px=4)

        # IMPORTANT: `torso` cushion ONLY expands under foreground layers drawn ON TOP OF `torso`
        # (`head_upper`, `shoulder_armor_near`, `arm_near`), NEVER over `tail` or `arm_far`!
        fg_over_torso = c_masks["head_upper"] | c_masks["shoulder_armor_near"] | c_masks["arm_near"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        layers["torso"] = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=steel_fill_m,
        )

        layers["head_upper"] = build_part_layer(canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0)

        fg_over_sh_near = c_masks["arm_near"]
        u_sh_near = ndi.binary_erosion(c_masks["shoulder_armor_near"] | fg_over_sh_near, iterations=1)
        layers["shoulder_armor_near"] = build_part_layer(
            canvas,
            c_masks["shoulder_armor_near"],
            u_sh_near & fg_over_sh_near,
            cushion_px=5,
            allowed_fill_mask=steel_fill_m,
        )
        layers["arm_near"] = build_part_layer(canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0)

        pivots = {
            "shoulder_armor_near": to_canvas_pt(65.0, 38.0),
            "arm_near": to_canvas_pt(73.0, 46.0),
            "arm_far": to_canvas_pt(20.0, 38.0),
            "tail": to_canvas_pt(30.0, 58.0),
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
        (mid_x - 18 - drift, max(2, top_y + 14 - drift // 2)),
        (mid_x + 18 + drift, max(2, top_y + 16 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (254, 254, 254, 255)
    return out


def apply_aggron_mouth(
    frame: np.ndarray, line_cell: List[Tuple[float, float]], amount: int, head_dx: int, head_dy: int
) -> np.ndarray:
    """Small face-following mouth opening (replaces the old jaw drop + rectangular pocket).

    1. carve a lens of outline ink below the lower-lip line (the lip moves down by `amount` px in the middle only,
       both corners stay closed, nothing leaves the silhouette);
    2. the old lower-lip outline pixels inside the lens become the dark-red mouth interior, so the gap reads as
       the open mouth instead of a thicker outline.
    """
    line = mouth_open.transform_points(line_cell, dx=head_dx, dy=head_dy)
    out = mouth_open.open_mouth(frame, line, amount=amount, deep=MOUTH_INK, inner=None, peak=0.6)
    up = mouth_open.transform_points(line, dy=-1)
    red = mouth_open.open_mouth(out, up, amount=1, deep=MOUTH_INNER, inner=None, peak=0.6)
    was_ink = np.all(frame[..., :3] == np.array(MOUTH_INK, dtype=np.uint8), axis=2) & (frame[..., 3] > 0)
    changed = np.any(red != out, axis=2) & was_ink
    out[changed] = red[changed]
    return out


def synthesize_aggron_mega_frame(
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
    sh_far_dx, sh_far_dy = 0, 0
    sh_near_dx, sh_near_dy = 0, 0
    anear_dx, anear_dy = 0, 0
    afar_dx, afar_dy = 0, 0
    tail_dx, tail_dy = 0, 0
    leg_L_dx, leg_L_dy = 0, 0
    leg_R_dx, leg_R_dy = 0, 0

    sh_near_ang = 0.0
    anear_ang = 0.0
    afar_ang = 0.0
    tail_ang = 0.0

    if action == "idle":
        # Steel Fortress Kaiju Breathing + Deep Metallic Roar / Bark + Circular Claw Readiness + Tail Sway:
        # 1. Both armored legs (`leg_L`, `leg_R`) stay 100% planted on the ground `(0, 0)`.
        # 2. Colossal steel torso & horned helmet rise and sink (`-1..+1px`) with heavy mechanical breathing.
        # 3. `lower_jaw` opens `+1px` during the primary roar (`frames 2..7`) and secondary bark (`frames 11..13`),
        #    exposing the dark-red mouth interior pocket `(156, 49, 65)`.
        # 4. Both spiked steel gauntlets (`arm_near`, `arm_far`) rotate (`±3.0°`) and orbit menacingly.
        # 5. The massive segmented steel tail (`tail`) sways and rotates (`±3.0°`) from the tail socket!
        body_dy = int(round(1.2 * math.sin(phase - 0.2)))
        body_dx = int(round(0.6 * max(0.0, math.sin(phase - 0.2)) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        jaw_open = 1 if (frame_idx in (2, 3, 4, 5, 6, 7) or frame_idx in (11, 12)) else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw no longer drops; opening drawn by mouth_open

        sh_far_dx = body_dx
        sh_far_dy = body_dy + int(round(0.8 * math.sin(phase - 0.4)))

        sh_near_dx = body_dx
        sh_near_dy = body_dy + int(round(0.9 * math.sin(phase - 0.4)))
        sh_near_ang = 1.6 * math.sin(phase - 0.4) * (-facing)

        anear_dx = body_dx + int(round(1.2 * math.cos(phase - 0.5) * facing))
        anear_dy = body_dy + int(round(1.4 * math.sin(phase - 0.5)))
        anear_ang = 3.0 * math.sin(phase - 0.5) * (-facing)

        afar_dx = body_dx + int(round(1.2 * math.cos(phase - 0.8) * facing))
        afar_dy = body_dy + int(round(1.4 * math.sin(phase - 0.8)))
        afar_ang = 3.0 * math.sin(phase - 0.8) * facing

        tail_dx = body_dx
        tail_dy = int(round(1.5 * math.sin(phase - 1.1)))
        tail_ang = 2.8 * math.sin(phase - 1.1) * facing

    elif action == "walk":
        # Earth-Shaking Bipedal Fortress Stomp Gait:
        # Alternating left/right armored leg stomp (`lift_px=2, stride_px=2`),
        # 2-beat heavy upper-body stomp bounce, counter-rotating spiked gauntlets (`±4.5°` rotation),
        # rhythmic metallic roar/bark (`+1px`) on each stomp, and heavy counter-rotating steel tail!
        bounce = int(round(-1.5 * abs(sin1)))
        sway_x = int(round(1.0 * sin1 * facing))

        torso_dx, torso_dy = sway_x, bounce
        head_dx = sway_x + int(round(0.6 * cos1 * facing))
        head_dy = bounce

        jaw_open = 1 if math.sin(2.0 * phase) > 0.15 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw no longer drops; opening drawn by mouth_open

        sh_far_dx = sway_x
        sh_far_dy = bounce + int(round(0.9 * math.sin(2.0 * phase - 0.4)))

        sh_near_dx = sway_x
        sh_near_dy = bounce + int(round(1.0 * math.sin(2.0 * phase - 0.4)))
        sh_near_ang = 2.0 * sin1 * (-facing)

        anear_dx = sway_x + int(round(1.8 * sin1 * facing))
        anear_dy = bounce + int(round(1.4 * cos1))
        anear_ang = 4.2 * sin1 * (-facing)

        afar_dx = sway_x + int(round(-1.8 * sin1 * facing))
        afar_dy = bounce + int(round(-1.4 * cos1))
        afar_ang = -4.2 * sin1 * facing

        tail_dx = sway_x
        tail_dy = bounce + int(round(1.6 * math.sin(2.0 * phase - 1.0)))
        tail_ang = 3.2 * math.sin(2.0 * phase - 1.0) * facing

        if is_back:
            leg_L_dx, leg_L_dy = foot_step_offset(phase, facing, lift_px=1, stride_px=1)
        else:
            leg_L_dx, leg_L_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        leg_R_dx, leg_R_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Mega Aggron "Thunderous Kaiju Roar + Spinning Heavy Slam / Metal Claw Lariat & Tail Whip Crush"!
        # Frames 0..3 (Wind-up & Thunderous Kaiju Roar):
        #   - Braces legs, pulls colossal steel body back (`-3px`), flares both giant shoulder horns (`-1px`),
        #   - Opens `lower_jaw` wide (`+2px` thunderous Kaiju roar!),
        #   - Cocks both spiked steel gauntlets high at the shoulder/elbow sockets (`-8°` rotation).
        # Frames 4..9 (Spinning Heavy Bomber Lariat & Steel Tail Whip Smash):
        #   - Explosive forward fortress charge (`+5px` body, `+6px` head horns & spiked gauntlets, lead foot `+3px`),
        #   - High-torque spinning Heavy Slam / Metal Claw lariat (`+9° -> -7° -> +8°` pivot rotation + circular smash arc),
        #   - Giant segmented steel tail executes a rapid wrecking-ball spin oscillation (`±5.5°` around tail socket),
        #   - Jaws clamp shut on Heavy Slam impact (`0px` at frames 6..7) and unleash a follow-through roar (`+1..2px` at 8..9)!
        # Frames 10..15: Heavy follow-through recovery to fortress stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.65 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        body_dx = int(round(4.8 * k * facing))
        if frame_idx in (1, 2, 3):
            body_dy = -1 if frame_idx in (2, 3) else 0
        elif frame_idx in (5, 6, 7, 8):
            body_dy = 1
        else:
            body_dy = 0

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = int(round(5.5 * k * facing))
        head_dy = body_dy

        # Thunderous +2px wide roar on wind-up & early lunge (frames 1..5), snap shut on impact (6..7), follow-through roar (8..9)
        if frame_idx in (2, 3, 4):
            jaw_open = 2
        elif frame_idx in (1, 5, 8, 9):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw no longer drops; opening drawn by mouth_open

        # Giant shoulder pauldrons flare and drive into the Heavy Slam
        sh_far_dx = body_dx
        sh_far_dy = body_dy - (1 if frame_idx in (2, 3, 5, 6, 7) else 0)

        sh_near_dx = int(round(5.0 * k * facing))
        sh_near_dy = body_dy - (1 if frame_idx in (2, 3, 5, 6) else 0)
        sh_spin_table = [0.0, -1.5, -2.5, -3.0, 2.5, 3.5, -2.5, 3.0, -2.0, 1.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        sh_near_ang = sh_spin_table[frame_idx] * (-facing)

        # Spiked steel claw gauntlets: cock back on wind-up, then slam forward (+6px) with a fierce rotational lariat spin!
        anear_dx = int(round(6.0 * k * facing))
        anear_dy = body_dy + (int(round(-1.5 * k)) if frame_idx in (4, 5) else int(round(1.8 * max(0.0, k))))
        afar_dx = int(round(5.8 * k * facing))
        afar_dy = body_dy + int(round(1.6 * k))

        spin_table_near = [0.0, -4.0, -6.5, -8.5, 6.0, 9.0, -7.0, 8.0, -5.0, 3.5, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        spin_table_far = [0.0, 4.0, 6.5, 8.5, -6.0, -9.0, 7.0, -8.0, 5.0, -3.5, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        anear_ang = spin_table_near[frame_idx] * (-facing)
        afar_ang = spin_table_far[frame_idx] * facing

        # Segmented steel tail: whips and spins rapidly around the tail socket during strike frames (4..9)!
        tail_spin_ang = [0.0, -2.0, -3.5, -5.0, 5.0, -5.5, 5.5, -5.0, 4.0, -2.5, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        tail_whip_dy = [0, 1, 1, 2, -2, 2, -2, 2, -1, 1, 0, 0, 0, 0, 0, 0]
        tail_dx = body_dx
        tail_dy = body_dy + tail_whip_dy[frame_idx]
        tail_ang = tail_spin_ang[frame_idx] * facing

        # Lead armored foot stomps powerfully into the lunge
        lead_x = int(round(3.2 * max(0.0, k) * facing)) if k >= 0 else int(round(1.8 * k * facing))
        rear_x = int(round(2.4 * k * facing))
        if not is_back:
            leg_L_dx, leg_L_dy = lead_x, 0
            leg_R_dx, leg_R_dy = rear_x, 0
        else:
            leg_R_dx, leg_R_dy = lead_x, 0
            leg_L_dx, leg_L_dy = rear_x, 0

    elif action == "hurt":
        # Heavy Steel Fortress knockback + hit-shake + pained Kaiju roar (`lower_jaw` open +1..2px) + gauntlets flung back + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-3.5 * env * facing)) + jit
        body_dy = int(round(-1.0 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-1.2 * env))

        jaw_open = 1 if 1 <= frame_idx <= 7 else 0  # hurt growl: small (1 px) opening only
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw no longer drops; opening drawn by mouth_open

        sh_far_dx = body_dx
        sh_far_dy = body_dy - (1 if 2 <= frame_idx <= 5 else 0)

        sh_near_dx = body_dx
        sh_near_dy = body_dy - (1 if 2 <= frame_idx <= 5 else 0)
        sh_near_ang = -2.5 * env * (-facing)

        anear_dx = body_dx + int(round(-1.2 * env * facing))
        anear_dy = body_dy + int(round(-1.5 * env))
        anear_ang = -5.5 * env * (-facing)

        afar_dx = body_dx + int(round(-1.2 * env * facing))
        afar_dy = body_dy + int(round(-1.5 * env))
        afar_ang = 5.5 * env * facing

        tail_dx = body_dx
        tail_dy = int(round(1.2 * env))
        tail_ang = -3.5 * env * facing

        slide_x = int(round(body_dx * 0.65))
        leg_L_dx, leg_R_dx = slide_x, slide_x

    tail_rot = rotate_layer_nearest(layers["tail"], tail_ang, pivots["tail"])
    afar_rot = rotate_layer_nearest(layers["arm_far"], afar_ang, pivots["arm_far"])
    sh_near_rot = rotate_layer_nearest(layers["shoulder_armor_near"], sh_near_ang, pivots["shoulder_armor_near"])
    anear_rot = rotate_layer_nearest(layers["arm_near"], anear_ang, pivots["arm_near"])

    if not is_back:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(layers["shoulder_armor_far"], sh_far_dx, sh_far_dy),
            shift_layer_int(afar_rot, afar_dx, afar_dy),
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(sh_near_rot, sh_near_dx, sh_near_dy),
            shift_layer_int(anear_rot, anear_dx, anear_dy),
        ]
    else:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(layers["shoulder_armor_far"], sh_far_dx, sh_far_dy),
            shift_layer_int(afar_rot, afar_dx, afar_dy),
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(sh_near_rot, sh_near_dx, sh_near_dy),
            shift_layer_int(anear_rot, anear_dx, anear_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    if not is_back and jaw_open > 0 and "mouth_line" in pivots:
        out = apply_aggron_mouth(out, pivots["mouth_line"], jaw_open, head_dx, head_dy)
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

    # Tight bounds of front (91x80) and back (80x89)
    max_w, max_h = 91, 89
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
                frame_rgba = synthesize_aggron_mega_frame(
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
        "generator": "forge_aggron_mega_actions",
        "key": "0306_aggron-mega",
        "dex": 306,
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

    print(f"Built 0306_aggron-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
