#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Swampert (`0260_swampert-mega`) modeled after Normal Swampert's official GIF
(`0260_swampert/normal/front.gif`) and Mega Swampert's anatomical design (`front.png` & `back.png`):
  - Exact 10-layer anatomical rigging with true anatomical joint pivots (shoulder sockets & tail base):
      1. `tail_fin` (giant black fan-shaped tail fin — 100% separated from legs and arms, flexing and
                     whipping powerfully during breathing, stomps, and spinning Hammer Arm attacks!)
      2. `head_fins` (twin tall black dorsal head-fins — bristle and vibrate during roars and lunges)
      3. `arm_far` (far-side colossal muscular arm with orange nodules & grey knuckles — pivoted at the
                    far shoulder socket with `rotate_layer_nearest` for heavy wind-up and spinning smash!)
      4. `leg_L` (left hind leg & 3-toed foot — 100% grounded `(0,0)` in `idle`, heavy kaiju stomp in `walk`)
      5. `torso` (massive blue muscular upper body, dorsal orange bumps, white belly, bottom tail-base ridge,
                  and dark red-brown mouth interior pocket when `lower_jaw` opens)
      6. `leg_R` (right hind leg & 3-toed foot — 100% grounded `(0,0)` in `idle`, alternating stomp in `walk`)
      7. `cheek_gills` (left & right orange 3-pronged cheek gills — tremble and flare during roars!)
      8. `lower_jaw` (white lower jaw — opens `+1px` for rhythmic roaring/barking in `idle`/`walk` and
                      `+2px` wide for a ferocious Kaiju roar during `attack` wind-up, follow-through, and `hurt`!)
      9. `head_upper` (upper snout, cranium, and fierce yellow-orange eyes — 100% rigid integer motion
                       `shift_layer_int` so the eyes and face never deform by a single pixel)
     10. `arm_near` (foreground colossal muscular arm with giant orange plates & 3 grey fist knuckles —
                     pivoted at the near shoulder socket with `rotate_layer_nearest` for dynamic spinning
                     Hammer Arm / Waterfall crashes!)
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
import mouth_open  # noqa: E402  (small face-following mouth opening, 2026-11-04)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0260_swampert/mega/swampert-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0260_swampert/mega/swampert-mega"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16

# 2026-11-04 mouth fix: the lower jaw no longer drops (it read as a rectangle + jutting chin).
# Closed mouth line on native front.png (hinge at the cheek -> lip tip), measured on the still.
MOUTH_LINE_NATIVE = [(39.0, 57.0), (36.0, 57.0), (35.0, 58.0), (34.0, 59.0)]
MOUTH_DEEP = (24, 24, 24)       # outline colour (darkest palette colour)
MOUTH_INNER = (142, 54, 37)     # existing dark red-brown (mouth interior), only shown at a 2 px roar
# Only face / jaw pixels may be carved (never the orange gills or arms drawn over the head)
MOUTH_CARVABLE = {(131, 164, 222), (156, 197, 222), (222, 238, 230), (172, 189, 180), (106, 123, 115), (24, 24, 24)}


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
        ((r == 24) & (g == 24) & (b == 24))
        | ((r == 142) & (g == 54) & (b == 37))
    )


def is_blue_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 90) & (g == 131) & (b == 197))
        | ((r == 131) & (g == 164) & (b == 222))
        | ((r == 90) & (g == 90) & (b == 139))
    )


def is_fin_grey_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 82) & (g == 74) & (b == 82))
        | ((r == 106) & (g == 106) & (b == 139))
    )


def is_orange_gill_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 232) & (g == 119) & (b == 71))
        | ((r == 209) & (g == 79) & (b == 64))
        | ((r == 142) & (g == 54) & (b == 37))
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
    Segments native 96x96 `front.png` into 10 exact anatomical masks following 1px outlines:
      1. `tail_fin`: Giant black fan-shaped tail fin behind the right back (`x=70..86, y=24..68`).
      2. `head_fins`: Upper crowns of the twin black dorsal head-fins (`x=20..58, y=14..26`).
      3. `arm_far`: Far-side left colossal muscular arm with orange nodules & 3 grey knuckles (`x=10..30, y=52..75`).
      4. `leg_L`: Grounded left hind leg & 3-toed foot (`x=40..52, y=72..79`).
      5. `torso`: Massive blue muscular back/chest, dorsal orange bumps, white belly, and tail-base underside.
      6. `leg_R`: Grounded right hind leg & 3-toed foot (`x=67..80, y=72..80`).
      7. `cheek_gills`: Left (`x=8..22, y=36..51`) & right (`x=39..62, y=20..60`) orange 3-pronged cheek gills.
      8. `lower_jaw`: White lower jaw and chin outline (`x=31..43, y=56..60`).
      9. `head_upper`: Forehead fins, blue snout, and fierce yellow-orange eyes (`x=20..44, y=27..59`).
     10. `arm_near`: Foreground right colossal muscular arm with orange plates & 3 grey knuckles (`x=48..78, y=38..71`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    orange_m = is_orange_gill_fill(arr96)

    # 1. Tail fin (giant black fan behind right side, strictly to the right of the 1px black arm/body contour)
    tail_zone = (
        ((yy >= 24) & (yy <= 37) & (xx >= 70))
        | ((yy >= 38) & (yy <= 42) & (xx >= 71))
        | ((yy >= 43) & (yy <= 45) & (xx >= 72))
        | ((yy >= 46) & (yy <= 47) & (xx >= 73))
        | ((yy >= 48) & (yy <= 50) & (xx >= 74))
        | ((yy == 51) & (xx >= 75))
        | ((yy == 52) & (xx >= 76))
        | ((yy >= 53) & (yy <= 59) & (xx >= 77))
        | ((yy == 60) & (xx >= 79))
        | ((yy == 61) & (xx >= 78))
        | ((yy >= 62) & (yy <= 63) & (xx >= 79))
        | ((yy == 64) & (xx >= 78))
        | ((yy >= 65) & (yy <= 68) & (xx >= 79))
    )
    tail_fin = a & tail_zone

    # 2. Cheek gills (left 3-pronged orange gill + right 3-pronged orange gill, excluding dorsal back bump)
    gill_L_zone = (
        ((yy >= 36) & (yy <= 50) & (xx <= 22))
        | ((yy == 51) & (xx >= 17) & (xx <= 22))
        | ((yy == 52) & (xx >= 21) & (xx <= 22))
    )
    gill_R_tip_zone = (yy >= 20) & (yy <= 26) & (xx >= 59) & (xx <= 62)
    gill_R_body_zone = (
        ((yy == 27) & (xx >= 58) & (xx <= 60))
        | ((yy == 28) & (xx >= 55) & (xx <= 57))
        | ((yy == 29) & (xx >= 53) & (xx <= 57))
        | ((yy == 30) & (xx >= 51) & (xx <= 57))
        | ((yy == 31) & (xx >= 50) & (xx <= 56))
        | ((yy == 32) & (xx >= 49) & (xx <= 55))
        | ((yy == 33) & (xx >= 49) & (xx <= 54))
        | ((yy == 34) & (xx >= 49) & (xx <= 53))
        | ((yy >= 35) & (yy <= 37) & (xx >= 48) & (xx <= 52))
        | ((yy == 38) & (xx >= 48) & (xx <= 57))
        | ((yy == 39) & (xx >= 46) & (xx <= 57))
        | ((yy == 40) & (xx >= 45) & (xx <= 56))
        | ((yy >= 41) & (yy <= 42) & (xx >= 43) & (xx <= 55))
        | ((yy >= 43) & (yy <= 44) & (xx >= 42) & (xx <= 55))
        | ((yy >= 45) & (yy <= 46) & (xx >= 41) & (xx <= 54))
        | ((yy >= 47) & (yy <= 48) & (xx >= 40) & (xx <= 50))
        | ((yy == 49) & (xx >= 40) & (xx <= 49))
        | ((yy == 50) & (xx >= 40) & (xx <= 47))
        | ((yy == 51) & (xx >= 40) & (xx <= 50))
        | ((yy >= 52) & (yy <= 53) & (xx >= 40) & (xx <= 49))
        | ((yy == 54) & (xx >= 40) & (xx <= 47))
        | ((yy == 55) & (xx >= 40) & (xx <= 46))
        | ((yy == 56) & (xx >= 41) & (xx <= 46))
        | ((yy >= 57) & (yy <= 58) & (xx >= 42) & (xx <= 45))
        | ((yy == 59) & (xx >= 44) & (xx <= 45))
        | ((yy == 60) & (xx == 44))
    )
    cheek_gills = a & (gill_L_zone | gill_R_tip_zone | (gill_R_body_zone & orange_m))

    # 3. Head fins (upper dorsal crowns of the twin black head-fins)
    head_fins_zone = (yy >= 14) & (yy <= 26) & (xx >= 20) & (xx <= 58)
    head_fins = a & head_fins_zone & (~cheek_gills)

    # 4. Arm far (far-side left colossal muscular arm + orange bumps + 3 grey knuckles)
    arm_far_zone = (
        ((yy == 52) & (xx >= 17) & (xx <= 20))
        | ((yy >= 53) & (yy <= 54) & (xx >= 17) & (xx <= 23))
        | ((yy >= 55) & (yy <= 56) & (xx <= 15) & (xx <= 23))
        | ((yy == 57) & (xx >= 15) & (xx <= 25))
        | ((yy == 58) & (xx >= 14) & (xx <= 26))
        | ((yy == 59) & (xx >= 13) & (xx <= 27))
        | ((yy == 60) & (xx >= 12) & (xx <= 29))
        | ((yy >= 61) & (yy <= 75) & (xx <= 30))
    )
    arm_far = a & arm_far_zone & (~cheek_gills)

    # 5. Lower jaw (white lower jaw + chin outline, strictly excluding upper snout's bottom outline at y=60, x=31..35)
    jaw_zone = (
        ((yy == 56) & (xx == 40))
        | ((yy == 57) & (xx >= 40) & (xx <= 41))
        | ((yy == 58) & (xx >= 36) & (xx <= 41))
        | ((yy == 59) & (xx >= 36) & (xx <= 40))
        | ((yy == 60) & (xx >= 36) & (xx <= 38))
    )
    lower_jaw = a & jaw_zone & (~cheek_gills)

    # 6. Head upper (forehead fins, blue snout, and yellow-orange eyes — 100% rigid, including bottom snout outline y=60, x=31..35)
    head_upper_zone = (
        ((yy == 27) & (xx >= 22) & (xx <= 54))
        | ((yy >= 28) & (yy <= 34) & (xx >= 20) & (xx <= 48))
        | ((yy >= 35) & (yy <= 40) & (xx >= 20) & (xx <= 45))
        | ((yy >= 41) & (yy <= 44) & (xx >= 20) & (xx <= 42))
        | ((yy >= 45) & (yy <= 51) & (xx >= 21) & (xx <= 40))
        | ((yy >= 52) & (yy <= 54) & (xx >= 23) & (xx <= 40))
        | ((yy >= 55) & (yy <= 56) & (xx >= 24) & (xx <= 39))
        | ((yy == 57) & (xx >= 26) & (xx <= 39))
        | ((yy == 58) & (xx >= 27) & (xx <= 35))
        | ((yy == 59) & (xx >= 28) & (xx <= 35))
        | ((yy == 60) & (xx >= 31) & (xx <= 35))
    )
    head_upper = a & head_upper_zone & (~cheek_gills) & (~arm_far) & (~lower_jaw) & (~head_fins)

    # 7. Arm near (foreground right colossal muscular arm + orange plates + 3 grey fist knuckles)
    arm_near_zone = (
        ((yy == 38) & (xx >= 63) & (xx <= 69))
        | ((yy == 39) & (xx >= 59) & (xx <= 69))
        | ((yy == 40) & (xx >= 58) & (xx <= 69))
        | ((yy == 41) & (xx >= 57) & (xx <= 69))
        | ((yy == 42) & (xx >= 56) & (xx <= 70))
        | ((yy == 43) & (xx >= 55) & (xx <= 71))
        | ((yy == 44) & (xx >= 56) & (xx <= 72))
        | ((yy == 45) & (xx >= 55) & (xx <= 72))
        | ((yy == 46) & (xx >= 54) & (xx <= 73))
        | ((yy == 47) & (xx >= 52) & (xx <= 73))
        | ((yy == 48) & (xx >= 51) & (xx <= 74))
        | ((yy == 49) & (xx >= 50) & (xx <= 74))
        | ((yy == 50) & (xx >= 48) & (xx <= 74))
        | ((yy == 51) & (xx >= 51) & (xx <= 75))
        | ((yy >= 52) & (yy <= 53) & (xx >= 50) & (xx <= 76))
        | ((yy == 54) & (xx >= 48) & (xx <= 76))
        | ((yy == 55) & (xx >= 49) & (xx <= 76))
        | ((yy == 56) & (xx >= 50) & (xx <= 77))
        | ((yy >= 57) & (yy <= 58) & (xx >= 51) & (xx <= 77))
        | ((yy == 59) & (xx >= 52) & (xx <= 77))
        | ((yy == 60) & (xx >= 52) & (xx <= 78))
        | ((yy == 61) & (xx >= 52) & (xx <= 77))
        | ((yy >= 62) & (yy <= 63) & (xx >= 51) & (xx <= 78))
        | ((yy == 64) & (xx >= 51) & (xx <= 77))
        | ((yy == 65) & (xx >= 50) & (xx <= 78))
        | ((yy == 66) & (xx >= 50) & (xx <= 76))
        | ((yy == 67) & (xx >= 50) & (xx <= 75))
        | ((yy == 68) & (xx >= 51) & (xx <= 74))
        | ((yy == 69) & (xx >= 53) & (xx <= 72))
        | ((yy == 70) & (xx >= 59) & (xx <= 71))
        | ((yy == 71) & (xx >= 61) & (xx <= 70))
    )
    arm_near = a & arm_near_zone & (~tail_fin) & (~cheek_gills)

    # 8. Leg L (grounded left hind leg & 3-toed foot)
    leg_L_zone = (yy >= 72) & (yy <= 79) & (xx >= 40) & (xx <= 52)
    leg_L = a & leg_L_zone

    # 9. Leg R (grounded right hind leg & 3-toed foot)
    leg_R_zone = (
        ((yy >= 72) & (yy <= 73) & (xx >= 68) & (xx <= 80))
        | ((yy == 74) & (xx >= 67) & (xx <= 80))
        | ((yy >= 75) & (yy <= 80) & (xx >= 68) & (xx <= 80))
    )
    leg_R = a & leg_R_zone

    used = (
        tail_fin
        | cheek_gills
        | head_fins
        | arm_far
        | lower_jaw
        | head_upper
        | arm_near
        | leg_L
        | leg_R
    )
    torso = a & (~used)

    return {
        "tail_fin": tail_fin,
        "head_fins": head_fins,
        "arm_far": arm_far,
        "leg_L": leg_L,
        "torso": torso,
        "leg_R": leg_R,
        "cheek_gills": cheek_gills,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
        "arm_near": arm_near,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 10 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    orange_m = is_orange_gill_fill(arr96)

    # 1. Tail fin (giant rising fan blade on the left of back.png, above y=47 where vertical columns align)
    tail_zone = (yy >= 28) & (yy <= 47) & (xx <= 23)
    tail_fin = a & tail_zone

    # 2. Cheek gills (left orange gill tip + right 2-pronged orange cheek gill)
    gill_L_zone = (yy >= 32) & (yy <= 39) & (xx >= 34) & (xx <= 39) & orange_m
    gill_R_zone = (yy >= 18) & (yy <= 45) & (xx >= 72)
    cheek_gills = a & (gill_L_zone | gill_R_zone)

    # 3. Head fins (upper dorsal crowns of the twin black head-fins)
    head_fins_zone = (yy >= 18) & (yy <= 25) & (xx >= 35) & (xx <= 71)
    head_fins = a & head_fins_zone & (~cheek_gills)

    # 4. Head upper (part of torso on back.png so dorsal spine and blue back stay 100% continuous)
    head_upper = np.zeros_like(a)

    # 5. Arm far (far-side left shoulder & upper arm on back.png)
    arm_far_zone = (
        ((yy == 36) & (xx >= 34) & (xx <= 38))
        | ((yy >= 37) & (yy <= 50) & (xx >= 25) & (xx <= 38))
        | ((yy == 51) & (xx >= 26) & (xx <= 30))
    )
    arm_far = a & arm_far_zone & (~cheek_gills)

    # 6. Arm near (foreground right extended muscular arm & orange knuckles on back.png)
    arm_near_zone = (
        ((yy >= 46) & (yy <= 47) & (xx >= 75))
        | ((yy >= 48) & (yy <= 50) & (xx >= 74))
        | ((yy >= 51) & (yy <= 52) & (xx >= 66))
        | ((yy == 53) & (xx >= 60))
        | ((yy == 54) & (xx >= 61))
        | ((yy == 55) & (xx >= 62))
        | ((yy >= 56) & (yy <= 68) & (xx >= 64))
    )
    arm_near = a & arm_near_zone

    # 7. Leg L (grounded left hind foot on back.png)
    leg_L_zone = (yy >= 76) & (yy <= 80) & (xx >= 25) & (xx <= 35)
    leg_L = a & leg_L_zone

    # 8. Leg R (grounded right hind leg & foot on back.png)
    leg_R_zone = (yy >= 73) & (yy <= 81) & (xx >= 46) & (xx <= 63)
    leg_R = a & leg_R_zone

    empty = np.zeros_like(a)
    used = (
        tail_fin
        | cheek_gills
        | head_fins
        | head_upper
        | arm_far
        | arm_near
        | leg_L
        | leg_R
    )
    torso = a & (~used)

    return {
        "tail_fin": tail_fin,
        "head_fins": head_fins,
        "arm_far": arm_far,
        "leg_L": leg_L,
        "torso": torso,
        "leg_R": leg_R,
        "cheek_gills": cheek_gills,
        "lower_jaw": empty,
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

    blue_fill_m = is_blue_body_fill(canvas)
    fin_fill_m = is_fin_grey_fill(canvas)
    orange_fill_m = is_orange_gill_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order:
        # 1. tail_fin -> 2. head_fins -> 3. arm_far -> 4. leg_L -> 5. torso -> 6. leg_R
        # -> 7. lower_jaw -> 8. head_upper -> 9. cheek_gills -> 10. arm_near
        fg_over_tail = c_masks["torso"] | c_masks["arm_near"] | c_masks["leg_R"]
        u_tail = ndi.binary_erosion(c_masks["tail_fin"] | fg_over_tail, iterations=1)
        layers["tail_fin"] = build_part_layer(
            canvas,
            c_masks["tail_fin"],
            u_tail & fg_over_tail,
            cushion_px=7,
            allowed_fill_mask=fin_fill_m,
        )

        fg_over_hfins = c_masks["head_upper"] | c_masks["torso"] | c_masks["cheek_gills"]
        u_hfins = ndi.binary_erosion(c_masks["head_fins"] | fg_over_hfins, iterations=1)
        layers["head_fins"] = build_part_layer(
            canvas,
            c_masks["head_fins"],
            u_hfins & fg_over_hfins,
            cushion_px=4,
            allowed_fill_mask=fin_fill_m,
        )

        fg_over_afar = c_masks["head_upper"] | c_masks["lower_jaw"] | c_masks["torso"] | c_masks["cheek_gills"]
        u_afar = ndi.binary_erosion(c_masks["arm_far"] | fg_over_afar, iterations=1)
        layers["arm_far"] = build_part_layer(
            canvas,
            c_masks["arm_far"],
            u_afar & fg_over_afar,
            cushion_px=5,
            allowed_fill_mask=blue_fill_m,
        )

        layers["leg_L"] = build_foot_stem_layer(canvas, c_masks["leg_L"], c_masks["torso"], stem_px=4)
        layers["leg_R"] = build_foot_stem_layer(canvas, c_masks["leg_R"], c_masks["torso"], stem_px=4)

        # IMPORTANT: `torso` cushion ONLY expands under foreground layers drawn ON TOP OF `torso`
        # (`leg_R`, `lower_jaw`, `head_upper`, `cheek_gills`, `arm_near`), NEVER over `tail_fin` or `head_fins`!
        fg_over_torso = (
            c_masks["leg_R"]
            | c_masks["lower_jaw"]
            | c_masks["head_upper"]
            | c_masks["cheek_gills"]
            | c_masks["arm_near"]
        )
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=blue_fill_m,
        )

        # Fill a clean blue shoulder/ribcage backing under `arm_near`'s upper/rear edge
        # (using pure mid-blue body fill `(90, 131, 197)` + 1px black outer contour `(24, 24, 24)`
        # so `arm_near`'s orange plates or grey knuckles are NEVER duplicated onto `torso`!)
        dist_to_torso = ndi.distance_transform_edt(~c_masks["torso"])
        dist_to_tail_or_bg = ndi.distance_transform_edt(c_masks["torso"] | c_masks["arm_near"] | c_masks["leg_R"])
        sh_cap = c_masks["arm_near"] & (
            (dist_to_torso <= 8) | (ndi.distance_transform_edt(~c_masks["tail_fin"]) <= 7)
        )
        # Exclude the front grey knuckles (`is_fin_grey_fill`) from the ribcage backing
        sh_cap = sh_cap & (~fin_fill_m)
        sh_edge = sh_cap & (dist_to_tail_or_bg == 1)
        torso_lyr[sh_cap] = (90, 131, 197, 255)
        torso_lyr[sh_edge] = (24, 24, 24, 255)

        # (2026-11-04) no rectangular mouth-interior pocket any more; the mouth opens via mouth_open.
        layers["torso"] = torso_lyr

        layers["lower_jaw"] = build_part_layer(
            canvas, c_masks["lower_jaw"], np.zeros_like(c_masks["lower_jaw"]), cushion_px=0
        )

        fg_over_head = c_masks["cheek_gills"]
        u_head = ndi.binary_erosion(c_masks["head_upper"] | fg_over_head, iterations=1)
        layers["head_upper"] = build_part_layer(
            canvas,
            c_masks["head_upper"],
            u_head & fg_over_head,
            cushion_px=3,
            allowed_fill_mask=blue_fill_m,
        )

        fg_over_gills = c_masks["arm_near"]
        u_gills = ndi.binary_erosion(c_masks["cheek_gills"] | fg_over_gills, iterations=1)
        layers["cheek_gills"] = build_part_layer(
            canvas,
            c_masks["cheek_gills"],
            u_gills & fg_over_gills,
            cushion_px=2,
            allowed_fill_mask=orange_fill_m,
        )

        layers["arm_near"] = build_part_layer(
            canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0
        )

        pivots = {
            "arm_near": to_canvas_pt(62.0, 44.0),  # Right shoulder socket on front.png
            "arm_far": to_canvas_pt(21.0, 56.0),   # Left shoulder socket on front.png
            "tail_fin": to_canvas_pt(72.0, 52.0),  # Tail fin root behind hip on front.png
            "mouth_line": [to_canvas_pt(mx, my) for mx, my in MOUTH_LINE_NATIVE],
        }

    else:
        # Back view draw order:
        # 1. tail_fin -> 2. cheek_gills -> 3. head_fins -> 4. arm_far -> 5. leg_L -> 6. leg_R
        # -> 7. torso -> 8. arm_near
        layers["lower_jaw"] = np.zeros_like(canvas)
        layers["head_upper"] = np.zeros_like(canvas)

        fg_over_tail = c_masks["torso"]
        u_tail = ndi.binary_erosion(c_masks["tail_fin"] | fg_over_tail, iterations=1)
        layers["tail_fin"] = build_part_layer(
            canvas,
            c_masks["tail_fin"],
            u_tail & fg_over_tail,
            cushion_px=3,
            allowed_fill_mask=fin_fill_m,
        )

        fg_over_gills = c_masks["head_fins"] | c_masks["torso"] | c_masks["arm_far"]
        u_gills = ndi.binary_erosion(c_masks["cheek_gills"] | fg_over_gills, iterations=1)
        layers["cheek_gills"] = build_part_layer(
            canvas,
            c_masks["cheek_gills"],
            u_gills & fg_over_gills,
            cushion_px=3,
            allowed_fill_mask=orange_fill_m,
        )

        fg_over_hfins = c_masks["torso"]
        u_hfins = ndi.binary_erosion(c_masks["head_fins"] | fg_over_hfins, iterations=1)
        layers["head_fins"] = build_part_layer(
            canvas,
            c_masks["head_fins"],
            u_hfins & fg_over_hfins,
            cushion_px=3,
            allowed_fill_mask=fin_fill_m,
        )

        yy_c, _ = np.mgrid[0:cell_h, 0:cell_w]
        fg_over_afar = c_masks["torso"]
        u_afar = ndi.binary_erosion(c_masks["arm_far"] | fg_over_afar, iterations=1)
        layers["arm_far"] = build_part_layer(
            canvas,
            c_masks["arm_far"],
            u_afar & fg_over_afar,
            cushion_px=4,
            allowed_fill_mask=blue_fill_m,
        )

        layers["leg_L"] = build_foot_stem_layer(canvas, c_masks["leg_L"], c_masks["torso"], stem_px=4)
        layers["leg_R"] = build_foot_stem_layer(canvas, c_masks["leg_R"], c_masks["torso"], stem_px=4)

        armpit_y_max = dst_y0 + (56 - y0)
        fg_over_torso = c_masks["arm_near"] & (yy_c <= armpit_y_max)
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr_back = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=blue_fill_m,
        )
        layers["torso"] = torso_lyr_back

        layers["arm_near"] = build_part_layer(
            canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0
        )

        pivots = {
            "arm_near": to_canvas_pt(66.0, 52.0),  # Right shoulder socket on back.png
            "arm_far": to_canvas_pt(35.0, 44.0),   # Left shoulder socket on back.png
            "tail_fin": to_canvas_pt(16.0, 52.0),  # Tail fin root on back.png
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
        (mid_x - 18 - drift, max(2, top_y + 10 - drift // 2)),
        (mid_x + 18 + drift, max(2, top_y + 12 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (246, 180, 65, 255)
    return out


def synthesize_swampert_mega_frame(
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
    hfins_dx, hfins_dy = 0, 0
    gills_dx, gills_dy = 0, 0
    jaw_dx, jaw_dy = 0, 0
    jaw_open = 0
    anear_dx, anear_dy = 0, 0
    afar_dx, afar_dy = 0, 0
    tail_dx, tail_dy = 0, 0
    leg_L_dx, leg_L_dy = 0, 0
    leg_R_dx, leg_R_dy = 0, 0

    anear_ang = 0.0
    afar_ang = 0.0
    tail_ang = 0.0

    if action == "idle":
        # Heavy Kaiju hulk breathing + menacing roar/bark (`lower_jaw` opens `+1px`) +
        # circular muscular arm readiness (`arm_near`, `arm_far` pivot & pump) +
        # trembling cheek gills & bristling head fins + flexing giant tail fin:
        # 1. Both hind legs (`leg_L`, `leg_R`) stay 100% planted on the ground `(0, 0)`.
        # 2. Upper body (`torso`, `head_upper`) rises and sinks (`-1..+1px`) while leaning into the breath.
        # 3. `lower_jaw` opens `+1px` during the inhalation/roar peak, revealing the dark-red mouth pocket.
        # 4. Both colossal muscular arms trace a heavy orbital stance with shoulder rotation (`±2.5°`).
        # 5. `tail_fin` flexes and sways (`±2.5°`) with phase delay!
        body_dy = int(round(1.2 * math.sin(phase - 0.2)))
        body_dx = int(round(0.6 * max(0.0, math.sin(phase - 0.2)) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        jaw_open = 1 if math.sin(phase - 0.1) > 0.15 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays with the head; mouth opens via mouth_open

        hfins_dx = head_dx
        hfins_dy = head_dy - (1 if frame_idx in (4, 5, 6) else 0)

        gills_dx = head_dx
        gills_dy = head_dy + (1 if frame_idx in (5, 7) else 0)

        anear_dx = body_dx + int(round(1.2 * math.cos(phase - 0.5) * facing))
        anear_dy = body_dy + int(round(1.5 * math.sin(phase - 0.5)))
        anear_ang = 2.5 * math.sin(phase - 0.5) * (-facing)

        afar_dx = body_dx + int(round(1.2 * math.cos(phase - 0.8) * facing))
        afar_dy = body_dy + int(round(1.5 * math.sin(phase - 0.8)))
        afar_ang = -2.5 * math.sin(phase - 0.8) * facing

        tail_dx = body_dx
        tail_dy = body_dy + (0 if is_back else int(round(1.4 * math.sin(phase - 1.1))))
        tail_ang = (1.6 if is_back else 2.5) * math.sin(phase - 1.1) * facing

    elif action == "walk":
        # Heavy Kaiju knuckle-stomp march:
        # Alternating left/right hind leg stomp (`lift_px=2, stride_px=2`),
        # 2-beat upper-body bounce, powerful counter-rotating muscular arms (`±4.5°` shoulder rotation),
        # ferocious barking roar (`lower_jaw` opens `+1px`) on each stride, and tail fin counterbalance!
        bounce = int(round(-1.5 * abs(sin1)))
        sway_x = int(round(1.2 * sin1 * facing))

        torso_dx, torso_dy = sway_x, bounce
        head_dx = sway_x + int(round(0.6 * cos1 * facing))
        head_dy = bounce + int(round(0.6 * math.sin(2.0 * phase - 0.3)))

        jaw_open = 1 if math.sin(2.0 * phase) > 0.15 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays with the head; mouth opens via mouth_open

        hfins_dx = head_dx
        hfins_dy = head_dy - (1 if jaw_open > 0 else 0)

        gills_dx = head_dx
        gills_dy = head_dy + int(round(0.7 * math.sin(2.0 * phase - 0.4)))

        anear_dx = sway_x + int(round(2.0 * sin1 * facing))
        anear_dy = bounce + int(round(1.6 * cos1))
        anear_ang = 4.5 * sin1 * (-facing)

        afar_dx = sway_x + int(round(-2.0 * sin1 * facing))
        afar_dy = bounce + int(round(-1.6 * cos1))
        afar_ang = -4.5 * sin1 * facing

        tail_dx = sway_x
        tail_dy = bounce + (0 if is_back else int(round(1.6 * math.sin(2.0 * phase - 1.0))))
        tail_ang = (2.0 if is_back else 3.0) * math.sin(2.0 * phase - 1.0) * facing

        leg_L_dx, leg_L_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        leg_R_dx, leg_R_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Mega Swampert "Kaiju Roar + Spinning Double Hammer Arm & Waterfall Crash"!
        # Frames 0..3 (Wind-up & Big Kaiju Roar):
        #   - Hauls colossal body back (`-3px`), bristles `head_fins` (`-1px`), trembles `cheek_gills`,
        #   - Opens `lower_jaw` WIDE (`+2px` ferocious Kaiju roar!),
        #   - Winds both giant muscular arms high at the shoulder sockets (`-9°` rotation + lift).
        # Frames 4..9 (Spinning Hammer Arm Smash & Waterfall Lunge):
        #   - Explosive forward lunge (`+5px` body, `+6px` giant arms, lead leg stomps `+3px` forward),
        #   - High-speed spinning Hammer Arm rotation around shoulder sockets (`-9° -> +10° -> -7° -> +8°`)
        #     combined with a circular smash trajectory,
        #   - Giant `tail_fin` whips and spins (`±5.5°`) like a hydro-propeller,
        #   - Jaws clamp on impact (`0px` at frames 6..7) and let out a victorious `+2px` roar at frames 8..9!
        # Frames 10..15: Heavy follow-through recovery to stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.65 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        body_dx = int(round(4.5 * k * facing))
        if frame_idx in (1, 2, 3):
            body_dy = -1 if frame_idx in (2, 3) else 0
        elif frame_idx in (6, 7, 8):
            body_dy = 1
        else:
            body_dy = 0

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = int(round(4.8 * k * facing))
        head_dy = body_dy

        # Ferocious +2px wide roar on wind-up (frames 2..4) and follow-through roar (frames 8..9)!
        if frame_idx in (2, 3, 4, 8):
            jaw_open = 2
        elif frame_idx in (1, 5, 9):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays with the head; mouth opens via mouth_open

        # Head fins bristle (-1px) and cheek gills vibrate during roar & smash
        hfins_dx = head_dx
        hfins_dy = head_dy - (1 if frame_idx in (2, 3, 4, 6, 7, 8) else 0)

        gills_vib = [0, 0, -1, 1, -1, 1, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0]
        gills_dx = head_dx
        gills_dy = head_dy + gills_vib[frame_idx]

        # Colossal muscular arms: wind up high, then execute a powerful spinning Hammer Arm smash arc!
        anear_arc_dy = [0, -1, -2, -2, -1, 1, 2, 1, -1, 1, 0, 0, 0, 0, 0, 0]
        afar_arc_dy = [0, -1, -2, -2, 1, 2, -1, 2, 1, 0, 0, 0, 0, 0, 0, 0]
        anear_dx = int(round(5.0 * k * facing))
        anear_dy = body_dy + anear_arc_dy[frame_idx]
        afar_dx = int(round(5.2 * k * facing))
        afar_dy = body_dy + afar_arc_dy[frame_idx]

        spin_table_near = [0.0, -4.5, -7.5, -9.5, 6.0, 9.5, -6.5, 8.0, -4.5, 3.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        spin_table_far = [0.0, 4.5, 7.5, 9.5, -6.5, -9.5, 7.5, -8.0, 5.0, -3.5, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        anear_ang = spin_table_near[frame_idx] * (-facing)
        afar_ang = spin_table_far[frame_idx] * facing

        # Giant tail fin whips and spins like a hydro-propeller during the strike!
        tail_spin_ang = [0.0, -2.5, -4.0, -5.0, 5.0, -5.5, 5.5, -5.0, 4.0, -2.5, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        tail_whip_dy = [0, 1, 1, 2, -2, 2, -2, 2, -1, 1, 0, 0, 0, 0, 0, 0]
        tail_dx = body_dx
        tail_dy = body_dy + (0 if is_back else tail_whip_dy[frame_idx])
        tail_ang = (0.55 if is_back else 1.0) * tail_spin_ang[frame_idx] * facing

        # Lead leg stomps into the lunge while rear leg braces
        lead_x = int(round(3.0 * max(0.0, k) * facing)) if k >= 0 else int(round(1.8 * k * facing))
        rear_x = int(round(2.2 * k * facing))
        if not is_back:
            leg_L_dx, leg_L_dy = lead_x, 0
            leg_R_dx, leg_R_dy = rear_x, 0
        else:
            leg_R_dx, leg_R_dy = lead_x, 0
            leg_L_dx, leg_L_dy = rear_x, 0

    elif action == "hurt":
        # Heavy Kaiju knockback + hit-shake + pained roar (`lower_jaw` open `+2px`) + flung arms + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-3.5 * env * facing)) + jit
        body_dy = int(round(-1.0 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-1.5 * env))

        jaw_open = 2 if frame_idx in (2, 3, 4) else (1 if 1 <= frame_idx <= 7 else 0)
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays with the head; mouth opens via mouth_open

        hfins_dx = head_dx
        hfins_dy = head_dy - (1 if 2 <= frame_idx <= 6 else 0)

        gills_dx = head_dx
        gills_dy = head_dy - (1 if frame_idx in (2, 4, 6) else 0)

        anear_dx = body_dx + int(round(-1.5 * env * facing))
        anear_dy = body_dy + int(round(-1.5 * env))
        anear_ang = -5.5 * env * (-facing)

        afar_dx = body_dx + int(round(-1.5 * env * facing))
        afar_dy = body_dy + int(round(-1.5 * env))
        afar_ang = 5.5 * env * facing

        tail_dx = body_dx
        tail_dy = body_dy if is_back else int(round(1.2 * env))
        tail_ang = (-1.8 if is_back else -3.0) * env * facing

        slide_x = int(round(body_dx * 0.65))
        leg_L_dx, leg_R_dx = slide_x, slide_x

    tail_rot = layers["tail_fin"].copy() if is_back else rotate_layer_nearest(layers["tail_fin"], tail_ang, pivots["tail_fin"])
    afar_rot = layers["arm_far"].copy() if is_back else rotate_layer_nearest(layers["arm_far"], afar_ang, pivots["arm_far"])
    anear_rot = rotate_layer_nearest(layers["arm_near"], anear_ang, pivots["arm_near"])

    if not is_back:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(layers["head_fins"], hfins_dx, hfins_dy),
            shift_layer_int(afar_rot, afar_dx, afar_dy),
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(layers["cheek_gills"], gills_dx, gills_dy),
            shift_layer_int(anear_rot, anear_dx, anear_dy),
        ]
    else:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(layers["cheek_gills"], gills_dx, gills_dy),
            shift_layer_int(layers["head_fins"], hfins_dx, hfins_dy),
            shift_layer_int(afar_rot, afar_dx, afar_dy),
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(anear_rot, anear_dx, anear_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)

    # Small face-following mouth opening (front views only; back views show no mouth)
    if not is_back and pivots.get("mouth_line"):
        if action == "attack":
            mouth_amt = 2 if frame_idx in (2, 3, 4, 8) else (1 if jaw_open > 0 else 0)
        else:
            mouth_amt = 1 if jaw_open > 0 else 0
        if mouth_amt > 0:
            line = mouth_open.transform_points(pivots["mouth_line"], dx=head_dx, dy=head_dy)
            opened = mouth_open.open_mouth(out, line, amount=mouth_amt, deep=MOUTH_DEEP,
                                           inner=MOUTH_INNER, peak=0.6)
            changed = np.any(opened != out, axis=2)
            if np.any(changed):
                ok = np.zeros(changed.shape, dtype=bool)
                for col in MOUTH_CARVABLE:
                    ok |= np.all(out[:, :, :3] == np.array(col, dtype=np.uint8), axis=2)
                keep = changed & ok
                out = out.copy()
                out[keep] = opened[keep]
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

    # Tight bounds of front (79x67) and back (89x64)
    max_w, max_h = 89, 67
    cell_w = 114
    cell_h = 92
    foot_y = 84

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
                frame_rgba = synthesize_swampert_mega_frame(
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
        "generator": "forge_swampert_mega_actions",
        "key": "0260_swampert-mega",
        "dex": 260,
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

    print(f"Built 0260_swampert-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
