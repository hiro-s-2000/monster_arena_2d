#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Primal Groudon (`0383_groudon-primal`) modeled directly after Primal Groudon's
official 50-frame GIF (`0383_groudon/mega/groudon-primal/front.gif`) and native `front.png` / `back.png`:
  - Exact anatomical 9-layer rigging with true joint pivots (shoulder sockets & tail root socket):
      1. `tail` (massive 4-bladed crimson & magma-veined tail with charcoal side spikes — 100% separated
                 from both hind legs so the tail never steps with the feet; pivoted at the tail socket
                 `rotate_layer_nearest` for heavy kaiju tail sweeps and spinning Precipice Blades / Eruption slams!)
      2. `dorsal_spikes` (crimson & magma-veined dorsal armor plates and charcoal neck/back spikes —
                 flares and bristles independently during roars and eruptions)
      3. `arm_far` (far-side dark-charcoal & crimson arm with 4 white claws — pivoted at the far shoulder
                 socket `rotate_layer_nearest` for ferocious rotational claw swings)
      4. `leg_L` (far-side magma-cored crimson hind leg & 3 white claws — 100% grounded `(0, 0)` in `idle`
                 while the upper body flexes at the hip sockets, and earth-shaking kaiju stomp in `walk`)
      5. `torso` (crimson & golden-magma chest/back carapace, dark charcoal segmented belly armor,
                 upper thighs, and glowing magma/crimson mouth interior pocket behind `lower_jaw`)
      6. `leg_R` (near-side magma-cored crimson hind leg & 3 white claws — 100% grounded `(0, 0)` in `idle`
                 and alternating heavy kaiju stomp in `walk`)
      7. `lower_jaw` (charcoal-grey toothed lower jaw between the crimson cheek guards — opens `+1px` in
                 `idle` / `walk` for deep continental kaiju roars/barks, and drops `+2px` wide in `attack`
                 & `hurt` to expose the glowing golden-magma & crimson throat interior!)
      8. `head_upper` (4-bladed crimson crown helmet, glowing golden/orange eye, upper jaw fangs, and
                 left/right crimson cheek guards — 100% rigid integer translation `shift_layer_int` ONLY
                 so the face, eye, and crown blades never deform by a single pixel!)
      9. `arm_near` (foreground giant spherical crimson shoulder pauldron, Omega-emblem magma gauntlet,
                 and 4 massive white claws — pivoted at the shoulder socket `rotate_layer_nearest` for
                 dynamic rotational wind-ups and earth-shattering Precipice Blades / Hammer Arm smashes!)
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
import mouth_open  # noqa: E402  (2026-11-04: small face-following mouth opening, no jaw drop)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0383_groudon/mega/groudon-primal"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0383_groudon/mega/groudon-primal"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16

# Closed mouth line on native `front.png` (96x96), ordered hinge/cheek corner -> lip tip (snout faces left).
# The upper teeth row is y=34; the opening is carved just below it, inside the face.
# Back views (N/NE/NW) get no opening.
MOUTH_LINE_FRONT_NATIVE = [(26.0, 34.0), (20.0, 34.0), (15.0, 34.0)]
MOUTH_INNER = (124, 33, 41)  # existing dark-crimson palette colour (visible mouth interior)


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
        ((r <= 16) & (g <= 16) & (b <= 16))
        | ((r == 39) & (g == 35) & (b == 36))
        | ((r == 52) & (g == 52) & (b == 52))
    )


def is_armor_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    # Crimson, maroon, dark-charcoal armor, or magma gold (excluding black outline & pure white claw tips)
    return a & (~is_dark_outline(rgba)) & (~((r >= 200) & (g >= 200) & (b >= 200)))


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
    Segments native 96x96 `front.png` (facing left) into 9 exact anatomical masks
    following the artist's 1px dark outlines:
      1. `tail`: Massive 4-bladed crimson & magma-veined tail with 3 charcoal side spikes (`x=61..90, y=29..74`).
      2. `dorsal_spikes`: Crimson & magma-veined neck/back dorsal armor ridge + 3 neck charcoal spikes +
                          2 mid-back charcoal spikes (`x=30..69, y=18..61`).
      3. `arm_far`: Far-side left dark-charcoal & crimson forearm + 4 white claws (`x=12..35, y=46..65`).
      4. `leg_L`: Far-side left grounded hind shin, ankle & 3 white claws (`x=20..44, y=73..83`).
      5. `torso`: Central dark-charcoal segmented chest/belly, magma veins, upper thighs, and mouth pocket.
      6. `leg_R`: Near-side right grounded hind shin, ankle & 3 white claws (`x=55..74, y=75..88`).
      7. `lower_jaw`: Charcoal-grey toothed lower jaw between the crimson cheek guards (`x=17..26, y=35..39`).
      8. `head_upper`: 4-bladed crimson crown helmet, glowing yellow/orange eye, upper jaw teeth, and
                       left/right crimson cheek guards (`x=10..35, y=11..45`).
      9. `arm_near`: Foreground spherical crimson shoulder pauldron, Omega-emblem magma gauntlet, and
                     4 massive white claws (`x=41..62, y=35..70`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Lower jaw (charcoal-grey toothed lower jaw between the left & right crimson cheek guards)
    jaw_zone = (
        ((yy == 35) & (xx >= 18) & (xx <= 26))
        | ((yy == 36) & (xx >= 18) & (xx <= 26))
        | ((yy == 37) & (xx >= 19) & (xx <= 26))
        | ((yy == 38) & (xx >= 20) & (xx <= 26))
        | ((yy == 39) & (xx >= 22) & (xx <= 26))
    )
    lower_jaw = a & jaw_zone

    # 2. Head upper (4-bladed crimson crown, glowing eye, upper teeth, left cheek guard x=14..22, right cheek guard x=27..33)
    head_zone = (
        ((yy >= 11) & (yy <= 17) & (xx <= 35))
        | ((yy >= 18) & (yy <= 24) & (xx <= 34))
        | ((yy == 25) & (xx <= 30))
        | ((yy == 26) & (xx <= 30))
        | ((yy == 27) & (xx <= 30))
        | ((yy >= 28) & (yy <= 34) & (xx <= 33))
        | ((yy >= 35) & (yy <= 40) & (((xx >= 14) & (xx <= 21)) | ((xx >= 27) & (xx <= 33))))
        | ((yy >= 41) & (yy <= 42) & (((xx >= 17) & (xx <= 22)) | ((xx >= 27) & (xx <= 33))))
        | ((yy == 43) & (((xx >= 18) & (xx <= 22)) | ((xx >= 27) & (xx <= 32))))
        | ((yy == 44) & (((xx >= 18) & (xx <= 21)) | ((xx >= 27) & (xx <= 30))))
        | ((yy == 45) & (xx >= 18) & (xx <= 20))
    )
    head_upper = a & head_zone & (~lower_jaw)

    # 3. Arm near (foreground spherical crimson shoulder pauldron + Omega magma gauntlet + 4 white claws)
    arm_near_zone = (
        ((yy == 35) & (xx >= 51) & (xx <= 58))
        | ((yy == 36) & (xx >= 49) & (xx <= 60))
        | ((yy == 37) & (xx >= 48) & (xx <= 61))
        | ((yy >= 38) & (yy <= 42) & (xx >= 48) & (xx <= 62))
        | ((yy >= 43) & (yy <= 46) & (xx >= 49) & (xx <= 60))
        | ((yy == 47) & (xx >= 48) & (xx <= 59))
        | ((yy == 48) & (xx >= 46) & (xx <= 59))
        | ((yy == 49) & (xx >= 45) & (xx <= 60))
        | ((yy == 50) & (xx >= 44) & (xx <= 60))
        | ((yy >= 51) & (yy <= 53) & (xx >= 43) & (xx <= 61))
        | ((yy == 54) & (xx >= 42) & (xx <= 61))
        | ((yy >= 55) & (yy <= 57) & (xx >= 41) & (xx <= 60))
        | ((yy >= 58) & (yy <= 60) & (xx >= 41) & (xx <= 60))
        | ((yy == 61) & (xx >= 41) & (xx <= 59))
        | ((yy >= 62) & (yy <= 64) & (xx >= 41) & (xx <= 58))
        | ((yy >= 65) & (yy <= 66) & (xx >= 41) & (xx <= 57))
        | ((yy == 67) & (xx >= 43) & (xx <= 56))
        | ((yy == 68) & (xx >= 44) & (xx <= 56))
        | ((yy == 69) & (xx >= 44) & (xx <= 55))
        | ((yy == 70) & (xx >= 50) & (xx <= 53))
    )
    arm_near = a & arm_near_zone

    # 4. Dorsal spikes (crimson/magma upper-back ridge behind head + 3 neck charcoal spikes + 2 mid-back charcoal spikes)
    dorsal_upper = (
        ((yy >= 18) & (yy <= 24) & (xx >= 35) & (xx <= 60))
        | ((yy >= 25) & (yy <= 27) & (xx >= 31) & (xx <= 60))
        | ((yy >= 28) & (yy <= 34) & (xx >= 34) & (xx <= 61))
        | ((yy == 35) & (xx >= 39) & (xx <= 61))
        | ((yy == 36) & (xx >= 42) & (xx <= 61))
    ) & (~head_upper) & (~arm_near)
    dorsal_mid_spikes = (
        ((yy == 46) & (xx >= 59) & (xx <= 64))
        | ((yy == 47) & (xx >= 58) & (xx <= 66))
        | ((yy == 48) & (xx >= 58) & (xx <= 67))
        | ((yy == 49) & (xx >= 59) & (xx <= 68))
        | ((yy == 50) & (xx >= 59) & (xx <= 65))
        | ((yy == 54) & (xx >= 59) & (xx <= 63))
        | ((yy == 55) & (xx >= 59) & (xx <= 66))
        | ((yy == 56) & (xx >= 59) & (xx <= 68))
        | ((yy == 57) & (xx >= 58) & (xx <= 66))
        | ((yy == 58) & (xx >= 58) & (xx <= 63))
        | ((yy == 59) & (xx >= 58) & (xx <= 60))
    ) & (~arm_near)
    dorsal_spikes = a & (dorsal_upper | dorsal_mid_spikes)

    # 5. Tail (4 tall crimson tail blades + golden-magma tail body + 3 charcoal tail-side spikes on the right)
    tail_zone = (
        ((yy >= 29) & (yy <= 45) & (xx >= 61))
        | ((yy >= 46) & (yy <= 60) & (xx >= 62))
        | ((yy == 61) & (xx >= 66))
        | ((yy == 62) & (xx >= 67))
        | ((yy >= 63) & (yy <= 64) & (xx >= 68))
        | ((yy >= 65) & (yy <= 66) & (xx >= 69))
        | ((yy >= 67) & (yy <= 69) & (xx >= 70))
        | ((yy >= 70) & (yy <= 71) & (xx >= 71))
        | ((yy >= 72) & (yy <= 74) & (xx >= 70))
    )
    tail = a & tail_zone & (~arm_near) & (~dorsal_spikes)

    # 6. Arm far (far-side left charcoal/crimson forearm + 4 white claws)
    arm_far_zone = (
        ((yy >= 46) & (yy <= 57) & (xx <= 33))
        | ((yy == 58) & (xx <= 28))
        | ((yy == 59) & (xx <= 25))
        | ((yy == 60) & (xx <= 24))
        | ((yy >= 61) & (yy <= 65) & (xx <= 22))
    )
    arm_far = a & arm_far_zone & (~head_upper) & (~lower_jaw)

    # 7. Grounded Left Hind Leg (`leg_L`: lower shin, ankle & 3 white claws at y=73..83, x=20..44)
    leg_L_zone = (yy >= 73) & (yy <= 83) & (xx >= 20) & (xx <= 45)
    leg_L = a & leg_L_zone

    # 8. Grounded Right Hind Leg (`leg_R`: lower shin, ankle & 3 white claws at y=75..88, x=55..74)
    leg_R_zone = (yy >= 75) & (yy <= 88) & (xx >= 55) & (xx <= 74)
    leg_R = a & leg_R_zone & (~tail)

    used = (
        tail
        | dorsal_spikes
        | arm_far
        | leg_L
        | leg_R
        | lower_jaw
        | head_upper
        | arm_near
    )
    torso = a & (~used)

    return {
        "tail": tail,
        "dorsal_spikes": dorsal_spikes,
        "arm_far": arm_far,
        "leg_L": leg_L,
        "torso": torso,
        "leg_R": leg_R,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
        "arm_near": arm_near,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` (facing right) into matching 9 anatomical masks:
      1. `tail`: 4 crimson tail blades on the left + golden-magma tail base + lower charcoal tail spikes (`x=7..55, y=55..84`).
      2. `dorsal_spikes`: 3 charcoal neck/head-side spikes (`x=67..83, y=22..40`) + 2 back-shoulder charcoal spikes (`x=53..66, y=43..59`).
      3. `arm_far`: Far-left hip/side armor protrusion (`x=29..32, y=55..66`).
      4. `leg_L`: Far-side hind foot claws visible at (`x=67..74, y=75..78`).
      5. `torso`: Massive crimson & golden-magma back carapace and upper thigh.
      6. `leg_R`: Near-side crimson hind ankle, heel & white claws (`x=51..75, y=74..85`).
      7. `lower_jaw`: Right-side charcoal lower jaw & teeth (`x=80..87, y=33..40`).
      8. `head_upper`: 4-bladed crimson crown, glowing yellow eye, upper snout, and crimson cheek guard (`x=58..88, y=12..44`).
      9. `arm_near`: Near-right spherical crimson shoulder + Omega magma gauntlet + 4 white claws (`x=57..88, y=38..67`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Lower jaw (right-side charcoal lower jaw & teeth below the upper snout outline)
    jaw_zone = (
        ((yy == 33) & (xx >= 81) & (xx <= 87))
        | ((yy == 34) & (xx >= 80) & (xx <= 86))
        | ((yy == 35) & (xx >= 80) & (xx <= 85))
        | ((yy == 36) & (xx >= 80) & (xx <= 84))
        | ((yy == 37) & (xx >= 81) & (xx <= 83))
        | ((yy == 38) & (xx >= 81) & (xx <= 81))
    )
    lower_jaw = a & jaw_zone

    # 2. Dorsal spikes (3 charcoal neck spikes at y=22..40, x=67..83 + 2 shoulder charcoal spikes at y=43..59, x=53..65)
    neck_spikes = (
        ((yy == 23) & (xx >= 67) & (xx <= 75))
        | ((yy == 24) & (xx >= 67) & (xx <= 77))
        | ((yy == 25) & (xx >= 67) & (xx <= 76))
        | ((yy == 26) & (xx >= 63) & (xx <= 74))
        | ((yy == 27) & (xx >= 62) & (xx <= 72))
        | ((yy >= 28) & (yy <= 29) & (xx >= 62) & (xx <= 73))
        | ((yy == 30) & (xx >= 62) & (xx <= 73))
        | ((yy == 31) & (xx >= 63) & (xx <= 69))
        | ((yy == 32) & (xx >= 64) & (xx <= 70))
        | ((yy == 33) & (xx >= 62) & (xx <= 70))
        | ((yy >= 34) & (yy <= 36) & (xx >= 62) & (xx <= 73))
        | ((yy == 37) & (xx >= 63) & (xx <= 71))
        | ((yy == 38) & (xx >= 66) & (xx <= 71))
        | ((yy == 39) & (xx >= 65) & (xx <= 70))
        | ((yy == 40) & (xx >= 66) & (xx <= 70))
    )
    back_sh_spikes = (
        ((yy >= 44) & (yy <= 50) & (xx >= 54) & (xx <= 58))
        | ((yy == 52) & (xx >= 55) & (xx <= 60))
        | ((yy >= 53) & (yy <= 54) & (xx >= 53) & (xx <= 62))
        | ((yy >= 55) & (yy <= 56) & (xx >= 53) & (xx <= 63))
        | ((yy >= 57) & (yy <= 58) & (xx >= 54) & (xx <= 64))
    )
    dorsal_spikes = a & (neck_spikes | back_sh_spikes) & (~lower_jaw)

    # 3. Head upper (4-bladed crown, eye, upper jaw snout, and crimson cheek guard at x=72..78, y=31..43)
    head_zone = (
        ((yy >= 12) & (yy <= 21) & (xx >= 58))
        | ((yy >= 22) & (yy <= 32) & (xx >= 70))
        | ((yy >= 33) & (yy <= 43) & (xx >= 72))
    )
    head_upper = a & head_zone & (~lower_jaw) & (~dorsal_spikes)

    # 4. Arm near (spherical crimson shoulder pauldron + Omega magma gauntlet + 4 white claws)
    arm_near_zone = (
        ((yy >= 38) & (yy <= 44) & (xx >= 57) & (xx <= 71))
        | ((yy >= 45) & (yy <= 49) & (xx >= 59) & (xx <= 73))
        | ((yy >= 50) & (yy <= 52) & (xx >= 61) & (xx <= 83))
        | ((yy >= 53) & (yy <= 58) & (xx >= 64) & (xx <= 88))
        | ((yy >= 59) & (yy <= 60) & (xx >= 64) & (xx <= 86))
        | ((yy == 61) & (xx >= 65) & (xx <= 86))
        | ((yy == 62) & (xx >= 67) & (xx <= 86))
        | ((yy == 63) & (xx >= 69) & (xx <= 86))
        | ((yy == 64) & (xx >= 70) & (xx <= 84))
        | ((yy == 65) & (xx >= 72) & (xx <= 82))
        | ((yy == 66) & (xx >= 74) & (xx <= 81))
        | ((yy == 67) & (xx >= 75) & (xx <= 79))
    )
    arm_near = a & arm_near_zone & (~head_upper) & (~dorsal_spikes)

    # 5. Grounded Near Hind Foot (`leg_R`: near-side ankle, heel & both white claws at y=73..85, x=50..76)
    leg_R_zone = (yy >= 73) & (yy <= 85) & (xx >= 50) & (xx <= 76)
    leg_R = a & leg_R_zone

    # 6. Far-left leg thigh edge (`leg_L`: far-left hip/thigh armor edge at x=29..31, y=56..66)
    leg_L_zone = (yy >= 56) & (yy <= 66) & (xx >= 29) & (xx <= 31)
    leg_L = a & leg_L_zone

    # 7. Tail (4 crimson tail blades on the left + lower charcoal tail body & spikes below the carapace outline)
    tail_zone = (
        ((yy >= 55) & (yy <= 71) & (xx <= 27))
        | ((yy == 72) & (xx <= 26))
        | ((yy == 73) & ((xx <= 26) | ((xx >= 42) & (xx <= 49))))
        | ((yy == 74) & ((xx <= 27) | ((xx >= 42) & (xx <= 51))))
        | ((yy == 75) & ((xx <= 28) | ((xx >= 38) & (xx <= 52))))
        | ((yy == 76) & (xx <= 53))
        | ((yy >= 77) & (yy <= 84) & (xx <= 52))
    )
    tail = a & tail_zone & (~leg_R) & (~leg_L)

    arm_far = np.zeros_like(a)

    used = (
        tail
        | dorsal_spikes
        | arm_far
        | leg_L
        | leg_R
        | lower_jaw
        | head_upper
        | arm_near
    )
    torso = a & (~used)

    return {
        "tail": tail,
        "dorsal_spikes": dorsal_spikes,
        "arm_far": arm_far,
        "leg_L": leg_L,
        "torso": torso,
        "leg_R": leg_R,
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
    dst_y0 = (foot_y + 1 - sh) if is_back else (foot_y - sh)

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

    body_fill_m = is_armor_body_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order:
        # 1. tail -> 2. dorsal_spikes -> 3. arm_far -> 4. leg_L -> 5. leg_R
        # -> 6. torso -> 7. lower_jaw -> 8. head_upper -> 9. arm_near
        fg_over_tail = c_masks["dorsal_spikes"] | c_masks["torso"] | c_masks["arm_near"] | c_masks["leg_R"]
        u_tail = ndi.binary_erosion(c_masks["tail"] | fg_over_tail, iterations=1)
        layers["tail"] = build_part_layer(
            canvas,
            c_masks["tail"],
            u_tail & fg_over_tail,
            cushion_px=6,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_dorsal = c_masks["torso"] | c_masks["head_upper"] | c_masks["arm_near"]
        u_dorsal = ndi.binary_erosion(c_masks["dorsal_spikes"] | fg_over_dorsal, iterations=1)
        layers["dorsal_spikes"] = build_part_layer(
            canvas,
            c_masks["dorsal_spikes"],
            u_dorsal & fg_over_dorsal,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_afar = c_masks["torso"] | c_masks["leg_L"]
        u_afar = ndi.binary_erosion(c_masks["arm_far"] | fg_over_afar, iterations=1)
        layers["arm_far"] = build_part_layer(
            canvas,
            c_masks["arm_far"],
            u_afar & fg_over_afar,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        layers["leg_L"] = build_foot_stem_layer(canvas, c_masks["leg_L"], c_masks["torso"], stem_px=4)
        layers["leg_R"] = build_foot_stem_layer(canvas, c_masks["leg_R"], c_masks["torso"], stem_px=4)

        # IMPORTANT: `torso` cushion ONLY expands under foreground layers drawn ON TOP OF `torso`
        # (`lower_jaw`, `head_upper`, `arm_near`), NEVER over `tail`, `dorsal_spikes`, or `arm_far`!
        fg_over_torso = c_masks["lower_jaw"] | c_masks["head_upper"] | c_masks["arm_near"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=body_fill_m,
        )

        # (2026-11-04) The rectangular mouth-interior pocket is no longer painted; the mouth opens via
        # `mouth_open.open_mouth` along MOUTH_LINE_FRONT_NATIVE on the composed frame instead.
        layers["torso"] = torso_lyr

        layers["lower_jaw"] = build_part_layer(canvas, c_masks["lower_jaw"], np.zeros_like(c_masks["lower_jaw"]), cushion_px=0)
        layers["head_upper"] = build_part_layer(canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0)
        layers["arm_near"] = build_part_layer(canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0)

        pivots = {
            "arm_near": to_canvas_pt(54.0, 43.0),
            "arm_far": to_canvas_pt(30.0, 51.0),
            "tail": to_canvas_pt(64.0, 56.0),
            "mouth": [to_canvas_pt(mx, my) for mx, my in MOUTH_LINE_FRONT_NATIVE],
        }

    else:
        # Back view draw order:
        # 1. tail -> 2. dorsal_spikes -> 3. leg_L -> 4. leg_R
        # -> 5. torso -> 6. lower_jaw -> 7. head_upper -> 8. arm_near
        layers["arm_far"] = np.zeros_like(canvas)

        fg_over_tail = c_masks["torso"] | c_masks["leg_L"] | c_masks["leg_R"]
        u_tail = ndi.binary_erosion(c_masks["tail"] | fg_over_tail, iterations=1)
        layers["tail"] = build_part_layer(
            canvas,
            c_masks["tail"],
            u_tail & fg_over_tail,
            cushion_px=6,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_dorsal = c_masks["torso"] | c_masks["head_upper"] | c_masks["arm_near"]
        u_dorsal = ndi.binary_erosion(c_masks["dorsal_spikes"] | fg_over_dorsal, iterations=1)
        layers["dorsal_spikes"] = build_part_layer(
            canvas,
            c_masks["dorsal_spikes"],
            u_dorsal & fg_over_dorsal,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_legL = c_masks["torso"]
        u_legL = ndi.binary_erosion(c_masks["leg_L"] | fg_over_legL, iterations=1)
        layers["leg_L"] = build_part_layer(
            canvas,
            c_masks["leg_L"],
            u_legL & fg_over_legL,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )
        layers["leg_R"] = build_foot_stem_layer(canvas, c_masks["leg_R"], c_masks["torso"], stem_px=4)

        # IMPORTANT: `torso` cushion ONLY expands under foreground layers drawn ON TOP OF `torso`
        # (`lower_jaw`, `head_upper`, `arm_near`), NEVER over `tail`, `dorsal_spikes`, or `leg_L`!
        fg_over_torso = c_masks["lower_jaw"] | c_masks["head_upper"] | c_masks["arm_near"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=body_fill_m,
        )

        # (2026-11-04) No mouth-interior pocket in back views (no opening there).
        layers["torso"] = torso_lyr

        layers["lower_jaw"] = build_part_layer(canvas, c_masks["lower_jaw"], np.zeros_like(c_masks["lower_jaw"]), cushion_px=0)
        layers["head_upper"] = build_part_layer(canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0)
        layers["arm_near"] = build_part_layer(canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0)

        pivots = {
            "arm_near": to_canvas_pt(64.0, 46.0),
            "arm_far": to_canvas_pt(31.0, 60.0),
            "tail": to_canvas_pt(30.0, 69.0),
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
            out[sy:sy + 2, sx:sx + 2] = (251, 251, 251, 255)
    return out


def synthesize_groudon_primal_frame(
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
    dorsal_dx, dorsal_dy = 0, 0
    anear_dx, anear_dy = 0, 0
    afar_dx, afar_dy = 0, 0
    tail_dx, tail_dy = 0, 0
    leg_L_dx, leg_L_dy = 0, 0
    leg_R_dx, leg_R_dy = 0, 0

    anear_ang = 0.0
    afar_ang = 0.0
    tail_ang = 0.0

    if action == "idle":
        # Continental Magma Kaiju Breathing + Thunderous Roar/Bark + Circular Omega-Arm & 4-Blade Tail Rotation:
        # Modeled directly after Primal Groudon's official 50-frame GIF (`front.gif`):
        # 1. Both massive magma-cored hind legs (`leg_L`, `leg_R`) stay 100% planted on the ground `(0, 0)`.
        # 2. Colossal magma torso & 4-bladed crown head rise and sink (`-1..+1px`) with deep volcanic breathing.
        # 3. `lower_jaw` opens `+1px` during the primary roar (`frames 2..7`) and secondary bark (`frames 11..13`),
        #    exposing the glowing golden-magma & crimson mouth interior pocket.
        # 4. Both 4-clawed Omega arms (`arm_near`, `arm_far`) rotate (`±3.2°`) and orbit menacingly from their shoulder sockets.
        # 5. The 4-bladed crimson tail (`tail`) sways and rotates (`±3.0°`) heavily from the tail root socket!
        body_dy = int(round(1.2 * math.sin(phase - 0.2)))
        body_dx = int(round(0.6 * max(0.0, math.sin(phase - 0.2)) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        jaw_open = 1 if (frame_idx in (2, 3, 4, 5, 6, 7) or frame_idx in (11, 12)) else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; opening is carved by mouth_open

        dorsal_dx = body_dx
        dorsal_dy = body_dy + int(round(0.9 * math.sin(phase - 0.5)))

        anear_dx = body_dx + int(round(1.2 * math.cos(phase - 0.5) * facing))
        anear_dy = body_dy + int(round(1.4 * math.sin(phase - 0.5)))
        anear_ang = 3.2 * math.sin(phase - 0.5) * (-facing)

        afar_dx = body_dx + int(round(1.2 * math.cos(phase - 0.8) * facing))
        afar_dy = body_dy + int(round(1.4 * math.sin(phase - 0.8)))
        afar_ang = 3.2 * math.sin(phase - 0.8) * facing

        tail_dx = body_dx
        tail_dy = int(round(1.5 * math.sin(phase - 1.1)))
        tail_ang = 3.0 * math.sin(phase - 1.1) * facing

    elif action == "walk":
        # `biped_kaiju_stomper`: Earth-Shaking Continental Kaiju Stomp Gait!
        # Alternating left/right magma-cored leg stomp (`lift_px=2, stride_px=2`),
        # 2-beat heavy tectonic bounce, counter-rotating Omega claw arms (`±4.5°` rotation),
        # rhythmic magma roar/bark (`+1px`) on each stomp, and heavy counter-rotating 4-bladed tail!
        bounce = int(round(-1.5 * abs(sin1)))
        sway_x = int(round(1.0 * sin1 * facing))

        torso_dx, torso_dy = sway_x, bounce
        head_dx = sway_x + int(round(0.6 * cos1 * facing))
        head_dy = bounce

        jaw_open = 1 if math.sin(2.0 * phase) > 0.15 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; opening is carved by mouth_open

        dorsal_dx = sway_x
        dorsal_dy = bounce + int(round(1.0 * math.sin(2.0 * phase - 0.4)))

        anear_dx = sway_x + int(round(1.8 * sin1 * facing))
        anear_dy = bounce + int(round(1.4 * cos1))
        anear_ang = 4.5 * sin1 * (-facing)

        afar_dx = sway_x + int(round(-1.8 * sin1 * facing))
        afar_dy = bounce + int(round(-1.4 * cos1))
        afar_ang = -4.5 * sin1 * facing

        tail_dx = sway_x
        tail_dy = bounce + int(round(1.6 * math.sin(2.0 * phase - 1.0)))
        tail_ang = 3.4 * math.sin(2.0 * phase - 1.0) * facing

        if is_back:
            leg_L_dx, leg_L_dy = foot_step_offset(phase, facing, lift_px=1, stride_px=1)
        else:
            leg_L_dx, leg_L_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        leg_R_dx, leg_R_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Primal Groudon "Volcanic Kaiju Roar + Spinning Precipice Blades / Hammer Arm & Eruption Tail Crush"!
        # Frames 0..3 (Wind-up & Thunderous Eruption Roar):
        #   - Braces hind legs, pulls colossal magma body back (`-3px`), flares `dorsal_spikes` (`-1px`),
        #   - Opens `lower_jaw` wide (`+2px` volcanic Kaiju roar exposing the golden-magma throat!),
        #   - Winds both 4-clawed Omega arms high at the shoulder sockets (`-8.5°` rotation).
        # Frames 4..9 (Spinning Precipice Blades / Hammer Arm Lariat & 4-Blade Tail Crush):
        #   - Explosive forward tectonic charge (`+5px` body, `+5.5px` head, `+6px` Omega arms, lead foot `+3px`),
        #   - High-torque rotational claw & arm smash (`+9.5° -> -7.5° -> +8.5°` shoulder pivot rotation + smash arc),
        #   - Massive 4-bladed crimson tail executes a rapid rotational wrecking-ball sweep (`±5.8°` around tail socket),
        #   - Jaws roar wide (`+2px` at frames 2..5), clamp shut on tectonic impact (`0px` at frames 6..7),
        #     and unleash a follow-through volcanic roar (`+1px` at frames 8..9)!
        # Frames 10..15: Heavy follow-through recovery to continental stance.
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
        head_dx = int(round(5.4 * k * facing))
        head_dy = body_dy

        # +2px wide volcanic roar on wind-up & early lunge (frames 2..5), snap shut on impact (6..7), follow-through roar (8..9)
        if frame_idx in (3, 4):
            jaw_open = 2
        elif frame_idx in (1, 2, 5, 8, 9):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; opening is carved by mouth_open

        # Dorsal armor plates & charcoal spikes bristle upward (-1px) during roar & eruption strike
        dorsal_dx = body_dx
        dorsal_dy = body_dy - (1 if frame_idx in (2, 3, 5, 6, 7, 8) else 0)

        # 4-clawed Omega arms: wind high on wind-up, then slam forward (+6px) with a fierce rotational smash!
        anear_dx = int(round(6.0 * k * facing))
        anear_dy = body_dy + (int(round(-1.5 * k)) if frame_idx in (4, 5) else int(round(1.8 * max(0.0, k))))
        afar_dx = int(round(5.8 * k * facing))
        afar_dy = body_dy + int(round(1.6 * k))

        spin_table_near = [0.0, -4.5, -7.0, -9.0, 6.5, 9.5, -7.5, 8.5, -5.5, 3.5, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        spin_table_far = [0.0, 4.5, 7.0, 9.0, -6.5, -9.5, 7.5, -8.5, 5.5, -3.5, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        anear_ang = spin_table_near[frame_idx] * (-facing)
        afar_ang = spin_table_far[frame_idx] * facing

        # 4-bladed crimson tail: whips and spins rapidly around the tail root socket during strike frames (4..9)!
        tail_spin_ang = [0.0, -2.2, -3.8, -5.2, 5.2, -5.8, 5.8, -5.2, 4.2, -2.5, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        tail_whip_dy = [0, 1, 1, 2, -2, 2, -2, 2, -1, 1, 0, 0, 0, 0, 0, 0]
        tail_dx = body_dx
        tail_dy = body_dy + tail_whip_dy[frame_idx]
        tail_ang = tail_spin_ang[frame_idx] * facing

        # Lead hind leg stomps powerfully into the lunge
        lead_x = int(round(3.2 * max(0.0, k) * facing)) if k >= 0 else int(round(1.8 * k * facing))
        rear_x = int(round(2.4 * k * facing))
        if not is_back:
            leg_L_dx, leg_L_dy = lead_x, 0
            leg_R_dx, leg_R_dy = rear_x, 0
        else:
            leg_R_dx, leg_R_dy = lead_x, 0
            leg_L_dx, leg_L_dy = rear_x, 0

    elif action == "hurt":
        # Heavy Kaiju knockback + hit-shake + pained volcanic roar (`lower_jaw` open +1..2px) + Omega arms flung back + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-3.5 * env * facing)) + jit
        body_dy = int(round(-1.0 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-1.2 * env))

        jaw_open = 1 if 1 <= frame_idx <= 7 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; opening is carved by mouth_open

        dorsal_dx = body_dx
        dorsal_dy = body_dy - (1 if 2 <= frame_idx <= 5 else 0)

        anear_dx = body_dx + int(round(-1.2 * env * facing))
        anear_dy = body_dy + int(round(-1.5 * env))
        anear_ang = -5.5 * env * (-facing)

        afar_dx = body_dx + int(round(-1.2 * env * facing))
        afar_dy = body_dy + int(round(-1.5 * env))
        afar_ang = 5.5 * env * facing

        tail_dx = body_dx
        tail_dy = int(round(1.2 * env))
        tail_ang = -3.8 * env * facing

        slide_x = int(round(body_dx * 0.65))
        leg_L_dx, leg_R_dx = slide_x, slide_x

    tail_rot = rotate_layer_nearest(layers["tail"], tail_ang, pivots["tail"])
    afar_rot = rotate_layer_nearest(layers["arm_far"], afar_ang, pivots["arm_far"])
    anear_rot = rotate_layer_nearest(layers["arm_near"], anear_ang, pivots["arm_near"])

    if not is_back:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(layers["dorsal_spikes"], dorsal_dx, dorsal_dy),
            shift_layer_int(afar_rot, afar_dx, afar_dy),
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(anear_rot, anear_dx, anear_dy),
        ]
    else:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(layers["dorsal_spikes"], dorsal_dx, dorsal_dy),
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(anear_rot, anear_dx, anear_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    if jaw_open > 0 and not is_back and "mouth" in pivots:
        line = mouth_open.transform_points(pivots["mouth"], dx=head_dx, dy=head_dy)
        lum = palette_rgba[:, :3].astype(int).sum(axis=1)
        deep = tuple(int(v) for v in palette_rgba[int(np.argmin(lum)), :3])
        out = mouth_open.open_mouth(out, line, amount=jaw_open, deep=deep, inner=MOUTH_INNER, peak=0.65)
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    front_raw = np.array(Image.open(MEGA_DIR / "front.png").convert("RGBA"))
    back_raw = np.array(Image.open(MEGA_DIR / "back.png").convert("RGBA"))

    palette_rgba = extract_palette(front_raw, back_raw)

    # Tight bounds of front (81x78) and back (82x74)
    max_w, max_h = 82, 78
    cell_w = 114
    cell_h = 98
    foot_y = 90

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
                frame_rgba = synthesize_groudon_primal_frame(
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

    # Preserve existing high-quality official 50-frame `front.gif` & `front-reverse.gif`,
    # and cleanly replace broken legacy warped `back.gif` & `back-reverse.gif` with 16-frame `idle` N loop!
    idle_sheet = sheets_by_action["idle"]
    back_frames = [idle_sheet[4 * cell_h:5 * cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    back_rev_frames = [f[:, ::-1, :].copy() for f in back_frames]

    # (2026-11-04 mouth fix) front/back png/gif are left untouched: only write them if missing.
    if not (MEGA_DIR / "back.gif").exists():
        save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
    if not (MEGA_DIR / "back-reverse.gif").exists():
        save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)

    meta = {
        "generator": "forge_groudon_primal_actions",
        "key": "0383_groudon-primal",
        "dex": 383,
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
            if (MEGA_DIR / gif_name).exists():
                shutil.copy2(MEGA_DIR / gif_name, SYNC_MEGA_DIR / gif_name)

    print(f"Built 0383_groudon-primal -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
