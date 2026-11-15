#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Metagross (`0376_metagross-mega`) modeled after Normal Metagross's official
88-frame animation (`0376_metagross/normal/front.gif`) and Mega Metagross's 4-armed supercomputer
steel-fortress anatomy (`front.png` & `back.png`):
  - Exact 8-layer anatomical rigging with true joint pivots (4 giant steel claw-arm shoulder sockets +
    rear thruster-leg pivots):
      1. `rear_claws_legs` (dorsal armor hump, left/right lateral silver claws, and rear Beldum thruster legs —
                            pivoted in sync with magnetic levitation and turbine spins!)
      2. `claw_arm_lower_L` (lower-left giant steel claw gauntlet & 3 silver claws — pivoted at the lower-left
                             shoulder socket via `rotate_layer_nearest` for turbine/drill rotation & Comet Punch!)
      3. `claw_arm_lower_R` (lower-right giant steel claw gauntlet & 3 silver claws — pivoted at the lower-right
                             shoulder socket via `rotate_layer_nearest` for phase-shifted turbine/drill rotation!)
      4. `torso_dome` (central blue steel saucer dome + dark-crimson/black mechanical roar core pocket behind
                       `lower_jaw_spike` exposed when Mega Metagross roars!)
      5. `lower_jaw_spike` (lower jaw steel plate + central sharp silver chin spike — opens `+1px` for rhythmic
                            metallic roar/bark in `idle`/`walk` and `+2px` wide for a thunderous mechanical roar
                            during `attack` wind-up, spinning Comet Punch lunge, and `hurt` recoil!)
      6. `head_golden_x` (golden X-cross armor plate + fierce glowing red eyes + dark eye sockets — 100% rigid
                          integer motion `shift_layer_int` so the golden X and red eyes never deform by a single pixel!)
      7. `claw_arm_upper_L` (upper-left giant steel claw gauntlet & 3 silver claws — pivoted at the upper-left
                             shoulder socket via `rotate_layer_nearest` for high-speed turbine/drill rotation!)
      8. `claw_arm_upper_R` (upper-right foreground giant steel claw gauntlet & 3 silver claws — pivoted at the
                             upper-right shoulder socket via `rotate_layer_nearest` for explosive spinning
                             Comet Punch & Bullet Punch barrages!)
"""

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mouth_open  # noqa: E402

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0376_metagross/mega/metagross-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0376_metagross/mega/metagross-mega"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16

# 2026-11-04 mouth fix: closed mouth line of the front still (native 96x96 coords), the dark seam under the
# golden X between its two lower arms (y=50, x=44..53), ordered hinge -> lip. Front.png faces left, so the
# right end is the back. Back views have no visible mouth.
MOUTH_LINE_NATIVE = [(53.0, 50.0), (49.0, 50.0), (44.0, 50.0)]
MOUTH_DEEP = (16, 16, 16)      # darkest palette colour (outline)
MOUTH_INNER = (213, 74, 82)    # existing red (the crimson core glimpsed at the roar peak)


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
        | ((r == 49) & (g == 57) & (b == 98))
    )


def is_blue_steel_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 65) & (g == 98) & (b == 148))
        | ((r == 74) & (g == 131) & (b == 197))
        | ((r == 123) & (g == 164) & (b == 246))
        | ((r == 49) & (g == 57) & (b == 98))
    )


def is_silver_claw_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 161) & (g == 161) & (b == 161))
        | ((r == 172) & (g == 172) & (b == 172))
        | ((r == 205) & (g == 205) & (b == 205))
        | ((r == 231) & (g == 231) & (b == 231))
        | ((r == 115) & (g == 106) & (b == 115))
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
    Preserves exact palette colors with zero interpolation blur and cleans up tiny 1-3px edge crumbs.
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

    out_a = out[:, :, 3] >= 128
    if np.any(out_a):
        lbl, n_cc = ndi.label(out_a, structure=np.ones((3, 3)))
        if n_cc > 1:
            for c_i in range(1, n_cc + 1):
                cc_m = lbl == c_i
                if int(np.sum(cc_m)) <= 3:
                    out[cc_m] = 0
    return out


def composite_layers(canvas_shape: Tuple[int, int, int], layers: List[np.ndarray]) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=np.uint8)
    for lyr in layers:
        m = lyr[..., 3] >= 128
        out[m] = lyr[m]
    return out


def segment_front_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `front.png` into 8 exact anatomical masks following 1px outlines:
      1. `rear_claws_legs`: Dorsal armor hump (`y=16..23`) + left/right lateral silver spikes + right-rear leg (`y=32..50`).
      2. `claw_arm_lower_L`: Lower-left giant steel claw gauntlet & 3 silver claws (`x=0..35, y=42..80`).
      3. `claw_arm_lower_R`: Lower-right giant steel claw gauntlet & 3 silver claws (`x=57..87, y=49..85`).
      4. `torso_dome`: Central blue steel saucer dome (`x=31..70, y=22..51`).
      5. `lower_jaw_spike`: Lower jaw steel armor plate & central sharp silver chin spike (`x=42..59, y=51..63`).
      6. `head_golden_x`: Golden X-cross armor plate, glowing red eyes & dark eye sockets (`x=37..65, y=23..52`).
      7. `claw_arm_upper_L`: Upper-left giant steel claw gauntlet & 3 silver claws (`x=0..34, y=11..38`).
      8. `claw_arm_upper_R`: Upper-right giant steel claw gauntlet & 3 silver claws (`x=65..95, y=9..40`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    r = arr96[:, :, 0].astype(np.int32)
    g = arr96[:, :, 1].astype(np.int32)
    b = arr96[:, :, 2].astype(np.int32)
    is_gold = a & (
        ((r == 230) & (g == 156) & (b == 0))
        | ((r == 255) & (g == 213) & (b == 0))
        | ((r == 180) & (g == 65) & (b == 0))
        | ((r == 255) & (g == 255) & (b == 98))
    )
    is_red_eye = a & (r == 213) & (g == 74) & (b == 82)

    # 1. claw_arm_upper_L (upper-left giant steel claw arm)
    upper_L_zone = (
        ((yy >= 11) & (yy <= 22) & (xx <= 33))
        | ((yy >= 23) & (yy <= 26) & (xx <= 34))
        | ((yy == 27) & (xx <= 33))
        | ((yy >= 28) & (yy <= 29) & (xx <= 32))
        | ((yy == 30) & (xx <= 31))
        | ((yy >= 31) & (yy <= 32) & (xx <= 30))
        | ((yy >= 33) & (yy <= 38) & (xx <= 29))
    )
    claw_arm_upper_L = a & upper_L_zone & (~is_gold)

    # 2. claw_arm_upper_R (upper-right giant steel claw arm)
    upper_R_zone = (
        ((yy >= 9) & (yy <= 25) & (xx >= 65))
        | ((yy >= 26) & (yy <= 28) & (xx >= 67))
        | ((yy >= 29) & (yy <= 31) & (xx >= 65))
        | ((yy == 32) & (xx >= 66))
        | ((yy == 33) & (xx >= 67))
        | ((yy == 34) & (xx >= 70))
        | ((yy >= 35) & (yy <= 40) & (xx >= 74))
    )
    claw_arm_upper_R = a & upper_R_zone & (~is_gold)

    # 3. claw_arm_lower_L (lower-left giant steel claw arm)
    lower_L_zone = (
        ((yy == 42) & (xx <= 33))
        | ((yy >= 43) & (yy <= 80) & (xx <= 35))
    )
    claw_arm_lower_L = a & lower_L_zone

    # 4. claw_arm_lower_R (lower-right giant steel claw arm including upper-arm shoulder joint at y=41..50)
    lower_R_zone = (
        ((yy == 41) & (xx >= 77) & (xx <= 82))
        | ((yy >= 42) & (yy <= 43) & (xx >= 76) & (xx <= 82))
        | ((yy == 44) & (xx >= 75) & (xx <= 81))
        | ((yy == 45) & (xx >= 74) & (xx <= 81))
        | ((yy == 46) & (xx >= 73) & (xx <= 80))
        | ((yy >= 47) & (yy <= 48) & (xx >= 72) & (xx <= 79))
        | ((yy == 49) & (xx >= 67) & (xx <= 79))
        | ((yy == 50) & (xx >= 67) & (xx <= 80))
        | ((yy >= 51) & (yy <= 53) & (xx >= 67))
        | ((yy >= 54) & (yy <= 85) & (xx >= 57))
    )
    claw_arm_lower_R = a & lower_R_zone

    # 5. rear_claws_legs (dorsal armor hump + left/right lateral silver crescent claws)
    rear_zone = (
        ((yy >= 16) & (yy <= 21) & (xx >= 35) & (xx <= 58))
        | ((yy == 22) & (xx >= 35) & (xx <= 46))
        | ((yy == 23) & (xx >= 35) & (xx <= 36))
        # Left lateral silver spike
        | ((yy == 32) & (xx >= 33) & (xx <= 35))
        | ((yy == 33) & (xx >= 32) & (xx <= 35))
        | ((yy >= 34) & (yy <= 35) & (xx >= 31) & (xx <= 36))
        | ((yy == 36) & (xx >= 31) & (xx <= 35))
        | ((yy == 37) & (xx >= 30) & (xx <= 35))
        | ((yy == 38) & (xx >= 30) & (xx <= 33))
        | ((yy == 39) & (xx >= 31) & (xx <= 33))
        | ((yy == 40) & (xx >= 32) & (xx <= 33))
        # Right lateral silver crescent claw (y=35..47)
        | ((yy == 35) & (xx >= 69) & (xx <= 71))
        | ((yy == 36) & (xx >= 69) & (xx <= 73))
        | ((yy == 37) & (xx >= 68) & (xx <= 74))
        | ((yy == 38) & (xx >= 67) & (xx <= 75))
        | ((yy >= 39) & (yy <= 41) & (xx >= 72) & (xx <= 76))
        | ((yy == 42) & (xx >= 72) & (xx <= 75))
        | ((yy == 43) & (xx >= 71) & (xx <= 75))
        | ((yy == 44) & (xx >= 71) & (xx <= 74))
        | ((yy == 45) & (xx >= 70) & (xx <= 73))
        | ((yy == 46) & (xx >= 69) & (xx <= 72))
        | ((yy == 47) & (xx >= 70) & (xx <= 71))
    )
    rear_claws_legs = (
        a
        & rear_zone
        & (~claw_arm_upper_L)
        & (~claw_arm_upper_R)
        & (~claw_arm_lower_L)
        & (~claw_arm_lower_R)
        & (~is_gold)
    )

    # 6. head_golden_x (Golden X-cross armor + red eyes + eye socket dark pixels)
    eye_socket_zone = (
        ((yy >= 38) & (yy <= 42) & (xx >= 41) & (xx <= 45))
        | ((yy >= 38) & (yy <= 42) & (xx >= 52) & (xx <= 58))
    )
    head_golden_x = (
        a
        & (is_gold | is_red_eye | eye_socket_zone)
        & (~claw_arm_upper_L)
        & (~claw_arm_upper_R)
    )

    # 7. lower_jaw_spike (lower jaw steel armor plate + central silver chin spike)
    jaw_zone = (
        ((yy >= 51) & (yy <= 53) & (xx >= 42) & (xx <= 62))
        | ((yy >= 54) & (yy <= 63) & (xx >= 44) & (xx <= 59))
    )
    lower_jaw_spike = (
        a
        & jaw_zone
        & (~head_golden_x)
        & (~claw_arm_lower_L)
        & (~claw_arm_lower_R)
    )

    # 8. torso_dome (central blue steel saucer dome)
    used = (
        claw_arm_upper_L
        | claw_arm_upper_R
        | claw_arm_lower_L
        | claw_arm_lower_R
        | rear_claws_legs
        | head_golden_x
        | lower_jaw_spike
    )
    torso_dome = a & (~used)

    return {
        "rear_claws_legs": rear_claws_legs,
        "claw_arm_lower_L": claw_arm_lower_L,
        "claw_arm_lower_R": claw_arm_lower_R,
        "torso_dome": torso_dome,
        "lower_jaw_spike": lower_jaw_spike,
        "head_golden_x": head_golden_x,
        "claw_arm_upper_L": claw_arm_upper_L,
        "claw_arm_upper_R": claw_arm_upper_R,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 8 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. claw_arm_upper_R (massive forward-right upper steel claw arm y=21..42, x=50..91)
    upper_R_zone = (
        ((yy >= 21) & (yy <= 29) & (xx >= 62))
        | ((yy == 30) & (xx >= 59))
        | ((yy == 31) & (xx >= 55))
        | ((yy == 32) & (xx >= 53))
        | ((yy == 33) & (xx >= 51))
        | ((yy == 34) & (xx >= 50))
        | ((yy >= 35) & (yy <= 36) & (xx >= 51))
        | ((yy == 37) & (xx >= 52))
        | ((yy == 38) & (xx >= 56))
        | ((yy >= 39) & (yy <= 42) & (xx >= 66))
    )
    claw_arm_upper_R = a & upper_R_zone

    # 2. claw_arm_lower_R (massive forward-right lower steel claw arm y=55..75, x=51..91)
    lower_R_zone = (
        ((yy >= 55) & (yy <= 59) & (xx >= 66))
        | ((yy >= 60) & (yy <= 62) & (xx >= 51))
        | ((yy >= 63) & (yy <= 64) & (xx >= 53))
        | ((yy >= 65) & (yy <= 66) & (xx >= 55))
        | ((yy == 67) & (xx >= 59))
        | ((yy >= 68) & (yy <= 75) & (xx >= 64))
    )
    claw_arm_lower_R = a & lower_R_zone

    # 3. claw_arm_upper_L (far-left upper rear thruster leg y=35..48, x=0..30)
    upper_L_zone = (
        ((yy >= 35) & (yy <= 38) & (xx <= 19))
        | ((yy == 39) & (xx <= 26))
        | ((yy == 40) & (xx <= 23))
        | ((yy == 41) & (xx <= 25))
        | ((yy == 42) & (xx <= 27))
        | ((yy == 43) & (xx <= 30))
        | ((yy == 44) & (xx <= 17))
        | ((yy == 45) & (xx <= 16))
        | ((yy == 46) & (xx <= 14))
        | ((yy == 47) & (xx <= 11))
        | ((yy == 48) & (xx <= 8))
    )
    claw_arm_upper_L = a & upper_L_zone

    # 4. claw_arm_lower_L (far-left lower rear thruster leg y=51..61, x=10..29)
    lower_L_zone = (
        ((yy >= 51) & (yy <= 52) & (xx <= 20))
        | ((yy == 53) & (xx <= 22))
        | ((yy == 54) & (xx <= 23))
        | ((yy == 55) & (xx <= 26))
        | ((yy >= 56) & (yy <= 57) & (xx <= 29))
        | ((yy == 58) & (xx <= 25))
        | ((yy == 59) & (xx <= 21))
        | ((yy == 60) & (xx <= 19))
        | ((yy == 61) & (xx <= 17))
    )
    claw_arm_lower_L = a & lower_L_zone

    # 5. rear_claws_legs (central pair of rear Beldum thruster legs protruding toward the camera)
    rear_zone = (
        ((yy == 33) & (xx >= 36) & (xx <= 45))
        | ((yy == 34) & (xx >= 35) & (xx <= 49))
        | ((yy == 35) & (xx >= 34) & (xx <= 50))
        | ((yy >= 36) & (yy <= 37) & (xx >= 32) & (xx <= 51))
        | ((yy == 38) & (xx >= 31) & (xx <= 55))
        | ((yy >= 39) & (yy <= 40) & (xx >= 30) & (xx <= 58))
        | ((yy >= 41) & (yy <= 42) & (xx >= 30) & (xx <= 56))
        | ((yy >= 43) & (yy <= 44) & (xx >= 31) & (xx <= 48))
        | ((yy == 45) & (xx >= 32) & (xx <= 47))
        | ((yy == 46) & (xx >= 33) & (xx <= 46))
        | ((yy == 51) & (xx >= 36) & (xx <= 42))
        | ((yy == 52) & (xx >= 33) & (xx <= 53))
        | ((yy == 53) & (xx >= 33) & (xx <= 54))
        | ((yy >= 54) & (yy <= 58) & (xx >= 32) & (xx <= 55))
        | ((yy == 59) & (xx >= 34) & (xx <= 54))
        | ((yy == 60) & (xx >= 35) & (xx <= 50))
    )
    rear_claws_legs = (
        a
        & rear_zone
        & (~claw_arm_upper_R)
        & (~claw_arm_lower_R)
        & (~claw_arm_upper_L)
        & (~claw_arm_lower_L)
    )

    # 6. head_golden_x (rear silver cross spikes & dorsal crest tips)
    head_zone = (
        ((yy >= 24) & (yy <= 33) & (xx >= 47) & (xx <= 59))
        | ((yy >= 44) & (yy <= 54) & (xx >= 54) & (xx <= 64))
    )
    head_golden_x = (
        a
        & head_zone
        & (~claw_arm_upper_R)
        & (~claw_arm_lower_R)
        & (~rear_claws_legs)
    )

    empty = np.zeros_like(a)
    used = (
        claw_arm_upper_R
        | claw_arm_lower_R
        | claw_arm_upper_L
        | claw_arm_lower_L
        | rear_claws_legs
        | head_golden_x
    )
    torso_dome = a & (~used)

    return {
        "rear_claws_legs": rear_claws_legs,
        "claw_arm_lower_L": claw_arm_lower_L,
        "claw_arm_lower_R": claw_arm_lower_R,
        "torso_dome": torso_dome,
        "lower_jaw_spike": empty,
        "head_golden_x": head_golden_x,
        "claw_arm_upper_L": claw_arm_upper_L,
        "claw_arm_upper_R": claw_arm_upper_R,
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
    Strictly ensures no background layer is ever added to `foreground_mask`.
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
    # Back sprite is 55px tall (floating higher); center it vertically with front sprite's body mass
    dst_y0 = (foot_y - 8 - sh) if is_back else (foot_y - sh)

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

    blue_fill_m = is_blue_steel_fill(canvas)
    silver_fill_m = is_silver_claw_fill(canvas)
    steel_or_silver_m = blue_fill_m | silver_fill_m
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order (strictly Back-to-Front):
        # 1. claw_arm_lower_L -> 2. claw_arm_lower_R -> 3. rear_claws_legs ->
        # 4. torso_dome -> 5. lower_jaw_spike -> 6. head_golden_x ->
        # 7. claw_arm_upper_L -> 8. claw_arm_upper_R
        fg_over_lowL = (
            c_masks["rear_claws_legs"]
            | c_masks["torso_dome"]
            | c_masks["lower_jaw_spike"]
            | c_masks["head_golden_x"]
        )
        u_lowL = ndi.binary_erosion(c_masks["claw_arm_lower_L"] | fg_over_lowL, iterations=1)
        layers["claw_arm_lower_L"] = build_part_layer(
            canvas,
            c_masks["claw_arm_lower_L"],
            u_lowL & fg_over_lowL,
            cushion_px=2,
            allowed_fill_mask=blue_fill_m,
        )

        fg_over_lowR = (
            c_masks["rear_claws_legs"]
            | c_masks["torso_dome"]
            | c_masks["lower_jaw_spike"]
            | c_masks["head_golden_x"]
            | c_masks["claw_arm_upper_R"]
        )
        u_lowR = ndi.binary_erosion(c_masks["claw_arm_lower_R"] | fg_over_lowR, iterations=1)
        layers["claw_arm_lower_R"] = build_part_layer(
            canvas,
            c_masks["claw_arm_lower_R"],
            u_lowR & fg_over_lowR,
            cushion_px=2,
            allowed_fill_mask=blue_fill_m,
        )

        fg_over_rear = (
            c_masks["torso_dome"]
            | c_masks["head_golden_x"]
            | c_masks["claw_arm_upper_L"]
            | c_masks["claw_arm_upper_R"]
        )
        u_rear = ndi.binary_erosion(c_masks["rear_claws_legs"] | fg_over_rear, iterations=1)
        layers["rear_claws_legs"] = build_part_layer(
            canvas,
            c_masks["rear_claws_legs"],
            u_rear & fg_over_rear,
            cushion_px=2,
            allowed_fill_mask=steel_or_silver_m,
        )

        # IMPORTANT: `torso_dome` is in front of `rear_claws_legs` and `claw_arm_lower_L/R`.
        # Its `foreground_mask` ONLY contains layers drawn ON TOP OF `torso_dome`!
        fg_over_torso = (
            c_masks["lower_jaw_spike"]
            | c_masks["head_golden_x"]
            | c_masks["claw_arm_upper_L"]
            | c_masks["claw_arm_upper_R"]
        )
        u_torso = ndi.binary_erosion(c_masks["torso_dome"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso_dome"],
            u_torso & fg_over_torso,
            cushion_px=2,
            allowed_fill_mask=blue_fill_m,
        )

        # 2026-11-04 (mouth fix): the rectangular crimson "roar core" pocket behind the jaw is no longer painted.
        # The jaw stays put and `mouth_open.open_mouth` carves a small lens along the mouth line instead.
        layers["torso_dome"] = torso_lyr

        fg_over_jaw = c_masks["head_golden_x"]
        u_jaw = ndi.binary_erosion(c_masks["lower_jaw_spike"] | fg_over_jaw, iterations=1)
        layers["lower_jaw_spike"] = build_part_layer(
            canvas,
            c_masks["lower_jaw_spike"],
            u_jaw & fg_over_jaw,
            cushion_px=2,
            allowed_fill_mask=blue_fill_m,
        )

        layers["head_golden_x"] = build_part_layer(
            canvas,
            c_masks["head_golden_x"],
            np.zeros_like(c_masks["head_golden_x"]),
            cushion_px=0,
        )

        layers["claw_arm_upper_L"] = build_part_layer(
            canvas,
            c_masks["claw_arm_upper_L"],
            np.zeros_like(c_masks["claw_arm_upper_L"]),
            cushion_px=0,
        )
        layers["claw_arm_upper_R"] = build_part_layer(
            canvas,
            c_masks["claw_arm_upper_R"],
            np.zeros_like(c_masks["claw_arm_upper_R"]),
            cushion_px=0,
        )

        pivots = {
            "rear_claws_legs": to_canvas_pt(52.0, 36.0),
            "claw_arm_upper_L": to_canvas_pt(31.0, 27.0),
            "claw_arm_upper_R": to_canvas_pt(68.0, 30.0),
            "claw_arm_lower_L": to_canvas_pt(32.0, 46.0),
            "claw_arm_lower_R": to_canvas_pt(75.0, 44.0),
        }

    else:
        # Back view draw order (strictly Back-to-Front):
        # 1. claw_arm_upper_R -> 2. claw_arm_lower_R -> 3. claw_arm_upper_L -> 4. claw_arm_lower_L ->
        # 5. torso_dome -> 6. head_golden_x -> 7. rear_claws_legs
        layers["lower_jaw_spike"] = np.zeros_like(canvas)

        fg_over_upR = c_masks["torso_dome"] | c_masks["head_golden_x"] | c_masks["rear_claws_legs"]
        u_upR = ndi.binary_erosion(c_masks["claw_arm_upper_R"] | fg_over_upR, iterations=1)
        layers["claw_arm_upper_R"] = build_part_layer(
            canvas,
            c_masks["claw_arm_upper_R"],
            u_upR & fg_over_upR,
            cushion_px=5,
            allowed_fill_mask=blue_fill_m,
        )

        fg_over_lowR = c_masks["torso_dome"] | c_masks["head_golden_x"] | c_masks["rear_claws_legs"]
        u_lowR = ndi.binary_erosion(c_masks["claw_arm_lower_R"] | fg_over_lowR, iterations=1)
        layers["claw_arm_lower_R"] = build_part_layer(
            canvas,
            c_masks["claw_arm_lower_R"],
            u_lowR & fg_over_lowR,
            cushion_px=5,
            allowed_fill_mask=blue_fill_m,
        )

        fg_over_upL = c_masks["torso_dome"] | c_masks["rear_claws_legs"]
        u_upL = ndi.binary_erosion(c_masks["claw_arm_upper_L"] | fg_over_upL, iterations=1)
        layers["claw_arm_upper_L"] = build_part_layer(
            canvas,
            c_masks["claw_arm_upper_L"],
            u_upL & fg_over_upL,
            cushion_px=4,
            allowed_fill_mask=blue_fill_m,
        )

        fg_over_lowL = c_masks["torso_dome"] | c_masks["rear_claws_legs"]
        u_lowL = ndi.binary_erosion(c_masks["claw_arm_lower_L"] | fg_over_lowL, iterations=1)
        layers["claw_arm_lower_L"] = build_part_layer(
            canvas,
            c_masks["claw_arm_lower_L"],
            u_lowL & fg_over_lowL,
            cushion_px=4,
            allowed_fill_mask=blue_fill_m,
        )

        fg_over_torso = c_masks["head_golden_x"] | c_masks["rear_claws_legs"]
        u_torso = ndi.binary_erosion(c_masks["torso_dome"] | fg_over_torso, iterations=1)
        layers["torso_dome"] = build_part_layer(
            canvas,
            c_masks["torso_dome"],
            u_torso & fg_over_torso,
            cushion_px=5,
            allowed_fill_mask=blue_fill_m,
        )

        fg_over_head = c_masks["rear_claws_legs"]
        u_head = ndi.binary_erosion(c_masks["head_golden_x"] | fg_over_head, iterations=1)
        layers["head_golden_x"] = build_part_layer(
            canvas,
            c_masks["head_golden_x"],
            u_head & fg_over_head,
            cushion_px=3,
            allowed_fill_mask=silver_fill_m,
        )

        layers["rear_claws_legs"] = build_part_layer(
            canvas,
            c_masks["rear_claws_legs"],
            np.zeros_like(c_masks["rear_claws_legs"]),
            cushion_px=0,
        )

        pivots = {
            "rear_claws_legs": to_canvas_pt(48.0, 47.0),
            "claw_arm_upper_L": to_canvas_pt(23.0, 41.0),
            "claw_arm_upper_R": to_canvas_pt(56.0, 34.0),
            "claw_arm_lower_L": to_canvas_pt(25.0, 56.0),
            "claw_arm_lower_R": to_canvas_pt(57.0, 61.0),
        }

    return layers, pivots


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
        (mid_x - 20 - drift, max(2, top_y + 12 - drift // 2)),
        (mid_x + 20 + drift, max(2, top_y + 14 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (255, 255, 98, 255)
    return out


def synthesize_metagross_mega_frame(
    layers: Dict[str, np.ndarray],
    pivots: Dict[str, Tuple[float, float]],
    canvas_shape: Tuple[int, int, int],
    action: str,
    frame_idx: int,
    facing: float,
    is_back: bool,
    palette_rgba: np.ndarray,
    mouth_line: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    t = frame_idx / float(N_FRAMES)
    phase = 2.0 * math.pi * t
    sin1 = math.sin(phase)
    cos1 = math.cos(phase)

    torso_dx, torso_dy = 0, 0
    head_dx, head_dy = 0, 0
    jaw_dx, jaw_dy = 0, 0
    jaw_open = 0
    rear_dx, rear_dy = 0, 0

    upL_dx, upL_dy = 0, 0
    upR_dx, upR_dy = 0, 0
    lowL_dx, lowL_dy = 0, 0
    lowR_dx, lowR_dy = 0, 0

    rear_ang = 0.0
    upL_ang = 0.0
    upR_ang = 0.0
    lowL_ang = 0.0
    lowR_ang = 0.0

    # `side_sign` is `+1.0` when unflipped (`facing == -1.0` on front, `+1.0` on back),
    # and `-1.0` when horizontally flipped (`flip_x == True`).
    side_sign = facing if is_back else (-facing)

    if action == "idle":
        # Magnetic Levitation Hover + 4-Arm Turbine Rotor Drive + Metallic Kaiju Roar / Bark:
        # 1. Heavy electromagnetic hover oscillation (`-2..+2px` vertical float + subtle forward sway).
        # 2. `lower_jaw_spike` opens `+1px` during the primary metallic roar (`frames 2..7`) and
        #    secondary bark (`frames 11..13`), exposing the glowing crimson interior core!
        # 3. All 4 giant steel claw arms (`claw_arm_upper_L/R`, `claw_arm_lower_L/R`) continuously
        #    rotate around their shoulder sockets (`±8.0°..9.5°`) with phase offset like a massive
        #    4-bladed magnetic turbine!
        hover_dy = int(round(1.6 * math.sin(phase - 0.2)))
        hover_dx = int(round(0.8 * math.cos(phase - 0.2) * facing))

        torso_dx, torso_dy = hover_dx, hover_dy
        head_dx, head_dy = hover_dx, hover_dy

        if frame_idx in (2, 3, 4, 5, 6, 7) or frame_idx in (11, 12, 13):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; opening is carved by mouth_open

        rear_dx = hover_dx
        rear_dy = hover_dy + int(round(0.8 * math.sin(phase - 0.8)))
        rear_ang = 3.5 * math.sin(phase - 0.6) * side_sign

        up_wave = math.sin(phase - 0.4)
        low_wave = math.sin(phase - 1.4)
        upL_ang = 8.0 * up_wave * side_sign
        upR_ang = -8.5 * up_wave * side_sign
        lowL_ang = -9.0 * low_wave * side_sign
        lowR_ang = 9.0 * low_wave * side_sign

        upL_dx = hover_dx + int(round(-0.8 * up_wave * side_sign))
        upL_dy = hover_dy + int(round(1.2 * up_wave))
        upR_dx = hover_dx + int(round(0.8 * up_wave * side_sign))
        upR_dy = hover_dy + int(round(-1.2 * up_wave))

        lowL_dx = hover_dx + int(round(0.8 * low_wave * side_sign))
        lowL_dy = hover_dy + int(round(1.4 * low_wave))
        lowR_dx = hover_dx + int(round(-0.8 * low_wave * side_sign))
        lowR_dy = hover_dy + int(round(-1.4 * low_wave))

    elif action == "walk":
        # High-Speed Electromagnetic Glide + Alternating 4-Claw Turbine Thruster Swing + Roaring Chant:
        glide_dx = int(round(1.8 * sin1 * facing))
        glide_dy = int(round(-1.5 * abs(sin1) + 0.8 * math.cos(2.0 * phase)))

        torso_dx, torso_dy = glide_dx, glide_dy
        head_dx = glide_dx + int(round(0.7 * cos1 * facing))
        head_dy = glide_dy

        jaw_open = 1 if math.sin(2.0 * phase) > 0.10 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; opening is carved by mouth_open

        rear_dx = glide_dx + int(round(-1.0 * cos1 * facing))
        rear_dy = glide_dy + int(round(1.0 * sin1))
        rear_ang = 5.0 * sin1 * side_sign

        # Diagonal claw-arm pairs (`upper_L + lower_R` vs `upper_R + lower_L`) spin and thrust in opposition
        upL_ang = 11.5 * sin1 * side_sign
        upR_ang = 11.5 * cos1 * (-side_sign)
        lowL_ang = -12.5 * cos1 * side_sign
        lowR_ang = -12.5 * sin1 * (-side_sign)

        upL_dx = glide_dx + int(round(1.5 * cos1 * facing))
        upL_dy = glide_dy + int(round(-1.4 * sin1))
        upR_dx = glide_dx + int(round(-1.5 * cos1 * facing))
        upR_dy = glide_dy + int(round(1.4 * sin1))

        lowL_dx = glide_dx + int(round(-1.6 * sin1 * facing))
        lowL_dy = glide_dy + int(round(1.5 * cos1))
        lowR_dx = glide_dx + int(round(1.6 * sin1 * facing))
        lowR_dy = glide_dy + int(round(-1.5 * cos1))

    elif action == "attack":
        # Signature Mega Metagross "4-Claw Turbine Drill Spin + Thunderous Mechanical Roar (`+2px` jaw) +
        # Comet Punch & Bullet Punch Super-Heavy Lunge (`+7px`)"!
        # Frames 0..3 (Wind-up & X-Turbine Spread Charge + Full Mechanical Roar):
        #   - Coils back (`-3px` back, `-1px` magnetic lift),
        #   - Opens `lower_jaw_spike` wide (`+2px` roar exposing crimson core!),
        #   - Spreads all 4 giant steel claw arms wide into a charged X-formation (`±16°..18°` rotation).
        # Frames 4..9 (High-Speed Turbine Drill Spin & Comet Punch / Bullet Punch Rush `+7px` Lunge):
        #   - Explosive forward magnetic rocket lunge (`+7px` forward),
        #   - All 4 giant steel claw arms spin rapidly like a 4-bladed drill turbine (`+20° -> -18° -> +19° -> -15°`)
        #     while thrusting `+8px` forward in an overwhelming Comet Punch & Bullet Punch barrage!
        #   - `lower_jaw_spike` stays wide open (`+2px`) roaring throughout the strike!
        # Frames 10..15: Electromagnetic braking & turbine spin-down recovery.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.55 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.22 * math.sin(u * math.pi)

        body_dx = int(round(7.0 * k * facing))
        body_dy_table = [0, -1, -2, -1, 1, 2, 1, 0, 1, 0, -1, 0, 0, 0, 0, 0]
        body_dy = body_dy_table[frame_idx]

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        if frame_idx in (3, 4, 5):
            jaw_open = 2  # roar peak
        elif frame_idx in (1, 2, 6, 7, 8, 9):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; opening is carved by mouth_open

        rear_ang_table = [0.0, -4.0, -7.0, -9.0, 8.0, 11.0, -9.0, 8.0, -6.0, 4.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rear_dx = int(round(6.0 * k * facing))
        rear_dy = body_dy
        rear_ang = rear_ang_table[frame_idx] * side_sign

        # 4 Giant Steel Claw Arms: X-Charge (0..3) -> High-Speed Turbine Drill Spin & Comet Punch Rush (4..9) -> Recovery (10..15)
        upL_spin = [0.0, -8.0, -14.0, -18.0, 16.0, 20.0, -17.0, 19.0, -15.0, 10.0, -4.5, -1.5, 0.0, 0.0, 0.0, 0.0]
        upR_spin = [0.0, 8.0, 14.0, 18.0, -16.0, -20.0, 17.0, -19.0, 15.0, -10.0, 4.5, 1.5, 0.0, 0.0, 0.0, 0.0]
        lowL_spin = [0.0, 7.5, 13.5, 17.5, -17.0, 19.5, -18.5, 16.5, -13.5, 9.0, -4.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        lowR_spin = [0.0, -7.5, -13.5, -17.5, 17.0, -19.5, 18.5, -16.5, 13.5, -9.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        upL_ang = upL_spin[frame_idx] * side_sign
        upR_ang = upR_spin[frame_idx] * side_sign
        lowL_ang = lowL_spin[frame_idx] * side_sign
        lowR_ang = lowR_spin[frame_idx] * side_sign

        punch_rush_dx = int(round(8.2 * k * facing))
        # Alternating Comet Punch / Bullet Punch extensions on frames 4..9
        upL_ext = int(round((2.0 if frame_idx in (4, 6, 8) else (-1.0 if frame_idx in (5, 7) else 0.0)) * facing))
        upR_ext = int(round((2.0 if frame_idx in (5, 7, 9) else (-1.0 if frame_idx in (4, 6, 8) else 0.0)) * facing))
        lowL_ext = int(round((2.0 if frame_idx in (5, 7) else (-1.0 if frame_idx in (4, 6, 8) else 0.0)) * facing))
        lowR_ext = int(round((2.0 if frame_idx in (4, 6, 8) else (-1.0 if frame_idx in (5, 7) else 0.0)) * facing))

        up_dy_shift = -2 if frame_idx in (2, 3, 5, 7) else (2 if frame_idx in (4, 6, 8) else 0)
        low_dy_shift = 2 if frame_idx in (2, 3, 5, 7) else (-2 if frame_idx in (4, 6, 8) else 0)

        upL_dx, upL_dy = punch_rush_dx + upL_ext, body_dy + up_dy_shift
        upR_dx, upR_dy = punch_rush_dx + upR_ext, body_dy - up_dy_shift
        lowL_dx, lowL_dy = punch_rush_dx + lowL_ext, body_dy + low_dy_shift
        lowR_dx, lowR_dy = punch_rush_dx + lowR_ext, body_dy - low_dy_shift

    elif action == "hurt":
        # Heavy Knockback Recoil + Hit-Shake + Pained Metallic Roar (`lower_jaw_spike` open +2px) +
        # 4 Giant Claw Arms Flared Wide + Electromagnetic Sparks!
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-4.0 * env * facing)) + jit
        body_dy = int(round(-1.5 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        jaw_open = 1 if 1 <= frame_idx <= 7 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the head; opening is carved by mouth_open

        rear_dx = body_dx
        rear_dy = body_dy + int(round(-1.0 * env))
        rear_ang = -7.0 * env * side_sign

        upL_ang = -14.0 * env * side_sign
        upR_ang = 14.0 * env * side_sign
        lowL_ang = 13.5 * env * side_sign
        lowR_ang = -13.5 * env * side_sign

        upL_dx, upL_dy = body_dx - int(round(1.5 * env * side_sign)), body_dy - int(round(1.5 * env))
        upR_dx, upR_dy = body_dx + int(round(1.5 * env * side_sign)), body_dy - int(round(1.5 * env))
        lowL_dx, lowL_dy = body_dx - int(round(1.5 * env * side_sign)), body_dy + int(round(1.5 * env))
        lowR_dx, lowR_dy = body_dx + int(round(1.5 * env * side_sign)), body_dy + int(round(1.5 * env))

    rear_rot = rotate_layer_nearest(layers["rear_claws_legs"], rear_ang, pivots["rear_claws_legs"])
    upL_rot = rotate_layer_nearest(layers["claw_arm_upper_L"], upL_ang, pivots["claw_arm_upper_L"])
    upR_rot = rotate_layer_nearest(layers["claw_arm_upper_R"], upR_ang, pivots["claw_arm_upper_R"])
    lowL_rot = rotate_layer_nearest(layers["claw_arm_lower_L"], lowL_ang, pivots["claw_arm_lower_L"])
    lowR_rot = rotate_layer_nearest(layers["claw_arm_lower_R"], lowR_ang, pivots["claw_arm_lower_R"])

    if not is_back:
        comp = [
            shift_layer_int(lowL_rot, lowL_dx, lowL_dy),
            shift_layer_int(lowR_rot, lowR_dx, lowR_dy),
            shift_layer_int(rear_rot, rear_dx, rear_dy),
            shift_layer_int(layers["torso_dome"], torso_dx, torso_dy),
            shift_layer_int(layers["lower_jaw_spike"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_golden_x"], head_dx, head_dy),
            shift_layer_int(upL_rot, upL_dx, upL_dy),
            shift_layer_int(upR_rot, upR_dx, upR_dy),
        ]
    else:
        comp = [
            shift_layer_int(upR_rot, upR_dx, upR_dy),
            shift_layer_int(lowR_rot, lowR_dx, lowR_dy),
            shift_layer_int(upL_rot, upL_dx, upL_dy),
            shift_layer_int(lowL_rot, lowL_dx, lowL_dy),
            shift_layer_int(layers["torso_dome"], torso_dx, torso_dy),
            shift_layer_int(layers["head_golden_x"], head_dx, head_dy),
            shift_layer_int(rear_rot, rear_dx, rear_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    if mouth_line is not None and jaw_open > 0:
        line = mouth_open.transform_points(mouth_line, dx=head_dx, dy=head_dy)
        opened = mouth_open.open_mouth(out, line, amount=jaw_open, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=0.5)
        # never carve into the upper claw arms drawn over the face
        arms = (comp[-1][..., 3] >= 128) | (comp[-2][..., 3] >= 128)
        opened[arms] = out[arms]
        out = opened
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

    # Tight bounds of front (96x77) and back (92x55), padded for 4-arm turbine spin & +8px Comet Punch lunge
    max_w, max_h = 96, 77
    cell_w = 124
    cell_h = 108
    foot_y = 96

    mouth_lines: Dict[str, Optional[List[Tuple[float, float]]]] = {}
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
        if use_back:
            mouth_lines[d_name] = None
        else:
            # native -> cell: same placement as build_rig_layers (front still is 96 wide, x0=0, y0=9)
            fy_ys = np.where(front_raw[:, :, 3] >= 128)[0]
            ox = (cell_w - 96) // 2
            oy = (foot_y - (int(fy_ys.max()) + 1 - int(fy_ys.min()))) - int(fy_ys.min())
            pts = [(x + ox, y + oy) for x, y in MOUTH_LINE_NATIVE]
            mouth_lines[d_name] = mouth_open.transform_points(pts, mirror_w=cell_w) if flip_x else pts

    sheets_by_action: Dict[str, np.ndarray] = {}
    master_rows: List[np.ndarray] = []

    for act in ACTIONS:
        act_sheet = np.zeros((8 * cell_h, N_FRAMES * cell_w, 4), dtype=np.uint8)
        for d_idx, d_name in enumerate(DIRS):
            layers, pivots, facing, use_back = dir_rigs[d_name]
            for f_idx in range(N_FRAMES):
                frame_rgba = synthesize_metagross_mega_frame(
                    layers,
                    pivots,
                    (cell_h, cell_w, 4),
                    act,
                    f_idx,
                    facing,
                    use_back,
                    palette_rgba,
                    mouth_lines[d_name],
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

    # Replace warped 19f `front.gif`, `front-reverse.gif`, `back.gif`, `back-reverse.gif`
    # with clean 16-frame `idle` S/N loop!
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
        "generator": "forge_metagross_mega_actions",
        "key": "0376_metagross-mega",
        "dex": 376,
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

    print(f"Built 0376_metagross-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
