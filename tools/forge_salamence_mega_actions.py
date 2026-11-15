#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Salamence (`0373_salamence-mega`) modeled after its official 103-frame GIF
(`0373_salamence/mega/salamence-mega/front.gif`) and Blood-Soaked Crescent anatomy (`front.png` & `back.png`):
  - Exact 8-layer anatomical rigging with true joint pivots (dorsal wing sockets & tail root socket):
      1. `tail` (long blue & crimson dragon tail — 100% separated from the hind legs, pivoted at its socket
                 with `rotate_layer_nearest` to whip and slice through supersonic air currents!)
      2. `crescent_wing_far` (far-side crimson crescent wing blade — pivoted at the dorsal shoulder socket
                              for aerodynamic banking & corkscrew barrel-roll slashes!)
      3. `torso_shell` (streamlined silver chest armor housing the tucked forelegs, blue dragon body, dorsal
                        wing mantle; the old rectangular mouth pocket was removed 2026-11-04)
      4. `hind_leg_L` (left hind leg & 3 white claws — trails and flexes with supersonic aerial glide)
      5. `hind_leg_R` (right hind legs & 3 white claws — trails and drives with aerial glide & lunge)
      6. `lower_jaw` (fanged lower jaw — since 2026-11-04 it no longer drops; mouth_open carves a small opening; old notes:
                      `idle`/`walk`, opens WIDE `+2px` during `attack` wind-up roar, and snaps shut `-1px`
                      on supersonic Double-Edge / Dragon Rush impact!)
      7. `crescent_wing_near` (giant foreground Blood-Soaked Crescent wing blade — pivoted at its dorsal root
                               socket for powerful banking and high-speed corkscrew spinning slashes!)
      8. `head_upper` (6 blue head/cheek fins, red forehead crest, fierce yellow eyes, and upper fangs —
                       100% rigid integer motion `shift_layer_int` so the face and eyes never deform by 1px!)
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
import mouth_open  # noqa: E402  (2026-11-04: small face-following mouth opening)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0373_salamence/mega/salamence-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0373_salamence/mega/salamence-mega"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16
VERSION = "salamence_mega_v1m"

# 2026-11-04 mouth fix: closed mouth line on the native 96px front.png, ordered hinge (back) -> lip tip (front).
# The jaw layer no longer drops; open_mouth carves a small lens below this line inside the face.
MOUTH_LINE_FRONT96 = [(32.0, 37.0), (27.0, 37.0), (24.0, 37.0), (21.0, 37.0)]
MOUTH_DEEP = (16, 16, 16)     # darkest palette colour (outline)
MOUTH_INNER = (96, 40, 40)    # existing dark mouth red


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
        | ((r == 0) & (g == 46) & (b == 63))
        | ((r == 96) & (g == 40) & (b == 40))
    )


def is_blue_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 35) & (g == 161) & (b == 211))
        | ((r == 30) & (g == 118) & (b == 150))
        | ((r == 37) & (g == 80) & (b == 99))
    )


def is_red_wing_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 163) & (g == 57) & (b == 57))
        | ((r == 234) & (g == 83) & (b == 80))
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
    # Keep only the primary connected component so nearest-neighbor wing/tail rotation never leaves a 1px stray speck
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
    Segments native 96x96 `front.png` into 8 exact anatomical masks following 1px outlines:
      1. `tail`: Whip-like blue & crimson dragon tail tip extending behind the left leg (`x=11..25, y=58..66`).
      2. `crescent_wing_far`: Far-side crimson crescent wing blade (`x=2..37, y=18..30`).
      3. `torso_shell`: Streamlined silver chest armor blades, lower silver keel, blue torso, dorsal wing collar,
                        and rear hip coil.
      4. `hind_leg_L`: Left blue leg and 3 white claws (`x=20..33, y=65..78`).
      5. `hind_leg_R`: Right legs and 3-clawed feet (`x=54..83, y=57..83`).
      6. `lower_jaw`: Fanged lower jaw and lower tongue (`x=20..32, y=38..45`).
      7. `crescent_wing_near`: Giant foreground crimson crescent wing blade (`x=42..92, y=12..56`).
      8. `head_upper`: 6 blue head/cheek fins, red forehead crest, yellow eyes, snout, and upper teeth (`x=10..45, y=15..45`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Lower jaw (open red lower jaw, white lower fangs, and lower pink tongue)
    lower_jaw = a & (
        ((yy == 38) & (xx >= 24) & (xx <= 30))
        | ((yy == 39) & (xx >= 23) & (xx <= 30))
        | ((yy == 40) & (xx >= 23) & (xx <= 31))
        | ((yy == 41) & (xx >= 22) & (xx <= 31))
        | ((yy == 42) & (xx >= 21) & (xx <= 31))
        | ((yy == 43) & (xx >= 20) & (xx <= 32))
        | ((yy == 44) & (xx >= 21) & (xx <= 29))
        | ((yy == 45) & (xx >= 22) & (xx <= 27))
    )

    # 2. Head upper (6 blue head/cheek fins, red forehead crest, yellow eyes, snout, and upper teeth)
    head_upper = (
        a
        & (
            # Far-left upper blue horn
            ((yy >= 15) & (yy <= 18) & (xx >= 13) & (xx <= 17))
            | ((yy == 19) & (xx >= 11) & (xx <= 18))
            | ((yy == 20) & (xx >= 13) & (xx <= 19))
            | ((yy == 21) & (xx >= 16) & (xx <= 20))
            | ((yy == 22) & (xx >= 17) & (xx <= 21))
            | ((yy == 23) & (xx >= 18) & (xx <= 22))
            | ((yy == 24) & (xx >= 18) & (xx <= 23))
            | ((yy == 25) & (xx >= 18) & (xx <= 24))
            | ((yy == 26) & (xx >= 18) & (xx <= 21))
            | ((yy == 27) & (xx == 18))
            # Near-right upper blue horn
            | ((yy == 15) & (xx >= 42) & (xx <= 43))
            | ((yy == 16) & (xx >= 41) & (xx <= 43))
            | ((yy == 17) & (xx >= 40) & (xx <= 43))
            | ((yy == 18) & (xx >= 39) & (xx <= 42))
            | ((yy >= 19) & (yy <= 20) & (xx >= 38) & (xx <= 42))
            | ((yy == 21) & (xx >= 37) & (xx <= 41))
            | ((yy == 22) & (xx >= 37) & (xx <= 40))
            | ((yy == 23) & (xx >= 36) & (xx <= 39))
            | ((yy == 24) & (xx >= 36) & (xx <= 40))
            | ((yy == 25) & (xx >= 33) & (xx <= 39))
            | ((yy == 26) & (xx >= 32) & (xx <= 38))
            | ((yy >= 27) & (yy <= 28) & (xx >= 31) & (xx <= 38))
            | ((yy == 29) & (xx >= 30) & (xx <= 38))
            | ((yy == 30) & (xx >= 29) & (xx <= 37))
            # Red forehead crest & brow
            | ((yy == 26) & (xx >= 22) & (xx <= 26))
            | ((yy == 27) & (xx >= 21) & (xx <= 25))
            | ((yy == 28) & (xx >= 20) & (xx <= 27))
            | ((yy == 29) & (xx >= 19) & (xx <= 26))
            | ((yy == 30) & (xx >= 10) & (xx <= 28))
            # Main snout, yellow eyes, middle fins, upper teeth
            | ((yy == 31) & (xx >= 10) & (xx <= 45))
            | ((yy == 32) & (xx >= 11) & (xx <= 45))
            | ((yy == 33) & (xx >= 11) & (xx <= 44))
            | ((yy == 34) & (xx >= 12) & (xx <= 41))
            | ((yy == 35) & (xx >= 12) & (xx <= 39))
            | ((yy == 36) & (xx >= 13) & (xx <= 39))
            | ((yy == 37) & (xx >= 16) & (xx <= 38))
            # Far-left 3rd lower cheek fin
            | ((yy == 38) & (xx >= 19) & (xx <= 23))
            | ((yy == 39) & (xx >= 18) & (xx <= 22))
            | ((yy == 40) & (xx >= 17) & (xx <= 22))
            | ((yy == 41) & (xx >= 16) & (xx <= 21))
            | ((yy == 42) & (xx >= 15) & (xx <= 20))
            | ((yy == 43) & (xx >= 14) & (xx <= 18))
            | ((yy == 44) & (xx >= 14) & (xx <= 16))
            # Near-right 3rd lower cheek fin
            | ((yy == 38) & (xx >= 32) & (xx <= 39))
            | ((yy == 39) & (xx >= 32) & (xx <= 39))
            | ((yy == 40) & (xx >= 33) & (xx <= 40))
            | ((yy == 41) & (xx >= 34) & (xx <= 41))
            | ((yy == 42) & (xx >= 36) & (xx <= 41))
            | ((yy == 43) & (xx >= 36) & (xx <= 42))
            | ((yy == 44) & (xx >= 37) & (xx <= 43))
            | ((yy == 45) & (xx >= 37) & (xx <= 38))
        )
        & (~lower_jaw)
    )

    # 3. Crescent wing far (far-side left crimson wing blade)
    crescent_wing_far = a & (yy >= 18) & (yy <= 30) & (xx <= 37) & (~head_upper)

    # 4. Crescent wing near (giant foreground right crimson crescent wing blade)
    crescent_wing_near = (
        a
        & (
            ((yy >= 12) & (yy <= 46) & (xx >= 42))
            | ((yy == 47) & (xx >= 55))
            | ((yy == 48) & (xx >= 49))
            | ((yy == 49) & (xx >= 51))
            | ((yy == 50) & (xx >= 53))
            | ((yy == 51) & (xx >= 56))
            | ((yy == 52) & (xx >= 59))
            | ((yy == 53) & (xx >= 61))
            | ((yy == 54) & (xx >= 63))
            | ((yy == 55) & (xx >= 65))
            | ((yy == 56) & (xx >= 67) & (xx <= 75))
        )
        & (~head_upper)
        & (~((yy >= 26) & (yy <= 46) & (xx <= 54)))
    )

    # 5. Tail (whip-like left tail tip sticking out behind the left leg at x=11..25, y=58..66)
    tail = a & (yy >= 58) & (yy <= 66) & (xx <= 25)

    # 6. Hind leg L (left blue leg and 3 white claws)
    hind_leg_L = (
        a
        & (
            ((yy == 65) & (xx >= 26) & (xx <= 28))
            | ((yy == 66) & (xx >= 26) & (xx <= 29))
            | ((yy == 67) & (xx >= 24) & (xx <= 30))
            | ((yy == 68) & (xx >= 24) & (xx <= 30))
            | ((yy == 69) & (xx >= 24) & (xx <= 31))
            | ((yy == 70) & (xx >= 24) & (xx <= 32))
            | ((yy == 71) & (xx >= 23) & (xx <= 32))
            | ((yy == 72) & (xx >= 23) & (xx <= 33))
            | ((yy >= 73) & (yy <= 78) & (xx >= 20) & (xx <= 31))
        )
        & (~tail)
    )

    # 7. Hind leg R (foreground right leg x=54..70, y=68..83 + rear right leg x=68..83, y=57..78)
    hind_leg_R = a & (
        ((yy >= 68) & (yy <= 70) & (xx >= 54) & (xx <= 66))
        | ((yy >= 71) & (yy <= 73) & (xx >= 54) & (xx <= 67))
        | ((yy >= 74) & (yy <= 83) & (xx >= 55) & (xx <= 70))
        | ((yy == 57) & (xx >= 68) & (xx <= 73))
        | ((yy >= 58) & (yy <= 60) & (xx >= 68) & (xx <= 75))
        | ((yy >= 61) & (yy <= 62) & (xx >= 68) & (xx <= 77))
        | ((yy >= 63) & (yy <= 64) & (xx >= 68) & (xx <= 78))
        | ((yy >= 65) & (yy <= 66) & (xx >= 68) & (xx <= 79))
        | ((yy >= 67) & (yy <= 78) & (xx >= 70) & (xx <= 83))
    )

    used = lower_jaw | head_upper | crescent_wing_far | crescent_wing_near | tail | hind_leg_L | hind_leg_R
    torso_shell = a & (~used)

    return {
        "tail": tail,
        "crescent_wing_far": crescent_wing_far,
        "torso_shell": torso_shell,
        "hind_leg_L": hind_leg_L,
        "hind_leg_R": hind_leg_R,
        "lower_jaw": lower_jaw,
        "crescent_wing_near": crescent_wing_near,
        "head_upper": head_upper,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 8 anatomical masks following 1px outlines.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Head upper on back (blue head, 2 upper horns, 3 left cheek fins, red forehead crest, and upper neck)
    head_upper = a & (
        ((yy == 10) & (((xx >= 56) & (xx <= 57)) | ((xx >= 72) & (xx <= 74))))
        | ((yy == 11) & (((xx >= 56) & (xx <= 58)) | ((xx >= 72) & (xx <= 74))))
        | ((yy == 12) & (((xx >= 56) & (xx <= 59)) | ((xx >= 71) & (xx <= 74))))
        | ((yy == 13) & (((xx >= 56) & (xx <= 60)) | ((xx >= 71) & (xx <= 74))))
        | ((yy == 14) & (((xx >= 57) & (xx <= 60)) | ((xx >= 71) & (xx <= 74))))
        | ((yy == 15) & (((xx >= 57) & (xx <= 59)) | ((xx >= 70) & (xx <= 73))))
        | ((yy == 16) & (((xx >= 57) & (xx <= 59)) | ((xx >= 70) & (xx <= 72))))
        | ((yy == 17) & (((xx >= 58) & (xx <= 60)) | ((xx >= 70) & (xx <= 72))))
        | ((yy == 18) & (((xx >= 58) & (xx <= 62)) | ((xx >= 69) & (xx <= 72))))
        | ((yy == 19) & (xx >= 58) & (xx <= 73))
        | ((yy >= 20) & (yy <= 22) & (xx >= 59) & (xx <= 74))
        | ((yy >= 23) & (yy <= 24) & (xx >= 52) & (xx <= 74))
        | ((yy == 25) & (xx >= 54) & (xx <= 75))
        | ((yy == 26) & (xx >= 57) & (xx <= 74))
        | ((yy == 27) & (xx >= 60) & (xx <= 73))
        | ((yy == 28) & (xx >= 58) & (xx <= 73))
        | ((yy == 29) & (xx >= 56) & (xx <= 72))
        | ((yy == 30) & (xx >= 55) & (xx <= 70))
        | ((yy == 31) & (xx >= 55) & (xx <= 69))
        | ((yy == 32) & (((xx >= 55) & (xx <= 57)) | ((xx >= 60) & (xx <= 69))))
        | ((yy == 33) & (xx >= 60) & (xx <= 68))
        | ((yy == 34) & (xx >= 59) & (xx <= 66))
        | ((yy == 35) & (xx >= 58) & (xx <= 63))
        | ((yy == 36) & (xx >= 57) & (xx <= 59))
    )

    # 2. Giant sweeping crimson crescent wing on back (y=7..65)
    wing_all = (
        a
        & (
            (yy <= 58)
            | ((yy == 59) & (xx >= 14))
            | ((yy == 60) & (xx >= 16) & (xx <= 65))
            | ((yy == 61) & (xx >= 17) & (xx <= 63))
            | ((yy == 62) & (xx >= 20) & (xx <= 60))
            | ((yy == 63) & (xx >= 24) & (xx <= 55))
            | ((yy == 64) & (xx >= 28) & (xx <= 49))
            | ((yy == 65) & (xx >= 34) & (xx <= 49))
        )
        & (~head_upper)
    )
    crescent_wing_far = np.zeros_like(a)
    crescent_wing_near = wing_all

    # 3. Hind leg L on back (far-left leg x=1..12, y=63..77)
    hind_leg_L = a & (
        ((yy >= 63) & (yy <= 67) & (xx <= 12))
        | ((yy >= 68) & (yy <= 74) & (xx <= 11))
        | ((yy >= 75) & (yy <= 77) & (xx <= 12))
    )

    # 4. Hind leg R on back (middle-right leg x=37..53, y=71..81 + far-right leg x=58..73, y=67..74)
    hind_leg_R = a & (
        ((yy == 71) & (xx >= 37) & (xx <= 52))
        | ((yy >= 72) & (yy <= 73) & (xx >= 38) & (xx <= 51))
        | ((yy >= 74) & (yy <= 75) & (xx >= 39) & (xx <= 51))
        | ((yy >= 76) & (yy <= 77) & (xx >= 40) & (xx <= 51))
        | ((yy == 78) & (xx >= 40) & (xx <= 52))
        | ((yy == 79) & (xx >= 41) & (xx <= 53))
        | ((yy == 80) & (xx >= 44) & (xx <= 53))
        | ((yy == 81) & (xx >= 44) & (xx <= 52))
        | ((yy >= 67) & (yy <= 74) & (xx >= 58) & (xx <= 73))
    )

    # 5. Tail on back (sweeping U-shaped dragon tail at the bottom y=70..87, x=13..65)
    tail = (
        a
        & (
            ((yy == 70) & (xx >= 13) & (xx <= 25))
            | ((yy >= 71) & (yy <= 73) & (xx >= 13) & (xx <= 24))
            | ((yy == 74) & (xx >= 13) & (xx <= 25))
            | ((yy == 75) & (((xx >= 13) & (xx <= 26)) | ((xx >= 62) & (xx <= 65))))
            | ((yy == 76) & (((xx >= 14) & (xx <= 29)) | ((xx >= 61) & (xx <= 65))))
            | ((yy == 77) & (((xx >= 14) & (xx <= 32)) | ((xx >= 60) & (xx <= 64))))
            | ((yy == 78) & (((xx >= 14) & (xx <= 35)) | ((xx >= 57) & (xx <= 63))))
            | ((yy == 79) & (((xx >= 15) & (xx <= 38)) | ((xx >= 55) & (xx <= 62))))
            | ((yy == 80) & (((xx >= 16) & (xx <= 43)) | ((xx >= 54) & (xx <= 61))))
            | ((yy >= 81) & (yy <= 87))
        )
        & (~hind_leg_R)
    )

    lower_jaw = np.zeros_like(a)
    used = head_upper | crescent_wing_far | crescent_wing_near | hind_leg_L | hind_leg_R | tail | lower_jaw
    torso_shell = a & (~used)

    return {
        "tail": tail,
        "crescent_wing_far": crescent_wing_far,
        "torso_shell": torso_shell,
        "hind_leg_L": hind_leg_L,
        "hind_leg_R": hind_leg_R,
        "lower_jaw": lower_jaw,
        "crescent_wing_near": crescent_wing_near,
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

    blue_fill_m = is_blue_body_fill(canvas)
    red_fill_m = is_red_wing_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order (back-to-front):
        # 1. tail -> 2. crescent_wing_far -> 3. torso_shell -> 4. hind_leg_L -> 5. hind_leg_R
        # -> 6. lower_jaw -> 7. crescent_wing_near -> 8. head_upper
        fg_over_tail = c_masks["torso_shell"] | c_masks["hind_leg_L"]
        u_tail = ndi.binary_erosion(c_masks["tail"] | fg_over_tail, iterations=1)
        layers["tail"] = build_part_layer(
            canvas,
            c_masks["tail"],
            u_tail & fg_over_tail,
            cushion_px=6,
            allowed_fill_mask=blue_fill_m,
        )

        # Cushion `crescent_wing_far` only behind the lower head/forehead (`y >= dst_y0 + 11`) so no red
        # pixels ever protrude past the upper blue horns when the wing banks!
        yy_c, _ = np.mgrid[0:cell_h, 0:cell_w]
        fg_over_wfar = (c_masks["torso_shell"] | c_masks["head_upper"]) & (yy_c >= dst_y0 + 11)
        u_wfar = ndi.binary_erosion(c_masks["crescent_wing_far"] | fg_over_wfar, iterations=1)
        layers["crescent_wing_far"] = build_part_layer(
            canvas,
            c_masks["crescent_wing_far"],
            u_wfar & fg_over_wfar,
            cushion_px=5,
            allowed_fill_mask=red_fill_m,
        )

        # IMPORTANT: `torso_shell` cushion ONLY expands under foreground layers drawn ON TOP OF `torso_shell`
        # (`hind_leg_L`, `hind_leg_R`, `lower_jaw`, `crescent_wing_near`, `head_upper`),
        # NEVER over background layers (`tail`, `crescent_wing_far`)!
        fg_over_torso = (
            c_masks["hind_leg_L"]
            | c_masks["hind_leg_R"]
            | c_masks["lower_jaw"]
            | c_masks["crescent_wing_near"]
            | c_masks["head_upper"]
        )
        u_torso = ndi.binary_erosion(c_masks["torso_shell"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso_shell"],
            u_torso & fg_over_torso,
            cushion_px=7,
            allowed_fill_mask=blue_fill_m,
        )

        # 2026-11-04: the rectangular pink/crimson mouth pocket behind `lower_jaw` was removed (it read as a
        # rectangle / jutting chin). The mouth now opens with mouth_open.open_mouth on the composed frame.
        layers["torso_shell"] = torso_lyr

        layers["hind_leg_L"] = build_part_layer(
            canvas, c_masks["hind_leg_L"], np.zeros_like(c_masks["hind_leg_L"]), cushion_px=0
        )
        layers["hind_leg_R"] = build_part_layer(
            canvas, c_masks["hind_leg_R"], np.zeros_like(c_masks["hind_leg_R"]), cushion_px=0
        )

        jaw_lyr = build_part_layer(
            canvas, c_masks["lower_jaw"], np.zeros_like(c_masks["lower_jaw"]), cushion_px=0
        )
        layers["lower_jaw"] = jaw_lyr

        # `crescent_wing_near` already has its complete 1px outline; zero cushion onto `head_upper` prevents
        # any red pixels from bleeding onto the right blue horn tip!
        layers["crescent_wing_near"] = build_part_layer(
            canvas,
            c_masks["crescent_wing_near"],
            np.zeros_like(c_masks["crescent_wing_near"]),
            cushion_px=0,
        )

        layers["head_upper"] = build_part_layer(
            canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0
        )

        pivots = {
            "tail": to_canvas_pt(25.0, 63.0),
            "crescent_wing_far": to_canvas_pt(25.0, 26.0),
            "crescent_wing_near": to_canvas_pt(55.0, 42.0),
        }
        mouth_line = [to_canvas_pt(x, y) for x, y in MOUTH_LINE_FRONT96]

    else:
        # Back view draw order (back-to-front):
        # 1. tail -> 2. crescent_wing_far -> 3. torso_shell -> 4. hind_leg_L -> 5. hind_leg_R
        # -> 6. lower_jaw -> 7. head_upper -> 8. crescent_wing_near
        fg_over_tail = c_masks["torso_shell"] | c_masks["hind_leg_L"] | c_masks["hind_leg_R"]
        u_tail = ndi.binary_erosion(c_masks["tail"] | fg_over_tail, iterations=1)
        layers["tail"] = build_part_layer(
            canvas,
            c_masks["tail"],
            u_tail & fg_over_tail,
            cushion_px=6,
            allowed_fill_mask=blue_fill_m,
        )

        layers["crescent_wing_far"] = np.zeros_like(canvas)

        # IMPORTANT: `torso_shell` cushion ONLY expands under foreground layers drawn ON TOP OF `torso_shell`
        # (`hind_leg_L`, `hind_leg_R`, `head_upper`, `crescent_wing_near`), NEVER over `tail`!
        fg_over_torso = (
            c_masks["hind_leg_L"]
            | c_masks["hind_leg_R"]
            | c_masks["head_upper"]
            | c_masks["crescent_wing_near"]
        )
        u_torso = ndi.binary_erosion(c_masks["torso_shell"] | fg_over_torso, iterations=1)
        layers["torso_shell"] = build_part_layer(
            canvas,
            c_masks["torso_shell"],
            u_torso & fg_over_torso,
            cushion_px=7,
            allowed_fill_mask=blue_fill_m,
        )

        layers["hind_leg_L"] = build_part_layer(
            canvas, c_masks["hind_leg_L"], np.zeros_like(c_masks["hind_leg_L"]), cushion_px=0
        )
        layers["hind_leg_R"] = build_part_layer(
            canvas, c_masks["hind_leg_R"], np.zeros_like(c_masks["hind_leg_R"]), cushion_px=0
        )
        layers["lower_jaw"] = np.zeros_like(canvas)

        # Cushion `head_upper`'s lower neck (`y >= dst_y0 + 14`) with blue fill under `crescent_wing_near`
        # so when the giant crescent wing banks across the back of the neck, only blue neck pixels are revealed!
        yy_c, _ = np.mgrid[0:cell_h, 0:cell_w]
        fg_over_head = c_masks["crescent_wing_near"] & (yy_c >= dst_y0 + 14)
        u_head = ndi.binary_erosion(c_masks["head_upper"] | fg_over_head, iterations=1)
        layers["head_upper"] = build_part_layer(
            canvas,
            c_masks["head_upper"],
            u_head & fg_over_head,
            cushion_px=6,
            allowed_fill_mask=blue_fill_m,
        )

        layers["crescent_wing_near"] = build_part_layer(
            canvas,
            c_masks["crescent_wing_near"],
            np.zeros_like(c_masks["crescent_wing_near"]),
            cushion_px=0,
        )

        pivots = {
            "tail": to_canvas_pt(22.0, 72.0),
            "crescent_wing_far": to_canvas_pt(48.0, 48.0),
            "crescent_wing_near": to_canvas_pt(48.0, 48.0),
        }
        mouth_line = []  # back views: mouth not visible, no opening

    pivots["mouth_line"] = mouth_line
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
        (mid_x - 19 - drift, max(2, top_y + 13 - drift // 2)),
        (mid_x + 19 + drift, max(2, top_y + 15 - drift // 2)),
    ]
    # Uses Mega Salamence's native eye-yellow palette color (241, 249, 79) so off_palette_colors == 0
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (241, 249, 79, 255)
    return out


def synthesize_salamence_mega_frame(
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
    wfar_dx, wfar_dy = 0, 0
    wnear_dx, wnear_dy = 0, 0
    leg_L_dx, leg_L_dy = 0, 0
    leg_R_dx, leg_R_dy = 0, 0
    tail_dx, tail_dy = 0, 0

    wfar_ang = 0.0
    wnear_ang = 0.0
    tail_ang = 0.0

    if action == "idle":
        # Predatory Supersonic Gliding Hover + Blood-Soaked Crescent Wing Bank + Dragon Roar:
        # 1. Streamlined dragon body glides smoothly in an updraft figure-8 (`±1px` x, `-2..+1px` y).
        # 2. Giant crimson crescent wings (`crescent_wing_near`, `crescent_wing_far`) bank and roll (`±4.2°`)
        #    around their dorsal sockets while the dragon `tail` whips (`±3.8°`) in the slipstream.
        # 3. `lower_jaw` opens `+1..+2px` during the primary dragon roar (`frames 2..7`, peak `+2px` at `3..5`)
        #    and barks (`+1px` at `frames 11..12`), exposing the crimson/pink mouth cavity!
        body_dy = int(round(1.5 * math.sin(phase - 0.2)))
        body_dx = int(round(1.0 * math.cos(phase - 0.2) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        if frame_idx in (3, 4, 5):
            jaw_open = 1
        elif frame_idx in (2, 6, 7, 11, 12):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # 2026-11-04: jaw stays put (no chin drop); mouth opens via mouth_open

        wnear_dx = body_dx
        wnear_dy = body_dy + int(round(0.8 * math.sin(phase - 0.5)))
        wnear_ang = 3.2 * math.sin(phase - 0.5) * (-facing)

        wfar_dx = body_dx
        wfar_dy = body_dy + int(round(0.8 * math.sin(phase - 0.8)))
        wfar_ang = 3.2 * math.sin(phase - 0.8) * facing

        leg_L_dx = body_dx
        leg_L_dy = body_dy + int(round(0.6 * math.sin(phase - 0.6)))
        leg_R_dx = body_dx
        leg_R_dy = body_dy + int(round(0.6 * math.sin(phase - 0.9)))

        tail_dx = body_dx
        tail_dy = body_dy
        tail_ang = (2.6 if is_back else 3.8) * math.sin(phase - 1.0) * facing

    elif action == "walk":
        # Supersonic Jet-Stream Glide + Dynamic Crescent Wing Roll Sweep + Ferocious Barking Jaw:
        # 2-cycle jet-stream propulsion (`2.0 * phase`), forward-leaning aerial surge (`surge_x`, `surge_y`),
        # counter-banking crimson crescent wings (`±4.8°`), high-cadence rudder tail whip, trailing hind legs,
        # and rhythmic dragon roar/bark (`+1..+2px`) on each forward beat!
        surge_x = int(round(2.0 * sin1 * facing))
        surge_y = int(round(1.5 * math.sin(2.0 * phase - 0.3)))

        torso_dx, torso_dy = surge_x, surge_y
        head_dx = surge_x + int(round(0.8 * cos1 * facing))
        head_dy = surge_y

        s2 = math.sin(2.0 * phase)
        if s2 > 0.65:
            jaw_open = 1
        elif s2 > 0.15:
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # 2026-11-04: jaw stays put (no chin drop); mouth opens via mouth_open

        wnear_dx = surge_x
        wnear_dy = surge_y + int(round(1.0 * math.sin(2.0 * phase - 0.4)))
        wnear_ang = 4.6 * math.sin(2.0 * phase - 0.4) * (-facing)

        wfar_dx = surge_x
        wfar_dy = surge_y + int(round(-1.0 * math.sin(2.0 * phase - 0.4)))
        wfar_ang = -4.2 * math.sin(2.0 * phase - 0.4) * facing

        leg_L_dx = surge_x + int(round(1.0 * cos1 * facing))
        leg_L_dy = surge_y + int(round(-0.8 * max(0.0, sin1)))
        leg_R_dx = surge_x + int(round(-1.0 * cos1 * facing))
        leg_R_dy = surge_y + int(round(-0.8 * max(0.0, -sin1)))

        tail_dx = surge_x
        tail_dy = surge_y
        tail_ang = (3.0 if is_back else 4.8) * math.sin(2.0 * phase - 0.8) * facing

    elif action == "attack":
        # Signature Mega Salamence "Thunderous Dragon Roar + Corkscrew Spinning Double-Edge / Dragon Rush"!
        # Frames 0..3 (Wind-up & Wide Dragon Roar):
        #   - Coils back (`-4px`), tilts giant crimson crescent wings steep back (`-6.5°`),
        #   - Opens `lower_jaw` WIDE (`+2px` ferocious dragon roar!).
        # Frames 4..10 (Supersonic Corkscrew Barrel-Roll Lunge & Jaw Crunch!):
        #   - Explosive supersonic forward lunge (`+7px` body & head),
        #   - High-speed corkscrew spinning bank across both crimson crescent wings (`±6.8°`) and tail (`±6.5°`),
        #   - Fanged jaws clamp shut (`-1px` at frames 6..7) on impact, then unleash a follow-through roar (`+2px` at frames 9..10)!
        # Frames 11..15: Aerodynamic recovery to hover stance.
        lunge_x_table = [0, -2, -3, -4, 3, 6, 7, 7, 5, 3, 2, 0, -1, -1, 0, 0]
        lunge_y_table = [0, -1, -1, -2, 0, 1, 2, 1, 0, -1, 0, 1, 0, 0, 0, 0]

        body_dx = int(round(lunge_x_table[frame_idx] * facing))
        body_dy = lunge_y_table[frame_idx]

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + (int(round(1.0 * facing)) if frame_idx in (5, 6, 7) else 0)
        head_dy = body_dy

        # Jaw curve: wide +2px roar on wind-up (1..4), snap shut -1px on impact (6..7), follow-through roar +2px (9..10)
        jaw_open_table = [0, 1, 2, 2, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0]  # mouth_open amount (2 only at roar peaks)
        jaw_open = jaw_open_table[frame_idx]
        jaw_dx = head_dx
        jaw_dy = head_dy  # 2026-11-04: jaw stays put (no chin drop); mouth opens via mouth_open

        # Corkscrew spinning wing & tail angles during strike frames (4..10)
        roll_wnear = [0.0, -3.5, -5.0, -6.5, 6.5, -7.0, 7.0, -6.5, 5.0, -3.8, 2.2, -1.0, 0.0, 0.0, 0.0, 0.0]
        roll_wfar = [0.0, 3.0, 4.5, 5.5, -5.5, 6.0, -6.0, 5.5, -4.5, 3.2, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        roll_tail = [0.0, -3.0, -4.5, -6.0, 6.0, -6.5, 6.5, -6.0, 4.5, -3.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        corkscrew_dy = [0, 0, -1, -1, 1, -2, 2, -2, 1, -1, 0, 0, 0, 0, 0, 0]

        wnear_dx = body_dx + (int(round(1.0 * facing)) if frame_idx in (5, 6, 7) else 0)
        wnear_dy = body_dy + corkscrew_dy[frame_idx]
        wnear_ang = roll_wnear[frame_idx] * (-facing)

        wfar_dx = body_dx
        wfar_dy = body_dy - (corkscrew_dy[frame_idx] // 2)
        wfar_ang = roll_wfar[frame_idx] * facing

        leg_L_dx = body_dx + (int(round(-1.0 * facing)) if frame_idx in (5, 6, 7) else 0)
        leg_L_dy = body_dy + (corkscrew_dy[frame_idx] // 2)
        leg_R_dx = body_dx
        leg_R_dy = body_dy - (corkscrew_dy[frame_idx] // 2)

        tail_dx = body_dx
        tail_dy = body_dy
        tail_ang = (0.55 if is_back else 1.0) * roll_tail[frame_idx] * facing

    elif action == "hurt":
        # Mid-air shock knockback + hit-shake + pained dragon roar (`lower_jaw` open `+2px`) + wings flung back + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-4.0 * env * facing)) + jit
        body_dy = int(round(-1.2 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-1.5 * env))

        if frame_idx in (2, 3, 4, 5):
            jaw_open = 1
        elif frame_idx in (1, 6, 7, 8):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # 2026-11-04: jaw stays put (no chin drop); mouth opens via mouth_open

        wnear_dx = body_dx + int(round(-1.0 * env * facing))
        wnear_dy = body_dy + int(round(-1.2 * env))
        wnear_ang = -5.5 * env * (-facing)

        wfar_dx = body_dx + int(round(-1.0 * env * facing))
        wfar_dy = body_dy + int(round(-1.2 * env))
        wfar_ang = 5.0 * env * facing

        leg_L_dx, leg_L_dy = body_dx, body_dy
        leg_R_dx, leg_R_dy = body_dx, body_dy

        tail_dx, tail_dy = body_dx, body_dy
        tail_ang = (-3.0 if is_back else -5.0) * env * facing

    tail_rot = rotate_layer_nearest(layers["tail"], tail_ang, pivots["tail"])
    wfar_rot = rotate_layer_nearest(layers["crescent_wing_far"], wfar_ang, pivots["crescent_wing_far"])
    wnear_rot = rotate_layer_nearest(layers["crescent_wing_near"], wnear_ang, pivots["crescent_wing_near"])

    if not is_back:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(wfar_rot, wfar_dx, wfar_dy),
            shift_layer_int(layers["torso_shell"], torso_dx, torso_dy),
            shift_layer_int(layers["hind_leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["hind_leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(wnear_rot, wnear_dx, wnear_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
        ]
    else:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(wfar_rot, wfar_dx, wfar_dy),
            shift_layer_int(layers["torso_shell"], torso_dx, torso_dy),
            shift_layer_int(layers["hind_leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["hind_leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(wnear_rot, wnear_dx, wnear_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    mouth_line = pivots.get("mouth_line") or []
    if (not is_back) and mouth_line and jaw_open > 0:
        line = mouth_open.transform_points(mouth_line, dx=head_dx, dy=head_dy)
        out = mouth_open.open_mouth(out, line, amount=jaw_open, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=0.55)
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

    # Tight bounds of front (91x72) and back (94x81)
    max_w, max_h = 94, 81
    cell_w = 124
    cell_h = 104
    foot_y = 94

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
                frame_rgba = synthesize_salamence_mega_frame(
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

    # Preserve existing official 103-frame `front.gif` / `front-reverse.gif`,
    # and cleanly replace the warped 20-frame `back.gif` / `back-reverse.gif` with 16-frame `idle` N loop!
    idle_sheet = sheets_by_action["idle"]
    back_frames = [idle_sheet[4 * cell_h:5 * cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    back_rev_frames = [f[:, ::-1, :].copy() for f in back_frames]

    save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
    save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)

    meta = {
        "generator": "forge_salamence_mega_actions",
        "key": "0373_salamence-mega",
        "dex": 373,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "foot_y": foot_y,
        "sprite_h": max_h,
        "frames": N_FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "palette_colors": int(len(palette_rgba)),
        "off_palette_colors": 0,
        "v": VERSION,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Sync to pokemon_arena_2d if present
    if SYNC_MEGA_DIR.exists():
        SYNC_OUT_DIR.mkdir(parents=True, exist_ok=True)
        for f_path in OUT_DIR.iterdir():
            if f_path.is_file():
                shutil.copy2(f_path, SYNC_OUT_DIR / f_path.name)
        for gif_name in ("back.gif", "back-reverse.gif"):
            shutil.copy2(MEGA_DIR / gif_name, SYNC_MEGA_DIR / gif_name)

    print(f"Built 0373_salamence-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
