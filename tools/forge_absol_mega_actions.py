#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Absol (`0359_absol-mega`) modeled after Normal Absol's official GIF
(`0359_absol/normal/front.gif`) and Mega Absol's winged disaster-beast anatomy (`front.png` & `back.png`):
  - Exact 10-layer anatomical rigging with true joint pivots (shoulder wing sockets & tail root socket):
      1. `tail` (serrated navy-blue scimitar tail — 100% separated from both hind legs, pivoted at the tail root
                 socket so it whips and slices smoothly without detaching!)
      2. `wing_far` (far-side radiant white angelic wing-fur — pivoted at the far shoulder socket for majestic
                     flapping and spinning slash deployment)
      3. `hind_leg_L` (far-side shaded hind leg — 100% grounded `(0,0)` in `idle`, diagonal gallop in `walk`)
      4. `front_leg_L` (far-side shaded foreleg — 100% grounded `(0,0)` in `idle`, diagonal gallop in `walk`)
      5. `hind_leg_R` (near-side hind leg with 3 heel spikes & navy claws — 100% grounded `(0,0)` in `idle`)
      6. `torso` (white chest mane, silver-white body, and crimson/dark-red mouth interior pocket behind the jaw)
      7. `front_leg_R` (near-side foreleg with long white elbow fur & 3 navy claws — 100% grounded `(0,0)` in `idle`,
                        powerful leaping slash step in `attack`)
      8. `lower_jaw` (navy-blue lower jaw — opens `+1..+2px` for ferocious disaster-beast roar/bark in `idle`/`walk`
                      and opens wide `+2px` during `attack` wind-up/slash roar and `hurt` reaction!)
      9. `head_upper` (crescent scythe horn, head crest, forehead emblem, glowing red eye, and long white bang —
                       100% rigid integer motion `shift_layer_int` so the red eye and scythe horn never deform!)
     10. `wing_near` (foreground radiant white angelic wing — pivoted at the near shoulder socket for sweeping
                      flaps in `idle`/`walk` and high-speed spinning Night Slash / Psycho Cut rotation in `attack`!)
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
import mouth_open  # noqa: E402  (2026-11-04: small mouth opening along the face instead of a jaw drop)

# 2026-11-04 (user request): the mouth no longer opens by dropping the lower-jaw layer over a rectangular
# crimson pocket (read as a rectangle + jutting chin). The jaw stays put and `mouth_open.open_mouth` carves a
# small lens along the closed mouth line of the native front still (hinge -> lip tip, native 96px coords).
MOUTH_LINE_FRONT_NATIVE = [(35.0, 41.0), (32.0, 41.0), (29.0, 41.0)]
MOUTH_DEEP = (0, 0, 0)          # darkest palette colour (outline)
MOUTH_INNER = (123, 41, 49)     # native dark crimson (palette)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0359_absol/mega/absol-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0359_absol/mega/absol-mega"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16


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
        | ((r == 41) & (g == 57) & (b == 57))
        | ((r == 65) & (g == 74) & (b == 106))
    )


def is_white_fur_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 180) & (g == 180) & (b == 213))
        | ((r == 213) & (g == 222) & (b == 238))
        | ((r == 253) & (g == 253) & (b == 253))
        | ((r == 255) & (g == 255) & (b == 255))
        | ((r == 139) & (g == 139) & (b == 172))
    )


def is_navy_skin_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 65) & (g == 74) & (b == 106))
        | ((r == 82) & (g == 90) & (b == 123))
        | ((r == 115) & (g == 123) & (b == 172))
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


def build_leg_stem_layer(
    canvas: np.ndarray,
    leg_mask: np.ndarray,
    upper_mask: np.ndarray,
    stem_px: int = 4,
) -> np.ndarray:
    lyr = np.zeros_like(canvas)
    lyr[leg_mask] = canvas[leg_mask]
    f_ys, f_xs = np.where(leg_mask)
    if len(f_ys) == 0:
        return lyr
    for x_col in np.unique(f_xs):
        col_ys = f_ys[f_xs == x_col]
        cy_min = int(col_ys.min())
        fill_px = canvas[cy_min, x_col]
        for dy_up in range(1, stem_px + 1):
            py = cy_min - dy_up
            if py >= 0 and upper_mask[py, x_col]:
                if not is_dark_outline(canvas[py:py + 1, x_col:x_col + 1])[0, 0]:
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
    Segments native 96x96 `front.png` into 10 exact anatomical masks following 1px outlines:
      1. `tail`: Serrated navy-blue scimitar tail (`x=60..81, y=42..52`) — 100% separated from `hind_leg_R`.
      2. `wing_far`: Far-side radiant white angelic wing-fur (`x=14..25, y=16..45`).
      3. `hind_leg_L`: Far-side shaded hind leg under belly (`x=47..57, y=58..74`).
      4. `front_leg_L`: Far-side shaded foreleg under chest (`x=34..42, y=60..69`).
      5. `hind_leg_R`: Near-side hind leg with 3 heel spikes and navy claws (`x=55..72, y=57..76`).
      6. `torso`: White chest mane, silver-white back/belly/pelvis.
      7. `front_leg_R`: Near-side foreleg with long white elbow fur and 3 navy claws (`x=32..52, y=58..82`).
      8. `lower_jaw`: Navy-blue lower jaw and chin outline (`x=28..36, y=41..44`).
      9. `head_upper`: Crescent scythe horn, head crest, forehead diamond, glowing red eye, upper snout,
                       and long white left bang (`x=19..48, y=13..55`).
     10. `wing_near`: Foreground right radiant white angelic wing (`x=42..73, y=18..49`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Tail (serrated navy scimitar tail extending right from the hip)
    tail = a & (
        ((yy == 42) & (xx >= 70))
        | ((yy == 43) & (xx >= 68))
        | ((yy == 44) & (xx >= 67))
        | ((yy == 45) & (xx >= 66))
        | ((yy == 46) & (xx >= 65))
        | ((yy == 47) & (xx >= 63))
        | ((yy == 48) & (xx >= 60))
        | ((yy == 49) & (xx >= 61))
        | ((yy == 50) & (xx >= 61))
        | ((yy == 51) & (xx >= 64))
        | ((yy == 52) & (xx >= 65))
    )

    # 2. Wing far (left-side white wing feathers behind the crescent horn & head)
    wing_far = a & (
        ((yy >= 16) & (yy <= 21) & (xx <= 22))
        | ((yy == 22) & (xx <= 21))
        | ((yy == 23) & (xx <= 20))
        | ((yy >= 24) & (yy <= 25) & (xx <= 19))
        | ((yy >= 26) & (yy <= 31) & (xx <= 18))
        | ((yy >= 32) & (yy <= 34) & (xx <= 19))
        | ((yy >= 35) & (yy <= 36) & (xx <= 20))
        | ((yy >= 37) & (yy <= 39) & (xx <= 24))
        | ((yy >= 40) & (yy <= 43) & (xx <= 25))
        | ((yy >= 44) & (yy <= 45) & (xx <= 24))
    )

    # 3. Wing near (large right-side radiant white angelic wing)
    wing_near = a & (
        ((yy >= 18) & (yy <= 30) & (xx >= 50))
        | ((yy == 31) & (xx >= 50))
        | ((yy == 32) & (xx >= 49))
        | ((yy >= 33) & (yy <= 34) & (xx >= 48))
        | ((yy >= 35) & (yy <= 38) & (xx >= 47))
        | ((yy == 39) & (xx >= 46))
        | ((yy >= 40) & (yy <= 41) & (xx >= 45))
        | ((yy == 42) & (xx >= 45) & (xx <= 63))
        | ((yy == 43) & (xx >= 44) & (xx <= 58))
        | ((yy == 44) & (xx >= 44) & (xx <= 55))
        | ((yy == 45) & (xx >= 43) & (xx <= 55))
        | ((yy == 46) & (xx >= 43) & (xx <= 52))
        | ((yy == 47) & (xx >= 43) & (xx <= 49))
        | ((yy >= 48) & (yy <= 49) & (xx >= 42) & (xx <= 48))
    )

    # 4. Lower jaw (navy-blue lower jaw and chin outline strictly along and below the mouth line y=42..44)
    lower_jaw = a & (
        ((yy == 42) & (xx >= 29) & (xx <= 35))
        | ((yy == 43) & (xx >= 29) & (xx <= 36))
        | ((yy == 44) & (xx >= 31) & (xx <= 35))
    )

    # 5. Head upper (crescent scythe horn, head crest, forehead emblem, glowing red eye, upper snout, and long white bang)
    head_upper = (
        a
        & (
            ((yy >= 13) & (yy <= 36) & (xx <= 49))
            | ((yy >= 37) & (yy <= 41) & (xx >= 25) & (xx <= 46))
            | ((yy == 42) & (((xx >= 26) & (xx <= 28)) | ((xx >= 36) & (xx <= 45))))
            | ((yy == 43) & (((xx >= 26) & (xx <= 28)) | ((xx >= 37) & (xx <= 44))))
            | ((yy == 44) & (((xx >= 25) & (xx <= 30)) | ((xx >= 36) & (xx <= 44))))
            | ((yy == 45) & (xx >= 37) & (xx <= 43))
            | ((yy >= 46) & (yy <= 47) & (xx >= 36) & (xx <= 42))
            | ((yy >= 48) & (yy <= 49) & (xx >= 35) & (xx <= 41))
            | ((yy >= 50) & (yy <= 51) & (xx >= 35) & (xx <= 40))
            | ((yy >= 52) & (yy <= 53) & (xx >= 36) & (xx <= 39))
            | ((yy == 54) & (xx >= 37) & (xx <= 39))
            | ((yy == 55) & (xx >= 38) & (xx <= 39))
        )
        & (~wing_far)
        & (~wing_near)
        & (~lower_jaw)
    )

    # 6. Front leg R (foreground left foreleg with long white elbow fur & 3 navy claws)
    front_leg_R = a & (
        ((yy >= 58) & (yy <= 61) & (xx >= 43) & (xx <= 48))
        | ((yy >= 62) & (yy <= 64) & (xx >= 42) & (xx <= 47))
        | ((yy == 65) & (xx >= 41) & (xx <= 47))
        | ((yy == 66) & (xx >= 41) & (xx <= 52))
        | ((yy == 67) & (xx >= 41) & (xx <= 51))
        | ((yy == 68) & (xx >= 40) & (xx <= 50))
        | ((yy == 69) & (xx >= 39) & (xx <= 49))
        | ((yy == 70) & (xx >= 38) & (xx <= 50))
        | ((yy == 71) & (xx >= 37) & (xx <= 51))
        | ((yy >= 72) & (yy <= 73) & (xx >= 36) & (xx <= 48))
        | ((yy >= 74) & (yy <= 82) & (xx >= 32) & (xx <= 45))
    )

    # 7. Front leg L (far-side shaded foreleg under the chest mane)
    front_leg_L = (
        a
        & (
            ((yy == 60) & (xx >= 40) & (xx <= 42))
            | ((yy == 61) & (xx >= 39) & (xx <= 41))
            | ((yy == 62) & (xx >= 38) & (xx <= 40))
            | ((yy == 63) & ((xx == 34) | ((xx >= 38) & (xx <= 39))))
            | ((yy == 64) & (xx >= 34) & (xx <= 40))
            | ((yy >= 65) & (yy <= 67) & (xx >= 35) & (xx <= 40))
            | ((yy == 68) & (xx >= 37) & (xx <= 39))
            | ((yy == 69) & (xx >= 37) & (xx <= 38))
        )
        & (~front_leg_R)
    )

    # 8. Hind leg L (far-side shaded hind leg under belly)
    hind_leg_L = (
        a
        & (
            ((yy >= 58) & (yy <= 59) & (xx >= 49) & (xx <= 54))
            | ((yy == 60) & (xx >= 48) & (xx <= 55))
            | ((yy >= 61) & (yy <= 62) & (xx >= 48) & (xx <= 56))
            | ((yy == 63) & (xx >= 48) & (xx <= 56))
            | ((yy == 64) & (xx >= 48) & (xx <= 57))
            | ((yy == 65) & (xx >= 48) & (xx <= 57))
            | ((yy == 66) & (xx >= 53) & (xx <= 56))
            | ((yy == 67) & (xx >= 52) & (xx <= 55))
            | ((yy >= 68) & (yy <= 69) & (xx >= 50) & (xx <= 56))
            | ((yy == 70) & (xx >= 51) & (xx <= 55))
            | ((yy == 71) & (xx >= 52) & (xx <= 54))
            | ((yy >= 72) & (yy <= 73) & (xx >= 49) & (xx <= 53))
            | ((yy == 74) & (xx >= 47) & (xx <= 52))
        )
        & (~front_leg_R)
    )

    # 9. Hind leg R (near-side hind leg with 3 heel fur spikes & 3 navy claws)
    hind_leg_R = (
        a
        & (
            ((yy == 57) & (xx == 64))
            | ((yy >= 58) & (yy <= 60) & (xx >= 55) & (xx <= 65))
            | ((yy >= 61) & (yy <= 65) & (xx >= 57) & (xx <= 72))
            | ((yy >= 66) & (yy <= 76) & (xx >= 58) & (xx <= 72))
        )
        & (~hind_leg_L)
    )

    used = (
        tail
        | wing_far
        | wing_near
        | lower_jaw
        | head_upper
        | front_leg_R
        | front_leg_L
        | hind_leg_L
        | hind_leg_R
    )
    torso = a & (~used)

    return {
        "tail": tail,
        "wing_far": wing_far,
        "hind_leg_L": hind_leg_L,
        "front_leg_L": front_leg_L,
        "hind_leg_R": hind_leg_R,
        "torso": torso,
        "front_leg_R": front_leg_R,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
        "wing_near": wing_near,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 10 anatomical masks following 1px outlines.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    navy_m = is_navy_skin_fill(arr96) | (
        a & (arr96[..., 0] == 41) & (arr96[..., 1] == 57) & (arr96[..., 2] == 57)
    )

    # 1. Tail on back (`y=39..49, x=7..31`)
    tail = a & (
        ((yy >= 39) & (yy <= 42) & (xx <= 25))
        | ((yy == 43) & (xx <= 27))
        | ((yy == 44) & (xx <= 28))
        | ((yy == 45) & (xx <= 30))
        | ((yy == 46) & (xx <= 31))
        | ((yy == 47) & (xx <= 31))
        | ((yy == 48) & (xx <= 26))
        | ((yy == 49) & (xx <= 25))
    )

    # 2. Wing far on back (left shoulder white wing `y=22..43, x=41..50`)
    wing_far = a & (
        ((yy >= 22) & (yy <= 39) & (xx >= 41) & (xx <= 50))
        | ((yy == 40) & (xx >= 42) & (xx <= 48))
        | ((yy == 41) & (xx >= 42) & (xx <= 47))
        | ((yy == 42) & (xx >= 43) & (xx <= 45))
        | ((yy == 43) & (xx >= 44) & (xx <= 48))
    )

    # 3. Wing near on back (right shoulder radiant white wing `y=28..53, x=58..77`, excluding the navy scythe horn)
    horn_back = a & (yy >= 20) & (yy <= 31) & (xx >= 64) & (xx <= 72) & (
        navy_m
        | ((yy <= 28) & (xx <= 67))
        | ((yy == 28) & (xx >= 70))
        | ((yy >= 29) & (yy <= 31) & (xx >= 64) & (xx <= 66))
    )
    wing_near = (
        a
        & (
            ((yy >= 26) & (yy <= 31) & (xx >= 67))
            | ((yy >= 32) & (yy <= 39) & (xx >= 61))
            | ((yy >= 40) & (yy <= 41) & (xx >= 60))
            | ((yy >= 42) & (yy <= 44) & (xx >= 58))
            | ((yy >= 45) & (yy <= 48) & (xx >= 60))
            | ((yy >= 49) & (yy <= 53) & (xx >= 63))
        )
        & (~horn_back)
    )

    # 4. Head upper on back (crescent scythe horn + white head hair `y=20..46, x=48..72`)
    head_upper = (
        a
        & (
            horn_back
            | ((yy >= 28) & (yy <= 42) & (xx >= 48) & (xx <= 65))
            | ((yy >= 43) & (yy <= 45) & (xx >= 49) & (xx <= 60))
        )
        & (~wing_far)
        & (~wing_near)
    )

    # 5. Hind leg L on back (far left hind leg with spiky heel fur `y=55..73, x=19..34`)
    hind_leg_L = a & (
        ((yy >= 55) & (yy <= 59) & (xx >= 21) & (xx <= 35))
        | ((yy >= 60) & (yy <= 61) & (xx >= 21) & (xx <= 34))
        | ((yy >= 62) & (yy <= 63) & (xx >= 19) & (xx <= 32))
        | ((yy >= 64) & (yy <= 73) & (xx >= 19) & (xx <= 28))
    )

    # 6. Hind leg R on back (near left hind leg `y=60..78, x=32..45`)
    hind_leg_R = (
        a
        & (yy >= 60)
        & (yy <= 78)
        & (xx >= 32)
        & (xx <= 45)
        & (~hind_leg_L)
    )

    # 7. Front leg L on back (far dark shaded foreleg under belly `y=60..72, x=46..54`)
    front_leg_L = a & (
        ((yy >= 60) & (yy <= 63) & (xx >= 46) & (xx <= 53))
        | ((yy >= 64) & (yy <= 72) & (xx >= 46) & (xx <= 54))
    )

    # 8. Front leg R on back (near right foreleg `y=57..75, x=53..68`)
    front_leg_R = (
        a
        & (
            ((yy >= 57) & (yy <= 61) & (xx >= 54) & (xx <= 63))
            | ((yy >= 62) & (yy <= 75) & (xx >= 53) & (xx <= 68))
        )
        & (~front_leg_L)
    )

    empty = np.zeros_like(a)
    used = (
        tail
        | wing_far
        | wing_near
        | head_upper
        | hind_leg_L
        | hind_leg_R
        | front_leg_L
        | front_leg_R
    )
    torso = a & (~used)

    return {
        "tail": tail,
        "wing_far": wing_far,
        "hind_leg_L": hind_leg_L,
        "front_leg_L": front_leg_L,
        "hind_leg_R": hind_leg_R,
        "torso": torso,
        "front_leg_R": front_leg_R,
        "lower_jaw": empty,
        "head_upper": head_upper,
        "wing_near": wing_near,
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

    white_fill_m = is_white_fur_fill(canvas)
    navy_fill_m = is_navy_skin_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order (back-to-front):
        # 1. tail -> 2. wing_far -> 3. hind_leg_L -> 4. front_leg_L -> 5. hind_leg_R
        # -> 6. torso -> 7. front_leg_R -> 8. lower_jaw -> 9. head_upper -> 10. wing_near
        fg_over_tail = c_masks["torso"] | c_masks["hind_leg_R"]
        u_tail = ndi.binary_erosion(c_masks["tail"] | fg_over_tail, iterations=1)
        layers["tail"] = build_part_layer(
            canvas,
            c_masks["tail"],
            u_tail & fg_over_tail,
            cushion_px=6,
            allowed_fill_mask=navy_fill_m,
        )

        fg_over_wfar = c_masks["torso"] | c_masks["head_upper"] | c_masks["lower_jaw"]
        u_wfar = ndi.binary_erosion(c_masks["wing_far"] | fg_over_wfar, iterations=1)
        layers["wing_far"] = build_part_layer(
            canvas,
            c_masks["wing_far"],
            u_wfar & fg_over_wfar,
            cushion_px=5,
            allowed_fill_mask=white_fill_m,
        )

        layers["hind_leg_L"] = build_leg_stem_layer(
            canvas, c_masks["hind_leg_L"], c_masks["torso"] | c_masks["hind_leg_R"], stem_px=4
        )
        layers["front_leg_L"] = build_leg_stem_layer(
            canvas, c_masks["front_leg_L"], c_masks["torso"] | c_masks["front_leg_R"], stem_px=4
        )
        layers["hind_leg_R"] = build_leg_stem_layer(
            canvas, c_masks["hind_leg_R"], c_masks["torso"], stem_px=4
        )
        layers["front_leg_R"] = build_leg_stem_layer(
            canvas, c_masks["front_leg_R"], c_masks["torso"], stem_px=4
        )

        # IMPORTANT: `torso` cushion ONLY expands under foreground layers drawn ON TOP OF `torso`
        # (`front_leg_R`, `lower_jaw`, `head_upper`, `wing_near`), NEVER over `tail`, `wing_far`, or background legs!
        fg_over_torso = (
            c_masks["front_leg_R"]
            | c_masks["lower_jaw"]
            | c_masks["head_upper"]
            | c_masks["wing_near"]
        )
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=white_fill_m,
        )

        # (2026-11-04) no crimson mouth pocket any more: the mouth is carved by mouth_open on the composed frame.
        layers["torso"] = torso_lyr

        jaw_lyr = build_part_layer(
            canvas, c_masks["lower_jaw"], np.zeros_like(c_masks["lower_jaw"]), cushion_px=0
        )
        layers["lower_jaw"] = jaw_lyr

        # `head_upper` gets a small white-fur cushion under `wing_near` so wing rotation never leaves a seam
        fg_over_head = c_masks["wing_near"]
        u_head = ndi.binary_erosion(c_masks["head_upper"] | fg_over_head, iterations=1)
        layers["head_upper"] = build_part_layer(
            canvas,
            c_masks["head_upper"],
            u_head & fg_over_head,
            cushion_px=4,
            allowed_fill_mask=white_fill_m,
        )

        layers["wing_near"] = build_part_layer(
            canvas, c_masks["wing_near"], np.zeros_like(c_masks["wing_near"]), cushion_px=0
        )

        pivots = {
            "tail": to_canvas_pt(60.0, 48.0),
            "wing_far": to_canvas_pt(25.0, 39.0),
            "wing_near": to_canvas_pt(46.0, 41.0),
            "mouth_line": [to_canvas_pt(x, y) for x, y in MOUTH_LINE_FRONT_NATIVE],
        }

    else:
        # Back view draw order (back-to-front):
        # 1. tail -> 2. wing_far -> 3. front_leg_L -> 4. hind_leg_L -> 5. torso
        # -> 6. hind_leg_R -> 7. front_leg_R -> 8. head_upper -> 9. wing_near
        layers["lower_jaw"] = np.zeros_like(canvas)

        fg_over_tail = c_masks["torso"]
        u_tail = ndi.binary_erosion(c_masks["tail"] | fg_over_tail, iterations=1)
        layers["tail"] = build_part_layer(
            canvas,
            c_masks["tail"],
            u_tail & fg_over_tail,
            cushion_px=6,
            allowed_fill_mask=navy_fill_m,
        )

        fg_over_wfar = c_masks["torso"] | c_masks["head_upper"]
        u_wfar = ndi.binary_erosion(c_masks["wing_far"] | fg_over_wfar, iterations=1)
        layers["wing_far"] = build_part_layer(
            canvas,
            c_masks["wing_far"],
            u_wfar & fg_over_wfar,
            cushion_px=5,
            allowed_fill_mask=white_fill_m,
        )

        layers["front_leg_L"] = build_leg_stem_layer(
            canvas, c_masks["front_leg_L"], c_masks["torso"], stem_px=4
        )
        layers["hind_leg_L"] = build_leg_stem_layer(
            canvas, c_masks["hind_leg_L"], c_masks["torso"], stem_px=4
        )
        layers["hind_leg_R"] = build_leg_stem_layer(
            canvas, c_masks["hind_leg_R"], c_masks["torso"], stem_px=4
        )
        layers["front_leg_R"] = build_leg_stem_layer(
            canvas, c_masks["front_leg_R"], c_masks["torso"], stem_px=4
        )

        fg_over_torso = (
            c_masks["hind_leg_R"]
            | c_masks["front_leg_R"]
            | c_masks["head_upper"]
            | c_masks["wing_near"]
        )
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        layers["torso"] = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=white_fill_m,
        )

        fg_over_head = c_masks["wing_near"]
        u_head = ndi.binary_erosion(c_masks["head_upper"] | fg_over_head, iterations=1)
        layers["head_upper"] = build_part_layer(
            canvas,
            c_masks["head_upper"],
            u_head & fg_over_head,
            cushion_px=5,
            allowed_fill_mask=white_fill_m | navy_fill_m,
        )

        layers["wing_near"] = build_part_layer(
            canvas, c_masks["wing_near"], np.zeros_like(c_masks["wing_near"]), cushion_px=0
        )

        pivots = {
            "tail": to_canvas_pt(30.0, 46.0),
            "wing_far": to_canvas_pt(47.0, 41.0),
            "wing_near": to_canvas_pt(59.0, 42.0),
        }

    return layers, pivots


def paw_step_offset(phase: float, facing: float, lift_px: int = 2, stride_px: int = 2) -> Tuple[int, int]:
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
    left_x = int(xs.min())
    right_x = int(xs.max())
    drift = frame_idx - 2
    pts = [
        (max(2, left_x - 5 - drift), max(2, top_y + 10 - drift // 2)),
        (min(w - 4, right_x + 3 + drift), max(2, top_y + 12 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (205, 41, 32, 255)
    return out


def synthesize_absol_mega_frame(
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
    tail_dx, tail_dy = 0, 0
    fl_L_dx, fl_L_dy = 0, 0
    fl_R_dx, fl_R_dy = 0, 0
    hl_L_dx, hl_L_dy = 0, 0
    hl_R_dx, hl_R_dy = 0, 0

    wfar_ang = 0.0
    wnear_ang = 0.0
    tail_ang = 0.0

    if action == "idle":
        # Grounded 4-Legged Disaster-Beast Breathing + Angelic Wing Sweep + Scimitar Tail Whip + Ferocious Roar/Bark:
        # 1. All 4 paws (`front_leg_L/R`, `hind_leg_L/R`) stay 100% planted on the ground `(0, 0)`.
        # 2. Body and head breathe & prowl rhythmically (`-1..+1px`), modeled on Normal Absol's official GIF.
        # 3. `lower_jaw` opens `+1..+2px` during the primary roar (`frames 2..6`) and barks (`+1px` at `frames 11..12`),
        #    revealing the crimson mouth interior pocket.
        # 4. Radiant white wings (`wing_near`, `wing_far`) and scimitar `tail` pivot smoothly with phase lag!
        body_dy = int(round(1.3 * math.sin(phase - 0.2)))
        body_dx = int(round(0.7 * math.cos(phase - 0.2) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        if frame_idx in (3, 4, 5):
            jaw_open = 2
        elif frame_idx in (2, 6, 11, 12):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy

        wnear_dx = body_dx
        wnear_dy = body_dy + int(round(1.0 * math.sin(phase - 0.5)))
        wnear_ang = 4.5 * math.sin(phase - 0.5) * facing

        wfar_dx = body_dx
        wfar_dy = body_dy + int(round(1.0 * math.sin(phase - 0.8)))
        wfar_ang = 4.5 * math.sin(phase - 0.8) * (-facing)

        tail_dx = body_dx
        tail_dy = body_dy + int(round(1.0 * math.sin(phase - 1.1)))
        tail_ang = 4.2 * math.sin(phase - 1.1) * facing

    elif action == "walk":
        # Swift Quadruped Diagonal Trot/Gallop + Sweeping Angelic Wings + Roaring Jaw Bark:
        # Diagonal pairs `(front_leg_L + hind_leg_R)` and `(front_leg_R + hind_leg_L)` alternate in crisp
        # arc strides (`lift_px=2, stride_px=2`), while the white angelic wings and scimitar tail sweep
        # dynamically (`±6.0°`) and the lower jaw barks (`+1..+2px`) on each stride!
        bounce = int(round(-1.5 * abs(sin1)))
        sway_x = int(round(1.4 * sin1 * facing))

        torso_dx, torso_dy = sway_x, bounce
        head_dx = sway_x + int(round(0.8 * cos1 * facing))
        head_dy = bounce + int(round(0.8 * math.sin(2.0 * phase - 0.3)))

        s2 = math.sin(2.0 * phase)
        if s2 > 0.65:
            jaw_open = 2
        elif s2 > 0.15:
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy

        wnear_dx = sway_x
        wnear_dy = bounce + int(round(1.2 * math.sin(2.0 * phase - 0.5)))
        wnear_ang = 6.0 * math.sin(2.0 * phase - 0.5) * facing

        wfar_dx = sway_x
        wfar_dy = bounce + int(round(1.2 * math.sin(2.0 * phase - 0.8)))
        wfar_ang = 6.0 * math.sin(2.0 * phase - 0.8) * (-facing)

        tail_dx = sway_x
        tail_dy = bounce + int(round(1.2 * math.sin(2.0 * phase - 0.9)))
        tail_ang = 5.5 * math.sin(2.0 * phase - 0.9) * facing

        fl_L_dx, fl_L_dy = paw_step_offset(phase, facing, lift_px=2, stride_px=2)
        hl_R_dx, hl_R_dy = paw_step_offset(phase + 0.35, facing, lift_px=2, stride_px=2)
        fl_R_dx, fl_R_dy = paw_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)
        hl_L_dx, hl_L_dy = paw_step_offset(phase + math.pi + 0.35, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Mega Absol "Disaster Roar + Spinning Angelic-Wing Night Slash & Psycho Cut Lunge"!
        # Frames 0..3 (Wind-up & Ferocious Disaster Roar):
        #   - Crouches low and coils backward (`-3px` x, `+1px` y),
        #   - Spreads and rotates both radiant white angelic wings high (`-8.5°`),
        #   - Opens `lower_jaw` wide (`+2px` ferocious roar exposing crimson maw!).
        # Frames 4..9 (Spinning Wing-Storm & Scythe-Horn Night Slash Lunge):
        #   - Explosive leaping slash lunge (`+5px` body, `+6px` crescent scythe horn & head, forelegs leap `+4px`),
        #   - Radiant white wings (`wing_near`, `wing_far`) and scimitar `tail` execute a high-speed spinning
        #     vortex slash oscillation (`+9.5°` <-> `-7.5°` around shoulder & tail sockets!),
        #   - `lower_jaw` roars wide (`+2px`) through the slash and snaps/barks on follow-through!
        # Frames 10..15: Follow-through landing & stance recovery.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.65 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        body_dx = int(round(5.0 * k * facing))
        if frame_idx in (1, 2, 3):
            body_dy = 1
        elif frame_idx in (4, 5, 6):
            body_dy = -2
        elif frame_idx in (7, 8):
            body_dy = 1
        else:
            body_dy = 0

        torso_dx, torso_dy = body_dx, body_dy
        head_slash_dy = [0, 1, 1, 1, -2, -2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0]
        head_dx = int(round(6.0 * k * facing))
        head_dy = head_slash_dy[frame_idx]

        # Wide +2px roar during wind-up and slash peak (frames 1..6), snap at frame 7, bark at frames 8..9
        if frame_idx in (1, 2, 3, 4, 5, 6):
            jaw_open = 2
        elif frame_idx in (8, 9):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy

        # High-speed spinning angelic-wing & scimitar-tail slash rotation table
        wing_spin_near = [0.0, -5.5, -8.5, -10.5, 9.5, 11.5, -9.5, 10.0, -6.5, 4.5, -2.5, -1.0, 0.0, 0.0, 0.0, 0.0]
        wing_spin_far = [0.0, 5.5, 8.5, 10.5, -9.5, -11.5, 9.5, -10.0, 6.5, -4.5, 2.5, 1.0, 0.0, 0.0, 0.0, 0.0]
        tail_spin_ang = [0.0, -5.0, -7.5, -9.5, 9.5, -10.0, 10.0, -8.0, 6.0, -3.5, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0]

        wnear_dx = int(round(5.5 * k * facing))
        wnear_dy = body_dy + (-1 if frame_idx in (2, 3, 4, 5) else 0)
        wnear_ang = wing_spin_near[frame_idx] * facing

        wfar_dx = int(round(5.5 * k * facing))
        wfar_dy = body_dy + (-1 if frame_idx in (2, 3, 4, 5) else 0)
        wfar_ang = wing_spin_far[frame_idx] * (-facing)

        tail_dx = body_dx
        tail_dy = body_dy + (-1 if frame_idx in (4, 6, 8) else (1 if frame_idx in (5, 7) else 0))
        tail_ang = tail_spin_ang[frame_idx] * facing

        # Forelegs leap & slash forward; hind legs drive the lunge
        fore_x = int(round(4.0 * k * facing))
        fore_y = -2 if frame_idx in (4, 5) else (-1 if frame_idx == 6 else 0)
        hind_x = int(round(3.0 * k * facing))
        hind_y = -1 if frame_idx in (4, 5) else 0

        fl_R_dx, fl_R_dy = fore_x, fore_y
        fl_L_dx, fl_L_dy = int(round(3.5 * k * facing)), fore_y
        hl_R_dx, hl_R_dy = hind_x, hind_y
        hl_L_dx, hl_L_dy = hind_x, hind_y

    elif action == "hurt":
        # Knockback + hit-shake + pained disaster roar (`lower_jaw` open +2px) + wings flared + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-3.5 * env * facing)) + jit
        body_dy = int(round(-1.0 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-1.0 * env * facing))
        head_dy = int(round(-1.5 * env))

        jaw_open = 2 if frame_idx in (2, 3, 4, 5) else (1 if 1 <= frame_idx <= 7 else 0)
        jaw_dx = head_dx
        jaw_dy = head_dy

        wnear_dx = body_dx
        wnear_dy = body_dy + int(round(-1.5 * env))
        wnear_ang = -7.0 * env * facing

        wfar_dx = body_dx
        wfar_dy = body_dy + int(round(-1.5 * env))
        wfar_ang = 7.0 * env * (-facing)

        tail_dx = body_dx
        tail_dy = int(round(1.2 * env))
        tail_ang = -5.5 * env * facing

        slide_x = int(round(body_dx * 0.65))
        fl_L_dx, fl_R_dx = slide_x, slide_x
        hl_L_dx, hl_R_dx = slide_x, slide_x

    tail_rot = rotate_layer_nearest(layers["tail"], tail_ang, pivots["tail"])
    wfar_rot = rotate_layer_nearest(layers["wing_far"], wfar_ang, pivots["wing_far"])
    wnear_rot = rotate_layer_nearest(layers["wing_near"], wnear_ang, pivots["wing_near"])

    if not is_back:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(wfar_rot, wfar_dx, wfar_dy),
            shift_layer_int(layers["hind_leg_L"], hl_L_dx, hl_L_dy),
            shift_layer_int(layers["front_leg_L"], fl_L_dx, fl_L_dy),
            shift_layer_int(layers["hind_leg_R"], hl_R_dx, hl_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["front_leg_R"], fl_R_dx, fl_R_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(wnear_rot, wnear_dx, wnear_dy),
        ]
    else:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(wfar_rot, wfar_dx, wfar_dy),
            shift_layer_int(layers["front_leg_L"], fl_L_dx, fl_L_dy),
            shift_layer_int(layers["hind_leg_L"], hl_L_dx, hl_L_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["hind_leg_R"], hl_R_dx, hl_R_dy),
            shift_layer_int(layers["front_leg_R"], fl_R_dx, fl_R_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(wnear_rot, wnear_dx, wnear_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    mouth_line = pivots.get("mouth_line")
    if (not is_back) and mouth_line and jaw_open > 0:
        line = mouth_open.transform_points(mouth_line, dx=head_dx, dy=head_dy)
        amount = jaw_open if action == "attack" else min(1, jaw_open)  # growl 1, roar peak 2
        out = mouth_open.open_mouth(out, line, amount=amount, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=0.6)
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

    # Tight bounds of front (68x70) and back (71x59)
    max_w, max_h = 71, 70
    cell_w = 104
    cell_h = 96
    foot_y = 88

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
                frame_rgba = synthesize_absol_mega_frame(
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
        "generator": "forge_absol_mega_actions",
        "key": "0359_absol-mega",
        "dex": 359,
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

    print(f"Built 0359_absol-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
