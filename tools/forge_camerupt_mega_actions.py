#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Camerupt (`0323_camerupt-mega`) modeled after Normal Camerupt's official GIF
(`0323_camerupt/normal/front.gif`) and Mega Camerupt's Erupting Magma Volcano Kaiju anatomy (`front.png` & `back.png`):
  - Exact 9-layer anatomical rigging with true rotational pivot at the active volcano crater center:
      1. `hind_foot_L` (left rear hoof — 100% grounded `(0,0)` in `idle`, heavy 4-legged diagonal stomp in `walk`)
      2. `hind_foot_R` (right rear hoof — 100% grounded `(0,0)` in `idle`, heavy 4-legged diagonal stomp in `walk`)
      3. `front_foot_L` (left front hoof — 100% grounded `(0,0)` in `idle`, alternating stomp in `walk`,
                         lifted high and slammed down in `attack` Earth Power / Eruption!)
      4. `front_foot_R` (right front hoof — 100% grounded `(0,0)` in `idle`, alternating stomp in `walk`,
                         heavy earth-shattering stomp in `attack`!)
      5. `torso_fur_skirt` (shaggy red-orange volcanic fur skirt + dark-maroon mouth interior pocket behind `lower_jaw`)
      6. `volcano_crater` (colossal dorsal rock volcano cone, flowing molten lava streams, and black volcanic boulders —
                           pulses and heaves upward during breathing and seismic eruptions)
      7. `magma_plume_top` (churning yellow/orange/red lava plume crown at the crater mouth — pivoted at the crater center
                            with `rotate_layer_nearest` so the molten magma swirls/rotates and erupts skyward!)
      8. `lower_jaw` (tan lower jaw & chin — opens `+1px` for deep volcanic roar/bark in `idle`/`walk` and
                      `+2px` wide for a thunderous Kaiju Eruption roar in `attack` and `hurt`!)
      9. `head_upper` (dark-grey M forehead emblem, red tuft ears, half-lidded eyes, nostrils, and upper snout —
                       100% rigid integer motion `shift_layer_int` so the face, eyes, and M-mark never deform by 1px!)
"""

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mouth_open  # noqa: E402  (small face-following mouth opening, 2026-11-04)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0323_camerupt/mega/camerupt-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0323_camerupt/mega/camerupt-mega"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16
VERSION = "camerupt_mega_v1m"

# Closed mouth line on the native 96x96 front.png, ordered hinge/corner -> lip tip (2026-11-04 mouth fix).
# The back.png views (N/NE/NW) get no opening.
MOUTH_LINE_FRONT = [(21.0, 59.0), (19.0, 60.0), (14.0, 60.0)]
MOUTH_DEEP = (24, 24, 24)       # darkest palette colour (outline)
MOUTH_INNER = (115, 41, 57)     # existing dark maroon (mouth interior) palette colour


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
        | ((r == 115) & (g == 41) & (b == 57))
    )


def is_fur_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 246) & (g == 139) & (b == 98))
        | ((r == 238) & (g == 106) & (b == 74))
        | ((r == 197) & (g == 57) & (b == 32))
    )


def is_magma_or_stone_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 247) & (g == 206) & (b == 63))
        | ((r == 255) & (g == 202) & (b == 20))
        | ((r == 255) & (g == 147) & (b == 20))
        | ((r == 255) & (g == 64) & (b == 20))
        | ((r == 155) & (g == 31) & (b == 31))
        | ((r == 213) & (g == 213) & (b == 180))
        | ((r == 172) & (g == 164) & (b == 139))
        | ((r == 58) & (g == 58) & (b == 58))
        | ((r == 74) & (g == 74) & (b == 74))
        | ((r == 98) & (g == 98) & (b == 98))
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
      1. `magma_plume_top`: Erupting yellow/orange/red lava crown at the volcano crater mouth (`y=14..23, x=35..63`).
      2. `volcano_crater`: Colossal dorsal rock volcano cone, flowing molten lava streams, and black volcanic boulders (`y=23..59, x=35..73`).
      3. `hind_foot_L`: Rear-left hoof (tucked behind front-left skirt in front view).
      4. `hind_foot_R`: Rear-right grounded dark-grey hoof (`x=61..71, y=73..76`).
      5. `front_foot_L`: Front-left grounded dark-grey hoof (`x=16..30, y=73..77`).
      6. `front_foot_R`: Front-right foreground grounded dark-grey hoof (`x=38..52, y=76..81`).
      7. `torso_fur_skirt`: Shaggy red-orange volcanic fur skirt covering the body flanks.
      8. `lower_jaw`: Tan lower jaw and chin outline (`x=13..31, y=60..64`).
      9. `head_upper`: Red head crest tufts, ears, dark-grey M forehead emblem, half-lidded eyes, nostrils, and upper snout (`x=10..41, y=23..60`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Magma plume top (swirling yellow/orange/red lava crown at the top of the crater)
    magma_plume_top = a & (
        ((yy >= 14) & (yy <= 22) & (xx >= 35) & (xx <= 63))
        | ((yy == 23) & (xx >= 38) & (xx <= 55))
    )

    # 2. Lower jaw (tan lower jaw & chin outline below nostrils and upper fangs)
    lower_jaw = a & (
        ((yy == 60) & (xx >= 26) & (xx <= 31))
        | ((yy == 61) & (xx >= 13) & (xx <= 30))
        | ((yy == 62) & (xx >= 14) & (xx <= 28))
        | ((yy == 63) & (xx >= 15) & (xx <= 27))
        | ((yy == 64) & (xx >= 17) & (xx <= 25))
    )

    # 3. Head upper (red crest hair, ears, M forehead emblem, eyes, upper snout & nostrils)
    head_zone = (
        ((yy >= 23) & (yy <= 30) & (xx <= 32))
        | ((yy >= 31) & (yy <= 35) & (xx <= 37))
        | ((yy >= 36) & (yy <= 37) & (xx <= 36))
        | ((yy == 38) & ((xx <= 36) | ((xx >= 38) & (xx <= 41))))
        | ((yy == 39) & (xx <= 41))
        | ((yy >= 40) & (yy <= 41) & (xx <= 39))
        | ((yy == 42) & (xx <= 38))
        | ((yy == 43) & (xx <= 37))
        | ((yy == 44) & (xx <= 36))
        | ((yy == 45) & (xx <= 35))
        | ((yy >= 46) & (yy <= 52) & (xx <= 32))
        | ((yy >= 53) & (yy <= 57) & (xx <= 33))
        | ((yy >= 58) & (yy <= 59) & (xx <= 32))
        | ((yy == 60) & (xx >= 13) & (xx <= 31))
    )
    head_upper = a & head_zone & (~lower_jaw) & (~magma_plume_top)

    # 4. Volcano crater (rock volcano cone, flowing lava streams, and black volcanic boulder collar)
    volcano_zone = (
        ((yy >= 23) & (yy <= 45) & (xx >= 35))
        | ((yy == 46) & (xx >= 36) & (xx <= 70))
        | ((yy == 47) & (xx >= 36) & (xx <= 71))
        | ((yy == 48) & (xx >= 37) & (xx <= 72))
        | ((yy == 49) & (xx >= 37) & (xx <= 71))
        | ((yy == 50) & (xx >= 37) & (xx <= 70))
        | ((yy == 51) & (xx >= 38) & (xx <= 69))
        | ((yy == 52) & (xx >= 38) & (xx <= 68))
        | ((yy == 53) & (xx >= 39) & (xx <= 66))
        | ((yy == 54) & (xx >= 39) & (xx <= 65))
        | ((yy == 55) & (xx >= 40) & (xx <= 64))
        | ((yy == 56) & (xx >= 41) & (xx <= 53))
        | ((yy == 57) & (xx >= 41) & (xx <= 52))
        | ((yy == 58) & (xx >= 42) & (xx <= 51))
        | ((yy == 59) & (xx >= 44) & (xx <= 46))
        | ((yy == 59) & (xx >= 49) & (xx <= 51))
    )
    volcano_crater = a & volcano_zone & (~magma_plume_top) & (~head_upper)

    # 5. Four hooves (`front_foot_L`, `front_foot_R`, `hind_foot_L`, `hind_foot_R`)
    front_foot_L = a & (
        ((yy == 73) & (((xx >= 19) & (xx <= 22)) | ((xx >= 25) & (xx <= 27))))
        | ((yy == 74) & (xx >= 16) & (xx <= 28))
        | ((yy == 75) & (xx >= 17) & (xx <= 28))
        | ((yy == 76) & (xx >= 18) & (xx <= 30))
        | ((yy == 77) & (xx >= 21) & (xx <= 25))
    )
    front_foot_R = a & (
        ((yy == 76) & (((xx >= 40) & (xx <= 42)) | ((xx >= 46) & (xx <= 48))))
        | ((yy >= 77) & (yy <= 81) & (xx >= 38) & (xx <= 52))
    )
    hind_foot_R = a & (
        ((yy == 73) & (xx >= 67) & (xx <= 69))
        | ((yy == 74) & (xx >= 65) & (xx <= 71))
        | ((yy == 75) & (xx >= 62) & (xx <= 70))
        | ((yy == 76) & (xx >= 61) & (xx <= 66))
    )
    hind_foot_L = np.zeros_like(a)

    used = (
        magma_plume_top
        | lower_jaw
        | head_upper
        | volcano_crater
        | front_foot_L
        | front_foot_R
        | hind_foot_R
        | hind_foot_L
    )
    torso_fur_skirt = a & (~used)

    return {
        "magma_plume_top": magma_plume_top,
        "volcano_crater": volcano_crater,
        "hind_foot_L": hind_foot_L,
        "hind_foot_R": hind_foot_R,
        "front_foot_L": front_foot_L,
        "front_foot_R": front_foot_R,
        "torso_fur_skirt": torso_fur_skirt,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 9 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Magma plume top (swirling yellow/orange/red lava crown at crater mouth y=13..21)
    magma_plume_top = a & (
        ((yy >= 13) & (yy <= 20) & (xx >= 34) & (xx <= 61))
        | ((yy == 21) & (xx >= 37) & (xx <= 53))
    )

    # 2. Lower jaw (tan lower jaw & lower teeth on right profile y=59..62, x=71..83)
    lower_jaw = a & (
        ((yy == 59) & (xx >= 75) & (xx <= 80))
        | ((yy == 60) & (xx >= 71) & (xx <= 83))
        | ((yy == 61) & (xx >= 72) & (xx <= 80))
        | ((yy == 62) & (xx >= 73) & (xx <= 78))
    )

    # 3. Head upper (red head crest tufts, ears, right eye, nostrils, and upper snout)
    head_zone = (
        ((yy >= 23) & (yy <= 35) & (xx >= 64))
        | ((yy == 36) & (xx >= 73))
        | ((yy == 37) & (xx >= 74))
        | ((yy == 38) & (xx >= 73))
        | ((yy >= 39) & (yy <= 42) & (xx >= 74))
        | ((yy >= 43) & (yy <= 44) & (xx >= 73))
        | ((yy == 45) & (xx >= 71))
        | ((yy >= 46) & (yy <= 52) & (xx >= 72))
        | ((yy == 53) & (xx >= 70))
        | ((yy >= 54) & (yy <= 58) & (xx >= 70))
        | ((yy == 59) & (xx >= 71))
        | ((yy == 60) & (xx >= 81))
    )
    head_upper = a & head_zone & (~lower_jaw)

    # 4. Volcano crater (rock volcano cone, flowing lava streams, and black volcanic boulder collar)
    volcano_zone = (
        ((yy >= 21) & (yy <= 35) & (xx >= 30) & (xx <= 63))
        | ((yy == 36) & (xx >= 29) & (xx <= 72))
        | ((yy == 37) & (xx >= 24) & (xx <= 73))
        | ((yy == 38) & (xx >= 23) & (xx <= 72))
        | ((yy == 39) & (xx >= 19) & (xx <= 73))
        | ((yy >= 40) & (yy <= 41) & (xx >= 18) & (xx <= 73))
        | ((yy == 42) & (xx >= 17) & (xx <= 73))
        | ((yy == 43) & (xx >= 16) & (xx <= 72))
        | ((yy == 44) & (xx >= 16) & (xx <= 72) & (~((xx == 21) | (xx == 28))))
        | ((yy == 45) & (xx >= 16) & (xx <= 70) & (~((xx >= 21) & (xx <= 30))))
        | ((yy == 46) & (((xx >= 17) & (xx <= 19)) | ((xx >= 32) & (xx <= 71))))
        | ((yy == 47) & (xx >= 32) & (xx <= 71))
        | ((yy == 48) & (xx >= 32) & (xx <= 71))
        | ((yy == 49) & (xx >= 33) & (xx <= 71))
        | ((yy == 50) & (xx >= 35) & (xx <= 71))
        | ((yy == 51) & (xx >= 35) & (xx <= 71))
        | ((yy == 52) & (xx >= 36) & (xx <= 71))
        | (
            (yy == 53)
            & (
                ((xx >= 37) & (xx <= 39))
                | ((xx >= 42) & (xx <= 45))
                | ((xx >= 48) & (xx <= 60))
                | ((xx >= 63) & (xx <= 69))
            )
        )
        | ((yy == 54) & (((xx >= 50) & (xx <= 59)) | ((xx >= 64) & (xx <= 67))))
        | ((yy == 55) & (xx >= 51) & (xx <= 57))
        | ((yy == 56) & (xx >= 52) & (xx <= 55))
    )
    volcano_crater = a & volcano_zone & (~magma_plume_top) & (~head_upper)

    # 5. Four hooves on back.png
    hind_foot_L = a & (
        ((yy == 76) & (xx >= 20) & (xx <= 27))
        | ((yy == 77) & (xx >= 22) & (xx <= 29))
    )
    hind_foot_R = a & (
        ((yy == 77) & (((xx >= 32) & (xx <= 35)) | ((xx >= 42) & (xx <= 43))))
        | ((yy >= 78) & (yy <= 81) & (xx >= 32) & (xx <= 46))
    )
    front_foot_R = a & (
        ((yy == 76) & (((xx >= 54) & (xx <= 57)) | ((xx >= 61) & (xx <= 65))))
        | ((yy >= 77) & (yy <= 79) & (xx >= 55) & (xx <= 67))
    )
    front_foot_L = np.zeros_like(a)

    used = (
        magma_plume_top
        | lower_jaw
        | head_upper
        | volcano_crater
        | hind_foot_L
        | hind_foot_R
        | front_foot_R
        | front_foot_L
    )
    torso_fur_skirt = a & (~used)

    return {
        "magma_plume_top": magma_plume_top,
        "volcano_crater": volcano_crater,
        "hind_foot_L": hind_foot_L,
        "hind_foot_R": hind_foot_R,
        "front_foot_L": front_foot_L,
        "front_foot_R": front_foot_R,
        "torso_fur_skirt": torso_fur_skirt,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
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
        fill_px = (58, 58, 58, 255)
        for dy_up in range(1, stem_px + 1):
            py = cy_min - dy_up
            if py >= 0 and torso_mask[py, x_col]:
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
    places onto the target `(cell_h, cell_w)` canvas with hooves aligned to `foot_y`,
    and builds cushioned RGBA layers + volcano crater pivot.
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

    fur_fill_m = is_fur_fill(canvas)
    volcano_fill_m = is_magma_or_stone_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    # 1..4: Four hooves with upward dark-grey ankle stems behind `torso_fur_skirt`
    layers["hind_foot_L"] = build_foot_stem_layer(canvas, c_masks["hind_foot_L"], c_masks["torso_fur_skirt"], stem_px=4)
    layers["hind_foot_R"] = build_foot_stem_layer(canvas, c_masks["hind_foot_R"], c_masks["torso_fur_skirt"], stem_px=4)
    layers["front_foot_L"] = build_foot_stem_layer(canvas, c_masks["front_foot_L"], c_masks["torso_fur_skirt"], stem_px=4)
    layers["front_foot_R"] = build_foot_stem_layer(canvas, c_masks["front_foot_R"], c_masks["torso_fur_skirt"], stem_px=4)

    # 5: `torso_fur_skirt` — ONLY expands cushion under foreground layers drawn ON TOP OF `torso_fur_skirt`
    # (`volcano_crater`, `lower_jaw`, `head_upper`), NEVER over background hooves!
    fg_over_torso = c_masks["volcano_crater"] | c_masks["lower_jaw"] | c_masks["head_upper"]
    u_torso = ndi.binary_erosion(c_masks["torso_fur_skirt"] | fg_over_torso, iterations=1)
    torso_lyr = build_part_layer(
        canvas,
        c_masks["torso_fur_skirt"],
        u_torso & fg_over_torso,
        cushion_px=5,
        allowed_fill_mask=fur_fill_m,
    )
    layers["torso_fur_skirt"] = torso_lyr

    # 6: `volcano_crater` — expands lava/stone cushion ONLY under `magma_plume_top` and `head_upper`,
    # NEVER over `torso_fur_skirt`!
    fg_over_volcano = c_masks["magma_plume_top"] | c_masks["head_upper"]
    u_volcano = ndi.binary_erosion(c_masks["volcano_crater"] | fg_over_volcano, iterations=1)
    volcano_lyr = build_part_layer(
        canvas,
        c_masks["volcano_crater"],
        u_volcano & fg_over_volcano,
        cushion_px=6,
        allowed_fill_mask=volcano_fill_m,
    )
    # Reinforce the inner magma column neck under `magma_plume_top` so when the lava crown
    # erupts `-3px` upward and spins (`±9.5°`), a solid molten magma neck connects it seamlessly!
    plume_ys, plume_xs = np.where(c_masks["magma_plume_top"])
    if len(plume_ys) > 0:
        py_max = int(plume_ys.max())
        for py in range(max(0, py_max - 4), py_max + 1):
            row_xs = plume_xs[plume_ys == py]
            if len(row_xs) >= 4:
                px0 = int(row_xs.min()) + 1
                px1 = int(row_xs.max()) - 1
                for px in range(px0, px1 + 1):
                    if c_masks["magma_plume_top"][py, px]:
                        if px in (px0, px1):
                            volcano_lyr[py, px] = (255, 64, 20, 255)
                        elif px in (px0 + 1, px1 - 1):
                            volcano_lyr[py, px] = (255, 147, 20, 255)
                        else:
                            volcano_lyr[py, px] = (247, 206, 63, 255) if not is_back else (255, 202, 20, 255)
    layers["volcano_crater"] = volcano_lyr

    # 7: `magma_plume_top` — Swirling/erupting molten magma crown at the crater mouth
    layers["magma_plume_top"] = build_part_layer(
        canvas,
        c_masks["magma_plume_top"],
        np.zeros_like(c_masks["magma_plume_top"]),
        cushion_px=0,
    )

    # 8 & 9: `head_upper` and `lower_jaw` (the jaw always moves with the head; no jaw drop).
    head_lyr = build_part_layer(
        canvas,
        c_masks["head_upper"],
        np.zeros_like(c_masks["head_upper"]),
        cushion_px=0,
    )
    # (2026-11-04) The rectangular dark-maroon mouth pocket is no longer painted; the mouth opens with
    # mouth_open.open_mouth along the closed mouth line instead (jaw layer stays in place).
    layers["head_upper"] = head_lyr

    layers["lower_jaw"] = build_part_layer(
        canvas,
        c_masks["lower_jaw"],
        np.zeros_like(c_masks["lower_jaw"]),
        cushion_px=0,
    )

    if not is_back:
        pivots = {
            "magma_plume_top": to_canvas_pt(49.0, 19.0),
            "mouth_line": [to_canvas_pt(x, y) for x, y in MOUTH_LINE_FRONT],
        }
    else:
        pivots = {
            "magma_plume_top": to_canvas_pt(47.5, 17.0),
            "mouth_line": [],
        }

    return layers, pivots


def foot_step_offset(phase: float, facing: float, lift_px: int = 2, stride_px: int = 2) -> Tuple[int, int]:
    p = phase % (2.0 * math.pi)
    s = math.sin(p)
    c = math.cos(p)
    dx = int(round(stride_px * c * facing))
    dy = int(round(-lift_px * max(0.0, s)))
    return dx, dy


def add_erupting_magma_sparks(canvas: np.ndarray, crater_pivot: Tuple[float, float], frame_idx: int) -> np.ndarray:
    """
    Adds palette-locked molten magma ejecta sparks `(247, 206, 63)` and `(255, 64, 20)` bordered by
    `(255, 147, 20)` above the erupting volcano crater during `attack` eruption frames (`4..10`).
    """
    if frame_idx not in (4, 5, 6, 7, 8, 9, 10):
        return canvas
    out = canvas.copy()
    h, w = out.shape[:2]
    cx = int(round(crater_pivot[0]))
    cy = int(round(crater_pivot[1]))
    step = frame_idx - 4
    offsets = [
        (-8 - step, -8 - (step // 2), (247, 206, 63, 255)),
        (8 + step, -7 - (step // 2), (255, 147, 20, 255)),
        (0 + (1 if step % 2 == 0 else -1), -10 - (step // 2), (255, 64, 20, 255)),
    ]
    for ox, oy, col in offsets:
        px = cx + ox
        py = cy + oy
        if 1 <= px < w - 2 and 1 <= py < h - 2:
            out[py:py + 2, px:px + 2] = col
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
            out[sy:sy + 2, sx:sx + 2] = (248, 248, 248, 255)
    return out


def synthesize_camerupt_mega_frame(
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
    volcano_dx, volcano_dy = 0, 0
    plume_dx, plume_dy = 0, 0
    head_dx, head_dy = 0, 0
    jaw_dx, jaw_dy = 0, 0
    jaw_open = 0

    ffL_dx, ffL_dy = 0, 0
    ffR_dx, ffR_dy = 0, 0
    hfL_dx, hfL_dy = 0, 0
    hfR_dx, hfR_dy = 0, 0

    plume_ang = 0.0

    if action == "idle":
        # Active Magma Volcano Pulsation + Swirling Lava Crown Rotation + Deep Kaiju Bellow / Roar:
        # 1. All 4 hooves (`front_foot_L`, `front_foot_R`, `hind_foot_L`, `hind_foot_R`) stay 100% grounded `(0, 0)`.
        # 2. Shaggy volcanic fur skirt & dorsal volcano cone heave rhythmically with magma pressure.
        # 3. `magma_plume_top` swirls/rotates (`±4.5°` around crater pivot) and bubbles upward (`-1..+1px`).
        # 4. `lower_jaw` opens `+1px` during the primary low-frequency bellow (`frames 2..7`) and secondary bark (`frames 11..13`),
        #    exposing the dark-maroon mouth interior `(115, 41, 57)`.
        body_dy = int(round(1.0 * math.sin(phase - 0.2)))
        body_dx = int(round(0.5 * max(0.0, math.sin(phase - 0.2)) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        volcano_dx = body_dx
        volcano_dy = body_dy + int(round(-0.6 * max(0.0, math.sin(phase))))

        plume_dx = volcano_dx + int(round(0.8 * math.cos(phase) * facing))
        plume_dy = volcano_dy + int(round(-1.2 * max(0.0, math.sin(phase))))
        plume_ang = 4.5 * math.sin(phase) * facing

        head_dx = body_dx
        head_dy = body_dy

        jaw_open = 1 if (frame_idx in (2, 3, 4, 5, 6, 7) or frame_idx in (11, 12)) else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays with the head; opening is carved by mouth_open

    elif action == "walk":
        # Earth-Shaking Quadruped Diagonal Stomp Gait (`quadruped_beast`):
        # Diagonal pairs `(front_foot_L + hind_foot_R)` and `(front_foot_R + hind_foot_L)` stomp in opposition,
        # shaggy volcanic fur skirt and dorsal volcano cone heave with 2-beat cadence,
        # lava plume crown swirls (`±5.0°`), and `lower_jaw` barks/roars (`+1px`) on each heavy stomp!
        bounce = int(round(-1.4 * abs(sin1)))
        sway_x = int(round(1.0 * sin1 * facing))

        torso_dx, torso_dy = sway_x, bounce
        volcano_dx = sway_x
        volcano_dy = bounce + int(round(0.8 * math.sin(2.0 * phase - 0.4)))

        plume_dx = volcano_dx + int(round(1.0 * cos1 * facing))
        plume_dy = volcano_dy + int(round(-1.0 * abs(cos1)))
        plume_ang = 5.0 * sin1 * facing

        head_dx = sway_x + int(round(0.6 * cos1 * facing))
        head_dy = bounce

        jaw_open = 1 if math.sin(2.0 * phase) > 0.15 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays with the head; opening is carved by mouth_open

        ffL_dx, ffL_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        hfR_dx, hfR_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        ffR_dx, ffR_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)
        hfL_dx, hfL_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Mega Camerupt "Thunderous Kaiju Roar + Swirling Magma Eruption (ふんか) & Earth Power Front-Hoof Crush"!
        # Frames 0..3 (Wind-up & Magma Compression Roar):
        #   - Compresses body down (`+1px`) and pulls back (`-3px`), pressurizing the dorsal volcano crater,
        #   - Opens `lower_jaw` wide (`+2px` thunderous Kaiju roar!),
        #   - Lifts front hooves (`front_foot_L`, `front_foot_R` `-2px` up) to prepare an earth-shattering stomp!
        # Frames 4..9 (Cataclysmic Magma Eruption & Earth Power Front-Hoof Slam):
        #   - Explosive forward lunge (`+5px` body & head), front hooves slam down hard into the earth (`+3px` forward, `0px` grounded),
        #   - Dorsal volcano cone heaves upward (`-2px`) while `magma_plume_top` erupts skyward (`-3px`) and executes a
        #     high-speed vortex rotation (`+9.5° -> -9.0° -> +9.0° -> -8.0°` around the crater center) with flying magma ejecta sparks!
        #   - `lower_jaw` stays wide open (`+2px` at frames 4..8) unleashing a continuous volcanic roar!
        # Frames 10..15: Heavy follow-through recovery to grounded stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.60 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        body_dx = int(round(5.0 * k * facing))
        if frame_idx in (1, 2, 3):
            body_dy = 1
        elif frame_idx in (4, 5, 6, 7):
            body_dy = -1
        else:
            body_dy = 0

        torso_dx, torso_dy = body_dx, body_dy

        # Dorsal volcano cone heaves upward during the eruption peak (frames 4..8)
        volcano_erupt_dy = [0, 1, 1, 1, -2, -2, -2, -2, -1, -1, 0, 0, 0, 0, 0, 0]
        volcano_dx = body_dx
        volcano_dy = volcano_erupt_dy[frame_idx]

        # Swirling magma plume crown erupts skyward (-3px) and spins violently like a lava vortex!
        plume_erupt_dy = [0, 1, 1, 0, -3, -3, -3, -3, -2, -2, -1, 0, 0, 0, 0, 0]
        plume_vortex_ang = [0.0, -3.5, -6.0, -8.0, 8.5, -9.5, 9.5, -9.0, 8.0, -6.0, 3.5, -1.5, 0.0, 0.0, 0.0, 0.0]
        plume_dx = volcano_dx
        plume_dy = plume_erupt_dy[frame_idx]
        plume_ang = plume_vortex_ang[frame_idx] * facing

        head_dx = int(round(5.4 * k * facing))
        head_dy = body_dy

        # Thunderous +2px wide Kaiju roar during wind-up and full eruption (frames 2..8)!
        if frame_idx in (3, 4, 5, 6, 7):
            jaw_open = 2
        elif frame_idx in (1, 2, 8, 9, 10):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays with the head; opening is carved by mouth_open

        # Front hooves rear up (-2px) during wind-up (frames 1..3), then slam down (+3px forward) on eruption impact!
        if frame_idx in (1, 2, 3):
            ffL_dx = int(round(-1.5 * facing))
            ffR_dx = int(round(-1.5 * facing))
            ffL_dy = -2 if frame_idx in (2, 3) else -1
            ffR_dy = -2 if frame_idx in (2, 3) else -1
        else:
            lead_x = int(round(3.5 * max(0.0, k) * facing)) if k >= 0 else int(round(1.5 * k * facing))
            ffL_dx, ffL_dy = lead_x, 0
            ffR_dx, ffR_dy = lead_x, 0

        rear_x = int(round(2.4 * k * facing))
        hfL_dx, hfL_dy = rear_x, 0
        hfR_dx, hfR_dy = rear_x, 0

    elif action == "hurt":
        # Heavy Volcanic Kaiju knockback + hit-shake + pained roar (`lower_jaw` open +2px) + lava plume jolt + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-3.5 * env * facing)) + jit
        body_dy = int(round(-1.0 * env))

        torso_dx, torso_dy = body_dx, body_dy
        volcano_dx = body_dx
        volcano_dy = int(round(-1.4 * env))

        plume_dx = volcano_dx + int(round(-1.0 * env * facing))
        plume_dy = int(round(-2.0 * env))
        plume_ang = -7.0 * env * facing

        head_dx = body_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-1.2 * env))

        jaw_open = 1 if 1 <= frame_idx <= 7 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays with the head; opening is carved by mouth_open

        slide_x = int(round(body_dx * 0.65))
        ffL_dx, ffR_dx, hfL_dx, hfR_dx = slide_x, slide_x, slide_x, slide_x

    plume_rot = rotate_layer_nearest(layers["magma_plume_top"], plume_ang, pivots["magma_plume_top"])

    comp = [
        shift_layer_int(layers["hind_foot_L"], hfL_dx, hfL_dy),
        shift_layer_int(layers["hind_foot_R"], hfR_dx, hfR_dy),
        shift_layer_int(layers["front_foot_L"], ffL_dx, ffL_dy),
        shift_layer_int(layers["front_foot_R"], ffR_dx, ffR_dy),
        shift_layer_int(layers["torso_fur_skirt"], torso_dx, torso_dy),
        shift_layer_int(layers["volcano_crater"], volcano_dx, volcano_dy),
        shift_layer_int(plume_rot, plume_dx, plume_dy),
        shift_layer_int(layers["head_upper"], head_dx, head_dy),
        shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
    ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    if jaw_open > 0 and pivots.get("mouth_line"):
        line = mouth_open.transform_points(pivots["mouth_line"], dx=head_dx, dy=head_dy)
        out = mouth_open.open_mouth(out, line, amount=jaw_open, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=0.6)
    if action == "attack":
        out = add_erupting_magma_sparks(out, pivots["magma_plume_top"], frame_idx)
    elif action == "hurt":
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

    # Tight bounds of front (75x68) and back (83x69)
    max_w, max_h = 83, 69
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
                frame_rgba = synthesize_camerupt_mega_frame(
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
        "generator": "forge_camerupt_mega_actions",
        "key": "0323_camerupt-mega",
        "dex": 323,
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
        for gif_name in ("front.gif", "front-reverse.gif", "back.gif", "back-reverse.gif"):
            shutil.copy2(MEGA_DIR / gif_name, SYNC_MEGA_DIR / gif_name)

    print(f"Built 0323_camerupt-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
