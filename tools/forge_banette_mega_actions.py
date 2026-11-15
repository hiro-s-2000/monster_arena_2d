#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Banette (`0354_banette-mega`) modeled after Mega Banette's official 47-frame GIF
(`0354_banette/mega/banette-mega/front.gif`) and native sprites (`front.png` & `back.png`):
  - Exact 8-layer anatomical rigging with true joint pivots:
      1. `head_tail_plume` (long undulating ghost head-tail plume with golden zipper rings — pivoted at
                            the back of the cranium via `rotate_layer_nearest` so it waves like a grudge!)
      2. `arm_far` (far-side giant unzipped cursed claw arm with pink grudge core fingers — pivoted at
                    the far shoulder socket via `rotate_layer_nearest` for wide puppet swings & spinning slashes!)
      3. `leg_L` (left unzipped doll hem & pink cursed leg tip — ghostly puppet hover & alternating step)
      4. `leg_R` (right unzipped golden-trimmed hem & pink cursed leg tip — alternating puppet step)
      5. `torso` (dark doll torso, unzipped chest/belly pink curse core, and glowing pink/crimson cursed
                  mouth interior pocket revealed when the golden zipper jaw opens!)
      6. `lower_jaw` (golden zipper lower teeth, pull-tab, and chin — opens `+1px` for sinister cackling/roaring
                      in `idle`/`walk` and `+2px` wide for a ferocious cursed roar in `attack` and `hurt`!)
      7. `head_upper` (unzipped head crest, glowing pink eyes with white pupils, and upper golden zipper teeth —
                       100% rigid integer motion `shift_layer_int` so the face and eyes never deform by a single pixel!)
      8. `arm_near` (foreground giant unzipped cursed claw arm — pivoted at the near shoulder socket via
                     `rotate_layer_nearest` for dynamic spinning Shadow Claw / Phantom Force slashes!)
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
MEGA_DIR = ROOT / "monsterSprite/gen3/0354_banette/mega/banette-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0354_banette/mega/banette-mega"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16

# Mouth fix (2026-11-04, user request): the zipper mouth no longer opens by dropping the `lower_jaw` layer and
# revealing a rectangular pink pocket (read as a jutting chin). The jaw stays put and `mouth_open.open_mouth`
# carves a small lens-shaped gap just below the closed zipper line (front views only; back views show no mouth).
# Closed zipper line on native front.png (96x96), ordered back corner (right) -> front lip (left, facing side).
MOUTH_LINE_FRONT_96 = [(48.0, 48.0), (36.0, 48.0)]
MOUTH_DEEP = (1, 1, 1)        # darkest palette colour (outline)
MOUTH_INNER = (128, 49, 61)   # existing dark crimson palette colour (cursed mouth interior)
MOUTH_PEAK = 0.55


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
        ((r == 1) & (g == 1) & (b == 1))
        | ((r == 50) & (g == 44) & (b == 51))
        | ((r == 82) & (g == 57) & (b == 0))
        | ((r == 128) & (g == 49) & (b == 61))
    )


def is_grey_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 77) & (g == 70) & (b == 79))
        | ((r == 104) & (g == 95) & (b == 107))
        | ((r == 124) & (g == 119) & (b == 125))
        | ((r == 145) & (g == 140) & (b == 146))
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
    Segments native 96x96 `front.png` into 8 anatomical masks:
      1. `head_tail_plume`: Ghostly tail plume extending from the back of the head (`y=15..42, x=50..69`).
      2. `arm_far`: Left (far-side) giant unzipped cursed claw arm (`y=49..75, x=13..39`).
      3. `leg_L`: Left unzipped doll hem & pink cursed leg tip (`y=71..78, x=36..44`).
      4. `leg_R`: Right pink cursed leg tip & golden zipper hem (`y=71..79, x=45..57`).
      5. `torso`: Central dark doll torso & unzipped pink curse core (`y=53..73, x=30..57`).
      6. `lower_jaw`: Lower row of golden zipper teeth, left zipper pull-tab, and chin (`y=49..52, x=32..54`).
      7. `head_upper`: Head zipper crest, glowing pink eyes, cranium, and upper zipper teeth (`y=22..48, x=30..58`).
      8. `arm_near`: Right (foreground) giant unzipped cursed claw arm (`y=48..75, x=50..81`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. `head_tail_plume` (back of head ghostly plume extending up-right)
    plume_zone = (
        ((yy >= 15) & (yy <= 27) & (xx >= 50))
        | ((yy == 28) & (xx >= 55))
        | ((yy == 29) & (xx >= 54))
        | ((yy >= 30) & (yy <= 31) & (xx >= 55))
        | ((yy >= 32) & (yy <= 33) & (xx >= 56))
        | ((yy >= 34) & (yy <= 37) & (xx >= 57))
        | ((yy >= 38) & (yy <= 40) & (xx >= 58))
        | ((yy >= 41) & (yy <= 42) & (xx >= 59))
    )
    head_tail_plume = a & plume_zone

    # 2. `lower_jaw` (lower golden zipper teeth + left pull-tab at x=32..35 + chin outline)
    jaw_zone = (
        ((yy == 49) & (xx >= 32) & (xx <= 54))
        | ((yy == 50) & (xx >= 32) & (xx <= 53))
        | ((yy == 51) & (xx >= 32) & (xx <= 51))
        | ((yy == 52) & (xx >= 33) & (xx <= 49))
    )
    lower_jaw = a & jaw_zone

    # 3. `head_upper` (head crest, glowing pink eyes, upper golden zipper teeth y=47..48)
    head_zone = (yy >= 22) & (yy <= 48) & (xx >= 30) & (xx <= 58)
    head_upper = a & head_zone & (~head_tail_plume) & (~lower_jaw)

    # 4. `arm_far` (left giant unzipped claw arm)
    arm_far_zone = (
        ((yy >= 49) & (yy <= 52) & (xx <= 28))
        | ((yy >= 53) & (yy <= 55) & (xx <= 36))
        | ((yy >= 56) & (yy <= 57) & (xx <= 39))
        | ((yy == 58) & (xx <= 37))
        | ((yy == 59) & (xx <= 35))
        | ((yy >= 60) & (yy <= 61) & (xx <= 34))
        | ((yy >= 62) & (yy <= 63) & (xx <= 29))
        | ((yy == 64) & (xx <= 30))
        | ((yy == 65) & (xx <= 31))
        | ((yy == 66) & (xx <= 32))
        | ((yy >= 67) & (yy <= 75) & (xx <= 31))
    )
    arm_far = a & arm_far_zone

    # 5. `arm_near` (right foreground giant unzipped claw arm)
    arm_near_zone = (
        ((yy >= 48) & (yy <= 52) & (xx >= 58))
        | ((yy == 53) & (xx >= 56))
        | ((yy == 54) & (xx >= 53))
        | ((yy >= 55) & (yy <= 57) & (xx >= 50))
        | ((yy == 58) & (xx >= 52))
        | ((yy == 59) & (xx >= 54))
        | ((yy == 60) & (xx >= 56))
        | ((yy >= 61) & (yy <= 64) & (xx >= 57))
        | ((yy >= 65) & (yy <= 66) & (xx >= 59))
        | ((yy >= 67) & (yy <= 68) & (xx >= 60))
        | ((yy >= 69) & (yy <= 75) & (xx >= 61))
    )
    arm_near = a & arm_near_zone

    # 6. `leg_L` and `leg_R` (lower doll skirt spikes & pink cursed leg tips)
    leg_L_zone = (
        ((yy >= 71) & (yy <= 72) & (xx >= 34) & (xx <= 43))
        | ((yy >= 73) & (yy <= 78) & (xx >= 34) & (xx <= 44))
    )
    leg_L = a & leg_L_zone & (~arm_far)

    leg_R_zone = (
        ((yy >= 71) & (yy <= 72) & (xx >= 45) & (xx <= 57))
        | ((yy >= 73) & (yy <= 79) & (xx >= 45) & (xx <= 57))
    )
    leg_R = a & leg_R_zone & (~arm_near)

    used = head_tail_plume | lower_jaw | head_upper | arm_far | arm_near | leg_L | leg_R
    torso = a & (~used)

    return {
        "head_tail_plume": head_tail_plume,
        "arm_far": arm_far,
        "leg_L": leg_L,
        "leg_R": leg_R,
        "torso": torso,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
        "arm_near": arm_near,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 8 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. `head_tail_plume` (long undulating ghost tail plume extending up-left from back of head)
    plume_zone = (
        ((yy >= 16) & (yy <= 20) & (xx <= 30))
        | ((yy == 21) & (xx <= 43))
        | ((yy == 22) & (xx <= 44))
        | ((yy >= 23) & (yy <= 26) & (xx <= 45))
        | ((yy >= 27) & (yy <= 28) & (xx <= 43))
        | ((yy == 29) & (xx <= 42))
        | ((yy == 30) & (xx <= 40))
        | ((yy == 31) & (xx <= 38))
        | ((yy >= 32) & (yy <= 33) & (xx <= 32))
        | ((yy == 34) & (xx <= 30))
    )
    head_tail_plume = a & plume_zone

    # 2. `lower_jaw` (golden zipper pull-tab peeking on the right side of the jaw in back view: y=49..52, x=56..60)
    jaw_zone = (
        ((yy == 49) & (xx >= 58) & (xx <= 60))
        | ((yy == 50) & (xx >= 56) & (xx <= 60))
        | ((yy == 51) & (xx >= 57) & (xx <= 60))
        | ((yy == 52) & (xx >= 58) & (xx <= 59))
    )
    lower_jaw = a & jaw_zone

    # 3. `head_upper` (back of cranium & ears: y=20..51, x=33..61)
    head_zone = (
        ((yy >= 20) & (yy <= 48) & (xx >= 33) & (xx <= 61))
        | ((yy == 49) & (xx >= 39) & (xx <= 57))
        | ((yy == 50) & (xx >= 41) & (xx <= 55))
        | ((yy == 51) & (xx >= 43) & (xx <= 52))
    )
    head_upper = a & head_zone & (~head_tail_plume) & (~lower_jaw)

    # 4. `arm_far` (left giant unzipped claw arm in back view)
    arm_far_zone = (
        ((yy >= 48) & (yy <= 54) & (xx <= 39))
        | ((yy >= 55) & (yy <= 56) & (xx <= 41))
        | ((yy == 57) & (xx <= 40))
        | ((yy == 58) & (xx <= 38))
        | ((yy >= 59) & (yy <= 60) & (xx <= 37))
        | ((yy >= 61) & (yy <= 62) & (xx <= 32))
        | ((yy == 63) & (xx <= 33))
        | ((yy == 64) & (xx <= 34))
        | ((yy == 65) & (xx <= 35))
        | ((yy >= 66) & (yy <= 74) & (xx <= 34))
    )
    arm_far = a & arm_far_zone

    # 5. `arm_near` (right giant unzipped claw arm in back view)
    arm_near_zone = (
        ((yy >= 47) & (yy <= 52) & (xx >= 60))
        | ((yy >= 53) & (yy <= 56) & (xx >= 53))
        | ((yy == 57) & (xx >= 55))
        | ((yy == 58) & (xx >= 57))
        | ((yy == 59) & (xx >= 59))
        | ((yy >= 60) & (yy <= 61) & (xx >= 60))
        | ((yy >= 62) & (yy <= 63) & (xx >= 61))
        | ((yy >= 64) & (yy <= 65) & (xx >= 62))
        | ((yy >= 66) & (yy <= 67) & (xx >= 63))
        | ((yy >= 68) & (yy <= 74) & (xx >= 64))
    )
    arm_near = a & arm_near_zone & (~lower_jaw)

    # 6. `leg_L` and `leg_R` (lower doll hem & pink cursed leg tips in back view)
    leg_L_zone = (
        ((yy >= 70) & (yy <= 71) & (xx >= 37) & (xx <= 46))
        | ((yy >= 72) & (yy <= 77) & (xx >= 37) & (xx <= 47))
    )
    leg_L = a & leg_L_zone & (~arm_far)

    leg_R_zone = (
        ((yy >= 70) & (yy <= 71) & (xx >= 48) & (xx <= 60))
        | ((yy >= 72) & (yy <= 78) & (xx >= 48) & (xx <= 60))
    )
    leg_R = a & leg_R_zone & (~arm_near)

    used = head_tail_plume | lower_jaw | head_upper | arm_far | arm_near | leg_L | leg_R
    torso = a & (~used)

    return {
        "head_tail_plume": head_tail_plume,
        "arm_far": arm_far,
        "leg_L": leg_L,
        "leg_R": leg_R,
        "torso": torso,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
        "arm_near": arm_near,
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
    places onto `(cell_h, cell_w)` canvas with feet aligned to `foot_y`, and builds cushioned
    RGBA layers + anatomical joint pivots.
    Note: Back-most layers (`torso`, etc.) are NEVER added to `foreground_mask` of layers in front of them.
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

    grey_fill_m = is_grey_body_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order (back to front):
        # 1. head_tail_plume -> 2. arm_far -> 3. leg_L -> 4. leg_R -> 5. torso -> 6. lower_jaw -> 7. head_upper -> 8. arm_near
        layers["head_tail_plume"] = build_part_layer(
            canvas,
            c_masks["head_tail_plume"],
            c_masks["head_upper"],
            cushion_px=5,
            allowed_fill_mask=grey_fill_m,
        )

        # `arm_far` cushioned only where `torso` overlaps in front of it
        layers["arm_far"] = build_part_layer(
            canvas,
            c_masks["arm_far"],
            c_masks["torso"],
            cushion_px=4,
            allowed_fill_mask=grey_fill_m,
        )

        layers["leg_L"] = build_foot_stem_layer(canvas, c_masks["leg_L"], c_masks["torso"], stem_px=4)
        layers["leg_R"] = build_foot_stem_layer(canvas, c_masks["leg_R"], c_masks["torso"], stem_px=4)

        # `torso` cushioned under `lower_jaw`, `head_upper`, and `arm_near`
        fg_over_torso = c_masks["lower_jaw"] | c_masks["head_upper"] | c_masks["arm_near"]
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=grey_fill_m,
        )
        layers["torso"] = torso_lyr

        # `lower_jaw` (drawn over `head_upper`'s mouth pocket)
        layers["lower_jaw"] = build_part_layer(
            canvas,
            c_masks["lower_jaw"],
            np.zeros_like(c_masks["lower_jaw"]),
            cushion_px=0,
            allowed_fill_mask=None,
        )

        # `head_upper` (rigid integer motion only; includes the cursed pink/crimson mouth interior pocket
        # directly behind the golden zipper teeth so opening `lower_jaw` by +1..+2px always reveals a
        # seamless, zero-bleed cursed mouth cavity that moves 100% locked with the head!)
        head_lyr = build_part_layer(
            canvas,
            c_masks["head_upper"],
            np.zeros_like(c_masks["head_upper"]),
            cushion_px=0,
        )
        # (2026-11-04) The rectangular pink mouth-interior pocket is no longer painted; see MOUTH_LINE_FRONT_96.
        layers["head_upper"] = head_lyr

        # `arm_near` (foreground giant unzipped claw arm, cushioned 3px into torso shoulder socket)
        layers["arm_near"] = build_part_layer(
            canvas,
            c_masks["arm_near"],
            c_masks["torso"],
            cushion_px=3,
            allowed_fill_mask=grey_fill_m,
        )

        pivots = {
            "head_tail_plume": to_canvas_pt(55.0, 34.0),
            "arm_far": to_canvas_pt(37.0, 56.0),
            "arm_near": to_canvas_pt(52.0, 56.0),
            "mouth_line": [to_canvas_pt(mx, my) for mx, my in MOUTH_LINE_FRONT_96],
        }

    else:
        # Back view draw order (back to front):
        # 1. arm_far -> 2. leg_L -> 3. leg_R -> 4. torso -> 5. lower_jaw -> 6. arm_near -> 7. head_upper -> 8. head_tail_plume
        layers["arm_far"] = build_part_layer(
            canvas,
            c_masks["arm_far"],
            c_masks["torso"],
            cushion_px=4,
            allowed_fill_mask=grey_fill_m,
        )

        layers["leg_L"] = build_foot_stem_layer(canvas, c_masks["leg_L"], c_masks["torso"], stem_px=4)
        layers["leg_R"] = build_foot_stem_layer(canvas, c_masks["leg_R"], c_masks["torso"], stem_px=4)

        fg_over_torso = c_masks["lower_jaw"] | c_masks["arm_near"] | c_masks["head_upper"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        layers["torso"] = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=5,
            allowed_fill_mask=grey_fill_m,
        )

        layers["lower_jaw"] = build_part_layer(
            canvas,
            c_masks["lower_jaw"],
            c_masks["head_upper"],
            cushion_px=2,
            allowed_fill_mask=None,
        )

        layers["arm_near"] = build_part_layer(
            canvas,
            c_masks["arm_near"],
            c_masks["head_upper"],
            cushion_px=2,
            allowed_fill_mask=grey_fill_m,
        )

        # In back view, `head_tail_plume` emerges from the upper-left of `head_upper`, so cushion `head_upper` under it
        layers["head_upper"] = build_part_layer(
            canvas,
            c_masks["head_upper"],
            c_masks["head_tail_plume"],
            cushion_px=5,
            allowed_fill_mask=grey_fill_m,
        )

        layers["head_tail_plume"] = build_part_layer(
            canvas,
            c_masks["head_tail_plume"],
            np.zeros_like(c_masks["head_tail_plume"]),
            cushion_px=0,
        )

        pivots = {
            "head_tail_plume": to_canvas_pt(41.0, 29.0),
            "arm_far": to_canvas_pt(40.0, 56.0),
            "arm_near": to_canvas_pt(55.0, 55.0),
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
            out[sy:sy + 2, sx:sx + 2] = (238, 189, 90, 255)
    return out


def synthesize_banette_mega_frame(
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
    jaw_open = 0   # back views: legacy zipper pull-tab jiggle (no visible mouth)
    mouth_amt = 0  # front views: small mouth opening via mouth_open
    plume_dx, plume_dy = 0, 0
    anear_dx, anear_dy = 0, 0
    afar_dx, afar_dy = 0, 0
    leg_L_dx, leg_L_dy = 0, 0
    leg_R_dx, leg_R_dy = 0, 0

    plume_ang = 0.0
    anear_ang = 0.0
    afar_ang = 0.0

    if action == "idle":
        # Eerie ghostly puppet levitation + rotating giant cursed claw arms + cackling/roaring golden zipper jaw!
        hover_dy = int(round(1.4 * math.sin(phase)))
        sway_dx = int(round(0.8 * math.cos(phase) * facing))

        torso_dx, torso_dy = sway_dx, hover_dy
        head_dx = sway_dx
        head_dy = hover_dy + int(round(0.5 * math.sin(phase - 0.4)))

        # Sinister golden zipper jaw cackle / roar in idle: opens +2px at peak frame 4, +1px on 2,3,5,6,11,12
        if frame_idx in (4,):
            jaw_open = 2
        elif frame_idx in (2, 3, 5, 6, 11, 12):
            jaw_open = 1
        else:
            jaw_open = 0
        mouth_amt = 1 if jaw_open else 0
        jaw_dx = head_dx
        jaw_dy = head_dy + jaw_open

        # Undulating ghost head-tail plume
        plume_dx, plume_dy = head_dx, head_dy
        plume_ang = 6.5 * math.sin(phase - 0.6) * facing

        # Both giant unzipped cursed claw arms slowly rotate/swing like a possessed marionette
        anear_dx = sway_dx
        anear_dy = hover_dy + int(round(1.0 * math.sin(phase - 0.3)))
        anear_ang = 7.0 * math.sin(phase - 0.3) * facing

        afar_dx = sway_dx
        afar_dy = hover_dy + int(round(-1.0 * math.sin(phase - 0.3)))
        afar_ang = -6.5 * math.sin(phase - 0.3) * facing

        # Floating puppet legs dangle with slight phase lag
        leg_L_dx = sway_dx
        leg_L_dy = int(round(1.0 * math.sin(phase - 0.5)))
        leg_R_dx = sway_dx
        leg_R_dy = int(round(1.0 * math.sin(phase - 0.8)))

    elif action == "walk":
        # Possessed marionette stalk with wide rotating claw-arm swings + barking golden zipper jaw!
        bounce = int(round(-1.5 * abs(sin1)))
        sway_x = int(round(1.6 * sin1 * facing))

        torso_dx, torso_dy = sway_x, bounce
        head_dx = sway_x + int(round(0.6 * cos1 * facing))
        head_dy = bounce + int(round(0.8 * math.sin(2.0 * phase - 0.3)))

        # Barking/laughing golden zipper jaw on each stalk beat
        jaw_open = 1 if math.sin(2.0 * phase) > 0.05 else 0
        mouth_amt = jaw_open
        jaw_dx = head_dx
        jaw_dy = head_dy + jaw_open

        plume_dx, plume_dy = head_dx, head_dy
        plume_ang = 8.5 * math.sin(phase - 0.5) * facing

        # Giant cursed claw arms swing & rotate boldly in counter-phase
        anear_dx = sway_x + int(round(1.2 * sin1 * facing))
        anear_dy = bounce + int(round(1.0 * cos1))
        anear_ang = 10.5 * sin1 * facing

        afar_dx = sway_x + int(round(-1.2 * sin1 * facing))
        afar_dy = bounce + int(round(-1.0 * cos1))
        afar_ang = -10.0 * sin1 * facing

        leg_L_dx, leg_L_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        leg_R_dx, leg_R_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Mega Banette "Cursed Zipper Roar + Spinning Giant Shadow Claw & Phantom Force Slash"!
        # Frames 0..3 (Wind-up & Wide Zipper Roar):
        #   - Coils back (`-3px`), rotates both giant cursed claw arms high/back (`-13°`),
        #   - Whips `head_tail_plume` back (`-11°`),
        #   - Opens golden zipper `lower_jaw` WIDE (`+2px` ferocious grudge roar exposing pink curse mouth!).
        # Frames 4..9 (Spinning Shadow Claw & Phantom Force Lunge):
        #   - Lunges forward (`+6px` toward target!) while dynamically spinning/whirling both giant
        #     unzipped claw arms (`+16° -> -14° -> +15° -> -11° -> +8°`) around their shoulder sockets!
        #   - Keeps `lower_jaw` wide open (`+2px` on frames 4..6, `+1px` on frames 7..8) in a fierce roar!
        # Frames 10..15: Follow-through recovery to floating stance.
        lunge_table = [0, -2, -3, -3, 3, 6, 6, 5, 4, 2, 0, -1, -1, 0, 0, 0]
        body_dy_table = [0, -1, -2, -1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]

        # Dynamic spinning claw-slash rotation tables (degrees)
        anear_spin_table = [0.0, -7.0, -12.0, -14.0, 15.0, 17.0, -14.0, 15.0, -11.0, 7.0, -3.0, -1.5, 0.0, 0.0, 0.0, 0.0]
        afar_spin_table = [0.0, 6.0, 11.0, 13.0, -14.0, -16.0, 13.0, -14.0, 10.0, -6.0, 2.5, 1.0, 0.0, 0.0, 0.0, 0.0]
        plume_spin_table = [0.0, -5.0, -9.0, -11.0, 11.0, 13.0, 9.0, -8.0, 6.0, 3.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0]

        body_dx = int(round(lunge_table[frame_idx] * facing))
        body_dy = body_dy_table[frame_idx]

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + (int(round(1.0 * facing)) if frame_idx in (4, 5, 6, 7) else 0)
        head_dy = body_dy

        # Ferocious +2px wide zipper roar during wind-up & slash peak (frames 1..6), +1px on 7..8
        if frame_idx in (1, 2, 3, 4, 5, 6):
            jaw_open = 2
        elif frame_idx in (7, 8):
            jaw_open = 1
        else:
            jaw_open = 0
        # Small face-following mouth: 2 px only at the roar peak (3..5), 1 px growl around it
        mouth_amt = 2 if frame_idx in (3, 4, 5) else (1 if jaw_open else 0)
        jaw_dx = head_dx
        jaw_dy = head_dy + jaw_open

        plume_dx, plume_dy = head_dx, head_dy
        plume_ang = plume_spin_table[frame_idx] * facing

        # Both giant claw arms thrust forward & whirl in a ferocious Shadow Claw tornado slash
        arm_thrust_x = int(round(1.5 * facing)) if frame_idx in (4, 5, 6, 7, 8) else 0
        anear_dx = body_dx + arm_thrust_x
        anear_dy = body_dy + (-1 if frame_idx in (2, 3, 6) else (1 if frame_idx in (4, 5, 7) else 0))
        anear_ang = anear_spin_table[frame_idx] * facing

        afar_dx = body_dx + arm_thrust_x
        afar_dy = body_dy + (1 if frame_idx in (2, 3, 6) else (-1 if frame_idx in (4, 5, 7) else 0))
        afar_ang = afar_spin_table[frame_idx] * facing

        leg_L_dx = int(round(body_dx * 0.85))
        leg_L_dy = body_dy
        leg_R_dx = int(round(body_dx * 0.75))
        leg_R_dy = body_dy

    elif action == "hurt":
        # Knockback recoil + hit-shake + zipper jaw flung open (`+2px`) + claw arms jolted back + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-3.5 * env * facing)) + jit
        body_dy = int(round(-1.2 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-1.6 * env))

        jaw_open = 2 if frame_idx in (2, 3, 4) else (1 if 1 <= frame_idx <= 7 else 0)
        mouth_amt = 1 if jaw_open else 0
        jaw_dx = head_dx
        jaw_dy = head_dy + jaw_open

        plume_dx, plume_dy = head_dx, head_dy
        plume_ang = -9.0 * env * facing

        anear_dx = body_dx + int(round(-1.0 * env * facing))
        anear_dy = int(round(-2.0 * env))
        anear_ang = -11.0 * env * facing

        afar_dx = body_dx + int(round(-1.0 * env * facing))
        afar_dy = int(round(-1.5 * env))
        afar_ang = 10.0 * env * facing

        slide_x = int(round(body_dx * 0.75))
        leg_L_dx, leg_L_dy = slide_x, int(round(-0.8 * env))
        leg_R_dx, leg_R_dy = slide_x, int(round(-0.8 * env))

    if not is_back:
        # Front views: the jaw stays locked to the head (no chin drop); the mouth is carved after compositing.
        jaw_dx, jaw_dy = head_dx, head_dy

    plume_rot = rotate_layer_nearest(layers["head_tail_plume"], plume_ang, pivots["head_tail_plume"])
    anear_rot = rotate_layer_nearest(layers["arm_near"], anear_ang, pivots["arm_near"])
    afar_rot = rotate_layer_nearest(layers["arm_far"], afar_ang, pivots["arm_far"])

    if not is_back:
        comp = [
            shift_layer_int(plume_rot, plume_dx, plume_dy),
            shift_layer_int(afar_rot, afar_dx, afar_dy),
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(anear_rot, anear_dx, anear_dy),
        ]
    else:
        comp = [
            shift_layer_int(afar_rot, afar_dx, afar_dy),
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(anear_rot, anear_dx, anear_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(plume_rot, plume_dx, plume_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    if not is_back and mouth_amt > 0 and "mouth_line" in pivots:
        line = mouth_open.transform_points(pivots["mouth_line"], dx=head_dx, dy=head_dy)
        out = mouth_open.open_mouth(out, line, amount=mouth_amt, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=MOUTH_PEAK)
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

    # Tight bounds of front (69x65) and back (75x63)
    max_w, max_h = 75, 65
    cell_w = 96
    cell_h = 96
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
                frame_rgba = synthesize_banette_mega_frame(
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

    # Preserve official 47-frame `front.gif` (and `front-reverse.gif`), and cleanly replace broken
    # 19-frame warped `back.gif` / `back-reverse.gif` with our zero-distortion 16-frame `idle` N loop!
    idle_sheet = sheets_by_action["idle"]
    back_frames = [idle_sheet[4 * cell_h:5 * cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    back_rev_frames = [f[:, ::-1, :].copy() for f in back_frames]

    save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
    save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)

    meta = {
        "generator": "forge_banette_mega_actions",
        "key": "0354_banette-mega",
        "dex": 354,
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
        for gif_name in ("back.gif", "back-reverse.gif"):
            shutil.copy2(MEGA_DIR / gif_name, SYNC_MEGA_DIR / gif_name)

    print(f"Built 0354_banette-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
