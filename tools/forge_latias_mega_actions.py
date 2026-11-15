#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Latias (`0380_latias-mega` / メガラティアス) modeled after Mega Latias's
official 95-frame animated GIF (`front.gif`) and native 96x96 sprites (`front.png` & `back.png`):
  - Exact 7-layer anatomical rigging with true joint pivots (shoulder wing sockets & individual stabilizer roots):
      1. `tail_fin_L` (left rear stabilizer leg-fin — pivoted directly at its own root socket `rotate_layer_nearest`
                       to vector jet airflow with zero root displacement)
      2. `tail_fin_R` (right rear stabilizer leg-fin — pivoted directly at its own root socket `rotate_layer_nearest`
                       to bank and Rudder-stabilize during high-speed rolls)
      3. `jet_wing_far` (far-side giant lavender-and-white delta-jet wing & 3-clawed arm — pivoted at
                         its shoulder socket `rotate_layer_nearest` for banking & supersonic aileron rolls)
      4. `torso` (streamlined lavender-and-white jet fuselage + seamless dark-violet oral cavity pocket
                  behind `lower_jaw` so opening the mouth never leaves transparent gaps)
      5. `lower_jaw` (white lower jaw & chin — opens `+1..+2px` for Eon Flute / Mist Ball resonance
                      roar/bark in `idle`/`walk`, and opens wide **`+2px` for a thunderous Mist Ball &
                      Dragon Pulse roar** during `attack` wind-up and supersonic dive!)
      6. `head_upper` (white forehead triangle emblem, twin feathered ear-horns, glowing amber-yellow eye,
                       and upper snout — 100% rigid integer translation `shift_layer_int` ONLY so the face,
                       amber eye, forehead mark, and ear-horns never deform by even 1 pixel!)
      7. `jet_wing_near` (foreground colossal lavender-and-white delta-jet main wing & 3-clawed arm —
                          pivoted at its shoulder socket `rotate_layer_nearest` for dramatic banking,
                          supersonic corkscrew aileron rolls, and Mist Ball / Dragon Dive strikes!)
"""

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0380_latias/mega/latias-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0380_latias/mega/latias-mega"
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
        ((r == 16) & (g == 16) & (b == 16))
        | ((r == 72) & (g == 63) & (b == 115))
        | ((r == 82) & (g == 57) & (b == 119))
        | ((r == 82) & (g == 94) & (b == 97))
    )


def is_latias_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 104) & (g == 91) & (b == 166))
        | ((r == 120) & (g == 77) & (b == 185))
        | ((r == 125) & (g == 132) & (b == 134))
        | ((r == 135) & (g == 119) & (b == 217))
        | ((r == 152) & (g == 118) & (b == 198))
        | ((r == 160) & (g == 167) & (b == 169))
        | ((r == 173) & (g == 173) & (b == 217))
        | ((r == 206) & (g == 206) & (b == 242))
        | ((r == 211) & (g == 217) & (b == 220))
        | ((r == 242) & (g == 242) & (b == 255))
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
    a = out[..., 3] >= 128
    if np.any(a):
        # 1. Keep only the primary connected component so nearest-neighbor wing rotation never leaves a 1px stray speck
        lbl, n_comp = ndi.label(a)
        if n_comp > 1:
            sizes = ndi.sum(a, lbl, range(1, n_comp + 1))
            main_id = int(np.argmax(sizes)) + 1
            stray = a & (lbl != main_id)
            out[stray] = 0
            a = out[..., 3] >= 128
        # 2. Seal any 1..2px enclosed transparent pinholes caused by nearest-neighbor rotation rounding
        inv = ~a
        lbl_inv, _ = ndi.label(inv)
        bg_lbl = lbl_inv[0, 0]
        pinholes = (lbl_inv != bg_lbl) & inv
        if np.any(pinholes):
            fill_src = a & (~is_dark_outline(out))
            if not np.any(fill_src):
                fill_src = a
            _, (ny_idx, nx_idx) = ndi.distance_transform_edt(~fill_src, return_indices=True)
            py, px = np.nonzero(pinholes)
            out[py, px] = out[ny_idx[py, px], nx_idx[py, px]]
    return out


def segment_front_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `front.png` into 7 exact anatomical masks following 1px outlines:
      1. `tail_fin_L`: Left rear stabilizer leg-fin (`y=69..82, x=33..43`).
      2. `tail_fin_R`: Right rear stabilizer leg-fin (`y=68..81, x=50..66`).
      3. `jet_wing_far`: Far-side left lavender delta-jet wing & white arm socket (`x=2..29, y=46..67`).
      4. `torso`: Streamlined white/gray neck, chest, and purple lower jet fuselage (`y=38..70`).
      5. `lower_jaw`: White lower jaw & chin (`y=36..39, x=11..24`).
      6. `head_upper`: Purple upper snout, white forehead triangle mark, amber-yellow eye, and twin
                       feathered ear-horns (`y=13..38, x=11..40`).
      7. `jet_wing_near`: Colossal foreground right lavender delta-jet main wing (`x=46..92, y=43..65`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Far-side left delta-jet wing
    jet_wing_far = a & (
        ((yy == 46) & (xx <= 24))
        | ((yy == 47) & (xx <= 26))
        | ((yy >= 48) & (yy <= 54) & (xx <= 27))
        | ((yy >= 55) & (yy <= 59) & (xx <= 28))
        | ((yy >= 60) & (yy <= 64) & (xx <= 29))
        | ((yy >= 65) & (yy <= 67) & (xx <= 15))
    )

    # 2. Foreground right colossal delta-jet main wing (strictly outside the 1px black torso border at y=62..65)
    jet_wing_near = a & (
        ((yy == 43) & (xx >= 50))
        | ((yy == 44) & (xx >= 48))
        | ((yy == 45) & (xx >= 47))
        | ((yy >= 46) & (yy <= 54) & (xx >= 46))
        | ((yy == 55) & (xx >= 47))
        | ((yy >= 56) & (yy <= 57) & (xx >= 48))
        | ((yy == 58) & (xx >= 49))
        | ((yy == 59) & (xx >= 50))
        | ((yy >= 60) & (yy <= 61) & (xx >= 51))
        | ((yy == 62) & (xx >= 53))
        | ((yy == 63) & (xx >= 54))
        | ((yy == 64) & (xx >= 55))
        | ((yy == 65) & (xx >= 57))
    )

    # 3. Left & Right rear stabilizer leg-fins (individually segmented at their natural roots)
    tail_fin_L = a & (yy >= 69) & (xx <= 45) & (~jet_wing_far)
    tail_fin_R = a & (yy >= 68) & (xx >= 48) & (~jet_wing_near)

    # 4. White lower jaw & chin
    lower_jaw = a & (
        ((yy == 36) & (xx >= 11) & (xx <= 23))
        | ((yy == 37) & (xx >= 12) & (xx <= 22))
        | ((yy == 38) & (xx >= 13) & (xx <= 24))
        | ((yy == 39) & (xx >= 16) & (xx <= 22))
    )

    # 5. Upper head, forehead triangle, amber eye, and twin ear-horns
    head_upper = (
        a
        & (
            (yy <= 35)
            | ((yy == 36) & (xx >= 24) & (xx <= 40))
            | ((yy == 37) & (xx >= 23) & (xx <= 40))
            | ((yy == 38) & (xx >= 25) & (xx <= 33))
        )
        & (~lower_jaw)
    )

    used = jet_wing_far | jet_wing_near | tail_fin_L | tail_fin_R | lower_jaw | head_upper
    torso = a & (~used)

    return {
        "tail_fin_L": tail_fin_L,
        "tail_fin_R": tail_fin_R,
        "jet_wing_far": jet_wing_far,
        "torso": torso,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
        "jet_wing_near": jet_wing_near,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 7 anatomical masks following the true 1px outlines.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Far-side left delta-jet wing & white 3-clawed hand (`x=0..41, y=37..62`)
    jet_wing_far = a & (
        ((yy == 37) & (xx <= 41))
        | ((yy == 38) & (xx <= 40))
        | ((yy == 39) & (xx <= 39))
        | ((yy == 40) & (xx <= 38))
        | ((yy == 41) & (xx <= 37))
        | ((yy == 42) & (xx <= 36))
        | ((yy == 43) & (xx <= 35))
        | ((yy == 44) & (xx <= 34))
        | ((yy >= 45) & (yy <= 46) & (xx <= 33))
        | ((yy >= 47) & (yy <= 48) & (xx <= 32))
        | ((yy >= 49) & (yy <= 50) & (xx <= 31))
        | ((yy == 51) & (xx <= 30))
        | ((yy == 52) & (xx <= 29))
        | ((yy == 53) & (xx <= 28))
        | ((yy >= 54) & (yy <= 55) & (xx <= 27))
        | ((yy >= 56) & (yy <= 57) & (xx <= 26))
        | ((yy >= 58) & (yy <= 60) & (xx <= 25))
        | ((yy == 61) & (xx <= 24))
        | ((yy == 62) & (xx <= 19))
    )

    # 2. Foreground right colossal delta-jet wing & white 3-clawed hand (`x=39..95, y=38..66`)
    jet_wing_near = a & (
        ((yy == 38) & (xx >= 68))
        | ((yy == 39) & (xx >= 61))
        | ((yy == 40) & (xx >= 59))
        | ((yy >= 41) & (yy <= 42) & (xx >= 58))
        | ((yy == 43) & (xx >= 57))
        | ((yy == 44) & (xx >= 56))
        | ((yy >= 45) & (yy <= 46) & (xx >= 55))
        | ((yy == 47) & (xx >= 53))
        | ((yy == 48) & (xx >= 51))
        | ((yy >= 49) & (yy <= 50) & (xx >= 50))
        | ((yy == 51) & (xx >= 49))
        | ((yy >= 52) & (yy <= 53) & (xx >= 48))
        | ((yy >= 54) & (yy <= 55) & (xx >= 47))
        | ((yy == 56) & (xx >= 46))
        | ((yy == 57) & (xx >= 44))
        | ((yy == 58) & (xx >= 43))
        | ((yy >= 59) & (yy <= 60) & (xx >= 42))
        | ((yy == 61) & (xx >= 41))
        | ((yy == 62) & (xx >= 40))
        | ((yy >= 63) & (yy <= 64) & (xx >= 39))
        | ((yy == 65) & (xx >= 40) & (xx <= 48))
        | ((yy == 66) & (xx >= 41) & (xx <= 45))
    )

    # 3. Left & Right rear stabilizer leg-fins (`y=64..79`)
    tail_fin_L = a & (yy >= 64) & (xx <= 38) & (~jet_wing_far) & (~jet_wing_near)
    tail_fin_R = a & (yy >= 65) & (xx >= 42) & (~jet_wing_far) & (~jet_wing_near)

    # 4. Lower jaw tip peeking on 3/4 back view (`y=36..38, x>=72`)
    lower_jaw = (
        a
        & (yy >= 36)
        & (yy <= 38)
        & (xx >= 72)
        & (~jet_wing_near)
    )

    # 5. Back of head & twin feathered ear-horns (`y=16..38, x>=55`)
    head_upper = (
        a
        & (
            (yy <= 36)
            | ((yy == 37) & (xx >= 56))
            | ((yy == 38) & (xx >= 55))
        )
        & (~jet_wing_far)
        & (~jet_wing_near)
        & (~lower_jaw)
    )

    used = jet_wing_far | jet_wing_near | tail_fin_L | tail_fin_R | lower_jaw | head_upper
    torso = a & (~used)

    return {
        "tail_fin_L": tail_fin_L,
        "tail_fin_R": tail_fin_R,
        "jet_wing_far": jet_wing_far,
        "torso": torso,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
        "jet_wing_near": jet_wing_near,
    }


def build_tail_fin_with_stem(
    canvas: np.ndarray,
    fin_mask: np.ndarray,
    torso_mask: np.ndarray,
    default_rgb: Tuple[int, int, int],
) -> np.ndarray:
    lyr = build_part_layer(canvas, fin_mask, np.zeros_like(fin_mask), cushion_px=0)
    t_ys, t_xs = np.where(fin_mask)
    if len(t_ys) > 0:
        ty_min = int(t_ys.min())
        for px in np.unique(t_xs[t_ys <= ty_min + 3]):
            col_ys = t_ys[t_xs == px]
            cy0 = int(col_ys.min())
            if cy0 - 1 >= 0 and torso_mask[cy0 - 1, px]:
                stem_col = canvas[cy0 - 1, px].copy()
                if is_dark_outline(stem_col[None, None, :])[0, 0]:
                    stem_col = np.array([default_rgb[0], default_rgb[1], default_rgb[2], 255], dtype=np.uint8)
                for dy_up in range(1, 5):
                    if cy0 - dy_up >= 0 and torso_mask[cy0 - dy_up, px]:
                        lyr[cy0 - dy_up, px] = stem_col
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

    def to_canvas_y(ny: float) -> int:
        return int(round(float(dst_y0) + (ny - float(y0))))

    canvas = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
    canvas[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = src_96[y0:y1, x0:x1]

    c_masks: Dict[str, np.ndarray] = {}
    for k, m96 in masks_96.items():
        cm = np.zeros((cell_h, cell_w), dtype=bool)
        cm[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = m96[y0:y1, x0:x1]
        c_masks[k] = cm

    body_fill_m = is_latias_body_fill(canvas)
    cy_grid, _ = np.mgrid[0:cell_h, 0:cell_w]
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order (back-to-front):
        # 1. tail_fin_L -> 2. tail_fin_R -> 3. jet_wing_far -> 4. torso -> 5. lower_jaw -> 6. head_upper -> 7. jet_wing_near
        layers["tail_fin_L"] = build_tail_fin_with_stem(
            canvas, c_masks["tail_fin_L"], c_masks["torso"], (82, 57, 119)
        )
        layers["tail_fin_R"] = build_tail_fin_with_stem(
            canvas, c_masks["tail_fin_R"], c_masks["torso"], (120, 77, 185)
        )

        fg_over_wfar = ndi.binary_erosion(c_masks["torso"] | c_masks["head_upper"], iterations=1)
        layers["jet_wing_far"] = build_part_layer(
            canvas,
            c_masks["jet_wing_far"],
            fg_over_wfar,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        # IMPORTANT: `torso` cushion ONLY expands under foreground layers drawn ON TOP OF `torso`
        # (`lower_jaw`, `head_upper`, and the upper shoulder socket of `jet_wing_near` at `y <= 57`),
        # NEVER into the open air under the lower outer pod of `jet_wing_near` (`y >= 58`)!
        socket_y_max = to_canvas_y(57.0)
        wnear_socket = c_masks["jet_wing_near"] & (cy_grid <= socket_y_max)
        fg_over_torso = c_masks["lower_jaw"] | c_masks["head_upper"] | wnear_socket
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        # Paint deep violet/purple oral cavity pocket `(82, 57, 119)` / `(120, 77, 185)` bordered by `(16, 16, 16)`
        # behind `lower_jaw` AND add a 3px upper mouth stem onto `lower_jaw` so when `lower_jaw` drops `+1..+2px`
        # for a Mist Ball / Dragon Pulse roar, zero transparent gaps ever appear!
        jaw_ys, jaw_xs = np.where(c_masks["lower_jaw"])
        if len(jaw_ys) > 0:
            jy_min = int(jaw_ys.min())
            for py in range(jy_min, jy_min + 4):
                row_xs = jaw_xs[jaw_ys == py]
                if len(row_xs) > 0:
                    jx_min, jx_max = int(row_xs.min()), int(row_xs.max())
                    for px in range(jx_min, jx_max + 1):
                        if c_masks["lower_jaw"][py, px]:
                            if not flip_x:
                                if px <= jx_min + 1:
                                    torso_lyr[py, px] = (16, 16, 16, 255)
                                elif px >= jx_max - 2:
                                    torso_lyr[py, px] = (160, 167, 169, 255)
                                elif py == jy_min:
                                    torso_lyr[py, px] = (82, 57, 119, 255)
                                else:
                                    torso_lyr[py, px] = (120, 77, 185, 255)
                            else:
                                if px >= jx_max - 1:
                                    torso_lyr[py, px] = (16, 16, 16, 255)
                                elif px <= jx_min + 2:
                                    torso_lyr[py, px] = (160, 167, 169, 255)
                                elif py == jy_min:
                                    torso_lyr[py, px] = (82, 57, 119, 255)
                                else:
                                    torso_lyr[py, px] = (120, 77, 185, 255)
        layers["torso"] = torso_lyr

        jaw_lyr = build_part_layer(
            canvas, c_masks["lower_jaw"], np.zeros_like(c_masks["lower_jaw"]), cushion_px=0
        )
        if len(jaw_ys) > 0:
            jy_min = int(jaw_ys.min())
            top_xs = jaw_xs[jaw_ys == jy_min]
            if len(top_xs) > 0:
                tx_min, tx_max = int(top_xs.min()), int(top_xs.max())
                for dy_up in range(1, 3):
                    py = jy_min - dy_up
                    if py >= 0:
                        for px in range(tx_min, tx_max + 1):
                            if not flip_x:
                                if px <= tx_min + 1:
                                    jaw_lyr[py, px] = (16, 16, 16, 255)
                                elif px >= tx_max - 2:
                                    jaw_lyr[py, px] = (160, 167, 169, 255)
                                else:
                                    jaw_lyr[py, px] = (82, 57, 119, 255)
                            else:
                                if px >= tx_max - 1:
                                    jaw_lyr[py, px] = (16, 16, 16, 255)
                                elif px <= tx_min + 2:
                                    jaw_lyr[py, px] = (160, 167, 169, 255)
                                else:
                                    jaw_lyr[py, px] = (82, 57, 119, 255)
        layers["lower_jaw"] = jaw_lyr

        layers["head_upper"] = build_part_layer(
            canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0
        )
        layers["jet_wing_near"] = build_part_layer(
            canvas, c_masks["jet_wing_near"], np.zeros_like(c_masks["jet_wing_near"]), cushion_px=0
        )

        pivots = {
            "tail_fin_L": to_canvas_pt(36.0, 69.0),
            "tail_fin_R": to_canvas_pt(53.0, 68.0),
            "jet_wing_far": to_canvas_pt(28.0, 54.0),
            "jet_wing_near": to_canvas_pt(49.0, 53.0),
        }

    else:
        # Back view draw order (back-to-front):
        # 1. tail_fin_L -> 2. tail_fin_R -> 3. jet_wing_far -> 4. lower_jaw -> 5. torso -> 6. head_upper -> 7. jet_wing_near
        layers["tail_fin_L"] = build_tail_fin_with_stem(
            canvas, c_masks["tail_fin_L"], c_masks["torso"], (104, 91, 166)
        )
        layers["tail_fin_R"] = build_tail_fin_with_stem(
            canvas, c_masks["tail_fin_R"], c_masks["torso"], (104, 91, 166)
        )

        fg_over_wfar = ndi.binary_erosion(c_masks["torso"] | c_masks["head_upper"], iterations=1)
        layers["jet_wing_far"] = build_part_layer(
            canvas,
            c_masks["jet_wing_far"],
            fg_over_wfar,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_jaw = c_masks["head_upper"] | c_masks["jet_wing_near"]
        u_jaw = ndi.binary_erosion(c_masks["lower_jaw"] | fg_over_jaw, iterations=1)
        layers["lower_jaw"] = build_part_layer(
            canvas,
            c_masks["lower_jaw"],
            u_jaw & fg_over_jaw,
            cushion_px=3,
            allowed_fill_mask=body_fill_m,
        )

        # IMPORTANT: `torso` cushion ONLY expands under foreground layers drawn ON TOP OF `torso`
        # (`head_upper`, and the inner shoulder/body overlap of `jet_wing_near` at `y <= 60`),
        # NEVER over background layers (`tail_fin_L`, `tail_fin_R`, `jet_wing_far`)!
        socket_y_max = to_canvas_y(60.0)
        wnear_socket = c_masks["jet_wing_near"] & (cy_grid <= socket_y_max)
        fg_over_torso = c_masks["head_upper"] | wnear_socket
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        layers["torso"] = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_head = c_masks["jet_wing_near"]
        u_head = ndi.binary_erosion(c_masks["head_upper"] | fg_over_head, iterations=1)
        layers["head_upper"] = build_part_layer(
            canvas,
            c_masks["head_upper"],
            u_head & fg_over_head,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        layers["jet_wing_near"] = build_part_layer(
            canvas, c_masks["jet_wing_near"], np.zeros_like(c_masks["jet_wing_near"]), cushion_px=0
        )

        pivots = {
            "tail_fin_L": to_canvas_pt(34.0, 64.0),
            "tail_fin_R": to_canvas_pt(48.0, 65.0),
            "jet_wing_far": to_canvas_pt(33.0, 48.0),
            "jet_wing_near": to_canvas_pt(53.0, 49.0),
        }

    return layers, pivots


def add_mist_ball_roar_particles(
    canvas: np.ndarray,
    frame_idx: int,
    facing: float,
    is_back: bool,
    intensity: int = 1,
) -> np.ndarray:
    """
    Adds crisp palette-locked psychic Mist Ball / Dragon Pulse luster particles
    (`(250, 250, 250)`, `(242, 242, 255)`, `(152, 118, 198)`, `(255, 205, 90)`)
    radiating in the open air ahead of Mega Latias's snout during resonance roars and dives!
    """
    out = canvas.copy()
    a = out[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return out
    h, w = out.shape[:2]
    near_body = ndi.binary_dilation(a, iterations=2)
    mid_x = int(np.median(xs)) + int(round((20 if not is_back else 16) * facing))
    top_y = int(ys.min()) + (16 if not is_back else 12)
    drift = (frame_idx % 5) + 1

    offsets = [
        (int(round((10 + drift * 2) * facing)), -2 - drift),
        (int(round((14 + drift * 2) * facing)), 4 + drift),
    ]
    if intensity >= 2:
        offsets.append((int(round((8 + drift * 3) * facing)), 9 + drift))
        offsets.append((int(round((-22 - drift) * facing)), 4 - drift))

    for idx, (ox, oy) in enumerate(offsets):
        px = mid_x + ox
        py = top_y + oy
        if 2 <= px < w - 3 and 2 <= py < h - 3:
            if not np.any(near_body[py:py + 2, px:px + 2]):
                core_col = (250, 250, 250, 255) if idx % 2 == 0 else (255, 205, 90, 255)
                halo_col = (152, 118, 198, 255) if idx % 2 == 0 else (206, 206, 242, 255)
                out[py, px] = core_col
                out[py, px + 1] = halo_col
                out[py + 1, px] = halo_col
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
            out[sy:sy + 2, sx:sx + 2] = (255, 205, 90, 255)
            out[sy, sx] = (250, 250, 250, 255)
    return out


def synthesize_latias_mega_frame(
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
    wfar_dx, wfar_dy = 0, 0
    wnear_dx, wnear_dy = 0, 0
    tL_dx, tL_dy = 0, 0
    tR_dx, tR_dy = 0, 0

    wfar_ang = 0.0
    wnear_ang = 0.0
    tL_ang = 0.0
    tR_ang = 0.0
    roar_fx = 0

    if action == "idle":
        # Jet-Stream Figure-8 Hovering + Delta Wing Banking Roll + Mist Ball Resonance Roar/Bark:
        # 1. Streamlined Eon-jet body hovers in a smooth figure-8 thermal current (`-2..+2px` y, `±1px` x).
        # 2. Colossal lavender delta-jet wings (`jet_wing_near`, `jet_wing_far`) and twin rear stabilizer fins
        #    (`tail_fin_L`, `tail_fin_R`) bank and roll (`±4.8°`) with phase lag, matching the official 95-frame GIF!
        # 3. White `lower_jaw` opens `+1..+2px` at frames `3..7` (primary Eon Flute resonance roar) and
        #    `+1px` at frames `11..13` (secondary bark), emitting Mist Ball psychic sparkles!
        body_dy = int(round(2.0 * math.sin(phase - 0.25)))
        body_dx = int(round(1.0 * math.cos(phase - 0.25) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        if frame_idx in (4, 5, 6):
            jaw_open = 2
            roar_fx = 1
        elif frame_idx in (3, 7, 11, 12):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy + jaw_open

        wnear_dx = body_dx
        wnear_dy = body_dy + int(round(1.0 * math.sin(phase - 0.6)))
        wnear_ang = 4.8 * math.sin(phase - 0.6) * (-facing)

        wfar_dx = body_dx
        wfar_dy = body_dy + int(round(1.0 * math.sin(phase - 0.9)))
        wfar_ang = 4.8 * math.sin(phase - 0.9) * facing

        tL_dx, tL_dy = body_dx, body_dy
        tR_dx, tR_dy = body_dx, body_dy
        tL_ang = 4.5 * math.sin(phase - 1.0) * facing
        tR_ang = 4.5 * math.sin(phase - 1.3) * facing

    elif action == "walk":
        # Supersonic Jet-Glide Forward + Dynamic Aileron-Roll Wing Banking + Rhythmic Jet Roar:
        # 2-cycle high-speed jet surge (`2.0 * phase`), forward-leaning aerodynamic glide (`surge_x`, `surge_y`),
        # counter-rolling delta-jet main wings (`±7.0°`) and stabilizer tail fins (`±5.5°`),
        # plus rhythmic `+1..+2px` resonance roar on each jet thrust beat!
        surge_x = int(round(2.0 * sin1 * facing))
        surge_y = int(round(1.5 * math.sin(2.0 * phase - 0.3)))

        torso_dx, torso_dy = surge_x, surge_y
        head_dx = surge_x + int(round(1.0 * cos1 * facing))
        head_dy = surge_y

        s2 = math.sin(2.0 * phase)
        if s2 > 0.65:
            jaw_open = 2
            roar_fx = 1
        elif s2 > 0.15:
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy + jaw_open

        wnear_dx = surge_x + int(round(1.0 * cos1 * facing))
        wnear_dy = surge_y + int(round(1.2 * math.sin(2.0 * phase - 0.5)))
        wnear_ang = 7.0 * math.sin(2.0 * phase - 0.5) * (-facing)

        wfar_dx = surge_x + int(round(-1.0 * cos1 * facing))
        wfar_dy = surge_y + int(round(-1.2 * math.sin(2.0 * phase - 0.5)))
        wfar_ang = -7.0 * math.sin(2.0 * phase - 0.5) * facing

        tL_dx, tL_dy = surge_x, surge_y
        tR_dx, tR_dy = surge_x, surge_y
        tL_ang = 5.5 * math.sin(2.0 * phase - 0.8) * facing
        tR_ang = 5.5 * math.sin(2.0 * phase - 1.1) * facing

    elif action == "attack":
        # Signature Mega Latias "Mist Ball Roar + Supersonic Aileron-Roll Dragon Dive (`ミストボール` / `ドラゴンダイブ`)"!
        # Frames 0..3 (Wind-up & Thunderous Mist Ball Roar):
        #   - Pulls jet fuselage back (`-4px`) and climbs (`-2px`), banking the giant delta wings wide (`-9.5°`),
        #   - Opens `lower_jaw` WIDE (`+2px` full resonance roar!) while gathering psychic Mist Ball energy.
        # Frames 4..10 (Supersonic Corkscrew Aileron-Roll Dive & Mist Ball Barrage!):
        #   - Rockets forward (`+7px` Mach dive!) while executing a high-speed corkscrew aileron roll
        #     (`wnear ±12.0°`, `wfar ∓12.0°`, `tail_fin_L/R ±8.0°` with counter-rolling vertical wing sweep),
        #   - Roars fiercely (`+2px` open mouth at frames 4..6 & 9..10, snapping to `+1px` at 7..8)
        #     while unleashing psychic Mist Ball / Dragon Pulse shockwaves!
        # Frames 11..15: Air-brake follow-through recovery to hover stance.
        lunge_x_table = [0, -2, -3, -4, 3, 6, 7, 7, 5, 3, 2, 0, -1, -1, 0, 0]
        lunge_y_table = [0, -1, -2, -2, 0, 2, 2, 1, 0, -1, 0, 1, 0, 0, 0, 0]

        body_dx = int(round(lunge_x_table[frame_idx] * facing))
        body_dy = lunge_y_table[frame_idx]

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + (int(round(1.0 * facing)) if frame_idx in (5, 6, 7) else 0)
        head_dy = body_dy

        # Jaw curve: wide +2px roar during wind-up (1..3) and supersonic dive (4..6, 9..10)
        jaw_open_table = [0, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 0, 0, 0, 0]
        jaw_open = jaw_open_table[frame_idx]
        jaw_dx = head_dx
        jaw_dy = head_dy + jaw_open

        if frame_idx in (2, 3, 4, 5, 6, 9, 10):
            roar_fx = 2 if frame_idx in (3, 5, 6, 9) else 1

        # High-speed Corkscrew Aileron-Roll angles across both giant delta wings & twin rear tail fins!
        roll_wnear = [0.0, -4.5, -7.5, -9.5, 11.0, -12.0, 12.0, -11.0, 9.0, -6.5, 4.0, -1.5, 0.0, 0.0, 0.0, 0.0]
        roll_wfar = [0.0, 4.5, 7.5, 9.5, -11.0, 12.0, -12.0, 11.0, -9.0, 6.5, -4.0, 1.5, 0.0, 0.0, 0.0, 0.0]
        roll_tail = [0.0, -3.5, -5.5, -7.5, 7.5, -8.0, 8.0, -7.5, 6.0, -4.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        aileron_dy = [0, 0, -1, -1, 1, -2, 2, -2, 1, -1, 0, 0, 0, 0, 0, 0]

        wnear_dx = body_dx + (int(round(1.0 * facing)) if frame_idx in (5, 6, 7) else 0)
        wnear_dy = body_dy + aileron_dy[frame_idx]
        wnear_ang = roll_wnear[frame_idx] * (-facing)

        wfar_dx = body_dx
        wfar_dy = body_dy - aileron_dy[frame_idx]
        wfar_ang = roll_wfar[frame_idx] * facing

        tL_dx, tL_dy = body_dx, body_dy
        tR_dx, tR_dy = body_dx, body_dy
        tL_ang = roll_tail[frame_idx] * facing
        tR_ang = -roll_tail[frame_idx] * facing

    elif action == "hurt":
        # Air-turbulence knockback + hit-shake + pained Eon cry (`lower_jaw` open `+2px`) + wings flared back + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-4.0 * env * facing)) + jit
        body_dy = int(round(-1.2 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-1.0 * env * facing))
        head_dy = int(round(-1.5 * env))

        if frame_idx in (2, 3, 4, 5):
            jaw_open = 2
        elif frame_idx in (1, 6, 7):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy + jaw_open

        wnear_dx = body_dx + int(round(-1.0 * env * facing))
        wnear_dy = body_dy + int(round(-1.5 * env))
        wnear_ang = -8.0 * env * (-facing)

        wfar_dx = body_dx + int(round(-1.0 * env * facing))
        wfar_dy = body_dy + int(round(-1.5 * env))
        wfar_ang = 8.0 * env * facing

        tL_dx, tL_dy = body_dx, body_dy
        tR_dx, tR_dy = body_dx, body_dy
        tL_ang = -5.5 * env * facing
        tR_ang = -5.5 * env * facing

    tL_rot = rotate_layer_nearest(layers["tail_fin_L"], tL_ang, pivots["tail_fin_L"])
    tR_rot = rotate_layer_nearest(layers["tail_fin_R"], tR_ang, pivots["tail_fin_R"])
    wfar_rot = rotate_layer_nearest(layers["jet_wing_far"], wfar_ang, pivots["jet_wing_far"])
    wnear_rot = rotate_layer_nearest(layers["jet_wing_near"], wnear_ang, pivots["jet_wing_near"])

    if not is_back:
        comp = [
            shift_layer_int(tL_rot, tL_dx, tL_dy),
            shift_layer_int(tR_rot, tR_dx, tR_dy),
            shift_layer_int(wfar_rot, wfar_dx, wfar_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(wnear_rot, wnear_dx, wnear_dy),
        ]
    else:
        comp = [
            shift_layer_int(tL_rot, tL_dx, tL_dy),
            shift_layer_int(tR_rot, tR_dx, tR_dy),
            shift_layer_int(wfar_rot, wfar_dx, wfar_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(wnear_rot, wnear_dx, wnear_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    if roar_fx > 0:
        out = add_mist_ball_roar_particles(out, frame_idx, facing, is_back, intensity=roar_fx)
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

    # Tight bounds of native 96x96 front (91x70) and back (96x64)
    max_w, max_h = 96, 70
    cell_w = 120
    cell_h = 98
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
                frame_rgba = synthesize_latias_mega_frame(
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

    # Preserve official 95-frame `front.gif` / `front-reverse.gif`, and replace broken 20f `back.gif` / `back-reverse.gif`
    # with clean 16-frame `idle` N loop!
    idle_sheet = sheets_by_action["idle"]
    back_frames = [idle_sheet[4 * cell_h:5 * cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    back_rev_frames = [f[:, ::-1, :].copy() for f in back_frames]

    save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
    save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)

    meta = {
        "generator": "forge_latias_mega_actions",
        "key": "0380_latias-mega",
        "dex": 380,
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

    print(f"Built 0380_latias-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
