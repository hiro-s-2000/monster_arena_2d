#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Chimecho (`0358_chimecho-mega`) modeled after Normal Chimecho's official GIF
(`0358_chimecho/normal/front.gif`) and Mega Chimecho's multi-chime wind-bell anatomical design (`front.png` & `back.png`):
  - Exact 9-layer anatomical rigging with true suspension & ribbon joint pivots:
      1. `wing_ribbon_L` (upper-left giant red-and-white wing ribbon — pivoted at the left crown socket
                          with `rotate_layer_nearest` for sweeping wind-chime waves and Boomburst spiral spins!)
      2. `wing_ribbon_R` (upper-right giant red-and-white wing ribbon — pivoted at the right crown socket
                          with `rotate_layer_nearest` for counter-phase soaring and rotational burst flares!)
      3. `chime_inner_L` (left inner pair of blue & green mini wind-chimes and paper strips — pivoted at the
                          left inner ring loop for phase-delayed wind-chime oscillation and spiral spin!)
      4. `chime_inner_R` (right inner pair of green & blue mini wind-chimes and paper strips — pivoted at the
                          right inner ring loop for phase-delayed wind-chime oscillation and spiral spin!)
      5. `gold_frame` (ornate golden wind-bell outer cage, twin inner loops, and left/right golden arms —
                       pivoted at the upper golden suspension ring so the entire golden wind-chime cage tilts
                       and spins dynamically during resonant roaring & attack!)
      6. `crown_knob` (top golden suction-cup crown — anchors the upper wing ribbons and golden bell frame)
      7. `chime_outer_L` (far-left hanging red-zigzag mini chime & paper strip — pivoted at the left golden arm tip)
      8. `chime_outer_R` (far-right hanging red-zigzag mini chime & paper strip — pivoted at the right golden arm tip)
      9. `tail_main` (central Chimecho's long blue-to-crimson main tanzaku tail ribbon — cushioned at the neck
                      and behind the rear bridge on `back.png`, pivoted at the neck socket with `rotate_layer_nearest`
                      for graceful wind-chime waves and dramatic helical spins!)
     10. `head_bell` / `head_roar1` / `head_roar2` (central blue Chimecho bell head with yellow eyes and pink cheeks —
                      moved ONLY via 100% rigid integer `shift_layer_int` so the eyes and face never deform by 1px,
                      with an expressive roaring/singing mouth & lower jaw opening wide for Uproar / Boomburst!)
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
MEGA_DIR = ROOT / "monsterSprite/gen3/0358_chimecho/mega/chimecho-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0358_chimecho/mega/chimecho-mega"
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
        | ((r == 65) & (g == 74) & (b == 106))
        | ((r == 123) & (g == 41) & (b == 65))
    )


def build_part_layer(
    src: np.ndarray,
    mask: np.ndarray,
    foreground_mask: np.ndarray,
    cushion_px: int = 4,
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
    Segments native 96x96 `front.png` into 9 exact anatomical masks with zero stray pixels:
      1. `crown_knob`: Top golden suction-cup crown (`x=36..49, y=31..38`).
      2. `wing_ribbon_L`: Upper-left giant red-and-white wing ribbon (`x=6..38, y=15..37`).
      3. `wing_ribbon_R`: Upper-right giant red-and-white wing ribbon (`x=48..83, y=8..36`).
      4. `head_bell`: Central blue Chimecho bell head with yellow eyes & pink cheeks (`x=33..50, y=44..56`).
      5. `tail_main`: Central Chimecho's long blue-to-crimson tanzaku tail ribbon (`x=37..52, y=57..80`).
      6. `chime_outer_L`: Far-left hanging red-zigzag mini chime & paper strip (`x=9..18, y=61..79`).
      7. `chime_outer_R`: Far-right hanging red-zigzag mini chime & paper strip (`x=80..90, y=63..81`).
      8. `chime_inner_L`: Left inner pair of blue & green mini wind-chimes (`x=22..40, y=66..84`).
      9. `chime_inner_R`: Right inner pair of green & blue mini wind-chimes (`x=54..75, y=67..85`).
     10. `gold_frame`: Ornate golden wind-bell outer cage, inner double loops, and left/right golden arms.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    r, g, b = arr96[:, :, 0], arr96[:, :, 1], arr96[:, :, 2]

    is_gold = (
        ((r == 164) & (g == 115) & (b == 82))
        | ((r == 230) & (g == 164) & (b == 74))
        | ((r == 255) & (g == 213) & (b == 90))
        | ((r == 255) & (g == 230) & (b == 148))
    )
    is_blk = ((r == 0) & (g == 0) & (b == 0)) | ((r == 16) & (g == 16) & (b == 16))
    is_white = (
        ((r == 255) & (g == 255) & (b == 255))
        | ((r == 244) & (g == 252) & (b == 255))
        | ((r == 215) & (g == 215) & (b == 247))
    )

    # 1. Crown knob
    crown_knob = (
        a
        & (yy >= 31)
        & (yy <= 38)
        & (xx >= 36)
        & (xx <= 49)
        & (is_gold | is_white | ((yy <= 37) & is_blk & (xx >= 37) & (xx <= 48)))
    )

    # 2. Upper wing ribbons L & R
    wing_ribbon_L = (
        a
        & (~crown_knob)
        & (yy <= 37)
        & (xx <= 38)
        & (~((yy == 37) & (xx >= 34) & (is_gold | is_blk)))
    )
    wing_ribbon_R = a & (~crown_knob) & (yy <= 36) & (xx >= 48)

    # 3. Central Chimecho bell head
    head_zone = (
        ((yy == 44) & (xx >= 38) & (xx <= 41))
        | ((yy == 45) & (xx >= 36) & (xx <= 45))
        | ((yy >= 46) & (yy <= 50) & (xx >= 33) & (xx <= 50))
        | ((yy >= 51) & (yy <= 53) & (xx >= 33) & (xx <= 50))
        | ((yy == 54) & (xx >= 35) & (xx <= 49))
        | ((yy == 55) & (xx >= 37) & (xx <= 47))
        | ((yy == 56) & (xx >= 38) & (xx <= 46))
    )
    head_bell = a & head_zone

    # 4. Central Chimecho main tanzaku tail ribbon
    tail_zone = (
        ((yy >= 57) & (yy <= 63) & (xx >= 37) & (xx <= 45))
        | ((yy >= 64) & (yy <= 67) & (xx >= 38) & (xx <= 46))
        | ((yy >= 68) & (yy <= 80) & (xx >= 40) & (xx <= 52))
    )
    tail_main = a & tail_zone & (~head_bell)

    # 5. Outer hanging mini chimes L & R
    chime_outer_L = a & (yy >= 61) & (yy <= 79) & (xx <= 18)
    chime_outer_R = a & (yy >= 63) & (yy <= 81) & (xx >= 80)

    # 6. Inner hanging mini chime pairs L & R
    is_chime_col = (~is_gold) & (~((r == 0) & (g == 0) & (b == 0)))
    stud_L = ((yy == 67) & (xx >= 31) & (xx <= 33)) | ((yy == 66) & (xx >= 25) & (xx <= 26))
    stud_R = ((yy == 69) & (xx >= 61) & (xx <= 63)) | ((yy == 67) & (xx >= 69) & (xx <= 71))
    chime_inner_L = a & (~tail_main) & (yy >= 66) & (xx >= 22) & (xx <= 41) & (is_chime_col | stud_L)
    chime_inner_R = a & (~tail_main) & (yy >= 67) & (xx >= 54) & (xx <= 77) & (is_chime_col | stud_R)

    # 7. Golden wind-bell frame & arms
    assigned = (
        crown_knob
        | wing_ribbon_L
        | wing_ribbon_R
        | head_bell
        | tail_main
        | chime_outer_L
        | chime_outer_R
        | chime_inner_L
        | chime_inner_R
    )
    gold_frame = a & (~assigned)

    return {
        "crown_knob": crown_knob,
        "wing_ribbon_L": wing_ribbon_L,
        "wing_ribbon_R": wing_ribbon_R,
        "head_bell": head_bell,
        "tail_main": tail_main,
        "chime_outer_L": chime_outer_L,
        "chime_outer_R": chime_outer_R,
        "chime_inner_L": chime_inner_L,
        "chime_inner_R": chime_inner_R,
        "gold_frame": gold_frame,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 9 exact anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    r, g, b = arr96[:, :, 0], arr96[:, :, 1], arr96[:, :, 2]

    is_gold = (
        ((r == 164) & (g == 115) & (b == 82))
        | ((r == 230) & (g == 164) & (b == 74))
        | ((r == 255) & (g == 213) & (b == 90))
        | ((r == 255) & (g == 230) & (b == 148))
    )
    is_blk = ((r == 0) & (g == 0) & (b == 0)) | ((r == 16) & (g == 16) & (b == 16))
    is_white = (
        ((r == 255) & (g == 255) & (b == 255))
        | ((r == 244) & (g == 252) & (b == 255))
        | ((r == 215) & (g == 215) & (b == 247))
    )

    # 1. Crown knob on back.png
    crown_knob = (
        a
        & (yy >= 31)
        & (yy <= 38)
        & (xx >= 44)
        & (xx <= 57)
        & (is_gold | is_white | ((yy <= 37) & is_blk & (xx >= 45) & (xx <= 56)))
    )

    # 2. Upper wing ribbons L & R on back.png
    wing_ribbon_L = (
        a
        & (~crown_knob)
        & (yy <= 37)
        & (xx <= 46)
        & (~((yy == 37) & (xx >= 42) & (is_gold | is_blk)))
    )
    wing_ribbon_R = a & (~crown_knob) & (yy <= 36) & (xx >= 56)

    # 3. Central Chimecho bell head on back.png
    head_zone = (
        ((yy == 44) & (xx >= 51) & (xx <= 56))
        | ((yy == 45) & (xx >= 49) & (xx <= 58))
        | ((yy >= 46) & (yy <= 49) & (xx >= 46) & (xx <= 61))
        | ((yy == 50) & (xx >= 45) & (xx <= 61))
        | ((yy == 51) & (xx >= 45) & (xx <= 59))
        | ((yy == 52) & (xx >= 45) & (xx <= 58))
        | ((yy == 53) & (xx >= 45) & (xx <= 57))
        | ((yy == 54) & (xx >= 46) & (xx <= 56))
        | ((yy == 55) & (xx >= 47) & (xx <= 56))
        | ((yy == 56) & (xx >= 48) & (xx <= 55))
    )
    head_bell = a & head_zone

    # 4. Central Chimecho main tail ribbon on back.png (passes behind gold bridge at y=63..68)
    is_chime_col = (~is_gold) & (~((r == 0) & (g == 0) & (b == 0)))
    tail_zone = (
        ((yy >= 57) & (yy <= 63) & (xx >= 50) & (xx <= 56) & is_chime_col)
        | ((yy >= 68) & (yy <= 80) & (xx >= 43) & (xx <= 55))
    )
    tail_main = a & tail_zone & (~head_bell)

    # 5. Outer hanging mini chimes L & R on back.png
    chime_outer_L = a & (yy >= 63) & (yy <= 79) & (xx <= 15)
    chime_outer_R = a & (yy >= 61) & (yy <= 78) & (xx >= 77)

    # 6. Inner hanging mini chime pairs L & R on back.png
    stud_L = ((yy == 67) & (xx >= 22) & (xx <= 23)) | ((yy == 69) & (xx >= 32) & (xx <= 34))
    stud_R = ((yy == 67) & (xx >= 60) & (xx <= 62)) | ((yy == 66) & (xx >= 69) & (xx <= 70))
    chime_inner_L = a & (~tail_main) & (yy >= 67) & (xx >= 19) & (xx <= 40) & (is_chime_col | stud_L)
    chime_inner_R = a & (~tail_main) & (yy >= 66) & (xx >= 54) & (xx <= 73) & (is_chime_col | stud_R)

    assigned = (
        crown_knob
        | wing_ribbon_L
        | wing_ribbon_R
        | head_bell
        | tail_main
        | chime_outer_L
        | chime_outer_R
        | chime_inner_L
        | chime_inner_R
    )
    gold_frame = a & (~assigned)

    return {
        "crown_knob": crown_knob,
        "wing_ribbon_L": wing_ribbon_L,
        "wing_ribbon_R": wing_ribbon_R,
        "head_bell": head_bell,
        "tail_main": tail_main,
        "chime_outer_L": chime_outer_L,
        "chime_outer_R": chime_outer_R,
        "chime_inner_L": chime_inner_L,
        "chime_inner_R": chime_inner_R,
        "gold_frame": gold_frame,
    }


def make_roaring_head_variant(
    head_96: np.ndarray,
    is_back: bool,
    open_level: int = 1,
) -> np.ndarray:
    """
    Creates a 100% crisp, palette-locked sonic-roar / Uproar / Boomburst expression (`open_level = 1` or `2`)
    on the native 96x96 `head_bell` layer BEFORE flipping/placing on the canvas:
      - Keeps both yellow eyes (`y=46..49`) and pink cheeks 100% untouched.
      - On `front.png`:
          - `open_level = 1` (resonant chime singing / soft roar): opens a crisp 4x2 crimson mouth at `x=39..42, y=53..54`
            and shifts the lower chin rim down by `+1px` at `y=55..56`.
          - `open_level = 2` (full Boomburst / Uproar sonic roar!): opens a wide 6x3 resonant crimson/pink mouth
            at `x=38..43, y=52..54` with deep maroon interior `(123, 41, 65)` & bright coral tongue `(246, 82, 123)`,
            and shifts the lower jaw rim down to `y=55..57` for an unmistakable, powerful roaring bell face!
      - On `back.png`:
          - Expands the dorsal crimson zigzag acoustic resonance band and lowers the rear jaw rim by `+1px`.
    """
    out = head_96.copy()
    if open_level <= 0:
        return out

    DEEP_RED = (123, 41, 65, 255)
    MID_RED = (180, 65, 90, 255)
    BRIGHT_PINK = (246, 82, 123, 255)
    SOFT_PINK = (250, 152, 176, 255)
    LIGHT_BLUE = (197, 230, 255, 255)
    MID_BLUE = (156, 197, 230, 255)
    SHADE_BLUE = (106, 148, 180, 255)
    DARK_OUTLINE = (65, 74, 106, 255)
    BLACK_OUTLINE = (16, 16, 16, 255)

    if not is_back:
        if open_level == 1:
            # Crisp singing / barking chime mouth
            out[53, 39] = DEEP_RED
            out[53, 40] = MID_RED
            out[53, 41] = BRIGHT_PINK
            out[53, 42] = DEEP_RED
            out[54, 39] = DARK_OUTLINE
            out[54, 40] = BRIGHT_PINK
            out[54, 41] = MID_RED
            out[54, 42] = DARK_OUTLINE
            # Lower chin shifted +1px
            for x in range(39, 44):
                out[55, x] = MID_BLUE
                out[56, x] = DARK_OUTLINE
        else:
            # Wide Boomburst / Uproar roaring mouth!
            out[52, 39] = DEEP_RED
            out[52, 40] = DEEP_RED
            out[52, 41] = MID_RED
            out[52, 42] = DEEP_RED

            out[53, 38] = DEEP_RED
            out[53, 39] = MID_RED
            out[53, 40] = BRIGHT_PINK
            out[53, 41] = BRIGHT_PINK
            out[53, 42] = MID_RED
            out[53, 43] = DEEP_RED

            out[54, 38] = DARK_OUTLINE
            out[54, 39] = BRIGHT_PINK
            out[54, 40] = SOFT_PINK
            out[54, 41] = BRIGHT_PINK
            out[54, 42] = MID_RED
            out[54, 43] = DARK_OUTLINE

            out[55, 38] = SHADE_BLUE
            out[55, 39] = MID_BLUE
            out[55, 40] = LIGHT_BLUE
            out[55, 41] = MID_BLUE
            out[55, 42] = SHADE_BLUE
            out[55, 43] = DARK_OUTLINE

            out[56, 38] = DARK_OUTLINE
            out[56, 39] = DARK_OUTLINE
            out[56, 40] = BLACK_OUTLINE
            out[56, 41] = BLACK_OUTLINE
            out[56, 42] = DARK_OUTLINE
    else:
        # Dorsal acoustic resonance pulse on back head
        if open_level >= 1:
            for x in range(48, 56):
                if out[50, x, 3] >= 128:
                    out[50, x] = BRIGHT_PINK
        if open_level >= 2:
            for x in range(48, 55):
                out[55, x] = SHADE_BLUE
                out[56, x] = DARK_OUTLINE

    return out


def build_rig_layers(
    base_96: np.ndarray,
    is_back: bool,
    flip_x: bool,
    cell_w: int,
    cell_h: int,
    foot_y: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
    """
    Segments native `front.png` or `back.png`, builds roaring head variants (`head_roar1`, `head_roar2`),
    optionally flips horizontally for SE/E/NW, places onto the target `(cell_h, cell_w)` canvas,
    and constructs cushioned RGBA layers + anatomical joint pivots.
    """
    raw_masks = segment_back_native(base_96) if is_back else segment_front_native(base_96)

    head_base_96 = np.zeros_like(base_96)
    head_base_96[raw_masks["head_bell"]] = base_96[raw_masks["head_bell"]]
    head_roar1_96 = make_roaring_head_variant(head_base_96, is_back=is_back, open_level=1)
    head_roar2_96 = make_roaring_head_variant(head_base_96, is_back=is_back, open_level=2)

    # On back.png, `tail_main` passes behind the golden bridge at y=63..68 (x=48..54).
    # We pre-fill that hidden bridge span on a dedicated `tail_96` source copy so that when `tail_main`
    # rotates and swings behind the golden bridge, it never shows a gap!
    tail_base_96 = np.zeros_like(base_96)
    tail_base_96[raw_masks["tail_main"]] = base_96[raw_masks["tail_main"]]
    if is_back:
        for y_b in range(62, 69):
            x_left = 50 - (y_b - 62) // 2
            x_right = x_left + 5
            tail_base_96[y_b, x_left] = (65, 74, 106, 255)
            for x_m in range(x_left + 1, x_right):
                tail_base_96[y_b, x_m] = (156, 197, 230, 255)
            tail_base_96[y_b, x_right] = (65, 74, 106, 255)

    src_96 = base_96[:, ::-1, :].copy() if flip_x else base_96.copy()
    head_r1_96 = head_roar1_96[:, ::-1, :].copy() if flip_x else head_roar1_96.copy()
    head_r2_96 = head_roar2_96[:, ::-1, :].copy() if flip_x else head_roar2_96.copy()
    tail_src_96 = tail_base_96[:, ::-1, :].copy() if flip_x else tail_base_96.copy()
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

    layers: Dict[str, np.ndarray] = {}

    # 1. `wing_ribbon_L` & `wing_ribbon_R`: cushion into `crown_knob` and `gold_frame` (drawn in front of wing roots)
    fg_over_wings = c_masks["crown_knob"] | c_masks["gold_frame"]
    u_wing_L = ndi.binary_erosion(c_masks["wing_ribbon_L"] | fg_over_wings, iterations=1)
    layers["wing_ribbon_L"] = build_part_layer(
        canvas,
        c_masks["wing_ribbon_L"],
        u_wing_L & fg_over_wings,
        cushion_px=3,
    )
    u_wing_R = ndi.binary_erosion(c_masks["wing_ribbon_R"] | fg_over_wings, iterations=1)
    layers["wing_ribbon_R"] = build_part_layer(
        canvas,
        c_masks["wing_ribbon_R"],
        u_wing_R & fg_over_wings,
        cushion_px=3,
    )

    # 2. `chime_inner_L` & `chime_inner_R`: cushion into `gold_frame` (drawn over their suspension loops)
    fg_over_inner = c_masks["gold_frame"]
    u_in_L = ndi.binary_erosion(c_masks["chime_inner_L"] | fg_over_inner, iterations=1)
    layers["chime_inner_L"] = build_part_layer(
        canvas,
        c_masks["chime_inner_L"],
        u_in_L & fg_over_inner,
        cushion_px=2,
    )
    u_in_R = ndi.binary_erosion(c_masks["chime_inner_R"] | fg_over_inner, iterations=1)
    layers["chime_inner_R"] = build_part_layer(
        canvas,
        c_masks["chime_inner_R"],
        u_in_R & fg_over_inner,
        cushion_px=2,
    )

    # 3. `chime_outer_L` & `chime_outer_R`: cushion into `gold_frame` arm tips
    u_out_L = ndi.binary_erosion(c_masks["chime_outer_L"] | fg_over_inner, iterations=1)
    layers["chime_outer_L"] = build_part_layer(
        canvas,
        c_masks["chime_outer_L"],
        u_out_L & fg_over_inner,
        cushion_px=2,
    )
    u_out_R = ndi.binary_erosion(c_masks["chime_outer_R"] | fg_over_inner, iterations=1)
    layers["chime_outer_R"] = build_part_layer(
        canvas,
        c_masks["chime_outer_R"],
        u_out_R & fg_over_inner,
        cushion_px=2,
    )

    # 4. `gold_frame`: on `front.png`, `head_bell` and `tail_main` are in front of `gold_frame` (so `gold_frame`
    #    cushions under `crown_knob`, `head_bell`, `tail_main`). On `back.png`, `gold_frame` is in front of
    #    `head_bell` and `tail_main` (so `gold_frame` only cushions under `crown_knob`).
    if not is_back:
        fg_over_gold = c_masks["crown_knob"] | c_masks["head_bell"] | c_masks["tail_main"]
    else:
        fg_over_gold = c_masks["crown_knob"]
    u_gold = ndi.binary_erosion(c_masks["gold_frame"] | fg_over_gold, iterations=1)
    layers["gold_frame"] = build_part_layer(
        canvas,
        c_masks["gold_frame"],
        u_gold & fg_over_gold,
        cushion_px=3,
    )

    # 5. `crown_knob`: top golden suction-cup crown
    layers["crown_knob"] = build_part_layer(
        canvas,
        c_masks["crown_knob"],
        np.zeros_like(c_masks["crown_knob"]),
        cushion_px=0,
    )

    # 6. `tail_main`: central Chimecho long tanzaku tail ribbon (cushioned into `head_bell` at the neck)
    tail_canvas = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
    tail_canvas[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = tail_src_96[y0:y1, x0:x1]
    tail_mask_full = tail_canvas[:, :, 3] >= 128
    u_tail = ndi.binary_erosion(tail_mask_full | c_masks["head_bell"], iterations=1)
    # Build neck cushion into `head_bell` using `canvas`
    neck_cushioned = build_part_layer(
        canvas,
        c_masks["tail_main"],
        u_tail & c_masks["head_bell"],
        cushion_px=3,
    )
    m_tc = tail_canvas[:, :, 3] >= 128
    neck_cushioned[m_tc] = tail_canvas[m_tc]
    layers["tail_main"] = neck_cushioned

    # 7. `head_bell`, `head_roar1`, `head_roar2`: 100% rigid layers (zero cushion needed)
    layers["head_bell"] = build_part_layer(
        canvas,
        c_masks["head_bell"],
        np.zeros_like(c_masks["head_bell"]),
        cushion_px=0,
    )

    head_r1_canvas = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
    head_r1_canvas[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = head_r1_96[y0:y1, x0:x1]
    layers["head_roar1"] = head_r1_canvas

    head_r2_canvas = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
    head_r2_canvas[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = head_r2_96[y0:y1, x0:x1]
    layers["head_roar2"] = head_r2_canvas

    if not is_back:
        pivots = {
            "wing_ribbon_L": to_canvas_pt(37.0, 35.0),
            "wing_ribbon_R": to_canvas_pt(48.0, 34.0),
            "gold_frame": to_canvas_pt(43.0, 41.0),
            "tail_main": to_canvas_pt(41.0, 57.0),
            "chime_outer_L": to_canvas_pt(14.0, 61.0),
            "chime_outer_R": to_canvas_pt(85.0, 63.0),
            "chime_inner_L": to_canvas_pt(31.0, 67.0),
            "chime_inner_R": to_canvas_pt(65.0, 68.0),
        }
    else:
        pivots = {
            "wing_ribbon_L": to_canvas_pt(45.0, 35.0),
            "wing_ribbon_R": to_canvas_pt(56.0, 34.0),
            "gold_frame": to_canvas_pt(51.0, 41.0),
            "tail_main": to_canvas_pt(53.0, 57.0),
            "chime_outer_L": to_canvas_pt(11.0, 63.0),
            "chime_outer_R": to_canvas_pt(82.0, 61.0),
            "chime_inner_L": to_canvas_pt(28.0, 67.0),
            "chime_inner_R": to_canvas_pt(61.0, 67.0),
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
        (mid_x - 18 - drift, max(2, top_y + 12 - drift // 2)),
        (mid_x + 18 + drift, max(2, top_y + 14 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (244, 252, 255, 255)
    return out


def synthesize_chimecho_mega_frame(
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

    body_dx, body_dy = 0, 0
    head_dx, head_dy = 0, 0
    tail_dx, tail_dy = 0, 0

    wingL_ang = 0.0
    wingR_ang = 0.0
    gold_ang = 0.0
    tail_ang = 0.0
    outL_ang = 0.0
    outR_ang = 0.0
    inL_ang = 0.0
    inR_ang = 0.0
    roar_level = 0

    if action == "idle":
        # Aerial wind-bell resonance + phase-shifted ribbon waves + spiral wind-chime sway + sonic singing:
        # 1. Whole wind-chime body floats in a gentle aerial figure-8 wave (`dy = ±2px`, `dx = ±1px`).
        # 2. Golden bell cage (`gold_frame`) pendulums softly (`±3.2°`) around the top crown knob.
        # 3. Left & right giant wing ribbons (`wing_ribbon_L`, `wing_ribbon_R`) wave gracefully (`±5.5°`).
        # 4. Central tanzaku tail (`tail_main`) and the 4 mini-chime groups (`chime_outer_L/R`, `chime_inner_L/R`)
        #    rotate with phase delays (`±8.5°`, `±9.0°`, `±6.5°`) like a real wind-chime dancing in the breeze!
        # 5. At the resonant peak of the chime (`frames 2..6`), Mega Chimecho opens its mouth (`roar_level=1..2`)
        #    to ring out a harmonic sonic roar!
        body_dy = int(round(1.8 * math.sin(phase - 0.3)))
        body_dx = int(round(1.0 * cos1 * facing))

        head_dx = body_dx
        head_dy = body_dy
        tail_dx = body_dx
        tail_dy = body_dy

        if frame_idx in (3, 4):
            roar_level = 2
        elif frame_idx in (2, 5, 6):
            roar_level = 1

        gold_ang = 3.2 * math.sin(phase - 0.4) * facing
        wingL_ang = 5.5 * math.sin(phase - 0.6)
        wingR_ang = -5.5 * math.sin(phase - 0.6)

        tail_ang = 8.5 * math.sin(phase - 0.9) * facing
        outL_ang = 9.0 * math.sin(phase - 1.1) * facing
        outR_ang = 9.0 * math.sin(phase - 0.7) * facing
        inL_ang = 6.5 * math.sin(phase - 1.3) * facing
        inR_ang = 6.5 * math.sin(phase - 0.5) * facing

    elif action == "walk":
        # Dynamic aerial float-glide & large rotational wind-bell swing:
        # Mega Chimecho glides forward through the air with a 2-beat buoyant cadence while the golden bell cage,
        # giant wing ribbons, central tanzaku tail, and all 6 mini-chimes swing and spin in wide wind-swept arcs!
        glide_dy = int(round(-2.0 * abs(sin1)))
        glide_dx = int(round(2.2 * sin1 * facing))

        body_dx, body_dy = glide_dx, glide_dy
        head_dx = glide_dx + int(round(0.6 * cos1 * facing))
        head_dy = glide_dy
        tail_dx = head_dx
        tail_dy = glide_dy

        if math.sin(2.0 * phase) > 0.25:
            roar_level = 1

        gold_ang = 5.5 * sin1 * facing
        wingL_ang = 7.5 * sin1
        wingR_ang = -7.5 * sin1

        tail_ang = 12.5 * math.sin(phase - 0.7) * facing
        outL_ang = 13.0 * math.sin(phase - 0.9) * facing
        outR_ang = 13.0 * math.sin(phase - 0.5) * facing
        inL_ang = 9.5 * math.sin(phase - 1.0) * facing
        inR_ang = 9.5 * math.sin(phase - 0.4) * facing

    elif action == "attack":
        # Signature Mega Chimecho "Helical Wind-Bell Spin & Boomburst / Psychic Shockwave Roar":
        # Frames 0..3 (Rotational Wind-up & Sonic Charge):
        #   - Pulls back (`-3px`) and coils up (`-1px`), winding the golden bell cage, giant wing ribbons,
        #     central tanzaku tail, and 6 mini-chimes tightly into a reverse spiral while opening its mouth
        #     (`roar_level=1 -> 2`) to gather sonic resonance!
        # Frames 4..9 (Spinning Boomburst Roar & Forward Lunge `+6px`!):
        #   - Lunges forward (`+6px`) and surges upward (`-2px`),
        #   - Mouth opens wide (`roar_level=2`) in a ferocious Boomburst / Uproar sonic roar!
        #   - The golden bell cage (`gold_frame`), giant wing ribbons (`wing_ribbon_L/R`), central tanzaku tail
        #     (`tail_main`), and all 6 mini-chimes (`chime_outer_L/R`, `chime_inner_L/R`) whip into a high-speed
        #     helical spin oscillation (`±15°` on tail & outer chimes, `±10°` on wing ribbons, `±7°` on gold cage)
        #     creating a dazzling 3D spinning wind-bell vortex!
        # Frames 10..15: Resonant follow-through back to neutral hover.
        lunge_table = [0.0, -0.45, -0.85, -1.0, 0.75, 1.0, 0.95, 0.85, 0.65, 0.40, 0.15, -0.10, -0.05, 0.0, 0.0, 0.0]
        lift_table = [0, 1, 1, 1, -2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 0]

        k = lunge_table[frame_idx]
        body_dx = int(round(5.5 * k * facing))
        body_dy = lift_table[frame_idx]

        head_dx = int(round(6.0 * k * facing))
        head_dy = body_dy
        tail_dx = head_dx
        tail_dy = body_dy

        # Sonic Roar (Boomburst / Uproar) mouth expression
        if frame_idx in (3, 4, 5, 6, 7, 8):
            roar_level = 2
        elif frame_idx in (1, 2, 9, 10):
            roar_level = 1
        else:
            roar_level = 0

        # Helical spinning tables for the golden cage, wing ribbons, central tail, and 6 mini-chimes
        gold_spin = [0.0, -3.5, -6.0, -7.5, 7.5, -7.0, 7.5, -6.5, 6.0, -4.5, 2.5, -1.2, 0.5, 0.0, 0.0, 0.0]
        wingL_spin = [0.0, -5.0, -8.5, -10.5, 11.0, 12.5, -9.5, 11.5, -8.5, 9.0, -4.5, 2.0, -0.8, 0.0, 0.0, 0.0]
        wingR_spin = [0.0, 5.0, 8.5, 10.5, -11.0, -12.5, 9.5, -11.5, 8.5, -9.0, 4.5, -2.0, 0.8, 0.0, 0.0, 0.0]
        tail_spin = [0.0, -7.0, -12.0, -15.0, 16.0, -15.0, 15.5, -14.0, 12.5, -9.5, 5.5, -2.5, 1.0, 0.0, 0.0, 0.0]
        outL_spin = [0.0, -8.0, -13.0, -16.0, 17.0, -16.0, 16.0, -14.5, 13.0, -10.0, 6.0, -3.0, 1.0, 0.0, 0.0, 0.0]
        outR_spin = [0.0, -8.0, -13.0, -16.0, -17.0, 16.0, -16.0, 14.5, -13.0, 10.0, -6.0, 3.0, -1.0, 0.0, 0.0, 0.0]
        inL_spin = [0.0, -6.0, -10.0, -12.5, 13.5, -13.0, 13.0, -11.5, 10.0, -7.5, 4.5, -2.0, 0.8, 0.0, 0.0, 0.0]
        inR_spin = [0.0, -6.0, -10.0, -12.5, -13.5, 13.0, -13.0, 11.5, -10.0, 7.5, -4.5, 2.0, -0.8, 0.0, 0.0, 0.0]

        gold_ang = gold_spin[frame_idx] * facing
        wingL_ang = wingL_spin[frame_idx]
        wingR_ang = wingR_spin[frame_idx]
        tail_ang = tail_spin[frame_idx] * facing
        outL_ang = outL_spin[frame_idx] * facing
        outR_ang = outR_spin[frame_idx] * facing
        inL_ang = inL_spin[frame_idx] * facing
        inR_ang = inR_spin[frame_idx] * facing

    elif action == "hurt":
        # Knockback + hit-shake + pained sonic cry (`roar_level=2`) + upward-flung tanzaku ribbons + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-4.0 * env * facing)) + jit
        body_dy = int(round(-1.5 * env))

        head_dx = body_dx + int(round(-0.6 * env * facing))
        head_dy = body_dy
        tail_dx = head_dx
        tail_dy = body_dy

        roar_level = 2 if frame_idx in (2, 3, 4, 5) else (1 if frame_idx in (1, 6, 7) else 0)

        gold_ang = -5.5 * env * facing
        wingL_ang = -8.5 * env
        wingR_ang = 8.5 * env
        tail_ang = -14.0 * env * facing
        outL_ang = -14.5 * env * facing
        outR_ang = 14.5 * env * facing
        inL_ang = -11.0 * env * facing
        inR_ang = 11.0 * env * facing

    wingL_rot = rotate_layer_nearest(layers["wing_ribbon_L"], wingL_ang, pivots["wing_ribbon_L"])
    wingR_rot = rotate_layer_nearest(layers["wing_ribbon_R"], wingR_ang, pivots["wing_ribbon_R"])
    gold_rot = rotate_layer_nearest(layers["gold_frame"], gold_ang, pivots["gold_frame"])
    tail_rot = rotate_layer_nearest(layers["tail_main"], tail_ang, pivots["tail_main"])
    outL_rot = rotate_layer_nearest(layers["chime_outer_L"], outL_ang, pivots["chime_outer_L"])
    outR_rot = rotate_layer_nearest(layers["chime_outer_R"], outR_ang, pivots["chime_outer_R"])
    inL_rot = rotate_layer_nearest(layers["chime_inner_L"], inL_ang, pivots["chime_inner_L"])
    inR_rot = rotate_layer_nearest(layers["chime_inner_R"], inR_ang, pivots["chime_inner_R"])

    if roar_level == 2:
        active_head = layers["head_roar2"]
    elif roar_level == 1:
        active_head = layers["head_roar1"]
    else:
        active_head = layers["head_bell"]

    if not is_back:
        # On front view, `tail_main` and `head_bell` hang in front of the golden inner bridge
        comp = [
            shift_layer_int(wingL_rot, body_dx, body_dy),
            shift_layer_int(wingR_rot, body_dx, body_dy),
            shift_layer_int(inL_rot, body_dx, body_dy),
            shift_layer_int(inR_rot, body_dx, body_dy),
            shift_layer_int(gold_rot, body_dx, body_dy),
            shift_layer_int(layers["crown_knob"], body_dx, body_dy),
            shift_layer_int(outL_rot, body_dx, body_dy),
            shift_layer_int(outR_rot, body_dx, body_dy),
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(active_head, head_dx, head_dy),
        ]
    else:
        # On back view, `tail_main` and `head_bell` hang behind the golden rear bridge (`gold_frame`)
        comp = [
            shift_layer_int(wingL_rot, body_dx, body_dy),
            shift_layer_int(wingR_rot, body_dx, body_dy),
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(active_head, head_dx, head_dy),
            shift_layer_int(inL_rot, body_dx, body_dy),
            shift_layer_int(inR_rot, body_dx, body_dy),
            shift_layer_int(gold_rot, body_dx, body_dy),
            shift_layer_int(layers["crown_knob"], body_dx, body_dy),
            shift_layer_int(outL_rot, body_dx, body_dy),
            shift_layer_int(outR_rot, body_dx, body_dy),
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    front_raw = np.array(Image.open(MEGA_DIR / "front.png").convert("RGBA"))
    back_raw = np.array(Image.open(MEGA_DIR / "back.png").convert("RGBA"))

    palette_rgba = extract_palette(front_raw, back_raw)

    # Tight bounds of front (85x80) and back (88x80)
    max_w, max_h = 88, 80
    cell_w = 116
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
                frame_rgba = synthesize_chimecho_mega_frame(
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
        "generator": "forge_chimecho_mega_actions",
        "key": "0358_chimecho-mega",
        "dex": 358,
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

    print(f"Built 0358_chimecho-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
