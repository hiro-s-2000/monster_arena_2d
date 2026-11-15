#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Gardevoir (`0282_gardevoir-mega`) modeled after Normal Gardevoir's official GIF
(`0282_gardevoir/normal/front.gif`) and Mega Gardevoir's ball-gown anatomical design (`front.png` & `back.png`):
  - Exact 9-layer anatomical rigging following the true 1px pixel-art pleat lines & joint pivots:
      1. `dress_back_skirt` (flowing bridal train, deep inner blue-silver skirt slit folds, and rear hem —
                             undulates in phase-delayed waltz waves while providing an interior under-skirt cushion)
      2. `dress_side_L` (left ball-gown wing pleat bounded by its exact 1px crease line — pivoted at the left
                         waist socket with `rotate_layer_nearest` for graceful waltz spins and Hyper Voice flare!)
      3. `dress_side_R` (right ball-gown wing pleat bounded by its exact 1px crease line — pivoted at the right
                         waist socket with `rotate_layer_nearest` for sweeping waltz turns!)
      4. `dress_center` (central front/back gown panel bounded by its exact left & right 1px pleat lines —
                         sways in counter-phase to create a true 3D spinning waltz silhouette)
      5. `arm_far` (far-side white opera-gloved arm — deeply cushioned into the shoulder socket and pivoted
                    for graceful conducting and rotational casting)
      6. `torso` (slender white neck, upper chest, far-side crimson crystal tip, and upper corset dome)
      7. `chest_plate` (twin opened crimson "Heart Crystal" chest plates — resonates and pulses forward during
                        Hyper Voice / Psychic invocation!)
      8. `head` / `head_roar1` / `head_roar2` (emerald bob hair, white 3-spike facial crest, and crimson eyes —
                                               moved ONLY via 100% rigid integer `shift_layer_int` so the eyes and
                                               face never distort by 1px, with a crisp 1-2px singing/roaring mouth
                                               opening during Hyper Voice & hurt!)
      9. `arm_near` (foreground white opera-gloved arm — pivoted at the near shoulder socket, drawing into the
                     heart crystal on wind-up and sweeping wide open during the spinning Hyper Voice release!)
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
MEGA_DIR = ROOT / "monsterSprite/gen3/0282_gardevoir/mega/gardevoir-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0282_gardevoir/mega/gardevoir-mega"
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
        | ((r == 49) & (g == 123) & (b == 65))
        | ((r == 51) & (g == 128) & (b == 70))
        | ((r == 131) & (g == 41) & (b == 74))
        | ((r == 128) & (g == 51) & (b == 64))
        | ((r == 133) & (g == 133) & (b == 166))
    )


def is_white_dress_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 230) & (g == 230) & (b == 242))
        | ((r == 195) & (g == 195) & (b == 217))
        | ((r == 205) & (g == 205) & (b == 230))
        | ((r == 207) & (g == 207) & (b == 229))
        | ((r == 238) & (g == 238) & (b == 255))
        | ((r == 242) & (g == 242) & (b == 255))
        | ((r == 164) & (g == 180) & (b == 205))
        | ((r == 163) & (g == 163) & (b == 191))
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
    Segments native 96x96 `front.png` into 9 exact anatomical masks aligned to the 1px pixel-art crease lines:
      1. `dress_back_skirt`:
           - Left bridal train (`y=57..80`, strictly left of the 1px crease `x_crease_L1(y)`),
           - Deep blue-silver inner skirt slit (`y=58..82`, between `x_slit_L(y)` and `x_slit_R(y)`),
           - Far-right shaded rear pleat (`y=61..82`, strictly right of `x=71`).
      2. `dress_side_L`: Left wing pleat (`y=49..82`), including its left 1px crease `x_crease_L1(y)`
                         so its outer contour stays 100% smooth and unbroken!
      3. `dress_side_R`: Right wing pleat (`y=49..83`), bounded by `x_slit_R(y)` on the left and `x=71` on the right.
      4. `dress_center`: Central bridal gown pleat (`y=47..84`), bounded by its exact left (`x_crease_L2(y)`)
                         and right (`x_slit_L(y)`) 1px crease lines.
      5. `arm_far`: Far-side left white opera-gloved arm (`x=26..53, y=34..46`).
      6. `torso`: Neck, upper chest, far-side red heart crystal tip, and upper corset dome (`y=31..49`).
      7. `chest_plate`: Foreground opened crimson Heart Crystal plate (`x=54..64, y=32..40`).
      8. `head`: Emerald bob hair, white 3-spike cheek crest, and crimson eyes (`x=44..68, y=10..31`).
      9. `arm_near`: Foreground right white opera-gloved arm (`x=63..80, y=36..48`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Head (emerald hair, white facial crest, crimson eyes)
    head_zone = (
        (yy <= 30)
        | ((yy == 31) & (xx >= 46) & (xx <= 55))
        | ((yy == 32) & (xx >= 48) & (xx <= 53))
    )
    head = a & head_zone

    # 2. Arm far (left opera-gloved arm)
    arm_far_zone = (
        ((yy >= 34) & (yy <= 36) & (xx <= 53))
        | ((yy == 37) & (xx <= 50))
        | ((yy == 38) & (xx <= 49))
        | ((yy == 39) & (xx <= 47))
        | ((yy >= 40) & (yy <= 46) & (xx <= 42))
    )
    arm_far = a & arm_far_zone & (~head)

    # 3. Arm near (right opera-gloved arm)
    arm_near_zone = (
        ((yy >= 36) & (yy <= 39) & (xx >= 64))
        | ((yy >= 40) & (yy <= 48) & (xx >= 63))
    )
    arm_near = a & arm_near_zone & (~head)

    # 4. Foreground crimson Heart Crystal chest plate (x=54..64, y=32..40)
    r = arr96[:, :, 0].astype(np.int32)
    is_red_crystal = (r == 213) | (r == 131)
    chest_plate_zone = (
        ((yy >= 32) & (yy <= 40) & (xx >= 54) & (xx <= 64) & is_red_crystal)
        | ((yy == 39) & (xx == 61))
        | ((yy == 40) & (xx >= 56) & (xx <= 60))
    )
    chest_plate = a & chest_plate_zone & (~head) & (~arm_near)

    # 5. Torso (neck, upper chest, and slender waist stem down to y=45)
    torso_zone = (
        ((yy >= 31) & (yy <= 42) & (xx >= 50) & (xx <= 63))
        | ((yy == 43) & (xx >= 51) & (xx <= 57))
        | ((yy == 44) & (xx >= 52) & (xx <= 57))
        | ((yy == 45) & (xx >= 53) & (xx <= 56))
    )
    torso = a & torso_zone & (~head) & (~arm_far) & (~arm_near) & (~chest_plate)

    # 6. Skirt region (all lower gown pixels)
    skirt_all = a & (~head) & (~arm_far) & (~arm_near) & (~chest_plate) & (~torso)

    # Exact 1px crease line lookup tables from pixel inspection of `front.png`:
    # Crease L1: separates left bridal train (`dress_back_skirt`) from `dress_side_L` for y=57..81
    crease_L1 = np.full(h, -1, dtype=np.int32)
    for y_val, x_val in [
        (57, 43), (58, 43), (59, 43), (60, 42), (61, 42), (62, 42), (63, 41), (64, 41),
        (65, 40), (66, 40), (67, 39), (68, 39), (69, 38), (70, 37), (71, 36), (72, 35),
        (73, 34), (74, 33), (75, 32), (76, 31), (77, 30), (78, 28), (79, 26), (80, 24), (81, 26),
    ]:
        crease_L1[y_val] = x_val

    # Crease L2: separates `dress_side_L` from `dress_center` for y=47..83
    crease_L2 = np.full(h, 46, dtype=np.int32)
    for y_val, x_val in [
        (47, 50), (48, 49), (49, 48), (50, 47), (51, 47), (52, 46), (53, 45), (54, 45),
        (55, 44), (56, 44), (57, 44), (58, 51), (59, 50), (60, 50), (61, 50), (62, 49),
        (63, 49), (64, 49), (65, 48), (66, 48), (67, 48), (68, 47), (69, 47), (70, 46),
        (71, 46), (72, 45), (73, 45), (74, 44), (75, 44), (76, 43), (77, 42), (78, 41),
        (79, 40), (80, 39), (81, 38), (82, 38), (83, 39), (84, 41),
    ]:
        crease_L2[y_val] = x_val

    # Slit L & Slit R: bound the central blue-silver inner skirt slit (`dress_back_skirt`)
    slit_L = np.full(h, 60, dtype=np.int32)
    slit_R = np.full(h, 60, dtype=np.int32)
    for y_val, (xl, xr) in [
        (54, (60, 61)), (55, (60, 61)), (56, (60, 61)), (57, (60, 61)),
        (58, (60, 61)), (59, (60, 61)), (60, (60, 61)), (61, (60, 62)),
        (62, (60, 62)), (63, (60, 62)), (64, (60, 62)), (65, (60, 62)),
        (66, (59, 62)), (67, (59, 61)), (68, (59, 61)), (69, (59, 61)),
        (70, (58, 61)), (71, (58, 61)), (72, (58, 60)), (73, (57, 60)),
        (74, (57, 60)), (75, (56, 60)), (76, (56, 59)), (77, (55, 59)),
        (78, (55, 59)), (79, (54, 58)), (80, (53, 58)), (81, (52, 57)),
        (82, (51, 56)), (83, (50, 57)), (84, (50, 57)),
    ]:
        slit_L[y_val] = xl
        slit_R[y_val] = xr

    c_L1_map = crease_L1[yy]
    c_L2_map = crease_L2[yy]
    s_L_map = slit_L[yy]
    s_R_map = slit_R[yy]

    # Left rear train (strictly left of Crease L1), central inner slit (strictly between Slit L and Slit R, plus bottom hem),
    # and far-right rear shadow pleat (`x >= 72` for `y >= 61`)
    back_train_mask = (yy >= 57) & (c_L1_map > 0) & (xx < c_L1_map)
    inner_slit_mask = (
        ((yy >= 61) & (yy <= 80) & (xx > s_L_map) & (xx < s_R_map))
        | ((yy >= 81) & (yy <= 82) & (xx >= 52) & (xx <= 56))
    )
    right_rear_mask = (
        ((yy >= 61) & (yy <= 78) & (xx >= 72))
        | ((yy == 79) & (xx >= 71))
        | ((yy >= 80) & (yy <= 82) & (xx >= 70))
    )
    dress_back_skirt = skirt_all & (back_train_mask | inner_slit_mask | right_rear_mask)

    # Central gown panel (includes its left 1px crease `c_L2_map` and right 1px slit outline `s_L_map`)
    center_mask = (xx >= c_L2_map) & (xx <= s_L_map)
    dress_center = skirt_all & (~dress_back_skirt) & center_mask

    # Left & right wing pleats (include their own 1px outer creases!)
    dress_side_L = skirt_all & (~dress_back_skirt) & (~dress_center) & (xx < c_L2_map)
    dress_side_R = skirt_all & (~dress_back_skirt) & (~dress_center) & (xx > s_L_map)

    return {
        "dress_back_skirt": dress_back_skirt,
        "dress_side_L": dress_side_L,
        "dress_side_R": dress_side_R,
        "dress_center": dress_center,
        "arm_far": arm_far,
        "torso": torso,
        "chest_plate": chest_plate,
        "head": head,
        "arm_near": arm_near,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 9 anatomical masks aligned to the 1px crease lines.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Head (emerald bob hair, white facial spikes, crimson side eye; y=9..30)
    head_zone = (
        (yy <= 29)
        | ((yy == 30) & (xx >= 32) & (xx <= 46))
        | ((yy == 31) & (xx >= 39) & (xx <= 44))
    )
    head = a & head_zone

    # 2. Arm far (right extended opera-gloved arm on back.png: x=43..68, y=32..44)
    arm_far_zone = (
        ((yy >= 32) & (yy <= 38) & (xx >= 44))
        | ((yy == 39) & (xx >= 45))
        | ((yy >= 40) & (yy <= 44) & (xx >= 47))
    )
    arm_far = a & arm_far_zone & (~head)

    # 3. Arm near (left extended opera-gloved arm on back.png: x=15..36, y=34..46)
    arm_near_zone = (
        ((yy >= 34) & (yy <= 37) & (xx <= 35))
        | ((yy == 38) & (xx <= 31))
        | ((yy >= 39) & (yy <= 46) & (xx <= 30))
    )
    arm_near = a & arm_near_zone & (~head)

    # 4. Dorsal crimson Heart Crystal plates on back.png (x=32..42, y=31..37)
    r = arr96[:, :, 0].astype(np.int32)
    is_red_crystal = (r == 217) | (r == 128) | (r == 255)
    chest_plate_zone = (yy >= 31) & (yy <= 37) & (xx >= 31) & (xx <= 43) & is_red_crystal
    chest_plate = a & chest_plate_zone & (~head) & (~arm_near) & (~arm_far)

    # 5. Torso (upper back and slender waist stem down to y=43)
    torso_zone = (
        ((yy >= 31) & (yy <= 40) & (xx >= 31) & (xx <= 45))
        | ((yy == 41) & (xx >= 35) & (xx <= 43))
        | ((yy == 42) & (xx >= 37) & (xx <= 42))
        | ((yy == 43) & (xx >= 38) & (xx <= 42))
    )
    torso = a & torso_zone & (~head) & (~arm_far) & (~arm_near) & (~chest_plate)

    # 6. Skirt region on back.png
    skirt_all = a & (~head) & (~arm_far) & (~arm_near) & (~chest_plate) & (~torso)

    # Exact 1px crease lines on `back.png`:
    # - Far-left blue-grey shadow fold (`x < crease_BL1(y)` for `y >= 54`)
    # - Left/Center crease `crease_BL2(y)` (`x=36..38` for `y=48..85`)
    # - Center inner slit (`x=38..43` for `y=66..84`)
    # - Center/Right crease `crease_BR1(y)` (`x=43..58` for `y=48..85`)
    # - Right/Train crease `crease_BR2(y)` (`x=52..72` for `y=56..83`, separating `dress_side_R` from the bright right train)
    crease_BL1 = np.full(h, -1, dtype=np.int32)
    for y_val in range(54, 86):
        if y_val <= 55:
            crease_BL1[y_val] = 28
        elif y_val <= 58:
            crease_BL1[y_val] = 27
        elif y_val <= 62:
            crease_BL1[y_val] = 26
        elif y_val <= 74:
            crease_BL1[y_val] = 25
        elif y_val <= 80:
            crease_BL1[y_val] = 26
        else:
            crease_BL1[y_val] = 27

    crease_BL2 = np.full(h, 36, dtype=np.int32)
    for y_val in range(48, 87):
        if y_val <= 74:
            crease_BL2[y_val] = 36
        elif y_val <= 80:
            crease_BL2[y_val] = 37
        else:
            crease_BL2[y_val] = 38

    crease_BR1 = np.full(h, 43, dtype=np.int32)
    for y_val, x_val in [
        (48, 43), (49, 43), (50, 44), (51, 44), (52, 44), (53, 44), (54, 44),
        (55, 45), (56, 45), (57, 45), (58, 45), (59, 45), (60, 46), (61, 46),
        (62, 46), (63, 46), (64, 47), (65, 47), (66, 47), (67, 48), (68, 48),
        (69, 48), (70, 49), (71, 49), (72, 50), (73, 50), (74, 51), (75, 51),
        (76, 52), (77, 52), (78, 53), (79, 54), (80, 55), (81, 56), (82, 57),
        (83, 58), (84, 58), (85, 57), (86, 57),
    ]:
        crease_BR1[y_val] = x_val

    crease_BR2 = np.full(h, 99, dtype=np.int32)
    for y_val, x_val in [
        (56, 52), (57, 52), (58, 52), (59, 53), (60, 53), (61, 53), (62, 54),
        (63, 54), (64, 55), (65, 55), (66, 56), (67, 56), (68, 57), (69, 57),
        (70, 58), (71, 59), (72, 60), (73, 61), (74, 62), (75, 63), (76, 64),
        (77, 65), (78, 66), (79, 68), (80, 70), (81, 71), (82, 72), (83, 72),
    ]:
        crease_BR2[y_val] = x_val

    c_BL1_map = crease_BL1[yy]
    c_BL2_map = crease_BL2[yy]
    c_BR1_map = crease_BR1[yy]
    c_BR2_map = crease_BR2[yy]

    is_inner_blue = (arr96[:, :, 0] == 163) & (arr96[:, :, 1] == 163) & (arr96[:, :, 2] == 191)
    back_skirt_zone = (
        ((yy >= 54) & (c_BL1_map > 0) & (xx < c_BL1_map))
        | ((yy >= 56) & (xx > c_BR2_map))
        | ((yy >= 66) & (xx > c_BL2_map) & (xx <= 43) & is_inner_blue)
    )
    dress_back_skirt = skirt_all & back_skirt_zone

    center_zone = (xx >= c_BL2_map) & (xx <= c_BR1_map)
    dress_center = skirt_all & (~dress_back_skirt) & center_zone

    dress_side_L = skirt_all & (~dress_back_skirt) & (~dress_center) & (xx < c_BL2_map)
    dress_side_R = skirt_all & (~dress_back_skirt) & (~dress_center) & (xx > c_BR1_map)

    return {
        "dress_back_skirt": dress_back_skirt,
        "dress_side_L": dress_side_L,
        "dress_side_R": dress_side_R,
        "dress_center": dress_center,
        "arm_far": arm_far,
        "torso": torso,
        "chest_plate": chest_plate,
        "head": head,
        "arm_near": arm_near,
    }


def make_roaring_head_variant(
    head_96: np.ndarray,
    is_back: bool,
    open_level: int = 1,
) -> np.ndarray:
    """
    Creates a 100% crisp, palette-locked Hyper Voice singing/roaring facial expression (`open_level = 1` or `2`)
    on the native 96x96 head layer BEFORE flipping/placing on the canvas:
      - Keeps the crimson eyes, emerald hair, and 3-spike cheek crest 100% untouched.
      - Opens a small, elegant crimson singing/roaring mouth (`(213, 74, 115)` / `(131, 41, 74)` on front,
        `(217, 87, 108)` / `(128, 51, 64)` on back) near the lower chin (`x=56..57, y=26..27` on front;
        `x=46, y=24..25` on back).
    """
    out = head_96.copy()
    if open_level <= 0:
        return out
    if not is_back:
        if open_level == 1:
            out[26, 56] = (131, 41, 74, 255)
            out[26, 57] = (213, 74, 115, 255)
        else:
            out[26, 56] = (131, 41, 74, 255)
            out[26, 57] = (213, 74, 115, 255)
            out[27, 56] = (213, 74, 115, 255)
            out[27, 57] = (131, 41, 74, 255)
            out[28, 56] = (230, 230, 242, 255)
    else:
        if open_level == 1:
            out[24, 46] = (128, 51, 64, 255)
        else:
            out[24, 46] = (217, 87, 108, 255)
            out[25, 46] = (128, 51, 64, 255)
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
    Segments native `front.png` or `back.png`, builds roaring/singing head variants,
    optionally flips horizontally for SE/E/NW, places onto the target `(cell_h, cell_w)` canvas
    with the dress hem aligned to `foot_y`, and constructs cushioned RGBA layers + joint pivots.
    """
    raw_masks = segment_back_native(base_96) if is_back else segment_front_native(base_96)

    head_base_96 = np.zeros_like(base_96)
    head_base_96[raw_masks["head"]] = base_96[raw_masks["head"]]
    head_roar1_96 = make_roaring_head_variant(head_base_96, is_back=is_back, open_level=1)
    head_roar2_96 = make_roaring_head_variant(head_base_96, is_back=is_back, open_level=2)

    src_96 = base_96[:, ::-1, :].copy() if flip_x else base_96.copy()
    head_r1_96 = head_roar1_96[:, ::-1, :].copy() if flip_x else head_roar1_96.copy()
    head_r2_96 = head_roar2_96[:, ::-1, :].copy() if flip_x else head_roar2_96.copy()
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

    white_fill_m = is_white_dress_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    # Erode the total gown silhouette by 2px so internal cushions NEVER bleed outside the 1px outer contour!
    all_skirt_mask = (
        c_masks["dress_back_skirt"]
        | c_masks["dress_side_L"]
        | c_masks["dress_side_R"]
        | c_masks["dress_center"]
        | c_masks["torso"]
    )
    safe_skirt_interior = ndi.binary_erosion(all_skirt_mask, iterations=2)

    # 1. `dress_back_skirt`: cushion strictly inside `safe_skirt_interior` under `dress_side_L`, `dress_side_R`, `dress_center`.
    #    Use nearest non-outline pixel from `dress_back_skirt` itself (`allowed_fill_mask=None`) so the blue-silver
    #    inner slit cushions with blue-silver while the white rear train cushions with white!
    fg_over_back_skirt = (c_masks["dress_side_L"] | c_masks["dress_side_R"] | c_masks["dress_center"]) & safe_skirt_interior
    layers["dress_back_skirt"] = build_part_layer(
        canvas,
        c_masks["dress_back_skirt"],
        fg_over_back_skirt,
        cushion_px=6,
        allowed_fill_mask=None,
    )

    # 2. `dress_side_L` & `dress_side_R`: cushion strictly inside `safe_skirt_interior` under `dress_center` and `torso`
    fg_over_side_L = ((c_masks["dress_center"] | c_masks["torso"]) & safe_skirt_interior) & (~c_masks["dress_back_skirt"])
    layers["dress_side_L"] = build_part_layer(
        canvas,
        c_masks["dress_side_L"],
        fg_over_side_L,
        cushion_px=6,
        allowed_fill_mask=white_fill_m,
    )

    # For `dress_side_R`, allow cushion into `dress_center` ONLY above the central slit (`y < dst_y0 + 48`),
    # never across the blue-silver slit!
    yy_c, _ = np.mgrid[0:cell_h, 0:cell_w]
    upper_dome_zone = yy_c < (dst_y0 + 48)
    fg_over_side_R = (
        ((c_masks["torso"] | (c_masks["dress_center"] & upper_dome_zone)) & safe_skirt_interior)
        & (~c_masks["dress_back_skirt"])
    )
    layers["dress_side_R"] = build_part_layer(
        canvas,
        c_masks["dress_side_R"],
        fg_over_side_R,
        cushion_px=6,
        allowed_fill_mask=white_fill_m,
    )

    # 3. `dress_center`: cushion strictly inside `safe_skirt_interior` under `torso`
    fg_over_center = c_masks["torso"] & safe_skirt_interior
    layers["dress_center"] = build_part_layer(
        canvas,
        c_masks["dress_center"],
        fg_over_center,
        cushion_px=4,
        allowed_fill_mask=white_fill_m,
    )

    # 4. `arm_far`: deep shoulder cushion under `torso`, `chest_plate`, and `head` so shoulder never gaps
    fg_over_afar = c_masks["torso"] | c_masks["chest_plate"] | c_masks["head"]
    u_afar = ndi.binary_erosion(c_masks["arm_far"] | fg_over_afar, iterations=1)
    layers["arm_far"] = build_part_layer(
        canvas,
        c_masks["arm_far"],
        u_afar & fg_over_afar,
        cushion_px=6,
        allowed_fill_mask=white_fill_m,
    )

    # 5. `torso`: IMPORTANT — `foreground_mask` ONLY includes foreground layers drawn ON TOP OF `torso`
    #    (`chest_plate`, `head`, `arm_near`), NEVER skirt layers or `arm_far`!
    fg_over_torso = c_masks["chest_plate"] | c_masks["head"] | c_masks["arm_near"]
    u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
    layers["torso"] = build_part_layer(
        canvas,
        c_masks["torso"],
        u_torso & fg_over_torso,
        cushion_px=5,
        allowed_fill_mask=white_fill_m,
    )

    # 6. `chest_plate`: crimson heart crystal plate
    fg_over_plate = c_masks["head"] | c_masks["arm_near"]
    u_plate = ndi.binary_erosion(c_masks["chest_plate"] | fg_over_plate, iterations=1)
    layers["chest_plate"] = build_part_layer(
        canvas,
        c_masks["chest_plate"],
        u_plate & fg_over_plate,
        cushion_px=2,
    )

    # 7. `head`, `head_roar1`, `head_roar2`: 100% rigid layers (zero cushion needed)
    layers["head"] = build_part_layer(canvas, c_masks["head"], np.zeros_like(c_masks["head"]), cushion_px=0)

    head_r1_canvas = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
    head_r1_canvas[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = head_r1_96[y0:y1, x0:x1]
    layers["head_roar1"] = head_r1_canvas

    head_r2_canvas = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
    head_r2_canvas[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = head_r2_96[y0:y1, x0:x1]
    layers["head_roar2"] = head_r2_canvas

    # 8. `arm_near`: foreground opera-gloved arm with 3px shoulder-socket cushion into `torso`
    fg_shoulder_near = c_masks["torso"] & (~is_dark_outline(canvas))
    layers["arm_near"] = build_part_layer(
        canvas,
        c_masks["arm_near"],
        fg_shoulder_near,
        cushion_px=3,
        allowed_fill_mask=white_fill_m,
    )

    if not is_back:
        pivots = {
            "dress_side_L": to_canvas_pt(54.0, 46.0),
            "dress_side_R": to_canvas_pt(55.0, 46.0),
            "arm_far": to_canvas_pt(51.0, 35.0),
            "arm_near": to_canvas_pt(64.0, 38.0),
        }
    else:
        pivots = {
            "dress_side_L": to_canvas_pt(39.0, 44.0),
            "dress_side_R": to_canvas_pt(40.0, 44.0),
            "arm_far": to_canvas_pt(44.0, 33.0),
            "arm_near": to_canvas_pt(34.0, 35.0),
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
        (mid_x - 15 - drift, max(2, top_y + 11 - drift // 2)),
        (mid_x + 15 + drift, max(2, top_y + 13 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (238, 238, 255, 255)
    return out


def synthesize_gardevoir_mega_frame(
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

    bskirt_dx, bskirt_dy = 0, 0
    sideL_dx, sideL_dy = 0, 0
    sideR_dx, sideR_dy = 0, 0
    center_dx, center_dy = 0, 0
    afar_dx, afar_dy = 0, 0
    torso_dx, torso_dy = 0, 0
    plate_dx, plate_dy = 0, 0
    head_dx, head_dy = 0, 0
    anear_dx, anear_dy = 0, 0

    sideL_ang = 0.0
    sideR_ang = 0.0
    afar_ang = 0.0
    anear_ang = 0.0
    roar_level = 0

    if action == "idle":
        # Graceful psychic levitation + waltz-spinning ball-gown pleats + resonant heart crystal + aria breathing:
        # 1. Whole body levitates in a serene psychic sine wave (`-2..+1px`).
        # 2. Left & right ball-gown wing pleats (`dress_side_L`, `dress_side_R`) rotate around the waist sockets
        #    (`±2.8°`) in a continuous waltz rhythm while `dress_center` and `dress_back_skirt` sway in counter-phase!
        # 3. Both opera-gloved arms (`arm_far`, `arm_near`) conduct the psychic melody (`±4.5°` shoulder rotation).
        # 4. At the peak of the breath/aria (`frames 2..5`), the mouth opens (`roar_level=1`) and the crimson
        #    Heart Crystal (`chest_plate`) resonates upward/forward!
        float_dy = int(round(1.6 * math.sin(phase - 0.3)))
        sway_dx = int(round(0.8 * cos1 * facing))

        torso_dx, torso_dy = sway_dx, float_dy
        head_dx = sway_dx
        head_dy = float_dy

        if frame_idx in (2, 3, 4, 5):
            roar_level = 1

        plate_dx = torso_dx + (int(round(0.8 * facing)) if frame_idx in (2, 3, 4, 5) else 0)
        plate_dy = torso_dy - (1 if frame_idx in (3, 4) else 0)

        # Waltz-spinning gown pleats
        sideL_dx, sideL_dy = sway_dx, float_dy
        sideR_dx, sideR_dy = sway_dx, float_dy
        sideL_ang = 2.6 * math.sin(phase - 0.5)
        sideR_ang = -2.6 * math.sin(phase - 0.5)

        center_dx = sway_dx + int(round(1.0 * math.sin(phase) * facing))
        center_dy = float_dy
        bskirt_dx = sway_dx + int(round(-1.0 * math.sin(phase - 0.8) * facing))
        bskirt_dy = float_dy + int(round(0.8 * math.cos(phase - 0.8)))

        # Conducting arms
        afar_dx = sway_dx
        afar_dy = float_dy + int(round(0.8 * math.sin(phase - 0.7)))
        afar_ang = 4.5 * math.sin(phase - 0.6) * facing

        anear_dx = sway_dx
        anear_dy = float_dy + int(round(0.8 * math.sin(phase - 0.9)))
        anear_ang = -4.5 * math.sin(phase - 0.8) * facing

    elif action == "walk":
        # Sweeping psychic waltz-glide & rotational dress billow:
        # 2-beat levitating waltz cadence where the ball-gown wings spin and flare (`±3.8°`)
        # as Mega Gardevoir glides forward, arms sweeping in an elegant waltz arc, singing softly on each beat.
        glide_dy = int(round(-1.8 * abs(sin1)))
        glide_dx = int(round(1.5 * sin1 * facing))

        torso_dx, torso_dy = glide_dx, glide_dy
        head_dx = glide_dx + int(round(0.5 * cos1 * facing))
        head_dy = glide_dy

        if math.sin(2.0 * phase) > 0.25:
            roar_level = 1

        plate_dx = torso_dx + int(round(0.6 * max(0.0, sin1) * facing))
        plate_dy = torso_dy

        sideL_dx = glide_dx
        sideL_dy = glide_dy
        sideR_dx = glide_dx
        sideR_dy = glide_dy
        sideL_ang = 3.8 * sin1
        sideR_ang = 3.5 * sin1

        center_dx = glide_dx + int(round(1.2 * cos1 * facing))
        center_dy = glide_dy
        bskirt_dx = glide_dx + int(round(-1.2 * cos1 * facing))
        bskirt_dy = glide_dy + int(round(1.0 * math.sin(2.0 * phase - 0.7)))

        afar_dx = glide_dx
        afar_dy = glide_dy + int(round(0.8 * cos1))
        afar_ang = 5.5 * sin1 * facing

        anear_dx = glide_dx
        anear_dy = glide_dy + int(round(-0.8 * cos1))
        anear_ang = 5.5 * sin1 * facing

    elif action == "attack":
        # Signature Mega Gardevoir "Grand Waltz Pirouette & Hyper Voice Roaring Aria / Moonblast Release":
        # Frames 0..3 (Wind-up & Psychic Gathering):
        #   - Draws back (`-2px`) and lowers slightly (`+1px`), folding both opera-gloved arms inward toward the
        #     crimson Heart Crystal (`+7.5°` / `-7.5°`) while furling the ball-gown wings inward (`-3.5°` / `+3.5°`).
        # Frames 4..9 (Spinning Waltz Flare & Hyper Voice Roaring Release!):
        #   - Surges forward (`+4px`) and upward (`-2px` levitation peak),
        #   - Mouth opens wide (`roar_level=2`) for a majestic Hyper Voice roar/aria!
        #   - Crimson Heart Crystal (`chest_plate`) pulses `+1px` forward & `-1px` up in psychic resonance!
        #   - Both opera-gloved arms fling wide open (`-10°` / `+10°` shoulder rotation + forward reach),
        #   - Ball-gown wings (`dress_side_L`, `dress_side_R`) execute a dramatic rotational waltz spin & flare
        #     (`+5.0°` / `-5.0°` with waltz oscillation across frames 4..9) while `dress_center` and
        #     `dress_back_skirt` whip in opposite directions to create a breathtaking 3D spinning gown effect!
        # Frames 10..15: Graceful waltz follow-through back to stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.60 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        body_dx = int(round(3.8 * k * facing))
        if frame_idx in (1, 2, 3):
            body_dy = 1
        elif frame_idx in (4, 5, 6, 7):
            body_dy = -2
        elif frame_idx in (8, 9):
            body_dy = -1
        else:
            body_dy = 0

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = int(round(4.0 * k * facing))
        head_dy = body_dy

        # Hyper Voice roaring/singing mouth expression
        if frame_idx in (4, 5, 6, 7):
            roar_level = 2
        elif frame_idx in (2, 3, 8, 9):
            roar_level = 1
        else:
            roar_level = 0

        # Resonant crimson Heart Crystal pulse
        plate_dx = torso_dx + (int(round(1.0 * facing)) if frame_idx in (4, 5, 6, 7, 8) else 0)
        plate_dy = torso_dy - (1 if frame_idx in (4, 5, 6, 7) else 0)

        # Spinning waltz ball-gown choreography (furl in on wind-up -> dramatic rotational flare & waltz spin on release)
        waltz_L_table = [0.0, -1.8, -2.8, -3.5, 3.5, 4.2, -3.2, 3.8, -2.8, 3.0, -1.8, 0.8, 0.0, 0.0, 0.0, 0.0]
        waltz_R_table = [0.0, 1.8, 2.8, 3.5, -3.5, -4.2, 3.2, -3.8, 2.8, -3.0, 1.8, -0.8, 0.0, 0.0, 0.0, 0.0]
        waltz_sway = [0, -1, -1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0]

        sideL_dx = body_dx + int(round(0.8 * waltz_sway[frame_idx] * facing))
        sideL_dy = body_dy
        sideR_dx = body_dx - int(round(0.8 * waltz_sway[frame_idx] * facing))
        sideR_dy = body_dy

        sideL_ang = waltz_L_table[frame_idx]
        sideR_ang = waltz_R_table[frame_idx]

        center_dx = body_dx + int(round(1.5 * waltz_sway[frame_idx] * facing))
        center_dy = body_dy
        bskirt_dx = body_dx - int(round(1.5 * waltz_sway[frame_idx] * facing))
        bskirt_dy = body_dy + (1 if frame_idx in (4, 6, 8) else (-1 if frame_idx in (5, 7) else 0))

        # Opera-gloved arms: draw in toward chest crystal (0..3) -> sweep wide open in spinning aria (4..9)
        arm_near_table = [0.0, 3.5, 6.5, 8.0, -7.5, -10.0, 6.5, -8.5, 5.0, -6.0, 2.5, 1.0, 0.0, 0.0, 0.0, 0.0]
        arm_far_table = [0.0, -3.5, -6.5, -8.0, 7.5, 10.0, -6.5, 8.5, -5.0, 6.0, -2.5, -1.0, 0.0, 0.0, 0.0, 0.0]

        anear_dx = body_dx
        anear_dy = body_dy + (-1 if frame_idx in (4, 5, 6, 7) else 0)
        anear_ang = arm_near_table[frame_idx] * (-facing)

        afar_dx = body_dx
        afar_dy = body_dy + (-1 if frame_idx in (4, 5, 6, 7) else 0)
        afar_ang = arm_far_table[frame_idx] * facing

    elif action == "hurt":
        # Knockback + hit-shake + pained cry (`roar_level=2`) + ball-gown upward billow + flung arms + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-3.5 * env * facing)) + jit
        body_dy = int(round(-1.2 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.6 * env * facing))
        head_dy = int(round(-1.4 * env))

        roar_level = 2 if frame_idx in (2, 3, 4, 5) else (1 if frame_idx in (1, 6, 7) else 0)

        plate_dx = torso_dx
        plate_dy = torso_dy - (1 if 2 <= frame_idx <= 5 else 0)

        # Ball-gown pleats billow outward from the impact shockwave
        sideL_dx = body_dx
        sideL_dy = body_dy
        sideR_dx = body_dx
        sideR_dy = body_dy
        sideL_ang = 4.0 * env
        sideR_ang = -4.0 * env

        center_dx = body_dx + int(round(0.8 * env * facing))
        center_dy = body_dy
        bskirt_dx = body_dx + int(round(-1.0 * env * facing))
        bskirt_dy = body_dy + int(round(0.8 * env))

        afar_dx = body_dx
        afar_dy = body_dy - int(round(1.0 * env))
        afar_ang = -6.5 * env * facing

        anear_dx = body_dx
        anear_dy = body_dy - int(round(1.0 * env))
        anear_ang = 6.5 * env * (-facing)

    sideL_rot = rotate_layer_nearest(layers["dress_side_L"], sideL_ang, pivots["dress_side_L"])
    sideR_rot = rotate_layer_nearest(layers["dress_side_R"], sideR_ang, pivots["dress_side_R"])
    afar_rot = rotate_layer_nearest(layers["arm_far"], afar_ang, pivots["arm_far"])
    anear_rot = rotate_layer_nearest(layers["arm_near"], anear_ang, pivots["arm_near"])

    if roar_level == 2:
        active_head = layers["head_roar2"]
    elif roar_level == 1:
        active_head = layers["head_roar1"]
    else:
        active_head = layers["head"]

    comp = [
        shift_layer_int(layers["dress_back_skirt"], bskirt_dx, bskirt_dy),
        shift_layer_int(sideL_rot, sideL_dx, sideL_dy),
        shift_layer_int(sideR_rot, sideR_dx, sideR_dy),
        shift_layer_int(layers["dress_center"], center_dx, center_dy),
        shift_layer_int(afar_rot, afar_dx, afar_dy),
        shift_layer_int(layers["torso"], torso_dx, torso_dy),
        shift_layer_int(layers["chest_plate"], plate_dx, plate_dy),
        shift_layer_int(active_head, head_dx, head_dy),
        shift_layer_int(anear_rot, anear_dx, anear_dy),
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

    # Tight bounds of front (67x75) and back (66x78)
    max_w, max_h = 67, 78
    cell_w = 104
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
                frame_rgba = synthesize_gardevoir_mega_frame(
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
        "generator": "forge_gardevoir_mega_actions",
        "key": "0282_gardevoir-mega",
        "dex": 282,
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

    print(f"Built 0282_gardevoir-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
