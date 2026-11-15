#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Altaria (`0334_altaria-mega`) modeled after Altaria's official 81-frame
sprite animation (`0334_altaria/normal/front.gif`) and Mega Altaria's anatomical design (`front.png` & `back.png`):
  - Exact 10-layer anatomical rigging with true joint & cloud-vortex pivots:
      1. `tail_ribbons` (long sky-blue 3-ribbon tail — pivoted at the tail root socket via
                         `rotate_layer_nearest` so the ribbons sway and whip gracefully in midair!)
      2. `cloud_mantle_back` (massive billowing cumulus cloud mantle behind the body — pivoted around
                              its dorsal vortex center via `rotate_layer_nearest` so the giant cloud
                              swirls and rotates like a living storm cloud!)
      3. `cloud_wing_L` (left billowing cotton-cloud wing — pivoted at the left shoulder socket via
                         `rotate_layer_nearest` for grand wing flaps and spinning Sky Attack rolls!)
      4. `foot_L` (left blue 3-toed talon foot — hovers and swings in sync with aerial flight)
      5. `neck_torso` (white fluffy neck-cloud collar, sky-blue neck & chest, and dark mauve/pink
                       singing/roaring mouth interior pocket behind `lower_beak`)
      6. `foot_R` (right blue 3-toed talon foot — phase-shifted aerial hover & stride)
      7. `cloud_wing_R` (foreground right billowing cotton-cloud wing — pivoted at the right chest/shoulder
                         socket via `rotate_layer_nearest` for dynamic counter-rotating wing flaps and
                         vortex spins over the dorsal cloud mantle!)
      8. `cloud_hat_crest` (fluffy cotton-cloud hat & twin white cheek/ear plumes — sways and pivots with
                            phase lag behind head movements)
      9. `lower_beak` (lower beak & chin — opens `+1px` for melodic singing/roaring in `idle`/`walk`
                       and `+2px` wide for a resonant Hyper Voice / Dragon Pulse roar during `attack`
                       wind-up, spinning lunge, and `hurt` reaction!)
     10. `head_upper` (sky-blue head crest, cranium, black eyes, white round cheeks, and white upper beak —
                       100% rigid integer motion `shift_layer_int` so the eyes and face never deform by a single pixel!)
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
MEGA_DIR = ROOT / "monsterSprite/gen3/0334_altaria/mega/altaria-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0334_altaria/mega/altaria-mega"
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
        | ((r == 75) & (g == 105) & (b == 115))
        | ((r == 115) & (g == 115) & (b == 156))
        | ((r == 164) & (g == 136) & (b == 159))
    )


def is_cloud_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 252) & (g == 252) & (b == 252))
        | ((r == 222) & (g == 230) & (b == 238))
        | ((r == 205) & (g == 205) & (b == 230))
        | ((r == 172) & (g == 180) & (b == 197))
    )


def is_blue_body_pixel(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 193) & (g == 232) & (b == 244))
        | ((r == 149) & (g == 209) & (b == 229))
        | ((r == 116) & (g == 163) & (b == 179))
        | ((r == 75) & (g == 105) & (b == 115))
        | ((r == 16) & (g == 16) & (b == 16))
    )


def is_blue_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 193) & (g == 232) & (b == 244))
        | ((r == 149) & (g == 209) & (b == 229))
        | ((r == 116) & (g == 163) & (b == 179))
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

    out_a = out[:, :, 3] >= 128
    if np.any(out_a):
        lbl, n_cc = ndi.label(out_a, structure=np.ones((3, 3)))
        if n_cc > 1:
            for c_i in range(1, n_cc + 1):
                cc_m = lbl == c_i
                if int(np.sum(cc_m)) <= 4:
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
    Segments native 96x96 `front.png` (facing left-front in aerial hovering song pose) into 10 exact anatomical masks:
      1. `tail_ribbons`: Long sky-blue 3-ribbon tail (`y=70..93, x=2..68`).
      2. `cloud_mantle_back`: Giant billowing dorsal cumulus cloud mantle (`y=2..72`).
      3. `cloud_wing_L`: Left billowing cotton-cloud wing (`y=17..56, x=4..35`).
      4. `foot_L`: Left blue 3-toed talon foot (`y=60..69, x=28..39`).
      5. `neck_torso`: Fluffy white neck-cloud collar + sky-blue neck & chest (`y=38..67, x=30..55`).
      6. `foot_R`: Right blue 3-toed talon foot (`y=62..71, x=45..53`).
      7. `cloud_wing_R`: Foreground right billowing cotton-cloud wing (`y=13..58, x=45..85`).
      8. `cloud_hat_crest`: Fluffy white cloud hat & right ear/cheek plume (`y=12..37, x=22..52`).
      9. `lower_beak`: Lower beak & chin outline (`y=41..42, x=29..36`).
     10. `head_upper`: Sky-blue head crest, cranium, black eyes, white cheeks, and upper beak (`y=22..40, x=27..39`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    blue_px = is_blue_body_pixel(arr96)

    # 1. `lower_beak` (lower beak & chin outline for singing/Hyper Voice roar)
    lower_beak = a & (
        ((yy == 41) & (xx >= 29) & (xx <= 36))
        | ((yy == 42) & (xx >= 30) & (xx <= 36))
    )

    # 2. `head_upper` (sky-blue head crest y=22..29, cranium & eyes y=30..37, white upper beak & round cheek y=38..40)
    head_zone = (
        ((yy == 22) & (xx >= 32) & (xx <= 33))
        | ((yy >= 23) & (yy <= 27) & (xx >= 31) & (xx <= 33))
        | ((yy == 28) & (xx >= 32) & (xx <= 33))
        | ((yy == 29) & (xx >= 30) & (xx <= 36))
        | ((yy == 30) & (xx >= 28) & (xx <= 36))
        | ((yy == 31) & (xx >= 28) & (xx <= 37))
        | ((yy == 32) & (xx >= 28) & (xx <= 38))
        | ((yy >= 33) & (yy <= 37) & (xx >= 27) & (xx <= 39))
        | ((yy == 38) & (xx >= 29) & (xx <= 39))
        | ((yy == 39) & (xx >= 28) & (xx <= 39))
        | ((yy == 40) & (xx >= 28) & (xx <= 38))
    )
    head_upper = a & head_zone & (~lower_beak)

    # 3. `cloud_hat_crest` (round cotton-cloud hat on top of head + white right ear/cheek plume)
    hat_zone = (
        ((yy == 12) & (xx >= 36) & (xx <= 38))
        | ((yy == 13) & (xx >= 34) & (xx <= 39))
        | ((yy == 14) & (xx >= 33) & (xx <= 39))
        | ((yy == 15) & (xx >= 32) & (xx <= 39))
        | ((yy == 16) & (xx >= 28) & (xx <= 39))
        | ((yy == 17) & (xx >= 26) & (xx <= 39))
        | ((yy == 18) & (xx >= 25) & (xx <= 39))
        | ((yy == 19) & (xx >= 24) & (xx <= 40))
        | ((yy == 20) & (xx >= 24) & (xx <= 42))
        | ((yy == 21) & (xx >= 24) & (xx <= 43))
        | ((yy >= 22) & (yy <= 23) & (xx >= 24) & (xx <= 43))
        | ((yy >= 24) & (yy <= 25) & (xx >= 25) & (xx <= 43))
        | ((yy == 26) & (xx >= 25) & (xx <= 44))
        | ((yy == 27) & (xx >= 24) & (xx <= 44))
        | ((yy == 28) & (xx >= 23) & (xx <= 44))
        | ((yy == 29) & (xx >= 23) & (xx <= 52))
        | ((yy == 30) & (xx >= 23) & (xx <= 51))
        | ((yy == 31) & (xx >= 23) & (xx <= 50))
        | ((yy == 32) & (xx >= 24) & (xx <= 49))
        | ((yy == 33) & (xx >= 24) & (xx <= 48))
        | ((yy == 34) & (xx >= 25) & (xx <= 47))
        | ((yy == 35) & (xx >= 40) & (xx <= 46))
        | ((yy == 36) & (xx >= 40) & (xx <= 44))
        | ((yy == 37) & (xx >= 40) & (xx <= 42))
    )
    cloud_hat_crest = a & hat_zone & (~head_upper) & (~lower_beak)

    # 4. `cloud_wing_L` (left billowing cotton-cloud wing y=17..56, x=4..35)
    wing_L_zone = (
        ((yy >= 17) & (yy <= 20) & (xx <= 23))
        | ((yy == 21) & (xx <= 22))
        | ((yy >= 22) & (yy <= 23) & (xx <= 23))
        | ((yy >= 24) & (yy <= 26) & (xx <= 24))
        | ((yy == 27) & (xx <= 23))
        | ((yy >= 28) & (yy <= 31) & (xx <= 22))
        | ((yy >= 32) & (yy <= 33) & (xx <= 23))
        | ((yy >= 34) & (yy <= 35) & (xx <= 24))
        | ((yy >= 36) & (yy <= 38) & (xx <= 25))
        | ((yy == 39) & (xx <= 26))
        | ((yy == 40) & (xx <= 27))
        | ((yy == 41) & (xx <= 28))
        | ((yy >= 42) & (yy <= 45) & (xx <= 29))
        | ((yy >= 46) & (yy <= 47) & (xx <= 30))
        | ((yy >= 48) & (yy <= 49) & (xx <= 31))
        | ((yy >= 50) & (yy <= 55) & (xx <= 34))
        | ((yy == 56) & (xx >= 25) & (xx <= 34))
    )
    cloud_wing_L = a & wing_L_zone & (~head_upper) & (~lower_beak) & (~cloud_hat_crest)

    # 5. `cloud_wing_R` (foreground right billowing cotton-cloud wing bounded by its internal 1px outline)
    wing_R_zone = (
        ((yy == 13) & (xx >= 70) & (xx <= 77))
        | ((yy == 14) & (xx >= 66) & (xx <= 79))
        | ((yy == 15) & (xx >= 63) & (xx <= 79))
        | ((yy == 16) & (xx >= 61) & (xx <= 74))
        | ((yy == 17) & (xx >= 61) & (xx <= 77))
        | ((yy == 18) & (xx >= 62) & (xx <= 81))
        | ((yy == 19) & (xx >= 61) & (xx <= 83))
        | ((yy == 20) & (xx >= 61) & (xx <= 84))
        | ((yy == 21) & (xx >= 58) & (xx <= 85))
        | ((yy == 22) & (xx >= 55) & (xx <= 85))
        | ((yy == 23) & (xx >= 54) & (xx <= 84))
        | ((yy == 24) & (xx >= 53) & (xx <= 83))
        | ((yy == 25) & (xx >= 53) & (xx <= 84))
        | ((yy >= 26) & (yy <= 27) & (xx >= 51) & (xx <= 85))
        | ((yy >= 28) & (yy <= 31) & (xx >= 51) & (xx <= 85))
        | ((yy == 32) & (xx >= 50) & (xx <= 84))
        | ((yy == 33) & (xx >= 52) & (xx <= 83))
        | ((yy == 34) & (xx >= 52) & (xx <= 84))
        | ((yy >= 35) & (yy <= 37) & (xx >= 52) & (xx <= 83))
        | ((yy == 38) & (xx >= 50) & (xx <= 83))
        | ((yy == 39) & (xx >= 50) & (xx <= 81))
        | ((yy == 40) & (xx >= 50) & (xx <= 81))
        | ((yy >= 41) & (yy <= 43) & (xx >= 49) & (xx <= 80))
        | ((yy == 44) & (xx >= 47) & (xx <= 81))
        | ((yy == 45) & (xx >= 46) & (xx <= 81))
        | ((yy == 46) & (xx >= 46) & (xx <= 80))
        | ((yy == 47) & (xx >= 45) & (xx <= 77))
        | ((yy == 48) & (xx >= 45) & (xx <= 74))
        | ((yy >= 49) & (yy <= 50) & (xx >= 45) & (xx <= 73))
        | ((yy == 51) & (xx >= 47) & (xx <= 72))
        | ((yy == 52) & (xx >= 47) & (xx <= 71))
        | ((yy == 53) & (xx >= 48) & (xx <= 70))
        | ((yy == 54) & (xx >= 48) & (xx <= 66))
        | ((yy == 55) & (xx >= 47) & (xx <= 64))
        | ((yy == 56) & (xx >= 48) & (xx <= 63))
        | ((yy == 57) & (xx >= 48) & (xx <= 62))
        | ((yy == 58) & (xx >= 50) & (xx <= 58))
    )
    cloud_wing_R = a & wing_R_zone & (~cloud_hat_crest)

    # 6. `foot_L` (left blue 3-toed talon foot y=60..69, x=28..39)
    foot_L_zone = (
        ((yy == 60) & (xx >= 30) & (xx <= 31))
        | ((yy == 61) & (xx >= 29) & (xx <= 34))
        | ((yy == 62) & (xx >= 29) & (xx <= 35))
        | ((yy == 63) & (xx >= 29) & (xx <= 34))
        | ((yy == 64) & (xx >= 29) & (xx <= 38))
        | ((yy == 65) & (xx >= 29) & (xx <= 39))
        | ((yy == 66) & (xx >= 30) & (xx <= 39))
        | ((yy == 67) & (xx >= 32) & (xx <= 34))
        | ((yy == 68) & (xx >= 33) & (xx <= 35))
        | ((yy == 69) & (xx == 34))
    )
    foot_L = a & foot_L_zone & (blue_px | ((yy == 66) & (xx == 31)))

    # 7. `foot_R` (right blue 3-toed talon foot y=62..71, x=45..53)
    foot_R_zone = (
        ((yy == 62) & (xx >= 47) & (xx <= 49))
        | ((yy == 63) & (xx >= 46) & (xx <= 50))
        | ((yy == 64) & (xx >= 45) & (xx <= 51))
        | ((yy == 65) & (xx >= 45) & (xx <= 52))
        | ((yy == 66) & (xx >= 45) & (xx <= 52))
        | ((yy == 67) & (xx >= 48) & (xx <= 53))
        | ((yy == 68) & (xx >= 48) & (xx <= 53))
        | ((yy == 69) & (xx >= 48) & (xx <= 51))
        | ((yy == 70) & (xx >= 48) & (xx <= 50))
        | ((yy == 71) & (xx >= 48) & (xx <= 49))
    )
    foot_R = a & foot_R_zone & blue_px

    # 8. `tail_ribbons` (long sky-blue 3-ribbon tail y=70..93)
    tail_ribbons = a & (yy >= 70) & blue_px & (~foot_R)

    # 9. `neck_torso` (white neck-cloud collar y=38..50 + sky-blue neck, chest, and belly y=45..67)
    neck_collar_zone = (
        ((yy == 38) & (xx >= 40) & (xx <= 42))
        | ((yy == 39) & (xx >= 40) & (xx <= 43))
        | ((yy == 40) & (xx >= 39) & (xx <= 43))
        | ((yy == 41) & (xx >= 37) & (xx <= 43))
        | ((yy == 42) & (xx >= 37) & (xx <= 44))
        | ((yy == 43) & (xx >= 30) & (xx <= 44))
        | ((yy == 44) & (xx >= 30) & (xx <= 44))
        | ((yy == 45) & (xx >= 30) & (xx <= 43))
        | ((yy == 46) & (xx >= 30) & (xx <= 43))
        | ((yy == 47) & (xx >= 31) & (xx <= 41))
        | ((yy == 48) & (xx >= 32) & (xx <= 39))
        | ((yy == 49) & (xx >= 32) & (xx <= 38))
        | ((yy == 50) & (xx >= 35) & (xx <= 37))
    )
    blue_torso_zone = (yy >= 44) & (yy <= 67) & (xx >= 34) & (xx <= 56) & blue_px
    neck_torso = (
        a
        & (neck_collar_zone | blue_torso_zone)
        & (~head_upper)
        & (~lower_beak)
        & (~cloud_hat_crest)
        & (~cloud_wing_L)
        & (~cloud_wing_R)
        & (~foot_L)
        & (~foot_R)
    )

    # 10. `cloud_mantle_back` (all remaining dorsal cumulus cloud mantle & lower cloud skirt)
    used = (
        lower_beak
        | head_upper
        | cloud_hat_crest
        | cloud_wing_L
        | cloud_wing_R
        | foot_L
        | foot_R
        | tail_ribbons
        | neck_torso
    )
    cloud_mantle_back = a & (~used)

    return {
        "tail_ribbons": tail_ribbons,
        "cloud_mantle_back": cloud_mantle_back,
        "cloud_wing_L": cloud_wing_L,
        "foot_L": foot_L,
        "neck_torso": neck_torso,
        "foot_R": foot_R,
        "cloud_wing_R": cloud_wing_R,
        "cloud_hat_crest": cloud_hat_crest,
        "lower_beak": lower_beak,
        "head_upper": head_upper,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` (3/4 rear aerial view) into matching 10 anatomical masks:
      1. `tail_ribbons`: Sky-blue 3-ribbon tail extending bottom-right (`y=72..89, x=63..86`).
      2. `foot_L`: Left blue foot (`y=75..81, x=31..38`).
      3. `foot_R`: Right blue foot (`y=78..84, x=61..73`).
      4. `cloud_hat_crest`: Upper cotton-cloud crown/hat lobe (`y=6..38, x=15..66`).
      5. `cloud_wing_L`: Left billowing cotton-cloud wing (`y=24..83, x=0..46`).
      6. `cloud_wing_R`: Right billowing cotton-cloud wing (`y=8..71, x=60..95`).
      7. `cloud_mantle_back`: Central foreground dorsal cumulus cloud lobe (`y=22..83, x=31..79`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    blue_px = is_blue_body_pixel(arr96)
    empty = np.zeros_like(a)

    # 1. `foot_L` (left blue foot at bottom-left y=75..81, x=31..38)
    foot_L = a & blue_px & (yy >= 75) & (yy <= 81) & (xx >= 31) & (xx <= 39)

    # 2. `foot_R` (right blue foot at bottom-center-right y=78..84, x=60..71)
    foot_R_zone = (
        ((yy == 78) & (xx >= 61) & (xx <= 66))
        | ((yy == 79) & (xx >= 60) & (xx <= 68))
        | ((yy == 80) & (xx >= 61) & (xx <= 70))
        | ((yy == 81) & (xx >= 61) & (xx <= 71))
        | ((yy == 82) & (xx >= 64) & (xx <= 71))
        | ((yy == 83) & (xx >= 66) & (xx <= 71))
        | ((yy == 84) & (xx >= 69) & (xx <= 71))
    )
    foot_R = a & blue_px & foot_R_zone

    # 3. `tail_ribbons` (sky-blue 3-ribbon tail at bottom-right y=72..89, x=63..86)
    tail_ribbons = a & blue_px & (yy >= 72) & (~foot_L) & (~foot_R)

    # 4. `cloud_mantle_back` (central prominent billowing dorsal cloud lobe bounded by its 1px outline)
    central_lobe_zone = (
        ((yy == 22) & (xx >= 50) & (xx <= 55))
        | ((yy == 23) & (xx >= 48) & (xx <= 62))
        | ((yy == 24) & (xx >= 46) & (xx <= 67))
        | ((yy == 25) & (xx >= 45) & (xx <= 71))
        | ((yy == 26) & (xx >= 41) & (xx <= 72))
        | ((yy == 27) & (xx >= 40) & (xx <= 72))
        | ((yy == 28) & (xx >= 39) & (xx <= 73))
        | ((yy == 29) & (xx >= 38) & (xx <= 73))
        | ((yy == 30) & (xx >= 37) & (xx <= 75))
        | ((yy == 31) & (xx >= 36) & (xx <= 75))
        | ((yy == 32) & (xx >= 36) & (xx <= 76))
        | ((yy == 33) & (xx >= 35) & (xx <= 77))
        | ((yy == 34) & (xx >= 35) & (xx <= 78))
        | ((yy >= 35) & (yy <= 38) & (xx >= 33) & (xx <= 78))
        | ((yy == 39) & (xx >= 32) & (xx <= 77))
        | ((yy == 40) & (xx >= 31) & (xx <= 78))
        | ((yy == 41) & (xx >= 31) & (xx <= 77))
        | ((yy >= 42) & (yy <= 43) & (xx >= 32) & (xx <= 76))
        | ((yy >= 44) & (yy <= 45) & (xx >= 33) & (xx <= 75))
        | ((yy == 46) & (xx >= 34) & (xx <= 74))
        | ((yy == 47) & (xx >= 35) & (xx <= 73))
        | ((yy >= 48) & (yy <= 50) & (xx >= 34) & (xx <= 73))
        | ((yy >= 51) & (yy <= 55) & (xx >= 33) & (xx <= 71))
        | ((yy == 56) & (xx >= 34) & (xx <= 70))
        | ((yy == 57) & (xx >= 34) & (xx <= 71))
        | ((yy == 58) & (xx >= 35) & (xx <= 71))
        | ((yy == 59) & (xx >= 36) & (xx <= 67))
        | ((yy == 60) & (xx >= 37) & (xx <= 66))
        | ((yy == 61) & (xx >= 38) & (xx <= 65))
        | ((yy == 62) & (xx >= 31) & (xx <= 66))
        | ((yy == 63) & (xx >= 31) & (xx <= 66))
        | ((yy == 64) & (xx >= 32) & (xx <= 66))
        | ((yy == 65) & (xx >= 32) & (xx <= 67))
        | ((yy == 66) & (xx >= 32) & (xx <= 66))
        | ((yy >= 67) & (yy <= 68) & (xx >= 32) & (xx <= 64))
        | ((yy == 69) & (xx >= 31) & (xx <= 64))
        | ((yy == 70) & (xx >= 31) & (xx <= 63))
        | ((yy == 71) & (xx >= 31) & (xx <= 63))
        | ((yy >= 72) & (yy <= 76) & (xx >= 34) & (xx <= 64))
        | ((yy >= 77) & (yy <= 80) & (xx >= 36) & (xx <= 60))
        | ((yy >= 81) & (yy <= 83) & (xx >= 39) & (xx <= 55))
    )
    cloud_mantle_back = a & central_lobe_zone & (~foot_L) & (~foot_R) & (~tail_ribbons)

    # 5. `cloud_hat_crest` (upper cotton-cloud crown/hat lobe y=6..38, x=15..66)
    hat_back_zone = (
        ((yy >= 6) & (yy <= 23) & (xx <= 65))
        | ((yy >= 24) & (yy <= 29) & (xx >= 16) & (xx <= 65))
        | ((yy >= 30) & (yy <= 34) & (xx >= 19) & (xx <= 45))
        | ((yy == 35) & (xx >= 20) & (xx <= 40))
        | ((yy == 36) & (xx >= 21) & (xx <= 38))
        | ((yy == 37) & (xx >= 22) & (xx <= 36))
        | ((yy == 38) & (xx >= 24) & (xx <= 33))
    )
    cloud_hat_crest = a & hat_back_zone & (~cloud_mantle_back)

    # 6. `cloud_wing_L` (left billowing cotton-cloud wing y=24..83, x=0..46)
    cloud_wing_L = (
        a
        & (xx <= 46)
        & (~cloud_mantle_back)
        & (~cloud_hat_crest)
        & (~foot_L)
        & (~foot_R)
        & (~tail_ribbons)
    )

    # 7. `cloud_wing_R` (right billowing cotton-cloud wing y=8..71, x=55..95)
    used = (
        foot_L
        | foot_R
        | tail_ribbons
        | cloud_mantle_back
        | cloud_hat_crest
        | cloud_wing_L
    )
    cloud_wing_R = a & (~used)

    return {
        "tail_ribbons": tail_ribbons,
        "cloud_mantle_back": cloud_mantle_back,
        "cloud_wing_L": cloud_wing_L,
        "foot_L": foot_L,
        "neck_torso": empty,
        "foot_R": foot_R,
        "cloud_wing_R": cloud_wing_R,
        "cloud_hat_crest": cloud_hat_crest,
        "lower_beak": empty,
        "head_upper": empty,
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
    and builds cushioned RGBA layers + true anatomical joint/vortex pivots.
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
    dst_y0 = (foot_y - 2 - sh) if is_back else (foot_y - sh)

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

    cloud_fill_m = is_cloud_fill(canvas)
    blue_fill_m = is_blue_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order (strictly Back-to-Front):
        # 1. tail_ribbons -> 2. cloud_mantle_back -> 3. cloud_wing_L -> 4. foot_L ->
        # 5. neck_torso -> 6. foot_R -> 7. cloud_wing_R -> 8. cloud_hat_crest ->
        # 9. lower_beak -> 10. head_upper

        fg_over_tail = c_masks["cloud_mantle_back"] | c_masks["foot_R"]
        u_tail = ndi.binary_erosion(c_masks["tail_ribbons"] | fg_over_tail, iterations=1)
        layers["tail_ribbons"] = build_part_layer(
            canvas,
            c_masks["tail_ribbons"],
            u_tail & fg_over_tail,
            cushion_px=8,
            allowed_fill_mask=blue_fill_m,
        )

        fg_over_mantle = (
            c_masks["cloud_wing_L"]
            | c_masks["foot_L"]
            | c_masks["neck_torso"]
            | c_masks["foot_R"]
            | c_masks["cloud_wing_R"]
            | c_masks["cloud_hat_crest"]
            | c_masks["lower_beak"]
            | c_masks["head_upper"]
        )
        layers["cloud_mantle_back"] = build_part_layer(
            canvas,
            c_masks["cloud_mantle_back"],
            fg_over_mantle,
            cushion_px=16,
            allowed_fill_mask=cloud_fill_m,
        )

        fg_over_wingL = (
            c_masks["foot_L"]
            | c_masks["neck_torso"]
            | c_masks["cloud_hat_crest"]
            | c_masks["lower_beak"]
            | c_masks["head_upper"]
        )
        layers["cloud_wing_L"] = build_part_layer(
            canvas,
            c_masks["cloud_wing_L"],
            fg_over_wingL,
            cushion_px=10,
            allowed_fill_mask=cloud_fill_m,
        )

        layers["foot_L"] = build_part_layer(
            canvas,
            c_masks["foot_L"],
            np.zeros_like(c_masks["foot_L"]),
            cushion_px=0,
        )

        # IMPORTANT: `neck_torso` is in front of `cloud_mantle_back`, `cloud_wing_L`, and `foot_L`,
        # so its `foreground_mask` ONLY includes layers drawn AFTER `neck_torso`:
        fg_over_torso = (
            c_masks["foot_R"]
            | c_masks["cloud_wing_R"]
            | c_masks["cloud_hat_crest"]
            | c_masks["lower_beak"]
            | c_masks["head_upper"]
        )
        torso_lyr = build_part_layer(
            canvas,
            c_masks["neck_torso"],
            fg_over_torso,
            cushion_px=9,
            allowed_fill_mask=(blue_fill_m | cloud_fill_m),
        )

        # Paint resonant singing/Hyper Voice mouth interior pocket behind `lower_beak`
        # using strictly locked palette colors: dark outline `(16, 16, 16)`, mauve `(164, 136, 159)`,
        # and pink `(222, 173, 196)`!
        beak_ys, beak_xs = np.where(c_masks["lower_beak"])
        if len(beak_ys) > 0:
            by_min = int(beak_ys.min())
            bx_min, bx_max = int(beak_xs.min()), int(beak_xs.max())
            for py in range(by_min, by_min + 2):
                for px in range(bx_min, bx_max + 1):
                    if c_masks["lower_beak"][py, px]:
                        if px == bx_min or px == bx_max:
                            torso_lyr[py, px] = (16, 16, 16, 255)
                        elif py == by_min:
                            torso_lyr[py, px] = (164, 136, 159, 255)
                        else:
                            torso_lyr[py, px] = (222, 173, 196, 255)
        layers["neck_torso"] = torso_lyr

        layers["foot_R"] = build_part_layer(
            canvas,
            c_masks["foot_R"],
            np.zeros_like(c_masks["foot_R"]),
            cushion_px=0,
        )

        fg_over_wingR = c_masks["cloud_hat_crest"] | c_masks["head_upper"]
        layers["cloud_wing_R"] = build_part_layer(
            canvas,
            c_masks["cloud_wing_R"],
            fg_over_wingR,
            cushion_px=8,
            allowed_fill_mask=cloud_fill_m,
        )

        fg_over_hat = c_masks["lower_beak"] | c_masks["head_upper"]
        layers["cloud_hat_crest"] = build_part_layer(
            canvas,
            c_masks["cloud_hat_crest"],
            fg_over_hat,
            cushion_px=8,
            allowed_fill_mask=cloud_fill_m,
        )

        layers["lower_beak"] = build_part_layer(
            canvas,
            c_masks["lower_beak"],
            np.zeros_like(c_masks["lower_beak"]),
            cushion_px=0,
        )
        layers["head_upper"] = build_part_layer(
            canvas,
            c_masks["head_upper"],
            np.zeros_like(c_masks["head_upper"]),
            cushion_px=0,
        )

        pivots = {
            "tail_ribbons": to_canvas_pt(62.0, 72.0),       # Tail ribbon root under cloud skirt
            "cloud_mantle_back": to_canvas_pt(55.0, 40.0),  # Dorsal cumulus cloud vortex center
            "cloud_wing_L": to_canvas_pt(30.0, 44.0),       # Left shoulder cloud-wing socket
            "cloud_wing_R": to_canvas_pt(49.0, 46.0),       # Right chest/shoulder cloud-wing socket
            "cloud_hat_crest": to_canvas_pt(35.0, 31.0),    # Crown of head for cloud hat & ear plume
        }

    else:
        # Back view draw order (strictly Back-to-Front):
        # 1. tail_ribbons -> 2. foot_L -> 3. foot_R -> 4. cloud_hat_crest ->
        # 5. cloud_wing_L -> 6. cloud_wing_R -> 7. cloud_mantle_back

        fg_over_tail = c_masks["foot_R"] | c_masks["cloud_wing_R"] | c_masks["cloud_mantle_back"]
        layers["tail_ribbons"] = build_part_layer(
            canvas,
            c_masks["tail_ribbons"],
            fg_over_tail,
            cushion_px=8,
            allowed_fill_mask=blue_fill_m,
        )

        fg_over_footL = c_masks["cloud_wing_L"] | c_masks["cloud_mantle_back"]
        layers["foot_L"] = build_part_layer(
            canvas,
            c_masks["foot_L"],
            fg_over_footL,
            cushion_px=5,
            allowed_fill_mask=blue_fill_m,
        )

        fg_over_footR = c_masks["cloud_mantle_back"]
        layers["foot_R"] = build_part_layer(
            canvas,
            c_masks["foot_R"],
            fg_over_footR,
            cushion_px=5,
            allowed_fill_mask=blue_fill_m,
        )

        fg_over_hat = c_masks["cloud_wing_L"] | c_masks["cloud_wing_R"] | c_masks["cloud_mantle_back"]
        layers["cloud_hat_crest"] = build_part_layer(
            canvas,
            c_masks["cloud_hat_crest"],
            fg_over_hat,
            cushion_px=14,
            allowed_fill_mask=cloud_fill_m,
        )

        fg_over_wingL = c_masks["cloud_mantle_back"]
        layers["cloud_wing_L"] = build_part_layer(
            canvas,
            c_masks["cloud_wing_L"],
            fg_over_wingL,
            cushion_px=14,
            allowed_fill_mask=cloud_fill_m,
        )

        fg_over_wingR = c_masks["cloud_mantle_back"]
        layers["cloud_wing_R"] = build_part_layer(
            canvas,
            c_masks["cloud_wing_R"],
            fg_over_wingR,
            cushion_px=14,
            allowed_fill_mask=cloud_fill_m,
        )

        layers["cloud_mantle_back"] = build_part_layer(
            canvas,
            c_masks["cloud_mantle_back"],
            np.zeros_like(c_masks["cloud_mantle_back"]),
            cushion_px=0,
        )

        layers["neck_torso"] = np.zeros_like(canvas)
        layers["lower_beak"] = np.zeros_like(canvas)
        layers["head_upper"] = np.zeros_like(canvas)

        pivots = {
            "tail_ribbons": to_canvas_pt(65.0, 74.0),       # Tail ribbon root on back.png
            "cloud_mantle_back": to_canvas_pt(52.0, 50.0),  # Central dorsal cloud vortex center
            "cloud_wing_L": to_canvas_pt(33.0, 48.0),       # Left cloud-wing socket on back.png
            "cloud_wing_R": to_canvas_pt(66.0, 42.0),       # Right cloud-wing socket on back.png
            "cloud_hat_crest": to_canvas_pt(44.0, 26.0),    # Upper cloud-crown pivot on back.png
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
        (mid_x - 20 - drift, max(2, top_y + 16 - drift // 2)),
        (mid_x + 20 + drift, max(2, top_y + 18 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            # Pure locked-palette white/pink impact spark `(252, 252, 252)` + `(255, 197, 238)`
            out[sy:sy + 2, sx:sx + 2] = (252, 252, 252, 255)
            out[sy + 1, sx + 1] = (255, 197, 238, 255)
    return out


def synthesize_altaria_mega_frame(
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

    # Sign helper so left/right cloud wings flare/rotate symmetrically outward/upward:
    # On unflipped native front (`facing = -1.0`), `cloud_wing_L` is on the left (`-x`) and `cloud_wing_R` on the right (`+x`).
    # On unflipped native back (`facing = +1.0`), `cloud_wing_L` is on the left (`-x`) and `cloud_wing_R` on the right (`+x`).
    side_sign = facing if is_back else (-facing)

    torso_dx, torso_dy = 0, 0
    head_dx, head_dy = 0, 0
    beak_dx, beak_dy = 0, 0
    hat_dx, hat_dy = 0, 0
    wingL_dx, wingL_dy = 0, 0
    wingR_dx, wingR_dy = 0, 0
    mantle_dx, mantle_dy = 0, 0
    footL_dx, footL_dy = 0, 0
    footR_dx, footR_dy = 0, 0
    tail_dx, tail_dy = 0, 0

    wingL_ang = 0.0
    wingR_ang = 0.0
    mantle_ang = 0.0
    hat_ang = 0.0
    tail_ang = 0.0

    if action == "idle":
        # Modeled directly after Altaria's official 81-frame hovering animation (`0334_altaria/normal/front.gif`):
        # 1. Graceful aerial hovering (`±2px` vertical float + subtle forward drift).
        # 2. Both cotton-cloud wings (`cloud_wing_L`, `cloud_wing_R`) rotate around their shoulder sockets (`±8°`),
        #    while the giant dorsal cumulus cloud mantle (`cloud_mantle_back`) swirls in a phase-shifted vortex (`±6°`)!
        # 3. `lower_beak` opens `+1px` to sing a resonant Fairy/Dragon aria.
        # 4. Long sky-blue 3-ribbon tail (`tail_ribbons`) waves and pivots (`±4.5°`) beneath the cloud skirt.
        hover_dy = int(round(1.8 * math.sin(phase - 0.2)))
        hover_dx = int(round(0.8 * math.cos(phase - 0.2) * facing))

        torso_dx, torso_dy = hover_dx, hover_dy
        head_dx, head_dy = hover_dx, hover_dy

        # Melodic singing/barking beak opening (+1px)
        beak_open = 1 if math.sin(phase - 0.1) > 0.10 else 0
        beak_dx = head_dx
        beak_dy = head_dy + beak_open

        # Fluffy cloud hat & ear plumes follow head with slight phase lag
        hat_dx = head_dx
        hat_dy = head_dy + int(round(0.7 * math.sin(phase - 0.6)))
        hat_ang = 2.8 * math.sin(phase - 0.6) * (-facing)

        # Billowing cotton-cloud wings flap & rotate symmetrically while dorsal mantle swirls
        wing_wave = math.sin(phase - 0.45)
        mantle_wave = math.sin(phase - 1.25)

        wingL_ang = 7.5 * wing_wave * side_sign
        wingR_ang = -7.5 * wing_wave * side_sign
        mantle_ang = 5.0 * mantle_wave * (-facing)

        wingL_dx = hover_dx
        wingL_dy = hover_dy + int(round(1.0 * wing_wave))
        wingR_dx = hover_dx
        wingR_dy = hover_dy + int(round(-1.0 * wing_wave))
        mantle_dx = hover_dx
        mantle_dy = hover_dy + int(round(1.0 * mantle_wave))

        # Aerial hovering talon feet
        footL_dx = hover_dx + int(round(0.6 * sin1 * facing))
        footL_dy = hover_dy + int(round(1.0 * math.sin(phase - 0.5)))
        footR_dx = hover_dx + int(round(-0.6 * sin1 * facing))
        footR_dy = hover_dy + int(round(1.0 * math.sin(phase - 0.9)))

        # Flowing 3-ribbon tail wave (root stays locked to mantle_dx/mantle_dy so it never detaches)
        tail_dx = mantle_dx
        tail_dy = mantle_dy
        tail_ang = 5.0 * math.sin(phase - 1.1) * facing

    elif action == "walk":
        # Dynamic soaring cloud-glide with grand rotational wing-flaps (`±10.5°`), swirling cumulus vortex (`±7°`),
        # singing/roaring beak cadence (`+1px`), and streaming tail ribbons (`±6.5°`):
        glide_dy = int(round(-1.8 * math.sin(phase)))
        glide_dx = int(round(1.5 * math.cos(phase) * facing))

        torso_dx, torso_dy = glide_dx, glide_dy
        head_dx = glide_dx + int(round(0.8 * sin1 * facing))
        head_dy = glide_dy + int(round(0.6 * cos1))

        beak_open = 1 if math.sin(2.0 * phase) > 0.15 else 0
        beak_dx = head_dx
        beak_dy = head_dy + beak_open

        hat_dx = head_dx
        hat_dy = head_dy + int(round(0.6 * math.sin(phase - 0.5)))
        hat_ang = 3.2 * math.sin(phase - 0.5) * (-facing)

        flap_wave = math.sin(phase - 0.35)
        vortex_wave = math.cos(phase - 0.4)

        wingL_ang = 10.5 * flap_wave * side_sign
        wingR_ang = -10.5 * flap_wave * side_sign
        mantle_ang = 6.5 * vortex_wave * (-facing)

        wingL_dx = glide_dx
        wingL_dy = glide_dy + int(round(1.4 * flap_wave))
        wingR_dx = glide_dx
        wingR_dy = glide_dy + int(round(-1.2 * flap_wave))
        mantle_dx = glide_dx
        mantle_dy = glide_dy + int(round(1.0 * vortex_wave))

        footL_dx = glide_dx + int(round(1.4 * cos1 * facing))
        footL_dy = glide_dy + int(round(-1.4 * max(0.0, sin1)))
        footR_dx = glide_dx + int(round(-1.4 * cos1 * facing))
        footR_dy = glide_dy + int(round(-1.4 * max(0.0, -sin1)))

        tail_dx = mantle_dx
        tail_dy = mantle_dy
        tail_ang = 6.5 * math.sin(phase - 0.9) * facing

    elif action == "attack":
        # Signature Mega Altaria "Hyper Voice Resonant Roar + Spinning Cumulus Vortex Sky Attack (`+6px` Lunge)"!
        # Frames 0..3 (Wind-up & Resonant Hyper Voice Roar):
        #   - Draws back in midair (`-3px` back, `-1px` rise),
        #   - Folds both cotton-cloud wings back (`-13°`) while coiling the dorsal cloud mantle (`-9°`),
        #   - Opens `lower_beak` WIDE (`+2px`) for a booming Pixilate Hyper Voice / Dragon Pulse roar!
        # Frames 4..9 (Spinning Cumulus Vortex Sky Attack + Hyper Voice Release `+6px` Lunge):
        #   - Explosive airborne forward lunge (`+6px` forward!),
        #   - Both cotton-cloud wings (`cloud_wing_L`, `cloud_wing_R`) and the giant dorsal cumulus cloud mantle
        #     (`cloud_mantle_back`) execute a high-speed 3D vortex rotation (`+14° -> -12° -> +13° -> -10°`)
        #     like a spinning hurricane of clouds!
        #   - `lower_beak` stays wide open (`+2px`) roaring through frames 2..7 as the soundwave/dive strikes!
        #   - `tail_ribbons` whip dynamically (`±8.5°`) behind the spinning cloud vortex!
        # Frames 10..15: Graceful aerial follow-through recovery to hover stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.55 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        body_dx = int(round(6.0 * k * facing))
        body_dy_table = [0, -1, -2, -2, -1, 1, 1, 0, -1, -1, 0, 0, 0, 0, 0, 0]
        body_dy = body_dy_table[frame_idx]

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx
        head_dy = body_dy

        # Wide +2px Hyper Voice roar from wind-up peak through lunge impact!
        if frame_idx in (2, 3, 4, 5, 6, 7):
            beak_open = 2
        elif frame_idx in (1, 8, 9):
            beak_open = 1
        else:
            beak_open = 0
        beak_dx = head_dx
        beak_dy = head_dy + beak_open

        hat_ang_table = [0.0, -2.5, -4.5, -5.5, 4.5, 6.0, -4.5, 4.5, -3.5, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        hat_dx = head_dx
        hat_dy = head_dy
        hat_ang = hat_ang_table[frame_idx] * (-facing)

        # High-speed Spinning Cloud Wings & Swirling Cumulus Mantle Vortex!
        wingL_spin = [0.0, -6.5, -11.0, -13.5, 12.5, 14.5, -12.0, 13.0, -10.5, 7.0, -3.5, -1.5, 0.0, 0.0, 0.0, 0.0]
        wingR_spin = [0.0, 6.5, 11.0, 13.5, -12.5, -14.5, 12.0, -13.0, 10.5, -7.0, 3.5, 1.5, 0.0, 0.0, 0.0, 0.0]
        mantle_spin = [0.0, -4.5, -7.5, -9.5, 8.5, -10.0, 9.5, -8.5, 7.0, -4.5, 2.5, 1.0, 0.0, 0.0, 0.0, 0.0]

        wingL_ang = wingL_spin[frame_idx] * side_sign
        wingR_ang = wingR_spin[frame_idx] * side_sign
        mantle_ang = mantle_spin[frame_idx] * (-facing)

        wL_dy_lift = -1 if frame_idx in (2, 3, 5, 7) else (1 if frame_idx in (4, 6, 8) else 0)
        wR_dy_lift = 1 if frame_idx in (2, 3, 5, 7) else (-1 if frame_idx in (4, 6, 8) else 0)

        wingL_dx, wingL_dy = body_dx, body_dy + wL_dy_lift
        wingR_dx, wingR_dy = body_dx, body_dy + wR_dy_lift
        mantle_dx, mantle_dy = body_dx, body_dy

        footL_dx = body_dx + int(round(0.8 * k * facing))
        footL_dy = body_dy + (1 if frame_idx in (4, 5, 6) else (-1 if frame_idx in (2, 3) else 0))
        footR_dx = body_dx + int(round(0.4 * k * facing))
        footR_dy = body_dy + (1 if frame_idx in (5, 6, 7) else (-1 if frame_idx in (2, 3) else 0))

        tail_spin = [0.0, -3.5, -6.5, -8.5, 7.5, 9.0, -7.5, 7.0, -5.5, 3.5, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        tail_dx = mantle_dx
        tail_dy = mantle_dy
        tail_ang = tail_spin[frame_idx] * facing

    elif action == "hurt":
        # Aerial knockback recoil + hit-shake + pained roar (`lower_beak` open +2px) + cloud wings flared wide + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-4.0 * env * facing)) + jit
        body_dy = int(round(-1.2 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx
        head_dy = body_dy

        beak_open = 2 if frame_idx in (2, 3, 4, 5) else (1 if 1 <= frame_idx <= 7 else 0)
        beak_dx = head_dx
        beak_dy = head_dy + beak_open

        hat_dx = head_dx
        hat_dy = head_dy
        hat_ang = -4.5 * env * (-facing)

        wingL_ang = -11.0 * env * side_sign
        wingR_ang = 11.0 * env * side_sign
        mantle_ang = -7.5 * env * (-facing)

        wingL_dx, wingL_dy = body_dx, body_dy - int(round(1.0 * env))
        wingR_dx, wingR_dy = body_dx, body_dy + int(round(1.0 * env))
        mantle_dx, mantle_dy = body_dx, body_dy

        footL_dx = body_dx + int(round(1.0 * env * facing))
        footL_dy = body_dy + int(round(1.0 * env))
        footR_dx = body_dx + int(round(0.6 * env * facing))
        footR_dy = body_dy + int(round(1.2 * env))

        tail_dx = mantle_dx
        tail_dy = mantle_dy
        tail_ang = -6.5 * env * facing

    tail_rot = rotate_layer_nearest(layers["tail_ribbons"], tail_ang, pivots["tail_ribbons"])
    mantle_rot = rotate_layer_nearest(layers["cloud_mantle_back"], mantle_ang, pivots["cloud_mantle_back"])
    wingL_rot = rotate_layer_nearest(layers["cloud_wing_L"], wingL_ang, pivots["cloud_wing_L"])
    wingR_rot = rotate_layer_nearest(layers["cloud_wing_R"], wingR_ang, pivots["cloud_wing_R"])
    hat_rot = rotate_layer_nearest(layers["cloud_hat_crest"], hat_ang, pivots["cloud_hat_crest"])

    if not is_back:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(mantle_rot, mantle_dx, mantle_dy),
            shift_layer_int(wingL_rot, wingL_dx, wingL_dy),
            shift_layer_int(layers["foot_L"], footL_dx, footL_dy),
            shift_layer_int(layers["neck_torso"], torso_dx, torso_dy),
            shift_layer_int(layers["foot_R"], footR_dx, footR_dy),
            shift_layer_int(wingR_rot, wingR_dx, wingR_dy),
            shift_layer_int(hat_rot, hat_dx, hat_dy),
            shift_layer_int(layers["lower_beak"], beak_dx, beak_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
        ]
    else:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(layers["foot_L"], footL_dx, footL_dy),
            shift_layer_int(layers["foot_R"], footR_dx, footR_dy),
            shift_layer_int(hat_rot, hat_dx, hat_dy),
            shift_layer_int(wingL_rot, wingL_dx, wingL_dy),
            shift_layer_int(wingR_rot, wingR_dx, wingR_dy),
            shift_layer_int(mantle_rot, mantle_dx, mantle_dy),
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

    # Tight bounds of front (92x92) and back (96x84), padded for cloud-wing vortex spin & +6px lunge
    max_w, max_h = 96, 92
    cell_w = 128
    cell_h = 120
    foot_y = 110

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
                frame_rgba = synthesize_altaria_mega_frame(
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

    # Replace warped 20f `front.gif` / `front-reverse.gif` and `back.gif` / `back-reverse.gif`
    # with clean, zero-distortion 16-frame `idle` S/N loops!
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
        "generator": "forge_altaria_mega_actions",
        "key": "0334_altaria-mega",
        "dex": 334,
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

    print(f"Built 0334_altaria-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, foot_y={foot_y}, sprite_h={max_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
