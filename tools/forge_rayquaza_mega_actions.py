#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Rayquaza (`0384_rayquaza-mega`) modeled after Normal Rayquaza's official 74-frame
sprite animation (`0384_rayquaza/normal/front.gif`) and Mega Rayquaza's anatomical design (`front.png` & `back.png`):
  - Exact 9-layer anatomical rigging with true joint, streamer-root, and serpentine coil vortex pivots:
      1. `gold_streamers_back` (long luminous gold/orange streamers trailing from the delta horns & wingtips —
                                pivoted at their upper root via `rotate_layer_nearest` so they whip and spiral
                                dynamically in the stratospheric wind!)
      2. `tail_streamers` (twin golden tail-blade horns, floating emerald tail-fins, and trailing tail streamer —
                           pivoted at the tail-crest socket via `rotate_layer_nearest` for serpentine whip motion!)
      3. `lower_tail_coil` (lower U-shaped serpentine body loop, glowing yellow/white orbs, and tail coil —
                            pivoted around its lower coil center with phase-shifted serpentine glide & corkscrew spin!)
      4. `mid_coil_loop` (middle emerald serpentine coil loop, golden-trimmed side fins, and spiraling orb rings —
                          pivoted and shifted in phase with `lower_tail_coil` to create living ascending-dragon undulation!)
      5. `arm_L` (dorsal golden-edged delta shoulder wing-blades + far-side 3-clawed dragon arm —
                  pivoted at the shoulder socket via `rotate_layer_nearest` for majestic wing flares & slashes!)
      6. `upper_neck_torso` (upper emerald/charcoal serpentine neck & chest with glowing golden orb rings, plus
                             a deep crimson mouth interior pocket behind `lower_jaw` — background layers are NEVER
                             included in its `foreground_mask`!)
      7. `arm_R` (foreground 3-clawed dragon arm — pivoted at the shoulder socket via `rotate_layer_nearest`
                  for menacing claw readiness and Dragon Ascent slashes!)
      8. `lower_jaw` (emerald/gold lower delta jaw-horn + 3-plume luminous gold jaw whiskers — opens `+1..+2px`
                      for a thunderous Sky God roar in `idle`/`walk` and **`+3px` wide-open roar** during `attack`
                      wind-up, corkscrew Dragon Ascent lunge, and `hurt` recoil!)
      9. `head_upper` (golden Delta (Δ) forehead horn, vertical front delta blade, black occipital horns,
                       fierce glowing yellow/crimson eye, and upper jaw — 100% rigid integer motion
                       `shift_layer_int` ONLY so the eye and face never deform by a single pixel!)
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
import mouth_open  # noqa: E402  (small mouth opening along the face, 2026-11-04)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0384_rayquaza/mega/rayquaza-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0384_rayquaza/mega/rayquaza-mega"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16

# 2026-11-04 mouth fix (user request): the jaw layer no longer drops/rotates and no rectangular crimson pocket is
# painted. Instead `mouth_open.open_mouth` carves a small lens-shaped gap along the lower-lip line of front.png.
# Native front.png coords (head faces down-left), ordered hinge/corner -> lip tip; the jaw opens toward MOUTH_NORMAL.
# back.png views show no mouth -> no opening.
MOUTH_LINE_FRONT = [(18.0, 45.0), (18.0, 47.0), (17.0, 49.0), (16.0, 51.0), (15.0, 53.0)]
MOUTH_NORMAL_FRONT = (1.0, 0.0)
MOUTH_DEEP = (1, 1, 1)        # darkest palette colour (outline)
MOUTH_INNER = (156, 41, 82)   # existing dark crimson mouth colour of front.png


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
        | ((r == 34) & (g == 82) & (b == 62))
        | ((r == 131) & (g == 98) & (b == 49))
    )


def is_emerald_or_charcoal_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 61) & (g == 125) & (b == 109))
        | ((r == 54) & (g == 176) & (b == 117))
        | ((r == 96) & (g == 210) & (b == 147))
        | ((r == 63) & (g == 63) & (b == 63))
    )


def is_gold_streamer_pixel(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 251) & (g == 226) & (b == 126))
        | ((r == 252) & (g == 148) & (b == 54))
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
    Segments native 96x96 `front.png` (facing left-front in ascending dragon coil pose) into 9 exact anatomical masks:
      1. `gold_streamers_back`: Long luminous gold/orange streamers trailing from the delta horn tips across the sky (`y=0..30`).
      2. `tail_streamers`: Twin golden tail-blade horns (`y=33..62, x=59..78`), floating dorsal tail-fin (`y=58..68, x=52..58`),
                           and right-side tail-tip streamer (`y=52..65, x=73..95`).
      3. `lower_tail_coil`: Lower U-shaped serpentine body loop (`y=82..95, x=31..68`) + ascending right tail coil & fins (`y=56..90, x=54..85`).
      4. `mid_coil_loop`: Middle-left emerald serpentine coil loop, golden-edged side fins, and spiraling gold ring (`y=56..86, x=24..57`).
      5. `arm_L`: Dorsal golden-edged delta shoulder wing-blades (`y=12..33, x=33..76`).
      6. `upper_neck_torso`: Upper emerald/charcoal serpentine neck & chest with glowing orange orb rings (`y=23..66, x=26..56`).
      7. `arm_R`: Foreground right 3-clawed dragon arm (`y=33..58, x=34..49`).
      8. `lower_jaw`: Lower delta jaw-blade horn (`y=43..70, x=11..31`) + 3-plume luminous gold jaw whiskers (`y=31..63, x=24..43`).
      9. `head_upper`: Vertical left delta blade (`x=0..10, y=10..70`), golden Delta forehead horn (`x=11..35, y=10..42`),
                       black occipital horns, glowing eye (`y=45..49, x=11..15`), and upper jaw/teeth (`y=43..56, x=11..21`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    streamer_px = is_gold_streamer_pixel(arr96)

    # 1. `gold_streamers_back` (upper sky streamers trailing from left/right delta horn tips y=0..31)
    # Strictly restrict `y >= 10` to `streamer_px` (`S`/`O`) so the dorsal delta wing-blade tip (`x=58..76`) stays 100% intact on `arm_L`!
    upper_streamer_zone = (
        (yy <= 9)
        | ((yy == 10) & ((xx <= 4) | (xx >= 34)))
        | ((yy >= 11) & (yy <= 31) & (xx >= 58) & streamer_px)
    )
    gold_streamers_back = a & upper_streamer_zone

    # 2. `head_upper` (left vertical delta blade x=0..10, forehead delta horn, cranium, eye, and upper jaw teeth)
    head_upper_zone = (
        ((yy >= 10) & (yy <= 40) & (xx <= 31))
        | ((yy == 10) & (xx >= 32) & (xx <= 33))
        | ((yy == 11) & (xx >= 31) & (xx <= 33))
        | ((yy == 12) & (xx >= 30) & (xx <= 33))
        | ((yy == 13) & (xx >= 30) & (xx <= 34))
        | ((yy == 14) & (xx >= 30) & (xx <= 34))
        | ((yy == 15) & (xx >= 29) & (xx <= 33))
        | ((yy == 16) & (xx >= 28) & (xx <= 33))
        | ((yy == 17) & (xx >= 28) & (xx <= 33))
        | ((yy == 18) & (xx >= 27) & (xx <= 32))
        | ((yy == 19) & (xx >= 26) & (xx <= 32))
        | ((yy == 20) & (xx >= 26) & (xx <= 31))
        | ((yy == 21) & (xx >= 25) & (xx <= 31))
        | ((yy == 22) & (xx >= 24) & (xx <= 31))
        | ((yy >= 41) & (yy <= 42) & (xx <= 23))
        | ((yy == 43) & (xx <= 23))
        | ((yy == 44) & (xx <= 22))
        | ((yy == 45) & (xx <= 22))
        | ((yy == 46) & (xx <= 22))
        | ((yy == 47) & (xx <= 21))
        | ((yy == 48) & (xx <= 21))
        | ((yy == 49) & (xx <= 20))
        | ((yy == 50) & (xx <= 20))
        | ((yy == 51) & (xx <= 19))
        | ((yy == 52) & (xx <= 19))
        | ((yy == 53) & (xx <= 18))
        | ((yy == 54) & (xx <= 16))
        | ((yy == 55) & (xx <= 15))
        | ((yy == 56) & (xx <= 15))
        | ((yy >= 57) & (yy <= 70) & (xx <= 10))
    )
    head_upper = a & head_upper_zone & (~gold_streamers_back)

    # 3. `lower_jaw` (lower delta jaw-blade horn x=11..31, y=38..70 + luminous gold jaw whiskers x=24..43, y=31..62)
    jaw_blade_zone = (
        ((yy == 38) & (xx >= 29) & (xx <= 33))
        | ((yy == 39) & (xx >= 28) & (xx <= 33))
        | ((yy == 40) & (xx >= 24) & (xx <= 34))
        | ((yy >= 41) & (yy <= 42) & (xx >= 24) & (xx <= 34))
        | ((yy >= 43) & (yy <= 56) & (xx >= 16) & (xx <= 34))
        | ((yy >= 57) & (yy <= 70) & (xx >= 11) & (xx <= 31))
    )
    jaw_whisker_zone = (
        ((yy >= 31) & (yy <= 37) & (xx >= 34) & (xx <= 42) & streamer_px)
        | ((yy >= 38) & (yy <= 56) & (xx >= 27) & (xx <= 43) & streamer_px)
        | ((yy >= 57) & (yy <= 62) & (xx >= 24) & (xx <= 32) & streamer_px)
    )
    lower_jaw = a & (jaw_blade_zone | jaw_whisker_zone) & (~head_upper)

    # 4. `arm_R` (foreground 3-clawed right dragon arm y=33..58, x=34..49)
    arm_R_zone = (
        ((yy == 33) & (xx >= 40) & (xx <= 46))
        | ((yy == 34) & (xx >= 39) & (xx <= 47))
        | ((yy == 35) & (xx >= 39) & (xx <= 47))
        | ((yy == 36) & (xx >= 39) & (xx <= 48))
        | ((yy == 37) & (xx >= 40) & (xx <= 48))
        | ((yy == 38) & (xx >= 42) & (xx <= 49))
        | ((yy == 39) & (xx >= 44) & (xx <= 49))
        | ((yy == 40) & (xx >= 44) & (xx <= 49))
        | ((yy == 41) & (xx >= 45) & (xx <= 48))
        | ((yy == 42) & (xx >= 44) & (xx <= 48))
        | ((yy == 43) & (xx >= 44) & (xx <= 47))
        | ((yy == 44) & (xx >= 43) & (xx <= 47))
        | ((yy == 45) & (xx >= 42) & (xx <= 46))
        | ((yy == 46) & (xx >= 41) & (xx <= 46))
        | ((yy == 47) & (xx >= 40) & (xx <= 46))
        | ((yy == 48) & (xx >= 39) & (xx <= 46))
        | ((yy == 49) & (xx >= 38) & (xx <= 45))
        | ((yy == 50) & (xx >= 37) & (xx <= 45))
        | ((yy == 51) & (xx >= 36) & (xx <= 45))
        | ((yy == 52) & (xx >= 35) & (xx <= 45))
        | ((yy == 53) & (xx >= 34) & (xx <= 44))
        | ((yy == 54) & (xx >= 34) & (xx <= 44))
        | ((yy == 55) & (xx >= 34) & (xx <= 44))
        | ((yy == 56) & (xx >= 36) & (xx <= 43))
        | ((yy == 57) & (xx >= 36) & (xx <= 42))
        | ((yy == 58) & (xx >= 36) & (xx <= 41))
    )
    arm_R = a & arm_R_zone & (~lower_jaw)

    # 5. `arm_L` (dorsal golden-edged delta shoulder wing-blades y=12..32, x=32..76)
    arm_L_zone = (
        ((yy >= 12) & (yy <= 22) & (xx >= 32) & (xx <= 76))
        | ((yy == 23) & (xx >= 36) & (xx <= 76))
        | ((yy == 24) & (xx >= 37) & (xx <= 76))
        | ((yy == 25) & (xx >= 44) & (xx <= 76))
        | ((yy == 26) & (xx >= 46) & (xx <= 76))
        | ((yy == 27) & (xx >= 48) & (xx <= 76))
        | ((yy == 28) & (xx >= 50) & (xx <= 76))
        | ((yy == 29) & (xx >= 52) & (xx <= 76))
        | ((yy == 30) & (xx >= 53) & (xx <= 76))
        | ((yy == 31) & (xx >= 55) & (xx <= 76))
        | ((yy == 32) & (xx >= 56) & (xx <= 76))
    )
    arm_L = a & arm_L_zone & (~gold_streamers_back) & (~head_upper)

    # 6. `tail_streamers` (twin golden tail-blade horns y=33..62, x=59..78 + right tail-tip fin & streamer y=52..65, x=73..95
    #                      + floating middle small fin y=58..68, x=52..58)
    tail_horns_zone = (
        ((yy >= 33) & (yy <= 53) & (xx >= 59) & (xx <= 95))
        | ((yy >= 54) & (yy <= 56) & (xx >= 66) & (xx <= 95))
        | ((yy == 57) & (xx >= 68) & (xx <= 95))
        | ((yy == 58) & (xx >= 72) & (xx <= 95))
        | ((yy >= 59) & (yy <= 62) & (xx >= 74) & (xx <= 95))
        | ((yy >= 63) & (yy <= 66) & (xx >= 78) & (xx <= 95))
        | ((yy >= 58) & (yy <= 68) & (xx >= 52) & (xx <= 58))
    )
    tail_streamers = a & tail_horns_zone

    # 7. `upper_neck_torso` (upper neck & black/emerald chest with glowing orange orb rings y=23..65, x=24..58)
    neck_torso_zone = (
        ((yy >= 23) & (yy <= 62) & (xx >= 24) & (xx <= 58))
        | ((yy == 63) & (xx >= 36) & (xx <= 51))
        | ((yy == 64) & (xx >= 42) & (xx <= 51))
        | ((yy == 65) & (xx >= 43) & (xx <= 50))
    )
    upper_neck_torso = (
        a
        & neck_torso_zone
        & (~head_upper)
        & (~lower_jaw)
        & (~arm_R)
        & (~arm_L)
        & (~tail_streamers)
    )

    # 8. `mid_coil_loop` (middle-left emerald coil & golden-trimmed side fins y=63..86, x=24..58 + spiraling gold streamer)
    mid_coil_zone = (
        ((yy >= 63) & (yy <= 78) & (xx >= 24) & (xx <= 58))
        | ((yy >= 79) & (yy <= 81) & (xx >= 28) & (xx <= 54))
        | ((yy >= 82) & (yy <= 86) & (xx >= 30) & (xx <= 52))
        | ((yy == 87) & (xx >= 31) & (xx <= 43))
        | ((yy == 88) & (xx >= 33) & (xx <= 41))
    )
    mid_coil_loop = (
        a
        & mid_coil_zone
        & (~upper_neck_torso)
        & (~lower_jaw)
        & (~tail_streamers)
    )

    # 9. `lower_tail_coil` (lower U-shaped belly loop y=83..95 + ascending right tail coil, orbs, and fins y=54..95)
    used = (
        gold_streamers_back
        | head_upper
        | lower_jaw
        | arm_R
        | arm_L
        | tail_streamers
        | upper_neck_torso
        | mid_coil_loop
    )
    lower_tail_coil = a & (~used)

    return {
        "gold_streamers_back": gold_streamers_back,
        "tail_streamers": tail_streamers,
        "lower_tail_coil": lower_tail_coil,
        "mid_coil_loop": mid_coil_loop,
        "arm_L": arm_L,
        "upper_neck_torso": upper_neck_torso,
        "arm_R": arm_R,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` (3/4 rear ascending dragon coil view facing right) into matching 9 anatomical masks:
      1. `gold_streamers_back`: Luminous gold/orange streamers trailing across the upper sky (`y=0..46, x=10..93`).
      2. `tail_streamers`: Twin tall golden tail-blade horns on the left (`x=0..18, y=22..75`), tail-tip fin,
                           and trailing streamer (`x=14..56, y=46..63`).
      3. `lower_tail_coil`: Left-to-center lower serpentine arch, fins, and glowing orbs (`x=9..57, y=63..93`).
      4. `mid_coil_loop`: Center-to-right lower U-coil, golden-trimmed dorsal fins, and bottom orb loop (`x=42..90, y=61..95`).
      5. `arm_L`: Dorsal golden-edged delta shoulder wing-blades (`y=7..24, x=39..78`) + left 3-clawed dragon arm (`y=27..45, x=50..58`).
      6. `upper_neck_torso`: Upper-right serpentine neck & dorsal column (`y=23..62, x=56..90`).
      7. `arm_R`: Right-side dragon arm & claws (`y=31..48, x=75..86`).
      8. `lower_jaw`: Right lower delta jaw-horn (`y=49..60, x=83..94`) + luminous gold cheek/jaw whiskers (`y=35..58, x=68..85`).
      9. `head_upper`: Right vertical delta blade (`x=88..95, y=11..60`), black occipital horns (`y=27..48, x=73..91`),
                       and upper right horn crest (`y=11..24, x=74..80`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    streamer_px = is_gold_streamer_pixel(arr96)

    # 1. `gold_streamers_back` (upper sky streamers + left loop streamer y=0..46, strictly `streamer_px` near wings/horns)
    sky_streamer_zone = (
        (yy <= 6)
        | ((yy >= 7) & (yy <= 21) & ((xx <= 38) | (xx >= 78)) & streamer_px)
        | ((yy >= 22) & (yy <= 46) & (xx >= 19) & (xx <= 57) & streamer_px)
    )
    gold_streamers_back = a & sky_streamer_zone

    # 2. `head_upper` (right vertical golden delta blade x=88..95, y=11..60 + black occipital horn x=75..91, y=27..48 + upper horn x=74..80, y=11..24)
    head_back_zone = (
        ((yy >= 11) & (yy <= 60) & (xx >= 88))
        | ((yy >= 11) & (yy <= 24) & (xx >= 74) & (xx <= 81))
        | ((yy >= 27) & (yy <= 48) & (xx >= 78) & (xx <= 91))
    )
    head_upper = a & head_back_zone & (~gold_streamers_back)

    # 3. `lower_jaw` (right lower delta jaw-blade y=49..60, x=83..94 + luminous gold cheek/jaw whiskers)
    jaw_back_zone = (
        ((yy >= 49) & (yy <= 60) & (xx >= 83) & (xx <= 94))
        | ((yy >= 34) & (yy <= 58) & (xx >= 67) & (xx <= 87) & streamer_px)
    )
    lower_jaw = a & jaw_back_zone & (~head_upper) & (~gold_streamers_back)

    # 4. `arm_R` (right dragon arm & claws y=31..48, x=73..84)
    arm_R_zone = (yy >= 31) & (yy <= 48) & (xx >= 73) & (xx <= 84)
    arm_R = a & arm_R_zone & (~head_upper) & (~lower_jaw)

    # 5. `arm_L` (dorsal golden-edged shoulder wing-blades y=7..24, x=39..77 + left 3-clawed arm y=27..45, x=50..58)
    arm_L_zone = (
        ((yy >= 7) & (yy <= 22) & (xx >= 39) & (xx <= 77))
        | ((yy >= 23) & (yy <= 24) & (xx >= 48) & (xx <= 65))
        | ((yy >= 27) & (yy <= 45) & (xx >= 50) & (xx <= 58))
    )
    arm_L = a & arm_L_zone & (~gold_streamers_back) & (~head_upper)

    # 6. `tail_streamers` (twin tall golden tail-blade horns x=0..18, y=22..75 + middle floating fin & streamer x=14..56, y=46..62)
    tail_streamers_zone = (
        ((yy >= 22) & (yy <= 71) & (xx <= 18))
        | ((yy >= 72) & (yy <= 75) & (xx <= 8))
        | ((yy >= 46) & (yy <= 62) & (xx >= 14) & (xx <= 56))
    )
    tail_streamers = a & tail_streamers_zone & (~gold_streamers_back)

    # 7. `upper_neck_torso` (upper-right serpentine neck & dorsal body y=22..62, x=56..90)
    neck_back_zone = (yy >= 22) & (yy <= 62) & (xx >= 56) & (xx <= 90)
    upper_neck_torso = (
        a
        & neck_back_zone
        & (~gold_streamers_back)
        & (~head_upper)
        & (~lower_jaw)
        & (~arm_R)
        & (~arm_L)
        & (~tail_streamers)
    )

    # 8. `lower_tail_coil` (left-to-center lower serpentine arch, tail base, and dorsal fins x=8..55, y=63..93)
    lower_coil_zone = (
        ((yy >= 63) & (yy <= 78) & (xx >= 8) & (xx <= 55))
        | ((yy >= 79) & (yy <= 93) & (xx >= 8) & (xx <= 46))
    )
    lower_tail_coil = a & lower_coil_zone & (~tail_streamers)

    # 9. `mid_coil_loop` (center-to-right lower U-coil, golden-trimmed dorsal fins, and bottom orb loop)
    used = (
        gold_streamers_back
        | head_upper
        | lower_jaw
        | arm_R
        | arm_L
        | tail_streamers
        | upper_neck_torso
        | lower_tail_coil
    )
    mid_coil_loop = a & (~used)

    return {
        "gold_streamers_back": gold_streamers_back,
        "tail_streamers": tail_streamers,
        "lower_tail_coil": lower_tail_coil,
        "mid_coil_loop": mid_coil_loop,
        "arm_L": arm_L,
        "upper_neck_torso": upper_neck_torso,
        "arm_R": arm_R,
        "lower_jaw": lower_jaw,
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

    body_fill_m = is_emerald_or_charcoal_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    # Draw order (strictly Back-to-Front for both Front and Back views):
    # 1. gold_streamers_back -> 2. tail_streamers -> 3. lower_tail_coil -> 4. mid_coil_loop ->
    # 5. arm_L -> 6. upper_neck_torso -> 7. arm_R -> 8. lower_jaw -> 9. head_upper

    layers["gold_streamers_back"] = build_part_layer(
        canvas,
        c_masks["gold_streamers_back"],
        c_masks["arm_L"] | c_masks["head_upper"],
        cushion_px=3,
    )

    fg_over_tail_s = c_masks["lower_tail_coil"] | c_masks["mid_coil_loop"]
    layers["tail_streamers"] = build_part_layer(
        canvas,
        c_masks["tail_streamers"],
        fg_over_tail_s,
        cushion_px=5,
        allowed_fill_mask=body_fill_m,
    )

    fg_over_lower_coil = c_masks["mid_coil_loop"] | c_masks["upper_neck_torso"]
    layers["lower_tail_coil"] = build_part_layer(
        canvas,
        c_masks["lower_tail_coil"],
        fg_over_lower_coil,
        cushion_px=6,
        allowed_fill_mask=body_fill_m,
    )

    fg_over_mid_coil = c_masks["upper_neck_torso"] | c_masks["lower_jaw"]
    layers["mid_coil_loop"] = build_part_layer(
        canvas,
        c_masks["mid_coil_loop"],
        fg_over_mid_coil,
        cushion_px=6,
        allowed_fill_mask=body_fill_m,
    )

    fg_over_armL = c_masks["upper_neck_torso"] | c_masks["head_upper"]
    layers["arm_L"] = build_part_layer(
        canvas,
        c_masks["arm_L"],
        fg_over_armL,
        cushion_px=5,
        allowed_fill_mask=body_fill_m,
    )

    # CRITICAL RULE: `upper_neck_torso` is in front of `gold_streamers_back`, `tail_streamers`,
    # `lower_tail_coil`, `mid_coil_loop`, and `arm_L`. Those background layers are NEVER in `fg_over_torso`!
    fg_over_torso = c_masks["arm_R"] | c_masks["lower_jaw"] | c_masks["head_upper"]
    torso_lyr = build_part_layer(
        canvas,
        c_masks["upper_neck_torso"],
        fg_over_torso,
        cushion_px=6,
        allowed_fill_mask=body_fill_m,
    )

    # (2026-11-04) The rectangular crimson "mouth interior pocket" that used to be painted here is removed:
    # the mouth now opens with `mouth_open.open_mouth` along the face instead of a dropped jaw.

    layers["upper_neck_torso"] = torso_lyr

    fg_over_armR = c_masks["lower_jaw"] | c_masks["head_upper"]
    layers["arm_R"] = build_part_layer(
        canvas,
        c_masks["arm_R"],
        fg_over_armR,
        cushion_px=4,
        allowed_fill_mask=body_fill_m,
    )

    layers["lower_jaw"] = build_part_layer(
        canvas,
        c_masks["lower_jaw"],
        c_masks["head_upper"],
        cushion_px=2,
    )

    layers["head_upper"] = build_part_layer(
        canvas,
        c_masks["head_upper"],
        np.zeros_like(c_masks["head_upper"]),
        cushion_px=0,
    )

    if not is_back:
        pivots = {
            "gold_streamers_back": to_canvas_pt(34.0, 11.0),  # Upper delta horn streamer root
            "tail_streamers": to_canvas_pt(68.0, 57.0),       # Twin golden tail-horn socket
            "lower_tail_coil": to_canvas_pt(55.0, 81.0),      # Lower serpentine coil vortex center
            "mid_coil_loop": to_canvas_pt(40.0, 72.0),        # Middle serpentine coil loop pivot
            "arm_L": to_canvas_pt(38.0, 24.0),                # Dorsal delta shoulder-wing socket
            "arm_R": to_canvas_pt(43.0, 36.0),                # Foreground right dragon-arm shoulder socket
            "lower_jaw": to_canvas_pt(25.0, 42.0),            # Jaw hinge socket
        }
        pivots["mouth_line"] = [to_canvas_pt(mx, my) for mx, my in MOUTH_LINE_FRONT]
        nx, ny = MOUTH_NORMAL_FRONT
        pivots["mouth_normal"] = ((-nx) if flip_x else nx, ny)
    else:
        pivots = {
            "gold_streamers_back": to_canvas_pt(74.0, 12.0),  # Upper horn streamer root on back.png
            "tail_streamers": to_canvas_pt(12.0, 70.0),       # Twin golden tail-horn socket on back.png
            "lower_tail_coil": to_canvas_pt(34.0, 77.0),      # Lower left-center serpentine arch pivot
            "mid_coil_loop": to_canvas_pt(67.0, 76.0),        # Middle-right serpentine U-coil pivot
            "arm_L": to_canvas_pt(60.0, 26.0),                # Dorsal shoulder-wing & left arm socket
            "arm_R": to_canvas_pt(76.0, 35.0),                # Right dragon-arm shoulder socket
            "lower_jaw": to_canvas_pt(83.0, 46.0),            # Jaw hinge socket on back.png
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
        (mid_x - 22 - drift, max(2, top_y + 18 - drift // 2)),
        (mid_x + 22 + drift, max(2, top_y + 20 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            # Pure locked-palette white/gold impact spark `(252, 252, 252)` + `(246, 222, 0)`
            out[sy:sy + 2, sx:sx + 2] = (252, 252, 252, 255)
            out[sy + 1, sx + 1] = (246, 222, 0, 255)
    return out


def synthesize_rayquaza_mega_frame(
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
    armL_dx, armL_dy = 0, 0
    armR_dx, armR_dy = 0, 0
    mid_dx, mid_dy = 0, 0
    low_dx, low_dy = 0, 0
    tail_s_dx, tail_s_dy = 0, 0
    sky_s_dx, sky_s_dy = 0, 0

    sky_s_ang = 0.0
    tail_s_ang = 0.0
    low_ang = 0.0
    mid_ang = 0.0
    armL_ang = 0.0
    armR_ang = 0.0
    jaw_ang = 0.0  # the jaw layer never rotates/drops any more (2026-11-04 mouth fix)
    mouth_amt = 0

    if action == "idle":
        # Modeled after Rayquaza's official 74-frame ascending-dragon hover (`0384_rayquaza/normal/front.gif`):
        # 1. Majestic stratospheric coiling float (`±2px` vertical wave) with phase-shifted serpentine coil undulation.
        # 2. Both `mid_coil_loop` (`±5.5°`) and `lower_tail_coil` (`±7.0°`) rotate and glide in phase lag,
        #    while `gold_streamers_back` (`±8.5°`) and `tail_streamers` (`±7.5°`) spiral in the sky wind!
        # 3. `lower_jaw` opens `+1..+2px` to unleash a thunderous Sky-God roar!
        hover_dy = int(round(1.8 * math.sin(phase - 0.2)))
        hover_dx = int(round(0.8 * math.cos(phase - 0.2) * facing))

        torso_dx, torso_dy = hover_dx, hover_dy
        head_dx, head_dy = hover_dx, hover_dy

        # Thunderous Sky-God roaring jaw cadence (+1..+2px)
        roar_w = math.sin(phase - 0.1)
        jaw_open = 2 if roar_w > 0.55 else (1 if roar_w > 0.05 else 0)
        mouth_amt = 1 if jaw_open > 0 else 0
        jaw_dx, jaw_dy = head_dx, head_dy

        # Dorsal shoulder wing-blades (`arm_L`) & foreground 3-clawed dragon arm (`arm_R`)
        armL_ang = 5.5 * math.sin(phase - 0.5) * (-facing)
        armL_dx, armL_dy = hover_dx, hover_dy + int(round(0.8 * math.sin(phase - 0.5)))

        armR_ang = 6.0 * math.sin(phase - 0.7) * (-facing)
        armR_dx = hover_dx + int(round(0.8 * math.cos(phase - 0.7) * facing))
        armR_dy = hover_dy + int(round(1.0 * math.sin(phase - 0.7)))

        # Serpentine ascending-dragon coil wave (`mid_coil_loop` & `lower_tail_coil`)
        mid_wave = math.sin(phase - 0.85)
        low_wave = math.sin(phase - 1.55)
        mid_ang = 5.5 * mid_wave * facing
        low_ang = -7.0 * low_wave * facing

        mid_dx = hover_dx + int(round(1.0 * mid_wave * facing))
        mid_dy = int(round(1.5 * math.sin(phase - 0.7)))

        low_dx = int(round(1.4 * low_wave * facing))
        low_dy = int(round(1.6 * math.sin(phase - 1.3)))

        # Luminous golden streamers & twin tail horns whipping in stratospheric wind
        sky_s_dx, sky_s_dy = head_dx, head_dy
        sky_s_ang = 8.0 * math.sin(phase - 0.9) * facing

        tail_s_dx, tail_s_dy = low_dx, low_dy
        tail_s_ang = 7.5 * math.sin(phase - 1.8) * facing

    elif action == "walk":
        # Ascending Dragon S-Curve Serpentine Glide + Corkscrew Coil Rotation + Barking/Roaring Jaw:
        # Serpentine coils (`mid_coil_loop`, `lower_tail_coil`) undulate forward/backward in an S-wave
        # with rotational swirl (`±8.5°`), while golden streamers (`±10.5°`) trail and whip!
        glide_dy = int(round(-1.8 * math.sin(phase)))
        glide_dx = int(round(1.6 * math.cos(phase) * facing))

        torso_dx, torso_dy = glide_dx, glide_dy
        head_dx = glide_dx + int(round(0.8 * sin1 * facing))
        head_dy = glide_dy + int(round(0.8 * cos1))

        # Roaring/barking jaw on 2-beat flight cadence (+1..+2px)
        bark_w = math.sin(2.0 * phase)
        jaw_open = 2 if bark_w > 0.65 else (1 if bark_w > 0.10 else 0)
        mouth_amt = 1 if jaw_open > 0 else 0
        jaw_dx, jaw_dy = head_dx, head_dy

        armL_ang = 8.5 * math.sin(phase - 0.35) * (-facing)
        armL_dx, armL_dy = glide_dx, glide_dy + int(round(1.2 * math.sin(phase - 0.35)))

        armR_ang = -8.0 * math.sin(phase - 0.35) * (-facing)
        armR_dx = glide_dx + int(round(1.4 * sin1 * facing))
        armR_dy = glide_dy + int(round(1.2 * cos1))

        # S-wave serpentine coil propulsion
        mid_wave = math.sin(phase - 0.7)
        low_wave = math.sin(phase - 1.5)
        mid_ang = 8.0 * mid_wave * facing
        low_ang = -9.5 * low_wave * facing

        mid_dx = int(round(2.0 * math.cos(phase - 0.6) * facing))
        mid_dy = int(round(-1.6 * math.sin(phase - 0.6)))

        low_dx = int(round(-2.2 * math.cos(phase - 0.6) * facing))
        low_dy = int(round(1.6 * math.sin(phase - 1.2)))

        sky_s_dx, sky_s_dy = head_dx, head_dy
        sky_s_ang = 10.5 * math.sin(phase - 0.8) * facing

        tail_s_dx, tail_s_dy = low_dx, low_dy
        tail_s_ang = 10.0 * math.sin(phase - 1.6) * facing

    elif action == "attack":
        # Signature Mega Rayquaza "Delta Stream Great Roar (`+3px` Jaw) + Corkscrew Dragon Ascent (ガリョウテンセイ / 画竜点睛) `+7px` Lunge"!
        # Frames 0..3 (Wind-up & Heaven-Shaking +3px Roar):
        #   - Coils the serpentine body tightly backward (`-4px` back, `-2px` high),
        #   - Flares dorsal delta wing-blades & golden streamers high (`-14°`),
        #   - Opens `lower_jaw` FULL WIDE (`+3px` + `-5°` jaw pivot) for a heaven-shaking Sky-God roar!
        # Frames 4..9 (Supersonic Corkscrew Dragon Ascent / Draco Meteor Lunge `+7px`):
        #   - Explosive supersonic lunge (`+7px` forward!),
        #   - `gold_streamers_back`, `tail_streamers`, `mid_coil_loop`, `lower_tail_coil`, `arm_L`, and `arm_R`
        #     execute a high-speed corkscrew spiral rotation (`+16° -> -15° -> +14° -> -12°`) like a drilling emerald dragon!
        #   - `lower_jaw` stays wide open (`+3px` -> `+2px`) roaring ferociously as the Delta horns cleave the target!
        # Frames 10..15: Follow-through recovery to ascending-dragon stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.60 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.22 * math.sin(u * math.pi)

        body_dx = int(round(7.0 * k * facing))
        body_dy_table = [0, -1, -2, -2, -1, 1, 2, 1, -1, -1, 0, 0, 0, 0, 0, 0]
        body_dy = body_dy_table[frame_idx]

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        # Massive +3px wide-open Sky-God roar during wind-up peak and Dragon Ascent lunge!
        jaw_open_table = [0, 2, 3, 3, 3, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0]
        jaw_open = jaw_open_table[frame_idx]
        mouth_amt = 2 if jaw_open >= 3 else (1 if jaw_open > 0 else 0)
        jaw_dx, jaw_dy = head_dx, head_dy

        # Corkscrew spiral rotation tables for Dragon Ascent (ガリョウテンセイ)
        corkscrew_A = [0.0, -5.0, -8.5, -10.5, 10.5, -11.5, 10.5, -9.5, 7.5, -5.0, 2.5, -1.0, 0.0, 0.0, 0.0, 0.0]
        corkscrew_B = [0.0, 5.0, 8.5, 10.5, -10.5, 11.5, -10.5, 9.5, -7.5, 5.0, -2.5, 1.0, 0.0, 0.0, 0.0, 0.0]
        coil_spin_M = [0.0, -4.5, -7.5, -9.5, 10.0, -11.0, 10.0, -8.5, 6.5, -4.5, 2.5, -1.0, 0.0, 0.0, 0.0, 0.0]
        coil_spin_L = [0.0, 4.5, 8.0, 10.0, -10.5, 11.5, -10.5, 9.0, -7.0, 4.5, -2.5, 1.0, 0.0, 0.0, 0.0, 0.0]

        armL_ang = corkscrew_A[frame_idx] * (-facing)
        armR_ang = corkscrew_B[frame_idx] * (-facing)
        mid_ang = coil_spin_M[frame_idx] * facing
        low_ang = coil_spin_L[frame_idx] * facing
        sky_s_ang = corkscrew_A[frame_idx] * facing
        tail_s_ang = corkscrew_B[frame_idx] * facing

        swirl_y = -2 if frame_idx in (4, 6, 8) else (2 if frame_idx in (5, 7, 9) else 0)

        armL_dx = body_dx
        armL_dy = body_dy + (-1 if frame_idx in (2, 3, 5, 7) else (1 if frame_idx in (4, 6, 8) else 0))

        armR_dx = body_dx + (2 * int(round(facing)) if 4 <= frame_idx <= 8 else 0)
        armR_dy = body_dy + (1 if frame_idx in (4, 6, 8) else (-1 if frame_idx in (2, 3, 5, 7) else 0))

        mid_dx = int(round(5.0 * k * facing))
        mid_dy = body_dy + swirl_y

        low_dx = int(round(3.5 * k * facing))
        low_dy = -swirl_y

        sky_s_dx, sky_s_dy = head_dx, head_dy
        tail_s_dx, tail_s_dy = low_dx, low_dy

    elif action == "hurt":
        # Aerial knockback recoil + hit-shake + wide-open pained roar (`+2..+3px`) + flared coils/streamers + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-4.0 * env * facing)) + jit
        body_dy = int(round(-1.5 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        jaw_open = 3 if frame_idx in (2, 3, 4) else (2 if frame_idx in (1, 5, 6) else (1 if frame_idx == 7 else 0))
        mouth_amt = 1 if jaw_open > 0 else 0
        jaw_dx, jaw_dy = head_dx, head_dy

        armL_ang = -11.0 * env * (-facing)
        armR_ang = 11.0 * env * (-facing)
        mid_ang = -8.5 * env * facing
        low_ang = 9.5 * env * facing
        sky_s_ang = -12.0 * env * facing
        tail_s_ang = 11.0 * env * facing

        armL_dx, armL_dy = body_dx, body_dy - int(round(1.0 * env))
        armR_dx, armR_dy = body_dx, body_dy + int(round(1.0 * env))
        mid_dx, mid_dy = int(round(-2.5 * env * facing)) + jit, int(round(1.0 * env))
        low_dx, low_dy = int(round(-1.5 * env * facing)), int(round(1.2 * env))
        sky_s_dx, sky_s_dy = head_dx, head_dy
        tail_s_dx, tail_s_dy = low_dx, low_dy

    sky_s_rot = rotate_layer_nearest(layers["gold_streamers_back"], sky_s_ang, pivots["gold_streamers_back"])
    tail_s_rot = rotate_layer_nearest(layers["tail_streamers"], tail_s_ang, pivots["tail_streamers"])
    low_rot = rotate_layer_nearest(layers["lower_tail_coil"], low_ang, pivots["lower_tail_coil"])
    mid_rot = rotate_layer_nearest(layers["mid_coil_loop"], mid_ang, pivots["mid_coil_loop"])
    armL_rot = rotate_layer_nearest(layers["arm_L"], armL_ang, pivots["arm_L"])
    armR_rot = rotate_layer_nearest(layers["arm_R"], armR_ang, pivots["arm_R"])
    jaw_rot = rotate_layer_nearest(layers["lower_jaw"], jaw_ang, pivots["lower_jaw"])

    comp = [
        shift_layer_int(sky_s_rot, sky_s_dx, sky_s_dy),
        shift_layer_int(tail_s_rot, tail_s_dx, tail_s_dy),
        shift_layer_int(low_rot, low_dx, low_dy),
        shift_layer_int(mid_rot, mid_dx, mid_dy),
        shift_layer_int(armL_rot, armL_dx, armL_dy),
        shift_layer_int(layers["upper_neck_torso"], torso_dx, torso_dy),
        shift_layer_int(armR_rot, armR_dx, armR_dy),
        shift_layer_int(jaw_rot, jaw_dx, jaw_dy),
        shift_layer_int(layers["head_upper"], head_dx, head_dy),
    ]

    out = composite_layers(canvas_shape, comp)
    if mouth_amt > 0 and not is_back and "mouth_line" in pivots:
        line = mouth_open.transform_points(pivots["mouth_line"], dx=head_dx, dy=head_dy)
        out = mouth_open.open_mouth(
            out, line, amount=mouth_amt, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=0.5,
            normal=pivots["mouth_normal"],
        )
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

    # Full 96x96 native bounds padded generously for +7px Dragon Ascent lunge & corkscrew streamer rotation
    max_w, max_h = 96, 96
    cell_w = 132
    cell_h = 124
    foot_y = 112

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
                frame_rgba = synthesize_rayquaza_mega_frame(
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
        "generator": "forge_rayquaza_mega_actions",
        "key": "0384_rayquaza-mega",
        "dex": 384,
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

    print(
        f"Built 0384_rayquaza-mega -> {OUT_DIR} "
        f"(cell={cell_w}x{cell_h}, foot_y={foot_y}, sprite_h={max_h}, colors={len(palette_rgba)})"
    )


if __name__ == "__main__":
    main()
