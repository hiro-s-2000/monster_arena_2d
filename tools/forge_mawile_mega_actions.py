#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Mawile (`0303_mawile-mega`) modeled after Normal Mawile's official GIF
(`0303_mawile/normal/front.gif`) and Mega Mawile's anatomical design (`front.png` & `back.png`):
  - Exact 10-layer anatomical rigging with true anatomical joint pivots:
      1. `mega_jaw_far_upper` (far-side colossal black-steel upper jaw with horns, yellow spot & fangs —
                               rotates dynamically around the occipital hinge pivot!)
      2. `mega_jaw_far_lower` (far-side colossal black-steel lower jaw & fangs — rotates with the far jaw
                               and opens/snaps vertically to roar and crunch!)
      3. `arm_L` (left yellow/pink kimono sleeve arm)
      4. `leg_L` (left pink hakama leg & black foot — 100% grounded `(0,0)` in `idle`, agile stride in `walk`)
      5. `torso` (central yellow upper body & pink hakama waist)
      6. `leg_R` (right pink hakama leg & black foot — 100% grounded `(0,0)` in `idle`, alternating stride in `walk`)
      7. `head` (cute yellow face, black hair crown & crimson eye — 100% rigid integer motion `shift_layer_int`
                 so the face and eye never deform by a single pixel!)
      8. `arm_R` (foreground right kimono sleeve arm & 3-fingered black hand — swings at the shoulder pivot)
      9. `mega_jaw_near_lower` (near-side colossal black-steel lower jaw, pink mouth interior, lower fangs &
                                trailing steel tail-tendril — rotates around the near occipital hinge and opens
                                wide `+2px` to roar or snaps `-1px` shut to crunch!)
     10. `mega_jaw_near_upper` (near-side colossal black-steel upper jaw with large yellow oval spot & upper fangs —
                                rotates around the near occipital hinge in spinning twin-jaw slams!)
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
MEGA_DIR = ROOT / "monsterSprite/gen3/0303_mawile/mega/mawile-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0303_mawile/mega/mawile-mega"
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
    )


def is_steel_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 54) & (g == 54) & (b == 54))
        | ((r == 74) & (g == 74) & (b == 74))
        | ((r == 82) & (g == 82) & (b == 82))
        | ((r == 115) & (g == 115) & (b == 115))
    )


def is_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 211) & (g == 126) & (b == 169))
        | ((r == 213) & (g == 131) & (b == 172))
        | ((r == 160) & (g == 71) & (b == 112))
        | ((r == 156) & (g == 74) & (b == 106))
        | ((r == 189) & (g == 98) & (b == 139))
        | ((r == 184) & (g == 96) & (b == 136))
        | ((r == 152) & (g == 72) & (b == 104))
        | ((r == 255) & (g == 197) & (b == 90))
        | ((r == 222) & (g == 148) & (b == 65))
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
    Segments native 96x96 `front.png` into 10 exact anatomical masks following 1px outlines:
      1. `mega_jaw_far_upper`: Far-side colossal black-steel upper jaw (`x=12..73, y=14..38`)
      2. `mega_jaw_far_lower`: Far-side colossal black-steel lower jaw & fangs (`x=12..40, y=35..44`)
      3. `mega_jaw_near_upper`: Near-side colossal black-steel upper jaw with large yellow spot & upper fangs (`x=23..71, y=26..53`)
      4. `mega_jaw_near_lower`: Near-side colossal black-steel lower jaw, pink mouth interior, lower fangs & steel tendril (`x=22..72, y=38..67`)
      5. `head`: Cute yellow face, black hair crown & crimson eye (`x=70..79, y=41..54`)
      6. `arm_R`: Foreground right kimono sleeve arm & 3-fingered black hand (`x=68..82, y=56..68`)
      7. `arm_L`: Left chest/shoulder kimono sleeve (`x=62..68, y=58..62`)
      8. `leg_L`: Grounded left pink hakama leg & black foot (`x=49..60, y=70..79`)
      9. `leg_R`: Grounded right pink hakama leg & black foot (`x=61..76, y=70..81`)
     10. `torso`: Central yellow upper body & pink hakama waist (`x=54..75, y=45..71`)
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Head (black hair crown + cute yellow face + crimson eye on the right)
    head_zone = (
        ((yy == 41) & (xx >= 70) & (xx <= 76))
        | ((yy == 42) & (xx >= 70) & (xx <= 77))
        | ((yy == 43) & (xx >= 70) & (xx <= 78))
        | ((yy == 44) & (xx >= 70) & (xx <= 78))
        | ((yy == 45) & (xx >= 73) & (xx <= 79))
        | ((yy == 46) & (xx >= 73) & (xx <= 79))
        | ((yy >= 47) & (yy <= 52) & (xx >= 72) & (xx <= 79))
        | ((yy == 53) & (xx >= 71) & (xx <= 78))
        | ((yy == 54) & (xx >= 71) & (xx <= 75))
    )
    head = a & head_zone

    # 2. Arm R (right kimono sleeve + 3-fingered black hand)
    arm_R_zone = (
        ((yy == 56) & (xx >= 68))
        | ((yy == 57) & (xx >= 68))
        | ((yy == 58) & (xx >= 69))
        | ((yy == 59) & (xx >= 69))
        | ((yy == 60) & (xx >= 69))
        | ((yy == 61) & (xx >= 69))
        | ((yy >= 62) & (yy <= 66) & (xx >= 70))
        | ((yy == 67) & (xx >= 71))
        | ((yy == 68) & (xx >= 74))
    )
    arm_R = a & arm_R_zone & (~head)

    # 3. Leg L & Leg R (pink hakama lower legs + black feet, split cleanly along x=60)
    leg_L_zone = (yy >= 67) & (yy <= 79) & (xx >= 48) & (xx <= 60)
    leg_R_zone = (yy >= 67) & (yy <= 81) & (xx >= 61) & (xx <= 77) & (~arm_R_zone)
    leg_L = a & leg_L_zone
    leg_R = a & leg_R_zone

    # 4. Arm L (left shoulder/sleeve under the steel tendril)
    arm_L_zone = (
        ((yy == 58) & (xx >= 64) & (xx <= 68))
        | ((yy == 59) & (xx >= 62) & (xx <= 67))
        | ((yy == 60) & (xx >= 61) & (xx <= 67))
        | ((yy == 61) & (xx >= 60) & (xx <= 66))
    )
    arm_L = a & arm_L_zone

    # 5. Torso (yellow neck/chest + upper pink hakama waist)
    torso_zone = (
        ((yy >= 45) & (yy <= 54) & (xx >= 68) & (xx <= 72))
        | ((yy == 55) & (xx >= 67) & (xx <= 72))
        | ((yy == 56) & (xx >= 66) & (xx <= 67))
        | ((yy == 57) & (xx >= 65) & (xx <= 67))
        | ((yy >= 58) & (yy <= 69) & (xx >= 55) & (xx <= 73))
    )
    # Exclude the steel tendril pixels on the left side of the waist (`x<=60` at `y=58..67`)
    steel_tendril_cut = (
        ((yy == 58) & (xx <= 63))
        | ((yy == 59) & (xx <= 61))
        | ((yy == 60) & (xx <= 60))
        | ((yy == 61) & (xx <= 59))
        | ((yy == 62) & (xx <= 57))
        | ((yy == 63) & (xx <= 55))
        | ((yy == 64) & (xx <= 54))
        | ((yy == 65) & (xx <= 52))
        | ((yy == 66) & (xx <= 51))
        | ((yy == 67) & (xx <= 50))
    )
    torso = a & torso_zone & (~steel_tendril_cut) & (~head) & (~arm_R) & (~arm_L) & (~leg_L) & (~leg_R)

    body_all = head | arm_R | arm_L | leg_L | leg_R | torso
    jaws_all = a & (~body_all)

    # Now split `jaws_all` into the 4 colossal jaw components:
    # A. `mega_jaw_far_lower` (far lower jaw tip & lower teeth at y=35..44, x=12..40)
    far_lower_zone = (
        ((yy == 35) & (xx >= 25) & (xx <= 26))
        | ((yy == 36) & (((xx >= 22) & (xx <= 28)) | ((xx >= 37) & (xx <= 38))))
        | ((yy == 37) & (((xx >= 21) & (xx <= 28)) | ((xx >= 33) & (xx <= 39))))
        | ((yy == 38) & ((xx <= 28) | ((xx >= 33) & (xx <= 39))))
        | ((yy == 39) & ((xx <= 28) | ((xx >= 33) & (xx <= 40))))
        | ((yy == 40) & (xx <= 29))
        | ((yy == 41) & (xx <= 29))
        | ((yy == 42) & (xx <= 28))
        | ((yy == 43) & (xx <= 28))
        | ((yy == 44) & (xx <= 27))
    )
    mega_jaw_far_lower = jaws_all & far_lower_zone

    # B. `mega_jaw_far_upper` (far upper jaw at y=14..37 + far occipital arch at x>=68, y=31..40)
    far_upper_zone = (
        (yy <= 25)
        | ((yy == 26) & ((xx <= 46) | (xx >= 58)))
        | ((yy == 27) & ((xx <= 44) | (xx >= 66)))
        | ((yy == 28) & ((xx <= 43) | (xx >= 68)))
        | ((yy == 29) & ((xx <= 42) | (xx >= 69)))
        | ((yy == 30) & ((xx <= 42) | (xx >= 69)))
        | ((yy == 31) & ((xx <= 42) | (xx >= 70)))
        | ((yy == 32) & ((xx <= 42) | (xx >= 70)))
        | ((yy == 33) & ((xx <= 42) | (xx >= 70)))
        | ((yy == 34) & ((xx <= 41) | (xx >= 71)))
        | ((yy == 35) & (xx == 38))
        | ((yy >= 35) & (yy <= 40) & (xx >= 71))
    )
    mega_jaw_far_upper = jaws_all & far_upper_zone & (~mega_jaw_far_lower)

    near_jaws = jaws_all & (~mega_jaw_far_upper) & (~mega_jaw_far_lower)

    # C. `mega_jaw_near_upper` (near colossal upper jaw with large yellow oval spot & upper teeth row)
    near_upper_zone = (
        ((yy >= 26) & (yy <= 36) & (xx <= 70))
        | ((yy == 37) & (xx >= 40) & (xx <= 62))
        | ((yy == 38) & (xx >= 42) & (xx <= 61))
        | ((yy == 39) & (xx >= 42) & (xx <= 59))
        | ((yy == 40) & (xx >= 31) & (xx <= 59))
        | ((yy == 41) & (xx >= 30) & (xx <= 59))
        | ((yy == 42) & (xx >= 30) & (xx <= 59))
        | ((yy == 43) & (xx >= 29) & (xx <= 58))
        | ((yy == 44) & (xx >= 26) & (xx <= 56))
        | ((yy == 45) & (xx >= 27) & (xx <= 55))
        | ((yy == 46) & (xx >= 23) & (xx <= 54))
        | ((yy == 47) & (xx >= 23) & (xx <= 48))
        | ((yy == 48) & (xx >= 23) & (xx <= 45))
        | ((yy == 49) & (xx >= 24) & (xx <= 44))
        | ((yy == 50) & (xx >= 26) & (xx <= 43))
        | ((yy == 51) & (xx >= 28) & (xx <= 39))
        | ((yy == 52) & (xx >= 29) & (xx <= 39))
        | ((yy == 53) & (xx >= 30) & (xx <= 39))
    )
    mega_jaw_near_upper = near_jaws & near_upper_zone

    # D. `mega_jaw_near_lower` (near colossal lower jaw, pink mouth interior, lower fangs & steel tendril)
    mega_jaw_near_lower = near_jaws & (~mega_jaw_near_upper)

    return {
        "mega_jaw_far_upper": mega_jaw_far_upper,
        "mega_jaw_far_lower": mega_jaw_far_lower,
        "arm_L": arm_L,
        "leg_L": leg_L,
        "torso": torso,
        "leg_R": leg_R,
        "head": head,
        "arm_R": arm_R,
        "mega_jaw_near_lower": mega_jaw_near_lower,
        "mega_jaw_near_upper": mega_jaw_near_upper,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 10 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Head (left-facing yellow face, crimson eye, and black crown on left side of back.png)
    head_zone = (
        ((yy >= 44) & (yy <= 45) & (xx >= 15) & (xx <= 21))
        | ((yy >= 46) & (yy <= 53) & (xx >= 14) & (xx <= 21))
        | ((yy == 54) & (xx >= 15) & (xx <= 24))
        | ((yy == 55) & (xx >= 16) & (xx <= 26))
    )
    head = a & head_zone

    # 2. Arm L (extended left kimono sleeve & 3-fingered black hand on the far left of back.png)
    arm_L_zone = (
        ((yy >= 60) & (yy <= 69) & (xx >= 8) & (xx <= 21) & (xx <= (21 - (yy - 60) // 2)))
    )
    arm_L = a & arm_L_zone

    # 3. Arm R (right kimono sleeve & black hand resting in front of the right hip)
    arm_R_zone = (
        ((yy >= 59) & (yy <= 65) & (xx >= 29) & (xx <= 36))
        | ((yy >= 66) & (yy <= 70) & (xx >= 29) & (xx <= 36))
    )
    arm_R = a & arm_R_zone

    # 4. Leg L & Leg R (pink hakama legs & black feet at the bottom of back.png)
    leg_L_zone = (yy >= 67) & (yy <= 81) & (xx >= 13) & (xx <= 24)
    leg_R_zone = (yy >= 67) & (yy <= 82) & (xx >= 25) & (xx <= 42)
    leg_L = a & leg_L_zone
    leg_R = a & leg_R_zone & (~arm_R)

    # 5. Torso (neck, yellow shoulders, and upper pink/magenta hakama back)
    torso_zone = (
        ((yy >= 56) & (yy <= 59) & (xx >= 16) & (xx <= 33))
        | ((yy >= 60) & (yy <= 69) & (xx >= 16) & (xx <= 37))
    )
    torso = a & torso_zone & (~head) & (~arm_L) & (~arm_R) & (~leg_L) & (~leg_R)

    body_all = head | arm_L | arm_R | leg_L | leg_R | torso
    jaws_all = a & (~body_all)

    # Split `jaws_all` on `back.png` into the 4 colossal jaw components:
    # A. `mega_jaw_near_upper`: Upper colossal jaw with 3 horns, yellow oval spot, upper snout & upper teeth (`y=12..43`)
    near_upper_zone = (
        (yy <= 34)
        | ((yy == 35) & ((xx <= 48) | (xx >= 52)))
        | ((yy == 36) & ((xx <= 36) | ((xx >= 42) & (xx <= 48)) | ((xx >= 55) & (xx <= 66))))
        | ((yy == 37) & ((xx <= 36) | ((xx >= 43) & (xx <= 48)) | ((xx >= 63) & (xx <= 65))))
        | ((yy >= 38) & (yy <= 43) & (xx <= 21))
    )
    mega_jaw_near_upper = jaws_all & near_upper_zone

    # B. `mega_jaw_near_lower`: Upper colossal jaw's lower mandible, pink mouth lining & lower fangs (`y=35..51, x>=35`)
    near_lower_zone = (
        ((yy >= 35) & (yy <= 41) & (xx >= 37) & (xx <= 72))
        | ((yy >= 42) & (yy <= 46) & (xx >= 38) & (xx <= 85))
        | ((yy >= 47) & (yy <= 50) & (xx >= 42) & (xx <= 85))
    )
    mega_jaw_near_lower = jaws_all & near_lower_zone & (~mega_jaw_near_upper)

    rem_jaws = jaws_all & (~mega_jaw_near_upper) & (~mega_jaw_near_lower)

    # C. `mega_jaw_far_upper`: Lower/far jaw's upper section (`y=36..49, x=22..45`)
    far_upper_zone = (yy >= 36) & (yy <= 49) & (xx >= 22) & (xx <= 45)
    mega_jaw_far_upper = rem_jaws & far_upper_zone

    # D. `mega_jaw_far_lower`: Lower/far jaw's lower mandible (`y=50..60, x>=35`) & twin steel tail-tendrils (`x=21..49, y=44..68`)
    mega_jaw_far_lower = rem_jaws & (~mega_jaw_far_upper)

    return {
        "mega_jaw_far_upper": mega_jaw_far_upper,
        "mega_jaw_far_lower": mega_jaw_far_lower,
        "arm_L": arm_L,
        "leg_L": leg_L,
        "torso": torso,
        "leg_R": leg_R,
        "head": head,
        "arm_R": arm_R,
        "mega_jaw_near_lower": mega_jaw_near_lower,
        "mega_jaw_near_upper": mega_jaw_near_upper,
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
    IMPORTANT: For every layer, `foreground_mask` ONLY includes layers drawn ON TOP OF that layer,
    never background layers!
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

    steel_fill_m = is_steel_fill(canvas)
    body_fill_m = is_body_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order (back to front):
        # 1. mega_jaw_far_lower
        # 2. mega_jaw_far_upper
        # 3. arm_L
        # 4. leg_L
        # 5. torso
        # 6. leg_R
        # 7. head
        # 8. arm_R
        # 9. mega_jaw_near_lower
        # 10. mega_jaw_near_upper

        fg_9_10 = c_masks["mega_jaw_near_lower"] | c_masks["mega_jaw_near_upper"]
        fg_8_10 = c_masks["arm_R"] | fg_9_10
        fg_7_10 = c_masks["head"] | fg_8_10
        fg_6_10 = c_masks["leg_R"] | fg_7_10
        fg_5_10 = c_masks["torso"] | fg_6_10
        fg_4_10 = c_masks["leg_L"] | fg_5_10
        fg_3_10 = c_masks["arm_L"] | fg_4_10
        fg_2_10 = c_masks["mega_jaw_far_upper"] | fg_3_10

        u_jfl = ndi.binary_erosion(c_masks["mega_jaw_far_lower"] | fg_2_10, iterations=1)
        layers["mega_jaw_far_lower"] = build_part_layer(
            canvas,
            c_masks["mega_jaw_far_lower"],
            u_jfl & fg_2_10,
            cushion_px=7,
            allowed_fill_mask=steel_fill_m,
        )

        u_jfu = ndi.binary_erosion(c_masks["mega_jaw_far_upper"] | fg_3_10, iterations=1)
        layers["mega_jaw_far_upper"] = build_part_layer(
            canvas,
            c_masks["mega_jaw_far_upper"],
            u_jfu & fg_3_10,
            cushion_px=6,
            allowed_fill_mask=steel_fill_m,
        )

        u_al = ndi.binary_erosion(c_masks["arm_L"] | fg_5_10, iterations=1)
        layers["arm_L"] = build_part_layer(
            canvas,
            c_masks["arm_L"],
            u_al & fg_5_10,
            cushion_px=3,
            allowed_fill_mask=body_fill_m,
        )

        layers["leg_L"] = build_foot_stem_layer(canvas, c_masks["leg_L"], c_masks["torso"], stem_px=4)
        layers["leg_R"] = build_foot_stem_layer(canvas, c_masks["leg_R"], c_masks["torso"], stem_px=4)

        # IMPORTANT: `torso` only expands under `leg_R`, `head`, `arm_R` (never into the open-air steel tendril on the left!)
        fg_over_torso = c_masks["leg_R"] | c_masks["head"] | c_masks["arm_R"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        layers["torso"] = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        # `head` is 100% rigid integer shift
        layers["head"] = build_part_layer(canvas, c_masks["head"], np.zeros_like(c_masks["head"]), cushion_px=0)

        u_ar = ndi.binary_erosion(c_masks["arm_R"] | fg_9_10, iterations=1)
        layers["arm_R"] = build_part_layer(
            canvas,
            c_masks["arm_R"],
            u_ar & fg_9_10,
            cushion_px=3,
            allowed_fill_mask=body_fill_m,
        )

        # Near lower jaw has a dark/magenta mouth-hinge cushion under `mega_jaw_near_upper` so opening `+1..+2px` never leaves a hole!
        u_jnl = ndi.binary_erosion(c_masks["mega_jaw_near_lower"] | c_masks["mega_jaw_near_upper"], iterations=1)
        layers["mega_jaw_near_lower"] = build_part_layer(
            canvas,
            c_masks["mega_jaw_near_lower"],
            u_jnl & c_masks["mega_jaw_near_upper"],
            cushion_px=5,
        )

        layers["mega_jaw_near_upper"] = build_part_layer(
            canvas,
            c_masks["mega_jaw_near_upper"],
            np.zeros_like(c_masks["mega_jaw_near_upper"]),
            cushion_px=0,
        )

        pivots = {
            "jaw_far": to_canvas_pt(65.0, 35.0),
            "jaw_near": to_canvas_pt(67.0, 42.0),
            "arm_R": to_canvas_pt(69.0, 58.0),
            "arm_L": to_canvas_pt(65.0, 58.0),
        }

    else:
        # Back view draw order (back to front):
        # 1. arm_L
        # 2. leg_L
        # 3. torso
        # 4. leg_R
        # 5. arm_R
        # 6. head
        # 7. mega_jaw_far_lower
        # 8. mega_jaw_far_upper
        # 9. mega_jaw_near_lower
        # 10. mega_jaw_near_upper

        fg_9_10 = c_masks["mega_jaw_near_lower"] | c_masks["mega_jaw_near_upper"]
        fg_8_10 = c_masks["mega_jaw_far_upper"] | fg_9_10
        fg_7_10 = c_masks["mega_jaw_far_lower"] | fg_8_10
        fg_6_10 = c_masks["head"] | fg_7_10
        fg_5_10 = c_masks["arm_R"] | fg_6_10
        fg_4_10 = c_masks["leg_R"] | fg_5_10
        fg_3_10 = c_masks["torso"] | fg_4_10

        u_al = ndi.binary_erosion(c_masks["arm_L"] | fg_3_10, iterations=1)
        layers["arm_L"] = build_part_layer(
            canvas,
            c_masks["arm_L"],
            u_al & fg_3_10,
            cushion_px=3,
            allowed_fill_mask=body_fill_m,
        )

        layers["leg_L"] = build_foot_stem_layer(canvas, c_masks["leg_L"], c_masks["torso"], stem_px=4)
        layers["leg_R"] = build_foot_stem_layer(canvas, c_masks["leg_R"], c_masks["torso"], stem_px=4)

        fg_over_torso = c_masks["leg_R"] | c_masks["arm_R"] | c_masks["head"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        layers["torso"] = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        u_ar = ndi.binary_erosion(c_masks["arm_R"] | fg_6_10, iterations=1)
        layers["arm_R"] = build_part_layer(
            canvas,
            c_masks["arm_R"],
            u_ar & fg_6_10,
            cushion_px=3,
            allowed_fill_mask=body_fill_m,
        )

        layers["head"] = build_part_layer(canvas, c_masks["head"], np.zeros_like(c_masks["head"]), cushion_px=0)

        u_jfl = ndi.binary_erosion(c_masks["mega_jaw_far_lower"] | fg_8_10, iterations=1)
        layers["mega_jaw_far_lower"] = build_part_layer(
            canvas,
            c_masks["mega_jaw_far_lower"],
            u_jfl & fg_8_10,
            cushion_px=6,
            allowed_fill_mask=steel_fill_m,
        )

        u_jfu = ndi.binary_erosion(c_masks["mega_jaw_far_upper"] | fg_9_10, iterations=1)
        layers["mega_jaw_far_upper"] = build_part_layer(
            canvas,
            c_masks["mega_jaw_far_upper"],
            u_jfu & fg_9_10,
            cushion_px=6,
            allowed_fill_mask=steel_fill_m,
        )

        u_jnl = ndi.binary_erosion(c_masks["mega_jaw_near_lower"] | c_masks["mega_jaw_near_upper"], iterations=1)
        layers["mega_jaw_near_lower"] = build_part_layer(
            canvas,
            c_masks["mega_jaw_near_lower"],
            u_jnl & c_masks["mega_jaw_near_upper"],
            cushion_px=5,
        )

        layers["mega_jaw_near_upper"] = build_part_layer(
            canvas,
            c_masks["mega_jaw_near_upper"],
            np.zeros_like(c_masks["mega_jaw_near_upper"]),
            cushion_px=0,
        )

        # Shared occipital hinge on `back.png` so the colossal back jaws never split at the arch seam
        pivots = {
            "jaw_far": to_canvas_pt(24.0, 38.0),
            "jaw_near": to_canvas_pt(24.0, 38.0),
            "arm_R": to_canvas_pt(30.0, 59.0),
            "arm_L": to_canvas_pt(19.0, 60.0),
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
        (mid_x - 16 - drift, max(2, top_y + 10 - drift // 2)),
        (mid_x + 16 + drift, max(2, top_y + 12 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (255, 197, 90, 255)
    return out


def synthesize_mawile_mega_frame(
    layers: Dict[str, np.ndarray],
    pivots: Dict[str, Tuple[float, float]],
    canvas_shape: Tuple[int, int, int],
    action: str,
    frame_idx: int,
    facing: float,
    is_back: bool,
    palette_rgba: np.ndarray,
) -> np.ndarray:
    """
    Synthesizes one frame of Mega Mawile (`0303_mawile-mega`) with:
      - Colossal Twin Black-Steel Jaws (`mega_jaw_far_*`, `mega_jaw_near_*`) that rotate around the occipital hinges
        and open/snap vertically to roar, bark, and crunch!
      - 100% rigid integer `head` motion (`shift_layer_int`) so the cute face and crimson eye never distort.
      - Grounded hakama feet (`leg_L`, `leg_R`) in `idle` and alternating steps in `walk`.
    """
    t = frame_idx / float(N_FRAMES)
    phase = 2.0 * math.pi * t
    sin1 = math.sin(phase)
    cos1 = math.cos(phase)

    torso_dx, torso_dy = 0, 0
    head_dx, head_dy = 0, 0
    arm_L_dx, arm_L_dy = 0, 0
    arm_R_dx, arm_R_dy = 0, 0
    leg_L_dx, leg_L_dy = 0, 0
    leg_R_dx, leg_R_dy = 0, 0

    jfar_dx, jfar_dy = 0, 0
    jnear_dx, jnear_dy = 0, 0
    jfar_open_dy = 0
    jnear_open_dy = 0

    jfar_ang = 0.0
    jnear_ang = 0.0
    arm_R_ang = 0.0
    arm_L_ang = 0.0

    if action == "idle":
        # Poised body breathing + alternating twin-jaw pivot rotation & ferocious roaring/barking open-close!
        # 1. Both feet (`leg_L`, `leg_R`) stay 100% planted on the ground `(0, 0)`.
        # 2. Cute body breathes rhythmically (`-1..+1px`).
        # 3. The 2 colossal black-steel jaws (`mega_jaw_far`, `mega_jaw_near`) weave and rotate in phase-shifted arcs
        #    (`±4.2°`) while opening their lower mandibles (`+1px`) to roar and snapping (`-1px`) to intimidate!
        body_dy = int(round(1.0 * math.sin(phase - 0.2)))
        body_dx = int(round(0.5 * math.cos(phase - 0.2) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        arm_R_dx = body_dx + int(round(0.8 * math.cos(phase - 0.5) * (-facing)))
        arm_R_dy = body_dy + int(round(1.0 * math.sin(phase - 0.5)))
        arm_R_ang = 2.5 * math.sin(phase - 0.5) * facing

        arm_L_dx = body_dx
        arm_L_dy = body_dy + int(round(0.8 * math.sin(phase - 0.8)))

        # Twin colossal jaws: orbital rotation + roaring mandible oscillation
        jnear_dx = body_dx + int(round(1.4 * math.cos(phase - 0.4) * facing))
        jnear_dy = body_dy + int(round(1.2 * math.sin(phase - 0.4)))
        jnear_ang = 4.0 * math.sin(phase - 0.4) * facing
        jnear_open_dy = 1 if math.sin(phase - 0.2) > 0.15 else (-1 if math.sin(phase - 0.2) < -0.55 else 0)

        jfar_dx = body_dx + int(round(1.4 * math.cos(phase - 0.9) * facing))
        jfar_dy = body_dy + int(round(1.2 * math.sin(phase - 0.9)))
        jfar_ang = (jnear_ang if is_back else (4.2 * math.sin(phase - 0.9) * facing))
        jfar_open_dy = 1 if math.sin(phase - 0.8) > 0.15 else (-1 if math.sin(phase - 0.8) < -0.55 else 0)

    elif action == "walk":
        # Light kimono-hakama step + counter-swinging sleeves + heavy twin-jaw rotational swagger & barking!
        bounce = int(round(-1.5 * abs(sin1)))
        sway_x = int(round(1.2 * sin1 * facing))

        torso_dx, torso_dy = sway_x, bounce
        head_dx = sway_x + int(round(0.6 * cos1 * facing))
        head_dy = bounce

        arm_R_dx = sway_x + int(round(-1.4 * sin1 * facing))
        arm_R_dy = bounce + int(round(1.2 * cos1))
        arm_R_ang = 4.0 * sin1 * facing

        arm_L_dx = sway_x + int(round(1.2 * sin1 * facing))
        arm_L_dy = bounce + int(round(-1.0 * cos1))
        arm_L_ang = -3.5 * sin1 * facing

        # Twin colossal jaws swing and rotate rhythmically on each stride while barking (`+1px` open)
        jnear_dx = sway_x + int(round(1.6 * cos1 * facing))
        jnear_dy = bounce + int(round(1.4 * math.sin(2.0 * phase - 0.4)))
        jnear_ang = 4.8 * math.sin(phase - 0.3) * facing
        jnear_open_dy = 1 if math.sin(2.0 * phase) > 0.1 else 0

        jfar_dx = sway_x + int(round(1.6 * math.cos(phase - 0.6) * facing))
        jfar_dy = bounce + int(round(1.4 * math.sin(2.0 * phase - 1.0)))
        jfar_ang = (jnear_ang if is_back else (4.8 * math.sin(phase - 0.8) * facing))
        jfar_open_dy = 1 if math.sin(2.0 * phase - 0.8) > 0.1 else 0

        leg_L_dx, leg_L_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        leg_R_dx, leg_R_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Mega Mawile "Twin-Jaw Spinning Iron Head & Play Rough Double-Crunch + Ferocious Roar"!
        # Frames 0..3 (Wind-up & Twin-Jaw Wide Roar):
        #   - Body coils back (`-3px`), both colossal jaws rear high (`-7.0°` / `-6.0°` pivot rotation)
        #     and open wide (`+2px` lower jaw drop) in a terrifying twin roar!
        # Frames 4..9 (Spinning Twin-Jaw Lunge & Double Crunch!):
        #   - Explosive forward lunge (`+4px` body, `+6px` colossal jaws),
        #   - Both colossal jaws execute a high-speed spinning whip-slash (`+7.5°` -> `-6.0°` -> `+7.0°` rotation)
        #     while snapping their fanged mandibles shut (`-1px` crunch on frames 5..6, wide `+2px` roar on frame 7,
        #     second `-1px` crunch on frame 8)!
        # Frames 10..15: Follow-through recovery to stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.65 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        body_dx = int(round(4.0 * k * facing))
        if frame_idx in (1, 2, 3):
            body_dy = 1
        elif frame_idx in (4, 5):
            body_dy = -1
        elif frame_idx in (6, 7, 8):
            body_dy = 1
        else:
            body_dy = 0

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = int(round(4.2 * k * facing))
        head_dy = body_dy

        arm_R_dx = body_dx + int(round(-1.5 * k * facing))
        arm_R_dy = body_dy + (1 if frame_idx in (2, 3) else (-1 if frame_idx in (5, 6, 7) else 0))
        arm_R_ang = (-5.5 if frame_idx in (2, 3) else (6.0 if frame_idx in (5, 6, 7) else 0.0)) * facing

        arm_L_dx = body_dx
        arm_L_dy = body_dy

        # Twin colossal jaws: massive forward thrust (`+6px`) + spinning rotational whip-slash + roaring/crunching bite!
        jaw_thrust = int(round(6.0 * k * facing))
        spin_near_table = [0.0, -4.0, -6.0, -7.0, 5.0, 7.5, -5.5, 6.5, -4.5, 3.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        spin_far_table = [0.0, -3.5, -5.5, -6.5, 4.0, 6.5, -4.5, 5.5, -3.5, 2.5, -1.5, -0.5, 0.0, 0.0, 0.0, 0.0]

        jnear_dx = jaw_thrust
        jnear_dy = body_dy + (-1 if frame_idx in (2, 3, 4) else (1 if frame_idx in (5, 6, 8) else 0))
        jnear_ang = spin_near_table[frame_idx] * facing

        jfar_dx = jaw_thrust
        jfar_dy = body_dy + (-1 if frame_idx in (1, 2, 3) else (1 if frame_idx in (6, 7, 8) else 0))
        jfar_ang = (jnear_ang if is_back else (spin_far_table[frame_idx] * facing))

        # Jaw open/crunch table: wide +2px roar on windup (1..4) and mid-combo (7), tight -1px crunch on impact (5..6, 8..9)
        open_near_table = [0, 1, 2, 2, 2, -1, -1, 2, -1, 1, 0, 0, 0, 0, 0, 0]
        open_far_table = [0, 1, 2, 2, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0]
        jnear_open_dy = open_near_table[frame_idx]
        jfar_open_dy = open_far_table[frame_idx]

        # Lead foot steps into the lunge
        lead_x = int(round(3.0 * max(0.0, k) * facing)) if k >= 0 else int(round(1.5 * k * facing))
        rear_x = int(round(2.0 * k * facing))
        leg_L_dx, leg_L_dy = lead_x, 0
        leg_R_dx, leg_R_dy = rear_x, 0

    elif action == "hurt":
        # Knockback + hit-shake + twin colossal jaws flung upward & roaring open (`+2px`) + sparks!
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-3.5 * env * facing)) + jit
        body_dy = int(round(-1.0 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.5 * env * facing))
        head_dy = int(round(-1.2 * env))

        arm_R_dx = body_dx + int(round(-1.0 * env * facing))
        arm_R_dy = body_dy + int(round(-1.5 * env))
        arm_R_ang = -5.5 * env * facing

        arm_L_dx = body_dx
        arm_L_dy = body_dy + int(round(-1.0 * env))

        jnear_dx = body_dx + int(round(-1.5 * env * facing))
        jnear_dy = body_dy + int(round(-2.0 * env))
        jnear_ang = -6.0 * env * facing
        jnear_open_dy = 2 if frame_idx in (2, 3, 4, 5) else (1 if 1 <= frame_idx <= 7 else 0)

        jfar_dx = body_dx + int(round(-1.5 * env * facing))
        jfar_dy = body_dy + int(round(-2.0 * env))
        jfar_ang = (jnear_ang if is_back else (-6.5 * env * facing))
        jfar_open_dy = 1 if frame_idx in (2, 3, 4, 5) else 0

        slide_x = int(round(body_dx * 0.65))
        leg_L_dx, leg_R_dx = slide_x, slide_x

    # Rotate jaw and arm layers around their anatomical pivots
    jfl_rot = rotate_layer_nearest(layers["mega_jaw_far_lower"], jfar_ang, pivots["jaw_far"])
    jfu_rot = rotate_layer_nearest(layers["mega_jaw_far_upper"], jfar_ang, pivots["jaw_far"])
    jnl_rot = rotate_layer_nearest(layers["mega_jaw_near_lower"], jnear_ang, pivots["jaw_near"])
    jnu_rot = rotate_layer_nearest(layers["mega_jaw_near_upper"], jnear_ang, pivots["jaw_near"])
    arm_R_rot = rotate_layer_nearest(layers["arm_R"], arm_R_ang, pivots["arm_R"])
    arm_L_rot = rotate_layer_nearest(layers["arm_L"], arm_L_ang, pivots["arm_L"])

    if not is_back:
        comp = [
            shift_layer_int(jfl_rot, jfar_dx, jfar_dy + jfar_open_dy),
            shift_layer_int(jfu_rot, jfar_dx, jfar_dy),
            shift_layer_int(arm_L_rot, arm_L_dx, arm_L_dy),
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["head"], head_dx, head_dy),
            shift_layer_int(arm_R_rot, arm_R_dx, arm_R_dy),
            shift_layer_int(jnl_rot, jnear_dx, jnear_dy + jnear_open_dy),
            shift_layer_int(jnu_rot, jnear_dx, jnear_dy),
        ]
    else:
        comp = [
            shift_layer_int(arm_L_rot, arm_L_dx, arm_L_dy),
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(arm_R_rot, arm_R_dx, arm_R_dy),
            shift_layer_int(layers["head"], head_dx, head_dy),
            shift_layer_int(jfl_rot, jfar_dx, jfar_dy + jfar_open_dy),
            shift_layer_int(jfu_rot, jfar_dx, jfar_dy),
            shift_layer_int(jnl_rot, jnear_dx, jnear_dy + jnear_open_dy),
            shift_layer_int(jnu_rot, jnear_dx, jnear_dy),
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

    # Tight bounds of front (71x68) and back (78x71) -> generous padding for spinning colossal jaws!
    max_w, max_h = 78, 71
    cell_w = 104
    cell_h = 92
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
                frame_rgba = synthesize_mawile_mega_frame(
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
        "generator": "forge_mawile_mega_actions",
        "key": "0303_mawile-mega",
        "dex": 303,
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

    print(f"Built 0303_mawile-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
