#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Blaziken (`0257_blaziken-mega`) modeled after Mega Blaziken's official 90-frame
sprite animation (`0257_blaziken/mega/blaziken-mega/front.gif`) and anatomical design (`front.png` & `back.png`):
  - Exact 10-layer hierarchical anatomical rigging with true joint pivots (shoulder -> wrist -> flame ribbon FK,
    hip kicking-leg socket, and head plume base):
      1. `wrist_flame_far` (twin long blazing flame ribbons streaming from the far wrist — hierarchical FK
                            inheriting `arm_far` shoulder rotation + independent wrist flame whip!)
      2. `arm_far` (far-side black-clawed martial-arts hand & red forearm — pivoted at the far shoulder socket)
      3. `head_plumes` (twin tall cream/white V-shaped feather ear-plumes — flex and whip dynamically during
                        breathing, martial-arts footwork, and spinning Blaze Kick!)
      4. `leg_L` (left leg: raised Muay-Thai / Crane-stance kicking leg on `front.png`, or grounded pivot leg
                  on `back.png` — pivoted at the hip socket for high-angle spinning Blaze Kicks!)
      5. `leg_R` (right leg: grounded pivot leg on `front.png` (`(0,0)` planted in `idle`), or raised kicking
                  leg on `back.png`)
      6. `torso` (white feathered chest mantle, upward red chest horns, red waist plumage, upper pivot thigh,
                  and dark-red mouth interior pocket behind `lower_beak`)
      7. `lower_beak` (lower hooked beak / jaw — opens `+1px` for fierce martial-arts breathing/barking in
                       `idle`/`walk` and `+2px` wide for a ferocious roar during `attack` wind-up, spinning
                       Blaze Kick lunge, and `hurt` reaction!)
      8. `head_upper` (red V-flame horn crest, fierce blue/yellow eye, upper hooked beak, and cheek spikes —
                       100% rigid integer motion `shift_layer_int` so the eye and face never deform by a single pixel)
      9. `wrist_flame_near` (twin long blazing flame ribbons streaming from the foreground wrist — hierarchical FK
                             inheriting `arm_near` shoulder rotation + wrist flame spin!)
     10. `arm_near` (foreground red arm & 3-clawed black hand — pivoted at the near shoulder socket for
                     Sky Uppercut & spinning Blaze Kick combos!)
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
import mouth_open  # noqa: E402  (2026-11-04: small mouth opening along the face instead of a jaw drop)

# Mouth line on native front.png (beak faces left): hinge/corner (back of the beak) -> lip tip.
# Carved below the line (rows y=34..35), between the upper beak and the chin outline at y=36.
MOUTH_LINE_FRONT_NATIVE = [(28.0, 33.0), (22.0, 33.0)]
MOUTH_INNER = (131, 41, 41)   # dark red already in the palette (beak interior)


def mouth_deep(palette_rgba: np.ndarray) -> Tuple[int, int, int]:
    pal = palette_rgba[:, :3].astype(int)
    c = pal[int(np.argmin(pal.sum(axis=1)))]
    return int(c[0]), int(c[1]), int(c[2])


def carve_mouth(img: np.ndarray, pivots: Dict, amount: int, dx: int, dy: int, palette_rgba: np.ndarray) -> np.ndarray:
    line = pivots.get("mouth_line")
    if not line or amount <= 0:
        return img
    ln = mouth_open.transform_points(line, dx=dx, dy=dy)
    return mouth_open.open_mouth(img, ln, amount=int(amount), deep=mouth_deep(palette_rgba), inner=MOUTH_INNER, peak=0.6)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0257_blaziken/mega/blaziken-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0257_blaziken/mega/blaziken-mega"
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
        | ((r == 1) & (g == 1) & (b == 1))
        | ((r == 98) & (g == 82) & (b == 74))
    )


def is_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 222) & (g == 222) & (b == 180))
        | ((r == 189) & (g == 180) & (b == 148))
        | ((r == 148) & (g == 131) & (b == 98))
        | ((r == 238) & (g == 98) & (b == 98))
        | ((r == 189) & (g == 65) & (b == 65))
        | ((r == 131) & (g == 41) & (b == 41))
        | ((r == 41) & (g == 41) & (b == 41))
        | ((r == 65) & (g == 65) & (b == 65))
        | ((r == 31) & (g == 31) & (b == 31))
        | ((r == 53) & (g == 53) & (b == 53))
    )


def is_flame_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 255) & (g == 110) & (b == 69))
        | ((r == 246) & (g == 213) & (b == 98))
        | ((r == 222) & (g == 164) & (b == 82))
        | ((r == 240) & (g == 160) & (b == 16))
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


def guard_claw_outline(layer: np.ndarray, outline_rgb: Tuple[int, int, int] = (1, 1, 1)) -> np.ndarray:
    """
    Ensures bright cream claw pixels `(222, 222, 180)` inside `arm_near` / `arm_far`
    never touch transparent pixels directly after nearest-neighbor rotation.
    """
    out = layer.copy()
    a = out[:, :, 3] >= 128
    claw = a & (out[:, :, 0] == 222) & (out[:, :, 1] == 222) & (out[:, :, 2] == 180)
    if not np.any(claw):
        return out
    dil = ndi.binary_dilation(claw, iterations=1)
    exposed = dil & (~a)
    if np.any(exposed):
        out[exposed, 0] = outline_rgb[0]
        out[exposed, 1] = outline_rgb[1]
        out[exposed, 2] = outline_rgb[2]
        out[exposed, 3] = 255
    return out


def composite_layers(canvas_shape: Tuple[int, int, int], layers: List[np.ndarray]) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=np.uint8)
    for lyr in layers:
        m = lyr[..., 3] >= 128
        out[m] = lyr[m]
    return out


def segment_front_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `front.png` (facing left in Muay-Thai / Crane stance) into 10 exact anatomical masks:
      1. `wrist_flame_far`: Twin long flame ribbons rising from the far-left wrist (`x=0..50, y=0..36`).
      2. `arm_far`: Far-left 3-clawed black hand & red forearm (`x=4..21, y=31..41`).
      3. `head_plumes`: Twin tall cream/white V-shaped feather ear-plumes (`x=18..45, y=8..33`).
      4. `leg_L`: Raised left kicking leg (black bent knee & red feathered leg flare `x=3..21, y=43..74`).
      5. `leg_R`: Grounded right pivot lower leg, rear claw & red feathered foot flare (`x=16..37, y=76..95`).
      6. `torso`: White feathered chest mantle, upward red chest horns, red waist plumage, and upper pivot thigh.
      7. `lower_beak`: Lower hooked beak / chin (`x=22..27, y=34..36`).
      8. `head_upper`: Red V-flame horn crest, fierce blue/yellow eye, upper beak, and cheek spikes (`x=20..33, y=14..35`).
      9. `wrist_flame_near`: Twin long flame ribbons streaming right from the foreground wrist (`x=46..94, y=15..43`).
     10. `arm_near`: Foreground right red arm & 3-clawed black hand (`x=37..52, y=25..44`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    r, g, b = arr96[:, :, 0], arr96[:, :, 1], arr96[:, :, 2]

    is_flame_col = a & (
        ((r == 255) & (g == 110) & (b == 69))
        | ((r == 246) & (g == 213) & (b == 98))
        | ((r == 222) & (g == 164) & (b == 82))
    )
    eye_box = (yy >= 29) & (yy <= 31) & (xx >= 26) & (xx <= 29)

    # 1. wrist_flame_far
    wrist_flame_far = is_flame_col & (yy <= 36) & (xx <= 53) & (~eye_box) & (~((xx >= 44) & (yy >= 25)))

    # 2. wrist_flame_near: long twin flame ribbons extending right from the wrist cuff (`x >= 52, y >= 15`);
    #    keep the wrist flame cuff (`x=46..51, y=26..40`) attached to `arm_near` so the wrist bridge never detaches!
    wrist_flame_near = is_flame_col & (xx >= 52) & (yy >= 15) & (~wrist_flame_far)

    # 3. lower_beak
    lower_beak = a & (
        ((yy == 34) & (xx >= 22) & (xx <= 27))
        | ((yy == 35) & (xx >= 23) & (xx <= 26))
        | ((yy == 36) & (xx >= 24) & (xx <= 26))
    )

    # 4. arm_far
    arm_far = a & (~wrist_flame_far) & (~lower_beak) & (
        ((yy >= 31) & (yy <= 41) & (xx <= 16))
        | ((yy >= 35) & (yy <= 40) & (xx >= 15) & (xx <= 21))
    )

    # 5. head_upper (red horn crest + black right border at x=24, y=18..22 + eye + upper beak + cheek spikes)
    is_red_col = a & (
        ((r == 238) & (g == 98) & (b == 98))
        | ((r == 189) & (g == 65) & (b == 65))
        | ((r == 131) & (g == 41) & (b == 41))
    )
    head_upper_zone = (
        ((yy >= 14) & (yy <= 24) & (xx >= 20) & (xx <= 27) & (is_red_col | ((xx == 24) & (yy >= 18) & (yy <= 22)) | ((xx == 26) & (yy >= 18) & (yy <= 22))))
        | ((yy >= 25) & (yy <= 31) & (xx >= 21) & (xx <= 33) & (is_red_col | eye_box | ((yy == 30) & (xx == 25)) | ((yy == 31) & (xx == 33))))
        | ((yy == 32) & (xx >= 21) & (xx <= 32))
        | ((yy == 33) & (xx >= 22) & (xx <= 33))
        | ((yy == 34) & (xx >= 28) & (xx <= 32))
        | ((yy == 35) & (xx == 30))
    )
    head_upper = a & head_upper_zone & (~wrist_flame_far) & (~lower_beak)

    # 6. head_plumes (twin cream/white V-shaped feather ear-plumes)
    head_plumes_zone = (
        ((yy >= 8) & (yy <= 31) & (xx >= 17) & (xx <= 45))
        | ((yy >= 32) & (yy <= 33) & (xx >= 33) & (xx <= 37))
    )
    head_plumes = a & head_plumes_zone & (~wrist_flame_far) & (~head_upper) & (~lower_beak) & (~arm_far)

    # 7. arm_near (foreground right red arm, 3-clawed black hand, and wrist flame cuff at x=46..51;
    #              note x=39 at y=34..35 is chest mantle tip!)
    arm_near_zone = (
        ((yy >= 25) & (yy <= 32) & (xx >= 39) & (xx <= 52))
        | ((yy >= 33) & (yy <= 35) & (xx >= 40) & (xx <= 52))
        | ((yy >= 36) & (yy <= 38) & (xx >= 46) & (xx <= 52))
        | ((yy >= 38) & (yy <= 44) & (xx >= 37) & (xx <= 53))
        | ((yy >= 40) & (yy <= 41) & (xx >= 35) & (xx <= 36))
    )
    arm_near = a & arm_near_zone & (~wrist_flame_near) & (~head_plumes)

    # 8. leg_L (raised left kicking leg)
    leg_L_zone = (
        ((yy >= 43) & (yy <= 48) & (xx <= 20))
        | ((yy == 49) & (xx <= 15))
        | ((yy == 50) & (xx <= 15))
        | ((yy >= 51) & (yy <= 53) & (xx <= 16))
        | ((yy >= 54) & (yy <= 59) & (xx <= 17))
        | ((yy == 60) & (xx <= 19))
        | ((yy == 61) & (xx <= 21))
        | ((yy == 62) & (xx <= 21))
        | ((yy >= 63) & (yy <= 74) & (xx <= 22))
    )
    leg_L = a & leg_L_zone

    # 9. leg_R (grounded right pivot lower leg & feathered foot flare)
    leg_R_zone = (yy >= 76) & (xx >= 15) & (xx <= 39)
    leg_R = a & leg_R_zone

    used = (
        wrist_flame_far
        | wrist_flame_near
        | lower_beak
        | arm_far
        | head_upper
        | head_plumes
        | arm_near
        | leg_L
        | leg_R
    )
    torso = a & (~used)

    return {
        "wrist_flame_far": wrist_flame_far,
        "arm_far": arm_far,
        "head_plumes": head_plumes,
        "leg_L": leg_L,
        "leg_R": leg_R,
        "torso": torso,
        "lower_beak": lower_beak,
        "head_upper": head_upper,
        "wrist_flame_near": wrist_flame_near,
        "arm_near": arm_near,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` (facing right in 3/4 rear stance) into matching 10 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    r, g, b = arr96[:, :, 0], arr96[:, :, 1], arr96[:, :, 2]

    is_white_cape = a & (
        ((r == 222) & (g == 222) & (b == 180))
        | ((r == 189) & (g == 180) & (b == 148))
        | ((r == 148) & (g == 131) & (b == 98))
        | ((r == 98) & (g == 82) & (b == 74))
    )

    # 1. wrist_flame_far: twin flame ribbons extending to the left of the back cape (`x <= 39, y=16..50`)
    wrist_flame_far = a & (xx <= 39) & (yy >= 16) & (yy <= 50) & (~is_white_cape) & (~((yy == 33) & (xx >= 37)))

    # 2. lower_beak on 3/4 back profile
    lower_beak = a & (
        ((yy == 27) & (xx == 58))
        | ((yy == 28) & (xx >= 56) & (xx <= 58))
    )

    # 3. head_upper on 3/4 back profile
    head_upper_zone = (
        ((yy >= 15) & (yy <= 23) & (xx >= 55) & (xx <= 59))
        | ((yy >= 24) & (yy <= 28) & (xx >= 53) & (xx <= 59))
    )
    head_upper = a & head_upper_zone & (~lower_beak)

    # 4. head_plumes on back.png
    head_plumes = a & (yy <= 28) & (xx >= 35) & (~wrist_flame_far) & (~head_upper) & (~lower_beak)

    # 5. arm_near on back.png (right extended red arm & 3-clawed hand)
    arm_near_zone = (
        ((yy >= 30) & (yy <= 35) & (xx >= 62))
        | ((yy == 36) & (((xx >= 51) & (xx <= 54)) | (xx >= 66)))
        | ((yy == 37) & (((xx >= 50) & (xx <= 55)) | (xx >= 66)))
        | ((yy == 38) & (((xx >= 50) & (xx <= 56)) | (xx >= 66)))
        | ((yy == 39) & (((xx >= 52) & (xx <= 57)) | (xx >= 64)))
        | ((yy == 40) & (xx >= 53) & (xx <= 71))
        | ((yy == 41) & (xx >= 54) & (xx <= 70) & (~((r == 246) & (g == 213))))
        | ((yy == 42) & (xx >= 55) & (xx <= 65))
        | ((yy == 43) & (xx >= 57) & (xx <= 64))
        | ((yy == 44) & (xx >= 58) & (xx <= 63))
        | ((yy == 45) & (xx >= 59) & (xx <= 61))
    )
    arm_near = a & arm_near_zone

    # 6. wrist_flame_near on back.png (flame ribbon loop around the right wrist & forearm)
    is_flame_or_red = a & (
        ((r == 246) & (g == 213) & (b == 98))
        | ((r == 240) & (g == 160) & (b == 16))
        | ((r == 238) & (g == 98) & (b == 98))
        | ((r == 189) & (g == 65) & (b == 65))
        | ((r == 131) & (g == 41) & (b == 41))
    )
    wrist_flame_near_zone = (
        ((yy >= 29) & (yy <= 39) & (xx >= 54) & (xx <= 67))
        | ((yy >= 41) & (yy <= 47) & (xx >= 64) & (xx <= 71))
        | ((yy >= 48) & (yy <= 53) & (xx >= 55) & (xx <= 71))
        | ((yy >= 54) & (yy <= 55) & (xx >= 55) & (xx <= 67))
    )
    wrist_flame_near = a & wrist_flame_near_zone & (~arm_near) & (
        is_flame_or_red | ((r == 0) & (g == 0) & (b == 0) & (xx >= 64) & (yy <= 53))
    )

    # 7. leg_R on back.png (raised right kicking leg)
    leg_R_zone = (
        ((yy >= 48) & (yy <= 55) & (xx >= 55) & (xx <= 71))
        | ((yy >= 56) & (yy <= 71) & (xx >= 53) & (xx <= 71))
        | ((yy >= 72) & (yy <= 78) & (xx >= 54) & (xx <= 68))
    )
    leg_R = a & leg_R_zone & (~wrist_flame_near)

    # 8. leg_L on back.png (grounded left pivot lower leg & feathered foot flare)
    leg_L_zone = (yy >= 72) & (xx <= 55)
    leg_L = a & leg_L_zone & (~leg_R)

    empty = np.zeros_like(a)
    used = (
        wrist_flame_far
        | lower_beak
        | head_upper
        | head_plumes
        | arm_near
        | wrist_flame_near
        | leg_R
        | leg_L
    )
    torso = a & (~used)

    return {
        "wrist_flame_far": wrist_flame_far,
        "arm_far": empty,
        "head_plumes": head_plumes,
        "leg_L": leg_L,
        "leg_R": leg_R,
        "torso": torso,
        "lower_beak": lower_beak,
        "head_upper": head_upper,
        "wrist_flame_near": wrist_flame_near,
        "arm_near": arm_near,
    }


def build_foot_stem_layer(
    canvas: np.ndarray,
    foot_mask: np.ndarray,
    torso_mask: np.ndarray,
    stem_px: int = 5,
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

    body_fill_m = is_body_fill(canvas)
    flame_fill_m = is_flame_fill(canvas)
    layers: Dict[str, np.ndarray] = {}
    cy_grid, _ = np.mgrid[0:cell_h, 0:cell_w]

    if not is_back:
        # Front view draw order:
        # 1. wrist_flame_far -> 2. arm_far -> 3. head_plumes -> 4. leg_L -> 5. leg_R ->
        # 6. torso -> 7. lower_beak -> 8. head_upper -> 9. wrist_flame_near -> 10. arm_near
        fg_over_ffar = c_masks["head_plumes"] | c_masks["head_upper"]
        u_ffar = ndi.binary_erosion(c_masks["wrist_flame_far"] | fg_over_ffar, iterations=1)
        layers["wrist_flame_far"] = build_part_layer(
            canvas,
            c_masks["wrist_flame_far"],
            u_ffar & fg_over_ffar,
            cushion_px=2,
            allowed_fill_mask=flame_fill_m,
        )

        fg_over_afar = c_masks["torso"] | c_masks["head_plumes"] | c_masks["lower_beak"] | c_masks["head_upper"]
        u_afar = ndi.binary_erosion(c_masks["arm_far"] | fg_over_afar, iterations=1)
        layers["arm_far"] = build_part_layer(
            canvas,
            c_masks["arm_far"],
            u_afar & fg_over_afar,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_plumes = c_masks["torso"] | c_masks["head_upper"] | c_masks["lower_beak"]
        u_plumes = ndi.binary_erosion(c_masks["head_plumes"] | fg_over_plumes, iterations=1)
        layers["head_plumes"] = build_part_layer(
            canvas,
            c_masks["head_plumes"],
            u_plumes & fg_over_plumes,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_legL = c_masks["torso"]
        u_legL = ndi.binary_erosion(c_masks["leg_L"] | fg_over_legL, iterations=1)
        layers["leg_L"] = build_part_layer(
            canvas,
            c_masks["leg_L"],
            u_legL & fg_over_legL,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        layers["leg_R"] = build_foot_stem_layer(canvas, c_masks["leg_R"], c_masks["torso"], stem_px=5)

        # IMPORTANT: `torso` cushion ONLY expands under `lower_beak`, `head_upper`, and the red upper-arm root
        # of `arm_near` (`y_native >= 38`), NEVER behind the floating 3-clawed hand (`y_native <= 37`) or back layers!
        arm_root_y = dst_y0 + (38 - y0)
        arm_near_root = c_masks["arm_near"] & (cy_grid >= arm_root_y)
        fg_over_torso = c_masks["lower_beak"] | c_masks["head_upper"] | arm_near_root
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        # (2026-11-04) no rectangular mouth-interior pocket any more: the jaw stays put and the mouth is carved
        # along MOUTH_LINE_FRONT_NATIVE by mouth_open.open_mouth on the composed frame.
        layers["torso"] = torso_lyr

        layers["lower_beak"] = build_part_layer(canvas, c_masks["lower_beak"], np.zeros_like(c_masks["lower_beak"]), cushion_px=0)
        layers["head_upper"] = build_part_layer(canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0)

        fg_over_fnear = c_masks["arm_near"]
        u_fnear = ndi.binary_erosion(c_masks["wrist_flame_near"] | fg_over_fnear, iterations=1)
        layers["wrist_flame_near"] = build_part_layer(
            canvas,
            c_masks["wrist_flame_near"],
            u_fnear & fg_over_fnear,
            cushion_px=3,
            allowed_fill_mask=flame_fill_m,
        )

        layers["arm_near"] = build_part_layer(canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0)

        pivots = {
            "wrist_flame_far": to_canvas_pt(10.0, 33.0),   # Far wrist flame origin
            "arm_far": to_canvas_pt(21.0, 38.0),           # Far shoulder socket
            "head_plumes": to_canvas_pt(28.0, 28.0),       # Rear cranium plume base
            "leg_L": to_canvas_pt(21.0, 54.0),             # Left hip socket (raised kicking leg)
            "wrist_flame_near": to_canvas_pt(52.0, 33.0),  # Near wrist flame ribbon origin (right of wrist cuff)
            "arm_near": to_canvas_pt(37.0, 40.0),          # Near shoulder socket
            "mouth_line": [to_canvas_pt(x, y) for x, y in MOUTH_LINE_FRONT_NATIVE],
        }

    else:
        # Back view draw order:
        # 1. wrist_flame_far -> 2. head_plumes -> 3. leg_R (raised) -> 4. leg_L (grounded) ->
        # 5. torso -> 6. lower_beak -> 7. head_upper -> 8. wrist_flame_near -> 9. arm_near
        layers["arm_far"] = np.zeros_like(canvas)

        fg_over_ffar = c_masks["torso"] | c_masks["head_plumes"]
        u_ffar = ndi.binary_erosion(c_masks["wrist_flame_far"] | fg_over_ffar, iterations=1)
        layers["wrist_flame_far"] = build_part_layer(
            canvas,
            c_masks["wrist_flame_far"],
            u_ffar & fg_over_ffar,
            cushion_px=6,
            allowed_fill_mask=flame_fill_m,
        )

        fg_over_plumes = c_masks["torso"] | c_masks["head_upper"] | c_masks["lower_beak"]
        u_plumes = ndi.binary_erosion(c_masks["head_plumes"] | fg_over_plumes, iterations=1)
        layers["head_plumes"] = build_part_layer(
            canvas,
            c_masks["head_plumes"],
            u_plumes & fg_over_plumes,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_legR = c_masks["torso"] | c_masks["wrist_flame_near"]
        u_legR = ndi.binary_erosion(c_masks["leg_R"] | fg_over_legR, iterations=1)
        layers["leg_R"] = build_part_layer(
            canvas,
            c_masks["leg_R"],
            u_legR & fg_over_legR,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        layers["leg_L"] = build_foot_stem_layer(canvas, c_masks["leg_L"], c_masks["torso"], stem_px=5)

        arm_root_y = dst_y0 + (36 - y0)
        arm_near_root = c_masks["arm_near"] & (cy_grid >= arm_root_y)
        fg_over_torso = c_masks["lower_beak"] | c_masks["head_upper"] | c_masks["wrist_flame_near"] | arm_near_root
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )
        layers["torso"] = torso_lyr

        layers["lower_beak"] = build_part_layer(canvas, c_masks["lower_beak"], np.zeros_like(c_masks["lower_beak"]), cushion_px=0)
        layers["head_upper"] = build_part_layer(canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0)

        layers["wrist_flame_near"] = build_part_layer(
            canvas,
            c_masks["wrist_flame_near"],
            np.zeros_like(c_masks["wrist_flame_near"]),
            cushion_px=0,
        )

        layers["arm_near"] = build_part_layer(canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0)

        pivots = {
            "wrist_flame_far": to_canvas_pt(38.0, 39.0),   # Midpoint between the two left flame ribbons on back.png
            "arm_far": to_canvas_pt(40.0, 38.0),
            "head_plumes": to_canvas_pt(48.0, 26.0),       # Rear cranium plume base on back.png
            "leg_R": to_canvas_pt(55.0, 56.0),             # Right hip socket (raised kicking leg on back.png)
            "wrist_flame_near": to_canvas_pt(63.0, 38.0),  # Right wrist flame origin on back.png
            "arm_near": to_canvas_pt(53.0, 39.0),          # Right shoulder socket on back.png
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
        (mid_x - 16 - drift, max(2, top_y + 16 - drift // 2)),
        (mid_x + 16 + drift, max(2, top_y + 18 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (246, 213, 98, 255)
    return out


def synthesize_blaziken_mega_frame(
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
    beak_dx, beak_dy = 0, 0
    plumes_dx, plumes_dy = 0, 0
    anear_dx, anear_dy = 0, 0
    afar_dx, afar_dy = 0, 0
    fnear_dx, fnear_dy = 0, 0
    ffar_dx, ffar_dy = 0, 0
    kick_leg_dx, kick_leg_dy = 0, 0
    pivot_leg_dx, pivot_leg_dy = 0, 0

    plumes_ang = 0.0
    anear_ang = 0.0
    afar_ang = 0.0
    fnear_wrist_ang = 0.0
    ffar_wrist_ang = 0.0
    kick_leg_ang = 0.0

    if action == "idle":
        # Modeled directly after Mega Blaziken's official 90f GIF (`front.gif` loop f=0..17):
        # 1. Grounded pivot leg (`leg_R` on front, `leg_L` on back) stays 100% planted on the ground `(0, 0)`.
        # 2. Martial-arts Crane-stance breathing: torso and head bob rhythmically (`-1..+1px`),
        #    while `lower_beak` opens `+1px` for a fierce martial-arts kiai/roar breath!
        # 3. Raised kicking leg (`leg_L` on front, `leg_R` on back) bobs and flexes (`±3.0°`) at the hip socket.
        # 4. Both martial-arts hands (`arm_near`, `arm_far`) trace a rhythmic guard circle, while the
        #    long wrist flame ribbons (`wrist_flame_near`, `wrist_flame_far`) follow the shoulder FK +
        #    whip and swirl (`±4.0°`) around the wrist sockets!
        body_dy = int(round(1.3 * math.sin(phase - 0.2)))
        body_dx = int(round(0.6 * max(0.0, math.sin(phase - 0.2)) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        beak_open = 1 if math.sin(phase - 0.1) > 0.15 else 0
        beak_dx = head_dx
        beak_dy = head_dy

        plumes_dx = head_dx
        plumes_dy = head_dy + int(round(0.7 * math.sin(phase - 0.6)))
        plumes_ang = 0.0

        kick_leg_dx = body_dx
        kick_leg_dy = body_dy + int(round(-1.2 * math.sin(phase - 0.3)))
        kick_leg_ang = 3.0 * math.sin(phase - 0.3) * (-facing)

        anear_dx = body_dx + int(round(1.0 * math.cos(phase - 0.5) * facing))
        anear_dy = body_dy + int(round(1.4 * math.sin(phase - 0.5)))

        afar_dx = body_dx + int(round(1.0 * math.cos(phase - 0.9) * facing))
        afar_dy = body_dy + int(round(1.4 * math.sin(phase - 0.9)))

        fnear_dx, fnear_dy = anear_dx, anear_dy
        fnear_wrist_ang = 5.5 * math.sin(phase - 0.8) * facing

        ffar_dx = body_dx if is_back else afar_dx
        ffar_dy = (body_dy + int(round(0.8 * math.sin(phase - 1.1)))) if is_back else afar_dy
        ffar_wrist_ang = (2.8 * math.sin(phase - 1.1) * facing) if is_back else (6.0 * math.sin(phase - 1.1) * facing)

    elif action == "walk":
        # Agile Muay-Thai / Crane-stance martial-arts step & rhythmic knee drive:
        # Grounded pivot leg steps briskly (`lift_px=2, stride_px=2`), raised kicking leg drives
        # in counter-phase (`±5.5°` hip rotation), arms & flame ribbons counter-swing,
        # and `lower_beak` barks (`+1px`) on each stride!
        bounce = int(round(-1.6 * abs(sin1)))
        sway_x = int(round(1.4 * sin1 * facing))

        torso_dx, torso_dy = sway_x, bounce
        head_dx = sway_x + int(round(0.8 * cos1 * facing))
        head_dy = bounce + int(round(0.8 * math.sin(2.0 * phase - 0.3)))

        beak_open = 1 if math.sin(2.0 * phase) > 0.15 else 0
        beak_dx = head_dx
        beak_dy = head_dy

        plumes_dx = head_dx
        plumes_dy = head_dy + int(round(1.0 * math.sin(2.0 * phase - 0.6)))
        plumes_ang = 0.0

        kick_leg_dx = sway_x + int(round(1.8 * cos1 * facing))
        kick_leg_dy = bounce + int(round(-1.8 * sin1))
        kick_leg_ang = 5.5 * sin1 * (-facing)

        pivot_leg_dx, pivot_leg_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)

        anear_dx = sway_x + int(round(1.2 * sin1 * facing))
        anear_dy = bounce + int(round(1.5 * cos1))

        afar_dx = sway_x + int(round(-1.4 * sin1 * facing))
        afar_dy = bounce + int(round(-1.5 * cos1))

        fnear_dx, fnear_dy = anear_dx, anear_dy
        fnear_wrist_ang = 7.0 * math.sin(2.0 * phase - 0.7) * facing

        ffar_dx = sway_x if is_back else afar_dx
        ffar_dy = (bounce + int(round(1.0 * math.sin(2.0 * phase - 1.0)))) if is_back else afar_dy
        ffar_wrist_ang = (3.2 * math.sin(2.0 * phase - 1.0) * facing) if is_back else (7.5 * math.sin(2.0 * phase - 1.2) * facing)

    elif action == "attack":
        # Signature Mega Blaziken "Ferocious Roar + Spinning Blaze Kick & Sky Uppercut Cyclone"!
        # Modeled after the special action in official `front.gif` (`f=73..89`) + dynamic joint spin:
        # Frames 0..3 (Deep Coil & Ferocious Roar):
        #   - Crouches low (`+2px` down, `-3px` back), cocks kicking leg down/back (`-8°`),
        #   - Opens `lower_beak` wide (`+2px` ferocious roar!),
        #   - Both wrist flame ribbons flare high.
        # Frames 4..9 (Spinning Blaze Kick & Sky Uppercut Combo):
        #   - Explosive airborne forward lunge (`+5px` forward, `-2px` jump!),
        #   - Raised kicking leg (`leg_L` on front, `leg_R` on back) whips around the hip socket in a
        #     high-speed spinning Blaze Kick arc (`+12°` high kick -> `-8°` axe-kick -> `+10°` second spin kick!),
        #   - Foreground & far arms + long wrist flame ribbons execute a spinning fire-ribbon slash
        #     via hierarchical shoulder + wrist rotation,
        #   - `lower_beak` roars wide (`+2px`) during the spinning kick burst!
        # Frames 10..15: Agile landing & martial-arts stance recovery.
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
        body_dy_table = [0, 1, 2, 2, -1, -2, -2, -1, 0, 1, 1, 0, 0, 0, 0, 0]
        body_dy = body_dy_table[frame_idx]

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = int(round(5.2 * k * facing))
        head_dy = body_dy

        # Ferocious +2px wide roar on wind-up (1..3) and spinning Blaze Kick climax (4..6), snap on 7, kiai on 8..9!
        # (2026-11-04) small carved opening: 2 px only at the roar peak (3..5), 1 px otherwise.
        if frame_idx in (3, 4, 5):
            beak_open = 2
        elif frame_idx in (1, 2, 6, 8, 9):
            beak_open = 1
        else:
            beak_open = 0
        beak_dx = head_dx
        beak_dy = head_dy

        # Head plumes flare upward/back with the spinning jump kick without encroaching on the foreground hand gap
        plumes_dx = head_dx
        plumes_dy = head_dy - (1 if frame_idx in (2, 3, 5, 6, 8) else 0)
        plumes_ang = 0.0

        # Kicking leg (`leg_L` on front, `leg_R` on back): deep coil -> high spinning Blaze Kick -> axe kick -> second spin!
        kick_ang_table = [0.0, -4.5, -7.5, -9.0, 8.5, 12.5, 10.0, -8.0, 9.5, 5.0, -2.5, -1.0, 0.0, 0.0, 0.0, 0.0]
        kick_dy_table = [0, 1, 2, 2, -1, -3, -2, 2, -2, -1, 0, 0, 0, 0, 0, 0]
        kick_leg_dx = int(round(5.8 * k * facing))
        kick_leg_dy = body_dy + kick_dy_table[frame_idx]
        kick_leg_ang = kick_ang_table[frame_idx] * (-facing)

        # Grounded pivot leg leaps forward during airborne spinning kick (frames 4..8) and lands cleanly
        pivot_dy_table = [0, 0, 0, 0, -1, -2, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0]
        pivot_leg_dx = int(round(4.2 * k * facing))
        pivot_leg_dy = pivot_dy_table[frame_idx]

        # Arms & Wrist Flame Ribbons: circular martial-arts guard & spinning Cyclone Blaze Ribbon arcs!
        # Keep the intricate 3-clawed hands (`arm_near`, `arm_far`) 100% crisp via integer orbital/uppercut trajectories
        # (just like `head_upper`), while spinning the long wrist flame ribbons (`wrist_flame_near`, `wrist_flame_far`)
        # around the wrist sockets!
        anear_dy_table = [0, 1, 2, 2, -1, -2, -2, 0, -1, 0, 1, 0, 0, 0, 0, 0]
        anear_dx = body_dx + (1 if frame_idx in (1, 2, 3) else (-1 if frame_idx in (5, 6) else 0)) * int(facing)
        anear_dy = anear_dy_table[frame_idx]

        # Raise `arm_far` high during the high kick (frames 4..6, like official `front.gif` f=77..80!)
        afar_dy_table = [0, 1, 2, 2, -2, -4, -3, -1, -2, -1, 0, 0, 0, 0, 0, 0]
        afar_dx = int(round(5.2 * k * facing))
        afar_dy = afar_dy_table[frame_idx]

        wrist_spin_near = [0.0, -5.5, -9.0, -11.5, 8.5, 13.5, -10.5, 11.5, -8.5, 5.5, -3.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        wrist_spin_far = [0.0, 5.5, 9.0, 11.5, -8.5, -13.5, 10.5, -11.5, 8.5, -5.5, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        fnear_dx, fnear_dy = anear_dx, anear_dy
        fnear_wrist_ang = wrist_spin_near[frame_idx] * facing

        ffar_dx = body_dx if is_back else afar_dx
        ffar_dy = body_dy if is_back else afar_dy
        ffar_wrist_ang = (0.45 * wrist_spin_far[frame_idx] * facing) if is_back else (wrist_spin_far[frame_idx] * facing)

    elif action == "hurt":
        # Knockback recoil + hit-shake + pained roar (`lower_beak` open +2px) + flame ribbons whipped back + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-3.5 * env * facing)) + jit
        body_dy = int(round(-1.0 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-1.5 * env))

        beak_open = 1 if 1 <= frame_idx <= 7 else 0
        beak_dx = head_dx
        beak_dy = head_dy

        plumes_dx = head_dx
        plumes_dy = head_dy - (1 if 2 <= frame_idx <= 6 else 0)
        plumes_ang = 0.0

        kick_leg_dx = body_dx + int(round(-1.0 * env * facing))
        kick_leg_dy = body_dy + int(round(1.2 * env))
        kick_leg_ang = -6.0 * env * (-facing)

        pivot_leg_dx = int(round(body_dx * 0.65))
        pivot_leg_dy = 0

        anear_dx = body_dx + int(round(-1.0 * env * facing))
        anear_dy = body_dy + int(round(-1.2 * env))

        afar_dx = body_dx + int(round(-1.2 * env * facing))
        afar_dy = body_dy + int(round(-1.5 * env))

        fnear_dx, fnear_dy = anear_dx, anear_dy
        fnear_wrist_ang = -7.5 * env * facing

        ffar_dx = body_dx if is_back else afar_dx
        ffar_dy = body_dy if is_back else afar_dy
        ffar_wrist_ang = (-3.0 * env * facing) if is_back else (-7.5 * env * facing)

    plumes_rot = rotate_layer_nearest(layers["head_plumes"], plumes_ang, pivots["head_plumes"])
    anear_rot = guard_claw_outline(rotate_layer_nearest(layers["arm_near"], anear_ang, pivots["arm_near"]))

    # Rotate wrist flame ribbons around wrist sockets
    fnear_rot = rotate_layer_nearest(layers["wrist_flame_near"], fnear_wrist_ang, pivots["wrist_flame_near"])

    if not is_back:
        afar_rot = guard_claw_outline(rotate_layer_nearest(layers["arm_far"], afar_ang, pivots["arm_far"]))
        ffar_sh = rotate_layer_nearest(layers["wrist_flame_far"], afar_ang, pivots["arm_far"])
        ffar_rot = rotate_layer_nearest(ffar_sh, ffar_wrist_ang, pivots["wrist_flame_far"])
        legL_rot = rotate_layer_nearest(layers["leg_L"], kick_leg_ang, pivots["leg_L"])
        comp = [
            shift_layer_int(ffar_rot, ffar_dx, ffar_dy),
            shift_layer_int(afar_rot, afar_dx, afar_dy),
            shift_layer_int(plumes_rot, plumes_dx, plumes_dy),
            shift_layer_int(legL_rot, kick_leg_dx, kick_leg_dy),
            shift_layer_int(layers["leg_R"], pivot_leg_dx, pivot_leg_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["lower_beak"], beak_dx, beak_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(fnear_rot, fnear_dx, fnear_dy),
            shift_layer_int(anear_rot, anear_dx, anear_dy),
        ]
    else:
        ffar_rot = rotate_layer_nearest(layers["wrist_flame_far"], ffar_wrist_ang, pivots["wrist_flame_far"])
        legR_rot = rotate_layer_nearest(layers["leg_R"], kick_leg_ang, pivots["leg_R"])
        comp = [
            shift_layer_int(ffar_rot, ffar_dx, ffar_dy),
            shift_layer_int(plumes_rot, plumes_dx, plumes_dy),
            shift_layer_int(legR_rot, kick_leg_dx, kick_leg_dy),
            shift_layer_int(layers["leg_L"], pivot_leg_dx, pivot_leg_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["lower_beak"], beak_dx, beak_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(fnear_rot, fnear_dx, fnear_dy),
            shift_layer_int(anear_rot, anear_dx, anear_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    if not is_back:
        out = carve_mouth(out, pivots, beak_open, head_dx, head_dy, palette_rgba)
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

    # Tight bounds of front (95x96) and back (74x90)
    max_w, max_h = 95, 96
    cell_w = 124
    cell_h = 114
    foot_y = 106

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
                frame_rgba = synthesize_blaziken_mega_frame(
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

    # Preserve official 90f `front.gif` / `front-reverse.gif`, and replace warped 19f `back.gif` / `back-reverse.gif`
    # with clean 16-frame `idle` N loop!
    idle_sheet = sheets_by_action["idle"]
    back_frames = [idle_sheet[4 * cell_h:5 * cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    back_rev_frames = [f[:, ::-1, :].copy() for f in back_frames]

    save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
    save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)

    meta = {
        "generator": "forge_blaziken_mega_actions",
        "key": "0257_blaziken-mega",
        "dex": 257,
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

    print(f"Built 0257_blaziken-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
