#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Absol Z (`0359_absol-mega-z`) modeled after Normal Absol's official 108-frame GIF
(`0359_absol/normal/front.gif`) and Mega Absol Z's native 96x96 sprites (`front.png` & `back.png`):
  - Exact 10-layer anatomical rigging with true joint pivots:
      1. `scythe_tail` (rear 3-pronged slate-grey blade tail — 100% separated from the hind legs and
                        pivoted at the hip tail root via `rotate_layer_nearest` so it whips and slashes!)
      2. `scythe_wing_far` (far-side crimson wing spikes & shoulder fur flare — pivoted at the far shoulder
                            socket via `rotate_layer_nearest` for counter-phase wing rotation & slash flares)
      3. `hind_leg_L` (back-left leg + heel spur — 100% grounded `(0,0)` in `idle`, fast 4-legged diagonal
                       wind-gallop in `walk`)
      4. `hind_leg_R` (back-right leg under belly — 100% grounded `(0,0)` in `idle`, diagonal gallop in `walk`)
      5. `front_leg_R` (front-right leg stepping outward — 100% grounded `(0,0)` in `idle`, diagonal gallop
                        in `walk`, forward pounce in `attack`)
      6. `torso` (dark navy chest mane, long lower bang lock, torso, and thighs)
      7. `front_leg_L` (foreground main front-left leg with 3 silver-white claws — 100% grounded `(0,0)` in
                        `idle`, diagonal gallop in `walk`, slashing pounce in `attack`)
      8. `scythe_wing_main` (colossal 3-bladed crimson scythe wing — pivoted at the dorsal shoulder socket
                             via `rotate_layer_nearest` for dramatic rotary wing expansion & spinning Night Slash!)
      9. `lower_jaw` (roaring lower jaw — opens `+1px` for menacing barks in `idle`/`walk` and `+2px` wide
                      for a ferocious disaster-beast roar in `attack` and `hurt`!)
     10. `head_upper` (silver-white crescent scythe horn, right sub-horn, forehead crest, fierce crimson eye,
                       upper snout, and crimson roaring mouth cavity pocket — 100% rigid integer translation
                       `shift_layer_int` so the face, horns, and red eye never deform by a single pixel!)
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
import mouth_open  # noqa: E402  (small face-following mouth opening, 2026-11-04)

# Closed mouth line on native front.png (hinge/back of snout -> lip tip), measured on the still:
# the dark outline under the slate snout. Back views have no visible mouth (no line, no opening).
MOUTH_LINE_FRONT_NATIVE = [(57.0, 43.0), (58.0, 44.0), (59.0, 44.0), (60.0, 45.0), (62.0, 45.0)]
MOUTH_DEEP = (0, 0, 0)          # darkest palette colour (outline)
MOUTH_INNER = (124, 38, 66)     # existing dark crimson (mouth interior)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0359_absol/mega/absol-mega-z"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0359_absol/mega/absol-mega-z"
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
        | ((r == 8) & (g == 24) & (b == 43))
        | ((r == 41) & (g == 56) & (b == 51))
        | ((r == 124) & (g == 38) & (b == 66))
    )


def is_navy_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 38) & (g == 58) & (b == 84))
        | ((r == 53) & (g == 90) & (b == 125))
        | ((r == 46) & (g == 72) & (b == 103))
    )


def is_slate_horn_tail_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 87) & (g == 90) & (b == 124))
        | ((r == 119) & (g == 136) & (b == 170))
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
    Segments native 96x96 `front.png` into 10 anatomical masks:
      1. `scythe_tail`: 3-pronged slate-grey blade tail at rear-left (`y=42..55, x=29..39`).
      2. `scythe_wing_far`: Right-shoulder crimson spike & shoulder fur flare (`y=40..54, x=68..75`).
      3. `hind_leg_L`: Back-left leg + heel spur (`y=58..75, x=27..43`).
      4. `hind_leg_R`: Back-right leg under belly (`y=58..72, x=44..47` and `y=66..72, x=52..57`).
      5. `front_leg_R`: Front-right leg stepping outward (`y=62..75, x=58..76`).
      6. `torso`: Dark navy chest mane, lower bang lock, torso, and thighs.
      7. `front_leg_L`: Foreground central main front-left leg + 3 silver-white claws (`y=56..81, x=45..60`).
      8. `scythe_wing_main`: Colossal 3-bladed crimson wing on upper-left back + dark wing-arm base (`y=15..50, x=19..53`).
      9. `lower_jaw`: Roaring lower jaw (`y=43..45, x=57..64`).
     10. `head_upper`: Silver-white crescent scythe horn, right sub-horn, forehead crest, red eye, and upper snout (`y=21..45, x=47..74`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    r = arr96[:, :, 0].astype(np.int32)
    g = arr96[:, :, 1].astype(np.int32)
    b = arr96[:, :, 2].astype(np.int32)

    # 1. `scythe_tail` (rear-left 3-pronged slate-grey blade tail)
    is_slate_tail_col = ((r == 41) & (g == 56) & (b == 51)) | ((r == 87) & (g == 90) & (b == 124))
    tail_zone = (
        ((yy >= 42) & (yy <= 55) & (xx >= 29) & (xx <= 39) & is_slate_tail_col)
        | ((yy == 49) & (xx == 39))
    )
    scythe_tail = a & tail_zone

    # 2. `lower_jaw` (roaring lower jaw)
    jaw_zone = (
        ((yy == 43) & (xx >= 57) & (xx <= 64))
        | ((yy == 44) & (xx >= 58) & (xx <= 63))
        | ((yy == 45) & (xx >= 60) & (xx <= 62))
    )
    lower_jaw = a & jaw_zone

    # 3. `scythe_wing_far` (right-shoulder crimson spike & fur flare)
    wing_far_zone = (
        ((yy >= 40) & (yy <= 45) & (xx >= 71) & (xx <= 75))
        | ((yy >= 46) & (yy <= 51) & (xx >= 69) & (xx <= 75))
        | ((yy >= 52) & (yy <= 54) & (xx >= 68) & (xx <= 73))
    )
    scythe_wing_far = a & wing_far_zone

    # 4. `head_upper` (left scythe horn, right sub-horn, forehead crest, red eye, upper snout, upper hair)
    head_zone = (
        ((yy >= 21) & (yy <= 26) & (xx >= 48))
        | ((yy >= 27) & (yy <= 31) & (xx >= 49))
        | ((yy >= 32) & (yy <= 35) & (xx >= 48))
        | ((yy >= 36) & (yy <= 37) & (xx >= 47))
        | ((yy == 38) & (xx >= 48))
        | ((yy == 39) & (xx >= 49))
        | ((yy == 40) & (xx >= 50) & (xx <= 70))
        | ((yy == 41) & (xx >= 51) & (xx <= 70))
        | ((yy == 42) & (xx >= 54) & (xx <= 70))
        | ((yy >= 43) & (yy <= 45) & (xx >= 64) & (xx <= 70))
    )
    head_upper = a & head_zone & (~lower_jaw) & (~scythe_wing_far)

    # 5. `scythe_wing_main` (colossal 3-bladed crimson wing + wing-arm base)
    wing_main_zone = (
        ((yy >= 15) & (yy <= 40) & (xx <= 50))
        | ((yy == 41) & (xx <= 46))
        | ((yy >= 42) & (yy <= 50) & (xx <= 31))
    )
    scythe_wing_main = a & wing_main_zone & (~head_upper) & (~scythe_tail)

    # 6. `hind_leg_L` (back-left leg + heel spur)
    hind_L_zone = (yy >= 58) & (yy <= 75) & (xx >= 27) & (xx <= 43)
    hind_leg_L = a & hind_L_zone

    # 7. `hind_leg_R` (back-right leg & paw under belly: y=58..72, x=44..57 behind front_leg_L)
    hind_R_zone = (
        ((yy >= 58) & (yy <= 68) & (xx >= 44) & (xx <= 47))
        | ((yy >= 69) & (yy <= 72) & (xx >= 44) & (xx <= 47))
        | ((yy >= 66) & (yy <= 69) & (xx >= 53) & (xx <= 57))
        | ((yy == 70) & (xx >= 52) & (xx <= 57))
        | ((yy == 71) & (xx >= 53) & (xx <= 57))
        | ((yy == 72) & (xx >= 53) & (xx <= 56))
    )
    hind_leg_R = a & hind_R_zone

    # 8. `front_leg_L` (foreground main front-left leg + 3 silver-white claws)
    front_L_zone = (
        ((yy >= 56) & (yy <= 72) & (xx >= 48) & (xx <= 54))
        | ((yy >= 73) & (yy <= 76) & (xx >= 47) & (xx <= 55))
        | ((yy >= 77) & (yy <= 81) & (xx >= 45) & (xx <= 60))
    )
    front_leg_L = a & front_L_zone & (~hind_leg_R)

    # 9. `front_leg_R` (front-right leg stepping outward)
    front_R_zone = (yy >= 62) & (yy <= 75) & (xx >= 58) & (xx <= 76)
    front_leg_R = a & front_R_zone

    used = (
        scythe_tail
        | lower_jaw
        | scythe_wing_far
        | head_upper
        | scythe_wing_main
        | hind_leg_L
        | hind_leg_R
        | front_leg_L
        | front_leg_R
    )
    torso = a & (~used)

    return {
        "scythe_tail": scythe_tail,
        "scythe_wing_far": scythe_wing_far,
        "hind_leg_L": hind_leg_L,
        "hind_leg_R": hind_leg_R,
        "front_leg_R": front_leg_R,
        "torso": torso,
        "front_leg_L": front_leg_L,
        "scythe_wing_main": scythe_wing_main,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 10 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. `scythe_tail` (3-pronged slate-grey blade tail at rear-left: y=31..49, x=16..30)
    tail_zone = (
        ((yy >= 31) & (yy <= 47) & (xx >= 16) & (xx <= 30))
        | ((yy >= 48) & (yy <= 49) & (xx >= 22) & (xx <= 30))
    )
    scythe_tail = a & tail_zone

    # 2. `scythe_wing_far` (left-back crimson-tipped spikes: y=38..46, x=31..42)
    wing_far_zone = (
        ((yy == 38) & (xx >= 39) & (xx <= 42))
        | ((yy >= 39) & (yy <= 43) & (xx >= 31) & (xx <= 42))
        | ((yy >= 44) & (yy <= 45) & (xx >= 32) & (xx <= 42))
        | ((yy == 46) & (xx >= 33) & (xx <= 42))
    )
    scythe_wing_far = a & wing_far_zone

    # 3. `scythe_wing_main` (colossal 3-bladed crimson wing on right-back + wing-arm base)
    wing_main_zone = (
        ((yy >= 16) & (yy <= 21) & (xx >= 55))
        | ((yy >= 22) & (yy <= 26) & (xx >= 56))
        | ((yy >= 27) & (yy <= 32) & (xx >= 53))
        | ((yy >= 33) & (yy <= 39) & (xx >= 51))
        | ((yy == 40) & (xx >= 49))
        | ((yy == 41) & (xx >= 48))
        | ((yy == 42) & (xx >= 47))
        | ((yy >= 43) & (yy <= 45) & (xx >= 46))
        | ((yy >= 46) & (yy <= 51) & (xx >= 65))
    )
    scythe_wing_main = a & wing_main_zone

    # 4. `lower_jaw` (right-side jaw/cheek fur tuft under back of head: y=47..55, x=54..62)
    jaw_zone = (
        ((yy >= 47) & (yy <= 50) & (xx >= 55) & (xx <= 62))
        | ((yy >= 51) & (yy <= 55) & (xx >= 54) & (xx <= 61))
    )
    lower_jaw = a & jaw_zone & (~scythe_wing_main)

    # 5. `head_upper` (back of cranium + left & right horns: y=22..44, x=39..56)
    head_zone = (
        ((yy >= 22) & (yy <= 38) & (xx >= 39) & (xx <= 56))
        | ((yy >= 39) & (yy <= 44) & (xx >= 42) & (xx <= 50))
    )
    head_upper = a & head_zone & (~scythe_wing_main) & (~scythe_wing_far)

    # 6. `hind_leg_R` (near-center hind leg + heel spur: y=60..80, x=24..43)
    hind_R_zone = (
        ((yy == 60) & (xx >= 28) & (xx <= 31))
        | ((yy >= 61) & (yy <= 62) & (xx >= 28) & (xx <= 34))
        | ((yy >= 63) & (yy <= 64) & (xx >= 29) & (xx <= 35))
        | ((yy >= 65) & (yy <= 74) & (xx >= 28) & (xx <= 36))
        | ((yy == 75) & (xx >= 28) & (xx <= 37))
        | ((yy == 76) & (xx >= 27) & (xx <= 40))
        | ((yy == 77) & (xx >= 24) & (xx <= 42))
        | ((yy >= 78) & (yy <= 80) & (xx >= 24) & (xx <= 43))
    )
    hind_leg_R = a & hind_R_zone

    # 7. `hind_leg_L` (far-left hind leg + heel spur: y=59..77, x=12..27)
    hind_L_zone = (
        ((yy >= 59) & (yy <= 62) & (xx >= 17) & (xx <= 24))
        | ((yy >= 63) & (yy <= 64) & (xx >= 17) & (xx <= 25))
        | ((yy >= 65) & (yy <= 77) & (xx >= 12) & (xx <= 27))
    )
    hind_leg_L = a & hind_L_zone & (~hind_leg_R)

    # 8. `front_leg_L` (inner-front left leg behind hind_leg_R: y=63..76, x=35..44)
    front_L_zone = (yy >= 63) & (yy <= 76) & (xx >= 35) & (xx <= 44)
    front_leg_L = a & front_L_zone & (~hind_leg_R)

    # 9. `front_leg_R` (right front leg stepping outward + wrist spur: y=62..78, x=45..67)
    front_R_zone = (yy >= 62) & (yy <= 78) & (xx >= 45) & (xx <= 67)
    front_leg_R = a & front_R_zone

    used = (
        scythe_tail
        | scythe_wing_far
        | scythe_wing_main
        | lower_jaw
        | head_upper
        | hind_leg_L
        | hind_leg_R
        | front_leg_L
        | front_leg_R
    )
    torso = a & (~used)

    return {
        "scythe_tail": scythe_tail,
        "scythe_wing_far": scythe_wing_far,
        "hind_leg_L": hind_leg_L,
        "hind_leg_R": hind_leg_R,
        "front_leg_R": front_leg_R,
        "torso": torso,
        "front_leg_L": front_leg_L,
        "scythe_wing_main": scythe_wing_main,
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
    places onto `(cell_h, cell_w)` canvas with feet aligned to `foot_y`, and builds cushioned
    RGBA layers + anatomical joint pivots.
    IMPORTANT: Back-side layers are NEVER placed into `foreground_mask` of layers in front of them.
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

    navy_fill_m = is_navy_body_fill(canvas)
    slate_fill_m = is_slate_horn_tail_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order (back to front):
        # 1. scythe_tail -> 2. scythe_wing_far -> 3. scythe_wing_main -> 4. hind_leg_L -> 5. hind_leg_R
        # -> 6. front_leg_R -> 7. torso -> 8. front_leg_L -> 9. head_upper -> 10. lower_jaw
        layers["scythe_tail"] = build_part_layer(
            canvas,
            c_masks["scythe_tail"],
            c_masks["torso"],
            cushion_px=5,
            allowed_fill_mask=slate_fill_m,
        )

        layers["scythe_wing_far"] = build_part_layer(
            canvas,
            c_masks["scythe_wing_far"],
            c_masks["torso"] | c_masks["head_upper"],
            cushion_px=4,
            allowed_fill_mask=navy_fill_m,
        )

        layers["scythe_wing_main"] = build_part_layer(
            canvas,
            c_masks["scythe_wing_main"],
            c_masks["torso"] | c_masks["head_upper"],
            cushion_px=5,
            allowed_fill_mask=navy_fill_m,
        )

        layers["hind_leg_L"] = build_foot_stem_layer(canvas, c_masks["hind_leg_L"], c_masks["torso"], stem_px=4)
        # Bridge `hind_leg_R` behind `front_leg_L` so the far-right hind leg is a single solid limb even when `front_leg_L` swings!
        hind_R_bridged = build_part_layer(
            canvas,
            c_masks["hind_leg_R"],
            c_masks["front_leg_L"],
            cushion_px=5,
            allowed_fill_mask=navy_fill_m,
        )
        hind_R_stem = build_foot_stem_layer(canvas, c_masks["hind_leg_R"], c_masks["torso"], stem_px=4)
        m_stem = hind_R_stem[..., 3] >= 128
        hind_R_bridged[m_stem] = hind_R_stem[m_stem]
        layers["hind_leg_R"] = hind_R_bridged
        layers["front_leg_R"] = build_foot_stem_layer(canvas, c_masks["front_leg_R"], c_masks["torso"], stem_px=4)

        # `torso` cushioned ONLY under foreground layers drawn in front of `torso`
        fg_over_torso = c_masks["front_leg_L"] | c_masks["head_upper"] | c_masks["lower_jaw"]
        layers["torso"] = build_part_layer(
            canvas,
            c_masks["torso"],
            fg_over_torso,
            cushion_px=5,
            allowed_fill_mask=navy_fill_m,
        )

        layers["front_leg_L"] = build_foot_stem_layer(canvas, c_masks["front_leg_L"], c_masks["torso"], stem_px=4)

        # `head_upper`: rigid integer motion only; paint crimson roaring mouth cavity pocket + white fang
        # directly behind `lower_jaw` so opening `lower_jaw` by +1..+2px reveals a fierce roaring maw
        # that moves 100% locked with the cranium!
        head_lyr = build_part_layer(
            canvas,
            c_masks["head_upper"],
            np.zeros_like(c_masks["head_upper"]),
            cushion_px=0,
        )
        # (2026-11-04) the rectangular crimson mouth pocket is no longer painted; the mouth now opens with
        # mouth_open.open_mouth along MOUTH_LINE_FRONT_NATIVE while the jaw stays locked to the head.
        layers["head_upper"] = head_lyr

        layers["lower_jaw"] = build_part_layer(
            canvas,
            c_masks["lower_jaw"],
            np.zeros_like(c_masks["lower_jaw"]),
            cushion_px=0,
        )

        pivots = {
            "scythe_tail": to_canvas_pt(39.0, 52.0),
            "scythe_wing_far": to_canvas_pt(68.0, 47.0),
            "scythe_wing_main": to_canvas_pt(46.0, 41.0),
            "_mouth_line": [to_canvas_pt(mx, my) for mx, my in MOUTH_LINE_FRONT_NATIVE],
        }

    else:
        # Back view draw order (back to front):
        # 1. scythe_tail -> 2. scythe_wing_far -> 3. front_leg_L -> 4. hind_leg_L -> 5. front_leg_R
        # -> 6. torso -> 7. hind_leg_R -> 8. lower_jaw -> 9. head_upper -> 10. scythe_wing_main
        layers["scythe_tail"] = build_part_layer(
            canvas,
            c_masks["scythe_tail"],
            c_masks["torso"],
            cushion_px=5,
            allowed_fill_mask=slate_fill_m,
        )

        layers["scythe_wing_far"] = build_part_layer(
            canvas,
            c_masks["scythe_wing_far"],
            c_masks["torso"] | c_masks["head_upper"],
            cushion_px=4,
            allowed_fill_mask=navy_fill_m,
        )

        layers["front_leg_L"] = build_foot_stem_layer(canvas, c_masks["front_leg_L"], c_masks["torso"], stem_px=4)
        layers["hind_leg_L"] = build_foot_stem_layer(canvas, c_masks["hind_leg_L"], c_masks["torso"], stem_px=4)
        layers["front_leg_R"] = build_foot_stem_layer(canvas, c_masks["front_leg_R"], c_masks["torso"], stem_px=4)

        fg_over_torso = c_masks["hind_leg_R"] | c_masks["lower_jaw"] | c_masks["head_upper"] | c_masks["scythe_wing_main"]
        layers["torso"] = build_part_layer(
            canvas,
            c_masks["torso"],
            fg_over_torso,
            cushion_px=5,
            allowed_fill_mask=navy_fill_m,
        )

        layers["hind_leg_R"] = build_foot_stem_layer(canvas, c_masks["hind_leg_R"], c_masks["torso"], stem_px=4)

        layers["lower_jaw"] = build_part_layer(
            canvas,
            c_masks["lower_jaw"],
            c_masks["head_upper"] | c_masks["scythe_wing_main"],
            cushion_px=3,
            allowed_fill_mask=navy_fill_m,
        )

        layers["head_upper"] = build_part_layer(
            canvas,
            c_masks["head_upper"],
            c_masks["scythe_wing_main"],
            cushion_px=4,
            allowed_fill_mask=navy_fill_m,
        )

        layers["scythe_wing_main"] = build_part_layer(
            canvas,
            c_masks["scythe_wing_main"],
            np.zeros_like(c_masks["scythe_wing_main"]),
            cushion_px=0,
        )

        pivots = {
            "scythe_tail": to_canvas_pt(29.0, 48.0),
            "scythe_wing_far": to_canvas_pt(40.0, 45.0),
            "scythe_wing_main": to_canvas_pt(48.0, 43.0),
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
            out[sy:sy + 2, sx:sx + 2] = (255, 46, 97, 255)
    return out


def synthesize_absol_mega_z_frame(
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
    wmain_dx, wmain_dy = 0, 0
    wfar_dx, wfar_dy = 0, 0
    tail_dx, tail_dy = 0, 0
    hl_L_dx, hl_L_dy = 0, 0
    hl_R_dx, hl_R_dy = 0, 0
    fl_L_dx, fl_L_dy = 0, 0
    fl_R_dx, fl_R_dy = 0, 0

    wmain_ang = 0.0
    wfar_ang = 0.0
    tail_ang = 0.0

    if action == "idle":
        # Menacing disaster-beast stance modeled after Normal Absol's GIF + Mega Absol Z's crimson scythe wing:
        # 1. All 4 paws (`hind_leg_L`, `hind_leg_R`, `front_leg_L`, `front_leg_R`) stay 100% planted `(0, 0)`.
        # 2. Chest and head breathe rhythmically while `lower_jaw` opens (`+2px` peak roar at frame 4,
        #    `+1px` barks at frames 2,3,5,6,11,12) revealing the crimson mouth cavity and white fang!
        # 3. Colossal crimson `scythe_wing_main` and 3-pronged `scythe_tail` rotate/oscillate with phase lag.
        body_dy = int(round(1.2 * math.sin(phase - 0.2)))
        body_dx = int(round(0.6 * math.cos(phase) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx
        head_dy = body_dy + int(round(0.5 * math.sin(phase - 0.4)))

        if frame_idx in (2, 3, 4, 5, 6, 11, 12):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays locked to the head; mouth opens via mouth_open (no chin drop)

        wmain_dx, wmain_dy = body_dx, body_dy
        wmain_ang = 6.5 * math.sin(phase - 0.4) * facing

        wfar_dx, wfar_dy = body_dx, body_dy
        wfar_ang = -5.5 * math.sin(phase - 0.5) * facing

        tail_dx, tail_dy = body_dx, body_dy
        tail_ang = 7.0 * math.sin(phase - 0.8) * facing

    elif action == "walk":
        # High-speed 4-legged wind-gallop / diagonal trot (`quadruped_beast`):
        # Diagonal pairs `(front_leg_L + hind_leg_R)` and `(front_leg_R + hind_leg_L)` bound in opposition,
        # while the crimson `scythe_wing_main` and `scythe_tail` sweep in dynamic rotary arcs and
        # `lower_jaw` barks on each gallop stride!
        bounce = int(round(-1.6 * abs(sin1)))
        sway_x = int(round(1.5 * sin1 * facing))

        torso_dx, torso_dy = sway_x, bounce
        head_dx = sway_x + int(round(0.8 * cos1 * facing))
        head_dy = bounce + int(round(0.8 * math.sin(2.0 * phase - 0.3)))

        jaw_open = 1 if math.sin(2.0 * phase) > 0.05 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays locked to the head; mouth opens via mouth_open (no chin drop)

        wmain_dx, wmain_dy = sway_x, bounce
        wmain_ang = 9.5 * math.sin(phase - 0.3) * facing

        wfar_dx, wfar_dy = sway_x, bounce
        wfar_ang = -8.0 * math.sin(phase - 0.4) * facing

        tail_dx, tail_dy = sway_x, bounce
        tail_ang = 10.0 * math.sin(phase - 0.7) * facing

        # Diagonal quadruped pair A: front_leg_L + hind_leg_R
        fl_L_dx, fl_L_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        hl_R_dx, hl_R_dy = foot_step_offset(phase + 0.25, facing, lift_px=2, stride_px=2)

        # Diagonal quadruped pair B: front_leg_R + hind_leg_L
        fl_R_dx, fl_R_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)
        hl_L_dx, hl_L_dy = foot_step_offset(phase + math.pi + 0.25, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Mega Absol Z "Disaster Roar + Spinning Crimson Scythe-Wing Night Slash & Psycho Cut"!
        # Frames 0..3 (Low Wind-up & Ferocious Disaster Roar):
        #   - Crouches low (`+1..+2px`) and coils back (`-3px`),
        #   - Rotates the colossal crimson `scythe_wing_main` high back (`-14°`) to charge dark energy,
        #   - Opens `lower_jaw` WIDE (`+2px` ferocious roar exposing the crimson mouth and silver fang!).
        # Frames 4..9 (Spinning Scythe-Wing Slash & Pounce Lunge):
        #   - Explosive leaping pounce forward (`+6px` toward the target, `-2px` aerial lift on frames 4..5!),
        #   - Dynamically spins/whirls the colossal crimson `scythe_wing_main` (`+17° -> -14° -> +16° -> -11° -> +7°`)
        #     and 3-pronged `scythe_tail` (`+15° -> -12° -> +13° -> -9°`) in a whirlwind Night Slash / Psycho Cut!
        #   - Keeps `lower_jaw` wide open (`+2px` on frames 4..6, `+1px` on frames 7..8) in a fierce roar!
        # Frames 10..15: Follow-through landing & stance recovery.
        lunge_table = [0, -2, -3, -3, 3, 6, 6, 5, 4, 2, 0, -1, -1, 0, 0, 0]
        body_dy_table = [0, 1, 2, 1, -2, -2, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0]

        wmain_spin_table = [0.0, -7.0, -12.0, -14.0, 16.0, 18.0, -14.0, 15.0, -11.0, 7.0, -3.0, -1.5, 0.0, 0.0, 0.0, 0.0]
        wfar_spin_table = [0.0, 6.0, 10.0, 12.0, -13.0, -15.0, 12.0, -13.0, 9.0, -6.0, 2.5, 1.0, 0.0, 0.0, 0.0, 0.0]
        tail_spin_table = [0.0, -6.0, -10.0, -12.0, 14.0, 15.0, -12.0, 13.0, -9.0, 5.0, -2.5, -1.0, 0.0, 0.0, 0.0, 0.0]

        body_dx = int(round(lunge_table[frame_idx] * facing))
        body_dy = body_dy_table[frame_idx]

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + (int(round(1.0 * facing)) if frame_idx in (4, 5, 6, 7) else 0)
        head_dy = body_dy

        if frame_idx in (2, 3, 4, 5):
            jaw_open = 2
        elif frame_idx in (1, 6, 7, 8):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays locked to the head; mouth opens via mouth_open (no chin drop)

        wing_thrust_x = int(round(1.5 * facing)) if frame_idx in (4, 5, 6, 7, 8) else 0
        wmain_dx = body_dx + wing_thrust_x
        wmain_dy = body_dy
        wmain_ang = wmain_spin_table[frame_idx] * facing

        wfar_dx = body_dx + wing_thrust_x
        wfar_dy = body_dy
        wfar_ang = wfar_spin_table[frame_idx] * facing

        tail_dx = body_dx
        tail_dy = body_dy
        tail_ang = tail_spin_table[frame_idx] * facing

        # Front legs pounce and slash forward during the aerial strike (frames 4..7)
        if frame_idx in (4, 5, 6, 7):
            pounce_x = int(round((lunge_table[frame_idx] + 1) * facing))
            fl_L_dx, fl_L_dy = pounce_x, body_dy - 1
            fl_R_dx, fl_R_dy = pounce_x, body_dy - 1
            hl_L_dx, hl_L_dy = int(round(body_dx * 0.85)), max(0, body_dy)
            hl_R_dx, hl_R_dy = int(round(body_dx * 0.85)), max(0, body_dy)
        else:
            fl_L_dx, fl_L_dy = int(round(body_dx * 0.85)), 0
            fl_R_dx, fl_R_dy = int(round(body_dx * 0.85)), 0
            hl_L_dx, hl_L_dy = int(round(body_dx * 0.75)), 0
            hl_R_dx, hl_R_dy = int(round(body_dx * 0.75)), 0

    elif action == "hurt":
        # Knockback recoil + hit-shake + ferocious roar (`+2px` jaw open) + crimson wing jolted back + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-3.5 * env * facing)) + jit
        body_dy = int(round(-1.0 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-1.5 * env))

        jaw_open = 1 if 1 <= frame_idx <= 7 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays locked to the head; mouth opens via mouth_open (no chin drop)

        wmain_dx = body_dx + int(round(-1.0 * env * facing))
        wmain_dy = int(round(-1.5 * env))
        wmain_ang = -11.0 * env * facing

        wfar_dx = body_dx
        wfar_dy = body_dy
        wfar_ang = 9.0 * env * facing

        tail_dx = body_dx
        tail_dy = body_dy
        tail_ang = -10.0 * env * facing

        slide_x = int(round(body_dx * 0.75))
        hl_L_dx, hl_L_dy = slide_x, 0
        hl_R_dx, hl_R_dy = slide_x, 0
        fl_L_dx, fl_L_dy = slide_x, int(round(-0.8 * env))
        fl_R_dx, fl_R_dy = slide_x, int(round(-0.8 * env))

    tail_rot = rotate_layer_nearest(layers["scythe_tail"], tail_ang, pivots["scythe_tail"])
    wfar_rot = rotate_layer_nearest(layers["scythe_wing_far"], wfar_ang, pivots["scythe_wing_far"])
    wmain_rot = rotate_layer_nearest(layers["scythe_wing_main"], wmain_ang, pivots["scythe_wing_main"])

    if not is_back:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(wfar_rot, wfar_dx, wfar_dy),
            shift_layer_int(wmain_rot, wmain_dx, wmain_dy),
            shift_layer_int(layers["hind_leg_L"], hl_L_dx, hl_L_dy),
            shift_layer_int(layers["hind_leg_R"], hl_R_dx, hl_R_dy),
            shift_layer_int(layers["front_leg_R"], fl_R_dx, fl_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["front_leg_L"], fl_L_dx, fl_L_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
        ]
    else:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(wfar_rot, wfar_dx, wfar_dy),
            shift_layer_int(layers["front_leg_L"], fl_L_dx, fl_L_dy),
            shift_layer_int(layers["hind_leg_L"], hl_L_dx, hl_L_dy),
            shift_layer_int(layers["front_leg_R"], fl_R_dx, fl_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["hind_leg_R"], hl_R_dx, hl_R_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(wmain_rot, wmain_dx, wmain_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    if (not is_back) and jaw_open > 0 and "_mouth_line" in pivots:
        m_line = mouth_open.transform_points(pivots["_mouth_line"], dx=head_dx, dy=head_dy)
        out = mouth_open.open_mouth(out, m_line, amount=jaw_open, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=0.6)
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

    # Tight bounds of front (58x67) and back (73x65)
    max_w, max_h = 73, 67
    cell_w = 96
    cell_h = 96
    foot_y = 84

    dir_rigs: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]], float, bool]] = {}
    for d_name in DIRS:
        use_back = d_name in ("N", "NE", "NW")
        src_96 = back_raw if use_back else front_raw
        if not use_back:
            # Native front.png faces slightly right (SE/S), so flip_x for SW/W to face left
            flip_x = d_name in ("SW", "W")
            facing = -1.0 if flip_x else 1.0
        else:
            # Native back.png faces slightly right-up (NE/N), so flip_x for NW to face left-up
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
                frame_rgba = synthesize_absol_mega_z_frame(
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

    # Replace broken 20-frame warped `front.gif` / `back.gif` (and reverse GIFs) with our
    # zero-distortion 16-frame `idle` S and N loops!
    idle_sheet = sheets_by_action["idle"]
    front_frames = [idle_sheet[0 * cell_h:1 * cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    front_rev_frames = [f[:, ::-1, :].copy() for f in front_frames]
    back_frames = [idle_sheet[4 * cell_h:5 * cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    back_rev_frames = [f[:, ::-1, :].copy() for f in back_frames]

    save_transparent_gif(front_frames, MEGA_DIR / "front.gif", duration_ms=75)
    save_transparent_gif(front_rev_frames, MEGA_DIR / "front-reverse.gif", duration_ms=75)
    save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
    save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)

    meta = {
        "generator": "forge_absol_mega_z_actions",
        "key": "0359_absol-mega-z",
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

    print(f"Built 0359_absol-mega-z -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
