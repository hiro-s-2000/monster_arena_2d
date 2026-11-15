#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Medicham (`0308_medicham-mega`) modeled after Mega Medicham's official 92-frame
sprite animation (`0308_medicham/mega/medicham-mega/front.gif`) and anatomical design (`front.png` & `back.png`):
  - Exact 10-layer anatomical rigging with true joint pivots (4 psychic aura arm mandala shoulder pivots +
    hip knee-kick socket):
      1. `aura_arms_upper_L` (upper-left floating white psychic aura giant arm — pivoted at the upper-left
                              dorsal chakra socket via `rotate_layer_nearest` for smooth mandala/wheel rotation!)
      2. `aura_arms_upper_R` (upper-right floating white psychic aura giant arm — pivoted at the upper-right
                              dorsal chakra socket via `rotate_layer_nearest` for mandala/wheel rotation!)
      3. `aura_arms_lower_L` (lower-left floating white psychic aura giant arm — phase-shifted wheel rotation
                              and thousand-palm flurry rush via `rotate_layer_nearest`!)
      4. `aura_arms_lower_R` (lower-right floating white psychic aura giant arm — phase-shifted wheel rotation
                              and thousand-palm flurry rush via `rotate_layer_nearest`!)
      5. `leg_R` (right pink balloon-pants leg & foot — grounded pivot foot `(0,0)` on `front.png`, or raised
                  knee-kick leg on `back.png`)
      6. `sash_ribbons` (golden waist/hip sash ribbons draping over both upper thighs — flutter dynamically
                         during levitation, yoga steps, and spinning High Jump Kick!)
      7. `leg_L` (left pink balloon-pants leg & foot — raised yoga knee-kick leg on `front.png`, or grounded
                  pivot leg on `back.png` — pivoted at the hip socket for spinning High Jump Kicks!)
      8. `torso` (central white body, praying/martial-arts hands, golden wristbands, and dark-crimson mouth
                  interior pocket behind `mouth_jaw`)
      9. `mouth_jaw` (lower red lip & chin — opens `+1px` for rhythmic yoga kiai breathing in `idle`/`walk`
                      and `+2px` wide for a fierce martial-arts roar during `attack` wind-up, spinning
                      High Jump Kick + 4-arm mandala rush, and `hurt` reaction!)
     10. `head` (white turban, red crest plume, golden forehead gem, fierce blue eyes, and upper lip —
                 100% rigid integer motion `shift_layer_int` so the eyes and face never deform by a single pixel!)
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
MEGA_DIR = ROOT / "monsterSprite/gen3/0308_medicham/mega/medicham-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0308_medicham/mega/medicham-mega"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16
VERSION = "medicham_mega_v1m"

# Closed mouth line on native `front.png` (96x96), ordered hinge (right/back of the face) -> lip tip (left/front).
# The pursed pink lips sit at x=42..45, y=40..43; the line runs along the upper-lip/lower-lip seam (y=41).
# `mouth_open.open_mouth` carves a small lens just below it (inside the lips), both corners stay closed.
MOUTH_LINE_NATIVE = [(46.0, 41.0), (41.0, 41.0)]
MOUTH_DEEP = (16, 16, 16)      # darkest palette colour (outline)
MOUTH_INNER = (131, 65, 74)    # existing dark-crimson mouth colour in the palette


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
        | ((r == 82) & (g == 65) & (b == 74))
    )


def is_aura_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 222) & (g == 222) & (b == 222))
        | ((r == 206) & (g == 206) & (b == 206))
        | ((r == 172) & (g == 164) & (b == 172))
        | ((r == 231) & (g == 227) & (b == 231))
        | ((r == 189) & (g == 175) & (b == 173))
    )


def is_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    return a & (~is_dark_outline(rgba))


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
    Segments native 96x96 `front.png` (facing left-front in floating yoga stance) into 10 exact anatomical masks:
      1. `aura_arms_upper_L`: Upper-left floating white psychic aura arm (`x=10..35, y=15..37`).
      2. `aura_arms_upper_R`: Upper-right floating white psychic aura arm (`x=58..84, y=15..38`).
      3. `aura_arms_lower_L`: Lower-left floating white psychic aura arm (`x=8..38, y=36..44`).
      4. `aura_arms_lower_R`: Lower-right floating white psychic aura arm (`x=52..86, y=35..44`).
      5. `leg_R`: Grounded right pink balloon-pants leg & grounded foot (`x=43..59, y=60..81`).
      6. `sash_ribbons`: Golden waist/hip sash ribbons draping over both upper thighs (`x=35..56, y=53..63`).
      7. `leg_L`: Raised left pink balloon-pants knee & left foot tip (`x=29..44, y=56..75`).
      8. `torso`: Central white upper body, praying hands, and golden wristbands (`x=30..59, y=44..56`).
      9. `mouth_jaw`: Lower red lip & chin outline (`x=42..46, y=42..44`).
     10. `head`: White turban, red crest plume, gold forehead gem, blue eyes, and upper lip (`x=32..59, y=14..44`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. aura_arms_upper_L
    aura_upper_L = a & (
        ((yy >= 15) & (yy <= 32) & (xx <= 30))
        | ((yy == 33) & (xx <= 31))
        | ((yy == 34) & (xx <= 31))
        | ((yy == 35) & (xx <= 32))
        | ((yy == 36) & (xx >= 30) & (xx <= 34))
        | ((yy == 37) & (xx >= 33) & (xx <= 35))
    )

    # 2. aura_arms_upper_R
    aura_upper_R = a & (
        ((yy >= 15) & (yy <= 34) & (xx >= 62))
        | ((yy == 35) & (xx >= 61) & (xx <= 63))
        | ((yy == 36) & (xx >= 60) & (xx <= 62))
        | ((yy == 37) & (xx >= 59) & (xx <= 61))
        | ((yy == 38) & (xx >= 58) & (xx <= 60))
    )

    # 3. aura_arms_lower_L
    aura_lower_L = a & (~aura_upper_L) & (
        ((yy == 36) & (xx <= 25))
        | ((yy >= 37) & (yy <= 38) & (xx <= 32))
        | ((yy >= 39) & (yy <= 44) & (xx <= 38))
    )

    # 4. aura_arms_lower_R
    aura_lower_R = a & (~aura_upper_R) & (
        ((yy == 35) & (xx >= 72))
        | ((yy == 36) & (xx >= 69))
        | ((yy == 37) & (xx >= 67))
        | ((yy == 38) & (xx >= 65))
        | ((yy >= 39) & (yy <= 44) & (xx >= 52))
    )

    # 5. mouth_jaw (lower red lip + chin outline for kiai/roar)
    mouth_jaw = a & (
        ((yy == 42) & (xx >= 43) & (xx <= 45))
        | ((yy == 43) & (xx >= 43) & (xx <= 45))
        | ((yy == 44) & (xx >= 44) & (xx <= 45))
    )

    # 6. head (white turban, red crest plume, gold forehead gem, blue eyes, upper lip)
    head_zone = (
        (yy <= 43)
        | ((yy == 44) & (xx >= 43) & (xx <= 47))
    )
    head = (
        a
        & head_zone
        & (~aura_upper_L)
        & (~aura_upper_R)
        & (~aura_lower_L)
        & (~aura_lower_R)
        & (~mouth_jaw)
    )

    # 7. leg_L (raised left pink balloon-pants leg + yellow patch + blue drops + left foot tip at y=71..75)
    leg_L_zone = (
        ((yy == 56) & (xx <= 37))
        | ((yy == 57) & (xx <= 40))
        | ((yy == 58) & (xx <= 41))
        | ((yy >= 59) & (yy <= 60) & (xx <= 42))
        | ((yy >= 61) & (yy <= 63) & (xx <= 43))
        | ((yy >= 64) & (yy <= 66) & (xx <= 44))
        | ((yy >= 67) & (yy <= 68) & (xx <= 43))
        | ((yy >= 69) & (yy <= 74) & (xx <= 42))
        | ((yy == 75) & (xx <= 41))
    )
    leg_L = a & leg_L_zone

    # 8. sash_ribbons (golden waist/hip sash ribbons draping over both upper thighs y=53..63)
    sash_zone = (
        ((yy == 53) & (xx >= 37) & (xx <= 40))
        | ((yy == 54) & (xx >= 36) & (xx <= 42))
        | ((yy == 55) & ((xx <= 43) | (xx >= 48)))
        | ((yy == 56) & (xx >= 38) & ((xx <= 44) | (xx >= 46)))
        | ((yy >= 57) & (yy <= 59) & (xx >= 41) & (xx <= 56))
        | ((yy == 60) & (xx >= 43) & ((xx <= 48) | (xx >= 53)))
        | ((yy == 61) & (((xx >= 44) & (xx <= 46)) | (xx >= 54)))
        | ((yy == 62) & (((xx >= 44) & (xx <= 46)) | ((xx >= 55) & (xx <= 56))))
        | ((yy == 63) & (xx >= 44) & (xx <= 45))
    )
    sash_ribbons = a & sash_zone & (~leg_L)

    # 9. leg_R (grounded right pink balloon-pants leg + grounded right foot y=60..81)
    leg_R = a & (yy >= 60) & (~leg_L) & (~sash_ribbons)

    # 10. torso (neck, upper body, praying hands & golden wristbands y=44..56)
    used = (
        aura_upper_L
        | aura_upper_R
        | aura_lower_L
        | aura_lower_R
        | mouth_jaw
        | head
        | leg_L
        | sash_ribbons
        | leg_R
    )
    torso = a & (~used)

    return {
        "aura_arms_upper_L": aura_upper_L,
        "aura_arms_upper_R": aura_upper_R,
        "aura_arms_lower_L": aura_lower_L,
        "aura_arms_lower_R": aura_lower_R,
        "leg_R": leg_R,
        "sash_ribbons": sash_ribbons,
        "leg_L": leg_L,
        "torso": torso,
        "mouth_jaw": mouth_jaw,
        "head": head,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` (facing right-rear in 3/4 rear yoga stance) into matching 10 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. aura_arms_upper_L
    aura_upper_L = a & (
        ((yy >= 15) & (yy <= 30) & (xx <= 36))
        | ((yy == 31) & (xx <= 38))
        | ((yy == 32) & (xx <= 40))
        | ((yy == 33) & (xx <= 40))
        | ((yy == 34) & (xx <= 41))
        | ((yy == 35) & (xx <= 43))
    )

    # 2. aura_arms_upper_R
    aura_upper_R = a & (
        ((yy >= 14) & (yy <= 26) & (xx >= 62))
        | ((yy >= 27) & (yy <= 29) & (xx >= 60))
        | ((yy == 30) & (xx >= 59))
        | ((yy == 31) & (xx >= 58))
        | ((yy == 32) & (xx >= 57))
        | ((yy == 33) & (xx >= 56))
        | ((yy == 34) & (xx >= 54))
    )

    # 3. aura_arms_lower_L
    aura_lower_L = a & (
        ((yy >= 38) & (yy <= 44) & (xx <= 37))
        | ((yy == 45) & (xx <= 43))
        | ((yy == 46) & (xx <= 44))
        | ((yy == 47) & (xx <= 45))
        | ((yy == 48) & (xx <= 43))
    )

    # 4. aura_arms_lower_R
    aura_lower_R = a & (
        ((yy >= 38) & (yy <= 44) & (xx >= 56))
        | ((yy == 45) & (xx >= 52))
        | ((yy == 46) & (xx >= 51))
        | ((yy == 47) & (((xx >= 51) & (xx <= 60)) | (xx >= 74)))
    )

    # 5. head (turban, red crest plume, side turban ears y=14..44)
    head = a & (yy <= 44) & (~aura_upper_L) & (~aura_upper_R) & (~aura_lower_L) & (~aura_lower_R)

    # 6. sash_ribbons (golden waist sash ribbons on back y=54..65)
    sash_zone = (
        ((yy == 54) & (xx >= 53))
        | ((yy == 55) & ((xx <= 46) | (xx >= 52)))
        | ((yy == 56) & ((xx <= 47) | (xx >= 51)))
        | ((yy == 57) & ((xx <= 47) | (xx >= 50)))
        | ((yy == 58) & (xx <= 60) & ((xx <= 48) | (xx >= 50)))
        | ((yy == 59) & (xx <= 56))
        | ((yy == 60) & (xx <= 54))
        | ((yy == 61) & (xx <= 52))
        | ((yy == 62) & (xx <= 50))
        | ((yy == 63) & (xx <= 49))
        | ((yy == 64) & (xx >= 40) & (xx <= 48))
        | ((yy == 65) & (xx >= 40) & (xx <= 45))
    )
    sash_ribbons = a & sash_zone

    # 7. leg_R (raised right balloon leg + right foot tip at y=73..76, x=51..64)
    leg_R_zone = (
        ((yy >= 58) & (yy <= 60) & (xx >= 55))
        | ((yy == 61) & (xx >= 53))
        | ((yy >= 62) & (yy <= 68) & (xx >= 51))
        | ((yy >= 69) & (yy <= 70) & (xx >= 52))
        | ((yy >= 71) & (yy <= 76) & (xx >= 53))
    )
    leg_R = a & leg_R_zone & (~sash_ribbons)

    # 8. leg_L (grounded left balloon leg + grounded left foot at y=63..84)
    leg_L = a & (yy >= 62) & (~sash_ribbons) & (~leg_R)

    # 9. torso (back spine, waist, and folded elbows with gold wristbands y=45..58)
    empty = np.zeros_like(a)
    used = (
        aura_upper_L
        | aura_upper_R
        | aura_lower_L
        | aura_lower_R
        | head
        | sash_ribbons
        | leg_R
        | leg_L
    )
    torso = a & (~used)

    return {
        "aura_arms_upper_L": aura_upper_L,
        "aura_arms_upper_R": aura_upper_R,
        "aura_arms_lower_L": aura_lower_L,
        "aura_arms_lower_R": aura_lower_R,
        "leg_R": leg_R,
        "leg_L": leg_L,
        "sash_ribbons": sash_ribbons,
        "torso": torso,
        "mouth_jaw": empty,
        "head": head,
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

    aura_fill_m = is_aura_fill(canvas)
    body_fill_m = is_body_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order (strictly Back-to-Front):
        # 1. aura_arms_upper_L -> 2. aura_arms_upper_R -> 3. aura_arms_lower_L -> 4. aura_arms_lower_R ->
        # 5. leg_R (grounded) -> 6. sash_ribbons -> 7. leg_L (raised knee) ->
        # 8. torso -> 9. mouth_jaw -> 10. head
        fg_over_auL = c_masks["head"]
        u_auL = ndi.binary_erosion(c_masks["aura_arms_upper_L"] | fg_over_auL, iterations=1)
        layers["aura_arms_upper_L"] = build_part_layer(
            canvas,
            c_masks["aura_arms_upper_L"],
            u_auL & fg_over_auL,
            cushion_px=3,
            allowed_fill_mask=aura_fill_m,
        )

        fg_over_auR = c_masks["head"]
        u_auR = ndi.binary_erosion(c_masks["aura_arms_upper_R"] | fg_over_auR, iterations=1)
        layers["aura_arms_upper_R"] = build_part_layer(
            canvas,
            c_masks["aura_arms_upper_R"],
            u_auR & fg_over_auR,
            cushion_px=3,
            allowed_fill_mask=aura_fill_m,
        )

        layers["aura_arms_lower_L"] = build_part_layer(
            canvas,
            c_masks["aura_arms_lower_L"],
            np.zeros_like(c_masks["aura_arms_lower_L"]),
            cushion_px=0,
        )
        layers["aura_arms_lower_R"] = build_part_layer(
            canvas,
            c_masks["aura_arms_lower_R"],
            np.zeros_like(c_masks["aura_arms_lower_R"]),
            cushion_px=0,
        )

        fg_over_legR = c_masks["sash_ribbons"] | c_masks["leg_L"] | c_masks["torso"]
        u_legR = ndi.binary_erosion(c_masks["leg_R"] | fg_over_legR, iterations=1)
        layers["leg_R"] = build_part_layer(
            canvas,
            c_masks["leg_R"],
            u_legR & fg_over_legR,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_sash = c_masks["leg_L"] | c_masks["torso"]
        u_sash = ndi.binary_erosion(c_masks["sash_ribbons"] | fg_over_sash, iterations=1)
        layers["sash_ribbons"] = build_part_layer(
            canvas,
            c_masks["sash_ribbons"],
            u_sash & fg_over_sash,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        layers["leg_L"] = build_part_layer(
            canvas,
            c_masks["leg_L"],
            np.zeros_like(c_masks["leg_L"]),
            cushion_px=0,
        )

        # IMPORTANT: `torso` is in front of `sash_ribbons`/`leg_L`/`leg_R`, so its `foreground_mask`
        # ONLY contains `mouth_jaw` and `head`!
        fg_over_torso = c_masks["mouth_jaw"] | c_masks["head"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        # (2026-11-04) No rectangular mouth-interior pocket any more: the jaw no longer drops; the mouth opens
        # with `mouth_open.open_mouth` along MOUTH_LINE_NATIVE on the composed frame instead.
        layers["torso"] = torso_lyr

        layers["mouth_jaw"] = build_part_layer(
            canvas,
            c_masks["mouth_jaw"],
            np.zeros_like(c_masks["mouth_jaw"]),
            cushion_px=0,
        )
        layers["head"] = build_part_layer(
            canvas,
            c_masks["head"],
            np.zeros_like(c_masks["head"]),
            cushion_px=0,
        )

        pivots = {
            "aura_arms_upper_L": to_canvas_pt(35.0, 36.0),
            "aura_arms_upper_R": to_canvas_pt(59.0, 37.0),
            "aura_arms_lower_L": to_canvas_pt(36.0, 42.0),
            "aura_arms_lower_R": to_canvas_pt(55.0, 42.0),
            "leg_L": to_canvas_pt(41.0, 59.0),
            "sash_ribbons": to_canvas_pt(45.0, 56.0),
            # list of points (not a pivot): closed mouth line in cell coords, mirrored for flipped views
            "mouth_line": [to_canvas_pt(mx, my) for mx, my in MOUTH_LINE_NATIVE],
        }

    else:
        # Back view draw order (strictly Back-to-Front):
        # 1. aura_arms_upper_L -> 2. aura_arms_upper_R -> 3. aura_arms_lower_L -> 4. aura_arms_lower_R ->
        # 5. leg_R (raised) -> 6. leg_L (grounded) -> 7. sash_ribbons -> 8. torso -> 9. head
        fg_over_auL = c_masks["head"]
        u_auL = ndi.binary_erosion(c_masks["aura_arms_upper_L"] | fg_over_auL, iterations=1)
        layers["aura_arms_upper_L"] = build_part_layer(
            canvas,
            c_masks["aura_arms_upper_L"],
            u_auL & fg_over_auL,
            cushion_px=2,
            allowed_fill_mask=aura_fill_m,
        )

        fg_over_auR = c_masks["head"]
        u_auR = ndi.binary_erosion(c_masks["aura_arms_upper_R"] | fg_over_auR, iterations=1)
        layers["aura_arms_upper_R"] = build_part_layer(
            canvas,
            c_masks["aura_arms_upper_R"],
            u_auR & fg_over_auR,
            cushion_px=2,
            allowed_fill_mask=aura_fill_m,
        )

        fg_over_alL = c_masks["torso"]
        u_alL = ndi.binary_erosion(c_masks["aura_arms_lower_L"] | fg_over_alL, iterations=1)
        layers["aura_arms_lower_L"] = build_part_layer(
            canvas,
            c_masks["aura_arms_lower_L"],
            u_alL & fg_over_alL,
            cushion_px=4,
            allowed_fill_mask=aura_fill_m,
        )

        fg_over_alR = c_masks["torso"]
        u_alR = ndi.binary_erosion(c_masks["aura_arms_lower_R"] | fg_over_alR, iterations=1)
        layers["aura_arms_lower_R"] = build_part_layer(
            canvas,
            c_masks["aura_arms_lower_R"],
            u_alR & fg_over_alR,
            cushion_px=4,
            allowed_fill_mask=aura_fill_m,
        )

        fg_over_legR = c_masks["leg_L"] | c_masks["sash_ribbons"]
        u_legR = ndi.binary_erosion(c_masks["leg_R"] | fg_over_legR, iterations=1)
        layers["leg_R"] = build_part_layer(
            canvas,
            c_masks["leg_R"],
            u_legR & fg_over_legR,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_legL = c_masks["sash_ribbons"]
        u_legL = ndi.binary_erosion(c_masks["leg_L"] | fg_over_legL, iterations=1)
        layers["leg_L"] = build_part_layer(
            canvas,
            c_masks["leg_L"],
            u_legL & fg_over_legL,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_sash = c_masks["torso"]
        u_sash = ndi.binary_erosion(c_masks["sash_ribbons"] | fg_over_sash, iterations=1)
        layers["sash_ribbons"] = build_part_layer(
            canvas,
            c_masks["sash_ribbons"],
            u_sash & fg_over_sash,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_torso = c_masks["head"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        layers["torso"] = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        layers["mouth_jaw"] = np.zeros_like(canvas)
        layers["head"] = build_part_layer(
            canvas,
            c_masks["head"],
            np.zeros_like(c_masks["head"]),
            cushion_px=0,
        )

        pivots = {
            "aura_arms_upper_L": to_canvas_pt(42.0, 34.0),
            "aura_arms_upper_R": to_canvas_pt(55.0, 33.0),
            "aura_arms_lower_L": to_canvas_pt(43.0, 46.0),
            "aura_arms_lower_R": to_canvas_pt(52.0, 46.0),
            "leg_R": to_canvas_pt(54.0, 61.0),
            "sash_ribbons": to_canvas_pt(50.0, 57.0),
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
        (mid_x - 16 - drift, max(2, top_y + 14 - drift // 2)),
        (mid_x + 16 + drift, max(2, top_y + 16 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (246, 222, 131, 255)
    return out


def synthesize_medicham_mega_frame(
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
    sash_dx, sash_dy = 0, 0
    kick_leg_dx, kick_leg_dy = 0, 0
    pivot_leg_dx, pivot_leg_dy = 0, 0

    auL_dx, auL_dy = 0, 0
    auR_dx, auR_dy = 0, 0
    alL_dx, alL_dy = 0, 0
    alR_dx, alR_dy = 0, 0

    auL_ang = 0.0
    auR_ang = 0.0
    alL_ang = 0.0
    alR_ang = 0.0
    kick_leg_ang = 0.0
    sash_ang = 0.0

    # Sign helper so left/right aura arms flare/rotate symmetrically outward/upward
    # On unflipped native canvas (`facing = -1.0` for front, `+1.0` for back),
    # `aura_arms_upper_L` & `aura_arms_lower_L` extend to the LEFT (`-x`) when not flipped,
    # and to the RIGHT (`+x`) when `flip_x=True`.
    # Specifically, for front: `facing == -1.0` means `flip_x == False`; `facing == +1.0` means `flip_x == True`.
    # For back: `facing == +1.0` means `flip_x == False`; `facing == -1.0` means `flip_x == True`.
    # Thus `side_sign = -facing` on front, and `+facing` on back (`+1` when unflipped, `-1` when flipped).
    side_sign = facing if is_back else (-facing)

    if action == "idle":
        # Modeled directly after Mega Medicham's official 92f GIF (`front.gif` loop f=0..15):
        # 1. Grounded pivot foot (`leg_R` on front, `leg_L` on back) stays 100% planted on the ground `(0, 0)`.
        # 2. Meditative yoga breathing + kiai chant (`mouth_jaw` opens `+1px`).
        # 3. 4 psychic aura arms (`aura_arms_upper_L/R`, `aura_arms_lower_L/R`) smoothly rotate like a
        #    thousand-armed mandala wheel around the dorsal shoulder chakra pivots with phase offset!
        body_dy = int(round(1.3 * math.sin(phase - 0.2)))
        body_dx = int(round(0.6 * max(0.0, math.sin(phase - 0.2)) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        jaw_open = 1 if math.sin(phase - 0.1) > 0.15 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the face; mouth opens via mouth_open

        sash_dx = body_dx
        sash_dy = body_dy + int(round(0.8 * math.sin(phase - 0.6)))

        kick_leg_dx = body_dx
        kick_leg_dy = body_dy + int(round(-1.2 * math.sin(phase - 0.3)))
        kick_leg_ang = 3.5 * math.sin(phase - 0.3) * (-facing)

        # Upper aura arms rotate in a rhythmic wheel arc (`±8.5°`), lower aura arms follow with phase lag (`±9.5°`)
        au_wave = math.sin(phase - 0.4)
        al_wave = math.sin(phase - 1.3)
        auL_ang = 8.5 * au_wave * side_sign
        auR_ang = -8.5 * au_wave * side_sign
        alL_ang = 9.5 * al_wave * side_sign
        alR_ang = -9.5 * al_wave * side_sign

        auL_dx, auL_dy = body_dx, body_dy + int(round(1.0 * au_wave))
        auR_dx, auR_dy = body_dx, body_dy + int(round(-1.0 * au_wave))
        alL_dx, alL_dy = body_dx, body_dy + int(round(1.0 * al_wave))
        alR_dx, alR_dy = body_dx, body_dy + int(round(-1.0 * al_wave))

    elif action == "walk":
        # Graceful aerial yoga martial-arts step + 4-arm mandala wheel swing + kiai chant:
        bounce = int(round(-1.6 * abs(sin1)))
        sway_x = int(round(1.4 * sin1 * facing))

        torso_dx, torso_dy = sway_x, bounce
        head_dx = sway_x + int(round(0.8 * cos1 * facing))
        head_dy = bounce + int(round(0.8 * math.sin(2.0 * phase - 0.3)))

        jaw_open = 1 if math.sin(2.0 * phase) > 0.15 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the face; mouth opens via mouth_open

        sash_dx = sway_x
        sash_dy = bounce + int(round(1.2 * math.sin(2.0 * phase - 0.6)))
        sash_ang = 2.5 * sin1 * facing

        kick_leg_dx = sway_x + int(round(1.6 * cos1 * facing))
        kick_leg_dy = bounce + int(round(-1.8 * sin1))
        kick_leg_ang = 6.0 * sin1 * (-facing)

        pivot_leg_dx, pivot_leg_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)

        # Asymmetric mandala wheel rotation during yoga footwork
        auL_ang = 11.0 * math.sin(phase - 0.2) * side_sign
        auR_ang = 11.0 * math.cos(phase - 0.2) * (-side_sign)
        alL_ang = 12.0 * math.sin(phase - 1.1) * side_sign
        alR_ang = 12.0 * math.cos(phase - 1.1) * (-side_sign)

        auL_dx, auL_dy = sway_x, bounce + int(round(1.2 * sin1))
        auR_dx, auR_dy = sway_x, bounce + int(round(-1.2 * sin1))
        alL_dx, alL_dy = sway_x, bounce + int(round(1.2 * cos1))
        alR_dx, alR_dy = sway_x, bounce + int(round(-1.2 * cos1))

    elif action == "attack":
        # Signature Mega Medicham "Thousand-Arm Asura Mandala Wheel Spin + Fierce Kiai Roar +
        # Spinning High Jump Knee Kick (`+6px` lunge)"!
        # Modeled after the climax of official `front.gif` (`f=68..88`) + martial-arts lunge:
        # Frames 0..3 (Wind-up & Mandala Wheel Charge + Kiai Roar):
        #   - Coils back and low (`-3px` back, `+2px` crouch),
        #   - Opens `mouth_jaw` wide (`+2px` kiai roar!),
        #   - Spreads all 4 psychic aura arms wide into an X-mandala wheel (`±16°` rotation).
        # Frames 4..9 (Thousand-Arm Wheel Spin Rush & Spinning High Jump Knee Kick `+6px` Lunge):
        #   - Explosive airborne forward lunge (`+6px` forward, `-2px` jump!),
        #   - Raised knee (`leg_L` on front, `leg_R` on back) executes a high-speed spinning High Jump Kick
        #     (`+14° -> -9° -> +12°` hip rotation + `+7px` knee thrust!),
        #   - All 4 psychic aura arms execute a full high-speed Mandala Wheel rotation (`±20°` alternating
        #     wheel spin + forward palm-strike rush `+7px`!),
        #   - `mouth_jaw` roars wide (`+2px`) on impact!
        # Frames 10..15: Graceful yoga landing & residual aura wheel follow-through.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.60 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        body_dx = int(round(6.0 * k * facing))
        body_dy_table = [0, 1, 2, 2, -1, -2, -2, -1, 0, 1, 1, 0, 0, 0, 0, 0]
        body_dy = body_dy_table[frame_idx]

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = int(round(6.0 * k * facing))
        head_dy = body_dy

        if frame_idx in (2, 3, 4, 5, 6):
            jaw_open = 2
        elif frame_idx in (1, 8, 9):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the face; mouth opens via mouth_open

        sash_dx = body_dx
        sash_dy = body_dy + (-1 if frame_idx in (4, 5, 6, 7) else 0)
        sash_ang_table = [0.0, -3.0, -5.0, -6.0, 5.0, 8.0, 6.0, -5.0, 4.0, 2.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        sash_ang = sash_ang_table[frame_idx] * (-facing)

        # Spinning High Jump Knee Kick trajectory
        kick_ang_table = [0.0, -5.0, -8.5, -10.5, 9.5, 14.0, 11.5, -8.5, 10.5, 5.5, -2.5, -1.0, 0.0, 0.0, 0.0, 0.0]
        kick_dy_table = [0, 1, 2, 2, -2, -3, -3, 1, -2, -1, 0, 0, 0, 0, 0, 0]
        kick_leg_dx = int(round(7.0 * k * facing))
        kick_leg_dy = body_dy + kick_dy_table[frame_idx]
        kick_leg_ang = kick_ang_table[frame_idx] * (-facing)

        # Grounded leg leaps into the air during the High Jump Kick (frames 4..8)
        pivot_dy_table = [0, 0, 0, 0, -1, -3, -3, -2, -1, 0, 0, 0, 0, 0, 0, 0]
        pivot_leg_dx = int(round(5.2 * k * facing))
        pivot_leg_dy = pivot_dy_table[frame_idx]

        # 4 Psychic Aura Arms: Mandala Wheel Charge (0..3) -> High-Speed Asura Wheel Spin & Rush (4..9) -> Recovery (10..15)
        auL_spin = [0.0, -8.0, -14.0, -18.0, 14.0, 20.0, -16.0, 18.0, -14.0, 9.0, -4.0, -1.5, 0.0, 0.0, 0.0, 0.0]
        auR_spin = [0.0, 8.0, 14.0, 18.0, -14.0, -20.0, 16.0, -18.0, 14.0, -9.0, 4.0, 1.5, 0.0, 0.0, 0.0, 0.0]
        alL_spin = [0.0, 7.0, 13.0, 17.0, -16.0, 19.0, -18.0, 15.0, -12.0, 8.0, -3.5, -1.0, 0.0, 0.0, 0.0, 0.0]
        alR_spin = [0.0, -7.0, -13.0, -17.0, 16.0, -19.0, 18.0, -15.0, 12.0, -8.0, 3.5, 1.0, 0.0, 0.0, 0.0, 0.0]

        auL_ang = auL_spin[frame_idx] * side_sign
        auR_ang = auR_spin[frame_idx] * side_sign
        alL_ang = alL_spin[frame_idx] * side_sign
        alR_ang = alR_spin[frame_idx] * side_sign

        aura_rush_dx = int(round(7.2 * k * facing))
        au_dy_lift = -2 if frame_idx in (2, 3, 5, 6, 8) else (1 if frame_idx in (4, 7) else 0)
        al_dy_lift = 2 if frame_idx in (2, 3, 5, 7) else (-2 if frame_idx in (4, 6, 8) else 0)

        auL_dx, auL_dy = aura_rush_dx, body_dy + au_dy_lift
        auR_dx, auR_dy = aura_rush_dx, body_dy + au_dy_lift
        alL_dx, alL_dy = aura_rush_dx, body_dy + al_dy_lift
        alR_dx, alR_dy = aura_rush_dx, body_dy + al_dy_lift

    elif action == "hurt":
        # Knockback recoil + hit-shake + pained roar (`mouth_jaw` open +2px) + 4 aura arms flared wide + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-4.0 * env * facing)) + jit
        body_dy = int(round(-1.0 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-1.5 * env))

        jaw_open = 1 if 1 <= frame_idx <= 7 else 0  # small pained opening (amount 1)
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays on the face; mouth opens via mouth_open

        sash_dx = body_dx
        sash_dy = body_dy + int(round(-1.0 * env))
        sash_ang = -4.5 * env * (-facing)

        kick_leg_dx = body_dx + int(round(-1.0 * env * facing))
        kick_leg_dy = body_dy + int(round(1.2 * env))
        kick_leg_ang = -6.5 * env * (-facing)

        pivot_leg_dx = int(round(body_dx * 0.65))
        pivot_leg_dy = 0

        auL_ang = -14.0 * env * side_sign
        auR_ang = 14.0 * env * side_sign
        alL_ang = 13.0 * env * side_sign
        alR_ang = -13.0 * env * side_sign

        auL_dx, auL_dy = body_dx, body_dy - int(round(1.5 * env))
        auR_dx, auR_dy = body_dx, body_dy - int(round(1.5 * env))
        alL_dx, alL_dy = body_dx, body_dy + int(round(1.2 * env))
        alR_dx, alR_dy = body_dx, body_dy + int(round(1.2 * env))

    auL_rot = rotate_layer_nearest(layers["aura_arms_upper_L"], auL_ang, pivots["aura_arms_upper_L"])
    auR_rot = rotate_layer_nearest(layers["aura_arms_upper_R"], auR_ang, pivots["aura_arms_upper_R"])
    alL_rot = rotate_layer_nearest(layers["aura_arms_lower_L"], alL_ang, pivots["aura_arms_lower_L"])
    alR_rot = rotate_layer_nearest(layers["aura_arms_lower_R"], alR_ang, pivots["aura_arms_lower_R"])
    sash_rot = rotate_layer_nearest(layers["sash_ribbons"], sash_ang, pivots["sash_ribbons"])

    if not is_back:
        legL_rot = rotate_layer_nearest(layers["leg_L"], kick_leg_ang, pivots["leg_L"])
        comp = [
            shift_layer_int(auL_rot, auL_dx, auL_dy),
            shift_layer_int(auR_rot, auR_dx, auR_dy),
            shift_layer_int(alL_rot, alL_dx, alL_dy),
            shift_layer_int(alR_rot, alR_dx, alR_dy),
            shift_layer_int(layers["leg_R"], pivot_leg_dx, pivot_leg_dy),
            shift_layer_int(sash_rot, sash_dx, sash_dy),
            shift_layer_int(legL_rot, kick_leg_dx, kick_leg_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["mouth_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head"], head_dx, head_dy),
        ]
    else:
        legR_rot = rotate_layer_nearest(layers["leg_R"], kick_leg_ang, pivots["leg_R"])
        comp = [
            shift_layer_int(auL_rot, auL_dx, auL_dy),
            shift_layer_int(auR_rot, auR_dx, auR_dy),
            shift_layer_int(alL_rot, alL_dx, alL_dy),
            shift_layer_int(alR_rot, alR_dx, alR_dy),
            shift_layer_int(legR_rot, kick_leg_dx, kick_leg_dy),
            shift_layer_int(layers["leg_L"], pivot_leg_dx, pivot_leg_dy),
            shift_layer_int(sash_rot, sash_dx, sash_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["head"], head_dx, head_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    if not is_back and jaw_open > 0 and "mouth_line" in pivots:
        line = mouth_open.transform_points(pivots["mouth_line"], dx=head_dx, dy=head_dy)
        out = mouth_open.open_mouth(out, line, amount=min(2, jaw_open), deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=0.6)
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

    # Tight bounds of front (79x68) and back (74x71), padded for 4-arm mandala wheel spin & High Jump Kick
    max_w, max_h = 79, 71
    cell_w = 114
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
                frame_rgba = synthesize_medicham_mega_frame(
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

    # Preserve official 92f `front.gif` / `front-reverse.gif`, and replace warped 19f `back.gif` / `back-reverse.gif`
    # with clean 16-frame `idle` N loop!
    idle_sheet = sheets_by_action["idle"]
    back_frames = [idle_sheet[4 * cell_h:5 * cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    back_rev_frames = [f[:, ::-1, :].copy() for f in back_frames]

    save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
    save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)

    meta = {
        "generator": "forge_medicham_mega_actions",
        "key": "0308_medicham-mega",
        "dex": 308,
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
        for gif_name in ("back.gif", "back-reverse.gif"):
            shutil.copy2(MEGA_DIR / gif_name, SYNC_MEGA_DIR / gif_name)

    print(f"Built 0308_medicham-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
