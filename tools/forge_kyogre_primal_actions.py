#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Primal Kyogre (`0382_kyogre-primal`) modeled after Primal Kyogre's official 85-frame
sprite animation (`0382_kyogre/mega/kyogre-primal/front.gif`) and native anatomical design (`front.png` & `back.png`):
  - Exact 7-layer anatomical rigging with true joint pivots (pectoral wing sockets, dorsal root, and tail base):
      1. `tail_streamers` (4 long rear tail ribbons & caudal fin base — pivoted at the tail root socket via
                           `rotate_layer_nearest` so the 4 ribbons undulate and roll like deep-sea currents!)
      2. `fin_wing_far` (far-side giant pectoral fin wing with glowing golden Alpha symbol — pivoted at its shoulder
                         socket via `rotate_layer_nearest` for grand ocean-leviathan wing flaps & whirlpool rolls!)
      3. `dorsal_fins` (twin cyan-trimmed dorsal fins on the back — pivoted to sway and roll with swimming & roars)
      4. `torso` (royal navy & golden-lined leviathan whale body + deep crimson/navy roaring mouth pocket behind `lower_jaw`)
      [2026-11-04 mouth fix, v1m: the jaw no longer drops and no mouth pocket is painted; a small lens-shaped
       opening is carved along the art's mouth line with tools/mouth_open.py (1px growl, 2px only at the roar peak).]
      5. `lower_jaw` (white lower jaw & rear jaw fang — formerly opened `+1px` for leviathan roars in `idle`/`walk` and opens
                      WIDE `+2px` for a thunderous Origin Pulse / Water Spout (`こんげんのはどう` / `しおふき`) roar
                      during `attack` wind-up, whirlpool spin lunge, and `hurt` reaction!)
      6. `head_upper` (golden forehead Alpha crest, glowing yellow eyes, and snout/upper jaw —
                       100% rigid integer motion `shift_layer_int` so the eyes and face never deform by a single pixel!)
      7. `fin_wing_near` (foreground giant pectoral fin main wing with glowing golden Alpha symbol — pivoted at its
                          shoulder socket via `rotate_layer_nearest` for dynamic vortex rolls and Origin Pulse surges!)
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
import mouth_open  # noqa: E402

# Mouth fix (2026-11-04, user request): the jaw no longer drops; a small lens-shaped opening is carved along the
# closed mouth line of the art (native front.png coords, ordered hinge -> lip tip) with mouth_open.open_mouth.
MOUTH_LINE_FRONT_NATIVE = [(39.0, 71.0), (33.0, 71.0), (28.0, 72.0), (23.0, 72.0), (21.0, 71.0)]
MOUTH_DEEP = (16, 16, 16)      # darkest palette colour (outline)
MOUTH_INNER = (240, 92, 150)   # existing crimson/pink palette colour (maw interior)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0382_kyogre/mega/kyogre-primal"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0382_kyogre/mega/kyogre-primal"
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
        | ((r == 16) & (g == 28) & (b == 76))
        | ((r == 39) & (g == 36) & (b == 94))
    )


def is_kyogre_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 3) & (g == 78) & (b == 163))
        | ((r == 8) & (g == 42) & (b == 120))
        | ((r == 7) & (g == 136) & (b == 214))
        | ((r == 118) & (g == 138) & (b == 196))
        | ((r == 55) & (g == 51) & (b == 132))
        | ((r == 116) & (g == 101) & (b == 157))
        | ((r == 65) & (g == 121) & (b == 153))
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
    a = out[..., 3] >= 128
    if np.any(a):
        lbl, n_comp = ndi.label(a, structure=np.ones((3, 3)))
        if n_comp > 1:
            sizes = ndi.sum(a, lbl, range(1, n_comp + 1))
            main_id = int(np.argmax(sizes)) + 1
            stray = a & (lbl != main_id)
            out[stray] = 0
    return out


def segment_front_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `front.png` into 7 exact anatomical masks following 1px outlines:
      1. `tail_streamers`: 4 long rear tail ribbons & central caudal spine (`y=11..43, x=49..90`).
      2. `fin_wing_far`: Far-side left giant pectoral fin wing with golden Alpha orb (`y=28..54, x=0..33`).
      3. `dorsal_fins`: Twin cyan-trimmed dorsal fins on the back (`y=30..57, x=34..60`).
      4. `torso`: Royal navy & golden-lined leviathan body.
      5. `lower_jaw`: White lower jaw, mouth rim, and rear jaw fang (`y=72..77, x=20..49`).
      6. `head_upper`: Snout, golden forehead Alpha crest, glowing yellow eyes, and upper jaw/cheek (`y=52..71, x=19..51`).
      7. `fin_wing_near`: Foreground right giant pectoral fin main wing with golden Alpha orb (`y=49..75, x=51..95`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. `dorsal_fins` (twin cyan-trimmed dorsal fins on the back)
    dorsal_left = a & (
        ((yy >= 30) & (yy <= 39) & (xx >= 35) & (xx <= 47))
        | ((yy == 40) & (xx >= 35) & (xx <= 44))
        | ((yy == 41) & (xx >= 35) & (xx <= 41))
        | ((yy == 42) & (xx >= 34) & (xx <= 39))
        | ((yy == 43) & (xx >= 34) & (xx <= 37))
        | ((yy == 44) & (xx >= 34) & (xx <= 36))
    )
    dorsal_right = a & (
        ((yy == 36) & (xx >= 58) & (xx <= 61))
        | ((yy == 37) & (xx >= 56) & (xx <= 60))
        | ((yy == 38) & (xx >= 54) & (xx <= 59))
        | ((yy == 39) & (xx >= 47) & (xx <= 58))
        | ((yy == 40) & (xx >= 48) & (xx <= 55))
        | ((yy == 41) & (xx >= 48) & (xx <= 56))
        | ((yy >= 42) & (yy <= 50) & (xx >= 48) & (xx <= 60))
        | ((yy == 51) & (xx >= 48) & (xx <= 58))
        | ((yy == 52) & (xx >= 49) & (xx <= 58))
        | ((yy == 53) & (xx >= 53) & (xx <= 59))
        | ((yy >= 54) & (yy <= 56) & (xx >= 56) & (xx <= 60))
        | ((yy == 57) & (xx >= 56) & (xx <= 59))
    )
    dorsal_fins = dorsal_left | dorsal_right

    # 2. `tail_streamers` (4 long rear tail ribbons & central caudal tip behind `dorsal_fins` and `torso` hip)
    tail_streamers = (
        a
        & (
            ((yy >= 11) & (yy <= 35) & (xx >= 49))
            | ((yy >= 36) & (yy <= 38) & (xx >= 60))
            | ((yy == 39) & (xx >= 72))
            | ((yy == 40) & (xx >= 75))
            | ((yy == 41) & (xx >= 79))
        )
        & (~dorsal_fins)
    )

    # 3. `fin_wing_far` (far-left giant pectoral fin wing with golden Alpha orb y=28..54, x=0..33)
    fin_wing_far = a & (
        ((yy >= 28) & (yy <= 44) & (xx <= 33))
        | ((yy >= 45) & (yy <= 46) & (xx <= 32))
        | ((yy == 47) & (xx <= 30))
        | ((yy == 48) & (xx <= 29))
        | ((yy == 49) & (xx <= 28))
        | ((yy == 50) & (xx <= 27))
        | ((yy == 51) & (xx <= 26))
        | ((yy == 52) & (xx <= 25))
        | ((yy == 53) & (xx <= 24))
        | ((yy == 54) & (xx <= 23))
    )

    # 4. `fin_wing_near` (foreground right giant pectoral fin main wing with golden Alpha orb y=49..75, x=51..95)
    fin_wing_near = a & (
        ((yy == 49) & (xx >= 91))
        | ((yy == 50) & (xx >= 89))
        | ((yy == 51) & (xx >= 74))
        | ((yy == 52) & (xx >= 70))
        | ((yy == 53) & (xx >= 66))
        | ((yy == 54) & (xx >= 64))
        | ((yy == 55) & (xx >= 62))
        | ((yy >= 56) & (yy <= 57) & (xx >= 60))
        | ((yy >= 58) & (yy <= 61) & (xx >= 55))
        | ((yy >= 62) & (yy <= 64) & (xx >= 51))
        | ((yy == 65) & (xx >= 53))
        | ((yy == 66) & (xx >= 55))
        | ((yy == 67) & (xx >= 57))
        | ((yy == 68) & (xx >= 59))
        | ((yy == 69) & (xx >= 60))
        | ((yy == 70) & (xx >= 61))
        | ((yy == 71) & (xx >= 62))
        | ((yy >= 72) & (yy <= 75) & (xx >= 63))
    )

    # 5. `lower_jaw` (white lower jaw, mouth rim, and rear jaw fang y=72..77, x=20..49)
    lower_jaw = a & (yy >= 72) & (yy <= 77) & (xx >= 20) & (xx <= 49)

    # 6. `head_upper` (snout, golden forehead Alpha crest, glowing yellow eyes, upper jaw & cheek y=52..71, x=19..52)
    head_upper = (
        a
        & (
            ((yy == 52) & (xx >= 26) & (xx <= 38))
            | ((yy == 53) & (xx >= 25) & (xx <= 40))
            | ((yy == 54) & (xx >= 24) & (xx <= 42))
            | ((yy == 55) & (xx >= 24) & (xx <= 43))
            | ((yy == 56) & (xx >= 23) & (xx <= 44))
            | ((yy == 57) & (xx >= 23) & (xx <= 45))
            | ((yy >= 58) & (yy <= 64) & (xx >= 20) & (xx <= 46))
            | ((yy >= 65) & (yy <= 71) & (xx >= 19) & (xx <= 52))
        )
        & (~fin_wing_far)
        & (~dorsal_fins)
        & (~fin_wing_near)
        & (~lower_jaw)
    )

    used = (
        tail_streamers
        | fin_wing_far
        | dorsal_fins
        | fin_wing_near
        | lower_jaw
        | head_upper
    )
    torso = a & (~used)

    return {
        "tail_streamers": tail_streamers,
        "fin_wing_far": fin_wing_far,
        "dorsal_fins": dorsal_fins,
        "torso": torso,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
        "fin_wing_near": fin_wing_near,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 7 anatomical masks following the true 1px outlines:
      1. `tail_streamers`: 4 long rear tail ribbons & caudal root extending bottom-left (`y=52..83, x=0..44`).
      2. `fin_wing_far`: Far-left giant pectoral fin wing with golden Alpha orb (`y=28..46, x=6..36`).
      3. `dorsal_fins`: Twin cyan-trimmed dorsal fins on the back (`y=34..59, x=24..58`).
      4. `torso`: Dorsal leviathan body with glowing golden dorsal spine.
      5. `lower_jaw`: Empty on rear 3/4 view.
      6. `head_upper`: Upper-right cranium, snout highlight, and golden forehead Alpha crest (`y=25..40, x=45..73`).
      7. `fin_wing_near`: Foreground right giant pectoral fin main wing (`y=41..64, x=57..95`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    empty = np.zeros_like(a)

    # 1. `dorsal_fins` (twin dorsal fins on the back)
    dorsal_left = a & (
        ((yy == 34) & (xx >= 25) & (xx <= 36))
        | ((yy >= 35) & (yy <= 36) & (xx >= 24) & (xx <= 37))
        | ((yy == 37) & (xx >= 25) & (xx <= 36))
        | ((yy == 38) & (xx >= 25) & (xx <= 35))
        | ((yy >= 39) & (yy <= 40) & (xx >= 25) & (xx <= 34))
        | ((yy == 41) & (xx >= 25) & (xx <= 33))
        | ((yy == 42) & (xx >= 26) & (xx <= 33))
        | ((yy == 43) & (xx >= 26) & (xx <= 31))
        | ((yy >= 44) & (yy <= 46) & (xx >= 27) & (xx <= 30))
        | ((yy == 47) & (xx >= 27) & (xx <= 29))
        | ((yy >= 48) & (yy <= 49) & (xx >= 26) & (xx <= 28))
    )
    dorsal_right = a & (
        ((yy == 38) & (xx >= 45) & (xx <= 49))
        | ((yy == 39) & (xx >= 44) & (xx <= 51))
        | ((yy == 40) & (xx >= 44) & (xx <= 53))
        | ((yy == 41) & (xx >= 43) & (xx <= 54))
        | ((yy == 42) & (xx >= 43) & (xx <= 55))
        | ((yy == 43) & (xx >= 43) & (xx <= 56))
        | ((yy == 44) & (xx >= 43) & (xx <= 57))
        | ((yy == 45) & (xx >= 43) & (xx <= 58))
        | ((yy >= 46) & (yy <= 48) & (xx >= 42) & (xx <= 58))
        | ((yy == 49) & (xx >= 41) & (xx <= 58))
        | ((yy >= 50) & (yy <= 51) & (xx >= 39) & (xx <= 57))
        | ((yy == 52) & (xx >= 37) & (xx <= 57))
        | ((yy >= 53) & (yy <= 55) & (xx >= 36) & (xx <= 55))
        | ((yy >= 56) & (yy <= 57) & (xx >= 37) & (xx <= 50))
        | ((yy == 58) & (xx >= 38) & (xx <= 47))
        | ((yy == 59) & (xx >= 41) & (xx <= 44))
    )
    dorsal_fins = dorsal_left | dorsal_right

    # 2. `tail_streamers` (4 long rear tail ribbons & caudal root extending bottom-left y=52..83, x=0..44)
    tail_streamers = (
        a
        & (yy >= 52)
        & (xx <= 44)
        & (~dorsal_fins)
    )

    # 3. `fin_wing_far` (far-left giant pectoral fin wing y=28..46, x=6..36)
    fin_wing_far = (
        a
        & (
            ((yy >= 28) & (yy <= 33) & (xx <= 36))
            | ((yy == 34) & (xx <= 24))
            | ((yy >= 35) & (yy <= 36) & (xx <= 23))
            | ((yy >= 37) & (yy <= 41) & (xx <= 24))
            | ((yy >= 42) & (yy <= 43) & (xx <= 25))
            | ((yy >= 44) & (yy <= 46) & (xx <= 26))
        )
        & (~dorsal_fins)
    )

    # 4. `fin_wing_near` (foreground right giant pectoral fin wing y=41..64, x=57..95)
    fin_wing_near = (
        a
        & (
            ((yy == 41) & (xx >= 64))
            | ((yy == 42) & (xx >= 60))
            | ((yy >= 43) & (yy <= 52) & (xx >= 58))
            | ((yy >= 53) & (yy <= 55) & (xx >= 57))
            | ((yy >= 56) & (yy <= 64) & (xx >= 58))
        )
        & (~dorsal_fins)
    )

    # 5. `head_upper` (upper-right cranium, snout highlight, and golden forehead Alpha crest y=25..40, x=45..73)
    head_upper = (
        a
        & (
            ((yy >= 25) & (yy <= 37) & (xx >= 45))
            | ((yy >= 38) & (yy <= 40) & (xx >= 50))
        )
        & (~dorsal_fins)
        & (~fin_wing_near)
    )

    used = (
        tail_streamers
        | fin_wing_far
        | dorsal_fins
        | fin_wing_near
        | head_upper
    )
    torso = a & (~used)

    return {
        "tail_streamers": tail_streamers,
        "fin_wing_far": fin_wing_far,
        "dorsal_fins": dorsal_fins,
        "torso": torso,
        "lower_jaw": empty,
        "head_upper": head_upper,
        "fin_wing_near": fin_wing_near,
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
    dst_y0 = (foot_y - sh)

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

    body_fill_m = is_kyogre_body_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order (strictly Back-to-Front):
        # 1. tail_streamers -> 2. fin_wing_far -> 3. dorsal_fins ->
        # 4. torso -> 5. lower_jaw -> 6. head_upper -> 7. fin_wing_near

        fg_over_tail = c_masks["dorsal_fins"] | c_masks["torso"]
        u_tail = ndi.binary_erosion(c_masks["tail_streamers"] | fg_over_tail, iterations=1)
        layers["tail_streamers"] = build_part_layer(
            canvas,
            c_masks["tail_streamers"],
            u_tail & fg_over_tail,
            cushion_px=7,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_ffar = (
            c_masks["dorsal_fins"]
            | c_masks["torso"]
            | c_masks["lower_jaw"]
            | c_masks["head_upper"]
        )
        u_ffar = ndi.binary_erosion(c_masks["fin_wing_far"] | fg_over_ffar, iterations=1)
        layers["fin_wing_far"] = build_part_layer(
            canvas,
            c_masks["fin_wing_far"],
            u_ffar & fg_over_ffar,
            cushion_px=7,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_dfin = c_masks["torso"] | c_masks["head_upper"] | c_masks["fin_wing_near"]
        u_dfin = ndi.binary_erosion(c_masks["dorsal_fins"] | fg_over_dfin, iterations=1)
        layers["dorsal_fins"] = build_part_layer(
            canvas,
            c_masks["dorsal_fins"],
            u_dfin & fg_over_dfin,
            cushion_px=6,
            allowed_fill_mask=body_fill_m,
        )

        # IMPORTANT: `torso` cushion ONLY expands under foreground layers drawn ON TOP OF `torso`
        # (`lower_jaw`, `head_upper`, and the inner shoulder socket of `fin_wing_near`),
        # NEVER over background fins/tail streamers or out into the open wing blade!
        wing_socket_fg = c_masks["fin_wing_near"] & ndi.binary_dilation(c_masks["torso"], iterations=3)
        fg_over_torso = c_masks["lower_jaw"] | c_masks["head_upper"] | wing_socket_fg
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        # (Mouth fix 2026-11-04: the rectangular crimson mouth pocket behind `lower_jaw` is no longer painted.)
        layers["torso"] = torso_lyr

        jaw_lyr = build_part_layer(
            canvas,
            c_masks["lower_jaw"],
            np.zeros_like(c_masks["lower_jaw"]),
            cushion_px=0,
        )
        layers["lower_jaw"] = jaw_lyr

        layers["head_upper"] = build_part_layer(
            canvas,
            c_masks["head_upper"],
            np.zeros_like(c_masks["head_upper"]),
            cushion_px=0,
        )

        layers["fin_wing_near"] = build_part_layer(
            canvas,
            c_masks["fin_wing_near"],
            np.zeros_like(c_masks["fin_wing_near"]),
            cushion_px=0,
        )

        pivots = {
            "tail_streamers": to_canvas_pt(64.0, 40.0),  # Caudal base socket for the 4 tail ribbons
            "fin_wing_far": to_canvas_pt(28.0, 48.0),    # Far pectoral fin shoulder socket
            "dorsal_fins": to_canvas_pt(50.0, 49.0),     # Dorsal fin root along spine
            "fin_wing_near": to_canvas_pt(58.0, 61.0),   # Near giant pectoral wing shoulder socket
        }
        pivots["_mouth_line"] = [to_canvas_pt(mx, my) for mx, my in MOUTH_LINE_FRONT_NATIVE]

    else:
        # Back view draw order (strictly Back-to-Front):
        # 1. fin_wing_far -> 2. head_upper -> 3. lower_jaw ->
        # 4. torso -> 5. tail_streamers -> 6. dorsal_fins -> 7. fin_wing_near

        fg_over_ffar = c_masks["head_upper"] | c_masks["torso"] | c_masks["dorsal_fins"]
        u_ffar = ndi.binary_erosion(c_masks["fin_wing_far"] | fg_over_ffar, iterations=1)
        layers["fin_wing_far"] = build_part_layer(
            canvas,
            c_masks["fin_wing_far"],
            u_ffar & fg_over_ffar,
            cushion_px=7,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_head = c_masks["torso"] | c_masks["dorsal_fins"] | c_masks["fin_wing_near"]
        u_head = ndi.binary_erosion(c_masks["head_upper"] | fg_over_head, iterations=1)
        layers["head_upper"] = build_part_layer(
            canvas,
            c_masks["head_upper"],
            u_head & fg_over_head,
            cushion_px=6,
            allowed_fill_mask=body_fill_m,
        )

        layers["lower_jaw"] = np.zeros_like(canvas)

        # IMPORTANT: `torso` cushion ONLY expands under foreground layers drawn ON TOP OF `torso`
        # (`tail_streamers`, `dorsal_fins`, and inner socket of `fin_wing_near`), NEVER over `fin_wing_far` or `head_upper`!
        wing_socket_back = c_masks["fin_wing_near"] & ndi.binary_dilation(c_masks["torso"], iterations=3)
        fg_over_torso = c_masks["tail_streamers"] | c_masks["dorsal_fins"] | wing_socket_back
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        layers["torso"] = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_tail = c_masks["dorsal_fins"]
        u_tail = ndi.binary_erosion(c_masks["tail_streamers"] | fg_over_tail, iterations=1)
        layers["tail_streamers"] = build_part_layer(
            canvas,
            c_masks["tail_streamers"],
            u_tail & fg_over_tail,
            cushion_px=7,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_dfin = c_masks["fin_wing_near"]
        u_dfin = ndi.binary_erosion(c_masks["dorsal_fins"] | fg_over_dfin, iterations=1)
        layers["dorsal_fins"] = build_part_layer(
            canvas,
            c_masks["dorsal_fins"],
            u_dfin & fg_over_dfin,
            cushion_px=6,
            allowed_fill_mask=body_fill_m,
        )

        layers["fin_wing_near"] = build_part_layer(
            canvas,
            c_masks["fin_wing_near"],
            np.zeros_like(c_masks["fin_wing_near"]),
            cushion_px=0,
        )

        pivots = {
            "tail_streamers": to_canvas_pt(29.0, 56.0),  # Rear-left caudal base socket
            "fin_wing_far": to_canvas_pt(25.0, 39.0),    # Far pectoral fin shoulder socket
            "dorsal_fins": to_canvas_pt(45.0, 48.0),     # Dorsal fin root along spine
            "fin_wing_near": to_canvas_pt(60.0, 51.0),   # Near giant pectoral wing shoulder socket
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
        (mid_x - 20 - drift, max(2, top_y + 14 - drift // 2)),
        (mid_x + 20 + drift, max(2, top_y + 16 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (255, 244, 147, 255)
    return out


def synthesize_kyogre_primal_frame(
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
    mouth_amt = 0
    dfin_dx, dfin_dy = 0, 0
    ffar_dx, ffar_dy = 0, 0
    fnear_dx, fnear_dy = 0, 0
    tail_dx, tail_dy = 0, 0

    dfin_ang = 0.0
    ffar_ang = 0.0
    fnear_ang = 0.0
    tail_ang = 0.0

    if action == "idle":
        # Deep-Sea / Aerial Leviathan Hover + Grand Pectoral Wing Flap & Roll + Sea-God Roar:
        # 1. Giant whale torso glides smoothly in a majestic oceanic wave (`±1px` x, `±2px` y).
        # 2. Both colossal golden-Alpha pectoral wings (`fin_wing_near` & `fin_wing_far`) and the 4 rear
        #    tail ribbons (`tail_streamers`) pivot and roll (`±6.5°` / `±6.0°`) with oceanic phase lag.
        # 3. White `lower_jaw` opens `+1px` (and `+2px` at peak frame 4) during the sea-god roar (`frames 2..6`, `11..13`),
        #    revealing the glowing crimson/magenta maw interior while `head_upper` stays 100% rigid!
        body_dy = int(round(1.8 * math.sin(phase - 0.2)))
        body_dx = int(round(1.0 * math.cos(phase - 0.2) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        if frame_idx in (2, 3, 4, 5, 6, 11, 12):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays put; mouth opens via mouth_open
        mouth_amt = jaw_open

        dfin_dx, dfin_dy = body_dx, body_dy
        dfin_ang = 3.5 * math.sin(phase - 0.6) * facing

        fnear_dx = body_dx
        fnear_dy = body_dy + int(round(1.0 * math.sin(phase - 0.5)))
        fnear_ang = 6.5 * math.sin(phase - 0.5) * (-facing)

        ffar_dx = body_dx
        ffar_dy = body_dy + int(round(1.0 * math.sin(phase - 0.8)))
        ffar_ang = 6.5 * math.sin(phase - 0.8) * facing

        tail_dx = body_dx
        tail_dy = body_dy + int(round(-1.0 * math.sin(phase - 1.1)))
        tail_ang = 6.0 * math.sin(phase - 1.1) * facing

    elif action == "walk":
        # Powerful Oceanic Whirlpool Roll Flap & Surge Glide + Roaring Jaw:
        # 2-cycle pectoral wing propulsion & vortex roll (`±8.5°`), undulating 4 tail ribbons (`±7.5°`),
        # forward surging glide (`±2px` x, `±2px` y), and rhythmic `+1px` sea-god roar on each propulsion stroke!
        surge_x = int(round(2.0 * sin1 * facing))
        surge_y = int(round(1.6 * math.sin(2.0 * phase - 0.3)))

        torso_dx, torso_dy = surge_x, surge_y
        head_dx = surge_x + int(round(0.8 * cos1 * facing))
        head_dy = surge_y

        s2 = math.sin(2.0 * phase)
        jaw_open = 1 if s2 > 0.25 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays put; mouth opens via mouth_open
        mouth_amt = jaw_open

        dfin_dx, dfin_dy = surge_x, surge_y
        dfin_ang = 4.5 * math.sin(2.0 * phase - 0.5) * facing

        fnear_dx = surge_x + int(round(0.8 * cos1 * facing))
        fnear_dy = surge_y + int(round(1.2 * math.sin(2.0 * phase - 0.4)))
        fnear_ang = 8.5 * math.sin(2.0 * phase - 0.4) * (-facing)

        ffar_dx = surge_x + int(round(-0.8 * cos1 * facing))
        ffar_dy = surge_y + int(round(-1.2 * math.sin(2.0 * phase - 0.4)))
        ffar_ang = -8.5 * math.sin(2.0 * phase - 0.4) * facing

        tail_dx = surge_x
        tail_dy = surge_y + int(round(-1.0 * math.sin(2.0 * phase - 0.9)))
        tail_ang = 7.5 * math.sin(2.0 * phase - 0.9) * facing

    elif action == "attack":
        # Signature Primal Kyogre "Leviathan Sea-God Roar + Great Whirlpool Spin & Origin Pulse / Water Spout (`こんげんのはどう` / `しおふき`)"!
        # Frames 0..3 (Wind-up & Thunderous Sea-God Roar):
        #   - Pulls massive leviathan body back (`-4px`), flares both giant golden-Alpha pectoral wings (`-10°`),
        #   - Opens `lower_jaw` WIDE (`+2px` grand roaring maw!).
        # Frames 4..10 (Great Whirlpool Spin Lunge & Origin Pulse / Water Spout Unleashed!):
        #   - Explosive forward tidal surge (`+7px` snout & body),
        #   - High-speed whirlpool barrel-roll rotation across both giant pectoral wings (`fnear ±12.5°`, `ffar ∓12.5°`),
        #     the 4 tail ribbons (`tail ±9.5°`), and twin dorsal fins (`dfin ±6.5°`),
        #   - `lower_jaw` stays wide open (`+2px`) unleashing the Sea God's Origin Pulse roar!
        # Frames 11..15: Oceanic follow-through recovery to hover stance.
        lunge_x_table = [0, -2, -3, -4, 3, 6, 7, 7, 6, 4, 2, 0, -1, -1, 0, 0]
        lunge_y_table = [0, -1, -1, -2, 0, 1, 2, 1, 0, -1, 0, 1, 0, 0, 0, 0]

        body_dx = int(round(lunge_x_table[frame_idx] * facing))
        body_dy = lunge_y_table[frame_idx]

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + (int(round(1.0 * facing)) if frame_idx in (5, 6, 7, 8) else 0)
        head_dy = body_dy

        # Jaw curve: wide +2px roar during wind-up (1..3) AND throughout the Origin Pulse tidal surge (4..10)!
        jaw_open_table = [0, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  # 2 only at the roar peak
        jaw_open = jaw_open_table[frame_idx]
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays put; mouth opens via mouth_open
        mouth_amt = jaw_open

        # Great Whirlpool Roll angles across giant pectoral wings, dorsal fins, and 4 tail ribbons during frames (4..10)!
        roll_dfin = [0.0, -3.5, -5.0, -6.5, 6.0, -6.5, 6.5, -6.0, 5.0, -3.5, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        roll_fnear = [0.0, -5.5, -8.5, -11.0, 11.5, -12.5, 12.5, -11.5, 9.5, -6.5, 4.0, -1.5, 0.0, 0.0, 0.0, 0.0]
        roll_ffar = [0.0, 5.5, 8.5, 11.0, -11.5, 12.5, -12.5, 11.5, -9.5, 6.5, -4.0, 1.5, 0.0, 0.0, 0.0, 0.0]
        roll_tail = [0.0, -4.5, -7.0, -9.0, 9.0, -9.5, 9.5, -8.5, 7.0, -5.0, 3.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        vortex_dy = [0, 0, -1, -1, 1, -2, 2, -2, 1, -1, 0, 0, 0, 0, 0, 0]

        dfin_dx = body_dx
        dfin_dy = body_dy + (vortex_dy[frame_idx] // 2)
        dfin_ang = roll_dfin[frame_idx] * facing

        fnear_dx = body_dx + (int(round(1.0 * facing)) if frame_idx in (5, 6, 7) else 0)
        fnear_dy = body_dy + vortex_dy[frame_idx]
        fnear_ang = roll_fnear[frame_idx] * (-facing)

        ffar_dx = body_dx
        ffar_dy = body_dy - vortex_dy[frame_idx]
        ffar_ang = roll_ffar[frame_idx] * facing

        tail_dx = body_dx
        tail_dy = body_dy - vortex_dy[frame_idx]
        tail_ang = roll_tail[frame_idx] * facing

    elif action == "hurt":
        # Deep-sea shock knockback + hit-shake + pained Leviathan roar (`lower_jaw` open `+2px`) + wings flung back + golden sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-4.0 * env * facing)) + jit
        body_dy = int(round(-1.2 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-1.5 * env))

        if frame_idx in (1, 2, 3, 4, 5, 6, 7, 8):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # jaw stays put; mouth opens via mouth_open
        mouth_amt = jaw_open

        dfin_dx, dfin_dy = body_dx, body_dy
        dfin_ang = -5.0 * env * facing

        fnear_dx = body_dx + int(round(-1.0 * env * facing))
        fnear_dy = body_dy + int(round(-1.2 * env))
        fnear_ang = -8.5 * env * (-facing)

        ffar_dx = body_dx + int(round(-1.0 * env * facing))
        ffar_dy = body_dy + int(round(-1.2 * env))
        ffar_ang = 8.5 * env * facing

        tail_dx, tail_dy = body_dx, body_dy
        tail_ang = -7.0 * env * facing

    tail_rot = rotate_layer_nearest(layers["tail_streamers"], tail_ang, pivots["tail_streamers"])
    ffar_rot = rotate_layer_nearest(layers["fin_wing_far"], ffar_ang, pivots["fin_wing_far"])
    dfin_rot = rotate_layer_nearest(layers["dorsal_fins"], dfin_ang, pivots["dorsal_fins"])
    fnear_rot = rotate_layer_nearest(layers["fin_wing_near"], fnear_ang, pivots["fin_wing_near"])

    if not is_back:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(ffar_rot, ffar_dx, ffar_dy),
            shift_layer_int(dfin_rot, dfin_dx, dfin_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(fnear_rot, fnear_dx, fnear_dy),
        ]
    else:
        comp = [
            shift_layer_int(ffar_rot, ffar_dx, ffar_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(dfin_rot, dfin_dx, dfin_dy),
            shift_layer_int(fnear_rot, fnear_dx, fnear_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    mline = pivots.get("_mouth_line")
    if (not is_back) and mline and mouth_amt > 0:
        line = mouth_open.transform_points(mline, dx=head_dx, dy=head_dy)
        out = mouth_open.open_mouth(out, line, amount=mouth_amt, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=0.6)
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

    # Tight bounds of front (96x67) and back (96x59)
    max_w, max_h = 96, 67
    cell_w = 120
    cell_h = 104
    foot_y = 92

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
                frame_rgba = synthesize_kyogre_primal_frame(
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

    # Preserve existing high-quality 85-frame official `front.gif` & `front-reverse.gif`,
    # and cleanly replace broken 19-frame `back.gif` & `back-reverse.gif` with 16-frame `idle` N loop!
    idle_sheet = sheets_by_action["idle"]
    back_frames = [idle_sheet[4 * cell_h:5 * cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    back_rev_frames = [f[:, ::-1, :].copy() for f in back_frames]

    save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
    save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)

    meta = {
        "generator": "forge_kyogre_primal_actions",
        "version": "kyogre_primal_v1m",
        "key": "0382_kyogre-primal",
        "dex": 382,
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

    print(f"Built 0382_kyogre-primal -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
