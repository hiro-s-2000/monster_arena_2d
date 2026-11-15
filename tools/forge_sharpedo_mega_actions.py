#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Sharpedo (`0319_sharpedo-mega`) modeled after Normal Sharpedo's official GIF
(`0319_sharpedo/normal/front.gif`) and Mega Sharpedo's saw-snout torpedo shark anatomy (`front.png` & `back.png`):
  - Exact 8-layer anatomical rigging with true joint pivots (fin sockets & jet nozzle):
      1. `tail_rear` (rear jet-propulsion caudal nozzle & dorsal bump — vibrates and pivots with jet thrust)
      2. `pectoral_fin_far` (far-side blade pectoral fin — pivoted at its socket for torpedo roll & propulsion sweep)
      3. `ventral_fins` (long lower ventral/pelvic fin with yellow stripes — pivoted at its base to sweep with water flow)
      4. `dorsal_fin` (towering yellow-scarred dorsal fin — pivoted at the dorsal root to slice and roll through currents)
      5. `torso` (streamlined navy & white-gray shark body + crimson/pink mouth interior pocket behind the jaws)
      6. `lower_jaw` (massive saw-toothed lower jaw, pink tongue, and yellow chin spikes — opens `+1..+2px` for ferocious
                      shark roar/bark in `idle`/`walk`, opens wide `+2..+3px` during `attack` wind-up roar, and snaps shut
                      `-2px` for a devastating Crunch (`かみくだく`) bite on torpedo impact!)
      7. `head_upper` (`saw_snout_spikes` + yellow star scar + glowing red eye + black gills + upper jaw teeth —
                       100% rigid integer motion `shift_layer_int` so the saw snout, red eye, and teeth never deform!)
      8. `pectoral_fin_near` (foreground yellow-scarred blade pectoral fin — pivoted at its white socket ring for
                              high-speed corkscrew torpedo rolls and slicing propulsion!)
"""

import argparse
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
import mouth_open  # noqa: E402  (2026-11-04 small face-following mouth opening)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0319_sharpedo/mega/sharpedo-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

SYNC_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
SYNC_MEGA_DIR = SYNC_ROOT / "monsterSprite/gen3/0319_sharpedo/mega/sharpedo-mega"
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16

# 2026-11-04 mouth fix (user: rectangular mouth / jutting chin): the lower jaw no longer drops and no rectangular
# crimson pocket/stem is painted. Instead a small lens-shaped gap is carved with tools/mouth_open.py along the line
# between the pink tongue and the lower teeth (native front.png coords, hinge/back -> lip/front). The gap grows UP
# into the tongue (normal (0,-1)) so teeth, chin and outline stay intact. Back views have no visible mouth.
MOUTH_LINE_FRONT_NATIVE = [(42.0, 62.0), (38.0, 62.0), (34.0, 62.0)]
MOUTH_DEEP = (16, 16, 16)      # darkest palette colour (outline)
MOUTH_INNER = (172, 57, 82)    # existing crimson mouth-interior colour


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
        | ((r == 41) & (g == 57) & (b == 82))
    )


def is_shark_body_fill(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 57) & (g == 82) & (b == 148))
        | ((r == 82) & (g == 115) & (b == 189))
        | ((r == 49) & (g == 65) & (b == 115))
        | ((r == 172) & (g == 164) & (b == 180))
        | ((r == 230) & (g == 222) & (b == 246))
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
    # Keep only the primary connected component so nearest-neighbor fin rotation never leaves a 1px stray speck
    a = out[..., 3] >= 128
    if np.any(a):
        lbl, n_comp = ndi.label(a)
        if n_comp > 1:
            sizes = ndi.sum(a, lbl, range(1, n_comp + 1))
            main_id = int(np.argmax(sizes)) + 1
            stray = a & (lbl != main_id)
            out[stray] = 0
    return out


def segment_front_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `front.png` into 8 exact anatomical masks following 1px outlines:
      1. `tail_rear`: Rear jet-propulsion caudal fin & dorsal bump (`x=59..67, y=41..44 & 59..63`).
      2. `pectoral_fin_far`: Far-side left blade pectoral fin with yellow scar (`x=21..35, y=50..58`).
      3. `ventral_fins`: Long lower ventral/pelvic fin with two yellow stripes (`x=46..58, y=64..83`).
      4. `dorsal_fin`: Towering yellow-scarred dorsal fin (`x=46..64, y=11..36`).
      5. `torso`: Streamlined navy/gray shark body + upper mouth roof pocket.
      6. `lower_jaw`: Massive lower jaw, sharp white teeth, pink tongue, and yellow chin spikes (`x=30..48, y=56..70`).
      7. `head_upper`: Long white saw-toothed snout (`saw_snout_spikes`), yellow star scar, glowing red eye,
                       black gill slits, and upper jaw teeth (`x=15..61, y=26..57`).
      8. `pectoral_fin_near`: Foreground right blade pectoral fin with yellow cross scar (`x=56..80, y=51..62`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Dorsal fin (towering dorsal fin with yellow scars)
    dorsal_fin = a & (
        ((yy >= 11) & (yy <= 34) & (xx >= 46))
        | ((yy == 35) & (xx >= 50))
        | ((yy == 36) & (xx >= 54))
    )

    # 2. Pectoral fin far (left blade fin)
    pectoral_fin_far = a & (
        ((yy == 50) & (xx >= 29) & (xx <= 30))
        | ((yy == 51) & (xx >= 28) & (xx <= 35))
        | ((yy == 52) & (xx >= 26) & (xx <= 34))
        | ((yy == 53) & (xx >= 25) & (xx <= 33))
        | ((yy == 54) & (xx >= 23) & (xx <= 33))
        | ((yy == 55) & (xx >= 22) & (xx <= 32))
        | ((yy == 56) & (xx >= 21) & (xx <= 31))
        | ((yy == 57) & (xx >= 21) & (xx <= 30))
        | ((yy == 58) & (xx >= 22) & (xx <= 26))
    )

    # 3. Pectoral fin near (foreground right blade fin)
    pectoral_fin_near = a & (
        ((yy == 51) & (xx >= 56) & (xx <= 67))
        | ((yy == 52) & (xx >= 56))
        | ((yy == 53) & (xx >= 57))
        | ((yy == 54) & (xx >= 58))
        | ((yy == 55) & (xx >= 59))
        | ((yy == 56) & (xx >= 61))
        | ((yy == 57) & (xx >= 63))
        | ((yy == 58) & (xx >= 65))
        | ((yy >= 59) & (yy <= 62) & (xx >= 67))
    )

    # 4. Ventral fins (long lower fin below belly)
    ventral_fins = a & (
        ((yy == 64) & (xx >= 50) & (xx <= 57))
        | ((yy >= 65) & (yy <= 83) & (xx >= 46) & (xx <= 58))
    )

    # 5. Tail rear (rear jet propulsion bump & caudal tip)
    tail_rear = a & (
        ((yy >= 41) & (yy <= 44) & (xx >= 62))
        | ((yy == 59) & (xx >= 62) & (xx <= 66))
        | ((yy == 60) & (xx >= 59) & (xx <= 65))
        | ((yy == 61) & (xx >= 59) & (xx <= 64))
        | ((yy == 62) & (xx >= 60) & (xx <= 64))
        | ((yy == 63) & (xx >= 61) & (xx <= 62))
    )

    # 6. Lower jaw (pink tongue, lower white fangs, gray jaw frame, yellow chin spikes)
    lower_jaw = a & (
        ((yy == 56) & (xx >= 34) & (xx <= 44))
        | ((yy == 57) & (xx >= 31) & (xx <= 47))
        | ((yy == 58) & (xx >= 30) & (xx <= 47))
        | ((yy == 59) & (xx >= 30) & (xx <= 47))
        | ((yy == 60) & (xx >= 30) & (xx <= 47))
        | ((yy == 61) & (xx >= 31) & (xx <= 46))
        | ((yy == 62) & (xx >= 31) & (xx <= 46))
        | ((yy == 63) & (xx >= 31) & (xx <= 46))
        | ((yy == 64) & (xx >= 32) & (xx <= 45))
        | ((yy == 65) & (xx >= 33) & (xx <= 44))
        | ((yy == 66) & (xx >= 33) & (xx <= 44))
        | ((yy == 67) & (xx >= 33) & (xx <= 43))
        | ((yy == 68) & (xx >= 33) & (xx <= 42))
        | ((yy == 69) & (xx >= 34) & (xx <= 40))
        | ((yy == 70) & (xx >= 35) & (xx <= 38))
    )

    # 7. Head upper (white saw-toothed snout, yellow star scar, glowing red eye, black gills, and ALL upper jaw teeth)
    head_upper = (
        a
        & (
            ((yy >= 26) & (yy <= 47) & (xx <= 61))
            | ((yy == 48) & (xx <= 53))
            | ((yy == 49) & (xx <= 52))
            | ((yy == 50) & (((xx >= 31) & (xx <= 38)) | ((xx >= 40) & (xx <= 52))))
            | ((yy == 51) & (((xx >= 36) & (xx <= 37)) | ((xx >= 41) & (xx <= 52))))
            | ((yy == 52) & (((xx >= 35) & (xx <= 36)) | ((xx >= 44) & (xx <= 53))))
            | ((yy == 53) & (((xx >= 34) & (xx <= 36)) | ((xx >= 45) & (xx <= 53))))
            | ((yy == 54) & (((xx >= 34) & (xx <= 36)) | ((xx >= 46) & (xx <= 54))))
            | ((yy == 55) & (((xx >= 34) & (xx <= 36)) | ((xx >= 46) & (xx <= 55))))
            | ((yy == 56) & (xx >= 45) & (xx <= 57))
            | ((yy == 57) & (xx >= 55) & (xx <= 60))
        )
        & (~dorsal_fin)
        & (~tail_rear)
        & (~pectoral_fin_far)
    )

    used = (
        dorsal_fin
        | pectoral_fin_far
        | pectoral_fin_near
        | ventral_fins
        | tail_rear
        | lower_jaw
        | head_upper
    )
    torso = a & (~used)

    return {
        "tail_rear": tail_rear,
        "pectoral_fin_far": pectoral_fin_far,
        "ventral_fins": ventral_fins,
        "dorsal_fin": dorsal_fin,
        "torso": torso,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
        "pectoral_fin_near": pectoral_fin_near,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` into matching 8 anatomical masks following the true 1px outlines.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Dorsal fin on back (`y=10..33, x<=46`)
    dorsal_fin = a & (
        ((yy >= 10) & (yy <= 29) & (xx <= 46))
        | ((yy >= 30) & (yy <= 32) & (xx <= 45))
        | ((yy == 33) & (xx <= 42))
    )

    # 2. Pectoral fin far on back (left blade fin `y=44..53, x<=26`)
    pectoral_fin_far = a & (yy >= 44) & (yy <= 53) & (xx <= 26)

    # 3. Pectoral fin near on back (right blade fin `y=52..63, x>=56`)
    pectoral_fin_near = a & (
        ((yy == 52) & (xx >= 58))
        | ((yy == 53) & (xx >= 58))
        | ((yy == 54) & (xx >= 57))
        | ((yy >= 55) & (yy <= 59) & (xx >= 56))
        | ((yy == 60) & (xx >= 59))
        | ((yy == 61) & (xx >= 64))
        | ((yy == 62) & (xx >= 69))
        | ((yy == 63) & (xx >= 73))
    )

    # 4. Lower jaw tip peeking under the cheek on 3/4 back view (`y=62..65, x=61..70`)
    lower_jaw = (
        a
        & (
            ((yy == 62) & (xx >= 61) & (xx <= 68))
            | ((yy == 63) & (xx >= 62) & (xx <= 70))
            | ((yy == 64) & (xx >= 64) & (xx <= 70))
            | ((yy == 65) & (xx >= 66) & (xx <= 69))
        )
        & (~pectoral_fin_near)
    )

    # 5. Ventral fins on back (`y=65..85, x=43..56`)
    ventral_fins = a & (
        ((yy >= 65) & (yy <= 69) & (xx >= 49) & (xx <= 56))
        | ((yy >= 70) & (yy <= 85) & (xx >= 43) & (xx <= 56))
    )

    # 6. Tail rear on back (rear jet propulsion nozzle & yellow star scar at `y=56..68, x=26..39`)
    tail_rear = a & (
        ((yy == 56) & (xx >= 26) & (xx <= 33))
        | ((yy == 57) & (xx >= 26) & (xx <= 34))
        | ((yy == 58) & (xx >= 27) & (xx <= 34))
        | ((yy == 59) & (xx >= 29) & (xx <= 34))
        | ((yy == 60) & (xx >= 28) & (xx <= 39))
        | ((yy == 61) & (xx >= 28) & (xx <= 38))
        | ((yy >= 62) & (yy <= 65) & (xx >= 28) & (xx <= 37))
        | ((yy == 66) & (xx >= 29) & (xx <= 37))
        | ((yy == 67) & (xx >= 31) & (xx <= 37))
        | ((yy == 68) & (xx >= 34) & (xx <= 37))
    )

    # 7. Head upper on back (follows the natural diagonal 1px black gill/head seam `#` at `x=47..54`)
    head_upper = (
        a
        & (
            ((yy >= 23) & (yy <= 38) & (xx >= 47))
            | ((yy == 39) & (xx >= 54))
            | ((yy == 40) & (xx >= 53))
            | ((yy == 41) & (xx >= 52))
            | ((yy >= 42) & (yy <= 43) & (xx >= 51))
            | ((yy >= 44) & (yy <= 46) & (xx >= 50))
            | ((yy >= 47) & (yy <= 48) & (xx >= 49))
            | ((yy >= 49) & (yy <= 50) & (xx >= 48))
            | ((yy == 51) & (xx >= 47))
        )
        & (~dorsal_fin)
        & (~pectoral_fin_near)
    )

    used = (
        dorsal_fin
        | pectoral_fin_far
        | pectoral_fin_near
        | lower_jaw
        | ventral_fins
        | tail_rear
        | head_upper
    )
    torso = a & (~used)

    return {
        "tail_rear": tail_rear,
        "pectoral_fin_far": pectoral_fin_far,
        "ventral_fins": ventral_fins,
        "dorsal_fin": dorsal_fin,
        "torso": torso,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
        "pectoral_fin_near": pectoral_fin_near,
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
    places onto the target `(cell_h, cell_w)` canvas with bottom fin aligned to `foot_y`,
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
    dst_y0 = (foot_y + 2 - sh) if is_back else (foot_y - sh)

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

    body_fill_m = is_shark_body_fill(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order (back-to-front):
        # 1. tail_rear -> 2. pectoral_fin_far -> 3. ventral_fins -> 4. dorsal_fin
        # -> 5. torso -> 6. lower_jaw -> 7. head_upper -> 8. pectoral_fin_near
        fg_over_tail = c_masks["torso"] | c_masks["head_upper"] | c_masks["pectoral_fin_near"]
        u_tail = ndi.binary_erosion(c_masks["tail_rear"] | fg_over_tail, iterations=1)
        layers["tail_rear"] = build_part_layer(
            canvas,
            c_masks["tail_rear"],
            u_tail & fg_over_tail,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_pfar = c_masks["torso"] | c_masks["lower_jaw"] | c_masks["head_upper"]
        u_pfar = ndi.binary_erosion(c_masks["pectoral_fin_far"] | fg_over_pfar, iterations=1)
        layers["pectoral_fin_far"] = build_part_layer(
            canvas,
            c_masks["pectoral_fin_far"],
            u_pfar & fg_over_pfar,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_vfin = c_masks["torso"] | c_masks["lower_jaw"]
        u_vfin = ndi.binary_erosion(c_masks["ventral_fins"] | fg_over_vfin, iterations=1)
        layers["ventral_fins"] = build_part_layer(
            canvas,
            c_masks["ventral_fins"],
            u_vfin & fg_over_vfin,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_dfin = c_masks["torso"] | c_masks["head_upper"]
        u_dfin = ndi.binary_erosion(c_masks["dorsal_fin"] | fg_over_dfin, iterations=1)
        layers["dorsal_fin"] = build_part_layer(
            canvas,
            c_masks["dorsal_fin"],
            u_dfin & fg_over_dfin,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        # IMPORTANT: `torso` cushion ONLY expands under foreground layers drawn ON TOP OF `torso`
        # (`lower_jaw`, `head_upper`, `pectoral_fin_near`), NEVER over background fins!
        fg_over_torso = c_masks["lower_jaw"] | c_masks["head_upper"] | c_masks["pectoral_fin_near"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=body_fill_m,
        )

        # (2026-11-04) no rectangular crimson pocket behind the jaw any more: the jaw does not drop.
        layers["torso"] = torso_lyr

        jaw_lyr = build_part_layer(
            canvas, c_masks["lower_jaw"], np.zeros_like(c_masks["lower_jaw"]), cushion_px=0
        )
        # (2026-11-04) no crimson mouth stem above the jaw (it painted a rectangle over the art).
        layers["lower_jaw"] = jaw_lyr
        layers["head_upper"] = build_part_layer(
            canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0
        )
        layers["pectoral_fin_near"] = build_part_layer(
            canvas, c_masks["pectoral_fin_near"], np.zeros_like(c_masks["pectoral_fin_near"]), cushion_px=0
        )

        pivots = {
            "dorsal_fin": to_canvas_pt(52.0, 35.0),
            "pectoral_fin_far": to_canvas_pt(33.0, 52.0),
            "pectoral_fin_near": to_canvas_pt(58.0, 53.0),
            "ventral_fins": to_canvas_pt(52.0, 65.0),
            "tail_rear": to_canvas_pt(61.0, 60.0),
        }
        pivots["mouth_line"] = [to_canvas_pt(x, y) for x, y in MOUTH_LINE_FRONT_NATIVE]

    else:
        # Back view draw order (back-to-front):
        # 1. tail_rear -> 2. pectoral_fin_far -> 3. ventral_fins -> 4. dorsal_fin
        # -> 5. lower_jaw -> 6. torso -> 7. head_upper -> 8. pectoral_fin_near
        fg_over_tail = c_masks["torso"]
        u_tail = ndi.binary_erosion(c_masks["tail_rear"] | fg_over_tail, iterations=1)
        layers["tail_rear"] = build_part_layer(
            canvas,
            c_masks["tail_rear"],
            u_tail & fg_over_tail,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_pfar = c_masks["torso"]
        u_pfar = ndi.binary_erosion(c_masks["pectoral_fin_far"] | fg_over_pfar, iterations=1)
        layers["pectoral_fin_far"] = build_part_layer(
            canvas,
            c_masks["pectoral_fin_far"],
            u_pfar & fg_over_pfar,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_vfin = c_masks["torso"]
        u_vfin = ndi.binary_erosion(c_masks["ventral_fins"] | fg_over_vfin, iterations=1)
        layers["ventral_fins"] = build_part_layer(
            canvas,
            c_masks["ventral_fins"],
            u_vfin & fg_over_vfin,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_dfin = c_masks["torso"] | c_masks["head_upper"]
        u_dfin = ndi.binary_erosion(c_masks["dorsal_fin"] | fg_over_dfin, iterations=1)
        layers["dorsal_fin"] = build_part_layer(
            canvas,
            c_masks["dorsal_fin"],
            u_dfin & fg_over_dfin,
            cushion_px=5,
            allowed_fill_mask=body_fill_m,
        )

        fg_over_jaw = c_masks["torso"] | c_masks["pectoral_fin_near"]
        u_jaw = ndi.binary_erosion(c_masks["lower_jaw"] | fg_over_jaw, iterations=1)
        layers["lower_jaw"] = build_part_layer(
            canvas,
            c_masks["lower_jaw"],
            u_jaw & fg_over_jaw,
            cushion_px=4,
            allowed_fill_mask=body_fill_m,
        )

        # IMPORTANT: `torso` cushion ONLY expands under foreground layers drawn ON TOP OF `torso`
        # (`head_upper`, `pectoral_fin_near`), NEVER over background fins!
        fg_over_torso = c_masks["head_upper"] | c_masks["pectoral_fin_near"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        layers["torso"] = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=body_fill_m,
        )

        layers["head_upper"] = build_part_layer(
            canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0
        )
        layers["pectoral_fin_near"] = build_part_layer(
            canvas, c_masks["pectoral_fin_near"], np.zeros_like(c_masks["pectoral_fin_near"]), cushion_px=0
        )

        pivots = {
            "dorsal_fin": to_canvas_pt(40.0, 31.0),
            "pectoral_fin_far": to_canvas_pt(26.0, 49.0),
            "pectoral_fin_near": to_canvas_pt(58.0, 55.0),
            "ventral_fins": to_canvas_pt(50.0, 66.0),
            "tail_rear": to_canvas_pt(35.0, 61.0),
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
        (mid_x - 17 - drift, max(2, top_y + 14 - drift // 2)),
        (mid_x + 17 + drift, max(2, top_y + 16 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (255, 222, 98, 255)
    return out


def synthesize_sharpedo_mega_frame(
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
    jaw_open = 0  # (2026-11-04) now only drives the small mouth_open gap, never a jaw drop
    dfin_dx, dfin_dy = 0, 0
    pfar_dx, pfar_dy = 0, 0
    pnear_dx, pnear_dy = 0, 0
    vfin_dx, vfin_dy = 0, 0
    tail_dx, tail_dy = 0, 0

    dfin_ang = 0.0
    pfar_ang = 0.0
    pnear_ang = 0.0
    vfin_ang = 0.0
    tail_ang = 0.0

    if action == "idle":
        # Predatory Hydro-Hovering + Ferocious Shark Roar/Bark + Torpedo Fin Roll Readiness:
        # 1. Streamlined shark body hovers smoothly in a predatory figure-8 current (`±1px` x, `-2..+1px` y).
        # 2. Towering dorsal fin (`dorsal_fin`), blade pectoral fins (`pectoral_fin_near/far`), and `ventral_fins`
        #    pivot and roll (`±3.5°..±4.5°`) with phase lag as Mega Sharpedo treads water.
        # 3. Massive saw-toothed `lower_jaw` opens `+1..+2px` during the primary shark roar (`frames 2..7`)
        #    and snaps/barks (`+1px` at `frames 11..13`), exposing the crimson mouth interior!
        body_dy = int(round(1.5 * math.sin(phase - 0.2)))
        body_dx = int(round(1.0 * math.cos(phase - 0.2) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        if frame_idx in (3, 4, 5):
            jaw_open = 2
        elif frame_idx in (2, 6, 7, 11, 12):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # (2026-11-04) jaw stays on the head: no chin drop

        dfin_dx, dfin_dy = body_dx, body_dy
        dfin_ang = 3.2 * math.sin(phase - 0.6) * facing

        pnear_dx = body_dx
        pnear_dy = body_dy + int(round(0.8 * math.sin(phase - 0.5)))
        pnear_ang = 4.5 * math.sin(phase - 0.5) * (-facing)

        pfar_dx = body_dx
        pfar_dy = body_dy + int(round(0.8 * math.sin(phase - 0.8)))
        pfar_ang = 4.5 * math.sin(phase - 0.8) * facing

        vfin_dx = body_dx
        vfin_dy = body_dy
        vfin_ang = 3.8 * math.sin(phase - 1.0) * facing

        tail_dx = body_dx
        tail_dy = body_dy + int(round(0.8 * math.sin(2.0 * phase)))
        tail_ang = 3.0 * math.sin(2.0 * phase) * facing

    elif action == "walk":
        # High-Speed Jet-Propulsion Torpedo Glide + Corkscrew Fin Roll + Menacing Jaw Snap:
        # 2-cycle jet-propulsion surge (`2.0 * phase`), forward-leaning hydro-glide (`sway_x`, `surge_y`),
        # vigorous counter-rolling pectoral fins (`±6.5°`), slicing dorsal fin (`±4.2°`), sweeping ventral fin (`±5.0°`),
        # vibrating rear jet nozzle, and rhythmic shark roar/bark (`+1..+2px`) on each propulsion beat!
        surge_x = int(round(2.0 * sin1 * facing))
        surge_y = int(round(1.5 * math.sin(2.0 * phase - 0.3)))

        torso_dx, torso_dy = surge_x, surge_y
        head_dx = surge_x + int(round(0.8 * cos1 * facing))
        head_dy = surge_y

        s2 = math.sin(2.0 * phase)
        if s2 > 0.65:
            jaw_open = 2
        elif s2 > 0.15:
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # (2026-11-04) jaw stays on the head: no chin drop

        dfin_dx, dfin_dy = surge_x, surge_y
        dfin_ang = 4.2 * math.sin(2.0 * phase - 0.5) * facing

        pnear_dx = surge_x + int(round(0.8 * cos1 * facing))
        pnear_dy = surge_y + int(round(1.0 * math.sin(2.0 * phase - 0.4)))
        pnear_ang = 6.5 * math.sin(2.0 * phase - 0.4) * (-facing)

        pfar_dx = surge_x + int(round(-0.8 * cos1 * facing))
        pfar_dy = surge_y + int(round(-1.0 * math.sin(2.0 * phase - 0.4)))
        pfar_ang = -6.5 * math.sin(2.0 * phase - 0.4) * facing

        vfin_dx = surge_x
        vfin_dy = surge_y
        vfin_ang = 5.0 * math.sin(2.0 * phase - 0.9) * facing

        tail_dx = surge_x
        tail_dy = surge_y + int(round(1.0 * math.sin(4.0 * phase)))
        tail_ang = 4.5 * math.sin(4.0 * phase) * facing

    elif action == "attack":
        # Signature Mega Sharpedo "Thunderous Shark Roar + Corkscrew Torpedo Aqua Jet Spin & Jaw Crunch (`かみくだく`)"!
        # Frames 0..3 (Wind-up & Wide Shark Roar):
        #   - Pulls streamlined torpedo body back (`-4px`), flares dorsal & pectoral blades (`-8°`),
        #   - Opens `lower_jaw` WIDE (`+2..+3px` ferocious shark roar!).
        # Frames 4..10 (Corkscrew Torpedo Spin Lunge & Devastating Jaw Crunch!):
        #   - Explosive jet-propelled forward torpedo lunge (`+7px` snout & body),
        #   - High-speed corkscrew barrel-roll oscillation across all fins (`pnear ±11°`, `pfar ∓11°`, `dfin ±7°`, `vfin ±8°`),
        #   - Giant saw-jaws clamp shut (`-2px` at frames 6..7) for a bone-crushing Strong Jaw Crunch (`かみくだく`) on impact,
        #     then unleash a follow-through roar (`+2px` at frames 9..10)!
        # Frames 11..15: Hydro-brake follow-through recovery to hover stance.
        lunge_x_table = [0, -2, -3, -4, 3, 6, 7, 7, 5, 3, 2, 0, -1, -1, 0, 0]
        lunge_y_table = [0, -1, -1, -2, 0, 1, 2, 1, 0, -1, 0, 1, 0, 0, 0, 0]

        body_dx = int(round(lunge_x_table[frame_idx] * facing))
        body_dy = lunge_y_table[frame_idx]

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + (int(round(1.0 * facing)) if frame_idx in (5, 6, 7) else 0)
        head_dy = body_dy

        # Jaw curve: wide +2..+3px roar on wind-up (1..4), snap shut -2px on Crunch impact (6..7), follow-through roar +2px (9..10)
        jaw_open_table = [0, 2, 3, 3, 2, 0, -2, -2, -1, 2, 2, 1, 0, 0, 0, 0]
        jaw_open = jaw_open_table[frame_idx]
        jaw_dx = head_dx
        jaw_dy = head_dy  # (2026-11-04) jaw stays on the head: no chin drop

        # Corkscrew Torpedo Roll angles across all 5 fin layers during strike frames (4..10)!
        roll_dfin = [0.0, -3.5, -5.5, -7.0, 6.5, -7.0, 7.0, -6.5, 5.5, -4.0, 2.5, -1.0, 0.0, 0.0, 0.0, 0.0]
        roll_pnear = [0.0, -5.0, -8.0, -10.0, 10.5, -11.5, 11.5, -10.5, 8.5, -6.0, 3.5, -1.5, 0.0, 0.0, 0.0, 0.0]
        roll_pfar = [0.0, 5.0, 8.0, 10.0, -10.5, 11.5, -11.5, 10.5, -8.5, 6.0, -3.5, 1.5, 0.0, 0.0, 0.0, 0.0]
        roll_vfin = [0.0, -4.0, -6.0, -7.5, 7.5, -8.5, 8.5, -7.5, 6.0, -4.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        roll_tail = [0.0, -3.0, -5.0, -6.0, 6.5, -7.0, 7.0, -6.5, 5.0, -3.5, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        corkscrew_dy = [0, 0, -1, -1, 1, -2, 2, -2, 1, -1, 0, 0, 0, 0, 0, 0]

        dfin_dx = body_dx
        dfin_dy = body_dy + (corkscrew_dy[frame_idx] // 2)
        dfin_ang = roll_dfin[frame_idx] * facing

        pnear_dx = body_dx + (int(round(1.0 * facing)) if frame_idx in (5, 6, 7) else 0)
        pnear_dy = body_dy + corkscrew_dy[frame_idx]
        pnear_ang = roll_pnear[frame_idx] * (-facing)

        pfar_dx = body_dx
        pfar_dy = body_dy - corkscrew_dy[frame_idx]
        pfar_ang = roll_pfar[frame_idx] * facing

        vfin_dx = body_dx
        vfin_dy = body_dy - (corkscrew_dy[frame_idx] // 2)
        vfin_ang = roll_vfin[frame_idx] * facing

        tail_dx = body_dx
        tail_dy = body_dy + corkscrew_dy[frame_idx]
        tail_ang = roll_tail[frame_idx] * facing

    elif action == "hurt":
        # Hydro-shock knockback + hit-shake + pained shark roar (`lower_jaw` open `+2..+3px`) + fins flung back + sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-4.0 * env * facing)) + jit
        body_dy = int(round(-1.2 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx = body_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-1.5 * env))

        if frame_idx in (2, 3, 4):
            jaw_open = 3
        elif frame_idx in (1, 5, 6):
            jaw_open = 2
        elif frame_idx in (7, 8):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy  # (2026-11-04) jaw stays on the head: no chin drop

        dfin_dx, dfin_dy = body_dx, body_dy
        dfin_ang = -5.5 * env * facing

        pnear_dx = body_dx + int(round(-1.0 * env * facing))
        pnear_dy = body_dy + int(round(-1.2 * env))
        pnear_ang = -7.5 * env * (-facing)

        pfar_dx = body_dx + int(round(-1.0 * env * facing))
        pfar_dy = body_dy + int(round(-1.2 * env))
        pfar_ang = 7.5 * env * facing

        vfin_dx, vfin_dy = body_dx, body_dy
        vfin_ang = -5.5 * env * facing

        tail_dx, tail_dy = body_dx, body_dy
        tail_ang = -5.0 * env * facing

    tail_rot = rotate_layer_nearest(layers["tail_rear"], tail_ang, pivots["tail_rear"])
    pfar_rot = rotate_layer_nearest(layers["pectoral_fin_far"], pfar_ang, pivots["pectoral_fin_far"])
    vfin_rot = rotate_layer_nearest(layers["ventral_fins"], vfin_ang, pivots["ventral_fins"])
    dfin_rot = rotate_layer_nearest(layers["dorsal_fin"], dfin_ang, pivots["dorsal_fin"])
    pnear_rot = rotate_layer_nearest(layers["pectoral_fin_near"], pnear_ang, pivots["pectoral_fin_near"])

    if not is_back:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(pfar_rot, pfar_dx, pfar_dy),
            shift_layer_int(vfin_rot, vfin_dx, vfin_dy),
            shift_layer_int(dfin_rot, dfin_dx, dfin_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(pnear_rot, pnear_dx, pnear_dy),
        ]
    else:
        comp = [
            shift_layer_int(tail_rot, tail_dx, tail_dy),
            shift_layer_int(pfar_rot, pfar_dx, pfar_dy),
            shift_layer_int(vfin_rot, vfin_dx, vfin_dy),
            shift_layer_int(dfin_rot, dfin_dx, dfin_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(pnear_rot, pnear_dx, pnear_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    # (2026-11-04) small mouth opening that follows the face (1 px growl, 2 px only at the attack roar peak)
    if not is_back and "mouth_line" in pivots and jaw_open > 0:
        if action == "attack":
            mouth_amt = 2 if jaw_open >= 3 else 1
        else:
            mouth_amt = 1
        line = mouth_open.transform_points(pivots["mouth_line"], dx=jaw_dx, dy=jaw_dy)
        out = mouth_open.open_mouth(
            out, line, amount=mouth_amt, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=0.55, normal=(0.0, -1.0)
        )
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--gifs", action="store_true", help="also rewrite front/back(.reverse).gif from idle")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    front_raw = np.array(Image.open(MEGA_DIR / "front.png").convert("RGBA"))
    back_raw = np.array(Image.open(MEGA_DIR / "back.png").convert("RGBA"))

    palette_rgba = extract_palette(front_raw, back_raw)

    # Tight bounds of front (66x73) and back (67x76)
    max_w, max_h = 67, 76
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
                frame_rgba = synthesize_sharpedo_mega_frame(
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

    # (2026-11-04) front/back gifs are left untouched unless --gifs is given.
    if args.gifs:
        save_transparent_gif(front_frames, MEGA_DIR / "front.gif", duration_ms=75)
        save_transparent_gif(front_rev_frames, MEGA_DIR / "front-reverse.gif", duration_ms=75)
        save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
        save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)

    meta = {
        "generator": "forge_sharpedo_mega_actions",
        "v": "sharpedo_mega_v1m",
        "key": "0319_sharpedo-mega",
        "dex": 319,
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
        if args.gifs:
            for gif_name in ("front.gif", "front-reverse.gif", "back.gif", "back-reverse.gif"):
                shutil.copy2(MEGA_DIR / gif_name, SYNC_MEGA_DIR / gif_name)

    print(f"Built 0319_sharpedo-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
