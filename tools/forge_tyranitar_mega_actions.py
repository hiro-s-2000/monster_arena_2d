#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Tyranitar (`0248_tyranitar-mega`) modeled directly after Normal & Mega Tyranitar's
official GIF motion (`0248_tyranitar/normal/front.gif` & `mega/tyranitar-mega/front.gif`):
  - Exact color & connected-component anatomical layer separation (9 independent rig layers):
      1. `tail` (massive multi-spiked armored tail — completely separated from the hind legs so the tail
                 never tears or steps with the feet; sways with heavy phase delay)
      2. `back_spikes` (giant black/green dorsal spikes protruding from the shoulders and back —
                        independent phase-delayed kaiju dorsal tremor & flare during roar/attack)
      3. `arm_far` (far-side segmented green clawed arm with 3 white/grey claws)
      4. `foot_L` (far-side full muscular leg: thigh, pointed knee spike, ankle, and 3-clawed foot —
                   100% grounded `(0, 0)` in `idle` while the upper body flexes at the hip sockets,
                   and heavy alternating kaiju stomp in `walk`)
      5. `torso` (armored green chest with glowing red eye-like pectoral cores, segmented red belly core,
                  and pectoral shoulder horn)
      6. `foot_R` (near-side spherical thigh with black armor patches, pointed knee spike, ankle, and
                   3-clawed foot — 100% grounded `(0, 0)` in `idle` and alternating stomp in `walk`)
      7. `lower_jaw` (armored lower jaw, lower white fangs, and orange tongue — moves with the head; since
                      2026-11-04 the mouth opens as a small lens along the mouth line via `mouth_open.py`
                      (1 px growl in idle/walk/hurt, 2 px roar in the attack wind-up), then shuts on impact)
      8. `head_upper` (tall pale-green center horn, upper head, fierce orange eye, upper white fang,
                       and 3 rear head-crest spikes — 100% rigid integer translation so the face and eye
                       never deform by a single pixel)
      9. `arm_near` (foreground segmented green clawed arm with 3 white/grey claws)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import mouth_open  # noqa: E402

# 2026-11-04 mouth fix: closed mouth line on native 96x96 front.png (facing left), ordered hinge -> lip tip.
# The mouth opens as a small lens carved just below this line (inside the face) instead of dropping the jaw.
MOUTH_LINE_NATIVE = [(29, 25), (26, 25), (23, 25)]
MOUTH_DEEP = (16, 16, 16)      # outline colour (darkest palette entry)
MOUTH_INNER = (148, 41, 0)     # existing dark red-brown mouth interior colour

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen2/0248_tyranitar/mega/tyranitar-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

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
        | ((r == 74) & (g == 90) & (b == 57))
    )


def is_green_armor(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 172) & (g == 205) & (b == 156))
        | ((r == 115) & (g == 156) & (b == 98))
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


def composite_layers(canvas_shape: Tuple[int, int, int], layers: List[np.ndarray]) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=np.uint8)
    for lyr in layers:
        m = lyr[..., 3] >= 128
        out[m] = lyr[m]
    return out


def segment_front_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `front.png` (facing left) into 9 exact anatomical masks
    following the artist's 1px dark outlines:
      - `lower_jaw`: Lower jaw chin, lower white fangs, and orange tongue (`x=20..29, y=26..31`).
      - `head_upper`: Tall pale-green horn, upper head, fierce orange eye, upper fang, and 3 rear head spikes (`x=19..46, y=6..30`).
      - `arm_near`: Foreground right segmented arm + 3 white/grey claws (`x=38..60, y=43..63`).
      - `back_spikes`: Left 3 dorsal spikes (`x=3..22, y=29..65`) + right giant mid & lower dorsal spikes (`x=45..83, y=39..67`).
      - `arm_far`: Far-side left segmented arm + 3 white/grey claws (`x=4..21, y=44..60`).
      - `foot_L`: Far-side full left leg (thigh, pointed knee spike, ankle, and 3 claws) outside the black belly crease (`x=5..27, y=64..88`).
      - `foot_R`: Near-side full right leg (spherical thigh with black patches, knee spike, ankle, and 3 claws) (`x=37..60, y=61..89`).
      - `tail`: Massive multi-spiked armored tail extending to the right (`x=55..92, y=60..86`).
      - `torso`: Central green chest with glowing red eye-like cores, red segmented belly core, and pectoral shoulder horn.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Lower jaw (jaw chin, lower fangs, orange tongue)
    jaw_zone = (
        ((yy == 26) & (xx >= 23) & (xx <= 29))
        | ((yy == 27) & (xx >= 22) & (xx <= 27))
        | ((yy == 28) & (xx >= 21) & (xx <= 27))
        | ((yy == 29) & (xx >= 20) & (xx <= 27))
        | ((yy == 30) & (xx >= 21) & (xx <= 27))
        | ((yy == 31) & (xx >= 22) & (xx <= 25))
    )
    lower_jaw = a & jaw_zone

    # 2. Head upper (tall horn, upper head, eye, upper fangs, 3 head-back spikes)
    head_zone = (
        (yy <= 25)
        | ((yy >= 26) & (yy <= 28) & (xx >= 19) & (xx <= 46))
        | ((yy == 29) & (xx >= 20) & (xx <= 42))
        | ((yy == 30) & (xx >= 21) & (xx <= 40))
    )
    head_upper = a & head_zone & (~lower_jaw)

    # 3. Arm Near (foreground right clawed arm + 3 white/grey claws)
    arm_near_zone = (
        ((yy == 43) & (xx >= 42) & (xx <= 46))
        | ((yy == 44) & (xx >= 40) & (xx <= 48))
        | ((yy == 45) & (xx >= 39) & (xx <= 49))
        | ((yy == 46) & (xx >= 39) & (xx <= 50))
        | ((yy == 47) & (xx >= 38) & (xx <= 51))
        | ((yy == 48) & (xx >= 38) & (xx <= 52))
        | ((yy == 49) & (xx >= 38) & (xx <= 53))
        | ((yy == 50) & (xx >= 38) & (xx <= 55))
        | ((yy == 51) & (xx >= 38) & (xx <= 55))
        | ((yy == 52) & (xx >= 39) & (xx <= 56))
        | ((yy == 53) & (xx >= 39) & (xx <= 57))
        | ((yy == 54) & (xx >= 40) & (xx <= 58))
        | ((yy == 55) & (xx >= 41) & (xx <= 58))
        | ((yy == 56) & (xx >= 42) & (xx <= 58))
        | ((yy == 57) & (xx >= 44) & (xx <= 58))
        | ((yy == 58) & (xx >= 46) & (xx <= 58))
        | ((yy == 59) & (xx >= 49) & (xx <= 58))
        | ((yy == 60) & (xx >= 50) & (xx <= 58))
        | ((yy == 61) & (xx >= 51) & (xx <= 57))
        | ((yy == 62) & (xx >= 52) & (xx <= 57))
        | ((yy == 63) & (xx == 56))
    )
    arm_near = a & arm_near_zone

    # 4. Back Spikes (Left 3 dorsal spikes + Right mid & lower dorsal spikes)
    spike_L1 = (
        ((yy >= 29) & (yy <= 33) & (xx <= 19))
        | ((yy == 34) & (xx <= 20))
        | ((yy >= 35) & (yy <= 40) & (xx <= 22))
        | ((yy >= 41) & (yy <= 42) & (xx <= 21))
        | ((yy == 43) & (xx <= 20))
    )
    spike_L2 = (
        ((yy == 45) & (xx <= 11))
        | ((yy == 46) & (xx <= 12))
        | ((yy == 47) & (xx <= 11))
        | ((yy == 48) & (xx <= 10))
    )
    spike_L3 = (
        ((yy == 59) & (xx >= 14) & (xx <= 17))
        | ((yy == 60) & (xx >= 10) & (xx <= 17))
        | ((yy == 61) & (xx >= 9) & (xx <= 17))
        | ((yy == 62) & (xx >= 8) & (xx <= 17))
        | ((yy == 63) & (xx >= 7) & (xx <= 16))
        | ((yy == 64) & (xx >= 6) & (xx <= 15))
        | ((yy == 65) & (xx >= 6) & (xx <= 7))
    )
    spike_R2 = (
        ((yy == 39) & (xx >= 65))
        | ((yy == 40) & (xx >= 60))
        | ((yy == 41) & (xx >= 58))
        | ((yy == 42) & (xx >= 50))
        | ((yy == 43) & (xx >= 48))
        | ((yy >= 44) & (yy <= 50) & (xx >= 45))
    ) & (~arm_near)
    spike_R3 = (
        ((yy >= 55) & (yy <= 58) & (xx >= 58))
        | ((yy >= 59) & (yy <= 62) & (xx >= 58) & (xx <= 72))
        | ((yy == 63) & (xx >= 55) & (xx <= 72))
        | ((yy >= 64) & (yy <= 65) & (xx >= 56) & (xx <= 72))
        | ((yy >= 66) & (yy <= 67) & (xx >= 65) & (xx <= 72))
    ) & (~arm_near)
    back_spikes = a & (spike_L1 | spike_L2 | spike_L3 | spike_R2 | spike_R3)

    # 5. Arm Far (left extended clawed arm)
    arm_far_zone = (
        ((yy == 44) & (xx >= 16) & (xx <= 20))
        | ((yy == 45) & (xx >= 13) & (xx <= 20))
        | ((yy == 46) & (xx >= 13) & (xx <= 20))
        | ((yy == 47) & (xx >= 12) & (xx <= 20))
        | ((yy == 48) & (xx >= 11) & (xx <= 21))
        | ((yy == 49) & (xx >= 9) & (xx <= 20))
        | ((yy == 50) & (xx >= 8) & (xx <= 21))
        | ((yy == 51) & (xx >= 7) & (xx <= 21))
        | ((yy == 52) & (xx >= 7) & (xx <= 19))
        | ((yy == 53) & (xx >= 6) & (xx <= 18))
        | ((yy == 54) & (xx >= 6) & (xx <= 17))
        | ((yy >= 55) & (yy <= 60) & (xx <= 15))
    )
    arm_far = a & arm_far_zone & (~back_spikes)

    # 6. Grounded Left Hind Foot (`foot_L`: ankle and 3 white/grey claws below the horizontal ankle crease `y >= 81`)
    foot_L_zone = (yy >= 81) & (xx <= 24)
    foot_L = a & foot_L_zone

    # 7. Grounded Right Hind Foot (`foot_R`: ankle and 3 white/grey claws at `y=82..89, x=44..60`)
    foot_R_zone = (
        ((yy >= 82) & (yy <= 83) & (xx >= 45) & (xx <= 56))
        | ((yy >= 84) & (yy <= 89) & (xx >= 44) & (xx <= 60))
    )
    foot_R = a & foot_R_zone

    # 8. Tail (massive multi-spiked tail to the right of the right thigh outline `x >= 57`)
    tail_zone = (
        ((yy >= 60) & (yy <= 65) & (xx >= 73))
        | ((yy >= 61) & (yy <= 66) & (xx >= 57) & (~arm_near) & (~back_spikes))
        | ((yy >= 67) & (yy <= 72) & (xx >= 58))
        | ((yy >= 73) & (yy <= 76) & (xx >= 57))
        | ((yy >= 77) & (yy <= 80) & (xx >= 56))
        | ((yy >= 81) & (yy <= 83) & (xx >= 57))
        | ((yy >= 84) & (yy <= 86) & (xx >= 62))
    )
    tail = a & tail_zone & (~arm_near) & (~back_spikes) & (~foot_R)

    used = lower_jaw | head_upper | arm_near | back_spikes | arm_far | foot_L | foot_R | tail
    torso = a & (~used)

    return {
        "tail": tail,
        "back_spikes": back_spikes,
        "arm_far": arm_far,
        "foot_L": foot_L,
        "torso": torso,
        "foot_R": foot_R,
        "lower_jaw": lower_jaw,
        "head_upper": head_upper,
        "arm_near": arm_near,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` (facing right) into matching 9 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Head upper (horn + head + 3 left head crest spikes + eye + upper jaw)
    head_zone = (
        (yy <= 17)
        | ((yy >= 18) & (yy <= 29) & (xx >= 50))
        | ((yy == 30) & (xx >= 51) & (xx <= 63))
    )
    head_upper = a & head_zone

    # 2. Back Spikes (3 giant left dorsal spikes)
    spikes_zone = (
        ((yy >= 18) & (yy <= 29) & (xx <= 49))
        | ((yy == 30) & (xx <= 50))
        | ((yy >= 31) & (yy <= 36) & (xx <= 51))
        | ((yy == 37) & (xx <= 47))
        | ((yy >= 38) & (yy <= 40) & (xx <= 46))
        | ((yy == 41) & (xx <= 45))
        | ((yy >= 42) & (yy <= 44) & (xx <= 46))
        | ((yy >= 45) & (yy <= 46) & (xx <= 44))
        | ((yy >= 47) & (yy <= 51) & (xx <= 43))
        | ((yy == 52) & (xx <= 42))
        | ((yy >= 53) & (yy <= 54) & (xx <= 40))
    )
    back_spikes = a & spikes_zone

    # 3. Arm Near (extended right arm + claws)
    arm_near_zone = (
        ((yy == 39) & (xx >= 68))
        | ((yy >= 40) & (yy <= 43) & (xx >= 67))
        | ((yy >= 44) & (yy <= 47) & (xx >= 68))
        | ((yy >= 48) & (yy <= 49) & (xx >= 67))
        | ((yy >= 50) & (yy <= 51) & (xx >= 68))
        | ((yy == 52) & (xx >= 67))
        | ((yy == 53) & (xx >= 68))
        | ((yy >= 54) & (yy <= 57) & (xx >= 71))
    )
    arm_near = a & arm_near_zone

    # 4. Far-Left Leg Edge (`foot_L`) & Grounded Right Foot (`foot_R`)
    foot_L_zone = (yy >= 58) & (yy <= 70) & (xx >= 33) & (xx <= 39)
    foot_L = a & foot_L_zone

    foot_R_zone = (yy >= 78) & (yy <= 85) & (xx >= 63)
    foot_R = a & foot_R_zone

    # 5. Tail (circular socket on lower back + massive multi-spiked left tail)
    tail_zone = (
        ((yy == 64) & (xx >= 37) & (xx <= 40))
        | ((yy == 65) & (xx >= 38) & (xx <= 45))
        | ((yy == 66) & (xx >= 38) & (xx <= 46))
        | ((yy >= 67) & (yy <= 68) & (xx >= 37) & (xx <= 47))
        | ((yy == 69) & (xx <= 50))
        | ((yy >= 70) & (yy <= 71) & (xx <= 50))
        | ((yy >= 72) & (yy <= 74) & (xx <= 51))
        | ((yy >= 75) & (yy <= 76) & (xx <= 50))
        | ((yy == 77) & (xx <= 49))
        | ((yy >= 78) & (yy <= 93) & (xx <= 52))
    )
    tail = a & tail_zone & (~foot_L)

    used = head_upper | back_spikes | arm_near | foot_L | foot_R | tail
    torso = a & (~used)
    empty = np.zeros_like(a)

    return {
        "tail": tail,
        "back_spikes": back_spikes,
        "arm_far": empty,
        "foot_L": foot_L,
        "torso": torso,
        "foot_R": foot_R,
        "lower_jaw": empty,
        "head_upper": head_upper,
        "arm_near": arm_near,
    }


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


def build_rig_layers(
    base_96: np.ndarray,
    is_back: bool,
    flip_x: bool,
    cell_w: int,
    cell_h: int,
    foot_y: int,
) -> Dict[str, np.ndarray]:
    """
    Segments native `front.png` or `back.png`, optionally flips horizontally for SE/E/NW,
    places onto the target `(cell_h, cell_w)` canvas with feet aligned to `foot_y`,
    and builds cushioned RGBA layers.
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
    dst_y0 = (foot_y + 5 - sh) if is_back else (foot_y - sh)
    mouth_line = None if is_back else mouth_open.transform_points(
        MOUTH_LINE_NATIVE, dx=dst_x0 - x0, dy=dst_y0 - y0, mirror_w=(base_96.shape[1] if flip_x else None)
    )

    canvas = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
    canvas[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = src_96[y0:y1, x0:x1]

    c_masks: Dict[str, np.ndarray] = {}
    for k, m96 in masks_96.items():
        cm = np.zeros((cell_h, cell_w), dtype=bool)
        cm[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = m96[y0:y1, x0:x1]
        c_masks[k] = cm

    green_fill_m = is_green_armor(canvas)
    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front view draw order:
        # 1. tail -> 2. back_spikes -> 3. arm_far -> 4. foot_L -> 5. foot_R -> 6. torso -> 7. lower_jaw -> 8. head_upper -> 9. arm_near
        fg_over_tail = c_masks["back_spikes"] | c_masks["torso"] | c_masks["arm_near"]
        u_tail = ndi.binary_erosion(c_masks["tail"] | fg_over_tail, iterations=1)
        layers["tail"] = build_part_layer(
            canvas,
            c_masks["tail"],
            u_tail & fg_over_tail,
            cushion_px=6,
            allowed_fill_mask=green_fill_m,
        )

        fg_over_spikes = c_masks["arm_far"] | c_masks["torso"] | c_masks["arm_near"]
        u_spikes = ndi.binary_erosion(c_masks["back_spikes"] | fg_over_spikes, iterations=1)
        layers["back_spikes"] = build_part_layer(
            canvas,
            c_masks["back_spikes"],
            u_spikes & fg_over_spikes,
            cushion_px=5,
            allowed_fill_mask=green_fill_m,
        )

        u_afar = ndi.binary_erosion(c_masks["arm_far"] | c_masks["torso"], iterations=1)
        layers["arm_far"] = build_part_layer(
            canvas,
            c_masks["arm_far"],
            u_afar & c_masks["torso"],
            cushion_px=5,
            allowed_fill_mask=green_fill_m,
        )

        layers["foot_L"] = build_foot_stem_layer(canvas, c_masks["foot_L"], c_masks["torso"], stem_px=4)
        layers["foot_R"] = build_foot_stem_layer(canvas, c_masks["foot_R"], c_masks["torso"], stem_px=4)

        fg_over_torso = c_masks["lower_jaw"] | c_masks["head_upper"] | c_masks["arm_near"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=green_fill_m,
        )
        # (2026-11-04) No rectangular mouth-interior pocket any more: the jaw no longer drops; the mouth
        # opening is carved along the mouth line with `mouth_open.open_mouth` after compositing.
        layers["torso"] = torso_lyr

        layers["lower_jaw"] = build_part_layer(canvas, c_masks["lower_jaw"], np.zeros_like(c_masks["lower_jaw"]), cushion_px=0)
        layers["head_upper"] = build_part_layer(canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0)
        layers["arm_near"] = build_part_layer(canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0)

    else:
        # Back view draw order:
        # 1. back_spikes -> 2. foot_L -> 3. foot_R -> 4. torso -> 5. tail -> 6. head_upper -> 7. arm_near
        layers["arm_far"] = np.zeros_like(canvas)
        layers["lower_jaw"] = np.zeros_like(canvas)

        fg_over_spikes = c_masks["torso"] | c_masks["head_upper"]
        u_spikes = ndi.binary_erosion(c_masks["back_spikes"] | fg_over_spikes, iterations=1)
        layers["back_spikes"] = build_part_layer(
            canvas,
            c_masks["back_spikes"],
            u_spikes & fg_over_spikes,
            cushion_px=5,
            allowed_fill_mask=green_fill_m,
        )

        fg_over_footL = c_masks["torso"] | c_masks["tail"]
        u_footL = ndi.binary_erosion(c_masks["foot_L"] | fg_over_footL, iterations=1)
        layers["foot_L"] = build_part_layer(
            canvas,
            c_masks["foot_L"],
            u_footL & fg_over_footL,
            cushion_px=5,
            allowed_fill_mask=green_fill_m,
        )

        layers["foot_R"] = build_foot_stem_layer(canvas, c_masks["foot_R"], c_masks["torso"], stem_px=4)

        fg_over_torso = c_masks["tail"] | c_masks["head_upper"] | c_masks["arm_near"]
        u_torso = ndi.binary_erosion(c_masks["torso"] | fg_over_torso, iterations=1)
        torso_lyr = build_part_layer(
            canvas,
            c_masks["torso"],
            u_torso & fg_over_torso,
            cushion_px=6,
            allowed_fill_mask=green_fill_m,
        )
        # Preserve the 1px top/bottom outline of the shoulder cylinder (`arm_near` within 3px of `torso`)
        # so when `arm_near` extends forward in `attack`, the shoulder outline never notches!
        dist_to_torso = ndi.distance_transform_edt(~c_masks["torso"])
        sh_seam = c_masks["arm_near"] & (dist_to_torso >= 1) & (dist_to_torso <= 3)
        torso_lyr[sh_seam] = canvas[sh_seam]
        layers["torso"] = torso_lyr

        layers["tail"] = build_part_layer(canvas, c_masks["tail"], np.zeros_like(c_masks["tail"]), cushion_px=0)
        layers["head_upper"] = build_part_layer(canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0)
        layers["arm_near"] = build_part_layer(canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0)

    layers["_mouth_line"] = mouth_line
    return layers


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
        (mid_x - 16 - drift, max(2, top_y + 18 - drift // 2)),
        (mid_x + 16 + drift, max(2, top_y + 19 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (254, 254, 254, 255)
    return out


def synthesize_tyranitar_mega_frame(
    layers: Dict[str, np.ndarray],
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
    anear_dx, anear_dy = 0, 0
    afar_dx, afar_dy = 0, 0
    spikes_dx, spikes_dy = 0, 0
    tail_dx, tail_dy = 0, 0
    foot_L_dx, foot_L_dy = 0, 0
    foot_R_dx, foot_R_dy = 0, 0

    if action == "idle":
        # Modeled directly after Normal & Mega Tyranitar's official GIFs (`front.gif` & `back.gif`):
        # 1. Both muscular hind legs (`foot_L`, `foot_R`) stay 100% planted on the ground `(0, 0)`.
        # 2. Heavy Kaiju breathing crouch at the hip sockets: the red-cored chest (`torso`) and
        #    crowned head (`head_upper`) rise and sink (`-1..+1px`) while leaning slightly into the breath.
        # 3. `lower_jaw` opens `+1px` during the inhalation/growl peak, revealing the dark mouth interior.
        # 4. Both clawed arms (`arm_near`, `arm_far`) pump rhythmically up/down and forward/back.
        # 5. The giant dorsal spikes (`back_spikes`) and massive multi-pronged tail (`tail`) sway
        #    heavily with phase delay!
        body_dy = int(round(1.2 * math.sin(phase - 0.2)))
        body_dx = int(round(0.6 * max(0.0, math.sin(phase - 0.2)) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        jaw_open = 1 if math.sin(phase - 0.1) > 0.20 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy
        mouth_amt = jaw_open

        anear_dx = body_dx + int(round(1.2 * math.sin(phase - 0.6) * facing))
        anear_dy = body_dy + int(round(1.4 * math.sin(phase - 0.5)))

        afar_dx = body_dx + int(round(1.2 * math.sin(phase - 0.9) * facing))
        afar_dy = body_dy + int(round(1.4 * math.sin(phase - 0.8)))

        spikes_dx = body_dx
        spikes_dy = body_dy + int(round(0.9 * math.sin(phase - 0.5)))

        tail_dx = body_dx
        tail_dy = int(round(1.5 * math.sin(phase - 1.1)))

    elif action == "walk":
        # `biped_kaiju_stomper`: Earth-shaking bipedal Kaiju stomp gait!
        # Alternating left/right full-leg stomp (`lift_px=2, stride_px=2`),
        # 2-beat upper-body bounce, counter-swinging clawed arms (`arm_near` vs `arm_far`),
        # rhythmic jaw roar (`+1px`) on each stomp, and heavy phase-delayed tail counterbalance!
        bounce = int(round(-1.5 * abs(sin1)))
        sway_x = int(round(1.0 * sin1 * facing))

        torso_dx, torso_dy = sway_x, bounce
        head_dx, head_dy = sway_x, bounce

        jaw_open = 1 if math.sin(2.0 * phase) > 0.25 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy
        mouth_amt = jaw_open

        anear_dx = sway_x + int(round(1.6 * sin1 * facing))
        anear_dy = bounce + int(round(1.4 * cos1))

        afar_dx = sway_x + int(round(-1.6 * sin1 * facing))
        afar_dy = bounce + int(round(-1.4 * cos1))

        spikes_dx = sway_x
        spikes_dy = bounce + int(round(1.0 * math.sin(2.0 * phase - 0.5)))

        tail_dx = sway_x
        tail_dy = bounce + int(round(1.6 * math.sin(2.0 * phase - 1.0)))

        if is_back:
            foot_L_dx, foot_L_dy = foot_step_offset(phase, facing, lift_px=1, stride_px=1)
        else:
            foot_L_dx, foot_L_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        foot_R_dx, foot_R_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Mega Tyranitar "Stone Edge / Crunch / Earthquake / Hyper Beam" Kaiju Rampage!
        # Frames 0..3: Wind-up — pull massive body back (`-3px`), open `lower_jaw` wide (`+2px` ferocious roar!),
        #              cock both clawed arms back, and flare dorsal `back_spikes`!
        # Frames 4..9: Ferocious forward Kaiju lunge (`+5px` forward!), front leg stomps forward (`+3px`),
        #              jaws clamp shut (`0px` Crunch impact on frames 6..8!) while both clawed arms
        #              slam forward (`+6px`!) and dorsal spikes + giant tail whip upward!
        # Frames 10..15: Heavy follow-through recovery to stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.65 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        body_dx = int(round(4.5 * k * facing))
        if frame_idx in (1, 2, 3):
            body_dy = -1 if frame_idx in (2, 3) else 0
        elif frame_idx in (5, 6, 7, 8):
            body_dy = 1
        else:
            body_dy = 0

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        # Jaw drops +2px wide during wind-up & early lunge (frames 1..5), then snaps shut (0px) on impact (frames 6..8)!
        if frame_idx in (2, 3, 4):
            jaw_open = 2
        elif frame_idx in (1, 5):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy
        mouth_amt = jaw_open

        # Both clawed arms slam powerfully forward (+6px) on strike frames!
        anear_dx = int(round(5.8 * k * facing))
        anear_dy = body_dy + (int(round(-1.5 * k)) if frame_idx in (4, 5) else int(round(1.5 * max(0.0, k))))
        afar_dx = int(round(5.5 * k * facing))
        afar_dy = body_dy + int(round(1.5 * k))

        # Dorsal spikes flare upward (-1px) during lunge peak
        spikes_dx = body_dx
        spikes_dy = body_dy - (1 if frame_idx in (5, 6, 7, 8) else 0)

        # Giant tail counter-whips upward during forward smash
        tail_dx = body_dx
        tail_dy = body_dy + int(round(-1.8 * k))

        # Lead leg stomps into the lunge while rear leg braces cohesively
        lead_x = int(round(3.2 * max(0.0, k) * facing)) if k >= 0 else int(round(2.0 * k * facing))
        rear_x = int(round(2.5 * k * facing))
        if not is_back:
            foot_L_dx, foot_L_dy = lead_x, 0
            foot_R_dx, foot_R_dy = rear_x, 0
        else:
            foot_R_dx, foot_R_dy = lead_x, 0
            foot_L_dx, foot_L_dy = rear_x, 0

    elif action == "hurt":
        # Heavy Kaiju knockback + hit-shake + jaw jarred open (+1px) + arms flung back + hurt sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-3.5 * env * facing)) + jit
        body_dy = int(round(-1.0 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy

        jaw_dx = head_dx
        jaw_dy = head_dy
        mouth_amt = 1 if 1 <= frame_idx <= 7 else 0

        anear_dx = body_dx + int(round(-1.2 * env * facing))
        anear_dy = body_dy + int(round(-1.2 * env))
        afar_dx = body_dx + int(round(-1.2 * env * facing))
        afar_dy = body_dy + int(round(-1.2 * env))

        spikes_dx = body_dx
        spikes_dy = body_dy - (1 if 2 <= frame_idx <= 5 else 0)

        tail_dx = body_dx
        tail_dy = int(round(1.0 * env))

        slide_x = int(round(body_dx * 0.7))
        foot_L_dx, foot_R_dx = slide_x, slide_x

    if not is_back:
        comp = [
            shift_layer_int(layers["tail"], tail_dx, tail_dy),
            shift_layer_int(layers["back_spikes"], spikes_dx, spikes_dy),
            shift_layer_int(layers["arm_far"], afar_dx, afar_dy),
            shift_layer_int(layers["foot_L"], foot_L_dx, foot_L_dy),
            shift_layer_int(layers["foot_R"], foot_R_dx, foot_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(layers["arm_near"], anear_dx, anear_dy),
        ]
    else:
        comp = [
            shift_layer_int(layers["back_spikes"], spikes_dx, spikes_dy),
            shift_layer_int(layers["foot_L"], foot_L_dx, foot_L_dy),
            shift_layer_int(layers["foot_R"], foot_R_dx, foot_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["tail"], tail_dx, tail_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(layers["arm_near"], anear_dx, anear_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    line = layers.get("_mouth_line")
    if line and mouth_amt > 0:
        line = mouth_open.transform_points(line, dx=head_dx, dy=head_dy)
        out = mouth_open.open_mouth(out, line, amount=mouth_amt, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=0.6)
    if action == "hurt":
        out = add_hurt_sparks(out, frame_idx)
    return out


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

    # Tight bounds of front (90x84) and back (82x92)
    cell_w = 114
    cell_h = 108
    foot_y = 98

    dir_layers: Dict[str, Tuple[Dict[str, np.ndarray], float, bool]] = {}
    for d_name in DIRS:
        use_back = d_name in ("N", "NE", "NW")
        src_96 = back_raw if use_back else front_raw
        if not use_back:
            flip_x = d_name in ("SE", "E")
            facing = 1.0 if flip_x else -1.0
        else:
            flip_x = d_name in ("NW",)
            facing = -1.0 if flip_x else 1.0

        layers = build_rig_layers(
            src_96, is_back=use_back, flip_x=flip_x, cell_w=cell_w, cell_h=cell_h, foot_y=foot_y
        )
        dir_layers[d_name] = (layers, facing, use_back)

    sheets_by_action: Dict[str, np.ndarray] = {}
    master_rows: List[np.ndarray] = []

    for act in ACTIONS:
        act_sheet = np.zeros((8 * cell_h, N_FRAMES * cell_w, 4), dtype=np.uint8)
        for d_idx, d_name in enumerate(DIRS):
            layers, facing, use_back = dir_layers[d_name]
            for f_idx in range(N_FRAMES):
                frame_rgba = synthesize_tyranitar_mega_frame(
                    layers,
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

    # Note: Existing 102-frame official `front.gif` / `back.gif` in `MEGA_DIR` are high-quality
    # noise-free official animations, so we preserve them untouched per specification!

    meta = {
        "schema": "agent-sprite-forge.meta.v2",
        "key": "0248_tyranitar-mega",
        "archetype": "biped_kaiju_stomper",
        "rig_parts": [
            "tail",
            "back_spikes",
            "arm_far",
            "foot_L",
            "torso",
            "foot_R",
            "lower_jaw",
            "head_upper",
            "arm_near",
        ],
        "Idle": "../../monsterSprite/gen2/0248_tyranitar/mega/tyranitar-mega/agent-sprite-forge/idle.png",
        "Walk": "../../monsterSprite/gen2/0248_tyranitar/mega/tyranitar-mega/agent-sprite-forge/walk.png",
        "Attack": "../../monsterSprite/gen2/0248_tyranitar/mega/tyranitar-mega/agent-sprite-forge/attack.png",
        "Hurt": "../../monsterSprite/gen2/0248_tyranitar/mega/tyranitar-mega/agent-sprite-forge/hurt.png",
        "Sheet": "../../monsterSprite/gen2/0248_tyranitar/mega/tyranitar-mega/agent-sprite-forge/sheet.png",
        "Preview": "../../monsterSprite/gen2/0248_tyranitar/mega/tyranitar-mega/agent-sprite-forge/preview.png",
        "cols": N_FRAMES,
        "rows": 8,
        "cellW": cell_w,
        "cellH": cell_h,
        "footY": foot_y,
        "paletteCount": int(len(palette_rgba)),
        "directions": DIRS,
        "actions": {
            "idle": {"rowStart": 0, "rows": 8, "frames": N_FRAMES, "fps": 12},
            "walk": {"rowStart": 8, "rows": 8, "frames": N_FRAMES, "fps": 14},
            "attack": {"rowStart": 16, "rows": 8, "frames": N_FRAMES, "fps": 16},
            "hurt": {"rowStart": 24, "rows": 8, "frames": N_FRAMES, "fps": 12},
        },
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Successfully forged Mega Tyranitar 8-dir x 4-action bundle in {OUT_DIR} ({cell_w}x{cell_h} cell)")


if __name__ == "__main__":
    main()
