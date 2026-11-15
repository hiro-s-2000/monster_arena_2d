#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Ampharos (`0181_ampharos-mega`) modeled directly after Normal Ampharos's
official GIF motion (`0181_ampharos/normal/front.gif` & `back.gif`):
  - Exact color & connected-component anatomical layer separation:
      1. `head` (yellow head/snout, black/white eye, forehead red orb & white setting,
                 black-and-yellow striped ears/horns, and top head wool crest —
                 100% rigid integer translation so face/eye/horns never deform by 1 pixel)
      2. `mane_upper` (large billowing white/silver cloud wool mane behind the head and neck)
      3. `tail_wool` (magnificent cloud-covered tail with 4 embedded red orbs)
      4. `arm_far` (far-side yellow flipper arm)
      5. `arm_near` (foreground yellow flipper arm overlapping torso)
      6. `torso` (striped neck, yellow chest, white/silver belly, and yellow thighs)
      7. `foot_L`, `foot_R` (grounded hind feet in `idle`, alternating bipedal step in `walk`,
                             completely separated from `tail_wool`)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen2/0181_ampharos/mega/ampharos-mega"
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
        | ((r == 115) & (g == 115) & (b == 115))
        | ((r == 115) & (g == 75) & (b == 34))
        | ((r == 128) & (g == 37) & (b == 6))
    )


def build_part_layer(
    src: np.ndarray,
    mask: np.ndarray,
    foreground_mask: np.ndarray,
    cushion_px: int = 4,
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
    Segments native 96x96 `front.png` (facing left) into exact anatomical masks:
      - `head`: Yellow head/snout, black/white eye, forehead red orb & white setting,
                striped ears/horns, and top head wool crest (`y <= 33`).
      - `mane_upper`: Billowing white/silver cloud wool mane behind head/neck (`y=21..48, x>=40` + neck tuft `x=31..35, y=32..35`).
      - `tail_wool`: Fluffy white/silver tail wool + 4 red tail orbs (`x>=49, y=41..78`).
      - `arm_far`: Left extended yellow flipper arm (`x=17..30, y=45..58`).
      - `arm_near`: Right foreground yellow flipper arm (`x=39..52, y=48..60`).
      - `foot_L`: Left grounded foot (`x=21..32, y=74..80`).
      - `foot_R`: Right grounded foot (`x=42..50, y=76..84`).
      - `torso`: Striped neck (`y=34..41`), yellow chest, white/silver belly, and yellow thighs.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    r = arr96[:, :, 0].astype(np.int32)
    g = arr96[:, :, 1].astype(np.int32)
    b = arr96[:, :, 2].astype(np.int32)
    yy, xx = np.mgrid[0:h, 0:w]

    is_wht_fill = a & (((r == 248) & (g == 248)) | ((r == 191) & (g == 191)))
    is_red = a & (((r == 255) & (g == 73)) | ((r == 191) & (g == 55)) | ((r == 128) & (g == 37)))
    is_yel_fill = a & (((r == 255) & (g == 225)) | ((r == 229) & (g == 184)) | ((r == 179) & (g == 131)))

    lbl_wht, _ = ndi.label(is_wht_fill)
    lbl_yel, _ = ndi.label(is_yel_fill)

    def pick_labels(lbl_arr: np.ndarray, predicate) -> np.ndarray:
        out_m = np.zeros_like(a)
        for cid in range(1, int(lbl_arr.max()) + 1):
            cm = lbl_arr == cid
            ys, xs = np.where(cm)
            if len(ys) > 0 and predicate(int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()), int(len(ys))):
                out_m |= cm
        return out_m

    # 1. `mane_upper`: White/silver cloud mane behind head & neck (`x>=40, y=21..48`) + small left neck tuft (`x=31..35, y=32..35`)
    mane_wht = pick_labels(lbl_wht, lambda x0, x1, y0, y1, n: (n >= 100 and y0 >= 21 and y1 <= 48) or (31 <= x0 <= 35 and 31 <= y0 <= 35))
    mane_dil = ndi.binary_dilation(mane_wht, iterations=1)
    # Include the grey/black outer outline of `mane_upper`, but NOT the head/horn/neck outlines
    head_zone = (
        (yy <= 20)
        | ((yy == 21) & (xx <= 46))
        | ((yy == 22) & (xx <= 45))
        | ((yy == 23) & (xx <= 44))
        | ((yy == 24) & (xx <= 43))
        | ((yy >= 25) & (yy <= 27) & (xx <= 42))
        | ((yy >= 28) & (yy <= 29) & (xx <= 41))
        | ((yy >= 30) & (yy <= 31) & (xx <= 40))
        | ((yy >= 32) & (yy <= 33) & (xx >= 35) & (xx <= 39))
    )
    head_m = a & head_zone

    neck_torso_zone = (
        ((yy >= 34) & (yy <= 40) & (xx >= 33) & (xx <= 39))
        | ((yy >= 41) & (yy <= 45) & (xx <= 40))
        | ((yy == 46) & (xx <= 41))
        | ((yy == 47) & (xx <= 42))
        | ((yy >= 48) & (xx <= 48))
    )
    mane_m = (mane_wht | (mane_dil & a & is_dark_outline(arr96))) & (~head_m) & (~neck_torso_zone)
    # Also include the small trailing black lock tips of `mane_upper` at `y=37..39, x=60..64` and `y=42..48, x=43..65`
    mane_m |= a & (~head_m) & (~neck_torso_zone) & (yy >= 21) & (yy <= 48) & (xx <= 65)

    # 2. `tail_wool`: Fluffy cloud tail (`x>=49, y=41..78`) + 4 red tail orbs
    tail_wht = pick_labels(lbl_wht, lambda x0, x1, y0, y1, n: x0 >= 50 and y0 >= 41)
    tail_red = is_red & (yy >= 45)
    tail_seed = tail_wht | tail_red
    tail_dil = ndi.binary_dilation(tail_seed, iterations=2)
    right_thigh_zone = (xx <= 49) | ((xx == 50) & (yy >= 65) & (yy <= 73))
    tail_m = (tail_seed | (tail_dil & a & (xx >= 50))) & (~mane_m) & (~right_thigh_zone)

    # 3. `arm_far`: Left extended yellow flipper arm (`x=17..29, y=45..58`), leaving `torso`'s
    #    left belly outline (`G`/`S`/`K` at `x=26..27, y=47..57`) 100% intact on `torso`!
    arm_far_m = a & (
        ((yy == 45) & (xx <= 29))
        | ((yy == 46) & (xx <= 29))
        | ((yy == 47) & (xx <= 27))
        | ((yy == 48) & (xx <= 26))
        | ((yy == 49) & (xx <= 26))
        | ((yy == 50) & (xx <= 26))
        | ((yy >= 51) & (yy <= 56) & (xx <= 25))
        | ((yy == 57) & (xx <= 24))
        | ((yy == 58) & (xx <= 22))
    )

    # 4. `arm_near`: Right foreground yellow flipper arm (`x=39..52, y=48..60`)
    arm_near_m = a & (
        ((yy == 48) & (xx >= 41) & (xx <= 45))
        | ((yy == 49) & (xx >= 39) & (xx <= 46))
        | ((yy == 50) & (xx >= 39) & (xx <= 47))
        | ((yy == 51) & (xx >= 39) & (xx <= 48))
        | ((yy == 52) & (xx >= 39) & (xx <= 49))
        | ((yy == 53) & (xx >= 39) & (xx <= 50))
        | ((yy == 54) & (xx >= 40) & (xx <= 51))
        | ((yy == 55) & (xx >= 40) & (xx <= 52))
        | ((yy == 56) & (xx >= 41) & (xx <= 52))
        | ((yy == 57) & (xx >= 42) & (xx <= 52))
        | ((yy == 58) & (xx >= 43) & (xx <= 52))
        | ((yy == 59) & (xx >= 45) & (xx <= 52))
        | ((yy == 60) & (xx >= 48) & (xx <= 52))
    )

    # 5. `foot_L` and `foot_R`: Grounded feet below ankle narrowings (`y>=73` for L, `y>=75` for R)
    foot_L_m = a & (xx <= 32) & (yy >= 73)
    foot_R_m = a & (xx >= 41) & (xx <= 50) & (yy >= 75)

    # 6. `torso`: Remaining body pixels (striped neck, chest, white belly, left & right thighs)
    used = head_m | mane_m | tail_m | arm_far_m | arm_near_m | foot_L_m | foot_R_m
    torso_m = a & (~used)

    return {
        "mane_upper": mane_m,
        "tail_wool": tail_m,
        "arm_far": arm_far_m,
        "foot_L": foot_L_m,
        "foot_R": foot_R_m,
        "torso": torso_m,
        "arm_near": arm_near_m,
        "head": head_m,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` (facing right) into matching anatomical masks:
      - `mane_upper`: Giant billowing white/silver cloud wool mane on back of head/neck (`y=11..48, x=31..65`).
      - `head`: Right-facing yellow head/snout, forehead red orb tip, and horn tips (`x>=60, y=13..30` + left ear tip `x=48..51, y=14..16`).
      - `tail_wool`: Fluffy white/silver cloud tail + 4 red orbs sweeping left from lower back (`x=20..58, y=41..78`).
      - `arm_far`: Left yellow flipper arm (`x=44..52, y=48..56`).
      - `arm_near`: Right yellow flipper arm (`x=64..74, y=50..60`).
      - `foot_L`: Left hind foot (`x=44..52, y=74..80`).
      - `foot_R`: Right hind foot (`x=59..70, y=77..84`).
      - `torso`: Yellow back and thighs (`x=47..71, y=42..78`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    r = arr96[:, :, 0].astype(np.int32)
    g = arr96[:, :, 1].astype(np.int32)
    b = arr96[:, :, 2].astype(np.int32)
    yy, xx = np.mgrid[0:h, 0:w]

    is_wht_fill = a & (((r == 248) & (g == 248)) | ((r == 191) & (g == 191)))
    is_red = a & (((r == 255) & (g == 73)) | ((r == 191) & (g == 55)) | ((r == 128) & (g == 37)))
    lbl_wht, _ = ndi.label(is_wht_fill)

    def pick_labels(lbl_arr: np.ndarray, predicate) -> np.ndarray:
        out_m = np.zeros_like(a)
        for cid in range(1, int(lbl_arr.max()) + 1):
            cm = lbl_arr == cid
            ys, xs = np.where(cm)
            if len(ys) > 0 and predicate(int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()), int(len(ys))):
                out_m |= cm
        return out_m

    # 1. `head` on `back.png`: right-side horn, forehead red orb, white orb setting, and yellow cheek (`y=13..30, x>=60`) + left ear tip (`x=48..51, y=14..16`)
    head_right_zone = (
        ((yy == 13) & (xx >= 61))
        | ((yy == 14) & (xx >= 61))
        | ((yy == 15) & (xx >= 61))
        | ((yy == 16) & (xx >= 61))
        | ((yy == 17) & (xx >= 60))
        | ((yy == 18) & (xx >= 60))
        | ((yy == 19) & (xx >= 60))
        | ((yy == 20) & (xx >= 60))
        | ((yy == 21) & (xx >= 60))
        | ((yy == 22) & (xx >= 60))
        | ((yy == 23) & (xx >= 61))
        | ((yy == 24) & (xx >= 63))
        | ((yy == 25) & (xx >= 63))
        | ((yy == 26) & (xx >= 64))
        | ((yy == 27) & (xx >= 64))
        | ((yy == 28) & (xx >= 65))
        | ((yy == 29) & (xx >= 65))
        | ((yy == 30) & (xx >= 65))
    )
    head_left_ear = (yy >= 14) & (yy <= 16) & (xx >= 48) & (xx <= 51) & ((r == 16) & (g == 16) & (b == 16))
    head_m = a & (head_right_zone | head_left_ear)

    # 2. `mane_upper` on `back.png`: upper white/silver cloud mane (`y=11..48, x=31..65`)
    mane_wht = pick_labels(lbl_wht, lambda x0, x1, y0, y1, n: y0 <= 38 and x1 <= 65)
    mane_dil = ndi.binary_dilation(mane_wht, iterations=1)
    mane_m = (mane_wht | (mane_dil & a & is_dark_outline(arr96))) & (~head_m)
    # Include the trailing black lock tips of `mane_upper` at `y=37..46, x=30..35` and `y=45..48, x=41..55`
    mane_m |= a & (~head_m) & (yy >= 11) & (yy <= 48) & (xx >= 30) & (xx <= 64) & (
        (yy <= 41)
        | ((yy == 42) & (xx <= 60))
        | ((yy == 43) & (xx <= 59))
        | ((yy == 44) & (xx <= 58))
        | ((yy == 45) & (xx <= 57))
        | ((yy == 46) & (xx <= 57))
        | ((yy == 47) & (xx <= 55))
        | ((yy == 48) & (xx <= 53))
    )

    # 3. `tail_wool` on `back.png`: sweeping cloud tail + 4 red orbs (`x=20..58, y=41..78`)
    tail_wht = pick_labels(lbl_wht, lambda x0, x1, y0, y1, n: y0 >= 41)
    tail_red = is_red & (yy >= 45)
    tail_seed = tail_wht | tail_red
    tail_dil = ndi.binary_dilation(tail_seed, iterations=1)
    tail_m = (tail_seed | (tail_dil & a & is_dark_outline(arr96))) & (~mane_m)
    # Include outer black tips of tail wool at `x<=45, y>=41` and bottom tail seam at `y=73..74, x=43..58`
    tail_m |= a & (~mane_m) & (
        ((xx <= 43) & (yy >= 41))
        | ((yy == 73) & (xx <= 58))
        | ((yy == 74) & (xx <= 45))
        | ((yy == 74) & (xx >= 52) & (xx <= 58))
    )

    # 4. `arm_far`: Left yellow flipper arm (`x=44..52, y=48..56`)
    arm_far_m = a & (~mane_m) & (~tail_m) & (
        ((yy == 48) & (xx <= 52))
        | ((yy == 49) & (xx <= 51))
        | ((yy == 50) & (xx <= 51))
        | ((yy == 51) & (xx <= 50))
        | ((yy >= 52) & (yy <= 54) & (xx <= 50))
        | ((yy == 55) & (xx <= 49))
        | ((yy == 56) & (xx <= 48))
    )

    # 5. `arm_near`: Right yellow flipper arm (`x=64..74, y=50..60`)
    arm_near_m = a & (
        ((yy == 50) & (xx >= 66))
        | ((yy >= 51) & (yy <= 53) & (xx >= 65))
        | ((yy == 54) & (xx >= 64))
        | ((yy >= 55) & (yy <= 56) & (xx >= 65))
        | ((yy >= 57) & (yy <= 58) & (xx >= 66))
        | ((yy == 59) & (xx >= 67))
        | ((yy == 60) & (xx >= 69))
    )

    # 6. `foot_L` (`x=44..52, y=74..80`) and `foot_R` (`x=59..73, y=77..84` including right toe claw at x=71..72)
    foot_L_m = a & (~tail_m) & (xx >= 44) & (xx <= 52) & (yy >= 74)
    foot_R_m = a & (xx >= 59) & (xx <= 74) & (yy >= 77)

    # 7. `torso`: Remaining yellow back & thighs
    used = head_m | mane_m | tail_m | arm_far_m | arm_near_m | foot_L_m | foot_R_m
    torso_m = a & (~used)

    return {
        "mane_upper": mane_m,
        "tail_wool": tail_m,
        "arm_far": arm_far_m,
        "foot_L": foot_L_m,
        "foot_R": foot_R_m,
        "torso": torso_m,
        "arm_near": arm_near_m,
        "head": head_m,
    }


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
    and builds cushioned RGBA layers with view-accurate foreground masks.
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

    canvas = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
    canvas[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = src_96[y0:y1, x0:x1]

    c_masks: Dict[str, np.ndarray] = {}
    for k, m96 in masks_96.items():
        cm = np.zeros((cell_h, cell_w), dtype=bool)
        cm[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = m96[y0:y1, x0:x1]
        c_masks[k] = cm

    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Build natural right-torso contour behind `arm_near` so when `arm_near` swings forward/up,
        # `torso` never exposes an un-outlined lump protruding into `arm_near`'s silhouette!
        yy_c, xx_c = np.mgrid[0:cell_h, 0:cell_w]
        # Convert native x-coordinate helper to canvas coordinates
        def nat_x(nx: np.ndarray) -> np.ndarray:
            if not flip_x:
                return dst_x0 + (nx - x0)
            else:
                return dst_x0 + (sw - 1 - (nx - x0))

        # Native `y` for each canvas row `yy_c`
        nat_y = yy_c - dst_y0 + y0
        # Behind `arm_near` (native y=48..60), torso's natural right contour runs from native x=41 at y=48 to x=47 at y=60
        torso_under_arm_near = np.zeros((cell_h, cell_w), dtype=bool)
        torso_outline_under_arm = np.zeros((cell_h, cell_w), dtype=bool)
        for ny in range(48, 61):
            cy = ny - y0 + dst_y0
            if 0 <= cy < cell_h:
                lim_nx = 41 + (ny - 47) // 2
                for nx in range(38, lim_nx + 1):
                    cx = int(nat_x(nx))
                    if 0 <= cx < cell_w and c_masks["arm_near"][cy, cx]:
                        torso_under_arm_near[cy, cx] = True
                        if nx == lim_nx:
                            torso_outline_under_arm[cy, cx] = True

        # Shoulder attachment zone for `arm_far` (native y=46..55, with bottom outline at y=55)
        arm_far_shoulder_fg = c_masks["torso"] & (nat_y >= 46) & (nat_y <= 55)

        layers["mane_upper"] = build_part_layer(
            canvas, c_masks["mane_upper"], c_masks["head"] | c_masks["torso"], cushion_px=5
        )
        layers["tail_wool"] = build_part_layer(
            canvas, c_masks["tail_wool"], c_masks["torso"] & (nat_y >= 64) & (nat_y <= 73), cushion_px=4
        )
        layers["arm_far"] = build_part_layer(
            canvas, c_masks["arm_far"], arm_far_shoulder_fg, cushion_px=5
        )
        # Keep the top (native y=46) and bottom (native y=55) of `arm_far`'s shoulder cushion outlined
        arm_far_cush_only = (layers["arm_far"][:, :, 3] >= 128) & (~c_masks["arm_far"])
        layers["arm_far"][arm_far_cush_only & (nat_y == 46)] = [115, 75, 34, 255]
        layers["arm_far"][arm_far_cush_only & (nat_y >= 54)] = [16, 16, 16, 255]
        layers["torso"] = build_part_layer(
            canvas, c_masks["torso"], torso_under_arm_near | c_masks["head"], cushion_px=6
        )
        # Draw a crisp 1px brown/black contour along the newly exposed torso edge under `arm_near`
        layers["torso"][torso_outline_under_arm] = [115, 75, 34, 255]
        layers["arm_near"] = build_part_layer(
            canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0
        )
        layers["head"] = build_part_layer(
            canvas, c_masks["head"], np.zeros_like(c_masks["head"]), cushion_px=0
        )
    else:
        yy_c, xx_c = np.mgrid[0:cell_h, 0:cell_w]
        def nat_x_b(nx: int) -> int:
            if not flip_x:
                return int(dst_x0 + (nx - x0))
            else:
                return int(dst_x0 + (sw - 1 - (nx - x0)))

        nat_y = yy_c - dst_y0 + y0
        torso_under_arm_near = np.zeros((cell_h, cell_w), dtype=bool)
        torso_outline_under_arm = np.zeros((cell_h, cell_w), dtype=bool)
        for ny in range(50, 61):
            cy = ny - y0 + dst_y0
            if 0 <= cy < cell_h:
                lim_nx = 64 + (ny - 49) // 4
                for nx in range(60, lim_nx + 1):
                    cx = nat_x_b(nx)
                    if 0 <= cx < cell_w and c_masks["arm_near"][cy, cx]:
                        torso_under_arm_near[cy, cx] = True
                        if nx == lim_nx:
                            torso_outline_under_arm[cy, cx] = True

        arm_far_shoulder_fg = (c_masks["torso"] | c_masks["mane_upper"]) & (nat_y >= 48) & (nat_y <= 53)

        layers["arm_far"] = build_part_layer(
            canvas, c_masks["arm_far"], arm_far_shoulder_fg, cushion_px=3
        )
        layers["head"] = build_part_layer(
            canvas, c_masks["head"], c_masks["mane_upper"], cushion_px=4
        )
        layers["torso"] = build_part_layer(
            canvas,
            c_masks["torso"],
            torso_under_arm_near | c_masks["tail_wool"] | c_masks["mane_upper"],
            cushion_px=5,
        )
        layers["torso"][torso_outline_under_arm] = [115, 75, 34, 255]
        layers["arm_near"] = build_part_layer(
            canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0
        )
        layers["tail_wool"] = build_part_layer(
            canvas, c_masks["tail_wool"], c_masks["mane_upper"], cushion_px=3
        )
        layers["mane_upper"] = build_part_layer(
            canvas, c_masks["mane_upper"], np.zeros_like(c_masks["mane_upper"]), cushion_px=0
        )

    # Hind feet (`foot_L`, `foot_R`): 2px vertical upward ankle stem on interior columns (with dark outline on outer edges)
    stem_parent = c_masks["torso"] | (c_masks["tail_wool"] if is_back else np.zeros_like(c_masks["torso"]))
    for fk in ("foot_L", "foot_R"):
        fm = c_masks[fk]
        lyr = np.zeros_like(canvas)
        lyr[fm] = canvas[fm]
        f_ys, f_xs = np.where(fm)
        if len(f_ys) > 0:
            cy_min = int(f_ys.min())
            top_xs = np.sort(np.where(fm[cy_min, :])[0])
            if len(top_xs) >= 3:
                # Only extend the middle columns of the ankle upward by 2px so horizontal stride never pokes out
                inner_xs = top_xs[1:-1] if len(top_xs) >= 4 else top_xs
                for idx_c, x_col in enumerate(inner_xs):
                    top_px = canvas[cy_min, x_col]
                    for dy_up in range(1, 3):
                        py = cy_min - dy_up
                        if py >= 0 and stem_parent[py, x_col]:
                            if idx_c == 0 or idx_c == len(inner_xs) - 1:
                                lyr[py, x_col] = [16, 16, 16, 255]
                            else:
                                lyr[py, x_col] = top_px
        layers[fk] = lyr

    return layers


def wave_wool_layer(
    wool_layer: np.ndarray,
    base_dx: int,
    base_dy: int,
    facing: float,
    tip_shear_dx: int = 0,
    fluff_px: int = 0,
    is_tail: bool = False,
) -> np.ndarray:
    """
    Moves `mane_upper` or `tail_wool` by integer `(base_dx, base_dy)` and applies a gentle,
    coherent 1px cloud-tip wave (`tip_shear_dx`) or outward fluff expansion (`fluff_px`)
    while keeping thin diagonal lock tips (`y >= y_max - 5` on `mane_upper`) locked together
    so no 1px outline tip ever detaches.
    """
    shifted = shift_layer_int(wool_layer, base_dx, base_dy)
    if tip_shear_dx == 0 and fluff_px == 0:
        return shifted
    m = shifted[:, :, 3] >= 128
    if not np.any(m):
        return shifted
    h, w = shifted.shape[:2]
    ys, xs = np.where(m)
    y_min, y_max = int(ys.min()), int(ys.max())
    span_y = max(6, y_max - y_min + 1)
    cx = int(round(np.mean(xs)))

    out = np.zeros_like(shifted)
    for py in range(y_min, y_max + 1):
        if is_tail:
            # On `tail_wool`, the upper spire (`py <= y_min + span_y // 3`) sways as a coherent block
            row_dx = tip_shear_dx if (py - y_min) <= (span_y // 3) else 0
        else:
            # On `mane_upper`, the entire lower lock group (`py >= y_min + span_y // 2`) shifts as a single coherent block
            row_dx = tip_shear_dx if (py - y_min) >= (span_y // 2) else 0

        row_px = shifted[py]
        tmp_row = np.zeros_like(row_px)
        if row_dx == 0:
            tmp_row[:] = row_px
        elif row_dx > 0:
            tmp_row[row_dx:w] = row_px[0:w - row_dx]
        else:
            tmp_row[0:w + row_dx] = row_px[-row_dx:w]

        # Only apply `fluff_px` on thick interior rows (avoiding thin bottom lock tips where 1px diagonal outlines live)
        row_m = tmp_row[:, 3] >= 128
        r_xs = np.where(row_m)[0]
        if fluff_px > 0 and len(r_xs) >= 10 and (y_min + 3 <= py <= y_max - 6):
            out[py] = tmp_row
            if facing < 0:
                for px in r_xs:
                    if px >= cx + 2 and px + 1 < w:
                        out[py, px + 1] = tmp_row[px]
            else:
                for px in r_xs:
                    if px <= cx - 2 and px - 1 >= 0:
                        out[py, px - 1] = tmp_row[px]
        else:
            out[py] = tmp_row
    return out


def pulse_red_orbs(canvas: np.ndarray, pulse_level: int) -> np.ndarray:
    """
    Subtly gleams the red orbs in Mega Ampharos's tail and forehead using ONLY existing
    in-palette colors (`(255,73,13)` bright red-orange and `(248,248,248)` white highlight)
    during peak breath (`pulse_level=1`) or electric discharge (`pulse_level=2`).
    """
    if pulse_level <= 0:
        return canvas
    out = canvas.copy()
    a = out[:, :, 3] >= 128
    r = out[:, :, 0]
    g = out[:, :, 1]
    b = out[:, :, 2]

    # Mid red `(191, 55, 10)` next to bright red `(255, 73, 13)` gleams to `(255, 73, 13)`
    is_bright_red = a & (r == 255) & (g == 73) & (b == 13)
    is_mid_red = a & (r == 191) & (g == 55) & (b == 10)
    if np.any(is_bright_red) and np.any(is_mid_red):
        dil = ndi.binary_dilation(is_bright_red, iterations=1)
        gleam_m = is_mid_red & dil
        out[gleam_m, 0] = 255
        out[gleam_m, 1] = 73
        out[gleam_m, 2] = 13

    if pulse_level >= 2 and np.any(is_bright_red):
        # At peak attack discharge, 1 central pixel of each bright red cluster gleams to palette white `(248, 248, 248)`
        lbl, n_c = ndi.label(is_bright_red)
        for cid in range(1, n_c + 1):
            ys, xs = np.where(lbl == cid)
            if len(ys) >= 4:
                out[int(ys[0]), int(xs[0]), :3] = (248, 248, 248)
    return out


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
        (mid_x - 11 - drift, max(2, top_y + 2 - drift // 2)),
        (mid_x + 11 + drift, max(2, top_y + 3 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (248, 248, 248, 255)
    return out


def synthesize_ampharos_mega_frame(
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
    arm_near_dx, arm_near_dy = 0, 0
    arm_far_dx, arm_far_dy = 0, 0
    mane_dx, mane_dy, mane_shear, mane_fluff = 0, 0, 0, 0
    tail_dx, tail_dy, tail_shear, tail_fluff = 0, 0, 0, 0
    foot_L_dx, foot_L_dy = 0, 0
    foot_R_dx, foot_R_dy = 0, 0
    orb_pulse = 0

    if action == "idle":
        # Modeled directly after Normal Ampharos's GIF (`0181_ampharos/normal/front.gif`):
        # 1. Both feet (`foot_L`, `foot_R`) stay 100% grounded at `(0, 0)`.
        # 2. Torso & Head breathe rhythmically (Head is 100% rigid integer shift).
        # 3. Flipper arms (`arm_near`, `arm_far`) swing gently with the breath.
        # 4. Fluffy wool mane (`mane_upper`) and orb-studded tail (`tail_wool`) float & wave
        #    with phase delay, and the red orbs gleam softly at peak breath!
        torso_dy = int(round(1.0 * sin1))
        head_dy = int(round(1.5 * math.sin(phase - 0.25)))
        head_dx = int(round(0.8 * max(0.0, math.sin(phase - 0.25)) * facing))

        arm_near_dx = int(round(1.2 * math.sin(phase - 0.45) * facing))
        arm_near_dy = torso_dy + int(round(1.4 * math.sin(phase - 0.40)))
        arm_far_dx = int(round(1.2 * math.sin(phase - 0.70) * facing))
        arm_far_dy = torso_dy + int(round(1.4 * math.sin(phase - 0.65)))

        mane_dx = head_dx + int(round(-0.8 * math.sin(phase - 0.65) * facing))
        mane_dy = int(round(1.8 * math.sin(phase - 0.65)))
        mane_shear = int(round(-1.0 * math.sin(phase - 0.95) * facing))
        mane_fluff = 1 if math.sin(phase - 0.55) > 0.50 else 0

        tail_dx = int(round(-0.8 * math.sin(phase - 1.05) * facing))
        tail_dy = int(round(1.6 * math.sin(phase - 1.05)))
        tail_shear = int(round(1.0 * math.sin(phase - 1.35) * facing))
        tail_fluff = 1 if math.sin(phase - 0.95) > 0.50 else 0

        orb_pulse = 1 if math.sin(phase - 0.35) > 0.45 else 0

    elif action == "walk":
        # Bipedal alternating step (`foot_L` vs `foot_R`, lift=2px, stride=±2px),
        # 2-beat body bounce, opposite-phase arm swing, and rich phase-delayed wool billowing!
        bounce = int(round(-1.8 * abs(sin1)))
        sway_x = int(round(1.0 * sin1 * facing))
        torso_dx, torso_dy = sway_x, bounce

        head_dx = sway_x + int(round(1.0 * cos1 * facing))
        head_dy = bounce + int(round(1.2 * math.sin(2.0 * phase - 0.3)))

        # Arms swing in opposite phase to the stepping feet
        arm_near_dx = sway_x + int(round(2.0 * sin1 * facing))
        arm_near_dy = bounce + int(round(1.4 * cos1))
        arm_far_dx = sway_x + int(round(-2.0 * sin1 * facing))
        arm_far_dy = bounce + int(round(-1.4 * cos1))

        mane_dx = head_dx + int(round(-1.0 * math.sin(2.0 * phase - 0.6) * facing))
        mane_dy = bounce + int(round(1.6 * math.sin(2.0 * phase - 0.6)))
        mane_shear = int(round(-1.0 * math.sin(2.0 * phase - 1.0) * facing))
        mane_fluff = 1 if math.sin(2.0 * phase - 0.6) > 0.45 else 0

        tail_dx = sway_x + int(round(-1.0 * math.sin(2.0 * phase - 1.0) * facing))
        tail_dy = bounce + int(round(1.8 * math.sin(2.0 * phase - 1.0)))
        tail_shear = int(round(1.0 * math.sin(2.0 * phase - 1.4) * facing))
        tail_fluff = 1 if math.sin(2.0 * phase - 1.0) > 0.45 else 0

        foot_L_dx, foot_L_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        foot_R_dx, foot_R_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)
        orb_pulse = 1 if math.sin(2.0 * phase) > 0.40 else 0

    elif action == "attack":
        # Signature Mega Ampharos "Dragon Pulse / Thunderbolt" Discharge Lunge!
        # Frames 0..3: Charge wind-up (pull back, crouch +1px, arms & wool draw back, orbs charge)
        # Frames 4..9: Forward discharge thrust (front foot steps forward +2px, torso & head lunge
        #              forward +4..5px, both flipper arms thrust forward +5..6px, wool mane & tail
        #              billow/expand outward, and red orbs gleam at full intensity!)
        # Frames 10..15: Smooth follow-through recovery to idle stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.65 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        torso_dx = int(round(3.5 * k * facing))
        torso_dy = int(round(-1.0 * k)) if k > 0 else int(round(1.0 * abs(k)))

        head_dx = int(round(4.8 * k * facing))
        head_dy = int(round(1.2 * k)) if k > 0 else int(round(-1.0 * abs(k)))

        arm_near_dx = int(round(5.8 * k * facing))
        arm_near_dy = int(round(-1.5 * k))
        arm_far_dx = int(round(5.5 * k * facing))
        arm_far_dy = int(round(-1.2 * k))

        # Wool mane & tail billow dramatically during discharge (frames 4..9)
        mane_dx = head_dx + int(round(-1.5 * k * facing))
        mane_dy = head_dy + int(round(-2.0 * k))
        mane_shear = int(round(-1.0 * k * facing))
        mane_fluff = 1 if k > 0.30 else 0

        tail_dx = torso_dx + int(round(-1.5 * k * facing))
        tail_dy = int(round(-2.2 * k))
        tail_shear = int(round(1.0 * k * facing))
        tail_fluff = 1 if k > 0.30 else 0

        # Front foot steps into the lunge while rear foot stays planted on the ground
        step_x = int(round(2.2 * max(0.0, k) * facing))
        lift_y = -1 if frame_idx in (4, 5) else 0
        if facing < 0:
            foot_L_dx, foot_L_dy = step_x, lift_y
        else:
            foot_R_dx, foot_R_dy = step_x, lift_y

        orb_pulse = 2 if (4 <= frame_idx <= 8) else (1 if frame_idx in (2, 3, 9) else 0)

    elif action == "hurt":
        # Knockback + hit-shake + ruffled wool + hit sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        torso_dx = int(round(-3.5 * env * facing)) + jit
        torso_dy = int(round(-1.0 * env))

        head_dx = torso_dx + int(round(-1.0 * env * facing))
        head_dy = int(round(-1.8 * env))

        arm_near_dx = torso_dx + int(round(-1.5 * env * facing))
        arm_near_dy = int(round(-2.0 * env))
        arm_far_dx = torso_dx + int(round(-1.2 * env * facing))
        arm_far_dy = int(round(-1.5 * env))

        mane_dx = head_dx + int(round(-1.0 * env * facing))
        mane_dy = head_dy + int(round(-1.0 * env))
        mane_shear = int(round(-1.0 * env * facing))
        mane_fluff = 1 if 2 <= frame_idx <= 6 else 0

        tail_dx = torso_dx
        tail_dy = int(round(1.2 * env))
        tail_shear = int(round(-1.0 * env * facing))
        tail_fluff = 1 if 2 <= frame_idx <= 6 else 0

        slide_x = int(round(torso_dx * 0.6))
        foot_L_dx, foot_R_dx = slide_x, slide_x

    mane_layer = shift_layer_int(layers["mane_upper"], mane_dx, mane_dy)
    tail_layer = shift_layer_int(layers["tail_wool"], tail_dx, tail_dy)

    if not is_back:
        comp = [
            mane_layer,
            tail_layer,
            shift_layer_int(layers["arm_far"], arm_far_dx, arm_far_dy),
            shift_layer_int(layers["foot_L"], foot_L_dx, foot_L_dy),
            shift_layer_int(layers["foot_R"], foot_R_dx, foot_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["arm_near"], arm_near_dx, arm_near_dy),
            shift_layer_int(layers["head"], head_dx, head_dy),
        ]
    else:
        comp = [
            shift_layer_int(layers["arm_far"], arm_far_dx, arm_far_dy),
            shift_layer_int(layers["foot_L"], foot_L_dx, foot_L_dy),
            shift_layer_int(layers["foot_R"], foot_R_dx, foot_R_dy),
            shift_layer_int(layers["head"], head_dx, head_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["arm_near"], arm_near_dx, arm_near_dy),
            tail_layer,
            mane_layer,
        ]

    out = composite_layers(canvas_shape, comp)
    out = pulse_red_orbs(out, orb_pulse)
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

    # Tight bounds of front (61x74) and back (56x74)
    max_w, max_h = 61, 74
    cell_w = max_w + 21  # 82 (even)
    cell_h = max_h + 16  # 90 (even)
    foot_y = cell_h - 4  # 86

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
                frame_rgba = synthesize_ampharos_mega_frame(
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
        "generator": "forge_ampharos_mega_actions",
        "key": "0181_ampharos-mega",
        "dex": 181,
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
    print(f"Built 0181_ampharos-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
