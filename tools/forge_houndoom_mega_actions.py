#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Houndoom (`0229_houndoom-mega`) modeled directly after Normal Houndoom's
official GIF motion (`0229_houndoom/normal/front.gif` & `back.gif`):
  - Exact color & connected-component anatomical layer separation (10 independent rig layers):
      1. `tail` (devil's bifurcated whip tail anchored seamlessly to the rear spine)
      2. `leg_0`, `leg_1`, `leg_2`, `leg_3` (the 4 distinct quadruped legs with white bone ankle
                                             rings and red claws — 100% grounded `(0, 0)` in `idle`,
                                             diagonal quadruped trot in `walk`, front-paw lunge in `attack`)
      3. `torso` (dark charcoal body, orange underbelly, and 4 dorsal spine bone rings)
      4. `head_upper` (red-orange snout, upper fang, fierce red eye, cranium, and upper neck —
                       100% rigid integer translation so the face and eyes never deform by a single pixel)
      5. `bone_collar` (massive white skull & ribcage bone armor surrounding the neck and chest)
      6. `lower_jaw` (articulated orange lower jaw & fang that opens `+1px` during breathing, roar, and bite)
      7. `horns` (massive sweeping curved white bone horns atop the skull)
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mouth_open  # noqa: E402

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen2/0229_houndoom/mega/houndoom-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16
VERSION = "houndoom_mega_v1m"

# Closed mouth line on native front.png (facing left), ordered hinge -> lip tip (2026-11-04 mouth fix).
# The mouth opens as a small lens carved UPWARD from this slit into the upper snout (the lower jaw is only
# one pixel row thick, so carving downward would erase it). The lower fang stays visible in the gap.
FRONT_MOUTH_LINE_96 = [(24, 39), (15, 39)]
MOUTH_NORMAL = (0.0, -1.0)
MOUTH_INNER = (98, 41, 16)  # existing dark red-brown palette colour (palate) for the 2px roar gap


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
        | ((r == 24) & (g == 32) & (b == 41))
        | ((r == 131) & (g == 115) & (b == 139))
    )


def is_charcoal_body(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 49) & (g == 49) & (b == 57))
        | ((r == 74) & (g == 74) & (b == 82))
    )


def build_part_layer(
    src: np.ndarray,
    mask: np.ndarray,
    foreground_mask: np.ndarray,
    cushion_px: int = 4,
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
    Segments native 96x96 `front.png` (facing left) into 10 exact anatomical masks:
      - `horns`: Giant sweeping curved white horns (`y=6..30, x=15..34`).
      - `lower_jaw`: Articulated orange lower jaw & white fang (`y=39..41, x=16..26`).
      - `head_upper`: Red-orange snout, fierce red eye, cranium, and upper neck (`y=26..48, x=14..34`).
      - `bone_collar`: Massive white skull & ribcage bone armor collar (`y=31..65, x=15..48`).
      - `torso`: Charcoal body, orange chest/belly, and 4 dorsal spine bone rings (`y=45..66, x=29..61`).
      - `tail`: Devil's bifurcated whip tail (`y=26..54, x=61..80`).
      - `leg_0`: Front-left leg with white bone ankle rings & red claws (`y=65..85, x=14..30`).
      - `leg_1`: Front-right leg with white bone ankle rings & red claws (`y=65..88, x=35..47`).
      - `leg_2`: Rear-left inner leg (`y=65..83, x=45..54`).
      - `leg_3`: Rear-right outer leg with red claws (`y=65..85, x=52..65`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    r, g, b = arr96[:, :, 0], arr96[:, :, 1], arr96[:, :, 2]
    is_orange = ((r == 246) & (g == 156)) | ((r == 164) & (g == 90)) | ((r == 98) & (g == 41))

    # 1. Horns
    horns_m = a & (
        (yy <= 25)
        | ((yy == 26) & (((xx >= 18) & (xx <= 22)) | ((xx >= 26) & (xx <= 32))))
        | ((yy == 27) & (((xx >= 19) & (xx <= 20)) | ((xx >= 26) & (xx <= 32))))
        | ((yy == 28) & (xx >= 26) & (xx <= 31))
        | ((yy == 29) & (xx >= 27) & (xx <= 31))
        | ((yy == 30) & (xx >= 28) & (xx <= 30))
    )

    # 2. Lower Jaw (orange lower jaw + lower white fang at x=19, y=39)
    lower_jaw_m = a & (
        ((yy == 39) & (xx >= 19) & (xx <= 24))
        | ((yy == 40) & (xx >= 16) & (xx <= 23))
        | ((yy == 41) & (xx >= 17) & (xx <= 19))
    )

    # 3. Head Upper (snout, red eye, cranium, and neck column down to y=48)
    head_upper_m = a & (~horns_m) & (~lower_jaw_m) & (
        ((yy >= 26) & (yy <= 40) & (xx <= 34))
        | ((yy == 41) & (xx >= 23) & (xx <= 34))
        | ((yy == 42) & (xx >= 24) & (xx <= 33))
        | ((yy == 43) & (xx >= 24) & (xx <= 33))
        | ((yy == 44) & (xx >= 25) & (xx <= 33))
        | ((yy == 45) & (xx >= 25) & (xx <= 32))
        | ((yy == 46) & (xx >= 25) & (xx <= 31))
        | ((yy == 47) & (xx >= 25) & (xx <= 30))
        | ((yy == 48) & (xx >= 26) & (xx <= 27))
    )

    # 4. Tail (devil's bifurcated whip tail)
    tail_m = a & (
        ((yy <= 45) & (xx >= 58))
        | ((yy >= 46) & (yy <= 54) & (xx >= 61))
    )

    # 5. Bone Collar (massive white skull & ribcage armor collar)
    bone_collar_m = a & (~horns_m) & (~head_upper_m) & (~lower_jaw_m) & (~is_orange) & (
        ((yy >= 31) & (yy <= 44) & (xx <= 49))
        | ((yy == 45) & (xx <= 44))
        | ((yy >= 46) & (yy <= 59) & (xx <= 44))
        | ((yy == 60) & (xx <= 42))
        | ((yy == 61) & (xx <= 39))
        | ((yy == 62) & ((xx <= 29) | ((xx >= 32) & (xx <= 39))))
        | ((yy == 63) & (xx <= 29))
        | ((yy == 64) & (xx <= 28))
        | ((yy == 65) & (xx >= 25) & (xx <= 28))
    )

    # 6. Four Legs (`leg_0`, `leg_1`, `leg_2`, `leg_3`) and `torso`
    rem = a & (~horns_m) & (~head_upper_m) & (~lower_jaw_m) & (~tail_m) & (~bone_collar_m)
    leg_0_m = rem & (yy >= 65) & (xx <= 30) & (~is_orange)
    leg_1_m = rem & (
        ((yy >= 65) & (yy <= 67) & (xx >= 35) & (xx <= 42))
        | ((yy >= 68) & (yy <= 69) & (xx >= 35) & (xx <= 43))
        | ((yy == 70) & (xx >= 36) & (xx <= 44))
        | ((yy == 71) & (xx >= 36) & (xx <= 45))
        | ((yy >= 72) & (yy <= 76) & (xx >= 36) & (xx <= 46))
        | ((yy == 77) & (xx >= 37) & (xx <= 46))
        | ((yy >= 78) & (yy <= 80) & (xx >= 37) & (xx <= 47))
        | ((yy == 81) & (xx >= 37) & (xx <= 46))
        | ((yy == 82) & (xx >= 38) & (xx <= 45))
        | ((yy >= 83) & (yy <= 88) & (xx >= 35) & (xx <= 46))
    )
    leg_2_m = rem & (~leg_1_m) & (
        ((yy == 65) & (xx >= 45) & (xx <= 50))
        | ((yy == 66) & (xx >= 45) & (xx <= 51))
        | ((yy == 67) & (xx >= 45) & (xx <= 52))
        | ((yy >= 68) & (yy <= 69) & (xx >= 46) & (xx <= 53))
        | ((yy >= 70) & (yy <= 73) & (xx >= 47) & (xx <= 54))
        | ((yy >= 74) & (yy <= 83) & (xx >= 46) & (xx <= 53))
    )
    leg_3_m = rem & (~leg_1_m) & (~leg_2_m) & (yy >= 65) & (xx >= 52)
    torso_m = rem & (~leg_0_m) & (~leg_1_m) & (~leg_2_m) & (~leg_3_m)

    return {
        "tail": tail_m,
        "leg_0": leg_0_m,
        "leg_2": leg_2_m,
        "torso": torso_m,
        "leg_1": leg_1_m,
        "leg_3": leg_3_m,
        "head_upper": head_upper_m,
        "bone_collar": bone_collar_m,
        "lower_jaw": lower_jaw_m,
        "horns": horns_m,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` (facing right) into matching 10 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    r, g, b = arr96[:, :, 0], arr96[:, :, 1], arr96[:, :, 2]
    is_orange = ((r == 246) & (g == 156)) | ((r == 164) & (g == 90)) | ((r == 98) & (g == 41))
    is_bone = (
        ((r == 255) & (g == 255) & (b == 255))
        | ((r == 205) & (g == 213) & (b == 213))
        | ((r == 164) & (g == 156) & (b == 172))
        | ((r == 131) & (g == 115) & (b == 139))
    )

    # 1. Horns
    horns_m = a & (
        (yy <= 24)
        | ((yy == 25) & ((xx <= 60) | (xx >= 66)))
        | ((yy == 26) & ((xx <= 60) | (xx >= 65)))
        | ((yy == 27) & (xx >= 65))
        | ((yy == 28) & (((xx >= 58) & (xx <= 59)) | (xx >= 65)))
        | ((yy == 29) & (xx >= 65) & (xx <= 71))
        | ((yy == 30) & (xx >= 65) & (xx <= 69))
        | ((yy == 31) & (xx >= 67) & (xx <= 68))
    )

    # 2. Lower Jaw (y=37..39, x=71..76)
    lower_jaw_m = a & (
        ((yy == 37) & (xx >= 71) & (xx <= 76))
        | ((yy == 38) & (xx >= 71) & (xx <= 75))
        | ((yy == 39) & (xx >= 71) & (xx <= 74))
    )

    # 3. Head Upper & Bone Collar
    # Note: on `back.png`, the dark neck column at `x=54..58, y=34..40` belongs to `head_upper`,
    # while the two dorsal bone spikes (`x=43..54` and `x=58..70`) belong to `bone_collar`!
    right_collar_spike = (
        ((yy == 29) & (xx >= 58) & (xx <= 60))
        | ((yy == 30) & (xx >= 58) & (xx <= 61))
        | ((yy == 31) & (xx >= 58) & (xx <= 62))
        | ((yy == 32) & (xx >= 58) & (xx <= 63))
        | ((yy == 33) & (xx >= 58) & (xx <= 63))
        | ((yy == 34) & (xx >= 58) & (xx <= 64))
        | ((yy == 35) & (xx >= 58) & (xx <= 64))
        | ((yy == 36) & (xx >= 58) & (xx <= 65))
        | ((yy == 37) & (xx >= 58) & (xx <= 70))
        | ((yy >= 38) & (yy <= 40) & (xx >= 59) & (xx <= 70))
    )
    head_upper_m = a & (~horns_m) & (~lower_jaw_m) & (~right_collar_spike) & (
        ((yy >= 25) & (yy <= 40) & (xx >= 55))
    )

    bone_collar_m = a & (~horns_m) & (~head_upper_m) & (~lower_jaw_m) & (
        ((yy >= 29) & (yy <= 43) & (xx >= 42))
        | ((yy == 44) & (xx >= 44))
        | ((yy == 45) & (xx >= 50))
        | ((yy == 46) & (xx >= 52))
        | ((yy == 47) & (xx >= 53))
        | ((yy == 48) & (xx >= 55))
        | ((yy == 49) & (xx >= 56))
        | ((yy == 50) & (xx >= 57))
        | ((yy == 51) & (xx >= 58))
        | ((yy == 52) & (xx >= 58))
        | ((yy == 53) & (xx >= 59))
        | ((yy == 54) & (xx >= 59))
        | ((yy >= 55) & (yy <= 60) & (xx >= 59) & (~is_orange))
    )

    # 4. Tail
    tail_m = a & (yy <= 54) & (xx <= 29)

    # 5. Four Legs and Torso
    rem = a & (~horns_m) & (~head_upper_m) & (~lower_jaw_m) & (~bone_collar_m) & (~tail_m)
    leg_0_m = rem & (yy >= 64) & (xx <= 36)
    leg_1_m = rem & (
        ((yy >= 64) & (yy <= 65) & (xx >= 37) & (xx <= 48))
        | ((yy >= 66) & (yy <= 67) & (xx >= 37) & (xx <= 47))
        | ((yy == 68) & (xx >= 37) & (xx <= 46))
        | ((yy >= 69) & (yy <= 80) & (xx >= 37) & (xx <= 45))
        | ((yy >= 81) & (yy <= 87) & (xx >= 37) & (xx <= 46))
    )
    leg_2_m = rem & (~leg_1_m) & (yy >= 64) & (xx >= 46) & (xx <= 56)
    leg_3_m = rem & (yy >= 64) & (xx >= 59)
    torso_m = rem & (~leg_0_m) & (~leg_1_m) & (~leg_2_m) & (~leg_3_m)

    return {
        "tail": tail_m,
        "leg_0": leg_0_m,
        "leg_2": leg_2_m,
        "torso": torso_m,
        "leg_1": leg_1_m,
        "leg_3": leg_3_m,
        "head_upper": head_upper_m,
        "bone_collar": bone_collar_m,
        "lower_jaw": lower_jaw_m,
        "horns": horns_m,
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
    and builds cushioned RGBA layers in strict back-to-front draw order.
    """
    raw_masks = segment_back_native(base_96) if is_back else segment_front_native(base_96)
    # Prepare clean native layers before flipping so exact pixel coordinates on 96x96 can be sealed cleanly
    src_96 = base_96.copy()
    charcoal_96 = is_charcoal_body(src_96)
    yy96, xx96 = np.mgrid[0:96, 0:96]

    layers_96: Dict[str, np.ndarray] = {}

    # 1. `tail`: cushion 3px into rear spine of `torso`
    tail_torso_er = ndi.binary_erosion(raw_masks["tail"] | raw_masks["torso"], iterations=1)
    layers_96["tail"] = build_part_layer(
        src_96, raw_masks["tail"], tail_torso_er & raw_masks["torso"], cushion_px=3
    )

    # 2. `leg_0, leg_1, leg_2, leg_3`:
    # Purely vertical upward stem cushion (4px) into `torso`/`bone_collar` so stepping never opens a hip gap,
    # and NO horizontal protrusion into adjacent legs!
    leg_cut_nat_y = 64 if is_back else 65
    upper_body_96 = raw_masks["torso"] | raw_masks["bone_collar"]
    for lk in ("leg_0", "leg_1", "leg_2", "leg_3"):
        lm = raw_masks[lk]
        lyr = np.zeros_like(src_96)
        lyr[lm] = src_96[lm]
        top_xs = np.where(lm[leg_cut_nat_y, :])[0]
        for x_col in top_xs:
            fill_px = np.array([49, 49, 57, 255], dtype=np.uint8)
            for dy_s in range(0, 4):
                if leg_cut_nat_y + dy_s < 96 and charcoal_96[leg_cut_nat_y + dy_s, x_col]:
                    fill_px = src_96[leg_cut_nat_y + dy_s, x_col]
                    break
            for dy_up in range(1, 5):
                py = leg_cut_nat_y - dy_up
                if py >= 0 and upper_body_96[py, x_col]:
                    lyr[py, x_col] = fill_px
        if not is_back and lk == "leg_2":
            # Seal the 2px left outline of `leg_2` behind `leg_1`'s lower ankle ring (y=79..80, x=47)
            lyr[79, 47] = np.array([1, 1, 1, 255], dtype=np.uint8)
            lyr[80, 47] = np.array([1, 1, 1, 255], dtype=np.uint8)
        elif is_back and lk == "leg_2":
            # Seal the 2px left outline of `leg_2` behind `leg_1` (y=64..65, x=48)
            lyr[64, 48] = np.array([24, 32, 41, 255], dtype=np.uint8)
            lyr[65, 48] = np.array([24, 32, 41, 255], dtype=np.uint8)
        layers_96[lk] = lyr

    # 3. `torso`:
    # Cushion `torso` inside the eroded interior of `bone_collar` & `head_upper`
    torso_fg_union = raw_masks["torso"] | raw_masks["bone_collar"] | raw_masks["head_upper"]
    torso_fg_eroded = ndi.binary_erosion(torso_fg_union, iterations=1)
    fg_over_torso = torso_fg_eroded & (raw_masks["bone_collar"] | raw_masks["head_upper"]) & (yy96 >= 41)
    layers_96["torso"] = build_part_layer(
        src_96, raw_masks["torso"], fg_over_torso, cushion_px=5, allowed_fill_mask=charcoal_96
    )

    # 4. `head_upper`:
    # Cushion neck base inside `bone_collar` (charcoal fill) so differential head/collar motion never opens a hole!
    head_collar_union = ndi.binary_erosion(raw_masks["head_upper"] | raw_masks["bone_collar"], iterations=1)
    fg_collar_over_head = head_collar_union & raw_masks["bone_collar"] & (yy96 >= 29)
    head_lyr = build_part_layer(
        src_96, raw_masks["head_upper"], fg_collar_over_head, cushion_px=6, allowed_fill_mask=charcoal_96
    )
    # (2026-11-04) The rectangular dark mouth-interior pocket behind `lower_jaw` was removed: the jaw now
    # moves rigidly with the head and the mouth opens with `mouth_open.open_mouth` along the mouth line.
    layers_96["head_upper"] = head_lyr

    # 5. `bone_collar`, `lower_jaw`, `horns`:
    for k in ("bone_collar", "lower_jaw", "horns"):
        lyr = np.zeros_like(src_96)
        lyr[raw_masks[k]] = src_96[raw_masks[k]]
        layers_96[k] = lyr

    # Apply horizontal flip if requested and place onto target `(cell_h, cell_w)` canvas
    a96 = src_96[:, :, 3] >= 128
    ys, xs = np.where(a96)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x0 = (cell_w - sw) // 2
    dst_y0 = foot_y - sh

    layers: Dict[str, np.ndarray] = {}
    for k, lyr96 in layers_96.items():
        cropped = lyr96[y0:y1, x0:x1].copy()
        if flip_x:
            cropped = cropped[:, ::-1, :].copy()
        canvas_lyr = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
        canvas_lyr[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = cropped
        layers[k] = canvas_lyr

    return layers


def mouth_line_cell(base_96: np.ndarray, is_back: bool, flip_x: bool, cell_w: int, foot_y: int):
    """Mouth line in cell coordinates for this direction (None for back views: mouth not visible)."""
    if is_back:
        return None
    a96 = base_96[:, :, 3] >= 128
    ys, xs = np.where(a96)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x0 = (cell_w - sw) // 2
    dst_y0 = foot_y - sh
    local = [(x - x0, y - y0) for x, y in FRONT_MOUTH_LINE_96]
    pts = mouth_open.transform_points(local, mirror_w=sw if flip_x else None)
    return mouth_open.transform_points(pts, dx=dst_x0, dy=dst_y0)


def wave_tail_layer(
    tail_layer: np.ndarray,
    base_dx: int,
    base_dy: int,
    phase: float,
    facing: float,
    amp_y: float = 2.0,
) -> np.ndarray:
    """
    Waves Mega Houndoom's devil whip tail smoothly from its spine root (`u=0`, locked to `base_dx, base_dy`)
    to its bifurcated tip (`u=1`) using column-coherent integer vertical shear so the thin black tail never splits!
    """
    shifted = shift_layer_int(tail_layer, base_dx, base_dy)
    m = shifted[:, :, 3] >= 128
    if not np.any(m):
        return shifted
    h, w = shifted.shape[:2]
    ys, xs = np.where(m)
    x_min, x_max = int(xs.min()), int(xs.max())
    span_x = max(6, x_max - x_min + 1)

    # When facing left (-1.0), the tail extends to the right (`x_min` is root, `x_max` is tip).
    # When facing right (+1.0), the tail extends to the left (`x_max` is root, `x_min` is tip).
    out = np.zeros_like(shifted)
    for px in range(x_min, x_max + 1):
        if facing < 0:
            u_tip = min(1.0, max(0.0, (px - x_min) / float(span_x)))
        else:
            u_tip = min(1.0, max(0.0, (x_max - px) / float(span_x)))
        # Keep the first 20% at the root completely locked (`dy = 0`) so it never detaches from the spine
        u_eff = max(0.0, (u_tip - 0.15) / 0.85)
        dy = int(round(u_eff * amp_y * math.sin(phase - u_eff * 1.1)))
        col_px = shifted[:, px]
        if dy == 0:
            out[:, px] = col_px
        elif dy > 0:
            out[dy:h, px] = col_px[0:h - dy]
        else:
            out[0:h + dy, px] = col_px[-dy:h]
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
        (mid_x - 14 - drift, max(2, top_y + 22 - drift // 2)),
        (mid_x + 14 + drift, max(2, top_y + 23 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (252, 252, 252, 255)
    return out


def synthesize_houndoom_mega_frame(
    layers: Dict[str, np.ndarray],
    canvas_shape: Tuple[int, int, int],
    action: str,
    frame_idx: int,
    facing: float,
    is_back: bool,
    palette_rgba: np.ndarray,
    mouth_line=None,
) -> np.ndarray:
    t = frame_idx / float(N_FRAMES)
    phase = 2.0 * math.pi * t
    sin1 = math.sin(phase)
    cos1 = math.cos(phase)

    torso_dx, torso_dy = 0, 0
    collar_dx, collar_dy = 0, 0
    head_dx, head_dy = 0, 0
    horns_dx, horns_dy = 0, 0
    jaw_open = 0
    leg_offsets = [(0, 0), (0, 0), (0, 0), (0, 0)]
    tail_amp_y, tail_freq = 2.0, 1.0

    if action == "idle":
        # Modeled directly after Normal Houndoom's official GIF (`0229_houndoom/normal/front.gif`):
        # 1. All 4 paws (`leg_0..3`) stay 100% planted on the ground `(0, 0)`.
        # 2. Ferocious hellhound breathing: `torso` rises/sinks (`-1..+1px`), massive `bone_collar`
        #    heaves (`-1..+2px`) and thrusts slightly forward (`+1px`), `head_upper` & `horns`
        #    menacingly nod up/down (`-1..+2px`) and forward (`+1px`).
        # 3. `lower_jaw` opens (`+1px`) on inhalation/growl frames (`frame_idx in (2,3,4,5, 10,11,12)`),
        #    and the bifurcated devil `tail` whips smoothly!
        torso_dy = int(round(1.0 * sin1))
        collar_dy = int(round(1.6 * math.sin(phase - 0.20)))
        collar_dx = int(round(0.8 * max(0.0, math.sin(phase - 0.20)) * facing))

        head_dy = int(round(1.8 * math.sin(phase - 0.35)))
        head_dx = int(round(1.2 * max(0.0, math.sin(phase - 0.35)) * facing))
        horns_dx, horns_dy = head_dx, head_dy

        jaw_open = 1 if frame_idx in (2, 3, 4, 5, 11, 12, 13) else 0
        tail_amp_y, tail_freq = 2.2, 1.0

    elif action == "walk":
        # `quadruped_beast`: 4-legged diagonal alternating trot (`(leg_0, leg_2)` vs `(leg_1, leg_3)`),
        # 2-beat upper-body & bone-collar bounce, forward head/horn lead, and rhythmic tail whip!
        bounce = int(round(-1.6 * abs(sin1)))
        sway_x = int(round(1.2 * sin1 * facing))
        torso_dx, torso_dy = sway_x, bounce

        collar_dx = sway_x + int(round(0.8 * cos1 * facing))
        collar_dy = bounce + int(round(1.2 * math.sin(2.0 * phase - 0.3)))

        head_dx = sway_x + int(round(1.4 * cos1 * facing))
        head_dy = bounce + int(round(1.4 * math.sin(2.0 * phase - 0.45)))
        horns_dx, horns_dy = head_dx, head_dy

        jaw_open = 1 if frame_idx in (3, 4, 11, 12) else 0

        leg_offsets = [
            foot_step_offset(phase, facing, lift_px=2, stride_px=2),
            foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2),
            foot_step_offset(phase + 0.35, facing, lift_px=2, stride_px=2),
            foot_step_offset(phase + math.pi + 0.35, facing, lift_px=2, stride_px=2),
        ]
        tail_amp_y, tail_freq = 2.4, 2.0

    elif action == "attack":
        # Signature Mega Houndoom "Dark Pulse / Fire Blast / Crunch" Roar & Lunge!
        # Frames 0..3: Low predator crouch wind-up — pull body & skull armor back (`-3px`),
        #              lower head (`dy = +2px`), and open jaw (`jaw_open = 1`) charging hellfire!
        # Frames 4..9: Explosive forward lunge (`+5..6px` head/horns & bone_collar thrust!) —
        #              front paws step forward into the strike, jaw roars wide (`frames 4,5,6`) then
        #              snaps shut & crunches (`frame 7 -> 8`), while the devil tail whips high!
        # Frames 10..15: Smooth follow-through recovery to stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.60 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        torso_dx = int(round(4.0 * k * facing))
        torso_dy = int(round(-1.0 * k)) if k > 0 else int(round(1.5 * abs(k)))

        collar_dx = int(round(5.0 * k * facing))
        collar_dy = torso_dy + (int(round(-0.8 * k)) if k > 0 else int(round(0.8 * abs(k))))

        head_dx = int(round(6.0 * k * facing))
        if frame_idx in (1, 2, 3):
            head_dy = 2 if frame_idx in (2, 3) else 1
        elif frame_idx in (4, 5):
            head_dy = 1 if frame_idx == 4 else 0
        elif frame_idx in (6, 7, 8):
            head_dy = -1
        else:
            head_dy = 0
        horns_dx, horns_dy = head_dx, head_dy

        # Jaw opens during wind-up charge (1..3) and initial roar/bite (4..6, 8), snaps shut on crunch impact (7)!
        jaw_open = 1 if frame_idx in (1, 2, 3, 4, 5, 6, 8) else 0

        # Front paws step forcefully into the lunge while rear paws brace and push
        front_step = int(round(3.5 * max(0.0, k) * facing))
        front_back = int(round(2.0 * min(0.0, k) * facing))
        front_lift = -2 if frame_idx in (4, 5) else 0
        rear_slide = int(round(1.8 * max(0.0, k) * facing))
        if not is_back:
            # Front view (and flipped SE/E): leg_0 & leg_1 are front legs; leg_2 & leg_3 are rear legs
            leg_offsets = [
                (front_step + front_back, front_lift),
                (int(round((front_step + front_back) * 0.75)), front_lift // 2),
                (rear_slide, 0),
                (rear_slide, 0),
            ]
        else:
            # Back view (and flipped NW): leg_2 & leg_3 are front legs; leg_0 & leg_1 are rear legs
            leg_offsets = [
                (rear_slide, 0),
                (rear_slide, 0),
                (int(round((front_step + front_back) * 0.75)), front_lift // 2),
                (front_step + front_back, front_lift),
            ]
        tail_amp_y, tail_freq = 3.0, 2.0

    elif action == "hurt":
        # Knockback + hit-shake + head/horns recoiling upward + open jaw flinch + hurt sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        torso_dx = int(round(-3.6 * env * facing)) + jit
        torso_dy = int(round(-1.0 * env))

        collar_dx = torso_dx + int(round(-0.6 * env * facing))
        collar_dy = int(round(-1.5 * env))

        head_dx = collar_dx + int(round(-0.8 * env * facing))
        head_dy = int(round(-2.2 * env))
        horns_dx, horns_dy = head_dx, head_dy

        jaw_open = 1 if (2 <= frame_idx <= 7) else 0

        slide_x = int(round(torso_dx * 0.65))
        leg_offsets = [(slide_x, 0), (slide_x, 0), (slide_x, 0), (slide_x, 0)]
        tail_amp_y, tail_freq = 2.6, 2.0

    comp = [
        wave_tail_layer(layers["tail"], torso_dx, torso_dy, phase * tail_freq, facing, amp_y=tail_amp_y),
        shift_layer_int(layers["leg_0"], leg_offsets[0][0], leg_offsets[0][1]),
        shift_layer_int(layers["leg_2"], leg_offsets[2][0], leg_offsets[2][1]),
        shift_layer_int(layers["torso"], torso_dx, torso_dy),
        shift_layer_int(layers["leg_1"], leg_offsets[1][0], leg_offsets[1][1]),
        shift_layer_int(layers["leg_3"], leg_offsets[3][0], leg_offsets[3][1]),
        shift_layer_int(layers["head_upper"], head_dx, head_dy),
        shift_layer_int(layers["bone_collar"], collar_dx, collar_dy),
        shift_layer_int(layers["lower_jaw"], head_dx, head_dy),
        shift_layer_int(layers["horns"], horns_dx, horns_dy),
    ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    if jaw_open and mouth_line is not None:
        # Small mouth opening along the face's mouth line (no jaw drop). 2px only at the attack roar peak.
        amount = 2 if (action == "attack" and frame_idx in (4, 5)) else 1
        line = mouth_open.transform_points(mouth_line, dx=head_dx, dy=head_dy)
        dark = palette_rgba[int(np.argmin(palette_rgba[:, :3].astype(np.int32).sum(axis=1))), :3]
        out = mouth_open.open_mouth(
            out, line, amount=amount, deep=tuple(int(c) for c in dark), inner=MOUTH_INNER,
            peak=0.55, normal=MOUTH_NORMAL,
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    front_raw = np.array(Image.open(MEGA_DIR / "front.png").convert("RGBA"))
    back_raw = np.array(Image.open(MEGA_DIR / "back.png").convert("RGBA"))

    palette_rgba = extract_palette(front_raw, back_raw)

    # Tight bounds of front (67x83) and back (67x83)
    max_w, max_h = 67, 83
    cell_w = max_w + 21  # 88 (generous horizontal margin for +6px lunge & tail whip)
    cell_h = max_h + 17  # 100
    foot_y = cell_h - 4  # 96

    dir_layers: Dict[str, Tuple[Dict[str, np.ndarray], float, bool]] = {}
    mouth_lines: Dict[str, object] = {}
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
        mouth_lines[d_name] = mouth_line_cell(src_96, use_back, flip_x, cell_w, foot_y)

    sheets_by_action: Dict[str, np.ndarray] = {}
    master_rows: List[np.ndarray] = []

    for act in ACTIONS:
        act_sheet = np.zeros((8 * cell_h, N_FRAMES * cell_w, 4), dtype=np.uint8)
        for d_idx, d_name in enumerate(DIRS):
            layers, facing, use_back = dir_layers[d_name]
            for f_idx in range(N_FRAMES):
                frame_rgba = synthesize_houndoom_mega_frame(
                    layers,
                    (cell_h, cell_w, 4),
                    act,
                    f_idx,
                    facing,
                    use_back,
                    palette_rgba,
                    mouth_lines[d_name],
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

    # Also export clean 16-frame animated GIFs (`front.gif`, `front-reverse.gif`, `back.gif`, `back-reverse.gif`)
    # in `MEGA_DIR` to replace the 19-frame GIFs that had pink/red laser line noise artifacts.
    # (2026-11-04) Only with `--gifs`: the front/back png/gif are not overwritten by default.
    if "--gifs" in sys.argv:
        export_gifs(sheets_by_action, cell_w, cell_h)
    write_meta(palette_rgba, cell_w, cell_h, foot_y)


def export_gifs(sheets_by_action, cell_w, cell_h) -> None:
    idle_sheet = sheets_by_action["idle"]
    front_frames = [idle_sheet[0:cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    front_rev_frames = [f[:, ::-1, :].copy() for f in front_frames]
    back_frames = [idle_sheet[4 * cell_h:5 * cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    back_rev_frames = [f[:, ::-1, :].copy() for f in back_frames]

    save_transparent_gif(front_frames, MEGA_DIR / "front.gif", duration_ms=75)
    save_transparent_gif(front_rev_frames, MEGA_DIR / "front-reverse.gif", duration_ms=75)
    save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
    save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)


def write_meta(palette_rgba, cell_w, cell_h, foot_y) -> None:
    meta = {
        "schema": "agent-sprite-forge.meta.v2",
        "key": "0229_houndoom-mega",
        "archetype": "quadruped_beast",
        "rig_parts": [
            "tail",
            "leg_0",
            "leg_2",
            "torso",
            "leg_1",
            "leg_3",
            "head_upper",
            "bone_collar",
            "lower_jaw",
            "horns",
        ],
        "Idle": "../../monsterSprite/gen2/0229_houndoom/mega/houndoom-mega/agent-sprite-forge/idle.png",
        "Walk": "../../monsterSprite/gen2/0229_houndoom/mega/houndoom-mega/agent-sprite-forge/walk.png",
        "Attack": "../../monsterSprite/gen2/0229_houndoom/mega/houndoom-mega/agent-sprite-forge/attack.png",
        "Hurt": "../../monsterSprite/gen2/0229_houndoom/mega/houndoom-mega/agent-sprite-forge/hurt.png",
        "Sheet": "../../monsterSprite/gen2/0229_houndoom/mega/houndoom-mega/agent-sprite-forge/sheet.png",
        "Preview": "../../monsterSprite/gen2/0229_houndoom/mega/houndoom-mega/agent-sprite-forge/preview.png",
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
    print(f"Successfully forged Mega Houndoom 8-dir x 4-action bundle in {OUT_DIR} ({cell_w}x{cell_h} cell)")


if __name__ == "__main__":
    main()
