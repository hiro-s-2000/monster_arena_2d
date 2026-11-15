#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Scizor (`0212_scizor-mega`) modeled directly after Normal & Mega Scizor's
official GIF motion (`0212_scizor/normal/front.gif` & `0212_scizor/mega/scizor-mega/front.gif`, `back.gif`):
  - Exact color & connected-component anatomical layer separation (8 independent rig layers):
      1. `wing_L`, `wing_R` (translucent cyan-white 4-wing set with red upper-wing leading edges —
                             high-frequency insectoid flutter vibration in `idle`, `walk`, and `attack`)
      2. `pincer_far` (far-side giant serrated steel pincer & forearm with glowing cyan core)
      3. `leg_L`, `leg_R` (sharp red armored legs & silver claw tips — 100% grounded `(0, 0)` in `idle`,
                           crisp alternating bipedal step gait in `walk`, forward lunge step in `attack`)
      4. `torso` (neck joint, black-steel shoulder pauldrons, red thorax, segmented dark-steel abdomen
                  plates, hip armor, and pelvis)
      5. `head` (3-horned red cranium, dark-steel forehead crest, and fierce cyan eye — 100% rigid
                 integer translation so the face, horns, and eye never deform by a single pixel)
      6. `pincer_near` (foreground giant serrated saw-pincer & forearm with glowing cyan core and
                        white lower saw-blade — rhythmic fighting-stance undulation in `idle`,
                        counter-swing in `walk`, and explosive +6..7px Bullet Punch / X-Scissor
                        combo lunge in `attack`)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen2/0212_scizor/mega/scizor-mega"
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
        | ((r == 52) & (g == 52) & (b == 68))
        | ((r == 57) & (g == 57) & (b == 74))
        | ((r == 64) & (g == 64) & (b == 64))
        | ((r == 98) & (g == 41) & (b == 41))
        | ((r == 102) & (g == 41) & (b == 41))
    )


def is_red_armor(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    return a & (r >= 150) & (g <= 110)


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
    Segments native 96x96 `front.png` (facing left) into 8 exact anatomical masks:
      - `wing_L`: Lower-left small translucent wing behind left shoulder (`x=41..48, y=39..46`).
      - `wing_R`: Upper-right main wing (red leading edge + white veined membrane) and
                  lower-right small wing (`x=55..74, y=12..48`).
      - `pincer_far`: Right extended giant saw-pincer & forearm with cyan core (`x=60..86, y=36..66`).
      - `leg_L`: Left sharp armored leg & silver claw tip (`x=31..45, y=53..84`).
      - `leg_R`: Right sharp armored leg & silver claw tip (`x=55..67, y=55..87`).
      - `torso`: Neck, black-steel shoulder pauldrons, red chest, segmented abdomen, hip armor, and pelvis.
      - `head`: 3-horned red cranium, dark-steel forehead crest, and cyan eye (`x=40..61, y=7..30`).
      - `pincer_near`: Foreground left giant serrated saw-pincer (upper red blade, cyan core,
                       and lower white saw-blade) & forearm (`x=9..43, y=32..59`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    r = arr96[:, :, 0].astype(np.int32)
    yy, xx = np.mgrid[0:h, 0:w]

    is_red = a & ((r == 98) | (r == 164) | (r == 213) | (r == 246))
    lbl_red, _ = ndi.label(is_red)
    is_ws = a & ((r == 148) | (r == 180) | (r == 253))
    lbl_ws, _ = ndi.label(is_ws)
    is_cyan = a & ((r == 0) | (r == 32))

    # 1. Head: 3-horned cranium, dark slate crest, cyan eye, and complete 1px black outline
    head_zone = (
        ((yy <= 16) & (xx <= 62))
        | ((yy >= 17) & (yy <= 19) & (xx <= 61))
        | ((yy == 20) & (xx <= 59))
        | ((yy == 21) & (xx <= 58))
        | ((yy >= 22) & (yy <= 23) & (xx <= 59))
        | ((yy >= 24) & (yy <= 26) & (xx <= 58))
        | ((yy == 27) & (xx <= 57))
        | ((yy == 28) & (xx <= 56))
        | ((yy == 29) & (xx >= 44) & (xx <= 55))
        | ((yy == 30) & (xx >= 45) & (xx <= 52))
    )
    head_m = a & head_zone

    # 2. Wings (`wing_L` and `wing_R`):
    wing_upper_zone = (
        ((yy >= 12) & (yy <= 30) & (xx >= 56))
        | ((yy == 31) & (((xx >= 56) & (xx <= 58)) | (xx >= 66)))
        | ((yy == 32) & (((xx >= 56) & (xx <= 57)) | (xx >= 68)))
    ) & (~head_m)
    wing_lower_seed = (lbl_ws == 24) | (lbl_ws == 25)
    wing_lower_dil = ndi.binary_dilation(wing_lower_seed, iterations=1) & a & (
        ((xx >= 41) & (xx <= 48) & (yy >= 39) & (yy <= 46))
        | ((xx >= 55) & (xx <= 64) & (yy >= 39) & (yy <= 48))
    ) & (~is_red) & (~is_cyan) & (arr96[:, :, 0] != 52)
    wings_all = (a & wing_upper_zone) | wing_lower_seed | wing_lower_dil
    wing_L_m = wings_all & (xx <= 50)
    wing_R_m = wings_all & (xx > 50)

    # 3. Pincer Near (left giant saw-pincer & forearm):
    pincer_near_seed = (
        (lbl_red == 6)
        | (lbl_ws == 30)
        | (a & (xx <= 33) & (yy >= 32) & (yy <= 59))
        | (a & (xx >= 34) & (xx <= 38) & (yy >= 36) & (yy <= 51))
    )
    pincer_near_m = (
        pincer_near_seed
        | (
            ndi.binary_dilation(pincer_near_seed, iterations=1)
            & a & (xx <= 44) & (yy >= 32)
            & ((xx <= 34) | (yy <= 51))
            & (~wings_all)
        )
    )
    shoulder_L = a & (xx >= 39) & (xx <= 48) & (yy >= 31) & (yy <= 37) & (~(lbl_red == 6))
    shoulder_L |= a & (xx >= 42) & (xx <= 48) & (yy >= 38) & (yy <= 39) & (~(lbl_red == 6))
    hip_L = a & (xx >= 40) & (xx <= 47) & (yy >= 47) & (yy <= 53) & (~(lbl_red == 6))
    pincer_near_m = pincer_near_m & (~shoulder_L) & (~hip_L)

    # 4. Pincer Far (right giant saw-pincer & forearm):
    pincer_far_seed = (
        (lbl_red == 9) | (lbl_red == 10)
        | (lbl_ws == 23) | (lbl_ws == 35) | (lbl_ws == 37) | (lbl_ws == 38) | (lbl_ws == 39)
        | (a & (xx >= 64) & (yy >= 38) & (yy <= 66) & (~wings_all) & (~((xx <= 67) & (yy >= 55))))
    )
    pincer_far_m = (
        pincer_far_seed
        | (
            ndi.binary_dilation(pincer_far_seed, iterations=1)
            & a & (xx >= 60) & (yy >= 37) & (yy <= 66)
            & (~wings_all)
            & (~((xx <= 66) & (yy >= 55)))
            & (~((xx <= 67) & (yy <= 36)))
        )
    )

    # 5. Leg L (left armored leg & silver claw tip):
    leg_L_zone = (
        ((yy == 53) & (xx >= 35) & (xx <= 39))
        | ((yy == 54) & (xx >= 34) & (xx <= 41))
        | ((yy >= 55) & (yy <= 56) & (xx >= 33) & (xx <= 42))
        | ((yy == 57) & (xx >= 33) & (xx <= 43))
        | ((yy >= 58) & (yy <= 86) & (xx >= 31) & (xx <= 45))
    )
    leg_L_m = a & leg_L_zone & (~pincer_near_m)

    # 6. Leg R (right armored leg & silver claw tip):
    leg_R_zone = (
        ((yy == 55) & (xx >= 58) & (xx <= 62))
        | ((yy == 56) & (xx >= 56) & (xx <= 64))
        | ((yy >= 57) & (yy <= 60) & (xx >= 55) & (xx <= 67))
        | ((yy >= 61) & (yy <= 88) & (xx >= 56) & (xx <= 67))
    )
    leg_R_m = a & leg_R_zone & (~pincer_far_m)

    # 7. Torso: all remaining central body pixels
    used = head_m | wings_all | pincer_near_m | pincer_far_m | leg_L_m | leg_R_m
    torso_m = a & (~used)

    return {
        "wing_L": wing_L_m,
        "wing_R": wing_R_m,
        "pincer_far": pincer_far_m,
        "leg_L": leg_L_m,
        "leg_R": leg_R_m,
        "torso": torso_m,
        "head": head_m,
        "pincer_near": pincer_near_m,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` (facing right) into matching 8 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    r = arr96[:, :, 0].astype(np.int32)
    yy, xx = np.mgrid[0:h, 0:w]

    is_red = a & ((r == 102) | (r == 166) | (r == 217) | (r == 242))
    lbl_red, _ = ndi.label(is_red)
    is_ws = a & ((r == 115) | (r == 191) | (r == 248))
    lbl_ws, _ = ndi.label(is_ws)

    # 1. Head on back.png (3-horned cranium at y=5..26):
    head_zone = (
        ((yy <= 17) & (xx >= 35) & (xx <= 55))
        | ((yy >= 18) & (yy <= 20) & (xx >= 40) & (xx <= 54))
        | ((yy == 21) & (xx >= 40) & (xx <= 51))
        | ((yy >= 22) & (yy <= 23) & (xx >= 41) & (xx <= 50))
        | ((yy == 24) & (xx >= 42) & (xx <= 49))
        | ((yy == 25) & (xx >= 42) & (xx <= 49))
        | ((yy == 26) & (xx >= 44) & (xx <= 48))
    )
    head_m = a & head_zone

    # 2. Wings on back.png (`wing_L` and `wing_R`, including the right wing-root shoulder cap at x=51..57, y=31..36):
    wing_upper_L = a & (~head_m) & (
        ((yy >= 11) & (yy <= 27) & (xx <= 41))
        | ((yy == 28) & (xx <= 41))
        | ((yy >= 29) & (yy <= 31) & (xx <= 42))
    )
    wing_upper_R = a & (~head_m) & (
        ((yy >= 11) & (yy <= 36) & (xx >= 48))
    )
    wing_lower_seed = (lbl_ws == 4) | (lbl_ws == 5)
    wing_lower_dil = ndi.binary_dilation(wing_lower_seed, iterations=1) & a & (
        ((xx >= 31) & (xx <= 41) & (yy >= 31) & (yy <= 43))
        | ((xx >= 46) & (xx <= 56) & (yy >= 31) & (yy <= 43))
    ) & (~((xx >= 41) & (xx <= 46)))
    wings_all = wing_upper_L | wing_upper_R | wing_lower_seed | wing_lower_dil
    wing_L_m = wings_all & (xx <= 43)
    wing_R_m = wings_all & (xx >= 45)

    # 3. Pincer Far (left extended pincer on back.png):
    pincer_far_zone = (
        ((xx <= 30) & (yy >= 39) & (yy <= 65))
        | ((xx >= 31) & (xx <= 35) & (yy >= 39) & (yy <= 52) & (~wings_all))
    )
    hip_ball_L = a & (xx >= 34) & (yy >= 44) & (yy <= 54) & (~(lbl_red == 4)) & (~(lbl_ws == 8))
    pincer_far_m = a & pincer_far_zone & (~wings_all) & (~hip_ball_L)

    # 4. Pincer Near (right extended pincer & forearm on back.png):
    pincer_near_seed = (lbl_red == 3) | (lbl_red == 7) | (lbl_ws == 9) | (lbl_ws == 11)
    pincer_near_m = (
        pincer_near_seed
        | (a & (xx >= 58) & (yy >= 38) & (yy <= 65) & (~((xx <= 60) & (yy >= 53))))
        | (
            ndi.binary_dilation(pincer_near_seed, iterations=1)
            & a & (xx >= 54) & (yy >= 36) & (yy <= 65)
            & (~wings_all)
            & (~((xx <= 60) & (yy >= 53)))
        )
    )

    # 5. Leg L (left armored leg on back.png):
    leg_L_seed = (lbl_red == 9) | (lbl_ws == 13)
    leg_L_m = (
        leg_L_seed
        | (a & (xx >= 31) & (xx <= 41) & (yy >= 55) & (yy <= 87) & (~((xx >= 40) & (yy <= 61))))
        | (ndi.binary_dilation(leg_L_seed, iterations=1) & a & (xx >= 31) & (xx <= 42) & (yy >= 53))
    )

    # 6. Leg R (right armored leg on back.png):
    leg_R_seed = (lbl_red == 10) | (lbl_ws == 14) | (a & (r == 242))
    leg_R_m = (
        leg_R_seed
        | (a & (xx >= 50) & (xx <= 60) & (yy >= 55) & (yy <= 90) & (~pincer_near_m))
        | (ndi.binary_dilation(leg_R_seed, iterations=1) & a & (xx >= 49) & (xx <= 60) & (yy >= 54) & (~pincer_near_m))
    )

    used = head_m | wings_all | pincer_far_m | pincer_near_m | leg_L_m | leg_R_m
    torso_m = a & (~used)

    return {
        "wing_L": wing_L_m,
        "wing_R": wing_R_m,
        "pincer_far": pincer_far_m,
        "leg_L": leg_L_m,
        "leg_R": leg_R_m,
        "torso": torso_m,
        "head": head_m,
        "pincer_near": pincer_near_m,
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
    src_96 = base_96[:, ::-1, :].copy() if flip_x else base_96.copy()
    masks_96 = {k: (m[:, ::-1].copy() if flip_x else m.copy()) for k, m in raw_masks.items()}
    if flip_x:
        masks_96["wing_L"], masks_96["wing_R"] = masks_96["wing_R"], masks_96["wing_L"]
        masks_96["leg_L"], masks_96["leg_R"] = masks_96["leg_R"], masks_96["leg_L"]

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

    yy_c, xx_c = np.mgrid[0:cell_h, 0:cell_w]
    nat_y = yy_c - dst_y0 + y0
    red_fill_m = is_red_armor(canvas)

    layers: Dict[str, np.ndarray] = {}

    # 1. `wing_L`, `wing_R`:
    # Cushion ONLY inside the eroded interior of upper `torso` (`nat_y <= 43`) and lower `head` (`nat_y >= 18`)
    # so wing cushions NEVER protrude past the outer silhouette of the shoulder pauldrons or waist!
    fg_over_wings_raw = (c_masks["torso"] & (nat_y <= 43)) | (c_masks["head"] & (nat_y >= 18))
    fg_over_wings = ndi.binary_erosion(fg_over_wings_raw, iterations=2)
    layers["wing_L"] = build_part_layer(canvas, c_masks["wing_L"], fg_over_wings, cushion_px=4)
    layers["wing_R"] = build_part_layer(canvas, c_masks["wing_R"], fg_over_wings, cushion_px=4)

    # 2. `pincer_far`:
    # Cushion ONLY inside the eroded interior of `torso` at the red forearm joint
    # and strictly using `red_fill_m` so the cushion never protrudes past the outer silhouette!
    torso_eroded = ndi.binary_erosion(c_masks["torso"], iterations=2)
    if not is_back:
        fg_over_pfar = torso_eroded & (nat_y >= 35) & (nat_y <= 42)
        layers["pincer_far"] = build_part_layer(
            canvas, c_masks["pincer_far"], fg_over_pfar, cushion_px=4, allowed_fill_mask=red_fill_m
        )
    else:
        fg_over_pfar = torso_eroded & (nat_y >= 38) & (nat_y <= 45)
        layers["pincer_far"] = build_part_layer(
            canvas, c_masks["pincer_far"], fg_over_pfar, cushion_px=4, allowed_fill_mask=red_fill_m
        )

    # 3. `leg_L`, `leg_R`:
    # Red-only interior cushion inside `torso_eroded` + 5px vertical upward red stem inside `torso_eroded`:
    for fk in ("leg_L", "leg_R"):
        fm = c_masks[fk]
        lyr = build_part_layer(canvas, fm, torso_eroded, cushion_px=4, allowed_fill_mask=red_fill_m)
        f_ys, f_xs = np.where(fm)
        if len(f_ys) > 0:
            for x_col in np.unique(f_xs):
                col_ys = f_ys[f_xs == x_col]
                cy_min = int(col_ys.min())
                fill_px = canvas[cy_min, x_col]
                for y_scan in col_ys:
                    if not is_dark_outline(canvas[y_scan:y_scan + 1, x_col:x_col + 1])[0, 0]:
                        fill_px = canvas[y_scan, x_col]
                        break
                for dy_up in range(1, 6):
                    py = cy_min - dy_up
                    if py >= 0 and torso_eroded[py, x_col]:
                        lyr[py, x_col] = (
                            canvas[py, x_col]
                            if not is_dark_outline(canvas[py:py + 1, x_col:x_col + 1])[0, 0]
                            else fill_px
                        )
        layers[fk] = lyr

    # 4. `torso`:
    # Cushion inside eroded `head` (at neck `nat_y >= 24`) and eroded `pincer_near` (at forearm root `nat_y >= 36, nat_y <= 43`)
    head_eroded = ndi.binary_erosion(c_masks["head"], iterations=1)
    pnear_eroded = ndi.binary_erosion(c_masks["pincer_near"], iterations=2)
    fg_over_torso = (head_eroded & (nat_y >= 24)) | (pnear_eroded & (nat_y >= 36) & (nat_y <= 43))
    layers["torso"] = build_part_layer(canvas, c_masks["torso"], fg_over_torso, cushion_px=5)

    # 5. `head`: 100% rigid foreground layer (zero distortion)
    layers["head"] = build_part_layer(canvas, c_masks["head"], np.zeros_like(c_masks["head"]), cushion_px=0)

    # 6. `pincer_near`:
    if not is_back:
        layers["pincer_near"] = build_part_layer(
            canvas, c_masks["pincer_near"], np.zeros_like(c_masks["pincer_near"]), cushion_px=0
        )
    else:
        fg_over_pnear = torso_eroded & (nat_y >= 35) & (nat_y <= 42)
        layers["pincer_near"] = build_part_layer(
            canvas, c_masks["pincer_near"], fg_over_pnear, cushion_px=3, allowed_fill_mask=red_fill_m
        )

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
        (mid_x - 13 - drift, max(2, top_y + 3 - drift // 2)),
        (mid_x + 13 + drift, max(2, top_y + 4 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (253, 253, 253, 255)
    return out


def synthesize_scizor_mega_frame(
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
    pnear_dx, pnear_dy = 0, 0
    pfar_dx, pfar_dy = 0, 0
    wing_L_dx, wing_L_dy = 0, 0
    wing_R_dx, wing_R_dy = 0, 0
    leg_L_dx, leg_L_dy = 0, 0
    leg_R_dx, leg_R_dy = 0, 0

    # High-frequency insectoid wing vibration (4 cycles per 16 frames, matching Scizor's official GIF):
    flutter_sin = math.sin(4.0 * phase)
    flutter_cos = math.cos(4.0 * phase)

    if action == "idle":
        # Modeled directly after Normal & Mega Scizor's official GIFs:
        # 1. Both armored feet (`leg_L`, `leg_R`) stay 100% planted on the ground `(0, 0)`.
        # 2. Sharp fighting-stance breathing: `torso` and `head` bob rhythmically (`±1..2px`).
        # 3. Both giant serrated saw-pincers (`pincer_near`, `pincer_far`) weave up/down and
        #    forward/back in a menacing guard stance.
        # 4. Translucent cyan-white wings (`wing_L`, `wing_R`) vibrate in a high-frequency buzz!
        torso_dy = int(round(1.0 * sin1))
        head_dy = int(round(1.4 * math.sin(phase - 0.25)))
        head_dx = int(round(0.8 * max(0.0, math.sin(phase - 0.2)) * facing))

        pnear_dx = int(round(1.6 * math.sin(phase - 0.5) * facing))
        pnear_dy = torso_dy + int(round(1.8 * math.sin(phase - 0.4)))

        if not is_back:
            pfar_dx = int(round(1.4 * math.sin(phase - 1.1) * facing))
            pfar_dy = torso_dy + int(round(1.6 * math.sin(phase - 1.0)))

            buzz_x = int(round(-1.2 * max(0.0, flutter_sin) * facing))
            buzz_y = int(round(-1.2 * max(0.0, flutter_cos)))
            wing_L_dx, wing_L_dy = torso_dx + buzz_x, torso_dy + buzz_y
            wing_R_dx, wing_R_dy = torso_dx + buzz_x, torso_dy + buzz_y
        else:
            # On back views (`N, NE, NW`), `pincer_far` extends on the opposite lateral side from `pincer_near`
            pfar_dx = int(round(-1.2 * math.sin(phase - 1.1) * facing))
            pfar_dy = torso_dy + int(round(1.6 * math.sin(phase - 1.0)))

            spread_x = int(round(1.2 * max(0.0, flutter_sin)))
            buzz_y = int(round(-1.2 * max(0.0, flutter_cos)))
            wing_L_dx, wing_L_dy = torso_dx - spread_x, torso_dy + buzz_y
            wing_R_dx, wing_R_dy = torso_dx + spread_x, torso_dy + buzz_y

    elif action == "walk":
        # `biped_pincer_striker`: Crisp alternating bipedal step gait (`lift_px=2, stride_px=2`),
        # 2-beat body bounce, counter-swinging giant saw-pincers, and high-speed wing buzz!
        bounce = int(round(-1.6 * abs(sin1)))
        sway_x = int(round(1.2 * sin1 * facing))
        torso_dx, torso_dy = sway_x, bounce

        head_dx = sway_x + int(round(1.0 * cos1 * facing))
        head_dy = bounce + int(round(1.2 * math.sin(2.0 * phase - 0.3)))

        pnear_dx = sway_x + int(round(2.2 * sin1 * facing))
        pnear_dy = bounce + int(round(1.6 * cos1))

        if not is_back:
            pfar_dx = sway_x + int(round(-2.0 * sin1 * facing))
            pfar_dy = bounce + int(round(-1.5 * cos1))

            buzz_x = int(round(-1.4 * max(0.0, flutter_sin) * facing))
            buzz_y = int(round(-1.4 * max(0.0, flutter_cos)))
            wing_L_dx, wing_L_dy = sway_x + buzz_x, bounce + buzz_y
            wing_R_dx, wing_R_dy = sway_x + buzz_x, bounce + buzz_y
        else:
            pfar_dx = sway_x + int(round(1.5 * sin1 * facing))
            pfar_dy = bounce + int(round(-1.6 * cos1))

            spread_x = int(round(1.4 * max(0.0, flutter_sin)))
            buzz_y = int(round(-1.4 * max(0.0, flutter_cos)))
            wing_L_dx, wing_L_dy = sway_x - spread_x, bounce + buzz_y
            wing_R_dx, wing_R_dy = sway_x + spread_x, bounce + buzz_y

        leg_L_dx, leg_L_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        leg_R_dx, leg_R_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Mega Scizor "Bullet Punch / X-Scissor" High-Speed Combo Lunge!
        # Frames 0..3: Wind-up — pull body back (`-3px`), cock both giant saw-pincers back & wide, flare wings!
        # Frames 4..9: Bullet Punch & X-Scissor Strike — sharp forward lunge (`+5px` body/head),
        #              rapid 1-2 & simultaneous pincer thrust (`+6..7px` forward!) with intense wing buzz!
        # Frames 10..15: Follow-through recovery to fighting stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.65 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        torso_dx = int(round(4.5 * k * facing))
        torso_dy = int(round(1.0 * abs(k)))

        head_dx = int(round(5.2 * k * facing))
        head_dy = int(round(1.5 * k)) if k > 0 else int(round(-1.0 * abs(k)))

        if not is_back:
            # Front / Side-Front Two-stage Bullet Punch + X-Scissor pincer trajectory:
            if frame_idx in (1, 2, 3):
                pnear_dx = int(round(4.5 * k * facing))
                pnear_dy = -1
                pfar_dx = int(round(4.0 * k * facing))
                pfar_dy = -2 if frame_idx in (2, 3) else -1
            elif frame_idx in (4, 5):
                # Strike 1: High-speed Bullet Punch jab (`pincer_far` leads +6px, `pincer_near` +5.5px raised)
                pnear_dx = int(round(5.5 * k * facing))
                pnear_dy = -1
                pfar_dx = int(round(6.2 * k * facing))
                pfar_dy = 0
            elif frame_idx in (6, 7, 8):
                # Strike 2: Full-power X-Scissor / Dual Pincer Crush (+7px `pincer_near`, +6px `pincer_far` clamping!)
                pnear_dx = int(round(7.0 * k * facing))
                pnear_dy = 1 if frame_idx in (6, 7) else 0
                pfar_dx = int(round(6.0 * k * facing))
                pfar_dy = 1
            else:
                pnear_dx = int(round(5.0 * k * facing))
                pnear_dy = int(round(1.0 * k))
                pfar_dx = int(round(4.5 * k * facing))
                pfar_dy = int(round(1.0 * k))
        else:
            # Back view (`N, NE, NW`): Both pincers thrust forward-upward (`dy = -3..4px`) and outward
            pnear_dx = int(round(5.8 * k * facing))
            pnear_dy = int(round(-2.5 * k))
            pfar_dx = int(round(2.5 * k * facing))
            pfar_dy = int(round(-3.5 * k))

        # Intense wing flare & buzz during wind-up and strike
        wing_flare = 2 if (2 <= frame_idx <= 8) else 1
        if not is_back:
            buzz_x = int(round(-wing_flare * (1 if frame_idx % 2 == 0 else 0) * facing))
            buzz_y = int(round(-wing_flare * (1 if (frame_idx % 3) != 0 else 0)))
            wing_L_dx, wing_L_dy = torso_dx + buzz_x, torso_dy + buzz_y
            wing_R_dx, wing_R_dy = torso_dx + buzz_x, torso_dy + buzz_y
        else:
            spread = wing_flare if (frame_idx % 2 == 0) else 0
            buzz_y = -wing_flare if (frame_idx % 2 == 0) else -1
            wing_L_dx, wing_L_dy = torso_dx - spread, torso_dy + buzz_y
            wing_R_dx, wing_R_dy = torso_dx + spread, torso_dy + buzz_y

        # Lead foot steps powerfully into the lunge while rear foot anchors
        step_x = int(round(2.5 * max(0.0, k) * facing))
        if facing < 0:
            leg_L_dx, leg_L_dy = step_x, 0
        else:
            leg_R_dx, leg_R_dy = step_x, 0

    elif action == "hurt":
        # Knockback + hit-shake + pincers knocked upward/back + wing flare + hurt sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        torso_dx = int(round(-3.5 * env * facing)) + jit
        torso_dy = int(round(-1.0 * env))

        head_dx = torso_dx + int(round(-1.0 * env * facing))
        head_dy = int(round(-1.5 * env))

        pnear_dx = torso_dx + int(round(-1.8 * env * facing))
        pnear_dy = int(round(-2.2 * env))
        pfar_dx = torso_dx + int(round(-1.5 * env * facing))
        pfar_dy = int(round(-1.8 * env))

        if not is_back:
            wing_L_dx, wing_L_dy = torso_dx - int(round(1.0 * env * facing)), torso_dy - int(round(1.5 * env))
            wing_R_dx, wing_R_dy = torso_dx - int(round(1.0 * env * facing)), torso_dy - int(round(1.5 * env))
        else:
            wing_L_dx, wing_L_dy = torso_dx - int(round(1.5 * env)), torso_dy - int(round(1.5 * env))
            wing_R_dx, wing_R_dy = torso_dx + int(round(1.5 * env)), torso_dy - int(round(1.5 * env))

        slide_x = int(round(torso_dx * 0.6))
        leg_L_dx, leg_R_dx = slide_x, slide_x

    if not is_back:
        comp = [
            shift_layer_int(layers["wing_L"], wing_L_dx, wing_L_dy),
            shift_layer_int(layers["wing_R"], wing_R_dx, wing_R_dy),
            shift_layer_int(layers["pincer_far"], pfar_dx, pfar_dy),
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["head"], head_dx, head_dy),
            shift_layer_int(layers["pincer_near"], pnear_dx, pnear_dy),
        ]
    else:
        comp = [
            shift_layer_int(layers["pincer_far"], pfar_dx, pfar_dy),
            shift_layer_int(layers["pincer_near"], pnear_dx, pnear_dy),
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy),
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy),
            shift_layer_int(layers["wing_L"], wing_L_dx, wing_L_dy),
            shift_layer_int(layers["wing_R"], wing_R_dx, wing_R_dy),
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["head"], head_dx, head_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
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

    # Tight bounds of front (78x81) and back (72x86)
    max_w, max_h = 78, 86
    cell_w = max_w + 24  # 102 (generous horizontal margin for +7px Bullet Punch pincer thrust)
    cell_h = max_h + 16  # 102
    foot_y = cell_h - 4  # 98

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
                frame_rgba = synthesize_scizor_mega_frame(
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

    # Note: Mega Scizor's existing `front.gif` (84 frames, 92x82) and `back.gif` (20 frames, 83x89)
    # are genuine, artifact-free official pixel-art animations and are intentionally preserved intact!

    meta = {
        "generator": "forge_scizor_mega_actions",
        "key": "0212_scizor-mega",
        "dex": 212,
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
    print(f"Built 0212_scizor-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
