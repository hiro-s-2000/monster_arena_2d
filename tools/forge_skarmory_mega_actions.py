#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Skarmory (`0227_skarmory-mega`) modeled directly after Normal Skarmory's
official GIF motion (`0227_skarmory/normal/front.gif` & `back.gif`) and Mega Skarmory's
`avian_raptor_hover` anatomy (`front.png`, `back.png`):
  - Exact color & connected-component anatomical layer separation (8 independent rig layers):
      1. `wing_L` (far-side golden blade wing — dorsal golden blades behind the neck on `front.png`,
                   and swept-back golden underside blades on `back.png`)
      2. `tail` (signature crescent-hooked dark-navy steel tail feather)
      3. `leg_L`, `leg_R` (left and right legs, navy ankles, hip plumes, and giant hanging 3-clawed
                           golden/navy talons — 100% preserving the long golden talon geometry)
      4. `torso` (armored dark-navy neck & chest plates, golden body/back, and golden thighs/pelvis)
      5. `wing_R` (near-side giant blade wing — golden leading-edge blade + 3 dark-navy steel blades)
      6. `lower_jaw` (jagged white-toothed golden lower mandible with dark-crimson mouth interior
                      cushion — opens/closes `+1px` in `idle`, `walk`, `attack`, and `hurt`)
      7. `head_upper` (sharp golden upper beak, forehead crest, fierce crimson eye, and golden/navy
                       rear head-crest feathers — 100% rigid integer translation so the face, beak,
                       and crimson eye never deform by a single pixel)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mouth_open  # noqa: E402  (small face-following mouth opening, 2026-11-04)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen2/0227_skarmory/mega/skarmory-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16

# 2026-11-04 mouth fix: the jaw layer no longer drops; a small lens-shaped gap is carved along the closed mouth
# line instead (tools/mouth_open.py). Measured on front.png (native 96x96 coords), ordered hinge -> beak tip.
# The line is the red mouth line; the gap opens downward into the gum row so the lower teeth stay visible.
MOUTH_LINE_FRONT_NATIVE = [(17.0, 20.0), (9.0, 20.0)]
MOUTH_DEEP = (16, 16, 16)      # darkest palette colour (outline)
MOUTH_INNER = (131, 24, 32)    # existing dark-crimson mouth interior colour


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
        | ((r == 119) & (g == 63) & (b == 21))
    )


def is_gold_armor(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    return a & ((r == 200) | (r == 241) | (r == 255)) & (rgba[..., 1] >= 130)


def is_navy_armor(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    return a & ((r == 42) | (r == 85) | (r == 86))


def is_mouth_red(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    return a & (r == 131)


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
      - `wing_L`: Far-side golden blade wing rising behind the neck (`x=24..32, y=21..32`).
      - `tail`: Hooked dark-navy steel tail (`x=42..53, y=37..44`).
      - `leg_L`: Left leg shaft, navy ankle, and 2 golden + 2 navy hanging talons (`x=15..37, y=50..88`).
      - `leg_R`: Right leg shaft, hip plume, navy ankle, and 2 golden + 2 navy hanging talons (`x=38..65, y=46..87`).
      - `torso`: Armored dark-navy neck & chest plates, golden body, and golden thighs (`x=18..47, y=25..50`).
      - `wing_R`: Near-side giant golden & 3-navy-blade wing (`x=32..89, y=12..42`).
      - `lower_jaw`: Jagged white-toothed golden lower mandible (`x=9..23, y=20..25`).
      - `head_upper`: Sharp golden upper beak, forehead crest, fierce crimson eye, and golden/navy
                      rear head-crest feathers (`x=7..38, y=8..24`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    r = arr96[:, :, 0].astype(np.int32)
    yy, xx = np.mgrid[0:h, 0:w]

    is_navy = a & ((r == 42) | (r == 85) | (r == 86))
    is_gold = a & ((r == 200) | (r == 241) | (r == 255))
    lbl_n, _ = ndi.label(is_navy)
    lbl_g, _ = ndi.label(is_gold)

    # 1. `lower_jaw`:
    lower_jaw_zone = (
        ((yy == 20) & (xx >= 11) & (xx <= 16))
        | ((yy == 21) & (xx >= 9) & (xx <= 18))
        | ((yy == 22) & (xx >= 9) & (xx <= 21))
        | ((yy == 23) & (xx >= 10) & (xx <= 23))
        | ((yy == 24) & (xx >= 11) & (xx <= 23))
        | ((yy == 25) & (xx >= 13) & (xx <= 15))
    )
    lower_jaw_m = a & lower_jaw_zone

    # 2. `wing_L` (far-side golden wing blades behind neck):
    wing_L_zone = (
        ((yy == 21) & (xx >= 28) & (xx <= 30))
        | ((yy == 22) & (xx >= 27) & (xx <= 31))
        | ((yy == 23) & (xx >= 27) & (xx <= 32))
        | ((yy == 24) & (xx >= 26) & (xx <= 32))
        | ((yy >= 25) & (yy <= 26) & (xx >= 24) & (xx <= 29))
        | ((yy >= 27) & (yy <= 28) & (xx >= 24) & (xx <= 32))
        | ((yy == 29) & (xx >= 25) & (xx <= 32))
        | ((yy == 30) & (xx >= 26) & (xx <= 30))
        | ((yy == 31) & (xx >= 27) & (xx <= 29))
        | ((yy == 32) & (xx >= 29) & (xx <= 31))
    )
    wing_L_m = a & wing_L_zone

    # 3. `head_upper`:
    head_zone = (
        ((yy <= 19) & (xx <= 38))
        | ((yy == 20) & (xx <= 38))
        | ((yy == 21) & (xx <= 38) & (~((xx >= 27) & (xx <= 31))))
        | ((yy == 22) & (xx <= 38) & (~((xx >= 27) & (xx <= 32))))
        | ((yy == 23) & (xx >= 23) & (xx <= 26))
        | ((yy == 24) & (xx >= 24) & (xx <= 25))
    )
    head_upper_m = a & head_zone & (~lower_jaw_m) & (~wing_L_m)

    # 4. `tail` (hooked navy tail at x=42..53, y=37..44):
    tail_seed = (lbl_n == 11) | (lbl_n == 12)
    tail_m = tail_seed | (
        ndi.binary_dilation(tail_seed, iterations=1)
        & a & (xx >= 42) & (xx <= 53) & (yy >= 37) & (yy <= 44)
        & (~(lbl_g == 10)) & (~(lbl_n == 9))
    )

    # 5. `wing_R` (near-side golden leading-edge + 3 navy blades):
    wing_R_seed = (
        (lbl_g == 3)
        | (lbl_n == 4)
        | (lbl_n == 7)
        | (lbl_n == 8)
        | (lbl_n == 9)
        | (lbl_n == 10)
    )
    wing_R_m = (
        wing_R_seed
        | (
            ndi.binary_dilation(wing_R_seed, iterations=1)
            & a & (xx >= 32) & (yy >= 12) & (yy <= 42)
            & (~head_upper_m) & (~wing_L_m) & (lbl_g != 10)
        )
    ) & (~tail_m)

    # 6. `leg_L` (left lower leg shaft + navy ankle + 2 golden & 2 navy hanging talons, starting below the left thigh at y=50):
    leg_L_zone = (
        ((yy >= 50) & (yy <= 56) & (xx >= 31) & (xx <= 37))
        | ((yy >= 57) & (yy <= 62) & (xx >= 26) & (xx <= 37))
        | ((yy >= 61) & (yy <= 88) & (xx <= 37))
    )
    leg_L_m = a & leg_L_zone

    # 7. `leg_R` (right lower leg shaft + hip plume + navy ankle + 2 golden & 2 navy hanging talons):
    leg_R_zone = (
        ((yy >= 46) & (yy <= 58) & (xx >= 43) & (xx <= 56))
        | ((yy >= 50) & (yy <= 88) & (xx >= 38))
    )
    leg_R_m = a & leg_R_zone & (~leg_L_m)

    # 8. `torso`: remaining central body, segmented navy neck/chest, and golden thighs
    used = lower_jaw_m | head_upper_m | wing_L_m | wing_R_m | tail_m | leg_L_m | leg_R_m
    torso_m = a & (~used)

    return {
        "wing_L": wing_L_m,
        "tail": tail_m,
        "leg_L": leg_L_m,
        "leg_R": leg_R_m,
        "torso": torso_m,
        "wing_R": wing_R_m,
        "lower_jaw": lower_jaw_m,
        "head_upper": head_upper_m,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` (facing right) into matching 8 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    r = arr96[:, :, 0].astype(np.int32)
    yy, xx = np.mgrid[0:h, 0:w]

    is_navy = a & ((r == 42) | (r == 85) | (r == 86))
    is_gold = a & ((r == 200) | (r == 241) | (r == 255))
    lbl_n, _ = ndi.label(is_navy)
    lbl_g, _ = ndi.label(is_gold)

    # 1. `lower_jaw` on back.png (`x=78..87, y=14..20`):
    lower_jaw_zone = (
        ((yy == 14) & (xx >= 84))
        | ((yy == 15) & (xx >= 83))
        | ((yy == 16) & (xx >= 81))
        | ((yy == 17) & (xx >= 79))
        | ((yy == 18) & (xx >= 78))
        | ((yy == 19) & (xx >= 79))
        | ((yy == 20) & (xx >= 80))
    )
    lower_jaw_m = a & lower_jaw_zone

    # 2. `head_upper` on back.png (upper beak, eye, golden rear crest, and 2 navy crest feathers):
    head_zone = (
        ((yy <= 10) & (xx >= 58))
        | ((yy >= 11) & (yy <= 16) & (xx >= 57) & (~(lbl_g == 5)))
        | ((yy >= 17) & (yy <= 20) & (xx >= 68))
        | ((yy == 21) & (xx >= 68) & (xx <= 71))
    )
    head_upper_m = a & head_zone & (~lower_jaw_m)

    # 3. `tail` on back.png (hooked navy tail at x=17..42, y=28..42):
    tail_seed = (lbl_n == 11) | (lbl_n == 14)
    tail_m = tail_seed | (
        ndi.binary_dilation(tail_seed, iterations=2)
        & a & (xx >= 17) & (xx <= 42) & (yy >= 28) & (yy <= 42)
        & (~(lbl_n == 7)) & (~(lbl_n == 12)) & (~(lbl_n == 13))
        & (~is_gold)
    )

    # 4. `wing_R` on back.png (near-side right wing: upright golden leading blade G5 + 3 navy blades N7, N12, N13):
    wing_R_seed = (
        (lbl_g == 5)
        | (a & (yy == 11) & (xx >= 52) & (xx <= 55))
        | (lbl_n == 7)
        | (lbl_n == 12)
        | (lbl_n == 13)
    )
    wing_R_m = (
        wing_R_seed
        | (
            ndi.binary_dilation(wing_R_seed, iterations=1)
            & a & (xx >= 25) & (xx <= 65) & (yy >= 11) & (yy <= 47)
            & (~head_upper_m) & (~tail_m)
            & (~(lbl_g == 2)) & (~(lbl_g == 4)) & (~(lbl_g == 8)) & (~(lbl_g == 9))
            & (~(lbl_g == 10)) & (~(lbl_g == 11)) & (~(lbl_g == 12)) & (~(lbl_g == 13))
            & (~(lbl_g == 14)) & (~(lbl_g == 15))
            & (~(lbl_n == 10))
        )
    ) & (~head_upper_m) & (~tail_m)

    # 5. `wing_L` on back.png (far-side golden wing blades: gold 2, 4, 8, 9, 10, 11, 13):
    wing_L_seed = (
        (lbl_g == 2)
        | (lbl_g == 4)
        | (lbl_g == 8)
        | (lbl_g == 9)
        | (lbl_g == 10)
        | (lbl_g == 11)
        | (lbl_g == 13)
    )
    wing_L_m = (
        wing_L_seed
        | (
            ndi.binary_dilation(wing_L_seed, iterations=2)
            & a & (xx <= 58) & (yy <= 35)
            & (~head_upper_m) & (~tail_m) & (~wing_R_m)
            & (~(lbl_g == 12))
        )
    ) & (~head_upper_m) & (~tail_m) & (~wing_R_m)

    # 6. `leg_L` on back.png (left thigh/ankle + 3 left-sweeping talon arches at y>=47, x<=48):
    leg_L_zone = (
        ((yy >= 47) & (yy <= 48) & (xx >= 44) & (xx <= 49))
        | ((yy >= 49) & (yy <= 56) & (xx <= 48))
        | ((yy >= 57) & (yy <= 90) & (xx <= 48))
    )
    leg_L_m = a & leg_L_zone & (~wing_R_m)

    # 7. `leg_R` on back.png (right thigh/ankle + 2 right-sweeping talon arches at y>=49, x>=49):
    leg_R_zone = (
        ((yy >= 49) & (yy <= 52) & (xx >= 50) & (xx <= 56))
        | ((yy >= 53) & (yy <= 90) & (xx >= 49))
    )
    leg_R_m = a & leg_R_zone & (~leg_L_m)

    # 8. `torso`: neck, navy upper back armor, golden back, and golden pelvis
    used = lower_jaw_m | head_upper_m | tail_m | wing_L_m | wing_R_m | leg_L_m | leg_R_m
    torso_m = a & (~used)

    return {
        "wing_L": wing_L_m,
        "tail": tail_m,
        "leg_L": leg_L_m,
        "leg_R": leg_R_m,
        "torso": torso_m,
        "wing_R": wing_R_m,
        "lower_jaw": lower_jaw_m,
        "head_upper": head_upper_m,
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
    places onto the target `(cell_h, cell_w)` canvas with talons aligned to `foot_y`,
    and builds cushioned RGBA layers in strict back-to-front draw order.
    """
    raw_masks = segment_back_native(base_96) if is_back else segment_front_native(base_96)
    src_96 = base_96[:, ::-1, :].copy() if flip_x else base_96.copy()
    masks_96 = {k: (m[:, ::-1].copy() if flip_x else m.copy()) for k, m in raw_masks.items()}
    if flip_x:
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
    gold_fill_m = is_gold_armor(canvas)
    navy_fill_m = is_navy_armor(canvas)
    mouth_red_m = is_mouth_red(canvas)

    layers: Dict[str, np.ndarray] = {}

    if not is_back:
        # Front draw order: `wing_L` -> `tail` -> `wing_R` -> `far_leg` -> `near_leg` -> `torso` -> `lower_jaw` -> `head_upper`
        body_union = ndi.binary_erosion(
            c_masks["torso"] | c_masks["head_upper"] | c_masks["wing_R"] | c_masks["wing_L"] | c_masks["tail"],
            iterations=1,
        )
        fg_over_wing_L = body_union & (c_masks["torso"] | c_masks["head_upper"]) & (nat_y >= 20) & (nat_y <= 34)
        layers["wing_L"] = build_part_layer(
            canvas, c_masks["wing_L"], fg_over_wing_L, cushion_px=4, allowed_fill_mask=gold_fill_m
        )

        fg_over_tail = body_union & (c_masks["torso"] | c_masks["wing_R"]) & (nat_y >= 35) & (nat_y <= 44)
        layers["tail"] = build_part_layer(
            canvas, c_masks["tail"], fg_over_tail, cushion_px=4, allowed_fill_mask=navy_fill_m
        )

        fg_over_wing_R = body_union & c_masks["torso"] & (nat_y >= 29) & (nat_y <= 44)
        layers["wing_R"] = build_part_layer(
            canvas, c_masks["wing_R"], fg_over_wing_R, cushion_px=6, allowed_fill_mask=(navy_fill_m | gold_fill_m)
        )

        # `leg_L`, `leg_R` cushioned upward into `torso`'s thighs (`nat_y >= 44..50`) and across the lower-leg overlap
        leg_torso_union = ndi.binary_erosion(c_masks["torso"] | c_masks["leg_L"] | c_masks["leg_R"], iterations=1)
        near_lk = "leg_L" if flip_x else "leg_R"
        far_lk = "leg_R" if flip_x else "leg_L"

        fg_over_far_leg = leg_torso_union & (
            (c_masks["torso"] & (nat_y >= 44) & (nat_y <= 50))
            | (c_masks[near_lk] & (nat_y >= 50) & (nat_y <= 58))
        )
        layers[far_lk] = build_part_layer(
            canvas, c_masks[far_lk], fg_over_far_leg, cushion_px=6, allowed_fill_mask=gold_fill_m
        )

        fg_over_near_leg = leg_torso_union & (
            (c_masks["torso"] & (nat_y >= 44) & (nat_y <= 50))
            | (c_masks[far_lk] & (nat_y >= 50) & (nat_y <= 58))
        )
        layers[near_lk] = build_part_layer(
            canvas, c_masks[near_lk], fg_over_near_leg, cushion_px=6, allowed_fill_mask=gold_fill_m
        )

        # `torso` cushioned inside eroded neck/jaw boundary
        head_jaw_eroded = ndi.binary_erosion(c_masks["head_upper"] | c_masks["lower_jaw"] | c_masks["torso"], iterations=1)
        fg_over_torso = head_jaw_eroded & (c_masks["head_upper"] | c_masks["lower_jaw"]) & (nat_y >= 22) & (nat_y <= 27)
        layers["torso"] = build_part_layer(
            canvas, c_masks["torso"], fg_over_torso, cushion_px=5, allowed_fill_mask=(navy_fill_m | gold_fill_m)
        )

        # `lower_jaw` cushioned upward inside `head_upper` with dark-red mouth fill
        head_eroded = ndi.binary_erosion(c_masks["head_upper"] | c_masks["lower_jaw"], iterations=1) & c_masks["head_upper"]
        fg_over_jaw = head_eroded & (nat_y >= 18) & (nat_y <= 23)
        layers["lower_jaw"] = build_part_layer(
            canvas, c_masks["lower_jaw"], fg_over_jaw, cushion_px=3, allowed_fill_mask=(mouth_red_m | gold_fill_m)
        )

        # `head_upper`: 100% rigid foreground layer (zero distortion)
        layers["head_upper"] = build_part_layer(
            canvas, c_masks["head_upper"], np.zeros_like(c_masks["head_upper"]), cushion_px=0
        )

    else:
        # Back draw order: `wing_L` -> `tail` -> `leg_L` -> `leg_R` -> `torso` -> `wing_R` -> `lower_jaw` -> `head_upper`
        upper_union = ndi.binary_erosion(
            c_masks["wing_L"] | c_masks["wing_R"] | c_masks["torso"] | c_masks["tail"] | c_masks["head_upper"],
            iterations=1,
        )
        fg_over_wing_L = (
            upper_union
            & (c_masks["wing_R"] | c_masks["torso"] | c_masks["tail"])
            & (nat_y >= 17)
            & (nat_y <= 38)
        )
        layers["wing_L"] = build_part_layer(
            canvas, c_masks["wing_L"], fg_over_wing_L, cushion_px=6, allowed_fill_mask=gold_fill_m
        )

        fg_over_tail = (
            upper_union
            & (c_masks["wing_R"] | c_masks["torso"] | c_masks["wing_L"])
            & (nat_y >= 26)
            & (nat_y <= 42)
        )
        layers["tail"] = build_part_layer(
            canvas, c_masks["tail"], fg_over_tail, cushion_px=6, allowed_fill_mask=navy_fill_m
        )

        leg_union = ndi.binary_erosion(c_masks["torso"] | c_masks["leg_L"] | c_masks["leg_R"], iterations=1)
        for lk in ("leg_L", "leg_R"):
            other_lk = "leg_R" if lk == "leg_L" else "leg_L"
            fg_over_leg = leg_union & (
                (c_masks["torso"] & (nat_y >= 43) & (nat_y <= 54))
                | (c_masks[other_lk] & (nat_y >= 54) & (nat_y <= 62))
            )
            layers[lk] = build_part_layer(
                canvas, c_masks[lk], fg_over_leg, cushion_px=6, allowed_fill_mask=(gold_fill_m | navy_fill_m)
            )

        # `torso` cushioned underneath `wing_R` (`nat_y >= 27..45`) with `cushion_px=10` and `gold_fill_m` ONLY
        # so the golden back is 100% solid gold when `wing_R` flaps or slashes!
        fg_over_torso_wing = upper_union & c_masks["wing_R"] & (nat_y >= 27) & (nat_y <= 45)
        fg_over_torso_head = upper_union & c_masks["head_upper"] & (nat_y >= 18) & (nat_y <= 23)
        lyr_torso = build_part_layer(
            canvas, c_masks["torso"], fg_over_torso_wing, cushion_px=10, allowed_fill_mask=gold_fill_m
        )
        lyr_torso_neck = build_part_layer(
            canvas, c_masks["torso"], fg_over_torso_head, cushion_px=4, allowed_fill_mask=gold_fill_m
        )
        m_neck = lyr_torso_neck[..., 3] >= 128
        lyr_torso[m_neck] = lyr_torso_neck[m_neck]
        layers["torso"] = lyr_torso

        # `wing_R` cushioned ONLY downward into `torso`'s golden back (`nat_y=30..36`)
        # so `G5` keeps its 100% razor-sharp upper blade silhouette (`nat_y=11..29`) with zero lumps!
        fg_over_wing_R = upper_union & c_masks["torso"] & (nat_y >= 30) & (nat_y <= 36)
        layers["wing_R"] = build_part_layer(
            canvas, c_masks["wing_R"], fg_over_wing_R, cushion_px=4, allowed_fill_mask=gold_fill_m
        )

        # `lower_jaw` cushioned upward into `head_upper`
        head_eroded = ndi.binary_erosion(c_masks["head_upper"] | c_masks["lower_jaw"], iterations=1) & c_masks["head_upper"]
        fg_over_jaw = head_eroded & (nat_y >= 13) & (nat_y <= 18)
        layers["lower_jaw"] = build_part_layer(
            canvas, c_masks["lower_jaw"], fg_over_jaw, cushion_px=3, allowed_fill_mask=(mouth_red_m | gold_fill_m)
        )

        # `head_upper`: 100% rigid face/beak/eye/crest, with 3px navy feather extension behind G5 (`nat_y=11..16`)
        fg_over_head_crest = upper_union & c_masks["wing_R"] & (nat_y >= 11) & (nat_y <= 16)
        layers["head_upper"] = build_part_layer(
            canvas, c_masks["head_upper"], fg_over_head_crest, cushion_px=3, allowed_fill_mask=navy_fill_m
        )

    return layers


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
        (mid_x - 14 - drift, max(2, top_y + 10 - drift // 2)),
        (mid_x + 14 + drift, max(2, top_y + 12 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (242, 242, 242, 255)
    return out


def synthesize_skarmory_mega_frame(
    layers: Dict[str, np.ndarray],
    canvas_shape: Tuple[int, int, int],
    action: str,
    frame_idx: int,
    facing: float,
    is_back: bool,
    palette_rgba: np.ndarray,
    mouth_line: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    t = frame_idx / float(N_FRAMES)
    phase = 2.0 * math.pi * t
    sin1 = math.sin(phase)
    cos1 = math.cos(phase)

    torso_dx, torso_dy = 0, 0
    head_dx, head_dy = 0, 0
    jaw_dx, jaw_dy = 0, 0
    mouth_amt = 0
    wing_L_dx, wing_L_dy = 0, 0
    wing_R_dx, wing_R_dy = 0, 0
    tail_dx, tail_dy = 0, 0
    leg_L_dx, leg_L_dy = 0, 0
    leg_R_dx, leg_R_dy = 0, 0

    if action == "idle":
        # Modeled after Normal Skarmory's official GIF (`0227_skarmory/normal/front.gif` & `back.gif`)
        # and Mega Skarmory's hovering raptor anatomy (`avian_raptor_hover`):
        # 1. Hovering raptor breathing: `torso` and `head_upper` rise and dip smoothly (`±1..2px`),
        #    with `lower_jaw` opening `+1px` during the shriek/exhale phase.
        # 2. Dynamic metallic blade-wing stroke: `wing_L` and `wing_R` flap up and down (`±2px`)
        #    and sweep outward (`±1px`) in counter-phase to the body hover.
        # 3. Hooked `tail` and giant hanging golden talons (`leg_L`, `leg_R`) flex with natural
        #    aerodynamic phase lag while preserving 100% of the 3-claw talon geometry.
        torso_dy = int(round(1.2 * sin1))
        head_dy = int(round(1.5 * math.sin(phase - 0.25)))
        head_dx = int(round(0.8 * max(0.0, math.sin(phase - 0.2)) * facing))

        jaw_open = 1 if math.sin(phase - 0.1) > 0.15 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy
        mouth_amt = jaw_open

        flap_y = int(round(-2.0 * math.sin(phase + 0.35)))
        flap_x = int(round(-1.2 * math.cos(phase + 0.35) * facing))
        wing_R_dx = torso_dx + flap_x
        wing_R_dy = torso_dy + flap_y
        wing_L_dx = torso_dx + int(round(-0.8 * math.cos(phase + 0.5) * facing))
        wing_L_dy = torso_dy + int(round(-1.6 * math.sin(phase + 0.5)))

        tail_dx = torso_dx
        tail_dy = torso_dy + int(round(-0.8 * cos1))

        # Subtle hovering talon flex (never squashing or grounding the hanging 3-claw talons)
        leg_L_dx = int(round(0.8 * math.sin(phase - 0.4) * facing))
        leg_L_dy = int(round(0.8 * math.sin(phase - 0.3)))
        leg_R_dx = int(round(-0.8 * math.sin(phase - 0.4) * facing))
        leg_R_dy = int(round(0.8 * math.sin(phase - 0.6)))

    elif action == "walk":
        # `avian_raptor_hover`: Forward-leaning aerial glide & powerful blade-wing flap!
        # 2-beat body glide cadence, large blade-wing strokes (`±2..3px`), head thrust,
        # and alternating front/back swinging of the long golden talons (`leg_L` vs `leg_R`)!
        bounce = int(round(-1.5 * abs(sin1)))
        sway_x = int(round(1.4 * sin1 * facing))
        torso_dx, torso_dy = sway_x, bounce

        head_dx = sway_x + int(round(1.0 * cos1 * facing))
        head_dy = bounce + int(round(1.0 * math.sin(2.0 * phase - 0.3)))

        jaw_open = 1 if cos1 > 0.2 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy
        mouth_amt = jaw_open

        flap_y = int(round(-2.5 * sin1))
        flap_x = int(round(-1.5 * cos1 * facing))
        wing_R_dx = sway_x + flap_x
        wing_R_dy = bounce + flap_y
        wing_L_dx = sway_x + int(round(-1.2 * cos1 * facing))
        wing_L_dy = bounce + int(round(-2.0 * sin1))

        tail_dx = sway_x
        tail_dy = bounce + int(round(1.0 * cos1))

        # Alternating raptor talon pendulum swing (`leg_L` vs `leg_R`)
        if not is_back:
            leg_L_dx = sway_x + int(round(1.0 * cos1 * facing))
            leg_L_dy = bounce + int(round(-1.5 * max(0.0, sin1)))
            leg_R_dx = sway_x + int(round(-1.0 * cos1 * facing))
            leg_R_dy = bounce + int(round(-1.5 * max(0.0, -sin1)))
        else:
            # On back.png, keep relative lateral separation smoothly bounded so the ankle bridge stays seamless
            leg_L_dx = sway_x + int(round(1.2 * cos1 * facing))
            leg_L_dy = bounce + int(round(-1.2 * max(0.0, sin1)))
            leg_R_dx = sway_x + int(round(0.6 * cos1 * facing))
            leg_R_dy = bounce + int(round(-1.2 * max(0.0, -sin1)))

    elif action == "attack":
        # Signature Mega Skarmory "Brave Bird / Steel Wing / Drill Peck" High-Speed Dive Slash!
        # Frames 0..3: Soaring wind-up — rise up & back (`dx = -3px, dy = -2px`), flare both giant
        #              golden/navy blade wings high (`dy = -4px`), and open the toothed beak wide!
        # Frames 4..9: Explosive Dive Slash & Talon Strike — surge `+6px` forward (`head_dx = +6px`,
        #              `torso_dx = +5px`), slash both giant blade wings powerfully forward & down
        #              (`wing_R_dx = +6px, dy = +2px`), and rake the giant golden talons forward (`+5..6px`)!
        # Frames 10..15: Follow-through recovery to hovering stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.65 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        torso_dx = int(round(5.0 * k * facing))
        if frame_idx in (1, 2, 3):
            torso_dy = -2 if frame_idx in (2, 3) else -1
        elif frame_idx in (4, 5, 6, 7, 8):
            torso_dy = 1 if frame_idx in (5, 6, 7) else 0
        else:
            torso_dy = 0

        head_dx = int(round(6.0 * k * facing))
        head_dy = torso_dy + (1 if frame_idx in (5, 6, 7) else 0)

        jaw_open = (2 if frame_idx in (2, 3) else 1) if frame_idx in (1, 2, 3, 4, 5, 6) else 0
        jaw_dx = head_dx
        jaw_dy = head_dy
        mouth_amt = jaw_open

        # Blade wings (`wing_R`, `wing_L`): flare high on wind-up (0..3f), then slash forward & down (4..8f)!
        if frame_idx in (1, 2, 3):
            wing_R_dx = torso_dx + int(round(-1.5 * abs(k) * facing))
            wing_R_dy = torso_dy - (2 if frame_idx in (2, 3) else 1)
            wing_L_dx = torso_dx + int(round(-1.0 * abs(k) * facing))
            wing_L_dy = torso_dy - (2 if frame_idx in (2, 3) else 1)
        elif frame_idx in (4, 5, 6, 7, 8):
            wing_R_dx = int(round(6.2 * k * facing))
            wing_R_dy = torso_dy + (2 if frame_idx in (5, 6, 7) else 1)
            wing_L_dx = int(round(5.5 * k * facing))
            wing_L_dy = torso_dy + (1 if frame_idx in (5, 6, 7) else 0)
        else:
            wing_R_dx = torso_dx
            wing_R_dy = torso_dy
            wing_L_dx = torso_dx
            wing_L_dy = torso_dy

        tail_dx = torso_dx
        tail_dy = torso_dy + (-1 if frame_idx in (5, 6, 7) else 0)

        # Giant golden talons (`leg_L`, `leg_R`) rake forward into the strike!
        if not is_back:
            leg_L_dx = int(round(6.0 * k * facing))
            leg_L_dy = torso_dy - (1 if frame_idx in (4, 5, 6, 7) else 0)
            leg_R_dx = int(round(5.2 * k * facing))
            leg_R_dy = torso_dy
        else:
            leg_L_dx = int(round(5.5 * k * facing))
            leg_L_dy = torso_dy - (1 if frame_idx in (4, 5, 6, 7) else 0)
            leg_R_dx = int(round(5.5 * k * facing))
            leg_R_dy = torso_dy - (1 if frame_idx in (4, 5, 6, 7) else 0)

    elif action == "hurt":
        # Knockback + hit-shake + wings & head flung back + open shriek jaw + hurt sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        torso_dx = int(round(-3.8 * env * facing)) + jit
        torso_dy = int(round(-1.2 * env))

        head_dx = torso_dx + int(round(-1.0 * env * facing))
        head_dy = torso_dy + int(round(-1.0 * env))

        jaw_dx = head_dx
        jaw_dy = head_dy
        mouth_amt = 1 if (1 <= frame_idx <= 7) else 0

        wing_R_dx = torso_dx + int(round(-1.5 * env * facing))
        wing_R_dy = torso_dy + int(round(-1.8 * env))
        wing_L_dx = torso_dx + int(round(-1.2 * env * facing))
        wing_L_dy = torso_dy + int(round(-1.5 * env))

        tail_dx = torso_dx
        tail_dy = torso_dy + int(round(1.0 * env))

        leg_L_dx = int(round(torso_dx * 0.75))
        leg_L_dy = torso_dy
        leg_R_dx = int(round(torso_dx * 0.75))
        leg_R_dy = torso_dy

    if not is_back:
        far_leg_lyr = (
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy)
            if facing < 0
            else shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy)
        )
        near_leg_lyr = (
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy)
            if facing < 0
            else shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy)
        )
        comp = [
            shift_layer_int(layers["wing_L"], wing_L_dx, wing_L_dy),
            shift_layer_int(layers["tail"], tail_dx, tail_dy),
            shift_layer_int(layers["wing_R"], wing_R_dx, wing_R_dy),
            far_leg_lyr,
            near_leg_lyr,
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
        ]
    else:
        far_leg_lyr = (
            shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy)
            if facing > 0
            else shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy)
        )
        near_leg_lyr = (
            shift_layer_int(layers["leg_R"], leg_R_dx, leg_R_dy)
            if facing > 0
            else shift_layer_int(layers["leg_L"], leg_L_dx, leg_L_dy)
        )
        comp = [
            shift_layer_int(layers["wing_L"], wing_L_dx, wing_L_dy),
            shift_layer_int(layers["tail"], tail_dx, tail_dy),
            far_leg_lyr,
            near_leg_lyr,
            shift_layer_int(layers["torso"], torso_dx, torso_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(layers["wing_R"], wing_R_dx, wing_R_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    if mouth_line and mouth_amt > 0 and not is_back:
        line = mouth_open.transform_points(mouth_line, dx=head_dx, dy=head_dy)
        out = mouth_open.open_mouth(out, line, amount=mouth_amt, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=0.6)
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

    # Tight bounds of front (83x81) and back (81x86)
    max_w, max_h = 83, 86
    cell_w = max_w + 23  # 106 (generous horizontal margin for +6px Brave Bird / Steel Wing slash)
    cell_h = max_h + 18  # 104
    foot_y = cell_h - 4  # 100

    def mouth_line_cell(src_96: np.ndarray, flip_x: bool) -> List[Tuple[float, float]]:
        # same placement as build_rig_layers(): optional mirror, then tight-bounds -> centred, feet on foot_y
        a96 = (src_96[:, ::-1, 3] if flip_x else src_96[:, :, 3]) >= 128
        ys_, xs_ = np.where(a96)
        bx0, by0 = int(xs_.min()), int(ys_.min())
        bw, bh = int(xs_.max()) + 1 - bx0, int(ys_.max()) + 1 - by0
        pts = mouth_open.transform_points(
            MOUTH_LINE_FRONT_NATIVE, mirror_w=(src_96.shape[1] if flip_x else None)
        )
        return [(x - bx0 + (cell_w - bw) // 2, y - by0 + (foot_y - bh)) for x, y in pts]

    dir_layers: Dict[str, Tuple[Dict[str, np.ndarray], float, bool]] = {}
    dir_mouth: Dict[str, Optional[List[Tuple[float, float]]]] = {}
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
        dir_mouth[d_name] = None if use_back else mouth_line_cell(src_96, flip_x)

    sheets_by_action: Dict[str, np.ndarray] = {}
    master_rows: List[np.ndarray] = []

    for act in ACTIONS:
        act_sheet = np.zeros((8 * cell_h, N_FRAMES * cell_w, 4), dtype=np.uint8)
        for d_idx, d_name in enumerate(DIRS):
            layers, facing, use_back = dir_layers[d_name]
            for f_idx in range(N_FRAMES):
                frame_rgba = synthesize_skarmory_mega_frame(
                    layers,
                    (cell_h, cell_w, 4),
                    act,
                    f_idx,
                    facing,
                    use_back,
                    palette_rgba,
                    dir_mouth[d_name],
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
    # in `MEGA_DIR` to replace the 20-frame GIFs that had white clouds, cyan lasers, and chest scratch artifacts:
    idle_sheet = sheets_by_action["idle"]
    front_frames = [idle_sheet[0:cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    front_rev_frames = [f[:, ::-1, :].copy() for f in front_frames]
    back_frames = [idle_sheet[4 * cell_h:5 * cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    back_rev_frames = [f[:, ::-1, :].copy() for f in back_frames]

    # 2026-11-04 mouth fix: front/back png/gif are left untouched unless explicitly requested.
    if "--write-gifs" in sys.argv:
        save_transparent_gif(front_frames, MEGA_DIR / "front.gif", duration_ms=75)
        save_transparent_gif(front_rev_frames, MEGA_DIR / "front-reverse.gif", duration_ms=75)
        save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
        save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)

    meta = {
        "schema": "agent-sprite-forge.meta.v2",
        "key": "0227_skarmory-mega",
        "archetype": "avian_raptor_hover",
        "rig_parts": [
            "wing_L",
            "tail",
            "leg_L",
            "leg_R",
            "torso",
            "wing_R",
            "lower_jaw",
            "head_upper",
        ],
        "Idle": "../../monsterSprite/gen2/0227_skarmory/mega/skarmory-mega/agent-sprite-forge/idle.png",
        "Walk": "../../monsterSprite/gen2/0227_skarmory/mega/skarmory-mega/agent-sprite-forge/walk.png",
        "Attack": "../../monsterSprite/gen2/0227_skarmory/mega/skarmory-mega/agent-sprite-forge/attack.png",
        "Hurt": "../../monsterSprite/gen2/0227_skarmory/mega/skarmory-mega/agent-sprite-forge/hurt.png",
        "Sheet": "../../monsterSprite/gen2/0227_skarmory/mega/skarmory-mega/agent-sprite-forge/sheet.png",
        "Preview": "../../monsterSprite/gen2/0227_skarmory/mega/skarmory-mega/agent-sprite-forge/preview.png",
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
    print(f"Successfully forged Mega Skarmory 8-dir x 4-action bundle in {OUT_DIR} ({cell_w}x{cell_h} cell)")


if __name__ == "__main__":
    main()
