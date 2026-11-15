#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Feraligatr (`0160_feraligatr-mega`) modeled directly after Normal Feraligatr's
official GIF motion (`0160_feraligatr/normal/front.gif` & `back.gif`):
  - Exact color-component anatomical layer separation:
      1. `helmet` (giant red skull-like hood visor over the head)
      2. `head_upper` (blue upper snout, yellow eye, and upper white fangs - 100% rigid integer motion)
      3. `lower_jaw` (cream/tan lower jaw - opens and closes cleanly beneath the upper fangs!)
      4. `arm_far` (far-side clawed arm with 2 red arm-blades)
      5. `arm_near` (foreground clawed arm with 3 red gauntlet-blades, overlapping in front of torso)
      6. `back_spines` (red dorsal spines along the back)
      7. `tail` (heavy armored tail with red & grey tail-fins)
      8. `torso` (central blue body, cream belly plates, and thighs)
      9. `foot_L`, `foot_R` (grounded hind feet in `idle`, heavy bipedal stomp gait in `walk`)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import mouth_open  # noqa: E402  (small mouth opening along the face, 2026-11-04)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen2/0160_feraligatr/mega/feraligatr-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16

# Closed mouth line on the front still (S/SW/W, facing left), cell coords, ordered hinge/corner -> lip tip.
# Upper lip with the fang tips sits on y=54; the opening is carved into the cream lower jaw just below it.
MOUTH_LINE_FRONT = [(48, 54), (43, 54), (39, 54)]
MOUTH_DEEP = (16, 16, 16)   # outline ink (darkest palette colour)
MOUTH_INNER = (131, 0, 8)   # existing dark red -> mouth interior on the roar peak only


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
        | ((r == 131) & (g == 0) & (b == 8))
        | ((r == 0) & (g == 100) & (b == 162))
        | ((r == 106) & (g == 82) & (b == 0))
    )


def build_part_layer(
    src: np.ndarray,
    mask: np.ndarray,
    foreground_mask: np.ndarray,
    cushion_px: int = 3,
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
    Segments native 96x96 `front.png` (facing left) using exact connected components
    of its interior fill colors, then assigns 1px outlines to their owning anatomical part.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    r = arr96[:, :, 0].astype(np.int32)
    g = arr96[:, :, 1].astype(np.int32)
    b = arr96[:, :, 2].astype(np.int32)
    yy, xx = np.mgrid[0:h, 0:w]

    is_red = a & (
        ((r == 242) & (g == 82) & (b == 96))
        | ((r == 192) & (g == 41) & (b == 51))
        | ((r == 131) & (g == 0) & (b == 8))
        | ((r == 255) & (g == 139) & (b == 131))
    )
    is_blue_fill = a & (
        ((r == 28) & (g == 154) & (b == 210))
        | ((r == 51) & (g == 215) & (b == 255))
        | ((r == 145) & (g == 237) & (b == 255))
    )
    is_tan = a & (
        ((r == 255) & (g == 230) & (b == 139))
        | ((r == 197) & (g == 163) & (b == 109))
        | ((r == 106) & (g == 82) & (b == 0))
    )
    is_grey = a & (
        ((r == 21) & (g == 62) & (b == 94))
        | ((r == 65) & (g == 106) & (b == 138))
    )
    is_white = a & (((r == 255) & (g == 255) & (b == 255)) | ((r == 197) & (g == 197) & (b == 197)))

    lbl_red, _ = ndi.label(is_red)
    lbl_blue, _ = ndi.label(is_blue_fill)
    lbl_tan, _ = ndi.label(is_tan)
    lbl_grey, _ = ndi.label(is_grey)

    def pick_labels(lbl_arr: np.ndarray, predicate) -> np.ndarray:
        out_m = np.zeros_like(a)
        for cid in range(1, int(lbl_arr.max()) + 1):
            cm = lbl_arr == cid
            ys, xs = np.where(cm)
            if len(ys) > 0 and predicate(int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()), int(len(ys))):
                out_m |= cm
        return out_m

    # 1. Helmet seed: top red hood visor + grey eye-patches + white triangle on helmet
    helmet_seed = (
        pick_labels(lbl_red, lambda x0, x1, y0, y1, n: y0 <= 36 and x0 <= 56)
        | (is_grey & (yy <= 28))
        | (is_white & (yy <= 28))
    )

    # 2. Head Upper seed: Blue comps 1 & 2 (y0 <= 38, x0 <= 45) + yellow eye iris + white eye/fangs
    head_seed = (
        pick_labels(lbl_blue, lambda x0, x1, y0, y1, n: n >= 50 and y0 <= 38 and x0 <= 45)
        | (is_tan & (yy <= 43))
        | (is_white & (yy > 28) & (yy <= 53))
    )

    # 3. Lower Jaw seed: Cream lower jaw (Tan comp at x=30..50, y=44..56)
    jaw_seed = pick_labels(lbl_tan, lambda x0, x1, y0, y1, n: n >= 80 and x0 <= 32 and y0 >= 43)

    # 4. Arm Far (left extended arm): Blue comp at x=13..33, y=54..73 + Red blades at x<=26, y>=50 + Grey claws at x<=20, y>=68
    arm_far_seed = (
        pick_labels(lbl_blue, lambda x0, x1, y0, y1, n: x1 <= 34 and y0 >= 52)
        | pick_labels(lbl_red, lambda x0, x1, y0, y1, n: x1 <= 28 and y0 >= 48)
        | (is_grey & (xx <= 20) & (yy >= 68))
    )

    # 5. Arm Near (foreground right-to-left sweeping arm with 3 red gauntlet blades + all 3 hand claws):
    #    Blue comp at x=35..63, y=46..83 (n=200) + Red blades (x0 in 35..60, y0 >= 41) + Shoulder pad & all 3 hand claws (x=31..42, y=73..85)
    arm_near_blue = (
        pick_labels(lbl_blue, lambda x0, x1, y0, y1, n: n >= 150 and 34 <= x0 <= 38 and y1 >= 80)
        | (is_blue_fill & (xx >= 31) & (xx <= 40) & (yy >= 73))
    )
    arm_near_red = pick_labels(lbl_red, lambda x0, x1, y0, y1, n: 35 <= x0 <= 61 and y0 >= 41)
    arm_near_grey = (
        pick_labels(lbl_grey, lambda x0, x1, y0, y1, n: 56 <= x0 <= 62 and y1 <= 48)  # shoulder pad
        | (is_grey & (xx >= 31) & (xx <= 42) & (yy >= 73))  # all 3 foreground claws of arm_near's hand!
    )
    arm_near_seed = arm_near_blue | arm_near_red | arm_near_grey

    # 6. Back Spines: Red dorsal fins on upper right back (x0 >= 62, y0 in 37..45)
    spines_seed = pick_labels(lbl_red, lambda x0, x1, y0, y1, n: x0 >= 62 and 37 <= y0 <= 45)

    # 7. Tail: Rightmost blue tail (x0 >= 79, y0 >= 52) + grey tail fin (x0 >= 79) + red tail tip (x0 >= 83)
    tail_seed = (
        pick_labels(lbl_blue, lambda x0, x1, y0, y1, n: x0 >= 79 and y0 >= 52)
        | (is_grey & (xx >= 79) & (yy >= 52))
        | pick_labels(lbl_red, lambda x0, x1, y0, y1, n: x0 >= 83 and y0 >= 50)
    )

    # 8. Foot R & Foot L (grounded hind feet):
    foot_R_seed = (a & (xx >= 60) & (yy >= 79)) & (~arm_near_seed)
    foot_L_seed = (
        is_blue_fill & (xx >= 48) & (xx <= 56) & (yy >= 73) & (yy <= 79) & (~arm_near_seed)
    )

    # 9. Torso: Remaining interior pixels
    assigned_seeds = (
        helmet_seed | head_seed | jaw_seed | arm_far_seed | arm_near_seed
        | spines_seed | tail_seed | foot_R_seed | foot_L_seed
    )
    torso_seed = a & (~assigned_seeds) & (~is_dark_outline(arr96))

    # Now assign all dark outline pixels (`(16,16,16)`, `(0,100,162)`, etc.) to their nearest seed,
    # with priority so foreground parts (`helmet`, `head_upper`, `arm_near`, `arm_far`) keep complete 1px outlines!
    def grow_outline(seed_m: np.ndarray, allowed_zone: np.ndarray, iters: int = 1) -> np.ndarray:
        dil = ndi.binary_dilation(seed_m, iterations=iters)
        return seed_m | (dil & a & allowed_zone & is_dark_outline(arr96))

    helmet_m = grow_outline(helmet_seed, yy <= 43)
    # Include the entire bottom hand/claw tip of arm_near (x=31..49, y=72..86) so zero claw pixels get left behind
    arm_near_m = grow_outline(arm_near_seed, (~helmet_m), iters=1) | (a & (xx >= 31) & (xx <= 49) & (yy >= 72))
    head_m = grow_outline(head_seed, (yy >= 31) & (yy <= 54) & (~helmet_m) & (~arm_near_m))
    jaw_m = grow_outline(jaw_seed, (yy >= 43) & (yy <= 58) & (~helmet_m) & (~head_m) & (~arm_near_m))
    arm_far_m = grow_outline(arm_far_seed, (~helmet_m) & (~head_m) & (~jaw_m) & (~arm_near_m)) | (a & (xx <= 26) & (yy >= 50))
    spines_m = grow_outline(spines_seed, (~helmet_m) & (~arm_near_m))
    tail_m = grow_outline(tail_seed, (xx >= 78) & (~spines_m))
    foot_R_m = grow_outline(foot_R_seed, (xx >= 60) & (yy >= 78) & (~arm_near_m))
    foot_L_m = grow_outline(foot_L_seed, (xx >= 48) & (xx <= 57) & (yy >= 72) & (~arm_near_m))

    used = helmet_m | head_m | jaw_m | arm_near_m | arm_far_m | spines_m | tail_m | foot_R_m | foot_L_m
    torso_m = a & (~used)

    return {
        "arm_far": arm_far_m,
        "back_spines": spines_m,
        "tail": tail_m,
        "foot_L": foot_L_m,
        "foot_R": foot_R_m,
        "torso": torso_m,
        "lower_jaw": jaw_m,
        "head_upper": head_m,
        "helmet": helmet_m,
        "arm_near": arm_near_m,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` (facing right) into matching anatomical layers.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    r = arr96[:, :, 0].astype(np.int32)
    g = arr96[:, :, 1].astype(np.int32)
    b = arr96[:, :, 2].astype(np.int32)
    yy, xx = np.mgrid[0:h, 0:w]

    is_red = a & (
        ((r == 242) & (g == 82) & (b == 96))
        | ((r == 192) & (g == 41) & (b == 51))
        | ((r == 131) & (g == 0) & (b == 8))
        | ((r == 255) & (g == 139) & (b == 131))
    )
    is_blue_fill = a & (
        ((r == 28) & (g == 154) & (b == 210))
        | ((r == 51) & (g == 215) & (b == 255))
    )
    is_grey = a & (
        ((r == 21) & (g == 62) & (b == 94))
        | ((r == 65) & (g == 106) & (b == 138))
    )
    is_white = a & ((r == 255) & (g == 255) & (b == 255))

    lbl_red, _ = ndi.label(is_red)
    lbl_blue, _ = ndi.label(is_blue_fill)

    def pick_labels(lbl_arr: np.ndarray, predicate) -> np.ndarray:
        out_m = np.zeros_like(a)
        for cid in range(1, int(lbl_arr.max()) + 1):
            cm = lbl_arr == cid
            ys, xs = np.where(cm)
            if len(ys) > 0 and predicate(int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()), int(len(ys))):
                out_m |= cm
        return out_m

    # 1. Helmet on back.png: Upper red hood (y <= 41, plus visor beak on right x >= 53, y <= 44)
    helmet_seed = (
        (is_red & ((yy <= 40) | ((xx >= 52) & (yy <= 44))))
        | (is_grey & (yy <= 33))
        | (is_white & (yy <= 33))
    )

    # 2. Back Dorsal Spines: Red spines on mid-back (y > 40, x <= 54, excluding tiny tail red at x <= 12, y >= 53)
    spines_seed = (is_red & (~helmet_seed) & (xx >= 13) & (xx <= 54) & (yy <= 58))

    # 3. Right Extended Arm (`arm_near`): Blue comp at x=48..87, y=48..80 + 3 red arm blades (x >= 56, y >= 46) + claws
    arm_near_blue = pick_labels(lbl_blue, lambda x0, x1, y0, y1, n: n >= 200 and x0 >= 46)
    arm_near_red = is_red & (xx >= 56) & (yy >= 46)
    arm_near_claws = is_grey & (xx >= 78) & (yy >= 76)
    arm_near_seed = arm_near_blue | arm_near_red | arm_near_claws

    # 4. Tail on back.png: Left sweeping tail (x <= 24, y >= 50) + red & grey tail fins
    tail_seed = (
        (a & (xx <= 23) & (yy >= 50) & (yy <= 78) & (~spines_seed))
        | (is_red & (xx <= 13) & (yy >= 52))
        | (is_grey & (xx <= 25) & (yy >= 52) & (yy <= 68))
    )

    # 5. Hind Feet (`foot_L` on left at x=16..30, y>=81; `foot_R` on right at x=44..67, y>=81)
    foot_L_seed = a & (xx >= 16) & (xx <= 30) & (yy >= 81)
    foot_R_seed = a & (xx >= 44) & (xx <= 68) & (yy >= 81)

    def grow_outline(seed_m: np.ndarray, allowed_zone: np.ndarray) -> np.ndarray:
        dil = ndi.binary_dilation(seed_m, iterations=1)
        return seed_m | (dil & a & allowed_zone & is_dark_outline(arr96))

    helmet_m = grow_outline(helmet_seed, yy <= 45)
    arm_near_m = grow_outline(arm_near_seed, (~helmet_m)) | (a & (xx >= 75) & (yy >= 68))
    spines_m = grow_outline(spines_seed, (~helmet_m) & (~arm_near_m))
    tail_m = grow_outline(tail_seed, (xx <= 25) & (~spines_m) & (yy <= 78))
    foot_L_m = grow_outline(foot_L_seed, (xx <= 31) & (yy >= 80))
    foot_R_m = grow_outline(foot_R_seed, (xx >= 43) & (xx <= 69) & (yy >= 80) & (~arm_near_m))

    used = helmet_m | arm_near_m | spines_m | tail_m | foot_L_m | foot_R_m
    torso_m = a & (~used)
    empty = np.zeros_like(a)

    return {
        "arm_far": empty,
        "back_spines": spines_m,
        "tail": tail_m,
        "foot_L": foot_L_m,
        "foot_R": foot_R_m,
        "torso": torso_m,
        "lower_jaw": empty,
        "head_upper": empty,
        "helmet": helmet_m,
        "arm_near": arm_near_m,
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
    Segments native `front.png` or `back.png`, optionally flips horizontally for E/SE/NW,
    places onto the target `(cell_h, cell_w)` canvas with feet aligned to `foot_y`,
    and builds cushioned RGBA layers in back-to-front draw order.
    """
    raw_masks = segment_back_native(base_96) if is_back else segment_front_native(base_96)
    src_96 = base_96[:, ::-1, :].copy() if flip_x else base_96.copy()
    masks_96 = {k: (m[:, ::-1].copy() if flip_x else m.copy()) for k, m in raw_masks.items()}

    # Compute tight bounding box of the whole sprite to position on `(cell_h, cell_w)`
    a96 = src_96[:, :, 3] >= 128
    ys, xs = np.where(a96)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x0 = (cell_w - sw) // 2
    dst_y0 = foot_y - sh

    canvas = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
    canvas[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = src_96[y0:y1, x0:x1]
    mirror_w = sw + 2 * dst_x0  # x -> mirror_w - 1 - x maps the unflipped placement onto the flipped one

    c_masks: Dict[str, np.ndarray] = {}
    for k, m96 in masks_96.items():
        cm = np.zeros((cell_h, cell_w), dtype=bool)
        cm[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = m96[y0:y1, x0:x1]
        c_masks[k] = cm

    # Back-to-front draw order:
    # 1. `arm_far` (far-side arm behind snout/body)
    # 2. `back_spines` (dorsal spines on back)
    # 3. `tail` (armored tail)
    # 4. `foot_L`, `foot_R` (hind feet)
    # 5. `torso` (main blue body & belly)
    # 6. `lower_jaw` (cream lower jaw behind upper fangs)
    # 7. `head_upper` (blue upper snout, eye, and upper white fangs)
    # 8. `helmet` (giant red hood visor over the head)
    # 9. `arm_near` (foreground clawed arm with 3 red gauntlet blades)
    order = [
        "arm_far",
        "back_spines",
        "tail",
        "foot_L",
        "foot_R",
        "torso",
        "lower_jaw",
        "head_upper",
        "helmet",
        "arm_near",
    ]

    layers: Dict[str, np.ndarray] = {}
    # Torso gets an 8px interior cushion ONLY under layers drawn ON TOP OF torso (`arm_near`, `lower_jaw`, `head_upper`, `helmet`),
    # never over `back_spines` or `tail` (which are drawn behind torso)!
    fg_over_torso = (
        c_masks["arm_near"]
        | c_masks["lower_jaw"]
        | c_masks["head_upper"]
        | c_masks["helmet"]
    )
    layers["torso"] = build_part_layer(canvas, c_masks["torso"], fg_over_torso, cushion_px=8)

    # Lower jaw gets a 6px upward/rightward cream cushion behind `head_upper`, `arm_near`, and `torso`
    layers["lower_jaw"] = build_part_layer(canvas, c_masks["lower_jaw"], c_masks["head_upper"] | c_masks["arm_near"] | c_masks["torso"], cushion_px=6)

    # Head upper gets an 8px blue cushion behind `helmet` and `arm_near` so shoulder/visor motion never leaves a gap!
    layers["head_upper"] = build_part_layer(canvas, c_masks["head_upper"], c_masks["helmet"] | c_masks["arm_near"], cushion_px=8)

    # Back spines and tail get 5px cushion under torso/helmet/arm_near
    layers["back_spines"] = build_part_layer(canvas, c_masks["back_spines"], c_masks["torso"] | c_masks["helmet"] | c_masks["arm_near"], cushion_px=5)
    layers["tail"] = build_part_layer(canvas, c_masks["tail"], c_masks["torso"], cushion_px=5)

    # Arm far gets 3px cushion under head/torso/jaw
    layers["arm_far"] = build_part_layer(canvas, c_masks["arm_far"], c_masks["torso"] | c_masks["lower_jaw"] | c_masks["head_upper"], cushion_px=3)

    # Foreground layers (`helmet`, `arm_near`) need zero outward cushion
    layers["helmet"] = build_part_layer(canvas, c_masks["helmet"], np.zeros_like(c_masks["helmet"]), cushion_px=0)
    layers["arm_near"] = build_part_layer(canvas, c_masks["arm_near"], np.zeros_like(c_masks["arm_near"]), cushion_px=0)

    # Hind feet (`foot_L`, `foot_R`): purely vertical 4px upward stem under `torso`
    for fk in ("foot_L", "foot_R"):
        fm = c_masks[fk]
        lyr = np.zeros_like(canvas)
        lyr[fm] = canvas[fm]
        f_ys, f_xs = np.where(fm)
        if len(f_ys) > 0:
            for x_col in np.unique(f_xs):
                col_ys = f_ys[f_xs == x_col]
                cy_min = int(col_ys.min())
                fill_px = canvas[cy_min, x_col]
                for dy_up in range(1, 5):
                    py = cy_min - dy_up
                    if py >= 0 and c_masks["torso"][py, x_col]:
                        lyr[py, x_col] = canvas[py, x_col] if not is_dark_outline(canvas[py:py + 1, x_col:x_col + 1])[0, 0] else fill_px
        layers[fk] = lyr

    # Mouth line for this view (front views only; back views show no mouth)
    if is_back:
        layers["_mouth_line"] = None
    else:
        layers["_mouth_line"] = mouth_open.transform_points(MOUTH_LINE_FRONT, mirror_w=mirror_w if flip_x else None)
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
        (mid_x - 12 - drift, max(2, top_y + 2 - drift // 2)),
        (mid_x + 12 + drift, max(2, top_y + 3 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (255, 255, 255, 255)
    return out


def synthesize_feraligatr_mega_frame(
    layers: Dict[str, np.ndarray],
    canvas_shape: Tuple[int, int, int],
    action: str,
    frame_idx: int,
    facing: float,
    palette_rgba: np.ndarray,
) -> np.ndarray:
    t = frame_idx / float(N_FRAMES)
    phase = 2.0 * math.pi * t
    sin1 = math.sin(phase)
    cos1 = math.cos(phase)

    torso_dx, torso_dy = 0, 0
    head_dx, head_dy = 0, 0
    helm_dx, helm_dy = 0, 0
    jaw_dx, jaw_dy = 0, 0
    arm_near_dx, arm_near_dy = 0, 0
    arm_far_dx, arm_far_dy = 0, 0
    mouth_amt = 0
    spines_dx, spines_dy = 0, 0
    tail_dx, tail_dy = 0, 0
    foot_L_dx, foot_L_dy = 0, 0
    foot_R_dx, foot_R_dy = 0, 0

    if action == "idle":
        # Modeled directly after Normal Feraligatr's GIF (`0160_feraligatr/normal/front.gif`):
        # 1. Hind feet (`foot_L`, `foot_R`) stay 100% planted on the ground `(0, 0)`.
        # 2. Reptilian breathing crouch: Torso & Head bob (`-1..+1px` / `-1..+2px`),
        #    Giant Red Helmet Visor (`helmet`) nods with a slight follow-through,
        #    Lower Jaw (`lower_jaw`) opens `+1px` during inhalation/growl,
        #    Both red-bladed arms (`arm_near`, `arm_far`) pump up/down and forward/back (`±1..2px`),
        #    and the armored tail (`tail`) sways up/down with phase delay!
        torso_dy = int(round(1.0 * sin1))
        head_dy = int(round(1.4 * math.sin(phase - 0.2)))
        head_dx = int(round(0.8 * max(0.0, math.sin(phase - 0.2)) * facing))

        # Helmet visor follows head with a 1px heavy visor nod at the bottom of the breath
        helm_dx = head_dx
        helm_dy = head_dy + (1 if math.sin(phase - 0.5) > 0.65 else 0)

        # Lower jaw opens by +1px relative to head_upper during breath peak
        # (2026-11-04) jaw stays on the head; the growl is a small 1px opening along the mouth line
        mouth_amt = 1 if math.sin(phase - 0.1) > 0.25 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy

        # Clawed arms pump rhythmically (like Normal Feraligatr's GIF!)
        arm_near_dx = int(round(1.4 * math.sin(phase - 0.6) * facing))
        arm_near_dy = torso_dy + int(round(1.6 * math.sin(phase - 0.5)))
        arm_far_dx = int(round(1.3 * math.sin(phase - 0.9) * facing))
        arm_far_dy = torso_dy + int(round(1.5 * math.sin(phase - 0.8)))

        spines_dx = torso_dx
        spines_dy = torso_dy + int(round(0.8 * math.sin(phase - 0.4)))

        tail_dx = torso_dx
        tail_dy = int(round(1.6 * math.sin(phase - 1.1)))

    elif action == "walk":
        # Heavy bipedal alligator stomp gait:
        # Alternating left/right hind foot stomp (`foot_L`, `foot_R`),
        # 2-beat body bounce, counter-swinging clawed arms (`arm_near` vs `arm_far`),
        # jaw champing on each step, and heavy tail counterbalance!
        bounce = int(round(-1.8 * abs(sin1)))
        sway_x = int(round(1.2 * sin1 * facing))
        torso_dx, torso_dy = sway_x, bounce

        head_dx = sway_x + int(round(1.0 * cos1 * facing))
        head_dy = bounce + int(round(1.2 * math.sin(2.0 * phase - 0.3)))
        helm_dx = head_dx
        helm_dy = head_dy + (1 if math.sin(2.0 * phase - 0.6) > 0.5 else 0)

        mouth_amt = 1 if math.sin(2.0 * phase) > 0.2 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy

        # Alternating arm swing opposite to hind legs
        arm_near_dx = sway_x + int(round(2.0 * sin1 * facing))
        arm_near_dy = bounce + int(round(1.5 * cos1))
        arm_far_dx = sway_x + int(round(-2.0 * sin1 * facing))
        arm_far_dy = bounce + int(round(-1.5 * cos1))

        spines_dx, spines_dy = sway_x, bounce + int(round(1.0 * math.sin(2.0 * phase - 0.5)))
        tail_dx = sway_x
        tail_dy = bounce + int(round(1.8 * math.sin(2.0 * phase - 1.0)))

        foot_L_dx, foot_L_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        foot_R_dx, foot_R_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Mega Feraligatr "Strong Jaw / Liquidation / Crunch" Attack!
        # Frames 0..3: Wind-up (pull back, helmet visor raises `-1px`, lower jaw drops wide `+2px`, arms cock back)
        # Frames 4..9: Ferocious forward lunge (`+5px` forward) + GIANT RED HELMET VISOR SLAMS DOWN (`+2px`)
        #              to meet the chomping jaw + BOTH RED-BLADED ARMS SLASH FORWARD (`+6px`) + tail whips up!
        # Frames 10..15: Recovery to idle stance.
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
        torso_dy = int(round(1.0 * abs(k)))

        head_dx = int(round(4.8 * k * facing))
        head_dy = int(round(1.5 * k)) if k > 0 else int(round(-1.0 * abs(k)))

        # During wind-up (k < 0) and early lunge (frames 3..5), jaw opens wide (+2px) and helmet lifts (-1px);
        # on crunch impact (frames 6..8), helmet slams down (+1..2px) and jaw snaps shut (0px)!
        if frame_idx in (2, 3, 4, 5):
            helm_dx = head_dx
            helm_dy = head_dy - 1
            jaw_dx = head_dx
            jaw_dy = head_dy
            mouth_amt = 2 if frame_idx in (3, 4) else 1
        elif frame_idx in (6, 7, 8):
            helm_dx = head_dx + int(round(1.0 * facing))
            helm_dy = head_dy + 1
            jaw_dx = head_dx
            jaw_dy = head_dy
        else:
            helm_dx, helm_dy = head_dx, head_dy
            jaw_dx, jaw_dy = head_dx, head_dy

        # Both red-gauntlet arms slash powerfully forward on strike frames!
        arm_near_dx = int(round(6.0 * k * facing))
        arm_near_dy = int(round(-2.0 * k)) if frame_idx in (4, 5) else int(round(2.0 * max(0.0, k)))
        arm_far_dx = int(round(5.5 * k * facing))
        arm_far_dy = int(round(1.8 * k))

        spines_dx = int(round(3.8 * k * facing))
        spines_dy = torso_dy + int(round(-1.0 * max(0.0, k)))

        tail_dx = int(round(2.5 * k * facing))
        tail_dy = int(round(-2.2 * k))

        # Front foot steps into the lunge
        step_x = int(round(2.0 * max(0.0, k) * facing))
        if facing < 0:
            foot_L_dx, foot_L_dy = step_x, 0
        else:
            foot_R_dx, foot_R_dy = step_x, 0

    elif action == "hurt":
        # Knockback + hit-shake + jaw jarred open (+1px) + arms flung back
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        torso_dx = int(round(-3.5 * env * facing)) + jit
        torso_dy = int(round(-1.0 * env))

        head_dx = torso_dx + int(round(-1.0 * env * facing))
        head_dy = int(round(-1.5 * env))
        helm_dx = head_dx
        helm_dy = head_dy - (1 if 2 <= frame_idx <= 6 else 0)
        jaw_dx = head_dx
        jaw_dy = head_dy
        mouth_amt = 1 if 1 <= frame_idx <= 7 else 0

        arm_near_dx = torso_dx + int(round(-1.5 * env * facing))
        arm_near_dy = int(round(-2.0 * env))
        arm_far_dx = torso_dx + int(round(-1.2 * env * facing))
        arm_far_dy = int(round(-1.5 * env))

        spines_dx, spines_dy = torso_dx, torso_dy
        tail_dx, tail_dy = torso_dx, int(round(1.2 * env))

        slide_x = int(round(torso_dx * 0.6))
        foot_L_dx, foot_R_dx = slide_x, slide_x

    comp = [
        shift_layer_int(layers["arm_far"], arm_far_dx, arm_far_dy),
        shift_layer_int(layers["back_spines"], spines_dx, spines_dy),
        shift_layer_int(layers["tail"], tail_dx, tail_dy),
        shift_layer_int(layers["foot_L"], foot_L_dx, foot_L_dy),
        shift_layer_int(layers["foot_R"], foot_R_dx, foot_R_dy),
        shift_layer_int(layers["torso"], torso_dx, torso_dy),
        shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
        shift_layer_int(layers["head_upper"], head_dx, head_dy),
        shift_layer_int(layers["helmet"], helm_dx, helm_dy),
        shift_layer_int(layers["arm_near"], arm_near_dx, arm_near_dy),
    ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    line = layers.get("_mouth_line")
    if mouth_amt > 0 and line:
        opened = mouth_open.open_mouth(
            out, mouth_open.transform_points(line, dx=head_dx, dy=head_dy),
            amount=mouth_amt, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=0.6,
        )
        # never carve through the foreground arm if it swings over the face
        arm_on_top = comp[-1][..., 3] >= 128
        opened[arm_on_top] = out[arm_on_top]
        out = opened
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

    # Tight bounds of front (90x78) and back (86x77)
    max_w, max_h = 90, 78
    cell_w = max_w + 20  # 110
    cell_h = max_h + 16  # 94
    foot_y = cell_h - 4  # 90

    dir_layers: Dict[str, Tuple[Dict[str, np.ndarray], float]] = {}
    for d_name in DIRS:
        use_back = d_name in ("N", "NE", "NW")
        src_96 = back_raw if use_back else front_raw
        if not use_back:
            flip_x = d_name in ("SE", "E")
            facing = 1.0 if flip_x else -1.0
        else:
            flip_x = d_name in ("NW",)
            facing = -1.0 if flip_x else 1.0

        layers = build_rig_layers(src_96, is_back=use_back, flip_x=flip_x, cell_w=cell_w, cell_h=cell_h, foot_y=foot_y)
        dir_layers[d_name] = (layers, facing)

    sheets_by_action: Dict[str, np.ndarray] = {}
    master_rows: List[np.ndarray] = []

    for act in ACTIONS:
        act_sheet = np.zeros((8 * cell_h, N_FRAMES * cell_w, 4), dtype=np.uint8)
        for d_idx, d_name in enumerate(DIRS):
            layers, facing = dir_layers[d_name]
            for f_idx in range(N_FRAMES):
                frame_rgba = synthesize_feraligatr_mega_frame(
                    layers,
                    (cell_h, cell_w, 4),
                    act,
                    f_idx,
                    facing,
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

    # Also export clean 16-frame animated GIFs (`front.gif`, `front-reverse.gif`, `back.gif`, `back-reverse.gif`)
    # in `MEGA_DIR` to replace the static 19-frame GIFs that had blue laser artifacts across the face:
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
        "generator": "forge_feraligatr_mega_actions",
        "key": "0160_feraligatr-mega",
        "dex": 160,
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
    print(f"Built 0160_feraligatr-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
