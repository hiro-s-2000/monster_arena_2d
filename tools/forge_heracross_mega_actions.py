#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Heracross (`0214_heracross-mega`) modeled directly after Normal Heracross's
official GIF motion (`0214_heracross/normal/front.gif` & `back.gif`):
  - Exact color & connected-component anatomical layer separation (7 independent rig layers):
      1. `arm_far` (far-side giant red-ringed spherical gauntlet & 3 white claws)
      2. `foot_L`, `foot_R` (full muscular thighs, knee spikes, ankles, and white-clawed feet —
                             100% grounded `(0, 0)` in `idle` while the upper body flexes at the hips,
                             heavy alternating bipedal march from the hip sockets in `walk`,
                             and deep power-step lunge in `attack`)
      3. `torso` (armored blue thorax, red collar/chest/hip sockets, segmented abdomen,
                  golden-yellow dorsal elytra shell, and spherical shoulder pauldrons)
      4. `head` (red forehead crest, fierce yellow eyes, and blue front nose-horn — 100% rigid
                 integer translation so the face and eyes never deform by a single pixel)
      5. `horn_crest` (massive Hercules dorsal horn & clubbed left/right antennae)
      6. `arm_near` (foreground upper arm cylinder + giant red-ringed spherical gauntlet & white claws —
                     rhythmic fighter guard in `idle`, counter-swing in `walk`, and explosive +6..7px
                     Megahorn / Close Combat / Pin Missile rush thrust in `attack`)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen2/0214_heracross/mega/heracross-mega"
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
        | ((r == 32) & (g == 56) & (b == 99))
        | ((r == 106) & (g == 90) & (b == 32))
    )


def is_blue_armor(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 66) & (g == 137) & (b == 178))
        | ((r == 31) & (g == 105) & (b == 145))
        | ((r == 108) & (g == 175) & (b == 209))
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
    Segments native 96x96 `front.png` (facing left) into 7 exact anatomical masks:
      - `arm_far`: Left extended giant red-ringed spherical gauntlet & 3 white claws (`x=3..36, y=53..76`).
      - `foot_L`: Full left leg (thigh, knee spike, ankle, and 3 white claws) outside the red hip socket (`x=29..42, y=65..80`).
      - `foot_R`: Full right leg (thigh, ankle, and white claws) outside the red hip socket (`x=58..70, y=66..84`).
      - `torso`: Armored blue chest/abdomen, red collar/hip sockets, yellow dorsal elytra tip, and spherical shoulder pauldrons.
      - `head`: Red forehead crest, yellow eyes, and blue front snout horn (`x=29..52, y=47..61`).
      - `horn_crest`: Giant Hercules horn and left/right clubbed antennae (`x=29..61, y=10..51`).
      - `arm_near`: Right upper arm cylinder + giant red-ringed spherical gauntlet & 3 white claws (`x=62..91, y=52..81`).
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    r = arr96[:, :, 0].astype(np.int32)
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Right antenna zone running diagonally from (x=45, y=51) to (x=61, y=35):
    r_ant_zone = (
        ((yy >= 31) & (yy <= 38) & (xx >= 55) & (xx <= 62))
        | ((yy == 39) & (xx >= 54) & (xx <= 58))
        | ((yy == 40) & (xx >= 53) & (xx <= 57))
        | ((yy == 41) & (xx >= 53) & (xx <= 56))
        | ((yy == 42) & (xx >= 52) & (xx <= 55))
        | ((yy == 43) & (xx >= 51) & (xx <= 54))
        | ((yy == 44) & (xx >= 50) & (xx <= 53))
        | ((yy == 45) & (xx >= 50) & (xx <= 52))
        | ((yy == 46) & (xx >= 49) & (xx <= 52))
        | ((yy == 47) & (xx >= 48) & (xx <= 51))
        | ((yy == 48) & (xx >= 47) & (xx <= 50))
        | ((yy == 49) & (xx >= 46) & (xx <= 49))
        | ((yy == 50) & (xx >= 45) & (xx <= 48))
        | ((yy == 51) & (xx >= 45) & (xx <= 47))
    )
    is_red_forehead = a & ((r == 190) | (r == 248)) & (xx >= 36) & (xx <= 49) & (yy >= 47) & (yy <= 55)

    horn_crest_m = a & (
        ((yy <= 46) & (~((yy >= 42) & (xx >= 53) & (~r_ant_zone))))
        | r_ant_zone
        | ((yy >= 47) & (yy <= 49) & (xx <= 37))
        | ((yy == 47) & (xx >= 38) & (xx <= 50) & (~is_red_forehead) & (r != 16))
    )

    # 2. Head: Red forehead crest + yellow eyes + blue front nose-horn:
    head_zone = (
        ((yy >= 47) & (yy <= 51) & (xx >= 34) & (xx <= 51))
        | ((yy >= 52) & (yy <= 53) & (xx >= 33) & (xx <= 51))
        | ((yy == 54) & (xx >= 30) & (xx <= 51))
        | ((yy >= 55) & (yy <= 56) & (xx >= 29) & (xx <= 51))
        | ((yy >= 57) & (yy <= 59) & (xx >= 30) & (xx <= 51))
        | ((yy == 60) & (xx >= 33) & (xx <= 47))
        | ((yy == 61) & (xx >= 37) & (xx <= 44))
    )
    head_m = a & head_zone & (~horn_crest_m)

    # 3. Arm Far (left extended spherical gauntlet + 3 white claws):
    arm_far_zone = (
        ((yy >= 53) & (yy <= 60) & (xx <= 32) & (~head_m) & (~horn_crest_m))
        | ((yy == 61) & (xx <= 36) & (~head_m))
        | ((yy >= 62) & (yy <= 76) & (xx <= 32))
    )
    arm_far_m = a & arm_far_zone

    # 4. Arm Near (right upper arm cylinder + giant spherical gauntlet + 3 white claws):
    arm_near_zone = (
        ((yy == 52) & (xx >= 63))
        | ((yy >= 53) & (yy <= 55) & (xx >= 62))
        | ((yy == 56) & (xx >= 63))
        | ((yy == 57) & (xx >= 65))
        | ((yy >= 58) & (yy <= 61) & (xx >= 68))
        | ((yy >= 62) & (yy <= 69) & (xx >= 66))
        | ((yy >= 70) & (yy <= 72) & (xx >= 67))
        | ((yy >= 73) & (yy <= 75) & (xx >= 68))
        | ((yy == 76) & (xx >= 69))
        | ((yy >= 77) & (yy <= 82) & (xx >= 71))
    )
    arm_near_m = a & arm_near_zone

    # 5. Full Left & Right Legs (`foot_L`, `foot_R` including thighs outside the red hip sockets):
    foot_L_zone = (
        ((yy == 65) & (xx >= 34) & (xx <= 35))
        | ((yy == 66) & (xx >= 34) & (xx <= 37))
        | ((yy == 67) & (xx >= 34) & (xx <= 38))
        | ((yy >= 68) & (yy <= 69) & (xx >= 34) & (xx <= 37))
        | ((yy >= 70) & (yy <= 71) & (xx >= 33) & (xx <= 38))
        | ((yy == 72) & (xx >= 33) & (xx <= 39))
        | ((yy == 73) & (xx >= 33) & (xx <= 40))
        | ((yy == 74) & (xx >= 34) & (xx <= 42))
        | ((yy == 75) & (xx >= 35) & (xx <= 40))
        | ((yy >= 76) & (yy <= 81) & (xx >= 28) & (xx <= 42))
    )
    foot_L_m = a & foot_L_zone & (~arm_far_m)

    foot_R_zone = (
        ((yy == 66) & (xx >= 59) & (xx <= 63))
        | ((yy == 67) & (xx >= 59) & (xx <= 65))
        | ((yy >= 68) & (yy <= 69) & (xx >= 58) & (xx <= 65))
        | ((yy >= 70) & (yy <= 72) & (xx >= 58) & (xx <= 66))
        | ((yy >= 73) & (yy <= 74) & (xx >= 59) & (xx <= 67))
        | ((yy == 75) & (xx >= 60) & (xx <= 67))
        | ((yy == 76) & (xx >= 61) & (xx <= 68))
        | ((yy >= 77) & (yy <= 84) & (xx >= 60) & (xx <= 70))
    )
    foot_R_m = a & foot_R_zone & (~arm_near_m)

    # 6. Torso: Remaining central body, red hip sockets, and shoulder pauldrons:
    used = horn_crest_m | head_m | arm_far_m | arm_near_m | foot_L_m | foot_R_m
    torso_m = a & (~used)

    return {
        "arm_far": arm_far_m,
        "foot_L": foot_L_m,
        "foot_R": foot_R_m,
        "torso": torso_m,
        "head": head_m,
        "horn_crest": horn_crest_m,
        "arm_near": arm_near_m,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` (facing right) into matching 7 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Horn Crest on back.png: Giant upper horn + left & right antennae (`y=10..45`):
    horn_crest_zone = (
        (yy <= 39)
        | ((yy >= 40) & (yy <= 42) & ((xx <= 33) | (xx >= 54) | ((xx >= 39) & (xx <= 51))))
        | ((yy >= 43) & (yy <= 45) & (xx >= 53))
    )
    horn_crest_m = a & horn_crest_zone

    # 2. Head / Upper Carapace Hood on back.png (`y=40..52`, above red dorsal collar line):
    head_zone = (
        ((yy >= 40) & (yy <= 45) & (xx >= 20) & (xx <= 53))
        | ((yy == 46) & (xx >= 20) & (xx <= 48))
        | ((yy >= 47) & (yy <= 52) & (xx >= 20) & (xx <= 47))
    )
    head_m = a & head_zone & (~horn_crest_m)

    # 3. Arm Far on back.png (left extended spherical gauntlet):
    arm_far_zone = (
        ((yy >= 54) & (yy <= 56) & (xx <= 21))
        | ((yy >= 57) & (yy <= 68) & (xx <= 22))
        | ((yy >= 69) & (yy <= 70) & (xx <= 20))
        | ((yy >= 71) & (yy <= 75) & (xx <= 19))
        | ((yy == 76) & (xx <= 18))
    )
    arm_far_m = a & arm_far_zone

    # 4. Arm Near on back.png (right upper arm cylinder + giant spherical gauntlet + white claws):
    arm_near_zone = (
        ((yy == 54) & (xx >= 54))
        | ((yy >= 55) & (yy <= 58) & (xx >= 52))
        | ((yy == 59) & (xx >= 53))
        | ((yy == 60) & (xx >= 54))
        | ((yy == 61) & (xx >= 55))
        | ((yy >= 62) & (yy <= 65) & (xx >= 57))
        | ((yy == 66) & (xx >= 58))
        | ((yy >= 67) & (yy <= 70) & (xx >= 59))
        | ((yy >= 71) & (yy <= 74) & (xx >= 60))
        | ((yy >= 75) & (yy <= 80) & (xx >= 61))
    )
    arm_near_m = a & arm_near_zone

    # 5. Full Left & Right Legs (`foot_L`, `foot_R` including thighs outside the golden elytra & red hip socket):
    foot_L_zone = (
        ((yy >= 68) & (yy <= 69) & (xx >= 21) & (xx <= 23))
        | ((yy >= 70) & (yy <= 72) & (xx >= 19) & (xx <= 24))
        | ((yy >= 73) & (yy <= 74) & (xx >= 19) & (xx <= 25))
        | ((yy == 75) & (xx >= 20) & (xx <= 26))
        | ((yy >= 76) & (yy <= 77) & (xx >= 20) & (xx <= 27))
        | ((yy == 78) & (xx >= 19) & (xx <= 28))
        | ((yy >= 79) & (yy <= 81) & (xx >= 19) & (xx <= 29))
    )
    foot_L_m = a & foot_L_zone & (~arm_far_m)

    foot_R_zone = (
        ((yy == 68) & (xx >= 44) & (xx <= 54))
        | ((yy == 69) & (xx >= 44) & (xx <= 55))
        | ((yy >= 70) & (yy <= 71) & (xx >= 43) & (xx <= 55))
        | ((yy >= 72) & (yy <= 74) & (xx >= 43) & (xx <= 54))
        | ((yy >= 75) & (yy <= 76) & (xx >= 43) & (xx <= 55))
        | ((yy == 77) & (xx >= 43) & (xx <= 54))
        | ((yy >= 78) & (yy <= 84) & (xx >= 47) & (xx <= 60))
    )
    foot_R_m = a & foot_R_zone & (~arm_near_m)

    # 6. Torso: Golden-yellow elytra shell, red trim lines, and right shoulder pauldron:
    used = horn_crest_m | head_m | arm_far_m | arm_near_m | foot_L_m | foot_R_m
    torso_m = a & (~used)

    return {
        "arm_far": arm_far_m,
        "foot_L": foot_L_m,
        "foot_R": foot_R_m,
        "torso": torso_m,
        "head": head_m,
        "horn_crest": horn_crest_m,
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
    Segments native `front.png` or `back.png`, optionally flips horizontally for SE/E/NW,
    places onto the target `(cell_h, cell_w)` canvas with feet aligned to `foot_y`,
    and builds cushioned RGBA layers in strict back-to-front draw order.
    """
    raw_masks = segment_back_native(base_96) if is_back else segment_front_native(base_96)
    src_96 = base_96[:, ::-1, :].copy() if flip_x else base_96.copy()
    masks_96 = {k: (m[:, ::-1].copy() if flip_x else m.copy()) for k, m in raw_masks.items()}
    if flip_x:
        masks_96["foot_L"], masks_96["foot_R"] = masks_96["foot_R"], masks_96["foot_L"]

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
    blue_fill_m = is_blue_armor(canvas)

    layers: Dict[str, np.ndarray] = {}

    # 1. `arm_far`:
    # Cushion ONLY inside the eroded interior of `torso` and `head` where `arm_far` joins the body
    body_union_far = c_masks["torso"] | c_masks["head"] | c_masks["arm_far"]
    body_eroded_far = ndi.binary_erosion(body_union_far, iterations=2) & (c_masks["torso"] | c_masks["head"])
    layers["arm_far"] = build_part_layer(
        canvas, c_masks["arm_far"], body_eroded_far, cushion_px=5, allowed_fill_mask=blue_fill_m
    )

    # 2. `foot_L`, `foot_R`:
    # Full thighs + feet cushioned inside eroded `torso` hip sockets so stepping and crouching never open a seam!
    leg_torso_union = ndi.binary_erosion(c_masks["torso"] | c_masks["foot_L"] | c_masks["foot_R"], iterations=1)
    hip_interior = leg_torso_union & c_masks["torso"] & (nat_y >= 62)
    for fk in ("foot_L", "foot_R"):
        fm = c_masks[fk]
        lyr = build_part_layer(canvas, fm, hip_interior, cushion_px=6, allowed_fill_mask=blue_fill_m)
        layers[fk] = lyr

    # 3. `torso`:
    # Cushion ONLY inside the eroded union of foreground layers drawn ON TOP OF `torso`
    # (`head`, `horn_crest`, `arm_near`), strictly using `blue_fill_m` so red/yellow trim never bleeds!
    upper_union = c_masks["torso"] | c_masks["head"] | c_masks["horn_crest"] | c_masks["arm_near"]
    upper_eroded = ndi.binary_erosion(upper_union, iterations=2)
    fg_over_torso = upper_eroded & (c_masks["head"] | c_masks["horn_crest"] | c_masks["arm_near"]) & (nat_y >= 42)
    layers["torso"] = build_part_layer(
        canvas, c_masks["torso"], fg_over_torso, cushion_px=6, allowed_fill_mask=blue_fill_m
    )

    # 4. `head` & 5. `horn_crest`:
    head_horn_union = ndi.binary_erosion(c_masks["head"] | c_masks["horn_crest"], iterations=2)
    fg_over_head = head_horn_union & c_masks["horn_crest"] & (nat_y >= 40) & (nat_y <= 50)
    layers["head"] = build_part_layer(
        canvas, c_masks["head"], fg_over_head, cushion_px=3, allowed_fill_mask=blue_fill_m
    )
    layers["horn_crest"] = build_part_layer(
        canvas, c_masks["horn_crest"], np.zeros_like(c_masks["horn_crest"]), cushion_px=0
    )

    # 6. `arm_near`:
    # Cushion the upper-arm root inside the spherical shoulder pauldron (`upper_eroded & torso`)
    torso_eroded = ndi.binary_erosion(c_masks["torso"], iterations=2)
    fg_over_anear = torso_eroded & (nat_y >= 51) & (nat_y <= 62)
    layers["arm_near"] = build_part_layer(
        canvas, c_masks["arm_near"], fg_over_anear, cushion_px=4, allowed_fill_mask=blue_fill_m
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
    # Place impact sparks around head/shoulder height (`top_y + 24`) rather than the very tip of the tall horn
    pts = [
        (mid_x - 15 - drift, max(2, top_y + 24 - drift // 2)),
        (mid_x + 15 + drift, max(2, top_y + 25 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (248, 248, 248, 255)
    return out


def synthesize_heracross_mega_frame(
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
    horn_dx, horn_dy = 0, 0
    anear_dx, anear_dy = 0, 0
    afar_dx, afar_dy = 0, 0
    foot_L_dx, foot_L_dy = 0, 0
    foot_R_dx, foot_R_dy = 0, 0

    if action == "idle":
        # Modeled directly after Normal Heracross's official GIF (`0214_heracross/normal/front.gif`):
        # 1. Both full legs (`foot_L`, `foot_R`) stay 100% planted on the ground `(0, 0)`.
        # 2. Heavy power-fighter breathing at the hip sockets: the armored upper body (`torso`),
        #    `head`, and massive Hercules `horn_crest` flex up and down (`-1..+2px`) and lean
        #    forward (`+1px`) on inhalation with zero seam along the beetle carapace dome!
        # 3. Both giant spherical gauntlets (`arm_near`, `arm_far`) pump rhythmically up/down
        #    and forward/back in a brawler guard stance!
        body_dy = int(round(1.5 * math.sin(phase - 0.2)))
        body_dx = int(round(0.8 * max(0.0, math.sin(phase - 0.2)) * facing))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy
        horn_dx, horn_dy = body_dx, body_dy

        anear_dx = body_dx + int(round(1.5 * math.sin(phase - 0.55) * facing))
        anear_dy = body_dy + int(round(1.6 * math.sin(phase - 0.45)))

        afar_dx = body_dx + int(round(1.4 * math.sin(phase - 1.05) * facing))
        afar_dy = body_dy + int(round(1.5 * math.sin(phase - 0.95)))

    elif action == "walk":
        # `biped_horn_grappler`: Heavy armored 2-legged march!
        # Alternating left/right full-leg stomp (`lift_px=2, stride_px=2`) from the red hip sockets,
        # 2-beat upper-body bounce, counter-swinging spherical gauntlets (`arm_near` vs `arm_far`),
        # and powerful rhythmic cadence of the giant Hercules horn!
        bounce = int(round(-1.6 * abs(sin1)))
        sway_x = int(round(1.2 * sin1 * facing))
        torso_dx, torso_dy = sway_x, bounce
        head_dx, head_dy = sway_x, bounce
        horn_dx, horn_dy = sway_x, bounce

        anear_dx = sway_x + int(round(2.2 * sin1 * facing))
        anear_dy = bounce + int(round(1.5 * cos1))

        afar_dx = sway_x + int(round(-2.0 * sin1 * facing))
        afar_dy = bounce + int(round(-1.5 * cos1))

        foot_L_dx, foot_L_dy = foot_step_offset(phase, facing, lift_px=2, stride_px=2)
        foot_R_dx, foot_R_dy = foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2)

    elif action == "attack":
        # Signature Mega Heracross "Megahorn / Close Combat / Pin Missile" Combo Rush!
        # Frames 0..3: Low crouch wind-up — sink upper body & giant Hercules horn deep (`dy = +2px`),
        #              pull back (`dx = -3px`), and cock both spherical gauntlets back!
        # Frames 4..9: Explosive forward lunge (`+5px` body/horn charge!) — drive the giant Hercules horn
        #              from low crouch (`dy = +2px -> +1px`) to a ferocious upward Megahorn uppercut
        #              (`dy = -2px`!) while unleashing a rapid +6..7px dual-gauntlet Close Combat /
        #              Pin Missile barrage!
        # Frames 10..15: Follow-through recovery to brawler stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.65 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        body_dx = int(round(5.2 * k * facing))
        if frame_idx in (1, 2, 3):
            # Deep crouch lowering the giant horn before the charge
            body_dy = 2 if frame_idx in (2, 3) else 1
        elif frame_idx == 4:
            # Drive horn forward low at start of charge
            body_dy = 1
        elif frame_idx in (5, 6, 7, 8):
            # Powerful upward Megahorn uppercut!
            body_dy = -2 if frame_idx in (6, 7) else -1
        else:
            body_dy = 0

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy
        horn_dx, horn_dy = body_dx, body_dy

        # Dual gauntlet rush (Close Combat / Pin Missile barrage +6..7px forward!):
        if frame_idx in (1, 2, 3):
            anear_dx = int(round(4.5 * k * facing))
            anear_dy = body_dy + 1
            afar_dx = int(round(4.0 * k * facing))
            afar_dy = body_dy + 1
        elif frame_idx in (4, 5):
            # Barrage Hit 1: `arm_far` leads +6.5px forward while `arm_near` drives +5.5px
            anear_dx = int(round(5.6 * k * facing))
            anear_dy = body_dy
            afar_dx = int(round(6.4 * k * facing))
            afar_dy = body_dy - 1
        elif frame_idx in (6, 7, 8):
            # Barrage Hit 2: `arm_near` blasts +6.5px forward while `arm_far` follows +5.8px
            anear_dx = int(round(6.6 * k * facing))
            anear_dy = body_dy + (1 if frame_idx in (6, 7) else 0)
            afar_dx = int(round(5.8 * k * facing))
            afar_dy = body_dy
        else:
            anear_dx = int(round(5.2 * k * facing))
            anear_dy = body_dy
            afar_dx = int(round(4.8 * k * facing))
            afar_dy = body_dy

        # Lead leg steps powerfully into the lunge while rear leg follows cohesively
        step_x = int(round(3.5 * max(0.0, k) * facing))
        back_step_x = int(round(2.5 * k * facing)) if k < 0 else int(round(2.8 * k * facing))
        if facing < 0:
            foot_L_dx, foot_L_dy = step_x, 0
            foot_R_dx, foot_R_dy = back_step_x, 0
        else:
            foot_R_dx, foot_R_dy = step_x, 0
            foot_L_dx, foot_L_dy = back_step_x, 0

    elif action == "hurt":
        # Knockback + hit-shake + gauntlets & giant horn flung upward/back + hurt sparks
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        body_dx = int(round(-3.8 * env * facing)) + jit
        body_dy = int(round(-1.5 * env))

        torso_dx, torso_dy = body_dx, body_dy
        head_dx, head_dy = body_dx, body_dy
        horn_dx, horn_dy = body_dx, body_dy

        anear_dx = body_dx + int(round(-1.8 * env * facing))
        anear_dy = body_dy + int(round(-1.0 * env))
        afar_dx = body_dx + int(round(-1.5 * env * facing))
        afar_dy = body_dy + int(round(-1.0 * env))

        slide_x = int(round(body_dx * 0.65))
        foot_L_dx, foot_R_dx = slide_x, slide_x

    comp = [
        shift_layer_int(layers["arm_far"], afar_dx, afar_dy),
        shift_layer_int(layers["foot_L"], foot_L_dx, foot_L_dy),
        shift_layer_int(layers["foot_R"], foot_R_dx, foot_R_dy),
        shift_layer_int(layers["torso"], torso_dx, torso_dy),
        shift_layer_int(layers["head"], head_dx, head_dy),
        shift_layer_int(layers["horn_crest"], horn_dx, horn_dy),
        shift_layer_int(layers["arm_near"], anear_dx, anear_dy),
    ]

    out = composite_layers(canvas_shape, comp)
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

    # Tight bounds of front (89x75) and back (83x75)
    max_w, max_h = 89, 75
    cell_w = max_w + 21  # 110 (generous horizontal margin for +6..7px Megahorn / Gauntlet rush)
    cell_h = max_h + 19  # 94
    foot_y = cell_h - 4  # 90

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
                frame_rgba = synthesize_heracross_mega_frame(
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

    # Also export clean 16-frame animated GIFs (`front.gif`, `front-reverse.gif`, `back.gif`, `back-reverse.gif`)
    # in `MEGA_DIR` to replace the static 20-frame GIFs that had green Pin Missile debris artifacts:
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
        "schema": "agent-sprite-forge.meta.v2",
        "key": "0214_heracross-mega",
        "archetype": "biped_horn_grappler",
        "rig_parts": [
            "arm_far",
            "foot_L",
            "foot_R",
            "torso",
            "head",
            "horn_crest",
            "arm_near",
        ],
        "Idle": "../../monsterSprite/gen2/0214_heracross/mega/heracross-mega/agent-sprite-forge/idle.png",
        "Walk": "../../monsterSprite/gen2/0214_heracross/mega/heracross-mega/agent-sprite-forge/walk.png",
        "Attack": "../../monsterSprite/gen2/0214_heracross/mega/heracross-mega/agent-sprite-forge/attack.png",
        "Hurt": "../../monsterSprite/gen2/0214_heracross/mega/heracross-mega/agent-sprite-forge/hurt.png",
        "Sheet": "../../monsterSprite/gen2/0214_heracross/mega/heracross-mega/agent-sprite-forge/sheet.png",
        "Preview": "../../monsterSprite/gen2/0214_heracross/mega/heracross-mega/agent-sprite-forge/preview.png",
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
    print(f"Successfully forged Mega Heracross 8-dir x 4-action bundle in {OUT_DIR} ({cell_w}x{cell_h} cell)")


if __name__ == "__main__":
    main()
