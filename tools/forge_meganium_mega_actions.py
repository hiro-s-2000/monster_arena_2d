#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Meganium (`0154_meganium-mega`) modeled directly after Normal Meganium's
official GIF motion (`front.gif` / `back.gif`):
  - Exact color-guided anatomical layer separation:
      1. `antennae` (yellow feelers with brown outline `(146,102,14)`)
      2. `head` (green head & eyes peeking out of the top flower tier)
      3. `tier1_red` (top red flower collar, outline `(103,0,31)`)
      4. `tier2_dpink` (upper-mid deep pink flower collar, outline `(172,16,71)`)
      5. `tier3_lpink` (lower-mid light pink flower collar, outline `(197,63,110)`)
      6. `tier4_white` (bottom white flower collar, outline `(108,108,108)`)
      7. `torso` (green lower body & tail)
      8. `leg_0, leg_1, leg_2, leg_3` (the 4 distinct legs/feet at the bottom)
  - 100% integer-pixel rigid translation on `head` and `torso` (zero facial/body crushing).
  - Grounded feet during `idle` (matching Normal Meganium's GIF), true 4-leg alternating gait
    during `walk`, 4-tier flower bloom + forward lunge during `attack`, and knockback flinch during `hurt`.
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen2/0154_meganium/mega/meganium-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16


def trim_tight_1x(arr: np.ndarray) -> np.ndarray:
    mask = arr[:, :, 3] >= 128
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = arr[y0:y1, x0:x1].copy()
    cropped[cropped[:, :, 3] < 128] = [0, 0, 0, 0]
    cropped[cropped[:, :, 3] >= 128, 3] = 255
    return cropped


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
    # Exclude dark outlines when extending interior cushion colors
    return a & (
        ((r == 16) & (g == 16) & (b == 16))
        | ((r == 103) & (g == 0) & (b == 31))
        | ((r == 172) & (g == 16) & (b == 71))
        | ((r == 197) & (g == 63) & (b == 110))
        | ((r == 108) & (g == 108) & (b == 108))
        | ((r == 146) & (g == 102) & (b == 14))
        | ((r == 16) & (g == 123) & (b == 49))
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


def build_ordered_layers(
    src: np.ndarray,
    ordered_masks: List[Tuple[str, np.ndarray]],
    cushion_px: int = 4,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    n = len(ordered_masks)
    for i, (name, m) in enumerate(ordered_masks):
        fg = np.zeros_like(m)
        for j in range(i + 1, n):
            fg |= ordered_masks[j][1]
        out[name] = build_part_layer(src, m, fg, cushion_px=cushion_px)
    return out


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


def segment_meganium_mega(base_arr: np.ndarray, is_back: bool) -> Dict[str, np.ndarray]:
    """
    Segments Mega Meganium (`front` or `back`, or their horizontal flips) into exact
    color-guided anatomical layers:
      - `antennae`: Top yellow/gold feelers above the head (`y < y0 + 16`)
      - `head`: Green head & forehead crest peeking out of the top flower ring (front only)
      - `tier1_red`: Top red flower tier
      - `tier2_dpink`: Second deep-pink flower tier
      - `tier3_lpink`: Third light-pink flower tier
      - `tier4_white`: Fourth white/lavender flower tier
      - `torso`: Lower green body + tail
      - `leg_0, leg_1, leg_2, leg_3`: The 4 distinct bottom feet/legs
    """
    h, w = base_arr.shape[:2]
    a = base_arr[:, :, 3] >= 128
    r = base_arr[:, :, 0].astype(np.int32)
    g = base_arr[:, :, 1].astype(np.int32)
    b = base_arr[:, :, 2].astype(np.int32)

    ys, xs = np.where(a)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    yy, xx = np.mgrid[0:h, 0:w]

    # 1. Yellow/Gold Antennae colors
    is_yellow = a & (
        ((r == 146) & (g == 102) & (b == 14))
        | ((r == 230) & (g == 172) & (b == 0))
        | ((r == 255) & (g == 231) & (b == 79))
    )
    if not is_back:
        # On front view, antennae are at the top (yy < y0 + 16); lower yellow is the forehead V-crest on the head
        antennae_m = is_yellow & (yy < y0 + 15)
    else:
        antennae_m = is_yellow & (yy < y0 + 18)

    # 2. Green pixels (head on front + lower body/legs on both front and back)
    is_green = a & (
        ((r == 16) & (g == 123) & (b == 49))
        | ((r == 54) & (g == 135) & (b == 112))
        | ((r == 90) & (g == 127) & (b == 143))
        | ((r == 90) & (g == 171) & (b == 99))
        | ((r == 125) & (g == 206) & (b == 86))
        | ((r == 192) & (g == 211) & (b == 219))
        | ((r == 203) & (g == 250) & (b == 145))
    )

    # 3. Head (front view only: upper green region + forehead V-crest + enclosed eye/outline pixels)
    head_m = np.zeros_like(a)
    if not is_back:
        head_seed = (is_green | (is_yellow & (~antennae_m))) & (yy < y0 + 30)
        # Include dark outline/eye/pupil pixels immediately adjacent to or inside the head
        head_dil = ndi.binary_dilation(head_seed, iterations=2)
        head_m = a & (~antennae_m) & (yy < y0 + 29) & (
            head_seed
            | (head_dil & ((r == 16) & (g == 16) & (b == 16)))
            | (head_dil & ((r == 255) & (g == 255) & (b == 255)) & (yy < y0 + 25))
            | (head_dil & ((r == 214) & (g == 35) & (b == 31)) & (yy < y0 + 24) & (ndi.distance_transform_edt(~head_seed) <= 1))
        )

    # 4. Flower Tiers (strictly separated by their dedicated fill & outline colors!)
    rem_flower = a & (~antennae_m) & (~head_m)

    # Tier 1: Red flower
    t1_seed = rem_flower & (
        ((r == 103) & (g == 0) & (b == 31))
        | ((r == 214) & (g == 35) & (b == 31))
        | ((r == 255) & (g == 96) & (b == 83) & (yy < y0 + 40))
    )
    # Tier 2: Deep Pink flower
    t2_seed = rem_flower & (~t1_seed) & (
        ((r == 172) & (g == 16) & (b == 71))
        | ((r == 248) & (g == 106) & (b == 157))
        | ((r == 251) & (g == 146) & (b == 185))
    )
    # Tier 3: Light Pink flower
    t3_seed = rem_flower & (~t1_seed) & (~t2_seed) & (
        ((r == 197) & (g == 63) & (b == 110))
        | ((r == 255) & (g == 188) & (b == 211))
        | ((r == 255) & (g == 216) & (b == 230))
    )
    # Tier 4: White / Lavender flower
    t4_seed = rem_flower & (~t1_seed) & (~t2_seed) & (~t3_seed) & (
        ((r == 108) & (g == 108) & (b == 108))
        | ((r == 164) & (g == 151) & (b == 174))
        | ((r == 206) & (g == 201) & (b == 211))
        | ((r == 208) & (g == 208) & (b == 208))
        | ((r == 246) & (g == 164) & (b == 156))
        | ((r == 255) & (g == 255) & (b == 255))
        | ((r == 255) & (g == 96) & (b == 83) & (yy >= y0 + 40))
    )

    # Assign any black outline `(16,16,16)` in the flower zone (`yy < y0 + 66` on front, `yy < y0 + 60` on back)
    # to the nearest flower tier if closer to flower than to green body
    flower_union = t1_seed | t2_seed | t3_seed | t4_seed
    lower_green = is_green & (~head_m)
    black_px = a & (~antennae_m) & (~head_m) & ((r == 16) & (g == 16) & (b == 16))
    dist_flw = ndi.distance_transform_edt(~flower_union)
    dist_grn = ndi.distance_transform_edt(~lower_green)
    flower_black = black_px & (dist_flw <= 1) & (dist_flw < dist_grn)

    # Distribute `flower_black` to whichever tier is nearest
    for b_y, b_x in zip(*np.where(flower_black)):
        for tier_m in (t1_seed, t2_seed, t3_seed, t4_seed):
            if np.any(tier_m[max(0, b_y - 1):min(h, b_y + 2), max(0, b_x - 1):min(w, b_x + 2)]):
                tier_m[b_y, b_x] = True
                break

    # 5. Lower Green Body + 4 Distinct Legs/Feet
    body_all = a & (~antennae_m) & (~head_m) & (~t1_seed) & (~t2_seed) & (~t3_seed) & (~t4_seed)

    # Identify the 4 distinct feet at the bottom using the foot-arch level (`yy >= y1 - 11`)
    foot_cut_y = y1 - 11
    foot_band = body_all & (yy >= foot_cut_y)
    lbl, n_comp = ndi.label(foot_band)
    foot_lobes = []
    for cid in range(1, n_comp + 1):
        cm = lbl == cid
        if int(np.sum(cm)) >= 6:
            cx_mean = float(np.mean(np.where(cm)[1]))
            foot_lobes.append((cx_mean, cm))
    foot_lobes.sort(key=lambda item: item[0])

    leg_masks = [np.zeros_like(a) for _ in range(4)]
    for i, (_, cm) in enumerate(foot_lobes[:4]):
        leg_masks[i] = cm

    legs_union = leg_masks[0] | leg_masks[1] | leg_masks[2] | leg_masks[3]
    torso_m = body_all & (~legs_union)

    # Order from back to front so upper flower tiers overlap lower flower tiers,
    # `tier4_white` overlaps `torso`, `torso` overlaps `leg_0..3`, and `head` + `antennae` sit on top!
    ordered_specs = [
        ("leg_0", leg_masks[0]),
        ("leg_1", leg_masks[1]),
        ("leg_2", leg_masks[2]),
        ("leg_3", leg_masks[3]),
        ("torso", torso_m),
        ("tier4_white", t4_seed),
        ("tier3_lpink", t3_seed),
        ("tier2_dpink", t2_seed),
        ("tier1_red", t1_seed),
        ("head", head_m),
        ("antennae", antennae_m),
    ]
    layers = build_ordered_layers(base_arr, ordered_specs, cushion_px=4)

    # 1. For each leg (`leg_0..3`), build a purely vertical 3px upward extension directly above
    #    that leg's top cut row (`foot_cut_y`), without horizontal expansion into the inner-leg arch:
    for leg_idx, (leg_name, leg_m) in enumerate(ordered_specs[:4]):
        lyr = np.zeros_like(base_arr)
        lyr[leg_m] = base_arr[leg_m]
        top_row_xs = np.where(leg_m[foot_cut_y, :])[0] if 0 <= foot_cut_y < h else np.array([], dtype=int)
        for x_col in top_row_xs:
            # Copy interior green from `foot_cut_y` (or `torso_m` right above) for 3px vertically
            fill_px = base_arr[foot_cut_y, x_col]
            for dy_up in range(1, 4):
                py = foot_cut_y - dy_up
                if py >= 0 and torso_m[py, x_col]:
                    lyr[py, x_col] = base_arr[py, x_col] if not is_dark_outline(base_arr[py:py+1, x_col:x_col+1])[0, 0] else fill_px
        layers[leg_name] = lyr

    # 2. Prevent `tier1_red`, `tier2_dpink`, etc. from extending upward into `antennae_m`
    #    (since `antennae` project into open sky above the head/flower, not in front of red petals):
    layers["tier1_red"] = build_part_layer(base_arr, t1_seed, head_m, cushion_px=3)
    layers["head"] = build_part_layer(base_arr, head_m, antennae_m & (yy >= y0 + 13), cushion_px=2)
    layers["antennae"] = build_part_layer(base_arr, antennae_m, np.zeros_like(a), cushion_px=0)
    return layers


def wave_antennae_layer(
    ant_layer: np.ndarray,
    base_dx: int,
    base_dy: int,
    phase: float,
    amp_y: float = 1.6,
    amp_x: float = 1.2,
) -> np.ndarray:
    """
    Waves the yellow feelers smoothly from their base to their tips (matching Normal Meganium's GIF!)
    using row-coherent integer shear + vertical follow-through so thin outlines never split or gap.
    """
    follow_dy = int(round(0.6 * amp_y * math.sin(phase - 0.45)))
    shifted = shift_layer_int(ant_layer, base_dx, base_dy + follow_dy)
    m = shifted[:, :, 3] >= 128
    if not np.any(m):
        return shifted
    h, w = shifted.shape[:2]
    ys, xs = np.where(m)
    y_max = int(ys.max())
    y_min = int(ys.min())
    span_y = max(4, y_max - y_min + 1)

    out = np.zeros_like(shifted)
    for py in range(y_min, y_max + 1):
        u_tip = min(1.0, max(0.0, (y_max - py) / float(span_y)))
        dx = int(round(u_tip * amp_x * math.sin(phase - u_tip * 0.9)))
        row_px = shifted[py]
        if dx == 0:
            out[py] = row_px
        elif dx > 0:
            out[py, dx:w] = row_px[0:w - dx]
        else:
            out[py, 0:w + dx] = row_px[-dx:w]
    return out


def rustle_flower_tier(
    tier_layer: np.ndarray,
    dx: int,
    dy: int,
    bloom_px: int = 0,
) -> np.ndarray:
    """
    Moves a flower collar tier by integer `(dx, dy)` and optionally expands/rustles its
    left and right outer petal tips outward by `bloom_px` (`-1`, `0`, or `+1` px) so the
    flower collar breathes and blooms just like Normal Meganium's GIF!
    """
    shifted = shift_layer_int(tier_layer, dx, dy)
    if bloom_px == 0:
        return shifted
    m = shifted[:, :, 3] >= 128
    if not np.any(m):
        return shifted
    h, w = shifted.shape[:2]
    ys, xs = np.where(m)
    cx = int(round(np.mean(xs)))
    out = shifted.copy() if bloom_px > 0 else np.zeros_like(shifted)

    for py, px in zip(ys, xs):
        dist_x = px - cx
        if abs(dist_x) >= 8:
            step = bloom_px if dist_x > 0 else -bloom_px
            nx = max(0, min(w - 1, px + step))
            out[py, nx] = shifted[py, px]
        else:
            out[py, px] = shifted[py, px]
    return out


def foot_step_offset(phase: float, facing: float, lift_px: int = 2, stride_px: int = 2) -> Tuple[int, int]:
    """
    Clean integer-pixel quadruped foot step cycle:
    Half the cycle on the ground sliding back, half the cycle lifted by `1..2px` stepping forward.
    """
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
        (mid_x - 10 - drift, max(2, top_y - 2 - drift // 2)),
        (mid_x + 10 + drift, max(2, top_y - 1 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (255, 255, 255, 255)
    return out


def synthesize_meganium_mega_frame(
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

    # Default offsets (all integer pixels!)
    torso_dx, torso_dy = 0, 0
    head_dx, head_dy = 0, 0
    t1_dx, t1_dy, t1_bloom = 0, 0, 0
    t2_dx, t2_dy, t2_bloom = 0, 0, 0
    t3_dx, t3_dy, t3_bloom = 0, 0, 0
    t4_dx, t4_dy, t4_bloom = 0, 0, 0
    leg_offsets = [(0, 0), (0, 0), (0, 0), (0, 0)]
    ant_amp_y, ant_amp_x, ant_freq = 1.6, 1.0, 1.0

    if action == "idle":
        # Modeled directly after Normal Meganium's GIF:
        # 1. Feet stay 100% planted on the ground (`leg_offsets = (0,0)`).
        # 2. Body breathes gently (`-1..+1px`), Head nods rhythmically (`-2..+2px` vertical + `1px` forward nod),
        #    4 Flower tiers cascade with phase delay (`tier1 -> tier2 -> tier3 -> tier4`) and bloom `1px` at peak breath!
        torso_dy = int(round(1.0 * sin1))
        head_dy = int(round(2.0 * math.sin(phase - 0.25)))
        head_dx = int(round(1.0 * max(0.0, math.sin(phase - 0.25)) * facing))

        t1_dy = int(round(1.8 * math.sin(phase - 0.35)))
        t2_dy = int(round(1.6 * math.sin(phase - 0.65)))
        t3_dy = int(round(1.4 * math.sin(phase - 0.95)))
        t4_dy = int(round(1.2 * math.sin(phase - 1.25)))

        t1_bloom = 1 if math.sin(phase - 0.35) > 0.45 else 0
        t2_bloom = 1 if math.sin(phase - 0.65) > 0.40 else 0
        t3_bloom = 1 if math.sin(phase - 0.95) > 0.40 else 0
        t4_bloom = 1 if math.sin(phase - 1.25) > 0.45 else 0

        ant_amp_y, ant_amp_x, ant_freq = 2.0, 1.5, 1.0

    elif action == "walk":
        # Quadruped 4-leg alternating gait + rhythmic double-bounce of head & 4-tier flower cascade!
        bounce = int(round(-2.0 * abs(sin1)))
        sway_x = int(round(1.2 * sin1 * facing))
        torso_dx, torso_dy = sway_x, bounce
        head_dx = sway_x + int(round(1.2 * cos1 * facing))
        head_dy = bounce + int(round(1.5 * math.sin(2.0 * phase - 0.3)))

        # Cascading flower bounce
        t1_dx, t1_dy = head_dx, bounce + int(round(1.6 * math.sin(2.0 * phase - 0.4)))
        t2_dx, t2_dy = sway_x, bounce + int(round(1.6 * math.sin(2.0 * phase - 0.8)))
        t3_dx, t3_dy = sway_x, bounce + int(round(1.5 * math.sin(2.0 * phase - 1.2)))
        t4_dx, t4_dy = sway_x, bounce + int(round(1.3 * math.sin(2.0 * phase - 1.6)))
        t1_bloom = 1 if math.sin(2.0 * phase - 0.4) > 0.45 else 0
        t2_bloom = 1 if math.sin(2.0 * phase - 0.8) > 0.35 else 0
        t3_bloom = 1 if math.sin(2.0 * phase - 1.2) > 0.35 else 0
        t4_bloom = 1 if math.sin(2.0 * phase - 1.6) > 0.45 else 0

        # 4-leg quadruped diagonal gait: (leg_0, leg_2) vs (leg_1, leg_3)
        leg_offsets = [
            foot_step_offset(phase, facing, lift_px=2, stride_px=2),
            foot_step_offset(phase + math.pi, facing, lift_px=2, stride_px=2),
            foot_step_offset(phase + 0.4, facing, lift_px=2, stride_px=2),
            foot_step_offset(phase + math.pi + 0.4, facing, lift_px=2, stride_px=2),
        ]
        ant_amp_y, ant_amp_x, ant_freq = 2.2, 1.8, 2.0

    elif action == "attack":
        # 3-phase Solar Beam / Petal Blizzard attack:
        # Frames 0..3: Wind-up crouch (pull back, petals close slightly)
        # Frames 4..9: Full forward lunge + 4-tier flower collar FULL BLOOM + antennae whip forward!
        # Frames 10..15: Smooth recovery to idle pose
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.60 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        torso_dx = int(round(3.5 * k * facing))
        torso_dy = int(round(-1.0 * k)) if k > 0 else int(round(1.2 * abs(k)))
        head_dx = int(round(5.0 * k * facing))
        head_dy = int(round(2.0 * k)) if k > 0 else int(round(-1.0 * abs(k)))

        # Cascading forward thrust & full flower bloom on strike frames (4..9)
        t1_dx = int(round(4.6 * k * facing))
        t2_dx = int(round(4.2 * k * facing))
        t3_dx = int(round(3.8 * k * facing))
        t4_dx = int(round(3.4 * k * facing))

        t1_dy = torso_dy + int(round(-1.5 * k))
        t2_dy = torso_dy + int(round(-1.0 * k))
        t3_dy = torso_dy + int(round(-0.5 * k))
        t4_dy = torso_dy

        bloom = 1 if k > 0.35 else (-1 if k < -0.35 else 0)
        t1_bloom = bloom
        t2_bloom = bloom * (2 if k > 0.70 else 1)
        t3_bloom = bloom * (2 if k > 0.70 else 1)
        t4_bloom = bloom

        # Front legs step forward into the lunge, rear legs brace
        front_step = int(round(2.5 * max(0.0, k) * facing))
        front_lift = int(round(-2.0 * max(0.0, math.sin(max(0.0, (frame_idx - 3) / 4.0) * math.pi)))) if 3 <= frame_idx <= 6 else 0
        if facing < 0:
            leg_offsets = [(front_step, front_lift), (front_step // 2, 0), (0, 0), (0, 0)]
        else:
            leg_offsets = [(0, 0), (0, 0), (front_step // 2, 0), (front_step, front_lift)]

        ant_amp_y, ant_amp_x, ant_freq = 2.6, 2.0, 2.0

    elif action == "hurt":
        # Knockback + hit-shake + ruffled flower tiers
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        torso_dx = int(round(-3.5 * env * facing)) + jit
        torso_dy = int(round(-1.0 * env))
        head_dx = torso_dx + int(round(-1.0 * env * facing))
        head_dy = int(round(-2.0 * env))

        t1_dx, t1_dy, t1_bloom = head_dx, int(round(-2.0 * env)), (1 if 2 <= frame_idx <= 6 else 0)
        t2_dx, t2_dy, t2_bloom = torso_dx, int(round(-1.6 * env)), (1 if 2 <= frame_idx <= 6 else 0)
        t3_dx, t3_dy, t3_bloom = torso_dx, int(round(-1.2 * env)), (1 if 2 <= frame_idx <= 6 else 0)
        t4_dx, t4_dy, t4_bloom = torso_dx, int(round(-0.8 * env)), 0

        slide_x = int(round(torso_dx * 0.6))
        leg_offsets = [(slide_x, 0), (slide_x, 0), (slide_x, 0), (slide_x, 0)]
        ant_amp_y, ant_amp_x, ant_freq = 2.4, 1.8, 2.0

    comp = [
        shift_layer_int(layers["leg_0"], leg_offsets[0][0], leg_offsets[0][1]),
        shift_layer_int(layers["leg_1"], leg_offsets[1][0], leg_offsets[1][1]),
        shift_layer_int(layers["leg_2"], leg_offsets[2][0], leg_offsets[2][1]),
        shift_layer_int(layers["leg_3"], leg_offsets[3][0], leg_offsets[3][1]),
        shift_layer_int(layers["torso"], torso_dx, torso_dy),
        rustle_flower_tier(layers["tier4_white"], t4_dx, t4_dy, t4_bloom),
        rustle_flower_tier(layers["tier3_lpink"], t3_dx, t3_dy, t3_bloom),
        rustle_flower_tier(layers["tier2_dpink"], t2_dx, t2_dy, t2_bloom),
        rustle_flower_tier(layers["tier1_red"], t1_dx, t1_dy, t1_bloom),
        shift_layer_int(layers["head"], head_dx, head_dy),
        wave_antennae_layer(
            layers["antennae"],
            head_dx if np.any(layers["head"][:, :, 3] >= 128) else t1_dx,
            head_dy if np.any(layers["head"][:, :, 3] >= 128) else t1_dy,
            phase * ant_freq,
            amp_y=ant_amp_y,
            amp_x=ant_amp_x,
        ),
    ]

    out = composite_layers(canvas_shape, comp)
    if action != "hurt":
        out = snap_to_palette(out, palette_rgba)
    else:
        out = snap_to_palette(out, palette_rgba)
        out = add_hurt_sparks(out, frame_idx)
    return out


def save_transparent_gif(frames_rgba: List[np.ndarray], out_path: Path, duration_ms: int = 75) -> None:
    pil_frames: List[Image.Image] = []
    for arr in frames_rgba:
        # Build paletted frame with index 0 reserved for transparent
        a = arr[:, :, 3] >= 128
        rgb = arr[:, :, :3]
        # Collect unique colors across the frame (max 255)
        uniq = np.unique(rgb[a], axis=0) if np.any(a) else np.zeros((0, 3), dtype=np.uint8)
        pal_flat = [0, 0, 0]  # index 0 = transparent
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
    # 8 rows (directions) x 16 cols (4 cols per action: idle, walk, attack, hurt)
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

    front_crop = trim_tight_1x(front_raw)
    back_crop = trim_tight_1x(back_raw)
    palette_rgba = extract_palette(front_crop, back_crop)

    max_h = max(front_crop.shape[0], back_crop.shape[0])
    max_w = max(front_crop.shape[1], back_crop.shape[1])
    cell_w = max_w + 20
    cell_h = max_h + 16
    if cell_w % 2 == 1:
        cell_w += 1
    if cell_h % 2 == 1:
        cell_h += 1
    foot_y = cell_h - 4

    # Pre-segment the 4 base views (front_left=SW/W/S, front_right=SE/E, back_right=NE/N, back_left=NW)
    dir_layers: Dict[str, Tuple[Dict[str, np.ndarray], float]] = {}
    for d_name in DIRS:
        use_back = d_name in ("N", "NE", "NW")
        src = back_crop.copy() if use_back else front_crop.copy()
        # Native front_crop faces left (-1); native back_crop faces right (+1)
        if not use_back:
            if d_name in ("SE", "E"):
                src = src[:, ::-1, :].copy()
                facing = 1.0
            else:
                facing = -1.0
        else:
            if d_name in ("NW",):
                src = src[:, ::-1, :].copy()
                facing = -1.0
            else:
                facing = 1.0

        canvas = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
        sh, sw = src.shape[:2]
        x0 = (cell_w - sw) // 2
        y0 = foot_y - sh
        canvas[y0:y0 + sh, x0:x0 + sw] = src
        layers = segment_meganium_mega(canvas, is_back=use_back)
        dir_layers[d_name] = (layers, facing)

    sheets_by_action: Dict[str, np.ndarray] = {}
    master_rows: List[np.ndarray] = []

    for act in ACTIONS:
        act_sheet = np.zeros((8 * cell_h, N_FRAMES * cell_w, 4), dtype=np.uint8)
        for d_idx, d_name in enumerate(DIRS):
            layers, facing = dir_layers[d_name]
            for f_idx in range(N_FRAMES):
                frame_rgba = synthesize_meganium_mega_frame(
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
    # in `MEGA_DIR` so GIF viewers/loaders also get the smooth Meganium-style animation without laser artifacts:
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
        "generator": "forge_meganium_mega_actions",
        "key": "0154_meganium-mega",
        "dex": 154,
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
    print(f"Built 0154_meganium-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
