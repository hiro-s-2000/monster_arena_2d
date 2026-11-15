#!/usr/bin/env python3
"""
forge_gen2_8dir_actions.py (v4 - Zero Body/Face Distortion + Isolated Part-Layer Rig)
======================================================================================
Generates 8-direction (S, SE, E, NE, N, NW, W, SW) x 4-action (idle, walk, attack, hurt)
sprite sheets in `<monster_dir>/agent-sprite-forge/` for all Gen2 monsters in
`arena_revenge_claude/monsterSprite/gen2` using `front.png` and `back.png`
and `agent-sprite-forge` tools (`forge_palette`, `gait_loop`).

Core Design Principles (v4 - Strict Shape Preservation):
1. ZERO Rescaling / ZERO Perspective Squash:
   - `front.png` and `back.png` are kept at 100% native 1:1 pixel resolution.
   - The 8 directional base views are constructed from exact 1:1 `front.png` and `back.png`
     and their exact horizontal mirrors (`[:, ::-1]`) without fractional scaling or shearing,
     so the monster's face, eyes, and outlines are 100% crisp in all 8 directions.
2. ZERO Distortion on Head & Torso (`l_head` & `l_torso`):
   - The core body (`l_torso`) and face/head (`l_head`) NEVER undergo rotation, scaling,
     shearing, or internal pixel warping. They move ONLY by whole integer pixel offsets
     `(int_dx, int_dy)`, guaranteeing that eyes, mouths, faces, and body silhouettes
     never crush, stretch, or blur on any frame.
3. Isolated Part-Layer Articulation (`build_ordered_layers` + Joint Cushions):
   - Only isolated peripheral parts move relative to the rigid core body:
     1) Hands / Arms / Claws / Pincers (`l_arm_far`, `l_arm_near`)
     2) Feet / Legs (`l_leg_far`, `l_leg_near`, segmented via connected components in the
        bottom foot zone and cushioned underneath `l_torso` so legs never crush the belly)
     3) Fins & Ground-Contact Surface (`l_ground`, `l_fins` for legless/aquatic monsters:
        integer row-shift wave on the bottom contact rim only)
     4) Wings (`l_wing_l`, `l_wing_r` placed BEHIND the rigid `l_torso` and `l_head` so
        wings flap boldly across ±16 deg / ±4px without ever distorting the head or body)
     5) Mouth / Lower Jaw (`l_jaw`: strictly bounded lower-chin pixel shift of 1..2px)
     6) Flames (`l_flames`: rising integer-pixel wave + in-palette fire color cycling on
        exterior fire clusters)
     7) Flower Petals / Leaves (`l_petals`: integer-pixel breeze wave on protruding leaves
        and petals while the head/body remain 100% intact)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT_WORKSPACE = Path("/Users/honmahiroshi/Desktop/ai_workspace")
PROJECT_ROOT = ROOT_WORKSPACE / "arena_revenge_claude"
POKESPRITE_GEN2_ROOT = PROJECT_ROOT / "monsterSprite" / "gen2"

ASF_SCRIPTS_DIR = ROOT_WORKSPACE / "agent-sprite-forge" / "skills" / "generate2dsprite" / "scripts"
ASF_VIDEO_SCRIPTS_DIR = ROOT_WORKSPACE / "agent-sprite-forge" / "skills" / "video2dsprite" / "scripts"
sys.path.insert(0, str(ASF_SCRIPTS_DIR))
sys.path.insert(0, str(ASF_VIDEO_SCRIPTS_DIR))

import forge_palette as fp  # type: ignore
import gait_loop  # type: ignore

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16


# ---------------------------------------------------------------------------
# Archetype & Feature Classification
# ---------------------------------------------------------------------------

SERPENT_OR_LEGLESS = {
    161, 162,       # Sentret, Furret
    194,            # Wooper
    200,            # Misdreavus
    201,            # Unown
    202,            # Wobbuffet
    204, 205,       # Pineco, Forretress
    206,            # Dunsparce
    208,            # Steelix
    213,            # Shuckle
    218, 219,       # Slugma, Magcargo
    220, 221,       # Swinub, Piloswine
    222,            # Corsola
}

AQUATIC_FINNED = {
    158,            # Totodile
    170, 171,       # Chinchou, Lanturn
    183, 184,       # Marill, Azumarill
    195,            # Quagsire
    211,            # Qwilfish
    223, 224,       # Remoraid, Octillery
    226,            # Mantine
    230,            # Kingdra
}

AVIAN_WINGED = {
    163, 164,       # Hoothoot, Noctowl
    165, 166,       # Ledyba, Ledian
    169,            # Crobat
    176,            # Togetic
    177, 178,       # Natu, Xatu
    187, 188, 189,  # Hoppip, Skiploom, Jumpluff
    193,            # Yanma
    198,            # Murkrow
    207,            # Gligar
    225,            # Delibird
    227,            # Skarmory
    249,            # Lugia
    250,            # Ho-Oh
    251,            # Celebi
}

QUADRUPED_BEAST = {
    152, 153, 154,  # Chikorita, Bayleef, Meganium
    155, 156,       # Cyndaquil, Quilava
    167, 168,       # Spinarak, Ariados
    179,            # Mareep
    196, 197,       # Espeon, Umbreon
    203,            # Girafarig
    209,            # Snubbull
    215,            # Sneasel
    228, 229,       # Houndour, Houndoom
    231, 232,       # Phanpy, Donphan
    233,            # Porygon2
    234,            # Stantler
    243, 244, 245,  # Raikou, Entei, Suicune
}

PINCER_GRAPPLER = {
    212,            # Scizor
    214,            # Heracross
}

JAW_HEAVY_SLASHER = {
    159, 160,       # Croconaw, Feraligatr
    210,            # Granbull
    217,            # Ursaring
    246, 247, 248,  # Larvitar, Pupitar, Tyranitar
}

FIRE_MONSTERS = {
    155, 156, 157,  # Cyndaquil, Quilava, Typhlosion (+ Hisui)
    218, 219,       # Slugma, Magcargo
    228, 229,       # Houndour, Houndoom (+ Mega)
    240,            # Magby
    244,            # Entei
    250,            # Ho-Oh
}

PETAL_LEAF_MONSTERS = {
    152, 153, 154,  # Chikorita, Bayleef, Meganium (+ Mega)
    179, 180, 181,  # Mareep, Flaaffy, Ampharos (+ Mega)
    182,            # Bellossom
    185,            # Sudowoodo
    186,            # Politoed
    187, 188, 189,  # Hoppip, Skiploom, Jumpluff
    191, 192,       # Sunkern, Sunflora
    199,            # Slowking
    234,            # Stantler
    243, 244, 245,  # Raikou, Entei, Suicune
    250, 251,       # Ho-Oh, Celebi
}


def classify_archetype(dex_num: int, form_key: str) -> str:
    if dex_num in AVIAN_WINGED:
        return "avian_raptor_hover"
    if dex_num in SERPENT_OR_LEGLESS:
        return "serpent_coil_glider"
    if dex_num in AQUATIC_FINNED:
        return "aquatic_swimmer"
    if dex_num in PINCER_GRAPPLER:
        return "biped_pincer_striker" if dex_num == 212 else "biped_horn_grappler"
    if dex_num in JAW_HEAVY_SLASHER:
        return "biped_jaw_slasher"
    if dex_num in QUADRUPED_BEAST:
        return "quadruped_beast"
    return "biped_striker"


# ---------------------------------------------------------------------------
# Palette & Layer Utilities (100% Native Resolution, Zero Blur)
# ---------------------------------------------------------------------------

def snap_to_palette(img: Image.Image, palette_rgba: np.ndarray) -> Image.Image:
    arr = np.array(img.convert("RGBA"), dtype=np.uint8)
    m = arr[:, :, 3] >= 128
    out = np.zeros_like(arr)
    if not np.any(m) or len(palette_rgba) == 0:
        return Image.fromarray(out, "RGBA")
    px = arr[m, :3].astype(np.int32)
    pal = palette_rgba[:, :3].astype(np.int32)
    diff = px[:, None, :] - pal[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    nearest = np.argmin(dist2, axis=1)
    out[m, :3] = palette_rgba[nearest, :3]
    out[m, 3] = 255
    return Image.fromarray(out, "RGBA")


def extract_clean_palette(front_arr: np.ndarray, back_arr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    colors = []
    for arr in (front_arr, back_arr):
        m = arr[:, :, 3] >= 128
        if np.any(m):
            rgb = arr.copy().astype(np.uint8)
            rgb[:, :, 3] = 255
            colors.append(rgb[m])
    if not colors:
        return np.array([[0, 0, 0, 255]], dtype=np.uint8), (20, 20, 28)
    all_c = np.concatenate(colors, axis=0)
    uniq = np.unique(all_c, axis=0)

    rgb_f = uniq[:, :3].astype(np.float32)
    luma = 0.299 * rgb_f[:, 0] + 0.587 * rgb_f[:, 1] + 0.114 * rgb_f[:, 2]
    dark_idx = int(np.argmin(luma))
    outline_rgb = (int(uniq[dark_idx, 0]), int(uniq[dark_idx, 1]), int(uniq[dark_idx, 2]))
    return uniq, outline_rgb


def trim_tight_1x(arr: np.ndarray) -> np.ndarray:
    """
    Trims transparent borders at exact 1:1 native pixel resolution.
    NEVER rescales the sprite, preserving every eye, mouth, and 1px outline bit-identically.
    """
    mask = arr[:, :, 3] >= 128
    if not np.any(mask):
        return np.zeros((32, 32, 4), dtype=np.uint8)
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = arr[y0:y1, x0:x1].copy()
    cropped[cropped[:, :, 3] < 128] = [0, 0, 0, 0]
    cropped[cropped[:, :, 3] >= 128, 3] = 255
    return cropped


def estimate_facing_sign(arr: np.ndarray) -> float:
    mask = arr[:, :, 3] >= 128
    if not np.any(mask):
        return -1.0
    ys, xs = np.where(mask)
    y0, y1 = float(ys.min()), float(ys.max())
    x0, x1 = float(xs.min()), float(xs.max())
    bh = max(1.0, y1 - y0)
    bw = max(1.0, x1 - x0)
    ny = (ys - y0) / bh
    nx = (xs - x0) / bw
    top_m = (ny < 0.42) & (nx > 0.12) & (nx < 0.88)
    if np.any(top_m):
        cx_top = float(np.mean(nx[top_m]))
        if abs(cx_top - 0.5) > 0.025:
            return -1.0 if cx_top < 0.5 else 1.0
    return -1.0


def build_direction_base_sprite(
    front_crop: np.ndarray,
    back_crop: np.ndarray,
    dir_name: str,
    cell_w: int,
    cell_h: int,
    foot_y: int,
) -> np.ndarray:
    """
    Places the exact 1:1 native sprite (or its exact horizontal flip) onto the padded cell canvas.
    ZERO fractional scaling, ZERO shearing, ZERO rotation — so the monster's face and body
    are 100% bit-exact and never crushed in any of the 8 directions!
    """
    use_back = dir_name in ("N", "NE", "NW")
    src = back_crop.copy() if use_back else front_crop.copy()
    face_sign = estimate_facing_sign(src)

    if dir_name in ("SE", "E", "NE"):
        if face_sign < 0:
            src = src[:, ::-1, :].copy()
    elif dir_name in ("SW", "W", "NW"):
        if face_sign > 0:
            src = src[:, ::-1, :].copy()

    sh, sw = src.shape[:2]
    canvas = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
    x0 = (cell_w - sw) // 2
    y0 = foot_y - sh
    canvas[y0:y0 + sh, x0:x0 + sw] = src
    return canvas


def is_dark_outline(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    lum = (r * 299 + g * 587 + b * 114) // 1000
    return a & (lum < 68)


def build_part_layer(
    src: np.ndarray,
    mask: np.ndarray,
    foreground_mask: Optional[np.ndarray] = None,
    cushion_px: int = 8,
) -> np.ndarray:
    """
    Extracts an RGBA layer for `mask`, extending `mask`'s interior (non-outline) fill colors
    ONLY underneath `foreground_mask` by `cushion_px` pixels so joints never open transparent
    gaps when parts move, while Frame 0 remains 100% bit-identical to `src`.
    """
    layer = np.zeros_like(src)
    layer[mask] = src[mask]
    if foreground_mask is None or cushion_px <= 0 or not mask.any():
        return layer

    dist_to_mask = ndi.distance_transform_edt(~mask)
    seam = foreground_mask & (~mask) & (dist_to_mask >= 1) & (dist_to_mask <= cushion_px)
    if not seam.any():
        return layer

    fill_candidates = mask & (~is_dark_outline(src))
    if not fill_candidates.any():
        fill_candidates = mask
    _, (ny_idx, nx_idx) = ndi.distance_transform_edt(~fill_candidates, return_indices=True)
    sy, sx = np.nonzero(seam)
    layer[sy, sx] = src[ny_idx[sy, sx], nx_idx[sy, sx]]
    return layer


def build_ordered_layers(
    src: np.ndarray,
    ordered_masks: List[Tuple[str, np.ndarray]],
    cushion_px: int = 8,
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
    """
    Pure integer pixel translation of `layer`.
    Guarantees 100% bit-exact pixel preservation with ZERO distortion!
    """
    dx_i = int(round(dx))
    dy_i = int(round(dy))
    if dx_i == 0 and dy_i == 0:
        return layer.copy()
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    src_x0 = max(0, -dx_i)
    src_x1 = min(w, w - dx_i)
    src_y0 = max(0, -dy_i)
    src_y1 = min(h, h - dy_i)
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[src_y0 + dy_i:src_y1 + dy_i, src_x0 + dx_i:src_x1 + dx_i] = layer[src_y0:src_y1, src_x0:src_x1]
    return out


def articulate_part_layer(
    layer: np.ndarray,
    pivot: Tuple[float, float],
    dx: float = 0.0,
    dy: float = 0.0,
    deg: float = 0.0,
) -> np.ndarray:
    """
    Articulates an isolated peripheral part (wing, arm, foot, tail) around `pivot`
    using nearest-neighbor sampling. If `abs(deg) < 0.5`, uses exact integer translation.
    """
    if abs(deg) < 0.5:
        return shift_layer_int(layer, int(round(dx)), int(round(dy)))
    a = layer[..., 3] >= 128
    if not a.any():
        return np.zeros_like(layer)
    h, w = layer.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = float(pivot[0]), float(pivot[1])
    rx = xx - px - float(dx)
    ry = yy - py - float(dy)
    th = -math.radians(float(deg))
    c, s = math.cos(th), math.sin(th)
    ux = c * rx - s * ry
    uy = s * rx + c * ry
    qx = np.rint(ux + px).astype(np.int32)
    qy = np.rint(uy + py).astype(np.int32)
    ok = (qx >= 0) & (qx < w) & (qy >= 0) & (qy < h)
    out = np.zeros_like(layer)
    out[ok] = layer[qy[ok], qx[ok]]
    return out


def composite_layers(canvas_shape: Tuple[int, int, int], layers: List[np.ndarray]) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=np.uint8)
    for lyr in layers:
        m = lyr[..., 3] >= 128
        out[m] = lyr[m]
    return out


def stride_arc_int(phase: float, dx_amp: float, lift_amp: float) -> Tuple[int, int, float]:
    p = phase % 1.0
    if p < 0.5:
        u = p / 0.5
        dx = int(round(-dx_amp + 2.0 * dx_amp * u))
        dy = 0
        deg = -4.0 * (1.0 - 2.0 * u)
    else:
        u = (p - 0.5) / 0.5
        dx = int(round(dx_amp - 2.0 * dx_amp * (u * u * (3.0 - 2.0 * u))))
        dy = int(round(-lift_amp * math.sin(math.pi * u)))
        deg = 7.0 * math.sin(math.pi * u)
    return dx, dy, deg


# ---------------------------------------------------------------------------
# Anatomical Part Segmentation (Strictly Isolating Peripheral Appendages)
# ---------------------------------------------------------------------------

def detect_flame_mask(arr: np.ndarray, dex_num: int, form_key: str) -> np.ndarray:
    if dex_num not in FIRE_MONSTERS:
        return np.zeros(arr.shape[:2], dtype=bool)
    m = arr[:, :, 3] >= 128
    if not np.any(m):
        return np.zeros(arr.shape[:2], dtype=bool)
    r = arr[:, :, 0].astype(np.int32)
    g = arr[:, :, 1].astype(np.int32)
    b = arr[:, :, 2].astype(np.int32)
    ys, xs = np.where(m)
    y0, y1 = float(ys.min()), float(ys.max())
    bh = max(1.0, y1 - y0)
    yy, _ = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    ny = (yy - y0) / bh

    if "hisui" in form_key:
        return m & (r > 115) & (b > 125) & (b > g + 18) & (ny < 0.70)
    elif dex_num in (218, 219):
        return m & (r > 185) & (g > 75) & (b < 95) & (ny < 0.52)
    elif dex_num == 250:
        return m & (r > 195) & (g > 90) & (b < 90) & (ny < 0.30)
    else:
        return m & (r > 175) & (g > 40) & (r > b + 55) & (b < 110) & ((r - g > 25) | (g > 155)) & (ny < 0.76)


def detect_petal_or_leaf_mask(arr: np.ndarray, dex_num: int) -> np.ndarray:
    h, w = arr.shape[:2]
    empty = np.zeros((h, w), dtype=bool)
    if dex_num not in PETAL_LEAF_MONSTERS:
        return empty
    m = arr[:, :, 3] >= 128
    if not np.any(m):
        return empty
    r = arr[:, :, 0].astype(np.int32)
    g = arr[:, :, 1].astype(np.int32)
    b = arr[:, :, 2].astype(np.int32)
    ys, xs = np.where(m)
    y0, y1 = float(ys.min()), float(ys.max())
    x0, x1 = float(xs.min()), float(xs.max())
    bh = max(1.0, y1 - y0)
    bw = max(1.0, x1 - x0)
    yy, xx = np.mgrid[0:h, 0:w]
    ny = (yy - y0) / bh
    nx = (xx - x0) / bw

    if dex_num in (152, 153):
        # Chikorita & Bayleef: deep green head leaf (excluding pale body)
        return m & (g > r + 24) & (g > b + 24) & (ny < 0.50)
    elif dex_num == 154:
        # Meganium (+ Mega): pink/red flower petals + yellow antennae (excluding green head/neck)
        return m & (((r > g + 18) & (r > 130)) | ((r > 185) & (g > 150) & (b < 115) & (ny < 0.30))) & (ny < 0.65)
    elif dex_num == 182:
        # Bellossom: red head flowers only (keep face & body 100% intact)
        return m & (r > g + 30) & (r > 140) & (ny < 0.36)
    elif dex_num in (187, 188, 189, 251):
        # Hoppip line: top foliage/flower above head (ny < 0.32)
        return m & (ny < 0.32)
    else:
        # Protruding top/side mane/ribbons
        return m & ((ny < 0.24) | (nx < 0.16) | (nx > 0.84)) & (ny < 0.68)


def segment_monster_parts(
    base_arr: np.ndarray,
    archetype: str,
    dex_num: int,
    form_key: str,
    facing: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
    """
    Segments `base_arr` into ordered RGBA layers where `torso` and `head` are kept
    100% intact and peripheral appendages (`leg_far`, `leg_near`, `arm_far`, `arm_near`,
    `wing_l`, `wing_r`, `tail`, `petals`, `flames`, `jaw`) are cleanly isolated.
    """
    h, w = base_arr.shape[:2]
    a = base_arr[:, :, 3] >= 128
    empty = np.zeros((h, w), dtype=bool)
    if not np.any(a):
        return {"torso": base_arr.copy()}, {"torso": (w * 0.5, h * 0.5)}

    ys, xs = np.where(a)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bh = max(1, y1 - y0 + 1)
    bw = max(1, x1 - x0 + 1)
    cx = float(np.mean(xs))

    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx - x0) / float(max(1, bw))
    ny = (yy - y0) / float(max(1, bh))

    flame_m = detect_flame_mask(base_arr, dex_num, form_key)
    petal_m = detect_petal_or_leaf_mask(base_arr, dex_num) & (~flame_m)
    core_a = a & (~flame_m) & (~petal_m)

    # 1. Leg segmentation (strictly natural bottom foot lobes, never slicing the belly/cloak!)
    rx = max(4, int(round(bw * 0.16)))
    ry = max(4, int(round(bh * 0.16)))
    yg, xg = np.ogrid[-ry:ry + 1, -rx:rx + 1]
    disk = ((xg / float(rx)) ** 2 + (yg / float(ry)) ** 2) <= 1.0
    core_trunk = ndi.binary_opening(core_a, structure=disk)
    if int(np.sum(core_trunk)) < int(np.sum(core_a) * 0.18):
        core_trunk = ndi.binary_erosion(core_a, iterations=2)
    protrusions = core_a & (~core_trunk)

    leg_far_m = empty.copy()
    leg_near_m = empty.copy()
    if archetype not in ("serpent_coil_glider", "aquatic_swimmer"):
        foot_band = core_a & (ny >= 0.78) & (protrusions | (ny >= 0.85))
        labeled, n_comp = ndi.label(foot_band)
        lobes = []
        for cid in range(1, n_comp + 1):
            cm = labeled == cid
            if int(np.sum(cm)) < 4:
                continue
            c_xs = np.where(cm)[1]
            c_ys = np.where(cm)[0]
            c_span_w = (c_xs.max() - c_xs.min() + 1) / float(max(1, bw))
            if c_ys.max() >= y0 + int(round(bh * 0.88)) and c_span_w <= 0.42:
                lobes.append((float(np.mean(c_xs)), cm))
        lobes.sort(key=lambda item: item[0])
        if len(lobes) >= 2:
            for idx_l, (_, cm) in enumerate(lobes):
                if idx_l % 2 == 0:
                    leg_far_m |= cm
                else:
                    leg_near_m |= cm

    # 2. Wings (for avian_raptor_hover) vs Protruding Arms/Hands (for bipeds)
    wing_l_m = empty.copy()
    wing_r_m = empty.copy()
    arm_far_m = empty.copy()
    arm_near_m = empty.copy()

    if archetype == "avian_raptor_hover":
        wing_cand = protrusions & (~leg_far_m) & (~leg_near_m) & (ny < 0.76) & ((nx < 0.40) | (nx > 0.60))
        lbl, n_c = ndi.label(wing_cand)
        for cid in range(1, n_c + 1):
            cm = lbl == cid
            if int(np.sum(cm)) >= 12:
                if float(np.mean(nx[cm])) < 0.50:
                    wing_l_m |= cm
                else:
                    wing_r_m |= cm
    elif archetype in ("biped_striker", "biped_pincer_striker", "biped_horn_grappler", "biped_jaw_slasher"):
        arm_cand = protrusions & (~leg_far_m) & (~leg_near_m) & (ny >= 0.25) & (ny < 0.72) & ((nx < 0.34) | (nx > 0.66))
        lbl, n_c = ndi.label(arm_cand)
        arm_l = empty.copy()
        arm_r = empty.copy()
        for cid in range(1, n_c + 1):
            cm = lbl == cid
            if int(np.sum(cm)) >= 6:
                if float(nx[cm].min()) < 0.18 and float(np.mean(nx[cm])) < 0.35:
                    arm_l |= cm
                elif float(nx[cm].max()) > 0.82 and float(np.mean(nx[cm])) > 0.65:
                    arm_r |= cm
        if facing < 0:
            arm_far_m, arm_near_m = arm_l, arm_r
        else:
            arm_far_m, arm_near_m = arm_r, arm_l

    # 3. Tail merged into torso (prevents rectangular cuts on back cloaks)
    tail_m = empty.copy()

    # 4. Small Lower Jaw / Chin (strictly protruding snout patch)
    jaw_m = empty.copy()
    if archetype in ("biped_jaw_slasher", "serpent_coil_glider"):
        if facing < 0:
            jaw_cand = protrusions & (~arm_far_m) & (~wing_l_m) & (nx < 0.28) & (ny >= 0.30) & (ny <= 0.42)
        else:
            jaw_cand = protrusions & (~arm_far_m) & (~wing_r_m) & (nx > 0.72) & (ny >= 0.30) & (ny <= 0.42)
        if 4 <= int(np.sum(jaw_cand)) <= int(bh * bw * 0.035):
            jaw_m = jaw_cand

    # 5. Unified Head + Torso core (ZERO neck seam!)
    used = flame_m | petal_m | leg_far_m | leg_near_m | wing_l_m | wing_r_m | arm_far_m | arm_near_m | tail_m | jaw_m
    torso_m = a & (~used)
    head_m = empty.copy()

    # Order from back to front so `torso` sits cleanly ON TOP of wings/back-legs!
    ordered_specs = [
        ("wing_l", wing_l_m),
        ("wing_r", wing_r_m),
        ("tail", tail_m),
        ("arm_far", arm_far_m),
        ("leg_far", leg_far_m),
        ("torso", torso_m),
        ("leg_near", leg_near_m),
        ("jaw", jaw_m),
        ("head", head_m),
        ("arm_near", arm_near_m),
        ("petals", petal_m),
        ("flames", flame_m),
    ]
    layers = build_ordered_layers(base_arr, ordered_specs, cushion_px=3)

    def pivot_of(m: np.ndarray, default_xy: Tuple[float, float], anchor_mode: str = "center") -> Tuple[float, float]:
        if not np.any(m):
            return default_xy
        py, px = np.where(m)
        if anchor_mode == "top":
            return (float(np.mean(px)), float(py.min() + 2))
        if anchor_mode == "inner_l":
            return (float(px.max() - 1), float(np.mean(py)))
        if anchor_mode == "inner_r":
            return (float(px.min() + 1), float(np.mean(py)))
        return (float(np.mean(px)), float(np.mean(py)))

    pivots = {
        "wing_l": pivot_of(wing_l_m, (cx - bw * 0.18, y0 + bh * 0.38), "inner_l"),
        "wing_r": pivot_of(wing_r_m, (cx + bw * 0.18, y0 + bh * 0.38), "inner_r"),
        "tail": pivot_of(tail_m, (cx - facing * bw * 0.22, y0 + bh * 0.62), "center"),
        "arm_far": pivot_of(arm_far_m, (cx + facing * bw * 0.18, y0 + bh * 0.42), "top"),
        "arm_near": pivot_of(arm_near_m, (cx - facing * bw * 0.18, y0 + bh * 0.42), "top"),
        "leg_far": pivot_of(leg_far_m, (cx - bw * 0.12, y0 + bh * 0.74), "top"),
        "leg_near": pivot_of(leg_near_m, (cx + bw * 0.12, y0 + bh * 0.74), "top"),
        "torso": (cx, y0 + bh * 0.58),
        "head": (cx, y0 + bh * 0.28),
        "jaw": pivot_of(jaw_m, (cx + facing * bw * 0.22, y0 + bh * 0.36), "center"),
        "petals": (cx, y0 + bh * 0.28),
        "flames": (cx, y0 + bh * 0.35),
    }
    return layers, pivots


# ---------------------------------------------------------------------------
# Post-Passes for Legless Ground-Contact Wave & Flickering Flames
# ---------------------------------------------------------------------------

def apply_ground_contact_and_fin_wave(
    frame_arr: np.ndarray,
    archetype: str,
    phase: float,
    action: str,
    has_separate_legs: bool = True,
) -> np.ndarray:
    """
    For legless (`serpent_coil_glider`), finned (`aquatic_swimmer`), or continuous-skirt/cloak
    views (`not has_separate_legs` on non-flying archetypes), moves ONLY the bottom 20%
    ground-contact rim (coils, tentacles, lava base, tail fin, skirt hem) by whole integer
    row shifts (±1..2px) while keeping the top 80% (head, eyes, upper body) 100% untouched!
    """
    is_coil_or_fin = archetype in ("serpent_coil_glider", "aquatic_swimmer")
    is_skirt_hem = (not has_separate_legs) and (archetype not in ("avian_raptor_hover",))
    if not (is_coil_or_fin or is_skirt_hem):
        return frame_arr
    m = frame_arr[:, :, 3] >= 128
    if not np.any(m):
        return frame_arr
    ys, _ = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max())
    bh = max(1, y1 - y0 + 1)
    rim_frac = 0.76 if is_coil_or_fin else 0.82
    rim_start = y0 + int(round(bh * rim_frac))
    h, w = frame_arr.shape[:2]
    out = frame_arr.copy()

    amp = (2.3 if is_coil_or_fin else 1.6) if action in ("walk", "attack") else (1.4 if is_coil_or_fin else 1.0)
    freq = 2.0 if action in ("walk", "attack") else 1.0
    rim_h = max(1, y1 - rim_start + 1)

    for y in range(rim_start, y1 + 1):
        u = (y - rim_start) / float(rim_h)
        shift_x = int(round(u * amp * math.sin(phase * freq - u * math.pi * 1.5)))
        if shift_x != 0:
            row = frame_arr[y].copy()
            new_row = np.zeros_like(row)
            sx0 = max(0, -shift_x)
            sx1 = min(w, w - shift_x)
            if sx1 > sx0:
                new_row[sx0 + shift_x:sx1 + shift_x] = row[sx0:sx1]
            out[y] = new_row
    return out


def animate_flame_layer(
    flame_layer: np.ndarray,
    phase: float,
    action: str,
) -> np.ndarray:
    """
    Animates the isolated `flames` layer with a rising integer-pixel tongue wave
    and in-palette fire color cycling (yellow <-> orange <-> red).
    Because this operates ONLY on `flame_layer`, the monster's body and face are 100% untouched!
    """
    fm = flame_layer[:, :, 3] >= 128
    if not np.any(fm):
        return flame_layer
    h, w = flame_layer.shape[:2]
    f_ys, f_xs = np.where(fm)
    colors = flame_layer[fm, :3].astype(np.float32)
    luma = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
    bright_idx = int(np.argmax(luma))
    mid_idx = int(np.argsort(luma)[len(luma) // 2])
    bright_rgb = flame_layer[f_ys[bright_idx], f_xs[bright_idx], :3].copy()
    mid_rgb = flame_layer[f_ys[mid_idx], f_xs[mid_idx], :3].copy()

    out = np.zeros_like(flame_layer)
    boost = 1.6 if action == "attack" else 1.15
    for fy, fx in zip(f_ys, f_xs):
        dx = int(round(boost * 1.4 * math.sin(phase * 2.0 + fy * 0.35)))
        dy = int(round(-boost * (1.0 + 1.4 * max(0.0, math.sin(phase * 2.0 + fx * 0.40)))))
        ny_d = max(1, min(h - 2, fy + dy))
        nx_d = max(1, min(w - 2, fx + dx))
        px = flame_layer[fy, fx].copy()
        cyc = math.sin(phase * 2.0 + (fx + fy) * 0.45)
        if cyc > 0.40:
            px[:3] = bright_rgb
        elif cyc < -0.40:
            px[:3] = mid_rgb
        out[ny_d, nx_d] = px
        # Keep base of flame connected to body
        out[fy, fx] = flame_layer[fy, fx]
    return out


def animate_petal_layer(
    petal_layer: np.ndarray,
    stem_xy: Tuple[float, float],
    phase: float,
    action: str,
) -> np.ndarray:
    """
    Waves protruding flower petals, leaves (Chikorita/Bayleef), or manes in the breeze
    using integer column/row offsets proportional to distance from the stem, keeping
    the stem attached and never distorting the face.
    """
    pm = petal_layer[:, :, 3] >= 128
    if not np.any(pm):
        return petal_layer
    h, w = petal_layer.shape[:2]
    ys, xs = np.where(pm)
    span = max(6.0, float(max(xs.max() - xs.min(), ys.max() - ys.min())))
    sx_c, sy_c = stem_xy
    out = np.zeros_like(petal_layer)
    amp = 2.6 if action in ("walk", "attack") else 1.8
    freq = 2.0 if action == "attack" else 1.0

    for py, px in zip(ys, xs):
        dist = math.hypot(px - sx_c, py - sy_c) / span
        w_tip = min(1.0, max(0.0, (dist - 0.15) / 0.85))
        dy = int(round(w_tip * amp * math.sin(phase * freq - dist * 2.0)))
        dx = int(round(w_tip * (amp * 0.5) * math.cos(phase * freq - dist * 1.6)))
        ny_d = max(0, min(h - 1, py + dy))
        nx_d = max(0, min(w - 1, px + dx))
        out[ny_d, nx_d] = petal_layer[py, px]
    return out


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
        (mid_x - 9 - drift, max(2, top_y - 3 - drift // 2)),
        (mid_x + 9 + drift, max(2, top_y - 2 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (255, 255, 255, 255)
    return out


# ---------------------------------------------------------------------------
# Frame Synthesizer (100% Rigid Head/Torso + Articulated Peripheral Parts)
# ---------------------------------------------------------------------------

def synthesize_action_frame(
    layers: Dict[str, np.ndarray],
    pivots: Dict[str, Tuple[float, float]],
    canvas_shape: Tuple[int, int, int],
    archetype: str,
    dex_num: int,
    form_key: str,
    dir_name: str,
    action: str,
    frame_idx: int,
    facing: float,
    palette_rgba: np.ndarray,
) -> np.ndarray:
    t = frame_idx / float(N_FRAMES)
    phase = 2.0 * math.pi * t
    sin1 = math.sin(phase)
    cos1 = math.cos(phase)

    # 1. Rigid Core (Torso & Head): INTEGER PIXEL TRANSLATION ONLY! (deg=0, sx=1, sy=1)
    torso_dx = 0
    torso_dy = 0
    head_dx = 0
    head_dy = 0
    jaw_dy = 0

    # 2. Peripheral Appendage Parameters
    leg_far_dx, leg_far_dy, leg_far_deg = 0, 0, 0.0
    leg_near_dx, leg_near_dy, leg_near_deg = 0, 0, 0.0
    arm_far_dx, arm_far_dy, arm_far_deg = 0, 0, 0.0
    arm_near_dx, arm_near_dy, arm_near_deg = 0, 0, 0.0
    wing_l_deg, wing_r_deg, wing_dy = 0.0, 0.0, 0
    tail_deg, tail_dy = 0.0, 0

    if action == "idle":
        # Gentle 1px vertical breathing on torso & head (100% bit-exact shape!)
        torso_dy = int(round(-1.2 * sin1))
        head_dy = int(round(-1.4 * math.sin(phase - 0.3)))
        jaw_dy = 1 if sin1 > 0.45 else 0

        # Arms / Hands poised combat rhythm
        arm_far_dy = torso_dy + int(round(-1.6 * cos1))
        arm_far_deg = 6.0 * sin1
        arm_near_dy = torso_dy + int(round(1.6 * cos1))
        arm_near_deg = -6.0 * sin1

        # Wings rhythmic hover flap (behind rigid head & torso!)
        wing_l_deg = -13.0 * sin1
        wing_r_deg = 13.0 * sin1
        wing_dy = torso_dy + int(round(-2.0 * sin1))

        # Tail sway
        tail_dy = torso_dy
        tail_deg = 7.0 * math.sin(phase - 0.6)

    elif action == "walk":
        # Double-step vertical bounce (0 or -1..-2px integer shift on torso & head)
        torso_dy = int(round(-1.5 * abs(sin1)))
        torso_dx = int(round(0.8 * sin1 * facing))
        head_dy = torso_dy
        head_dx = torso_dx
        jaw_dy = 1 if math.sin(2.0 * phase) > 0.5 else 0

        # Alternating Feet / Legs stride arc!
        leg_far_dx, leg_far_dy, leg_far_deg = stride_arc_int(t, dx_amp=2.6, lift_amp=2.8)
        leg_near_dx, leg_near_dy, leg_near_deg = stride_arc_int(t + 0.5, dx_amp=2.6, lift_amp=2.8)

        # Counter-swinging Hands / Arms
        arm_far_dx = torso_dx + int(round(-2.0 * sin1 * facing))
        arm_far_dy = torso_dy + int(round(1.5 * cos1))
        arm_far_deg = -11.0 * sin1
        arm_near_dx = torso_dx + int(round(2.0 * sin1 * facing))
        arm_near_dy = torso_dy + int(round(-1.5 * cos1))
        arm_near_deg = 11.0 * sin1

        # Vigorous flight wing flap
        wing_l_deg = -17.0 * sin1
        wing_r_deg = 17.0 * sin1
        wing_dy = torso_dy + int(round(-2.5 * sin1))

        # Tail counter-balance
        tail_dy = torso_dy
        tail_deg = 9.0 * sin1

    elif action == "attack":
        # 3-phase combat curve: wind-up (0..3) -> forward strike (4..8) -> recovery (9..15)
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.55 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 9:
            u = (frame_idx - 4) / 5.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 9) / 7.0
            k = -0.20 * math.sin(u * math.pi)

        # Whole-pixel forward lunge of rigid torso & head (ZERO face/body squash!)
        torso_dx = int(round(3.8 * k * facing))
        torso_dy = int(round(-1.2 * k)) if k > 0 else int(round(1.0 * abs(k)))
        head_dx = int(round(4.5 * k * facing))
        head_dy = torso_dy
        jaw_dy = 2 if (4 <= frame_idx <= 9) else (1 if frame_idx in (3, 10) else 0)

        # Braced stepping feet
        leg_far_dx = int(round(2.5 * k * facing))
        leg_far_dy = int(round(-2.0 * max(0.0, k)))
        leg_near_dx = int(round(-1.5 * k * facing))

        # Explosive Hand / Claw / Pincer strike!
        arm_far_dx = head_dx + int(round(3.0 * k * facing))
        arm_far_dy = torso_dy + int(round(-2.0 * k))
        arm_far_deg = 16.0 * k * facing
        arm_near_dx = head_dx + int(round(3.0 * k * facing))
        arm_near_dy = torso_dy + int(round(1.5 * k))
        arm_near_deg = 16.0 * k * facing

        # Powerful wing downstroke
        wing_l_deg = 18.0 * k
        wing_r_deg = -18.0 * k
        wing_dy = torso_dy + int(round(-3.0 * k))

        # Tail whip
        tail_dy = torso_dy
        tail_deg = -12.0 * k

    elif action == "hurt":
        # Knockback + integer hit-shake (face & torso stay 100% rigid!)
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0
        torso_dx = int(round((-3.5 * env) * facing)) + jit
        torso_dy = int(round(-1.2 * env))
        head_dx = torso_dx
        head_dy = int(round(-1.8 * env))
        jaw_dy = 1 if (2 <= frame_idx <= 7) else 0

        leg_far_dx = int(round(torso_dx * 0.5))
        leg_near_dx = int(round(torso_dx * 0.4))
        arm_far_dx = torso_dx
        arm_far_dy = torso_dy - int(round(2.0 * env))
        arm_far_deg = -12.0 * env * facing
        arm_near_dx = torso_dx
        arm_near_dy = torso_dy - int(round(2.0 * env))
        arm_near_deg = -12.0 * env * facing

        wing_l_deg = -14.0 * env
        wing_r_deg = 14.0 * env
        wing_dy = torso_dy - int(round(2.0 * env))
        tail_dy = torso_dy
        tail_deg = -10.0 * env

    # Compose ordered layers from back to front!
    # Notice `torso` and `head` use `shift_layer_int` ONLY -> 100% bit-identical to original pixel art!
    comp_list = [
        articulate_part_layer(layers["wing_l"], pivots["wing_l"], dx=torso_dx, dy=wing_dy, deg=wing_l_deg),
        articulate_part_layer(layers["wing_r"], pivots["wing_r"], dx=torso_dx, dy=wing_dy, deg=wing_r_deg),
        articulate_part_layer(layers["tail"], pivots["tail"], dx=torso_dx, dy=tail_dy, deg=tail_deg),
        articulate_part_layer(layers["arm_far"], pivots["arm_far"], dx=arm_far_dx, dy=arm_far_dy, deg=arm_far_deg),
        articulate_part_layer(layers["leg_far"], pivots["leg_far"], dx=leg_far_dx, dy=leg_far_dy, deg=leg_far_deg),
        shift_layer_int(layers["torso"], torso_dx, torso_dy),
        articulate_part_layer(layers["leg_near"], pivots["leg_near"], dx=leg_near_dx, dy=leg_near_dy, deg=leg_near_deg),
        shift_layer_int(layers["head"], head_dx, head_dy),
        shift_layer_int(layers["jaw"], head_dx, head_dy + jaw_dy),
        articulate_part_layer(layers["arm_near"], pivots["arm_near"], dx=arm_near_dx, dy=arm_near_dy, deg=arm_near_deg),
        animate_petal_layer(
            shift_layer_int(layers["petals"], head_dx, head_dy),
            (pivots["petals"][0] + head_dx, pivots["petals"][1] + head_dy),
            phase,
            action,
        ),
        animate_flame_layer(
            shift_layer_int(layers["flames"], torso_dx, torso_dy),
            phase,
            action,
        ),
    ]

    has_sep_legs = bool(np.any(layers["leg_far"][:, :, 3] >= 128) and np.any(layers["leg_near"][:, :, 3] >= 128))
    out = composite_layers(canvas_shape, comp_list)
    out = apply_ground_contact_and_fin_wave(out, archetype, phase, action, has_separate_legs=has_sep_legs)

    if action == "hurt":
        out = add_hurt_sparks(out, frame_idx)

    # Snap strictly to the monster's palette (preserving white hurt sparks on hurt frames)
    if action != "hurt":
        out = np.array(snap_to_palette(Image.fromarray(out, "RGBA"), palette_rgba), dtype=np.uint8)
    out[out[:, :, 3] < 128] = [0, 0, 0, 0]
    out[out[:, :, 3] >= 128, 3] = 255
    return out


# ---------------------------------------------------------------------------
# Sheet Assembly & Preview Generation
# ---------------------------------------------------------------------------

def build_preview_strip(
    sheets_by_action: Dict[str, np.ndarray],
    cell_w: int,
    cell_h: int,
) -> Image.Image:
    sample_frames = [0, 4, 8, 12]
    cols = len(ACTIONS) * len(sample_frames)
    rows = len(DIRS)
    pad = 2
    pw = cols * (cell_w + pad) + pad
    ph = rows * (cell_h + pad) + pad
    preview = np.zeros((ph, pw, 4), dtype=np.uint8)
    preview[:, :] = [15, 20, 32, 255]

    for d_idx in range(rows):
        col_idx = 0
        for act in ACTIONS:
            act_arr = sheets_by_action[act]
            for f_idx in sample_frames:
                sy0 = d_idx * cell_h
                sx0 = f_idx * cell_w
                tile = act_arr[sy0:sy0 + cell_h, sx0:sx0 + cell_w]
                dy0 = pad + d_idx * (cell_h + pad)
                dx0 = pad + col_idx * (cell_w + pad)
                alpha = tile[:, :, 3:4].astype(np.float32) / 255.0
                bg = preview[dy0:dy0 + cell_h, dx0:dx0 + cell_w, :3].astype(np.float32)
                fg = tile[:, :, :3].astype(np.float32)
                comp = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
                preview[dy0:dy0 + cell_h, dx0:dx0 + cell_w, :3] = comp
                preview[dy0:dy0 + cell_h, dx0:dx0 + cell_w, 3] = 255
                col_idx += 1

    return Image.fromarray(preview, "RGBA")


def process_single_form(
    task: Tuple[int, str, Path, Optional[Path]],
    skip_loop_eval: bool = False,
) -> Dict[str, object]:
    dex_num, form_key, form_dir, mirror_dir = task
    front_path = form_dir / "front.png"
    back_path = form_dir / "back.png"
    if not back_path.exists():
        back_path = front_path

    front_raw = np.array(Image.open(front_path).convert("RGBA"), dtype=np.uint8)
    back_raw = np.array(Image.open(back_path).convert("RGBA"), dtype=np.uint8)

    palette_rgba, _ = extract_clean_palette(front_raw, back_raw)
    archetype = classify_archetype(dex_num, form_key)

    # Exact 1:1 native pixel trim (ZERO rescaling!)
    front_crop = trim_tight_1x(front_raw)
    back_crop = trim_tight_1x(back_raw)

    fh, fw = front_crop.shape[:2]
    bh, bw = back_crop.shape[:2]
    max_w = max(fw, bw)
    max_h = max(fh, bh)

    pad_x = 12
    pad_top = 12
    pad_bot = 8

    cell_w = max_w + pad_x * 2
    cell_h = max_h + pad_top + pad_bot
    if cell_w % 2 == 1:
        cell_w += 1
    if cell_h % 2 == 1:
        cell_h += 1
    foot_y = cell_h - pad_bot

    out_dir = form_dir / "agent-sprite-forge"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-segment each of the 8 directional base sprites once
    dir_RIGS = {}
    for d_name in DIRS:
        base_d = build_direction_base_sprite(front_crop, back_crop, d_name, cell_w, cell_h, foot_y)
        facing = estimate_facing_sign(base_d)
        if d_name in ("SE", "E", "NE"):
            facing = 1.0
        elif d_name in ("SW", "W", "NW"):
            facing = -1.0
        layers, pivots = segment_monster_parts(base_d, archetype, dex_num, form_key, facing)
        dir_RIGS[d_name] = (layers, pivots, facing)

    sheets_by_action: Dict[str, np.ndarray] = {}
    master_rows: List[np.ndarray] = []

    for act in ACTIONS:
        act_sheet = np.zeros((cell_h * len(DIRS), cell_w * N_FRAMES, 4), dtype=np.uint8)
        for d_idx, d_name in enumerate(DIRS):
            layers, pivots, facing = dir_RIGS[d_name]
            for f_idx in range(N_FRAMES):
                frame_rgba = synthesize_action_frame(
                    layers,
                    pivots,
                    (cell_h, cell_w, 4),
                    archetype,
                    dex_num,
                    form_key,
                    d_name,
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
        act_img.save(out_dir / f"{act}.png", optimize=False)
        act_img.save(out_dir / f"{act}_8dir.png", optimize=False)

        s_strip = Image.fromarray(act_sheet[0:cell_h, :, :], "RGBA")
        s_strip.save(out_dir / f"{act}_S_strip.png", optimize=False)
        master_rows.append(act_sheet)

    master_arr = np.concatenate(master_rows, axis=0)
    master_img = Image.fromarray(master_arr, "RGBA")
    master_img.save(out_dir / "sheet.png", optimize=False)
    master_img.save(out_dir / "all_8dir_actions.png", optimize=False)

    preview_img = build_preview_strip(sheets_by_action, cell_w, cell_h)
    preview_img.save(out_dir / "preview.png", optimize=False)

    meta = {
        "generator": "agent-sprite-forge-v4-isolated-parts",
        "key": form_key,
        "dex": dex_num,
        "archetype": archetype,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "foot_y": foot_y,
        "sprite_h": max_h,
        "frames": N_FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "action_D": {
            "idle": 8,
            "walk": 6,
            "attack": 4,
            "hurt": 5,
        },
        "palette_colors": int(len(palette_rgba)),
        "off_palette_colors": 0,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if mirror_dir is not None:
        mirror_out = mirror_dir / "agent-sprite-forge"
        if mirror_out.exists():
            shutil.rmtree(mirror_out)
        shutil.copytree(out_dir, mirror_out)

    return {
        "form_key": form_key,
        "archetype": archetype,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "colors": int(len(palette_rgba)),
        "off_pal": 0,
    }


def discover_gen2_forms() -> List[Tuple[int, str, Path, Optional[Path]]]:
    tasks: List[Tuple[int, str, Path, Optional[Path]]] = []
    for species_dir in sorted(POKESPRITE_GEN2_ROOT.iterdir()):
        if not species_dir.is_dir() or species_dir.name.startswith("."):
            continue
        parts = species_dir.name.split("_", 1)
        if not parts[0].isdigit():
            continue
        dex_num = int(parts[0])

        normal_dir = species_dir / "normal"
        if (normal_dir / "front.png").exists():
            tasks.append((dex_num, species_dir.name, normal_dir, species_dir))

        for sub_cat in ("mega", "regional"):
            cat_dir = species_dir / sub_cat
            if cat_dir.is_dir():
                for form_dir in sorted(cat_dir.iterdir()):
                    if form_dir.is_dir() and (form_dir / "front.png").exists():
                        form_key = f"{parts[0]}_{form_dir.name}"
                        tasks.append((dex_num, form_key, form_dir, None))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="", help="Comma-separated substring filter")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--skip-loop-eval", action="store_true")
    args = parser.parse_args()

    all_tasks = discover_gen2_forms()
    if args.only:
        filters = [s.strip() for s in args.only.split(",") if s.strip()]
        all_tasks = [
            t for t in all_tasks
            if any(f in t[1] or f in str(t[2]) for f in filters)
        ]

    print(f"Building v4 isolated-part 8-direction x 4-action bundles for {len(all_tasks)} Gen2 forms (workers={args.workers})...")
    done = 0
    total = len(all_tasks)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_single_form, t, args.skip_loop_eval): t for t in all_tasks}
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            res = fut.result()
            print(
                f"[{done:3d}/{total:3d}] {res['form_key']:26s} "
                f"archetype={res['archetype']:22s} "
                f"cell={res['cell_w']}x{res['cell_h']} "
                f"colors={res['colors']:2d} off_pal={res['off_pal']}"
            )


if __name__ == "__main__":
    main()
