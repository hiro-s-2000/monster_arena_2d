#!/usr/bin/env python3
"""Anatomical Part-Rig + agent-sprite-forge Action Sheet Builder (`actions/sheet.png`).

Implements the `monster-action-forge-architect` skill:
1. Extracts and locks the exact RGBA palette (`generate2dsprite.palette.v1`).
2. Segments each monster into anatomical layers (legs, arms/pincers/gauntlets, head/jaw/hood/horn,
   wings, tail, serpent coils, floating debris) with automatic interior-color joint socket cushions
   so limbs and weapons articulate cleanly around anatomical pivots with zero transparent gaps,
   zero outline tearing, and zero limb/tail fusion.
3. Generates 4 distinct, character-accurate actions (idle: 16f, walk: 16f, attack: 16f, hurt: 12f)
   tailored to the monster's locomotion & weapon archetype.
4. Scores and validates the generated loops with agent-sprite-forge's `gait_loop.py`.
5. Updates `<form>/actions/{sheet.png, preview.png, meta.json}` and `data/dir8.js`.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

GAME = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
GEN2 = GAME / "monsterSprite" / "gen2"
DIR8 = GAME / "data" / "dir8.js"

FORGE_SCRIPTS = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge/skills/video2dsprite/scripts"
)
sys.path.insert(0, str(FORGE_SCRIPTS))
import gait_loop  # noqa: E402

PAD_X = 16
PAD_TOP = 14
PAD_BOT = 6

ACTIONS = [
    ("idle", 16, 0.085),
    ("walk", 16, 0.065),
    ("attack", 16, 0.055),
    ("hurt", 12, 0.055),
]


def interp_keys(t: float, pts: list[tuple[float, float]]) -> float:
    """Smooth hermite piecewise interpolation through (t, value) keyframes."""
    if t <= pts[0][0]:
        return float(pts[0][1])
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t <= t1:
            u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            u = u * u * (3.0 - 2.0 * u)
            return float(v0 + (v1 - v0) * u)
    return float(pts[-1][1])


def is_dark_outline(rgba: np.ndarray) -> np.ndarray:
    """Detect dark exterior/interior outline pixels so joint cushions only use body fill colors."""
    a = rgba[..., 3] > 16
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    lum = (r * 299 + g * 587 + b * 114) // 1000
    return a & (lum < 68)


def build_part_layer(
    src: np.ndarray,
    mask: np.ndarray,
    foreground_mask: np.ndarray | None = None,
    cushion_px: int = 12,
) -> np.ndarray:
    """Extract an RGBA layer for `mask`, extending `mask`'s own interior (non-outline) fill colors
    ONLY underneath `foreground_mask` (layers drawn on top of this layer) by `cushion_px` pixels.
    This guarantees Frame 0 is 100% bit-identical to `front.png` while preventing any gaps when
    foreground layers rotate or translate."""
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
    _, (ny, nx) = ndi.distance_transform_edt(~fill_candidates, return_indices=True)
    sy, sx = np.nonzero(seam)
    layer[sy, sx] = src[ny[sy, sx], nx[sy, sx]]
    return layer


def build_ordered_layers(
    src: np.ndarray,
    ordered_masks: list[tuple[str, np.ndarray]],
    cushion_px: int = 12,
) -> dict[str, np.ndarray]:
    """Build back-to-front layers where each layer cushions ONLY underneath masks drawn after it."""
    out: dict[str, np.ndarray] = {}
    n = len(ordered_masks)
    for i, (name, m) in enumerate(ordered_masks):
        fg = np.zeros_like(m)
        for j in range(i + 1, n):
            fg |= ordered_masks[j][1]
        out[name] = build_part_layer(src, m, fg, cushion_px=cushion_px)
    return out


def transform_layer(
    layer: np.ndarray,
    pivot: tuple[float, float],
    dx: float = 0.0,
    dy: float = 0.0,
    deg: float = 0.0,
    sx: float = 1.0,
    sy: float = 1.0,
    shear_x: float = 0.0,
    wave_amp: float = 0.0,
    wave_phase: float = 0.0,
    wave_len: float = 24.0,
) -> np.ndarray:
    """Inverse nearest-neighbor sampling of `layer` around `pivot = (px, py)` in source coords."""
    a = layer[..., 3] > 16
    if not a.any():
        return np.zeros_like(layer)
    H, W = layer.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    px, py = float(pivot[0]), float(pivot[1])

    # Translate target back by (dx, dy)
    rx = xx - px - float(dx)
    ry = yy - py - float(dy)

    if wave_amp != 0.0:
        rx = rx - float(wave_amp) * np.sin(float(wave_phase) + (yy - py) / max(4.0, float(wave_len)) * 2.0 * np.pi)

    if shear_x != 0.0:
        rx = rx - float(shear_x) * (ry / max(10.0, H * 0.25))

    # Inverse rotation by -deg
    if deg != 0.0:
        th = -math.radians(float(deg))
        c, s = math.cos(th), math.sin(th)
        ux = c * rx - s * ry
        uy = s * rx + c * ry
    else:
        ux, uy = rx, ry

    # Inverse scale
    ux = ux / max(0.2, float(sx))
    uy = uy / max(0.2, float(sy))

    qx = np.rint(ux + px).astype(np.int32)
    qy = np.rint(uy + py).astype(np.int32)
    ok = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
    out = np.zeros_like(layer)
    out[ok] = layer[qy[ok], qx[ok]]
    return out


def composite_layers(canvas_shape: tuple[int, int, int], layers: list[np.ndarray]) -> np.ndarray:
    """Alpha-over composite ordered layers (back to front) with crisp binary alpha."""
    out = np.zeros(canvas_shape, dtype=np.uint8)
    for lyr in layers:
        m = lyr[..., 3] > 16
        out[m] = lyr[m]
    return out


def add_hurt_sparks(canvas: np.ndarray, k: int) -> np.ndarray:
    """Add crisp GBA/DS pixel impact sweat/spark droplets near the top-back of the head on hurt frames."""
    if k not in (1, 2, 3, 4, 5):
        return canvas
    out = canvas.copy()
    a = out[..., 3] > 16
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return out
    H, W = out.shape[:2]
    top_y = int(ys.min())
    mid_x = int(np.median(xs))
    drift = k - 1
    pts = [
        (mid_x - 10 - drift, max(2, top_y - 3 - drift // 2)),
        (mid_x + 10 + drift, max(2, top_y - 2 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < W - 3 and 1 <= sy < H - 3:
            out[sy : sy + 2, sx : sx + 2] = (255, 255, 255, 255)
    return out


def place_in_padded_canvas(src96: np.ndarray, crop_box: tuple[int, int, int, int]) -> tuple[np.ndarray, int, int]:
    """Place a 96x96 source sprite into a padded cell canvas and return (canvas, ox, oy)
    so 96x96 coords (x, y) map to (x + ox, y + oy)."""
    x0, y0, x1, y1 = crop_box
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    cw = bw + PAD_X * 2
    ch = bh + PAD_TOP + PAD_BOT
    ox = PAD_X - x0
    oy = PAD_TOP - y0
    canvas = np.zeros((ch, cw, 4), dtype=np.uint8)
    canvas[PAD_TOP : PAD_TOP + bh, PAD_X : PAD_X + bw] = src96[y0 : y1 + 1, x0 : x1 + 1]
    return canvas, ox, oy


def coord_grids(canvas: np.ndarray, ox: int, oy: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (xx96, yy96) grids in original 96x96 sprite coordinates for every pixel of `canvas`."""
    H, W = canvas.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    return xx - ox, yy - oy


def stride_arc(phase: float, dx_amp: float, lift_amp: float) -> tuple[float, float, float]:
    """Natural foot trajectory over phase in [0, 1):
    - [0.0, 0.5): stance phase (foot on ground y=0, slides back from -dx_amp to +dx_amp)
    - [0.5, 1.0): swing phase (foot lifts up by lift_amp and swings forward from +dx_amp to -dx_amp)
    Returns (dx, dy, deg)."""
    p = phase % 1.0
    if p < 0.5:
        u = p / 0.5
        dx = -dx_amp + 2.0 * dx_amp * u
        dy = 0.0
        deg = -6.0 * (1.0 - 2.0 * u)
    else:
        u = (p - 0.5) / 0.5
        dx = dx_amp - 2.0 * dx_amp * (u * u * (3.0 - 2.0 * u))
        dy = -lift_amp * math.sin(math.pi * u)
        deg = 10.0 * math.sin(math.pi * u)
    return dx, dy, deg


# ==============================================================================
# 1. 0154_meganium-mega (Quadruped Plant Dinosaur + Multi-layered Flower Ruff)
# ==============================================================================
def rig_meganium_mega(canvas: np.ndarray, ox: int, oy: int) -> dict[str, list[np.ndarray]]:
    a = canvas[..., 3] > 16
    r, g, b = canvas[..., 0].astype(int), canvas[..., 1].astype(int), canvas[..., 2].astype(int)
    xx, yy = coord_grids(canvas, ox, oy)

    # Anatomical masks in 96x96 space
    antennae_m = a & (yy <= 26) & ((yy <= 16) | ((r > 155) & (g > 115) & (b < 95) & (xx <= 55)))
    head_m = a & (~antennae_m) & (yy <= 36) & (xx >= 27) & (xx <= 50) & ((g > r - 10) | (yy <= 29))
    leg_fl_m = a & (yy >= 72) & (xx <= 43)
    leg_fr_m = a & (yy >= 72) & (xx >= 44) & (xx <= 64)
    leg_hr_m = a & (yy >= 71) & (xx >= 65) & (yy >= 68 + (xx - 65) * 0.35)
    tail_m = a & (~leg_hr_m) & (yy >= 61) & (xx >= 71)
    ruff_m = a & (~antennae_m) & (~head_m) & (~leg_fl_m) & (~leg_fr_m) & (~leg_hr_m) & (yy <= 71)
    torso_m = a & (~antennae_m) & (~head_m) & (~ruff_m) & (~leg_fl_m) & (~leg_fr_m) & (~leg_hr_m) & (~tail_m)

    L = build_ordered_layers(
        canvas,
        [
            ("tail", tail_m),
            ("leg_hr", leg_hr_m),
            ("leg_fl", leg_fl_m),
            ("torso", torso_m),
            ("leg_fr", leg_fr_m),
            ("ruff", ruff_m),
            ("head", head_m),
            ("ant", antennae_m),
        ],
    )
    l_tail, l_leg_hr, l_leg_fl, l_torso, l_leg_fr, l_ruff, l_head, l_ant = (
        L["tail"], L["leg_hr"], L["leg_fl"], L["torso"], L["leg_fr"], L["ruff"], L["head"], L["ant"]
    )

    p_tail = (71 + ox, 68 + oy)
    p_hr = (70 + ox, 73 + oy)
    p_fl = (36 + ox, 74 + oy)
    p_fr = (54 + ox, 74 + oy)
    p_torso = (52 + ox, 70 + oy)
    p_ruff = (48 + ox, 48 + oy)
    p_head = (39 + ox, 28 + oy)
    p_ant = (38 + ox, 21 + oy)

    rows: dict[str, list[np.ndarray]] = {"idle": [], "walk": [], "attack": [], "hurt": []}

    # IDLE (16f): gentle breathing of torso & flower ruff, antennae & tail swaying in phase lag
    for k in range(16):
        ph = 2.0 * math.pi * k / 16.0
        breath = math.sin(ph)
        sway = math.sin(ph - 0.6)
        by = round(-1.2 * breath)
        rows["idle"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_tail, p_tail, dy=by, deg=5.0 * sway),
                    transform_layer(l_leg_hr, p_hr),
                    transform_layer(l_leg_fl, p_fl),
                    transform_layer(l_torso, p_torso, dy=by),
                    transform_layer(l_leg_fr, p_fr),
                    transform_layer(l_ruff, p_ruff, dy=by, deg=2.0 * breath, sx=1.0 + 0.015 * breath),
                    transform_layer(l_head, p_head, dy=by, deg=1.8 * breath),
                    transform_layer(l_ant, p_ant, dy=by, deg=6.0 * sway),
                ],
            )
        )

    # WALK (16f): true 4-legged diagonal trot (leg_fl + leg_hr vs leg_fr)
    for k in range(16):
        t = k / 16.0
        dx1, dy1, deg1 = stride_arc(t, dx_amp=2.6, lift_amp=2.8)
        dx2, dy2, deg2 = stride_arc(t + 0.5, dx_amp=2.6, lift_amp=2.8)
        bob = round(-1.2 * abs(math.sin(2.0 * math.pi * t)))
        sway = math.sin(2.0 * math.pi * t)
        rows["walk"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_tail, p_tail, dx=-0.5, dy=bob, deg=7.0 * sway),
                    transform_layer(l_leg_hr, p_hr, dx=dx1, dy=dy1, deg=deg1),
                    transform_layer(l_leg_fl, p_fl, dx=dx1, dy=dy1, deg=deg1),
                    transform_layer(l_torso, p_torso, dx=-0.5, dy=bob),
                    transform_layer(l_leg_fr, p_fr, dx=dx2, dy=dy2, deg=deg2),
                    transform_layer(l_ruff, p_ruff, dx=-1.0 + 0.6 * sway, dy=bob, deg=2.5 * sway),
                    transform_layer(l_head, p_head, dx=-1.0 + 0.6 * sway, dy=bob, deg=2.0 * sway),
                    transform_layer(l_ant, p_ant, dx=-1.0 + 0.6 * sway, dy=bob, deg=8.0 * math.sin(2.0 * math.pi * t - 0.5)),
                ],
            )
        )

    # ATTACK (16f): wind-up brace -> full head + petal ruff + antennae solar/petal thrust (face stays 100% intact!)
    for k in range(16):
        t = k / 15.0
        lunge = interp_keys(t, [(0.0, 0.0), (0.22, 3.0), (0.42, -6.5), (0.65, -5.5), (1.0, 0.0)])
        drop = interp_keys(t, [(0.0, 0.0), (0.22, 1.5), (0.42, 1.0), (0.65, 0.5), (1.0, 0.0)])
        flare = interp_keys(t, [(0.0, 1.0), (0.22, 0.96), (0.42, 1.06), (0.65, 1.04), (1.0, 1.0)])
        tilt = interp_keys(t, [(0.0, 0.0), (0.22, -7.0), (0.42, 10.0), (0.65, 8.0), (1.0, 0.0)])
        rows["attack"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_tail, p_tail, dx=lunge * 0.4, dy=drop, deg=-tilt * 0.8),
                    transform_layer(l_leg_hr, p_hr, dx=max(0.0, lunge * 0.3)),
                    transform_layer(l_leg_fl, p_fl, dx=min(0.0, lunge * 0.6)),
                    transform_layer(l_torso, p_torso, dx=lunge * 0.5, dy=drop),
                    transform_layer(l_leg_fr, p_fr, dx=min(0.0, lunge * 0.45)),
                    transform_layer(l_ruff, p_ruff, dx=lunge * 0.85, dy=drop, deg=tilt * 0.6, sx=flare, sy=flare),
                    transform_layer(l_head, p_head, dx=lunge, dy=drop + (1.0 if lunge < -2 else 0.0), deg=tilt * 0.5),
                    transform_layer(l_ant, p_ant, dx=lunge * 1.05, dy=drop, deg=tilt * 1.4),
                ],
            )
        )

    # HURT (12f): knockback recoil to the right + petal ruffle + sweat sparks
    for k in range(12):
        t = k / 11.0
        rec = interp_keys(t, [(0.0, 0.0), (0.18, 1.0), (0.45, 0.7), (0.75, 0.25), (1.0, 0.0)])
        jit = (1.0 if k % 2 == 0 else -1.0) if 1 <= k <= 5 else 0.0
        dx = 4.5 * rec + jit
        frame = composite_layers(
            canvas.shape,
            [
                transform_layer(l_tail, p_tail, dx=dx * 0.7, dy=1.5 * rec, deg=-8.0 * rec),
                transform_layer(l_leg_hr, p_hr, dx=dx * 0.4),
                transform_layer(l_leg_fl, p_fl, dx=dx * 0.5, dy=-1.0 * rec),
                transform_layer(l_torso, p_torso, dx=dx * 0.75, dy=1.5 * rec),
                transform_layer(l_leg_fr, p_fr, dx=dx * 0.45),
                transform_layer(l_ruff, p_ruff, dx=dx * 0.9, dy=2.0 * rec, deg=-7.0 * rec),
                transform_layer(l_head, p_head, dx=dx, dy=2.0 * rec, deg=-9.0 * rec),
                transform_layer(l_ant, p_ant, dx=dx, dy=2.0 * rec, deg=-14.0 * rec),
            ],
        )
        rows["hurt"].append(add_hurt_sparks(frame, k))
    return rows


# ==============================================================================
# 2. 0160_feraligatr-mega (Biped Crocodile + Giant Red Jaw-Hood + Spiked Arms)
# ==============================================================================
def rig_feraligatr_mega(canvas: np.ndarray, ox: int, oy: int) -> dict[str, list[np.ndarray]]:
    a = canvas[..., 3] > 16
    r, g, b = canvas[..., 0].astype(int), canvas[..., 1].astype(int), canvas[..., 2].astype(int)
    xx, yy = coord_grids(canvas, ox, oy)

    # Red upper helmet-hood (hinged at skull back (57, 34))
    hood_m = a & (yy <= 42) & (xx <= 65) & ((yy <= 35) | (r > b + 20) | (xx <= 28))
    # Right arm (viewer's far left spiked arm)
    arm_r_m = a & (yy >= 48) & (xx <= 35) & (yy <= 78) & (~((xx >= 27) & (yy <= 58) & (b > r + 20)))
    # Near left arm (diagonal from shoulder (58,42) down-left to claws at (31..53, 77..85))
    arm_l_m = (
        a
        & (~hood_m)
        & (~arm_r_m)
        & (yy >= 42)
        & (xx >= 30)
        & (xx <= 66)
        & (yy >= 42 + (xx - 64) * 0.8)
        & (yy <= 85 - np.maximum(0, xx - 48) * 0.85)
        & (~((xx >= 50) & (yy >= 68)))
    )
    # Near left hind leg & foot (x=61..79, y=60..86)
    leg_l_m = a & (~arm_l_m) & (xx >= 61) & (xx <= 81) & (yy >= 60) & (~((xx >= 76) & (yy <= 74)))
    # Far right hind foot (x=47..60, y=68..80)
    leg_r_m = a & (~arm_l_m) & (~leg_l_m) & (xx >= 47) & (xx <= 61) & (yy >= 68)
    # Tail & back spikes
    tail_m = a & (~leg_l_m) & (xx >= 75) & (yy >= 50)
    # Blue crocodile head & lower yellow jaw
    head_m = a & (~hood_m) & (~arm_r_m) & (~arm_l_m) & (xx >= 25) & (xx <= 61) & (yy >= 34) & (yy <= 60)
    torso_m = a & (~hood_m) & (~arm_r_m) & (~arm_l_m) & (~leg_l_m) & (~leg_r_m) & (~tail_m) & (~head_m)

    L = build_ordered_layers(
        canvas,
        [
            ("arm_r", arm_r_m),
            ("leg_r", leg_r_m),
            ("tail", tail_m),
            ("torso", torso_m),
            ("head", head_m),
            ("leg_l", leg_l_m),
            ("arm_l", arm_l_m),
            ("hood", hood_m),
        ],
    )
    l_arm_r, l_leg_r, l_tail, l_torso, l_head, l_leg_l, l_arm_l, l_hood = (
        L["arm_r"], L["leg_r"], L["tail"], L["torso"], L["head"], L["leg_l"], L["arm_l"], L["hood"]
    )

    p_arm_r = (32 + ox, 53 + oy)
    p_leg_r = (55 + ox, 70 + oy)
    p_tail = (76 + ox, 64 + oy)
    p_torso = (62 + ox, 58 + oy)
    p_head = (52 + ox, 47 + oy)
    p_leg_l = (69 + ox, 66 + oy)
    p_arm_l = (59 + ox, 46 + oy)
    p_hood = (57 + ox, 34 + oy)

    rows: dict[str, list[np.ndarray]] = {"idle": [], "walk": [], "attack": [], "hurt": []}

    # IDLE (16f): menacing jaw-hood breathing + spiked arms flexing
    for k in range(16):
        ph = 2.0 * math.pi * k / 16.0
        bth = math.sin(ph)
        lag = math.sin(ph - 0.5)
        by = round(-1.2 * bth)
        rows["idle"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_arm_r, p_arm_r, dy=by, deg=4.0 * lag),
                    transform_layer(l_leg_r, p_leg_r),
                    transform_layer(l_tail, p_tail, dy=by, deg=4.5 * lag),
                    transform_layer(l_torso, p_torso, dy=by),
                    transform_layer(l_head, p_head, dy=by, deg=1.5 * bth),
                    transform_layer(l_leg_l, p_leg_l),
                    transform_layer(l_arm_l, p_arm_l, dy=by, deg=-4.0 * lag),
                    transform_layer(l_hood, p_hood, dy=by, deg=-3.5 * bth),
                ],
            )
        )

    # WALK (16f): heavy bipedal stomp on hind legs (leg_l / leg_r) while clawed arms counter-swing
    for k in range(16):
        t = k / 16.0
        dx_l, dy_l, deg_l = stride_arc(t, dx_amp=2.8, lift_amp=3.0)
        dx_r, dy_r, deg_r = stride_arc(t + 0.5, dx_amp=2.6, lift_amp=2.6)
        sw = math.sin(2.0 * math.pi * t)
        bob = round(-1.4 * abs(sw))
        rows["walk"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_arm_r, p_arm_r, dx=-1.5 * sw, dy=bob, deg=-9.0 * sw),
                    transform_layer(l_leg_r, p_leg_r, dx=dx_r, dy=dy_r, deg=deg_r),
                    transform_layer(l_tail, p_tail, dy=bob, deg=6.0 * sw),
                    transform_layer(l_torso, p_torso, dx=-0.5, dy=bob, deg=1.5 * sw),
                    transform_layer(l_head, p_head, dx=-1.0, dy=bob, deg=1.5 * sw),
                    transform_layer(l_leg_l, p_leg_l, dx=dx_l, dy=dy_l, deg=deg_l),
                    transform_layer(l_arm_l, p_arm_l, dx=1.8 * sw, dy=bob, deg=10.0 * sw),
                    transform_layer(l_hood, p_hood, dx=-1.0, dy=bob, deg=-3.0 * abs(sw)),
                ],
            )
        )

    # ATTACK (16f): Hood opens wide (+14 deg upward) & arms raise -> ferocious Crunch bite (-11 deg slam) + double claw slash!
    for k in range(16):
        t = k / 15.0
        lunge = interp_keys(t, [(0.0, 0.0), (0.24, 3.0), (0.44, -7.5), (0.66, -5.5), (1.0, 0.0)])
        hood_deg = interp_keys(t, [(0.0, 0.0), (0.24, 14.0), (0.44, -11.0), (0.66, -7.0), (1.0, 0.0)])
        arm_deg = interp_keys(t, [(0.0, 0.0), (0.24, 14.0), (0.44, -16.0), (0.66, -11.0), (1.0, 0.0)])
        body_dy = interp_keys(t, [(0.0, 0.0), (0.24, -1.5), (0.44, 2.0), (0.66, 1.0), (1.0, 0.0)])
        rows["attack"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_arm_r, p_arm_r, dx=lunge * 1.05, dy=body_dy, deg=arm_deg),
                    transform_layer(l_leg_r, p_leg_r, dx=min(0.0, lunge * 0.5)),
                    transform_layer(l_tail, p_tail, dx=lunge * 0.4, dy=body_dy, deg=-arm_deg * 0.4),
                    transform_layer(l_torso, p_torso, dx=lunge * 0.75, dy=body_dy, deg=arm_deg * 0.2),
                    transform_layer(l_head, p_head, dx=lunge * 0.9, dy=body_dy, deg=-hood_deg * 0.25),
                    transform_layer(l_leg_l, p_leg_l, dx=max(0.0, lunge * 0.3)),
                    transform_layer(l_arm_l, p_arm_l, dx=lunge * 0.95, dy=body_dy, deg=arm_deg),
                    transform_layer(l_hood, p_hood, dx=lunge * 0.9, dy=body_dy, deg=hood_deg),
                ],
            )
        )

    # HURT (12f): knockback + hood & arms thrown back
    for k in range(12):
        t = k / 11.0
        rec = interp_keys(t, [(0.0, 0.0), (0.18, 1.0), (0.45, 0.7), (0.75, 0.25), (1.0, 0.0)])
        jit = (1.0 if k % 2 == 0 else -1.0) if 1 <= k <= 5 else 0.0
        dx = 5.0 * rec + jit
        frame = composite_layers(
            canvas.shape,
            [
                transform_layer(l_arm_r, p_arm_r, dx=dx, dy=1.5 * rec, deg=-10.0 * rec),
                transform_layer(l_leg_r, p_leg_r, dx=dx * 0.5),
                transform_layer(l_tail, p_tail, dx=dx * 0.6, dy=1.5 * rec, deg=-8.0 * rec),
                transform_layer(l_torso, p_torso, dx=dx * 0.8, dy=1.5 * rec, deg=-5.0 * rec),
                transform_layer(l_head, p_head, dx=dx * 0.9, dy=1.5 * rec, deg=-8.0 * rec),
                transform_layer(l_leg_l, p_leg_l, dx=dx * 0.4),
                transform_layer(l_arm_l, p_arm_l, dx=dx * 0.9, dy=1.5 * rec, deg=-12.0 * rec),
                transform_layer(l_hood, p_hood, dx=dx * 0.95, dy=1.0 * rec, deg=-12.0 * rec),
            ],
        )
        rows["hurt"].append(add_hurt_sparks(frame, k))
    return rows


# ==============================================================================
# 3. 0181_ampharos-mega (Biped Dragon-Sheep + Flowing Wool Mane & Gem Tail)
# ==============================================================================
def rig_ampharos_mega(canvas: np.ndarray, ox: int, oy: int) -> dict[str, list[np.ndarray]]:
    a = canvas[..., 3] > 16
    r, g, b = canvas[..., 0].astype(int), canvas[..., 1].astype(int), canvas[..., 2].astype(int)
    xx, yy = coord_grids(canvas, ox, oy)

    # White wool mane behind head/neck (x >= 40, y <= 51, non-yellow)
    mane_m = a & (xx >= 40) & (yy <= 51) & (~((r > 180) & (g > 130) & (b < 100) & (xx <= 46)))
    # Gem-studded wool tail on the right (x >= 50, y >= 42, excluding yellow left thigh/leg)
    tail_m = a & (xx >= 50) & (yy >= 42) & (~((r > 170) & (g > 120) & (b < 110) & (xx <= 54) & (yy >= 54)))
    # Right arm (viewer's left yellow arm)
    arm_r_m = a & (xx <= 30) & (yy >= 46) & (yy <= 59)
    # Left arm (viewer's right yellow arm)
    arm_l_m = a & (~tail_m) & (xx >= 40) & (xx <= 55) & (yy >= 47) & (yy <= 62) & (b < 120)
    # Right leg (viewer's left leg, x <= 35, y >= 59)
    leg_r_m = a & (~arm_r_m) & (xx <= 35) & (yy >= 59)
    # Left leg (viewer's center-right leg, 40 <= x <= 53, y >= 61)
    leg_l_m = a & (~tail_m) & (~arm_l_m) & (xx >= 39) & (xx <= 53) & (yy >= 60)
    # Head & neck & horns
    head_m = a & (~mane_m) & (yy <= 42) & (xx <= 47)
    torso_m = a & (~mane_m) & (~tail_m) & (~arm_r_m) & (~arm_l_m) & (~leg_r_m) & (~leg_l_m) & (~head_m)

    L = build_ordered_layers(
        canvas,
        [
            ("mane", mane_m),
            ("tail", tail_m),
            ("arm_r", arm_r_m),
            ("leg_r", leg_r_m),
            ("torso", torso_m),
            ("leg_l", leg_l_m),
            ("arm_l", arm_l_m),
            ("head", head_m),
        ],
    )
    l_mane, l_tail, l_arm_r, l_leg_r, l_torso, l_leg_l, l_arm_l, l_head = (
        L["mane"], L["tail"], L["arm_r"], L["leg_r"], L["torso"], L["leg_l"], L["arm_l"], L["head"]
    )

    p_mane = (41 + ox, 31 + oy)
    p_tail = (51 + ox, 68 + oy)
    p_arm_r = (30 + ox, 49 + oy)
    p_leg_r = (29 + ox, 61 + oy)
    p_torso = (37 + ox, 56 + oy)
    p_leg_l = (46 + ox, 62 + oy)
    p_arm_l = (41 + ox, 50 + oy)
    p_head = (37 + ox, 39 + oy)

    rows: dict[str, list[np.ndarray]] = {"idle": [], "walk": [], "attack": [], "hurt": []}

    # IDLE (16f): fluffy mane & gem tail wave in S-curve, gentle breathing
    for k in range(16):
        ph = 2.0 * math.pi * k / 16.0
        bth = math.sin(ph)
        wave = math.sin(ph - 0.65)
        by = round(-1.2 * bth)
        rows["idle"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_mane, p_mane, dy=by, deg=5.5 * wave, wave_amp=1.0, wave_phase=ph),
                    transform_layer(l_tail, p_tail, dy=by, deg=6.5 * wave, wave_amp=1.2, wave_phase=ph),
                    transform_layer(l_arm_r, p_arm_r, dy=by, deg=5.0 * bth),
                    transform_layer(l_leg_r, p_leg_r),
                    transform_layer(l_torso, p_torso, dy=by),
                    transform_layer(l_leg_l, p_leg_l),
                    transform_layer(l_arm_l, p_arm_l, dy=by, deg=-5.0 * bth),
                    transform_layer(l_head, p_head, dy=by, deg=2.0 * bth),
                ],
            )
        )

    # WALK (16f): crisp alternating 2-legged steps (never fusing into 1 leg!) + tail & mane floating behind
    for k in range(16):
        t = k / 16.0
        dx_r, dy_r, deg_r = stride_arc(t, dx_amp=2.6, lift_amp=2.8)
        dx_l, dy_l, deg_l = stride_arc(t + 0.5, dx_amp=2.6, lift_amp=2.8)
        sw = math.sin(2.0 * math.pi * t)
        bob = round(-1.2 * abs(sw))
        rows["walk"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_mane, p_mane, dx=-0.5, dy=bob, deg=6.0 * sw, wave_amp=1.2, wave_phase=2.0 * math.pi * t),
                    transform_layer(l_tail, p_tail, dx=-0.5, dy=bob, deg=7.0 * sw, wave_amp=1.3, wave_phase=2.0 * math.pi * t),
                    transform_layer(l_arm_r, p_arm_r, dx=-0.5, dy=bob, deg=-10.0 * sw),
                    transform_layer(l_leg_r, p_leg_r, dx=dx_r, dy=dy_r, deg=deg_r),
                    transform_layer(l_torso, p_torso, dx=-0.5, dy=bob),
                    transform_layer(l_leg_l, p_leg_l, dx=dx_l, dy=dy_l, deg=deg_l),
                    transform_layer(l_arm_l, p_arm_l, dx=-0.5, dy=bob, deg=10.0 * sw),
                    transform_layer(l_head, p_head, dx=-0.8, dy=bob, deg=2.5 * sw),
                ],
            )
        )

    # ATTACK (16f): grounded thunder-charge -> forward palm & horn discharge (feet stay planted, no rocket stretch!)
    for k in range(16):
        t = k / 15.0
        lunge = interp_keys(t, [(0.0, 0.0), (0.25, 2.5), (0.44, -5.5), (0.66, -4.5), (1.0, 0.0)])
        tilt = interp_keys(t, [(0.0, 0.0), (0.25, -8.0), (0.44, 11.0), (0.66, 8.0), (1.0, 0.0)])
        arm_deg = interp_keys(t, [(0.0, 0.0), (0.25, -14.0), (0.44, 20.0), (0.66, 15.0), (1.0, 0.0)])
        flare_deg = interp_keys(t, [(0.0, 0.0), (0.25, 10.0), (0.44, -14.0), (0.66, -10.0), (1.0, 0.0)])
        rows["attack"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_mane, p_mane, dx=lunge * 0.85, deg=flare_deg, wave_amp=1.5, wave_phase=t * 6.0),
                    transform_layer(l_tail, p_tail, dx=lunge * 0.4, deg=-flare_deg * 0.9, wave_amp=1.5, wave_phase=t * 6.0),
                    transform_layer(l_arm_r, p_arm_r, dx=lunge * 1.1, deg=arm_deg),
                    transform_layer(l_leg_r, p_leg_r, dx=min(0.0, lunge * 0.45)),
                    transform_layer(l_torso, p_torso, dx=lunge * 0.65, deg=tilt * 0.5),
                    transform_layer(l_leg_l, p_leg_l, dx=max(0.0, lunge * 0.25)),
                    transform_layer(l_arm_l, p_arm_l, dx=lunge * 1.1, deg=arm_deg),
                    transform_layer(l_head, p_head, dx=lunge * 0.95, dy=1.0 if lunge < -2 else 0.0, deg=tilt * 0.8),
                ],
            )
        )

    # HURT (12f)
    for k in range(12):
        t = k / 11.0
        rec = interp_keys(t, [(0.0, 0.0), (0.18, 1.0), (0.45, 0.7), (0.75, 0.25), (1.0, 0.0)])
        jit = (1.0 if k % 2 == 0 else -1.0) if 1 <= k <= 5 else 0.0
        dx = 4.5 * rec + jit
        frame = composite_layers(
            canvas.shape,
            [
                transform_layer(l_mane, p_mane, dx=dx, dy=1.0 * rec, deg=-10.0 * rec),
                transform_layer(l_tail, p_tail, dx=dx * 0.6, dy=1.0 * rec, deg=10.0 * rec),
                transform_layer(l_arm_r, p_arm_r, dx=dx, deg=-12.0 * rec),
                transform_layer(l_leg_r, p_leg_r, dx=dx * 0.45),
                transform_layer(l_torso, p_torso, dx=dx * 0.75, dy=1.5 * rec, deg=-6.0 * rec),
                transform_layer(l_leg_l, p_leg_l, dx=dx * 0.35),
                transform_layer(l_arm_l, p_arm_l, dx=dx * 0.85, deg=-12.0 * rec),
                transform_layer(l_head, p_head, dx=dx * 0.95, dy=1.5 * rec, deg=-9.0 * rec),
            ],
        )
        rows["hurt"].append(add_hurt_sparks(frame, k))
    return rows


# ==============================================================================
# 4. 0208_steelix-mega (Armored Serpent + S-Coil Glide + Floating Magnetic Debris)
# ==============================================================================
def rig_steelix_mega(canvas: np.ndarray, ox: int, oy: int) -> dict[str, list[np.ndarray]]:
    a = canvas[..., 3] > 16
    xx, yy = coord_grids(canvas, ox, oy)

    # Floating magnetic debris stones (top-left, bottom-left, top-center, top-right)
    labeled, n_comp = ndi.label(a)
    comp_sizes = ndi.sum(a, labeled, range(1, n_comp + 1))
    debris_m = np.zeros_like(a)
    for cid in range(1, n_comp + 1):
        if comp_sizes[cid - 1] < 180:
            debris_m |= labeled == cid

    body_a = a & (~debris_m)
    # Massive armored head & jaw on the left (x <= 60, y <= 67)
    jaw_m = body_a & (xx <= 53) & (yy >= 49) & (yy <= 68) & (yy >= 48 + (xx - 15) * 0.22)
    head_m = body_a & (~jaw_m) & (xx <= 60) & (yy <= 65) & (~((xx >= 50) & (yy >= 54)))
    # Upper neck arch + top-right prism spike (x >= 48, y <= 60)
    arch_m = body_a & (~head_m) & (~jaw_m) & (yy <= 60)
    # Lower serpent coil + bottom prism spikes (y >= 57)
    coil_m = body_a & (~head_m) & (~jaw_m) & (~arch_m)

    L = build_ordered_layers(
        canvas,
        [
            ("coil", coil_m),
            ("arch", arch_m),
            ("jaw", jaw_m),
            ("head", head_m),
            ("debris", debris_m),
        ],
    )
    l_coil, l_arch, l_jaw, l_head, l_debris = L["coil"], L["arch"], L["jaw"], L["head"], L["debris"]

    p_coil = (66 + ox, 74 + oy)
    p_arch = (62 + ox, 45 + oy)
    p_head = (46 + ox, 42 + oy)
    p_jaw = (44 + ox, 50 + oy)
    p_deb = (45 + ox, 40 + oy)

    rows: dict[str, list[np.ndarray]] = {"idle": [], "walk": [], "attack": [], "hurt": []}

    # IDLE (16f): heavy serpent coil breathing + levitating magnetic debris + subtle jaw flex
    for k in range(16):
        ph = 2.0 * math.pi * k / 16.0
        bth = math.sin(ph)
        orb = math.sin(ph - 0.8)
        rows["idle"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_coil, p_coil, wave_amp=1.2, wave_phase=ph, wave_len=28.0),
                    transform_layer(l_arch, p_arch, dy=round(-1.2 * bth), deg=2.0 * bth),
                    transform_layer(l_jaw, p_jaw, dy=round(-1.2 * bth) + (1 if bth > 0.3 else 0), deg=-2.5 * bth),
                    transform_layer(l_head, p_head, dy=round(-1.5 * bth), deg=2.0 * bth),
                    transform_layer(l_debris, p_deb, dx=round(1.2 * orb), dy=round(-2.2 * math.cos(ph))),
                ],
            )
        )

    # WALK (16f): true S-wave serpent slither on lower coil (NO fake legs!) + head weaving forward
    for k in range(16):
        ph = 2.0 * math.pi * k / 16.0
        slither = math.sin(ph)
        bob = round(-1.5 * math.cos(ph))
        rows["walk"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(
                        l_coil,
                        p_coil,
                        dx=1.5 * slither,
                        wave_amp=3.2,
                        wave_phase=ph,
                        wave_len=22.0,
                    ),
                    transform_layer(l_arch, p_arch, dx=-1.0 + 1.5 * math.sin(ph - 0.6), dy=bob, deg=3.5 * slither),
                    transform_layer(l_jaw, p_jaw, dx=-1.5 + 1.2 * slither, dy=bob, deg=-3.0 * slither),
                    transform_layer(l_head, p_head, dx=-1.5 + 1.2 * slither, dy=bob, deg=3.0 * slither),
                    transform_layer(l_debris, p_deb, dx=round(-1.0 + 1.8 * math.cos(ph)), dy=round(2.0 * slither)),
                ],
            )
        )

    # ATTACK (16f): Serpent coils back high & opens massive iron jaw -> slams forward & crunches jaw shut!
    for k in range(16):
        t = k / 15.0
        lunge = interp_keys(t, [(0.0, 0.0), (0.26, 4.0), (0.45, -7.5), (0.66, -5.5), (1.0, 0.0)])
        head_dy = interp_keys(t, [(0.0, 0.0), (0.26, -3.0), (0.45, 2.5), (0.66, 1.5), (1.0, 0.0)])
        head_deg = interp_keys(t, [(0.0, 0.0), (0.26, -10.0), (0.45, 12.0), (0.66, 8.0), (1.0, 0.0)])
        jaw_open = interp_keys(t, [(0.0, 0.0), (0.26, -12.0), (0.38, -14.0), (0.48, 4.0), (0.68, 2.0), (1.0, 0.0)])
        rows["attack"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_coil, p_coil, dx=-lunge * 0.35, wave_amp=2.5, wave_phase=t * 2.0 * math.pi),
                    transform_layer(l_arch, p_arch, dx=lunge * 0.55, dy=head_dy * 0.6, deg=head_deg * 0.6),
                    transform_layer(l_jaw, p_jaw, dx=lunge, dy=head_dy - jaw_open * 0.18, deg=head_deg + jaw_open),
                    transform_layer(l_head, p_head, dx=lunge, dy=head_dy, deg=head_deg),
                    transform_layer(l_debris, p_deb, dx=lunge * 1.15, dy=-head_dy * 0.8, deg=head_deg * 1.2),
                ],
            )
        )

    # HURT (12f)
    for k in range(12):
        t = k / 11.0
        rec = interp_keys(t, [(0.0, 0.0), (0.18, 1.0), (0.45, 0.7), (0.75, 0.25), (1.0, 0.0)])
        jit = (1.0 if k % 2 == 0 else -1.0) if 1 <= k <= 5 else 0.0
        dx = 5.0 * rec + jit
        frame = composite_layers(
            canvas.shape,
            [
                transform_layer(l_coil, p_coil, dx=dx * 0.5, wave_amp=2.0 * rec, wave_phase=k * 1.1),
                transform_layer(l_arch, p_arch, dx=dx * 0.8, dy=1.5 * rec, deg=-7.0 * rec),
                transform_layer(l_jaw, p_jaw, dx=dx, dy=2.5 * rec, deg=-12.0 * rec),
                transform_layer(l_head, p_head, dx=dx, dy=1.5 * rec, deg=-8.0 * rec),
                transform_layer(l_debris, p_deb, dx=dx * 1.2, dy=-2.0 * rec),
            ],
        )
        rows["hurt"].append(add_hurt_sparks(frame, k))
    return rows


# ==============================================================================
# 5. 0212_scizor-mega (Biped Armored Mantis + Giant Toothed Pincers + Wings)
# ==============================================================================
def rig_scizor_mega(canvas: np.ndarray, ox: int, oy: int) -> dict[str, list[np.ndarray]]:
    a = canvas[..., 3] > 16
    r, g, b = canvas[..., 0].astype(int), canvas[..., 1].astype(int), canvas[..., 2].astype(int)
    xx, yy = coord_grids(canvas, ox, oy)

    # Translucent wings (upper right & inner white/grey wing pixels)
    wings_m = a & (
        ((xx >= 55) & (yy <= 40))
        | ((r > 175) & (g > 175) & (b > 185) & (yy >= 11) & (yy <= 51) & (xx >= 42))
    )
    # Right pincer (viewer's left giant toothed pincer: upper red blade + lower silver blade)
    pincer_r_m = a & (xx <= 42) & (yy >= 31) & (yy <= 62) & (~((xx >= 33) & (yy >= 53)))
    pincer_r_low_m = pincer_r_m & (yy >= 46) & (xx <= 37)
    pincer_r_up_m = pincer_r_m & (~pincer_r_low_m)
    # Left pincer (viewer's right giant pincer: x >= 63, y >= 39)
    pincer_l_m = a & (~wings_m) & (xx >= 63) & (yy >= 39) & (~((xx <= 67) & (yy >= 56)))
    # Right leg (viewer's left pointed leg: x <= 46, y >= 53)
    leg_r_m = a & (~pincer_r_m) & (xx <= 46) & (yy >= 53)
    # Left leg (viewer's right pointed leg: 55 <= x <= 69, y >= 55)
    leg_l_m = a & (~pincer_l_m) & (xx >= 55) & (xx <= 69) & (yy >= 55)
    # Head (3-horned helmet: y <= 32, 39 <= x <= 63)
    head_m = a & (~wings_m) & (yy <= 32) & (xx >= 39) & (xx <= 63)
    torso_m = a & (~wings_m) & (~pincer_r_m) & (~pincer_l_m) & (~leg_r_m) & (~leg_l_m) & (~head_m)

    L = build_ordered_layers(
        canvas,
        [
            ("wings", wings_m),
            ("leg_r", leg_r_m),
            ("leg_l", leg_l_m),
            ("torso", torso_m),
            ("head", head_m),
            ("pincer_l", pincer_l_m),
            ("pincer_r_low", pincer_r_low_m),
            ("pincer_r_up", pincer_r_up_m),
        ],
    )
    l_wings, l_leg_r, l_leg_l, l_torso, l_head, l_pincer_l, l_pincer_r_low, l_pincer_r_up = (
        L["wings"], L["leg_r"], L["leg_l"], L["torso"], L["head"], L["pincer_l"], L["pincer_r_low"], L["pincer_r_up"]
    )

    p_wings = (54 + ox, 34 + oy)
    p_leg_r = (41 + ox, 57 + oy)
    p_leg_l = (60 + ox, 58 + oy)
    p_torso = (51 + ox, 47 + oy)
    p_head = (50 + ox, 28 + oy)
    p_pin_l = (65 + ox, 44 + oy)
    p_pin_r = (40 + ox, 43 + oy)
    p_pin_r_jaw = (35 + ox, 47 + oy)

    rows: dict[str, list[np.ndarray]] = {"idle": [], "walk": [], "attack": [], "hurt": []}

    # IDLE (16f): high-frequency insect wing buzz + poised pincers breathing + jaw micro-snap
    for k in range(16):
        ph = 2.0 * math.pi * k / 16.0
        bth = math.sin(ph)
        buzz = 4.0 * (1.0 if k % 2 == 0 else -1.0)
        by = round(-1.2 * bth)
        rows["idle"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_wings, p_wings, dy=by, deg=buzz),
                    transform_layer(l_leg_r, p_leg_r),
                    transform_layer(l_leg_l, p_leg_l),
                    transform_layer(l_torso, p_torso, dy=by),
                    transform_layer(l_head, p_head, dy=by, deg=1.5 * bth),
                    transform_layer(l_pincer_l, p_pin_l, dy=by, deg=-4.0 * bth),
                    transform_layer(l_pincer_r_low, p_pin_r_jaw, dy=by, deg=3.5 * bth - 3.0 * max(0.0, bth)),
                    transform_layer(l_pincer_r_up, p_pin_r, dy=by, deg=3.5 * bth),
                ],
            )
        )

    # WALK (16f): crisp martial dash-step (legs never cross into an X or fuse!) + buzzing wings
    for k in range(16):
        t = k / 16.0
        dx_r, dy_r, deg_r = stride_arc(t, dx_amp=2.6, lift_amp=2.6)
        dx_l, dy_l, deg_l = stride_arc(t + 0.5, dx_amp=2.6, lift_amp=2.6)
        sw = math.sin(2.0 * math.pi * t)
        bob = round(-1.2 * abs(sw))
        buzz = 5.0 * (1.0 if k % 2 == 0 else -1.0)
        rows["walk"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_wings, p_wings, dx=-0.8, dy=bob, deg=buzz),
                    transform_layer(l_leg_r, p_leg_r, dx=dx_r, dy=dy_r, deg=deg_r),
                    transform_layer(l_leg_l, p_leg_l, dx=dx_l, dy=dy_l, deg=deg_l),
                    transform_layer(l_torso, p_torso, dx=-0.8, dy=bob, deg=2.0 * sw),
                    transform_layer(l_head, p_head, dx=-1.0, dy=bob, deg=2.0 * sw),
                    transform_layer(l_pincer_l, p_pin_l, dx=-0.8 + 1.5 * sw, dy=bob, deg=9.0 * sw),
                    transform_layer(l_pincer_r_low, p_pin_r, dx=-0.8 - 1.5 * sw, dy=bob, deg=-8.0 * sw),
                    transform_layer(l_pincer_r_up, p_pin_r, dx=-0.8 - 1.5 * sw, dy=bob, deg=-8.0 * sw),
                ],
            )
        )

    # ATTACK (16f): High-speed Bullet Punch / X-Scissor thrust + toothed pincer snap!
    for k in range(16):
        t = k / 15.0
        lunge = interp_keys(t, [(0.0, 0.0), (0.22, 3.0), (0.40, -7.5), (0.64, -6.0), (1.0, 0.0)])
        pin_thrust = interp_keys(t, [(0.0, 0.0), (0.22, 4.0), (0.40, -10.5), (0.64, -8.5), (1.0, 0.0)])
        pin_deg = interp_keys(t, [(0.0, 0.0), (0.22, -12.0), (0.40, 12.0), (0.64, 8.0), (1.0, 0.0)])
        jaw_snap = interp_keys(t, [(0.0, 0.0), (0.22, -10.0), (0.34, -12.0), (0.44, 6.0), (0.66, 4.0), (1.0, 0.0)])
        buzz = 7.0 * (1.0 if k % 2 == 0 else -1.0)
        rows["attack"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_wings, p_wings, dx=lunge * 0.7, deg=buzz - 8.0),
                    transform_layer(l_leg_r, p_leg_r, dx=min(0.0, lunge * 0.55)),
                    transform_layer(l_leg_l, p_leg_l, dx=max(0.0, lunge * 0.25)),
                    transform_layer(l_torso, p_torso, dx=lunge * 0.7, dy=1.0 if lunge < -2 else 0.0, deg=pin_deg * 0.35),
                    transform_layer(l_head, p_head, dx=lunge * 0.85, dy=1.0 if lunge < -2 else 0.0, deg=pin_deg * 0.3),
                    transform_layer(l_pincer_l, p_pin_l, dx=pin_thrust * 0.85, dy=-1.5 if lunge < -2 else 0.0, deg=pin_deg * 1.3),
                    transform_layer(
                        l_pincer_r_low,
                        p_pin_r_jaw,
                        dx=pin_thrust,
                        dy=1.0 if lunge < -2 else 0.0,
                        deg=pin_deg + jaw_snap,
                    ),
                    transform_layer(
                        l_pincer_r_up,
                        p_pin_r,
                        dx=pin_thrust,
                        dy=1.0 if lunge < -2 else 0.0,
                        deg=pin_deg,
                    ),
                ],
            )
        )

    # HURT (12f): crisp recoil without squashing into a 4-legged bug
    for k in range(12):
        t = k / 11.0
        rec = interp_keys(t, [(0.0, 0.0), (0.18, 1.0), (0.45, 0.7), (0.75, 0.25), (1.0, 0.0)])
        jit = (1.0 if k % 2 == 0 else -1.0) if 1 <= k <= 5 else 0.0
        dx = 4.5 * rec + jit
        frame = composite_layers(
            canvas.shape,
            [
                transform_layer(l_wings, p_wings, dx=dx, dy=1.0 * rec, deg=-10.0 * rec),
                transform_layer(l_leg_r, p_leg_r, dx=dx * 0.45),
                transform_layer(l_leg_l, p_leg_l, dx=dx * 0.35),
                transform_layer(l_torso, p_torso, dx=dx * 0.8, dy=1.0 * rec, deg=-6.0 * rec),
                transform_layer(l_head, p_head, dx=dx * 0.9, dy=1.5 * rec, deg=-8.0 * rec),
                transform_layer(l_pincer_l, p_pin_l, dx=dx * 0.9, dy=1.0 * rec, deg=-10.0 * rec),
                transform_layer(l_pincer_r_low, p_pin_r, dx=dx, dy=1.0 * rec, deg=-12.0 * rec),
                transform_layer(l_pincer_r_up, p_pin_r, dx=dx, dy=1.0 * rec, deg=-10.0 * rec),
            ],
        )
        rows["hurt"].append(add_hurt_sparks(frame, k))
    return rows


# ==============================================================================
# 6. 0214_heracross-mega (Biped Rhino Beetle + Tall Horn + Heavy Rounded Forearms)
# ==============================================================================
def rig_heracross_mega(canvas: np.ndarray, ox: int, oy: int) -> dict[str, list[np.ndarray]]:
    a = canvas[..., 3] > 16
    xx, yy = coord_grids(canvas, ox, oy)

    # Right heavy rounded forearm (viewer's left gauntlet: x <= 33, y >= 53)
    arm_r_m = a & (xx <= 33) & (yy >= 53)
    # Left heavy rounded forearm (viewer's right gauntlet: x >= 68, y >= 54, excluding left foot at x<=69, y>=76)
    arm_l_m = a & (xx >= 67) & (yy >= 53) & (~((xx <= 69) & (yy >= 75)))
    # Right leg & white-clawed foot (viewer's bottom-left foot: 33 <= x <= 44, y >= 66)
    leg_r_m = a & (~arm_r_m) & (xx >= 32) & (xx <= 45) & (yy >= 66)
    # Left leg & white-clawed foot (viewer's bottom-right foot: 58 <= x <= 70, y >= 67)
    leg_l_m = a & (~arm_l_m) & (xx >= 58) & (xx <= 70) & (yy >= 67)
    # Tall central horn + antennae + brow/eyes (y <= 50, 26 <= x <= 62)
    horn_head_m = a & (yy <= 50) & (xx >= 26) & (xx <= 62)
    torso_m = a & (~arm_r_m) & (~arm_l_m) & (~leg_r_m) & (~leg_l_m) & (~horn_head_m)

    L = build_ordered_layers(
        canvas,
        [
            ("arm_r", arm_r_m),
            ("leg_r", leg_r_m),
            ("torso", torso_m),
            ("leg_l", leg_l_m),
            ("horn", horn_head_m),
            ("arm_l", arm_l_m),
        ],
    )
    l_arm_r, l_leg_r, l_torso, l_leg_l, l_horn, l_arm_l = (
        L["arm_r"], L["leg_r"], L["torso"], L["leg_l"], L["horn"], L["arm_l"]
    )

    p_arm_r = (32 + ox, 59 + oy)
    p_leg_r = (39 + ox, 68 + oy)
    p_torso = (51 + ox, 60 + oy)
    p_leg_l = (64 + ox, 69 + oy)
    p_horn = (46 + ox, 47 + oy)
    p_arm_l = (67 + ox, 57 + oy)

    rows: dict[str, list[np.ndarray]] = {"idle": [], "walk": [], "attack": [], "hurt": []}

    # IDLE (16f): powerful beetle breathing + gauntlets & horn poised
    for k in range(16):
        ph = 2.0 * math.pi * k / 16.0
        bth = math.sin(ph)
        by = round(-1.2 * bth)
        rows["idle"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_arm_r, p_arm_r, dy=by, deg=4.5 * bth),
                    transform_layer(l_leg_r, p_leg_r),
                    transform_layer(l_torso, p_torso, dy=by),
                    transform_layer(l_leg_l, p_leg_l),
                    transform_layer(l_horn, p_horn, dy=by, deg=2.5 * bth),
                    transform_layer(l_arm_l, p_arm_l, dy=by, deg=-4.5 * bth),
                ],
            )
        )

    # WALK (16f): 16 full frames of sturdy bipedal beetle march (feet stay 100% visible!) + gauntlet pump
    for k in range(16):
        t = k / 16.0
        dx_r, dy_r, deg_r = stride_arc(t, dx_amp=2.4, lift_amp=2.5)
        dx_l, dy_l, deg_l = stride_arc(t + 0.5, dx_amp=2.4, lift_amp=2.5)
        sw = math.sin(2.0 * math.pi * t)
        bob = round(-1.2 * abs(sw))
        rows["walk"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_arm_r, p_arm_r, dx=-1.5 * sw, dy=bob, deg=-9.0 * sw),
                    transform_layer(l_leg_r, p_leg_r, dx=dx_r, dy=dy_r, deg=deg_r),
                    transform_layer(l_torso, p_torso, dx=-0.5, dy=bob, deg=2.0 * sw),
                    transform_layer(l_leg_l, p_leg_l, dx=dx_l, dy=dy_l, deg=deg_l),
                    transform_layer(l_horn, p_horn, dx=-0.8, dy=bob, deg=3.0 * sw),
                    transform_layer(l_arm_l, p_arm_l, dx=1.5 * sw, dy=bob, deg=9.0 * sw),
                ],
            )
        )

    # ATTACK (16f): Low crouch lowering the giant horn (+16 deg forward-down dip) -> explosive Megahorn uppercut (-16 deg) + gauntlet drive!
    for k in range(16):
        t = k / 15.0
        lunge = interp_keys(t, [(0.0, 0.0), (0.28, 3.0), (0.46, -7.5), (0.68, -5.5), (1.0, 0.0)])
        crouch = interp_keys(t, [(0.0, 0.0), (0.28, 3.0), (0.46, -2.0), (0.68, -1.0), (1.0, 0.0)])
        # Positive deg rotates counter-clockwise/clockwise: in transform_layer, +deg tips top (ry<0) to the right (+rx), -deg tips top to the left (-rx)!
        horn_deg = interp_keys(t, [(0.0, 0.0), (0.28, -18.0), (0.46, 14.0), (0.68, 9.0), (1.0, 0.0)])
        arm_dx = interp_keys(t, [(0.0, 0.0), (0.28, 3.5), (0.46, -9.0), (0.68, -7.0), (1.0, 0.0)])
        arm_deg = interp_keys(t, [(0.0, 0.0), (0.28, -12.0), (0.46, 16.0), (0.68, 11.0), (1.0, 0.0)])
        rows["attack"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_arm_r, p_arm_r, dx=arm_dx, dy=crouch, deg=arm_deg),
                    transform_layer(l_leg_r, p_leg_r, dx=min(0.0, lunge * 0.55)),
                    transform_layer(l_torso, p_torso, dx=lunge * 0.75, dy=crouch, deg=horn_deg * 0.35),
                    transform_layer(l_leg_l, p_leg_l, dx=max(0.0, lunge * 0.25)),
                    transform_layer(l_horn, p_horn, dx=lunge, dy=crouch, deg=horn_deg),
                    transform_layer(l_arm_l, p_arm_l, dx=arm_dx * 0.9, dy=crouch, deg=arm_deg),
                ],
            )
        )

    # HURT (12f)
    for k in range(12):
        t = k / 11.0
        rec = interp_keys(t, [(0.0, 0.0), (0.18, 1.0), (0.45, 0.7), (0.75, 0.25), (1.0, 0.0)])
        jit = (1.0 if k % 2 == 0 else -1.0) if 1 <= k <= 5 else 0.0
        dx = 4.5 * rec + jit
        frame = composite_layers(
            canvas.shape,
            [
                transform_layer(l_arm_r, p_arm_r, dx=dx, dy=1.5 * rec, deg=-10.0 * rec),
                transform_layer(l_leg_r, p_leg_r, dx=dx * 0.45),
                transform_layer(l_torso, p_torso, dx=dx * 0.8, dy=1.5 * rec, deg=-6.0 * rec),
                transform_layer(l_leg_l, p_leg_l, dx=dx * 0.35),
                transform_layer(l_horn, p_horn, dx=dx * 0.95, dy=1.5 * rec, deg=10.0 * rec),
                transform_layer(l_arm_l, p_arm_l, dx=dx * 0.9, dy=1.5 * rec, deg=-10.0 * rec),
            ],
        )
        rows["hurt"].append(add_hurt_sparks(frame, k))
    return rows


# ==============================================================================
# 7. 0227_skarmory-mega (Avian Raptor Hover + Dark Navy Blade Wings + Long Golden Talons)
# ==============================================================================
def rig_skarmory_mega(canvas: np.ndarray, ox: int, oy: int) -> dict[str, list[np.ndarray]]:
    a = canvas[..., 3] > 16
    xx, yy = coord_grids(canvas, ox, oy)

    # Long sweeping blade wings (x >= 32, y <= 46)
    wings_m = a & (xx >= 32) & (yy <= 46) & (~((xx <= 38) & (yy >= 34)))
    # Head, crest & open toothed beak (x <= 39, y <= 29)
    head_m = a & (~wings_m) & (xx <= 39) & (yy <= 29)
    # Long 3-pronged golden talons hanging below (y >= 56)
    # Right talon cluster (viewer's left 2 long golden talons: x <= 35, y >= 56)
    talons_r_m = a & (yy >= 56) & (xx <= 35)
    # Left talon cluster (viewer's right 2 long golden talons: x >= 36, y >= 56)
    talons_l_m = a & (yy >= 56) & (xx >= 36)
    # Raptor torso, armored breast & tail feathers
    torso_m = a & (~wings_m) & (~head_m) & (~talons_r_m) & (~talons_l_m)

    L = build_ordered_layers(
        canvas,
        [
            ("wings", wings_m),
            ("talons_l", talons_l_m),
            ("torso", torso_m),
            ("talons_r", talons_r_m),
            ("head", head_m),
        ],
    )
    l_wings, l_talons_l, l_torso, l_talons_r, l_head = (
        L["wings"], L["talons_l"], L["torso"], L["talons_r"], L["head"]
    )

    p_wings = (34 + ox, 34 + oy)
    p_tal_l = (43 + ox, 57 + oy)
    p_torso = (34 + ox, 42 + oy)
    p_tal_r = (32 + ox, 57 + oy)
    p_head = (23 + ox, 27 + oy)

    rows: dict[str, list[np.ndarray]] = {"idle": [], "walk": [], "attack": [], "hurt": []}

    # IDLE (16f): rhythmic aerial hover + blade wing flap + long golden talons swaying intact!
    for k in range(16):
        ph = 2.0 * math.pi * k / 16.0
        hov = math.sin(ph)
        lag = math.sin(ph - 0.6)
        by = round(-2.2 * hov)
        rows["idle"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_wings, p_wings, dy=by, deg=-10.0 * hov),
                    transform_layer(l_talons_l, p_tal_l, dy=by, deg=5.5 * lag),
                    transform_layer(l_torso, p_torso, dy=by, deg=1.5 * hov),
                    transform_layer(l_talons_r, p_tal_r, dy=by, deg=4.5 * lag),
                    transform_layer(l_head, p_head, dy=by, deg=-2.5 * hov),
                ],
            )
        )

    # WALK (16f): aerodynamic forward raptor glide + vigorous blade-wing strokes + alternating talon air-stride
    for k in range(16):
        ph = 2.0 * math.pi * k / 16.0
        flap = math.sin(ph)
        by = round(-2.0 * math.cos(ph))
        rows["walk"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_wings, p_wings, dx=-1.5, dy=by, deg=-14.0 * flap),
                    transform_layer(l_talons_l, p_tal_l, dx=-1.5 - 1.5 * flap, dy=by, deg=-10.0 * flap),
                    transform_layer(l_torso, p_torso, dx=-1.5, dy=by, deg=-4.0),
                    transform_layer(l_talons_r, p_tal_r, dx=-1.5 + 1.5 * flap, dy=by, deg=10.0 * flap),
                    transform_layer(l_head, p_head, dx=-2.0, dy=by, deg=-3.0 + 2.0 * flap),
                ],
            )
        )

    # ATTACK (16f): High aerial wind-up -> ferocious Talon Dive & Brave Bird slash with all 4 golden talons raking forward!
    for k in range(16):
        t = k / 15.0
        lunge = interp_keys(t, [(0.0, 0.0), (0.25, 4.5), (0.44, -8.5), (0.66, -6.5), (1.0, 0.0)])
        dive_dy = interp_keys(t, [(0.0, 0.0), (0.25, -3.5), (0.44, 2.0), (0.66, 1.0), (1.0, 0.0)])
        wing_deg = interp_keys(t, [(0.0, 0.0), (0.25, -16.0), (0.44, 14.0), (0.66, 9.0), (1.0, 0.0)])
        # Positive deg on talons (ry > 0) swings the long golden claws forward-left!
        talon_deg = interp_keys(t, [(0.0, 0.0), (0.25, -12.0), (0.44, 20.0), (0.66, 15.0), (1.0, 0.0)])
        rows["attack"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_wings, p_wings, dx=lunge * 0.85, dy=dive_dy, deg=wing_deg),
                    transform_layer(l_talons_l, p_tal_l, dx=lunge * 1.05, dy=dive_dy, deg=talon_deg * 0.9),
                    transform_layer(l_torso, p_torso, dx=lunge * 0.9, dy=dive_dy, deg=-talon_deg * 0.25),
                    transform_layer(l_talons_r, p_tal_r, dx=lunge * 1.15, dy=dive_dy, deg=talon_deg * 1.1),
                    transform_layer(l_head, p_head, dx=lunge * 1.05, dy=dive_dy + (1.0 if lunge < -2 else 0.0), deg=-talon_deg * 0.35),
                ],
            )
        )

    # HURT (12f)
    for k in range(12):
        t = k / 11.0
        rec = interp_keys(t, [(0.0, 0.0), (0.18, 1.0), (0.45, 0.7), (0.75, 0.25), (1.0, 0.0)])
        jit = (1.0 if k % 2 == 0 else -1.0) if 1 <= k <= 5 else 0.0
        dx = 5.0 * rec + jit
        frame = composite_layers(
            canvas.shape,
            [
                transform_layer(l_wings, p_wings, dx=dx, dy=-1.0 * rec, deg=12.0 * rec),
                transform_layer(l_talons_l, p_tal_l, dx=dx * 0.8, dy=1.0 * rec, deg=-10.0 * rec),
                transform_layer(l_torso, p_torso, dx=dx * 0.85, dy=1.0 * rec, deg=6.0 * rec),
                transform_layer(l_talons_r, p_tal_r, dx=dx * 0.85, dy=1.0 * rec, deg=-12.0 * rec),
                transform_layer(l_head, p_head, dx=dx, dy=1.0 * rec, deg=8.0 * rec),
            ],
        )
        rows["hurt"].append(add_hurt_sparks(frame, k))
    return rows


# ==============================================================================
# 8. 0229_houndoom-mega (Quadruped Hellhound + Tall Curved Horns + 4 Distinct Legs)
# ==============================================================================
def rig_houndoom_mega(canvas: np.ndarray, ox: int, oy: int) -> dict[str, list[np.ndarray]]:
    a = canvas[..., 3] > 16
    xx, yy = coord_grids(canvas, ox, oy)

    # Head, orange snout & tall white curved horns (x <= 37, y <= 44)
    head_m = a & (xx <= 37) & (yy <= 44) & (~((xx >= 22) & (yy >= 39) & (xx - yy > -14)))
    # Forked devil tail whipping up on the right (x >= 62, y <= 58)
    tail_m = a & (xx >= 62) & (yy <= 58)
    # 4 distinct canine legs:
    # 1) Front-right leg (viewer's leftmost leg: x <= 31, y >= 65)
    leg_fr_m = a & (xx <= 31) & (yy >= 65)
    # 2) Front-left leg (viewer's second leg: 33 <= x <= 46, y >= 65, plus paw at y>=79, x<=49)
    leg_fl_m = a & (xx >= 33) & (yy >= 65) & ((xx <= 46) | ((yy >= 79) & (xx <= 50)))
    # 3) Hind-right inner leg (viewer's third leg: 46 <= x <= 55, 63 <= y <= 83, excluding leg_fl paw)
    leg_hr_m = a & (~leg_fl_m) & (xx >= 46) & (xx <= 55) & (yy >= 63)
    # 4) Hind-left outer leg (viewer's rightmost leg: x >= 55, y >= 62)
    leg_hl_m = a & (~leg_hr_m) & (xx >= 55) & (yy >= 62)
    # Torso & white bone skull-collar
    torso_m = a & (~head_m) & (~tail_m) & (~leg_fr_m) & (~leg_fl_m) & (~leg_hr_m) & (~leg_hl_m)

    L = build_ordered_layers(
        canvas,
        [
            ("tail", tail_m),
            ("leg_hr", leg_hr_m),
            ("leg_fr", leg_fr_m),
            ("torso", torso_m),
            ("leg_hl", leg_hl_m),
            ("leg_fl", leg_fl_m),
            ("head", head_m),
        ],
    )
    l_tail, l_leg_hr, l_leg_fr, l_torso, l_leg_hl, l_leg_fl, l_head = (
        L["tail"], L["leg_hr"], L["leg_fr"], L["torso"], L["leg_hl"], L["leg_fl"], L["head"]
    )

    p_tail = (63 + ox, 54 + oy)
    p_hr = (50 + ox, 65 + oy)
    p_fr = (26 + ox, 66 + oy)
    p_torso = (44 + ox, 56 + oy)
    p_hl = (58 + ox, 64 + oy)
    p_fl = (40 + ox, 66 + oy)
    p_head = (28 + ox, 40 + oy)

    rows: dict[str, list[np.ndarray]] = {"idle": [], "walk": [], "attack": [], "hurt": []}

    # IDLE (16f): fierce breathing of bone collar, horns poised, forked tail whipping in S-wave
    for k in range(16):
        ph = 2.0 * math.pi * k / 16.0
        bth = math.sin(ph)
        tw = math.sin(ph - 0.7)
        by = round(-1.2 * bth)
        rows["idle"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_tail, p_tail, dy=by, deg=8.0 * tw),
                    transform_layer(l_leg_hr, p_hr),
                    transform_layer(l_leg_fr, p_fr),
                    transform_layer(l_torso, p_torso, dy=by),
                    transform_layer(l_leg_hl, p_hl),
                    transform_layer(l_leg_fl, p_fl),
                    transform_layer(l_head, p_head, dy=by, deg=-2.5 * bth),
                ],
            )
        )

    # WALK (16f): authentic 4-legged diagonal canine trot! Pair A (leg_fr + leg_hl) vs Pair B (leg_fl + leg_hr)
    for k in range(16):
        t = k / 16.0
        dx_a, dy_a, deg_a = stride_arc(t, dx_amp=2.8, lift_amp=2.8)
        dx_b, dy_b, deg_b = stride_arc(t + 0.5, dx_amp=2.8, lift_amp=2.8)
        sw = math.sin(2.0 * math.pi * t)
        bob = round(-1.3 * abs(sw))
        rows["walk"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_tail, p_tail, dx=-0.5, dy=bob, deg=9.0 * sw),
                    transform_layer(l_leg_hr, p_hr, dx=dx_b, dy=dy_b, deg=deg_b),
                    transform_layer(l_leg_fr, p_fr, dx=dx_a, dy=dy_a, deg=deg_a),
                    transform_layer(l_torso, p_torso, dx=-0.5, dy=bob, deg=1.5 * sw),
                    transform_layer(l_leg_hl, p_hl, dx=dx_a, dy=dy_a, deg=deg_a),
                    transform_layer(l_leg_fl, p_fl, dx=dx_b, dy=dy_b, deg=deg_b),
                    transform_layer(l_head, p_head, dx=-1.0, dy=bob, deg=-3.0 * sw),
                ],
            )
        )

    # ATTACK (16f): Canine pounce & horn/fang gore (hind legs drive on ground, front paws claw forward, horns gore!)
    for k in range(16):
        t = k / 15.0
        lunge = interp_keys(t, [(0.0, 0.0), (0.24, 3.0), (0.44, -7.0), (0.66, -5.5), (1.0, 0.0)])
        head_deg = interp_keys(t, [(0.0, 0.0), (0.24, 10.0), (0.44, -14.0), (0.66, -10.0), (1.0, 0.0)])
        front_paw_deg = interp_keys(t, [(0.0, 0.0), (0.24, -10.0), (0.44, 16.0), (0.66, 12.0), (1.0, 0.0)])
        rows["attack"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_tail, p_tail, dx=lunge * 0.5, deg=-head_deg * 1.1),
                    transform_layer(l_leg_hr, p_hr, dx=max(0.0, lunge * 0.3)),
                    transform_layer(l_leg_fr, p_fr, dx=lunge * 0.9, dy=-1.5 if lunge < -2 else 0.0, deg=front_paw_deg),
                    transform_layer(l_torso, p_torso, dx=lunge * 0.75, dy=1.0 if lunge < -2 else 0.0, deg=head_deg * 0.25),
                    transform_layer(l_leg_hl, p_hl, dx=max(0.0, lunge * 0.25)),
                    transform_layer(l_leg_fl, p_fl, dx=lunge * 0.85, dy=-1.0 if lunge < -2 else 0.0, deg=front_paw_deg * 0.85),
                    transform_layer(l_head, p_head, dx=lunge * 1.05, dy=1.5 if lunge < -2 else 0.0, deg=head_deg),
                ],
            )
        )

    # HURT (12f): standing knockback recoil (never squashing the head into a sitting blob!)
    for k in range(12):
        t = k / 11.0
        rec = interp_keys(t, [(0.0, 0.0), (0.18, 1.0), (0.45, 0.7), (0.75, 0.25), (1.0, 0.0)])
        jit = (1.0 if k % 2 == 0 else -1.0) if 1 <= k <= 5 else 0.0
        dx = 4.5 * rec + jit
        frame = composite_layers(
            canvas.shape,
            [
                transform_layer(l_tail, p_tail, dx=dx * 0.6, deg=-12.0 * rec),
                transform_layer(l_leg_hr, p_hr, dx=dx * 0.4),
                transform_layer(l_leg_fr, p_fr, dx=dx * 0.6, dy=-1.0 * rec),
                transform_layer(l_torso, p_torso, dx=dx * 0.75, dy=1.0 * rec, deg=5.0 * rec),
                transform_layer(l_leg_hl, p_hl, dx=dx * 0.35),
                transform_layer(l_leg_fl, p_fl, dx=dx * 0.55),
                transform_layer(l_head, p_head, dx=dx * 0.9, dy=1.0 * rec, deg=9.0 * rec),
            ],
        )
        rows["hurt"].append(add_hurt_sparks(frame, k))
    return rows


# ==============================================================================
# 9. 0248_tyranitar-mega (Biped Kaiju + Swept Dorsal Spikes + Grounded Tail)
# ==============================================================================
def rig_tyranitar_mega(canvas: np.ndarray, ox: int, oy: int) -> dict[str, list[np.ndarray]]:
    a = canvas[..., 3] > 16
    xx, yy = coord_grids(canvas, ox, oy)

    # Right arm (viewer's left clawed arm: x <= 21, 43 <= y <= 64)
    arm_r_m = a & (xx <= 21) & (yy >= 43) & (yy <= 64)
    # Left arm (viewer's center-right clawed arm: 39 <= x <= 60, 44 <= y <= 65)
    arm_l_m = a & (xx >= 39) & (xx <= 60) & (yy >= 44) & (yy <= 65) & (yy <= 46 + (xx - 39) * 1.1)
    # Long swept-back dorsal spikes (x >= 46, 28 <= y <= 69, excluding arm_l)
    spikes_m = a & (~arm_l_m) & (xx >= 46) & (yy >= 28) & (yy <= 69) & (~((xx <= 58) & (yy >= 60)))
    # Grounded segmented kaiju tail (x >= 59, y >= 62, strictly separated from left leg!)
    tail_m = a & (~spikes_m) & (xx >= 59) & (yy >= 62)
    # Right kaiju leg & clawed foot (viewer's left leg: x <= 29, y >= 66)
    leg_r_m = a & (xx <= 29) & (yy >= 66)
    # Left kaiju leg & clawed foot (viewer's center-right leg: 38 <= x <= 60, y >= 63, excluding tail)
    leg_l_m = a & (~tail_m) & (~arm_l_m) & (xx >= 38) & (xx <= 60) & (yy >= 63)
    # Kaiju head, roaring jaw & tall horn (y <= 35, 17 <= x <= 48)
    head_m = a & (yy <= 35) & (xx >= 17) & (xx <= 48)
    torso_m = a & (~arm_r_m) & (~arm_l_m) & (~spikes_m) & (~tail_m) & (~leg_r_m) & (~leg_l_m) & (~head_m)

    L = build_ordered_layers(
        canvas,
        [
            ("tail", tail_m),
            ("spikes", spikes_m),
            ("arm_r", arm_r_m),
            ("leg_r", leg_r_m),
            ("torso", torso_m),
            ("leg_l", leg_l_m),
            ("head", head_m),
            ("arm_l", arm_l_m),
        ],
    )
    l_tail, l_spikes, l_arm_r, l_leg_r, l_torso, l_leg_l, l_head, l_arm_l = (
        L["tail"], L["spikes"], L["arm_r"], L["leg_r"], L["torso"], L["leg_l"], L["head"], L["arm_l"]
    )

    p_tail = (60 + ox, 75 + oy)
    p_spikes = (46 + ox, 44 + oy)
    p_arm_r = (20 + ox, 49 + oy)
    p_leg_r = (21 + ox, 68 + oy)
    p_torso = (35 + ox, 58 + oy)
    p_leg_l = (49 + ox, 66 + oy)
    p_head = (33 + ox, 33 + oy)
    p_arm_l = (42 + ox, 48 + oy)

    rows: dict[str, list[np.ndarray]] = {"idle": [], "walk": [], "attack": [], "hurt": []}

    # IDLE (16f): heavy kaiju breathing, dorsal spikes & arms pulsing, tail grounded & steady
    for k in range(16):
        ph = 2.0 * math.pi * k / 16.0
        bth = math.sin(ph)
        lag = math.sin(ph - 0.6)
        by = round(-1.2 * bth)
        rows["idle"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_tail, p_tail, deg=2.5 * lag),
                    transform_layer(l_spikes, p_spikes, dy=by, deg=3.0 * lag),
                    transform_layer(l_arm_r, p_arm_r, dy=by, deg=4.5 * bth),
                    transform_layer(l_leg_r, p_leg_r),
                    transform_layer(l_torso, p_torso, dy=by),
                    transform_layer(l_leg_l, p_leg_l),
                    transform_layer(l_head, p_head, dy=by, deg=-2.0 * bth),
                    transform_layer(l_arm_l, p_arm_l, dy=by, deg=-4.5 * bth),
                ],
            )
        )

    # WALK (16f): earth-shaking kaiju stomp! Only leg_r and leg_l step; tail stays grounded as counterbalance!
    for k in range(16):
        t = k / 16.0
        dx_r, dy_r, deg_r = stride_arc(t, dx_amp=2.6, lift_amp=3.2)
        dx_l, dy_l, deg_l = stride_arc(t + 0.5, dx_amp=2.6, lift_amp=3.2)
        sw = math.sin(2.0 * math.pi * t)
        bob = round(-1.5 * abs(sw))
        rows["walk"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_tail, p_tail, dx=-0.4, deg=3.5 * sw),
                    transform_layer(l_spikes, p_spikes, dx=-0.6, dy=bob, deg=3.5 * sw),
                    transform_layer(l_arm_r, p_arm_r, dx=-0.6 - 1.2 * sw, dy=bob, deg=-9.0 * sw),
                    transform_layer(l_leg_r, p_leg_r, dx=dx_r, dy=dy_r, deg=deg_r),
                    transform_layer(l_torso, p_torso, dx=-0.6, dy=bob, deg=1.8 * sw),
                    transform_layer(l_leg_l, p_leg_l, dx=dx_l, dy=dy_l, deg=deg_l),
                    transform_layer(l_head, p_head, dx=-0.9, dy=bob, deg=-2.5 * sw),
                    transform_layer(l_arm_l, p_arm_l, dx=-0.6 + 1.5 * sw, dy=bob, deg=10.0 * sw),
                ],
            )
        )

    # ATTACK (16f): Kaiju wind-up roar -> massive forward body-slam + double claw crush!
    for k in range(16):
        t = k / 15.0
        lunge = interp_keys(t, [(0.0, 0.0), (0.26, 3.5), (0.45, -7.5), (0.66, -5.5), (1.0, 0.0)])
        tilt = interp_keys(t, [(0.0, 0.0), (0.26, 7.0), (0.45, -9.0), (0.66, -6.5), (1.0, 0.0)])
        arm_deg = interp_keys(t, [(0.0, 0.0), (0.26, 14.0), (0.45, -18.0), (0.66, -13.0), (1.0, 0.0)])
        drop = interp_keys(t, [(0.0, 0.0), (0.26, -1.0), (0.45, 2.0), (0.66, 1.0), (1.0, 0.0)])
        rows["attack"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_tail, p_tail, dx=lunge * 0.5, deg=-tilt * 0.5),
                    transform_layer(l_spikes, p_spikes, dx=lunge * 0.75, dy=drop, deg=tilt * 0.5),
                    transform_layer(l_arm_r, p_arm_r, dx=lunge * 0.95, dy=drop, deg=arm_deg),
                    transform_layer(l_leg_r, p_leg_r, dx=min(0.0, lunge * 0.5)),
                    transform_layer(l_torso, p_torso, dx=lunge * 0.75, dy=drop, deg=tilt * 0.35),
                    transform_layer(l_leg_l, p_leg_l, dx=max(0.0, lunge * 0.25)),
                    transform_layer(l_head, p_head, dx=lunge * 0.9, dy=drop, deg=tilt * 0.7),
                    transform_layer(l_arm_l, p_arm_l, dx=lunge * 0.85, dy=drop, deg=arm_deg),
                ],
            )
        )

    # HURT (12f)
    for k in range(12):
        t = k / 11.0
        rec = interp_keys(t, [(0.0, 0.0), (0.18, 1.0), (0.45, 0.7), (0.75, 0.25), (1.0, 0.0)])
        jit = (1.0 if k % 2 == 0 else -1.0) if 1 <= k <= 5 else 0.0
        dx = 4.5 * rec + jit
        frame = composite_layers(
            canvas.shape,
            [
                transform_layer(l_tail, p_tail, dx=dx * 0.45, deg=-5.0 * rec),
                transform_layer(l_spikes, p_spikes, dx=dx * 0.85, dy=1.0 * rec, deg=8.0 * rec),
                transform_layer(l_arm_r, p_arm_r, dx=dx, dy=1.0 * rec, deg=-10.0 * rec),
                transform_layer(l_leg_r, p_leg_r, dx=dx * 0.45),
                transform_layer(l_torso, p_torso, dx=dx * 0.75, dy=1.5 * rec, deg=5.0 * rec),
                transform_layer(l_leg_l, p_leg_l, dx=dx * 0.35),
                transform_layer(l_head, p_head, dx=dx * 0.9, dy=1.5 * rec, deg=8.0 * rec),
                transform_layer(l_arm_l, p_arm_l, dx=dx * 0.9, dy=1.5 * rec, deg=-12.0 * rec),
            ],
        )
        rows["hurt"].append(add_hurt_sparks(frame, k))
    return rows


# ==============================================================================
# Generic Anatomical Classifier & Part-Rig for Any Future Monster
# ==============================================================================
def rig_generic_anatomical(canvas: np.ndarray, ox: int, oy: int) -> dict[str, list[np.ndarray]]:
    """Automatic anatomical part segmentation (head, arms/wings, left/right legs, tail, torso)
    so any future monster gets clean part-isolated locomotion and attack motions."""
    a = canvas[..., 3] > 16
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bh = max(1, y1 - y0 + 1)
    bw = max(1, x1 - x0 + 1)
    cx = float(xs.mean())
    yy, xx = np.mgrid[0 : canvas.shape[0], 0 : canvas.shape[1]]

    # Detect bottom ground-contact lobes (legs vs tail)
    leg_top = y0 + int(bh * 0.70)
    bot_mask = a & (yy >= leg_top)
    labeled, n_lobes = ndi.label(bot_mask)
    lobe_xs = []
    for i in range(1, n_lobes + 1):
        m = labeled == i
        if m.sum() >= 6:
            lobe_xs.append((float(np.nonzero(m)[1].mean()), m))
    lobe_xs.sort(key=lambda t: t[0])

    tail_m = a & (xx >= cx + bw * 0.24) & (yy >= y0 + int(bh * 0.45))
    leg_r_m = a & (~tail_m) & (yy >= leg_top) & (xx < cx)
    leg_l_m = a & (~tail_m) & (yy >= leg_top) & (xx >= cx)
    head_m = a & (yy <= y0 + int(bh * 0.36)) & (xx <= cx + bw * 0.18)
    arm_r_m = a & (~head_m) & (~leg_r_m) & (xx < cx - bw * 0.16) & (yy >= y0 + int(bh * 0.34)) & (yy < leg_top)
    arm_l_m = a & (~head_m) & (~leg_l_m) & (~tail_m) & (xx > cx + bw * 0.12) & (yy >= y0 + int(bh * 0.34)) & (yy < leg_top)
    torso_m = a & (~tail_m) & (~leg_r_m) & (~leg_l_m) & (~head_m) & (~arm_r_m) & (~arm_l_m)

    L = build_ordered_layers(
        canvas,
        [
            ("tail", tail_m),
            ("arm_r", arm_r_m),
            ("leg_r", leg_r_m),
            ("torso", torso_m),
            ("leg_l", leg_l_m),
            ("head", head_m),
            ("arm_l", arm_l_m),
        ],
    )
    l_tail, l_arm_r, l_leg_r, l_torso, l_leg_l, l_head, l_arm_l = (
        L["tail"], L["arm_r"], L["leg_r"], L["torso"], L["leg_l"], L["head"], L["arm_l"]
    )

    p_tail = (cx + bw * 0.2, y0 + bh * 0.65)
    p_arm_r = (cx - bw * 0.15, y0 + bh * 0.45)
    p_leg_r = (cx - bw * 0.12, leg_top)
    p_torso = (cx, y0 + bh * 0.55)
    p_leg_l = (cx + bw * 0.10, leg_top)
    p_head = (cx - bw * 0.05, y0 + bh * 0.32)
    p_arm_l = (cx + bw * 0.12, y0 + bh * 0.45)

    rows: dict[str, list[np.ndarray]] = {"idle": [], "walk": [], "attack": [], "hurt": []}
    for k in range(16):
        ph = 2.0 * math.pi * k / 16.0
        bth = math.sin(ph)
        by = round(-1.2 * bth)
        rows["idle"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_tail, p_tail, dy=by, deg=5.0 * bth),
                    transform_layer(l_arm_r, p_arm_r, dy=by, deg=4.0 * bth),
                    transform_layer(l_leg_r, p_leg_r),
                    transform_layer(l_torso, p_torso, dy=by),
                    transform_layer(l_leg_l, p_leg_l),
                    transform_layer(l_head, p_head, dy=by, deg=-2.0 * bth),
                    transform_layer(l_arm_l, p_arm_l, dy=by, deg=-4.0 * bth),
                ],
            )
        )
    for k in range(16):
        t = k / 16.0
        dx_r, dy_r, deg_r = stride_arc(t, dx_amp=2.5, lift_amp=2.6)
        dx_l, dy_l, deg_l = stride_arc(t + 0.5, dx_amp=2.5, lift_amp=2.6)
        sw = math.sin(2.0 * math.pi * t)
        bob = round(-1.2 * abs(sw))
        rows["walk"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_tail, p_tail, dx=-0.5, dy=bob, deg=6.0 * sw),
                    transform_layer(l_arm_r, p_arm_r, dx=-0.5, dy=bob, deg=-9.0 * sw),
                    transform_layer(l_leg_r, p_leg_r, dx=dx_r, dy=dy_r, deg=deg_r),
                    transform_layer(l_torso, p_torso, dx=-0.5, dy=bob),
                    transform_layer(l_leg_l, p_leg_l, dx=dx_l, dy=dy_l, deg=deg_l),
                    transform_layer(l_head, p_head, dx=-0.8, dy=bob, deg=-2.5 * sw),
                    transform_layer(l_arm_l, p_arm_l, dx=-0.5, dy=bob, deg=9.0 * sw),
                ],
            )
        )
    for k in range(16):
        t = k / 15.0
        lunge = interp_keys(t, [(0.0, 0.0), (0.25, 3.0), (0.45, -6.5), (0.66, -5.0), (1.0, 0.0)])
        deg = interp_keys(t, [(0.0, 0.0), (0.25, -12.0), (0.45, 16.0), (0.66, 12.0), (1.0, 0.0)])
        rows["attack"].append(
            composite_layers(
                canvas.shape,
                [
                    transform_layer(l_tail, p_tail, dx=lunge * 0.4, deg=-deg * 0.5),
                    transform_layer(l_arm_r, p_arm_r, dx=lunge * 1.1, deg=deg),
                    transform_layer(l_leg_r, p_leg_r, dx=min(0.0, lunge * 0.5)),
                    transform_layer(l_torso, p_torso, dx=lunge * 0.7, deg=-deg * 0.25),
                    transform_layer(l_leg_l, p_leg_l, dx=max(0.0, lunge * 0.25)),
                    transform_layer(l_head, p_head, dx=lunge * 0.95, deg=-deg * 0.4),
                    transform_layer(l_arm_l, p_arm_l, dx=lunge * 1.1, deg=deg),
                ],
            )
        )
    for k in range(12):
        t = k / 11.0
        rec = interp_keys(t, [(0.0, 0.0), (0.18, 1.0), (0.45, 0.7), (0.75, 0.25), (1.0, 0.0)])
        jit = (1.0 if k % 2 == 0 else -1.0) if 1 <= k <= 5 else 0.0
        dx = 4.5 * rec + jit
        frame = composite_layers(
            canvas.shape,
            [
                transform_layer(l_tail, p_tail, dx=dx * 0.5, deg=-8.0 * rec),
                transform_layer(l_arm_r, p_arm_r, dx=dx, deg=-10.0 * rec),
                transform_layer(l_leg_r, p_leg_r, dx=dx * 0.45),
                transform_layer(l_torso, p_torso, dx=dx * 0.75, dy=1.0 * rec, deg=5.0 * rec),
                transform_layer(l_leg_l, p_leg_l, dx=dx * 0.35),
                transform_layer(l_head, p_head, dx=dx * 0.9, dy=1.0 * rec, deg=8.0 * rec),
                transform_layer(l_arm_l, p_arm_l, dx=dx * 0.9, deg=-10.0 * rec),
            ],
        )
        rows["hurt"].append(add_hurt_sparks(frame, k))
    return rows


RIG_BUILDERS = {
    "0154_meganium-mega": ("quadruped_beast", rig_meganium_mega),
    "0160_feraligatr-mega": ("biped_jaw_slasher", rig_feraligatr_mega),
    "0181_ampharos-mega": ("biped_tail_mage", rig_ampharos_mega),
    "0208_steelix-mega": ("serpent_coil_glider", rig_steelix_mega),
    "0212_scizor-mega": ("biped_pincer_striker", rig_scizor_mega),
    "0214_heracross-mega": ("biped_horn_grappler", rig_heracross_mega),
    "0227_skarmory-mega": ("avian_raptor_hover", rig_skarmory_mega),
    "0229_houndoom-mega": ("quadruped_beast", rig_houndoom_mega),
    "0248_tyranitar-mega": ("biped_kaiju_stomper", rig_tyranitar_mega),
}


def score_with_forge(frames: list[np.ndarray], fps_num: int, kind: str) -> dict:
    """Validate and score generated animation loops using agent-sprite-forge's `gait_loop.py`."""
    with tempfile.TemporaryDirectory(prefix=f"forge-{kind}-") as tmp:
        raw = Path(tmp) / "raw"
        raw.mkdir(parents=True)
        for i, fr in enumerate(frames):
            Image.fromarray(fr, "RGBA").save(raw / f"frame_{i:04d}.png")
        try:
            analysis = gait_loop.analyse(
                raw,
                gait_loop.parse_fps(f"{fps_num}/1"),
                kind="idle",
                state=kind,
                dedupe_cap=0,
                min_seconds=0.3,
                max_seconds=3.0,
            )
            rec = gait_loop.recommend(analysis, "auto")
            w = rec["window"]
            return {
                "policy": rec.get("policy"),
                "start": int(w["start"]),
                "endExclusive": int(w["endExclusive"]),
                "score": round(float(w.get("score", 0.0)), 4),
            }
        except Exception as exc:
            return {"policy": "full", "note": type(exc).__name__}


def build_form(fid: str, form_dir: Path, out_dir: Path) -> dict:
    src96 = np.array(Image.open(form_dir / "front.png").convert("RGBA"))
    a = src96[..., 3] > 16
    ys, xs = np.nonzero(a)
    crop_box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    bh = crop_box[3] - crop_box[1] + 1

    # Extract locked palette (generate2dsprite.palette.v1)
    rgb = src96[a][:, :3]
    uniq = np.unique(rgb, axis=0)
    palette_hex = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in uniq]

    canvas, ox, oy = place_in_padded_canvas(src96, crop_box)
    ch, cw = canvas.shape[:2]

    archetype, builder = RIG_BUILDERS.get(fid, ("generic_anatomical", rig_generic_anatomical))
    rows = builder(canvas, ox, oy)

    # Validate with agent-sprite-forge gait_loop
    forge_scores = {
        "idle": score_with_forge(rows["idle"], 12, "idle"),
        "walk": score_with_forge(rows["walk"], 15, "walk"),
    }

    order = [act[0] for act in ACTIONS]
    ncol = max(len(rows[k]) for k in order)
    sheet = np.zeros((ch * len(order), cw * ncol, 4), dtype=np.uint8)
    for ri, key in enumerate(order):
        for fi, fr in enumerate(rows[key]):
            sheet[ri * ch : (ri + 1) * ch, fi * cw : (fi + 1) * cw] = fr

    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_img = Image.fromarray(sheet, "RGBA")
    sheet_img.save(out_dir / "sheet.png", optimize=True)

    pv = sheet_img.resize((sheet.shape[1] * 2, sheet.shape[0] * 2), Image.Resampling.NEAREST)
    bg = Image.new("RGB", pv.size, (16, 20, 32))
    bg.paste(pv, mask=pv.split()[-1])
    bg.save(out_dir / "preview.png")

    idle_n, idle_d = ACTIONS[0][1], ACTIONS[0][2]
    dir8_entry = {
        "layout": "actions",
        "cw": int(cw),
        "ch": int(ch),
        "h": int(bh),
        "n": [idle_n] * 8,
        "d": [[idle_d] * idle_n for _ in range(8)],
        "an": [a_spec[1] for a_spec in ACTIONS],
        "ad": [a_spec[2] for a_spec in ACTIONS],
        "raw": 1,
        "arow": 0,
        "src": "agent-sprite-forge-anatomical",
        "archetype": archetype,
        "facing": "left",
    }
    meta = dict(dir8_entry)
    meta["_paletteCount"] = len(palette_hex)
    meta["_forgeLoop"] = forge_scores
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n")
    return dir8_entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="Comma-separated form IDs")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))

    raw = DIR8.read_text()
    at = raw.index("window.ARENA_DIR8")
    head = raw[:at] + "window.ARENA_DIR8"
    body = raw[at:].split("=", 1)[1]
    man = json.loads(body.strip().rstrip(";"))

    for form in sorted(GEN2.glob("*/mega/*")):
        if not form.is_dir() or not (form / "front.png").exists():
            continue
        fid = form.parent.parent.name.split("_")[0] + "_" + form.name
        if only and fid not in only:
            continue
        entry = build_form(fid, form, form / "actions")
        man[fid] = entry
        print(
            f"[OK] {fid:22s} archetype={entry['archetype']:22s} cell={entry['cw']}x{entry['ch']} an={entry['an']}",
            flush=True,
        )

    DIR8.write_text(head.rstrip() + " = " + json.dumps(man, separators=(",", ":")) + ";\n")


if __name__ == "__main__":
    main()
