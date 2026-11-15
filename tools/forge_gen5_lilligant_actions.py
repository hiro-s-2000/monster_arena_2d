#!/usr/bin/env python3
"""Lilligant normal (0549) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x21-72 y14-80, 2123 px, 15 colours,
alpha 0/255, sole y80):
  Two eyes, no mouth. Iris pixels are the brown, orange and rust dots that
  touch the face white: left eye x40-42 y36-38 (mean x 41), right eye
  x48-50 y36-39 (mean x 49). Fourteen pixels, mean x 45.0.
  The chin is the cream row y41 x44-48, mean x 46.0, right of the eyes.
  The still faces screen-right. Pink (255, 180, 156) is only in the petals
  (y19-30), and the cream has no slit, so the roar does not invent a mouth,
  a wedge, or a ring. It lifts the flower and the two side leaves.
  The flower is the orange bloom x39-64 y14-35. The left leaf is the pointed
  dark-green lobe x21-42 y38-49. The right leaf is the dark sleeve
  x48-72 y46-73. Feet split at y74: the long sole reaches y80 at x44-46,
  the short toe ends at y77 on the right.
normal/back.png (bbox x20-73 y13-79, 2128 px, 14 colours, all in the front
  palette, sole y79): no eyes, no cream, no mouth. The bloom mean x is left
  of the torso mean x, so this back faces away to the left. One pixel of
  drop puts its sole on the same ground as the front. Same leaves and feet.
  No wings, horns, fangs, tail, hands, or pincers are drawn.

Directions, front faces right, back faces away-left:
  S SE E front as drawn (screen-right). W SW front mirrored (screen-left).
  N NW back as drawn. NE back mirrored.
  E is screen-right, W is screen-left, N is the back.
  Attack 5-8 is that back at equal scale. S SE E show N. W SW show N
  flipped. Back rows stay the back. No |cos| squash and no in-plane spin.

  idle 16: feet planted, flower and leaves breathe. Frame 0 is the still.
  walk 16: the two drawn feet step. Leaves and the flower sway.
  attack: 0 still, 1-4 flower and leaves rise (no mouth), 5-8 the back
          still, 9 wind-up, 10-12 the leaves, the flower and the feet
          strike, 11-12 impact, 13-15 return.
  hurt 12: white flash, knock-back lean (<=38 deg) and recover.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_lilligant_actions.py
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen5_attack_strike import finish, save_peaks

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen5/0549_lilligant/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0549_lilligant/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0549_lilligant"
VERSION = "lilligant_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 208, 208
OFF_X = (CELL_W - 96) // 2
OFF_Y = 80
FRONT_FOOT = 80
BACK_FOOT = 79
BACK_DROP = 1
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-right. Back as drawn looks away to the left.
VIEW_OF = {
    "S": ("front", False), "SE": ("front", False), "E": ("front", False),
    "NE": ("back", True),
    "N": ("back", False), "NW": ("back", False),
    "W": ("front", True), "SW": ("front", True),
}
MAX_LEAN = 38.0
# Wind-up ceiling. The strike is drawn up to 34, above this, so finish()
# does not push a frame past the angle this script already paints.
# Frames 5-8 are cleared after finish() so the back still is not swung.
PEAKS = {"bloom": 14.0, "leaf": 16.0, "kick": 10.0}
LIMBS = ("bloom", "leaf_l", "leaf_r", "leg_l", "leg_r")

INK = (16, 16, 16)
LIME = (197, 238, 123)
GREEN = (57, 172, 57)
OLIVE = (156, 180, 98)
ORANGE = (255, 106, 57)
DG = (74, 131, 74)
FOREST = (49, 90, 49)
BROWN = (115, 65, 32)
VINE = (82, 98, 41)
WHITE = (255, 255, 255)
RUST = (189, 98, 57)
GOLD = (255, 222, 65)
TAN = (189, 164, 82)
CREAM = (205, 197, 189)
PINK = (255, 180, 156)
PETAL = (ORANGE, RUST, PINK)
DARK = (GREEN, DG, FOREST)
CROWN = (GOLD, TAN, BROWN, WHITE, INK)
EYE_COLS = (BROWN, ORANGE, RUST)

SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)


def is_color(img, color):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(color, np.uint8), axis=-1)


def any_color(img, cols):
    m = np.zeros(img.shape[:2], bool)
    for c in cols:
        m |= is_color(img, c)
    return m


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1, constant_values=False)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def shift(layer, dx, dy):
    dx, dy = int(round(dx)), int(round(dy))
    if dx == 0 and dy == 0:
        return layer.copy()
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = layer[sy0:sy1, sx0:sx1]
    return out


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest in-plane rotation. Positive is clockwise on screen. No squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    sc = 4
    up = np.repeat(np.repeat(layer, sc, axis=0), sc, axis=1)
    h, w = up.shape[:2]
    px, py = pivot[0] * sc + 1.5, pivot[1] * sc + 1.5
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = xx - px, yy - py
    r = math.radians(ang)
    c, s = math.cos(r), math.sin(r)
    sx = np.rint(px + c * rx + s * ry).astype(np.int32)
    sy = np.rint(py - s * rx + c * ry).astype(np.int32)
    ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    out = np.zeros_like(up)
    out[ok] = up[sy[ok], sx[ok]]
    return out[sc // 2::sc, sc // 2::sc].copy()


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1, constant_values=False)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img):
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        p = np.pad(a, 1, constant_values=False)
        n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        hole = (~a) & (n == 4)
        if not hole.any():
            break
        up = np.zeros_like(out)
        up[1:] = out[:-1]
        out[hole] = up[hole]
    return out


def snap(img, palette):
    out = img.copy()
    op = out[..., 3] >= 128
    out[~op] = 0
    if not op.any():
        return out
    px = out[op][:, :3].astype(np.int32)
    pal = palette.astype(np.int32)
    d = ((px[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    out[op, :3] = palette[d.argmin(axis=1)]
    out[op, 3] = 255
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def put(img, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 1 <= x < w - 1 and 1 <= y < h - 1 and img[y, x, 3] < 128:
        img[y, x, :3] = col
        img[y, x, 3] = 255


def paint_cushion(layer, allowed, steps):
    """Grow this layer's own colours into allowed pixels. Hidden at rest."""
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        up = np.zeros_like(src)
        dn = np.zeros_like(src)
        lt = np.zeros_like(src)
        rt = np.zeros_like(src)
        up[:-1] = src[1:]
        dn[1:] = src[:-1]
        lt[:, :-1] = src[:, 1:]
        rt[:, 1:] = src[:, :-1]
        for g in (up, dn, lt, rt):
            take = (acc[..., 3] < 128) & allowed & (g[..., 3] >= 128)
            acc[take] = g[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def component_list(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    groups = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        groups.append(pts)
    return groups


def largest_component(mask):
    groups = component_list(mask)
    if not groups:
        return 0
    return max(len(g) for g in groups)


def check_partition(op, masks):
    acc = np.zeros_like(op)
    for name, m in masks.items():
        if (m & acc).any():
            raise SystemExit(f"overlap in {name}")
        if (m & ~op).any():
            raise SystemExit(f"{name} left the still")
        acc |= m
    if int(acc.sum()) != int(op.sum()):
        raise SystemExit(f"parts do not partition the still: {int(acc.sum())} vs {int(op.sum())}")


def touches(part, body):
    return bool((part & dilate(body, 1)).any())


def flood_from(seed, allow):
    out = np.zeros_like(allow, bool)
    seen = np.zeros_like(allow, bool)
    ys, xs = np.nonzero(seed & allow)
    stack = [(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())]
    for y, x in stack:
        seen[y, x] = True
    while stack:
        y, x = stack.pop()
        out[y, x] = True
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < 96 and 0 <= nx < 96 and allow[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    return out


def absorb_touching(mask, extra):
    out = mask.copy()
    while True:
        add = dilate(out, 1) & extra & ~out
        if not add.any():
            return out
        out |= add


def absorb_specks(masks):
    """Tiny torso islands that touch one limb belong to that limb."""
    names = [n for n in LIMBS if n in masks]
    torso = masks["torso"]
    changed = True
    while changed:
        changed = False
        for g in component_list(torso):
            if len(g) > 8:
                continue
            speck = np.zeros_like(torso)
            for y, x in g:
                speck[y, x] = True
            hits = [n for n in names if touches(speck, masks[n])]
            if len(hits) != 1:
                continue
            masks[hits[0]] |= speck
            torso = torso & ~speck
            changed = True
            break
    masks["torso"] = torso


def eye_mask(img):
    """Brown, orange and rust dots inside the white face. Not the petals."""
    yy, xx = np.mgrid[0:96, 0:96]
    white = is_color(img, WHITE)
    cream = is_color(img, CREAM)
    face_w = white & (yy >= 34) & (yy <= 40) & (xx >= 38) & (xx <= 52)
    iris = any_color(img, EYE_COLS) & (yy >= 35) & (yy <= 40) & (xx >= 38) & (xx <= 52)
    return iris & dilate(face_w | (cream & (yy <= 41)), 1)


def face_ban(img, eyes):
    yy = np.mgrid[0:96, 0:96][0]
    cream = is_color(img, CREAM) & (yy <= 42)
    white = is_color(img, WHITE) & (yy >= 34) & (yy <= 41)
    return dilate(cream | white | eyes, 1)


def validate_parts(kind, op, masks, foot):
    check_partition(op, masks)
    torso = masks["torso"]
    if largest_component(torso) != int(torso.sum()):
        sizes = sorted((len(g) for g in component_list(torso)), reverse=True)
        raise SystemExit(f"{kind} torso fell apart {sizes[:6]}")
    for name in LIMBS:
        part = masks[name]
        if int(part.sum()) < 8:
            raise SystemExit(f"{kind} {name} is too small: {int(part.sum())}")
        if not touches(part, torso):
            raise SystemExit(f"{kind} {name} does not touch the body")
        if largest_component(part) != int(part.sum()):
            sizes = sorted((len(g) for g in component_list(part)), reverse=True)
            raise SystemExit(f"{kind} {name} fell apart {sizes[:4]}")
    yy = np.mgrid[0:96, 0:96][0]
    grounded = op & (yy == foot)
    feet = masks["leg_l"] | masks["leg_r"]
    if int(grounded.sum()) < 1 or int((grounded & feet).sum()) != int(grounded.sum()):
        raise SystemExit(f"{kind} ground line is not a foot")


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    eyes = eye_mask(img)
    if int(eyes.sum()) != 14:
        raise SystemExit(f"front eyes changed: {int(eyes.sum())}")
    ex = float(np.nonzero(eyes)[1].mean())
    chin = is_color(img, CREAM) & (yy == 41)
    cx = float(np.nonzero(chin)[1].mean())
    if not (cx > ex):
        raise SystemExit(f"front does not face right: chin {cx} eye {ex}")
    petal = any_color(img, PETAL) & ~eyes & (yy <= 36)
    crown = any_color(img, CROWN) & (yy <= 32) & (xx >= 32) & ~eyes
    bloom = flood_from(petal, petal | crown)
    bloom = absorb_touching(bloom, any_color(img, DARK) & (yy <= 35) & ~eyes)
    ban = face_ban(img, eyes)
    ink = is_color(img, INK)
    dark = any_color(img, DARK)
    allow_l = ((dark | ink) & (xx <= 42) & (yy >= 36) & (yy <= 50) & ~bloom & ~ban)
    seed_l = dark & (xx <= 30) & (yy >= 36) & (yy <= 48)
    leaf_l = flood_from(seed_l, allow_l)
    allow_r = ((dark | ink) & (xx >= 48) & (yy >= 46) & ~bloom & ~leaf_l & ~ban)
    seed_r = dark & (xx >= 58) & (yy >= 50)
    leaf_r = flood_from(seed_r, allow_r)
    leg = op & (yy >= 74) & ~bloom & ~leaf_l & ~leaf_r
    leg_l = leg & (xx < 50)
    leg_r = leg & (xx >= 50)
    masks = {
        "bloom": bloom, "leaf_l": leaf_l, "leaf_r": leaf_r,
        "leg_l": leg_l, "leg_r": leg_r,
    }
    used = np.zeros_like(op)
    for m in masks.values():
        used |= m
    masks["torso"] = op & ~used
    absorb_specks(masks)
    if int((eyes & masks["torso"]).sum()) != int(eyes.sum()):
        raise SystemExit("front eyes left the face")
    validate_parts("front", op, masks, FRONT_FOOT)
    return masks, ex, cx


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    if any_color(img, (CREAM,)).any() or eye_mask_back(img):
        raise SystemExit("back gained a face")
    petal = any_color(img, PETAL)
    crown = any_color(img, CROWN) & (yy <= 34)
    bloom = flood_from(petal, petal | crown)
    bloom = absorb_touching(bloom, any_color(img, DARK) & (yy <= 36))
    ink = is_color(img, INK)
    dark = any_color(img, DARK)
    allow_l = (dark | ink) & (xx <= 40) & (yy >= 46) & ~bloom
    seed_l = dark & (xx <= 34) & (yy >= 48)
    leaf_l = flood_from(seed_l, allow_l)
    allow_r = (dark | ink) & (xx >= 50) & (yy >= 46) & ~bloom & ~leaf_l
    seed_r = dark & (xx >= 60) & (yy >= 48)
    leaf_r = flood_from(seed_r, allow_r)
    leg = op & (yy >= 74) & ~bloom & ~leaf_l & ~leaf_r
    leg_l = leg & (xx < 52)
    leg_r = leg & (xx >= 52)
    masks = {
        "bloom": bloom, "leaf_l": leaf_l, "leaf_r": leaf_r,
        "leg_l": leg_l, "leg_r": leg_r,
    }
    used = np.zeros_like(op)
    for m in masks.values():
        used |= m
    masks["torso"] = op & ~used
    absorb_specks(masks)
    by, bx = np.nonzero(masks["torso"])
    fy, fx = np.nonzero(masks["bloom"])
    if float(fx.mean()) >= float(bx.mean()):
        raise SystemExit(
            f"back bloom is not left of the body: {float(fx.mean())} vs {float(bx.mean())}"
        )
    validate_parts("back", op, masks, BACK_FOOT)
    return masks, float(fx.mean()), float(bx.mean())


def eye_mask_back(img):
    """Back has no face white cluster. True only if a front-like eye appears."""
    return bool(is_color(img, CREAM).any())


def place(arr, dy=0):
    if arr.ndim == 2:
        out = np.zeros((CELL_H, CELL_W), bool)
        y0 = OFF_Y + dy
        out[y0:y0 + 96, OFF_X:OFF_X + 96] = arr
        return out
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    y0 = OFF_Y + dy
    out[y0:y0 + 96, OFF_X:OFF_X + 96] = arr
    return out


def joint_band(part, torso, depth):
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 3)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def pivot_of(part, torso):
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 4)
    if not touch.any():
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(torso)
    cx, cy = float(tx.mean()), float(ty.mean())
    d = (xs.astype(np.float64) - cx) ** 2 + (ys.astype(np.float64) - cy) ** 2
    near = d <= d.min() + 2.0
    return float(xs[near].mean()), float(ys[near].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs.astype(np.float64) - pivot[0]) ** 2 + (ys.astype(np.float64) - pivot[1]) ** 2
    i = int(d.argmax())
    return float(xs[i]), float(ys[i])


def signs_for(layer, pivot, facing):
    """Which rotation sign lifts the tip, and which sends it toward facing."""
    if int((layer[..., 3] >= 128).sum()) < 4:
        return {"up": 1.0, "fwd": 1.0}

    def tip_after(ang):
        return tip_xy(rotate_nearest(layer, ang, pivot), pivot)

    x_pos, y_pos = tip_after(12.0)
    x_neg, y_neg = tip_after(-12.0)
    up = -1.0 if y_neg < y_pos else 1.0
    if facing > 0:
        fwd = 1.0 if x_pos >= x_neg else -1.0
    else:
        fwd = 1.0 if x_pos <= x_neg else -1.0
    return {"up": up, "fwd": fwd}


def build_view(img, masks, kind, mirror):
    dy = 0 if kind == "front" else BACK_DROP
    canvas = place(img, dy)
    placed = {k: place(v, dy) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    if kind == "front":
        facing = 1.0 if not mirror else -1.0
    else:
        facing = -1.0 if not mirror else 1.0
    pivots = {}
    signs = {}
    depth = {"bloom": 8, "leaf_l": 6, "leaf_r": 6, "leg_l": 5, "leg_r": 5}
    for name in LIMBS:
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth[name]), depth[name],
        )
        pivots[name] = pivot_of(placed[name], torso_m)
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "has_mouth": False,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        bloom=0.0, leaf=0.0, kick=0.0, mode="up",
        step_l=(0, 0), step_r=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0, head_dx=0, head_dy=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay on the ground. The flower and the side leaves breathe.
        p["bloom"] = wave(f, n, 5.0, phase=0.4)
        p["leaf"] = wave(f, n, 6.0, phase=1.1)
        p["mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(3 * sa)), -int(round(4 * max(0.0, sa))))
        p["step_r"] = (int(round(3 * sb)), -int(round(4 * max(0.0, sb))))
        p["leaf"] = wave(f, n, 8.0, cycles=2.0, phase=0.2)
        p["bloom"] = wave(f, n, 4.0, cycles=2.0, phase=0.8)
        p["mode"] = "fwd"
    elif action == "attack":
        # 0 rest. 1-4 raise the flower and the leaves. No mouth.
        # 5-8 the back, equal scale, cleared after finish().
        # 9 wind-up. 10-12 strike with the leaves, the flower and the feet.
        # 13-15 return. 15 is the still.
        bloom = [0, 8, 12, 16, 18, 0, 0, 0, 0, 14, 24, 30, 30, 10, 4, 0]
        leaf = [0, 8, 14, 18, 20, 0, 0, 0, 0, 18, 28, 34, 34, 12, 4, 0]
        kick = [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 22, 34, 34, 12, 4, 0]
        bdx = [0, -1, -1, -1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        p["bloom"] = float(bloom[f])
        p["leaf"] = float(leaf[f])
        p["kick"] = float(kick[f])
        p["bdx"] = int(bdx[f])
        p["mode"] = "up" if f <= 4 else "fwd"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["bloom"] = 8.0 * env
        p["leaf"] = 10.0 * env
        p["mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["bloom"] = 0.0
        p["leaf"] = 0.0
        p["kick"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["spark"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["head_dx"] = 0
        p["head_dy"] = 0
        p["turned"] = True
        p["mode"] = "up"
    return p


def part_angle(rig, name, mode, amp):
    if abs(amp) < 0.45 or name not in rig["signs"]:
        return 0.0
    sign = rig["signs"][name]
    if mode == "up":
        return sign["up"] * amp
    return sign["fwd"] * amp


def moved(layer, ang, pivot, dx, dy):
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, pivot)
    return shift(layer, dx, dy)


def limb_order(fc):
    # The leaf on the facing side is pasted last. The flower stays in front.
    if fc < 0:
        return ("leg_r", "leg_l", "leaf_r", "leaf_l", "bloom")
    return ("leg_l", "leg_r", "leaf_l", "leaf_r", "bloom")


def limb_delta(rig, p, name):
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    if name == "leg_l":
        sx, sy = p["step_l"]
        ang = part_angle(rig, name, "fwd", p["kick"])
        return ang, bdx + int(round(fc * sx)), bdy + int(sy)
    if name == "leg_r":
        sx, sy = p["step_r"]
        ang = part_angle(rig, name, "fwd", p["kick"])
        return ang, bdx + int(round(fc * sx)), bdy + int(sy)
    if name in ("leaf_l", "leaf_r"):
        return part_angle(rig, name, p["mode"], p["leaf"]), bdx, bdy
    if name == "bloom":
        return part_angle(rig, name, p["mode"], p["bloom"]), bdx, bdy
    return 0.0, bdx, bdy


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    for name in limb_order(fc):
        ang, dx, dy = limb_delta(rig, p, name)
        paste(out, moved(L[name], ang, piv[name], dx, dy))
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK)
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def lean_angle(facing, deg):
    """Head moves opposite facing. Positive rotation is clockwise on screen."""
    return facing * deg


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (GOLD, WHITE, LIME)
    r = 3 + int(age) * 2
    step = 1 if fc >= 0 else -1
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2))):
        put(img, ox + step * dx, oy + dy, cols[i % 3])


def forward_tip(rig, p):
    fc = rig["facing"]
    best = None
    for name in ("leaf_l", "leaf_r", "bloom", "leg_l", "leg_r"):
        ang, dx, dy = limb_delta(rig, p, name)
        layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
        m = layer[..., 3] >= 128
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
        pt = (float(xs[i]), float(ys[i]))
        if best is None:
            best = pt
        elif fc > 0 and pt[0] > best[0]:
            best = pt
        elif fc < 0 and pt[0] < best[0]:
            best = pt
    if best is None:
        return rig["pivots"]["center"]
    return best


def rig_for(rigs, d, p):
    """Attack 5-8 shows this side's back at equal scale. Back rows stay."""
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", mirror)]


def moving(p):
    if p["step_l"] != (0, 0) or p["step_r"] != (0, 0):
        return True
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    if abs(p["bloom"]) >= 0.45 or abs(p["leaf"]) >= 0.45 or abs(p["kick"]) >= 0.45:
        return True
    if abs(p["lean"]) > 0.4:
        return True
    return False


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    if not moving(p):
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            body = rotate_nearest(body, lean_angle(fc, p["lean"]), rear_pivot(body, fc))
        body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = forward_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name in LIMBS:
        ang, dx, dy = limb_delta(rig, p, name)
        layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def assemble(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                t = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png").convert("RGBA"))
                sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = t
        sheets[act] = sh
    return sheets


def write_outputs(sheets):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for act, sh in sheets.items():
        im = Image.fromarray(sh, "RGBA")
        im.save(OUT_DIR / f"{act}.png")
        im.save(OUT_DIR / f"{act}_8dir.png")
        Image.fromarray(sh[:CELL_H], "RGBA").save(OUT_DIR / f"{act}_S_strip.png")
    master = np.concatenate([sheets[a] for a in ACTIONS], axis=0)
    Image.fromarray(master, "RGBA").save(OUT_DIR / "sheet.png")
    return master


def tile_grid(tiles, cols, bg=(18, 22, 34, 255)):
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    out = np.full((rows * th, cols * tw, 4), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        m = t[..., 3] >= 128
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
    return out


def content_box(canvases, pad=8):
    acc = np.zeros_like(canvases[0][..., 3], bool)
    for c in canvases:
        acc |= c[..., 3] >= 128
    ys, xs = np.nonzero(acc)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(CELL_H, int(ys.max()) + pad + 8)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(CELL_W, int(xs.max()) + pad)
    return y0, y1, x0, x1


def preview_image(sheets):
    pick = {"idle": 4, "walk": 4, "attack": 11, "hurt": 3}
    tiles = []
    for act in ACTIONS:
        for di in range(8):
            f = pick[act]
            tiles.append(sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W])
    return Image.fromarray(tile_grid(tiles, 8), "RGBA")


def motion_strip(sheets, di, box):
    y0, y1, x0, x1 = box
    rows = []
    for act in ACTIONS:
        row = []
        for f in range(16):
            if f < FRAMES[act]:
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                row.append(t[y0:y1, x0:x1])
            else:
                row.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def locked_palette(front_path, back_path, work):
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
    imgs = [np.array(Image.open(p).convert("RGBA")) for p in (front_path, back_path)]
    own = []
    for im in imgs:
        cols = np.unique(im[im[..., 3] >= 128][:, :3], axis=0)
        own.extend(tuple(int(v) for v in c) for c in cols)
    own = sorted(set(own))

    def matches():
        if not (lock_dir / "palette.json").exists():
            return False
        data = json.loads((lock_dir / "palette.json").read_text())
        got = {tuple(int(c["hex"][i:i + 2], 16) for i in (1, 3, 5)) for c in data["colors"]}
        return got == set(own)

    if not matches():
        py = str(FORGE / ".venv/bin/python")
        hexes = ["#%02x%02x%02x" % c for c in own]
        for d in (pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        r = subprocess.run(
            [py, "-I", str(PALETTE_TOOL), "build", "--input", str(front_path),
             "--colors", str(len(hexes)), "--reserve", *hexes, "--output-dir", str(pal_dir)],
            capture_output=True, text=True,
        )
        if r.returncode:
            raise SystemExit("palette build failed: " + r.stdout + r.stderr)
        r = subprocess.run(
            [py, "-I", str(PALETTE_TOOL), "lock", "--palette", str(pal_dir / "palette.json"),
             "--source", str(front_path), "--strict", "--output-dir", str(lock_dir)],
            capture_output=True, text=True,
        )
        if r.returncode:
            raise SystemExit("palette lock failed: " + r.stdout + r.stderr)
    data = json.loads((lock_dir / "palette.json").read_text())
    cols = np.array([[int(c["hex"][i:i + 2], 16) for i in (1, 3, 5)] for c in data["colors"]], np.uint8)
    got = {tuple(int(v) for v in c) for c in cols}
    if got != set(own):
        raise SystemExit(f"locked palette differs from the stills: {sorted(got ^ set(own))}")
    return cols


def params_hash():
    src = Path(__file__).read_bytes() + (SRC_DIR / "front.png").read_bytes() + (SRC_DIR / "back.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def crown_shift(img, facing, deg):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    i = int(ys.argmin())
    piv = rear_pivot(img, facing)
    rx = float(xs[i]) - piv[0]
    ry = float(ys[i]) - piv[1]
    ang = lean_angle(facing, deg)
    r = math.radians(ang)
    c, s = math.cos(r), math.sin(r)
    return (piv[0] + c * rx + s * ry) - float(xs[i])


def build_rigs():
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    back = np.array(Image.open(SRC_DIR / "back.png").convert("RGBA"))
    if front.shape[:2] != (96, 96) or back.shape[:2] != (96, 96):
        raise SystemExit("expected 96x96 stills")
    for im in (front, back):
        im[im[..., 3] < 128] = 0
        im[im[..., 3] >= 128, 3] = 255
    if int(np.nonzero(front[..., 3])[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    if int(np.nonzero(back[..., 3])[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line changed")
    fm, ex, cx = segment_front(front)
    bm, bloom_x, body_x = segment_back(back)
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs["E"]["facing"] != 1.0 or rigs["W"]["facing"] != -1.0:
        raise SystemExit("front facing is not right-as-drawn")
    if rigs["N"]["facing"] != -1.0 or rigs["NE"]["facing"] != 1.0:
        raise SystemExit("back facing is not away-left as drawn")
    if rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("this still has no mouth")
    for key in (("front", False), ("front", True), ("back", False)):
        rig = rigs[key]
        fc = rig["facing"]
        shift_x = crown_shift(rig["canvas"], fc, MAX_LEAN)
        if fc > 0 and not (shift_x < -5):
            raise SystemExit(f"hurt lean does not tip the head back ({shift_x})")
        if fc < 0 and not (shift_x > 5):
            raise SystemExit(f"hurt lean does not tip the head back ({shift_x})")
    return front, back, fm, bm, rigs, (ex, cx, bloom_x, body_x)


def save_parts(work, front, fm, back, bm):
    cols = {
        "bloom": (255, 140, 40, 255),
        "leaf_l": (40, 200, 90, 255),
        "leaf_r": (40, 160, 220, 255),
        "leg_l": (220, 60, 40, 255),
        "leg_r": (200, 80, 220, 255),
        "torso": (170, 170, 170, 255),
    }
    tiles = []
    for masks in (fm, bm):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 3, 0), 3, 1))
    Image.fromarray(np.concatenate(tiles, axis=1), "RGBA").save(work / "parts.png")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    return n


def publish(bounds, sprite_h):
    import gen5_publish

    entry = {
        "dex": 549, "form": "normal", "variant": "lilligant", "gen": "gen5",
        "dir": "monsterSprite/gen5/0549_lilligant/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def frame_metrics(rigs, palette):
    pal = {tuple(int(v) for v in c) for c in palette}
    off = set()
    partial = 0
    clip = []
    bodies = []
    for d in DIRS:
        for act in ACTIONS:
            for f in range(FRAMES[act]):
                t = synth(rigs, d, act, f, palette)
                a = t[..., 3]
                partial += int(((a != 0) & (a != 255)).sum())
                op = a == 255
                if op.any():
                    used = {tuple(int(v) for v in c) for c in np.unique(t[op][:, :3], axis=0)}
                    off |= used - pal
                if op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
                if act in ("attack", "hurt", "walk"):
                    bodies.append(largest_fraction(t))
    return {
        "off_palette": len(off),
        "partial_alpha": partial,
        "edge_touch": len(clip),
        "edge_touching": clip[:8],
        "body_main_min": round(min(bodies), 3) if bodies else 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    save_peaks(Path(__file__).stem, PEAKS)
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    palette = locked_palette(SRC_DIR / "front.png", SRC_DIR / "back.png", work)
    front, back, fm, bm, rigs, measured = build_rigs()
    save_parts(work, front, fm, back, bm)
    ex, cx, bloom_x, body_x = measured
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    print("measure", "eye_x", round(ex, 2), "chin_x", round(cx, 2),
          "back_bloom_x", round(bloom_x, 2), "back_body_x", round(body_x, 2), flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    turn_s = synth(rigs, "S", "attack", 6, palette, effects=False)
    turn_w = synth(rigs, "W", "attack", 6, palette, effects=False)
    turn_n = synth(rigs, "N", "attack", 6, palette, effects=False)
    if np.any(turn_s != rigs["N"]["canvas"]):
        raise SystemExit("S turn is not the N back at equal scale")
    if np.any(turn_w != rigs["NE"]["canvas"]):
        raise SystemExit("W turn is not the mirrored back at equal scale")
    if np.any(turn_n != rigs["N"]["canvas"]):
        raise SystemExit("N turn left the back still")

    if args.samples:
        lost = {}
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost[f"{act}:{d}:{f}"] = bad
        print("limb lost", lost or "none", flush=True)
        metrics = frame_metrics(rigs, palette)
        print("qa_samples", "rest_diff", int(sum(rest_diff.values())),
              "off_palette", metrics["off_palette"],
              "partial_alpha", metrics["partial_alpha"],
              "edge_touch", metrics["edge_touch"],
              "body_main_min", metrics["body_main_min"],
              "clip", metrics["edge_touching"], flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(float(p["bloom"]), 1), round(float(p["leaf"]), 1),
                        round(float(p["kick"]), 1), p["mode"], p["turned"],
                        p["spin"], p["bdx"], p["spark"]))
        print("attack poses", atk, flush=True)
        leans = [round(float(pose("hurt", f)["lean"]), 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        box = content_box([rigs[d]["canvas"] for d in DIRS])
        y0, y1, x0, x1 = box
        tiles = []
        for d in ("S", "W", "N", "E"):
            for act, f in (
                ("idle", 0), ("idle", 8),
                ("walk", 4), ("walk", 12),
                ("attack", 0), ("attack", 3), ("attack", 6), ("attack", 11),
                ("hurt", 3),
            ):
                tiles.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
        sheet = np.repeat(np.repeat(tile_grid(tiles, 9), 2, 0), 2, 1)
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        print("samples", work / "samples.png", flush=True)
        return

    frames_dir = work / "frames"
    state_p = work / "state.json"
    digest = params_hash()
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    if args.force or state.get("hash") != digest:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        state = {"hash": digest, "done": []}
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = len(DIRS) * len(ACTIONS)
    for act in ACTIONS:
        for d in DIRS:
            key = f"{act}/{d}"
            if key in state["done"]:
                continue
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    box = content_box([rigs[d]["canvas"] for d in DIRS])
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.repeat(np.repeat(np.concatenate(strips, axis=0), 2, 0), 2, 1)
    Image.fromarray(motion, "RGBA").save(OUT_DIR / "preview_motion.png")
    preview_image(sheets).save(OUT_DIR / "preview.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", OUT_DIR / "actions" / "preview.png")

    pal = {tuple(int(v) for v in c) for c in palette}
    op = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[op][:, :3], axis=0)}
    off = sorted(used - pal)
    partial = int(((master[..., 3] != 0) & (master[..., 3] != 255)).sum())
    clip = []
    bodies = []
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = t[..., 3] == 255
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
                if act in ("attack", "hurt", "walk"):
                    bodies.append(largest_fraction(t))
    import gen5_publish
    bounds = gen5_publish.measure_idle_bounds(OUT_DIR / "sheet.png", CELL_W, CELL_H, rows=8)
    sprite_h = max(bounds[i + 3] for i in range(0, 32, 4))
    idle0 = {}
    for di, d in enumerate(DIRS):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        idle0[d] = int(np.any(t != rigs[d]["canvas"], axis=2).sum())
    report = {
        "version": VERSION, "hash": digest, "rest_diff": rest_diff, "idle0": idle0,
        "off_palette_colors": len(off), "partial_alpha": partial, "edge_touching": clip[:12],
        "edge_n": len(clip), "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "bounds": bounds, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (
        not any(rest_diff.values()) and not any(idle0.values()) and not off
        and partial == 0 and not clip and min(bodies) >= 0.90
    )
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print("qa_full", "rest_diff", int(sum(rest_diff.values()) + sum(idle0.values())),
          "off_palette", len(off), "partial_alpha", partial, "edge_touch", len(clip),
          "body_main_min", report["body_main_min"], "clip", clip[:8], flush=True)
    meta = {
        "generator": "forge_gen5_lilligant_actions", "version": VERSION, "key": KEY,
        "dex": 549, "variant": "lilligant", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped flower dancer (two eyes, no mouth, bloom, two side leaves, two feet)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-right (chin mean x 46.0 > eye mean x 45.0); back bloom is left of the body",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    if args.publish:
        print("synced", sync_outputs(), flush=True)
        publish(bounds, sprite_h)
        print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
