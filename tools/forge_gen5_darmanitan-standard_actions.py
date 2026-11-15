#!/usr/bin/env python3
"""Darmanitan standard (0555) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x15-72 y13-77, 1847 px, 15 colours,
alpha 0/255, sole y77):
  Eyes (white 246,246,246 / grey 139,139,156 / pupil 74,74,82, y<=41) mean
  x 43.53. The open mouth is the tooth block y>=44, x35-52, tip x35.
  Tip is left of the eyes, so the still faces screen-left.
  Two flame brows, flame colours only: left x34-41 y13-35 (105 px), right
  x43-65 y16-37 (157 px). They do not touch each other.
  Near fist hangs free of the torso from the armpit notch: y>=52, x15-32
  (243 px). Far fist is the separate right lump y>=60, x62-72 (73 px).
  Soles are one mass under the belly, split at x50: left x42-50 and right
  x51-59, both reach y77. No wings, tail, horns, or extra legs.
normal/back.png (bbox x20-72 y12-72, 1624 px, 12 colours, all inside the
front palette, sole y72): no eyes and no teeth. Two flame brows, the left
one 8-connected so the diagonal pixel at (18, 34) stays on that brow
(96 and 121 px). Side fists separate at the wrists: left y>=62 x20-30
(48 px), right y>=59 x57-72 (156 px). Centre soles x36-52 y>=66, split at
x43, both reach y72. Back is placed 5 px lower so both soles share FOOT_Y.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (away, forward to screen-right). NW back mirrored.
  Attack 5-8 is that direction's back at equal scale. W becomes N, S becomes
  NW. No |cos| squash and no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_darmanitan-standard_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0555_darmanitan-standard/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0555_darmanitan-standard/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0555_darmanitan-standard"
VERSION = "darmanitan-standard_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 224, 224
OFF_X = (CELL_W - 96) // 2
FRONT_FOOT = 77
BACK_FOOT = 72
# Back sole is 5 px higher in its 96 canvas, so the back buffer sits lower.
# 24 px under the shared sole keeps that buffer inside the cell.
FOOT_Y = CELL_H - 24
# Front as drawn looks screen-left. W keeps that picture. Its back is N.
# S is the mirrored front, so its back is N mirrored. Equal scale, no squash.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# finish() may raise the punch this far. The hit is drawn at 34, above this,
# so the strike is not stretched past the angle this script already draws.
PEAKS = {"arm": 18.0}
LIMBS = ("arm_l", "arm_r", "foot_l", "foot_r", "brow_l", "brow_r")

INK = (0, 0, 0)
WHITE = (246, 246, 246)
GREY = (139, 139, 156)
PUPIL = (74, 74, 82)
TAN = (238, 164, 106)
YELLOW = (238, 205, 16)
FLAME_RED = (238, 16, 16)
DEEP = (189, 0, 0)
GOLD = (189, 148, 41)
OLIVE = (106, 90, 16)
FLAME_COLS = (FLAME_RED, DEEP, YELLOW, OLIVE, GOLD)
MOUTH_COLS = (WHITE, GREY, PUPIL)

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
    """4x nearest in-plane rotation. Positive follows the shared forge formula. No squash."""
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


def paint_cushion(layer, allowed, steps, dirs=("up", "down", "left", "right")):
    """Grow this layer's own colours into allowed pixels. Hidden at rest."""
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        grown = []
        if "up" in dirs:
            g = np.zeros_like(src)
            g[:-1] = src[1:]
            grown.append(g)
        if "down" in dirs:
            g = np.zeros_like(src)
            g[1:] = src[:-1]
            grown.append(g)
        if "left" in dirs:
            g = np.zeros_like(src)
            g[:, :-1] = src[:, 1:]
            grown.append(g)
        if "right" in dirs:
            g = np.zeros_like(src)
            g[:, 1:] = src[:, :-1]
            grown.append(g)
        for g in grown:
            take = (acc[..., 3] < 128) & allowed & (g[..., 3] >= 128)
            acc[take] = g[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def flood(mask, seed, diag=False):
    h, w = mask.shape
    y0, x0 = seed
    if not mask[y0, x0]:
        raise SystemExit(f"seed {seed} is empty")
    out = np.zeros_like(mask)
    stack = [(int(y0), int(x0))]
    out[y0, x0] = True
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diag:
        neigh = neigh + ((1, 1), (1, -1), (-1, 1), (-1, -1))
    while stack:
        y, x = stack.pop()
        for dy, dx in neigh:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                stack.append((ny, nx))
    return out


def components(mask, diag=False):
    seen = np.zeros_like(mask, bool)
    groups = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        m = flood(mask, (int(y0), int(x0)), diag=diag)
        seen |= m
        groups.append(m)
    groups.sort(key=lambda m: int(m.sum()), reverse=True)
    return groups


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


def largest_component(mask):
    groups = components(mask)
    if not groups:
        return 0
    return int(groups[0].sum())


def touches(part, body):
    return bool((part & dilate(body, 1)).any())


def flame_mask(img):
    return any_color(img, FLAME_COLS)


def brows_from(img):
    """Two flame brows. A diagonal speck stays with the brow it touches."""
    groups = components(flame_mask(img), diag=True)
    big = [g for g in groups if int(g.sum()) >= 12]
    if len(big) != 2:
        raise SystemExit(f"expected two flame brows, got {len(big)}")
    big.sort(key=lambda m: float(np.nonzero(m)[1].mean()))
    return big[0], big[1]


def cut_part(op, y0, x_lo, x_hi, seed):
    yy, xx = np.mgrid[0:op.shape[0], 0:op.shape[1]]
    region = op & (yy >= y0) & (xx >= x_lo) & (xx <= x_hi)
    return flood(region, seed)


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    brow_l, brow_r = brows_from(img)
    used = brow_l | brow_r
    arm_l = cut_part(op & ~used, 52, 0, 40, (65, 22))
    arm_r = cut_part(op & ~used, 60, 60, 95, (64, 66))
    used = used | arm_l | arm_r
    soles = op & ~used & (yy >= 68) & (xx >= 40) & (xx <= 60)
    foot_l = soles & (xx <= 50)
    foot_r = soles & (xx >= 51)
    used = used | foot_l | foot_r
    torso = op & ~used
    masks = {
        "brow_l": brow_l, "brow_r": brow_r,
        "arm_l": arm_l, "arm_r": arm_r,
        "foot_l": foot_l, "foot_r": foot_r,
        "torso": torso,
    }
    check_partition(op, masks)
    eye = any_color(img, MOUTH_COLS) & (yy <= 41)
    teeth = any_color(img, MOUTH_COLS) & (yy >= 44)
    if int(eye.sum()) < 12 or int(teeth.sum()) < 40:
        raise SystemExit("front face markings moved")
    eye_x = float(np.nonzero(eye)[1].mean())
    tip_x = int(np.nonzero(teeth)[1].min())
    if tip_x >= eye_x - 1.0:
        raise SystemExit(f"front does not face left: tip {tip_x} eye {eye_x:.2f}")
    expect = {
        "brow_l": 105, "brow_r": 157, "arm_l": 243, "arm_r": 73,
        "foot_l": 67, "foot_r": 79,
    }
    for name, n in expect.items():
        if int(masks[name].sum()) != n:
            raise SystemExit(f"front {name} size {int(masks[name].sum())} != {n}")
        if largest_component(masks[name]) != n:
            raise SystemExit(f"front {name} fell apart")
        if not touches(masks[name], torso):
            raise SystemExit(f"front {name} does not touch the body")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("front torso fell apart")
    if int(np.nonzero(foot_l)[0].max()) != FRONT_FOOT or int(np.nonzero(foot_r)[0].max()) != FRONT_FOOT:
        raise SystemExit("front soles left the ground line")
    if int((eye & torso).sum()) != int(eye.sum()) or int((teeth & torso).sum()) != int(teeth.sum()):
        raise SystemExit("front face left the head")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    if any_color(img, MOUTH_COLS).any():
        raise SystemExit("back gained a face colour")
    brow_l, brow_r = brows_from(img)
    used = brow_l | brow_r
    arm_l = cut_part(op & ~used, 62, 0, 32, (64, 24))
    arm_r = cut_part(op & ~used, 59, 56, 95, (64, 64))
    used = used | arm_l | arm_r
    soles = op & ~used & (yy >= 66) & (xx >= 36) & (xx <= 52)
    foot_l = soles & (xx <= 43)
    foot_r = soles & (xx >= 44)
    used = used | foot_l | foot_r
    torso = op & ~used
    masks = {
        "brow_l": brow_l, "brow_r": brow_r,
        "arm_l": arm_l, "arm_r": arm_r,
        "foot_l": foot_l, "foot_r": foot_r,
        "torso": torso,
    }
    check_partition(op, masks)
    expect = {
        "brow_l": 96, "brow_r": 121, "arm_l": 48, "arm_r": 156,
        "foot_l": 27, "foot_r": 26,
    }
    for name, n in expect.items():
        if int(masks[name].sum()) != n:
            raise SystemExit(f"back {name} size {int(masks[name].sum())} != {n}")
        if name.startswith("brow"):
            # The left brow includes one diagonal pixel. 4-connected size is the rest.
            if largest_component(masks[name]) < n - 1:
                raise SystemExit(f"back {name} fell apart")
        elif largest_component(masks[name]) != n:
            raise SystemExit(f"back {name} fell apart")
        if not touches(masks[name], torso):
            raise SystemExit(f"back {name} does not touch the body")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("back torso fell apart")
    if int(np.nonzero(foot_l)[0].max()) != BACK_FOOT or int(np.nonzero(foot_r)[0].max()) != BACK_FOOT:
        raise SystemExit("back soles left the ground line")
    return masks


def place(arr, src_foot):
    off_y = FOOT_Y - int(src_foot)
    if off_y < 0 or off_y + arr.shape[0] > CELL_H:
        raise SystemExit("placement leaves the cell")
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    out[off_y:off_y + arr.shape[0], OFF_X:OFF_X + arr.shape[1]] = arr
    return out


def joint_band(part, torso, depth):
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 3)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def pivot_of(layer, torso_layer):
    part = layer[..., 3] >= 128
    torso = torso_layer[..., 3] >= 128
    touch = part & dilate(torso, 2)
    if not touch.any():
        touch = part & dilate(torso, 6)
    if not touch.any():
        ys, xs = np.nonzero(part)
        if len(xs) == 0:
            return 0.0, 0.0
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(torso)
    cx, cy = float(tx.mean()), float(ty.mean())
    d = (xs - cx) ** 2 + (ys - cy) ** 2
    near = d <= d.min() + 8.0
    return float(xs[near].mean()), float(ys[near].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs - pivot[0]) ** 2 + (ys - pivot[1]) ** 2
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


def build_view(img, masks, kind, mirror, src_foot):
    canvas = place(img, src_foot)
    placed = {k: place(v, src_foot) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    # Feet keep a few pixels of their own colour up into the ankle, so a
    # 1 px breath does not open a hole. The torso is drawn over that cushion.
    torso_allow = np.zeros_like(canvas)
    torso_allow[torso_m] = canvas[torso_m]
    for name in ("foot_l", "foot_r"):
        layers[name] = paint_cushion(layers[name], torso_m, 5, dirs=("up",))
    for name, depth in (("arm_l", 6), ("arm_r", 6), ("brow_l", 5), ("brow_r", 5)):
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth,
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in LIMBS:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    mouth = any_color(canvas, MOUTH_COLS) & (np.arange(CELL_H)[:, None] >= (FOOT_Y - src_foot + 44))
    # Tooth pixels only. The upper eye band stays above the mouth.
    eye_band = any_color(canvas, MOUTH_COLS) & (np.arange(CELL_H)[:, None] <= (FOOT_Y - src_foot + 41))
    teeth = any_color(canvas, MOUTH_COLS) & ~eye_band
    has_mouth = bool(kind == "front" and teeth.any())
    if has_mouth:
        my, mx = np.nonzero(teeth)
        pivots["mouth"] = (float(mx.mean()), float(my.mean()))
        j = int(mx.argmax()) if facing > 0 else int(mx.argmin())
        pivots["muzzle"] = (float(mx[j]), float(my[j]))
    else:
        pivots["mouth"] = pivots["center"]
        pivots["muzzle"] = pivots["center"]
    # The big fist is the one that hits. Front: the near left arm. Back: the right fist.
    striker = "arm_l" if kind == "front" else "arm_r"
    order_back = ("foot_l", "foot_r")
    order_front = ("arm_r", "arm_l", "brow_l", "brow_r") if kind == "front" else (
        "arm_l", "arm_r", "brow_l", "brow_r"
    )
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "has_mouth": has_mouth,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
        "striker": striker, "order_back": order_back, "order_front": order_front,
        "src_foot": src_foot,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        arm=0.0, arm_mode="fwd",
        brow=0.0,
        step_l=(0, 0), step_r=(0, 0),
        wedge=0, ring=-1, spark=0,
        lean=0.0, knock=0, flash=False,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. Flames breathe. Frame 0 is the still.
        p["brow"] = wave(f, n, 5.0, phase=0.4)
        p["bdy"] = -1 if wave(f, n, 1.0, phase=0.2) > 0.55 else 0
        p["arm_mode"] = "up"
    elif action == "walk":
        # Two drawn soles, opposite steps. The planted sole does not rise.
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * sb)), -int(round(3 * max(0.0, sb))))
        p["arm"] = wave(f, n, 10.0, cycles=1.0)
        p["brow"] = wave(f, n, 4.0, phase=0.6, cycles=2.0)
        p["bdy"] = -1 if abs(math.sin(u)) > 0.85 else 0
        p["arm_mode"] = "fwd"
    elif action == "attack":
        # 0 rest. 1-4 roar, fists and brows lift. 5-8 the back, equal scale.
        # 9 wind-up. 10-12 the near fist punches. 13-15 return. 15 is the still.
        arm = [0, 6, 10, 14, 14, 0, 0, 0, 0, 20, 30, 34, 34, 16, 6, 0]
        brow = [0, 8, 14, 18, 18, 0, 0, 0, 0, 6, 8, 6, 4, 2, 0, 0]
        bdx = [0, -1, -1, -1, -1, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        p["arm"] = float(arm[f])
        p["brow"] = float(brow[f])
        p["bdx"] = int(bdx[f])
        p["arm_mode"] = "up" if f <= 4 else "fwd"
        p["turned"] = 5 <= f <= 8
        if 1 <= f <= 4:
            p["wedge"] = [0, 1, 2, 3, 3][f]
        if 2 <= f <= 4:
            p["ring"] = f - 2
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.64, 0.44, 0.26, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        # The back is the still, at equal scale. finish() must not swing it.
        p["arm"] = 0.0
        p["brow"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["wedge"] = 0
        p["ring"] = -1
        p["spark"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["turned"] = True
    if action == "attack" and abs(float(p["arm"])) > 34.05:
        raise SystemExit(f"punch drawn past 34 deg at frame {f}: {p['arm']}")
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


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    far_scale = 0.7 if rig["kind"] == "front" else 1.0
    angles = {
        "arm_l": part_angle(rig, "arm_l", p["arm_mode"], p["arm"] * (1.0 if rig["striker"] == "arm_l" else far_scale)),
        "arm_r": part_angle(rig, "arm_r", p["arm_mode"], p["arm"] * (1.0 if rig["striker"] == "arm_r" else far_scale)),
        "brow_l": part_angle(rig, "brow_l", "up", p["brow"]),
        "brow_r": part_angle(rig, "brow_r", "up", p["brow"]),
    }
    steps = {"foot_l": p["step_l"], "foot_r": p["step_r"]}
    for name in rig["order_back"]:
        sx, sy = steps[name]
        paste(out, shift(L[name], bdx + int(round(fc * sx)), int(sy)))
    paste(out, shift(L["torso"], bdx, bdy))
    for name in rig["order_front"]:
        paste(out, moved(L[name], angles[name], piv[name], bdx, bdy))
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


def mouth_wedge(img, origin, fc, age):
    """A few pixels in front of the existing teeth. Empty pixels only."""
    ox, oy = origin
    cols = (INK, WHITE, FLAME_RED)
    for i in range(max(0, int(age))):
        x = ox + fc * (2 + i)
        span = 1 if i == 0 else 2
        for dy in range(-span, span + 1):
            put(img, x, oy + dy, cols[min(i, 2)])


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, YELLOW, FLAME_RED)
    rad = 8 + age * 5
    for deg in range(-50, 51, 8):
        a = math.radians(deg)
        x = ox + fc * rad * math.cos(a)
        y = oy + rad * math.sin(a) * 0.55
        put(img, x, y, cols[age % 3])


def strike_tip(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    name = rig["striker"]
    ang = part_angle(rig, name, p["arm_mode"], p["arm"])
    layer = moved(L[name], ang, piv[name], bdx, bdy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (YELLOW, WHITE, FLAME_RED)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2), (2, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    moved_part = (
        abs(p["arm"]) >= 0.45 or abs(p["brow"]) >= 0.45
        or p["step_l"] != (0, 0) or p["step_r"] != (0, 0)
        or p["bdx"] or p["bdy"]
    )
    fc = rig["facing"]
    if not moved_part and abs(p["lean"]) <= 0.4 and not p["knock"]:
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            body = rotate_nearest(body, -fc * p["lean"], rear_pivot(body, fc))
        body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if rig["has_mouth"] and not p["turned"]:
        if p["wedge"]:
            mouth_wedge(img, rig["pivots"]["muzzle"], fc, p["wedge"])
        if p["ring"] >= 0:
            roar_rings(img, rig["pivots"]["muzzle"], fc, p["ring"])
        if p["wedge"] or p["ring"] >= 0:
            img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = strike_tip(rig, p)
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
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    far_scale = 0.7 if rig["kind"] == "front" else 1.0
    angles = {
        "arm_l": part_angle(rig, "arm_l", p["arm_mode"], p["arm"] * (1.0 if rig["striker"] == "arm_l" else far_scale)),
        "arm_r": part_angle(rig, "arm_r", p["arm_mode"], p["arm"] * (1.0 if rig["striker"] == "arm_r" else far_scale)),
        "brow_l": part_angle(rig, "brow_l", "up", p["brow"]),
        "brow_r": part_angle(rig, "brow_r", "up", p["brow"]),
        "foot_l": 0.0, "foot_r": 0.0,
    }
    steps = {"foot_l": p["step_l"], "foot_r": p["step_r"], "arm_l": (0, 0), "arm_r": (0, 0),
             "brow_l": (0, 0), "brow_r": (0, 0)}
    lost = []
    for name in LIMBS:
        if int((L[name][..., 3] >= 128).sum()) == 0:
            continue
        if name in ("foot_l", "foot_r"):
            sx, sy = steps[name]
            layer = shift(L[name], bdx + int(round(fc * sx)), int(sy))
        else:
            layer = moved(L[name], angles[name], piv[name], bdx, bdy)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 2)).any():
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
    for img in imgs:
        cols = np.unique(img[img[..., 3] >= 128][:, :3], axis=0)
        own.extend(tuple(int(v) for v in c) for c in cols)
    own = sorted(set(own))
    if not (lock_dir / "palette.json").exists():
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
    want = set(own)
    if got != want:
        raise SystemExit(f"locked palette differs from the stills: {sorted(got ^ want)}")
    return cols


def params_hash():
    src = Path(__file__).read_bytes() + (SRC_DIR / "front.png").read_bytes() + (SRC_DIR / "back.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


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
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {}
    for kind, img, masks, foot in (
        ("front", front, fm, FRONT_FOOT),
        ("back", back, bm, BACK_FOOT),
    ):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror, foot)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs[("front", False)]["facing"] != -1.0 or rigs[("back", False)]["facing"] != 1.0:
        raise SystemExit("facing locked to the wrong side")
    if not rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("mouth is on the wrong view")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "brow_l": (255, 80, 40, 255),
        "brow_r": (255, 200, 40, 255),
        "arm_l": (40, 200, 90, 255),
        "arm_r": (40, 140, 220, 255),
        "foot_l": (220, 80, 180, 255),
        "foot_r": (180, 80, 220, 255),
        "torso": (170, 170, 170, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
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
        "dex": 555, "form": "normal", "variant": "darmanitan-standard", "gen": "gen5",
        "dir": "monsterSprite/gen5/0555_darmanitan-standard/normal",
        "sheetDir": "agent-sprite-forge",
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
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()

    save_peaks(Path(__file__).stem, PEAKS)
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    palette = locked_palette(SRC_DIR / "front.png", SRC_DIR / "back.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)

    zero = base_pose()
    compose_diff = {}
    for key in (("front", False), ("front", True), ("back", False), ("back", True)):
        got = compose(rigs[key], zero)
        compose_diff[str(key)] = int(np.any(got != rigs[key]["canvas"], axis=2).sum())
    print("compose zero", compose_diff, flush=True)
    if any(compose_diff.values()):
        raise SystemExit("zero pose does not rebuild the still")

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

    turn_w = synth(rigs, "W", "attack", 6, palette, effects=False)
    turn_s = synth(rigs, "S", "attack", 6, palette, effects=False)
    turn_n = synth(rigs, "N", "attack", 6, palette, effects=False)
    if np.any(turn_w != rigs["N"]["canvas"]):
        raise SystemExit("W turn is not the N back at equal scale")
    if np.any(turn_s != rigs["NW"]["canvas"]):
        raise SystemExit("S turn is not N flipped at equal scale")
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
        rest_sum = int(sum(rest_diff.values()))
        print(
            f"静止差分 {rest_sum} パレット外 {metrics['off_palette']} "
            f"中間アルファ {metrics['partial_alpha']} 枠接触 {metrics['edge_touch']}",
            flush=True,
        )
        print("qa_samples", "rest_diff", rest_sum,
              "off_palette", metrics["off_palette"],
              "partial_alpha", metrics["partial_alpha"],
              "edge_touch", metrics["edge_touch"],
              "body_main_min", metrics["body_main_min"],
              "clip", metrics["edge_touching"], flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(float(p["arm"]), 1), p["arm_mode"], p["turned"],
                        p["spin"], p["wedge"], p["bdx"], round(float(p["brow"]), 1)))
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
    rest_sum = int(sum(rest_diff.values()))
    print(
        f"静止差分 {rest_sum} パレット外 {len(off)} 中間アルファ {partial} 枠接触 {len(clip)}",
        flush=True,
    )
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
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_darmanitan-standard_actions", "version": VERSION, "key": KEY,
        "dex": 555, "variant": "darmanitan-standard", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped heavy puncher (open toothed mouth, two flame brows, near fist and far fist, two short soles, no tail)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (mouth tip x 35 < eye mean x 43.53)",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    if args.no_publish:
        print("qa ok; publish skipped", flush=True)
        return
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
