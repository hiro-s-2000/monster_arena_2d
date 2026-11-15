#!/usr/bin/env python3
"""Yamask normal (0562) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x20-72 y28-65, 749 px, alpha 0/255,
11 colours, foot y65):
  One red eye, mean x 46.87 (x44-50 y36-43). The gold mask (the face it
  holds) has mean x 44.02, left of that eye, so the still faces screen-left.
  No mouth is drawn on the ghost or as an open cavity in the mask. The roar
  does not invent a wedge or a ring. It lifts the arms and the mask.
  Left arm x20-40 y46-56. Right arm x60-72 y48-58. Mask x38-51 y48-65,
  including the small chin point under the gold. No legs, wings, horns,
  fangs, or pincers. The ghost floats.

normal/back.png (bbox x20-70 y28-65, 718 px, foot y65):
  The eye sits on the right of the head (mean x 50.62 vs head x 47.1), and
  the gold is the mask's right edge, so back as drawn looks away-right.
  Left arm x20-40 y43-52. Right arm x59-70 y47-57, touching the mask.
  Two ink pixels at x59 y50-51 belong to that arm.

Directions, front faces left, back faces away-right:
  S SE E front mirrored (screen-right). SW W front as drawn (screen-left).
  NE back as drawn. N NW back mirrored.
  E is screen-right, W is screen-left, N is the back.
  Attack 5-8 swaps kind and keeps the mirror. S becomes the N still.
  Equal scale. No width squash. No in-plane spin.

  idle 16: float. Arms, mask, and a 1px bob. No feet.
  walk 16: glide. Arms alternate. No new feet.
  attack: 0 still, 1-4 raise the arms and the mask (no mouth),
          5-8 the opposite still, 9-12 both arms and the mask strike,
          11-12 impact, 13-15 recover.
  hurt 12: flash, knock-back lean (<=38 deg) and recover.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_yamask_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0562_yamask/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0562_yamask/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0562_yamask"
VERSION = "yamask_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 192
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 24
FRONT_FOOT = 65
BACK_FOOT = 65
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. Back as drawn looks away to the right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "SW": ("front", False), "W": ("front", False),
    "NE": ("back", False), "N": ("back", True), "NW": ("back", True),
}
MAX_LEAN = 38.0
# Angles this script already draws. finish() will not raise a frame past them.
PEAKS = {"arm_l": 34.0, "arm_r": 34.0, "mask": 34.0}

INK = (16, 16, 16)
DK = (49, 49, 49)
MID = (82, 82, 82)
GOLD_D = (139, 90, 32)
GOLD_M = (205, 131, 16)
SILVER = (222, 222, 222)
GOLD = (246, 172, 41)
GOLD_L = (255, 222, 123)
RED_DD = (90, 0, 0)
RED_D = (172, 0, 0)
RED = (255, 0, 0)
GOLD_COLS = (GOLD, GOLD_M, GOLD_D, GOLD_L, SILVER)
RED_COLS = (RED, RED_D, RED_DD)
OUTLINE_KEEP = (INK, DK, RED, RED_D, RED_DD)

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


def dilate8(mask):
    p = np.pad(mask, 1, constant_values=False)
    h, w = mask.shape
    out = mask.copy()
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + h, dx:dx + w]
    return out


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    groups = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
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
    groups.sort(key=len, reverse=True)
    return groups


def mask_of(pts, shape):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m


def largest_component(mask):
    groups = components(mask)
    if not groups:
        return 0
    return len(groups[0])


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
    """4x nearest in-plane rotation (positive = clockwise on screen). No squash."""
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


def put(img, x, y, col, over=False):
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 1 <= x < w - 1 and 1 <= y < h - 1 and (over or img[y, x, 3] < 128):
        img[y, x, :3] = col
        img[y, x, 3] = 255


def paint_cushion(layer, allowed, steps, dirs=("up", "down", "left", "right")):
    """Grow the layer's own colours into allowed pixels (hidden at rest)."""
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        for d in dirs:
            mv = np.zeros_like(src)
            if d == "up":
                mv[:-1] = src[1:]
            elif d == "down":
                mv[1:] = src[:-1]
            elif d == "left":
                mv[:, :-1] = src[:, 1:]
            else:
                mv[:, 1:] = src[:, :-1]
            take = (acc[..., 3] < 128) & allowed & (mv[..., 3] >= 128)
            acc[take] = mv[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def grow_mask(op, gold, y0, y1, x0, x1):
    yy, xx = np.mgrid[0:96, 0:96]
    mask = gold.copy()
    box = (yy >= y0) & (yy <= y1) & (xx >= x0) & (xx <= x1)
    for _ in range(6):
        grow = op & ~mask & dilate(mask, 1) & box
        if not grow.any():
            break
        mask |= grow
    chin = op & (yy >= 64) & dilate(mask, 2) & (xx >= x0) & (xx <= x1 + 2)
    mask |= chin
    return mask


def attach_islands(masks):
    """Small torso islands join the part they actually touch."""
    gs = components(masks["torso"])
    if len(gs) <= 1:
        return masks
    shape = masks["torso"].shape
    keep = mask_of(gs[0], shape)
    for g in gs[1:]:
        blob = mask_of(g, shape)
        best, score = None, 0
        for name in ("mask", "arm_l", "arm_r"):
            s = int((blob & dilate(masks[name], 1)).sum())
            if s > score:
                best, score = name, s
        if best is None:
            raise SystemExit(f"torso island of {len(g)} touches no part")
        masks[best] = masks[best] | blob
    masks["torso"] = keep
    return masks


def segment(img, kind):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    gold = any_color(img, GOLD_COLS)
    if kind == "front":
        mask = grow_mask(op, gold, 50, 63, 37, 51)
        arm_y = 46
    else:
        mask = grow_mask(op, gold, 51, 65, 50, 59)
        arm_y = 43
    pool = op & (xx <= 40) & (yy >= arm_y) & ~mask
    gs = components(pool)
    if not gs:
        raise SystemExit(f"{kind} has no left arm")
    arm_l = mask_of(gs[0], op.shape)
    arm_r = op & (xx >= 60) & (yy >= arm_y) & ~mask & (yy < 64)
    torso = op & ~mask & ~arm_l & ~arm_r
    masks = {"torso": torso, "arm_l": arm_l, "arm_r": arm_r, "mask": mask}
    masks = attach_islands(masks)
    check_partition(op, masks)
    for name, m in masks.items():
        n = len(components(m))
        if n != 1:
            raise SystemExit(f"{kind} {name} fell apart into {n}")
    red = any_color(img, RED_COLS)
    if ((masks["mask"] | masks["arm_l"] | masks["arm_r"]) & red).any():
        raise SystemExit(f"{kind} eye left the body")
    if kind == "front":
        eye = red & (yy <= 45)
        if int(eye.sum()) < 30:
            raise SystemExit("front eye missing")
        mask_x = float(np.nonzero(masks["mask"])[1].mean())
        eye_x = float(np.nonzero(eye)[1].mean())
        if mask_x >= eye_x - 1.0:
            raise SystemExit(f"front mask x {mask_x:.2f} is not left of eye x {eye_x:.2f}")
    else:
        eye = red & (yy <= 42)
        head = op & (yy <= 42)
        if int(eye.sum()) < 20:
            raise SystemExit("back eye missing")
        eye_x = float(np.nonzero(eye)[1].mean())
        head_x = float(np.nonzero(head)[1].mean())
        if eye_x <= head_x + 2.0:
            raise SystemExit(f"back eye x {eye_x:.2f} is not right of head x {head_x:.2f}")
    if int(masks["arm_l"].sum()) < 80 or int(masks["arm_r"].sum()) < 60:
        raise SystemExit(f"{kind} arm collapsed")
    if int(masks["mask"].sum()) < 100:
        raise SystemExit(f"{kind} mask collapsed")
    return masks


def place(arr, src_foot):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    y0 = FOOT_Y - src_foot
    x0 = OFF_X
    h, w = arr.shape[:2]
    if y0 < 0 or x0 < 0 or y0 + h > CELL_H or x0 + w > CELL_W:
        raise SystemExit(f"place overflow y {y0} x {x0}")
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def joint_band(part, body, depth):
    touch = part & dilate(body, 1)
    if not touch.any():
        touch = part & dilate8(body)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def pivot_of(part, body):
    touch = part & dilate(body, 1)
    if not touch.any():
        touch = part & dilate8(body)
    if not touch.any():
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(body)
    cx, cy = float(tx.mean()), float(ty.mean())
    d = (xs.astype(np.float64) - cx) ** 2 + (ys.astype(np.float64) - cy) ** 2
    near = d <= d.min() + 0.5
    return float(xs[near].mean()), float(ys[near].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs.astype(np.float64) - pivot[0]) ** 2 + (ys.astype(np.float64) - pivot[1]) ** 2
    i = int(d.argmax())
    return float(xs[i]), float(ys[i])


def signs_for(layer, pivot, facing):
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


def part_anchor(name, placed):
    host = placed["torso"]
    part = placed[name]
    if name != "mask" and not (part & dilate(host, 1)).any():
        host = host | placed["mask"]
    return host


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
    facing = 1.0 if mirror else -1.0
    if kind == "back":
        facing = -facing
    pivots = {}
    signs = {}
    torso_m = placed["torso"]
    for name in ("arm_l", "arm_r", "mask"):
        anchor = part_anchor(name, placed)
        band = joint_band(placed[name], torso_m, 4)
        if band.any():
            layers["torso"] = paint_cushion(layers["torso"], band, 4)
        pivots[name] = pivot_of(placed[name], anchor)
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, head_dx=0, head_dy=0,
        arm_l=0.0, arm_r=0.0, mask=0.0,
        arm_mode="up", mask_mode="up",
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0,
    )


def clear_turn(p):
    p["arm_l"] = 0.0
    p["arm_r"] = 0.0
    p["mask"] = 0.0
    p["bdx"] = 0
    p["bdy"] = 0
    p["head_dx"] = 0
    p["head_dy"] = 0
    p["spark"] = 0
    p["spin"] = 0
    p["lean"] = 0.0
    p["knock"] = 0
    p["arm_mode"] = "up"
    p["mask_mode"] = "up"
    p["turned"] = True
    p["flash"] = False
    return p


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        sway = wave(f, n, 6.0, phase=0.4)
        p["arm_l"] = sway
        p["arm_r"] = sway
        p["mask"] = wave(f, n, 4.0, phase=1.1)
        bob = wave(f, n, 1.0)
        p["bdy"] = -1 if bob > 0.55 else 0
    elif action == "walk":
        bob = wave(f, n, 1.0, cycles=2.0)
        p["bdy"] = -1 if bob > 0.35 else 0
        p["bdx"] = int(round(wave(f, n, 1.0, cycles=2.0, phase=0.5)))
        p["arm_l"] = wave(f, n, 12.0, phase=0.2)
        p["arm_r"] = wave(f, n, 12.0, phase=math.pi)
        p["mask"] = wave(f, n, 5.0, phase=0.8)
    elif action == "attack":
        # 0 rest. 1-4 raise the arms and the mask. There is no mouth.
        # 5-8 are cleared after finish() so the opposite still stays exact.
        # 9-12 the arms and the mask strike. Values sit on finish()'s envelope.
        arm = [0, 8, 14, 20, 18, 0, 0, 0, 0, 34, 34, 34, 22, 8, 4, 0]
        mask = [0, 8, 14, 18, 16, 0, 0, 0, 0, 34, 34, 34, 20, 8, 3, 0]
        bdx = [0, -1, -1, -1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        bdy = [0, -1, -1, -2, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        p["arm_l"] = float(arm[f])
        p["arm_r"] = float(arm[f])
        p["mask"] = float(mask[f])
        p["bdx"] = int(bdx[f])
        p["bdy"] = int(bdy[f])
        p["arm_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["mask_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.66, 0.46, 0.28, 0.14, 0.05, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["arm_l"] = 12.0 * env
        p["arm_r"] = 12.0 * env
        p["mask"] = 8.0 * env
        p["arm_mode"] = "up"
        p["mask_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p = clear_turn(p)
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


def body_shift(rig, p):
    fc = rig["facing"]
    return int(round(fc * p["bdx"])), int(round(p["bdy"]))


def angle_of(rig, p, name):
    if name in ("arm_l", "arm_r"):
        return part_angle(rig, name, p["arm_mode"], p[name])
    if name == "mask":
        return part_angle(rig, name, p["mask_mode"], p["mask"])
    return 0.0


def part_layer(rig, name, p):
    if name not in rig["layers"]:
        return None
    bdx, bdy = body_shift(rig, p)
    return moved(rig["layers"][name], angle_of(rig, p, name), rig["pivots"][name], bdx, bdy)


def compose(rig, p):
    out = np.zeros_like(rig["canvas"])
    bdx, bdy = body_shift(rig, p)
    paste(out, shift(rig["layers"]["torso"], bdx, bdy))
    for name in ("arm_l", "arm_r", "mask"):
        layer = part_layer(rig, name, p)
        if layer is not None:
            paste(out, layer)
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = any_color(out, OUTLINE_KEEP)
    out[op & ~keep, :3] = np.array(SILVER, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def facing_tip(layer, facing):
    m = layer[..., 3] >= 128
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if facing > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (GOLD_L, RED, SILVER, GOLD)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r), (2, r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def limb_names(rig):
    return [n for n in ("arm_l", "arm_r", "mask") if n in rig["layers"]]


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name in limb_names(rig):
        layer = part_layer(rig, name, p)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate8(other)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    kind, mirror = VIEW_OF[d]
    if p["turned"]:
        kind = "back" if kind == "front" else "front"
    return rigs[(kind, mirror)]


def moving(p):
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["arm_l"]) >= 0.45 or abs(p["arm_r"]) >= 0.45 or abs(p["mask"]) >= 0.45:
        return True
    return False


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["arm_l"]) >= 0.45 or abs(p["arm_r"]) >= 0.45 or abs(p["mask"]) >= 0.45:
        return True
    return False


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    body = compose(rig, p)
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], piv)
        grounded = body[..., 3] >= 128
        if grounded.any():
            bot = int(np.nonzero(grounded)[0].max())
            body = shift(body, 0, FOOT_Y - bot)
    if rotated(p) or moving(p):
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        layer = part_layer(rig, "mask", p)
        tip = facing_tip(layer, fc) if layer is not None else None
        if tip is not None:
            impact(img, (tip[0] + fc * 2, tip[1]), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


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


def crop_box(canvas, pad=8):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad - 18)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 10)
    x0 = max(0, int(xs.min()) - pad - 22)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 22)
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
    tiles = []
    for act in ACTIONS:
        for f in range(16):
            if f < FRAMES[act]:
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                tiles.append(t[y0:y1, x0:x1])
            else:
                tiles.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
    return tile_grid(tiles, 16)


def frame_faults(img, pal):
    a = img[..., 3]
    partial = int(((a != 0) & (a != 255)).sum())
    op = a >= 128
    edge = bool(op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any())
    off = set()
    solid = a == 255
    if solid.any():
        for c in np.unique(img[solid][:, :3], axis=0):
            t = tuple(int(v) for v in c)
            if t not in pal:
                off.add(t)
    return off, partial, edge


def locked_palette(front_path, back_path, work):
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
    img = np.array(Image.open(front_path).convert("RGBA"))
    own = np.unique(img[img[..., 3] >= 128][:, :3], axis=0)
    if not (lock_dir / "palette.json").exists():
        py = str(FORGE / ".venv/bin/python")
        hexes = ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in own]
        for d in (pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        r = subprocess.run(
            [py, "-I", str(PALETTE_TOOL), "build", "--input", str(front_path),
             "--colors", str(len(hexes)), "--reserve", *hexes,
             "--output-dir", str(pal_dir)],
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
    want = {tuple(int(v) for v in c) for c in own}
    if got != want:
        raise SystemExit(f"locked palette differs from the still: {sorted(got ^ want)}")
    back = np.array(Image.open(back_path).convert("RGBA"))
    extra = []
    for c in np.unique(back[back[..., 3] >= 128][:, :3], axis=0):
        t = tuple(int(v) for v in c)
        if t not in got:
            extra.append(c)
    if extra:
        cols = np.vstack([cols, np.array(extra, np.uint8)])
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
    fm = segment(front, "front")
    bm = segment(back, "back")
    rigs = {}
    for kind, img, masks, foot in (
        ("front", front, fm, FRONT_FOOT),
        ("back", back, bm, BACK_FOOT),
    ):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror, foot)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs["W"]["facing"] != -1.0 or rigs["E"]["facing"] != 1.0:
        raise SystemExit("front facing is not left-as-drawn")
    if rigs["NE"]["facing"] != 1.0 or rigs["N"]["facing"] != -1.0:
        raise SystemExit("back facing is not away-right as drawn")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "arm_l": (80, 170, 220, 255),
        "arm_r": (240, 140, 60, 255),
        "mask": (240, 210, 60, 255),
        "torso": (90, 90, 110, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 3, 0), 3, 1))
    gap = np.zeros((tiles[0].shape[0], 8, 4), np.uint8)
    Image.fromarray(np.concatenate([tiles[0], gap, tiles[1]], axis=1), "RGBA").save(work / "parts.png")


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
        "dex": 562, "form": "normal", "variant": "yamask", "gen": "gen5",
        "dir": "monsterSprite/gen5/0562_yamask/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def tip_shift(rigs, d, name):
    rest = pose("attack", 0)
    hit = pose("attack", 11)
    rig_a = rig_for(rigs, d, rest)
    rig_b = rig_for(rigs, d, hit)
    la = part_layer(rig_a, name, rest)
    lb = part_layer(rig_b, name, hit)
    if la is None or lb is None:
        return None
    fa = rig_b["facing"]
    a = facing_tip(la, fa)
    b = facing_tip(lb, fa)
    if a is None or b is None:
        return None
    return round((b[0] - a[0]) * fa, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    args = ap.parse_args()

    save_peaks(Path(__file__).stem, PEAKS)
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    palette = locked_palette(SRC_DIR / "front.png", SRC_DIR / "back.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, flush=True)
    print("parts", {k: int(v.sum()) for k, v in fm.items()}, {k: int(v.sum()) for k, v in bm.items()}, flush=True)
    print("signs S", rigs["S"]["signs"], "N", rigs["N"]["signs"], flush=True)

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
        r3 = synth(rigs, d, "attack", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r3 != rigs[d]["canvas"], axis=2).sum())
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    turn_diff = {}
    for d in DIRS:
        turn_diff[d] = 0
        for f in (5, 6, 7, 8):
            p = pose("attack", f)
            rig = rig_for(rigs, d, p)
            img = synth(rigs, d, "attack", f, palette, effects=True)
            turn_diff[d] += int(np.any(img != rig["canvas"], axis=2).sum())
    print("turn diff", turn_diff, flush=True)
    if any(turn_diff.values()):
        raise SystemExit("attack 5-8 is not the opposite still")

    strike = {name: tip_shift(rigs, "S", name) for name in ("arm_l", "arm_r", "mask")}
    print("strike dx S", strike, flush=True)

    pal = {tuple(int(v) for v in c) for c in palette}
    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        y0, y1, x0, x1 = box
        off = set()
        partial = 0
        edge = []
        lost = []
        bodies = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    img = synth(rigs, d, act, f, palette)
                    o, part, ed = frame_faults(img, pal)
                    off |= o
                    partial += part
                    if ed:
                        edge.append(f"{act}:{d}:{f}")
                    if act in ("attack", "hurt", "walk"):
                        bodies.append(largest_fraction(img))
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        rest_n = int((rigs["S"]["canvas"][..., 3] >= 128).sum())
        hit = synth(rigs, "S", "attack", 11, palette, effects=False)
        back_hit = synth(rigs, "N", "attack", 11, palette, effects=False)
        print("opaque S0", rest_n, "S11", int((hit[..., 3] >= 128).sum()),
              "N11", int((back_hit[..., 3] >= 128).sum()), flush=True)
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["arm_l"], 1), round(p["arm_r"], 1), round(p["mask"], 1),
                        p["arm_mode"], p["mask_mode"], p["turned"], p["spin"], p["bdx"], p["bdy"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        report = {
            "rest_diff_max": max(rest_diff.values()),
            "turn_diff_max": max(turn_diff.values()),
            "off_palette": len(off),
            "partial_alpha": partial,
            "edge_n": len(edge),
            "edge_touching": edge[:12],
            "disconnected": len(lost),
            "body_main_min": round(min(bodies), 3) if bodies else 1,
            "strike_dx_S": strike,
        }
        print("qa_samples", json.dumps(report), flush=True)
        print("samples", work, flush=True)
        if any(v is None or v <= 0 for v in strike.values()):
            raise SystemExit(f"strike does not move forward: {strike}")
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
            if key in state["done"] and not args.force:
                continue
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.exists() and fp.stat().st_size > 0 and not args.force:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            if key not in state["done"]:
                state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    y0, y1, x0, x1 = box
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.concatenate(strips, axis=0)
    motion = np.repeat(np.repeat(motion, 2, 0), 2, 1)
    Image.fromarray(motion, "RGBA").save(OUT_DIR / "preview_motion.png")
    preview_image(sheets).save(OUT_DIR / "preview.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", OUT_DIR / "actions" / "preview.png")

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
        "turn_diff": turn_diff,
        "off_palette_colors": len(off), "partial_alpha": partial, "edge_touching": clip[:12],
        "edge_n": len(clip), "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "bounds": bounds, "cell": [CELL_W, CELL_H, FOOT_Y],
        "strike_dx_S": strike,
    }
    ok = (not any(rest_diff.values()) and not any(idle0.values()) and not any(turn_diff.values())
          and not off and partial == 0 and not clip
          and all(v is not None and v > 0 for v in strike.values()))
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_yamask_actions", "version": VERSION, "key": KEY,
        "dex": 562, "variant": "yamask", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "floating ghost (two arms, held mask, no mouth, no legs)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (mask mean x left of the red eye); back eye is on the right, away-right as drawn",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
