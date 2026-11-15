#!/usr/bin/env python3
"""Sewaddle normal (0540) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x31-60 y29-63, 664 px, alpha 0/255,
14 colours, foot y63):
  One eye: white+grey at x47-48 y47-48, mean x 47.5.
  Mouth: orange (230,131,8) and brown (148,82,24), 15 px, x39-45 y49-52,
  mean x 42.0. Mouth x is left of the eye, so the still faces screen-left.
  The leaf hood is the green mass from the top, plus the outline that touches
  only that mass. Two soles reach y63: left x46-47, right x51-52.
  No wing, horn, fang, hand, tail, or pincer is drawn.

normal/back.png (bbox x39-66 y25-61, 606 px, foot y61, 8 colours, all in the
  front palette). No eye and no mouth. The gold/olive fleck mean x is 56,
  right of the body, so the back still looks away to the right. One sole
  at y61 x48-49. Placed 2 px lower so both soles share the front foot line.

Directions, front faces left, back faces away-right:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn. NW back mirrored.
  E is screen-right, W is screen-left, N is the back.
  Attack 5-8 is that back at equal scale: W and SW show N, S SE E show N
  flipped. Back rows stay the back. No |cos| squash and no in-plane spin.

  idle 16: feet planted, the leaf hood breathes.
  walk 16: the two drawn feet alternate. The back has one foot.
  attack: 0 still, 1-4 leaf rises and the mouth wedge and ring (front only),
          5-8 the back still, 9-12 the leaf and the near foot strike,
          11-12 impact, 13-15 recover.
  hurt 12: white flash, knock-back lean (<=38 deg) and recover.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_sewaddle_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0540_sewaddle/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0540_sewaddle/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0540_sewaddle"
VERSION = "sewaddle_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 192
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 24
BACK_DROP = 2
FRONT_FOOT = 63
BACK_FOOT = 61
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. Back as drawn looks away to the right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Roar peaks. The leaf strike is drawn at 34, above these, so finish()
# does not push a frame past the angle this script already paints.
PEAKS = {"leaf": 18.0, "kick": 8.0}

INK = (0, 0, 0)
DARK = (57, 98, 24)
GREY = (74, 74, 74)
MID = (90, 156, 49)
BROWN = (148, 82, 24)
LIME = (148, 205, 8)
OLIVE = (156, 123, 8)
SILVER = (205, 205, 213)
GOLD = (213, 172, 0)
ORANGE = (230, 131, 8)
CREAM = (255, 246, 172)
YELLOW = (255, 222, 49)
WHITE = (255, 255, 255)
PEACH = (255, 172, 74)
GREEN = (LIME, MID, DARK)
FACE_COLS = (YELLOW, CREAM, GOLD, OLIVE)
MOUTH_COLS = (ORANGE, BROWN)

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


def flood(allowed, seeds):
    out = seeds & allowed
    while True:
        nxt = dilate(out, 1) & allowed
        if np.array_equal(nxt, out):
            return out
        out = nxt


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
    groups = component_list(mask)
    if not groups:
        return 0
    return max(len(g) for g in groups)


def require_piece(name, mask):
    if not mask.any():
        raise SystemExit(f"{name} is empty")
    if largest_component(mask) != int(mask.sum()):
        raise SystemExit(f"{name} fell apart")


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    green = any_color(img, GREEN)
    ink = is_color(img, INK)
    face = any_color(img, FACE_COLS)
    eye = any_color(img, (WHITE, GREY)) & (yy >= 46) & (yy <= 49) & (xx >= 46) & (xx <= 49)
    mouth = any_color(img, MOUTH_COLS) & (yy <= 55) & (xx <= 46)
    leaf = flood(green & (yy <= 55), green & (yy <= 34))
    sole_l = op & (yy >= 62) & (xx <= 48)
    sole_r = op & (yy >= 62) & (xx >= 50)
    band = ~face & ~mouth & ~eye
    foot_l = flood(op & (yy >= 58) & (xx <= 49) & band, sole_l)
    foot_r = flood(op & (yy >= 58) & (xx >= 49) & band & ~foot_l, sole_r)
    body_green = green & ~leaf
    blocked = face | mouth | eye | foot_l | foot_r | body_green
    for _ in range(4):
        near = ink & dilate(leaf, 1) & ~leaf & ~dilate(blocked, 1)
        if not near.any():
            break
        leaf |= near
    torso = op & ~leaf & ~foot_l & ~foot_r
    masks = {"leaf": leaf, "foot_l": foot_l, "foot_r": foot_r, "torso": torso}
    check_partition(op, masks)
    for name in ("leaf", "foot_l", "foot_r", "torso"):
        require_piece(name, masks[name])
        if name != "torso" and not (masks[name] & dilate(torso, 1)).any():
            raise SystemExit(f"front {name} does not touch the body")
    if int(eye.sum()) != 4 or (eye & ~torso).any():
        raise SystemExit("front eye left the head")
    if int(mouth.sum()) != 15 or (mouth & ~torso).any():
        raise SystemExit("front mouth left the head")
    eye_x = float(np.nonzero(eye)[1].mean())
    mouth_x = float(np.nonzero(mouth)[1].mean())
    if mouth_x >= eye_x - 1.0:
        raise SystemExit(f"front does not face left: mouth {mouth_x:.1f} eye {eye_x:.1f}")
    if not sole_l.any() or not (foot_l & (yy == FRONT_FOOT)).any():
        raise SystemExit("front left sole left the ground")
    if not (foot_r & (yy == FRONT_FOOT)).any():
        raise SystemExit("front right sole left the ground")
    if int(leaf.sum()) < 200:
        raise SystemExit("front leaf hood is missing")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    for c in (YELLOW, CREAM, WHITE, GREY, SILVER, PEACH):
        if is_color(img, c).any():
            raise SystemExit(f"back gained a front-only colour {c}")
    accent = any_color(img, (GOLD, OLIVE))
    if not accent.any() or float(np.nonzero(accent)[1].mean()) < 54:
        raise SystemExit("back accent is not on the right")
    sole = op & (yy >= 60) & (xx >= 45) & (xx <= 52)
    foot = flood(op & (yy >= 58) & (xx >= 44) & (xx <= 55), sole)
    torso = op & ~foot
    masks = {"foot": foot, "torso": torso}
    check_partition(op, masks)
    require_piece("foot", foot)
    require_piece("torso", torso)
    if not (foot & dilate(torso, 1)).any():
        raise SystemExit("back foot does not touch the body")
    if not (foot & (yy == BACK_FOOT)).any():
        raise SystemExit("back sole left the ground")
    if is_color(img, ORANGE).any() and is_color(img, YELLOW).any():
        raise SystemExit("back grew a mouth")
    return masks


def place(arr, dy=0):
    extra = arr.shape[2:]
    out = np.zeros((CELL_H, CELL_W) + extra, arr.dtype)
    y0 = OFF_Y + int(dy)
    x0 = OFF_X
    h, w = arr.shape[:2]
    if y0 < 0 or x0 < 0 or y0 + h > CELL_H or x0 + w > CELL_W:
        raise SystemExit(f"place overflow y {y0} x {x0}")
    out[y0:y0 + h, x0:x0 + w] = arr
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
    """Contact pixel of the part nearest the torso."""
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
    near = d <= d.min() + 0.5
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
    dy = BACK_DROP if kind == "back" else 0
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
        facing = -1.0 if not mirror else 1.0
    else:
        facing = 1.0 if not mirror else -1.0
    pivots = {}
    signs = {}
    for name in list(layers):
        if name == "torso":
            continue
        depth = 5 if name == "leaf" else 4
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth
        )
        pivots[name] = pivot_of(placed[name], torso_m)
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    mouth = any_color(canvas, MOUTH_COLS) & (canvas[..., 3] >= 128)
    # The back has orange-brown feet and no mouth. The mouth sits above the feet.
    if kind == "front":
        my, mx = np.nonzero(mouth)
        # Feet are the low orange-brown. The mouth is the high cluster.
        keep = my < (OFF_Y + 56)
        if not keep.any():
            raise SystemExit("front mouth missing after place")
        pivots["mouth"] = (float(mx[keep].mean()), float(my[keep].mean()))
        has_mouth = True
    else:
        pivots["mouth"] = pivots["center"]
        has_mouth = False
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas, "has_mouth": has_mouth,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, leaf=0.0, leaf_mode="up", kick=0.0,
        step_l=(0, 0), step_r=(0, 0), step_b=(0, 0),
        lean=0.0, knock=0, flash=False, wedge=0, ring=-1, spark=0,
        turned=False, spin=0, head_dx=0, head_dy=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay on the ground. The hood breathes. Frame 0 is the still.
        p["leaf"] = wave(f, n, 6.0, phase=0.4)
        p["leaf_mode"] = "up"
    elif action == "walk":
        sa = math.sin(2 * math.pi * f / n)
        lift_l = max(0.0, sa)
        lift_r = max(0.0, -sa)
        p["step_l"] = (int(round(2 * lift_l)), -int(round(2 * lift_l)))
        p["step_r"] = (int(round(2 * lift_r)), -int(round(2 * lift_r)))
        p["step_b"] = p["step_l"]
        p["leaf"] = wave(f, n, 8.0, phase=0.3)
        p["leaf_mode"] = "up"
        p["bdx"] = int(round(wave(f, n, 1.0, cycles=2.0)))
    elif action == "attack":
        # 0 rest. 1-4 raise the hood and, on the front, the mouth.
        # 5-8 the back still, cleared again after finish().
        # 9-12 the hood and the near foot strike. 11-12 impact. 13-15 recover.
        leaf = [0, 8, 14, 18, 18, 0, 0, 0, 0, 30, 34, 34, 34, 16, 6, 0]
        kick = [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 12, 12, 6, 2, 0]
        bdx = [0, -1, -1, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        wedge = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ring = [-1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        p["leaf"] = float(leaf[f])
        p["kick"] = float(kick[f])
        p["bdx"] = int(bdx[f])
        p["wedge"] = int(wedge[f])
        p["ring"] = int(ring[f])
        p["leaf_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["turned"] = 5 <= f <= 8
        if 9 <= f <= 12:
            p["step_l"] = (2, -1)
            p["step_b"] = (1, 0)
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    # 5-8 is the back still at equal scale. Drop the wind-up finish() restored.
    if action == "attack" and 5 <= f <= 8:
        p["leaf"] = 0.0
        p["kick"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["step_b"] = (0, 0)
        p["wedge"] = 0
        p["ring"] = -1
        p["spark"] = 0
        p["spin"] = 0
        p["head_dx"] = 0
        p["head_dy"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["turned"] = True
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


def step_of(p, name):
    if name == "foot_l":
        return p["step_l"]
    if name == "foot_r":
        return p["step_r"]
    if name == "foot":
        return p["step_b"]
    return (0, 0)


def angle_of(rig, p, name):
    if name == "leaf":
        return part_angle(rig, name, p["leaf_mode"], p["leaf"])
    if name in ("foot_l", "foot"):
        return part_angle(rig, name, "fwd", p["kick"])
    return 0.0


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])

    def one(name):
        sx, sy = step_of(p, name)
        ang = angle_of(rig, p, name)
        dx = bdx + int(round(fc * sx))
        dy = bdy + int(sy)
        return moved(L[name], ang, piv[name], dx, dy)

    paste(out, shift(L["torso"], bdx, bdy))
    for name in ("foot_r", "foot_l", "foot", "leaf"):
        if name in L:
            paste(out, one(name))
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = any_color(out, (INK, GREY, DARK))
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def leaf_tip(rig, p):
    name = "leaf" if "leaf" in rig["layers"] else None
    if name is None:
        return rig["pivots"]["center"]
    fc = rig["facing"]
    layer = moved(
        rig["layers"][name],
        angle_of(rig, p, name),
        rig["pivots"][name],
        int(round(fc * p["bdx"])),
        int(round(p["bdy"])),
    )
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def mouth_wedge(img, origin, fc, age):
    """A few pixels in front of the existing mouth. Empty pixels only."""
    ox, oy = origin
    cols = (INK, ORANGE, YELLOW)
    for i in range(max(0, int(age))):
        x = ox + fc * (2 + i)
        span = 1 if i == 0 else 2
        for dy in range(-span, span + 1):
            put(img, x, oy + dy, cols[min(i, 2)])


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, CREAM, YELLOW)
    rad = 6 + int(age) * 3
    for deg in range(-48, 49, 8):
        a = math.radians(deg)
        x = ox + fc * rad * math.cos(a)
        y = oy + rad * math.sin(a) * 0.55
        put(img, x, y, cols[int(age) % 3])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, CREAM, LIME)
    r = 2 + int(age) * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (1, -r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def rig_for(rigs, d, p):
    """Attack 5-8 shows this side's back at equal scale. Back rows stay."""
    if not p.get("turned"):
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", mirror)]


def moving(p):
    if p["step_l"] != (0, 0) or p["step_r"] != (0, 0) or p["step_b"] != (0, 0):
        return True
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["leaf"]) >= 0.45 or abs(p["kick"]) >= 0.45:
        return True
    return False


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["leaf"]) >= 0.45 or abs(p["kick"]) >= 0.45:
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
            body = rotate_nearest(body, -fc * p["lean"], rear_pivot(body, fc))
        if rotated(p):
            body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if rig["has_mouth"] and not p["turned"]:
        mx, my = rig["pivots"]["mouth"]
        origin = (mx + int(round(fc * p["bdx"])), my + int(round(p["bdy"])))
        if p["wedge"]:
            mouth_wedge(img, origin, fc, p["wedge"])
        if p["ring"] >= 0:
            roar_rings(img, origin, fc, p["ring"])
        if p["wedge"] or p["ring"] >= 0:
            img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"] and "leaf" in rig["layers"]:
        tx, ty = leaf_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def limb_names(rig):
    return [n for n in rig["layers"] if n != "torso"]


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    for name in limb_names(rig):
        sx, sy = step_of(p, name)
        ang = angle_of(rig, p, name)
        layer = moved(
            rig["layers"][name], ang, rig["pivots"][name],
            bdx + int(round(fc * sx)), bdy + int(sy),
        )
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


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
    y0 = max(0, int(ys.min()) - pad - 14)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 18)
    x0 = max(0, int(xs.min()) - pad - 18)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 18)
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


def frame_gate(img, pal):
    """One cell: off-palette count, partial alpha, edge touch."""
    partial = int(((img[..., 3] != 0) & (img[..., 3] != 255)).sum())
    a = img[..., 3] == 255
    edge = 1 if (a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any()) else 0
    off = 0
    if a.any():
        used = {tuple(int(v) for v in c) for c in np.unique(img[a][:, :3], axis=0)}
        off = len(used - pal)
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
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs["W"]["facing"] != -1.0 or rigs["E"]["facing"] != 1.0:
        raise SystemExit("front facing is not left-as-drawn")
    if rigs["N"]["facing"] != 1.0 or rigs["NW"]["facing"] != -1.0:
        raise SystemExit("back facing is not away-right as drawn")
    if not rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("mouth is on the wrong view")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "leaf": (40, 180, 60, 255),
        "foot_l": (240, 180, 40, 255),
        "foot_r": (240, 80, 40, 255),
        "foot": (240, 180, 40, 255),
        "torso": (180, 190, 80, 255),
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
        "dex": 540, "form": "normal", "variant": "sewaddle", "gen": "gen5",
        "dir": "monsterSprite/gen5/0540_sewaddle/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def measure_synth(rigs, palette):
    pal = {tuple(int(v) for v in c) for c in palette}
    off = 0
    partial = 0
    edge = 0
    bodies = []
    turn_diff = 0
    for d in DIRS:
        for act in ACTIONS:
            for f in range(FRAMES[act]):
                img = synth(rigs, d, act, f, palette, effects=True)
                o, p, e = frame_gate(img, pal)
                off += o
                partial += p
                edge += e
                if act in ("attack", "hurt", "walk"):
                    bodies.append(largest_fraction(img))
                if act == "attack" and 5 <= f <= 8:
                    rig = rig_for(rigs, d, pose(act, f))
                    turn_diff += int(np.any(img != rig["canvas"], axis=2).sum())
    return {
        "off_palette": off,
        "partial_alpha": partial,
        "edge": edge,
        "turn_diff": turn_diff,
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
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, flush=True)
    print(
        "parts",
        {k: int(v.sum()) for k, v in fm.items()},
        {k: int(v.sum()) for k, v in bm.items()},
        "palette", int(len(palette)),
        flush=True,
    )

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
    rest_sum = int(sum(rest_diff.values()))
    print("rest diff", rest_diff, flush=True)
    if rest_sum:
        raise SystemExit("frame 0 does not match the still")

    if args.samples:
        gate = measure_synth(rigs, palette)
        lost = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["leaf"], 1), round(p["kick"], 1), p["leaf_mode"],
                        p["turned"], p["spin"], p["bdx"], p["wedge"], p["ring"], p["spark"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        print(
            f"gate rest_diff_sum={rest_sum} off_palette={gate['off_palette']} "
            f"partial_alpha={gate['partial_alpha']} edge={gate['edge']} "
            f"turn_diff={gate['turn_diff']} body_main_min={gate['body_main_min']}",
            flush=True,
        )
        box = crop_box(rigs["S"]["canvas"])
        y0, y1, x0, x1 = box
        for d in ("S", "W", "N", "E"):
            rows = []
            for act in ACTIONS:
                frames = []
                for f in range(16):
                    if f < FRAMES[act]:
                        frames.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
                    else:
                        frames.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
                rows.append(tile_grid(frames, 16))
            sheet = np.concatenate(rows, axis=0)
            z = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
            Image.fromarray(z, "RGBA").save(work / f"samples_{d}.png")
        print("samples", work, flush=True)
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
    box = crop_box(rigs["S"]["canvas"])
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.concatenate(strips, axis=0)
    motion = np.repeat(np.repeat(motion, 2, 0), 2, 1)
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
    ok = (not any(rest_diff.values()) and not any(idle0.values()) and not off
          and partial == 0 and not clip and min(bodies) >= 0.9)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    print(
        f"gate rest_diff_sum={int(sum(rest_diff.values()))} off_palette={len(off)} "
        f"partial_alpha={partial} edge={len(clip)} idle0={int(sum(idle0.values()))}",
        flush=True,
    )
    meta = {
        "generator": "forge_gen5_sewaddle_actions", "version": VERSION, "key": KEY,
        "dex": 540, "variant": "sewaddle", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "larva (leaf hood, one mouth, one eye, two stubby feet, no hands)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front mouth mean x 42.0 is left of the eye mean x 47.5; back accent is screen-right",
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
