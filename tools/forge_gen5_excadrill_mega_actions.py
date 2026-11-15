#!/usr/bin/env python3
"""Mega Excadrill (0530) 8-direction actions from this form's own stills.

Measured on mega/excadrill-mega/front.png (96x96, bbox x0-94 y11-84, 22 colours,
alpha 0/255) and back.png (bbox x0-95 y18-78, 16 colours, all inside the front palette):

  Front faces screen-left. The pink mouth tip is x57.8 y41-42 (5 px). The two
  red eyes on y49 sit at x59-60 and x64-65, centre x62.0, to the right of the mouth.
  White tusks sit in the mouth (x57-66 y41-46). There is a mouth, so the roar
  uses a wedge and a ring. The back still has no mouth, so those directions do not
  grow one.

  Head drill: metal cone y11-40 x22-75, red blade on its right edge.
  Near claw (screen-left, the big drill): x0-48 y44-79.
  Far claw (screen-right): x71-94 y45-84, the lowest pixel of the still.
  Feet are stubs under the belly, split by a transparent gap at y72:
  left foot x46-56 y72-76, right toe x68-73 y72-74. No wings, no tail.

  Back as drawn has the big claw on the screen-right (x58-95) and the small claw
  on the screen-left (x0-21). Feet: left x18-25 y74-76, right x27-50 y74-78.
  Bottoms are aligned to the front's lowest pixel so both stand on one line.

Directions: S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
N NE back as drawn. NW back mirrored. E is screen-right, W is screen-left, N is the back.
A turn keeps that back at equal scale. The body is never laid flat and never scaled by |cos|.

  idle 16: feet planted, drill and claws sway. Frame 0 is the still.
  walk 16: the two stubs step. Claws swing. No extra legs.
  attack 16: 0 still, 1-4 roar (drill and claws up, mouth wedge and ring on the front),
             5-8 the matching back still, 9-12 claw chop and drill thrust (strike <= 34 deg),
             13-15 recover.
  hurt 12: lean at most 38 deg about the rear foot, then recover.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen5_excadrill_mega_actions.py [--samples] [--force] [--publish]
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
SRC_DIR = ROOT / "monsterSprite/gen5/0530_excadrill/mega/excadrill-mega"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0530_excadrill/mega/excadrill-mega/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0530_excadrill-mega"
VERSION = "excadrill-mega_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 224, 196
OFF_X = 64
OFF_Y = 52
FRONT_FOOT = 84
BACK_FOOT = 78
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left (mouth x < eye x). Its big claw is on the left.
# Back as drawn has that big claw on the right, so it matches the right-facing front.
# S (front mirrored) turns into N (back as drawn).
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Wind-up peaks. The claw chop is drawn at 34 deg, above these.
# Frames 5-8 are cleared after finish() so the turned still is not swung.
PEAKS = {"drill": 20.0, "claw_l": 20.0, "claw_r": 14.0}
LIMBS = ("leg_l", "leg_r", "claw_l", "claw_r", "drill")

INK = (16, 16, 16)
BROWN = (65, 49, 41)
BROWN_L = (97, 79, 71)
BROWN_D = (41, 32, 24)
BROWN_COLS = (BROWN, BROWN_L, BROWN_D)
PINK = (238, 131, 123)
PINK_D = (205, 115, 106)
PINK_COLS = (PINK, PINK_D)
WHITE = (255, 255, 255)
SILVER = (235, 235, 235)
STEEL = (190, 193, 199)
STEEL_M = (150, 153, 160)
STEEL_D = (105, 108, 114)
STEEL_K = (80, 83, 89)
ORANGE = (255, 122, 78)
RED = (233, 97, 66)
RUST = (210, 83, 57)
EYE = (189, 65, 65)
EYE_D = (131, 65, 65)
LEG_COLS = BROWN_COLS + (INK,)

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


def grow(seed, allowed, n):
    out = seed & allowed
    for _ in range(n):
        nxt = dilate(out, 1) & allowed
        if int(nxt.sum()) == int(out.sum()):
            return out
        out = nxt
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


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return len(components(op)[0]) / float(op.sum())


def check_partition(op, masks):
    acc = np.zeros_like(op)
    for name, m in masks.items():
        if (m & ~op).any():
            raise SystemExit(f"{name} left the still")
        if (m & acc).any():
            raise SystemExit(f"overlap in {name}")
        acc |= m
    if int(acc.sum()) != int(op.sum()):
        raise SystemExit(f"parts do not partition the still: {int(acc.sum())} vs {int(op.sum())}")


def drop_specks(mask, speck=3):
    """Leave 1-3 px flakes in the torso. A real foot stays."""
    groups = components(mask)
    if len(groups) <= 1:
        return mask
    out = np.zeros_like(mask)
    for g in groups:
        if len(g) <= speck:
            continue
        for y, x in g:
            out[y, x] = True
    return out


def require_piece(name, mask, minimum):
    if int(mask.sum()) < minimum:
        raise SystemExit(f"{name} shrank: {int(mask.sum())}")
    sizes = [len(g) for g in components(mask)]
    if sizes[0] != int(mask.sum()):
        raise SystemExit(f"{name} fell apart: {sizes[:4]}")


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


def _rotate_crop(layer, ang, pivot):
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


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest rotation. Positive is clockwise. No width squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    a = layer[..., 3] >= 128
    ys, xs = np.nonzero(a)
    rad = float(np.hypot(xs - pivot[0], ys - pivot[1]).max()) + 3.0
    x0 = max(0, int(math.floor(pivot[0] - rad)))
    x1 = min(layer.shape[1], int(math.ceil(pivot[0] + rad)) + 1)
    y0 = max(0, int(math.floor(pivot[1] - rad)))
    y1 = min(layer.shape[0], int(math.ceil(pivot[1] + rad)) + 1)
    sub = layer[y0:y1, x0:x1]
    rot = _rotate_crop(sub, ang, (pivot[0] - x0, pivot[1] - y0))
    out = np.zeros_like(layer)
    out[y0:y1, x0:x1] = rot
    return out


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
    """Grow this layer's colours into allowed pixels. The part covers them at rest."""
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        for name, src_slice in (
            ("up", src[1:]),
            ("down", src[:-1]),
            ("left", src[:, 1:]),
            ("right", src[:, :-1]),
        ):
            grown = np.zeros_like(src)
            if name == "up":
                grown[:-1] = src_slice
            elif name == "down":
                grown[1:] = src_slice
            elif name == "left":
                grown[:, :-1] = src_slice
            else:
                grown[:, 1:] = src_slice
            take = (acc[..., 3] < 128) & allowed & (grown[..., 3] >= 128)
            acc[take] = grown[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    brown = any_color(img, BROWN_COLS)
    pink = any_color(img, PINK_COLS)
    allow = op & (yy <= 40) & (xx >= 20) & (xx <= 78) & ~brown & ~pink
    drill = grow(allow & (yy <= 18), allow, 40)
    cla = op & (yy >= 44) & (xx <= 52) & ~drill & ~brown
    claw_l = grow(cla & (xx <= 6), cla, 50)
    cra = op & (yy >= 44) & (xx >= 68) & ~drill & ~brown & ~claw_l
    claw_r = grow(cra & (xx >= 88), cra, 50)
    feet = any_color(img, LEG_COLS) & (yy >= 72) & ~claw_l & ~claw_r & ~drill
    leg_l = drop_specks(feet & (xx >= 46) & (xx <= 56))
    leg_r = drop_specks(feet & (xx >= 68) & (xx <= 73))
    used = drill | claw_l | claw_r | leg_l | leg_r
    torso = op & ~used
    masks = {
        "drill": drill, "claw_l": claw_l, "claw_r": claw_r,
        "leg_l": leg_l, "leg_r": leg_r, "torso": torso,
    }
    check_partition(op, masks)
    require_piece("drill", drill, 700)
    require_piece("claw_l", claw_l, 900)
    require_piece("claw_r", claw_r, 400)
    require_piece("leg_l", leg_l, 20)
    require_piece("leg_r", leg_r, 6)
    if int((pink & torso).sum()) != int(pink.sum()) or int(pink.sum()) < 4:
        raise SystemExit("mouth left the face")
    eye = (is_color(img, EYE) | is_color(img, EYE_D)) & (yy == 49) & (xx >= 54) & (xx <= 70)
    if int((eye & torso).sum()) != int(eye.sum()) or int(eye.sum()) < 4:
        raise SystemExit("eyes left the face")
    tip = float(np.nonzero(pink)[1].mean())
    ex = float(np.nonzero(eye)[1].mean())
    if tip >= ex:
        raise SystemExit(f"front does not face left: mouth {tip:.1f} eye {ex:.1f}")
    if int(np.nonzero(op)[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    brown = any_color(img, BROWN_COLS)
    allow = op & (yy <= 41) & (xx >= 18) & (xx <= 74) & ~brown
    drill = grow(allow & (yy <= 22), allow, 40)
    cra = op & (yy >= 42) & (xx >= 58) & ~drill & ~brown
    claw_r = grow(cra & (xx >= 90), cra, 50)
    cla = op & (yy >= 46) & (xx <= 24) & ~drill & ~brown & ~claw_r
    claw_l = grow(cla & (xx <= 6) & (yy >= 56), cla, 40)
    feet = any_color(img, LEG_COLS) & (yy >= 74) & ~claw_l & ~claw_r & ~drill
    # Transparent gap at x25-26 separates the short left foot from the long right foot.
    leg_l = drop_specks(feet & (xx <= 25))
    leg_r = drop_specks(feet & (xx >= 27))
    used = drill | claw_l | claw_r | leg_l | leg_r
    torso = op & ~used
    masks = {
        "drill": drill, "claw_l": claw_l, "claw_r": claw_r,
        "leg_l": leg_l, "leg_r": leg_r, "torso": torso,
    }
    check_partition(op, masks)
    require_piece("drill", drill, 500)
    require_piece("claw_l", claw_l, 60)
    require_piece("claw_r", claw_r, 600)
    require_piece("leg_l", leg_l, 12)
    require_piece("leg_r", leg_r, 30)
    if any_color(img, PINK_COLS).any():
        raise SystemExit("back grew a mouth")
    if float(np.nonzero(claw_r)[1].mean()) <= float(np.nonzero(claw_l)[1].mean()):
        raise SystemExit("back big claw is not on the right")
    if int(np.nonzero(op)[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line changed")
    return masks


def place(arr, y_shift):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    y0 = OFF_Y + y_shift
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


def pivot_of(layer, torso_layer):
    part = layer[..., 3] >= 128
    torso = torso_layer[..., 3] >= 128
    touch = part & dilate(torso, 2)
    if not touch.any():
        touch = part & dilate(torso, 6)
    if not touch.any():
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(torso)
    cx, cy = float(tx.mean()), float(ty.mean())
    d = (xs.astype(np.float32) - cx) ** 2 + (ys.astype(np.float32) - cy) ** 2
    near = d <= d.min() + 8.0
    return float(xs[near].mean()), float(ys[near].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs - pivot[0]) ** 2 + (ys - pivot[1]) ** 2
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


def build_view(img, masks, kind, mirror):
    y_shift = 0 if kind == "front" else (FRONT_FOOT - BACK_FOOT)
    canvas = place(img, y_shift)
    placed = {k: place(v, y_shift) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    base_torso = layers["torso"].copy()
    torso_m = placed["torso"]
    depth = {"leg_l": 5, "leg_r": 5, "claw_l": 8, "claw_r": 8, "drill": 8}
    for name in LIMBS:
        if not placed[name].any():
            continue
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth[name]), depth[name]
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in LIMBS:
        if not placed[name].any():
            continue
        pivots[name] = pivot_of(layers[name], base_torso)
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    pink = any_color(canvas, PINK_COLS)
    mouth = None
    if pink.any():
        py, px = np.nonzero(pink)
        if facing > 0:
            i = int(px.argmax())
        else:
            i = int(px.argmin())
        mouth = (float(px[i]), float(py[i]))
    counts = {n: int(placed[n].sum()) for n in ("claw_l", "claw_r")}
    strike = max(counts, key=counts.get)
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
        "has_mouth": mouth is not None, "mouth": mouth, "strike": strike,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        drill=0.0, claw_l=0.0, claw_r=0.0,
        drill_mode="up", claw_mode="up",
        step_l=(0, 0), step_r=(0, 0),
        lean=0.0, knock=0, flash=False,
        wedge=0, ring=0, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        p["drill"] = wave(f, n, 5.0, phase=0.4)
        p["claw_l"] = wave(f, n, 3.0, phase=0.8)
        p["claw_r"] = wave(f, n, 3.0, phase=0.8 + math.pi)
        p["drill_mode"] = "up"
        p["claw_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        p["step_l"] = (int(round(2 * max(0.0, sa))), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * max(0.0, -sa))), -int(round(3 * max(0.0, -sa))))
        p["claw_l"] = wave(f, n, 8.0, phase=0.3)
        p["claw_r"] = wave(f, n, 8.0, phase=0.3 + math.pi)
        p["drill"] = wave(f, n, 4.0, phase=0.6)
        p["claw_mode"] = "fwd"
        p["drill_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 raise the drills and open the mouth. 5-8 the back still.
        # 9-12 chop with the claws and thrust the head drill. 13-15 recover.
        drill = [0, 8, 12, 16, 20, 0, 0, 0, 0, 18, 24, 30, 28, 12, 4, 0]
        claw_l = [0, 8, 12, 16, 20, 0, 0, 0, 0, 22, 28, 34, 34, 16, 6, 0]
        claw_r = [0, 8, 10, 12, 14, 0, 0, 0, 0, 16, 22, 28, 26, 10, 4, 0]
        p["drill"] = float(drill[f])
        p["claw_l"] = float(claw_l[f])
        p["claw_r"] = float(claw_r[f])
        strike = 9 <= f <= 14
        p["drill_mode"] = "fwd" if strike else "up"
        p["claw_mode"] = "fwd" if strike else "up"
        p["turned"] = 5 <= f <= 8
        if 1 <= f <= 4:
            p["wedge"] = (2, 3, 4, 4)[f - 1]
            p["ring"] = (1, 2, 3, 3)[f - 1]
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.64, 0.42, 0.24, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["drill"] = 6.0 * env
        p["claw_l"] = 4.0 * env
        p["claw_r"] = 4.0 * env
        p["drill_mode"] = "up"
        p["claw_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["drill"] = 0.0
        p["claw_l"] = 0.0
        p["claw_r"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["wedge"] = 0
        p["ring"] = 0
        p["spark"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["turned"] = True
    return p


def part_angle(rig, name, mode, amp):
    if abs(amp) < 0.45 or name not in rig["signs"]:
        return 0.0
    sign = rig["signs"][name]
    if mode == "fwd":
        return sign["fwd"] * amp
    return sign["up"] * amp


def moved(layer, ang, pivot, dx, dy):
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, pivot)
    return shift(layer, dx, dy)


def limb_layers(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    parts = {}
    for name in LIMBS:
        if name not in piv:
            continue
        if name == "leg_l":
            sx, sy = p["step_l"]
            ang = 0.0
        elif name == "leg_r":
            sx, sy = p["step_r"]
            ang = 0.0
        elif name == "drill":
            sx, sy = 0, 0
            ang = part_angle(rig, name, p["drill_mode"], p["drill"])
        else:
            sx, sy = 0, 0
            ang = part_angle(rig, name, p["claw_mode"], p["claw_l" if name == "claw_l" else "claw_r"])
        dx = bdx + int(round(fc * sx))
        dy = bdy + int(sy)
        parts[name] = moved(L[name], ang, piv[name], dx, dy)
    return parts


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    parts = limb_layers(rig, p)
    present = [n for n in ("leg_l", "leg_r") if n in parts]
    if present:
        foot_x = {}
        for n in present:
            xs = np.nonzero(rig["layers"][n][..., 3] >= 128)[1]
            foot_x[n] = float(xs.mean()) if len(xs) else 0.0
        if fc > 0:
            near = max(present, key=lambda n: foot_x[n])
        else:
            near = min(present, key=lambda n: foot_x[n])
        order = [n for n in present if n != near]
        order.append(near)
        for n in order:
            paste(out, parts[n])
    claws = [n for n in ("claw_l", "claw_r") if n in parts]
    claws.sort(key=lambda n: int((rig["layers"][n][..., 3] >= 128).sum()))
    for n in claws:
        paste(out, parts[n])
    if "drill" in parts:
        paste(out, parts["drill"])
    return out


def rear_pivot(rig, body):
    """Rear stub, not the drill claw that also reaches the ground."""
    fc = rig["facing"]
    names = [n for n in ("leg_l", "leg_r") if n in rig["pivots"]]
    if not names:
        a = body[..., 3] >= 128
        ys, xs = np.nonzero(a)
        yb = int(ys.max())
        band = ys >= yb - 2
        xb = int(xs[band].min()) if fc > 0 else int(xs[band].max())
        return float(xb), float(yb)
    def cx(n):
        xs = np.nonzero(rig["layers"][n][..., 3] >= 128)[1]
        return float(xs.mean())
    rear = min(names, key=cx) if fc > 0 else max(names, key=cx)
    layer = rig["layers"][rear]
    a = layer[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 1
    return float(xs[band].mean()), float(yb)


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK)
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def mouth_fx(img, origin, fc, wedge, ring):
    ox, oy = origin
    cols = (INK, PINK, PINK_D, WHITE)
    for i in range(1, wedge + 1):
        put(img, ox + fc * i, oy + (i // 2), cols[i % 4])
        if i > 1:
            put(img, ox + fc * i, oy + (i // 2) - 1, INK)
    if ring <= 0:
        return
    rad = 6 + ring * 4
    for deg in range(-50, 51, 8):
        a = math.radians(deg)
        x = ox + fc * (4 + rad * math.cos(a))
        y = oy + rad * math.sin(a) * 0.72
        col = (ORANGE, WHITE, STEEL)[(deg // 8) % 3]
        put(img, x, y, col)


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, ORANGE, SILVER)
    r = 3 + age * 3
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (1, -r // 2), (1, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def weapon_tip(rig, p):
    parts = limb_layers(rig, p)
    name = rig["strike"]
    layer = parts.get(name)
    if layer is None or not np.any(layer[..., 3] >= 128):
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    fc = rig["facing"]
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name, layer in limb_layers(rig, p).items():
        limb = layer[..., 3] >= 128
        if not limb.any():
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    return rigs[("back" if kind == "front" else "front", not mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    moved_part = abs(p["drill"]) >= 0.45 or abs(p["claw_l"]) >= 0.45 or abs(p["claw_r"]) >= 0.45
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(rig, body)
        piv = (piv[0] - fc * p["knock"], piv[1])
        body = rotate_nearest(body, -fc * p["lean"], piv)
    if moved_part or abs(p["lean"]) > 0.4:
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    if rig["has_mouth"] and rig["mouth"] is not None and (p["wedge"] or p["ring"]) and not p["turned"]:
        mx, my = rig["mouth"]
        mouth_fx(img, (mx + int(round(fc * p["bdx"])), my + int(p["bdy"])), fc, p["wedge"], p["ring"])
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = weapon_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
    return snap(img, palette)


def metrics_of(img, palette_set):
    a = img[..., 3]
    partial = int(((a != 0) & (a != 255)).sum())
    op = a == 255
    edge = bool(op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any())
    off = 0
    if op.any():
        cols = {tuple(int(v) for v in c) for c in np.unique(img[op][:, :3], axis=0)}
        off = len(cols - palette_set)
    return off, partial, edge


def tile_grid(tiles, cols, bg=(15, 20, 31, 255)):
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    out = np.full((rows * th, cols * tw, 4), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        m = t[..., 3] >= 128
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
    return out


def crop_box(canvas, pad=8):
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(canvas.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(canvas.shape[1], int(xs.max()) + 1 + pad)
    return y0, y1, x0, x1


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


def preview_image(sheets):
    pick = {"idle": 0, "walk": 4, "attack": 11, "hurt": 3}
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
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    fa = rigs[("front", False)]["canvas"]
    ba = rigs[("back", False)]["canvas"]
    if int(np.nonzero(fa[..., 3] >= 128)[0].max()) != int(np.nonzero(ba[..., 3] >= 128)[0].max()):
        raise SystemExit("front and back feet do not share a line")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "drill": (80, 180, 255, 255), "claw_l": (255, 80, 40, 255),
        "claw_r": (255, 180, 40, 255), "leg_l": (80, 220, 80, 255),
        "leg_r": (180, 80, 220, 255), "torso": (90, 90, 90, 255),
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
        "dex": 530, "form": "mega", "variant": "excadrill-mega", "gen": "gen5",
        "dir": "monsterSprite/gen5/0530_excadrill/mega/excadrill-mega",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


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
    pal_set = {tuple(int(v) for v in c) for c in palette}
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS},
          "mouth", rigs["W"]["has_mouth"], rigs["N"]["has_mouth"], flush=True)

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
        p = pose("attack", 6)
        rig = rig_for(rigs, d, p)
        img = synth(rigs, d, "attack", 6, palette, effects=False)
        turn_diff[d] = int(np.any(img != rig["canvas"], axis=2).sum())
    print("turn diff", turn_diff, flush=True)
    if any(turn_diff.values()):
        raise SystemExit("turned frame is not the back still")

    if args.samples:
        y0, y1, x0, x1 = crop_box(rigs["S"]["canvas"])
        off_n = 0
        partial_n = 0
        edge_n = 0
        edge_list = []
        lost = []
        blank = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
        for d in DIRS:
            rows = []
            keep = d in ("S", "W", "N", "E")
            for act in ACTIONS:
                frames = []
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
                    img = synth(rigs, d, act, f, palette)
                    off, partial, edge = metrics_of(img, pal_set)
                    off_n += off
                    partial_n += partial
                    if edge:
                        edge_n += 1
                        edge_list.append(f"{act}:{d}:{f}")
                    groups = components(img[..., 3] >= 128)
                    islands = [len(g) for g in groups[1:] if len(g) >= 12]
                    if islands:
                        lost.append(f"{act}:{d}:{f}:island{islands[0]}")
                    if keep:
                        frames.append(img[y0:y1, x0:x1])
                if keep:
                    while len(frames) < 16:
                        frames.append(blank.copy())
                    rows.append(tile_grid(frames, 16))
            if keep:
                sheet = np.concatenate(rows, axis=0)
                z = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
                Image.fromarray(z, "RGBA").save(work / f"samples_{d}.png")
            print("sampled", d, flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["drill"], 1), round(p["claw_l"], 1), round(p["claw_r"], 1),
                        p["claw_mode"], p["turned"], p["spin"], p["wedge"], p["ring"]))
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        print("rest_diff", rest_diff, flush=True)
        print("off_palette", off_n, flush=True)
        print("partial_alpha", partial_n, flush=True)
        print("edge", edge_n, edge_list[:12], flush=True)
        print("attack poses", atk, flush=True)
        print("hurt lean", leans, "max", max(leans), flush=True)
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

    op = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[op][:, :3], axis=0)}
    off = sorted(used - pal_set)
    partial = int(((master[..., 3] != 0) & (master[..., 3] != 255)).sum())
    clip = []
    bodies = []
    idle0 = {}
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = t[..., 3] == 255
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
                if act in ("attack", "hurt", "walk"):
                    bodies.append(largest_fraction(t))
    for di, d in enumerate(DIRS):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        idle0[d] = int(np.any(t != rigs[d]["canvas"], axis=2).sum())
    import gen5_publish
    bounds = gen5_publish.measure_idle_bounds(OUT_DIR / "sheet.png", CELL_W, CELL_H, rows=8)
    sprite_h = max(bounds[i + 3] for i in range(0, 32, 4))
    report = {
        "version": VERSION, "hash": digest, "rest_diff": rest_diff, "idle0": idle0,
        "off_palette_colors": len(off), "partial_alpha": partial, "edge_touching": clip[:12],
        "edge_n": len(clip), "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (not any(rest_diff.values()) and not any(idle0.values()) and not off
          and partial == 0 and not clip)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps({**report, "bounds": bounds}, ensure_ascii=False, indent=1))
    print("rest_diff", rest_diff, flush=True)
    print("off_palette", len(off), flush=True)
    print("partial_alpha", partial, flush=True)
    print("edge", len(clip), clip[:12], flush=True)
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_excadrill_mega_actions", "version": VERSION, "key": KEY,
        "dex": 530, "variant": "excadrill-mega", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped mole (head drill, two drill claws, two stub feet, open mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (mouth x 57.8 < eye x 62.0); back big claw is on the right",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    if args.publish:
        publish(bounds, sprite_h)
        print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
