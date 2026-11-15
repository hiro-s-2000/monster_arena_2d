#!/usr/bin/env python3
"""Buneary (0427) 8-direction action sheet from this form's 96x96 front.png.

Measured on normal/front.png: 885 opaque pixels, bbox x32-63 y19-75 (32x57),
12 colours, alpha already 0 or 255. No enclosed hole.

Biped. Two feet split at the ground: y74 (44-50) and (53-57), y75 (45-49)
and (54-56). The movable pads are only y72-75. No long legs are added.
One upright ear (cream tip y19-30, brown stalk, pink inner) ends at y41 x32-48.
One rolled ear: the single cream puff beside the cheek, x52-63 y42-52.
It is not a second upright ear and it is not the belly (belly cream starts ~y63).
Six pink/maroon pixels at x40-42 y52-54 are the snout mouth. Eyes stay on the face
(white at x38 y50 and x48 y52). The snout is left of the body centre, so the still
faces left. Unflipped dirs are N NW W SW. S SE E NE are mirrored. E faces right.

  idle   : ears sway from their sockets. Frame 0 is the still. Feet stay down.
  walk   : the two short feet alternate. No extra limb.
  attack : 0-4 mouth wedge plus rings, 5-10 one-piece in-plane 360 (no |cos|),
           11-12 impact, 13-15 the still.
  hurt   : one-piece lean, at most 38 degrees, snout up and back. No belly flip.

Palette-locked to this still. front/back png and gif are not written.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0427_buneary/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0427_buneary/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 176, 156
VERSION = "buneary_v2"
KEY = "0427_buneary"
MEGA_KEY = "0428_lopunny-mega"

# Colours measured on this still. Not a mega Lopunny mask.
INK = (16, 16, 16)
CREAM = (238, 222, 164)
TAN = (197, 164, 123)
GRAY = (131, 106, 90)
DARK = (82, 57, 57)
WHITE = (255, 255, 255)
PINK = (197, 90, 123)
MAROON = (123, 57, 65)
LPINK = (255, 139, 164)
PUFF = (CREAM, TAN, GRAY)
MOUTH_COLS = (PINK, MAROON, LPINK)
OUTLINE = (INK, DARK)

DRAW = ("torso", "foot_l", "foot_r", "mouth", "ear_roll", "ear_up")
# Fixed dot count so roar and impact stay inside the opaque-ratio band.
RING = ((10, -3), (11, -1), (12, 0), (11, 1), (10, 3),
        (15, -4), (16, -2), (17, 0), (16, 2), (15, 4))
WEDGE = ((3, 0, LPINK), (4, -1, LPINK), (4, 1, MAROON))
BURST = ((8, -4), (10, -2), (12, 0), (10, 2), (8, 4),
         (14, -5), (16, -2), (18, 0), (16, 2), (14, 5))


def flood(seeds, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow, dtype=bool)
    q = deque()
    ys, xs = np.nonzero(seeds & allow)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if not seen[y, x]:
            seen[y, x] = True
            q.append((int(y), int(x)))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and allow[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def components(mask):
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    out = []
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
        xs2 = [p[1] for p in pts]
        ys2 = [p[0] for p in pts]
        out.append((len(pts), min(xs2), min(ys2), max(xs2), max(ys2)))
    return out


def colored(img, cols):
    rgb = img[..., :3]
    a = img[..., 3] >= 128
    hit = np.zeros(a.shape, dtype=bool)
    for col in cols:
        hit |= np.all(rgb == np.array(col, np.uint8), axis=-1)
    return a & hit


def segment(img):
    """Cut this still only: one upright ear, one rolled puff, two short feet, one mouth."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    if int(a.sum()) != 885:
        raise SystemExit(f"still opaque changed: {int(a.sum())}")
    yy, xx = np.mgrid[0:h, 0:w]
    ys, xs = np.nonzero(a)
    if int(xs.min()) != 32 or int(xs.max()) != 63 or int(ys.min()) != 19 or int(ys.max()) != 75:
        raise SystemExit("still bbox changed; remeasure before forging")
    ground = components(a & (yy >= 74))
    if len(ground) != 2:
        raise SystemExit(f"expected 2 feet on the ground row, got {ground}")

    ear_up = flood(a & (yy <= 21) & (xx <= 50), a & (yy <= 41) & (xx <= 50))
    puff = colored(img, PUFF)
    ear_roll = flood(
        puff & (xx >= 53) & (yy >= 42) & (yy <= 52),
        puff & (xx >= 52) & (yy >= 42) & (yy <= 52),
    )
    outline = colored(img, OUTLINE)
    ear_roll = ear_roll | (outline & dilate(ear_roll, 1) & (xx >= 51) & (yy >= 41) & (yy <= 54))
    mouth = colored(img, MOUTH_COLS) & (yy >= 50) & (yy <= 56) & (xx <= 46)
    foot_l = a & (yy >= 72) & (xx <= 50)
    foot_r = a & (yy >= 72) & (xx >= 53)
    parts = {
        "ear_up": ear_up, "ear_roll": ear_roll, "mouth": mouth,
        "foot_l": foot_l, "foot_r": foot_r,
    }
    used = np.zeros_like(a)
    for name, m in parts.items():
        if np.any(used & m):
            raise SystemExit(f"overlap at {name}")
        used |= m
    parts["torso"] = a & ~used
    if int(used.sum() + parts["torso"].sum()) != int(a.sum()):
        raise SystemExit("masks do not cover the still")

    eu = components(ear_up)
    er = components(ear_roll)
    if len(eu) != 1 or not (200 <= eu[0][0] <= 280) or eu[0][2] > 21 or eu[0][3] > 50:
        raise SystemExit(f"upright ear is not the tall left ear: {eu}")
    if len(er) != 1 or not (70 <= er[0][0] <= 160) or er[0][4] > 56 or er[0][2] < 40:
        raise SystemExit(f"rolled ear is not the single cheek puff: {er}")
    if int(mouth.sum()) != 6:
        raise SystemExit(f"mouth pixel count changed: {int(mouth.sum())}")
    for name, limit in (("foot_l", 40), ("foot_r", 40)):
        info = components(parts[name])
        if len(info) != 1 or info[0][0] < 12 or info[0][0] > limit or (info[0][4] - info[0][2]) > 5:
            raise SystemExit(f"{name} is not a short foot: {info}")
    # Mouth must sit left of the body centre. That is the facing measure.
    mx = float(np.nonzero(mouth)[1].mean())
    if mx >= 47:
        raise SystemExit(f"snout is not left of centre: {mx}")
    return parts


def edge_pivot(mask, side):
    ys, xs = np.nonzero(mask)
    if side == "bottom":
        sel = ys >= ys.max() - 1
    elif side == "top":
        sel = ys <= ys.min() + 1
    elif side == "left":
        sel = xs <= xs.min() + 1
    else:
        sel = xs >= xs.max() - 1
    return float(xs[sel].mean()), float(ys[sel].mean())


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def require_colors(palette):
    for col in (INK, CREAM, TAN, GRAY, DARK, WHITE, PINK, MAROON, LPINK):
        if not np.any(np.all(palette == np.array(col, np.uint8), axis=1)):
            raise SystemExit(f"still is missing measured colour {col}")


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    for i in range(0, len(ty), 300):
        sl = slice(i, i + 300)
        d = (ty[sl, None] - pts[None, :, 0]) ** 2 + (tx[sl, None] - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def build_rig(img, masks, flip):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    placed = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), dtype=bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm
    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr
    # Joint colour only under the socket, so a moved ear or foot does not leave a hole.
    for part, rad in (("ear_up", 2), ("ear_roll", 2), ("foot_l", 1), ("foot_r", 1)):
        zone = placed[part] & dilate(placed["torso"], rad)
        fill = nearest_color(canvas, zone, placed["torso"])
        write = zone & (layers["torso"][..., 3] < 128)
        layers["torso"][write] = fill[write]

    pivots = {
        "ear_up": edge_pivot(placed["ear_up"], "bottom"),
        "ear_roll": edge_pivot(placed["ear_roll"], "left"),
        "foot_l": edge_pivot(placed["foot_l"], "top"),
        "foot_r": edge_pivot(placed["foot_r"], "top"),
    }
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    fys, fxs = np.nonzero(placed["foot_l"] | placed["foot_r"])
    sole = (float(fxs[fys == fys.max()].mean()), float(fys.max()))
    mys, mxs = np.nonzero(placed["mouth"])
    mouth = (float(mxs.min()), float(mys[mxs == mxs.min()].mean()))
    soles = {}
    for name in ("foot_l", "foot_r"):
        sy, sx = np.nonzero(placed[name])
        yb = int(sy.max())
        soles[name] = [(int(x), yb) for x in sx[sy == yb]]

    if flip:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}

        def mx(x):
            return CELL_W - 1 - x

        pivots = {k: (mx(x), y) for k, (x, y) in pivots.items()}
        center = (mx(center[0]), center[1])
        sole = (mx(sole[0]), sole[1])
        mouth = (mx(mouth[0]), mouth[1])
        soles = {k: [(mx(x), y) for x, y in pts] for k, pts in soles.items()}
        facing = 1.0
    else:
        facing = -1.0
    rest = compose({k: layers[k] for k in DRAW})
    if np.any(rest != canvas):
        raise SystemExit("rest pose does not match the still")
    return {
        "layers": layers, "pivots": pivots, "center": center, "sole": sole,
        "mouth": mouth, "facing": facing, "rest": canvas, "soles": soles,
        "counts": {k: int(np.count_nonzero(layers[k][..., 3] >= 128)) for k in DRAW},
    }


def shift(layer, dx, dy):
    dx_i, dy_i = int(round(dx)), int(round(dy))
    if dx_i == 0 and dy_i == 0:
        return layer.copy()
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0, sx1 = max(0, -dx_i), min(w, w - dx_i)
    sy0, sy1 = max(0, -dy_i), min(h, h - dy_i)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy_i:sy1 + dy_i, sx0 + dx_i:sx1 + dx_i] = layer[sy0:sy1, sx0:sx1]
    return out


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest in-plane rotate. 0 and 360 stay exact. No |cos| squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.05 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    scale = 4
    up = np.repeat(np.repeat(layer, scale, axis=0), scale, axis=1)
    px, py = pivot[0] * scale + 1.5, pivot[1] * scale + 1.5
    rad = math.radians(ang)
    c, s = math.cos(rad), math.sin(rad)
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = xx - px, yy - py
    src_x = np.rint(px + c * rx + s * ry).astype(np.int32)
    src_y = np.rint(py - s * rx + c * ry).astype(np.int32)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    out[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    return out[scale // 2::scale, scale // 2::scale].copy()


def despeckle(img):
    """Drop a pixel only when all four neighbours are empty. After rotation only."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img, target):
    """Fill interior holes back toward the still count. Do not grow past it."""
    out = img.copy()
    for need in (4, 3, 3):
        a = out[..., 3] >= 128
        have = int(a.sum())
        if have >= target:
            break
        p = np.pad(a, 1)
        n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        holes = (~a) & (n >= need)
        ys, xs = np.nonzero(holes)
        if len(ys) == 0:
            continue
        room = target - have
        if len(ys) > room:
            ys, xs = ys[:room], xs[:room]
        for y, x in zip(ys.tolist(), xs.tolist()):
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                ny, nx = y + dy, x + dx
                if out[ny, nx, 3] >= 128:
                    out[y, x] = out[ny, nx]
                    break
    return out


def rotate_part(layer, angle, pivot, target):
    spun = rotate_nearest(layer, angle, pivot)
    if abs(angle) < 0.05:
        return spun
    return despeckle(fill_pinholes(spun, target))


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
        out[..., 3] = 0
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = a.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    mapped = palette[dist.argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    out[~a] = 0
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def compose(layers):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in DRAW:
        paste(img, layers[name])
    return img


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero at frame 0 and at frame n, so the loop closes on the still."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        ear_up=0.0, ear_roll=0.0, lift_l=0.0, lift_r=0.0, fwd_l=0.0, fwd_r=0.0,
        jaw=0, spin=0.0, recoil=0.0, knock=0.0, rings=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        # Feet do not move. Only the two ears, from their own sockets.
        p["ear_up"] = wave(f, n, 8.0)
        p["ear_roll"] = wave(f, n, 5.0, phase=0.7)
    elif action == "walk":
        step = wave(f, n, 1.0, cycles=2.0)
        p["lift_l"] = max(0.0, step) * 2.0
        p["lift_r"] = max(0.0, -step) * 2.0
        p["fwd_l"] = max(0.0, step) * 2.0
        p["fwd_r"] = max(0.0, -step) * 2.0
        p["ear_up"] = wave(f, n, 6.0, cycles=2.0)
        p["ear_roll"] = wave(f, n, 4.0, cycles=2.0, phase=math.pi)
    elif action == "attack":
        if f <= 4:
            p["jaw"] = (1, 2, 2, 2, 1)[f]
            p["ear_up"] = (6, 10, 12, 10, 6)[f]
            p["ear_roll"] = (3, 4, 5, 4, 3)[f]
            p["rings"] = 1
        elif f <= 10:
            p["spin"] = 60.0 * (f - 4)
        elif f in (11, 12):
            p["burst"] = 1
    else:
        env = math.sin(math.pi * f / 11.0)
        p["recoil"] = 38.0 * env
        p["knock"] = 4.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 5 else 0
    return p


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def open_mouth(img, mouth, facing, drop):
    """Drop the existing snout pixels and leave a same-palette wedge. No new jaw."""
    if drop <= 0:
        paste(img, mouth)
        return img
    home = mouth
    dropped = shift(mouth, facing * 1.0, drop)
    hole = (home[..., 3] >= 128) & (dropped[..., 3] < 128) & (img[..., 3] < 128)
    img[hole, :3] = MAROON
    img[hole, 3] = 255
    paste(img, dropped)
    ys, xs = np.nonzero(dropped[..., 3] >= 128)
    if len(xs) == 0:
        return img
    tip_x = float(xs.min() if facing < 0 else xs.max())
    tip_y = float(ys[xs == (xs.min() if facing < 0 else xs.max())].mean())
    for dx, dy, col in WEDGE:
        put(img, tip_x + facing * dx, tip_y + dy, col)
    return img


def dots(out, origin, facing, table, age=0):
    ox, oy = origin
    bump = age % 2
    cols = (WHITE, LPINK, CREAM)
    for i, (dx, dy) in enumerate(table):
        put(out, ox + facing * (dx + bump), oy + dy, cols[i % 3])
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def sparks(out, facing, frame_idx):
    a = out[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return out
    x = int(xs.max() if facing < 0 else xs.min())
    y = int(ys.min()) + 8
    for i, col in enumerate((WHITE, CREAM, LPINK)):
        put(out, x - facing * (4 + i + frame_idx), y + i * 3, col)
    return out


def synth(rig, action, f, palette):
    layers, pivots = rig["layers"], rig["pivots"]
    facing = rig["facing"]
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = (p["spin"] % 360.0) > 0.5
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        if spinning:
            img = rotate_nearest(rig["rest"], p["spin"], rig["center"])
        else:
            # Left-facing (facing < 0) leans the snout up and backward.
            img = rotate_nearest(rig["rest"], -facing * p["recoil"], rig["sole"])
            img = shift(img, -facing * p["knock"], 0)
        img = despeckle(fill_pinholes(img, rig["counts"]["body"]))
    else:
        moved = {
            "torso": layers["torso"],
            "mouth": layers["mouth"],
            "foot_l": shift(layers["foot_l"], p["fwd_l"] * facing, -p["lift_l"]),
            "foot_r": shift(layers["foot_r"], p["fwd_r"] * facing, -p["lift_r"]),
            "ear_up": rotate_part(layers["ear_up"], p["ear_up"], pivots["ear_up"], rig["counts"]["ear_up"]),
            "ear_roll": rotate_part(
                layers["ear_roll"], p["ear_roll"], pivots["ear_roll"], rig["counts"]["ear_roll"]
            ),
        }
        img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        for name in ("torso", "foot_l", "foot_r", "ear_roll", "ear_up"):
            paste(img, moved[name])
        img = open_mouth(img, moved["mouth"], facing, int(p["jaw"]))
    img = snap(img, palette)
    if p["flash"]:
        img = flash_white(img)
    if p["rings"] and not spinning and not recoiling:
        img = dots(img, rig["mouth"], facing, RING, age=f)
    if p["burst"]:
        img = dots(img, rig["mouth"], facing, BURST, age=0)
    if p["sparks"] and recoiling:
        img = sparks(img, facing, f)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (SRC_DIR / "front.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def assemble(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png"))
                sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = tile
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
    Image.fromarray(master, "RGBA").save(OUT_DIR / "all_8dir_actions.png")
    return master


def contact_sheet(sheets, row):
    """Every frame of one direction, cropped and scaled so the feet and ears read."""
    probe = sheets["idle"][row * CELL_H:(row + 1) * CELL_H, :CELL_W]
    ys, xs = np.nonzero(probe[..., 3] >= 128)
    pad = 36
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad)
    scale = 2
    cw, ch = (x1 - x0) * scale, (y1 - y0) * scale
    cols = 8
    blocks = []
    for act in ACTIONS:
        band = sheets[act][row * CELL_H:(row + 1) * CELL_H]
        frames = []
        for f in range(FRAMES[act]):
            tile = band[y0:y1, f * CELL_W + x0:f * CELL_W + x1]
            frames.append(np.repeat(np.repeat(tile, scale, 0), scale, 1))
        blocks.append(frames)
    rows = sum(math.ceil(len(b) / cols) for b in blocks)
    bg = np.full((rows * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    r = 0
    for frames in blocks:
        for i, up in enumerate(frames):
            rr, cc = divmod(i, cols)
            y, x = (r + rr) * ch, cc * cw
            m = up[..., 3] >= 128
            bg[y:y + ch, x:x + cw][m] = up[m]
        r += math.ceil(len(frames) / cols)
    return Image.fromarray(bg, "RGBA")


def south_attack_ratios(sheets):
    base = int((sheets["attack"][:CELL_H, :CELL_W, 3] >= 128).sum())
    ratios = []
    counts = []
    for f in range(FRAMES["attack"]):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        n = int((tile[..., 3] >= 128).sum())
        counts.append(n)
        ratios.append(n / base if base else 0.0)
    return counts, ratios


def largest_fraction(img):
    a = img[..., 3] >= 128
    if not np.any(a):
        return 1.0
    best = 0
    total = int(a.sum())
    seen = np.zeros_like(a)
    h, w = a.shape
    ys, xs = np.nonzero(a)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        n = 0
        while stack:
            y, x = stack.pop()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        best = max(best, n)
    return best / total


def save_parts(img, masks, path):
    cols = {
        "ear_up": (255, 213, 74),
        "ear_roll": (255, 139, 164),
        "mouth": (238, 70, 70),
        "foot_l": (80, 160, 255),
        "foot_r": (80, 220, 120),
        "torso": (160, 140, 120),
    }
    part = np.zeros((*img.shape[:2], 4), np.uint8)
    for name, col in cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    big = np.repeat(np.repeat(part, 8, 0), 8, 1)
    still = np.repeat(np.repeat(img, 8, 0), 8, 1)
    gap = np.zeros((big.shape[0], 8, 4), np.uint8)
    Image.fromarray(np.concatenate([still, gap, big], axis=1), "RGBA").save(path)


def with_flock(lock_path, fn):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def object_span(text, key):
    token = f'"{key}":'
    start = text.index(token)
    i = start + len(token)
    while text[i].isspace():
        i += 1
    if text[i] != "{":
        raise SystemExit(f"{key} value is not an object")
    depth = 0
    end = i
    while end < len(text):
        ch = text[end]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, end + 1
        end += 1
    raise SystemExit(f"unclosed {key}")


def publish_dir8(sprite_h):
    entry = {
        "dex": 427, "form": "normal", "variant": "buneary", "gen": "gen4",
        "dir": "monsterSprite/gen4/0427_buneary/normal",
        "sheetDir": "agent-sprite-forge", "layout": "actions_8dir", "nf": 16,
        "cw": CELL_W, "ch": CELL_H, "w": CELL_W, "h": int(sprite_h), "fy": FOOT_Y,
        "an": [16, 16, 16, 12], "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        mega_s, mega_e = object_span(text, MEGA_KEY)
        mega_raw = text[mega_s:mega_e]
        key_at, val_end = object_span(text, KEY)
        val_start = text.index("{", key_at)
        text = text[:val_start] + blob + text[val_end:]
        if object_span(text, MEGA_KEY)[0] != mega_s and mega_raw not in text:
            raise SystemExit("0428_lopunny-mega key string changed")
        mega2_s, mega2_e = object_span(text, MEGA_KEY)
        if text[mega2_s:mega2_e] != mega_raw:
            raise SystemExit("0428_lopunny-mega key string changed")
        data = json.loads(text[len(prefix):].rstrip().rstrip(";"))
        got = data[KEY]
        if got["v"] != VERSION or got["layout"] != "actions_8dir" or got["an"] != [16, 16, 16, 12]:
            raise SystemExit("dir8 entry did not stick")
        if got["raw"] != 1 or got["sheetDir"] != "agent-sprite-forge" or got["h"] != int(sprite_h):
            raise SystemExit("dir8 entry fields mismatch")
        if json.dumps(data[MEGA_KEY], ensure_ascii=False, separators=(",", ":")) not in mega_raw:
            raise SystemExit("0428_lopunny-mega parsed value changed")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(text)
        tmp.replace(DIR8)
        return mega_raw

    return with_flock(DIR8_LOCK, edit)


def update_queue(status):
    def edit():
        fd = os.open(QUEUE, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with os.fdopen(os.dup(fd), "r+", encoding="utf-8") as handle:
                raw = handle.read()
                data = json.loads(raw)
                found = False
                for item in data["items"]:
                    if item.get("id") != KEY:
                        continue
                    item["status"] = status
                    item.pop("claim", None)
                    found = True
                if not found:
                    raise SystemExit("queue has no 0427_buneary")
                # Surgical write would be safer, but rewrite under the same lock
                # after a fresh read so only this id's status and claim change.
                before = json.loads(raw)
                for b, a in zip(before["items"], data["items"]):
                    if b.get("id") == KEY:
                        continue
                    if b != a:
                        raise SystemExit(f"queue neighbor changed: {b.get('id')}")
                text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
                after = json.loads(text)
                for b, a in zip(before["items"], after["items"]):
                    if b.get("id") == KEY:
                        if a.get("status") != status or "claim" in a:
                            raise SystemExit("queue status did not stick")
                        continue
                    if b != a:
                        raise SystemExit(f"queue neighbor changed: {b.get('id')}")
                tmp = QUEUE.with_suffix(".json.tmp")
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def qa_report(master, sheets, palette, ratios, sprite_h):
    opaque = master[..., 3] >= 128
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    off = []
    if np.any(opaque):
        cols = np.unique(master[opaque][:, :3], axis=0)
        pal = {tuple(int(c) for c in row) for row in palette}
        off = [tuple(int(c) for c in row) for row in cols if tuple(int(c) for c in row) not in pal]
    clip = 0
    for act in ACTIONS:
        for di in range(8):
            for f in range(FRAMES[act]):
                tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = tile[..., 3] >= 128
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip += 1
    spin_body = []
    for f in range(5, 11):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        spin_body.append({"f": f, "main": round(largest_fraction(tile), 4), "ratio": round(ratios[f], 4)})
    rest = sheets["idle"][:CELL_H, :CELL_W]
    hurt = sheets["hurt"][:CELL_H, 5 * CELL_W:6 * CELL_W]
    def width(tile):
        xs = np.nonzero(tile[..., 3] >= 128)[1]
        return int(xs.max() - xs.min() + 1) if len(xs) else 0
    hurt_squash = width(hurt) < int(width(rest) * 0.75)
    bad_ratio = [round(r, 4) for r in ratios if r < 0.90 or r > 1.08]
    idle0 = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(idle0[..., 3] >= 128)[0]
    h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    return {
        "alphas": alphas,
        "off_palette": len(off),
        "clip_frames": clip,
        "sprite_h": h,
        "hurt_squash": bool(hurt_squash),
        "bad_ratio": bad_ratio,
        "spin_body": spin_body,
        "h_matches": h == sprite_h,
    }


def soles_hold(img, pts):
    return all(img[y, x, 3] >= 128 for x, y in pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    require_colors(palette)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("feet 2 facing left parts", counts, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    # Whole-body spin fills back to the still count, which includes every part once.
    body_n = int((rigs["S"]["rest"][..., 3] >= 128).sum())
    for rig in rigs.values():
        rig["counts"]["body"] = body_n

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    save_parts(front, masks, work / "parts.png")

    rest = synth(rigs["W"], "idle", 0, palette)
    if np.any(rest != rigs["W"]["rest"]):
        raise SystemExit("idle frame 0 does not match the still")
    for f in range(16):
        idle = synth(rigs["W"], "idle", f, palette)
        if not soles_hold(idle, rigs["W"]["soles"]["foot_l"]):
            raise SystemExit(f"left foot left the ground on idle {f}")
        if not soles_hold(idle, rigs["W"]["soles"]["foot_r"]):
            raise SystemExit(f"right foot left the ground on idle {f}")
        if f == 0 and int(pose("hurt", f)["recoil"]) > 38:
            raise SystemExit("recoil exceeds 38")
    if max(pose("hurt", f)["recoil"] for f in range(12)) > 38.001:
        raise SystemExit("recoil exceeds 38")

    if args.samples:
        sheets = {}
        for act in ACTIONS:
            sheet = np.zeros((CELL_H, FRAMES[act] * CELL_W, 4), np.uint8)
            for f in range(FRAMES[act]):
                sheet[0:CELL_H, f * CELL_W:(f + 1) * CELL_W] = synth(rigs["S"], act, f, palette)
            sheets[act] = sheet
        contact_sheet(sheets, 0).save(work / "samples_S.png")
        counts_a, ratios = south_attack_ratios(sheets)
        print("ATTACK_S_RATIOS", flush=True)
        for f, (c, r) in enumerate(zip(counts_a, ratios)):
            tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            print(
                f"attack S {f:02d} opaque {c} ratio {r:.4f} main {largest_fraction(tile):.3f}",
                flush=True,
            )
        print(f"min {min(ratios):.4f} max {max(ratios):.4f}", flush=True)
        bad = [round(r, 4) for r in ratios if r < 0.90 or r > 1.08]
        if bad:
            raise SystemExit(f"attack opaque ratio out of range at {bad}")
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
        for direction in DIRS:
            key = f"{act}/{direction}"
            if key in state["done"]:
                continue
            od = frames_dir / act / direction
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[direction], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    write_outputs(sheets)
    preview = contact_sheet(sheets, 0)
    preview.save(OUT_DIR / "preview.png")
    (SRC_DIR / "actions").mkdir(parents=True, exist_ok=True)
    preview.save(SRC_DIR / "actions" / "preview.png")
    motion = contact_sheet(sheets, DIRS.index("W"))
    motion.save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    counts_a, ratios = south_attack_ratios(sheets)
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(counts_a, ratios)):
        print(f"attack S {f:02d} opaque {c} ratio {r:.4f}", flush=True)
    print(f"min {min(ratios):.4f} max {max(ratios):.4f} h {sprite_h}", flush=True)

    report = qa_report(sheets["attack"] if False else np.concatenate([sheets[a] for a in ACTIONS], axis=0), sheets, palette, ratios, sprite_h)
    # The concat above is the master. Reuse the written pixels via sheets only.
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            try:
                subprocess.run(
                    [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                     "--fps", "12", "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                    capture_output=True, text=True, timeout=90, check=False,
                )
            except subprocess.TimeoutExpired:
                report[f"gait_{act}"] = "timeout"
                continue
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)

    bad_spin = [s for s in report["spin_body"] if s["main"] < 0.90]
    failed = (
        report["off_palette"] or report["clip_frames"] or report["alphas"] != [0, 255]
        or report["bad_ratio"] or bad_spin or report["hurt_squash"] or not report["h_matches"]
    )
    if failed:
        if args.publish:
            update_queue("needs_fix")
        raise SystemExit(f"qa failed {report} spin={bad_spin}")

    meta = {
        "generator": "forge_buneary_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 427,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "unflipped": sorted(UNFLIPPED),
        "still_faces": "left",
        "feet": 2,
        "rolled_ears": 1,
        "mouth_pixels": counts["mouth"],
        "parts": counts,
        "palette_colors": int(len(palette)),
        "off_palette_colors": 0,
        "attack_s_ratios": [round(r, 4) for r in ratios],
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.publish and not args.no_sync:
        copied = 0
        SYNC_OUT.mkdir(parents=True, exist_ok=True)
        for fp in OUT_DIR.iterdir():
            if fp.is_file():
                shutil.copy2(fp, SYNC_OUT / fp.name)
                copied += 1
        publish_dir8(sprite_h)
        update_queue("done")
        arena_bytes = (SYNC_OUT / "sheet.png").stat().st_size
        print(f"synced {copied} arena_sheet_bytes {arena_bytes}", flush=True)
    print(f"h {sprite_h} cw {CELL_W} ch {CELL_H} fy {FOOT_Y}", flush=True)


if __name__ == "__main__":
    main()
