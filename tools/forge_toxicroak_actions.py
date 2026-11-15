#!/usr/bin/env python3
"""Toxicroak (0454) normal 8-direction action sheet from this form's front.png.

Measured on normal/front.png: 1684 opaque pixels, bbox x20-75 y19-76 (56x58),
15 colours, alpha already 0 or 255. Adult biped frog. Two feet:
left sole x30-46 y68-74, right sole x55-67 y68-76.
The hanging fists end at y67 (left x20-31, right x62-75) and are not feet.
The red throat sac is x28-44 y41-53, pink mouth x32-34 y44-46, and is not a foot.
Yellow eyes x35-39 y30-33 sit left of the body centroid x45.1. The still faces left.
Unflipped dirs are N NW W SW. S SE E NE are mirrored. E faces right.

  idle   : crest, throat, and fists sway from their sockets. Frame 0 is the still.
  walk   : the two soles alternate. Fists swing as arms. Throat is not a foot.
  attack : 0-4 the existing throat plus rings, 5-10 one-piece in-plane 360,
           11-12 impact, 13-15 the still.
  hurt   : one-piece lean, at most 38 degrees. No |cos| squash.

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
SRC_DIR = ROOT / "monsterSprite/gen4/0454_toxicroak/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0454_toxicroak/normal/agent-sprite-forge")
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
VERSION = "toxicroak_v2"
KEY = "0454_toxicroak"
PROTECTED = (
    "0450_hippowdon",
    "0451_skorupi",
    "0452_drapion",
    "0453_croagunk",
    "0455_carnivine",
    "0456_finneon",
)

INK = (16, 16, 16)
BLUE = (106, 115, 213)
NAVY = (74, 74, 139)
DARK = (49, 49, 65)
RED = (230, 82, 90)
DRED = (156, 57, 57)
LBLUE = (131, 156, 255)
WHITE = (255, 255, 255)
INDIGO = (57, 57, 82)
GRAY = (180, 180, 197)
MGRAY = (123, 123, 139)
PINK = (255, 156, 164)
SGRAY = (82, 82, 82)
YELLOW = (255, 213, 24)
GOLD = (180, 139, 24)
PALETTE_COLS = (
    INK, BLUE, NAVY, DARK, RED, DRED, LBLUE, WHITE, INDIGO, GRAY, MGRAY, PINK, SGRAY, YELLOW, GOLD,
)

DRAW = ("torso", "foot_l", "foot_r", "arm_l", "arm_r", "throat", "crest")
RING = ((8, -3), (10, -1), (10, 1), (8, 3), (13, -2), (15, 0), (13, 2), (18, -1), (18, 1))
BURST = ((7, -4), (9, -2), (11, 0), (9, 2), (7, 4), (14, -3), (16, 0), (14, 3), (20, -1), (20, 1))


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
    a = img[..., 3] == 255
    hit = np.zeros(a.shape, dtype=bool)
    for col in cols:
        hit |= np.all(rgb == np.array(col, np.uint8), axis=-1)
    return a & hit


def segment(img):
    """Two soles, two fists, one throat, one crest. Arms and throat are not feet."""
    h, w = img.shape[:2]
    a = img[..., 3] == 255
    if int(a.sum()) != 1684:
        raise SystemExit(f"still opaque changed: {int(a.sum())}")
    if img.shape[0] != 96 or img.shape[1] != 96:
        raise SystemExit("front is not 96x96")
    yy, xx = np.mgrid[0:h, 0:w]
    ys, xs = np.nonzero(a)
    if int(xs.min()) != 20 or int(xs.max()) != 75 or int(ys.min()) != 19 or int(ys.max()) != 76:
        raise SystemExit("still bbox changed; remeasure before forging")
    ground = components(a & (yy >= 73))
    if len(ground) != 2:
        raise SystemExit(f"expected 2 feet on the ground row, got {ground}")

    red = colored(img, (RED, DRED, PINK))
    ink = colored(img, (INK,))
    pink = colored(img, (PINK,))
    crest = flood(a & (yy <= 21), a & (yy <= 28) & (xx >= 24) & (xx <= 48))
    throat = flood(pink, (red | ink) & (xx >= 26) & (xx <= 45) & (yy >= 40) & (yy <= 53))
    arm_l = flood(a & (xx <= 25) & (yy >= 62) & (yy <= 67), a & (xx <= 34) & (yy >= 54) & (yy <= 67))
    arm_r = flood(a & (xx >= 69) & (yy >= 63) & (yy <= 67), a & (xx >= 62) & (yy >= 54) & (yy <= 67))
    foot_l = a & (yy >= 68) & (xx <= 52)
    foot_r = a & (yy >= 68) & (xx >= 53)
    parts = {
        "crest": crest, "throat": throat, "arm_l": arm_l, "arm_r": arm_r,
        "foot_l": foot_l, "foot_r": foot_r,
    }
    expect = {
        "crest": 114, "throat": 171, "arm_l": 124, "arm_r": 142,
        "foot_l": 82, "foot_r": 89,
    }
    used = np.zeros_like(a)
    for name, m in parts.items():
        info = components(m)
        if len(info) != 1 or int(m.sum()) != expect[name]:
            raise SystemExit(f"{name} mask changed: n={int(m.sum())} comps={info}")
        if np.any(used & m):
            raise SystemExit(f"overlap at {name}")
        used |= m
    parts["torso"] = a & ~used
    if int(parts["torso"].sum()) != 962 or int(used.sum()) + 962 != 1684:
        raise SystemExit(f"torso coverage changed: {int(parts['torso'].sum())}")
    if int(parts["arm_l"][68:].sum()) or int(parts["arm_r"][68:].sum()):
        raise SystemExit("an arm reached the ground")
    if int(parts["throat"][54:].sum()) or int((parts["foot_l"] | parts["foot_r"])[:68].sum()):
        raise SystemExit("throat and feet share a row")
    if int((pink & parts["throat"]).sum()) != 9:
        raise SystemExit("pink mouth is not inside the throat")
    eyes = colored(img, (YELLOW, GOLD))
    if int((eyes & (crest | throat | foot_l | foot_r | arm_l | arm_r)).sum()):
        raise SystemExit("eyes leaked into a moving part")
    throat_x = float(np.nonzero(throat)[1].mean())
    eye_x = float(np.nonzero(eyes)[1].mean())
    body_x = float(xs.mean())
    if not (throat_x < body_x - 4 and eye_x < body_x):
        raise SystemExit(f"still is not facing left: throat {throat_x} eye {eye_x} body {body_x}")
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
    a = img[..., 3] == 255
    return np.unique(img[a][:, :3], axis=0)


def require_colors(palette):
    if len(palette) != 15:
        raise SystemExit(f"expected 15 colours, got {len(palette)}")
    for col in PALETTE_COLS:
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
    a = img[..., 3] == 255
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
    for part, rad in (("crest", 2), ("throat", 2), ("arm_l", 1), ("arm_r", 1), ("foot_l", 1), ("foot_r", 1)):
        zone = placed[part] & dilate(placed["torso"], rad)
        fill = nearest_color(canvas, zone, placed["torso"])
        write = zone & (layers["torso"][..., 3] == 0)
        layers["torso"][write] = fill[write]

    pivots = {
        "crest": edge_pivot(placed["crest"], "bottom"),
        "throat": edge_pivot(placed["throat"], "right"),
        "arm_l": edge_pivot(placed["arm_l"], "top"),
        "arm_r": edge_pivot(placed["arm_r"], "top"),
        "foot_l": edge_pivot(placed["foot_l"], "top"),
        "foot_r": edge_pivot(placed["foot_r"], "top"),
    }
    oys, oxs = np.nonzero(canvas[..., 3] == 255)
    center = (float(oxs.mean()), float(oys.mean()))
    fys, fxs = np.nonzero(placed["foot_l"] | placed["foot_r"])
    sole = (float(fxs[fys >= fys.max() - 1].mean()), float(fys.max()))
    mouth_m = np.all(layers["throat"][..., :3] == np.array(PINK, np.uint8), axis=-1)
    mouth_m &= layers["throat"][..., 3] == 255
    mys, mxs = np.nonzero(mouth_m)
    if len(mxs) != 9:
        raise SystemExit(f"pink mouth count changed: {len(mxs)}")
    mouth = (float(mxs.mean()), float(mys.mean()))
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
        "counts": {k: int(np.count_nonzero(layers[k][..., 3] == 255)) for k in DRAW},
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
    if abs(ang) < 0.05 or not np.any(layer[..., 3] == 255):
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
    a = out[..., 3] == 255
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img, target):
    """Fill interior holes back toward the still count. Do not grow past it."""
    out = img.copy()
    for need in (4, 3, 3):
        a = out[..., 3] == 255
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
                if out[ny, nx, 3] == 255:
                    out[y, x] = out[ny, nx]
                    break
    return out


def rotate_part(layer, angle, pivot, target):
    spun = rotate_nearest(layer, angle, pivot)
    if abs(angle) % 360.0 < 0.05:
        return spun
    return despeckle(fill_pinholes(spun, target))


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] == 255
    if not np.any(a):
        out[...] = 0
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
    m = layer[..., 3] == 255
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
        crest=0.0, throat=0.0, arm_l=0.0, arm_r=0.0,
        lift_l=0.0, lift_r=0.0, fwd_l=0.0, fwd_r=0.0,
        jaw=0, spin=0.0, recoil=0.0, knock=0.0, lunge=0,
        rings=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["crest"] = wave(f, n, 6.0)
        p["throat"] = wave(f, n, 5.0, phase=0.5)
        p["arm_l"] = wave(f, n, 4.0, phase=0.8)
        p["arm_r"] = wave(f, n, 4.0, phase=1.4)
    elif action == "walk":
        step = wave(f, n, 1.0, cycles=2.0)
        p["lift_l"] = max(0.0, step) * 4.0
        p["lift_r"] = max(0.0, -step) * 4.0
        p["fwd_l"] = max(0.0, step) * 2.0
        p["fwd_r"] = max(0.0, -step) * 2.0
        p["arm_l"] = wave(f, n, 16.0, cycles=2.0, phase=math.pi)
        p["arm_r"] = wave(f, n, 16.0, cycles=2.0)
        p["throat"] = wave(f, n, 4.0, cycles=2.0, phase=0.4)
        p["crest"] = wave(f, n, 5.0, cycles=2.0)
    elif action == "attack":
        if f <= 4:
            p["jaw"] = (1, 2, 2, 2, 1)[f]
            p["crest"] = (4, 8, 10, 6, 3)[f]
            p["rings"] = 1
        elif f <= 10:
            p["spin"] = 60.0 * (f - 4)
        elif f in (11, 12):
            p["burst"] = 1
            p["lunge"] = 2
    else:
        env = math.sin(math.pi * f / 11.0)
        p["recoil"] = 38.0 * env
        p["knock"] = 3.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def open_throat(img, throat, facing, drop):
    """Shift the existing sac forward and keep the vacated pixels. No new jaw."""
    if drop <= 0:
        paste(img, throat)
        return img
    dropped = shift(throat, facing * float(drop), 0)
    home = throat[..., 3] == 255
    cover = dropped[..., 3] == 255
    hole = home & ~cover
    img[hole, :3] = RED
    img[hole, 3] = 255
    paste(img, dropped)
    return img


def dots(out, origin, facing, table, age=0):
    ox, oy = origin
    bump = age % 2
    cols = (WHITE, RED, YELLOW)
    for i, (dx, dy) in enumerate(table):
        put(out, ox + facing * (dx + bump), oy + dy, cols[i % 3])
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] == 255
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def sparks(out, facing, frame_idx):
    a = out[..., 3] == 255
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return out
    x = int(xs.max() if facing < 0 else xs.min())
    y = int(ys.min()) + 10
    for i, col in enumerate((WHITE, YELLOW, RED)):
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
            img = rotate_nearest(rig["rest"], facing * p["recoil"], rig["sole"])
            img = shift(img, -facing * p["knock"], 0)
        ang = p["spin"] if spinning else p["recoil"]
        if ang % 360.0 > 0.5:
            img = despeckle(fill_pinholes(img, rig["counts"]["body"]))
    else:
        moved = {
            "torso": layers["torso"],
            "foot_l": shift(layers["foot_l"], p["fwd_l"] * facing, -p["lift_l"]),
            "foot_r": shift(layers["foot_r"], p["fwd_r"] * facing, -p["lift_r"]),
            "arm_l": rotate_part(layers["arm_l"], p["arm_l"], pivots["arm_l"], rig["counts"]["arm_l"]),
            "arm_r": rotate_part(layers["arm_r"], p["arm_r"], pivots["arm_r"], rig["counts"]["arm_r"]),
            "crest": rotate_part(layers["crest"], p["crest"], pivots["crest"], rig["counts"]["crest"]),
            "throat": rotate_part(layers["throat"], p["throat"], pivots["throat"], rig["counts"]["throat"]),
        }
        img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        for name in ("torso", "foot_l", "foot_r", "arm_l", "arm_r", "crest"):
            paste(img, moved[name])
        img = open_throat(img, moved["throat"], facing, int(p["jaw"]))
        if p["lunge"]:
            img = shift(img, facing * p["lunge"], 0)
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


def one_row(rig, palette, action):
    sheet = np.zeros((CELL_H, FRAMES[action] * CELL_W, 4), np.uint8)
    for f in range(FRAMES[action]):
        sheet[0:CELL_H, f * CELL_W:(f + 1) * CELL_W] = synth(rig, action, f, palette)
    return sheet


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
    probe = sheets["idle"][row * CELL_H:(row + 1) * CELL_H, :CELL_W]
    ys, xs = np.nonzero(probe[..., 3] == 255)
    pad = 40
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
            m = up[..., 3] == 255
            bg[y:y + ch, x:x + cw][m] = up[m]
        r += math.ceil(len(frames) / cols)
    return Image.fromarray(bg, "RGBA")


def south_attack_ratios(sheets):
    base_tile = sheets["attack"][:CELL_H, :CELL_W]
    base = int((base_tile[..., 3] == 255).sum())
    ratios = []
    counts = []
    for f in range(FRAMES["attack"]):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        n = int((tile[..., 3] == 255).sum())
        counts.append(n)
        ratios.append(n / base if base else 0.0)
    return counts, ratios


def largest_fraction(img):
    a = img[..., 3] == 255
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
        "crest": (255, 213, 74),
        "throat": (238, 70, 70),
        "arm_l": (255, 160, 40),
        "arm_r": (255, 210, 80),
        "foot_l": (80, 160, 255),
        "foot_r": (80, 220, 120),
        "torso": (70, 70, 90),
    }
    part = np.zeros((*img.shape[:2], 4), np.uint8)
    for name, col in cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    big = np.repeat(np.repeat(part, 4, 0), 4, 1)
    still = np.repeat(np.repeat(img, 4, 0), 4, 1)
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
    in_str = False
    esc = False
    while end < len(text):
        ch = text[end]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return start, end + 1
        end += 1
    raise SystemExit(f"unclosed {key}")


def publish_dir8(sprite_h):
    entry = {
        "dex": 454, "form": "normal", "variant": "toxicroak", "gen": "gen4",
        "dir": "monsterSprite/gen4/0454_toxicroak/normal",
        "sheetDir": "agent-sprite-forge", "layout": "actions_8dir", "nf": 16,
        "cw": CELL_W, "ch": CELL_H, "w": CELL_W, "h": int(sprite_h), "fy": FOOT_Y,
        "an": [16, 16, 16, 12], "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix) or "\n" in text.strip():
            raise SystemExit("dir8.js is not a single assignment line")
        protected = {}
        for key in PROTECTED:
            s, e = object_span(text, key)
            protected[key] = text[s:e]
        key_at, val_end = object_span(text, KEY)
        val_start = text.index("{", key_at)
        text = text[:val_start] + blob + text[val_end:]
        if not text.endswith("\n"):
            text += "\n"
        if text.count("\n") != 1:
            raise SystemExit("dir8.js picked up a line break")
        for key, raw in protected.items():
            s, e = object_span(text, key)
            if text[s:e] != raw:
                raise SystemExit(f"{key} changed while publishing toxicroak")
        data = json.loads(text[len(prefix):].rstrip().rstrip(";"))
        got = data[KEY]
        if got["v"] != VERSION or got["layout"] != "actions_8dir" or got["an"] != [16, 16, 16, 12]:
            raise SystemExit("dir8 entry did not stick")
        if got["raw"] != 1 or got["sheetDir"] != "agent-sprite-forge" or got["h"] != int(sprite_h):
            raise SystemExit("dir8 entry fields mismatch")
        for key, raw in protected.items():
            if json.dumps(data[key], ensure_ascii=False, separators=(",", ":")) not in raw:
                raise SystemExit(f"{key} parsed value changed")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(text)
        tmp.replace(DIR8)
        return True

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
                    raise SystemExit("queue has no 0454_toxicroak")
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
    opaque = master[..., 3] == 255
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
                a = tile[..., 3] == 255
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip += 1
    spin_body = []
    for f in range(5, 11):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        spin_body.append({"f": f, "main": round(largest_fraction(tile), 4), "ratio": round(ratios[f], 4)})
    rest = sheets["idle"][:CELL_H, :CELL_W]
    hurt = sheets["hurt"][:CELL_H, 5 * CELL_W:6 * CELL_W]

    def width(tile):
        xs = np.nonzero(tile[..., 3] == 255)[1]
        return int(xs.max() - xs.min() + 1) if len(xs) else 0

    hurt_squash = width(hurt) < int(width(rest) * 0.75)
    bad_ratio = [round(r, 4) for r in ratios if r < 0.90 or r > 1.08]
    idle0 = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(idle0[..., 3] == 255)[0]
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
    return all(img[y, x, 3] == 255 for x, y in pts)


def print_ratios(counts, ratios, sheets):
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(counts, ratios)):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        print(
            f"attack S {f:02d} opaque {c} ratio {r:.4f} main {largest_fraction(tile):.3f}",
            flush=True,
        )
    print(f"min {min(ratios):.4f} max {max(ratios):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if set(np.unique(front[..., 3]).tolist()) - {0, 255}:
        raise SystemExit("front alpha is not binary")
    palette = extract_palette(front)
    require_colors(palette)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("feet 2 facing left parts", counts, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    body_n = int((rigs["W"]["rest"][..., 3] == 255).sum())
    if body_n != 1684:
        raise SystemExit(f"placed still count changed: {body_n}")
    for rig in rigs.values():
        rig["counts"]["body"] = body_n

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    save_parts(front, masks, work / "parts.png")

    rest = synth(rigs["W"], "idle", 0, palette)
    if np.any(rest != rigs["W"]["rest"]):
        raise SystemExit("idle frame 0 does not match the still")
    if np.any(synth(rigs["W"], "walk", 0, palette) != rigs["W"]["rest"]):
        raise SystemExit("walk frame 0 does not match the still")
    for f in range(16):
        idle = synth(rigs["W"], "idle", f, palette)
        if not soles_hold(idle, rigs["W"]["soles"]["foot_l"]):
            raise SystemExit(f"left foot left the ground on idle {f}")
        if not soles_hold(idle, rigs["W"]["soles"]["foot_r"]):
            raise SystemExit(f"right foot left the ground on idle {f}")
        step = pose("walk", f)
        walked = synth(rigs["W"], "walk", f, palette)
        if step["lift_l"] < 0.5 and not soles_hold(walked, rigs["W"]["soles"]["foot_l"]):
            raise SystemExit(f"left sole slipped on walk {f}")
        if step["lift_r"] < 0.5 and not soles_hold(walked, rigs["W"]["soles"]["foot_r"]):
            raise SystemExit(f"right sole slipped on walk {f}")
    if max(pose("hurt", f)["recoil"] for f in range(12)) > 38.001:
        raise SystemExit("recoil exceeds 38")

    if args.samples:
        for label in ("S", "W"):
            sheets = {act: one_row(rigs[label], palette, act) for act in ACTIONS}
            contact_sheet(sheets, 0).save(work / f"samples_{label}.png")
            if label == "S":
                counts_a, ratios = south_attack_ratios(sheets)
                print_ratios(counts_a, ratios, sheets)
                bad = [round(r, 4) for r in ratios if r < 0.90 or r > 1.08]
                if bad:
                    raise SystemExit(f"attack opaque ratio out of range at {bad}")
                weak = [f for f in range(5, 11) if largest_fraction(sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]) < 0.90]
                if weak:
                    raise SystemExit(f"spin shattered at frames {weak}")
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
    nys = np.nonzero(neutral[..., 3] == 255)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    counts_a, ratios = south_attack_ratios(sheets)
    print_ratios(counts_a, ratios, sheets)
    print(f"h {sprite_h}", flush=True)

    master = np.concatenate([sheets[a] for a in ACTIONS], axis=0)
    report = qa_report(master, sheets, palette, ratios, sprite_h)
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
        "generator": "forge_toxicroak_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 454,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "unflipped": sorted(UNFLIPPED),
        "still_faces": "left",
        "feet": 2,
        "throat_pixels": counts["throat"],
        "arms_are_feet": False,
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
