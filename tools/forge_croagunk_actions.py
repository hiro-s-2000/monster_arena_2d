#!/usr/bin/env python3
"""Croagunk (0453) normal 8-direction action sheet from this form's 96x96 front.png.

Measured on normal/front.png: 957 opaque pixels, bbox x29-66 y27-68 (38x42),
14 colours, alpha already 0 or 255. Small biped frog. Two feet on the ground.
Left foot x38-48 y61-66 (49 px). Right foot x54-63 y61-68 (59 px).
Arms end in red fingers: left x29-42 y49-59 (91 px), right x55-66 y51-60
(106 px). Those are not feet. Yellow cheek marks are not feet and the mouth
is closed (no transparent gap), so the roar puffs those marks plus rings.
No new mouth. Front cheek centroid is left of the body and the eye sits
behind it, so the still faces left. Unflipped dirs are N NW W SW.
S SE E NE are mirrored. E faces right.

  idle   : cheeks and arms sway. Frame 0 is the still. Feet stay planted.
  walk   : the two feet alternate. Arms swing as arms.
  attack : 0-4 cheek puff plus rings, 5-10 one-piece in-plane 360,
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
SRC_DIR = ROOT / "monsterSprite/gen4/0453_croagunk/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0453_croagunk/normal/agent-sprite-forge")
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
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "croagunk_v2"
KEY = "0453_croagunk"

INK = (16, 16, 16)
NAVY = (49, 49, 65)
INDIGO = (74, 74, 139)
GRAY = (82, 82, 82)
BLUE = (106, 115, 213)
SLATE = (123, 123, 139)
SKY = (131, 156, 255)
DRED = (156, 57, 57)
GOLD = (180, 139, 24)
SILVER = (180, 180, 197)
RED = (230, 82, 90)
PINK = (255, 156, 164)
YELLOW = (255, 213, 24)
WHITE = (255, 255, 255)
PALETTE_COLS = (
    INK, NAVY, INDIGO, GRAY, BLUE, SLATE, SKY, DRED, GOLD, SILVER, RED, PINK, YELLOW, WHITE,
)
YELLOW_COLS = (YELLOW, GOLD)
RED_COLS = (DRED, RED, PINK)

DRAW = ("torso", "foot_l", "foot_r", "arm_l", "arm_r", "cheek_l", "cheek_r")
RING = ((4, -3), (6, -1), (6, 1), (8, -2), (9, 0), (8, 2), (11, -1), (11, 1))
BURST = ((5, -4), (7, -2), (8, 0), (7, 2), (5, 4), (11, -2), (13, 0), (11, 2))


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
    """Two feet, two arms, two yellow cheek marks. The mouth stays closed."""
    h, w = img.shape[:2]
    a = img[..., 3] == 255
    if int(a.sum()) != 957:
        raise SystemExit(f"still opaque changed: {int(a.sum())}")
    if img.shape[0] != 96 or img.shape[1] != 96:
        raise SystemExit("front is not 96x96")
    yy, xx = np.mgrid[0:h, 0:w]
    ys, xs = np.nonzero(a)
    if int(xs.min()) != 29 or int(xs.max()) != 66 or int(ys.min()) != 27 or int(ys.max()) != 68:
        raise SystemExit("still bbox changed; remeasure before forging")
    ground = components(a & (yy >= 61))
    if len(ground) != 2:
        raise SystemExit(f"expected 2 feet on the ground row, got {ground}")

    yellow = colored(img, YELLOW_COLS)
    red = colored(img, RED_COLS)
    foot_l = a & (yy >= 61) & (xx <= 50)
    foot_r = a & (yy >= 61) & (xx >= 52)
    arm_l = flood(red & (xx < 45), a & (xx <= 43) & (yy >= 49) & (yy <= 59))
    arm_r = flood(red & (xx > 55), a & (xx >= 55) & (yy >= 51) & (yy <= 60))
    cheek_l = yellow & (xx < 42)
    cheek_r = yellow & (xx >= 42)
    parts = {
        "foot_l": foot_l, "foot_r": foot_r,
        "arm_l": arm_l, "arm_r": arm_r,
        "cheek_l": cheek_l, "cheek_r": cheek_r,
    }
    expect = {
        "foot_l": 49, "foot_r": 59, "arm_l": 91, "arm_r": 106,
        "cheek_l": 5, "cheek_r": 11,
    }
    expect_comps = {
        "foot_l": 1, "foot_r": 1, "arm_l": 1, "arm_r": 1,
        "cheek_l": 2, "cheek_r": 1,
    }
    used = np.zeros_like(a)
    for name, m in parts.items():
        info = components(m)
        if len(info) != expect_comps[name] or int(m.sum()) != expect[name]:
            raise SystemExit(f"{name} mask changed: n={int(m.sum())} comps={info}")
        if np.any(used & m):
            raise SystemExit(f"overlap at {name}")
        used |= m
    parts["torso"] = a & ~used
    if int(parts["torso"].sum()) != 636 or int(used.sum()) + 636 != 957:
        raise SystemExit(f"torso coverage changed: {int(parts['torso'].sum())}")
    if components(parts["torso"]) and len(components(parts["torso"])) != 1:
        raise SystemExit("torso is not one piece")
    # The head has no transparent mouth. Yellow marks sit on opaque skin.
    head = a & (yy <= 45) & (xx >= 29) & (xx <= 66)
    span_holes = 0
    for y in range(27, 46):
        row = np.nonzero(a[y])[0]
        if len(row) == 0:
            continue
        span_holes += int((~a[y, int(row.min()):int(row.max()) + 1]).sum())
    if span_holes > 4:
        raise SystemExit(f"head gained an opening: {span_holes}")
    front_x = float(np.nonzero(cheek_l)[1].mean())
    eye = red & (yy < 48)
    eye_x = float(np.nonzero(eye)[1].mean())
    body_x = float(xs.mean())
    if not (front_x < 40.0 and front_x < body_x and eye_x > front_x):
        raise SystemExit(f"facing changed: cheek {front_x} eye {eye_x} body {body_x}")
    if int(eye.sum()) != 29:
        raise SystemExit(f"eye changed: {int(eye.sum())}")
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
    if len(palette) != 14:
        raise SystemExit(f"expected 14 colours, got {len(palette)}")
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
    for part, rad in (
        ("arm_l", 1), ("arm_r", 1), ("foot_l", 1), ("foot_r", 1),
        ("cheek_l", 2), ("cheek_r", 2),
    ):
        zone = placed[part] & dilate(placed["torso"], rad)
        fill = nearest_color(canvas, zone, placed["torso"])
        write = zone & (layers["torso"][..., 3] == 0)
        layers["torso"][write] = fill[write]

    pivots = {
        "arm_l": edge_pivot(placed["arm_l"], "right"),
        "arm_r": edge_pivot(placed["arm_r"], "left"),
        "foot_l": edge_pivot(placed["foot_l"], "top"),
        "foot_r": edge_pivot(placed["foot_r"], "top"),
        "cheek_l": edge_pivot(placed["cheek_l"], "right"),
        "cheek_r": edge_pivot(placed["cheek_r"], "left"),
    }
    oys, oxs = np.nonzero(canvas[..., 3] == 255)
    center = (float(oxs.mean()), float(oys.mean()))
    fys, fxs = np.nonzero(placed["foot_l"] | placed["foot_r"])
    sole = (float(fxs[fys >= fys.max() - 1].mean()), float(fys.max()))
    cys, cxs = np.nonzero(placed["cheek_l"])
    cheek = (float(cxs.min()), float(cys[cxs == cxs.min()].mean()))
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
        cheek = (mx(cheek[0]), cheek[1])
        soles = {k: [(mx(x), y) for x, y in pts] for k, pts in soles.items()}
        facing = 1.0
    else:
        facing = -1.0
    rest = compose({k: layers[k] for k in DRAW})
    if np.any(rest != canvas):
        raise SystemExit("rest pose does not match the still")
    return {
        "layers": layers, "pivots": pivots, "center": center, "sole": sole,
        "cheek": cheek, "facing": facing, "rest": canvas, "soles": soles,
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
    h, w = out.shape[:2]
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
                if 0 <= ny < h and 0 <= nx < w and out[ny, nx, 3] == 255:
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


def paste_empty(dst, layer):
    m = (layer[..., 3] == 255) & (dst[..., 3] == 0)
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
        arm_l=0.0, arm_r=0.0, lift_l=0.0, lift_r=0.0, fwd_l=0.0, fwd_r=0.0,
        puff=0, spin=0.0, recoil=0.0, knock=0.0, lunge=0,
        rings=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["arm_l"] = wave(f, n, 6.0)
        p["arm_r"] = wave(f, n, 5.0, phase=0.7)
        p["puff"] = wave(f, n, 1.15)
    elif action == "walk":
        step = wave(f, n, 1.0, cycles=2.0)
        p["lift_l"] = max(0.0, step) * 3.0
        p["lift_r"] = max(0.0, -step) * 3.0
        p["fwd_l"] = max(0.0, step) * 2.0
        p["fwd_r"] = max(0.0, -step) * 2.0
        p["arm_l"] = wave(f, n, 12.0, cycles=2.0, phase=math.pi)
        p["arm_r"] = wave(f, n, 12.0, cycles=2.0)
        p["puff"] = wave(f, n, 0.8, cycles=2.0)
    elif action == "attack":
        if f <= 4:
            p["puff"] = (2, 3, 3, 3, 2)[f]
            p["arm_l"] = (8, 12, 14, 10, 6)[f]
            p["arm_r"] = (-8, -12, -14, -10, -6)[f]
            p["rings"] = 1
        elif f <= 10:
            p["spin"] = 60.0 * (f - 4)
        elif f in (11, 12):
            p["burst"] = 1
            p["lunge"] = 2
            p["puff"] = 1
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


def edge_tip(layer, direction):
    ys, xs = np.nonzero(layer[..., 3] == 255)
    if len(xs) == 0:
        return None
    if direction < 0:
        x = int(xs.min())
    else:
        x = int(xs.max())
    y = float(ys[xs == x].mean())
    return float(x), y


def puff_cheeks(img, layers, facing, amount):
    """Move the existing yellow marks. Do not draw a mouth."""
    n = int(round(amount))
    if n <= 0:
        paste(img, layers["cheek_l"])
        paste(img, layers["cheek_r"])
        return None
    # Front sac (cheek_l) goes forward and a little up. Inner mark only rises.
    front = shift(layers["cheek_l"], facing * n, -max(1, n // 2))
    inner = shift(layers["cheek_r"], 0, -n)
    paste(img, front)
    paste(img, inner)
    if n >= 2:
        paste_empty(img, shift(layers["cheek_l"], facing * (n + 1), -max(1, n // 2)))
        paste_empty(img, shift(layers["cheek_r"], 0, -(n + 1)))
    return front


def dots(out, origin, facing, table, age=0):
    if origin is None:
        return out
    ox, oy = origin
    bump = age % 2
    cols = (YELLOW, WHITE, SKY)
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
    y = int(ys.min()) + 6
    for i, col in enumerate((WHITE, YELLOW, SKY)):
        put(out, x - facing * (3 + i + frame_idx), y + i * 2, col)
    return out


def synth(rig, action, f, palette):
    layers, pivots = rig["layers"], rig["pivots"]
    facing = rig["facing"]
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = (p["spin"] % 360.0) > 0.5
    recoiling = p["recoil"] > 0.4
    front = None
    if spinning or recoiling:
        if spinning:
            img = rotate_nearest(rig["rest"], p["spin"], rig["center"])
        else:
            # Left-facing head leans up and backward. Positive screen angle dropped it.
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
        }
        img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        for name in ("torso", "foot_l", "foot_r", "arm_l", "arm_r"):
            paste(img, moved[name])
        front = puff_cheeks(img, layers, facing, p["puff"])
        if p["lunge"]:
            img = shift(img, facing * p["lunge"], 0)
            if front is not None:
                front = shift(front, facing * p["lunge"], 0)
    img = snap(img, palette)
    if p["flash"]:
        img = flash_white(img)
    if p["rings"] and not spinning and not recoiling:
        tip = edge_tip(front, facing) if front is not None else rig["cheek"]
        img = dots(img, tip, facing, RING, age=f)
    if p["burst"]:
        img = dots(img, rig["cheek"], facing, BURST, age=0)
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
    pad = 28
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
        "cheek_l": (255, 213, 24),
        "cheek_r": (255, 160, 40),
        "arm_l": (255, 80, 80),
        "arm_r": (255, 180, 80),
        "foot_l": (80, 160, 255),
        "foot_r": (80, 220, 120),
        "torso": (70, 70, 90),
    }
    part = np.zeros((*img.shape[:2], 4), np.uint8)
    for name, col in cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    big = np.repeat(np.repeat(part, 6, 0), 6, 1)
    still = np.repeat(np.repeat(img, 6, 0), 6, 1)
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
        "dex": 453, "form": "normal", "variant": "croagunk", "gen": "gen4",
        "dir": "monsterSprite/gen4/0453_croagunk/normal",
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
        before = json.loads(text[len(prefix):].rstrip().rstrip(";"))
        if KEY not in before:
            raise SystemExit("dir8 has no 0453_croagunk")
        key_at, val_end = object_span(text, KEY)
        val_start = text.index("{", key_at)
        text = text[:val_start] + blob + text[val_end:]
        if not text.endswith("\n"):
            text += "\n"
        if "\n" in text.strip():
            raise SystemExit("dir8.js became multi-line")
        after = json.loads(text[len(prefix):].rstrip().rstrip(";"))
        if set(after) != set(before):
            raise SystemExit("dir8 key set changed")
        for k, old in before.items():
            if k == KEY:
                continue
            if after[k] != old:
                raise SystemExit(f"dir8 neighbor changed: {k}")
        got = after[KEY]
        if got != entry:
            raise SystemExit("dir8 entry did not stick")
        if got["v"] != VERSION or got["layout"] != "actions_8dir" or got["an"] != [16, 16, 16, 12]:
            raise SystemExit("dir8 entry did not stick")
        if got["raw"] != 1 or got["sheetDir"] != "agent-sprite-forge" or got["h"] != int(sprite_h):
            raise SystemExit("dir8 entry fields mismatch")
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
                    raise SystemExit("queue has no 0453_croagunk")
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


def soles_hold(img, rest, pts):
    for x, y in pts:
        if img[y, x, 3] != 255 or not np.array_equal(img[y, x], rest[y, x]):
            return False
    return True


def print_ratios(counts, ratios, sheets):
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(counts, ratios)):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        print(
            f"attack S {f:02d} opaque {c} ratio {r:.4f} main {largest_fraction(tile):.3f}",
            flush=True,
        )
    print(f"min {min(ratios):.4f} max {max(ratios):.4f}", flush=True)


def check_row(sheets, label):
    counts_a, ratios = south_attack_ratios(sheets)
    if label == "S":
        print_ratios(counts_a, ratios, sheets)
    bad = [round(r, 4) for r in ratios if r < 0.90 or r > 1.08]
    if bad:
        raise SystemExit(f"{label} attack opaque ratio out of range at {bad}")
    weak = [
        f for f in range(5, 11)
        if largest_fraction(sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]) < 0.90
    ]
    if weak:
        raise SystemExit(f"{label} spin shattered at frames {weak}")
    return counts_a, ratios


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
    if body_n != 957:
        raise SystemExit(f"placed still count changed: {body_n}")
    for rig in rigs.values():
        rig["counts"]["body"] = body_n

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    save_parts(front, masks, work / "parts.png")

    for label, rig in rigs.items():
        rest = synth(rig, "idle", 0, palette)
        if np.any(rest != rig["rest"]):
            raise SystemExit(f"{label} idle frame 0 does not match the still")
        for f in range(16):
            idle = synth(rig, "idle", f, palette)
            if not soles_hold(idle, rig["rest"], rig["soles"]["foot_l"]):
                raise SystemExit(f"{label} left foot left the ground on idle {f}")
            if not soles_hold(idle, rig["rest"], rig["soles"]["foot_r"]):
                raise SystemExit(f"{label} right foot left the ground on idle {f}")
    if max(pose("hurt", f)["recoil"] for f in range(12)) > 38.001:
        raise SystemExit("recoil exceeds 38")

    if args.samples:
        for label in ("S", "W"):
            sheets = {act: one_row(rigs[label], palette, act) for act in ACTIONS}
            contact_sheet(sheets, 0).save(work / f"samples_{label}.png")
            check_row(sheets, label)
        print(f"cell {CELL_W}x{CELL_H} fy {FOOT_Y}", flush=True)
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
    print(f"feet 2 cell {CELL_W}x{CELL_H} h {sprite_h}", flush=True)

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
        "generator": "forge_croagunk_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 453,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "unflipped": sorted(UNFLIPPED),
        "still_faces": "left",
        "feet": 2,
        "mouth": "closed",
        "cheek_pixels": counts["cheek_l"] + counts["cheek_r"],
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
    print(f"h {sprite_h} cw {CELL_W} ch {CELL_H} fy {FOOT_Y} feet 2", flush=True)


if __name__ == "__main__":
    main()
