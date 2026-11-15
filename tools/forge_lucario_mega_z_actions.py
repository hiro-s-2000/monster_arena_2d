#!/usr/bin/env python3
"""Mega Lucario Z (0448_lucario-mega-z) 8-direction action sheet.

Left-facing biped. The snout is left of the eye. One arm reaches back to the
right, the other hangs forward on the left. Two shins: the near foot is the
only one on the ground, the far shin stops higher. No wings and no wide stance.

  idle   : breath, a small arm and shin sway, crest tilt. Frame 0 is the still.
  walk   : short in-place march. Arms swing. Shins do not split into a stride.
  attack : 0-4 roar (small slit along the mouth line + rings), 5-10 one-piece 360 spin,
           11-12 impact on the forward paw, 13-15 recover.
  hurt   : knock-back, recoil at most 38 degrees, no belly flip. 12 frames.

Every opaque pixel is snapped to the still's palette. Alpha is 0 or 255.
front/back png and gif are not written.

  --force     rebuild frames
  --samples   write parts.png and samples.png, then stop
  --no-sync   do not copy the sheet to pokemon_arena_2d
  --skip-qa   skip gait_loop.py
"""

import argparse
import hashlib
import json
import math
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
import mouth_open

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen4/0448_lucario/mega/lucario-mega-z"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0448_lucario/mega/lucario-mega-z/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# The still faces left. Screen-right directions are the mirror.
FACE_RIGHT = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "lucario_mega_z_v2m"

INK = (16, 16, 16)
WHITE = (255, 255, 255)
YELLOW = (234, 234, 136)
CYAN = (61, 202, 221)
PALE = (205, 217, 221)
EYE = (74, 156, 239)

# Closed mouth line on the still (front.png coords), hinge -> lip tip. It runs
# along the underside of the pale upper muzzle, in front of the cyan tendril,
# so the opening is a small slit inside the snout. The jaw never moves.
MOUTH_LINE_STILL = [(34, 34), (32, 34), (30, 34)]
# Attack mouth after the shared strike hook: small growl, 2 px at the roar peak.
ATTACK_MOUTH = [1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def leftmost_run(row):
    xs = np.nonzero(row)[0]
    if len(xs) == 0:
        return None
    end = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x > end + 1:
            break
        end = x
    return int(xs[0]), end


def segment(img):
    """Parts of this left-facing still. Masks partition every opaque pixel."""
    a = img[..., 3] >= 128
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    fore = np.zeros_like(a)
    for y in range(42, 67):
        run = leftmost_run(a[y])
        if run is None:
            continue
        start, end = run
        if end <= 32 and start <= 32:
            fore[y, start:end + 1] = True
        elif y <= 53 and start <= 32:
            cut = min(end, 31)
            fore[y, start:cut + 1] = True
        elif y >= 54 and start <= 22:
            cut = min(end, 24)
            fore[y, start:cut + 1] = True
    fore &= a

    # Long rear arm, from the shoulder gap out to the right. Tips with a 1px
    # break stay on the arm. The torso's right edge stays put.
    rear = a & (yy >= 42) & (yy <= 57) & (xx >= 54)

    # Near shin plus the only foot that reaches the ground. Far shin stops at y74.
    near = a & (yy >= 66) & (xx >= 31) & (xx <= 41) & ~fore
    far = a & (yy >= 66) & (yy <= 74) & (xx >= 43) & (xx <= 53) & ~fore & ~near

    crest = a & (yy <= 22) & ~fore & ~rear & ~near & ~far
    head = a & (yy >= 23) & (yy <= 41) & ~crest & ~fore & ~rear & ~near & ~far
    torso = a & ~(crest | head | fore | rear | near | far)
    parts = {
        "crest": crest, "head": head, "fore": fore, "rear": rear,
        "near": near, "far": far, "torso": torso,
    }
    own = None
    overlap = 0
    names = list(parts)
    for name in names:
        own = parts[name].copy() if own is None else (own | parts[name])
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap += int((parts[names[i]] & parts[names[j]]).sum())
    if int(own.sum()) != int(a.sum()) or overlap:
        raise SystemExit(
            f"part masks do not partition the still ({int(own.sum())} vs {int(a.sum())}, overlap {overlap})"
        )
    return parts


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


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
    """4x nearest rotation. Copies palette pixels. Does not invent colours."""
    ang = angle_deg % 360.0
    if min(ang, 360.0 - ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    up = np.repeat(np.repeat(layer, 4, axis=0), 4, axis=1)
    px, py = pivot[0] * 4 + 1.5, pivot[1] * 4 + 1.5
    rad = math.radians(angle_deg)
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
    return out[1::4, 1::4].copy()


def content_point(x, y, angle_deg, pivot):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    vx, vy = x - pivot[0], y - pivot[1]
    return pivot[0] + c * vx - s * vy, pivot[1] + s * vx + c * vy


def despeckle(img):
    """Drop only pixels whose 4-neighbours are all empty."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    step = 400
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None] - pts[None, :, 0]) ** 2 + (tx[sl, None] - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
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
    return out


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def socket_pivot(part, body):
    """Contact pixel of the part nearest the body centre."""
    contact = part & dilate(body, 2)
    if not np.any(contact):
        contact = part
    by, bx = np.nonzero(body if np.any(body) else contact)
    bcx, bcy = float(bx.mean()), float(by.mean())
    ys, xs = np.nonzero(contact)
    d = (xs.astype(np.int32) - int(bcx)) ** 2 + (ys.astype(np.int32) - int(bcy)) ** 2
    i = int(d.argmin())
    return float(xs[i]), float(ys[i])


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
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm

    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr

    # Socket colour under a limb, only where that limb already covered the body.
    for part_name, under, rad in (
        ("fore", "torso", 4),
        ("rear", "torso", 4),
        ("near", "torso", 3),
        ("far", "torso", 3),
        ("crest", "head", 3),
    ):
        zone = placed[part_name] & dilate(placed[under], rad)
        if np.any(zone):
            fill = nearest_color(canvas, zone, placed[under])
            empty = layers[under][..., 3] < 128
            write = zone & empty
            layers[under][write] = fill[write]

    pivots = {
        "fore": socket_pivot(placed["fore"], placed["torso"]),
        "rear": socket_pivot(placed["rear"], placed["torso"]),
        "near": socket_pivot(placed["near"], placed["torso"]),
        "far": socket_pivot(placed["far"], placed["torso"] | placed["head"]),
        "crest": socket_pivot(placed["crest"], placed["head"]),
    }
    opaque = np.zeros((CELL_H, CELL_W), bool)
    for pm in placed.values():
        opaque |= pm
    oys, oxs = np.nonzero(opaque)
    center = (float(oxs.mean()), float(oys.mean()))

    # Snout tip: leftmost pixel of the muzzle, which sits left of the eye.
    head = placed["head"]
    hy, hx = np.nonzero(head)
    eye = head & np.all(canvas[..., :3] == EYE, axis=2)
    ey, ex = np.nonzero(eye)
    eye_x = float(ex.mean()) if len(ex) else float(np.median(hx))
    snout = head & (np.arange(CELL_H)[:, None] >= int(np.median(hy)))
    sy, sx = np.nonzero(snout)
    if len(sx) == 0:
        sy, sx = hy, hx
    tip = int(sx.min())
    mouth = (float(tip), float(np.median(sy[sx <= tip + 1])))
    if not (mouth[0] < eye_x):
        raise SystemExit(f"expected a left-facing snout, mouth x={mouth[0]} eye x={eye_x}")

    mouth_line = [(x - x0 + dst_x, y - y0 + dst_y) for x, y in MOUTH_LINE_STILL]
    if flip:
        mouth_line = mouth_open.transform_points(mouth_line, mirror_w=CELL_W)
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
        mouth = (CELL_W - 1 - mouth[0], mouth[1])
    facing = 1.0 if flip else -1.0
    return layers, pivots, center, mouth, facing, mouth_line


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, fore=0.0, rear=0.0, near=0.0, far=0.0, crest=0.0,
        jaw=0, spin=0.0, recoil=0.0, ring=-1, rings=0, lines=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["dy"] = wave(f, n, 1.4)
        p["fore"] = wave(f, n, 5.0, phase=0.4)
        p["rear"] = wave(f, n, -4.0, phase=0.4)
        p["near"] = wave(f, n, 3.0, phase=1.1)
        p["far"] = wave(f, n, -3.0, phase=1.1)
        p["crest"] = wave(f, n, 4.0, phase=0.2)
    elif action == "walk":
        p["dy"] = wave(f, n, 1.6, cycles=2.0)
        p["dx"] = wave(f, n, 1.0, cycles=2.0)
        p["fore"] = wave(f, n, 16.0, cycles=2.0)
        p["rear"] = wave(f, n, -13.0, cycles=2.0)
        p["near"] = wave(f, n, 11.0, cycles=2.0)
        p["far"] = wave(f, n, -9.0, cycles=2.0)
        p["crest"] = wave(f, n, 3.0, cycles=2.0)
        p["lines"] = 2
    elif action == "attack":
        # 0-4 roar, 5-10 in-place spin, 11-12 impact, 13-15 recover.
        crest_t = [4, 10, 14, 14, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        jaw_t = [1, 2, 3, 3, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        fore_t = [-6, -12, -14, -10, -4, 0, 0, 0, 0, 0, 0, 16, 12, 5, 2, 0]
        rear_t = [5, 9, 10, 6, 2, 0, 0, 0, 0, 0, 0, -8, -4, 0, 0, 0]
        p["crest"] = crest_t[f]
        p["jaw"] = jaw_t[f]
        p["fore"] = fore_t[f]
        p["rear"] = rear_t[f]
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
            p["lines"] = 4
            p["fore"] = p["rear"] = p["crest"] = 0.0
            p["jaw"] = 0
        if 0 <= f <= 4:
            p["ring"], p["rings"] = f, 2
        if f in (11, 12):
            p["burst"] = 1
    else:
        env = [0, 0.4, 0.75, 1.0, 0.88, 0.62, 0.4, 0.22, 0.1, 0.03, 0.0, 0][f]
        p["dx"] = -5.0 * env
        p["dy"] = -1.0 * env
        p["recoil"] = 38.0 * env
        p["jaw"] = 1 if 2 <= f <= 5 else 0
        p["crest"] = 5.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 5 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def limb_tip(layer, facing):
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    if len(xs) == 0:
        return None
    if facing > 0:
        x = int(xs.max())
        y = int(np.median(ys[xs >= x - 1]))
    else:
        x = int(xs.min())
        y = int(np.median(ys[xs <= x + 1]))
    return x, y


def roar_rings(out, mouth, facing, age, strength):
    mx, my = mouth
    cols = (WHITE, YELLOW, CYAN)
    for r_i in range(strength):
        rad = 6 + age * 3 + r_i * 5
        col = cols[min(r_i, 2)]
        for deg in range(-50, 51, 8):
            ang = math.radians(deg)
            put(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.72, col)
    return out


def speed_lines(out, facing, n, seed, length):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    rear_x = (x0 + (x1 - x0) // 6) if facing > 0 else (x1 - (x1 - x0) // 6)
    y_lo = y0 + (y1 - y0) // 3
    for _k in range(n):
        y = int(rng.integers(y_lo, max(y_lo + 1, y1 - 4)))
        off = int(rng.integers(2, 6))
        leng = int(rng.integers(max(3, length // 2), length + 1))
        for i in range(leng):
            put(out, rear_x - facing * (off + i), y, WHITE if i < 2 else PALE)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(size // 2, size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), WHITE if r < size - 2 else YELLOW)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, WHITE)
    return out


def sparks(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    d = frame_idx
    cx = (x0 + x1) // 2
    pts = [
        (cx - facing * (6 + d), y0 + 4 - d),
        (cx + facing * (4 + d), y0 + 8),
        (cx - facing * 2, y0 - d),
    ]
    for sx, sy in pts:
        put(out, sx, sy, WHITE)
        put(out, sx - facing, sy, YELLOW)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def synth(rig, action, f, palette):
    layers, pivots, center, mouth, facing, mouth_line = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)
    if action == "attack":
        # The shared hook forces jaw 3 on frames 5-14; this rig opens small.
        p["jaw"] = ATTACK_MOUTH[f]

    fwd = facing
    bdx, bdy = p["dx"] * fwd, p["dy"]
    sgn = -1.0 if facing > 0 else 1.0

    def moved(name, ang):
        return shift(rotate_nearest(layers[name], ang * sgn, pivots[name]), bdx, bdy)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    pivot = (center[0] + bdx, center[1] + bdy)
    fore = moved("fore", p["fore"])
    rear = moved("rear", p["rear"])
    near = moved("near", p["near"])
    far = moved("far", p["far"])
    crest = moved("crest", p["crest"])
    torso = shift(layers["torso"], bdx, bdy)
    head = shift(layers["head"], bdx, bdy)

    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    # Torso and head first, so the socket fill stays hidden under the limbs.
    for lyr in (torso, head, far, rear, near, fore, crest):
        paste(body, lyr)
    part_moved = any(abs(p[k]) >= 0.45 for k in ("fore", "rear", "near", "far", "crest"))
    if part_moved and not spinning and not recoiling:
        body = despeckle(body)

    if spinning:
        body = rotate_nearest(body, p["spin"], pivot)
        body = despeckle(body)
    elif recoiling:
        # Lean back, away from the facing. Stays at or under 38 degrees.
        body = rotate_nearest(body, -p["recoil"] * fwd, pivot)
        body = despeckle(body)

    mx, my = mouth[0] + bdx, mouth[1] + bdy
    if recoiling:
        mx, my = content_point(mx, my, -p["recoil"] * fwd, pivot)
    anchor = (int(round(mx)), int(round(my)))

    img = snap(body, palette)
    if p["jaw"] and not spinning:
        line = mouth_open.transform_points(mouth_line, dx=bdx, dy=bdy)
        if recoiling:
            line = mouth_open.transform_points(line, angle=-p["recoil"] * fwd, pivot=pivot)
        img = mouth_open.open_mouth(img, line, amount=min(2, int(p["jaw"])), deep=INK, inner=None, peak=0.6)
    if p["flash"]:
        img = flash_white(img)
    if p["rings"] and p["ring"] >= 0 and not spinning:
        img = roar_rings(img, anchor, fwd, p["ring"], p["rings"])
    if p["lines"]:
        img = speed_lines(img, fwd, p["lines"], seed=f * 17 + (1 if action == "walk" else 9), length=9)
    if p["burst"]:
        tip = limb_tip(fore, fwd)
        if tip is None:
            tip = (anchor[0] + int(6 * fwd), anchor[1] + 8)
        img = impact_burst(img, tip, 9 if f >= 12 else 7)
    if p["sparks"]:
        img = sparks(img, f, fwd)
    return snap(img, palette)


def largest_component(img):
    a = img[..., 3] >= 128
    seen = np.zeros_like(a)
    best = 0
    n_comp = 0
    h, w = a.shape
    ys, xs = np.nonzero(a)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        n_comp += 1
        q = deque([(y, x)])
        seen[y, x] = True
        n = 0
        while q:
            cy, cx = q.popleft()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        if n > best:
            best = n
    return best, int(a.sum()), n_comp


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (MEGA_DIR / "front.png").read_bytes()
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


def preview_image(sheets):
    samples = {"idle": (0, 4, 8, 12), "walk": (0, 4, 8, 12), "attack": (2, 7, 11, 14), "hurt": (1, 3, 6, 10)}
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            x = si * CELL_W
            y = ai * CELL_H
            m = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 6
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][2 * CELL_H:3 * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            a = tile[..., 3] >= 128
            if not np.any(a):
                continue
            ys, xs = np.nonzero(a)
            y0, y1 = max(0, int(ys.min()) - pad), min(CELL_H, int(ys.max()) + 1 + pad)
            x0, x1 = max(0, int(xs.min()) - pad), min(CELL_W, int(xs.max()) + 1 + pad)
            row.append(tile[y0:y1, x0:x1])
        crops.append(row)
    cell_h = max(c.shape[0] for row in crops for c in row)
    cell_w = max(c.shape[1] for row in crops for c in row)
    cols = max(len(row) for row in crops)
    bg = np.full((len(crops) * cell_h, cols * cell_w, 4), (15, 20, 31, 255), np.uint8)
    for ri, row in enumerate(crops):
        for ci, c in enumerate(row):
            y = ri * cell_h + (cell_h - c.shape[0]) // 2
            x = ci * cell_w + (cell_w - c.shape[1]) // 2
            m = c[..., 3] >= 128
            bg[y:y + c.shape[0], x:x + c.shape[1]][m] = c[m]
    return Image.fromarray(bg, "RGBA")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(MEGA_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), flush=True)
    for name, m in masks.items():
        print(f"  {name} {bbox(m)}", flush=True)

    rigs = {}
    for d in DIRS:
        rigs[d] = build_rig(front, masks, flip=d in FACE_RIGHT)

    rest = synth(rigs["W"], "idle", 0, palette)
    a = front[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    dst_x = (CELL_W - (x1 - x0)) // 2
    dst_y = FOOT_Y - (y1 - y0)
    placed = np.zeros_like(rest)
    placed[dst_y:dst_y + (y1 - y0), dst_x:dst_x + (x1 - x0)] = front[y0:y1, x0:x1]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (180, 180, 190), "head": (240, 200, 40), "crest": (255, 80, 220),
        "fore": (40, 180, 80), "rear": (220, 60, 60), "near": (40, 90, 220), "far": (255, 140, 40),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    spin_bad = False
    for f in (5, 6, 7, 8, 9, 10):
        tile = synth(rigs["W"], "attack", f, palette)
        best, total, n_comp = largest_component(tile)
        print(f"spin W {f} largest {best}/{total} comps {n_comp}", flush=True)
        if best < int(total * 0.92):
            spin_bad = True
    if spin_bad:
        raise SystemExit("spin shattered the body")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 4, 8, 12)]
        want += [("attack", "W", f) for f in (0, 2, 4, 5, 7, 10, 11, 15)]
        want += [("attack", "E", f) for f in (2, 7)]
        want += [("hurt", "W", f) for f in (0, 1, 3, 6, 11)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
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
                if fp.exists() and fp.stat().st_size > 0:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions_dir = MEGA_DIR / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    preview.save(actions_dir / "preview.png")
    contact = motion_contact(sheets)
    contact.save(work / "review_E.png")
    contact.save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    partial = int(np.sum((master[..., 3] > 0) & (master[..., 3] < 255)))
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{d}:{f}")
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "partial_alpha": partial,
        "edge_touching": {k: len(v) for k, v in clip.items()},
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:800], flush=True)

    meta = {
        "generator": "forge_lucario_mega_z_actions", "version": VERSION, "key": "0448_lucario-mega-z",
        "dex": 448, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS, "face_right": sorted(FACE_RIGHT),
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "partial_alpha": partial, "source": "front.png 96x96", "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
