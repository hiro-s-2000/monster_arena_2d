#!/usr/bin/env python3
"""Mega Abomasnow (0460_abomasnow-mega) 8-direction action sheet.

Measured on this form's front.png (96x96, opaque 4009, bbox x7-88 y11-84).
Frontal. The highest pixels are the tree on the right shoulder, not a profile
head. The left ice and the right ice club are the side branches. The back tree
is not a limb. Torso and thighs are one fur mass. Only the outer feet split
around y74-84. The short center tuft stays on the skirt.

  Facing: front. Unflipped S SE E NE keep the tree on the viewer's right.
  N NW W SW are mirrored. E faces screen-right (rings go right).

  idle   : breath, tree rustle, arms sway. Feet planted. Frame 0 is the still.
  walk   : in-place stomp. Feet lift a few pixels, no stride. Skirt stays one mass.
  attack : 0-4 small mouth opening (mouth_open) plus rings and the arms. 5-10 one-piece 360 spin.
           11-12 impact. 13-15 recover.
  hurt   : whole-body recoil, 38 degrees at most. No belly mirror, no squash.

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.

  --force     rebuild frames
  --samples   parts.png and samples.png, then stop
  --no-sync   do not copy to pokemon_arena_2d
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
import mouth_open  # noqa: E402  (small face-following mouth opening, 2026-11-04)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen4/0460_abomasnow/mega/abomasnow-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0460_abomasnow/mega/abomasnow-mega/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
PY = FORGE / ".venv/bin/python"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still is the front view. Screen-right rows keep it. The other four are the mirror.
UNFLIPPED = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "abomasnow_mega_v2m"

WHITE = (254, 254, 254)
ICE = (153, 205, 255)
GREEN = (82, 164, 148)
PINK = (222, 164, 246)
BROWN = (176, 88, 82)
INK = (32, 41, 49)
PALE = (197, 213, 222)

# Closed mouth line on front.png (still coords), left corner -> right corner.
# The face is frontal, so the small lens opens straight down below the existing
# mouth (ink corners at x37 / x43, tongue pink in between; line kept 1 px inside). Jaw never moves.
MOUTH_LINE = [(38, 59), (40, 59), (42, 59)]
# Opening amount per frame (px). 1 = growl, 2 = roar peak. Replaces the old
# downward wedge (attached_mouth) and the gen4_attack_strike jaw=3 push.
MOUTH_ATTACK = [1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]


def flood(seeds, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow)
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


def segment(img):
    """Partition this still. The tree stays one mass. Feet are only the outer split."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = img[..., :3]
    ice = a & (
        np.all(rgb == ICE, axis=2) | np.all(rgb == (107, 161, 202), axis=2)
    )
    pink = a & np.all(rgb == PINK, axis=2)
    brown = a & np.all(rgb == BROWN, axis=2)

    seed = np.zeros_like(a)
    seed[11, 72] = True
    allow_t = a & (((yy <= 42) & (xx >= 50)) | ((yy <= 22) & (xx >= 66)))
    tree = flood(seed, allow_t)

    seed = np.zeros_like(a)
    seed[12, 16] = True
    arm_l = flood(seed, a & (yy <= 24) & (xx <= 36) & ~tree)

    seed = np.zeros_like(a)
    iys, ixs = np.nonzero(ice & (yy >= 48) & (xx >= 74))
    if len(iys) == 0:
        raise SystemExit("right ice branch is missing")
    seed[int(iys[0]), int(ixs[0])] = True
    arm_r = flood(seed, a & (yy >= 46) & (yy <= 56) & (xx >= 72) & ~tree & ~arm_l)

    foot_l = a & (yy >= 74) & (xx <= 28) & ~tree & ~arm_l & ~arm_r
    foot_r = a & (yy >= 74) & (xx >= 54) & ~tree & ~arm_l & ~arm_r
    mouth = (brown | (pink & (yy >= 55))) & ~tree & ~arm_l & ~arm_r & ~foot_l & ~foot_r
    torso = a & ~(tree | arm_l | arm_r | foot_l | foot_r)

    parts = {
        "tree": tree, "arm_l": arm_l, "arm_r": arm_r,
        "foot_l": foot_l, "foot_r": foot_r, "torso": torso,
    }
    own = None
    for m in parts.values():
        own = m.copy() if own is None else (own | m)
    overlap = 0
    names = list(parts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap += int((parts[names[i]] & parts[names[j]]).sum())
    if int(own.sum()) != int(a.sum()) or overlap:
        raise SystemExit(
            f"part masks do not partition the still ({int(own.sum())} vs {int(a.sum())}, overlap {overlap})"
        )
    if int(foot_l.sum()) < 40 or int(foot_r.sum()) < 40:
        raise SystemExit("feet collapsed")
    parts["mouth"] = mouth
    return parts


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
    """Drop only pixels whose 4-neighbours are all empty. After rotation only."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


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


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def attached_mouth(img, tip, amount):
    """Short downward wedge on the existing mouth. No new jaw."""
    if amount <= 0:
        return img
    tx, ty = tip
    for i in range(1, amount + 1):
        half = 1 if i < amount else 2
        for dx in range(-half, half + 1):
            if abs(dx) == half:
                col = INK
            elif i == amount and dx == 0:
                col = BROWN
            else:
                col = PINK
            put(img, tx + dx, ty + i, col)
    return img


def roar_rings(out, mouth, facing, age, strength):
    mx, my = mouth
    cols = (WHITE, ICE, GREEN)
    for r_i in range(strength):
        rad = 6 + age * 3 + r_i * 5
        col = cols[min(r_i, 2)]
        for deg in range(-50, 51, 8):
            ang = math.radians(deg)
            put(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.72, col)
    return out


def speed_lines(out, facing, n, seed, length):
    ys, xs = np.nonzero(out[..., 3] >= 128)
    if len(ys) == 0:
        return out
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    rng = np.random.default_rng(seed)
    rear_x = (x0 + (x1 - x0) // 5) if facing > 0 else (x1 - (x1 - x0) // 5)
    y_lo = y0 + (y1 - y0) // 3
    y_hi = y0 + (y1 - y0) * 2 // 3
    for _k in range(n):
        y = int(rng.integers(y_lo, max(y_lo + 1, y_hi)))
        off = int(rng.integers(4, 9))
        length_i = int(rng.integers(max(3, length // 2), length + 1))
        for i in range(length_i):
            put(out, rear_x - facing * (off + i), y, WHITE if i < 2 else PALE)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(size // 2, size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), WHITE if r < size - 2 else GREEN)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, WHITE)
    return out


def sparks(out, frame_idx, facing):
    ys, xs = np.nonzero(out[..., 3] >= 128)
    if len(ys) == 0:
        return out
    x0, y0, x1 = int(xs.min()), int(ys.min()), int(xs.max())
    d = frame_idx
    cx = (x0 + x1) // 2
    pts = [
        (cx - facing * (6 + d * 2), y0 + 10),
        (cx + facing * (5 + d), y0 + 6),
        (cx - facing * 2, y0 + 18),
    ]
    for sx, sy in pts:
        put(out, sx, sy, WHITE)
        put(out, sx - facing, sy + 1, ICE)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    rgb = out[..., :3].astype(np.int16)
    ink = a & (rgb[..., 0] < 50) & (rgb[..., 1] < 60) & (rgb[..., 2] < 70)
    out[a & ~ink, :3] = WHITE
    return out


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
        if name == "mouth":
            continue
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr

    body_names = ("torso", "tree", "arm_l", "arm_r", "foot_l", "foot_r")
    cushions = (
        ("tree", "torso", 3),
        ("arm_l", "torso", 3),
        ("arm_r", "torso", 3),
        ("foot_l", "torso", 2),
        ("foot_r", "torso", 2),
    )
    for part_name, under, rad in cushions:
        zone = placed[part_name] & dilate(placed[under], rad)
        if not np.any(zone):
            continue
        fill = nearest_color(canvas, zone, placed[under])
        empty = layers[under][..., 3] < 128
        write = zone & empty
        layers[under][write] = fill[write]

    opaque = np.zeros((CELL_H, CELL_W), bool)
    for name in body_names:
        opaque |= placed[name]
    oys, oxs = np.nonzero(opaque)
    center = (float(oxs.mean()), float(oys.mean()))
    pivots = {
        "tree": socket_pivot(placed["tree"], placed["torso"]),
        "arm_l": socket_pivot(placed["arm_l"], placed["torso"]),
        "arm_r": socket_pivot(placed["arm_r"], placed["torso"]),
        "foot_l": socket_pivot(placed["foot_l"], placed["torso"]),
        "foot_r": socket_pivot(placed["foot_r"], placed["torso"]),
    }
    mys, mxs = np.nonzero(placed["mouth"])
    if len(mys) == 0:
        raise SystemExit("mouth pixels missing")
    mouth = (float(np.median(mxs)), float(np.median(mys)))
    tip = (float(np.median(mxs[mys >= mys.max() - 1])), float(mys.max()))
    mline = mouth_open.transform_points(MOUTH_LINE, dx=dst_x - x0, dy=dst_y - y0)

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
        mouth = (CELL_W - 1 - mouth[0], mouth[1])
        tip = (CELL_W - 1 - tip[0], tip[1])
        mline = mouth_open.transform_points(mline, mirror_w=CELL_W)
    facing = 1.0 if not flip else -1.0
    return layers, pivots, center, mouth, tip, facing, mline


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, tree=0.0, arm_l=0.0, arm_r=0.0,
        foot_l=0.0, foot_r=0.0, jaw=0, spin=0.0, recoil=0.0,
        ring=-1, rings=0, lines=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["dy"] = wave(f, n, 1.0)
        p["tree"] = wave(f, n, 3.5, phase=0.4)
        p["arm_l"] = wave(f, n, 4.0, phase=0.7)
        p["arm_r"] = wave(f, n, -4.0, phase=1.1)
    elif action == "walk":
        p["dy"] = wave(f, n, 1.0, cycles=2.0)
        p["tree"] = wave(f, n, 4.0, phase=0.5, cycles=2.0)
        p["arm_l"] = wave(f, n, 8.0, cycles=2.0)
        p["arm_r"] = wave(f, n, -8.0, phase=math.pi, cycles=2.0)
        # Up only. A downward shift would open a gap and read as a stride.
        p["foot_l"] = min(0.0, wave(f, n, 3.0, cycles=2.0))
        p["foot_r"] = min(0.0, wave(f, n, 3.0, phase=math.pi, cycles=2.0))
        p["lines"] = 2
    elif action == "attack":
        # 0-4 roar, 5-10 in-place spin, 11-12 impact, 13-15 recover.
        arm_l = [6, 10, 12, 12, 8, 0, 0, 0, 0, 0, 0, 8, 5, 3, 1, 0]
        arm_r = [-8, -16, -24, -22, -12, 0, 0, 0, 0, 0, 0, -14, -8, -3, 0, 0]
        tree = [2, 4, 5, 5, 3, 0, 0, 0, 0, 0, 0, 3, 2, 1, 0, 0]
        jaw = [1, 2, 3, 3, 2, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0]
        p["arm_l"] = arm_l[f]
        p["arm_r"] = arm_r[f]
        p["tree"] = tree[f]
        p["jaw"] = jaw[f]
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
            p["lines"] = 3
        if 0 <= f <= 4:
            p["ring"], p["rings"] = f, 3
        if f in (11, 12):
            p["burst"] = 1
            p["dx"] = 2.0
    else:
        env = [0, 0.45, 0.85, 1.0, 0.92, 0.7, 0.48, 0.3, 0.16, 0.08, 0.02, 0][f]
        p["dx"] = -6.0 * env
        p["dy"] = -1.0 * env
        p["recoil"] = 38.0 * env
        p["jaw"] = 2 if 2 <= f <= 5 else 0
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def synth(rig, action, f, palette, effects=True):
    layers, pivots, center, mouth, tip, facing, mline = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)
    # Small face-following mouth instead of the jaw wedge (ignore the shared jaw push).
    if action == "attack":
        p["jaw"] = MOUTH_ATTACK[f]
    elif action == "hurt":
        p["jaw"] = 1 if p["jaw"] else 0

    fwd = facing
    bdx, bdy = p["dx"] * fwd, p["dy"]

    def moved(name, ang, extra_dy=0.0):
        return shift(rotate_nearest(layers[name], ang * fwd, pivots[name]), bdx, bdy + extra_dy)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    tree = moved("tree", p["tree"])
    arm_l = moved("arm_l", p["arm_l"])
    arm_r = moved("arm_r", p["arm_r"])
    # Feet translate up only. The root shape stays. No stride, no thin legs.
    foot_l = shift(layers["foot_l"], bdx, bdy + p["foot_l"])
    foot_r = shift(layers["foot_r"], bdx, bdy + p["foot_r"])
    torso = shift(layers["torso"], bdx, bdy)

    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for lyr in (torso, tree, foot_l, foot_r, arm_l, arm_r):
        paste(body, lyr)

    pivot = (center[0] + bdx, center[1] + bdy)
    rotated = False
    if spinning:
        body = rotate_nearest(body, p["spin"] * fwd, pivot)
        rotated = True
    elif recoiling:
        body = rotate_nearest(body, -p["recoil"] * fwd, pivot)
        rotated = True
    elif any(abs(p[k]) >= 0.45 for k in ("tree", "arm_l", "arm_r")):
        rotated = True
    if rotated:
        body = despeckle(body)

    mx, my = mouth[0] + bdx, mouth[1] + bdy
    tx, ty = tip[0] + bdx, tip[1] + bdy
    if recoiling:
        ang = -p["recoil"] * fwd
        mx, my = content_point(mx, my, ang, pivot)
        tx, ty = content_point(tx, ty, ang, pivot)
    anchor = (int(round(mx)), int(round(my)))
    tip_xy = (int(round(tx)), int(round(ty)))

    img = snap(body, palette)
    if not effects:
        return img
    if p["jaw"] and not spinning:
        line = mouth_open.transform_points(mline, dx=bdx, dy=bdy)
        if recoiling:
            line = [content_point(x, y, -p["recoil"] * fwd, pivot) for x, y in line]
        img = mouth_open.open_mouth(img, line, amount=int(p["jaw"]), deep=INK, inner=BROWN, peak=0.5)
    if p["flash"]:
        img = flash_white(img)
    if p["rings"] and p["ring"] >= 0 and not spinning:
        img = roar_rings(img, anchor, fwd, p["ring"], p["rings"])
    if p["lines"]:
        img = speed_lines(img, fwd, p["lines"], seed=f * 17 + (0 if action == "walk" else 5), length=8)
    if p["burst"]:
        img = impact_burst(img, (anchor[0] + fwd * 6, anchor[1] + 2), 9 if f >= 12 else 7)
    if p["sparks"]:
        img = sparks(img, f, fwd)
    return snap(img, palette)


def largest_component(img):
    a = img[..., 3] >= 128
    seen = np.zeros_like(a)
    best = 0
    second = 0
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
            second = best
            best = n
        elif n > second:
            second = n
    total = int(a.sum())
    return best, second, total, n_comp


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
        Image.fromarray(sh, "RGBA").save(OUT_DIR / f"{act}.png")
    master = np.concatenate([sheets[a] for a in ACTIONS], axis=0)
    Image.fromarray(master, "RGBA").save(OUT_DIR / "sheet.png")
    return master


def preview_image(sheets):
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (2, 6, 10, 14),
        "attack": (2, 7, 11, 14),
        "hurt": (1, 3, 6, 11),
    }
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
    pad = 4
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
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
    act = OUT_DIR / "actions" / "preview.png"
    if act.exists():
        dest = SYNC_OUT / "actions"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(act, dest / "preview.png")
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
    counts = {k: int(v.sum()) for k, v in masks.items() if k != "mouth"}
    counts["mouth"] = int(masks["mouth"].sum())
    print("parts", counts, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}

    rest = synth(rigs["S"], "idle", 0, palette)
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
        "torso": (180, 180, 190, 255),
        "tree": (40, 170, 70, 255),
        "arm_l": (80, 160, 255, 255),
        "arm_r": (255, 120, 40, 255),
        "foot_l": (230, 60, 80, 255),
        "foot_r": (240, 210, 40, 255),
        "mouth": (255, 0, 200, 255),
    }
    for name, col in part_cols.items():
        part[masks[name]] = col
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    for label, act, f, effects in (
        ("spin", "attack", 5, False),
        ("spin", "attack", 6, False),
        ("spin", "attack", 7, False),
        ("spin", "attack", 8, False),
        ("spin", "attack", 9, False),
        ("hurt", "hurt", 3, False),
    ):
        tile = synth(rigs["S"], act, f, palette, effects=effects)
        best, second, total, n_comp = largest_component(tile)
        frac = best / total if total else 1
        print(f"{label} S {f} largest {best}/{total} second {second} comps {n_comp} frac {frac:.4f}", flush=True)
        if frac < 0.97 or second > 80:
            raise SystemExit(f"{label} frame {f} shattered (frac {frac:.3f}, second {second})")

    if args.samples:
        want = [("idle", "S", f) for f in (0, 4, 8, 12)]
        want += [("walk", "S", f) for f in (0, 2, 4, 6, 8, 10, 14)]
        want += [("attack", "S", f) for f in (0, 2, 4, 5, 6, 7, 8, 10, 11, 12, 15)]
        want += [("attack", "E", f) for f in (2, 7)]
        want += [("attack", "W", f) for f in (2, 7)]
        want += [("hurt", "S", f) for f in (0, 1, 3, 6, 11)]
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
                if fp.exists() and fp.stat().st_size > 0 and not args.force:
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
    actions = OUT_DIR / "actions"
    actions.mkdir(parents=True, exist_ok=True)
    preview.save(actions / "preview.png")
    contact = motion_contact(sheets)
    contact.save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 74

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = set(int(v) for v in np.unique(master[..., 3]))
    clip = {}
    spin_bad = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and 5 <= f <= 9:
                    best, second, total_px, _n = largest_component(tile)
                    # Rings are absent on spin. Speed lines are the extra specks.
                    if total_px and (best / total_px < 0.95 or second > 120):
                        spin_bad.append(f"{d}:{f}:{second}")
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": sorted(alphas),
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "spin_bad": spin_bad,
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run(
                [str(PY), "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:900], flush=True)

    meta = {
        "generator": "forge_abomasnow_mega_actions", "version": VERSION,
        "key": "0460_abomasnow-mega", "dex": 460,
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "facing": "front",
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped_kaiju_stomper",
        "limbs": "right-shoulder tree, two ice branches, two root feet, skirt is torso",
        "source": "front.png 96x96", "gifs_untouched": True,
        "layout": "actions_8dir", "sheetDir": "agent-sprite-forge",
        "an": [16, 16, 16, 12], "raw": 1,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (work / "sprite_h.txt").write_text(str(sprite_h))

    if off or alphas - {0, 255} or clip or spin_bad:
        raise SystemExit(f"qa failed off={len(off)} alphas={alphas} clip={clip} spin_bad={spin_bad}")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
