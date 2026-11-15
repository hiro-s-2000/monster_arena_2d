#!/usr/bin/env python3
"""Mega Staraptor (0398_staraptor-mega) 8-direction action sheet from front.png.

Archetype: avian_raptor_hover. The still is one connected 96x96 bird.
Measured on front.png: yellow eye (255,148,41) at about x 58-63, y 25-26.
The beak tip is the pale point at x 82, y 19-22, so beak_x > eye_x and the
still faces right. The red crest sits behind the eye (x 31-61). Unflipped
directions are S SE E NE. E faces screen-right. N NW W SW are mirrored.

Parts, from this still only: red crest, head (crown, eye, beak), left wing
(image-left feathers, including the lower-left tips), right wing, two short
orange talons, a lower-right tail tip, and the torso. No extra legs or jaw.
Lower-left fragments stay on the left wing.

  idle   : hover. Wings, crest, tail, and talons sway. Frame 0 is the still.
  walk   : nose-down glide, two wingbeats, speed lines. Talons do not stride.
  attack : roar (small beak opening along the beak line + crest + rings) -> in-place 360 spin -> impact -> recover.
  hurt   : flash, backward recoil of at most 34 degrees, sparks, recover.

The spin is one rigid 4x-nearest rotation of the whole body. Hurt does not
mirror the belly and does not squash. Every opaque pixel is snapped to the
still palette. Alpha is only 0 or 255. front/back png and gif are not written.

Resumable via agent-sprite-forge/_work. --force only if that checkpoint is corrupt.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike
import mouth_open

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen4/0398_staraptor/mega/staraptor-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0398_staraptor/mega/staraptor-mega/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces right (beak tip x > eye x). Confirmed again in measure_facing().
FACE_RIGHT = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 208, 208, 180
VERSION = "staraptor_mega_v2m"

INK = (0, 0, 0)
MOUTH = (56, 46, 42)
WHITE = (248, 248, 248)
RED = (255, 96, 53)
GOLD = (255, 228, 131)
ORANGE = (255, 148, 41)
RED_DARK = (156, 65, 65)
RED_DEEP = (106, 45, 45)
CLAW_BROWN = (123, 65, 32)
CLAW_ORANGE = (189, 98, 0)
PEACH = (245, 165, 112)

# Closed beak line measured on front.png (still coords), hinge near the eye -> hooked tip.
# The opening is carved just under it, inside the white lower beak/face (2026-11-04 mouth fix).
MOUTH_LINE_STILL = [(65, 23), (71, 23), (76, 22), (79, 21)]
MOUTH_PEAK = 0.65


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def is_col(img, rgb):
    a = img[..., 3] >= 128
    return a & np.all(img[..., :3] == np.array(rgb, np.uint8), axis=-1)


def segment(img):
    """Partition every opaque pixel of this still. Cuts are from its own silhouette."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    black = is_col(img, INK)
    red = is_col(img, RED) | is_col(img, RED_DARK) | is_col(img, RED_DEEP)
    # Main crest is the red mass behind the eye. The fleck near x 70 is a shoulder mark.
    crest = red & (xx <= 64) & (yy <= 32)
    eye = is_col(img, ORANGE) & (yy <= 32) & (xx >= 54) & (xx <= 66)
    eye = eye | (is_col(img, (255, 255, 255)) & (yy <= 32) & (xx >= 50))
    for _ in range(2):
        grow = dilate(crest, 1) & black & (xx <= 66) & (yy <= 34) & ~dilate(eye, 2)
        crest = crest | grow

    # Skull, crown, face, and beak. Shoulder red stays off the head.
    head = a & (yy <= 36) & (xx >= 30) & ~crest & ~red

    claw = (
        is_col(img, CLAW_BROWN) | is_col(img, CLAW_ORANGE) | is_col(img, ORANGE) | is_col(img, PEACH)
    )
    claw = claw & (yy >= 60) & (xx >= 50) & (xx <= 74) & ~head & ~crest
    talon_lo = claw & (yy >= 70)
    talon_hi = claw & (yy < 70)

    def with_outline(mask):
        near = dilate(mask, 1) & black & (yy >= 58) & (yy <= 82) & (xx >= 48) & (xx <= 76)
        return mask | near

    talon_lo = with_outline(talon_lo)
    talon_hi = with_outline(talon_hi) & ~talon_lo

    # Hanging tip under the right wing, not a third leg.
    tail = a & (yy >= 88) & (xx >= 74)
    wing_l = a & (yy >= 38) & (xx <= 44) & ~tail & ~talon_lo & ~talon_hi & ~head & ~crest
    wing_r = a & (yy >= 40) & (xx >= 78) & ~tail & ~talon_lo & ~talon_hi & ~head & ~crest
    used = crest | head | wing_l | wing_r | talon_lo | talon_hi | tail
    torso = a & ~used
    parts = {
        "crest": crest, "head": head, "wing_l": wing_l, "wing_r": wing_r,
        "talon_lo": talon_lo, "talon_hi": talon_hi, "tail": tail, "torso": torso,
    }
    own = None
    for m in parts.values():
        own = m.copy() if own is None else (own | m)
    if int(own.sum()) != int(a.sum()) or any(int((parts[i] & parts[j]).sum()) for i in parts for j in parts if i < j):
        raise SystemExit("part masks do not partition the still")
    return parts


def measure_facing(img):
    a = img[..., 3] >= 128
    eye = is_col(img, ORANGE) & a
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    eye = eye & (yy <= 32) & (xx >= 54) & (xx <= 66)
    ey, ex = np.nonzero(eye)
    if len(ex) == 0:
        raise SystemExit("eye pixels not found")
    eye_x = float(np.median(ex))
    eye_y = float(np.median(ey))
    band = a & (np.abs(yy - eye_y) <= 8) & (yy <= 32)
    by, bx = np.nonzero(band)
    tip_x = int(bx.max())
    # Beak is the forward point. Crest red is behind the eye, so the far x is the beak.
    faces_right = tip_x > eye_x
    return faces_right, eye_x, eye_y, tip_x


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def edge_pivot(mask, side):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0, 0.0
    if side == "right":
        sel = xs >= xs.max() - 2
    elif side == "left":
        sel = xs <= xs.min() + 2
    else:
        sel = ys <= ys.min() + 2
    return float(xs[sel].mean()), float(ys[sel].mean())


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
    if abs(angle_deg) < 0.45 or not np.any(layer[..., 3] >= 128):
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


def attached_mouth(img, facing, amount):
    """Wedge grown from the existing beak tip. No new jaw shape."""
    if amount <= 0:
        return img
    a = img[..., 3] >= 128
    rgb = img[..., :3].astype(np.int16)
    h, w = img.shape[:2]
    upper = np.arange(h)[:, None] < int(h * 0.62)
    eye = upper & a & (rgb[..., 0] > 240) & (rgb[..., 1] > 120) & (rgb[..., 1] < 190) & (rgb[..., 2] < 80)
    ey, ex = np.nonzero(eye)
    if len(ey) == 0:
        return img
    eye_y, eye_x = int(np.median(ey)), int(np.median(ex))
    yy = np.arange(h)[:, None]
    band = a & upper & (np.abs(yy - eye_y) <= 8)
    if facing > 0:
        band &= np.arange(w)[None, :] >= eye_x - 2
    else:
        band &= np.arange(w)[None, :] <= eye_x + 2
    by, bx = np.nonzero(band)
    if len(bx) == 0:
        return img
    if facing > 0:
        tip_x = int(bx.max())
        tip_y = int(np.median(by[bx >= tip_x - 1]))
        step = 1
    else:
        tip_x = int(bx.min())
        tip_y = int(np.median(by[bx <= tip_x + 1]))
        step = -1
    for i in range(amount + 1):
        for dy in (-1, 0, 1):
            if i == 0 and dy != 0:
                continue
            col = INK if abs(dy) == 1 or i == amount else MOUTH
            put(img, tip_x + step * i, tip_y + dy + (1 if i else 0), col)
    return img


def roar_rings(out, mouth, facing, age, strength):
    mx, my = mouth
    cols = (WHITE, RED, GOLD)
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
    rear_x = (x0 + (x1 - x0) // 5) if facing > 0 else (x1 - (x1 - x0) // 5)
    y_lo = y0 + (y1 - y0) // 3
    y_hi = y0 + (y1 - y0) * 2 // 3
    for _k in range(n):
        y = int(rng.integers(y_lo, max(y_lo + 1, y_hi)))
        off = int(rng.integers(3, 8))
        L = int(rng.integers(max(3, length // 2), length + 1))
        for i in range(L):
            put(out, rear_x - facing * (off + i), y, WHITE if i < 2 else (176, 176, 176))
    return out


def impact_burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(size // 2, size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), WHITE if r < size - 2 else GOLD)
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
        (cx - facing * (6 + d * 2), y0 + 8 - d),
        (cx + facing * (4 + d), y0 + 4),
        (cx - facing * 2, y0 + 14),
    ]
    for sx, sy in pts:
        put(out, sx, sy, WHITE)
        put(out, sx - facing, sy + 1, GOLD)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def largest_fraction(img):
    a = img[..., 3] >= 128
    if not np.any(a):
        return 1.0, 0
    seen = np.zeros_like(a)
    h, w = a.shape
    best = 0
    total = int(a.sum())
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
        if n > best:
            best = n
    return best / total, total


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

    # Joint colour lives under the part, so a swing does not open a hole.
    # At rest the part is drawn later and covers it.
    cushions = (
        ("wing_l", "torso", 5),
        ("wing_r", "torso", 5),
        ("tail", "torso", 4),
        ("talon_lo", "torso", 3),
        ("talon_hi", "torso", 3),
        ("crest", "head", 4),
    )
    for part_name, under, rad in cushions:
        zone = placed[part_name] & dilate(placed[under], rad)
        if not np.any(zone):
            continue
        fill = nearest_color(canvas, zone, placed[under])
        empty = layers[under][..., 3] < 128
        write = zone & empty
        layers[under][write] = fill[write]

    pivots = {
        "wing_l": edge_pivot(placed["wing_l"], "right"),
        "wing_r": edge_pivot(placed["wing_r"], "left"),
        "crest": edge_pivot(placed["crest"], "right"),
        "tail": edge_pivot(placed["tail"], "top"),
        "talon_lo": edge_pivot(placed["talon_lo"], "top"),
        "talon_hi": edge_pivot(placed["talon_hi"], "top"),
    }
    hys, hxs = np.nonzero(placed["head"])
    neck = (float(np.median(hxs[hys >= hys.max() - 2])), float(hys.max()))
    opaque = placed["torso"] | placed["head"] | placed["wing_l"] | placed["wing_r"]
    oys, oxs = np.nonzero(opaque | placed["crest"] | placed["tail"] | placed["talon_lo"] | placed["talon_hi"])
    center = (float(oxs.mean()), float(oys.mean()))

    mouth = [(x - x0 + dst_x, y - y0 + dst_y) for x, y in MOUTH_LINE_STILL]
    if flip:
        mouth = mouth_open.transform_points(mouth, mirror_w=CELL_W)
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        neck = (CELL_W - 1 - neck[0], neck[1])
        center = (CELL_W - 1 - center[0], center[1])
    facing = -1.0 if flip else 1.0
    return layers, pivots, neck, center, facing, mouth


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, tilt=0.0, spin=0.0, jaw=0,
        wing=0.0, crest=0.0, tail=0.0, talon=0.0,
        ring=-1, rings=0, lines=0, burst=0, flash=0, sparks=0, recoil=0.0,
    )
    if action == "idle":
        p["dy"] = wave(f, n, 1.6)
        p["wing"] = wave(f, n, 8.0)
        p["crest"] = wave(f, n, 4.0, phase=0.4)
        p["tail"] = wave(f, n, 6.0, phase=-0.8)
        p["talon"] = wave(f, n, 5.0, phase=1.1)
    elif action == "walk":
        p["dy"] = wave(f, n, 1.2, cycles=2.0)
        p["dx"] = wave(f, n, 2.4, cycles=2.0)
        p["tilt"] = -6.0
        p["wing"] = wave(f, n, 14.0, cycles=2.0)
        p["crest"] = wave(f, n, 3.0, cycles=2.0)
        p["tail"] = wave(f, n, 8.0, phase=-0.7, cycles=2.0)
        p["talon"] = wave(f, n, 4.0, phase=0.5, cycles=2.0)
        p["lines"] = 3
    elif action == "attack":
        # 0-4 roar, 5-10 in-place spin, 11-12 impact, 13-15 recover.
        crest_t = [6, 12, 16, 16, 12, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 0]
        wing_t = [4, 10, 12, 10, 6, 0, 0, 0, 0, 0, 0, 8, 6, 3, 0, 0]
        jaw_t = [1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
        p["crest"] = crest_t[f]
        p["wing"] = wing_t[f]
        p["jaw"] = jaw_t[f]
        p["tail"] = wing_t[f] * 0.4
        p["talon"] = -3.0 if f <= 4 else 0.0
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
            p["lines"] = 4
        if 1 <= f <= 4:
            p["ring"], p["rings"] = f - 1, 3
        if f in (11, 12):
            p["burst"] = 1
    else:
        env = [0, 0.45, 0.85, 1.0, 0.92, 0.7, 0.48, 0.3, 0.16, 0.08, 0.02, 0][f]
        p["dx"] = -7.0 * env
        p["dy"] = -1.0 * env
        p["recoil"] = 34.0 * env
        p["jaw"] = 1 if 1 <= f <= 7 else 0
        p["wing"] = 12.0 * env
        p["crest"] = 10.0 * env
        p["tail"] = -8.0 * env
        p["talon"] = 8.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def synth(rig, action, f, palette):
    layers, pivots, neck, center, facing, mouth = rig
    p = pose(action, f)
    own_jaw = p["jaw"]
    p = gen4_attack_strike.finish(p, action, f, __file__)
    # The shared PEAKS table pushes jaw to 3; this species keeps its own small beak opening.
    p["jaw"] = own_jaw
    head_dx, head_dy = 0, 0
    body_ang = 0.0

    fwd = facing
    bdx, bdy = p["dx"] * fwd, p["dy"]
    sgn = fwd

    def moved(name, ang, dx, dy, pivot=None):
        pv = pivots[name] if pivot is None else pivot
        return shift(rotate_nearest(layers[name], ang * sgn, pv), dx, dy)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4

    # The tail tip hangs off the right wing. Same pivot and angle keeps that feather one piece.
    wing_l = moved("wing_l", p["wing"], bdx, bdy)
    wr_ang = -p["wing"] * 0.85
    wing_r = moved("wing_r", wr_ang, bdx, bdy)
    tail = moved("tail", wr_ang, bdx, bdy, pivot=pivots["wing_r"])
    talon_lo = moved("talon_lo", p["talon"], bdx, bdy)
    talon_hi = moved("talon_hi", -p["talon"], bdx, bdy)
    # Crest flares back (up off the skull). Head stays an integer shift of the body.
    crest = moved("crest", -p["crest"], bdx, bdy)
    torso = shift(layers["torso"], bdx, bdy)
    head = shift(layers["head"], bdx, bdy)

    # Torso and head sit under the parts that carry their joint cushions.
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for lyr in (torso, head, wing_l, wing_r, tail, talon_hi, talon_lo, crest):
        paste(body, lyr)
    part_moved = any(abs(p[k]) >= 0.45 for k in ("wing", "crest", "tail", "talon"))
    if part_moved and not spinning and not recoiling:
        body = despeckle(body)

    if abs(p["tilt"]) >= 0.45 and not spinning and not recoiling:
        pivot = (center[0] + bdx, center[1] + bdy)
        tilt_ang = p["tilt"] * fwd
        # Tilt wings, torso, tail, and talons. Keep the head pixels upright and
        # slide them so the neck stays on the tilted chest.
        head_m = head[..., 3] >= 128
        crest_m = crest[..., 3] >= 128
        held = body.copy()
        held[head_m | crest_m] = 0
        held = rotate_nearest(held, tilt_ang, pivot)
        nx, ny = content_point(neck[0] + bdx, neck[1] + bdy, tilt_ang, pivot)
        extra_x = int(round(nx - (neck[0] + bdx)))
        extra_y = int(round(ny - (neck[1] + bdy)))
        body = held
        head_dx, head_dy = extra_x, extra_y
        paste(body, shift(head, extra_x, extra_y))
        paste(body, shift(crest, extra_x, extra_y))
        body = despeckle(body)

    if spinning:
        body = rotate_nearest(body, p["spin"], center)
        body_ang = p["spin"]
        body = despeckle(body)
    elif recoiling:
        # Nose up, away from the beak. Cap stays under 38 degrees. No belly mirror.
        body = rotate_nearest(body, -p["recoil"] * fwd, center)
        body_ang = -p["recoil"] * fwd
        body = despeckle(body)

    img = snap(body, palette)
    if p["jaw"] and not spinning:
        # Small lens along the beak line; no wedge grown past the beak tip.
        line = mouth_open.transform_points(mouth, dx=bdx + head_dx, dy=bdy + head_dy)
        if abs(body_ang) >= 0.45:
            # Body was shifted first, then rotated about the (unshifted) centre.
            line = mouth_open.transform_points(line, angle=body_ang, pivot=center)
        img = mouth_open.open_mouth(img, line, amount=min(2, int(p["jaw"])), deep=INK, inner=RED_DEEP, peak=MOUTH_PEAK)
    if p["flash"]:
        img = flash_white(img)
    if p["rings"] and p["ring"] >= 0 and not spinning:
        bb = body_box(img)
        if bb:
            # Mouth is the forward edge of the head band, not the wingtip.
            my = int(round(center[1] - 28))
            mx = (bb[2] - 8) if facing > 0 else (bb[0] + 8)
            # Prefer the beak we just drew: search the eye band.
            img = roar_rings(img, (mx, my), fwd, p["ring"], p["rings"])
    if p["lines"]:
        img = speed_lines(img, fwd, p["lines"], seed=f * 17 + (0 if action == "walk" else 3), length=10)
    if p["burst"]:
        bb2 = body_box(img)
        if bb2:
            tip_x = bb2[2] - 6 if fwd > 0 else bb2[0] + 6
            tip_y = int(round(center[1] - 24))
            img = impact_burst(img, (tip_x, tip_y), 10 if f >= 12 else 8)
    if p["sparks"]:
        img = sparks(img, f, fwd)
    return snap(img, palette)


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
    samples = {"idle": (0, 4, 8, 12), "walk": (0, 4, 8, 12), "attack": (2, 4, 7, 11), "hurt": (1, 3, 6, 9)}
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


def publish_dir8(sprite_h):
    text = DIR8.read_text()
    prefix = "window.ARENA_DIR8 = "
    if not text.startswith(prefix):
        raise SystemExit("dir8.js header changed; not publishing")
    entry = {
        "dex": 398, "form": "mega", "variant": "staraptor-mega", "gen": "gen4",
        "dir": "monsterSprite/gen4/0398_staraptor/mega/staraptor-mega",
        "sheetDir": "agent-sprite-forge", "layout": "actions_8dir", "nf": 16,
        "cw": CELL_W, "ch": CELL_H, "w": CELL_W, "h": int(sprite_h), "fy": FOOT_Y,
        "an": [FRAMES[a] for a in ACTIONS],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    key = '"0398_staraptor-mega":'
    if key not in text:
        raise SystemExit("dir8 key missing")
    start = text.index(key) + len(key)
    while text[start].isspace():
        start += 1
    if text[start] != "{":
        raise SystemExit("dir8 value is not an object")
    depth = 0
    end = None
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    if end is None:
        raise SystemExit("dir8 value did not close")
    text = text[:start] + blob + text[end:]
    DIR8.write_text(text)
    json.loads(text[len(prefix):].rstrip().rstrip(";"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(MEGA_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    faces_right, eye_x, eye_y, tip_x = measure_facing(front)
    if not faces_right:
        raise SystemExit(f"expected beak tip x > eye x, got tip {tip_x} eye {eye_x}")
    print(f"facing right eye_x={eye_x:.1f} eye_y={eye_y:.1f} beak_x={tip_x}", flush=True)

    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), "sum", sum(counts.values()), flush=True)

    rigs = {}
    for d in DIRS:
        rigs[d] = build_rig(front, masks, flip=d not in FACE_RIGHT)

    rest = synth(rigs["E"], "idle", 0, palette)
    a = front[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    dst_x = (CELL_W - (x1 - x0)) // 2
    dst_y = FOOT_Y - (y1 - y0)
    placed = np.zeros_like(rest)
    placed[dst_y:dst_y + (y1 - y0), dst_x:dst_x + (x1 - x0)] = front[y0:y1, x0:x1]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    diff_n = int(differ.sum())
    print("rest-pose pixels changed", diff_n, flush=True)
    if diff_n:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (190, 190, 198), "wing_l": (220, 60, 60), "wing_r": (60, 90, 220),
        "tail": (40, 180, 80), "talon_lo": (240, 200, 40), "talon_hi": (240, 140, 40),
        "head": (240, 240, 240), "crest": (255, 40, 40),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "E", f) for f in (0, 4, 8, 12)]
        want += [("walk", "E", f) for f in (0, 4, 8, 12)]
        want += [("attack", "E", f) for f in (0, 2, 4, 5, 7, 8, 10, 11, 13)]
        want += [("attack", "W", f) for f in (3, 7)]
        want += [("hurt", "E", f) for f in (0, 1, 3, 6, 11)]
        tiles = []
        for act, d, f in want:
            tile = synth(rigs[d], act, f, palette)
            frac, total = largest_fraction(tile)
            edge = tile[..., 3] >= 128
            touch = bool(edge[0].any() or edge[-1].any() or edge[:, 0].any() or edge[:, -1].any())
            print(f"sample {act} {d} {f:02d} main={frac:.3f} n={total} edge={touch}", flush=True)
            tiles.append(tile)
        cols = 6
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
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview_image(sheets).save(OUT_DIR / "preview.png")
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
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = set(np.unique(master[..., 3]).tolist())
    clip = {}
    spin_frac = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and d == "E" and 5 <= f <= 10:
                    frac, _n = largest_fraction(tile)
                    spin_frac[f"{f}"] = round(frac, 3)
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": sorted(alphas),
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "spin_main_frac_E": spin_frac,
        "facing": "right", "beak_x": tip_x, "eye_x": eye_x,
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                report[f"gait_{act}"] = "skipped_existing"
                continue
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:900], flush=True)

    meta = {
        "generator": "forge_staraptor_mega_actions", "version": VERSION, "key": "0398_staraptor-mega",
        "dex": 398, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS, "face_right": sorted(FACE_RIGHT),
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "facing": "right", "beak_x": tip_x, "eye_x": eye_x,
        "source": "front.png 96x96", "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        if off or clip:
            raise SystemExit("refusing to publish with palette or edge errors")
        publish_dir8(sprite_h)
        print("dir8 published", VERSION, flush=True)


if __name__ == "__main__":
    main()
