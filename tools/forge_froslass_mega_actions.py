#!/usr/bin/env python3
"""Mega Froslass (0478_froslass-mega) 8-direction action sheet from front.png.

Archetype: floating kimono ghost (queue label avian_raptor_hover, but this
still has no wings and no legs). Measured on front.png: opaque 2492, bbox
(21,1)-(75,94), one connected piece. Eyes (yellow/orange) sit just right of
center. The red at the lower face is the existing mouth, not a jaw to invent.
Sleeves hang around y 50-64. The hem from y 67 down, tassels included, is one
cloth shape. Splits in the hem are not feet.

Unflipped rows are S SE E NE (eyes stay right of center). N NW W SW are the
horizontal mirror, so E faces screen-right and W faces screen-left.

  idle   : hover. Horns, hair, sleeves, and the hem sway. Frame 0 is the still.
  walk   : glide. Same cloth trails backward. No foot cycle and no invented legs.
  attack : small open of the existing mouth plus horn flare and rings (0-4),
           in-place 360 of the whole body as one piece (5-10), impact (11-12),
           recover (13-15).
  hurt   : flash, backward recoil of at most 36 degrees, sparks, recover.
           No belly mirror and no vertical squash.

The head is only an integer shift except inside the rigid spin and the recoil.
Every opaque pixel is snapped to the still palette. Alpha is only 0 or 255.
front/back png and gif are not written.

Resumable via agent-sprite-forge/_work. --force clears only this form's frames.
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

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen4/0478_froslass/mega/froslass-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0478_froslass/mega/froslass-mega/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
FACE_RIGHT = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 176, 156
VERSION = "froslass_mega_v2"

INK = (16, 16, 16)
RED = (221, 51, 43)
DRED = (106, 49, 49)
PINK = (252, 126, 106)
WHITE = (255, 255, 255)
ICE = (201, 238, 255)
CYAN = (113, 208, 255)
YEL = (255, 213, 90)
ORG = (222, 148, 57)


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def components(mask):
    """4-connected components as boolean masks, largest first."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    found = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        pix = []
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        m = np.zeros_like(mask)
        for y, x in pix:
            m[y, x] = True
        found.append(m)
    found.sort(key=lambda m: int(m.sum()), reverse=True)
    return found


def segment(img):
    """Partition this still. Horns, hair, sleeves, and one hem. No legs."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    # Free spikes only. The joining row stays on the head so a flare does not split the skull.
    horn_comps = components(a & (yy <= 11))
    if len(horn_comps) < 2:
        raise SystemExit("expected two separate horns above the face")
    horns = horn_comps[:2]
    mean_x = []
    for m in horns:
        mean_x.append(float(np.nonzero(m)[1].mean()))
    horn_l = horns[int(np.argmin(mean_x))]
    horn_r = horns[int(np.argmax(mean_x))]
    bridge = a & (yy >= 12) & (yy <= 13)
    head = (a & (yy >= 14) & (yy <= 46) & (xx >= 43) & (xx <= 64)) | bridge
    head = head & ~horn_l & ~horn_r
    used = horn_l | horn_r | head
    hair_l = a & ~used & (xx <= 42) & (yy >= 14) & (yy <= 49)
    hair_r = a & ~used & (xx >= 65) & (yy >= 14) & (yy <= 49)
    used = used | hair_l | hair_r
    sleeve_l = a & ~used & (xx <= 44) & (yy >= 50) & (yy <= 66)
    sleeve_r = a & ~used & (xx >= 62) & (yy >= 50) & (yy <= 66)
    used = used | sleeve_l | sleeve_r
    # One cloth hem, tassels included. The gaps near the bottom are not feet.
    hem = a & ~used & (yy >= 67)
    torso = a & ~(horn_l | horn_r | head | hair_l | hair_r | sleeve_l | sleeve_r | hem)
    parts = {
        "horn_l": horn_l, "horn_r": horn_r, "head": head,
        "hair_l": hair_l, "hair_r": hair_r,
        "sleeve_l": sleeve_l, "sleeve_r": sleeve_r,
        "hem": hem, "torso": torso,
    }
    own = None
    for m in parts.values():
        own = m.copy() if own is None else (own | m)
    overlap = 0
    names = list(parts)
    for i, ni in enumerate(names):
        for nj in names[i + 1:]:
            overlap += int((parts[ni] & parts[nj]).sum())
    if int(own.sum()) != int(a.sum()) or overlap:
        raise SystemExit(f"part masks do not partition the still (sum {int(own.sum())} opaque {int(a.sum())} overlap {overlap})")
    return parts


def measure_eyes(img):
    a = img[..., 3] >= 128
    yy = np.mgrid[0:img.shape[0], 0:img.shape[1]][0]
    rgb = img[..., :3]
    eye = a & (yy <= 28) & (
        np.all(rgb == np.array(YEL, np.uint8), axis=-1)
        | np.all(rgb == np.array(ORG, np.uint8), axis=-1)
    )
    ys, xs = np.nonzero(eye)
    if len(xs) == 0:
        raise SystemExit("eye pixels not found")
    return float(np.mean(xs)), float(np.mean(ys))


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


def small_mouth(img, amount):
    """Open the existing red mouth by 1px into its own black outline. No new jaw."""
    if amount <= 0:
        return img
    out = img.copy()
    rgb = out[..., :3]
    a = out[..., 3] >= 128
    red = a & np.all(rgb == np.array(RED, np.uint8), axis=-1)
    ys, xs = np.nonzero(red)
    if len(xs) < 8:
        return out
    # Only the lower lip's middle, where a black outline pixel already sits underneath.
    x_lo, x_hi = int(np.median(xs)) - 1, int(np.median(xs)) + 1
    opened = 0
    for x in range(max(0, x_lo), min(img.shape[1], x_hi + 1)):
        col = np.nonzero(red[:, x])[0]
        if len(col) == 0:
            continue
        y = int(col.max())
        if y + 1 >= img.shape[0]:
            continue
        below = out[y + 1, x]
        if int(below[3]) < 128 or tuple(int(v) for v in below[:3]) != INK:
            continue
        out[y + 1, x, :3] = RED
        out[y + 1, x, 3] = 255
        out[y, x, :3] = INK
        out[y, x, 3] = 255
        opened += 1
    if opened == 0:
        return img
    return out


def roar_rings(out, mouth, facing, age, strength):
    """Ice rings in front of the existing mouth. Not a beak wedge."""
    mx, my = mouth
    cols = (WHITE, ICE, CYAN)
    for r_i in range(strength):
        rad = 8 + age * 3 + r_i * 5
        col = cols[min(r_i, 2)]
        # Lower arc so the rings leave the eyes. A little bias toward the facing.
        for deg in range(15, 166, 12):
            ang = math.radians(deg if facing > 0 else (180 - deg))
            put(out, mx + rad * math.cos(ang), my + 2 + rad * math.sin(ang) * 0.85, col)
    return out


def speed_lines(out, facing, n, seed, length):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    rear_x = (x0 + (x1 - x0) // 6) if facing > 0 else (x1 - (x1 - x0) // 6)
    y_lo = y0 + (y1 - y0) // 3
    y_hi = y0 + (y1 - y0) * 2 // 3
    for _k in range(n):
        y = int(rng.integers(y_lo, max(y_lo + 1, y_hi)))
        off = int(rng.integers(4, 9))
        length_i = int(rng.integers(max(3, length // 2), length + 1))
        for i in range(length_i):
            put(out, rear_x - facing * (off + i), y, WHITE if i < 2 else ICE)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(size // 2, size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), WHITE if r < size - 2 else CYAN)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, WHITE)
    return out


def sparks(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, _y1 = bb
    d = frame_idx
    cx = (x0 + x1) // 2
    pts = [
        (cx - facing * (6 + d * 2), y0 + 10 - d),
        (cx + facing * (5 + d), y0 + 6),
        (cx - facing * 2, y0 + 16),
    ]
    for sx, sy in pts:
        put(out, sx, sy, WHITE)
        put(out, sx - facing, sy + 1, ICE)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def contact_pivot(part, anchor):
    """Contact pixel of `part` nearest the anchor's centre."""
    ys, xs = np.nonzero(part)
    if len(ys) == 0:
        return 0.0, 0.0
    ay, ax = np.nonzero(anchor)
    if len(ay) == 0:
        return float(xs.mean()), float(ys.mean())
    acx, acy = float(ax.mean()), float(ay.mean())
    touch = part & dilate(anchor, 2)
    ty, tx = np.nonzero(touch)
    if len(ty) == 0:
        ty, tx = ys, xs
    d = (tx.astype(np.int32) - int(round(acx))) ** 2 + (ty.astype(np.int32) - int(round(acy))) ** 2
    i = int(d.argmin())
    return float(tx[i]), float(ty[i])


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

    # Joint colour lives under the cloth, covered at rest by the part itself.
    cushions = (
        ("horn_l", "head", 3),
        ("horn_r", "head", 3),
        ("hair_l", "head", 4),
        ("hair_r", "head", 4),
        ("sleeve_l", "torso", 5),
        ("sleeve_r", "torso", 5),
        ("hem", "torso", 5),
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
        "horn_l": contact_pivot(placed["horn_l"], placed["head"]),
        "horn_r": contact_pivot(placed["horn_r"], placed["head"]),
        "hair_l": contact_pivot(placed["hair_l"], placed["head"]),
        "hair_r": contact_pivot(placed["hair_r"], placed["head"]),
        "sleeve_l": contact_pivot(placed["sleeve_l"], placed["torso"] | placed["head"]),
        "sleeve_r": contact_pivot(placed["sleeve_r"], placed["torso"] | placed["head"]),
        "hem": contact_pivot(placed["hem"], placed["torso"]),
    }
    opaque = np.zeros((CELL_H, CELL_W), bool)
    for pm in placed.values():
        opaque |= pm
    oys, oxs = np.nonzero(opaque)
    center = (float(oxs.mean()), float(oys.mean()))

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
    facing = -1.0 if flip else 1.0
    return layers, pivots, center, facing


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, tilt=0.0, spin=0.0, jaw=0,
        horn=0.0, hair=0.0, sleeve=0.0, hem=0.0, trail=0.0,
        ring=-1, rings=0, lines=0, burst=0, flash=0, sparks=0, recoil=0.0,
    )
    if action == "idle":
        p["dy"] = wave(f, n, 1.4)
        p["horn"] = wave(f, n, 3.2, phase=0.5)
        p["hair"] = wave(f, n, 5.5, phase=-0.4)
        p["sleeve"] = wave(f, n, 6.5, phase=0.9)
        p["hem"] = wave(f, n, 3.6, phase=-1.1)
    elif action == "walk":
        # Glide. Cloth trails back. The face is not tilted and no feet are drawn.
        p["dy"] = wave(f, n, 1.5, cycles=2.0)
        p["dx"] = wave(f, n, 1.8, cycles=2.0)
        p["horn"] = wave(f, n, 2.4, cycles=2.0)
        p["hair"] = wave(f, n, 7.0, phase=-0.5, cycles=2.0)
        p["sleeve"] = wave(f, n, 9.0, phase=0.6, cycles=2.0)
        p["hem"] = wave(f, n, 4.5, phase=-0.8, cycles=2.0)
        p["trail"] = 6.0
        p["lines"] = 3
    elif action == "attack":
        # 0-4 roar, 5-10 in-place spin, 11-12 impact, 13-15 recover.
        horn_t = [4, 10, 14, 14, 8, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0]
        hair_t = [2, 6, 8, 6, 3, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0]
        sleeve_t = [3, 8, 11, 8, 4, 0, 0, 0, 0, 0, 0, 5, 2, 0, 0, 0]
        hem_t = [1, 3, 4, 3, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0]
        jaw_t = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        p["horn"] = horn_t[f]
        p["hair"] = hair_t[f]
        p["sleeve"] = sleeve_t[f]
        p["hem"] = hem_t[f]
        p["jaw"] = jaw_t[f]
        if 5 <= f <= 9:
            p["spin"] = 60.0 * (f - 4)
        elif f == 10:
            p["spin"] = 0.0
        if 5 <= f <= 10:
            p["lines"] = 4
        if 1 <= f <= 4:
            p["ring"], p["rings"] = f - 1, 3
        if f in (11, 12):
            p["burst"] = 1
    else:
        env = [0, 0.45, 0.82, 1.0, 0.88, 0.66, 0.44, 0.26, 0.12, 0.05, 0.0, 0][f]
        p["dx"] = -6.0 * env
        p["dy"] = -1.0 * env
        p["recoil"] = 36.0 * env
        p["horn"] = 6.0 * env
        p["hair"] = 8.0 * env
        p["sleeve"] = 10.0 * env
        p["hem"] = -4.0 * env
        p["jaw"] = 1 if 2 <= f <= 5 else 0
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def mouth_point(img, facing, center):
    rgb = img[..., :3]
    a = img[..., 3] >= 128
    red = a & (
        np.all(rgb == np.array(RED, np.uint8), axis=-1)
        | np.all(rgb == np.array(DRED, np.uint8), axis=-1)
        | np.all(rgb == np.array(PINK, np.uint8), axis=-1)
    )
    ys, xs = np.nonzero(red)
    if len(xs) == 0:
        return center[0] + facing * 4, center[1] - 10
    return float(np.median(xs)), float(np.median(ys))


def synth(rig, action, f, palette):
    layers, pivots, center, facing = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    fwd = facing
    bdx, bdy = p["dx"] * fwd, p["dy"]
    sgn = fwd

    def moved(name, ang):
        return shift(rotate_nearest(layers[name], ang * sgn, pivots[name]), bdx, bdy)

    trail = p["trail"]
    horn_l = moved("horn_l", p["horn"])
    horn_r = moved("horn_r", -p["horn"])
    hair_l = moved("hair_l", p["hair"] + trail)
    hair_r = moved("hair_r", -(p["hair"] + trail))
    sleeve_l = moved("sleeve_l", p["sleeve"] + trail)
    sleeve_r = moved("sleeve_r", -(p["sleeve"] + trail))
    hem = moved("hem", p["hem"] + trail * 0.45)
    torso = shift(layers["torso"], bdx, bdy)
    head = shift(layers["head"], bdx, bdy)

    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for lyr in (torso, head, hem, hair_l, hair_r, sleeve_l, sleeve_r, horn_l, horn_r):
        paste(body, lyr)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    part_moved = any(abs(p[k]) >= 0.45 for k in ("horn", "hair", "sleeve", "hem")) or abs(trail) >= 0.45
    if part_moved and not spinning and not recoiling:
        body = despeckle(body)

    pivot = (center[0] + bdx, center[1] + bdy)
    if spinning:
        body = despeckle(rotate_nearest(body, p["spin"] * fwd, pivot))
    elif recoiling:
        # Lean back. Cap is 36 degrees. No belly-up mirror and no squash.
        body = despeckle(rotate_nearest(body, -p["recoil"] * fwd, pivot))
    elif abs(p["tilt"]) >= 0.45:
        body = despeckle(rotate_nearest(body, p["tilt"] * fwd, pivot))

    img = snap(body, palette)
    if p["jaw"] and not spinning:
        img = small_mouth(img, int(p["jaw"]))
    if p["flash"]:
        img = flash_white(img)
    if p["rings"] and p["ring"] >= 0 and not spinning:
        mx, my = mouth_point(img, fwd, pivot)
        img = roar_rings(img, (mx, my), fwd, p["ring"], p["rings"])
    if p["lines"]:
        img = speed_lines(img, fwd, p["lines"], seed=f * 17 + (0 if action == "walk" else 5), length=9)
    if p["burst"]:
        mx, my = mouth_point(img, fwd, pivot)
        img = impact_burst(img, (mx + fwd * 10, my + 4), 10 if f >= 12 else 8)
    if p["sparks"]:
        img = sparks(img, f, fwd)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (MEGA_DIR / "front.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def component_stats(img):
    a = img[..., 3] >= 128
    if not np.any(a):
        return 1.0, 0, 0
    sizes = [int(m.sum()) for m in components(a)]
    total = sum(sizes)
    best = sizes[0]
    second = sizes[1] if len(sizes) > 1 else 0
    return best / total, second, total


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
    samples = {"idle": (0, 4, 8, 12), "walk": (0, 4, 8, 12), "attack": (2, 4, 7, 11), "hurt": (1, 3, 6, 10)}
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
    """Copy sheet files. Never copy _work, and never touch front/back stills."""
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.rglob("*"):
        if not fp.is_file():
            continue
        rel = fp.relative_to(OUT_DIR)
        if rel.parts[0] == "_work":
            continue
        dest = SYNC_OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fp, dest)
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
    eye_x, eye_y = measure_eyes(front)
    if eye_x <= 48:
        raise SystemExit(f"expected eyes right of center, got {eye_x:.1f}")
    print(f"eyes x={eye_x:.1f} y={eye_y:.1f} (native faces screen-right)", flush=True)

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
        "torso": (80, 80, 96), "head": (255, 214, 214),
        "horn_l": (70, 140, 255), "horn_r": (40, 90, 220),
        "hair_l": (255, 170, 50), "hair_r": (255, 110, 30),
        "sleeve_l": (40, 200, 120), "sleeve_r": (20, 140, 80),
        "hem": (180, 80, 220),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "E", f) for f in (0, 4, 8, 12)]
        want += [("walk", "E", f) for f in (0, 4, 8, 12)]
        want += [("attack", "E", f) for f in (0, 2, 4, 5, 7, 9, 10, 11, 15)]
        want += [("attack", "W", f) for f in (3, 7)]
        want += [("hurt", "E", f) for f in (0, 1, 3, 6, 11)]
        tiles = []
        for act, d, f in want:
            tile = synth(rigs[d], act, f, palette)
            frac, second, total = component_stats(tile)
            edge = tile[..., 3] >= 128
            touch = bool(edge[0].any() or edge[-1].any() or edge[:, 0].any() or edge[:, -1].any())
            print(f"sample {act} {d} {f:02d} main={frac:.3f} second={second} n={total} edge={touch}", flush=True)
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
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions_dir = OUT_DIR / "actions"
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
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = set(np.unique(master[..., 3]).tolist())
    clip = {}
    spin_frac = {}
    spin_second = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and d == "E" and 5 <= f <= 10:
                    frac, second, _n = component_stats(tile)
                    spin_frac[f"{f}"] = round(frac, 3)
                    spin_second[f"{f}"] = second
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": sorted(alphas),
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "spin_main_frac_E": spin_frac, "spin_second_E": spin_second,
        "facing": "front-right-bias", "eye_x": eye_x, "eye_y": eye_y,
        "cell": [CELL_W, CELL_H, FOOT_Y],
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
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1200], flush=True)

    meta = {
        "generator": "forge_froslass_mega_actions", "version": VERSION, "key": "0478_froslass-mega",
        "dex": 478, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS, "face_right": sorted(FACE_RIGHT),
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "facing": "front-right-bias", "eye_x": eye_x, "eye_y": eye_y,
        "source": "front.png 96x96", "gifs_untouched": True,
        "layout": "actions_8dir", "sheetDir": "agent-sprite-forge", "raw": 1,
        "an": [16, 16, 16, 12],
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if off or clip or any(v < 0.9 for v in spin_frac.values()):
        print("QA gate failed", flush=True)
    if not args.no_sync and not off and not clip:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
