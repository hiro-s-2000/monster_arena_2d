#!/usr/bin/env python3
"""Mismagius (0429) 8-direction action sheet from this folder's 96x96 front.png.

Measured on this still only (opaque 1603, bbox x21-73 y15-79, 53x65, 12 colours,
alpha 0/255, one 4-connected component). It faces the viewer: both red eyes sit
under the brim around y38-42, with gold and yellow glints and two white pixels.
A separate three-pixel red mark on the chest (y52-57) is not a mouth. back.png
has no face. No feet, no legs, no mouth. The hem ends in a point. It floats.

Parts, from this still only:
  brim_l   left jagged hat wing, including the upper-left spike (x<=36, y22-48).
  brim_r   right hat wing and its magenta trim (not the hat tip).
  hair_l   lower-left magenta tatter. Not a foot.
  hair_r   right hanging magenta lock / sleeve tatter. Not a foot.
Eye colours (red, dark red, white, gold, yellow) stay on the body. The hat
cone, face, neck, and cloak stay one body. Nothing below the hem is drawn.

idle   : planted. Brim and hair sway. Frame 0 is the still.
walk   : hover bob. Hair and brim sway. Not a stomp. No legs.
attack : 0-4 flare the brim, hair, and sleeve plus sparse rings,
         5-10 one-piece in-plane 360 spin (no |cos| squash),
         11-12 impact, 13-15 recover.
hurt   : rigid lean of at most 38 degrees.

Palette-locked. Alpha is only 0 or 255. front/back png and gif are not written.
"""

import argparse
import fcntl
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
STILL_DIR = ROOT / "monsterSprite/gen4/0429_mismagius/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0429_mismagius/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 176, 155
VERSION = "mismagius_v2"
KEY = "0429_mismagius"
UNFLIPPED = {"S", "SE", "E", "NE"}
MAX_RECOIL = 38.0
MOVING = ("brim_l", "brim_r", "hair_l", "hair_r")

MID = (106, 74, 148)
SHADE = (74, 57, 90)
DARK = (49, 32, 57)
PINK = (180, 98, 180)
INK = (16, 16, 16)
GRAY = (90, 65, 98)
MPINK = (131, 82, 131)
RED = (238, 57, 16)
DRED = (148, 57, 90)
GOLD = (205, 156, 0)
YEL = (246, 222, 57)
WHITE = (255, 255, 255)

# Outward flare in degrees. Frame 0 is a small open with no rings, so the
# spin of that same pose stays near ratio 1. Frame 15 is closed.
FLARE = [4.0, 9.0, 14.0, 16.0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0, 6.0, 3.0, 0.0]
RING = [-1, 0, 1, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
FLARE_AMP = {"brim_l": 1.0, "brim_r": 1.0, "hair_l": 1.15, "hair_r": 1.15}
SWAY_AMP = {"brim_l": 0.55, "brim_r": 0.55, "hair_l": 1.2, "hair_r": 1.2}


def is_color(img, color):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(color, np.uint8), axis=-1)


def component_stats(mask):
    seen = np.zeros_like(mask)
    h, w = mask.shape
    n = 0
    best = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        n += 1
        stack = [(y0, x0)]
        seen[y0, x0] = True
        count = 0
        while stack:
            y, x = stack.pop()
            count += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if count > best:
            best = count
    return n, best


def segment(img):
    """Hat wings, hair tatters, and one cloak body. No legs and no mouth."""
    op = img[..., 3] >= 128
    h, w = op.shape
    yy, xx = np.mgrid[0:h, 0:w]
    pink = is_color(img, PINK) | is_color(img, MPINK)
    face_col = (
        is_color(img, RED)
        | is_color(img, DRED)
        | is_color(img, WHITE)
        | is_color(img, GOLD)
        | is_color(img, YEL)
    )
    # Left jagged wing of the hat, including the upper-left spike.
    brim_l = op & (xx <= 36) & (yy >= 22) & (yy <= 48)
    # Right wing. The hat tip (pink above y26, x<60) stays on the cone.
    brim_r = op & (yy >= 26) & (yy <= 50) & (xx >= 62)
    brim_r |= pink & (yy >= 26) & (yy <= 48) & (xx >= 54)
    # Hanging magenta tatters. The left spike is part of brim_l, not a foot.
    hair_l = pink & (yy >= 60) & (xx <= 50)
    hair_r = pink & (yy >= 54) & (xx >= 54)
    parts = {
        "brim_l": brim_l,
        "brim_r": brim_r,
        "hair_l": hair_l,
        "hair_r": hair_r,
    }
    owned = np.zeros_like(op)
    overlap = 0
    for name in MOVING:
        parts[name] = parts[name] & op & ~face_col & ~owned
        overlap += int((owned & parts[name]).sum())
        owned |= parts[name]
    body = op & ~owned
    parts["body"] = body
    if int(owned.sum()) + int(body.sum()) != int(op.sum()) or overlap:
        raise SystemExit(
            f"masks do not partition the still: overlap {overlap} covered {int((owned | body).sum())} opaque {int(op.sum())}"
        )
    if int((face_col & ~body).sum()):
        raise SystemExit("an eye or the chest mark left the body")
    counts = {k: int(v.sum()) for k, v in parts.items()}
    if counts["body"] < 1100 or counts["brim_l"] < 80 or counts["brim_r"] < 40:
        raise SystemExit(f"hat wings did not separate: {counts}")
    if counts["hair_l"] < 8 or counts["hair_r"] < 20:
        raise SystemExit(f"hair tatters missing: {counts}")
    n_body, best_body = component_stats(body)
    if best_body < int(body.sum()) * 0.92:
        raise SystemExit(f"body split: components {n_body} largest {best_body} of {int(body.sum())}")
    # The hem is the lowest opaque row. No pixel is a foot below it.
    ys = np.nonzero(op)[0]
    if int(ys.max()) > 80:
        raise SystemExit("pixels below the hem; a foot may have been invented")
    return parts


def measure_face(img, masks):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    red = is_color(img, RED)
    eyes = red & (yy <= 46)
    chest = red & (yy >= 50)
    if int(eyes.sum()) < 6 or int(chest.sum()) < 3:
        raise SystemExit("eyes or chest mark missing; facing cannot be fixed")
    return {
        "eyes": int(eyes.sum()),
        "chest": int(chest.sum()),
        "gold": int(is_color(img, GOLD).sum()),
        "white": int(is_color(img, WHITE).sum()),
        "feet": 0,
        "mouth": 0,
        "centroid_x": round(float(xs.mean()), 2),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "parts": {k: int(v.sum()) for k, v in masks.items()},
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
    """4x nearest in-plane rotate. A multiple of 360 stays exact. No squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
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
    """Drop only pixels whose four neighbours are all empty. After a warp only."""
    out = img.copy()
    a = out[..., 3] >= 128
    pad = np.pad(a, 1)
    n = (
        pad[:-2, 1:-1].astype(np.int16)
        + pad[2:, 1:-1]
        + pad[1:-1, :-2]
        + pad[1:-1, 2:]
    )
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
    for i in range(0, len(ty), 400):
        sl = slice(i, i + 400)
        diff_y = ty[sl, None].astype(np.int32) - pts[None, :, 0]
        diff_x = tx[sl, None].astype(np.int32) - pts[None, :, 1]
        dist = diff_y * diff_y + diff_x * diff_x
        pick[sl] = dist.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def extract_palette(img):
    opaque = img[..., 3] >= 128
    return np.unique(img[opaque][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    opaque = out[..., 3] >= 128
    if not np.any(opaque):
        out[..., 3] = 0
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = opaque.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    mapped = palette[dist.argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)
    out[~opaque] = 0
    return out


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def side_of(mask, center_x):
    xs = np.nonzero(mask)[1]
    return -1.0 if float(xs.mean()) < center_x else 1.0


def socket_pivot(mask, center):
    """Contact nearest the cloak, so the wing hinges instead of orbiting."""
    ys, xs = np.nonzero(mask)
    d = (xs - center[0]) ** 2 + (ys - center[1]) ** 2
    k = min(8, len(xs))
    idx = np.argpartition(d, k - 1)[:k]
    return float(xs[idx].mean()), float(ys[idx].mean())


def build_rig(img, masks, flip):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - (sh - 1)
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]

    placed = {}
    for name, mask in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = mask[y0:y1, x0:x1]
        placed[name] = pm

    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr

    if flip:
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
        canvas = np.ascontiguousarray(canvas[:, ::-1])

    opaque = canvas[..., 3] >= 128
    oys, oxs = np.nonzero(opaque)
    center = (float(oxs.mean()), float(oys.mean()))
    bys, bxs = np.nonzero(placed["body"])
    body_center = (float(bxs.mean()), float(bys.mean()))
    pivots = {}
    sides = {}
    yy = np.arange(CELL_H)[:, None]
    xx = np.arange(CELL_W)[None, :]
    for name in MOVING:
        pivots[name] = socket_pivot(placed[name], body_center)
        sides[name] = side_of(placed[name], body_center[0])
        py = pivots[name][1]
        px = pivots[name][0]
        root = placed[name] & ((yy - py) ** 2 + (xx - px) ** 2 <= 2.5 ** 2)
        fill = nearest_color(canvas, root, placed["body"])
        write = root & (layers["body"][..., 3] < 128)
        layers["body"][write] = fill[write]
    facing = -1.0 if flip else 1.0
    return layers, pivots, sides, center, body_center, facing, canvas


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(dx=0.0, dy=0.0, sway=0.0, flare=0.0, spin=0.0, recoil=0.0, ring=-1, burst=0, flash=0, sparks=0)
    if action == "idle":
        p["sway"] = wave(f, n, 4.0)
        p["dy"] = wave(f, n, -1.0)
    elif action == "walk":
        p["sway"] = wave(f, n, 7.0, cycles=2.0)
        p["dy"] = wave(f, n, -2.0, cycles=2.0)
    elif action == "attack":
        p["flare"] = FLARE[f]
        p["ring"] = RING[f]
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
        if f in (11, 12):
            p["burst"] = 2 if f == 11 else 1
        if f == 13:
            p["dy"] = -1.0
    else:
        env = [0.0, 0.55, 1.0, 0.82, 0.58, 0.38, 0.22, 0.10, 0.03, 0.0, 0.0, 0.0][f]
        p["recoil"] = MAX_RECOIL * env
        p["dx"] = -4.0 * env
        p["dy"] = -1.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def compose(layers, pivots, sides, p):
    """Cloak stays put. Brim, hair, and the sleeve hinge from their sockets."""
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(body, shift(layers["body"], p["dx"], p["dy"]))
    for name in MOVING:
        side = sides[name]
        ang = (-side) * p["flare"] * FLARE_AMP[name] + p["sway"] * SWAY_AMP[name]
        spun = rotate_nearest(layers[name], ang, pivots[name])
        paste(body, shift(spun, p["dx"], p["dy"]))
    return body


def roar_rings(out, center, age):
    """Sparse rings outside the cloak. Age 0 is one ring, 2 is three."""
    cx, cy = center
    cols = (YEL, GOLD, WHITE)
    radii = (42, 48, 54)
    steps = (28, 32, 36)
    drawn = 0
    for ri in range(age + 1):
        rad = radii[ri]
        step = steps[ri]
        col = cols[ri]
        deg = 0
        while deg < 360:
            ang = math.radians(deg)
            drawn += put(out, cx + rad * math.cos(ang), cy + rad * math.sin(ang) * 0.92, col)
            deg += step
    return out, drawn


def impact_burst(out, center, size):
    cx, cy = center
    ticks = [(0, -40), (18, -34), (-18, -34), (30, -18), (-30, -18), (34, -4), (-34, -4), (8, -44), (-8, -44)]
    col = YEL if size > 1 else WHITE
    drawn = 0
    for dx, dy in ticks[: 6 + size]:
        drawn += put(out, cx + dx, cy + dy, col)
        if size > 1:
            drawn += put(out, cx + dx, cy + dy - 1, GOLD)
    return out, drawn


def sparks(out, frame_idx, center):
    cx, cy = center
    d = frame_idx
    pts = [(cx - 22 - d, cy - 30), (cx + 22 + d, cy - 28), (cx, cy - 40 - d)]
    drawn = 0
    for sx, sy in pts:
        drawn += put(out, sx, sy, YEL)
        drawn += put(out, sx, sy + 1, WHITE)
    return out, drawn


def flash_white(img):
    out = img.copy()
    opaque = out[..., 3] >= 128
    black = opaque & np.all(out[..., :3] == np.array(INK, np.uint8), axis=-1)
    out[opaque & ~black, :3] = WHITE
    return out


def largest_fraction(img):
    opaque = img[..., 3] >= 128
    if not np.any(opaque):
        return 1.0, 0
    seen = np.zeros_like(opaque)
    h, w = opaque.shape
    best = 0
    total = int(opaque.sum())
    ys, xs = np.nonzero(opaque)
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
                if 0 <= ny < h and 0 <= nx < w and opaque[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if n > best:
            best = n
    return best / total, total


def synth(rig, action, f, palette, effects=True):
    layers, pivots, sides, center, body_center, facing, _canvas = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        # One piece. Spin the frame-0 pose so the opaque count stays with frame 0.
        base = pose(action, 0)
        base = gen4_attack_strike.finish(base, action, 0, __file__)

        base["dx"] *= facing
        body = compose(layers, pivots, sides, base)
        if recoiling:
            body = shift(body, p["dx"] * facing, p["dy"])
            pivot = (center[0] + p["dx"] * facing, center[1] + p["dy"])
            body = rotate_nearest(body, -p["recoil"] * facing, pivot)
        else:
            body = rotate_nearest(body, p["spin"] * facing, center)
        body = despeckle(body)
    else:
        moved = dict(p)
        moved["dx"] = p["dx"] * facing
        body = compose(layers, pivots, sides, moved)
        if abs(p["sway"]) > 0.45 or abs(p["flare"]) > 0.45 or abs(p["dy"]) > 0.45 or abs(p["dx"]) > 0.45:
            body = despeckle(body)

    img = snap(body, palette)
    if not effects:
        return img
    ring_at = (body_center[0] + p["dx"] * facing, body_center[1] + p["dy"])
    if p["flash"]:
        img = flash_white(img)
    if p["ring"] >= 0 and not spinning:
        img, _drawn = roar_rings(img, ring_at, p["ring"])
    if p["burst"]:
        img, _drawn = impact_burst(img, body_center, p["burst"])
    if p["sparks"]:
        img, _drawn = sparks(img, f, body_center)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (STILL_DIR / "front.png").read_bytes()
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
    return master


def preview_image(sheets):
    samples = {"idle": (0, 4, 8, 12), "walk": (0, 4, 8, 12), "attack": (2, 6, 8, 11), "hurt": (1, 3, 6, 10)}
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            m = tile[..., 3] >= 128
            bg[ai * CELL_H:(ai + 1) * CELL_H, si * CELL_W:(si + 1) * CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 4
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            opaque = tile[..., 3] >= 128
            if not np.any(opaque):
                continue
            ys, xs = np.nonzero(opaque)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(CELL_H, int(ys.max()) + 1 + pad)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(CELL_W, int(xs.max()) + 1 + pad)
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


SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    return n


def south_attack_ratios(sheets):
    row = sheets["attack"][:CELL_H]
    counts = []
    for f in range(16):
        tile = row[:, f * CELL_W:(f + 1) * CELL_W]
        counts.append(int((tile[..., 3] == 255).sum()))
    base = counts[0]
    return counts, [c / base for c in counts]


def _entry_span(text, key):
    token = f'"{key}":'
    start = text.index(token)
    i = start + len(token)
    while text[i] != "{":
        i += 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return start, j + 1
    raise SystemExit(f"dir8 entry did not close: {key}")


def publish_dir8(sprite_h):
    entry = {
        "dex": 429,
        "form": "normal",
        "variant": "mismagius",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0429_mismagius/normal",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir",
        "nf": 16,
        "cw": CELL_W,
        "ch": CELL_H,
        "w": CELL_W,
        "h": int(sprite_h),
        "fy": int(FOOT_Y),
        "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1,
        "src": "agent-sprite-forge",
        "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    DIR8_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with DIR8_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            text = DIR8.read_text()
            prefix = "window.ARENA_DIR8 = "
            if not text.startswith(prefix):
                raise SystemExit("dir8.js header changed; not publishing")
            keys = re.findall(r'"(\d{4}_[^"]+)":\{', text)
            if keys.count(KEY) != 1:
                raise SystemExit("0429_mismagius key missing or duplicated")
            idx = keys.index(KEY)
            neighbor = keys[idx + 1] if idx + 1 < len(keys) else keys[idx - 1]
            other = keys[idx - 1] if idx > 0 else neighbor
            if other != "0428_lopunny-mega":
                raise SystemExit(f"unexpected dir8 neighbor {other}")
            nb_s, nb_e = _entry_span(text, neighbor)
            ot_s, ot_e = _entry_span(text, other)
            neighbor_blob = text[nb_s:nb_e]
            other_blob = text[ot_s:ot_e]
            s, e = _entry_span(text, KEY)
            text = text[:s] + f'"{KEY}":' + blob + text[e:]
            if neighbor_blob not in text or other_blob not in text:
                raise SystemExit("neighbor dir8 key changed")
            if text.count('"0428_lopunny-mega":') != 1:
                raise SystemExit("lopunny-mega key touched")
            body = text[len(prefix):].strip()
            if body.endswith(";"):
                body = body[:-1].strip()
            json.loads(body)
            if text.count(f'"v":"{VERSION}"') != 1:
                raise SystemExit("dir8 version stamp collided")
            if "\n" in text.strip():
                raise SystemExit("dir8.js would no longer be one line")
            DIR8.write_text(text if text.endswith("\n") else text + "\n")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return neighbor


def update_queue(status):
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            text = QUEUE.read_text()
            data = json.loads(text)
            before = {item.get("id"): (item.get("status"), item.get("claim")) for item in data["items"]}
            start = text.index('"id": "0429_mismagius"')
            nxt = text.find('"id":', start + 10)
            end = len(text) if nxt < 0 else nxt
            block = text[start:end]
            if '"status":' not in block or '"claim": "parallel-batch"' not in block and status == "done":
                # claim may already be gone only if we are rewriting a failed pass; require it on first publish
                if '"claim": "parallel-batch"' not in block and before.get(KEY, (None, None))[1] not in (None, "parallel-batch"):
                    raise SystemExit("queue claim is not parallel-batch")
            if '"status":' not in block:
                raise SystemExit("queue status missing")
            block2 = re.sub(r'"status": "[^"]*"', f'"status": "{status}"', block, count=1)
            mid, n_mid = re.subn(r'\n\s*"claim": "[^"]*",', "", block2, count=1)
            if n_mid:
                block2 = mid
            else:
                block2 = re.sub(r',\s*\n\s*"claim": "[^"]*"', "", block2, count=1)
            text = text[:start] + block2 + text[end:]
            data = json.loads(text)
            ids = [item.get("id") for item in data["items"]]
            if ids.count(KEY) != 1:
                raise SystemExit("queue id collision")
            for item in data["items"]:
                if item.get("id") == KEY:
                    if item.get("status") != status or "claim" in item:
                        raise SystemExit("queue update did not stick")
                elif (item.get("status"), item.get("claim")) != before.get(item.get("id")):
                    raise SystemExit(f"queue neighbor changed: {item.get('id')}")
            QUEUE.write_text(text if text.endswith("\n") else text + "\n")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(STILL_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    if set(np.unique(front[..., 3]).tolist()) - {0, 255}:
        raise SystemExit("still alpha is not only 0 or 255")
    palette = extract_palette(front)
    masks = segment(front)
    face_info = measure_face(front, masks)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("facing viewer", face_info, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}

    rest = synth(rigs["S"], "idle", 0, palette, effects=False)
    placed = rigs["S"][6]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "body": (90, 70, 140),
        "brim_l": (80, 180, 255),
        "brim_r": (255, 180, 40),
        "hair_l": (255, 80, 160),
        "hair_r": (255, 240, 80),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "S", f) for f in (0, 4, 8, 12)]
        want += [("walk", "S", f) for f in (0, 2, 4, 6, 8, 10)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in (0, 1, 2, 3, 6, 9)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        base = int((synth(rigs["S"], "attack", 0, palette)[..., 3] == 255).sum())
        print("ATTACK_S_RATIOS", flush=True)
        ratios = []
        for f in range(16):
            tile = synth(rigs["S"], "attack", f, palette)
            n = int((tile[..., 3] == 255).sum())
            frac, _total = largest_fraction(tile)
            ratio = n / base
            ratios.append(ratio)
            edge = tile[..., 3]
            touch = bool(edge[0].any() or edge[-1].any() or edge[:, 0].any() or edge[:, -1].any())
            print(f"attack S {f:02d} opaque {n} ratio {ratio:.3f} main {frac:.3f} edge {touch}", flush=True)
        print(f"min {min(ratios):.3f} max {max(ratios):.3f}", flush=True)
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
    (STILL_DIR / "actions").mkdir(parents=True, exist_ok=True)
    preview.save(STILL_DIR / "actions" / "preview.png")
    contact = motion_contact(sheets)
    contact.save(OUT_DIR / "preview_motion.png")
    contact.save(work / "review_S.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    counts_s, ratios = south_attack_ratios(sheets)
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(counts_s, ratios)):
        print(f"attack S {f:02d} opaque {c} ratio {r:.3f}", flush=True)
    print(f"min {min(ratios):.3f} max {max(ratios):.3f} h {sprite_h} cw {CELL_W} ch {CELL_H} fy {FOOT_Y}", flush=True)

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] == 255
    partial = (master[..., 3] != 0) & (master[..., 3] != 255)
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)} if opaque.any() else set()
    off = sorted(used - pal)
    alphas = sorted({int(v) for v in np.unique(master[..., 3])})
    clip = {}
    spin_stats = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and d == "S" and 5 <= f <= 10:
                    frac, total_px = largest_fraction(tile)
                    spin_stats.append({"f": f, "opaque": total_px, "main": round(frac, 3)})
    bad_ratio = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": alphas, "partial_alpha": int(partial.sum()),
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "attack_s_ratios": [round(r, 3) for r in ratios],
        "attack_s_min": round(min(ratios), 3), "attack_s_max": round(max(ratios), 3),
        "spin_body": spin_stats, "rest_diff": 0,
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = sys.executable
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
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1800], flush=True)

    bad_spin = [s for s in spin_stats if s["main"] < 0.90]
    failed = bool(off or clip or partial.any() or alphas != [0, 255] or bad_ratio or bad_spin)
    if failed:
        if not args.no_publish:
            update_queue("needs_fix")
        raise SystemExit(f"qa failed off={len(off)} clip={clip} alphas={alphas} ratios={bad_ratio} spin={bad_spin}")

    meta = {
        "generator": "forge_mismagius_actions", "version": VERSION, "key": KEY,
        "dex": 429, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "still_faces": "viewer",
        "face": face_info, "parts": counts, "feet": 0, "mouth": 0,
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "attack_s_min": round(min(ratios), 3), "attack_s_max": round(max(ratios), 3),
        "attack_s_ratios": [round(r, 3) for r in ratios],
        "source": "front.png 96x96", "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.no_publish:
        print("no-publish", flush=True)
        return
    copied = 0 if args.no_sync else sync_outputs()
    neighbor = publish_dir8(sprite_h)
    update_queue("done")
    arena_sheet = SYNC_OUT / "sheet.png"
    arena_bytes = arena_sheet.stat().st_size if arena_sheet.is_file() else 0
    print(f"synced {copied} neighbor {neighbor} arena_sheet_bytes {arena_bytes}", flush=True)


if __name__ == "__main__":
    main()
