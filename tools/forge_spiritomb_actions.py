#!/usr/bin/env python3
"""Spiritomb (0442) 8-direction action sheet from this folder's 96x96 front.png.

Measured on this still only (opaque 1908, bbox x20-74 y22-73, 55x52, 15 colours,
alpha 0/255, one 4-connected component, centroid 45.89, 47.46). It faces the
viewer. Twenty yellow pixels are spirit eyes inside the purple swirl, with green
faces around them. There is no separate jaw and no feet. The grey-brown odd
keystone sits at the bottom (stone colours plus the black outline from y63 and
the plum collar from y57). The keystone and the swirling spirits are one body.

Parts, from this still only:
  seat   the keystone and the collar it sits in. It does not walk.
  cloud  the purple swirl and the spirit faces. It hinges on the neck.

idle   : keystone planted on the foot line. The swirl rocks a few degrees.
walk   : one-piece hover bob. Not a stomp. No legs.
attack : 0-4 swell the swirl and flare rings (no new mouth),
         5-10 one-piece in-plane 360 spin of that frame-0 pose (no |cos| squash),
         11-12 impact, 13-15 recover.
hurt   : the whole body leans at most 38 degrees.

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
STILL_DIR = ROOT / "monsterSprite/gen4/0442_spiritomb/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0442_spiritomb/normal/agent-sprite-forge"
)
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
VERSION = "spiritomb_v2"
KEY = "0442_spiritomb"
UNFLIPPED = {"S", "SE", "E", "NE"}
MAX_RECOIL = 38.0

LIGHT = (197, 115, 189)
MID = (164, 82, 156)
GREEN = (98, 180, 65)
PLUM = (115, 74, 123)
DGREEN = (49, 98, 74)
STONE = (131, 106, 98)
INK = (16, 16, 16)
LIME = (189, 213, 65)
PALE = (180, 156, 148)
PURPLE = (82, 65, 106)
DUSK = (82, 65, 82)
YELLOW = (255, 222, 16)
MGREEN = (74, 139, 65)
TAN = (156, 131, 106)
VDGREEN = (49, 65, 49)
STONE_COLORS = (PURPLE, DUSK, PALE, TAN, STONE)

# Frame 0 is the still plus one ring, so the spun body stays near that count.
SWELL = [0.00, 0.018, 0.028, 0.032, 0.020, 0, 0, 0, 0, 0, 0, 0, 0, 0.012, 0.0, 0.0]
TURN = [0.0, 3.0, 6.0, 8.0, 4.0, 0, 0, 0, 0, 0, 0, 0, 0, 2.0, 0.0, 0.0]
RING = [0, 1, 2, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


def is_color(img, color):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(color, np.uint8), axis=-1)


def component_stats(mask):
    seen = np.zeros_like(mask, dtype=bool)
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
    """Keystone seat and one spirit cloud. No legs and no invented mouth."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    seat = np.zeros_like(op)
    for color in STONE_COLORS:
        seat |= is_color(img, color)
    seat |= is_color(img, INK) & (yy >= 63)
    seat |= is_color(img, PLUM) & (yy >= 57)
    seat &= op
    cloud = op & ~seat
    if int(seat.sum()) + int(cloud.sum()) != int(op.sum()):
        raise SystemExit("keystone and swirl do not partition the still")
    if int(seat.sum()) < 180 or int(cloud.sum()) < 1400:
        raise SystemExit(f"seat/cloud split looks wrong: seat {int(seat.sum())} cloud {int(cloud.sum())}")
    n_cloud, best_cloud = component_stats(cloud)
    if best_cloud < int(cloud.sum()) * 0.97:
        raise SystemExit(f"swirl split before motion: {n_cloud} largest {best_cloud}")
    ys = np.nonzero(op)[0]
    if int(ys.max()) > 74:
        raise SystemExit("pixels below the keystone; a foot may have been invented")
    return {"seat": seat, "cloud": cloud}


def measure_face(img, masks):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    yellow = is_color(img, YELLOW)
    if int(yellow.sum()) < 16:
        raise SystemExit("spirit eyes missing; facing cannot be fixed")
    # Yellow sits in the swirl, not on a jaw. Nothing below the keystone.
    return {
        "spirit_eyes": int(yellow.sum()),
        "green_faces": int(is_color(img, GREEN).sum()),
        "mouth": 0,
        "feet": 0,
        "centroid_x": round(float(xs.mean()), 2),
        "centroid_y": round(float(ys.mean()), 2),
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


def _resample(layer, src_x, src_y):
    h, w = layer.shape[:2]
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(layer)
    vy, vx = np.nonzero(valid)
    out[vy, vx] = layer[src_y[vy, vx], src_x[vy, vx]]
    return out


def _xform_nearest(layer, pivot, map_xy):
    """4x nearest resample. map_xy(rx, ry) -> (src_x, src_y) in the upscaled grid."""
    if not np.any(layer[..., 3] >= 128):
        return layer.copy()
    scale = 4
    up = np.repeat(np.repeat(layer, scale, axis=0), scale, axis=1)
    px, py = pivot[0] * scale + 1.5, pivot[1] * scale + 1.5
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    src_x, src_y = map_xy(xx - px, yy - py, px, py)
    src_x = np.rint(src_x).astype(np.int32)
    src_y = np.rint(src_y).astype(np.int32)
    out = _resample(up, src_x, src_y)
    return out[scale // 2::scale, scale // 2::scale].copy()


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest in-plane rotate. A multiple of 360 stays exact. No squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    rad = math.radians(ang)
    c, s = math.cos(rad), math.sin(rad)

    def map_xy(rx, ry, px, py):
        return px + c * rx + s * ry, py - s * rx + c * ry

    return _xform_nearest(layer, pivot, map_xy)


def scale_nearest(layer, scale, pivot):
    """4x nearest swell around a pivot. Scale 1 stays exact. No axis squash."""
    if abs(scale - 1.0) < 1e-4 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    inv = 1.0 / scale

    def map_xy(rx, ry, px, py):
        return px + rx * inv, py + ry * inv

    return _xform_nearest(layer, pivot, map_xy)


def despeckle(img):
    """Drop only pixels whose four neighbours are all empty. After a warp only."""
    out = img.copy()
    a = out[..., 3] >= 128
    pad = np.pad(a, 1, constant_values=False)
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


def neck_pivot(cloud, seat):
    """Contact along the top of the keystone, so the swirl hinges instead of orbiting."""
    pad = np.pad(seat, 1, constant_values=False)
    touch = cloud & (pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:])
    ys, xs = np.nonzero(touch)
    if len(xs) < 4:
        raise SystemExit("swirl does not touch the keystone")
    top = int(ys.min())
    band = ys <= top + 5
    return float(xs[band].mean()), float(ys[band].mean())


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
    cys, cxs = np.nonzero(placed["cloud"])
    cloud_center = (float(cxs.mean()), float(cys.mean()))
    neck = neck_pivot(placed["cloud"], placed["seat"])
    # Joint cushion under the swirl, covered at rest, so a swell does not open a hole.
    yy = np.arange(CELL_H)[:, None]
    xx = np.arange(CELL_W)[None, :]
    root = placed["cloud"] & ((yy - neck[1]) ** 2 + (xx - neck[0]) ** 2 <= 5.0 ** 2)
    joint = nearest_color(canvas, root, placed["cloud"])
    layers["joint"] = joint
    facing = -1.0 if flip else 1.0
    pivots = {"neck": neck, "cloud": cloud_center, "body": center}
    return layers, pivots, center, facing, canvas


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, sway=0.0, swell=0.0, spin=0.0, recoil=0.0,
        ring=-1, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["sway"] = wave(f, n, 4.5)
    elif action == "walk":
        p["sway"] = wave(f, n, 6.5, cycles=2.0)
        p["dy"] = wave(f, n, -2.0, cycles=2.0)
    elif action == "attack":
        p["swell"] = SWELL[f]
        p["sway"] = TURN[f]
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
        p["flash"] = 1 if f == 2 else 0
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def compose(layers, pivots, p):
    """Keystone stays rigid. The swirl swells and rocks from the neck, then both bob."""
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    cloud = layers["cloud"]
    if abs(p["swell"]) > 1e-4:
        cloud = scale_nearest(cloud, 1.0 + p["swell"], pivots["cloud"])
    if abs(p["sway"]) > 0.45:
        cloud = rotate_nearest(cloud, p["sway"], pivots["neck"])
    cloud = shift(cloud, p["dx"], p["dy"])
    seat = shift(layers["seat"], p["dx"], p["dy"])
    joint = shift(layers["joint"], p["dx"], p["dy"])
    paste(body, joint)
    paste(body, cloud)
    paste(body, seat)
    return body


def roar_rings(out, center, age):
    """Sparse rings outside the swirl. Age 0 is one ring."""
    cx, cy = center
    cols = (YELLOW, LIME, LIGHT)
    radii = (38, 44, 49)
    steps = (30, 26, 24)
    drawn = 0
    for ri in range(age + 1):
        rad = radii[ri]
        step = steps[ri]
        col = cols[ri]
        deg = ri * 8
        while deg < 360 + ri * 8:
            ang = math.radians(deg)
            drawn += put(out, cx + rad * math.cos(ang), cy - 6 + rad * math.sin(ang) * 0.86, col)
            deg += step
    return out, drawn


def impact_burst(out, center, size):
    cx, cy = center
    ticks = [(0, -42), (20, -34), (-20, -34), (34, -12), (-34, -12), (16, 22), (-16, 22), (0, 30), (28, 8), (-28, 8)]
    col = YELLOW if size > 1 else LIME
    drawn = 0
    for dx, dy in ticks[: 6 + size * 2]:
        drawn += put(out, cx + dx, cy + dy, col)
        if size > 1:
            drawn += put(out, cx + dx, cy + dy - 1, LIGHT)
    return out, drawn


def sparks(out, frame_idx, center):
    cx, cy = center
    d = frame_idx
    pts = [(cx - 24 - d, cy - 28), (cx + 24 + d, cy - 26), (cx, cy - 40 - d)]
    drawn = 0
    for sx, sy in pts:
        drawn += put(out, sx, sy, YELLOW)
        drawn += put(out, sx, sy + 1, LIME)
    return out, drawn


def flash_swirl(img, seat_hint):
    """Yellow flash on the swirl only. The keystone and the ink stay."""
    out = img.copy()
    opaque = out[..., 3] >= 128
    ink = opaque & np.all(out[..., :3] == np.array(INK, np.uint8), axis=-1)
    stone = np.zeros(opaque.shape, bool)
    for color in STONE_COLORS:
        stone |= opaque & np.all(out[..., :3] == np.array(color, np.uint8), axis=-1)
    # seat_hint is unused once colours identify the stone; kept so the call site stays local.
    del seat_hint
    out[opaque & ~ink & ~stone, :3] = np.array(YELLOW, np.uint8)
    return out


def largest_fraction(img):
    opaque = img[..., 3] >= 128
    if not np.any(opaque):
        return 1.0, 0
    seen = np.zeros_like(opaque, dtype=bool)
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
    layers, pivots, center, facing, _canvas = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        # One piece. Spin or lean the frame-0 pose so the spirits cannot fly off.
        base = pose(action, 0)
        base = gen4_attack_strike.finish(base, action, 0, __file__)

        base["sway"] = base["sway"] * facing
        base["dx"] = base["dx"] * facing
        body = compose(layers, pivots, base)
        if recoiling:
            body = shift(body, p["dx"] * facing, p["dy"])
            pivot = (center[0] + p["dx"] * facing, center[1] + p["dy"])
            body = rotate_nearest(body, -p["recoil"] * facing, pivot)
        else:
            body = rotate_nearest(body, p["spin"] * facing, center)
        body = despeckle(body)
    else:
        moved = dict(p)
        moved["sway"] = p["sway"] * facing
        moved["dx"] = p["dx"] * facing
        body = compose(layers, pivots, moved)
        if abs(p["sway"]) > 0.45 or abs(p["swell"]) > 1e-4 or abs(p["dy"]) > 0.45 or abs(p["dx"]) > 0.45:
            body = despeckle(body)

    img = snap(body, palette)
    if not effects:
        return img
    ring_at = (center[0] + p["dx"] * facing, center[1] + p["dy"])
    if p["flash"]:
        img = flash_swirl(img, None)
    if p["ring"] >= 0 and not spinning:
        img, _drawn = roar_rings(img, ring_at, p["ring"])
    if p["burst"]:
        img, _drawn = impact_burst(img, center, p["burst"])
    if p["sparks"]:
        img, _drawn = sparks(img, f, center)
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
        "dex": 442,
        "form": "normal",
        "variant": "spiritomb",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0442_spiritomb/normal",
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
            if "\n" in text.strip():
                raise SystemExit("dir8.js is no longer one line")
            keys = re.findall(r'"(\d{4}_[^"]+)":\{', text)
            if keys.count(KEY) != 1:
                raise SystemExit("0442_spiritomb key missing or duplicated")
            idx = keys.index(KEY)
            prev_key = keys[idx - 1]
            next_key = keys[idx + 1]
            prev_s, prev_e = _entry_span(text, prev_key)
            next_s, next_e = _entry_span(text, next_key)
            prev_blob = text[prev_s:prev_e]
            next_blob = text[next_s:next_e]
            s, e = _entry_span(text, KEY)
            new_text = text[:s] + f'"{KEY}":' + blob + text[e:]
            if prev_blob not in new_text or next_blob not in new_text:
                raise SystemExit("neighbor dir8 key changed")
            if new_text.count(f'"{prev_key}":') != 1 or new_text.count(f'"{next_key}":') != 1:
                raise SystemExit("neighbor key count changed")
            body = new_text[len(prefix):].strip()
            if body.endswith(";"):
                body = body[:-1].strip()
            data = json.loads(body)
            if list(data.keys())[idx] != KEY:
                raise SystemExit("dir8 key order changed")
            if data[KEY] != entry:
                raise SystemExit("dir8 entry did not round-trip")
            if new_text.count(f'"v":"{VERSION}"') != 1:
                raise SystemExit("dir8 version stamp collided")
            if "\n" in new_text.strip():
                raise SystemExit("dir8.js would no longer be one line")
            # Keep the file's existing newline style.
            if text.endswith("\n") and not new_text.endswith("\n"):
                new_text += "\n"
            if (not text.endswith("\n")) and new_text.endswith("\n"):
                new_text = new_text[:-1]
            DIR8.write_text(new_text)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return prev_key, next_key


def update_queue(status):
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            text = QUEUE.read_text()
            data = json.loads(text)
            before = {item.get("id"): (item.get("status"), item.get("claim")) for item in data["items"]}
            if before.get(KEY, (None, None))[1] != "parallel-batch" and status == "done":
                if before.get(KEY, (None, None))[0] != "in_progress":
                    raise SystemExit(f"queue claim is not parallel-batch: {before.get(KEY)}")
            start = text.index('"id": "0442_spiritomb"')
            nxt = text.find('"id":', start + 10)
            end = len(text) if nxt < 0 else nxt
            block = text[start:end]
            if '"status":' not in block:
                raise SystemExit("queue status missing")
            block2 = re.sub(r'"status": "[^"]*"', f'"status": "{status}"', block, count=1)
            mid, n_mid = re.subn(r',\n\s*"claim": "[^"]*"', "", block2, count=1)
            if not n_mid:
                mid, n_mid = re.subn(r'\n\s*"claim": "[^"]*",', "", block2, count=1)
            if n_mid:
                block2 = mid
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
            if not text.endswith("\n"):
                text += "\n"
            QUEUE.write_text(text)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def attack_report(rig, palette):
    base_img = synth(rig, "attack", 0, palette)
    base = int((base_img[..., 3] == 255).sum())
    print("ATTACK_S_RATIOS", flush=True)
    ratios = []
    rows = []
    for f in range(16):
        tile = synth(rig, "attack", f, palette)
        n = int((tile[..., 3] == 255).sum())
        frac, _total = largest_fraction(tile)
        ratio = n / base
        ratios.append(ratio)
        edge = tile[..., 3]
        touch = bool(edge[0].any() or edge[-1].any() or edge[:, 0].any() or edge[:, -1].any())
        line = f"attack S {f:02d} opaque {n} ratio {ratio:.3f} main {frac:.3f} edge {touch}"
        print(line, flush=True)
        rows.append({"f": f, "opaque": n, "ratio": ratio, "main": frac, "edge": touch})
    print(f"min {min(ratios):.3f} max {max(ratios):.3f}", flush=True)
    return ratios, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front_path = STILL_DIR / "front.png"
    front_bytes = front_path.read_bytes()
    front = np.array(Image.open(front_path).convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    if set(np.unique(front[..., 3]).tolist()) - {0, 255}:
        raise SystemExit("still alpha is not only 0 or 255")
    palette = extract_palette(front)
    if len(palette) != 15:
        raise SystemExit(f"expected 15 still colours, got {len(palette)}")
    masks = segment(front)
    face_info = measure_face(front, masks)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("facing viewer", face_info, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}

    rest = synth(rigs["S"], "idle", 0, palette, effects=False)
    placed = rigs["S"][4]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part[masks["seat"]] = (180, 156, 148, 255)
    part[masks["cloud"]] = (164, 82, 156, 255)
    # Mark the measured spirit eyes so the part map shows they stayed on the cloud.
    eyes = is_color(front, YELLOW)
    part[eyes] = (255, 222, 16, 255)
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        ratios, _rows = attack_report(rigs["S"], palette)
        want = [("idle", "S", f) for f in (0, 4, 8, 12)]
        want += [("walk", "S", f) for f in (0, 2, 4, 6, 8, 10)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in (0, 1, 2, 3, 6, 9)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        cols = 8
        rows_n = math.ceil(len(tiles) / cols)
        sheet = np.full((rows_n * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        bad = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
        if bad:
            raise SystemExit(f"sample ratios outside 0.90-1.08: {bad}")
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
    if front_path.read_bytes() != front_bytes:
        raise SystemExit("front.png changed during the forge")
    if failed:
        if not args.no_publish:
            update_queue("needs_fix")
        raise SystemExit(f"qa failed off={len(off)} clip={clip} alphas={alphas} ratios={bad_ratio} spin={bad_spin}")

    meta = {
        "generator": "forge_spiritomb_actions", "version": VERSION, "key": KEY,
        "dex": 442, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "still_faces": "viewer",
        "archetype": "keystone_spirit_hover",
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
    neighbors = publish_dir8(sprite_h)
    update_queue("done")
    arena_sheet = SYNC_OUT / "sheet.png"
    arena_bytes = arena_sheet.stat().st_size if arena_sheet.is_file() else 0
    print(f"synced {copied} neighbors {neighbors} arena_sheet_bytes {arena_bytes}", flush=True)


if __name__ == "__main__":
    main()
