#!/usr/bin/env python3
"""Drifblim (0426) 8-direction action sheet from this folder's 96x96 front.png.

Measured on this still only (opaque 1755, bbox x18-77 y19-77, 60x59, 15 colours,
alpha 0/255). The balloon faces the viewer: yellow X at about x34-44 y41-49,
two red cheek dots at y40-43, white cloud on top. Both sides of the balloon
are drawn, so south stays unflipped. N NW W SW mirror.

Arms, counted from the pixels that leave the balloon below y59. Nothing else
is a limb. No legs are drawn.
  arm_l    left ribbon, one piece, three-prong hand (x18-33).
  arm_in   inner right ribbon, the long three-prong hand.
  arm_out  outer right ribbon. It does not touch arm_in.
The red patch under the balloon (about x37-51 y65-71) stays on the body.
It is not a jaw and not feet. There is no mouth, so the roar swells the
whole balloon and draws rings.

idle   : balloon planted. The three ribbons sway. Frame 0 is the still.
walk   : one-piece bob. Ribbons sway. Not a stomp.
attack : 0-4 swell plus rings, 5-10 one-piece in-plane 360 spin,
         11-12 impact, 13-15 settle. No |cos| squash.
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
STILL_DIR = ROOT / "monsterSprite/gen4/0426_drifblim/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0426_drifblim/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 192, 184, 164
VERSION = "drifblim_v2"
KEY = "0426_drifblim"
UNFLIPPED = {"S", "SE", "E", "NE"}
MAX_RECOIL = 38.0
ARMS = ("arm_l", "arm_in", "arm_out")

YELLOW = (255, 222, 8)
WHITE = (255, 255, 255)
RED = (230, 74, 82)
DARK_RED = (148, 57, 90)
GOLD = (189, 156, 0)
PALE = (213, 180, 255)
BROWN = (98, 82, 57)
INK = (16, 16, 16)

# Whole-body swell. Frame 0 and frame 15 are 1 so the loop meets the still.
SWELL = [1.0, 1.010, 1.016, 1.020, 1.012, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# Ring age. -1 draws none.
RING = [-1, 0, 1, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


def is_color(img, color):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(color, np.uint8), axis=-1)


def flood_from(op, block, seeds):
    h, w = op.shape
    seen = np.zeros_like(op)
    stack = []
    for y, x in seeds:
        if op[y, x] and not block[y, x] and not seen[y, x]:
            seen[y, x] = True
            stack.append((int(y), int(x)))
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and op[ny, nx] and not seen[ny, nx] and not block[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    return seen


def component_count(mask):
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
    """Three ribbons below the balloon. The red underside stays on the body."""
    op = img[..., 3] >= 128
    h, w = op.shape
    yy, xx = np.mgrid[0:h, 0:w]
    red = is_color(img, RED) | is_color(img, DARK_RED)
    yellow = is_color(img, YELLOW)
    # Do not climb into the balloon, the cloud, the X, or the red underside.
    block = (yy <= 59) | red
    block |= (yy >= 60) & (xx >= 34) & (xx <= 49)
    block |= ~op

    left_seeds = list(zip(*np.nonzero(yellow & (xx < 34) & (yy > 64))))
    in_seeds = list(zip(*np.nonzero(yellow & (xx >= 52) & (xx <= 66) & (yy >= 70))))
    out_seeds = list(zip(*np.nonzero(yellow & (xx >= 66) & (yy >= 63) & (yy <= 69))))
    if not left_seeds or not in_seeds or not out_seeds:
        raise SystemExit("an arm seed is missing; the still was not read")
    arm_l = flood_from(op, block, left_seeds)
    arm_in = flood_from(op, block, in_seeds)
    arm_out = flood_from(op, block, out_seeds)
    parts = {"arm_l": arm_l, "arm_in": arm_in, "arm_out": arm_out}
    own = np.zeros_like(op)
    overlap = 0
    for name in ARMS:
        overlap += int((own & parts[name]).sum())
        own |= parts[name]
    body = op & ~own
    parts = {"body": body, "arm_l": arm_l, "arm_in": arm_in, "arm_out": arm_out}
    if int(own.sum()) + int(body.sum()) != int(op.sum()) or overlap:
        raise SystemExit(
            f"masks do not partition the still: overlap {overlap} covered {int((own | body).sum())} opaque {int(op.sum())}"
        )
    counts = {k: int(v.sum()) for k, v in parts.items()}
    if not (90 <= counts["arm_l"] <= 180 and 100 <= counts["arm_in"] <= 220 and 60 <= counts["arm_out"] <= 170):
        raise SystemExit(f"arm sizes are not the three ribbons: {counts}")
    if component_count(arm_l)[0] != 1 or component_count(arm_in)[0] != 1 or component_count(arm_out)[0] != 1:
        raise SystemExit("an arm mask split or merged; recount the ribbons")
    x_mark = yellow & (yy >= 40) & (yy <= 52) & (xx >= 33) & (xx <= 46)
    if int((x_mark & ~body).sum()) or int(x_mark.sum()) < 40:
        raise SystemExit("the yellow X left the balloon")
    if int((red & ~body).sum()) or int(red.sum()) < 50:
        raise SystemExit("the red underside left the balloon")
    return parts


def measure_face(img, masks):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    yellow = is_color(img, YELLOW)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    x_mark = yellow & (yy <= 52)
    cheeks = (is_color(img, RED) | is_color(img, DARK_RED)) & (yy <= 50)
    if int(x_mark.sum()) < 40 or int(cheeks.sum()) < 4:
        raise SystemExit("face marks missing; facing cannot be fixed")
    # Both cheeks and the full round outline: this still faces the viewer.
    return {
        "x_mark": int(x_mark.sum()),
        "cheeks": int(cheeks.sum()),
        "arms": 3,
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


def scale_nearest(layer, scale, pivot):
    """4x nearest grow around pivot. Scale 1 stays exact. Not a |cos| squash."""
    if abs(scale - 1.0) < 0.002 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    factor = 4
    up = np.repeat(np.repeat(layer, factor, axis=0), factor, axis=1)
    px, py = pivot[0] * factor + 1.5, pivot[1] * factor + 1.5
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    src_x = np.rint(px + (xx - px) / scale).astype(np.int32)
    src_y = np.rint(py + (yy - py) / scale).astype(np.int32)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    out[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    return out[factor // 2::factor, factor // 2::factor].copy()


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


def hang_pivot(mask, center_x):
    """Top of a ribbon, on the edge that touches the balloon."""
    ys, xs = np.nonzero(mask)
    top = int(ys.min())
    sel = ys <= top + 1
    inward = 1.0 if float(xs.mean()) < center_x else -1.0
    x = float(xs[sel].max() if inward > 0 else xs[sel].min())
    return (x, float(ys[sel].mean())), inward


def build_rig(img, masks, flip):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
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

    # Body colour under each ribbon socket, hidden while the ribbon is down.
    yy = np.arange(CELL_H)[:, None]
    for name in ARMS:
        pm = placed[name]
        top = int(np.nonzero(pm)[0].min())
        band = pm & (yy <= top + 2)
        fill = nearest_color(canvas, band, placed["body"])
        write = band & (layers["body"][..., 3] < 128)
        layers["body"][write] = fill[write]

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
    for name in ARMS:
        pivots[name], _inward = hang_pivot(placed[name], center[0])
    facing = -1.0 if flip else 1.0
    return layers, pivots, center, body_center, facing, canvas


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(dx=0.0, dy=0.0, sway=0.0, swell=1.0, spin=0.0, recoil=0.0, ring=-1, burst=0, flash=0, sparks=0)
    if action == "idle":
        # Planted. Only the ribbons move. Frame 0 is the still.
        p["sway"] = wave(f, n, 6.0)
    elif action == "walk":
        p["sway"] = wave(f, n, 8.0, cycles=2.0)
        p["dy"] = wave(f, n, -2.5, cycles=2.0)
    elif action == "attack":
        p["swell"] = SWELL[f]
        p["ring"] = RING[f]
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
        if f in (11, 12):
            p["burst"] = 2 if f == 11 else 1
        if f in (13, 14):
            p["dy"] = -1.0 if f == 13 else 0.0
    else:
        env = [0.0, 0.55, 1.0, 0.82, 0.62, 0.42, 0.24, 0.12, 0.04, 0.0, 0.0, 0.0][f]
        p["recoil"] = MAX_RECOIL * env
        p["dx"] = -4.0 * env
        p["dy"] = -1.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def compose(layers, pivots, p):
    """Balloon stays one piece. Ribbons swing from their sockets."""
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(body, shift(layers["body"], p["dx"], p["dy"]))
    for name in ARMS:
        spun = rotate_nearest(layers[name], p["sway"], pivots[name])
        paste(body, shift(spun, p["dx"], p["dy"]))
    return body


def roar_rings(out, center, age):
    """Sparse rings outside the balloon. Age 0 is one ring, 2 is three."""
    cx, cy = center
    cols = (YELLOW, WHITE, RED)
    radii = (36, 41, 46)
    steps = (30, 26, 24)
    for ri in range(age + 1):
        rad = radii[ri]
        step = steps[ri]
        col = cols[ri]
        deg = 0
        while deg < 360:
            ang = math.radians(deg)
            put(out, cx + rad * math.cos(ang), cy + rad * math.sin(ang), col)
            deg += step
    return out


def impact_burst(out, center, size):
    cx, cy = center
    # Upper ticks, clear of the hanging hands. Fixed count when they land on air.
    ticks = [(0, -38), (16, -34), (-16, -34), (28, -22), (-28, -22), (34, -8), (-34, -8), (10, -42), (-10, -42)]
    col = YELLOW if size > 1 else WHITE
    for dx, dy in ticks[: 6 + size]:
        put(out, cx + dx, cy + dy, col)
        if size > 1:
            put(out, cx + dx, cy + dy - 1, GOLD)
    return out


def sparks(out, frame_idx, center):
    cx, cy = center
    d = frame_idx
    pts = [(cx - 18 - d, cy - 28), (cx + 18 + d, cy - 26), (cx, cy - 36 - d)]
    for sx, sy in pts:
        put(out, sx, sy, YELLOW)
        put(out, sx, sy + 1, WHITE)
    return out


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
    layers, pivots, center, body_center, facing, _canvas = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    swelling = p["swell"] > 1.002
    if spinning or recoiling or swelling:
        rest = pose(action, 0)
        rest = gen4_attack_strike.finish(rest, action, 0, __file__)

        body = compose(layers, pivots, rest)
        if recoiling:
            body = shift(body, p["dx"] * facing, p["dy"])
            pivot = (center[0] + p["dx"] * facing, center[1] + p["dy"])
            body = rotate_nearest(body, -p["recoil"] * facing, pivot)
        elif spinning:
            body = rotate_nearest(body, p["spin"] * facing, center)
        else:
            body = scale_nearest(body, p["swell"], center)
        body = despeckle(body)
    else:
        moved = dict(p)
        moved["dx"] = p["dx"] * facing
        body = compose(layers, pivots, moved)
        if abs(p["sway"]) > 0.45:
            body = despeckle(body)

    img = snap(body, palette)
    if not effects:
        return img
    ring_at = (body_center[0], body_center[1] + p["dy"])
    if p["flash"]:
        img = flash_white(img)
    if p["ring"] >= 0 and not spinning:
        img = roar_rings(img, ring_at, p["ring"])
    if p["burst"]:
        img = impact_burst(img, body_center, p["burst"])
    if p["sparks"]:
        img = sparks(img, f, body_center)
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
    Image.fromarray(master, "RGBA").save(OUT_DIR / "all_8dir_actions.png")
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
    "all_8dir_actions.png",
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
        counts.append(int((tile[..., 3] >= 128).sum()))
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
        "dex": 426,
        "form": "normal",
        "variant": "drifblim",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0426_drifblim/normal",
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
                raise SystemExit("0426_drifblim key missing or duplicated")
            idx = keys.index(KEY)
            neighbor = keys[idx + 1] if idx + 1 < len(keys) else keys[idx - 1]
            other = keys[idx - 1] if idx > 0 else neighbor
            nb_s, nb_e = _entry_span(text, neighbor)
            ot_s, ot_e = _entry_span(text, other)
            neighbor_blob = text[nb_s:nb_e]
            other_blob = text[ot_s:ot_e]
            s, e = _entry_span(text, KEY)
            text = text[:s] + f'"{KEY}":' + blob + text[e:]
            if neighbor_blob not in text or other_blob not in text:
                raise SystemExit("neighbor dir8 key changed")
            body = text[len(prefix):].strip()
            if body.endswith(";"):
                body = body[:-1].strip()
            json.loads(body)
            if text.count(f'"v":"{VERSION}"') != 1:
                raise SystemExit("dir8 version stamp collided")
            DIR8.write_text(text)
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
            start = text.index('"id": "0426_drifblim"')
            nxt = text.find('"id":', start + 10)
            end = len(text) if nxt < 0 else nxt
            block = text[start:end]
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
    palette = extract_palette(front)
    masks = segment(front)
    face_info = measure_face(front, masks)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("facing viewer", face_info, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}

    rest = synth(rigs["S"], "idle", 0, palette)
    placed = rigs["S"][5]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "body": (150, 110, 190),
        "arm_l": (255, 210, 40),
        "arm_in": (80, 200, 120),
        "arm_out": (80, 140, 255),
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
        base = int((synth(rigs["S"], "attack", 0, palette)[..., 3] >= 128).sum())
        print("ATTACK_S_RATIOS", flush=True)
        ratios = []
        for f in range(16):
            tile = synth(rigs["S"], "attack", f, palette)
            n = int((tile[..., 3] >= 128).sum())
            frac, _total = largest_fraction(tile)
            ratio = n / base
            ratios.append(ratio)
            print(f"attack S {f:02d} opaque {n} ratio {ratio:.3f} main {frac:.3f}", flush=True)
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
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
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
        "off_palette_colors": len(off), "alphas": alphas,
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
    failed = bool(off or clip or alphas != [0, 255] or bad_ratio or bad_spin)
    if failed:
        if not args.no_publish:
            update_queue("needs_fix")
        raise SystemExit(f"qa failed off={len(off)} clip={clip} alphas={alphas} ratios={bad_ratio} spin={bad_spin}")

    meta = {
        "generator": "forge_drifblim_actions", "version": VERSION, "key": KEY,
        "dex": 426, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "still_faces": "viewer",
        "face": face_info, "parts": counts, "arms": 3,
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
