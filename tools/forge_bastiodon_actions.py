#!/usr/bin/env python3
"""Bastiodon (0411) 8-direction action sheet from this front.png only.

Measured on normal/front.png (96x96, alpha already 0 or 255, 13 colours):
  opaque bbox x14-81 y20-76 (68x57). Centroid about (44.1, 50.2). 2555 pixels.
  Faces left. Four yellow eye-spots sit on the shield at x23-44 y32-46.
  The gold belly is to the right of the shield (x51-72). The tail tip is the
  rear nub x77-81 y50-60 and stays on the body. The face plate is the head.
  Four feet actually reach the ground: x17-30 sole y74, x36-45 sole y74,
  x50-61 sole y76 (claws), x67-70 sole y72. No fifth leg.
  A jaw exists: red mouth pixels and the lip directly under them. Roar drops
  that lip and draws rings toward screen-left. Eyes and the shield stay put.

Idle frame 0 is the still. Walk is a 4-beat diagonal trot on those feet.
Attack is roar (0-4), one-piece 360 spin (5-10), impact (11-12), recover (13-15).
Hurt recoils at most 38 degrees, no |cos| squash. front/back png and gif are not written.
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
SPECIES = ROOT / "monsterSprite/gen4/0411_bastiodon/normal"
OUT_DIR = SPECIES / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0411_bastiodon/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left, so east-side rows are the mirror. E looks screen-right.
FACE_RIGHT = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 184, 176, 152
VERSION = "bastiodon_v2"
KEY = "0411_bastiodon"
NEIGHBOR = "0412_burmy"

STEEL = (98, 98, 106)
DARK = (65, 57, 74)
BLACK = (16, 16, 16)
LIGHT = (213, 213, 213)
GOLD = (255, 205, 123)
MID = (139, 139, 156)
VDARK = (41, 41, 49)
YELLOW = (230, 205, 65)
NEAR = (32, 32, 32)
TAN = (197, 156, 90)
OLIVE = (131, 123, 90)
BROWN = (115, 90, 57)
RED = (180, 57, 74)

FOOTS = ("fl", "mid", "hr", "far")


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
    """4x nearest. Positive degrees turn clockwise on screen. Exact colours."""
    if abs(angle_deg) % 360 < 0.01 or not np.any(layer[..., 3] >= 128):
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
    """Drop only pixels whose 4-neighbourhood is empty. Keeps eyes and outlines."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        q = deque([(y, x)])
        seen[y, x] = True
        pts = []
        while q:
            cy, cx = q.popleft()
            pts.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        out.append(np.array(pts))
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


def fill_pinholes(img, max_size=8):
    a = img[..., 3] >= 128
    empty = ~a
    h, w = a.shape
    seen = np.zeros_like(empty)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if empty[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if empty[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and empty[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    holes = empty & ~seen
    if not np.any(holes):
        return img
    out = img.copy()
    for pts in components(holes):
        if len(pts) > max_size:
            continue
        mask = np.zeros_like(holes)
        mask[pts[:, 0], pts[:, 1]] = True
        fill = nearest_color(out, mask, a)
        out[mask] = fill[mask]
    return out


def bridge_parts(img, max_gap=2):
    a = img[..., 3] >= 128
    parts = components(a)
    if len(parts) <= 1:
        return img
    parts.sort(key=len, reverse=True)
    main = np.zeros_like(a)
    main[parts[0][:, 0], parts[0][:, 1]] = True
    out = img.copy()
    changed = False
    for pts in parts[1:]:
        gap_lim = 3 if len(pts) > 80 else max_gap
        frag = np.zeros_like(a)
        frag[pts[:, 0], pts[:, 1]] = True
        edge = frag.copy()
        grown = None
        for _ in range(gap_lim):
            p = np.pad(edge, 1)
            dil = edge | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
            if np.any(dil & main):
                grown = dil & ~a
                break
            edge = dil & ~main
        else:
            continue
        if grown is None or not np.any(grown):
            main = main | frag
            continue
        fill = nearest_color(out, grown, out[..., 3] >= 128)
        out[grown] = fill[grown]
        changed = True
        main = main | grown | frag
        a = out[..., 3] >= 128
    return out if changed else img


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


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def is_col(img, colour):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(colour, np.uint8), axis=-1)


def segment(img):
    """Feet are the four ground columns. The shield, eyes, belly, and tail stay on the body."""
    h, w = img.shape[:2]
    opaque = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    # Gaps measured at y70 split the soles. Do not flood up into the belly.
    bounds = {
        "fl": (yy >= 70) & (xx <= 33),
        "mid": (yy >= 70) & (xx >= 34) & (xx <= 48),
        "hr": (yy >= 70) & (xx >= 49) & (xx <= 63),
        "far": (yy >= 70) & (xx >= 64),
    }
    feet = {}
    claimed = np.zeros_like(opaque)
    for name in FOOTS:
        foot = opaque & bounds[name] & ~claimed
        feet[name] = foot
        claimed |= foot

    def cushion(mask, rows=2):
        ys = np.nonzero(mask)[0]
        if len(ys) == 0:
            return np.zeros_like(mask)
        return mask & (yy < int(ys.min()) + rows)

    hip = np.zeros_like(opaque)
    for name in FOOTS:
        hip |= cushion(feet[name])
    legs = claimed
    # Jaw: the lip directly under each red mouth pixel, plus the yellow chin
    # under the eye row. The four eye-spots (y < 54) stay on the shield.
    red = is_col(img, RED)
    yellow = is_col(img, YELLOW)
    black = is_col(img, BLACK)
    under = np.zeros_like(opaque)
    rys, rxs = np.nonzero(red)
    for y, x in zip(rys.tolist(), rxs.tolist()):
        y0, y1 = y + 1, min(h, y + 4)
        x0, x1 = max(0, x - 1), min(w, x + 2)
        under[y0:y1, x0:x1] = True
    under &= opaque & (yy <= 66) & ~legs
    chin = yellow & (yy >= 57) & ~legs
    jaw = under | chin
    pad = np.pad(jaw, 1)
    near = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    rim = black & near & (yy >= 57) & (yy <= 66) & ~red & ~legs
    jaw = jaw | rim
    body = (opaque & ~legs) | (opaque & hip)
    masks = {"body": body, "jaw": jaw, "hip": hip}
    masks.update(feet)
    return masks


def place_geometry(img):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    cy = float(ys.mean())
    cx = float(xs.mean())
    pivot = (dst_x + (cx - x0), dst_y + (cy - y0))
    return (y0, y1, x0, x1, dst_y, dst_x, pivot)


def build_layers(img, masks):
    y0, y1, x0, x1, dst_y, dst_x, pivot = place_geometry(img)
    sh, sw = y1 - y0, x1 - x0
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    layers = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr
    return layers, pivot, canvas


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def gait(f, n, phase):
    """Positive half of a 2-cycle wave. Zero at both loop seams."""
    return max(0.0, wave(f, n, 1.0, phase=phase, cycles=2))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        bdx=0, bdy=0,
        fdx=(0, 0, 0, 0), fdy=(0, 0, 0, 0),
        jaw=0, spin=0.0, hurt=0.0,
        rings=0, ring=0, burst=0, sparks=0, lines=0,
    )
    if action == "idle":
        p["bdy"] = -int(round(max(0.0, wave(f, n, 1.0, cycles=1))))
    elif action == "walk":
        p["bdy"] = -int(round(max(0.0, wave(f, n, 1.0, cycles=2))))
        # Diagonal pairs: near-front with near-hind, mid with far-hind.
        lifts, swings = [], []
        for pair in (0, 1, 0, 1):
            s = gait(f, n, 0.0 if pair == 0 else math.pi)
            lifts.append(-int(round(3 * s)))
            swings.append(-int(round(2 * s)))
        p["fdy"] = tuple(lifts)
        p["fdx"] = tuple(swings)
    elif action == "attack":
        jaw = [0, 2, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        p["jaw"] = jaw[f]
        if 1 <= f <= 4:
            p["rings"] = 2 if f >= 2 else 1
            p["ring"] = f - 1
        if 5 <= f <= 10:
            ang = 60.0 * (f - 4)
            p["spin"] = 0.0 if ang % 360 == 0 else ang
            if p["spin"]:
                p["lines"] = 3
        if f in (11, 12):
            p["burst"] = 9 if f == 11 else 6
    else:
        env = math.sin(math.pi * f / (n - 1))
        p["hurt"] = 38.0 * env
        p["bdx"] = int(round(3 * env))
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def open_jaw(img, jaw_layer, amount):
    """Drop the existing lip. The gap is mouth-black from the palette."""
    if amount <= 0 or not np.any(jaw_layer[..., 3] >= 128):
        return img
    out = img.copy()
    jm = jaw_layer[..., 3] >= 128
    out[jm] = 0
    moved = shift(jaw_layer, 0, amount)
    paste(out, moved)
    vacated = jm & (out[..., 3] < 128)
    out[vacated, :3] = BLACK
    out[vacated, 3] = 255
    return out


def roar_rings(out, mouth, age, strength):
    """Rings leave the mouth toward screen-left (this still faces left)."""
    mx, my = mouth
    cols = (LIGHT, YELLOW, GOLD)
    for r_i in range(strength):
        rad = 5 + age * 3 + r_i * 4
        col = cols[r_i % 3]
        for deg in range(-56, 57, 14):
            ang = math.radians(deg)
            put(out, mx - rad * math.cos(ang), my + rad * math.sin(ang) * 0.62, col)
    return out


def speed_lines(out, n, seed):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    rear = x1 - 2
    for k in range(n):
        y = int(rng.integers(y0 + 6, max(y0 + 7, y1 - 6)))
        length = int(rng.integers(5, 8))
        gap = int(rng.integers(2, 5))
        for i in range(length):
            put(out, rear + gap + i, y, LIGHT if i < 2 else STEEL)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    cols = (LIGHT, YELLOW, GOLD)
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(2, size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), cols[r % 3])
    put(out, cx, cy, LIGHT)
    return out


def sparks(out, frame_idx):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    d = frame_idx
    cx = (x0 + x1) // 2
    pts = [(x1 + 2 + d, y0 + 8), (x0 - 2 - d, y0 + 12), (cx, y0 - 2 - d)]
    for sx, sy in pts:
        put(out, sx, sy, LIGHT)
        put(out, sx + 1, sy, YELLOW)
    return out


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mouth_anchor(jaw_layer, fallback):
    ys, xs = np.nonzero(jaw_layer[..., 3] >= 128)
    if len(ys) == 0:
        return fallback
    # Front of a left-facing jaw is the leftmost lip pixel.
    tip = int(xs.min())
    ys_at = ys[xs == tip]
    return tip, int(np.median(ys_at))


def repair(img):
    img = fill_pinholes(img)
    img = bridge_parts(img)
    img = despeckle(img)
    return img


def synth(layers, pivot, action, f, palette, mouth):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    bdx, bdy = p["bdx"], p["bdy"]
    body = shift(layers["body"], bdx, bdy)
    legs = [shift(layers[name], p["fdx"][i], p["fdy"][i]) for i, name in enumerate(FOOTS)]
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(img, body)
    # Far hind, near hind, mid, then the near front foot.
    for leg in (legs[3], legs[2], legs[1], legs[0]):
        paste(img, leg)
    jaw = shift(layers["jaw"], bdx, bdy)
    spinning = abs(p["spin"]) > 0.01
    recoiling = abs(p["hurt"]) > 0.01
    if p["jaw"] and not spinning and not recoiling:
        img = open_jaw(img, jaw, p["jaw"])
    if spinning or recoiling:
        ang = p["spin"] if spinning else p["hurt"]
        img = rotate_nearest(img, ang, (pivot[0] + bdx, pivot[1] + bdy))
        img = repair(img)
    img = snap(img, palette)
    if p["lines"] and spinning:
        img = speed_lines(img, p["lines"], seed=f * 17 + 11)
    if p["rings"]:
        tip = mouth_anchor(jaw, mouth)
        img = roar_rings(img, (tip[0], tip[1] + p["jaw"]), p["ring"], p["rings"])
    if p["burst"]:
        tip = mouth_anchor(shift(layers["jaw"], bdx, bdy), mouth)
        img = impact_burst(img, (tip[0] - 5, tip[1]), p["burst"])
    if p["sparks"]:
        img = sparks(img, f)
    return snap(img, palette)


def facing_flip(img, direction):
    if direction in FACE_RIGHT:
        return img[:, ::-1].copy()
    return img


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (SPECIES / "front.png").read_bytes()
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
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 2, 4, 7, 8, 10, 12, 14),
        "attack": (0, 2, 4, 6, 8, 10, 11, 13),
        "hurt": (0, 2, 4, 6, 8, 10),
    }
    cols = max(len(v) for v in samples.values())
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
            # West row is the unflipped still, so the shield stays on the left.
            tile = sheets[act][6 * CELL_H:7 * CELL_H, f * CELL_W:(f + 1) * CELL_W]
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


def main_component_fraction(img):
    a = img[..., 3] >= 128
    parts = components(a)
    if not parts:
        return 0.0, 0
    parts.sort(key=len, reverse=True)
    total = int(a.sum())
    return len(parts[0]) / max(1, total), len(parts)


def south_attack_ratios(sheets):
    row = sheets["attack"][:CELL_H]
    counts = []
    for f in range(FRAMES["attack"]):
        tile = row[:, f * CELL_W:(f + 1) * CELL_W]
        counts.append(int((tile[..., 3] == 255).sum()))
    base = max(1, counts[0])
    return counts, [c / base for c in counts]


def qa_report(master, sheets, palette, frames_dir, digest, counts, sprite_h, skip_qa):
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = np.unique(master[..., 3])
    bad_alpha = [int(v) for v in alphas if int(v) not in (0, 255)]
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{d}:{f}")
    _counts, ratios = south_attack_ratios(sheets)
    report = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_h,
        "off_palette_colors": len(off),
        "bad_alpha": bad_alpha,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "attack_ratios": [round(v, 4) for v in ratios],
    }
    if not skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        qa = OUT_DIR / "_work" / "qa"
        qa.mkdir(parents=True, exist_ok=True)
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
    return report


def object_slice(text, key):
    token = f'"{key}":'
    start = text.index(token)
    i = start + len(token)
    while i < len(text) and text[i] != "{":
        i += 1
    depth = 0
    end = i
    while end < len(text):
        ch = text[end]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end += 1
                break
        end += 1
    return start, end, text[start:end]


def with_flock(lock_path, fn):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def publish_dir8(sprite_h):
    entry = {
        "dex": 411, "form": "normal", "variant": "bastiodon", "gen": "gen4",
        "dir": "monsterSprite/gen4/0411_bastiodon/normal",
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
        if f'"{NEIGHBOR}":' not in text:
            raise SystemExit(f"neighbor {NEIGHBOR} missing")
        _ns, _ne, neighbor_before = object_slice(text, NEIGHBOR)
        key = f'"{KEY}":'
        if key in text:
            start, end, _old = object_slice(text, KEY)
            # object_slice includes the key. Replace only the object value.
            colon = text.index(":", start)
            val = colon + 1
            text = text[:val] + blob + text[end:]
        else:
            text = text.replace(prefix + "{", prefix + "{" + key + blob + ",", 1)
        _ns, _ne, neighbor_after = object_slice(text, NEIGHBOR)
        if neighbor_before != neighbor_after:
            raise SystemExit("neighbor key string changed; not publishing")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(text)
        json.loads(text[len(prefix):].rstrip().rstrip(";"))
        tmp.replace(DIR8)
        return True

    with_flock(DIR8_LOCK, edit)


def update_queue(status):
    def edit():
        fd = os.open(QUEUE, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with os.fdopen(os.dup(fd), "r", encoding="utf-8") as handle:
                data = json.loads(handle.read())
            found = False
            for item in data["items"]:
                if item.get("id") != KEY:
                    continue
                item["status"] = status
                item.pop("claim", None)
                found = True
                break
            if not found:
                raise SystemExit("queue has no 0411_bastiodon")
            raw = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            tmp = QUEUE.with_suffix(".json.tmp")
            tmp.write_text(raw, encoding="utf-8")
            tmp.replace(QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def measure_facing(img):
    yellow = is_col(img, YELLOW)
    gold = is_col(img, GOLD)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    eyes = yellow & (yy <= 50)
    ey, ex = np.nonzero(eyes)
    gy, gx = np.nonzero(gold)
    eye_x = int(np.median(ex)) if len(ex) else 99
    belly_x = int(np.median(gx)) if len(gx) else -1
    return eye_x, belly_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SPECIES / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    opaque = front[..., 3] >= 128
    covered = masks["body"] | masks["fl"] | masks["mid"] | masks["hr"] | masks["far"]
    missing = int((opaque & ~covered).sum())
    extra = int((covered & ~opaque).sum())
    counts = {k: int(v.sum()) for k, v in masks.items()}
    counts["missing"] = missing
    counts["extra"] = extra
    counts["feet"] = 4
    print("parts", counts, "palette", len(palette), flush=True)
    if missing or extra:
        raise SystemExit("segmentation does not cover the still")
    if counts["jaw"] < 8:
        raise SystemExit("jaw mask is too small; the mouth was not found")
    eye_x, belly_x = measure_facing(front)
    print("facing check eye_x", eye_x, "belly_x", belly_x, flush=True)
    if not (eye_x < belly_x):
        raise SystemExit("expected a left-facing shield")

    red = is_col(front, RED)
    rys, rxs = np.nonzero(red)
    order = int(np.argmin(rxs))
    mouth_src = (int(rxs[order]), int(rys[order]))
    print("mouth_src", mouth_src, "red", int(red.sum()), flush=True)

    layers, pivot, canvas = build_layers(front, masks)
    y0, y1, x0, x1, dst_y, dst_x, _pivot = place_geometry(front)
    mouth = (dst_x + (mouth_src[0] - x0), dst_y + (mouth_src[1] - y0))
    print("pivot", [round(v, 2) for v in pivot], "mouth", mouth, flush=True)

    rest = synth(layers, pivot, "idle", 0, palette, mouth)
    differ = np.any(rest != canvas, axis=2) & ((rest[..., 3] >= 128) | (canvas[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "body": (70, 78, 96), "fl": (220, 60, 60), "mid": (40, 90, 220),
        "hr": (240, 200, 40), "far": (80, 200, 120), "jaw": (255, 80, 200),
        "hip": (255, 255, 255),
    }
    for name, col in part_cols.items():
        if name == "hip":
            continue
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    part[masks["hip"], :3] = (255, 255, 255)
    part[masks["hip"], 3] = 255
    part[masks["jaw"], :3] = (255, 80, 200)
    part[masks["jaw"], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", f) for f in (0, 4, 8, 12)]
        want += [("walk", f) for f in (0, 2, 4, 6, 8, 12)]
        want += [("attack", f) for f in (0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13)]
        want += [("hurt", f) for f in (0, 1, 3, 6, 11)]
        tiles = [synth(layers, pivot, act, f, palette, mouth) for act, f in want]
        for act, f in (("attack", 6), ("attack", 7), ("attack", 8), ("hurt", 6)):
            img = synth(layers, pivot, act, f, palette, mouth)
            frac, n = main_component_fraction(img)
            print(f"cc {act}:{f} main {frac:.3f} parts {n}", flush=True)
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
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest and not args.force:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                img = facing_flip(synth(layers, pivot, act, f, palette, mouth), d)
                Image.fromarray(img, "RGBA").save(tmp)
                tmp.replace(fp)
            if key not in state["done"]:
                state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions_dir = SPECIES / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    preview.save(actions_dir / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    west = sheets["idle"][6 * CELL_H:7 * CELL_H, :CELL_W]
    differ = np.any(west != canvas, axis=2) & ((west[..., 3] >= 128) | (canvas[..., 3] >= 128))
    counts_a, ratios = south_attack_ratios(sheets)
    print("W idle0 pixels changed", int(differ.sum()), "sprite_h", sprite_h, flush=True)
    print("south attack opaque", counts_a, flush=True)
    print("south attack ratios", [round(v, 4) for v in ratios], flush=True)
    bad_ratio = [round(v, 4) for v in ratios if v < 0.90 or v > 1.08]
    if bad_ratio or int(differ.sum()) != 0:
        raise SystemExit(f"qa failed ratios {bad_ratio} idle_diff {int(differ.sum())}")

    report = qa_report(master, sheets, palette, frames_dir, digest, counts, sprite_h, args.skip_qa)
    qa = work / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False), flush=True)
    if report["off_palette_colors"] or report["bad_alpha"] or report["edge_touching"]:
        raise SystemExit("qa failed palette, alpha, or edge")

    meta = {
        "generator": "forge_bastiodon_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 411,
        "form": "normal",
        "variant": "bastiodon",
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "face_right": sorted(FACE_RIGHT),
        "feet": 4,
        "palette_colors": int(len(palette)),
        "off_palette_colors": report["off_palette_colors"],
        "attack_ratios": report["attack_ratios"],
        "source": "front.png 96x96",
        "gifs_untouched": True,
        "face_plate": "attached",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        publish_dir8(sprite_h)
        update_queue("done")
        print("published", KEY, "h", sprite_h, flush=True)


if __name__ == "__main__":
    main()
