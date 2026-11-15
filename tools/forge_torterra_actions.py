#!/usr/bin/env python3
"""Torterra (0389) 8-direction action sheet from this front.png only.

Measured on normal/front.png (96x96, alpha already 0 or 255, 13 colours):
  opaque bbox x12-83 y11-87 (72x77). Centroid about (48.9, 50.1), max radius 48.8.
  Faces left: the dark-grey beak tip sits left of the tan eye/cheek.
  The shell, white spikes, and the tree (canopy plus trunk) are one body.
  Three feet actually separate from the belly: near-left, long centre, far-right.
  A fourth leg is not a separate component, so none is invented.
  The beak is the mouth. Roar drops that existing jaw and draws rings.

Idle frame 0 is the still. Walk plants the three feet in turn. Attack is
roar (0-4), one-piece 360 spin (5-10), impact (11-12), recover (13-15).
Hurt recoils at most 38 degrees, no belly-flip squash. Shell and tree never
leave the body. front/back png and gif are not written.
"""

import argparse
import fcntl
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

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SPECIES = ROOT / "monsterSprite/gen4/0389_torterra/normal"
OUT_DIR = SPECIES / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0389_torterra/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left, so east-side directions are the mirror. E looks screen-right.
FACE_RIGHT = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 184, 176, 152
VERSION = "torterra_v2"
KEY = "0389_torterra"

BLACK = (16, 16, 16)
WHITE = (230, 230, 230)
LGRAY = (189, 189, 189)
MGRAY = (164, 164, 164)
GREEN = (65, 172, 65)
DGREEN = (24, 98, 24)
BROWN = (139, 106, 65)
DBROWN = (98, 74, 57)
TAN = (180, 131, 82)
STEEL = (106, 115, 123)
DSTEEL = (74, 82, 90)

LEG_COLOURS = {BROWN, DBROWN, TAN, WHITE, LGRAY, MGRAY}
BEAK_COLOURS = {STEEL, DSTEEL}


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
    """4x nearest. Positive degrees turn clockwise on screen. Exact palette."""
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
        arr = np.array(pts)
        out.append(arr)
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
    """Fill only tiny enclosed holes left by nearest rotation. Open gaps stay open."""
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
    """Rejoin a fragment rotation pulled a couple of pixels off the shell or the beak."""
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
        # A nicked trunk is large and close. Specks are small. Rings are added later.
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


def colour_mask(img, colours):
    rgb = img[..., :3]
    a = img[..., 3] >= 128
    m = np.zeros(a.shape, bool)
    for c in colours:
        m |= a & np.all(rgb == np.array(c, np.uint8), axis=-1)
    return m


def add_outline(mask, black, bounds):
    """Sole and claw outline travels with the foot. One black pixel of contact is enough."""
    out = mask.copy()
    for _ in range(2):
        p = np.pad(out, 1)
        near = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        out = out | (black & near & bounds & ~out)
    return out


def segment(img):
    """Partition this still. Shell, tree, head, and spikes stay on the body."""
    h, w = img.shape[:2]
    opaque = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = img[..., :3]
    black = opaque & np.all(rgb == np.array(BLACK, np.uint8), axis=-1)
    leg_col = colour_mask(img, LEG_COLOURS)
    beak_col = colour_mask(img, BEAK_COLOURS)

    # Beak: the dark-grey component whose tip is the low grey pixel near the head.
    beak_allow = beak_col & (yy >= 56) & (yy <= 74) & (xx >= 12) & (xx <= 38)
    tip_ys, tip_xs = np.nonzero(beak_allow)
    beak = np.zeros_like(opaque)
    if len(tip_ys):
        # Lowest grey, then the leftmost of those, is the beak tip (faces left).
        low = tip_ys.max()
        tip_x = int(tip_xs[tip_ys == low].min())
        seed = np.zeros_like(opaque)
        seed[int(low), tip_x] = True
        beak = flood(seed, beak_allow)
    bys = np.nonzero(beak)[0]
    jaw = np.zeros_like(opaque)
    if len(bys):
        # Lower tip of the existing beak is the jaw. The upper beak stays on the head.
        jaw = beak & (yy >= int(bys.max()) - 6)
        p = np.pad(jaw, 1)
        near = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        jaw = jaw | (black & near & (yy >= int(bys.max()) - 6) & (xx <= 38) & ~beak_col)

    def leg_from(seed_x, seed_y, bounds):
        seed = np.zeros_like(opaque)
        if not (0 <= seed_y < h and 0 <= seed_x < w):
            return seed
        # Walk up onto a leg-coloured pixel if the seed itself is outline.
        if not (leg_col[seed_y, seed_x] & bounds[seed_y, seed_x]):
            found = None
            for rad in range(1, 5):
                for dy in range(-rad, rad + 1):
                    for dx in range(-rad, rad + 1):
                        ny, nx = seed_y + dy, seed_x + dx
                        if 0 <= ny < h and 0 <= nx < w and leg_col[ny, nx] and bounds[ny, nx]:
                            found = (ny, nx)
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                return seed
            seed_y, seed_x = found
        seed[seed_y, seed_x] = True
        core = flood(seed, leg_col & bounds & ~beak & ~jaw)
        return add_outline(core, black & ~beak & ~jaw, bounds)

    # Cuts follow the transparent gaps measured between the three feet.
    # Left foot x12-27 below the beak. Centre foot x44-64 from the ankle down.
    # Right foot x66-78. Do not cross those gaps into the belly.
    fl_bounds = (yy >= 64) & (xx <= 28)
    mid_bounds = (yy >= 74) & (xx >= 44) & (xx <= 64)
    hr_bounds = (yy >= 73) & (xx >= 66)
    fl = leg_from(18, 80, fl_bounds)
    mid = leg_from(57, 84, mid_bounds)
    hr = leg_from(72, 78, hr_bounds)
    # A pixel belongs to one leg. Near foot wins a shared outline.
    mid = mid & ~fl
    hr = hr & ~fl & ~mid

    def cushion(mask, rows=5):
        ys = np.nonzero(mask)[0]
        if len(ys) == 0:
            return np.zeros_like(mask)
        return mask & (yy < int(ys.min()) + rows)

    hip = cushion(fl) | cushion(mid) | cushion(hr)
    legs = fl | mid | hr
    body = (opaque & ~legs) | (opaque & hip)
    # Jaw stays on the body until the roar drops it. Hip stays on both.
    return {"body": body, "fl": fl, "mid": mid, "hr": hr, "jaw": jaw, "hip": hip}


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
    placed = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr
    return layers, placed, pivot, canvas


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def foot_window(f, n, slot, slots=3, cycles=2):
    """Each foot owns one beat per cycle. Value is 0 at the loop ends and at every plant."""
    ph = (cycles * f / n) % 1.0
    width = 1.0 / slots
    start = slot * width
    if ph < start or ph > start + width:
        return 0.0
    t = (ph - start) / width
    return math.sin(math.pi * t)


def pose(action, f):
    n = FRAMES[action]
    p = dict(bdx=0, bdy=0, fdx=(0, 0, 0), fdy=(0, 0, 0), jaw=0, spin=0.0, hurt=0.0,
             rings=0, ring=0, burst=0, flash=0, sparks=0, lines=0)
    if action == "idle":
        # Whole shell, tree, and head breathe up by one pixel. Feet stay down.
        p["bdy"] = -int(round(max(0.0, wave(f, n, 1.0))))
    elif action == "walk":
        p["bdy"] = -int(round(max(0.0, math.sin(2 * math.pi * 2 * f / n))))
        lifts = []
        swings = []
        for slot in (0, 1, 2):
            s = foot_window(f, n, slot)
            lifts.append(-int(round(3 * s)))
            # Forward is -x on this left-facing still. The foot returns to the plant.
            swings.append(-int(round(2 * s)))
        p["fdy"] = tuple(lifts)
        p["fdx"] = tuple(swings)
    elif action == "attack":
        jaw = [0, 1, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        p["jaw"] = jaw[f]
        if 1 <= f <= 4:
            p["rings"] = 2 if f >= 2 else 1
            p["ring"] = f - 1
        if 5 <= f <= 10:
            ang = 60.0 * (f - 4)
            p["spin"] = 0.0 if ang % 360 == 0 else ang
            p["lines"] = 4
        if f in (11, 12):
            p["burst"] = 12 if f == 11 else 8
        if f == 11:
            p["flash"] = 0
    else:
        env = math.sin(math.pi * f / (n - 1))
        ang = 38.0 * env
        p["hurt"] = ang
        p["bdx"] = int(round(4 * env))
        p["jaw"] = 2 if 2 <= f <= 6 else (1 if f in (1, 7) else 0)
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def open_jaw(img, jaw_layer, amount):
    """Drop the existing beak tip. The gap is mouth-black from the palette."""
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
    """Rings leave the beak toward screen-left (this still faces left)."""
    mx, my = mouth
    cols = (WHITE, LGRAY, GREEN)
    for r_i in range(strength):
        rad = 6 + age * 3 + r_i * 5
        col = cols[min(r_i, 2)]
        for deg in range(-50, 51, 8):
            ang = math.radians(deg)
            # Facing -1: the arc opens to the left of the beak.
            put(out, mx - rad * math.cos(ang), my + rad * math.sin(ang) * 0.72, col)
    return out


def speed_lines(out, n, seed):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    # Streaks sit behind the shell (right side of a left-facing body).
    rear = x1 - 2
    for k in range(n):
        y = int(rng.integers(y0 + 8, max(y0 + 9, y1 - 8)))
        length = int(rng.integers(6, 11))
        gap = int(rng.integers(3, 7))
        for i in range(length):
            put(out, rear + gap + i, y, WHITE if i < 2 else LGRAY)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    cols = (WHITE, TAN, LGRAY)
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(max(2, size // 3), size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), cols[r % 3])
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, WHITE)
    return out


def sparks(out, frame_idx):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    d = frame_idx
    cx = (x0 + x1) // 2
    pts = [(x1 + 2 + d, y0 + 6), (x0 - 2 - d, y0 + 10), (cx, y0 - 2 - d)]
    for sx, sy in pts:
        put(out, sx, sy, WHITE)
        put(out, sx + 1, sy, TAN)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., 0].astype(np.int16) + out[..., 1] + out[..., 2] < 80)
    out[a & ~dark, :3] = WHITE
    return out


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def jaw_tip(jaw_layer):
    ys, xs = np.nonzero(jaw_layer[..., 3] >= 128)
    if len(ys) == 0:
        return None
    low = ys.max()
    tip_x = int(xs[ys == low].min())
    return tip_x, int(low)


def repair(img):
    img = fill_pinholes(img)
    img = bridge_parts(img)
    img = despeckle(img)
    return img


def synth(layers, pivot, action, f, palette):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    bdx, bdy = p["bdx"], p["bdy"]
    body = shift(layers["body"], bdx, bdy)
    names = ("fl", "mid", "hr")
    legs = [shift(layers[name], p["fdx"][i], p["fdy"][i]) for i, name in enumerate(names)]
    # Far foot, then centre, then the near left foot.
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(img, body)
    for leg in (legs[2], legs[1], legs[0]):
        paste(img, leg)
    jaw = shift(layers["jaw"], bdx, bdy)
    spinning = abs(p["spin"]) > 0.01
    recoiling = abs(p["hurt"]) > 0.01
    if p["jaw"] and not spinning:
        img = open_jaw(img, jaw, p["jaw"])
    if spinning or recoiling:
        # One bitmap: shell, tree, head, beak, and feet rotate together.
        ang = p["spin"] if spinning else p["hurt"]
        img = rotate_nearest(img, ang, pivot)
        img = repair(img)
    img = snap(img, palette)
    if p["lines"] and spinning:
        img = speed_lines(img, p["lines"], seed=f * 17 + 3)
    if p["rings"]:
        tip = jaw_tip(jaw)
        if tip and not spinning:
            # The open jaw dropped; rings start at the dropped tip.
            img = roar_rings(img, (tip[0], tip[1] + p["jaw"]), p["ring"], p["rings"])
    if p["burst"]:
        tip = jaw_tip(shift(layers["jaw"], bdx, bdy))
        if tip:
            img = impact_burst(img, (tip[0] - 6, tip[1] - 2), p["burst"])
    if p["flash"]:
        img = flash_white(img)
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
            # West row is the unflipped still, so the beak stays on the left.
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
    report = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_h,
        "off_palette_colors": len(off),
        "bad_alpha": bad_alpha,
        "edge_touching": {k: len(v) for k, v in clip.items()},
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SPECIES / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    opaque = front[..., 3] >= 128
    covered = masks["body"] | masks["fl"] | masks["mid"] | masks["hr"]
    missing = int((opaque & ~covered).sum())
    extra = int((covered & ~opaque).sum())
    counts = {k: int(v.sum()) for k, v in masks.items()}
    counts["missing"] = missing
    counts["extra"] = extra
    print("parts", counts, "palette", len(palette), flush=True)
    if missing or extra:
        raise SystemExit("segmentation does not cover the still")

    # Beak tip is left of the tan cheek, so the still faces left.
    beak = colour_mask(front, BEAK_COLOURS) & masks["jaw"]
    by, bx = np.nonzero(beak)
    tan = colour_mask(front, {TAN})
    ty, tx = np.nonzero(tan & (np.arange(96)[:, None] >= 60) & (np.arange(96)[:, None] <= 72) & (np.arange(96)[None, :] <= 42))
    if len(bx) and len(tx):
        print("facing check beak_tip_x", int(bx[by == by.max()].min()), "cheek_x", int(np.median(tx)), flush=True)

    layers, placed, pivot, canvas = build_layers(front, masks)
    print("pivot", [round(v, 2) for v in pivot], flush=True)

    rest = synth(layers, pivot, "idle", 0, palette)
    differ = np.any(rest != canvas, axis=2) & ((rest[..., 3] >= 128) | (canvas[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "body": (40, 160, 60), "fl": (220, 60, 60), "mid": (40, 90, 220),
        "hr": (240, 200, 40), "jaw": (40, 220, 220), "hip": (255, 255, 255),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", f) for f in (0, 4, 8, 12)]
        want += [("walk", f) for f in (0, 2, 3, 4, 7, 8, 12, 14)]
        want += [("attack", f) for f in (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13)]
        want += [("hurt", f) for f in (0, 1, 3, 5, 8, 11)]
        tiles = []
        for act, f in want:
            tiles.append(synth(layers, pivot, act, f, palette))
        for act, f in (("attack", 6), ("attack", 7), ("attack", 8), ("hurt", 5)):
            img = synth(layers, pivot, act, f, palette)
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
                if key in state.get("done", []) and fp.exists() and fp.stat().st_size > 0:
                    continue
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest and not args.force:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                img = facing_flip(synth(layers, pivot, act, f, palette), d)
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

    # West idle 0 is the unflipped still. South is its mirror.
    west = sheets["idle"][6 * CELL_H:7 * CELL_H, :CELL_W]
    differ = np.any(west != canvas, axis=2) & ((west[..., 3] >= 128) | (canvas[..., 3] >= 128))
    print("W idle0 pixels changed", int(differ.sum()), "sprite_h", sprite_h, flush=True)

    report = qa_report(master, sheets, palette, frames_dir, digest, counts, sprite_h, args.skip_qa)
    qa = work / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:800], flush=True)

    meta = {
        "generator": "forge_torterra_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 389,
        "form": "normal",
        "variant": "torterra",
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "face_right": sorted(FACE_RIGHT),
        "palette_colors": int(len(palette)),
        "off_palette_colors": report["off_palette_colors"],
        "source": "front.png 96x96",
        "gifs_untouched": True,
        "shell_and_tree": "attached",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
