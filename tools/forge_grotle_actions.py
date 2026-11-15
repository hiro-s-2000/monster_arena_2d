#!/usr/bin/env python3
"""Grotle (0388_grotle) 8-direction x 4-action sheet, from this form's front.png.

Measured on the 96x96 still (opaque bbox x20-75, y15-72). The beak tip
(x≈21, y≈50) is left of the white eye (x≈32, y≈42), so the drawing faces
left. Unflipped directions are N NW W SW. S SE E NE are the horizontal mirror.
E faces screen-right.

Body plan, from these pixels only: a planted quadruped turtle. Two separate
leaf crests sit on a rigid shell. The yellow head and closed beak have no
mouth slit, so they stay on the torso (no invented jaw). Three feet are
visible — front (image left), middle (lowest), rear (image right). No fourth
leg is drawn.

  idle   : feet planted. The two crests rustle. Frame 0 is the still.
  walk   : in-place trot of the three existing feet. Crests counter-sway.
  attack : 0-4 crest flare plus rings at the beak; 5-10 one-piece 360 spin;
           11-12 impact; 13-15 recover.
  hurt   : knock-back and a recoil of at most 38 degrees. No belly flip.

Every opaque pixel is snapped to the still palette. Alpha is 0 or 255.
front/back png and gif are not written.

  --force     rebuild frames
  --samples   write _work/parts.png and _work/samples.png, then stop
  --no-sync   do not copy sheets to pokemon_arena_2d
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

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SPECIES = ROOT / "monsterSprite/gen4/0388_grotle/normal"
OUT_DIR = SPECIES / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0388_grotle/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left, so the north-west quadrant keeps the drawing.
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "grotle_v2"

WHITE = (255, 255, 255)
YELLOW = (238, 213, 65)
TAN = (222, 180, 65)
INK = (16, 16, 24)
GREEN = (98, 180, 106)
DGREEN = (57, 139, 65)
OLIVE = (57, 90, 41)


def flood(seeds, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow)
    q = deque()
    for y, x in seeds:
        y, x = int(y), int(x)
        if 0 <= y < h and 0 <= x < w and allow[y, x] and not seen[y, x]:
            seen[y, x] = True
            q.append((y, x))
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


def iscol(rgb, opaque, *cs):
    m = np.zeros(opaque.shape, bool)
    for c in cs:
        m |= np.all(rgb == np.array(c, np.uint8), axis=2)
    return opaque & m


def segment(img):
    """Partition this still. Crests are the two bushes. Legs are the three feet."""
    h, w = img.shape[:2]
    opaque = img[..., 3] >= 128
    rgb = img[..., :3]
    yy, xx = np.mgrid[0:h, 0:w]
    bush = iscol(rgb, opaque, GREEN, DGREEN)
    ys, xs = np.nonzero(bush & (xx <= 40))
    if len(ys) == 0:
        raise SystemExit("left bush not found")
    left_bush = flood([(int(ys[0]), int(xs[0]))], bush)
    right_bush = bush & ~left_bush
    extra_allow = iscol(rgb, opaque, OLIVE, INK) & (yy <= 38)
    crest_all = flood(
        [(int(y), int(x)) for y, x in zip(*np.nonzero(bush & (yy <= 38)))],
        bush | extra_allow,
    )
    extra = crest_all & ~bush
    crest_l = left_bush.copy()
    crest_r = right_bush.copy()
    if np.any(extra):
        dist_l = _side_dist(extra, left_bush)
        dist_r = _side_dist(extra, right_bush)
        ey, ex = np.nonzero(extra)
        take_l = dist_l[ey, ex] <= dist_r[ey, ex]
        crest_l[ey[take_l], ex[take_l]] = True
        crest_r[ey[~take_l], ex[~take_l]] = True

    # Feet only. The green chest above y61 stays on the torso. No fourth leg.
    leg_f = opaque & (xx <= 37) & (yy >= 61)
    leg_m = opaque & (xx >= 38) & (xx <= 58) & (yy >= 64)
    leg_r = opaque & (xx >= 59) & (yy >= 57)
    leg_m &= ~leg_f
    leg_r &= ~leg_f & ~leg_m
    used = crest_l | crest_r | leg_f | leg_m | leg_r
    torso = opaque & ~used
    parts = {
        "crest_l": crest_l,
        "crest_r": crest_r,
        "leg_f": leg_f,
        "leg_m": leg_m,
        "leg_r": leg_r,
        "torso": torso,
    }
    cover = np.zeros_like(opaque)
    for m in parts.values():
        if np.any(cover & m):
            raise SystemExit("part masks overlap")
        cover |= m
    if int(cover.sum()) != int(opaque.sum()):
        raise SystemExit(f"part cover {int(cover.sum())} != opaque {int(opaque.sum())}")
    return parts


def _side_dist(extra, seed):
    h, w = extra.shape
    dist = np.full((h, w), 9999, np.int16)
    q = deque()
    for y, x in zip(*np.nonzero(seed)):
        y, x = int(y), int(x)
        dist[y, x] = 0
        q.append((y, x, 0))
    while q:
        y, x, d = q.popleft()
        nd = d + 1
        if nd > 16:
            continue
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and extra[ny, nx] and dist[ny, nx] > nd:
                dist[ny, nx] = nd
                q.append((ny, nx, nd))
    return dist


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
    """4x nearest rotation. Palette stays exact."""
    if abs(angle_deg) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    # 360 lands on the unrotated pixels. The float path is not identity.
    ang = angle_deg % 360.0
    if ang < 0.45 or ang > 359.55:
        return layer.copy()
    up = np.repeat(np.repeat(layer, 4, axis=0), 4, axis=1)
    px, py = pivot[0] * 4 + 1.5, pivot[1] * 4 + 1.5
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
    return out[1::4, 1::4].copy()


def content_point(x, y, angle_deg, pivot):
    ang = angle_deg % 360.0
    if abs(angle_deg) < 0.45 or ang < 0.45 or ang > 359.55:
        return float(x), float(y)
    rad = math.radians(ang)
    c, s = math.cos(rad), math.sin(rad)
    vx, vy = x - pivot[0], y - pivot[1]
    return pivot[0] + c * vx - s * vy, pivot[1] + s * vx + c * vy


def despeckle(img):
    """Drop only pixels whose four neighbours are empty. Eyes and outlines stay."""
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
        d = (ty[sl, None].astype(np.int32) - pts[None, :, 0]) ** 2 + (
            tx[sl, None].astype(np.int32) - pts[None, :, 1]
        ) ** 2
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


def band_mask(part, from_top):
    """from_top True keeps the upper rows (a hip). False keeps the lower rows (a root)."""
    ys, xs = np.nonzero(part)
    if len(ys) == 0:
        return np.zeros_like(part)
    if from_top:
        keep = ys <= ys.min() + 2
    else:
        keep = ys >= ys.max() - 2
    out = np.zeros_like(part)
    out[ys[keep], xs[keep]] = True
    return out


def socket_fill(canvas, placed):
    """Torso-coloured cushion under each joint, hidden while the part is at rest."""
    torso = placed["torso"]
    spec = (
        ("crest_l", False, 3),
        ("crest_r", False, 3),
        ("leg_f", True, 2),
        ("leg_m", True, 2),
        ("leg_r", True, 2),
    )
    lyr = np.zeros_like(canvas)
    lyr[torso] = canvas[torso]
    for name, hip, rad in spec:
        zone = band_mask(placed[name], hip) & dilate(torso, rad)
        if not np.any(zone):
            continue
        fill = nearest_color(canvas, zone, torso)
        empty = lyr[..., 3] < 128
        lyr[zone & empty] = fill[zone & empty]
    return lyr


def pivot_of(part, hip):
    band = band_mask(part, from_top=hip)
    ys, xs = np.nonzero(band if np.any(band) else part)
    return float(xs.mean()), float(ys.mean())


def build_rig(img, masks, flip):
    opaque = img[..., 3] >= 128
    ys, xs = np.nonzero(opaque)
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
    layers["torso"] = socket_fill(canvas, placed)

    pivots = {
        "crest_l": pivot_of(placed["crest_l"], hip=False),
        "crest_r": pivot_of(placed["crest_r"], hip=False),
        "leg_f": pivot_of(placed["leg_f"], hip=True),
        "leg_m": pivot_of(placed["leg_m"], hip=True),
        "leg_r": pivot_of(placed["leg_r"], hip=True),
    }
    # Beak tip: leftmost opaque pixel in the head band, not a claw.
    head = opaque & (np.arange(img.shape[0])[:, None] >= 46) & (np.arange(img.shape[0])[:, None] <= 54)
    hy, hx = np.nonzero(head)
    tip = int(np.argmin(hx))
    snout = (float(hx[tip] - x0 + dst_x), float(hy[tip] - y0 + dst_y))

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        snout = (CELL_W - 1 - snout[0], snout[1])
    facing = 1.0 if flip else -1.0
    return layers, pivots, snout, facing, (dst_x, dst_y, sw, sh)


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    """Angles are for the unflipped left-facing drawing. Positive crest angle
    moves that bush's top toward screen-right. Positive leg angle swings the
    foot toward screen-left (forward). dx > 0 moves the body screen-right
    (backward)."""
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, tilt=0.0, spin=0.0,
        crest_l=0.0, crest_r=0.0, leg_f=0.0, leg_m=0.0, leg_r=0.0,
        rings=0, ring_age=0, lines=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        rustle = wave(f, n, 1.0)
        p["crest_l"] = -8.0 * rustle
        p["crest_r"] = 7.0 * wave(f, n, 1.0, phase=0.7)
    elif action == "walk":
        p["leg_f"] = wave(f, n, 26.0, cycles=2.0)
        p["leg_m"] = wave(f, n, 22.0, cycles=2.0, phase=math.pi)
        p["leg_r"] = wave(f, n, 16.0, cycles=2.0, phase=math.pi * 0.85)
        p["crest_l"] = -wave(f, n, 6.0, cycles=2.0)
        p["crest_r"] = wave(f, n, 5.0, cycles=2.0, phase=0.6)
    elif action == "attack":
        flare = (8, 14, 18, 18, 12, 0, 0, 0, 0, 0, 0, 4, 2, 2, 1, 0)
        dxs = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, -1, 0, 0)
        rings = (1, 2, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        rage = (0, 1, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        lines = (0, 0, 0, 0, 0, 4, 5, 5, 5, 4, 2, 0, 0, 0, 0, 0)
        burst = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0)
        p["crest_l"] = -float(flare[f])
        p["crest_r"] = float(flare[f])
        p["dx"] = float(dxs[f])
        p["rings"] = rings[f]
        p["ring_age"] = rage[f]
        p["lines"] = lines[f]
        p["burst"] = burst[f]
        if 5 <= f <= 10:
            p["spin"] = 60.0 * (f - 4)
    else:
        env = math.sin(math.pi * f / 11.0)
        p["dx"] = 6.0 * env
        p["dy"] = -1.0 * env
        p["tilt"] = 38.0 * env
        p["crest_l"] = 10.0 * env
        p["crest_r"] = 8.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 5 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def roar_rings(out, origin, facing, age, strength):
    ox, oy = origin
    cols = (WHITE, YELLOW, TAN)
    for r_i in range(strength):
        rad = 6 + age * 3 + r_i * 4
        col = cols[min(r_i, 2)]
        for deg in range(-50, 51, 8):
            ang = math.radians(deg)
            put(out, ox + facing * rad * math.cos(ang), oy + rad * math.sin(ang) * 0.72, col)
    return out


def speed_lines(out, facing, n, seed, length):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    rear_x = (x1 - (x1 - x0) // 6) if facing < 0 else (x0 + (x1 - x0) // 6)
    y_lo = y0 + (y1 - y0) // 3
    y_hi = y0 + (y1 - y0) * 3 // 4
    for k in range(n):
        y = int(rng.integers(y_lo, max(y_lo + 1, y_hi)))
        off = int(rng.integers(3, 8))
        span = int(rng.integers(max(3, length // 2), length + 1))
        for i in range(span):
            put(out, rear_x - facing * (off + i), y, WHITE if i < 2 else YELLOW)
    return out


def impact_burst(out, origin, facing, size):
    ox, oy = origin
    cx, cy = ox + facing * 8, oy
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
    # Backward of the nose.
    back = -facing
    pts = (
        (cx + back * (8 + d * 2), y0 + 8),
        (cx + back * (4 + d), y0 + 16),
        (cx + back * (10 + d), (y0 + y1) // 2),
    )
    for sx, sy in pts:
        put(out, sx, sy, WHITE)
        put(out, sx + back, sy, YELLOW)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def synth(rig, action, f, palette):
    layers, pivots, snout, facing, _place = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    flip = facing > 0
    mirror = -1.0 if flip else 1.0
    bdx = p["dx"] * mirror
    bdy = p["dy"]

    def moved(name, ang):
        return shift(rotate_nearest(layers[name], ang * mirror, pivots[name]), bdx, bdy)

    spinning = abs(p["spin"]) > 0.5
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(body, shift(layers["torso"], bdx, bdy))
    for name in ("leg_r", "leg_m", "leg_f", "crest_l", "crest_r"):
        paste(body, moved(name, p[name]))
    articulated = any(abs(p[k]) >= 0.45 for k in ("crest_l", "crest_r", "leg_f", "leg_m", "leg_r"))
    if articulated:
        body = despeckle(body)

    bb = body_box(body)
    center = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0) if bb else (CELL_W / 2.0, FOOT_Y - 30.0)
    sx = snout[0] + bdx
    sy = snout[1] + bdy
    if spinning:
        ang = p["spin"] * mirror
        body = despeckle(rotate_nearest(body, ang, center))
        sx, sy = content_point(sx, sy, ang, center)
    elif abs(p["tilt"]) >= 0.45:
        ang = p["tilt"] * mirror
        body = despeckle(rotate_nearest(body, ang, center))
        sx, sy = content_point(sx, sy, ang, center)

    img = snap(body, palette)
    if p["flash"]:
        img = flash_white(img)
    if p["rings"]:
        img = roar_rings(img, (sx, sy), facing, p["ring_age"], p["rings"])
    if p["lines"]:
        img = speed_lines(img, facing, p["lines"], seed=f * 17 + (1 if flip else 0), length=9)
    if p["burst"]:
        img = impact_burst(img, (sx, sy), facing, 10 if f >= 12 else 8)
    if p["sparks"]:
        img = sparks(img, f, facing)
    return snap(img, palette)


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
    samples = {"idle": (0, 4, 8, 12), "walk": (0, 2, 6, 10), "attack": (2, 7, 11, 14), "hurt": (1, 4, 6, 10)}
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            y = ai * CELL_H
            x = si * CELL_W
            m = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    """East row, every frame, cropped. E is the mirrored right-facing side."""
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


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    sizes = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        q = deque([(y, x)])
        seen[y, x] = True
        n = 0
        while q:
            cy, cx = q.popleft()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes


def sync_outputs():
    """Copy sheets only. _work stays on the revenge side."""
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def placed_still(front):
    opaque = front[..., 3] >= 128
    ys, xs = np.nonzero(opaque)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    placed = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    placed[dst_y:dst_y + sh, dst_x:dst_x + sw] = front[y0:y1, x0:x1]
    return placed


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
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    still = placed_still(front)
    rest_w = synth(rigs["W"], "idle", 0, palette)
    rest_s = synth(rigs["S"], "idle", 0, palette)
    diff_w = int(np.any(rest_w != still, axis=2).sum())
    mirror = still[:, ::-1].copy()
    diff_s = int(np.any(rest_s != mirror, axis=2).sum())
    print("rest-pose W", diff_w, "S", diff_s, flush=True)
    if diff_w or diff_s:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (70, 74, 88),
        "crest_l": (40, 210, 70),
        "crest_r": (40, 140, 255),
        "leg_f": (255, 50, 50),
        "leg_m": (255, 190, 40),
        "leg_r": (210, 60, 220),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 12)]
        want += [("attack", "W", f) for f in (0, 2, 4, 5, 6, 7, 8, 9, 11, 12, 15)]
        want += [("hurt", "W", f) for f in (0, 1, 3, 5, 8, 11)]
        want += [("idle", "S", 0), ("walk", "E", 4), ("attack", "E", 7)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        for label, tile in (("walk4", tiles[6]), ("atk7", tiles[13]), ("hurt5", tiles[20])):
            sizes = components(tile[..., 3] >= 128)
            print(label, "components", sizes[:6], flush=True)
        tw, th = CELL_W, CELL_H
        cols = 6
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * th, cols * tw, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = tile[m]
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
    actions_dir = SPECIES / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    preview.save(actions_dir / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    clip = {}
    split = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act in ("attack", "hurt") and d == "W":
                    sizes = components(c)
                    if sizes and sizes[0] < 0.9 * int(c.sum()):
                        split.setdefault(act, []).append({"f": f, "sizes": sizes[:5]})
    report = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_h,
        "off_palette_colors": len(off),
        "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "split_W": split,
    }
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
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
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1200], flush=True)

    meta = {
        "generator": "forge_grotle_actions",
        "version": VERSION,
        "key": "0388_grotle",
        "dex": 388,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED),
        "facing": "left",
        "palette_colors": int(len(palette)),
        "off_palette_colors": len(off),
        "source": "front.png 96x96",
        "gifs_untouched": True,
        "anatomy": "two shell bushes, three visible feet, closed beak, no invented mouth",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
