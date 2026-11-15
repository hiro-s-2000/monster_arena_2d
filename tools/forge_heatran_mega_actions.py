#!/usr/bin/env python3
"""Mega Heatran (0485_heatran-mega) 8-direction action sheet from front.png.

Archetype: quadruped_beast, but the still shows two supports, not four feet.
Measured on front.png (96x96, opaque 3434, bbox x 0-95, y 17-78):
  top 22% centroid x is about 54.6, so the still faces right.
  The yellow mouth flame is the jaw (about x 40-58, y 27-42), under the brow.
  The tan hump on the upper right is the back ridge, not a limb.
  Red and orange on the shell are lava, not eyes. The cross stays on the shell.
  Supports: image-left foot y>=68 x<=36, image-right foot y>=71 x>=54.

Unflipped directions are S SE E NE. E faces screen-right. N NW W SW are mirrored.

  idle   : the shell breathes one pixel. Feet stay planted. Frame 0 is the still.
  walk   : two-support tread. One foot is always down. The body does not rear.
  attack : v3 (2026-11-04, user: "the yellow mouth moves unnaturally; make it a
           powerful stomp"). The mouth flame never rotates. 0-4 rear up on the hind
           support while the fore foot lifts high, 5 slam, 6-11 ground shock rings,
           lava splash, rock debris and a short shake, 12-15 settle.
  hurt   : backward slide and nose-up recoil of at most 36 degrees. No belly mirror.

The turn is one rigid 4x-nearest rotation. This shell stays one piece through
180 degrees, so the turn is a full circle and still readable. Alpha is only
0 or 255. Every opaque pixel is snapped to the still palette.
front/back png and gif are not written.
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
MEGA_DIR = ROOT / "monsterSprite/gen4/0485_heatran/mega/heatran-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0485_heatran/mega/heatran-mega/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
FACE_RIGHT = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
# Wide shell. 90 degrees swings the nose about 51px below the centroid,
# so the cell is taller than the 168 guide or the roll clips the feet.
CELL_W, CELL_H, FOOT_Y = 200, 200, 156
VERSION = "heatran_mega_v3"

INK = (25, 25, 25)
INK2 = (24, 24, 24)
WHITE = (255, 255, 255)
SHELL = (238, 243, 247)
STEEL = (192, 203, 216)
GOLD = (255, 219, 76)
YELLOW = (255, 255, 101)
RED = (253, 83, 53)
GOLD2 = (251, 219, 76)
YELLOW2 = (251, 255, 101)


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def is_any(img, colors):
    a = img[..., 3] >= 128
    hit = np.zeros(a.shape, bool)
    rgb = img[..., :3]
    for c in colors:
        hit |= np.all(rgb == np.array(c, np.uint8), axis=-1)
    return a & hit


def largest_component(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    seen = np.zeros_like(mask)
    h, w = mask.shape
    best = None
    best_n = 0
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
        if len(pix) > best_n:
            best_n = len(pix)
            best = pix
    out = np.zeros_like(mask)
    for y, x in best:
        out[y, x] = True
    return out


def segment(img):
    """Partition this still. The cross, the tan ridge, and shell lava stay on the torso."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    yellow = is_any(img, (YELLOW, YELLOW2))
    gold = is_any(img, (GOLD, GOLD2))
    ink = is_any(img, (INK, INK2))

    # Mouth flame under the brow. Stray yellow lower on the shell is lava, not the jaw.
    jaw = yellow & (yy >= 26) & (yy <= 43) & (xx >= 36) & (xx <= 62)
    jaw = jaw | (dilate(jaw, 1) & gold & (yy >= 26) & (yy <= 43) & (xx >= 36) & (xx <= 62))
    jaw = jaw | (dilate(jaw, 1) & ink & (yy >= 25) & (yy <= 44) & (xx >= 34) & (xx <= 64))
    jaw = largest_component(jaw)

    # Arch under the belly. Image-left is the rear support, image-right is the fore support.
    leg_l = largest_component(a & (yy >= 68) & (xx <= 36) & ~jaw)
    leg_r = largest_component(a & (yy >= 71) & (xx >= 54) & ~jaw)
    used = jaw | leg_l | leg_r
    torso = a & ~used
    parts = {"torso": torso, "jaw": jaw, "leg_l": leg_l, "leg_r": leg_r}
    own = torso | jaw | leg_l | leg_r
    if int(own.sum()) != int(a.sum()):
        raise SystemExit(f"part masks dropped pixels {int(a.sum()) - int(own.sum())}")
    names = list(parts)
    for i, ni in enumerate(names):
        for nj in names[i + 1 :]:
            both = int((parts[ni] & parts[nj]).sum())
            if both:
                raise SystemExit(f"part overlap {ni}&{nj} {both}")
    return parts


def measure_facing(img):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max())
    cut = y0 + int(0.22 * (y1 - y0 + 1))
    top = a.copy()
    top[cut:] = False
    txs = np.nonzero(top)[1]
    if len(txs) == 0:
        raise SystemExit("top band empty")
    return float(txs.mean()) > 48.0, float(txs.mean())


def edge_pivot(mask, side):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0, 0.0
    if side == "top":
        sel = ys <= ys.min() + 1
    elif side == "bottom":
        sel = ys >= ys.max() - 1
    else:
        sel = xs >= xs.max() - 1
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
        out[sy0 + dy_i : sy1 + dy_i, sx0 + dx_i : sx1 + dx_i] = layer[sy0:sy1, sx0:sx1]
    return out


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest rotation. Copies palette pixels. Does not invent colours."""
    if abs(angle_deg) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    # Wrap into (-180, 180] so a finished circle is not resampled.
    angle_deg = (angle_deg + 180.0) % 360.0 - 180.0
    if abs(angle_deg) < 0.45:
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


def roar_rings(out, mouth, facing, age, strength):
    mx, my = mouth
    cols = (YELLOW, GOLD, RED)
    for r_i in range(strength):
        # Large enough to clear the shell. The mouth sits inside the head.
        rad = 28 + age * 7 + r_i * 9
        col = cols[min(r_i, 2)]
        for deg in range(-48, 49, 8):
            ang = math.radians(deg)
            put(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.62, col)
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
        leng = int(rng.integers(max(3, length // 2), length + 1))
        for i in range(leng):
            put(out, rear_x - facing * (off + i), y, SHELL if i < 2 else STEEL)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(size // 2, size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), YELLOW if r < size - 2 else GOLD)
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
        (cx - facing * (8 + d * 2), y0 + 6),
        (cx + facing * (6 + d), y0 + 10),
        (cx - facing * 3, y0 + 16),
    ]
    for sx, sy in pts:
        put(out, sx, sy, YELLOW)
        put(out, sx - facing, sy + 1, RED)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    rgb = out[..., :3]
    ink = a & (
        np.all(rgb == np.array(INK, np.uint8), axis=-1)
        | np.all(rgb == np.array(INK2, np.uint8), axis=-1)
    )
    out[a & ~ink, :3] = WHITE
    return out


def largest_fraction(img):
    a = img[..., 3] >= 128
    if not np.any(a):
        return 1.0, 0, 0
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
    return best / total, total, best


def ground_shock(out, foot, facing, age):
    """v3 stomp FX: flat ground rings, lava splash and rock debris from the fore foot."""
    fx, fy = foot
    gy = FOOT_Y
    cols = (WHITE, YELLOW, GOLD, RED)
    # Two flat rings on the ground, the inner one one step behind.
    for k, lag in enumerate((0, 2)):
        a = age - lag
        if a < 0:
            continue
        rx = 8 + a * 7
        ry = 2 + a * 1.2
        col = cols[min(a // 2 + k, 3)]
        for deg in range(0, 360, 6):
            ang = math.radians(deg)
            x = fx + rx * math.cos(ang)
            y = gy + 1 + ry * math.sin(ang)
            if math.sin(ang) < -0.2 and abs(x - fx) < 14:
                continue  # keep the ring from drawing over the foot
            put(out, x, y, col)
    # Lava splash: arcs up and out from the impact, falling back.
    for i, (vx, vy) in enumerate(((-2.4, -5.0), (2.6, -5.6), (-1.2, -6.8), (1.4, -7.2),
                                  (3.6, -4.0), (-3.4, -3.6), (0.4, -8.0))):
        t = age + 1
        x = fx + vx * t * 1.6
        y = fy - 2 + vy * t + 0.9 * t * t
        if y < gy:
            c1 = YELLOW if i % 2 else RED
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                put(out, x + dx, y + dy, c1 if dy == 0 else GOLD)
    # Rock debris thrown sideways along the ground.
    for i, (vx, vy) in enumerate(((-4.0, -3.0), (4.4, -2.6), (-5.6, -1.8), (6.0, -2.2))):
        t = age + 1
        x = fx + vx * t * 1.4
        y = gy - 1 + vy * t + 0.7 * t * t
        if y <= gy and age <= 5:
            for dx, dy in ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, 2)):
                put(out, x + dx, y + dy, INK if dy == 2 else (STEEL if (dx + dy) % 2 else SHELL))
    # Cracks of glowing lava on the ground for the first frames.
    if age <= 4:
        for d in (-1, 1):
            for i in range(4, 8 + age * 4):
                put(out, fx + d * i, gy + 1 + (i // 5) % 2, RED if i % 3 else GOLD)
    return out


def hip_cap(torso_mask, leg_mask, reach=3):
    """Duplicate a few torso pixels above the ankle so a short lift does not open a hole."""
    ys = np.nonzero(leg_mask)[0]
    if len(ys) == 0:
        return np.zeros_like(leg_mask)
    top = int(ys.min())
    yy = np.arange(leg_mask.shape[0])[:, None]
    band = (yy >= top - reach) & (yy < top)
    return torso_mask & dilate(leg_mask, 2) & band


def build_rig(img, masks, flip):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y : dst_y + sh, dst_x : dst_x + sw] = img[y0:y1, x0:x1]

    placed = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y : dst_y + sh, dst_x : dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm

    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr

    # Socket under the jaw, covered at rest. Shows the mouth cavity when the jaw swings.
    zone = placed["jaw"] & dilate(placed["torso"], 2)
    if np.any(zone):
        fill = nearest_color(canvas, zone, placed["torso"])
        empty = layers["torso"][..., 3] < 128
        write = zone & empty
        layers["torso"][write] = fill[write]

    for leg in ("leg_l", "leg_r"):
        cap = hip_cap(placed["torso"], placed[leg], 3)
        layers[leg][cap] = canvas[cap]

    pivots = {
        "jaw": edge_pivot(placed["jaw"], "top"),
        "leg_l": edge_pivot(placed["leg_l"], "top"),
        "leg_r": edge_pivot(placed["leg_r"], "top"),
    }
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
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
    """Feet tables are 0 at frame 0 and at the loop. One support is always down."""
    n = FRAMES[action]
    p = dict(
        body_dy=0.0,
        jaw=0.0,
        ll_dy=0,
        lr_dy=0,
        ll_dx=0,
        lr_dx=0,
        ll_sw=0.0,
        lr_sw=0.0,
        spin=0.0,
        recoil=0.0,
        slide=0.0,
        ring=-1,
        rings=0,
        lines=0,
        burst=0,
        flash=0,
        sparks=0,
        rear=0.0,
        shock=-1,
        shake=0,
    )
    if action == "idle":
        p["body_dy"] = wave(f, n, 1.0)
    elif action == "walk":
        p["body_dy"] = wave(f, n, 1.0, cycles=2.0)
        # Rear support steps in the first half, fore support in the second. Never both up.
        p["ll_dy"] = [0, -1, -3, -4, -4, -3, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["ll_dx"] = [0, 1, 2, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["ll_sw"] = [0, 6, 12, 14, 12, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["lr_dy"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -3, -4, -4, -3, -1, 0][f]
        p["lr_dx"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 2, 1, 0][f]
        p["lr_sw"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 12, 14, 12, 6, 0, 0][f]
        p["lines"] = 2
    elif action == "attack":
        # v3 stomp. Mouth flame stays fixed to the head (jaw 0 on every frame).
        # rear: nose-up degrees around the hind sole. Negative = lean into the slam.
        p["rear"] = [0, 4, 8, 11, 12, -4, -3, -2, -1, -1, 0, 0, 0, 0, 0, 0][f]
        # Fore support (leg_r in the right-facing still) lifts high, then slams.
        p["lr_dy"] = [0, -2, -4, -6, -6, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["lr_dx"] = [0, 1, 2, 2, 2, 3, 3, 2, 2, 1, 1, 1, 0, 0, 0, 0][f]
        p["lr_sw"] = [0, 3, 6, 8, 8, -3, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["body_dy"] = [0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0][f]
        p["shock"] = [-1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, -1, -1, -1, -1][f]
        p["shake"] = [0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0][f]
    else:
        env = [0, 0.45, 0.85, 1.0, 0.92, 0.7, 0.48, 0.3, 0.16, 0.08, 0.02, 0][f]
        p["slide"] = -6.5 * env
        p["recoil"] = 36.0 * env
        p["jaw"] = 0.0  # v3: the mouth flame does not swing on its own.
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def jaw_tip(layer, facing):
    m = layer[..., 3] >= 128
    if not np.any(m):
        return None
    ys, xs = np.nonzero(m)
    tip_y = int(ys.max())
    band = ys >= tip_y - 1
    tip_x = int(np.median(xs[band]))
    return tip_x + int(round(facing * 2)), tip_y


def synth(rig, action, f, palette):
    layers, pivots, center, facing = rig
    p = pose(action, f)
    # v3: gen4_attack_strike.finish() pushed the mouth flame to 54 degrees here;
    # this species now has its own stomp, so the shared hook is not applied.

    fwd = facing
    bdy = p["body_dy"]

    # Negative swing swings the sole toward +x before the facing multiply.
    def leg(name, swing, dx, dy):
        ang = -float(swing) * fwd
        return shift(rotate_nearest(layers[name], ang, pivots[name]), dx * fwd, dy)

    # Negative angle swings the hanging flame toward +x (forward when facing right).
    jaw = shift(rotate_nearest(layers["jaw"], -p["jaw"] * fwd, pivots["jaw"]), 0, bdy)
    torso = shift(layers["torso"], p["slide"] * fwd, bdy)
    # Slide the jaw and the feet with the knockback. Feet do not take the breath.
    if p["slide"]:
        jaw = shift(jaw, p["slide"] * fwd, 0)
    leg_l = leg("leg_l", p["ll_sw"], p["ll_dx"] + p["slide"], p["ll_dy"])
    leg_r = leg("leg_r", p["lr_sw"], p["lr_dx"] + p["slide"], p["lr_dy"])

    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for lyr in (torso, jaw, leg_l, leg_r):
        paste(body, lyr)

    if abs(p["rear"]) > 0.4:
        hm = layers["leg_l"][..., 3] >= 128
        hy, hx = np.nonzero(hm)
        sole = (float(hx[hy >= hy.max() - 1].mean()), float(hy.max()))
        body = rotate_nearest(body, -p["rear"] * fwd, sole)
        body = despeckle(body)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        pivot = (center[0] + p["slide"] * fwd, center[1] + bdy)
        # Nose up, away from the ground. Cap stays under 38 degrees. No belly mirror.
        ang = p["spin"] if spinning else (-p["recoil"] * fwd)
        body = rotate_nearest(body, ang, pivot)
        body = despeckle(body)

    img = snap(body, palette)
    if p["flash"]:
        img = flash_white(img)
    tip = jaw_tip(jaw, fwd)
    if p["rings"] and p["ring"] >= 0 and not spinning and tip is not None:
        img = roar_rings(img, tip, fwd, p["ring"], p["rings"])
    if p["lines"]:
        img = speed_lines(img, fwd, p["lines"], seed=f * 17 + (0 if action == "walk" else 3), length=8)
    if p["burst"] and tip is not None:
        img = impact_burst(img, (tip[0] + int(round(fwd * 4)), tip[1]), 9 if f >= 12 else 7)
    if p["sparks"]:
        img = sparks(img, f, fwd)
    if p["shock"] >= 0:
        fm = leg_r[..., 3] >= 128
        fy_, fx_ = np.nonzero(fm)
        foot = (float(fx_.mean()), float(fy_.max()))
        img = ground_shock(img, foot, fwd, p["shock"])
    if p["shake"]:
        img = shift(img, 0, p["shake"])
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
                sh[di * CELL_H : (di + 1) * CELL_H, f * CELL_W : (f + 1) * CELL_W] = tile
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
        "walk": (0, 4, 8, 12),
        "attack": (2, 4, 7, 11),
        "hurt": (1, 3, 6, 9),
    }
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W : (f + 1) * CELL_W]
            x = si * CELL_W
            y = ai * CELL_H
            m = tile[..., 3] >= 128
            bg[y : y + CELL_H, x : x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 4
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][2 * CELL_H : 3 * CELL_H, f * CELL_W : (f + 1) * CELL_W]
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
            bg[y : y + c.shape[0], x : x + c.shape[1]][m] = c[m]
    return Image.fromarray(bg, "RGBA")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def place_still(front):
    a = front[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    dst_x = (CELL_W - (x1 - x0)) // 2
    dst_y = FOOT_Y - (y1 - y0)
    placed = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    placed[dst_y : dst_y + (y1 - y0), dst_x : dst_x + (x1 - x0)] = front[y0:y1, x0:x1]
    return placed


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
    faces_right, top_x = measure_facing(front)
    if not faces_right:
        raise SystemExit(f"expected top-band centroid x > 48, got {top_x:.2f}")
    print(f"facing right top_centroid_x={top_x:.2f}", flush=True)

    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    for name, m in masks.items():
        bb_ys, bb_xs = np.nonzero(m)
        print(
            f"part {name} n={int(m.sum())} bbox x {int(bb_xs.min())}-{int(bb_xs.max())} y {int(bb_ys.min())}-{int(bb_ys.max())}",
            flush=True,
        )
    print("palette", len(palette), "sum", sum(counts.values()), flush=True)

    rigs = {}
    for d in DIRS:
        rigs[d] = build_rig(front, masks, flip=d not in FACE_RIGHT)

    placed = place_still(front)
    rest = synth(rigs["E"], "idle", 0, palette)
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    diff_n = int(differ.sum())
    print("rest-pose pixels changed", diff_n, flush=True)
    if diff_n:
        raise SystemExit("idle frame 0 does not match the still")
    rest_w = synth(rigs["W"], "idle", 0, palette)
    mirror = placed[:, ::-1].copy()
    differ_w = np.any(rest_w != mirror, axis=2) & ((rest_w[..., 3] >= 128) | (mirror[..., 3] >= 128))
    diff_w = int(differ_w.sum())
    print("mirror rest-pose pixels changed", diff_w, flush=True)
    if diff_w:
        raise SystemExit("flipped idle frame 0 does not match the mirrored still")

    both_up = []
    for f in range(16):
        pf = pose("walk", f)
        if pf["ll_dy"] < 0 and pf["lr_dy"] < 0:
            both_up.append(f)
    if both_up:
        raise SystemExit(f"walk rears onto two feet at {both_up}")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (180, 180, 190),
        "jaw": (255, 220, 40),
        "leg_l": (60, 200, 90),
        "leg_r": (60, 140, 255),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "E", f) for f in (0, 4, 8, 12)]
        want += [("walk", "E", f) for f in (0, 4, 8, 12)]
        want += [("attack", "E", f) for f in (0, 2, 4, 5, 6, 7, 8, 9, 11, 13, 15)]
        want += [("hurt", "E", f) for f in (0, 1, 3, 6, 11)]
        want += [("attack", "W", f) for f in (2, 7)]
        want += [("walk", "W", 4), ("hurt", "W", 3)]
        tiles = []
        for act, d, f in want:
            tile = synth(rigs[d], act, f, palette)
            frac, total, best = largest_fraction(tile)
            edge = tile[..., 3] >= 128
            touch = bool(edge[0].any() or edge[-1].any() or edge[:, 0].any() or edge[:, -1].any())
            print(
                f"sample {act} {d} {f:02d} main={frac:.3f} n={total} best={best} edge={touch}",
                flush=True,
            )
            tiles.append(tile)
        cols = 6
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H : (r + 1) * CELL_H, c * CELL_W : (c + 1) * CELL_W][m] = tile[m]
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
    hurt_best = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H : (di + 1) * CELL_H, f * CELL_W : (f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and d == "E" and 5 <= f <= 10:
                    frac, _n, best = largest_fraction(tile)
                    spin_frac[str(f)] = {"frac": round(frac, 3), "best": best}
                if act == "hurt" and d == "E" and f in (1, 3, 6):
                    frac, _n, best = largest_fraction(tile)
                    hurt_best[str(f)] = {"frac": round(frac, 3), "best": best}
    report = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_h,
        "off_palette_colors": len(off),
        "alphas": sorted(alphas),
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "spin_main_E": spin_frac,
        "hurt_main_E": hurt_best,
        "facing": "right",
        "top_centroid_x": round(top_x, 2),
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
                [
                    py,
                    "-I",
                    str(GAIT_LOOP),
                    "select",
                    "--frames-dir",
                    str(frames_dir / act / "S"),
                    "--fps",
                    "12",
                    "--output-dir",
                    str(od),
                    "--kind",
                    kind,
                    "--no-aids",
                ],
                capture_output=True,
                text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1200], flush=True)

    meta = {
        "generator": "forge_heatran_mega_actions",
        "version": VERSION,
        "key": "0485_heatran-mega",
        "dex": 485,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "face_right": sorted(FACE_RIGHT),
        "palette_colors": int(len(palette)),
        "off_palette_colors": len(off),
        "facing": "right",
        "top_centroid_x": round(top_x, 2),
        "source": "front.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    hard = []
    if off:
        hard.append("off_palette")
    if alphas - {0, 255}:
        hard.append("alpha")
    if clip:
        hard.append("edge")
    for info in list(spin_frac.values()) + list(hurt_best.values()):
        if info["best"] < 3000:
            hard.append("shattered")
            break
    if hard:
        print("QA hard fail", hard, flush=True)
        return

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
