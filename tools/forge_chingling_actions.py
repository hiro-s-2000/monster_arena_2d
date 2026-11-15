#!/usr/bin/env python3
"""Chingling (0433) 8-direction action sheet from the normal 96x96 front.png.

Measured on this still only (opaque 563, bbox x32-63 y30-66, 12 colours).
One yellow bell. Two red-and-pale ribbon curls sit on top (left x38-46 y30-44,
right x50-62 y33-47) and do not touch each other. The white shine is a 2x2 at
(41-42, 48-49), left of the body centre because the light is upper-left, not a
profile eye. The bottom is one bell mouth: a gap at y64 and a clapper at
x47-50 y65-66. No legs, no feet, no face-mouth. The still faces the viewer.
S SE E NE stay unflipped. N NW W SW mirror.

idle   : curls sway. The bell bobs up at most 1px and is planted on frame 0.
walk   : the same bell bobs up (two hops), not a step. Curls flutter.
attack : 0 planted, 1-4 curls flare out and the lower rim widens, plus rings,
         5-10 one-piece in-plane 360 (no |cos| squash), 11-12 impact, 13-15 return.
hurt   : whole bell leans about the clapper, at most 38 degrees.

Palette-locked. Alpha is only 0 or 255. Off-palette pixels are cleared.
front/back png and gif are not written.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
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
STILL_DIR = ROOT / "monsterSprite/gen4/0433_chingling/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0433_chingling/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "chingling_v2"
KEY = "0433_chingling"
UNFLIPPED = {"S", "SE", "E", "NE"}

YELLOW = (255, 213, 90)
GOLD = (189, 156, 74)
BLACK = (0, 0, 0)
RED = (230, 98, 82)
DARK = (106, 57, 49)
BROWN = (172, 82, 49)
PALE = (213, 213, 246)
INK = (98, 82, 74)
MID = (131, 106, 74)
LAV = (164, 164, 205)
HI = (255, 230, 148)
WHITE = (255, 255, 255)
RIBBON_COLS = (RED, DARK, BROWN, PALE, LAV)
BODY_COLS = (YELLOW, GOLD, HI, WHITE)

# Attack tables. Frame 0 is the planted still so the spin ratio has a stable base.
# Curl numbers are outward magnitudes in degrees. Lip is the lower-rim widen in px.
CURL_L = [0, 8, 14, 18, 12, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 0]
CURL_R = [0, 10, 16, 20, 14, 0, 0, 0, 0, 0, 0, 0, 0, 7, 3, 0]
WIDEN = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
RINGS = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
BURST = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0]
HURT = [0.0, 0.40, 0.72, 1.0, 0.84, 0.60, 0.38, 0.20, 0.08, 0.0, 0.0, 0.0]


def color_mask(img, cols):
    op = img[..., 3] >= 128
    m = np.zeros(op.shape, bool)
    for c in cols:
        m |= op & np.all(img[..., :3] == np.array(c, np.uint8), axis=-1)
    return m


def components(mask, eight=False):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    steps = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx] if eight else ((1, 0), (-1, 0), (0, 1), (0, -1))
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in steps:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        comp = np.zeros_like(mask)
        for y, x in pts:
            comp[y, x] = True
        out.append(comp)
    return out


def segment(img):
    """Two ribbon curls plus one bell. Shared outline grows onto the part it touches."""
    op = img[..., 3] >= 128
    ribbon = color_mask(img, RIBBON_COLS)
    body = color_mask(img, BODY_COLS)
    if int((ribbon & body).sum()):
        raise SystemExit("ribbon and bell colours overlap")
    # One ribbon pixel at (45, 34) only touches the left curl on a diagonal.
    comps = components(ribbon, eight=True)
    if len(comps) != 2:
        raise SystemExit(f"expected 2 ribbon curls, got {len(comps)}")
    comps.sort(key=lambda m: float(np.nonzero(m)[1].mean()))
    labels = np.zeros(op.shape, np.uint8)
    labels[comps[0]] = 1
    labels[comps[1]] = 2
    labels[body] = 3
    unknown = op & (labels == 0)
    h, w = op.shape

    def neigh_labs(y, x):
        labs = []
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and labels[ny, nx]:
                labs.append(int(labels[ny, nx]))
        return labs

    pending = list(zip(*np.nonzero(unknown)))
    guard = 0
    while pending and guard < 8:
        guard += 1
        next_pending = []
        progressed = False
        for y, x in pending:
            labs = set(neigh_labs(y, x))
            if len(labs) == 1:
                labels[y, x] = labs.pop()
                progressed = True
            else:
                next_pending.append((y, x))
        pending = next_pending
        if not progressed:
            break
    for y, x in pending:
        labs = neigh_labs(y, x)
        if not labs:
            raise SystemExit(f"orphan outline pixel {(x, y)}")
        counts = {1: 0, 2: 0, 3: 0}
        for lab in labs:
            counts[lab] += 1
        if counts[3] >= counts[1] and counts[3] >= counts[2]:
            labels[y, x] = 3
        elif counts[1] >= counts[2]:
            labels[y, x] = 1
        else:
            labels[y, x] = 2
    parts = {"curl_l": labels == 1, "curl_r": labels == 2, "bell": labels == 3}
    own = parts["curl_l"] | parts["curl_r"] | parts["bell"]
    if int(own.sum()) != int(op.sum()) or int(parts["curl_l"].sum()) < 60 or int(parts["curl_r"].sum()) < 70:
        raise SystemExit(
            f"bad partition covered {int(own.sum())} opaque {int(op.sum())} "
            f"L {int(parts['curl_l'].sum())} R {int(parts['curl_r'].sum())} bell {int(parts['bell'].sum())}"
        )
    if int(parts["bell"].sum()) < 250:
        raise SystemExit("bell collapsed")
    return parts


def measure_face(img):
    op = img[..., 3] >= 128
    white = op & np.all(img[..., :3] == np.array(WHITE, np.uint8), axis=-1)
    ys, xs = np.nonzero(op)
    wy, wx = np.nonzero(white)
    return {
        "shine": [int(v) for v in wx.tolist()],
        "shine_y": [int(v) for v in wy.tolist()],
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "opaque": int(op.sum()),
        "faces": "viewer",
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
    """4x nearest. Multiples of 360 stay exact. No |cos| squash."""
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
    """Drop only pixels whose four neighbours are all empty. After rotation only."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int32) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_cracks(img):
    """Close 1px cracks left by nearest rotation so the bell stays one piece."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int32) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    hole = (~a) & (n >= 3)
    ys, xs = np.nonzero(hole)
    h, w = a.shape
    for y, x in zip(ys.tolist(), xs.tolist()):
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and out[ny, nx, 3] >= 128:
                out[y, x] = out[ny, nx]
                break
    return out


def dilate(mask, radius):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask.copy()
    h, w = mask.shape
    out = mask.copy()
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        out[y0:y1, x0:x1] = True
    return out


def contact_pivot(curl, bell):
    touch = curl & dilate(bell, 1)
    ys, xs = np.nonzero(touch)
    if len(ys) == 0:
        ys, xs = np.nonzero(curl)
        if len(ys) == 0:
            return (0.0, 0.0)
        keep = ys >= int(ys.max()) - 2
        ys, xs = ys[keep], xs[keep]
    return (float(xs.mean()), float(ys.mean()))


def away_sign(layer, pivot, center_x):
    """+1 if a positive angle swings this curl away from the bell centre."""
    a0 = layer[..., 3] >= 128
    c0 = float(np.nonzero(a0)[1].mean())
    rot = rotate_nearest(layer, 12.0, pivot)
    a1 = rot[..., 3] >= 128
    if not np.any(a1):
        return 1.0
    c1 = float(np.nonzero(a1)[1].mean())
    if c0 <= center_x:
        return 1.0 if c1 < c0 - 0.2 else -1.0
    return 1.0 if c1 > c0 + 0.2 else -1.0


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
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm

    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr

    pivot_l = contact_pivot(placed["curl_l"], placed["bell"])
    pivot_r = contact_pivot(placed["curl_r"], placed["bell"])
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    foot_y = int(oys.max())
    foot_x = float(np.median(oxs[oys >= foot_y - 1]))
    foot = (foot_x, float(foot_y))
    sign_l = away_sign(layers["curl_l"], pivot_l, center[0])
    sign_r = away_sign(layers["curl_r"], pivot_r, center[0])

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivot_l = (CELL_W - 1 - pivot_l[0], pivot_l[1])
        pivot_r = (CELL_W - 1 - pivot_r[0], pivot_r[1])
        center = (CELL_W - 1 - center[0], center[1])
        foot = (CELL_W - 1 - foot[0], foot[1])
        canvas = canvas[:, ::-1].copy()

    facing = 1.0 if not flip else -1.0
    return {
        "layers": layers,
        "pivot_l": pivot_l,
        "pivot_r": pivot_r,
        "center": center,
        "foot": foot,
        "facing": facing,
        "sign_l": sign_l,
        "sign_r": sign_r,
        "canvas": canvas,
    }


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def hop(f, n, amp, cycles):
    """Upward bob only. Zero on frame 0 and on the loop point. Never sinks the clapper."""
    return -max(0.0, wave(f, n, amp, cycles=cycles))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, curl_l=0.0, curl_r=0.0, widen=0,
        spin=0.0, recoil=0.0, rings=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["dy"] = hop(f, n, 1.0, 1.0)
        p["curl_l"] = wave(f, n, 8.0)
        p["curl_r"] = wave(f, n, 9.0, phase=0.5)
    elif action == "walk":
        p["dy"] = hop(f, n, 3.0, 2.0)
        p["curl_l"] = wave(f, n, 11.0, cycles=2.0)
        p["curl_r"] = wave(f, n, 12.0, phase=0.4, cycles=2.0)
    elif action == "attack":
        p["curl_l"] = float(CURL_L[f])
        p["curl_r"] = float(CURL_R[f])
        p["widen"] = int(WIDEN[f])
        p["rings"] = int(RINGS[f])
        p["burst"] = int(BURST[f])
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
    else:
        env = HURT[f]
        p["recoil"] = 38.0 * env
        p["dx"] = -4.0 * env
        p["dy"] = -1.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def widen_bottom(layer, amount, foot_y):
    """Grow the existing lower silhouette outward by whole pixels. Not a squash."""
    if amount <= 0:
        return layer
    out = layer.copy()
    a = out[..., 3] >= 128
    y_lo = max(0, int(foot_y) - 7)
    for y in range(y_lo, int(foot_y) + 1):
        xs = np.nonzero(a[y])[0]
        if len(xs) < 2:
            continue
        left, right = int(xs[0]), int(xs[-1])
        for k in range(1, amount + 1):
            xl, xr = left - k, right + k
            if xl >= 2 and out[y, xl, 3] < 128:
                out[y, xl] = out[y, left]
            if xr < out.shape[1] - 2 and out[y, xr, 3] < 128:
                out[y, xr] = out[y, right]
    return out


def compose(rig, p, body_dx, body_dy):
    layers = rig["layers"]
    facing = rig["facing"]
    ang_l = facing * rig["sign_l"] * p["curl_l"]
    ang_r = facing * rig["sign_r"] * p["curl_r"]
    bell = widen_bottom(layers["bell"], p["widen"], rig["foot"][1])
    curl_l = layers["curl_l"]
    curl_r = layers["curl_r"]
    if abs(ang_l) >= 0.45:
        curl_l = despeckle(rotate_nearest(curl_l, ang_l, rig["pivot_l"]))
    if abs(ang_r) >= 0.45:
        curl_r = despeckle(rotate_nearest(curl_r, ang_r, rig["pivot_r"]))
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(body, bell)
    paste(body, curl_l)
    paste(body, curl_r)
    return shift(body, body_dx, body_dy)


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def paint_ring(out, center, radius, count, colors):
    cx, cy = center
    n = 0
    for i in range(count):
        ang = 2 * math.pi * i / count
        col = colors[i % len(colors)]
        n += put(out, cx + radius * math.cos(ang), cy + radius * 0.82 * math.sin(ang), col)
    return n


def roar_rings(out, center, level):
    if level <= 0:
        return out
    paint_ring(out, center, 26, 8, (PALE, RED, HI, YELLOW))
    if level >= 2:
        paint_ring(out, center, 32, 10, (HI, PALE, WHITE, RED))
    return out


def impact_burst(out, center, size):
    cx, cy = center
    ticks = [
        (20, -4), (-20, -4), (0, -22), (0, 16),
        (14, -14), (-14, -14), (16, 10), (-16, 10),
        (26, 2), (-26, 2),
    ]
    col = RED if size > 1 else HI
    alt = YELLOW
    for i, (dx, dy) in enumerate(ticks[: 6 + 2 * min(size, 2)]):
        put(out, cx + dx, cy + dy, col if i % 2 == 0 else alt)
        if size > 1 and i < 6:
            put(out, cx + dx + (1 if dx >= 0 else -1), cy + dy, WHITE)
    return out


def sparks(out, frame_idx, center):
    cx, cy = center
    d = frame_idx
    pts = [(cx - 18 - d, cy - 16), (cx + 16 + d, cy - 14), (cx - 6, cy + 18)]
    for sx, sy in pts:
        put(out, sx, sy, HI)
        put(out, sx + 1, sy, RED)
    return out


def flash_hit(img):
    out = img.copy()
    a = out[..., 3] >= 128
    keep = a & (
        np.all(out[..., :3] == np.array(BLACK, np.uint8), axis=-1)
        | np.all(out[..., :3] == np.array(WHITE, np.uint8), axis=-1)
    )
    out[a & ~keep, :3] = HI
    return out


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
        out[..., 3] = 0
        out[~a] = 0
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
    # mapped rows are palette entries, so every opaque pixel is on-palette.
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    out[~a] = 0
    return out


def largest_fraction(img):
    a = img[..., 3] >= 128
    if not np.any(a):
        return 1.0, 0, 0
    seen = np.zeros_like(a)
    h, w = a.shape
    best = 0
    ncomp = 0
    total = int(a.sum())
    ys, xs = np.nonzero(a)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        ncomp += 1
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
    return best / total, total, ncomp


def synth(rig, action, f, palette, effects=True):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        rest = pose(action, 0)
        rest = gen4_attack_strike.finish(rest, action, 0, __file__)

        body = compose(rig, rest, 0, 0)
        if recoiling:
            dx_i = int(round(p["dx"] * rig["facing"]))
            dy_i = int(round(p["dy"]))
            body = shift(body, dx_i, dy_i)
            pivot = (rig["foot"][0] + dx_i, rig["foot"][1] + dy_i)
            ang = -p["recoil"] * rig["facing"]
        else:
            pivot = rig["center"]
            ang = p["spin"] * rig["facing"]
        body = rotate_nearest(body, ang, pivot)
        # 360 lands on the still. Do not fill cracks into that exact frame.
        norm = ang % 360.0
        if norm > 180.0:
            norm -= 360.0
        if abs(norm) >= 0.45:
            body = fill_cracks(body)
            body = despeckle(body)
    else:
        dx_i = int(round(p["dx"] * rig["facing"]))
        dy_i = int(round(p["dy"]))
        body = compose(rig, p, dx_i, dy_i)

    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_hit(img)
    if p["rings"] and not spinning:
        img = roar_rings(img, rig["center"], p["rings"])
    if p["burst"] and not spinning:
        img = impact_burst(img, rig["center"], p["burst"])
    if p["sparks"]:
        img = sparks(img, f, rig["center"])
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
    """South rows in one fixed window so the bob, spin, and lean stay in place."""
    idle0 = sheets["idle"][:CELL_H, :CELL_W]
    a = idle0[..., 3] >= 128
    ys, xs = np.nonzero(a)
    pad = 36
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad)
    ch, cw = y1 - y0, x1 - x0
    bg = np.full((len(ACTIONS) * ch, 16 * cw, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for f in range(FRAMES[act]):
            tile = sheets[act][y0:y1, f * CELL_W + x0:f * CELL_W + x1]
            m = tile[..., 3] >= 128
            bg[ai * ch:(ai + 1) * ch, f * cw:(f + 1) * cw][m] = tile[m]
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


def foot_row(tile):
    ys = np.nonzero(tile[..., 3] >= 128)[0]
    return int(ys.max()) if len(ys) else -1


def with_flock(lock_path, fn):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def publish_dir8(sprite_h, foot_y):
    entry = {
        "dex": 433, "form": "normal", "variant": "chingling", "gen": "gen4",
        "dir": "monsterSprite/gen4/0433_chingling/normal",
        "sheetDir": "agent-sprite-forge", "layout": "actions_8dir", "nf": 16,
        "cw": CELL_W, "ch": CELL_H, "w": CELL_W, "h": int(sprite_h), "fy": int(foot_y),
        "an": [16, 16, 16, 12], "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        raw = text[len(prefix):].strip()
        if raw.endswith(";"):
            raw = raw[:-1]
        data = json.loads(raw)
        if KEY not in data:
            raise SystemExit(f"{KEY} missing from dir8")
        others = {k: data[k] for k in data if k != KEY}
        data[KEY] = entry
        for k, v in others.items():
            if data[k] != v:
                raise SystemExit(f"dir8 key {k} changed")
        out = prefix + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";"
        parsed = json.loads(out[len(prefix):-1])
        if list(parsed) != list(data) or parsed[KEY] != entry:
            raise SystemExit("dir8 rewrite mismatched")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(out)
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
                raise SystemExit("queue has no 0433_chingling")
            raw = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            tmp = QUEUE.with_suffix(".json.tmp")
            tmp.write_text(raw, encoding="utf-8")
            os.replace(tmp, QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front_path = STILL_DIR / "front.png"
    front_hash = hashlib.sha1(front_path.read_bytes()).hexdigest()
    front = np.array(Image.open(front_path).convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    face_info = measure_face(front)
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("facing", face_info, "parts", counts, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    print(
        "signs", rigs["S"]["sign_l"], rigs["S"]["sign_r"],
        "pivots", rigs["S"]["pivot_l"], rigs["S"]["pivot_r"],
        "center", rigs["S"]["center"], "foot", rigs["S"]["foot"],
        flush=True,
    )

    rest = synth(rigs["S"], "idle", 0, palette, effects=False)
    placed = rigs["S"]["canvas"]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    mirrored = synth(rigs["N"], "idle", 0, palette, effects=False)
    if not np.array_equal(rest, mirrored[:, ::-1]):
        raise SystemExit("north idle 0 is not the mirror of south")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {"curl_l": (220, 60, 70), "curl_r": (70, 90, 200), "bell": (240, 200, 60)}
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "S", f) for f in range(16)]
        want += [("walk", "S", f) for f in range(16)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in range(12)]
        tiles = {(act, f): synth(rigs["S"], act, f, palette) for act, _d, f in want}
        base = int((tiles[("attack", 0)][..., 3] == 255).sum())
        print("ATTACK_S_RATIOS", flush=True)
        ratios = []
        for f in range(16):
            tile = tiles[("attack", f)]
            n = int((tile[..., 3] == 255).sum())
            frac, _total, ncomp = largest_fraction(tile)
            ratio = n / base
            ratios.append(ratio)
            print(f"attack S {f:02d} opaque {n} ratio {ratio:.3f} main {frac:.3f} parts {ncomp}", flush=True)
        print(f"min {min(ratios):.3f} max {max(ratios):.3f}", flush=True)
        idle4 = tiles[("idle", 4)]
        north4 = synth(rigs["N"], "idle", 4, palette, effects=False)
        south4 = synth(rigs["S"], "idle", 4, palette, effects=False)
        print("idle4 mirror diff", int(np.any(south4 != north4[:, ::-1], axis=2).sum()), flush=True)
        print("idle feet", [foot_row(tiles[("idle", f)]) for f in range(16)], flush=True)
        print("walk feet", [foot_row(tiles[("walk", f)]) for f in (0, 2, 4, 6)], flush=True)
        hurt = tiles[("hurt", 3)]
        frac, _t, ncomp = largest_fraction(hurt)
        print(f"hurt peak main {frac:.3f} parts {ncomp} foot {foot_row(hurt)}", flush=True)
        # Fixed window of the south actions, zoomed so the bell reads.
        idle0 = tiles[("idle", 0)]
        a = idle0[..., 3] >= 128
        ys, xs = np.nonzero(a)
        pad = 34
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(CELL_H, int(ys.max()) + 1 + pad)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(CELL_W, int(xs.max()) + 1 + pad)
        ch, cw = y1 - y0, x1 - x0
        sheet = np.full((4 * ch, 16 * cw, 4), (15, 20, 31, 255), np.uint8)
        for ai, act in enumerate(ACTIONS):
            for f in range(FRAMES[act]):
                tile = tiles[(act, f)][y0:y1, x0:x1]
                m = tile[..., 3] >= 128
                sheet[ai * ch:(ai + 1) * ch, f * cw:(f + 1) * cw][m] = tile[m]
        zoom = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
        Image.fromarray(zoom, "RGBA").save(work / "samples.png")
        print("samples", work / "samples.png", flush=True)
        bad = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
        if bad:
            raise SystemExit(f"sample ratios outside 0.90-1.08: {bad}")
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

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    foot_y = int(nys.max()) if len(nys) else FOOT_Y

    counts_s, ratios = south_attack_ratios(sheets)
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(counts_s, ratios)):
        print(f"attack S {f:02d} opaque {c} ratio {r:.3f}", flush=True)
    print(f"min {min(ratios):.3f} max {max(ratios):.3f} h {sprite_h} fy {foot_y}", flush=True)

    if hashlib.sha1(front_path.read_bytes()).hexdigest() != front_hash:
        raise SystemExit("front.png changed")

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)} if np.any(opaque) else set()
    off = sorted(used - pal)
    alphas = sorted({int(v) for v in np.unique(master[..., 3])})
    clip = {}
    spin_stats = []
    hurt_stats = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and d == "S" and 5 <= f <= 10:
                    frac, total, ncomp = largest_fraction(tile)
                    spin_stats.append({"f": f, "opaque": total, "main": round(frac, 3), "parts": ncomp})
                if act == "hurt" and d == "S" and f == 3:
                    frac, total, ncomp = largest_fraction(tile)
                    hurt_stats.append({"opaque": total, "main": round(frac, 3), "parts": ncomp})
    bad_ratio = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "cell": [CELL_W, CELL_H, foot_y],
        "off_palette_colors": len(off), "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "attack_s_ratios": [round(r, 3) for r in ratios],
        "attack_s_min": round(min(ratios), 3), "attack_s_max": round(max(ratios), 3),
        "spin_body": spin_stats, "hurt_peak": hurt_stats, "rest_diff": 0,
    }
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
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1800], flush=True)

    bad_spin = [s for s in spin_stats if s["main"] < 0.90]
    bad_hurt = [s for s in hurt_stats if s["main"] < 0.90]
    failed = bool(off or clip or alphas != [0, 255] or bad_ratio or bad_spin or bad_hurt)
    if failed:
        if args.publish:
            update_queue("needs_fix")
        raise SystemExit(
            f"qa failed off={len(off)} clip={clip} alphas={alphas} ratios={bad_ratio} spin={bad_spin} hurt={bad_hurt}"
        )

    meta = {
        "generator": "forge_chingling_actions", "version": VERSION, "key": KEY,
        "dex": 433, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": foot_y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "still_faces": "viewer",
        "archetype": "bell_hover",
        "face": face_info, "parts": counts,
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "attack_s_min": round(min(ratios), 3), "attack_s_max": round(max(ratios), 3),
        "attack_s_ratios": [round(r, 3) for r in ratios],
        "source": "front.png 96x96", "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
        print("arena sheet bytes", (SYNC_OUT / "sheet.png").stat().st_size, flush=True)
    if args.publish:
        publish_dir8(sprite_h, foot_y)
        update_queue("done")
        print("published", KEY, "h", sprite_h, "cw", CELL_W, "ch", CELL_H, "fy", foot_y, flush=True)


if __name__ == "__main__":
    main()
