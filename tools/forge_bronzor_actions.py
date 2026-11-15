#!/usr/bin/env python3
"""Bronzor (0436) 8-direction action sheet from the normal 96x96 front.png.

Measured on this still only (opaque 802, bbox x33-63 y30-66, 8 colours).
One floating hex disc. Six rim points, two yellow eyes (dark vertical slits),
and a 4px cyan glint between the eyes. No legs, no mouth, no wings.
The still faces the viewer. S SE E NE stay unflipped. N NW W SW mirror.

idle   : planted on frame 0. The disc bobs up at most 1px. Rim tips breathe.
walk   : two upward bobs, not a step. Rim tips pulse with the lift.
attack : 0 planted, 1-4 eyes and rim flare plus rings, 5-10 one-piece
         in-plane 360 (no |cos| squash), 11-12 impact, 13-15 return.
hurt   : the disc leans about its lower point, at most 38 degrees.

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
STILL_DIR = ROOT / "monsterSprite/gen4/0436_bronzor/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0436_bronzor/normal/agent-sprite-forge")
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
VERSION = "bronzor_v2"
KEY = "0436_bronzor"
UNFLIPPED = {"S", "SE", "E", "NE"}

DARK = (24, 90, 123)
MID = (41, 131, 164)
OUTLINE = (16, 57, 74)
BLACK = (16, 16, 16)
LIGHT = (65, 164, 197)
YELLOW = (255, 222, 115)
OLIVE = (148, 180, 139)
CYAN = (98, 197, 230)
EYE_COLORS = (YELLOW, OLIVE, CYAN)
PROTECT = (BLACK, OUTLINE)

# Frame 0 stays the planted still so the spin ratio has a stable base.
EYE_AMT = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
RIM_AMT = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
RINGS = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
BURST = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0]
HURT = [0.0, 0.35, 0.70, 1.0, 0.82, 0.58, 0.36, 0.18, 0.06, 0.0, 0.0, 0.0]


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


def ang_dist(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def knob_tips(op):
    """Six hex points: outer pixels of the radial maxima. Not a second body."""
    ys, xs = np.nonzero(op)
    if len(ys) == 0:
        raise SystemExit("empty still")
    cx, cy = float(xs.mean()), float(ys.mean())
    h, w = op.shape
    outline = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        edge = False
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w and op[ny, nx]):
                edge = True
                break
        if edge:
            outline.append((x, y))
    bins = [[] for _ in range(72)]
    for x, y in outline:
        ang = math.atan2(y - cy, x - cx)
        r = math.hypot(x - cx, y - cy)
        b = int(((ang + math.pi) / (2 * math.pi)) * 72) % 72
        bins[b].append(r)
    rad = np.array([max(b) if b else 0.0 for b in bins], np.float64)
    sm = np.convolve(np.r_[rad[-2:], rad, rad[:2]], np.ones(5) / 5.0, mode="valid")
    peaks = []
    for i in range(72):
        if sm[i] >= sm[(i - 1) % 72] and sm[i] > sm[(i + 1) % 72] and sm[i] > 14.0:
            deg = (i + 0.5) / 72.0 * 360.0 - 180.0
            peaks.append((float(sm[i]), deg))
    peaks.sort(reverse=True)
    chosen = []
    for _r, deg in peaks:
        if all(ang_dist(deg, other) > 36.0 for other in chosen):
            chosen.append(deg)
        if len(chosen) == 6:
            break
    if len(chosen) != 6:
        raise SystemExit(f"expected 6 rim points, got {len(chosen)} from {peaks}")
    tips = np.zeros_like(op)
    opaque_pts = list(zip(ys.tolist(), xs.tolist()))
    for deg in chosen:
        sector = []
        for y, x in opaque_pts:
            ang = math.degrees(math.atan2(y - cy, x - cx))
            r = math.hypot(x - cx, y - cy)
            if ang_dist(ang, deg) <= 16.0:
                sector.append((r, y, x))
        if not sector:
            continue
        mx = max(s[0] for s in sector)
        for r, y, x in sector:
            if r >= mx - 1.25:
                tips[y, x] = True
    return tips, chosen


def segment(img):
    op = img[..., 3] >= 128
    eye = color_mask(img, EYE_COLORS)
    comps = components(eye, eight=False)
    if len(comps) != 3:
        raise SystemExit(f"expected 2 eyes plus glint, got {len(comps)}")
    comps.sort(key=lambda m: float(np.nonzero(m)[1].mean()))
    tips, peak_degs = knob_tips(op)
    tips = tips & ~eye
    if int(tips.sum()) < 12 or int(tips.sum()) > 80:
        raise SystemExit(f"rim tips out of range: {int(tips.sum())}")
    if int((tips & eye).sum()):
        raise SystemExit("rim tips overlap the eyes")
    parts = {
        "eye_l": comps[0],
        "glint": comps[1],
        "eye_r": comps[2],
        "tips": tips,
        "disc": op,
    }
    if int(parts["eye_l"].sum()) < 10 or int(parts["eye_r"].sum()) < 10:
        raise SystemExit("an eye collapsed")
    if int(parts["glint"].sum()) < 2:
        raise SystemExit("face glint missing")
    parts["peaks"] = peak_degs
    return parts


def measure_face(img, parts):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    return {
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "opaque": int(op.sum()),
        "eyes": [int(parts["eye_l"].sum()), int(parts["glint"].sum()), int(parts["eye_r"].sum())],
        "tips": int(parts["tips"].sum()),
        "faces": "viewer",
        "limbs": 0,
        "mouth": 0,
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


def shift_mask(mask, dx, dy):
    layer = np.zeros(mask.shape + (4,), np.uint8)
    layer[mask, 3] = 255
    return shift(layer, dx, dy)[..., 3] >= 128


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
    pad = np.pad(a, 1)
    n = pad[:-2, 1:-1].astype(np.int32) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_cracks(img):
    """Close 1px cracks left by nearest rotation so the disc stays one piece."""
    out = img.copy()
    a = out[..., 3] >= 128
    pad = np.pad(a, 1)
    n = pad[:-2, 1:-1].astype(np.int32) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
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

    def place(mask):
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = mask[y0:y1, x0:x1]
        return pm

    placed = {
        "eye_l": place(masks["eye_l"]),
        "glint": place(masks["glint"]),
        "eye_r": place(masks["eye_r"]),
        "tips": place(masks["tips"]),
        "disc": place(masks["disc"]),
    }
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    foot_y = int(oys.max())
    foot_x = float(np.median(oxs[oys >= foot_y - 1]))
    foot = (foot_x, float(foot_y))
    if flip:
        canvas = canvas[:, ::-1].copy()
        placed = {k: v[:, ::-1].copy() for k, v in placed.items()}
        center = (CELL_W - 1 - center[0], center[1])
        foot = (CELL_W - 1 - foot[0], foot[1])
    return {
        "canvas": canvas,
        "placed": placed,
        "center": center,
        "foot": foot,
        "facing": 1.0 if not flip else -1.0,
    }


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def hop(f, n, amp, cycles):
    """Upward bob only. Zero on frame 0 and on the loop point."""
    return -max(0.0, wave(f, n, amp, cycles=cycles))


def pose(action, f):
    n = FRAMES[action]
    p = dict(dx=0.0, dy=0.0, eye=0, rim=0, spin=0.0, recoil=0.0, rings=0, burst=0, flash=0, sparks=0)
    if action == "idle":
        p["dy"] = hop(f, n, 1.0, 1.0)
        p["rim"] = 1 if wave(f, n, 1.0) > 0.65 else 0
    elif action == "walk":
        p["dy"] = hop(f, n, 3.0, 2.0)
        p["rim"] = 1 if wave(f, n, 1.0, cycles=2.0) > 0.45 else 0
    elif action == "attack":
        p["eye"] = int(EYE_AMT[f])
        p["rim"] = int(RIM_AMT[f])
        p["rings"] = int(RINGS[f])
        p["burst"] = int(BURST[f])
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
    else:
        env = HURT[f]
        p["recoil"] = 38.0 * env
        p["dx"] = -3.0 * env
        p["dy"] = -1.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def extend_tips(layer, tip_mask, center, amount):
    """Copy existing rim-point pixels outward. The disc underneath stays."""
    if amount <= 0:
        return layer
    out = layer.copy()
    h, w = out.shape[:2]
    ys, xs = np.nonzero(tip_mask & (layer[..., 3] >= 128))
    cx, cy = center
    for y, x in zip(ys.tolist(), xs.tolist()):
        vx, vy = x - cx, y - cy
        mag = math.hypot(vx, vy)
        if mag < 1.0:
            continue
        ux, uy = vx / mag, vy / mag
        for step in range(1, int(amount) + 1):
            nx = int(round(x + ux * step))
            ny = int(round(y + uy * step))
            if 0 <= nx < w and 0 <= ny < h and out[ny, nx, 3] < 128:
                out[ny, nx] = layer[y, x]
                # Bridge a diagonal gap so the point stays on the disc.
                mx = int(round(x + ux * (step - 0.5)))
                my = int(round(y + uy * (step - 0.5)))
                if 0 <= mx < w and 0 <= my < h and out[my, mx, 3] < 128:
                    out[my, mx] = layer[y, x]
    return out


def flare_eyes(img, masks, amount):
    """Grow each eye and the glint by whole pixels. Pupils and ink stay."""
    if amount <= 0:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    protect = np.zeros((h, w), bool)
    for c in PROTECT:
        protect |= np.all(out[..., :3] == np.array(c, np.uint8), axis=-1)
    for mask in masks:
        dil = mask.copy()
        for _ in range(int(amount)):
            pad = np.pad(dil, 1)
            dil = dil | pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
        grow = dil & ~mask & (out[..., 3] >= 128) & ~protect
        ey, ex = np.nonzero(mask)
        if len(ey) == 0 or not np.any(grow):
            continue
        cols = img[ey, ex, :3]
        gy, gx = np.nonzero(grow)
        for y, x in zip(gy.tolist(), gx.tolist()):
            d = (ey.astype(np.int32) - y) ** 2 + (ex.astype(np.int32) - x) ** 2
            out[y, x, :3] = cols[int(np.argmin(d))]
            out[y, x, 3] = 255
    return out


def compose(rig, p):
    dx_i = int(round(p["dx"] * rig["facing"]))
    dy_i = int(round(p["dy"]))
    body = shift(rig["canvas"], dx_i, dy_i)
    if p["rim"] or p["eye"]:
        center = (rig["center"][0] + dx_i, rig["center"][1] + dy_i)
        if p["rim"]:
            tips = shift_mask(rig["placed"]["tips"], dx_i, dy_i)
            body = extend_tips(body, tips, center, p["rim"])
        if p["eye"]:
            eyes = [shift_mask(rig["placed"][name], dx_i, dy_i) for name in ("eye_l", "glint", "eye_r")]
            body = flare_eyes(body, eyes, p["eye"])
    return body


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
    for i in range(count):
        ang = 2 * math.pi * i / count
        col = colors[i % len(colors)]
        put(out, cx + radius * math.cos(ang), cy + radius * math.sin(ang), col)


def roar_rings(out, center, level):
    if level <= 0:
        return out
    paint_ring(out, center, 26, 8, (YELLOW, CYAN, LIGHT, OLIVE))
    if level >= 2:
        paint_ring(out, center, 33, 8, (LIGHT, YELLOW, CYAN, OLIVE))
    return out


def impact_burst(out, center, size):
    cx, cy = center
    ticks = [(18, -6), (-18, -6), (0, -20), (0, 16), (14, -14), (-14, -14), (16, 10), (-16, 10)]
    cols = (YELLOW, CYAN, LIGHT, OLIVE)
    n = 8 if size > 1 else 6
    for i, (dx, dy) in enumerate(ticks[:n]):
        put(out, cx + dx, cy + dy, cols[i % 4])
        if size > 1:
            put(out, cx + dx + (1 if dx >= 0 else -1), cy + dy, LIGHT)
    return out


def sparks(out, frame_idx, center):
    cx, cy = center
    d = frame_idx
    pts = [(cx - 16 - d, cy - 14), (cx + 14 + d, cy - 12), (cx - 4, cy + 18)]
    for sx, sy in pts:
        put(out, sx, sy, YELLOW)
        put(out, sx + 1, sy, CYAN)
    return out


def flash_hit(img):
    out = img.copy()
    a = out[..., 3] >= 128
    keep = np.zeros(a.shape, bool)
    for c in (BLACK, OUTLINE, YELLOW, OLIVE, CYAN):
        keep |= np.all(out[..., :3] == np.array(c, np.uint8), axis=-1)
    out[a & ~keep, :3] = LIGHT
    return out


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
        out[..., :] = 0
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
    pal = {tuple(int(v) for v in c) for c in palette}
    ys, xs = np.nonzero(a)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if tuple(int(v) for v in out[y, x, :3]) not in pal:
            out[y, x] = 0
            a[y, x] = False
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

        body = compose(rig, rest)
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
        norm = ang % 360.0
        if norm > 180.0:
            norm -= 360.0
        if abs(norm) >= 0.45:
            body = despeckle(body)
            _frac, total, ncomp = largest_fraction(body)
            if ncomp > 1 or total < 780:
                body = despeckle(fill_cracks(body))
    else:
        body = compose(rig, p)

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
    pad = 40
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


def assert_in_progress():
    import os as _rebuild_os
    if _rebuild_os.environ.get("GEN4_ATTACK_REBUILD") == "1":
        return
    data = json.loads(QUEUE.read_text(encoding="utf-8"))
    for item in data["items"]:
        if item.get("id") == KEY:
            if item.get("status") != "in_progress":
                raise SystemExit(f"{KEY} status is {item.get('status')}, not drawing")
            return
    raise SystemExit(f"queue has no {KEY}")


def publish_dir8(sprite_h, foot_y):
    entry = {
        "dex": 436,
        "form": "normal",
        "variant": "bronzor",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0436_bronzor/normal",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir",
        "nf": 16,
        "cw": CELL_W,
        "ch": CELL_H,
        "w": CELL_W,
        "h": int(sprite_h),
        "fy": int(foot_y),
        "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1,
        "src": "agent-sprite-forge",
        "v": VERSION,
    }

    def edit():
        text = DIR8.read_text(encoding="utf-8")
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        newline = text.endswith("\n")
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
        if newline:
            out += "\n"
        parsed = json.loads(out[len(prefix):].strip()[:-1])
        if list(parsed) != list(data) or parsed[KEY] != entry:
            raise SystemExit("dir8 rewrite mismatched")
        for k, v in others.items():
            if parsed[k] != v:
                raise SystemExit(f"dir8 key {k} changed after dumps")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(out, encoding="utf-8")
        tmp.replace(DIR8)
        return True

    with_flock(DIR8_LOCK, edit)


def update_queue(status):
    def edit():
        fd = os.open(QUEUE, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with os.fdopen(os.dup(fd), "r", encoding="utf-8") as handle:
                raw_text = handle.read()
            data = json.loads(raw_text)
            found = False
            for item in data["items"]:
                if item.get("id") != KEY:
                    continue
                item["status"] = status
                item.pop("claim", None)
                found = True
                break
            if not found:
                raise SystemExit(f"queue has no {KEY}")
            raw = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            old = json.loads(raw_text)
            new = json.loads(raw)
            for a, b in zip(old["items"], new["items"]):
                if a.get("id") != KEY and a != b:
                    raise SystemExit(f"queue item changed {a.get('id')}")
            if len(old["items"]) != len(new["items"]):
                raise SystemExit("queue length changed")
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

    assert_in_progress()
    front_path = STILL_DIR / "front.png"
    front_hash = hashlib.sha1(front_path.read_bytes()).hexdigest()
    front = np.array(Image.open(front_path).convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    face_info = measure_face(front, masks)
    counts = {k: int(v.sum()) for k, v in masks.items() if k != "peaks"}
    print("facing", face_info, "parts", counts, "peaks", [round(p, 1) for p in masks["peaks"]], "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    print("center", rigs["S"]["center"], "foot", rigs["S"]["foot"], "facing", rigs["S"]["facing"], flush=True)

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
    part_cols = {
        "disc": (40, 110, 150),
        "tips": (255, 180, 40),
        "eye_l": (255, 230, 80),
        "eye_r": (255, 230, 80),
        "glint": (120, 220, 255),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        tiles = {(act, f): synth(rigs["S"], act, f, palette) for act in ACTIONS for f in range(FRAMES[act])}
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
        north4 = synth(rigs["N"], "idle", 4, palette)
        south4 = synth(rigs["S"], "idle", 4, palette)
        print("idle4 mirror diff", int(np.any(south4 != north4[:, ::-1], axis=2).sum()), flush=True)
        print("idle feet", [foot_row(tiles[("idle", f)]) for f in range(16)], flush=True)
        print("walk feet", [foot_row(tiles[("walk", f)]) for f in range(16)], flush=True)
        hurt = tiles[("hurt", 3)]
        frac, _t, ncomp = largest_fraction(hurt)
        print(f"hurt peak main {frac:.3f} parts {ncomp} foot {foot_row(hurt)}", flush=True)
        idle0 = tiles[("idle", 0)]
        a = idle0[..., 3] >= 128
        ys, xs = np.nonzero(a)
        pad = 40
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
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        frac, _total, ncomp = largest_fraction(tile)
        print(f"attack S {f:02d} opaque {c} ratio {r:.3f} main {frac:.3f} parts {ncomp}", flush=True)
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
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:2000], flush=True)

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
        "generator": "forge_bronzor_actions", "version": VERSION, "key": KEY,
        "dex": 436, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": foot_y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "still_faces": "viewer",
        "archetype": "disc_hover",
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
