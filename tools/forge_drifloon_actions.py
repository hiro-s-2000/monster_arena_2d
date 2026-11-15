#!/usr/bin/env python3
"""Drifloon (0425) 8-direction action sheet from the normal 96x96 front.png.

Measured on this still only (opaque 736, bbox x31-63 y21-76, 12 colours, alpha 0/255).
Cloud puff y<=28 (118). Purple balloon y29-55 (528) with the yellow mark on its
left-centre. Two strings y>=56 (90) end in yellow hands at y>=70. No mouth and
no legs. Both hands and the mark are visible, so the still faces the viewer.
S SE E NE stay unflipped. N NW W SW mirror.

idle   : balloon hovers. String tips stay on the idle-0 pixels. Frame 0 is the still.
walk   : the same bob, two cycles. Not a biped stomp.
attack : 0-4 balloon swell plus rings, 5-10 one-piece in-plane 360 spin,
         11-12 impact, 13-15 return. No |cos| squash.
hurt   : rigid recoil of at most 38 degrees.

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
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
STILL_DIR = ROOT / "monsterSprite/gen4/0425_drifloon/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0425_drifloon/normal/agent-sprite-forge")
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
VERSION = "drifloon_v2"
KEY = "0425_drifloon"
UNFLIPPED = {"S", "SE", "E", "NE"}
MAX_RECOIL = 38.0

INK = (16, 16, 16)
YELLOW = (255, 222, 8)
WHITE = (255, 255, 255)
PALE = (197, 213, 230)

# Linear swell of the balloon. Frames 5-10 are the rigid spin, so swell is 1 there.
SWELL = [1.00, 1.02, 1.03, 1.03, 1.02, 1, 1, 1, 1, 1, 1, 1, 1, 1.015, 1.01, 1.00]
# Ring age. -1 draws none. Frame 0 and frame 15 match, so the loop meets.
RING = [-1, 0, 1, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1]
HURT_ENV = [0.0, 0.42, 0.74, 1.0, 0.82, 0.55, 0.32, 0.16, 0.06, 0.0, 0.0, 0.0]
EIGHT = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def components(mask):
    seen = np.zeros(mask.shape, bool)
    h, w = mask.shape
    groups = []
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
            for dy, dx in EIGHT:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        groups.append(pts)
    groups.sort(key=len, reverse=True)
    return groups


def largest_fraction(img):
    mask = img[..., 3] >= 128
    total = int(mask.sum())
    if total == 0:
        return 1.0, 0
    groups = components(mask)
    return len(groups[0]) / total, total


def segment(img):
    """Cloud, balloon, and two strings. The yellow mark stays on the balloon."""
    op = img[..., 3] >= 128
    if int(op.sum()) != 736:
        raise SystemExit(f"expected 736 opaque pixels, got {int(op.sum())}")
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    cloud = op & (yy <= 28)
    strings = op & (yy >= 56)
    balloon = op & ~cloud & ~strings
    groups = components(strings)
    if len(groups) != 2:
        raise SystemExit(f"expected two strings, found {len(groups)}")
    groups.sort(key=lambda pts: float(np.mean([p[1] for p in pts])))
    shaft_l = np.zeros_like(op)
    shaft_r = np.zeros_like(op)
    tip_l = np.zeros_like(op)
    tip_r = np.zeros_like(op)
    for mask_s, mask_t, pts in ((shaft_l, tip_l, groups[0]), (shaft_r, tip_r, groups[1])):
        for y, x in pts:
            if y >= 70:
                mask_t[y, x] = True
            else:
                mask_s[y, x] = True
    parts = {
        "cloud": cloud,
        "balloon": balloon,
        "shaft_l": shaft_l,
        "shaft_r": shaft_r,
        "tip_l": tip_l,
        "tip_r": tip_r,
    }
    own = np.zeros_like(op)
    overlap = 0
    for name in parts:
        overlap += int((own & parts[name]).sum())
        own |= parts[name]
    if int(own.sum()) != int(op.sum()) or overlap:
        raise SystemExit(
            f"masks do not partition the still: covered {int(own.sum())} opaque {int(op.sum())} overlap {overlap}"
        )
    for name, need in (("cloud", 80), ("balloon", 400), ("shaft_l", 8), ("shaft_r", 8), ("tip_l", 12), ("tip_r", 12)):
        if int(parts[name].sum()) < need:
            raise SystemExit(f"{name} collapsed: {int(parts[name].sum())}")
    return parts


def measure_face(img, masks):
    """Both hands and an interior yellow mark: the still faces the viewer."""
    op = img[..., 3] >= 128
    yellow = op & np.all(img[..., :3] == np.array(YELLOW, np.uint8), axis=-1)
    mark = int((yellow & ~masks["tip_l"] & ~masks["tip_r"]).sum())
    if mark < 40:
        raise SystemExit(f"yellow mark on the balloon is missing ({mark})")
    ys, xs = np.nonzero(op)
    my, mx = np.nonzero(yellow & ~masks["tip_l"] & ~masks["tip_r"])
    if int(mx.min()) <= int(xs.min()) or int(mx.max()) >= int(xs.max()):
        raise SystemExit("yellow mark touches the silhouette edge; facing must be remeasured")
    for name in ("tip_l", "tip_r"):
        if int((yellow & masks[name]).sum()) < 4:
            raise SystemExit(f"{name} has no yellow hand")
    return {
        "faces": "viewer",
        "mark": mark,
        "centroid_x": round(float(xs.mean()), 2),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
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
    """4x nearest. A multiple of 360 stays exact."""
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


def forward_map(layer, angle_deg, pivot):
    """Same transform as rotate_nearest, source pixel to dest cell."""
    ang = math.radians(angle_deg)
    c, s = math.cos(ang), math.sin(ang)
    scale = 4
    px, py = pivot[0] * scale + 1.5, pivot[1] * scale + 1.5
    dest = {}
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    for y, x in zip(ys.tolist(), xs.tolist()):
        sx, sy = x * scale + 1.5, y * scale + 1.5
        dx, dy = sx - px, sy - py
        ox = px + c * dx - s * dy
        oy = py + s * dx + c * dy
        dest[(y, x)] = (int(np.rint((oy - 1.5) / scale)), int(np.rint((ox - 1.5) / scale)))
    return dest


def neighbor_edges(mask):
    pix = set((int(y), int(x)) for y, x in zip(*np.nonzero(mask)))
    edges = []
    for y, x in pix:
        for dy, dx in EIGHT:
            q = (y + dy, x + dx)
            if q in pix and (y, x) < q:
                edges.append(((y, x), q))
    return edges


def rotate_keep(layer, angle_deg, pivot):
    """4x nearest, then put back string pixels the downsample dropped.

    The diagonal strings are one pixel wide. Inverse 4x splits them off the
    balloon. Stamping each source pixel and bridging a 1px neighbor gap keeps
    the whole body one piece. This is not a |cos| squash.
    """
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    base = rotate_nearest(layer, angle_deg, pivot)
    dest = forward_map(layer, angle_deg, pivot)
    img = base
    h, w = img.shape[:2]
    for (y, x), (yy, xx) in dest.items():
        if 0 <= yy < h and 0 <= xx < w and img[yy, xx, 3] < 128:
            img[yy, xx] = layer[y, x]
    for a, b in neighbor_edges(layer[..., 3] >= 128):
        ya, xa = dest[a]
        yb, xb = dest[b]
        if max(abs(ya - yb), abs(xa - xb)) != 2:
            continue
        my = int(round((ya + yb) / 2))
        mx = int(round((xa + xb) / 2))
        if 0 <= my < h and 0 <= mx < w and img[my, mx, 3] < 128 and img[ya, xa, 3] >= 128 and img[yb, xb, 3] >= 128:
            img[my, mx] = img[ya, xa]
    return img


def scale_nearest(layer, factor, pivot):
    if abs(factor - 1.0) < 0.004 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    scale = 4
    up = np.repeat(np.repeat(layer, scale, axis=0), scale, axis=1)
    px, py = pivot[0] * scale + 1.5, pivot[1] * scale + 1.5
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    src_x = np.rint(px + (xx - px) / factor).astype(np.int32)
    src_y = np.rint(py + (yy - py) / factor).astype(np.int32)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    out[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    return out[scale // 2::scale, scale // 2::scale].copy()


def despeckle(img):
    """Drop only 8-isolated specks. A 4-neighbor test erases the diagonal strings."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n8 = (
        p[:-2, :-2].astype(np.int16) + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    )
    out[a & (n8 == 0)] = 0
    return out


def extract_palette(img):
    opaque = img[..., 3] >= 128
    return np.unique(img[opaque][:, :3], axis=0)


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
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def top_pivot(mask):
    ys, xs = np.nonzero(mask)
    top = int(ys.min())
    sel = ys == top
    return float(xs[sel].mean()), float(top)


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
    layers = {}
    for name, mask in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = mask[y0:y1, x0:x1]
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        placed[name] = pm
        layers[name] = lyr
    env = np.zeros_like(canvas)
    env[placed["cloud"] | placed["balloon"]] = canvas[placed["cloud"] | placed["balloon"]]
    layers["envelope"] = env

    pivots = {"shaft_l": top_pivot(placed["shaft_l"]), "shaft_r": top_pivot(placed["shaft_r"])}
    eys, exs = np.nonzero(placed["cloud"] | placed["balloon"])
    env_center = (float(exs.mean()), float(eys.mean()))
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    tip = placed["tip_l"] | placed["tip_r"]

    if flip:
        def fl(arr):
            return arr[:, ::-1].copy()

        layers = {k: fl(v) for k, v in layers.items()}
        placed = {k: fl(v) for k, v in placed.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
        env_center = (CELL_W - 1 - env_center[0], env_center[1])
        canvas = fl(canvas)
        tip = tip[:, ::-1].copy()

    return {
        "layers": layers,
        "pivots": pivots,
        "center": center,
        "env_center": env_center,
        "facing": -1.0 if flip else 1.0,
        "canvas": canvas,
        "tip": tip,
    }


def pose(action, f):
    p = {
        "dx": 0.0, "dy": 0.0, "sway": 0.0, "swell": 1.0,
        "spin": 0.0, "recoil": 0.0, "ring": -1, "burst": 0, "sparks": 0,
    }
    n = FRAMES[action]
    if action == "idle":
        p["dy"] = wave(f, n, 1.5, phase=math.pi / 2)
        p["sway"] = wave(f, n, 4.0)
    elif action == "walk":
        p["dy"] = wave(f, n, 2.0, phase=math.pi / 2, cycles=2.0)
        p["dx"] = wave(f, n, 1.0, cycles=2.0)
        p["sway"] = wave(f, n, 5.0, cycles=2.0)
    elif action == "attack":
        p["swell"] = SWELL[f]
        p["ring"] = RING[f]
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
            p["swell"] = 1.0
            p["ring"] = -1
        if f == 11:
            p["burst"] = 2
        elif f == 12:
            p["burst"] = 1
    else:
        env = HURT_ENV[f]
        p["recoil"] = min(MAX_RECOIL, MAX_RECOIL * env)
        p["dx"] = -3.0 * env
        p["dy"] = -1.0 * env
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    if p["recoil"] > MAX_RECOIL + 0.01:
        raise SystemExit("recoil exceeded 38 degrees")
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def bresenham(y0, x0, y1, x1):
    pts = []
    dy, dx = abs(y1 - y0), abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy
    y, x = y0, x0
    while True:
        pts.append((y, x))
        if y == y1 and x == x1 or len(pts) > 12:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return pts


def link_pair(img, a_mask, b_mask):
    """Paint the gap between two parts that the still had touching. Short gaps only."""
    if not np.any(a_mask) or not np.any(b_mask):
        return
    ay, ax = np.nonzero(a_mask)
    by, bx = np.nonzero(b_mask)
    cheb = np.maximum(np.abs(ay[:, None] - by[None, :]), np.abs(ax[:, None] - bx[None, :]))
    flat = int(cheb.argmin())
    i, j = divmod(flat, cheb.shape[1])
    dist = int(cheb[i, j])
    if dist <= 1 or dist > 6:
        return
    y0, x0 = int(ay[i]), int(ax[i])
    y1, x1 = int(by[j]), int(bx[j])
    col = img[y0, x0].copy()
    for y, x in bresenham(y0, x0, y1, x1)[1:-1]:
        if img[y, x, 3] < 128:
            img[y, x] = col


def compose(rig, p):
    dx = int(round(p["dx"] * rig["facing"]))
    dy = int(round(p["dy"]))
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    env = rig["layers"]["envelope"]
    if abs(p["swell"] - 1.0) >= 0.004:
        env = scale_nearest(env, p["swell"], rig["env_center"])
    env = shift(env, dx, dy)
    paste(body, env)
    moved = {}
    for name in ("shaft_l", "shaft_r"):
        layer = shift(rig["layers"][name], dx, dy)
        if abs(p["sway"]) >= 0.45:
            px, py = rig["pivots"][name]
            layer = rotate_keep(layer, p["sway"] * rig["facing"], (px + dx, py + dy))
            layer = despeckle(layer)
        moved[name] = layer
        paste(body, layer)
    # Tips are planted on the idle-0 pixels. Paste them last so a bob cannot drag them.
    for name in ("tip_l", "tip_r"):
        paste(body, rig["layers"][name])
    if dx or dy or abs(p["swell"] - 1.0) >= 0.004 or abs(p["sway"]) >= 0.45:
        env_a = env[..., 3] >= 128
        for name, tip_name in (("shaft_l", "tip_l"), ("shaft_r", "tip_r")):
            shaft_a = moved[name][..., 3] >= 128
            link_pair(body, env_a, shaft_a)
            link_pair(body, shaft_a, rig["layers"][tip_name][..., 3] >= 128)
    return body


def roar_rings(out, center, age):
    cx, cy = center
    cols = (YELLOW, WHITE, PALE)
    radii = (28, 34, 40)
    for ri in range(int(age) + 1):
        rad = radii[ri]
        col = cols[ri]
        deg = 0
        while deg < 360:
            ang = math.radians(deg)
            put(out, cx + rad * math.cos(ang), cy + rad * math.sin(ang), col)
            deg += 45
    return out


def impact_burst(out, center, size):
    cx, cy = center
    ticks = ((30, -6), (-30, -6), (0, -34), (22, -20), (-22, -20), (24, 10), (-24, 10), (0, 18))
    col = YELLOW if size > 1 else WHITE
    for dx, dy in ticks[: 6 + size]:
        put(out, cx + dx, cy + dy, col)
        if size > 1:
            put(out, cx + dx, cy + dy - 1, PALE)
    return out


def sparks(out, center, frame_idx):
    cx, cy = center
    d = frame_idx
    for sx, sy in ((cx - 24 - d, cy - 28), (cx + 24 + d, cy - 26), (cx, cy - 36)):
        put(out, sx, sy, WHITE)
        put(out, sx, sy + 1, PALE)
    return out


def synth(rig, action, f, palette, effects=True):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        rest = {k: 0.0 for k in p}
        rest["swell"] = 1.0
        rest["ring"] = -1
        body = compose(rig, rest)
        dx = int(round(p["dx"] * rig["facing"]))
        dy = int(round(p["dy"]))
        if dx or dy:
            body = shift(body, dx, dy)
        pivot = (rig["center"][0] + dx, rig["center"][1] + dy)
        ang = -p["recoil"] * rig["facing"] if recoiling else p["spin"] * rig["facing"]
        body = rotate_keep(body, ang, pivot)
        body = despeckle(body)
    else:
        body = compose(rig, p)
    img = snap(body, palette)
    if not effects:
        return img
    if p["ring"] >= 0 and not spinning:
        img = roar_rings(img, rig["env_center"], p["ring"])
    if p["burst"]:
        img = impact_burst(img, rig["env_center"], p["burst"])
    if p["sparks"]:
        img = sparks(img, rig["env_center"], f)
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
    """South row, one fixed window, so the bob is not recentered away from the tips."""
    pad = 6
    rows = []
    boxes = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            row.append(tile)
            a = tile[..., 3] >= 128
            if np.any(a):
                ys, xs = np.nonzero(a)
                boxes.append((int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())))
        rows.append(row)
    y0 = max(0, min(b[0] for b in boxes) - pad)
    y1 = min(CELL_H, max(b[1] for b in boxes) + 1 + pad)
    x0 = max(0, min(b[2] for b in boxes) - pad)
    x1 = min(CELL_W, max(b[3] for b in boxes) + 1 + pad)
    ch, cw = y1 - y0, x1 - x0
    cols = max(FRAMES.values())
    bg = np.full((len(ACTIONS) * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    for ri, row in enumerate(rows):
        for ci, tile in enumerate(row):
            crop = tile[y0:y1, x0:x1]
            m = crop[..., 3] >= 128
            bg[ri * ch:(ri + 1) * ch, ci * cw:(ci + 1) * cw][m] = crop[m]
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
    work = SYNC_OUT / "_work"
    if work.exists():
        raise SystemExit("refusing to treat arena _work as a sheet")
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
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, j + 1
    raise SystemExit(f"dir8 entry did not close: {key}")


def with_flock(lock_path, fn):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def publish_dir8(sprite_h):
    entry = {
        "dex": 425,
        "form": "normal",
        "variant": "drifloon",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0425_drifloon/normal",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir",
        "nf": 16,
        "cw": CELL_W,
        "ch": CELL_H,
        "w": CELL_W,
        "h": int(sprite_h),
        "fy": FOOT_Y,
        "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1,
        "src": "agent-sprite-forge",
        "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        keys = re.findall(r'"(\d{4}_[^"]+)":\{', text)
        if keys.count(KEY) != 1:
            raise SystemExit("0425_drifloon key missing or duplicated")
        idx = keys.index(KEY)
        neighbor = keys[idx + 1] if idx + 1 < len(keys) else keys[idx - 1]
        other = keys[idx - 1] if idx else neighbor
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
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(text)
        tmp.replace(DIR8)
        return neighbor

    return with_flock(DIR8_LOCK, edit)


def update_queue(status):
    def edit():
        text = QUEUE.read_text()
        data = json.loads(text)
        before = {item.get("id"): (item.get("status"), item.get("claim")) for item in data["items"]}
        token = '"id": "0425_drifloon"'
        pos = text.index(token)
        obj_start = text.rfind("{", 0, pos)
        depth = 0
        obj_end = None
        for j in range(obj_start, len(text)):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    obj_end = j + 1
                    break
        if obj_end is None:
            raise SystemExit("queue object did not close")
        block = text[obj_start:obj_end]
        block2, n_status = re.subn(r'"status"\s*:\s*"[^"]*"', f'"status": "{status}"', block, count=1)
        if n_status != 1:
            raise SystemExit("queue status missing")
        block3, n_claim = re.subn(r',\s*"claim"\s*:\s*"[^"]*"', "", block2, count=1)
        if not n_claim:
            block3, n_claim = re.subn(r'"claim"\s*:\s*"[^"]*"\s*,\s*', "", block2, count=1)
        text = text[:obj_start] + block3 + text[obj_end:]
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
        tmp = QUEUE.with_suffix(".json.tmp")
        tmp.write_text(text if text.endswith("\n") else text + "\n")
        tmp.replace(QUEUE)

    with_flock(QUEUE_LOCK, edit)


def qa_report(master, sheets, palette, ratios, sprite_h, rig):
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted({int(v) for v in np.unique(master[..., 3])})
    clip = {}
    spin_stats = []
    hurt_widths = []
    rest_w = None
    tip_drift = []
    for act in ACTIONS:
        sheet = sheets[act]
        for di, direction in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sheet[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{direction}:{f}")
                if act == "attack" and direction == "S" and 5 <= f <= 10:
                    frac, total = largest_fraction(tile)
                    spin_stats.append({"f": f, "opaque": total, "main": round(frac, 3)})
                if act == "hurt" and direction == "S":
                    xs = np.nonzero(c)[1]
                    width = int(xs.max() - xs.min() + 1) if len(xs) else 0
                    if f == 0:
                        rest_w = width
                    hurt_widths.append(width)
                if direction == "S" and act in ("idle", "walk"):
                    if not np.array_equal(tile[rig["tip"]], rig["canvas"][rig["tip"]]):
                        tip_drift.append(f"{act}:{f}")
    bad_ratio = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
    squash = False
    if rest_w:
        squash = min(hurt_widths) < int(rest_w * 0.85)
    return {
        "version": VERSION,
        "sprite_h": sprite_h,
        "off_palette_colors": len(off),
        "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "attack_s_ratios": [round(r, 3) for r in ratios],
        "attack_s_min": round(min(ratios), 3),
        "attack_s_max": round(max(ratios), 3),
        "spin_body": spin_stats,
        "hurt_squash": squash,
        "tip_drift": tip_drift,
        "rest_diff": 0,
        "_off": off,
        "_clip": clip,
        "_bad_ratio": bad_ratio,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(STILL_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    face_info = measure_face(front, masks)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("facing", face_info, "parts", counts, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    rest = synth(rigs["S"], "idle", 0, palette, effects=False)
    placed = rigs["S"]["canvas"]
    if not np.array_equal(rest, placed):
        differ = int((np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))).sum())
        raise SystemExit(f"idle frame 0 does not match the still ({differ})")
    west = synth(rigs["W"], "idle", 0, palette, effects=False)
    if not np.array_equal(west, rigs["W"]["canvas"]):
        raise SystemExit("mirrored idle frame 0 does not match the mirrored still")
    print("rest-pose pixels changed 0", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "cloud": (197, 213, 230), "balloon": (148, 106, 230),
        "shaft_l": (16, 16, 16), "shaft_r": (90, 90, 98),
        "tip_l": (255, 222, 8), "tip_r": (205, 180, 0),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        sheets = {}
        for act in ACTIONS:
            sheet = np.zeros((CELL_H, max(FRAMES.values()) * CELL_W, 4), np.uint8)
            for f in range(FRAMES[act]):
                sheet[0:CELL_H, f * CELL_W:(f + 1) * CELL_W] = synth(rigs["S"], act, f, palette)
            sheets[act] = sheet
        contact = motion_contact(sheets)
        contact.save(work / "samples_motion.png")
        _counts, ratios = south_attack_ratios(sheets)
        print("ATTACK_S_RATIOS", flush=True)
        for f, (c, r) in enumerate(zip(_counts, ratios)):
            frac, _total = largest_fraction(sheets["attack"][:, f * CELL_W:(f + 1) * CELL_W])
            print(f"attack S {f:02d} opaque {c} ratio {r:.3f} main {frac:.3f}", flush=True)
        print(f"min {min(ratios):.3f} max {max(ratios):.3f}", flush=True)
        bad = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
        if bad:
            raise SystemExit(f"attack opaque ratio out of range at {bad}")
        print("samples", work / "samples_motion.png", flush=True)
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
        for direction in DIRS:
            key = f"{act}/{direction}"
            if key in state["done"]:
                continue
            od = frames_dir / act / direction
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[direction], act, f, palette), "RGBA").save(tmp)
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
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    counts_s, ratios = south_attack_ratios(sheets)
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(counts_s, ratios)):
        print(f"attack S {f:02d} opaque {c} ratio {r:.3f}", flush=True)
    print(f"min {min(ratios):.3f} max {max(ratios):.3f} h {sprite_h}", flush=True)

    report = qa_report(master, sheets, palette, ratios, sprite_h, rigs["S"])
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            try:
                subprocess.run(
                    [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                     "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                    capture_output=True, text=True, timeout=90,
                )
            except subprocess.TimeoutExpired:
                report[f"gait_{act}"] = "timeout"
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    public = {k: v for k, v in report.items() if not k.startswith("_")}
    (qa / "qa.json").write_text(json.dumps(public, ensure_ascii=False, indent=1))
    print(json.dumps(public, ensure_ascii=False)[:1600], flush=True)

    bad_spin = [s for s in report["spin_body"] if s["main"] < 0.90]
    failed = (
        report["_off"] or report["_clip"] or report["alphas"] != [0, 255]
        or report["_bad_ratio"] or bad_spin or report["hurt_squash"] or report["tip_drift"]
    )
    if failed:
        if not args.samples:
            update_queue("needs_fix")
        raise SystemExit(
            f"qa failed off={len(report['_off'])} clip={report['_clip']} alphas={report['alphas']} "
            f"ratios={report['_bad_ratio']} spin={bad_spin} squash={report['hurt_squash']} tips={report['tip_drift']}"
        )

    meta = {
        "generator": "forge_drifloon_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 425,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED),
        "still_faces": "viewer",
        "archetype": "balloon_strings",
        "face": face_info,
        "parts": counts,
        "palette_colors": int(len(palette)),
        "off_palette_colors": 0,
        "attack_s_min": round(min(ratios), 3),
        "attack_s_max": round(max(ratios), 3),
        "source": "front.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.publish and not args.no_sync:
        copied = sync_outputs()
        neighbor = publish_dir8(sprite_h)
        update_queue("done")
        arena_bytes = (SYNC_OUT / "sheet.png").stat().st_size
        print(f"synced {copied} neighbor {neighbor} arena_sheet_bytes {arena_bytes}", flush=True)
    print(f"h {sprite_h} cw {CELL_W} ch {CELL_H} fy {FOOT_Y}", flush=True)


if __name__ == "__main__":
    main()
