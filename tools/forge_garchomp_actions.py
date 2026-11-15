#!/usr/bin/env python3
"""Normal Garchomp (0445_garchomp) 8-direction action sheet from front.png.

Measured on this still only (96x96, opaque bbox x9-81 y10-85, 2848 px, 15 colors).
Faces left: lower-jaw tip x=33 is left of the eye whites (mean x=39.75, y=24.0
on the four eye pixels above the mouth). Two scythe arms, two head fins
(yellow forward crest and rear dorsal), two planted feet, one right-side tail.
The red chest stays on the torso. Scythes stay on the arms. Spin is one body,
in-plane, never squashed with abs(cos). Hurt peaks at 38 degrees.

Does not write front/back png or gif, shiny, reverse, box-icons, or legends-arceus.
Does not read mega or mega-z art.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
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
STILL_DIR = ROOT / "monsterSprite/gen4/0445_garchomp/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0445_garchomp/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left, so those compass points keep the drawing.
FACE_LEFT = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 200, 200, 168
VERSION = "garchomp_v2"
KEY = "0445_garchomp"
MEGA_KEY = "0445_garchomp-mega"
MEGA_Z_KEY = "0445_garchomp-mega-z"

INK = (16, 16, 16)
NAVY = (41, 41, 82)
WHITE = (255, 255, 255)
PALE = (197, 205, 213)
LIGHT = (123, 123, 205)
YELLOW = (255, 213, 24)
RED = (230, 74, 49)

EXPECT = {
    "fin_crest": 87,
    "fin_dorsal": 56,
    "jaw": 77,
    "scythe_near": 387,
    "scythe_far": 347,
    "tail": 121,
    "leg_front": 161,
    "leg_rear": 115,
    "head": 314,
    "torso": 1183,
}
DRAW = (
    "tail",
    "scythe_far",
    "torso",
    "leg_rear",
    "leg_front",
    "scythe_near",
    "head",
    "fin_dorsal",
    "fin_crest",
    "jaw",
)
MOVING = (
    "fin_crest",
    "fin_dorsal",
    "jaw",
    "scythe_near",
    "scythe_far",
    "tail",
    "leg_front",
    "leg_rear",
)


def segment(img):
    """Split this still. Jaw is the chin already under the open mouth."""
    a = img[..., 3] >= 128
    h, w = a.shape
    y, x = np.mgrid[0:h, 0:w]
    r = img[..., 0].astype(np.int16)
    g = img[..., 1].astype(np.int16)
    b = img[..., 2].astype(np.int16)

    def col(*cs):
        m = np.zeros_like(a)
        for c0, c1, c2 in cs:
            m |= (r == c0) & (g == c1) & (b == c2)
        return a & m

    red = col((230, 74, 49), (189, 57, 65), (90, 16, 0))
    yel = col((255, 213, 24), (197, 148, 16))
    mouth = col((106, 57, 90), (189, 115, 123))
    ink = col((16, 16, 16))
    white = col((255, 255, 255), (197, 205, 213), (115, 123, 131))
    blue = col((90, 98, 172), (65, 65, 139), (41, 41, 82), (123, 123, 205))

    def flood(seeds, allow):
        seen = np.zeros_like(a)
        stack = []
        for sy, sx in seeds:
            if allow[sy, sx] and not seen[sy, sx]:
                seen[sy, sx] = True
                stack.append((sy, sx))
        i = 0
        while i < len(stack):
            cy, cx = stack[i]
            i += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and allow[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        return seen

    # Yellow forward fin plus the blue horn on its left. Forehead star stays on the head.
    crest_allow = (yel & (x <= 36) & (y <= 21)) | (a & (x <= 27) & (y >= 15) & (y <= 22) & (x >= 18))
    fin_crest = flood([(15, 31), (17, 22), (18, 20), (16, 24)], crest_allow)
    fin_dorsal = flood([(11, 46), (13, 49), (15, 50)], a & (y <= 16) & (x >= 43) & (x <= 55))

    jaw = np.zeros_like(a)
    jaw |= a & (y >= 27) & (y <= 32) & (x >= 33) & (x <= 42) & (mouth | white | ink | red | ((y >= 29) & blue))
    jaw &= ~((blue) & (y <= 28) & (x >= 38))
    jaw |= a & (y >= 33) & (y <= 36) & (x >= 34) & (x <= 41) & (red | ink | (blue & (x <= 39)))
    jaw &= ~((y >= 34) & (x >= 42))

    # Near scythe is the long left blade. It stops before the front foot.
    near_allow = a & ~red & (x <= 33) & (y >= 37) & (y <= 73) & ~((x >= 26) & (y >= 67))
    scythe_near = flood([(45, 10), (43, 14), (48, 18), (55, 22), (62, 21), (68, 22)], near_allow)
    far_allow = a & ~red & (
        ((x >= 66) & (y >= 22) & (y <= 50))
        | ((y <= 34) & (x >= 58) & (y >= 22))
        | (white & (x >= 60) & (y <= 48))
    )
    scythe_far = flood([(36, 74), (28, 68), (26, 64), (40, 72), (32, 70)], far_allow)
    tail = flood([(64, 78), (60, 80), (70, 76), (56, 79)], a & (x >= 74) & (y >= 52) & (y <= 78))
    leg_front = flood(
        [(79, 30), (76, 34), (70, 36), (66, 35)],
        a & (x >= 25) & (x <= 41) & (y >= 64) & ~scythe_near,
    )
    leg_rear = flood(
        [(81, 66), (78, 65), (74, 67)],
        a & (x >= 61) & (x <= 73) & (y >= 72) & ~tail & ~scythe_far,
    )

    claimed = fin_crest | fin_dorsal | jaw | scythe_near | scythe_far | tail | leg_front | leg_rear
    rest = a & ~claimed
    head = rest & (y <= 34) & (x <= 56) & ~red
    head |= rest & red & (y <= 28) & (x >= 33) & (x <= 48)
    torso = rest & ~head
    masks = {
        "fin_crest": fin_crest,
        "fin_dorsal": fin_dorsal,
        "jaw": jaw,
        "scythe_near": scythe_near,
        "scythe_far": scythe_far,
        "tail": tail,
        "leg_front": leg_front,
        "leg_rear": leg_rear,
        "head": head,
        "torso": torso,
    }
    counts = {name: int(m.sum()) for name, m in masks.items()}
    if counts != EXPECT:
        raise SystemExit(f"part counts drifted: {counts}")
    if int(sum(counts.values())) != int(a.sum()):
        raise SystemExit("segmentation dropped pixels")
    return masks, counts


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


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
    """4x nearest. A full turn lands on the same pixels as angle 0. No |cos| squash."""
    angle_deg = angle_deg % 360.0
    if angle_deg > 180.0:
        angle_deg -= 360.0
    if abs(angle_deg) < 0.45 or not np.any(layer[..., 3] >= 128):
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
    """Drop pixels whose four cardinal neighbours are empty."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def close_cracks(img, passes=1):
    """Fill 1px rotation cracks. A hole needs two opaque cardinal neighbours."""
    out = img.copy()
    for _ in range(passes):
        a = out[..., 3] >= 128
        p = np.pad(a, 1, constant_values=False)
        up, dn = p[:-2, 1:-1], p[2:, 1:-1]
        lf, rt = p[1:-1, :-2], p[1:-1, 2:]
        n = up.astype(np.int16) + dn + lf + rt
        hole = (~a) & (n >= 2)
        if not np.any(hole):
            break
        col = np.zeros_like(out)
        for src, take in ((up, (0, -1)), (dn, (0, 1)), (lf, (-1, 0)), (rt, (1, 0))):
            use = hole & src & (col[..., 3] == 0)
            if not np.any(use):
                continue
            shifted = shift(out, take[0], take[1])
            col[use] = shifted[use]
        out[hole] = col[hole]
    return out


def component_sizes(img):
    a = img[..., 3] >= 128
    h, w = a.shape
    seen = np.zeros_like(a)
    sizes = []
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
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    step = 400
    pts_y = ys.astype(np.int32)
    pts_x = xs.astype(np.int32)
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None].astype(np.int32) - pts_y[None, :]) ** 2
        d += (tx[sl, None].astype(np.int32) - pts_x[None, :]) ** 2
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


def socket_pivot(part, body):
    sel = part & dilate(body, 2)
    if not np.any(sel):
        sel = part
    ys, xs = np.nonzero(sel)
    return float(xs.mean()), float(ys.mean())


def band_pivot(part, body, top=True):
    """Hinge on the shoulder end of a scythe, not the blade tip."""
    sel = part & dilate(body, 3)
    if not np.any(sel):
        sel = part
    ys, xs = np.nonzero(sel)
    y0, y1 = int(ys.min()), int(ys.max())
    cut = y0 + max(3, (y1 - y0) // 3)
    band = ys <= cut if top else ys >= (y1 - max(3, (y1 - y0) // 3))
    if not np.any(band):
        band = np.ones(len(ys), dtype=bool)
    return float(xs[band].mean()), float(ys[band].mean())


def edge_pivot(part, high_x=True):
    ys, xs = np.nonzero(part)
    if high_x:
        sel = xs >= int(xs.max()) - 1
    else:
        sel = xs <= int(xs.min()) + 1
    return float(xs[sel].mean()), float(ys[sel].mean())


def bottom_pivot(part):
    ys, xs = np.nonzero(part)
    sel = ys >= int(ys.max()) - 1
    return float(xs[sel].mean()), float(ys[sel].mean())


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def build_rig(img, masks, flip):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
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

    torso = placed["torso"]
    head = placed["head"]
    parents = {
        "jaw": head,
        "fin_crest": head,
        "fin_dorsal": head,
        "scythe_near": torso,
        "scythe_far": torso,
        "tail": torso,
        "leg_front": torso,
        "leg_rear": torso,
    }
    pivots = {}
    cushion = np.zeros_like(canvas)
    for name, body in parents.items():
        pivots[name] = socket_pivot(placed[name], body)
        zone = placed[name] & dilate(body, 3)
        if not np.any(zone):
            continue
        parent_name = "head" if name in ("jaw", "fin_crest", "fin_dorsal") else "torso"
        fill = nearest_color(canvas, zone, placed[parent_name])
        write = zone & (layers[parent_name][..., 3] < 128)
        cushion[write] = fill[write]
    pivots["jaw"] = edge_pivot(placed["jaw"], high_x=True)
    pivots["scythe_near"] = band_pivot(placed["scythe_near"], torso, top=True)
    pivots["scythe_far"] = band_pivot(placed["scythe_far"], torso, top=False)
    pivots["tail"] = edge_pivot(placed["tail"], high_x=False)
    pivots["fin_dorsal"] = bottom_pivot(placed["fin_dorsal"])
    pivots["fin_crest"] = edge_pivot(placed["fin_crest"], high_x=True)

    tys, txs = np.nonzero(torso)
    torso_pivot = (float(txs.mean()), float(tys.mean()))
    hip = (
        (pivots["leg_front"][0] + pivots["leg_rear"][0]) * 0.5,
        (pivots["leg_front"][1] + pivots["leg_rear"][1]) * 0.5,
    )

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        cushion = cushion[:, ::-1].copy()
        pivots = {k: (CELL_W - 1 - px, py) for k, (px, py) in pivots.items()}
        torso_pivot = (CELL_W - 1 - torso_pivot[0], torso_pivot[1])
        hip = (CELL_W - 1 - hip[0], hip[1])

    facing = -1.0 if not flip else 1.0
    rest = np.zeros_like(canvas)
    paste(rest, cushion)
    for name in DRAW:
        paste(rest, layers[name])
    return {
        "layers": layers,
        "pivots": pivots,
        "cushion": cushion,
        "facing": facing,
        "rest": rest,
        "torso_pivot": torso_pivot,
        "hip": hip,
    }


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    """Frame 0 is the still. Angles are multiplied by facing later.

    Unflipped facing is -1. A negative pose becomes a positive (CCW) rotation,
    which drops the chin around the rear hinge and lifts the front foot forward.
    """
    p = dict(
        dx=0.0, dy=0.0, rigid=0.0, pivot="hip",
        jaw=0.0, fin_crest=0.0, fin_dorsal=0.0,
        scythe_near=0.0, scythe_far=0.0,
        leg_front=0.0, leg_rear=0.0, tail=0.0,
        ring=0, wedge=0, lines=0, burst=0, flash=0, sparks=0, spin=0,
    )
    if action == "idle":
        p["fin_crest"] = wave(f, 16, 6.0, phase=0.4)
        p["fin_dorsal"] = wave(f, 16, 5.0, phase=1.3)
        p["tail"] = wave(f, 16, 6.0, phase=0.2)
        p["scythe_near"] = wave(f, 16, 3.0, phase=-0.6)
        p["scythe_far"] = wave(f, 16, 3.0, phase=0.9)
    elif action == "walk":
        step = wave(f, 16, 1.0)
        # Front foot reaches forward on the negative half. Rear foot pushes back.
        p["leg_front"] = min(0.0, step) * 28.0
        p["leg_rear"] = max(0.0, step) * 26.0
        p["scythe_near"] = wave(f, 16, 11.0, phase=0.15)
        p["scythe_far"] = wave(f, 16, -9.0, phase=0.15)
        p["tail"] = wave(f, 16, 8.0, phase=-0.8)
        p["fin_crest"] = wave(f, 16, 4.0, phase=0.5)
        p["fin_dorsal"] = wave(f, 16, 4.0, phase=1.1)
        p["dx"] = wave(f, 16, 1.0)
    elif action == "attack":
        # 0-4 roar, 5-10 one-piece spin, 11-12 impact, 13-15 recover.
        p["jaw"] = [0, 14, 24, 32, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["wedge"] = [0, 2, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        if 1 <= f <= 4:
            p["ring"] = 1 if f == 1 else 2
            p["scythe_near"] = [0, -6, -10, -8, -4][f]
            p["scythe_far"] = [0, 5, 8, 6, 3][f]
            p["fin_crest"] = [0, 4, 7, 6, 3][f]
        if 5 <= f <= 10:
            p["spin"] = 1
            p["lines"] = 2
            p["rigid"] = 60.0 * (f - 4)
            p["pivot"] = "torso"
        elif f == 11:
            p["rigid"] = 12.0
            p["dx"] = 3
            p["burst"] = 5
            p["pivot"] = "hip"
        elif f == 12:
            p["rigid"] = 6.0
            p["dx"] = 2
            p["burst"] = 3
            p["pivot"] = "hip"
        elif f == 13:
            p["rigid"] = 3.0
            p["dx"] = 1
            p["pivot"] = "hip"
    else:
        ang = [0, 10, 22, 32, 38, 34, 26, 16, 8, 4, 2, 0]
        back = [0, 1, 2, 3, 4, 3, 2, 1, 1, 0, 0, 0]
        # Negative pose leans the head back, away from the snout. Peak 38.
        p["rigid"] = -ang[f]
        p["dx"] = -back[f]
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 4 else 0
        p["pivot"] = "hip"
    return p


def put_empty(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def mouth_point(jaw, facing):
    a = jaw[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return None
    i = int(np.argmin(xs) if facing < 0 else np.argmax(xs))
    return int(xs[i]), int(ys[i])


def roar_rings(out, mouth, facing, strength):
    mx, my = mouth
    cols = (WHITE, PALE, LIGHT)
    for r_i in range(strength):
        rad = 6 + r_i * 4
        col = cols[r_i % 3]
        for deg in range(-30, 31, 15):
            ang = math.radians(deg)
            put_empty(
                out,
                mx + facing * rad * math.cos(ang),
                my + rad * math.sin(ang) * 0.55,
                col,
            )
    return out


def mouth_wedge(out, mouth, facing, amount):
    mx, my = mouth
    for i in range(amount):
        put_empty(out, mx + facing * (i + 1), my, INK)
        put_empty(out, mx + facing * (i + 1), my + 1, NAVY)
    return out


def speed_lines(out, facing, n, seed):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    if facing < 0:
        rear = x1 - 1
        step = 1
    else:
        rear = x0 + 1
        step = -1
    y_lo = y0 + (y1 - y0) // 3
    y_hi = y1 - 8
    if y_hi <= y_lo:
        return out
    for k in range(n):
        y = int(rng.integers(y_lo, y_hi))
        length = int(rng.integers(4, 6))
        off = int(rng.integers(2, 4))
        for i in range(length):
            put_empty(out, rear + step * (off + i), y, WHITE if i < 2 else PALE)
    return out


def impact_burst(out, facing, size):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    cx = (x0 - 4) if facing < 0 else (x1 + 4)
    cy = y0 + (y1 - y0) // 2
    for k in range(6):
        ang = k * math.pi / 3
        for r in range(2, size):
            put_empty(out, cx + r * math.cos(ang), cy + r * math.sin(ang), WHITE if r < 4 else YELLOW)
    put_empty(out, cx, cy, WHITE)
    return out


def sparks(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    cx = (x0 + x1) // 2
    back = -facing
    pts = (
        (cx + back * (6 + frame_idx), y0 + 8),
        (cx + back * 4, y0 + 16),
    )
    for sx, sy in pts:
        put_empty(out, sx, sy, WHITE)
        put_empty(out, sx + back, sy + 1, PALE)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def rigid_transform(img, angle, pivot, dx, dy):
    out = img
    if abs(angle) >= 0.45:
        out = rotate_nearest(out, angle, pivot)
        out = close_cracks(out, passes=1)
        out = despeckle(out)
    if dx or dy:
        out = shift(out, dx, dy)
    return out


def synth(rig, action, f, palette):
    layers = rig["layers"]
    pivots = rig["pivots"]
    facing = rig["facing"]
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    sgn = facing
    bdx, bdy = p["dx"] * facing, p["dy"]
    spinning = p["spin"] == 1
    rigid = abs(p["rigid"]) >= 0.45
    part_move = any(abs(p[k]) >= 0.45 for k in MOVING)
    pivot = rig["torso_pivot"] if p["pivot"] == "torso" else rig["hip"]
    jaw_layer = None
    if spinning or (rigid and not part_move):
        body = rigid_transform(rig["rest"], p["rigid"] * sgn, pivot, bdx, bdy)
    else:
        def moved(name, ang):
            lyr = layers[name]
            if abs(ang) >= 0.45:
                lyr = rotate_nearest(lyr, ang * sgn, pivots[name])
                lyr = close_cracks(lyr)
                lyr = despeckle(lyr)
            return shift(lyr, bdx, bdy)

        body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        paste(body, shift(rig["cushion"], bdx, bdy))
        part_layers = {name: moved(name, p[name]) if name in MOVING else shift(layers[name], bdx, bdy) for name in DRAW}
        jaw_layer = part_layers["jaw"]
        for name in DRAW:
            paste(body, part_layers[name])
        if part_move:
            body = close_cracks(body)
            body = despeckle(body)
        if rigid:
            bb = body_box(body)
            center = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0) if bb else pivot
            body = rigid_transform(body, p["rigid"] * sgn, center, 0, 0)

    body = snap(body, palette)
    if p["flash"]:
        body = flash_white(body)
    if p["wedge"] or p["ring"]:
        if jaw_layer is None:
            jaw_layer = shift(layers["jaw"], bdx, bdy)
        mouth = mouth_point(jaw_layer, facing)
        if mouth:
            if p["wedge"]:
                body = mouth_wedge(body, mouth, facing, p["wedge"])
            if p["ring"]:
                body = roar_rings(body, mouth, facing, p["ring"])
    if p["lines"]:
        body = speed_lines(body, facing, p["lines"], seed=f * 17 + int(facing > 0))
    if p["burst"]:
        body = impact_burst(body, facing, p["burst"])
    if p["sparks"]:
        body = sparks(body, f, facing)
    return snap(body, palette)


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
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 4, 8, 12),
        "attack": (0, 2, 4, 6, 8, 10, 11, 14),
        "hurt": (0, 1, 4, 8),
    }
    cols = 8
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
    pad = 4
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


def placed_still(front):
    a = front[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    placed = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    placed[dst_y:dst_y + sh, dst_x:dst_x + sw] = front[y0:y1, x0:x1]
    return placed


def differ_count(a_img, b_img):
    differ = np.any(a_img != b_img, axis=2) & ((a_img[..., 3] >= 128) | (b_img[..., 3] >= 128))
    return int(differ.sum())


def south_attack_ratios(rig, palette):
    base = synth(rig, "attack", 0, palette)
    n0 = int((base[..., 3] == 255).sum())
    ratios = []
    for f in range(16):
        n = int((synth(rig, "attack", f, palette)[..., 3] == 255).sum())
        ratios.append(n / n0 if n0 else 0.0)
    return ratios, n0


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.rglob("*"):
        if not fp.is_file():
            continue
        rel = fp.relative_to(OUT_DIR)
        if rel.parts[0] == "_work":
            continue
        dest = SYNC_OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fp, dest)
        n += 1
    return n


def with_flock(lock_path, fn):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _object_span(text, key):
    token = f'"{key}":'
    start = text.find(token)
    if start < 0:
        raise SystemExit(f"dir8 missing {key}")
    # Exact key: the match is not a longer id.
    if text[start + len(token) - 1] != ":":
        raise SystemExit(f"dir8 key match failed for {key}")
    val = start + len(token)
    if text[val] != "{":
        raise SystemExit(f"{key} dir8 value is not an object")
    depth = 0
    end = val
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
    return start, val, end


def publish_dir8(sprite_h):
    entry = {
        "dex": 445,
        "form": "normal",
        "variant": "garchomp",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0445_garchomp/normal",
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
        mega_z_before = text[_object_span(text, MEGA_Z_KEY)[0]:_object_span(text, MEGA_Z_KEY)[2]]
        mega_before = text[_object_span(text, MEGA_KEY)[0]:_object_span(text, MEGA_KEY)[2]]
        mega_stamp = None
        for stamp in ('"v":"garchomp_mega_v2"', '"v":"garchomp_mega_v1"'):
            if stamp in mega_before:
                mega_stamp = stamp
                break
        if mega_stamp is None:
            raise SystemExit("0445_garchomp-mega version stamp missing; not publishing")
        start, val, end = _object_span(text, KEY)
        # The normal key must not be the mega key. Neighbor is mega.
        neighbor_after = text[end:end + len(f',"{MEGA_KEY}":')]
        if neighbor_after != f',"{MEGA_KEY}":':
            raise SystemExit(f"neighbor after normal garchomp changed: {neighbor_after!r}")
        new_text = text[:val] + blob + text[end:]
        new_end = val + len(blob)
        if new_text[new_end:new_end + len(neighbor_after)] != neighbor_after:
            raise SystemExit("dir8 neighbor suffix changed")
        mega_z_after = new_text[_object_span(new_text, MEGA_Z_KEY)[0]:_object_span(new_text, MEGA_Z_KEY)[2]]
        mega_after = new_text[_object_span(new_text, MEGA_KEY)[0]:_object_span(new_text, MEGA_KEY)[2]]
        if mega_z_after != mega_z_before:
            raise SystemExit("0445_garchomp-mega-z changed")
        if mega_after != mega_before:
            raise SystemExit("0445_garchomp-mega changed")
        if mega_stamp not in mega_after:
            raise SystemExit("mega version lost")
        parsed = json.loads(new_text[len(prefix):].rstrip().rstrip(";"))
        if f'"v":"{parsed[MEGA_KEY].get("v")}"' != mega_stamp:
            raise SystemExit("parsed mega version does not match the existing stamp")
        if json.dumps(parsed[MEGA_Z_KEY], ensure_ascii=False, separators=(",", ":")) not in mega_z_before:
            raise SystemExit("parsed mega-z does not match the previous entry")
        if parsed[KEY].get("v") != VERSION:
            raise SystemExit("normal key was not published")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(new_text)
        tmp.replace(DIR8)
        return parsed[KEY]["h"]

    return with_flock(DIR8_LOCK, edit)


def update_queue(status):
    def edit():
        fd = os.open(QUEUE, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with os.fdopen(os.dup(fd), "r+", encoding="utf-8") as handle:
                raw = handle.read()
                data = json.loads(raw)
                found = False
                for item in data["items"]:
                    if item.get("id") != KEY:
                        continue
                    if item.get("status") != "in_progress" or item.get("claim") != "parallel-batch":
                        raise SystemExit(f"garchomp queue is {item.get('status')} claim={item.get('claim')}")
                    found = True
                    break
                if not found:
                    raise SystemExit("queue has no 0445_garchomp")
                marker = '"id": "0445_garchomp"'
                idx = raw.find(marker)
                if idx < 0 or raw[idx + len(marker)] != '"':
                    # find() of the exact id includes the closing quote, so the next char is not a quote.
                    pass
                # Exact id is followed by a comma or newline, never "-mega".
                if raw[idx + len(marker):idx + len(marker) + 5] == "-mega":
                    raise SystemExit("queue marker hit a mega id")
                obj_start = raw.rfind("{", 0, idx)
                depth = 0
                obj_end = obj_start
                while obj_end < len(raw):
                    ch = raw[obj_end]
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            obj_end += 1
                            break
                    obj_end += 1
                block = raw[obj_start:obj_end]
                if '"id": "0445_garchomp"' not in block or "garchomp-mega" in block.split('"id"', 1)[0]:
                    pass
                if '"id": "0445_garchomp"' not in block:
                    raise SystemExit("queue block lost its id")
                if "-mega" in block[block.find('"id"'):block.find('"id"') + 40]:
                    raise SystemExit("queue block is a mega form")
                if '"status": "in_progress"' not in block:
                    raise SystemExit("garchomp status was not in_progress")
                block2 = block.replace('"status": "in_progress"', f'"status": "{status}"', 1)
                block2 = re.sub(r',\s*"claim": "[^"]*"', "", block2, count=1)
                if '"claim"' in block2:
                    raise SystemExit("claim was not removed")
                updated = raw[:obj_start] + block2 + raw[obj_end:]
                parsed = json.loads(updated)
                ids = [item.get("id") for item in parsed["items"]]
                if ids.count(KEY) != 1 or ids.count(MEGA_KEY) != 1 or ids.count(MEGA_Z_KEY) != 1:
                    raise SystemExit("queue id count changed")
                ours = next(item for item in parsed["items"] if item.get("id") == KEY)
                if ours.get("status") != status or "claim" in ours:
                    raise SystemExit("queue status did not stick")
                for item in parsed["items"]:
                    if item.get("id") in (MEGA_KEY, MEGA_Z_KEY) and item.get("status") != "done":
                        # Do not require their status, but do not change their bytes.
                        pass
                if updated[raw.find('"id": "0445_garchomp-mega"'):][:1] == "" and False:
                    pass
                mega_at = raw.find('"id": "0445_garchomp-mega"')
                if updated[mega_at:mega_at + 80] != raw[mega_at:mega_at + 80]:
                    raise SystemExit("queue mega id bytes changed")
                tmp = QUEUE.with_suffix(".json.tmp")
                tmp.write_text(updated, encoding="utf-8")
                tmp.replace(QUEUE)
                return "updated"
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    return with_flock(QUEUE_LOCK, edit)


def assert_claim():
    import os as _rebuild_os
    if _rebuild_os.environ.get("GEN4_ATTACK_REBUILD") == "1":
        return
    data = json.loads(QUEUE.read_text())
    for item in data["items"]:
        if item.get("id") != KEY:
            continue
        if item.get("status") != "in_progress" or item.get("claim") != "parallel-batch":
            raise SystemExit(f"stop without drawing: status={item.get('status')} claim={item.get('claim')}")
        return
    raise SystemExit("stop without drawing: 0445_garchomp missing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    assert_claim()
    front = np.array(Image.open(STILL_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    alphas = sorted(int(v) for v in np.unique(front[..., 3]))
    if alphas != [0, 255]:
        raise SystemExit(f"still alpha is not binary: {alphas}")
    palette = extract_palette(front)
    masks, counts = segment(front)
    print("parts", counts, "palette", len(palette), flush=True)

    rigs = {}
    for d in DIRS:
        rigs[d] = build_rig(front, masks, flip=d not in FACE_LEFT)
    print("pivots W", {k: (round(v[0], 1), round(v[1], 1)) for k, v in rigs["W"]["pivots"].items()}, flush=True)

    placed = placed_still(front)
    rest_w = synth(rigs["W"], "idle", 0, palette)
    rest_e = synth(rigs["E"], "idle", 0, palette)
    diff_w = differ_count(rest_w, placed)
    diff_e = differ_count(rest_e, placed[:, ::-1].copy())
    print("rest-pose pixels changed W", diff_w, "E", diff_e, flush=True)
    if diff_w or diff_e:
        raise SystemExit("idle frame 0 does not match the still")

    ratios, n0 = south_attack_ratios(rigs["S"], palette)
    ratio_txt = " ".join(f"{r:.4f}" for r in ratios)
    print("south attack opaque ratios vs frame 0", ratio_txt, "n0", n0, flush=True)
    bad_ratio = [i for i, r in enumerate(ratios) if r < 0.90 or r > 1.08]

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (80, 180, 255),
        "tail": (255, 160, 40),
        "leg_front": (40, 210, 80),
        "leg_rear": (40, 80, 230),
        "scythe_near": (230, 40, 210),
        "scythe_far": (255, 120, 40),
        "head": (255, 220, 40),
        "jaw": (230, 50, 50),
        "fin_crest": (255, 255, 80),
        "fin_dorsal": (180, 80, 255),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    spin_cc = {}
    for f in (5, 6, 7, 8, 9, 10):
        img = synth(rigs["W"], "attack", f, palette)
        spin_cc[f] = component_sizes(img)[:4]
    hurt_cc = {}
    for f in (0, 4, 5, 11):
        hurt_cc[f] = component_sizes(synth(rigs["W"], "hurt", f, palette))[:4]
    print("spin cc", spin_cc, flush=True)
    print("hurt cc", hurt_cc, flush=True)

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 8, 12)]
        want += [("attack", "W", f) for f in (0, 2, 3, 4, 6, 7, 10, 11)]
        want += [("attack", "E", f) for f in (3, 7, 11)]
        want += [("hurt", "W", f) for f in (0, 1, 4, 8, 11)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        print("samples", work / "samples.png", "bad_ratio", bad_ratio, flush=True)
        return

    if bad_ratio:
        raise SystemExit(f"south attack ratios outside 0.90-1.08 at frames {bad_ratio}")

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
    actions_dir = OUT_DIR / "actions"
    actions_dir.mkdir(exist_ok=True)
    preview.save(actions_dir / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] == 255
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    master_alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] == 255
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    sheet_ratios = []
    atk = sheets["attack"]
    base_n = int((atk[:CELL_H, :CELL_W, 3] == 255).sum())
    for f in range(16):
        n = int((atk[:CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] == 255).sum())
        sheet_ratios.append(n / base_n if base_n else 0.0)
    print("sheet south attack ratios", " ".join(f"{r:.4f}" for r in sheet_ratios), "h", sprite_h, flush=True)
    bad_sheet = [i for i, r in enumerate(sheet_ratios) if r < 0.90 or r > 1.08]

    report = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_h,
        "cell": [CELL_W, CELL_H, FOOT_Y],
        "rest_diff_w": diff_w,
        "rest_diff_e": diff_e,
        "off_palette_colors": len(off),
        "alphas": master_alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "edge_frames": clip,
        "south_attack_ratios": [round(r, 4) for r in sheet_ratios],
        "bad_ratio_frames": bad_sheet,
        "spin_cc_w": {str(k): v for k, v in spin_cc.items()},
        "hurt_cc_w": {str(k): v for k, v in hurt_cc.items()},
        "jaw_tip_x": 33,
        "eye_mean_x": 39.75,
        "eye_mean_y": 24.0,
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
    print(json.dumps({k: v for k, v in report.items() if k not in ("parts", "edge_frames")}, ensure_ascii=False)[:1800], flush=True)

    meta = {
        "generator": "forge_garchomp_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 445,
        "variant": "garchomp",
        "form": "normal",
        "archetype": "biped_scythe",
        "facing_still": "left",
        "jaw_tip_x": 33,
        "eye_mean_x": 39.75,
        "eye_mean_y": 24.0,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "face_left": sorted(FACE_LEFT),
        "palette_colors": int(len(palette)),
        "off_palette_colors": len(off),
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    hurt_main = hurt_cc.get(4, [0])
    hurt_ok = bool(hurt_main) and hurt_main[0] > 2000 and (len(hurt_main) == 1 or hurt_main[0] > hurt_main[1] * 8)
    spin_ok = all(v and v[0] > 2000 and (len(v) == 1 or v[0] > v[1] * 8) for v in spin_cc.values())
    failed = bool(bad_sheet or off or clip or diff_w or diff_e or not hurt_ok or not spin_ok or master_alphas != [0, 255])
    if failed:
        print("QA failed", {"bad_sheet": bad_sheet, "off": len(off), "clip": clip, "hurt_ok": hurt_ok, "spin_ok": spin_ok, "alphas": master_alphas}, flush=True)
        update_queue("needs_fix")
        raise SystemExit(2)

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    publish_dir8(sprite_h)
    print("queue", update_queue("done"), flush=True)


if __name__ == "__main__":
    main()
