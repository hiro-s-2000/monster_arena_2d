#!/usr/bin/env python3
"""Rampardos (0409_rampardos) 8-direction action sheet from the 96x96 front still.

Measured on front.png (opaque bbox x 20-74, y 9-86, 2307 px, 14 colors):
biped, facing left. Left horn tip x=22 and jaw tip x=27 are left of the red
eye (mean x=40.83, y=26.17). Tail stub is the right lobe (x>=66, y 48-64).
Unflipped dirs are N NW W SW. Flipped dirs are S SE E NE so E faces screen-right.

The blue skull dome and both white horns are one head and stay attached.
The gray lower jaw already in the still opens for the roar, plus rings.
Two legs, one short arm, one tail. No second head and no extra legs.
Spin and hurt rotate the whole body (hurt peaks at 38 degrees). No |cos| squash.

Does not write front/back png or gif, shiny, reverse, box-icons, or legends-arceus.
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
STILL_DIR = ROOT / "monsterSprite/gen4/0409_rampardos/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0409_rampardos/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left, so the left-facing compass keeps the drawing.
FACE_LEFT = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 200, 200, 168
VERSION = "rampardos_v2"
KEY = "0409_rampardos"

INK = (16, 16, 16)
DARK = (65, 65, 65)
WHITE = (255, 255, 255)
PALE = (189, 213, 222)
LIGHT = (115, 172, 238)
BLUE = (57, 131, 213)
RED = (246, 82, 98)

# Counts from the measured masks. Drift means the cut moved.
EXPECT = {
    "jaw": 132,
    "tail": 119,
    "leg_front": 271,
    "leg_rear": 295,
    "arm": 65,
    "head": 511,
    "torso": 914,
}
DRAW = ("tail", "leg_rear", "torso", "leg_front", "arm", "head", "jaw")
MOVING = ("jaw", "arm", "tail", "leg_front", "leg_rear")


def segment(img):
    """Split the still. Horns stay inside the head. Jaw is the gray muzzle only."""
    a = img[..., 3] >= 128
    r = img[..., 0].astype(np.int16)
    g = img[..., 1].astype(np.int16)
    b = img[..., 2].astype(np.int16)
    blue = a & (b > r + 25) & (b > 140)
    h, w = a.shape
    jaw = np.zeros_like(a)
    tail = np.zeros_like(a)
    leg_f = np.zeros_like(a)
    leg_b = np.zeros_like(a)
    arm = np.zeros_like(a)
    for y in range(h):
        for x in range(w):
            if not a[y, x]:
                continue
            # Lower jaw, below the mouth line, eye pixels excluded.
            if (not blue[y, x]) and 24 <= y <= 34 and x <= 46:
                line = 22 + (x - 22) * 0.28
                if y >= line and not (38 <= x <= 43 and 24 <= y <= 28):
                    jaw[y, x] = True
            if x >= 66 and 46 <= y <= 64:
                tail[y, x] = True
    for y in range(h):
        for x in range(w):
            if not a[y, x] or tail[y, x] or jaw[y, x]:
                continue
            if y >= 62 and x <= 42:
                leg_f[y, x] = True
            elif y >= 64 and x >= 50 and x <= 68 and (y >= 70 or x >= 54):
                leg_b[y, x] = True
    for y in range(48, 62):
        for x in range(20, 31):
            if a[y, x] and not leg_f[y, x] and not jaw[y, x] and not blue[y, x]:
                arm[y, x] = True
    head = np.zeros_like(a)
    for y in range(h):
        for x in range(w):
            if a[y, x] and y <= 36 and not (
                jaw[y, x] or tail[y, x] or leg_f[y, x] or leg_b[y, x] or arm[y, x]
            ):
                head[y, x] = True
    torso = a & ~(jaw | tail | leg_f | leg_b | arm | head)
    masks = {
        "jaw": jaw,
        "tail": tail,
        "leg_front": leg_f,
        "leg_rear": leg_b,
        "arm": arm,
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
    """Drop pixels whose four cardinal neighbours are empty. Eyes stay."""
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


def edge_pivot(part, high_x=True):
    """Hinge on the body side of a part, not the centroid."""
    ys, xs = np.nonzero(part)
    if high_x:
        sel = xs >= int(xs.max()) - 1
    else:
        sel = xs <= int(xs.min()) + 1
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
        "arm": torso,
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
        parent_name = "head" if name == "jaw" else "torso"
        fill = nearest_color(canvas, zone, placed[parent_name])
        write = zone & (layers[parent_name][..., 3] < 128)
        cushion[write] = fill[write]
    # Jaw opens around the rear hinge. The arm swings from the shoulder.
    pivots["jaw"] = edge_pivot(placed["jaw"], high_x=True)
    pivots["arm"] = edge_pivot(placed["arm"], high_x=True)

    tys, txs = np.nonzero(torso)
    torso_pivot = (float(txs.mean()), float(tys.mean()))
    hip = (
        (pivots["leg_front"][0] + pivots["leg_rear"][0]) * 0.5,
        (pivots["leg_front"][1] + pivots["leg_rear"][1]) * 0.5,
    )

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        cushion = cushion[:, ::-1].copy()
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
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

    Unflipped facing is -1, so a positive pose angle rotates negatively.
    Front leg lifts on a positive rotation. Rear leg lifts on a negative one.
    The jaw opens downward on a negative rotation around the rear hinge.
    """
    p = dict(
        dx=0.0, dy=0.0, rigid=0.0, pivot="hip",
        jaw=0.0, arm=0.0, leg_front=0.0, leg_rear=0.0, tail=0.0,
        ring=0, wedge=0, lines=0, burst=0, flash=0, sparks=0, spin=0,
    )
    if action == "idle":
        p["tail"] = wave(f, 16, 7.0, phase=0.5)
        p["arm"] = wave(f, 16, 5.0, phase=-0.4)
    elif action == "walk":
        step = wave(f, 16, 34.0)
        # Negative pose lifts the front leg after * facing(-1).
        p["leg_front"] = min(0.0, step)
        p["leg_rear"] = max(0.0, step)
        p["arm"] = wave(f, 16, -14.0, phase=0.4)
        p["tail"] = wave(f, 16, 8.0, phase=-0.7)
        p["dx"] = wave(f, 16, 1.0)
    elif action == "attack":
        # 0-4 roar, 5-10 one-piece spin, 11-12 impact, 13-15 recover.
        jaw = [0, 16, 26, 34, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        p["jaw"] = jaw[f]
        p["wedge"] = [0, 2, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        if 1 <= f <= 4:
            p["ring"] = 1 if f == 1 else 2
        if 5 <= f <= 10:
            p["spin"] = 1
            p["lines"] = 3
            # 60 degrees per frame, upright again on frame 10.
            p["rigid"] = 60.0 * (f - 4)
            p["pivot"] = "torso"
        if f == 11:
            # Positive pose * facing(-1) leans the head forward.
            p["rigid"] = 14.0
            p["dx"] = 4
            p["burst"] = 6
        elif f == 12:
            p["rigid"] = 8.0
            p["dx"] = 2
            p["burst"] = 4
        elif f == 13:
            p["dx"] = 1
            p["rigid"] = 3.0
        elif f == 14:
            p["dx"] = 1
    else:
        ang = [0, 12, 24, 32, 38, 34, 26, 18, 10, 5, 2, 0]
        back = [0, 2, 3, 4, 5, 4, 3, 2, 1, 1, 0, 0]
        # Negative pose leans the skull back, away from the snout. Peak 38.
        p["rigid"] = -ang[f]
        p["dx"] = -back[f]
        p["dy"] = [0, -1, -1, -2, -2, -1, -1, 0, 0, 0, 0, 0][f]
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 5 else 0
        p["pivot"] = "hip"
    return p


def put_empty(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def mouth_point(jaw, facing):
    """Front of the upper jaw. Not the horn tip, which sticks out past the mouth."""
    a = jaw[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return None
    y_cut = np.quantile(ys.astype(np.float32), 0.45)
    band = ys <= y_cut
    if not np.any(band):
        band = np.ones(len(ys), dtype=bool)
    sel = np.flatnonzero(band)
    i = int(sel[np.argmin(xs[band]) if facing < 0 else np.argmax(xs[band])])
    return int(xs[i]), int(ys[i])


def roar_rings(out, mouth, facing, strength):
    mx, my = mouth
    cols = (WHITE, PALE, LIGHT)
    for r_i in range(strength):
        rad = 5 + r_i * 4
        col = cols[r_i % 3]
        for deg in range(-36, 37, 12):
            ang = math.radians(deg)
            put_empty(
                out,
                mx + facing * rad * math.cos(ang),
                my + rad * math.sin(ang) * 0.62,
                col,
            )
    return out


def mouth_wedge(out, mouth, facing, amount):
    mx, my = mouth
    for i in range(amount):
        put_empty(out, mx + facing * (i + 1), my + 1, INK)
        put_empty(out, mx + facing * (i + 1), my + 2, DARK)
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
    y_hi = y1 - 6
    if y_hi <= y_lo:
        return out
    for k in range(n):
        y = int(rng.integers(y_lo, y_hi))
        length = int(rng.integers(4, 7))
        off = int(rng.integers(2, 5))
        for i in range(length):
            put_empty(out, rear + step * (off + i), y, WHITE if i < 2 else PALE)
    return out


def impact_burst(out, facing, size):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    cx = (x0 - 3) if facing < 0 else (x1 + 3)
    cy = y0 + (y1 - y0) // 2
    for k in range(6):
        ang = k * math.pi / 3
        for r in range(2, size):
            put_empty(out, cx + r * math.cos(ang), cy + r * math.sin(ang), WHITE if r < 4 else LIGHT)
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
        (cx + back * (6 + frame_idx), y0 + 6),
        (cx + back * 5, y0 + 14),
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
        out = close_cracks(out)
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
    if spinning or (rigid and not part_move):
        body = rigid_transform(rig["rest"], p["rigid"] * sgn, pivot, bdx, bdy)
    else:
        def moved(name, ang):
            lyr = rotate_nearest(layers[name], ang * sgn, pivots[name])
            return shift(lyr, bdx, bdy)

        body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        paste(body, shift(rig["cushion"], bdx, bdy))
        part_layers = {
            "tail": moved("tail", p["tail"]),
            "leg_rear": moved("leg_rear", p["leg_rear"]),
            "torso": shift(layers["torso"], bdx, bdy),
            "leg_front": moved("leg_front", p["leg_front"]),
            "arm": moved("arm", p["arm"]),
            "head": shift(layers["head"], bdx, bdy),
            "jaw": moved("jaw", p["jaw"]),
        }
        for name in DRAW:
            paste(body, part_layers[name])
        if part_move or rigid:
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
        mouth = mouth_point(layers["jaw"], facing)
        if mouth:
            mouth = (mouth[0] + bdx, mouth[1] + bdy)
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


def publish_dir8(sprite_h):
    entry = {
        "dex": 409,
        "form": "normal",
        "variant": "rampardos",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0409_rampardos/normal",
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
        key = f'"{KEY}":'
        start = text.index(key)
        val = start + len(key)
        if text[val] != "{":
            raise SystemExit("rampardos dir8 value is not an object")
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
        neighbor_after = text[end:end + 18]
        neighbor_before = text[start - 40:start]
        if ',"0410_shieldon":' not in neighbor_after and not neighbor_after.startswith(',"0410_shieldon"'):
            raise SystemExit(f"neighbor after rampardos changed: {neighbor_after!r}")
        new_text = text[:val] + blob + text[end:]
        new_end = val + len(blob)
        if new_text[new_end:new_end + 18] != neighbor_after:
            raise SystemExit("dir8 neighbor suffix changed")
        if new_text[start - 40:start] != neighbor_before:
            raise SystemExit("dir8 neighbor prefix changed")
        if new_text.count('"' + "0410_shieldon" + '"') != text.count('"' + "0410_shieldon" + '"'):
            raise SystemExit("0410_shieldon key count changed")
        if new_text.count('"' + "0408_cranidos" + '"') != text.count('"' + "0408_cranidos" + '"'):
            raise SystemExit("0408_cranidos key count changed")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(new_text)
        json.loads(new_text[len(prefix):].rstrip().rstrip(";"))
        tmp.replace(DIR8)
        return neighbor_before, neighbor_after

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
                    if item.get("claim") not in (None, "parallel-batch") and status == "done":
                        raise SystemExit(f"rampardos claim is {item.get('claim')}")
                    item["status"] = status
                    item.pop("claim", None)
                    found = True
                    break
                if not found:
                    raise SystemExit("queue has no 0409_rampardos")
                # Surgical write so other items keep their bytes.
                marker = '"id": "0409_rampardos"'
                idx = raw.find(marker)
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
                if '"status": "in_progress"' not in block and status == "done":
                    if '"status": "done"' in block and '"claim"' not in block:
                        return "already-done"
                    raise SystemExit("rampardos status was not in_progress")
                block2 = block.replace('"status": "in_progress"', f'"status": "{status}"', 1)
                block2 = block2.replace('"status": "needs_fix"', f'"status": "{status}"', 1)
                # Drop the claim property and the comma that introduced it.
                block2 = re.sub(r',\s*"claim": "[^"]*"', "", block2, count=1)
                if block2 == block and status != "done":
                    raise SystemExit("queue block did not change")
                updated = raw[:obj_start] + block2 + raw[obj_end:]
                json.loads(updated)
                # Confirm only this id's status line flipped.
                if updated.count(f'"id": "{KEY}"') != 1:
                    raise SystemExit("queue id count changed")
                tmp = QUEUE.with_suffix(".json.tmp")
                tmp.write_text(updated, encoding="utf-8")
                tmp.replace(QUEUE)
                return "updated"
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    return with_flock(QUEUE_LOCK, edit)


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
    masks, counts = segment(front)
    print("parts", counts, "palette", len(palette), flush=True)

    rigs = {}
    for d in DIRS:
        rigs[d] = build_rig(front, masks, flip=d not in FACE_LEFT)

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
        "arm": (230, 40, 210),
        "head": (255, 220, 40),
        "jaw": (230, 50, 50),
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
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] == 255
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    # Recompute south ratios from the written sheet so publish uses the same pixels.
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
        "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "edge_frames": clip,
        "south_attack_ratios": [round(r, 4) for r in sheet_ratios],
        "bad_ratio_frames": bad_sheet,
        "spin_cc_w": {str(k): v for k, v in spin_cc.items()},
        "hurt_cc_w": {str(k): v for k, v in hurt_cc.items()},
        "eye_mean_x": 40.833,
        "snout_x": 22,
        "jaw_tip_x": 27,
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
    print(json.dumps({k: v for k, v in report.items() if k not in ("parts", "edge_frames")}, ensure_ascii=False)[:1600], flush=True)

    meta = {
        "generator": "forge_rampardos_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 409,
        "variant": "rampardos",
        "form": "normal",
        "archetype": "biped",
        "facing_still": "left",
        "snout_x": 22,
        "jaw_tip_x": 27,
        "eye_mean_x": 40.833,
        "eye_mean_y": 26.167,
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
    hurt_ok = len(hurt_main) >= 1 and (len(hurt_main) == 1 or hurt_main[0] > hurt_main[1] * 8)
    spin_ok = all(v and v[0] > 1500 for v in spin_cc.values())
    failed = bool(bad_sheet or off or clip or diff_w or diff_e or not hurt_ok or not spin_ok or alphas != [0, 255])
    if failed:
        print("QA failed", {"bad_sheet": bad_sheet, "off": len(off), "clip": clip, "hurt_ok": hurt_ok, "spin_ok": spin_ok, "alphas": alphas}, flush=True)
        if args.publish:
            update_queue("needs_fix")
        raise SystemExit(2)

    if args.publish and not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
        publish_dir8(sprite_h)
        print("queue", update_queue("done"), flush=True)
    elif not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
