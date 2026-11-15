#!/usr/bin/env python3
"""Turtwig (0387) 8-direction action sheet from the normal 96x96 front.png.

Measured on this still only (opaque 748, bbox x32-62 y27-68, 14 colours).
The one white eye pixel is (45, 48). In that row the face runs x32-52, so
the snout (x32) is left of the eye. The still faces left. Unflipped directions
are N NW W SW. S SE E NE are mirrored so E faces screen-right.

Parts from these pixels, not from another species:
  leaf_l / leaf_r  sprout blades (y27-36). The roar flares these. No crest was invented.
  jaw              bright yellow cheek, x>=40 and y<60. There is no mouth slit, so no wedge.
  foot_a/b/c       the three protrusions under the belly (y>=65, and the right nub y>=64).
                   A fourth leg is not in the still and is not added.
  torso            head, eye, brown stem, shell rim, belly. Stays one piece.

idle   : feet and shell planted. Leaves sway. Frame 0 is the still.
walk   : in-place quadruped shuffle of the three real feet. Two steps per loop.
attack : 0-4 leaf flare + jaw drop + rings, 5-10 one-piece 360 spin,
         11-12 impact, 13-15 recover.
hurt   : rigid recoil of at most 34 degrees. No squash, no belly mirror.

Palette-locked. Alpha is only 0 or 255. front/back png and gif are not written.
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
STILL_DIR = ROOT / "monsterSprite/gen4/0387_turtwig/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0387_turtwig/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "turtwig_v2"

INK = (0, 0, 0)
YELLOW = (246, 230, 90)
WHITE = (255, 255, 255)
LEAF_HI = (123, 213, 106)
LEAF_MID = (57, 164, 65)
LEAF_DK = (57, 90, 41)
STEM_DK = (98, 74, 57)
STEM_MID = (139, 106, 65)
STEM_TAN = (180, 131, 82)


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


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def is_col(img, rgb):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(rgb, np.uint8), axis=-1)


def measure_facing(img):
    """Snout versus the single white eye pixel. Left snout means the still faces left."""
    white = is_col(img, WHITE)
    ys, xs = np.nonzero(white)
    if len(xs) != 1:
        raise SystemExit(f"expected one eye highlight, found {len(xs)}")
    eye_x, eye_y = int(xs[0]), int(ys[0])
    op = img[..., 3] >= 128
    # The eye's own row is the face. Lower rows are the shell and must not count as a snout.
    cols = np.nonzero(op[eye_y])[0]
    left, right = int(cols.min()), int(cols.max())
    faces_left = (eye_x - left) > (right - eye_x)
    return faces_left, eye_x, eye_y, left, right


def segment(img):
    """Partition this still. Cuts are its own sprout, yellow cheek, and three feet."""
    h, w = img.shape[:2]
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    ink = is_col(img, INK)
    yellow = is_col(img, YELLOW)
    leaf_g = is_col(img, LEAF_HI) | is_col(img, LEAF_MID)
    leaf_e = is_col(img, LEAF_DK)
    brown = is_col(img, STEM_DK) | is_col(img, STEM_MID) | is_col(img, STEM_TAN)

    # Blades stop at the brown stem. x-limits keep the two leaves from bridging.
    allow = (leaf_g | leaf_e | ink) & (yy <= 36) & ~brown
    leaf_l = flood(leaf_g & (xx <= 45) & (yy <= 35), allow & (xx <= 45))
    leaf_r = flood(leaf_g & (xx >= 47) & (yy <= 35), allow & (xx >= 47))
    leaf_l = leaf_l | (dilate(leaf_l, 1) & ink & (yy <= 36) & (xx <= 46))
    leaf_r = leaf_r | (dilate(leaf_r, 1) & ink & (yy <= 36) & (xx >= 46))
    both = leaf_l & leaf_r
    leaf_l = leaf_l & ~both
    leaf_r = leaf_r & ~both

    # Yellow cheek under the eye. Foot yellow is y>=65 and is not part of the jaw.
    jaw = yellow & (yy < 60) & (xx >= 40)

    # Three belly protrusions. Columns come from the bottom contour valleys.
    foot_a = op & (xx >= 38) & (xx <= 44) & (yy >= 65)
    foot_b = op & (xx >= 48) & (xx <= 54) & (yy >= 65)
    foot_c = op & (xx >= 56) & (yy >= 64)

    used = leaf_l | leaf_r | jaw | foot_a | foot_b | foot_c
    torso = op & ~used
    parts = {
        "leaf_l": leaf_l, "leaf_r": leaf_r, "jaw": jaw,
        "foot_a": foot_a, "foot_b": foot_b, "foot_c": foot_c,
        "torso": torso,
    }
    own = None
    for m in parts.values():
        own = m.copy() if own is None else (own | m)
    overlap = 0
    names = list(parts)
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            overlap += int((parts[na] & parts[nb]).sum())
    if int(own.sum()) != int(op.sum()) or overlap:
        raise SystemExit(f"masks do not partition the still: covered {int(own.sum())} opaque {int(op.sum())} overlap {overlap}")
    return parts


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
    """4x nearest. Multiples of 360 stay exact so the spin lands on the still."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
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


def despeckle(img):
    """Drop only pixels whose four neighbours are all empty. After rotation only."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def seal_cracks(img):
    """Fill a 1px crack that already has 3 opaque neighbours. Does not bridge the toe gaps."""
    a = img[..., 3] >= 128
    if not np.any(a):
        return img
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    hole = (~a) & (n >= 3)
    if not np.any(hole):
        return img
    ys, xs = np.nonzero(a)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = img[ys, xs]
    ty, tx = np.nonzero(hole)
    pick = np.empty(len(ty), np.int32)
    for i in range(0, len(ty), 400):
        sl = slice(i, i + 400)
        d = (ty[sl, None] - pts[None, :, 0]) ** 2 + (tx[sl, None] - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    out = img.copy()
    out[ty, tx] = cols[pick]
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
    for i in range(0, len(ty), 400):
        sl = slice(i, i + 400)
        d = (ty[sl, None] - pts[None, :, 0]) ** 2 + (tx[sl, None] - pts[None, :, 1]) ** 2
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
    """Paint a palette colour only on a transparent pixel, inset from the cell edge."""
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def socket_pivot(part, inward):
    """Joint at the stem side of a leaf. inward is +1 for a leaf whose socket is on the right."""
    ys, xs = np.nonzero(part)
    if len(ys) == 0:
        return 0.0, 0.0
    bottom = ys.max()
    sel = ys >= bottom - 2
    if inward > 0:
        x = float(xs[sel].max())
    else:
        x = float(xs[sel].min())
    return x, float(ys[sel].mean())


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

    # Socket paint is hidden by the part at rest. Stem or face colour, never a copy of the part.
    # Feet cushion the ankle (top row) only. A sole cushion would stay planted and hide the step.
    def cushion(part_name, rows, top=False):
        pm = placed[part_name]
        if not np.any(pm):
            return
        ys_p = np.nonzero(pm)[0]
        yy = np.arange(CELL_H)[:, None]
        if top:
            edge = int(ys_p.min())
            zone = pm & (yy <= edge + rows - 1)
        else:
            edge = int(ys_p.max())
            zone = pm & (yy >= edge - (rows - 1))
        fill = nearest_color(canvas, zone, placed["torso"])
        empty = layers["torso"][..., 3] < 128
        write = zone & empty
        layers["torso"][write] = fill[write]

    cushion("leaf_l", 2)
    cushion("leaf_r", 2)
    cushion("jaw", 8)
    cushion("foot_a", 1, top=True)
    cushion("foot_b", 1, top=True)
    cushion("foot_c", 1, top=True)

    pivots = {
        "leaf_l": socket_pivot(placed["leaf_l"], inward=1),
        "leaf_r": socket_pivot(placed["leaf_r"], inward=-1),
    }
    oys, oxs = np.nonzero(op_canvas(canvas))
    center = (float(oxs.mean()), float(oys.mean()))

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
    # Original still faces left. A flip turns it to screen-right.
    facing = 1.0 if flip else -1.0
    return layers, pivots, center, facing


def op_canvas(canvas):
    return canvas[..., 3] >= 128


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def foot_step(f, pair):
    """Two-step trot. pair 0 lifts on the first half, pair 1 on the second. Zero at frame 0 and 16."""
    phase = 0.0 if pair == 0 else math.pi
    th = 2 * math.pi * 2 * (f / 16.0) + phase
    s = math.sin(th) - math.sin(phase)
    c = math.cos(th) - math.cos(phase)
    lift = max(0.0, s) * 3.0
    swing = -c * 1.1
    return swing, -lift


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, leaf=0.0, leaf_spread=0, jaw=0,
        spin=0.0, recoil=0.0, ring=-1, rings=0, lines=0, burst=0, flash=0, sparks=0,
        feet=0,
    )
    if action == "idle":
        p["leaf"] = wave(f, n, 10.0, phase=0.5)
    elif action == "walk":
        p["leaf"] = wave(f, n, 8.0, phase=0.4, cycles=2.0)
        p["feet"] = 1
    elif action == "attack":
        leaf_t = [8, 16, 22, 22, 14, 0, 0, 0, 0, 0, 0, 8, 4, 2, 0, 0]
        spread_t = [1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        jaw_t = [1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        p["leaf"] = float(leaf_t[f])
        p["leaf_spread"] = spread_t[f]
        p["jaw"] = jaw_t[f]
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
            p["lines"] = 3
        if 1 <= f <= 4:
            p["ring"], p["rings"] = f - 1, 3
        if f in (11, 12):
            p["burst"] = 1
    else:
        env = [0.0, 0.5, 0.85, 1.0, 0.88, 0.66, 0.42, 0.24, 0.12, 0.05, 0.0, 0.0][f]
        p["recoil"] = 32.0 * env
        p["dx"] = -5.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 5 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def compose(layers, pivots, facing, p, center):
    """Pose the parts. Spin and recoil ignore this and rotate the rest pose instead."""
    fwd = facing
    bdx, bdy = p["dx"] * fwd, p["dy"]
    spread = int(p["leaf_spread"])
    # Positive leaf angle lifts a blade. Spread pushes the tips apart after the mirror.
    leaf_l = shift(
        rotate_nearest(layers["leaf_l"], -p["leaf"] * fwd, pivots["leaf_l"]),
        bdx - spread * fwd, bdy - spread,
    )
    leaf_r = shift(
        rotate_nearest(layers["leaf_r"], p["leaf"] * fwd, pivots["leaf_r"]),
        bdx + spread * fwd, bdy - spread,
    )
    jaw = shift(layers["jaw"], bdx, bdy + int(p["jaw"]))
    torso = shift(layers["torso"], bdx, bdy)
    feet = []
    for name, pair in (("foot_a", 0), ("foot_b", 1), ("foot_c", 0)):
        sx, sy = (0.0, 0.0)
        if p["feet"]:
            sx, sy = foot_step(p["frame"], pair)
        feet.append(shift(layers[name], bdx + sx * fwd, bdy + sy))
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(body, torso)
    for lyr in feet:
        paste(body, lyr)
    paste(body, jaw)
    paste(body, leaf_l)
    paste(body, leaf_r)
    return body


def snout_point(img, facing, center):
    a = img[..., 3] >= 128
    h = img.shape[0]
    yy = np.arange(h)[:, None]
    y0, y1 = center[1] - 8, center[1] + 6
    band = a & (yy >= y0) & (yy <= y1)
    by, bx = np.nonzero(band)
    if len(bx) == 0:
        bb = body_box(img)
        if bb is None:
            return center
        return (bb[2] if facing > 0 else bb[0], (bb[1] + bb[3]) // 2)
    if facing > 0:
        x = int(bx.max())
        y = int(np.median(by[bx >= x - 1]))
    else:
        x = int(bx.min())
        y = int(np.median(by[bx <= x + 1]))
    return x, y


def roar_rings(out, mouth, facing, age, strength):
    mx, my = mouth
    cols = (YELLOW, LEAF_HI, WHITE)
    for r_i in range(strength):
        rad = 6 + age * 3 + r_i * 4
        col = cols[min(r_i, 2)]
        for deg in range(-55, 56, 8):
            ang = math.radians(deg)
            put(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.7, col)
    return out


def speed_lines(out, facing, n, seed, length):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    rear_x = (x0 + (x1 - x0) // 5) if facing > 0 else (x1 - (x1 - x0) // 5)
    y_lo = y0 + (y1 - y0) // 3
    y_hi = y0 + (y1 - y0) * 2 // 3
    for _k in range(n):
        y = int(rng.integers(y_lo, max(y_lo + 1, y_hi)))
        off = int(rng.integers(3, 7))
        L = int(rng.integers(max(3, length // 2), length + 1))
        for i in range(L):
            put(out, rear_x - facing * (off + i), y, WHITE if i < 2 else LEAF_HI)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(size // 2, size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), WHITE if r < size - 2 else YELLOW)
    return out


def sparks(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    d = frame_idx
    cx = (x0 + x1) // 2
    pts = [
        (cx - facing * (4 + d), y0 + 2),
        (cx + facing * (3 + d), y0 + 6),
        (cx - facing * 2, y0 - 2),
    ]
    for sx, sy in pts:
        put(out, sx, sy, WHITE)
        put(out, sx - facing, sy + 1, YELLOW)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def largest_fraction(img):
    a = img[..., 3] >= 128
    if not np.any(a):
        return 1.0, 0
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
    return best / total, total


def synth(rig, action, f, palette, effects=True):
    layers, pivots, center, facing = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    p["frame"] = f
    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        rest = pose(action, f)
        rest = gen4_attack_strike.finish(rest, action, f, __file__)

        rest["leaf"] = 0.0
        rest["leaf_spread"] = 0
        rest["jaw"] = 0
        rest["feet"] = 0
        rest["dx"] = 0.0
        rest["dy"] = 0.0
        rest["frame"] = 0
        body = compose(layers, pivots, facing, rest, center)
        if recoiling:
            body = shift(body, p["dx"] * facing, 0)
            bb = body_box(body)
            pivot = center if bb is None else ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
            # Nose up, away from the snout. Under 38 degrees. No belly mirror and no squash.
            spun = rotate_nearest(body, -p["recoil"] * facing, pivot)
        else:
            spun = rotate_nearest(body, p["spin"], center)
        if spun is not body and not np.array_equal(spun, body):
            body = seal_cracks(spun)
            body = despeckle(body)
        else:
            body = spun
    else:
        body = compose(layers, pivots, facing, p, center)

    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    if p["rings"] and p["ring"] >= 0 and not spinning:
        img = roar_rings(img, snout_point(img, facing, center), facing, p["ring"], p["rings"])
    if p["lines"]:
        img = speed_lines(img, facing, p["lines"], seed=f * 17 + 5, length=8)
    if p["burst"]:
        sx, sy = snout_point(img, facing, center)
        img = impact_burst(img, (sx + facing * 5, sy), 9 if f >= 12 else 7)
    if p["sparks"]:
        img = sparks(img, f, facing)
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
    samples = {"idle": (0, 4, 8, 12), "walk": (0, 4, 8, 12), "attack": (2, 4, 7, 11), "hurt": (1, 3, 6, 10)}
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            m = tile[..., 3] >= 128
            bg[ai * CELL_H:(ai + 1) * CELL_H, si * CELL_W:(si + 1) * CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    """East row, every frame, cropped. Roar, spin, and the foot shuffle in one picture."""
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


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(STILL_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    faces_left, eye_x, eye_y, left, right = measure_facing(front)
    if not faces_left:
        raise SystemExit(f"expected a left-facing still, eye {eye_x} span {left}-{right}")
    unflipped = {"N", "NW", "W", "SW"}
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("facing left", "eye", eye_x, eye_y, "span", left, right, "parts", counts, "palette", len(palette), flush=True)

    rigs = {}
    for d in DIRS:
        rigs[d] = build_rig(front, masks, flip=d not in unflipped)

    rest = synth(rigs["W"], "idle", 0, palette)
    op = front[..., 3] >= 128
    ys, xs = np.nonzero(op)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    dst_x = (CELL_W - (x1 - x0)) // 2
    dst_y = FOOT_Y - (y1 - y0)
    placed = np.zeros_like(rest)
    placed[dst_y:dst_y + (y1 - y0), dst_x:dst_x + (x1 - x0)] = front[y0:y1, x0:x1]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (190, 190, 170), "leaf_l": (40, 180, 60), "leaf_r": (20, 120, 40),
        "jaw": (240, 200, 40), "foot_a": (220, 60, 60), "foot_b": (40, 90, 220), "foot_c": (220, 40, 180),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10)]
        want += [("attack", "W", f) for f in (0, 2, 3, 5, 6, 7, 8, 11, 13)]
        want += [("attack", "E", f) for f in (2, 7, 11)]
        want += [("hurt", "W", f) for f in (0, 1, 3, 6, 9)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        zoom = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
        Image.fromarray(zoom, "RGBA").save(work / "samples_x2.png")
        # Body-only opaque counts through the spin, before rings and lines.
        base = int((synth(rigs["W"], "idle", 0, palette, effects=False)[..., 3] >= 128).sum())
        for f in range(16):
            body = synth(rigs["W"], "attack", f, palette, effects=False)
            n = int((body[..., 3] >= 128).sum())
            frac, _ = largest_fraction(body)
            print(f"attack W {f:02d} opaque {n} ({n / base:.3f}) main {frac:.3f}", flush=True)
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
    alphas = sorted({int(v) for v in np.unique(master[..., 3])})
    clip = {}
    spin_stats = []
    base_n = None
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and d == "W" and 5 <= f <= 10:
                    body = synth(rigs["W"], "attack", f, palette, effects=False)
                    n = int((body[..., 3] >= 128).sum())
                    frac, _ = largest_fraction(body)
                    if base_n is None:
                        base_n = int((synth(rigs["W"], "idle", 0, palette, effects=False)[..., 3] >= 128).sum())
                    spin_stats.append({"f": f, "opaque": n, "ratio": round(n / base_n, 3), "main": round(frac, 3)})
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "spin_body": spin_stats, "rest_diff": 0,
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
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1200], flush=True)

    bad_spin = [s for s in spin_stats if s["ratio"] < 0.82 or s["main"] < 0.90]
    if off or clip or alphas != [0, 255] or bad_spin:
        raise SystemExit(f"qa failed off={len(off)} clip={clip} alphas={alphas} spin={bad_spin}")

    meta = {
        "generator": "forge_turtwig_actions", "version": VERSION, "key": "0387_turtwig",
        "dex": 387, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "face_right": ["S", "SE", "E", "NE"], "still_faces": "left",
        "eye": [eye_x, eye_y], "parts": counts,
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "source": "front.png 96x96", "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
