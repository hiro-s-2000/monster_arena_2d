#!/usr/bin/env python3
"""Kricketot (0401) 8-direction sheet from this form's 96x96 front.png.

Measured on normal/front.png (opaque 956, bbox x30-61 y26-70, 32x45, 12 colours,
alpha already 0 or 255). One 4-connected silhouette.

  Two antennae, not more. Left stalk x30-48 y26-39. Right stalk is the tip
  x48-56 y26-33 plus the base x46-52 y32-39. No wings, no tail, no extra legs.
  Two feet already drawn, separated by a gap:
    screen-left foot  x35-41 y65-68
    screen-right foot x49-54 y66-70 (the lowest pixel is this foot)
  One arm sticks out on the screen-left, x33-37 y50-57. The red right flank
  is body colour, not a second arm. The isolated brown oval x41-45 y37-40 is
  the eye and stays on the head. The brown mark inside the red mustache
  (x39-47 y44-49) is the mouth. The eye is left of the body centroid (45.5)
  and the only arm is on the left, so the still faces left.
  Unflipped dirs are N NW W SW. Flipped dirs are S SE E NE. E faces screen-right.

  idle   : 1px breath, antennae sway, feet planted. Frame 0 is the still.
  walk   : the two existing feet step in turn. Antennae and the one arm follow.
  attack : 0-4 mouth plus antennae and a few rings; 5-10 in-place 360 of the
           whole body, no squash; 11-12 a small impact; 13-15 settle.
  hurt   : one-piece lean, 38 degrees max, no belly flip.

Palette-locked. front/back png and gif are not written.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_kricketot_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen4/0401_kricketot/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0401_kricketot/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left. These four keep the drawing; the other four are mirrored.
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "kricketot_v2"
KEY = "0401_kricketot"

INK = (16, 16, 16)
CREAM = (255, 238, 172)
GRAY = (123, 115, 98)
RED = (230, 106, 98)
DARK = (82, 74, 65)
DRED = (180, 74, 57)
GOLD = (230, 197, 106)
BROWN = (106, 65, 32)
DBROWN = (90, 32, 0)
OLIVE = (139, 106, 8)
MOUTH = (148, 82, 24)
LIGHT = (156, 156, 148)
PARTS = ("body", "antenna_l", "antenna_r", "arm", "foot_l", "foot_r", "mouth")


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


def label_fill(mask):
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    seen = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    nxt = 1
    boxes = {}
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
        for cy, cx in pts:
            labels[cy, cx] = nxt
        ys2 = np.array([p[0] for p in pts])
        xs2 = np.array([p[1] for p in pts])
        boxes[nxt] = (int(xs2.min()), int(ys2.min()), int(xs2.max()), int(ys2.max()), len(pts), float(xs2.mean()), float(ys2.mean()))
        nxt += 1
    return labels, boxes


def colour_mask(img, colour):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(colour, np.uint8), axis=-1)


def segment(img):
    """Two antennae, one arm, two feet, the mustache mouth. The eye stays on the body."""
    h, w = img.shape[:2]
    opaque = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    ink = colour_mask(img, INK)
    fill = opaque & ~ink
    labels, boxes = label_fill(fill)
    ant_l = np.zeros_like(opaque)
    ant_r = np.zeros_like(opaque)
    arm = np.zeros_like(opaque)
    foot_l = np.zeros_like(opaque)
    foot_r = np.zeros_like(opaque)
    eye = np.zeros_like(opaque)
    for lab, (x0, y0, x1, y1, n, cx, cy) in boxes.items():
        m = labels == lab
        # The eye is a small brown oval above the mustache. Check it before the stalks.
        if 36 <= cy <= 41 and 40 <= cx <= 46 and n < 30 and y0 >= 36:
            eye |= m
        elif cy < 42 and y1 <= 40 and cx < 47:
            ant_l |= m
        elif cy < 42 and y1 <= 40 and cx >= 46:
            ant_r |= m
        elif y0 >= 48 and y1 <= 60 and x1 <= 40 and n < 80:
            arm |= m
        elif y0 >= 64 and cx < 46:
            foot_l |= m
        elif y0 >= 64 and cx >= 46:
            foot_r |= m
    # Centre-forehead olive belongs to the head, even if a stalk flood touched it.
    centre = (yy >= 33) & (xx >= 42) & (xx <= 51) & colour_mask(img, OLIVE)
    ant_l &= ~centre
    ant_r &= ~centre
    mouth_col = colour_mask(img, MOUTH) | colour_mask(img, BROWN)
    mouth = mouth_col & (yy >= 44) & (yy <= 49) & (xx >= 39) & (xx <= 47) & ~eye
    parts = {
        "antenna_l": ant_l,
        "antenna_r": ant_r,
        "arm": arm,
        "foot_l": foot_l,
        "foot_r": foot_r,
        "mouth": mouth,
    }
    body_fill = fill & ~ant_l & ~ant_r & ~arm & ~foot_l & ~foot_r & ~mouth
    # Outline that touches only one part moves with that part. Shared outline stays on the body.
    touch_body = dilate(body_fill | eye, 1)
    for name in ("antenna_l", "antenna_r", "arm", "foot_l", "foot_r", "mouth"):
        private = dilate(parts[name], 1) & ink & ~touch_body
        # An outline pixel claimed by two parts stays on the body.
        parts[name] = parts[name] | private
    overlap_ink = np.zeros_like(opaque)
    claimed = np.zeros_like(opaque)
    for name in ("antenna_l", "antenna_r", "arm", "foot_l", "foot_r", "mouth"):
        overlap_ink |= claimed & parts[name]
        claimed |= parts[name]
    for name in ("antenna_l", "antenna_r", "arm", "foot_l", "foot_r", "mouth"):
        parts[name] &= ~overlap_ink
    union = np.zeros_like(opaque)
    for name in ("antenna_l", "antenna_r", "arm", "foot_l", "foot_r", "mouth"):
        union |= parts[name]
    # Shared outline and the eye stay on the body. Nothing is dropped.
    body = opaque & ~union
    parts["body"] = body
    parts["eye"] = eye & body
    return parts


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
    """4x nearest. Width is never scaled. 0 and 360 return the same pixels."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.05 or not np.any(layer[..., 3] >= 128):
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
    """Drop a pixel only when all four neighbours are empty. After rotation only."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


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
    return out


def put_empty(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def edge_rows(mask, rows, side):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros_like(mask)
    yy = np.arange(mask.shape[0])[:, None]
    xx = np.arange(mask.shape[1])[None, :]
    if side == "top":
        return mask & (yy <= int(ys.min()) + rows - 1)
    if side == "bottom":
        return mask & (yy >= int(ys.max()) - (rows - 1))
    if side == "right":
        return mask & (xx >= int(xs.max()) - (rows - 1))
    return mask & (xx <= int(xs.min()) + rows - 1)


def pivot_of(mask, side):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return (0.0, 0.0)
    if side == "bottom":
        lim = int(ys.max())
        sel = ys >= lim - 1
    elif side == "top":
        lim = int(ys.min())
        sel = ys <= lim + 1
    elif side == "right":
        lim = int(xs.max())
        sel = xs >= lim - 1
    else:
        lim = int(xs.min())
        sel = xs <= lim + 1
    return float(xs[sel].mean()), float(ys[sel].mean())


def place_geometry(img):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    return y0, y1, x0, x1, dst_y, dst_x


def build_rig(img, masks, flip):
    y0, y1, x0, x1, dst_y, dst_x = place_geometry(img)
    sh, sw = y1 - y0, x1 - x0
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    placed = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm
    # Socket copies stay under the moving part so a 1-2px lift does not open a hole.
    cushion = (
        edge_rows(placed["antenna_l"], 2, "bottom")
        | edge_rows(placed["antenna_r"], 2, "bottom")
        | edge_rows(placed["arm"], 2, "right")
        | edge_rows(placed["foot_l"], 2, "top")
        | edge_rows(placed["foot_r"], 2, "top")
        | edge_rows(placed["mouth"], 2, "top")
    )
    layers = {}
    for name in PARTS:
        lyr = np.zeros_like(canvas)
        lyr[placed[name]] = canvas[placed[name]]
        layers[name] = lyr
    layers["body"][cushion] = canvas[cushion]
    pivots = {
        "antenna_l": pivot_of(placed["antenna_l"], "bottom"),
        "antenna_r": pivot_of(placed["antenna_r"], "bottom"),
        "arm": pivot_of(placed["arm"], "right"),
        "foot_l": pivot_of(placed["foot_l"], "top"),
        "foot_r": pivot_of(placed["foot_r"], "top"),
        "mouth": pivot_of(placed["mouth"], "top"),
    }
    bys, bxs = np.nonzero(canvas[..., 3] >= 128)
    sole = (float(np.median(bxs[bys >= bys.max() - 1])), float(bys.max()))
    center = (float(bxs.mean()), float(bys.mean()))
    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        sole = (CELL_W - 1 - sole[0], sole[1])
        center = (CELL_W - 1 - center[0], center[1])
        canvas = canvas[:, ::-1].copy()
    facing = 1.0 if flip else -1.0
    pivots["sole"] = sole
    pivots["center"] = center
    return layers, canvas, pivots, facing


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        bdy=0, lift_l=0, lift_r=0, step_l=0, step_r=0,
        ant_l=0.0, ant_r=0.0, arm_dx=0, jaw=0,
        spin=0.0, hurt=0.0, knock=0.0, rings=0, ring=0,
        lines=0, burst=0, sparks=0,
    )
    if action == "idle":
        p["bdy"] = -int(round(max(0.0, wave(f, n, 1.0))))
        p["ant_l"] = wave(f, n, 6.0)
        p["ant_r"] = wave(f, n, 6.0, phase=0.9)
        p["arm_dx"] = int(round(wave(f, n, 1.0)))
    elif action == "walk":
        s = math.sin(2 * math.pi * 2 * f / n)
        p["bdy"] = -1 if abs(s) > 0.65 else 0
        p["lift_l"] = int(round(2 * max(0.0, s)))
        p["lift_r"] = int(round(2 * max(0.0, -s)))
        p["step_l"] = 1 if p["lift_l"] else 0
        p["step_r"] = 1 if p["lift_r"] else 0
        p["ant_l"] = wave(f, n, 8.0, cycles=2.0)
        p["ant_r"] = wave(f, n, 8.0, cycles=2.0, phase=math.pi)
        p["arm_dx"] = int(round(wave(f, n, 1.0, cycles=2.0)))
    elif action == "attack":
        # 0 still, 1-4 roar, 5-10 one-piece spin, 11-12 impact, 13-15 settle.
        p["jaw"] = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["ant_l"] = [0, 10, 16, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0][f]
        p["ant_r"] = [0, -10, -16, -16, -8, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0][f]
        p["arm_dx"] = [0, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["rings"] = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["ring"] = [0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        if 5 <= f <= 10:
            p["spin"] = 60.0 * (f - 4)
        if 6 <= f <= 9:
            p["lines"] = 2
        if f in (11, 12):
            p["burst"] = 1
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 38.0 * env
        p["knock"] = 3.0 * env
        p["sparks"] = 1 if 2 <= f <= 5 else 0
    return p


def seal_joint(img, part, pivot, radius=3):
    """Fill a 1px crack at the socket with a neighbouring body colour. Not the foot gap."""
    a = img[..., 3] >= 128
    p = part[..., 3] >= 128
    if not np.any(p):
        return img
    body = a & ~p
    gap = dilate(p, 1) & dilate(body, 1) & ~a
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    gap &= (xx - pivot[0]) ** 2 + (yy - pivot[1]) ** 2 <= float(radius * radius)
    ys, xs = np.nonzero(gap)
    if len(ys) == 0:
        return img
    by, bx = np.nonzero(body)
    out = img.copy()
    for y, x in zip(ys.tolist()[:12], xs.tolist()[:12]):
        d = (bx - x) ** 2 + (by - y) ** 2
        i = int(np.argmin(d))
        out[y, x, :3] = img[int(by[i]), int(bx[i]), :3]
        out[y, x, 3] = 255
    return out


def assemble_pose(layers, pivots, p, facing):
    """Body, antennae, the one arm, two feet, then the mouth. Rest pose matches the still."""
    bdy = p["bdy"]
    sgn = -facing
    body = shift(layers["body"], 0, bdy)
    ant_l = shift(rotate_nearest(layers["antenna_l"], p["ant_l"] * sgn, pivots["antenna_l"]), 0, bdy)
    ant_r = shift(rotate_nearest(layers["antenna_r"], p["ant_r"] * sgn, pivots["antenna_r"]), 0, bdy)
    arm = shift(layers["arm"], facing * p["arm_dx"], bdy)
    foot_l = shift(layers["foot_l"], facing * p["step_l"], -p["lift_l"])
    foot_r = shift(layers["foot_r"], facing * p["step_r"], -p["lift_r"])
    mouth = shift(layers["mouth"], 0, bdy + p["jaw"])
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for lyr in (body, ant_l, ant_r, arm, foot_l, foot_r):
        paste(img, lyr)
    if p["jaw"]:
        home = shift(layers["mouth"], 0, bdy)
        vacated = (home[..., 3] >= 128) & (mouth[..., 3] < 128) & (img[..., 3] < 128)
        img[vacated, :3] = DRED
        img[vacated, 3] = 255
    paste(img, mouth)
    if p["ant_l"] or p["ant_r"] or bdy:
        img = seal_joint(img, ant_l, (pivots["antenna_l"][0], pivots["antenna_l"][1] + bdy))
        img = seal_joint(img, ant_r, (pivots["antenna_r"][0], pivots["antenna_r"][1] + bdy))
    if p["arm_dx"] or bdy:
        img = seal_joint(img, arm, (pivots["arm"][0] + facing * p["arm_dx"], pivots["arm"][1] + bdy))
    if p["lift_l"] or p["step_l"]:
        img = seal_joint(img, foot_l, pivots["foot_l"])
    if p["lift_r"] or p["step_r"]:
        img = seal_joint(img, foot_r, pivots["foot_r"])
    return img


def roar_rings(out, mouth, facing, age, strength):
    """A short arc in front of the mouth. Few pixels, so the silhouette stays near the still."""
    mx, my = mouth
    cols = (RED, INK, GOLD)
    for i in range(strength):
        rad = 6 + age * 2 + i * 3
        col = cols[i % 3]
        for deg in (-40, -20, 0, 20, 40):
            ang = math.radians(deg)
            put_empty(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.45, col)
    return out


def speed_lines(out, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rear = x1 + 4 if facing < 0 else x0 - 4
    step = -facing
    span = max(1, y1 - y0)
    for k, frac in enumerate((0.32, 0.62)):
        y = y0 + int(span * frac)
        for i in range(4):
            put_empty(out, rear + step * i, y, GOLD if i < 2 else INK)
    return out


def impact_burst(out, center, facing):
    cx, cy = center
    cols = (RED, GOLD, INK)
    dots = ((5, 0), (8, 0), (6, -3), (6, 3), (4, -2), (4, 2))
    for i, (dx, dy) in enumerate(dots):
        put_empty(out, cx + facing * dx, cy + dy, cols[i % 3])
    return out


def sparks(out, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    x = x1 + 2 if facing < 0 else x0 - 2
    y = y0 + 8
    put_empty(out, x, y, CREAM)
    put_empty(out, x - facing * 2, y + 2, INK)
    put_empty(out, x - facing, y - 2, GOLD)
    return out


def neutral_bitmap(layers):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in PARTS:
        paste(img, layers[name])
    return img


def synth(rig, action, f, palette):
    layers, _canvas, pivots, facing = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = (p["spin"] % 360.0) > 0.5
    recoiling = p["hurt"] > 0.45
    if spinning or recoiling:
        # One bitmap. A |cos| squash turns this 32px body into a line.
        base = neutral_bitmap(layers)
        if spinning:
            img = rotate_nearest(base, p["spin"], pivots["center"])
        else:
            # Left-facing leans the face up and backward. Angle stays <= 38.
            img = rotate_nearest(base, -facing * p["hurt"], pivots["sole"])
            img = shift(img, -facing * p["knock"], 0)
        img = despeckle(img)
    else:
        img = assemble_pose(layers, pivots, p, facing)
    img = snap(img, palette)
    if p["rings"] and not spinning:
        mouth = (pivots["mouth"][0], pivots["mouth"][1] + p["jaw"] + p["bdy"])
        img = roar_rings(img, mouth, facing, p["ring"], p["rings"])
    if p["lines"]:
        img = speed_lines(img, facing)
    if p["burst"]:
        bb = body_box(img)
        if bb:
            tip = bb[2] + 3 if facing > 0 else bb[0] - 3
            img = impact_burst(img, (tip, (bb[1] + bb[3]) // 2), facing)
    if p["sparks"]:
        img = sparks(img, facing)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (SRC_DIR / "front.png").read_bytes()
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
    """South row, every frame, cropped so the 32px cricket and the spin are readable."""
    scale = 3
    x0, x1 = 48, 132
    y0, y1 = 88, 160
    cw, ch = (x1 - x0) * scale, (y1 - y0) * scale
    cols = 8
    blocks = []
    for act in ACTIONS:
        band = sheets[act][:CELL_H]
        frames = []
        for f in range(FRAMES[act]):
            tile = band[y0:y1, f * CELL_W + x0:f * CELL_W + x1]
            frames.append(np.repeat(np.repeat(tile, scale, 0), scale, 1))
        blocks.append(frames)
    rows = sum(math.ceil(len(b) / cols) for b in blocks)
    bg = np.full((rows * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    r = 0
    for frames in blocks:
        for i, up in enumerate(frames):
            rr, cc = divmod(i, cols)
            y = (r + rr) * ch
            x = cc * cw
            m = up[..., 3] >= 128
            bg[y:y + ch, x:x + cw][m] = up[m]
        r += math.ceil(len(frames) / cols)
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 6
    crops = []
    for act in ACTIONS:
        row = []
        band = sheets[act][6 * CELL_H:7 * CELL_H]
        for f in range(FRAMES[act]):
            tile = band[:, f * CELL_W:(f + 1) * CELL_W]
            a = tile[..., 3] >= 128
            if not np.any(a):
                continue
            ys, xs = np.nonzero(a)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(CELL_H, int(ys.max()) + 1 + pad)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(CELL_W, int(xs.max()) + 1 + pad)
            crop = tile[y0:y1, x0:x1]
            row.append(np.repeat(np.repeat(crop, 3, 0), 3, 1))
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
        "dex": 401, "form": "normal", "variant": "kricketot", "gen": "gen4",
        "dir": "monsterSprite/gen4/0401_kricketot/normal",
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
        key = f'"{KEY}":'
        if key in text:
            start = text.index(key) + len(key)
            end = start
            depth = 0
            begun = False
            while end < len(text):
                ch = text[end]
                if ch == "{":
                    depth += 1
                    begun = True
                elif ch == "}":
                    depth -= 1
                    if begun and depth == 0:
                        end += 1
                        break
                end += 1
            text = text[:start] + blob + text[end:]
        else:
            text = text.replace(prefix + "{", prefix + "{" + key + blob + ",", 1)
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(text)
        json.loads(text[len(prefix):].rstrip().rstrip(";"))
        tmp.replace(DIR8)

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
                raise SystemExit("queue has no 0401_kricketot")
            tmp = QUEUE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def attack_ratios(rig, palette):
    rows = []
    base = None
    for f in range(16):
        fr = synth(rig, "attack", f, palette)
        c = fr[..., 3] >= 128
        n = int(c.sum())
        if f == 0:
            base = n
        ys, xs = np.nonzero(c)
        bw = int(xs.max() - xs.min() + 1) if len(xs) else 0
        bh = int(ys.max() - ys.min() + 1) if len(ys) else 0
        sizes = components(c)
        main = sizes[0] / n if sizes and n else 0
        rows.append({
            "f": f, "opaque": n,
            "ratio": round(n / base, 4) if base else 0,
            "w": bw, "h": bh, "main": round(main, 4),
            "parts": sizes[:4],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    opaque = front[..., 3] >= 128
    covered = np.zeros_like(opaque)
    overlap = 0
    for name in PARTS:
        overlap += int((covered & masks[name]).sum())
        covered |= masks[name]
    counts = {k: int(masks[k].sum()) for k in PARTS}
    counts["eye"] = int(masks["eye"].sum())
    counts["missing"] = int((opaque & ~covered).sum())
    counts["overlap"] = overlap
    print("parts", counts, "palette", len(palette), flush=True)
    if counts["missing"] or overlap:
        raise SystemExit("segmentation does not partition the still")
    if counts["antenna_l"] < 40 or counts["antenna_r"] < 40:
        raise SystemExit("an antenna mask collapsed")
    if counts["foot_l"] < 12 or counts["foot_r"] < 12:
        raise SystemExit("a foot mask collapsed")
    if counts["arm"] < 12:
        raise SystemExit("the one arm mask collapsed")
    if counts["mouth"] < 6:
        raise SystemExit("mouth mask collapsed")
    if np.any(masks["mouth"] & masks["eye"]):
        raise SystemExit("mouth ate the eye")
    eye_ys, eye_xs = np.nonzero(masks["eye"])
    op_ys, op_xs = np.nonzero(opaque)
    eye_x = float(eye_xs.mean()) if len(eye_xs) else 99
    body_x = float(op_xs.mean())
    arm_x = float(np.nonzero(masks["arm"])[1].mean())
    print("facing eye_x", round(eye_x, 1), "arm_x", round(arm_x, 1), "body_x", round(body_x, 1), flush=True)
    if not (eye_x < body_x and arm_x < body_x):
        raise SystemExit("expected a left-facing still")

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}

    rest = synth(rigs["W"], "idle", 0, palette)
    canvas = rigs["W"][1]
    differ = np.any(rest != canvas, axis=2) & ((rest[..., 3] >= 128) | (canvas[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()):
        raise SystemExit("idle frame 0 does not match the still")
    south = synth(rigs["S"], "idle", 0, palette)
    mirror = canvas[:, ::-1].copy()
    ed = np.any(south != mirror, axis=2) & ((south[..., 3] >= 128) | (mirror[..., 3] >= 128))
    print("south mirror pixels changed", int(ed.sum()), flush=True)
    if int(ed.sum()):
        raise SystemExit("south idle 0 is not the mirror of the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "body": (197, 170, 90), "antenna_l": (220, 60, 60), "antenna_r": (40, 90, 220),
        "arm": (40, 180, 80), "foot_l": (240, 140, 40), "foot_r": (240, 200, 40),
        "mouth": (230, 80, 160),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    rows = attack_ratios(rigs["S"], palette)
    ratios = [row["ratio"] for row in rows]
    print("south attack opaque ratios", " ".join(f"{r:.4f}" for r in ratios), flush=True)
    bad_ratio = [row for row in rows if row["ratio"] < 0.90 or row["ratio"] > 1.08]
    rest_w = rows[0]["w"]
    rest_h = rows[0]["h"]
    thin = [row for row in rows if 5 <= row["f"] <= 10 and (row["w"] < rest_w * 0.75 or row["h"] < rest_h * 0.75)]
    weak = [row for row in rows if 5 <= row["f"] <= 10 and row["main"] < 0.90]
    broken = []
    for f in range(5, 11):
        if rows[f]["main"] < 0.90 or not rows[f]["parts"] or rows[f]["parts"][0] < 800:
            broken.append(("attack", f, rows[f]["parts"]))
    for f in (1, 3, 5, 6, 9):
        fr = synth(rigs["W"], "hurt", f, palette)
        sizes = components(fr[..., 3] >= 128)
        if not sizes or sizes[0] < 800:
            broken.append(("hurt", f, sizes[:4]))
    print("piece check", broken if broken else "ok", "thin", thin, "weak", weak, flush=True)
    if bad_ratio or thin or weak or broken:
        raise SystemExit("attack ratio or spin failed before writing the sheet")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "W", f) for f in (0, 2, 4, 5, 7, 9, 10, 11, 13, 15)]
        want += [("attack", "S", f) for f in (2, 7, 11)]
        want += [("hurt", "W", f) for f in (0, 3, 5, 8, 11)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        cols = 8
        x0, x1, y0, y1 = 48, 132, 88, 160
        zw, zh = (x1 - x0) * 4, (y1 - y0) * 4
        rows_n = math.ceil(len(tiles) / cols)
        sheet = np.full((rows_n * zh, cols * zw, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            up = np.repeat(np.repeat(tile[y0:y1, x0:x1], 4, 0), 4, 1)
            m = up[..., 3] >= 128
            sheet[r * zh:(r + 1) * zh, c * zw:(c + 1) * zw][m] = up[m]
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
    actions = SRC_DIR / "actions"
    actions.mkdir(exist_ok=True)
    preview.save(actions / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    # South row is the flipped still. Opaque height is the same 45px body.
    na = neutral[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = [int(v) for v in np.unique(master[..., 3])]
    bad_alpha = [v for v in alphas if v not in (0, 255)]
    clip = {}
    attack_s = []
    base_n = int((sheets["attack"][:CELL_H, :CELL_W, 3] >= 128).sum())
    rest_w = 0
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and d == "S":
                    n = int(c.sum())
                    ys, xs = np.nonzero(c)
                    bw = int(xs.max() - xs.min() + 1) if len(xs) else 0
                    bh = int(ys.max() - ys.min() + 1) if len(ys) else 0
                    if f == 0:
                        rest_w = bw
                    sizes = components(c)
                    main = sizes[0] / n if sizes and n else 0
                    attack_s.append({
                        "f": f, "opaque": n,
                        "ratio": round(n / base_n, 4) if base_n else 0,
                        "w": bw, "h": bh, "main": round(main, 4),
                    })
    ratios = [row["ratio"] for row in attack_s]
    print("south attack opaque ratios", " ".join(f"{r:.4f}" for r in ratios), flush=True)
    bad_ratio = [row for row in attack_s if row["ratio"] < 0.90 or row["ratio"] > 1.08]
    thin = [row for row in attack_s if 5 <= row["f"] <= 10 and (row["w"] < rest_w * 0.75 or row["h"] < sprite_h * 0.75)]
    weak = [row for row in attack_s if 5 <= row["f"] <= 10 and row["main"] < 0.90]
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "attack_S_opaque": attack_s,
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        qa = work / "qa"
        qa.mkdir(exist_ok=True)
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
    (work / "qa").mkdir(exist_ok=True)
    (work / "qa" / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("parts", "attack_S_opaque")}, ensure_ascii=False)[:900], flush=True)

    meta = {
        "generator": "forge_kricketot_actions", "version": VERSION, "key": KEY,
        "dex": 401, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "source": "front.png 96x96", "gifs_untouched": True,
        "facing": "left", "antennae": 2, "feet": 2, "arms": 1,
        "archetype": "biped_cricket",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = not off and not bad_alpha and not clip and not bad_ratio and not thin and not weak and sprite_h > 0
    if not ok:
        print("QA FAIL", "off", len(off), "alpha", bad_alpha, "clip", clip,
              "bad_ratio", bad_ratio, "thin", thin, "weak", weak, flush=True)
        update_queue("needs_fix")
        raise SystemExit(1)
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    publish_dir8(sprite_h)
    update_queue("done")
    print("published", KEY, VERSION, "h", sprite_h, flush=True)


if __name__ == "__main__":
    main()
