#!/usr/bin/env python3
"""Mime Jr. (0439) 8-direction action sheet from this form's 96x96 front.png.

Measured on normal/front.png only (no other species' mask or rig):
  931 opaque pixels, alpha already 0 or 255, bbox x28-66 y20-77, 13 colours.
  Pompom tuft: white ball y20-27 plus the curled stem y27-31 (one piece).
  Blue bob hat/hair y32-57 frames a pink face. Face, hair and body = torso.
  Eyes: ink with one white glint each, glints at (36,52) and (47,53).
  Red nose ball x37-43 y53-57. Mouth: maroon smile (41..44,58) + (45,57),(46,57).
  Nose mean x (40) is left of the body centroid: the still faces LEFT.
  So N NW W SW keep the pixels and S SE E NE are mirrored (E = screen right).
  Two thin pink arms:
    arm_l  x28-41 y61-65, straight out forward (screen left), shoulder x41.
    arm_r  x56-65 y54-60, raised back/up (screen right), shoulder (55,59).
  Two blue feet:
    foot_near  the pointed foot y72-77, the only sole on the ground (y77).
    foot_far   the bent back foot x52-59 y66-71, off the ground.
  No tail, no wings, no extra legs.

  idle   : mime sway. Arms drift in counter-phase, pompom bobs, far foot taps.
           Frame 0 and the loop end are the still.
  walk   : tiptoe two-step. Pointed foot lifts and reaches, far foot swings
           opposite, arms counter-swing, pompom trails.
  attack : 0-4 roar: the existing mouth opens (wedge) + three roar rings, arms
           flare up, pompom whips back. 5-12 mime "barrier push": the forward
           arm thrusts, the back arm swings over, a psychic wall flies forward,
           11-12 impact shatter. 13-15 settle to the still.
           No whole-body planar spin (2026-11-04 rule).
  hurt   : 12 frames, white flash, posed flail rotated as ONE piece about the
           sole (max 34 deg), knock back, recover to the still.

Palette-locked to this still. front/back png and gif are never written.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen4_attack_strike  # noqa: E402

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
SRC_DIR = ROOT / "monsterSprite/gen4/0439_mime-jr/normal"
ALT_SRC = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0439_mime-jr/normal")
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = ALT_SRC / "agent-sprite-forge"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
BOUNDS = ROOT / "data/sprite_bounds.js"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

KEY = "0439_mime-jr"
VERSION = "mime-jr_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148

# Colours measured on this still.
INK = (0, 0, 0)
NAVY = (41, 49, 90)
BLUE = (49, 74, 148)
SKY = (82, 106, 213)
GRAY = (106, 98, 106)
MAROON = (115, 32, 74)
LILAC = (189, 172, 189)
RED = (197, 32, 74)
ROSE = (222, 123, 172)
PEARL = (222, 213, 222)
ORANGE = (238, 90, 57)
PINK = (255, 197, 213)
WHITE = (255, 255, 255)
PALETTE_REQUIRED = (INK, NAVY, BLUE, SKY, GRAY, MAROON, LILAC, RED, ROSE, PEARL, ORANGE, PINK, WHITE)
HAIR = (NAVY, BLUE, SKY)

# cushion first, back limbs, torso, front limbs, pompom on top.
DRAW = ("cushion", "foot_far", "arm_r", "torso", "foot_near", "arm_l", "tuft")
PARTS = ("tuft", "arm_l", "arm_r", "foot_near", "foot_far")
EXPECT = {"tuft": 71, "arm_l": 46, "arm_r": 42, "foot_near": 34, "foot_far": 26}

HURT_DEG = [0, 14, 26, 32, 34, 30, 22, 14, 7, 3, 1, 0]
HURT_DX = [0, 1, 2, 3, 3, 3, 2, 2, 1, 1, 0, 0]


# ---------------------------------------------------------------- helpers

def flood(seeds, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow, dtype=bool)
    q = deque()
    for y, x in zip(*np.nonzero(seeds & allow)):
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


def comp_sizes(mask):
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    sizes = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        n = 0
        while stack:
            y, x = stack.pop()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        sizes.append(n)
    return sorted(sizes, reverse=True)


def color_mask(img, cols):
    m = np.zeros(img.shape[:2], dtype=bool)
    for c in cols:
        m |= np.all(img[..., :3] == np.array(c, np.uint8), axis=-1)
    return m & (img[..., 3] >= 128)


def px(img, x, y):
    return tuple(int(v) for v in img[y, x, :3])


# ---------------------------------------------------------------- segmentation

def segment(img):
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    if int(a.sum()) != 931:
        raise SystemExit(f"still opaque changed: {int(a.sum())}")
    ys, xs = np.nonzero(a)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    if bbox != (28, 20, 66, 77):
        raise SystemExit(f"still bbox changed: {bbox}")
    for x, y in ((36, 52), (47, 53)):
        if px(img, x, y) != WHITE:
            raise SystemExit(f"eye glint moved at {x},{y}")
    for x, y in ((35, 52), (37, 52), (46, 53), (48, 53)):
        if px(img, x, y) != INK:
            raise SystemExit(f"eye ink moved at {x},{y}")
    for x, y in ((41, 58), (42, 58), (43, 58), (44, 58), (45, 57), (46, 57)):
        if px(img, x, y) != MAROON:
            raise SystemExit(f"mouth pixel moved at {x},{y}")
    nose = color_mask(img, (ORANGE, RED)) & (np.mgrid[0:h, 0:w][0] <= 57)
    nys, nxs = np.nonzero(nose)
    if not (float(nxs.mean()) < float(xs.mean()) - 3):
        raise SystemExit("nose is not left of centre; remeasure facing")

    yy, xx = np.mgrid[0:h, 0:w]
    hair = color_mask(img, HAIR)
    tuft = a & (yy <= 31)
    arm_l = a & (xx <= 41) & (yy >= 61) & (yy <= 65) & ~((yy == 61) & (xx >= 37))
    arm_r = a & (xx >= 56) & (yy >= 54) & (yy <= 60) & ~hair & ~((yy <= 56) & (xx < 60))
    foot_near = a & (yy >= 72)
    foot_far = flood(a & (yy == 68) & (xx == 57), a & (xx >= 52) & (yy >= 66) & (yy <= 71)
                     & ~color_mask(img, (MAROON, PINK, ROSE, RED, ORANGE, WHITE)))
    parts = {"tuft": tuft, "arm_l": arm_l, "arm_r": arm_r, "foot_near": foot_near, "foot_far": foot_far}
    used = np.zeros_like(a)
    for name, m in parts.items():
        if np.any(used & m):
            raise SystemExit(f"overlap at {name}")
        used |= m
    parts["torso"] = a & ~used
    for name, n in EXPECT.items():
        got = comp_sizes(parts[name])
        if len(got) != 1 or got[0] != n:
            raise SystemExit(f"{name} is not the measured single piece: {got}")
    tor = comp_sizes(parts["torso"])
    if len(tor) != 1:
        raise SystemExit(f"torso is not one piece: {tor}")
    # only the pointed foot touches the ground.
    if not np.any(foot_near[77]) or np.any(foot_far[72:]):
        raise SystemExit("ground contact changed")
    return parts


# ---------------------------------------------------------------- rig

def edge_pivot(mask, side):
    ys, xs = np.nonzero(mask)
    if side == "bottom":
        sel = ys >= ys.max() - 1
    elif side == "top":
        sel = ys <= ys.min() + 1
    elif side == "left":
        sel = xs <= xs.min() + 1
    else:
        sel = xs >= xs.max() - 1
    return float(xs[sel].mean()), float(ys[sel].mean())


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    d = (ty[:, None].astype(np.int32) - ys[None, :].astype(np.int32)) ** 2 + \
        (tx[:, None].astype(np.int32) - xs[None, :].astype(np.int32)) ** 2
    layer[ty, tx] = cols[d.argmin(axis=1)]
    return layer


def build_rig(img, masks, flip):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - (sh - 1)
    ox, oy = dst_x - x0, dst_y - y0  # still -> cell offset

    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    placed = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), dtype=bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm
    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr
    # Joint cushion: torso-coloured fill under the joints only.
    cushion = np.zeros_like(canvas)
    for part, rad in (("arm_l", 2), ("arm_r", 2), ("foot_near", 2), ("foot_far", 2), ("tuft", 1)):
        zone = placed[part] & dilate(placed["torso"], rad)
        fill = nearest_color(canvas, zone, placed["torso"] & ~color_mask(canvas, (WHITE,)))
        cushion[zone] = fill[zone]
    layers["cushion"] = cushion

    pivots = {
        "tuft": edge_pivot(placed["tuft"], "bottom"),
        "arm_l": edge_pivot(placed["arm_l"], "right"),
        "arm_r": (55.5 + ox, 59.5 + oy),
        "foot_near": edge_pivot(placed["foot_near"], "top"),
        "foot_far": edge_pivot(placed["foot_far"], "top"),
    }
    sole_y = FOOT_Y
    sole_x = float(np.nonzero(placed["foot_near"][sole_y])[0].mean())
    mouth = [(x + ox, y + oy) for x, y in ((41, 58), (42, 58), (43, 58), (44, 58), (45, 57), (46, 57))]
    mouth_c = (43.0 + ox, 59.0 + oy)
    hand_l = (28.0 + ox, 62.5 + oy)
    hand_r = (65.0 + ox, 55.0 + oy)
    center = (float(np.nonzero(canvas[..., 3] >= 128)[1].mean()), float(dst_y + sh / 2))
    top = (49.0 + ox, 22.0 + oy)

    def mx(p):
        return (CELL_W - 1 - p[0], p[1])

    if flip:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        pivots = {k: mx(v) for k, v in pivots.items()}
        mouth = [(CELL_W - 1 - x, y) for x, y in mouth]
        mouth_c, hand_l, hand_r, center, top = map(mx, (mouth_c, hand_l, hand_r, center, top))
        sole_x = CELL_W - 1 - sole_x
        facing = 1.0
    else:
        facing = -1.0
    rig = {
        "layers": layers, "pivots": pivots, "facing": facing, "rest": canvas,
        "sole": (sole_x, float(sole_y)), "mouth": mouth, "mouth_c": mouth_c,
        "hand_l": hand_l, "hand_r": hand_r, "center": center, "top": top,
        "counts": {k: int(np.count_nonzero(layers[k][..., 3] >= 128)) for k in DRAW},
        "body_n": int(np.count_nonzero(canvas[..., 3] >= 128)),
    }
    if np.any(compose(layers) != canvas):
        raise SystemExit("rest pose does not match the still")
    return rig


# ---------------------------------------------------------------- raster ops

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
    """4x nearest rotate (positive = clockwise on screen). 0 stays exact."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.05 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    # Work on a window around the part for speed.
    r = int(math.hypot(max(abs(xs - pivot[0]).max(), 1), max(abs(ys - pivot[1]).max(), 1))) + 3
    h, w = layer.shape[:2]
    wx0, wx1 = max(0, int(pivot[0]) - r), min(w, int(pivot[0]) + r + 2)
    wy0, wy1 = max(0, int(pivot[1]) - r), min(h, int(pivot[1]) + r + 2)
    sub = layer[wy0:wy1, wx0:wx1]
    scale = 4
    up = np.repeat(np.repeat(sub, scale, axis=0), scale, axis=1)
    px_, py_ = (pivot[0] - wx0) * scale + 1.5, (pivot[1] - wy0) * scale + 1.5
    rad = math.radians(ang)
    c, s = math.cos(rad), math.sin(rad)
    uh, uw = up.shape[:2]
    yy, xx = np.mgrid[0:uh, 0:uw]
    rx, ry = xx - px_, yy - py_
    src_x = np.rint(px_ + c * rx + s * ry).astype(np.int32)
    src_y = np.rint(py_ - s * rx + c * ry).astype(np.int32)
    valid = (src_x >= 0) & (src_x < uw) & (src_y >= 0) & (src_y < uh)
    rot = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    rot[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    out = np.zeros_like(layer)
    out[wy0:wy1, wx0:wx1] = rot[scale // 2::scale, scale // 2::scale]
    return out


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img, target):
    out = img.copy()
    for need in (4, 3, 3):
        a = out[..., 3] >= 128
        have = int(a.sum())
        if have >= target:
            break
        p = np.pad(a, 1)
        n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        ys, xs = np.nonzero((~a) & (n >= need))
        room = target - have
        for y, x in list(zip(ys.tolist(), xs.tolist()))[:room]:
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                ny, nx = y + dy, x + dx
                if out[ny, nx, 3] >= 128:
                    out[y, x] = out[ny, nx]
                    break
    return out


def rotate_part(layer, angle, pivot, target):
    spun = rotate_nearest(layer, angle, pivot)
    if abs(angle) < 0.05:
        return spun
    return despeckle(fill_pinholes(spun, target))


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
        out[...] = 0
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = a.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    mapped = palette[(diff * diff).sum(axis=2).argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv.reshape(-1)]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    out[~a] = 0
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def compose(layers):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in DRAW:
        paste(img, layers[name])
    return img


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero at frame 0 and frame n so the loop closes on the still."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


# ---------------------------------------------------------------- poses

def pose(action, f):
    """Angles in degrees, + = the named motion (see synth for the sign map).

    arm_l  + raises the forward arm, - pushes it down/forward
    arm_r  + swings the back arm forward over the head
    tuft   + whips the pompom backward
    far    + kicks the far foot backward
    """
    n = FRAMES[action]
    p = dict(arm_l=0.0, arm_r=0.0, tuft=0.0, far=0.0, near_lift=0, near_fwd=0, near_rot=0.0,
             arm_l_dx=0, mouth=0, rings=0, wall=-1, burst=0, flash=0, hurt=0.0, knock=0,
             sparks=0, glint=0, spin=0)
    if action == "idle":
        p["arm_l"] = wave(f, n, 9.0)
        p["arm_r"] = wave(f, n, -10.0, phase=0.0)
        p["tuft"] = wave(f, n, 12.0, phase=0.6, cycles=1.0)
        p["far"] = wave(f, n, 6.0, cycles=2.0)
    elif action == "walk":
        step = math.sin(2 * math.pi * 2 * f / n)  # two steps per loop
        p["near_lift"] = int(round(max(0.0, step) * 3))
        p["near_fwd"] = int(round(max(0.0, step) * 2))
        p["near_rot"] = max(0.0, step) * -10.0
        p["far"] = wave(f, n, 16.0, cycles=2.0, phase=math.pi)
        p["arm_l"] = wave(f, n, 12.0, cycles=2.0, phase=math.pi)
        p["arm_r"] = wave(f, n, 12.0, cycles=2.0)
        p["tuft"] = wave(f, n, 14.0, cycles=2.0, phase=-0.8)
    elif action == "attack":
        roar_l = (0.0, 14.0, 26.0, 32.0, 30.0)
        roar_r = (0.0, -8.0, -14.0, -18.0, -16.0)
        roar_t = (0.0, 12.0, 22.0, 26.0, 22.0)
        if f <= 4:
            p["arm_l"] = roar_l[f]
            p["arm_r"] = roar_r[f]
            p["tuft"] = roar_t[f]
            p["mouth"] = (0, 1, 2, 2, 2)[f]
            p["rings"] = f
            p["far"] = (0.0, 6.0, 12.0, 14.0, 12.0)[f]
        elif f <= 12:
            # thrust: forward arm pushes level-down and reaches, back arm swings over.
            k = f - 5
            p["arm_l"] = (-6.0, -12.0, -12.0, -10.0, -10.0, -8.0, -6.0, -4.0)[k]
            p["arm_l_dx"] = (2, 3, 3, 3, 3, 2, 2, 1)[k]
            p["arm_r"] = (14.0, 28.0, 34.0, 34.0, 32.0, 28.0, 22.0, 14.0)[k]
            p["tuft"] = (-10.0, -24.0, -30.0, -28.0, -22.0, -16.0, -10.0, -6.0)[k]
            p["far"] = (18.0, 24.0, 24.0, 22.0, 20.0, 16.0, 12.0, 8.0)[k]
            p["mouth"] = 1 if k <= 1 else 0
            p["wall"] = k if k <= 5 else -1
            p["burst"] = 1 if k in (6, 7) else 0
            p["glint"] = 1 if k <= 3 else 0
        else:
            k = f - 13
            p["arm_l"] = (-2.0, -1.0, 0.0)[k]
            p["arm_r"] = (8.0, 3.0, 0.0)[k]
            p["tuft"] = (-3.0, 2.0, 0.0)[k]
            p["far"] = (4.0, 1.0, 0.0)[k]
    else:  # hurt
        p["hurt"] = HURT_DEG[f]
        p["knock"] = HURT_DX[f]
        p["flash"] = 1 if f in (1, 2) else 0
        p["sparks"] = 1 if 1 <= f <= 5 else 0
        env = HURT_DEG[f] / 34.0
        p["arm_l"] = 22.0 * env
        p["arm_r"] = -14.0 * env
        p["tuft"] = 20.0 * env
        p["far"] = -10.0 * env
        p["mouth"] = 1 if 2 <= f <= 6 else 0
    return p


# ---------------------------------------------------------------- fx

def put(out, x, y, col, over=False):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and (over or out[y, x, 3] < 128):
        out[y, x, :3] = col
        out[y, x, 3] = 255


def arc(out, c, facing, r, span, cols, gap=0):
    seen = set()
    steps = max(8, int(r * math.radians(2 * span) * 2))
    for i in range(steps + 1):
        t = math.radians(-span + 2 * span * i / steps)
        x = int(round(c[0] + facing * r * math.cos(t)))
        y = int(round(c[1] + r * math.sin(t)))
        if (x, y) in seen:
            continue
        seen.add((x, y))
        if gap and (len(seen) // gap) % 2 == 1:
            continue
        put(out, x, y, cols[len(seen) % len(cols)])


def star(out, x, y, size, col, core=WHITE):
    put(out, x, y, core)
    for d in range(1, size + 1):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            put(out, x + dx * d, y + dy * d, col if d > 1 else core)


def wall(out, base, facing, k):
    """Mime barrier: a glassy pane that flies forward from the palm."""
    hx, hy = base
    x = hx + facing * (2 + 3 * k)
    top, bot = int(hy - 16 - k), int(hy + 11 + k // 2)
    width = 5
    for y in range(top, bot + 1):
        lean = (y - top) // 10  # slight perspective tilt
        for dx in range(width):
            xx = x + facing * (dx + lean)
            if dx in (0, width - 1):
                col = SKY
            elif (y - top + dx) % 7 == 0:
                col = WHITE
            else:
                col = PEARL if (y + dx) % 2 else LILAC
            put(out, xx, y, col)
    for dx in range(width):
        put(out, x + facing * dx, top - 1, BLUE)
        put(out, x + facing * (dx + (bot - top) // 10), bot + 1, BLUE)
    for i in range(6):
        put(out, x + facing * (1 + i // 3), top + 2 + i, WHITE, over=True)
    for j, yy in enumerate((top + 3, top + 10, hy, bot - 3)):
        for t in range(2, 4 + 2 * k):
            if (t + j) % 3:
                put(out, x - facing * t, yy, (LILAC, SKY)[j % 2])


def burst(out, base, facing, k):
    hx, hy = base
    cx, cy = hx + facing * 22, hy - 3
    r0 = 4 + 5 * (k - 6)
    for i in range(12):
        t = 2 * math.pi * i / 12 + 0.2
        for d in range(r0, r0 + 5 + (i % 2) * 3):
            put(out, cx + d * math.cos(t), cy + d * math.sin(t), (WHITE, PINK, SKY)[(d - r0) % 3])
    star(out, cx, cy, 3 if k == 6 else 4, PINK)
    # pane shards
    for i, (dx, dy) in enumerate(((-6, -10), (7, -8), (-8, 6), (6, 9), (11, -1), (-11, 0))):
        s = 1.4 + 1.2 * (k - 6)
        put(out, cx + dx * s * 0.7, cy + dy * s * 0.7, SKY)
        put(out, cx + dx * s * 0.7 + 1, cy + dy * s * 0.7, PEARL)


# ---------------------------------------------------------------- synth

def open_mouth(layer, rig, level):
    """Wedge on the existing smile. Pixels hang off the head, never move it."""
    if level <= 0:
        return layer
    out = layer.copy()
    pts = rig["mouth"]
    row = [p for p in pts if p[1] == pts[0][1]]  # the 4-pixel line
    for x, y in row:
        out[y, x, :3] = INK
        out[y, x, 3] = 255
    if level >= 2:
        xs = sorted(x for x, _ in row)
        y = row[0][1]
        for i, x in enumerate(xs):
            out[y + 1, x, :3] = MAROON if i in (0, len(xs) - 1) else RED
            out[y + 1, x, 3] = 255
        for x in xs[1:-1]:
            out[y + 2, x, :3] = INK
            out[y + 2, x, 3] = 255
    return out


def pose_parts(rig, p):
    L, P, fc = rig["layers"], rig["pivots"], rig["facing"]
    cnt = rig["counts"]
    torso = open_mouth(L["torso"], rig, p["mouth"])
    moved = {
        "cushion": L["cushion"],
        "torso": torso,
        # unflipped (facing -1): + raise of the forward arm is clockwise.
        "arm_l": shift(rotate_part(L["arm_l"], -fc * p["arm_l"], P["arm_l"], cnt["arm_l"]),
                       fc * p["arm_l_dx"], 0),
        "arm_r": rotate_part(L["arm_r"], fc * p["arm_r"], P["arm_r"], cnt["arm_r"]),
        "tuft": rotate_part(L["tuft"], -fc * p["tuft"], P["tuft"], cnt["tuft"]),
        "foot_far": rotate_part(L["foot_far"], -fc * p["far"], P["foot_far"], cnt["foot_far"]),
        "foot_near": shift(rotate_part(L["foot_near"], -fc * p["near_rot"], P["foot_near"], cnt["foot_near"]),
                           fc * p["near_fwd"], -p["near_lift"]),
    }
    return compose(moved)


def synth(rig, action, f, palette, fx=True):
    fc = rig["facing"]
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)
    img = pose_parts(rig, p)
    if p["hurt"] > 0.05:
        # lean back about the sole, as one piece.
        img = rotate_nearest(img, -fc * p["hurt"], rig["sole"])
        img = shift(img, -fc * p["knock"], 0)
        img = despeckle(fill_pinholes(img, rig["body_n"]))
    img = snap(img, palette)
    if p["flash"]:
        a = img[..., 3] >= 128
        ink = np.all(img[..., :3] == np.array(INK, np.uint8), axis=-1)
        img[a & ~ink, :3] = WHITE
    if not fx:
        return img
    hx, hy = rig["hand_l"]
    hand = (hx + fc * p["arm_l_dx"], hy)
    if p["rings"]:
        mc = (rig["mouth_c"][0] + fc * 3, rig["mouth_c"][1])
        radii = {1: (5,), 2: (5, 9), 3: (6, 10, 15), 4: (8, 13, 18)}[p["rings"]]
        for i, r in enumerate(radii):
            cols = (WHITE, PEARL) if i == 0 else ((PINK, WHITE) if i == 1 else (SKY, PEARL))
            arc(img, mc, fc, r, 50 - 4 * i, cols, gap=3 if i == 2 else 0)
        tx, ty = rig["top"]
        if p["rings"] >= 2:
            star(img, tx - fc * 7, ty - 3, 2, PINK)
    if p["glint"]:
        star(img, hand[0] + fc * 1, hand[1] - 3, 2, SKY)
    if p["wall"] >= 0:
        wall(img, hand, fc, p["wall"])
    if p["burst"]:
        burst(img, hand, fc, 6 + (f - 11))
    if p["sparks"]:
        sx, sy = rig["center"]
        k = f
        for i, (dx, dy) in enumerate(((10, -14), (14, -6), (8, -22), (16, -18))):
            if (i + k) % 2 == 0:
                star(img, sx + fc * dx - fc * p["knock"] * 2, sy + dy, 1 + (i % 2), (PINK, SKY)[i % 2])
    return snap(img, palette)


# ---------------------------------------------------------------- output

def extract_palette(img):
    return np.unique(img[img[..., 3] >= 128][:, :3], axis=0)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes()
    src += (SRC_DIR / "front.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def assemble(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png").convert("RGBA"))
                sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = tile
        sheets[act] = sh
    return sheets


def contact_sheet(sheets, rows, scale=2):
    """All frames of the given direction rows, cropped, one band per action."""
    pad = 34
    probe = sheets["idle"][:CELL_H, :CELL_W]
    ys, xs = np.nonzero(probe[..., 3] >= 128)
    x0, x1 = max(0, int(xs.min()) - pad - 8), min(CELL_W, int(xs.max()) + 1 + pad + 8)
    y0, y1 = max(0, int(ys.min()) - 22), min(CELL_H, int(ys.max()) + 6)
    cw, ch = (x1 - x0) * scale, (y1 - y0) * scale
    lines = []
    for di in rows:
        for act in ACTIONS:
            lines.append((act, di))
    bg = np.full((len(lines) * ch, 16 * cw, 4), (24, 28, 40, 255), np.uint8)
    for r, (act, di) in enumerate(lines):
        band = sheets[act][di * CELL_H:(di + 1) * CELL_H]
        for f in range(FRAMES[act]):
            tile = band[y0:y1, f * CELL_W + x0:f * CELL_W + x1]
            up = np.repeat(np.repeat(tile, scale, 0), scale, 1)
            y, x = r * ch, f * cw
            if (f + r) % 2:
                bg[y:y + ch, x:x + cw, :3] = (30, 35, 50)
            m = up[..., 3] >= 128
            bg[y:y + ch, x:x + cw][m] = up[m]
    return Image.fromarray(bg, "RGBA")


def dirs_sheet(sheets, scale=2):
    """Idle frame 0 of the 8 directions side by side (direction check)."""
    tiles = [sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W] for di in range(8)]
    row = np.concatenate(tiles, axis=1)
    big = np.repeat(np.repeat(row, scale, 0), scale, 1)
    bg = np.full(big.shape, (24, 28, 40, 255), np.uint8)
    m = big[..., 3] >= 128
    bg[m] = big[m]
    return Image.fromarray(bg, "RGBA")


def body_box(tile):
    ys, xs = np.nonzero(tile[..., 3] >= 128)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def largest_fraction(img):
    a = img[..., 3] >= 128
    sizes = comp_sizes(a)
    return sizes[0] / sum(sizes) if sizes else 1.0


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
        "dex": 439, "form": "normal", "variant": "mime-jr", "gen": "gen4",
        "dir": "monsterSprite/gen4/0439_mime-jr/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed")
        token = '"' + KEY + '":'
        pos = text.find(token)
        if pos < 0 or text.find(token, pos + 1) != -1:
            raise SystemExit("dir8 key not unique")
        val_at = pos + len(token)
        _old, end = json.JSONDecoder().raw_decode(text, val_at)
        out = text[:val_at] + blob + text[end:]
        before = json.loads(text[len(prefix):].strip().rstrip(";"))
        after = json.loads(out[len(prefix):].strip().rstrip(";"))
        if after[KEY] != entry or list(after) != list(before):
            raise SystemExit("dir8 edit failed")
        for k in before:
            if k != KEY and before[k] != after[k]:
                raise SystemExit(f"dir8 neighbour changed: {k}")
        tmp = DIR8.with_suffix(f".js.tmp.{os.getpid()}")
        tmp.write_text(out)
        tmp.replace(DIR8)

    with_flock(DIR8_LOCK, edit)
    return entry


def publish_bounds(sheets):
    vals = []
    for di in range(8):
        tile = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        x0, y0, x1, y1 = body_box(tile)
        vals += [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    new_val = json.dumps(vals, separators=(",", ":"))

    def edit():
        text = BOUNDS.read_text()
        s_at = text.find('"s":{')
        token = '"' + KEY + '":'
        pos = text.find(token, s_at)
        if s_at < 0 or pos < 0 or text.find(token, pos + len(token)) != -1:
            raise SystemExit("sprite_bounds s key not unique")
        val_at = pos + len(token)
        _old, end = json.JSONDecoder().raw_decode(text, val_at)
        out = text[:val_at] + new_val + text[end:]
        tmp = BOUNDS.with_suffix(f".js.tmp.{os.getpid()}")
        tmp.write_text(out)
        tmp.replace(BOUNDS)

    with_flock(DIR8_LOCK, edit)
    return vals


def update_queue(status, note=None):
    def edit():
        raw = QUEUE.read_text(encoding="utf-8")
        at = raw.find('"id": "' + KEY + '"')
        if at < 0:
            raise SystemExit("queue has no item")
        obj_start = raw.rfind("{", 0, at)
        _obj, obj_end = json.JSONDecoder().raw_decode(raw, obj_start)
        block = raw[obj_start:obj_end]
        i = block.find('"status": "')
        j = block.find('"', i + len('"status": "'))
        block2 = block[:i] + f'"status": "{status}"' + block[j + 1:]
        if note:
            k = block2.find('"notes": "')
            e = block2.find('"', k + len('"notes": "'))
            while block2[e - 1] == "\\":
                e = block2.find('"', e + 1)
            block2 = block2[:e] + " | " + note.replace('"', "'") + block2[e:]
        json.loads(block2)
        out = raw[:obj_start] + block2 + raw[obj_end:]
        before, after = json.loads(raw), json.loads(out)
        for b, a in zip(before["items"], after["items"]):
            if b.get("id") != KEY and b != a:
                raise SystemExit("queue neighbour changed")
        tmp = QUEUE.with_suffix(f".json.tmp.{os.getpid()}")
        tmp.write_text(out, encoding="utf-8")
        tmp.replace(QUEUE)

    with_flock(QUEUE_LOCK, edit)


def qa_report(sheets, palette, rigs):
    master = np.concatenate([sheets[a] for a in ACTIONS], axis=0)
    opaque = master[..., 3] == 255
    pal = {tuple(int(c) for c in row) for row in palette}
    cols = np.unique(master[opaque][:, :3], axis=0)
    off = [tuple(int(c) for c in row) for row in cols if tuple(int(c) for c in row) not in pal]
    partial = int(((master[..., 3] > 0) & (master[..., 3] < 255)).sum())
    clip = 0
    for act in ACTIONS:
        for di in range(8):
            for f in range(FRAMES[act]):
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = t[..., 3] >= 128
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip += 1
    still_diff = 0
    for di, d in enumerate(DIRS):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        still_diff += int(np.any(t != rigs[d]["rest"]))
    body = {}
    for act in ACTIONS:
        worst = 1.0
        for f in range(FRAMES[act]):
            for d in ("S", "W"):
                worst = min(worst, largest_fraction(synth(rigs[d], act, f, palette, fx=False)))
        body[act] = round(worst, 4)
    return {
        "alphas": sorted(int(v) for v in np.unique(master[..., 3])),
        "partial_alpha": partial, "off_palette": len(off), "clip_frames": clip,
        "still_diff_dirs": still_diff, "body_main_min": body,
        "hurt_deg_max": max(HURT_DEG),
    }


def run_gait(frames_dir, work, report):
    if not GAIT_LOOP.exists():
        return
    qa = work / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    for act, kind in (("idle", "hover"), ("walk", "hover")):
        od = qa / f"gait_{act}_S"
        if od.exists():
            shutil.rmtree(od)
        r = subprocess.run(
            [str(FORGE / ".venv/bin/python"), "-I", str(GAIT_LOOP), "select",
             "--frames-dir", str(frames_dir / act / "S"), "--fps", "12",
             "--output-dir", str(od), "--kind", kind, "--no-aids"],
            capture_output=True, text=True,
        )
        rep = od / "loop-report.json"
        if rep.exists():
            report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
        else:
            report[f"gait_{act}"] = (r.stderr or r.stdout).strip()[-300:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true", help="after visual approval: dir8 + bounds + sync + queue done")
    args = ap.parse_args()

    front_path = SRC_DIR / "front.png"
    if not front_path.exists():
        front_path = ALT_SRC / "front.png"
    front_bytes = front_path.read_bytes()
    front = np.array(Image.open(front_path).convert("RGBA"))
    if front.shape[:2] != (96, 96):
        raise SystemExit(f"expected 96x96, got {front.shape}")
    if sorted(int(v) for v in np.unique(front[..., 3])) != [0, 255]:
        raise SystemExit("still alpha is not 0/255")
    palette = extract_palette(front)
    for col in PALETTE_REQUIRED:
        if not np.any(np.all(palette == np.array(col, np.uint8), axis=1)):
            raise SystemExit(f"missing measured colour {col}")
    if len(palette) != 13:
        raise SystemExit(f"expected 13 colours, got {len(palette)}")
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("mime-jr parts", counts, flush=True)
    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    for d, rig in rigs.items():
        if np.any(synth(rig, "idle", 0, palette) != rig["rest"]):
            raise SystemExit(f"idle 0 drifted on {d}")
        if np.any(synth(rig, "attack", 15, palette) != rig["rest"]):
            raise SystemExit(f"attack end is not the still on {d}")

    if args.samples:
        sheets = {}
        for act in ACTIONS:
            sh = np.zeros((8 * CELL_H, 16 * CELL_W, 4), np.uint8)
            for di, d in enumerate(DIRS):
                if d not in ("S", "W"):
                    continue
                for f in range(FRAMES[act]):
                    sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = synth(rigs[d], act, f, palette)
            sheets[act] = sh
        contact_sheet(sheets, [DIRS.index("S"), DIRS.index("W")]).save(work / "samples.png")
        for act in ACTIONS:
            worst = min(largest_fraction(synth(rigs["W"], act, f, palette, fx=False)) for f in range(FRAMES[act]))
            print(act, "body main min", round(worst, 3), flush=True)
        print("samples ->", work / "samples.png", flush=True)
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
            if key in state["done"] and all((od / f"{f:02d}.png").exists() and (od / f"{f:02d}.png").stat().st_size > 0
                                            for f in range(FRAMES[act])):
                continue
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.exists() and fp.stat().st_size > 0:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            if key not in state["done"]:
                state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    master = np.concatenate([sheets[a] for a in ACTIONS], axis=0)
    Image.fromarray(master, "RGBA").save(OUT_DIR / "sheet.png")
    preview = contact_sheet(sheets, [DIRS.index("S")])
    preview.save(OUT_DIR / "preview.png")
    (SRC_DIR / "actions").mkdir(parents=True, exist_ok=True)
    preview.save(SRC_DIR / "actions" / "preview.png")
    contact_sheet(sheets, [DIRS.index("W"), DIRS.index("E")]).save(OUT_DIR / "preview_motion.png")
    dirs_sheet(sheets).save(OUT_DIR / "preview_dirs.png")

    idle0 = sheets["idle"][:CELL_H, :CELL_W]
    x0, y0, x1, y1 = body_box(idle0)
    sprite_h = y1 - y0 + 1
    report = qa_report(sheets, palette, rigs)
    report["sprite_h"] = sprite_h
    report["foot_y"] = y1
    if not args.skip_qa:
        run_gait(frames_dir, work, report)
    (work / "qa").mkdir(exist_ok=True)
    (work / "qa" / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)

    if front_path.read_bytes() != front_bytes:
        raise SystemExit("front.png changed")
    meta = {
        "generator": "forge_mime_jr_actions", "version": VERSION, "key": KEY, "dex": 439,
        "variant": "mime-jr", "form": "normal", "archetype": "small biped mime (2 arms, 2 feet, pompom)",
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "unflipped": sorted(UNFLIPPED), "still_faces": "left",
        "parts": counts, "palette_colors": int(len(palette)), "qa": report, "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = (report["off_palette"] or report["clip_frames"] or report["partial_alpha"]
              or report["alphas"] != [0, 255] or report["still_diff_dirs"]
              or min(report["body_main_min"].values()) < 0.9 or y1 != FOOT_Y)
    if failed:
        raise SystemExit(f"qa failed: {report}")
    if args.publish:
        if not args.no_sync:
            SYNC_OUT.mkdir(parents=True, exist_ok=True)
            n = 0
            for fp in OUT_DIR.iterdir():
                if fp.is_file():
                    shutil.copy2(fp, SYNC_OUT / fp.name)
                    n += 1
            print("synced", n, "files ->", SYNC_OUT, flush=True)
        entry = publish_dir8(sprite_h)
        vals = publish_bounds(sheets)
        update_queue("done")
        print("dir8", entry, "bounds", vals, flush=True)
    print(f"h {sprite_h} cw {CELL_W} ch {CELL_H} fy {FOOT_Y}", flush=True)


if __name__ == "__main__":
    main()
