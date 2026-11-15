#!/usr/bin/env python3
"""Kricketune (0402) 8-direction sheet from this form's front.png only.

Measured on normal/front.png (96x96, alpha already 0 or 255, 1414 opaque, 13 colours):
  bbox x20-69 y19-76 (50x58). Centroid about (45.9, 47.6). One component.
  Two white eyes at (41, 34) and (49, 34). The left scythe tip is x=20,
  left of both eyes, so the still leads screen-left.
  Two antenna stalks on the crown. Two scythe arms (the only arms). The black
  mustache stays on the face. The mouth is the existing 10-pixel cluster at
  x43-46 y40-42 (cream, orange, dark brown). It is not the belly stripe.
  Two feet split at y>=71 (gap at x41 and x47). No third leg and no extra arm.

  idle   : 1px whole-body breath. Antennae and the two scythes sway. Feet stay on.
  walk   : the two feet step in turn. Scythes counter-swing. Antennae lag.
  attack : 0-4 the existing mouth drops plus rings; 5-10 in-place 360 of the
           whole still (one piece, no squash); 11-12 impact; 13-15 rest.
  hurt   : one-piece recoil, 38 degrees max, leaning back over the feet.

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.
"""

import argparse
import fcntl
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
NORMAL = ROOT / "monsterSprite/gen4/0402_kricketune/normal"
OUT_DIR = NORMAL / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0402_kricketune/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Left-leading still. These four are mirrored so E faces screen-right.
FLIP_DIRS = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "kricketune_v2"
KEY = "0402_kricketune"

BLACK = (16, 16, 16)
RED = (222, 90, 82)
WHITE = (255, 255, 255)
CREAM = (255, 213, 131)
GOLD = (189, 156, 98)
ORANGE = (148, 82, 24)
MOUTH_BROWN = (90, 57, 32)

DRAW = ["torso", "leg_r", "leg_l", "blade_r", "blade_l", "antenna_r", "antenna_l", "mouth"]
SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "all_8dir_actions.png",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)


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
        pad = np.pad(out, 1)
        out = out | pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    return out


def is_col(img, rgb):
    a = img[..., 3] >= 128
    c = img[..., :3]
    return a & (c[..., 0] == rgb[0]) & (c[..., 1] == rgb[1]) & (c[..., 2] == rgb[2])


def segment(img):
    """Two scythes, two antennae, two feet, one mouth. Mustache stays on the torso."""
    h, w = img.shape[:2]
    opaque = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    # The mouth is only this cluster. Belly gold is not a mouth.
    mouth = is_col(img, MOUTH_BROWN)
    ys, xs = np.nonzero(mouth)
    if len(ys) == 0:
        raise SystemExit("kricketune mouth brown is missing")
    near = np.zeros_like(opaque)
    for y, x in zip(ys.tolist(), xs.tolist()):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and ny <= 43 and (
                    is_col(img, ORANGE)[ny, nx] or is_col(img, CREAM)[ny, nx] or mouth[ny, nx]
                ):
                    near[ny, nx] = True
    mouth = near

    label = np.full((h, w), "", dtype=object)
    label[mouth] = "mouth"
    seeds = {
        "torso": [(48, 36), (45, 34), (47, 50), (45, 60), (44, 66), (42, 39), (45, 43), (50, 40)],
        "blade_l": [(22, 20), (23, 30), (28, 40), (29, 47)],
        "blade_r": [(60, 28), (63, 50), (67, 45)],
        "antenna_l": [(40, 19), (37, 25), (35, 28)],
        "antenna_r": [(52, 19), (52, 25), (50, 28)],
        "leg_l": [(36, 72), (35, 73)],
        "leg_r": [(52, 73), (53, 75)],
    }
    q = deque()
    for name, pts in seeds.items():
        planted = 0
        for x, y in pts:
            if not opaque[y, x] or label[y, x] != "":
                continue
            label[y, x] = name
            q.append((y, x))
            planted += 1
        if planted == 0:
            raise SystemExit(f"kricketune seed missed {name}")
    while q:
        y, x = q.popleft()
        lab = label[y, x]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and opaque[ny, nx] and label[ny, nx] == "":
                label[ny, nx] = lab
                q.append((ny, nx))
    if int((opaque & (label == "")).sum()):
        raise SystemExit("kricketune watershed left pixels unlabeled")

    # Feet are only the split below the belly. Gold corners above y71 stay torso.
    label[(label == "leg_l") | (label == "leg_r")] = "torso"
    label[opaque & (yy >= 71) & (xx <= 40)] = "leg_l"
    label[opaque & (yy >= 71) & (xx >= 48)] = "leg_r"
    # The right scythe merges into the belly at y54. Below that is abdomen.
    label[(label == "blade_r") & (yy >= 54)] = "torso"
    # The left scythe's hip seam is abdomen, not a third arm.
    label[(label == "blade_l") & (yy >= 46) & (xx >= 33)] = "torso"

    parts = {name: (label == name) for name in (
        "torso", "blade_l", "blade_r", "antenna_l", "antenna_r", "leg_l", "leg_r", "mouth"
    )}
    cover = np.zeros_like(opaque)
    for name, mask in parts.items():
        if np.any(cover & mask):
            raise SystemExit(f"kricketune masks overlap at {name}")
        cover |= mask
    if int((cover != opaque).sum()):
        raise SystemExit(f"kricketune masks missed {int((cover != opaque).sum())} pixels")
    need = {
        "torso": 600, "blade_l": 40, "blade_r": 40, "antenna_l": 20, "antenna_r": 20,
        "leg_l": 8, "leg_r": 8, "mouth": 8,
    }
    for name, n in need.items():
        if int(parts[name].sum()) < n:
            raise SystemExit(f"part {name} collapsed: {int(parts[name].sum())}")
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
    """4x nearest. Rigid: no scale and no mirror. Exact colours."""
    ang = float(angle_deg) % 360.0
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
    """Drop only pixels whose four neighbours are empty."""
    out = img.copy()
    a = out[..., 3] >= 128
    pad = np.pad(a, 1)
    n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    step = 400
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None].astype(np.int32) - ys[None, :].astype(np.int32)) ** 2
        d += (tx[sl, None].astype(np.int32) - xs[None, :].astype(np.int32)) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def fill_pinholes(img):
    """Plug only fully enclosed single holes. Does not bridge the leg gap."""
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        pad = np.pad(a, 1)
        n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        hole = (~a) & (n == 4)
        if not np.any(hole):
            break
        fill = nearest_color(out, hole, a)
        out[hole] = fill[hole]
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
    """Draw one palette pixel only on empty space, inside the cell margin."""
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return True
    return False


def contact_pivot(part, body):
    ys, xs = np.nonzero(part)
    if len(ys) == 0:
        return 0.0, 0.0
    touch = part & dilate(body, 1)
    if np.any(touch):
        ys, xs = np.nonzero(touch)
    by, bx = np.nonzero(body)
    bcy, bcx = float(by.mean()), float(bx.mean())
    d = (ys.astype(np.float32) - bcy) ** 2 + (xs.astype(np.float32) - bcx) ** 2
    i = int(d.argmin())
    return float(xs[i]), float(ys[i])


def add_socket(host, host_mask, part_mask, canvas, n=1):
    zone = part_mask & dilate(host_mask, n)
    if not np.any(zone):
        return
    fill = nearest_color(canvas, zone, host_mask)
    write = zone & (host[..., 3] < 128)
    host[write] = fill[write]


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
    for name, mask in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = mask[y0:y1, x0:x1]
        placed[name] = pm

    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr

    body = placed["torso"]
    for name in ("blade_l", "blade_r", "antenna_l", "antenna_r", "leg_l", "leg_r", "mouth"):
        add_socket(layers["torso"], body, placed[name], canvas, n=1)

    pivots = {
        name: contact_pivot(placed[name], body)
        for name in ("blade_l", "blade_r", "antenna_l", "antenna_r")
    }
    opy, opx = np.nonzero(canvas[..., 3] >= 128)
    center = (float(opx.mean()), float(opy.mean()))
    sole_y = float(opy.max())
    sole_x = float(np.median(opx[opy >= sole_y - 1]))
    sole = (sole_x, sole_y)
    still = canvas.copy()

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
        sole = (CELL_W - 1 - sole[0], sole[1])
        still = still[:, ::-1].copy()
    facing = -1.0 if not flip else 1.0
    return {
        "layers": layers,
        "pivots": pivots,
        "center": center,
        "sole": sole,
        "facing": facing,
        "flip": flip,
        "still": still,
    }


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, blade_l=0.0, blade_r=0.0, ant_l=0.0, ant_r=0.0,
        lift_l=0, lift_r=0, mouth=0, spin=0.0, hurt=0.0,
        rings=0, ring=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["dy"] = wave(f, n, 1.0)
        # Right scythe detaches if its angle is positive. Keep that swing <= 0.
        p["ant_l"] = wave(f, n, 4.0, phase=0.5)
        p["ant_r"] = wave(f, n, -3.0, phase=0.2)
        p["blade_l"] = wave(f, n, 4.0, phase=0.9)
        p["blade_r"] = -abs(wave(f, n, 4.0, phase=0.9))
    elif action == "walk":
        step = wave(f, n, 1.0, cycles=2.0)
        p["dx"] = step
        p["dy"] = wave(f, n, 1.0, cycles=2.0)
        p["lift_l"] = int(round(max(0.0, step) * 3.0))
        p["lift_r"] = int(round(max(0.0, -step) * 3.0))
        p["blade_l"] = wave(f, n, 10.0, cycles=2.0)
        p["blade_r"] = -abs(wave(f, n, 10.0, cycles=2.0))
        p["ant_l"] = wave(f, n, 4.0, cycles=2.0, phase=0.5)
        p["ant_r"] = wave(f, n, -3.0, cycles=2.0, phase=0.5)
    elif action == "attack":
        if f <= 4:
            p["mouth"] = 2
            p["blade_l"] = (8, 14, 18, 14, 8)[f]
            p["blade_r"] = (-6, -10, -14, -10, -6)[f]
            p["ant_l"] = (4, 8, 10, 8, 4)[f]
            p["ant_r"] = (-4, -7, -9, -7, -4)[f]
            p["rings"] = 1
            p["ring"] = f
        elif f <= 10:
            p["spin"] = 60.0 * (f - 4)
        elif f in (11, 12):
            p["burst"] = 1
        elif f == 13:
            p["blade_l"] = 5
            p["blade_r"] = -4
        elif f == 14:
            p["blade_l"] = 2
            p["blade_r"] = -2
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 38.0 * env
        p["dx"] = 4.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 5 else 0
        if p["hurt"] > 38.0 + 1e-6:
            raise SystemExit("hurt angle exceeded 38")
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def part_moved(layer, pivot, ang, dx, dy):
    lyr = rotate_nearest(layer, ang, pivot) if abs(ang) >= 0.45 else layer
    return shift(lyr, dx, dy)


def mouth_anchor(layer, facing):
    a = layer[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return None
    x = int(xs.min() if facing < 0 else xs.max())
    y = int(np.median(ys[xs == x])) if np.any(xs == x) else int(np.median(ys))
    return x, y


def open_mouth(img, mouth_src, bdx, bdy, drop):
    """Drop the existing mouth. The gap is palette black, not a new jaw."""
    if drop <= 0:
        paste(img, shift(mouth_src, bdx, bdy))
        return img
    home = shift(mouth_src, bdx, bdy)
    dropped = shift(mouth_src, bdx, bdy + drop)
    hole = (home[..., 3] >= 128) & (dropped[..., 3] < 128) & (img[..., 3] < 128)
    img[hole, :3] = BLACK
    img[hole, 3] = 255
    paste(img, dropped)
    return img


def roar_rings(out, mouth, facing):
    """Fixed 24 dots in the empty space ahead of the scythe. Same count every roar frame."""
    mx, my = mouth
    cols = (WHITE, RED, CREAM)
    drawn = 0
    for i, rad in enumerate((28, 34, 40)):
        col = cols[i]
        for deg in range(-50, 51, 14):
            ang = math.radians(deg)
            x = mx + facing * rad * math.cos(ang)
            y = my + rad * math.sin(ang) * 0.42
            if put_empty(out, x, y, col):
                drawn += 1
    return out, drawn


def impact_burst(out, mouth, facing):
    mx, my = mouth
    cols = (WHITE, GOLD, RED)
    for i, rad in enumerate((22, 30, 38)):
        col = cols[i]
        for deg in range(-60, 61, 17):
            ang = math.radians(deg)
            put_empty(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.4, col)
    return out


def sparks(out, frame_idx, facing):
    ys, xs = np.nonzero(out[..., 3] >= 128)
    if len(ys) == 0:
        return out
    x = int(xs.min() if facing < 0 else xs.max())
    y = int(np.median(ys[xs == x])) if np.any(xs == x) else int(ys.min())
    d = frame_idx
    pts = (
        (x + facing * (4 + d), y - 6, WHITE),
        (x + facing * (6 + d), y - 2, RED),
        (x + facing * (3 + d), y + 3, CREAM),
    )
    for sx, sy, col in pts:
        put_empty(out, sx, sy, col)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def synth(rig, action, f, palette):
    layers = rig["layers"]
    pivots = rig["pivots"]
    facing = rig["facing"]
    flip = rig["flip"]
    ang_sign = -1.0 if flip else 1.0
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    if p["hurt"] > 38.0 + 1e-6:
        raise SystemExit("hurt angle exceeded 38")

    spinning = p["spin"] > 0.5 and (p["spin"] % 360.0) > 0.5
    recoiling = p["hurt"] > 0.45
    if spinning or recoiling:
        if spinning:
            img = rotate_nearest(rig["still"], p["spin"] * ang_sign, rig["center"])
        else:
            # Left-leading still: positive angle kicks the crown up and back.
            img = rotate_nearest(rig["still"], p["hurt"] * ang_sign, rig["sole"])
            img = shift(img, int(round(-facing * p["dx"])), 0)
        img = fill_pinholes(img)
        img = despeckle(img)
    else:
        bdx = int(round(p["dx"] * facing))
        bdy = int(round(p["dy"]))
        img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        paste(img, shift(layers["torso"], bdx, bdy))
        for name, lift in (("leg_r", p["lift_r"]), ("leg_l", p["lift_l"])):
            step = facing if lift else 0
            paste(img, shift(layers[name], bdx + step, bdy - lift))
        for name, key in (("blade_r", "blade_r"), ("blade_l", "blade_l"), ("antenna_r", "ant_r"), ("antenna_l", "ant_l")):
            paste(img, part_moved(layers[name], pivots[name], p[key] * ang_sign, bdx, bdy))
        img = open_mouth(img, layers["mouth"], bdx, bdy, int(p["mouth"]))

    img = snap(img, palette)
    if p["flash"]:
        img = flash_white(img)
        img = snap(img, palette)
    anchor = None
    if p["rings"] or p["burst"]:
        dropped = shift(layers["mouth"], 0, int(p["mouth"]))
        anchor = mouth_anchor(dropped, facing)
    if p["rings"] and anchor:
        img, _drawn = roar_rings(img, anchor, facing)
    if p["burst"] and anchor:
        img = impact_burst(img, anchor, facing)
    if p["sparks"]:
        img = sparks(img, f, facing)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (NORMAL / "front.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def stack_sheets(frames_dir):
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
    """South row, every frame, scaled so the scythes and mouth are readable."""
    scale = 2
    pad = 6
    crops = []
    for act in ACTIONS:
        band = sheets[act][:CELL_H]
        row = []
        for f in range(FRAMES[act]):
            tile = band[:, f * CELL_W:(f + 1) * CELL_W]
            a = tile[..., 3] >= 128
            if not np.any(a):
                crop = tile[:8, :8]
            else:
                ys, xs = np.nonzero(a)
                y0 = max(0, int(ys.min()) - pad)
                y1 = min(CELL_H, int(ys.max()) + 1 + pad)
                x0 = max(0, int(xs.min()) - pad)
                x1 = min(CELL_W, int(xs.max()) + 1 + pad)
                crop = tile[y0:y1, x0:x1]
            row.append(np.repeat(np.repeat(crop, scale, 0), scale, 1))
        crops.append(row)
    cell_h = max(c.shape[0] for row in crops for c in row)
    cell_w = max(c.shape[1] for row in crops for c in row)
    cols = 8
    rows = sum(math.ceil(len(row) / cols) for row in crops)
    bg = np.full((rows * cell_h, cols * cell_w, 4), (15, 20, 31, 255), np.uint8)
    r = 0
    for row in crops:
        for i, c in enumerate(row):
            rr, cc = divmod(i, cols)
            y = (r + rr) * cell_h + (cell_h - c.shape[0]) // 2
            x = cc * cell_w + (cell_w - c.shape[1]) // 2
            m = c[..., 3] >= 128
            bg[y:y + c.shape[0], x:x + c.shape[1]][m] = c[m]
        r += math.ceil(len(row) / cols)
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 4
    crops = []
    for act in ACTIONS:
        row = []
        band = sheets[act][:CELL_H]
        for f in range(FRAMES[act]):
            tile = band[:, f * CELL_W:(f + 1) * CELL_W]
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


def south_attack_ratios(sheets):
    band = sheets["attack"][:CELL_H]
    counts = []
    for f in range(FRAMES["attack"]):
        tile = band[:, f * CELL_W:(f + 1) * CELL_W]
        counts.append(int((tile[..., 3] >= 128).sum()))
    base = counts[0] if counts[0] else 1
    return counts, [c / base for c in counts]


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    return n


def publish_dir8(sprite_h):
    """Replace only 0402_kricketune. The rest of dir8.js stays byte-for-byte."""
    entry = {
        "dex": 402,
        "form": "normal",
        "variant": "kricketune",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0402_kricketune/normal",
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
    key = f'"{KEY}":'
    DIR8_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with DIR8_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        if key not in text:
            raise SystemExit("dir8 is missing 0402_kricketune")
        start = text.index(key) + len(key)
        i = start
        while text[i] != "{":
            i += 1
        depth = 0
        end = None
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end is None:
            raise SystemExit("dir8 entry braces did not close")
        text = text[:i] + blob + text[end:]
        json.loads(text[len(prefix):].rstrip().rstrip(";"))
        if text.count(f'"v":"{VERSION}"') != 1:
            raise SystemExit("dir8 version stamp collided")
        if text.count(key) != 1:
            raise SystemExit("dir8 key was duplicated")
        DIR8.write_text(text)
        fcntl.flock(lock, fcntl.LOCK_UN)


def update_queue(status):
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(QUEUE.read_text())
        found = False
        for item in data["items"]:
            if item.get("id") == KEY:
                item["status"] = status
                item.pop("claim", None)
                found = True
                break
        if not found:
            raise SystemExit("queue item 0402_kricketune missing")
        QUEUE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)


def edge_touches(img):
    m = img[..., 3] >= 128
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front_path = NORMAL / "front.png"
    front_bytes = front_path.read_bytes()
    front = np.array(Image.open(front_path).convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), flush=True)

    eyes = is_col(front, WHITE)
    ey, ex = np.nonzero(eyes)
    blade_ys, blade_xs = np.nonzero(masks["blade_l"])
    tip_x = int(blade_xs.min())
    eye_xs = sorted(int(v) for v in ex.tolist())
    print("facing tip_x", tip_x, "eyes", list(zip(ex.tolist(), ey.tolist())), flush=True)
    if not eye_xs or not all(tip_x < x for x in eye_xs):
        raise SystemExit("expected the left scythe to lead both eyes")

    rigs = {d: build_rig(front, masks, flip=d in FLIP_DIRS) for d in DIRS}
    print(
        "pivots L", [round(v, 1) for v in rigs["W"]["pivots"]["blade_l"]],
        "R", [round(v, 1) for v in rigs["W"]["pivots"]["blade_r"]],
        "center", [round(v, 1) for v in rigs["W"]["center"]],
        flush=True,
    )

    rest = synth(rigs["W"], "idle", 0, palette)
    if not np.array_equal(rest, rigs["W"]["still"]):
        differ = int(np.any(rest != rigs["W"]["still"], axis=2).sum())
        raise SystemExit(f"idle frame 0 does not match the still ({differ} pixels)")
    south = synth(rigs["S"], "idle", 0, palette)
    mirrored = rigs["W"]["still"][:, ::-1].copy()
    if not np.array_equal(south, mirrored):
        differ = int(np.any(south != mirrored, axis=2).sum())
        raise SystemExit(f"south idle 0 is not the mirror ({differ} pixels)")
    print("rest-pose pixels changed 0", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (80, 140, 200), "blade_l": (220, 60, 60), "blade_r": (255, 140, 40),
        "antenna_l": (40, 200, 80), "antenna_r": (170, 80, 220),
        "leg_l": (255, 220, 40), "leg_r": (40, 220, 220), "mouth": (255, 255, 255),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 6, 0), 6, 1), "RGBA").save(work / "parts.png")

    atk = [synth(rigs["S"], "attack", f, palette) for f in range(16)]
    counts_a = [int((fr[..., 3] >= 128).sum()) for fr in atk]
    base_n = counts_a[0]
    ratios = [c / base_n for c in counts_a]
    print("south attack opaque", counts_a, flush=True)
    print("south attack ratios", " ".join(f"{r:.4f}" for r in ratios), flush=True)
    print("min", f"{min(ratios):.4f}", "max", f"{max(ratios):.4f}", flush=True)
    bad = [i for i, r in enumerate(ratios) if r < 0.90 or r > 1.08]
    if bad:
        raise SystemExit(f"attack opaque ratio out of range at frames {bad}")
    clipped = [f for f, fr in enumerate(atk) if edge_touches(fr)]
    if clipped:
        raise SystemExit(f"south attack touches the cell edge at {clipped}")

    broken = []
    for f, fr in enumerate(atk):
        if f < 5 or f > 10:
            continue
        sizes = components(fr[..., 3] >= 128)
        body = [s for s in sizes if s > 12]
        if len(body) != 1:
            broken.append(("attack", f, sizes[:6]))
    for f in (0, 1, 3, 6, 9, 11):
        fr = synth(rigs["S"], "hurt", f, palette)
        if edge_touches(fr):
            broken.append(("hurt-edge", f, []))
        sizes = components(fr[..., 3] >= 128)
        body = [s for s in sizes if s > 12]
        if len(body) != 1:
            broken.append(("hurt", f, sizes[:6]))
    print("piece check", broken if broken else "ok", flush=True)
    if broken:
        raise SystemExit("spin or recoil broke the body")

    if front_path.read_bytes() != front_bytes:
        raise SystemExit("front.png was modified")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in range(12)]
        tiles = []
        for act, d, f in want:
            if act == "attack" and d == "S":
                tiles.append(atk[f])
            else:
                tiles.append(synth(rigs[d], act, f, palette))
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        zooms = []
        for tile in tiles:
            a = tile[..., 3] >= 128
            ys, xs = np.nonzero(a)
            y0, y1 = int(ys.min()) - 8, int(ys.max()) + 9
            x0, x1 = int(xs.min()) - 8, int(xs.max()) + 9
            y0, x0 = max(0, y0), max(0, x0)
            y1, x1 = min(CELL_H, y1), min(CELL_W, x1)
            crop = tile[y0:y1, x0:x1]
            zooms.append(np.repeat(np.repeat(crop, 2, 0), 2, 1))
        zh = max(z.shape[0] for z in zooms)
        zw = max(z.shape[1] for z in zooms)
        zrows = math.ceil(len(zooms) / cols)
        zsh = np.full((zrows * zh, cols * zw, 4), (15, 20, 31, 255), np.uint8)
        for i, z in enumerate(zooms):
            r, c = divmod(i, cols)
            y = r * zh + (zh - z.shape[0]) // 2
            x = c * zw + (zw - z.shape[1]) // 2
            m = z[..., 3] >= 128
            zsh[y:y + z.shape[0], x:x + z.shape[1]][m] = z[m]
        Image.fromarray(zsh, "RGBA").save(work / "samples_zoom.png")
        print("samples", work / "samples_zoom.png", flush=True)
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
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if (
                    not args.force
                    and state.get("hash") == digest
                    and fp.exists()
                    and fp.stat().st_size > 0
                ):
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            if key not in state["done"]:
                state["done"].append(key)
            state["hash"] = digest
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = stack_sheets(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    (NORMAL / "actions").mkdir(exist_ok=True)
    preview.save(NORMAL / "actions" / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    counts_a, ratios = south_attack_ratios(sheets)
    print("south attack opaque", counts_a, flush=True)
    print("south attack ratios", " ".join(f"{r:.4f}" for r in ratios), flush=True)
    print("min", f"{min(ratios):.4f}", "max", f"{max(ratios):.4f}", flush=True)
    bad = [i for i, r in enumerate(ratios) if r < 0.90 or r > 1.08]
    if bad:
        raise SystemExit(f"attack opaque ratio out of range at frames {bad}")

    neutral = sheets["idle"][6 * CELL_H:7 * CELL_H, :CELL_W]
    if not np.array_equal(neutral, rigs["W"]["still"]):
        raise SystemExit("sheet W idle 0 drifted from the still")
    nys = np.nonzero(sheets["idle"][:CELL_H, :CELL_W, 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    bad_alpha = [v for v in alphas if v not in (0, 255)]
    clip = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
    spin_parts = []
    for f in range(5, 11):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        spin_parts.append(components(tile[..., 3] >= 128)[:3])
    report = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_h,
        "off_palette": len(off),
        "alphas": alphas,
        "bad_alpha": bad_alpha,
        "edge_touching": clip,
        "attack_counts": counts_a,
        "attack_ratios": [round(r, 4) for r in ratios],
        "attack_ratio_min": round(min(ratios), 4),
        "attack_ratio_max": round(max(ratios), 4),
        "spin_components": spin_parts,
        "tip_x": tip_x,
        "eye_xs": eye_xs,
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "off_palette", "alphas", "edge_touching", "sprite_h", "attack_ratio_min", "attack_ratio_max", "spin_components"
    )}, ensure_ascii=False), flush=True)

    meta = {
        "generator": "forge_kricketune_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 402,
        "form": "normal",
        "variant": "kricketune",
        "archetype": "biped_cricket_scythe",
        "facing": "left",
        "unflipped": ["N", "NW", "W", "SW"],
        "flipped": ["S", "SE", "E", "NE"],
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "palette_colors": int(len(palette)),
        "off_palette_colors": len(off),
        "arms": 2,
        "antennae": 2,
        "feet": 2,
        "mustache": "on torso",
        "source": "front.png 96x96",
        "gifs_untouched": True,
        "attack_ratio_min": round(min(ratios), 4),
        "attack_ratio_max": round(max(ratios), 4),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = (
        len(off) != 0
        or bad_alpha
        or clip
        or min(ratios) < 0.90
        or max(ratios) > 1.08
        or any(len([s for s in sizes if s > 12]) != 1 for sizes in spin_parts)
    )
    if failed:
        print("QA failed", flush=True)
        if args.publish:
            raise SystemExit("refusing to publish a failed sheet")
        return

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        publish_dir8(sprite_h)
        update_queue("done")
        print("published", VERSION, flush=True)
    if front_path.read_bytes() != front_bytes:
        raise SystemExit("front.png was modified")


if __name__ == "__main__":
    main()
