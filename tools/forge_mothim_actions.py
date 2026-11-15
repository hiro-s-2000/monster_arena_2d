#!/usr/bin/env python3
"""Mothim (0414_mothim) 8-direction action sheet from the normal front.png.

Measured on this still only (96x96, opaque 1697, bbox x10-85 y26-69, 13 colours).
Light-gray eyes (y<48) sit at x=40.6. The opaque centroid is x=46.9, so the
face is left of the body. The still faces left. S SE E NE are mirrored so E
points screen-right. W matches the still.

Two antennae (orange stalks above the eyes, cut at y<=36), one left wing
(x<=33 and y<=53, feet start lower), one right wing (outer x>=63, upper runs
only so the right foot stays on the body). No extra wings and no mouth.
Roar flares those antennae and the two wings, plus rings. The spin turns the
whole moth as one piece. Feet stay on the body and idle frame 0 is the still.

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.
"""

import argparse
import fcntl
import hashlib
import json
import math
import re
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
SPECIES = ROOT / "monsterSprite/gen4/0414_mothim/normal"
OUT_DIR = SPECIES / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0414_mothim/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"
KEY = "0414_mothim"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left. These four are mirrored so E points screen-right.
MIRROR = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 200, 192, 168
VERSION = "mothim_v2"
KEEP = 6

BLACK = (16, 16, 16)
ORANGE = (230, 106, 41)
YELLOW = (246, 213, 123)
PALE = (205, 205, 205)
PARTS = ("body", "left_wing", "right_wing", "ant_l", "ant_r")


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def segment(img):
    """Two wings, two antennae, and the body (head, abdomen, both feet).

    Cuts are the gaps measured on this still. The left wing stops at x=33, y=53
    so the left foot (y>=55) stays. The right wing takes the outer membrane and
    the upper runs only; columns x52-62 below y54 are the abdomen and the right
    foot. Antennae are the stalks above the eye bar and stop at y=36.
    """
    h, w = img.shape[:2]
    opaque = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    left_wing = opaque & (xx <= 33) & (yy <= 53)
    right_wing = opaque & (
        (xx >= 63)
        | ((xx >= 57) & (yy <= 54))
        | ((xx >= 52) & (yy <= 53))
    )
    ant_l = opaque & (xx >= 35) & (xx <= 41) & (yy >= 26) & (yy <= 36)
    ant_r = opaque & (xx >= 43) & (xx <= 51) & (yy >= 26) & (yy <= 36)
    body = opaque & ~left_wing & ~right_wing & ~ant_l & ~ant_r
    masks = {
        "body": body,
        "left_wing": left_wing,
        "right_wing": right_wing,
        "ant_l": ant_l,
        "ant_r": ant_r,
    }
    own = body | left_wing | right_wing | ant_l | ant_r
    overlap = (
        int((left_wing & right_wing).sum())
        + int((left_wing & ant_l).sum())
        + int((left_wing & ant_r).sum())
        + int((right_wing & ant_l).sum())
        + int((right_wing & ant_r).sum())
        + int((ant_l & ant_r).sum())
        + int((body & (left_wing | right_wing | ant_l | ant_r)).sum())
    )
    if int(own.sum()) != int(opaque.sum()) or overlap:
        raise SystemExit(f"masks do not partition the still overlap={overlap}")
    if min(int(masks[k].sum()) for k in PARTS) <= 0:
        raise SystemExit("a mothim part mask is empty")
    return masks


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
    """4x nearest, then a 6-of-16 mode. Multiples of 360 stay exact.

    Positive degrees are clockwise. No |cos| squash. The kept colour is the
    mode of the block, so the palette does not grow.
    """
    if abs(angle_deg) % 360 < 0.05 or not np.any(layer[..., 3] >= 128):
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
    outu = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    outu[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    H, W = layer.shape[:2]
    blk = outu.reshape(H, 4, W, 4, 4)
    opa = blk[..., 3] >= 128
    cnt = opa.sum(axis=(1, 3))
    out = np.zeros_like(layer)
    ys, xs = np.nonzero(cnt >= KEEP)
    for y, x in zip(ys.tolist(), xs.tolist()):
        cols = blk[y, :, x, :, :3][opa[y, :, x, :]]
        if len(cols) == 0:
            continue
        uniq, counts = np.unique(cols, axis=0, return_counts=True)
        out[y, x, :3] = uniq[int(counts.argmax())]
        out[y, x, 3] = 255
    return out


def despeckle(img):
    """Drop pixels whose four neighbours are empty. After a rotation only."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


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
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    ys, xs = np.nonzero(mask)
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
        out.append(pts)
    return out


def edge_pivot(part, side):
    ys, xs = np.nonzero(part)
    if side == "inner_right":
        edge = xs.max()
        sel = xs >= edge - 1
    else:
        edge = xs.min()
        sel = xs <= edge + 1
    return float(xs[sel].mean()), float(ys[sel].mean())


def base_pivot(part):
    ys, xs = np.nonzero(part)
    bottom = ys.max()
    sel = ys >= bottom - 1
    return float(xs[sel].mean()), float(bottom)


def build_rig(img, masks):
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
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm

    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr

    # A one-pixel joint cushion, hidden while the part rests on it.
    for part_name in ("left_wing", "right_wing", "ant_l", "ant_r"):
        zone = placed[part_name] & dilate(placed["body"], 1)
        if not np.any(zone):
            continue
        fill = nearest_color(canvas, zone, placed["body"])
        write = zone & (layers["body"][..., 3] < 128)
        layers["body"][write] = fill[write]

    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    bottom = int(oys.max())
    foot_sel = oys >= bottom - 1
    foot_pivot = (float(oxs[foot_sel].mean()), float(bottom))
    eye = placed["body"] & np.all(canvas[..., :3] == np.array(PALE, np.uint8), axis=-1)
    ey, ex = np.nonzero(eye & (np.arange(CELL_H)[:, None] < foot_pivot[1] - 18))
    if len(ex) == 0:
        ey, ex = np.nonzero(eye)
    face = (float(ex.mean()), float(ey.mean())) if len(ex) else center
    pivots = {
        "left_wing": edge_pivot(placed["left_wing"], "inner_right"),
        "right_wing": edge_pivot(placed["right_wing"], "inner_left"),
        "ant_l": base_pivot(placed["ant_l"]),
        "ant_r": base_pivot(placed["ant_r"]),
    }
    return layers, pivots, center, foot_pivot, face, canvas


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def iwave(f, n, amp, phase=0.0, cycles=1.0):
    return int(round(wave(f, n, amp, phase, cycles)))


def pose(action, f):
    n = FRAMES[action]
    p = dict(dx=0, dy=0, wing=0.0, ant=0.0, tilt=0.0, spin=0.0, hurt=0.0,
             rings=0, ring=0, burst=0, sparks=0)
    if action == "idle":
        p["dy"] = iwave(f, n, 1.0)
        p["wing"] = wave(f, n, 10.0)
        p["ant"] = wave(f, n, 7.0, phase=-0.5)
    elif action == "walk":
        p["dx"] = iwave(f, n, 1.0)
        p["dy"] = iwave(f, n, 1.0, cycles=2.0)
        p["tilt"] = wave(f, n, 4.0)
        p["wing"] = wave(f, n, 16.0, cycles=2.0)
        p["ant"] = wave(f, n, 8.0, phase=-0.4, cycles=2.0)
    elif action == "attack":
        # 0 is the still. 1-4 flare antennae and wings. 5-10 one full turn.
        wing = [0, 6, 10, 12, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ant = [0, 8, 14, 16, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        p["wing"] = float(wing[f])
        p["ant"] = float(ant[f])
        if 1 <= f <= 4:
            p["rings"] = 2 if f in (2, 3) else 1
            p["ring"] = f - 1
        if 5 <= f <= 10:
            ang = 60.0 * (f - 4)
            p["spin"] = 0.0 if ang % 360 == 0 else ang
        if f in (11, 12):
            p["burst"] = 1
    else:
        env = [0.0, 0.35, 0.65, 0.88, 1.0, 0.92, 0.72, 0.48, 0.28, 0.12, 0.04, 0.0][f]
        # Negative raises the left-hand face. Cap is 38 degrees.
        p["hurt"] = -38.0 * env
        p["dx"] = int(round(3 * env))
        p["sparks"] = 1 if 2 <= f <= 5 else 0
    if abs(p["hurt"]) > 38.01:
        raise SystemExit("hurt angle exceeds 38")
    return p


def turn(layer, angle, pivot):
    if abs(angle) < 0.05 or abs(angle) % 360 < 0.05:
        return layer
    return despeckle(rotate_nearest(layer, angle, pivot))


def leading_edge(img, face):
    """Still faces left, so the roar sits just past the left-hand wing tip."""
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    return float(xs.min()), float(face[1])


def roar_rings(out, face, age, strength):
    """Rings in front of the wings. There is no mouth on this still."""
    ox, oy = leading_edge(out, face)
    cols = (PALE, YELLOW, ORANGE)
    for r_i in range(strength):
        rad = 4 + age * 2 + r_i * 3
        col = cols[r_i % 3]
        for deg in range(-50, 51, 25):
            ang = math.radians(deg)
            put(out, ox - 3 - rad * math.cos(ang), oy + rad * math.sin(ang) * 0.75, col)
    return out


def impact_burst(out, face):
    ox, oy = leading_edge(out, face)
    cols = (YELLOW, ORANGE, PALE)
    for k in range(7):
        ang = math.radians(-60 + k * 20)
        put(out, ox - 4 - 5 * math.cos(ang), oy + 4 * math.sin(ang), cols[k % 3])
        put(out, ox - 4 - 8 * math.cos(ang), oy + 6 * math.sin(ang), cols[(k + 1) % 3])
    return out


def sparks(out, frame_idx):
    a = out[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return out
    x0, y0 = int(xs.min()), int(ys.min())
    d = frame_idx
    for sx, sy, col in (
        (x0 - 2 - d, y0 + 8, PALE),
        (x0 - 1, y0 - 2 - d, YELLOW),
        ((x0 + int(xs.max())) // 2, y0 - 3 - d, ORANGE),
    ):
        put(out, sx, sy, col)
        put(out, sx + 1, sy, PALE)
    return out


def synth(rig, action, f, palette):
    layers, pivots, center, foot_pivot, face = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = abs(p["spin"]) > 0.05
    recoiling = abs(p["hurt"]) > 0.05
    wing = 0.0 if spinning or recoiling else p["wing"]
    ant = 0.0 if spinning or recoiling else p["ant"]
    # Positive wing lifts both tips. Positive ant flares both stalks outward.
    left = turn(layers["left_wing"], -wing, pivots["left_wing"])
    right = turn(layers["right_wing"], wing, pivots["right_wing"])
    ant_l = turn(layers["ant_l"], -ant, pivots["ant_l"])
    ant_r = turn(layers["ant_r"], ant, pivots["ant_r"])
    img = np.zeros_like(layers["body"])
    paste(img, layers["body"])
    paste(img, left)
    paste(img, right)
    paste(img, ant_l)
    paste(img, ant_r)
    if abs(p["tilt"]) >= 0.45 and not spinning and not recoiling:
        img = turn(img, p["tilt"], center)
    if spinning:
        img = turn(img, p["spin"], center)
    elif recoiling:
        img = turn(img, p["hurt"], foot_pivot)
    img = shift(img, p["dx"], p["dy"])
    img = snap(img, palette)
    if p["rings"] and not spinning:
        img = roar_rings(img, face, p["ring"], p["rings"])
    if p["burst"]:
        img = impact_burst(img, face)
    if p["sparks"]:
        img = sparks(img, f)
    return snap(img, palette)


def facing_flip(img, direction):
    if direction in MIRROR:
        return img[:, ::-1].copy()
    return img


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (SPECIES / "front.png").read_bytes()
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
        "attack": (0, 2, 4, 6, 8, 10, 11, 15),
        "hurt": (0, 2, 4, 6, 8, 11),
    }
    cols = max(len(v) for v in samples.values())
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            y = ai * CELL_H
            x = si * CELL_W
            m = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 4
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
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


def opaque_count(tile):
    return int((tile[..., 3] == 255).sum())


def south_attack_ratios(sheets):
    row = sheets["attack"][:CELL_H]
    base = opaque_count(row[:, :CELL_W])
    ratios = []
    for f in range(FRAMES["attack"]):
        n = opaque_count(row[:, f * CELL_W:(f + 1) * CELL_W])
        ratios.append(n / base if base else 0.0)
    return base, ratios


def largest_fraction(tile):
    a = tile[..., 3] >= 128
    parts = components(a)
    if not parts:
        return 0.0
    parts.sort(key=len, reverse=True)
    return len(parts[0]) / max(1, int(a.sum()))


def qa_numbers(master, sheets, palette, canvas):
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = [int(v) for v in np.unique(master[..., 3])]
    bad_alpha = [v for v in alphas if v not in (0, 255)]
    clip = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
    west = sheets["idle"][6 * CELL_H:7 * CELL_H, :CELL_W]
    differ = np.any(west != canvas, axis=2) & ((west[..., 3] >= 128) | (canvas[..., 3] >= 128))
    base, ratios = south_attack_ratios(sheets)
    bad_ratio = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
    spin_frac = []
    for f in range(5, 11):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        spin_frac.append(round(largest_fraction(tile), 3))
    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] == 255)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    return {
        "off": off,
        "bad_alpha": bad_alpha,
        "clip": clip,
        "rest_diff": int(differ.sum()),
        "base": base,
        "ratios": ratios,
        "bad_ratio": bad_ratio,
        "spin_frac": spin_frac,
        "sprite_h": sprite_h,
    }


def _entry_span(text, key):
    token = f'"{key}":'
    start = text.index(token)
    i = start + len(token)
    while text[i] != "{":
        i += 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return start, j + 1
    raise SystemExit(f"dir8 entry did not close: {key}")


def publish_dir8(sprite_h):
    entry = {
        "dex": 414,
        "form": "normal",
        "variant": "mothim",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0414_mothim/normal",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir",
        "nf": 16,
        "cw": CELL_W,
        "ch": CELL_H,
        "w": CELL_W,
        "h": int(sprite_h),
        "fy": int(FOOT_Y),
        "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1,
        "src": "agent-sprite-forge",
        "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    DIR8_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with DIR8_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        keys = re.findall(r'"(\d{4}_[^"]+)":\{', text)
        if KEY not in keys:
            raise SystemExit("0414_mothim missing from dir8.js")
        idx = keys.index(KEY)
        neighbor = keys[idx + 1] if idx + 1 < len(keys) else keys[idx - 1]
        nb_s, nb_e = _entry_span(text, neighbor)
        neighbor_blob = text[nb_s:nb_e]
        s, e = _entry_span(text, KEY)
        text = text[:s] + f'"{KEY}":' + blob + text[e:]
        if neighbor_blob not in text:
            raise SystemExit("neighbor dir8 key changed")
        body = text[len(prefix):].strip()
        if body.endswith(";"):
            body = body[:-1].strip()
        json.loads(body)
        if text.count(f'"v":"{VERSION}"') != 1:
            raise SystemExit("dir8 version stamp collided")
        DIR8.write_text(text)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return neighbor


def update_queue(status):
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = QUEUE.read_text()
        start = text.index('"id": "0414_mothim"')
        nxt = text.find('"id":', start + 10)
        end = len(text) if nxt < 0 else nxt
        block = text[start:end]
        if '"status":' not in block:
            raise SystemExit("queue status missing")
        block2 = re.sub(r'"status": "[^"]*"', f'"status": "{status}"', block, count=1)
        mid, n_mid = re.subn(r'\n\s*"claim": "[^"]*",', "", block2, count=1)
        if n_mid:
            block2 = mid
        else:
            block2 = re.sub(r',\s*\n\s*"claim": "[^"]*"', "", block2, count=1)
        text = text[:start] + block2 + text[end:]
        data = json.loads(text)
        ids = [item.get("id") for item in data["items"]]
        if ids.count(KEY) != 1:
            raise SystemExit("queue id collision")
        for item in data["items"]:
            if item.get("id") == KEY:
                if item.get("status") != status or "claim" in item:
                    raise SystemExit("queue update did not stick")
        QUEUE.write_text(text if text.endswith("\n") else text + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front_path = SPECIES / "front.png"
    front = np.array(Image.open(front_path).convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), flush=True)

    rgb = front[..., :3]
    opaque = front[..., 3] >= 128
    pale = opaque & np.all(rgb == np.array(PALE, np.uint8), axis=-1)
    eyes = pale & (np.arange(96)[:, None] < 48)
    ey, ex = np.nonzero(eyes)
    face_x = float(ex.mean())
    by, bx = np.nonzero(opaque)
    body_x = float(bx.mean())
    print(f"facing face_x {face_x:.1f} body_x {body_x:.1f}", flush=True)
    if not (face_x < body_x - 2):
        raise SystemExit("facing check failed; eyes are not left of the body")

    layers, pivots, center, foot_pivot, face, canvas = build_rig(front, masks)
    rig = (layers, pivots, center, foot_pivot, face)
    print("pivot", [round(v, 2) for v in center], "foot", [round(v, 2) for v in foot_pivot],
          "face", [round(v, 2) for v in face], flush=True)

    rest = synth(rig, "idle", 0, palette)
    differ = np.any(rest != canvas, axis=2) & ((rest[..., 3] >= 128) | (canvas[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "body": (70, 90, 140),
        "left_wing": (210, 70, 70),
        "right_wing": (230, 190, 60),
        "ant_l": (230, 230, 230),
        "ant_r": (80, 210, 210),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", f) for f in (0, 4, 8, 12)]
        want += [("walk", f) for f in (0, 4, 8, 12)]
        want += [("attack", f) for f in range(16)]
        want += [("hurt", f) for f in (0, 2, 4, 6, 8, 11)]
        tiles = [facing_flip(synth(rig, act, f, palette), "S") for act, f in want]
        base = opaque_count(tiles[8])
        print("sample attack S ratios", flush=True)
        for i, f in enumerate(range(16)):
            n = opaque_count(tiles[8 + i])
            print(f"  {f:02d} {n} {n / base:.3f}", flush=True)
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
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
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if key in state.get("done", []) and fp.exists() and fp.stat().st_size > 0 and not args.force:
                    continue
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest and not args.force:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(facing_flip(synth(rig, act, f, palette), d), "RGBA").save(tmp)
                tmp.replace(fp)
            if key not in state["done"]:
                state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions_dir = SPECIES / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    preview.save(actions_dir / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    report = qa_numbers(master, sheets, palette, canvas)
    base = report["base"]
    ratios = report["ratios"]
    print("attack S opaque ratios", flush=True)
    for f, r in enumerate(ratios):
        n = int(round(r * base))
        print(f"  {f:02d} {n} {r:.3f}", flush=True)
    print(f"min {min(ratios):.3f} max {max(ratios):.3f} h {report['sprite_h']}", flush=True)

    qa = work / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    gait = {}
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
                gait[act] = json.loads(rep.read_text()).get("recommendation")
    summary = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": report["sprite_h"],
        "off_palette": len(report["off"]),
        "bad_alpha": report["bad_alpha"],
        "clip": report["clip"],
        "rest_diff": report["rest_diff"],
        "attack_S_ratios": [round(r, 3) for r in ratios],
        "spin_main_frac": report["spin_frac"],
        "gait": gait,
    }
    (qa / "qa.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "parts"}, ensure_ascii=False)[:1200], flush=True)

    meta = {
        "generator": "forge_mothim_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 414,
        "form": "normal",
        "variant": "mothim",
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": report["sprite_h"],
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "mirror": sorted(MIRROR),
        "palette_colors": int(len(palette)),
        "off_palette_colors": len(report["off"]),
        "source": "front.png 96x96",
        "gifs_untouched": True,
        "wings": 2,
        "antennae": 2,
        "mouth": 0,
        "face_x": round(face_x, 2),
        "body_x": round(body_x, 2),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = (
        report["off"] or report["bad_alpha"] or report["clip"] or report["rest_diff"]
        or report["bad_ratio"] or any(v < 0.90 for v in report["spin_frac"])
    )
    if failed:
        raise SystemExit(
            f"qa failed off={len(report['off'])} alpha={report['bad_alpha']} "
            f"clip={report['clip'][:8]} rest={report['rest_diff']} "
            f"ratio={report['bad_ratio']} spin={report['spin_frac']}"
        )
    if args.publish and not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
        neighbor = publish_dir8(report["sprite_h"])
        update_queue("done")
        print("published", neighbor, flush=True)


if __name__ == "__main__":
    main()
