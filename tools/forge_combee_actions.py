#!/usr/bin/env python3
"""Combee (0415_combee) 8-direction action sheet from the normal front.png.

Measured on this still only (96x96, opaque 933, bbox x18-77 y32-64, 9 colours).
One component. Not a profile: three viewer-facing bees (two eye-pixels each)
sit as a triangle, upper-left, upper-right, and lower-center. Two antenna
stalks occupy y32-37. Two wings: pale membrane plus blue-gray tip and
highlight, left x18-27 (80px) and right x58-77 (158px). The right wing is
larger, so the centroid sits left of the bbox center. That is wing bulk, not
a left-facing beak. No mouth and no queen abdomen. South keeps the still.
N NW W SW are mirrored.

Roar flares the two wings and the two stalks, plus rings. The spin turns the
whole cluster as one piece (no |cos| squash). Idle keeps the lowest pixel on
frame 0. Walk is a wing beat. Palette-locked. Alpha is 0 or 255.
front/back png and gif are not written.
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
SPECIES = ROOT / "monsterSprite/gen4/0415_combee/normal"
OUT_DIR = SPECIES / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0415_combee/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"
KEY = "0415_combee"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Front view. South is the still. The opposite four are mirrored.
MIRROR = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "combee_v2"
KEEP = 6

BLACK = (16, 16, 16)
GOLD = (255, 197, 0)
ORANGE = (238, 115, 0)
PALE = (238, 238, 255)
WING_COLORS = (PALE, (82, 90, 115), (189, 189, 213))
BODY_COLORS = (GOLD, ORANGE, (106, 90, 0), (164, 123, 0), (57, 57, 65))
PARTS = ("body", "wing_l", "wing_r", "ant_l", "ant_r")


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def components(mask, diag=False):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    ys, xs = np.nonzero(mask)
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diag:
        steps = steps + ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        q = deque([(y, x)])
        seen[y, x] = True
        pts = []
        while q:
            cy, cx = q.popleft()
            pts.append((cy, cx))
            for dy, dx in steps:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        out.append(pts)
    return out


def color_mask(img, colors):
    rgb = img[..., :3]
    opaque = img[..., 3] >= 128
    hit = np.zeros(opaque.shape, bool)
    for col in colors:
        hit |= np.all(rgb == np.array(col, np.uint8), axis=-1)
    return hit & opaque


def segment(img):
    """Three fused bees, two wings, two antennae. No fourth bee.

    Antennae are every opaque pixel at y<=37 (two stalks). Wing seeds are the
    pale membrane, the blue-gray tip, and the highlight. Body seeds are the
    gold, orange, brown, and dark-gray hexes. Black outline pixels go to the
    nearest seed. The lower bee stays a hex face, not an abdomen.
    """
    h, w = img.shape[:2]
    opaque = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    ant = opaque & (yy <= 37)
    ant_l = ant & (xx < 40)
    ant_r = ant & (xx >= 40)
    wing_seed = color_mask(img, WING_COLORS) & ~ant
    body_seed = color_mask(img, BODY_COLORS) & ~ant & ~wing_seed
    seeds = {
        "ant_l": ant_l,
        "ant_r": ant_r,
        "wing_l": wing_seed & (xx < 40),
        "wing_r": wing_seed & (xx >= 40),
        "body": body_seed,
    }
    if min(int(v.sum()) for v in seeds.values()) <= 0:
        raise SystemExit("combee seed mask is empty")
    assigned = np.zeros_like(opaque)
    for m in seeds.values():
        assigned |= m
    rest = opaque & ~assigned
    pts = {}
    for name, m in seeds.items():
        ys, xs = np.nonzero(m)
        pts[name] = np.stack([ys, xs], axis=1).astype(np.int32)
    masks = {name: m.copy() for name, m in seeds.items()}
    ry, rx = np.nonzero(rest)
    for y, x in zip(ry.tolist(), rx.tolist()):
        best, bestd = None, 10 ** 9
        for name, p in pts.items():
            d = int(((p[:, 0] - y) ** 2 + (p[:, 1] - x) ** 2).min())
            if d < bestd:
                bestd = d
                best = name
        masks[best][y, x] = True
    own = np.zeros_like(opaque)
    overlap = 0
    for m in masks.values():
        overlap += int((own & m).sum())
        own |= m
    if int(own.sum()) != int(opaque.sum()) or overlap:
        raise SystemExit(f"masks do not partition the still overlap={overlap}")
    if components(masks["body"]) and len(components(masks["body"])) != 1:
        raise SystemExit("body is not one fused cluster")
    if len(components(masks["wing_l"])) != 1 or len(components(masks["wing_r"])) != 1:
        raise SystemExit("a wing split")
    if len(components(masks["ant_l"])) != 1 or len(components(masks["ant_r"])) != 1:
        raise SystemExit("an antenna split")
    return masks


def count_bees(img):
    """Each bee has one pair of black eye pixels. Expect 3 bees, 6 eyes."""
    opaque = img[..., 3] >= 128
    black = opaque & np.all(img[..., :3] == np.array(BLACK, np.uint8), axis=-1)
    eyes = []
    for pts in components(black):
        if len(pts) != 2:
            continue
        cy = sum(p[0] for p in pts) / 2
        cx = sum(p[1] for p in pts) / 2
        if 44 <= cy <= 57 and 29 <= cx <= 52:
            eyes.append((cx, cy))
    return len(eyes) // 2, len(eyes)


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

    Positive degrees are clockwise. No |cos| squash.
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
        d = (ty[sl, None].astype(np.int32) - pts[None, :, 0]) ** 2
        d += (tx[sl, None].astype(np.int32) - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx, :3] = cols[pick, :3]
    layer[ty, tx, 3] = 255
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

    # Joint cushion under the wings and stalks, hidden while they rest.
    for part_name in ("wing_l", "wing_r", "ant_l", "ant_r"):
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
    black = np.all(canvas[..., :3] == np.array(BLACK, np.uint8), axis=-1)
    eye = placed["body"] & black
    ey, ex = np.nonzero(eye)
    face = (float(ex.mean()), float(ey.mean())) if len(ex) else center
    pivots = {
        "wing_l": edge_pivot(placed["wing_l"], "inner_right"),
        "wing_r": edge_pivot(placed["wing_r"], "inner_left"),
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
    p = dict(dx=0, dy=0, wing=0.0, ant=0.0, spin=0.0, hurt=0.0,
             rings=0, ring=0, burst=0, sparks=0)
    if action == "idle":
        # Contact stays on frame 0. Only the wings and stalks move.
        p["wing"] = wave(f, n, 8.0)
        p["ant"] = wave(f, n, 6.0, phase=-0.5)
    elif action == "walk":
        p["wing"] = wave(f, n, 16.0, cycles=2.0)
        p["ant"] = wave(f, n, 7.0, phase=-0.4, cycles=2.0)
    elif action == "attack":
        wing = [0, 8, 12, 14, 10, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 0]
        ant = [0, 8, 12, 14, 8, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0]
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
        p["hurt"] = -36.0 * env
        p["dy"] = int(round(-2 * env))
        p["sparks"] = 1 if 2 <= f <= 5 else 0
    if abs(p["hurt"]) > 38.01:
        raise SystemExit("hurt angle exceeds 38")
    return p


def turn(layer, angle, pivot):
    if abs(angle) < 0.05 or abs(angle) % 360 < 0.05:
        return layer
    return despeckle(rotate_nearest(layer, angle, pivot))


def roar_rings(out, center, age, strength):
    """Rings outside the cluster. This still has no mouth."""
    ox, oy = center
    cols = (PALE, GOLD, ORANGE)
    for r_i in range(strength):
        rad = 36 + age * 2 + r_i * 4
        step = 24
        for deg in range(0, 360, step):
            ang = math.radians(deg)
            put(out, ox + rad * math.cos(ang), oy + rad * math.sin(ang) * 0.72, cols[r_i % 3])
    return out


def impact_burst(out, center):
    ox, oy = center
    cols = (GOLD, ORANGE, PALE)
    for k in range(10):
        ang = math.radians(-90 + k * 36)
        put(out, ox + 28 * math.cos(ang), oy + 18 * math.sin(ang), cols[k % 3])
        put(out, ox + 34 * math.cos(ang), oy + 22 * math.sin(ang), cols[(k + 1) % 3])
    return out


def sparks(out, frame_idx):
    a = out[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return out
    x0, y0 = int(xs.min()), int(ys.min())
    x1 = int(xs.max())
    d = frame_idx
    for sx, sy, col in (
        (x0 - 2 - d, y0 + 6, PALE),
        (x1 + 2 + d, y0 + 6, GOLD),
        ((x0 + x1) // 2, y0 - 3 - d, ORANGE),
    ):
        put(out, sx, sy, col)
    return out


def synth(rig, action, f, palette):
    layers, pivots, center, foot_pivot, face = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = abs(p["spin"]) > 0.05
    recoiling = abs(p["hurt"]) > 0.05
    wing = 0.0 if spinning or recoiling else p["wing"]
    ant = 0.0 if spinning or recoiling else p["ant"]
    # Positive wing lifts both tips. Positive ant flares the right stalk out.
    left = turn(layers["wing_l"], -wing, pivots["wing_l"])
    right = turn(layers["wing_r"], wing, pivots["wing_r"])
    ant_l = turn(layers["ant_l"], -ant, pivots["ant_l"])
    ant_r = turn(layers["ant_r"], ant, pivots["ant_r"])
    img = np.zeros_like(layers["body"])
    paste(img, layers["body"])
    paste(img, left)
    paste(img, right)
    paste(img, ant_l)
    paste(img, ant_r)
    if spinning:
        img = turn(img, p["spin"], center)
    elif recoiling:
        img = turn(img, p["hurt"], foot_pivot)
    img = shift(img, p["dx"], p["dy"])
    img = snap(img, palette)
    if p["rings"] and not spinning:
        img = roar_rings(img, center, p["ring"], p["rings"])
    if p["burst"]:
        img = impact_burst(img, center)
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


def bottom_y(tile):
    ys = np.nonzero(tile[..., 3] == 255)[0]
    return int(ys.max()) if len(ys) else -1


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
    south = sheets["idle"][:CELL_H, :CELL_W]
    differ = np.any(south != canvas, axis=2) & ((south[..., 3] >= 128) | (canvas[..., 3] >= 128))
    base, ratios = south_attack_ratios(sheets)
    bad_ratio = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
    spin_frac = []
    for f in range(5, 11):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        spin_frac.append(round(largest_fraction(tile), 3))
    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] == 255)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    idle_bottoms = []
    for f in range(FRAMES["idle"]):
        tile = sheets["idle"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        idle_bottoms.append(bottom_y(tile))
    walk_bottoms = []
    for f in range(FRAMES["walk"]):
        tile = sheets["walk"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        walk_bottoms.append(bottom_y(tile))
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
        "idle_bottoms": idle_bottoms,
        "walk_bottoms": walk_bottoms,
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
        "dex": 415,
        "form": "normal",
        "variant": "combee",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0415_combee/normal",
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
        if text.count(f'"{KEY}":') != 1:
            raise SystemExit("0415_combee key is not unique")
        keys = re.findall(r'"(\d{4}_[^"]+)":\{', text)
        if KEY not in keys:
            raise SystemExit("0415_combee missing from dir8.js")
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
        start = text.index('"id": "0415_combee"')
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
    bees, eyes = count_bees(front)
    print("parts", counts, "palette", len(palette), "bees", bees, "eyes", eyes, flush=True)
    if bees != 3 or eyes != 6:
        raise SystemExit(f"expected 3 bees and 6 eyes, got bees={bees} eyes={eyes}")
    if counts["wing_l"] <= 0 or counts["wing_r"] <= 0:
        raise SystemExit("missing a wing")

    by, bx = np.nonzero(front[..., 3] >= 128)
    print(
        f"bbox x{int(bx.min())}-{int(bx.max())} y{int(by.min())}-{int(by.max())} "
        f"centroid {float(bx.mean()):.1f},{float(by.mean()):.1f}",
        flush=True,
    )

    layers, pivots, center, foot_pivot, face, canvas = build_rig(front, masks)
    rig = (layers, pivots, center, foot_pivot, face)
    print(
        "pivot", [round(v, 2) for v in center],
        "foot", [round(v, 2) for v in foot_pivot],
        "face", [round(v, 2) for v in face],
        flush=True,
    )

    rest = synth(rig, "idle", 0, palette)
    differ = np.any(rest != canvas, axis=2) & ((rest[..., 3] >= 128) | (canvas[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")
    if bottom_y(rest) != bottom_y(canvas):
        raise SystemExit("idle frame 0 contact moved")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "body": (70, 90, 140),
        "wing_l": (210, 70, 70),
        "wing_r": (230, 190, 60),
        "ant_l": (230, 230, 230),
        "ant_r": (80, 210, 210),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        attack = [facing_flip(synth(rig, "attack", f, palette), "S") for f in range(16)]
        base = opaque_count(attack[0])
        print("sample attack S ratios", flush=True)
        bad = []
        for f, tile in enumerate(attack):
            n = opaque_count(tile)
            r = n / base
            print(f"  {f:02d} {n} {r:.3f}", flush=True)
            if r < 0.90 or r > 1.08:
                bad.append(round(r, 3))
        idle_b = [bottom_y(facing_flip(synth(rig, "idle", f, palette), "S")) for f in range(16)]
        walk_b = [bottom_y(facing_flip(synth(rig, "walk", f, palette), "S")) for f in (0, 4, 8, 12)]
        print("idle bottoms", idle_b, "walk sample bottoms", walk_b, flush=True)
        if any(b != idle_b[0] for b in idle_b):
            raise SystemExit("idle contact left frame 0")
        cols = 8
        tiles = attack
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        print("samples", work / "samples.png", "bad", bad, flush=True)
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
    print(
        f"min {min(ratios):.3f} max {max(ratios):.3f} h {report['sprite_h']} "
        f"idle_bottoms {report['idle_bottoms']}",
        flush=True,
    )

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
    contact_ok = len(set(report["idle_bottoms"])) == 1
    summary = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "bees": bees,
        "wings": 2,
        "mouth": 0,
        "sprite_h": report["sprite_h"],
        "off_palette": len(report["off"]),
        "bad_alpha": report["bad_alpha"],
        "clip": report["clip"],
        "rest_diff": report["rest_diff"],
        "attack_S_ratios": [round(r, 3) for r in ratios],
        "spin_main_frac": report["spin_frac"],
        "idle_contact_planted": contact_ok,
        "gait": gait,
    }
    (qa / "qa.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "parts"}, ensure_ascii=False)[:1500], flush=True)

    meta = {
        "generator": "forge_combee_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 415,
        "form": "normal",
        "variant": "combee",
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
        "bees": 3,
        "wings": 2,
        "antennae": 2,
        "mouth": 0,
        "parts": counts,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = (
        report["off"] or report["bad_alpha"] or report["clip"] or report["rest_diff"]
        or report["bad_ratio"] or any(v < 0.90 for v in report["spin_frac"])
        or not contact_ok
    )
    if failed:
        if args.publish:
            update_queue("needs_fix")
        raise SystemExit(
            f"qa failed off={len(report['off'])} alpha={report['bad_alpha']} "
            f"clip={report['clip'][:8]} rest={report['rest_diff']} "
            f"ratio={report['bad_ratio']} spin={report['spin_frac']} contact={contact_ok}"
        )
    if args.publish and not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
        neighbor = publish_dir8(report["sprite_h"])
        update_queue("done")
        print("published", neighbor, flush=True)


if __name__ == "__main__":
    main()
