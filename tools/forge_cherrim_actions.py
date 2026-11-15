#!/usr/bin/env python3
"""Cherrim overcast (0421) 8-direction sheet from this folder's 96x96 front.png.

Measured on this still only (opaque 985, bbox x33-62 y24-70, 13 colours, alpha 0/255).
No eyes and no mouth. The pink cherry sits on top (y24-35). Green leaves collar
y31-46. Two purple cloak petals meet near x49 (left 259, right 218). The pink
stubs at y62-70 are the base already in the picture (68 px). They are not a
gait, and no extra legs are drawn. This is the closed overcast flower: petals
flare a few degrees, they do not open into the sunshine face.

The still shows both petals and both stubs, so it faces the viewer.
S SE E NE stay unflipped. N NW W SW mirror.

idle   : base, leaves, and petals planted. The cherry bobs. Frame 0 is the still.
walk   : one-piece hop and sideways sway. The stubs do not step.
attack : 0-4 petal flare plus rings, 5-10 one-piece in-plane 360 spin,
         11-12 impact, 13-15 close. No |cos| squash.
hurt   : rigid lean of at most 38 degrees.

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
STILL_DIR = ROOT / "monsterSprite/gen4/0421_cherrim/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0421_cherrim/normal/agent-sprite-forge")
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
VERSION = "cherrim_v2"
KEY = "0421_cherrim"
UNFLIPPED = {"S", "SE", "E", "NE"}
MAX_RECOIL = 38.0

PINK = {(205, 82, 123), (164, 57, 98), (222, 115, 148), (123, 41, 74)}
GREEN = {(82, 172, 90), (65, 123, 74), (90, 197, 90), (16, 82, 32)}
PURPLE = {(106, 65, 123), (131, 90, 172), (156, 123, 189), (74, 41, 65)}
INK = (16, 16, 16)
PINK_LT = (222, 115, 148)
PINK_MID = (205, 82, 123)
GREEN_LT = (90, 197, 90)
PURPLE_MID = (131, 90, 172)

# Roar flare in degrees. Frames 5-10 are the closed-body spin, so the flare is 0.
OPEN = [0, 8, 12, 14, 10, 0, 0, 0, 0, 0, 0, 0, 0, 6, 4, 0]
RING = [-1, 0, 1, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


def color_mask(img, colors):
    rgb = img[..., :3]
    opaque = img[..., 3] >= 128
    mask = np.zeros(opaque.shape, bool)
    for color in colors:
        mask |= opaque & np.all(rgb == np.array(color, np.uint8), axis=-1)
    return mask


def flood_count(mask):
    return int(mask.sum())


def components(mask):
    seen = np.zeros_like(mask)
    count = 0
    best = 0
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        count += 1
        stack = [(y0, x0)]
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
        if n > best:
            best = n
    return count, best


def segment(img):
    """Cherry, leaf collar, two cloak petals, and the pink base. No invented limbs."""
    opaque = img[..., 3] >= 128
    if int(opaque.sum()) != 985:
        raise SystemExit(f"still opaque changed: {int(opaque.sum())}")
    h, w = opaque.shape
    yy, xx = np.mgrid[0:h, 0:w]
    pink = color_mask(img, PINK)
    green = color_mask(img, GREEN)
    purple = color_mask(img, PURPLE)
    ink = opaque & np.all(img[..., :3] == np.array(INK, np.uint8), axis=-1)
    if int((pink | green | purple | ink).sum()) != int(opaque.sum()):
        raise SystemExit("a still pixel is outside the measured colours")
    if int((pink & (yy > 40) & (yy < 60)).sum()) != 0:
        raise SystemExit("pink appeared between the cherry and the base")
    labeled = np.zeros(opaque.shape, np.int16)
    labeled[pink & (yy <= 40)] = 1
    labeled[green] = 2
    labeled[purple] = 3
    labeled[pink & (yy >= 60)] = 4
    src_y, src_x = np.nonzero(labeled > 0)
    src = np.stack([src_y, src_x], axis=1).astype(np.int32)
    src_lab = labeled[src_y, src_x]
    uy, ux = np.nonzero(opaque & (labeled == 0))
    unk = np.stack([uy, ux], axis=1).astype(np.int32)
    assign = np.empty(len(unk), np.int16)
    for i in range(0, len(unk), 300):
        sl = slice(i, i + 300)
        dist = (unk[sl, None, 0] - src[None, :, 0]) ** 2 + (unk[sl, None, 1] - src[None, :, 1]) ** 2
        assign[sl] = src_lab[dist.argmin(axis=1)]
    labeled[uy, ux] = assign
    petal = labeled == 3
    labeled[petal & (xx >= 49)] = 5
    names = {1: "blossom", 2: "leaves", 3: "petal_l", 5: "petal_r", 4: "base"}
    parts = {name: labeled == idx for idx, name in names.items()}
    own = np.zeros_like(opaque)
    overlap = 0
    for name in ("blossom", "leaves", "petal_l", "petal_r", "base"):
        overlap += int((own & parts[name]).sum())
        own |= parts[name]
    if int(own.sum()) != int(opaque.sum()) or overlap:
        raise SystemExit(f"masks do not partition the still: covered {int(own.sum())} overlap {overlap}")
    expect = {"blossom": 80, "leaves": 200, "petal_l": 150, "petal_r": 150, "base": 40}
    for name, minimum in expect.items():
        if flood_count(parts[name]) < minimum:
            raise SystemExit(f"{name} collapsed: {flood_count(parts[name])}")
        ncomp, _best = components(parts[name])
        if ncomp != 1:
            raise SystemExit(f"{name} is not one piece ({ncomp})")
    ys, xs = np.nonzero(opaque)
    if (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) != (33, 24, 62, 70):
        raise SystemExit("still bbox changed; remeasure before forging")
    return parts


def measure_face(parts):
    """Both petals and both sides of the base are visible: the still faces the viewer."""
    center = 47.5
    for name in ("petal_l", "petal_r", "base"):
        xs = np.nonzero(parts[name])[1]
        if float(xs.min()) >= center or float(xs.max()) <= center and name == "base":
            pass
    base_x = np.nonzero(parts["base"])[1]
    left = np.nonzero(parts["petal_l"])[1]
    right = np.nonzero(parts["petal_r"])[1]
    if int(base_x.min()) >= 47 or int(base_x.max()) <= 48:
        raise SystemExit("base does not sit under both sides of the flower")
    if int(left.max()) < 40 or int(right.min()) > 55:
        raise SystemExit("petals are not a closed front pair")
    return {
        "still_faces": "viewer",
        "blossom": flood_count(parts["blossom"]),
        "leaves": flood_count(parts["leaves"]),
        "petal_l": flood_count(parts["petal_l"]),
        "petal_r": flood_count(parts["petal_r"]),
        "base": flood_count(parts["base"]),
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
    """4x nearest. A multiple of 360 stays exact. No |cos| squash."""
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
    n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img, target):
    """Fill rotation pinholes with a neighbour colour. Do not grow past the still."""
    out = img.copy()
    limit = int(target * 1.04)
    for neighbors in (4, 4, 3):
        a = out[..., 3] >= 128
        have = int(a.sum())
        if have >= target or have >= limit:
            break
        pad = np.pad(a, 1)
        n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        holes = (~a) & (n >= neighbors)
        ys, xs = np.nonzero(holes)
        if len(ys) == 0:
            continue
        room = limit - have
        if room <= 0:
            break
        if len(ys) > room:
            ys, xs = ys[:room], xs[:room]
        for y, x in zip(ys.tolist(), xs.tolist()):
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < out.shape[0] and 0 <= nx < out.shape[1] and out[ny, nx, 3] >= 128:
                    out[y, x] = out[ny, nx]
                    break
    return out


def extract_palette(img):
    opaque = img[..., 3] >= 128
    return np.unique(img[opaque][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    opaque = out[..., 3] >= 128
    if not np.any(opaque):
        out[..., 3] = 0
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = opaque.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    mapped = palette[dist.argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)
    out[~opaque] = 0
    return out


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def side_of(mask, center_x):
    xs = np.nonzero(mask)[1]
    return -1.0 if float(xs.mean()) < center_x else 1.0


def top_socket(mask, inward):
    ys, xs = np.nonzero(mask)
    top = int(ys.min())
    sel = ys <= top + 2
    x = float(xs[sel].max() if inward > 0 else xs[sel].min())
    return x, float(ys[sel].mean())


def inner_strip(mask, inward, width=2):
    out = np.zeros_like(mask)
    rows = np.unique(np.nonzero(mask)[0])
    for y in rows.tolist():
        xs = np.nonzero(mask[y])[0]
        take = xs[-width:] if inward > 0 else xs[:width]
        out[y, take] = True
    return out


def outside_dots(canvas, face):
    """Fixed ring dots in the empty margin above the cherry. Same dots every roar."""
    cx, cy = face
    dots = []
    for ring, rad, step in ((0, 24, 22), (1, 30, 20), (2, 36, 18)):
        deg = 214
        while deg <= 326:
            ang = math.radians(deg)
            x = int(round(cx + rad * math.cos(ang)))
            y = int(round(cy + rad * math.sin(ang)))
            if 2 <= x < CELL_W - 2 and 2 <= y < CELL_H - 2 and int(canvas[y - 2:y + 3, x - 2:x + 3, 3].max()) == 0:
                dots.append((x, y, ring))
            deg += step
    if len(dots) < 12:
        raise SystemExit(f"roar rings landed on the flower ({len(dots)} dots)")
    return dots


def build_rig(img, masks, flip):
    opaque = img[..., 3] >= 128
    ys, xs = np.nonzero(opaque)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    if dst_y < 16 or CELL_H - FOOT_Y < 16:
        raise SystemExit("cell padding is tighter than the still margin")
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]

    placed = {}
    for name, mask in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = mask[y0:y1, x0:x1]
        placed[name] = pm

    layers = {}
    for name, pm in placed.items():
        layer = np.zeros_like(canvas)
        layer[pm] = canvas[pm]
        layers[name] = layer

    # Seam copy stays put while the petals flare, so the cloak does not crack open
    # into a face. It is covered exactly while the petals are closed.
    seam = inner_strip(placed["petal_l"], inward=1) | inner_strip(placed["petal_r"], inward=-1)
    cushion = np.zeros_like(canvas)
    cushion[seam] = canvas[seam]
    layers["cushion"] = cushion

    pivots = {
        "petal_l": top_socket(placed["petal_l"], inward=1),
        "petal_r": top_socket(placed["petal_r"], inward=-1),
    }
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    by, bx = np.nonzero(placed["blossom"])
    face = (float(bx.mean()), float(by.mean()))
    base_y, base_x = np.nonzero(placed["base"])
    bottom = int(base_y.max())
    foot_sel = base_y >= bottom - 1
    foot = (float(base_x[foot_sel].mean()), float(bottom))
    dots = outside_dots(canvas, face)

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        placed = {k: v[:, ::-1].copy() for k, v in placed.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
        face = (CELL_W - 1 - face[0], face[1])
        foot = (CELL_W - 1 - foot[0], foot[1])
        dots = [(CELL_W - 1 - x, y, ring) for x, y, ring in dots]
        canvas = canvas[:, ::-1].copy()

    sides = {name: side_of(placed[name], center[0]) for name in ("petal_l", "petal_r")}
    facing = 1.0 if not flip else -1.0
    return {
        "layers": layers,
        "pivots": pivots,
        "sides": sides,
        "center": center,
        "face": face,
        "foot": foot,
        "facing": facing,
        "canvas": canvas,
        "dots": dots,
    }


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(dx=0.0, dy=0.0, lift=0.0, open=0.0, spin=0.0, recoil=0.0, ring=-1, burst=0, sparks=0)
    if action == "idle":
        # Cherry only. Leaves, petals, and the base stay on the frame-0 feet.
        p["lift"] = -max(0.0, wave(f, n, 1.0, cycles=2.0))
    elif action == "walk":
        p["dx"] = wave(f, n, 2.0, cycles=2.0)
        p["dy"] = -3.0 * max(0.0, wave(f, n, 1.0, cycles=2.0))
    elif action == "attack":
        p["open"] = float(OPEN[f])
        p["ring"] = RING[f]
        if f in (2, 3, 4):
            p["lift"] = -1.0
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
        if f in (11, 12):
            p["burst"] = 2 if f == 11 else 1
    else:
        env = [0.0, 0.45, 0.75, 1.0, 0.82, 0.58, 0.36, 0.18, 0.08, 0.0, 0.0, 0.0][f]
        p["recoil"] = MAX_RECOIL * env
        p["dx"] = -2.0 * env
        p["dy"] = -1.0 * env
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def paste(dst, layer):
    mask = layer[..., 3] >= 128
    dst[mask] = layer[mask]


def compose(rig, p):
    """Base stays down. Petals flare from their top sockets. Cherry bobs on its own."""
    layers = rig["layers"]
    facing = rig["facing"]
    bdx, bdy = p["dx"] * facing, p["dy"]
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(body, shift(layers["cushion"], bdx, bdy))
    paste(body, shift(layers["base"], bdx, bdy))
    for name in ("petal_l", "petal_r"):
        ang = (-rig["sides"][name]) * p["open"]
        spun = rotate_nearest(layers[name], ang, rig["pivots"][name])
        paste(body, shift(spun, bdx, bdy))
    paste(body, shift(layers["leaves"], bdx, bdy))
    paste(body, shift(layers["blossom"], bdx, bdy + p["lift"]))
    return body


def paint_dots(out, dots, age, cols):
    for x, y, ring in dots:
        if ring <= age:
            put(out, x, y, cols[ring % len(cols)])
    return out


def impact_burst(out, face, size):
    cx, cy = face
    ticks = [(0, -30), (-16, -26), (16, -26), (-28, -16), (28, -16), (0, -38), (-22, -32), (22, -32)]
    col = PINK_LT if size > 1 else PINK_MID
    for dx, dy in ticks[: 6 + size]:
        put(out, cx + dx, cy + dy, col)
    return out


def sparks(out, frame_idx, face):
    cx, cy = face
    d = frame_idx
    pts = [(cx - 10 - d, cy - 22), (cx + 10 + d, cy - 20), (cx, cy - 28)]
    for sx, sy in pts:
        put(out, sx, sy, PINK_LT)
        put(out, sx, sy + 1, GREEN_LT)
    return out


def largest_fraction(img):
    opaque = img[..., 3] >= 128
    if not np.any(opaque):
        return 1.0, 0
    _count, best = components(opaque)
    total = int(opaque.sum())
    return best / total, total


def synth(rig, action, f, palette, effects=True):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        rest_p = pose(action, 0)
        rest_p = gen4_attack_strike.finish(rest_p, action, 0, __file__)

        rest_p["open"] = 0.0
        rest_p["lift"] = 0.0
        rest_p["dx"] = 0.0
        rest_p["dy"] = 0.0
        rest_p["spin"] = 0.0
        rest_p["recoil"] = 0.0
        body = compose(rig, rest_p)
        target = int((body[..., 3] >= 128).sum())
        if recoiling:
            body = shift(body, p["dx"] * rig["facing"], p["dy"])
            pivot = (rig["foot"][0] + p["dx"] * rig["facing"], rig["foot"][1] + p["dy"])
            body = rotate_nearest(body, -p["recoil"] * rig["facing"], pivot)
        else:
            body = rotate_nearest(body, p["spin"] * rig["facing"], rig["center"])
        body = despeckle(body)
        body = fill_pinholes(body, target)
    else:
        body = compose(rig, p)
        if p["open"] > 0.45:
            body = despeckle(body)
    img = snap(body, palette)
    if not effects:
        return img
    if p["ring"] >= 0 and not spinning:
        img = paint_dots(img, rig["dots"], p["ring"], (PINK_LT, PINK_MID, GREEN_LT))
    if p["burst"]:
        img = impact_burst(img, rig["face"], p["burst"])
    if p["sparks"]:
        moved = (rig["face"][0] + p["dx"] * rig["facing"], rig["face"][1] + p["dy"])
        img = sparks(img, f, moved)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (STILL_DIR / "front.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def assemble(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sheet = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, direction in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = np.array(Image.open(frames_dir / act / direction / f"{f:02d}.png"))
                sheet[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = tile
        sheets[act] = sheet
    return sheets


def write_outputs(sheets):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for act, sheet in sheets.items():
        image = Image.fromarray(sheet, "RGBA")
        image.save(OUT_DIR / f"{act}.png")
        image.save(OUT_DIR / f"{act}_8dir.png")
        Image.fromarray(sheet[:CELL_H], "RGBA").save(OUT_DIR / f"{act}_S_strip.png")
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
            mask = tile[..., 3] >= 128
            bg[ai * CELL_H:(ai + 1) * CELL_H, si * CELL_W:(si + 1) * CELL_W][mask] = tile[mask]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 4
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            opaque = tile[..., 3] >= 128
            if not np.any(opaque):
                continue
            ys, xs = np.nonzero(opaque)
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
        for ci, crop in enumerate(row):
            y = ri * cell_h + (cell_h - crop.shape[0]) // 2
            x = ci * cell_w + (cell_w - crop.shape[1]) // 2
            mask = crop[..., 3] >= 128
            bg[y:y + crop.shape[0], x:x + crop.shape[1]][mask] = crop[mask]
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
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return start, j + 1
    raise SystemExit(f"dir8 entry did not close: {key}")


def publish_dir8(sprite_h):
    entry = {
        "dex": 421,
        "form": "normal",
        "variant": "cherrim",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0421_cherrim/normal",
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
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise SystemExit("dir8.js header changed; not publishing")
        keys = re.findall(r'"(\d{4}_[^"]+)":\{', text)
        if keys.count(KEY) != 1:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise SystemExit("0421_cherrim key missing or duplicated")
        idx = keys.index(KEY)
        neighbor = keys[idx + 1] if idx + 1 < len(keys) else keys[idx - 1]
        other = keys[idx - 1] if idx > 0 else neighbor
        nb_s, nb_e = _entry_span(text, neighbor)
        ot_s, ot_e = _entry_span(text, other)
        neighbor_blob = text[nb_s:nb_e]
        other_blob = text[ot_s:ot_e]
        s, e = _entry_span(text, KEY)
        text = text[:s] + f'"{KEY}":' + blob + text[e:]
        if neighbor_blob not in text or other_blob not in text:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise SystemExit("neighbor dir8 key changed")
        body = text[len(prefix):].strip()
        if body.endswith(";"):
            body = body[:-1].strip()
        json.loads(body)
        if text.count(f'"v":"{VERSION}"') != 1:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise SystemExit("dir8 version stamp collided")
        DIR8.write_text(text)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return neighbor


def update_queue(status):
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = QUEUE.read_text()
        data = json.loads(text)
        before = {item.get("id"): (item.get("status"), item.get("claim")) for item in data["items"]}
        start = text.index('"id": "0421_cherrim"')
        nxt = text.find('"id":', start + 10)
        end = len(text) if nxt < 0 else nxt
        block = text[start:end]
        if '"status":' not in block:
            fcntl.flock(lock, fcntl.LOCK_UN)
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
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise SystemExit("queue id collision")
        for item in data["items"]:
            if item.get("id") == KEY:
                if item.get("status") != status or "claim" in item:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    raise SystemExit("queue update did not stick")
            else:
                if (item.get("status"), item.get("claim")) != before.get(item.get("id")):
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    raise SystemExit(f"queue neighbor changed: {item.get('id')}")
        QUEUE.write_text(text if text.endswith("\n") else text + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)


def qa_report(master, sheets, palette, ratios, sprite_h):
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted({int(v) for v in np.unique(master[..., 3])})
    clip = {}
    spin_stats = []
    hurt_widths = []
    rest_w = None
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
    face_info = measure_face(masks)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("facing", face_info, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    rest = synth(rigs["S"], "idle", 0, palette, effects=False)
    placed = rigs["S"]["canvas"]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "blossom": (222, 115, 148),
        "leaves": (90, 197, 90),
        "petal_l": (156, 123, 189),
        "petal_r": (106, 65, 123),
        "base": (205, 82, 123),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "S", f) for f in range(16)]
        want += [("walk", "S", f) for f in range(16)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in range(12)]
        want += [("attack", "W", f) for f in (3, 7, 11)]
        tiles = [(act, synth(rigs[d], act, f, palette)) for act, d, f in want]
        sheets = {}
        for act in ACTIONS:
            width = FRAMES[act] * CELL_W
            sheet = np.zeros((CELL_H, max(FRAMES.values()) * CELL_W, 4), np.uint8)
            for f in range(FRAMES[act]):
                tile = synth(rigs["S"], act, f, palette)
                sheet[0:CELL_H, f * CELL_W:(f + 1) * CELL_W] = tile
            sheets[act] = sheet
        contact = motion_contact(sheets)
        contact.save(work / "samples_motion.png")
        base = int((synth(rigs["S"], "attack", 0, palette)[..., 3] >= 128).sum())
        print("ATTACK_S_RATIOS", flush=True)
        ratios = []
        for f in range(16):
            tile = synth(rigs["S"], "attack", f, palette)
            n = int((tile[..., 3] >= 128).sum())
            frac, _total = largest_fraction(tile)
            ratio = n / base
            ratios.append(ratio)
            print(f"attack S {f:02d} opaque {n} ratio {ratio:.3f} main {frac:.3f}", flush=True)
        print(f"min {min(ratios):.3f} max {max(ratios):.3f}", flush=True)
        hurt = synth(rigs["S"], "hurt", 3, palette)
        hxs = np.nonzero(hurt[..., 3] >= 128)[1]
        rest_xs = np.nonzero(rest[..., 3] >= 128)[1]
        print(
            "hurt peak width",
            int(hxs.max() - hxs.min() + 1),
            "rest width",
            int(rest_xs.max() - rest_xs.min() + 1),
            "samples",
            work / "samples_motion.png",
            flush=True,
        )
        bad = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
        if bad:
            raise SystemExit(f"attack opaque ratio out of range at {bad}")
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

    report = qa_report(master, sheets, palette, ratios, sprite_h)
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
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
    public = {k: v for k, v in report.items() if not k.startswith("_")}
    (qa / "qa.json").write_text(json.dumps(public, ensure_ascii=False, indent=1))
    print(json.dumps(public, ensure_ascii=False)[:1600], flush=True)

    bad_spin = [s for s in report["spin_body"] if s["main"] < 0.90]
    failed = report["_off"] or report["_clip"] or report["alphas"] != [0, 255] or report["_bad_ratio"] or bad_spin or report["hurt_squash"]
    if failed:
        if args.publish:
            update_queue("needs_fix")
        raise SystemExit(
            f"qa failed off={len(report['_off'])} clip={report['_clip']} alphas={report['alphas']} "
            f"ratios={report['_bad_ratio']} spin={bad_spin} squash={report['hurt_squash']}"
        )

    meta = {
        "generator": "forge_cherrim_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 421,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED),
        "still_faces": "viewer",
        "face": face_info,
        "parts": counts,
        "palette_colors": int(len(palette)),
        "off_palette_colors": 0,
        "attack_s_min": round(min(ratios), 3),
        "attack_s_max": round(max(ratios), 3),
        "source": "front.png 96x96 overcast",
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
