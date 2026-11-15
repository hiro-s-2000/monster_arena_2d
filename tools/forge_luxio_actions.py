#!/usr/bin/env python3
"""Luxio (0404) 8-direction action sheet from this form's 96x96 front.png.

Measured on normal/front.png (opaque 1369, bbox x18-77 y24-71, 60x48, 13 colours,
alpha already 0 or 255). The snout tip is x=18. The eye shines sit near x=26 and
x=39 (mean x=34.5), so the still faces left. Unflipped dirs are N NW W SW.
Flipped dirs are S SE E NE, and E faces screen-right.

Three paws touch the ground. Each one is a single component. No fourth leg
is drawn, so none is invented:

  front paw  x27-35 y61-69  (62 px)  far front, gold ankle band
  mid paw    x37-50 y61-71  (103 px) near front, longer, yellow band
  rear paw   x52-60 y61-69  (62 px)  the one visible hind paw
  tail       x64-77 y44-64  (153 px) stalk plus the yellow star, above the ground

The mouth is the five red pixels at x28-30 y43-45. The roar drops those pixels
and adds rings. It does not grow a new jaw.

  idle   : 1px breath, tail sway. Feet stay planted. Frame 0 is the still.
  walk   : the two front paws alternate. The hind paw steps with the far front.
  attack : 0-4 roar; 5-10 in-place 360 of the whole body (no squash); 11-12
           impact; 13-15 settle.
  hurt   : one-piece lean, 38 degrees max, no belly flip.

Palette-locked. front/back png and gif are not written.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_luxio_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen4/0404_luxio/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0404_luxio/normal/agent-sprite-forge")
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
VERSION = "luxio_v2"
KEY = "0404_luxio"

INK = (16, 16, 16)
YELLOW = (255, 230, 90)
GOLD = (180, 156, 90)
WHITE = (255, 255, 255)
RED = (230, 74, 82)
DRED = (148, 57, 82)


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


def segment(img):
    """Cuts measured on this still. Three paws, one tail, the red mouth."""
    h, w = img.shape[:2]
    opaque = img[..., 3] >= 128
    rgb = img[..., :3]
    yy, xx = np.mgrid[0:h, 0:w]
    red = opaque & (
        np.all(rgb == np.array(RED, np.uint8), axis=-1)
        | np.all(rgb == np.array(DRED, np.uint8), axis=-1)
    )
    # The mouth only. Eye shines are white and stay on the head.
    jaw = red & (yy >= 42) & (yy <= 46) & (xx >= 26) & (xx <= 32)

    # Feet split below y60. y60 still joins the two front paws.
    front = opaque & (yy >= 61) & (xx <= 36)
    mid = opaque & (yy >= 61) & (xx >= 37) & (xx <= 51)
    rear = opaque & (yy >= 61) & (xx >= 52) & (xx <= 63)
    # Star and stalk. One component, clear of every paw.
    tail = opaque & (xx >= 64) & (yy <= 64) & ~front & ~mid & ~rear
    body = opaque & ~front & ~mid & ~rear & ~tail & ~jaw
    return {"body": body, "front": front, "mid": mid, "rear": rear, "tail": tail, "jaw": jaw}


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
    """Drop a pixel only when all four neighbours are empty."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img):
    """Close a 1px hole that already has four opaque neighbours. No outline growth."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    hole = (~a) & (n == 4)
    if not np.any(hole):
        return out
    ys, xs = np.nonzero(hole)
    for y, x in zip(ys.tolist(), xs.tolist()):
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            ny, nx = y + dy, x + dx
            if a[ny, nx]:
                out[y, x] = out[ny, nx]
                break
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


def top_rows(mask, rows):
    ys = np.nonzero(mask)[0]
    if len(ys) == 0:
        return np.zeros_like(mask)
    return mask & (np.arange(mask.shape[0])[:, None] <= int(ys.min()) + rows - 1)


def root_cols(mask, cols):
    """Leftmost columns of a rightward tail. That edge is the socket."""
    xs = np.nonzero(mask)[1]
    if len(xs) == 0:
        return np.zeros_like(mask)
    return mask & (np.arange(mask.shape[1])[None, :] <= int(xs.min()) + cols - 1)


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

    # Ankle and tail-root copies stay on the torso so a lift does not open a hole.
    cushion = (
        top_rows(placed["front"], 3)
        | top_rows(placed["mid"], 3)
        | top_rows(placed["rear"], 3)
        | root_cols(placed["tail"], 3)
    )
    layers = {}
    for name in ("body", "front", "mid", "rear", "tail", "jaw"):
        lyr = np.zeros_like(canvas)
        lyr[placed[name]] = canvas[placed[name]]
        layers[name] = lyr
    layers["body"][cushion] = canvas[cushion]

    txs = np.nonzero(placed["tail"])[1]
    root_x = int(txs.min())
    sel = placed["tail"][:, root_x:root_x + 2]
    rys, rxs = np.nonzero(sel)
    tail_pivot = (float(rxs.mean() + root_x), float(rys.mean()))
    bys, bxs = np.nonzero(canvas[..., 3] >= 128)
    sole = (float(np.median(bxs[bys >= bys.max() - 1])), float(bys.max()))
    center = (float(bxs.mean()), float(bys.mean()))
    jys, jxs = np.nonzero(placed["jaw"])
    if len(jxs):
        # Front of the red mouth, on the snout side (left before any flip).
        mouth = (float(jxs.min()), float(np.median(jys)))
    else:
        mouth = (float(dst_x), float(dst_y + sh // 2))

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        tail_pivot = (CELL_W - 1 - tail_pivot[0], tail_pivot[1])
        sole = (CELL_W - 1 - sole[0], sole[1])
        center = (CELL_W - 1 - center[0], center[1])
        mouth = (CELL_W - 1 - mouth[0], mouth[1])
        canvas = canvas[:, ::-1].copy()
    facing = 1.0 if flip else -1.0
    return layers, canvas, {"tail": tail_pivot, "sole": sole, "center": center, "mouth": mouth}, facing


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        bdy=0, lift_f=0, lift_m=0, lift_r=0, step_f=0, step_m=0, step_r=0,
        tail=0.0, jaw=0, spin=0.0, hurt=0.0, knock=0.0,
        rings=0, ring=0, burst=0, sparks=0, flash=0,
    )
    if action == "idle":
        p["bdy"] = -1 if wave(f, n, 1.0) > 0.45 else 0
        p["tail"] = wave(f, n, 8.0)
    elif action == "walk":
        # Two plants per loop. Far front + the one hind paw, then the near front.
        s = math.sin(2 * math.pi * 2 * f / n)
        side = int(round(5 * max(0.0, s)))
        near = int(round(5 * max(0.0, -s)))
        p["lift_f"] = side
        p["lift_r"] = side
        p["lift_m"] = near
        p["step_f"] = 2 if side else 0
        p["step_r"] = 2 if side else 0
        p["step_m"] = 2 if near else 0
        p["bdy"] = -1 if (side or near) else 0
        p["tail"] = -10.0 * s
    elif action == "attack":
        p["jaw"] = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["rings"] = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["ring"] = max(0, f - 1)
        if 5 <= f <= 10:
            # Six poses of one turn. Frame 10 is still turned, not a flat line.
            p["spin"] = [60.0, 120.0, 180.0, 240.0, 300.0, 345.0][f - 5]
        if f in (11, 12):
            p["burst"] = 1
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 38.0 * env
        p["knock"] = 4.0 * env
        p["sparks"] = 1 if 2 <= f <= 5 else 0
        p["flash"] = 1 if f == 1 else 0
    return p


def assemble_pose(layers, pivots, p, facing):
    """Torso, tail, three paws, then the red mouth. Rest pose matches the still."""
    body = shift(layers["body"], 0, p["bdy"])
    tail = shift(rotate_nearest(layers["tail"], p["tail"] * facing, pivots["tail"]), 0, p["bdy"])
    front = shift(layers["front"], facing * p["step_f"], -p["lift_f"])
    mid = shift(layers["mid"], facing * p["step_m"], -p["lift_m"])
    rear = shift(layers["rear"], facing * p["step_r"], -p["lift_r"])
    jaw = shift(layers["jaw"], 0, p["bdy"] + p["jaw"])
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for lyr in (body, tail, front, mid, rear):
        paste(img, lyr)
    if p["jaw"]:
        home = shift(layers["jaw"], 0, p["bdy"])
        vacated = (home[..., 3] >= 128) & (jaw[..., 3] < 128) & (img[..., 3] < 128)
        img[vacated, :3] = DRED
        img[vacated, 3] = 255
        # A short wedge of the same red, stuck to the mouth and pointing forward.
        my = int(round(pivots["mouth"][1] + p["bdy"] + p["jaw"]))
        mx = int(round(pivots["mouth"][0]))
        for i in range(1, int(p["jaw"]) + 1):
            put_empty(img, mx + facing * i, my, RED)
    paste(img, jaw)
    return img


def roar_rings(out, mouth, facing, age, strength):
    mx, my = mouth
    cols = (YELLOW, WHITE, GOLD)
    for i in range(strength):
        rad = 12 + age * 2 + i * 4
        col = cols[i % 3]
        for deg in range(-36, 37, 12):
            ang = math.radians(deg)
            put_empty(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.55, col)
    return out


def impact_burst(out, center):
    cx, cy = center
    cols = (YELLOW, WHITE, GOLD)
    for k in range(6):
        ang = k * math.pi / 3
        put_empty(out, cx + 5 * math.cos(ang), cy + 5 * math.sin(ang), cols[k % 3])
        put_empty(out, cx + 8 * math.cos(ang), cy + 8 * math.sin(ang), WHITE)
    return out


def sparks(out, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, _y1 = bb
    y = y0 + 3
    x = x0 + 8 if facing < 0 else x1 - 8
    put_empty(out, x, y, WHITE)
    put_empty(out, x + facing * 2, y + 2, YELLOW)
    put_empty(out, x - facing * 3, y + 1, GOLD)
    return out


def flash_white(img):
    out = img.copy()
    m = out[..., 3] >= 128
    out[m, :3] = WHITE
    return out


def neutral_bitmap(layers):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in ("body", "tail", "front", "mid", "rear", "jaw"):
        paste(img, layers[name])
    return img


def synth(rig, action, f, palette):
    layers, _canvas, pivots, facing = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = (p["spin"] % 360.0) > 0.5
    recoiling = p["hurt"] > 0.45
    if spinning or recoiling:
        # One bitmap. A |cos| squash would turn this body into a line.
        base = neutral_bitmap(layers)
        if spinning:
            img = rotate_nearest(base, p["spin"], pivots["center"])
        else:
            # Left-facing (facing < 0) leans the snout up and backward.
            img = rotate_nearest(base, -facing * p["hurt"], pivots["sole"])
            img = shift(img, -facing * p["knock"], 0)
        img = fill_pinholes(img)
        img = despeckle(img)
    else:
        img = assemble_pose(layers, pivots, p, facing)
    img = snap(img, palette)
    if p["flash"]:
        img = flash_white(img)
    if p["rings"] and not spinning:
        mouth = (pivots["mouth"][0], pivots["mouth"][1] + p["jaw"])
        img = roar_rings(img, mouth, facing, p["ring"], p["rings"])
    if p["burst"]:
        bb = body_box(img)
        if bb:
            tip = bb[2] + 6 if facing > 0 else bb[0] - 6
            img = impact_burst(img, (tip, (bb[1] + bb[3]) // 2))
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
    """South row, every frame, scaled so the three paws and the spin can be read."""
    scale = 3
    x0, x1 = 36, 146
    y0, y1 = 78, 164
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
    """West row (unflipped, faces left), every frame, cropped."""
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
        "dex": 404, "form": "normal", "variant": "luxio", "gen": "gen4",
        "dir": "monsterSprite/gen4/0404_luxio/normal",
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
        if key not in text:
            text = text.replace(prefix + "{", prefix + "{" + key + blob + ",", 1)
        else:
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
                raise SystemExit("queue has no 0404_luxio")
            tmp = QUEUE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def attack_report(sheets, sprite_h):
    attack_s = []
    base_n = int((sheets["attack"][:CELL_H, :CELL_W, 3] >= 128).sum())
    rest_w = 0
    rest_h = 0
    for f in range(FRAMES["attack"]):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        c = tile[..., 3] >= 128
        n = int(c.sum())
        ys, xs = np.nonzero(c)
        bw = int(xs.max() - xs.min() + 1) if len(xs) else 0
        bh = int(ys.max() - ys.min() + 1) if len(ys) else 0
        if f == 0:
            rest_w, rest_h = bw, bh
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
    thin = [
        row for row in attack_s
        if 5 <= row["f"] <= 10 and (row["w"] < rest_w * 0.75 or row["h"] < max(rest_h, sprite_h) * 0.75)
    ]
    weak = [row for row in attack_s if 5 <= row["f"] <= 10 and row["main"] < 0.90]
    return attack_s, ratios, bad_ratio, thin, weak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    opaque = front[..., 3] >= 128
    exclusive = ["body", "front", "mid", "rear", "tail", "jaw"]
    covered = np.zeros_like(opaque)
    overlap = 0
    for name in exclusive:
        overlap += int((covered & masks[name]).sum())
        covered |= masks[name]
    counts = {k: int(masks[k].sum()) for k in exclusive}
    counts["feet"] = 3
    counts["missing"] = int((opaque & ~covered).sum())
    counts["overlap"] = overlap
    print("parts", counts, "palette", len(palette), flush=True)
    if counts["missing"] or overlap:
        raise SystemExit("segmentation does not partition the still")
    for paw in ("front", "mid", "rear"):
        if components(masks[paw]) != [counts[paw]]:
            raise SystemExit(f"{paw} paw is not one piece")
        if counts[paw] < 8:
            raise SystemExit(f"{paw} paw collapsed")
    if components(masks["tail"]) != [counts["tail"]]:
        raise SystemExit("tail is not one piece")
    white = opaque & np.all(front[..., :3] == np.array(WHITE, np.uint8), axis=-1)
    eye_x = float(np.nonzero(white)[1].mean())
    snout_x = int(np.nonzero(opaque & (np.arange(96)[:, None] >= 40) & (np.arange(96)[:, None] <= 48))[1].min())
    print("facing snout_x", snout_x, "eye_x", round(eye_x, 1), "feet", 3, flush=True)
    if snout_x >= eye_x:
        raise SystemExit("expected a left-facing still")

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}

    rest = synth(rigs["W"], "idle", 0, palette)
    canvas = rigs["W"][1]
    differ = np.any(rest != canvas, axis=2) & ((rest[..., 3] >= 128) | (canvas[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()):
        raise SystemExit("idle frame 0 does not match the still")
    for act in ("walk", "attack", "hurt"):
        fr = synth(rigs["W"], act, 0, palette)
        ed = np.any(fr != canvas, axis=2) & ((fr[..., 3] >= 128) | (canvas[..., 3] >= 128))
        if int(ed.sum()):
            raise SystemExit(f"{act} frame 0 does not match the still")
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
        "body": (70, 90, 140), "front": (220, 60, 60), "mid": (240, 180, 40),
        "rear": (40, 180, 80), "tail": (160, 80, 220), "jaw": (255, 255, 255),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    broken = []
    for f in range(5, 11):
        fr = synth(rigs["W"], "attack", f, palette)
        sizes = components(fr[..., 3] >= 128)
        if not sizes or sizes[0] < 1100:
            broken.append(("attack", f, sizes[:4]))
    for f in (1, 3, 5, 6, 9):
        fr = synth(rigs["W"], "hurt", f, palette)
        sizes = components(fr[..., 3] >= 128)
        if not sizes or sizes[0] < 1100:
            broken.append(("hurt", f, sizes[:4]))
    print("piece check", broken if broken else "ok", flush=True)
    if broken:
        raise SystemExit("spin or recoil broke the body")

    # South attack ratios before the long render, so a bad spin can be rebuilt.
    south_tiles = [synth(rigs["S"], "attack", f, palette) for f in range(16)]
    dummy = {"attack": np.zeros((CELL_H, 16 * CELL_W, 4), np.uint8)}
    for f, tile in enumerate(south_tiles):
        dummy["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W] = tile
    _rows, ratios, bad_ratio, thin, weak = attack_report(dummy, 48)
    if bad_ratio or thin or weak:
        print("QA FAIL early", "bad_ratio", bad_ratio, "thin", thin, "weak", weak, flush=True)
        raise SystemExit(1)

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "W", f) for f in (0, 2, 4, 5, 7, 9, 10, 11, 13, 15)]
        want += [("hurt", "W", f) for f in (0, 3, 5, 8, 11)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        cols = 8
        x0, x1, y0, y1 = 40, 140, 88, 160
        zw, zh = (x1 - x0) * 3, (y1 - y0) * 3
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * zh, cols * zw, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            up = np.repeat(np.repeat(tile[y0:y1, x0:x1], 3, 0), 3, 1)
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
    # South idle 0 is the mirrored still. Height is the opaque span.
    # The game reads h from the south idle pose. Use the unflipped height too.
    west = sheets["idle"][6 * CELL_H:7 * CELL_H, :CELL_W]
    na = west[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = [int(v) for v in np.unique(master[..., 3])]
    bad_alpha = [v for v in alphas if v not in (0, 255)]
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{d}:{f}")
    attack_s, ratios, bad_ratio, thin, weak = attack_report(sheets, sprite_h)
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "attack_S_opaque": attack_s,
        "ratio_min": min(ratios) if ratios else 0,
        "ratio_max": max(ratios) if ratios else 0,
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
        "generator": "forge_luxio_actions", "version": VERSION, "key": KEY,
        "dex": 404, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "source": "front.png 96x96", "gifs_untouched": True,
        "facing": "left", "feet": 3, "archetype": "quadruped_side_three_visible_paws",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = not off and not bad_alpha and not clip and not bad_ratio and not thin and not weak and sprite_h > 0
    if not ok:
        print("QA FAIL", "off", len(off), "alpha", bad_alpha, "clip", clip,
              "bad_ratio", bad_ratio, "thin", thin, "weak", weak, flush=True)
        raise SystemExit(1)
    if args.publish and not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        publish_dir8(sprite_h)
        update_queue("done")
        print("published", KEY, VERSION, "h", sprite_h, flush=True)
    else:
        print("PASS unpublished", KEY, "h", sprite_h, "ratios",
              f"{min(ratios):.4f}", f"{max(ratios):.4f}", flush=True)


if __name__ == "__main__":
    main()
