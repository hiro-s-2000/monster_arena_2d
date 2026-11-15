#!/usr/bin/env python3
"""Glameow (0431) 8-direction action sheet from this form's 96x96 front.png.

Measured on normal/front.png (opaque 1367, bbox x18-72 y19-76, 55x58, 15 colours,
alpha already 0 or 255). One main silhouette plus three whisker strokes and five
tiny gap specks. The head is the left edge (whiskers at x18, yellow eye at x27)
and the coil tail reaches x72, so the still faces left.
Unflipped dirs are N NW W SW. Flipped dirs are S SE E NE, and E faces screen-right.

Two paws touch the ground. The coil is a tail, not a foot. There is no third leg
and no mouth opening (yellow eye, blue eye, tan nose, white muzzle only).

  upright ear  x25-36 y27-40   the pointed ear, pink inside, white tip
  side ear     stays on the head (pink triangle). Not a second animal.
  front paw    x33-41 y58-76   white toes, the only pixels on y76
  rear paw     x42-52 y64-75   pale toes, ends at y75
  tail coil    x48-72 y19-40   upper spring, rooted on the lower loops
  whiskers     three dark strokes at y45, y46, y50, disconnected

  idle   : 1px breath. Upper coil sways. Feet stay planted. Frame 0 is the still.
  walk   : the two measured paws alternate. The tail is not a third foot.
  attack : 0 onset (still), 1-4 roar (ear, whiskers, rings, no invented mouth),
           5-10 one-piece in-plane 360, 11-12 impact, 13-15 rest.
  hurt   : one-piece lean about the sole, 38 degrees max, no belly flip.

Palette-locked. front/back png and gif are not written.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_glameow_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen4/0431_glameow/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0431_glameow/normal/agent-sprite-forge")
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
VERSION = "glameow_v2"
KEY = "0431_glameow"

INK = (0, 0, 0)
SLATE = (65, 65, 82)
MID = (106, 115, 131)
FUR = (139, 156, 189)
WHITE = (255, 255, 255)
PINK = (255, 123, 156)
PINK_D = (213, 74, 131)
YELLOW = (255, 222, 115)
PALE = (205, 205, 222)


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


def colour_mask(img, colour):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(colour, np.uint8), axis=-1)


def segment(img):
    """Cuts measured on this still. Two paws, one coil, the upright ear, three whiskers."""
    h, w = img.shape[:2]
    opaque = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    pink = colour_mask(img, PINK) | colour_mask(img, PINK_D)
    white = colour_mask(img, WHITE)

    # Upper spring only. The lower loops stay on the torso so the coil does not tear.
    tip = np.zeros_like(opaque)
    tip[19, 49] = True
    tail_allow = opaque & (yy <= 40) & (xx >= 48) & ~((yy >= 37) & (xx <= 55))
    tail = flood(tip, tail_allow)

    # Near foreleg. Lowest pixels are the black toe outline on y76.
    front_seed = np.zeros_like(opaque)
    front_seed[76, 35] = True
    front_seed[76, 36] = True
    front = flood(front_seed, opaque & (yy >= 58) & (xx <= 41))

    # Far hind paw. Pale toes at x43-45, not the tail.
    rear_seed = np.zeros_like(opaque)
    rear_seed[74, 43:46] = True
    rear_seed[75, 43:46] = True
    rear = flood(rear_seed, opaque & (yy >= 64) & (xx >= 41) & (xx <= 52) & ~front)

    # Pointed ear above the skull. The face (y>=41) stays on the body.
    ear_seed = (pink & (xx <= 34) & (yy <= 40)) | (white & (yy <= 33) & (xx >= 28) & (xx <= 36))
    ear = flood(ear_seed, opaque & (yy <= 40) & (xx <= 36) & (xx >= 24))

    whisk = np.zeros_like(opaque)
    whisk[45, 18:21] = True
    whisk[46, 21:24] = True
    whisk[50, 19:22] = True
    whisk &= opaque

    front = front & ~tail & ~ear & ~whisk
    rear = rear & ~tail & ~ear & ~whisk & ~front
    ear = ear & ~tail & ~whisk
    body = opaque & ~front & ~rear & ~tail & ~ear & ~whisk
    return {
        "body": body, "front": front, "rear": rear, "tail": tail,
        "ear": ear, "whisk": whisk,
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
    """4x nearest. Width is never scaled by |cos|. 0 and 360 return the same pixels."""
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
    pad = np.pad(a, 1)
    n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
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


def top_rows(mask, rows):
    ys = np.nonzero(mask)[0]
    if len(ys) == 0:
        return np.zeros_like(mask)
    return mask & (np.arange(mask.shape[0])[:, None] <= int(ys.min()) + rows - 1)


def bottom_rows(mask, rows):
    ys = np.nonzero(mask)[0]
    if len(ys) == 0:
        return np.zeros_like(mask)
    return mask & (np.arange(mask.shape[0])[:, None] >= int(ys.max()) - (rows - 1))


def row_pivot(mask, rows, bottom=True):
    band = bottom_rows(mask, rows) if bottom else top_rows(mask, rows)
    ys, xs = np.nonzero(band)
    if len(ys) == 0:
        ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


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

    # Joint copies stay on the torso. At rest the real part is drawn on top.
    cushion = (
        top_rows(placed["front"], 3)
        | top_rows(placed["rear"], 3)
        | bottom_rows(placed["tail"], 2)
        | bottom_rows(placed["ear"], 2)
    )
    names = ("body", "front", "rear", "tail", "ear", "whisk")
    layers = {}
    for name in names:
        lyr = np.zeros_like(canvas)
        lyr[placed[name]] = canvas[placed[name]]
        layers[name] = lyr
    layers["body"][cushion] = canvas[cushion]

    tail_pivot = row_pivot(placed["tail"], 2, bottom=True)
    ear_pivot = row_pivot(placed["ear"], 2, bottom=True)
    bys, bxs = np.nonzero(canvas[..., 3] >= 128)
    sole = (float(np.median(bxs[bys >= bys.max() - 1])), float(bys.max()))
    center = (float(bxs.mean()), float(bys.mean()))
    wys, wxs = np.nonzero(placed["whisk"])
    mouth = (float(wxs.min()), float(np.median(wys)))

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        tail_pivot = (CELL_W - 1 - tail_pivot[0], tail_pivot[1])
        ear_pivot = (CELL_W - 1 - ear_pivot[0], ear_pivot[1])
        sole = (CELL_W - 1 - sole[0], sole[1])
        center = (CELL_W - 1 - center[0], center[1])
        mouth = (CELL_W - 1 - mouth[0], mouth[1])
        canvas = canvas[:, ::-1].copy()
    facing = 1.0 if flip else -1.0
    pivots = {"tail": tail_pivot, "ear": ear_pivot, "sole": sole, "center": center, "mouth": mouth}
    return layers, canvas, pivots, facing


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        bdy=0, lift_f=0, lift_r=0, step_f=0, step_r=0,
        tail=0.0, ear=0.0, flare=0, spin=0.0, hurt=0.0, knock=0.0,
        rings=0, ring=0, lines=0, burst=0, sparks=0, lunge=0,
    )
    if action == "idle":
        breath = wave(f, n, 1.0)
        p["bdy"] = -1 if breath > 0.45 else 0
        p["tail"] = wave(f, n, 7.0)
    elif action == "walk":
        # Two steps. Front paw, then the rear paw. Both planted on frame 0.
        s = math.sin(2 * math.pi * 2 * f / n)
        p["bdy"] = -1 if abs(s) > 0.65 else 0
        p["lift_f"] = int(round(2 * max(0.0, s)))
        p["lift_r"] = int(round(2 * max(0.0, -s)))
        p["step_f"] = 1 if p["lift_f"] else 0
        p["step_r"] = 1 if p["lift_r"] else 0
        p["tail"] = wave(f, n, 10.0, cycles=2.0)
    elif action == "attack":
        # 0 still onset, 1-4 roar, 5-10 one-piece spin, 11-12 impact, 13-15 rest.
        p["ear"] = [0, 10, 16, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["flare"] = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["rings"] = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["ring"] = max(0, f - 1)
        p["tail"] = [0, 4, 6, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        if 5 <= f <= 10:
            p["spin"] = 60.0 * (f - 4)
        if 6 <= f <= 9:
            p["lines"] = 2
        if f in (11, 12):
            p["burst"] = 1
            p["lunge"] = 1
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 38.0 * env
        p["knock"] = 3.0 * env
        p["sparks"] = 1 if 2 <= f <= 5 else 0
    return p


def assemble_pose(layers, pivots, p, facing):
    """Coil, torso, hind paw, forepaw, ear, whiskers. Rest pose matches the still."""
    bdy = p["bdy"]
    body = shift(layers["body"], 0, bdy)
    tail = shift(rotate_nearest(layers["tail"], p["tail"] * facing, pivots["tail"]), 0, bdy)
    ear = shift(rotate_nearest(layers["ear"], -p["ear"] * facing, pivots["ear"]), 0, bdy)
    front = shift(layers["front"], facing * p["step_f"], -p["lift_f"])
    rear = shift(layers["rear"], facing * p["step_r"], -p["lift_r"])
    whisk = shift(layers["whisk"], facing * p["flare"], bdy)
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(img, body)
    for lyr in (tail, rear, front, ear, whisk):
        paste(img, lyr)
    return img


def roar_rings(out, mouth, facing, age, strength):
    """A few rings in front of the muzzle. No mouth wedge — the still has none."""
    mx, my = mouth
    cols = (WHITE, PINK, INK)
    for i in range(strength):
        rad = 5 + age * 2 + i * 3
        col = cols[i % 3]
        for deg in (-32, 0, 32):
            ang = math.radians(deg)
            put_empty(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.5, col)
    return out


def speed_lines(out, facing, n):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rear = x1 + 4 if facing < 0 else x0 - 4
    step = -facing
    span = max(8, (y1 - y0) // 3)
    for k in range(n):
        y = y0 + span + k * 6
        for i in range(3):
            put_empty(out, rear + step * (i + 1), y, WHITE if i == 0 else INK)
    return out


def impact_burst(out, center):
    cx, cy = center
    cols = (YELLOW, WHITE, PINK, INK)
    for k in range(4):
        ang = k * math.pi / 2 + 0.4
        put_empty(out, cx + 4 * math.cos(ang), cy + 4 * math.sin(ang), cols[k])
        put_empty(out, cx + 7 * math.cos(ang), cy + 7 * math.sin(ang), WHITE if k % 2 == 0 else PINK)
    return out


def sparks(out, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    y = y0 + 6
    x = x0 + 6 if facing < 0 else x1 - 6
    put_empty(out, x, y, WHITE)
    put_empty(out, x + facing * 3, y + 2, YELLOW)
    put_empty(out, x - facing * 2, y + 3, PINK)
    return out


def neutral_bitmap(layers):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(img, layers["body"])
    for name in ("tail", "rear", "front", "ear", "whisk"):
        paste(img, layers[name])
    return img


def synth(rig, action, f, palette):
    layers, _canvas, pivots, facing = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = (p["spin"] % 360.0) > 0.5
    recoiling = p["hurt"] > 0.45
    if spinning or recoiling:
        # One bitmap. A |cos| squash turns the coil and the legs into scraps.
        base = neutral_bitmap(layers)
        if spinning:
            img = rotate_nearest(base, p["spin"], pivots["center"])
        else:
            img = rotate_nearest(base, -facing * p["hurt"], pivots["sole"])
            img = shift(img, -facing * p["knock"], 0)
        img = despeckle(img)
    elif p["lunge"]:
        img = shift(neutral_bitmap(layers), facing * p["lunge"], 0)
    else:
        img = assemble_pose(layers, pivots, p, facing)
    img = snap(img, palette)
    if p["rings"] and not spinning and not recoiling:
        mouth = (pivots["mouth"][0] + facing * p["flare"], pivots["mouth"][1] + p["bdy"])
        img = roar_rings(img, mouth, facing, p["ring"], p["rings"])
    if p["lines"]:
        img = speed_lines(img, facing, p["lines"])
    if p["burst"]:
        bb = body_box(img)
        if bb:
            tip = bb[2] + 4 if facing > 0 else bb[0] - 4
            img = impact_burst(img, (tip, bb[1] + (bb[3] - bb[1]) // 3))
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
    return master


def preview_image(sheets):
    """South row, every frame, scaled so the cat can be read."""
    scale = 2
    x0, x1 = 36, 148
    y0, y1 = 64, 160
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
            yy0 = max(0, int(ys.min()) - pad)
            yy1 = min(CELL_H, int(ys.max()) + 1 + pad)
            xx0 = max(0, int(xs.min()) - pad)
            xx1 = min(CELL_W, int(xs.max()) + 1 + pad)
            crop = tile[yy0:yy1, xx0:xx1]
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
        "dex": 431, "form": "normal", "variant": "glameow", "gen": "gen4",
        "dir": "monsterSprite/gen4/0431_glameow/normal",
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
            raise SystemExit("dir8.js has no 0431_glameow")
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
        json.loads(text[start:end])
        text = text[:start] + blob + text[end:]
        if text.count(key) != 1:
            raise SystemExit("dir8 replace duplicated or dropped 0431_glameow")
        if '"0430_honchkrow":' not in text or '"0432_purugly":' not in text:
            raise SystemExit("dir8 neighbour key missing after replace")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(text)
        data = json.loads(text[len(prefix):].rstrip().rstrip(";"))
        if data[KEY]["v"] != VERSION or data[KEY]["an"] != [16, 16, 16, 12]:
            raise SystemExit("dir8 entry did not parse back")
        if data[KEY]["layout"] != "actions_8dir" or data[KEY]["raw"] != 1:
            raise SystemExit("dir8 layout or raw mismatch")
        if "0430_honchkrow" not in data or "0432_purugly" not in data:
            raise SystemExit("dir8 neighbours dropped")
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
                raise SystemExit("queue has no 0431_glameow")
            tmp = QUEUE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def assert_claim():
    import os as _rebuild_os
    if _rebuild_os.environ.get("GEN4_ATTACK_REBUILD") == "1":
        return
    data = json.loads(QUEUE.read_text(encoding="utf-8"))
    for item in data["items"]:
        if item.get("id") != KEY:
            continue
        if item.get("status") != "in_progress" or item.get("claim") != "parallel-batch":
            raise SystemExit(
                f"queue is {item.get('status')} claim={item.get('claim')}; not drawing"
            )
        return
    raise SystemExit("queue has no 0431_glameow")


def south_attack_report(tiles):
    base = int((tiles[0][..., 3] >= 128).sum())
    rows = []
    for f, tile in enumerate(tiles):
        c = tile[..., 3] >= 128
        n = int(c.sum())
        ys, xs = np.nonzero(c)
        bw = int(xs.max() - xs.min() + 1) if len(xs) else 0
        bh = int(ys.max() - ys.min() + 1) if len(ys) else 0
        sizes = components(c)
        main = sizes[0] / n if sizes and n else 0
        rows.append({
            "f": f, "opaque": n, "ratio": (n / base) if base else 0,
            "w": bw, "h": bh, "main": main,
        })
    return rows


def qa_fail(rows, rest_w, rest_h):
    bad = []
    thin = []
    weak = []
    for row in rows:
        if row["ratio"] < 0.90 or row["ratio"] > 1.08:
            bad.append(row["f"])
        if 5 <= row["f"] <= 10:
            if row["w"] < rest_w * 0.55 or row["h"] < rest_h * 0.55:
                thin.append(row["f"])
            if row["main"] < 0.90:
                weak.append(row["f"])
    return bad, thin, weak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    assert_claim()
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    opaque = front[..., 3] >= 128
    exclusive = ["body", "front", "rear", "tail", "ear", "whisk"]
    covered = np.zeros_like(opaque)
    overlap = 0
    for name in exclusive:
        overlap += int((covered & masks[name]).sum())
        covered |= masks[name]
    counts = {k: int(masks[k].sum()) for k in exclusive}
    counts["missing"] = int((opaque & ~covered).sum())
    counts["overlap"] = overlap
    print("parts", counts, "palette", len(palette), flush=True)
    if counts["missing"] or overlap:
        raise SystemExit("segmentation does not partition the still")
    for name in ("front", "rear", "tail", "ear"):
        sizes = components(masks[name])
        if sizes != [counts[name]]:
            raise SystemExit(f"{name} is not one piece: {sizes}")
    if components(masks["whisk"]) != [3, 3, 3]:
        raise SystemExit(f"whiskers changed: {components(masks['whisk'])}")
    if counts["front"] < 40 or counts["rear"] < 40:
        raise SystemExit("a paw mask collapsed")
    tys = np.nonzero(masks["tail"])[0]
    fys = np.nonzero(masks["front"])[0]
    rys = np.nonzero(masks["rear"])[0]
    if int(tys.max()) >= 50 or int(fys.min()) < 50 or int(rys.min()) < 50:
        raise SystemExit("tail and feet ranges crossed")
    if int((masks["tail"] & (np.arange(front.shape[0])[:, None] >= 74)).sum()):
        raise SystemExit("tail was treated as a foot")
    eye = colour_mask(front, (41, 115, 213)) | (
        colour_mask(front, YELLOW) & (np.arange(front.shape[0])[:, None] >= 45)
        & (np.arange(front.shape[1])[None, :] <= 30)
    )
    if not np.any(eye & masks["body"]):
        raise SystemExit("eyes left the head")
    # Facing: whiskers and the yellow eye are left of the coil.
    eye_x = float(np.nonzero(eye)[1].mean())
    tail_x = float(np.nonzero(masks["tail"])[1].mean())
    print("facing eye_x", round(eye_x, 1), "tail_x", round(tail_x, 1), "paws", 2, flush=True)
    if eye_x >= tail_x:
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
        dlt = np.any(fr != canvas, axis=2) & ((fr[..., 3] >= 128) | (canvas[..., 3] >= 128))
        if int(dlt.sum()):
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
        "body": (90, 140, 220), "front": (220, 60, 60), "rear": (40, 180, 90),
        "tail": (230, 120, 40), "ear": (255, 220, 40), "whisk": (0, 220, 220),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    atk = [synth(rigs["S"], "attack", f, palette) for f in range(16)]
    rows = south_attack_report(atk)
    ratios = [row["ratio"] for row in rows]
    print("south attack opaque", [row["opaque"] for row in rows], flush=True)
    print("south attack ratios", " ".join(f"{r:.4f}" for r in ratios), flush=True)
    print("min", round(min(ratios), 4), "max", round(max(ratios), 4), flush=True)
    rest_w = rows[0]["w"]
    rest_h = rows[0]["h"]
    bad, thin, weak = qa_fail(rows, rest_w, rest_h)
    if bad or thin or weak:
        print("early QA FAIL", "bad", bad, "thin", thin, "weak", weak, flush=True)
        if args.publish:
            update_queue("needs_fix")
        raise SystemExit(1)

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "W", f) for f in (0, 2, 4, 6, 8, 11)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        cols = 8
        x0, x1, y0, y1 = 36, 148, 64, 160
        zw, zh = (x1 - x0) * 2, (y1 - y0) * 2
        nrows = math.ceil(len(tiles) / cols)
        sheet = np.full((nrows * zh, cols * zw, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            up = np.repeat(np.repeat(tile[y0:y1, x0:x1], 2, 0), 2, 1)
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
                    sizes = components(c)
                    main = sizes[0] / n if sizes and n else 0
                    attack_s.append({
                        "f": f, "opaque": n,
                        "ratio": round(n / base_n, 4) if base_n else 0,
                        "w": bw, "h": bh, "main": round(main, 4),
                    })
    ratios = [row["ratio"] for row in attack_s]
    print("south attack opaque", [row["opaque"] for row in attack_s], flush=True)
    print("south attack ratios", " ".join(f"{r:.4f}" for r in ratios), flush=True)
    print("min", round(min(ratios), 4), "max", round(max(ratios), 4), flush=True)
    bad_ratio = [row for row in attack_s if row["ratio"] < 0.90 or row["ratio"] > 1.08]
    thin = [row for row in attack_s if 5 <= row["f"] <= 10 and (row["w"] < rest_w * 0.55 or row["h"] < rest_h * 0.55)]
    weak = [row for row in attack_s if 5 <= row["f"] <= 10 and row["main"] < 0.90]
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "attack_S_opaque": attack_s,
        "ratio_min": round(min(ratios), 4), "ratio_max": round(max(ratios), 4),
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        qa = work / "qa"
        qa.mkdir(exist_ok=True)
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                continue
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
        "generator": "forge_glameow_actions", "version": VERSION, "key": KEY,
        "dex": 431, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "source": "front.png 96x96", "gifs_untouched": True,
        "facing": "left", "paws": 2, "mouth": False, "tails": 1,
        "archetype": "quadruped_two_visible_paws_coil_tail",
        "south_attack_ratio_min": round(min(ratios), 4),
        "south_attack_ratio_max": round(max(ratios), 4),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = not off and not bad_alpha and not clip and not bad_ratio and not thin and not weak and sprite_h > 0
    if not ok:
        print("QA FAIL", "off", len(off), "alpha", bad_alpha, "clip", clip,
              "bad_ratio", bad_ratio, "thin", thin, "weak", weak, flush=True)
        if args.publish:
            update_queue("needs_fix")
        raise SystemExit(1)
    if args.publish and not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
        publish_dir8(sprite_h)
        update_queue("done")
        print("published", KEY, VERSION, "h", sprite_h, flush=True)
    elif args.publish and args.no_sync:
        publish_dir8(sprite_h)
        update_queue("done")
        print("published without sync", KEY, VERSION, "h", sprite_h, flush=True)
    else:
        print("QA numeric pass; dir8 not edited", flush=True)


if __name__ == "__main__":
    main()
