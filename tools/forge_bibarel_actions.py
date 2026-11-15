#!/usr/bin/env python3
"""Bibarel (0400_bibarel) 8-direction action sheet from the 96x96 front still.

Archetype: heavy beaver, front three-quarter. One round torso, two drawn paws,
one paddle tail on the right, and one incisor muzzle (the existing buck teeth).
The nose sits between the eyes and the tail is the rear, so the still faces left.
Unflipped: N NW W SW. Flipped: S SE E NE, so E faces screen-right.

  idle   : feet planted. Body breathes 1px. The single tail wags.
  walk   : the two existing paws alternate. Tail counters. No extra legs.
  attack : frames 0-4 drop the existing muzzle and add rings. Frames 5-10 are an
           in-place 360 spin of the whole still (one piece, no squash). 11-12
           impact dots. 13-15 settle.
  hurt   : whole-body recoil, 38 degrees max, leaning toward the tail.

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.
"""

import argparse
import fcntl
import hashlib
import json
import math
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
NORMAL = ROOT / "monsterSprite/gen4/0400_bibarel/normal"
OUT_DIR = NORMAL / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0400_bibarel/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Left-facing still. These four are the horizontal mirror so E points screen-right.
FLIP_DIRS = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "bibarel_v2"
KEY = "0400_bibarel"

INK = (16, 16, 16)
WHITE = (255, 255, 255)
CREAM = (230, 213, 156)
TAN = (189, 131, 74)


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
    return out


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def bbox_of(mask):
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
    """4x nearest. 0 and 360 degrees return the same pixels."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    up = np.repeat(np.repeat(layer, 4, axis=0), 4, axis=1)
    px, py = pivot[0] * 4 + 1.5, pivot[1] * 4 + 1.5
    rad = math.radians(ang)
    co, s = math.cos(rad), math.sin(rad)
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = xx - px, yy - py
    src_x = np.rint(px + co * rx + s * ry).astype(np.int32)
    src_y = np.rint(py - s * rx + co * ry).astype(np.int32)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    out[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    return out[1::4, 1::4].copy()


def despeckle(img):
    """Drop only pixels whose four neighbours are all empty."""
    out = img.copy()
    opaque = out[..., 3] >= 128
    p = np.pad(opaque, 1)
    n = (
        p[:-2, 1:-1].astype(np.int16)
        + p[2:, 1:-1]
        + p[1:-1, :-2]
        + p[1:-1, 2:]
    )
    out[opaque & (n == 0)] = 0
    return out


def fill_pinholes(img):
    """Fill enclosed holes of 1 or 2 pixels so a spin does not punch the body."""
    opaque = img[..., 3] >= 128
    trans = ~opaque
    h, w = trans.shape
    seen = np.zeros_like(trans)
    from collections import deque

    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if trans[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if trans[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and trans[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    enc = trans & ~seen
    if not np.any(enc):
        return img
    # Only tiny pinholes. The tail notch and the gap between the paws stay open.
    seen_e = np.zeros_like(enc)
    out = img.copy()
    ys, xs = np.nonzero(enc)
    src_y, src_x = np.nonzero(opaque)
    src = np.stack([src_y, src_x], axis=1).astype(np.int32)
    cols = img[src_y, src_x]
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen_e[y, x]:
            continue
        q = deque([(y, x)])
        seen_e[y, x] = True
        comp = [(y, x)]
        while q:
            cy, cx = q.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and enc[ny, nx] and not seen_e[ny, nx]:
                    seen_e[ny, nx] = True
                    q.append((ny, nx))
                    comp.append((ny, nx))
        if len(comp) > 2:
            continue
        for cy, cx in comp:
            d = (src[:, 0] - cy) ** 2 + (src[:, 1] - cx) ** 2
            out[cy, cx] = cols[int(d.argmin())]
            out[cy, cx, 3] = 255
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
    step = 400
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None].astype(np.int32) - pts[None, :, 0]) ** 2
        d += (tx[sl, None].astype(np.int32) - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def segment(img):
    """Partition the still. One tail, two paws, one incisor muzzle, the torso."""
    opaque = img[..., 3] >= 128
    rgb = img[..., :3]
    h, w = opaque.shape
    yy, xx = np.mgrid[0:h, 0:w]

    tail = np.zeros_like(opaque)
    for y in range(35, 48):
        xs = [int(v) for v in np.nonzero(opaque[y])[0]]
        if not xs:
            continue
        runs = []
        start = prev = xs[0]
        for x in xs[1:]:
            if x == prev + 1:
                prev = x
            else:
                runs.append((start, prev))
                start = prev = x
        runs.append((start, prev))
        if runs[-1][0] >= 54:
            x0, x1 = runs[-1]
            tail[y, x0:x1 + 1] = True
    tail[48:68, 58:] = opaque[48:68, 58:]

    light = ((rgb[..., 0] == 255) & (rgb[..., 1] == 255) & (rgb[..., 2] == 255)) | (
        (rgb[..., 0] == 197) & (rgb[..., 1] == 197) & (rgb[..., 2] == 180)
    )
    core = opaque & light & (yy >= 37) & (yy <= 40) & (xx >= 28) & (xx <= 36)
    black = (rgb[..., 0] == 16) & (rgb[..., 1] == 16) & (rgb[..., 2] == 16)
    dark = (rgb[..., 0] == 57) & (rgb[..., 1] == 49) & (rgb[..., 2] == 41)
    rim_allow = (black | dark) & (yy >= 37) & (yy <= 41) & (xx >= 27) & (xx <= 37)
    muzzle = core | (rim_allow & dilate(core, 1))

    foot_l = opaque & (yy >= 71) & (xx <= 36)
    foot_r = opaque & (yy >= 71) & (xx >= 41)
    parts = (tail, muzzle, foot_l, foot_r)
    overlap = (tail & muzzle) | (tail & foot_l) | (tail & foot_r) | (muzzle & foot_l) | (muzzle & foot_r) | (foot_l & foot_r)
    if np.any(overlap):
        raise SystemExit("bibarel parts overlap")
    torso = opaque & ~tail & ~muzzle & ~foot_l & ~foot_r
    if int((torso | tail | muzzle | foot_l | foot_r).sum()) != int(opaque.sum()):
        raise SystemExit("bibarel parts do not cover the still")
    return {"torso": torso, "tail": tail, "muzzle": muzzle, "foot_l": foot_l, "foot_r": foot_r}


def place_still(img):
    opaque = img[..., 3] >= 128
    ys, xs = np.nonzero(opaque)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    origin = (dst_x - x0, dst_y - y0)
    return canvas, origin


def move_mask(mask, origin):
    ox, oy = origin
    out = np.zeros((CELL_H, CELL_W), bool)
    ys, xs = np.nonzero(mask)
    out[ys + oy, xs + ox] = True
    return out


def build_rig(img, masks):
    placed, origin = place_still(img)
    moved = {name: move_mask(mask, origin) for name, mask in masks.items()}
    layers = {}
    for name, mask in moved.items():
        layer = np.zeros_like(placed)
        layer[mask] = placed[mask]
        layers[name] = layer

    # Mouth behind the existing teeth. Covered while the muzzle is shut.
    layers["torso"][moved["muzzle"], :3] = INK
    layers["torso"][moved["muzzle"], 3] = 255

    feet = moved["foot_l"] | moved["foot_r"]
    ankle = moved["torso"] & dilate(feet, 1)
    hip = moved["tail"] & dilate(moved["torso"], 1)
    socket = np.zeros_like(placed)
    socket[ankle] = placed[ankle]
    hip_fill = nearest_color(placed, hip, moved["torso"])
    socket[hip] = hip_fill[hip]

    tail_ys, tail_xs = np.nonzero(moved["tail"])
    left = tail_xs.min()
    sel = tail_xs <= left + 2
    tail_pivot = (float(tail_xs[sel].mean()), float(tail_ys[sel].mean()))

    opaque = placed[..., 3] >= 128
    ys, xs = np.nonzero(opaque)
    centroid = (float(xs.mean()), float(ys.mean()))
    foot_pivot = (float(xs.mean()), float(ys.max()))
    return {
        "layers": layers,
        "socket": socket,
        "placed": placed,
        "tail_pivot": tail_pivot,
        "centroid": centroid,
        "foot_pivot": foot_pivot,
    }


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2.0 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def step_lift(f, n, phase):
    return max(0.0, wave(f, n, 1.0, phase=phase, cycles=2.0))


def pose(action, f):
    n = FRAMES[action]
    p = {
        "bdy": 0, "tail": 0.0,
        "ldx": 0, "ldy": 0, "rdx": 0, "rdy": 0,
        "mdx": 0, "mdy": 0, "rings": 0, "impact": 0,
        "spin": 0.0, "hurt": 0.0, "hdx": 0, "sweat": 0,
    }
    if action == "idle":
        p["bdy"] = int(round(wave(f, n, 1.4)))
        p["tail"] = wave(f, n, 5.0, phase=0.5)
    elif action == "walk":
        p["bdy"] = int(round(wave(f, n, 1.2, cycles=2.0)))
        p["tail"] = wave(f, n, 6.0, cycles=2.0, phase=0.4)
        ll = step_lift(f, n, 0.0)
        rl = step_lift(f, n, math.pi)
        # Facing left: a lifted paw reaches toward the nose (negative x).
        p["ldy"] = -int(round(4.0 * ll))
        p["ldx"] = -int(round(2.0 * ll))
        p["rdy"] = -int(round(4.0 * rl))
        p["rdx"] = -int(round(2.0 * rl))
    elif action == "attack":
        muzzle_dy = (0, 1, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
        muzzle_dx = (0, -1, -1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        rings = (0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        p["mdy"] = muzzle_dy[f]
        p["mdx"] = muzzle_dx[f]
        p["rings"] = rings[f]
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
        if f in (11, 12):
            p["impact"] = 1
    else:
        # Lean toward the tail. +38 degrees in this rotator swings the head to +x.
        angles = (0, 12, 24, 32, 38, 34, 24, 14, 8, 4, 2, 0)
        shifts = (0, 1, 2, 3, 4, 3, 2, 1, 1, 0, 0, 0)
        p["hurt"] = float(angles[f])
        p["hdx"] = shifts[f]
        p["sweat"] = 1 if 2 <= f <= 5 else 0
    return p


def paste(dst, layer):
    mask = layer[..., 3] >= 128
    dst[mask] = layer[mask]


def put_empty(img, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 1 <= x < w - 1 and 1 <= y < h - 1 and img[y, x, 3] == 0:
        img[y, x, :3] = col
        img[y, x, 3] = 255


def forward_anchor(img):
    """Left edge of the head, in front of the cheek. The tail is not the front."""
    opaque = img[..., 3] >= 128
    ys, xs = np.nonzero(opaque)
    if len(xs) == 0:
        return None
    y1 = int(ys.min() + 0.48 * (ys.max() - ys.min()))
    sel = ys <= y1
    x = int(xs[sel].min())
    y = int(np.median(ys[sel][xs[sel] <= x + 1]))
    return x, y


def roar_rings(img, mouth, strength):
    """Sparse arcs in the empty space in front of the muzzle. They do not repaint the body."""
    if mouth is None or strength <= 0:
        return img
    mx, my = mouth
    cols = (WHITE, CREAM, TAN)
    radii = (5, 9, 13)[:strength]
    for i, rad in enumerate(radii):
        col = cols[i]
        for deg in range(-50, 51, 20):
            ang = math.radians(deg)
            put_empty(img, mx - 2 - rad * math.cos(ang), my + rad * math.sin(ang) * 0.7, col)
    return img


def impact_dots(img, mouth):
    if mouth is None:
        return img
    mx, my = mouth
    for k in range(5):
        ang = (k - 2) * 0.45
        put_empty(img, mx - 6 - 4 * math.cos(ang), my + 6 * math.sin(ang), WHITE)
        put_empty(img, mx - 10 - 3 * math.cos(ang), my + 9 * math.sin(ang), CREAM)
    return img


def sweat(img, anchor):
    if anchor is None:
        return img
    ax, ay = anchor
    for dx, dy in ((6, -8), (9, -5), (4, -12), (12, -9)):
        put_empty(img, ax + dx, ay + dy, WHITE)
    return img


def compose(rig, p):
    layers = rig["layers"]
    bdy = p["bdy"]
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(body, rig["socket"])
    paste(body, shift(layers["torso"], 0, bdy))
    tail = rotate_nearest(layers["tail"], p["tail"], rig["tail_pivot"])
    paste(body, shift(tail, 0, bdy))
    paste(body, shift(layers["foot_l"], p["ldx"], p["ldy"]))
    paste(body, shift(layers["foot_r"], p["rdx"], p["rdy"]))
    paste(body, shift(layers["muzzle"], p["mdx"], bdy + p["mdy"]))
    if abs(p["tail"]) >= 0.45:
        body = despeckle(body)
    return body


def finish_body(img, pivot, angle):
    turned = rotate_nearest(img, angle, pivot)
    turned = despeckle(turned)
    return fill_pinholes(turned)


def synth(rig, action, f, palette, flip):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    if p["spin"] > 0.45:
        img = finish_body(rig["placed"], rig["centroid"], p["spin"])
    elif p["hurt"] > 0.45 or p["hdx"]:
        leaned = finish_body(rig["placed"], rig["foot_pivot"], p["hurt"])
        img = shift(leaned, p["hdx"], 0)
    else:
        img = compose(rig, p)
    img = snap(img, palette)
    if p["rings"]:
        img = roar_rings(img, forward_anchor(img), p["rings"])
    if p["impact"]:
        img = impact_dots(img, forward_anchor(img))
    if p["sweat"]:
        opaque = rig["placed"][..., 3] >= 128
        ys, xs = np.nonzero(opaque)
        top = int(ys.min())
        band = xs[ys <= top + 6]
        img = sweat(img, (int(np.median(band)) + p["hdx"], top))
    img = snap(img, palette)
    if flip:
        img = img[:, ::-1].copy()
    return img


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (NORMAL / "front.png").read_bytes()
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
        "attack": (2, 4, 7, 11),
        "hurt": (2, 4, 6, 9),
    }
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            y, x = ai * CELL_H, si * CELL_W
            mask = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][mask] = tile[mask]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 6
    crops = []
    for act in ACTIONS:
        # East row is index 2. It is the mirrored still, nose to screen-right.
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][2 * CELL_H:3 * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            opaque = tile[..., 3] >= 128
            if not np.any(opaque):
                continue
            ys, xs = np.nonzero(opaque)
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
            mask = c[..., 3] >= 128
            bg[y:y + c.shape[0], x:x + c.shape[1]][mask] = c[mask]
    return Image.fromarray(bg, "RGBA")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def sprite_height(sheets):
    neutral = sheets["idle"][:CELL_H, :CELL_W]
    ys = np.nonzero(neutral[..., 3] >= 128)[0]
    return int(ys.max() - ys.min() + 1)


def south_attack_ratios(sheets):
    row = sheets["attack"][:CELL_H]
    counts = []
    for f in range(16):
        tile = row[:, f * CELL_W:(f + 1) * CELL_W]
        counts.append(int((tile[..., 3] >= 128).sum()))
    base = counts[0]
    ratios = [c / base for c in counts]
    return counts, ratios


def components(mask):
    from collections import deque

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


def qa_report(sheets, master, palette, digest, counts):
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = set(int(v) for v in np.unique(master[..., 3]))
    clip = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
    atk_counts, ratios = south_attack_ratios(sheets)
    spin_parts = []
    for f in range(5, 11):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        sizes = components(tile[..., 3] >= 128)
        spin_parts.append(sizes[:3])
    return {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_height(sheets),
        "off_palette": len(off),
        "alphas": sorted(alphas),
        "edge_touching": clip,
        "attack_counts": atk_counts,
        "attack_ratios": [round(r, 4) for r in ratios],
        "attack_ratio_min": min(ratios),
        "attack_ratio_max": max(ratios),
        "spin_components": spin_parts,
    }


def publish_dir8(sprite_h):
    """Replace only 0400_bibarel. The rest of dir8.js stays byte-for-byte."""
    entry = {
        "dex": 400,
        "form": "normal",
        "variant": "bibarel",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0400_bibarel/normal",
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
        if key in text:
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
        else:
            text = text.replace(prefix + "{", prefix + "{" + key + blob + ",", 1)
        json.loads(text[len(prefix):].rstrip().rstrip(";"))
        # The replaced entry is the only one that should carry this version.
        if text.count(f'"v":"{VERSION}"') != 1:
            raise SystemExit("dir8 version stamp collided")
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
            raise SystemExit("queue item 0400_bibarel missing")
        QUEUE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front_path = NORMAL / "front.png"
    front = np.array(Image.open(front_path).convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), flush=True)
    rig = build_rig(front, masks)

    rest = synth(rig, "idle", 0, palette, flip=False)
    if not np.array_equal(rest, rig["placed"]):
        differ = np.any(rest != rig["placed"], axis=2)
        raise SystemExit(f"idle frame 0 does not match the still ({int(differ.sum())} pixels)")
    flipped = synth(rig, "idle", 0, palette, flip=True)
    if not np.array_equal(flipped, rig["placed"][:, ::-1]):
        raise SystemExit("flipped idle frame 0 is not the mirrored still")
    print("rest-pose pixels changed 0", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (190, 170, 120),
        "tail": (40, 90, 220),
        "muzzle": (220, 40, 40),
        "foot_l": (40, 180, 80),
        "foot_r": (230, 140, 40),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", False, f) for f in (0, 4, 8, 12)]
        want += [("walk", False, f) for f in (0, 2, 4, 6, 8, 12)]
        want += [("attack", False, f) for f in (0, 2, 3, 5, 7, 9, 11, 15)]
        want += [("hurt", False, f) for f in (0, 2, 4, 8)]
        tiles = [synth(rig, act, f, palette, flip) for act, flip, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            mask = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][mask] = tile[mask]
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
                Image.fromarray(synth(rig, act, f, palette, d in FLIP_DIRS), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    (OUT_DIR / "actions").mkdir(parents=True, exist_ok=True)
    preview.save(OUT_DIR / "actions" / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    report = qa_report(sheets, master, palette, digest, counts)
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
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

    ratios = report["attack_ratios"]
    print("SOUTH attack opaque ratios:", " ".join(f"{r:.4f}" for r in ratios), flush=True)
    print(
        f"SOUTH attack opaque min {report['attack_ratio_min']:.4f} max {report['attack_ratio_max']:.4f}",
        flush=True,
    )
    print(
        json.dumps(
            {k: report[k] for k in ("off_palette", "alphas", "edge_touching", "sprite_h", "spin_components")},
            ensure_ascii=False,
        ),
        flush=True,
    )

    meta = {
        "generator": "forge_bibarel_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 400,
        "archetype": "quadruped_beaver",
        "facing": "left",
        "unflipped": ["N", "NW", "W", "SW"],
        "flipped": ["S", "SE", "E", "NE"],
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": report["sprite_h"],
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "palette_colors": int(len(palette)),
        "off_palette_colors": report["off_palette"],
        "attack_ratio_min": report["attack_ratio_min"],
        "attack_ratio_max": report["attack_ratio_max"],
        "source": "front.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = (
        report["off_palette"] != 0
        or set(report["alphas"]) != {0, 255}
        or report["edge_touching"]
        or report["attack_ratio_min"] < 0.90
        or report["attack_ratio_max"] > 1.08
    )
    if failed:
        print("QA failed", flush=True)
        if args.publish:
            raise SystemExit("refusing to publish a failed sheet")
        return

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        publish_dir8(report["sprite_h"])
        update_queue("done")
        print("published", VERSION, flush=True)


if __name__ == "__main__":
    main()
