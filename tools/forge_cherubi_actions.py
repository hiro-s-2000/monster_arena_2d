#!/usr/bin/env python3
"""Cherubi (0420) 8-direction action sheet from the normal 96x96 front.png.

Measured on this still only (opaque 577, bbox x33-63 y32-63, 13 colours).
Two round cherries on one stem. The big cherry is the lower left (pink x<=50,
y45-62). The small cherry is the upper right (pink x>=53, y44-53). Leaves sit
above y42. The neck under them is a few green pixels plus outline, not legs.
Both faces are dots on the front of the cherries. White shines are (36,51)
and (45,53), inset from the silhouette, so the still faces the viewer.
S SE E NE stay unflipped. N NW W SW mirror. No arms, no legs, no flower cloak.

idle   : leaves sway about the stem socket. Cherries and the stem contact
         stay on the idle-0 foot. Frame 0 is the still.
walk   : one-piece hop and a 1px sway. Not a biped stomp.
attack : 0-4 cherries bob (no mouth to open) plus rings, 5-10 one-piece
         in-plane 360 spin, 11-12 impact, 13-15 return.
hurt   : whole-body lean around the foot, at most 38 degrees. No squash.

Palette-locked. Alpha is only 0 or 255. front/back png and gif are not written.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
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
STILL_DIR = ROOT / "monsterSprite/gen4/0420_cherubi/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0420_cherubi/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 176, 156
VERSION = "cherubi_v2"
KEY = "0420_cherubi"
NEIGHBOR = "0001_bulbasaur"
UNFLIPPED = {"S", "SE", "E", "NE"}

PINK = (222, 98, 131)
PINK_DARK = (172, 65, 106)
LIGHT = (115, 205, 90)
GREEN = (82, 164, 74)
GREEN_MID = (65, 148, 65)
GREEN_DARK = (57, 115, 74)
OUTLINE = (16, 16, 16)
TEAL = (49, 65, 82)
DEEP = (131, 49, 82)
SHADOW = (106, 57, 74)
BROWN = (65, 49, 49)
HIGH = (255, 139, 172)
WHITE = (255, 255, 255)
PINK_COLS = (PINK, PINK_DARK, HIGH, DEEP, SHADOW)
GREEN_COLS = (LIGHT, GREEN, GREEN_MID, GREEN_DARK)

# Roar bob is upward (negative y). Frame 0 matches frame 15. Spin frames ignore these.
BOB_L = [0, -1, -2, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0]
BOB_R = [0, -2, -3, -3, -2, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0]
LEAF = [0, -6, -8, -8, -6, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0]
RINGS = [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2]
BURST = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0]
HURT = [0.0, 0.45, 0.75, 1.0, 0.82, 0.58, 0.36, 0.18, 0.08, 0.0, 0.0, 0.0]


def color_mask(img, cols):
    op = img[..., 3] >= 128
    m = np.zeros(op.shape, bool)
    for c in cols:
        m |= op & np.all(img[..., :3] == np.array(c, np.uint8), axis=-1)
    return m


def segment(img):
    """Leaves, stem neck, and the two cherries. No invented limbs."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    green = color_mask(img, GREEN_COLS)
    pink = color_mask(img, PINK_COLS)
    left = pink & (xx <= 51)
    right = pink & (xx >= 52)
    leaf_g = green & (yy <= 42)
    stem_g = green & (yy >= 43)
    other = op & ~green & ~pink
    h, w = op.shape

    def touches(mask, y, x):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                return True
        return False

    leaf_o = np.zeros_like(op)
    stem_o = np.zeros_like(op)
    left_o = np.zeros_like(op)
    right_o = np.zeros_like(op)
    ys, xs = np.nonzero(other)
    for y, x in zip(ys.tolist(), xs.tolist()):
        tl, tr = touches(left, y, x), touches(right, y, x)
        tleaf, tstem = touches(leaf_g, y, x), touches(stem_g, y, x)
        neck = 46 <= x <= 54 and 43 <= y <= 47 and (tleaf or tstem or touches(green, y, x))
        if neck and (tl or tr or tstem or tleaf):
            stem_o[y, x] = True
        elif (tl or tr) and not (tleaf or tstem):
            if tr and not tl:
                right_o[y, x] = True
            else:
                left_o[y, x] = True
        elif y <= 42 and (tleaf or not (tl or tr)):
            leaf_o[y, x] = True
        elif tl or tr:
            if tr and not tl:
                right_o[y, x] = True
            else:
                left_o[y, x] = True
        elif y >= 43:
            stem_o[y, x] = True
        else:
            leaf_o[y, x] = True

    parts = {
        "leaf": leaf_g | leaf_o,
        "stem": stem_g | stem_o,
        "cherry_l": left | left_o,
        "cherry_r": right | right_o,
    }
    own = np.zeros_like(op)
    overlap = 0
    for name in parts:
        overlap += int((own & parts[name]).sum())
        own |= parts[name]
    if int(own.sum()) != int(op.sum()) or overlap:
        raise SystemExit(
            f"masks do not partition the still: covered {int(own.sum())} opaque {int(op.sum())} overlap {overlap}"
        )
    if int(parts["cherry_l"].sum()) < 200 or int(parts["cherry_r"].sum()) < 40:
        raise SystemExit("a cherry collapsed")
    if int(parts["leaf"].sum()) < 80 or int(parts["stem"].sum()) < 4:
        raise SystemExit("leaf or stem collapsed")
    return parts


def measure_face(img):
    op = img[..., 3] >= 128
    white = op & np.all(img[..., :3] == np.array(WHITE, np.uint8), axis=-1)
    ys, xs = np.nonzero(op)
    wy, wx = np.nonzero(white)
    if len(wx) != 2:
        raise SystemExit(f"expected two eye shines, found {len(wx)}")
    return {
        "eyes": [int(v) for v in wx.tolist()],
        "eye_y": [int(v) for v in wy.tolist()],
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "opaque": int(op.sum()),
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
    """4x nearest. Multiples of 360 stay exact. No |cos| squash."""
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
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int32) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    cols = src[ys, xs]
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
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
        return 1
    return 0


def dilate(mask, radius):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask.copy()
    h, w = mask.shape
    out = mask.copy()
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        out[y0:y1, x0:x1] = True
    return out


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

    socket_zone = placed["leaf"] & dilate(placed["stem"], 2)
    socket = nearest_color(canvas, socket_zone, placed["leaf"])
    layers["socket"] = socket

    sy, sx = np.nonzero(socket_zone)
    if len(sy) == 0:
        raise SystemExit("leaf does not meet the stem")
    leaf_pivot = (float(sx.mean()), float(sy.mean()))

    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    foot_y = int(oys.max())
    foot_x = float(np.median(oxs[oys >= foot_y - 1]))
    foot = (foot_x, float(foot_y))
    fruit = placed["cherry_l"] | placed["cherry_r"]
    fy, fx = np.nonzero(fruit)
    face = (float(fx.mean()), float(fy.mean()))

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        leaf_pivot = (CELL_W - 1 - leaf_pivot[0], leaf_pivot[1])
        center = (CELL_W - 1 - center[0], center[1])
        foot = (CELL_W - 1 - foot[0], foot[1])
        face = (CELL_W - 1 - face[0], face[1])
        canvas = canvas[:, ::-1].copy()

    facing = 1.0 if not flip else -1.0
    return layers, leaf_pivot, center, foot, face, facing, canvas


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, sway=0.0, leaf=0.0, bob_l=0.0, bob_r=0.0,
        spin=0.0, recoil=0.0, rings=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["sway"] = wave(f, n, 8.0)
    elif action == "walk":
        p["sway"] = wave(f, n, 10.0, cycles=2.0)
        p["dy"] = -3.0 * max(0.0, wave(f, n, 1.0, cycles=2.0))
        p["dx"] = wave(f, n, 1.0, cycles=2.0)
    elif action == "attack":
        p["leaf"] = float(LEAF[f])
        p["bob_l"] = float(BOB_L[f])
        p["bob_r"] = float(BOB_R[f])
        p["rings"] = int(RINGS[f])
        p["burst"] = int(BURST[f])
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
    else:
        env = HURT[f]
        p["recoil"] = 38.0 * env
        p["dx"] = -4.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def compose(layers, leaf_pivot, facing, p, body_dx, body_dy):
    """Stem stays put relative to the body anchor. Leaves sway. Cherries bob."""
    ang = facing * (p["sway"] + p["leaf"])
    leaf = layers["leaf"]
    if abs(ang) >= 0.45:
        leaf = rotate_nearest(leaf, ang, leaf_pivot)
        leaf = despeckle(leaf)
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(body, shift(layers["socket"], body_dx, body_dy))
    paste(body, shift(layers["stem"], body_dx, body_dy))
    paste(body, shift(leaf, body_dx, body_dy))
    paste(body, shift(layers["cherry_l"], body_dx, body_dy + p["bob_l"]))
    paste(body, shift(layers["cherry_r"], body_dx, body_dy + p["bob_r"]))
    return body


def paint_ring(out, center, radius, count, colors):
    cx, cy = center
    n = 0
    for i in range(count):
        ang = 2 * math.pi * i / count
        col = colors[i % len(colors)]
        n += put(out, cx + radius * math.cos(ang), cy + radius * math.sin(ang), col)
    return n


def roar_rings(out, center, level):
    """level 1 is one ring, 2 is two. Dots stay off the sprite."""
    if level <= 0:
        return out
    paint_ring(out, center, 30, 16, (LIGHT, PINK, HIGH))
    if level >= 2:
        paint_ring(out, center, 36, 20, (PINK, LIGHT, HIGH))
    return out


def impact_burst(out, center, size):
    cx, cy = center
    ticks = [
        (22, 0), (-22, 0), (0, -24), (0, 20),
        (16, -14), (-16, -14), (16, 12), (-16, 12),
        (28, -6), (-28, -6), (26, 8), (-26, 8),
    ]
    col = HIGH if size > 1 else PINK
    alt = LIGHT
    for i, (dx, dy) in enumerate(ticks[: 8 + 4 * min(size, 2)]):
        put(out, cx + dx, cy + dy, col if i % 2 == 0 else alt)
        if size > 1:
            put(out, cx + dx + (1 if dx >= 0 else -1), cy + dy, WHITE)
    return out


def sparks(out, frame_idx, center):
    cx, cy = center
    d = frame_idx
    pts = [(cx - 20 - d, cy - 22), (cx + 18 + d, cy - 20), (cx - 8, cy + 22)]
    for sx, sy in pts:
        put(out, sx, sy, LIGHT)
        put(out, sx, sy + 1, HIGH)
    return out


def flash_hit(img):
    out = img.copy()
    a = out[..., 3] >= 128
    keep = a & (
        np.all(out[..., :3] == np.array(OUTLINE, np.uint8), axis=-1)
        | np.all(out[..., :3] == np.array(WHITE, np.uint8), axis=-1)
    )
    out[a & ~keep, :3] = HIGH
    return out


def largest_fraction(img):
    a = img[..., 3] >= 128
    if not np.any(a):
        return 1.0, 0, 0
    seen = np.zeros_like(a)
    h, w = a.shape
    best = 0
    ncomp = 0
    total = int(a.sum())
    ys, xs = np.nonzero(a)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        ncomp += 1
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
    return best / total, total, ncomp


def synth(rig, action, f, palette, effects=True):
    layers, leaf_pivot, center, foot, face, facing, _canvas = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        rest = pose(action, 0)
        rest = gen4_attack_strike.finish(rest, action, 0, __file__)

        rest["leaf"] = 0.0
        rest["sway"] = 0.0
        rest["bob_l"] = 0.0
        rest["bob_r"] = 0.0
        rest["dx"] = 0.0
        rest["dy"] = 0.0
        body = compose(layers, leaf_pivot, facing, rest, 0, 0)
        if recoiling:
            dx_i = int(round(p["dx"] * facing))
            dy_i = int(round(p["dy"]))
            body = shift(body, dx_i, dy_i)
            pivot = (foot[0] + dx_i, foot[1] + dy_i)
            body = rotate_nearest(body, -p["recoil"] * facing, pivot)
        else:
            body = rotate_nearest(body, p["spin"] * facing, center)
        body = despeckle(body)
    else:
        dx_i = int(round(p["dx"] * facing))
        dy_i = int(round(p["dy"]))
        body = compose(layers, leaf_pivot, facing, p, dx_i, dy_i)

    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_hit(img)
    if p["rings"] and not spinning:
        img = roar_rings(img, center, p["rings"])
    if p["burst"] and not spinning:
        img = impact_burst(img, center, p["burst"])
    if p["sparks"]:
        img = sparks(img, f, center)
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
    samples = {"idle": (0, 4, 8, 12), "walk": (0, 4, 8, 12), "attack": (2, 6, 8, 11), "hurt": (1, 3, 6, 10)}
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            m = tile[..., 3] >= 128
            bg[ai * CELL_H:(ai + 1) * CELL_H, si * CELL_W:(si + 1) * CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    """South rows in one fixed window so the hop, bob, spin, and lean stay in place."""
    idle0 = sheets["idle"][:CELL_H, :CELL_W]
    a = idle0[..., 3] >= 128
    ys, xs = np.nonzero(a)
    pad = 40
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad)
    ch, cw = y1 - y0, x1 - x0
    bg = np.full((len(ACTIONS) * ch, 16 * cw, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for f in range(FRAMES[act]):
            tile = sheets[act][y0:y1, f * CELL_W + x0:f * CELL_W + x1]
            m = tile[..., 3] >= 128
            bg[ai * ch:(ai + 1) * ch, f * cw:(f + 1) * cw][m] = tile[m]
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


def foot_row(tile):
    ys = np.nonzero(tile[..., 3] >= 128)[0]
    return int(ys.max()) if len(ys) else -1


def with_flock(lock_path, fn):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def object_span(text, key):
    token = f'"{key}":'
    i = text.find(token)
    if i < 0:
        return None
    j = i + len(token)
    while j < len(text) and text[j] in " \n\t":
        j += 1
    if j >= len(text) or text[j] != "{":
        raise SystemExit(f"{key} value is not an object")
    depth = 0
    k = j
    while k < len(text):
        ch = text[k]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i, k + 1
        k += 1
    raise SystemExit(f"unclosed {key}")


def publish_dir8(sprite_h):
    entry = {
        "dex": 420, "form": "normal", "variant": "cherubi", "gen": "gen4",
        "dir": "monsterSprite/gen4/0420_cherubi/normal",
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
        before = object_span(text, NEIGHBOR)
        if before is None:
            raise SystemExit(f"neighbor {NEIGHBOR} missing")
        neighbor = text[before[0]:before[1]]
        token = f'"{KEY}":'
        span = object_span(text, KEY)
        if span is None:
            text = text.replace(prefix + "{", prefix + "{" + token + blob + ",", 1)
        else:
            text = text[:span[0]] + token + blob + text[span[1]:]
        after = object_span(text, NEIGHBOR)
        if after is None or text[after[0]:after[1]] != neighbor:
            raise SystemExit(f"neighbor {NEIGHBOR} changed; dir8 not published")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(text)
        json.loads(text[len(prefix):].rstrip().rstrip(";"))
        tmp.replace(DIR8)
        return neighbor

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
                raise SystemExit("queue has no 0420_cherubi")
            raw = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            tmp = QUEUE.with_suffix(".json.tmp")
            tmp.write_text(raw, encoding="utf-8")
            os.replace(tmp, QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front_path = STILL_DIR / "front.png"
    front_hash = hashlib.sha1(front_path.read_bytes()).hexdigest()
    front = np.array(Image.open(front_path).convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    face_info = measure_face(front)
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("facing viewer", face_info, "parts", counts, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}

    rest = synth(rigs["S"], "idle", 0, palette, effects=False)
    placed = rigs["S"][6]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    idle_feet = [foot_row(synth(rigs["S"], "idle", f, palette, effects=False)) for f in range(16)]
    if len(set(idle_feet)) != 1:
        raise SystemExit(f"idle stem contact moved: {idle_feet}")
    print("idle foot row", idle_feet[0], flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "leaf": (40, 180, 60), "stem": (230, 200, 40),
        "cherry_l": (220, 80, 120), "cherry_r": (160, 40, 200),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "S", f) for f in (0, 4, 8, 12)]
        want += [("walk", "S", f) for f in (0, 2, 4, 6, 8, 10)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in (0, 1, 3, 6, 9)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        base_tile = synth(rigs["S"], "attack", 0, palette)
        base = int((base_tile[..., 3] >= 128).sum())
        print("ATTACK_S_RATIOS", flush=True)
        ratios = []
        for f in range(16):
            tile = synth(rigs["S"], "attack", f, palette)
            n = int((tile[..., 3] >= 128).sum())
            frac, _total, ncomp = largest_fraction(tile)
            ratio = n / base
            ratios.append(ratio)
            print(f"attack S {f:02d} opaque {n} ratio {ratio:.3f} main {frac:.3f} parts {ncomp}", flush=True)
        print(f"min {min(ratios):.3f} max {max(ratios):.3f}", flush=True)
        for f in (0, 2, 4, 8):
            tile = synth(rigs["S"], "walk", f, palette, effects=False)
            print(f"walk S {f:02d} foot {foot_row(tile)}", flush=True)
        hurt = synth(rigs["S"], "hurt", 3, palette)
        frac, _t, ncomp = largest_fraction(hurt)
        print(f"hurt peak main {frac:.3f} parts {ncomp} foot {foot_row(hurt)}", flush=True)
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
    contact.save(OUT_DIR / "preview_motion.png")
    contact.save(work / "review_S.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    counts_s, ratios = south_attack_ratios(sheets)
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(counts_s, ratios)):
        print(f"attack S {f:02d} opaque {c} ratio {r:.3f}", flush=True)
    print(f"min {min(ratios):.3f} max {max(ratios):.3f} h {sprite_h}", flush=True)

    if hashlib.sha1(front_path.read_bytes()).hexdigest() != front_hash:
        raise SystemExit("front.png changed")

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted({int(v) for v in np.unique(master[..., 3])})
    clip = {}
    spin_stats = []
    hurt_stats = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and d == "S" and 5 <= f <= 10:
                    frac, total, ncomp = largest_fraction(tile)
                    spin_stats.append({"f": f, "opaque": total, "main": round(frac, 3), "parts": ncomp})
                if act == "hurt" and d == "S" and f == 3:
                    frac, total, ncomp = largest_fraction(tile)
                    hurt_stats.append({"opaque": total, "main": round(frac, 3), "parts": ncomp})
    bad_ratio = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "cell": [CELL_W, CELL_H, FOOT_Y],
        "off_palette_colors": len(off), "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "attack_s_ratios": [round(r, 3) for r in ratios],
        "attack_s_min": round(min(ratios), 3), "attack_s_max": round(max(ratios), 3),
        "spin_body": spin_stats, "hurt_peak": hurt_stats, "rest_diff": 0,
        "idle_foot": idle_feet[0],
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
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1800], flush=True)

    bad_spin = [s for s in spin_stats if s["main"] < 0.90]
    bad_hurt = [s for s in hurt_stats if s["main"] < 0.90]
    failed = bool(off or clip or alphas != [0, 255] or bad_ratio or bad_spin or bad_hurt)
    if failed:
        if args.publish:
            update_queue("needs_fix")
        raise SystemExit(
            f"qa failed off={len(off)} clip={clip} alphas={alphas} ratios={bad_ratio} spin={bad_spin} hurt={bad_hurt}"
        )

    meta = {
        "generator": "forge_cherubi_actions", "version": VERSION, "key": KEY,
        "dex": 420, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "still_faces": "viewer",
        "archetype": "two_cherries_on_stem",
        "face": face_info, "parts": counts,
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "attack_s_min": round(min(ratios), 3), "attack_s_max": round(max(ratios), 3),
        "attack_s_ratios": [round(r, 3) for r in ratios],
        "source": "front.png 96x96", "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
        print("arena sheet bytes", (SYNC_OUT / "sheet.png").stat().st_size, flush=True)
    if args.publish:
        publish_dir8(sprite_h)
        update_queue("done")
        print("published", KEY, "h", sprite_h, "cw", CELL_W, "ch", CELL_H, "fy", FOOT_Y, flush=True)


if __name__ == "__main__":
    main()
