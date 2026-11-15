#!/usr/bin/env python3
"""Mega Garchomp (0445_garchomp-mega) 8-direction action sheet from front.png.

Archetype: biped shark. Measured on the 96x96 still, bbox (8,8)-(87,86).
Yellow eyes sit at about x 37-47, y 12-18. The red snout tip reaches x 83,
so the still faces right. Unflipped directions are S SE E NE. N NW W SW are
mirrored. E faces screen-right.

Parts that already exist:
  left red scythe (x 8-30, y 24-68)
  snout (x 60-83, y 20-34), pivoted where it meets the face
  right hip fin (x 74-87, y 52-71), not a wing and not a third leg
  two feet: left y>=70 x<=40, right y>=74 x>=55
  head holds the eyes and the 11-pixel mouth at about (44-48, 22-25)

There is no second scythe sticking out. The red chest stays on the torso.
The roar tips the existing snout and draws rings. It does not paint a jaw.

  idle   : feet planted. Scythe, fin, and snout sway. Frame 0 is the still.
  walk   : the two feet step. The scythe and fin counter-swing.
  attack : snout roar + rings, in-place 360 spin, impact at the snout, recover.
  hurt   : flash and a rigid recoil of at most 34 degrees. No belly mirror.

Every opaque pixel snaps to the still palette. Alpha is 0 or 255.
front/back png and gif are not written.
"""

import argparse
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
MEGA_DIR = ROOT / "monsterSprite/gen4/0445_garchomp/mega/garchomp-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0445_garchomp/mega/garchomp-mega/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces right (snout to the right of the eyes).
FACE_RIGHT = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 200, 200, 172
VERSION = "garchomp_mega_v2"

INK = (16, 16, 16)
WHITE = (245, 245, 245)
YELLOW = (255, 213, 24)
RED = (230, 74, 49)
PALE = (197, 205, 213)
RED_COLS = ((189, 57, 65), (230, 74, 49), (126, 33, 33))
EYE_COLS = ((255, 213, 24), (197, 148, 16))


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


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


def segment(img):
    """Partition this still. Cuts follow its scythe, snout, fin, and two feet."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    rgb = img[..., :3]
    yy, xx = np.mgrid[0:h, 0:w]
    red = np.zeros_like(a)
    for col in RED_COLS:
        red |= a & np.all(rgb == np.array(col, np.uint8), axis=-1)

    # Left scythe. It leaves the body on the left and ends before the feet.
    arm_l = flood(a & (xx <= 12) & (yy >= 40) & (yy <= 60), a & (xx <= 30) & (yy >= 22) & (yy <= 69))
    hook = red & (xx <= 28) & (yy >= 60) & (yy <= 69)
    arm_l = arm_l | hook

    # Snout, including the dark tip past the red. It does not include the mouth.
    snout = flood(a & (xx >= 73) & (yy >= 20) & (yy <= 26), a & (xx >= 60) & (yy <= 34))

    # Right hip fin. Separate from both feet. Not a wing.
    fin = flood(a & (xx >= 80) & (yy >= 52) & (yy <= 60), a & (xx >= 74) & (yy >= 50) & (yy <= 72))

    leg_l = a & (yy >= 70) & (xx <= 40) & ~arm_l & ~snout & ~fin
    leg_r = a & (yy >= 74) & (xx >= 55) & ~arm_l & ~snout & ~fin & ~leg_l
    head = a & (yy <= 40) & (xx >= 30) & (xx <= 68) & ~arm_l & ~snout & ~fin & ~leg_l & ~leg_r
    used = arm_l | snout | fin | leg_l | leg_r | head
    torso = a & ~used
    parts = {
        "arm_l": arm_l, "snout": snout, "fin": fin,
        "leg_l": leg_l, "leg_r": leg_r, "head": head, "torso": torso,
    }
    own = None
    for m in parts.values():
        own = m.copy() if own is None else (own | m)
    overlap = 0
    names = list(parts)
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            overlap += int((parts[n1] & parts[n2]).sum())
    if int(own.sum()) != int(a.sum()) or overlap:
        raise SystemExit(f"masks failed cover={int(own.sum())} opaque={int(a.sum())} overlap={overlap}")
    return parts


def measure_facing(img):
    """Snout tip is the rightmost pixel ahead of the yellow eyes."""
    a = img[..., 3] >= 128
    rgb = img[..., :3]
    eye = np.zeros_like(a)
    for col in EYE_COLS:
        eye |= a & np.all(rgb == np.array(col, np.uint8), axis=-1)
    eye &= np.arange(a.shape[0])[:, None] <= 20
    ey, ex = np.nonzero(eye)
    if len(ex) < 4:
        raise SystemExit("expected the yellow eyes")
    eye_x = float(np.mean(ex))
    band = a & (np.arange(a.shape[0])[:, None] >= 20) & (np.arange(a.shape[0])[:, None] <= 28)
    tip_x = int(np.nonzero(band)[1].max())
    faces_right = tip_x > eye_x + 8
    return faces_right, eye_x, tip_x


def edge_pivot(mask, side):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0, 0.0
    if side == "right":
        sel = xs >= xs.max() - 2
    elif side == "left":
        sel = xs <= xs.min() + 2
    elif side == "top":
        sel = ys <= ys.min() + 2
    else:
        sel = ys >= ys.max() - 2
    return float(xs[sel].mean()), float(ys[sel].mean())


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
    if abs(angle_deg) < 0.45 or not np.any(layer[..., 3] >= 128):
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
    out = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    out[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    return out[1::4, 1::4].copy()


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    step = 300
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
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
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def roar_rings(out, origin, facing, age, strength):
    mx, my = origin
    cols = (WHITE, RED, YELLOW)
    for r_i in range(strength):
        rad = 5 + age * 2 + r_i * 4
        col = cols[min(r_i, 2)]
        for deg in range(-46, 47, 8):
            ang = math.radians(deg)
            put(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.7, col)
    return out


def speed_lines(out, facing, n, seed, length):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    rear_x = (x1 - (x1 - x0) // 6) if facing < 0 else (x0 + (x1 - x0) // 6)
    y_lo = y0 + (y1 - y0) // 3
    y_hi = y0 + (y1 - y0) * 2 // 3
    for _k in range(n):
        y = int(rng.integers(y_lo, max(y_lo + 1, y_hi)))
        off = int(rng.integers(2, 6))
        length_i = int(rng.integers(4, length + 1))
        for i in range(length_i):
            put(out, rear_x - facing * (off + i), y, WHITE if i < 2 else YELLOW)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(max(2, size // 2), size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), WHITE if r < size - 2 else RED)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, WHITE)
    return out


def sparks(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, _y1 = bb
    d = frame_idx
    cx = (x0 + x1) // 2
    pts = [
        (cx - facing * (4 + d), y0 + 4),
        (cx + facing * 3, y0 + 10),
        (cx - facing * 2, y0 + 16),
    ]
    for sx, sy in pts:
        put(out, sx, sy, WHITE)
        put(out, sx - facing, sy + 1, YELLOW)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 90)
    out[a & ~dark, :3] = WHITE
    return out


def largest_fraction(img):
    a = img[..., 3] >= 128
    if not np.any(a):
        return 1.0
    seen = np.zeros_like(a)
    h, w = a.shape
    best = 0
    total = int(a.sum())
    ys, xs = np.nonzero(a)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
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
        best = max(best, n)
    return best / total


def build_rig(img, masks, flip):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
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

    cushions = (
        ("arm_l", "torso", 4),
        ("snout", "head", 3),
        ("fin", "torso", 3),
        ("leg_l", "torso", 3),
        ("leg_r", "torso", 3),
    )
    for part_name, under, rad in cushions:
        if not np.any(placed[part_name]) or not np.any(placed[under]):
            continue
        zone = placed[part_name] & dilate(placed[under], rad)
        if not np.any(zone):
            continue
        fill = nearest_color(canvas, zone, placed[under])
        write = zone & (layers[under][..., 3] < 128)
        layers[under][write] = fill[write]

    pivots = {
        "arm_l": edge_pivot(placed["arm_l"], "right"),
        "snout": edge_pivot(placed["snout"], "left"),
        "fin": edge_pivot(placed["fin"], "left"),
        "leg_l": edge_pivot(placed["leg_l"], "top"),
        "leg_r": edge_pivot(placed["leg_r"], "top"),
    }
    oys, oxs = np.nonzero(a)
    center = (float(oxs.mean()) - x0 + dst_x, float(oys.mean()) - y0 + dst_y)
    hys, hxs = np.nonzero(placed["snout"])
    if len(hxs):
        tip_sel = hxs >= hxs.max() - 1
        bite = (float(hxs[tip_sel].mean()), float(hys[tip_sel].mean()))
    else:
        bite = center

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
        bite = (CELL_W - 1 - bite[0], bite[1])
    # +1 faces screen-right. The still is already right-facing, so flip is the left view.
    facing = -1.0 if flip else 1.0
    return layers, pivots, center, bite, facing


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, spin=0.0, recoil=0.0,
        snout=0.0, fin=0.0, arm=0.0, leg=0.0,
        ring=-1, rings=0, lines=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["snout"] = wave(f, n, 3.0, phase=0.4)
        p["fin"] = wave(f, n, 5.0, phase=-0.7)
        p["arm"] = wave(f, n, 4.0, phase=0.5)
    elif action == "walk":
        p["dx"] = wave(f, n, 1.5, cycles=2.0)
        p["dy"] = wave(f, n, 0.8, cycles=2.0)
        p["leg"] = wave(f, n, 12.0, cycles=2.0)
        p["arm"] = wave(f, n, -8.0, cycles=2.0)
        p["fin"] = wave(f, n, 7.0, phase=-0.6, cycles=2.0)
        p["snout"] = wave(f, n, 3.0, cycles=2.0)
        p["lines"] = 2
    elif action == "attack":
        snout_t = [4, 10, 14, 12, 6, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
        arm_t = [-4, -8, -10, -4, 8, 0, 0, 0, 0, 0, 0, 6, 3, 0, 0, 0]
        p["snout"] = snout_t[f]
        p["arm"] = arm_t[f]
        p["fin"] = snout_t[f] * 0.35
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
            p["lines"] = 3
        if 1 <= f <= 4:
            p["ring"], p["rings"] = f - 1, 2
        if f in (11, 12):
            p["burst"] = 1
    else:
        env = [0, 0.4, 0.8, 1.0, 0.9, 0.65, 0.4, 0.22, 0.1, 0.04, 0.0, 0][f]
        p["dx"] = -6.0 * env
        p["recoil"] = 34.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def synth(rig, action, f, palette):
    layers, pivots, center, bite, facing = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    fwd = facing
    bdx, bdy = p["dx"] * fwd, p["dy"]
    sgn = 1.0 if fwd > 0 else -1.0
    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4

    def moved(name, ang, dx, dy):
        return shift(rotate_nearest(layers[name], ang * sgn, pivots[name]), dx, dy)

    leg_dy = 0.0 if action == "idle" else bdy
    snout = moved("snout", p["snout"], bdx, bdy)
    fin = moved("fin", p["fin"], bdx, bdy)
    arm = moved("arm_l", p["arm"], bdx, bdy)
    leg_l = moved("leg_l", p["leg"], bdx, leg_dy)
    leg_r = moved("leg_r", -p["leg"], bdx, leg_dy)
    torso = shift(layers["torso"], bdx, bdy)
    head = shift(layers["head"], bdx, bdy)

    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for lyr in (torso, head, leg_l, leg_r, fin, arm, snout):
        paste(body, lyr)
    if any(abs(p[k]) >= 0.45 for k in ("snout", "fin", "arm", "leg")) and not spinning and not recoiling:
        body = despeckle(body)

    if spinning:
        body = despeckle(rotate_nearest(body, p["spin"], center))
    elif recoiling:
        body = despeckle(rotate_nearest(body, -p["recoil"] * fwd, center))

    img = snap(body, palette)
    if p["rings"] and p["ring"] >= 0 and not spinning:
        img = roar_rings(img, (bite[0] + bdx, bite[1] + bdy), fwd, p["ring"], p["rings"])
    if p["lines"]:
        img = speed_lines(img, fwd, p["lines"], seed=f * 17 + (1 if action == "walk" else 9), length=8)
    if p["burst"]:
        img = impact_burst(img, (bite[0] + 4 * fwd, bite[1]), 9 if f >= 12 else 7)
    if p["sparks"]:
        img = sparks(img, f, fwd)
    if p["flash"]:
        img = flash_white(img)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (MEGA_DIR / "front.png").read_bytes()
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
    samples = {"idle": (0, 4, 8, 12), "walk": (0, 4, 8, 12), "attack": (2, 4, 7, 11), "hurt": (1, 3, 6, 9)}
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            y, x = ai * CELL_H, si * CELL_W
            m = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 4
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            # E is the unflipped right-facing pose.
            tile = sheets[act][2 * CELL_H:3 * CELL_H, f * CELL_W:(f + 1) * CELL_W]
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


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def publish_dir8(sprite_h):
    text = DIR8.read_text()
    prefix = "window.ARENA_DIR8 = "
    if not text.startswith(prefix):
        raise SystemExit("dir8.js header changed; not publishing")
    entry = {
        "dex": 445, "form": "mega", "variant": "garchomp-mega", "gen": "gen4",
        "dir": "monsterSprite/gen4/0445_garchomp/mega/garchomp-mega",
        "sheetDir": "agent-sprite-forge", "layout": "actions_8dir", "nf": 16,
        "cw": CELL_W, "ch": CELL_H, "w": CELL_W, "h": int(sprite_h), "fy": FOOT_Y,
        "an": [FRAMES[a] for a in ACTIONS],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    key = '"0445_garchomp-mega":'
    if key not in text:
        text = text.replace(prefix + "{", prefix + "{" + key + blob + ",", 1)
    else:
        start = text.index(key) + len(key)
        while text[start].isspace():
            start += 1
        if text[start] != "{":
            raise SystemExit("dir8 value is not an object")
        depth = 0
        end = None
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end is None:
            raise SystemExit("dir8 value did not close")
        text = text[:start] + blob + text[end:]
    DIR8.write_text(text)
    json.loads(text[len(prefix):].rstrip().rstrip(";"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(MEGA_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    faces_right, eye_x, tip_x = measure_facing(front)
    if not faces_right:
        raise SystemExit(f"expected the snout tip right of the eyes, tip {tip_x} eye {eye_x}")
    print(f"facing right eye_x={eye_x:.1f} snout_tip_x={tip_x}", flush=True)

    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), "sum", sum(counts.values()), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in FACE_RIGHT) for d in DIRS}

    rest = synth(rigs["E"], "idle", 0, palette)
    a = front[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    dst_x = (CELL_W - (x1 - x0)) // 2
    dst_y = FOOT_Y - (y1 - y0)
    placed = np.zeros_like(rest)
    placed[dst_y:dst_y + (y1 - y0), dst_x:dst_x + (x1 - x0)] = front[y0:y1, x0:x1]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    diff_n = int(differ.sum())
    print("rest-pose pixels changed", diff_n, flush=True)
    if diff_n:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (180, 180, 190), "head": (250, 250, 250), "snout": (230, 40, 40),
        "fin": (40, 90, 220), "arm_l": (40, 180, 80),
        "leg_l": (240, 200, 40), "leg_r": (240, 140, 20),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "E", f) for f in (0, 4, 8, 12)]
        want += [("walk", "E", f) for f in (0, 4, 8, 12)]
        want += [("attack", "E", f) for f in (0, 2, 4, 5, 7, 8, 10, 11, 13)]
        want += [("attack", "W", f) for f in (3, 7)]
        want += [("hurt", "E", f) for f in (0, 1, 3, 6, 11)]
        tiles = []
        for act, d, f in want:
            tile = synth(rigs[d], act, f, palette)
            edge = tile[..., 3] >= 128
            touch = bool(edge[0].any() or edge[-1].any() or edge[:, 0].any() or edge[:, -1].any())
            print(f"sample {act} {d} {f:02d} main={largest_fraction(tile):.3f} edge={touch}", flush=True)
            tiles.append(tile)
        cols = 6
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
        if frames_dir.exists() and state.get("hash") not in (None, digest):
            shutil.rmtree(frames_dir)
        elif args.force and frames_dir.exists():
            shutil.rmtree(frames_dir)
        state = {"hash": digest, "done": []}
    frames_dir.mkdir(parents=True, exist_ok=True)

    total = len(DIRS) * len(ACTIONS)
    for act in ACTIONS:
        for d in DIRS:
            key = f"{act}/{d}"
            if key in state["done"] and state.get("hash") == digest:
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
    preview_image(sheets).save(OUT_DIR / "preview.png")
    contact = motion_contact(sheets)
    contact.save(work / "review_E.png")
    contact.save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = set(int(v) for v in np.unique(master[..., 3]).tolist())
    clip = {}
    spin_frac = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and d == "E" and 5 <= f <= 10:
                    spin_frac[str(f)] = round(largest_fraction(tile), 3)
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": sorted(alphas),
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "spin_main_frac_E": spin_frac, "facing": "right",
        "eye_x": eye_x, "snout_tip_x": tip_x,
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "hover"), ("walk", "gait")):
            od = qa / f"gait_{act}_E"
            if od.exists():
                report[f"gait_{act}"] = "skipped_existing"
                continue
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "E"),
                 "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:900], flush=True)

    meta = {
        "generator": "forge_garchomp_mega_actions", "version": VERSION, "key": "0445_garchomp-mega",
        "dex": 445, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "face_right": sorted(FACE_RIGHT), "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "facing": "right", "eye_x": eye_x, "snout_tip_x": tip_x,
        "source": "front.png 96x96", "gifs_untouched": True, "mouth": "snout_and_rings",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        if off or clip:
            raise SystemExit("refusing to publish with palette or edge errors")
        publish_dir8(sprite_h)
        print("dir8 published", VERSION, flush=True)


if __name__ == "__main__":
    main()
