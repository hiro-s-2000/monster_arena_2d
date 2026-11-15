#!/usr/bin/env python3
"""Honchkrow normal (0430) 8-direction sheet from this form's front.png only.

Measured on normal/front.png (96x96, alpha already 0 or 255, 1637 opaque, 15 colours):
  bbox x21-74 y20-75 (54x56). Centroid about (44.3, 46.8).
  Faces left: beak tip (olive, x=24 y=38) is left of the one red eye (x=37 y=34).
  One beak (yellow, gold, olive). No second beak. One red eye, not two.
  Hat/crest is the fedora above y=32, including its rear brim.
  Red plume is the rear feather at x>=59 y40-55. It is not a wing and not a beak.
  The coat is one torso. No separate wing lobe is drawn.
  Two talons, split at the ankle gap (not one fused foot, not four):
    near/left  x34-47, toes end near y72
    far/right  x48-54, two claw tips at y75

  idle   : feet planted. Crest and plume sway. Frame 0 is the still.
  walk   : two-foot bird step. Those two talons only, plus crest and plume.
  attack : 0-4 the existing beak opens plus rings; 5-10 in-place 360 of the
           whole bird (coat included); 11-12 impact; 13-15 rest. No |cos| squash.
  hurt   : one-piece recoil, 38 degrees max.

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_honchkrow_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen4/0430_honchkrow/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0430_honchkrow/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left. Right-facing rows are the mirror. E looks screen-right.
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "honchkrow_v2"
KEY = "0430_honchkrow"

BLK = (16, 16, 16)
NAVY = (57, 57, 90)
BLUE = (57, 90, 156)
WHTF = (238, 238, 246)
LGRY = (164, 164, 180)
DPUR = (41, 32, 57)
GRY = (82, 82, 82)
DRED = (74, 32, 32)
MRED = (139, 49, 57)
RED = (222, 82, 90)
DGRY = (49, 49, 57)
YEL = (246, 222, 57)
WHT = (255, 255, 255)
OLV = (90, 90, 57)
GLD = (172, 148, 57)
BEAK_COLOURS = (YEL, GLD, OLV)
RED_COLOURS = (RED, MRED, DRED)

DRAW = ["torso", "plume", "crest", "talon_r", "talon_l", "beak", "jaw"]


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
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def is_col(img, rgb):
    a = img[..., 3] >= 128
    c = img[..., :3]
    return a & (c[..., 0] == rgb[0]) & (c[..., 1] == rgb[1]) & (c[..., 2] == rgb[2])


def segment(img):
    """Hat, one beak, rear red plume, two talons, one coat. Cuts are from this still only."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    beak_all = np.zeros_like(a)
    for c in BEAK_COLOURS:
        beak_all |= is_col(img, c)
    red = np.zeros_like(a)
    for c in RED_COLOURS:
        red |= is_col(img, c)
    blk = is_col(img, BLK)

    # Lower mandible is the tip side of the one beak. y=37 is the mouth line.
    jaw = beak_all & (yy >= 37)
    beak = beak_all & ~jaw

    # Rear feather. The single red eye pixel (x=37) stays on the head.
    plume_cols = red & (xx >= 55)
    outline_allow = (plume_cols | blk) & (xx >= 58) & (yy >= 38) & (yy <= 57)
    plume = plume_cols | flood(plume_cols, outline_allow)

    # Fedora, including the rear brim. The face and the eye start at y=33.
    crest = a & (yy <= 32) & ~plume & ~beak_all

    feet = a & (yy >= 64) & ~plume & ~crest & ~beak_all
    talon_l = feet & (((yy <= 70) & (xx <= 47)) | ((yy >= 71) & (xx <= 42)))
    talon_r = feet & ~talon_l

    used = jaw | beak | plume | crest | talon_l | talon_r
    torso = a & ~used
    parts = {
        "torso": torso, "plume": plume, "crest": crest,
        "beak": beak, "jaw": jaw, "talon_l": talon_l, "talon_r": talon_r,
    }
    cover = np.zeros_like(a)
    for name, m in parts.items():
        if np.any(cover & m):
            raise SystemExit(f"honchkrow masks overlap at {name}")
        cover |= m
    missed = int((cover != a).sum())
    if missed:
        raise SystemExit(f"honchkrow masks missed {missed} pixels")
    if int(beak.sum()) < 8 or int(jaw.sum()) < 8:
        raise SystemExit("beak mask collapsed")
    if int(crest.sum()) < 150:
        raise SystemExit("crest mask collapsed")
    if int(plume.sum()) < 80 or int(plume.sum()) > 200:
        raise SystemExit(f"plume mask out of range {int(plume.sum())}")
    if int(talon_l.sum()) < 30 or int(talon_r.sum()) < 30:
        raise SystemExit("talon mask collapsed")
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
    """4x nearest. Rigid: no scale and no mirror. Exact colours. No |cos| squash."""
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
    """Drop only pixels whose four neighbours are empty. After rotation only."""
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
    """Plug only fully enclosed holes. Does not web the talons."""
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        p = np.pad(a, 1)
        n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
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
    """Nearest palette colour. Off-palette pixels are cleared. Alpha is 0 or 255."""
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
    nearest = dist.argmin(axis=1)
    # A colour that is not already in the palette stays mapped, then any miss is cleared.
    mapped = palette[nearest]
    exact = dist.min(axis=1) == 0
    flat = flat.copy()
    flat[sel] = mapped[inv]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    alpha = np.where(a, 255, 0).astype(np.uint8)
    bad = np.zeros(sel.shape, bool)
    bad[sel] = ~exact[inv]
    alpha.reshape(-1)[bad] = 0
    out[..., 3] = alpha
    out[..., :3][out[..., 3] == 0] = 0
    return out


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        out[y, x, :3] = col
        out[y, x, 3] = 255


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


def add_socket(host, host_mask, part_mask, canvas, n=3):
    zone = part_mask & dilate(host_mask, n)
    if not np.any(zone):
        return
    fill = nearest_color(canvas, zone, host_mask)
    write = zone & (host[..., 3] < 128)
    host[write] = fill[write]


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

    body = placed["torso"]
    add_socket(layers["torso"], body, placed["crest"], canvas)
    add_socket(layers["torso"], body, placed["plume"], canvas)
    add_socket(layers["torso"], body, placed["talon_l"], canvas)
    add_socket(layers["torso"], body, placed["talon_r"], canvas)

    pivots = {
        "crest": contact_pivot(placed["crest"], body),
        "plume": contact_pivot(placed["plume"], body),
        "talon_l": contact_pivot(placed["talon_l"], body),
        "talon_r": contact_pivot(placed["talon_r"], body),
    }
    opy, opx = np.nonzero(canvas[..., 3] >= 128)
    center = (float(opx.mean()), float(opy.mean()))
    ty, tx = np.nonzero(placed["torso"])
    sole = (float(np.median(tx)), float(ty.max()))

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
        sole = (CELL_W - 1 - sole[0], sole[1])
    facing = -1.0 if not flip else 1.0
    return layers, pivots, center, sole, facing, flip


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, crest=0.0, plume=0.0,
        tal_l=0.0, tal_r=0.0, lift_l=0.0, lift_r=0.0, jaw=0, spin=0.0, hurt=0.0,
        rings=0, ring=0, burst=0, sparks=0,
    )
    if action == "idle":
        # Feet stay down. Only the hat and the rear feather move.
        p["crest"] = wave(f, n, 5.0, phase=0.4)
        p["plume"] = wave(f, n, -7.0, phase=-0.6)
    elif action == "walk":
        # Two steps. Ankles stay put; each talon lifts on its own beat.
        s = math.sin(2 * math.pi * 2.0 * f / n)
        p["dy"] = s
        p["tal_l"] = 32.0 * s
        p["tal_r"] = -28.0 * s
        p["lift_l"] = -2.0 if s > 0.0 else 0.0
        p["lift_r"] = -2.0 if s < 0.0 else 0.0
        p["crest"] = wave(f, n, 6.0, cycles=2.0, phase=0.5)
        p["plume"] = wave(f, n, -9.0, cycles=2.0, phase=-0.3)
    elif action == "attack":
        if f <= 4:
            p["jaw"] = (1, 2, 3, 2, 1)[f]
            p["rings"] = 2
            p["ring"] = f
            p["crest"] = (0, 6, 10, 8, 4)[f]
            p["plume"] = (0, -8, -12, -8, -4)[f]
        elif f <= 10:
            p["spin"] = 60.0 * (f - 4)
        elif f in (11, 12):
            p["burst"] = 6 if f == 11 else 5
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 38.0 * env
        p["dx"] = 5.0 * env
        p["sparks"] = 1 if 2 <= f <= 5 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def beak_colours_of(layer):
    a = layer[..., 3] >= 128
    rgb = layer[..., :3]
    g = np.zeros(a.shape, bool)
    for c in BEAK_COLOURS:
        g |= a & (rgb[..., 0] == c[0]) & (rgb[..., 1] == c[1]) & (rgb[..., 2] == c[2])
    return g


def beak_tip(layers_moved, facing):
    g = np.zeros(layers_moved[0].shape[:2], bool)
    for lyr in layers_moved:
        g |= beak_colours_of(lyr)
    ys, xs = np.nonzero(g)
    if len(xs) == 0:
        return None
    if facing > 0:
        x = int(xs.max())
        y = int(np.median(ys[xs >= x - 1]))
    else:
        x = int(xs.min())
        y = int(np.median(ys[xs <= x + 1]))
    return x, y


def open_beak(img, jaw_src, bdx, bdy, drop, facing):
    """Drop the existing lower beak. The gap is palette black. No second beak."""
    home = shift(jaw_src, bdx, bdy)
    dropped = shift(jaw_src, bdx, bdy + max(0, drop))
    hole = (home[..., 3] >= 128) & (dropped[..., 3] < 128) & (img[..., 3] < 128)
    img[hole, :3] = BLK
    img[hole, 3] = 255
    paste(img, dropped)
    return img


def roar_rings(out, mouth, facing, age, strength):
    mx, my = mouth
    cols = (WHT, LGRY, RED)
    for r_i in range(strength):
        rad = 7 + age * 2 + r_i * 5
        col = cols[min(r_i, 2)]
        for deg in range(-42, 43, 7):
            ang = math.radians(deg)
            put(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.62, col)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    cols = (WHT, YEL, RED)
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(2, size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), cols[r % 3])
    put(out, cx, cy, WHT)
    return out


def sparks(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    cx = (x0 + x1) // 2
    d = frame_idx
    pts = (
        (cx - facing * (4 + d), y0 + 3),
        (cx + facing * (3 + d), y0 + 8),
        (cx, y0 + 1),
    )
    for sx, sy in pts:
        put(out, sx, sy, WHT)
        put(out, sx + facing, sy + 1, RED)
    return out


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def compose(moved):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in DRAW:
        paste(img, moved[name])
    return img


def part_moved(layers, pivots, ang, dx, dy, pivot_name):
    lyr = layers
    if abs(ang) >= 0.45:
        lyr = rotate_nearest(layers, ang, pivots[pivot_name])
    return shift(lyr, dx, dy)


def synth(rig, action, f, palette):
    layers, pivots, center, sole, facing, flip = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    # Authored x is left-facing screen space. Right-facing rigs mirror it.
    x_mul = -1 if flip else 1
    bdx = int(round(p["dx"] * x_mul))
    bdy = int(round(p["dy"]))
    ang_mul = x_mul

    spinning = p["spin"] > 0.5 and (p["spin"] % 360.0) > 0.5
    recoiling = p["hurt"] > 0.45
    if spinning or recoiling:
        rest = {name: layers[name] for name in DRAW}
        base = compose(rest)
        if spinning:
            img = rotate_nearest(base, p["spin"] * ang_mul, center)
        else:
            # Positive angle on the left-facing still kicks the beak up and back.
            img = shift(rotate_nearest(base, p["hurt"] * ang_mul, sole), bdx, 0)
        img = fill_pinholes(img)
        img = despeckle(img)
    else:
        moved = {
            "torso": shift(layers["torso"], bdx, bdy),
            "plume": part_moved(layers["plume"], pivots, p["plume"] * ang_mul, bdx, bdy, "plume"),
            "crest": part_moved(layers["crest"], pivots, p["crest"] * ang_mul, bdx, bdy, "crest"),
            "talon_l": part_moved(
                layers["talon_l"], pivots, p["tal_l"] * ang_mul, bdx, bdy + int(round(p["lift_l"])), "talon_l"
            ),
            "talon_r": part_moved(
                layers["talon_r"], pivots, p["tal_r"] * ang_mul, bdx, bdy + int(round(p["lift_r"])), "talon_r"
            ),
            "beak": shift(layers["beak"], bdx, bdy),
            "jaw": shift(layers["jaw"], bdx, bdy),
        }
        img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        for name in ("torso", "plume", "crest", "talon_r", "talon_l", "beak"):
            paste(img, moved[name])
        if p["jaw"]:
            img = open_beak(img, layers["jaw"], bdx, bdy, int(p["jaw"]), facing)
        else:
            paste(img, moved["jaw"])

    img = snap(img, palette)
    if p["rings"] and not spinning and not recoiling:
        tip = beak_tip(
            (
                shift(layers["beak"], bdx, bdy),
                shift(layers["jaw"], bdx, bdy + int(p["jaw"])),
            ),
            facing,
        )
        if tip:
            img = roar_rings(img, tip, facing, p["ring"], p["rings"])
    if p["burst"]:
        tip = beak_tip((img,), facing)
        if tip:
            img = impact_burst(img, (tip[0] + facing * 6, tip[1]), p["burst"])
    if p["sparks"]:
        img = sparks(img, f, facing)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (SRC_DIR / "front.png").read_bytes()
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
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 2, 4, 6, 8, 10, 12, 14),
        "attack": (0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15),
        "hurt": (0, 2, 4, 6, 8, 11),
    }
    cols = max(len(v) for v in samples.values())
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            m = tile[..., 3] >= 128
            y = ai * CELL_H
            x = si * CELL_W
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 4
    crops = []
    for act in ACTIONS:
        row = []
        band = sheets[act][0:CELL_H]
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


def opaque_count(tile):
    return int((tile[..., 3] == 255).sum())


def south_attack_ratios(sheets):
    counts = []
    band = sheets["attack"][:CELL_H]
    for f in range(FRAMES["attack"]):
        counts.append(opaque_count(band[:, f * CELL_W:(f + 1) * CELL_W]))
    base = counts[0]
    ratios = [c / base for c in counts]
    return counts, ratios


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


def assert_claim():
    import os as _rebuild_os
    if _rebuild_os.environ.get("GEN4_ATTACK_REBUILD") == "1":
        return
    data = json.loads(QUEUE.read_text())
    for item in data["items"]:
        if item.get("id") != KEY:
            continue
        if item.get("status") != "in_progress" or item.get("claim") != "parallel-batch":
            raise SystemExit(f"stop: {KEY} is {item.get('status')} claim={item.get('claim')}")
        return
    raise SystemExit(f"stop: queue has no {KEY}")


def publish_dir8(sprite_h):
    entry = {
        "dex": 430, "form": "normal", "variant": "honchkrow", "gen": "gen4",
        "dir": "monsterSprite/gen4/0430_honchkrow/normal",
        "sheetDir": "agent-sprite-forge", "layout": "actions_8dir", "nf": 16,
        "cw": CELL_W, "ch": CELL_H, "w": CELL_W, "h": int(sprite_h), "fy": FOOT_Y,
        "an": [16, 16, 16, 12], "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        payload = text[len(prefix):].strip()
        if payload.endswith(";"):
            payload = payload[:-1]
        data = json.loads(payload)
        data[KEY] = entry
        blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        # Only this key may change.
        if set(data) != set(json.loads(payload)):
            raise SystemExit("dir8 key set changed")
        out = prefix + blob + ";\n"
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(out)
        json.loads(out[len(prefix):].rstrip().rstrip(";"))
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
                if item.get("status") != "in_progress" or item.get("claim") != "parallel-batch":
                    raise SystemExit(f"stop: lost claim before queue write {item.get('status')}")
                item["status"] = status
                item.pop("claim", None)
                found = True
                break
            if not found:
                raise SystemExit(f"queue has no {KEY}")
            tmp = QUEUE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def measure_facing(img):
    beak = np.zeros(img.shape[:2], bool)
    for c in BEAK_COLOURS:
        beak |= is_col(img, c)
    by, bx = np.nonzero(beak)
    beak_x = int(bx.min())
    red = np.zeros(img.shape[:2], bool)
    for c in RED_COLOURS:
        red |= is_col(img, c)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    eye = red & (xx < 50) & (yy < 40)
    ey, ex = np.nonzero(eye)
    eye_x = int(np.median(ex)) if len(ex) else -1
    return beak_x, eye_x


def body_pieces(img):
    return [s for s in components(img[..., 3] >= 128) if s > 12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    assert_claim()
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), flush=True)
    beak_x, eye_x = measure_facing(front)
    print("facing check beak_tip_x", beak_x, "eye_x", eye_x, flush=True)
    if not (beak_x < eye_x):
        raise SystemExit("expected a left-facing still")

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    print("pivot crest", [round(v, 1) for v in rigs["W"][1]["crest"]],
          "plume", [round(v, 1) for v in rigs["W"][1]["plume"]],
          "center", [round(v, 1) for v in rigs["W"][2]], flush=True)

    rest = synth(rigs["W"], "idle", 0, palette)
    a = front[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    placed = np.zeros_like(rest)
    dst_x = (CELL_W - (x1 - x0)) // 2
    dst_y = FOOT_Y - (y1 - y0)
    placed[dst_y:dst_y + (y1 - y0), dst_x:dst_x + (x1 - x0)] = front[y0:y1, x0:x1]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()):
        raise SystemExit("idle frame 0 does not match the still")

    east = synth(rigs["E"], "idle", 0, palette)
    flipped = placed[:, ::-1].copy()
    ed = np.any(east != flipped, axis=2) & ((east[..., 3] >= 128) | (flipped[..., 3] >= 128))
    print("east mirror pixels changed", int(ed.sum()), flush=True)
    if int(ed.sum()):
        raise SystemExit("east idle 0 is not the mirror of the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (70, 110, 170), "plume": (220, 40, 50), "crest": (40, 70, 200),
        "beak": (255, 200, 40), "jaw": (255, 120, 20),
        "talon_l": (40, 200, 80), "talon_r": (180, 70, 220),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    broken = []
    for f in range(5, 11):
        fr = synth(rigs["S"], "attack", f, palette)
        sizes = body_pieces(fr)
        if len(sizes) != 1:
            broken.append(("attack", f, sizes[:6]))
    for f in (1, 3, 6, 9):
        fr = synth(rigs["S"], "hurt", f, palette)
        sizes = body_pieces(fr)
        if len(sizes) != 1:
            broken.append(("hurt", f, sizes[:6]))
    print("piece check", broken if broken else "ok", flush=True)
    if broken:
        raise SystemExit("spin or recoil broke the body")

    if args.samples:
        atk = [synth(rigs["S"], "attack", f, palette) for f in range(16)]
        base = opaque_count(atk[0])
        ratios = [opaque_count(t) / base for t in atk]
        print("south attack opaque", [opaque_count(t) for t in atk], flush=True)
        print("south attack ratios", [round(v, 4) for v in ratios], flush=True)
        print("min", round(min(ratios), 4), "max", round(max(ratios), 4), flush=True)
        tiles = atk
        tiles += [synth(rigs["W"], "walk", f, palette) for f in range(16)]
        tiles += [synth(rigs["S"], "hurt", f, palette) for f in range(12)]
        tiles += [synth(rigs["W"], "idle", f, palette) for f in (0, 4, 8, 12)]
        cols = 16
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
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

    total = len(DIRS) * sum(FRAMES.values())
    done = 0
    for act in ACTIONS:
        for d in DIRS:
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                done += 1
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
            print(f"[{done}/{total}] {act}/{d}", flush=True)
    state["done"] = [f"{act}/{d}" for act in ACTIONS for d in DIRS]
    state["hash"] = digest
    state_p.write_text(json.dumps(state))

    sheets = stack_sheets(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    (SRC_DIR / "actions").mkdir(exist_ok=True)
    preview.save(SRC_DIR / "actions" / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    west = sheets["idle"][6 * CELL_H:7 * CELL_H, :CELL_W]
    wd = np.any(west != placed, axis=2) & ((west[..., 3] >= 128) | (placed[..., 3] >= 128))
    print("W idle0 pixels changed", int(wd.sum()), "sprite_h", sprite_h, flush=True)

    counts_a, ratios = south_attack_ratios(sheets)
    print("south attack opaque", counts_a, flush=True)
    print("south attack ratios", [round(v, 4) for v in ratios], flush=True)
    print("min", round(min(ratios), 4), "max", round(max(ratios), 4), flush=True)

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    bad_alpha = [v for v in alphas if v not in (0, 255)]
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    ratio_fail = [i for i, v in enumerate(ratios) if v < 0.90 or v > 1.08]
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": alphas, "bad_alpha": bad_alpha,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "south_attack_opaque": counts_a,
        "south_attack_ratios": [round(v, 4) for v in ratios],
        "ratio_min": round(min(ratios), 4), "ratio_max": round(max(ratios), 4),
        "ratio_fail_frames": ratio_fail,
        "beak_tip_x": beak_x, "eye_x": eye_x,
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
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False), flush=True)

    meta = {
        "generator": "forge_honchkrow_actions", "version": VERSION, "key": KEY,
        "dex": 430, "form": "normal", "variant": "honchkrow",
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "source": "front.png 96x96", "gifs_untouched": True,
        "facing": "left", "beak_tip_x": beak_x, "eye_x": eye_x,
        "beak": 1, "eyes": 1, "talons": 2, "wings": "coat",
        "south_attack_ratio_min": round(min(ratios), 4),
        "south_attack_ratio_max": round(max(ratios), 4),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = bool(off or bad_alpha or clip or ratio_fail or int(wd.sum()) or sprite_h <= 0)
    if failed:
        if not args.no_publish:
            update_queue("needs_fix")
        raise SystemExit("qa failed")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if not args.no_publish:
        publish_dir8(sprite_h)
        update_queue("done")
        print("published", KEY, VERSION, "h", sprite_h, flush=True)


if __name__ == "__main__":
    main()
