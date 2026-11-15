#!/usr/bin/env python3
"""Roserade (0407_roserade) 8-direction action sheet from this form's front.png.

Measured on normal/front.png (96x96, alpha already 0 or 255, 15 colours, 1674
opaque, one component). Opaque bbox x19-77, y19-76.

No open mouth. Burgundy flecks at (41, 42) and (47-49, 42-43) stay on the
torso; they are not a jaw. The roar flares the two bouquets and draws rings.

Eyes are yellow. A 4px cluster sits at x40-41, y38-40. The larger cluster is
x47-51, y40-42. Combined eye mean x is 45.92. White and lavender hair (mean x
49.46) runs out to x64, to the right of both eyes, so the hair trails right
and the face points left. The pink bouquet (x19-36, y32-50) is the forward
hand. The blue bouquet (x62-76, y42-59) is the rear hand. Unflipped dirs are
N NW W SW. Flipped dirs are S SE E NE, so E faces screen-right.

Biped. The two bouquets are the hands. Feet split below y71: left shoe
x40-45 (yellow toes x41-42), right shoe x53-58 (yellow toes x56-57). The
dress point between them (about x48-49) stays on the cape. No third arm and
no extra leg. Cape and dress stay on the torso.

  idle   : frame 0 is the still. Bouquets sway. Feet stay planted.
  walk   : the two shoes step in turn. Bouquets counter-swing.
  attack : 0-4 bouquet flare plus rings; 5-10 one-piece 360 (no |cos| squash);
           11-12 impact; 13-15 recover.
  hurt   : one-piece recoil, 38 degrees max, around the soles.

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
NORMAL = ROOT / "monsterSprite/gen4/0407_roserade/normal"
OUT_DIR = NORMAL / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0407_roserade/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left. These four keep the drawing. E is the mirror and faces right.
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 192, 180, 160
VERSION = "roserade_v2"
KEY = "0407_roserade"

PINK = (255, 98, 131)
PINK_D = (189, 65, 106)
BURG = (123, 57, 90)
BLUE = (82, 115, 238)
BLUE_D = (74, 90, 189)
BLUE_N = (65, 74, 131)
WHITE = (255, 255, 255)
LAV = (213, 205, 222)
YEL = (246, 213, 74)
PINK_COLS = (PINK, PINK_D, BURG)
BLUE_COLS = (BLUE, BLUE_D, BLUE_N)
DRAW = ("torso", "leg_l", "leg_r", "rose_blue", "rose_pink")


def extract_palette(img):
    opaque = img[..., 3] >= 128
    return np.unique(img[opaque][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    opaque = out[..., 3] >= 128
    if not np.any(opaque):
        out[..., 3] = 0
        out[..., :3] = 0
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
    out[~opaque, :3] = 0
    return out


def color_mask(img, colors):
    opaque = img[..., 3] >= 128
    m = np.zeros(opaque.shape, bool)
    for c in colors:
        m |= opaque & np.all(img[..., :3] == np.array(c, np.uint8), axis=2)
    return m


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        pts = []
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        out.append(pts)
    out.sort(key=len, reverse=True)
    return out


def grow(seed, allow, n):
    out = seed.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        nbr = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        out = out | (nbr & allow)
    return out


def segment(img):
    """Two bouquets, two shoes, everything else on the torso (cape included)."""
    opaque = img[..., 3] >= 128
    h, w = opaque.shape
    yy, xx = np.indices((h, w))
    pink = color_mask(img, PINK_COLS)
    blue = color_mask(img, BLUE_COLS)
    white = color_mask(img, (WHITE,))
    lav = color_mask(img, (LAV,))
    yellow = color_mask(img, (YEL,))
    blocked = white | lav | yellow

    pink_seed = np.zeros_like(opaque)
    pink_parts = components(pink)
    if not pink_parts or len(pink_parts[0]) < 100:
        raise SystemExit("pink bouquet missing")
    for y, x in pink_parts[0]:
        pink_seed[y, x] = True
    # Chest flecks are the smaller pink components. They stay on the torso.
    blue_seed = np.zeros_like(opaque)
    blue_parts = components(blue)
    if not blue_parts or len(blue_parts[0]) < 100:
        raise SystemExit("blue bouquet missing")
    for y, x in blue_parts[0]:
        blue_seed[y, x] = True

    allow_pink = opaque & (xx <= 40) & ~blocked & ~blue
    allow_blue = opaque & (xx >= 58) & ~blocked & ~pink
    rose_pink = grow(pink_seed, allow_pink | pink_seed, 2)
    rose_blue = grow(blue_seed, allow_blue | blue_seed, 2)

    leg_l = opaque & (yy >= 71) & (xx <= 46)
    leg_r = opaque & (yy >= 71) & (xx >= 53)
    rose_pink &= ~leg_l & ~leg_r
    rose_blue &= ~leg_l & ~leg_r & ~rose_pink
    leg_l &= ~rose_pink & ~rose_blue
    leg_r &= ~rose_pink & ~rose_blue & ~leg_l
    torso = opaque & ~rose_pink & ~rose_blue & ~leg_l & ~leg_r
    masks = {
        "rose_pink": rose_pink,
        "rose_blue": rose_blue,
        "leg_l": leg_l,
        "leg_r": leg_r,
        "torso": torso,
    }
    acc = np.zeros(opaque.shape, np.uint8)
    for m in masks.values():
        acc += m.astype(np.uint8)
    if int(acc.max()) > 1 or not np.array_equal(acc > 0, opaque):
        raise SystemExit("segmentation overlap or drop")
    counts = {k: int(v.sum()) for k, v in masks.items()}
    if counts["rose_pink"] < 220 or counts["rose_blue"] < 160:
        raise SystemExit(f"bouquet too small: {counts}")
    if counts["leg_l"] < 18 or counts["leg_r"] < 18:
        raise SystemExit(f"a shoe is missing: {counts}")
    if counts["leg_l"] > 80 or counts["leg_r"] > 80:
        raise SystemExit(f"a shoe ate the dress: {counts}")
    return masks


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


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
    """4x nearest. A full turn lands on the same pixels as angle 0. No |cos| squash."""
    angle_deg = angle_deg % 360.0
    if angle_deg > 180.0:
        angle_deg -= 360.0
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
    """Drop pixels whose four cardinal neighbours are empty."""
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
    pts_y = ys.astype(np.int32)
    pts_x = xs.astype(np.int32)
    step = 400
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None].astype(np.int32) - pts_y[None, :]) ** 2
        d += (tx[sl, None].astype(np.int32) - pts_x[None, :]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def socket_pivot(part, body):
    sel = part & dilate(body, 3)
    if not np.any(sel):
        sel = part
    ys, xs = np.nonzero(sel)
    return float(xs.mean()), float(ys.mean())


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


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
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm

    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr

    torso = placed["torso"]
    pivots = {}
    cushion = np.zeros_like(canvas)
    for name in ("rose_pink", "rose_blue", "leg_l", "leg_r"):
        pivots[name] = socket_pivot(placed[name], torso)
        zone = placed[name] & dilate(torso, 3)
        if not np.any(zone):
            continue
        fill = nearest_color(canvas, zone, torso)
        write = zone & (layers["torso"][..., 3] < 128)
        cushion[write] = fill[write]

    opy, opx = np.nonzero(canvas[..., 3] >= 128)
    center = (float(opx.mean()), float(opy.mean()))
    soles = []
    for name in ("leg_l", "leg_r"):
        fys, fxs = np.nonzero(placed[name])
        if len(fys) == 0:
            continue
        yb = int(fys.max())
        xb = float(np.median(fxs[fys >= yb - 1]))
        soles.append((xb, float(yb)))
    if len(soles) != 2:
        raise SystemExit("expected two soles")
    sole = ((soles[0][0] + soles[1][0]) * 0.5, max(soles[0][1], soles[1][1]))

    if flip:
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        cushion = np.ascontiguousarray(cushion[:, ::-1])
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
        sole = (CELL_W - 1 - sole[0], sole[1])

    rest = np.zeros_like(canvas)
    paste(rest, cushion)
    for name in DRAW:
        paste(rest, layers[name])
    facing = -1.0 if not flip else 1.0
    return {
        "layers": layers,
        "pivots": pivots,
        "cushion": cushion,
        "rest": rest,
        "center": center,
        "sole": sole,
        "facing": facing,
    }


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    p = dict(
        dx=0.0, dy=0.0,
        rose_pink=0.0, rose_blue=0.0, flare=0,
        lift_l=0.0, lift_r=0.0, foot_l=0.0, foot_r=0.0,
        spin=0.0, recoil=0.0, lean=0.0,
        rings=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["rose_pink"] = wave(f, 16, 7.0, phase=0.4)
        p["rose_blue"] = wave(f, 16, -6.0, phase=-0.5)
    elif action == "walk":
        step = wave(f, 16, 1.0, cycles=2.0)
        p["dx"] = step
        p["lift_l"] = max(0.0, step) * 4.0
        p["lift_r"] = max(0.0, -step) * 4.0
        p["foot_l"] = max(0.0, step) * 2.0
        p["foot_r"] = max(0.0, -step) * 2.0
        p["rose_pink"] = wave(f, 16, -16.0, cycles=2.0)
        p["rose_blue"] = wave(f, 16, 16.0, cycles=2.0)
    elif action == "attack":
        if f <= 4:
            ang = (0, 10, 16, 18, 12)[f]
            p["rose_pink"] = ang
            p["rose_blue"] = -ang
            p["flare"] = (0, 1, 2, 2, 1)[f]
            p["rings"] = 1 if f else 0
        elif f <= 10:
            p["spin"] = 60.0 * (f - 4)
        elif f == 11:
            p["lean"] = 10.0
            p["dx"] = 2.0
            p["burst"] = 1
        elif f == 12:
            p["lean"] = 6.0
            p["dx"] = 1.0
            p["burst"] = 1
        elif f == 13:
            p["lean"] = 4.0
        elif f == 14:
            p["lean"] = 2.0
    else:
        ang = (0, 16, 28, 34, 38, 34, 26, 18, 10, 5, 2, 0)
        back = (0, 2, 3, 4, 4, 3, 2, 2, 1, 1, 0, 0)
        p["recoil"] = ang[f]
        p["dx"] = -back[f]
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 5 else 0
    return p


def put_empty(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def flare_tip(dst, source, amount, col):
    """Extend that bouquet's own tip by a few pixels. No new limb."""
    if amount <= 0:
        return
    a = source[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return
    cx, cy = float(xs.mean()), float(ys.mean())
    d = (xs.astype(np.float32) - cx) ** 2 + (ys.astype(np.float32) - cy) ** 2
    i = int(d.argmax())
    tx, ty = float(xs[i]), float(ys[i])
    vx, vy = tx - cx, ty - cy
    norm = math.hypot(vx, vy) or 1.0
    ux, uy = vx / norm, vy / norm
    for k in range(1, int(amount) + 1):
        put_empty(dst, tx + ux * k, ty + uy * k, col)


def leading_point(torso, facing):
    a = torso[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return None
    y_cut = float(ys.min()) + 0.42 * float(ys.max() - ys.min())
    band = ys <= y_cut
    if not np.any(band):
        band = np.ones(len(ys), dtype=bool)
    sel = np.flatnonzero(band)
    i = int(sel[np.argmin(xs[band]) if facing < 0 else np.argmax(xs[band])])
    return int(xs[i]), int(ys[i])


def roar_rings(out, origin, facing):
    """Sparse forward arcs. Same dot set on every roar frame."""
    ox, oy = origin
    ox += facing * 3
    cols = (WHITE, YEL, PINK)
    for i, rad in enumerate((9, 14, 19)):
        col = cols[i]
        for deg in range(-50, 51, 25):
            ang = math.radians(deg)
            put_empty(out, ox + facing * rad * math.cos(ang), oy + rad * math.sin(ang) * 0.5, col)
    return out


def impact_burst(out, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    cx = (x0 - 3) if facing < 0 else (x1 + 3)
    cy = y0 + (y1 - y0) // 2
    for k in range(8):
        ang = k * math.pi / 4
        put_empty(out, cx + 6 * math.cos(ang), cy + 5 * math.sin(ang), WHITE)
        put_empty(out, cx + 10 * math.cos(ang), cy + 8 * math.sin(ang), YEL)
    return out


def sparks(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    cx = (x0 + x1) // 2
    back = -facing
    pts = (
        (cx + back * (6 + frame_idx), y0 + 6, WHITE),
        (cx + back * (4 + frame_idx), y0 + 14, YEL),
        (cx + back * 8, y0 + 22, PINK),
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


def moved_parts(rig, p):
    layers = rig["layers"]
    pivots = rig["pivots"]
    facing = rig["facing"]
    sgn = facing
    bdx, bdy = p["dx"] * facing, p["dy"]
    out = {
        "torso": shift(layers["torso"], bdx, bdy),
        "leg_l": shift(layers["leg_l"], bdx + p["foot_l"] * facing, bdy - p["lift_l"]),
        "leg_r": shift(layers["leg_r"], bdx + p["foot_r"] * facing, bdy - p["lift_r"]),
    }
    for name in ("rose_pink", "rose_blue"):
        spun = rotate_nearest(layers[name], p[name] * sgn, pivots[name])
        extra = 0.0
        if p["flare"]:
            # Push each bouquet away from the torso centre, in whole pixels.
            cx = pivots[name][0]
            extra = -p["flare"] if cx < rig["center"][0] else p["flare"]
        out[name] = shift(spun, bdx + extra, bdy)
    return out, bdx, bdy


def synth(rig, action, f, palette):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    facing = rig["facing"]
    sgn = facing
    part_move = any(abs(p[k]) >= 0.45 for k in ("rose_pink", "rose_blue", "lift_l", "lift_r", "foot_l", "foot_r", "dx", "dy", "flare"))
    spinning = (p["spin"] % 360.0) > 0.5
    recoiling = p["recoil"] > 0.45
    leaning = p["lean"] > 0.45
    if spinning or recoiling or leaning:
        if spinning:
            img = rotate_nearest(rig["rest"], p["spin"] * sgn, rig["center"])
        elif recoiling:
            img = rotate_nearest(rig["rest"], p["recoil"] * sgn, rig["sole"])
            img = shift(img, p["dx"] * facing, 0)
        else:
            img = rotate_nearest(rig["rest"], -p["lean"] * sgn, rig["center"])
            img = shift(img, p["dx"] * facing, 0)
        img = despeckle(img)
    elif part_move:
        moved, _, _ = moved_parts(rig, p)
        img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        paste(img, shift(rig["cushion"], p["dx"] * facing, p["dy"]))
        for name in DRAW:
            paste(img, moved[name])
        img = despeckle(img)
        if p["flare"]:
            flare_tip(img, moved["rose_pink"], p["flare"], PINK)
            flare_tip(img, moved["rose_blue"], p["flare"], BLUE)
    else:
        img = rig["rest"].copy()

    img = snap(img, palette)
    if p["flash"]:
        img = flash_white(img)
    if p["rings"]:
        mouth = leading_point(img, facing)
        if mouth:
            img = roar_rings(img, mouth, facing)
    if p["burst"]:
        img = impact_burst(img, facing)
    if p["sparks"]:
        img = sparks(img, f, facing)
    return snap(img, palette)


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


def south_attack_ratios(sheets):
    band = sheets["attack"][:CELL_H]
    counts = []
    for f in range(FRAMES["attack"]):
        tile = band[:, f * CELL_W:(f + 1) * CELL_W]
        counts.append(int((tile[..., 3] == 255).sum()))
    base = counts[0] if counts[0] else 1
    return counts, [c / base for c in counts]


def sprite_metrics(sheets):
    tile = sheets["idle"][:CELL_H, :CELL_W]
    ys = np.nonzero(tile[..., 3] == 255)[0]
    if len(ys) == 0:
        return 0, 0
    return int(ys.max() - ys.min() + 1), int(ys.max())


def component_sizes(img):
    a = img[..., 3] >= 128
    h, w = a.shape
    seen = np.zeros_like(a)
    sizes = []
    ys, xs = np.nonzero(a)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        n = 0
        while q:
            y, x = q.popleft()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes


def preview_image(sheets):
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 2, 4, 8, 12),
        "attack": (0, 3, 4, 6, 8, 10, 11, 13),
        "hurt": (0, 2, 4, 8),
    }
    cols = 8
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
    """South row, every frame, cropped so the spin and the two shoes are readable."""
    pad = 8
    scale = 2
    crops = []
    for act in ACTIONS:
        row = []
        band = sheets[act][:CELL_H]
        for f in range(FRAMES[act]):
            tile = band[:, f * CELL_W:(f + 1) * CELL_W]
            a = tile[..., 3] >= 128
            if not np.any(a):
                row.append(np.zeros((8, 8, 4), np.uint8))
                continue
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
    for fp in OUT_DIR.rglob("*"):
        if not fp.is_file():
            continue
        rel = fp.relative_to(OUT_DIR)
        if rel.parts[0] == "_work":
            continue
        dest = SYNC_OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fp, dest)
        n += 1
    return n


def placed_still(front, flip):
    opaque = front[..., 3] >= 128
    ys, xs = np.nonzero(opaque)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    placed = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    placed[dst_y:dst_y + sh, dst_x:dst_x + sw] = front[y0:y1, x0:x1]
    if flip:
        placed = np.ascontiguousarray(placed[:, ::-1])
    return placed


def differ_count(a_img, b_img):
    differ = np.any(a_img != b_img, axis=2) & ((a_img[..., 3] >= 128) | (b_img[..., 3] >= 128))
    return int(differ.sum())


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


def publish_dir8(sprite_h, foot_y):
    entry = {
        "dex": 407,
        "form": "normal",
        "variant": "roserade",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0407_roserade/normal",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir",
        "nf": 16,
        "cw": CELL_W,
        "ch": CELL_H,
        "w": CELL_W,
        "h": int(sprite_h),
        "fy": int(foot_y),
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
            raise SystemExit("0407_roserade missing from dir8.js")
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
        start = text.index('"id": "0407_roserade"')
        nxt = text.find('"id":', start + 10)
        end = len(text) if nxt < 0 else nxt
        block = text[start:end]
        if '"status":' not in block:
            raise SystemExit("queue status missing")
        block2 = re.sub(r'"status": "[^"]*"', f'"status": "{status}"', block, count=1)
        # Drop claim without leaving a trailing comma on the previous field.
        mid, n_mid = re.subn(r'\n\s*"claim": "[^"]*",', "", block2, count=1)
        if n_mid:
            block2 = mid
        else:
            block2 = re.sub(r',\s*\n\s*"claim": "[^"]*"', "", block2, count=1)
        if block2 == block and status == "done" and '"status": "done"' in block and '"claim"' not in block:
            pass
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


def part_preview(front, masks):
    col = {
        "torso": (90, 140, 90, 255),
        "rose_pink": (230, 60, 110, 255),
        "rose_blue": (70, 90, 230, 255),
        "leg_l": (240, 210, 40, 255),
        "leg_r": (40, 220, 220, 255),
    }
    img = np.zeros((*front.shape[:2], 4), np.uint8)
    for name in DRAW:
        img[masks[name]] = col[name]
    return Image.fromarray(np.repeat(np.repeat(img, 4, 0), 4, 1), "RGBA")


def qa_report(sheets, palette, digest, counts):
    master = np.concatenate([sheets[a] for a in ACTIONS], axis=0)
    opaque = master[..., 3] == 255
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)} if np.any(opaque) else set()
    off = sorted(used - pal)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    clip = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] == 255
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
    atk_counts, ratios = south_attack_ratios(sheets)
    h, fy = sprite_metrics(sheets)
    spin_parts = []
    for f in range(5, 11):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        spin_parts.append(component_sizes(tile)[:3])
    return {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": h,
        "foot_y": fy,
        "off_palette": len(off),
        "alphas": alphas,
        "edge_touching": clip,
        "attack_counts": atk_counts,
        "attack_ratios": [round(r, 4) for r in ratios],
        "attack_ratio_min": min(ratios),
        "attack_ratio_max": max(ratios),
        "spin_components": spin_parts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(NORMAL / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), flush=True)
    print("facing left; unflipped N NW W SW; flipped S SE E NE", flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    rest_w = synth(rigs["W"], "idle", 0, palette)
    rest_e = synth(rigs["E"], "idle", 0, palette)
    diff_w = differ_count(rest_w, placed_still(front, False))
    diff_e = differ_count(rest_e, placed_still(front, True))
    print("rest-pose pixels changed W", diff_w, "E", diff_e, flush=True)
    if diff_w or diff_e:
        raise SystemExit("idle frame 0 does not match the still")

    # Pink bouquet is screen-left on the unflipped still, screen-right after the east mirror.
    def cx(layer):
        xs = np.nonzero(layer[..., 3] >= 128)[1]
        return float(xs.mean()) if len(xs) else 0.0

    if cx(rigs["W"]["layers"]["rose_pink"]) >= cx(rigs["W"]["layers"]["rose_blue"]):
        raise SystemExit("unflipped bouquets are on the wrong sides")
    if cx(rigs["E"]["layers"]["rose_pink"]) <= cx(rigs["E"]["layers"]["rose_blue"]):
        raise SystemExit("east mirror did not turn the bouquets")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part_preview(front, masks).save(work / "parts.png")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 12)]
        want += [("attack", "W", f) for f in (0, 3, 5, 7, 10, 11, 15)]
        want += [("attack", "S", f) for f in (0, 6, 11)]
        want += [("attack", "E", f) for f in (0, 7)]
        want += [("hurt", "W", f) for f in (0, 4, 11)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
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
    write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions_dir = OUT_DIR / "actions"
    actions_dir.mkdir(exist_ok=True)
    preview.save(actions_dir / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    report = qa_report(sheets, palette, digest, counts)
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
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
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    ratios = report["attack_ratios"]
    print("SOUTH attack opaque ratios:", " ".join(f"{r:.4f}" for r in ratios), flush=True)
    print(
        f"SOUTH attack opaque min {report['attack_ratio_min']:.4f} max {report['attack_ratio_max']:.4f}",
        flush=True,
    )
    print(
        json.dumps(
            {k: report[k] for k in ("off_palette", "alphas", "edge_touching", "sprite_h", "foot_y", "spin_components")},
            ensure_ascii=False,
        ),
        flush=True,
    )

    meta = {
        "generator": "forge_roserade_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 407,
        "variant": "roserade",
        "form": "normal",
        "archetype": "biped",
        "facing_still": "left",
        "eye_mean_x": 45.92,
        "hair_mean_x": 49.46,
        "hair_max_x": 64,
        "unflipped": ["N", "NW", "W", "SW"],
        "flipped": ["S", "SE", "E", "NE"],
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": report["sprite_h"],
        "opaque_bottom": report["foot_y"],
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "parts": counts,
        "palette_colors": int(len(palette)),
        "off_palette_colors": report["off_palette"],
        "attack_ratios": ratios,
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
        or report["sprite_h"] < 40
    )
    if failed:
        print("QA failed", flush=True)
        if args.publish:
            update_queue("needs_fix")
            raise SystemExit("refusing to publish a failed sheet")
        return

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        neighbor = publish_dir8(report["sprite_h"], FOOT_Y)
        update_queue("done")
        print("published", VERSION, "neighbor", neighbor, flush=True)


if __name__ == "__main__":
    main()
