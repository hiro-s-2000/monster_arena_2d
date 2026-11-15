#!/usr/bin/env python3
"""Buizel (0418_buizel) 8-direction action sheet from the 96x96 front still.

Measured on front.png, not borrowed from another species:
  two planted feet (y>=68, split by the gap at x 40-45)
  two flotation tails on the rear (right runs starting at x>=56, y<=62),
  split at x=64 into the inner fork and the outer fork. Tails are not feet.
  two blue chest flippers, and one mouth (ink at y=40, x=45-46) under the eyes.
Eyes sit near x=42; tail yellow sits near x=64, so the still faces left.
Unflipped N NW W SW. Flipped S SE E NE, so E points screen-right.
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
NORMAL = ROOT / "monsterSprite/gen4/0418_buizel/normal"
OUT_DIR = NORMAL / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0418_buizel/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
FLIP_DIRS = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 176, 156
VERSION = "buizel_v2"
KEY = "0418_buizel"

INK = (24, 24, 24)
WHITE = (255, 255, 255)
CREAM = (222, 197, 131)
YELLOW = (246, 246, 180)
CHEEK = (255, 222, 0)


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
    """4x nearest-neighbor. 0 and 360 return the same pixels. No |cos| squash."""
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
    pad = np.pad(opaque, 1)
    n = (
        pad[:-2, 1:-1].astype(np.int16)
        + pad[2:, 1:-1]
        + pad[1:-1, :-2]
        + pad[1:-1, 2:]
    )
    out[opaque & (n == 0)] = 0
    return out


def fill_pinholes(img):
    """Fill enclosed holes of 1 or 2 pixels. Open gaps (feet, tail fork) stay open."""
    opaque = img[..., 3] >= 128
    trans = ~opaque
    h, w = trans.shape
    seen = np.zeros_like(trans)
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
    """Two feet, two rear flotation tails, two flippers, one existing mouth."""
    opaque = img[..., 3] >= 128
    rgb = img[..., :3]
    h, w = opaque.shape
    yy, xx = np.mgrid[0:h, 0:w]

    tail = np.zeros_like(opaque)
    for y in range(44, 63):
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
        for x0, x1 in runs:
            if x0 >= 56:
                tail[y, x0:x1 + 1] = True
    tail_inner = tail & (xx < 64)
    tail_outer = tail & (xx >= 64)

    foot_l = opaque & (yy >= 68) & (xx <= 42)
    foot_r = opaque & (yy >= 68) & (xx >= 46)
    blue = opaque & (
        ((rgb[..., 0] == 32) & (rgb[..., 1] == 98) & (rgb[..., 2] == 164))
        | ((rgb[..., 0] == 98) & (rgb[..., 1] == 189) & (rgb[..., 2] == 230))
    )
    arm_l = blue & (xx < 45)
    arm_r = blue & (xx >= 45)
    ink = opaque & (rgb[..., 0] == 24) & (rgb[..., 1] == 24) & (rgb[..., 2] == 24)
    cream = opaque & (rgb[..., 0] == 222) & (rgb[..., 1] == 197) & (rgb[..., 2] == 131)
    mouth = ink & (yy == 40) & (xx >= 44) & (xx <= 47)
    jaw = mouth | (cream & (yy == 41) & (xx >= 44) & (xx <= 47))

    parts = {
        "tail_inner": tail_inner,
        "tail_outer": tail_outer,
        "foot_l": foot_l,
        "foot_r": foot_r,
        "arm_l": arm_l,
        "arm_r": arm_r,
        "jaw": jaw,
    }
    if int(tail_inner.sum()) < 20 or int(tail_outer.sum()) < 15:
        raise SystemExit("buizel flotation tails did not separate")
    if int(foot_l.sum()) < 10 or int(foot_r.sum()) < 10:
        raise SystemExit("buizel feet missing")
    if int(jaw.sum()) < 4:
        raise SystemExit("buizel mouth missing")
    stacked = np.zeros(opaque.shape, np.uint8)
    for mask in parts.values():
        stacked += mask.astype(np.uint8)
    if int((stacked > 1).sum()):
        raise SystemExit("buizel parts overlap")
    torso = opaque & (stacked == 0)
    parts["torso"] = torso
    if int(stacked.sum() + torso.sum()) != int(opaque.sum()):
        raise SystemExit("buizel parts do not cover the still")
    return parts


def measure_facing(img):
    opaque = img[..., 3] >= 128
    rgb = img[..., :3]
    h, w = opaque.shape
    yy, xx = np.mgrid[0:h, 0:w]
    eye = opaque & (yy < 48) & (
        ((rgb[..., 0] == 255) & (rgb[..., 1] == 255) & (rgb[..., 2] == 255))
        | ((rgb[..., 0] == 106) & (rgb[..., 1] == 106) & (rgb[..., 2] == 106))
    )
    tail = opaque & (xx >= 56) & (yy <= 62) & (rgb[..., 0] == 246) & (rgb[..., 1] == 246)
    ey, ex = np.nonzero(eye)
    ty, tx = np.nonzero(tail)
    if len(ex) == 0 or len(tx) == 0:
        raise SystemExit("could not measure buizel face or tail")
    face_x = float(ex.mean())
    tail_x = float(tx.mean())
    if not face_x < tail_x:
        raise SystemExit(f"expected left-facing still, face_x={face_x:.1f} tail_x={tail_x:.1f}")
    return face_x, tail_x


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


def pivot_of(mask, side="left"):
    ys, xs = np.nonzero(mask)
    edge = xs.min() if side == "left" else xs.max()
    sel = xs <= edge + 1 if side == "left" else xs >= edge - 1
    return float(xs[sel].mean()), float(ys[sel].mean())


def build_rig(img, masks):
    placed, origin = place_still(img)
    moved = {name: move_mask(mask, origin) for name, mask in masks.items()}
    layers = {}
    for name, mask in moved.items():
        layer = np.zeros_like(placed)
        layer[mask] = placed[mask]
        layers[name] = layer

    torso = moved["torso"]
    feet = moved["foot_l"] | moved["foot_r"]
    tails = moved["tail_inner"] | moved["tail_outer"]
    arms = moved["arm_l"] | moved["arm_r"]
    ankle = torso & dilate(feet, 1)
    hip = tails & dilate(torso, 1)
    socket = np.zeros_like(placed)
    socket[ankle] = placed[ankle]
    hip_fill = nearest_color(placed, hip, torso)
    socket[hip] = hip_fill[hip]
    arm_fill = nearest_color(placed, arms, torso)
    socket[arms] = arm_fill[arms]

    opaque = placed[..., 3] >= 128
    ys, xs = np.nonzero(opaque)
    return {
        "layers": layers,
        "socket": socket,
        "placed": placed,
        "jaw_rest": layers["jaw"],
        "inner_pivot": pivot_of(moved["tail_inner"], "left"),
        "outer_pivot": pivot_of(moved["tail_outer"], "left"),
        "centroid": (float(xs.mean()), float(ys.mean())),
        "foot_pivot": (float(xs.mean()), float(ys.max())),
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
        "bdy": 0,
        "inner": 0.0,
        "outer": 0.0,
        "ldx": 0,
        "ldy": 0,
        "rdx": 0,
        "rdy": 0,
        "aldx": 0,
        "aldy": 0,
        "ardx": 0,
        "ardy": 0,
        "mdy": 0,
        "rings": 0,
        "impact": 0,
        "spin": 0.0,
        "hurt": 0.0,
        "hdx": 0,
        "sweat": 0,
    }
    if action == "idle":
        p["bdy"] = int(round(wave(f, n, 0.55)))
        p["inner"] = wave(f, n, 6.0, phase=0.4)
        p["outer"] = wave(f, n, 7.0, phase=0.4 + math.pi)
    elif action == "walk":
        p["bdy"] = int(round(wave(f, n, 0.55, cycles=2.0)))
        ll = step_lift(f, n, 0.0)
        rl = step_lift(f, n, math.pi)
        # Facing left: a lifted paw reaches toward the nose (negative x).
        p["ldy"] = -int(round(3.0 * ll))
        p["ldx"] = -int(round(ll))
        p["rdy"] = -int(round(3.0 * rl))
        p["rdx"] = -int(round(rl))
        p["aldy"] = -int(round(ll))
        p["aldx"] = -int(round(ll))
        p["ardy"] = -int(round(rl))
        p["ardx"] = -int(round(rl))
        p["inner"] = wave(f, n, 8.0, cycles=2.0, phase=0.5)
        p["outer"] = wave(f, n, 9.0, cycles=2.0, phase=0.5 + math.pi)
    elif action == "attack":
        muzzle_dy = (1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
        rings = (1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        p["mdy"] = muzzle_dy[f]
        p["rings"] = rings[f]
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
        if f in (11, 12):
            p["impact"] = 1
    else:
        angles = (0, 12, 22, 32, 38, 34, 24, 14, 8, 4, 2, 0)
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


def mouth_anchor(img, jaw_layer):
    opaque = jaw_layer[..., 3] >= 128
    if np.any(opaque):
        ys, xs = np.nonzero(opaque)
        return int(xs.min()), int(np.median(ys))
    opaque = img[..., 3] >= 128
    ys, xs = np.nonzero(opaque)
    if len(xs) == 0:
        return None
    y1 = int(ys.min() + 0.42 * (ys.max() - ys.min()))
    sel = ys <= y1
    x = int(xs[sel].min())
    y = int(np.median(ys[sel][xs[sel] <= x + 1]))
    return x, y


def roar_rings(img, mouth, strength):
    """Sparse arcs in the empty space in front of the muzzle. Facing left."""
    if mouth is None or strength <= 0:
        return img
    mx, my = mouth
    cols = (WHITE, YELLOW, CREAM)
    radii = (7, 11, 15)[:strength]
    for i, rad in enumerate(radii):
        col = cols[i]
        for deg in range(-48, 49, 24):
            ang = math.radians(deg)
            put_empty(img, mx - 3 - rad * math.cos(ang), my + rad * math.sin(ang) * 0.62, col)
    return img


def impact_dots(img, mouth):
    if mouth is None:
        return img
    mx, my = mouth
    for k in range(5):
        ang = (k - 2) * 0.55
        put_empty(img, mx - 8 - 5 * math.cos(ang), my + 7 * math.sin(ang), WHITE)
        put_empty(img, mx - 12 - 4 * math.cos(ang), my + 10 * math.sin(ang), CHEEK)
    return img


def sweat(img, anchor):
    if anchor is None:
        return img
    ax, ay = anchor
    for dx, dy in ((5, -8), (8, -4), (3, -12), (10, -9)):
        put_empty(img, ax + dx, ay + dy, WHITE)
    return img


def compose(rig, p):
    layers = rig["layers"]
    bdy = p["bdy"]
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(body, rig["socket"])
    paste(body, shift(layers["torso"], 0, bdy))
    paste(body, shift(layers["arm_l"], p["aldx"], bdy + p["aldy"]))
    paste(body, shift(layers["arm_r"], p["ardx"], bdy + p["ardy"]))
    inner = rotate_nearest(layers["tail_inner"], p["inner"], rig["inner_pivot"])
    outer = rotate_nearest(layers["tail_outer"], p["outer"], rig["outer_pivot"])
    paste(body, shift(inner, 0, bdy))
    paste(body, shift(outer, 0, bdy))
    paste(body, shift(layers["foot_l"], p["ldx"], p["ldy"]))
    paste(body, shift(layers["foot_r"], p["rdx"], p["rdy"]))
    jaw = shift(layers["jaw"], 0, bdy + p["mdy"])
    if p["mdy"]:
        vacated = (shift(layers["jaw"], 0, bdy)[..., 3] >= 128) & (body[..., 3] < 128)
        body[vacated, :3] = INK
        body[vacated, 3] = 255
    paste(body, jaw)
    if abs(p["inner"]) >= 0.45 or abs(p["outer"]) >= 0.45:
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
        jaw_now = img
    elif p["hurt"] > 0.45 or p["hdx"]:
        leaned = finish_body(rig["placed"], rig["foot_pivot"], p["hurt"])
        img = shift(leaned, p["hdx"], 0)
        jaw_now = img
    else:
        img = compose(rig, p)
        jaw_now = shift(rig["jaw_rest"], 0, p["bdy"] + p["mdy"])
    img = snap(img, palette)
    anchor = mouth_anchor(img, jaw_now if p["spin"] <= 0.45 and p["hurt"] <= 0.45 else img)
    if p["rings"]:
        img = roar_rings(img, anchor, p["rings"])
    if p["impact"]:
        img = impact_dots(img, anchor)
    if p["sweat"]:
        opaque = img[..., 3] >= 128
        ys, xs = np.nonzero(opaque)
        top = int(ys.min())
        band = xs[ys <= top + 4]
        img = sweat(img, (int(np.median(band)) + 4, top))
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


def _tight_rows(sheets, row_index):
    pad = 4
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            y0 = row_index * CELL_H
            tile = sheets[act][y0:y0 + CELL_H, f * CELL_W:(f + 1) * CELL_W]
            opaque = tile[..., 3] >= 128
            if not np.any(opaque):
                continue
            ys, xs = np.nonzero(opaque)
            top = max(0, int(ys.min()) - pad)
            bot = min(CELL_H, int(ys.max()) + 1 + pad)
            left = max(0, int(xs.min()) - pad)
            right = min(CELL_W, int(xs.max()) + 1 + pad)
            row.append(tile[top:bot, left:right])
        crops.append(row)
    return crops


def _pack(crops, scale):
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
    if scale != 1:
        bg = np.repeat(np.repeat(bg, scale, axis=0), scale, axis=1)
    return Image.fromarray(bg, "RGBA")


def preview_image(sheets):
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 4, 8, 12),
        "attack": (2, 4, 7, 11),
        "hurt": (2, 4, 6, 9),
    }
    crops = []
    for act in ACTIONS:
        row = []
        for f in samples[act]:
            tile = sheets[act][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            opaque = tile[..., 3] >= 128
            ys, xs = np.nonzero(opaque)
            pad = 6
            top = max(0, int(ys.min()) - pad)
            bot = min(CELL_H, int(ys.max()) + 1 + pad)
            left = max(0, int(xs.min()) - pad)
            right = min(CELL_W, int(xs.max()) + 1 + pad)
            row.append(tile[top:bot, left:right])
        crops.append(row)
    return _pack(crops, 3)


def motion_contact(sheets):
    return _pack(_tight_rows(sheets, 0), 2)


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file() and not fp.name.startswith("_"):
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
        "dex": 418,
        "form": "normal",
        "variant": "buizel",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0418_buizel/normal",
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
    DIR8_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with DIR8_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not before.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        keys = re.findall(r'"(\d{4}_[^"]+)":\{', before)
        if KEY not in keys:
            raise SystemExit("0418_buizel missing from dir8.js")
        idx = keys.index(KEY)
        neighbor = keys[idx + 1] if idx + 1 < len(keys) else keys[idx - 1]
        nb_s, nb_e = _entry_span(before, neighbor)
        neighbor_blob = before[nb_s:nb_e]
        s, e = _entry_span(before, KEY)
        after = before[:s] + f'"{KEY}":' + blob + before[e:]
        if neighbor_blob not in after:
            raise SystemExit("neighbor dir8 key changed")
        masked_before = before[:s] + before[e:]
        s2, e2 = _entry_span(after, KEY)
        masked_after = after[:s2] + after[e2:]
        if masked_before != masked_after:
            raise SystemExit("dir8 changed outside 0418_buizel")
        body = after[len(prefix):].strip()
        if body.endswith(";"):
            body = body[:-1].strip()
        json.loads(body)
        if after.count(f'"v":"{VERSION}"') != 1:
            raise SystemExit("dir8 version stamp collided")
        if re.findall(r'"(\d{4}_[^"]+)":\{', after) != keys:
            raise SystemExit("dir8 key set changed")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(after)
        tmp.replace(DIR8)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return neighbor


def update_queue(status):
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(QUEUE.read_text())
        found = False
        for item in data["items"]:
            if item.get("id") != KEY:
                continue
            item["status"] = status
            item.pop("claim", None)
            found = True
            break
        if not found:
            raise SystemExit("queue item 0418_buizel missing")
        ids = [item.get("id") for item in data["items"]]
        if ids.count(KEY) != 1:
            raise SystemExit("queue id collision")
        raw = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        tmp = QUEUE.with_suffix(".json.tmp")
        tmp.write_text(raw, encoding="utf-8")
        tmp.replace(QUEUE)
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
    face_x, tail_x = measure_facing(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print(
        "parts", counts, "palette", len(palette),
        f"face_x {face_x:.2f} tail_x {tail_x:.2f} facing left",
        "feet 2 tails 2",
        flush=True,
    )
    rig = build_rig(front, masks)

    rest = synth(rig, "idle", 0, palette, flip=False)
    if not np.array_equal(rest, rig["placed"]):
        differ = np.any(rest != rig["placed"], axis=2)
        raise SystemExit(f"idle frame 0 does not match the still ({int(differ.sum())} pixels)")
    # Frame 10 used to be the end of a full turn, so it matched the still.
    # The strike now holds the tails out through that frame. Frame 15 recovers.
    for act, frame in (("walk", 0), ("attack", 15), ("hurt", 0), ("hurt", 11)):
        got = synth(rig, act, frame, palette, flip=False)
        if not np.array_equal(got, rig["placed"]):
            differ = int(np.any(got != rig["placed"], axis=2).sum())
            raise SystemExit(f"{act} frame {frame} does not match the still ({differ} pixels)")
    flipped = synth(rig, "idle", 0, palette, flip=True)
    if not np.array_equal(flipped, rig["placed"][:, ::-1]):
        raise SystemExit("flipped idle frame 0 is not the mirrored still")
    print("rest-pose pixels changed 0", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (180, 100, 40),
        "tail_inner": (40, 90, 220),
        "tail_outer": (80, 220, 220),
        "foot_l": (40, 200, 80),
        "foot_r": (230, 200, 40),
        "arm_l": (220, 40, 180),
        "arm_r": (255, 140, 180),
        "jaw": (255, 30, 30),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", False, f) for f in (0, 4, 8, 12)]
        want += [("walk", False, f) for f in (0, 2, 4, 6, 8, 10)]
        want += [("attack", False, f) for f in (0, 2, 4, 5, 7, 10, 11, 15)]
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
        crops = []
        row = []
        for tile in tiles:
            opaque = tile[..., 3] >= 128
            ys, xs = np.nonzero(opaque)
            pad = 4
            crop = tile[max(0, int(ys.min()) - pad):int(ys.max()) + 1 + pad,
                        max(0, int(xs.min()) - pad):int(xs.max()) + 1 + pad]
            row.append(crop)
            if len(row) == 8:
                crops.append(row)
                row = []
        if row:
            crops.append(row)
        _pack(crops, 3).save(work / "samples_zoom.png")
        print("samples", work / "samples_zoom.png", flush=True)
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
    report["face_x"] = round(face_x, 2)
    report["tail_x"] = round(tail_x, 2)
    report["facing"] = "left"
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
        "generator": "forge_buizel_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 418,
        "archetype": "biped_flotation_tails",
        "feet": 2,
        "flotation_tails": 2,
        "mouth": True,
        "facing": "left",
        "face_x": round(face_x, 2),
        "tail_x": round(tail_x, 2),
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
    spin_bad = any(not sizes or sizes[0] < 700 for sizes in report["spin_components"])
    if spin_bad:
        failed = True
        print("spin body is not one piece", report["spin_components"], flush=True)
    if failed:
        print("QA failed", flush=True)
        if args.publish:
            update_queue("needs_fix")
            raise SystemExit("refusing to publish a failed sheet")
        return

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        neighbor = publish_dir8(report["sprite_h"])
        update_queue("done")
        print("published", VERSION, "neighbor", neighbor, flush=True)


if __name__ == "__main__":
    main()
