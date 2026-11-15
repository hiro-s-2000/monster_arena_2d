#!/usr/bin/env python3
"""Staravia (0397) 8-direction sheet from this front.png only.

Measured on normal/front.png (96x96, alpha already 0 or 255, 12 colours):
  opaque bbox x32-68 y21-72 (37x52), 1071 pixels, one component.
  Centroid about (48.4, 48.9). Max radius about 28.4.
  Faces left: beak tip x=38, eye-mask median x=50.
  Beak is the orange/brown cluster x38-43 y38-43. Lower mandible is y>=41
  and x>=40, plus y>=42. That is the mouth. No second beak is drawn.
  Two crests sit at y<=30 (92 px, one piece, with the existing gap).
  Two feet separate at y>=64: near x34-48 (68), far x49-59 (57).
  One rear fan, the right-hand run on y38-45 (60 px), touches the body
  only on y46. No second wing is drawn. Folded wing pixels stay on the body.

Idle frame 0 is the still. Walk steps the two feet. Attack is a beak roar
with rings (0-4), an in-place 360 spin (5-10), impact (11-12), recover.
Hurt recoils at most 38 degrees. No row-squash. front/back png/gif are not written.
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
SPECIES = ROOT / "monsterSprite/gen4/0397_staravia/normal"
OUT_DIR = SPECIES / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0397_staravia/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
PY = FORGE / ".venv/bin/python"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left, so the east group is the mirror. E looks screen-right.
FACE_RIGHT = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "staravia_v2"
KEY = "0397_staravia"

BLACK = (16, 16, 16)
PLUM = (57, 49, 57)
DGRAY = (57, 57, 57)
GRAY = (90, 82, 90)
LGRAY = (180, 180, 180)
BROWN = (123, 65, 32)
ORANGE = (189, 98, 0)
BRIGHT = (255, 148, 41)
WHITE = (255, 255, 255)
BEAK_COLOURS = {BROWN, ORANGE, BRIGHT}


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
    out = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        q = deque([(int(y), int(x))])
        seen[y, x] = True
        pts = []
        while q:
            cy, cx = q.popleft()
            pts.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        out.append(np.array(pts, np.int32))
    out.sort(key=len, reverse=True)
    return out


def colour_mask(img, colours):
    rgb = img[..., :3]
    a = img[..., 3] >= 128
    m = np.zeros(a.shape, bool)
    for c in colours:
        m |= a & np.all(rgb == np.array(c, np.uint8), axis=-1)
    return m


def segment(img):
    """Partition this still. No extra wing, foot, or mouth."""
    opaque = img[..., 3] >= 128
    h, w = opaque.shape
    yy, xx = np.mgrid[0:h, 0:w]
    beak = colour_mask(img, BEAK_COLOURS) & (yy <= 46) & (xx <= 48)
    # Lower mandible of the existing beak. Upper bright-orange tip stays on the head.
    jaw = (beak & (yy >= 42)) | (beak & (yy == 41) & (xx >= 40))

    crest = opaque & (yy <= 30)

    # Right-hand run on the rows where a gap separates the rear fan from the chest.
    fan = np.zeros_like(opaque)
    for y in range(38, 46):
        xs = np.nonzero(opaque[y])[0]
        if len(xs) == 0:
            continue
        runs = []
        start = prev = int(xs[0])
        for x in xs[1:]:
            x = int(x)
            if x == prev + 1:
                prev = x
            else:
                runs.append((start, prev))
                start = prev = x
        runs.append((start, prev))
        if len(runs) >= 2:
            s, e = runs[-1]
            fan[y, s:e + 1] = True

    feet_zone = opaque & (yy >= 64) & ~jaw & ~fan & ~crest
    foot_parts = components(feet_zone)
    if len(foot_parts) < 2:
        raise SystemExit("expected two feet below y64")
    foot_parts = foot_parts[:2]
    foot_parts.sort(key=lambda p: float(p[:, 1].mean()))
    near = np.zeros_like(opaque)
    far = np.zeros_like(opaque)
    near[foot_parts[0][:, 0], foot_parts[0][:, 1]] = True
    far[foot_parts[1][:, 0], foot_parts[1][:, 1]] = True

    body = opaque & ~jaw & ~crest & ~fan & ~near & ~far

    def rows_from_top(mask, n):
        ys = np.nonzero(mask)[0]
        if len(ys) == 0:
            return np.zeros_like(mask)
        return mask & (yy <= int(ys.min()) + n - 1)

    def rows_from_bottom(mask, n):
        ys = np.nonzero(mask)[0]
        if len(ys) == 0:
            return np.zeros_like(mask)
        return mask & (yy >= int(ys.max()) - n + 1)

    # Socket copies so a 1px lift does not open a hole. Not a new limb.
    hip = rows_from_top(near, 2) | rows_from_top(far, 2)
    crest_base = rows_from_bottom(crest, 1)
    fan_base = rows_from_bottom(fan, 1)
    body = body | hip | crest_base | fan_base
    return {
        "body": body, "jaw": jaw, "crest": crest, "fan": fan,
        "near": near, "far": far,
    }


def bbox_mask(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def place_geometry(img):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    cx, cy = float(xs.mean()), float(ys.mean())
    pivot = (dst_x + (cx - x0), dst_y + (cy - y0))
    return y0, y1, x0, x1, dst_y, dst_x, pivot


def build_layers(img, masks):
    y0, y1, x0, x1, dst_y, dst_x, pivot = place_geometry(img)
    sh, sw = y1 - y0, x1 - x0
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    layers = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr
    return layers, pivot, canvas


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
    """4x nearest. Positive degrees turn clockwise on screen."""
    if abs(angle_deg) % 360 < 0.01 or not np.any(layer[..., 3] >= 128):
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
    """Drop only pixels whose 4-neighbourhood is empty."""
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
    step = 400
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None] - pts[None, :, 0]) ** 2 + (tx[sl, None] - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def fill_pinholes(img, max_size=4):
    a = img[..., 3] >= 128
    empty = ~a
    h, w = a.shape
    seen = np.zeros_like(empty)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if empty[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if empty[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and empty[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    holes = empty & ~seen
    if not np.any(holes):
        return img
    out = img.copy()
    for pts in components(holes):
        if len(pts) > max_size:
            continue
        mask = np.zeros_like(holes)
        mask[pts[:, 0], pts[:, 1]] = True
        fill = nearest_color(out, mask, a)
        out[mask] = fill[mask]
    return out


def bridge_parts(img, max_gap=2):
    """Rejoin a fragment the spin pulled a couple of pixels off the body."""
    a = img[..., 3] >= 128
    parts = components(a)
    if len(parts) <= 1:
        return img
    main = np.zeros_like(a)
    main[parts[0][:, 0], parts[0][:, 1]] = True
    out = img.copy()
    changed = False
    for pts in parts[1:]:
        gap_lim = 3 if len(pts) > 12 else max_gap
        frag = np.zeros_like(a)
        frag[pts[:, 0], pts[:, 1]] = True
        edge = frag.copy()
        grown = None
        for _ in range(gap_lim):
            p = np.pad(edge, 1)
            dil = edge | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
            if np.any(dil & main):
                grown = dil & ~a
                break
            edge = dil & ~main
        else:
            continue
        if grown is None or not np.any(grown):
            main = main | frag
            continue
        fill = nearest_color(out, grown, out[..., 3] >= 128)
        out[grown] = fill[grown]
        changed = True
        main = main | grown | frag
        a = out[..., 3] >= 128
    return out if changed else img


def repair(img):
    img = fill_pinholes(img)
    img = bridge_parts(img)
    img = despeckle(img)
    return img


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
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


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return True
    return False


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def iwave(f, n, amp=1.0, phase=0.0, cycles=1.0):
    return int(round(wave(f, n, amp, phase, cycles)))


def foot_window(f, n, slot, slots=2, cycles=2):
    ph = (cycles * f / n) % 1.0
    width = 1.0 / slots
    start = slot * width
    if ph < start or ph > start + width:
        return 0.0
    t = (ph - start) / width
    return math.sin(math.pi * t)


def pose(action, f):
    n = FRAMES[action]
    p = dict(bdx=0, bdy=0, near=(0, 0), far=(0, 0), crest=0, fan=0, jaw=0,
             spin=0.0, hurt=0.0, rings=0, ring=0, lines=0, burst=0, flash=0, sparks=0)
    if action == "idle":
        p["bdy"] = iwave(f, n, 1.0)
        p["crest"] = iwave(f, n, 1.0, phase=0.4)
        p["fan"] = iwave(f, n, 1.0, phase=-0.8)
    elif action == "walk":
        lifts = []
        swings = []
        for slot in (0, 1):
            s = foot_window(f, n, slot)
            lifts.append(-int(round(3 * s)))
            # Forward is -x on this left-facing still.
            swings.append(-int(round(2 * s)))
        # Foot offsets are relative to the body. A planted foot stays down.
        p["near"] = (swings[0], lifts[0])
        p["far"] = (swings[1], lifts[1])
        p["fan"] = iwave(f, n, 1.0, cycles=2.0)
        p["crest"] = iwave(f, n, 1.0, phase=0.6, cycles=2.0)
    elif action == "attack":
        jaw = [1, 2, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        rings = [1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        age = [0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        p["jaw"] = jaw[f]
        p["rings"] = rings[f]
        p["ring"] = age[f]
        if 5 <= f <= 12:
            # Beak stays open. The near foot uses the walk kick (2px forward, 3px up).
            # Forward is -x on this left-facing still. The body does not spin.
            k = (0.45, 0.8, 1.0, 1.0, 0.75, 0.4, 0.55, 0.25)[f - 5]
            p["jaw"] = 3
            p["fan"] = 1 if f % 2 == 0 else -1
            p["crest"] = -1 if f % 2 == 0 else 1
            if f <= 10:
                p["near"] = (int(round(-2 * k)), int(round(-3 * k)))
        if 5 <= f <= 9:
            p["lines"] = 3
        if f in (11, 12):
            p["burst"] = 7 if f == 11 else 5
    else:
        env = math.sin(math.pi * f / (n - 1))
        p["hurt"] = 38.0 * env
        p["bdx"] = int(round(4 * env))
        p["bdy"] = int(round(-1 * env))
        p["jaw"] = 2 if 2 <= f <= 5 else (1 if f in (1, 6) else 0)
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def beak_tip(layers, bdx, bdy):
    """Leftmost pixel of the upper beak, which stays on the body."""
    m = layers["body"][..., 3] >= 128
    # Upper beak colours live just above the jaw. Use the jaw's rest bbox as a guide.
    jy, jx = np.nonzero(layers["jaw"][..., 3] >= 128)
    if len(jx) == 0:
        return None
    y0 = int(jy.min()) - 4
    y1 = int(jy.max()) + 2
    x1 = int(jx.max()) + 2
    region = m.copy()
    region[:max(0, y0 + bdy), :] = False
    region[min(region.shape[0], y1 + bdy + 1):, :] = False
    region[:, min(region.shape[1], x1 + bdx + 1):] = False
    ys, xs = np.nonzero(region)
    if len(xs) == 0:
        return int(jx.min()) + bdx, int(np.median(jy)) + bdy
    tip = int(xs.min())
    return tip, int(np.median(ys[xs <= tip + 1]))


def roar_rings(out, tip, age, strength):
    """Rings leave the beak toward screen-left. This still faces left."""
    mx, my = tip
    cols = (WHITE, LGRAY, BRIGHT)
    for r_i in range(strength):
        rad = 5 + age * 2 + r_i * 4
        col = cols[r_i % 3]
        for deg in range(-42, 43, 14):
            ang = math.radians(deg)
            put(out, mx - rad * math.cos(ang), my + rad * math.sin(ang) * 0.7, col)
    return out


def speed_lines(out, n, seed):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    rear = x1 - 1
    for k in range(n):
        y = int(rng.integers(y0 + 6, max(y0 + 7, y1 - 6)))
        length = 4
        gap = 3 + k
        for i in range(length):
            put(out, rear + gap + i, y, WHITE if i < 2 else LGRAY)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    cols = (WHITE, BRIGHT, LGRAY)
    for k in range(6):
        ang = k * math.pi / 3
        for r in range(2, size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), cols[r % 3])
    put(out, cx, cy, WHITE)
    return out


def sparks(out, frame_idx):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    d = frame_idx
    cx = (x0 + x1) // 2
    pts = [(x1 + 3 + d, y0 + 8), (x0 - 3 - d, y0 + 12), (cx, y0 - 3 - d)]
    for sx, sy in pts:
        put(out, sx, sy, WHITE)
        put(out, sx + 1, sy, BRIGHT)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., 0].astype(np.int16) + out[..., 1] + out[..., 2] < 80)
    out[a & ~dark, :3] = WHITE
    return out


def compose(layers, p):
    """Integer part shifts. Zero pose reproduces the still."""
    bdx, bdy = p["bdx"], p["bdy"]
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(img, shift(layers["body"], bdx, bdy))
    paste(img, shift(layers["fan"], bdx, bdy + p["fan"]))
    paste(img, shift(layers["far"], bdx + p["far"][0], bdy + p["far"][1]))
    paste(img, shift(layers["near"], bdx + p["near"][0], bdy + p["near"][1]))
    paste(img, shift(layers["crest"], bdx, bdy + p["crest"]))
    jaw_rest = shift(layers["jaw"], bdx, bdy)
    jaw_open = shift(layers["jaw"], bdx, bdy + p["jaw"])
    if p["jaw"]:
        vacated = (jaw_rest[..., 3] >= 128) & (img[..., 3] < 128)
        img[vacated, :3] = BLACK
        img[vacated, 3] = 255
    paste(img, jaw_open)
    return img


def synth(layers, pivot, canvas, action, f, palette):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = abs(p["spin"]) > 0.01
    recoiling = abs(p["hurt"]) > 0.01
    if spinning or recoiling:
        # One bitmap. Crest, fan, beak, and both feet rotate together.
        img = shift(canvas, p["bdx"], p["bdy"])
        if recoiling and p["jaw"]:
            jaw_rest = shift(layers["jaw"], p["bdx"], p["bdy"])
            jaw_open = shift(layers["jaw"], p["bdx"], p["bdy"] + p["jaw"])
            held = jaw_rest[..., 3] >= 128
            img[held] = 0
            img[held, :3] = BLACK
            img[held, 3] = 255
            paste(img, jaw_open)
        ang = p["spin"] if spinning else p["hurt"]
        origin = (pivot[0] + p["bdx"], pivot[1] + p["bdy"])
        img = rotate_nearest(img, ang, origin)
        img = repair(img)
    elif all(p[k] == 0 for k in ("bdx", "bdy", "crest", "fan", "jaw")) and p["near"] == (0, 0) and p["far"] == (0, 0):
        img = canvas.copy()
    else:
        img = compose(layers, p)
    img = snap(img, palette)
    if p["lines"] and spinning:
        img = speed_lines(img, p["lines"], seed=f * 17 + 3)
    if p["rings"] and not spinning:
        tip = beak_tip(layers, p["bdx"], p["bdy"])
        if tip:
            img = roar_rings(img, tip, p["ring"], p["rings"])
    if p["burst"]:
        tip = beak_tip(layers, 0, 0)
        if tip:
            img = impact_burst(img, (tip[0] - 4, tip[1]), p["burst"])
    if p["flash"]:
        img = flash_white(img)
    if p["sparks"]:
        img = sparks(img, f)
    return snap(img, palette)


def facing_flip(img, direction):
    if direction in FACE_RIGHT:
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
    """South row, enough frames to read the roar, the spin, and the recoil."""
    samples = {
        "idle": (0, 2, 4, 8, 12, 15),
        "walk": (0, 2, 4, 6, 10, 14),
        "attack": (0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 15),
        "hurt": (0, 1, 3, 5, 6, 8, 11),
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
    pad = 6
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


def sync_outputs():
    """Copy sheet files only. _work stays on the revenge side."""
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def main_component_fraction(img):
    a = img[..., 3] >= 128
    parts = components(a)
    if not parts:
        return 0.0, 0, 0
    total = int(a.sum())
    return len(parts[0]) / max(1, total), len(parts), total


def south_attack_ratios(sheets):
    ratios = []
    counts = []
    base = None
    for f in range(FRAMES["attack"]):
        tile = sheets["attack"][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        c = int((tile[..., 3] >= 128).sum())
        counts.append(c)
        if f == 0:
            base = c
        ratios.append(c / max(1, base))
    return counts, ratios


def qa_report(master, sheets, palette, frames_dir, digest, counts, sprite_h, skip_qa):
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = np.unique(master[..., 3])
    bad_alpha = [int(v) for v in alphas if int(v) not in (0, 255)]
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{d}:{f}")
    atk_counts, atk_ratios = south_attack_ratios(sheets)
    report = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_h,
        "off_palette_colors": len(off),
        "bad_alpha": bad_alpha,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "south_attack_opaque": atk_counts,
        "south_attack_ratios": [round(r, 4) for r in atk_ratios],
        "south_attack_ratio_min": round(min(atk_ratios), 4),
        "south_attack_ratio_max": round(max(atk_ratios), 4),
    }
    if not skip_qa and GAIT_LOOP.exists():
        qa = OUT_DIR / "_work" / "qa"
        qa.mkdir(parents=True, exist_ok=True)
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                report[f"gait_{act}"] = "skipped_existing_output"
                continue
            subprocess.run(
                [str(PY), "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SPECIES / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    opaque = front[..., 3] >= 128
    exclusive = {k: masks[k].copy() for k in ("jaw", "crest", "fan", "near", "far")}
    # Body carries socket copies. Coverage uses the body pixels that are not copies.
    body_only = masks["body"].copy()
    for k, m in exclusive.items():
        body_only &= ~m
    covered = body_only | exclusive["jaw"] | exclusive["crest"] | exclusive["fan"] | exclusive["near"] | exclusive["far"]
    missing = int((opaque & ~covered).sum())
    extra = int((covered & ~opaque).sum())
    overlap = 0
    names = ["body_only", "jaw", "crest", "fan", "near", "far"]
    parts_ex = {"body_only": body_only, **exclusive}
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            overlap += int((parts_ex[n1] & parts_ex[n2]).sum())
    counts = {k: int(v.sum()) for k, v in masks.items()}
    counts["missing"] = missing
    counts["extra"] = extra
    counts["overlap"] = overlap
    print("parts", counts, "palette", len(palette), flush=True)
    if missing or extra or overlap:
        raise SystemExit("segmentation does not partition the still")

    beak = colour_mask(front, BEAK_COLOURS)
    by, bx = np.nonzero(beak & (np.arange(96)[:, None] <= 46))
    eye = colour_mask(front, {BLACK}) & (np.arange(96)[:, None] >= 34) & (np.arange(96)[:, None] <= 42)
    eye = eye & (np.arange(96)[None, :] >= 44) & (np.arange(96)[None, :] <= 52)
    ey, ex = np.nonzero(eye)
    if len(bx) and len(ex):
        print("facing beak_tip_x", int(bx.min()), "eye_x", round(float(np.median(ex)), 1), flush=True)

    layers, pivot, canvas = build_layers(front, masks)
    print("pivot", [round(v, 2) for v in pivot], flush=True)

    rest = synth(layers, pivot, canvas, "idle", 0, palette)
    differ = np.any(rest != canvas, axis=2) & ((rest[..., 3] >= 128) | (canvas[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")
    rebuilt = compose(layers, pose("idle", 0))
    differ_b = np.any(rebuilt != canvas, axis=2) & ((rebuilt[..., 3] >= 128) | (canvas[..., 3] >= 128))
    print("rebuilt rest pixels changed", int(differ_b.sum()), flush=True)
    if int(differ_b.sum()) != 0:
        raise SystemExit("part composite does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "body": (90, 90, 98), "fan": (40, 90, 220), "near": (220, 60, 60),
        "far": (240, 200, 40), "crest": (40, 180, 80), "jaw": (255, 148, 41),
    }
    # Show exclusive body so socket copies do not hide the parts.
    show = {"body": body_only, "fan": masks["fan"], "near": masks["near"],
            "far": masks["far"], "crest": masks["crest"], "jaw": masks["jaw"]}
    for name, col in part_cols.items():
        part[show[name], :3] = col
        part[show[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", f) for f in (0, 4, 8, 12)]
        want += [("walk", f) for f in (0, 2, 6, 10, 14)]
        want += [("attack", f) for f in (0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 15)]
        want += [("hurt", f) for f in (0, 1, 3, 5, 8, 11)]
        tiles = [synth(layers, pivot, canvas, act, f, palette) for act, f in want]
        for act, f in (("attack", 5), ("attack", 6), ("attack", 7), ("attack", 8), ("hurt", 5)):
            img = synth(layers, pivot, canvas, act, f, palette)
            frac, n, total = main_component_fraction(img)
            bb = body_box(img)
            hgt = (bb[3] - bb[1] + 1) if bb else 0
            wid = (bb[2] - bb[0] + 1) if bb else 0
            print(f"cc {act}:{f} main {frac:.3f} parts {n} opaque {total} box {wid}x{hgt}", flush=True)
        base = int((canvas[..., 3] >= 128).sum())
        print("sample attack ratios vs still", flush=True)
        for f in range(16):
            img = synth(layers, pivot, canvas, "attack", f, palette)
            c = int((img[..., 3] >= 128).sum())
            print(f"  {f:02d} {c} {c / base:.4f}", flush=True)
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
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if key in state.get("done", []) and fp.exists() and fp.stat().st_size > 0:
                    continue
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest and not args.force:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                img = facing_flip(synth(layers, pivot, canvas, act, f, palette), d)
                Image.fromarray(img, "RGBA").save(tmp)
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

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    west = sheets["idle"][6 * CELL_H:7 * CELL_H, :CELL_W]
    differ = np.any(west != canvas, axis=2) & ((west[..., 3] >= 128) | (canvas[..., 3] >= 128))
    print("W idle0 pixels changed", int(differ.sum()), "sprite_h", sprite_h, flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("west idle 0 does not match the still")

    counts_a, ratios = south_attack_ratios(sheets)
    print("south attack opaque", counts_a, flush=True)
    print("south attack ratios", " ".join(f"{r:.4f}" for r in ratios), flush=True)
    print("ratio min", f"{min(ratios):.4f}", "max", f"{max(ratios):.4f}", flush=True)
    bad = [i for i, r in enumerate(ratios) if r < 0.90 or r > 1.08]
    if bad:
        raise SystemExit(f"south attack opaque ratio out of range at frames {bad}")

    for f in range(5, 11):
        tile = sheets["attack"][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        frac, n, total = main_component_fraction(tile)
        bb = body_box(tile)
        hgt = (bb[3] - bb[1] + 1) if bb else 0
        print(f"spin S:{f} main {frac:.3f} parts {n} opaque {total} h {hgt}", flush=True)
        if frac < 0.90 or hgt < 24:
            raise SystemExit(f"spin frame {f} shattered or squashed")

    report = qa_report(master, sheets, palette, frames_dir, digest, counts, sprite_h, args.skip_qa)
    qa = work / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1200], flush=True)
    if report["off_palette_colors"] or report["bad_alpha"] or report["edge_touching"]:
        raise SystemExit("palette, alpha, or edge check failed")

    meta = {
        "generator": "forge_staravia_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 397,
        "form": "normal",
        "variant": "staravia",
        "archetype": "biped_bird",
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "face_right": sorted(FACE_RIGHT),
        "palette_colors": int(len(palette)),
        "off_palette_colors": report["off_palette_colors"],
        "source": "front.png 96x96",
        "gifs_untouched": True,
        "wings_invented": 0,
        "feet": 2,
        "rear_fan": 1,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
