#!/usr/bin/env python3
"""Mega Latias (0380_latias-mega) 8-direction x 4-action sheet, from the v2 120x120 still.

Archetype: avian_raptor_hover, legless (no invented feet). The v2 drawing faces right:
a delta wing, a rigid head with the hand-drawn eye, a small chin, one forelimb, a
side fin and a ribbon tail. Directions are that drawing and its horizontal mirror.

  idle   : hover. Wings, forelimb and tail sway. A short growl (chin + roar ring).
  walk   : nose-down glide, two wingbeats per loop, speed lines. No foot cycle.
  attack : roar (chin drops, rings) -> 360 degree corkscrew -> dive strike -> recover.
  hurt   : white flash, knock-back, a half tumble, sparks, recover. 12 frames.

The head is an integer shift except during the corkscrew, where the whole body rolls.
Wings, the forelimb and the fins rotate about their sockets. Every pixel is snapped
to the v2 palette. front/back png and gif are not written.

Resumable: finished frames live in agent-sprite-forge/_work/frames/<action>/<dir>/NN.png
with a parameter hash in _work/state.json. A re-run skips finished frames.
  --force     rebuild every frame
  --samples   write _work/parts.png and _work/samples.png, then stop
  --no-sync   do not copy the sheet to pokemon_arena_2d
  --skip-qa   skip gait_loop.py
  --publish   register the sheet in data/dir8.js (only after the preview looks right)

Run:
  <agent-sprite-forge>/.venv/bin/python tools/forge_latias_mega_actions_v2.py
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

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0380_latias/mega/latias-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen3/0380_latias/mega/latias-mega/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
FACE_RIGHT = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 200, 188, 168
VERSION = "latias_mega_v2"

# Colours taken from the v2 palette (measured on front.png).
WHITE = (236, 237, 238)
LAV = (141, 137, 191)
AMBER = (232, 178, 72)
MOUTH = (40, 34, 64)
INK = (16, 16, 16)


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


def segment(img):
    """Seven parts of the right-facing v2 still. Masks partition every opaque pixel."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    rgb = img[..., :3].astype(np.int16)
    white = a & (rgb[..., 0] > 190) & (rgb[..., 1] > 190) & (rgb[..., 2] > 190)
    amber = a & (rgb[..., 0] > 160) & (rgb[..., 1] > 80) & (rgb[..., 2] < 120)
    yy, xx = np.mgrid[0:h, 0:w]

    seed = np.zeros_like(a)
    seed[9, 4] = True
    wing = flood(seed, a & ~white)
    wing = wing | (white & (xx <= 52) & dilate(wing, 1))

    horn_allow = a & (yy <= 28) & (xx >= 64) & ~dilate(wing, 1)
    horns = flood(horn_allow & (yy <= 22), horn_allow)
    face = flood(amber, a & (yy <= 46) & (xx >= 60) & ~wing & ~horns)
    face = face | flood(face, a & (yy <= 48) & (xx >= 72) & ~wing)
    head_all = face | horns
    jaw = head_all & white & (yy >= 38) & (yy <= 48) & (xx >= 72)
    head = head_all & ~jaw

    ys, xs = np.nonzero(a & (yy >= 50) & (yy <= 75) & (xx >= 100))
    arm_seed = np.zeros_like(a)
    i = int(np.argmax(xs))
    arm_seed[int(ys[i]), int(xs[i])] = True
    arm = flood(arm_seed, a & (yy >= 44) & (yy <= 80) & (xx >= 78) & ~head & ~jaw & ~wing)

    rest = a & ~wing & ~head & ~jaw & ~arm
    side = rest & (yy >= 82) & (yy <= 98) & (xx >= 58)
    tail = rest & (yy >= 92) & ~side
    torso = rest & ~side & ~tail
    # Anything the bounds missed stays on the torso so the silhouette cannot lose pixels.
    torso = torso | (a & ~(wing | head | jaw | arm | side | tail))
    return {
        "wing": wing, "head": head, "jaw": jaw, "arm": arm,
        "side": side, "tail": tail, "torso": torso,
    }


def bbox(mask):
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
    """4x nearest rotation. Palette stays exact. Angle follows the content-rotation used below."""
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


def content_point(x, y, angle_deg, pivot):
    """Where a point moves under rotate_nearest (inverse of that function's sample map)."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    vx, vy = x - pivot[0], y - pivot[1]
    return pivot[0] + c * vx - s * vy, pivot[1] + s * vx + c * vy


def roll_frame(img, theta, cy):
    """Aileron roll: squash by |cos|, vertical mirror past 90 degrees (belly-up)."""
    c = math.cos(theta)
    if abs(c - 1.0) < 1e-3:
        return img.copy()
    scale = max(0.34, abs(c))
    h = img.shape[0]
    out = np.zeros_like(img)
    for y in range(h):
        if c >= 0:
            sy = cy + (y - cy) / scale
        else:
            sy = cy - (y - cy) / scale
        sy = int(round(sy))
        if 0 <= sy < h:
            out[y] = img[sy]
    return out


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    spur = a & (n == 0)
    out[spur] = 0
    return out


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def nearest_color(src, target_mask, from_mask):
    """Paint target pixels with the nearest opaque colour from from_mask."""
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int16)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    # Chunked nearest so a long edge does not build a giant matrix.
    pick = np.empty(len(ty), np.int32)
    step = 400
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None].astype(np.int32) - pts[None, :, 0]) ** 2 + (tx[sl, None].astype(np.int32) - pts[None, :, 1]) ** 2
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


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def attached_mouth(img, facing, amount):
    """Dark mouth wedge grown out of the snout tip, so the chin never floats free."""
    if amount <= 0:
        return img
    a = img[..., 3] >= 128
    rgb = img[..., :3].astype(np.int16)
    amber = a & (rgb[..., 0] > 160) & (rgb[..., 1] > 80) & (rgb[..., 2] < 120)
    ey, ex = np.nonzero(amber)
    if len(ey) == 0:
        return img
    eye_y, eye_x = int(np.median(ey)), int(np.median(ex))
    yy = np.arange(img.shape[0])[:, None]
    band = a & (np.abs(yy - eye_y) <= 5)
    if facing > 0:
        band &= np.arange(img.shape[1])[None, :] >= eye_x
    else:
        band &= np.arange(img.shape[1])[None, :] <= eye_x
    by, bx = np.nonzero(band)
    if len(bx) == 0:
        return img
    if facing > 0:
        tip_x = int(bx.max())
        tip_y = int(np.median(by[bx >= tip_x - 1]))
        step = 1
    else:
        tip_x = int(bx.min())
        tip_y = int(np.median(by[bx <= tip_x + 1]))
        step = -1
    for i in range(amount + 1):
        for dy in (-1, 0, 1):
            if i == 0 and dy != 0:
                continue
            col = INK if abs(dy) == 1 or i == amount else MOUTH
            put(img, tip_x + step * i, tip_y + dy + (1 if i else 0), col)
    return img


def roar_rings(out, mouth, facing, age, strength):
    mx, my = mouth
    cols = (WHITE, LAV, AMBER)
    for r_i in range(strength):
        rad = 5 + age * 3 + r_i * 5
        col = cols[min(r_i, 2)]
        for deg in range(-55, 56, 7):
            ang = math.radians(deg)
            put(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.85, col)
    return out


def speed_lines(out, facing, n, seed, length):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    # Streaks leave the lower rear, under the wing, so they do not read as broken wing pixels.
    rear_x = (x0 + (x1 - x0) // 5) if facing > 0 else (x1 - (x1 - x0) // 5)
    y_lo = y0 + (y1 - y0) // 2
    for k in range(n):
        y = int(rng.integers(y_lo, max(y_lo + 1, y1 - 2)))
        off = int(rng.integers(2, 7))
        L = int(rng.integers(max(3, length // 2), length + 1))
        for i in range(L):
            put(out, rear_x - facing * (off + i), y, WHITE if i < 2 else LAV)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(size // 2, size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), WHITE if r < size - 2 else AMBER)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, WHITE)
    return out


def sparks(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    d = frame_idx
    cx = (x0 + x1) // 2
    pts = [(cx - facing * (6 + d * 2), y0 + 6 - d), (cx + facing * (8 + d), y0 + 2 - d // 2),
           (cx - facing * (2 + d), y0 - 1 - d)]
    for sx, sy in pts:
        put(out, sx, sy, WHITE)
        put(out, sx + facing, sy, AMBER)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def socket_pivot(part, body):
    sel = part & dilate(body, 6)
    if not np.any(sel):
        sel = part
    ys, xs = np.nonzero(sel)
    return float(xs.mean()), float(ys.mean())


def top_pivot(part):
    ys, xs = np.nonzero(part)
    top = ys.min()
    sel_y = ys <= top + 2
    return float(xs[sel_y].mean()), float(ys[sel_y].mean())


def build_rig(img, masks, palette_unused, flip):
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

    # A short mouth stem on the chin, sitting under the head. The head is drawn
    # after the chin, so the stem is hidden until the chin drops.
    jaw_m = placed["jaw"]
    if np.any(jaw_m):
        jys, jxs = np.nonzero(jaw_m)
        top = int(jys.min())
        xs = np.unique(jxs[jys <= top + 1])
        x_lo, x_hi = int(xs.min()), int(xs.max())
        for dy in range(1, 4):
            y = top - dy
            if y < 0:
                break
            for x in range(x_lo, x_hi + 1):
                if not placed["head"][y, x]:
                    continue
                edge = x <= x_lo + 1 or x >= x_hi - 1
                layers["jaw"][y, x, :3] = INK if edge else MOUTH
                layers["jaw"][y, x, 3] = 255

    # Socket fill stays on the torso, under the wing and the forelimb.
    for part_name, rad in (("wing", 5), ("arm", 4)):
        zone = placed[part_name] & dilate(placed["torso"], rad)
        if np.any(zone):
            fill = nearest_color(canvas, zone, placed["torso"])
            empty = layers["torso"][..., 3] < 128
            write = zone & empty
            layers["torso"][write] = fill[write]

    pivots = {
        "wing": socket_pivot(placed["wing"], placed["torso"]),
        "arm": socket_pivot(placed["arm"], placed["torso"]),
        "tail": top_pivot(placed["tail"]),
        "side": socket_pivot(placed["side"], placed["torso"]),
    }
    # Neck joint: bottom-centre of the head, so a body tilt carries the face without rotating the eye.
    hys, hxs = np.nonzero(placed["head"])
    neck = (float(np.median(hxs[hys >= hys.max() - 2])), float(hys.max()))

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        neck = (CELL_W - 1 - neck[0], neck[1])
    facing = -1.0 if flip else 1.0
    return layers, pivots, neck, facing


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Sine that is exactly 0 on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(dx=0.0, dy=0.0, tilt=0.0, roll=0.0, head_dx=0.0, head_dy=0.0, jaw=0,
             wing=0.0, arm=0.0, tail=0.0, side=0.0, ring=-1, rings=0, lines=0, burst=0, flash=0, sparks=0)
    if action == "idle":
        p["dy"] = wave(f, n, 2.0)
        p["dx"] = wave(f, n, 0.6)
        p["wing"] = wave(f, n, 8.0)
        p["arm"] = wave(f, n, 7.0, phase=-0.6)
        p["tail"] = wave(f, n, 6.0, phase=-0.9)
        p["side"] = wave(f, n, 5.0, phase=-1.2)
        growl = {9: 1, 10: 2, 11: 2, 12: 1}
        p["jaw"] = growl.get(f, 0)
        if f in (10, 11, 12):
            p["ring"], p["rings"] = f - 10, 1
    elif action == "walk":
        beat = wave(f, n, 1.0, cycles=2.0)
        p["dy"] = wave(f, n, 1.4, cycles=2.0)
        p["dx"] = wave(f, n, 2.2)
        p["tilt"] = -7.0 + wave(f, n, 2.0, cycles=2.0)
        p["wing"] = wave(f, n, 15.0, cycles=2.0)
        p["arm"] = wave(f, n, -12.0, cycles=2.0)
        p["tail"] = wave(f, n, 8.0, phase=-0.8, cycles=2.0)
        p["side"] = wave(f, n, -7.0, phase=-0.8, cycles=2.0)
        p["jaw"] = 1 if beat > 0.65 else 0
        p["lines"] = 3
    elif action == "attack":
        dx_t = [0, -2, -3, -4, -3, -1, 2, 4, 6, 8, 9, 10, 8, 4, 2, 0]
        dy_t = [0, -1, -3, -4, -3, -2, -1, 0, 1, 2, 2, 3, 2, 1, 0, 0]
        tilt_t = [0, 6, 12, 14, 12, 6, 0, -6, -10, -12, -14, -12, -8, -4, -2, 0]
        jaw_t = [0, 1, 3, 3, 3, 2, 1, 1, 1, 1, 2, 3, 2, 1, 0, 0]
        wing_t = [0, 12, 20, 24, 20, 8, -8, -16, -16, -12, -18, -14, -8, -4, 0, 0]
        p["dx"], p["dy"] = dx_t[f], dy_t[f]
        p["tilt"], p["jaw"] = tilt_t[f], jaw_t[f]
        p["wing"], p["arm"] = wing_t[f], -wing_t[f] * 0.85
        p["tail"], p["side"] = wing_t[f] * 0.45, -wing_t[f] * 0.35
        # The face stays on the neck. The roar is the mouth wedge, the rings and the raised wings.
        p["head_dy"] = 0
        p["head_dx"] = 0
        if 5 <= f <= 10:
            p["roll"] = 2 * math.pi * (f - 4) / 6.0
        if 2 <= f <= 5:
            p["ring"], p["rings"] = f - 2, 3
        if 6 <= f <= 11:
            p["lines"] = 5
        if f in (11, 12):
            p["burst"] = 1
    else:
        env = math.sin(math.pi * min(1.0, f / 9.0))
        p["dx"] = 5.0 * env
        p["dy"] = -2.0 * env
        p["tilt"] = 14.0 * env
        if 2 <= f <= 6:
            # A wobble, not a belly-up flip. Past 90 degrees the row-squash tears the body apart.
            p["roll"] = math.radians(38) * math.sin(math.pi * (f - 2) / 4.0)
        p["jaw"] = 2 if 1 <= f <= 5 else (1 if f in (6, 7) else 0)
        p["wing"], p["arm"] = -16 * env, 12 * env
        p["tail"], p["side"] = 10 * env, -8 * env
        p["head_dx"] = 0
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def synth(rig, action, f, palette):
    layers, pivots, neck, facing = rig
    p = pose(action, f)
    fwd = facing  # +1 faces right
    bdx, bdy = p["dx"] * fwd, p["dy"]
    # head_dx > 0 pulls the face backward (opposite the facing).
    hdx = bdx - p["head_dx"] * fwd
    hdy = bdy + p["head_dy"]
    # The chin stays on the head. An open mouth is drawn onto the snout, not by tearing the chin off.
    jdx, jdy = hdx, hdy
    sgn = -1.0 if facing < 0 else 1.0

    def moved(name, ang):
        return shift(rotate_nearest(layers[name], ang * sgn, pivots[name]), bdx, bdy)

    torso = shift(layers["torso"], bdx, bdy)
    wing = moved("wing", p["wing"])
    arm = moved("arm", p["arm"])
    tail = moved("tail", p["tail"])
    side = moved("side", p["side"])
    head = shift(layers["head"], hdx, hdy)
    jaw = shift(layers["jaw"], jdx, jdy)

    spinning = abs(p["roll"]) > 0.2
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    # Tail and wing sit behind the chest. The forelimb and the side fin sit in front.
    for lyr in (tail, torso, wing, side, arm):
        paste(body, lyr)
    moved = any(abs(p[k]) >= 0.45 for k in ("wing", "arm", "tail", "side"))
    if moved:
        body = despeckle(body)

    bb = body_box(body)
    center = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2) if bb else (CELL_W / 2, FOOT_Y - 40)
    tilt_ang = -p["tilt"] * fwd

    if spinning:
        for lyr in (jaw, head):
            paste(body, lyr)
        body = roll_frame(body, p["roll"], center[1])
        if tilt_ang:
            body = rotate_nearest(body, tilt_ang, center)
        body = despeckle(body)
        img = body
    else:
        if tilt_ang:
            body = despeckle(rotate_nearest(body, tilt_ang, center))
            nx, ny = content_point(neck[0] + bdx, neck[1] + bdy, tilt_ang, center)
            extra_x = int(round(nx - (neck[0] + bdx)))
            extra_y = int(round(ny - (neck[1] + bdy)))
            head = shift(head, extra_x, extra_y)
            jaw = shift(jaw, extra_x, extra_y)
        img = body
        paste(img, jaw)
        paste(img, head)

    img = snap(img, palette)
    if p["jaw"] and not spinning:
        img = attached_mouth(img, fwd, int(p["jaw"]))
    if p["flash"]:
        img = flash_white(img)
    if p["rings"] and p["ring"] >= 0:
        bbj = body_box(jaw if not spinning else img)
        if bbj:
            mx = bbj[2] if facing > 0 else bbj[0]
            my = (bbj[1] + bbj[3]) // 2
            img = roar_rings(img, (mx, my), fwd, p["ring"], p["rings"])
    if p["lines"]:
        img = speed_lines(img, fwd, p["lines"], seed=f * 17 + (0 if action == "walk" else 3), length=12 if action == "attack" else 8)
    if p["burst"]:
        bb2 = body_box(img)
        if bb2:
            tip_x = bb2[2] + 4 if fwd > 0 else bb2[0] - 4
            img = impact_burst(img, (tip_x, bb2[3] - 16), 11 if f >= 12 else 8)
    if p["sparks"]:
        img = sparks(img, f, fwd)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + (MEGA_DIR / "front.png").read_bytes()
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
    """South row, four sample frames of each action, on a dark field. This is the file to look at."""
    samples = {"idle": (0, 4, 8, 11), "walk": (0, 4, 8, 12), "attack": (2, 3, 7, 11), "hurt": (1, 4, 7, 10)}
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            if f >= FRAMES[act]:
                continue
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            x = si * CELL_W
            y = ai * CELL_H
            m = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    """Every east-facing frame, cropped, so the roar and the spin can be read in one picture."""
    pad = 8
    crops = []
    labels_h = 0
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][2 * CELL_H:3 * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            a = tile[..., 3] >= 128
            if not np.any(a):
                continue
            ys, xs = np.nonzero(a)
            y0, y1 = max(0, ys.min() - pad), min(CELL_H, ys.max() + 1 + pad)
            x0, x1 = max(0, xs.min() - pad), min(CELL_W, xs.max() + 1 + pad)
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
    return Image.fromarray(bg, "RGBA"), labels_h


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
        "dex": 380, "form": "mega", "variant": "latias-mega", "gen": "gen3",
        "dir": "monsterSprite/gen3/0380_latias/mega/latias-mega",
        "sheetDir": "agent-sprite-forge", "layout": "actions_8dir", "nf": 16,
        "cw": CELL_W, "ch": CELL_H, "w": CELL_W, "h": int(sprite_h), "fy": FOOT_Y,
        "an": [FRAMES[a] for a in ACTIONS],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    key = '"0380_latias-mega":'
    if key in text:
        # Replace the existing object. It contains no nested braces.
        start = text.index(key) + len(key)
        end = text.index("}", start) + 1
        text = text[:start] + blob + text[end:]
    else:
        text = text.replace(prefix + "{", prefix + "{" + key + blob + ",", 1)
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
    if front.shape[0] != 120 or front.shape[1] != 120:
        raise SystemExit(f"expected the v2 120x120 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), flush=True)

    rigs = {}
    for d in DIRS:
        rigs[d] = build_rig(front, masks, palette, flip=d not in FACE_RIGHT)

    # Rest pose must reproduce the still. Compare the east (unflipped) idle frame 0.
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
    print("rest-pose pixels changed", int(differ.sum()), flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {"torso": (190, 190, 198), "wing": (220, 60, 60), "tail": (40, 90, 220),
                 "side": (220, 40, 180), "arm": (40, 180, 80), "head": (240, 200, 40), "jaw": (40, 220, 220)}
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "E", f) for f in (0, 4, 8, 11)] + [("walk", "E", f) for f in (0, 4, 8, 12)]
        want += [("attack", "E", f) for f in (0, 2, 3, 5, 7, 11, 13)] + [("attack", "W", f) for f in (3, 7)]
        want += [("hurt", "E", f) for f in (1, 4, 7)]
        tiles = []
        for act, d, f in want:
            tiles.append(synth(rigs[d], act, f, palette))
        tw, th = CELL_W, CELL_H
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * th, cols * tw, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = tile[m]
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
    preview_image(sheets).save(OUT_DIR / "preview.png")
    contact, _ = motion_contact(sheets)
    contact.save(work / "review_E.png")
    contact.save(OUT_DIR / "preview_motion.png")

    # Opaque height of the neutral south frame. The game scales the body by this.
    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 108

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{d}:{f}")
    report = {"version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
              "off_palette_colors": len(off), "edge_touching": {k: len(v) for k, v in clip.items()}}
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run([py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                            "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                           capture_output=True, text=True)
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:800], flush=True)

    meta = {"generator": "forge_latias_mega_actions_v2", "version": VERSION, "key": "0380_latias-mega",
            "dex": 380, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
            "frames": FRAMES, "directions": DIRS, "actions": ACTIONS, "face_right": sorted(FACE_RIGHT),
            "palette_colors": int(len(palette)), "off_palette_colors": len(off),
            "source": "v2 front.png 120x120", "gifs_untouched": True}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        publish_dir8(sprite_h)
        print("dir8 published", VERSION, flush=True)


if __name__ == "__main__":
    main()
