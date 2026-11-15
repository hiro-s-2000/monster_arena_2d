#!/usr/bin/env python3
"""Gallade (0475) normal 8-direction action sheet from this form's front.png.

Measured on normal/front.png: 96x96, 1324 opaque, bbox x21-74 y14-82, 15
colours, alpha 0/255, one 4-connected piece.

Biped. The head faces screen-right: the face white (x51-63) and the single red
eye (x48-53 y33-36) sit right of the teal head crest (x37-57 y14-23). There is
no drawn mouth, so the roar flares the existing crest and the side spike rides
with the head; rings come off the face. Both arms are akimbo elbow blades:
  arm_b (screen-left, rear when facing right)  x21-43 y41-53, shoulder (42,49)
  arm_f (screen-right, front)                  x52-75 y44-56, shoulder (58,46)
The red chest fin stays on the torso. Two long splayed legs from y61:
  leg_b x<=48, leg_f x>=49, hip pivots near (43,61) and (55,61).

Unflipped dirs are S SE E NE (the still faces right). N NW W SW are mirrored.

  idle   : feet planted, breathing bob of the upper body, blades sway, crest
           breathes. Frame 0 and the loop end are the still.
  walk   : the two legs alternate (lift, swing forward, plant), arms counter-
           swing, upper body bobs.
  attack : 0-4 crest flare + psychic rings, front blade raised (wind-up).
           5-7 pirouette: the whole body turns around (facing mirror with a
           narrowed transition, never an in-plane rotation) with speed lines.
           8-10 front blade slash down + rear blade cross slash with a
           crescent psycho-cut arc. 11-12 impact. 13-15 recover to the still.
  hurt   : 12 frames, white flash, backward lean <= 30 degrees, recover.

Palette-locked to this still. front/back png and gif are not written.
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

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0475_gallade/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0475_gallade/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
BOUNDS = ROOT / "data/sprite_bounds.js"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
UNFLIPPED = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "gallade_v2"
KEY = "0475_gallade"

C_LT = (197, 205, 222)
C_INK = (16, 16, 16)
C_GRN = (90, 180, 106)
C_DGN = (65, 131, 90)
C_SLT = (57, 82, 82)
C_WHT = (238, 238, 255)
C_GRY = (123, 139, 172)
C_STL = (115, 148, 164)
C_TEA = (123, 197, 180)
C_RED = (230, 82, 57)
C_PAL = (164, 180, 205)
C_PNK = (255, 156, 131)
C_PUR = (255, 255, 255)

STILL_N = 1324
STILL_BBOX = (21, 74, 14, 82)


# ---------------------------------------------------------------- helpers
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
            q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and allow[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


def component_sizes(mask):
    seen = np.zeros_like(mask)
    h, w = mask.shape
    sizes = []
    ys, xs = np.nonzero(mask)
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
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
        out[...] = 0
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = a.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    mapped = palette[dist.argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv.reshape(-1)]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    out[~a] = 0
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
    """4x nearest rotation, positive = clockwise on screen. No squash."""
    if not np.any(layer[..., 3] >= 128):
        return layer.copy()
    norm = angle_deg % 360.0
    if min(norm, 360.0 - norm) < 0.45:
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
    spun = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    spun[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    sh, sw = layer.shape[:2]
    inv = spun[2::4, 2::4].copy()
    # Centre sample only: forward filling adds a fringe on 1px outlines.
    # Close 1px cracks inside the part (3+ opaque neighbours).
    for _ in range(2):
        al = inv[..., 3] >= 128
        pp = np.pad(al, 1)
        n = pp[:-2, 1:-1].astype(np.int16) + pp[2:, 1:-1] + pp[1:-1, :-2] + pp[1:-1, 2:]
        hole = (~al) & (n >= 3)
        if not np.any(hole):
            break
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            src = np.roll(np.roll(inv, -dy, 0), -dx, 1)
            take = hole & (src[..., 3] >= 128) & (inv[..., 3] < 128)
            inv[take] = src[take]
    return inv


def content_point(x, y, angle_deg, pivot):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    vx, vy = x - pivot[0], y - pivot[1]
    return pivot[0] + c * vx - s * vy, pivot[1] + s * vx + c * vy


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int64)
    for i in range(0, len(ty), 400):
        sl = slice(i, i + 400)
        d = (ty[sl, None].astype(np.int32) - pts[None, :, 0]) ** 2
        d += (tx[sl, None].astype(np.int32) - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def seal_cracks(img):
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        p = np.pad(a, 1)
        n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        hole = (~a) & (n >= 3)
        if not np.any(hole):
            break
        fill = nearest_color(out, hole, a)
        out[hole] = fill[hole]
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def touches_edge(tile):
    c = tile[..., 3] >= 128
    if not np.any(c):
        return False
    return bool(c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any())


def narrow_x(img, scale, cx):
    """Narrow horizontally around cx for a turn frame. Union sampling keeps it one piece."""
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    for x in range(w):
        s0 = cx + (x - 0.5 - cx) / scale
        s1 = cx + (x + 0.5 - cx) / scale
        a0, a1 = int(math.floor(min(s0, s1) + 0.5)), int(math.floor(max(s0, s1) - 0.5 + 1e-9))
        # Centre column first, then the others fill gaps.
        mid = int(round(cx + (x - cx) / scale))
        order = [mid] + [c for c in range(a0, a1 + 1) if c != mid]
        for sx in order:
            if not (0 <= sx < w):
                continue
            col = img[:, sx]
            use = (out[:, x, 3] < 128) & (col[:, 3] >= 128)
            out[use, x] = col[use]
    return out


# ---------------------------------------------------------------- measure
def segment(img):
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    if (h, w) != (96, 96):
        raise SystemExit("front is not 96x96")
    if int(a.sum()) != STILL_N:
        raise SystemExit(f"still opaque changed: {int(a.sum())}")
    ys, xs = np.nonzero(a)
    if (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())) != STILL_BBOX:
        raise SystemExit("still bbox changed; remeasure before forging")
    if component_sizes(a) != [STILL_N]:
        raise SystemExit("still is not one 4-connected piece")
    rgb = img[..., :3]
    yy, xx = np.mgrid[0:h, 0:w]

    def is_col(c):
        return a & np.all(rgb == np.array(c, np.uint8), axis=-1)

    head = a & (yy <= 40)
    crest = head & (yy <= 23)
    arm_b = flood(a & (yy == 42) & (xx <= 24), a & (yy >= 41) & (yy <= 53) & (xx <= 43))
    allow_f = a & ~head & (yy >= 44) & (yy <= 56) & (
        ((yy <= 49) & (xx >= 57)) | ((yy >= 50) & (xx >= 53))
    )
    arm_f = flood(a & (yy == 48) & (xx >= 66) & (xx <= 72), allow_f)
    legs = a & (yy >= 61)
    leg_b = legs & (xx <= 48)
    leg_f = legs & (xx >= 49)
    torso = a & ~head & ~arm_b & ~arm_f & ~legs
    head = head & ~crest
    parts = {"crest": crest, "head": head, "arm_b": arm_b, "arm_f": arm_f,
             "leg_b": leg_b, "leg_f": leg_f, "torso": torso}
    total = sum(int(m.sum()) for m in parts.values())
    if total != STILL_N:
        raise SystemExit(f"parts overlap or miss pixels: {total}")
    for name, m in parts.items():
        sz = component_sizes(m)
        if name != "torso" and len(sz) != 1:
            raise SystemExit(f"{name} is not one piece: {sz[:5]}")
    eye = is_col(C_RED) & (yy <= 40)
    if int(eye.sum()) < 3:
        raise SystemExit("eye red missing")
    eye_x = float(np.nonzero(eye)[1].mean())
    crest_x = float(np.nonzero(crest)[1].mean())
    face_white = is_col(C_WHT) & (yy <= 40)
    face_x = float(np.nonzero(face_white)[1].mean())
    if not (face_x > eye_x - 2 and face_x > crest_x):
        raise SystemExit(f"head is not facing right: face {face_x:.1f} eye {eye_x:.1f} crest {crest_x:.1f}")
    if int((is_col(C_RED) & (arm_b | arm_f)).sum()):
        raise SystemExit("chest red leaked onto an arm")
    info = {k: int(v.sum()) for k, v in parts.items()}
    info.update(eye_x=round(eye_x, 1), face_x=round(face_x, 1), crest_x=round(crest_x, 1))
    return parts, info


# ---------------------------------------------------------------- rig
def build_rig(img, masks, flip):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    ox, oy = dst_x - x0, dst_y - y0
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    canvas[..., 3] = np.where(canvas[..., 3] >= 128, 255, 0).astype(np.uint8)
    canvas[canvas[..., 3] == 0] = 0

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

    def P(x, y):
        return (x + ox, y + oy)

    pivots = {
        "arm_b": P(42, 49), "arm_f": P(58, 46),
        "leg_b": P(43, 61), "leg_f": P(55, 61),
        "crest": P(50, 23), "hip": P(48, 59),
    }
    yy, xx = np.mgrid[0:CELL_H, 0:CELL_W]

    # Torso sockets under the shoulders: torso colour where a moving arm leaves
    # the shoulder or the hand at the hip. Hidden under the arm at rest.
    for part, rad in (("arm_b", 2), ("arm_f", 2)):
        zone = placed[part] & dilate(placed["torso"], rad)
        fill = nearest_color(canvas, zone, placed["torso"])
        write = zone & (layers["torso"][..., 3] < 128)
        layers["torso"][write] = fill[write]
    # Crest socket: head colour under the crest base.
    zone = placed["crest"] & dilate(placed["head"], 2)
    fill = nearest_color(canvas, zone, placed["head"])
    write = zone & (layers["head"][..., 3] < 128)
    layers["head"][write] = fill[write]
    # Leg columns extend 2px up under the skirt so a 1px body bob or a lifted
    # hip never opens a gap. Legs are drawn before the torso.
    for leg in ("leg_b", "leg_f"):
        m = placed[leg]
        lys, lxs = np.nonzero(m)
        top = int(lys.min())
        for x in np.unique(lxs[lys == top]).tolist():
            for dy in (1, 2, 3):
                y = top - dy
                if placed["torso"][y, x] and layers[leg][y, x, 3] == 0:
                    layers[leg][y, x] = canvas[top, x]

    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    hy, hx = np.nonzero(placed["head"] & (yy >= oy + 28))
    face = (float(hx.max()) + 2, float(hy.mean()))
    fys, fxs = np.nonzero(placed["arm_f"])
    tip_f = (float(fxs.max()), float(fys[fxs >= fxs.max() - 1].mean()))
    bys, bxs = np.nonzero(placed["arm_b"])
    tip_b = (float(bxs.min()), float(bys[bxs <= bxs.min() + 1].mean()))
    soles = {}
    for leg in ("leg_b", "leg_f"):
        sy, sx = np.nonzero(placed[leg])
        yb = int(sy.max())
        soles[leg] = [(int(x), yb) for x in sx[sy == yb]]
    sole = (float(np.mean([p[0] for v in soles.values() for p in v])), float(FOOT_Y - 1))
    facing = 1.0
    if flip:
        def mx(p):
            return (CELL_W - 1 - p[0], p[1])
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        pivots = {k: mx(v) for k, v in pivots.items()}
        center, face, tip_f, tip_b, sole = mx(center), mx(face), mx(tip_f), mx(tip_b), mx(sole)
        soles = {k: [(CELL_W - 1 - x, y) for x, y in v] for k, v in soles.items()}
        facing = -1.0
    return {
        "layers": layers, "pivots": pivots, "center": center, "face": face,
        "tip_f": tip_f, "tip_b": tip_b, "sole": sole, "soles": soles,
        "facing": facing, "rest": canvas, "body_n": int((canvas[..., 3] >= 128).sum()),
    }


# ---------------------------------------------------------------- motion
def wave(f, n, amp, phase=0.0, cycles=1.0):
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def pose(action, f):
    """Angles are in the unflipped (right-facing) frame, positive = clockwise.

    arm_f: negative raises the front elbow blade. arm_b: positive raises the
    rear elbow blade. leg: positive swings the foot toward screen-left.
    """
    n = FRAMES[action]
    p = dict(
        arm_f=0.0, arm_b=0.0, leg_b=0.0, leg_f=0.0, lift_b=0, lift_f=0,
        fwd_b=0, fwd_f=0, crest=0.0, bob=0, turn=0, narrow=1.0,
        recoil=0.0, knock=0, flash=0, rings=-1, lines=0, slash=-1, burst=0,
        sparks=0, aura=0,
    )
    if action == "idle":
        breath = (1 - math.cos(2 * math.pi * f / n)) / 2
        p["bob"] = -1 if breath > 0.5 else 0
        p["arm_f"] = wave(f, n, -4.0)
        p["arm_b"] = wave(f, n, 4.0)
        p["crest"] = wave(f, n, -3.0, cycles=1.0)
    elif action == "walk":
        step = math.sin(2 * math.pi * 2 * f / n)
        lb = max(0.0, step)
        lf = max(0.0, -step)
        p["lift_b"] = int(round(3 * lb))
        p["lift_f"] = int(round(3 * lf))
        p["fwd_b"] = int(round(2 * lb))
        p["fwd_f"] = int(round(2 * lf))
        p["leg_b"] = -9.0 * lb
        p["leg_f"] = -9.0 * lf
        p["bob"] = -1 if abs(step) > 0.6 else 0
        p["arm_f"] = 7.0 * step
        p["arm_b"] = 7.0 * step
        p["crest"] = wave(f, n, -2.0, cycles=2.0)
        p["lines"] = 1 if abs(step) > 0.6 else 0
    elif action == "attack":
        # 0-4 crest flare, psychic rings, front blade raised.
        arm_f = [0, -12, -22, -30, -32, -10, 18, 24, 24, 24, 24, 20, 14, 8, 3, 0]
        arm_b = [0, 4, 8, 10, 10, 6, 18, 26, 26, 26, 26, 22, 14, 8, 3, 0]
        crest = [0, -4, -7, -8, -8, -4, 0, 0, 0, 0, 0, -3, -2, -1, 0, 0]
        leg_f = [0, 0, -2, -4, -4, -2, -8, -10, -10, -10, -10, -8, -5, -2, 0, 0]
        bob = [0, 0, 1, 1, 1, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0]
        p["arm_f"], p["arm_b"], p["crest"] = arm_f[f], arm_b[f], crest[f]
        p["leg_f"], p["bob"] = leg_f[f], bob[f]
        if 1 <= f <= 4:
            p["rings"] = f - 1
            p["aura"] = 1
        if 5 <= f <= 10:
            p["slash"] = min(f - 5, 2)
        if f in (11, 12):
            p["burst"] = f - 10
    else:
        env = [0, 0.5, 0.85, 1.0, 0.9, 0.7, 0.48, 0.3, 0.16, 0.06, 0.0, 0][f]
        p["recoil"] = 26.0 * env
        p["knock"] = int(round(3 * env))
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 5 else 0
        p["arm_f"] = -14.0 * env
        p["arm_b"] = 14.0 * env
        p["crest"] = 4.0 * env
    return p


# ---------------------------------------------------------------- fx
def put(out, x, y, col, over=False):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and (over or out[y, x, 3] < 128):
        out[y, x, :3] = col
        out[y, x, 3] = 255


def psychic_rings(out, origin, facing, age):
    ox, oy = origin
    cols = (C_WHT, C_PNK, C_TEA)
    for r_i in range(2):
        rad = 5 + age * 3 + r_i * 5
        col = cols[(age + r_i) % 3]
        for deg in range(-60, 61, 7):
            ang = math.radians(deg)
            put(out, ox + facing * rad * math.cos(ang), oy + rad * math.sin(ang) * 0.8, col)


def aura(out, facing, f):
    bb = body_box(out)
    if not bb:
        return
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(31 + f)
    for _ in range(6):
        x = int(rng.integers(x0 - 4, x1 + 5))
        y = int(rng.integers(y0, y1 - 10))
        put(out, x, y, C_PNK if rng.random() < 0.5 else C_TEA)


def speed_lines(out, facing, n, seed, length):
    bb = body_box(out)
    if not bb:
        return
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    for _ in range(n * 2):
        side = 1 if rng.random() < 0.5 else -1
        y = int(rng.integers(y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4))
        base = x1 + 3 if side > 0 else x0 - 3
        leng = int(rng.integers(4, length + 1))
        for i in range(leng):
            put(out, base + side * i, y, C_WHT if i < 2 else C_TEA)


def slash_arc(out, pivot, facing, stage):
    """Crescent psycho-cut sweep in front of the front blade."""
    px, py = pivot
    rad = 26 + stage * 2
    span = [(-70, 10), (-60, 50), (-20, 70)][stage]
    for deg in range(span[0], span[1] + 1, 3):
        ang = math.radians(deg)
        for k, col in ((0, C_WHT), (1, C_PNK), (2, C_TEA)):
            r = rad - k
            put(out, px + facing * r * math.cos(ang), py + r * math.sin(ang), col)


def burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4 + math.pi / 8
        for r in range(max(2, size // 2), size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang),
                C_WHT if r < size - 2 else C_PNK, over=True)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, C_WHT, over=True)


def sparks(out, f, facing):
    bb = body_box(out)
    if not bb:
        return
    x0, y0, x1, _ = bb
    cx = (x0 + x1) // 2
    for sx, sy in ((cx + facing * (6 + f), y0 + 4), (cx + facing * 12, y0 + 12), (cx - facing * 8, y0 + 8)):
        put(out, sx, sy, C_WHT)
        put(out, sx + facing, sy + 1, C_PNK)


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 80)
    out[a & ~dark, :3] = C_WHT
    return out


# ---------------------------------------------------------------- synth
def synth(rigs, d, action, f, palette, with_fx=True):
    p = pose(action, f)
    rig = rigs[d]
    if p["turn"]:
        # Mid-pirouette: show the body turned around (mirrored rig of this row).
        mate = {"S": "SW", "SE": "W", "E": "W", "NE": "NW",
                "N": "NE", "NW": "E", "W": "E", "SW": "S"}[d]
        rig = rigs[mate]
    L = rig["layers"]
    pv = rig["pivots"]
    fc = rig["facing"]

    def rot(name, ang, pivot=None):
        return rotate_nearest(L[name], fc * ang, pv[name] if pivot is None else pivot)

    turned = False
    if p["recoil"] > 0.4:
        arm_f = rot("arm_f", p["arm_f"])
        arm_b = rot("arm_b", p["arm_b"])
        crest = rot("crest", p["crest"])
        leg_b, leg_f = L["leg_b"], L["leg_f"]
        turned = True
    else:
        arm_f = rot("arm_f", p["arm_f"])
        arm_b = rot("arm_b", p["arm_b"])
        crest = rot("crest", p["crest"])
        leg_b = shift(rot("leg_b", p["leg_b"]), fc * p["fwd_b"], -p["lift_b"])
        leg_f = shift(rot("leg_f", p["leg_f"]), fc * p["fwd_f"], -p["lift_f"])
        turned = any(abs(p[k]) >= 0.45 for k in ("arm_f", "arm_b", "crest", "leg_b", "leg_f"))
    bob = p["bob"]
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    upper = np.zeros_like(body)
    for lyr in (L["torso"], arm_b, arm_f, L["head"], crest):
        paste(upper, lyr)
    if p["recoil"] > 0.4:
        # Backward lean of the upper body as an integer row shear about the hip
        # (crisp rows, no resampling). Head and crest move rigidly with the
        # neck row, so the face never resamples. Feet stay planted.
        hip_y = int(pv["hip"][1])
        tan = math.tan(math.radians(p["recoil"]))
        neck_y = hip_y - 19
        body_u = np.zeros_like(body)
        for lyr in (L["torso"], arm_b, arm_f):
            paste(body_u, lyr)
        sheared = np.zeros_like(body)
        for y in range(CELL_H):
            off = int(round(max(0, hip_y - y) * tan)) if y <= hip_y else 0
            off = -int(fc) * off
            if off:
                sheared[y] = np.roll(body_u[y], off, axis=0)
            else:
                sheared[y] = body_u[y]
        head_off = -int(fc) * int(round((hip_y - neck_y) * tan))
        head_l = np.zeros_like(body)
        paste(head_l, L["head"])
        paste(head_l, crest)
        upper = sheared
        paste(upper, shift(head_l, head_off, 1 if p["recoil"] > 12 else 0))
        upper = shift(upper, -fc * p["knock"], 0)
    paste(body, leg_b)
    paste(body, leg_f)
    paste(body, shift(upper, 0, bob))
    if turned:
        body = despeckle(body)

    tip_f = content_point(*rig["tip_f"], fc * p["arm_f"], pv["arm_f"])
    tip_b = content_point(*rig["tip_b"], fc * p["arm_b"], pv["arm_b"])
    if p["recoil"] > 0.4:
        body = seal_cracks(body)
        body = despeckle(body)
    if p["narrow"] < 0.99:
        body = narrow_x(body, p["narrow"], rig["center"][0])

    img = snap(body, palette)
    if not with_fx:
        return img
    face = (rig["face"][0], rig["face"][1] + bob)
    if p["aura"]:
        aura(img, fc, f)
    if p["rings"] >= 0:
        psychic_rings(img, (face[0] + fc * 2, face[1] - 2), fc, p["rings"])
    if p["lines"]:
        speed_lines(img, fc, p["lines"], seed=f * 13 + (5 if action == "walk" else 11), length=8)
    if p["slash"] >= 0:
        slash_arc(img, (pv["arm_f"][0], pv["arm_f"][1] + bob), fc, p["slash"])
    if p["burst"]:
        cx = pv["arm_f"][0] + fc * (30 + 2 * p["burst"])
        burst(img, (cx, pv["arm_f"][1] + 10), 7 + 3 * p["burst"])
    if p["sparks"]:
        sparks(img, f, -fc)
    if p["flash"]:
        img = flash_white(img)
    return snap(img, palette)


# ---------------------------------------------------------------- outputs
def params_hash():
    src = Path(__file__).read_bytes() + (SRC_DIR / "front.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def assemble(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png").convert("RGBA"))
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
    """All 8 directions: idle 0, walk 4, attack 3/6/9/11, hurt 3."""
    picks = [("idle", 0), ("walk", 4), ("walk", 12), ("attack", 3), ("attack", 6),
             ("attack", 9), ("attack", 11), ("hurt", 3)]
    pad = 8
    crop = (CELL_W // 2 - 70, CELL_H - 140, CELL_W // 2 + 70, CELL_H - 8)
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    bg = np.full((8 * ch, len(picks) * cw, 4), (15, 20, 31, 255), np.uint8)
    for di in range(8):
        for pi, (act, f) in enumerate(picks):
            tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            t = tile[crop[1]:crop[3], crop[0]:crop[2]]
            m = t[..., 3] >= 128
            bg[di * ch:(di + 1) * ch, pi * cw:(pi + 1) * cw][m] = t[m]
    del pad
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets, row="S", scale=2):
    ri = DIRS.index(row)
    crop = (CELL_W // 2 - 64, CELL_H - 136, CELL_W // 2 + 64, CELL_H - 12)
    cw, ch = (crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale
    cols = 8
    rows = sum(math.ceil(FRAMES[a] / cols) for a in ACTIONS)
    bg = np.full((rows * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    r = 0
    for act in ACTIONS:
        for f in range(FRAMES[act]):
            rr, cc = divmod(f, cols)
            tile = sheets[act][ri * CELL_H:(ri + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            t = tile[crop[1]:crop[3], crop[0]:crop[2]]
            t = np.repeat(np.repeat(t, scale, 0), scale, 1)
            m = t[..., 3] >= 128
            bg[(r + rr) * ch:(r + rr + 1) * ch, cc * cw:(cc + 1) * cw][m] = t[m]
        r += math.ceil(FRAMES[act] / cols)
    return Image.fromarray(bg, "RGBA")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def with_flock(lock_path, fn):
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def replace_key_value(text, prefix_hint, key, new_val):
    token = '"' + key + '":'
    start = text.find(token)
    if start < 0 or text.find(token, start + len(token)) != -1:
        raise SystemExit(f"{prefix_hint}: key {key} missing or duplicated")
    val_at = start + len(token)
    _old, end = json.JSONDecoder().raw_decode(text, val_at)
    return text[:val_at] + new_val + text[end:]


def publish_dir8(sprite_h):
    entry = {
        "dex": 475, "form": "normal", "variant": "gallade", "gen": "gen4",
        "dir": "monsterSprite/gen4/0475_gallade/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    new_val = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js prefix changed")
        out = replace_key_value(text, "dir8", KEY, new_val)
        before = json.loads(text[len(prefix):].strip().rstrip(";"))
        after = json.loads(out[len(prefix):].strip().rstrip(";"))
        if after[KEY] != entry or list(after) != list(before):
            raise SystemExit("dir8 edit failed")
        for k in before:
            if k != KEY and before[k] != after[k]:
                raise SystemExit(f"dir8 neighbour changed: {k}")
        if len(out) - len(text) != len(new_val) - len(json.dumps(before[KEY], separators=(",", ":"), ensure_ascii=False)):
            raise SystemExit("dir8 length delta unexpected")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(out)
        tmp.replace(DIR8)

    with_flock(DIR8_LOCK, edit)


def publish_bounds(sheets):
    vals = []
    for di in range(8):
        tile = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        x0, y0, x1, y1 = body_box(tile)
        vals += [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    new_val = json.dumps(vals, separators=(",", ":"))
    text = BOUNDS.read_text()
    s_at = text.find('"s":{')
    token = '"' + KEY + '":'
    pos = text.find(token, s_at)
    if s_at < 0 or pos < 0 or text.find(token, pos + len(token)) != -1:
        raise SystemExit("sprite_bounds s key not unique")
    val_at = pos + len(token)
    _old, end = json.JSONDecoder().raw_decode(text, val_at)
    out = text[:val_at] + new_val + text[end:]
    tmp = BOUNDS.with_suffix(".js.tmp")
    tmp.write_text(out)
    tmp.replace(BOUNDS)
    return vals


def update_queue(status, note=None):
    def edit():
        raw = QUEUE.read_text(encoding="utf-8")
        i = raw.find('"id": "' + KEY + '",')
        if i < 0:
            raise SystemExit("queue item missing")
        j = raw.find('"status": "', i)
        k = raw.find('"', j + len('"status": "'))
        out = raw[:j] + '"status": "' + status + raw[k:]
        if note:
            n0 = out.find('"notes": "', i)
            n1 = out.find('"', n0 + len('"notes": "'))
            while out[n1 - 1] == "\\":
                n1 = out.find('"', n1 + 1)
            out = out[:n1] + " " + json.dumps(note)[1:-1] + out[n1:]
        json.loads(out)
        tmp = QUEUE.with_suffix(".json.tmp")
        tmp.write_text(out, encoding="utf-8")
        tmp.replace(QUEUE)

    with_flock(QUEUE_LOCK, edit)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true",
                    help="after visual approval: dir8 + sprite_bounds + queue done")
    args = ap.parse_args()

    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if set(np.unique(front[..., 3]).tolist()) - {0, 255}:
        raise SystemExit("front alpha is not binary")
    palette = extract_palette(front)
    masks, info = segment(front)
    print("parts", info, "palette", len(palette), flush=True)
    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}

    for d in DIRS:
        for act in ACTIONS:
            r0 = synth(rigs, d, act, 0, palette, with_fx=False)
            if np.any(r0 != rigs[d]["rest"]):
                raise SystemExit(f"{act} frame 0 differs from the still ({d})")
        for act in ("idle", "walk"):
            n = FRAMES[act]
            end, rest = pose(act, n), pose("hurt", 0)
            if any(abs(float(end[k]) - float(rest[k])) > 1e-6 for k in end):
                raise SystemExit(f"{act} loop end is not the still")
    if rigs["E"]["face"][0] <= rigs["E"]["center"][0] or rigs["W"]["face"][0] >= rigs["W"]["center"][0]:
        raise SystemExit("E must face screen-right and W screen-left")
    print("still diff 0 for all dirs; E faces right, W faces left", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    pcols = {"torso": (180, 180, 190), "head": (250, 250, 250), "crest": (60, 220, 220),
             "arm_b": (40, 90, 220), "arm_f": (40, 180, 80),
             "leg_b": (240, 200, 40), "leg_f": (240, 120, 20)}
    part = np.zeros((96, 96, 4), np.uint8)
    for name, col in pcols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "S", f) for f in (0, 4, 8, 12)]
        want += [("walk", "S", f) for f in (0, 2, 4, 6, 10, 12)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("attack", "W", f) for f in (3, 6, 9, 11)]
        want += [("hurt", "S", f) for f in (0, 1, 3, 5, 8, 11)]
        tiles = []
        crop = (CELL_W // 2 - 64, CELL_H - 136, CELL_W // 2 + 64, CELL_H - 12)
        for act, d, f in want:
            bare = synth(rigs, d, act, f, palette, with_fx=False)
            tile = synth(rigs, d, act, f, palette)
            sizes = component_sizes(bare[..., 3] >= 128)
            print(f"sample {act} {d} {f:02d} comps={sizes[:4]} edge={touches_edge(tile)}", flush=True)
            tiles.append(tile[crop[1]:crop[3], crop[0]:crop[2]])
        cols = 8
        th, tw = tiles[0].shape[:2]
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * th, cols * tw, 4), (15, 20, 31, 255), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            m = t[..., 3] >= 128
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
        sheet = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        zt = []
        for act, d, fs in (("attack", "S", range(16)), ("hurt", "S", range(12)), ("walk", "S", range(0, 16, 2))):
            for f in fs:
                t = synth(rigs, d, act, f, palette)
                zt.append(t[CELL_H - 120:CELL_H - 12, CELL_W // 2 - 54:CELL_W // 2 + 54])
        zh, zw = zt[0].shape[:2]
        zc = 8
        zs = np.full((math.ceil(len(zt) / zc) * zh, zc * zw, 4), (15, 20, 31, 255), np.uint8)
        for i, t in enumerate(zt):
            r, c = divmod(i, zc)
            m = t[..., 3] >= 128
            zs[r * zh:(r + 1) * zh, c * zw:(c + 1) * zw][m] = t[m]
        Image.fromarray(np.repeat(np.repeat(zs, 3, 0), 3, 1), "RGBA").save(work / "zoom.png")
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
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions_dir = SRC_DIR / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    preview.save(actions_dir / "preview.png")
    motion_contact(sheets, "S").save(OUT_DIR / "preview_motion.png")
    motion_contact(sheets, "W").save(work / "review_W.png")

    idle0 = sheets["idle"][:CELL_H, :CELL_W]
    bb = body_box(idle0)
    sprite_h = bb[3] - bb[1] + 1
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]).tolist())
    edge = []
    split = []
    still_diff = 0
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                if touches_edge(tile):
                    edge.append(f"{act}:{d}:{f}")
                bare = synth(rigs, d, act, f, palette, with_fx=False)
                sizes = component_sizes(bare[..., 3] >= 128)
                if len([s for s in sizes if s >= 20]) != 1:
                    split.append({"a": act, "d": d, "f": f, "sizes": sizes[:5]})
            if act != "attack":
                t0 = sheets[act][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
                still_diff += int(np.any(t0 != rigs[d]["rest"], axis=2).sum())
    report = {
        "version": VERSION, "hash": digest, "parts": info, "sprite_h": int(sprite_h),
        "off_palette": len(off), "alphas": alphas, "edge_touching": len(edge),
        "edge_list": edge[:20], "still_diff": still_diff, "split": split[:20],
        "split_n": len(split), "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "hover"), ("walk", "gait")):
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
            else:
                report[f"gait_{act}"] = "no report"
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False)[:1500], flush=True)
    meta = {
        "generator": "forge_gallade_actions", "version": VERSION, "key": KEY, "dex": 475,
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(sprite_h),
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "still_faces": "right",
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "source": "front.png 96x96", "gifs_untouched": True,
        "roar": "existing_crest_flare_plus_psychic_rings",
        "attack": "pirouette turn (facing mirror, no in-plane spin) + elbow-blade slash",
        "layout": "actions_8dir", "sheetDir": "agent-sprite-forge", "raw": 1,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    failed = report["off_palette"] or report["edge_touching"] or report["still_diff"] or report["split_n"] or alphas != [0, 255]
    if failed:
        print("QA FAILED", flush=True)
        return
    if args.publish:
        publish_dir8(sprite_h)
        vals = publish_bounds(sheets)
        update_queue("done")
        print("published dir8", VERSION, "h", sprite_h, "bounds", vals[:4], flush=True)


if __name__ == "__main__":
    main()
