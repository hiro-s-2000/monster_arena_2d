#!/usr/bin/env python3
"""Regigigas (0486) normal 8-direction action sheet from this form's front.png / back.png.

Measured on normal/front.png (96x96, alpha 0/255, 15 colours):
  bbox x9-86 y14-81. Heavy biped golem (biped_kaiju_stomper without a tail).
  No mouth and no eyes with pupils: the face is two columns of three dots
  (red x35-37 / x55-57 y34-36, blue y40-42, grey y46-48) either side of the
  gold centre stripe. Faces slightly screen-left: the dot pair centre (x47)
  sits left of the torso centre (x52) and the left hand reaches further out.
  arm_l   left gold shoulder plate x18-31 y19-50 + hanging gauntlet hand
          x9-32 y43-62 (one piece, pivot at the shoulder socket ~x31 y27)
  arm_r   striped upper arm x71-86 y28-45 that leaves the right plate, and
          the gold-cuffed hand x60-82 y44-61 (pivot at the socket ~x73 y29).
          The right plate itself stays on the torso.
  leg_l   x24-45 y61-80 (white thigh, moss foot)
  leg_r   x55-82 y62-81
  torso   gold dome, moss tufts, face dots, right plate, lavender belly.

back.png (96x96, 12 colours): left plate + hand x7-22, right plate ring
x52-69 y13-50 with the arm inside it, arm and gold-cuffed hand x60-88 y26-61,
legs below y63.

Directions: front still S SW W unflipped, SE E mirrored. back still N NE
unflipped, NW mirrored. E looks screen-right, W screen-left, N is the back.

  idle   : heavy breathing (torso 1px), hands sway 1-2 deg, the face dots
           light one pair after another. Frame 0 and the loop end are the still.
  walk   : heavy alternating stomps (lift + hip swing), arms swing against
           the legs, 1px body bob, dust at each landing.
  attack : 0-4 roar without a mouth: both arms flare outward, all six dots
           blaze and white/cyan rings leave the face (back: from the dome).
           5-6 arms raised high (<= 34 deg), 7-8 double-hand ground slam
           (arms swing in and drop, body dips), 9-12 shock rings, cracks and
           flying rock chunks, 13-15 recover. No whole-body rotation.
  hurt   : 12 frames, white flash, one-piece recoil (<= 16 deg) around the
           rear foot, arms flung out, sparks, recover.

Palette-locked to each still. front/back png and gif are not written.
Resumable via agent-sprite-forge/_work. --force clears only this form's frames.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_regigigas_actions.py [--samples]
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
SRC_DIR = ROOT / "monsterSprite/gen4/0486_regigigas/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0486_regigigas/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
VIEW = {
    "S": ("front", False), "SW": ("front", False), "W": ("front", False),
    "SE": ("front", True), "E": ("front", True),
    "N": ("back", False), "NE": ("back", False), "NW": ("back", True),
}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "regigigas_v1"
KEY = "0486_regigigas"

INK = (16, 24, 16)
WHITE = (246, 246, 246)
LAV = (213, 205, 213)
GOLD = (197, 156, 82)
GOLD_L = (246, 213, 90)
GOLD_D = (123, 98, 32)
MOSS = (32, 131, 106)
MOSS_L = (74, 164, 98)
SLATE = (57, 74, 90)
CYAN = (90, 197, 246)
RED = (238, 90, 106)
GREY = (180, 189, 197)
DOTS = (RED, CYAN, GREY)

PARTS = ("arm_l", "arm_r", "leg_l", "leg_r")


# ---------------------------------------------------------------- helpers
def flood(seed_yx, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow)
    y, x = seed_yx
    if not allow[y, x]:
        raise SystemExit(f"seed {seed_yx} is not inside its allow mask")
    q = deque([(y, x)])
    seen[y, x] = True
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
    """4x nearest. Rigid, exact colours. Positive angle turns a point left of
    the pivot upward (a point below the pivot swings toward -x)."""
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
    for i in range(0, len(ty), 400):
        sl = slice(i, i + 400)
        d = (ty[sl, None].astype(np.int32) - ys[None, :].astype(np.int32)) ** 2
        d += (tx[sl, None].astype(np.int32) - xs[None, :].astype(np.int32)) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def fill_pinholes(img):
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
    flat[sel] = mapped[inv.reshape(-1)]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    out[~a] = 0
    return out


def put(out, x, y, col, only_empty=False):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        if only_empty and out[y, x, 3] >= 128:
            return
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


def top_pivot(part):
    ys, xs = np.nonzero(part)
    top = ys <= int(ys.min()) + 1
    return float(np.median(xs[top])), float(ys.min())


def add_socket(host, host_mask, part_mask, canvas, n=4):
    zone = part_mask & dilate(host_mask, n)
    if not np.any(zone):
        return
    fill = nearest_color(canvas, zone, host_mask)
    write = zone & (host[..., 3] < 128)
    host[write] = fill[write]


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


# ---------------------------------------------------------------- segment
def gold_mask(img, cols):
    m = np.zeros(img.shape[:2], bool)
    for c in cols:
        m |= is_col(img, c)
    return m


def segment(img, kind):
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    if kind == "front":
        # left plate + hand: everything left of the plate's inner outline
        bx = np.where(yy <= 45, 31, 32)
        arm_l = flood((55, 15), a & (xx <= bx) & (yy <= 62) & (yy >= 14))
        # right plate (gold, above the cuff) stays on the torso
        gold = gold_mask(img, (GOLD, GOLD_L, GOLD_D))
        plate_r = flood((33, 68), gold & (xx >= 62) & (yy <= 45))
        allow_r = a & ~plate_r & (yy <= 61) & (
            ((yy >= 28) & (yy <= 43) & (xx >= 72)) | ((yy >= 44) & (xx >= 60)))
        arm_r = flood((52, 66), allow_r)
        leg_l = flood((75, 30), a & (yy >= 61) & (xx >= 24) & (xx <= 46) & ~arm_l)
        leg_r = flood((75, 70), a & (yy >= 62) & (xx >= 55) & ~arm_r)
    else:
        gold = gold_mask(img, (GOLD, GOLD_L, GOLD_D))
        plate_l = gold & (xx <= 24)
        bound = np.full(h, -1)
        for y in range(h):
            xs = np.nonzero(plate_l[y])[0]
            if len(xs):
                bound[y] = xs.max() + 1
        rows = np.nonzero(bound >= 0)[0]
        for y in range(int(rows.min()), int(rows.max()) + 1):
            if bound[y] < 0:  # a plate row without gold (outline only)
                bound[y] = max(bound[y - 1], bound[y + 1])
        bx = bound[:, None] * np.ones((1, w), int)
        arm_l = flood((45, 9), a & (xx <= bx) & (yy <= 62))
        plate_r = flood((25, 58), gold & (xx >= 50) & (xx <= 70) & (yy <= 50))
        arm_r = flood((45, 80), a & ~plate_r & (yy >= 24) & (yy <= 61) & (xx >= 59))
        leg_l = flood((75, 20), a & (yy >= 63) & (xx <= 36) & ~arm_l)
        leg_r = flood((75, 60), a & (yy >= 63) & (xx >= 44) & ~arm_r)
    parts = {"arm_l": arm_l, "arm_r": arm_r, "leg_l": leg_l, "leg_r": leg_r}
    cover = np.zeros_like(a)
    for name, m in parts.items():
        if np.any(cover & m):
            raise SystemExit(f"{kind}: {name} overlaps another part")
        cover |= m
    parts = {"torso": a & ~cover, **parts}
    if kind == "front":
        dots = gold_mask(img, DOTS)
        if np.any(dots & ~parts["torso"]):
            raise SystemExit("a face dot left the torso")
    for name in PARTS:
        if int(parts[name].sum()) < 60:
            raise SystemExit(f"{kind}: {name} collapsed ({int(parts[name].sum())})")
    return parts


def mask_overlay(img, parts, path):
    cols = {"torso": (90, 90, 90), "arm_l": (40, 200, 60), "arm_r": (60, 120, 255),
            "leg_l": (240, 200, 40), "leg_r": (230, 60, 200)}
    out = np.zeros(img.shape[:2] + (3,), np.uint8)
    for name, m in parts.items():
        out[m] = cols[name]
    both = np.concatenate([img[..., :3] * (img[..., 3:] >= 128), out], axis=1)
    Image.fromarray(both.astype(np.uint8)).resize((both.shape[1] * 5, both.shape[0] * 5), Image.NEAREST).save(path)


# ---------------------------------------------------------------- rig
def placement(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return y0, y1, x0, x1, (CELL_W - (x1 - x0)) // 2, FOOT_Y - (y1 - y0)


def placed_still(img, flip=False):
    y0, y1, x0, x1, dx, dy = placement(img)
    out = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    out[dy:dy + (y1 - y0), dx:dx + (x1 - x0)] = img[y0:y1, x0:x1]
    return out[:, ::-1].copy() if flip else out


SHOULDER = {
    # (x, y) in the 96x96 still: the socket where each arm leaves its plate
    "front": {"arm_l": (31.0, 27.0), "arm_r": (73.0, 30.0)},
    "back": {"arm_l": (16.0, 26.0), "arm_r": (63.0, 32.0)},
}


def build_rig(img, masks, kind, flip):
    y0, y1, x0, x1, dst_x, dst_y = placement(img)
    sh, sw = y1 - y0, x1 - x0
    canvas = placed_still(img)
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
    under = np.zeros_like(canvas)
    for name in PARTS:
        add_socket(under, body, placed[name], canvas)
    layers["under"] = under

    to_cell = lambda p: (p[0] - x0 + dst_x, p[1] - y0 + dst_y)  # noqa: E731
    pivots = {k: to_cell(v) for k, v in SHOULDER[kind].items()}
    for name in ("leg_l", "leg_r"):
        pivots[name] = top_pivot(placed[name])
    feet = {}
    for name in ("leg_l", "leg_r"):
        ly, lx = np.nonzero(placed[name])
        low = ly >= int(ly.max()) - 1
        feet[name] = (float(np.median(lx[low])), float(ly.max()))
    hands = {}
    for name in ("arm_l", "arm_r"):
        ly, lx = np.nonzero(placed[name])
        low = ly >= int(ly.max()) - 6
        hands[name] = (float(lx[low].mean()), float(ly.max()))
    opy, opx = np.nonzero(canvas[..., 3] >= 128)
    ground = int(opy.max())
    face = -1.0 if kind == "front" else 1.0
    rear = max(feet.values(), key=lambda p: -face * p[0])
    dots = {}
    if kind == "front":
        for i, c in enumerate(DOTS):
            dots[i] = is_col(canvas, c)
        fy, fx = np.nonzero(dots[0] | dots[1] | dots[2])
        face_pt = (float(fx.mean()), float(fy.mean()))
    else:
        dome = canvas[..., 3] >= 128
        ys, xs = np.nonzero(dome)
        top = ys <= int(ys.min()) + 8
        face_pt = (float(xs[top].mean()), float(ys.min()) + 8)
    if flip:
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        mx = lambda p: (CELL_W - 1 - p[0], p[1])  # noqa: E731
        pivots = {k: mx(v) for k, v in pivots.items()}
        feet = {k: mx(v) for k, v in feet.items()}
        hands = {k: mx(v) for k, v in hands.items()}
        rear = mx(rear)
        face_pt = mx(face_pt)
        dots = {k: v[:, ::-1].copy() for k, v in dots.items()}
        face = -face
    # outward sign per arm: the hand swings away from the body centre
    cx = (opx.min() + opx.max()) / 2.0
    if flip:
        cx = CELL_W - 1 - cx
    out_sign = {}
    for name in ("arm_l", "arm_r"):
        out_sign[name] = -1.0 if hands[name][0] < cx else 1.0
    if kind == "front":
        draw = ["under", "leg_l", "leg_r", "torso", "arm_l", "arm_r"]
    else:
        draw = ["under", "leg_l", "leg_r", "torso", "arm_l", "arm_r"]
    return {
        "kind": kind, "flip": flip, "layers": layers, "pv": pivots, "face": face,
        "ground": ground, "feet": feet, "hands": hands, "rear": rear, "dots": dots,
        "face_pt": face_pt, "out": out_sign, "draw": draw,
    }


# ---------------------------------------------------------------- poses
def pose(action, f):
    n = FRAMES[action]
    p = dict(
        bdy=0, bdx=0, arm_l=0.0, arm_r=0.0, hand_dy=0, leg_l=0.0, leg_r=0.0,
        lift_l=0, lift_r=0, dots=-1, blaze=0, rings=0, ring_age=0, dust=0,
        slam=0, shock=0, rocks=0, hurt=0.0, hdx=0, flash=0, sparks=0, aura=0,
    )
    if action == "idle":
        p["bdy"] = -int(round(max(0.0, math.sin(2 * math.pi * f / n)) * 1.0))
        p["arm_l"] = wave(f, n, 2.5, phase=0.4)
        p["arm_r"] = wave(f, n, 2.5, phase=0.4)
        # dots light one pair after another: red, blue, grey
        p["dots"] = {4: 0, 5: 0, 6: 1, 7: 1, 8: 2, 9: 2}.get(f, -1)
    elif action == "walk":
        ph = 2 * math.pi * 2 * f / n
        sw = math.sin(ph)
        sw_c = math.cos(ph)
        p["leg_l"] = 8.0 * sw
        p["leg_r"] = -8.0 * sw
        p["lift_l"] = int(round(4 * max(0.0, sw_c)))
        p["lift_r"] = int(round(4 * max(0.0, -sw_c)))
        p["bdy"] = -1 if abs(sw_c) > 0.7 else 0
        p["arm_l"] = -6.0 * sw
        p["arm_r"] = 6.0 * sw
        # dust on landing (just after each foot comes down)
        p["dust"] = {4: 1, 5: 2, 12: 1, 13: 2}.get(f, 0)
        p["dust_foot"] = "leg_l" if f in (4, 5) else "leg_r"
    elif action == "attack":
        raise_ = (6, 14, 22, 27, 30, 33, 34, 4, -10, -12, -12, -10, -6, -3, -1, 0)
        hdy = (0, 0, -1, -1, -1, -2, -2, 2, 3, 3, 3, 2, 1, 0, 0, 0)
        bdy = (0, -1, -1, -2, -2, -2, -2, 1, 2, 2, 2, 1, 1, 0, 0, 0)
        p["arm_l"] = p["arm_r"] = raise_[f]
        p["hand_dy"] = hdy[f]
        p["bdy"] = bdy[f]
        p["bdx"] = (0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 1, 0, 0, 0)[f]  # toward the face
        p["blaze"] = 1 if 1 <= f <= 10 else 0
        if 1 <= f <= 5:
            p["rings"] = 3 if f >= 2 else 2
            p["ring_age"] = f - 1
        p["aura"] = f - 4 if 5 <= f <= 6 else 0
        p["slam"] = 1 if f in (7, 8) else 0
        p["shock"] = f - 7 if 8 <= f <= 12 else 0
        p["rocks"] = f - 7 if 8 <= f <= 13 else 0
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 16.0 * env
        p["hdx"] = int(round(3.0 * env))
        p["arm_l"] = p["arm_r"] = 14.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = f if 2 <= f <= 6 else 0
    return p


# ---------------------------------------------------------------- effects
def roar_rings(out, center, face, age, strength):
    mx, my = center
    cols = (WHITE, CYAN, GREY)
    for r_i in range(strength):
        rad = 8 + age * 4 + r_i * 5
        col = cols[min(r_i, 2)]
        for deg in range(-60, 61, 4):
            ang = math.radians(deg)
            put(out, mx + face * rad * math.cos(ang), my + rad * math.sin(ang) * 0.85, col, True)
            if r_i == 0 and abs(deg) <= 40:
                put(out, mx + face * (rad + 1) * math.cos(ang), my + (rad + 1) * math.sin(ang) * 0.85, CYAN, True)


def aura_spark(out, center, k):
    cx, cy = center
    for i in range(6):
        ang = i * math.pi / 3 + k * 0.4
        for r in range(4 + k * 2, 8 + k * 3):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), (WHITE, GOLD_L)[r % 2], True)


def ground_shock(out, center, age, ground):
    fx = center[0]
    for i, col in enumerate((WHITE, GOLD_L, GREY)):
        rx = 8 + age * 7 + i * 4
        ry = max(2, rx * 0.25)
        if rx > 80:
            continue
        for deg in range(0, 360, 5):
            ang = math.radians(deg)
            x = fx + rx * math.cos(ang)
            y = ground + 1 + ry * math.sin(ang)
            if y > ground - 3:
                put(out, x, y, col, True)


def cracks(out, center, ground, k):
    fx = center[0]
    for sgn in (-1, 1):
        x, y = fx, ground + 2
        for i in range(6 + k * 3):
            x += sgn * 1
            if i % 3 == 2:
                y += (1 if (i // 3) % 2 == 0 else -1)
            put(out, x, y, INK, False)


def slam_flash(out, center, ground):
    cx = center[0]
    for k in range(8):
        ang = math.pi + k * math.pi / 7
        for r in range(3, 13, 1):
            put(out, cx + r * math.cos(ang) * 1.4, ground - 2 + r * math.sin(ang) * 0.9,
                (WHITE, GOLD_L)[(r // 2) % 2], True)


def rock(out, cx, cy, r, col):
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if abs(dx) + abs(dy) <= r:
                edge = abs(dx) + abs(dy) == r
                put(out, cx + dx, cy + dy, INK if edge else col, True)


def rocks(out, center, ground, k):
    cx = center[0]
    specs = ((-14, 1.6, GOLD), (-6, 2.4, MOSS), (5, 2.2, SLATE), (13, 1.5, GOLD), (-20, 1.1, MOSS_L), (19, 1.2, SLATE))
    for i, (vx, vy, col) in enumerate(specs):
        t = k
        x = cx + vx * t * 0.6
        y = ground - 2 - (vy * 5 * t - 1.6 * t * t)
        if y > ground:
            continue
        rock(out, x, y, 3 if i < 2 else 2, col)


def dust(out, foot, k):
    fx, fy = foot
    for sgn in (-1, 1):
        for i in range(3):
            x = fx + sgn * (9 + k * 3 + i * 2)
            y = fy - i - (k - 1)
            put(out, x, y, GREY if i else WHITE, True)
            put(out, x + sgn, y, GREY, True)


def sparks(out, k, face, box):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    pts = ((cx + face * (10 + k * 3), y0 + 8 - k), (cx - face * (4 + k * 2), y0 - 2 - k),
           (cx + face * 3, y0 + 18 - k))
    for sx, sy in pts:
        r = 2 if k % 2 else 3
        for d in range(-r, r + 1):
            put(out, sx + d, sy, WHITE, True)
            put(out, sx, sy + d, GOLD_L if abs(d) == r else WHITE, True)


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    ink = is_col(out, INK)
    out[a & ~ink, :3] = WHITE
    return out


def light_dots(img, rig, which, bdx, bdy, full):
    """Light one dot pair (which=0,1,2) or all of them (full). The dot turns
    white and its dark outline ring takes the dot colour: integer, in place."""
    if rig["kind"] != "front":
        return
    picks = (0, 1, 2) if full else (which,)
    for i in picks:
        d = rig["dots"].get(i)
        if d is None or not np.any(d):
            continue
        m = shift(np.dstack([d.astype(np.uint8)] * 4), bdx, bdy)[..., 0] > 0
        ring = dilate(m, 1) & ~m & (img[..., 3] >= 128)
        ring &= ~is_col(img, WHITE)
        img[ring, :3] = DOTS[i]
        img[m, :3] = WHITE


# ---------------------------------------------------------------- synth
def part_moved(layer, pivot, ang, dx, dy):
    lyr = layer
    if abs(ang) >= 0.45:
        lyr = fill_pinholes(rotate_nearest(layer, ang, pivot))
    return shift(lyr, dx, dy)


def assemble_body(rig, p):
    L, pv = rig["layers"], rig["pv"]
    face = rig["face"]
    bdx, bdy = int(round(face * p["bdx"])), int(p["bdy"])
    moved = {}
    for name in ("under", "torso"):
        moved[name] = shift(L[name], bdx, bdy)
    for arm in ("arm_l", "arm_r"):
        # outward = hand away from the body. rotate_nearest's positive angle
        # lifts a point left of the pivot (a hanging hand swings to -x), so a
        # hand left of centre goes out with +, one right of centre with -.
        ang = p[arm] if rig["out"][arm] < 0 else -p[arm]
        moved[arm] = part_moved(L[arm], pv[arm], ang, bdx, bdy + int(p["hand_dy"]))
    fwd = face
    for leg, key in (("leg_l", "lift_l"), ("leg_r", "lift_r")):
        deg = -fwd * p[leg]
        moved[leg] = part_moved(L[leg], pv[leg], deg, bdx, -int(p[key]))
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in rig["draw"]:
        paste(img, moved[name])
    if p["blaze"]:
        light_dots(img, rig, 0, bdx, bdy, True)
    elif p["dots"] >= 0:
        light_dots(img, rig, p["dots"], bdx, bdy, False)
    return img


def slam_point(rig, p):
    face = rig["face"]
    hx = [rig["hands"][a][0] for a in ("arm_l", "arm_r")]
    x = sum(hx) / 2.0 + face * (p["bdx"] + 6)
    return (x, rig["ground"])


def synth(rig, action, f, palette, effects=True):
    p = pose(action, f)
    face = rig["face"]
    if p["hurt"] > 0.45 or p["hdx"]:
        base = assemble_body(rig, p)
        img = rotate_nearest(base, -face * p["hurt"], rig["rear"])
        img = shift(img, -face * p["hdx"], 0)
        img = despeckle(fill_pinholes(img))
    else:
        img = assemble_body(rig, p)
    img = snap(img, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    bdx, bdy = int(round(face * p["bdx"])), int(p["bdy"])
    fp = (rig["face_pt"][0] + bdx, rig["face_pt"][1] + bdy)
    if p["rings"]:
        roar_rings(img, (fp[0] + face * 6, fp[1]), face, p["ring_age"], p["rings"])
    if p["aura"]:
        for arm in ("arm_l", "arm_r"):
            bb = body_box(shift(rig["layers"][arm], 0, 0))
            hx = rig["hands"][arm][0] + rig["out"][arm] * 10
            aura_spark(img, (hx, rig["hands"][arm][1] - 26), p["aura"])
    sp = slam_point(rig, p)
    if p["slam"]:
        slam_flash(img, sp, rig["ground"])
    if p["shock"]:
        ground_shock(img, sp, p["shock"], rig["ground"])
        cracks(img, sp, rig["ground"], min(p["shock"], 3))
    if p["rocks"]:
        rocks(img, sp, rig["ground"], p["rocks"])
    if p.get("dust"):
        dust(img, rig["feet"][p["dust_foot"]], p["dust"])
    if p["sparks"]:
        bb = body_box(img)
        if bb:
            sparks(img, p["sparks"], face, bb)
    return snap(img, palette)


# ---------------------------------------------------------------- output
def params_hash():
    src = (Path(__file__).read_bytes() + (SRC_DIR / "front.png").read_bytes()
           + (SRC_DIR / "back.png").read_bytes())
    return hashlib.sha1(src).hexdigest()[:12]


def stack_sheets(frames_dir):
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
            bg[ai * CELL_H:(ai + 1) * CELL_H, si * CELL_W:(si + 1) * CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    """Every frame of every action, cropped, for S (front) and N (back)."""
    pad = 3
    rows = []
    for di in (0, 4):
        for act in ACTIONS:
            row = []
            band = sheets[act][di * CELL_H:(di + 1) * CELL_H]
            for f in range(FRAMES[act]):
                tile = band[:, f * CELL_W:(f + 1) * CELL_W]
                a = tile[..., 3] >= 128
                if not np.any(a):
                    continue
                ys, xs = np.nonzero(a)
                y0, y1 = max(0, int(ys.min()) - pad), min(CELL_H, int(ys.max()) + 1 + pad)
                x0, x1 = max(0, int(xs.min()) - pad), min(CELL_W, int(xs.max()) + 1 + pad)
                row.append(tile[y0:y1, x0:x1])
            rows.append(row)
    ch = max(c.shape[0] for r in rows for c in r)
    cw = max(c.shape[1] for r in rows for c in r)
    ncol = max(len(r) for r in rows)
    bg = np.full((len(rows) * ch, ncol * cw, 4), (15, 20, 31, 255), np.uint8)
    for ri, r in enumerate(rows):
        for ci, c in enumerate(r):
            y = ri * ch + (ch - c.shape[0])
            x = ci * cw + (cw - c.shape[1]) // 2
            m = c[..., 3] >= 128
            bg[y:y + c.shape[0], x:x + c.shape[1]][m] = c[m]
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


def body_pieces(img):
    return [s for s in components(img[..., 3] >= 128) if s > 6]


def diff_count(a_img, b_img):
    d = np.any(a_img != b_img, axis=2) & ((a_img[..., 3] >= 128) | (b_img[..., 3] >= 128))
    return int(d.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    stills = {k: np.array(Image.open(SRC_DIR / f"{k}.png").convert("RGBA")) for k in ("front", "back")}
    for k, im in stills.items():
        if im.shape[:2] != (96, 96):
            raise SystemExit(f"expected 96x96 {k}.png, got {im.shape}")
    palettes = {k: extract_palette(im) for k, im in stills.items()}
    masks = {k: segment(im, k) for k, im in stills.items()}
    counts = {k: {n: int(m.sum()) for n, m in v.items()} for k, v in masks.items()}
    print("parts", json.dumps(counts), flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    for k in ("front", "back"):
        mask_overlay(stills[k], masks[k], work / f"mask_{k}.png")

    rigs = {d: build_rig(stills[VIEW[d][0]], masks[VIEW[d][0]], VIEW[d][0], VIEW[d][1]) for d in DIRS}
    for d in ("S", "N", "E"):
        print(d, "pivots", {k: tuple(round(v, 1) for v in p) for k, p in rigs[d]["pv"].items()},
              "out", rigs[d]["out"], "rear", rigs[d]["rear"], flush=True)

    bad = {}
    for d in DIRS:
        kind, flip = VIEW[d]
        ref = placed_still(stills[kind], flip)
        n = diff_count(synth(rigs[d], "idle", 0, palettes[kind]), ref)
        if n:
            bad[d] = n
    print("still diff", bad if bad else 0, flush=True)
    if bad:
        raise SystemExit("idle frame 0 does not match the still")

    broken = []
    for d in ("S", "N", "E"):
        kind = VIEW[d][0]
        for act in ACTIONS:
            for f in range(FRAMES[act]):
                fr = synth(rigs[d], act, f, palettes[kind], effects=False)
                pcs = body_pieces(fr)
                if len(pcs) != 1:
                    broken.append((d, act, f, pcs[:4]))
    print("piece check", broken if broken else "ok", flush=True)

    if args.samples:
        want = [("idle", "S", f) for f in (0, 4, 6, 8)]
        want += [("walk", "S", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in range(12)]
        want += [("walk", "N", f) for f in (0, 2, 4, 6)]
        want += [("attack", "N", f) for f in (2, 4, 6, 7, 8, 10, 12, 15)]
        want += [("attack", "E", f) for f in (3, 6, 8, 10)]
        want += [("idle", d, 0) for d in DIRS]
        tiles = [synth(rigs[d], act, f, palettes[VIEW[d][0]]) for act, d, f in want]
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
    if broken:
        raise SystemExit("a pose broke the body")

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
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palettes[VIEW[d][0]]), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"] = sorted(set(state.get("done", [])) | {f"{act}/{d}"})
            state_p.write_text(json.dumps(state))
            print(f"[{done}/{total}] {act}/{d}", flush=True)

    sheets = stack_sheets(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    (SRC_DIR / "actions").mkdir(exist_ok=True)
    preview.save(SRC_DIR / "actions" / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1)
    bounds = []
    for di in range(8):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        ys, xs = np.nonzero(t[..., 3] >= 128)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    union = np.unique(np.concatenate(list(palettes.values())), axis=0)
    pal = {tuple(int(v) for v in c) for c in union}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    off_view = 0
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            band = sheets[act][di * CELL_H:(di + 1) * CELL_H]
            pv = {tuple(int(v) for v in c) for c in palettes[VIEW[d][0]]}
            u = {tuple(int(v) for v in c) for c in np.unique(band[band[..., 3] >= 128][:, :3], axis=0)}
            off_view += len(u - pv)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    still_bad = 0
    for di, d in enumerate(DIRS):
        kind, flip = VIEW[d]
        still_bad += diff_count(sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W], placed_still(stills[kind], flip))
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "off_view_palette": off_view, "alphas": alphas,
        "edge_touching": sum(len(v) for v in clip.values()), "still_diff": still_bad,
        "bounds_s": bounds, "views": {d: list(VIEW[d]) for d in DIRS},
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
            else:
                report[f"gait_{act}"] = (r.stdout + r.stderr).strip()[-300:]
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False), flush=True)

    meta = {
        "generator": "forge_regigigas_actions", "version": VERSION, "key": KEY,
        "dex": 486, "form": "normal", "variant": "regigigas",
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "views": {d: {"still": VIEW[d][0], "mirrored": VIEW[d][1]} for d in DIRS},
        "archetype": "biped_kaiju_stomper (heavy golem, no tail, no mouth: face dots)",
        "off_palette_colors": len(off), "source": "front.png + back.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = bool(off or off_view or clip or still_bad or alphas not in ([0, 255], [0]))
    if failed:
        raise SystemExit("qa failed")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
