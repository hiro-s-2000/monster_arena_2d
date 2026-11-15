#!/usr/bin/env python3
"""Heatran (0485) normal 8-direction action sheet from this form's front.png / back.png.

Measured on normal/front.png (96x96, alpha 0/255, 2393 opaque, 14 colours):
  bbox x9-88 y21-75. Heavy crouched quadruped (quadruped_beast). The steel
  helmet head sits left of the red-brown shell, so the still faces screen-left
  (head centroid x is left of the body centroid).
  head    grey helmet x23-58 y24-47, side horns, two red eyes (255,82,57)
          at x31-34 / x45-49 y40-43. Stays on the torso, integer moves only.
  jaw     the row of white fangs under the eyes, x30-54 y48-53. It drops by
          whole pixels; the gap shows the ink mouth and a lava glow.
  shell   the red-brown dome x50-76 y21-60 with orange lava spots (torso).
  leg_l   left foreleg x11-29 y43-69 (grey shoulder, orange spot, toes).
  leg_c   right foreleg x56-76 y54-75 (white knee, maroon foot, lowest claw).
  leg_r   right hind leg x74-89 y41-68 (grey band, maroon foot, grey toes).
  The maroon lump under the belly (x30-43 y56-63) is the hidden fourth leg; it
  stays on the torso.

back.png (96x96, 2341 opaque, front palette + one grey) faces up-right: the
shell fills the top, the helmet horns peek out at the top-right (x49-73
y26-37), no mouth is drawn. Legs: leg_l x10-29 y44-66, leg_c x45-70 y52-71,
leg_r x70-90 y45-68 (the fore leg on that side).

Directions: front still S SW W unflipped, SE E mirrored. back still N NE
unflipped, NW mirrored. E looks screen-right, W screen-left, N is the back.

  idle   : slow heavy breathing (the upper body rises 1px, legs planted), the
           lava spots on the shell pulse. Frame 0 and the loop end are the still.
  walk   : diagonal trot (leg_l + leg_r against leg_c), lifted swing foot,
           1px stomp bob.
  attack : 0-4 roar: the fang jaw drops, the mouth glows with lava, the shell
           spots blaze, fire roar rings leave the mouth (back rows: from the
           helmet horns). 5-10 a magma-storm vortex circles the body (the body
           itself is not rotated, as the 2026-11-04 rule asks) while the
           foreleg rears up and slams down with flame arcs. 9-12 a lava
           eruption and ground shock under the claw. 13-15 recover.
  hurt   : 12 frames, white flash, recoil of the one-piece body (<= 16 deg)
           around the rear foot, embers, recover.

Palette-locked to each still. front/back png and gif are not written.
Resumable via agent-sprite-forge/_work. --force clears only this form's frames.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_heatran_actions.py [--samples] [--masks]
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen4_attack_strike  # noqa: E402

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0485_heatran/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0485_heatran/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# (still, mirrored)
VIEW = {
    "S": ("front", False), "SW": ("front", False), "W": ("front", False),
    "SE": ("front", True), "E": ("front", True),
    "N": ("back", False), "NE": ("back", False), "NW": ("back", True),
}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "heatran_v1"
KEY = "0485_heatran"

INK = (24, 24, 24)
LAVA_Y = (255, 164, 16)
LAVA_O = (205, 131, 41)
LAVA_R = (255, 82, 57)
LAVA_D = (180, 65, 41)
HOT_W = (255, 255, 255)
PALE = (230, 230, 238)
EYE = (255, 82, 57)

PARTS = ("leg_l", "leg_c", "leg_r", "jaw")

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
    the pivot upward (counter-clockwise on screen)."""
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


def top_pivot(part):
    """Hip/shoulder joint of a leg: middle of its top two rows."""
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


def segment(img, kind):
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    empty = np.zeros_like(a)
    if kind == "front":
        jaw = a & (yy >= 48) & (yy <= 53) & (xx >= 30) & (xx <= 54)
        leg_l = flood((66, 14), a & (((xx <= 28) & (yy >= 43)) | ((xx <= 32) & (yy >= 59))) & ~jaw)
        leg_r = flood((60, 84), a & (((xx >= 75) & (yy >= 42)) | ((xx >= 77) & (yy >= 58))))
        leg_c = flood((73, 66), a & (((yy >= 54) & (xx >= 60)) | (yy >= 63) & (xx >= 55)) & (xx <= 78) & ~leg_r)
    else:
        jaw = empty
        leg_l = flood((62, 15), a & (xx <= 28) & (yy >= 45))
        leg_r = flood((63, 85), a & (xx >= 71) & (yy >= 51))
        leg_c = flood((68, 55), a & (yy >= 56) & (xx >= 46) & (xx <= 70) & ~leg_r)
    parts = {"leg_l": leg_l, "leg_c": leg_c, "leg_r": leg_r, "jaw": jaw}
    cover = np.zeros_like(a)
    for name, m in parts.items():
        if np.any(cover & m):
            raise SystemExit(f"{kind}: {name} overlaps another part")
        cover |= m
    parts = {"torso": a & ~cover, **parts}
    if kind == "front":
        eye = is_col(img, EYE)
        if np.any(eye & ~parts["torso"]):
            raise SystemExit("an eye left the head")
    for name in ("leg_l", "leg_c", "leg_r"):
        if int(parts[name].sum()) < 40:
            raise SystemExit(f"{kind}: {name} collapsed ({int(parts[name].sum())})")
    return parts


def mask_overlay(img, parts, path):
    cols = {"torso": (90, 90, 90), "leg_l": (40, 200, 60), "leg_c": (60, 120, 255),
            "leg_r": (240, 200, 0), "jaw": (255, 0, 200)}
    out = np.zeros(img.shape[:2] + (3,), np.uint8) + 20
    for k, m in parts.items():
        out[m] = (np.array(cols[k]) * 0.6 + img[m][:, :3] * 0.4).astype(np.uint8)
    Image.fromarray(out).resize((img.shape[1] * 6, img.shape[0] * 6), Image.NEAREST).save(path)


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
    # joint cushions on their own layer under everything: a moved leg never
    # leaves a hole and the rest pose stays the exact still
    under = np.zeros_like(canvas)
    for name in ("leg_l", "leg_c", "leg_r"):
        add_socket(under, body, placed[name], canvas)
    layers["under"] = under

    pivots = {}
    for name in ("leg_l", "leg_c", "leg_r"):
        pivots[name] = top_pivot(placed[name])
    feet = {}
    for name in ("leg_l", "leg_c", "leg_r"):
        ly, lx = np.nonzero(placed[name])
        low = ly >= int(ly.max()) - 1
        feet[name] = (float(np.median(lx[low])), float(ly.max()))
    opy, opx = np.nonzero(canvas[..., 3] >= 128)
    ground = int(opy.max())
    # front faces left, back faces right (before mirroring)
    face = -1.0 if kind == "front" else 1.0
    rear = max(feet.values(), key=lambda p: -face * p[0])
    lava = is_col(canvas, LAVA_Y) | is_col(canvas, LAVA_O)
    if kind == "front":
        # helmet head: everything grey above the fang row and left of the shell
        head_box = (24, 47, 23, 58)
    else:
        head_box = (26, 37, 48, 74)
    hb = np.zeros((CELL_H, CELL_W), bool)
    hy0, hy1, hx0, hx1 = head_box
    hb[dst_y + hy0 - y0:dst_y + hy1 - y0 + 1, dst_x + hx0 - x0:dst_x + hx1 - x0 + 1] = True
    hb &= canvas[..., 3] >= 128
    if flip:
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        mx = lambda p: (CELL_W - 1 - p[0], p[1])  # noqa: E731
        pivots = {k: mx(v) for k, v in pivots.items()}
        feet = {k: mx(v) for k, v in feet.items()}
        rear = mx(rear)
        lava = lava[:, ::-1].copy()
        hb = hb[:, ::-1].copy()
        face = -face
    # strike claw: the foreleg nearest the facing side
    strike = "leg_l" if kind == "front" else "leg_r"
    draw = ["under", "leg_r", "leg_l", "torso", "leg_c", "jaw"] if kind == "front" else \
           ["under", "leg_l", "torso", "leg_r", "leg_c", "jaw"]
    return {
        "kind": kind, "flip": flip, "layers": layers, "pv": pivots, "face": face,
        "ground": ground, "feet": feet, "rear": rear, "lava": lava, "head": hb,
        "strike": strike, "draw": draw,
    }


# ---------------------------------------------------------------- poses
def pose(action, f):
    n = FRAMES[action]
    p = dict(
        bdy=0, bdx=0, jaw=0, leg_l=0.0, leg_c=0.0, leg_r=0.0,
        lift_l=0, lift_c=0, lift_r=0, strike=0.0, slift=0, brace=0.0,
        rings=0, ring_age=0, glow=0, pulse=0, vortex=-1, slash=0, shock=0,
        erupt=0, hurt=0.0, hdx=0, flash=0, embers=0, spin=0,
    )
    if action == "idle":
        # slow heavy breath: up one pixel in the middle of the loop
        p["bdy"] = -1 if 4 <= f <= 10 else 0
        p["pulse"] = 1 if f in (6, 7, 8, 9) else 0
    elif action == "walk":
        ph = 2 * math.pi * 2 * f / n
        sw = math.sin(ph)
        # diagonal pair A = leg_l + leg_r, B = leg_c, opposite phase
        p["leg_l"] = 14.0 * sw
        p["leg_r"] = 12.0 * sw
        p["leg_c"] = -14.0 * sw
        swing = math.cos(ph)  # > 0: pair A travels forward
        p["lift_l"] = int(round(3 * max(0.0, swing)))
        p["lift_r"] = int(round(2 * max(0.0, swing)))
        p["lift_c"] = int(round(3 * max(0.0, -swing)))
        p["bdy"] = -1 if abs(swing) > 0.7 else 0
    elif action == "attack":
        jaw = (1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 3, 2, 1, 0, 0)
        strike = (0, 0, 0, 2, 4, 9, 13, 15, 5, 2, 2, 2, 1, 0, 0, 0)
        slift = (0, 0, 0, 1, 2, 4, 6, 7, 1, 0, 0, 0, 0, 0, 0, 0)
        bdy = (0, -1, -1, -2, -2, -2, -2, -1, 1, 1, 0, 0, 0, 0, 0, 0)
        brace = (0, 0, 0, 0, 1, 3, 5, 5, 1, 0, 0, 0, 0, 0, 0, 0)
        p["jaw"], p["strike"], p["slift"], p["bdy"], p["brace"] = jaw[f], strike[f], slift[f], bdy[f], brace[f]
        if f <= 5:
            p["rings"] = 1 if f == 0 else (2 if f == 1 else 3)
            p["ring_age"] = f
        p["glow"] = 1 if 1 <= f <= 12 else 0
        p["vortex"] = f - 4 if 5 <= f <= 11 else -1
        p["slash"] = f - 7 if f in (8, 9) else 0
        p["shock"] = f - 8 if 9 <= f <= 13 else 0
        p["erupt"] = f - 8 if 9 <= f <= 12 else 0
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 16.0 * env
        p["hdx"] = int(round(3.0 * env))
        p["jaw"] = int(round(2.0 * env))
        p["leg_l"] = -6.0 * env
        p["leg_r"] = 4.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["embers"] = f if 2 <= f <= 7 else 0
    return p


# ---------------------------------------------------------------- effects
def mouth_point(rig, bdx, bdy, jaw):
    """Front: the middle of the dropped fang row. Back: the helmet horn tip."""
    face = rig["face"]
    if rig["kind"] == "front":
        lyr = shift(rig["layers"]["jaw"], bdx, bdy + jaw)
        ys, xs = np.nonzero(lyr[..., 3] >= 128)
        if len(xs) == 0:
            return None
        x = int(xs.min()) if face < 0 else int(xs.max())
        return x + int(face) * 2, int(ys.mean())
    hb = shift(np.dstack([rig["head"].astype(np.uint8)] * 4), bdx, bdy)[..., 0] > 0
    ys, xs = np.nonzero(hb)
    x = int(xs.max()) if face > 0 else int(xs.min())
    return x + int(face) * 2, int(ys.min()) + 4


def roar_rings(out, mouth, face, age, strength):
    mx, my = mouth
    cols = (LAVA_Y, LAVA_R, LAVA_D)
    for r_i in range(strength):
        rad = 5 + age * 3 + r_i * 4
        col = cols[min(r_i, 2)]
        for deg in range(-60, 61, 5):
            ang = math.radians(deg)
            put(out, mx + face * rad * math.cos(ang), my + rad * math.sin(ang) * 0.8, col, True)
            if r_i == 0 and abs(deg) <= 35:
                put(out, mx + face * (rad + 1) * math.cos(ang), my + (rad + 1) * math.sin(ang) * 0.8, HOT_W, True)


def vortex(out, box, k, face):
    """Magma storm: two flame streams circling the body (the body stays still)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, y0 + (y1 - y0) * 0.55
    rx = (x1 - x0) / 2.0 + 7 + min(k, 3)
    ry = rx * 0.36
    turn = -face * k * 0.85  # spins toward the facing side on the near half
    for s in range(2):
        for i in range(26):
            t = turn + s * math.pi + i * 0.075
            x = cx + rx * math.cos(t)
            y = cy + ry * math.sin(t) - i * 0.25 * (1 if s == 0 else -1)
            col = (HOT_W, LAVA_Y, LAVA_Y, LAVA_O, LAVA_R, LAVA_D)[min(5, i // 5)]
            put(out, x, y, col, True)
            if i < 14:
                put(out, x, y - 1, LAVA_Y if i > 3 else HOT_W, True)
    # rising cinders above the shell
    for j in range(5):
        ex = cx + (j - 2) * (rx / 2.6) + math.sin(k + j) * 3
        ey = y0 - 3 - ((k * 3 + j * 5) % 14)
        put(out, ex, ey, LAVA_Y if j % 2 else LAVA_R, True)


def slash_arcs(out, foot, face, k):
    fx, fy = foot
    for i in range(3):
        r = 9 + i * 3 + k * 2
        for deg in range(-75, 25, 6):
            ang = math.radians(deg)
            x = fx + face * (4 + r * math.cos(ang) * 0.7)
            y = fy - 10 + r * math.sin(ang)
            put(out, x, y, (LAVA_O, HOT_W, LAVA_Y)[i], True)


def ground_shock(out, foot, age, ground):
    fx, _ = foot
    for i, col in enumerate((LAVA_Y, LAVA_R, LAVA_D)):
        rx = 6 + age * 6 + i * 4
        ry = max(2, rx * 0.26)
        if rx > 70:
            continue
        for deg in range(0, 360, 6):
            ang = math.radians(deg)
            x = fx + rx * math.cos(ang)
            y = ground + 1 + ry * math.sin(ang)
            if y > ground - 2:
                put(out, x, y, col, True)


def eruption(out, foot, face, k, ground):
    """Lava pillars bursting out of the ground in front of the claw."""
    fx, _ = foot
    heights = (10, 18, 24, 16) [k - 1]
    for j, off in enumerate((4, 11, 18, 25)):
        hgt = int(heights * (1.0 - 0.18 * j))
        px = fx + face * off
        for yv in range(hgt):
            y = ground - yv
            frac = yv / max(1, hgt)
            col = HOT_W if frac > 0.85 else (LAVA_Y if frac > 0.5 else (LAVA_O if frac > 0.2 else LAVA_R))
            wob = int(round(math.sin(yv * 0.7 + j + k) * 1))
            put(out, px + wob, y, col, True)
            put(out, px + wob + 1, y, LAVA_R if frac < 0.6 else LAVA_Y, True)
        for d in range(3):
            put(out, px + (d - 1) * 2, ground - hgt - 2 - (k + d) % 3, LAVA_Y, True)


def embers(out, frame_idx, face, box):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    d = frame_idx
    pts = ((cx + face * (8 + d * 2), y0 + 4 - d), (cx - face * (4 + d), y0 - 2 - d),
           (cx + face * 18, y0 + 12 - d), (cx - face * 16, y0 + 8 - d))
    for i, (sx, sy) in enumerate(pts):
        put(out, sx, sy, LAVA_Y if (d + i) % 2 else LAVA_R, True)
        if d % 2 == 0:
            put(out, sx, sy - 1, HOT_W, True)


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    ink = is_col(out, INK)
    out[a & ~ink, :3] = PALE
    return out


# ---------------------------------------------------------------- synth
def part_moved(layer, pivot, ang, dx, dy):
    lyr = layer
    if abs(ang) >= 0.45:
        lyr = fill_pinholes(rotate_nearest(layer, ang, pivot))
    return shift(lyr, dx, dy)


def assemble_body(rig, p):
    L, pv, face = rig["layers"], rig["pv"], rig["face"]
    bdx, bdy = int(p["bdx"]), int(p["bdy"])
    fwd = -face  # positive angle moves a hanging foot toward -x
    legs = {}
    for leg, key in (("leg_l", "lift_l"), ("leg_c", "lift_c"), ("leg_r", "lift_r")):
        deg = fwd * p[leg]
        lift = int(p[key])
        if leg == rig["strike"]:
            deg += fwd * p["strike"]
            lift += int(p["slift"])
        elif p.get("brace"):
            deg -= fwd * p["brace"]
        legs[leg] = (deg, lift)
    moved = {
        "under": shift(L["under"], bdx, bdy),
        "torso": shift(L["torso"], bdx, bdy),
        "jaw": shift(L["jaw"], bdx, bdy + int(p["jaw"])),
    }
    for leg, (deg, lift) in legs.items():
        moved[leg] = part_moved(L[leg], pv[leg], deg, bdx, -lift)
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in rig["draw"]:
        if name == "jaw" and rig["kind"] == "front" and int(p["jaw"]) > 0:
            # open mouth: the strip the fang row left is ink with lava inside
            home = shift(L["jaw"], bdx, bdy)
            hm = home[..., 3] >= 128
            jm = moved["jaw"][..., 3] >= 128
            ys, xs = np.nonzero(hm)
            top = int(ys.min())
            span = np.zeros_like(hm)
            for x in np.unique(xs):
                col = hm[:, x]
                yt = int(np.nonzero(col)[0].min())
                span[yt:yt + int(p["jaw"]) + 1, x] = True
            hole = span & ~jm
            img[hole, :3] = INK
            img[hole, 3] = 255
            jd = int(p["jaw"])
            if jd >= 2:
                # lava glow in the lower rows of the open mouth, ink rim on top
                above = np.zeros_like(hole)
                above[1:] = hole[:-1]
                inner = hole & above
                if p["glow"]:
                    # keep an ink rim at both mouth corners
                    hx = np.nonzero(hole.any(axis=0))[0]
                    inner[:, :int(hx.min()) + 3] = False
                    inner[:, int(hx.max()) - 2:] = False
                    img[inner, :3] = LAVA_O
                    low = inner.copy()
                    low[:-1] &= ~inner[1:]
                    img[inner & ~low, :3] = LAVA_Y
        paste(img, moved[name])
    if p["glow"] or p["pulse"]:
        lava = shift(np.dstack([rig["lava"].astype(np.uint8)] * 4), bdx, bdy)[..., 0] > 0
        lava &= img[..., 3] >= 128
        if p["glow"]:
            img[lava & is_col(img, LAVA_O), :3] = LAVA_Y
            core = lava & ~dilate(~lava, 1)
            img[core, :3] = HOT_W
        else:
            img[lava & is_col(img, LAVA_O), :3] = LAVA_Y
    return img


def synth(rig, action, f, palette, effects=True):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)
    face = rig["face"]
    if p["hurt"] > 0.45 or p["hdx"]:
        base = assemble_body(rig, p)
        # head goes up and back: a point on the facing side rises
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
    box = body_box(img)
    if p["rings"]:
        m = mouth_point(rig, int(p["bdx"]), int(p["bdy"]), int(p["jaw"]))
        if m:
            roar_rings(img, m, face, p["ring_age"], p["rings"])
    if p["vortex"] >= 0 and box:
        vortex(img, box, p["vortex"], face)
    foot = rig["feet"][rig["strike"]]
    if p["slash"]:
        slash_arcs(img, (foot[0], foot[1] - 2), face, p["slash"])
    if p["shock"]:
        ground_shock(img, (foot[0] + face * 3, foot[1]), p["shock"], rig["ground"])
    if p["erupt"]:
        eruption(img, foot, face, p["erupt"], rig["ground"])
    if p["embers"] and box:
        embers(img, p["embers"], face, box)
    return snap(img, palette)


# ---------------------------------------------------------------- output
def params_hash():
    src = (Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes()
           + (SRC_DIR / "front.png").read_bytes() + (SRC_DIR / "back.png").read_bytes())
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
    ap.add_argument("--masks", action="store_true")
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

    # facing: the helmet head (top rows left of the shell) vs the body centroid
    fa = stills["front"][..., 3] >= 128
    eye = is_col(stills["front"], EYE)
    ey, ex = np.nonzero(eye)
    by, bx = np.nonzero(fa)
    print("facing check: eye x", float(ex.mean()), "body x", float(bx.mean()), flush=True)
    if not ex.mean() < bx.mean():
        raise SystemExit("front.png is expected to face left")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    if args.masks:
        for k in ("front", "back"):
            mask_overlay(stills[k], masks[k], work / f"masks_{k}.png")
        print("masks", work, flush=True)
        return

    rigs = {d: build_rig(stills[VIEW[d][0]], masks[VIEW[d][0]], VIEW[d][0], VIEW[d][1]) for d in DIRS}
    for d in ("W", "N"):
        print(d, "pivots", {k: tuple(round(v, 1) for v in p) for k, p in rigs[d]["pv"].items()},
              "rear", rigs[d]["rear"], flush=True)

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
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in range(12)]
        want += [("walk", "N", f) for f in (0, 2, 4, 6)]
        want += [("attack", "N", f) for f in (2, 4, 6, 7, 9, 11)]
        want += [("attack", "E", f) for f in (3, 6, 7, 9, 10, 12)]
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
        "generator": "forge_heatran_actions", "version": VERSION, "key": KEY,
        "dex": 485, "form": "normal", "variant": "heatran",
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "views": {d: {"still": VIEW[d][0], "mirrored": VIEW[d][1]} for d in DIRS},
        "archetype": "quadruped_beast (heavy lava-dome crouch, 3 drawn legs)",
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
