#!/usr/bin/env python3
"""Glaceon normal (0471) 8-direction action sheet from this form's front.png only.

Measured on normal/front.png (96x96, opaque 1302, bbox x15-82 y22-74, 68x53,
11 colours, alpha already 0 or 255). Faces left: the two eye highlights
(238,255,255) sit at (30,45) and (37,47), the snout tip is x~27, and the
diamond tail is on the right (x63-82 y38-57). Unflipped rows are N NW W SW;
S SE E NE are mirrored so E looks screen-right.

Quadruped (quadruped_beast). Four feet are drawn:
  near fore  x40-45 y63-74  light blue, paw on y73.
  far fore   x46-48 y63-72  dark strip behind the near fore.
  far hind   x53-59 y63-73  dark.
  near hind  x57-66 y63-74  light blue, paw on y74.
Thighs above y63 stay on the torso (cushion), so legs never tear off.
Head extras (not legs): tall ear x28-38 y22-35, flat ear x49-61 y34-42,
left ribbon with dark diamond x15-27 y44-58, front ribbon with dark diamond
x25-40 y53-63, diamond tail.
No mouth line is drawn, so the roar does not invent one: it flares the ears,
the two ribbons and the tail, and draws roar rings and ice shards from the snout.

  idle   : feet planted. Frame 0 is the still. 1px breathing, ears/ribbons/tail sway.
  walk   : diagonal trot (near fore + far hind, far fore + near hind), two steps.
  attack : 0 still, 1-4 ice roar (flare + rings + shards), 5-10 leaping
           360 turn about the vertical axis (front -> back -> back -> mirrored
           front -> front, using this form's own back.png; the sprite is
           never laid on its side) with frost streaks, 11-12 landing
           ice burst, 13-15 recover.
  hurt   : 12 frames, white flash, one-piece recoil about the hind sole (<=34 deg).

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_glaceon_actions.py [--samples] [--force] [--no-sync] [--skip-qa]
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

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
import gen4_attack_strike  # noqa: E402  (shared gen4 attack hook)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0471_glaceon/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0471_glaceon/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
BOUNDS = ROOT / "data/sprite_bounds.js"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 192, 184, 160
VERSION = "glaceon_v2"
KEY = "0471_glaceon"

INK = (16, 16, 16)
NAVY = (49, 57, 74)
DUSK = (57, 57, 82)
DIAMOND = (65, 90, 106)
SLATE = (82, 90, 131)
CAP2 = (82, 98, 156)
BLUE = (82, 156, 222)
MIST = (123, 156, 180)
ICE = (148, 230, 238)
WHITE = (238, 255, 255)

PARTS = ("body", "ear_up", "ear_flat", "ribbon_l", "ribbon_f", "tail",
         "fore_near", "fore_far", "hind_far", "hind_near")
DRAW_ORDER = ("tail", "hind_far", "fore_far", "ribbon_l", "body", "hind_near",
              "fore_near", "ribbon_f", "ear_flat", "ear_up")


# ---------------------------------------------------------------- utilities
def colour_mask(img, colour):
    return (img[..., 3] == 255) & np.all(img[..., :3] == np.array(colour, np.uint8), axis=-1)


def label(mask):
    h, w = mask.shape
    lab = np.zeros(mask.shape, np.int32)
    sizes = []
    cid = 0
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if lab[y, x]:
            continue
        cid += 1
        q = deque([(y, x)])
        lab[y, x] = cid
        n = 0
        while q:
            cy, cx = q.popleft()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = cid
                    q.append((ny, nx))
        sizes.append(n)
    return lab, sizes


def components(mask):
    return sorted(label(mask)[1], reverse=True)


def adjacent(a, b):
    """Pixels of a that touch b (4-neighbour)."""
    p = np.pad(b, 1)
    nb = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return a & nb


def extract_palette(img):
    return np.unique(img[img[..., 3] == 255][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    flat = out[..., :3].reshape(-1, 3).copy()
    sel = a.reshape(-1)
    if sel.any():
        uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
        diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
        dist = (diff * diff).sum(axis=2)
        flat[sel] = palette[dist.argmin(axis=1)][inv.reshape(-1)]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    out[~a, :3] = 0
    return out


def shift(layer, dx, dy):
    dx, dy = int(round(dx)), int(round(dy))
    if dx == 0 and dy == 0:
        return layer.copy()
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = layer[sy0:sy1, sx0:sx1]
    return out


def rotate_nearest(layer, angle_deg, pivot):
    """4x upscale, nearest in-plane rotation, nearest back down. Never |cos| squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.05 or not np.any(layer[..., 3] == 255):
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
    a = out[..., 3] == 255
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def knit(img, limit=8):
    """Join a tiny diagonal speck to the main body with one copied palette pixel."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(limit):
        a = out[..., 3] == 255
        lab, sizes = label(a)
        if len(sizes) <= 1:
            break
        small = 1 + int(np.argmin(sizes))
        if sizes[small - 1] > 4:
            break
        filled = False
        sy, sx = np.nonzero(lab == small)
        for y, x in zip(sy.tolist(), sx.tolist()):
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w) or a[ny, nx]:
                    continue
                for dy2, dx2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    qy, qx = ny + dy2, nx + dx2
                    if 0 <= qy < h and 0 <= qx < w and lab[qy, qx] not in (0, small):
                        out[ny, nx, :3] = out[y, x, :3]
                        out[ny, nx, 3] = 255
                        filled = True
                        break
                if filled:
                    break
            if filled:
                break
        if not filled:
            break
    return out


def paste(dst, layer):
    m = layer[..., 3] == 255
    dst[m] = layer[m]


def put_empty(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """amp * (sin(2pi*cycles*f/n + phase) - sin(phase)); zero on 0 and on the loop end."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


# ---------------------------------------------------------------- segmentation
def segment(img):
    h, w = img.shape[:2]
    op = img[..., 3] == 255
    yy, xx = np.mgrid[0:h, 0:w]
    ink = colour_mask(img, INK)
    dark = colour_mask(img, NAVY) | colour_mask(img, SLATE)

    # Legs below the thigh line y63.
    legs = op & (yy >= 63) & (xx >= 39)
    fore = legs & (xx <= 50)
    hind = legs & (xx >= 51)
    fore_far_c = fore & dark & (xx >= 46)
    fore_near_c = fore & ~ink & ~fore_far_c
    hind_far_c = hind & dark & (xx <= 60)
    hind_near_c = hind & ~ink & ~hind_far_c

    def split(region, a_core, b_core):
        """Outline pixels go to the closest coloured core (ties go to b, the near leg)."""
        a_m = a_core.copy()
        b_m = b_core.copy()
        ys, xs = np.nonzero(region & ~a_core & ~b_core)
        ay, ax = np.nonzero(a_core)
        by, bx = np.nonzero(b_core)
        for y, x in zip(ys.tolist(), xs.tolist()):
            da = np.min((ay - y) ** 2 + (ax - x) ** 2) if len(ay) else 1e9
            db = np.min((by - y) ** 2 + (bx - x) ** 2) if len(by) else 1e9
            if da < db:
                a_m[y, x] = True
            else:
                b_m[y, x] = True
        return a_m, b_m

    fore_far, fore_near = split(fore, fore_far_c, fore_near_c)
    hind_far, hind_near = split(hind, hind_far_c, hind_near_c)

    ear_up = op & (yy <= 35) & (xx <= 40)
    ear_flat = op & (yy <= 42) & (xx >= 50) & (xx <= 62)
    ribbon_l = op & (((yy >= 44) & (yy <= 52) & (xx <= 27)) | ((yy >= 53) & (yy <= 58) & (xx <= 23)))
    ribbon_f = op & (yy >= 53) & (yy <= 63) & (xx >= 24) & (xx <= 40) & ~fore_near & ~fore_far
    tail = op & (xx >= 64) & (yy <= 56)
    tail |= op & (xx >= 62) & (yy <= 52) & ~ear_flat
    used = fore_far | fore_near | hind_far | hind_near | ear_up | ear_flat | ribbon_l | ribbon_f | tail
    body = op & ~used
    return {
        "body": body, "ear_up": ear_up, "ear_flat": ear_flat, "ribbon_l": ribbon_l,
        "ribbon_f": ribbon_f, "tail": tail, "fore_near": fore_near, "fore_far": fore_far,
        "hind_far": hind_far, "hind_near": hind_near,
    }


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), int(mask.sum()))


# ---------------------------------------------------------------- rig
def place_geometry(img):
    ys, xs = np.nonzero(img[..., 3] == 255)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    dst_x = (CELL_W - (x1 - x0)) // 2
    dst_y = FOOT_Y - (y1 - y0)
    return y0, y1, x0, x1, dst_y, dst_x


def placed_still(img):
    y0, y1, x0, x1, dy, dx = place_geometry(img)
    c = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    c[dy:dy + y1 - y0, dx:dx + x1 - x0] = img[y0:y1, x0:x1]
    return c


def contact_pivot(part, body):
    """Mean of the part pixels touching the body: the joint the part swings on."""
    touch = adjacent(part, body)
    ys, xs = np.nonzero(touch if touch.any() else part)
    return (float(xs.mean()), float(ys.mean()))


def build_rig(img, masks, flip):
    y0, y1, x0, x1, dst_y, dst_x = place_geometry(img)
    canvas = placed_still(img)
    placed = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + y1 - y0, dst_x:dst_x + x1 - x0] = m[y0:y1, x0:x1]
        placed[name] = pm
    layers = {}
    for name in PARTS:
        lyr = np.zeros_like(canvas)
        lyr[placed[name]] = canvas[placed[name]]
        layers[name] = lyr
    # Joint cushion: the first rows of each limb/ribbon also live on the body layer,
    # so a lifted leg or swung ribbon never opens a hole at the joint. The parts are
    # drawn over the body, so the zero pose is the still exactly.
    body = placed["body"]
    cushion = np.zeros_like(body)
    yy = np.arange(CELL_H)[:, None]
    for name in ("fore_near", "fore_far", "hind_far", "hind_near"):
        ys = np.nonzero(placed[name])[0]
        cushion |= placed[name] & (yy <= ys.min() + 1)
    for name in ("ear_up", "ear_flat", "ribbon_l", "ribbon_f", "tail"):
        near = placed[name] & dilate(body, 2)
        cushion |= near
    layers["body"][cushion] = canvas[cushion]

    pivots = {}
    for name in ("ear_up", "ear_flat", "ribbon_l", "ribbon_f", "tail"):
        pivots[name] = contact_pivot(placed[name], body)
    ys, xs = np.nonzero(canvas[..., 3] == 255)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    for name in ("fore_near", "fore_far", "hind_far", "hind_near"):
        ly, lx = np.nonzero(placed[name])
        top = ly <= ly.min() + 1
        pivots[name] = (float(lx[top].mean()), float(ly.min()))
    hy, hx = np.nonzero(placed["hind_near"] | placed["hind_far"])
    pivots["hind_sole"] = (float(hx.mean()), float(hy.max()))
    wy, wx = np.nonzero(np.all(canvas[..., :3] == np.array(WHITE, np.uint8), axis=-1) & (canvas[..., 3] == 255))
    eye_x = float(wx.min())
    pivots["snout"] = (eye_x - 4.0, float(wy.mean()) + 3.0)
    pivots["front_x"] = float(xs.min())
    # This form's own back.png, same foot line and centre. It faces away to the
    # right, so the left-facing rig turns through its mirror first.
    bimg = np.array(Image.open(SRC_DIR / "back.png").convert("RGBA"))
    back = placed_still(bimg)
    back_own = back[:, ::-1].copy() if not flip else back
    back_other = back_own[:, ::-1].copy()
    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        canvas = canvas[:, ::-1].copy()
        for k in list(pivots):
            if k == "front_x":
                pivots[k] = CELL_W - 1 - pivots[k]
            else:
                pivots[k] = (CELL_W - 1 - pivots[k][0], pivots[k][1])
    facing = 1.0 if flip else -1.0  # +1 = looks screen-right
    return {"layers": layers, "canvas": canvas, "pivots": pivots, "facing": facing,
            "back_own": back_own, "back_other": back_other}


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


# ---------------------------------------------------------------- poses
def pose(action, f):
    n = FRAMES[action]
    p = dict(bdx=0, bdy=0, legs={k: (0.0, 0) for k in ("fore_near", "fore_far", "hind_far", "hind_near")},
             ear_up=0.0, ear_flat=0.0, ribbon_l=0.0, ribbon_f=0.0, tail=0.0,
             turn=-1, lift=0, hurt=0.0, knock=0, flash=False,
             rings=0, ring_age=0, shards=0, frost=0, burst=0, sparks=0)
    if action == "idle":
        br = wave(f, n, 1.0, phase=0.0)
        p["bdy"] = -1 if br > 0.5 else 0
        p["ear_up"] = wave(f, n, 4.0, phase=0.5)
        p["ear_flat"] = wave(f, n, 5.0, phase=1.1)
        p["ribbon_l"] = wave(f, n, 7.0, phase=0.3)
        p["ribbon_f"] = wave(f, n, 6.0, phase=0.9)
        p["tail"] = wave(f, n, 6.0, phase=1.6)
    elif action == "walk":
        th = 2 * math.pi * 2 * f / n
        s = math.sin(th)
        # Lifted pair swings forward about the hip; the planted pair sweeps back.
        la = int(round(2 * max(0.0, s)))
        lb = int(round(2 * max(0.0, -s)))
        aa = 16.0 * s
        p["legs"] = {
            "fore_near": (aa, la), "hind_far": (aa, la),
            "fore_far": (-aa, lb), "hind_near": (-aa, lb),
        }
        p["bdy"] = -1 if abs(s) > 0.7 else 0
        p["ear_up"] = wave(f, n, 3.0, cycles=2)
        p["ear_flat"] = wave(f, n, 4.0, cycles=2, phase=0.8)
        p["ribbon_l"] = wave(f, n, 9.0, cycles=2, phase=0.4)
        p["ribbon_f"] = wave(f, n, 8.0, cycles=2, phase=1.0)
        p["tail"] = wave(f, n, 8.0, cycles=2, phase=1.5)
    elif action == "attack":
        # 0 still | 1-4 ice roar | 5-10 leaping 360 | 11-12 landing burst | 13-15 recover
        flare = [0, 0.45, 1.0, 1.0, 0.6, 0, 0, 0, 0, 0, 0, 0.3, 0.2, 0.1, 0.0, 0][f]
        # Signs: positive swings the tip backward over the top / forward under the chin.
        p["ear_up"] = 18.0 * flare       # tall ear pinned back
        p["ear_flat"] = 14.0 * flare     # flat ear pinned back and down
        p["ribbon_l"] = 38.0 * flare     # head ribbons fling up and forward
        p["ribbon_f"] = 30.0 * flare
        p["tail"] = -24.0 * flare        # diamond tail raised
        # bdx is a whole-body integer lunge toward the facing side (negative = rear back).
        p["bdx"] = [0, -1, -2, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["bdy"] = [0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0][f]
        p["rings"] = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["ring_age"] = f
        p["shards"] = [0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        if 5 <= f <= 10:
            # view index: 0 own, 1 back (own side), 2 back (other side), 3 mirrored front
            p["turn"] = [0, 1, 2, 3, 0, 0][f - 5]
            p["lift"] = int(round(16 * math.sin(math.pi * (f - 4) / 7)))
            p["frost"] = f - 4
        if f in (11, 12):
            p["burst"] = f - 10
    else:
        env = math.sin(math.pi * min(f, 10) / 10.0) if f <= 10 else 0.0
        p["hurt"] = 34.0 * env
        p["knock"] = int(round(3 * env))
        p["flash"] = f == 1
        p["sparks"] = 1 if 2 <= f <= 5 else 0
    return p


def assemble_pose(rig, p):
    L, P, fc = rig["layers"], rig["pivots"], rig["facing"]
    bdx = fc * p["bdx"]
    bdy = p["bdy"]
    out = {}
    out["body"] = shift(L["body"], bdx, bdy)
    # Swing angles: positive = tip moves backward/up relative to facing.
    for name in ("ear_up", "ear_flat", "ribbon_l", "ribbon_f", "tail"):
        ang = p[name]
        out[name] = shift(rotate_nearest(L[name], -fc * ang, P[name]), bdx, bdy)
    for name, (ang, lift) in p["legs"].items():
        lyr = L[name]
        if abs(ang) > 0.05:
            lyr = despeckle(rotate_nearest(lyr, -fc * ang, P[name]))
        out[name] = shift(lyr, bdx, -lift)
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in DRAW_ORDER:
        paste(img, out[name])
    return img


def neutral_bitmap(rig):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in DRAW_ORDER:
        paste(img, rig["layers"][name])
    return img


# ---------------------------------------------------------------- effects
def arc(out, cx, cy, facing, rad, col, span=40, squash=0.75):
    steps = max(4, int(rad * math.radians(2 * span)))
    for i in range(steps + 1):
        deg = -span + 2 * span * i / steps
        a = math.radians(deg)
        put_empty(out, cx + facing * rad * math.cos(a), cy + rad * math.sin(a) * squash, col)


def roar_rings(out, snout, facing, age, strength):
    sx, sy = snout
    cols = (WHITE, ICE, BLUE)
    for i in range(strength):
        rad = 5 + age * 2 + i * 5
        arc(out, sx, sy, facing, rad, cols[i % 3])
        arc(out, sx, sy, facing, rad + 1, INK if i == strength - 1 else cols[i % 3], span=30)


def shard(out, x, y, size=2):
    """A tiny ice diamond: ink rim, ice fill, white glint."""
    x, y = int(round(x)), int(round(y))
    for dy in range(-size - 1, size + 2):
        for dx in range(-size - 1, size + 2):
            d = abs(dx) + abs(dy)
            if d == size + 1:
                put_empty(out, x + dx, y + dy, INK)
    for dy in range(-size, size + 1):
        for dx in range(-size, size + 1):
            d = abs(dx) + abs(dy)
            if d <= size:
                put_empty(out, x + dx, y + dy, WHITE if (dx, dy) == (-1, -1) or d == 0 and size == 1 else ICE)


def ice_shards(out, snout, facing, count):
    """Icy wind: diamonds blown out of the roar, further each frame."""
    sx, sy = snout
    spots = [(12, -7, 2), (18, 5, 3), (27, -2, 2), (33, -12, 2), (36, 9, 2), (44, -4, 3)]
    n = {1: 2, 2: 4, 3: 6}[count]
    for i in range(n):
        dx, dy, sz = spots[i]
        shard(out, sx + facing * (dx + count * 3), sy + dy, sz)


def frost_trail(out, center, k, facing):
    """Two curved frost streaks trailing the spin, plus a few sparkles."""
    cx, cy = center
    lead = -facing * k * 60.0
    for base in (0.0, 180.0):
        for r, col, span in ((40, WHITE, 50), (42, ICE, 70), (44, BLUE, 40)):
            steps = int(r * math.radians(span))
            for i in range(steps):
                deg = base + lead + facing * (20 + span * i / steps)
                a = math.radians(deg)
                put_empty(out, cx + r * math.cos(a), cy + r * math.sin(a), col)
    for j in range(4):
        a = math.radians(k * 37 + j * 90 + 45)
        shard(out, cx + 48 * math.cos(a), cy + 46 * math.sin(a), 1)


def landing_burst(out, facing, stage):
    ys, xs = np.nonzero(out[..., 3] == 255)
    if len(xs) == 0:
        return out
    front = float(xs.min() if facing < 0 else xs.max())
    gy = float(ys.max())
    cx = front + facing * 3
    # Ice crystals burst up out of the ground right in front: tapered spikes.
    spikes = ((0, 16, 3), (7, 22, 4), (14, 13, 3), (20, 9, 2), (-3, 9, 2))
    for dx, hgt, half in spikes:
        x0 = cx + facing * dx
        hh = hgt if stage == 1 else int(hgt * 0.7)
        for t in range(hh + 1):
            wd = max(0, int(round(half * (1 - t / max(1, hh)))))
            y = gy - t
            for u in range(-wd, wd + 1):
                col = WHITE if u == -wd + 1 and t > 1 else (ICE if u <= 0 else BLUE)
                put_empty(out, x0 + u, y, col)
            put_empty(out, x0 - wd - 1, y, INK)
            put_empty(out, x0 + wd + 1, y, INK)
        put_empty(out, x0, gy - hh - 1, INK)
    # Frost ring skating out across the floor.
    rad = 20 if stage == 1 else 30
    for k in range(40):
        a = 2 * math.pi * k / 40
        put_empty(out, cx + facing * 8 + rad * math.cos(a), gy + 2 + rad * 0.16 * math.sin(a),
                  (WHITE, ICE, BLUE)[k % 3])
    return out


def hurt_sparks(out, facing):
    ys, xs = np.nonzero(out[..., 3] == 255)
    if len(xs) == 0:
        return out
    y = int(ys.min()) + 6
    x = int(xs.min() + 6 if facing < 0 else xs.max() - 6)
    for dx, dy, col in ((0, -4, WHITE), (-3, -2, ICE), (3, -2, ICE), (-5, 2, WHITE), (5, 1, WHITE)):
        put_empty(out, x + dx, y + dy, col)
    return out


def flash(img):
    out = img.copy()
    a = out[..., 3] == 255
    ink = np.all(out[..., :3] == np.array(INK, np.uint8), axis=-1)
    out[a & ~ink, :3] = WHITE
    return out


# ---------------------------------------------------------------- synth
def synth(rig, action, f, palette, effects=True):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    fc = rig["facing"]
    P = rig["pivots"]
    if action == "attack" and 5 <= f <= 10:
        view = (neutral_bitmap(rig), rig["back_own"], rig["back_other"],
                neutral_bitmap(rig)[:, ::-1].copy())[p["turn"]]
        img = shift(view, fc * p["bdx"], -p["lift"])
    elif action == "hurt" and p["hurt"] > 0.45:
        img = rotate_nearest(neutral_bitmap(rig), -fc * p["hurt"], P["hind_sole"])
        img = shift(img, -fc * p["knock"], 0)
        img = knit(despeckle(img))
    else:
        img = assemble_pose(rig, p)
        if action == "hurt" and p["knock"]:
            img = shift(img, -fc * p["knock"], 0)
    img = snap(img, palette)
    if p["flash"]:
        img = flash(img)
    body_only = img.copy()
    if effects:
        snout = (P["snout"][0] + fc * p["bdx"], P["snout"][1] + p["bdy"])
        if p["rings"]:
            roar_rings(img, snout, fc, p["ring_age"], p["rings"])
        if p["shards"] and f <= 4:
            ice_shards(img, snout, fc, p["shards"])
        if p["frost"] and action == "attack" and 5 <= f <= 10:
            frost_trail(img, (P["center"][0], P["center"][1] - p["lift"]), p["frost"], fc)
        if p["burst"]:
            landing_burst(img, fc, p["burst"])
        if p["sparks"]:
            hurt_sparks(img, fc)
    return snap(img, palette), body_only


# ---------------------------------------------------------------- outputs
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
    """Every action, every direction, every frame at 1x on a dark backdrop."""
    x0, x1, y0, y1 = 24, CELL_W - 24, 40, CELL_H - 4
    cw, ch = x1 - x0, y1 - y0
    rows = len(ACTIONS) * 8
    cols = max(FRAMES.values())
    bg = np.full((rows * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    r = 0
    for act in ACTIONS:
        sh = sheets[act]
        for di in range(8):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H + y0:di * CELL_H + y1, f * CELL_W + x0:f * CELL_W + x1]
                m = tile[..., 3] == 255
                bg[r * ch:(r + 1) * ch, f * cw:(f + 1) * cw][m] = tile[m]
            r += 1
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    """South and East rows of each action at 2x, cropped to motion."""
    pad = 6
    rows = []
    for act in ACTIONS:
        for di in (0, 2):
            row = []
            band = sheets[act][di * CELL_H:(di + 1) * CELL_H]
            for f in range(FRAMES[act]):
                tile = band[:, f * CELL_W:(f + 1) * CELL_W]
                row.append(tile)
            rows.append(row)
    allm = np.zeros((CELL_H, CELL_W), bool)
    for row in rows:
        for t in row:
            allm |= t[..., 3] == 255
    ys, xs = np.nonzero(allm)
    yy0, yy1 = max(0, ys.min() - pad), min(CELL_H, ys.max() + 1 + pad)
    xx0, xx1 = max(0, xs.min() - pad), min(CELL_W, xs.max() + 1 + pad)
    ch, cw = (yy1 - yy0) * 2, (xx1 - xx0) * 2
    cols = max(FRAMES.values())
    bg = np.full((len(rows) * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    for ri, row in enumerate(rows):
        for ci, t in enumerate(row):
            c = np.repeat(np.repeat(t[yy0:yy1, xx0:xx1], 2, 0), 2, 1)
            m = c[..., 3] == 255
            bg[ri * ch:(ri + 1) * ch, ci * cw:(ci + 1) * cw][m] = c[m]
            bg[ri * ch:(ri + 1) * ch, ci * cw] = (40, 48, 64, 255)
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
        if rel.name.startswith(("front", "back")):
            raise SystemExit(f"refusing to copy {rel}")
        dest = SYNC_OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fp, dest)
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


def _object_span(text, key):
    token = f'"{key}":'
    start = text.find(token)
    if start < 0 or text.find(token, start + 1) >= 0:
        raise SystemExit(f"{key} not found exactly once")
    val = start + len(token)
    if text[val] != "{":
        raise SystemExit(f"{key} value is not an object")
    depth = 0
    for j in range(val, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return start, val, j + 1
    raise SystemExit(f"{key} object did not close")


def publish_dir8(sprite_h):
    entry = {
        "dex": 471, "form": "normal", "variant": "glaceon", "gen": "gen4",
        "dir": "monsterSprite/gen4/0471_glaceon/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed")
        _s, val, end = _object_span(text, KEY)
        new_text = text[:val] + blob + text[end:]
        if new_text[:val] != text[:val] or new_text[val + len(blob):] != text[end:]:
            raise SystemExit("dir8 bytes outside the entry changed")
        parsed = json.loads(new_text[len(prefix):].rstrip().rstrip(";"))
        if parsed[KEY] != entry:
            raise SystemExit("dir8 entry did not parse back")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(new_text)
        tmp.replace(DIR8)

    with_flock(DIR8_LOCK, edit)


def publish_bounds(bounds):
    """Replace only s["0471_glaceon"] in sprite_bounds.js with the sheet's idle-0 boxes."""
    text = BOUNDS.read_text()
    token = f'"{KEY}":['
    i = text.find('"s":{')
    if i < 0:
        raise SystemExit("sprite_bounds has no s map")
    j = text.find(token, i)
    if j < 0:
        raise SystemExit("sprite_bounds has no s entry for glaceon")
    k = text.index("]", j)
    blob = token + ",".join(str(int(v)) for v in bounds) + "]"
    new = text[:j] + blob + text[k + 1:]
    tmp = BOUNDS.with_suffix(".js.tmp")
    tmp.write_text(new)
    tmp.replace(BOUNDS)


def update_queue(status, note=None):
    def edit():
        raw = QUEUE.read_text()
        marker = f'"id": "{KEY}"'
        if raw.count(marker) != 1:
            raise SystemExit("queue id count")
        idx = raw.index(marker)
        start = raw.rfind("{", 0, idx)
        depth = 0
        end = None
        for j in range(start, len(raw)):
            if raw[j] == "{":
                depth += 1
            elif raw[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        block = raw[start:end]
        block2, n = re.subn(r'"status": "[^"]*"', f'"status": "{status}"', block, count=1)
        if n != 1:
            raise SystemExit("queue status not found")
        if note:
            block2 = block2.replace('"notes": "', '"notes": "' + note.replace('"', "'") + " | ", 1)
        new = raw[:start] + block2 + raw[end:]
        old = json.loads(raw)
        data = json.loads(new)
        for a, b in zip(old["items"], data["items"]):
            if a.get("id") != KEY and a != b:
                raise SystemExit("queue neighbour changed")
        tmp = QUEUE.with_suffix(".json.tmp")
        tmp.write_text(new)
        tmp.replace(QUEUE)

    with_flock(QUEUE_LOCK, edit)


def differ_count(a, b):
    d = np.any(a != b, axis=2) & ((a[..., 3] == 255) | (b[..., 3] == 255))
    return int(d.sum())


def params_hash():
    return hashlib.sha1((SRC_DIR / "back.png").read_bytes() + Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (SRC_DIR / "front.png").read_bytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true",
                    help="after a visually checked pass: write the dir8 key, sprite_bounds s[], queue done")
    args = ap.parse_args()

    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if front.shape[:2] != (96, 96):
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    if sorted(int(v) for v in np.unique(front[..., 3])) != [0, 255]:
        raise SystemExit("still alpha is not binary")
    palette = extract_palette(front)
    masks = segment(front)
    op = front[..., 3] == 255
    cover = np.zeros_like(op)
    overlap = 0
    for name in PARTS:
        overlap += int((cover & masks[name]).sum())
        cover |= masks[name]
    counts = {k: int(masks[k].sum()) for k in PARTS}
    counts["missing"] = int((op & ~cover).sum())
    counts["overlap"] = overlap
    print("parts", counts, "palette", len(palette), flush=True)
    for name in PARTS:
        print(" ", name, bbox_of(masks[name]), "cc", components(masks[name])[:3], flush=True)
    if counts["missing"] or overlap:
        raise SystemExit("segmentation does not partition the still")

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    still = placed_still(front)
    print("pivots W", {k: (tuple(round(x, 1) for x in v) if isinstance(v, tuple) else round(v, 1))
                       for k, v in rigs["W"]["pivots"].items()}, flush=True)
    for act in ACTIONS:
        for d, ref in (("W", still), ("E", still[:, ::-1].copy())):
            d0 = differ_count(synth(rigs[d], act, 0, palette)[0], ref)
            if d0:
                raise SystemExit(f"{act} {d} frame 0 differs from the still by {d0}")
    for act in ("idle", "walk"):
        p_end = pose(act, FRAMES[act] - 1)
        _ = p_end
    print("still diff 0 (all actions, W and E)", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((96, 96, 4), np.uint8)
    cols = {"body": (150, 150, 170), "ear_up": (255, 140, 40), "ear_flat": (255, 220, 80),
            "ribbon_l": (220, 40, 180), "ribbon_f": (160, 60, 255), "tail": (40, 160, 90),
            "fore_near": (220, 60, 60), "fore_far": (120, 30, 30),
            "hind_far": (40, 80, 200), "hind_near": (80, 200, 240)}
    for name, col in cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    base = int((still[..., 3] == 255).sum())
    body_report = {}
    for act in ("attack", "hurt"):
        for f in range(FRAMES[act]):
            _img, body = synth(rigs["W"], act, f, palette)
            sizes = components(body[..., 3] == 255)
            body_report[f"{act}{f:02d}"] = (sizes[:3], round(sizes[0] / base, 3))
    print("body components", body_report, flush=True)

    if args.samples:
        want = [(a, d) for a in ACTIONS for d in ("W", "S", "N")]
        tw, th = CELL_W, CELL_H
        sheet = np.full((len(want) * th, 16 * tw, 4), (15, 20, 31, 255), np.uint8)
        for r, (act, d) in enumerate(want):
            for f in range(FRAMES[act]):
                tile = synth(rigs[d], act, f, palette)[0]
                m = tile[..., 3] == 255
                sheet[r * th:(r + 1) * th, f * tw:(f + 1) * tw][m] = tile[m]
                sheet[r * th:(r + 1) * th, f * tw] = (40, 48, 64, 255)
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        # zoomed attack/hurt W for close inspection
        zoom = []
        for act in ("idle", "walk", "attack", "hurt"):
            row = [synth(rigs["W"], act, f, palette)[0][30:CELL_H, 20:CELL_W - 20] for f in range(FRAMES[act])]
            while len(row) < 16:
                row.append(np.zeros_like(row[0]))
            zoom.append(np.concatenate(row, axis=1))
        z = np.concatenate(zoom, axis=0)
        bgz = np.full(z.shape, (15, 20, 31, 255), np.uint8)
        m = z[..., 3] == 255
        bgz[m] = z[m]
        Image.fromarray(bgz, "RGBA").save(work / "samples_W.png")
        print("samples", work / "samples.png", flush=True)
        return

    frames_dir = work / "frames"
    state_p = work / "state.json"
    if args.force and frames_dir.exists():
        shutil.rmtree(frames_dir)
    digest = params_hash()
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    if state.get("hash") != digest:
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
                if key in state["done"] and fp.exists() and fp.stat().st_size > 0:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palette)[0], "RGBA").save(tmp)
                tmp.replace(fp)
            if key not in state["done"]:
                state["done"].append(key)
                state_p.write_text(json.dumps(state))
            print(f"[{state['done'].index(key) + 1}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    prev = preview_image(sheets)
    prev.save(OUT_DIR / "preview.png")
    (SRC_DIR / "actions").mkdir(exist_ok=True)
    prev.save(SRC_DIR / "actions" / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    bounds = []
    for di in range(8):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        ys, xs = np.nonzero(t[..., 3] == 255)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    sprite_h = bounds[3]

    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] == 255][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = [int(v) for v in np.unique(master[..., 3])]
    clip = []
    still_diff = 0
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            ref = still if d in UNFLIPPED else still[:, ::-1].copy()
            t0 = sh[di * CELL_H:(di + 1) * CELL_H, :CELL_W]
            still_diff += differ_count(t0, ref)
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] == 255
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
    print("sprite_h", sprite_h, "off", len(off), "alphas", alphas, "clip", clip, "still_diff", still_diff, flush=True)
    report = {"version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
              "off_palette_colors": len(off), "alphas": alphas, "edge_touching": len(clip),
              "still_diff": still_diff, "bounds": bounds, "body_components": body_report}
    if not args.skip_qa and GAIT_LOOP.exists():
        qa = work / "qa"
        qa.mkdir(exist_ok=True)
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run(
                [str(FORGE / ".venv/bin/python"), "-I", str(GAIT_LOOP), "select", "--frames-dir",
                 str(frames_dir / act / "S"), "--fps", "12", "--output-dir", str(od),
                 "--kind", "hover", "--no-aids"], capture_output=True, text=True)
            rep = od / "loop-report.json"
            report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation") if rep.exists() else (r.stderr or r.stdout)[-300:]
            print("gait", act, report[f"gait_{act}"], flush=True)
    (work / "qa").mkdir(exist_ok=True)
    (work / "qa" / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    meta = {"generator": "forge_glaceon_actions", "version": VERSION, "key": KEY, "dex": 471,
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
            "frames": FRAMES, "directions": DIRS, "actions": ACTIONS, "unflipped": sorted(UNFLIPPED),
            "palette_colors": int(len(palette)), "off_palette_colors": len(off),
            "source": "front.png 96x96", "gifs_untouched": True, "facing": "left",
            "archetype": "quadruped_beast", "feet": 4, "mouth": "none_ear_ribbon_flare"}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if off or alphas not in ([0, 255], [0]) or clip or still_diff:
        print("QA FAIL", flush=True)
        raise SystemExit(1)
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    print("QA OK; bounds", bounds, flush=True)
    if args.publish:
        publish_dir8(sprite_h)
        publish_bounds(bounds)
        update_queue("done")
        print("published", KEY, VERSION, "h", sprite_h, flush=True)


if __name__ == "__main__":
    main()
