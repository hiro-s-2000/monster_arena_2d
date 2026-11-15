#!/usr/bin/env python3
"""Magmortar (0467_magmortar) normal 8-direction action sheet from the 96x96 front still.

Measured on normal/front.png: 96x96, 2934 opaque, bbox x17-83 y9-82, 14 colours,
alpha already 0 or 255.

Heavy biped (kaiju stomper with cannon arms). Parts that the still really has:
  head + yellow crest  x33-53 y9-40. Mouth: black upper-lip line y33 x36-41 with
                       grey teeth (x35, x42-44), dark interior y34 x37-41, pink
                       lower lip y35-36 and chin y37-39. Eye glint x42-45 y31-33.
  flame_l              red flame + orange shoulder pad left of the head x21-34 y12-39
  flame_r              red flame + orange shoulder pad right of the head x50-76 y14-44
  arm_l                yellow cannon, hangs down-left, claws x17-24 y62-65
  arm_r                yellow cannon with black cuff, hangs down-right, claws y68-73
  tail                 dark red flame tail behind the right arm x72-83 y33-58
  foot_l / foot_r      dark shin + red foot + white claws (y69-78 / y71-82)
  torso                collar, yellow belly with red pattern, pink thighs

The lips (x35) are left of the eye glint (mean x 43.5) and the tail shows on the
right, so the still faces left. Unflipped dirs are N NW W SW; S SE E NE mirror it.

  idle   : breathing bob of the upper body, flames flicker, tail sways, cannons drift.
  walk   : heavy alternating stomp of the two real feet, cannons swing, flames trail.
  attack : 0-4 roar (lower jaw drops, cannons flex up, flames flare, roar rings),
           5-10 one-piece 360 jump-spin inside a fire-spin ring,
           11-12 near cannon thrust forward + fire blast, 13-15 recover to the still.
  hurt   : white flash, one-piece lean back on the rear foot (max 28 deg), embers, recover.

Palette-locked to this still. Does not write front/back png or gif, shiny, reverse,
box-icons or legends-arceus.
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
STILL_DIR = ROOT / "monsterSprite/gen4/0467_magmortar/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0467_magmortar/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"

KEY = "0467_magmortar"
VERSION = "magmortar_v2"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# The still faces left, so the left-facing compass keeps the drawing.
FACE_LEFT = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 192, 184, 162

# Measured colours (front.png).
INK = (16, 16, 16)
DGRY = (74, 74, 65)
BRN = (106, 49, 41)
DGLD = (131, 98, 32)
CRIM = (172, 57, 82)
ROSE = (197, 90, 106)
GOLD = (205, 164, 57)
CLAW = (205, 213, 230)
RED = (213, 49, 49)
RED2 = (230, 65, 49)
ORG = (246, 98, 49)
PINK = (255, 148, 164)
YEL = (255, 197, 57)
WHITE = (255, 255, 255)
CODE = "abcdefghijklmn"
COLS = (INK, DGRY, BRN, DGLD, CRIM, ROSE, GOLD, CLAW, RED, RED2, ORG, PINK, YEL, WHITE)

OPAQUE = 2934
BBOX = (17, 83, 9, 82)

# Interior floods: (seed (x, y), allowed colour codes, (x0, x1, y0, y1) box).
FLOODS = {
    "tail": ([(80, 45), (81, 37)], "eic", (72, 84, 32, 60)),
    "foot_l": ([(30, 74)], "bjceh", (18, 36, 69, 80)),
    "foot_r": ([(62, 77), (60, 72)], "bjcehn", (55, 72, 71, 84)),
    "arm_l": ([(21, 52)], "mgdbhn", (15, 33, 40, 66)),
    "arm_r": ([(70, 60), (73, 71)], "mgdbhn", (59, 82, 42, 74)),
    "flame_l": ([(28, 24)], "ik", (19, 35, 11, 40)),
    "flame_r": ([(60, 30), (71, 17)], "ik", (49, 77, 13, 45)),
}
EXPECT = {
    "tail": 0, "foot_l": 0, "foot_r": 0, "arm_l": 0, "arm_r": 0,
    "flame_l": 0, "flame_r": 0, "head": 0, "torso": 0,
}
# Joint pivots (still coords). Arms turn at the shoulder, flames at the pad base.
PIVOTS = {
    "arm_l": (30.0, 42.0),
    "arm_r": (63.0, 46.0),
    "flame_l": (32.0, 35.0),
    "flame_r": (53.0, 38.0),
    "tail": (74.0, 53.0),
}
MOUTH_TIP = (35, 35)  # left end of the lips; the roar leaves from here
DRAW = ("arm_r", "tail", "leg_ext", "foot_r", "foot_l", "torso", "flame_r", "flame_l", "head", "arm_l")
ROTATING = ("arm_l", "arm_r", "flame_l", "flame_r", "tail")


# ----------------------------------------------------------------- utilities

def code_map(img):
    a = img[..., 3] >= 128
    code = np.full(a.shape, ".", dtype="<U1")
    for ch, col in zip(CODE, COLS):
        code[a & np.all(img[..., :3] == np.array(col, np.uint8), axis=2)] = ch
    return code


def flood(seeds, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow, dtype=bool)
    q = deque()
    for x, y in seeds:
        if not allow[y, x]:
            raise SystemExit(f"flood seed not allowed: {(x, y)}")
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


def dilate(mask, n=1, diag=False):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        nxt = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        if diag:
            nxt |= p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
        out = nxt
    return out


def box_mask(shape, box):
    x0, x1, y0, y1 = box
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)


def component_sizes(img_or_mask):
    a = img_or_mask if img_or_mask.ndim == 2 else img_or_mask[..., 3] >= 128
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


def torso_crumbs(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    comps = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        comp = np.zeros_like(mask)
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        while q:
            y, x = q.popleft()
            comp[y, x] = True
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        comps.append(comp)
    comps.sort(key=lambda c: int(c.sum()), reverse=True)
    return comps[1:]


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ---------------------------------------------------------------- segmentation

def segment(img):
    a = img[..., 3] >= 128
    if img.shape[:2] != (96, 96):
        raise SystemExit("front is not 96x96")
    if int(a.sum()) != OPAQUE:
        raise SystemExit(f"still opaque changed: {int(a.sum())}")
    ys, xs = np.nonzero(a)
    if (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())) != BBOX:
        raise SystemExit("still bbox changed; remeasure before forging")
    code = code_map(img)
    if np.any(a & (code == ".")):
        raise SystemExit("still has a colour that was not measured")
    owner = np.full(a.shape, "", dtype="<U8")
    shape = a.shape
    # 1. interiors
    for name, (seeds, cols, box) in FLOODS.items():
        allow = np.isin(code, list(cols)) & box_mask(shape, box) & (owner == "")
        owner[flood(seeds, allow)] = name
    # 2. each part takes the outline ring that touches its interior
    outline = np.isin(code, ["a", "c", "d"])
    for name, (_s, _c, box) in FLOODS.items():
        ring = dilate(owner == name, 1, diag=True) & outline & (owner == "") & box_mask(shape, box)
        owner[ring] = name
    # 3. head: what is left above the collar between the two shoulder flames
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    collar = (code == "b") & (yy >= 37)
    head = a & (owner == "") & (yy <= 38) & (xx >= 33) & (xx <= 54) & ~collar
    head |= a & (owner == "") & (yy >= 39) & (yy <= 40) & (xx >= 37) & (xx <= 50) & ~collar
    owner[head] = "head"
    owner[a & (owner == "")] = "torso"
    # Torso crumbs cut off from the main body (flame wisps, claw outlines) join
    # the part they touch most, so no grey island is left behind when it moves.
    for _ in range(3):
        crumbs = torso_crumbs(owner == "torso")
        if not crumbs:
            break
        for comp in crumbs:
            ring = dilate(comp, 1) & ~comp & a
            names_near, votes = np.unique(owner[ring], return_counts=True)
            best = [(v, n) for n, v in zip(names_near.tolist(), votes.tolist()) if n not in ("", "torso")]
            if best:
                owner[comp] = max(best)[1]
    names = list(FLOODS) + ["head", "torso"]
    masks = {n: owner == n for n in names}
    # Stray single pixels of a part that are not joined to it go to the torso.
    for n in list(FLOODS):
        m = masks[n]
        if not np.any(m):
            raise SystemExit(f"part {n} is empty")
    counts = {n: int(m.sum()) for n, m in masks.items()}
    if sum(counts.values()) != OPAQUE:
        raise SystemExit("segmentation dropped pixels")
    # Mouth / jaw / eye checks.
    jaw = masks["head"] & (yy >= 35) & (xx >= 34) & (xx <= 49)
    upper_lip = (code == "a") & (yy == 33) & (xx >= 36) & (xx <= 41)
    interior = (code == "b") & (yy == 34) & (xx >= 37) & (xx <= 41)
    eye = np.isin(code, ["h", "n"]) & (yy >= 31) & (yy <= 33) & (xx >= 42) & (xx <= 45)
    if int(upper_lip.sum()) != 6 or int(interior.sum()) != 5:
        raise SystemExit(f"mouth changed: lip {int(upper_lip.sum())} interior {int(interior.sum())}")
    if int(eye.sum()) < 4:
        raise SystemExit("eye glint changed")
    eye_x = float(np.nonzero(eye)[1].mean())
    if not MOUTH_TIP[0] < eye_x:
        raise SystemExit("lips are not left of the eye; facing changed")
    for n, (px, py) in PIVOTS.items():
        if not dilate(masks[n], 2, diag=True)[int(py), int(px)]:
            raise SystemExit(f"pivot {n} is off its part")
    return masks, counts, jaw, eye_x


# ---------------------------------------------------------------- transforms

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
    """4x nearest in-plane rotate (+ = clockwise on screen). 360 lands on 0."""
    ang = angle_deg % 360.0
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
    inv = out[1::4, 1::4].copy()
    # Forward-map pass so no thin outline drops out of the downsample.
    a = layer[..., 3] >= 128
    ys, xs = np.nonzero(a)
    dxs = xs.astype(np.float64) - pivot[0]
    dys = ys.astype(np.float64) - pivot[1]
    tx = np.rint(pivot[0] + c * dxs - s * dys).astype(np.int32)
    ty = np.rint(pivot[1] + s * dxs + c * dys).astype(np.int32)
    H, W = layer.shape[:2]
    ok = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)
    fwd = np.zeros_like(layer)
    fwd[ty[ok], tx[ok]] = layer[ys[ok], xs[ok]]
    miss = (fwd[..., 3] >= 128) & (inv[..., 3] < 128)
    inv[miss] = fwd[miss]
    return inv


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def close_cracks(img, passes=2):
    out = img.copy()
    for _ in range(passes):
        a = out[..., 3] >= 128
        p = np.pad(a, 1, constant_values=False)
        up, dn = p[:-2, 1:-1], p[2:, 1:-1]
        lf, rt = p[1:-1, :-2], p[1:-1, 2:]
        hole = (~a) & (((up & dn) | (lf & rt)))
        if not np.any(hole):
            break
        col = np.zeros_like(out)
        for src, take in ((up, (0, 1)), (dn, (0, -1)), (lf, (1, 0)), (rt, (-1, 0))):
            use = hole & src & (col[..., 3] == 0)
            if np.any(use):
                col[use] = shift(out, take[0], take[1])[use]
        out[hole] = col[hole]
    return out


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


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


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


# ------------------------------------------------------------------- the rig

def placement(img):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    dst_x = (CELL_W - (x1 - x0)) // 2
    dst_y = FOOT_Y - (y1 - y0)
    return x0, y0, x1, y1, dst_x - x0, dst_y - y0


def placed_still(front):
    x0, y0, x1, y1, ox, oy = placement(front)
    out = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    out[y0 + oy:y1 + oy, x0 + ox:x1 + ox] = front[y0:y1, x0:x1]
    out[out[..., 3] < 128] = 0
    return out


def build_rig(img, masks, jaw, flip):
    x0, y0, x1, y1, ox, oy = placement(img)
    canvas = placed_still(img)

    def place(m):
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[y0 + oy:y1 + oy, x0 + ox:x1 + ox] = m[y0:y1, x0:x1]
        return pm

    placed = {n: place(m) for n, m in masks.items()}
    jaw_p = place(jaw)
    layers = {}
    for n, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[n] = lyr
    torso, head = placed["torso"], placed["head"]

    # Joint cushions: where a part sits on its parent, fill with the parent colour
    # underneath, so a part that turns away never leaves a hole in the body.
    cushion = np.zeros_like(canvas)
    parents = {
        "arm_l": ("torso",), "arm_r": ("torso",), "tail": ("torso",),
        "flame_l": ("head", "torso"), "flame_r": ("head", "torso"),
    }
    for n, par in parents.items():
        pm = np.zeros_like(torso)
        for p in par:
            pm |= placed[p]
        zone = placed[n] & dilate(pm, 3)
        fill = nearest_color(canvas, zone, pm)
        cushion[zone] = fill[zone]

    # Leg extension: copy each shin's top row up under the torso, so a lifted
    # torso or a stepping foot never opens a gap at the knee.
    ext = np.zeros_like(canvas)
    for n in ("foot_l", "foot_r"):
        ys, xs = np.nonzero(placed[n])
        ytop = int(ys.min())
        row = [(int(x), ytop) for x in xs[ys == ytop]]
        for k in range(1, 5):
            for x, y in row:
                yy = y - k
                if torso[yy, x] and ext[yy, x, 3] == 0:
                    ext[yy, x] = canvas[y, x]
    layers["leg_ext"] = ext

    pivots = {n: (px + ox, py + oy) for n, (px, py) in PIVOTS.items()}
    mouth = (MOUTH_TIP[0] + ox, MOUTH_TIP[1] + oy)
    tys, txs = np.nonzero(torso | head)
    center = (float(txs.mean()), float(tys.mean()))
    soles = []
    for n in ("foot_l", "foot_r"):
        ys, xs = np.nonzero(placed[n])
        soles.append((float(xs.mean()), float(ys.max())))
    sole_mid = ((soles[0][0] + soles[1][0]) / 2.0, max(soles[0][1], soles[1][1]))

    facing = -1.0
    if flip:
        def mx(x):
            return CELL_W - 1 - x

        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        cushion = np.ascontiguousarray(cushion[:, ::-1])
        jaw_p = np.ascontiguousarray(jaw_p[:, ::-1])
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        pivots = {k: (mx(x), y) for k, (x, y) in pivots.items()}
        mouth = (mx(mouth[0]), mouth[1])
        center = (mx(center[0]), center[1])
        soles = [(mx(x), y) for x, y in soles]
        sole_mid = (mx(sole_mid[0]), sole_mid[1])
        facing = 1.0

    rest = np.zeros_like(canvas)
    paste(rest, cushion)
    for n in DRAW:
        paste(rest, layers[n])
    if np.any(rest != canvas):
        raise SystemExit("rest pose does not match the still")
    return {
        "layers": layers, "cushion": cushion, "jaw": jaw_p, "pivots": pivots,
        "mouth": mouth, "center": center, "soles": soles, "sole_mid": sole_mid,
        "facing": facing, "rest": canvas,
    }


# ------------------------------------------------------------------- poses

def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * f / n
    return amp * (math.sin(ph + phase) - math.sin(phase))


def blank_pose():
    return dict(
        body_dy=0, body_dx=0, lift_l=0, lift_r=0, fwd_l=0, fwd_r=0,
        arm_l=0.0, arm_r=0.0, flame_l=0.0, flame_r=0.0, tail=0.0,
        jaw=0, ring=0, flare=0, spin=None, spin_dy=0, firering=0,
        lean=0.0, knock=0, flash=0, embers=0, blast=0, dust=0,
    )


def pose(action, f):
    """Angles are for the left-facing still (+ = clockwise on screen)."""
    p = blank_pose()
    n = FRAMES[action]
    if action == "idle":
        # Heavy breathing: the upper body sinks 1px twice per loop. Flames flicker.
        p["body_dy"] = int(round(max(0.0, wave(f, n, 1.0, cycles=2.0))))
        p["flame_l"] = wave(f, n, 5.0, cycles=2.0)
        p["flame_r"] = wave(f, n, -5.0, phase=0.9, cycles=2.0)
        p["tail"] = wave(f, n, 7.0, phase=0.4)
        p["arm_l"] = wave(f, n, 4.0, phase=0.3)
        p["arm_r"] = wave(f, n, -4.0, phase=0.3)
        p["flare"] = (0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1)[f % n]
    elif action == "walk":
        # Two heavy stomps per loop. Lift is only on the rising half of each step.
        step = math.sin(2 * math.pi * f / n)
        p["lift_l"] = int(round(max(0.0, step) * 4.0))
        p["lift_r"] = int(round(max(0.0, -step) * 4.0))
        p["fwd_l"] = int(round(max(0.0, step) * 2.0))
        p["fwd_r"] = int(round(max(0.0, -step) * 2.0))
        # body rides up when a foot is lifted, drops on the stomp
        p["body_dy"] = -int(round(abs(step) * 1.4))
        p["arm_l"] = wave(f, n, -10.0)
        p["arm_r"] = wave(f, n, -10.0)
        p["flame_l"] = wave(f, n, 6.0, phase=0.6, cycles=2.0)
        p["flame_r"] = wave(f, n, 6.0, phase=0.6, cycles=2.0)
        p["tail"] = wave(f, n, 9.0, phase=-0.7)
        p["dust"] = 1 if f in (1, 9) else 0
        p["flare"] = (0, 1, 2, 2, 1, 0, 1, 2, 2, 1, 0, 1, 2, 2, 1, 0)[f % n]
    elif action == "attack":
        if f <= 4:
            # Roar: jaw drops on the existing lips, cannons flex up, flames flare.
            p["jaw"] = (0, 1, 2, 3, 3)[f]
            p["ring"] = (0, 1, 2, 3, 3)[f]
            k = (0.0, 0.45, 0.8, 1.0, 1.0)[f]
            p["arm_l"] = 42.0 * k
            p["arm_r"] = -42.0 * k
            p["flame_l"] = -8.0 * k
            p["flame_r"] = 8.0 * k
            p["tail"] = 12.0 * k
            p["flare"] = (0, 2, 3, 4, 4)[f]
            p["body_dy"] = (0, 1, 1, 0, 0)[f]
        elif f <= 10:
            # One-piece jump-spin inside a ring of fire (the spin turns the still).
            p["spin"] = 60.0 * (f - 4)
            p["spin_dy"] = (-6, -12, -15, -15, -12, -6)[f - 5]
            p["firering"] = f - 4
        elif f <= 12:
            # Land and fire the near cannon forward.
            p["arm_l"] = (66.0, 70.0)[f - 11]
            p["arm_r"] = (-22.0, -18.0)[f - 11]
            p["body_dx"] = (2, 3)[f - 11]  # recoil backwards
            p["blast"] = (1, 2)[f - 11]
            p["dust"] = 1 if f == 11 else 0
            p["jaw"] = 2
            p["flame_l"] = -6.0
            p["flame_r"] = 6.0
            p["flare"] = 3
        else:
            k = (0.5, 0.2, 0.0)[f - 13]
            p["arm_l"] = 66.0 * k
            p["arm_r"] = -20.0 * k
            p["body_dx"] = int(round(2 * k))
            p["jaw"] = 1 if f == 13 else 0
            p["flare"] = (2, 1, 0)[f - 13]
    else:
        ang = (0, 10, 20, 26, 28, 25, 20, 14, 9, 5, 2, 0)
        back = (0, 2, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0)
        p["lean"] = float(ang[f])
        p["knock"] = back[f]
        p["flash"] = 1 if f == 1 else 0
        p["embers"] = 1 if 2 <= f <= 7 else 0
    return p


# ------------------------------------------------------------------ effects

def put_empty(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return True
    return False


def put_any(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def open_jaw(img, jaw, drop):
    """Drop the existing lower lip/chin and show the dark mouth behind it."""
    if drop <= 0:
        return img
    out = img.copy()
    part = np.zeros_like(img)
    part[jaw] = img[jaw]
    ys, xs = np.nonzero(jaw)
    ytop = int(ys.min())
    top_xs = xs[ys == ytop]
    xa, xb = int(top_xs.min()), int(top_xs.max())
    out[jaw] = 0
    for k in range(drop):
        y = ytop + k
        for x in range(xa, xb + 1):
            edge = x in (xa, xb)
            out[y, x, :3] = INK if edge else DGRY
            out[y, x, 3] = 255
        # a hot glow deep in the throat when the mouth is wide
        if drop >= 2 and k == drop - 1:
            mid = (xa + xb) // 2
            for x in range(mid - 2, mid + 3):
                out[y, x, :3] = ORG
    moved = shift(part, 0, drop)
    m = moved[..., 3] >= 128
    out[m] = moved[m]
    return out


def roar_rings(out, mouth, facing, strength):
    mx, my = mouth
    cols = (YEL, ORG, WHITE)
    for r_i in range(strength):
        rad = 10 + r_i * 6
        col = cols[r_i % 3]
        for deg in range(-60, 61, 6):
            ang = math.radians(deg)
            put_empty(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.8, col)
    return out


def flare_tip(img, layer, pivot, amount, seed):
    """Lick the existing flame tip a few pixels further out (fire colours only)."""
    if amount <= 0:
        return img
    a = layer[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return img
    d = (xs - pivot[0]) ** 2 + (ys - pivot[1]) ** 2
    order = np.argsort(-d)[:3]
    cols = (RED2, ORG, YEL, RED)
    rng = np.random.default_rng(seed)
    for j, i in enumerate(order):
        tx, ty = float(xs[i]), float(ys[i])
        vx, vy = tx - pivot[0], ty - pivot[1]
        nrm = math.hypot(vx, vy) or 1.0
        ux, uy = vx / nrm, vy / nrm
        reach = amount if j == 0 else max(0, amount - 2)
        wob = float(rng.integers(-1, 2)) * 0.6
        for k in range(1, reach + 1):
            put_empty(img, tx + ux * k + wob * uy, ty + uy * k - wob * ux, cols[min(k - 1, 3)])
    return img


def fire_ball(out, cx, cy, r):
    for y in range(int(cy - r - 1), int(cy + r + 2)):
        for x in range(int(cx - r - 1), int(cx + r + 2)):
            d = math.hypot(x - cx, y - cy)
            if d <= r:
                col = WHITE if d < r * 0.35 else YEL if d < r * 0.65 else ORG if d < r * 0.88 else RED
                put_any(out, x, y, col)


def cannon_tip(layer, pivot):
    a = layer[..., 3] >= 128
    ys, xs = np.nonzero(a)
    d = (xs - pivot[0]) ** 2 + (ys - pivot[1]) ** 2
    far = np.argsort(-d)[:6]
    return float(xs[far].mean()), float(ys[far].mean())


def blast(out, tip, pivot, facing, stage):
    tx, ty = tip
    vx, vy = tx - pivot[0], ty - pivot[1]
    nrm = math.hypot(vx, vy) or 1.0
    ux, uy = vx / nrm, vy / nrm
    if stage == 1:
        fire_ball(out, tx + ux * 6, ty + uy * 6, 4.5)
        for k in range(8):
            ang = k * math.pi / 4 + 0.3
            for r in range(6, 9):
                put_empty(out, tx + ux * 6 + r * math.cos(ang), ty + uy * 6 + r * math.sin(ang), YEL)
    else:
        cx, cy = tx + ux * 15, ty + uy * 15
        fire_ball(out, cx, cy, 7.0)
        for k in range(10):
            ang = k * math.pi / 5 + 0.15
            for r in range(9, 14):
                put_empty(out, cx + r * math.cos(ang), cy + r * math.sin(ang), YEL if r < 12 else ORG)
        # muzzle trail between cannon and fireball
        for k in range(2, 8):
            put_empty(out, tx + ux * k, ty + uy * k, ORG)
            put_empty(out, tx + ux * k - uy, ty + uy * k + ux, YEL)
    return out


def fire_ring(out, center, step, facing):
    """Fire Spin: an orbit of flame dashes that turns with the body."""
    cx, cy = center
    cols = (RED2, ORG, YEL, ORG)
    base = math.radians(60.0 * step) * facing
    for arm in range(3):
        a0 = base + arm * 2 * math.pi / 3
        for k in range(10):
            ang = a0 - facing * k * 0.09
            r = 46 - k * 0.3
            col = cols[min(k // 3, 3)]
            put_empty(out, cx + r * math.cos(ang), cy + 0.55 * r * math.sin(ang), col)
            put_empty(out, cx + (r - 1) * math.cos(ang), cy + 0.55 * (r - 1) * math.sin(ang), col)


def speed_arcs(out, center, step):
    cx, cy = center
    for k in range(4):
        ang = math.radians(60.0 * step + 90 * k + 20)
        for j in range(5):
            a = ang + j * 0.06
            put_empty(out, cx + 40 * math.cos(a), cy + 40 * math.sin(a), WHITE if j < 2 else YEL)


def dust(out, soles, facing):
    for sx, sy in soles:
        for k, (dx, dy) in enumerate(((-6, 0), (-8, -1), (6, 0), (8, -1), (-4, -2), (4, -2))):
            put_empty(out, sx + dx, sy + dy, CLAW if k < 4 else WHITE)
    return out


def embers(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    cx = (x0 + x1) // 2
    back = -facing
    pts = ((cx + back * (10 + frame_idx), y0 + 6), (cx + back * (4 + 2 * frame_idx), y0 - 2),
           (cx + back * 16, y0 + 14), (cx - back * 6, y0 + 2 - frame_idx))
    for i, (sx, sy) in enumerate(pts):
        put_empty(out, sx, sy, WHITE if i % 2 == 0 else YEL)
        put_empty(out, sx + back, sy + 1, ORG)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 120)
    out[a & ~dark, :3] = WHITE
    return out


# --------------------------------------------------------------------- synth

def compose_parts(rig, p):
    L = rig["layers"]
    piv = rig["pivots"]
    sgn = -rig["facing"]  # angles are authored for the left-facing still
    bdx = int(p["body_dx"]) * (-rig["facing"])
    bdy = int(p["body_dy"])

    def rot(name):
        ang = p[name] * sgn
        return shift(rotate_nearest(L[name], ang, piv[name]), bdx, bdy)

    parts = {
        "arm_r": rot("arm_r"),
        "tail": rot("tail"),
        "leg_ext": L["leg_ext"],
        "foot_r": shift(L["foot_r"], rig["facing"] * p["fwd_r"], -p["lift_r"]),
        "foot_l": shift(L["foot_l"], rig["facing"] * p["fwd_l"], -p["lift_l"]),
        "torso": shift(L["torso"], bdx, bdy),
        "flame_r": rot("flame_r"),
        "flame_l": rot("flame_l"),
        "head": shift(L["head"], bdx, bdy),
        "arm_l": rot("arm_l"),
    }
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(img, shift(rig["cushion"], bdx, bdy))
    for n in DRAW:
        paste(img, parts[n])
    if any(abs(p[n]) >= 0.45 for n in ROTATING):
        img = despeckle(img)
    if p["jaw"]:
        img = open_jaw(img, shift_mask(rig["jaw"], bdx, bdy), int(p["jaw"]))
    return img, parts, (bdx, bdy)


def shift_mask(m, dx, dy):
    lyr = np.zeros(m.shape + (4,), np.uint8)
    lyr[m, 3] = 255
    return shift(lyr, dx, dy)[..., 3] >= 128


def synth(rig, action, f, palette):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    facing = rig["facing"]
    piv = rig["pivots"]
    if p["spin"] is not None:
        img = rotate_nearest(rig["rest"], p["spin"] * facing, rig["center"])
        img = close_cracks(img)
        img = despeckle(img)
        img = shift(img, 0, p["spin_dy"])
        img = snap(img, palette)
        c = (rig["center"][0], rig["center"][1] + p["spin_dy"])
        fire_ring(img, c, p["firering"], facing)
        speed_arcs(img, c, p["firering"])
        return snap(img, palette)
    if p["lean"] >= 0.45:
        # One-piece lean back away from the facing side, rocking on the rear foot.
        img = rotate_nearest(rig["rest"], -facing * p["lean"], rig["soles"][1])
        img = close_cracks(img)
        img = despeckle(img)
        img = shift(img, -facing * p["knock"], 0)
        img = snap(img, palette)
        if p["flash"]:
            img = flash_white(img)
        if p["embers"]:
            img = embers(img, f, facing)
        return snap(img, palette)

    img, parts, (bdx, bdy) = compose_parts(rig, p)
    img = snap(img, palette)
    if p["flare"]:
        for n in ("flame_l", "flame_r"):
            pv = (piv[n][0] + bdx, piv[n][1] + bdy)
            img = flare_tip(img, parts[n], pv, int(p["flare"]), seed=f * 7 + len(n))
    if p["ring"]:
        mouth = (rig["mouth"][0] + bdx, rig["mouth"][1] + bdy + p["jaw"])
        img = roar_rings(img, mouth, facing, int(p["ring"]))
    if p["blast"]:
        pv = (piv["arm_l"][0] + bdx, piv["arm_l"][1] + bdy)
        tip = cannon_tip(parts["arm_l"], pv)
        img = blast(img, tip, pv, facing, int(p["blast"]))
    if p["dust"]:
        img = dust(img, rig["soles"], facing)
    if p["flash"]:
        img = flash_white(img)
    return snap(img, palette)


# ------------------------------------------------------------------- output

def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (STILL_DIR / "front.png").read_bytes()
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


def tile_grid(tiles, cols, crop=None, scale=1, labels=None):
    if crop:
        x0, y0, x1, y1 = crop
        tiles = [t[y0:y1, x0:x1] for t in tiles]
    th, tw = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    bg = np.full((rows * th, cols * tw, 4), (15, 20, 31, 255), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        cell = bg[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
        cell[(r + c) % 2 == 0] = cell[(r + c) % 2 == 0]
        m = t[..., 3] >= 128
        cell[m] = t[m]
        cell[0, :] = (40, 48, 64, 255)
        cell[:, 0] = (40, 48, 64, 255)
    if scale > 1:
        bg = np.repeat(np.repeat(bg, scale, 0), scale, 1)
    return Image.fromarray(bg, "RGBA")


def preview_image(sheets):
    """South row, every action, sampled frames."""
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 2, 4, 8, 10, 12),
        "attack": (0, 3, 5, 6, 7, 8, 11, 12),
        "hurt": (0, 1, 4, 8),
    }
    tiles = []
    cols = 8
    for act in ACTIONS:
        row = [sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W] for f in samples[act]]
        row += [np.zeros((CELL_H, CELL_W, 4), np.uint8)] * (cols - len(row))
        tiles += row
    return tile_grid(tiles, cols)


def motion_contact(sheets):
    """Every frame of the E row (screen-right) and the W row."""
    tiles = []
    for di in (2, 6):
        for act in ACTIONS:
            row = [sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                   for f in range(FRAMES[act])]
            row += [np.zeros((CELL_H, CELL_W, 4), np.uint8)] * (16 - len(row))
            tiles += row
    return tile_grid(tiles, 16, crop=(20, 30, CELL_W - 20, CELL_H - 4))


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def differ_count(a_img, b_img):
    differ = np.any(a_img != b_img, axis=2) & ((a_img[..., 3] >= 128) | (b_img[..., 3] >= 128))
    return int(differ.sum())


def body_pieces(tile, min_big=70):
    sizes = component_sizes(tile)
    return [s for s in sizes if s >= min_big], sizes


def publish_dir8(sprite_h):
    entry = {
        "dex": 467, "form": "normal", "variant": "magmortar", "gen": "gen4",
        "dir": "monsterSprite/gen4/0467_magmortar/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    text = DIR8.read_text()
    prefix = "window.ARENA_DIR8 = "
    if not text.startswith(prefix):
        raise SystemExit("dir8.js does not start with the assignment")
    body_end = text.rstrip().rfind(";")
    before = json.loads(text[len(prefix):body_end])
    token = '"' + KEY + '":'
    start = text.find(token)
    if start < 0 or text.find(token, start + 1) != -1:
        raise SystemExit("dir8 key missing or duplicated")
    val_at = start + len(token)
    _old, end = json.JSONDecoder().raw_decode(text, val_at)
    new_val = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    out = text[:val_at] + new_val + text[end:]
    after = json.loads(out[len(prefix):out.rstrip().rfind(";")])
    if list(after) != list(before):
        raise SystemExit("dir8 key order changed")
    for k, v in before.items():
        if k != KEY and after[k] != v:
            raise SystemExit(f"dir8 neighbour changed: {k}")
    if after[KEY] != entry:
        raise SystemExit("dir8 entry did not stick")
    tmp = DIR8.with_suffix(".js.tmp")
    tmp.write_text(out)
    tmp.replace(DIR8)


def set_queue_status(status, note=None):
    text = QUEUE.read_text(encoding="utf-8")
    before = json.loads(text)
    anchor = '"id": "' + KEY + '"'
    i = text.find(anchor)
    if i < 0:
        raise SystemExit("queue has no item")
    j = text.find('"status": "', i)
    k = text.find('"', j + len('"status": "'))
    if j < 0 or j - i > 300:
        raise SystemExit("queue status field not found near the item")
    out = text[:j] + '"status": "' + status + text[k:]
    if note:
        nj = out.find('"notes": "', i)
        nk = nj + len('"notes": "')
        out = out[:nk] + note.replace('"', "'") + " " + out[nk:]
    after = json.loads(out)
    for b, a in zip(before["items"], after["items"]):
        if b.get("id") == KEY:
            if a.get("status") != status:
                raise SystemExit("queue status did not stick")
            continue
        if a != b:
            raise SystemExit(f"queue neighbour changed: {b.get('id')}")
    tmp = QUEUE.with_suffix(".json.tmp")
    tmp.write_text(out, encoding="utf-8")
    tmp.replace(QUEUE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true",
                    help="after the sheet was looked at: write the dir8 key and mark the queue done")
    ap.add_argument("--needs-fix", default=None, help="mark the queue needs_fix with this note")
    args = ap.parse_args()

    if args.needs_fix:
        set_queue_status("needs_fix", args.needs_fix)
        print("queue needs_fix", flush=True)
        return

    front = np.array(Image.open(STILL_DIR / "front.png").convert("RGBA"))
    if set(np.unique(front[..., 3]).tolist()) - {0, 255}:
        raise SystemExit("front alpha is not binary")
    palette = extract_palette(front)
    masks, counts, jaw, eye_x = segment(front)
    print("parts", counts, "palette", len(palette), "eye_x", round(eye_x, 1), flush=True)

    rigs = {d: build_rig(front, masks, jaw, flip=d not in FACE_LEFT) for d in DIRS}
    placed = placed_still(front)
    diff_w = differ_count(synth(rigs["W"], "idle", 0, palette), placed)
    diff_e = differ_count(synth(rigs["E"], "idle", 0, palette), placed[:, ::-1])
    print("rest-pose pixels changed W", diff_w, "E", diff_e, flush=True)
    if diff_w or diff_e:
        raise SystemExit("idle frame 0 does not match the still")
    for d in DIRS:
        for act in ("idle", "walk", "hurt"):
            if np.any(synth(rigs[d], act, 0, palette) != rigs[d]["rest"]):
                raise SystemExit(f"{act} frame 0 is not the still ({d})")
    for act in ("idle", "walk"):
        p_end = pose(act, FRAMES[act])
        if any(abs(p_end[k]) > 0.45 for k in ROTATING) or p_end["body_dy"]:
            raise SystemExit(f"{act} loop end is not the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part_cols = {
        "torso": (150, 150, 160), "head": (255, 220, 40), "flame_l": (255, 80, 20),
        "flame_r": (255, 140, 0), "arm_l": (40, 180, 255), "arm_r": (20, 80, 220),
        "tail": (160, 40, 160), "foot_l": (40, 200, 80), "foot_r": (20, 120, 60),
    }
    vis = np.zeros((96, 96, 4), np.uint8)
    for n, col in part_cols.items():
        vis[masks[n], :3] = col
        vis[masks[n], 3] = 255
    vis[jaw, :3] = (255, 255, 255)
    Image.fromarray(np.repeat(np.repeat(vis, 6, 0), 6, 1), "RGBA").save(work / "parts.png")

    spin_cc = {f: body_pieces(synth(rigs["W"], "attack", f, palette))[0] for f in range(5, 11)}
    hurt_cc = {f: body_pieces(synth(rigs["W"], "hurt", f, palette))[0] for f in range(12)}
    print("spin big pieces", spin_cc, flush=True)
    print("hurt big pieces", hurt_cc, flush=True)

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "W", f) for f in range(16)]
        want += [("hurt", "W", f) for f in (0, 1, 2, 4, 6, 8, 10, 11)]
        want += [(a, d, f) for d in ("S", "E", "N") for a, f in (("idle", 0), ("attack", 3), ("attack", 7), ("attack", 12))]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        tile_grid(tiles, 8, crop=(16, 40, CELL_W - 16, CELL_H - 6), scale=2).save(work / "samples.png")
        print("samples", work / "samples.png", flush=True)
        return

    frames_dir = work / "frames"
    state_p = work / "state.json"
    digest = params_hash()
    state = {}
    if state_p.exists():
        try:
            state = json.loads(state_p.read_text())
        except json.JSONDecodeError:
            state = {}
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
            if key in state["done"] and all(
                (od / f"{f:02d}.png").exists() and (od / f"{f:02d}.png").stat().st_size > 0
                for f in range(FRAMES[act])
            ):
                continue
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.exists() and fp.stat().st_size > 0:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            if key not in state["done"]:
                state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    preview.save(OUT_DIR / "actions" / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1)

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    clip, split = [], []
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = t[..., 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.append(f"{act}/{d}/{f}")
                big, _ = body_pieces(t)
                fireball = act == "attack" and f in (11, 12) and len(big) == 2 and big[1] < 200
                if len(big) != 1 and not fireball:
                    split.append(f"{act}/{d}/{f}:{big}")
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "rest_diff_w": diff_w, "rest_diff_e": diff_e, "off_palette_colors": len(off),
        "alphas": alphas, "edge_touching": len(clip), "edge_list": clip[:20],
        "body_split": len(split), "split_list": split[:20],
        "spin_big_w": {str(k): v for k, v in spin_cc.items()},
        "hurt_big_w": {str(k): v for k, v in hurt_cc.items()},
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
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:2000], flush=True)

    meta = {
        "generator": "forge_magmortar_actions", "version": VERSION, "key": KEY, "dex": 467,
        "variant": "magmortar", "form": "normal", "archetype": "biped_kaiju_stomper (cannon arms)",
        "facing_still": "left", "mouth_tip_x": MOUTH_TIP[0], "eye_mean_x": round(eye_x, 1),
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS, "face_left": sorted(FACE_LEFT),
        "parts": counts, "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        if off or clip or split or diff_w or diff_e or alphas != [0, 255]:
            raise SystemExit("qa failed; not publishing")
        publish_dir8(sprite_h)
        set_queue_status("done")
        print("dir8 + queue done", VERSION, "h", sprite_h, flush=True)


if __name__ == "__main__":
    main()
