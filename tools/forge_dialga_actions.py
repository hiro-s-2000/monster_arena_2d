#!/usr/bin/env python3
"""Dialga (0483) normal 8-direction action sheet from this form's front.png / back.png.

Measured on normal/front.png (96x96, alpha 0/255, 3384 opaque, 13 colours):
  bbox x8-86 y5-91. Faces screen-left: the red eye (x16-17 y22-23) sits right
  of the snout tip (x10). Heavy quadruped steel dragon.
  jaw     lower jaw under the snout, x10-16 y27-32, hinge about (17,27)
  head    crest + snout + eye on the torso layer (moves by integer pixels only)
  neck    long neck with ridge spikes, stays on the torso
  crystal the blue chest gem x16-23 y50-57 (colour 32,74,255), stays on the torso
  fins    the pale dorsal fan of blades above the back, x47-82 y21-44
  tail    the spiked tail running right, x70-90 y44-61
  leg_a   left foreleg x8-27 y60-85 (the most forward claw)
  leg_b   middle leg x31-50 y62-91 (lowest toe, drawn nearest)
  leg_c   hind leg x55-72 y64-86
  The striped steel plate between leg_a and leg_b stays on the torso.

back.png (96x96, 3111 opaque, front palette + one blue) faces up-right:
  head and crest top-right, dorsal fan on the left of the neck, a blade on the
  right of the neck, tail running down-left, three legs:
  leg_a far hind x29-40 y75-85, leg_b near hind x47-61 y71-90, leg_c fore x66-88 y67-85.

Directions: front still S SW W unflipped, SE E mirrored. back still N NE
unflipped, NW mirrored. E looks screen-right, W screen-left, N is the back.

  idle   : heavy breathing (torso 1px, legs planted), fan and tail sway,
           the chest gem twinkles. Frame 0 and the loop end are the still.
  walk   : diagonal trot of the three drawn legs (leg_a + leg_c against
           leg_b), lifted swing foot, 1px stomp bob, tail and fan swing.
  attack : 0-4 roar: the real lower jaw drops, the fan flares, the tail
           rises, the gem glows and steel-blue roar rings leave the mouth
           (back rows: from the head, no mouth is drawn there).
           5-6 the fore claw rears up (no whole-body rotation),
           7-8 claw slam with slash arcs, 9-12 ground shock rings and a
           time burst of diamond shards, 13-15 recover.
  hurt   : 12 frames, white flash, recoil of the one-piece body (<= 18 deg)
           around the rear foot, shards, recover.

Palette-locked to each still. front/back png and gif are not written.
Resumable via agent-sprite-forge/_work. --force clears only this form's frames.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_dialga_actions.py [--samples]
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
SRC_DIR = ROOT / "monsterSprite/gen4/0483_dialga/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0483_dialga/normal/agent-sprite-forge"
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
VERSION = "dialga_v1"
KEY = "0483_dialga"

INK = (16, 16, 16)
NAVY = (32, 65, 106)
STEEL = (139, 164, 180)
ICE = (156, 213, 246)
WHITE = (222, 238, 255)
PALE = (189, 205, 222)
MID = (57, 123, 180)
GEM = (32, 74, 255)

PARTS = ("fins", "fin_r", "tail", "leg_a", "leg_b", "leg_c", "jaw")


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


def segment(img, kind):
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    empty = np.zeros_like(a)
    if kind == "front":
        jaw = a & (yy >= 27) & (yy <= 32) & (xx >= 10) & (xx <= 16)
        fins = flood((25, 61), a & (yy <= 44) & (xx >= 47))
        tail = flood((48, 84), a & (xx >= 70) & (yy >= 44) & (yy <= 61) & ~fins)
        leg_c = flood((84, 69), a & (yy >= 64) & (xx >= 55) & ~tail)
        lim = np.where(yy <= 68, np.minimum(21 + (yy - 60), 26), 30)
        leg_a = flood((84, 13), a & (yy >= 60) & (xx <= lim))
        allow_b = a & (yy >= 62) & (xx >= 31) & (xx <= 52) & ~((xx >= 46) & (yy <= 65)) & ~leg_a
        leg_b = flood((89, 46), allow_b)
        fin_r = empty
    else:
        jaw = empty
        fins = flood((30, 26), a & (yy <= 48) & (xx <= 58) & (xx >= 20))
        fin_r = flood((42, 84), a & (xx >= 77) & (yy >= 38) & (yy <= 49))
        tail = flood((60, 9), a & (xx <= 38) & (yy >= 55) & (yy <= 72) & ~fins)
        leg_a = flood((84, 33), a & (yy >= 75) & (xx >= 28) & (xx <= 42))
        leg_b = flood((89, 55), a & (yy >= 71) & (xx >= 47) & (xx <= 61))
        leg_c = flood((84, 80), a & (yy >= 67) & (xx >= 66))
    parts = {"fins": fins, "fin_r": fin_r, "tail": tail, "leg_a": leg_a,
             "leg_b": leg_b, "leg_c": leg_c, "jaw": jaw}
    cover = np.zeros_like(a)
    for name, m in parts.items():
        if np.any(cover & m):
            raise SystemExit(f"{kind}: {name} overlaps another part")
        cover |= m
    parts = {"torso": a & ~cover, **parts}
    if kind == "front":
        eye = is_col(img, (246, 82, 65))
        if np.any(eye & ~parts["torso"]):
            raise SystemExit("the eye left the head")
    for name in ("leg_a", "leg_b", "leg_c", "tail", "fins"):
        if int(parts[name].sum()) < 40:
            raise SystemExit(f"{kind}: {name} collapsed ({int(parts[name].sum())})")
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
    # joint cushions live on their own layer under everything, so a moved
    # part never leaves a hole and the rest pose stays the exact still
    under = np.zeros_like(canvas)
    for name in PARTS:
        if np.any(placed[name]):
            add_socket(under, body, placed[name], canvas)
    layers["under"] = under

    pivots = {}
    for name in PARTS:
        if not np.any(placed[name]):
            pivots[name] = (0.0, 0.0)
        elif name.startswith("leg"):
            pivots[name] = top_pivot(placed[name])
        else:
            pivots[name] = contact_pivot(placed[name], body)
    if kind == "front":
        jy, jx = np.nonzero(placed["jaw"])
        pivots["jaw"] = (float(jx.max()) + 1.0, float(jy.min()))
    opy, opx = np.nonzero(canvas[..., 3] >= 128)
    ground = int(opy.max())
    feet = {}
    for name in ("leg_a", "leg_b", "leg_c"):
        ly, lx = np.nonzero(placed[name])
        low = ly >= int(ly.max()) - 1
        feet[name] = (float(np.median(lx[low])), float(ly.max()))
    # rear foot: away from the facing side. front faces left, back faces right.
    face = -1.0 if kind == "front" else 1.0
    rear = max(feet.values(), key=lambda p: -face * p[0])
    gem = np.zeros((CELL_H, CELL_W), bool)
    if kind == "front":
        gem = is_col(canvas, GEM)
    if flip:
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        mx = lambda p: (CELL_W - 1 - p[0], p[1])  # noqa: E731
        pivots = {k: mx(v) for k, v in pivots.items()}
        feet = {k: mx(v) for k, v in feet.items()}
        rear = mx(rear)
        gem = gem[:, ::-1].copy()
        face = -face
    # strike claw: the most forward foreleg
    strike = "leg_a" if kind == "front" else "leg_c"
    draw = ["under", "fins", "tail", "leg_c", "leg_a", "torso", "fin_r", "leg_b", "jaw"]
    return {
        "kind": kind, "flip": flip, "layers": layers, "pv": pivots, "face": face,
        "ground": ground, "feet": feet, "rear": rear, "gem": gem, "strike": strike,
        "draw": draw,
    }


# ---------------------------------------------------------------- poses
def pose(action, f):
    n = FRAMES[action]
    p = dict(
        bdy=0, bdx=0, fins=0.0, fin_r=0.0, tail=0.0, jaw=0.0,
        leg_a=0.0, leg_b=0.0, leg_c=0.0, lift_a=0, lift_b=0, lift_c=0,
        strike=0.0, slift=0, rings=0, ring_age=0, glow=0, twinkle=0,
        slash=0, shock=0, burst=0, hurt=0.0, hdx=0, flash=0, shards=0, spin=0,
    )
    if action == "idle":
        # breathing: the upper body rises one pixel and settles, legs planted
        p["bdy"] = -int(round(max(0.0, math.sin(2 * math.pi * f / n)) * 1.0))
        p["fins"] = wave(f, n, 4.0, phase=0.6)
        p["fin_r"] = wave(f, n, 3.0, phase=0.6)
        p["tail"] = wave(f, n, 5.0, phase=1.2)
        p["twinkle"] = 1 if f in (9, 10) else 0
    elif action == "walk":
        ph = 2 * math.pi * 2 * f / n
        sw = math.sin(ph)
        # A pair = leg_a + leg_c, B = leg_b, opposite phase
        p["leg_a"] = 16.0 * sw
        p["leg_c"] = 16.0 * sw
        p["leg_b"] = -16.0 * sw
        swing_a = math.cos(ph)  # >0: A pair travels forward
        p["lift_a"] = int(round(3 * max(0.0, swing_a)))
        p["lift_c"] = int(round(2 * max(0.0, swing_a)))
        p["lift_b"] = int(round(3 * max(0.0, -swing_a)))
        p["bdy"] = -1 if abs(swing_a) > 0.7 else 0
        p["tail"] = wave(f, n, 7.0, cycles=2.0, phase=0.9)
        p["fins"] = wave(f, n, 3.0, cycles=2.0, phase=1.6)
        p["fin_r"] = wave(f, n, 3.0, cycles=2.0, phase=1.6)
    elif action == "attack":
        jaw = (10, 20, 28, 32, 30, 22, 14, 0, 0, 6, 8, 10, 6, 3, 0, 0)
        fins = (3, 6, 9, 11, 11, 11, 11, 8, 8, 8, 7, 6, 4, 2, 1, 0)
        tail = (2, 5, 8, 9, 8, 10, 12, 4, 2, 2, 2, 2, 1, 1, 0, 0)
        strike = (0, 0, 2, 6, 12, 26, 34, 12, 3, 3, 3, 3, 2, 1, 0, 0)
        slift = (0, 0, 0, 1, 2, 4, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0)
        bdy = (0, 0, -1, -1, -1, -2, -2, 0, 1, 1, 0, 0, 0, 0, 0, 0)
        p["jaw"], p["fins"], p["tail"] = jaw[f], fins[f], tail[f]
        p["fin_r"] = fins[f] * 0.6
        p["strike"], p["slift"], p["bdy"] = strike[f], slift[f], bdy[f]
        # planted legs brace: lean back on the rear legs during the rear-up
        brace = (0, 0, 0, 1, 2, 4, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0)[f]
        p["brace"] = brace
        if 1 <= f <= 5:
            p["rings"] = 3 if f >= 2 else 2
            p["ring_age"] = f - 1
        p["glow"] = 1 if 2 <= f <= 10 else 0
        p["slash"] = f - 6 if f in (7, 8) else 0
        p["shock"] = f - 7 if 8 <= f <= 12 else 0
        p["burst"] = (5, 4) [f - 11] if f in (11, 12) else 0
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 18.0 * env
        p["hdx"] = int(round(3.0 * env))
        p["jaw"] = 14.0 * env
        p["fins"] = -6.0 * env
        p["tail"] = 8.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["shards"] = f if 2 <= f <= 6 else 0
    return p


# ---------------------------------------------------------------- effects
def mouth_point(rig, bdx, bdy):
    """Front: the jaw hinge front. Back: the head tip (top-right of the crest)."""
    face = rig["face"]
    if rig["kind"] == "front":
        lyr = shift(rig["layers"]["jaw"], bdx, bdy)
    else:
        lyr = shift(rig["layers"]["torso"], bdx, bdy)
        a = lyr[..., 3] >= 128
        ys, xs = np.nonzero(a)
        cut = int(ys.min()) + 18
        lyr = lyr.copy()
        lyr[cut:] = 0
    ys, xs = np.nonzero(lyr[..., 3] >= 128)
    if len(xs) == 0:
        return None
    x = int(xs.max()) if face > 0 else int(xs.min())
    y = int(np.median(ys[xs == x]))
    return x + int(face) * 2, y


def roar_rings(out, mouth, face, age, strength):
    mx, my = mouth
    cols = (WHITE, ICE, STEEL)
    for r_i in range(strength):
        rad = 5 + age * 3 + r_i * 4
        col = cols[min(r_i, 2)]
        for deg in range(-54, 55, 5):
            ang = math.radians(deg)
            put(out, mx + face * rad * math.cos(ang), my + rad * math.sin(ang) * 0.8, col, True)
            if r_i == 0 and abs(deg) <= 35:
                put(out, mx + face * (rad + 1) * math.cos(ang), my + (rad + 1) * math.sin(ang) * 0.8, ICE, True)


def slash_arcs(out, foot, face, k):
    fx, fy = foot
    for i in range(3):
        r = 9 + i * 3 + k * 2
        for deg in range(-70, 25, 6):
            ang = math.radians(deg)
            x = fx + face * (4 + r * math.cos(ang) * 0.7)
            y = fy - 10 + r * math.sin(ang)
            put(out, x, y, WHITE if i == 1 else ICE, True)


def ground_shock(out, foot, age, ground):
    fx, _ = foot
    for i, col in enumerate((WHITE, ICE, STEEL)):
        rx = 6 + age * 5 + i * 4
        ry = max(2, rx * 0.28)
        if rx > 60:
            continue
        for deg in range(0, 360, 7):
            ang = math.radians(deg)
            x = fx + rx * math.cos(ang)
            y = ground + 1 + ry * math.sin(ang)
            if y > ground - 2:
                put(out, x, y, col, True)


def diamond(out, cx, cy, r, col, edge):
    for dy in range(-r, r + 1):
        span = r - abs(dy)
        for dx in range(-span, span + 1):
            c = edge if abs(dx) == span else col
            put(out, cx + dx, cy + dy, c, True)


def time_burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4 + math.pi / 8
        for r in range(3, 3 + size * 3, 2):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang) * 0.75, (ICE, WHITE)[r % 2], True)
    for k in range(5):
        ang = k * 2 * math.pi / 5 + size
        d = 6 + size * 3
        diamond(out, cx + d * math.cos(ang), cy - 6 + d * math.sin(ang) * 0.6, 2, WHITE, MID)


def shards(out, frame_idx, face, box):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    d = frame_idx
    pts = ((cx + face * (6 + d * 2), y0 + 6 - d), (cx - face * (2 + d), y0 - 2 - d),
           (cx + face * 2, y0 + 14 - d))
    for sx, sy in pts:
        diamond(out, sx, sy, 2 if d % 2 else 1, WHITE, ICE)


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    ink = is_col(out, INK)
    out[a & ~ink, :3] = WHITE
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
    ang = {
        "fins": -face * p["fins"],        # blades lean back
        "fin_r": -face * p["fin_r"],
        "tail": face * p["tail"],          # tail tip rises
        "jaw": face * p["jaw"],            # jaw tip drops
    }
    legs = {}
    for leg, key in (("leg_a", "lift_a"), ("leg_b", "lift_b"), ("leg_c", "lift_c")):
        deg = fwd * p[leg]
        lift = int(p[key])
        if leg == rig["strike"]:
            deg += fwd * p["strike"]
            lift += int(p["slift"])
        elif p.get("brace"):
            # planted legs push back a little while the claw rears up
            deg -= fwd * p["brace"]
        legs[leg] = (deg, lift)
    moved = {}
    upper = ("under", "torso", "fins", "fin_r", "tail", "jaw")
    for name in upper:
        a_ = ang.get(name, 0.0)
        if name in ("torso", "under"):
            moved[name] = shift(L[name], bdx, bdy)
        else:
            jdy = (1 if p["jaw"] >= 18 else 0) if name == "jaw" else 0
            moved[name] = part_moved(L[name], pv[name], a_, bdx, bdy + jdy)
    for leg, (deg, lift) in legs.items():
        # legs follow the body sideways, stay on the ground vertically
        moved[leg] = part_moved(L[leg], pv[leg], deg, bdx, -lift + min(0, bdy) * 0)
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in rig["draw"]:
        if name == "jaw" and rig["kind"] == "front" and abs(p["jaw"]) >= 0.45:
            # mouth interior: the hole the dropped jaw leaves is ink
            home = shift(L["jaw"], bdx, bdy)
            hole = (home[..., 3] >= 128) & (moved["jaw"][..., 3] < 128)
            img[hole, :3] = INK
            img[hole, 3] = 255
            # the gap between the hinge and the dropped tip
            jm = moved["jaw"][..., 3] >= 128
            gap = dilate(home[..., 3] >= 128, 1) & dilate(jm, 1) & (img[..., 3] < 128) & ~jm
            img[gap, :3] = INK
            img[gap, 3] = 255
        paste(img, moved[name])
    if p["glow"] or p["twinkle"]:
        gem = rig["gem"]
        if np.any(gem):
            g = shift(np.dstack([gem.astype(np.uint8)] * 4), bdx, bdy)[..., 0] > 0
            gy, gx = np.nonzero(g)
            if p["glow"]:
                # light the upper-left of the gem
                cy, cx = gy.mean(), gx.mean()
                hi = g & (np.mgrid[0:CELL_H, 0:CELL_W][0] <= cy)
                img[hi, :3] = ICE
                img[g & (np.mgrid[0:CELL_H, 0:CELL_W][0] <= cy - 1.5), :3] = WHITE
            else:
                top = int(gy.min())
                row = g & (np.mgrid[0:CELL_H, 0:CELL_W][0] == top + 1)
                img[row, :3] = WHITE
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
    if p["rings"]:
        m = mouth_point(rig, int(p["bdx"]), int(p["bdy"]))
        if m:
            roar_rings(img, m, face, p["ring_age"], p["rings"])
    foot = rig["feet"][rig["strike"]]
    if p["slash"]:
        slash_arcs(img, (foot[0], foot[1] - 2), face, p["slash"])
    if p["shock"]:
        ground_shock(img, (foot[0] + face * 3, foot[1]), p["shock"], rig["ground"])
    if p["burst"]:
        time_burst(img, (foot[0] + face * 12, foot[1] - 10), p["burst"])
    if p["shards"]:
        bb = body_box(img)
        if bb:
            shards(img, p["shards"], face, bb)
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
    eye = is_col(stills["front"], (246, 82, 65))
    ey, ex = np.nonzero(eye)
    jy, jx = np.nonzero(masks["front"]["jaw"])
    print("facing check: jaw tip x", int(jx.min()), "eye x", float(ex.mean()), flush=True)
    if not jx.min() < ex.mean():
        raise SystemExit("front.png is expected to face left")

    rigs = {d: build_rig(stills[VIEW[d][0]], masks[VIEW[d][0]], VIEW[d][0], VIEW[d][1]) for d in DIRS}
    for d in ("W", "N"):
        print(d, "pivots", {k: tuple(round(v, 1) for v in p) for k, p in rigs[d]["pv"].items()},
              "rear", rigs[d]["rear"], flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)

    # still check for every direction
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
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in range(12)]
        want += [("walk", "N", f) for f in (0, 2, 4, 6)]
        want += [("attack", "N", f) for f in (2, 4, 6, 7, 9, 11)]
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
    # per-view palette check
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
        "generator": "forge_dialga_actions", "version": VERSION, "key": KEY,
        "dex": 483, "form": "normal", "variant": "dialga",
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "views": {d: {"still": VIEW[d][0], "mirrored": VIEW[d][1]} for d in DIRS},
        "archetype": "quadruped_beast (heavy steel dragon, 3 drawn legs)",
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
