#!/usr/bin/env python3
"""Lucario (0448) normal 8-direction action sheet from this form's 96x96 front.png.

Not Riolu, not Mega Lucario, not Mega Lucario Z. No rig or colour mask is
borrowed from those scripts. Parts come from rules measured on this still.

Measured on normal/front.png (alpha already 0/255, 13 colours):
  bbox x25-71 y17-79. Ears x38-51 y17-27. Head x35-52 y26-40.
  Snout tip x36 y34-36 is left of the red eye x45-48 y33-35, so the still
  faces LEFT. Unflipped dirs are N NW W SW, mirrored dirs are S SE E NE.
  Jaw: chin under the snout x37-46 y39-40 (dark outline row y39 above it).
  Black aura tassels behind the head x52-61 y29-41.
  Near arm: raised silver fist x25-36 y41-53, blue forearm to the shoulder x40 y46.
  Far arm: blue forearm x51-56 y45-53, silver fist x56-68 y42-52 with the white paw spike.
  Torso: cream chest x42-50 y41-55, waist y53-57.
  Tail: blue, x57-71 y54-64, off the right thigh.
  Legs: front thigh x32-43 y58-66, shin+foot x27-41 y66-77;
        back thigh x44-57 y58-66, shin+foot x49-67 y66-79.
  Archetype: biped (two legs, two arms with fists, tail, jaw, tassels).

  idle   : breathing, tassels and tail sway, fists bob. Frame 0 and loop end = still.
  walk   : alternating hip swing with knee bend, arms counter-swing, tail wag.
  attack : 0-4 roar (jaw opens as a wedge, rings, tassels flare, fist chambers),
           5-10 Aura Sphere palm thrust with a forward lunge (no flat body spin),
           11-12 impact burst, 13-15 recover.
  hurt   : 12 frames, white flash, one-piece recoil lean (<= 30 deg) and recover.

Head moves by integer pixels only, so the eye and face never resample.
Palette-locked to this still. front/back png and gif are never written.
"""

import argparse
import fcntl
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
import gen4_attack_strike  # noqa: E402  (drops any in-plane attack spin)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0448_lucario/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0448_lucario/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
BOUNDS = ROOT / "data/sprite_bounds.js"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

KEY = "0448_lucario"
VERSION = "lucario_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148

INK = (16, 16, 16)
DARK = (57, 57, 57)
GRAY = (90, 90, 90)
SILVER = (131, 131, 131)
LGRAY = (213, 213, 213)
WHITE = (255, 255, 255)
BLUE = (74, 156, 238)
MID = (41, 115, 156)
NAVY = (16, 74, 123)
CREAM = (230, 230, 156)
RED = (222, 65, 65)
DRED = (164, 49, 49)
FX_COLS = (INK, DARK, GRAY, WHITE, LGRAY, BLUE, MID, NAVY, RED, DRED)

# Back to front. Far limbs and tail behind the torso, near arm last.
DRAW = ("tail", "arm_b", "thigh_b", "shin_b", "torso", "thigh_f", "shin_f",
        "tassel", "head", "jaw", "arm_f")

# Joint pivots in still coordinates (x, y), measured on the contact pixels.
PIVOT = {
    "arm_f": (40.0, 46.0),
    "arm_b": (51.0, 46.0),
    "tail": (57.0, 60.0),
    "tassel": (53.0, 33.0),
    "thigh_f": (38.0, 58.0),
    "thigh_b": (50.0, 58.0),
    "shin_f": (38.0, 66.0),
    "shin_b": (53.0, 66.0),
}
PARENT = {
    "arm_f": "torso", "arm_b": "torso", "tail": ("thigh_b", "torso"),
    "tassel": "head", "thigh_f": "torso", "thigh_b": "torso",
    "shin_f": "thigh_f", "shin_b": "thigh_b",
}
EXPECT = None  # filled after the first measured run (see segment())


# ---------------------------------------------------------------- segmentation
def segment(img):
    a = img[..., 3] >= 128
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = img[..., :3].astype(np.int32)

    tassel = a & (yy <= 41) & ((xx >= 53) | ((yy >= 37) & (xx >= 52)))
    head = a & (yy <= 40) & (xx >= 35) & ~tassel
    # Lower jaw: chin rows under the snout outline, left of the cheek.
    jaw = head & (yy >= 39) & (xx <= 44)
    head = head & ~jaw
    arm_f = a & (yy >= 41) & (yy <= 54) & (xx <= 40)
    arm_b = a & ~tassel & (yy >= 41) & (yy <= 53) & ((xx >= 52) | ((yy >= 45) & (xx >= 51)))
    tail = a & ~arm_b & (yy >= 53) & (yy <= 64) & ((xx >= 58) | ((yy <= 62) & (xx >= 57)))
    legs = a & (yy >= 58) & ~tail
    front = legs & (xx <= 43)
    back = legs & (xx >= 44)
    thigh_f = front & (yy <= 65)
    shin_f = front & (yy >= 66)
    thigh_b = back & (yy <= 65)
    shin_b = back & (yy >= 66)
    torso = a & ~(tassel | head | jaw | arm_f | arm_b | tail | legs)
    masks = {
        "tassel": tassel, "head": head, "jaw": jaw, "arm_f": arm_f, "arm_b": arm_b,
        "tail": tail, "thigh_f": thigh_f, "shin_f": shin_f, "thigh_b": thigh_b,
        "shin_b": shin_b, "torso": torso,
    }
    total = sum(int(m.sum()) for m in masks.values())
    if total != int(a.sum()):
        raise SystemExit("segmentation overlaps or drops pixels")
    counts = {k: int(m.sum()) for k, m in masks.items()}
    if EXPECT is not None and counts != EXPECT:
        raise SystemExit(f"part counts drifted: {counts}")
    del rgb
    return masks, counts


# ---------------------------------------------------------------- raster helpers
def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


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
    """Rotate about pivot at 4x with nearest sampling, then back to 1x.

    Positive angle is counter-clockwise on screen: a point below the pivot
    moves left, a point left of the pivot moves up.
    """
    if abs(angle_deg) < 0.45:
        return layer.copy()
    a = layer[..., 3] >= 128
    if not np.any(a):
        return layer.copy()
    ys, xs = np.nonzero(a)
    px, py = pivot
    r = int(math.ceil(max(np.hypot(xs - px, ys - py).max(), 1.0))) + 2
    h, w = a.shape
    x0, x1 = max(0, int(px) - r), min(w, int(px) + r + 2)
    y0, y1 = max(0, int(py) - r), min(h, int(py) + r + 2)
    crop = layer[y0:y1, x0:x1]
    up = np.repeat(np.repeat(crop, 4, axis=0), 4, axis=1)
    cpx, cpy = (px - x0) * 4 + 1.5, (py - y0) * 4 + 1.5
    rad = math.radians(-angle_deg)  # sampling is the inverse map
    c, s = math.cos(rad), math.sin(rad)
    uh, uw = up.shape[:2]
    gy, gx = np.mgrid[0:uh, 0:uw]
    rx, ry = gx - cpx, gy - cpy
    sx = np.rint(cpx + c * rx - s * ry).astype(np.int32)
    sy = np.rint(cpy + s * rx + c * ry).astype(np.int32)
    ok = (sx >= 0) & (sx < uw) & (sy >= 0) & (sy < uh)
    rot = np.zeros_like(up)
    vy, vx = np.nonzero(ok)
    rot[vy, vx] = up[sy[vy, vx], sx[vy, vx]]
    out = np.zeros_like(layer)
    out[y0:y1, x0:x1] = rot[1::4, 1::4]
    return out


def rot_point(pt, angle_deg, pivot):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    dx, dy = pt[0] - pivot[0], pt[1] - pivot[1]
    # Forward map matching rotate_nearest: + is counter-clockwise on screen.
    return (pivot[0] + c * dx - s * dy, pivot[1] + s * dx + c * dy)


def despeckle(img):
    """Drop opaque pixels whose four cardinal neighbours are all empty."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def close_cracks(img, passes=2):
    """Fill 1px rotation cracks: an empty pixel with 3+ opaque cardinal neighbours."""
    out = img.copy()
    for _ in range(passes):
        a = out[..., 3] >= 128
        p = np.pad(a, 1)
        up, dn, lf, rt = p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
        n = up.astype(np.int16) + dn + lf + rt
        hole = (~a) & (n >= 3)
        if not np.any(hole):
            break
        col = np.zeros_like(out)
        for src, (dx, dy) in ((up, (0, 1)), (dn, (0, -1)), (lf, (1, 0)), (rt, (-1, 0))):
            use = hole & src & (col[..., 3] == 0)
            if np.any(use):
                col[use] = shift(out, dx, dy)[use]
        out[hole] = col[hole]
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    out[~a] = 0
    if not np.any(a):
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = a.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    mapped = palette[(diff * diff).sum(axis=2).argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv.reshape(-1)]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    return out


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


# ---------------------------------------------------------------- rig
def placement(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    dst_x = (CELL_W - (x1 - x0)) // 2
    dst_y = FOOT_Y - (y1 - y0)
    return x0, y0, x1, y1, dst_x - x0, dst_y - y0


def place(arr, img_shape_hw, off):
    ox, oy = off
    h, w = img_shape_hw
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    out[oy:oy + h, ox:ox + w] = arr
    return out


def build_rig(img, masks, flip):
    x0, y0, x1, y1, ox, oy = placement(img)
    h, w = img.shape[:2]
    canvas = place(img, (h, w), (ox, oy))
    pm = {k: place(m, (h, w), (ox, oy)) for k, m in masks.items()}
    layers = {}
    for k, m in pm.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    pivots = {k: (x + ox, y + oy) for k, (x, y) in PIVOT.items()}

    # Joint cushion: the part's own root pixels next to its parent stay put
    # underneath everything, so a swinging limb never opens a hole at the joint.
    cushion = np.zeros_like(canvas)
    for name, par in PARENT.items():
        pars = par if isinstance(par, tuple) else (par,)
        body = np.zeros((CELL_H, CELL_W), bool)
        for p in pars:
            body |= pm[p]
        zone = pm[name] & dilate(body, 2)
        cushion[zone] = canvas[zone]
    # Mouth hinge in still coords: cheek corner behind the chin.
    hinge = (46 + ox, 39 + oy)
    join_x = 37 + ox
    tip = (36 + ox, 39 + oy)
    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        cushion = cushion[:, ::-1].copy()
        canvas = canvas[:, ::-1].copy()
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        hinge = (CELL_W - 1 - hinge[0], hinge[1])
        tip = (CELL_W - 1 - tip[0], tip[1])
        join_x = CELL_W - 1 - join_x
    rest = np.zeros_like(canvas)
    paste(rest, cushion)
    for name in DRAW:
        paste(rest, layers[name])
    return {
        "layers": layers, "pivots": pivots, "cushion": cushion, "rest": rest,
        "flip": flip, "facing": 1.0 if flip else -1.0, "hinge": hinge, "tip": tip,
        "canvas": canvas, "join_x": join_x,
        "foot_row": int(np.nonzero(canvas[..., 3] >= 128)[0].max()),
    }


# ---------------------------------------------------------------- motion
def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * f / n
    return amp * (math.sin(ph + phase) - math.sin(phase))


def ease(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def pose(action, f):
    """Angles are in still (left-facing) space: + is counter-clockwise.

    For legs + swings the foot forward (left), for the near arm + lifts the fist,
    for the tail and tassels + lowers the tip (they hang to the right).
    """
    p = dict(dx=0, dy=0, lean=0.0, arm_f=0.0, arm_b=0.0, tail=0.0, tassel=0.0,
             thigh_f=0.0, thigh_b=0.0, knee_f=0.0, knee_b=0.0, jaw=0, ring=0,
             orb=None, burst=0, flash=0, sparks=0, lines=0, spin=0, aura=0, reach=0,
             plant=0, ground=0)
    n = FRAMES[action]
    if action == "idle":
        # Calm aura stance: one slow breath, tassels and tail trail it.
        # Upper body sinks 1 px on the out-breath; the shins stay planted.
        p["dy"] = 1 if 5 <= f <= 10 else 0
        p["plant"] = 1
        p["tassel"] = wave(f, n, 7.0, phase=0.6)
        p["tail"] = wave(f, n, -6.0, phase=-0.4)
        p["arm_f"] = wave(f, n, 5.0, phase=0.0)
        p["arm_b"] = wave(f, n, 4.0, phase=0.9)
    elif action == "walk":
        ph = 2 * math.pi * f / n
        # Frame 0 is the still's own stride (front foot ahead, back foot behind).
        # 0-8: front leg sweeps back on the ground, back leg swings through with
        # the knee lifted. 8-16: the mirror. The ground rule keeps a sole down.
        A = 15.0
        p["thigh_f"] = A * (math.cos(ph) - 1.0)
        p["thigh_b"] = -A * (math.cos(ph) - 1.0)
        p["knee_b"] = -38.0 * max(0.0, math.sin(ph)) ** 1.2
        p["knee_f"] = -38.0 * max(0.0, -math.sin(ph)) ** 1.2
        p["arm_f"] = 10.0 * (math.cos(ph) - 1.0) * 0.8
        p["arm_b"] = -9.0 * (math.cos(ph) - 1.0) * 0.8
        p["ground"] = 1
        p["tail"] = wave(f, n, 8.0, phase=0.0, cycles=2.0)
        p["tassel"] = wave(f, n, 6.0, phase=0.5, cycles=2.0)
    elif action == "attack":
        # 0-4 roar and chamber, 5-10 aura sphere palm, 11-12 impact, 13-15 recover.
        jaw = [0, 1, 2, 3, 3, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        ring = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        tas = [0, -10, -20, -26, -24, -16, -18, -20, -20, -18, -16, -12, -8, -4, -1, 0]
        arm_f = [0, -12, -30, -46, -52, -14, 4, 6, 6, 6, 5, 4, 3, 2, 1, 0]
        reach = [0, 0, 0, 0, 0, 3, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0]
        arm_b = [0, 8, 16, 22, 24, 10, -4, -8, -8, -8, -6, -4, -2, -1, 0, 0]
        tail = [0, -6, -12, -16, -14, 4, 10, 12, 12, 10, 8, 6, 4, 2, 1, 0]
        th_f = [0, 0, -2, -4, -4, 8, 14, 16, 16, 16, 14, 10, 6, 3, 1, 0]
        th_b = [0, 0, 2, 4, 4, -6, -10, -12, -12, -12, -10, -8, -5, -2, -1, 0]
        kn_f = [0, 0, -4, -8, -8, -6, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        dx = [0, 0, 1, 1, 1, -1, -3, -4, -4, -4, -3, -2, -2, -1, 0, 0]
        dy = [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        p.update(jaw=jaw[f], ring=ring[f], tassel=tas[f], arm_f=arm_f[f], arm_b=arm_b[f],
                 tail=tail[f], thigh_f=th_f[f], thigh_b=th_b[f], knee_f=kn_f[f],
                 dx=dx[f], dy=0, reach=reach[f], ground=1)
        # Aura sphere: charge at the fist, fly forward, burst.
        orb = {2: (0, 2), 3: (0, 3), 4: (0, 3), 5: (1, 4), 6: (2, 5), 7: (5, 6),
               8: (11, 6), 9: (17, 6), 10: (23, 6)}
        if f in orb:
            p["orb"] = orb[f]
        if f in (6, 7, 8, 9, 10):
            p["lines"] = 3
        if f == 11:
            p["burst"] = 12
        elif f == 12:
            p["burst"] = 8
        if 2 <= f <= 4:
            p["aura"] = 1
    else:
        ang = [0, -10, -18, -22, -24, -22, -17, -11, -6, -3, -1, 0]
        back = [0, 2, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0]
        p["lean"] = ang[f]
        p["dx"] = back[f]
        p["dy"] = [0, -1, -2, -2, -2, -1, -1, 0, 0, 0, 0, 0][f]
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
        p["jaw"] = [0, 1, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0][f]
    return p


# ---------------------------------------------------------------- effects
def put_empty(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def put_any(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def open_jaw(img, jaw_layer, rig, drop, dx, dy):
    """Hinge the existing chin open as a wedge: the tip drops `drop` px, the hinge 0."""
    if drop <= 0:
        paste(img, shift(jaw_layer, dx, dy))
        return img
    hx = rig["hinge"][0] + dx
    tx = rig["tip"][0] + dx
    span = abs(hx - tx) or 1
    lay = shift(jaw_layer, dx, dy)
    a = lay[..., 3] >= 128
    ys, xs = np.nonzero(a)
    moved = np.zeros_like(lay)
    for y, x in zip(ys.tolist(), xs.tolist()):
        t = min(1.0, max(0.0, abs(hx - x) / span))
        k = int(round(drop * t))
        moved[y + k, x] = lay[y, x]
    # Mouth cavity between the upper snout and the dropped chin.
    top = rig["hinge"][1] + dy  # first chin row
    for x in sorted(set(xs.tolist())):
        t = min(1.0, max(0.0, abs(hx - x) / span))
        k = int(round(drop * t))
        for j in range(k):
            y = top + j
            col = DRED if j == k - 1 and k >= 2 else INK
            if abs(x - tx) <= 1 and j == 0:
                col = WHITE if k >= 2 else INK  # fang glint at the tip
            put_any(img, x, y, col)
    paste(img, moved)
    return img


def roar_rings(out, origin, facing, strength):
    ox, oy = origin
    cols = (WHITE, BLUE, MID)
    for r_i in range(strength):
        rad = 6 + r_i * 5
        for deg in range(-55, 56, 9):
            ang = math.radians(deg)
            put_empty(out, ox + facing * rad * math.cos(ang), oy + rad * math.sin(ang) * 0.8, cols[r_i % 3])


def aura_glow(out, body_mask, frame):
    """Thin blue aura rim around the silhouette while charging."""
    rim = dilate(body_mask, 1) & ~body_mask
    ys, xs = np.nonzero(rim)
    if len(ys) == 0:
        return
    top = int(np.nonzero(body_mask)[0].min())
    for y, x in zip(ys.tolist(), xs.tolist()):
        # Upper body only (head, tassels, arms), every other pixel, flickering.
        if y < top + 40 and (x + y + frame) % 2 == 0:
            put_empty(out, x, y, BLUE if (y // 2 + frame) % 2 else MID)


def draw_orb(out, cx, cy, r):
    for y in range(int(cy - r - 1), int(cy + r + 2)):
        for x in range(int(cx - r - 1), int(cx + r + 2)):
            d = math.hypot(x - cx, y - cy)
            if d <= r - 2.2:
                col = WHITE
            elif d <= r - 1.0:
                col = BLUE
            elif d <= r:
                col = MID
            elif d <= r + 0.7:
                col = NAVY
            else:
                continue
            put_any(out, x, y, col)


def speed_lines(out, facing, n, seed, box):
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = box
    rear = x1 + 2 if facing < 0 else x0 - 2
    step = 1 if facing < 0 else -1
    for _ in range(n):
        y = int(rng.integers(y0 + 6, y1 - 6))
        ln = int(rng.integers(5, 10))
        off = int(rng.integers(1, 5))
        for i in range(ln):
            put_empty(out, rear + step * (off + i), y, BLUE if i < 2 else MID)


def impact_burst(out, cx, cy, size):
    for k in range(12):
        ang = k * math.pi / 6 + 0.13
        for r in range(2, size):
            col = WHITE if r < size * 0.45 else (BLUE if r < size * 0.75 else MID)
            put_any(out, cx + r * math.cos(ang), cy + r * math.sin(ang), col)
    for y in range(-2, 3):
        for x in range(-2, 3):
            if abs(x) + abs(y) <= 2:
                put_any(out, cx + x, cy + y, WHITE)


def hurt_sparks(out, f, facing, box):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    back = -facing
    for sx, sy in ((cx + back * (10 + f), y0 + 4), (cx + back * (5 + f), y0 - 2), (cx + back * 8, y0 + 12)):
        put_empty(out, sx, sy, WHITE)
        put_empty(out, sx + back, sy + 1, LGRAY)
        put_empty(out, sx, sy + 1, BLUE)


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


# ---------------------------------------------------------------- synth
def stretch_arm(layer, rig, k):
    """Push the near fist forward by k px. The forearm is cut at the measured
    column x=37 (rows 49-52: outline, blue, blue, outline) and that column is
    repeated across the gap, so the arm gets longer without new colours."""
    if k <= 0:
        return layer
    jx = rig["join_x"]
    fwd = -1 if not rig["flip"] else 1
    out = np.zeros_like(layer)
    w = layer.shape[1]
    xs = np.arange(w)
    ahead = (xs < jx) if fwd < 0 else (xs > jx)
    moved = layer.copy()
    moved[:, ~ahead] = 0
    out = shift(moved, fwd * k, 0)
    behind = layer.copy()
    behind[:, ahead] = 0
    paste(out, behind)
    col = layer[:, jx]
    for i in range(1, k + 1):
        x = jx + fwd * i
        m = col[:, 3] >= 128
        out[m, x] = col[m]
    return out


def synth(rig, action, f, palette):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)
    L = rig["layers"]
    P = rig["pivots"]
    sg = -1.0 if rig["flip"] else 1.0  # mirror angles for flipped dirs
    facing = rig["facing"]
    dx = int(p["dx"]) * (-1 if rig["flip"] else 1)  # pose dx<0 is forward (left in still)
    dy = int(p["dy"])

    def rot(name, ang, pivot=None):
        return rotate_nearest(L[name], ang * sg, pivot or P[name])

    def leg(th, sh, ang, knee, shin_dy=0):
        shin = rot(sh, knee)
        both = L[th].copy()
        paste(both, shin)
        both = rotate_nearest(both, ang * sg, P[th])
        if shin_dy:
            # Planted: cushion stays with the body, the shin is pinned by shin_dy.
            lower = shift(rot(sh, knee), 0, shin_dy)
            both = L[th].copy()
            paste(both, lower)
        return both

    parts = {}
    parts["tail"] = rot("tail", p["tail"])
    parts["arm_b"] = rot("arm_b", p["arm_b"])
    back_leg = leg("thigh_b", "shin_b", p["thigh_b"], p["knee_b"])
    front_leg = leg("thigh_f", "shin_f", p["thigh_f"], p["knee_f"])
    parts["torso"] = L["torso"]
    parts["tassel"] = rot("tassel", p["tassel"])
    parts["head"] = L["head"]
    arm_layer = stretch_arm(L["arm_f"], rig, int(p["reach"]))
    parts["arm_f"] = rotate_nearest(arm_layer, p["arm_f"] * sg, P["arm_f"])

    if p["ground"]:
        # Lowest sole lands on the rest foot line; the lifted foot stays up.
        low = max(int(np.nonzero(lg[..., 3] >= 128)[0].max()) for lg in (back_leg, front_leg))
        dy += rig["foot_row"] - low
    shin_dy = -dy if p["plant"] else 0
    if shin_dy:
        # Idle breath: the upper body sinks, the shins and soles do not.
        back_leg = leg("thigh_b", "shin_b", p["thigh_b"], p["knee_b"], shin_dy)
        front_leg = leg("thigh_f", "shin_f", p["thigh_f"], p["knee_f"], shin_dy)
    body = np.zeros_like(L["torso"])
    paste(body, rig["cushion"])
    paste(body, parts["tail"])
    paste(body, parts["arm_b"])
    paste(body, back_leg)
    paste(body, parts["torso"])
    paste(body, front_leg)
    paste(body, parts["tassel"])
    paste(body, parts["head"])
    moving = any(abs(p[k]) >= 0.45 for k in ("tail", "arm_b", "thigh_b", "thigh_f", "knee_f", "knee_b", "tassel", "arm_f"))
    if moving:
        body = despeckle(body)
    body = shift(body, dx, dy)
    open_jaw(body, L["jaw"], rig, int(p["jaw"]), dx, dy)
    arm = shift(parts["arm_f"], dx, dy)
    paste(body, arm)
    if moving:
        body = close_cracks(body, 1)

    if abs(p["lean"]) >= 0.45:
        # Hurt: the posed body tips back as one piece around the feet.
        bb = body_box(body)
        piv = ((bb[0] + bb[2]) / 2.0, float(bb[3] - 6))
        body = rotate_nearest(body, p["lean"] * sg, piv)
        body = close_cracks(body, 2)
        body = despeckle(body)

    body = snap(body, palette)
    if p["flash"]:
        body = flash_white(body)
    box = body_box(body)
    if p["aura"]:
        aura_glow(body, body[..., 3] >= 128, f)
    if p["ring"]:
        tx = rig["tip"][0] + dx + int(facing) * 1
        roar_rings(body, (tx, rig["tip"][1] + dy - 2), facing, p["ring"])
    if p["lines"]:
        speed_lines(body, facing, p["lines"], f * 31 + int(rig["flip"]), box)
    # Fist tip of the near arm, after its pose, is where the sphere starts.
    # Measured fist front in the still: 15 px ahead of the shoulder, 1 px lower.
    fist = rot_point((P["arm_f"][0] + facing * (15 + int(p["reach"])), P["arm_f"][1] + 1), p["arm_f"] * sg, P["arm_f"])
    fist = (fist[0] + dx, fist[1] + dy)
    if p["orb"] is not None:
        travel, r = p["orb"]
        cx = fist[0] + facing * (r + 1 + travel)
        draw_orb(body, cx, fist[1], r)
    if p["burst"]:
        cx = fist[0] + facing * (23 + 8)
        impact_burst(body, cx, fist[1], p["burst"])
    if p["sparks"]:
        hurt_sparks(body, f, facing, box)
    return snap(body, palette)


# ---------------------------------------------------------------- sheets, preview
def params_hash():
    src = (Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes()
           + (SRC_DIR / "front.png").read_bytes())
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
    samples = {"idle": (0, 4, 8, 12), "walk": (0, 2, 4, 8, 12), "attack": (0, 3, 6, 8, 11, 13), "hurt": (0, 1, 4, 8)}
    cols = 6
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            m = tile[..., 3] >= 128
            bg[ai * CELL_H:(ai + 1) * CELL_H, si * CELL_W:(si + 1) * CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets, row=2):
    pad = 4
    crops = []
    for act in ACTIONS:
        r = []
        for f in range(FRAMES[act]):
            tile = sheets[act][row * CELL_H:(row + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            bb = body_box(tile)
            if not bb:
                continue
            x0, y0, x1, y1 = bb
            r.append(tile[max(0, y0 - pad):min(CELL_H, y1 + 1 + pad), max(0, x0 - pad):min(CELL_W, x1 + 1 + pad)])
        crops.append(r)
    ch = max(c.shape[0] for r in crops for c in r)
    cw = max(c.shape[1] for r in crops for c in r)
    cols = max(len(r) for r in crops)
    bg = np.full((len(crops) * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    for ri, r in enumerate(crops):
        for ci, c in enumerate(r):
            y = ri * ch + (ch - c.shape[0]) // 2
            x = ci * cw + (cw - c.shape[1]) // 2
            m = c[..., 3] >= 128
            bg[y:y + c.shape[0], x:x + c.shape[1]][m] = c[m]
    return Image.fromarray(bg, "RGBA")


def tile_grid(tiles, cols=8, scale=1):
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        m = t[..., 3] >= 128
        sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = t[m]
    im = Image.fromarray(sheet, "RGBA")
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    return im


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    sub = OUT_DIR / "actions"
    if sub.exists():
        (SYNC_OUT / "actions").mkdir(exist_ok=True)
        for fp in sub.iterdir():
            if fp.is_file():
                shutil.copy2(fp, SYNC_OUT / "actions" / fp.name)
                n += 1
    return n


# ---------------------------------------------------------------- publishing
def with_flock(lock_path, fn):
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def publish_dir8(sprite_h):
    entry = {
        "dex": 448, "form": "normal", "variant": "lucario", "gen": "gen4",
        "dir": "monsterSprite/gen4/0448_lucario/normal",
        "sheetDir": "agent-sprite-forge", "layout": "actions_8dir", "nf": 16,
        "cw": CELL_W, "ch": CELL_H, "w": CELL_W, "h": int(sprite_h), "fy": FOOT_Y,
        "an": [16, 16, 16, 12], "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    new_val = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js prefix changed")
        token = '"' + KEY + '":'
        pos = text.find(token)
        if pos < 0 or text.find(token, pos + 1) != -1:
            raise SystemExit("dir8 key not unique")
        val_at = pos + len(token)
        _old, end = json.JSONDecoder().raw_decode(text, val_at)
        out = text[:val_at] + new_val + text[end:]
        before = json.loads(text[len(prefix):].strip().rstrip(";"))
        after = json.loads(out[len(prefix):].strip().rstrip(";"))
        if after[KEY] != entry or list(after) != list(before):
            raise SystemExit("dir8 edit failed")
        for k in before:
            if k != KEY and before[k] != after[k]:
                raise SystemExit(f"dir8 neighbour changed: {k}")
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


def update_queue(status):
    def edit():
        raw = QUEUE.read_text(encoding="utf-8")
        i = raw.find('"id": "' + KEY + '",')
        if i < 0:
            raise SystemExit("queue item missing")
        j = raw.find('"status": "', i)
        k = raw.find('"', j + len('"status": "'))
        out = raw[:j] + '"status": "' + status + raw[k:]
        json.loads(out)
        tmp = QUEUE.with_suffix(".json.tmp")
        tmp.write_text(out, encoding="utf-8")
        tmp.replace(QUEUE)

    with_flock(QUEUE_LOCK, edit)


# ---------------------------------------------------------------- main
def differ_count(a_img, b_img):
    d = np.any(a_img != b_img, axis=2) & ((a_img[..., 3] >= 128) | (b_img[..., 3] >= 128))
    return int(d.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true",
                    help="after visual approval: dir8 + sprite_bounds + queue done")
    ap.add_argument("--parts", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if front.shape[:2] != (96, 96):
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    if set(np.unique(front[..., 3]).tolist()) - {0, 255}:
        raise SystemExit("front alpha is not binary")
    palette = extract_palette(front)
    pal_set = {tuple(int(v) for v in c) for c in palette}
    if not set(FX_COLS) <= pal_set:
        raise SystemExit("fx colours are not in the still palette")
    masks, counts = segment(front)
    print("parts", counts, "palette", len(palette), flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    if args.parts:
        cols = {"tassel": (255, 80, 20), "head": (255, 220, 40), "jaw": (255, 0, 255),
                "arm_f": (40, 180, 255), "arm_b": (20, 80, 220), "tail": (160, 90, 40),
                "thigh_f": (40, 200, 80), "shin_f": (20, 120, 40), "thigh_b": (200, 120, 255),
                "shin_b": (120, 60, 180), "torso": (180, 180, 190)}
        part = np.zeros_like(front)
        for k, c in cols.items():
            part[masks[k], :3] = c
            part[masks[k], 3] = 255
        Image.fromarray(np.concatenate([front, part], 1), "RGBA").resize((192 * 6, 96 * 6), Image.NEAREST).save(work / "parts.png")
        print("parts", work / "parts.png")
        return

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    placed = rigs["W"]["canvas"]
    diff_w = differ_count(synth(rigs["W"], "idle", 0, palette), placed)
    diff_e = differ_count(synth(rigs["E"], "idle", 0, palette), placed[:, ::-1].copy())
    print("rest-pose pixels changed W", diff_w, "E", diff_e, flush=True)
    if diff_w or diff_e:
        raise SystemExit("idle frame 0 does not match the still")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "W", f) for f in range(16)]
        want += [("hurt", "W", f) for f in (0, 1, 2, 4, 6, 8, 10, 11)]
        want += [("attack", d, 7) for d in DIRS]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        tile_grid(tiles, 8, 1).save(work / "samples.png")
        # Zoomed crops for checking the face and joints.
        groups = {"idle_walk": tiles[:12], "attack": tiles[12:28], "hurt_dirs": tiles[28:]}
        for gname, gt in groups.items():
            crops = [t[74:152, 20:156] for t in gt]
            chh, cw = 78, 136
            g = np.full((math.ceil(len(crops) / 4) * chh, 4 * cw, 4), (15, 20, 31, 255), np.uint8)
            for i, c in enumerate(crops):
                r, cc = divmod(i, 4)
                m = c[..., 3] >= 128
                g[r * chh:(r + 1) * chh, cc * cw:(cc + 1) * cw][m] = c[m]
                g[r * chh:(r + 1) * chh, cc * cw] = (60, 60, 80, 255)
            Image.fromarray(g).resize((g.shape[1] * 3, g.shape[0] * 3), Image.NEAREST).save(work / f"zoom_{gname}.png")
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
            if key in state["done"] and all((od / f"{f:02d}.png").exists() for f in range(FRAMES[act])):
                continue
            od.mkdir(parents=True, exist_ok=True)
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
    motion_contact(sheets, 2).save(OUT_DIR / "preview_motion.png")
    motion_contact(sheets, 6).save(work / "preview_motion_W.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    bb = body_box(neutral)
    sprite_h = bb[3] - bb[1] + 1

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal_set)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    body_cc = {}
    for act in ("attack", "hurt"):
        for f in range(FRAMES[act]):
            img = sheets[act][6 * CELL_H:7 * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            body_cc[f"{act}:{f}"] = component_sizes(img)[:3]
    loop_end = {}
    for act in ("idle", "walk"):
        n = FRAMES[act]
        f0 = sheets[act][:CELL_H, :CELL_W]
        fl = sheets[act][:CELL_H, (n - 1) * CELL_W:n * CELL_W]
        loop_end[act] = differ_count(f0, fl)
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": int(sprite_h),
        "rest_diff_w": diff_w, "rest_diff_e": diff_e,
        "off_palette_colors": len(off), "alphas": alphas,
        "edge_touching": sum(len(v) for v in clip.values()), "edge_detail": clip,
        "last_vs_first_px": loop_end, "body_cc_W": body_cc,
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            res = subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                capture_output=True, text=True)
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
            else:
                report[f"gait_{act}"] = (res.stderr or res.stdout).strip()[-300:]
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("parts", "body_cc_W")}, ensure_ascii=False)[:1500], flush=True)

    meta = {
        "generator": "forge_lucario_actions", "version": VERSION, "key": KEY, "dex": 448,
        "variant": "lucario", "form": "normal", "archetype": "biped_aura_fighter",
        "facing_still": "left", "snout_tip_x": 36, "eye_x": [45, 48],
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(sprite_h),
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = (len(off) == 0 and report["edge_touching"] == 0 and diff_w == 0 and diff_e == 0)
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        if not ok:
            update_queue("needs_fix")
            raise SystemExit("QC failed, not published")
        publish_dir8(sprite_h)
        print("bounds", publish_bounds(sheets), flush=True)
        update_queue("done")
        print("published dir8 + bounds, queue done", flush=True)


if __name__ == "__main__":
    main()
