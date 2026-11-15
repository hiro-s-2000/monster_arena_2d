#!/usr/bin/env python3
"""Mega Lucario (0448_lucario-mega) 8-direction action sheet from front.png.

Archetype: biped, facing left. Opaque 1582, bbox (24,15)-(71,80).
The head sits on the left (top 22% centroid x about 42.5). The short snout
is the left muzzle and stops before the red eye. A small aura spike is the
crest above the skull. Two paws stay against the torso: the front paw ends
near x34, the rear fist starts near x62. No wings and no long tail.

The feet are already separate. Around y75 the left foot is x26-36 and the
right foot is x51-60, with a gap that must stay open. Both feet stay planted
on idle. The spin is one rigid turn of the whole still, so that gap is not
squeezed shut and no limb is invented.

The still faces left, so N NW W SW stay unflipped and S SE E NE are mirrored.
E faces screen-right. W faces screen-left.

  idle   : feet planted. Crest, short snout, and the two paws sway. Frame 0 is the still.
  walk   : the two feet step in place. The paws swing. No hop, no extra limbs.
  attack : 0-4 short-snout howl plus rings, 5-10 in-place 360 spin, 11-12 impact, 13-15 recover.
  hurt   : rigid recoil of at most 38 degrees around the feet. No belly mirror, no squash.

Every opaque pixel snaps to the still palette. Alpha is 0 or 255.
front/back png and gif are not written. This rig is not the mega-z rig.
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
import mouth_open

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen4/0448_lucario/mega/lucario-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0448_lucario/mega/lucario-mega/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left. E is the mirror, so it faces screen-right.
FACE_LEFT = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "lucario_mega_v2m"

INK = (16, 16, 16)
DARK_RED = (154, 38, 38)
RED = (222, 65, 65)
WHITE = (252, 252, 252)
CREAM = (230, 208, 131)
BLUE = (64, 124, 220)
NAVY = (23, 57, 104)

# Closed mouth line on the still (front.png coords), hinge -> lip tip. The grey
# muzzle points left; its lower lip is row 35 under the line, tip at x34.
# The opening is carved 1-2 px under this line, inside the muzzle, with both
# corners closed (mouth_open helper). The "snout" layer is the ear, not the jaw.
MOUTH_LINE_STILL = [(41.0, 34.0), (34.0, 34.0)]
# Opening per attack frame: 1 growl, 2 at the roar peak, 1 through the strike.
MOUTH_OPEN_T = [1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]


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
            q.append((int(y), int(x)))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and allow[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


def segment(img):
    """Cuts follow this still's snout, crest, two paws, and two feet."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    # Short muzzle. The red eye starts at x42, y30, and stays on the head.
    snout = flood(
        a & (xx <= 37) & (yy >= 18) & (yy <= 23),
        a & (xx <= 41) & (yy >= 17) & (yy <= 30),
    )
    eye = a & (yy >= 30) & (yy <= 34) & (xx >= 41)
    snout = snout & ~eye

    # Aura spike above the skull. It joins the head at y24.
    crest = a & (yy <= 23) & (xx >= 42)

    # Front paw stops at the black wrist, before the gold chest.
    arm_l = flood(
        a & (xx <= 28) & (yy >= 50) & (yy <= 60),
        a & (xx <= 34) & (yy >= 48) & (yy <= 63),
    )
    # Rear fist. It does not include the blue torso or the gold hip.
    arm_r = flood(
        a & (xx >= 67) & (yy >= 48) & (yy <= 54),
        a & (xx >= 62) & (yy >= 45) & (yy <= 56),
    )

    # Soles are already split. The right mask stops above the gold hip tuft.
    leg_l = flood(a & (yy >= 76) & (xx <= 36), a & (yy >= 69) & (xx <= 40))
    leg_r = flood(
        a & (yy >= 78) & (xx >= 52) & (xx <= 58),
        a & (yy >= 73) & (xx >= 50) & (xx <= 60),
    )

    used = snout | crest | arm_l | arm_r | leg_l | leg_r
    torso = a & ~used
    parts = {
        "snout": snout, "crest": crest, "arm_l": arm_l, "arm_r": arm_r,
        "leg_l": leg_l, "leg_r": leg_r, "torso": torso,
    }
    own = None
    for m in parts.values():
        own = m.copy() if own is None else (own | m)
    overlap = 0
    names = list(parts)
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            overlap += int((parts[n1] & parts[n2]).sum())
    if int(own.sum()) != int(a.sum()) or overlap:
        raise SystemExit(f"masks failed cover={int(own.sum())} opaque={int(a.sum())} overlap={overlap}")
    return parts


def measure_facing(img):
    """Snout tip is left of the red eye, so the still faces left."""
    a = img[..., 3] >= 128
    rgb = img[..., :3]
    yy = np.arange(a.shape[0])[:, None]
    eye = np.zeros_like(a)
    for col in (RED, WHITE):
        eye |= a & np.all(rgb == np.array(col, np.uint8), axis=-1)
    eye &= (yy >= 30) & (yy <= 34)
    ey, ex = np.nonzero(eye)
    if len(ex) < 3:
        raise SystemExit("expected the red eye")
    eye_x = float(np.mean(ex))
    band = a & (yy >= 18) & (yy <= 24)
    tip_x = int(np.nonzero(band)[1].min())
    faces_left = tip_x < eye_x - 4
    return faces_left, eye_x, tip_x


def edge_pivot(mask, side):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0, 0.0
    if side == "right":
        sel = xs >= xs.max() - 2
    elif side == "left":
        sel = xs <= xs.min() + 2
    elif side == "top":
        sel = ys <= ys.min() + 2
    else:
        sel = ys >= ys.max() - 2
    return float(xs[sel].mean()), float(ys[sel].mean())


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
    return out[1::4, 1::4].copy()


def despeckle(img):
    """Drop only pixels whose four neighbours are empty. Eyes and outlines stay."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    step = 300
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None] - pts[None, :, 0]) ** 2 + (tx[sl, None] - pts[None, :, 1]) ** 2
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
        out[..., 3] = 0
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


def howl_wedge(img, tip, facing, amount):
    """Short mouth wedge grown from the existing snout. It does not add a jaw."""
    if amount <= 0 or tip is None:
        return img
    tx, ty = tip
    step = 1 if facing > 0 else -1
    for i in range(1, amount + 1):
        put(img, tx + step * i, ty, DARK_RED if i < amount else INK)
        put(img, tx + step * i, ty + 1, INK)
    return img


def roar_rings(out, origin, facing, age, strength):
    mx, my = origin
    cols = (WHITE, BLUE, CREAM)
    for r_i in range(strength):
        rad = 6 + age * 2 + r_i * 4
        col = cols[min(r_i, 2)]
        for deg in range(-46, 47, 8):
            ang = math.radians(deg)
            put(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.72, col, only_empty=True)
    return out


def speed_lines(out, facing, n, seed, length):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    rear_x = (x1 - (x1 - x0) // 6) if facing < 0 else (x0 + (x1 - x0) // 6)
    y_lo = y0 + (y1 - y0) // 3
    y_hi = y0 + (y1 - y0) * 2 // 3
    for _k in range(n):
        y = int(rng.integers(y_lo, max(y_lo + 1, y_hi)))
        off = int(rng.integers(3, 7))
        length_i = int(rng.integers(4, length + 1))
        for i in range(length_i):
            put(out, rear_x - facing * (off + i), y, WHITE if i < 2 else BLUE, only_empty=True)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(max(2, size // 2), size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), WHITE if r < size - 2 else BLUE, only_empty=True)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, WHITE, only_empty=True)
    return out


def sparks(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, _y1 = bb
    d = frame_idx
    cx = (x0 + x1) // 2
    pts = [
        (cx - facing * (4 + d), y0 + 4),
        (cx + facing * 3, y0 + 10),
        (cx - facing * 2, y0 + 16),
    ]
    for sx, sy in pts:
        put(out, sx, sy, WHITE, only_empty=True)
        put(out, sx - facing, sy + 1, CREAM, only_empty=True)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 90)
    out[a & ~dark, :3] = WHITE
    return out


def largest_fraction(img):
    a = img[..., 3] >= 128
    if not np.any(a):
        return 1.0
    seen = np.zeros_like(a)
    h, w = a.shape
    best = 0
    total = int(a.sum())
    ys, xs = np.nonzero(a)
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
                if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        best = max(best, n)
    return best / total


def sole_gap(tile):
    """Largest empty run between the two soles. Toe holes are only a pixel or two."""
    a = tile[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return 0
    y1 = int(ys.max())
    y0 = max(0, y1 - 8)
    cols = a[y0:y1 + 1].any(axis=0)
    idx = np.nonzero(cols)[0]
    if len(idx) < 2:
        return 0
    return int(np.diff(idx).max() - 1)


def layer_front(layer, facing, lower=False):
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    if len(xs) == 0:
        return None
    if lower:
        cut = np.percentile(ys, 55)
        keep = ys >= cut
        if np.any(keep):
            xs, ys = xs[keep], ys[keep]
    if facing < 0:
        tip_x = int(xs.min())
        sel = xs <= tip_x + 1
    else:
        tip_x = int(xs.max())
        sel = xs >= tip_x - 1
    return int(round(float(xs[sel].mean()))), int(round(float(ys[sel].mean())))


def build_rig(img, masks, flip):
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

    cushions = (
        ("snout", "torso", 3),
        ("crest", "torso", 3),
        ("arm_l", "torso", 3),
        ("arm_r", "torso", 3),
        ("leg_l", "torso", 3),
        ("leg_r", "torso", 3),
    )
    for part_name, under, rad in cushions:
        if not np.any(placed[part_name]) or not np.any(placed[under]):
            continue
        zone = placed[part_name] & dilate(placed[under], rad)
        if not np.any(zone):
            continue
        fill = nearest_color(canvas, zone, placed[under])
        write = zone & (layers[under][..., 3] < 128)
        layers[under][write] = fill[write]

    pivots = {
        "snout": edge_pivot(placed["snout"], "right"),
        "crest": edge_pivot(placed["crest"], "bottom"),
        "arm_l": edge_pivot(placed["arm_l"], "right"),
        "arm_r": edge_pivot(placed["arm_r"], "left"),
        "leg_l": edge_pivot(placed["leg_l"], "top"),
        "leg_r": edge_pivot(placed["leg_r"], "top"),
    }
    oys, oxs = np.nonzero(a)
    center = (float(oxs.mean()) - x0 + dst_x, float(oys.mean()) - y0 + dst_y)
    feet_pts = []
    for name in ("leg_l", "leg_r"):
        fys, fxs = np.nonzero(placed[name])
        if len(fys) == 0:
            continue
        sel = fys >= fys.max() - 1
        feet_pts.append((float(fxs[sel].mean()), float(fys[sel].mean())))
    if len(feet_pts) == 2:
        feet = ((feet_pts[0][0] + feet_pts[1][0]) / 2.0, (feet_pts[0][1] + feet_pts[1][1]) / 2.0)
    else:
        feet = center

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
        feet = (CELL_W - 1 - feet[0], feet[1])
    mouth_line = [(x - x0 + dst_x, y - y0 + dst_y) for x, y in MOUTH_LINE_STILL]
    if flip:
        mouth_line = mouth_open.transform_points(mouth_line, mirror_w=CELL_W)
    facing = 1.0 if flip else -1.0
    return layers, pivots, center, feet, facing, mouth_line


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, spin=0.0, recoil=0.0,
        snout=0.0, crest=0.0, arm_l=0.0, arm_r=0.0, leg=0.0,
        mouth=0, ring=-1, rings=0, lines=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        # Feet stay down. The head does not translate.
        p["crest"] = wave(f, n, 6.0, phase=0.2)
        p["snout"] = wave(f, n, 2.5, phase=0.6)
        p["arm_l"] = wave(f, n, 3.0, phase=0.8)
        p["arm_r"] = wave(f, n, 2.0, phase=-0.5)
    elif action == "walk":
        p["dx"] = wave(f, n, 1.0, cycles=2.0)
        p["leg"] = wave(f, n, 9.0, cycles=2.0)
        p["arm_l"] = wave(f, n, -12.0, cycles=2.0)
        p["arm_r"] = wave(f, n, 8.0, cycles=2.0)
        p["crest"] = wave(f, n, 4.0, phase=-0.4, cycles=2.0)
    elif action == "attack":
        snout_t = [8, 14, 14, 10, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        mouth_t = [1, 2, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        crest_t = [3, 7, 9, 7, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        arm_l_t = [-3, -6, -8, -4, 0, 0, 0, 0, 0, 0, 0, 10, 6, 2, 0, 0]
        arm_r_t = [2, 4, 5, 3, 0, 0, 0, 0, 0, 0, 0, -4, -2, 0, 0, 0]
        p["snout"] = snout_t[f]
        p["mouth"] = mouth_t[f]
        p["crest"] = crest_t[f]
        p["arm_l"] = arm_l_t[f]
        p["arm_r"] = arm_r_t[f]
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
            p["lines"] = 3
        if 0 <= f <= 4:
            p["ring"], p["rings"] = f, 2
        if f in (11, 12):
            p["burst"] = 1
    else:
        env = [0, 0.42, 0.78, 1.0, 0.82, 0.55, 0.32, 0.16, 0.06, 0.0, 0.0, 0.0][f]
        p["dx"] = -4.0 * env
        p["recoil"] = 38.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def synth(rig, action, f, palette):
    layers, pivots, center, feet, facing, mouth_line = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)
    # The shared strike table forces "mouth" 3 on frames 5-12. That value only
    # fed the old wedge; the small face-following opening uses its own table.
    mouth_amt = MOUTH_OPEN_T[f] if action == "attack" else 0

    fwd = facing
    bdx, bdy = p["dx"] * fwd, p["dy"]
    sgn = 1.0 if fwd > 0 else -1.0
    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4

    def moved(name, ang, dx, dy):
        spun = rotate_nearest(layers[name], ang * sgn, pivots[name])
        if abs(ang) >= 0.45:
            spun = despeckle(spun)
        return shift(spun, dx, dy)

    # Feet take the body shift and no extra hop, so both soles stay on the line.
    snout = moved("snout", p["snout"], bdx, bdy)
    crest = moved("crest", p["crest"], bdx, bdy)
    arm_l = moved("arm_l", p["arm_l"], bdx, bdy)
    arm_r = moved("arm_r", p["arm_r"], bdx, bdy)
    leg_l = moved("leg_l", p["leg"], bdx, bdy)
    leg_r = moved("leg_r", -p["leg"], bdx, bdy)
    torso = shift(layers["torso"], bdx, bdy)

    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for lyr in (torso, leg_r, leg_l, arm_r, arm_l, snout, crest):
        paste(body, lyr)

    if spinning:
        # One piece. No scale, so the leg gap cannot close.
        body = despeckle(rotate_nearest(body, p["spin"], center))
    elif recoiling:
        body = despeckle(rotate_nearest(body, -p["recoil"] * fwd, feet))

    img = snap(body, palette)
    if mouth_amt and not spinning and not recoiling:
        line = mouth_open.transform_points(mouth_line, dx=bdx, dy=bdy)
        img = mouth_open.open_mouth(img, line, amount=mouth_amt, deep=INK, inner=DARK_RED, peak=0.65)
    if p["rings"] and p["ring"] >= 0 and not spinning:
        mouth = layer_front(snout, fwd, lower=True)
        if mouth is not None:
            img = roar_rings(img, (mouth[0] + fwd * (4 + p["mouth"]), mouth[1] + 1), fwd, p["ring"], p["rings"])
    if p["lines"]:
        img = speed_lines(img, fwd, p["lines"], seed=f * 19 + (1 if action == "walk" else 7), length=8)
    if p["burst"]:
        fist = layer_front(arm_l, fwd, lower=False)
        if fist is not None:
            img = impact_burst(img, (fist[0] + 6 * fwd, fist[1]), 8 if f >= 12 else 6)
    if p["sparks"]:
        img = sparks(img, f, fwd)
    if p["flash"]:
        img = flash_white(img)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (MEGA_DIR / "front.png").read_bytes()
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
    samples = {"idle": (0, 4, 8, 12), "walk": (0, 4, 8, 12), "attack": (2, 4, 7, 11), "hurt": (1, 3, 6, 9)}
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            y, x = ai * CELL_H, si * CELL_W
            m = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 4
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            # W is the unflipped left-facing pose.
            tile = sheets[act][6 * CELL_H:7 * CELL_H, f * CELL_W:(f + 1) * CELL_W]
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
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(MEGA_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    faces_left, eye_x, tip_x = measure_facing(front)
    if not faces_left:
        raise SystemExit(f"expected the snout left of the eye, tip {tip_x} eye {eye_x}")
    print(f"facing left eye_x={eye_x:.1f} snout_tip_x={tip_x}", flush=True)

    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), "sum", sum(counts.values()), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in FACE_LEFT) for d in DIRS}

    rest = synth(rigs["W"], "idle", 0, palette)
    a = front[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    dst_x = (CELL_W - (x1 - x0)) // 2
    dst_y = FOOT_Y - (y1 - y0)
    placed = np.zeros_like(rest)
    placed[dst_y:dst_y + (y1 - y0), dst_x:dst_x + (x1 - x0)] = front[y0:y1, x0:x1]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    diff_n = int(differ.sum())
    print("rest-pose pixels changed", diff_n, "sole_gap", sole_gap(rest), flush=True)
    if diff_n:
        raise SystemExit("idle frame 0 does not match the still")
    south = synth(rigs["S"], "idle", 0, palette)
    if not np.array_equal(south, rest[:, ::-1]):
        raise SystemExit("south idle 0 is not the mirror of west")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (180, 180, 190), "snout": (250, 220, 40), "crest": (230, 40, 40),
        "arm_l": (40, 180, 80), "arm_r": (40, 90, 220),
        "leg_l": (240, 140, 20), "leg_r": (240, 200, 40),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 4, 8, 12)]
        want += [("attack", "W", f) for f in (0, 2, 4, 5, 7, 8, 10, 11, 13, 15)]
        want += [("attack", "E", f) for f in (2, 7)]
        want += [("hurt", "W", f) for f in (0, 1, 3, 6, 11)]
        tiles = []
        for act, d, f in want:
            tile = synth(rigs[d], act, f, palette)
            edge = tile[..., 3] >= 128
            touch = bool(edge[0].any() or edge[-1].any() or edge[:, 0].any() or edge[:, -1].any())
            gap = sole_gap(tile) if not (act == "attack" and 5 <= f <= 9) and act != "hurt" else -1
            print(
                f"sample {act} {d} {f:02d} main={largest_fraction(tile):.3f} edge={touch} gap={gap}",
                flush=True,
            )
            tiles.append(tile)
        cols = 6
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
            if key in state["done"] and state.get("hash") == digest:
                continue
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest:
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
    contact = motion_contact(sheets)
    contact.save(work / "review_W.png")
    contact.save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = set(int(v) for v in np.unique(master[..., 3]).tolist())
    clip = {}
    spin_frac = {}
    hurt_frac = {}
    gaps = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and d == "W" and 5 <= f <= 10:
                    spin_frac[str(f)] = round(largest_fraction(tile), 3)
                if act == "hurt" and d == "W":
                    hurt_frac[str(f)] = round(largest_fraction(tile), 3)
                if act in ("idle", "walk") and d == "W":
                    gaps[f"{act}:{f}"] = sole_gap(tile)
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": sorted(alphas),
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "spin_main_frac_W": spin_frac, "hurt_main_frac_W": hurt_frac,
        "sole_gap_min": min(gaps.values()) if gaps else None,
        "facing": "left", "eye_x": eye_x, "snout_tip_x": tip_x,
        "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "idle"), ("walk", "gait")):
            od = qa / f"gait_{act}_W"
            if od.exists():
                report[f"gait_{act}"] = "skipped_existing"
                continue
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "W"),
                 "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1200], flush=True)

    meta = {
        "generator": "forge_lucario_mega_actions", "version": VERSION, "key": "0448_lucario-mega",
        "dex": 448, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "face_left": sorted(FACE_LEFT), "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "facing": "left", "eye_x": eye_x, "snout_tip_x": tip_x,
        "source": "front.png 96x96", "gifs_untouched": True, "mouth": "mouth_open_lens_and_rings",
        "layout": "actions_8dir", "an": [16, 16, 16, 12], "raw": 1,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    bad_spin = any(v < 0.9 for v in spin_frac.values())
    bad_hurt = any(v < 0.9 for v in hurt_frac.values())
    bad_gap = report["sole_gap_min"] is not None and report["sole_gap_min"] < 8
    if off or clip or alphas - {0, 255} or bad_spin or bad_hurt or bad_gap:
        raise SystemExit(
            f"QA failed off={len(off)} clip={clip} alphas={sorted(alphas)} "
            f"spin={spin_frac} hurt={hurt_frac} gap={report['sole_gap_min']}"
        )
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
