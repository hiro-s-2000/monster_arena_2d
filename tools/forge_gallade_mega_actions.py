#!/usr/bin/env python3
"""Mega Gallade (0475_gallade-mega) 8-direction action sheet from front.png.

Archetype: biped, still faces left. Measured on the 96x96 still (opaque 2837,
bbox x 8-87, y 11-84). Head centroid is left of center. The red mass at
x about 10-19 is the arm blade, not an eye. Chest-red flecks stay on the torso.
The right shape (x>=62) is the cape. Two legs: a left mass from y>=64, and a
right leg at x about 49-59. The white forearm beside the blade moves with it.
No second blade is drawn.

  idle   : feet planted. Blade and cape sway. Frame 0 is the still.
  walk   : the two legs step, the blade arm swings, the cape trails.
  attack : 0-4 short shout (existing chin + rings), 5-10 in-place 360 spin,
           11-12 blade impact, 13-15 recover.
  hurt   : rigid recoil of at most 38 degrees, no belly flip. 12 frames.

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.
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
MEGA_DIR = ROOT / "monsterSprite/gen4/0475_gallade/mega/gallade-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0475_gallade/mega/gallade-mega/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left (blade and head on the left, cape trailing right).
FACE_LEFT = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 192, 184, 156
VERSION = "gallade_mega_v2m"

C_RED = (217, 54, 81)
C_DRED = (128, 51, 64)
C_PINK = (255, 128, 149)
C_WHITE = (242, 242, 255)
C_INK = (16, 16, 16)
C_TEAL = (152, 217, 206)
C_GRAY = (89, 89, 128)
C_LT = (207, 207, 229)

# Closed mouth line on the 96x96 still (left-facing), hinge/back -> lip tip/front.
# The gray dash at (38-39, 36) is the existing mouth; the opening is carved from
# that row down, inside the white chin. 2026-11-04: replaces the 1-2 px chin drop.
MOUTH_LINE_STILL = [(43.0, 35.0), (40.0, 35.0), (37.0, 35.0)]


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
    """Partition this still. The left red is the blade. The right shape is the cape."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    rgb = img[..., :3]
    yy, xx = np.mgrid[0:h, 0:w]
    red = a & (
        np.all(rgb == np.array(C_RED, np.uint8), axis=-1)
        | np.all(rgb == np.array(C_DRED, np.uint8), axis=-1)
        | np.all(rgb == np.array(C_PINK, np.uint8), axis=-1)
    )
    # Left arm blade only. Chest flecks (x>24) stay off this mask.
    blade = flood(red & (xx <= 24), red & (xx <= 24))
    tip_allow = a & (xx <= 16) & (yy >= 64) & (yy <= 82)
    blade = blade | flood(a & (yy >= 76) & (xx >= 9) & (xx <= 15), tip_allow)
    rim = a & ~blade & dilate(blade, 1) & (xx <= 18) & (yy >= 22) & (yy <= 82)
    blade = blade | rim

    cape_allow = a & (xx >= 62) & (yy <= 68) & ~blade
    cape = flood(a & (xx >= 80) & (yy >= 40) & (yy <= 55), cape_allow)

    leg_r_allow = a & (yy >= 63) & (xx >= 46) & (xx <= 62) & ~blade & ~cape
    leg_r = flood(a & (yy >= 78) & (xx >= 50) & (xx <= 59), leg_r_allow)

    leg_l_allow = a & (yy >= 64) & (xx >= 16) & (xx <= 46) & ~blade & ~cape & ~leg_r
    leg_l = flood(a & (yy >= 80) & (xx >= 18) & (xx <= 28), leg_l_allow)

    head_allow = (a & ~blade & ~cape & ~leg_l & ~leg_r) & (
        ((yy <= 40) & (xx >= 26) & (xx <= 54))
        | ((yy <= 24) & (xx >= 18) & (xx <= 32))
    )
    head = flood(a & (yy <= 16) & (xx >= 32) & (xx <= 42), head_allow)
    horn = flood(a & (yy >= 15) & (yy <= 21) & (xx >= 20) & (xx <= 27), head_allow)
    head = head | horn

    # Forearm pixels that already sit on the blade. Not a second blade.
    arm_allow = (
        a & ~blade & ~head & ~cape & ~leg_l & ~leg_r
        & (xx <= 28) & (yy >= 24) & (yy <= 66)
    )
    arm = flood(dilate(blade, 1) & arm_allow, arm_allow)
    blade = blade | arm

    # Existing lower muzzle. A 1-2px drop is the shout. Not a new jaw.
    jaw = head & (yy >= 35) & (yy <= 38) & (xx >= 38) & (xx <= 46)
    head = head & ~jaw

    used = blade | cape | leg_l | leg_r | head | jaw
    torso = a & ~used
    parts = {
        "blade": blade, "cape": cape, "leg_l": leg_l, "leg_r": leg_r,
        "head": head, "jaw": jaw, "torso": torso,
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
    if int(jaw.sum()) < 8:
        raise SystemExit(f"mouth too small: {int(jaw.sum())}")
    left_red = red & (xx >= 10) & (xx <= 19)
    if int((left_red & blade).sum()) < int(0.9 * left_red.sum()):
        raise SystemExit("left red blade was not kept on the arm")
    if int((red & (xx >= 40) & blade).sum()):
        raise SystemExit("chest red was pulled onto the blade")
    return parts


def measure_facing(img, masks):
    """Blade tip is left of the head. That is the facing of this still."""
    by, bx = np.nonzero(masks["blade"])
    hy, hx = np.nonzero(masks["head"])
    if len(bx) == 0 or len(hx) == 0:
        raise SystemExit("blade or head missing")
    tip_x = int(bx.min())
    head_x = float(hx.mean())
    faces_left = tip_x < head_x - 8
    return faces_left, head_x, tip_x


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
    """4x nearest rotation. A cell stays opaque if any upscaled sample hit it."""
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
    blocks = spun.reshape(sh, 4, sw, 4, 4)
    picks = (
        blocks[:, 1, :, 1],
        blocks[:, 1, :, 2],
        blocks[:, 2, :, 1],
        blocks[:, 2, :, 2],
        blocks[:, 0, :, 1],
        blocks[:, 1, :, 0],
        blocks[:, 2, :, 2],
        blocks[:, 0, :, 2],
    )
    picked = picks[0].copy()
    empty = picked[..., 3] < 128
    for cand in picks[1:]:
        use = empty & (cand[..., 3] >= 128)
        picked[use] = cand[use]
        empty = picked[..., 3] < 128
    return picked.copy()


def content_point(x, y, angle_deg, pivot):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    vx, vy = x - pivot[0], y - pivot[1]
    return pivot[0] + c * vx - s * vy, pivot[1] + s * vx + c * vy


def despeckle(img):
    """Drop only pixels whose 4-neighborhood is empty. After rotation only."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def seal_cracks(img):
    """Fill 1px cracks (3 or 4 opaque neighbors) so a spin stays one piece."""
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
    step = 400
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None].astype(np.int32) - pts[None, :, 0]) ** 2
        d += (tx[sl, None].astype(np.int32) - pts[None, :, 1]) ** 2
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


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def roar_rings(out, origin, facing, age, strength):
    mx, my = origin
    cols = (C_WHITE, C_RED, C_PINK)
    for r_i in range(strength):
        rad = 6 + age * 2 + r_i * 4
        col = cols[min(r_i, 2)]
        for deg in range(-42, 43, 8):
            ang = math.radians(deg)
            put(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.65, col)
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
        leng = int(rng.integers(4, length + 1))
        for i in range(leng):
            put(out, rear_x - facing * (off + i), y, C_WHITE if i < 2 else C_TEAL)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(max(2, size // 2), size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), C_WHITE if r < size - 2 else C_RED)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, C_WHITE)
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
        put(out, sx, sy, C_WHITE)
        put(out, sx - facing, sy + 1, C_LT)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 80)
    out[a & ~dark, :3] = C_WHITE
    return out


def component_stats(img):
    a = img[..., 3] >= 128
    if not np.any(a):
        return 1.0, []
    seen = np.zeros_like(a)
    h, w = a.shape
    sizes = []
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
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes[0] / total, sizes[:4]


def contact_pivot(part, body, side):
    sel = part & dilate(body, 3)
    ys, xs = np.nonzero(sel if np.any(sel) else part)
    if len(ys) == 0:
        return 0.0, 0.0
    if side == "top":
        keep = ys <= ys.min() + 2
    elif side == "left":
        keep = xs <= xs.min() + 2
    elif side == "right":
        keep = xs >= xs.max() - 2
    else:
        keep = ys >= ys.max() - 2
    return float(xs[keep].mean()), float(ys[keep].mean())


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

    # Shoulder, hip, and cape socket. Hidden under the part at rest.
    for part_name, under, rad in (
        ("blade", "torso", 4),
        ("cape", "torso", 3),
        ("leg_l", "torso", 3),
        ("leg_r", "torso", 3),
    ):
        if not np.any(placed[part_name]) or not np.any(placed[under]):
            continue
        zone = placed[part_name] & dilate(placed[under], rad)
        if not np.any(zone):
            continue
        fill = nearest_color(canvas, zone, placed[under])
        write = zone & (layers[under][..., 3] < 128)
        layers[under][write] = fill[write]

    pivots = {
        "blade": contact_pivot(placed["blade"], placed["torso"], "top"),
        "cape": contact_pivot(placed["cape"], placed["torso"], "left"),
        "leg_l": contact_pivot(placed["leg_l"], placed["leg_l"], "top"),
        "leg_r": contact_pivot(placed["leg_r"], placed["leg_r"], "top"),
    }
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    hys, hxs = np.nonzero(placed["head"])
    face = (float(hxs.mean()), float(hys.mean())) if len(hxs) else center
    jys, jxs = np.nonzero(placed["jaw"])
    if len(jxs):
        mouth = (float(jxs.min()), float(jys.mean()))
    else:
        mouth = face
    bys, bxs = np.nonzero(placed["blade"])
    tip_sel = bxs <= bxs.min() + 1
    tip = (float(bxs[tip_sel].mean()), float(bys[tip_sel].mean()))
    mline = [(x + dst_x - x0, y + dst_y - y0) for x, y in MOUTH_LINE_STILL]

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
        face = (CELL_W - 1 - face[0], face[1])
        mouth = (CELL_W - 1 - mouth[0], mouth[1])
        tip = (CELL_W - 1 - tip[0], tip[1])
        mline = mouth_open.transform_points(mline, mirror_w=CELL_W)
    facing = 1.0 if flip else -1.0
    layers["_mouth_line"] = mline
    return layers, pivots, center, mouth, tip, facing


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        blade=0.0, cape=0.0, leg=0.0, jaw=0, spin=0.0, recoil=0.0,
        knock=0.0, ring=-1, rings=0, lines=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["blade"] = wave(f, n, 5.0)
        p["cape"] = wave(f, n, 7.0, phase=-0.8)
    elif action == "walk":
        p["leg"] = wave(f, n, 16.0, cycles=2.0)
        p["blade"] = wave(f, n, -12.0, cycles=2.0)
        p["cape"] = wave(f, n, 9.0, phase=-0.7, cycles=2.0)
        p["lines"] = 2
    elif action == "attack":
        # 0-4 shout, 5-10 spin, 11-12 impact, 13-15 recover.
        jaw_t = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        blade_t = [0, -8, -14, -16, -8, 0, 0, 0, 0, 0, 0, 20, 12, 6, 2, 0]
        cape_t = [0, 3, 6, 6, 3, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 0]
        p["jaw"] = jaw_t[f]
        p["blade"] = blade_t[f]
        p["cape"] = cape_t[f]
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
            p["lines"] = 3
            p["blade"] = 0.0
            p["cape"] = 0.0
            p["jaw"] = 0
        if 1 <= f <= 4:
            p["ring"], p["rings"] = f - 1, 2
        if f in (11, 12):
            p["burst"] = 1
    else:
        env = [0, 0.45, 0.8, 1.0, 0.85, 0.6, 0.38, 0.2, 0.08, 0.0, 0.0, 0][f]
        p["recoil"] = 38.0 * env
        p["knock"] = 6.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 5 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def synth(rig, action, f, palette, with_fx=True):
    layers, pivots, center, mouth, tip, facing = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    # The jaw value is now only the size of a small mouth opening along the face
    # (no chin drop, no rectangular pocket). 2 only at the roar peak.
    open_px = 0 if p["jaw"] <= 0 else (2 if (action == "attack" and f in (2, 3)) else 1)
    mline = layers["_mouth_line"]

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    # Positive blade/leg/cape is a forward swing on the unflipped still.
    sgn = -facing

    def moved(name, ang, dx, dy):
        return shift(rotate_nearest(layers[name], ang * sgn, pivots[name]), dx, dy)

    if spinning or recoiling:
        blade = layers["blade"]
        cape = layers["cape"]
        leg_l = layers["leg_l"]
        leg_r = layers["leg_r"]
        jaw = layers["jaw"]
        turned = False
    else:
        blade_ang = p["blade"]
        blade = moved("blade", blade_ang, 0, 0)
        cape = moved("cape", p["cape"], 0, 0)
        leg_l = moved("leg_l", p["leg"], 0, 0)
        leg_r = moved("leg_r", -p["leg"], 0, 0)
        jaw = layers["jaw"]
        turned = any(abs(p[k]) >= 0.45 for k in ("blade", "cape", "leg"))
        tip = content_point(tip[0], tip[1], blade_ang * sgn, pivots["blade"])
        mouth = (mouth[0], mouth[1] + p["jaw"])

    torso = layers["torso"]
    head = layers["head"]
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    # Torso first so socket cushions stay hidden until a part moves.
    for lyr in (torso, leg_r, leg_l, cape, blade, jaw, head):
        paste(body, lyr)
    if turned:
        body = despeckle(body)

    if spinning:
        body = rotate_nearest(body, p["spin"], center)
        body = seal_cracks(body)
        body = despeckle(body)
    elif recoiling:
        # Backward lean. facing<0 (left) is clockwise, never a belly mirror.
        body = rotate_nearest(body, facing * p["recoil"], center)
        body = seal_cracks(body)
        body = despeckle(body)
        body = shift(body, -facing * p["knock"], 0)

    img = snap(body, palette)
    if open_px and not (spinning or recoiling):
        img = mouth_open.open_mouth(img, mline, amount=open_px, deep=C_INK, inner=None, peak=0.6)
    if not with_fx:
        return img
    if p["rings"] and p["ring"] >= 0 and not spinning:
        img = roar_rings(img, (mouth[0] + facing * 3, mouth[1]), facing, p["ring"], p["rings"])
    if p["lines"]:
        img = speed_lines(img, facing, p["lines"], seed=f * 17 + (3 if action == "walk" else 9), length=7)
    if p["burst"]:
        img = impact_burst(img, (tip[0] + 3 * facing, tip[1]), 8)
    if p["sparks"]:
        img = sparks(img, f, facing)
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
    """Unflipped W row, so the left-facing still is what gets inspected."""
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 4, 8, 12),
        "attack": (2, 6, 8, 11),
        "hurt": (0, 3, 6, 11),
    }
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    row = DIRS.index("W")
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][row * CELL_H:(row + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            y, x = ai * CELL_H, si * CELL_W
            m = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 4
    crops = []
    row_i = DIRS.index("W")
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][row_i * CELL_H:(row_i + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
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


def touches_edge(tile):
    c = tile[..., 3] >= 128
    if not np.any(c):
        return False
    return bool(c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any())


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
    palette = extract_palette(front)
    masks = segment(front)
    faces_left, head_x, tip_x = measure_facing(front, masks)
    if not faces_left:
        raise SystemExit(f"expected blade tip left of the head, tip {tip_x} head {head_x}")
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print(f"facing left head_x={head_x:.1f} blade_tip_x={tip_x}", flush=True)
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
    placed[..., 3] = np.where(placed[..., 3] >= 128, 255, 0).astype(np.uint8)
    placed[placed[..., 3] == 0] = 0
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    diff_n = int(differ.sum())
    print("rest-pose pixels changed", diff_n, flush=True)
    if diff_n:
        raise SystemExit("idle frame 0 does not match the still")

    # Mirrored east must put the blade on the screen-right.
    e_tip = rigs["E"][4][0]
    e_mouth = rigs["E"][3][0]
    w_tip = rigs["W"][4][0]
    w_mouth = rigs["W"][3][0]
    print(f"W tip_x={w_tip:.1f} mouth_x={w_mouth:.1f}  E tip_x={e_tip:.1f} mouth_x={e_mouth:.1f}", flush=True)
    if not (w_tip < w_mouth and e_tip > e_mouth):
        raise SystemExit("facing mirror failed: E must point screen-right")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (180, 180, 190), "head": (250, 250, 250), "jaw": (230, 40, 40),
        "cape": (40, 90, 220), "blade": (40, 180, 80),
        "leg_l": (240, 200, 40), "leg_r": (240, 140, 20),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 4, 8, 12)]
        want += [("attack", "W", f) for f in (0, 2, 4, 5, 7, 8, 10, 11, 13, 15)]
        want += [("attack", "E", f) for f in (3, 7, 11)]
        want += [("hurt", "W", f) for f in (0, 1, 3, 6, 11)]
        tiles = []
        for act, d, f in want:
            bare = synth(rigs[d], act, f, palette, with_fx=False)
            tile = synth(rigs[d], act, f, palette, with_fx=True)
            frac, sizes = component_stats(bare)
            edge = touches_edge(tile)
            print(
                f"sample {act} {d} {f:02d} body={frac:.3f} sizes={sizes} "
                f"opaque={int((bare[..., 3] >= 128).sum())} edge={edge}",
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
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions_dir = MEGA_DIR / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    preview.save(actions_dir / "preview.png")
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
    alphas = sorted(int(v) for v in np.unique(master[..., 3]).tolist())
    clip = {}
    spin_frac = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                if touches_edge(tile):
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and d == "W" and 5 <= f <= 10:
                    frac, _sizes = component_stats(tile)
                    spin_frac[str(f)] = round(frac, 3)
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": alphas,
        "edge_touching": {k: v for k, v in clip.items()},
        "spin_main_frac_W": spin_frac, "facing": "left",
        "head_x": head_x, "blade_tip_x": tip_x,
        "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "hover"), ("walk", "gait")):
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
        "generator": "forge_gallade_mega_actions", "version": VERSION, "key": "0475_gallade-mega",
        "dex": 475, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "face_left": sorted(FACE_LEFT), "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "facing": "left",
        "head_x": head_x, "blade_tip_x": tip_x,
        "source": "front.png 96x96", "gifs_untouched": True,
        "mouth": "mouth_open_line_plus_rings",
        "layout": "actions_8dir", "sheetDir": "agent-sprite-forge", "raw": 1,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
