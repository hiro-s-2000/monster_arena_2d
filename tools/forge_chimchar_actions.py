#!/usr/bin/env python3
"""Chimchar (0390_chimchar) 8-direction action sheet from the measured 96x96 still.

The still is one piece, bbox x32-61 y27-71 (30x45). It faces left: both eyes sit
around x36 and x46, and the only flame (rear) is at x53-61 y53-63. Unflipped
directions are N NW W SW. S SE E NE are the horizontal mirror, so E faces
screen-right.

Parts that exist, and only those: head+torso, one hanging arm (screen-left,
y58-69), two feet (center foot y67-70, right leg y66-71), one tail flame.
No second tail, no extra arm, no invented jaw. The roar flares that flame and
opens the existing muzzle mark. The spin rotates the assembled body as one piece.

Run:
  <agent-sprite-forge>/.venv/bin/python -I tools/forge_chimchar_actions.py
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
MON_DIR = ROOT / "monsterSprite/gen4/0390_chimchar/normal"
OUT_DIR = MON_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0390_chimchar/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left (flame / rear on the right).
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "chimchar_v2"

INK = (16, 16, 16)
WHITE = (255, 255, 255)
YELLOW = (255, 213, 49)
RED = (230, 49, 49)
ORANGE = (222, 131, 0)
MOUTH = (230, 57, 65)
PINK = (238, 106, 106)


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
    """Partition every opaque pixel. Cuts follow the measured gaps, not another species."""
    h, w = img.shape[:2]
    rgb = img[..., :3]
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    red = (rgb[..., 0] == 230) & (rgb[..., 1] == 49) & (rgb[..., 2] == 49)
    yellow = (rgb[..., 0] == 255) & (rgb[..., 1] == 213) & (rgb[..., 2] == 49)
    pink = (rgb[..., 0] == 238) & (rgb[..., 1] == 106) & (rgb[..., 2] == 106)
    flame = op & (red | yellow | (pink & (xx >= 52) & (yy >= 52)))

    arm_seed = np.zeros_like(op)
    arm_seed[67, 37] = True
    arm = flood(arm_seed, op & (yy >= 58) & (xx <= 45) & ~flame)

    foot_l_seed = np.zeros_like(op)
    foot_l_seed[70, 45] = True
    foot_l = flood(foot_l_seed, op & (yy >= 67) & (xx >= 42) & (xx <= 48) & ~arm & ~flame)
    foot_r_seed = np.zeros_like(op)
    foot_r_seed[70, 55] = True
    foot_r = flood(foot_r_seed, op & (yy >= 66) & (xx >= 49) & ~arm & ~flame)

    torso = op & ~flame & ~arm & ~foot_l & ~foot_r
    # Existing muzzle mark (not the ear reds, not the tail). Anchor for the roar only.
    mouth_red = (rgb[..., 0] == 230) & (rgb[..., 1] == 57) & (rgb[..., 2] == 65)
    mouth = op & (mouth_red | pink) & (xx >= 41) & (xx <= 49) & (yy >= 43) & (yy <= 50) & torso
    parts = {"flame": flame, "arm": arm, "foot_l": foot_l, "foot_r": foot_r, "torso": torso}
    if int(sum(int(m.sum()) for m in parts.values())) != int(op.sum()):
        raise SystemExit("segmentation dropped or double-counted pixels")
    return parts, mouth


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
    """4x nearest rotation. Angle 0 and 360 return the same pixels."""
    if abs(angle_deg) < 0.05 or abs(abs(angle_deg) - 360.0) < 0.05:
        return layer.copy()
    if not np.any(layer[..., 3] >= 128):
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


def despeckle(img):
    """Drop only pixels whose 4-neighbours are all empty. Eyes and outline stay."""
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
    step = 400
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None].astype(np.int32) - pts[None, :, 0]) ** 2 + (
            tx[sl, None].astype(np.int32) - pts[None, :, 1]
        ) ** 2
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


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    sizes = []
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
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
    return sizes


def top_pivot(mask):
    ys, xs = np.nonzero(mask)
    top = int(ys.min())
    sel = ys <= top + 1
    return float(xs[sel].mean()), float(ys[sel].mean())


def base_pivot(part, body):
    contact = part & dilate(body, 2)
    if not np.any(contact):
        contact = part
    ys, xs = np.nonzero(contact)
    return float(xs.mean()), float(ys.mean())


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def socket_cushion(torso_layer, canvas, torso_mask, part_mask):
    """Body colour under the joint only, hidden while the limb is at rest."""
    contact = part_mask & dilate(torso_mask, 1)
    if not np.any(contact):
        return
    fill = nearest_color(canvas, contact, torso_mask)
    empty = torso_layer[..., 3] < 128
    write = contact & empty
    torso_layer[write] = fill[write]


def outward_step(origin, target):
    vx, vy = target[0] - origin[0], target[1] - origin[1]
    m = max(abs(vx), abs(vy), 1e-6)
    return int(round(vx / m)), int(round(vy / m))


def build_rig(img, masks, mouth, flip):
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
    mouth_m = np.zeros((CELL_H, CELL_W), bool)
    mouth_m[dst_y:dst_y + sh, dst_x:dst_x + sw] = mouth[y0:y1, x0:x1]

    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr

    socket_cushion(layers["torso"], canvas, placed["torso"], placed["arm"])
    socket_cushion(layers["torso"], canvas, placed["torso"], placed["foot_l"])
    socket_cushion(layers["torso"], canvas, placed["torso"], placed["foot_r"])

    pivots = {
        "arm": top_pivot(placed["arm"]),
        "foot_l": top_pivot(placed["foot_l"]),
        "foot_r": top_pivot(placed["foot_r"]),
        "flame": base_pivot(placed["flame"], placed["torso"]),
    }
    # From the flame's socket toward its tip. The head sits above the tail, so the
    # torso centroid points the lick down into the feet.
    step = outward_step(pivots["flame"], centroid(placed["flame"]))
    ys_all, xs_all = np.nonzero(canvas[..., 3] >= 128)
    foot_pivot = (float(xs_all.min() + xs_all.max()) / 2.0, float(ys_all.max()))

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        placed = {k: v[:, ::-1].copy() for k, v in placed.items()}
        mouth_m = mouth_m[:, ::-1].copy()
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        step = (-step[0], step[1])
        foot_pivot = (CELL_W - 1 - foot_pivot[0], foot_pivot[1])

    facing = -1.0 if not flip else 1.0
    return {
        "layers": layers,
        "pivots": pivots,
        "mouth": mouth_m,
        "step": step,
        "foot_pivot": foot_pivot,
        "facing": facing,
    }


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after an integer number of cycles."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = {
        "bdx_fwd": 0,
        "bdx_back": 0,
        "bdy": 0.0,
        "arm": 0.0,
        "foot_l": 0.0,
        "foot_r": 0.0,
        "flame_ang": 0.0,
        "flame_ext": 0,
        "spin": 0.0,
        "hurt": 0.0,
        "mouth": 0,
        "rings": 0,
        "ring_age": 0,
        "lines": 0,
        "burst": 0,
        "flash": 0,
        "sparks": 0,
    }
    if action == "idle":
        # Feet stay planted. The tail flame licks; the arm breathes.
        p["arm"] = wave(f, n, 8.0, phase=0.5)
        p["flame_ang"] = wave(f, n, 8.0, phase=-0.7)
        p["flame_ext"] = int(round(max(0.0, wave(f, n, 1.6, cycles=2.0))))
    elif action == "walk":
        # Knuckle step: the one real arm swings, the two feet alternate, flame trails.
        p["bdy"] = wave(f, n, 1.0, cycles=2.0)
        p["arm"] = wave(f, n, 18.0, cycles=2.0)
        lift_l = max(0.0, wave(f, n, 1.0, cycles=2.0))
        lift_r = max(0.0, wave(f, n, 1.0, cycles=2.0, phase=math.pi))
        p["foot_l"] = lift_l * 28.0
        p["foot_r"] = -lift_r * 28.0
        p["flame_ang"] = wave(f, n, 11.0, cycles=2.0, phase=math.pi)
        p["flame_ext"] = 1 if abs(p["flame_ang"]) > 6 else 0
    elif action == "attack":
        if f <= 4:
            p["flame_ext"] = [1, 2, 3, 2, 1][f]
            p["flame_ang"] = [4, 8, 10, 6, 2][f]
            p["mouth"] = [1, 2, 3, 2, 1][f]
            p["bdx_back"] = [0, 1, 2, 1, 0][f]
            p["rings"] = [1, 2, 3, 2, 1][f]
            p["ring_age"] = f
        elif f <= 10:
            p["spin"] = 60.0 * (f - 4)
            p["lines"] = 4
        elif f <= 12:
            p["bdx_fwd"] = 3 if f == 11 else 2
            p["burst"] = 9 if f == 11 else 11
            p["flame_ext"] = 2
            p["mouth"] = 2
        else:
            p["bdx_fwd"] = [1, 0, 0][f - 13]
            p["flame_ext"] = [1, 0, 0][f - 13]
            p["mouth"] = [1, 0, 0][f - 13]
    else:
        # Peak is sin(5π/11)*38 ≈ 37.6°, never a belly flip and never a squash.
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 38.0 * env
        p["bdx_back"] = int(round(2.0 * env))
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def flame_pose(layer, pivot, angle, step, extend):
    """One tail. The original pixels always stay; a lick only grows out from them."""
    out = layer.copy()
    if abs(angle) >= 0.05:
        turned = rotate_nearest(layer, angle, pivot)
        turned = despeckle(turned)
        grow = (turned[..., 3] >= 128) & dilate(layer[..., 3] >= 128, 1)
        add = grow & (out[..., 3] < 128)
        out[add] = turned[add]
    sx, sy = step
    for s in range(1, int(extend) + 1):
        shifted = shift(layer, sx * s, sy * s)
        add = (shifted[..., 3] >= 128) & (out[..., 3] < 128)
        out[add] = shifted[add]
    return out


def fill_cracks(img, torso_mask, limb_mask):
    """One-pixel joint cracks only, painted with the nearest torso colour."""
    if not np.any(limb_mask) or not np.any(torso_mask):
        return img
    empty = img[..., 3] < 128
    gap = empty & dilate(torso_mask, 1) & dilate(limb_mask, 1)
    if not np.any(gap):
        return img
    fill = nearest_color(img, gap, torso_mask)
    img[gap] = fill[gap]
    return img


def mask_of(layer):
    return layer[..., 3] >= 128


def open_mouth(img, mouth_mask, amount):
    """Recolour chin pixels under the existing muzzle mark. No floating wedge."""
    if amount <= 0 or not np.any(mouth_mask):
        return img
    ys, xs = np.nonzero(mouth_mask)
    bottom = int(ys.max())
    # The lowest muzzle pixel is a single column. Open across the whole mark.
    x_lo, x_hi = int(xs.min()), int(xs.max())
    h = img.shape[0]
    for i in range(1, amount + 1):
        y = bottom + i
        if y >= h - 2:
            break
        for x in range(x_lo, x_hi + 1):
            if img[y, x, 3] < 128:
                continue
            if img[y - 1, x, 3] < 128:
                continue
            edge = x == x_lo or x == x_hi or i == amount
            img[y, x, :3] = INK if edge else MOUTH
            img[y, x, 3] = 255
    return img


def roar_rings(out, center, away, age, strength):
    """Short arcs off the flame tip, not a hoop around the feet."""
    cx, cy = center
    base = math.atan2(away[1], away[0])
    cols = (YELLOW, RED, WHITE, ORANGE)
    for i in range(strength):
        rad = 3 + age * 2 + i * 3
        col = cols[i % len(cols)]
        for deg in range(-60, 61, 15):
            ang = base + math.radians(deg)
            put(out, cx + rad * math.cos(ang), cy + rad * math.sin(ang), col)
    return out


def speed_lines(out, facing, n, seed):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rear = (x0 - 3) if facing > 0 else (x1 + 3)
    rng = np.random.default_rng(seed)
    lo = y0 + 6
    hi = max(lo + 1, y1 - 6)
    for k in range(n):
        y = int(rng.integers(lo, hi))
        length = int(rng.integers(4, 8))
        for i in range(length):
            put(out, rear - facing * (2 + i), y, YELLOW if i < 2 else ORANGE)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(2, size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), WHITE if r >= size - 2 else YELLOW)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, WHITE)
    return out


def sparks(out, facing, frame_idx):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    cx = (x0 + x1) // 2
    back = -facing
    pts = [
        (cx + back * (6 + frame_idx), y0 + 4),
        (cx + back * (3 + frame_idx), y0 + 10),
        (cx + back * (8 + frame_idx // 2), y0 + 16),
    ]
    for sx, sy in pts:
        put(out, sx, sy, YELLOW)
        put(out, sx + back, sy, RED)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def moved_limb(layer, angle, pivot, dx, dy):
    img = rotate_nearest(layer, angle, pivot)
    if abs(angle) >= 0.05 and abs(abs(angle) - 360.0) >= 0.05:
        img = despeckle(img)
    return shift(img, dx, dy)


def synth(rig, action, f, palette, with_fx=True):
    layers = rig["layers"]
    pivots = rig["pivots"]
    facing = rig["facing"]
    # Authored angles are for the left-facing still. Mirror them when the body is mirrored.
    sgn = 1.0 if facing < 0 else -1.0
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    dx = int(round(p["bdx_fwd"] * facing + p["bdx_back"] * (-facing)))
    dy = int(round(p["bdy"]))
    spinning = p["spin"] > 0.05
    hurting = p["hurt"] > 0.05

    def limb(name, ang):
        return moved_limb(layers[name], ang * sgn, pivots[name], dx, dy)

    flame = flame_pose(
        layers["flame"], pivots["flame"], p["flame_ang"] * sgn, rig["step"], p["flame_ext"]
    )
    flame = shift(flame, dx, dy)
    torso = shift(layers["torso"], dx, dy)
    arm = limb("arm", p["arm"])
    foot_l = limb("foot_l", p["foot_l"])
    foot_r = limb("foot_r", p["foot_r"])

    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    # Flame sits behind the hip. Feet, then the near arm, then the head/chest.
    paste(body, flame)
    paste(body, torso)
    paste(body, foot_l)
    paste(body, foot_r)
    paste(body, arm)
    # Filling the rest-pose diagonal gaps would add pixels the still does not have.
    limb_moved = abs(p["arm"]) >= 0.05 or abs(p["foot_l"]) >= 0.05 or abs(p["foot_r"]) >= 0.05
    if limb_moved and not spinning and not hurting:
        body = fill_cracks(body, mask_of(torso), mask_of(arm))
        body = fill_cracks(body, mask_of(torso), mask_of(foot_l))
        body = fill_cracks(body, mask_of(torso), mask_of(foot_r))

    if p["flash"]:
        body = flash_white(body)
    if hurting:
        # Rigid tip around the feet. No |cos| squash.
        body = rotate_nearest(body, p["hurt"] * sgn, rig["foot_pivot"])
        body = despeckle(body)
    elif spinning:
        bb = body_box(body)
        center = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0) if bb else (CELL_W / 2, FOOT_Y - 20)
        body = rotate_nearest(body, p["spin"] * sgn, center)
        body = despeckle(body)

    if p["mouth"] and not spinning and not hurting:
        mouth = shift(rig["mouth"].astype(np.uint8), dx, dy).astype(bool)
        body = open_mouth(body, mouth, int(p["mouth"]))

    img = body
    if with_fx:
        if p["rings"]:
            bb = body_box(flame)
            if bb:
                tip_x = bb[2] if rig["step"][0] >= 0 else bb[0]
                tip_y = bb[1] if rig["step"][1] <= 0 else bb[3]
                img = roar_rings(img, (tip_x, tip_y), rig["step"], p["ring_age"], p["rings"])
        if p["lines"]:
            img = speed_lines(img, facing, p["lines"], seed=f * 17 + int(facing > 0))
        if p["burst"]:
            bb = body_box(body)
            if bb:
                front_x = (bb[2] + 4) if facing > 0 else (bb[0] - 4)
                front_y = bb[1] + max(8, (bb[3] - bb[1]) // 3)
                img = impact_burst(img, (front_x, front_y), int(p["burst"]))
        if p["sparks"]:
            img = sparks(img, facing, f)
        img = snap(img, palette)
    else:
        img = snap(img, palette)
    return img


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (MON_DIR / "front.png").read_bytes()
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
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 4, 8, 12),
        "attack": (2, 7, 11, 14),
        "hurt": (1, 4, 6, 10),
    }
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
    pad = 10
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][2 * CELL_H:3 * CELL_H, f * CELL_W:(f + 1) * CELL_W]
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


def write_zoom(rigs, palette, rows, path, scale=4):
    """Cropped, nearest-scaled contact sheet so the 30px body can be judged by eye."""
    pad = 14
    crops = []
    for row in rows:
        line = []
        for act, d, f in row:
            tile = synth(rigs[d], act, f, palette)
            a = tile[..., 3] >= 128
            if not np.any(a):
                line.append(np.zeros((8, 8, 4), np.uint8))
                continue
            ys, xs = np.nonzero(a)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(CELL_H, int(ys.max()) + 1 + pad)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(CELL_W, int(xs.max()) + 1 + pad)
            line.append(tile[y0:y1, x0:x1])
        crops.append(line)
    ch = max(c.shape[0] for line in crops for c in line)
    cw = max(c.shape[1] for line in crops for c in line)
    cols = max(len(line) for line in crops)
    bg = np.full((len(crops) * ch, cols * cw, 4), (20, 24, 36, 255), np.uint8)
    for ri, line in enumerate(crops):
        for ci, c in enumerate(line):
            y = ri * ch + (ch - c.shape[0]) // 2
            x = ci * cw + (cw - c.shape[1]) // 2
            m = c[..., 3] >= 128
            bg[y:y + c.shape[0], x:x + c.shape[1]][m] = c[m]
    up = np.repeat(np.repeat(bg, scale, 0), scale, 1)
    Image.fromarray(up, "RGBA").save(path)


def rest_reference(front, flip):
    a = front[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    dst_x = (CELL_W - (x1 - x0)) // 2
    dst_y = FOOT_Y - (y1 - y0)
    placed = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    placed[dst_y:dst_y + (y1 - y0), dst_x:dst_x + (x1 - x0)] = front[y0:y1, x0:x1]
    if flip:
        placed = placed[:, ::-1].copy()
    return placed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(MON_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks, mouth = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    counts["mouth"] = int(mouth.sum())
    print("parts", counts, "palette", len(palette), flush=True)

    rigs = {}
    for d in DIRS:
        rigs[d] = build_rig(front, masks, mouth, flip=d not in UNFLIPPED)

    for d, flip in (("W", False), ("S", True)):
        rest = synth(rigs[d], "idle", 0, palette)
        ref = rest_reference(front, flip)
        differ = np.any(rest != ref, axis=2) & ((rest[..., 3] >= 128) | (ref[..., 3] >= 128))
        print(f"rest-pose {d} pixels changed", int(differ.sum()), flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (210, 190, 60),
        "arm": (40, 180, 80),
        "foot_l": (40, 120, 220),
        "foot_r": (220, 120, 40),
        "flame": (220, 40, 40),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    part[mouth, :3] = (255, 255, 255)
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10)]
        want += [("attack", "W", f) for f in range(16)]
        want += [("hurt", "W", f) for f in range(12)]
        want += [("attack", "E", f) for f in (0, 3, 7, 11)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        tw, th = CELL_W, CELL_H
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * th, cols * tw, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        zoom_rows = [
            [("idle", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)],
            [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)],
            [("attack", "W", f) for f in range(8)],
            [("attack", "W", f) for f in range(8, 16)],
            [("hurt", "W", f) for f in range(8)],
            [("hurt", "W", f) for f in range(8, 12)] + [("idle", "E", 0), ("attack", "E", 0), ("attack", "E", 7), ("attack", "E", 11)],
        ]
        write_zoom(rigs, palette, zoom_rows, work / "zoom.png")
        # Body connectivity on the spin, before effects.
        for f in range(5, 11):
            body = synth(rigs["W"], "attack", f, palette, with_fx=False)
            sizes = components(body[..., 3] >= 128)
            print(f"spin W f{f} components", sizes, flush=True)
        for f in (0, 3, 5, 6):
            body = synth(rigs["W"], "hurt", f, palette, with_fx=False)
            ang = pose("hurt", f)["hurt"]
            sizes = components(body[..., 3] >= 128)
            print(f"hurt W f{f} ang {ang:.2f} components", sizes, flush=True)
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
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions_dir = OUT_DIR / "actions"
    actions_dir.mkdir(exist_ok=True)
    preview.save(actions_dir / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = [int(v) for v in np.unique(master[..., 3])]
    clip = {}
    spin_parts = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and 5 <= f <= 10 and d == "W":
                    spin_parts[str(f)] = components(c)
    report = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_h,
        "off_palette_colors": len(off),
        "alpha_values": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "spin_components_W": spin_parts,
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "hover"), ("walk", "hover")):
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
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1200], flush=True)

    meta = {
        "generator": "forge_chimchar_actions",
        "version": VERSION,
        "key": "0390_chimchar",
        "dex": 390,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED),
        "facing": "left",
        "archetype": "biped_knuckle",
        "palette_colors": int(len(palette)),
        "off_palette_colors": len(off),
        "source": "front.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
