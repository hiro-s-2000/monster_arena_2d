#!/usr/bin/env python3
"""Mantyke (0458) 8-direction action sheet from this folder's 96x96 front.png.

Measured on this still only (opaque 1351, bbox x21-75 y25-71, 55x47, 11 colours,
alpha 0/255). Front view. Two cephalic fins (left lobe x27-36 y25-40, right lobe
x40-59 y26-40). Left eye is a wink (ink x30-37 y49-57); the open eye with white
sits at x43-48 y51-56. The smile is eight existing pixels around (49-53, 61-64).
Pectoral wings are the side membranes (left x<=32, right x>=63). The tail tip is
y>=68. There are no feet and no legs. Wings are not feet.

Parts, from this still only:
  left_ant / right_ant   cephalic fins. Not legs.
  left_wing / right_wing pectoral fins. Not feet.
  tail                   bottom tip. Not a foot.
  mouth                  the eight-pixel smile. Not a new jaw.
  torso                  head, eyes, belly, back. Everything else.

S SE E NE keep the still. N NW W SW are mirrored.
idle  : 1px hover. Frame 0 is the still. Wings and fins sway.
walk  : in-place wing flap, two beats, no travel, no invented feet.
attack: 0-4 wing and smile display plus rings, 5-10 one-piece in-plane
        360 spin (no |cos| squash), 11-12 impact, 13-15 recover.
hurt  : one-piece lean, at most 36 degrees.

Palette-locked. Alpha is only 0 or 255. front/back png and gif are not written.
"""

import argparse
import fcntl
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
STILL_DIR = ROOT / "monsterSprite/gen4/0458_mantyke/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0458_mantyke/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 147
VERSION = "mantyke_v2"
KEY = "0458_mantyke"
# Front-facing still. West-side rows are the mirror.
FLIP_DIRS = {"N", "NW", "W", "SW"}
MAX_RECOIL = 36.0

INK = (0, 0, 0)
DEEP = (16, 41, 82)
NAVY = (16, 57, 123)
BLUE = (16, 82, 172)
IRIS = (57, 115, 197)
SLATE = (74, 98, 115)
SKY = (98, 156, 213)
GRAY = (123, 148, 164)
PALE = (156, 180, 222)
BELLY = (180, 222, 255)
WHITE = (255, 255, 255)

SHEET_FILES = (
    "sheet.png",
    "preview.png",
    "preview_motion.png",
    "meta.json",
    "idle.png",
    "walk.png",
    "attack.png",
    "hurt.png",
    "idle_8dir.png",
    "walk_8dir.png",
    "attack_8dir.png",
    "hurt_8dir.png",
    "idle_S_strip.png",
    "walk_S_strip.png",
    "attack_S_strip.png",
    "hurt_S_strip.png",
)


def is_color(img, color):
    rgb = img[..., :3]
    return (img[..., 3] >= 128) & np.all(rgb == np.array(color, np.uint8), axis=-1)


def segment(img):
    """Geometric parts of this front-facing still. Masks partition every opaque pixel."""
    op = img[..., 3] >= 128
    h, w = op.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ink = is_color(img, INK)
    shade = is_color(img, GRAY) | is_color(img, SLATE)
    # Smile only. Flood the ink and shade that already touch the seed. Do not grow a jaw.
    allow = (ink | shade) & (yy >= 58) & (yy <= 66) & (xx >= 40) & (xx <= 57)
    mouth = np.zeros_like(op)
    if not allow[62, 52]:
        raise SystemExit("smile seed is missing")
    stack = [(62, 52)]
    mouth[62, 52] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and allow[ny, nx] and not mouth[ny, nx]:
                mouth[ny, nx] = True
                stack.append((ny, nx))
    if not (6 <= int(mouth.sum()) <= 14):
        ys, xs = np.nonzero(mouth)
        raise SystemExit(f"smile size {int(mouth.sum())} {list(zip(xs.tolist(), ys.tolist()))}")

    # Cephalic fins. Separate lobes above the head. Not legs.
    left_ant = op & (yy <= 40) & (xx <= 36)
    right_ant = op & (yy <= 40) & (xx >= 40)
    # Eyes stay on the torso so a wing flap does not tear the face.
    left_eye = op & (xx >= 30) & (xx <= 38) & (yy >= 48) & (yy <= 58)
    right_eye = op & (xx >= 42) & (xx <= 50) & (yy >= 49) & (yy <= 58)
    # Pectoral membranes outside the face. Not feet.
    left_wing = op & (yy >= 50) & (xx <= 32) & ~left_eye & ~mouth
    right_wing = op & (yy >= 50) & (yy <= 66) & (xx >= 63) & ~right_eye & ~mouth
    # Tail tip below the belly. Not a foot.
    tail = op & (yy >= 68)
    names = {
        "left_ant": left_ant,
        "right_ant": right_ant,
        "mouth": mouth,
        "left_wing": left_wing,
        "right_wing": right_wing,
        "tail": tail,
    }
    used = np.zeros_like(op)
    for mask in names.values():
        if np.any(used & mask):
            raise SystemExit("mantyke parts overlap")
        used |= mask
    torso = op & ~used
    names["torso"] = torso
    if int(used.sum() + torso.sum()) != int(op.sum()):
        raise SystemExit("mantyke parts do not partition the still")
    if int(left_wing.sum()) < 40 or int(right_wing.sum()) < 40:
        raise SystemExit("wings collapsed")
    return names


def measure_face(img, masks):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    return {
        "opaque": int(op.sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "parts": {k: int(v.sum()) for k, v in masks.items()},
        "facing": "front",
        "feet": 0,
        "mouth": int(masks["mouth"].sum()),
        "eyes": int(is_color(img, WHITE).sum()),
    }


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


def _resample(layer, src_x, src_y):
    h, w = layer.shape[:2]
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(layer)
    vy, vx = np.nonzero(valid)
    out[vy, vx] = layer[src_y[vy, vx], src_x[vy, vx]]
    return out


def _xform_nearest(layer, pivot, map_xy):
    """4x nearest resample. map_xy(rx, ry) -> source in the upscaled grid."""
    if not np.any(layer[..., 3] >= 128):
        return layer.copy()
    scale = 4
    up = np.repeat(np.repeat(layer, scale, axis=0), scale, axis=1)
    px, py = pivot[0] * scale + 1.5, pivot[1] * scale + 1.5
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    src_x, src_y = map_xy(xx - px, yy - py, px, py)
    src_x = np.rint(src_x).astype(np.int32)
    src_y = np.rint(src_y).astype(np.int32)
    out = _resample(up, src_x, src_y)
    return out[scale // 2::scale, scale // 2::scale].copy()


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest in-plane rotate. A multiple of 360 stays exact. No squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    rad = math.radians(ang)
    c, s = math.cos(rad), math.sin(rad)

    def map_xy(rx, ry, px, py):
        # Inverse of a pure rotation. Do not scale by abs(cos).
        return px + c * rx + s * ry, py - s * rx + c * ry

    return _xform_nearest(layer, pivot, map_xy)


def despeckle(img):
    """Drop only pixels whose four neighbours are all empty. After a warp only."""
    out = img.copy()
    a = out[..., 3] >= 128
    pad = np.pad(a, 1, constant_values=False)
    n = (
        pad[:-2, 1:-1].astype(np.int16)
        + pad[2:, 1:-1]
        + pad[1:-1, :-2]
        + pad[1:-1, 2:]
    )
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img):
    """Fill a transparent pixel only when all four neighbours are opaque."""
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        pad = np.pad(a, 1, constant_values=False)
        n = (
            pad[:-2, 1:-1].astype(np.int16)
            + pad[2:, 1:-1]
            + pad[1:-1, :-2]
            + pad[1:-1, 2:]
        )
        hole = (~a) & (n == 4)
        if not np.any(hole):
            break
        up = np.roll(out, 1, axis=0)
        out[hole] = up[hole]
        out[..., 3] = np.where(out[..., 3] >= 128, 255, 0)
    return out


def extract_palette(img):
    opaque = img[..., 3] >= 128
    return np.unique(img[opaque][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    opaque = out[..., 3] >= 128
    if not np.any(opaque):
        out[...] = 0
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = opaque.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    mapped = palette[dist.argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)
    out[~opaque] = 0
    return out


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1, constant_values=False)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def hinge(part, body):
    """Mean of the part pixels that touch the body. Falls back to the part centre."""
    touch = part & dilate(body, 2)
    if not np.any(touch):
        touch = part
    ys, xs = np.nonzero(touch)
    return float(xs.mean()), float(ys.mean())


def _belly_fill(canvas, mouth_m):
    """Belly colour under the smile, so a 1px drop does not leave a hole."""
    out = np.zeros_like(canvas)
    ys, xs = np.nonzero(mouth_m)
    for y, x in zip(ys.tolist(), xs.tolist()):
        placed = False
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < canvas.shape[0]
                and 0 <= nx < canvas.shape[1]
                and canvas[ny, nx, 3] >= 128
                and not mouth_m[ny, nx]
            ):
                out[y, x] = canvas[ny, nx]
                placed = True
                break
        if not placed:
            out[y, x, :3] = BELLY
            out[y, x, 3] = 255
    return out


def build_rig(img, masks, flip):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - (sh - 1)
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]

    placed = {}
    for name, mask in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = mask[y0:y1, x0:x1]
        placed[name] = pm

    if flip:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}

    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr

    # One-pixel root under each fin. Covered at rest, so frame 0 matches.
    torso_m = placed["torso"]
    for name in ("left_ant", "right_ant", "left_wing", "right_wing", "tail"):
        touch = placed[name] & dilate(torso_m, 1)
        if np.any(touch):
            layers["torso"][touch] = layers[name][touch]
    # Smile covers this fill at rest. Moving it reveals belly, not a new mouth.
    layers["torso"][placed["mouth"]] = _belly_fill(canvas, placed["mouth"])[placed["mouth"]]

    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    facing = -1.0 if flip else 1.0
    pivots = {
        "left_ant": hinge(placed["left_ant"], placed["torso"]),
        "right_ant": hinge(placed["right_ant"], placed["torso"]),
        "left_wing": hinge(placed["left_wing"], placed["torso"]),
        "right_wing": hinge(placed["right_wing"], placed["torso"]),
        "tail": hinge(placed["tail"], placed["torso"]),
        "body": center,
    }
    return layers, pivots, center, facing, canvas


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Sine that is exactly 0 on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0,
        left_wing=0.0, right_wing=0.0,
        left_ant=0.0, right_ant=0.0,
        tail=0.0, mouth=0.0,
        spin=0.0, recoil=0.0,
        ring=-1, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        # Hover. Both wing tips rise together. Frame 0 is the still.
        p["dy"] = wave(f, n, -1.0)
        p["left_wing"] = wave(f, n, 10.0, phase=0.3)
        p["right_wing"] = wave(f, n, -10.0, phase=0.3)
        p["left_ant"] = wave(f, n, 7.0, phase=0.9)
        p["right_ant"] = wave(f, n, -7.0, phase=0.9)
        p["tail"] = wave(f, n, 6.0, phase=1.5)
    elif action == "walk":
        # In-place flap. Two beats. The body does not travel and grows no feet.
        p["dy"] = wave(f, n, -1.0, cycles=2.0)
        p["left_wing"] = wave(f, n, 22.0, cycles=2.0)
        p["right_wing"] = wave(f, n, -22.0, cycles=2.0)
        p["left_ant"] = wave(f, n, 12.0, cycles=2.0, phase=0.45)
        p["right_ant"] = wave(f, n, -12.0, cycles=2.0, phase=0.45)
        p["tail"] = wave(f, n, 9.0, cycles=2.0, phase=0.9)
    elif action == "attack":
        # 0-4 wing and smile display, 5-10 spin, 11-12 impact, 13-15 recover.
        p["left_wing"] = [0, 12, 22, 30, 16, 0, 0, 0, 0, 0, 0, 0, 0, 10, 4, 0][f]
        p["right_wing"] = [0, -12, -22, -30, -16, 0, 0, 0, 0, 0, 0, 0, 0, -10, -4, 0][f]
        p["left_ant"] = [0, 8, 14, 18, 10, 0, 0, 0, 0, 0, 0, 0, 0, 6, 2, 0][f]
        p["right_ant"] = [0, -8, -14, -18, -10, 0, 0, 0, 0, 0, 0, 0, 0, -6, -2, 0][f]
        p["tail"] = [0, -6, -10, -12, -6, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 0][f]
        p["mouth"] = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["dy"] = [0, -1, -2, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        if 1 <= f <= 4:
            p["ring"] = f - 1
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
        if f in (11, 12):
            p["burst"] = 2 if f == 11 else 1
    else:
        env = [0.0, 0.42, 0.78, 1.0, 0.80, 0.55, 0.32, 0.16, 0.05, 0.0, 0.0, 0.0][f]
        p["recoil"] = MAX_RECOIL * env
        p["dx"] = -2.0 * env
        p["dy"] = -1.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def moved_part(layer, angle, pivot, dx, dy):
    if abs(angle) >= 0.45:
        layer = rotate_nearest(layer, angle, pivot)
    return shift(layer, dx, dy)


def compose(layers, pivots, p, facing):
    """Fins move on their own hinges. Nothing here is a foot."""
    bdx = p["dx"] * facing
    bdy = p["dy"]
    torso = shift(layers["torso"], bdx, bdy)
    tail = moved_part(layers["tail"], p["tail"] * facing, pivots["tail"], bdx, bdy)
    left_wing = moved_part(layers["left_wing"], p["left_wing"] * facing, pivots["left_wing"], bdx, bdy)
    right_wing = moved_part(layers["right_wing"], p["right_wing"] * facing, pivots["right_wing"], bdx, bdy)
    left_ant = moved_part(layers["left_ant"], p["left_ant"] * facing, pivots["left_ant"], bdx, bdy)
    right_ant = moved_part(layers["right_ant"], p["right_ant"] * facing, pivots["right_ant"], bdx, bdy)
    mouth = shift(layers["mouth"], bdx, bdy + p["mouth"])
    body = np.zeros_like(torso)
    for lyr in (torso, tail, left_wing, right_wing, left_ant, right_ant, mouth):
        paste(body, lyr)
    return body


def front_point(img, facing):
    """Just outside the facing silhouette, at smile height. Not a new mouth."""
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    if len(xs) == 0:
        return CELL_W // 2, CELL_H // 2
    y0, y1 = int(ys.min()), int(ys.max())
    smile_y = y0 + int(0.70 * (y1 - y0))
    if facing > 0:
        return float(xs.max()) + 5.0, float(smile_y)
    return float(xs.min()) - 5.0, float(smile_y)


def fin_rings(img, origin, facing, age):
    """Sparse rings outside the wing. They are not part of the body."""
    ox, oy = origin
    cols = (WHITE, BELLY, SKY, BLUE)
    for ri in range(age + 1):
        rad = 3 + ri * 3
        col = cols[ri % 4]
        for deg in range(-60, 61, 30):
            ang = math.radians(deg)
            put(img, ox + facing * rad * math.cos(ang), oy + rad * math.sin(ang), col)
    return img


def impact_burst(img, origin, facing, size):
    ox, oy = origin
    pts = [(2, 0), (6, -3), (6, 3), (9, -1), (9, 1), (12, 0), (4, -6), (4, 6)]
    cols = (WHITE, BELLY, SKY, BLUE)
    n = 4 + size * 2
    for i, (dx, dy) in enumerate(pts[:n]):
        put(img, ox + facing * dx, oy + dy, cols[i % 4])
    return img


def sparks(img, frame_idx, facing, center):
    cx, cy = center
    d = frame_idx
    pts = (
        (cx - facing * (8 + d), cy - 16),
        (cx + facing * (6 + d), cy - 18),
        (cx - facing * 2, cy - 20 - d),
    )
    for sx, sy in pts:
        put(img, sx, sy, WHITE)
        put(img, sx + facing, sy, BELLY)
    return img


def flash_hit(img):
    """One-frame blink. Outline, eyes, and the smile stay. Body goes pale."""
    out = img.copy()
    opaque = out[..., 3] >= 128
    rgb = out[..., :3]
    keep = np.zeros(opaque.shape, bool)
    for color in (INK, WHITE, IRIS, GRAY, SLATE):
        keep |= opaque & np.all(rgb == np.array(color, np.uint8), axis=-1)
    out[opaque & ~keep, :3] = np.array(BELLY, np.uint8)
    return out


def largest_fraction(img):
    opaque = img[..., 3] >= 128
    if not np.any(opaque):
        return 1.0, 0
    seen = np.zeros_like(opaque, dtype=bool)
    h, w = opaque.shape
    best = 0
    total = int(opaque.sum())
    ys, xs = np.nonzero(opaque)
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
                if 0 <= ny < h and 0 <= nx < w and opaque[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if n > best:
            best = n
    return best / total, total


def synth(rig, action, f, palette, effects=True):
    layers, pivots, center, facing, canvas = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        # One piece. Spin or lean the still itself. Do not squash with |cos|.
        body = canvas.copy()
        if recoiling:
            body = shift(body, p["dx"] * facing, p["dy"])
            pivot = (center[0] + p["dx"] * facing, center[1] + p["dy"])
            body = rotate_nearest(body, -p["recoil"] * facing, pivot)
            body = fill_pinholes(body)
            body = despeckle(body)
        else:
            ang = p["spin"] * facing
            turned = abs(ang % 360.0) > 0.45 and abs((ang % 360.0) - 360.0) > 0.45
            if turned:
                body = rotate_nearest(body, ang, center)
                body = fill_pinholes(body)
                body = despeckle(body)
    else:
        body = compose(layers, pivots, p, facing)
        moved = (
            abs(p["left_wing"]) > 0.45
            or abs(p["right_wing"]) > 0.45
            or abs(p["left_ant"]) > 0.45
            or abs(p["right_ant"]) > 0.45
            or abs(p["tail"]) > 0.45
        )
        if moved:
            body = fill_pinholes(body)
            body = despeckle(body)
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_hit(img)
    if p["ring"] >= 0 and not spinning:
        img = fin_rings(img, front_point(img, facing), facing, p["ring"])
    if p["burst"]:
        img = impact_burst(img, front_point(img, facing), facing, p["burst"])
    if p["sparks"]:
        img = sparks(img, f, facing, center)
    return snap(img, palette)


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
    return master


def preview_image(sheets):
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (2, 6, 10, 14),
        "attack": (3, 6, 8, 11),
        "hurt": (1, 3, 6, 10),
    }
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            x = si * CELL_W
            y = ai * CELL_H
            m = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    """South row, every frame, cropped. Wings, smile, spin, and the lean."""
    pad = 6
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
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
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    return n


def _entry_span(text, key):
    token = f'"{key}":'
    start = text.index(token)
    i = start + len(token)
    while text[i] != "{":
        i += 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return start, j + 1
    raise SystemExit(f"dir8 entry did not close: {key}")


def publish_dir8(sprite_h):
    entry = {
        "dex": 458,
        "form": "normal",
        "variant": "mantyke",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0458_mantyke/normal",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir",
        "nf": 16,
        "cw": CELL_W,
        "ch": CELL_H,
        "w": CELL_W,
        "h": int(sprite_h),
        "fy": int(FOOT_Y),
        "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1,
        "src": "agent-sprite-forge",
        "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    DIR8_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with DIR8_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            text = DIR8.read_text()
            prefix = "window.ARENA_DIR8 = "
            if not text.startswith(prefix):
                raise SystemExit("dir8.js header changed; not publishing")
            if "\n" in text.strip():
                raise SystemExit("dir8.js is no longer one line")
            keys = re.findall(r'"(\d{4}_[^"]+)":\{', text)
            if keys.count(KEY) > 1:
                raise SystemExit("0458_mantyke key duplicated")
            if keys.count(KEY) != 1:
                raise SystemExit("0458_mantyke missing from dir8")
            idx = keys.index(KEY)
            prev_key = keys[idx - 1] if idx > 0 else None
            next_key = keys[idx + 1] if idx + 1 < len(keys) else None
            prev_blob = text[_entry_span(text, prev_key)[0]:_entry_span(text, prev_key)[1]] if prev_key else None
            next_blob = text[_entry_span(text, next_key)[0]:_entry_span(text, next_key)[1]] if next_key else None
            s, e = _entry_span(text, KEY)
            new_text = text[:s] + f'"{KEY}":' + blob + text[e:]
            if prev_blob and prev_blob not in new_text:
                raise SystemExit("neighbor dir8 key changed")
            if next_blob and next_blob not in new_text:
                raise SystemExit("neighbor dir8 key changed")
            body = new_text[len(prefix):].strip()
            if body.endswith(";"):
                body = body[:-1].strip()
            data = json.loads(body)
            if data.get(KEY) != entry:
                raise SystemExit("dir8 entry did not round-trip")
            if new_text.count(f'"v":"{VERSION}"') != 1:
                raise SystemExit("dir8 version stamp collided")
            if "\n" in new_text.strip():
                raise SystemExit("dir8.js would no longer be one line")
            if text.endswith("\n") and not new_text.endswith("\n"):
                new_text += "\n"
            if (not text.endswith("\n")) and new_text.endswith("\n"):
                new_text = new_text[:-1]
            DIR8.write_text(new_text)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def update_queue(status):
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            text = QUEUE.read_text()
            data = json.loads(text)
            before = {item.get("id"): (item.get("status"), item.get("claim")) for item in data["items"]}
            if status in ("done", "needs_fix"):
                st, claim = before.get(KEY, (None, None))
                if not (st == "in_progress" and claim == "parallel-batch"):
                    raise SystemExit(f"queue claim is not parallel-batch: {before.get(KEY)}")
            start = text.index('"id": "0458_mantyke"')
            nxt = text.find('"id":', start + 10)
            end = len(text) if nxt < 0 else nxt
            block = text[start:end]
            if '"status":' not in block:
                raise SystemExit("queue status missing")
            block2 = re.sub(r'"status": "[^"]*"', f'"status": "{status}"', block, count=1)
            mid, n_mid = re.subn(r',\n\s*"claim": "[^"]*"', "", block2, count=1)
            if not n_mid:
                mid, n_mid = re.subn(r'\n\s*"claim": "[^"]*",', "", block2, count=1)
            if n_mid:
                block2 = mid
            text = text[:start] + block2 + text[end:]
            data = json.loads(text)
            ids = [item.get("id") for item in data["items"]]
            if ids.count(KEY) != 1:
                raise SystemExit("queue id collision")
            for item in data["items"]:
                if item.get("id") == KEY:
                    if item.get("status") != status or "claim" in item:
                        raise SystemExit("queue update did not stick")
                elif (item.get("status"), item.get("claim")) != before.get(item.get("id")):
                    raise SystemExit(f"queue neighbor changed: {item.get('id')}")
            if not text.endswith("\n"):
                text += "\n"
            QUEUE.write_text(text)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def assert_claim():
    import os as _rebuild_os
    if _rebuild_os.environ.get("GEN4_ATTACK_REBUILD") == "1":
        return
    data = json.loads(QUEUE.read_text())
    for item in data["items"]:
        if item.get("id") == KEY:
            if item.get("status") != "in_progress" or item.get("claim") != "parallel-batch":
                raise SystemExit(f"stop without drawing: {item.get('status')} {item.get('claim')}")
            return
    raise SystemExit("0458_mantyke missing from queue")


def attack_report(rig, palette):
    base_img = synth(rig, "attack", 0, palette)
    base = int((base_img[..., 3] == 255).sum())
    print("ATTACK_S_RATIOS", flush=True)
    ratios = []
    for f in range(16):
        tile = synth(rig, "attack", f, palette)
        n = int((tile[..., 3] == 255).sum())
        frac, _total = largest_fraction(tile)
        ratio = n / base if base else 0.0
        ratios.append(ratio)
        edge = tile[..., 3]
        touch = bool(edge[0].any() or edge[-1].any() or edge[:, 0].any() or edge[:, -1].any())
        print(
            f"attack S {f:02d} opaque {n} ratio {ratio:.3f} main {frac:.3f} edge {touch}",
            flush=True,
        )
    print(f"min {min(ratios):.3f} max {max(ratios):.3f}", flush=True)
    return ratios


def qa_sheet(master, sheets, palette, frames_dir, digest, sprite_h, rest_diff, skip_qa):
    ratios_counts = []
    row = sheets["attack"][:CELL_H]
    for f in range(16):
        tile = row[:, f * CELL_W:(f + 1) * CELL_W]
        ratios_counts.append(int((tile[..., 3] == 255).sum()))
    base = ratios_counts[0]
    ratios = [c / base for c in ratios_counts]
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(ratios_counts, ratios)):
        print(f"attack S {f:02d} opaque {c} ratio {r:.3f}", flush=True)
    print(f"min {min(ratios):.3f} max {max(ratios):.3f}", flush=True)

    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    partial = int(((master[..., 3] != 0) & (master[..., 3] != 255)).sum())
    clip = []
    spin_stats = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = tile[..., 3] == 255
                if a.size and (a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any()):
                    clip.append(f"{act}:{d}:{f}")
                if act == "attack" and 5 <= f <= 10:
                    frac, total_px = largest_fraction(tile)
                    spin_stats.append({"d": d, "f": f, "opaque": total_px, "main": round(frac, 3)})
    report = {
        "version": VERSION,
        "hash": digest,
        "sprite_h": sprite_h,
        "cell": [CELL_W, CELL_H],
        "foot_y": FOOT_Y,
        "feet": 0,
        "mouth": 8,
        "rest_diff": rest_diff,
        "off_palette_colors": len(off),
        "partial_alpha": partial,
        "edge_touching": clip,
        "attack_s_ratios": [round(r, 3) for r in ratios],
        "attack_s_min": round(min(ratios), 3),
        "attack_s_max": round(max(ratios), 3),
        "spin": [s for s in spin_stats if s["d"] == "S"],
    }
    if not skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        qa = OUT_DIR / "_work" / "qa"
        qa.mkdir(parents=True, exist_ok=True)
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
    ok = (
        rest_diff == 0
        and not off
        and partial == 0
        and not clip
        and all(0.90 <= r <= 1.08 for r in ratios)
        and all(s["main"] >= 0.90 for s in spin_stats)
    )
    report["ok"] = ok
    return report, ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    assert_claim()
    front = np.array(Image.open(STILL_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    partial_src = int(((front[..., 3] != 0) & (front[..., 3] != 255)).sum())
    if partial_src:
        raise SystemExit(f"still has partial alpha {partial_src}")
    palette = extract_palette(front)
    masks = segment(front)
    info = measure_face(front, masks)
    print(
        "parts", info["parts"], "palette", len(palette),
        "facing", info["facing"], "feet", info["feet"], "mouth", info["mouth"],
        flush=True,
    )

    rigs = {}
    for d in DIRS:
        rigs[d] = build_rig(front, masks, flip=d in FLIP_DIRS)

    rest = synth(rigs["E"], "idle", 0, palette, effects=False)
    canvas = rigs["E"][4]
    rest_diff = int(np.any(rest != canvas, axis=2).sum())
    rest_s = synth(rigs["S"], "idle", 0, palette, effects=False)
    rest_s_diff = int(np.any(rest_s != rigs["S"][4], axis=2).sum())
    print("rest-pose pixels changed", rest_diff, "south", rest_s_diff, flush=True)
    if rest_diff or rest_s_diff:
        bad = rest if rest_diff else rest_s
        ref = canvas if rest_diff else rigs["S"][4]
        ys, xs = np.where(np.any(bad != ref, axis=2))
        print("mismatch", list(zip(xs[:12].tolist(), ys[:12].tolist())), flush=True)
        raise SystemExit("frame 0 does not match the still")

    neutral = rest_s
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    print(
        f"cell {CELL_W}x{CELL_H} foot {FOOT_Y} height {sprite_h} feet 0 mouth {info['mouth']}",
        flush=True,
    )

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (80, 170, 210),
        "left_ant": (240, 210, 80),
        "right_ant": (240, 140, 40),
        "left_wing": (80, 200, 120),
        "right_wing": (40, 140, 90),
        "tail": (220, 80, 160),
        "mouth": (255, 60, 60),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    ratios = attack_report(rigs["S"], palette)
    bad = any(r < 0.90 or r > 1.08 for r in ratios)
    if bad:
        print("attack ratios outside 0.90-1.08", flush=True)

    if args.samples:
        want = [("attack", "S", f) for f in range(16)]
        want += [("idle", "S", f) for f in (0, 4, 8, 12)]
        want += [("walk", "S", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("hurt", "S", f) for f in (0, 1, 3, 6, 11)]
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
        print("samples", work / "samples.png", flush=True)
        if bad:
            raise SystemExit(2)
        return

    if bad:
        raise SystemExit(2)

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
                if fp.exists() and fp.stat().st_size > 0 and not args.force:
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
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    report, ratios = qa_sheet(master, sheets, palette, frames_dir, digest, sprite_h, rest_diff, args.skip_qa)
    (work / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "spin"}, ensure_ascii=False)[:900], flush=True)

    meta = {
        "generator": "forge_mantyke_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 458,
        "variant": "mantyke",
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "flip_dirs": sorted(FLIP_DIRS),
        "palette_colors": int(len(palette)),
        "off_palette_colors": report["off_palette_colors"],
        "attack_s_ratios": report["attack_s_ratios"],
        "parts": info["parts"],
        "facing": "front",
        "feet": 0,
        "mouth": info["mouth"],
        "source": "front.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not report["ok"]:
        if not args.no_publish:
            update_queue("needs_fix")
        raise SystemExit("qa failed; not publishing")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if not args.no_publish:
        publish_dir8(sprite_h)
        update_queue("done")
        print("published", VERSION, flush=True)


if __name__ == "__main__":
    main()
