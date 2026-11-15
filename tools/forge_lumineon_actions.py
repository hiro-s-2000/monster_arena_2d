#!/usr/bin/env python3
"""Lumineon (0457) 8-direction action sheet from this folder's 96x96 front.png.

Measured on this still only (opaque 1548, bbox x20-76 y20-76, 57x57, 10 colours,
alpha 0/255, one 4-connected body). The white eye pixel is (29, 58). A pink
eyespot sits at x64-67 y51-54. There is no mouth. The still faces left: the eye
is left of the rear fin.

Parts, from this still only. None of these are feet:
  crest   dorsal fin, x25-33 y20-32. Narrow base.
  wing    rear fin tip, x60-76 y28-34. Not a foot.
  pec     front pectoral membrane, x20-27 y51-69. Not a foot.
  tail    ventral lobes, y>=70. Not a foot.
  torso   body, eye, eyespot. Everything else.

Left-facing still. S SE E NE are mirrored so E points screen-right.
N NW W SW keep the still so W points screen-left.
idle  : hover. Frame 0 is the still. Fins and tail sway.
walk  : in-place fin beats, two cycles, no travel, no feet.
attack: 0-4 fin and tail roar plus rings (no invented mouth),
        5-10 one-piece in-plane 360 spin (no |cos| squash),
        11-12 impact, 13-15 recover.
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
STILL_DIR = ROOT / "monsterSprite/gen4/0457_lumineon/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0457_lumineon/normal/agent-sprite-forge"
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
VERSION = "lumineon_v2"
KEY = "0457_lumineon"
# Left-facing still. Mirror the east-side rows.
FLIP_DIRS = {"S", "SE", "E", "NE"}
MAX_RECOIL = 36.0
FROZEN_KEYS = ("0456_finneon",)

INK = (16, 16, 16)
DEEP = (41, 41, 57)
NAVY = (49, 65, 90)
SLATE = (82, 106, 139)
PURPLE = (115, 65, 123)
STEEL = (115, 148, 172)
CYAN = (148, 213, 230)
PINK = (197, 74, 148)
PALE = (197, 230, 246)
WHITE = (238, 255, 255)

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


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1, constant_values=False)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def segment(img):
    """Geometric parts of this left-facing still. Masks partition every opaque pixel."""
    op = img[..., 3] >= 128
    h, w = op.shape
    yy, xx = np.mgrid[0:h, 0:w]
    fincol = is_color(img, CYAN) | is_color(img, PALE) | is_color(img, SLATE) | is_color(img, STEEL)
    bodycol = (
        is_color(img, NAVY)
        | is_color(img, DEEP)
        | is_color(img, PINK)
        | is_color(img, PURPLE)
        | is_color(img, WHITE)
    )
    ink = is_color(img, INK)
    # Dorsal fin above the back. Not a foot.
    crest = op & (yy <= 32) & (xx <= 36)
    # Rear fin tip. The long attached fringe stays on the torso.
    wing_core = fincol & (yy <= 34) & (xx >= 58) & ~crest
    wing_ink = ink & dilate(wing_core, 1) & ~dilate(bodycol, 1) & ~crest
    wing = wing_core | wing_ink
    # Front pectoral membrane, left of the eye. Not a foot.
    pec_core = fincol & (xx <= 27) & (yy >= 50) & (yy <= 69) & ~crest & ~wing
    pec_ink = ink & dilate(pec_core, 1) & ~dilate(bodycol, 1) & ~crest & ~wing
    pec = pec_core | pec_ink
    # Ventral lobes. Not feet.
    tail = op & (yy >= 70) & ~crest & ~wing & ~pec
    names = {"crest": crest, "wing": wing, "pec": pec, "tail": tail}
    used = np.zeros_like(op)
    for mask in names.values():
        if np.any(used & mask):
            raise SystemExit("lumineon parts overlap")
        used |= mask
    torso = op & ~used
    names["torso"] = torso
    if int(used.sum() + torso.sum()) != int(op.sum()):
        raise SystemExit("lumineon parts do not partition the still")
    eye = is_color(img, WHITE) | is_color(img, PINK) | is_color(img, PURPLE)
    if int((eye & used).sum()) != 0:
        raise SystemExit("eye or eyespot left the torso")
    if int(is_color(img, WHITE).sum()) != 1:
        raise SystemExit("white eye pixel missing")
    if int(crest.sum()) < 40 or int(wing.sum()) < 40 or int(tail.sum()) < 40:
        raise SystemExit("a fin collapsed")
    return names


def measure_face(img, masks):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    eye = is_color(img, WHITE)
    ey, ex = np.nonzero(eye)
    wing_ys, wing_xs = np.nonzero(masks["wing"])
    face_x = float(ex.mean())
    rear_x = float(wing_xs.mean())
    if not face_x < rear_x:
        raise SystemExit(f"expected left-facing still, eye {face_x:.1f} rear {rear_x:.1f}")
    return {
        "opaque": int(op.sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "sprite_wh": [int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        "parts": {k: int(v.sum()) for k, v in masks.items()},
        "facing": "left",
        "feet": 0,
        "mouth": 0,
        "eye_x": face_x,
        "rear_x": rear_x,
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


def hinge(part, body):
    """Mean of the part pixels that touch the body. The free tip swings around this."""
    touch = part & dilate(body, 1)
    if int(touch.sum()) < 3:
        touch = part & dilate(body, 2)
    if not np.any(touch):
        touch = part
    ys, xs = np.nonzero(touch)
    return float(xs.mean()), float(ys.mean())


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
    for name in ("crest", "wing", "pec", "tail"):
        touch = placed[name] & dilate(torso_m, 1)
        if np.any(touch):
            layers["torso"][touch] = layers[name][touch]

    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    mirror = -1.0 if flip else 1.0
    pivots = {
        "crest": hinge(placed["crest"], placed["torso"]),
        "wing": hinge(placed["wing"], placed["torso"]),
        "pec": hinge(placed["pec"], placed["torso"]),
        "tail": hinge(placed["tail"], placed["torso"]),
        "body": center,
    }
    return layers, pivots, center, mirror, canvas


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Sine that is exactly 0 on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def flap(f, n, amp, cycles=1.0):
    """Non-negative hump. Exactly 0 on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (1.0 - math.cos(ph)) * 0.5


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0,
        crest=0.0, wing=0.0, pec=0.0, tail=0.0,
        flare=0.0,
        spin=0.0, recoil=0.0, back=0.0,
        ring=-1, burst=0, sparks=0,
    )
    if action == "idle":
        # Hover. Frame 0 is the still. Fins sway. The tail only wags on its safe side.
        p["dy"] = wave(f, n, -1.0)
        p["crest"] = wave(f, n, 10.0, phase=0.4)
        p["wing"] = -flap(f, n, 14.0)
        p["pec"] = wave(f, n, 8.0, phase=1.2)
        p["tail"] = -flap(f, n, 10.0, cycles=1.0)
    elif action == "walk":
        # In-place glide. Two beats. No feet and no travel.
        p["dy"] = wave(f, n, -1.0, cycles=2.0)
        p["crest"] = wave(f, n, 12.0, cycles=2.0, phase=0.3)
        p["wing"] = -flap(f, n, 18.0, cycles=2.0)
        p["pec"] = wave(f, n, 10.0, cycles=2.0, phase=0.8)
        p["tail"] = -flap(f, n, 12.0, cycles=2.0)
    elif action == "attack":
        # 0-4 fin roar, 5-10 spin, 11-12 impact, 13-15 recover. No mouth.
        p["crest"] = [0, 8, 14, 18, 10, 0, 0, 0, 0, 0, 0, 0, 0, 8, 4, 0][f]
        p["wing"] = [0, -8, -16, -24, -12, 0, 0, 0, 0, 0, 0, 0, 0, -8, -4, 0][f]
        p["pec"] = [0, 6, 12, 14, 8, 0, 0, 0, 0, 0, 0, 0, 0, 6, 2, 0][f]
        p["tail"] = [0, -6, -10, -14, -8, 0, 0, 0, 0, 0, 0, 0, 0, -6, -2, 0][f]
        p["flare"] = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["dy"] = [0, -1, -1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        if 1 <= f <= 4:
            p["ring"] = f - 1
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
        if f in (11, 12):
            p["burst"] = 2 if f == 11 else 1
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.82, 0.55, 0.32, 0.16, 0.06, 0.0, 0.0, 0.0][f]
        p["recoil"] = MAX_RECOIL * env
        p["back"] = 2.0 * env
        p["dy"] = -1.0 * env
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def moved_part(layer, angle, pivot, dx, dy, flare, outward):
    if abs(angle) >= 0.45:
        layer = rotate_nearest(layer, angle, pivot)
    if flare:
        ox, oy = outward
        layer = shift(layer, ox * flare, oy * flare)
    return shift(layer, dx, dy)


def outward_step(layer, pivot):
    """One pixel from the hinge toward the free tip. Zero when the part is empty."""
    op = layer[..., 3] >= 128
    ys, xs = np.nonzero(op)
    if len(xs) == 0:
        return 0.0, 0.0
    dx = float(xs.mean()) - pivot[0]
    dy = float(ys.mean()) - pivot[1]
    norm = math.hypot(dx, dy) or 1.0
    sx = int(round(dx / norm))
    sy = int(round(dy / norm))
    return sx, sy


def compose(layers, pivots, p, mirror):
    """Fins and the tail move on their hinges. Nothing here is a foot."""
    bdx = p["dx"] * mirror
    bdy = p["dy"]
    torso = shift(layers["torso"], bdx, bdy)
    order = ("tail", "wing", "pec", "crest")
    moved = {}
    for name in order:
        ang = p[name] * mirror
        dx, dy = outward_step(layers[name], pivots[name])
        moved[name] = moved_part(
            layers[name], ang, pivots[name], bdx, bdy, p["flare"], (dx, dy)
        )
    body = np.zeros_like(torso)
    paste(body, torso)
    for name in order:
        paste(body, moved[name])
    return body


def head_point(img):
    """Empty space just past the eye, along the eye-from-centroid direction. Not a mouth."""
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    if len(xs) == 0:
        return CELL_W / 2, CELL_H / 2
    cx, cy = float(xs.mean()), float(ys.mean())
    white = op & np.all(img[..., :3] == np.array(WHITE, np.uint8), axis=-1)
    if np.any(white):
        wy, wx = np.nonzero(white)
        ex, ey = float(wx.mean()), float(wy.mean())
    else:
        ex, ey = float(xs.min()), cy
    dx, dy = ex - cx, ey - cy
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    x, y = ex, ey
    h, w = img.shape[:2]
    for _ in range(28):
        x += ux
        y += uy
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h and img[yi, xi, 3] >= 128):
            return x + ux * 4.0, y + uy * 3.0
    return ex + ux * 8.0, ey + uy * 4.0


def fin_rings(img, origin, age):
    """Sparse rings in front of the eye. They are not a mouth and not body."""
    ox, oy = origin
    cols = (WHITE, PALE, CYAN, PINK)
    for ri in range(age + 1):
        rad = 5 + ri * 3
        col = cols[ri % 4]
        for deg in (-50, 0, 50):
            ang = math.radians(deg)
            put(img, ox + rad * math.cos(ang), oy + rad * math.sin(ang) * 0.65, col)
    return img


def impact_burst(img, origin, size):
    ox, oy = origin
    pts = [(0, 0), (4, -2), (4, 2), (7, 0), (3, -5), (3, 5), (9, -1), (9, 1)]
    cols = (WHITE, PALE, CYAN, PINK)
    n = 4 + size * 2
    for i, (dx, dy) in enumerate(pts[:n]):
        put(img, ox + dx, oy + dy, cols[i % 4])
    return img


def sparks(img, frame_idx):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    if len(xs) == 0:
        return img
    top = int(ys.min())
    left = int(xs.min())
    right = int(xs.max())
    put(img, left + 6, top - 3 - (frame_idx % 2), WHITE)
    put(img, (left + right) // 2, top - 4, PALE)
    put(img, right - 8, top - 3, CYAN)
    return img


def components(img):
    """4-connected components of alpha==255. Returns count, largest fraction, total."""
    opaque = img[..., 3] == 255
    if not np.any(opaque):
        return 0, 1.0, 0
    seen = np.zeros_like(opaque, dtype=bool)
    h, w = opaque.shape
    best = 0
    ncomp = 0
    total = int(opaque.sum())
    ys, xs = np.nonzero(opaque)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        ncomp += 1
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
    return ncomp, best / total, total


def synth(rig, action, f, palette, effects=True):
    layers, pivots, center, mirror, canvas = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        # One piece. Spin or lean the still itself. Do not squash with |cos|.
        body = canvas.copy()
        if recoiling:
            body = shift(body, p["back"] * mirror, p["dy"])
            pivot = (center[0] + p["back"] * mirror, center[1] + p["dy"])
            body = rotate_nearest(body, p["recoil"] * mirror, pivot)
            body = fill_pinholes(body)
            body = despeckle(body)
        else:
            ang = p["spin"] * mirror
            turned = abs(ang % 360.0) > 0.45 and abs((ang % 360.0) - 360.0) > 0.45
            if turned:
                body = rotate_nearest(body, ang, center)
                body = fill_pinholes(body)
                body = despeckle(body)
    else:
        body = compose(layers, pivots, p, mirror)
        moved = (
            abs(p["crest"]) > 0.45
            or abs(p["wing"]) > 0.45
            or abs(p["pec"]) > 0.45
            or abs(p["tail"]) > 0.45
            or abs(p["dy"]) > 0.45
            or p["flare"]
        )
        if moved:
            body = fill_pinholes(body)
            body = despeckle(body)
    img = snap(body, palette)
    if not effects:
        return img
    if p["ring"] >= 0 and not spinning:
        img = fin_rings(img, head_point(img), p["ring"])
    if p["burst"]:
        img = impact_burst(img, head_point(img), p["burst"])
    if p["sparks"]:
        img = sparks(img, f)
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
    """South row, every frame, cropped to the fish."""
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


def _key_blob(text, key):
    s, e = _entry_span(text, key)
    return text[s:e]


def publish_dir8(sprite_h):
    entry = {
        "dex": 457,
        "form": "normal",
        "variant": "lumineon",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0457_lumineon/normal",
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
            stripped = text.strip()
            if "\n" in stripped:
                raise SystemExit("dir8.js is no longer one line")
            body = stripped[len(prefix):]
            if body.endswith(";"):
                body = body[:-1].strip()
            else:
                raise SystemExit("dir8.js has no trailing semicolon")
            data = json.loads(body)
            if KEY not in data:
                raise SystemExit("0457_lumineon missing from dir8")
            keys = re.findall(r'"(\d{4}_[^"]+)":\{', text)
            if keys.count(KEY) != 1:
                raise SystemExit(f"{KEY} key count is {keys.count(KEY)}")
            frozen = {k: _key_blob(text, k) for k in FROZEN_KEYS}
            if text.count('"v":"finneon_v2"') != 1:
                raise SystemExit("finneon_v2 stamp missing before edit")
            idx = keys.index(KEY)
            prev_key = keys[idx - 1] if idx > 0 else None
            next_key = keys[idx + 1] if idx + 1 < len(keys) else None
            prev_blob = _key_blob(text, prev_key) if prev_key else None
            next_blob = _key_blob(text, next_key) if next_key else None
            s, e = _entry_span(text, KEY)
            new_text = text[:s] + f'"{KEY}":' + blob + text[e:]
            if prev_blob and prev_blob not in new_text:
                raise SystemExit("neighbor dir8 key changed")
            if next_blob and next_blob not in new_text:
                raise SystemExit("neighbor dir8 key changed")
            for k, old in frozen.items():
                if _key_blob(new_text, k) != old:
                    raise SystemExit(f"refusing to touch {k}")
            new_body = new_text.strip()[len(prefix):]
            if new_body.endswith(";"):
                new_body = new_body[:-1].strip()
            parsed = json.loads(new_body)
            if parsed.get(KEY) != entry:
                raise SystemExit("dir8 entry did not round-trip")
            if new_text.count(f'"v":"{VERSION}"') != 1:
                raise SystemExit("dir8 version stamp collided")
            if new_text.count('"v":"finneon_v2"') != 1:
                raise SystemExit("finneon_v2 was touched")
            if "\n" in new_text.strip():
                raise SystemExit("dir8.js would no longer be one line")
            if not new_text.endswith("\n"):
                new_text += "\n"
            DIR8.write_text(new_text)
            written = DIR8.read_text()
            for k, old in frozen.items():
                if _key_blob(written, k) != old:
                    raise SystemExit(f"dir8 wrote over {k}")
            if _key_blob(written, KEY) != f'"{KEY}":' + blob:
                raise SystemExit("dir8 lumineon entry mismatch after write")
            if written.count('"v":"finneon_v2"') != 1:
                raise SystemExit("finneon_v2 changed on disk")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def update_queue(status):
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            text = QUEUE.read_text()
            data = json.loads(text)
            before = {item.get("id"): (item.get("status"), item.get("claim")) for item in data["items"]}
            st, claim = before.get(KEY, (None, None))
            if not (st == "in_progress" and claim == "parallel-batch"):
                raise SystemExit(f"queue claim is not parallel-batch: {before.get(KEY)}")
            start = text.index('"id": "0457_lumineon"')
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
    raise SystemExit("0457_lumineon missing from queue")


def south_report(rig, palette):
    """Opaque ratios of every south frame against the still. Spin component counts."""
    base_img = synth(rig, "idle", 0, palette, effects=False)
    base = int((base_img[..., 3] == 255).sum())
    ratios = []
    print("SOUTH_RATIOS", flush=True)
    for act in ACTIONS:
        for f in range(FRAMES[act]):
            tile = synth(rig, act, f, palette)
            n = int((tile[..., 3] == 255).sum())
            ncomp, frac, _total = components(tile)
            ratio = n / base if base else 0.0
            ratios.append((act, f, ratio, ncomp, frac, n))
            extra = ""
            if act == "attack" and 5 <= f <= 10:
                extra = " SPIN"
            print(
                f"{act:6} S {f:02d} opaque {n} ratio {ratio:.3f} comps {ncomp} main {frac:.3f}{extra}",
                flush=True,
            )
    vals = [r[2] for r in ratios]
    print(f"min {min(vals):.3f} max {max(vals):.3f}", flush=True)
    return ratios, base


def qa_sheet(master, sheets, palette, frames_dir, digest, sprite_h, rest_diff, skip_qa):
    base = None
    ratios = []
    print("SOUTH_RATIOS", flush=True)
    for act in ACTIONS:
        row = sheets[act][:CELL_H]
        for f in range(FRAMES[act]):
            tile = row[:, f * CELL_W:(f + 1) * CELL_W]
            n = int((tile[..., 3] == 255).sum())
            if base is None:
                base = n
            ratio = n / base if base else 0.0
            ratios.append(ratio)
            print(f"{act:6} S {f:02d} opaque {n} ratio {ratio:.3f}", flush=True)
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
                    ncomp, frac, total_px = components(tile)
                    spin_stats.append({
                        "d": d, "f": f, "opaque": total_px,
                        "comps": ncomp, "main": round(frac, 3),
                    })
    print("SPIN_COMPONENTS", flush=True)
    for s in spin_stats:
        print(f"attack {s['d']} {s['f']:02d} comps {s['comps']} main {s['main']:.3f} opaque {s['opaque']}", flush=True)
    report = {
        "version": VERSION,
        "hash": digest,
        "sprite_h": sprite_h,
        "cell": [CELL_W, CELL_H],
        "foot_y": FOOT_Y,
        "feet": 0,
        "mouth": 0,
        "rest_diff": rest_diff,
        "off_palette_colors": len(off),
        "partial_alpha": partial,
        "edge_touching": clip,
        "south_ratios": [round(r, 3) for r in ratios],
        "south_min": round(min(ratios), 3),
        "south_max": round(max(ratios), 3),
        "spin": spin_stats,
        "max_recoil": MAX_RECOIL,
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
        and sprite_h > 33
        and all(0.90 <= r <= 1.08 for r in ratios)
        and all(s["comps"] == 1 for s in spin_stats)
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
        "wh", info["sprite_wh"], "eye_x", round(info["eye_x"], 1),
        "rear_x", round(info["rear_x"], 1),
        flush=True,
    )
    if info["sprite_wh"][1] <= 33:
        raise SystemExit(f"height {info['sprite_wh'][1]} is not larger than finneon")
    if info["feet"] != 0:
        raise SystemExit("feet must be 0")

    rigs = {}
    for d in DIRS:
        rigs[d] = build_rig(front, masks, flip=d in FLIP_DIRS)

    rest_w = synth(rigs["W"], "idle", 0, palette, effects=False)
    rest_w_diff = int(np.any(rest_w != rigs["W"][4], axis=2).sum())
    rest_s = synth(rigs["S"], "idle", 0, palette, effects=False)
    rest_s_diff = int(np.any(rest_s != rigs["S"][4], axis=2).sum())
    print("rest-pose pixels changed", rest_w_diff, "west", rest_s_diff, "south", flush=True)
    if rest_w_diff or rest_s_diff:
        bad = rest_w if rest_w_diff else rest_s
        ref = rigs["W"][4] if rest_w_diff else rigs["S"][4]
        ys, xs = np.where(np.any(bad != ref, axis=2))
        print("mismatch", list(zip(xs[:12].tolist(), ys[:12].tolist())), flush=True)
        raise SystemExit("frame 0 does not match the still")
    rest_diff = rest_w_diff + rest_s_diff

    neutral = rest_s
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    print(
        f"cell {CELL_W}x{CELL_H} foot {FOOT_Y} height {sprite_h} feet 0 mouth 0",
        flush=True,
    )

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (80, 140, 210),
        "crest": (240, 210, 80),
        "wing": (80, 200, 120),
        "pec": (240, 140, 40),
        "tail": (220, 80, 160),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    ratios, _base = south_report(rigs["S"], palette)
    bad = any(r[2] < 0.90 or r[2] > 1.08 for r in ratios)
    spin_bad = any(r[3] != 1 for r in ratios if r[0] == "attack" and 5 <= r[1] <= 10)
    if bad or spin_bad:
        print("south ratios or spin components failed", flush=True)

    if args.samples:
        want = [("attack", "S", f) for f in range(16)]
        want += [("idle", "S", f) for f in (0, 4, 8, 12)]
        want += [("walk", "S", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("hurt", "S", f) for f in (0, 1, 3, 6, 11)]
        want += [("attack", "W", f) for f in (0, 3, 7, 11)]
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
        if bad or spin_bad:
            raise SystemExit(2)
        return

    if bad or spin_bad:
        if not args.no_publish:
            update_queue("needs_fix")
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
    report, ratios = qa_sheet(
        master, sheets, palette, frames_dir, digest, sprite_h, rest_diff, args.skip_qa
    )
    (work / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    brief = {k: v for k, v in report.items() if k not in ("spin", "south_ratios")}
    print(json.dumps(brief, ensure_ascii=False), flush=True)
    print(
        f"FEET {info['feet']} cell {CELL_W}x{CELL_H} fy {FOOT_Y} sprite_h {sprite_h} "
        f"parts {info['parts']}",
        flush=True,
    )

    meta = {
        "generator": "forge_lumineon_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 457,
        "variant": "lumineon",
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
        "south_ratios": report["south_ratios"],
        "parts": info["parts"],
        "facing": "left",
        "feet": 0,
        "mouth": 0,
        "source": "front.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not report["ok"]:
        if not args.no_publish:
            update_queue("needs_fix")
        raise SystemExit("qa failed; not publishing")
    if not args.no_sync:
        n = sync_outputs()
        arena = SYNC_OUT / "sheet.png"
        print("synced files", n, "arena_bytes", arena.stat().st_size if arena.is_file() else 0, flush=True)
    if not args.no_publish:
        publish_dir8(sprite_h)
        update_queue("done")
        print("published", VERSION, flush=True)


if __name__ == "__main__":
    main()
