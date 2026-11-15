#!/usr/bin/env python3
"""Gible (0443) 8-direction action sheet from this folder's 96x96 front.png.

Measured on this still only (opaque 1304, bbox x25-69 y20-70, 45x51, 15 colours,
alpha 0/255). The snout and the red mouth sit left of the cyan eye, so the still
faces left. Parts, from this still only:

  dorsal     head crest, y<=34. It is not a second body.
  jaw        lower maw and chin, y56-65 and x<=52. Upper teeth stay on the head.
  front_foot leading stub, y>=66 and x<=44 (under the chin).
  back_foot  trailing stub, y>=66 and x>=54.
  arm        the small rear spike, x>=64 and y41-47. One fin, not a new limb.
  tail       the rear nub, x>=66 and y50-54.
  torso      head, eye, upper mouth, belly, everything else.

S SE E NE are the horizontal mirror (screen-right). N NW W SW keep the still.
idle  : crest and rear fin sway. Frame 0 is the still. Feet stay on the ground.
walk  : alternating 2px step of the two measured stubs. No extra legs.
attack: 0-4 the existing jaw drops and a red wedge plus rings, 5-10 one-piece
        in-plane 360 spin (no |cos| squash), 11-12 bite and a small impact,
        13-15 close.
hurt  : one-piece lean, at most 34 degrees.

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
STILL_DIR = ROOT / "monsterSprite/gen4/0443_gible/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0443_gible/normal/agent-sprite-forge"
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
VERSION = "gible_v2"
KEY = "0443_gible"
# The still faces left, so the screen-right rows are mirrored.
FLIP_DIRS = {"S", "SE", "E", "NE"}
MAX_RECOIL = 34.0

INK = (16, 16, 16)
BLUE = (82, 148, 172)
SHADE = (65, 106, 131)
DEEP = (49, 65, 82)
LIGHT = (139, 197, 213)
CYAN = (65, 213, 222)
MOUTH = (222, 90, 41)
MOUTH_LT = (205, 115, 115)
MOUTH_DK = (123, 24, 16)
MOUTH_MID = (172, 57, 16)
WHITE = (255, 255, 255)
PALE = (197, 205, 213)
REDS = (MOUTH, MOUTH_LT, MOUTH_DK, MOUTH_MID, (90, 16, 0))

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
    """Geometric parts of this left-facing still. Masks partition every opaque pixel."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    front_foot = op & (yy >= 66) & (xx <= 44)
    back_foot = op & (yy >= 66) & (xx >= 54)
    dorsal = op & (yy <= 34) & ~front_foot & ~back_foot
    arm = op & (yy >= 41) & (yy <= 47) & (xx >= 64) & ~dorsal
    tail = op & (yy >= 50) & (yy <= 54) & (xx >= 66)
    jaw = op & (yy >= 56) & (yy <= 65) & (xx <= 52)
    used = front_foot | back_foot | dorsal | arm | tail | jaw
    torso = op & ~used
    masks = {
        "dorsal": dorsal,
        "jaw": jaw,
        "front_foot": front_foot,
        "back_foot": back_foot,
        "arm": arm,
        "tail": tail,
        "torso": torso,
    }
    total = sum(int(m.sum()) for m in masks.values())
    if total != int(op.sum()):
        raise SystemExit(f"parts do not partition the still: {total} vs {int(op.sum())}")
    overlap = 0
    names = list(masks)
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            overlap += int((masks[n1] & masks[n2]).sum())
    if overlap:
        raise SystemExit(f"part masks overlap: {overlap}")
    if int(front_foot.sum()) < 20 or int(back_foot.sum()) < 20:
        raise SystemExit("a measured foot is missing")
    if int(jaw.sum()) < 80 or int(dorsal.sum()) < 80 or int(torso.sum()) < 700:
        raise SystemExit(
            f"jaw/crest/torso split looks wrong: jaw {int(jaw.sum())} "
            f"dorsal {int(dorsal.sum())} torso {int(torso.sum())}"
        )
    if int(arm.sum()) < 6 or int(arm.sum()) > 80:
        raise SystemExit(f"rear fin cut looks wrong: {int(arm.sum())}")
    eye = is_color(img, CYAN)
    if int((eye & torso).sum()) != int(eye.sum()):
        raise SystemExit("the cyan eye was cut off the head")
    red = np.zeros_like(op)
    for color in REDS:
        red |= is_color(img, color)
    rys, rxs = np.nonzero(red)
    eys, exs = np.nonzero(eye)
    if float(rxs.mean()) >= float(exs.mean()):
        raise SystemExit("mouth is not left of the eye; facing was not remeasured")
    return masks


def measure_face(img, masks):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    return {
        "opaque": int(op.sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "parts": {k: int(v.sum()) for k, v in masks.items()},
        "facing": "left",
        "feet": 2,
        "eyes": int(is_color(img, CYAN).sum()),
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


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    for i in range(0, len(ty), 400):
        sl = slice(i, i + 400)
        diff_y = ty[sl, None].astype(np.int32) - pts[None, :, 0]
        diff_x = tx[sl, None].astype(np.int32) - pts[None, :, 1]
        dist = diff_y * diff_y + diff_x * diff_x
        pick[sl] = dist.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    layer[ty, tx, 3] = 255
    return layer


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


def contact_pivot(part, body, toward):
    """Joint on the part edge that touches the body, biased toward a side."""
    touch = part & dilate(body, 2)
    if not np.any(touch):
        touch = part
    ys, xs = np.nonzero(touch)
    if toward == "low":
        band = ys >= ys.max() - 1
    elif toward == "high":
        band = ys <= ys.min() + 1
    elif toward == "left":
        band = xs <= xs.min() + 1
    else:
        band = xs >= xs.max() - 1
    return float(xs[band].mean()), float(ys[band].mean())


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

    # One-pixel cushions under moving parts. Covered at rest, so frame 0 matches.
    torso_m = placed["torso"]
    for name in ("dorsal", "jaw", "arm", "tail", "front_foot", "back_foot"):
        touch = placed[name] & dilate(torso_m, 1)
        if np.any(touch):
            layers["torso"][touch] = layers[name][touch]

    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    facing = 1.0 if flip else -1.0
    pivots = {
        "dorsal": contact_pivot(placed["dorsal"], placed["torso"], "low"),
        "arm": contact_pivot(placed["arm"], placed["torso"], "left" if facing > 0 else "right"),
        "tail": contact_pivot(placed["tail"], placed["torso"], "left" if facing > 0 else "right"),
        "jaw": contact_pivot(placed["jaw"], placed["torso"], "high"),
        "body": center,
    }
    # Jaw hinge is the rear of the maw (toward the torso), not the snout lip.
    jys, jxs = np.nonzero(placed["jaw"])
    top = int(jys.min())
    band = jys <= top + 1
    if facing > 0:
        hx = float(jxs[band].min())
    else:
        hx = float(jxs[band].max())
    pivots["jaw"] = (hx, float(top))
    return layers, pivots, center, facing, canvas


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Sine that is exactly 0 on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0,
        jaw=0, fin=0.0, arm=0.0, tail=0.0,
        front_dx=0.0, front_dy=0.0,
        back_dx=0.0, back_dy=0.0,
        spin=0.0, recoil=0.0,
        ring=-1, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["dy"] = wave(f, n, -1.0)
        p["fin"] = wave(f, n, 4.0, phase=-0.4)
        p["arm"] = wave(f, n, 5.0, phase=0.8)
        p["tail"] = wave(f, n, 4.0, phase=1.1)
    elif action == "walk":
        step = wave(f, n, 1.0, cycles=2.0)
        p["dy"] = wave(f, n, -1.0, cycles=2.0)
        p["fin"] = wave(f, n, 6.0, cycles=2.0)
        p["arm"] = wave(f, n, -8.0, cycles=2.0)
        p["tail"] = wave(f, n, 5.0, cycles=2.0, phase=-0.7)
        # Two measured stubs, opposite phase. Lift is 2px, about half a foot.
        p["front_dy"] = -max(0.0, step * 2.0)
        p["front_dx"] = step * 1.5
        p["back_dy"] = -max(0.0, -step * 2.0)
        p["back_dx"] = -step * 1.5
    elif action == "attack":
        # 0-4 roar, 5-10 spin, 11-12 impact, 13-15 recover. Jaw drop is integer px.
        p["jaw"] = [0, 1, 2, 3, 2, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0][f]
        p["fin"] = [0, 4, 8, 11, 6, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0][f]
        if 2 <= f <= 4:
            p["ring"] = f - 2
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
        if f in (11, 12):
            p["burst"] = 2 if f == 11 else 1
    else:
        env = [0.0, 0.42, 0.78, 1.0, 0.80, 0.55, 0.32, 0.16, 0.05, 0.0, 0.0, 0.0][f]
        p["recoil"] = MAX_RECOIL * env
        p["dx"] = -3.0 * env
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


def paint_jaw_wedge(img, jaw_closed, jaw_open):
    """Red mouth wedge in the rows the existing jaw just left. No new colours."""
    was = jaw_closed[..., 3] >= 128
    now = img[..., 3] >= 128
    opened = jaw_open[..., 3] >= 128
    gap = was & ~now & ~opened
    if not np.any(gap):
        return img
    ys = np.nonzero(was)[0]
    top = int(ys.min())
    yy = np.arange(img.shape[0])[:, None]
    gap &= yy <= top + 3
    if not np.any(gap):
        return img
    # Centre of the maw goes red. The outer column stays the original outline colour.
    xs = np.nonzero(gap)[1]
    x_lo, x_hi = int(xs.min()), int(xs.max())
    edge = gap.copy()
    edge[:] = False
    edge[:, x_lo] = gap[:, x_lo]
    edge[:, x_hi] = gap[:, x_hi]
    img[gap] = (*MOUTH, 255)
    src = jaw_closed
    img[edge] = src[edge]
    img[edge, 3] = 255
    return img


def compose(layers, pivots, p, facing):
    """Feet do not inherit the body bob, so a planted stub stays on the ground."""
    bdx = p["dx"] * facing
    bdy = p["dy"]
    torso = shift(layers["torso"], bdx, bdy)
    dorsal = moved_part(layers["dorsal"], p["fin"] * facing, pivots["dorsal"], bdx, bdy)
    arm = moved_part(layers["arm"], p["arm"] * facing, pivots["arm"], bdx, bdy)
    tail = moved_part(layers["tail"], p["tail"] * facing, pivots["tail"], bdx, bdy)
    jaw_closed = shift(layers["jaw"], bdx, bdy)
    jaw = shift(jaw_closed, 0, int(p["jaw"]))
    front = shift(layers["front_foot"], p["front_dx"] * facing, p["front_dy"])
    back = shift(layers["back_foot"], p["back_dx"] * facing, p["back_dy"])
    body = np.zeros_like(torso)
    for lyr in (tail, torso, dorsal, arm, back, jaw, front):
        paste(body, lyr)
    if int(p["jaw"]) > 0:
        body = paint_jaw_wedge(body, jaw_closed, jaw)
    return body


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mouth_point(img, facing):
    red = np.zeros(img.shape[:2], bool)
    for color in REDS:
        red |= is_color(img, color)
    ys, xs = np.nonzero(red)
    if len(xs) == 0:
        bb = body_box(img)
        if not bb:
            return CELL_W // 2, CELL_H // 2
        x = bb[2] if facing > 0 else bb[0]
        return x, (bb[1] + bb[3]) // 2
    if facing > 0:
        tip = int(xs.max())
        y = int(np.median(ys[xs >= tip - 1]))
        return tip + 2, y
    tip = int(xs.min())
    y = int(np.median(ys[xs <= tip + 1]))
    return tip - 2, y


def roar_rings(img, origin, facing, age):
    """Sparse rings in front of the maw. They are not part of the body."""
    ox, oy = origin
    cols = (WHITE, MOUTH, LIGHT)
    for ri in range(age + 1):
        rad = 8 + ri * 4
        col = cols[ri]
        for deg in range(-48, 49, 16):
            ang = math.radians(deg)
            put(img, ox + facing * rad * math.cos(ang), oy + rad * math.sin(ang) * 0.72, col)
    return img


def impact_burst(img, origin, facing, size):
    ox, oy = origin
    pts = [(5, 0), (8, -2), (8, 2), (4, -4), (4, 4), (11, 0), (7, -5), (7, 5)]
    cols = (WHITE, MOUTH, LIGHT, PALE)
    for i, (dx, dy) in enumerate(pts[: 4 + size * 2]):
        put(img, ox + facing * dx, oy + dy, cols[i % 4])
    return img


def sparks(img, frame_idx, facing, center):
    cx, cy = center
    d = frame_idx
    pts = (
        (cx - facing * (10 + d), cy - 16),
        (cx + facing * (8 + d), cy - 20),
        (cx - facing * 4, cy - 22 - d),
    )
    for sx, sy in pts:
        put(img, sx, sy, WHITE)
        put(img, sx + facing, sy, LIGHT)
    return img


def flash_hit(img):
    """One-frame blink. The blue body goes light. Mouth, eye, teeth, and ink stay."""
    out = img.copy()
    opaque = out[..., 3] >= 128
    rgb = out[..., :3]
    keep = np.zeros(opaque.shape, bool)
    for color in (INK, WHITE, PALE, CYAN, *REDS):
        keep |= opaque & np.all(rgb == np.array(color, np.uint8), axis=-1)
    out[opaque & ~keep, :3] = np.array(LIGHT, np.uint8)
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
    layers, pivots, center, facing, _canvas = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        # One piece. Spin or lean the frame-0 pose. Do not squash with |cos|.
        base = pose(action, 0)
        base = gen4_attack_strike.finish(base, action, 0, __file__)

        body = compose(layers, pivots, base, facing)
        if recoiling:
            body = shift(body, p["dx"] * facing, p["dy"])
            pivot = (center[0] + p["dx"] * facing, center[1] + p["dy"])
            # Negative angle lifts the snout side (a lean back, not a belly flip).
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
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_hit(img)
    if p["ring"] >= 0 and not spinning:
        img = roar_rings(img, mouth_point(img, facing), facing, p["ring"])
    if p["burst"]:
        img = impact_burst(img, mouth_point(img, facing), facing, p["burst"])
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
    """South row, every frame, cropped. Roar, spin, steps, and the lean."""
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
        "dex": 443,
        "form": "normal",
        "variant": "gible",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0443_gible/normal",
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
                raise SystemExit("0443_gible key duplicated")
            if keys.count(KEY) == 1:
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
            else:
                anchor = '"0442_spiritomb":'
                if anchor not in text:
                    raise SystemExit("0443_gible missing and 0442_spiritomb anchor missing")
                _s, e = _entry_span(text, "0442_spiritomb")
                new_text = text[:e] + "," + f'"{KEY}":' + blob + text[e:]
            if "0445_garchomp-mega" in text and text.count('"0445_garchomp-mega"') != new_text.count('"0445_garchomp-mega"'):
                raise SystemExit("garchomp mega key touched")
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
            if status == "done" and __import__("os").environ.get("GEN4_ATTACK_REBUILD") != "1":
                st, claim = before.get(KEY, (None, None))
                if not (st == "in_progress" and claim == "parallel-batch"):
                    raise SystemExit(f"queue claim is not parallel-batch: {before.get(KEY)}")
            start = text.index('"id": "0443_gible"')
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
    raise SystemExit("0443_gible missing from queue")


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
    ratios = []
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
                if act == "attack" and d == "S" and 5 <= f <= 10:
                    frac, total_px = largest_fraction(tile)
                    spin_stats.append({"f": f, "opaque": total_px, "main": round(frac, 3)})
    report = {
        "version": VERSION,
        "hash": digest,
        "sprite_h": sprite_h,
        "rest_diff": rest_diff,
        "off_palette_colors": len(off),
        "partial_alpha": partial,
        "edge_touching": clip,
        "attack_s_ratios": [round(r, 3) for r in ratios],
        "attack_s_min": round(min(ratios), 3),
        "attack_s_max": round(max(ratios), 3),
        "spin": spin_stats,
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
    palette = extract_palette(front)
    masks = segment(front)
    info = measure_face(front, masks)
    print("parts", info["parts"], "palette", len(palette), "facing", info["facing"], flush=True)

    rigs = {}
    for d in DIRS:
        rigs[d] = build_rig(front, masks, flip=d in FLIP_DIRS)

    rest = synth(rigs["W"], "idle", 0, palette, effects=False)
    canvas = rigs["W"][4]
    rest_diff = int(np.any(rest != canvas, axis=2).sum())
    rest_s = synth(rigs["S"], "idle", 0, palette, effects=False)
    rest_s_diff = int(np.any(rest_s != rigs["S"][4], axis=2).sum())
    print("rest-pose pixels changed", rest_diff, "south", rest_s_diff, flush=True)
    if rest_diff or rest_s_diff:
        raise SystemExit("frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (80, 140, 170),
        "dorsal": (240, 200, 40),
        "jaw": (220, 60, 40),
        "front_foot": (40, 200, 80),
        "back_foot": (40, 80, 200),
        "arm": (220, 40, 180),
        "tail": (240, 140, 40),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    ratios = attack_report(rigs["S"], palette)
    if any(r < 0.90 or r > 1.08 for r in ratios):
        print("attack ratios outside 0.90-1.08", flush=True)
        if args.samples:
            raise SystemExit(2)

    if args.samples:
        want = [("idle", "S", f) for f in (0, 4, 8, 12)]
        want += [("walk", "S", f) for f in (0, 2, 6, 10, 14)]
        want += [("attack", "S", f) for f in (0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 15)]
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
        return

    if any(r < 0.90 or r > 1.08 for r in ratios):
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
        "generator": "forge_gible_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 443,
        "variant": "gible",
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
        "facing": "left",
        "source": "front.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not report["ok"]:
        raise SystemExit("qa failed; not publishing")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if not args.no_publish:
        publish_dir8(sprite_h)
        update_queue("done")
        print("published", VERSION, flush=True)


if __name__ == "__main__":
    main()
