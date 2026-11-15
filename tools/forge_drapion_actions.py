#!/usr/bin/env python3
"""Drapion (0452) 8-direction action sheet from this folder's 96x96 front.png.

Measured on this still only (opaque 3013, bbox x8-87 y17-78, 80x62, 13 colours,
alpha 0/255). Large adult scorpion, frontal. Mouth magenta sits under the tail
(mouth centroid x44 y36). Two pincers flank the mouth. The stinger is the crown
above the face, not a leg. Ground contacts at y>=71 are four feet only:

  stinger    tail crown, y<=30 and x32-64. Not a leg.
  left_claw  left pincer, x<=26 down to its lower finger (y<=61, x<=22). Not a leg.
  right_claw right pincer, x>=72 and y24-54. Not a leg.
  leg_l      foot x32-40, y>=71.
  leg_ml     short foot x42-46, y>=72. The belly nub at x53-61 is not a foot.
  leg_r      foot x70-78, y>=71.
  leg_fr     foot x80-87, y>=71.
  jaw        lower mouth, y36-39 and x38-50. Upper face stays on the torso.
  torso      head, belly, hips, everything else.

S SE E NE keep the still (front). N NW W SW are mirrored. E faces screen-right
by the mirror of the front. idle: frame 0 is the still, four feet planted.
walk: those four feet only. attack: 0-4 both claws and a small jaw plus rings,
5-10 one-piece in-plane 360 spin (no |cos| squash), 11-12 impact, 13-15 recover.
hurt: one-piece lean, at most 36 degrees.

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
STILL_DIR = ROOT / "monsterSprite/gen4/0452_drapion/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0452_drapion/normal/agent-sprite-forge"
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
# Wide crown. Spin radius is about 45px, so the cell is a little past 176x168.
CELL_W, CELL_H, FOOT_Y = 192, 184, 124
VERSION = "drapion_v2"
KEY = "0452_drapion"
# Frontal still. South and east show it. North and west are the mirror.
FLIP_DIRS = {"N", "NW", "W", "SW"}
MAX_RECOIL = 36.0
FORBIDDEN_KEYS = (
    "0449_hippopotas",
    "0450_hippowdon",
    "0451_skorupi",
    "0453_croagunk",
    "0454_toxicroak",
    "0455_carnivine",
    "0456_finneon",
    "0457_lumineon",
)

INK = (16, 16, 16)
DPUR = (74, 49, 98)
MPUR = (139, 90, 156)
LPUR = (197, 123, 180)
MED = (123, 65, 123)
WHITE = (255, 255, 255)
VDARK = (57, 24, 57)
GRAY = (156, 172, 180)
TEAL = (41, 90, 98)
PALE = (213, 205, 230)
CYAN = (16, 139, 172)
MAG = (131, 49, 123)
PINK = (222, 148, 205)
MOUTH_COLS = (MAG, PINK, VDARK)

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
    """Geometric parts of this frontal still. Masks partition every opaque pixel."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    stinger = op & (yy <= 30) & (xx >= 32) & (xx <= 64)
    left_claw = op & ~stinger & (
        ((yy <= 52) & (xx <= 26))
        | ((yy >= 53) & (yy <= 61) & (xx <= 22))
    )
    right_claw = op & ~stinger & (xx >= 72) & (yy >= 24) & (yy <= 54)
    leg_l = op & (yy >= 71) & (xx >= 32) & (xx <= 40)
    leg_ml = op & (yy >= 72) & (xx >= 42) & (xx <= 46)
    leg_r = op & (yy >= 71) & (xx >= 70) & (xx <= 78)
    leg_fr = op & (yy >= 71) & (xx >= 80) & (xx <= 87)
    jaw = op & (yy >= 36) & (yy <= 39) & (xx >= 38) & (xx <= 50)
    for extra in (stinger, left_claw, right_claw, leg_l, leg_ml, leg_r, leg_fr):
        jaw &= ~extra
        leg_l &= ~extra if extra is not leg_l else leg_l
    # legs must not keep claw pixels; the cuts above already exclude them
    leg_l &= ~(stinger | left_claw | right_claw | jaw)
    leg_ml &= ~(stinger | left_claw | right_claw | jaw | leg_l)
    leg_r &= ~(stinger | left_claw | right_claw | jaw | leg_l | leg_ml)
    leg_fr &= ~(stinger | left_claw | right_claw | jaw | leg_l | leg_ml | leg_r)
    used = stinger | left_claw | right_claw | leg_l | leg_ml | leg_r | leg_fr | jaw
    torso = op & ~used
    masks = {
        "stinger": stinger,
        "left_claw": left_claw,
        "right_claw": right_claw,
        "leg_l": leg_l,
        "leg_ml": leg_ml,
        "leg_r": leg_r,
        "leg_fr": leg_fr,
        "jaw": jaw,
        "torso": torso,
    }
    total = sum(int(m.sum()) for m in masks.values())
    if total != int(op.sum()):
        raise SystemExit(f"parts do not partition the still: {total} vs {int(op.sum())}")
    names = list(masks)
    overlap = 0
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            overlap += int((masks[n1] & masks[n2]).sum())
    if overlap:
        raise SystemExit(f"part masks overlap: {overlap}")
    if int(left_claw.sum()) < 350 or int(right_claw.sum()) < 200:
        raise SystemExit(
            f"a pincer is missing: left {int(left_claw.sum())} right {int(right_claw.sum())}"
        )
    if int(stinger.sum()) < 200:
        raise SystemExit(f"stinger split looks wrong: {int(stinger.sum())}")
    for name in ("leg_l", "leg_ml", "leg_r", "leg_fr"):
        if int(masks[name].sum()) < 8:
            raise SystemExit(f"foot missing: {name} {int(masks[name].sum())}")
    if int(jaw.sum()) < 30 or int(torso.sum()) < 1200:
        raise SystemExit(f"jaw/torso split looks wrong: jaw {int(jaw.sum())} torso {int(torso.sum())}")
    # Claws flank the mouth. The tail crown is above the mouth. Not a side-on skorupi.
    mouth = np.zeros_like(op)
    for color in (MAG,):
        mouth |= is_color(img, color)
    if int(mouth.sum()) < 20:
        raise SystemExit("magenta mouth is missing")
    mys, mxs = np.nonzero(mouth)
    lxs = np.nonzero(left_claw)[1]
    rxs = np.nonzero(right_claw)[1]
    sys_, sxs = np.nonzero(stinger)
    if float(lxs.mean()) >= float(mxs.mean()) or float(rxs.mean()) <= float(mxs.mean()):
        raise SystemExit("pincers do not flank the mouth")
    if float(sys_.mean()) >= float(mys.mean()):
        raise SystemExit("stinger is not above the mouth")
    if int(op.sum()) < 2500:
        raise SystemExit("still is too small for adult drapion")
    return masks


def measure_face(img, masks):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    return {
        "opaque": int(op.sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "sprite_wh": [int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        "parts": {k: int(v.sum()) for k, v in masks.items()},
        "facing": "front",
        "feet": 4,
        "claws": 2,
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
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    px, py = float(pivot[0]), float(pivot[1])
    corners = ((x0, y0), (x1 - 1, y0), (x0, y1 - 1), (x1 - 1, y1 - 1))
    reach = max(math.hypot(x - px, y - py) for x, y in corners) + 2.0
    rad = int(math.ceil(reach))
    sx0 = int(math.floor(px)) - rad
    sy0 = int(math.floor(py)) - rad
    side = rad * 2 + 1
    crop = np.zeros((side, side, 4), np.uint8)
    src_x0, src_y0 = max(0, sx0), max(0, sy0)
    src_x1, src_y1 = min(layer.shape[1], sx0 + side), min(layer.shape[0], sy0 + side)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return layer.copy()
    crop[src_y0 - sy0:src_y0 - sy0 + (src_y1 - src_y0), src_x0 - sx0:src_x0 - sx0 + (src_x1 - src_x0)] = (
        layer[src_y0:src_y1, src_x0:src_x1]
    )
    rad_ang = math.radians(ang)
    c, s = math.cos(rad_ang), math.sin(rad_ang)

    def map_xy(rx, ry, ppx, ppy):
        return ppx + c * rx + s * ry, ppy - s * rx + c * ry

    turned = _xform_nearest(crop, (px - sx0, py - sy0), map_xy)
    out = np.zeros_like(layer)
    dy0 = max(0, sy0)
    dx0 = max(0, sx0)
    dy1 = min(layer.shape[0], sy0 + side)
    dx1 = min(layer.shape[1], sx0 + side)
    if dy1 <= dy0 or dx1 <= dx0:
        return out
    out[dy0:dy1, dx0:dx1] = turned[dy0 - sy0:dy0 - sy0 + (dy1 - dy0), dx0 - sx0:dx0 - sx0 + (dx1 - dx0)]
    return out


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


def contact_pivot(part, body, toward):
    """Joint on the part edge that touches the body, biased toward a side."""
    touch = part & dilate(body, 3)
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

    torso_m = placed["torso"]
    for name in ("stinger", "left_claw", "right_claw", "jaw", "leg_l", "leg_ml", "leg_r", "leg_fr"):
        touch = placed[name] & dilate(torso_m, 1)
        if np.any(touch):
            layers["torso"][touch] = layers[name][touch]

    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    # Unflipped still is the front. Flipped rows mirror the snap.
    facing = -1.0 if flip else 1.0
    body = placed["torso"] | placed["stinger"] | placed["jaw"]
    if facing > 0:
        left_side, right_side = "right", "left"
    else:
        left_side, right_side = "left", "right"
    pivots = {
        "stinger": contact_pivot(placed["stinger"], placed["torso"], "low"),
        "left_claw": contact_pivot(placed["left_claw"], body, left_side),
        "right_claw": contact_pivot(placed["right_claw"], body, right_side),
        "body": center,
    }
    jys, jxs = np.nonzero(placed["jaw"])
    top = int(jys.min())
    band = jys <= top + 1
    pivots["jaw"] = (float(jxs[band].mean()), float(top))
    return layers, pivots, center, facing, canvas


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Sine that is exactly 0 on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0,
        jaw=0, sting=0.0, lclaw=0.0, rclaw=0.0,
        lin=0.0, rin=0.0,
        l_dx=0.0, l_dy=0.0,
        ml_dx=0.0, ml_dy=0.0,
        r_dx=0.0, r_dy=0.0,
        fr_dx=0.0, fr_dy=0.0,
        spin=0.0, recoil=0.0,
        ring=-1, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["dy"] = wave(f, n, -1.0)
        p["sting"] = wave(f, n, 3.5, phase=0.5)
        p["lclaw"] = wave(f, n, 2.5, phase=0.9)
        p["rclaw"] = wave(f, n, 2.5, phase=0.9)
    elif action == "walk":
        step = wave(f, n, 1.0, cycles=2.0)
        p["dy"] = wave(f, n, -1.0, cycles=2.0)
        p["sting"] = wave(f, n, 4.0, cycles=2.0, phase=0.6)
        p["lclaw"] = wave(f, n, 5.0, cycles=2.0, phase=0.4)
        p["rclaw"] = wave(f, n, -5.0, cycles=2.0, phase=0.4)
        # Four measured feet. Diagonal pairs. Lift is 2px. Planted at frame 0.
        lift_a = -max(0.0, step * 2.0)
        lift_b = -max(0.0, -step * 2.0)
        p["l_dy"] = lift_a
        p["fr_dy"] = lift_a
        p["l_dx"] = -max(0.0, step)
        p["fr_dx"] = max(0.0, step)
        p["ml_dy"] = lift_b
        p["r_dy"] = lift_b
        p["ml_dx"] = -max(0.0, -step)
        p["r_dx"] = max(0.0, -step)
    elif action == "attack":
        # 0-4 claw snap and a small jaw, 5-10 spin, 11-12 impact, 13-15 recover.
        p["jaw"] = [0, 1, 2, 3, 2, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0][f]
        p["sting"] = [0, 3, 6, 8, 4, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0][f]
        # Negative drops the left tip. Positive drops the right tip.
        p["lclaw"] = [0, -6, -12, -16, -8, 0, 0, 0, 0, 0, 0, 5, 2, 0, 0, 0][f]
        p["rclaw"] = [0, 6, 12, 16, 8, 0, 0, 0, 0, 0, 0, -5, -2, 0, 0, 0][f]
        p["lin"] = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["rin"] = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        if 2 <= f <= 4:
            p["ring"] = f - 2
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
        if f in (11, 12):
            p["burst"] = 2 if f == 11 else 1
    else:
        env = [0.0, 0.45, 0.80, 1.0, 0.82, 0.55, 0.32, 0.16, 0.05, 0.0, 0.0, 0.0][f]
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


def paint_jaw_wedge(img, jaw_closed, jaw_open):
    """Magenta wedge in the rows the existing jaw just left. No new colours."""
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
    xs = np.nonzero(gap)[1]
    x_lo, x_hi = int(xs.min()), int(xs.max())
    edge = gap.copy()
    edge[:] = False
    edge[:, x_lo] = gap[:, x_lo]
    edge[:, x_hi] = gap[:, x_hi]
    img[gap] = (*MAG, 255)
    src = jaw_closed
    img[edge] = src[edge]
    img[edge, 3] = 255
    return img


def compose(layers, pivots, p, facing):
    """Feet do not inherit the body bob, so a planted foot stays on the ground."""
    bdx = p["dx"] * facing
    bdy = p["dy"]
    torso = shift(layers["torso"], bdx, bdy)
    stinger = moved_part(layers["stinger"], p["sting"] * facing, pivots["stinger"], bdx, bdy)
    left = moved_part(
        layers["left_claw"], p["lclaw"] * facing, pivots["left_claw"],
        bdx + p["lin"] * facing, bdy,
    )
    right = moved_part(
        layers["right_claw"], p["rclaw"] * facing, pivots["right_claw"],
        bdx - p["rin"] * facing, bdy,
    )
    jaw_closed = shift(layers["jaw"], bdx, bdy)
    jaw = shift(jaw_closed, 0, int(p["jaw"]))
    leg_l = shift(layers["leg_l"], p["l_dx"] * facing, p["l_dy"])
    leg_ml = shift(layers["leg_ml"], p["ml_dx"] * facing, p["ml_dy"])
    leg_r = shift(layers["leg_r"], p["r_dx"] * facing, p["r_dy"])
    leg_fr = shift(layers["leg_fr"], p["fr_dx"] * facing, p["fr_dy"])
    body = np.zeros_like(torso)
    for lyr in (leg_fr, leg_r, leg_ml, leg_l, torso, jaw, stinger, right, left):
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
    """Centre of the open mouth, then a step in front of it (down, frontal)."""
    red = np.zeros(img.shape[:2], bool)
    for color in (MAG, PINK):
        red |= is_color(img, color)
    ys, xs = np.nonzero(red)
    if len(xs) == 0:
        bb = body_box(img)
        if not bb:
            return CELL_W // 2, CELL_H // 2
        return (bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2
    # Prefer the lower mouth, not the pink bulb on the tail.
    low = ys >= np.percentile(ys, 60)
    xs, ys = xs[low], ys[low]
    return int(np.median(xs)), int(np.median(ys)) + 2


def roar_rings(img, origin, age):
    """Sparse rings in front of the mouth. They are not part of the body."""
    ox, oy = origin
    cols = (WHITE, MAG, CYAN)
    for ri in range(age + 1):
        rad = 8 + ri * 4
        col = cols[ri]
        for deg in range(-50, 51, 16):
            ang = math.radians(deg)
            put(img, ox + rad * math.sin(ang), oy + rad * math.cos(ang) * 0.72, col)
    return img


def impact_burst(img, origin, size):
    ox, oy = origin
    pts = [(0, 6), (-4, 8), (4, 8), (-7, 4), (7, 4), (0, 12), (-3, 3), (3, 3)]
    cols = (WHITE, MAG, CYAN, PALE)
    for i, (dx, dy) in enumerate(pts[: 4 + size * 2]):
        put(img, ox + dx, oy + dy, cols[i % 4])
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
        put(img, sx + facing, sy, CYAN)
    return img


def flash_hit(img):
    """One-frame blink. Body purples go light. Mouth, claws, eyes, and ink stay."""
    out = img.copy()
    opaque = out[..., 3] >= 128
    rgb = out[..., :3]
    keep = np.zeros(opaque.shape, bool)
    for color in (INK, WHITE, PALE, GRAY, CYAN, TEAL, MAG, PINK, VDARK):
        keep |= opaque & np.all(rgb == np.array(color, np.uint8), axis=-1)
    out[opaque & ~keep, :3] = np.array(LPUR, np.uint8)
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
        img = roar_rings(img, mouth_point(img, facing), p["ring"])
    if p["burst"]:
        img = impact_burst(img, mouth_point(img, facing), p["burst"])
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
    """South row, every frame, cropped. Claws, spin, steps, and the lean."""
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
        "dex": 452,
        "form": "normal",
        "variant": "drapion",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0452_drapion/normal",
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
            if keys.count(KEY) != 1:
                raise SystemExit(f"{KEY} key count is {keys.count(KEY)}")
            frozen = {k: _key_blob(text, k) for k in FORBIDDEN_KEYS}
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
            written = DIR8.read_text()
            for k, old in frozen.items():
                if _key_blob(written, k) != old:
                    raise SystemExit(f"dir8 wrote over {k}")
            if _key_blob(written, KEY) != f'"{KEY}":' + blob:
                raise SystemExit("dir8 drapion entry mismatch after write")
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
            start = text.index('"id": "0452_drapion"')
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
    raise SystemExit("0452_drapion missing from queue")


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
        "feet": 4,
        "cell": [CELL_W, CELL_H, FOOT_Y],
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
        and 58 <= sprite_h <= 68
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
    print(
        "parts", info["parts"],
        "palette", len(palette),
        "facing", info["facing"],
        "wh", info["sprite_wh"],
        "feet", info["feet"],
        flush=True,
    )
    if info["sprite_wh"][1] < 58 or info["sprite_wh"][0] < 70:
        raise SystemExit(f"sprite {info['sprite_wh']} is not adult drapion")

    rigs = {}
    for d in DIRS:
        rigs[d] = build_rig(front, masks, flip=d in FLIP_DIRS)

    rest = synth(rigs["S"], "idle", 0, palette, effects=False)
    canvas = rigs["S"][4]
    rest_diff = int(np.any(rest != canvas, axis=2).sum())
    rest_w = synth(rigs["W"], "idle", 0, palette, effects=False)
    rest_w_diff = int(np.any(rest_w != rigs["W"][4], axis=2).sum())
    print("rest-pose pixels changed", rest_diff, "west", rest_w_diff, flush=True)
    if rest_diff or rest_w_diff:
        raise SystemExit("frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (80, 140, 210),
        "stinger": (240, 200, 40),
        "left_claw": (255, 80, 40),
        "right_claw": (255, 40, 180),
        "leg_l": (40, 220, 80),
        "leg_ml": (80, 255, 180),
        "leg_r": (40, 80, 255),
        "leg_fr": (180, 80, 255),
        "jaw": (255, 0, 80),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    ratios = attack_report(rigs["S"], palette)
    if any(r < 0.90 or r > 1.08 for r in ratios):
        print("attack ratios outside 0.90-1.08", flush=True)
        if args.samples or args.no_publish:
            raise SystemExit(2)
        update_queue("needs_fix")
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
    print(
        f"FEET {info['feet']} cell {CELL_W}x{CELL_H} fy {FOOT_Y} sprite_h {sprite_h} "
        f"parts {info['parts']}",
        flush=True,
    )

    meta = {
        "generator": "forge_drapion_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 452,
        "variant": "drapion",
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
        "feet": 4,
        "claws": 2,
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
        print("untouched keys", ",".join(FORBIDDEN_KEYS), flush=True)


if __name__ == "__main__":
    main()
