#!/usr/bin/env python3
"""Abomasnow (0460) normal 8-direction action sheet from this form's front.png.

Measured on normal/front.png only: 3006 opaque pixels, bbox x8-87 y14-82
(80x69), 14 colours, alpha already 0 or 255. Adult biped, frontal. Both
purple eyes are drawn (left 20 px around x38.7, right 29 px around x49.6).
Eye centroid x=45.1 sits 2.4 px left of the body centroid. That is a front
view, not a profile, so the still is South. Unflipped dirs are S SE E NE.
N NW W SW are mirrored.

Two feet, both green pine shapes under the cloak, split by a gap:
  foot_l x24-42 y61-78, foot_r x51-69 y61-82.
The outer pine masses are the arms (arm_l x about 8-27, arm_r x about 68-87).
The white snow cloak and the arms are not extra legs. No mouth gap, so the
attack display swings the existing arms and draws rings. No new jaw.

  idle   : arms sway. Frame 0 is the still. Both soles stay planted.
  walk   : the two feet alternate. Arms swing as arms. Cloak stays one mass.
  attack : 0-4 arm display plus rings, 5-10 one-piece in-plane 360,
           11-12 impact, 13-15 the still. No |cos| squash.
  hurt   : one-piece lean, at most 36 degrees. No |cos| squash.

Palette-locked to this still. front/back png and gif are not written.
The mega form and 0460_abomasnow-mega are not opened or edited.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0460_abomasnow/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0460_abomasnow/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
UNFLIPPED = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "abomasnow_v2"
KEY = "0460_abomasnow"
MEGA_KEY = "0460_abomasnow-mega"

INK = (32, 41, 49)
WHITE = (255, 255, 255)
SNOW = (197, 213, 222)
MIST = (172, 189, 205)
SLATE = (98, 115, 131)
STEEL = (148, 156, 180)
LEAF = (82, 164, 148)
MOSS = (41, 131, 115)
PINE = (32, 98, 90)
BARK = (41, 82, 90)
MINT = (156, 230, 205)
LILAC = (222, 164, 246)
VIOLET = (156, 115, 230)
PLUM = (98, 65, 189)
PALETTE_COLS = (
    INK, WHITE, SNOW, MIST, SLATE, STEEL, LEAF, MOSS, PINE, BARK, MINT, LILAC, VIOLET, PLUM,
)
GREEN_COLS = (LEAF, MOSS, PINE, BARK, MINT)
PURPLE_COLS = (LILAC, VIOLET, PLUM)
DRAW = ("torso", "foot_l", "foot_r", "arm_l", "arm_r")
RING = ((-14, -16), (-8, -18), (0, -20), (8, -18), (14, -16), (-18, -10), (18, -10), (0, -24))
BURST = ((-16, -8), (-10, -14), (0, -16), (10, -14), (16, -8), (-12, 6), (12, 6), (0, 8))


def flood_components(mask):
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append(pts)
    return out


def components(mask):
    out = []
    for pts in flood_components(mask):
        xs2 = [p[1] for p in pts]
        ys2 = [p[0] for p in pts]
        out.append((len(pts), min(xs2), min(ys2), max(xs2), max(ys2)))
    return out


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        pad = np.pad(out, 1)
        out = out | pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    return out


def colored(img, cols):
    rgb = img[..., :3]
    a = img[..., 3] == 255
    hit = np.zeros(a.shape, dtype=bool)
    for col in cols:
        hit |= np.all(rgb == np.array(col, np.uint8), axis=-1)
    return a & hit


def segment(img):
    """Two pine feet, two pine arms. Snow cloak stays on the torso."""
    h, w = img.shape[:2]
    a = img[..., 3] == 255
    if int(a.sum()) != 3006:
        raise SystemExit(f"still opaque changed: {int(a.sum())}")
    if img.shape[0] != 96 or img.shape[1] != 96:
        raise SystemExit("front is not 96x96")
    ys, xs = np.nonzero(a)
    if int(xs.min()) != 8 or int(xs.max()) != 87 or int(ys.min()) != 14 or int(ys.max()) != 82:
        raise SystemExit("still bbox changed; remeasure before forging")
    green = colored(img, GREEN_COLS)
    ink = colored(img, (INK,))
    purple = colored(img, PURPLE_COLS)
    if int(purple.sum()) != 49:
        raise SystemExit(f"face purple changed: {int(purple.sum())}")
    parts = {k: np.zeros_like(a) for k in ("arm_l", "arm_r", "foot_l", "foot_r")}
    large, small = [], []
    for pts in flood_components(green):
        n = len(pts)
        cx = sum(p[1] for p in pts) / n
        cy = sum(p[0] for p in pts) / n
        (large if n >= 80 else small).append((n, cx, cy, pts))
    if len(large) != 4:
        raise SystemExit(f"expected 4 pine masses, got {len(large)}")
    for _n, cx, cy, pts in large:
        if cy < 64 and cx < 40:
            name = "arm_l"
        elif cy < 64 and cx > 55:
            name = "arm_r"
        elif cx < 48:
            name = "foot_l"
        else:
            name = "foot_r"
        for y, x in pts:
            parts[name][y, x] = True
    cent = {}
    for name, mask in parts.items():
        py, px = np.nonzero(mask)
        if len(px) == 0:
            raise SystemExit(f"missing {name}")
        cent[name] = (float(px.mean()), float(py.mean()))
    for _n, cx, cy, pts in small:
        best = min(cent, key=lambda k: (cx - cent[k][0]) ** 2 + (cy - cent[k][1]) ** 2)
        dist = math.hypot(cx - cent[best][0], cy - cent[best][1])
        if dist <= 12:
            for y, x in pts:
                parts[best][y, x] = True
    trans = ~a
    for name, mask in list(parts.items()):
        others = np.zeros_like(a)
        for other, other_mask in parts.items():
            if other != name:
                others |= other_mask
        border = dilate(mask, 1) & ink & dilate(trans, 1) & ~dilate(others, 1)
        parts[name] = mask | border
    used = np.zeros_like(a)
    for mask in parts.values():
        used |= mask
    torso = a & ~used
    for pts in flood_components(torso):
        if len(pts) > 100:
            continue
        cy = sum(p[0] for p in pts) / len(pts)
        cx = sum(p[1] for p in pts) / len(pts)
        best, dist = None, 1e9
        for name, mask in parts.items():
            py, px = np.nonzero(mask)
            dmin = ((py - cy) ** 2 + (px - cx) ** 2).min()
            if dmin < dist:
                dist = dmin
                best = name
        if dist <= 2.5 ** 2:
            for y, x in pts:
                parts[best][y, x] = True
                torso[y, x] = False
    parts["torso"] = torso
    expect = {"arm_l": 257, "arm_r": 296, "foot_l": 188, "foot_r": 229, "torso": 2036}
    expect_comps = {"arm_l": 1, "arm_r": 1, "foot_l": 1, "foot_r": 2, "torso": 1}
    used = np.zeros_like(a)
    for name in DRAW:
        mask = parts[name]
        info = components(mask)
        if len(info) != expect_comps[name] or int(mask.sum()) != expect[name]:
            raise SystemExit(f"{name} mask changed: n={int(mask.sum())} comps={info}")
        if np.any(used & mask):
            raise SystemExit(f"overlap at {name}")
        used |= mask
    if int(used.sum()) != 3006:
        raise SystemExit(f"coverage changed: {int(used.sum())}")
    if int((green & parts["torso"]).sum()) != 0:
        raise SystemExit("green remained on the snow cloak")
    yy = np.mgrid[0:h, 0:w][0]
    ground = np.zeros_like(a)
    for name in ("foot_l", "foot_r"):
        py = np.nonzero(parts[name])[0]
        ground |= parts[name] & (yy >= int(py.max()) - 1)
    if len(components(ground)) != 2:
        raise SystemExit(f"expected 2 feet, got {components(ground)}")
    eye_x = float(np.nonzero(purple)[1].mean())
    body_x = float(xs.mean())
    if not (eye_x < body_x and abs(eye_x - body_x) < 6):
        raise SystemExit(f"facing changed: eye {eye_x} body {body_x}")
    return parts


def edge_pivot(mask, side):
    ys, xs = np.nonzero(mask)
    if side == "bottom":
        sel = ys >= ys.max() - 1
    elif side == "top":
        sel = ys <= ys.min() + 1
    elif side == "left":
        sel = xs <= xs.min() + 1
    else:
        sel = xs >= xs.max() - 1
    return float(xs[sel].mean()), float(ys[sel].mean())


def contact_pivot(part, torso):
    contact = part & dilate(torso, 1)
    if not np.any(contact):
        return edge_pivot(part, "top")
    ys, xs = np.nonzero(contact)
    top = ys <= ys.min() + 2
    return float(xs[top].mean()), float(ys[top].mean())


def extract_palette(img):
    opaque = img[..., 3] == 255
    return np.unique(img[opaque][:, :3], axis=0)


def require_colors(palette):
    if len(palette) != 14:
        raise SystemExit(f"expected 14 colours, got {len(palette)}")
    for col in PALETTE_COLS:
        if not np.any(np.all(palette == np.array(col, np.uint8), axis=1)):
            raise SystemExit(f"still is missing measured colour {col}")


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    for i in range(0, len(ty), 300):
        sl = slice(i, i + 300)
        dist = (ty[sl, None] - pts[None, :, 0]) ** 2 + (tx[sl, None] - pts[None, :, 1]) ** 2
        pick[sl] = dist.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def build_rig(img, masks, flip):
    opaque = img[..., 3] == 255
    ys, xs = np.nonzero(opaque)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    placed = {}
    for name, mask in masks.items():
        pm = np.zeros((CELL_H, CELL_W), dtype=bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = mask[y0:y1, x0:x1]
        placed[name] = pm
    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr
    for part, rad in (("arm_l", 2), ("arm_r", 2), ("foot_l", 2), ("foot_r", 2)):
        zone = placed[part] & dilate(placed["torso"], rad)
        fill = nearest_color(canvas, zone, placed["torso"])
        write = zone & (layers["torso"][..., 3] == 0)
        layers["torso"][write] = fill[write]
    pivots = {
        "arm_l": contact_pivot(placed["arm_l"], placed["torso"]),
        "arm_r": contact_pivot(placed["arm_r"], placed["torso"]),
        "foot_l": contact_pivot(placed["foot_l"], placed["torso"]),
        "foot_r": contact_pivot(placed["foot_r"], placed["torso"]),
    }
    oys, oxs = np.nonzero(canvas[..., 3] == 255)
    center = (float(oxs.mean()), float(oys.mean()))
    soles = {}
    for name in ("foot_l", "foot_r"):
        sy, sx = np.nonzero(placed[name])
        yb = int(sy.max())
        soles[name] = [(int(x), yb) for x in sx[sy == yb]]
    sole_pts = soles["foot_l"] + soles["foot_r"]
    sole = (float(sum(x for x, _y in sole_pts) / len(sole_pts)), float(max(y for _x, y in sole_pts)))
    purple = colored(canvas, PURPLE_COLS)
    py, px = np.nonzero(purple)
    face = (float(px.mean()), float(py.mean()))
    outward = {}
    for name in ("foot_l", "foot_r"):
        fx = np.nonzero(placed[name])[1]
        outward[name] = -1.0 if float(fx.mean()) < center[0] else 1.0

    if flip:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}

        def mx(x):
            return CELL_W - 1 - x

        pivots = {k: (mx(x), y) for k, (x, y) in pivots.items()}
        center = (mx(center[0]), center[1])
        sole = (mx(sole[0]), sole[1])
        face = (mx(face[0]), face[1])
        soles = {k: [(mx(x), y) for x, y in pts] for k, pts in soles.items()}
        outward = {k: -v for k, v in outward.items()}
        facing = -1.0
    else:
        facing = 1.0
    rest = compose(layers)
    if np.any(rest != canvas):
        raise SystemExit("rest pose does not match the still")
    return {
        "layers": layers,
        "pivots": pivots,
        "center": center,
        "sole": sole,
        "face": face,
        "facing": facing,
        "rest": canvas,
        "soles": soles,
        "outward": outward,
        "counts": {k: int(np.count_nonzero(layers[k][..., 3] == 255)) for k in DRAW},
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


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest in-plane rotate. 0 and 360 stay exact. No |cos| squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.05 or not np.any(layer[..., 3] == 255):
        return layer.copy()
    scale = 4
    up = np.repeat(np.repeat(layer, scale, axis=0), scale, axis=1)
    px, py = pivot[0] * scale + 1.5, pivot[1] * scale + 1.5
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
    return out[scale // 2::scale, scale // 2::scale].copy()


def despeckle(img):
    """Drop a pixel only when all four neighbours are empty. After rotation only."""
    out = img.copy()
    opaque = out[..., 3] == 255
    pad = np.pad(opaque, 1)
    n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    out[opaque & (n == 0)] = 0
    return out


def fill_pinholes(img, target):
    """Fill interior holes back toward the still count. Do not grow past it."""
    out = img.copy()
    h, w = out.shape[:2]
    for need in (4, 3, 3):
        opaque = out[..., 3] == 255
        have = int(opaque.sum())
        if have >= target:
            break
        pad = np.pad(opaque, 1)
        n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        holes = (~opaque) & (n >= need)
        ys, xs = np.nonzero(holes)
        if len(ys) == 0:
            continue
        room = target - have
        if len(ys) > room:
            ys, xs = ys[:room], xs[:room]
        for y, x in zip(ys.tolist(), xs.tolist()):
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and out[ny, nx, 3] == 255:
                    out[y, x] = out[ny, nx]
                    break
    return out


def rotate_part(layer, angle, pivot, target):
    spun = rotate_nearest(layer, angle, pivot)
    if abs(angle) % 360.0 < 0.05:
        return spun
    return despeckle(fill_pinholes(spun, target))


def snap(img, palette):
    out = img.copy()
    opaque = out[..., 3] == 255
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


def paste(dst, layer):
    mask = layer[..., 3] == 255
    dst[mask] = layer[mask]


def compose(layers):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in DRAW:
        paste(img, layers[name])
    return img


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero at frame 0 and at frame n, so the loop closes on the still."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        arm_l=0.0, arm_r=0.0, lift_l=0.0, lift_r=0.0, fwd_l=0.0, fwd_r=0.0,
        spin=0.0, recoil=0.0, knock=0.0, lunge=0, rings=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["arm_l"] = wave(f, n, 4.0)
        p["arm_r"] = wave(f, n, 3.5, phase=0.8)
    elif action == "walk":
        step = wave(f, n, 1.0, cycles=2.0)
        p["lift_l"] = max(0.0, step) * 4.0
        p["lift_r"] = max(0.0, -step) * 4.0
        p["fwd_l"] = max(0.0, step) * 2.0
        p["fwd_r"] = max(0.0, -step) * 2.0
        p["arm_l"] = -step * 10.0
        p["arm_r"] = -step * 10.0
    elif action == "attack":
        if f <= 4:
            p["arm_l"] = (8, 14, 18, 12, 6)[f]
            p["arm_r"] = (-8, -14, -18, -12, -6)[f]
            p["rings"] = 1
        elif f <= 10:
            p["spin"] = 60.0 * (f - 4)
        elif f in (11, 12):
            p["arm_l"] = (-12, -6)[f - 11]
            p["arm_r"] = (12, 6)[f - 11]
            p["burst"] = 1
            p["lunge"] = 2
    else:
        env = math.sin(math.pi * f / 11.0)
        p["recoil"] = 36.0 * env
        p["knock"] = 2.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def dots(out, origin, table, cols):
    if origin is None:
        return out
    ox, oy = origin
    for i, (dx, dy) in enumerate(table):
        put(out, ox + dx, oy + dy, cols[i % len(cols)])
    return out


def flash_white(img):
    """One-frame hit. Keep the ink outline so the pine arms and cloak stay readable."""
    out = img.copy()
    opaque = out[..., 3] == 255
    ink = np.all(out[..., :3] == np.array(INK, np.uint8), axis=-1)
    out[opaque & ~ink, :3] = WHITE
    return out


def sparks(out, facing, frame_idx):
    opaque = out[..., 3] == 255
    ys, xs = np.nonzero(opaque)
    if len(xs) == 0:
        return out
    x = int(xs.mean())
    y = int(ys.min()) + 4
    for i, col in enumerate((WHITE, LILAC, MINT)):
        put(out, x - facing * (4 + i + frame_idx), y + i * 2, col)
    return out


def synth(rig, action, f, palette):
    layers, pivots = rig["layers"], rig["pivots"]
    facing = rig["facing"]
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = (p["spin"] % 360.0) > 0.5
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        if spinning:
            img = rotate_nearest(rig["rest"], p["spin"], rig["center"])
        else:
            img = rotate_nearest(rig["rest"], facing * p["recoil"], rig["sole"])
            img = shift(img, -facing * p["knock"], 0)
        ang = p["spin"] if spinning else p["recoil"]
        if ang % 360.0 > 0.5:
            img = despeckle(fill_pinholes(img, rig["counts"]["body"]))
    else:
        moved = {
            "torso": layers["torso"],
            "foot_l": shift(layers["foot_l"], p["fwd_l"] * rig["outward"]["foot_l"], -p["lift_l"]),
            "foot_r": shift(layers["foot_r"], p["fwd_r"] * rig["outward"]["foot_r"], -p["lift_r"]),
            "arm_l": rotate_part(layers["arm_l"], p["arm_l"], pivots["arm_l"], rig["counts"]["arm_l"]),
            "arm_r": rotate_part(layers["arm_r"], p["arm_r"], pivots["arm_r"], rig["counts"]["arm_r"]),
        }
        img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        for name in DRAW:
            paste(img, moved[name])
        if p["lunge"]:
            img = shift(img, facing * p["lunge"], 0)
    img = snap(img, palette)
    ring_cols = (WHITE, LILAC, MINT, LEAF)
    if p["flash"]:
        img = flash_white(img)
    if p["rings"] and not spinning and not recoiling:
        img = dots(img, rig["face"], RING, ring_cols)
    if p["burst"]:
        img = dots(img, rig["face"], BURST, ring_cols)
    if p["sparks"] and recoiling:
        img = sparks(img, facing, f)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (SRC_DIR / "front.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def one_row(rig, palette, action):
    sheet = np.zeros((CELL_H, FRAMES[action] * CELL_W, 4), np.uint8)
    for f in range(FRAMES[action]):
        sheet[0:CELL_H, f * CELL_W:(f + 1) * CELL_W] = synth(rig, action, f, palette)
    return sheet


def assemble(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, direction in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = np.array(Image.open(frames_dir / act / direction / f"{f:02d}.png"))
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


def contact_sheet(sheets, row):
    probe = sheets["idle"][row * CELL_H:(row + 1) * CELL_H, :CELL_W]
    ys, xs = np.nonzero(probe[..., 3] == 255)
    pad = 28
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad)
    scale = 2
    cw, ch = (x1 - x0) * scale, (y1 - y0) * scale
    cols = 8
    blocks = []
    for act in ACTIONS:
        band = sheets[act][row * CELL_H:(row + 1) * CELL_H]
        frames = []
        for f in range(FRAMES[act]):
            tile = band[y0:y1, f * CELL_W + x0:f * CELL_W + x1]
            frames.append(np.repeat(np.repeat(tile, scale, 0), scale, 1))
        blocks.append(frames)
    rows = sum(math.ceil(len(b) / cols) for b in blocks)
    bg = np.full((rows * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    r = 0
    for frames in blocks:
        for i, up in enumerate(frames):
            rr, cc = divmod(i, cols)
            y, x = (r + rr) * ch, cc * cw
            mask = up[..., 3] == 255
            bg[y:y + ch, x:x + cw][mask] = up[mask]
        r += math.ceil(len(frames) / cols)
    return Image.fromarray(bg, "RGBA")


def south_attack_ratios(sheets):
    base_tile = sheets["attack"][:CELL_H, :CELL_W]
    base = int((base_tile[..., 3] == 255).sum())
    ratios = []
    counts = []
    for f in range(FRAMES["attack"]):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        n = int((tile[..., 3] == 255).sum())
        counts.append(n)
        ratios.append(n / base if base else 0.0)
    return counts, ratios


def largest_fraction(img):
    opaque = img[..., 3] == 255
    if not np.any(opaque):
        return 1.0
    best = 0
    total = int(opaque.sum())
    seen = np.zeros_like(opaque)
    h, w = opaque.shape
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
        best = max(best, n)
    return best / total


def save_parts(img, masks, path):
    cols = {
        "arm_l": (255, 80, 80),
        "arm_r": (255, 180, 80),
        "foot_l": (80, 160, 255),
        "foot_r": (80, 220, 120),
        "torso": (70, 70, 90),
    }
    part = np.zeros((*img.shape[:2], 4), np.uint8)
    for name, col in cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    big = np.repeat(np.repeat(part, 4, 0), 4, 1)
    still = np.repeat(np.repeat(img, 4, 0), 4, 1)
    gap = np.zeros((big.shape[0], 8, 4), np.uint8)
    Image.fromarray(np.concatenate([still, gap, big], axis=1), "RGBA").save(path)


def with_flock(lock_path, fn):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def object_span(text, key):
    token = f'"{key}":'
    start = text.index(token)
    i = start + len(token)
    while text[i].isspace():
        i += 1
    if text[i] != "{":
        raise SystemExit(f"{key} value is not an object")
    depth = 0
    end = i
    in_str = False
    esc = False
    while end < len(text):
        ch = text[end]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return start, end + 1
        end += 1
    raise SystemExit(f"unclosed {key}")


def publish_dir8(sprite_h):
    entry = {
        "dex": 460,
        "form": "normal",
        "variant": "abomasnow",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0460_abomasnow/normal",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir",
        "nf": 16,
        "cw": CELL_W,
        "ch": CELL_H,
        "w": CELL_W,
        "h": int(sprite_h),
        "fy": FOOT_Y,
        "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1,
        "src": "agent-sprite-forge",
        "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix) or "\n" in text.strip():
            raise SystemExit("dir8.js is not a single assignment line")
        before = json.loads(text[len(prefix):].rstrip().rstrip(";"))
        if KEY not in before:
            raise SystemExit("dir8 has no 0460_abomasnow")
        if before[MEGA_KEY].get("v") != "abomasnow_mega_v1":
            raise SystemExit("mega key is not abomasnow_mega_v1 before edit")
        key_at, val_end = object_span(text, KEY)
        val_start = text.index("{", key_at)
        text = text[:val_start] + blob + text[val_end:]
        if not text.endswith("\n"):
            text += "\n"
        if "\n" in text.strip():
            raise SystemExit("dir8.js became multi-line")
        after = json.loads(text[len(prefix):].rstrip().rstrip(";"))
        if set(after) != set(before):
            raise SystemExit("dir8 key set changed")
        for k, old in before.items():
            if k == KEY:
                continue
            if after[k] != old:
                raise SystemExit(f"dir8 neighbor changed: {k}")
        if after[MEGA_KEY] != before[MEGA_KEY] or after[MEGA_KEY].get("v") != "abomasnow_mega_v1":
            raise SystemExit("mega key changed")
        got = after[KEY]
        if got != entry:
            raise SystemExit("dir8 entry did not stick")
        if got["v"] != VERSION or got["layout"] != "actions_8dir" or got["an"] != [16, 16, 16, 12]:
            raise SystemExit("dir8 entry did not stick")
        if got["raw"] != 1 or got["sheetDir"] != "agent-sprite-forge" or got["h"] != int(sprite_h):
            raise SystemExit("dir8 entry fields mismatch")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(text)
        tmp.replace(DIR8)
        return True

    return with_flock(DIR8_LOCK, edit)


def update_queue(status):
    def edit():
        fd = os.open(QUEUE, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with os.fdopen(os.dup(fd), "r+", encoding="utf-8") as handle:
                raw = handle.read()
                data = json.loads(raw)
                found = False
                for item in data["items"]:
                    if item.get("id") != KEY:
                        continue
                    item["status"] = status
                    item.pop("claim", None)
                    found = True
                if not found:
                    raise SystemExit("queue has no 0460_abomasnow")
                before = json.loads(raw)
                for b, a in zip(before["items"], data["items"]):
                    if b.get("id") == KEY:
                        continue
                    if b != a:
                        raise SystemExit(f"queue neighbor changed: {b.get('id')}")
                text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
                after = json.loads(text)
                for b, a in zip(before["items"], after["items"]):
                    if b.get("id") == KEY:
                        if a.get("status") != status or "claim" in a:
                            raise SystemExit("queue status did not stick")
                        continue
                    if b != a:
                        raise SystemExit(f"queue neighbor changed: {b.get('id')}")
                tmp = QUEUE.with_suffix(".json.tmp")
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def qa_report(master, sheets, palette, ratios, sprite_h):
    opaque = master[..., 3] == 255
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    off = []
    if np.any(opaque):
        cols = np.unique(master[opaque][:, :3], axis=0)
        pal = {tuple(int(c) for c in row) for row in palette}
        off = [tuple(int(c) for c in row) for row in cols if tuple(int(c) for c in row) not in pal]
    clip = 0
    for act in ACTIONS:
        for di in range(8):
            for f in range(FRAMES[act]):
                tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = tile[..., 3] == 255
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip += 1
    spin_body = []
    for f in range(5, 11):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        spin_body.append({"f": f, "main": round(largest_fraction(tile), 4), "ratio": round(ratios[f], 4)})
    rest = sheets["idle"][:CELL_H, :CELL_W]
    hurt = sheets["hurt"][:CELL_H, 5 * CELL_W:6 * CELL_W]

    def width(tile):
        xs = np.nonzero(tile[..., 3] == 255)[1]
        return int(xs.max() - xs.min() + 1) if len(xs) else 0

    hurt_squash = width(hurt) < int(width(rest) * 0.75)
    bad_ratio = [round(r, 4) for r in ratios if r < 0.90 or r > 1.08]
    idle0 = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(idle0[..., 3] == 255)[0]
    h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    return {
        "alphas": alphas,
        "off_palette": len(off),
        "clip_frames": clip,
        "sprite_h": h,
        "hurt_squash": bool(hurt_squash),
        "bad_ratio": bad_ratio,
        "spin_body": spin_body,
        "h_matches": h == sprite_h,
    }


def soles_hold(img, rest, pts):
    for x, y in pts:
        if img[y, x, 3] != 255 or not np.array_equal(img[y, x], rest[y, x]):
            return False
    return True


def print_ratios(counts, ratios, sheets):
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(counts, ratios)):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        print(
            f"attack S {f:02d} opaque {c} ratio {r:.4f} main {largest_fraction(tile):.3f}",
            flush=True,
        )
    print(f"min {min(ratios):.4f} max {max(ratios):.4f}", flush=True)


def check_row(sheets, label):
    counts_a, ratios = south_attack_ratios(sheets)
    if label == "S":
        print_ratios(counts_a, ratios, sheets)
    bad = [round(r, 4) for r in ratios if r < 0.90 or r > 1.08]
    if bad:
        raise SystemExit(f"{label} attack opaque ratio out of range at {bad}")
    weak = [
        f for f in range(5, 11)
        if largest_fraction(sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]) < 0.90
    ]
    if weak:
        raise SystemExit(f"{label} spin shattered at frames {weak}")
    return counts_a, ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if set(np.unique(front[..., 3]).tolist()) - {0, 255}:
        raise SystemExit("front alpha is not binary")
    palette = extract_palette(front)
    require_colors(palette)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("feet 2 facing front parts", counts, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    body_n = int((rigs["S"]["rest"][..., 3] == 255).sum())
    if body_n != 3006:
        raise SystemExit(f"placed still count changed: {body_n}")
    for rig in rigs.values():
        rig["counts"]["body"] = body_n

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    save_parts(front, masks, work / "parts.png")

    for label, rig in rigs.items():
        rest = synth(rig, "idle", 0, palette)
        if np.any(rest != rig["rest"]):
            raise SystemExit(f"{label} idle frame 0 does not match the still")
        for f in range(16):
            idle = synth(rig, "idle", f, palette)
            if not soles_hold(idle, rig["rest"], rig["soles"]["foot_l"]):
                raise SystemExit(f"{label} left foot left the ground on idle {f}")
            if not soles_hold(idle, rig["rest"], rig["soles"]["foot_r"]):
                raise SystemExit(f"{label} right foot left the ground on idle {f}")
            walked = synth(rig, "walk", f, palette)
            wp = pose("walk", f)
            if wp["lift_l"] < 0.5 and wp["fwd_l"] < 0.5:
                if not soles_hold(walked, rig["rest"], rig["soles"]["foot_l"]):
                    raise SystemExit(f"{label} left foot slipped on walk {f}")
            if wp["lift_r"] < 0.5 and wp["fwd_r"] < 0.5:
                if not soles_hold(walked, rig["rest"], rig["soles"]["foot_r"]):
                    raise SystemExit(f"{label} right foot slipped on walk {f}")
    if max(pose("hurt", f)["recoil"] for f in range(12)) > 38.001:
        raise SystemExit("recoil exceeds 38")

    if args.samples:
        for label in ("S", "W"):
            sheets = {act: one_row(rigs[label], palette, act) for act in ACTIONS}
            contact_sheet(sheets, 0).save(work / f"samples_{label}.png")
            check_row(sheets, label)
        nys = np.nonzero(rigs["S"]["rest"][..., 3] == 255)[0]
        sprite_h = int(nys.max() - nys.min() + 1)
        print(f"feet 2 cell {CELL_W}x{CELL_H} fy {FOOT_Y} h {sprite_h}", flush=True)
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
        for direction in DIRS:
            key = f"{act}/{direction}"
            if key in state["done"]:
                continue
            od = frames_dir / act / direction
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[direction], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    write_outputs(sheets)
    preview = contact_sheet(sheets, 0)
    preview.save(OUT_DIR / "preview.png")
    preview.save(OUT_DIR / "preview_motion.png")
    (SRC_DIR / "actions").mkdir(parents=True, exist_ok=True)
    preview.save(SRC_DIR / "actions" / "preview.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] == 255)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    counts_a, ratios = south_attack_ratios(sheets)
    print_ratios(counts_a, ratios, sheets)
    print(f"feet 2 cell {CELL_W}x{CELL_H} h {sprite_h}", flush=True)

    master = np.concatenate([sheets[a] for a in ACTIONS], axis=0)
    report = qa_report(master, sheets, palette, ratios, sprite_h)
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            try:
                subprocess.run(
                    [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                     "--fps", "12", "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                    capture_output=True, text=True, timeout=90, check=False,
                )
            except subprocess.TimeoutExpired:
                report[f"gait_{act}"] = "timeout"
                continue
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)

    bad_spin = [s for s in report["spin_body"] if s["main"] < 0.90]
    failed = (
        report["off_palette"] or report["clip_frames"] or report["alphas"] != [0, 255]
        or report["bad_ratio"] or bad_spin or report["hurt_squash"] or not report["h_matches"]
    )
    if failed:
        if args.publish:
            update_queue("needs_fix")
        raise SystemExit(f"qa failed {report} spin={bad_spin}")

    meta = {
        "generator": "forge_abomasnow_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 460,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "unflipped": sorted(UNFLIPPED),
        "still_faces": "front",
        "feet": 2,
        "mouth": "none",
        "cloak": "torso",
        "parts": counts,
        "palette_colors": int(len(palette)),
        "off_palette_colors": 0,
        "attack_s_ratios": [round(r, 4) for r in ratios],
        "mega_key_edited": False,
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.publish and not args.no_sync:
        copied = 0
        SYNC_OUT.mkdir(parents=True, exist_ok=True)
        for fp in OUT_DIR.iterdir():
            if fp.is_file():
                shutil.copy2(fp, SYNC_OUT / fp.name)
                copied += 1
        publish_dir8(sprite_h)
        update_queue("done")
        arena_bytes = (SYNC_OUT / "sheet.png").stat().st_size
        print(f"synced {copied} arena_sheet_bytes {arena_bytes}", flush=True)
    print(f"h {sprite_h} cw {CELL_W} ch {CELL_H} fy {FOOT_Y} feet 2", flush=True)


if __name__ == "__main__":
    main()
