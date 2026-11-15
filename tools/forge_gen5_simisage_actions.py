#!/usr/bin/env python3
"""Simisage normal (0512) 8-direction actions from this form's own stills.

front.png 96x96, bbox x18-78 y10-79, foot y79, alpha 0/255.
  Eye whites sit at x33-48 y34-43, centroid x41.6.
  The muzzle (cream, not the hands) starts at x34, left of the eyes, so the
  still faces screen-left. There is no open mouth: y43-50 x36-53 is solid,
  so the roar does not cut a wedge or draw rings.
  Parts that are actually drawn: leaf crest, two side leaves, two arms with
  cream hands, a thin tail, two legs with cream feet.
back.png bbox x14-69 y14-82, foot y82. The crest tip is x54-55, so the back
  still faces screen-right. The tail curls to the screen-left.

Directions: S SE E look screen-right (front mirrored). W SW look screen-left
(front as drawn). N NE show the back as drawn (away-right). NW is that back
mirrored (away-left). Attack 5-8 is that back at equal scale, not a squash.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen5_simisage_actions.py [--samples] [--force] [--publish]
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen5_attack_strike import finish, save_peaks

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen5/0512_simisage/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0512_simisage/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0512_simisage"
VERSION = "simisage_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 192
OFF_X = (CELL_W - 96) // 2
OFF_Y = 40
FRONT_FOOT = 79
BACK_FOOT = 82
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn faces screen-left. Back as drawn faces away-right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Wind-up peaks. The slash itself is drawn at 34, above these, so finish()
# does not push the strike past the pose this script already draws.
PEAKS = {
    "crest": 16.0, "ear_l": 12.0, "ear_r": 12.0,
    "arm_l": 18.0, "arm_r": 18.0, "tail": 16.0,
}
ROTATE = ("crest", "ear_l", "ear_r", "arm_l", "arm_r", "tail")
LIMBS = ROTATE + ("foot_l", "foot_r")

INK = (0, 0, 0)
GREEN = (24, 164, 82)
DGREEN = (8, 115, 49)
XDARK = (24, 74, 41)
MINT = (131, 205, 156)
CREAM = (255, 205, 123)
TAN = (205, 164, 90)
BROWN = (115, 82, 32)
BROWN2 = (164, 123, 57)
WHITE = (255, 255, 255)
GREY = (82, 82, 82)
LGRAY = (156, 156, 156)
MGRAY = (197, 197, 197)
GREENS = (GREEN, DGREEN, XDARK, MINT)
SKIN_COLS = (CREAM, TAN, BROWN, BROWN2)
LIGHTS = (WHITE, GREY, LGRAY, MGRAY)

SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)


def is_color(img, color):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(color, np.uint8), axis=-1)


def any_color(img, cols):
    m = np.zeros(img.shape[:2], bool)
    for c in cols:
        m |= is_color(img, c)
    return m


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1, constant_values=False)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def flood(allowed, seeds):
    out = seeds & allowed
    while True:
        nxt = dilate(out, 1) & allowed
        if np.array_equal(nxt, out):
            return out
        out = nxt


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
    """4x nearest rotation. Positive is clockwise. No width squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    sc = 4
    up = np.repeat(np.repeat(layer, sc, axis=0), sc, axis=1)
    h, w = up.shape[:2]
    px, py = pivot[0] * sc + 1.5, pivot[1] * sc + 1.5
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = xx - px, yy - py
    r = math.radians(ang)
    c, s = math.cos(r), math.sin(r)
    sx = np.rint(px + c * rx + s * ry).astype(np.int32)
    sy = np.rint(py - s * rx + c * ry).astype(np.int32)
    ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    out = np.zeros_like(up)
    out[ok] = up[sy[ok], sx[ok]]
    return out[sc // 2::sc, sc // 2::sc].copy()


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1, constant_values=False)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img):
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        p = np.pad(a, 1, constant_values=False)
        n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        hole = (~a) & (n == 4)
        if not hole.any():
            break
        up = np.zeros_like(out)
        up[1:] = out[:-1]
        out[hole] = up[hole]
    return out


def snap(img, palette):
    out = img.copy()
    op = out[..., 3] >= 128
    out[~op] = 0
    if not op.any():
        return out
    px = out[op][:, :3].astype(np.int32)
    pal = palette.astype(np.int32)
    d = ((px[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    out[op, :3] = palette[d.argmin(axis=1)]
    out[op, 3] = 255
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def put(img, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 1 <= x < w - 1 and 1 <= y < h - 1 and img[y, x, 3] < 128:
        img[y, x, :3] = col
        img[y, x, 3] = 255


def paint_cushion(layer, allowed, steps):
    """Grow this layer's colours into allowed pixels. Hidden while the limb rests."""
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        for name in ("up", "down", "left", "right"):
            grown = np.zeros_like(src)
            if name == "up":
                grown[:-1] = src[1:]
            elif name == "down":
                grown[1:] = src[:-1]
            elif name == "left":
                grown[:, :-1] = src[:, 1:]
            else:
                grown[:, 1:] = src[:, :-1]
            take = (acc[..., 3] < 128) & allowed & (grown[..., 3] >= 128)
            acc[take] = grown[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def component_list(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    groups = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
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
        groups.append(pts)
    return groups


def largest_component(mask):
    groups = component_list(mask)
    if not groups:
        return 0
    return max(len(g) for g in groups)


def check_partition(op, masks):
    acc = np.zeros_like(op)
    for name, m in masks.items():
        if (m & acc).any():
            raise SystemExit(f"overlap in {name}")
        if (m & ~op).any():
            raise SystemExit(f"{name} left the still")
        acc |= m
    if int(acc.sum()) != int(op.sum()):
        raise SystemExit(f"parts do not partition the still: {int(acc.sum())} vs {int(op.sum())}")


def require_piece(name, mask, minimum):
    if int(mask.sum()) < minimum:
        raise SystemExit(f"{name} is too small ({int(mask.sum())})")
    if largest_component(mask) != int(mask.sum()):
        raise SystemExit(f"{name} fell apart ({largest_component(mask)} vs {int(mask.sum())})")


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    greens = any_color(img, GREENS)
    skin = any_color(img, SKIN_COLS)
    lights = any_color(img, LIGHTS)
    ink = is_color(img, INK)
    leaf = (greens | ink) & ~lights
    crest = op & (yy <= 32) & (xx < 56)
    ear_l = flood(leaf & (yy >= 32) & (yy <= 46) & (xx <= 40) & ~crest, greens & (xx <= 36) & (yy >= 33) & (yy <= 44))
    ear_r = flood(leaf & (yy >= 31) & (yy <= 46) & (xx >= 52) & ~crest, greens & (xx >= 56) & (yy >= 32) & (yy <= 44))
    # Hands stay outside the chest. Light ruff pixels stay on the torso.
    arm_l = flood(
        op & (xx <= 34) & (yy >= 46) & (yy <= 67) & ~ear_l & ~crest & ~lights,
        skin & (xx <= 32) & (yy >= 47) & (yy <= 64),
    )
    arm_r = flood(
        op & (xx >= 60) & (yy >= 47) & (yy <= 65) & ~ear_r & ~crest & ~lights,
        skin & (xx >= 66) & (yy >= 50) & (yy <= 62),
    )
    tail = flood(
        leaf & (xx >= 54) & (xx <= 66) & (yy >= 62) & (yy <= 76) & ~arm_r,
        op & (xx >= 57) & (xx <= 64) & (yy >= 64) & (yy <= 70) & ~arm_r,
    )
    foot_l = flood(
        op & (yy >= 68) & (xx <= 40) & ~arm_l & ~ear_l,
        skin & (yy >= 73) & (xx <= 40),
    )
    foot_r = flood(
        op & (yy >= 67) & (xx >= 44) & (xx <= 66) & ~arm_r & ~tail & ~ear_r,
        skin & (yy >= 73) & (xx >= 48),
    )
    used = crest | ear_l | ear_r | arm_l | arm_r | tail | foot_l | foot_r
    torso = op & ~used
    masks = {
        "crest": crest, "ear_l": ear_l, "ear_r": ear_r,
        "arm_l": arm_l, "arm_r": arm_r, "tail": tail,
        "foot_l": foot_l, "foot_r": foot_r, "torso": torso,
    }
    check_partition(op, masks)
    eyes = lights & (yy >= 33) & (yy <= 45)
    if int(eyes.sum()) < 20 or int((eyes & torso).sum()) != int(eyes.sum()):
        raise SystemExit(f"eyes left the head ({int((eyes & torso).sum())}/{int(eyes.sum())})")
    muzzle = skin & (yy >= 42) & (yy <= 58) & (xx >= 34) & (xx <= 52)
    if int(muzzle.sum()) < 40 or int((muzzle & torso).sum()) != int(muzzle.sum()):
        raise SystemExit("muzzle was cut off the torso")
    eye_x = float(np.nonzero(is_color(img, WHITE) & (yy < 45))[1].mean())
    my, mx = np.nonzero(muzzle)
    snout_x = float(mx.min())
    if not snout_x < eye_x:
        raise SystemExit(f"front no longer faces left (snout {snout_x} eye {eye_x})")
    for name, minimum in (
        ("crest", 200), ("ear_l", 80), ("ear_r", 80), ("arm_l", 60), ("arm_r", 60),
        ("tail", 30), ("foot_l", 50), ("foot_r", 50), ("torso", 400),
    ):
        require_piece(name, masks[name], minimum)
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    greens = any_color(img, GREENS)
    skin = any_color(img, SKIN_COLS)
    lights = any_color(img, LIGHTS)
    ink = is_color(img, INK)
    leaf = (greens | ink) & ~lights
    crest = op & (yy <= 33) & (xx >= 44) & (xx <= 62)
    top_ys, top_xs = np.nonzero(op)
    top = int(top_ys.min())
    if int(top_xs[top_ys == top].mean()) < 48:
        raise SystemExit("back crest is no longer on the right")
    ear_l = flood(
        leaf & (xx <= 42) & (yy >= 33) & (yy <= 50) & ~crest,
        leaf & (xx <= 38) & (yy >= 34) & (yy <= 44),
    )
    ear_r = flood(
        leaf & (xx >= 58) & (yy >= 33) & (yy <= 48) & ~crest,
        leaf & (xx >= 60) & (yy >= 36) & (yy <= 46),
    )
    arm_l = flood(
        op & (xx <= 38) & (yy >= 48) & (yy <= 66) & ~ear_l & ~crest & ~lights,
        skin & (xx <= 36) & (yy >= 49) & (yy <= 64),
    )
    arm_r = flood(
        op & (xx >= 56) & (yy >= 46) & (yy <= 68) & ~crest & ~ear_r & ~lights,
        (skin & (xx >= 50) & (yy >= 48) & (yy <= 66)) | (op & (xx >= 64) & (yy >= 50) & (yy <= 64) & ~crest),
    )
    foot_l = flood(
        op & (yy >= 72) & (xx >= 30) & (xx <= 46) & ~arm_l,
        skin & (yy >= 76) & (xx >= 30) & (xx <= 44),
    )
    foot_r = flood(
        op & (yy >= 68) & (xx >= 48) & ~arm_r & ~crest,
        skin & (yy >= 76) & (xx >= 50),
    )
    tail = flood(
        leaf & (xx <= 32) & (yy >= 52) & (yy <= 78) & ~foot_l & ~arm_l & ~ear_l & ~crest,
        op & (xx <= 24) & (yy >= 58) & (yy <= 74),
    )
    used = crest | ear_l | ear_r | arm_l | arm_r | tail | foot_l | foot_r
    torso = op & ~used
    masks = {
        "crest": crest, "ear_l": ear_l, "ear_r": ear_r,
        "arm_l": arm_l, "arm_r": arm_r, "tail": tail,
        "foot_l": foot_l, "foot_r": foot_r, "torso": torso,
    }
    check_partition(op, masks)
    for name, minimum in (
        ("crest", 120), ("ear_l", 40), ("ear_r", 30), ("arm_l", 20), ("arm_r", 20),
        ("tail", 60), ("foot_l", 20), ("foot_r", 40), ("torso", 400),
    ):
        require_piece(name, masks[name], minimum)
    return masks


def place(arr, y_shift):
    if arr.ndim == 2:
        out = np.zeros((CELL_H, CELL_W), bool)
    else:
        out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    y0 = OFF_Y + y_shift
    x0 = OFF_X
    h, w = arr.shape[:2]
    if y0 < 0 or x0 < 0 or y0 + h > CELL_H or x0 + w > CELL_W:
        raise SystemExit(f"place overflow y {y0} x {x0}")
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def joint_band(part, torso, depth):
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 4)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def pivot_of(layer, torso_layer):
    part = layer[..., 3] >= 128
    torso = torso_layer[..., 3] >= 128
    touch = part & dilate(torso, 2)
    if not touch.any():
        touch = part & dilate(torso, 6)
    if not touch.any():
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(torso)
    cx, cy = float(tx.mean()), float(ty.mean())
    d = (xs - cx) ** 2 + (ys - cy) ** 2
    near = d <= d.min() + 6.0
    return float(xs[near].mean()), float(ys[near].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs - pivot[0]) ** 2 + (ys - pivot[1]) ** 2
    i = int(d.argmax())
    return float(xs[i]), float(ys[i])


def signs_for(layer, pivot, facing):
    if int((layer[..., 3] >= 128).sum()) < 4:
        return {"up": 1.0, "fwd": 1.0}

    def tip_after(ang):
        return tip_xy(rotate_nearest(layer, ang, pivot), pivot)

    x_pos, y_pos = tip_after(12.0)
    x_neg, y_neg = tip_after(-12.0)
    up = -1.0 if y_neg < y_pos else 1.0
    if facing > 0:
        fwd = 1.0 if x_pos >= x_neg else -1.0
    else:
        fwd = 1.0 if x_pos <= x_neg else -1.0
    return {"up": up, "fwd": fwd}


def build_view(img, masks, kind, mirror):
    y_shift = 0 if kind == "front" else (FRONT_FOOT - BACK_FOOT)
    canvas = place(img, y_shift)
    placed = {k: place(v, y_shift) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    depth = {"crest": 5, "ear_l": 4, "ear_r": 4, "arm_l": 6, "arm_r": 6, "tail": 5, "foot_l": 6, "foot_r": 6}
    for name in LIMBS:
        if not placed[name].any():
            continue
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth[name]), depth[name] + 2
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in ROTATE:
        if int((layers[name][..., 3] >= 128).sum()) < 4:
            continue
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    for name in ("foot_l", "foot_r"):
        pivots[name] = pivot_of(layers[name], layers["torso"])
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    foot_cx = {}
    for n in ("foot_l", "foot_r"):
        if placed[n].any():
            foot_cx[n] = float(np.nonzero(placed[n])[1].mean())
    if facing > 0:
        near = max(foot_cx, key=lambda n: foot_cx[n])
    else:
        near = min(foot_cx, key=lambda n: foot_cx[n])
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas, "near_foot": near,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        crest=0.0, ear_l=0.0, ear_r=0.0, arm_l=0.0, arm_r=0.0, tail=0.0,
        crest_mode="up", ear_mode="up", arm_mode="up", tail_mode="up",
        thrust=0, step_l=(0, 0), step_r=(0, 0), kick=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0, turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        p["crest"] = wave(f, n, 4.0, phase=0.4)
        p["ear_l"] = wave(f, n, 3.0, phase=0.9)
        p["ear_r"] = wave(f, n, 3.0, phase=1.7)
        p["arm_l"] = wave(f, n, 4.0, phase=0.2)
        p["arm_r"] = wave(f, n, 4.0, phase=1.1)
        p["tail"] = wave(f, n, 6.0, phase=0.6)
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        p["step_l"] = (int(round(2 * max(0.0, sa))), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * max(0.0, -sa))), -int(round(3 * max(0.0, -sa))))
        p["arm_l"] = wave(f, n, 8.0, phase=0.0)
        p["arm_r"] = wave(f, n, 8.0, phase=math.pi)
        p["arm_mode"] = "fwd"
        p["crest"] = wave(f, n, 3.0, phase=0.3)
        p["ear_l"] = wave(f, n, 3.0, phase=0.8)
        p["ear_r"] = wave(f, n, 3.0, phase=0.8 + math.pi)
        p["tail"] = wave(f, n, 10.0, phase=math.pi / 2)
        p["tail_mode"] = "fwd"
    elif action == "attack":
        # 0 rest. 1-4 raise the crest, leaves and arms. No mouth cavity.
        # 5-8 the back still, equal scale. 9-12 hand slash and tail whip. 13-15 recover.
        crest = [0, 6, 12, 16, 16, 0, 0, 0, 0, 16, 14, 12, 8, 4, 0, 0]
        ears = [0, 4, 8, 12, 12, 0, 0, 0, 0, 10, 8, 6, 4, 2, 0, 0]
        arms = [0, 6, 12, 16, 18, 0, 0, 0, 0, 22, 28, 34, 34, 12, 4, 0]
        tail = [0, 4, 8, 12, 14, 0, 0, 0, 0, 18, 28, 34, 26, 10, 4, 0]
        p["crest"] = float(crest[f])
        p["ear_l"] = float(ears[f])
        p["ear_r"] = float(ears[f])
        p["arm_l"] = float(arms[f])
        p["arm_r"] = float(arms[f])
        p["tail"] = float(tail[f])
        p["arm_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["tail_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["thrust"] = 2 if 10 <= f <= 12 else (1 if f in (9, 13) else 0)
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["kick"] = (1, -1)
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["crest"] = 10.0 * env
        p["ear_l"] = 6.0 * env
        p["ear_r"] = 6.0 * env
        p["arm_l"] = 8.0 * env
        p["arm_r"] = 8.0 * env
        p["tail"] = 8.0 * env
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        for key in ("crest", "ear_l", "ear_r", "arm_l", "arm_r", "tail"):
            p[key] = 0.0
        p["thrust"] = 0
        p["bdx"] = 0
        p["bdy"] = 0
        p["kick"] = (0, 0)
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["spin"] = 0
        p["spark"] = 0
        p["lean"] = 0.0
    if abs(p["lean"]) > MAX_LEAN + 0.01:
        raise SystemExit(f"lean {p['lean']} exceeds {MAX_LEAN}")
    return p


def part_angle(rig, name, mode, amp):
    if abs(amp) < 0.45 or name not in rig["signs"]:
        return 0.0
    sign = rig["signs"][name]
    if mode == "up":
        return sign["up"] * amp
    return sign["fwd"] * amp


def moved(layer, ang, pivot, dx, dy):
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, pivot)
    return shift(layer, dx, dy)


def limb_layers(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    thrust = int(p["thrust"])
    parts = {}
    for name in LIMBS:
        if name == "foot_l":
            sx, sy = p["step_l"]
            ang = 0.0
            extra = 0
        elif name == "foot_r":
            sx, sy = p["step_r"]
            ang = 0.0
            extra = 0
        elif name in ("arm_l", "arm_r"):
            sx, sy = 0, 0
            ang = part_angle(rig, name, p["arm_mode"], p[name])
            extra = thrust
        elif name == "tail":
            sx, sy = 0, 0
            ang = part_angle(rig, name, p["tail_mode"], p["tail"])
            extra = thrust
        elif name == "crest":
            sx, sy = 0, 0
            ang = part_angle(rig, name, p["crest_mode"], p["crest"])
            extra = 0
        else:
            sx, sy = 0, 0
            ang = part_angle(rig, name, p["ear_mode"], p[name])
            extra = 0
        if name == rig["near_foot"]:
            kx, ky = p["kick"]
            sx += kx
            sy += ky
        if name not in piv:
            parts[name] = L[name]
            continue
        dx = bdx + int(round(fc * (sx + extra)))
        dy = bdy + int(sy)
        parts[name] = moved(L[name], ang, piv[name], dx, dy)
    return parts


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    parts = limb_layers(rig, p)
    for name in ("tail", "foot_l", "foot_r", "ear_l", "ear_r", "arm_l", "arm_r", "crest"):
        paste(out, parts[name])
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK)
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def weapon_tip(rig, p):
    parts = limb_layers(rig, p)
    fc = rig["facing"]
    best = None
    best_x = None
    for name in ("arm_l", "arm_r", "tail"):
        layer = parts[name]
        if not np.any(layer[..., 3] >= 128):
            continue
        ys, xs = np.nonzero(layer[..., 3] >= 128)
        i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
        x = float(xs[i])
        if best_x is None or (fc > 0 and x > best_x) or (fc < 0 and x < best_x):
            best_x = x
            best = (x, float(ys[i]))
    if best is None:
        return rig["pivots"]["center"]
    return best


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, CREAM, GREEN)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2), (2, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name, layer in limb_layers(rig, p).items():
        limb = layer[..., 3] >= 128
        if not limb.any():
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    """Attack 5-8 faces away. Front rows become that side's back, unscaled."""
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", not mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    rotated = any(abs(p[k]) >= 0.45 for k in ROTATE) or abs(p["lean"]) > 0.4
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], piv)
    if rotated:
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = weapon_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
    return snap(img, palette)


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def assemble(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                t = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png").convert("RGBA"))
                sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = t
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


def tile_grid(tiles, cols, bg=(18, 22, 34, 255)):
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    out = np.full((rows * th, cols * tw, 4), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        m = t[..., 3] >= 128
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
    return out


def crop_box(canvas, pad_x=28, pad_y=22):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad_y)
    y1 = min(CELL_H, int(ys.max()) + 1 + 16)
    x0 = max(0, int(xs.min()) - pad_x)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad_x)
    return y0, y1, x0, x1


def preview_image(sheets):
    pick = {"idle": 0, "walk": 4, "attack": 11, "hurt": 3}
    tiles = []
    for act in ACTIONS:
        for di in range(8):
            f = pick[act]
            tiles.append(sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W])
    return Image.fromarray(tile_grid(tiles, 8), "RGBA")


def motion_strip(sheets, di, box):
    y0, y1, x0, x1 = box
    tiles = []
    for act in ACTIONS:
        for f in range(16):
            if f < FRAMES[act]:
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                tiles.append(t[y0:y1, x0:x1])
            else:
                tiles.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
    return tile_grid(tiles, 16)


def still_colors(*imgs):
    cols = []
    seen = set()
    for img in imgs:
        for c in np.unique(img[img[..., 3] >= 128][:, :3], axis=0):
            t = tuple(int(v) for v in c)
            if t not in seen:
                seen.add(t)
                cols.append(t)
    return cols


def locked_palette(front_path, back_path, work):
    front = np.array(Image.open(front_path).convert("RGBA"))
    back = np.array(Image.open(back_path).convert("RGBA"))
    own = still_colors(front, back)
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
    if not (lock_dir / "palette.json").exists():
        py = str(FORGE / ".venv/bin/python")
        hexes = ["#%02x%02x%02x" % c for c in own]
        for d in (pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        r = subprocess.run(
            [py, "-I", str(PALETTE_TOOL), "build", "--input", str(front_path),
             "--colors", str(len(hexes)), "--reserve", *hexes, "--output-dir", str(pal_dir)],
            capture_output=True, text=True,
        )
        if r.returncode:
            raise SystemExit("palette build failed: " + r.stdout + r.stderr)
        r = subprocess.run(
            [py, "-I", str(PALETTE_TOOL), "lock", "--palette", str(pal_dir / "palette.json"),
             "--source", str(front_path), "--strict", "--output-dir", str(lock_dir)],
            capture_output=True, text=True,
        )
        if r.returncode:
            raise SystemExit("palette lock failed: " + r.stdout + r.stderr)
    data = json.loads((lock_dir / "palette.json").read_text())
    cols = np.array([[int(c["hex"][i:i + 2], 16) for i in (1, 3, 5)] for c in data["colors"]], np.uint8)
    got = {tuple(int(v) for v in c) for c in cols}
    want = set(own)
    if got != want:
        raise SystemExit(f"locked palette differs from the stills: {sorted(got ^ want)}")
    return cols


def params_hash():
    src = Path(__file__).read_bytes() + (SRC_DIR / "front.png").read_bytes() + (SRC_DIR / "back.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def build_rigs():
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    back = np.array(Image.open(SRC_DIR / "back.png").convert("RGBA"))
    if front.shape[:2] != (96, 96) or back.shape[:2] != (96, 96):
        raise SystemExit("expected 96x96 stills")
    for im in (front, back):
        im[im[..., 3] < 128] = 0
        im[im[..., 3] >= 128, 3] = 255
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "crest": (255, 70, 70, 255),
        "ear_l": (255, 170, 40, 255),
        "ear_r": (255, 220, 60, 255),
        "arm_l": (70, 170, 255, 255),
        "arm_r": (40, 70, 230, 255),
        "tail": (190, 70, 255, 255),
        "foot_l": (70, 230, 90, 255),
        "foot_r": (30, 160, 70, 255),
        "torso": (180, 180, 180, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        pair = np.concatenate([img, part], axis=1)
        tiles.append(np.repeat(np.repeat(pair, 4, 0), 4, 1))
    Image.fromarray(np.concatenate(tiles, axis=0), "RGBA").save(work / "parts.png")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    return n


def publish(bounds, sprite_h):
    import gen5_publish

    entry = {
        "dex": 512, "form": "normal", "variant": "simisage", "gen": "gen5",
        "dir": "monsterSprite/gen5/0512_simisage/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def assert_turns(rigs, palette):
    for d, back_d in (("S", "N"), ("E", "N"), ("SE", "N"), ("W", "NW"), ("SW", "NW")):
        got = synth(rigs, d, "attack", 6, palette, effects=False)
        if not np.array_equal(got, rigs[back_d]["canvas"]):
            raise SystemExit(f"attack turn {d} is not the {back_d} still")
    for d in ("N", "NE", "NW"):
        got = synth(rigs, d, "attack", 7, palette, effects=False)
        if not np.array_equal(got, rigs[d]["canvas"]):
            raise SystemExit(f"attack turn {d} changed the back still")
    for d in DIRS:
        a0 = synth(rigs, d, "attack", 0, palette, effects=False)
        a15 = synth(rigs, d, "attack", 15, palette, effects=False)
        if not np.array_equal(a0, rigs[d]["canvas"]) or not np.array_equal(a15, rigs[d]["canvas"]):
            raise SystemExit(f"attack ends do not match the still for {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    save_peaks(Path(__file__).stem, PEAKS)
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    palette = locked_palette(SRC_DIR / "front.png", SRC_DIR / "back.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS},
          "near", {d: rigs[d]["near_foot"] for d in ("S", "W", "N", "E")}, flush=True)

    rest_diff = {}
    for d in DIRS:
        r = synth(rigs, d, "idle", 0, palette, effects=False)
        rest_diff[d] = int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")
    assert_turns(rigs, palette)

    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        y0, y1, x0, x1 = box
        for d in ("S", "W", "N", "E"):
            frames = []
            for act, fs in (
                ("idle", (0, 4, 8, 12)),
                ("walk", (0, 2, 4, 6, 8, 10, 12, 14)),
                ("attack", (0, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15)),
                ("hurt", (0, 1, 3, 6, 11)),
            ):
                for f in fs:
                    frames.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
            sheet = tile_grid(frames, len(frames))
            z = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
            Image.fromarray(z, "RGBA").save(work / f"samples_{d}.png")
        zoom_frames = []
        for act, f in (
            ("idle", 0), ("walk", 4), ("walk", 12),
            ("attack", 0), ("attack", 4), ("attack", 5), ("attack", 8),
            ("attack", 9), ("attack", 11), ("attack", 12), ("attack", 15),
            ("hurt", 3),
        ):
            zoom_frames.append(synth(rigs, "S", act, f, palette)[y0:y1, x0:x1])
        for act, f in (("attack", 11), ("hurt", 3), ("walk", 4), ("idle", 0)):
            zoom_frames.append(synth(rigs, "N", act, f, palette)[y0:y1, x0:x1])
        z = tile_grid(zoom_frames, 8)
        Image.fromarray(np.repeat(np.repeat(z, 3, 0), 3, 1), "RGBA").save(work / "samples_zoom.png")
        lost = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["arm_l"], 1), p["arm_mode"], p["turned"], p["spin"], round(p["tail"], 1)))
        print("attack poses", atk, flush=True)
        hurt = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", hurt, "max", max(hurt), flush=True)
        print("samples", work, flush=True)
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
            if key in state["done"] and not args.force:
                continue
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.exists() and fp.stat().st_size > 0 and not args.force:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    y0, y1, x0, x1 = box
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.concatenate(strips, axis=0)
    Image.fromarray(np.repeat(np.repeat(motion, 2, 0), 2, 1), "RGBA").save(OUT_DIR / "preview_motion.png")
    preview_image(sheets).save(OUT_DIR / "preview.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", OUT_DIR / "actions" / "preview.png")

    pal = {tuple(int(v) for v in c) for c in palette}
    op = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[op][:, :3], axis=0)}
    off = sorted(used - pal)
    partial = int(((master[..., 3] != 0) & (master[..., 3] != 255)).sum())
    clip = []
    bodies = []
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = t[..., 3] == 255
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
                if act in ("attack", "hurt", "walk"):
                    bodies.append(largest_fraction(t))
    import gen5_publish
    bounds = gen5_publish.measure_idle_bounds(OUT_DIR / "sheet.png", CELL_W, CELL_H, rows=8)
    sprite_h = max(bounds[i + 3] for i in range(0, 32, 4))
    idle0 = {}
    for di, d in enumerate(DIRS):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        idle0[d] = int(np.any(t != rigs[d]["canvas"], axis=2).sum())
    report = {
        "version": VERSION, "hash": digest, "rest_diff": rest_diff, "idle0": idle0,
        "off_palette_colors": len(off), "partial_alpha": partial, "edge_touching": clip[:12],
        "edge_n": len(clip), "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "bounds": bounds, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (not any(rest_diff.values()) and not any(idle0.values()) and not off
          and partial == 0 and not clip and min(bodies) >= 0.9)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_simisage_actions", "version": VERSION, "key": KEY,
        "dex": 512, "variant": "simisage", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped monkey (leaf crest, two side leaves, two arms, thin tail, two feet, closed mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front eye centroid x41.6, muzzle starts x34 so the still faces screen-left; back crest tip is on the right",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    if args.publish:
        print("synced", sync_outputs(), flush=True)
        publish(bounds, sprite_h)
        print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
