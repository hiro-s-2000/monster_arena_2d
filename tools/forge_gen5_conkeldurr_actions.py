#!/usr/bin/env python3
"""Conkeldurr normal (0534) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x1-82 y15-79, 3332 px, 13 colours,
alpha 0/255):
  The red mouth is one blob at x34-41 y32-39 (mean x 37.5). The only white
  pixels are the eye at x45-46 y32-33 (mean x 45.7), to the right of the mouth,
  so the still faces screen-left. The mouth is drawn, so the roar uses a wedge
  and a ring in these colours. Nothing else is invented as a mouth.
  Crest: the tan horn at x43-50 y15-22.
  Snout-side pillar (grey, shorter): x1-18 y28-72. Rear pillar (grey, the
  ground contact): x58-78 y28-79. A small grey chest mark at x39-45 y41-45
  stays on the torso.
  Snout-side hand x2-30 y21-48. Rear hand x56-82 y22-52. Both are tan and
  already grip the pillars. Two feet under the split at y65: snout-side
  x35-45 ends y69, rear x51-58 ends y69. The rear pillar, not a foot, is the
  lowest pixel (y79).
normal/back.png (bbox x16-85 y17-73, 2599 px, 10 colours, all in the front
  palette, no red and no white). Crest x43-50 y17-24. Short pillar x18-33
  y45-68. Long pillar x68-85 y33-73, shifted down 6px so its sole shares the
  front foot line. Hands x16-40 y26-46 and x62-83 y25-48. Feet from y68:
  x27-42 and x43-53.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn. NW back mirrored.
  E is screen-right, W is screen-left, N is the back. Attack 5-8 uses that
  back at equal scale. No width squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_conkeldurr_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0534_conkeldurr/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0534_conkeldurr/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0534_conkeldurr"
VERSION = "conkeldurr_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 224, 208
OFF_X = (CELL_W - 96) // 2
# 40px under the source: the 38 degree lean swings the rear pillar 39px below the sole.
OFF_Y = CELL_H - 96 - 40
FRONT_FOOT = 79
BACK_FOOT = 73
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. Back as drawn is the away-right row.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Wind-up peaks finish() may raise frames 5-12 toward. The pillar strike is
# drawn at 34 and 18 degrees, above these, so finish does not pass it.
PEAKS = {"pillar_l": 16.0, "pillar_r": 12.0, "crest": 8.0}
LIMBS = ("crest", "arm_l", "arm_r", "leg_l", "leg_r", "pillar_l", "pillar_r")

INK = (16, 16, 16)
GREY = ((139, 139, 131), (106, 106, 98), (57, 57, 49))
GREY_L = (139, 139, 131)
RED = (172, 41, 65)
RED_D = (115, 41, 65)
WHITE = (255, 255, 255)
PINK = (197, 106, 180)
TAN = (197, 156, 115)

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


def components(mask):
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
    groups.sort(key=len, reverse=True)
    return groups


def mask_from(groups_pts):
    m = np.zeros((96, 96), bool)
    for y, x in groups_pts:
        m[y, x] = True
    return m


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
    """4x nearest rotation. Positive angle moves a point above the pivot to screen-left."""
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


def reconnect(img):
    """Bridge a 1px gap opened by a swing. Effects are drawn later."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(3):
        groups = components(out[..., 3] >= 128)
        if len(groups) <= 1:
            return out
        main = np.zeros((h, w), bool)
        for y, x in groups[0]:
            main[y, x] = True
        changed = False
        for g in groups[1:]:
            if len(g) < 4:
                continue
            best = None
            for y, x in g:
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and main[ny, nx]:
                            dist = max(abs(dy), abs(dx))
                            if best is None or dist < best[0]:
                                best = (dist, y, x, ny, nx)
            if best is None or best[0] > 2:
                continue
            _, y, x, ny, nx = best
            if abs(ny - y) == 1 and abs(nx - x) == 1:
                candidates = ((y, nx), (ny, x))
            else:
                sy = 0 if ny == y else (1 if ny > y else -1)
                sx = 0 if nx == x else (1 if nx > x else -1)
                candidates = ((y + sy, x + sx),)
            for fy, fx in candidates:
                if 0 <= fy < h and 0 <= fx < w and out[fy, fx, 3] < 128:
                    out[fy, fx, :3] = out[y, x, :3]
                    out[fy, fx, 3] = 255
                    changed = True
                    break
        if not changed:
            return out
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
    """Grow this layer's own colours into allowed pixels. Hidden at rest."""
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        grown = []
        up = np.zeros_like(src)
        dn = np.zeros_like(src)
        lt = np.zeros_like(src)
        rt = np.zeros_like(src)
        up[:-1] = src[1:]
        dn[1:] = src[:-1]
        lt[:, :-1] = src[:, 1:]
        rt[:, 1:] = src[:, :-1]
        grown.extend((up, dn, lt, rt))
        for g in grown:
            take = (acc[..., 3] < 128) & allowed & (g[..., 3] >= 128)
            acc[take] = g[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


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


def largest_component(mask):
    groups = components(mask)
    if not groups:
        return 0
    return len(groups[0])


def touches(part, body):
    return bool((part & dilate(body, 1)).any())


def solid(name, mask):
    if not mask.any():
        raise SystemExit(f"{name} is empty")
    if largest_component(mask) != int(mask.sum()):
        raise SystemExit(f"{name} fell apart")


def pillars_of(img):
    """Two grey columns plus the ink shell and the sole chips in each column."""
    op = img[..., 3] >= 128
    grey = any_color(img, GREY)
    ink = is_color(img, INK)
    yy, xx = np.mgrid[0:96, 0:96]
    big = []
    for g in components(grey):
        if len(g) < 80:
            continue
        big.append(mask_from(g))
    if len(big) != 2:
        raise SystemExit(f"expected two pillars, got {len(big)}")
    big.sort(key=lambda m: float(np.nonzero(m)[1].mean()))
    out = []
    for core0 in big:
        ys, xs = np.nonzero(core0)
        x0, x1 = int(xs.min()) - 2, int(xs.max()) + 2
        y0, y1 = int(ys.min()) - 1, int(ys.max()) + 3
        band = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        core = core0 | (grey & band)
        shell = dilate(core, 1) & ink & band
        shell2 = dilate(core | shell, 1) & ink & band
        out.append((core | shell | shell2) & op)
    if (out[0] & out[1]).any():
        raise SystemExit("pillars overlap")
    return out[0], out[1]


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    pillar_l, pillar_r = pillars_of(img)
    rest = op & ~pillar_l & ~pillar_r
    crest = rest & (yy <= 22) & (xx >= 36) & (xx <= 56)
    arm_l = rest & (xx <= 30) & (yy <= 50) & ~crest
    arm_r = rest & (xx >= 56) & (yy <= 52) & ~crest
    leg_l = rest & (yy >= 65) & (xx >= 28) & (xx <= 47)
    leg_r = rest & (yy >= 65) & (xx >= 48) & (xx <= 60)
    used = pillar_l | pillar_r | crest | arm_l | arm_r | leg_l | leg_r
    torso = op & ~used
    masks = {
        "pillar_l": pillar_l, "pillar_r": pillar_r, "crest": crest,
        "arm_l": arm_l, "arm_r": arm_r, "leg_l": leg_l, "leg_r": leg_r,
        "torso": torso,
    }
    check_partition(op, masks)
    for name in LIMBS:
        solid(name, masks[name])
    solid("torso", torso)
    if not touches(pillar_l, arm_l):
        raise SystemExit("snout pillar left the hand")
    if not touches(pillar_r, torso | arm_r):
        raise SystemExit("rear pillar left the body")
    for name in ("crest", "arm_l", "arm_r", "leg_l", "leg_r"):
        if not touches(masks[name], torso):
            raise SystemExit(f"{name} does not touch the body")
    mouth = any_color(img, (RED, RED_D))
    eye = is_color(img, WHITE)
    if int((mouth & torso).sum()) != int(mouth.sum()) or int(mouth.sum()) < 40:
        raise SystemExit("mouth left the head")
    if int((eye & torso).sum()) != int(eye.sum()) or int(eye.sum()) < 3:
        raise SystemExit("eye left the head")
    if float(np.nonzero(mouth)[1].mean()) >= float(np.nonzero(eye)[1].mean()) - 4:
        raise SystemExit("front does not face left")
    if int(np.nonzero(op)[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    if int(np.nonzero(pillar_r)[0].max()) != FRONT_FOOT:
        raise SystemExit("rear pillar is not the ground contact")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    if any_color(img, (RED, RED_D)).any() or is_color(img, WHITE).any():
        raise SystemExit("back grew a face")
    pillar_l, pillar_r = pillars_of(img)
    rest = op & ~pillar_l & ~pillar_r
    crest = rest & (yy <= 24) & (xx >= 38) & (xx <= 55)
    arm_l = rest & (xx <= 40) & (yy >= 25) & (yy <= 46) & ~crest
    arm_r = rest & (xx >= 62) & (yy <= 48) & ~crest
    leg_l = rest & (yy >= 68) & (xx >= 24) & (xx <= 42)
    leg_r = rest & (yy >= 68) & (xx >= 43) & (xx <= 68)
    used = pillar_l | pillar_r | crest | arm_l | arm_r | leg_l | leg_r
    torso = op & ~used
    masks = {
        "pillar_l": pillar_l, "pillar_r": pillar_r, "crest": crest,
        "arm_l": arm_l, "arm_r": arm_r, "leg_l": leg_l, "leg_r": leg_r,
        "torso": torso,
    }
    check_partition(op, masks)
    for name in LIMBS:
        solid(name, masks[name])
    solid("torso", torso)
    for name in ("crest", "arm_l", "arm_r", "leg_l", "leg_r", "pillar_l", "pillar_r"):
        if not touches(masks[name], torso | arm_l | arm_r):
            raise SystemExit(f"back {name} does not touch the body")
    if int(np.nonzero(op)[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line changed")
    if int(np.nonzero(pillar_r)[0].max()) != BACK_FOOT:
        raise SystemExit("back long pillar is not the ground contact")
    return masks


def place(arr, y_shift):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    y0 = OFF_Y + y_shift
    x0 = OFF_X
    h, w = arr.shape[:2]
    if y0 < 0 or x0 < 0 or y0 + h > CELL_H or x0 + w > CELL_W:
        raise SystemExit(f"place overflow y {y0} x {x0}")
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def joint_band(part, anchor, depth):
    touch = part & dilate(anchor, 1)
    if not touch.any():
        touch = part & dilate(anchor, 3)
    if not touch.any():
        return np.zeros_like(part)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def pivot_of(layer, anchor_layer):
    part = layer[..., 3] >= 128
    anchor = anchor_layer[..., 3] >= 128
    touch = part & dilate(anchor, 2)
    if not touch.any():
        touch = part & dilate(anchor, 6)
    if not touch.any():
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ay, ax = np.nonzero(anchor)
    cx, cy = float(ax.mean()), float(ay.mean())
    d = (xs - cx) ** 2 + (ys - cy) ** 2
    near = d <= d.min() + 8.0
    return float(xs[near].mean()), float(ys[near].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs - pivot[0]) ** 2 + (ys - pivot[1]) ** 2
    i = int(d.argmax())
    return float(xs[i]), float(ys[i])


def signs_for(layer, pivot, facing):
    """Which rotation sign lifts the tip, and which sends it toward facing."""
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
    for name, depth in (
        ("crest", 4), ("arm_l", 4), ("arm_r", 4),
        ("leg_l", 4), ("leg_r", 4), ("pillar_r", 5), ("pillar_l", 3),
    ):
        layers["torso"] = paint_cushion(layers["torso"], joint_band(placed[name], torso_m, depth), depth)
    # The snout pillar hangs from the hand. Keep hand colour under that grip.
    layers["arm_l"] = paint_cushion(layers["arm_l"], joint_band(placed["pillar_l"], placed["arm_l"], 5), 5)
    layers["arm_r"] = paint_cushion(layers["arm_r"], joint_band(placed["pillar_r"], placed["arm_r"], 5), 5)
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    anchors = {
        "crest": layers["torso"],
        "arm_l": layers["torso"],
        "arm_r": layers["torso"],
        "leg_l": layers["torso"],
        "leg_r": layers["torso"],
        "pillar_l": layers["arm_l"],
        "pillar_r": layers["arm_r"],
    }
    pivots = {}
    signs = {}
    for name in LIMBS:
        pivots[name] = pivot_of(layers[name], anchors[name])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    if int(ys.max()) != FOOT_Y:
        raise SystemExit(f"{kind} foot is not on the ground line")
    red = any_color(canvas, (RED, RED_D))
    if red.any():
        ry, rx = np.nonzero(red)
        pivots["mouth"] = (float(rx.mean()), float(ry.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        crest=0.0, pillar_l=0.0, pillar_r=0.0, arm_l=0.0, arm_r=0.0,
        pillar_mode="up", arm_mode="up",
        step_l=(0, 0), step_r=(0, 0),
        lean=0.0, knock=0, flash=False, wedge=0, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        p["crest"] = wave(f, n, 5.0, phase=0.4)
        p["pillar_l"] = wave(f, n, 3.0, phase=0.2)
        p["pillar_r"] = wave(f, n, 3.0, phase=0.2 + math.pi)
        p["arm_l"] = wave(f, n, 3.0, phase=0.8)
        p["arm_r"] = wave(f, n, 3.0, phase=0.8 + math.pi)
        p["pillar_mode"] = "up"
        p["arm_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        p["step_l"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(-2 * sa)), -int(round(3 * max(0.0, -sa))))
        p["bdy"] = -1 if abs(sa) > 0.7 else 0
        p["crest"] = wave(f, n, 4.0)
        p["pillar_l"] = wave(f, n, 5.0, phase=0.4)
        p["pillar_r"] = wave(f, n, 5.0, phase=0.4 + math.pi)
        p["arm_l"] = wave(f, n, 4.0, phase=0.6)
        p["arm_r"] = wave(f, n, 4.0, phase=0.6 + math.pi)
        p["pillar_mode"] = "fwd"
        p["arm_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 roar (mouth wedge only where a mouth is drawn).
        # 5-8 the back still, equal scale. 9-12 the two pillars strike.
        # 13-15 recover. 15 is the still.
        crest = [0, 6, 10, 14, 14, 0, 0, 0, 0, 8, 8, 6, 4, 2, 0, 0]
        pillar_l = [0, 4, 8, 12, 12, 0, 0, 0, 0, 18, 26, 34, 34, 16, 6, 0]
        pillar_r = [0, 3, 6, 8, 8, 0, 0, 0, 0, 8, 14, 18, 16, 8, 3, 0]
        arm_l = [0, 3, 6, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        arm_r = [0, 2, 4, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        bdx = [0, 0, -1, -1, -1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0]
        p["crest"] = float(crest[f])
        p["pillar_l"] = float(pillar_l[f])
        p["pillar_r"] = float(pillar_r[f])
        p["arm_l"] = float(arm_l[f])
        p["arm_r"] = float(arm_r[f])
        p["bdx"] = int(bdx[f])
        p["pillar_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["arm_mode"] = "up"
        p["wedge"] = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["turned"] = 5 <= f <= 8
        if f in (10, 11, 12):
            p["step_l"] = (1, -1)
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.9, 0.7, 0.5, 0.32, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["crest"] = 8.0 * env
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    # Frames 5-8 are the back still at equal scale. finish() would swing the
    # wind-up here; drop it so the turned silhouette stays the still.
    if action == "attack" and 5 <= f <= 8:
        p["crest"] = 0.0
        p["pillar_l"] = 0.0
        p["pillar_r"] = 0.0
        p["arm_l"] = 0.0
        p["arm_r"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["wedge"] = 0
        p["spark"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["spin"] = 0
        p["flash"] = False
    return p


def part_angle(rig, name, mode, amp):
    if abs(amp) < 0.45 or name not in rig["signs"]:
        return 0.0
    return rig["signs"][name][mode] * amp


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
    parts = {}
    for name in LIMBS:
        if name == "leg_l":
            sx, sy = p["step_l"]
            ang = 0.0
            dy = int(sy)
        elif name == "leg_r":
            sx, sy = p["step_r"]
            ang = 0.0
            dy = int(sy)
        elif name in ("pillar_l", "pillar_r"):
            sx, sy = 0, 0
            ang = part_angle(rig, name, p["pillar_mode"], p[name])
            dy = 0
        elif name in ("arm_l", "arm_r"):
            sx, sy = 0, 0
            ang = part_angle(rig, name, p["arm_mode"], p[name])
            dy = bdy
        else:
            sx, sy = 0, 0
            ang = part_angle(rig, name, "up", p["crest"])
            dy = bdy
        dx = bdx + int(round(fc * sx))
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
    # Pillars last so the hand-coloured grip stays hidden until they swing.
    for name in ("crest", "arm_l", "arm_r", "leg_l", "leg_r", "pillar_r", "pillar_l"):
        paste(out, parts[name])
    return out


def moving(p):
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    if p["step_l"] != (0, 0) or p["step_r"] != (0, 0):
        return True
    for key in ("crest", "pillar_l", "pillar_r", "arm_l", "arm_r"):
        if abs(p[key]) >= 0.45:
            return True
    return False


def rig_for(rigs, d, p):
    """Attack 5-8 faces away. Front rows become that side's back, unscaled."""
    if not p.get("turned"):
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", not mirror)]


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


def mouth_wedge(img, origin, fc, age):
    """Wedge and ring in front of the drawn mouth. Only this still's colours."""
    ox, oy = origin
    cols = (RED, RED_D, INK, WHITE, PINK)
    for i, (dx, dy) in enumerate(((1, 0), (2, -1), (2, 1), (3, 0), (2, 0), (1, -1), (1, 1))):
        if i > 2 + age:
            break
        put(img, ox + fc * dx, oy + dy, cols[i % 5])
    rad = 7 + age * 3
    for deg in range(-40, 41, 8):
        a = math.radians(deg)
        x = ox + fc * (3 + rad * math.cos(a))
        y = oy + rad * math.sin(a) * 0.55
        put(img, x, y, cols[(deg // 8) % 5])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, GREY_L, INK, TAN)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r, -2), (r, 2), (2, -r), (2, r), (r + 1, 0))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def facing_tip(layer, facing):
    m = layer[..., 3] >= 128
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if facing > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    parts = limb_layers(rig, p)
    for name, layer in parts.items():
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    if not moving(p) and abs(p["lean"]) <= 0.4:
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            piv = rear_pivot(body, fc)
            body = rotate_nearest(body, fc * p["lean"], piv)
        body = reconnect(fill_pinholes(despeckle(body)))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["wedge"] and "mouth" in rig["pivots"] and not p["turned"]:
        bdx = int(round(fc * p["bdx"]))
        bdy = int(p["bdy"])
        mx, my = rig["pivots"]["mouth"]
        mouth_wedge(img, (mx + bdx + fc * 2, my + bdy), fc, p["wedge"])
        img = snap(img, palette)
    if p["spark"] and not p["turned"]:
        layer = limb_layers(rig, p)["pillar_l"]
        tip = facing_tip(layer, fc)
        if tip is not None:
            impact(img, (tip[0] + fc * 2, tip[1]), fc, p["spark"])
            img = snap(img, palette)
    return img


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


def tile_grid(tiles, cols, bg=(15, 20, 31, 255)):
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    out = np.full((rows * th, cols * tw, 4), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        m = t[..., 3] >= 128
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
    return out


def preview_image(sheets):
    pick = {"idle": 4, "walk": 4, "attack": 11, "hurt": 3}
    tiles = []
    for act in ACTIONS:
        for di in range(8):
            f = pick[act]
            tiles.append(sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W])
    return Image.fromarray(tile_grid(tiles, 8), "RGBA")


def motion_strip(sheets, di):
    tiles = []
    for act in ACTIONS:
        for f in range(16):
            if f < FRAMES[act]:
                tiles.append(sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W])
            else:
                tiles.append(np.zeros((CELL_H, CELL_W, 4), np.uint8))
    return tile_grid(tiles, 16)


def frame_faults(img, pal):
    a = img[..., 3]
    partial = int(((a != 0) & (a != 255)).sum())
    op = a >= 128
    edge = bool(op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any())
    off = set()
    solid_px = a == 255
    if solid_px.any():
        for c in np.unique(img[solid_px][:, :3], axis=0):
            t = tuple(int(v) for v in c)
            if t not in pal:
                off.add(t)
    return off, partial, edge


def upper_mean_x(img):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    top = int(ys.min())
    band = (np.mgrid[0:img.shape[0], 0:img.shape[1]][0] <= top + 14) & a
    return float(np.nonzero(band)[1].mean())


def locked_palette(front_path, back_path, work):
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
    img = np.array(Image.open(front_path).convert("RGBA"))
    own = np.unique(img[img[..., 3] >= 128][:, :3], axis=0)
    if not (lock_dir / "palette.json").exists():
        py = str(FORGE / ".venv/bin/python")
        hexes = ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in own]
        for d in (pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        r = subprocess.run(
            [py, "-I", str(PALETTE_TOOL), "build", "--input", str(front_path),
             "--colors", str(len(hexes)), "--reserve", *hexes,
             "--output-dir", str(pal_dir)],
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
    want = {tuple(int(v) for v in c) for c in own}
    if got != want:
        raise SystemExit(f"locked palette differs from the still: {sorted(got ^ want)}")
    back = np.array(Image.open(back_path).convert("RGBA"))
    extra = []
    for c in np.unique(back[back[..., 3] >= 128][:, :3], axis=0):
        t = tuple(int(v) for v in c)
        if t not in got:
            extra.append(c)
    if extra:
        cols = np.vstack([cols, np.array(extra, np.uint8)])
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
    if rigs["W"]["facing"] != -1.0 or rigs["E"]["facing"] != 1.0:
        raise SystemExit("front facing is not left-as-drawn")
    if rigs["N"]["facing"] != 1.0 or rigs["NW"]["facing"] != -1.0:
        raise SystemExit("back facing is not away-right as drawn")
    if "mouth" not in rigs["S"]["pivots"] or "mouth" in rigs["N"]["pivots"]:
        raise SystemExit("mouth is on the wrong view")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "pillar_l": (180, 180, 170, 255),
        "pillar_r": (90, 90, 80, 255),
        "arm_l": (220, 160, 90, 255),
        "arm_r": (160, 100, 50, 255),
        "leg_l": (80, 180, 80, 255),
        "leg_r": (40, 120, 40, 255),
        "crest": (220, 60, 60, 255),
        "torso": (200, 100, 170, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 3, 0), 3, 1))
    gap = np.zeros((tiles[0].shape[0], 8, 4), np.uint8)
    Image.fromarray(np.concatenate([tiles[0], gap, tiles[1]], axis=1), "RGBA").save(work / "parts.png")


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
        "dex": 534, "form": "normal", "variant": "conkeldurr", "gen": "gen5",
        "dir": "monsterSprite/gen5/0534_conkeldurr/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def tip_shift(rigs, d, name):
    rest = pose("attack", 0)
    hit = pose("attack", 11)
    ra = rig_for(rigs, d, rest)
    rb = rig_for(rigs, d, hit)
    la = limb_layers(ra, rest)[name]
    lb = limb_layers(rb, hit)[name]
    fa = rb["facing"]
    a = facing_tip(la, fa)
    b = facing_tip(lb, fa)
    if a is None or b is None:
        return None
    return round((b[0] - a[0]) * fa, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    args = ap.parse_args()

    save_peaks(Path(__file__).stem, PEAKS)
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    palette = locked_palette(SRC_DIR / "front.png", SRC_DIR / "back.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    print("parts", {k: int(v.sum()) for k, v in fm.items()}, {k: int(v.sum()) for k, v in bm.items()}, flush=True)

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
        r3 = synth(rigs, d, "attack", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r3 != rigs[d]["canvas"], axis=2).sum())
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    turn_diff = {}
    for d in DIRS:
        p = pose("attack", 6)
        rig = rig_for(rigs, d, p)
        img = synth(rigs, d, "attack", 6, palette, effects=True)
        turn_diff[d] = int(np.any(img != rig["canvas"], axis=2).sum())
    print("turn diff", turn_diff, flush=True)
    if any(turn_diff.values()):
        raise SystemExit("attack 5-8 is not the back still")

    strike = {name: tip_shift(rigs, "S", name) for name in ("pillar_l", "pillar_r")}
    print("strike dx S", strike, flush=True)

    pal = {tuple(int(v) for v in c) for c in palette}
    if args.samples:
        off = set()
        partial = 0
        edge = []
        lost = []
        bodies = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    img = synth(rigs, d, act, f, palette)
                    o, part, ed = frame_faults(img, pal)
                    off |= o
                    partial += part
                    if ed:
                        edge.append(f"{act}:{d}:{f}")
                    if act in ("attack", "hurt", "walk"):
                        bodies.append(largest_fraction(img))
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        hurt0 = synth(rigs, "S", "hurt", 0, palette, effects=False)
        hurt3 = synth(rigs, "S", "hurt", 3, palette, effects=False)
        head_dx = round(upper_mean_x(hurt3) - upper_mean_x(hurt0), 2)
        print("hurt head dx S", head_dx, "lean", round(pose("hurt", 3)["lean"], 1), flush=True)
        print("disconnected", lost[:12], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["pillar_l"], 1), round(p["pillar_r"], 1), round(p["crest"], 1),
                        p["pillar_mode"], p["turned"], p["spin"], p["wedge"], p["bdx"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        report = {
            "rest_diff_max": max(rest_diff.values()),
            "turn_diff_max": max(turn_diff.values()),
            "off_palette": len(off),
            "partial_alpha": partial,
            "edge_n": len(edge),
            "edge_touching": edge[:12],
            "disconnected": len(lost),
            "body_main_min": round(min(bodies), 3) if bodies else 1,
            "hurt_head_dx_S": head_dx,
        }
        print("qa_samples", json.dumps(report), flush=True)
        print(
            f"静止差分 {report['rest_diff_max']} パレット外 {report['off_palette']} "
            f"中間アルファ {report['partial_alpha']} 枠接触 {report['edge_n']}",
            flush=True,
        )
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
            if key not in state["done"]:
                state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview_image(sheets).save(OUT_DIR / "preview.png")
    motion = np.concatenate([motion_strip(sheets, DIRS.index(d)) for d in ("S", "W", "N", "E")], axis=0)
    Image.fromarray(motion, "RGBA").save(OUT_DIR / "preview_motion.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", OUT_DIR / "actions" / "preview.png")

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
        "turn_diff": turn_diff,
        "off_palette_colors": len(off), "partial_alpha": partial, "edge_touching": clip[:12],
        "edge_n": len(clip), "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "bounds": bounds, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (
        not any(rest_diff.values()) and not any(idle0.values()) and not any(turn_diff.values())
        and not off and partial == 0 and not clip
    )
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    print(
        f"静止差分 {max(rest_diff.values())} パレット外 {len(off)} "
        f"中間アルファ {partial} 枠接触 {len(clip)}",
        flush=True,
    )
    meta = {
        "generator": "forge_gen5_conkeldurr_actions", "version": VERSION, "key": KEY,
        "dex": 534, "variant": "conkeldurr", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped with two concrete pillars, a crest, an open mouth, and two feet",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (mouth mean x37.5 < eye mean x45.7); back has no face and is away-right as drawn",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
