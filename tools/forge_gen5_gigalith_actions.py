#!/usr/bin/env python3
"""Gigalith normal (0526) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x9-88 y8-83, 2933 px, 13 colours,
alpha 0/255) and normal/back.png (bbox x8-82 y10-85, 2669 px, 11 colours,
all already in the front palette). Lowest pixel: front y83, back y85.
The back is shifted up 2px so both ground lines share y83.

Facing: the ten gold eye pixels sit at y31-35, mean x 42.0 (left cluster
x35-36, right cluster x43-46). At the eye rows the head silhouette starts
at x30, left of the eyes, so the still faces screen-left. There is no mouth.
Pink (238, 115, 115) is crystal shine on the crests, and the grey facets at
y59-66 are rock, not teeth. The roar does not invent a mouth.

Front parts, screen axes as drawn:
  horn_l crest tip x34-48 y8-28. horn_r crest tip x47-58 y10-28.
  The crystal below the eyes stays on the face so the left eye is not cut off.
  arm_l the hanging left crystal club (three shards, x9-36 y44-80).
  arm_r the right crystal, including the spike that reaches y83.
  Two blue feet under the hips: leg_l x22-36 y73-79, leg_r x58-73 y73-80.
  Eyes, grey facets, and the cheek crystals stay on the torso.
Back, no eyes and no mouth:
  horn_r is the top spike. horn_l is the upper-left crystal plate.
  arm_l the left hanging club. arm_r the right crystal that reaches y85.
  Two feet below the hip gap.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn. NW back mirrored.
  Attack 5-8 shows that direction's back at equal scale. No |cos| squash,
  no in-plane body spin.
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
SRC_DIR = ROOT / "monsterSprite/gen5/0526_gigalith/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0526_gigalith/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0526_gigalith"
VERSION = "gigalith_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 240, 288
OFF_X = (CELL_W - 96) // 2
FRONT_FOOT = 83
BACK_FOOT = 85
# Back sole is 2px lower. Shift it up. The 38 degree hurt lean drops the
# far foot about 40px, so the cell is taller than it is wide.
OFF_Y = 128
FOOT_Y = OFF_Y + FRONT_FOOT
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Roar ceilings. The crystal strike is drawn at 34 and 30, above these,
# so finish() does not shrink the hit.
PEAKS = {"horn_l": 14.0, "horn_r": 12.0, "arm_l": 12.0, "arm_r": 10.0}
LIMBS = ("horn_l", "horn_r", "arm_l", "arm_r", "leg_l", "leg_r")

INK = (0, 0, 0)
NAVY = (32, 32, 41)
DBLUE = (32, 41, 65)
MBLUE = (41, 57, 106)
BLUE = (65, 82, 139)
GREY = (74, 74, 74)
SILVER = (131, 131, 131)
DRED = (106, 32, 32)
RED = (156, 49, 49)
BRIGHT = (230, 49, 49)
PINK = (238, 115, 115)
GOLD_D = (172, 131, 24)
GOLD = (222, 189, 90)
RED_COLS = (DRED, RED, BRIGHT, PINK)
BLUE_COLS = (NAVY, DBLUE, MBLUE, BLUE)
GOLD_COLS = (GOLD_D, GOLD)

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
    if not out.any():
        return out
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
    """4x nearest in-plane rotation. No width squash."""
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


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def check_partition(op, masks):
    acc = np.zeros_like(op)
    for name, m in masks.items():
        if (m & acc).any():
            raise SystemExit(f"overlap in {name}")
        if (m & ~op).any():
            raise SystemExit(f"{name} left the still")
        acc |= m
    if int(acc.sum()) != int(op.sum()):
        raise SystemExit(f"parts do not partition: {int(acc.sum())} vs {int(op.sum())}")


def absorb_islands(torso, limbs):
    """Torso islands join the limb they touch. The main body stays."""
    groups = component_list(torso)
    if len(groups) <= 1:
        return torso
    groups.sort(key=len, reverse=True)
    keep = np.zeros_like(torso)
    for y, x in groups[0]:
        keep[y, x] = True
    for g in groups[1:]:
        sm = np.zeros_like(torso)
        for y, x in g:
            sm[y, x] = True
        best, score = None, 0
        for name, m in limbs.items():
            s = int((sm & dilate(m, 2)).sum())
            if s > score:
                best, score = name, s
        if best is None:
            raise SystemExit(f"torso island touches nothing ({len(g)} px)")
        limbs[best] |= sm
        keep &= ~sm
    return keep


def crystal_from(img, seed, box, red):
    ink = is_color(img, INK)
    op = img[..., 3] >= 128
    allowed = (red | ink) & box & op
    return flood(allowed, seed & allowed)


def segment(img, kind):
    """Crests, two crystal arms, two feet. Face and rock facets stay on the torso."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    red = any_color(img, RED_COLS)
    if kind == "front":
        # Crest tips only. The crystal below y28 is the face and holds the
        # left eye, so it stays on the torso.
        horn_l = crystal_from(
            img, red & (xx >= 38) & (xx <= 46) & (yy <= 16),
            (xx >= 34) & (xx <= 48) & (yy <= 28), red,
        )
        horn_r = crystal_from(
            img, red & (xx >= 49) & (xx <= 56) & (yy <= 16),
            (xx >= 47) & (xx <= 58) & (yy <= 28), red,
        )
        claimed = horn_l | horn_r
        arm_l = crystal_from(
            img, red & (xx <= 32) & (yy >= 46) & ~claimed,
            (xx <= 36) & (yy >= 44) & ~claimed, red,
        )
        arm_r = crystal_from(
            img, red & (xx >= 64) & (yy >= 50) & ~claimed,
            (xx >= 57) & (yy >= 32) & ~claimed, red,
        )
        leg_y, leg_xl, leg_xr = 73, 40, 58
    else:
        horn_r = crystal_from(
            img, red & (xx >= 46) & (xx <= 54) & (yy <= 18),
            (xx >= 44) & (xx <= 56) & (yy <= 28), red,
        )
        horn_l = crystal_from(
            img, red & (xx >= 28) & (xx <= 36) & (yy >= 24) & (yy <= 32),
            (xx >= 26) & (xx <= 45) & (yy >= 22) & (yy <= 50) & ~horn_r, red,
        )
        claimed = horn_l | horn_r
        arm_l = crystal_from(
            img, red & (xx <= 24) & (yy >= 48) & ~claimed,
            (xx <= 40) & (yy >= 45) & ~claimed, red,
        )
        arm_r = crystal_from(
            img, red & (xx >= 64) & (yy >= 70) & ~claimed,
            (xx >= 55) & (yy >= 44) & ~claimed, red,
        )
        leg_y, leg_xl, leg_xr = 73, 36, 52
    crystals = horn_l | horn_r | arm_l | arm_r
    rest = op & ~crystals
    leg_l = rest & (yy >= leg_y) & (xx <= leg_xl)
    leg_r = rest & (yy >= leg_y) & (xx >= leg_xr)
    limbs = {
        "horn_l": horn_l, "horn_r": horn_r, "arm_l": arm_l, "arm_r": arm_r,
        "leg_l": leg_l, "leg_r": leg_r,
    }
    torso = absorb_islands(rest & ~(leg_l | leg_r), limbs)
    masks = dict(limbs)
    masks["torso"] = torso
    check_partition(op, masks)
    if len(component_list(torso)) != 1:
        raise SystemExit(f"{kind} torso fell apart")
    for name in LIMBS:
        part = masks[name]
        if int(part.sum()) < 8:
            raise SystemExit(f"{kind} {name} is too small ({int(part.sum())})")
        if not (part & dilate(torso, 1)).any():
            raise SystemExit(f"{kind} {name} does not touch the body")
    foot = int(np.nonzero(op)[0].max())
    want = FRONT_FOOT if kind == "front" else BACK_FOOT
    if foot != want:
        raise SystemExit(f"{kind} sole moved: {foot}")
    if kind == "front":
        gold = any_color(img, GOLD_COLS)
        if int((gold & torso).sum()) != int(gold.sum()) or int(gold.sum()) != 10:
            raise SystemExit("front eyes left the head")
        ys, xs = np.nonzero(gold)
        eye = float(xs.mean())
        band = op & (yy >= 28) & (yy <= 38) & (xx >= 28) & (xx <= 66)
        tip = int(np.nonzero(band)[1].min())
        if tip >= eye - 1.0:
            raise SystemExit(f"front does not face left: tip {tip} eye {eye:.1f}")
    else:
        if any_color(img, GOLD_COLS).any():
            raise SystemExit("back grew eyes")
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


def joint_band(part, torso, depth):
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 3)
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
        ("horn_l", 4), ("horn_r", 4), ("arm_l", 5), ("arm_r", 5),
        ("leg_l", 6), ("leg_r", 6),
    ):
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth + 1,
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
        strike, off = "arm_l", "arm_r"
    else:
        facing = -1.0 if mirror else 1.0
        strike, off = "arm_r", "arm_l"
    pivots = {}
    signs = {}
    for name in LIMBS:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing,
        "strike": strike, "off": off,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        horn_l=0.0, horn_r=0.0, horn_mode="up",
        arm_l=0.0, arm_r=0.0, arm_mode="up",
        step_l=(0, 0), step_r=(0, 0),
        spark=0, lean=0.0, knock=0, flash=False,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay. The crests and crystal arms breathe.
        p["horn_l"] = wave(f, n, 5.0, phase=0.4)
        p["horn_r"] = wave(f, n, 4.0, phase=1.1)
        p["arm_l"] = wave(f, n, 3.0, phase=0.2)
        p["arm_r"] = wave(f, n, 3.0, phase=0.8)
        p["horn_mode"] = "up"
        p["arm_mode"] = "up"
    elif action == "walk":
        # Two drawn feet, opposite steps. One foot stays down.
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * sb)), -int(round(3 * max(0.0, sb))))
        p["arm_l"] = 8.0 * -sa
        p["arm_r"] = 8.0 * -sb
        p["horn_l"] = wave(f, n, 3.0, phase=0.3)
        p["horn_r"] = wave(f, n, 3.0, phase=0.9)
        p["horn_mode"] = "up"
        p["arm_mode"] = "fwd"
    elif action == "attack":
        # 0 rest. 1-4 crests and crystals rise. No mouth.
        # 5-8 the back, equal scale. 9-12 the crystals strike and the
        # near foot stomps. 13-15 return. 15 is the still.
        horn_l = [0, 6, 10, 14, 14, 0, 0, 0, 0, 8, 14, 18, 16, 8, 3, 0]
        horn_r = [0, 5, 9, 12, 12, 0, 0, 0, 0, 6, 12, 16, 14, 6, 2, 0]
        arm_l = [0, 4, 8, 12, 12, 0, 0, 0, 0, 16, 26, 34, 34, 12, 4, 0]
        arm_r = [0, 3, 6, 10, 10, 0, 0, 0, 0, 12, 22, 30, 28, 10, 3, 0]
        p["horn_l"] = float(horn_l[f])
        p["horn_r"] = float(horn_r[f])
        p["arm_l"] = float(arm_l[f])
        p["arm_r"] = float(arm_r[f])
        p["horn_mode"] = "up" if f <= 4 else "fwd"
        p["arm_mode"] = "up" if f <= 4 else "fwd"
        p["turned"] = 5 <= f <= 8
        if f == 9:
            p["step_l"] = (1, -2)
        elif f == 10:
            p["step_l"] = (2, -3)
        elif f == 11:
            p["step_l"] = (2, 0)
        elif f == 12:
            p["step_l"] = (1, 0)
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.80, 1.0, 0.85, 0.62, 0.42, 0.24, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["horn_l"] = 0.0
        p["horn_r"] = 0.0
        p["arm_l"] = 0.0
        p["arm_r"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["spark"] = 0
        p["spin"] = 0
        p["turned"] = True
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


def limb_motion(rig, p):
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    ang = {
        "horn_l": part_angle(rig, "horn_l", p["horn_mode"], p["horn_l"]),
        "horn_r": part_angle(rig, "horn_r", p["horn_mode"], p["horn_r"]),
        "arm_l": part_angle(rig, "arm_l", p["arm_mode"], p["arm_l"]),
        "arm_r": part_angle(rig, "arm_r", p["arm_mode"], p["arm_r"]),
        "leg_l": 0.0,
        "leg_r": 0.0,
    }
    step = {"leg_l": p["step_l"], "leg_r": p["step_r"]}
    delta = {}
    for name in LIMBS:
        if name in step:
            sx, sy = step[name]
            delta[name] = (bdx + int(round(fc * sx)), bdy + int(sy))
        else:
            delta[name] = (bdx, bdy)
    return ang, delta


def compose(rig, p):
    L = rig["layers"]
    piv = rig["pivots"]
    ang, delta = limb_motion(rig, p)
    out = np.zeros_like(rig["canvas"])
    bdx, bdy = delta["horn_l"]
    paste(out, shift(L["torso"], bdx, bdy))
    order = ("leg_l", "leg_r", rig["off"], "horn_l", "horn_r", rig["strike"])
    seen = set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        dx, dy = delta[name]
        paste(out, moved(L[name], ang[name], piv[name], dx, dy))
    return out


def spark(img, origin, fc, age):
    ox, oy = origin
    cols = (GOLD, PINK, BRIGHT, GOLD_D)
    r = 4 + age * 3
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2), (2, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK)
    out[op & ~keep, :3] = np.array(PINK, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def weapon_tip(rig, p):
    name = rig["strike"]
    fc = rig["facing"]
    ang, delta = limb_motion(rig, p)
    dx, dy = delta[name]
    layer = moved(rig["layers"][name], ang[name], rig["pivots"][name], dx, dy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def limb_lost(rig, p):
    body = compose(rig, p)
    ang, delta = limb_motion(rig, p)
    lost = []
    for name in LIMBS:
        dx, dy = delta[name]
        layer = moved(rig["layers"][name], ang[name], rig["pivots"][name], dx, dy)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    other = "back" if kind == "front" else "front"
    return rigs[(other, mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    moved_part = (
        abs(p["horn_l"]) >= 0.45 or abs(p["horn_r"]) >= 0.45
        or abs(p["arm_l"]) >= 0.45 or abs(p["arm_r"]) >= 0.45
        or p["step_l"] != (0, 0) or p["step_r"] != (0, 0)
        or p["bdx"] or p["bdy"]
    )
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        # Head goes backward, away from the face.
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, fc * p["lean"], piv)
    if moved_part or abs(p["lean"]) > 0.4:
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = weapon_tip(rig, p)
        spark(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


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


def tile_grid(tiles, cols, bg=(18, 22, 32, 255)):
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


def content_box(canvases, pad=8):
    acc = np.zeros(canvases[0].shape[:2], bool)
    for c in canvases:
        acc |= c[..., 3] >= 128
    ys, xs = np.nonzero(acc)
    y0 = max(0, int(ys.min()) - pad - 28)
    y1 = min(CELL_H, int(ys.max()) + pad + 18)
    x0 = max(0, int(xs.min()) - pad - 36)
    x1 = min(CELL_W, int(xs.max()) + pad + 36)
    return y0, y1, x0, x1


def motion_strip(sheets, di, box):
    y0, y1, x0, x1 = box
    rows = []
    for act in ACTIONS:
        row = []
        for f in range(16):
            if f < FRAMES[act]:
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                row.append(t[y0:y1, x0:x1])
            else:
                row.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def frame_metrics(img, palette):
    """Off-palette colours, partial alpha, and border contact for one cell."""
    pal = {tuple(int(v) for v in c) for c in palette}
    op = img[..., 3] == 255
    partial = int(((img[..., 3] != 0) & (img[..., 3] != 255)).sum())
    off = 0
    if op.any():
        used = {tuple(int(v) for v in c) for c in np.unique(img[op][:, :3], axis=0)}
        off = len(used - pal)
    a = img[..., 3] == 255
    edge = int(a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any())
    return off, partial, edge


def locked_palette(front_path, back_path, work):
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
    imgs = [np.array(Image.open(p).convert("RGBA")) for p in (front_path, back_path)]
    own = []
    for img in imgs:
        cols = np.unique(img[img[..., 3] >= 128][:, :3], axis=0)
        own.extend(tuple(int(v) for v in c) for c in cols)
    own = sorted(set(own))
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
    fm = segment(front, "front")
    bm = segment(back, "back")
    rigs = {
        ("front", False): build_view(front, fm, "front", False),
        ("front", True): build_view(front, fm, "front", True),
        ("back", False): build_view(back, bm, "back", False),
        ("back", True): build_view(back, bm, "back", True),
    }
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    fa = rigs[("front", False)]["canvas"]
    ba = rigs[("back", False)]["canvas"]
    if int(np.nonzero(fa[..., 3] >= 128)[0].max()) != int(np.nonzero(ba[..., 3] >= 128)[0].max()):
        raise SystemExit("front and back feet do not share a line")
    if rigs[("front", False)]["facing"] != -1.0 or rigs[("back", False)]["facing"] != 1.0:
        raise SystemExit("facing locked to the wrong side")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "torso": (70, 130, 210, 255),
        "horn_l": (40, 190, 70, 255),
        "horn_r": (220, 50, 50, 255),
        "arm_l": (70, 220, 220, 255),
        "arm_r": (190, 80, 220, 255),
        "leg_l": (250, 250, 250, 255),
        "leg_r": (255, 210, 40, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 4, 0), 4, 1))
    Image.fromarray(np.concatenate(tiles, axis=1), "RGBA").save(work / "parts.png")


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
        "dex": 526, "form": "normal", "variant": "gigalith", "gen": "gen5",
        "dir": "monsterSprite/gen5/0526_gigalith/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


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
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    print("strike", {d: rigs[d]["strike"] for d in ("S", "W", "N", "E")}, flush=True)

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
    rest_sum = int(sum(rest_diff.values()))
    print("rest diff", rest_diff, "sum", rest_sum, flush=True)
    if rest_sum:
        raise SystemExit("frame 0 does not match the still")

    turn_w = synth(rigs, "W", "attack", 6, palette, effects=False)
    turn_s = synth(rigs, "S", "attack", 6, palette, effects=False)
    if np.any(turn_w != rigs["N"]["canvas"]):
        raise SystemExit("W turn is not the N back at equal scale")
    if np.any(turn_s != rigs["NW"]["canvas"]):
        raise SystemExit("S turn is not N flipped at equal scale")

    if args.samples:
        lost = {}
        off = partial = edge = 0
        bodies = []
        picks = {
            "idle": (0, 8),
            "walk": (0, 4, 12),
            "attack": (0, 2, 4, 6, 8, 11, 12, 15),
            "hurt": (0, 3, 6, 11),
        }
        for d in DIRS:
            for act, fs in picks.items():
                for f in fs:
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost[f"{act}:{d}:{f}"] = bad
                    t = synth(rigs, d, act, f, palette)
                    o, pa, ed = frame_metrics(t, palette)
                    off += o
                    partial += pa
                    edge += ed
                    if act in ("walk", "attack", "hurt") and f != 0:
                        bodies.append(largest_fraction(t))
        leans = [round(pose("hurt", f)["lean"], 2) for f in range(12)]
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["horn_l"], 1), round(p["horn_r"], 1),
                        round(p["arm_l"], 1), round(p["arm_r"], 1),
                        int(p["turned"]), int(p["spin"])))
        print("limb lost", lost or "none", flush=True)
        print("hurt lean", leans, "max", max(leans), flush=True)
        print("attack poses", atk, flush=True)
        print(
            "SAMPLE rest_diff", rest_sum,
            "off_palette", off,
            "partial_alpha", partial,
            "edge", edge,
            "body_min", round(min(bodies), 3) if bodies else 1,
            flush=True,
        )
        box = content_box([rigs[d]["canvas"] for d in DIRS])
        y0, y1, x0, x1 = box
        tiles = []
        for d in ("S", "W", "N", "E"):
            for act, f in (
                ("idle", 0), ("idle", 8),
                ("walk", 4), ("walk", 12),
                ("attack", 0), ("attack", 3), ("attack", 6), ("attack", 8),
                ("attack", 11), ("attack", 14),
                ("hurt", 1), ("hurt", 3),
            ):
                t = synth(rigs, d, act, f, palette)
                tiles.append(t[y0:y1, x0:x1])
        sheet = np.repeat(np.repeat(tile_grid(tiles, 12), 2, 0), 2, 1)
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
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest and not args.force:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    box = content_box([rigs[d]["canvas"] for d in DIRS])
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.repeat(np.repeat(np.concatenate(strips, axis=0), 2, 0), 2, 1)
    Image.fromarray(motion, "RGBA").save(OUT_DIR / "preview_motion.png")
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
    ok = (
        not any(rest_diff.values()) and not any(idle0.values()) and not off
        and partial == 0 and not clip and min(bodies) >= 0.90
    )
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_gigalith_actions", "version": VERSION, "key": KEY,
        "dex": 526, "variant": "gigalith", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped rock (two crystal crests, two crystal arms, two feet, no mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (head tip x30 < eye mean x42); gold pixels are eyes",
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
