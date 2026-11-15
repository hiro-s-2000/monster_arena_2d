#!/usr/bin/env python3
"""Leavanny normal (0542) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x27-69 y13-87, 1237 px, 14 colours,
alpha 0/255, sole y87):
  Two eyes, no mouth. Maroon (131, 49, 65), red (255, 49, 74) and white
  (255, 255, 255) form the left eye at x39-40 y36-38 (mean x 39.6) and the
  right eye at x49-51 y37-39 (mean x 49.83). Pair mean x is 45.18.
  The yellow muzzle (255, 246, 32) in y30-43 is x41-52. Its chin, the lowest
  yellow row y40 x44-46, has mean x 45.0, left of the eyes, and the left eye
  sits on the left contour. The still faces screen-left.
  The red and white pixels are the eyes, so the roar does not invent a mouth,
  a wedge, or a ring. It raises the two antennae and the two leaf arms.
  Antennae are the thin stalks above y19: left x36-39 y14-18, right x49-53
  y13-18. Leaf arms are the separated side blades: left x27-36 y50-63, right
  x57-69 y53-65. Legs split at y73. The screen-left leg keeps the long gold
  leaf foot down to y87. The screen-right leg ends at y76. Face, hood and
  dress stay on the torso. No wings, horns, tail, or extra legs.
normal/back.png (bbox x30-71 y11-87, 1142 px, 10 colours, all in the front
  palette, sole y87): no eyes and no mouth. Antennae y11-18, leaf arms as the
  side lobes, two legs from y69. The screen-right leg keeps the gold foot
  that reaches y87. Same ground line, so no vertical shift.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (away, screen-right). NW back mirrored.
  E is screen-right, W is screen-left, N is the back.
  Attack 5-8 shows that direction's back at equal scale. Back rows stay.
  No |cos| squash and no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_leavanny_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0542_leavanny/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0542_leavanny/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0542_leavanny"
VERSION = "leavanny_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 256, 256
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 24
FRONT_FOOT = 87
BACK_FOOT = 87
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. W keeps that picture. Its back is N.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Wind-up ceiling. The leaf slash is drawn at 34, above this, so finish()
# does not push a frame past the angle this script already paints.
# Frames 5-8 are cleared after finish() so the back still is not swung.
PEAKS = {"arm": 16.0}
LIMBS = ("arm_l", "arm_r", "leg_l", "leg_r", "ant_l", "ant_r")

INK = (0, 0, 0)
LIME = (123, 205, 8)
YELLOW = (255, 246, 32)
WHITE = (255, 255, 255)
RED = (255, 49, 74)
MAROON = (131, 49, 65)
EYE_COLS = (MAROON, RED, WHITE)

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
    """4x nearest in-plane rotation. Positive is clockwise on screen. No squash."""
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


def runs_of(row):
    xs = np.nonzero(row)[0]
    if len(xs) == 0:
        return []
    out = []
    s = int(xs[0])
    prev = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x > prev + 1:
            out.append((s, prev))
            s = x
        prev = x
    out.append((s, prev))
    return out


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


def touches(part, body):
    return bool((part & dilate(body, 1)).any())


def side_blades(op, y0, y1, left_max, right_min):
    """Separated side runs. The dress between the gaps stays off the blades."""
    left = np.zeros_like(op)
    right = np.zeros_like(op)
    for y in range(y0, y1 + 1):
        rs = runs_of(op[y])
        if len(rs) < 2:
            continue
        a, b = rs[0]
        if b <= left_max and rs[1][0] > b + 1:
            left[y, a:b + 1] = True
        c, d = rs[-1]
        if c >= right_min and rs[-2][1] < c - 1:
            right[y, c:d + 1] = True
    return left, right


def legs_from(op, y0):
    """Two columns from the first split, then only the pixels that continue them."""
    h, w = op.shape
    left = np.zeros_like(op)
    right = np.zeros_like(op)
    rs = runs_of(op[y0])
    if len(rs) < 2:
        raise SystemExit(f"legs do not split at y{y0}")
    left[y0, rs[0][0]:rs[0][1] + 1] = True
    right[y0, rs[-1][0]:rs[-1][1] + 1] = True
    for y in range(y0 + 1, h):
        rs = runs_of(op[y])
        if not rs:
            break
        prev_l = left[y - 1]
        prev_r = right[y - 1]
        if not prev_l.any() and not prev_r.any():
            break
        wide_l = prev_l.copy()
        wide_r = prev_r.copy()
        wide_l[1:] |= prev_l[:-1]
        wide_l[:-1] |= prev_l[1:]
        wide_r[1:] |= prev_r[:-1]
        wide_r[:-1] |= prev_r[1:]
        for a, b in rs:
            if b - a + 1 > 16:
                continue
            seg = np.zeros(w, bool)
            seg[a:b + 1] = True
            hit_l = bool((seg & wide_l).any())
            hit_r = bool((seg & wide_r).any())
            if hit_l and not hit_r:
                left[y, a:b + 1] = True
            elif hit_r and not hit_l:
                right[y, a:b + 1] = True
            elif hit_l and hit_r:
                mx = 0.5 * (a + b)
                lxs = np.nonzero(prev_l)[0]
                rxs = np.nonzero(prev_r)[0]
                lc = float(lxs.mean()) if len(lxs) else 0.0
                rc = float(rxs.mean()) if len(rxs) else 99.0
                if abs(mx - lc) <= abs(mx - rc):
                    left[y, a:b + 1] = True
                else:
                    right[y, a:b + 1] = True
    return left, right


def antennae(op, y_cut, single):
    """Thin stalks at or above y_cut. `single` is which stalk exists alone."""
    left = np.zeros_like(op)
    right = np.zeros_like(op)
    for y in range(0, y_cut + 1):
        rs = runs_of(op[y])
        if not rs:
            continue
        if len(rs) == 1:
            a, b = rs[0]
            if single == "left":
                left[y, a:b + 1] = True
            else:
                right[y, a:b + 1] = True
        else:
            a, b = rs[0]
            c, d = rs[-1]
            left[y, a:b + 1] = True
            right[y, c:d + 1] = True
    return left, right


def segment(img, kind):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    if kind == "front":
        arm_l, arm_r = side_blades(op, 48, 70, 40, 57)
        leg_l, leg_r = legs_from(op, 73)
        ant_l, ant_r = antennae(op, 18, "right")
        foot = FRONT_FOOT
    else:
        arm_l, arm_r = side_blades(op, 49, 68, 46, 58)
        leg_l, leg_r = legs_from(op, 69)
        ant_l, ant_r = antennae(op, 18, "left")
        foot = BACK_FOOT
    masks = {
        "arm_l": arm_l, "arm_r": arm_r,
        "leg_l": leg_l, "leg_r": leg_r,
        "ant_l": ant_l, "ant_r": ant_r,
    }
    # A blade cut can leave a few dress pixels that only touched the blade.
    # Those belong to the blade, not to a second torso.
    used = arm_l | arm_r | leg_l | leg_r | ant_l | ant_r
    rest = op & ~used
    groups = component_list(rest)
    groups.sort(key=len, reverse=True)
    for g in groups[1:]:
        if len(g) > 12:
            raise SystemExit(f"{kind} torso lost a large piece ({len(g)} px)")
        speck = np.zeros_like(op)
        for y, x in g:
            speck[y, x] = True
        hits = [name for name in LIMBS if touches(speck, masks[name])]
        if len(hits) != 1:
            raise SystemExit(f"{kind} speck {g[:4]} touches {hits}")
        masks[hits[0]] |= speck
    used = np.zeros_like(op)
    for name in LIMBS:
        used |= masks[name]
    masks["torso"] = op & ~used
    torso = masks["torso"]
    check_partition(op, masks)
    for name in LIMBS:
        part = masks[name]
        if int(part.sum()) < 8:
            raise SystemExit(f"{kind} {name} is too small: {int(part.sum())}")
        if not touches(part, torso):
            raise SystemExit(f"{kind} {name} does not touch the body")
        if largest_component(part) != int(part.sum()):
            raise SystemExit(f"{kind} {name} fell apart")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit(f"{kind} torso fell apart")
    grounded = op & (yy == foot)
    feet = masks["leg_l"] | masks["leg_r"]
    if int(grounded.sum()) < 1 or int((grounded & feet).sum()) != int(grounded.sum()):
        raise SystemExit(f"{kind} ground line is not a foot")
    if kind == "front":
        eyes = any_color(img, EYE_COLS)
        if int(eyes.sum()) != 11 or int((eyes & torso).sum()) != 11:
            raise SystemExit("front eyes left the head")
        ex = float(np.nonzero(eyes)[1].mean())
        yellow = is_color(img, YELLOW) & (yy >= 30) & (yy <= 43)
        chin_y = int(np.nonzero(yellow)[0].max())
        chin_x = float(np.nonzero(yellow & (yy == chin_y))[1].mean())
        if not (chin_x < ex):
            raise SystemExit(f"front does not face left: chin {chin_x} eye {ex}")
        left_eye = eyes & (xx < 45)
        right_eye = eyes & (xx >= 45)
        if int(left_eye.sum()) < 4 or int(right_eye.sum()) < 4:
            raise SystemExit("front is missing an eye")
    else:
        if any_color(img, EYE_COLS).any():
            raise SystemExit("back gained an eye colour")
    return masks


def place(arr):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    out[OFF_Y:OFF_Y + 96, OFF_X:OFF_X + 96] = arr
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
        if len(xs) == 0:
            return 0.0, 0.0
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(torso)
    cx, cy = float(tx.mean()), float(ty.mean())
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

    x_pos, y_pos = tip_after(18.0)
    x_neg, y_neg = tip_after(-18.0)
    up = -1.0 if y_neg < y_pos else 1.0
    if facing > 0:
        fwd = 1.0 if x_pos >= x_neg else -1.0
    else:
        fwd = 1.0 if x_pos <= x_neg else -1.0
    return {"up": up, "fwd": fwd}


def build_view(img, masks, kind, mirror):
    canvas = place(img)
    placed = {k: place(v) for k, v in masks.items()}
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
        ("arm_l", 6), ("arm_r", 6), ("leg_l", 6), ("leg_r", 6), ("ant_l", 4), ("ant_r", 4),
    ):
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth,
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in LIMBS:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "has_mouth": False,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        arm=0.0, arm_mode="up",
        ant_l=0.0, ant_r=0.0,
        step_l=(0, 0), step_r=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay. Antennae and the leaf blades breathe. Frame 0 is the still.
        p["ant_l"] = wave(f, n, 8.0, phase=0.4)
        p["ant_r"] = wave(f, n, 7.0, phase=1.3)
        p["arm"] = wave(f, n, 5.0, phase=0.8)
        p["arm_mode"] = "up"
    elif action == "walk":
        # Two drawn legs, opposite steps. A planted sole does not rise.
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * sb)), -int(round(3 * max(0.0, sb))))
        p["arm"] = wave(f, n, 8.0, cycles=2.0, phase=0.3)
        p["arm_mode"] = "fwd"
        p["ant_l"] = wave(f, n, 6.0, cycles=2.0, phase=0.6)
        p["ant_r"] = wave(f, n, 6.0, cycles=2.0, phase=1.8)
    elif action == "attack":
        # 0 rest. 1-4 raise the antennae and the leaf arms (no mouth).
        # 5-8 the back, equal scale, cleared after finish().
        # 9 wind-up. 10-12 the leaf slash. 13-15 return. 15 is the still.
        arm = [0, 8, 14, 18, 20, 0, 0, 0, 0, 18, 28, 34, 34, 12, 4, 0]
        ant = [0, 4, 8, 10, 10, 0, 0, 0, 0, 4, 2, 0, 0, 0, 0, 0]
        bdx = [0, -1, -1, -1, -1, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        p["arm"] = float(arm[f])
        p["ant_l"] = float(ant[f])
        p["ant_r"] = float(ant[f])
        p["bdx"] = int(bdx[f])
        p["arm_mode"] = "up" if f <= 4 else "fwd"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.64, 0.44, 0.26, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["arm"] = 10.0 * env
        p["arm_mode"] = "up"
        p["ant_l"] = 8.0 * env
        p["ant_r"] = 8.0 * env
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["arm"] = 0.0
        p["ant_l"] = 0.0
        p["ant_r"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["spark"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["turned"] = True
        p["arm_mode"] = "up"
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


def limb_order(fc):
    # The blade on the facing side is pasted last so the slash stays in front.
    if fc < 0:
        return ("leg_l", "leg_r", "arm_r", "arm_l", "ant_l", "ant_r")
    return ("leg_r", "leg_l", "arm_l", "arm_r", "ant_l", "ant_r")


def limb_delta(rig, p, name):
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    if name == "leg_l":
        sx, sy = p["step_l"]
        return 0.0, bdx + int(round(fc * sx)), bdy + int(sy)
    if name == "leg_r":
        sx, sy = p["step_r"]
        return 0.0, bdx + int(round(fc * sx)), bdy + int(sy)
    if name in ("arm_l", "arm_r"):
        return part_angle(rig, name, p["arm_mode"], p["arm"]), bdx, bdy
    if name == "ant_l":
        return part_angle(rig, name, "up", p["ant_l"]), bdx, bdy
    if name == "ant_r":
        return part_angle(rig, name, "up", p["ant_r"]), bdx, bdy
    return 0.0, bdx, bdy


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    for name in limb_order(fc):
        ang, dx, dy = limb_delta(rig, p, name)
        paste(out, moved(L[name], ang, piv[name], dx, dy))
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


def top_x(img):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    top = ys <= int(ys.min()) + 2
    return float(xs[top].mean())


def lean_angle(facing, deg):
    """Head moves opposite facing. Positive rotation is clockwise on screen."""
    return facing * deg


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (YELLOW, WHITE, LIME)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2), (2, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def forward_tip(rig, p):
    fc = rig["facing"]
    best = None
    for name in ("arm_l", "arm_r"):
        ang, dx, dy = limb_delta(rig, p, name)
        layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
        m = layer[..., 3] >= 128
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
        pt = (float(xs[i]), float(ys[i]))
        if best is None:
            best = pt
        elif fc > 0 and pt[0] > best[0]:
            best = pt
        elif fc < 0 and pt[0] < best[0]:
            best = pt
    if best is None:
        return rig["pivots"]["center"]
    return best


def rig_for(rigs, d, p):
    """Attack 5-8 shows this side's back at equal scale. Back rows stay."""
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", mirror)]


def moving(p):
    if p["step_l"] != (0, 0) or p["step_r"] != (0, 0):
        return True
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    if abs(p["arm"]) >= 0.45 or abs(p["ant_l"]) >= 0.45 or abs(p["ant_r"]) >= 0.45:
        return True
    if abs(p["lean"]) > 0.4:
        return True
    return False


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        body = rotate_nearest(body, lean_angle(fc, p["lean"]), rear_pivot(body, fc))
    if moving(p) or abs(p["lean"]) > 0.4:
        body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    else:
        img = rig["canvas"].copy()
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = forward_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name in LIMBS:
        ang, dx, dy = limb_delta(rig, p, name)
        layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


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


def content_box(canvases, pad=8):
    acc = np.zeros_like(canvases[0][..., 3], bool)
    for c in canvases:
        acc |= c[..., 3] >= 128
    ys, xs = np.nonzero(acc)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(CELL_H, int(ys.max()) + pad + 8)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(CELL_W, int(xs.max()) + pad)
    return y0, y1, x0, x1


def preview_image(sheets):
    pick = {"idle": 4, "walk": 4, "attack": 11, "hurt": 3}
    tiles = []
    for act in ACTIONS:
        for di in range(8):
            f = pick[act]
            tiles.append(sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W])
    return Image.fromarray(tile_grid(tiles, 8), "RGBA")


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


def locked_palette(front_path, back_path, work):
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
    imgs = [np.array(Image.open(p).convert("RGBA")) for p in (front_path, back_path)]
    own = []
    for img in imgs:
        cols = np.unique(img[img[..., 3] >= 128][:, :3], axis=0)
        own.extend(tuple(int(v) for v in c) for c in cols)
    own = sorted(set(own))

    def matches():
        if not (lock_dir / "palette.json").exists():
            return False
        data = json.loads((lock_dir / "palette.json").read_text())
        got = {tuple(int(c["hex"][i:i + 2], 16) for i in (1, 3, 5)) for c in data["colors"]}
        return got == set(own)

    if not matches():
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
    if got != set(own):
        raise SystemExit(f"locked palette differs from the stills: {sorted(got ^ set(own))}")
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
    if int(np.nonzero(front[..., 3])[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    if int(np.nonzero(back[..., 3])[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line changed")
    fm = segment(front, "front")
    bm = segment(back, "back")
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs[("front", False)]["facing"] != -1.0 or rigs[("back", False)]["facing"] != 1.0:
        raise SystemExit("facing locked to the wrong side")
    if rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("this still has no mouth")
    # のけぞり: the original crown pixel moves opposite the facing.
    # The new top row is a risen flank, so it is not the crown.
    for key in (("front", False), ("front", True), ("back", False)):
        rig = rigs[key]
        fc = rig["facing"]
        shift = crown_shift(rig["canvas"], fc, MAX_LEAN)
        if fc > 0 and not (shift < -5):
            raise SystemExit(f"hurt lean does not tip the head back ({shift})")
        if fc < 0 and not (shift > 5):
            raise SystemExit(f"hurt lean does not tip the head back ({shift})")
    return front, back, fm, bm, rigs


def crown_shift(img, facing, deg):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    i = int(ys.argmin())
    piv = rear_pivot(img, facing)
    rx = float(xs[i]) - piv[0]
    ry = float(ys[i]) - piv[1]
    ang = lean_angle(facing, deg)
    r = math.radians(ang)
    c, s = math.cos(r), math.sin(r)
    return (piv[0] + c * rx + s * ry) - float(xs[i])


def save_parts(work, front, fm, back, bm):
    cols = {
        "arm_l": (220, 60, 40, 255),
        "arm_r": (255, 180, 40, 255),
        "leg_l": (40, 200, 90, 255),
        "leg_r": (40, 160, 220, 255),
        "ant_l": (200, 80, 220, 255),
        "ant_r": (80, 80, 220, 255),
        "torso": (170, 170, 170, 255),
    }
    tiles = []
    for masks in (fm, bm):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 3, 0), 3, 1))
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
        "dex": 542, "form": "normal", "variant": "leavanny", "gen": "gen5",
        "dir": "monsterSprite/gen5/0542_leavanny/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def frame_metrics(rigs, palette):
    pal = {tuple(int(v) for v in c) for c in palette}
    off = set()
    partial = 0
    clip = []
    bodies = []
    for d in DIRS:
        for act in ACTIONS:
            for f in range(FRAMES[act]):
                t = synth(rigs, d, act, f, palette)
                a = t[..., 3]
                partial += int(((a != 0) & (a != 255)).sum())
                op = a == 255
                if op.any():
                    used = {tuple(int(v) for v in c) for c in np.unique(t[op][:, :3], axis=0)}
                    off |= used - pal
                if op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
                if act in ("attack", "hurt", "walk"):
                    bodies.append(largest_fraction(t))
    return {
        "off_palette": len(off),
        "partial_alpha": partial,
        "edge_touch": len(clip),
        "edge_touching": clip[:8],
        "body_main_min": round(min(bodies), 3) if bodies else 1,
    }


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

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    turn_w = synth(rigs, "W", "attack", 6, palette, effects=False)
    turn_s = synth(rigs, "S", "attack", 6, palette, effects=False)
    turn_n = synth(rigs, "N", "attack", 6, palette, effects=False)
    if np.any(turn_w != rigs["N"]["canvas"]):
        raise SystemExit("W turn is not the N back at equal scale")
    if np.any(turn_s != rigs["NW"]["canvas"]):
        raise SystemExit("S turn is not the mirrored back at equal scale")
    if np.any(turn_n != rigs["N"]["canvas"]):
        raise SystemExit("N turn left the back still")

    if args.samples:
        lost = {}
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost[f"{act}:{d}:{f}"] = bad
        print("limb lost", lost or "none", flush=True)
        metrics = frame_metrics(rigs, palette)
        print("qa_samples", "rest_diff", int(sum(rest_diff.values())),
              "off_palette", metrics["off_palette"],
              "partial_alpha", metrics["partial_alpha"],
              "edge_touch", metrics["edge_touch"],
              "body_main_min", metrics["body_main_min"],
              "clip", metrics["edge_touching"], flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(float(p["arm"]), 1), p["arm_mode"], p["turned"],
                        p["spin"], round(float(p["ant_l"]), 1), p["bdx"], p["spark"]))
        print("attack poses", atk, flush=True)
        leans = [round(float(pose("hurt", f)["lean"]), 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        box = content_box([rigs[d]["canvas"] for d in DIRS])
        y0, y1, x0, x1 = box
        tiles = []
        for d in ("S", "W", "N", "E"):
            for act, f in (
                ("idle", 0), ("idle", 8),
                ("walk", 4), ("walk", 12),
                ("attack", 0), ("attack", 3), ("attack", 6), ("attack", 11),
                ("hurt", 3),
            ):
                tiles.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
        sheet = np.repeat(np.repeat(tile_grid(tiles, 9), 2, 0), 2, 1)
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
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest:
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
    ok = (
        not any(rest_diff.values()) and not any(idle0.values()) and not off
        and partial == 0 and not clip and min(bodies) >= 0.90
    )
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print("qa_sheet", "rest_diff", int(sum(rest_diff.values()) + sum(idle0.values())),
          "off_palette", len(off), "partial_alpha", partial, "edge_touch", len(clip),
          "body_main_min", report["body_main_min"], "clip", clip[:8], flush=True)
    meta = {
        "generator": "forge_gen5_leavanny_actions", "version": VERSION, "key": KEY,
        "dex": 542, "variant": "leavanny", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped leaf-arm striker (two eyes, no mouth, two antennae, two leaf blades, two legs, gold leaf foot)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (yellow chin mean x 45.0 < eye mean x 45.18)",
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
