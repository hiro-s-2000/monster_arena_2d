#!/usr/bin/env python3
"""Mega Audino (0531) 8-direction actions from this form's own stills.

Measured on mega/audino-mega/front.png (96x96, bbox x26-69 y16-78, 1709 px,
12 colours, alpha 0/255, foot y78):
  Eyes are (223,123,149), 12 px, mean x 44.5. The nose is the closed brown
  mark (164,139,98), 4 px, mean x 43.75. The nose is left of the eyes, so
  the still faces screen-left. There is no mouth cavity, so the roar does
  not draw a wedge or rings. It lifts the pink feeler tips already on the
  ears and the two pink hands already at the hips.
  Left ear lobe x30-43 y16-27 (high tip y16 x31-32) and its hanging flap
  x26-34 y39-46. Right ear lobe x53-69 y17-27 and flap x60-68 y40-50.
  Snout-side hand x31-34 y64-71. Far hand x58-63 y64-71.
  Feet from y72: snout-side x36-45 ends y77, far foot x48-57 ends y78.
mega/audino-mega/back.png (bbox x26-69 y16-79, 1649 px, foot y79) has its
  own colours. The high ear tip is y16 x65-66, on the screen-right, so this
  back faces away toward screen-right. It is shifted up 1px onto the front
  foot line. No mouth, no tail.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (away, snout side screen-right). NW back mirrored.
  E is screen-right, W is screen-left, N is the back. Attack 5-8 swaps to
  the opposite view at equal scale. No width squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_audino_mega_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0531_audino/mega/audino-mega"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0531_audino/mega/audino-mega/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0531_audino-mega"
VERSION = "audino-mega_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 224, 224
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 32
FRONT_FOOT = 78
BACK_FOOT = 79
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. Back as drawn looks away to the right.
# S (front mirrored) turns into N (back as drawn) by flipping the mirror flag.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Roar / turn hold. The hand strike is drawn up to 34 deg, above these.
PEAKS = {"ear": 16.0, "hand": 14.0}
LIMBS = (
    "ear_l", "ear_r", "flap_l", "flap_r",
    "hand_fore", "hand_rear", "leg_l", "leg_r",
)
ROTATE = ("ear_l", "ear_r", "flap_l", "flap_r", "hand_fore", "hand_rear")
TIP = {
    "ear_l": "top", "ear_r": "top",
    "flap_l": "bottom", "flap_r": "bottom",
    "hand_fore": "far", "hand_rear": "far",
}

INK_F = (24, 24, 24)
INK_B = (16, 16, 16)
WHITE = (251, 251, 251)
PINK = (255, 179, 174)
CREAM = (246, 230, 172)
EYE = (223, 123, 149)
NOSE = (164, 139, 98)
BPINK = (255, 191, 202)
BDARK = (179, 89, 104)

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
    """4x nearest in-plane rotation. Positive angle follows the shared forge formula. No squash."""
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
    """Bridge a 1px gap opened by a turn. Copies a colour already in the sprite."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(3):
        groups = component_list(out[..., 3] >= 128)
        if len(groups) <= 1:
            return out
        groups.sort(key=len, reverse=True)
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
        grown_up = np.zeros_like(src)
        grown_dn = np.zeros_like(src)
        grown_lt = np.zeros_like(src)
        grown_rt = np.zeros_like(src)
        grown_up[:-1] = src[1:]
        grown_dn[1:] = src[:-1]
        grown_lt[:, :-1] = src[:, 1:]
        grown_rt[:, 1:] = src[:, :-1]
        for grown in (grown_up, grown_dn, grown_lt, grown_rt):
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


def row_spans(op, y):
    xs = np.nonzero(op[y])[0]
    if len(xs) == 0:
        return []
    groups = []
    start = prev = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x > prev + 1:
            groups.append((start, prev))
            start = x
        prev = x
    groups.append((start, prev))
    return groups


def upper_lobe(op, y0, y1, side, split_x):
    m = np.zeros_like(op)
    for y in range(y0, y1 + 1):
        for a, b in row_spans(op, y):
            if side == "L" and b < split_x:
                m[y, a:b + 1] = True
            elif side == "R" and a > split_x:
                m[y, a:b + 1] = True
    return m & op


def side_flap(op, y0, y1, side, max_x_for_left, min_x_for_right):
    m = np.zeros_like(op)
    for y in range(y0, y1 + 1):
        spans = row_spans(op, y)
        if len(spans) < 2:
            continue
        if side == "L":
            a, b = spans[0]
            if b <= max_x_for_left:
                m[y, a:b + 1] = op[y, a:b + 1]
        else:
            a, b = spans[-1]
            if a >= min_x_for_right:
                m[y, a:b + 1] = op[y, a:b + 1]
    return m


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


def absorb_feelers(masks, img, colors):
    """A feeler pixel sitting on the outline next to a part travels with that part."""
    feel = any_color(img, colors)
    used = np.zeros(feel.shape, bool)
    for m in masks.values():
        used |= m
    free = feel & ~used
    for name in ("ear_l", "ear_r", "flap_l", "flap_r", "hand_fore", "hand_rear"):
        extra = free & dilate(masks[name], 1)
        if extra.any():
            masks[name] = masks[name] | extra
            free = free & ~extra


def require_solid(masks, torso):
    for name in LIMBS:
        if int(masks[name].sum()) < 8:
            raise SystemExit(f"{name} is too small")
        if not touches(masks[name], torso):
            raise SystemExit(f"{name} does not touch the body")
        if largest_component(masks[name]) != int(masks[name].sum()):
            raise SystemExit(f"{name} fell apart")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("torso fell apart")


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    ear_l = upper_lobe(op, 16, 27, "L", 48)
    ear_r = upper_lobe(op, 16, 27, "R", 50)
    flap_l = side_flap(op, 39, 46, "L", 38, 0)
    flap_r = side_flap(op, 39, 50, "R", 0, 58)
    hand_fore = op & (yy >= 64) & (yy <= 71) & (xx <= 34)
    hand_rear = op & (yy >= 64) & (yy <= 71) & (xx >= 58)
    leg_l = op & (yy >= 72) & (xx <= 46)
    leg_r = op & (yy >= 72) & (xx >= 47)
    masks = {
        "ear_l": ear_l, "ear_r": ear_r, "flap_l": flap_l, "flap_r": flap_r,
        "hand_fore": hand_fore, "hand_rear": hand_rear, "leg_l": leg_l, "leg_r": leg_r,
    }
    absorb_feelers(masks, img, (PINK,))
    used = np.zeros_like(op)
    for m in masks.values():
        used |= m
    torso = op & ~used
    masks["torso"] = torso
    check_partition(op, masks)
    eye = is_color(img, EYE)
    nose = is_color(img, NOSE)
    if int((eye & torso).sum()) != int(eye.sum()) or int(eye.sum()) != 12:
        raise SystemExit("front eyes left the head")
    if int((nose & torso).sum()) != int(nose.sum()) or int(nose.sum()) != 4:
        raise SystemExit("front nose left the head")
    if float(np.nonzero(nose)[1].mean()) >= float(np.nonzero(eye)[1].mean()):
        raise SystemExit("front does not face left")
    if not ear_l[16, 31]:
        raise SystemExit("front high feeler left the left ear")
    pink = is_color(img, PINK)
    if int((pink & (hand_fore | hand_rear | ear_l | ear_r)).sum()) != int(pink.sum()):
        raise SystemExit("front pink left the ears and hands")
    if int(np.nonzero(leg_r)[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line moved")
    require_solid(masks, torso)
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    ear_l = upper_lobe(op, 16, 27, "L", 46)
    ear_r = upper_lobe(op, 16, 27, "R", 50)
    flap_l = side_flap(op, 41, 49, "L", 40, 0)
    flap_r = side_flap(op, 41, 48, "R", 0, 58)
    hand_rear = op & (yy >= 63) & (yy <= 70) & (xx <= 36)
    hand_fore = op & (yy >= 63) & (yy <= 70) & (xx >= 59)
    leg_l = op & (yy >= 72) & (xx <= 47)
    leg_r = op & (yy >= 72) & (xx >= 48)
    masks = {
        "ear_l": ear_l, "ear_r": ear_r, "flap_l": flap_l, "flap_r": flap_r,
        "hand_fore": hand_fore, "hand_rear": hand_rear, "leg_l": leg_l, "leg_r": leg_r,
    }
    absorb_feelers(masks, img, (BPINK, BDARK))
    used = np.zeros_like(op)
    for m in masks.values():
        used |= m
    torso = op & ~used
    masks["torso"] = torso
    check_partition(op, masks)
    if not masks["ear_r"][16, 65]:
        raise SystemExit("back high tip is not on the right ear")
    if int(np.nonzero(leg_r)[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line moved")
    if float(np.nonzero(masks["hand_fore"])[1].mean()) <= float(np.nonzero(torso)[1].mean()):
        raise SystemExit("back snout-side hand is not on the right")
    require_solid(masks, torso)
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


def tip_xy(layer, pivot, mode):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    if mode == "top":
        band = ys <= ys.min() + 1
        return float(xs[band].mean()), float(ys[band].mean())
    if mode == "bottom":
        band = ys >= ys.max() - 1
        return float(xs[band].mean()), float(ys[band].mean())
    d = (xs - pivot[0]) ** 2 + (ys - pivot[1]) ** 2
    i = int(d.argmax())
    return float(xs[i]), float(ys[i])


def signs_for(layer, pivot, facing, mode):
    """Which rotation sign lifts the tip, and which sends it toward facing."""
    if int((layer[..., 3] >= 128).sum()) < 4:
        return {"up": 1.0, "fwd": 1.0}

    def tip_after(ang):
        return tip_xy(rotate_nearest(layer, ang, pivot), pivot, mode)

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
        ("ear_l", 4), ("ear_r", 4), ("flap_l", 4), ("flap_r", 4),
        ("hand_fore", 4), ("hand_rear", 4), ("leg_l", 3), ("leg_r", 3),
    ):
        layers["torso"] = paint_cushion(layers["torso"], joint_band(placed[name], torso_m, depth), depth)
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in ROTATE:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing, TIP[name])
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        ear=0.0, hand=0.0, hand_rear=0.0,
        hand_mode="up",
        step_l=(0, 0), step_r=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The ears and the hanging flaps sway. Frame 0 is the still.
        p["ear"] = wave(f, n, 6.0, phase=0.4)
    elif action == "walk":
        # Two drawn feet, opposite steps. One foot stays down each frame.
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * sb)), -int(round(3 * max(0.0, sb))))
        p["ear"] = wave(f, n, 5.0)
        p["hand"] = wave(f, n, 5.0, phase=0.3)
        p["hand_rear"] = wave(f, n, 5.0, phase=0.3 + math.pi)
    elif action == "attack":
        # 0 rest. 1-4 feelers and hands up (no mouth). 5-8 back view, equal scale.
        # 9-10 wind the snout-side hand (8-22 deg). 11-12 strike at 34. 13-15 recover.
        ear = [0, 6, 12, 16, 16, 14, 14, 14, 14, 12, 10, 8, 6, 3, 0, 0]
        hand = [0, 6, 10, 14, 14, 8, 8, 8, 8, 18, 22, 34, 34, 12, 4, 0]
        hand_rear = [0, 4, 6, 8, 8, 4, 4, 4, 4, -6, -10, -14, -12, -4, 0, 0]
        bdx = [0, 0, -1, -1, -1, 0, 0, 0, 0, 1, 1, 2, 2, 1, 0, 0]
        p["ear"] = float(ear[f])
        p["hand"] = float(hand[f])
        p["hand_rear"] = float(hand_rear[f])
        p["bdx"] = int(bdx[f])
        p["hand_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.75, 1.0, 0.85, 0.62, 0.42, 0.25, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["ear"] = 8.0 * env
        p["flash"] = f == 1
    return finish(p, action, f, __file__)


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


def part_motion(rig, p, name):
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    if name == "leg_l":
        sx, sy = p["step_l"]
        ang = 0.0
    elif name == "leg_r":
        sx, sy = p["step_r"]
        ang = 0.0
    elif name in ("ear_l", "ear_r", "flap_l", "flap_r"):
        sx, sy = 0, 0
        ang = part_angle(rig, name, "up", p["ear"])
    elif name == "hand_fore":
        sx, sy = 0, 0
        ang = part_angle(rig, name, p["hand_mode"], p["hand"])
    else:
        sx, sy = 0, 0
        ang = part_angle(rig, name, p["hand_mode"], p["hand_rear"])
    dx = bdx + int(round(fc * sx))
    return ang, dx, int(sy)


def compose(rig, p):
    L = rig["layers"]
    bdx = int(round(rig["facing"] * p["bdx"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, 0))
    for name in ("flap_l", "flap_r", "leg_l", "leg_r", "ear_l", "ear_r", "hand_rear", "hand_fore"):
        ang, dx, dy = part_motion(rig, p, name)
        paste(out, moved(L[name], ang, rig["pivots"].get(name, (0, 0)), dx, dy))
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK_F) | is_color(out, INK_B)
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def fore_tip(rig, p):
    fc = rig["facing"]
    ang, dx, dy = part_motion(rig, p, "hand_fore")
    layer = moved(rig["layers"]["hand_fore"], ang, rig["pivots"]["hand_fore"], dx, dy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, PINK, CREAM)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2), (2, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def limb_layers(rig, p):
    parts = {}
    for name in LIMBS:
        ang, dx, dy = part_motion(rig, p, name)
        pivot = rig["pivots"].get(name, (0.0, 0.0))
        parts[name] = moved(rig["layers"][name], ang, pivot, dx, dy)
    return parts


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name, layer in limb_layers(rig, p).items():
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    rig = rigs[d]
    if p["turned"]:
        kind, mirror = VIEW_OF[d]
        rig = rigs[("back" if kind == "front" else "front", not mirror)]
    return rig


def any_motion(p):
    return (
        abs(p["ear"]) >= 0.45 or abs(p["hand"]) >= 0.45 or abs(p["hand_rear"]) >= 0.45
        or p["step_l"] != (0, 0) or p["step_r"] != (0, 0) or p["bdx"] or p["bdy"]
        or p["knock"] or abs(p["lean"]) > 0.4
    )


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], piv)
    if any_motion(p):
        body = reconnect(fill_pinholes(despeckle(body)))
        img = snap(body, palette)
    else:
        img = rig["canvas"].copy()
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = fore_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
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


def tile_grid(tiles, cols, bg=(18, 22, 34, 255)):
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    out = np.full((rows * th, cols * tw, 4), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        m = t[..., 3] >= 128
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
    return out


def crop_box(canvas, pad=8):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad - 24)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 16)
    x0 = max(0, int(xs.min()) - pad - 28)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 28)
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
    tiles = []
    for act in ACTIONS:
        for f in range(16):
            if f < FRAMES[act]:
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                tiles.append(t[y0:y1, x0:x1])
            else:
                tiles.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
    return tile_grid(tiles, 16)


def frame_metrics(img, palette):
    pal = {tuple(int(v) for v in c) for c in palette}
    a = img[..., 3]
    partial = int(((a != 0) & (a != 255)).sum())
    op = a == 255
    off = 0
    if op.any():
        used = {tuple(int(v) for v in c) for c in np.unique(img[op][:, :3], axis=0)}
        off = len(used - pal)
    edge = int(op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any())
    return off, partial, edge


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
    if int(np.nonzero(front[..., 3])[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot changed")
    if int(np.nonzero(back[..., 3])[0].max()) != BACK_FOOT:
        raise SystemExit("back foot changed")
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {
        ("front", False): build_view(front, fm, "front", False),
        ("front", True): build_view(front, fm, "front", True),
        ("back", False): build_view(back, bm, "back", False),
        ("back", True): build_view(back, bm, "back", True),
    }
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "ear_l": (80, 170, 220), "ear_r": (240, 200, 60),
        "flap_l": (220, 80, 80), "flap_r": (80, 200, 120),
        "hand_fore": (240, 140, 40), "hand_rear": (180, 80, 220),
        "leg_l": (80, 220, 220), "leg_r": (240, 160, 190),
        "torso": (180, 180, 190),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m, :3] = cols[name]
            part[m, 3] = 255
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
        "dex": 531, "form": "mega", "variant": "audino-mega", "gen": "gen5",
        "dir": "monsterSprite/gen5/0531_audino/mega/audino-mega",
        "sheetDir": "agent-sprite-forge",
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
    ap.add_argument("--no-publish", action="store_true")
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

    if args.samples:
        off_n = partial_n = edge_n = 0
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    img = synth(rigs, d, act, f, palette)
                    off, partial, edge = frame_metrics(img, palette)
                    off_n += off
                    partial_n += partial
                    edge_n += edge
        print("off_palette", off_n, flush=True)
        print("partial_alpha", partial_n, flush=True)
        print("edge_touch", edge_n, flush=True)
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
            atk.append((f, round(p["ear"], 1), round(p["hand"], 1), round(p["hand_rear"], 1),
                        p["hand_mode"], p["turned"], p["spin"], p["bdx"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        box = crop_box(rigs["S"]["canvas"])
        y0, y1, x0, x1 = box
        for d in ("S", "W", "N", "E"):
            rows = []
            for act in ACTIONS:
                frames = []
                for f in range(16):
                    if f < FRAMES[act]:
                        frames.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
                    else:
                        frames.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
                rows.append(tile_grid(frames, 16))
            sheet = np.concatenate(rows, axis=0)
            z = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
            Image.fromarray(z, "RGBA").save(work / f"samples_{d}.png")
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
    box = crop_box(rigs["S"]["canvas"])
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.concatenate(strips, axis=0)
    motion = np.repeat(np.repeat(motion, 2, 0), 2, 1)
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
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = t[..., 3] == 255
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
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
        "edge_n": len(clip), "sprite_h": sprite_h, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (not any(rest_diff.values()) and not any(idle0.values()) and not off
          and partial == 0 and not clip)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in (
        "rest_diff", "idle0", "off_palette_colors", "partial_alpha", "edge_n", "ok", "sprite_h"
    )}), flush=True)
    meta = {
        "generator": "forge_gen5_audino_mega_actions", "version": VERSION, "key": KEY,
        "dex": 531, "variant": "audino-mega", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped with feeler ears, two pink hands, two feet, no mouth cavity",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (nose x 43.75 < eye x 44.5); back high ear tip is screen-right",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    if args.no_publish:
        print("qa ok; publish skipped", flush=True)
        return
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
