#!/usr/bin/env python3
"""Petilil normal (0548) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x35-61 y25-69, foot y69, alpha 0/255):
  Crown tip x43-44 (mean 43.5). Pupils (123,49,57) at x42 and x49, y55-56
  (mean x 45.5). The leaf tip is left of the eyes, so the still faces
  screen-left. Pink (189,139,148) sits on those pupils, not under the face:
  there is no mouth, so the roar does not invent a wedge or a ring.
  Three crown leaves (bright / mid / dark green plus their ink, y<=41),
  split from the y<=32 gaps. Two leaf feet: the ground nub at y69 is the
  left foot, the right nub ends at y68. The pale dress and eyes stay the torso.
normal/back.png (bbox x34-60 y25-69, foot y69) has no eyes. Its crown tip
  is x50-51, right of the body, so the back faces screen-right.

Directions (left-facing still): S SE E are the front mirrored, W SW the front
as drawn, NE the back mirrored, N NW the back as drawn. E is screen-right,
W is screen-left, N is the back. Attack 5-8 shows that back at equal scale.
No |cos| squash, no laid-over body.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen5_petilil_actions.py [--samples] [--force] [--publish]
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen5_attack_strike import finish, save_peaks

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen5/0548_petilil/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0548_petilil/normal/agent-sprite-forge"
)

KEY = "0548_petilil"
VERSION = "petilil_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 176, 176
FRONT_FOOT = 69
BACK_FOOT = 69
OFF_X = (CELL_W - 96) // 2
# The still already has 26 clear rows under the sole. That is the pad below the foot.
OFF_Y = CELL_H - 96
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn faces screen-left. Mirror groups follow that.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", True),
    "N": ("back", False), "NW": ("back", False),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Roar ceilings. The leaf slap is drawn higher (34 / 28 / 26), so finish()
# does not shrink the hit and does not push it past those drawn angles.
PEAKS = {"leaf_l": 18.0, "leaf_c": 22.0, "leaf_r": 16.0}
LIMBS = ("leaf_l", "leaf_c", "leaf_r", "foot_l", "foot_r")

INK = (16, 16, 16)
PALE = (197, 238, 123)
BRIGHT = (57, 172, 57)
DGREEN = (49, 90, 49)
WHITE = (255, 255, 255)
MGREEN = (57, 131, 74)
OLIVE = (156, 189, 74)
OLIVE2 = (98, 123, 49)
PALE2 = (205, 205, 189)
MAROON = (123, 49, 57)
PINK = (189, 139, 148)
BACK_HI = (180, 213, 98)
LEAF_COLS = (BRIGHT, DGREEN, MGREEN)
EYE_MARK = (MAROON, PINK)

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


def _rotate_full(layer, ang, pivot):
    """4x nearest of the whole cell. Reference for the windowed rotate."""
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


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest in-plane rotation. Positive is clockwise on screen. No squash."""
    ang = float(angle_deg) % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    px, py = float(pivot[0]), float(pivot[1])
    rad = math.sqrt(((xs - px) ** 2 + (ys - py) ** 2).max())
    pad = int(math.ceil(rad)) + 4
    h, w = layer.shape[:2]
    x0 = max(0, int(math.floor(px)) - pad)
    y0 = max(0, int(math.floor(py)) - pad)
    x1 = min(w, int(math.ceil(px)) + pad + 1)
    y1 = min(h, int(math.ceil(py)) + pad + 1)
    crop = layer[y0:y1, x0:x1]
    sc = 4
    up = np.repeat(np.repeat(crop, sc, axis=0), sc, axis=1)
    uh, uw = up.shape[:2]
    ppx = (px - x0) * sc + 1.5
    ppy = (py - y0) * sc + 1.5
    yy, xx = np.mgrid[0:uh, 0:uw]
    rx, ry = xx - ppx, yy - ppy
    r = math.radians(ang)
    c, s = math.cos(r), math.sin(r)
    sx = np.rint(ppx + c * rx + s * ry).astype(np.int32)
    sy = np.rint(ppy - s * rx + c * ry).astype(np.int32)
    ok = (sx >= 0) & (sx < uw) & (sy >= 0) & (sy < uh)
    out = np.zeros_like(up)
    out[ok] = up[sy[ok], sx[ok]]
    small = out[sc // 2::sc, sc // 2::sc]
    dest = np.zeros_like(layer)
    sh, sw = small.shape[:2]
    dest[y0:y0 + sh, x0:x0 + sw] = small
    return dest


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
    dist = ((px[:, None, :] - palette.astype(np.int32)[None, :, :]) ** 2).sum(axis=2)
    out[op, :3] = palette[dist.argmin(axis=1)]
    out[op, 3] = 255
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def put(img, x, y, col, over=False):
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 1 <= x < w - 1 and 1 <= y < h - 1 and (over or img[y, x, 3] < 128):
        img[y, x, :3] = col
        img[y, x, 3] = 255


def paint_cushion(layer, allowed, steps):
    """Grow the torso's own colours into the hidden joint band."""
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        for d in ("up", "down", "left", "right"):
            mv = np.zeros_like(src)
            if d == "up":
                mv[:-1] = src[1:]
            elif d == "down":
                mv[1:] = src[:-1]
            elif d == "left":
                mv[:, :-1] = src[:, 1:]
            else:
                mv[:, 1:] = src[:, :-1]
            take = (acc[..., 3] < 128) & allowed & (mv[..., 3] >= 128)
            acc[take] = mv[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    sizes = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        y0 = int(y0)
        x0 = int(x0)
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
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        sizes.append(n)
    return sorted(sizes, reverse=True)


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    sizes = components(op)
    return sizes[0] / float(op.sum())


def split_three(crown, yy):
    """Three leaves. Seeds are the separate tips at y<=32; the base is shared."""
    h, w = crown.shape
    upper = crown & (yy <= 32)
    seen = np.zeros_like(upper, bool)
    comps = []
    for y, x in zip(*np.nonzero(upper)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        stack = [(y, x)]
        seen[y, x] = True
        pts = []
        while stack:
            cy, cx = stack.pop()
            pts.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and upper[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        xs = np.array([p[1] for p in pts], np.float64)
        comps.append((float(xs.mean()), pts))
    comps.sort(key=lambda t: t[0])
    if len(comps) != 3:
        raise SystemExit(f"crown tips {len(comps)} != 3")
    labels = np.zeros((h, w), np.int16)
    qs = [deque() for _ in comps]
    for i, (_, pts) in enumerate(comps):
        for cy, cx in pts:
            labels[cy, cx] = i + 1
            qs[i].append((cy, cx))
    while any(qs):
        for i, q in enumerate(qs):
            if not q:
                continue
            cy, cx = q.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and crown[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = i + 1
                    q.append((ny, nx))
    if int((crown & (labels == 0)).sum()):
        raise SystemExit("crown pixel left unlabeled")
    return labels


def segment(img, kind):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    op = img[..., 3] >= 128
    leaf = any_color(img, LEAF_COLS)
    ink = is_color(img, INK)
    crown_g = flood(leaf, leaf & (yy <= 32))
    crown_ink = flood(ink & (yy <= 41), dilate(crown_g, 1) & ink & (yy <= 41))
    crown = crown_g | crown_ink
    labels = split_three(crown, yy)
    leaves = {
        "leaf_l": labels == 1,
        "leaf_c": labels == 2,
        "leaf_r": labels == 3,
    }
    olive = any_color(img, (OLIVE, OLIVE2, BACK_HI))
    foot_allowed = (ink | olive) & (yy >= 66)
    foot_l = flood(foot_allowed & (xx <= 48), op & (yy >= 68) & (xx <= 48) & foot_allowed)
    foot_r = flood(foot_allowed & (xx >= 49), op & (yy >= 67) & (xx >= 50) & foot_allowed)
    foot_r &= ~foot_l
    used = foot_l | foot_r
    for m in leaves.values():
        used |= m
    torso = op & ~used
    masks = dict(leaves)
    masks["foot_l"] = foot_l
    masks["foot_r"] = foot_r
    masks["torso"] = torso
    cover = np.zeros_like(op)
    for m in masks.values():
        if (cover & m).any():
            raise SystemExit(f"{kind} masks overlap")
        cover |= m
    if int((op & ~cover).sum()) or int(cover.sum()) != int(op.sum()):
        raise SystemExit(f"{kind} partition {int(cover.sum())} != {int(op.sum())}")
    for name in LIMBS:
        if int(masks[name].sum()) < 4:
            raise SystemExit(f"{kind} {name} is too small")
        if not (masks[name] & dilate(torso, 1)).any():
            raise SystemExit(f"{kind} {name} does not touch the torso")
    if kind == "back" and (any_color(img, (WHITE, PALE2, MAROON, PINK)).any()):
        raise SystemExit("back grew a face")
    if kind == "front":
        marks = any_color(img, EYE_MARK)
        eyes = is_color(img, WHITE)
        if marks.any() and (marks & ~dilate(eyes, 1)).any():
            raise SystemExit("a mark sits outside the eyes; do not treat it as a mouth")
    return masks


def place(arr, y_shift=0):
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
    for name, depth in (
        ("leaf_l", 5), ("leaf_c", 5), ("leaf_r", 5),
        ("foot_l", 3), ("foot_r", 3),
    ):
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], placed["torso"], depth), depth,
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
        strike = "leaf_l"
    else:
        facing = -1.0 if mirror else 1.0
        strike = "leaf_r"
    pivots = {}
    signs = {}
    for name in ("leaf_l", "leaf_c", "leaf_r"):
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "strike": strike,
        "has_mouth": False, "layers": layers, "pivots": pivots, "signs": signs,
        "canvas": canvas,
    }


def base_pose():
    return dict(
        leaf_l=0.0, leaf_c=0.0, leaf_r=0.0, leaf_mode="up",
        step_l=(0, 0), step_r=(0, 0),
        bdx=0, bdy=0,
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The three leaves breathe in a breeze. The face does not slide.
        p["leaf_c"] = wave(f, n, 5.0, phase=0.4)
        p["leaf_l"] = wave(f, n, 4.0, phase=1.1)
        p["leaf_r"] = wave(f, n, 4.0, phase=1.1 + math.pi)
        p["leaf_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * sa)), -int(round(2 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * sb)), -int(round(2 * max(0.0, sb))))
        p["leaf_c"] = wave(f, n, 6.0, cycles=2.0)
        p["leaf_l"] = wave(f, n, 5.0, phase=0.5)
        p["leaf_r"] = wave(f, n, 5.0, phase=0.5 + math.pi)
        p["leaf_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 raise the leaves (no mouth). 5-8 the back, equal scale.
        # 9-12 the leaves slap forward. 13-15 return. Feet stay planted.
        leaf_c = [0, 6, 12, 18, 22, 0, 0, 0, 0, 18, 28, 34, 34, 12, 4, 0]
        leaf_l = [0, 4, 8, 14, 18, 0, 0, 0, 0, 14, 22, 28, 28, 8, 3, 0]
        leaf_r = [0, 4, 8, 12, 16, 0, 0, 0, 0, 12, 20, 26, 26, 8, 3, 0]
        p["leaf_c"] = float(leaf_c[f])
        p["leaf_l"] = float(leaf_l[f])
        p["leaf_r"] = float(leaf_r[f])
        p["leaf_mode"] = "up" if f <= 4 else "fwd"
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["leaf_c"] = 14.0 * env
        p["leaf_l"] = 10.0 * env
        p["leaf_r"] = 10.0 * env
        p["leaf_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["leaf_l"] = 0.0
        p["leaf_c"] = 0.0
        p["leaf_r"] = 0.0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["bdx"] = 0
        p["bdy"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["spark"] = 0
        p["flash"] = False
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


def compose(rig, p):
    L = rig["layers"]
    piv = rig["pivots"]
    fc = rig["facing"]
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], int(round(fc * p["bdx"])), int(p["bdy"])))
    for name, key in (("foot_l", "step_l"), ("foot_r", "step_r")):
        sx, sy = p[key]
        paste(out, shift(L[name], int(round(fc * sx)), int(sy)))
    order = ("leaf_r", "leaf_c", "leaf_l") if fc < 0 else ("leaf_l", "leaf_c", "leaf_r")
    for name in order:
        ang = part_angle(rig, name, p["leaf_mode"], p[name])
        paste(out, moved(L[name], ang, piv[name], int(round(fc * p["bdx"])), int(p["bdy"])))
    return out


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", mirror)]


def weapon_tip(rig, p):
    name = rig["strike"]
    fc = rig["facing"]
    ang = part_angle(rig, name, p["leaf_mode"], p[name])
    layer = moved(rig["layers"][name], ang, rig["pivots"][name], int(round(fc * p["bdx"])), int(p["bdy"]))
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def spark(img, origin, fc, age):
    """A few impact pixels in front of the leaf tip. Not a mouth."""
    ox, oy = origin
    cols = (PALE, BRIGHT, WHITE)
    r = 3 + int(age)
    for i, (dx, dy) in enumerate(((r, -1), (r, 1), (r + 1, 0), (2, -2), (2, 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


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


def limb_lost(rig, p):
    body = compose(rig, p)
    fc = rig["facing"]
    lost = []
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    for name in LIMBS:
        if name.startswith("foot"):
            sx, sy = p["step_" + name[-1]]
            layer = shift(rig["layers"][name], bdx + int(round(fc * sx)), bdy + int(sy))
        else:
            ang = part_angle(rig, name, p["leaf_mode"], p[name])
            layer = moved(rig["layers"][name], ang, rig["pivots"][name], bdx, bdy)
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
    body = compose(rig, p)
    rotated = (
        abs(p["leaf_l"]) >= 0.45 or abs(p["leaf_c"]) >= 0.45 or abs(p["leaf_r"]) >= 0.45
        or abs(p["lean"]) > 0.4
    )
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
    if p["spark"]:
        tip = weapon_tip(rig, p)
        spark(img, tip, fc, p["spark"])
    return snap(img, palette)


def measure_facing(front, back):
    op = front[..., 3] >= 128
    ys, xs = np.nonzero(op)
    top = int(ys.min())
    tip = float(xs[ys == top].mean())
    pupils = is_color(front, MAROON)
    eye = float(np.nonzero(pupils)[1].mean())
    bop = back[..., 3] >= 128
    bys, bxs = np.nonzero(bop)
    btop = int(bys.min())
    btip = float(bxs[bys == btop].mean())
    bcom = float(bxs.mean())
    return tip, eye, btip, bcom


def palette_of(front, back):
    cols = []
    seen = set()
    for img in (front, back):
        op = img[..., 3] >= 128
        for c in np.unique(img[op][:, :3], axis=0):
            t = tuple(int(v) for v in c)
            if t not in seen:
                seen.add(t)
                cols.append(t)
    return np.array(cols, np.uint8)


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
    pick = {"idle": 0, "walk": 4, "attack": 11, "hurt": 3}
    tiles = []
    for act in ACTIONS:
        for di in range(8):
            f = pick[act]
            tiles.append(sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W])
    return Image.fromarray(tile_grid(tiles, 8), "RGBA")


def motion_contact(sheets, di=0):
    pad = 4
    rows = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            row.append(sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W])
        rows.append(row)
    allop = np.zeros((CELL_H, CELL_W), bool)
    for row in rows:
        for t in row:
            allop |= t[..., 3] >= 128
    ys, xs = np.nonzero(allop)
    y0, y1 = max(0, int(ys.min()) - pad), min(CELL_H, int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(CELL_W, int(xs.max()) + 1 + pad)
    tiles = []
    for row in rows:
        for f in range(16):
            if f < len(row):
                tiles.append(row[f][y0:y1, x0:x1])
            else:
                tiles.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
    return Image.fromarray(tile_grid(tiles, 16), "RGBA")


def frame_metrics(img, pal):
    partial = int(((img[..., 3] != 0) & (img[..., 3] != 255)).sum())
    op = img[..., 3] == 255
    off = 0
    if op.any():
        used = {tuple(int(v) for v in c) for c in np.unique(img[op][:, :3], axis=0)}
        off = len(used - pal)
    edge = 1 if (op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any()) else 0
    return off, partial, edge


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
        "dex": 548, "form": "normal", "variant": "petilil", "gen": "gen5",
        "dir": "monsterSprite/gen5/0548_petilil/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def params_hash():
    src = Path(__file__).read_bytes() + (SRC_DIR / "front.png").read_bytes() + (SRC_DIR / "back.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def load_still(path):
    img = np.array(Image.open(path).convert("RGBA"))
    img[img[..., 3] < 128] = 0
    img[img[..., 3] >= 128, 3] = 255
    return img


def build_rigs():
    front = load_still(SRC_DIR / "front.png")
    back = load_still(SRC_DIR / "back.png")
    if front.shape[:2] != (96, 96) or back.shape[:2] != (96, 96):
        raise SystemExit("expected 96x96 stills")
    fys = int(np.nonzero(front[..., 3])[0].max())
    bys = int(np.nonzero(back[..., 3])[0].max())
    if fys != FRONT_FOOT or bys != BACK_FOOT:
        raise SystemExit(f"foot line changed front {fys} back {bys}")
    tip, eye, btip, bcom = measure_facing(front, back)
    print(
        "facing_measure",
        "front_tip", round(tip, 2), "pupil_x", round(eye, 2),
        "back_tip", round(btip, 2), "back_com", round(bcom, 2),
        flush=True,
    )
    if not (tip < eye):
        raise SystemExit("front does not face left")
    if not (btip > bcom):
        raise SystemExit("back does not face right")
    fm = segment(front, "front")
    bm = segment(back, "back")
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "leaf_l": (80, 140, 210, 255), "leaf_c": (240, 200, 40, 255),
        "leaf_r": (220, 40, 40, 255), "foot_l": (40, 200, 80, 255),
        "foot_r": (240, 120, 40, 255), "torso": (200, 80, 220, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 3, 0), 3, 1))
    Image.fromarray(np.concatenate(tiles, axis=1), "RGBA").save(work / "parts.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    save_peaks(Path(__file__).stem, PEAKS)
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    front, back, fm, bm, rigs = build_rigs()
    palette = palette_of(front, back)
    pal = {tuple(int(v) for v in c) for c in palette}
    save_parts(work, front, fm, back, bm)
    print(
        "parts front", {k: int(v.sum()) for k, v in fm.items()},
        "back", {k: int(v.sum()) for k, v in bm.items()},
        "palette", len(palette), "foot", FOOT_Y, flush=True,
    )
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    print("mouth", False, "strike", {d: rigs[d]["strike"] for d in ("S", "W", "N", "E")}, flush=True)

    layer = rigs["W"]["layers"]["leaf_c"]
    piv = rigs["W"]["pivots"]["leaf_c"]
    if np.any(rotate_nearest(layer, 17.0, piv) != _rotate_full(layer, 17.0, piv)):
        raise SystemExit("windowed rotate does not match the full 4x rotate")
    print("rotate_match", 1, flush=True)

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

    for f in (5, 6, 7, 8):
        turn_w = synth(rigs, "W", "attack", f, palette, effects=False)
        turn_s = synth(rigs, "S", "attack", f, palette, effects=False)
        turn_n = synth(rigs, "N", "attack", f, palette, effects=False)
        if np.any(turn_w != rigs["N"]["canvas"]):
            raise SystemExit(f"W attack {f} is not the N back at equal scale")
        if np.any(turn_s != rigs["NE"]["canvas"]):
            raise SystemExit(f"S attack {f} is not the N back flipped at equal scale")
        if np.any(turn_n != rigs["N"]["canvas"]):
            raise SystemExit(f"N attack {f} left the back")
        if np.any(rigs["NE"]["canvas"] != rigs["N"]["canvas"][:, ::-1]):
            raise SystemExit("NE is not N flipped")
    print("turn_equal", 1, flush=True)

    if args.samples:
        lost = {}
        off = partial = edge = 0
        body_min = 1.0
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    img = synth(rigs, d, act, f, palette, effects=True)
                    o, p, e = frame_metrics(img, pal)
                    off += o
                    partial += p
                    edge += e
                    body_min = min(body_min, largest_fraction(img))
                    if act in ("attack", "hurt", "walk") and f in (2, 4, 6, 8, 11, 3, 12):
                        rig = rig_for(rigs, d, pose(act, f))
                        bad = limb_lost(rig, pose(act, f))
                        if bad:
                            lost[f"{act}:{d}:{f}"] = bad
        print("limb lost", lost or "none", flush=True)
        print(
            f"MEASURE rest_diff_sum={int(sum(rest_diff.values()))} off_palette={off} "
            f"partial_alpha={partial} edge_touch={edge} body_min={body_min:.3f}",
            flush=True,
        )
        tiles = []
        for d in ("S", "W", "N", "E"):
            for act, f in (
                ("idle", 0), ("idle", 8),
                ("walk", 4), ("walk", 12),
                ("attack", 0), ("attack", 3), ("attack", 6), ("attack", 8),
                ("attack", 11), ("attack", 14),
                ("hurt", 1), ("hurt", 3),
            ):
                tiles.append(synth(rigs, d, act, f, palette))
        ys, xs = np.nonzero(rigs["S"]["canvas"][..., 3] >= 128)
        pad = 18
        y0, y1 = max(0, int(ys.min()) - pad), min(CELL_H, int(ys.max()) + pad)
        x0, x1 = max(0, int(xs.min()) - pad), min(CELL_W, int(xs.max()) + pad)
        crops = [t[y0:y1, x0:x1] for t in tiles]
        sheet = np.repeat(np.repeat(tile_grid(crops, 12), 2, 0), 2, 1)
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
                if fp.exists() and fp.stat().st_size > 0:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview_image(sheets).save(OUT_DIR / "preview.png")
    motion_contact(sheets, 0).save(OUT_DIR / "preview_motion.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", OUT_DIR / "actions" / "preview.png")

    op = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[op][:, :3], axis=0)}
    off_cols = sorted(used - pal)
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
        "off_palette_colors": len(off_cols), "partial_alpha": partial,
        "edge_touching": clip[:12], "edge_n": len(clip),
        "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (
        not any(rest_diff.values()) and not any(idle0.values()) and not off_cols
        and partial == 0 and not clip and min(bodies) >= 0.90
    )
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(
        f"MEASURE rest_diff_sum={int(sum(rest_diff.values()))} off_palette={len(off_cols)} "
        f"partial_alpha={partial} edge_touch={len(clip)} body_min={report['body_main_min']}",
        flush=True,
    )
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_petilil_actions", "version": VERSION, "key": KEY,
        "dex": 548, "variant": "petilil", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off_cols),
        "archetype": "plant (three crown leaves, two leaf feet, no mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced files", sync_outputs(), flush=True)
    if args.publish:
        publish(bounds, sprite_h)
        print("published", KEY, VERSION, "h", sprite_h, flush=True)


if __name__ == "__main__":
    main()
