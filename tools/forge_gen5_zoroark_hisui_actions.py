#!/usr/bin/env python3
"""Hisuian Zoroark (0571 regional) 8-direction actions from this form's stills.

Measured on regional/zoroark-hisui/front.png (96x96, bbox x15-81 y5-81,
2633 px, one component, alpha 0/255, sole y81):
  Gold eyes (246, 238, 108), 5 px, mean x 37.2 y 41.2. The muzzle tip is
  the leftmost opaque pixel x15 y31, left of the eyes, so the still faces
  screen-left. The red snout is solid. Ink on the muzzle is the outline,
  not an interior mouth, so the roar does not invent a wedge or a ring.
  It lifts the crest and the hair locks that are already separate.
  Crest: red tuft x56-78 y5-16, behind the head (right of the eyes).
  Rear lock x66-81 y19-27. Side lock x73-81 y57-68.
  Fore foot (image-left, toward the face) x28-46 y60-76, clawed, off the
  ground. Rear foot x60-72 y64-81, the sole (y81, three pixels).
regional/zoroark-hisui/back.png (bbox x16-85 y4-81, 2722 px, one component,
  sole y81, no gold):
  The tall crest is x17-36 y4-15, on the left. That is the trailing crest
  of a right-facing back (the front crest sits on the right). The side
  mane is x63-84. No face and no mouth.
  Rear foot x22-38 y58-77. Fore foot x48-66 y58-81 holds the sole.
  Both soles are y81, so the two views share one ground line.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (screen-right). NW back mirrored.
  Attack 5-8 shows that direction's back at equal scale. W turns into N.
  S turns into NW. No |cos| squash and no in-plane body spin.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_zoroark_hisui_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0571_zoroark/regional/zoroark-hisui"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0571_zoroark/regional/zoroark-hisui/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0571_zoroark-hisui"
VERSION = "zoroark-hisui_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 256, 256
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 24
FRONT_FOOT = 81
BACK_FOOT = 81
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. W keeps that picture. Its back is N.
# S is the mirrored front, so its back is NW. Equal scale, no squash.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Wind-up ceiling. The kick is drawn at 34 and the mane whip at 22, above
# these, so finish() does not push a frame past the angle this script paints.
# Frames 5-8 are cleared after finish() so the back still is not swung.
PEAKS = {"kick": 16.0, "lock": 12.0}
LIMBS = ("foot_rear", "foot_fwd", "lock_a", "lock_b", "crest")

INK = (16, 16, 16)
WHITE = (250, 250, 250)
SILVER = (203, 207, 216)
GRAY = (146, 141, 150)
GRAY_L = (179, 179, 187)
GRAY_D = (74, 77, 83)
RED = (148, 36, 41)
RED_L = (213, 58, 62)
PINK = (240, 115, 118)
MAROON = (95, 0, 2)
DUST = (167, 72, 79)
BLUSH = (247, 172, 174)
WINE = (75, 22, 59)
PLUM = (109, 77, 98)
GOLD = (246, 238, 108)

RED_COLS = (RED, RED_L, PINK, MAROON, DUST, BLUSH)
GRAY_COLS = (GRAY, GRAY_L, GRAY_D, SILVER)
CLAW_COLS = (WINE, PLUM)
LIMB_COLS = GRAY_COLS + CLAW_COLS

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


def component_list(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    groups = []
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
        groups.append(pts)
    return groups


def mask_from_pts(shape, pts):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m


def largest_component(mask):
    groups = component_list(mask)
    if not groups:
        return 0
    return max(len(g) for g in groups)


def largest_mask(mask):
    groups = component_list(mask)
    if not groups:
        return mask
    groups.sort(key=len, reverse=True)
    return mask_from_pts(mask.shape, groups[0])


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


def edge_count(a, b):
    return int((a & dilate(b, 1)).sum())


def outer_ink(part, op, ink):
    """Outline that touches this part and does not also touch the rest."""
    rest = op & ~part & ~ink
    touch_part = ink & dilate(part, 1)
    touch_rest = ink & dilate(rest, 1)
    return touch_part & ~touch_rest


def red_where(red, keep):
    acc = np.zeros_like(red)
    for g in component_list(red):
        m = mask_from_pts(red.shape, g)
        if keep(m):
            acc |= m
    return acc


def bounds_of(mask):
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def segment(img, kind):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    ink = is_color(img, INK)
    red = any_color(img, RED_COLS)
    gray = any_color(img, GRAY_COLS)
    claw = any_color(img, CLAW_COLS)
    eyes = is_color(img, GOLD)
    if kind == "front":
        # Faces left: fore foot is image-left, rear foot holds the sole.
        fwd = (gray & (xx <= 48) & (yy >= 60) & (xx >= 26)) | (claw & (xx <= 48) & (yy >= 68))
        rear = (gray & (xx >= 56) & (yy >= 64)) | (claw & (xx >= 56) & (yy >= 70)) | (op & (yy == FRONT_FOOT))
        crest = red_where(red, lambda m: bounds_of(m)[1] <= 16 and bounds_of(m)[2] >= 48)
        lock_a = red_where(red, lambda m: 17 <= bounds_of(m)[0] and bounds_of(m)[1] <= 32 and bounds_of(m)[2] >= 64)
        # This red tuft touches the rear leg, not the torso, so it stays on that foot.
        thigh = red_where(red, lambda m: bounds_of(m)[0] >= 50 and bounds_of(m)[2] >= 70 and int(m.sum()) >= 20)
        rear = rear | thigh
        lock_b = np.zeros_like(op)
    else:
        # Faces right: fore foot is image-right and holds the sole.
        rear = (gray & (xx <= 40) & (yy >= 58)) | (claw & (xx <= 42) & (yy >= 68))
        fwd = (gray & (xx >= 48) & (yy >= 58) & (xx <= 70)) | (claw & (xx >= 50) & (yy >= 70)) | (op & (yy == BACK_FOOT))
        crest = red_where(red, lambda m: bounds_of(m)[1] <= 16 and bounds_of(m)[3] <= 42)
        lock_a = red_where(
            red, lambda m: bounds_of(m)[0] >= 18 and bounds_of(m)[1] <= 30 and bounds_of(m)[3] <= 32 and int(m.sum()) >= 20,
        )
        lock_b = red_where(red, lambda m: bounds_of(m)[2] >= 62 and bounds_of(m)[0] >= 20 and int(m.sum()) >= 40)
    fwd = largest_mask(fwd & op)
    rear = largest_mask(rear & op)

    def add_claws(foot):
        extra = np.zeros_like(foot)
        for g in component_list(claw & ~foot):
            m = mask_from_pts(foot.shape, g)
            if touches(m, foot) and int(np.nonzero(m)[0].min()) >= 60:
                extra |= m
        return foot | extra

    fwd, rear = add_claws(fwd), add_claws(rear)
    parts = {
        "foot_fwd": fwd, "foot_rear": rear,
        "crest": crest, "lock_a": lock_a, "lock_b": lock_b,
    }
    claimed = np.zeros_like(op)
    for name in ("foot_fwd", "foot_rear", "crest", "lock_a", "lock_b"):
        core = parts[name] & op & ~claimed & ~ink
        ink_add = outer_ink(core, op, ink) & ~claimed
        parts[name] = (core | ink_add) & ~claimed
        claimed |= parts[name]
    torso = op & ~claimed
    groups = component_list(torso)
    if not groups:
        raise SystemExit(f"{kind} torso vanished")
    groups.sort(key=len, reverse=True)
    for g in groups[1:]:
        speck = mask_from_pts(op.shape, g)
        best, best_n = None, 0
        for name, m in parts.items():
            e = edge_count(speck, m)
            if e > best_n:
                best, best_n = name, e
        if best is None:
            raise SystemExit(f"{kind} orphan speck at {g[0]}")
        parts[best] |= speck
        claimed |= speck
    parts["torso"] = op & ~claimed
    check_partition(op, parts)
    torso = parts["torso"]
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit(f"{kind} torso fell apart")
    for name in LIMBS:
        part = parts[name]
        if int(part.sum()) == 0:
            if not (kind == "front" and name == "lock_b"):
                raise SystemExit(f"{kind} {name} is missing")
            continue
        if int(part.sum()) < 40:
            raise SystemExit(f"{kind} {name} is too small ({int(part.sum())})")
        if largest_component(part) != int(part.sum()):
            raise SystemExit(f"{kind} {name} fell apart")
        if not touches(part, torso):
            raise SystemExit(f"{kind} {name} does not touch the body")
    sole = FRONT_FOOT if kind == "front" else BACK_FOOT
    line = op & (yy == sole)
    owner = parts["foot_rear"] if kind == "front" else parts["foot_fwd"]
    if int(line.sum()) < 1 or int((line & owner).sum()) != int(line.sum()):
        raise SystemExit(f"{kind} ground line is not the planted foot")
    if kind == "front":
        if int(eyes.sum()) != 5 or int((eyes & torso).sum()) != 5:
            raise SystemExit("eyes left the head")
        ex = float(np.nonzero(eyes)[1].mean())
        tip = int(np.nonzero(op)[1].min())
        if tip != 15 or not bool(torso[:, tip].any()) or not (tip < ex - 10):
            raise SystemExit(f"front does not face left: tip {tip} eye {ex}")
        crest_x = float(np.nonzero(parts["crest"])[1].mean())
        if not (crest_x > ex):
            raise SystemExit("front crest is not behind the face")
    else:
        if eyes.any():
            raise SystemExit("back grew eyes")
        cx = float(np.nonzero(torso)[1].mean())
        crest_x = float(np.nonzero(parts["crest"])[1].mean())
        lock_x = float(np.nonzero(parts["lock_b"])[1].mean())
        if not (crest_x < cx < lock_x):
            raise SystemExit(f"back is not facing right: crest {crest_x:.1f} body {cx:.1f} mane {lock_x:.1f}")
    return parts


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


def ground_pivot(layer, facing):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    yb = int(ys.max())
    band = ys >= yb - 1
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def shift(layer, dx, dy):
    if dx == 0 and dy == 0:
        return layer
    out = np.zeros_like(layer)
    h, w = layer.shape[:2]
    x0, x1 = max(0, dx), min(w, w + dx)
    y0, y1 = max(0, dy), min(h, h + dy)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    out[y0:y1, x0:x1] = layer[sy0:sy1, sx0:sx1]
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
        up = np.zeros_like(src)
        dn = np.zeros_like(src)
        lt = np.zeros_like(src)
        rt = np.zeros_like(src)
        up[:-1] = src[1:]
        dn[1:] = src[:-1]
        lt[:, :-1] = src[:, 1:]
        rt[:, 1:] = src[:, :-1]
        for g in (up, dn, lt, rt):
            take = (acc[..., 3] < 128) & allowed & (g[..., 3] >= 128)
            acc[take] = g[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


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
        ("foot_rear", 8), ("foot_fwd", 8),
        ("lock_a", 6), ("lock_b", 8), ("crest", 6),
    ):
        if int(placed[name].sum()) == 0:
            continue
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
        if int((layers[name][..., 3] >= 128).sum()) == 0:
            pivots[name] = (0.0, 0.0)
            signs[name] = {"up": 1.0, "fwd": 1.0}
            continue
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    pivots["rear_foot"] = ground_pivot(layers["foot_rear"], facing)
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
        kick=0.0, brace=0.0,
        crest=0.0, lock=0.0,
        lock_mode="up", foot_mode="fwd",
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay on the drawing. The crest and the hair locks sway.
        p["crest"] = wave(f, n, 6.0, phase=0.4)
        p["lock"] = wave(f, n, 8.0, phase=1.1)
        p["lock_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        # One clawed foot swings forward while the other stays planted.
        p["kick"] = 20.0 * max(0.0, math.sin(u))
        p["brace"] = 20.0 * max(0.0, math.sin(u + math.pi))
        p["foot_mode"] = "fwd"
        p["crest"] = wave(f, n, 5.0, cycles=2.0, phase=0.3)
        p["lock"] = wave(f, n, 8.0, cycles=2.0, phase=0.8)
        p["lock_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 raise the crest and the locks. No mouth.
        # 5-8 the back, equal scale, cleared after finish().
        # 9 wind-up. 10-12 fore-foot kick and the side lock. 13-15 return.
        crest = [0, 6, 10, 14, 16, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0]
        lock = [0, 6, 10, 12, 14, 0, 0, 0, 0, -12, 16, 22, 22, 8, 2, 0]
        kick = [0, 0, 0, 0, 0, 0, 0, 0, 0, -16, 24, 34, 34, 10, 3, 0]
        p["crest"] = float(crest[f])
        p["lock"] = float(lock[f])
        p["kick"] = float(kick[f])
        p["lock_mode"] = "up" if f <= 4 else "fwd"
        p["foot_mode"] = "fwd"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.64, 0.44, 0.26, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["crest"] = 6.0 * env
        p["lock"] = 10.0 * env
        p["lock_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["kick"] = 0.0
        p["brace"] = 0.0
        p["crest"] = 0.0
        p["lock"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["spark"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["flash"] = False
        p["turned"] = True
        p["lock_mode"] = "up"
        p["foot_mode"] = "fwd"
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


def limb_delta(rig, p, name):
    bdx = int(round(rig["facing"] * p["bdx"]))
    bdy = int(round(p["bdy"]))
    if name == "foot_fwd":
        return part_angle(rig, name, p["foot_mode"], p["kick"]), bdx, bdy
    if name == "foot_rear":
        return part_angle(rig, name, "fwd", p["brace"]), bdx, bdy
    if name == "crest":
        return part_angle(rig, name, "up", p["crest"]), bdx, bdy
    if name in ("lock_a", "lock_b"):
        return part_angle(rig, name, p["lock_mode"], p["lock"]), bdx, bdy
    return 0.0, bdx, bdy


def compose(rig, p):
    L = rig["layers"]
    piv = rig["pivots"]
    bdx = int(round(rig["facing"] * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    for name in LIMBS:
        if int((L[name][..., 3] >= 128).sum()) == 0:
            continue
        ang, dx, dy = limb_delta(rig, p, name)
        paste(out, moved(L[name], ang, piv[name], dx, dy))
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK)
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def lean_angle(facing, deg):
    """Head moves opposite facing. Positive rotation is clockwise on screen."""
    return facing * deg


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, PINK, BLUSH, RED_L)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2), (2, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def strike_tip(rig, p):
    fc = rig["facing"]
    best = None
    for name in ("foot_fwd", "lock_b"):
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
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    for key in ("kick", "brace", "crest", "lock"):
        if abs(p[key]) >= 0.45:
            return True
    if abs(p["lean"]) > 0.4:
        return True
    return False


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    for key in ("kick", "brace", "crest", "lock"):
        if abs(p[key]) >= 0.45:
            return True
    return False


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    fc = rig["facing"]
    knock_dx = int(round(-fc * p["knock"]))
    if knock_dx:
        body = shift(body, knock_dx, 0)
    if abs(p["lean"]) > 0.4:
        px = rig["pivots"]["rear_foot"][0] + knock_dx
        py = rig["pivots"]["rear_foot"][1]
        body = rotate_nearest(body, lean_angle(fc, p["lean"]), (px, py))
    if rotated(p):
        body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    elif moving(p):
        img = snap(body, palette)
    else:
        img = rig["canvas"].copy()
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = strike_tip(rig, p)
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
        if int((rig["layers"][name][..., 3] >= 128).sum()) == 0:
            continue
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


def tile_grid(tiles, cols, bg=(15, 20, 31, 255)):
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
        raise SystemExit("this still has no mouth cavity")
    fa = rigs[("front", False)]["canvas"]
    ba = rigs[("back", False)]["canvas"]
    if int(np.nonzero(fa[..., 3] >= 128)[0].max()) != int(np.nonzero(ba[..., 3] >= 128)[0].max()):
        raise SystemExit("front and back feet do not share a line")
    for key in (("front", False), ("back", False)):
        rig = rigs[key]
        shift_x = head_shift(rig, MAX_LEAN)
        fc = rig["facing"]
        if fc > 0 and not (shift_x < -5):
            raise SystemExit(f"hurt lean does not tip the head back ({shift_x})")
        if fc < 0 and not (shift_x > 5):
            raise SystemExit(f"hurt lean does not tip the head back ({shift_x})")
    return front, back, fm, bm, rigs


def head_shift(rig, deg):
    img = rig["canvas"]
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    i = int(ys.argmin())
    piv = rig["pivots"]["rear_foot"]
    rx = float(xs[i]) - piv[0]
    ry = float(ys[i]) - piv[1]
    ang = lean_angle(rig["facing"], deg)
    r = math.radians(ang)
    c, s = math.cos(r), math.sin(r)
    return (piv[0] + c * rx + s * ry) - float(xs[i])


def save_parts(work, front, fm, back, bm):
    cols = {
        "crest": (255, 220, 40, 255),
        "lock_a": (255, 80, 40, 255),
        "lock_b": (220, 40, 80, 255),
        "foot_fwd": (40, 200, 90, 255),
        "foot_rear": (40, 160, 220, 255),
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
        "dex": 571, "form": "regional", "variant": "zoroark-hisui", "gen": "gen5",
        "dir": "monsterSprite/gen5/0571_zoroark/regional/zoroark-hisui",
        "sheetDir": "agent-sprite-forge",
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


def check_pose_limits():
    for f in range(5, 9):
        p = pose("attack", f)
        if not p["turned"] or any(abs(float(p[k])) > 0.01 for k in ("kick", "brace", "crest", "lock", "lean", "spin")):
            raise SystemExit(f"turn frame {f} is not the still back")
        if p["knock"] or p["bdx"] or p["bdy"] or p["spark"]:
            raise SystemExit(f"turn frame {f} still moves")
    wind = pose("attack", 9)
    if not (8.0 <= abs(float(wind["kick"])) <= 22.0):
        raise SystemExit(f"windup left 8-22 ({wind['kick']})")
    for f in (10, 11, 12):
        kick = abs(float(pose("attack", f)["kick"]))
        if kick > 34.01:
            raise SystemExit(f"strike frame {f} is {kick}")
    leans = [float(pose("hurt", f)["lean"]) for f in range(12)]
    if max(leans) > 38.01:
        raise SystemExit(f"lean over 38 ({max(leans)})")
    if abs(float(pose("attack", 0)["kick"])) > 0.01 or abs(float(pose("attack", 15)["kick"])) > 0.01:
        raise SystemExit("attack does not start and end on the still")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()

    save_peaks(Path(__file__).stem, PEAKS)
    check_pose_limits()
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    palette = locked_palette(SRC_DIR / "front.png", SRC_DIR / "back.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    show = lambda masks: {k: int(v.sum()) for k, v in masks.items()}
    print("parts front", show(fm), "back", show(bm),
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    print("mouth", {d: rigs[d]["has_mouth"] for d in DIRS}, flush=True)

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
            atk.append((f, p["turned"], round(float(p["kick"]), 1), round(float(p["lock"]), 1),
                        round(float(p["crest"]), 1), p["lock_mode"], p["spark"], p["spin"]))
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
    print("qa_full", "rest_diff", int(sum(rest_diff.values()) + sum(idle0.values())),
          "off_palette", len(off), "partial_alpha", partial, "edge_touch", len(clip),
          "body_main_min", report["body_main_min"], "clip", clip[:8], flush=True)
    meta = {
        "generator": "forge_gen5_zoroark_hisui_actions", "version": VERSION, "key": KEY,
        "dex": 571, "variant": "zoroark-hisui", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped spectral fox (gold eyes, white mane, red hair locks, two clawed feet, no mouth cavity)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (muzzle tip x15 < eye mean x37.2); back faces screen-right",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    if not args.no_publish:
        publish(bounds, sprite_h)
        print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
