#!/usr/bin/env python3
"""Sigilyph normal (0561) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x8-87 y2-92, 2294 px, 13 colours,
alpha 0/255, lowest pixel y92 x63):
  White eye mean x 50.07 (y48.7). Green iris mean x 52.64, right of the white.
  The crest tip is the topmost pixels x84-87 y2, right of the eye.
  The still faces screen-right. No mouth, fangs, hands, horns, or feet are drawn.
  Left wing (screen-left) is the lower yellow panel plus its cyan/red eye,
  x8-44 y10-44. Right wing is the tall crest, x55-87 y2-42, with its own eye.
  The tail is the hanging totem x43-65 y66-92, yellow with a bottom eye.
  The central eye stays on the torso. The roar does not invent a mouth.

normal/back.png (bbox x8-88 y3-91, 2350 px, lowest y91, yellow is (230,172,0)):
  Crest tip x8-10 y3 is left of the white eye (mean x 41.0). Away-left as drawn.
  The tail is two lobes split by the body (x18-32 and the column beside it).

Directions, front faces right, back faces away-left:
  S SE E front as drawn (screen-right). SW W front mirrored (screen-left).
  N NW back as drawn (away-left). NE back mirrored (away-right).
  E is screen-right, W is screen-left, N is the back.
  Attack 5-8 swaps kind and keeps the mirror, so S becomes the N still
  at equal scale. No width squash and no in-plane spin.

  idle 16: hover. Both wings and the tail breathe. Frame 0 is the still.
  walk 16: glide. Wings flap, tail trails, a 1px bob. No new feet.
  attack: 0 still, 1-4 wings and tail rise (no mouth wedge), 5-8 the other
          side, 9-12 both wings and the tail strike, 11-12 impact, 13-15 recover.
  hurt 12: white flash, knock-back lean (<=38 deg) and recover.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_sigilyph_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0561_sigilyph/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0561_sigilyph/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0561_sigilyph"
VERSION = "sigilyph_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 256, 256
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 32
FRONT_FOOT = 92
BACK_FOOT = 91
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-right. Back as drawn looks away to the left.
VIEW_OF = {
    "S": ("front", False), "SE": ("front", False), "E": ("front", False),
    "SW": ("front", True), "W": ("front", True),
    "NE": ("back", True), "N": ("back", False), "NW": ("back", False),
}
MAX_LEAN = 38.0
# Angles this script already draws. finish() will not raise a frame past them.
PEAKS = {"wing": 34.0, "tail": 28.0}

INK = (16, 16, 16)
DK = (49, 49, 49)
YEL_F = (255, 197, 8)
YEL_B = (230, 172, 0)
CYA = (74, 172, 230)
WHT = (255, 255, 255)
GLD = (156, 123, 24)
RED = (222, 0, 0)
TEA = (41, 123, 139)
GRN = (57, 164, 74)
GRD = (49, 115, 57)
GRY = (148, 148, 148)
BRN = (98, 65, 16)
PNK = (148, 82, 115)
PATTERN = (CYA, RED, TEA, PNK)
EYE = (WHT, GRN, GRD, GRY)
OUTLINE_KEEP = (INK, DK)

SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)
PART_ORDER = ("tail", "tail_b", "wing_l", "wing_r")


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


def dilate8(mask):
    p = np.pad(mask, 1, constant_values=False)
    h, w = mask.shape
    out = mask.copy()
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + h, dx:dx + w]
    return out


def component_list(mask, conn=4):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    groups = []
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1)) if conn == 4 else tuple(
        (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)
    )
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
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        groups.append(pts)
    groups.sort(key=len, reverse=True)
    return groups


def mask_of(pts, shape):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m


def bounds_of(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()), float(xs.mean()), float(ys.mean())


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
    """4x nearest in-plane rotation (positive = clockwise on screen). No squash."""
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
    d = ((px[:, None, :] - palette.astype(np.int32)[None, :, :]) ** 2).sum(axis=2)
    out[op, :3] = palette[d.argmin(axis=1)]
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


def paint_cushion(layer, allowed, steps, dirs=("up", "down", "left", "right")):
    """Grow the layer's own colours into allowed pixels (hidden at rest)."""
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        for d in dirs:
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


def _assign_blob(blob, box, wing_l, wing_r, tail):
    x0, x1, y0, y1, cx, cy = box
    if y0 >= 64 and 16 <= cx <= 78:
        tail |= blob
    elif y1 <= 50 and x1 <= 48 and cx < 48:
        wing_l |= blob
    elif y1 <= 50 and x0 >= 50 and cx > 50:
        wing_r |= blob
    elif y0 >= 74:
        tail |= blob
    elif y1 <= 48 and x1 <= 48:
        wing_l |= blob
    elif y1 <= 48 and x0 >= 50:
        wing_r |= blob


def segment(img, yellow, kind):
    """Wings and the tail only. The central eye stays on the torso. No mouth."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    wing_l = np.zeros_like(op)
    wing_r = np.zeros_like(op)
    tail = np.zeros_like(op)
    for pts in component_list(is_color(img, yellow)):
        blob = mask_of(pts, op.shape)
        _assign_blob(blob, bounds_of(blob), wing_l, wing_r, tail)
    for pts in component_list(any_color(img, PATTERN)):
        blob = mask_of(pts, op.shape)
        box = bounds_of(blob)
        x0, x1, y0, y1, cx, cy = box
        if y0 >= 74:
            tail |= blob
        elif y1 <= 48 and x1 <= 48:
            wing_l |= blob
        elif y1 <= 48 and x0 >= 50:
            wing_r |= blob
        elif y1 <= 44 and cx < 46:
            wing_l |= blob
        elif y1 <= 44 and cx > 54:
            wing_r |= blob
    for col in (GLD, BRN):
        for pts in component_list(is_color(img, col)):
            blob = mask_of(pts, op.shape)
            box = bounds_of(blob)
            x0, x1, y0, y1, cx, cy = box
            if y0 >= 66:
                tail |= blob
            elif y1 <= 46 and x1 <= 46:
                wing_l |= blob
            elif y1 <= 46 and x0 >= 54:
                wing_r |= blob
    low = op & (yy >= 86) & (xx >= 16) & (xx <= 72)
    tail |= low & ~wing_l & ~wing_r

    outline = (is_color(img, INK) | is_color(img, DK)) & ~wing_l & ~wing_r & ~tail
    eye = any_color(img, EYE)
    torso_core = eye | (
        op & (yy >= 20) & (yy < 68) & (xx >= 36) & (xx <= 66) & ~wing_l & ~wing_r & ~tail
    )
    parts = {"wing_l": wing_l, "wing_r": wing_r, "tail": tail}
    assigned = wing_l | wing_r | tail
    pool = outline & ~assigned
    for _ in range(12):
        grew = False
        for pts in component_list(pool & ~assigned):
            blob = mask_of(pts, op.shape)
            touch = [n for n, m in parts.items() if (dilate(blob, 1) & m).any()]
            hits_core = bool((dilate(blob, 1) & torso_core).any())
            if len(touch) == 1 and not hits_core:
                parts[touch[0]] |= blob
                assigned |= blob
                grew = True
                break
        if not grew:
            break
    for _ in range(8):
        grew = False
        for pts in component_list(pool & ~assigned, conn=8):
            blob = mask_of(pts, op.shape)
            if len(pts) > 40:
                continue
            touch = [n for n, m in parts.items() if (dilate8(blob) & m).any()]
            hits_core = bool((dilate8(blob) & torso_core).any())
            if len(touch) == 1 and not hits_core:
                parts[touch[0]] |= blob
                assigned |= blob
                grew = True
                break
        if not grew:
            break

    def torso_now():
        return op & ~(parts["wing_l"] | parts["wing_r"] | parts["tail"])

    # Wing-eye scraps and tail scraps that the outline pass left on the torso.
    for _ in range(6):
        grew = False
        body = torso_now()
        for pts in component_list(body, conn=8):
            if len(pts) > 80:
                continue
            blob = mask_of(pts, op.shape)
            x0, x1, y0, y1, cx, cy = bounds_of(blob)
            touch = [n for n, m in parts.items() if (dilate8(blob) & m).any() or (dilate(blob, 2) & m).any()]
            if len(touch) != 1:
                continue
            name = touch[0]
            lateral = (name == "wing_r" and x0 >= 58 and y1 <= 46) or (
                name == "wing_l" and x1 <= 48 and y1 <= 48
            ) or (name == "tail" and y0 >= 60)
            if not lateral:
                continue
            # Keep the central eye on the torso.
            if 38 <= cx <= 64 and 40 <= cy <= 66:
                continue
            parts[name] |= blob
            grew = True
            break
        if not grew:
            break

    wing_l = parts["wing_l"]
    wing_r = parts["wing_r"]
    tail = parts["tail"]
    # Specks that do not 8-connect to the main piece stay on the torso.
    for name in ("wing_l", "wing_r", "tail"):
        groups = component_list(parts[name], conn=8)
        if len(groups) <= 1:
            continue
        keep = mask_of(groups[0], op.shape)
        for pts in groups[1:]:
            if len(pts) <= 20:
                continue
            keep |= mask_of(pts, op.shape)
        parts[name] = keep

    groups = component_list(parts["tail"], conn=8)
    masks = {"wing_l": parts["wing_l"], "wing_r": parts["wing_r"]}
    if len(groups) == 1:
        masks["tail"] = mask_of(groups[0], op.shape)
    elif len(groups) == 2:
        masks["tail"] = mask_of(groups[0], op.shape)
        masks["tail_b"] = mask_of(groups[1], op.shape)
    else:
        raise SystemExit(f"{kind} tail split into {len(groups)} pieces")
    torso = op & ~np.logical_or.reduce(list(masks.values()))
    masks["torso"] = torso
    check_partition(op, masks)
    for name, m in masks.items():
        if name == "torso":
            continue
        groups = component_list(m, conn=8)
        if len(groups) != 1:
            extra = bounds_of(mask_of(groups[1], op.shape))
            raise SystemExit(f"{kind} {name} is not one piece, next {extra}")
        if int(m.sum()) < 80:
            raise SystemExit(f"{kind} {name} collapsed ({int(m.sum())})")
        if not (m & dilate8(torso)).any():
            raise SystemExit(f"{kind} {name} does not touch the body")
    if kind == "front":
        if bounds_of(masks["wing_r"])[2] > 4:
            raise SystemExit("front right wing lost the crest")
        if bounds_of(masks["tail"])[2] < 64:
            raise SystemExit("front tail climbed into the eye")
        if "tail_b" in masks:
            raise SystemExit("front tail is one lobe")
    else:
        if bounds_of(masks["wing_l"])[2] > 6:
            raise SystemExit("back left wing lost the crest")
        if "tail_b" not in masks:
            raise SystemExit("back tail lobes merged")
        if bounds_of(masks["tail"])[2] < 60 or bounds_of(masks["tail_b"])[2] < 60:
            raise SystemExit("back tail climbed into the eye")
    tgroups = component_list(torso, conn=8)
    if not tgroups:
        raise SystemExit(f"{kind} torso is empty")
    for pts in tgroups[1:]:
        if len(pts) <= 36:
            continue
        extra = bounds_of(mask_of(pts, op.shape))
        raise SystemExit(f"{kind} torso island {len(pts)} {extra}")
    return masks


def place(arr, src_foot):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    y0 = FOOT_Y - src_foot
    x0 = OFF_X
    h, w = arr.shape[:2]
    if y0 < 0 or x0 < 0 or y0 + h > CELL_H or x0 + w > CELL_W:
        raise SystemExit(f"place overflow y {y0} x {x0}")
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def joint_band(part, body, depth):
    touch = part & dilate(body, 1)
    if not touch.any():
        touch = part & dilate8(body)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def pivot_of(part, body):
    touch = part & dilate(body, 1)
    if not touch.any():
        touch = part & dilate8(body)
    if not touch.any():
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(body)
    cx, cy = float(tx.mean()), float(ty.mean())
    d = (xs.astype(np.float64) - cx) ** 2 + (ys.astype(np.float64) - cy) ** 2
    near = d <= d.min() + 0.5
    return float(xs[near].mean()), float(ys[near].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs.astype(np.float64) - pivot[0]) ** 2 + (ys.astype(np.float64) - pivot[1]) ** 2
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


def build_view(img, masks, kind, mirror, src_foot):
    canvas = place(img, src_foot)
    placed = {k: place(v, src_foot) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    facing = -1.0 if mirror else 1.0
    if kind == "back":
        facing = -facing
    pivots = {}
    signs = {}
    order = [n for n in PART_ORDER if n in placed]
    for name in order:
        depth = 6 if name.startswith("wing") else 4
        layers["torso"] = paint_cushion(layers["torso"], joint_band(placed[name], torso_m, depth), depth)
        pivots[name] = pivot_of(placed[name], torso_m)
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, head_dx=0, head_dy=0,
        wing=0.0, tail=0.0,
        wing_mode="up", tail_mode="up",
        lean=0.0, knock=0, flash=False, spark=0, ring=-1,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        p["wing"] = wave(f, n, 7.0, phase=0.4)
        p["tail"] = wave(f, n, 5.0, phase=1.3)
        bob = wave(f, n, 1.0, phase=0.2)
        p["bdy"] = -int(round(max(0.0, bob)))
        p["wing_mode"] = "up"
        p["tail_mode"] = "up"
    elif action == "walk":
        # Glide. There are no feet. The wings do the moving.
        bob = wave(f, n, 1.0, cycles=2.0)
        p["bdy"] = -int(round(max(0.0, bob)))
        p["bdx"] = int(round(wave(f, n, 1.0, cycles=2.0)))
        p["wing"] = wave(f, n, 16.0, phase=0.15)
        p["tail"] = wave(f, n, 8.0, phase=1.7)
        p["wing_mode"] = "up"
        p["tail_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 the wings and the tail rise. There is no mouth.
        # 5-8 are cleared after finish() so the back still stays at equal scale.
        # 9-12 both wings and the tail strike. Values sit on finish()'s envelope.
        wing = [0, 8, 14, 18, 14, 0, 0, 0, 0, 34, 34, 34, 34, 10, 4, 0]
        tail = [0, 4, 8, 10, 8, 0, 0, 0, 0, 22, 28, 28, 22, 8, 3, 0]
        bdx = [0, -1, -1, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        p["wing"] = float(wing[f])
        p["tail"] = float(tail[f])
        p["bdx"] = int(bdx[f])
        p["wing_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["tail_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.66, 0.46, 0.28, 0.14, 0.05, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["wing"] = 10.0 * env
        p["tail"] = 8.0 * env
        p["wing_mode"] = "up"
        p["tail_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["wing"] = 0.0
        p["tail"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["head_dx"] = 0
        p["head_dy"] = 0
        p["spark"] = 0
        p["ring"] = -1
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["wing_mode"] = "up"
        p["tail_mode"] = "up"
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


def body_shift(rig, p):
    fc = rig["facing"]
    return int(round(fc * p["bdx"])), int(round(p["bdy"]))


def angle_of(rig, p, name):
    if name.startswith("wing"):
        return part_angle(rig, name, p["wing_mode"], p["wing"])
    if name.startswith("tail"):
        return part_angle(rig, name, p["tail_mode"], p["tail"])
    return 0.0


def limb_names(rig):
    return [n for n in PART_ORDER if n in rig["layers"]]


def part_layer(rig, name, p):
    if name not in rig["layers"]:
        return None
    bdx, bdy = body_shift(rig, p)
    return moved(rig["layers"][name], angle_of(rig, p, name), rig["pivots"][name], bdx, bdy)


def compose(rig, p):
    out = np.zeros_like(rig["canvas"])
    bdx, bdy = body_shift(rig, p)
    paste(out, shift(rig["layers"]["torso"], bdx, bdy))
    for name in limb_names(rig):
        layer = part_layer(rig, name, p)
        if layer is not None:
            paste(out, layer)
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = any_color(out, OUTLINE_KEEP)
    out[op & ~keep, :3] = np.array(WHT, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def facing_tip(layer, facing):
    m = layer[..., 3] >= 128
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if facing > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHT, CYA, RED, GLD, INK)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r), (2, r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 5])


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name in limb_names(rig):
        layer = part_layer(rig, name, p)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate8(other)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    kind, mirror = VIEW_OF[d]
    if p["turned"]:
        kind = "back" if kind == "front" else "front"
    return rigs[(kind, mirror)]


def moving(p):
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["wing"]) >= 0.45 or abs(p["tail"]) >= 0.45:
        return True
    return False


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["wing"]) >= 0.45 or abs(p["tail"]) >= 0.45:
        return True
    return False


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    body = compose(rig, p)
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], piv)
        grounded = body[..., 3] >= 128
        if grounded.any():
            bot = int(np.nonzero(grounded)[0].max())
            body = shift(body, 0, FOOT_Y - bot)
    if rotated(p) or moving(p):
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        for name in limb_names(rig):
            layer = part_layer(rig, name, p)
            if layer is None:
                continue
            tip = facing_tip(layer, fc)
            if tip is not None:
                impact(img, (tip[0] + fc * 2, tip[1]), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_component(mask):
    groups = component_list(mask)
    if not groups:
        return 0
    return len(groups[0])


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
    y0 = max(0, int(ys.min()) - pad - 18)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 10)
    x0 = max(0, int(xs.min()) - pad - 22)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 22)
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


def frame_faults(img, pal):
    a = img[..., 3]
    partial = int(((a != 0) & (a != 255)).sum())
    op = a >= 128
    edge = bool(op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any())
    off = set()
    solid = a == 255
    if solid.any():
        for c in np.unique(img[solid][:, :3], axis=0):
            t = tuple(int(v) for v in c)
            if t not in pal:
                off.add(t)
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


def assert_facing(front, back):
    yy, xx = np.mgrid[0:96, 0:96]
    fw = is_color(front, WHT)
    bw = is_color(back, WHT)
    eye_f = float(np.nonzero(fw)[1].mean())
    eye_b = float(np.nonzero(bw)[1].mean())
    top_f = (front[..., 3] >= 128) & (yy <= 4)
    top_b = (back[..., 3] >= 128) & (yy <= 5)
    tip_f = float(np.nonzero(top_f)[1].mean())
    tip_b = float(np.nonzero(top_b)[1].mean())
    if not (tip_f > eye_f > 48.0):
        raise SystemExit(f"front is not facing right: tip {tip_f:.1f} eye {eye_f:.1f}")
    if not (tip_b < eye_b < 46.0):
        raise SystemExit(f"back is not away-left: tip {tip_b:.1f} eye {eye_b:.1f}")
    green = is_color(front, GRN)
    if float(np.nonzero(green)[1].mean()) <= eye_f:
        raise SystemExit("front iris is not right of the white")


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
    if int((front[..., 3] >= 128).sum()) != 2294 or int((back[..., 3] >= 128).sum()) != 2350:
        raise SystemExit("opaque count changed")
    assert_facing(front, back)
    fm = segment(front, YEL_F, "front")
    bm = segment(back, YEL_B, "back")
    rigs = {}
    for kind, img, masks, foot in (
        ("front", front, fm, FRONT_FOOT),
        ("back", back, bm, BACK_FOOT),
    ):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror, foot)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs["W"]["facing"] != -1.0 or rigs["E"]["facing"] != 1.0:
        raise SystemExit("front facing is not right-as-drawn")
    if rigs["NE"]["facing"] != 1.0 or rigs["N"]["facing"] != -1.0:
        raise SystemExit("back facing is not away-left as drawn")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "wing_l": (80, 170, 230, 255),
        "wing_r": (255, 197, 8, 255),
        "tail": (222, 0, 0, 255),
        "tail_b": (57, 164, 74, 255),
        "torso": (49, 49, 49, 255),
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
        "dex": 561, "form": "normal", "variant": "sigilyph", "gen": "gen5",
        "dir": "monsterSprite/gen5/0561_sigilyph/normal", "sheetDir": "agent-sprite-forge",
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
    rig_a = rig_for(rigs, d, rest)
    rig_b = rig_for(rigs, d, hit)
    la = part_layer(rig_a, name, rest)
    lb = part_layer(rig_b, name, hit)
    if la is None or lb is None:
        return None
    fa = rig_b["facing"]
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
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, flush=True)
    print("parts", {k: int(v.sum()) for k, v in fm.items()}, {k: int(v.sum()) for k, v in bm.items()}, flush=True)
    print("signs S", rigs["S"]["signs"], "N", rigs["N"]["signs"], flush=True)

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
    for d in ("S", "N", "E", "W"):
        p = pose("attack", 6)
        rig = rig_for(rigs, d, p)
        img = synth(rigs, d, "attack", 6, palette, effects=True)
        turn_diff[d] = int(np.any(img != rig["canvas"], axis=2).sum())
    print("turn diff", turn_diff, flush=True)
    if any(turn_diff.values()):
        raise SystemExit("attack 5-8 is not the opposite still")

    strike = {name: tip_shift(rigs, "S", name) for name in ("wing_l", "wing_r", "tail")}
    print("strike dx S", strike, flush=True)

    pal = {tuple(int(v) for v in c) for c in palette}
    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        off = set()
        partial = 0
        edge = []
        lost = []
        bodies = []
        stored = {d: {a: [] for a in ACTIONS} for d in ("S", "W", "N", "E")}
        y0, y1, x0, x1 = box
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
                    if d in stored:
                        stored[d][act].append(img[y0:y1, x0:x1])
        for d in ("S", "W", "N", "E"):
            rows = []
            for act in ACTIONS:
                frames = list(stored[d][act])
                blank = np.zeros_like(frames[0])
                while len(frames) < 16:
                    frames.append(blank)
                rows.append(tile_grid(frames, 16))
            sheet = np.concatenate(rows, axis=0)
            z = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
            Image.fromarray(z, "RGBA").save(work / f"samples_{d}.png")
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["wing"], 1), round(p["tail"], 1), p["wing_mode"], p["turned"], p["spin"], p["ring"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        rest_sum = int(sum(rest_diff.values()))
        report = {
            "rest_diff_max": max(rest_diff.values()),
            "turn_diff_max": max(turn_diff.values()),
            "off_palette": len(off),
            "partial_alpha": partial,
            "edge_n": len(edge),
            "edge_touching": edge[:12],
            "disconnected": len(lost),
            "body_main_min": round(min(bodies), 3) if bodies else 1,
        }
        print("qa_samples", json.dumps(report), flush=True)
        print(
            f"qa_samples 静止差分 {rest_sum} パレット外 {len(off)} "
            f"中間アルファ {partial} 枠接触 {len(edge)}",
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
    y0, y1, x0, x1 = box
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.concatenate(strips, axis=0)
    motion = np.repeat(np.repeat(motion, 2, 0), 2, 1)
    Image.fromarray(motion, "RGBA").save(OUT_DIR / "preview_motion.png")
    preview_image(sheets).save(OUT_DIR / "preview.png")
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
    rest_sum = int(sum(rest_diff.values()) + sum(idle0.values()))
    report = {
        "version": VERSION, "hash": digest, "rest_diff": rest_diff, "idle0": idle0,
        "turn_diff": turn_diff,
        "off_palette_colors": len(off), "partial_alpha": partial, "edge_touching": clip[:12],
        "edge_n": len(clip), "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "bounds": bounds, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (not any(rest_diff.values()) and not any(idle0.values()) and not any(turn_diff.values())
          and not off and partial == 0 and not clip)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    print(
        f"qa_full 静止差分 {rest_sum} パレット外 {len(off)} "
        f"中間アルファ {partial} 枠接触 {len(clip)}",
        flush=True,
    )
    meta = {
        "generator": "forge_gen5_sigilyph_actions", "version": VERSION, "key": KEY,
        "dex": 561, "variant": "sigilyph", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "floating sigil (two wings with eyes, central eye on the torso, hanging tail, no mouth, no feet)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-right (iris x52.6 > white x50.1, crest x84-87); back crest x8 is away-left",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
