#!/usr/bin/env python3
"""Maractus normal (0556) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x19-76 y13-83, 1840 px, 12 colours,
alpha 0/255, sole y83):
  Two eyes. Each is a yellow disk with one gold pupil (180, 156, 16).
  Left pupil (41, 48), right pupil (50, 43). The disks are the yellow and
  olive pixels flooded from those pupils inside x38-56 y40-52: 43 px,
  mean x 49.07. The mouth is the ink bar y54 x47-50, mean x 48.5, left of
  the eyes. The nearer empty pixel on that row is also to the left (x44,
  five steps) rather than the right edge (x68). The still faces screen-left.
  The mouth is a drawn ink line, so the roar uses a wedge and a ring in
  this still's own colours. No new mouth is invented on the back.
  Head flower: pink and magenta, x48-68 y13-27. Left arm: the pink flower
  and its shaft, x19-43 y30-52. Right arm: the yellow flower and its shaft,
  x58-76 y38-57. Two feet split under y78. The long toe is the ground line
  y83 x41-43 and belongs to the screen-left foot.
normal/back.png (bbox x13-75 y11-81, 1931 px, 13 colours, sole y81):
  No gold pupils, no ink mouth. One back-only rose (205, 74, 139) stays in
  the palette. The head flower mean x is right of the torso, so this back
  faces screen-right. Dropped 2 px so both soles share y83. Same two arms
  and two feet. No wings, horns, fangs, tail, or pincers are drawn.

Directions, front faces left, back faces right:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (screen-right). NW back mirrored (screen-left).
  E is screen-right, W is screen-left, N is the back.
  Attack 5-8 is that back at equal scale. S turns into N by flipping the
  mirror. W turns into NW. Back rows stay the back. No |cos| squash and no
  in-plane spin.

  idle 16: feet planted, flower and arms breathe. Frame 0 is the still.
  walk 16: the two drawn feet step. Arms and the flower sway.
  attack: 0 still, 1-4 mouth wedge and ring, arms and flower rise,
          5-8 the back still, 9 wind-up, 10-12 the arms strike (feet kick),
          11-12 impact, 13-15 return.
  hurt 12: lime flash, knock-back lean (<=38 deg) and recover.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_maractus_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0556_maractus/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0556_maractus/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0556_maractus"
VERSION = "maractus_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 208, 208
OFF_X = (CELL_W - 96) // 2
OFF_Y = 80
FRONT_FOOT = 83
BACK_FOOT = 81
BACK_DROP = FRONT_FOOT - BACK_FOOT
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. Back as drawn looks screen-right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Wind-up ceiling. The arm strike is drawn up to 34, above these, so
# finish() does not push a frame past the angle this script already paints.
# Frames 5-8 are cleared after finish() so the back still is not swung.
PEAKS = {"arm_l": 18.0, "arm_r": 18.0, "bloom": 12.0, "kick": 8.0}
LIMBS = ("bloom", "arm_l", "arm_r", "leg_l", "leg_r")

INK = (16, 16, 16)
LIME = (90, 213, 106)
GREEN = (65, 156, 74)
TEAL = (16, 139, 115)
TEAL_DK = (24, 98, 82)
FOREST = (32, 74, 32)
DEEP = (8, 57, 49)
PINK = (230, 106, 164)
MAGENTA = (180, 49, 115)
ROSE = (205, 74, 139)
YELLOW = (255, 213, 0)
OLIVE = (131, 115, 16)
GOLD = (180, 156, 16)
PETAL = (PINK, MAGENTA, ROSE)
SPIKE = (YELLOW, OLIVE, GOLD)
BODY = (LIME, GREEN, FOREST, TEAL, TEAL_DK, DEEP, INK)
OUTLINE = (INK, FOREST, YELLOW, OLIVE, GOLD, PINK, MAGENTA, ROSE)

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


def flood_from(seed, allow):
    out = np.zeros_like(allow, bool)
    seen = np.zeros_like(allow, bool)
    ys, xs = np.nonzero(seed & allow)
    stack = [(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())]
    for y, x in stack:
        seen[y, x] = True
    while stack:
        y, x = stack.pop()
        out[y, x] = True
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < 96 and 0 <= nx < 96 and allow[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    return out


def absorb_touching(mask, extra):
    out = mask.copy()
    while True:
        add = dilate(out, 1) & extra & ~out
        if not add.any():
            return out
        out |= add


def mask_from_pts(shape, pts):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m


def absorb_islands(masks):
    """Torso islands that touch one limb belong to that limb."""
    torso = masks["torso"]
    names = [n for n in LIMBS if n in masks]
    changed = True
    while changed:
        changed = False
        groups = component_list(torso)
        if len(groups) <= 1:
            break
        groups.sort(key=len, reverse=True)
        for g in groups[1:]:
            speck = mask_from_pts(torso.shape, g)
            hits = [n for n in names if touches(speck, masks[n])]
            if not hits:
                ys = [p[0] for p in g]
                xs = [p[1] for p in g]
                raise SystemExit(f"torso island touches nothing x{min(xs)}-{max(xs)} y{min(ys)}-{max(ys)}")
            best = max(hits, key=lambda n: int((dilate(speck, 1) & masks[n]).sum()))
            masks[best] |= speck
            torso = torso & ~speck
            changed = True
            break
    masks["torso"] = torso


def keep_main(mask, name):
    """Drop nothing. A diagonal 1 px flower speck may sit beside the main bloom."""
    groups = component_list(mask)
    if len(groups) <= 1:
        return mask
    groups.sort(key=len, reverse=True)
    main = mask_from_pts(mask.shape, groups[0])
    near = dilate(main, 1)
    # 8-neighbour of the main mass, so a diagonal outline pixel can stay.
    p = np.pad(main, 1, constant_values=False)
    near8 = near | p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
    for g in groups[1:]:
        speck = mask_from_pts(mask.shape, g)
        if len(g) <= 2 and (speck & near8).any():
            continue
        ys = [pt[0] for pt in g]
        xs = [pt[1] for pt in g]
        raise SystemExit(f"{name} fell apart n={len(g)} x{min(xs)}-{max(xs)} y{min(ys)}-{max(ys)}")
    return mask


def eye_and_mouth(img):
    """Front eyes and the ink mouth. Back returns empty masks."""
    yy, xx = np.mgrid[0:96, 0:96]
    pupils = is_color(img, GOLD) & (yy >= 40) & (yy <= 52) & (xx >= 38) & (xx <= 56)
    if int(pupils.sum()) != 2:
        return pupils, np.zeros(img.shape[:2], bool), None, None
    disk = any_color(img, SPIKE) & (yy >= 40) & (yy <= 52) & (xx >= 38) & (xx <= 56)
    eyes = flood_from(pupils, disk)
    seed = is_color(img, INK) & (yy == 54) & (xx >= 47) & (xx <= 50)
    mouth = flood_from(seed, is_color(img, INK) & (yy >= 52) & (yy <= 56) & (xx >= 44) & (xx <= 52))
    return eyes, mouth, float(np.nonzero(eyes)[1].mean()), float(np.nonzero(mouth)[1].mean())


def foot_split(op, y0):
    """X where the two toes separate. The right run starts at the returned x."""
    for y in range(int(np.nonzero(op)[0].max()), y0 - 1, -1):
        xs = np.nonzero(op[y])[0]
        if len(xs) < 4:
            continue
        gaps = np.where(np.diff(xs) > 1)[0]
        if len(gaps) == 1:
            left = int(xs[gaps[0]])
            right = int(xs[gaps[0] + 1])
            return (left + right) // 2 + 1
    raise SystemExit("feet do not split")


def validate_parts(kind, op, masks, foot):
    check_partition(op, masks)
    torso = masks["torso"]
    if largest_component(torso) != int(torso.sum()):
        sizes = sorted((len(g) for g in component_list(torso)), reverse=True)
        raise SystemExit(f"{kind} torso fell apart {sizes[:6]}")
    for name in LIMBS:
        part = masks[name]
        if int(part.sum()) < 8:
            raise SystemExit(f"{kind} {name} is too small: {int(part.sum())}")
        if not touches(part, torso):
            raise SystemExit(f"{kind} {name} does not touch the body")
        masks[name] = keep_main(part, f"{kind} {name}")
    yy = np.mgrid[0:96, 0:96][0]
    grounded = op & (yy == foot)
    feet = masks["leg_l"] | masks["leg_r"]
    if int(grounded.sum()) < 1 or int((grounded & feet).sum()) != int(grounded.sum()):
        raise SystemExit(f"{kind} ground line is not a foot")


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    eyes, mouth, ex, mx = eye_and_mouth(img)
    if int(eyes.sum()) < 20 or int(mouth.sum()) != 4:
        raise SystemExit(f"front face changed eyes {int(eyes.sum())} mouth {int(mouth.sum())}")
    if not (mx < ex):
        raise SystemExit(f"front does not face left: mouth {mx} eye {ex}")
    petal = any_color(img, (PINK, MAGENTA)) & (yy <= 28) & (xx >= 44)
    crown = any_color(img, OUTLINE) & (yy <= 27) & (xx >= 46)
    bloom = flood_from(petal, petal | crown)
    bloom = absorb_touching(bloom, any_color(img, OUTLINE) & (yy <= 28) & (xx >= 44) & ~eyes & ~mouth)
    left_petal = any_color(img, PETAL) & (xx <= 42) & (yy >= 28)
    left_spike = any_color(img, SPIKE) & (xx <= 40) & (yy >= 34) & (yy <= 52)
    allow_l = (
        left_petal | left_spike
        | (any_color(img, BODY) & (xx <= 43) & (yy >= 32) & (yy <= 54) & ~bloom & ~eyes & ~mouth)
    )
    arm_l = flood_from(left_petal | (left_spike & (xx <= 34)), allow_l)
    right_spike = any_color(img, SPIKE) & (xx >= 64) & (yy >= 34) & (yy <= 54)
    allow_r = op & (xx >= 58) & (yy >= 34) & (yy <= 58) & ~bloom & ~arm_l & ~eyes & ~mouth
    arm_r = flood_from(right_spike, allow_r)
    split = foot_split(op, 78)
    leg = op & (yy >= 78) & ~bloom & ~arm_l & ~arm_r
    masks = {
        "bloom": bloom, "arm_l": arm_l, "arm_r": arm_r,
        "leg_l": leg & (xx < split), "leg_r": leg & (xx >= split),
    }
    used = np.zeros_like(op)
    for m in masks.values():
        used |= m
    masks["torso"] = op & ~used
    absorb_islands(masks)
    if int((eyes & masks["torso"]).sum()) != int(eyes.sum()):
        raise SystemExit("front eyes left the face")
    if int((mouth & masks["torso"]).sum()) != int(mouth.sum()):
        raise SystemExit("front mouth left the face")
    validate_parts("front", op, masks, FRONT_FOOT)
    return masks, ex, mx, eyes, mouth


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    eyes, mouth, _ex, _mx = eye_and_mouth(img)
    if eyes.any() or mouth.any():
        raise SystemExit("back gained a face")
    if not is_color(img, ROSE).any():
        raise SystemExit("back lost its rose flower")
    petal = any_color(img, (PINK, MAGENTA)) & (yy <= 28) & (xx >= 44)
    crown = any_color(img, OUTLINE) & (yy <= 27) & (xx >= 44)
    bloom = flood_from(petal, petal | crown)
    bloom = absorb_touching(bloom, any_color(img, OUTLINE) & (yy <= 28) & (xx >= 42))
    left_petal = any_color(img, PETAL) & (xx <= 42) & (yy >= 28)
    left_spike = any_color(img, SPIKE) & (xx <= 40) & (yy >= 34) & (yy <= 54)
    allow_l = (
        left_petal | left_spike
        | (any_color(img, BODY) & (xx <= 43) & (yy >= 28) & (yy <= 54) & ~bloom)
    )
    arm_l = flood_from(left_petal | (left_spike & (xx <= 34)), allow_l)
    right_spike = any_color(img, SPIKE) & (xx >= 64) & (yy >= 32) & (yy <= 56)
    allow_r = op & (xx >= 58) & (yy >= 32) & (yy <= 58) & ~bloom & ~arm_l
    arm_r = flood_from(right_spike, allow_r)
    split = foot_split(op, 76)
    leg = op & (yy >= 76) & ~bloom & ~arm_l & ~arm_r
    masks = {
        "bloom": bloom, "arm_l": arm_l, "arm_r": arm_r,
        "leg_l": leg & (xx < split), "leg_r": leg & (xx >= split),
    }
    used = np.zeros_like(op)
    for m in masks.values():
        used |= m
    masks["torso"] = op & ~used
    absorb_islands(masks)
    validate_parts("back", op, masks, BACK_FOOT)
    by, bx = np.nonzero(masks["torso"])
    fy, fx = np.nonzero(masks["bloom"])
    bloom_x, body_x = float(fx.mean()), float(bx.mean())
    if not (bloom_x > body_x):
        raise SystemExit(f"back crown is not right of the body: {bloom_x} vs {body_x}")
    return masks, bloom_x, body_x


def place(arr, dy=0):
    if arr.ndim == 2:
        out = np.zeros((CELL_H, CELL_W), bool)
        y0 = OFF_Y + dy
        out[y0:y0 + 96, OFF_X:OFF_X + 96] = arr
        return out
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    y0 = OFF_Y + dy
    out[y0:y0 + 96, OFF_X:OFF_X + 96] = arr
    return out


def joint_band(part, torso, depth):
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 3)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def pivot_of(part, torso):
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 4)
    if not touch.any():
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(torso)
    cx, cy = float(tx.mean()), float(ty.mean())
    d = (xs.astype(np.float64) - cx) ** 2 + (ys.astype(np.float64) - cy) ** 2
    near = d <= d.min() + 2.0
    return float(xs[near].mean()), float(ys[near].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs.astype(np.float64) - pivot[0]) ** 2 + (ys.astype(np.float64) - pivot[1]) ** 2
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


def build_view(img, masks, kind, mirror, mouth):
    dy = 0 if kind == "front" else BACK_DROP
    canvas = place(img, dy)
    placed = {k: place(v, dy) for k, v in masks.items()}
    mouth_m = place(mouth, dy) if mouth is not None else np.zeros((CELL_H, CELL_W), bool)
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
        mouth_m = np.ascontiguousarray(mouth_m[:, ::-1])
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    depth = {"bloom": 6, "arm_l": 5, "arm_r": 5, "leg_l": 4, "leg_r": 4}
    for name in LIMBS:
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth[name]), depth[name],
        )
        pivots[name] = pivot_of(placed[name], torso_m)
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    has_mouth = bool(mouth_m.any())
    if has_mouth:
        my, mx = np.nonzero(mouth_m)
        pivots["mouth"] = (float(mx.mean()), float(my.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "has_mouth": has_mouth,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        arm_l=0.0, arm_r=0.0, bloom=0.0, kick=0.0, mode="up",
        step_l=(0, 0), step_r=(0, 0),
        lean=0.0, knock=0, flash=False, wedge=0, ring=-1, spark=0,
        turned=False, spin=0, head_dx=0, head_dy=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay on the ground. The flower and the arms breathe.
        p["bloom"] = wave(f, n, 4.0, phase=0.4)
        p["arm_l"] = wave(f, n, 5.0, phase=0.8)
        p["arm_r"] = wave(f, n, 5.0, phase=1.6)
        p["mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * sb)), -int(round(3 * max(0.0, sb))))
        p["arm_l"] = wave(f, n, 7.0, cycles=2.0, phase=0.2)
        p["arm_r"] = wave(f, n, 7.0, cycles=2.0, phase=0.2 + math.pi)
        p["bloom"] = wave(f, n, 3.0, cycles=2.0, phase=0.9)
        p["mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 raise the arms and the flower. The mouth wedge is drawn
        # later, and only on a view that has the ink mouth.
        # 5-8 the back, equal scale, cleared after finish().
        # 9 wind-up (18). 10-12 the arms strike at 34. Feet kick. 13-15 return.
        arm = [0, 8, 12, 16, 18, 0, 0, 0, 0, 18, 28, 34, 34, 12, 4, 0]
        bloom = [0, 6, 10, 14, 16, 0, 0, 0, 0, 12, 22, 28, 28, 10, 4, 0]
        kick = [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 16, 22, 22, 8, 3, 0]
        bdx = [0, -1, -1, -1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        wedge = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ring = [-1, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        p["arm_l"] = float(arm[f])
        p["arm_r"] = float(arm[f])
        p["bloom"] = float(bloom[f])
        p["kick"] = float(kick[f])
        p["bdx"] = int(bdx[f])
        p["wedge"] = int(wedge[f])
        p["ring"] = int(ring[f])
        p["mode"] = "up" if f <= 4 else "fwd"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["bloom"] = 6.0 * env
        p["arm_l"] = 8.0 * env
        p["arm_r"] = 8.0 * env
        p["mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["arm_l"] = 0.0
        p["arm_r"] = 0.0
        p["bloom"] = 0.0
        p["kick"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["wedge"] = 0
        p["ring"] = -1
        p["spark"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["head_dx"] = 0
        p["head_dy"] = 0
        p["turned"] = True
        p["mode"] = "up"
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
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    if name == "leg_l":
        sx, sy = p["step_l"]
        ang = part_angle(rig, name, "fwd", p["kick"])
        return ang, bdx + int(round(fc * sx)), bdy + int(sy)
    if name == "leg_r":
        sx, sy = p["step_r"]
        ang = part_angle(rig, name, "fwd", p["kick"])
        return ang, bdx + int(round(fc * sx)), bdy + int(sy)
    if name == "arm_l":
        return part_angle(rig, name, p["mode"], p["arm_l"]), bdx, bdy
    if name == "arm_r":
        return part_angle(rig, name, p["mode"], p["arm_r"]), bdx, bdy
    if name == "bloom":
        return part_angle(rig, name, p["mode"], p["bloom"]), bdx, bdy
    return 0.0, bdx, bdy


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    for name in ("leg_l", "leg_r", "arm_l", "arm_r", "bloom"):
        ang, dx, dy = limb_delta(rig, p, name)
        paste(out, moved(L[name], ang, piv[name], dx, dy))
    return out


def flash_hit(img):
    """Hit flash in the still's own light green. The ink outline stays."""
    out = img.copy()
    op = out[..., 3] >= 128
    keep = any_color(out, (INK, DEEP))
    out[op & ~keep, :3] = np.array(LIME, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def lean_angle(facing, deg):
    """Head moves opposite facing. Positive rotation is clockwise on screen."""
    return facing * deg


def mouth_exit(img, origin, fc):
    """First empty pixel in front of the drawn mouth."""
    x = int(round(origin[0]))
    y = int(round(origin[1]))
    step = 1 if fc >= 0 else -1
    h, w = img.shape[:2]
    for _ in range(48):
        x += step
        if not (1 <= x < w - 1 and 1 <= y < h - 1):
            break
        if img[y, x, 3] < 128:
            return float(x), float(y)
    return float(x), float(y)


def mouth_wedge(img, origin, fc, age):
    """A few pixels in the gap in front of the existing mouth. Empty pixels only."""
    ox, oy = origin
    cols = (INK, YELLOW, PINK)
    step = 1 if fc >= 0 else -1
    for i in range(max(0, int(age))):
        x = ox + step * i
        span = 0 if i == 0 else 1
        for dy in range(-span, span + 1):
            put(img, x, oy + dy, cols[min(i, 2)])


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (YELLOW, PINK, LIME)
    rad = 5 + int(age) * 3
    step = 1.0 if fc >= 0 else -1.0
    for deg in range(-40, 41, 10):
        a = math.radians(deg)
        x = ox + step * rad * math.cos(a)
        y = oy + rad * math.sin(a) * 0.62
        put(img, x, y, cols[int(age) % 3])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (YELLOW, PINK, LIME)
    r = 3 + int(age) * 2
    step = 1 if fc >= 0 else -1
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2))):
        put(img, ox + step * dx, oy + dy, cols[i % 3])


def forward_tip(rig, p):
    fc = rig["facing"]
    best = None
    for name in ("arm_l", "arm_r", "bloom", "leg_l", "leg_r"):
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
    """Attack 5-8 shows this side's back at equal scale. Back rows stay.

    S is the mirrored front. N is the back as drawn. Turning flips the mirror
    so S becomes N and W becomes NW.
    """
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", not mirror)]


def moving(p):
    if p["step_l"] != (0, 0) or p["step_r"] != (0, 0):
        return True
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    if abs(p["bloom"]) >= 0.45 or abs(p["arm_l"]) >= 0.45 or abs(p["arm_r"]) >= 0.45:
        return True
    if abs(p["kick"]) >= 0.45:
        return True
    if abs(p["lean"]) > 0.4:
        return True
    return False


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    if not moving(p):
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            body = rotate_nearest(body, lean_angle(fc, p["lean"]), rear_pivot(body, fc))
        body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_hit(img), palette)
    if rig["has_mouth"] and not p["turned"] and (p["wedge"] or p["ring"] >= 0):
        mx, my = rig["pivots"]["mouth"]
        origin = mouth_exit(
            img, (mx + int(round(fc * p["bdx"])), my + int(round(p["bdy"]))), fc,
        )
        if p["wedge"]:
            mouth_wedge(img, origin, fc, p["wedge"])
        if p["ring"] >= 0:
            roar_rings(img, origin, fc, p["ring"])
        img = snap(img, palette)
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
    for im in imgs:
        cols = np.unique(im[im[..., 3] >= 128][:, :3], axis=0)
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
    fm, ex, mx, eyes, mouth = segment_front(front)
    bm, bloom_x, body_x = segment_back(back)
    empty = np.zeros((96, 96), bool)
    rigs = {}
    for kind, img, masks, mouth_m in (
        ("front", front, fm, mouth),
        ("back", back, bm, empty),
    ):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror, mouth_m)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs["E"]["facing"] != 1.0 or rigs["W"]["facing"] != -1.0:
        raise SystemExit("E/W facing is not screen-right / screen-left")
    if rigs["N"]["facing"] != 1.0 or rigs["NW"]["facing"] != -1.0:
        raise SystemExit("back facing is not screen-right as drawn")
    if not rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("mouth is on the front only")
    for key in (("front", False), ("front", True), ("back", False)):
        rig = rigs[key]
        fc = rig["facing"]
        shift_x = crown_shift(rig["canvas"], fc, MAX_LEAN)
        if fc > 0 and not (shift_x < -5):
            raise SystemExit(f"hurt lean does not tip the head back ({shift_x})")
        if fc < 0 and not (shift_x > 5):
            raise SystemExit(f"hurt lean does not tip the head back ({shift_x})")
    return front, back, fm, bm, rigs, (ex, mx, bloom_x, body_x)


def save_parts(work, front, fm, back, bm):
    cols = {
        "bloom": (255, 140, 180, 255),
        "arm_l": (255, 170, 40, 255),
        "arm_r": (40, 160, 220, 255),
        "leg_l": (220, 60, 40, 255),
        "leg_r": (160, 60, 200, 255),
        "torso": (80, 190, 90, 255),
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
        "dex": 556, "form": "normal", "variant": "maractus", "gen": "gen5",
        "dir": "monsterSprite/gen5/0556_maractus/normal", "sheetDir": "agent-sprite-forge",
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
    front, back, fm, bm, rigs, measured = build_rigs()
    save_parts(work, front, fm, back, bm)
    ex, mx, bloom_x, body_x = measured
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    print("measure", "eye_x", round(ex, 2), "mouth_x", round(mx, 2),
          "back_bloom_x", round(bloom_x, 2), "back_body_x", round(body_x, 2), flush=True)
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

    turn_s = synth(rigs, "S", "attack", 6, palette, effects=False)
    turn_w = synth(rigs, "W", "attack", 6, palette, effects=False)
    turn_n = synth(rigs, "N", "attack", 6, palette, effects=False)
    if np.any(turn_s != rigs["N"]["canvas"]):
        raise SystemExit("S turn is not the N back at equal scale")
    if np.any(turn_w != rigs["NW"]["canvas"]):
        raise SystemExit("W turn is not the mirrored back at equal scale")
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
            atk.append((f, round(float(p["arm_l"]), 1), round(float(p["bloom"]), 1),
                        round(float(p["kick"]), 1), p["mode"], p["turned"],
                        p["spin"], p["wedge"], p["bdx"], p["spark"]))
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
        and partial == 0 and not clip
    )
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print("qa_full", "rest_diff", int(sum(rest_diff.values()) + sum(idle0.values())),
          "off_palette", len(off), "partial_alpha", partial, "edge_touch", len(clip),
          "body_main_min", report["body_main_min"], "clip", clip[:8], flush=True)
    meta = {
        "generator": "forge_gen5_maractus_actions", "version": VERSION, "key": KEY,
        "dex": 556, "variant": "maractus", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "cactus (two yellow eyes, ink mouth, head flower, two flower arms, two feet)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (mouth mean x 48.5 < eye mean x 49.07); back crown is right of the body",
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
