#!/usr/bin/env python3
"""Simisear normal (0514) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x18-78 y17-75, 2027 px, 13 colours,
alpha 0/255):
  Pupils are the black pixels at x40 y36 and x49 y36. Midpoint x 44.5.
  The mouth is a closed black stroke at x41-49 y44 (centre x 45.0). Every
  one of those pixels has four opaque neighbours, so there is no cavity.
  The face points at the viewer. The roar does not invent a wedge or rings.
  It raises the flame crest and cocks the drawn hand and the tail.
  Crest spike x42-55 y17-24. Left hand with claws x18-38 y45-54.
  Right flame tail, separate from y39, joins the hip by y58.
  Feet split at y64. Left foot ends y75. Right foot ends y74.
  Ears, eyes, muzzle, and the white chest ruffs stay on the torso.

normal/back.png (bbox x11-77 y19-77, 1922 px, 12 colours, all in the front
  palette). No eyes and no mouth. Crest spike x42-56 y19-25. The flame tail
  is the left mass (x11-40, down to the curl at y74). One outstretched arm
  on the right, x55-77 y47-56, with claws. Feet: left sole y76, right sole
  y77. The back is shifted up 2px so both ground feet share y75.

Directions, frontal still, back tail on the left:
  S SE E front mirrored (strike reads screen-right). W SW front as drawn.
  N NE back as drawn. NW back mirrored.
  E is screen-right, W is screen-left, N is the back. Attack 5-8 shows that
  direction's back at equal scale. No width squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_simisear_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0514_simisear/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0514_simisear/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0514_simisear"
VERSION = "simisear_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 192
OFF_X = (CELL_W - 96) // 2
OFF_Y = (CELL_H - 96) // 2
FRONT_FOOT = 75
BACK_FOOT = 77
FOOT_Y = OFF_Y + FRONT_FOOT
# Front faces the viewer. Back as drawn has the flame on the left, so the
# away-right group uses that back unmirrored. S turns into N.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# finish() may open frames 5-12 toward these. The slap and the tail whip are
# drawn at 34 deg, above these, so finish does not pass them. Frames 5-8 are
# then cleared so the back still stays whole.
PEAKS = {"crest": 14.0, "arm": 16.0, "tail": 16.0}
LIMBS = ("crest", "arm", "tail", "leg_l", "leg_r")
SWING = ("crest", "arm", "tail")

INK = (0, 0, 0)
GREY_D = (82, 82, 82)
BROWN = (115, 82, 32)
RED_DD = (123, 49, 49)
GREY = (156, 156, 156)
BROWN_M = (164, 123, 57)
RED_D = (189, 65, 57)
GREY_L = (197, 197, 197)
GOLD = (205, 164, 90)
RED = (230, 82, 65)
ORANGE = (246, 148, 65)
CREAM = (255, 205, 123)
WHITE = (255, 255, 255)

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


def reconnect(img):
    """Close a 1-2px gap after a limb moves. Effects are drawn later."""
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


def row_segments(mask_row):
    xs = np.nonzero(mask_row)[0]
    if len(xs) == 0:
        return []
    segs = []
    start = prev = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x > prev + 1:
            segs.append((start, prev))
            start = x
        prev = x
    segs.append((start, prev))
    return segs


def paint_seg(mask, y, seg):
    if seg[0] > seg[1]:
        return
    mask[y, seg[0]:seg[1] + 1] = True


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
    groups = component_list(mask)
    if not groups:
        return 0
    return max(len(g) for g in groups)


def require_piece(name, mask):
    if not mask.any():
        raise SystemExit(f"{name} is missing")
    if largest_component(mask) != int(mask.sum()):
        raise SystemExit(f"{name} fell apart ({largest_component(mask)} vs {int(mask.sum())})")


def claim_specks(torso, limbs):
    """A torso speck that touches one limb rides with that limb."""
    groups = component_list(torso)
    if len(groups) <= 1:
        return torso
    groups.sort(key=len, reverse=True)
    keep = np.zeros_like(torso)
    for y, x in groups[0]:
        keep[y, x] = True
    h, w = torso.shape
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    for pts in groups[1:]:
        target = None
        for name, mask in limbs.items():
            hit = False
            for y, x in pts:
                for dy, dx in neigh:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                target = name
                break
        if target is None:
            for y, x in pts:
                keep[y, x] = True
        else:
            for y, x in pts:
                limbs[target][y, x] = True
    return keep


def sole_y(mask):
    ys = np.nonzero(mask)[0]
    if len(ys) == 0:
        raise SystemExit("empty sole")
    return int(ys.max())


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    crest = np.zeros_like(op)
    for y in range(17, 25):
        for seg in row_segments(op[y]):
            if 42 <= seg[0] and seg[1] <= 55:
                paint_seg(crest, y, seg)
    arm = np.zeros_like(op)
    for y in range(45, 55):
        segs = row_segments(op[y])
        if y == 50 and len(segs) == 1:
            paint_seg(arm, y, (segs[0][0], 33))
            continue
        for seg in segs:
            if seg[1] <= 38 and seg[0] <= 36:
                paint_seg(arm, y, seg)
    tail = np.zeros_like(op)
    rest = op & ~arm & ~crest
    for y in range(39, 59):
        segs = row_segments(rest[y])
        if len(segs) >= 2:
            for seg in segs:
                if seg[0] >= 56:
                    paint_seg(tail, y, seg)
        elif len(segs) == 1 and segs[0][1] >= 70 and segs[0][0] <= 40:
            x0 = max(segs[0][0], 62)
            if x0 <= segs[0][1]:
                paint_seg(tail, y, (x0, segs[0][1]))
    leg_l = np.zeros_like(op)
    leg_r = np.zeros_like(op)
    used = crest | arm | tail
    for y in range(64, 96):
        segs = row_segments((op & ~used)[y])
        if len(segs) >= 2:
            paint_seg(leg_l, y, segs[0])
            paint_seg(leg_r, y, segs[-1])
        elif len(segs) == 1:
            if segs[0][1] < 48:
                paint_seg(leg_l, y, segs[0])
            else:
                paint_seg(leg_r, y, segs[0])
    limbs = {"crest": crest, "arm": arm, "tail": tail, "leg_l": leg_l, "leg_r": leg_r}
    torso = claim_specks(op & ~crest & ~arm & ~tail & ~leg_l & ~leg_r, limbs)
    masks = dict(limbs)
    masks["torso"] = torso
    check_partition(op, masks)
    for name in LIMBS:
        require_piece(name, masks[name])
        if not (masks[name] & dilate(torso, 1)).any():
            raise SystemExit(f"front {name} does not touch the body")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("front torso fell apart")
    pupils = is_color(img, INK) & (yy == 36) & ((xx == 40) | (xx == 49))
    if int(pupils.sum()) != 2 or int((pupils & torso).sum()) != 2:
        raise SystemExit("front pupils left the head")
    mouth = is_color(img, INK) & (yy == 44) & (xx >= 41) & (xx <= 49)
    if int(mouth.sum()) != 9 or int((mouth & torso).sum()) != 9:
        raise SystemExit("front mouth line left the face")
    # Closed stroke: no empty neighbour, so it is not a cavity.
    for y, x in zip(*np.nonzero(mouth)):
        n = 0
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if op[y + dy, x + dx]:
                n += 1
        if n != 4:
            raise SystemExit("front mouth opened into a cavity")
    eye_x = (40.0 + 49.0) / 2.0
    mouth_x = float(np.nonzero(mouth)[1].mean())
    if abs(mouth_x - eye_x) > 2.0:
        raise SystemExit(f"front facing changed: mouth {mouth_x:.1f} eyes {eye_x:.1f}")
    if float(np.nonzero(arm)[1].mean()) >= 42:
        raise SystemExit("front hand is not on the left")
    if float(np.nonzero(tail)[1].mean()) <= 60:
        raise SystemExit("front tail is not on the right")
    if sole_y(leg_l) != FRONT_FOOT or sole_y(leg_r) != FRONT_FOOT - 1:
        raise SystemExit("front feet moved")
    ruff = is_color(img, WHITE) & (yy >= 42)
    if ruff.any() and int((ruff & arm).sum()):
        raise SystemExit("front ruff was cut into the hand")
    counts = {k: int(v.sum()) for k, v in masks.items()}
    expect = {"crest": 50, "arm": 128, "tail": 269, "leg_l": 99, "leg_r": 216, "torso": 1265}
    if counts != expect:
        raise SystemExit(f"front parts changed {counts}")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    if is_color(img, ORANGE).any():
        raise SystemExit("back gained the front ear orange")
    crest = np.zeros_like(op)
    for y in range(19, 26):
        for seg in row_segments(op[y]):
            if 42 <= seg[0] and seg[1] <= 56 and seg[1] - seg[0] <= 14:
                paint_seg(crest, y, seg)
    arm = np.zeros_like(op)
    for y in range(47, 57):
        segs = row_segments(op[y])
        if not segs:
            continue
        if y == 51 and segs[-1][1] >= 74 and segs[-1][0] < 60:
            paint_seg(arm, y, (63, segs[-1][1]))
            continue
        for seg in segs:
            if seg[0] >= 55 and seg[1] >= 64:
                paint_seg(arm, y, seg)
    tail = np.zeros_like(op)
    blocked = arm | crest
    for y in range(42, 75):
        segs = row_segments((op & ~blocked)[y])
        for seg in segs:
            if y <= 50 and seg[0] <= 36 and seg[1] <= 36:
                paint_seg(tail, y, seg)
            elif y == 51 and seg[0] <= 16:
                paint_seg(tail, y, (seg[0], min(seg[1], 36)))
            elif 52 <= y <= 65 and seg[0] <= 20 and seg[1] <= 41:
                paint_seg(tail, y, seg)
            elif 52 <= y <= 65 and seg[0] <= 16:
                paint_seg(tail, y, (seg[0], min(seg[1], 32)))
            elif 66 <= y <= 73 and seg[0] <= 22:
                paint_seg(tail, y, (seg[0], min(seg[1], 33)))
            elif y == 74 and seg[1] <= 33:
                paint_seg(tail, y, seg)
    leg_l = np.zeros_like(op)
    leg_r = np.zeros_like(op)
    used = crest | arm | tail
    for y in range(66, 96):
        segs = row_segments((op & ~used)[y])
        for seg in segs:
            if seg[0] >= 46:
                paint_seg(leg_r, y, seg)
            elif seg[1] <= 46 and seg[0] >= 30:
                paint_seg(leg_l, y, seg)
    limbs = {"crest": crest, "arm": arm, "tail": tail, "leg_l": leg_l, "leg_r": leg_r}
    torso = claim_specks(op & ~crest & ~arm & ~tail & ~leg_l & ~leg_r, limbs)
    masks = dict(limbs)
    masks["torso"] = torso
    check_partition(op, masks)
    for name in LIMBS:
        require_piece(name, masks[name])
        if not (masks[name] & dilate(torso, 1)).any():
            raise SystemExit(f"back {name} does not touch the body")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("back torso fell apart")
    face = is_color(img, ORANGE) | (is_color(img, CREAM) & (yy < 40))
    if face.any():
        raise SystemExit("back gained a face")
    if float(np.nonzero(tail)[1].mean()) >= float(np.nonzero(torso)[1].mean()):
        raise SystemExit("back tail is not on the left")
    if float(np.nonzero(arm)[1].mean()) <= 60:
        raise SystemExit("back arm is not on the right")
    if sole_y(leg_r) != BACK_FOOT or sole_y(leg_l) != BACK_FOOT - 1:
        raise SystemExit("back feet moved")
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
    depth = {"crest": 4, "arm": 5, "tail": 6, "leg_l": 5, "leg_r": 5}
    for name in LIMBS:
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth[name]), depth[name] + 1
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in SWING:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    for name in ("leg_l", "leg_r"):
        pivots[name] = pivot_of(layers[name], layers["torso"])
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    soles = {}
    for name in ("leg_l", "leg_r"):
        soles[name] = int(np.nonzero(placed[name])[0].max())
    ground = max(soles, key=soles.get)
    gap = int(soles[ground] - min(soles.values()))
    foot_cx = {n: float(np.nonzero(placed[n])[1].mean()) for n in ("leg_l", "leg_r")}
    if facing > 0:
        near = max(("leg_l", "leg_r"), key=lambda n: foot_cx[n])
    else:
        near = min(("leg_l", "leg_r"), key=lambda n: foot_cx[n])
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
        "ground_leg": ground, "sole_gap": gap, "near_foot": near,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        crest=0.0, arm=0.0, tail=0.0,
        crest_mode="up", arm_mode="back", tail_mode="back",
        thrust=0,
        step_l=(0, 0), step_r=(0, 0), kick=(0, 0),
        feet_follow=False,
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The crest, the hand, and the tail breathe.
        p["bdy"] = -1 if wave(f, n, 1.0, phase=0.4) > 0.55 else 0
        p["crest"] = wave(f, n, 7.0, phase=0.3)
        p["arm"] = wave(f, n, 5.0, phase=0.8)
        p["tail"] = wave(f, n, 9.0, phase=1.2)
        p["crest_mode"] = "up"
        p["arm_mode"] = "up"
        p["tail_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * max(0.0, sa))), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * max(0.0, sb))), -int(round(3 * max(0.0, sb))))
        p["crest"] = wave(f, n, 5.0, phase=0.2)
        p["arm"] = wave(f, n, 8.0, phase=0.4)
        p["tail"] = wave(f, n, 8.0, phase=math.pi / 2)
        p["crest_mode"] = "up"
        p["arm_mode"] = "up"
        p["tail_mode"] = "up"
        p["feet_follow"] = True
    elif action == "attack":
        # 0 rest. 1-4 crest up, hand and tail cock back. No mouth cavity.
        # 5-8 the back still, equal scale. 9-12 hand slap and tail whip.
        # 13-15 recover.
        crest = [0, 8, 14, 18, 20, 0, 0, 0, 0, 6, 4, 0, 0, 4, 0, 0]
        arm = [0, 6, 12, 16, 18, 0, 0, 0, 0, 18, 26, 34, 34, 14, 4, 0]
        tail = [0, 8, 14, 18, 20, 0, 0, 0, 0, 16, 26, 34, 34, 12, 4, 0]
        bdx = [0, 0, -1, -1, -1, 0, 0, 0, 0, 1, 1, 2, 1, 1, 0, 0]
        p["crest"] = float(crest[f])
        p["arm"] = float(arm[f])
        p["tail"] = float(tail[f])
        p["bdx"] = int(bdx[f])
        # The claw rises on every view. "fwd" drops the mirrored hand and, on
        # the back, drives the flame through the ground. Thrust is the reach.
        p["crest_mode"] = "up"
        p["arm_mode"] = "up"
        if 1 <= f <= 4:
            p["tail_mode"] = "back"
            p["thrust"] = -1 if f >= 2 else 0
        elif 9 <= f <= 14:
            p["tail_mode"] = "up"
            p["thrust"] = 2 if f in (11, 12) else (1 if f in (10, 13) else 0)
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["kick"] = (2, -2)
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.75, 1.0, 0.86, 0.64, 0.42, 0.24, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["crest"] = 12.0 * env
        p["arm"] = 8.0 * env
        p["tail"] = 10.0 * env
        p["crest_mode"] = "up"
        p["arm_mode"] = "up"
        p["tail_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["crest"] = 0.0
        p["arm"] = 0.0
        p["tail"] = 0.0
        p["thrust"] = 0
        p["bdx"] = 0
        p["bdy"] = 0
        p["kick"] = (0, 0)
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["spin"] = 0
        p["spark"] = 0
    if p["arm"] > 34.0:
        p["arm"] = 34.0
    if p["tail"] > 34.0:
        p["tail"] = 34.0
    if p["lean"] > MAX_LEAN:
        p["lean"] = MAX_LEAN
    return p


def part_angle(rig, name, mode, amp):
    if abs(amp) < 0.45 or name not in rig["signs"]:
        return 0.0
    sign = rig["signs"][name]
    if mode == "up":
        return sign["up"] * amp
    if mode == "back":
        return -sign["fwd"] * amp
    return sign["fwd"] * amp


def moved(layer, ang, pivot, dx, dy):
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, pivot)
    return shift(layer, dx, dy)


def plant_drop(rig, p):
    """When the long foot is up, drop so the short foot reaches the ground line."""
    gap = int(rig["sole_gap"])
    if gap <= 0:
        return 0
    g = rig["ground_leg"]
    sy = p["step_l"][1] if g == "leg_l" else p["step_r"][1]
    if g == rig["near_foot"]:
        sy += int(p["kick"][1])
    if sy < 0:
        return min(gap, -int(sy))
    return 0


def limb_layers(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    thrust = int(p["thrust"])
    parts = {}
    for name in LIMBS:
        if name == "leg_l":
            sx, sy = p["step_l"]
            ang = 0.0
            extra = 0
        elif name == "leg_r":
            sx, sy = p["step_r"]
            ang = 0.0
            extra = 0
        else:
            sx, sy = 0, 0
            mode = p[f"{name}_mode"]
            # Back flame's "forward" rotation points down through the floor.
            if name == "tail" and rig["kind"] == "back" and mode == "fwd":
                mode = "up"
            ang = part_angle(rig, name, mode, p[name])
            extra = thrust if name == "arm" else 0
        if name == rig["near_foot"]:
            kx, ky = p["kick"]
            sx += int(kx)
            sy += int(ky)
        if name in ("leg_l", "leg_r"):
            dy = (bdy if p["feet_follow"] else 0) + int(sy)
        else:
            dy = bdy + int(sy)
        dx = bdx + int(round(fc * (sx + extra)))
        parts[name] = moved(L[name], ang, piv[name], dx, dy)
    return parts


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    # Torso first. Limbs cover the cushion at rest, so frame 0 matches.
    paste(out, shift(L["torso"], bdx, bdy))
    parts = limb_layers(rig, p)
    for name in ("tail", "leg_l", "leg_r", "crest", "arm"):
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
    """The hand tip, toward facing. The slap lands there."""
    layer = limb_layers(rig, p)["arm"]
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    fc = rig["facing"]
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (ORANGE, CREAM, WHITE, GOLD)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -3), (2, 3))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


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
    """Attack 5-8 faces away. Front rows become that side's back, unscaled."""
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", not mirror)]


def body_moved(p):
    return (
        abs(p["crest"]) >= 0.45 or abs(p["arm"]) >= 0.45 or abs(p["tail"]) >= 0.45
        or p["step_l"] != (0, 0) or p["step_r"] != (0, 0)
        or p["kick"] != (0, 0) or p["thrust"] or p["bdx"] or p["bdy"]
        or p["knock"] or abs(p["lean"]) > 0.4
    )


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    if action == "walk" or (action == "attack" and p["kick"] != (0, 0)):
        p = dict(p)
        p["bdy"] = int(p["bdy"]) + plant_drop(rig, p)
        if action == "attack":
            p["feet_follow"] = True
    fc = rig["facing"]
    if not body_moved(p):
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            piv = rear_pivot(body, fc)
            body = rotate_nearest(body, -fc * p["lean"], piv)
            low = np.nonzero(body[..., 3] >= 128)[0]
            if len(low) and int(low.max()) > FOOT_Y:
                body = shift(body, 0, FOOT_Y - int(low.max()))
        body = reconnect(fill_pinholes(despeckle(body)))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = weapon_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def head_x(img):
    """Crest tip. Hurt should carry it backward."""
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0 = int(ys.min())
    band = ys <= y0 + 8
    return float(xs[band].mean())


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


def crop_box(canvas, pad=10):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad - 16)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 18)
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
        raise SystemExit("front foot line changed")
    if int(np.nonzero(back[..., 3])[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line changed")
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
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
        "crest": (255, 220, 40, 255),
        "arm": (40, 210, 80, 255),
        "tail": (255, 90, 40, 255),
        "leg_l": (180, 80, 255, 255),
        "leg_r": (80, 120, 255, 255),
        "torso": (160, 160, 160, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        a = img[..., 3] >= 128
        ys, xs = np.nonzero(a)
        y0, y1 = int(ys.min()) - 1, int(ys.max()) + 2
        x0, x1 = int(xs.min()) - 1, int(xs.max()) + 2
        tiles.append(np.repeat(np.repeat(part[y0:y1, x0:x1], 4, 0), 4, 1))
    h = max(t.shape[0] for t in tiles)
    padded = []
    for t in tiles:
        if t.shape[0] == h:
            padded.append(t)
            continue
        z = np.zeros((h, t.shape[1], 4), np.uint8)
        z[:t.shape[0]] = t
        padded.append(z)
    gap = np.zeros((h, 8, 4), np.uint8)
    Image.fromarray(np.concatenate([padded[0], gap, padded[1]], axis=1), "RGBA").save(work / "parts.png")


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
        "dex": 514, "form": "normal", "variant": "simisear", "gen": "gen5",
        "dir": "monsterSprite/gen5/0514_simisear/normal", "sheetDir": "agent-sprite-forge",
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
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, flush=True)
    print("near", {d: rigs[d]["near_foot"] for d in DIRS},
          "ground", {d: (rigs[d]["ground_leg"], rigs[d]["sole_gap"]) for d in ("S", "W", "N")},
          flush=True)

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
    leans = [pose("hurt", f)["lean"] for f in range(12)]
    if max(leans) > MAX_LEAN + 0.01 or leans[0] != 0 or leans[-1] != 0:
        raise SystemExit(f"hurt lean out of range {leans}")
    if pose("attack", 11)["spin"] != 0 or pose("attack", 6)["turned"] is not True:
        raise SystemExit("attack spin or turn flag is wrong")
    for d in ("W", "E", "N", "S"):
        rest = synth(rigs, d, "hurt", 0, palette, effects=False)
        hit = synth(rigs, d, "hurt", 3, palette, effects=False)
        fc = rigs[d]["facing"]
        dx = head_x(hit) - head_x(rest)
        if fc > 0 and dx >= -1:
            raise SystemExit(f"{d} hurt leaned forward ({dx:.1f})")
        if fc < 0 and dx <= 1:
            raise SystemExit(f"{d} hurt leaned forward ({dx:.1f})")

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
            ("attack", 0), ("attack", 4), ("attack", 6), ("attack", 8),
            ("attack", 11), ("attack", 12), ("attack", 15),
            ("hurt", 3), ("hurt", 6),
        ):
            zoom_frames.append(synth(rigs, "S", act, f, palette)[y0:y1, x0:x1])
        z = tile_grid(zoom_frames, 6)
        Image.fromarray(np.repeat(np.repeat(z, 3, 0), 3, 1), "RGBA").save(work / "samples_zoom.png")
        lost = []
        soles = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    if act == "walk" or (act == "attack" and p["kick"] != (0, 0)):
                        p = dict(p)
                        p["bdy"] = int(p["bdy"]) + plant_drop(rig, p)
                        if act == "attack":
                            p["feet_follow"] = True
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
                    if act in ("idle", "walk", "hurt") or (act == "attack" and f not in range(5, 9)):
                        img = synth(rigs, d, act, f, palette, effects=False)
                        ys = np.nonzero(img[..., 3] >= 128)[0]
                        if len(ys) and int(ys.max()) != FOOT_Y:
                            soles.append(f"{act}:{d}:{f}:{int(ys.max())}")
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        print("sole off", soles[:24], "count", len(soles), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["crest"], 1), round(p["arm"], 1), p["arm_mode"],
                        round(p["tail"], 1), p["turned"], p["spin"]))
        print("attack poses", atk, flush=True)
        print("hurt lean", [round(v, 1) for v in leans], "max", max(leans), flush=True)
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
            if key in state["done"]:
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
        "generator": "forge_gen5_simisear_actions", "version": VERSION, "key": KEY,
        "dex": 514, "variant": "simisear", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped fire monkey (flame crest, one clawed hand, flame tail, two feet, closed mouth line)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces the viewer (mouth x within 1px of the eye midpoint); back tail is left of the body",
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
