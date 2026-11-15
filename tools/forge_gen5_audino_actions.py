#!/usr/bin/env python3
"""Audino normal (0531) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x21-65 y19-70, 1505 px, 12 colours,
alpha 0/255):
  Eyes are #1862ac plus two white highlights, mean x 40.50 (y28-33).
  The nose is the five #a48b62 pixels at x38-42 y35-36, mean x 40.00, left of
  the eyes. The muzzle is closed cream. No mouth cavity is drawn, so the roar
  does not invent a wedge or rings. It raises the two ear feelers that are
  already there.
  Left feeler x21-31 y26-40 (cream/tan curl off the left ear). Right feeler
  x54-65 y31-44. One hand sticks out at x53-61 y51-53. Two feet split at y65:
  the short foot ends y69, the ground foot x43-51 ends y70. A 26px
  lavender/grey fluff sits on the rear (screen-right) at x53-56 y53-64.

normal/back.png (bbox x27-68 y23-73, 1452 px). No eyes. The rear fluff
  (#d5d5e6, #62626a, and #eeeef6 which is not in the front palette) is the
  left mass x27-41 y48-63. The cream curl is on the right, so this back
  faces screen-right. The ground foot ends y73. The back is shifted up 3px
  so both ground feet share one line.

Directions, front faces screen-left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (fluff left, cream right). NW back mirrored.
  E is screen-right, W is screen-left, N is the back. Attack 5-8 swaps to
  the opposite still at equal scale. No width squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_audino_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0531_audino/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0531_audino/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0531_audino"
VERSION = "audino_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 192
OFF_X = (CELL_W - 96) // 2
OFF_Y = (CELL_H - 96) // 2
FRONT_FOOT = 70
BACK_FOOT = 73
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left (nose x < eye x, rear fluff on the right).
# Back as drawn has the fluff on the left and the cream curl on the right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Amounts finish() may open on frames 5-12. The hit itself is drawn higher
# (tail 34, feelers 20, hand 4), so finish() does not lower it.
PEAKS = {"feeler_l": 8.0, "feeler_r": 8.0, "tail": 12.0, "hand": 3.0, "kick": 2.0}
ROTATE_PARTS = ("feeler_l", "feeler_r", "tail")

PINK = (222, 139, 148)
DPINK = (205, 106, 115)
BROWN = (115, 74, 74)
CREAM = (246, 230, 172)
TAN = (205, 180, 131)
GOLD = (164, 139, 98)
OLIVE = (106, 98, 74)
INK = (16, 16, 16)
BLUE = (24, 98, 172)
WHITE = (255, 255, 255)
LAV = (213, 213, 230)
GREY = (98, 98, 106)
PALE = (238, 238, 246)
FEELER_COLS = (CREAM, TAN, GOLD, OLIVE, INK)
BODY_BLOCK = (PINK, DPINK, BROWN, BLUE, WHITE)

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


def flood(mask, seeds):
    out = seeds & mask
    while True:
        nxt = dilate(out, 1) & mask
        if (nxt == out).all():
            return out
        out = nxt


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


def sole_y(mask):
    ys = np.nonzero(mask)[0]
    if len(ys) == 0:
        raise SystemExit("empty sole")
    return int(ys.max())


def feet_from(op, y0):
    """Two feet, split on the rows where this still actually gaps.

    A single bottom segment is the longer ground foot.
    """
    leg_l = np.zeros_like(op)
    leg_r = np.zeros_like(op)
    for y in range(y0, op.shape[0]):
        segs = row_segments(op[y])
        if len(segs) >= 2:
            a, b = segs[0]
            leg_l[y, a:b + 1] = True
            a, b = segs[-1]
            leg_r[y, a:b + 1] = True
        elif len(segs) == 1:
            a, b = segs[0]
            leg_r[y, a:b + 1] = True
    return leg_l, leg_r


def feeler_mask(img, box):
    """Curl of cream/tan/olive outside the pink head. The muzzle stays put."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    allow = op & any_color(img, FEELER_COLS)
    allow &= ~any_color(img, BODY_BLOCK)
    # This still's muzzle and belly sit at x32-53 from y32 down. Curls are outside it.
    allow &= ~((xx >= 32) & (xx <= 53) & (yy >= 32))
    x0, x1, y0, y1 = box
    seeds = allow & (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    return flood(allow, seeds)


def absorb_islands(torso, into):
    """Move tiny pieces that only touch `into` onto that part."""
    rest = torso.copy()
    dest = into.copy()
    for g in component_list(torso):
        if len(g) >= 40:
            continue
        mask = np.zeros_like(torso)
        for y, x in g:
            mask[y, x] = True
        if touches(mask, into):
            dest |= mask
            rest &= ~mask
    return rest, dest


def side_hand(op):
    """The one separated hand on the front, x>=53 at y51-53. The back has none."""
    hand = np.zeros_like(op)
    for y in range(51, 54):
        for a, b in row_segments(op[y]):
            if a >= 53:
                hand[y, a:b + 1] = True
    return hand


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    feeler_l = feeler_mask(img, (21, 31, 26, 42))
    feeler_r = feeler_mask(img, (54, 65, 31, 46))
    # Drop the lone grey pixel at y52. It is not 4-connected to the fluff.
    tail = any_color(img, (LAV, GREY)) & (yy >= 53)
    hand = side_hand(op)
    remain = op & ~feeler_l & ~feeler_r & ~tail & ~hand
    leg_l, leg_r = feet_from(remain, 65)
    torso = remain & ~leg_l & ~leg_r
    masks = {
        "feeler_l": feeler_l, "feeler_r": feeler_r, "tail": tail, "hand": hand,
        "leg_l": leg_l, "leg_r": leg_r, "torso": torso,
    }
    check_partition(op, masks)
    eyes = is_color(img, BLUE) | is_color(img, WHITE)
    nose = is_color(img, GOLD) & (yy >= 34) & (yy <= 37) & (xx >= 36) & (xx <= 44)
    if int(eyes.sum()) != 22 or int((eyes & torso).sum()) != 22:
        raise SystemExit("front eyes left the head")
    if int(nose.sum()) != 5:
        raise SystemExit(f"front nose moved: {int(nose.sum())}")
    if float(np.nonzero(nose)[1].mean()) >= float(np.nonzero(eyes)[1].mean()):
        raise SystemExit("front nose is not left of the eyes")
    if (feeler_l & eyes).any() or (feeler_r & eyes).any():
        raise SystemExit("a feeler ate an eye")
    if int(feeler_l.sum()) < 90 or float(np.nonzero(feeler_l)[1].mean()) > 32:
        raise SystemExit("front left feeler looks wrong")
    if int(feeler_r.sum()) < 90 or float(np.nonzero(feeler_r)[1].mean()) < 54:
        raise SystemExit("front right feeler looks wrong")
    if int(tail.sum()) < 20 or float(np.nonzero(tail)[1].mean()) < 50:
        raise SystemExit("front rear fluff is not on the right")
    if int(hand.sum()) < 12:
        raise SystemExit("front hand was not the separated stub")
    if sole_y(leg_r) != FRONT_FOOT or sole_y(leg_l) != FRONT_FOOT - 1:
        raise SystemExit("front feet moved")
    for name, m in masks.items():
        if name == "torso":
            continue
        if not touches(m, torso):
            raise SystemExit(f"{name} does not touch the body")
        if largest_component(m) != int(m.sum()):
            raise SystemExit(f"{name} fell apart")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("front torso fell apart")
    return masks, "leg_r", 1


def segment_back(img):
    op = img[..., 3] >= 128
    # The left curl is split by the pink ear into three pieces. It stays on the head.
    feeler_l = np.zeros_like(op)
    feeler_r = feeler_mask(img, (57, 68, 33, 48))
    tail = any_color(img, (LAV, GREY, PALE))
    hand = np.zeros_like(op)
    remain = op & ~feeler_l & ~feeler_r & ~tail & ~hand
    leg_l, leg_r = feet_from(remain, 69)
    torso = remain & ~leg_l & ~leg_r
    # Ink pixels on the fluff's outer edge are separated once the fluff colour leaves.
    torso, tail = absorb_islands(torso, tail)
    masks = {
        "feeler_l": feeler_l, "feeler_r": feeler_r, "tail": tail, "hand": hand,
        "leg_l": leg_l, "leg_r": leg_r, "torso": torso,
    }
    check_partition(op, masks)
    if any_color(img, (BLUE, WHITE)).any():
        raise SystemExit("back gained eyes")
    if int(feeler_r.sum()) < 80 or float(np.nonzero(feeler_r)[1].mean()) < 55:
        raise SystemExit("back cream curl is not on the right")
    if int(tail.sum()) < 100 or float(np.nonzero(tail)[1].mean()) > 42:
        raise SystemExit("back fluff is not on the left")
    if sole_y(leg_r) != BACK_FOOT or sole_y(leg_l) != BACK_FOOT - 1:
        raise SystemExit("back feet moved")
    for name, m in masks.items():
        if name in ("torso", "hand", "feeler_l") or not m.any():
            continue
        if not touches(m, torso):
            raise SystemExit(f"back {name} does not touch the body")
        if largest_component(m) != int(m.sum()):
            raise SystemExit(f"back {name} fell apart")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("back torso fell apart")
    return masks, "leg_r", 1


def place(arr, y_shift):
    y0 = OFF_Y + y_shift
    x0 = OFF_X
    h, w = arr.shape[:2]
    if y0 < 0 or x0 < 0 or y0 + h > CELL_H or x0 + w > CELL_W:
        raise SystemExit(f"place overflow y {y0} x {x0}")
    if arr.ndim == 2:
        out = np.zeros((CELL_H, CELL_W), bool)
    else:
        out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    out[y0:y0 + h, x0:x0 + w] = arr
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
    """4x nearest in-plane rotation. Positive is clockwise. No squash."""
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
    """Bridge a 1-2px gap with a colour the part already uses."""
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


def joint_band(part, torso, depth):
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 3)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def root_pivot(layer, torso_layer):
    """Top contact with the torso. Feelers and the fluff hang from there."""
    part = layer[..., 3] >= 128
    torso = torso_layer[..., 3] >= 128
    touch = part & dilate(torso, 2)
    if not touch.any():
        touch = part
    ys, xs = np.nonzero(touch)
    band = ys <= int(ys.min()) + 2
    return float(xs[band].mean()), float(ys[band].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return pivot
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


def build_view(img, masks, kind, mirror, ground_leg, sole_gap):
    y_shift = 0 if kind == "front" else (FRONT_FOOT - BACK_FOOT)
    canvas = place(img, y_shift)
    placed = {k: place(v, y_shift) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        if not m.any():
            continue
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    torso_m = placed["torso"]
    for name, depth in (("tail", 5), ("feeler_l", 4), ("feeler_r", 4), ("hand", 3),
                        ("leg_l", 3), ("leg_r", 3)):
        if name not in layers:
            continue
        layers["torso"] = paint_cushion(layers["torso"], joint_band(placed[name], torso_m, depth), depth)
    pivots = {}
    signs = {}
    for name in ROTATE_PARTS:
        if name not in layers:
            continue
        pivots[name] = root_pivot(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
        "ground_leg": ground_leg, "sole_gap": int(sole_gap),
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        feeler_l=0.0, feeler_r=0.0, feeler_lift=0.0,
        tail=0.0, tail_cock=0.0,
        hand=0, kick=0, kick_y=0,
        step_l=(0, 0), step_r=(0, 0),
        feet_follow=False,
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The chest lifts 1px. The two feelers sway apart.
        p["bdy"] = -1 if wave(f, n, 1.0, phase=0.4) > 0.55 else 0
        p["feeler_l"] = wave(f, n, 5.0, phase=0.5)
        p["feeler_r"] = wave(f, n, 5.0, phase=0.5 + math.pi)
        p["tail"] = wave(f, n, 6.0, phase=0.8)
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * max(0.0, sa))), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * max(0.0, sb))), -int(round(3 * max(0.0, sb))))
        p["feeler_l"] = wave(f, n, 7.0, phase=0.3)
        p["feeler_r"] = wave(f, n, 7.0, phase=0.3 + math.pi)
        p["tail"] = wave(f, n, 8.0, phase=math.pi / 2)
        p["feet_follow"] = True
    elif action == "attack":
        # 0 rest. 1-4 raise the feelers and cock the fluff (no mouth).
        # 5-8 the opposite still, equal scale, limbs only slightly open.
        # 9-12 the snout feelers and the tail come forward, the hand leaves
        # the hip, the snout foot kicks. 13-15 recover.
        lift = [0, 6, 12, 16, 18, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0]
        whip = [0, 0, 0, 0, 0, 8, 8, 8, 8, 12, 16, 20, 20, 8, 3, 0]
        cock = [0, 6, 12, 16, 18, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0]
        tail = [0, 0, 0, 0, 0, 12, 12, 12, 12, 18, 26, 34, 34, 12, 4, 0]
        hand = [0, 0, 0, 0, 0, 2, 3, 3, 3, 3, 3, 4, 4, 2, 0, 0]
        kick = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3, 3, 3, 1, 0, 0]
        kick_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, -1, 0, 0, 0]
        bdx = [0, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0]
        bdy = [0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        p["feeler_lift"] = float(lift[f])
        p["feeler_l"] = float(whip[f])
        p["feeler_r"] = float(whip[f])
        p["tail_cock"] = float(cock[f])
        p["tail"] = float(tail[f])
        p["hand"] = int(hand[f])
        p["kick"] = int(kick[f])
        p["kick_y"] = int(kick_y[f])
        p["bdx"] = int(bdx[f])
        p["bdy"] = int(bdy[f])
        p["turned"] = 5 <= f <= 8
        p["spark"] = f - 10 if f in (11, 12) else 0
    else:
        env = [0.0, 0.45, 0.75, 1.0, 0.82, 0.60, 0.40, 0.22, 0.10, 0.03, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["feeler_lift"] = 10.0 * env
        p["tail_cock"] = 12.0 * env
        p["flash"] = f == 1
    return finish(p, action, f, __file__)


def part_angle(rig, name, p):
    sgn = rig["signs"][name]
    if name.startswith("feeler"):
        return sgn["up"] * float(p["feeler_lift"]) + sgn["fwd"] * float(p[name])
    if name == "tail":
        return sgn["fwd"] * (float(p["tail"]) - float(p["tail_cock"]))
    return 0.0


def fore_name(rig):
    # Front faces left, so the snout foot is the left column.
    # Back faces right, so the snout foot is the right column.
    return "leg_l" if rig["kind"] == "front" else "leg_r"


def plant_drop(rig, p):
    """When the long foot is up, drop so the short foot reaches the ground line."""
    gap = int(rig["sole_gap"])
    if gap <= 0:
        return 0
    g = rig["ground_leg"]
    sy = p["step_r"][1] if g == "leg_r" else p["step_l"][1]
    if sy < 0:
        return min(gap, -int(sy))
    return 0


def moved(layer, ang, pivot, dx, dy):
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, pivot)
    return shift(layer, dx, dy)


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    fore = fore_name(rig)
    kick = int(p["kick"])
    kick_y = int(p["kick_y"])
    for name in ("tail", "feeler_l", "feeler_r"):
        if name not in L:
            continue
        ang = part_angle(rig, name, p)
        paste(out, moved(L[name], ang, rig["pivots"][name], bdx, bdy))
    if "hand" in L and int(p["hand"]):
        # The drawn hand sits on the rear hip. Push it out past that hip.
        hdx = bdx + int(round(-fc * p["hand"]))
        hdy = bdy + 1
        paste(out, shift(L["hand"], hdx, hdy))
    elif "hand" in L:
        paste(out, shift(L["hand"], bdx, bdy))
    leg_dy = bdy if p["feet_follow"] else 0
    for name in ("leg_l", "leg_r"):
        sx, sy = p["step_l"] if name == "leg_l" else p["step_r"]
        if name == fore:
            sx += kick
            sy += kick_y
        paste(out, shift(L[name], bdx + int(round(fc * sx)), leg_dy + int(sy)))
    return out


def body_moved(p):
    return (
        abs(float(p["feeler_l"])) >= 0.45 or abs(float(p["feeler_r"])) >= 0.45
        or abs(float(p["feeler_lift"])) >= 0.45
        or abs(float(p["tail"])) >= 0.45 or abs(float(p["tail_cock"])) >= 0.45
        or p["step_l"] != (0, 0) or p["step_r"] != (0, 0)
        or p["bdx"] or p["bdy"] or p["hand"] or p["kick"] or p["kick_y"]
        or p["knock"] or abs(p["lean"]) > 0.4
    )


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
    band = ys >= yb - 1
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def feeler_tip(rig, p, name):
    layer = rig["layers"][name]
    ang = part_angle(rig, name, p)
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    moved_layer = moved(layer, ang, rig["pivots"][name], bdx, bdy)
    return tip_xy(moved_layer, rig["pivots"][name])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (CREAM, PINK, WHITE)
    r = 4 + age * 3
    dots = ((r, 0), (r - 1, -3), (r - 1, 3), (r // 2, -4), (r // 2, 4))
    for i, (dx, dy) in enumerate(dots):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def rig_for(rigs, d, p):
    rig = rigs[d]
    if p["turned"]:
        kind, mirror = VIEW_OF[d]
        rig = rigs[("back" if kind == "front" else "front", not mirror)]
    return rig


def apply_contact(rig, p, action):
    """Copy pose and apply the ground drop for a kick or a walk step."""
    p = dict(p)
    p["step_l"] = tuple(p["step_l"])
    p["step_r"] = tuple(p["step_r"])
    if action == "walk":
        p["bdy"] = int(p["bdy"]) + plant_drop(rig, p)
    elif action == "attack" and rig["ground_leg"] == fore_name(rig) and int(p["kick_y"]) < 0:
        p["bdy"] = int(p["bdy"]) + int(rig["sole_gap"])
        p["feet_follow"] = True
    return p


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    p = apply_contact(rig, p, action)
    if not body_moved(p):
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        fc = rig["facing"]
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            piv = rear_pivot(body, fc)
            body = rotate_nearest(body, -fc * p["lean"], piv)
        body = reconnect(fill_pinholes(despeckle(body)))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        fc = rig["facing"]
        # Snout-side curl: front's left feeler, back's right feeler. Mirror keeps the name.
        name = "feeler_l" if rig["kind"] == "front" else "feeler_r"
        tx, ty = feeler_tip(rig, p, name)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    fore = fore_name(rig)
    for name in ("tail", "feeler_l", "feeler_r", "hand", "leg_l", "leg_r"):
        if name not in rig["layers"]:
            continue
        if name in ROTATE_PARTS:
            layer = moved(rig["layers"][name], part_angle(rig, name, p), rig["pivots"][name], bdx, bdy)
        elif name == "hand":
            extra = int(p["hand"])
            layer = shift(rig["layers"][name], bdx + int(round(-fc * extra)), bdy + (1 if extra else 0))
        else:
            sx, sy = p["step_l"] if name == "leg_l" else p["step_r"]
            if name == fore:
                sx += int(p["kick"])
                sy += int(p["kick_y"])
            leg_dy = bdy if p["feet_follow"] else 0
            layer = shift(rig["layers"][name], bdx + int(round(fc * sx)), leg_dy + int(sy))
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def head_x(img):
    """Front: the blue eyes. Back: the skull band under the ears."""
    eyes = is_color(img, BLUE) | is_color(img, WHITE)
    if int(eyes.sum()) >= 8:
        return float(np.nonzero(eyes)[1].mean())
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0 = int(ys.min())
    band = (ys >= y0 + 8) & (ys <= y0 + 18)
    if not band.any():
        band = ys <= y0 + 16
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


def crop_box(canvas, pad=14):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad - 24)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 20)
    x0 = max(0, int(xs.min()) - pad - 28)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 36)
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
    fm, fg, fgap = segment_front(front)
    bm, bg, bgap = segment_back(back)
    rigs = {}
    for kind, img, masks, gleg, gap in (
        ("front", front, fm, fg, fgap),
        ("back", back, bm, bg, bgap),
    ):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror, gleg, gap)
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
        "feeler_l": (80, 200, 240, 255),
        "feeler_r": (240, 180, 40, 255),
        "tail": (180, 80, 220, 255),
        "hand": (240, 80, 80, 255),
        "leg_l": (40, 210, 80, 255),
        "leg_r": (60, 80, 240, 255),
        "torso": (170, 170, 170, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            if m.any():
                part[m] = cols[name]
        a = img[..., 3] >= 128
        ys, xs = np.nonzero(a)
        y0, y1 = int(ys.min()) - 1, int(ys.max()) + 2
        x0, x1 = int(xs.min()) - 1, int(xs.max()) + 2
        tiles.append(np.repeat(np.repeat(part[y0:y1, x0:x1], 3, 0), 3, 1))
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
    act = OUT_DIR / "actions" / "preview.png"
    if act.is_file():
        dest = SYNC_OUT / "actions"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(act, dest / "preview.png")
        n += 1
    return n


def publish(bounds, sprite_h):
    import gen5_publish

    entry = {
        "dex": 531, "form": "normal", "variant": "audino", "gen": "gen5",
        "dir": "monsterSprite/gen5/0531_audino/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def assert_motion(rigs, palette):
    for d, expect_back in (("S", True), ("W", True), ("N", False), ("E", True)):
        p = pose("attack", 6)
        rig = rig_for(rigs, d, p)
        if expect_back and rig["kind"] != "back":
            raise SystemExit(f"{d} frame 6 did not turn to the back")
        if (not expect_back) and rig["kind"] != "front":
            raise SystemExit(f"{d} frame 6 left the front")
        if p["spin"] != 0:
            raise SystemExit("attack spin was not cleared")
    for f in list(range(0, 5)) + list(range(9, 16)):
        if pose("attack", f)["turned"]:
            raise SystemExit(f"attack {f} should face its own side")
    leans = [pose("hurt", f)["lean"] for f in range(12)]
    if max(leans) > MAX_LEAN + 0.01 or leans[0] != 0 or leans[-1] != 0:
        raise SystemExit(f"hurt lean out of range {leans}")
    tails = [abs(float(pose("attack", f)["tail"])) for f in range(16)]
    if max(tails) > 34.01:
        raise SystemExit(f"strike angle above 34: {max(tails)}")
    for d in ("W", "E", "N", "S"):
        rest = synth(rigs, d, "hurt", 0, palette, effects=False)
        hit = synth(rigs, d, "hurt", 3, palette, effects=False)
        fc = rigs[d]["facing"]
        dx = head_x(hit) - head_x(rest)
        if fc > 0 and dx >= -0.5:
            raise SystemExit(f"{d} hurt leaned the wrong way ({dx:.1f})")
        if fc < 0 and dx <= 0.5:
            raise SystemExit(f"{d} hurt leaned the wrong way ({dx:.1f})")


def measure_frames(rigs, palette):
    """The four gates, over every frame, from memory."""
    pal = {tuple(int(v) for v in c) for c in palette}
    off = set()
    partial = 0
    clip = []
    bodies = []
    for d in DIRS:
        for act in ACTIONS:
            for f in range(FRAMES[act]):
                img = synth(rigs, d, act, f, palette, effects=True)
                a = img[..., 3]
                partial += int(((a != 0) & (a != 255)).sum())
                op = a == 255
                if op.any():
                    used = {tuple(int(v) for v in c) for c in np.unique(img[op][:, :3], axis=0)}
                    off |= used - pal
                if op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
                if act in ("attack", "hurt", "walk"):
                    bodies.append(largest_fraction(img))
    return off, partial, clip, bodies


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
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    assert_motion(rigs, palette)

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
        off, partial, clip, bodies = measure_frames(rigs, palette)
        print("off_palette", len(off), flush=True)
        print("partial_alpha", partial, flush=True)
        print("edge_touching", len(clip), flush=True)
        if clip:
            print("edge", clip[:12], flush=True)
        lost = []
        soles = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    p = apply_contact(rig, p, act)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
                    if act in ("idle", "walk") or (act == "attack" and f not in range(5, 9)):
                        img = synth(rigs, d, act, f, palette, effects=False)
                        ys = np.nonzero(img[..., 3] >= 128)[0]
                        if len(ys) and int(ys.max()) != FOOT_Y:
                            soles.append(f"{act}:{d}:{f}:{int(ys.max())}")
        print("disconnected", lost[:12], "count", len(lost), flush=True)
        print("sole off", soles[:12], "count", len(soles), flush=True)
        if bodies:
            print("body_main_min", round(min(bodies), 3), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(float(p["feeler_l"]), 1), round(float(p["feeler_lift"]), 1),
                        round(float(p["tail"]), 1), p["hand"], p["kick"], p["turned"], p["spin"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        box = crop_box(rigs["S"]["canvas"])
        y0, y1, x0, x1 = box
        keys = [("idle", 0), ("idle", 8), ("walk", 4), ("walk", 12),
                ("attack", 0), ("attack", 4), ("attack", 6), ("attack", 8),
                ("attack", 11), ("attack", 15), ("hurt", 1), ("hurt", 3)]
        for d in ("W", "S", "N"):
            tiles = [synth(rigs, d, act, f, palette)[y0:y1, x0:x1] for act, f in keys]
            z = np.repeat(np.repeat(tile_grid(tiles, 6), 2, 0), 2, 1)
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
    box = crop_box(rigs["S"]["canvas"])
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
    print("rest diff", rest_diff, flush=True)
    print("off_palette", len(off), flush=True)
    print("partial_alpha", partial, flush=True)
    print("edge_touching", len(clip), flush=True)
    report = {
        "version": VERSION, "hash": digest, "rest_diff": rest_diff, "idle0": idle0,
        "off_palette_colors": len(off), "partial_alpha": partial, "edge_touching": clip[:12],
        "edge_n": len(clip), "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "bounds": bounds, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (not any(rest_diff.values()) and not any(idle0.values()) and not off
          and partial == 0 and not clip)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_audino_actions", "version": VERSION, "key": KEY,
        "dex": 531, "variant": "audino", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped (two ear feelers, two feet, rear fluff, one drawn hand, closed muzzle)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (nose x 40.0 < eye x 40.5); back fluff is left of the body",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
