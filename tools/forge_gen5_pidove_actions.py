#!/usr/bin/env python3
"""Pidove normal (0519) 8-direction action sheet from this form's own stills.

Measured on normal/front.png (96x96, bbox x34-65 y32-68, 674 opaque, foot y68,
15 colours, alpha 0/255):
  - Faces screen-left. The orange eye (white glint) sits at x45-49 y37-43,
    mean x 46.7. The pink beak is left of the eye, tip outline near x35-37.
  - One folded grey wing on the far side of the light belly (right of the
    chest, y45-55). A short tail tuft sticks off that wing at x60-65 y47-50.
  - Two pink feet. Ankles split at y61 (x45-47 and x49-52). The near foot
    (beak side) is the left one and reaches the ground at y68.
  - No separate crest feather: the grey cap stays on the head.

normal/back.png (bbox x30-61 y30-67, 733 opaque, foot y67) faces screen-right.
  The beak tip is x56, right of the eye (mean x 50.6). The tail is the left
  fan (x30-40). The right feather edge is the wing. Two feet, ankles at y61.

Directions: front still faces left, so S SE E mirror it (screen-right) and
W SW keep it. Back still faces right, so NE keeps it and N NW mirror it
(away, screen-left). E is screen-right, W is screen-left, N is the back.
Attack 5-8 swaps to that opposite still at equal scale (same mirror flag).
The body is never laid on its side and never squashed by |cos|.

  idle   : feet planted, wing settles, tail wags. Frame 0 is the still.
  walk   : the two drawn feet step (2px). The wing flaps. No new legs.
  attack : 0 still, 1-4 beak open (it has a mouth) plus a ring,
           5-8 the opposite still, 9-12 wing slap and beak peck, 13-15 recover.
  hurt   : 12 frames, lean at most 38 degrees around the rear foot.

Head offsets stay 0. Palette is this still's own colours. Alpha is 0 or 255.
front/back png and gif are never written.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen5_pidove_actions.py [--samples] [--force] [--publish]
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
SRC_DIR = ROOT / "monsterSprite/gen5/0519_pidove/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0519_pidove/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0519_pidove"
VERSION = "pidove_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 160, 168
FOOT_Y = 112
OFF_X = (CELL_W - 96) // 2
FRONT_FOOT = 68
BACK_FOOT = 67
# front.png looks screen-left, back.png looks screen-right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "SW": ("front", False), "W": ("front", False),
    "NE": ("back", False), "N": ("back", True), "NW": ("back", True),
}
MAX_LEAN = 38.0
LIMBS = ("beak", "wing", "tail", "foot_near", "foot_far")
# Max angles this script already draws. finish() may lift weaker frames
# up to these, and must not be given a higher target.
PEAKS = {"beak": 30.0, "wing": 34.0, "tail": 16.0}

INK = (0, 0, 0)
DARK = (41, 41, 41)
DEEP = (32, 32, 41)
GREY_D = (74, 74, 74)
GREY_M = (131, 131, 131)
GREY = (164, 164, 164)
LIGHT = (222, 222, 222)
SILVER = (197, 197, 205)
WHITE = (255, 255, 255)
PINK = (189, 98, 98)
PINK_L = (255, 156, 156)
PINK_D = (106, 49, 57)
GOLD = (255, 189, 41)
GOLD_D = (164, 106, 16)
GOLD_M = (213, 148, 24)
EYE_COLS = (GOLD, GOLD_D, GOLD_M, WHITE)
PINK_COLS = (PINK, PINK_L, PINK_D)
BELLY_COLS = (LIGHT, SILVER)
OUTLINE_KEEP = (INK, DARK, DEEP)

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


def component_list(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    groups = []
    for y0, x0 in zip(*np.nonzero(mask)):
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


def require_piece(name, mask):
    if not mask.any():
        raise SystemExit(f"{name} is empty")
    if largest_component(mask) != int(mask.sum()):
        raise SystemExit(f"{name} fell apart")


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


def land_point(x, y, pivot, ang_deg):
    """Where a source pixel lands under rotate_nearest."""
    ang = ang_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45:
        return float(x), float(y)
    r = math.radians(ang)
    c, s = math.cos(r), math.sin(r)
    rx, ry = x - pivot[0], y - pivot[1]
    return pivot[0] + c * rx - s * ry, pivot[1] + s * rx + c * ry


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


def split_feet(feet):
    """Two ankles on the top rows, toes assigned to the nearer ankle."""
    ys = np.nonzero(feet)[0]
    y_top = int(ys.min())
    top = feet[y_top:y_top + 2].any(axis=0)
    xs = np.nonzero(top)[0]
    segs = []
    s = prev = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x > prev + 1:
            segs.append((s, prev))
            s = x
        prev = x
    segs.append((s, prev))
    if len(segs) != 2:
        raise SystemExit(f"expected two ankles, got {segs}")
    yy, xx = np.mgrid[0:feet.shape[0], 0:feet.shape[1]]
    seed_l = feet & (yy <= y_top + 1) & (xx >= segs[0][0]) & (xx <= segs[0][1])
    seed_r = feet & (yy <= y_top + 1) & (xx >= segs[1][0]) & (xx <= segs[1][1])
    ly, lx = np.nonzero(seed_l)
    ry, rx = np.nonzero(seed_r)
    left = np.zeros_like(feet)
    right = np.zeros_like(feet)
    for y, x in zip(*np.nonzero(feet)):
        dl = ((lx - x) ** 2 + (ly - y) ** 2).min()
        dr = ((rx - x) ** 2 + (ry - y) ** 2).min()
        if dl <= dr:
            left[y, x] = True
        else:
            right[y, x] = True
    return left, right


def absorb_beak_islands(torso, beak):
    """Outline pixels the beak cut off the head stay on the beak."""
    groups = component_list(torso)
    groups.sort(key=len, reverse=True)
    main = np.zeros_like(torso)
    for y, x in groups[0]:
        main[y, x] = True
    for group in groups[1:]:
        island = np.zeros_like(torso)
        for y, x in group:
            island[y, x] = True
        if not (dilate(island, 1) & beak).any():
            ys, xs = np.nonzero(island)
            raise SystemExit(f"torso island is not the beak outline: x{int(xs.min())} y{int(ys.min())}")
        beak = beak | island
    return main, beak


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    pink = any_color(img, PINK_COLS)
    eye = any_color(img, EYE_COLS)
    belly = any_color(img, BELLY_COLS)
    pink_head = pink & (yy <= 50)
    band = dilate(pink_head, 1) & op & (xx <= 44) & (yy <= 44) & ~eye
    beak = flood(band, pink_head)
    feet = op & (yy >= 61)
    foot_l, foot_r = split_feet(feet)
    wing = np.zeros_like(op)
    for y in range(45, 61):
        cols = np.nonzero(belly[y])[0]
        if len(cols) == 0:
            continue
        cut = int(cols.max()) + 1
        wing[y, cut:] = op[y, cut:]
    wing &= ~feet & ~beak
    tail = wing & (xx >= 60) & (yy <= 50)
    wing = wing & ~tail
    used = beak | wing | tail | foot_l | foot_r
    torso = op & ~used
    torso, beak = absorb_beak_islands(torso, beak)
    # Near foot is the beak-side (left) foot. This still faces left.
    masks = {
        "beak": beak, "wing": wing, "tail": tail,
        "foot_near": foot_l, "foot_far": foot_r, "torso": torso,
    }
    check_partition(op, masks)
    tip = int(xx[beak].min())
    ex = float(xx[eye].mean())
    if tip >= ex - 2:
        raise SystemExit(f"front does not face left: beak {tip} eye {ex:.1f}")
    if int((eye & torso).sum()) != int(eye.sum()) or int(eye.sum()) < 15:
        raise SystemExit("front eye left the head")
    for name in LIMBS:
        require_piece(name, masks[name])
        anchor = torso | (wing if name == "tail" else np.zeros_like(torso))
        if not (masks[name] & dilate(anchor, 1)).any():
            raise SystemExit(f"front {name} does not touch the body")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("front torso fell apart")
    if int(np.nonzero(op)[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    if int(np.nonzero(foot_l)[0].max()) != FRONT_FOOT:
        raise SystemExit("front near foot left the ground")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    pink = any_color(img, PINK_COLS)
    eye = any_color(img, EYE_COLS)
    pink_head = pink & (yy <= 40)
    band = dilate(pink_head, 1) & op & (xx >= 52) & (yy <= 39) & ~eye
    beak = flood(band, pink_head)
    feet = op & (yy >= 61)
    foot_l, foot_r = split_feet(feet)
    tail_seed = op & (xx <= 33) & (yy >= 48) & (yy <= 55)
    tail_allow = op & (xx <= 40) & (yy >= 46) & (yy <= 58) & ~feet & ~beak
    tail = flood(tail_allow, tail_seed)
    wing = np.zeros_like(op)
    for y in range(46, 60):
        row = np.nonzero(op[y] & ~feet[y] & ~tail[y] & ~beak[y])[0]
        if len(row) == 0:
            continue
        cut = int(row.max()) - 5
        wing[y, cut:] = op[y, cut:]
    wing &= ~feet & ~tail & ~beak & ~eye
    used = beak | wing | tail | foot_l | foot_r
    torso = op & ~used
    # This still faces right, so the near foot is the right one.
    masks = {
        "beak": beak, "wing": wing, "tail": tail,
        "foot_near": foot_r, "foot_far": foot_l, "torso": torso,
    }
    check_partition(op, masks)
    tip = int(xx[beak].max())
    ex = float(xx[eye].mean())
    if tip <= ex + 2:
        raise SystemExit(f"back does not face right: beak {tip} eye {ex:.1f}")
    if int((eye & torso).sum()) != int(eye.sum()) or int(eye.sum()) < 10:
        raise SystemExit("back eye left the head")
    for name in LIMBS:
        require_piece(name, masks[name])
        anchor = torso | (wing if name == "tail" else np.zeros_like(torso))
        if not (masks[name] & dilate(anchor, 1)).any():
            raise SystemExit(f"back {name} does not touch the body")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("back torso fell apart")
    if int(np.nonzero(op)[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line changed")
    if int(np.nonzero(foot_l | foot_r)[0].max()) != BACK_FOOT:
        raise SystemExit("back feet left the ground")
    return masks


def place(arr, off_y):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    h, w = arr.shape[:2]
    if off_y < 0 or OFF_X < 0 or off_y + h > CELL_H or OFF_X + w > CELL_W:
        raise SystemExit(f"place overflow y {off_y}")
    out[off_y:off_y + h, OFF_X:OFF_X + w] = arr
    return out


def joint_band(part, body, depth):
    touch = part & dilate(body, 1)
    if not touch.any():
        touch = part & dilate(body, 3)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def pivot_of(part, body):
    touch = part & dilate(body, 1)
    if not touch.any():
        touch = part & dilate(body, 4)
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
    off_y = FOOT_Y - src_foot
    canvas = place(img, off_y)
    placed = {k: place(v, off_y) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    wing_m = placed["wing"]
    for name, depth in (("beak", 3), ("wing", 4), ("foot_near", 3), ("foot_far", 3)):
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth
        )
    # The tail root sits on the wing. Cushion stays wing-coloured.
    layers["wing"] = paint_cushion(
        layers["wing"], joint_band(placed["tail"], wing_m | torso_m, 3), 3
    )
    facing = 1.0 if mirror else -1.0
    if kind == "back":
        facing = -facing
    pivots = {}
    signs = {}
    for name in LIMBS:
        body = torso_m | (wing_m if name == "tail" else np.zeros_like(torso_m))
        pivots[name] = pivot_of(placed[name], body)
    for name in ("beak", "wing", "tail"):
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
        beak=0.0, wing=0.0, tail=0.0,
        beak_mode="up", wing_mode="up", tail_mode="fwd",
        step_near=(0, 0), step_far=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0, ring=-1,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        p["wing"] = wave(f, n, 7.0, phase=0.4)
        p["tail"] = wave(f, n, 10.0, phase=1.1)
        p["wing_mode"] = "up"
        p["tail_mode"] = "fwd"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_near"] = (int(round(2 * max(0.0, sa))), -int(round(2 * max(0.0, sa))))
        p["step_far"] = (int(round(2 * max(0.0, sb))), -int(round(2 * max(0.0, sb))))
        p["wing"] = wave(f, n, 12.0, phase=0.2)
        p["tail"] = wave(f, n, 8.0, phase=1.0, cycles=2.0)
        p["wing_mode"] = "up"
        p["tail_mode"] = "fwd"
    elif action == "attack":
        # 0 rest. 1-4 beak opens (the still has a mouth). 5-8 the other side.
        # 9-12 the wing and the beak strike. 11 is the hit. 13-15 recover.
        # Values on 9-14 already sit at or above finish()'s envelope.
        beak = [0, 8, 14, 18, 16, 0, 0, 0, 0, 30, 28, 30, 18, 8, 3, 0]
        wing = [0, 8, 14, 18, 20, 0, 0, 0, 0, 33, 32, 34, 22, 10, 4, 0]
        tail = [0, 4, 8, 12, 12, 0, 0, 0, 0, 16, 15, 16, 12, 6, 2, 0]
        bdx = [0, -1, -1, -1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        p["beak"] = float(beak[f])
        p["wing"] = float(wing[f])
        p["tail"] = float(tail[f])
        p["bdx"] = int(bdx[f])
        p["beak_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["wing_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["tail_mode"] = "fwd"
        p["turned"] = 5 <= f <= 8
        if 2 <= f <= 4:
            p["ring"] = f - 2
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.75, 1.0, 0.90, 0.70, 0.50, 0.32, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["wing"] = 12.0 * env
        p["tail"] = 10.0 * env
        p["wing_mode"] = "up"
        p["tail_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    # 5-8 is the opposite still at equal scale. Drop the strike so the
    # silhouette is not swung, and do not invent a mouth on the turn.
    if action == "attack" and 5 <= f <= 8:
        p["beak"] = 0.0
        p["wing"] = 0.0
        p["tail"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["head_dx"] = 0
        p["head_dy"] = 0
        p["step_near"] = (0, 0)
        p["step_far"] = (0, 0)
        p["spark"] = 0
        p["ring"] = -1
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
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


def part_layer(rig, name, p):
    """One part, with the tail riding on the wing so the tuft stays attached."""
    L = rig["layers"]
    piv = rig["pivots"]
    bdx, bdy = body_shift(rig, p)
    fc = rig["facing"]
    if name == "foot_near":
        sx, sy = p["step_near"]
        return shift(L[name], bdx + int(round(fc * sx)), bdy + int(sy))
    if name == "foot_far":
        sx, sy = p["step_far"]
        return shift(L[name], bdx + int(round(fc * sx)), bdy + int(sy))
    ang = part_angle(rig, name, p[f"{name}_mode"], p[name])
    if name != "tail":
        return moved(L[name], ang, piv[name], bdx, bdy)
    wing_ang = part_angle(rig, "wing", p["wing_mode"], p["wing"])
    layer = moved(L["tail"], wing_ang, piv["wing"], bdx, bdy)
    if abs(ang) < 0.45:
        return layer
    tx, ty = land_point(piv["tail"][0], piv["tail"][1], piv["wing"], wing_ang)
    return moved(layer, ang, (tx + bdx, ty + bdy), 0, 0)


def compose(rig, p):
    out = np.zeros_like(rig["canvas"])
    bdx, bdy = body_shift(rig, p)
    paste(out, shift(rig["layers"]["torso"], bdx, bdy))
    for name in ("foot_far", "foot_near", "wing", "tail", "beak"):
        paste(out, part_layer(rig, name, p))
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = any_color(out, OUTLINE_KEEP)
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
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


def roar(img, origin, fc, age):
    """Mouth wedge and a ring, drawn in this still's own colours."""
    ox, oy = origin
    cols = (INK, PINK, GREY, WHITE, GOLD)
    # The wedge shares an edge with the beak tip.
    for i, (dx, dy) in enumerate(((1, 0), (2, -1), (2, 1), (3, 0), (1, -1))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])
    rad = 7 + age * 4
    prev = None
    for deg in range(-50, 51, 6):
        a = math.radians(deg)
        x = ox + fc * (4 + rad * math.cos(a))
        y = oy + rad * math.sin(a) * 0.72
        put(img, x, y, cols[(deg // 6) % 5])
        if prev is not None:
            put(img, (x + prev[0]) / 2, (y + prev[1]) / 2, cols[age % 5])
        prev = (x, y)


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, GOLD, PINK_L, LIGHT)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r), (2, r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name in LIMBS:
        layer = part_layer(rig, name, p)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    kind, mirror = VIEW_OF[d]
    if p["turned"]:
        kind = "back" if kind == "front" else "front"
    return rigs[(kind, mirror)]


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    for key in ("beak", "wing", "tail"):
        if abs(p[key]) >= 0.45:
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
        # Nearest-neighbour lifts the rear toe by a pixel or two.
        # Put the lowest pixel back on the ground line.
        grounded = body[..., 3] >= 128
        if grounded.any():
            bot = int(np.nonzero(grounded)[0].max())
            body = shift(body, 0, FOOT_Y - bot)
    if rotated(p):
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["ring"] >= 0 and not p["turned"]:
        tip = facing_tip(part_layer(rig, "beak", p), fc)
        if tip is not None:
            roar(img, tip, fc, p["ring"])
            img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tip = facing_tip(part_layer(rig, "wing", p), fc)
        beak = facing_tip(part_layer(rig, "beak", p), fc)
        if tip is not None:
            impact(img, (tip[0] + fc * 2, tip[1]), fc, p["spark"])
        if beak is not None:
            impact(img, (beak[0] + fc * 2, beak[1]), fc, p["spark"])
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
    y0 = max(0, int(ys.min()) - pad - 22)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 20)
    x0 = max(0, int(xs.min()) - pad - 36)
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
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {}
    for kind, img, masks, foot in (
        ("front", front, fm, FRONT_FOOT),
        ("back", back, bm, BACK_FOOT),
    ):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror, foot)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "beak": (220, 40, 40, 255),
        "wing": (40, 140, 255, 255),
        "tail": (240, 200, 40, 255),
        "foot_near": (255, 140, 40, 255),
        "foot_far": (180, 70, 220, 255),
        "torso": (170, 170, 170, 255),
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
        "dex": 519, "form": "normal", "variant": "pidove", "gen": "gen5",
        "dir": "monsterSprite/gen5/0519_pidove/normal", "sheetDir": "agent-sprite-forge",
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

    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
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
            atk.append((f, round(p["beak"], 1), round(p["wing"], 1), round(p["tail"], 1),
                        p["beak_mode"], p["wing_mode"], p["turned"], p["spin"], p["bdx"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
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
        "generator": "forge_gen5_pidove_actions", "version": VERSION, "key": KEY,
        "dex": 519, "variant": "pidove", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "small pigeon (folded wing, short tail tuft, two feet, pink beak)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (beak x < eye x); back beak is screen-right",
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
