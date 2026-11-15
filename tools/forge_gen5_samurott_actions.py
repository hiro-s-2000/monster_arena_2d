#!/usr/bin/env python3
"""Samurott normal (0503) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x9-86 y8-87, 2907 px, alpha 0/255,
14 colours):
  Red eye is x26-27 y28-29 (mean x 26.7). Red nose tip is x16-17 y29-30
  (min x 16), left of the eye, so the still faces screen-left. There is no
  mouth slit and no mouth cavity, so the roar does not invent a wedge or rings.
  It raises the long horn and the white whiskers that are already drawn.
  Long horn: pale/cream/gold x11-24 y8-17, one spike on the helmet.
  Whiskers: three white tufts (blade x19-60 y22-46, brow, snout tuft).
  Near leg (screen-left) x9-35 y54-84, claw ends y84.
  Far leg reaches the ground at x60-63 y87, with the outer shin beside it.
  Rear fin (tail) is the separate right run x54-86 y39-53.
normal/back.png (bbox x6-88 y9-86, 3209 px, 15 colours) has the head on the
  right: the eye is the red/maroon pixel at x64 y26-27. The long shell horn
  is x62-76 y9-20 and the short horn is x40-56 y12-21. The white mane sits
  on that snout side. The tail fin is the separated left run. The lowest
  pixel is y86, so the back is shifted down 1px onto the front sole.
  The extra back colour (115, 57, 57) stays in the palette.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (snout side screen-right). NW back mirrored.
  E is screen-right, W is screen-left, N is the back. Attack 5-8 swaps to
  the matching back at equal scale. No width squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_samurott_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0503_samurott/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0503_samurott/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0503_samurott"
VERSION = "samurott_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 240, 220
OFF_X = (CELL_W - 96) // 2
OFF_Y = 46
FRONT_FOOT = 87
BACK_FOOT = 86
FOOT_Y = OFF_Y + FRONT_FOOT
# kind, mirror. Front as drawn looks screen-left. Back as drawn looks screen-right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Wind-up peaks finish() may raise frames 5-12 toward. The horn strike
# itself is drawn at 34 deg, above this peak. Frames 5-8 are zeroed after
# finish() so the turned silhouette is not swung.
PEAKS = {"horn": 18.0, "whisker": 12.0, "tail": 10.0}
LIMBS = ("tail", "leg_l", "leg_r", "horn", "horn2", "whisk_a", "whisk_b", "whisk_c")

BLUE = (33, 90, 156)
INK = (16, 16, 16)
NAVY = (33, 57, 99)
CREAM = (222, 206, 123)
WHITE = (255, 255, 255)
BROWN = (123, 99, 66)
TEAL = (132, 165, 165)
GRAY = (74, 74, 74)
GOLD = (173, 148, 90)
PALE = (255, 239, 148)
SLATE = (90, 115, 115)
SILVER = (198, 198, 206)
DARK = (16, 33, 74)
RED = (214, 82, 99)
MAROON = (115, 57, 57)
GOLD_COLS = (CREAM, BROWN, GOLD, PALE)
BODY_COLS = (BLUE, NAVY, SLATE, TEAL, DARK, RED, MAROON) + GOLD_COLS

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
    while True:
        nxt = dilate(out, 1) & allowed
        if np.array_equal(nxt, out):
            return out
        out = nxt


def components(mask):
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
    groups.sort(key=len, reverse=True)
    return groups


def largest(mask):
    gs = components(mask)
    out = np.zeros_like(mask)
    if not gs:
        return out
    for y, x in gs[0]:
        out[y, x] = True
    return out


def mask_from(pts):
    m = np.zeros((96, 96), bool)
    for y, x in pts:
        m[y, x] = True
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


def require_piece(name, mask):
    if not mask.any():
        return
    if len(components(mask)[0]) != int(mask.sum()):
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
        for name, src_slice in (
            ("up", src[1:]),
            ("down", src[:-1]),
            ("left", src[:, 1:]),
            ("right", src[:, :-1]),
        ):
            grown = np.zeros_like(src)
            if name == "up":
                grown[:-1] = src_slice
            elif name == "down":
                grown[1:] = src_slice
            elif name == "left":
                grown[:, :-1] = src_slice
            else:
                grown[:, 1:] = src_slice
            take = (acc[..., 3] < 128) & allowed & (grown[..., 3] >= 128)
            acc[take] = grown[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def outer_runs(img, side, y0, y1, limit):
    """Outer run on rows where a gap separates it from the rest of the body."""
    op = img[..., 3] >= 128
    m = np.zeros_like(op)
    for y in range(y0, y1):
        xs = np.nonzero(op[y])[0]
        if len(xs) == 0:
            continue
        if side == "left":
            end = int(xs[0])
            for x in xs[1:]:
                if int(x) > end + 1:
                    break
                end = int(x)
            if end > limit or int(xs[-1]) <= end:
                continue
            m[y, int(xs[0]):end + 1] = True
        else:
            end = int(xs[-1])
            for x in xs[-2::-1]:
                if int(x) < end - 1:
                    break
                end = int(x)
            if end < limit or int(xs[0]) >= end:
                continue
            m[y, end:int(xs[-1]) + 1] = True
    return m


def whisker_parts(img, core, banned, y_max):
    """Split the mane into tufts and keep each tuft's own outer ink."""
    bodyish = any_color(img, BODY_COLS) | banned
    parts = []
    claimed = banned.copy()
    for pts in components(core):
        if len(pts) < 20:
            continue
        comp = mask_from(pts)
        ink = is_color(img, INK) & dilate(comp, 1) & (np.mgrid[0:96, 0:96][0] <= y_max)
        ink &= ~dilate(bodyish, 1) & ~claimed & ~comp
        comp = comp | ink
        claimed |= comp
        parts.append(comp)
    parts.sort(key=lambda m: int(m.sum()), reverse=True)
    return parts[:3]


def blank_parts():
    z = np.zeros((96, 96), bool)
    return {name: z.copy() for name in LIMBS}


def absorb_near(dst, zone, reach):
    """Pull zone pixels that sit within `reach` of dst into dst."""
    if not dst.any() or not zone.any():
        return dst
    seed = dilate(dst, reach) & zone
    if not seed.any():
        return dst
    allow = zone | (dilate(dst, 1) & zone)
    return dst | flood(allow, seed)


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    parts = blank_parts()
    horn = largest(flood(
        any_color(img, GOLD_COLS + (INK,)) & (yy <= 17) & (xx <= 24),
        any_color(img, (PALE, CREAM)) & (yy <= 12) & (xx <= 16),
    ))
    parts["horn"] = horn
    core = flood(
        any_color(img, (WHITE, SILVER, GRAY)) & (yy <= 50) & ~horn,
        any_color(img, (WHITE, SILVER)) & ~horn,
    )
    tufts = whisker_parts(img, core & ~horn, horn, 50)
    for name, tuft in zip(("whisk_a", "whisk_b", "whisk_c"), tufts):
        parts[name] = tuft & ~horn
    whisk = parts["whisk_a"] | parts["whisk_b"] | parts["whisk_c"]
    tail_seed = outer_runs(img, "right", 36, 54, 54)
    tail = largest(flood(op & (xx >= 54) & (yy >= 36) & (yy <= 53) & ~horn & ~whisk, tail_seed))
    parts["tail"] = tail
    allow_l = op & (
        ((xx <= 35) & (yy >= 70))
        | ((xx <= 26) & (yy >= 54) & any_color(img, GOLD_COLS + (NAVY, INK, GRAY, BLUE)))
    )
    allow_l &= ~horn & ~whisk & ~tail
    leg_l = largest(flood(allow_l, op & (yy >= 80) & (xx <= 22)))
    leg_l = leg_l | (dilate(leg_l, 1) & allow_l)
    parts["leg_l"] = leg_l
    zone = op & (yy >= 73) & (xx >= 36) & ~leg_l & ~tail & ~whisk & ~horn
    leg_r = largest(flood(zone, op & (yy >= 84) & (xx >= 48)))
    shin_allow = any_color(img, GOLD_COLS + (INK, NAVY, GRAY)) & (xx >= 52) & (yy >= 58) & (yy <= 80)
    shin_allow &= ~leg_l & ~tail & ~whisk & ~horn
    leg_r = leg_r | flood(shin_allow, dilate(leg_r, 1) & shin_allow)
    outer = op & (xx >= 60) & (yy >= 64) & (yy <= 85) & ~leg_r & ~leg_l & ~tail & ~whisk & ~horn
    parts["leg_r"] = absorb_near(leg_r, outer, 6)
    used = np.zeros_like(op)
    for name in LIMBS:
        used |= parts[name]
    torso = op & ~used
    masks = dict(parts)
    masks["torso"] = torso
    check_partition(op, masks)
    for name in LIMBS:
        require_piece(name, masks[name])
        if masks[name].any() and not (masks[name] & dilate(torso, 2)).any():
            raise SystemExit(f"front {name} does not touch the body")
    eye = is_color(img, RED) & (xx >= 24) & (yy <= 32)
    nose = is_color(img, RED) & (xx <= 20)
    if int((eye & torso).sum()) != int(eye.sum()) or int(eye.sum()) < 3:
        raise SystemExit("eyes left the head")
    if int((nose & torso).sum()) != int(nose.sum()) or int(nose.sum()) < 3:
        raise SystemExit("nose left the head")
    tip = int(np.nonzero(nose)[1].min())
    ex = float(np.nonzero(eye)[1].mean())
    if not tip < ex - 1:
        raise SystemExit(f"front does not face left: nose {tip} eye {ex:.1f}")
    if int(np.nonzero(op)[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    if int(horn.sum()) < 40 or int(parts["leg_l"].sum()) < 150 or int(parts["leg_r"].sum()) < 150:
        raise SystemExit("front horn or legs shrank")
    if int(tail.sum()) < 80:
        raise SystemExit("front tail shrank")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    parts = blank_parts()
    horn = largest(flood(
        any_color(img, GOLD_COLS + (INK,)) & (yy <= 20) & (xx >= 62),
        any_color(img, (PALE, CREAM)) & (yy <= 14) & (xx >= 68),
    ))
    horn2 = largest(flood(
        any_color(img, GOLD_COLS + (INK,)) & (yy <= 21) & (xx >= 38) & (xx <= 56) & ~horn,
        any_color(img, (BROWN, CREAM, PALE)) & (yy <= 16) & (xx >= 42) & (xx <= 50),
    ))
    parts["horn"] = horn
    parts["horn2"] = horn2 & ~horn
    banned = horn | horn2
    core = flood(
        any_color(img, (WHITE, SILVER, GRAY)) & (yy <= 48) & (xx >= 50) & ~banned,
        any_color(img, (WHITE, SILVER)) & ~banned,
    )
    tufts = whisker_parts(img, core & ~banned, banned, 48)
    for name, tuft in zip(("whisk_a", "whisk_b", "whisk_c"), tufts):
        parts[name] = tuft & ~banned
    whisk = parts["whisk_a"] | parts["whisk_b"] | parts["whisk_c"]
    tail_seed = outer_runs(img, "left", 22, 50, 30)
    tail = largest(flood(
        op & (xx <= 30) & (yy >= 22) & (yy <= 49) & ~banned & ~whisk,
        tail_seed,
    ))
    parts["tail"] = tail
    leg_l = largest(flood(
        op & (xx <= 32) & (yy >= 68) & ~tail & ~whisk & ~banned,
        op & (yy >= 76) & (xx <= 18),
    ))
    parts["leg_l"] = leg_l
    br_zone = op & (yy >= 74) & (xx >= 40) & ~leg_l & ~tail & ~whisk & ~banned
    leg_r = largest(flood(br_zone, op & (yy >= 83)))
    outer = op & (xx >= 58) & (yy >= 66) & (yy <= 82) & ~leg_r & ~leg_l & ~tail & ~whisk & ~banned
    parts["leg_r"] = absorb_near(leg_r, outer, 6)
    used = np.zeros_like(op)
    for name in LIMBS:
        used |= parts[name]
    torso = op & ~used
    masks = dict(parts)
    masks["torso"] = torso
    check_partition(op, masks)
    for name in ("horn", "horn2", "whisk_a", "tail", "leg_l", "leg_r"):
        require_piece(name, masks[name])
        if masks[name].any() and not (masks[name] & dilate(torso, 2)).any():
            raise SystemExit(f"back {name} does not touch the body")
    eye = (is_color(img, RED) | is_color(img, MAROON)) & (yy <= 32)
    if int((eye & torso).sum()) != int(eye.sum()) or not eye.any():
        raise SystemExit("back eye left the head")
    if float(np.nonzero(eye)[1].mean()) < 55:
        raise SystemExit("back head is not on the right")
    if int(np.nonzero(op)[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line changed")
    if int(horn.sum()) < 40 or int(horn2.sum()) < 40 or int(leg_r.sum()) < 80:
        raise SystemExit("back horns or foot shrank")
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
    depth = {
        "tail": 5, "leg_l": 6, "leg_r": 6, "horn": 6, "horn2": 5,
        "whisk_a": 5, "whisk_b": 4, "whisk_c": 4,
    }
    for name in LIMBS:
        if not placed[name].any():
            continue
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth[name]), depth[name]
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in LIMBS:
        if not placed[name].any():
            continue
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    present = [n for n in ("leg_l", "leg_r") if placed[n].any()]
    foot_cx = {n: float(np.nonzero(placed[n])[1].mean()) for n in present}
    if not present:
        near = "leg_r"
    elif facing > 0:
        near = max(present, key=lambda n: foot_cx[n])
    else:
        near = min(present, key=lambda n: foot_cx[n])
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas, "near_foot": near,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, horn=0.0, whisker=0.0, tail=0.0,
        horn_mode="up", whisker_mode="up", tail_mode="up",
        thrust=0, step_l=(0, 0), step_r=(0, 0), kick=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0, turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. Horn, whiskers, and tail breathe.
        p["horn"] = wave(f, n, 5.0, phase=0.4)
        p["whisker"] = wave(f, n, 4.0, phase=0.9)
        p["tail"] = wave(f, n, 7.0, phase=0.2)
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        p["step_l"] = (int(round(2 * max(0.0, sa))), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * max(0.0, -sa))), -int(round(3 * max(0.0, -sa))))
        p["horn"] = wave(f, n, 3.0, phase=0.5)
        p["whisker"] = wave(f, n, 3.0, phase=0.3)
        p["tail"] = wave(f, n, 8.0, phase=math.pi / 2)
    elif action == "attack":
        # 0 rest. 1-4 raise the horn and whiskers (no mouth). 5-8 back view.
        # 9-12 horn thrust. 13-15 recover. 15 is the still.
        horn = [0, 8, 14, 18, 18, 0, 0, 0, 0, 24, 30, 34, 34, 14, 6, 0]
        whisker = [0, 6, 10, 14, 16, 0, 0, 0, 0, 12, 14, 16, 14, 8, 3, 0]
        tail = [0, 4, 8, 12, 12, 0, 0, 0, 0, 10, 12, 12, 10, 4, 2, 0]
        bdx = [0, -1, -1, -1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 1, 0, 0]
        thrust = [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0]
        p["horn"] = float(horn[f])
        p["whisker"] = float(whisker[f])
        p["tail"] = float(tail[f])
        p["bdx"] = int(bdx[f])
        p["thrust"] = int(thrust[f])
        p["horn_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["whisker_mode"] = "up"
        p["tail_mode"] = "up"
        p["turned"] = 5 <= f <= 8
        if f in (10, 11, 12):
            p["kick"] = {10: (1, -1), 11: (2, -1), 12: (1, 0)}[f]
        if f in (11, 12):
            p["spark"] = f - 10
        if f in (3, 4):
            p["bdy"] = -1
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["horn"] = 6.0 * env
        p["whisker"] = 5.0 * env
        p["tail"] = 8.0 * env
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    # Frames 5-8 are the other side at equal scale, the same still as that
    # direction's back or front row. finish() raises the wind-up on these
    # frames; drop it so the turned silhouette is not swung.
    if action == "attack" and 5 <= f <= 8:
        p["horn"] = 0.0
        p["whisker"] = 0.0
        p["tail"] = 0.0
        p["thrust"] = 0
        p["bdx"] = 0
        p["bdy"] = 0
        p["kick"] = (0, 0)
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


def limb_layers(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    thrust = int(p["thrust"])
    parts = {}
    for name in LIMBS:
        if name not in piv:
            continue
        if name == "leg_l":
            sx, sy = p["step_l"]
            ang = 0.0
            extra = 0
        elif name == "leg_r":
            sx, sy = p["step_r"]
            ang = 0.0
            extra = 0
        elif name in ("horn", "horn2"):
            sx, sy = 0, 0
            ang = part_angle(rig, name, p["horn_mode"], p["horn"])
            extra = thrust
        elif name.startswith("whisk"):
            sx, sy = 0, 0
            ang = part_angle(rig, name, p["whisker_mode"], p["whisker"])
            extra = 0
        else:
            sx, sy = 0, 0
            ang = part_angle(rig, name, p["tail_mode"], p["tail"])
            extra = 0
        if name == rig["near_foot"]:
            kx, ky = p["kick"]
            sx += kx
            sy += ky
        dx = bdx + int(round(fc * (sx + extra)))
        dy = bdy + int(sy)
        parts[name] = moved(L[name], ang, piv[name], dx, dy)
    return parts


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    out = np.zeros_like(rig["canvas"])
    # Torso first, including the joint cushion. Parts cover it at rest.
    paste(out, shift(L["torso"], bdx, bdy))
    parts = limb_layers(rig, p)
    if "tail" in parts:
        paste(out, parts["tail"])
    far = "leg_r" if rig["near_foot"] == "leg_l" else "leg_l"
    for name in (far, rig["near_foot"]):
        if name in parts:
            paste(out, parts[name])
    for name in ("horn", "horn2", "whisk_a", "whisk_b", "whisk_c"):
        if name in parts:
            paste(out, parts[name])
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK) | is_color(out, DARK)
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
    parts = limb_layers(rig, p)
    fc = rig["facing"]
    name = "horn" if "horn" in parts and np.any(parts["horn"][..., 3] >= 128) else None
    if name is None:
        return rig["pivots"]["center"]
    layer = parts[name]
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (PALE, WHITE, RED)
    r = 4 + age * 3
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2), (2, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])
        put(img, ox + fc * (dx + 1), oy + dy, cols[(i + 1) % 3])


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name, layer in limb_layers(rig, p).items():
        limb = layer[..., 3] >= 128
        if not limb.any():
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    return rigs[("back" if kind == "front" else "front", not mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    moved_part = (
        abs(p["horn"]) >= 0.45 or abs(p["whisker"]) >= 0.45 or abs(p["tail"]) >= 0.45
    )
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        # Rear foot stays down. Positive rotation is clockwise, so the lean
        # is the opposite of facing and the belly is not flipped.
        body = rotate_nearest(body, -fc * p["lean"], piv)
    if moved_part or abs(p["lean"]) > 0.4:
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = weapon_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
    return snap(img, palette)


def largest_component(mask):
    gs = components(mask)
    if not gs:
        return 0
    return len(gs[0])


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
    y0 = max(0, int(ys.min()) - pad - 28)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 16)
    x0 = max(0, int(xs.min()) - pad - 36)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 36)
    return y0, y1, x0, x1


def preview_image(sheets):
    pick = {"idle": 0, "walk": 4, "attack": 11, "hurt": 3}
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


def still_colors(*imgs):
    cols = []
    seen = set()
    for img in imgs:
        for c in np.unique(img[img[..., 3] >= 128][:, :3], axis=0):
            t = tuple(int(v) for v in c)
            if t not in seen:
                seen.add(t)
                cols.append(t)
    return cols


def locked_palette(front_path, back_path, work):
    front = np.array(Image.open(front_path).convert("RGBA"))
    back = np.array(Image.open(back_path).convert("RGBA"))
    own = still_colors(front, back)
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
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
        # Keep every still colour. The lock is built from the front, and the
        # back has one extra maroon pixel.
        missing = [c for c in own if c not in got]
        extra = [c for c in got if c not in want]
        if extra or not missing:
            raise SystemExit(f"locked palette differs from the stills: {sorted(got ^ want)}")
        cols = np.concatenate([cols, np.array(missing, np.uint8)], axis=0)
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
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "horn": (255, 50, 50, 255), "horn2": (255, 160, 40, 255),
        "whisk_a": (255, 220, 40, 255), "whisk_b": (80, 180, 255, 255),
        "whisk_c": (200, 80, 255, 255), "tail": (40, 210, 80, 255),
        "leg_l": (40, 120, 255, 255), "leg_r": (255, 120, 20, 255),
        "torso": (70, 70, 78, 255),
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
        "dex": 503, "form": "normal", "variant": "samurott", "gen": "gen5",
        "dir": "monsterSprite/gen5/0503_samurott/normal", "sheetDir": "agent-sprite-forge",
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
    print("facing", facing, "near", {d: rigs[d]["near_foot"] for d in ("S", "W", "N", "E")}, flush=True)

    rest_diff = {}
    for d in DIRS:
        r = synth(rigs, d, "idle", 0, palette, effects=False)
        rest_diff[d] = int(np.any(r != rigs[d]["canvas"], axis=2).sum())
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        y0, y1, x0, x1 = box
        for d in ("S", "W", "N", "E"):
            frames = []
            for act, fs in (
                ("idle", (0, 4, 8, 12)),
                ("walk", (0, 2, 4, 6, 8, 10, 12, 14)),
                ("attack", (0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15)),
                ("hurt", (0, 1, 3, 6, 11)),
            ):
                for f in fs:
                    frames.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
            sheet = tile_grid(frames, len(frames))
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
            atk.append((f, round(p["horn"], 1), p["horn_mode"], p["turned"], p["spin"], p["bdx"]))
        print("attack poses", atk, flush=True)
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
    y0, y1, x0, x1 = box
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.concatenate(strips, axis=0)
    motion_z = np.repeat(np.repeat(motion, 2, 0), 2, 1)
    Image.fromarray(motion_z, "RGBA").save(OUT_DIR / "preview_motion.png")
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
                    bare = synth(rigs, d, act, f, palette, effects=False)
                    bodies.append(largest_fraction(bare))
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
        "generator": "forge_gen5_samurott_actions", "version": VERSION, "key": KEY,
        "dex": 503, "variant": "samurott", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped horn striker (helmet horn, whiskers, rear fin, two legs)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (nose x < eye x); no mouth cavity",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit("qa failed; not publishing")
    if args.publish:
        print("synced", sync_outputs(), flush=True)
        publish(bounds, sprite_h)
        print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
