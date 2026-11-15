#!/usr/bin/env python3
"""Mega Scolipede (0545) 8-direction actions from this form's own stills.

Measured on mega/scolipede-mega/front.png (96x96, bbox x16-79 y0-95, 2329 px,
20 colours, alpha 0/255, one connected piece, foot y95):
  The teal horn tip is the single pixel at x25 y0. The eye is the navy pupil
  (32,41,49) at x30-33 y42-43, mean x 31.5, with the gold glint at x35-37 and
  the white glint at x39-40 on the same row. The horn tip is left of the eye,
  so the still faces screen-left. There is no drawn mouth: the blood pixels
  sit in the lower shell (y68-86), and the roar does not invent a wedge or rings.
  Horn: every opaque pixel at y<=37 (x18-44), one piece, rooted on the head.
  Rear claw: the separated right blade, flood of x>=65 y<=65 from the tip at
  x78-79 (76 px, x65-79 y56-65). It is not the belly.
  Four ventral prongs, the parts that split at y>=86, left to right:
  x24-31 y86-91, x39-46 y86-95 (the ground spike), x51-56 y86-91, x64-68 y86-92.
mega/scolipede-mega/back.png (bbox x17-79 y0-95, 2348 px, 17 colours, foot y95)
  has the horn tip at x70 y0, right of the body, and the rear claw on the left
  (x17-24 y54-62). This back faces screen-right. No pupil is drawn. Same part
  names: horn y<=32, left claw, four prongs at y>=86. Both feet are y95, so
  the back is not shifted. Back colours are a subset of the front palette.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (horn screen-right). NW back mirrored.
  E is screen-right, W is screen-left, N is the back. Attack 5-8 swaps to the
  matching opposite still at equal scale. No width squash, no in-plane spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_scolipede_mega_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0545_scolipede/mega/scolipede-mega"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0545_scolipede/mega/scolipede-mega/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0545_scolipede-mega"
VERSION = "scolipede-mega_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 256, 192
OFF_X = (CELL_W - 96) // 2
OFF_Y = 48
FRONT_FOOT = 95
BACK_FOOT = 95
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left (horn tip x < eye x). Back as drawn has the
# horn on the screen-right. S (front mirrored) turns into N (back as drawn).
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Roar / turn hold. The horn thrust is drawn at 34 deg, above these peaks.
PEAKS = {
    "horn": 22.0, "scythe": 16.0,
    "leg0": 8.0, "leg1": 8.0, "leg2": 8.0, "leg3": 8.0,
}
LIMBS = ("horn", "scythe", "leg0", "leg1", "leg2", "leg3")
LEG_NAMES = ("leg0", "leg1", "leg2", "leg3")

INK = (16, 16, 16)
PALE = (213, 187, 212)
MAGENTA = (205, 84, 170)
MAUVE = (144, 115, 139)
GREY = (80, 82, 76)
CRIMSON = (171, 37, 68)
DK = (50, 52, 47)
TEAL = (0, 224, 174)
PINK = (255, 100, 213)
PURPLE = (81, 65, 87)
MINT = (183, 255, 203)
DEEP = (121, 39, 117)
ROSE = (231, 74, 102)
TEAL_D = (0, 169, 144)
TEAL_B = (38, 232, 176)
PLUM = (117, 79, 103)
BLOOD = (98, 16, 16)
NAVY = (32, 41, 49)
GOLD = (255, 171, 39)
WHITE = (255, 229, 254)
IRIS = NAVY

SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)


def is_color(img, color):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(color, np.uint8), axis=-1)


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


def _rotate_crop(layer, ang, pivot):
    """4x nearest rotation of one crop. Positive follows the shared forge formula."""
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
    """Same rotation as the full-cell forge formula, cropped to the part. No squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    dx = max(abs(float(xs.min()) - pivot[0]), abs(float(xs.max()) - pivot[0]))
    dy = max(abs(float(ys.min()) - pivot[1]), abs(float(ys.max()) - pivot[1]))
    rad = int(math.ceil(math.hypot(dx, dy))) + 3
    x0 = max(0, int(math.floor(pivot[0] - rad)))
    y0 = max(0, int(math.floor(pivot[1] - rad)))
    x1 = min(layer.shape[1], int(math.ceil(pivot[0] + rad)) + 1)
    y1 = min(layer.shape[0], int(math.ceil(pivot[1] + rad)) + 1)
    crop = layer[y0:y1, x0:x1]
    turned = _rotate_crop(crop, ang, (pivot[0] - x0, pivot[1] - y0))
    out = np.zeros_like(layer)
    out[y0:y1, x0:x1] = turned
    return out


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


def mask_from(pts, shape):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m


def touches(part, body):
    return bool((part & dilate(body, 1)).any())


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()), int(mask.sum())


def segment(img, kind):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    if kind == "front":
        horn = op & (yy <= 37)
        seed = op & (xx >= 78) & (yy >= 56) & (yy <= 62)
        allowed = op & (yy <= 65) & (xx >= 65)
    else:
        horn = op & (yy <= 32)
        seed = op & (xx <= 19) & (yy >= 54) & (yy <= 60)
        allowed = op & (yy <= 62) & (xx <= 24) & (yy >= 50)
    if not (seed & allowed).any():
        raise SystemExit(f"{kind} claw seed missed the still")
    scythe = flood(allowed & ~horn, seed)
    leg_zone = op & (yy >= 86) & ~horn & ~scythe
    legs = []
    for pts in component_list(leg_zone):
        if len(pts) < 12:
            continue
        xs = [p[1] for p in pts]
        legs.append((sum(xs) / len(xs), mask_from(pts, op.shape)))
    legs.sort(key=lambda t: t[0])
    if len(legs) != 4:
        raise SystemExit(f"{kind} prongs {len(legs)}")
    masks = {"horn": horn, "scythe": scythe}
    for i, (_, m) in enumerate(legs):
        masks[f"leg{i}"] = m
    torso = op.copy()
    for m in masks.values():
        torso &= ~m
    masks["torso"] = torso
    check_partition(op, masks)
    expect = {
        "front": {"horn": 444, "scythe": 76, "leg0": 29, "leg1": 42, "leg2": 23, "leg3": 27},
        "back": {"horn": 383, "scythe": 39, "leg0": 30, "leg1": 49, "leg2": 17, "leg3": 34},
    }[kind]
    for name, n in expect.items():
        if int(masks[name].sum()) != n:
            raise SystemExit(f"{kind} {name} {int(masks[name].sum())} != {n}")
    for name in LIMBS:
        if not touches(masks[name], torso):
            raise SystemExit(f"{kind} {name} does not touch the body")
        if largest_component(masks[name]) != int(masks[name].sum()):
            raise SystemExit(f"{kind} {name} fell apart")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit(f"{kind} torso fell apart")
    top = int(np.nonzero(horn)[0].min())
    tip_x = float(np.nonzero(horn[top])[0].mean())
    if kind == "front":
        iris = is_color(img, IRIS)
        if int((iris & torso).sum()) != int(iris.sum()) or int(iris.sum()) != 4:
            raise SystemExit("front eye left the head")
        eye_x = float(np.nonzero(iris)[1].mean())
        if not tip_x < eye_x - 4:
            raise SystemExit(f"front does not face left: horn {tip_x:.1f} eye {eye_x:.1f}")
        if (is_color(img, GOLD) & ~torso).any() or (is_color(img, WHITE) & ~torso).any():
            raise SystemExit("front eye glint left the head")
    else:
        if tip_x < 60:
            raise SystemExit(f"back horn is not on the right: {tip_x:.1f}")
        sx = float(np.nonzero(scythe)[1].mean())
        if sx >= tip_x:
            raise SystemExit("back claw is not behind the horn")
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
    touch = part & dilate(torso, 2)
    if not touch.any():
        touch = part & dilate(torso, 5)
    band = touch.copy()
    for _ in range(depth):
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
    # The horn is long, so its root cushion can be deeper than the short claw and prongs.
    depths = {"horn": 5, "scythe": 3, "leg0": 3, "leg1": 3, "leg2": 3, "leg3": 3}
    for name, depth in depths.items():
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth + 1
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
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        horn=0.0, horn_mode="up",
        scythe=0.0, scythe_mode="up",
        leg0=0.0, leg1=0.0, leg2=0.0, leg3=0.0, leg_mode="up",
        lean=0.0, knock=0, flash=False, spark=0, turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        p["horn"] = wave(f, n, 7.0, phase=0.3)
        p["scythe"] = wave(f, n, 6.0, phase=1.4)
        for i, name in enumerate(LEG_NAMES):
            p[name] = wave(f, n, 5.0, phase=0.8 + i * 1.2)
    elif action == "walk":
        # The four drawn prongs step. The horn and rear claw counter-sway. No new legs.
        p["bdx"] = int(round(wave(f, n, 1.0, phase=0.2, cycles=2.0)))
        p["horn"] = wave(f, n, 10.0, phase=0.6)
        p["scythe"] = wave(f, n, 12.0, phase=2.2, cycles=2.0)
        for i, name in enumerate(LEG_NAMES):
            p[name] = wave(f, n, 16.0, phase=i * (math.pi / 2.0), cycles=2.0)
    elif action == "attack":
        # 0 rest. 1-4 raise the horn (no mouth). 5-8 the opposite still.
        # 9-12 horn thrust plus the claw and the prongs. 13-15 recover.
        horn = [0, 8, 14, 20, 22, 22, 22, 22, 22, 18, 28, 34, 34, 16, 6, 0]
        scythe = [0, 6, 10, 14, 16, 16, 16, 16, 16, 12, 22, 28, 26, 10, 4, 0]
        leg = [0, 3, 6, 8, 8, 8, 8, 8, 8, 8, 14, 18, 16, 6, 2, 0]
        p["horn"] = float(horn[f])
        p["scythe"] = float(scythe[f])
        for name in LEG_NAMES:
            p[name] = float(leg[f])
        strike = f >= 9
        p["horn_mode"] = "fwd" if strike else "up"
        p["scythe_mode"] = "fwd" if strike else "up"
        p["leg_mode"] = "fwd" if strike else "up"
        if f in (3, 4):
            p["bdx"] = -1
        if f in (11, 12):
            p["bdx"] = 1
            p["spark"] = f - 10
        p["turned"] = 5 <= f <= 8
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["horn"] = 14.0 * env
        p["scythe"] = 12.0 * env
        for name in LEG_NAMES:
            p[name] = 8.0 * env
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


def limb_amp(p, name):
    if name == "horn":
        return p["horn"], p["horn_mode"]
    if name == "scythe":
        return p["scythe"], p["scythe_mode"]
    return p[name], p["leg_mode"]


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    for name in ("leg0", "leg1", "leg2", "leg3", "scythe", "horn"):
        amp, mode = limb_amp(p, name)
        ang = part_angle(rig, name, mode, amp)
        paste(out, moved(L[name], ang, piv[name], bdx, bdy))
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
    band = ys >= yb - 1
    return float(xs[band].mean()), float(yb)


def horn_tip(rig, p):
    ang = part_angle(rig, "horn", p["horn_mode"], p["horn"])
    dx = int(round(rig["facing"] * p["bdx"]))
    layer = moved(rig["layers"]["horn"], ang, rig["pivots"]["horn"], dx, int(p["bdy"]))
    return tip_xy(layer, rig["pivots"]["horn"])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (TEAL, MINT, WHITE)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (0, -r), (0, r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def limb_layers(rig, p):
    L = rig["layers"]
    bdx = int(round(rig["facing"] * p["bdx"]))
    bdy = int(p["bdy"])
    parts = {}
    for name in LIMBS:
        amp, mode = limb_amp(p, name)
        ang = part_angle(rig, name, mode, amp)
        parts[name] = moved(L[name], ang, rig["pivots"][name], bdx, bdy)
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


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    moved_part = any(abs(limb_amp(p, name)[0]) >= 0.45 for name in LIMBS)
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], piv)
    if moved_part or abs(p["lean"]) > 0.4:
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = horn_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
    return snap(img, palette)


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


def crop_box(canvas, pad=10):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad - 28)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 16)
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


def locked_palette(front_path, work):
    img = np.array(Image.open(front_path).convert("RGBA"))
    own = np.unique(img[img[..., 3] >= 128][:, :3], axis=0)
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
    if not (lock_dir / "palette.json").exists():
        py = str(FORGE / ".venv/bin/python")
        hexes = ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in own]
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
    want = {tuple(int(v) for v in c) for c in own}
    if got != want:
        raise SystemExit(f"locked palette differs from the still: {sorted(got ^ want)}")
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
    fc = {tuple(int(v) for v in c) for c in np.unique(front[front[..., 3] >= 128][:, :3], axis=0)}
    bc = {tuple(int(v) for v in c) for c in np.unique(back[back[..., 3] >= 128][:, :3], axis=0)}
    if not bc <= fc:
        raise SystemExit(f"back colours outside the front palette: {sorted(bc - fc)}")
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
        "horn": (0, 220, 180, 255),
        "scythe": (255, 140, 40, 255),
        "leg0": (220, 40, 40, 255),
        "leg1": (240, 220, 40, 255),
        "leg2": (40, 120, 255, 255),
        "leg3": (180, 40, 220, 255),
        "torso": (170, 170, 170, 255),
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
        "dex": 545, "form": "mega", "variant": "scolipede-mega", "gen": "gen5",
        "dir": "monsterSprite/gen5/0545_scolipede/mega/scolipede-mega",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def measure_frame(img, palette_set):
    partial = int(((img[..., 3] != 0) & (img[..., 3] != 255)).sum())
    a = img[..., 3] == 255
    edge = bool(a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any())
    off = 0
    if a.any():
        used = {tuple(int(v) for v in c) for c in np.unique(img[a][:, :3], axis=0)}
        off = len(used - palette_set)
    return off, partial, edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    args = ap.parse_args()

    save_peaks(Path(__file__).stem, PEAKS)
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    palette = locked_palette(SRC_DIR / "front.png", work)
    palette_set = {tuple(int(v) for v in c) for c in palette}
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    print(
        "parts front", {k: int(v.sum()) for k, v in fm.items()},
        "back", {k: int(v.sum()) for k, v in bm.items()},
        "palette", len(palette), "foot", FOOT_Y, flush=True,
    )
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, flush=True)
    if rigs["W"]["facing"] >= 0 or rigs["E"]["facing"] <= 0 or rigs["N"]["facing"] <= 0:
        raise SystemExit(f"facing map is wrong: {facing}")

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
    rest_sum = int(sum(rest_diff.values()))
    print("rest diff", rest_diff, "sum", rest_sum, flush=True)
    if rest_sum:
        raise SystemExit("frame 0 does not match the still")

    if args.samples:
        box = crop_box(rigs["S"]["canvas"])
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
        off = 0
        partial = 0
        edge = []
        lost = []
        bodies = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
                    img = synth(rigs, d, act, f, palette)
                    o, pa, ed = measure_frame(img, palette_set)
                    off += o
                    partial += pa
                    if ed:
                        edge.append(f"{act}:{d}:{f}")
                    if act in ("attack", "hurt", "walk"):
                        bodies.append(largest_fraction(img))
        body_min = round(min(bodies), 3) if bodies else 1
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["horn"], 1), round(p["scythe"], 1), p["horn_mode"], p["turned"], p["spin"]))
        print("attack poses", atk, flush=True)
        print(
            f"MEASURE rest_diff_sum={rest_sum} off_palette={off} partial_alpha={partial} "
            f"edge_n={len(edge)} disconnected={len(lost)} body_min={body_min}",
            flush=True,
        )
        print("samples", work, flush=True)
        if off or partial or edge or lost or body_min < 0.9:
            raise SystemExit("sample qa failed")
        return

    box = crop_box(rigs["S"]["canvas"])
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
    Image.fromarray(motion, "RGBA").save(OUT_DIR / "preview_motion.png")
    preview_image(sheets).save(OUT_DIR / "preview.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", OUT_DIR / "actions" / "preview.png")

    op = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[op][:, :3], axis=0)}
    off_cols = sorted(used - palette_set)
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
    body_min = round(min(bodies), 3) if bodies else 1
    report = {
        "version": VERSION, "hash": digest, "rest_diff": rest_diff, "idle0": idle0,
        "off_palette_colors": len(off_cols), "partial_alpha": partial, "edge_touching": clip[:12],
        "edge_n": len(clip), "body_main_min": body_min,
        "sprite_h": sprite_h, "bounds": bounds, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (
        not any(rest_diff.values()) and not any(idle0.values()) and not off_cols
        and partial == 0 and not clip and body_min >= 0.9
    )
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    print(
        f"MEASURE rest_diff_sum={int(sum(rest_diff.values()))} off_palette={len(off_cols)} "
        f"partial_alpha={partial} edge_n={len(clip)} idle0_sum={int(sum(idle0.values()))} "
        f"body_min={body_min}",
        flush=True,
    )
    meta = {
        "generator": "forge_gen5_scolipede_mega_actions", "version": VERSION, "key": KEY,
        "dex": 545, "variant": "scolipede-mega", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off_cols),
        "archetype": "armored centipede (teal horn, rear claw, four ventral prongs, no mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (horn tip x25 < eye x31.5); back horn tip is x70",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
