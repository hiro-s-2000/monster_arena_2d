#!/usr/bin/env python3
"""Scolipede normal (0545) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x4-92 y7-88, 2612 px, 13 colours,
alpha 0/255, sole y88):
  Yellow eye (222, 172, 16), 5 px, mean x 33.0, y34-36.
  The forward horn tip is x4 at y13-14, left of the eye, so the still faces
  screen-left. The upper horn tip is x15-16 y7. No mouth colour is drawn.
  The roar raises those two horns. It does not invent a mouth, a wedge, or rings.
  Grey-blue legs: one near curtain from y44 down (the upper grey stays on the
  head, around the eye), plus separate feet. The ground toe is x49-52 y85-88.
normal/back.png (bbox x3-92 y9-87, 2534 px, 12 colours) is placed 1 px lower
  so its sole shares y88. The horn tip is x92 y12-13, right of the yellow
  mark at mean x 71.3, so this back faces screen-right. Two horns, a leg
  column, and separate feet. No mouth.

Directions, front faces left, back faces away-right:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn. NW back mirrored.
  Attack 5-8 is that back at equal scale: W and SW show N, S SE E show N
  flipped. Back rows stay the back. No |cos| squash and no in-plane spin.

  idle 16: feet planted, the two horns breathe.
  walk 16: the drawn feet step in a wave. The near leg curtain rocks.
  attack: 0 still, 1-4 horns rise (no mouth), 5-8 the back still,
          9-12 the horns strike (drawn at 34 deg) and the near legs drive,
          11-12 impact, 13-15 recover.
  hurt 12: flash, knock-back lean (<=38 deg) and recover.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_scolipede_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0545_scolipede/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0545_scolipede/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0545_scolipede"
VERSION = "scolipede_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 288, 288
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 32
BACK_DROP = 1
FRONT_FOOT = 88
BACK_FOOT = 88
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. Back as drawn looks away to the right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# finish() may lift a horn this far. The thrust is drawn at 34, above this,
# so the hit is not stretched past the angle this script already draws.
PEAKS = {"horn_up": 16.0, "horn_fwd": 16.0}

INK = (16, 16, 16)
YELLOW = (222, 172, 16)
GOLD = (131, 106, 24)
MAGENTA = (197, 24, 90)
PURPLE = (139, 57, 156)
GREY_COLS = ((65, 74, 98), (57, 57, 74), (32, 41, 49))
EYE_COLS = (YELLOW, GOLD)

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
    """4x nearest in-plane rotation. Positive follows the shared forge formula. No squash."""
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
        return np.zeros_like(mask)
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


def grow_root(horn, op, taken, y1, xpad=1):
    """Follow this horn down its own pixels until the neck."""
    h = horn.copy()
    yy, xx = np.mgrid[0:op.shape[0], 0:op.shape[1]]
    ys, xs = np.nonzero(h)
    if len(xs) == 0:
        return h
    x0, x1 = int(xs.min()) - xpad, int(xs.max()) + xpad
    for y in range(int(ys.max()) + 1, y1 + 1):
        p = np.pad(h, 1, constant_values=False)
        near = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        add = op & (yy == y) & near & ~taken & ~h & (xx >= x0) & (xx <= x1)
        if not add.any():
            break
        h |= add
    return h


def absorb_islands(horn, rest, limit=40):
    """A speck that touches only this horn belongs to the horn."""
    h = horn.copy()
    changed = True
    while changed:
        changed = False
        for g in component_list(rest):
            if len(g) > limit:
                continue
            m = mask_from_pts(rest.shape, g)
            if (m & dilate(h, 1)).any() and not (m & dilate(rest & ~m, 1)).any():
                h |= m
                rest = rest & ~m
                changed = True
                break
    return h, rest


def horns_of(op, kind):
    yy, xx = np.mgrid[0:op.shape[0], 0:op.shape[1]]
    if kind == "front":
        face = largest_mask(op & (yy <= 18) & (xx <= 17))
        dorsal = largest_mask(op & (yy <= 18) & (xx >= 18) & (xx <= 30))
        y1 = 22
    else:
        face = largest_mask(op & (yy <= 17) & (xx >= 84))
        dorsal = largest_mask(op & (yy <= 17) & (xx >= 72) & (xx < 84))
        y1 = 24
    taken = np.zeros_like(op)
    fwd = grow_root(face, op, taken, y1, 1)
    taken |= fwd
    up = grow_root(dorsal & ~taken, op, taken, y1, 1)
    rest = op & ~fwd & ~up
    up, rest = absorb_islands(up, rest)
    fwd, rest = absorb_islands(fwd, rest)
    return fwd, up


def segment(img, kind):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    fwd, up = horns_of(op, kind)
    eye = any_color(img, EYE_COLS)
    if (eye & (fwd | up)).any():
        raise SystemExit(f"{kind} eye rode a horn")
    if not eye.any():
        raise SystemExit(f"{kind} lost its eye mark")
    grey = np.zeros_like(op)
    for c in GREY_COLS:
        grey |= is_color(img, c)
    grey &= ~fwd & ~up & ~dilate(eye, 2)
    groups = component_list(grey)
    groups.sort(key=len, reverse=True)
    masks = {"horn_fwd": fwd, "horn_up": up}
    if not groups:
        raise SystemExit(f"{kind} has no legs")
    big = mask_from_pts(op.shape, groups[0])
    if int(np.nonzero(big)[0].min()) < 40:
        big = largest_mask(big & (yy >= 44))
    masks["leg_near"] = big
    feet = []
    for g in groups[1:]:
        m = mask_from_pts(op.shape, g) & ~big
        if int(m.sum()) < 5:
            continue
        if int(np.nonzero(m)[0].min()) < 50:
            continue
        feet.append(largest_mask(m))
    feet.sort(key=lambda m: float(np.nonzero(m)[1].mean()))
    for i, m in enumerate(feet):
        masks[f"foot_{i}"] = m
    used = np.zeros_like(op)
    for m in masks.values():
        used |= m
    torso = op & ~used
    main = largest_mask(torso)
    names = list(masks)
    for g in component_list(torso & ~main):
        m = mask_from_pts(op.shape, g)
        best, score = None, 0
        for k in names:
            s = int((m & dilate(masks[k], 1)).sum())
            if s > score:
                best, score = k, s
        if best is None:
            main |= m
        else:
            masks[best] |= m
    # A toe that only touches the leg curtain rides with that curtain.
    for name in list(masks):
        if not name.startswith("foot_"):
            continue
        if touches(masks[name], main):
            continue
        if touches(masks[name], masks["leg_near"]):
            masks["leg_near"] |= masks[name]
            del masks[name]
        else:
            main |= masks[name]
            del masks[name]
    masks["torso"] = main
    check_partition(op, masks)
    for name, m in masks.items():
        if name == "torso":
            continue
        if largest_component(m) != int(m.sum()):
            raise SystemExit(f"{kind} {name} fell apart")
        if not touches(m, main):
            raise SystemExit(f"{kind} {name} does not touch the body")
    if largest_component(main) != int(main.sum()):
        raise SystemExit(f"{kind} torso fell apart")
    tip_x = int(np.nonzero(fwd)[1].min() if kind == "front" else np.nonzero(fwd)[1].max())
    eye_x = float(np.nonzero(eye)[1].mean())
    if kind == "front" and not tip_x < eye_x - 1:
        raise SystemExit(f"front does not face left: tip {tip_x} eye {eye_x:.1f}")
    if kind == "back" and not tip_x > eye_x + 1:
        raise SystemExit(f"back does not face right: tip {tip_x} eye {eye_x:.1f}")
    sole = int(np.nonzero(op)[0].max())
    want = FRONT_FOOT if kind == "front" else BACK_FOOT
    if sole != want:
        raise SystemExit(f"{kind} sole {sole} != {want}")
    return masks


def place(arr, drop=0):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    y = OFF_Y + int(drop)
    out[y:y + arr.shape[0], OFF_X:OFF_X + arr.shape[1]] = arr
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

    x_pos, y_pos = tip_after(12.0)
    x_neg, y_neg = tip_after(-12.0)
    up = -1.0 if y_neg < y_pos else 1.0
    if facing > 0:
        fwd = 1.0 if x_pos >= x_neg else -1.0
    else:
        fwd = 1.0 if x_pos <= x_neg else -1.0
    return {"up": up, "fwd": fwd}


def build_view(img, masks, kind, mirror, drop):
    canvas = place(img, drop)
    placed = {k: place(v, drop) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    for name in list(layers):
        if name == "torso" or int(placed[name].sum()) == 0:
            continue
        depth = 5 if name.startswith("horn") or name == "leg_near" else 4
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth,
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots, signs = {}, {}
    for name in layers:
        if name == "torso":
            continue
        pivots[name] = pivot_of(layers[name], layers["torso"])
        if name.startswith("horn") or name == "leg_near":
            signs[name] = signs_for(layers[name], pivots[name], facing)
        else:
            signs[name] = {"up": 1.0, "fwd": 1.0}
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    feet = [k for k in layers if k.startswith("foot_")]
    feet.sort(key=lambda k: float(np.nonzero(placed[k])[1].mean()) if placed[k].any() else 0.0)
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "has_mouth": False,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
        "feet": feet,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        horn_up=0.0, horn_fwd=0.0, horn_mode="up",
        leg=0.0, leg_mode="fwd",
        steps=[(0, 0)] * 8,
        lean=0.0, knock=0, flash=False,
        spark=0, turned=False, spin=0,
        head_dx=0, head_dy=0,
    )


def foot_wave(f, n, count):
    steps = []
    for i in range(count):
        phase = i * 0.85
        w = wave(f, n, 1.0, phase=phase)
        dx = int(round(wave(f, n, 2.0, phase=phase)))
        dy = -int(round(2 * max(0.0, w)))
        steps.append((dx, dy))
    while len(steps) < 8:
        steps.append((0, 0))
    return steps


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The horns breathe. Frame 0 is the still.
        p["horn_up"] = wave(f, n, 5.0, phase=0.4)
        p["horn_fwd"] = wave(f, n, 4.0, phase=0.9)
        p["horn_mode"] = "up"
    elif action == "walk":
        p["steps"] = foot_wave(f, n, 8)
        p["leg"] = wave(f, n, 6.0, phase=0.3)
        p["leg_mode"] = "fwd"
        p["horn_up"] = wave(f, n, 4.0, phase=0.2)
        p["horn_fwd"] = wave(f, n, 3.0, phase=1.1)
        p["horn_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 raise the horns. No mouth.
        # 5-8 the back still, cleared again after finish().
        # 9-12 the horns thrust and the near legs drive. 11-12 impact.
        # 13-15 return. 15 is the still.
        horn = [0, 8, 14, 18, 20, 0, 0, 0, 0, 18, 28, 34, 34, 12, 4, 0]
        leg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 12, 12, 4, 0, 0]
        bdx = [0, -1, -1, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        p["horn_up"] = float(horn[f])
        p["horn_fwd"] = float(horn[f])
        p["leg"] = float(leg[f])
        p["bdx"] = int(bdx[f])
        p["horn_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["leg_mode"] = "fwd"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["horn_up"] = 0.0
        p["horn_fwd"] = 0.0
        p["leg"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["steps"] = [(0, 0)] * 8
        p["spark"] = 0
        p["spin"] = 0
        p["head_dx"] = 0
        p["head_dy"] = 0
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


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    leg_ang = part_angle(rig, "leg_near", p["leg_mode"], p["leg"])
    if "leg_near" in L:
        paste(out, moved(L["leg_near"], leg_ang, piv["leg_near"], bdx, bdy))
    for i, name in enumerate(rig["feet"]):
        sx, sy = p["steps"][i] if i < len(p["steps"]) else (0, 0)
        dx = bdx + int(round(fc * sx))
        dy = bdy + int(sy)
        paste(out, moved(L[name], 0.0, piv[name], dx, dy))
    for name in ("horn_up", "horn_fwd"):
        if name not in L:
            continue
        ang = part_angle(rig, name, p["horn_mode"], p[name])
        paste(out, moved(L[name], ang, piv[name], bdx, bdy))
    return out


def flash_gold(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK)
    out[op & ~keep, :3] = np.array(YELLOW, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def horn_tip(rig, p):
    name = "horn_fwd" if "horn_fwd" in rig["layers"] else "horn_up"
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    ang = part_angle(rig, name, p["horn_mode"], p[name])
    layer = moved(rig["layers"][name], ang, rig["pivots"][name], bdx, bdy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (YELLOW, MAGENTA, PURPLE)
    r = 3 + int(age) * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2), (2, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def rig_for(rigs, d, p):
    """Attack 5-8 shows this side's back at equal scale. Back rows stay."""
    if not p.get("turned"):
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", mirror)]


def moving(p):
    if any(step != (0, 0) for step in p["steps"]):
        return True
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["horn_up"]) >= 0.45 or abs(p["horn_fwd"]) >= 0.45 or abs(p["leg"]) >= 0.45:
        return True
    return False


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["horn_up"]) >= 0.45 or abs(p["horn_fwd"]) >= 0.45 or abs(p["leg"]) >= 0.45:
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
            body = rotate_nearest(body, -fc * p["lean"], rear_pivot(body, fc))
        if rotated(p):
            body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_gold(img), palette)
    if p["spark"] and action == "attack" and not p["turned"] and "horn_fwd" in rig["layers"]:
        tx, ty = horn_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def limb_names(rig):
    return [n for n in rig["layers"] if n != "torso"]


def limb_lost(rig, p):
    body = compose(rig, p)
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    lost = []
    feet = rig["feet"]
    for name in limb_names(rig):
        if name in ("horn_up", "horn_fwd"):
            ang = part_angle(rig, name, p["horn_mode"], p[name])
            dx, dy = bdx, bdy
        elif name == "leg_near":
            ang = part_angle(rig, name, p["leg_mode"], p["leg"])
            dx, dy = bdx, bdy
        else:
            i = feet.index(name)
            sx, sy = p["steps"][i] if i < len(p["steps"]) else (0, 0)
            ang = 0.0
            dx = bdx + int(round(fc * sx))
            dy = bdy + int(sy)
        layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
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
    for img in imgs:
        cols = np.unique(img[img[..., 3] >= 128][:, :3], axis=0)
        own.extend(tuple(int(v) for v in c) for c in cols)
    own = sorted(set(own))
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
        raise SystemExit(f"locked palette differs from the stills: {sorted(got ^ want)}")
    return cols


def params_hash():
    src = Path(__file__).read_bytes() + (SRC_DIR / "front.png").read_bytes() + (SRC_DIR / "back.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def shift_down(img, n):
    out = np.zeros_like(img)
    if n <= 0:
        return img.copy()
    out[n:] = img[:-n]
    return out


def build_rigs():
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    back = shift_down(np.array(Image.open(SRC_DIR / "back.png").convert("RGBA")), BACK_DROP)
    if front.shape[:2] != (96, 96) or back.shape[:2] != (96, 96):
        raise SystemExit("expected 96x96 stills")
    for im in (front, back):
        im[im[..., 3] < 128] = 0
        im[im[..., 3] >= 128, 3] = 255
    fm = segment(front, "front")
    bm = segment(back, "back")
    rigs = {}
    for kind, img, masks, drop in (
        ("front", front, fm, 0),
        ("back", back, bm, 0),
    ):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror, drop)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs[("front", False)]["facing"] != -1.0 or rigs[("back", False)]["facing"] != 1.0:
        raise SystemExit("facing locked to the wrong side")
    if rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("this still has no mouth")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "horn_fwd": (255, 210, 40, 255),
        "horn_up": (255, 120, 40, 255),
        "leg_near": (80, 200, 120, 255),
        "torso": (170, 170, 170, 255),
    }
    foot_cols = (
        (40, 160, 220, 255), (220, 80, 160, 255), (180, 180, 60, 255),
        (120, 80, 220, 255), (80, 220, 200, 255), (220, 140, 80, 255),
        (160, 80, 80, 255),
    )
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        fi = 0
        for name, m in masks.items():
            if name.startswith("foot_"):
                part[m] = foot_cols[fi % len(foot_cols)]
                fi += 1
            else:
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
        "dex": 545, "form": "normal", "variant": "scolipede", "gen": "gen5",
        "dir": "monsterSprite/gen5/0545_scolipede/normal", "sheetDir": "agent-sprite-forge",
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

    turn_w = synth(rigs, "W", "attack", 6, palette, effects=False)
    turn_s = synth(rigs, "S", "attack", 6, palette, effects=False)
    if np.any(turn_w != rigs["N"]["canvas"]):
        raise SystemExit("W turn is not the N back at equal scale")
    if np.any(turn_s != rigs["NW"]["canvas"]):
        raise SystemExit("S turn is not N flipped at equal scale")

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
            atk.append((f, round(float(p["horn_fwd"]), 1), p["horn_mode"], p["turned"],
                        p["spin"], round(float(p["leg"]), 1), p["bdx"]))
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
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_scolipede_actions", "version": VERSION, "key": KEY,
        "dex": 545, "variant": "scolipede", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "centipede (two horns, a near leg curtain, separate feet, no mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front horn tip x4 is left of eye mean x 33.0; back horn tip x92 is right of yellow mean x 71.3",
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
