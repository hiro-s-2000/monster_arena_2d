#!/usr/bin/env python3
"""Zorua normal (0570) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x28-67 y24-70, 1075 px, alpha 0/255,
sole y70). Opaque colours:
  slate (90,90,115), indigo (32,32,49), ink (16,16,16), maroon (106,32,57),
  crimson (172,16,65), mid (65,65,82), white (255,255,255), navy (49,49,65),
  cyan (24,180,180), teal (49,131,131).
  The eye is the cyan/teal/white cluster, mean x 40.29, y 47-51.
  The face tip is the head's leftmost pixel in that band, x 29.
  Tip is left of the eye, so the still faces screen-left.
  No mouth is drawn. White is the eye highlight, not teeth. The roar does not
  invent a mouth, a wedge, or rings. It lifts the red mane (x36-46 y24-37).
  Four short paws, split where they separate, y>=63:
    leg_a x<=41 (near fore, toward the face), leg_b x42-48 (reaches y70),
    leg_c x49-53, leg_d x>=54 (toward the tail).
  The tail is the right tuft that is already separated by a gap (runs starting
  at x>=56, y42-52) plus the fluff directly under that cut down to y62.
normal/back.png (bbox x29-67 y24-70, 1008 px, sole y70) adds only
  (65,8,16) on the paw bottoms. The mane sits on the upper right (x48-58
  y24-34), so the back as drawn looks away toward screen-right. The tail is
  the left tuft. No eyes and no mouth.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn. NW back mirrored.
  Attack 5-8 is that direction's back at equal scale. W becomes N, S becomes
  NW. No |cos| squash and no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_zorua_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0570_zorua/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0570_zorua/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0570_zorua"
VERSION = "zorua_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 176
OFF_X = (CELL_W - 96) // 2
SRC_FOOT = 70
# Full 96-canvas fits. 26 px under the sole (the still's own lowest pixel).
FOOT_Y = 150
OFF_Y = FOOT_Y - SRC_FOOT
# Front as drawn looks screen-left. W keeps that picture. Its back is N.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# finish() may lift the scratch this far. The hit is drawn at 34, above this,
# so the strike is not stretched past the angle this script already draws.
PEAKS = {"paw": 18.0}
LIMBS = ("mane", "tail", "leg_a", "leg_b", "leg_c", "leg_d")
LEG_NAMES = ("leg_a", "leg_b", "leg_c", "leg_d")

INK = (16, 16, 16)
INDIGO = (32, 32, 49)
NAVY = (49, 49, 65)
MID = (65, 65, 82)
SLATE = (90, 90, 115)
MAROON = (106, 32, 57)
CRIMSON = (172, 16, 65)
WHITE = (255, 255, 255)
CYAN = (24, 180, 180)
TEAL = (49, 131, 131)
PAWRED = (65, 8, 16)
RED_COLS = (CRIMSON, MAROON)
EYE_COLS = (CYAN, TEAL, WHITE)

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


def runs_of(row):
    xs = np.nonzero(row)[0]
    if len(xs) == 0:
        return []
    runs = []
    s = int(xs[0])
    p = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x > p + 1:
            runs.append((s, p))
            s = x
        p = x
    runs.append((s, p))
    return runs


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
    """4x nearest in-plane rotation (positive = clockwise). No width squash."""
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


def paint_cushion(layer, allowed, steps, dirs):
    """Grow this layer's own colours into allowed pixels. Hidden at rest."""
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
    """Zero on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def check_partition(op, masks):
    acc = np.zeros(op.shape, np.int16)
    for m in masks.values():
        acc += m.astype(np.int16)
    if int(acc.sum()) != int(op.sum()) or int(acc.max()) != 1 or int(acc[op].min()) != 1:
        raise SystemExit(f"parts do not partition the still: cover {int(acc.sum())} vs {int(op.sum())}")


def front_tail_mask(op):
    """Right tuft already split by a gap, plus the fluff under that cut."""
    mask = np.zeros_like(op)
    starts = []
    for y in range(42, 53):
        runs = runs_of(op[y])
        for i in range(1, len(runs)):
            gap = runs[i][0] - runs[i - 1][1] - 1
            if gap >= 2 and runs[i][0] >= 56:
                for x0, x1 in runs[i:]:
                    mask[y, x0:x1 + 1] = True
                    starts.append(x0)
                break
    if not starts:
        raise SystemExit("front tail tip not found")
    cut = min(starts)
    for y in range(53, 63):
        xs = np.nonzero(op[y])[0]
        take = xs[xs >= cut]
        if len(take):
            mask[y, take] = True
    return mask


def back_tail_mask(op):
    """Left tuft already split by a gap, plus the fluff under that cut."""
    mask = np.zeros_like(op)
    ends = []
    for y in range(38, 48):
        runs = runs_of(op[y])
        for i in range(len(runs) - 1):
            gap = runs[i + 1][0] - runs[i][1] - 1
            if gap >= 2 and runs[i][1] <= 42:
                for x0, x1 in runs[:i + 1]:
                    mask[y, x0:x1 + 1] = True
                    ends.append(x1)
                break
    if not ends:
        raise SystemExit("back tail tip not found")
    cut = max(ends)
    for y in range(48, 63):
        xs = np.nonzero(op[y])[0]
        take = xs[xs <= cut]
        if len(take):
            mask[y, take] = True
    return mask


def mane_mask(img, op, y_max, ink_x):
    yy, xx = np.mgrid[0:96, 0:96]
    red = any_color(img, RED_COLS)
    core = red & op & (yy <= y_max)
    ink = is_color(img, INK) & (yy <= y_max + 2) & ink_x
    return core | (dilate(core, 1) & ink)


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    mane = mane_mask(img, op, 37, xx >= 34)
    tail = front_tail_mask(op) & ~mane
    leg_band = op & (yy >= 63) & ~mane & ~tail
    legs = {
        "leg_a": leg_band & (xx <= 41),
        "leg_b": leg_band & (xx >= 42) & (xx <= 48),
        "leg_c": leg_band & (xx >= 49) & (xx <= 53),
        "leg_d": leg_band & (xx >= 54),
    }
    used = mane | tail
    for m in legs.values():
        used = used | m
    torso = op & ~used
    masks = {"mane": mane, "tail": tail, **legs, "torso": torso}
    check_partition(op, masks)
    eye = any_color(img, EYE_COLS)
    if int(eye.sum()) < 15 or int((eye & torso).sum()) != int(eye.sum()):
        raise SystemExit("eyes left the head")
    face = op & (yy >= 45) & (yy <= 52) & (xx <= 52)
    tip = int(np.nonzero(face)[1].min())
    eye_x = float(np.nonzero(eye)[1].mean())
    if tip >= eye_x:
        raise SystemExit(f"face tip x {tip} is not left of the eye {eye_x:.2f}")
    if not (70 <= int(mane.sum()) <= 120):
        raise SystemExit(f"front mane looks wrong: {int(mane.sum())}")
    if not (60 <= int(tail.sum()) <= 280):
        raise SystemExit(f"front tail looks wrong: {int(tail.sum())}")
    for name, m in legs.items():
        if int(m.sum()) < 15:
            raise SystemExit(f"{name} looks wrong: {int(m.sum())}")
    if int(np.nonzero(legs["leg_b"])[0].max()) != SRC_FOOT:
        raise SystemExit("near sole is not on the foot line")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    mane = mane_mask(img, op, 34, xx >= 45)
    tail = back_tail_mask(op) & ~mane
    leg_band = op & (yy >= 63) & ~mane & ~tail
    legs = {
        "leg_a": leg_band & (xx <= 37),
        "leg_b": leg_band & (xx >= 38) & (xx <= 44),
        "leg_c": leg_band & (xx >= 45) & (xx <= 50),
        "leg_d": leg_band & (xx >= 51),
    }
    used = mane | tail
    for m in legs.values():
        used = used | m
    torso = op & ~used
    masks = {"mane": mane, "tail": tail, **legs, "torso": torso}
    check_partition(op, masks)
    if any_color(img, EYE_COLS).any():
        raise SystemExit("back still grew an eye")
    paw = is_color(img, PAWRED)
    if int(paw.sum()) < 8 or int((paw & used).sum()) != int(paw.sum()):
        raise SystemExit("back paw red left the feet")
    if not (40 <= int(mane.sum()) <= 100):
        raise SystemExit(f"back mane looks wrong: {int(mane.sum())}")
    if not (40 <= int(tail.sum()) <= 280):
        raise SystemExit(f"back tail looks wrong: {int(tail.sum())}")
    for name, m in legs.items():
        if int(m.sum()) < 15:
            raise SystemExit(f"back {name} looks wrong: {int(m.sum())}")
    if int(np.nonzero(legs["leg_b"])[0].max()) != SRC_FOOT:
        raise SystemExit("back sole is not on the foot line")
    return masks


def place(arr):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    out[OFF_Y:OFF_Y + 96, OFF_X:OFF_X + 96] = arr
    if OFF_Y < 0 or OFF_X < 0 or OFF_Y + 96 > CELL_H or OFF_X + 96 > CELL_W:
        raise SystemExit("still does not fit in the cell")
    return out


def pivot_of(layer, torso_layer):
    part = layer[..., 3] >= 128
    torso = torso_layer[..., 3] >= 128
    touch = part & dilate(torso, 2)
    if not touch.any():
        touch = part & dilate(torso, 8)
    ys, xs = np.nonzero(part)
    if len(xs) == 0:
        return 0.0, 0.0
    if not touch.any():
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    # Root is the contact closest to the torso, not the free tip.
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
    canvas = place(img)
    placed = {k: place(v) for k, v in masks.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    raw_torso = layers["torso"].copy()
    leg_allow = np.zeros(canvas.shape[:2], bool)
    for name in LEG_NAMES:
        leg_allow |= placed[name]
    # Cushion sits under the part that is pasted later, so rest stays exact.
    layers["torso"] = paint_cushion(layers["torso"], leg_allow, 4, ("down",))
    layers["torso"] = paint_cushion(layers["torso"], placed["mane"], 4, ("up",))
    tail_dir = ("right",) if kind == "front" else ("left",)
    layers["torso"] = paint_cushion(layers["torso"], placed["tail"], 6, tail_dir)
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        raw_torso = np.ascontiguousarray(raw_torso[:, ::-1])
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in LIMBS:
        pivots[name] = pivot_of(layers[name], raw_torso)
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    # The paw that reaches toward the face. Front: near fore. Back: the paw
    # on the mane side, which is the right paw before mirroring.
    striker = "leg_a" if kind == "front" else "leg_d"
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "has_mouth": False,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
        "striker": striker,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        paw=0.0, paw_mode="fwd",
        mane=0.0, tail=0.0,
        step_a=(0, 0), step_b=(0, 0),
        lean=0.0, knock=0, flash=False,
        spark=0, turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. Mane and tail breathe. Frame 0 is the still.
        p["bdy"] = -1 if wave(f, n, 1.0) > 0.55 else 0
        p["mane"] = wave(f, n, 6.0, phase=0.4)
        p["tail"] = wave(f, n, 8.0, phase=0.9, cycles=1.0)
    elif action == "walk":
        # Diagonal trot: near fore with the far hind, the other pair opposite.
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_a"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["step_b"] = (int(round(2 * sb)), -int(round(3 * max(0.0, sb))))
        p["bdy"] = -1 if abs(math.sin(u)) > 0.85 else 0
        p["tail"] = wave(f, n, 10.0, cycles=2.0, phase=math.pi / 2)
        p["mane"] = wave(f, n, 4.0, phase=0.5)
    elif action == "attack":
        # 0 rest. 1-4 the mane flares (no mouth on this still).
        # 5-8 the back, equal scale, cleared after finish().
        # 9 the near paw pulls back. 10-12 it scratches. 13-15 return.
        mane = [0, 8, 12, 16, 16, 0, 0, 0, 0, 4, 4, 2, 0, 0, 0, 0]
        paw = [0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 28, 34, 34, 12, 4, 0]
        tail = [0, 4, 8, 10, 10, 0, 0, 0, 0, -6, 4, 8, 8, 4, 0, 0]
        bdx = [0, -1, -1, -1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0]
        p["mane"] = float(mane[f])
        p["paw"] = float(paw[f])
        p["tail"] = float(tail[f])
        p["bdx"] = int(bdx[f])
        p["paw_mode"] = "back" if f == 9 else "fwd"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.64, 0.44, 0.26, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        # The back is the still, at equal scale. finish() must not swing it.
        p["paw"] = 0.0
        p["mane"] = 0.0
        p["tail"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_a"] = (0, 0)
        p["step_b"] = (0, 0)
        p["spark"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["turned"] = True
        p["paw_mode"] = "fwd"
    if action == "attack" and abs(float(p["paw"])) > 34.05:
        raise SystemExit(f"scratch drawn past 34 deg at frame {f}: {p['paw']}")
    if action == "hurt" and abs(float(p["lean"])) > 38.05:
        raise SystemExit(f"hurt lean past 38 deg at frame {f}: {p['lean']}")
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


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    out = np.zeros_like(rig["canvas"])
    # Feet stay on their own step. The body bob does not lift a planted sole.
    steps = {
        "leg_a": p["step_a"], "leg_c": p["step_a"],
        "leg_b": p["step_b"], "leg_d": p["step_b"],
    }
    paste(out, shift(L["torso"], bdx, bdy))
    paste(out, moved(L["tail"], part_angle(rig, "tail", "up", p["tail"]), piv["tail"], bdx, bdy))
    paste(out, moved(L["mane"], part_angle(rig, "mane", "up", p["mane"]), piv["mane"], bdx, bdy))
    order = [n for n in LEG_NAMES if n != rig["striker"]] + [rig["striker"]]
    for name in order:
        sx, sy = steps[name]
        if name == rig["striker"] and abs(p["paw"]) >= 0.45:
            ang = part_angle(rig, name, p["paw_mode"], p["paw"])
            paste(out, moved(L[name], ang, piv[name], bdx + int(round(fc * sx)), bdy + int(sy)))
        else:
            paste(out, shift(L[name], bdx + int(round(fc * sx)), bdy + int(sy)))
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


def strike_tip(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    name = rig["striker"]
    ang = part_angle(rig, name, p["paw_mode"], p["paw"])
    layer = moved(L[name], ang, piv[name], bdx, bdy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    """A few palette pixels in the empty space in front of the paw."""
    ox, oy = origin
    cols = (CRIMSON, WHITE, INK)
    r = 3 + int(age)
    for i, (dx, dy) in enumerate(((r, 0), (r, -2), (r, 2), (r + 1, -1), (r + 1, 1))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    moved_part = (
        abs(p["paw"]) >= 0.45 or abs(p["mane"]) >= 0.45 or abs(p["tail"]) >= 0.45
        or p["step_a"] != (0, 0) or p["step_b"] != (0, 0)
        or p["bdx"] or p["bdy"]
    )
    fc = rig["facing"]
    if not moved_part and abs(p["lean"]) <= 0.4 and not p["knock"]:
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        rotated = abs(p["lean"]) > 0.4 or abs(p["paw"]) >= 0.45 or abs(p["mane"]) >= 0.45 or abs(p["tail"]) >= 0.45
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            body = rotate_nearest(body, -fc * p["lean"], rear_pivot(body, fc))
            body = fill_pinholes(body)
        if rotated:
            body = despeckle(body)
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"] and not rig["has_mouth"]:
        tx, ty = strike_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_component(mask):
    seen = np.zeros_like(mask, bool)
    best = 0
    h, w = mask.shape
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
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
        if n > best:
            best = n
    return best


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
             "--source", str(front_path), str(back_path), "--strict",
             "--output-dir", str(lock_dir)],
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


def build_rigs():
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    back = np.array(Image.open(SRC_DIR / "back.png").convert("RGBA"))
    if front.shape[:2] != (96, 96) or back.shape[:2] != (96, 96):
        raise SystemExit("expected 96x96 stills")
    for im in (front, back):
        im[im[..., 3] < 128] = 0
        im[im[..., 3] >= 128, 3] = 255
    if int(np.nonzero(front[..., 3])[0].max()) != SRC_FOOT:
        raise SystemExit("front foot line changed")
    if int(np.nonzero(back[..., 3])[0].max()) != SRC_FOOT:
        raise SystemExit("back foot line changed")
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {}
    for kind, img, masks in (
        ("front", front, fm),
        ("back", back, bm),
    ):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs[("front", False)]["facing"] != -1.0 or rigs[("back", False)]["facing"] != 1.0:
        raise SystemExit("facing locked to the wrong side")
    if rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("this still has no mouth")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "mane": (255, 80, 40, 255),
        "tail": (255, 200, 40, 255),
        "leg_a": (40, 200, 90, 255),
        "leg_b": (40, 140, 220, 255),
        "leg_c": (220, 80, 180, 255),
        "leg_d": (180, 80, 220, 255),
        "torso": (170, 170, 170, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
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
        "dex": 570, "form": "normal", "variant": "zorua", "gen": "gen5",
        "dir": "monsterSprite/gen5/0570_zorua/normal",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
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

    zero = base_pose()
    compose_diff = {}
    for key in (("front", False), ("front", True), ("back", False), ("back", True)):
        got = compose(rigs[key], zero)
        compose_diff[str(key)] = int(np.any(got != rigs[key]["canvas"], axis=2).sum())
    print("compose zero", compose_diff, flush=True)
    if any(compose_diff.values()):
        raise SystemExit("zero pose does not rebuild the still")

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
    turn_e = synth(rigs, "E", "attack", 6, palette, effects=False)
    if np.any(turn_w != rigs["N"]["canvas"]):
        raise SystemExit("W turn is not the N back at equal scale")
    if np.any(turn_s != rigs["NW"]["canvas"]):
        raise SystemExit("S turn is not N flipped at equal scale")
    if np.any(turn_e != rigs["NW"]["canvas"]):
        raise SystemExit("E turn is not N flipped at equal scale")
    if np.any(turn_n != rigs["N"]["canvas"]):
        raise SystemExit("N turn left the back still")

    if args.samples:
        metrics = frame_metrics(rigs, palette)
        rest_sum = int(sum(rest_diff.values()))
        print(
            f"静止差分 {rest_sum} パレット外 {metrics['off_palette']} "
            f"中間アルファ {metrics['partial_alpha']} 枠接触 {metrics['edge_touch']}",
            flush=True,
        )
        print("qa_samples", "rest_diff", rest_sum,
              "off_palette", metrics["off_palette"],
              "partial_alpha", metrics["partial_alpha"],
              "edge_touch", metrics["edge_touch"],
              "body_main_min", metrics["body_main_min"],
              "clip", metrics["edge_touching"], flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(float(p["paw"]), 1), p["paw_mode"], p["turned"],
                        p["spin"], round(float(p["mane"]), 1), p["bdx"]))
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
    rest_sum = int(sum(rest_diff.values()))
    print(
        f"静止差分 {rest_sum} パレット外 {len(off)} 中間アルファ {partial} 枠接触 {len(clip)}",
        flush=True,
    )
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
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_zorua_actions", "version": VERSION, "key": KEY,
        "dex": 570, "variant": "zorua", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "quadruped fox (cyan eye, red mane, four short paws, bushy tail, no mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (face tip x 29 < eye mean x 40.29)",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    if args.no_publish:
        print("qa ok; publish skipped", flush=True)
        return
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
