#!/usr/bin/env python3
"""Timburr normal (0532) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x24-70 y27-67, 1066 px, alpha 0/255,
15 colours, foot y67):
  Pink nose x38-45 y29-38, mean x 41.6. Eyes (white/silver) mean x 47.6
  at y38-45. The nose is left of the eyes, so the still faces screen-left.
  A small brown mouth sits at x50-54 y45-47 (10 px). The same brown is on
  the back face, so the roar uses that mouth. It does not invent a new one.
  The log is the wood colours: left shaft x24-37 y49-57 and right shaft
  with the pale cut end x50-70 y47-58. The body crosses the middle, so the
  two visible pieces are one beam and swing on one pivot.
  Lower pink (y44-58) is the chest stripe and stays on the torso.
  Feet split at y62: near/left x33-42 down to y67, far/right x47-54 to y65.

normal/back.png (bbox x25-76 y24-65, 1116 px, foot y65):
  Nose pink x42-48 y31-37. Eyes x46-53 sit to the right of that nose, and
  the cream cheek is on the right of the grey skull, so back as drawn looks
  away to the right. Mouth x53-56 y40-42. One wood beam x25-76 y42-60.
  Feet split at y60: left x27-39, right x43-52 down to y65.

Directions, front faces left, back faces away-right:
  S SE E front mirrored (screen-right). SW W front as drawn (screen-left).
  NE back as drawn. N NW back mirrored.
  E is screen-right, W is screen-left, N is the back.
  Attack 5-8 shows that direction's back still at equal scale (same mirror,
  no |cos| squash, body never laid on its side). S turns into the N still.

  idle 16: feet planted, 1px breath, nose and log rock.
  walk 16: the two feet that are drawn, one step cycle.
  attack: 0 still, 1-4 mouth wedge and ring, nose up, log windup,
          5-8 the back still, 9-12 the log strikes, 11-12 impact,
          13-15 recover.
  hurt 12: white flash, knock-back lean (<=38 deg) and recover.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_timburr_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0532_timburr/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0532_timburr/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0532_timburr"
VERSION = "timburr_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 176, 176
OFF_X = (CELL_W - 96) // 2
FRONT_FOOT = 67
BACK_FOOT = 65
FOOT_Y = 140
# Front as drawn looks screen-left. Back as drawn looks away to the right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "SW": ("front", False), "W": ("front", False),
    "NE": ("back", False), "N": ("back", True), "NW": ("back", True),
}
MAX_LEAN = 38.0
# Angles this script already draws. finish() will not raise a frame past them.
PEAKS = {"beam": 34.0, "nose": 12.0}

INK = (16, 16, 16)
CREAM = (197, 189, 172)
MID = (156, 139, 131)
DK = (90, 74, 74)
PINK = (238, 123, 139)
PINK_M = (197, 90, 115)
PINK_D = (131, 49, 65)
WHITE = (255, 255, 255)
SILVER = (197, 197, 197)
BROWN = (148, 98, 90)
BROWN_D = (106, 57, 49)
WOOD_D = (90, 57, 8)
WOOD_M = (172, 115, 8)
WOOD = (123, 82, 8)
WOOD_HI = (222, 180, 98)

PINK_COLS = (PINK, PINK_M, PINK_D)
WOOD_COLS = (WOOD_D, WOOD_M, WOOD, WOOD_HI)
BROWN_COLS = (BROWN, BROWN_D)
EYE_COLS = (WHITE, SILVER)
OUTLINE_KEEP = (INK, DK, BROWN_D)

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


def dilate8(mask):
    p = np.pad(mask, 1, constant_values=False)
    out = mask.copy()
    for dy in range(3):
        for dx in range(3):
            out = out | p[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return out


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
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
        out.append(n)
    return sorted(out, reverse=True)


def largest_component(mask):
    sizes = components(mask)
    return sizes[0] if sizes else 0


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def check_partition(op, masks):
    acc = np.zeros(op.shape, np.int16)
    for m in masks.values():
        acc += m.astype(np.int16)
    if int((acc > 1).sum()) or int((op & (acc == 0)).sum()):
        raise SystemExit("parts do not partition the still")


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
    """4x nearest in-plane rotation (positive = clockwise). No squash."""
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


def put(img, x, y, col, over=False):
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 1 <= x < w - 1 and 1 <= y < h - 1 and (over or img[y, x, 3] < 128):
        img[y, x, :3] = col
        img[y, x, 3] = 255


def paint_cushion(layer, allowed, steps, dirs=("up", "down", "left", "right")):
    acc = layer.copy()
    allow = allowed
    if allow.ndim == 3:
        allow = allow[..., 0]
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
            take = (acc[..., 3] < 128) & allow & (mv[..., 3] >= 128)
            acc[take] = mv[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    wood = any_color(img, WOOD_COLS)
    pink = any_color(img, PINK_COLS)
    mouth = any_color(img, BROWN_COLS)
    eye = any_color(img, EYE_COLS)
    ink = is_color(img, INK)
    # Nose is the pink oval on the head, left of the eyes. Chest pink stays.
    nose = pink & (yy <= 40)
    body_cols = op & ~wood & ~ink
    beam_ink = (ink & dilate(wood, 1)) & ~dilate(body_cols, 1)
    beam = (wood | beam_ink) & ~mouth & ~nose
    leg_l = op & (yy >= 62) & (xx <= 44) & ~beam
    leg_r = op & (yy >= 62) & (xx >= 45) & ~beam
    mouth = mouth & ~beam & ~nose & ~leg_l & ~leg_r
    nose = nose & ~beam & ~mouth
    used = beam | nose | mouth | leg_l | leg_r
    torso = op & ~used
    masks = {
        "beam": beam, "nose": nose, "mouth": mouth,
        "leg_l": leg_l, "leg_r": leg_r, "torso": torso,
    }
    check_partition(op, masks)
    if int((eye & torso).sum()) != int(eye.sum()) or eye.sum() < 8:
        raise SystemExit("front eyes left the head")
    if int(nose.sum()) < 30 or int(mouth.sum()) < 8 or int(beam.sum()) < 200:
        raise SystemExit("front nose, mouth, or beam collapsed")
    if largest_component(leg_l) != int(leg_l.sum()) or largest_component(leg_r) != int(leg_r.sum()):
        raise SystemExit("front feet fell apart")
    for name in ("nose", "mouth", "beam", "leg_l", "leg_r"):
        if not (masks[name] & dilate(torso, 1)).any():
            raise SystemExit(f"front {name} does not touch the body")
    ny = np.nonzero(nose)
    ey = np.nonzero(eye)
    if float(ny[1].mean()) >= float(ey[1].mean()):
        raise SystemExit("front nose is not left of the eyes")
    if int(np.nonzero(op)[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    wood = any_color(img, WOOD_COLS)
    pink = any_color(img, PINK_COLS)
    mouth = any_color(img, BROWN_COLS)
    eye = any_color(img, EYE_COLS)
    ink = is_color(img, INK)
    nose = pink & (yy <= 42)
    body_cols = op & ~wood & ~ink
    beam_ink = (ink & dilate(wood, 1)) & ~dilate(body_cols, 1)
    beam = (wood | beam_ink) & ~mouth & ~nose
    # Feet from the gap at y60: left column ends at x39, right starts at x43.
    leg_l = op & ~beam & (yy >= 60) & (xx <= 40)
    leg_r = op & ~beam & (yy >= 60) & (xx >= 43)
    mouth = mouth & ~beam & ~nose & ~leg_l & ~leg_r
    nose = nose & ~beam & ~mouth
    used = beam | nose | mouth | leg_l | leg_r
    torso = op & ~used
    masks = {
        "beam": beam, "nose": nose, "mouth": mouth,
        "leg_l": leg_l, "leg_r": leg_r, "torso": torso,
    }
    check_partition(op, masks)
    if int((eye & torso).sum()) != int(eye.sum()) or eye.sum() < 8:
        raise SystemExit("back eyes left the head")
    if int(nose.sum()) < 20 or int(mouth.sum()) < 4 or int(beam.sum()) < 400:
        raise SystemExit("back nose, mouth, or beam collapsed")
    if largest_component(leg_l) != int(leg_l.sum()) or largest_component(leg_r) != int(leg_r.sum()):
        raise SystemExit(f"back feet fell apart {components(leg_l)} {components(leg_r)}")
    for name in ("nose", "mouth", "beam", "leg_l", "leg_r"):
        if not (masks[name] & dilate(torso, 1)).any():
            raise SystemExit(f"back {name} does not touch the body")
    if float(np.nonzero(eye)[1].mean()) <= float(np.nonzero(nose)[1].mean()):
        raise SystemExit("back eye is not right of the nose")
    if int(np.nonzero(op)[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line changed")
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


def pivot_of(part, body):
    touch = part & dilate(body, 1)
    if not touch.any():
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    return float(xs.mean()), float(ys.mean())


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
    facing = 1.0 if mirror else -1.0
    if kind == "back":
        facing = -facing
    torso_m = placed["torso"]
    for ln in ("leg_l", "leg_r"):
        allowed = torso_m & dilate(placed[ln], 2)
        layers[ln] = paint_cushion(layers[ln], allowed, 6, dirs=("up", "left", "right"))
    for name, depth, reach in (("nose", 4, 2), ("mouth", 3, 2), ("beam", 12, 4)):
        allowed = placed[name] & dilate(torso_m, reach)
        layers["torso"] = paint_cushion(layers["torso"], allowed, depth)
    pivots = {}
    signs = {}
    for name in ("nose", "mouth", "beam"):
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
        beam=0.0, nose=0.0, mouth=0.0,
        beam_mode="up", nose_mode="up", mouth_mode="open",
        leg_l=(0, 0), leg_r=(0, 0),
        lean=0.0, knock=0, flash=False,
        ring=-1, spark=0, turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        p["bdy"] = -1 if wave(f, n, 1.0) > 0.55 else 0
        p["nose"] = wave(f, n, 6.0, phase=0.4)
        p["beam"] = wave(f, n, 3.0, phase=1.1)
        p["nose_mode"] = "up"
        p["beam_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa, sb = math.sin(u), math.sin(u + math.pi)
        p["leg_l"] = (int(round(2.0 * sa)), -int(round(3.0 * max(0.0, sa))))
        p["leg_r"] = (int(round(2.0 * sb)), -int(round(3.0 * max(0.0, sb))))
        bob = wave(f, n, 1.0, cycles=2.0)
        p["bdy"] = -1 if bob > 0.45 else 0
        p["beam"] = wave(f, n, 4.0, phase=0.6)
        p["nose"] = wave(f, n, 4.0, phase=0.9)
        p["beam_mode"] = "up"
        p["nose_mode"] = "up"
    elif action == "attack":
        # 0 still. 1-4 roar (env 0, so these stay). 5-8 cleared after finish.
        # 9-12 log strike at or above finish()'s envelope. 13-15 recover.
        beam = [0, 8, 14, 20, 18, 0, 0, 0, 0, 34, 34, 28, 18, 8, 4, 0]
        nose = [0, 6, 10, 12, 10, 0, 0, 0, 0, 12, 12, 9, 6, 3, 1, 0]
        mouth = [0, 8, 14, 18, 16, 0, 0, 0, 0, 8, 6, 4, 2, 0, 0, 0]
        p["beam"] = float(beam[f])
        p["nose"] = float(nose[f])
        p["mouth"] = float(mouth[f])
        p["beam_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["nose_mode"] = "fwd" if 9 <= f <= 12 else "up"
        p["mouth_mode"] = "open"
        p["turned"] = 5 <= f <= 8
        if 2 <= f <= 4:
            p["bdy"] = -1
        if 10 <= f <= 12:
            p["bdx"] = 1
        if 1 <= f <= 4:
            p["ring"] = f - 1
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.66, 0.46, 0.28, 0.14, 0.05, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["nose"] = 6.0 * env
        p["nose_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["beam"] = 0.0
        p["nose"] = 0.0
        p["mouth"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["head_dx"] = 0
        p["head_dy"] = 0
        p["leg_l"] = (0, 0)
        p["leg_r"] = (0, 0)
        p["lean"] = 0.0
        p["knock"] = 0
        p["spark"] = 0
        p["ring"] = -1
        p["spin"] = 0
        p["flash"] = False
        p["beam_mode"] = "up"
        p["nose_mode"] = "up"
        p["mouth_mode"] = "open"
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


def body_shift(p):
    return int(round(p["bdx"])), int(round(p["bdy"]))


def angle_of(rig, p, name):
    mode = p["beam_mode"] if name == "beam" else p["nose_mode"] if name == "nose" else p["mouth_mode"]
    amp = p[name]
    return part_angle(rig, name, mode, amp)


def compose(rig, p):
    fc = rig["facing"]
    bdx, bdy = body_shift(p)
    # bdx is authored in the facing direction. Feet do not take the breath.
    bdx = int(round(fc * bdx))
    out = np.zeros_like(rig["canvas"])
    for name in ("leg_l", "leg_r"):
        dx, dy = p[name]
        paste(out, shift(rig["layers"][name], int(round(fc * dx)), int(round(dy))))
    paste(out, shift(rig["layers"]["torso"], bdx, bdy))
    for name in ("nose", "mouth", "beam"):
        ang = angle_of(rig, p, name)
        paste(out, moved(rig["layers"][name], ang, rig["pivots"][name], bdx, bdy))
    return out


def rig_for(rigs, d, p):
    kind, mirror = VIEW_OF[d]
    if p["turned"] and kind == "front":
        kind = "back"
    return rigs[(kind, mirror)]


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    for name in ("beam", "nose", "mouth"):
        if abs(p[name]) >= 0.45:
            return True
    return False


def moving(p):
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    if p["leg_l"] != (0, 0) or p["leg_r"] != (0, 0):
        return True
    return rotated(p)


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


def part_layer(rig, name, p):
    fc = rig["facing"]
    bdx, bdy = body_shift(p)
    bdx = int(round(fc * bdx))
    if name in ("leg_l", "leg_r"):
        dx, dy = p[name]
        return shift(rig["layers"][name], int(round(fc * dx)), int(round(dy)))
    return moved(rig["layers"][name], angle_of(rig, p, name), rig["pivots"][name], bdx, bdy)


def roar(img, origin, fc, age):
    """Wedge and ring in front of the mouth that is already drawn."""
    ox, oy = origin
    cols = (INK, BROWN, PINK, CREAM, WHITE)
    for i, (dx, dy) in enumerate(((1, 0), (2, -1), (2, 1), (3, 0), (2, 0))):
        put(img, ox + fc * (dx + age), oy + dy, cols[i % 5])
    rad = 5 + age * 3
    prev = None
    for deg in range(-40, 41, 8):
        a = math.radians(deg)
        x = ox + fc * (3 + rad * math.cos(a))
        y = oy + rad * math.sin(a) * 0.55
        put(img, x, y, cols[(deg // 8) % 5])
        if prev is not None:
            put(img, (x + prev[0]) / 2, (y + prev[1]) / 2, cols[age % 5])
        prev = (x, y)


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, WOOD_HI, CREAM, INK)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (r + 1, -1), (r + 1, 1))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def limb_names(rig):
    return [n for n in ("leg_l", "leg_r", "nose", "mouth", "beam") if n in rig["layers"]]


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
    if rotated(p):
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["ring"] >= 0 and not p["turned"]:
        mouth = part_layer(rig, "mouth", p)
        tip = facing_tip(mouth, fc) if mouth is not None else None
        if tip is not None:
            roar(img, tip, fc, p["ring"])
            img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        beam = part_layer(rig, "beam", p)
        tip = facing_tip(beam, fc) if beam is not None else None
        if tip is not None:
            impact(img, (tip[0] + fc * 2, tip[1]), fc, p["spark"])
            img = snap(img, palette)
    return img


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


def crop_box(canvas, pad=6):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad - 16)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 14)
    x0 = max(0, int(xs.min()) - pad - 18)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 18)
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
    for kind, img, masks, foot in (
        ("front", front, fm, FRONT_FOOT),
        ("back", back, bm, BACK_FOOT),
    ):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror, foot)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs["W"]["facing"] != -1.0 or rigs["E"]["facing"] != 1.0:
        raise SystemExit("front facing is not left-as-drawn")
    if rigs["NE"]["facing"] != 1.0 or rigs["N"]["facing"] != -1.0:
        raise SystemExit("back facing is not away-right as drawn")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "beam": (180, 120, 40, 255),
        "nose": (230, 80, 120, 255),
        "mouth": (120, 50, 40, 255),
        "leg_l": (70, 140, 90, 255),
        "leg_r": (40, 90, 160, 255),
        "torso": (180, 180, 170, 255),
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
        "dex": 532, "form": "normal", "variant": "timburr", "gen": "gen5",
        "dir": "monsterSprite/gen5/0532_timburr/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def tip_shift(rigs, d, name):
    rest = pose("attack", 0)
    hit = pose("attack", 10)
    rig = rig_for(rigs, d, rest)
    la = part_layer(rig, name, rest)
    lb = part_layer(rig, name, hit)
    fa = rig["facing"]
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
        raise SystemExit("attack 5-8 is not the back still")

    strike = {name: tip_shift(rigs, "S", name) for name in ("beam", "nose", "mouth")}
    print("strike dx S", strike, flush=True)

    pal = {tuple(int(v) for v in c) for c in palette}
    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        y0, y1, x0, x1 = box
        off = set()
        partial = 0
        edge = []
        lost = []
        bodies = []
        stored = {d: {a: [] for a in ACTIONS} for d in ("S", "W", "N", "E")}
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
            atk.append((f, round(p["beam"], 1), round(p["nose"], 1), round(p["mouth"], 1),
                        p["beam_mode"], p["turned"], p["spin"], p["ring"], p["bdx"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
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
    strips = [motion_strip(sheets, DIRS.index(d), (y0, y1, x0, x1)) for d in ("S", "N", "E", "W")]
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
    meta = {
        "generator": "forge_gen5_timburr_actions", "version": VERSION, "key": KEY,
        "dex": 532, "variant": "timburr", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped (pink nose, small mouth, wooden beam, two stubby feet)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front nose mean x is left of the eyes; back eye is right of the nose (away-right)",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
