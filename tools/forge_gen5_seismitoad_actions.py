#!/usr/bin/env python3
"""Seismitoad normal (0537) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x10-80 y18-82, 2747 px, 13 colours,
alpha 0/255, ground y82):
  Orange mouth x26-44 y29-35 (tip x26, mean x36.0). Two silver eye shines at
  (29, 30) and (40, 32), mean x34.5. The mouth tip is left of the eyes, so the
  still faces screen-left. The mouth is drawn, so the roar uses a wedge and a
  ring in these colours. No wings, horns, tail, or pincers.
  Face-side arm (fingers) x10-26 y49-68. Far hand x72-80 y42-65.
  Short foot x25-42 y70-79. Ground foot x58-71 y71-82.
normal/back.png (bbox x18-85 y16-79, 2642 px, 10 colours, all in the front
  palette, ground y79). No mouth and no silver. One fist on the right,
  x76-85 y54-66. Feet: left x18-36 y71-77, ground x48-65 y71-79. The back is
  shifted down 3px so both ground feet share a line.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn. NW back mirrored.
  E is screen-right, W is screen-left, N is the back.
  Attack 5-8 swaps to that direction's opposite still at equal scale
  (S becomes N). No width squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_seismitoad_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0537_seismitoad/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0537_seismitoad/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0537_seismitoad"
VERSION = "seismitoad_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 192
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 24
FRONT_FOOT = 82
BACK_FOOT = 79
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. S (front mirrored) turns into N
# (back as drawn) by swapping kind and the mirror flag.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Strike angles this script already draws. finish() will not raise a frame past them.
PEAKS = {"hand": 34.0, "hand_far": 22.0}
FRONT_PARTS = ("hand_l", "hand_r", "foot_l", "foot_r", "torso")
BACK_PARTS = ("hand_r", "foot_l", "foot_r", "torso")

INK = (0, 0, 0)
ORANGE = (238, 90, 57)
ORANGE_D = (180, 57, 32)
CYAN = (8, 148, 213)
SILVER = (197, 197, 197)
MOUTH_COLS = (ORANGE, ORANGE_D)

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
    """4x nearest in-plane rotation. Positive is clockwise. No width squash."""
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


def paint_cushion(layer, allowed, steps, dirs=("up", "down", "left", "right")):
    """Grow this layer's own colours into allowed pixels. Hidden at rest."""
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        grown = []
        if "up" in dirs:
            g = np.zeros_like(src)
            g[:-1] = src[1:]
            grown.append(g)
        if "down" in dirs:
            g = np.zeros_like(src)
            g[1:] = src[:-1]
            grown.append(g)
        if "left" in dirs:
            g = np.zeros_like(src)
            g[:, :-1] = src[:, 1:]
            grown.append(g)
        if "right" in dirs:
            g = np.zeros_like(src)
            g[:, 1:] = src[:, :-1]
            grown.append(g)
        for g in grown:
            take = (acc[..., 3] < 128) & allowed & (g[..., 3] >= 128)
            acc[take] = g[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


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


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    hand_l = op & (xx <= 26) & (yy >= 49) & (yy <= 68)
    hand_r = op & (xx >= 72) & (yy >= 42) & (yy <= 65)
    foot_l = op & (yy >= 70) & (xx <= 42)
    foot_r = op & (yy >= 71) & (xx >= 58)
    used = hand_l | hand_r | foot_l | foot_r
    torso = op & ~used
    masks = {
        "hand_l": hand_l, "hand_r": hand_r,
        "foot_l": foot_l, "foot_r": foot_r, "torso": torso,
    }
    check_partition(op, masks)
    mouth = any_color(img, MOUTH_COLS)
    eyes = is_color(img, SILVER)
    if int(mouth.sum()) != 25 or int((mouth & torso).sum()) != 25:
        raise SystemExit("mouth left the face")
    if int(eyes.sum()) != 2 or int((eyes & torso).sum()) != 2:
        raise SystemExit("eyes left the face")
    my, mx = np.nonzero(mouth)
    ey, ex = np.nonzero(eyes)
    if int(mx.min()) >= float(ex.mean()):
        raise SystemExit("mouth tip is not left of the eyes")
    if int(np.nonzero(foot_r)[0].max()) != FRONT_FOOT:
        raise SystemExit("front ground foot moved")
    if int(hand_l.sum()) < 180 or int(hand_r.sum()) < 120:
        raise SystemExit("front hands look wrong")
    if not (hand_l & dilate(torso, 1)).any() or not (hand_r & dilate(torso, 1)).any():
        raise SystemExit("a front hand does not touch the body")
    if not (foot_l & dilate(torso, 1)).any() or not (foot_r & dilate(torso, 1)).any():
        raise SystemExit("a front foot does not touch the body")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    if any_color(img, MOUTH_COLS).any() or is_color(img, SILVER).any():
        raise SystemExit("back still grew a face")
    hand_r = op & (xx >= 76) & (yy >= 54) & (yy <= 66)
    foot_l = op & (yy >= 71) & (xx <= 36)
    foot_r = op & (yy >= 71) & (xx >= 48)
    used = hand_r | foot_l | foot_r
    torso = op & ~used
    masks = {"hand_r": hand_r, "foot_l": foot_l, "foot_r": foot_r, "torso": torso}
    check_partition(op, masks)
    if int(np.nonzero(foot_r)[0].max()) != BACK_FOOT:
        raise SystemExit("back ground foot moved")
    if int(hand_r.sum()) < 40:
        raise SystemExit("back fist looks wrong")
    if not (hand_r & dilate(torso, 1)).any():
        raise SystemExit("back fist does not touch the body")
    if not (foot_l & dilate(torso, 1)).any() or not (foot_r & dilate(torso, 1)).any():
        raise SystemExit("a back foot does not touch the body")
    return masks


def place(arr, y_shift):
    y0 = OFF_Y + y_shift
    x0 = OFF_X
    if arr.ndim == 2:
        out = np.zeros((CELL_H, CELL_W), arr.dtype)
        h, w = arr.shape
    else:
        out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
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
    near = d <= d.min() + 6.0
    return float(xs[near].mean()), float(ys[near].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs - pivot[0]) ** 2 + (ys - pivot[1]) ** 2
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


def facing_of(kind, mirror):
    if kind == "front":
        return 1.0 if mirror else -1.0
    return 1.0 if not mirror else -1.0


def mouth_anchor(canvas, facing):
    mouth = any_color(canvas, MOUTH_COLS)
    if not mouth.any():
        return None
    ys, xs = np.nonzero(mouth)
    i = int(xs.argmax()) if facing > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


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
    facing = facing_of(kind, mirror)
    torso_m = placed["torso"]
    hands = [n for n in ("hand_l", "hand_r") if n in placed]
    feet = [n for n in ("foot_l", "foot_r") if n in placed]
    for name in hands:
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, 5), 5
        )
    for name in feet:
        # Foot colours grown up under the body, covered by the torso at rest.
        layers[name] = paint_cushion(layers[name], torso_m, 3, dirs=("up",))
    pivots = {}
    signs = {}
    for name in hands:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    for name in feet:
        pivots[name] = pivot_of(layers[name], layers["torso"])
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
        "mouth": mouth_anchor(canvas, facing),
        "hands": hands, "feet": feet,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        hand=0.0, hand_far=0.0,
        hand_mode="up", hand_far_mode="up",
        step_l=(0, 0), step_r=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0, ring=-1,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        p["bdy"] = -1 if wave(f, n, 1.0) > 0.55 else 0
        p["hand"] = wave(f, n, 6.0, phase=0.4)
        p["hand_far"] = wave(f, n, 5.0, phase=0.9)
        p["hand_mode"] = "up"
        p["hand_far_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * sb)), -int(round(3 * max(0.0, sb))))
        p["bdy"] = -1 if abs(math.sin(u)) > 0.7 else 0
        p["hand"] = wave(f, n, 8.0, phase=0.2)
        p["hand_far"] = wave(f, n, 8.0, phase=0.2 + math.pi)
        p["hand_mode"] = "fwd"
        p["hand_far_mode"] = "fwd"
    elif action == "attack":
        # 0 rest. 1-4 mouth wedge and ring, hands rise. 5-8 cleared after
        # finish() so the opposite still stays at equal scale. 9-12 both
        # hands strike and the ground foot stomps. 13-15 recover. 15 is the still.
        hand = [0, 12, 18, 22, 16, 0, 0, 0, 0, 34, 34, 34, 24, 10, 4, 0]
        hand_far = [0, 8, 14, 18, 14, 0, 0, 0, 0, 22, 22, 18, 12, 6, 2, 0]
        bdy = [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        p["hand"] = float(hand[f])
        p["hand_far"] = float(hand_far[f])
        p["bdy"] = int(bdy[f])
        p["hand_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["hand_far_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["turned"] = 5 <= f <= 8
        if f == 9:
            p["step_r"] = (1, -2)
        elif f == 10:
            p["step_r"] = (2, -3)
        elif f == 11:
            p["step_r"] = (1, 0)
        if 1 <= f <= 4:
            p["ring"] = f - 1
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.85, 0.62, 0.42, 0.26, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["hand"] = 10.0 * env
        p["hand_far"] = 8.0 * env
        p["hand_mode"] = "up"
        p["hand_far_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["hand"] = 0.0
        p["hand_far"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["spark"] = 0
        p["ring"] = -1
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["hand_mode"] = "up"
        p["hand_far_mode"] = "up"
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


def body_delta(rig, p):
    fc = rig["facing"]
    return int(round(fc * p["bdx"])), int(round(p["bdy"]))


def limb_delta(rig, name, p):
    """Angle and pixel shift for one part. Feet stay on the ground when the body bobs."""
    fc = rig["facing"]
    bdx, bdy = body_delta(rig, p)
    if name == "foot_l":
        sx, sy = p["step_l"]
        return 0.0, int(round(fc * sx)), int(sy)
    if name == "foot_r":
        sx, sy = p["step_r"]
        return 0.0, int(round(fc * sx)), int(sy)
    if rig["kind"] == "front":
        if name == "hand_l":
            mode, amp = p["hand_mode"], p["hand"]
        else:
            mode, amp = p["hand_far_mode"], p["hand_far"]
    else:
        mode, amp = p["hand_mode"], p["hand"]
    ang = part_angle(rig, name, mode, amp)
    return ang, bdx, bdy


def part_layer(rig, name, p):
    if name not in rig["layers"]:
        return None
    ang, dx, dy = limb_delta(rig, name, p)
    pivot = rig["pivots"].get(name, (0.0, 0.0))
    return moved(rig["layers"][name], ang, pivot, dx, dy)


def compose(rig, p):
    bdx, bdy = body_delta(rig, p)
    out = np.zeros_like(rig["canvas"])
    for name in rig["feet"]:
        paste(out, part_layer(rig, name, p))
    paste(out, shift(rig["layers"]["torso"], bdx, bdy))
    for name in rig["hands"]:
        paste(out, part_layer(rig, name, p))
    return out


def moving(p):
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["hand"]) >= 0.45 or abs(p["hand_far"]) >= 0.45:
        return True
    if p["step_l"] != (0, 0) or p["step_r"] != (0, 0):
        return True
    return False


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["hand"]) >= 0.45 or abs(p["hand_far"]) >= 0.45:
        return True
    return False


def flash_silver(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK)
    out[op & ~keep, :3] = np.array(SILVER, np.uint8)
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
    if m is None or not m.any():
        return None
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if facing > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def roar(img, origin, fc, age):
    """Mouth wedge and a ring, in this still's own colours."""
    ox, oy = origin
    cols = (ORANGE, ORANGE_D, INK, CYAN)
    for i, (dx, dy) in enumerate(((2, 0), (3, -1), (3, 1), (4, 0), (5, -1), (5, 1), (4, -2), (4, 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])
    rad = 8 + age * 3
    prev = None
    for deg in range(-50, 51, 8):
        a = math.radians(deg)
        x = ox + fc * (6 + rad * math.cos(a))
        y = oy + rad * math.sin(a) * 0.55
        put(img, x, y, cols[(deg // 8) % 4])
        if prev is not None:
            put(img, (x + prev[0]) / 2, (y + prev[1]) / 2, cols[age % 4])
        prev = (x, y)


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (ORANGE, CYAN, SILVER, INK)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r), (2, r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def stomp_mark(img, origin, fc, age):
    ox, oy = origin
    cols = (CYAN, INK, ORANGE_D)
    for i, dx in enumerate(range(-6, 7, 3)):
        put(img, ox + dx, oy + 1, cols[i % 3])
        put(img, ox + fc * (3 + age) + dx, oy + 1, cols[(i + age) % 3])


def strike_hand(rig):
    if rig["kind"] == "front" and "hand_l" in rig["layers"]:
        return "hand_l"
    if "hand_r" in rig["layers"]:
        return "hand_r"
    return None


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name in list(rig["hands"]) + list(rig["feet"]):
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
        mirror = not mirror
    return rigs[(kind, mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    if not moving(p) and abs(p["lean"]) <= 0.4:
        img = rig["canvas"].copy()
    else:
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
        elif moving(p):
            body = fill_pinholes(body)
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_silver(img), palette)
    if p["ring"] >= 0 and not p["turned"] and rig["mouth"] is not None:
        ox, oy = rig["mouth"]
        bdx, bdy = body_delta(rig, p)
        roar(img, (ox + bdx + fc * 2, oy + bdy), fc, p["ring"])
        img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        name = strike_hand(rig)
        tip = None
        if name is not None:
            tip = facing_tip(part_layer(rig, name, p), fc)
        if tip is not None:
            impact(img, (tip[0] + fc * 2, tip[1]), fc, p["spark"])
        foot = part_layer(rig, "foot_r", p)
        if foot is not None:
            sole = facing_tip(foot, fc)
            if sole is not None:
                stomp_mark(img, (sole[0], FOOT_Y), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    h, w = op.shape
    seen = np.zeros_like(op, bool)
    best = 0
    ys, xs = np.nonzero(op)
    for y0, x0 in zip(ys, xs):
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
                if 0 <= ny < h and 0 <= nx < w and op[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if n > best:
            best = n
    return best / float(op.sum())


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
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 12)
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
    if rigs[("front", False)]["facing"] != -1.0 or rigs[("front", True)]["facing"] != 1.0:
        raise SystemExit("front facing locked to the wrong side")
    if rigs["S"]["mouth"] is None or rigs["N"]["mouth"] is not None:
        raise SystemExit("mouth is on the wrong view")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "hand_l": (40, 180, 255, 255),
        "hand_r": (180, 70, 220, 255),
        "foot_l": (40, 210, 80, 255),
        "foot_r": (80, 255, 180, 255),
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
    return n


def publish(bounds, sprite_h):
    import gen5_publish

    entry = {
        "dex": 537, "form": "normal", "variant": "seismitoad", "gen": "gen5",
        "dir": "monsterSprite/gen5/0537_seismitoad/normal", "sheetDir": "agent-sprite-forge",
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
    args = ap.parse_args()

    save_peaks(Path(__file__).stem, PEAKS)
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    palette = locked_palette(SRC_DIR / "front.png", SRC_DIR / "back.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, flush=True)
    print("parts", {k: int(v.sum()) for k, v in fm.items()},
          {k: int(v.sum()) for k, v in bm.items()}, "palette", len(palette), flush=True)

    rest_diff = {}
    compose_diff = {}
    blank = base_pose()
    for d in DIRS:
        rest_diff[d] = 0
        raw = compose(rigs[d], blank)
        compose_diff[d] = int(np.any(raw != rigs[d]["canvas"], axis=2).sum())
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
        r3 = synth(rigs, d, "attack", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r3 != rigs[d]["canvas"], axis=2).sum())
    print("rest diff", rest_diff, flush=True)
    print("compose diff", compose_diff, flush=True)
    if any(rest_diff.values()) or any(compose_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    turn_diff = {}
    for d in DIRS:
        p = pose("attack", 6)
        rig = rig_for(rigs, d, p)
        img = synth(rigs, d, "attack", 6, palette, effects=True)
        turn_diff[d] = int(np.any(img != rig["canvas"], axis=2).sum())
        for f in (5, 7, 8):
            img_f = synth(rigs, d, "attack", f, palette, effects=True)
            rig_f = rig_for(rigs, d, pose("attack", f))
            turn_diff[d] += int(np.any(img_f != rig_f["canvas"], axis=2).sum())
    print("turn diff", turn_diff, flush=True)
    if any(turn_diff.values()):
        raise SystemExit("attack 5-8 is not the opposite still")
    if rig_for(rigs, "S", pose("attack", 6)) is not rigs["N"]:
        raise SystemExit("S turn is not the N still")

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
                blank_tile = np.zeros_like(frames[0])
                while len(frames) < 16:
                    frames.append(blank_tile)
                rows.append(tile_grid(frames, 16))
            sheet = np.concatenate(rows, axis=0)
            z = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
            Image.fromarray(z, "RGBA").save(work / f"samples_{d}.png")
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["hand"], 1), round(p["hand_far"], 1),
                        p["hand_mode"], p["turned"], p["spin"], p["ring"], p["bdy"], p["step_r"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        report = {
            "rest_diff_max": max(rest_diff.values()),
            "compose_diff_max": max(compose_diff.values()),
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
        "generator": "forge_gen5_seismitoad_actions", "version": VERSION, "key": KEY,
        "dex": 537, "variant": "seismitoad", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "heavy biped toad (two hands, two feet, orange mouth, cranial bumps, no tail)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (mouth tip x26 < eye mean x34.5); back fist is on the right and has no mouth",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
