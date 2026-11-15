#!/usr/bin/env python3
"""Pansear normal (0513) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x27-68 y27-69, foot y69, 13 colours,
alpha 0/255, one piece):
  Eyes (white and grey) mean x 41.0, span x35-46 y47-51.
  Face cream is y49-54 and its right edge is x48, right of the eyes.
  The still faces screen-right. No mouth is drawn: the black under the chin
  is the outline, so the roar does not invent a mouth, a wedge, or a ring.
  It lifts the flame crest and the two ears that are already on the head.
  Left ear (orange core) x27-37 y36-49. Right ear x51-61 y38-47.
  Raised cream hand x54-61 y48-58. Hanging cream hand x31-39 y57-66.
  Feet: left x35-49 y65-68, right x50-55 y65-69. Tail curl x56-68 y55-66.
normal/back.png (bbox x27-68 y26-69, foot y69) has no eyes and no orange.
  Flame, two ears, left tail curl x27-40 y58-67, two feet. No arms are drawn.

Directions, still faces right:
  S SE E front as drawn (screen-right). W SW front mirrored (screen-left).
  NE back as drawn. N NW back mirrored. E is screen-right, W screen-left.
Attack 5-8 swaps to the opposite view at equal scale (S turns into the N back).
No width squash, no in-plane body spin. The strike is the raised hand, the
tail, and a short step of the near foot. Hurt leans at most 38 degrees.

  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_pansear_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0513_pansear/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0513_pansear/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0513_pansear"
VERSION = "pansear_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 176, 176
OFF_X = (CELL_W - 96) // 2
OFF_Y = 32
SRC_FOOT = 69
FOOT_Y = OFF_Y + SRC_FOOT
# kind, mirror. Front as drawn looks screen-right.
VIEW_OF = {
    "S": ("front", False), "SE": ("front", False), "E": ("front", False),
    "W": ("front", True), "SW": ("front", True),
    "NE": ("back", False), "N": ("back", True), "NW": ("back", True),
}
MAX_LEAN = 38.0
# finish() may raise frames 5-12 toward these. The hand strike is authored higher.
PEAKS = {"flame": 12.0, "ear": 12.0, "arm": 18.0, "tail": 12.0}
ORDER = ("torso", "foot_l", "foot_r", "tail", "arm_l", "flame", "ear_l", "ear_r", "arm_r")

INK = (0, 0, 0)
RED = (230, 82, 65)
RED_D = (189, 65, 57)
MAROON = (123, 49, 49)
ORANGE = (246, 148, 65)
CREAM = (255, 205, 123)
TAN = (205, 164, 90)
GOLD = (164, 123, 57)
BROWN = (115, 82, 32)
WHITE = (255, 255, 255)
GREY = (82, 82, 82)
GREY_M = (131, 131, 131)
GREY_L = (197, 197, 197)
EYE_COLS = (WHITE, GREY, GREY_M, GREY_L)
SKIN_COLS = (CREAM, TAN, GOLD, BROWN)

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


def put(img, x, y, col, over=False):
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 1 <= x < w - 1 and 1 <= y < h - 1 and (over or img[y, x, 3] < 128):
        img[y, x, :3] = col
        img[y, x, 3] = 255


def paint_cushion(layer, allowed, steps, dirs=("up", "down", "left", "right")):
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
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def check_partition(op, masks):
    acc = np.zeros_like(op)
    for name, m in masks.items():
        if (m & acc).any():
            raise SystemExit(f"overlap in {name}")
        acc |= m
    if int(acc.sum()) != int(op.sum()) or not np.array_equal(acc, op):
        raise SystemExit(f"parts do not partition the still: {int(acc.sum())} vs {int(op.sum())}")


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    orange = is_color(img, ORANGE)
    cream = is_color(img, CREAM)
    el = orange & (xx < 45)
    ear_l = dilate(el, 3) & op & (xx <= 38) & (yy >= 35) & (yy <= 49)
    ear_l &= ~((yy >= 47) & (xx >= 35))
    er = orange & (xx > 45)
    ear_r = dilate(er, 2) & op & (xx >= 51) & (yy <= 49) & (yy >= 36)
    ear_r &= ~cream
    ear_r &= ~(yy >= 48)
    flame = op & (yy <= 36) & (xx >= 36) & (xx <= 53)
    flame &= ~ear_l & ~ear_r
    hand = cream & (yy >= 48) & (xx >= 54)
    arm_r = hand | (dilate(hand, 1) & op & (xx >= 54) & (yy >= 48) & (yy <= 60))
    arm_r &= ~ear_r & ~flame
    hang = cream & (yy >= 58) & (xx <= 42)
    arm_l = hang | (dilate(hang, 1) & op & (xx <= 42) & (yy >= 57) & (yy <= 66))
    arm_l &= ~ear_l
    tail = op & (yy >= 55) & (yy <= 66) & (xx >= 60)
    tail |= op & (yy >= 64) & (yy <= 66) & (xx >= 56) & (xx <= 62)
    tail &= ~arm_r & ~arm_l
    low = op & (yy >= 65) & ~tail & ~arm_l & ~arm_r & ~ear_l
    foot_l = low & (xx <= 49)
    foot_r = low & (xx >= 50)
    used = flame | ear_l | ear_r | arm_l | arm_r | tail | foot_l | foot_r
    torso = op & ~used
    masks = {
        "flame": flame, "ear_l": ear_l, "ear_r": ear_r, "arm_l": arm_l, "arm_r": arm_r,
        "tail": tail, "foot_l": foot_l, "foot_r": foot_r, "torso": torso,
    }
    check_partition(op, masks)
    eye = any_color(img, EYE_COLS)
    if int((eye & torso).sum()) != int(eye.sum()) or eye.sum() < 16:
        raise SystemExit("eyes left the head")
    ex = float(np.nonzero(eye)[1].mean())
    face = any_color(img, SKIN_COLS) & (yy >= 49) & (yy <= 54) & (xx <= 52)
    if not face.any() or float(np.nonzero(face)[1].max()) <= ex:
        raise SystemExit("face tip is not right of the eyes")
    for name, m in masks.items():
        if name != "torso" and int((m & dilate(torso, 1)).sum()) < 3:
            raise SystemExit(f"{name} does not meet the torso")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    ear_l = op & (xx <= 43) & (yy >= 35) & (yy <= 49)
    ear_r = op & (xx >= 60) & (yy >= 35) & (yy <= 49)
    flame = op & (yy <= 35) & (xx >= 44) & (xx <= 60)
    flame &= ~ear_l & ~ear_r
    tail = op & (xx <= 40) & (yy >= 58)
    tail &= ~ear_l
    foot_r = op & (xx >= 55) & (yy >= 56) & (yy <= 69)
    foot_l = op & (xx >= 38) & (xx <= 52) & (yy >= 60) & (yy <= 68)
    foot_l &= ~tail & ~foot_r
    foot_r &= ~tail
    used = flame | ear_l | ear_r | tail | foot_l | foot_r
    torso = op & ~used
    masks = {
        "flame": flame, "ear_l": ear_l, "ear_r": ear_r,
        "tail": tail, "foot_l": foot_l, "foot_r": foot_r, "torso": torso,
    }
    check_partition(op, masks)
    if any_color(img, EYE_COLS).any() or is_color(img, ORANGE).any():
        raise SystemExit("back still grew a face")
    for name, m in masks.items():
        if name != "torso" and int((m & dilate(masks["torso"], 1)).sum()) < 2:
            raise SystemExit(f"back {name} does not meet the torso")
    return masks


def place(arr):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    out[OFF_Y:OFF_Y + 96, OFF_X:OFF_X + 96] = arr
    return out


def joint_band(part, torso, depth):
    touch = part & dilate(torso, 2)
    if not touch.any():
        touch = part & dilate(torso, 5)
    band = touch.copy()
    for _ in range(depth):
        band = dilate(band, 1) & part
    return band


def pivot_of(layer, anchor_layers):
    part = layer[..., 3] >= 128
    if not part.any():
        return 0.0, 0.0
    anchor = np.zeros(part.shape, bool)
    for a in anchor_layers:
        anchor |= a[..., 3] >= 128
    touch = part & dilate(anchor, 2)
    if not touch.any():
        touch = part & dilate(anchor, 6)
    if not touch.any():
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ay, ax = np.nonzero(anchor)
    cx, cy = float(ax.mean()), float(ay.mean())
    d = (xs - cx) ** 2 + (ys - cy) ** 2
    near = d <= d.min() + 4.0
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


def build_view(img, masks, kind, mirror):
    canvas = place(img)
    placed = {k: place(v) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    limbs = [k for k in ORDER if k in placed and k != "torso"]
    depth = {"flame": 4, "ear_l": 4, "ear_r": 4, "arm_l": 4, "arm_r": 4, "tail": 5,
             "foot_l": 3, "foot_r": 3}
    stump = np.zeros_like(torso_m)
    for name in limbs:
        stump |= joint_band(placed[name], torso_m, depth.get(name, 4))
    layers["torso"] = paint_cushion(layers["torso"], stump, 8)
    facing = 1.0 if not mirror else -1.0
    anchors = {
        "flame": ("torso",), "ear_l": ("torso",), "ear_r": ("torso",),
        "arm_l": ("torso",), "arm_r": ("ear_r", "torso"), "tail": ("torso",),
        "foot_l": ("torso",), "foot_r": ("torso",),
    }
    pivots = {}
    signs = {}
    for name in limbs:
        anch = [layers[a] for a in anchors[name] if a in layers]
        pivots[name] = pivot_of(layers[name], anch)
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    foot_cx = {}
    for name in ("foot_l", "foot_r"):
        fxs = np.nonzero(placed[name])[1]
        foot_cx[name] = float(fxs.mean()) if len(fxs) else pivots["center"][0]
    if facing > 0:
        near = "foot_r" if foot_cx["foot_r"] >= foot_cx["foot_l"] else "foot_l"
    else:
        near = "foot_l" if foot_cx["foot_l"] <= foot_cx["foot_r"] else "foot_r"
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas, "near_foot": near,
        "limbs": limbs,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, hand_dx=0, flame=0.0, ear=0.0, arm=0.0, arm_far=0.0, tail=0.0,
        ear_mode="up", arm_mode="fwd", tail_mode="fwd", oppose_arms=False,
        step_l=(0, 0), step_r=(0, 0), kick=(0, 0), lean=0.0, knock=0,
        flash=False, spark=0, turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        breath = wave(f, n, 1.0, phase=0.4)
        p["bdy"] = -1 if breath > 0.45 else 0
        p["flame"] = wave(f, n, 6.0, phase=0.2, cycles=2.0)
        p["ear"] = wave(f, n, 5.0, phase=0.9)
        p["arm"] = wave(f, n, 4.0, phase=0.3)
        p["tail"] = wave(f, n, 7.0, phase=1.1)
        p["oppose_arms"] = True
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        p["bdy"] = -1 if abs(math.sin(2 * u)) > 0.72 else 0
        p["flame"] = 5.0 * math.sin(2 * u)
        p["ear"] = 4.0 * sa
        p["arm"] = 12.0 * sa
        p["tail"] = 8.0 * math.sin(u + 1.2)
        p["step_l"] = (int(round(2.0 * max(0.0, sa))), -int(round(2.0 * max(0.0, sa))))
        p["step_r"] = (int(round(2.0 * max(0.0, -sa))), -int(round(2.0 * max(0.0, -sa))))
        p["oppose_arms"] = True
    elif action == "attack":
        # 0 rest. 1-4 roar (crest and ears; no mouth). 5-8 back view.
        # 9 wind-up. 10-12 hand strike and tail lash. 13-15 recover.
        p["flame"] = [0, 6, 10, 12, 12, 12, 12, 12, 12, 12, 12, 10, 8, 3, 0, 0][f]
        p["ear"] = [0, 4, 8, 12, 12, 12, 12, 12, 12, 12, 10, 8, 6, 2, 0, 0][f]
        p["arm"] = [0, 4, 8, 12, 14, -18, -18, -18, -18, 18, 32, 32, 20, 8, 2, 0][f]
        p["arm_far"] = [0, -4, -6, -8, -8, -8, -8, -8, -6, -10, -12, -12, -6, -2, 0, 0][f]
        p["tail"] = [0, 4, 8, 12, 12, 12, 12, 12, 12, -12, -16, -16, -10, -4, 0, 0][f]
        # The raised hand already points forward, so rotation alone barely
        # extends it. A few pixels of reach make the strike read.
        p["hand_dx"] = [0, 0, -1, -1, -1, 0, 0, 0, 0, 1, 3, 4, 2, 1, 0, 0][f]
        if f in (2, 3, 4):
            p["bdx"] = -1
        elif f in (10, 11, 12):
            p["bdx"] = 1
        if f in (3, 4, 10):
            p["bdy"] = -1
        if f in (10, 11):
            p["kick"] = (2, -1)
        if f in (11, 12):
            p["spark"] = 2 if f == 11 else 1
        p["turned"] = 5 <= f <= 8
    else:
        env = [0.0, 0.55, 0.90, 1.0, 0.85, 0.62, 0.42, 0.26, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["flame"] = 10.0 * env
        p["ear"] = 8.0 * env
        p["arm"] = 10.0 * env
        p["arm_far"] = 8.0 * env
        p["tail"] = 12.0 * env
        p["arm_mode"] = "back"
        p["tail_mode"] = "up"
        p["flash"] = f == 1
        p["spark"] = 1 if f in (2, 3) else 0
    return finish(p, action, f, __file__)


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


def part_transform(rig, p, name):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    if name == "torso":
        return shift(L["torso"], bdx, bdy)
    if name.startswith("foot"):
        fwd, lift = p["step_l"] if name == "foot_l" else p["step_r"]
        if name == rig["near_foot"]:
            kf, ky = p["kick"]
            fwd += kf
            lift += ky
        return shift(L[name], bdx + int(round(fc * fwd)), int(lift))
    if name == "flame":
        ang = part_angle(rig, name, "up", p["flame"])
    elif name.startswith("ear"):
        ang = part_angle(rig, name, p["ear_mode"], p["ear"])
    elif name == "arm_r":
        ang = part_angle(rig, name, p["arm_mode"], p["arm"])
        extra = int(round(fc * p.get("hand_dx", 0)))
        return moved(L[name], ang, rig["pivots"][name], bdx + extra, bdy)
    elif name == "arm_l":
        amp = -p["arm"] if p.get("oppose_arms") else p["arm_far"]
        ang = part_angle(rig, name, p["arm_mode"], amp)
    elif name == "tail":
        ang = part_angle(rig, name, p["tail_mode"], p["tail"])
    else:
        ang = 0.0
    return moved(L[name], ang, rig["pivots"][name], bdx, bdy)


def compose(rig, p):
    out = np.zeros_like(rig["canvas"])
    for name in ORDER:
        if name not in rig["layers"]:
            continue
        paste(out, part_transform(rig, p, name))
    return out


def spark_at(img, origin, fc, age):
    ox, oy = origin
    r = 3 + age
    cols = (ORANGE, CREAM, WHITE)
    for i, (dx, dy) in enumerate(((r, 0), (0, -r), (-2, -1), (1, 2), (fc * 2, -2))):
        put(img, ox + dx, oy + dy, cols[i % 3])


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


def limb_tip(layer, facing):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    if facing > 0:
        i = int(xs.argmax())
    else:
        i = int(xs.argmin())
    return int(xs[i]), int(ys[i])


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rigs[d]
    if p["turned"]:
        kind, mirror = VIEW_OF[d]
        rig = rigs[("back" if kind == "front" else "front", not mirror)]
    body = compose(rig, p)
    moved_any = (
        abs(p["lean"]) > 0.4 or abs(p["flame"]) >= 0.45 or abs(p["ear"]) >= 0.45
        or abs(p["arm"]) >= 0.45 or abs(p["arm_far"]) >= 0.45 or abs(p["tail"]) >= 0.45
    )
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], piv)
        moved_any = True
    if moved_any:
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    if p["spark"] and action == "attack" and not p["turned"] and "arm_r" in rig["layers"]:
        hand = part_transform(rig, p, "arm_r")
        hx, hy = limb_tip(hand, fc)
        spark_at(img, (hx + fc * 2, hy), fc, p["spark"])
    elif p["spark"] and action == "hurt":
        cx, cy = rig["pivots"]["center"]
        spark_at(img, (cx - fc * 8, cy - 10), -fc, 1)
    return snap(img, palette)


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    h, w = op.shape
    seen = np.zeros_like(op, bool)
    best = 0
    total = int(op.sum())
    for y0, x0 in zip(*np.nonzero(op)):
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
    return best / float(total)


def disconnected_limbs(rig, p):
    body = compose(rig, p)
    lost = []
    for name in rig["limbs"]:
        layer = part_transform(rig, p, name)
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


def crop_box(canvas, pad=6):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad - 8)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 10)
    x0 = max(0, int(xs.min()) - pad - 16)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 16)
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
        for f in range(FRAMES[act]):
            t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            tiles.append(t[y0:y1, x0:x1])
        for _ in range(16 - FRAMES[act]):
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
    if int(np.nonzero(front[..., 3])[0].max()) != SRC_FOOT:
        raise SystemExit("front foot line changed")
    if int(np.nonzero(back[..., 3])[0].max()) != SRC_FOOT:
        raise SystemExit("back foot line changed")
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
        "flame": (220, 40, 40, 255), "ear_l": (240, 160, 20, 255), "ear_r": (240, 100, 180, 255),
        "arm_l": (40, 90, 220, 255), "arm_r": (20, 180, 80, 255), "tail": (160, 40, 220, 255),
        "foot_l": (240, 220, 40, 255), "foot_r": (40, 200, 210, 255), "torso": (180, 180, 180, 255),
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
        "dex": 513, "form": "normal", "variant": "pansear", "gen": "gen5",
        "dir": "monsterSprite/gen5/0513_pansear/normal", "sheetDir": "agent-sprite-forge",
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
    palette = locked_palette(SRC_DIR / "front.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    print("pivots S", {k: tuple(round(x, 1) for x in v) for k, v in rigs["S"]["pivots"].items()},
          "near", rigs["S"]["near_foot"], flush=True)

    rest_diff = {}
    for d in DIRS:
        r = synth(rigs, d, "idle", 0, palette, effects=False)
        rest_diff[d] = int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
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
                ("attack", (0, 2, 4, 5, 6, 8, 9, 10, 11, 12, 15)),
                ("hurt", (0, 1, 3, 6, 11)),
            ):
                for f in fs:
                    frames.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
            sheet = tile_grid(frames, len(frames))
            z = np.repeat(np.repeat(sheet, 3, 0), 3, 1)
            Image.fromarray(z, "RGBA").save(work / f"samples_{d}.png")
        for d in ("S", "N", "W"):
            row = [synth(rigs, d, "attack", f, palette)[y0:y1, x0:x1] for f in range(16)]
            z = np.repeat(np.repeat(tile_grid(row, 16), 3, 0), 3, 1)
            Image.fromarray(z, "RGBA").save(work / f"attack_{d}.png")
            row = [synth(rigs, d, "walk", f, palette)[y0:y1, x0:x1] for f in range(16)]
            z = np.repeat(np.repeat(tile_grid(row, 16), 3, 0), 3, 1)
            Image.fromarray(z, "RGBA").save(work / f"walk_{d}.png")
        lost = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rigs[d]
                    if p["turned"]:
                        kind, mirror = VIEW_OF[d]
                        rig = rigs[("back" if kind == "front" else "front", not mirror)]
                    bad = disconnected_limbs(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        print("disconnected", lost[:30], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["flame"], 1), round(p["ear"], 1), round(p["arm"], 1),
                        round(p["tail"], 1), p["turned"], p["spin"]))
        print("attack", atk, flush=True)
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
                    bodies.append(largest_fraction(t))
    bounds = []
    for di in range(8):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        ys, xs = np.nonzero(t[..., 3] == 255)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
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
        "generator": "forge_gen5_pansear_actions", "version": VERSION, "key": KEY,
        "dex": 513, "variant": "pansear", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped flame striker (crest, two ears, raised hand, hanging hand, two feet, curled tail)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-right (face cream xmax > eye mean x); no mouth is drawn",
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
