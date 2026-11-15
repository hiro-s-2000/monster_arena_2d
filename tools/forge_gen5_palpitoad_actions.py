#!/usr/bin/env python3
"""Palpitoad normal (0536) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x29-68 y25-71, foot y71, 15 colours,
alpha 0/255, one piece):
  Eye whites (39,42) (40,43) (41,44) (52,44) (50,45) (51,45), mean x 45.5.
  Mouth tan/brown sits at y43-45, xmin 38, left of the eyes.
  Light-grey tail stub x57-68 y62-66 is the rear, on the right.
  The still faces screen-left. Crown is the mint dome y25-36.
  Right cheek disc is the tympanum x59-66 y34-48, not an arm.
  Two blue feet, three toes each. Left toes end y70, right toes end y71.
normal/back.png (bbox x29-66 y26-69, foot y69) has no cream and no mouth.
  White tail x30-38 y55-59 is on the left. Blue back, mint bumps, two feet.

Directions, still faces left (the reverse of a right-facing still):
  W SW front as drawn (screen-left). S SE E front mirrored (screen-right).
  N NW back as drawn. NE back mirrored.
  E is screen-right, W is screen-left. N is the back row.
Attack 5-8 swaps to that direction's back at equal scale (S turns into N).
No width squash and no in-plane body spin. Roar uses the mouth wedge and a
ring only on views that have the mouth. The strike is the two feet and the
tail. Hurt leans back at most 38 degrees.

  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_palpitoad_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0536_palpitoad/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0536_palpitoad/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0536_palpitoad"
VERSION = "palpitoad_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 176, 176
OFF_X = (CELL_W - 96) // 2
OFF_Y = 32
SRC_FOOT = 71
FOOT_Y = OFF_Y + SRC_FOOT
# Front as drawn looks screen-left, so screen-left rows keep it.
VIEW_OF = {
    "W": ("front", False), "SW": ("front", False),
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "N": ("back", False), "NW": ("back", False),
    "NE": ("back", True),
}
MAX_LEAN = 38.0
# finish() may raise frames 5-12 toward these. Authored strikes already meet them.
PEAKS = {"foot_lead": 30.0, "foot_trail": 18.0, "tail": 22.0}
ORDER = ("torso", "foot_l", "foot_r", "tail", "tymp", "crown")

INK = (0, 0, 0)
GREY = (98, 98, 98)
DGREY = (65, 65, 65)
VDGREY = (41, 41, 41)
LGREY = (197, 197, 197)
WHITE = (255, 255, 255)
CREAM = (255, 222, 189)
TAN = (197, 156, 131)
BROWN = (148, 90, 74)
MINT = (131, 230, 213)
CYAN = (8, 148, 213)
TEAL = (82, 164, 180)
DTEAL = (24, 98, 115)
BLUE = (41, 106, 172)
NAVY = (16, 65, 123)
RIM = (INK, DTEAL, TEAL, DGREY, VDGREY)
MOUTH_COLS = (TAN, BROWN)

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
    """4x nearest in-plane rotation. Positive moves a top pixel left. No squash."""
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


def split_feet(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise SystemExit("no feet")
    lo, hi = int(xs.min()), int(xs.max())
    span = hi - lo
    mid0 = lo + max(1, span // 3)
    mid1 = hi - max(1, span // 3)
    valley = min(range(mid0, mid1 + 1), key=lambda x: int(mask[:, x].sum()))
    xx = np.arange(mask.shape[1])[None, :]
    left = mask & (xx <= valley)
    right = mask & (xx > valley)
    if not left.any() or not right.any():
        raise SystemExit(f"foot split failed at x{valley}")
    return left, right


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    dome = is_color(img, MINT) & (yy <= 37) & (xx >= 36) & (xx <= 56)
    rim = dilate(dome, 1) & op & (yy <= 37) & any_color(img, RIM)
    crown = dome | rim
    # Right cheek disc. The cyan flank below stays on the torso.
    region = op & (xx >= 58) & (yy >= 33) & (yy <= 48)
    tymp = flood(region, region & (xx >= 63) & (yy >= 36) & (yy <= 46))
    tymp &= ~crown
    tail_box = op & (xx >= 56) & (yy >= 61) & (yy <= 67)
    tail = flood(tail_box, tail_box & is_color(img, LGREY))
    tail &= ~crown & ~tymp
    foot = op & (yy >= 66) & ~tail & ~crown & ~tymp
    foot_l, foot_r = split_feet(foot)
    used = crown | tymp | tail | foot_l | foot_r
    torso = op & ~used
    masks = {
        "crown": crown, "tymp": tymp, "tail": tail,
        "foot_l": foot_l, "foot_r": foot_r, "torso": torso,
    }
    check_partition(op, masks)
    eye = is_color(img, WHITE)
    if int((eye & torso).sum()) != int(eye.sum()) or int(eye.sum()) != 6:
        raise SystemExit("eyes left the head")
    mouth = any_color(img, MOUTH_COLS) & (yy <= 48)
    if not mouth.any():
        raise SystemExit("front mouth missing")
    ex = float(np.nonzero(eye)[1].mean())
    mx = int(np.nonzero(mouth)[1].min())
    if mx >= ex:
        raise SystemExit(f"mouth tip x{mx} is not left of the eyes {ex:.1f}")
    for name, m in masks.items():
        if name == "torso":
            continue
        if int(m.sum()) < 12:
            raise SystemExit(f"{name} too small {int(m.sum())}")
        if int((m & dilate(torso, 1)).sum()) < 2:
            raise SystemExit(f"{name} does not meet the torso")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    if any_color(img, (CREAM, TAN, BROWN)).any():
        raise SystemExit("back still grew a belly")
    # White and light-grey stub only. Flooding the box would swallow the blue rump.
    core = (is_color(img, WHITE) | is_color(img, LGREY) | is_color(img, GREY)) & (yy >= 53) & (xx <= 46)
    tail = core | (dilate(core, 1) & is_color(img, INK) & (yy >= 53) & (yy <= 64) & (xx <= 48))
    bump_box = op & (xx >= 57) & (yy >= 30) & (yy <= 45)
    tymp = flood(bump_box, bump_box & is_color(img, MINT))
    tymp &= ~tail
    foot = op & (yy >= 67) & ~tail & ~tymp
    foot_l, foot_r = split_feet(foot)
    used = tail | tymp | foot_l | foot_r
    torso = op & ~used
    masks = {"tymp": tymp, "tail": tail, "foot_l": foot_l, "foot_r": foot_r, "torso": torso}
    check_partition(op, masks)
    if is_color(img, WHITE).any() and int((is_color(img, WHITE) & tail).sum()) != int(is_color(img, WHITE).sum()):
        raise SystemExit("back white left the tail")
    tx = float(np.nonzero(tail)[1].mean())
    cx = float(np.nonzero(op)[1].mean())
    if tx >= cx:
        raise SystemExit("back tail is not on the left")
    for name, m in masks.items():
        if name == "torso":
            continue
        if int(m.sum()) < 8:
            raise SystemExit(f"back {name} too small {int(m.sum())}")
        if int((m & dilate(torso, 1)).sum()) < 1:
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
    depth = {"crown": 5, "tymp": 5, "tail": 6, "foot_l": 5, "foot_r": 5}
    stump = np.zeros_like(torso_m)
    for name in limbs:
        stump |= joint_band(placed[name], torso_m, depth.get(name, 4))
    layers["torso"] = paint_cushion(layers["torso"], stump, 8)
    # Unmirrored front faces left. Unmirrored back has the tail on the left,
    # so that rear side trails a screen-right facing.
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in limbs:
        pivots[name] = pivot_of(layers[name], [layers["torso"]])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    foot_cx = {}
    for name in ("foot_l", "foot_r"):
        fxs = np.nonzero(layers[name][..., 3] >= 128)[1]
        foot_cx[name] = float(fxs.mean()) if len(fxs) else pivots["center"][0]
    if facing > 0:
        lead = "foot_r" if foot_cx["foot_r"] >= foot_cx["foot_l"] else "foot_l"
    else:
        lead = "foot_l" if foot_cx["foot_l"] <= foot_cx["foot_r"] else "foot_r"
    trail = "foot_r" if lead == "foot_l" else "foot_l"
    mouth = any_color(canvas, MOUTH_COLS)
    if mouth.any():
        my, mx = np.nonzero(mouth)
        # The face tip is the mouth edge in the facing direction.
        if facing > 0:
            tip = mx >= np.percentile(mx, 70)
        else:
            tip = mx <= np.percentile(mx, 30)
        pivots["mouth"] = (float(mx[tip].mean()), float(my[tip].mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas, "lead_foot": lead,
        "trail_foot": trail, "limbs": limbs, "has_mouth": bool(mouth.any()),
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, crown=0.0, tymp=0.0, tail=0.0,
        foot_lead=0.0, foot_trail=0.0,
        step_l=(0, 0), step_r=(0, 0), kick=(0, 0),
        lean=0.0, knock=0, flash=False, roar=0, turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        breath = wave(f, n, 1.0, phase=0.5)
        p["bdy"] = -1 if breath > 0.55 else 0
        p["crown"] = wave(f, n, 8.0, phase=-3 * math.pi / 8, cycles=2.0)
        p["tymp"] = wave(f, n, 6.0, phase=-3 * math.pi / 8, cycles=2.0)
        p["tail"] = wave(f, n, 7.0, phase=0.4)
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["bdy"] = -1 if abs(math.sin(2 * u)) > 0.75 else 0
        p["crown"] = wave(f, n, 5.0, cycles=2.0)
        p["tymp"] = wave(f, n, 4.0)
        p["tail"] = wave(f, n, 8.0, phase=1.1)
        p["step_l"] = (int(round(2.0 * sa)), -int(round(2.0 * max(0.0, sa))))
        p["step_r"] = (int(round(2.0 * sb)), -int(round(2.0 * max(0.0, sb))))
    elif action == "attack":
        # 0 rest. 1-4 roar. 5-8 back view, limbs cocked. 9-12 stomp and tail.
        # 13-15 return to the still. Abs values on 5-15 meet finish()'s envelope.
        p["crown"] = [0, 8, 14, 16, 14, 12, 12, 12, 10, 8, 6, 4, 2, 0, 0, 0][f]
        p["tymp"] = [0, 6, 10, 12, 12, 10, 10, 10, 8, 8, 6, 4, 2, 0, 0, 0][f]
        p["tail"] = [0, 4, 8, 12, 12, -22, -22, -22, -22, 22, 22, 22, 14, 6, 2, 0][f]
        p["foot_lead"] = [0, 0, 0, 0, 0, -30, -30, -30, -30, 30, 30, 30, 16, 8, 2, 0][f]
        p["foot_trail"] = [0, 0, 0, 0, 0, -18, -18, -18, -18, 18, 18, 18, 10, 4, 2, 0][f]
        if f in (2, 3, 4):
            p["bdy"] = -1
            p["roar"] = f
        elif f == 1:
            p["roar"] = 1
        if f in (10, 11, 12):
            p["bdx"] = 1
            p["kick"] = (3, -1 if f == 10 else 0)
        elif f in (5, 6, 7, 8):
            p["kick"] = (-1, -2)
        p["turned"] = 5 <= f <= 8
    else:
        env = [0.0, 0.55, 0.90, 1.0, 0.85, 0.62, 0.42, 0.26, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["crown"] = 12.0 * env
        p["tymp"] = 10.0 * env
        p["tail"] = 14.0 * env
        p["foot_lead"] = 12.0 * env
        p["foot_trail"] = 8.0 * env
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


def part_transform(rig, p, name):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    if name == "torso":
        return shift(L["torso"], bdx, bdy)
    if name in ("foot_l", "foot_r"):
        fwd, lift = p["step_l"] if name == "foot_l" else p["step_r"]
        if name == rig["lead_foot"]:
            ang = part_angle(rig, name, "fwd", p["foot_lead"])
            kf, ky = p["kick"]
            fwd += kf
            lift += ky
        else:
            ang = part_angle(rig, name, "fwd", p["foot_trail"])
        return moved(L[name], ang, rig["pivots"][name], bdx + int(round(fc * fwd)), bdy + int(lift))
    if name == "crown":
        ang = part_angle(rig, name, "up", p["crown"])
    elif name == "tymp":
        ang = part_angle(rig, name, "up", p["tymp"])
    elif name == "tail":
        ang = part_angle(rig, name, "fwd", p["tail"])
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


def roar_fx(img, origin, facing, age):
    """Mouth wedge just outside the face, and a ring. Palette colours only."""
    ox, oy = origin
    h, w = img.shape[:2]
    x, y = int(round(ox)), int(round(oy))
    step = 1 if facing > 0 else -1
    for _ in range(16):
        nx = x + step
        if not (1 <= nx < w - 1 and 1 <= y < h - 1):
            return
        if img[y, nx, 3] < 128:
            x = nx
            break
        x = nx
    else:
        return
    cols = (INK, BROWN, TAN, CYAN)
    for i, col in enumerate(cols[: 3 + (age > 1)]):
        put(img, x + step * i, y, col)
        if i and i < 3:
            put(img, x + step * i, y - 1, BROWN if i == 1 else INK)
    r = 6 + age * 2
    for k in range(12):
        ang = 2 * math.pi * k / 12
        put(img, ox + r * math.cos(ang), oy + r * 0.65 * math.sin(ang), CYAN if k % 2 == 0 else WHITE)


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


def rig_for(rigs, d, p):
    if p["turned"]:
        kind, mirror = VIEW_OF[d]
        return rigs[("back" if kind == "front" else "front", not mirror)]
    return rigs[d]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    moved_any = (
        abs(p["lean"]) > 0.4 or abs(p["crown"]) >= 0.45 or abs(p["tymp"]) >= 0.45
        or abs(p["tail"]) >= 0.45 or abs(p["foot_lead"]) >= 0.45 or abs(p["foot_trail"]) >= 0.45
        or p["bdx"] or p["bdy"] or any(p["step_l"]) or any(p["step_r"]) or any(p["kick"])
    )
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
        moved_any = True
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        # Measured: fc * lean tips the head toward the face. Negate so it leans back.
        body = rotate_nearest(body, -fc * p["lean"], piv)
        moved_any = True
    if moved_any:
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    if p["roar"] and rig["has_mouth"] and not p["turned"]:
        roar_fx(img, rig["pivots"]["mouth"], fc, p["roar"] - 1)
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


def head_mean_x(img):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    top = ys <= int(ys.min()) + 10
    return float(xs[top].mean())


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
        "crown": (240, 80, 80, 255), "tymp": (80, 180, 240, 255),
        "tail": (240, 200, 40, 255), "foot_l": (80, 200, 80, 255),
        "foot_r": (200, 80, 200, 255), "torso": (180, 180, 180, 255),
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
        "dex": 536, "form": "normal", "variant": "palpitoad", "gen": "gen5",
        "dir": "monsterSprite/gen5/0536_palpitoad/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def sheet_metrics(master, sheets, palette, rigs):
    pal = {tuple(int(v) for v in c) for c in palette}
    op = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[op][:, :3], axis=0)} if op.any() else set()
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
    idle0 = {}
    for di, d in enumerate(DIRS):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        idle0[d] = int(np.any(t != rigs[d]["canvas"], axis=2).sum())
    return off, partial, clip, bodies, idle0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
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
    print("pivots W", {k: tuple(round(x, 1) for x in v) for k, v in rigs["W"]["pivots"].items()},
          "lead", rigs["W"]["lead_foot"], "facing", rigs["W"]["facing"], flush=True)
    print("pivots N", {k: tuple(round(x, 1) for x in v) for k, v in rigs["N"]["pivots"].items()},
          "lead", rigs["N"]["lead_foot"], "facing", rigs["N"]["facing"], flush=True)

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

    # Backward lean: W faces left, so the head moves right. E faces right, head moves left.
    idle_w = synth(rigs, "W", "idle", 0, palette, effects=False)
    hurt_w = synth(rigs, "W", "hurt", 3, palette, effects=False)
    idle_e = synth(rigs, "E", "idle", 0, palette, effects=False)
    hurt_e = synth(rigs, "E", "hurt", 3, palette, effects=False)
    dx_w = head_mean_x(hurt_w) - head_mean_x(idle_w)
    dx_e = head_mean_x(hurt_e) - head_mean_x(idle_e)
    print("hurt head dx W", round(dx_w, 2), "E", round(dx_e, 2), flush=True)
    if dx_w <= 0.5 or dx_e >= -0.5:
        raise SystemExit("hurt lean is not backward")

    if args.samples:
        y0, y1, x0, x1 = crop_box(rigs["S"]["canvas"])
        pal = {tuple(int(v) for v in c) for c in palette}
        off = set()
        partial = 0
        clip = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    t = synth(rigs, d, act, f, palette)
                    a = t[..., 3]
                    partial += int(((a != 0) & (a != 255)).sum())
                    if (a[0] == 255).any() or (a[-1] == 255).any() or (a[:, 0] == 255).any() or (a[:, -1] == 255).any():
                        clip.append(f"{act}:{d}:{f}")
                    op = a == 255
                    if op.any():
                        cols = {tuple(int(v) for v in c) for c in np.unique(t[op][:, :3], axis=0)}
                        off |= cols - pal
        print("off_palette", len(off), flush=True)
        print("partial_alpha", partial, flush=True)
        print("edge_touching", len(clip), clip[:8], flush=True)
        lost = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    bad = disconnected_limbs(rig_for(rigs, d, p), p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        print("disconnected", lost[:12], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["crown"], 1), round(p["tail"], 1), round(p["foot_lead"], 1),
                        p["turned"], p["roar"], p["spin"]))
        print("attack", atk, flush=True)
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
            z = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
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
    motion = np.concatenate(strips, axis=0)
    motion_z = np.repeat(np.repeat(motion, 2, 0), 2, 1)
    Image.fromarray(motion_z, "RGBA").save(OUT_DIR / "preview_motion.png")
    preview_image(sheets).save(OUT_DIR / "preview.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", OUT_DIR / "actions" / "preview.png")

    off, partial, clip, bodies, idle0 = sheet_metrics(master, sheets, palette, rigs)
    bounds = []
    for di in range(8):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        ys, xs = np.nonzero(t[..., 3] == 255)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    sprite_h = max(bounds[i + 3] for i in range(0, 32, 4))
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
        "generator": "forge_gen5_palpitoad_actions", "version": VERSION, "key": KEY,
        "dex": 536, "variant": "palpitoad", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "toad biped (mint crown, right tympanum, cream belly, mouth, tail stub, two feet)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (mouth xmin left of eye-white mean; tail stub is the rear on the right)",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
