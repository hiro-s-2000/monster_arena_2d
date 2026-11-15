#!/usr/bin/env python3
"""Victini normal (0494) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x28-68 y21-74, alpha 0/255, one piece):
  Eyes (blue) sit at mean x 55.8. The pink mouth is x56-60 mean x 58.0, y53-57,
  so the face tip is to the right of the eyes. The still faces screen-right.
  V crest: left prong x28-52 y33-45, right spike x54-61 y21-40.
  Near arm (image left) x33-44 y55-67, far arm x62-68 y57-68.
  Toes: left x45-53 y72-74, right x57-63 y72-73. No separate wings on the front.
normal/back.png (bbox x29-63 y22-74, foot y74) has no eyes and no mouth.
  Both crest prongs, two side wings, and the two orange toes. No mouth is invented.

Directions, still faces right:
  S SE E front as drawn (screen-right). W SW front mirrored (screen-left).
  NE back as drawn. N NW back mirrored. E is screen-right, W screen-left, N is back.
Attack 5-8 swaps to the opposite view at equal scale (S turns into the N back).
No width squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12. Feet stay on the ground line.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_victini_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0494_victini/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0494_victini/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0494_victini"
VERSION = "victini_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 176, 176
OFF_X = (CELL_W - 96) // 2
OFF_Y = 40
SRC_FOOT = 74
FOOT_Y = OFF_Y + SRC_FOOT
# kind, mirror. Front as drawn looks screen-right.
VIEW_OF = {
    "S": ("front", False), "SE": ("front", False), "E": ("front", False),
    "W": ("front", True), "SW": ("front", True),
    "NE": ("back", False), "N": ("back", True), "NW": ("back", True),
}
MAX_LEAN = 38.0
# Wind-up peaks finish() may raise frames 5-12 toward. Impact frames are authored
# at least this high, and the punch itself is drawn larger (about 34 deg).
PEAKS = {"ear": 16.0, "arm": 22.0, "wing": 22.0}

INK = (0, 0, 0)
ORANGE = (255, 106, 24)
ORANGE_D = (189, 74, 0)
ORANGE_K = (139, 49, 16)
CREAM = (255, 230, 172)
CREAM_D = (197, 156, 90)
BROWN = (106, 74, 16)
BROWN_D = (131, 106, 57)
GREY = (57, 57, 57)
EYE = (57, 90, 172)
EYE_L = (115, 172, 255)
WHITE = (255, 255, 255)
PINK = (164, 90, 90)
PINK_L = (197, 115, 115)
SILVER = (197, 189, 189)
CREAM_COLS = (CREAM, CREAM_D)

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
    """4x nearest rotation. Positive is clockwise on screen. No squash."""
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


def paint_cushion(layer, allowed, steps):
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        for d in ("up", "down", "left", "right"):
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
    cream = any_color(img, CREAM_COLS)
    ear_r = op & (yy <= 40) & (xx >= 54)
    ear_l = op & (yy <= 45) & (xx <= 52) & ~ear_r
    cheek = op & (yy >= 44) & (xx >= 46) & (xx <= 54)
    ear_l = ear_l & ~cheek
    foot = op & (yy >= 72) & ~cream
    foot_l = foot & (xx <= 54)
    foot_r = foot & (xx >= 55)
    arm_l = op & (xx <= 44) & (yy >= 54) & (yy <= 67) & ~ear_l & ~foot_l
    arm_r = op & (xx >= 62) & (yy >= 57) & (yy <= 68) & ~foot_r & ~ear_r
    used = ear_l | ear_r | arm_l | arm_r | foot_l | foot_r
    torso = op & ~used
    masks = {
        "ear_l": ear_l, "ear_r": ear_r, "arm_l": arm_l, "arm_r": arm_r,
        "foot_l": foot_l, "foot_r": foot_r, "torso": torso,
    }
    check_partition(op, masks)
    eye = any_color(img, (EYE, EYE_L))
    pink = any_color(img, (PINK, PINK_L))
    if int((eye & torso).sum()) != int(eye.sum()) or eye.sum() < 30:
        raise SystemExit("eyes left the head")
    if int((pink & torso).sum()) != int(pink.sum()) or pink.sum() < 10:
        raise SystemExit("mouth left the head")
    ex = float(np.nonzero(eye)[1].mean())
    mx = float(np.nonzero(pink)[1].mean())
    if mx <= ex:
        raise SystemExit("mouth is not right of the eyes; facing changed")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    cream = any_color(img, CREAM_COLS)
    ear_l = op & (yy <= 42) & (xx <= 47)
    ear_r = op & (yy <= 42) & (xx >= 52)
    foot = op & (yy >= 71) & ~cream
    foot_l = foot & (xx < 48)
    foot_r = foot & (xx >= 48)
    wing_l = op & (yy >= 57) & (xx <= 44) & ~foot_l & ~foot_r
    wing_r = op & (yy >= 57) & (xx >= 55) & ~foot_l & ~foot_r
    used = ear_l | ear_r | wing_l | wing_r | foot_l | foot_r
    torso = op & ~used
    masks = {
        "ear_l": ear_l, "ear_r": ear_r, "wing_l": wing_l, "wing_r": wing_r,
        "foot_l": foot_l, "foot_r": foot_r, "torso": torso,
    }
    check_partition(op, masks)
    if any_color(img, (EYE, EYE_L, PINK, PINK_L)).any():
        raise SystemExit("back still grew a face")
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
    near = d <= d.min() + 4.0
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
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    limbs = [k for k in placed if k != "torso"]
    # Only the ankle/root stays painted under the part, so a lifted toe is not cloned.
    depth = {"ear_l": 5, "ear_r": 5, "arm_l": 4, "arm_r": 4, "wing_l": 4, "wing_r": 4,
             "foot_l": 3, "foot_r": 3}
    stump = np.zeros_like(torso_m)
    for name in limbs:
        stump |= joint_band(placed[name], torso_m, depth.get(name, 4))
    layers["torso"] = paint_cushion(layers["torso"], stump, 8)
    facing = 1.0 if not mirror else -1.0
    pivots = {name: pivot_of(layers[name], layers["torso"]) for name in limbs}
    signs = {name: signs_for(layers[name], pivots[name], facing) for name in limbs}
    pink = any_color(canvas, (PINK, PINK_L))
    has_mouth = bool(pink.any())
    muzzle = None
    if has_mouth:
        py, px = np.nonzero(pink)
        my = int(round(float(py.mean())))
        band = canvas[max(0, my - 2):my + 3]
        xs = np.nonzero((band[..., 3] >= 128).any(axis=0))[0]
        if facing > 0:
            muzzle = (int(xs.max()) + 2, my)
        else:
            muzzle = (int(xs.min()) - 2, my)
        pivots["mouth"] = (float(px.mean()), float(py.mean()))
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
        "pivots": pivots, "signs": signs, "canvas": canvas, "has_mouth": has_mouth,
        "muzzle": muzzle, "near_foot": near, "limbs": limbs,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, ear=0.0, arm=0.0, wing=0.0, ear_mode="up", arm_mode="fwd",
        wing_mode="up", kick=(0, 0), lean=0.0, knock=0, flash=False, ring=-1,
        wedge=0, spark=0, turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        breath = wave(f, n, 1.0)
        p["bdy"] = -1 if breath > 0.55 else 0
        p["ear"] = wave(f, n, 6.0, phase=0.5)
        p["wing"] = wave(f, n, 7.0, phase=0.8, cycles=2.0)
        p["arm"] = wave(f, n, 4.0, phase=0.3)
        p["ear_mode"] = "up"
        p["wing_mode"] = "up"
        p["arm_mode"] = "fwd"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        p["bdy"] = -1 if abs(math.sin(2 * u)) > 0.55 else 0
        p["ear"] = 5.0 * sa
        p["wing"] = 8.0 * math.sin(2 * u)
        p["arm"] = 14.0 * sa
        # Only the lifted toe steps forward. The planted toe stays on its mark.
        p["step_l"] = (int(round(2.0 * max(0.0, sa))), -int(round(2.0 * max(0.0, sa))))
        p["step_r"] = (int(round(2.0 * max(0.0, -sa))), -int(round(2.0 * max(0.0, -sa))))
        p["ear_mode"] = "up"
        p["wing_mode"] = "up"
        p["arm_mode"] = "fwd"
        p["oppose_arms"] = True
    elif action == "attack":
        # 0 rest. 1-4 roar. 5-8 back view. 9-12 strike. 13-15 recover.
        ears = [0, 8, 12, 16, 16, 16, 16, 16, 16, 18, 20, 22, 14, 6, 2, 0]
        arms = [0, 0, 0, 0, 0, -20, -22, -22, -22, 28, 34, 34, 26, 10, 4, 0]
        wings = [0, 4, 8, 12, 14, 20, 22, 22, 22, 22, 20, 16, 11, 4, 2, 0]
        p["ear"] = float(ears[f])
        p["arm"] = float(arms[f])
        p["wing"] = float(wings[f])
        if 1 <= f <= 8:
            p["ear_mode"] = "up"
        else:
            p["ear_mode"] = "fwd"
        # Frames 5-8 store a negative arm: pull back, in the forward basis.
        # finish() keeps it because the magnitude already meets the peak.
        p["arm_mode"] = "fwd"
        p["wing_mode"] = "up" if f <= 8 else "fwd"
        if f in (2, 3, 4):
            p["bdx"] = -1
        elif f in (10, 11, 12):
            p["bdx"] = 1
        if f in (1, 2):
            p["wedge"] = f
        elif f in (3, 4):
            p["wedge"] = 3
        if 2 <= f <= 4:
            p["ring"] = f - 2
        if f in (10, 11, 12):
            p["kick"] = {10: (2, -1), 11: (3, 0), 12: (1, 0)}[f]
        if f in (11, 12):
            p["spark"] = f - 10
        p["turned"] = 5 <= f <= 8
        p["bdy"] = -1 if f in (3, 4, 9) else 0
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(4 * env))
        p["ear"] = 8.0 * env
        p["arm"] = 10.0 * env
        p["wing"] = 8.0 * env
        p["ear_mode"] = "up"
        p["arm_mode"] = "back"
        p["wing_mode"] = "up"
        p["flash"] = f == 1
        p["spark"] = 1 if f in (2, 3) else 0
    if "step_l" not in p:
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
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


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = fc * p["bdx"]
    bdy = p["bdy"]
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    steps = {"foot_l": p["step_l"], "foot_r": p["step_r"]}
    kf, ky = p["kick"]
    for name in ("foot_l", "foot_r"):
        fwd, lift = steps[name]
        if name == rig["near_foot"]:
            fwd += kf
            lift += ky
        # Toes stay on the ground except when that foot is stepping or kicking.
        paste(out, shift(L[name], bdx + fc * fwd, lift))
    for name in rig["limbs"]:
        if name.startswith("foot_") or name == "torso":
            continue
        if name.startswith("ear"):
            ang = part_angle(rig, name, p["ear_mode"], p["ear"])
        elif name.startswith("arm"):
            amp = p["arm"]
            if name == "arm_r" and p.get("oppose_arms"):
                amp = -amp
            ang = part_angle(rig, name, p["arm_mode"], amp)
        elif name.startswith("wing"):
            ang = part_angle(rig, name, p["wing_mode"], p["wing"])
        else:
            ang = 0.0
        paste(out, moved(L[name], ang, piv[name], bdx, bdy))
    return out


def ellipse_pts(cx, cy, rx, ry, step):
    a = 0
    while a < 360:
        r = math.radians(a)
        yield cx + rx * math.cos(r), cy + ry * math.sin(r)
        a += step


def mouth_wedge(img, origin, fc, age):
    ox, oy = origin
    for i in range(age):
        x = ox + fc * (1 + i)
        span = i
        for dy in range(-span, span + 1):
            col = PINK if abs(dy) == span else PINK_L
            put(img, x, oy + dy, col)


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (ORANGE, CREAM, WHITE)
    rad = 8 + age * 5
    for deg in range(-48, 49, 8):
        a = math.radians(deg)
        x = ox + fc * rad * math.cos(a)
        y = oy + rad * math.sin(a) * 0.62
        put(img, x, y, cols[age % 3])


def spark(img, origin, fc, age):
    ox, oy = origin
    r = 3 + age * 2
    for dx, dy in ((r, 0), (-2, -2), (0, -r), (2, 2), (fc * r, -1)):
        put(img, ox + dx, oy + dy, ORANGE if (dx + dy) % 2 == 0 else CREAM)


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = any_color(out, (INK, GREY))
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def limb_linked(img, layer):
    limb = layer[..., 3] >= 128
    if not limb.any():
        return True
    body = (img[..., 3] >= 128) & ~limb
    return bool((limb & dilate(body, 1)).any())


def synth(rigs, d, action, f, palette, effects=True, report=None):
    p = pose(action, f)
    rig = rigs[d]
    if p["turned"]:
        kind, mirror = VIEW_OF[d]
        rig = rigs[("back" if kind == "front" else "front", not mirror)]
    body = compose(rig, p)
    rotated = abs(p["lean"]) > 0.4 or abs(p["ear"]) >= 0.45 or abs(p["arm"]) >= 0.45 or abs(p["wing"]) >= 0.45
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        # Rear foot stays down. Positive rotation is clockwise, so the lean
        # is the opposite of facing and the belly is not flipped.
        body = rotate_nearest(body, -fc * p["lean"], piv)
    if rotated or abs(p["lean"]) > 0.4:
        body = fill_pinholes(despeckle(body))
    if report is not None:
        lost = [name for name in rig["limbs"] if name != "torso" and not limb_linked(body, body)]
        # limb_linked against the whole image is meaningless; check each source part after compose
        report["pose"] = {k: p[k] for k in ("ear", "arm", "wing", "lean", "turned", "wedge")}
        report["lost"] = lost
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    if rig["has_mouth"] and rig["muzzle"] is not None:
        mx, my = rig["muzzle"]
        mx += fc * p["bdx"]
        my += p["bdy"]
        if p["wedge"] and not p["turned"]:
            mouth_wedge(img, (mx, my), fc, p["wedge"])
        if p["ring"] >= 0 and not p["turned"]:
            roar_rings(img, (mx + fc * 2, my), fc, p["ring"])
        if p["spark"] and action == "attack":
            spark(img, (mx + fc * 4, my - 2), fc, p["spark"])
    elif p["spark"] and action == "hurt":
        cx, cy = rig["pivots"]["center"]
        spark(img, (cx - fc * 6, cy - 8), -fc, 1)
    elif p["spark"] and action == "attack":
        cx, cy = rig["pivots"]["center"]
        spark(img, (cx + fc * 16, cy), fc, p["spark"])
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
    """Each moving part must still touch the composed body."""
    body = compose(rig, p)
    # Rebuild each limb the same way compose does and test adjacency.
    L = rig["layers"]
    fc = rig["facing"]
    bdx = fc * p["bdx"]
    bdy = p["bdy"]
    lost = []
    steps = {"foot_l": p.get("step_l", (0, 0)), "foot_r": p.get("step_r", (0, 0))}
    kf, ky = p["kick"]
    for name in rig["limbs"]:
        if name.startswith("foot"):
            fwd, lift = steps[name]
            if name == rig["near_foot"]:
                fwd += kf
                lift += ky
            layer = shift(L[name], bdx + fc * fwd, lift)
        elif name.startswith("ear"):
            ang = part_angle(rig, name, p["ear_mode"], p["ear"])
            layer = moved(L[name], ang, rig["pivots"][name], bdx, bdy)
        elif name.startswith("arm"):
            amp = p["arm"]
            if name == "arm_r" and p.get("oppose_arms"):
                amp = -amp
            ang = part_angle(rig, name, p["arm_mode"], amp)
            layer = moved(L[name], ang, rig["pivots"][name], bdx, bdy)
        elif name.startswith("wing"):
            ang = part_angle(rig, name, p["wing_mode"], p["wing"])
            layer = moved(L[name], ang, rig["pivots"][name], bdx, bdy)
        else:
            continue
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


def crop_box(canvas, pad=8):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 6)
    x0 = max(0, int(xs.min()) - pad - 18)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 18)
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
        for f in range(FRAMES[act]):
            t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            tiles.append(t[y0:y1, x0:x1])
        # pad the hurt row out to 16 so the grid stays even
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
        "ear_l": (220, 40, 40, 255), "ear_r": (240, 140, 20, 255),
        "arm_l": (40, 90, 220, 255), "arm_r": (20, 180, 80, 255),
        "wing_l": (180, 40, 200, 255), "wing_r": (80, 200, 220, 255),
        "foot_l": (240, 220, 40, 255), "foot_r": (255, 80, 180, 255),
        "torso": (210, 210, 210, 255),
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
        "dex": 494, "form": "normal", "variant": "victini", "gen": "gen5",
        "dir": "monsterSprite/gen5/0494_victini/normal", "sheetDir": "agent-sprite-forge",
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
                    rig = rigs[d]
                    if p["turned"]:
                        kind, mirror = VIEW_OF[d]
                        rig = rigs[("back" if kind == "front" else "front", not mirror)]
                    bad = disconnected_limbs(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, p["ear"], p["arm"], p["wing"], p["ear_mode"], p["arm_mode"], p["turned"], p["spin"]))
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
    # Motion sheet is the picture to judge: S, N, E, W, all frames, cropped.
    strips = []
    for d in ("S", "N", "E", "W"):
        strips.append(motion_strip(sheets, DIRS.index(d), box))
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
    # Idle frame 0 of every direction must still match the rig canvas.
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
        "generator": "forge_gen5_victini_actions", "version": VERSION, "key": KEY,
        "dex": 494, "variant": "victini", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped crest striker (V ears, small arms, back wings, two toes)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-right (mouth x > eye x)",
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
