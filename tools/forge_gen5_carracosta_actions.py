#!/usr/bin/env python3
"""Carracosta normal (0565) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, opaque 2361, bbox x15-80 y19-77, sole y77,
12 colours, alpha 0/255):
  One eye: the white pixel (230, 230, 230) at x41 y30, with the pale
  (164, 205, 213) specks beside it at y29-31. The head's leftmost opaque
  pixel above the flipper cut (y<=47) is left of that eye, so the still
  faces screen-left.
  No open mouth is drawn. There is no mouth colour and no tooth row.
  The pale oval (x39-51 y45-61, 89px) is the plastron, inset in the shell,
  so the roar does not invent a beak wedge or a ring. It lifts the flippers
  that are already in the still.
  Near flipper: the separated left column x15-36 y48-74 (blue and cyan).
  Near foot: x37-44 y66-72. Far foot: the toes that reach y77, x58-76 y72-77.
  Far flipper: blue on the right, x63-77 y52-69. Tail nub: x76-80 y56-64.
normal/back.png (opaque 2455, bbox x14-80 y18-78, sole y78, same 12 colours):
  Eyes at (62, 25) and (63, 26). The head's right edge above y40 is right
  of those eyes, so this still faces screen-right. Left flipper x14-40
  y48-73. Right foot reaches y78. No separate tail, no mouth.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (screen-right). NW back mirrored.
  Attack 5-8 is that direction's other side at equal scale: W becomes N,
  S becomes N flipped. No |cos| squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_carracosta_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0565_carracosta/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0565_carracosta/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0565_carracosta"
VERSION = "carracosta_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 224, 224
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 24
FRONT_FOOT = 77
BACK_FOOT = 78
FOOT_Y = OFF_Y + BACK_FOOT
# Front as drawn looks screen-left. W keeps that picture. Its back is N.
# S is the mirrored front, so its back is N mirrored. Equal scale, no squash.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# finish() may lift the flippers this far. The chop is drawn at 34, above
# this, so the hit is not stretched past the angle this script already draws.
PEAKS = {"arm": 18.0, "arm2": 14.0}

INK = (0, 0, 0)
DARK = (41, 49, 57)
NAVY = (41, 57, 98)
BLUE = (74, 148, 213)
MID = (57, 115, 156)
CYAN = (90, 172, 230)
SHELL = (98, 115, 131)
SHELL_D = (65, 74, 90)
SHELL_M = (82, 90, 106)
PALE = (164, 205, 213)
WHITE = (230, 230, 230)
TEAL = (57, 98, 131)
INK_COLS = (INK, DARK, NAVY)
BLUE_COLS = (BLUE, MID, CYAN)
FRONT_LIMBS = ("arm", "rarm", "foot", "far", "tail")
BACK_LIMBS = ("left", "right")

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
    h, w = allowed.shape
    out = np.zeros_like(allowed)
    stack = []
    for y, x in seeds:
        if 0 <= y < h and 0 <= x < w and allowed[y, x] and not out[y, x]:
            out[y, x] = True
            stack.append((int(y), int(x)))
    if not stack:
        raise SystemExit(f"flood missed every seed {seeds}")
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and allowed[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                stack.append((ny, nx))
    return out


def components(mask):
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
    groups.sort(key=len, reverse=True)
    return groups


def mask_from(shape, pts):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m


def largest_component(mask):
    groups = components(mask)
    if not groups:
        return 0
    return len(groups[0])


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


def bridge(op, part, ink):
    """Ink pixels sitting directly between two pixels of this part."""
    if not part.any():
        return part
    up = np.zeros_like(part)
    dn = np.zeros_like(part)
    lf = np.zeros_like(part)
    rt = np.zeros_like(part)
    up[1:] = part[:-1]
    dn[:-1] = part[1:]
    lf[:, 1:] = part[:, :-1]
    rt[:, :-1] = part[:, 1:]
    between = ink & op & ~part & ((up & dn) | (lf & rt))
    return part | between


def exclusive_outline(op, part, ink):
    body = op & ~part
    only = ink & dilate(part, 1) & ~dilate(body, 1)
    return part | only


def absorb_islands(op, parts):
    """Give a torso chip to a limb only when the limb borders it more than the body."""
    names = list(parts)
    for _ in range(4):
        used = np.zeros(op.shape, bool)
        for n in names:
            used |= parts[n]
        groups = components(op & ~used)
        if len(groups) <= 1:
            break
        main = mask_from(op.shape, groups[0])
        changed = False
        for g in groups[1:]:
            speck = mask_from(op.shape, g)
            near = dilate(speck, 1)
            best, best_n = None, -1
            for n in names:
                k = int((near & parts[n]).sum())
                if k > best_n:
                    best, best_n = n, k
            tb = int((near & main).sum())
            if best is not None and best_n > tb and best_n > 0:
                parts[best] |= speck
                changed = True
        if not changed:
            break
    return parts


def require_solid(name, mask, torso):
    if not mask.any():
        raise SystemExit(f"{name} is empty")
    if len(components(mask)) != 1:
        raise SystemExit(f"{name} fell apart ({len(components(mask))})")
    if not (mask & dilate(torso, 1)).any():
        raise SystemExit(f"{name} does not touch the body")


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    ink = any_color(img, INK_COLS)
    blue = any_color(img, BLUE_COLS)
    arm = flood(op & (yy >= 48) & (xx <= 36), [(56, 22), (64, 24), (70, 28)])
    arm = exclusive_outline(op, bridge(op, arm, ink), ink)
    arm = bridge(op, arm, ink)
    foot = flood(op & (yy >= 66) & (xx >= 32) & (xx <= 48) & ~arm, [(68, 40), (70, 36)])
    foot = exclusive_outline(op & ~arm, bridge(op & ~arm, foot, ink & ~arm), ink & ~arm)
    far = flood(op & (yy >= 72) & (xx >= 58) & (xx <= 77) & ~arm & ~foot, [(76, 66), (74, 70)])
    far = exclusive_outline(
        op & ~arm & ~foot, bridge(op & ~arm & ~foot, far, ink & ~arm & ~foot), ink & ~arm & ~foot,
    )
    tail = flood(
        op & (xx >= 76) & (yy >= 55) & (yy <= 64) & ~arm & ~foot & ~far,
        [(58, 78), (60, 79)],
    )
    tail = exclusive_outline(
        op & ~arm & ~foot & ~far,
        bridge(op & ~arm & ~foot & ~far, tail, ink & ~arm & ~foot & ~far),
        ink & ~arm & ~foot & ~far,
    )
    busy = arm | foot | far | tail
    rarm = flood((blue | ink) & (yy >= 52) & (yy <= 69) & (xx >= 63) & ~busy, [(58, 70), (62, 68), (64, 66)])
    rarm = exclusive_outline(op & ~busy, bridge(op & ~busy, rarm, ink & ~busy), ink & ~busy)
    parts = absorb_islands(op, {"arm": arm, "foot": foot, "far": far, "tail": tail, "rarm": rarm})
    used = np.zeros_like(op)
    for m in parts.values():
        used |= m
    torso = op & ~used
    parts["torso"] = torso
    check_partition(op, parts)
    if len(components(torso)) != 1:
        raise SystemExit("front torso fell apart")
    for name in FRONT_LIMBS:
        require_solid("front " + name, parts[name], torso)
    eye = is_color(img, WHITE)
    pale = is_color(img, PALE)
    if int(eye.sum()) != 1 or int((eye & torso).sum()) != 1:
        raise SystemExit("front eye left the head")
    if int((pale & torso).sum()) != int(pale.sum()) or int(pale.sum()) < 80:
        raise SystemExit("front plastron was cut")
    eye_x = float(np.nonzero(eye)[1][0])
    head = op & (yy <= 47)
    tip_x = int(np.nonzero(head)[1].min())
    if tip_x >= eye_x - 4:
        raise SystemExit(f"front does not face left: tip {tip_x} eye {eye_x}")
    if int(np.nonzero(parts["far"])[0].max()) != FRONT_FOOT:
        raise SystemExit("front ground foot moved")
    if int(parts["arm"].sum()) < 250 or int(parts["rarm"].sum()) < 150:
        raise SystemExit("front flippers look short")
    if int(parts["tail"].sum()) < 20:
        raise SystemExit("front tail looks short")
    return parts, {"eye_x": eye_x, "tip_x": tip_x, "has_mouth": False}


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    ink = any_color(img, INK_COLS)
    blue = any_color(img, BLUE_COLS)
    upper = flood(blue & (yy >= 48) & (xx <= 40), [(58, 22), (62, 20), (66, 24)])
    lower = flood(op & (yy >= 63) & (xx <= 40) & (xx >= 14), [(68, 24), (70, 30)])
    left = flood((upper | lower) & (xx <= 40), [(60, 22)])
    left = exclusive_outline(op, bridge(op, left, ink), ink)
    right = flood(op & (yy >= 67) & (xx >= 66) & ~left, [(74, 72), (77, 70)])
    right = exclusive_outline(op & ~left, bridge(op & ~left, right, ink & ~left), ink & ~left)
    parts = absorb_islands(op, {"left": left, "right": right})
    used = parts["left"] | parts["right"]
    torso = op & ~used
    parts["torso"] = torso
    check_partition(op, parts)
    if len(components(torso)) != 1:
        raise SystemExit("back torso fell apart")
    for name in BACK_LIMBS:
        require_solid("back " + name, parts[name], torso)
    eye = is_color(img, WHITE)
    if int(eye.sum()) != 2 or int((eye & torso).sum()) != 2:
        raise SystemExit("back eyes left the head")
    if any_color(img, (PALE,)).sum() > 6:
        raise SystemExit("back grew a plastron")
    eye_x = float(np.nonzero(eye)[1].mean())
    head = op & (yy <= 40) & (xx >= 48)
    tip_x = int(np.nonzero(head)[1].max())
    if tip_x <= eye_x + 2:
        raise SystemExit(f"back does not face right: tip {tip_x} eye {eye_x}")
    if int(np.nonzero(parts["right"])[0].max()) != BACK_FOOT:
        raise SystemExit("back ground foot moved")
    if int(parts["left"].sum()) < 250:
        raise SystemExit("back flipper looks short")
    return parts, {"eye_x": eye_x, "tip_x": tip_x, "has_mouth": False}


def place(arr):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    out[OFF_Y:OFF_Y + 96, OFF_X:OFF_X + 96] = arr
    return out


def joint_band(part, torso, depth):
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 3)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def paint_cushion(layer, allowed, steps):
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        grown = acc.copy()
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
            take = (grown[..., 3] < 128) & allowed & (mv[..., 3] >= 128)
            grown[take] = mv[take]
        acc = grown
    return acc


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


def build_view(img, masks, kind, mirror, info):
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
    names = FRONT_LIMBS if kind == "front" else BACK_LIMBS
    for name in names:
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, 6), 6,
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in names:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing,
        "has_mouth": False, "info": info,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
        "limbs": names,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, lift_all=False,
        arm=0.0, arm2=0.0, tail=0.0, kick=0.0,
        mode="up",
        step_a=(0, 0), step_b=(0, 0),
        lean=0.0, knock=0, flash=False,
        spark=0, turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Shell breathes. Flippers sway on their joints. Feet stay down.
        p["arm"] = wave(f, n, 3.5, phase=0.5)
        p["arm2"] = wave(f, n, 2.5, phase=1.1)
        p["tail"] = wave(f, n, 8.0, cycles=2.0, phase=0.4)
        p["bdy"] = -1 if wave(f, n, 1.0, phase=0.2) > 0.55 else 0
        p["mode"] = "up"
        p["lift_all"] = False
    elif action == "walk":
        # Two drawn feet, opposite steps. The planted sole does not rise.
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_a"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["step_b"] = (int(round(2 * sb)), -int(round(3 * max(0.0, sb))))
        p["arm"] = wave(f, n, 8.0, phase=math.pi)
        p["arm2"] = wave(f, n, 6.0, phase=0.0)
        p["tail"] = wave(f, n, 10.0, cycles=2.0, phase=math.pi / 2)
        p["bdy"] = -1 if abs(math.sin(u)) > 0.85 else 0
        p["mode"] = "fwd"
        p["lift_all"] = False
    elif action == "attack":
        # 0 rest. 1-4 raise the flippers (no mouth to open).
        # 5-8 the other side, equal scale.
        # 9 wind-up. 10-12 the near flipper strikes. 13-15 return.
        arm = [0, 8, 14, 18, 18, 0, 0, 0, 0, -18, 28, 34, 34, 12, 4, 0]
        arm2 = [0, 6, 10, 14, 14, 0, 0, 0, 0, -12, 20, 26, 26, 8, 2, 0]
        kick = [0, 0, 0, 0, 0, 0, 0, 0, 0, -10, 8, 16, 16, 6, 0, 0]
        tail = [0, 4, 8, 10, 10, 0, 0, 0, 0, 12, -8, -14, -14, -6, 0, 0]
        bdy = [0, -1, -1, -2, -2, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0]
        bdx = [0, -1, -1, -1, -1, 0, 0, 0, 0, -1, 1, 2, 2, 1, 0, 0]
        p["arm"] = float(arm[f])
        p["arm2"] = float(arm2[f])
        p["kick"] = float(kick[f])
        p["tail"] = float(tail[f])
        p["bdy"] = int(bdy[f])
        p["bdx"] = int(bdx[f])
        p["mode"] = "up" if f <= 4 else "fwd"
        p["lift_all"] = True
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.64, 0.44, 0.26, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["flash"] = f == 1
        p["lift_all"] = False
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        # The other side is the still, at equal scale. finish() must not swing it.
        p["arm"] = 0.0
        p["arm2"] = 0.0
        p["kick"] = 0.0
        p["tail"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_a"] = (0, 0)
        p["step_b"] = (0, 0)
        p["spark"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["lift_all"] = False
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


def layer_motion(rig, p, name, step_key, angle_key, with_body):
    """Return angle, dx, dy for one part. Feet stay down unless lift_all."""
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    sx, sy = p[step_key] if step_key else (0, 0)
    ang = part_angle(rig, name, p["mode"], p[angle_key]) if angle_key else 0.0
    dy = int(sy)
    if with_body or p["lift_all"]:
        dy += bdy
    dx = bdx + int(round(fc * sx))
    return ang, dx, dy


def front_plan():
    # Far foot and tail sit behind the shell. The near flipper paints last.
    return (
        ("far", "step_b", None, False),
        ("tail", None, "tail", True),
        ("rarm", None, "arm2", True),
        ("torso", None, None, True),
        ("foot", "step_a", "kick", False),
        ("arm", None, "arm", True),
    )


def back_plan():
    return (
        ("right", "step_b", "arm2", False),
        ("torso", None, None, True),
        ("left", "step_a", "arm", True),
    )


def compose(rig, p):
    L = rig["layers"]
    piv = rig["pivots"]
    plan = front_plan() if rig["kind"] == "front" else back_plan()
    out = np.zeros_like(rig["canvas"])
    for name, step_key, angle_key, with_body in plan:
        if name == "torso":
            bdx = int(round(rig["facing"] * p["bdx"]))
            paste(out, shift(L["torso"], bdx, int(round(p["bdy"]))))
            continue
        ang, dx, dy = layer_motion(rig, p, name, step_key, angle_key, with_body)
        paste(out, moved(L[name], ang, piv[name], dx, dy))
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = any_color(out, INK_COLS)
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def limb_tip(rig, p, name, step_key, angle_key, with_body):
    ang, dx, dy = layer_motion(rig, p, name, step_key, angle_key, with_body)
    layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    fc = rig["facing"]
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (CYAN, WHITE, BLUE)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2), (2, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    other = "back" if kind == "front" else "front"
    return rigs[(other, mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    moved_part = (
        abs(p["arm"]) >= 0.45 or abs(p["arm2"]) >= 0.45
        or abs(p["tail"]) >= 0.45 or abs(p["kick"]) >= 0.45
        or p["step_a"] != (0, 0) or p["step_b"] != (0, 0)
        or p["bdx"] or p["bdy"]
    )
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        px, py = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], (px, py))
    if moved_part or abs(p["lean"]) > 0.4 or p["knock"]:
        body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    else:
        img = rig["canvas"].copy()
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"] and not rig["has_mouth"]:
        # The strike lands on the flipper that is already drawn. No invented mouth.
        name = "arm" if rig["kind"] == "front" else "left"
        step_key = None if rig["kind"] == "front" else "step_a"
        with_body = True
        tx, ty = limb_tip(rig, p, name, step_key, "arm", with_body)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def limb_lost(rig, p):
    body = compose(rig, p)
    plan = front_plan() if rig["kind"] == "front" else back_plan()
    lost = []
    for name, step_key, angle_key, with_body in plan:
        if name == "torso":
            continue
        if int((rig["layers"][name][..., 3] >= 128).sum()) == 0:
            continue
        ang, dx, dy = layer_motion(rig, p, name, step_key, angle_key, with_body)
        layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
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
    fm, finfo = segment_front(front)
    bm, binfo = segment_back(back)
    rigs = {}
    for kind, img, masks, info in (
        ("front", front, fm, finfo), ("back", back, bm, binfo),
    ):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror, info)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs[("front", False)]["facing"] != -1.0 or rigs[("back", False)]["facing"] != 1.0:
        raise SystemExit("facing locked to the wrong side")
    if rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("this still has no open mouth")
    return front, back, fm, bm, rigs, finfo, binfo


def save_parts(work, front, fm, back, bm):
    cols = {
        "arm": (40, 180, 220, 255),
        "rarm": (240, 160, 40, 255),
        "foot": (40, 200, 90, 255),
        "far": (220, 80, 80, 255),
        "tail": (200, 80, 200, 255),
        "left": (40, 180, 220, 255),
        "right": (220, 80, 80, 255),
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
        "dex": 565, "form": "normal", "variant": "carracosta", "gen": "gen5",
        "dir": "monsterSprite/gen5/0565_carracosta/normal", "sheetDir": "agent-sprite-forge",
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
    front, back, fm, bm, rigs, finfo, binfo = build_rigs()
    save_parts(work, front, fm, back, bm)
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS},
          "front", finfo, "back", binfo, flush=True)

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
            atk.append((f, round(float(p["arm"]), 1), round(float(p["arm2"]), 1),
                        round(float(p["kick"]), 1), p["mode"], p["turned"], p["spin"]))
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
        and partial == 0 and not clip and min(bodies) >= 0.90
    )
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print("qa_full", "rest_diff", int(sum(rest_diff.values())),
          "off_palette", len(off), "partial_alpha", partial,
          "edge_touch", len(clip), "body_main_min", report["body_main_min"],
          "idle0", idle0, flush=True)
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_carracosta_actions", "version": VERSION, "key": KEY,
        "dex": 565, "variant": "carracosta", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped turtle (two flippers, two feet, plastron, small tail, no open mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": (
            f"front faces screen-left (tip x {finfo['tip_x']} < eye x {finfo['eye_x']}); "
            f"back faces screen-right (tip x {binfo['tip_x']} > eye x {binfo['eye_x']})"
        ),
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
