#!/usr/bin/env python3
"""Krookodile normal (0553) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x7-84 y13-84, 3185 px, 14 colours,
alpha 0/255) and normal/back.png (bbox x3-83 y12-86, 2745 px, 10 colours).
Every back colour is already in the front palette. Front ground is y84.
Back ground is y86, so the back sits 2px higher and the feet share one line.

Facing: the pink eyes (230,123,115) are 26 px, mean x 49.96, y 18.3.
The snout tip is x7 at y43, and the white teeth sit at x16-27 y43-48.
The tip is left of the eyes, so the still faces screen-left. The tail is the
right curl (x68-84 y38-57), including the white claw pixels that only touch
that curl. There is a drawn mouth, so the roar uses a wedge and rings.
The back has no mouth: its red marks are the crest, the tail is the left
run x3-23 y59-65, and the right protrusion x70-83 y34-48 is the arm.
That back faces screen-right.

Front parts, screen axes as drawn:
  Lower jaw x7-22 y48-54, hinge against the upper snout.
  Tail x68-84 y38-57. Screen-left foot y74-84. Screen-right foot y74-84.
  Eyes, upper snout, belly, and red bands stay on the torso.
Back parts: left tail tip, right arm, two feet. No jaw is invented.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (screen-right). NW back mirrored.
  Attack 5-8 is that facing's back still, equal scale. No width squash,
  no in-plane body spin. 0-4 roar from the mouth. 9-12 the jaw and tail strike.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_krookodile_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0553_krookodile/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0553_krookodile/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0553_krookodile"
VERSION = "krookodile_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 208, 208
OFF_X = (CELL_W - 96) // 2
OFF_Y = 80
FRONT_FOOT = 84
BACK_FOOT = 86
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. Back as drawn looks screen-right (tail on the left).
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Strike ceilings this script already draws. finish() will not go past them.
PEAKS = {"jaw": 34.0, "tail": 30.0, "arm": 22.0}

INK = (16, 16, 16)
NAVY = (24, 24, 32)
DARK = (32, 32, 41)
BROWN = (82, 41, 32)
CLAW = (82, 82, 98)
MAROON = (139, 49, 65)
RED = (197, 82, 82)
EYE = (230, 123, 115)
WHITE = (255, 255, 255)

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
    """4x nearest in-plane rotation. Positive angle follows the shared forge formula. No squash."""
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


def component_list(mask, diag=False):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    groups = []
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diag:
        nbrs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
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
            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        groups.append(pts)
    groups.sort(key=len, reverse=True)
    return groups


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


def row_runs(op, y):
    xs = np.nonzero(op[y])[0]
    if len(xs) == 0:
        return []
    segs = []
    s = p = int(xs[0])
    for x in map(int, xs[1:]):
        if x > p + 1:
            segs.append((s, p))
            s = x
        p = x
    segs.append((s, p))
    return segs


def drop_specks(mask, torso, keep=8):
    groups = component_list(mask, diag=True)
    if len(groups) <= 1:
        return
    for g in groups[1:]:
        if len(g) >= keep:
            continue
        for y, x in g:
            mask[y, x] = False
            torso[y, x] = True


def absorb_specks(masks, names, limit=12):
    """A cut can orphan a few torso pixels against a limb. Give them back."""
    torso = masks["torso"]
    changed = True
    while changed:
        changed = False
        groups = component_list(torso, diag=True)
        if len(groups) <= 1:
            break
        for g in groups[1:]:
            if len(g) > limit:
                continue
            pix = np.zeros_like(torso)
            for y, x in g:
                pix[y, x] = True
            for name in names:
                if (pix & dilate(masks[name], 1)).any():
                    masks[name] |= pix
                    torso[pix] = False
                    changed = True
                    break
    masks["torso"] = torso


def claim(order, op):
    acc = np.zeros_like(op)
    masks = {}
    names = []
    for name, m in order:
        m = m & ~acc & op
        masks[name] = m
        acc |= m
        names.append(name)
    masks["torso"] = op & ~acc
    for name in names:
        drop_specks(masks[name], masks["torso"])
    absorb_specks(masks, names)
    check_partition(op, masks)
    return masks


def require_solid(masks, names):
    torso = masks["torso"]
    for name in names:
        part = masks[name]
        if int(part.sum()) < 4:
            raise SystemExit(f"{name} is missing from the still")
        if not (part & dilate(torso, 1)).any():
            raise SystemExit(f"{name} does not touch the body")
        groups = component_list(part, diag=True)
        if len(groups[0]) != int(part.sum()):
            bits = []
            for g in groups[:6]:
                ys = [p[0] for p in g]
                xs = [p[1] for p in g]
                bits.append(f"n{len(g)}@{min(xs)},{min(ys)}")
            raise SystemExit(f"{name} fell apart ({len(groups)} pieces: {', '.join(bits)})")
    groups = component_list(torso, diag=True)
    if len(groups) > 1 and len(groups[1]) > 8:
        g = groups[1]
        ys = [p[0] for p in g]
        xs = [p[1] for p in g]
        raise SystemExit(
            f"torso fell apart ({len(groups)} pieces, next {len(g)} at {min(xs)},{min(ys)})"
        )


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    # Lower mandible under the tooth gap. Upper snout and eyes stay on the torso.
    jaw = op & (yy >= 48) & (yy <= 54) & (xx <= 22)
    # Right curl, including the white claws that only touch this curl.
    tail = op & (xx >= 68) & (yy >= 38) & (yy <= 58)
    leg_l = op & (yy >= 74) & (xx <= 38)
    leg_r = op & (yy >= 74) & (xx >= 58)
    masks = claim(
        (("jaw", jaw), ("tail", tail), ("leg_l", leg_l), ("leg_r", leg_r)),
        op,
    )
    eye = is_color(img, EYE)
    if int(eye.sum()) != 26 or int((eye & masks["torso"]).sum()) != 26:
        raise SystemExit("front eyes left the head")
    eye_x = float(np.nonzero(eye)[1].mean())
    tip_x = int(np.nonzero(op[30:56])[1].min())
    if tip_x >= eye_x - 20:
        raise SystemExit(f"front snout is not left of the eyes ({tip_x} vs {eye_x:.1f})")
    teeth = is_color(img, WHITE) & (yy >= 42) & (yy <= 50) & (xx <= 36)
    if int(teeth.sum()) < 8:
        raise SystemExit("front mouth teeth are missing")
    if int((teeth & (masks["jaw"] | masks["torso"])).sum()) != int(teeth.sum()):
        raise SystemExit("front teeth left the mouth")
    yb = int(np.nonzero(op)[0].max())
    if yb != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    if int(np.nonzero(masks["leg_l"])[0].max()) != FRONT_FOOT:
        raise SystemExit("front left foot does not reach the ground")
    if int(np.nonzero(masks["leg_r"])[0].max()) != FRONT_FOOT:
        raise SystemExit("front right foot does not reach the ground")
    require_solid(masks, ("jaw", "tail", "leg_l", "leg_r"))
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    tail = np.zeros_like(op)
    for y in range(54, 72):
        runs = row_runs(op, y)
        if not runs:
            continue
        a, b = runs[0]
        if a <= 12 and b <= 26:
            tail[y, a:b + 1] = True
    arm = np.zeros_like(op)
    for y in range(32, 50):
        runs = row_runs(op, y)
        if not runs:
            continue
        a, b = runs[-1]
        if a >= 70:
            arm[y, a:b + 1] = True
    arm |= op & (xx >= 76) & (yy >= 33) & (yy <= 48)
    # The y46-47 pixels between the arm outline and the ribs, measured as one orphan.
    arm |= op & (yy >= 46) & (yy <= 47) & (xx >= 71) & (xx <= 75)
    leg_l = op & (yy >= 78) & (xx <= 55)
    leg_r = op & (yy >= 78) & (xx >= 58)
    masks = claim(
        (("arm", arm), ("tail", tail), ("leg_l", leg_l), ("leg_r", leg_r)),
        op,
    )
    teeth = is_color(img, WHITE) & (yy >= 42) & (yy <= 50) & (xx <= 36)
    if int(teeth.sum()) != 0:
        raise SystemExit("back gained a mouth")
    crest = is_color(img, EYE)
    if int(crest.sum()) < 8 or int((crest & masks["torso"]).sum()) != int(crest.sum()):
        raise SystemExit("back crest left the torso")
    tail_x = float(np.nonzero(masks["tail"])[1].mean())
    arm_x = float(np.nonzero(masks["arm"])[1].mean())
    if tail_x >= 30 or arm_x <= 70:
        raise SystemExit(f"back sides swapped: tail {tail_x:.1f} arm {arm_x:.1f}")
    if int(np.nonzero(op)[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line changed")
    if int(np.nonzero(masks["leg_r"])[0].max()) != BACK_FOOT:
        raise SystemExit("back forward foot is not the right leg")
    require_solid(masks, ("arm", "tail", "leg_l", "leg_r"))
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
    names = [k for k in ("tail", "leg_l", "leg_r", "arm", "jaw") if k in placed]
    depth_of = {"jaw": 8, "tail": 6, "arm": 6, "leg_l": 5, "leg_r": 5}
    for name in names:
        if int(placed[name].sum()) < 4:
            continue
        depth = depth_of[name]
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in names:
        if int((layers[name][..., 3] >= 128).sum()) < 4:
            continue
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    has_mouth = False
    if kind == "front":
        teeth = is_color(img, WHITE)
        yy, xx = np.mgrid[0:96, 0:96]
        teeth = teeth & (yy >= 42) & (yy <= 50) & (xx <= 36)
        if teeth.any():
            has_mouth = True
            ty, tx = np.nonzero(teeth)
            i = int(tx.argmin())
            mx = OFF_X + float(tx[i])
            my = OFF_Y + y_shift + float(ty[i])
            if mirror:
                mx = CELL_W - 1 - mx
            pivots["muzzle"] = (mx, my)
    if not has_mouth:
        pivots["muzzle"] = pivots["center"]
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas, "has_mouth": has_mouth,
        "limbs": tuple(names),
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        jaw=0.0, tail=0.0, arm=0.0,
        jaw_mode="open", tail_mode="up", arm_mode="up",
        step_l=(0, 0), step_r=(0, 0),
        lean=0.0, knock=0, flash=False,
        wedge=0, ring=-1, spark=0,
        turned=False, spin=0, head_dx=0, head_dy=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay planted. The jaw breathes and the tail sways. No head slide.
        p["jaw"] = wave(f, n, 5.0, phase=0.4)
        p["tail"] = wave(f, n, 6.0, phase=0.8)
        p["arm"] = wave(f, n, 3.0, phase=0.2)
        p["jaw_mode"] = "open"
        p["tail_mode"] = "up"
        p["arm_mode"] = "up"
    elif action == "walk":
        # Two drawn feet, opposite steps. One foot stays down each frame.
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * sb)), -int(round(3 * max(0.0, sb))))
        p["tail"] = wave(f, n, 8.0, phase=math.pi)
        p["jaw"] = wave(f, n, 3.0, phase=0.3)
        p["arm"] = wave(f, n, 6.0, phase=0.2)
        p["jaw_mode"] = "open"
        p["tail_mode"] = "up"
        p["arm_mode"] = "fwd"
    elif action == "attack":
        # 0 rest. 1-4 the mouth opens (wedge and ring are drawn later).
        # 5-8 the back still, equal scale. 9-12 jaw bite and tail slap.
        # 13-15 recover. The head does not slide.
        jaw = [0, 8, 14, 18, 18, 0, 0, 0, 0, 22, 30, 34, 34, 14, 6, 0]
        tail = [0, 4, 8, 12, 12, 0, 0, 0, 0, 16, 24, 30, 28, 12, 4, 0]
        arm = [0, 4, 8, 12, 12, 0, 0, 0, 0, 12, 18, 22, 20, 10, 4, 0]
        p["jaw"] = float(jaw[f])
        p["tail"] = float(tail[f])
        p["arm"] = float(arm[f])
        p["jaw_mode"] = "fwd" if 9 <= f <= 14 else "open"
        p["tail_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["arm_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["wedge"] = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["ring"] = [-1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1][f]
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.64, 0.42, 0.24, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(4 * env))
        p["jaw"] = 8.0 * env
        p["tail"] = 10.0 * env
        p["arm"] = 6.0 * env
        p["jaw_mode"] = "open"
        p["tail_mode"] = "up"
        p["arm_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    # Frames 5-8 are the back still at equal scale. finish() would open the
    # strike on these frames; drop it so the turned silhouette stays whole.
    if action == "attack" and 5 <= f <= 8:
        p["jaw"] = 0.0
        p["tail"] = 0.0
        p["arm"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["wedge"] = 0
        p["ring"] = -1
        p["spark"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["head_dx"] = 0
        p["head_dy"] = 0
        p["turned"] = True
    return p


def part_angle(rig, name, mode, amp):
    if abs(amp) < 0.45 or name not in rig["signs"]:
        return 0.0
    sign = rig["signs"][name]
    if mode == "up":
        return sign["up"] * amp
    if mode == "open":
        return -sign["up"] * amp
    return sign["fwd"] * amp


def moved(layer, ang, pivot, dx, dy):
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, pivot)
    return shift(layer, dx, dy)


def limb_motion(rig, p, name):
    if name not in rig["pivots"]:
        return 0.0, 0, 0
    if name == "leg_l":
        sx, sy = p["step_l"]
        ang = 0.0
    elif name == "leg_r":
        sx, sy = p["step_r"]
        ang = 0.0
    elif name == "jaw":
        sx, sy = 0, 0
        ang = part_angle(rig, name, p["jaw_mode"], p["jaw"])
    elif name == "tail":
        sx, sy = 0, 0
        ang = part_angle(rig, name, p["tail_mode"], p["tail"])
    elif name == "arm":
        sx, sy = 0, 0
        ang = part_angle(rig, name, p["arm_mode"], p["arm"])
    else:
        sx, sy = 0, 0
        ang = 0.0
    fc = rig["facing"]
    dx = int(round(fc * p["bdx"])) + int(round(fc * sx))
    dy = int(round(p["bdy"])) + int(sy)
    return ang, dx, dy


def compose(rig, p):
    L = rig["layers"]
    bdx = int(round(rig["facing"] * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    for name in ("tail", "leg_l", "leg_r", "arm", "jaw"):
        if name not in L:
            continue
        ang, dx, dy = limb_motion(rig, p, name)
        if name not in rig["pivots"]:
            paste(out, shift(L[name], dx, dy))
            continue
        paste(out, moved(L[name], ang, rig["pivots"][name], dx, dy))
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


def mouth_wedge(img, origin, fc, age):
    """A few pixels in front of the existing teeth. Empty pixels only."""
    ox, oy = origin
    cols = (INK, RED, WHITE)
    for i in range(max(0, int(age))):
        x = ox + fc * (2 + i)
        span = 1 if i == 0 else 2
        for dy in range(-span, span + 1):
            put(img, x, oy + dy, cols[min(i, 2)])


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, EYE, RED)
    rad = 7 + age * 3
    for deg in range(-50, 51, 10):
        a = math.radians(deg)
        x = ox + fc * rad * math.cos(a)
        y = oy + rad * math.sin(a) * 0.55
        put(img, x, y, cols[age % 3])


def weapon_tip(rig, p):
    fc = rig["facing"]
    order = ("jaw", "arm", "tail")
    for name in order:
        if name not in rig["pivots"]:
            continue
        ang, dx, dy = limb_motion(rig, p, name)
        if abs(ang) < 0.45 and dx == 0 and dy == 0:
            continue
        layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
        m = layer[..., 3] >= 128
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
        return float(xs[i]), float(ys[i])
    return rig["pivots"]["center"]


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, RED, EYE, MAROON)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r), (2, r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    # Same facing, the back still. N stays on its own back. No width scale.
    facing = rigs[d]["facing"]
    return rigs[("back", facing < 0)]


def moving(p):
    return (
        abs(p["jaw"]) >= 0.45 or abs(p["tail"]) >= 0.45 or abs(p["arm"]) >= 0.45
        or p["step_l"] != (0, 0) or p["step_r"] != (0, 0)
        or p["bdx"] or p["bdy"] or p["knock"] or abs(p["lean"]) > 0.4
    )


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    if not moving(p):
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        fc = rig["facing"]
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            # Measured on this still: -facing * lean sends the eyes backward.
            body = rotate_nearest(body, -fc * p["lean"], rear_pivot(body, fc))
        body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    if not effects:
        return img
    fc = rig["facing"]
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if rig["has_mouth"] and not p["turned"]:
        if p["wedge"]:
            mouth_wedge(img, rig["pivots"]["muzzle"], fc, p["wedge"])
        if p["ring"] >= 0:
            roar_rings(img, rig["pivots"]["muzzle"], fc, p["ring"])
        if p["wedge"] or p["ring"] >= 0:
            img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = weapon_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    groups = component_list(op)
    return len(groups[0]) / float(op.sum())


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name in rig["limbs"]:
        if name not in rig["pivots"]:
            continue
        ang, dx, dy = limb_motion(rig, p, name)
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


def crop_box(canvas, pad=8):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad - 18)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 16)
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
    if rigs[("front", False)]["facing"] != -1.0 or rigs[("back", False)]["facing"] != 1.0:
        raise SystemExit("facing locked to the wrong side")
    if not rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("mouth is on the wrong view")
    for d in DIRS:
        p = base_pose()
        p["turned"] = True
        back_rig = rig_for(rigs, d, p)
        if back_rig["kind"] != "back" or back_rig["facing"] != rigs[d]["facing"]:
            raise SystemExit(f"{d} turn does not keep the back at the same facing")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "jaw": (255, 80, 40, 255),
        "arm": (40, 180, 255, 255),
        "tail": (180, 70, 220, 255),
        "leg_l": (40, 210, 80, 255),
        "leg_r": (80, 255, 180, 255),
        "torso": (170, 170, 170, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 4, 0), 4, 1))
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
        "dex": 553, "form": "normal", "variant": "krookodile", "gen": "gen5",
        "dir": "monsterSprite/gen5/0553_krookodile/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def frame_faults(img, palette_set):
    partial = int(((img[..., 3] != 0) & (img[..., 3] != 255)).sum())
    op = img[..., 3] == 255
    off = set()
    if op.any():
        used = {tuple(int(v) for v in c) for c in np.unique(img[op][:, :3], axis=0)}
        off = used - palette_set
    edge = bool(op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any())
    return off, partial, edge


def eye_dx(rest, img):
    def cx(im):
        m = is_color(im, EYE)
        if not m.any():
            return None
        return float(np.nonzero(m)[1].mean())
    a, b = cx(rest), cx(img)
    if a is None or b is None:
        return None
    return b - a


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
    print("facing", facing, flush=True)
    print("mouth", {d: rigs[d]["has_mouth"] for d in DIRS}, flush=True)

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

    # Hurt frame 3 is the peak lean. Eyes must travel backward, away from facing.
    for d, want in (("S", -1), ("W", 1)):
        hurt = synth(rigs, d, "hurt", 3, palette, effects=False)
        dx = eye_dx(rigs[d]["canvas"], hurt)
        print("hurt_eye_dx", d, None if dx is None else round(dx, 1), flush=True)
        if dx is None or dx * want < 8:
            raise SystemExit(f"hurt lean on {d} does not send the eyes backward")

    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        y0, y1, x0, x1 = box
        pal = {tuple(int(v) for v in c) for c in palette}
        off = set()
        partial = 0
        clip = []
        bodies = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    img = synth(rigs, d, act, f, palette)
                    bad, part, edge = frame_faults(img, pal)
                    off |= bad
                    partial += part
                    if edge:
                        clip.append(f"{act}:{d}:{f}")
                    if act in ("attack", "hurt", "walk"):
                        bodies.append(largest_fraction(img))
        for d in ("S", "W", "N", "E"):
            rows = []
            for act in ACTIONS:
                frames = []
                for f in range(16):
                    if f < FRAMES[act]:
                        frames.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
                    else:
                        frames.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
                rows.append(tile_grid(frames, 16))
            sheet = np.concatenate(rows, axis=0)
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
            atk.append((f, round(p["jaw"], 1), round(p["tail"], 1), round(p["arm"], 1),
                        p["jaw_mode"], p["turned"], p["spin"], p["wedge"], p["ring"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        print("qa_numbers", {
            "rest_diff": int(sum(rest_diff.values())),
            "off_palette_colors": len(off),
            "partial_alpha": partial,
            "edge_n": len(clip),
            "edge_touching": clip[:12],
            "body_main_min": round(min(bodies), 3) if bodies else 1,
            "disconnected": len(lost),
        }, flush=True)
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
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.concatenate(strips, axis=0)
    motion = np.repeat(np.repeat(motion, 2, 0), 2, 1)
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
    ok = (not any(rest_diff.values()) and not any(idle0.values()) and not off
          and partial == 0 and not clip)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_krookodile_actions", "version": VERSION, "key": KEY,
        "dex": 553, "variant": "krookodile", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped crocodile (lower jaw, tail, two feet, mouth with teeth; back arm and tail, no mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (snout tip x7 < eye mean x 49.96); back tail is on the left so it faces screen-right",
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
