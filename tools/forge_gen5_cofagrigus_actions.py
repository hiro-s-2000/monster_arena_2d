#!/usr/bin/env python3
"""Cofagrigus normal (0563) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x3-93 y10-85, 2963 px, 13 colours,
alpha 0/255, bottom y85). One connected silhouette.
  Eyes are red (255, 49, 32) plus dark red (156, 32, 65): left cluster
  x39-44 y40-43 (mean x 41.47), right cluster x49-54 y36-42 (mean x 51.18).
  Combined eye mean x is 47.24.
  The mouth is the gray (123, 123, 123) and pale (180, 189, 180) cluster
  at x45-51 y45-49, mean x 48.08, 1.75 px right of the eye midpoint.
  The still faces screen-right.
  No feet, wings, horns, tail, or pincers. The coffin floats. Four shadow
  arms are already drawn, gray (57, 57, 65), outside the coffin column
  x30-70:
    right upper x71-89 y10-47, left upper x3-29 y32-55,
    right lower x71-93 y51-71, left lower x6-29 y62-83.
  Gold trim and the face stay on the coffin. The arms are not given hands
  the still does not draw.
normal/back.png (bbox x2-92 y7-79, 2894 px, 9 colours, all in the front
  palette, bottom y79): no eyes and no mouth. Four arms outside x28-62:
    right upper x63-85 y7-45, left upper x2-27 y20-50,
    right lower x63-92 y43-68, left lower x5-27 y57-76.

Directions, front faces screen-right:
  S SE E front as drawn. W SW front mirrored.
  NE back as drawn. N NW back mirrored.
  E is screen-right, W is screen-left, N is the back row.
  Attack 5-8 shows that direction's back at equal scale (N, or N flipped).
  No |cos| squash and no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_cofagrigus_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0563_cofagrigus/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0563_cofagrigus/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0563_cofagrigus"
VERSION = "cofagrigus_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 256, 256
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 28
FRONT_FOOT = 85
BACK_FOOT = 79
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-right. E keeps that picture. Its back is NE,
# which is N flipped at equal scale. W is the mirrored front, so its back is N.
VIEW_OF = {
    "S": ("front", False), "SE": ("front", False), "E": ("front", False),
    "NE": ("back", False),
    "N": ("back", True), "NW": ("back", True),
    "W": ("front", True), "SW": ("front", True),
}
MAX_LEAN = 38.0
# finish() may push the arm key this far. The slap is drawn at 34, above
# this, so the hit is not stretched past the angle this script already draws.
PEAKS = {"arms": 18.0}
ARMS = ("arm_lu", "arm_ll", "arm_ru", "arm_rl")
ARM_ORDER = ("arm_ll", "arm_rl", "arm_lu", "arm_ru")
LEFT_ARMS = ("arm_lu", "arm_ll")
# Cuts measured so each still splits into one coffin and four arms.
FRONT_CUT = (30, 70)
BACK_CUT = (28, 62)

INK = (16, 16, 16)
BODY = (57, 57, 65)
GOLD = (255, 189, 49)
RED = (255, 49, 32)
PUPIL = (156, 32, 65)
MOUTH = (123, 123, 123)
PALE = (180, 189, 180)
WHITE = (255, 255, 255)
EYE_COLS = (RED, PUPIL)
MOUTH_COLS = (MOUTH, PALE)

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
    """4x nearest in-plane rotation. Positive is clockwise on screen. No squash."""
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
        grown = []
        up = np.zeros_like(src)
        dn = np.zeros_like(src)
        lt = np.zeros_like(src)
        rt = np.zeros_like(src)
        up[:-1] = src[1:]
        dn[1:] = src[:-1]
        lt[:, :-1] = src[:, 1:]
        rt[:, 1:] = src[:, :-1]
        grown.extend((up, dn, lt, rt))
        for g in grown:
            take = (acc[..., 3] < 128) & allowed & (g[..., 3] >= 128)
            acc[take] = g[take]
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


def touches(part, body):
    return bool((part & dilate(body, 1)).any())


def label_arms(arm_masks):
    stats = []
    for m in arm_masks:
        ys, xs = np.nonzero(m)
        stats.append((float(xs.mean()), float(ys.mean()), m))
    cx = sum(s[0] for s in stats) / float(len(stats))
    left = sorted([s for s in stats if s[0] < cx], key=lambda s: s[1])
    right = sorted([s for s in stats if s[0] >= cx], key=lambda s: s[1])
    if len(left) != 2 or len(right) != 2:
        raise SystemExit(f"expected two arms per side, got {len(left)} and {len(right)}")
    return {
        "arm_lu": left[0][2],
        "arm_ll": left[1][2],
        "arm_ru": right[0][2],
        "arm_rl": right[1][2],
    }


def split_arms(img, lo, hi):
    """Four arms already outside the coffin column. The column is the coffin."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    outer = op & ((xx < lo) | (xx > hi))
    groups = component_list(outer)
    big = [g for g in groups if len(g) >= 80]
    small = [g for g in groups if len(g) < 80]
    if len(big) != 4:
        raise SystemExit(f"expected 4 arms, got {len(big)}")
    arms = [mask_from_pts(op.shape, g) for g in big]
    for g in small:
        speck = mask_from_pts(op.shape, g)
        near = dilate(speck, 1)
        hit = None
        for i, arm in enumerate(arms):
            if (near & arm).any():
                hit = i
                break
        if hit is not None:
            arms[hit] |= speck
    named = label_arms(arms)
    torso = op.copy()
    for m in named.values():
        torso &= ~m
    groups = component_list(torso)
    if not groups:
        raise SystemExit("coffin is empty")
    groups.sort(key=len, reverse=True)
    main = mask_from_pts(op.shape, groups[0])
    for g in groups[1:]:
        speck = mask_from_pts(op.shape, g)
        if (dilate(speck, 2) & main).any():
            main |= speck
            continue
        near = dilate(speck, 1)
        hit = None
        for name, arm in named.items():
            if (near & arm).any():
                hit = name
                break
        if hit is None:
            raise SystemExit("a coffin speck touches nothing")
        named[hit] |= speck
    named["torso"] = main
    check_partition(op, named)
    if largest_component(main) != int(main.sum()):
        raise SystemExit("coffin fell apart")
    for name in ARMS:
        if not touches(named[name], main):
            raise SystemExit(f"{name} does not touch the coffin")
        if largest_component(named[name]) != int(named[name].sum()):
            raise SystemExit(f"{name} fell apart")
    return named


def segment_front(img):
    op = img[..., 3] >= 128
    masks = split_arms(img, *FRONT_CUT)
    eyes = any_color(img, EYE_COLS)
    mouth = any_color(img, MOUTH_COLS)
    if int(eyes.sum()) < 20 or int((eyes & masks["torso"]).sum()) != int(eyes.sum()):
        raise SystemExit("front eyes left the coffin")
    if int(mouth.sum()) < 8 or int((mouth & masks["torso"]).sum()) != int(mouth.sum()):
        raise SystemExit("front mouth left the coffin")
    eye_x = float(np.nonzero(eyes)[1].mean())
    mouth_x = float(np.nonzero(mouth)[1].mean())
    if mouth_x <= eye_x:
        raise SystemExit(f"front does not face right: mouth {mouth_x:.2f} eye {eye_x:.2f}")
    ys = np.nonzero(op)[0]
    if int(ys.max()) != FRONT_FOOT:
        raise SystemExit("front bottom line changed")
    bottom = op & (np.mgrid[0:96, 0:96][0] >= FRONT_FOOT)
    if not bottom.any() or int((bottom & masks["torso"]).sum()) != int(bottom.sum()):
        raise SystemExit("front bottom is not the coffin")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    masks = split_arms(img, *BACK_CUT)
    if any_color(img, EYE_COLS + MOUTH_COLS).any():
        raise SystemExit("back gained a face")
    ys = np.nonzero(op)[0]
    if int(ys.max()) != BACK_FOOT:
        raise SystemExit("back bottom line changed")
    return masks


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


def silhouette_tip(canvas, y, facing):
    y = int(round(y))
    h = canvas.shape[0]
    for dy in (0, -1, 1, -2, 2, -3, 3):
        yy = y + dy
        if yy < 0 or yy >= h:
            continue
        xs = np.nonzero(canvas[yy, :, 3] >= 128)[0]
        if len(xs) == 0:
            continue
        x = float(xs.max() if facing > 0 else xs.min())
        return x, float(yy)
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    i = int(xs.argmax() if facing > 0 else xs.argmin())
    return float(xs[i]), float(ys[i])


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
    if kind == "front":
        facing = -1.0 if mirror else 1.0
    else:
        facing = 1.0 if mirror else -1.0
    pivots = {}
    signs = {}
    for name in ARMS:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    torso_m = placed["torso"]
    for name in ARMS:
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, 8), 8,
        )
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    mouth = any_color(canvas, MOUTH_COLS)
    has_mouth = bool(mouth.any())
    if has_mouth:
        my, mx = np.nonzero(mouth)
        pivots["mouth"] = (float(mx.mean()), float(my.mean()))
        pivots["muzzle"] = silhouette_tip(canvas, float(my.mean()), facing)
    else:
        pivots["mouth"] = pivots["center"]
        pivots["muzzle"] = pivots["center"]
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "has_mouth": has_mouth,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        sway_l=0.0, sway_r=0.0,
        lift=0.0, arms=0.0,
        wedge=0, ring=-1,
        lean=0.0, knock=0, flash=False,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # The coffin floats. Frame 0 is the still. Arms sway, they do not walk.
        p["bdy"] = -1 if wave(f, n, 1.0, phase=0.25) > 0.55 else 0
        p["sway_l"] = wave(f, n, 5.0, phase=0.4)
        p["sway_r"] = wave(f, n, 5.0, phase=0.4 + math.pi)
    elif action == "walk":
        # No feet in the still, so no steps. The four arms row and the coffin bobs.
        p["bdy"] = -1 if wave(f, n, 1.0, phase=0.15) > 0.45 else 0
        p["sway_l"] = wave(f, n, 11.0, phase=0.0)
        p["sway_r"] = wave(f, n, 11.0, phase=math.pi)
    elif action == "attack":
        # 0 rest. 1-4 roar with the mouth that is drawn, arms lift.
        # 5-8 the back, equal scale. 9-10 cock the arms. 11-12 slap.
        # 13-15 return. 15 is the still.
        lift = [0, 4, 8, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        arms = [0, 0, 0, 0, 0, 0, 0, 0, 0, -18, -22, 34, 34, 12, 4, 0]
        p["lift"] = float(lift[f])
        p["arms"] = float(arms[f])
        p["turned"] = 5 <= f <= 8
        if 1 <= f <= 4:
            p["wedge"] = [0, 1, 2, 3, 3][f]
        if 2 <= f <= 4:
            p["ring"] = f - 2
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.64, 0.44, 0.26, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        # The back is the still, at equal scale. finish() must not swing it.
        p["sway_l"] = 0.0
        p["sway_r"] = 0.0
        p["lift"] = 0.0
        p["arms"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["wedge"] = 0
        p["ring"] = -1
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["turned"] = True
    if action == "attack" and abs(float(p["arms"])) > 34.05:
        p["arms"] = 34.0 if p["arms"] > 0 else -22.0
    return p


def arm_angle(rig, name, p):
    sign = rig["signs"][name]
    ang = sign["up"] * float(p["lift"])
    ang += sign["fwd"] * float(p["arms"])
    if name in LEFT_ARMS:
        ang += sign["fwd"] * float(p["sway_l"])
    else:
        ang += sign["fwd"] * float(p["sway_r"])
    return ang


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
    for name in ARM_ORDER:
        ang = arm_angle(rig, name, p)
        paste(out, moved(L[name], ang, piv[name], bdx, bdy))
    return out


def bridge_cracks(img):
    """Close a 1px resampling crack so a rotated arm stays on the coffin."""
    out = img.copy()
    op = out[..., 3] >= 128
    groups = component_list(op)
    if len(groups) <= 1:
        return out
    groups.sort(key=len, reverse=True)
    main = mask_from_pts(op.shape, groups[0])
    h, w = op.shape
    for g in groups[1:]:
        if len(g) < 20:
            continue
        part = mask_from_pts(op.shape, g)
        crack = dilate(main, 1) & dilate(part, 1) & (out[..., 3] < 128)
        if not crack.any():
            continue
        ys, xs = np.nonzero(crack)
        for y, x in zip(ys.tolist(), xs.tolist()):
            src = None
            for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and out[ny, nx, 3] >= 128:
                    src = out[ny, nx].copy()
                    break
            if src is None:
                continue
            out[y, x] = src
            out[y, x, 3] = 255
        main = main | part | crack
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = any_color(out, (INK, RED, PUPIL))
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
    """A few pixels in front of the existing mouth. Empty pixels only."""
    ox, oy = origin
    cols = (INK, RED, GOLD)
    for i in range(max(0, int(age))):
        x = ox + fc * (2 + i)
        span = 1 if i == 0 else 2
        for dy in range(-span, span + 1):
            put(img, x, oy + dy, cols[min(i, 2)])


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, GOLD, RED)
    rad = 8 + age * 4
    for deg in range(-50, 51, 8):
        a = math.radians(deg)
        x = ox + fc * (4 + rad * math.cos(a))
        y = oy + rad * math.sin(a) * 0.55
        put(img, x, y, cols[age % 3])


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    other = "back" if kind == "front" else "front"
    return rigs[(other, mirror)]


def motionless(p):
    return (
        abs(p["sway_l"]) < 0.45 and abs(p["sway_r"]) < 0.45
        and abs(p["lift"]) < 0.45 and abs(p["arms"]) < 0.45
        and not p["bdx"] and not p["bdy"]
        and abs(p["lean"]) < 0.4 and not p["knock"]
    )


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    if motionless(p):
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            px, py = rear_pivot(body, fc)
            body = rotate_nearest(body, -fc * p["lean"], (px, py))
            body = bridge_cracks(body)
        body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if rig["has_mouth"] and not p["turned"]:
        if p["wedge"]:
            mouth_wedge(img, rig["pivots"]["muzzle"], fc, p["wedge"])
        if p["ring"] >= 0:
            roar_rings(img, rig["pivots"]["muzzle"], fc, p["ring"])
        if p["wedge"] or p["ring"] >= 0:
            img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def limb_lost(rig, p):
    if motionless(p):
        return []
    body = compose(rig, p)
    if p["knock"]:
        body = shift(body, int(round(-rig["facing"] * p["knock"])), 0)
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    lost = []
    for name in ARMS:
        ang = arm_angle(rig, name, p)
        layer = moved(L[name], ang, piv[name], bdx, bdy)
        if p["knock"]:
            layer = shift(layer, int(round(-fc * p["knock"])), 0)
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
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs[("front", False)]["facing"] != 1.0 or rigs[("back", False)]["facing"] != -1.0:
        raise SystemExit("facing locked to the wrong side")
    if not rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("mouth is on the wrong view")
    back_l = rigs[("back", False)]["canvas"]
    back_r = rigs[("back", True)]["canvas"]
    if np.any(back_r != back_l[:, ::-1]):
        raise SystemExit("N is not the equal-scale flip of the back still")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "arm_lu": (220, 60, 40, 255),
        "arm_ll": (40, 200, 90, 255),
        "arm_ru": (40, 160, 220, 255),
        "arm_rl": (255, 180, 40, 255),
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
        "dex": 563, "form": "normal", "variant": "cofagrigus", "gen": "gen5",
        "dir": "monsterSprite/gen5/0563_cofagrigus/normal", "sheetDir": "agent-sprite-forge",
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

    turn_e = synth(rigs, "E", "attack", 6, palette, effects=False)
    turn_w = synth(rigs, "W", "attack", 6, palette, effects=False)
    turn_s = synth(rigs, "S", "attack", 6, palette, effects=False)
    if np.any(turn_e != rigs["NE"]["canvas"]) or np.any(turn_s != rigs["NE"]["canvas"]):
        raise SystemExit("E/S turn is not the unmirrored back at equal scale")
    if np.any(turn_w != rigs["N"]["canvas"]):
        raise SystemExit("W turn is not the N back at equal scale")
    if np.any(rigs["N"]["canvas"] != rigs["NE"]["canvas"][:, ::-1]):
        raise SystemExit("N is not NE flipped at equal scale")

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
            atk.append((f, round(float(p["arms"]), 1), round(float(p["lift"]), 1),
                        p["turned"], p["spin"], p["wedge"]))
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
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_cofagrigus_actions", "version": VERSION, "key": KEY,
        "dex": 563, "variant": "cofagrigus", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "floating sarcophagus (open mouth, four shadow arms, no feet)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-right (mouth mean x 48.08 > eye mean x 47.24)",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
