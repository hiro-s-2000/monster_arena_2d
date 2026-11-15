#!/usr/bin/env python3
"""Galarian Darmanitan Zen (0555) 8-direction actions from this form's own stills.

Measured on regional/darmanitan-galar-zen/front.png (96x96, bbox x24-75 y3-69,
1927 px, 15 colours, alpha 0/255) and back.png (bbox x25-68 y4-68, 1670 px,
13 colours). Every back colour is already in the front palette. Front ground
is y69. Back ground is y68, so the back sits one pixel lower and both feet
share that line.

Facing: the pale eye slit is x44-51 y35, mean x 47.5. The two red brows sit
at mean x 41.4 and x 54.0. The open mouth (pink interior plus white teeth,
218 px) runs x34-57, centroid x 45.39, left of the eye slit. The head band
y16-36 is centred at x 51.16, so the face sits left of the skull. The still
faces screen-left. The head flame starts at x27 y3, ahead of the face.
The back has the flame on the right (x50-68), no eye, no teeth, and no mouth.

Parts that are actually drawn:
  Head flame, cut at y<=20 so the brows and the belly mark stay on the body.
  Two brown mittens. The front ones are 166 and 133 px; the back ones are
  181 and 31 px. No separate feet: the base is one round belly, so the walk
  does not grow legs. It swings the mittens and the flame.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn. NW back mirrored.
  Attack 5-8 uses the other side at equal scale. No width squash, no
  in-plane body spin.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_darmanitan-galar-zen_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0555_darmanitan-standard/regional/darmanitan-galar-zen"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0555_darmanitan-standard/regional/darmanitan-galar-zen/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0555_darmanitan-galar-zen"
VERSION = "darmanitan-galar-zen_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 208, 208
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 16
FRONT_FOOT = 69
BACK_FOOT = 68
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. Back as drawn has the flame and the big mitten on the right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# finish() may raise these during frames 5-12. The punch is authored higher.
PEAKS = {"hand_big": 16.0, "hand_small": 12.0, "flame": 8.0}
LIMBS = ("flame", "hand_big", "hand_small")

INK = (0, 0, 0)
WHITE = (246, 246, 246)
PINK = (206, 58, 51)
RED = (201, 48, 35)
DRED = (146, 27, 27)
ORANGE = (233, 122, 40)
YELLOW = (238, 248, 162)
GOLD = (239, 166, 61)
FLAME_COLS = (DRED, RED, ORANGE, YELLOW, GOLD)
BROWN_COLS = ((112, 57, 24), (181, 87, 51), (226, 121, 58))
PALE = (211, 216, 213)

SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)


def is_color(img, color):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(color, np.uint8), axis=-1)


def is_cols(img, cols):
    mask = np.zeros(img.shape[:2], bool)
    op = img[..., 3] >= 128
    for color in cols:
        mask |= op & np.all(img[..., :3] == np.array(color, np.uint8), axis=-1)
    return mask


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
    """4x nearest in-plane rotation. Positive angle moves the top to the right. No squash."""
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


def mask_from(points, shape):
    mask = np.zeros(shape, bool)
    for y, x in points:
        mask[y, x] = True
    return mask


def ink_outline(part, op, ink):
    other = op & ~part & ~ink
    return ink & dilate(part, 1) & ~dilate(other, 1)


def flame_mask(img, op):
    """Head flame only. Brow gold and the belly mark share these colours and stay put."""
    col = is_cols(img, FLAME_COLS)
    col[21:, :] = False
    groups = component_list(col, diag=True)
    keep = []
    for g in groups:
        if min(p[0] for p in g) <= 16:
            keep.extend(g)
    if len(keep) < 80:
        raise SystemExit("head flame is missing")
    mask = mask_from(keep, op.shape)
    ys, xs = np.nonzero(mask)
    if int(ys.max()) > 20 or int(ys.min()) > 8:
        raise SystemExit(f"flame span changed y{int(ys.min())}-{int(ys.max())}")
    return mask


def hand_masks(img, op):
    brown = is_cols(img, BROWN_COLS)
    groups = component_list(brown, diag=True)
    if len(groups) != 2:
        raise SystemExit(f"expected two mittens, found {len(groups)}")
    hands = [mask_from(g, op.shape) for g in groups]
    hands.sort(key=lambda m: int(m.sum()), reverse=True)
    ink = is_color(img, INK) & op
    big = hands[0] | ink_outline(hands[0], op, ink)
    small = hands[1] | ink_outline(hands[1], op, ink)
    # A shared outline pixel would be claimed twice. Give it to the bigger mitten.
    small &= ~big
    return big, small


def segment(img, kind):
    op = img[..., 3] >= 128
    flame = flame_mask(img, op)
    ink = is_color(img, INK) & op
    flame = flame | ink_outline(flame, op, ink)
    hand_big, hand_small = hand_masks(img, op)
    hand_big &= ~flame
    hand_small &= ~flame & ~hand_big
    masks = {
        "flame": flame,
        "hand_big": hand_big,
        "hand_small": hand_small,
        "torso": op & ~flame & ~hand_big & ~hand_small,
    }
    check_partition(op, masks)
    require_solid(masks)
    if kind == "front":
        eye = is_color(img, PALE) & (np.arange(img.shape[0])[:, None] == 35)
        eye = eye[:, 44:52]
        if int(eye.sum()) != 8 or int((is_color(img, PALE) & masks["torso"] & (np.arange(img.shape[0])[:, None] == 35)).sum()) < 8:
            raise SystemExit("front eye slit left the face")
        teeth = is_color(img, WHITE)
        if int(teeth.sum()) != 89 or (teeth & ~masks["torso"]).any():
            raise SystemExit("front teeth left the mouth")
        mouth = teeth | (is_color(img, PINK) & (np.arange(img.shape[0])[:, None] >= 37))
        my, mx = np.nonzero(mouth)
        eye_x = 47.5
        if float(mx.min()) >= eye_x or float(mx.mean()) >= eye_x:
            raise SystemExit(f"front does not face left: mouth {float(mx.mean()):.2f} eye {eye_x}")
        ys, xs = np.nonzero(flame)
        if int(xs.max()) > 50:
            raise SystemExit("flame reached the far brow")
    else:
        if is_color(img, WHITE).any() or is_color(img, PINK).any():
            raise SystemExit("back grew a mouth")
        ys, xs = np.nonzero(masks["flame"])
        if float(xs.mean()) < 48:
            raise SystemExit("back flame is not on the right")
    return masks


def require_solid(masks):
    torso = masks["torso"]
    for name in LIMBS:
        part = masks[name]
        if int(part.sum()) < 20:
            raise SystemExit(f"{name} is missing from the still")
        if not (part & dilate(torso, 1)).any() and not (part & dilate(torso, 2)).any():
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
        raise SystemExit(f"torso fell apart ({len(groups)} pieces, next {len(groups[1])})")


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
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in LIMBS:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    torso_m = placed["torso"]
    for name, depth in (("flame", 4), ("hand_small", 5), ("hand_big", 5)):
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth
        )
    teeth = is_color(canvas, WHITE)
    has_mouth = bool(teeth.any())
    if has_mouth:
        my, mx = np.nonzero(teeth)
        pivots["mouth"] = (float(mx.mean()), float(my.mean()))
    else:
        pivots["mouth"] = None
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "has_mouth": has_mouth,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        flame=0.0, hand_big=0.0, hand_small=0.0,
        flame_mode="up", hand_big_mode="up", hand_small_mode="up",
        lean=0.0, knock=0, flash=False,
        wedge=0, ring=-1, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet are not drawn. The belly stays planted. The flame and mittens breathe.
        p["flame"] = wave(f, n, 6.0, phase=0.4)
        p["hand_big"] = wave(f, n, 3.0, phase=0.2)
        p["hand_small"] = wave(f, n, 3.0, phase=0.2 + math.pi)
    elif action == "walk":
        # No legs to step with. The two mittens swing, and the flame bobs.
        p["hand_big"] = wave(f, n, 11.0)
        p["hand_small"] = wave(f, n, 11.0, phase=math.pi)
        p["flame"] = wave(f, n, 5.0, phase=1.0)
        p["hand_big_mode"] = "fwd"
        p["hand_small_mode"] = "fwd"
    elif action == "attack":
        # 0 rest. 1-4 the open mouth roars and the flame lifts.
        # 5-8 the other side, equal scale. 9-12 both mittens strike.
        # 13-15 recover. 15 is the still. The head does not slide.
        big = [0, 4, 8, 10, 10, 6, 6, 8, 8, 18, 26, 34, 34, 12, 4, 0]
        small = [0, 3, 6, 8, 8, 4, 4, 6, 6, 12, 18, 24, 22, 8, 3, 0]
        flame = [0, 6, 10, 14, 14, 4, 4, 4, 4, 6, 8, 10, 8, 4, 2, 0]
        p["hand_big"] = float(big[f])
        p["hand_small"] = float(small[f])
        p["flame"] = float(flame[f])
        p["hand_big_mode"] = "fwd" if 9 <= f <= 12 else "up"
        p["hand_small_mode"] = "fwd" if 9 <= f <= 12 else "up"
        p["turned"] = 5 <= f <= 8
        if 1 <= f <= 4:
            p["wedge"] = [0, 1, 2, 3, 3][f]
            p["ring"] = f - 1
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.40, 0.72, 1.0, 0.84, 0.60, 0.38, 0.20, 0.08, 0.02, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["flame"] = 8.0 * env
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


def limb_motion(rig, p):
    ang = {}
    delta = {}
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    for name in LIMBS:
        ang[name] = part_angle(rig, name, p[f"{name}_mode"], p[name])
        delta[name] = (bdx, bdy)
    return ang, delta


def compose(rig, p):
    L = rig["layers"]
    ang, delta = limb_motion(rig, p)
    out = np.zeros_like(rig["canvas"])
    bdx = int(round(rig["facing"] * p["bdx"]))
    bdy = int(round(p["bdy"]))
    paste(out, shift(L["torso"], bdx, bdy))
    for name in ("flame", "hand_small", "hand_big"):
        dx, dy = delta[name]
        paste(out, moved(L[name], ang[name], rig["pivots"][name], dx, dy))
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


def mouth_exit(img, origin, fc):
    """First empty pixel in front of the mouth. The mouth already exists."""
    x, y = float(origin[0]), float(origin[1])
    h, w = img.shape[:2]
    for _ in range(24):
        nx = int(round(x + fc))
        ny = int(round(y))
        if not (0 <= nx < w and 0 <= ny < h):
            break
        if img[ny, nx, 3] < 128:
            return float(nx), float(y)
        x = nx
    return float(x), float(y)


def mouth_wedge(img, origin, fc, age):
    """A few pixels in front of the existing mouth. Empty pixels only."""
    ox, oy = origin
    cols = (PINK, RED, DRED)
    for i in range(max(0, int(age))):
        x = ox + fc * (1 + i)
        span = 1 if i == 0 else 2
        for dy in range(-span, span + 1):
            put(img, x, oy + dy, cols[min(i, 2)])


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (YELLOW, WHITE, GOLD, ORANGE)
    rad = 8 + age * 3
    for deg in range(-60, 61, 8):
        a = math.radians(deg)
        x = ox + fc * rad * math.cos(a)
        y = oy + rad * math.sin(a) * 0.62
        put(img, x, y, cols[age % 4])


def spark(img, origin, fc, age):
    ox, oy = origin
    cols = (GOLD, YELLOW, ORANGE, WHITE)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r), (2, r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def weapon_tip(rig, p):
    name = "hand_big"
    fc = rig["facing"]
    ang, delta = limb_motion(rig, p)
    dx, dy = delta[name]
    layer = moved(rig["layers"][name], ang[name], rig["pivots"][name], dx, dy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def limb_lost(rig, p):
    body = compose(rig, p)
    ang, delta = limb_motion(rig, p)
    lost = []
    for name in LIMBS:
        dx, dy = delta[name]
        layer = moved(rig["layers"][name], ang[name], rig["pivots"][name], dx, dy)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    other = "back" if kind == "front" else "front"
    return rigs[(other, not mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    moved_part = (
        abs(p["flame"]) >= 0.45 or abs(p["hand_big"]) >= 0.45 or abs(p["hand_small"]) >= 0.45
        or p["bdx"] or p["bdy"] or p["knock"] or abs(p["lean"]) > 0.4
    )
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        # Positive rotation moves the top right. Backward is opposite the facing.
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], piv)
    if moved_part or abs(p["lean"]) > 0.4:
        body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    else:
        img = rig["canvas"].copy()
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if rig["has_mouth"] and not p["turned"] and (p["wedge"] or p["ring"] >= 0):
        teeth = is_color(img, WHITE)
        if teeth.any():
            my, mx = np.nonzero(teeth)
            origin = mouth_exit(img, (float(mx.mean()), float(my.mean())), fc)
            if p["wedge"]:
                mouth_wedge(img, origin, fc, p["wedge"])
            if p["ring"] >= 0:
                roar_rings(img, (origin[0] + fc * 2, origin[1]), fc, p["ring"])
            img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = weapon_tip(rig, p)
        spark(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    groups = component_list(op)
    return len(groups[0]) / float(op.sum())


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
    fm = segment(front, "front")
    bm = segment(back, "back")
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
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "flame": (255, 140, 40, 255),
        "hand_big": (40, 180, 255, 255),
        "hand_small": (180, 70, 220, 255),
        "torso": (170, 170, 170, 255),
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
        "dex": 555, "form": "regional", "variant": "darmanitan-galar-zen", "gen": "gen5",
        "dir": "monsterSprite/gen5/0555_darmanitan-standard/regional/darmanitan-galar-zen",
        "sheetDir": "agent-sprite-forge",
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


def top_mean_x(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    limit = np.percentile(ys, 20)
    take = ys <= limit
    return float(xs[take].mean())


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
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, "mouth", {d: rigs[d]["has_mouth"] for d in DIRS}, flush=True)

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

    head_dx = top_mean_x(synth(rigs, "W", "hurt", 3, palette, effects=False)) - top_mean_x(rigs["W"]["canvas"])
    print("hurt head dx", round(head_dx, 2), flush=True)
    if head_dx < 2.0:
        raise SystemExit("hurt lean did not throw the head backward")

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
            atk.append((f, round(p["hand_big"], 1), round(p["hand_small"], 1), round(p["flame"], 1),
                        p["hand_big_mode"], p["turned"], p["spin"], p["wedge"], p["ring"]))
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
        "generator": "forge_gen5_darmanitan-galar-zen_actions", "version": VERSION, "key": KEY,
        "dex": 555, "variant": "darmanitan-galar-zen", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "round zen snowman (head flame, open mouth, two mittens, round belly, no separate feet)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (mouth centroid x 45.39 and mouth min x 34 are left of eye slit x 47.5); back flame is on the right and has no mouth",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
