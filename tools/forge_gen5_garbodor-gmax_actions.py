#!/usr/bin/env python3
"""Gigantamax Garbodor (0569) 8-direction actions from this form's own stills.

Measured on gigantamax/garbodor-gmax/front.png (96x96, bbox x0-95 y4-91,
5548 px, 15 colours, alpha 0/255) and back.png (bbox x0-95 y0-94, 5798 px,
12 colours, all already in the front palette). Lowest pixel: front y91,
back y94. The back is shifted up 3px so both ground lines share y91.

Facing: the two pink eyes (239, 143, 255), 46 px, sit at y19-25.
Mean eye x is 53.7 (left blob x43-47, right blob x59-63). The red mouth
(184, 41, 45) and (235, 73, 78) in the face, 35 px at x44-53 y56-64, has
mean x 48.4. The mouth is left of the eyes, so the still faces screen-left.
Pink and that red exist only on the front. Gold pixels are trash highlights
and the small bags, not the eyes.

Front parts, screen axes as drawn:
  spout  the brown cap on the crown, x42-53 y4-15.
  horn_r the brown spike on the upper right, x65-77 y7-19.
  arm_l  the left garbage claw, including the cyan and gold scrap on it.
  arm_r  the right garbage claw, including the red label at its tip.
  The pink eyes, the red mouth, the white teeth, and the grounded trash
  pile stay on the torso. There are no separate feet.
Back, no eyes and no mouth:
  spout on the crown. arm_l is the small upper-left scrap.
  arm_r is the long right claw, including the gold and lime bag.
  tail is the trash drag at the lower left, y83-94.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn. NW back mirrored.
  Attack 5-8 shows that direction's back at equal scale. No |cos| squash,
  no in-plane body spin.
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
SRC_DIR = ROOT / "monsterSprite/gen5/0569_garbodor/gigantamax/garbodor-gmax"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0569_garbodor/gigantamax/garbodor-gmax/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0569_garbodor-gmax"
VERSION = "garbodor-gmax_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 288, 320
OFF_X = (CELL_W - 96) // 2
FRONT_FOOT = 91
BACK_FOOT = 94
OFF_Y = 110
FOOT_Y = OFF_Y + FRONT_FOOT
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Roar ceilings. The claw strike is drawn at 34 and 28, above these,
# so finish() does not shrink the hit.
PEAKS = {
    "spout": 12.0, "horn_r": 10.0, "arm_l": 14.0, "arm_r": 12.0, "tail": 8.0,
}
MOVING = ("spout", "horn_r", "arm_l", "arm_r", "tail")

INK = (1, 1, 1)
DGREEN = (24, 57, 41)
GREEN = (41, 106, 74)
DK = (90, 74, 65)
BROWN = (123, 106, 90)
TAN = (164, 139, 115)
GREY = (189, 189, 189)
WHITE = (253, 253, 253)
CYAN = (0, 172, 213)
LIME = (107, 220, 39)
SKIN = (254, 188, 164)
GOLD = (230, 218, 0)
PINK = (239, 143, 255)
RED = (184, 41, 45)
RED2 = (235, 73, 78)
GREEN_COLS = (GREEN, DGREEN)
RED_COLS = (RED, RED2)
BAG_COLS = (TAN, BROWN, DK, INK, GREY)

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
    out = seeds & allowed
    if not out.any():
        return out
    while True:
        nxt = dilate(out, 1) & allowed
        if np.array_equal(nxt, out):
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
    """4x nearest in-plane rotation. No width squash."""
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


def largest_component(mask):
    groups = component_list(mask)
    if not groups:
        return 0
    return max(len(g) for g in groups)


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def check_partition(op, masks):
    acc = np.zeros_like(op)
    for name, m in masks.items():
        if (m & acc).any():
            raise SystemExit(f"overlap in {name}")
        if (m & ~op).any():
            raise SystemExit(f"{name} left the still")
        acc |= m
    if int(acc.sum()) != int(op.sum()):
        raise SystemExit(f"parts do not partition: {int(acc.sum())} vs {int(op.sum())}")


def absorb_islands(torso, limbs):
    """Torso islands join the limb they touch. The main body stays."""
    groups = component_list(torso)
    if len(groups) <= 1:
        return torso
    groups.sort(key=len, reverse=True)
    keep = np.zeros_like(torso)
    for y, x in groups[0]:
        keep[y, x] = True
    for g in groups[1:]:
        sm = np.zeros_like(torso)
        for y, x in g:
            sm[y, x] = True
        best, score = None, 0
        for name, m in limbs.items():
            s = int((sm & dilate(m, 2)).sum())
            if s > score:
                best, score = name, s
        if best is None:
            raise SystemExit(f"torso island touches nothing ({len(g)} px)")
        limbs[best] |= sm
        keep &= ~sm
    return keep


def segment(img, kind):
    """Claws, crown scraps, and the back drag. The face and the pile stay."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    green = any_color(img, GREEN_COLS)
    pink = is_color(img, PINK)
    red = any_color(img, RED_COLS)
    skin = is_color(img, SKIN)
    white = is_color(img, WHITE)
    if kind == "front":
        spout = flood(
            op & ~green & ~pink & (xx >= 40) & (xx <= 60) & (yy <= 16),
            op & (xx >= 44) & (xx <= 52) & (yy <= 8),
        )
        horn = flood(
            op & ~green & ~pink & ~spout & (xx >= 62) & (xx <= 82) & (yy <= 20),
            op & (xx >= 68) & (xx <= 76) & (yy <= 12),
        )
        chest_white = white & (xx >= 36) & (xx <= 52) & (yy >= 32) & (yy <= 46)
        mouth_block = (red | skin | white) & (yy >= 52) & (yy <= 74) & (xx >= 20) & (xx <= 70)
        allow_l = op & ~green & ~pink & ~spout & ~horn & ~chest_white & ~mouth_block
        allow_l &= (xx <= 46) & (yy <= 70) & ~((yy >= 52) & (xx >= 28))
        arm_l = flood(allow_l, allow_l & (xx <= 10) & (yy >= 8) & (yy <= 48))
        allow_r = op & ~green & ~pink & ~spout & ~horn & ~mouth_block & ~chest_white
        allow_r &= (xx >= 70) & (yy >= 20) & (yy <= 80)
        arm_r = flood(allow_r, allow_r & (xx >= 90))
        limbs = {"spout": spout, "horn_r": horn, "arm_l": arm_l, "arm_r": arm_r}
    else:
        if pink.any() or red.any():
            raise SystemExit("back grew a face")
        colorful = any_color(img, (GOLD, LIME, CYAN, SKIN, WHITE))
        spout = flood(
            op & ~green & (xx >= 46) & (xx <= 66) & (yy <= 10),
            op & (xx >= 52) & (xx <= 60) & (yy <= 4),
        )
        outer = op & ~green & (xx >= 76) & (yy <= 78)
        allow_r = op & ~green & ~spout & (yy <= 78) & ((xx >= 74) | (colorful & (xx >= 62)))
        arm_r = flood(allow_r, outer & (xx >= 88) & ~spout)
        allow_l = op & ~green & ~spout & ~arm_r & (xx <= 42) & (yy <= 16)
        arm_l = flood(allow_l, allow_l & (xx >= 26) & (xx <= 42))
        tail = flood(
            op & ~green & ~arm_l & ~arm_r & (yy >= 83) & (xx <= 46),
            op & (yy >= 89) & (xx <= 18),
        )
        limbs = {"spout": spout, "arm_l": arm_l, "arm_r": arm_r, "tail": tail}
    torso = absorb_islands(op & ~np.logical_or.reduce(list(limbs.values())), limbs)
    masks = dict(limbs)
    masks["torso"] = torso
    check_partition(op, masks)
    if len(component_list(torso)) != 1:
        raise SystemExit(f"{kind} torso fell apart")
    for name, part in limbs.items():
        if int(part.sum()) < 80:
            raise SystemExit(f"{kind} {name} is too small ({int(part.sum())})")
        if not (part & dilate(torso, 1)).any():
            raise SystemExit(f"{kind} {name} does not touch the body")
    foot = int(np.nonzero(op)[0].max())
    want = FRONT_FOOT if kind == "front" else BACK_FOOT
    if foot != want:
        raise SystemExit(f"{kind} sole moved: {foot}")
    if kind == "front":
        if int((pink & torso).sum()) != 46:
            raise SystemExit("front eyes left the head")
        mouth = red & (xx >= 40) & (xx <= 58) & (yy >= 54) & (yy <= 66)
        if int((mouth & torso).sum()) != int(mouth.sum()) or int(mouth.sum()) < 30:
            raise SystemExit("front mouth left the face")
        eye = float(np.nonzero(pink)[1].mean())
        mouth_x = float(np.nonzero(mouth)[1].mean())
        if not mouth_x < eye - 2.0:
            raise SystemExit(f"front does not face left: mouth {mouth_x:.1f} eye {eye:.1f}")
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
    for name in MOVING:
        if name not in placed:
            continue
        depth = 6 if name in ("arm_l", "arm_r", "tail") else 4
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth + 1,
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
        strike, off = "arm_l", "arm_r"
    else:
        facing = -1.0 if mirror else 1.0
        strike, off = "arm_r", "arm_l"
    pivots = {}
    signs = {}
    for name in MOVING:
        if name not in layers:
            continue
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    mouth = any_color(layers["torso"], RED_COLS)
    if mouth.any():
        my, mx = np.nonzero(mouth)
        pivots["mouth"] = (float(mx.mean()), float(my.mean()))
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing,
        "strike": strike, "off": off,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        spout=0.0, horn_r=0.0, arm_l=0.0, arm_r=0.0, tail=0.0,
        mode="up",
        roar=0, spark=0, lean=0.0, knock=0, flash=False,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # The pile stays on the ground. Claws and the crown scraps breathe.
        p["spout"] = wave(f, n, 5.0, phase=0.3)
        p["horn_r"] = wave(f, n, 4.0, phase=0.9)
        p["arm_l"] = wave(f, n, 3.0, phase=0.2)
        p["arm_r"] = wave(f, n, 3.0, phase=1.0)
        p["tail"] = wave(f, n, 6.0, phase=0.4)
        p["mode"] = "up"
    elif action == "walk":
        # No feet are drawn. The pile waddles one pixel and the claws swing.
        u = 2 * math.pi * f / n
        p["arm_l"] = 10.0 * math.sin(u)
        p["arm_r"] = 10.0 * math.sin(u + math.pi)
        p["spout"] = wave(f, n, 3.0, phase=0.2)
        p["horn_r"] = wave(f, n, 3.0, phase=0.8)
        p["tail"] = wave(f, n, 8.0, phase=0.5)
        p["bdx"] = int(round(wave(f, n, 1.2)))
        p["mode"] = "fwd"
    elif action == "attack":
        # 0 rest. 1-4 claws and scraps rise, and the front mouth roars.
        # 5-8 the back, equal scale. 9-12 the claws strike. 13-15 return.
        spout = [0, 4, 8, 12, 12, 0, 0, 0, 0, 6, 10, 12, 10, 4, 1, 0]
        horn = [0, 3, 6, 10, 10, 0, 0, 0, 0, 4, 8, 10, 8, 3, 1, 0]
        arm_l = [0, 6, 10, 14, 14, 0, 0, 0, 0, 18, 28, 34, 34, 12, 4, 0]
        arm_r = [0, 4, 8, 12, 12, 0, 0, 0, 0, 14, 22, 28, 26, 10, 3, 0]
        tail = [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 8, 6, 2, 0, 0]
        p["spout"] = float(spout[f])
        p["horn_r"] = float(horn[f])
        p["arm_l"] = float(arm_l[f])
        p["arm_r"] = float(arm_r[f])
        p["tail"] = float(tail[f])
        p["mode"] = "up" if f <= 4 else "fwd"
        p["turned"] = 5 <= f <= 8
        if 1 <= f <= 4:
            p["roar"] = f
        if f in (11, 12):
            p["spark"] = f - 10
        if f == 10:
            p["bdx"] = 1
        elif f in (11, 12):
            p["bdx"] = 2
    else:
        env = [0.0, 0.45, 0.80, 1.0, 0.85, 0.62, 0.42, 0.24, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        for key in MOVING:
            p[key] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["roar"] = 0
        p["spark"] = 0
        p["spin"] = 0
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


def limb_motion(rig, p):
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    ang = {}
    delta = {}
    for name in MOVING:
        if name not in rig["layers"]:
            continue
        ang[name] = part_angle(rig, name, p["mode"], p[name])
        delta[name] = (bdx, bdy)
    return ang, delta, bdx, bdy


def compose(rig, p):
    L = rig["layers"]
    piv = rig["pivots"]
    ang, delta, bdx, bdy = limb_motion(rig, p)
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    order = []
    if "tail" in L:
        order.append("tail")
    if rig["off"] in L:
        order.append(rig["off"])
    for name in ("spout", "horn_r"):
        if name in L:
            order.append(name)
    if rig["strike"] in L:
        order.append(rig["strike"])
    seen = set()
    for name in order:
        if name in seen or name not in L:
            continue
        seen.add(name)
        dx, dy = delta[name]
        paste(out, moved(L[name], ang[name], piv[name], dx, dy))
    return out


def roar(img, origin, fc, age):
    """Mouth wedge and rings. Age is 1..4. Only empty pixels, palette colours."""
    ox, oy = origin
    cols = (RED, WHITE, INK, RED2)
    span = 3 + age
    for i in range(span):
        x = ox + fc * (5 + i)
        put(img, x, oy, cols[i % 2])
        put(img, x, oy - 1 - (i % 2), WHITE if i % 2 == 0 else RED)
        put(img, x, oy + 1 + (i % 2), RED2 if i % 2 == 0 else INK)
    r = 7 + age * 2
    ring = ((r, 0), (r - 1, -3), (r - 1, 3), (r // 2, -4), (r // 2, 4), (3, -2))
    for i, (dx, dy) in enumerate(ring):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def slam(img, origin, fc, age):
    ox, oy = origin
    cols = (LIME, GOLD, WHITE, CYAN, INK)
    for i, (dx, dy) in enumerate(((4, -2), (6, 0), (5, 2), (3, -4), (7, 1), (2, 3))):
        put(img, ox + fc * (dx + age), oy + dy, cols[i % 5])


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


def weapon_tip(rig, p):
    name = rig["strike"]
    fc = rig["facing"]
    ang, delta, _bdx, _bdy = limb_motion(rig, p)
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
    ang, delta, _bdx, _bdy = limb_motion(rig, p)
    lost = []
    for name in MOVING:
        if name not in rig["layers"]:
            continue
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
    return rigs[(other, mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    moved_part = any(abs(p[k]) >= 0.45 for k in MOVING) or p["bdx"] or p["bdy"]
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, fc * p["lean"], piv)
    if moved_part or abs(p["lean"]) > 0.4:
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["roar"] and rig["kind"] == "front" and "mouth" in rig["pivots"]:
        mx, my = rig["pivots"]["mouth"]
        roar(img, (mx + fc * p["bdx"], my), fc, p["roar"])
        img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = weapon_tip(rig, p)
        slam(img, (tx, ty), fc, p["spark"])
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


def tile_grid(tiles, cols, bg=(18, 22, 32, 255)):
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    out = np.full((rows * th, cols * tw, 4), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        m = t[..., 3] >= 128
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
    return out


def preview_image(sheets):
    pick = {"idle": 4, "walk": 4, "attack": 11, "hurt": 3}
    tiles = []
    for act in ACTIONS:
        for di in range(8):
            f = pick[act]
            tiles.append(sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W])
    return Image.fromarray(tile_grid(tiles, 8), "RGBA")


def content_box(canvases, pad=8):
    acc = np.zeros(canvases[0].shape[:2], bool)
    for c in canvases:
        acc |= c[..., 3] >= 128
    ys, xs = np.nonzero(acc)
    y0 = max(0, int(ys.min()) - pad - 28)
    y1 = min(CELL_H, int(ys.max()) + pad + 18)
    x0 = max(0, int(xs.min()) - pad - 36)
    x1 = min(CELL_W, int(xs.max()) + pad + 36)
    return y0, y1, x0, x1


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


def frame_metrics(img, palette):
    """Off-palette colours, partial alpha, and border contact for one cell."""
    pal = {tuple(int(v) for v in c) for c in palette}
    op = img[..., 3] == 255
    partial = int(((img[..., 3] != 0) & (img[..., 3] != 255)).sum())
    off = 0
    if op.any():
        used = {tuple(int(v) for v in c) for c in np.unique(img[op][:, :3], axis=0)}
        off = len(used - pal)
    a = img[..., 3] == 255
    edge = int(a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any())
    return off, partial, edge


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
    fm = segment(front, "front")
    bm = segment(back, "back")
    rigs = {
        ("front", False): build_view(front, fm, "front", False),
        ("front", True): build_view(front, fm, "front", True),
        ("back", False): build_view(back, bm, "back", False),
        ("back", True): build_view(back, bm, "back", True),
    }
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    fa = rigs[("front", False)]["canvas"]
    ba = rigs[("back", False)]["canvas"]
    if int(np.nonzero(fa[..., 3] >= 128)[0].max()) != int(np.nonzero(ba[..., 3] >= 128)[0].max()):
        raise SystemExit("front and back feet do not share a line")
    if rigs[("front", False)]["facing"] != -1.0 or rigs[("back", False)]["facing"] != 1.0:
        raise SystemExit("facing locked to the wrong side")
    if "mouth" not in rigs[("front", False)]["pivots"]:
        raise SystemExit("front mouth anchor missing")
    if "mouth" in rigs[("back", False)]["pivots"]:
        raise SystemExit("back invented a mouth")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "torso": (70, 130, 210, 255),
        "spout": (40, 190, 70, 255),
        "horn_r": (220, 50, 50, 255),
        "arm_l": (70, 220, 220, 255),
        "arm_r": (190, 80, 220, 255),
        "tail": (255, 210, 40, 255),
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
        "dex": 569, "form": "gigantamax", "variant": "garbodor-gmax", "gen": "gen5",
        "dir": "monsterSprite/gen5/0569_garbodor/gigantamax/garbodor-gmax",
        "sheetDir": "agent-sprite-forge",
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
    palette = locked_palette(SRC_DIR / "front.png", SRC_DIR / "back.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    print("strike", {d: rigs[d]["strike"] for d in ("S", "W", "N", "E")}, flush=True)

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
    rest_sum = int(sum(rest_diff.values()))
    print("rest diff", rest_diff, "sum", rest_sum, flush=True)
    if rest_sum:
        raise SystemExit("frame 0 does not match the still")

    turn_w = synth(rigs, "W", "attack", 6, palette, effects=False)
    turn_s = synth(rigs, "S", "attack", 6, palette, effects=False)
    if np.any(turn_w != rigs["N"]["canvas"]):
        raise SystemExit("W turn is not the N back at equal scale")
    if np.any(turn_s != rigs["NW"]["canvas"]):
        raise SystemExit("S turn is not N flipped at equal scale")

    if args.samples:
        lost = {}
        off = partial = edge = 0
        bodies = []
        picks = {
            "idle": (0, 8),
            "walk": (0, 4, 12),
            "attack": (0, 2, 4, 6, 8, 11, 12, 15),
            "hurt": (0, 3, 6, 11),
        }
        for d in DIRS:
            for act, fs in picks.items():
                for f in fs:
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost[f"{act}:{d}:{f}"] = bad
                    t = synth(rigs, d, act, f, palette)
                    o, pa, ed = frame_metrics(t, palette)
                    off += o
                    partial += pa
                    edge += ed
                    if act in ("walk", "attack", "hurt") and f != 0:
                        bodies.append(largest_fraction(t))
        leans = [round(pose("hurt", f)["lean"], 2) for f in range(12)]
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["arm_l"], 1), round(p["arm_r"], 1),
                        round(p["spout"], 1), int(p["turned"]), int(p["spin"]), int(p["roar"])))
        end = synth(rigs, "W", "attack", 15, palette, effects=False)
        end_diff = int(np.any(end != rigs["W"]["canvas"], axis=2).sum())
        print("limb lost", lost or "none", flush=True)
        print("hurt lean", leans, "max", max(leans), flush=True)
        print("attack poses", atk, flush=True)
        print("attack15 diff", end_diff, flush=True)
        print(
            "SAMPLE rest_diff", rest_sum,
            "off_palette", off,
            "partial_alpha", partial,
            "edge", edge,
            "body_min", round(min(bodies), 3) if bodies else 1,
            flush=True,
        )
        box = content_box([rigs[d]["canvas"] for d in DIRS])
        y0, y1, x0, x1 = box
        tiles = []
        for d in ("S", "W", "N", "E"):
            for act, f in (
                ("idle", 0), ("idle", 8),
                ("walk", 4), ("walk", 12),
                ("attack", 0), ("attack", 3), ("attack", 6), ("attack", 8),
                ("attack", 11), ("attack", 14),
                ("hurt", 1), ("hurt", 3),
            ):
                t = synth(rigs, d, act, f, palette)
                tiles.append(t[y0:y1, x0:x1])
        sheet = np.repeat(np.repeat(tile_grid(tiles, 12), 2, 0), 2, 1)
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
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest and not args.force:
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
        "generator": "forge_gen5_garbodor-gmax_actions", "version": VERSION, "key": KEY,
        "dex": 569, "variant": "garbodor-gmax", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "grounded trash pile (two garbage claws, crown scraps, a mouth, no feet; back has a drag tail)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (mouth mean x48.4 < pink eye mean x53.7)",
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
