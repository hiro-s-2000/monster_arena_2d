#!/usr/bin/env python3
"""Mega Emboar (0500) 8-direction actions from this form's own stills.

Measured on mega/emboar-mega/front.png (96x96, bbox x5-91 y0-95, 4825 px,
15 colours, alpha 0/255, one connected piece):
  Nostrils (115,41,0) sit at x38 and x44, y30, mean x 41.0.
  Eyes are the white pair at x30-31 y31-36 and x48-50 y31-37, mean x 41.26.
  The nose is left of the eyes, so the still faces screen-left.
  The snout is orange (238,98,32) and rust around x32-51 y30-45, with a dark
  mouth at x34-50 y37-44 and two white tusks. The roar drops that jaw.
  It does not invent a second mouth.
  Screen-left flame wing x5-29 y12-30. Screen-right crest x46-80 y0-31.
  Hanging flame beard on the screen-right, x68-91 y66-95 (the rear side).
  Screen-left fist x7-23 y45-76. Screen-right fist x63-84 y42-64.
  Two mane tufts above the mask. Two hooves: x17-30 and x53-64, y90-95.
  Foot line y95. The beard tip also touches y95; the lean pivots on a hoof.
mega/emboar-mega/back.png (bbox x9-86 y3-93, 4404 px, 14 colours, foot y93)
  faces away with the flame beard on the screen-left, so the snout side is
  screen-right. Back is shifted down 2px so both feet share one line.
  Same parts: two mane tufts, left flame and right crest, beard, two arms,
  two hooves. No tusks and no mouth are drawn on the back.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (beard screen-left). NW back mirrored.
  E is screen-right, W is screen-left, N is the back. Attack 5-8 swaps to
  the matching back at equal scale. No width squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_emboar_mega_actions.py
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
import mouth_open

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen5/0500_emboar/mega/emboar-mega"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0500_emboar/mega/emboar-mega/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0500_emboar-mega"
VERSION = "emboar-mega_v1m"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 272, 272
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 36
FRONT_FOOT = 95
BACK_FOOT = 93
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left (nose x < eye x). The beard is the rear,
# on the screen-right. Back as drawn has that beard on the screen-left.
# S (front mirrored) turns into N (back as drawn) by flipping the mirror flag.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Roar / turn hold. The fist strike is drawn up to 34 deg, above these.
PEAKS = {
    "mane": 12.0, "flame": 10.0, "crest": 12.0, "beard": 8.0,
    "arm": 14.0, "arm_low": 10.0,
}
LIMBS = (
    "mane_l", "mane_r", "flame_l", "crest", "beard",
    "arm_l", "arm_r", "jaw", "leg_l", "leg_r",
)

INK = (16, 16, 16)
# Mouth fix (2026-11-04): the jaw no longer drops and no dark wedge is painted.
# A small opening is carved under the closed mouth line of the front still
# (the black line under the pig snout), hinge/cheek -> lip tip, in still
# coordinates. Back views have no mouth. Roar frames 1-4 open 1-2 px.
MOUTH_LINE_FRONT = [(43, 42), (41, 43), (36, 43)]
MOUTH_OPEN = [0, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
GREY_D = (32, 32, 32)
GREY = (49, 49, 49)
NOSE_C = (115, 41, 0)
DK = (123, 24, 16)
BROWN = (123, 90, 8)
MID = (164, 57, 49)
SILVER = (172, 172, 172)
GOLD = (180, 139, 32)
RUST = (189, 82, 32)
RED = (230, 49, 41)
DEEP = (238, 98, 32)
ORANGE = (238, 139, 8)
YELLOW = (238, 189, 8)
WHITE = (255, 255, 255)
PALE = (246, 246, 246)
DARKC = (INK, GREY_D, GREY)

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


def rot_point(x, y, angle_deg, pivot):
    """Same angle convention as rotate_nearest, in cell pixels."""
    r = math.radians(angle_deg)
    c, s = math.cos(r), math.sin(r)
    dx, dy = x - pivot[0], y - pivot[1]
    return pivot[0] + c * dx + s * dy, pivot[1] - s * dx + c * dy


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


def reconnect(img):
    """Close a 1px gap after a turn. Sparks are drawn later."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(3):
        groups = component_list(out[..., 3] >= 128)
        if len(groups) <= 1:
            return out
        groups.sort(key=len, reverse=True)
        main = np.zeros((h, w), bool)
        for y, x in groups[0]:
            main[y, x] = True
        changed = False
        for g in groups[1:]:
            if len(g) < 4:
                continue
            best = None
            for y, x in g:
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and main[ny, nx]:
                            dist = max(abs(dy), abs(dx))
                            if best is None or dist < best[0]:
                                best = (dist, y, x, ny, nx)
            if best is None or best[0] > 2:
                continue
            _, y, x, ny, nx = best
            if abs(ny - y) == 1 and abs(nx - x) == 1:
                candidates = ((y, nx), (ny, x))
            else:
                sy = 0 if ny == y else (1 if ny > y else -1)
                sx = 0 if nx == x else (1 if nx > x else -1)
                candidates = ((y + sy, x + sx),)
            for fy, fx in candidates:
                if 0 <= fy < h and 0 <= fx < w and out[fy, fx, 3] < 128:
                    out[fy, fx, :3] = out[y, x, :3]
                    out[fy, fx, 3] = 255
                    changed = True
                    break
        if not changed:
            return out
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


def comps_of(mask, eight=False):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    groups = []
    ys, xs = np.nonzero(mask)
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if eight:
        neigh = neigh + ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        groups.append(pts)
    groups.sort(key=len, reverse=True)
    return groups


def from_pts(pts, shape):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m


def largest8(mask):
    groups = comps_of(mask, eight=True)
    if not groups:
        return np.zeros_like(mask)
    return from_pts(groups[0], mask.shape)


def keep_big(mask, min_n=12):
    out = np.zeros_like(mask)
    for g in comps_of(mask, eight=False):
        if len(g) >= min_n:
            for y, x in g:
                out[y, x] = True
    return out


def touches(part, body):
    if not part.any() or not body.any():
        return False
    return bool((part & dilate(body, 1)).any())


def row_runs(op):
    rows = []
    for y in range(op.shape[0]):
        xs = np.nonzero(op[y])[0]
        segs = []
        if len(xs):
            s = p = int(xs[0])
            for x in xs[1:]:
                x = int(x)
                if x == p + 1:
                    p = x
                else:
                    segs.append((s, p))
                    s = p = x
            segs.append((s, p))
        rows.append(segs)
    return rows


def take_runs(op, pred):
    m = np.zeros_like(op)
    for y, segs in enumerate(row_runs(op)):
        for a, b in segs:
            if pred(y, a, b):
                m[y, a:b + 1] = True
    return m & op


def fill_gaps(arm, op, gap=8):
    """Pull opaque pixels that sit in a short gap between this part's own pixels."""
    out = arm.copy()
    h, w = arm.shape
    for y in range(h):
        xs = np.nonzero(arm[y])[0]
        if len(xs) < 2:
            continue
        for x0, x1 in zip(xs[:-1], xs[1:]):
            if 1 < int(x1 - x0) <= gap:
                out[y, int(x0):int(x1) + 1] |= op[y, int(x0):int(x1) + 1]
    return out & op


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


def require_parts(masks):
    torso = masks["torso"]
    main = comps_of(torso)
    if not main or len(main[0]) < 0.85 * int(torso.sum()):
        raise SystemExit(f"torso fell apart {[len(g) for g in main[:4]]}")
    for name in LIMBS:
        m = masks[name]
        if not m.any():
            if name == "jaw":
                continue
            raise SystemExit(f"{name} is empty")
        held = touches(m, torso) or any(
            touches(m, masks[other]) for other in LIMBS
            if other != name and masks[other].any() and touches(masks[other], torso)
        )
        if not held:
            raise SystemExit(f"{name} does not touch the body")
        cs = comps_of(m, eight=True)
        if len(cs[0]) < 0.8 * int(m.sum()):
            raise SystemExit(f"{name} fell apart {[len(g) for g in cs[:4]]}")


def segment(img, kind):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    dark = any_color(img, DARKC)
    if kind == "front":
        metal = any_color(img, (WHITE, SILVER))
    else:
        metal = any_color(img, (WHITE, SILVER, PALE))
    base = dark | metal

    def arm(box):
        return fill_gaps(largest8(base & box), op)

    if kind == "front":
        arm_l = arm((xx <= 26) & (yy >= 43) & (yy <= 78))
        arm_r = arm((xx >= 60) & (yy >= 40) & (yy <= 68))
        mane_l = largest8(dark & (yy <= 27) & (xx >= 28) & (xx <= 41))
        mane_r = largest8(dark & (yy <= 27) & (xx >= 42) & (xx <= 56))
        # Lower snout, tusks and the dark mouth. Eyes stay on the head.
        snout = any_color(img, (DEEP, RUST, NOSE_C, WHITE, SILVER, INK, GREY_D, GREY, DK))
        jaw = fill_gaps(snout & (yy >= 38) & (yy <= 45) & (xx >= 30) & (xx <= 52), op, gap=6)
        jaw = keep_big(jaw, min_n=8)
        used = arm_l | arm_r | mane_l | mane_r | jaw
        flame_l = keep_big(take_runs(op, lambda y, a, b: y <= 31 and b <= 32) & ~used)
        crest = take_runs(op, lambda y, a, b: y <= 31 and a >= 46) & ~used
        steal = crest & dark & (xx <= 56)
        mane_r = mane_r | steal
        crest = keep_big(crest & ~steal)
        used = used | flame_l | crest | mane_r
        beard = keep_big(take_runs(op, lambda y, a, b: y >= 66 and a >= 68) & ~used, min_n=8)
        used = used | beard
        leg_l = keep_big(op & ~used & (yy >= 90) & (xx <= 31), min_n=8)
        leg_r = keep_big(op & ~used & (yy >= 90) & (xx >= 50) & (xx <= 68), min_n=8)
        nose = is_color(img, NOSE_C)
        eye = is_color(img, WHITE) & (yy >= 31) & (yy <= 37) & (xx >= 25) & (xx <= 55)
        if int(nose.sum()) < 2 or int(eye.sum()) < 8:
            raise SystemExit("front nose or eyes were not found")
        nose_x = float(np.nonzero(nose)[1].mean())
        eye_x = float(np.nonzero(eye)[1].mean())
        if nose_x >= eye_x:
            raise SystemExit(f"front does not face left: nose {nose_x:.2f} eye {eye_x:.2f}")
        mouth = (any_color(img, (INK, DK, GREY_D)) & (yy >= 38) & (yy <= 44) & (xx >= 32) & (xx <= 50))
    else:
        arm_l = arm((xx <= 36) & (yy >= 36) & (yy <= 56))
        arm_r = arm((xx >= 58) & (yy >= 40) & (yy <= 78))
        mane_l = largest8(dark & (yy <= 28) & (xx >= 34) & (xx <= 48))
        mane_r = largest8(dark & (yy <= 28) & (xx >= 49) & (xx <= 62))
        jaw = np.zeros_like(op)
        used = arm_l | arm_r | mane_l | mane_r
        flame_l = take_runs(op, lambda y, a, b: y <= 36 and b <= 42) & ~used
        crest = keep_big(take_runs(op, lambda y, a, b: y <= 30 and a >= 64) & ~used, min_n=8)
        used = used | flame_l | crest
        # The center flame tip sits above the mane and is not a side run.
        top = op & ~used & (yy <= 16) & (xx < 60)
        flame_l = keep_big(flame_l | top, min_n=8)
        used = used | flame_l
        beard = keep_big(take_runs(op, lambda y, a, b: 50 <= y <= 87 and b <= 28) & ~used, min_n=8)
        used = used | beard
        leg_l = keep_big(op & ~used & (yy >= 88) & (xx <= 48), min_n=8)
        leg_r = keep_big(op & ~used & (yy >= 86) & (xx >= 54) & (xx <= 74), min_n=8)
        mouth = np.zeros_like(op)
        if float(np.nonzero(beard)[1].mean()) >= float(np.nonzero(op & ~beard)[1].mean()):
            raise SystemExit("back beard is not on the left")

    masks = {
        "arm_l": arm_l, "arm_r": arm_r, "mane_l": mane_l, "mane_r": mane_r,
        "flame_l": flame_l, "crest": crest, "beard": beard, "jaw": jaw,
        "leg_l": leg_l, "leg_r": leg_r,
    }
    used = np.zeros_like(op)
    for name in LIMBS:
        masks[name] = masks[name] & op & ~used
        used |= masks[name]
    masks["torso"] = op & ~used
    # A torso speck trapped in a limb travels with that limb.
    islands = comps_of(masks["torso"])
    if len(islands) > 1:
        for g in islands[1:]:
            if len(g) > 80:
                continue
            speck = from_pts(g, op.shape)
            host = None
            for name in LIMBS:
                if touches(speck, masks[name]):
                    host = name
                    break
            if host is None:
                continue
            masks[host] = masks[host] | speck
            masks["torso"] = masks["torso"] & ~speck
    # The long flame hangs on the fist. A separate speck touches the hip
    # and is not connected to that flame, so it stays on the torso.
    holder = "arm_r" if kind == "front" else "arm_l"
    kept = np.zeros_like(op)
    for comp in comps_of(masks["beard"], eight=False):
        piece = from_pts(comp, op.shape)
        if touches(piece, masks[holder]):
            kept |= piece
        else:
            masks["torso"] = masks["torso"] | piece
    if not kept.any():
        raise SystemExit("beard does not meet its arm")
    masks["beard"] = kept
    check_partition(op, masks)
    require_parts(masks)
    if kind == "front":
        if int(np.nonzero(masks["leg_l"] | masks["leg_r"] | masks["beard"])[0].max()) != FRONT_FOOT:
            raise SystemExit("front foot line moved")
        masks["mouth"] = mouth & masks["torso"] | (mouth & masks["jaw"])
    else:
        if int(np.nonzero(masks["leg_l"] | masks["leg_r"])[0].max()) != BACK_FOOT:
            raise SystemExit("back foot line moved")
        masks["mouth"] = mouth
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
    if not part.any():
        return 0.0, 0.0
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
        return {"up": 1.0, "fwd": 1.0, "down": -1.0}

    def tip_after(ang):
        return tip_xy(rotate_nearest(layer, ang, pivot), pivot)

    x_pos, y_pos = tip_after(12.0)
    x_neg, y_neg = tip_after(-12.0)
    up = -1.0 if y_neg < y_pos else 1.0
    if facing > 0:
        fwd = 1.0 if x_pos >= x_neg else -1.0
    else:
        fwd = 1.0 if x_pos <= x_neg else -1.0
    return {"up": up, "fwd": fwd, "down": -up}


def build_view(img, masks, kind, mirror):
    y_shift = 0 if kind == "front" else (FRONT_FOOT - BACK_FOOT)
    canvas = place(img, y_shift)
    placed = {k: place(v, y_shift) for k, v in masks.items() if k != "mouth"}
    mouth = place(masks["mouth"].astype(np.uint8), y_shift).astype(bool)
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
        mouth = np.ascontiguousarray(mouth[:, ::-1])
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    depths = {
        "mane_l": 3, "mane_r": 3, "flame_l": 4, "crest": 4, "beard": 4,
        "arm_l": 4, "arm_r": 4, "jaw": 3, "leg_l": 4, "leg_r": 4,
    }
    for name, depth in depths.items():
        if not placed[name].any():
            continue
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in LIMBS:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    holder = "arm_r" if kind == "front" else "arm_l"
    bm = layers["beard"][..., 3] >= 128
    am = layers[holder][..., 3] >= 128
    hit = bm & dilate(am, 1)
    if hit.any():
        hy, hx = np.nonzero(hit)
        pivots["beard_contact"] = (float(hx.mean()), float(hy.mean()))
    else:
        pivots["beard_contact"] = pivots["beard"]
    signs["beard"] = signs_for(layers["beard"], pivots["beard_contact"], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    muzzle = None
    if mouth.any():
        my, mx = np.nonzero(mouth)
        if facing > 0:
            i = int(mx.argmax())
        else:
            i = int(mx.argmin())
        muzzle = (float(mx[i]), float(my[i]))
    mouth_line = None
    if kind == "front":
        mouth_line = mouth_open.transform_points(
            MOUTH_LINE_FRONT, dx=OFF_X, dy=OFF_Y + y_shift)
        if mirror:
            mouth_line = mouth_open.transform_points(mouth_line, mirror_w=CELL_W)
    return {
        "mouth_line": mouth_line,
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
        "has_mouth": muzzle is not None, "muzzle": muzzle,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        mane=0.0, flame=0.0, crest=0.0, beard=0.0,
        arm=0.0, arm_low=0.0, jaw=0.0,
        arm_mode="up", arm_low_mode="up", jaw_mode="down",
        step_l=(0, 0), step_r=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0, wedge=0, ring=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The mane and the three flames move.
        p["mane"] = wave(f, n, 5.0, phase=0.4)
        p["flame"] = wave(f, n, 7.0, phase=1.1)
        p["crest"] = wave(f, n, 8.0, phase=0.2)
        p["beard"] = wave(f, n, 5.0, phase=1.8)
    elif action == "walk":
        # Two hooves, opposite steps. One hoof stays down each frame.
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * sb)), -int(round(3 * max(0.0, sb))))
        p["arm_mode"] = "fwd"
        p["arm_low_mode"] = "fwd"
        p["arm"] = wave(f, n, 8.0, phase=0.4)
        p["arm_low"] = wave(f, n, 8.0, phase=0.4 + math.pi)
        p["mane"] = wave(f, n, 4.0)
        p["flame"] = wave(f, n, 8.0, phase=0.6)
        p["crest"] = wave(f, n, 9.0, phase=0.3)
        p["beard"] = wave(f, n, 6.0, cycles=2.0, phase=1.0)
    elif action == "attack":
        # 0 rest. 1-4 jaw drop, mane and flames up, roar wedge and rings.
        # 5-8 the matching back, equal scale. 9-12 the raised fist strikes.
        # 13-15 recover. 15 is the still. Body does not slide.
        mane = [0, 6, 10, 12, 12, 8, 8, 8, 6, 4, 3, 2, 2, 1, 0, 0]
        flame = [0, 4, 8, 10, 10, 6, 6, 6, 4, 3, 3, 2, 2, 1, 0, 0]
        crest = [0, 6, 10, 12, 12, 8, 8, 8, 6, 4, 4, 3, 2, 1, 0, 0]
        beard = [0, 3, 6, 8, 8, 6, 6, 6, 6, 8, 10, 10, 6, 2, 0, 0]
        arm = [0, 4, 8, 10, 10, 6, 6, 6, 8, 20, 28, 34, 34, 12, 4, 0]
        arm_low = [0, 3, 6, 8, 8, 4, 4, 4, 4, 8, 14, 16, 12, 4, 0, 0]
        jaw = [0] * 16  # the jaw layer stays; mouth_open carves the opening
        p["mane"] = float(mane[f])
        p["flame"] = float(flame[f])
        p["crest"] = float(crest[f])
        p["beard"] = float(beard[f])
        p["arm"] = float(arm[f])
        p["arm_low"] = float(arm_low[f])
        p["jaw"] = float(jaw[f])
        p["arm_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["arm_low_mode"] = "up"
        p["jaw_mode"] = "down"
        p["turned"] = 5 <= f <= 8
        if 1 <= f <= 4:
            p["wedge"] = f if f < 4 else 3
            p["ring"] = max(0, f - 1)
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.64, 0.42, 0.24, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(4 * env))
        p["flash"] = f == 1
        p["mane"] = 8.0 * env
        p["crest"] = 8.0 * env
    return finish(p, action, f, __file__)


def part_angle(rig, name, mode, amp):
    if abs(amp) < 0.45 or name not in rig["signs"]:
        return 0.0
    sign = rig["signs"][name]
    return sign.get(mode, sign["up"]) * amp


def moved(layer, ang, pivot, dx, dy):
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, pivot)
    return shift(layer, dx, dy)


def limb_motion(rig, p, name):
    if name == "leg_l":
        return p["step_l"], 0.0
    if name == "leg_r":
        return p["step_r"], 0.0
    if name == "arm_r":
        return (0, 0), part_angle(rig, name, p["arm_mode"], p["arm"])
    if name == "arm_l":
        return (0, 0), part_angle(rig, name, p["arm_low_mode"], p["arm_low"])
    if name == "jaw":
        return (0, 0), part_angle(rig, name, p["jaw_mode"], p["jaw"])
    if name in ("mane_l", "mane_r"):
        return (0, 0), part_angle(rig, name, "up", p["mane"])
    if name == "flame_l":
        return (0, 0), part_angle(rig, name, "up", p["flame"])
    if name == "crest":
        return (0, 0), part_angle(rig, name, "up", p["crest"])
    if name == "beard":
        return (0, 0), part_angle(rig, name, "up", p["beard"])
    return (0, 0), 0.0


# Beard is a one-row neck into the hip. Arms paste after it used to erase
# that neck on the mirrored front and leave the flame floating. It is drawn
# after the arms so the neck stays. At rest the parts do not overlap.
ORDER = (
    "leg_l", "leg_r", "flame_l", "crest",
    "arm_l", "arm_r", "beard", "jaw", "mane_l", "mane_r",
)


def place_limb(rig, p, name, bdx, bdy):
    """Beard hangs on one fist, so it takes that fist's turn, then its own sway."""
    L = rig["layers"]
    fc = rig["facing"]
    if name == "beard":
        holder = "arm_r" if rig["kind"] == "front" else "arm_l"
        _, ang_arm = limb_motion(rig, p, holder)
        _, ang_extra = limb_motion(rig, p, "beard")
        layer = L["beard"]
        piv_c = rig["pivots"]["beard_contact"]
        if abs(ang_arm) >= 0.45:
            layer = rotate_nearest(layer, ang_arm, rig["pivots"][holder])
            piv_c = rot_point(piv_c[0], piv_c[1], ang_arm, rig["pivots"][holder])
        if abs(ang_extra) >= 0.45:
            layer = rotate_nearest(layer, ang_extra, piv_c)
        return shift(layer, bdx, bdy)
    (sx, sy), ang = limb_motion(rig, p, name)
    dx = bdx + int(round(fc * sx))
    dy = bdy + int(sy)
    return moved(L[name], ang, rig["pivots"][name], dx, dy)


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    for name in ORDER:
        paste(out, place_limb(rig, p, name, bdx, bdy))
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK)
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(img, facing):
    """Rear hoof, not the thin flame that also touches the ground."""
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    runs = []
    for y in range(max(0, yb - 3), yb + 1):
        cols = np.nonzero(a[y])[0]
        if len(cols) == 0:
            continue
        s = p = int(cols[0])
        found = []
        for x in list(cols[1:]) + [None]:
            if x is not None and int(x) == p + 1:
                p = int(x)
                continue
            if p - s + 1 >= 4:
                found.append((s, p, y))
            if x is None:
                break
            s = p = int(x)
        runs.extend(found)
    if not runs:
        band = ys >= yb - 2
        xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
        return float(xb), float(yb)
    if facing > 0:
        s, _p, _y = min(runs, key=lambda t: t[0])
        return float(s), float(yb)
    _s, p, _y = max(runs, key=lambda t: t[1])
    return float(p), float(yb)


def fore_tip(rig, p, name):
    fc = rig["facing"]
    (sx, sy), ang = limb_motion(rig, p, name)
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    dx = bdx + int(round(fc * sx))
    dy = bdy + int(sy)
    layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def mouth_wedge(img, origin, age):
    """Dark mouth interior in the gap the dropped jaw leaves. Down, not sideways."""
    ox, oy = origin
    for i in range(age):
        span = i
        for dx in range(-span, span + 1):
            col = ORANGE if abs(dx) == span else DK
            put(img, ox + dx, oy + 1 + i, col)


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (YELLOW, ORANGE, WHITE)
    rad = 12 + age * 4
    for deg in range(-60, 61, 10):
        a = math.radians(deg)
        x = ox + fc * rad * math.cos(a)
        y = oy + rad * math.sin(a) * 0.55
        put(img, x, y, cols[age % 3])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (YELLOW, WHITE, ORANGE)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2), (2, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def moving(p):
    if p["step_l"] != (0, 0) or p["step_r"] != (0, 0) or p["bdx"] or p["bdy"]:
        return True
    if abs(p["lean"]) > 0.4 or p["knock"]:
        return True
    for key in ("mane", "flame", "crest", "beard", "arm", "arm_low", "jaw"):
        if abs(p[key]) >= 0.45:
            return True
    return False


def rig_for(rigs, d, p):
    rig = rigs[d]
    if p["turned"]:
        kind, mirror = VIEW_OF[d]
        rig = rigs[("back" if kind == "front" else "front", not mirror)]
    return rig


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
            piv = rear_pivot(body, fc)
            body = rotate_nearest(body, -fc * p["lean"], piv)
        body = reconnect(fill_pinholes(despeckle(body)))
        img = snap(body, palette)
        amt = MOUTH_OPEN[f] if action == "attack" else 0
        if amt and rig["mouth_line"] is not None and not p["turned"]:
            line = mouth_open.transform_points(
                rig["mouth_line"], dx=int(round(rig["facing"] * p["bdx"])), dy=int(round(p["bdy"])))
            img = mouth_open.open_mouth(img, line, amount=amt, deep=INK, inner=DK, peak=0.6)
    if not effects:
        return img
    fc = rig["facing"]
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if rig["has_mouth"] and rig["muzzle"] is not None and not p["turned"]:
        mx, my = rig["muzzle"]
        if p["ring"]:
            roar_rings(img, (mx + fc * 2, my), fc, p["ring"])
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = fore_tip(rig, p, "arm_r")
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
    if p["wedge"] or p["ring"] or p["spark"] or p["flash"]:
        img = snap(img, palette)
    return img


def limb_layers(rig, p):
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    parts = {}
    for name in ORDER:
        parts[name] = place_limb(rig, p, name, bdx, bdy)
    return parts


def limb_lost(rig, p):
    if not moving(p):
        return []
    body = compose(rig, p)
    lost = []
    for name, layer in limb_layers(rig, p).items():
        limb = layer[..., 3] >= 128
        if not limb.any():
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    groups = component_list(op)
    return max(len(g) for g in groups) / float(op.sum())


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
    y0 = max(0, int(ys.min()) - pad - 28)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 24)
    x0 = max(0, int(xs.min()) - pad - 28)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 36)
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
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "mane_l": (80, 80, 255, 255), "mane_r": (180, 180, 255, 255),
        "flame_l": (255, 60, 40, 255), "crest": (255, 180, 40, 255),
        "beard": (255, 40, 40, 255), "arm_l": (40, 220, 80, 255),
        "arm_r": (255, 80, 200, 255), "jaw": (255, 255, 255, 255),
        "leg_l": (255, 220, 40, 255), "leg_r": (255, 140, 40, 255),
        "torso": (90, 90, 90, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            if name not in cols:
                continue
            part[m] = cols[name]
        a = img[..., 3] >= 128
        ys, xs = np.nonzero(a)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
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
        "dex": 500, "form": "mega", "variant": "emboar-mega", "gen": "gen5",
        "dir": "monsterSprite/gen5/0500_emboar/mega/emboar-mega",
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
    print("parts front", {k: int(v.sum()) for k, v in fm.items() if k != "mouth"},
          "back", {k: int(v.sum()) for k, v in bm.items() if k != "mouth"},
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, "muzzle", rigs["W"]["muzzle"], flush=True)

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

    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        y0, y1, x0, x1 = box
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
        clip = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
                    img = synth(rigs, d, act, f, palette)
                    a = img[..., 3] >= 128
                    if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                        clip.append(f"{act}:{d}:{f}")
                    groups = component_list(a)
                    groups.sort(key=len, reverse=True)
                    islands = [len(g) for g in groups[1:] if len(g) >= 12]
                    if islands:
                        lost.append(f"{act}:{d}:{f}:island{islands[0]}")
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        print("clip", clip[:12], "count", len(clip), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["jaw"], 1), round(p["arm"], 1), round(p["arm_low"], 1),
                        p["arm_mode"], p["turned"], p["spin"], p["wedge"], p["ring"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
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
          and partial == 0 and not clip and min(bodies) >= 0.9)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_emboar_mega_actions", "version": VERSION, "key": KEY,
        "dex": 500, "variant": "emboar-mega", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped fire boar (two fists, two hooves, mane, shoulder flames, hanging flame beard, jaw and tusks)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (nose x 41.0 < eye x 41.3); back beard is left of the body",
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
