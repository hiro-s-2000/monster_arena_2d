#!/usr/bin/env python3
"""Mamoswine normal (0473) 8-direction action sheet from this form's own stills.

Measured on normal/front.png (96x96, bbox x8-87 y13-81, 15 colours, alpha 0/255):
  - Faces screen-left (3/4 front). Eyes are the ink blobs at x36-38 y25-30 and
    x55-60 y23-30 inside the blue mask; the pink snout is x34-45 y32-41.
  - No mouth is drawn (it is hidden under the snout and beard), so the roar does
    not invent one: it raises the existing tusks, rears the body and puts roar
    rings and snow in front of the snout.
  - Two tusks: the big near tusk curls out to the left (x8-31 y38-67), the far
    tusk hangs from x48 y39 down the beard and curls right (x42-64 y39-71).
  - Legs (quadruped_beast, heavy): near fore x23-46 y>=69, dark far leg x47-56,
    off fore x56-72 y>=71, hind x>=74 y>=64. No tail is visible from the front.
normal/back.png (bbox x9-85 y13-81) faces away to the right: tusk x66-86 y39-69,
  tail tuft x13-25 y47-57, three leg columns split by ink at y72 (x<=29,
  x30-55, x>=56).

Directions: S SE E (front mirrored, looks screen-right), SW W (front as drawn,
looks screen-left), NE (back as drawn, away-right), N NW (back mirrored,
away-left). E = screen right, W = screen left, N = back, S = front.

  idle   : feet planted, 1px breathing, tusk sway, a frosty breath puff.
  walk   : heavy diagonal walk (near fore + hind, off fore + far leg), 1 cycle.
  attack : 0 still, 1-4 rear up + tusks raised + roar rings and snow,
           5-10 jumping 360 turn about the vertical axis through this form's own
           front/back views (the body is never laid on its side) with a blizzard
           swirl, 11-12 stomp landing with ice spikes and a ground ring,
           13-15 recover.
  hurt   : 12 frames, white flash, knock-back lean (<=10 deg) and recover.

Head (face) moves by integer pixels only, except in hurt's small lean.
Palette locked with agent-sprite-forge palette_tool.py. Alpha is 0 or 255.
front/back png and gif are never written.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_mamoswine_actions.py [--samples] [--force] [--no-sync] [--skip-qa] [--no-publish]
"""

import argparse
import fcntl
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0473_mamoswine/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0473_mamoswine/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
BOUNDS = ROOT / "data/sprite_bounds.js"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

KEY = "0473_mamoswine"
VERSION = "mamoswine_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
SRC_FOOT = 81
OFF_X = CELL_W // 2 - 48
OFF_Y = FOOT_Y - SRC_FOOT
# view, mirrored. front.png looks screen-left, back.png looks away to the right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "SW": ("front", False), "W": ("front", False),
    "NE": ("back", False), "N": ("back", True), "NW": ("back", True),
}
MAX_LEAN = 10.0

INK = (16, 16, 16)
DK = (57, 49, 49)
DKR = (90, 57, 57)
FUR_D = (115, 74, 65)
TUSK_D = (90, 82, 98)
BLUE_D = (65, 90, 139)
FUR = (156, 106, 74)
NOSE_D = (164, 98, 123)
GREY = (139, 139, 148)
BEIGE_D = (180, 164, 131)
NOSE = (230, 148, 189)
SILVER = (189, 189, 197)
BEIGE = (230, 222, 172)
WHITE = (238, 246, 246)
BLUE = (82, 148, 197)
TUSK_COLS = (WHITE, SILVER, GREY, TUSK_D)

SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)


# ---------------------------------------------------------------- helpers
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
    """4-connected flood of mask from the seed mask."""
    out = seeds & mask
    while True:
        nxt = dilate(out, 1) & mask
        if (nxt == out).all():
            return out
        out = nxt


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    sizes = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        n = 0
        while stack:
            y, x = stack.pop()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        sizes.append(n)
    return sorted(sizes, reverse=True)


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    s = components(op)
    return s[0] / float(op.sum())


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
    d = ((px[:, None, :] - palette.astype(np.int32)[None, :, :]) ** 2).sum(axis=2)
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
    """Grow the layer's own colours into allowed pixels (hidden at rest)."""
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        for d in dirs:
            mv = np.zeros_like(src)
            if d == "up":      # take from the pixel below
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
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


# ---------------------------------------------------------------- segmentation
def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    ink_like = any_color(img, (INK, TUSK_D))
    core = any_color(img, (WHITE, SILVER, GREY)) & (yy >= 38) & (yy <= 71)
    big_core = core & (xx <= 31)
    far_core = core & (xx >= 40) & (xx <= 66)
    big = big_core | (dilate(big_core, 1) & ink_like & (xx <= 32) & (yy >= 38) & (yy <= 68))
    far = far_core | (dilate(far_core, 1) & ink_like & (xx >= 40) & (xx <= 66) & (yy >= 38) & (yy <= 71))
    far &= ~big
    tusks = big | far
    leg_l = op & (yy >= 69) & (xx <= 46) & ~tusks
    leg_m = op & (yy >= 70) & (xx >= 47) & (xx <= 56) & ~tusks
    leg_r = op & (yy >= 71) & (xx >= 57) & (xx <= 73) & ~tusks
    leg_h = op & (yy >= 64) & (xx >= 74) & ~tusks
    used = big | far | leg_l | leg_m | leg_r | leg_h
    torso = op & ~used
    masks = {"tusk_big": big, "tusk_far": far, "leg_l": leg_l, "leg_m": leg_m,
             "leg_r": leg_r, "leg_h": leg_h, "torso": torso}
    check_partition(op, masks)
    # sanity from the pixels
    eyes = is_color(img, INK) & (yy >= 23) & (yy <= 30) & (xx >= 34) & (xx <= 61)
    if int((eyes & torso).sum()) != int(eyes.sum()) or eyes.sum() < 15:
        raise SystemExit("eyes left the head")
    nose = any_color(img, (NOSE, NOSE_D))
    if int((nose & torso).sum()) != int(nose.sum()):
        raise SystemExit("snout was cut")
    if big.sum() < 250 or far.sum() < 150:
        raise SystemExit(f"tusks look wrong: {int(big.sum())} {int(far.sum())}")
    nx = float(np.nonzero(nose)[1].mean())
    ex = float(np.nonzero(eyes)[1].mean())
    if nx >= ex + 2:
        raise SystemExit("snout is not on the left; facing changed")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    ink_like = any_color(img, (INK, TUSK_D))
    core = any_color(img, (WHITE, SILVER, GREY)) & (yy >= 39) & (xx >= 66)
    tusk = core | (dilate(core, 1) & ink_like & (xx >= 66) & (yy >= 39))
    tail = op & (xx >= 13) & (xx <= 25) & (yy >= 48) & (yy <= 57) & any_color(img, (FUR, INK, DK))
    tail &= ~(xx <= 14)
    lb_a = op & (yy >= 69) & (xx <= 29) & ~tusk
    lb_b = op & (yy >= 69) & (xx >= 30) & (xx <= 55) & ~tusk
    lb_c = op & (yy >= 69) & (xx >= 56) & ~tusk
    used = tusk | tail | lb_a | lb_b | lb_c
    torso = op & ~used
    masks = {"tusk": tusk, "tail": tail, "lb_a": lb_a, "lb_b": lb_b, "lb_c": lb_c, "torso": torso}
    check_partition(op, masks)
    if tusk.sum() < 150:
        raise SystemExit(f"back tusk looks wrong: {int(tusk.sum())}")
    return masks


def check_partition(op, masks):
    total = sum(int(m.sum()) for m in masks.values())
    if total != int(op.sum()):
        raise SystemExit(f"parts do not partition the still: {total} vs {int(op.sum())}")


# ---------------------------------------------------------------- rig
def place(arr):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    out[OFF_Y:OFF_Y + 96, OFF_X:OFF_X + 96] = arr
    return out


def enclosed(mask_part, body, reach=14):
    """Part pixels with body on the left, right, above and below (in reach)."""
    h, w = body.shape
    out = np.zeros_like(mask_part)
    for y, x in zip(*np.nonzero(mask_part)):
        l = body[y, max(0, x - reach):x].any()
        r = body[y, x + 1:x + 1 + reach].any()
        u = body[max(0, y - reach):y, x].any()
        d = body[y + 1:y + 1 + reach, x].any()
        out[y, x] = l and r and u and d
    return out


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
    if kind == "front":
        tusk_names = ("tusk_big", "tusk_far")
        leg_names = ("leg_l", "leg_m", "leg_r", "leg_h")
    else:
        tusk_names = ("tusk",)
        leg_names = ("lb_a", "lb_b", "lb_c")
    # Fur under the tusks where they lie over the body (hidden at rest).
    body = torso_m.copy()
    for ln in leg_names:
        body |= placed[ln]
    under = np.zeros_like(torso_m)
    for tn in tusk_names:
        under |= enclosed(placed[tn], body)
    if kind == "back":
        under |= enclosed(placed["tail"], body)
    layers["torso"] = paint_cushion(layers["torso"], under, 12)
    # Legs reach up under the torso so a body bob never opens a gap.
    for ln in leg_names:
        layers[ln] = paint_cushion(layers[ln], torso_m & ~under, 4, dirs=("up",))
    facing = 1.0 if mirror else -1.0
    if kind == "back":
        facing = -facing  # back.png as drawn looks right
    pivots = {}
    for tn in tusk_names:
        pivots[tn] = joint(placed[tn], torso_m, kind, tn, facing)
    if kind == "back":
        pivots["tail"] = joint(placed["tail"], torso_m, kind, "tail", facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    # hind foot (rear side, ground) for the hurt lean
    rear = xs.max() if facing < 0 else xs.min()
    pivots["rear_foot"] = (float(rear - facing * 10), float(FOOT_Y))
    nose = any_color(canvas, (NOSE, NOSE_D))
    if nose.any():
        ny, nx = np.nonzero(nose)
        pivots["snout"] = (float(nx.mean()), float(ny.mean()))
    else:
        pivots["snout"] = pivots["center"]
    return {"kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
            "pivots": pivots, "canvas": canvas, "legs": leg_names, "tusks": tusk_names}


def joint(part, torso, kind, name, facing):
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part
    ys, xs = np.nonzero(touch)
    if name == "tusk_far":
        band = ys <= ys.min() + 2          # hangs from its top
    elif name == "tusk_big":
        # root is the side toward the body (away from the curl)
        band = (xs >= xs.max() - 3) if facing < 0 else (xs <= xs.min() + 3)
    elif name == "tusk":
        band = ys <= ys.min() + 3
    else:  # tail
        band = (xs >= xs.max() - 1) if facing < 0 else (xs <= xs.min() + 1)
    return float(xs[band].mean()), float(ys[band].mean())


# ---------------------------------------------------------------- poses
def base_pose():
    return dict(bdx=0, bdy=0, hop=0, fore=0, legA=(0, 0), legB=(0, 0), tusk=0.0, tail=0.0,
                view=None, lean=0.0, knock=0, flash=False, ring=-1, snow=0,
                puff=-1, swirl=-1, land=0, spikes=0, stars=0)


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        p["bdy"] = -1 if wave(f, n, 1.0) > 0.55 else 0
        p["tusk"] = 1 if wave(f, n, 1.0, phase=0.6) > 0.7 else 0
        p["tail"] = wave(f, n, 10.0, phase=0.0, cycles=2.0)
        if 7 <= f <= 11:
            p["puff"] = f - 7
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa, sb = math.sin(u), math.sin(u + math.pi)
        p["legA"] = (int(round(1.5 * sa)), -int(round(3.2 * max(0.0, sa))))
        p["legB"] = (int(round(1.5 * sb)), -int(round(3.2 * max(0.0, sb))))
        p["bdy"] = -int(round(0.9 * abs(math.sin(u))))
        p["bdx"] = 0
        p["tusk"] = 1 if math.sin(2 * u) > 0.6 else 0
        p["tail"] = wave(f, n, 14.0, cycles=2.0, phase=math.pi / 2)
    elif action == "attack":
        # 1-4 rear up and roar
        if 1 <= f <= 4:
            p["bdy"] = [0, -1, -2, -2, -2][f]
            p["bdx"] = [0, 0, 1, 1, 1][f]           # weight back (sign = away from facing)
            p["fore"] = [0, -1, -2, -3, -3][f]
            p["tusk"] = [0, 1, 2, 3, 3][f]
            p["tail"] = [0, 10, 18, 22, 22][f]
            p["ring"] = f - 2 if f >= 2 else -1
            p["snow"] = f
        # 5-10 jumping turn through own views
        if 5 <= f <= 10:
            p["view"] = f - 5
            p["hop"] = [-4, -8, -10, -10, -7, -3][f - 5]
            p["tusk"] = 2
            p["legA"] = (0, -1)
            p["legB"] = (0, -1)
            p["swirl"] = f - 5
        # 11-12 stomp
        if f in (11, 12):
            p["bdy"] = 1 if f == 11 else 0
            p["tusk"] = -1 if f == 11 else 0
            p["land"] = f - 10
            p["spikes"] = 3 if f == 11 else 4
        if 13 <= f <= 15:
            p["tusk"] = 0
            p["spikes"] = [3, 2, 0][f - 13]
            p["land"] = 3 if f == 13 else 0
    else:  # hurt
        env = [0.0, 0.5, 0.9, 1.0, 0.85, 0.65, 0.45, 0.28, 0.14, 0.05, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(4 * env))
        p["tusk"] = 2 if env > 0.5 else (1 if env > 0.2 else 0)
        p["fore"] = -int(round(2 * env))
        p["tail"] = 18 * env
        p["flash"] = f == 1
        p["stars"] = f - 1 if 1 <= f <= 5 else 0
    return p


# ---------------------------------------------------------------- compose
def moved(layer, ang, pivot, dx, dy):
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, pivot)
    return shift(layer, dx, dy)


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = -fc * p["bdx"]        # positive bdx = weight backward
    bdy = p["bdy"] + p["hop"]
    hop = p["hop"]
    if rig["kind"] == "front":
        pair = {"leg_l": "legA", "leg_m": "legA", "leg_r": "legB", "leg_h": "legB"}
        fore = ("leg_l", "leg_r")
        order_back = ("leg_m", "leg_h", "leg_r", "leg_l")
    else:
        pair = {"lb_a": "legA", "lb_c": "legA", "lb_b": "legB"}
        fore = ("lb_c",) if fc > 0 else ("lb_a",)
        order_back = ("lb_a", "lb_c", "lb_b")
    out = np.zeros_like(rig["canvas"])
    for ln in order_back:
        ldx, ldy = p[pair[ln]]
        if ln in fore:
            ldy += p["fore"]
        paste(out, shift(L[ln], fc * ldx, ldy + hop))
    paste(out, shift(L["torso"], bdx, bdy))
    if rig["kind"] == "back":
        # the tail tuft sits on the rump; fur is painted under it
        paste(out, moved(L["tail"], -fc * p["tail"], piv["tail"], bdx, bdy))
    for tn in rig["tusks"]:
        # raise = forward and up by whole pixels; the tusk art is never resampled
        lift = int(round(p["tusk"]))
        fwd = (1 if lift >= 2 else 0) if tn != "tusk_far" else (1 if lift >= 3 else 0)
        if lift < 0:
            fwd = 0
        up = lift if tn != "tusk_far" else (lift + 1) // 2
        paste(out, shift(L[tn], bdx + fc * fwd, bdy - up))
    return out


def rest_of(rig):
    return rig["canvas"]


# ---------------------------------------------------------------- effects
def ellipse_pts(cx, cy, rx, ry, step_deg, a0=0, a1=360):
    for d in range(a0, a1, step_deg):
        a = math.radians(d)
        yield cx + rx * math.cos(a), cy + ry * math.sin(a)


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, BLUE, SILVER)
    for ri in range(age + 1):
        rad = 10 + ri * 8
        for deg in range(-56, 57, 5):
            a = math.radians(deg)
            x = ox + fc * rad * math.cos(a)
            y = oy + rad * math.sin(a) * 0.8
            put(img, x, y, cols[ri % 3])
            put(img, x + fc, y, BLUE if ri % 3 == 0 else WHITE)


def snow(img, origin, fc, f, count):
    ox, oy = origin
    rng = np.random.RandomState(73 + f)
    for i in range(count * 4):
        d = 6 + rng.randint(0, 10 + 6 * count)
        y = oy + rng.randint(-10, 11)
        x = ox + fc * d
        put(img, x, y, WHITE)
        if i % 3 == 0:
            put(img, x + fc, y, BLUE)


def breath_puff(img, origin, fc, age):
    ox, oy = origin
    x = ox + fc * (5 + age * 2)
    y = oy + 4 - age
    cols = (WHITE, SILVER)
    for dx, dy in ((0, 0), (1, 0), (0, -1), (1, -1), (-1, 0)):
        if age >= 3 and (dx, dy) in ((1, -1), (-1, 0)):
            continue
        put(img, x + fc * dx, y + dy, cols[(dx + dy) % 2])


def blizzard(img, center, age):
    """Swirling snow around the turning body (separate from the body)."""
    cx, cy = center
    for k in range(10):
        a = math.radians(k * 36 + age * 47)
        rx, ry = 50, 18
        x = cx + rx * math.cos(a)
        y = cy + 10 + ry * math.sin(a)
        put(img, x, y, WHITE if k % 2 else BLUE)
        put(img, x + 1, y, SILVER)


def ground_ring(img, fc, cx, age):
    rx = 30 + age * 10
    ry = 4 + age
    for x, y in ellipse_pts(cx, FOOT_Y + 1, rx, ry, 9):
        if y >= FOOT_Y:
            put(img, x, y, BEIGE_D if age % 2 else FUR)


def ice_spike(img, x, h):
    """A small ice spike standing on the ground line."""
    x = int(round(x))
    for k in range(h):
        y = FOOT_Y - k
        half = max(0, (h - k) // 4)
        for dx in range(-half - 1, half + 2):
            col = WHITE if dx == 0 else (BLUE if abs(dx) <= half else BLUE_D)
            put(img, x + dx, y, col, over=False)
    put(img, x, FOOT_Y - h, INK)


def spikes(img, fc, front_x, n):
    heights = (12, 17, 14, 9)
    for i in range(n):
        x = front_x + fc * (3 + i * 8)
        ice_spike(img, x, heights[i])


def hit_stars(img, origin, fc, age):
    ox, oy = origin
    r = 4 + age * 2
    for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r), (r - 2, -(r - 2)), (-(r - 2), r - 2)):
        put(img, ox + dx, oy + dy, WHITE)
    put(img, ox, oy, BEIGE)


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = any_color(out, (INK, DK))
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def body_front_x(img, fc):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    return int(xs.max()) if fc > 0 else int(xs.min())


# ---------------------------------------------------------------- synth
def turn_views(rigs, d):
    """Cycle front-left -> back-left -> back-right -> front-right (one turn)."""
    kind, mirror = VIEW_OF[d]
    own = ("front", "L" if not mirror else "R") if kind == "front" else ("back", "L" if mirror else "R")
    cyc = [("front", "L"), ("back", "L"), ("back", "R"), ("front", "R")]
    i = cyc.index(own)
    seq = [cyc[(i + 1) % 4], cyc[(i + 2) % 4], cyc[(i + 3) % 4], own, own, own]
    out = []
    for k, side in seq:
        if k == "front":
            out.append(rigs["front_R" if side == "R" else "front_L"])
        else:
            out.append(rigs["back_R" if side == "R" else "back_L"])
    return out


def synth(rigs, d, action, f, palette, effects=True):
    rig = rigs[d]
    fc = rig["facing"]
    p = pose(action, f)
    use = rig
    if p["view"] is not None:
        use = turn_views(rigs, d)[p["view"]]
    body = compose(use, p)
    if p["lean"] > 0.4 or p["knock"]:
        piv = rig["pivots"]["rear_foot"]
        body = shift(body, -fc * p["knock"], 0)
        body = rotate_nearest(body, -fc * p["lean"], (piv[0] - fc * p["knock"], piv[1]))
        body = fill_pinholes(body)
        body = despeckle(body)
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    sx, sy = rig["pivots"]["snout"]
    bdx = -fc * p["bdx"]
    snout = (sx + bdx + fc * 4, sy + p["bdy"] + 2)
    if rig["kind"] == "back":
        snout = (sx + fc * 8, sy + p["bdy"])
    if p["ring"] >= 0:
        roar_rings(img, (snout[0] + fc * 12, snout[1] - 2), fc, p["ring"])
    if p["snow"]:
        snow(img, snout, fc, f, p["snow"])
    if p["puff"] >= 0:
        breath_puff(img, snout, fc, p["puff"])
    cx, cy = rig["pivots"]["center"]
    if p["swirl"] >= 0:
        blizzard(img, (cx, cy + p["hop"]), p["swirl"])
    if p["land"]:
        ground_ring(img, fc, cx, p["land"])
    if p["spikes"]:
        spikes(img, fc, body_front_x(img, fc) if not p["land"] else body_front_x(body, fc), p["spikes"])
    if p["stars"]:
        hit_stars(img, (sx + fc * 6, sy - 6), fc, p["stars"] - 1)
    return snap(img, palette)


# ---------------------------------------------------------------- outputs
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


def tile_grid(tiles, cols, bg=(15, 20, 31, 255)):
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    out = np.full((rows * th, cols * tw, 4), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        m = t[..., 3] >= 128
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
    return out


def preview_image(sheets):
    """Rows: actions. Columns: 8 directions at a telling frame."""
    pick = {"idle": 0, "walk": 4, "attack": 3, "hurt": 3}
    tiles = []
    for act in ACTIONS:
        for di in range(8):
            f = pick[act]
            tiles.append(sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W])
    return Image.fromarray(tile_grid(tiles, 8), "RGBA")


def motion_contact(sheets, di=0):
    pad = 4
    rows = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            row.append(sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W])
        rows.append(row)
    # crop all tiles to a common box
    allop = np.zeros((CELL_H, CELL_W), bool)
    for row in rows:
        for t in row:
            allop |= t[..., 3] >= 128
    ys, xs = np.nonzero(allop)
    y0, y1 = max(0, ys.min() - pad), min(CELL_H, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(CELL_W, xs.max() + 1 + pad)
    tiles = []
    for row in rows:
        for f in range(16):
            tiles.append(row[f][y0:y1, x0:x1] if f < len(row) else np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
    return Image.fromarray(tile_grid(tiles, 16), "RGBA")


# ---------------------------------------------------------------- publish
def with_flock(path, fn):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _object_span(text, key):
    token = f'"{key}":'
    if text.count(token) != 1:
        raise SystemExit(f"{key} appears {text.count(token)} times")
    s = text.index(token)
    i = s + len(token)
    if text[i] != "{":
        raise SystemExit(f"{key} value is not an object")
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return s, i, j + 1
    raise SystemExit(f"{key} object did not close")


def publish_dir8(sprite_h):
    entry = {
        "dex": 473, "form": "normal", "variant": "mamoswine", "gen": "gen4",
        "dir": "monsterSprite/gen4/0473_mamoswine/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed")
        _s, val, end = _object_span(text, KEY)
        new_text = text[:val] + blob + text[end:]
        parsed = json.loads(new_text[len(prefix):].rstrip().rstrip(";"))
        old = json.loads(text[len(prefix):].rstrip().rstrip(";"))
        if parsed[KEY] != entry:
            raise SystemExit("dir8 entry did not parse back")
        for k in old:
            if k != KEY and old[k] != parsed.get(k):
                raise SystemExit(f"dir8 neighbour changed: {k}")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(new_text)
        tmp.replace(DIR8)

    with_flock(DIR8_LOCK, edit)


def publish_bounds(bounds):
    text = BOUNDS.read_text()
    token = f'"{KEY}":['
    i = text.find('"s":{')
    if i < 0:
        raise SystemExit("sprite_bounds has no s map")
    j = text.find(token, i)
    if j < 0:
        raise SystemExit("sprite_bounds has no s entry for mamoswine")
    k = text.index("]", j)
    blob = token + ",".join(str(int(v)) for v in bounds) + "]"
    new = text[:j] + blob + text[k + 1:]
    tmp = BOUNDS.with_suffix(".js.tmp")
    tmp.write_text(new)
    tmp.replace(BOUNDS)


def update_queue(status, note=None):
    def edit():
        raw = QUEUE.read_text()
        marker = f'"id": "{KEY}"'
        if raw.count(marker) != 1:
            raise SystemExit("queue id count")
        idx = raw.index(marker)
        start = raw.rfind("{", 0, idx)
        depth = 0
        end = None
        for j in range(start, len(raw)):
            if raw[j] == "{":
                depth += 1
            elif raw[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        block = raw[start:end]
        block2, n = re.subn(r'"status": "[^"]*"', f'"status": "{status}"', block, count=1)
        if n != 1:
            raise SystemExit("queue status not found")
        if note:
            block2 = block2.replace('"notes": "', '"notes": "' + note.replace('"', "'") + " | ", 1)
        new = raw[:start] + block2 + raw[end:]
        old = json.loads(raw)
        data = json.loads(new)
        for a, b in zip(old["items"], data["items"]):
            if a.get("id") != KEY and a != b:
                raise SystemExit("queue neighbour changed")
        tmp = QUEUE.with_suffix(".json.tmp")
        tmp.write_text(new)
        tmp.replace(QUEUE)

    with_flock(QUEUE_LOCK, edit)


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    return n


# ---------------------------------------------------------------- palette lock
def locked_palette(front_path, work):
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
        r = subprocess.run([py, "-I", str(PALETTE_TOOL), "build", "--input", str(front_path),
                            "--colors", str(len(hexes)), "--reserve", *hexes,
                            "--output-dir", str(pal_dir)], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("palette build failed: " + r.stdout + r.stderr)
        r = subprocess.run([py, "-I", str(PALETTE_TOOL), "lock", "--palette", str(pal_dir / "palette.json"),
                            "--source", str(front_path), "--strict", "--output-dir", str(lock_dir)],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("palette lock failed: " + r.stdout + r.stderr)
    data = json.loads((lock_dir / "palette.json").read_text())
    cols = np.array([[int(c["hex"][i:i + 2], 16) for i in (1, 3, 5)] for c in data["colors"]], np.uint8)
    got = {tuple(int(v) for v in c) for c in cols}
    want = {tuple(int(v) for v in c) for c in own}
    if got != want:
        raise SystemExit(f"locked palette differs from the still: {sorted(got ^ want)}")
    return cols


# ---------------------------------------------------------------- main
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
    fys = np.nonzero(front[..., 3])[0]
    bys = np.nonzero(back[..., 3])[0]
    if fys.max() != SRC_FOOT or bys.max() != SRC_FOOT:
        raise SystemExit("foot line changed")
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {
        "front_L": build_view(front, fm, "front", False),
        "front_R": build_view(front, fm, "front", True),
        "back_R": build_view(back, bm, "back", False),
        "back_L": build_view(back, bm, "back", True),
    }
    for d in DIRS:
        kind, mirror = VIEW_OF[d]
        if kind == "front":
            rigs[d] = rigs["front_R" if mirror else "front_L"]
        else:
            rigs[d] = rigs["back_L" if mirror else "back_R"]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = [(80, 140, 210), (240, 200, 40), (220, 40, 40), (40, 200, 80), (240, 120, 40),
            (200, 80, 220), (120, 220, 220)]
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for (name, m), c in zip(masks.items(), cols):
            part[m, :3] = c
            part[m, 3] = 255
        tiles.append(np.repeat(np.repeat(part, 3, 0), 3, 1))
    Image.fromarray(np.concatenate(tiles, axis=1), "RGBA").save(work / "parts.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    palette = locked_palette(SRC_DIR / "front.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()}, "palette", len(palette), flush=True)

    rest_diff = {}
    for d in DIRS:
        r = synth(rigs, d, "idle", 0, palette, effects=False)
        rest_diff[d] = int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    if args.samples:
        groups = []
        for d in ("W", "S", "N", "NE"):
            row = []
            for act, fs in (("idle", (0, 8, 9)), ("walk", (2, 4, 6, 10, 12, 14)),
                            ("attack", (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14)), ("hurt", (1, 3, 6, 10))):
                for f in fs:
                    row.append(synth(rigs, d, act, f, palette))
            groups.extend(row)
        Image.fromarray(tile_grid(groups, 25), "RGBA").save(work / "samples.png")
        # zoomed close-up of W
        tiles = [synth(rigs, "W", a, f, palette)[60:160, 20:156] for a, f in
                 (("idle", 0), ("walk", 4), ("walk", 12), ("attack", 3), ("attack", 6), ("attack", 7),
                  ("attack", 11), ("hurt", 3))]
        z = tile_grid(tiles, 4)
        Image.fromarray(np.repeat(np.repeat(z, 2, 0), 2, 1), "RGBA").save(work / "samples_zoom.png")
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
                if fp.exists() and fp.stat().st_size > 0:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview_image(sheets).save(OUT_DIR / "preview.png")
    motion_contact(sheets, 0).save(OUT_DIR / "preview_motion.png")
    motion_contact(sheets, 6).save(work / "preview_motion_W.png")
    motion_contact(sheets, 4).save(work / "preview_motion_N.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", OUT_DIR / "actions" / "preview.png")

    # ---------------- QC
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
                if act in ("attack", "hurt"):
                    bf = largest_fraction(synth(rigs, d, act, f, palette, effects=False))
                    bodies.append(bf)
    bounds = []
    for di in range(8):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        ys, xs = np.nonzero(t[..., 3] == 255)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    sprite_h = bounds[3]
    report = {
        "version": VERSION, "hash": digest, "rest_diff": rest_diff,
        "off_palette_colors": len(off), "partial_alpha": partial, "edge_touching": clip,
        "body_main_min": round(min(bodies), 3), "sprite_h": sprite_h, "bounds": bounds,
        "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        qa = work / "qa"
        qa.mkdir(parents=True, exist_ok=True)
        for act, kind in (("idle", "hover"), ("walk", "hover"), ("walk", "gait")):
            od = qa / f"gait_{act}_{kind}_S"
            if od.exists():
                shutil.rmtree(od)
            cmd = [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                   "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"]
            if kind == "gait":
                cmd += ["--state", "walk"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            rep = od / "loop-report.json"
            if rep.exists():
                rr = json.loads(rep.read_text())
                report[f"gait_{act}_{kind}"] = rr.get("recommendation", rr.get("status"))
            else:
                report[f"gait_{act}_{kind}"] = (r.stdout + r.stderr).strip()[-200:]
    ok = (not any(rest_diff.values()) and not off and partial == 0 and not clip
          and min(bodies) >= 0.97)
    report["ok"] = ok
    (qa_dir := work / "qa").mkdir(parents=True, exist_ok=True)
    (qa_dir / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False)[:1500], flush=True)

    meta = {
        "generator": "forge_mamoswine_actions", "version": VERSION, "key": KEY, "dex": 473,
        "variant": "mamoswine", "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y,
        "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "views": {d: list(VIEW_OF[d]) for d in DIRS}, "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "archetype": "quadruped_beast (heavy, tusks)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit("qa failed; not publishing")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if not args.no_publish:
        publish_dir8(sprite_h)
        publish_bounds(bounds)
        update_queue("done")
        print("published", KEY, VERSION, "h", sprite_h, flush=True)


if __name__ == "__main__":
    main()
