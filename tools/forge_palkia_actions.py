#!/usr/bin/env python3
"""Palkia (0484) normal 8-direction action sheet from this form's front.png / back.png.

Measured on normal/front.png (96x96, alpha 0/255, 3348 opaque, 15 colours):
  bbox x9-86 y15-81. Faces screen-left: the red eye (x16-18 y22-23) sits right
  of the snout tip (x9). Upright biped dragon with two claw arms, a long
  shoulder plate, a pearl on the near shoulder and a tail curled up behind.
  jaw     lower jaw under the snout, x9-20 y25-29 (mouth corner 'dpp' x12-14
          y25-26), hinge at its back top (21,25)
  head    long snout + crest + eye on the torso layer (integer moves only)
  pearl   pink/red pearl on the near shoulder x52-58 y31-38 (stays on the torso)
  plate   the long pale shoulder plate running right, x63-86 y22-38
  tail    the tail curling up behind, x66-86 y39-71 (base behind the near leg)
  arm_f   the far arm reaching forward, grey forearm + white claws x10-33 y43-53
  arm_n   the near arm folded against the chest, x55-67 y44-53
  leg_f   far leg (thigh x21-34 y54-63, shin + foot x21-41 y64-78)
  leg_n   near leg (thigh x49-65 y55-66, shin + foot x52-70 y67-81)

back.png (96x96, 3398 opaque, same palette) faces up-right:
  head and crest top-right (eye x76-78 y20-22), the pearl x59-67 y35-41,
  the shoulder plate fanning left x10-49 y21-47, tail curl x7-22 y36-67,
  the near arm reaching right x57-80 y49-60, far foot x27-48 y72-77,
  near leg x51-79 y68-83. No mouth is drawn on the back view.

Directions: front still S SW W unflipped, SE E mirrored. back still N NE
unflipped, NW mirrored. E looks screen-right, W screen-left, N is the back.

  idle   : breathing (upper body 1px, feet planted), plate and tail sway,
           arms drift, the pearl twinkles. Frame 0 and the loop end are the still.
  walk   : alternating biped steps (leg_f against leg_n) with the swing foot
           lifted, 1px bob, counter-swinging arms, tail and plate sway.
  attack : 0-4 roar: the real lower jaw drops, the plate flares, the tail
           rises, the pearl glows, violet roar rings leave the mouth (back
           rows: from the head tip) while the claw arm winds up.
           5-8 the claw arm cuts down (no whole-body rotation) with violet
           slash arcs, 9-12 a spatial rift tears open in front of the claw
           and bursts into shards, 13-15 recover.
  hurt   : 12 frames, white flash, recoil of the one-piece body (<= 18 deg)
           around the rear foot, violet shards, recover.

Palette-locked to each still. front/back png and gif are not written.
Resumable via agent-sprite-forge/_work. --force clears only this form's frames.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_palkia_actions.py [--samples]
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen4_attack_strike  # noqa: E402

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0484_palkia/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0484_palkia/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# (still, mirrored)
VIEW = {
    "S": ("front", False), "SW": ("front", False), "W": ("front", False),
    "SE": ("front", True), "E": ("front", True),
    "N": ("back", False), "NE": ("back", False), "NW": ("back", True),
}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "palkia_v1"
KEY = "0484_palkia"

INK = (16, 16, 16)
PURPLE = (139, 82, 148)
VIOLET = (189, 106, 213)
MAUVE = (205, 180, 213)
PALE = (238, 230, 255)
WHITE = (255, 255, 255)
PINK = (246, 164, 156)
RED = (246, 82, 65)
EYE = RED

PARTS = ("plate", "tail", "arm_f", "arm_n", "leg_f", "leg_n", "jaw")


# ---------------------------------------------------------------- helpers
def flood(seed_yx, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow)
    y, x = seed_yx
    if not allow[y, x]:
        raise SystemExit(f"seed {seed_yx} is not inside its allow mask")
    q = deque([(y, x)])
    seen[y, x] = True
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and allow[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def is_col(img, rgb):
    a = img[..., 3] >= 128
    c = img[..., :3]
    return a & (c[..., 0] == rgb[0]) & (c[..., 1] == rgb[1]) & (c[..., 2] == rgb[2])


def row_limit(yy, table, default):
    """Per-row x limit: table is a list of (y_from, y_to, x_max)."""
    lim = np.full(yy.shape, default, np.int32)
    for y0, y1, xm in table:
        lim[(yy >= y0) & (yy <= y1)] = xm
    return lim


def segment(img, kind):
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    empty = np.zeros_like(a)
    if kind == "front":
        jaw = a & (yy >= 25) & (yy <= 29) & (xx >= 9) & (xx <= 20)
        plate = flood((25, 75), a & (yy >= 22) & (yy <= 38) & (xx >= 63))
        arm_f = flood((49, 14), a & (xx <= 33) & (yy >= 43) & (yy <= 53))
        arm_n = flood((48, 60), a & (xx >= 55) & (xx <= 67) & (yy >= 44) & (yy <= 53))
        # far leg: thigh left of the torso stripe, then the whole shin + foot
        lim_f = row_limit(yy, [(54, 63, 34), (64, 64, 37), (65, 65, 35), (66, 66, 36),
                               (67, 68, 39), (69, 70, 40)], 41)
        leg_f = flood((76, 25), a & (yy >= 54) & (xx <= lim_f))
        # near leg: thigh x49-65, shin right of the crotch line, left of the tail
        lim_n = row_limit(yy, [(55, 66, 65), (67, 67, 65), (68, 68, 66), (69, 70, 67),
                               (71, 71, 68)], 70)
        lo_n = np.where(yy <= 66, 49, np.where(yy <= 72, 54, 51))
        leg_n = flood((79, 56), a & (yy >= 55) & (xx >= lo_n) & (xx <= lim_n) & ~arm_n)
        tail = flood((45, 80), a & (xx >= 66) & (yy >= 39) & (yy <= 71) & ~leg_n & ~arm_n)
    else:
        jaw = empty
        plate = flood((25, 20), a & (yy >= 21) & (yy <= 47)
                      & (((yy <= 27) & (xx <= 46)) | ((yy >= 28) & (xx <= 49))))
        tail = flood((45, 10), a & (xx <= 22) & (yy >= 36) & (yy <= 67) & ~plate)
        arm_f = flood((55, 70), a & (xx >= 57) & (yy >= 49) & (yy <= 60))
        arm_n = empty
        leg_f = flood((75, 29), a & (((yy >= 73) & (xx <= 50)) | ((yy == 72) & (xx >= 44) & (xx <= 48))))
        leg_n = flood((80, 60), a & (yy >= 68) & (xx >= np.where(yy >= 73, 51, 52)))
    parts = {"plate": plate, "tail": tail, "arm_f": arm_f, "arm_n": arm_n,
             "leg_f": leg_f, "leg_n": leg_n, "jaw": jaw}
    cover = np.zeros_like(a)
    for name, m in parts.items():
        if np.any(cover & m):
            raise SystemExit(f"{kind}: {name} overlaps another part")
        cover |= m
    parts = {"torso": a & ~cover, **parts}
    if kind == "front":
        eye = is_col(img, EYE) & (xx <= 30)
        if np.any(eye & ~parts["torso"]):
            raise SystemExit("the eye left the head")
    need = ("plate", "tail", "arm_f", "leg_f", "leg_n") + (("arm_n", "jaw") if kind == "front" else ())
    for name in need:
        if int(parts[name].sum()) < 30:
            raise SystemExit(f"{kind}: {name} collapsed ({int(parts[name].sum())})")
    return parts


def shift(layer, dx, dy):
    dx_i, dy_i = int(round(dx)), int(round(dy))
    if dx_i == 0 and dy_i == 0:
        return layer.copy()
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0, sx1 = max(0, -dx_i), min(w, w - dx_i)
    sy0, sy1 = max(0, -dy_i), min(h, h - dy_i)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy_i:sy1 + dy_i, sx0 + dx_i:sx1 + dx_i] = layer[sy0:sy1, sx0:sx1]
    return out


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest. Rigid, exact colours. Positive angle turns a point left of
    the pivot upward (counter-clockwise on screen)."""
    ang = float(angle_deg) % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    up = np.repeat(np.repeat(layer, 4, axis=0), 4, axis=1)
    px, py = pivot[0] * 4 + 1.5, pivot[1] * 4 + 1.5
    rad = math.radians(ang)
    c, s = math.cos(rad), math.sin(rad)
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = xx - px, yy - py
    src_x = np.rint(px + c * rx + s * ry).astype(np.int32)
    src_y = np.rint(py - s * rx + c * ry).astype(np.int32)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    out[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    return out[1::4, 1::4].copy()


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    for i in range(0, len(ty), 400):
        sl = slice(i, i + 400)
        d = (ty[sl, None].astype(np.int32) - ys[None, :].astype(np.int32)) ** 2
        d += (tx[sl, None].astype(np.int32) - xs[None, :].astype(np.int32)) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def fill_pinholes(img):
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        p = np.pad(a, 1)
        n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        hole = (~a) & (n == 4)
        if not np.any(hole):
            break
        fill = nearest_color(out, hole, a)
        out[hole] = fill[hole]
    return out


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    sizes = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        q = deque([(y, x)])
        seen[y, x] = True
        n = 0
        while q:
            cy, cx = q.popleft()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
        out[..., 3] = 0
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = a.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    mapped = palette[dist.argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv.reshape(-1)]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    out[~a] = 0
    return out


def put(out, x, y, col, only_empty=False):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        if only_empty and out[y, x, 3] >= 128:
            return
        out[y, x, :3] = col
        out[y, x, 3] = 255


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def contact_pivot(part, body):
    ys, xs = np.nonzero(part)
    if len(ys) == 0:
        return 0.0, 0.0
    touch = part & dilate(body, 1)
    if np.any(touch):
        ys, xs = np.nonzero(touch)
    by, bx = np.nonzero(body)
    bcy, bcx = float(by.mean()), float(bx.mean())
    d = (ys.astype(np.float32) - bcy) ** 2 + (xs.astype(np.float32) - bcx) ** 2
    i = int(d.argmin())
    return float(xs[i]), float(ys[i])


def top_pivot(part):
    """Hip joint of a leg: middle of its top two rows."""
    ys, xs = np.nonzero(part)
    top = ys <= int(ys.min()) + 1
    return float(np.median(xs[top])), float(ys.min())


def add_socket(host, host_mask, part_mask, canvas, n=4):
    zone = part_mask & dilate(host_mask, n)
    if not np.any(zone):
        return
    fill = nearest_color(canvas, zone, host_mask)
    write = zone & (host[..., 3] < 128)
    host[write] = fill[write]


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


# ---------------------------------------------------------------- rig
def placement(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return y0, y1, x0, x1, (CELL_W - (x1 - x0)) // 2, FOOT_Y - (y1 - y0)


def placed_still(img, flip=False):
    y0, y1, x0, x1, dx, dy = placement(img)
    out = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    out[dy:dy + (y1 - y0), dx:dx + (x1 - x0)] = img[y0:y1, x0:x1]
    return out[:, ::-1].copy() if flip else out


def pearl_mask(canvas, kind):
    h, w = canvas.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    a = canvas[..., 3] >= 128
    ys, xs = np.nonzero(a)
    # pearl box relative to the still's bbox (front x9 y15, back x7 y13)
    if kind == "front":
        bx0, by0, box = 9, 15, (52, 30, 59, 39)
    else:
        bx0, by0, box = 7, 13, (58, 34, 68, 42)
    ox, oy = int(xs.min()) - bx0, int(ys.max()) - (81 if kind == "front" else 83)
    x0, y0, x1, y1 = box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy
    inbox = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    m = (is_col(canvas, PINK) | is_col(canvas, RED) | is_col(canvas, WHITE)) & inbox
    return m


def build_rig(img, masks, kind, flip):
    y0, y1, x0, x1, dst_x, dst_y = placement(img)
    sh, sw = y1 - y0, x1 - x0
    canvas = placed_still(img)
    placed = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm
    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr
    body = placed["torso"]
    # joint cushions live on their own layer under everything, so a moved
    # part never leaves a hole and the rest pose stays the exact still
    under = np.zeros_like(canvas)
    rows = np.mgrid[0:CELL_H, 0:CELL_W][0]
    for name in PARTS:
        if np.any(placed[name]):
            pm = placed[name]
            if name.startswith("leg"):
                # only the hip band: a swung leg must not leave a stub hanging
                # down beside the crotch
                top = int(np.nonzero(pm)[0].min())
                pm = pm & (rows <= top + 13)
            add_socket(under, body, pm, canvas)
    layers["under"] = under

    pivots = {}
    for name in PARTS:
        if not np.any(placed[name]):
            pivots[name] = (0.0, 0.0)
        elif name.startswith("leg"):
            pivots[name] = top_pivot(placed[name])
        else:
            pivots[name] = contact_pivot(placed[name], body)
    if kind == "front":
        jy, jx = np.nonzero(placed["jaw"])
        pivots["jaw"] = (float(jx.max()) + 1.0, float(jy.min()))
    opy, opx = np.nonzero(canvas[..., 3] >= 128)
    ground = int(opy.max())
    feet = {}
    for name in ("leg_f", "leg_n"):
        ly, lx = np.nonzero(placed[name])
        low = ly >= int(ly.max()) - 1
        feet[name] = (float(np.median(lx[low])), float(ly.max()))
    # front faces left, back faces right
    face = -1.0 if kind == "front" else 1.0
    pearl = pearl_mask(canvas, kind)
    if flip:
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        mx = lambda p: (CELL_W - 1 - p[0], p[1])  # noqa: E731
        pivots = {k: mx(v) for k, v in pivots.items()}
        feet = {k: mx(v) for k, v in feet.items()}
        pearl = pearl[:, ::-1].copy()
        face = -face
    rear = max(feet.values(), key=lambda p: -face * p[0])
    rest_tip = claw_tip(layers["arm_f"], face)
    if kind == "front":
        draw = ["under", "plate", "tail", "leg_f", "arm_f", "torso", "leg_n", "arm_n", "jaw"]
    else:
        draw = ["under", "tail", "leg_f", "torso", "plate", "leg_n", "arm_f"]
    return {
        "kind": kind, "flip": flip, "layers": layers, "pv": pivots, "face": face,
        "ground": ground, "feet": feet, "rear": rear, "pearl": pearl, "draw": draw,
        "rest_tip": rest_tip,
    }


# ---------------------------------------------------------------- poses
def pose(action, f):
    n = FRAMES[action]
    p = dict(
        bdy=0, bdx=0, plate=0.0, tail=0.0, jaw=0.0, arm_f=0.0, arm_n=0.0,
        leg_f=0.0, leg_n=0.0, lift_f=0, lift_n=0,
        rings=0, ring_age=0, glow=0, twinkle=0, slash=0, rift=0, burst=0,
        hurt=0.0, hdx=0, flash=0, shards=0, plant=0, trail=0,
    )
    if action == "idle":
        # breathing: the upper body rises one pixel and settles, feet planted
        p["bdy"] = -int(round(max(0.0, math.sin(2 * math.pi * f / n)) * 1.0))
        p["plate"] = wave(f, n, 4.0, phase=0.5)
        p["tail"] = wave(f, n, 5.0, phase=1.3)
        p["arm_f"] = wave(f, n, 4.0, phase=0.2)
        p["arm_n"] = wave(f, n, 3.0, phase=0.9)
        p["twinkle"] = 1 if f in (9, 10) else 0
    elif action == "walk":
        ph = 2 * math.pi * 2 * f / n
        sw = math.sin(ph)
        p["leg_f"] = 14.0 * sw
        p["leg_n"] = -14.0 * sw
        swing = math.cos(ph)  # >0: the far leg travels forward
        p["lift_f"] = int(round(3 * max(0.0, swing)))
        p["lift_n"] = int(round(3 * max(0.0, -swing)))
        p["bdy"] = -1 if abs(swing) > 0.7 else 0
        # arms swing against the legs on the same side
        p["arm_f"] = -9.0 * sw
        p["arm_n"] = 7.0 * sw
        p["tail"] = wave(f, n, 7.0, cycles=2.0, phase=0.9)
        p["plate"] = wave(f, n, 4.0, cycles=2.0, phase=1.6)
    elif action == "attack":
        jaw = (8, 16, 24, 28, 28, 20, 10, 4, 4, 8, 10, 8, 5, 2, 0, 0)
        plate = (3, 7, 11, 14, 15, 12, 8, 6, 6, 8, 9, 7, 5, 3, 1, 0)
        tail = (2, 5, 8, 10, 10, 8, 4, 2, 2, 4, 5, 4, 3, 1, 0, 0)
        # claw arm: wind up (tip high), then cut down through the hit
        arm = (3, 9, 16, 22, 26, 10, -20, -32, -32, -30, -28, -24, -16, -8, -3, 0)
        arm_n = (1, 3, 6, 8, 10, 4, -8, -12, -12, -10, -9, -7, -4, -2, -1, 0)
        bdy = (0, 0, -1, -1, -1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
        # lean back for the roar, lunge forward into the cut (feet stay planted)
        bdx = (0, 0, -1, -1, -1, 1, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0)
        p["jaw"], p["plate"], p["tail"] = jaw[f], plate[f], tail[f]
        p["arm_f"], p["arm_n"] = arm[f], arm_n[f]
        p["bdy"], p["bdx"] = bdy[f], bdx[f]
        p["plant"] = 1
        if 1 <= f <= 5:
            p["rings"] = 3 if f >= 2 else 2
            p["ring_age"] = f - 1
        p["glow"] = 1 if 2 <= f <= 11 else 0
        p["trail"] = {6: (arm[4], arm[6]), 7: (arm[4], arm[7]), 8: (-10, -32)}.get(f, 0)
        p["slash"] = f - 5 if 6 <= f <= 8 else 0
        p["rift"] = f - 8 if 9 <= f <= 11 else 0
        p["burst"] = f - 10 if 11 <= f <= 13 else 0
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 18.0 * env
        p["hdx"] = int(round(3.0 * env))
        p["jaw"] = 14.0 * env
        p["plate"] = -6.0 * env
        p["tail"] = 8.0 * env
        p["arm_f"] = 12.0 * env
        p["arm_n"] = 8.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["shards"] = f if 2 <= f <= 6 else 0
    return p


# ---------------------------------------------------------------- effects
def mouth_point(rig, bdx, bdy):
    """Front: the jaw tip. Back: the head tip (top of the crest)."""
    face = rig["face"]
    if rig["kind"] == "front":
        lyr = shift(rig["layers"]["jaw"], bdx, bdy)
    else:
        lyr = shift(rig["layers"]["torso"], bdx, bdy)
        a = lyr[..., 3] >= 128
        ys, xs = np.nonzero(a)
        cut = int(ys.min()) + 14
        lyr = lyr.copy()
        lyr[cut:] = 0
    ys, xs = np.nonzero(lyr[..., 3] >= 128)
    if len(xs) == 0:
        return None
    x = int(xs.max()) if face > 0 else int(xs.min())
    y = int(np.median(ys[xs == x]))
    return x + int(face) * 2, y + 1


def roar_rings(out, mouth, face, age, strength):
    mx, my = mouth
    cols = (WHITE, VIOLET, PINK)
    for r_i in range(strength):
        rad = 5 + age * 3 + r_i * 4
        col = cols[min(r_i, 2)]
        for deg in range(-54, 55, 5):
            ang = math.radians(deg)
            put(out, mx + face * rad * math.cos(ang), my + rad * math.sin(ang) * 0.8, col, True)
            if r_i == 0 and abs(deg) <= 35:
                put(out, mx + face * (rad + 1) * math.cos(ang), my + (rad + 1) * math.sin(ang) * 0.8, PALE, True)


def slash_arcs(out, tip, face, k):
    """Three crescent cuts sweeping down in front of the claw."""
    tx, ty = tip
    for i, col in enumerate((VIOLET, WHITE, PINK)):
        r = 10 + i * 3 + k * 2
        for deg in range(-80, 50, 5):
            ang = math.radians(deg)
            x = tx + face * (2 + r * math.cos(ang) * 0.75)
            y = ty - 4 + r * math.sin(ang)
            put(out, x, y, col, True)


def claw_trail(out, pivot, rest_tip, face, a_from, a_to, k):
    """A thick crescent along the arc the claw tip really swept."""
    px, py = pivot
    vx, vy = rest_tip[0] - px, rest_tip[1] - py
    length = math.hypot(vx, vy)
    if length < 4:
        return
    ux, uy = vx / length, vy / length
    lo, hi = sorted((a_from, a_to))
    steps = max(2, int(abs(hi - lo) / 1.5))
    bands = ((-3, VIOLET), (-2, PINK), (-1, PALE), (0, WHITE), (1, WHITE), (2, PALE), (3, PINK), (4, VIOLET))
    if k >= 3:
        bands = ((0, VIOLET), (1, PINK), (2, VIOLET))
    for i in range(steps + 1):
        a = lo + (hi - lo) * i / steps
        th = math.radians(-face * a)
        c, s = math.cos(th), math.sin(th)
        dx, dy = c * ux - s * uy, s * ux + c * uy
        # the crescent is thickest in the middle of the sweep
        fat = math.sin(math.pi * i / steps)
        for off, col in bands:
            if abs(off - 0.5) > 1.0 + 3.0 * fat:
                continue
            r = length + 3 + off
            put(out, px + dx * r, py + dy * r, col, True)


def diamond(out, cx, cy, r, col, edge):
    for dy in range(-r, r + 1):
        span = r - abs(dy)
        for dx in range(-span, span + 1):
            c = edge if abs(dx) == span else col
            put(out, cx + dx, cy + dy, c, True)


ZIG = (0, 1, 2, 1, 0, -1, -2, -1)


def spatial_rift(out, center, size, face):
    """A jagged tear in space: ink core, violet lips, pink edge, pearl motes."""
    cx, cy = center
    half = 9 + size * 6
    open_w = 1 + size  # how far the lips part
    for t in range(-half, half + 1):
        z = ZIG[(t + 64) % len(ZIG)]
        taper = 1.0 - abs(t) / (half + 1)
        wid = max(0, int(round(open_w * taper)))
        x = cx + z
        y = cy + t
        for k in range(-wid, wid + 1):
            put(out, x + k, y, INK, True)
        put(out, x - wid - 1, y, VIOLET, True)
        put(out, x + wid + 1, y, VIOLET, True)
        if abs(t) % 3 == 0:
            put(out, x - wid - 2, y, PINK, True)
            put(out, x + wid + 2, y, PINK, True)
    for k in range(4 + size * 2):
        ang = k * 2.4 + size
        d = 6 + size * 3 + (k % 3) * 2
        put(out, cx + d * math.cos(ang) * 0.6, cy + d * math.sin(ang), (WHITE, PINK, VIOLET)[k % 3], True)


def rift_burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4 + math.pi / 8
        for r in range(4, 4 + size * 4, 2):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang) * 0.85, (VIOLET, WHITE)[r % 2], True)
    for k in range(6):
        ang = k * 2 * math.pi / 6 + size * 0.7
        d = 8 + size * 5
        diamond(out, cx + d * math.cos(ang), cy + d * math.sin(ang) * 0.7, 2 if size < 3 else 1, PALE, VIOLET)


def shards(out, frame_idx, face, box):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    d = frame_idx
    pts = ((cx + face * (6 + d * 2), y0 + 8 - d), (cx - face * (4 + d), y0 - 2 - d),
           (cx + face * 2, y0 + 16 - d))
    for sx, sy in pts:
        diamond(out, sx, sy, 2 if d % 2 else 1, PALE, VIOLET)


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    ink = is_col(out, INK)
    out[a & ~ink, :3] = WHITE
    return out


# ---------------------------------------------------------------- synth
def part_moved(layer, pivot, ang, dx, dy):
    lyr = layer
    if abs(ang) >= 0.45:
        lyr = fill_pinholes(rotate_nearest(layer, ang, pivot))
    return shift(lyr, dx, dy)


def claw_tip(layer, face):
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    if len(xs) == 0:
        return None
    x = int(xs.max()) if face > 0 else int(xs.min())
    sel = xs == x
    return float(x), float(np.median(ys[sel]))


def assemble_body(rig, p):
    L, pv, face = rig["layers"], rig["pv"], rig["face"]
    bdx, bdy = int(p["bdx"]) * int(face), int(p["bdy"])
    fwd = -face  # positive angle moves a hanging foot toward -x
    ang = {
        "plate": face * p["plate"],        # plate tip rises
        "tail": face * p["tail"],          # tail tip rises
        "jaw": face * p["jaw"],            # jaw tip drops
        "arm_f": -face * p["arm_f"],       # claw tip rises
        "arm_n": -face * p["arm_n"],
    }
    moved = {}
    for name in ("under", "torso", "plate", "tail", "jaw", "arm_f", "arm_n"):
        if name in ("torso", "under"):
            moved[name] = shift(L[name], bdx, bdy)
        else:
            jdy = (1 if p["jaw"] >= 18 else 0) if name == "jaw" else 0
            moved[name] = part_moved(L[name], pv[name], ang[name], bdx, bdy + jdy)
    for leg, key in (("leg_f", "lift_f"), ("leg_n", "lift_n")):
        # legs follow the body sideways, stay on the ground vertically.
        # With plant, the hip moves with the body and the leg leans back so
        # the foot stays where it was.
        deg = p[leg]
        if p.get("plant") and p["bdx"]:
            hip_y = pv[leg][1]
            deg -= math.degrees(math.atan2(float(p["bdx"]), max(8.0, rig["ground"] - hip_y)))
        moved[leg] = part_moved(L[leg], pv[leg], fwd * deg, bdx, -int(p[key]))
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in rig["draw"]:
        if name == "jaw" and rig["kind"] == "front" and abs(p["jaw"]) >= 0.45:
            # mouth interior: the hole the dropped jaw leaves is ink
            home = shift(L["jaw"], bdx, bdy)
            hole = (home[..., 3] >= 128) & (moved["jaw"][..., 3] < 128)
            img[hole, :3] = INK
            img[hole, 3] = 255
            jm = moved["jaw"][..., 3] >= 128
            gap = dilate(home[..., 3] >= 128, 1) & dilate(jm, 1) & (img[..., 3] < 128) & ~jm
            img[gap, :3] = INK
            img[gap, 3] = 255
        paste(img, moved[name])
    if p["glow"] or p["twinkle"]:
        pearl = rig["pearl"]
        if np.any(pearl):
            g = shift(np.dstack([pearl.astype(np.uint8)] * 4), bdx, bdy)[..., 0] > 0
            gy, gx = np.nonzero(g)
            yy = np.mgrid[0:CELL_H, 0:CELL_W][0]
            if p["glow"]:
                cy = gy.mean()
                red = g & is_col(img, RED)
                img[red, :3] = PINK
                img[g & (yy <= cy), :3] = WHITE
            else:
                top = int(gy.min())
                img[g & (yy == top + 1), :3] = WHITE
    return img, claw_tip(moved["arm_f"], face)


def synth(rig, action, f, palette, effects=True):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)
    face = rig["face"]
    base, tip = assemble_body(rig, p)
    if p["hurt"] > 0.45 or p["hdx"]:
        # head goes up and back: a point on the facing side rises
        img = rotate_nearest(base, -face * p["hurt"], rig["rear"])
        img = shift(img, -face * p["hdx"], 0)
        img = despeckle(fill_pinholes(img))
    else:
        img = base
    img = snap(img, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["rings"]:
        m = mouth_point(rig, int(p["bdx"]) * int(face), int(p["bdy"]))
        if m:
            roar_rings(img, m, face, p["ring_age"], p["rings"])
    if tip is not None:
        if p["trail"]:
            pv_ = rig["pv"]["arm_f"]
            bdx_ = int(p["bdx"]) * int(face)
            a0, a1 = p["trail"]
            claw_trail(img, (pv_[0] + bdx_, pv_[1] + int(p["bdy"])), rig["rest_tip"], face, a0, a1, p["slash"])
        rc = (tip[0] + face * 14, tip[1] - 8)
        if p["rift"]:
            spatial_rift(img, (int(rc[0]), int(rc[1])), p["rift"], face)
        if p["burst"]:
            rift_burst(img, (int(rc[0]), int(rc[1])), p["burst"])
    if p["shards"]:
        bb = body_box(img)
        if bb:
            shards(img, p["shards"], face, bb)
    return snap(img, palette)


# ---------------------------------------------------------------- output
def params_hash():
    src = (Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes()
           + (SRC_DIR / "front.png").read_bytes() + (SRC_DIR / "back.png").read_bytes())
    return hashlib.sha1(src).hexdigest()[:12]


def stack_sheets(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png").convert("RGBA"))
                sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = tile
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
    Image.fromarray(master, "RGBA").save(OUT_DIR / "all_8dir_actions.png")
    return master


def preview_image(sheets):
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 2, 4, 6, 8, 10, 12, 14),
        "attack": (0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15),
        "hurt": (0, 2, 4, 6, 8, 11),
    }
    cols = max(len(v) for v in samples.values())
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            m = tile[..., 3] >= 128
            bg[ai * CELL_H:(ai + 1) * CELL_H, si * CELL_W:(si + 1) * CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    """Every frame of every action, cropped, for S (front) and N (back)."""
    pad = 3
    rows = []
    for di in (0, 4):
        for act in ACTIONS:
            row = []
            band = sheets[act][di * CELL_H:(di + 1) * CELL_H]
            for f in range(FRAMES[act]):
                tile = band[:, f * CELL_W:(f + 1) * CELL_W]
                a = tile[..., 3] >= 128
                if not np.any(a):
                    continue
                ys, xs = np.nonzero(a)
                y0, y1 = max(0, int(ys.min()) - pad), min(CELL_H, int(ys.max()) + 1 + pad)
                x0, x1 = max(0, int(xs.min()) - pad), min(CELL_W, int(xs.max()) + 1 + pad)
                row.append(tile[y0:y1, x0:x1])
            rows.append(row)
    ch = max(c.shape[0] for r in rows for c in r)
    cw = max(c.shape[1] for r in rows for c in r)
    ncol = max(len(r) for r in rows)
    bg = np.full((len(rows) * ch, ncol * cw, 4), (15, 20, 31, 255), np.uint8)
    for ri, r in enumerate(rows):
        for ci, c in enumerate(r):
            y = ri * ch + (ch - c.shape[0])
            x = ci * cw + (cw - c.shape[1]) // 2
            m = c[..., 3] >= 128
            bg[y:y + c.shape[0], x:x + c.shape[1]][m] = c[m]
    return Image.fromarray(bg, "RGBA")


SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "all_8dir_actions.png",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    return n


def body_pieces(img):
    return [s for s in components(img[..., 3] >= 128) if s > 6]


def diff_count(a_img, b_img):
    d = np.any(a_img != b_img, axis=2) & ((a_img[..., 3] >= 128) | (b_img[..., 3] >= 128))
    return int(d.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    stills = {k: np.array(Image.open(SRC_DIR / f"{k}.png").convert("RGBA")) for k in ("front", "back")}
    for k, im in stills.items():
        if im.shape[:2] != (96, 96):
            raise SystemExit(f"expected 96x96 {k}.png, got {im.shape}")
    palettes = {k: extract_palette(im) for k, im in stills.items()}
    masks = {k: segment(im, k) for k, im in stills.items()}
    counts = {k: {n: int(m.sum()) for n, m in v.items()} for k, v in masks.items()}
    print("parts", json.dumps(counts), flush=True)
    yy, xx = np.mgrid[0:96, 0:96]
    eye = is_col(stills["front"], EYE) & (xx <= 30)
    ey, ex = np.nonzero(eye)
    jy, jx = np.nonzero(masks["front"]["jaw"])
    print("facing check: jaw tip x", int(jx.min()), "eye x", float(ex.mean()), flush=True)
    if not jx.min() < ex.mean():
        raise SystemExit("front.png is expected to face left")

    rigs = {d: build_rig(stills[VIEW[d][0]], masks[VIEW[d][0]], VIEW[d][0], VIEW[d][1]) for d in DIRS}
    for d in ("W", "N"):
        print(d, "pivots", {k: tuple(round(v, 1) for v in p) for k, p in rigs[d]["pv"].items()},
              "rear", rigs[d]["rear"], "pearl", int(rigs[d]["pearl"].sum()), flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)

    bad = {}
    for d in DIRS:
        kind, flip = VIEW[d]
        ref = placed_still(stills[kind], flip)
        n = diff_count(synth(rigs[d], "idle", 0, palettes[kind]), ref)
        if n:
            bad[d] = n
    print("still diff", bad if bad else 0, flush=True)
    if bad:
        raise SystemExit("idle frame 0 does not match the still")

    broken = []
    for d in ("S", "N", "E"):
        kind = VIEW[d][0]
        for act in ACTIONS:
            for f in range(FRAMES[act]):
                fr = synth(rigs[d], act, f, palettes[kind], effects=False)
                pcs = body_pieces(fr)
                if len(pcs) != 1:
                    broken.append((d, act, f, pcs[:4]))
    print("piece check", broken if broken else "ok", flush=True)

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in range(12)]
        want += [("walk", "N", f) for f in (0, 2, 4, 6)]
        want += [("attack", "N", f) for f in (2, 4, 6, 7, 9, 11)]
        want += [("idle", d, 0) for d in DIRS]
        tiles = [synth(rigs[d], act, f, palettes[VIEW[d][0]]) for act, d, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        print("samples", work / "samples.png", flush=True)
        return
    if broken:
        raise SystemExit("a pose broke the body")

    frames_dir = work / "frames"
    state_p = work / "state.json"
    digest = params_hash()
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    if args.force or state.get("hash") != digest:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        state = {"hash": digest, "done": []}
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = len(DIRS) * sum(FRAMES.values())
    done = 0
    for act in ACTIONS:
        for d in DIRS:
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                done += 1
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palettes[VIEW[d][0]]), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"] = sorted(set(state.get("done", [])) | {f"{act}/{d}"})
            state_p.write_text(json.dumps(state))
            print(f"[{done}/{total}] {act}/{d}", flush=True)

    sheets = stack_sheets(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    (SRC_DIR / "actions").mkdir(exist_ok=True)
    preview.save(SRC_DIR / "actions" / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1)
    bounds = []
    for di in range(8):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        ys, xs = np.nonzero(t[..., 3] >= 128)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    union = np.unique(np.concatenate(list(palettes.values())), axis=0)
    pal = {tuple(int(v) for v in c) for c in union}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    off_view = 0
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            band = sheets[act][di * CELL_H:(di + 1) * CELL_H]
            pv = {tuple(int(v) for v in c) for c in palettes[VIEW[d][0]]}
            u = {tuple(int(v) for v in c) for c in np.unique(band[band[..., 3] >= 128][:, :3], axis=0)}
            off_view += len(u - pv)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    still_bad = 0
    for di, d in enumerate(DIRS):
        kind, flip = VIEW[d]
        still_bad += diff_count(sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W], placed_still(stills[kind], flip))
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "off_view_palette": off_view, "alphas": alphas,
        "edge_touching": sum(len(v) for v in clip.values()), "still_diff": still_bad,
        "bounds_s": bounds, "views": {d: list(VIEW[d]) for d in DIRS},
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
            else:
                report[f"gait_{act}"] = (r.stdout + r.stderr).strip()[-300:]
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False), flush=True)

    meta = {
        "generator": "forge_palkia_actions", "version": VERSION, "key": KEY,
        "dex": 484, "form": "normal", "variant": "palkia",
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "views": {d: {"still": VIEW[d][0], "mirrored": VIEW[d][1]} for d in DIRS},
        "archetype": "biped_jaw_slasher (upright space dragon, claw arms, shoulder plate, curled tail)",
        "off_palette_colors": len(off), "source": "front.png + back.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = bool(off or off_view or clip or still_bad or alphas not in ([0, 255], [0]))
    if failed:
        raise SystemExit("qa failed")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
