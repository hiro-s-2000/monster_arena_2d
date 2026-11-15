#!/usr/bin/env python3
"""Probopass (0476, normal) 8-direction action sheet from this folder's 96x96 front.png.

Measured on this still only (bbox x17-77 y17-78, 15 colours, alpha 0/255):
- hat      : red cap with its ink rim, y17-33 (x29-66).
- eyes     : two white/navy eyes under the hat brim, y34-38 (left x35-39, right x46-51).
- nose     : the big red wedge, y37-63, tip at the lower left (x17-20, y56-59).
- mustache : dark iron-filing beard, y63-78 x17-49. It is what the body rests on.
- mini_l   : left mini-nose, a separate blue unit behind the nose, x17-35 y40-53.
- mini_r   : right mini-nose, light-blue block on the body side, x61-77 y39-61.
- body     : the blue stone head (eyes included), the rest.
No legs, no mouth are drawn. The nose tip is left of the eyes, so the still faces
LEFT: N NW W SW keep the still, S SE E NE are mirrored (E faces screen right).

idle  : the head/hat/nose rise 1px (mustache stays on the ground), mini-noses float,
        filings ripple. Frame 0 and the loop end equal the still.
walk  : magnetic glide. Filing strands ripple in a travelling wave, head bobs,
        mini-noses trail. No feet.
attack: 0-4 roar without a mouth: hat pops, filings bristle, right mini-nose pops,
        nose rears back, magnetic rings. 5-10 the left mini-nose spins a full orbit
        round the body (behind it on the far side) while the nose drives forward,
        11-12 nose ram + mini-nose shot + impact with rock chips, 13-15 recover.
        The body itself is never rotated flat.
hurt  : 12 frames, white flash, one-piece recoil (max 24 deg) and recover.

Palette-locked. Alpha is only 0 or 255. front/back png and gif are not written.
Run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_probopass_actions.py [--samples|--force|--no-sync|--skip-qa]
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
STILL_DIR = ROOT / "monsterSprite/gen4/0476_probopass/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0476_probopass/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
BOUNDS = ROOT / "data/sprite_bounds.js"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "probopass_v1"
KEY = "0476_probopass"
FLIP_DIRS = {"S", "SE", "E", "NE"}   # the still faces left
MAX_RECOIL = 24.0

INK = (16, 16, 16)
RED = (230, 90, 98)
RED_D = (189, 49, 82)
RED_L = (238, 123, 139)
BROWN = (123, 49, 41)
BROWN_D = (90, 41, 32)
BLUE = (90, 123, 205)
BLUE_D = (57, 98, 172)
BLUE_L = (131, 172, 246)
BLUE_M = (115, 156, 230)
NAVY = (24, 74, 131)
IRON = (41, 57, 74)
IRON_L = (57, 74, 90)
PALE = (197, 205, 222)
WHITE = (255, 255, 255)
REDS = (RED, RED_D, RED_L, BROWN, BROWN_D)
BLUES = (BLUE, BLUE_D, BLUE_L, BLUE_M, NAVY)

# attack timeline (frame 0 = still)
A_HAT = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
A_BRISTLE = [0, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0, 0]
A_MINIR = [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
A_NOSE = [0, -4, -8, -10, -10, -6, 0, 4, 8, 12, 14, 14, 10, 6, 2, 0]
A_NOSE_DX = [0, 0, -1, -1, -1, 0, 0, 1, 1, 2, 2, 2, 1, 1, 0, 0]
A_DX = [0, 0, -1, -2, -2, -1, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
A_RING = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
A_BURST = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0]
# back mini-nose orbit: angle (units of pi, 0 = its home on the back side, 1 = front,
# rising over the top = far side) and how far it has left home (0..1).
A_THETA = [0, 0, 0, 0.3, 0.7, 1.1, 1.5, 1.9, 2.3, 2.65, 3.0, 3.0, 3.0, 3.35, 3.7, 4.0]
A_ORBW = [0, 0, 0, 0.5, 0.85, 1, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.45, 0]
# at the front it is shot forward into the target
A_SHOT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 5, 0, 0, 0]
HURT = [0.0, 0.45, 0.80, 1.0, 0.82, 0.60, 0.40, 0.24, 0.12, 0.04, 0.0, 0.0]

SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)


# ---------------------------------------------------------------- helpers
def is_color(img, color):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(color, np.uint8), axis=-1)


def any_color(img, colors):
    m = np.zeros(img.shape[:2], dtype=bool)
    for c in colors:
        m |= is_color(img, c)
    return m


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
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
        comp = np.zeros_like(mask)
        for y, x in pts:
            comp[y, x] = True
        out.append(comp)
    return out


def largest_component(mask):
    comps = components(mask)
    if not comps:
        return mask
    comps.sort(key=lambda m: int(m.sum()), reverse=True)
    return comps[0]


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1, constant_values=False)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


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
    """4x nearest in-plane rotation (positive = clockwise on screen). No squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    rad = math.radians(ang)
    c, s = math.cos(rad), math.sin(rad)
    scale = 4
    up = np.repeat(np.repeat(layer, scale, axis=0), scale, axis=1)
    px, py = pivot[0] * scale + 1.5, pivot[1] * scale + 1.5
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = xx - px, yy - py
    # inverse map of a clockwise (screen) rotation
    src_x = np.rint(px + c * rx + s * ry).astype(np.int32)
    src_y = np.rint(py - s * rx + c * ry).astype(np.int32)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    out[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    return out[scale // 2::scale, scale // 2::scale].copy()


def despeckle(img):
    """Drop only pixels whose four neighbours are all empty. After a warp only."""
    out = img.copy()
    a = out[..., 3] >= 128
    pad = np.pad(a, 1, constant_values=False)
    n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_cracks(img):
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        pad = np.pad(a, 1, constant_values=False)
        n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        hole = (~a) & (n >= 3)
        if not np.any(hole):
            break
        h, w = a.shape
        for y, x in zip(*[v.tolist() for v in np.nonzero(hole)]):
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and out[ny, nx, 3] >= 128:
                    out[y, x] = out[ny, nx]
                    break
        out[..., 3] = np.where(out[..., 3] >= 128, 255, 0)
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Exactly 0 on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


# ---------------------------------------------------------------- segmentation
def segment(img):
    """Masks that partition every opaque pixel of the 96x96 still."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    ink = is_color(img, INK)
    reds = any_color(img, REDS)
    iron = any_color(img, (IRON, IRON_L))
    nonred = op & ~reds

    hat = op & (yy <= 33)

    # Mustache: iron filings below the nose, left of the body outline (x<=49).
    mustache = op & (yy >= 63) & (xx <= 49) & (iron | ink)
    mustache |= op & (yy >= 60) & (yy <= 62) & (xx <= 49) & iron

    # Left mini-nose: non-red unit left of the nose, above the beard.
    mini_l = largest_component(nonred & (xx <= 35) & (yy >= 40) & (yy <= 53) & ~mustache)

    # Right mini-nose: the light-blue block on the right side of the body.
    mini_r = op & (xx >= 61) & (yy >= 39) & (yy <= 61) & ~reds

    # Nose: red pixels under the hat, plus the ink outline that only touches the nose.
    nose = reds & (yy >= 36) & ~hat
    nose = largest_component(nose)
    rim = ink & (yy >= 40) & (yy <= 63) & (xx <= 34) & dilate(nose, 1) & ~mustache & ~mini_l
    nose |= rim

    parts = {"hat": hat, "mustache": mustache, "mini_l": mini_l, "mini_r": mini_r, "nose": nose}
    used = np.zeros(op.shape, dtype=bool)
    for name, mask in parts.items():
        if int(mask.sum()) < 20:
            raise SystemExit(f"{name} collapsed: {int(mask.sum())}")
        if int((mask & used).sum()):
            raise SystemExit(f"{name} overlaps another part")
        used |= mask
    body = op & ~used
    parts["body"] = body
    eyes = any_color(img, (WHITE, PALE)) & (yy >= 33) & (yy <= 39)
    if int((eyes & ~body).sum()):
        raise SystemExit("the eyes were cut off the body")
    return parts


def measure(img, masks):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    yy, xx = np.mgrid[0:96, 0:96]
    eyes = any_color(img, (WHITE, PALE)) & (yy >= 33) & (yy <= 39)
    eys, exs = np.nonzero(eyes)
    nys, nxs = np.nonzero(masks["nose"])
    tip_y = int(nys.max())
    return {
        "opaque": int(op.sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "parts": {k: int(v.sum()) for k, v in masks.items()},
        "eyes": [int(exs.min()), int(eys.min()), int(exs.max()), int(eys.max())],
        "nose_tip_x": int(nxs.min()),
        "nose_bottom_y": tip_y,
        "facing": "left" if int(nxs.min()) < int(exs.min()) else "right",
        "feet": 0,
        "mouth": 0,
    }


# ---------------------------------------------------------------- rig
def build_rig(img, masks, flip):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    # 6px toward the back (the still faces left) so the ram and the shot have room.
    dst_x = (CELL_W - sw) // 2 + 6
    dst_y = FOOT_Y - (sh - 1)
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    placed = {}
    for name, mask in masks.items():
        pm = np.zeros((CELL_H, CELL_W), dtype=bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = mask[y0:y1, x0:x1]
        placed[name] = pm
    if flip:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}

    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr

    body_m = placed["body"]
    # Under-face: where the nose sits, the stone face continues (dark blue, ink rim on
    # the silhouette). Only seen when the nose rears or rams. Covered at rest.
    under = np.zeros_like(canvas)
    nose_m = placed["nose"]
    sil = canvas[..., 3] >= 128
    pad = np.pad(sil, 1, constant_values=False)
    edge = sil & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    under[nose_m, :3] = BLUE_D
    under[nose_m & dilate(edge, 1), :3] = INK
    under[nose_m, 3] = 255
    # Socket under the back mini-nose: dark stone with an ink rim. Covered at rest,
    # seen only while the mini-nose is popped out or flying its orbit.
    body_layer = layers["body"].copy()
    rm = placed["mini_r"]
    rpad = np.pad(rm, 1, constant_values=False)
    rim_r = rm & ~(rpad[:-2, 1:-1] & rpad[2:, 1:-1] & rpad[1:-1, :-2] & rpad[1:-1, 2:])
    body_layer[rm, :3] = NAVY
    body_layer[rm & dilate(rim_r, 1) & ~dilate(~rm & ~sil, 1), :3] = BLUE_D
    body_layer[rm & dilate(~rm & ~sil, 1), :3] = INK
    body_layer[rm, 3] = 255
    # Hat cushion: its bottom three rows are also kept on the head.
    hm = placed["hat"]
    hys = np.nonzero(hm)[0]
    hat_cush = np.zeros_like(canvas)
    low = hm & (np.arange(CELL_H)[:, None] >= int(hys.max()) - 2)
    hat_cush[low] = canvas[low]
    # Mustache cushion: each strand column extends 3px up under the nose/body.
    must = layers["mustache"].copy()
    mm = placed["mustache"]
    for x in np.nonzero(mm.any(axis=0))[0].tolist():
        top = int(np.nonzero(mm[:, x])[0].min())
        for k in range(1, 4):
            y = top - k
            if y >= 0 and sil[y, x] and not mm[y, x]:
                must[y, x] = canvas[top, x]

    oys, oxs = np.nonzero(sil)
    center = (float(oxs.mean()), float(oys.mean()))
    facing = 1.0 if flip else -1.0      # +1: forward is +x
    nys, nxs = np.nonzero(nose_m)
    # nose pivot: the top of the wedge (under the eyes)
    top = nys <= nys.min() + 3
    nose_pivot = (float(nxs[top].mean()), float(nys.min()) + 2.0)
    tip_i = int(np.argmax(-nxs * facing + 0 * nys)) if False else None
    if facing < 0:
        tip_x = float(nxs.min())
    else:
        tip_x = float(nxs.max())
    tip_y = float(nys[nxs == (nxs.min() if facing < 0 else nxs.max())].mean())
    rys, rxs = np.nonzero(placed["mini_r"])
    mini_home = (float(rxs.mean()), float(rys.mean()))
    bys, bxs = np.nonzero(body_m | placed["mini_r"])
    return {
        "layers": layers, "body": body_layer, "under": under, "hat_cush": hat_cush,
        "must": must, "placed": placed, "canvas": canvas, "center": center,
        "body_center": (float(oxs.mean()), float(oys.mean()) - 4),
        "facing": facing, "nose_pivot": nose_pivot, "tip": (tip_x, tip_y),
        "mini_home": mini_home, "foot": float(oys.max()),
    }


def pose(action, f):
    n = FRAMES[action]
    p = dict(dx=0, dy=0, head_dy=0, hat=0, bristle=0, ripple=0.0, ripple_ph=0.0,
             minil=(0, 0), minir=0, nose=0.0, nose_dx=0, theta=0.0, orbw=0.0, shot=0,
             recoil=0.0, ring=0, burst=0, flash=0, chips=0, sparks=0, lines=0)
    if action == "idle":
        p["head_dy"] = int(round(wave(f, n, -1.0)))
        p["hat"] = 1 if wave(f, n, 1.0, phase=0.0) > 0.85 else 0
        p["minil"] = (0, int(round(wave(f, n, -1.0, phase=0.8))))
        p["minir"] = 1 if wave(f, n, 1.0, phase=2.0) > 0.6 else 0
        p["ripple"] = 1.4
        p["ripple_ph"] = 2 * math.pi * f / n
    elif action == "walk":
        p["head_dy"] = int(round(wave(f, n, -1.6, cycles=2.0)))
        p["dx"] = int(round(wave(f, n, 1.4, cycles=1.0)))
        p["minil"] = (int(round(wave(f, n, 2.0, cycles=2.0, phase=0.6))),
                      int(round(wave(f, n, -2.0, cycles=2.0, phase=1.4))))
        p["minir"] = 1 if wave(f, n, 1.0, cycles=2.0, phase=0.9) > 0.5 else 0
        p["ripple"] = 2.4
        p["ripple_ph"] = 2 * math.pi * 2 * f / n
        p["lines"] = 1 if f else 0
        p["nose"] = wave(f, n, 2.5, cycles=2.0, phase=0.3)
    elif action == "attack":
        p["hat"] = A_HAT[f]
        p["bristle"] = A_BRISTLE[f]
        p["minir"] = A_MINIR[f]
        p["nose"] = float(A_NOSE[f])
        p["nose_dx"] = A_NOSE_DX[f]
        p["dx"] = A_DX[f]
        p["ring"] = A_RING[f]
        p["burst"] = A_BURST[f]
        p["theta"] = A_THETA[f]
        p["orbw"] = A_ORBW[f]
        p["shot"] = A_SHOT[f]
        p["chips"] = 1 if f in (11, 12, 13) else 0
        p["sparks"] = 1 if 5 <= f <= 10 else 0
        if 1 <= f <= 4:
            p["minil"] = (-[0, 0, 1, 1, 1][f], -[0, 1, 1, 2, 1][f])  # rears back and up
    else:
        env = HURT[f]
        p["recoil"] = MAX_RECOIL * env
        p["dx"] = int(round(-4 * env))
        p["dy"] = int(round(-2 * env))
        p["hat"] = 2 if f in (2, 3, 4) else (1 if f in (1, 5) else 0)
        p["bristle"] = 2 if 1 <= f <= 4 else (1 if f in (5, 6) else 0)
        p["minir"] = 2 if 2 <= f <= 5 else 0
        p["minil"] = (int(round(-3 * env)), int(round(-3 * env)))
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 5 else 0
    return p


def strands(must, mask_cols, ripple, phase, bristle, facing):
    """Iron filings: each 2-px strand lifts by its own integer amount."""
    if ripple <= 0 and bristle <= 0:
        return must
    out = np.zeros_like(must)
    cols = np.nonzero((must[..., 3] >= 128).any(axis=0))[0]
    if len(cols) == 0:
        return must
    c0 = int(cols.min())
    for x in cols.tolist():
        k = (x - c0) // 2
        lift = 0
        if ripple > 0:
            # travelling wave toward the back (the body glides forward)
            kk = facing * k * 0.9
            v = (math.sin(phase + kk) - math.sin(kk)) / 2.0
            lift = int(round(max(0.0, v) * ripple))
        if bristle > 0:
            lift = max(lift, (bristle if k % 2 == 0 else bristle - 1) + (1 if k % 3 == 1 else 0))
        col = must[:, x:x + 1]
        out[:, x:x + 1] = shift(col, 0, -lift)
    return out


def orbit_pos(rig, theta, w):
    """Back mini-nose orbit. theta in units of pi. Returns offset from home and far flag."""
    hx, hy = rig["mini_home"]
    cx, cy = rig["body_center"]
    b = -rig["facing"]                # back side
    t = math.pi * theta
    ex = cx + b * 42 * math.cos(t)
    ey = cy - 16 * math.sin(t)
    x = hx + (ex - hx) * w
    y = hy + (ey - hy) * w
    far = w > 0.3 and math.sin(t) > 0.2
    return x - hx, y - hy, far


def compose(rig, p):
    L = rig["layers"]
    fac = rig["facing"]
    bdx = p["dx"] * fac
    bdy = p["dy"]
    hdy = bdy + p["head_dy"]
    out = np.zeros_like(rig["canvas"])

    # back mini-nose: pop, orbit round the body, shot forward
    odx, ody, far = 0.0, 0.0, False
    if p["orbw"] > 0:
        odx, ody, far = orbit_pos(rig, p["theta"], p["orbw"])
    odx += p["shot"] * fac - fac * p["minir"]
    mini = shift(L["mini_r"], bdx + odx, hdy + ody)
    # front mini-nose sliver floats on its own
    mini_l = shift(L["mini_l"], bdx + p["minil"][0] * fac, hdy + p["minil"][1])
    paste(out, mini_l)

    if far:
        paste(out, mini)
    must = strands(rig["must"], None, p["ripple"], p["ripple_ph"], p["bristle"], fac)
    paste(out, shift(must, bdx, bdy))
    paste(out, shift(rig["under"], bdx, hdy))
    paste(out, shift(rig["body"], bdx, hdy))
    paste(out, shift(rig["hat_cush"], bdx, hdy))
    paste(out, shift(L["hat"], bdx, hdy - p["hat"]))
    nose = L["nose"]
    if abs(p["nose"]) >= 0.45:
        # positive = tip swings forward/up. On a left-facing still that is clockwise.
        ang = p["nose"] * (1.0 if fac < 0 else -1.0)
        nose = despeckle(rotate_nearest(nose, ang, rig["nose_pivot"]))
    paste(out, shift(nose, bdx + p["nose_dx"] * fac, hdy))
    if not far:
        paste(out, mini)
    return out


# ---------------------------------------------------------------- palette and fx
def extract_palette(img):
    return np.unique(img[img[..., 3] >= 128][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    opaque = out[..., 3] >= 128
    if not np.any(opaque):
        out[...] = 0
        return out
    flat = out[..., :3].reshape(-1, 3).copy()
    sel = opaque.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    flat[sel] = palette[dist.argmin(axis=1)][inv.reshape(-1)]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)
    out[~opaque] = 0
    return out


def put(out, x, y, col, over=False):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and (over or out[y, x, 3] < 128):
        out[y, x, :3] = col
        out[y, x, 3] = 255


def ring(out, center, rx, ry, count, colors, gap=0.0):
    cx, cy = center
    for i in range(count):
        a = 2 * math.pi * i / count + gap
        put(out, cx + rx * math.cos(a), cy + ry * math.sin(a), colors[i % len(colors)])
        put(out, cx + rx * math.cos(a) + 1, cy + ry * math.sin(a), colors[(i + 1) % len(colors)])


def magnet_rings(img, rig, level, dx):
    """Magnetic roar: rings spread from the stone head (no mouth is drawn)."""
    cx, cy = rig["center"]
    c = (cx + dx, cy - 6)
    cols = (WHITE, BLUE_L, PALE, BLUE_M)
    if level >= 1:
        ring(img, c, 40, 30, 30, cols)
    if level >= 2:
        ring(img, c, 50, 37, 38, cols, gap=0.08)
    if level >= 3:
        ring(img, c, 60, 44, 46, (BLUE_L, WHITE, BLUE_M), gap=0.04)
    return img


def impact_burst(img, center, size):
    cx, cy = center
    cols = (WHITE, PALE, BLUE_L, RED_L)
    rays = 12 if size > 1 else 8
    r1, r2 = (4, 19) if size > 1 else (9, 17)
    for i in range(rays):
        a = 2 * math.pi * i / rays + 0.2
        for r in range(r1, r2, 1 if size > 1 else 2):
            put(img, cx + r * math.cos(a), cy + r * math.sin(a), cols[(i + r) % 4])
    if size > 1:
        for dx, dy in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)):
            put(img, cx + dx, cy + dy, WHITE)
    return img


def rock_chips(img, center, f, fac):
    cx, cy = center
    t = f - 10
    for i, (vx, vy) in enumerate(((5, -6), (7, -2), (4, 3), (8, -8), (3, -9))):
        x = cx + fac * vx * t * 0.8
        y = cy + vy * t * 0.7 + 1.2 * t * t
        col = (IRON_L, IRON, BROWN, IRON_L, PALE)[i]
        put(img, x, y, col)
        put(img, x + 1, y, INK)
        put(img, x, y + 1, col)
    return img


def sparks(img, rig, f):
    cx, cy = rig["center"]
    pts = ((-30, -24), (28, -28), (-8, -38), (34, 2), (-36, 4))
    for i, (sx, sy) in enumerate(pts):
        if (i + f) % 2:
            continue
        put(img, cx + sx, cy + sy, WHITE)
        put(img, cx + sx + 1, cy + sy, BLUE_L)
        put(img, cx + sx, cy + sy - 1, BLUE_L)
    return img


def speed_lines(img, rig, f):
    cx, cy = rig["center"]
    fac = rig["facing"]
    back = cx - fac * 40
    for i, yo in enumerate((-12, 4, 18)):
        if (f + i) % 3 == 0:
            continue
        for k in range(5 + 2 * i):
            put(img, back - fac * (k + (f % 4)), cy + yo, (PALE, BLUE_L)[k % 2])
    return img


def flash_hit(img):
    out = img.copy()
    m = any_color(out, BLUES + (RED, RED_L, RED_D))
    out[m, :3] = np.array(WHITE, np.uint8)
    m2 = any_color(out, (IRON, IRON_L, BROWN))
    out[m2, :3] = np.array(PALE, np.uint8)
    return out


def largest_fraction(img):
    opaque = img[..., 3] >= 128
    total = int(opaque.sum())
    if not total:
        return 1.0, 0
    comps = components(opaque)
    return max(int(c.sum()) for c in comps) / total, total


def synth(rig, action, f, palette, effects=True):
    p = pose(action, f)
    if p["recoil"] > 0.4:
        q = dict(p)
        q.update(dx=0, dy=0, recoil=0.0)
        body = compose(rig, q)
        body = shift(body, p["dx"] * rig["facing"], p["dy"])
        fac = rig["facing"]
        # pivot on the back-bottom corner: the heavy stone tips backwards
        ys, xs = np.nonzero(body[..., 3] >= 128)
        piv = (float(xs.max()) if fac < 0 else float(xs.min()), float(ys.max()))
        ang = p["recoil"] * (1.0 if fac < 0 else -1.0)
        body = rotate_nearest(body, ang, piv)
        body = despeckle(fill_cracks(body))
        # keep it above the frame bottom
        ys = np.nonzero(body[..., 3] >= 128)[0]
        if len(ys) and ys.max() > rig["foot"]:
            body = shift(body, 0, rig["foot"] - ys.max())
    else:
        body = compose(rig, p)
    img = snap(body, palette)
    if not effects:
        return img
    fac = rig["facing"]
    if p["flash"]:
        img = flash_hit(img)
    if p["ring"]:
        img = magnet_rings(img, rig, p["ring"], p["dx"] * fac)
    if p["sparks"]:
        img = sparks(img, rig, f)
    if p["lines"]:
        img = speed_lines(img, rig, f)
    hx, hy = rig["mini_home"]
    if p["orbw"] > 0.3 and 3 <= f <= 10:
        # magnetic trail behind the flying mini-nose
        for k, col in ((0.18, WHITE), (0.32, BLUE_L), (0.46, PALE), (0.60, BLUE_M)):
            ox, oy, _ = orbit_pos(rig, p["theta"] - k, p["orbw"])
            for dd in (-1, 0, 1):
                put(img, hx + ox + p["dx"] * fac, hy + oy + dd, col)
    if p["burst"] or p["chips"]:
        ox, oy, _ = orbit_pos(rig, 3.0, 1.0)
        hit = (hx + ox + fac * (p["dx"] + 8 + 9), hy + oy)
        if p["burst"]:
            img = impact_burst(img, hit, p["burst"])
        if p["chips"]:
            img = rock_chips(img, (hit[0] - fac * 4, hit[1]), f, fac)
    return snap(img, palette)


# ---------------------------------------------------------------- outputs
def params_hash():
    src = Path(__file__).read_bytes() + (STILL_DIR / "front.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def assemble(frames_dir):
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
    return master


def preview_image(sheets):
    samples = {"idle": (0, 4, 8, 12), "walk": (2, 6, 10, 14), "attack": (3, 7, 11, 13), "hurt": (1, 3, 6, 10)}
    bg = np.full((len(ACTIONS) * CELL_H, 4 * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            m = tile[..., 3] >= 128
            bg[ai * CELL_H:(ai + 1) * CELL_H, si * CELL_W:(si + 1) * CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    bg = np.full((len(ACTIONS) * CELL_H, 16 * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for f in range(FRAMES[act]):
            tile = sheets[act][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            m = tile[..., 3] >= 128
            bg[ai * CELL_H:(ai + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W][m] = tile[m]
    im = Image.fromarray(bg, "RGBA")
    return im.resize((im.width // 2, im.height // 2), Image.NEAREST)


def dirs_overview(sheets):
    bg = np.full((CELL_H, 8 * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for di in range(8):
        tile = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        m = tile[..., 3] >= 128
        bg[:, di * CELL_W:(di + 1) * CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    return n


def _entry_span(text, key):
    token = f'"{key}":'
    start = text.index(token)
    i = start + len(token)
    while text[i] != "{":
        i += 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return start, j + 1
    raise SystemExit(f"dir8 entry did not close: {key}")


def publish_dir8(sprite_h):
    entry = {
        "dex": 476, "form": "normal", "variant": "probopass", "gen": "gen4",
        "dir": "monsterSprite/gen4/0476_probopass/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": int(FOOT_Y), "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    with DIR8_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            text = DIR8.read_text()
            prefix = "window.ARENA_DIR8 = "
            if not text.startswith(prefix):
                raise SystemExit("dir8.js header changed; not publishing")
            keys = re.findall(r'"(\d{4}_[^"]+)":\{', text)
            if keys.count(KEY) > 1:
                raise SystemExit(f"{KEY} duplicated in dir8")
            if keys.count(KEY) == 1:
                s, e = _entry_span(text, KEY)
                new_text = text[:s] + f'"{KEY}":' + blob + text[e:]
            else:
                anchor = '"0477_dusknoir":'
                i = text.index(anchor)
                new_text = text[:i] + f'"{KEY}":' + blob + "," + text[i:]
            body = new_text[len(prefix):].strip()
            if body.endswith(";"):
                body = body[:-1].strip()
            old = json.loads(text[len(prefix):].strip().rstrip(";"))
            data = json.loads(body)
            if data.get(KEY) != entry:
                raise SystemExit("dir8 entry did not round-trip")
            for k, v in old.items():
                if k != KEY and data.get(k) != v:
                    raise SystemExit(f"dir8 neighbour changed: {k}")
            DIR8.write_text(new_text)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def bounds_from_sheet(master):
    out = []
    for k in range(8):
        tile = master[k * CELL_H:(k + 1) * CELL_H, :CELL_W, 3]
        ys, xs = np.nonzero(tile)
        out += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    return out


def publish_bounds(vals):
    text = BOUNDS.read_text()
    blob = "[" + ",".join(str(v) for v in vals) + "]"
    pat = re.compile(r'("s":\{.*?)"' + re.escape(KEY) + r'":\[[^\]]*\]', re.S)
    m = re.search(r'"' + re.escape(KEY) + r'":\[[0-9,]*\]', text[text.index('"s":{'):])
    if not m:
        raise SystemExit("sprite_bounds s entry missing")
    off = text.index('"s":{')
    s, e = off + m.start(), off + m.end()
    new_text = text[:s] + f'"{KEY}":' + blob + text[e:]
    BOUNDS.write_text(new_text)


def update_queue(status, note=None):
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            text = QUEUE.read_text()
            before = json.loads(text)
            start = text.index(f'"id": "{KEY}"')
            nxt = text.find('"id":', start + 10)
            end = len(text) if nxt < 0 else nxt
            block = text[start:end]
            block2 = re.sub(r'"status": "[^"]*"', f'"status": "{status}"', block, count=1)
            if note:
                block2 = re.sub(r'("notes": ")', r'\1' + note.replace("\\", "") + " ", block2, count=1)
            text2 = text[:start] + block2 + text[end:]
            data = json.loads(text2)
            for a, b in zip(before["items"], data["items"]):
                if a.get("id") != KEY and a != b:
                    raise SystemExit("queue neighbour changed")
            QUEUE.write_text(text2)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def qa_sheet(master, sheets, palette, frames_dir, rest_diff, skip_qa):
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    partial = int(((master[..., 3] != 0) & (master[..., 3] != 255)).sum())
    clip = []
    shattered = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = tile[..., 3] > 0
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
                if act == "hurt" or (act == "attack" and 5 <= f <= 12):
                    frac, _ = largest_fraction(tile)
                    if frac < 0.80:
                        shattered.append(f"{act}:{d}:{f}:{frac:.2f}")
    loop_end = {}
    for act in ("idle", "walk"):
        f0 = sheets[act][:CELL_H, :CELL_W]
        # the frame after the loop must equal frame 0 => compare pose(n) with frame 0
        loop_end[act] = 0
    report = {
        "version": VERSION, "cell": [CELL_W, CELL_H], "foot_y": FOOT_Y,
        "rest_diff": rest_diff, "off_palette_colors": len(off), "partial_alpha": partial,
        "edge_touching": clip, "shattered": shattered, "recoil_max": MAX_RECOIL,
    }
    if not skip_qa and GAIT_LOOP.exists():
        qa = OUT_DIR / "_work" / "qa"
        qa.mkdir(parents=True, exist_ok=True)
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run(
                [str(FORGE / ".venv/bin/python"), "-I", str(GAIT_LOOP), "select", "--frames-dir",
                 str(frames_dir / act / "S"), "--fps", "12", "--output-dir", str(od),
                 "--kind", "hover", "--no-aids"], capture_output=True, text=True)
            rep = od / "loop-report.json"
            report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation") if rep.exists() else (r.stderr or r.stdout)[-300:]
    report["ok"] = rest_diff == 0 and not off and partial == 0 and not clip and not shattered
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(STILL_DIR / "front.png").convert("RGBA"))
    if front.shape[:2] != (96, 96):
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    info = measure(front, masks)
    print("measure", json.dumps(info), "palette", len(palette), flush=True)
    if info["facing"] != "left":
        raise SystemExit("facing changed; re-check FLIP_DIRS")

    rigs = {d: build_rig(front, masks, flip=d in FLIP_DIRS) for d in DIRS}
    diffs = {}
    for d in ("W", "E", "S", "N"):
        rest = synth(rigs[d], "idle", 0, palette, effects=False)
        diffs[d] = int(np.any(rest != rigs[d]["canvas"], axis=2).sum())
        for act in ("idle", "walk"):
            end = synth(rigs[d], act, FRAMES[act], palette, effects=False)
            diffs[f"{d}_{act}_end"] = int(np.any(end != rigs[d]["canvas"], axis=2).sum())
        for act in ("attack", "hurt"):
            f0 = synth(rigs[d], act, 0, palette, effects=False)
            diffs[f"{d}_{act}0"] = int(np.any(f0 != rigs[d]["canvas"], axis=2).sum())
    print("rest diffs", diffs, flush=True)
    rest_diff = sum(diffs.values())
    if rest_diff:
        raise SystemExit("frame 0 / loop end does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((96, 96, 4), np.uint8)
    cols = {"body": (80, 140, 200), "hat": (230, 80, 80), "nose": (240, 170, 60),
            "mustache": (90, 90, 90), "mini_l": (80, 210, 120), "mini_r": (200, 90, 220)}
    for name, col in cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("attack", "W", f) for f in range(16)]
        want += [("attack", "E", f) for f in (2, 6, 8, 11)]
        want += [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("hurt", "W", f) for f in (0, 1, 2, 3, 5, 7, 9, 11)]
        want += [("idle", d, 0) for d in DIRS]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        ncol = 8
        rows = math.ceil(len(tiles) / ncol)
        sheet = np.full((rows * CELL_H, ncol * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, ncol)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
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
                if fp.exists() and fp.stat().st_size > 0:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview_image(sheets).save(OUT_DIR / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")
    dirs_overview(sheets).save(work / "dirs.png")
    (STILL_DIR / "actions").mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", STILL_DIR / "actions" / "preview.png")

    s0 = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(s0[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1)
    report = qa_sheet(master, sheets, palette, frames_dir, rest_diff, args.skip_qa)
    report["sprite_h"] = sprite_h
    report["fy_measured"] = int(nys.max())
    report["hash"] = digest
    (work / "qa").mkdir(parents=True, exist_ok=True)
    (work / "qa" / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False)[:1500], flush=True)

    meta = {
        "generator": "forge_probopass_actions", "version": VERSION, "key": KEY, "dex": 476,
        "variant": "probopass", "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y,
        "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "flip_dirs": sorted(FLIP_DIRS), "palette_colors": int(len(palette)),
        "off_palette_colors": report["off_palette_colors"], "parts": info["parts"],
        "facing": info["facing"], "feet": 0, "mouth": 0, "source": "front.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not report["ok"]:
        raise SystemExit("qa failed; not publishing")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if not args.no_publish:
        publish_dir8(sprite_h)
        publish_bounds(bounds_from_sheet(master))
        update_queue("done")
        print("published", VERSION, flush=True)


if __name__ == "__main__":
    main()
