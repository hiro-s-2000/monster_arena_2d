#!/usr/bin/env python3
"""Leafeon normal (0470) 8-direction sheet from this form's front.png only.

Measured on normal/front.png (96x96, opaque 1267, bbox x23-73 y23-72, 51x50,
14 colours, alpha already 0 or 255). Centroid about (46.1, 47.5). Faces left:
both eye highlights (x28,y48) and (x35,y49) sit on the cream head, and the leaf
tail occupies x58-73 y23-50. Unflipped rows are N NW W SW. S SE E NE are mirrored
so E looks screen-right.

Quadruped. Exactly four feet. Leaves are not legs and are not given paws.
  near fore   x32-44 y62-72   one paw, sole on y72. The far fore is occluded
              inside this silhouette (one paw, no second toe gap). Not invented.
  far hind    x48-54 y65-70   the higher paw. Splits from the near hind at y65.
  near hind   x55-63 y65-72   sole on y72. Leaf speck (64,65) is a tail-side
              leaf tip on this hock, not a fourth paw.
  shared hind thigh y63-64 stays on the torso so the two hocks are not torn apart.
  leaf ears   green x24-54 y27-48. Not legs.
  leaf tail   green x56-73 y23-52. Not a leg.
No separate mouth: the dark marks under the highlights are the eyes. Roar flares
the ears and the leaf tail, plus rings in front of the snout.

  idle   : feet planted. Frame 0 is the still. Chest breathes 1px. Ears and tail sway.
  walk   : diagonal trot. Near fore with near hind, far hind opposite. Two steps.
  attack : 0 still, 1-4 leaf roar plus rings, 5-10 one-piece in-plane 360
           (no |cos| squash), 11-12 impact, 13-15 rest.
  hurt   : one-piece recoil about the sole, 38 degrees max.

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_leafeon_actions.py
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0470_leafeon/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0470_leafeon/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left. These four keep the drawing. E is the mirror and looks screen-right.
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 192, 200, 172
VERSION = "leafeon_v2"
KEY = "0470_leafeon"
NEIGHBORS = ("0469_yanmega", "0471_glaceon")

INK = (16, 16, 16)
LEAF = (49, 156, 115)
DLEAF = (49, 98, 90)
MLEAF = (106, 189, 139)
CREAM = (238, 213, 156)
WHITE = (238, 255, 255)

PARTS = ("body", "fore", "hind_far", "hind_near", "tail", "ear_l", "ear_r")


def flood(seeds, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow)
    q = deque()
    ys, xs = np.nonzero(seeds & allow)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if not seen[y, x]:
            seen[y, x] = True
            q.append((int(y), int(x)))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and allow[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


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


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def colour_mask(img, colour):
    rgb = img[..., :3]
    return (img[..., 3] == 255) & np.all(rgb == np.array(colour, np.uint8), axis=-1)


def segment(img):
    """Three drawn paws plus the occluded far fore. Ears and the leaf tail stay off the legs."""
    h, w = img.shape[:2]
    opaque = img[..., 3] == 255
    yy, xx = np.mgrid[0:h, 0:w]
    green = colour_mask(img, LEAF) | colour_mask(img, DLEAF) | colour_mask(img, MLEAF)
    ink = colour_mask(img, INK)

    fore_seed = np.zeros_like(opaque)
    fore_seed[72, 33:36] = True
    fore = flood(fore_seed, opaque & (yy >= 62) & (xx <= 45))

    # Two hind paws. They share a thigh through y64, which stays on the torso.
    far_seed = np.zeros_like(opaque)
    far_seed[70, 49] = True
    hind_far = flood(far_seed, opaque & (yy >= 65) & (xx >= 46) & (xx <= 54) & ~fore)

    near_seed = np.zeros_like(opaque)
    near_seed[72, 56:59] = True
    hind_near = flood(near_seed, opaque & (yy >= 65) & (xx >= 55) & ~fore & ~hind_far)
    # One dark-green leaf tip, diagonal to the hock. Not a foot.
    if opaque[65, 64]:
        hind_near[65, 64] = True

    ear_g = green & (yy <= 48) & (xx <= 54)
    ear_ink = ink & dilate(ear_g, 1) & (yy <= 48) & (xx <= 55)
    ear = (ear_g | ear_ink) & ~fore & ~hind_far & ~hind_near
    ear_l = ear & (xx <= 36)
    ear_r = ear & (xx >= 37)
    if int(ear_l.sum()) < 8 or int(ear_r.sum()) < 8:
        ear_l = ear
        ear_r = np.zeros_like(ear)

    tail_g = green & (yy <= 52) & (xx >= 56)
    tail_ink = ink & dilate(tail_g, 1) & (yy <= 53) & (xx >= 55)
    tail = (tail_g | tail_ink) & ~fore & ~hind_far & ~hind_near & ~ear

    body = opaque & ~fore & ~hind_far & ~hind_near & ~tail & ~ear
    return {
        "body": body,
        "fore": fore,
        "hind_far": hind_far,
        "hind_near": hind_near,
        "tail": tail,
        "ear_l": ear_l,
        "ear_r": ear_r,
    }


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
    """4x nearest in-plane rotation. Width is never scaled by |cos|."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.05 or not np.any(layer[..., 3] == 255):
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
    """Drop a pixel only when all four neighbours are empty. After rotation only."""
    out = img.copy()
    a = out[..., 3] == 255
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def knit(img, limit=8):
    """Join a diagonal speck to the body with one copied palette pixel."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(limit):
        a = out[..., 3] == 255
        lab = np.zeros(a.shape, np.int32)
        sizes = []
        cid = 0
        ys, xs = np.nonzero(a)
        seen = np.zeros_like(a)
        for y, x in zip(ys.tolist(), xs.tolist()):
            if seen[y, x]:
                continue
            cid += 1
            q = deque([(y, x)])
            seen[y, x] = True
            lab[y, x] = cid
            n = 0
            while q:
                cy, cx = q.popleft()
                n += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        lab[ny, nx] = cid
                        q.append((ny, nx))
            sizes.append(n)
        if cid <= 1:
            break
        small = 1 + int(np.argmin(sizes))
        if sizes[small - 1] > 4:
            break
        filled = False
        sy, sx = np.nonzero(lab == small)
        for y, x in zip(sy.tolist(), sx.tolist()):
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w) or a[ny, nx]:
                    continue
                touch = False
                for dy2, dx2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    qy, qx = ny + dy2, nx + dx2
                    if 0 <= qy < h and 0 <= qx < w and lab[qy, qx] not in (0, small):
                        touch = True
                        break
                if touch:
                    out[ny, nx, :3] = out[y, x, :3]
                    out[ny, nx, 3] = 255
                    filled = True
                    break
            if filled:
                break
        if not filled:
            break
    return out


def extract_palette(img):
    a = img[..., 3] == 255
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] == 255
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
    flat[sel] = mapped[inv]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    return out


def put_empty(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def paste(dst, layer):
    m = layer[..., 3] == 255
    dst[m] = layer[m]


def top_rows(mask, rows):
    ys = np.nonzero(mask)[0]
    if len(ys) == 0:
        return np.zeros_like(mask)
    yy = np.arange(mask.shape[0])[:, None]
    return mask & (yy <= int(ys.min()) + rows - 1)


def bottom_rows(mask, rows):
    ys = np.nonzero(mask)[0]
    if len(ys) == 0:
        return np.zeros_like(mask)
    yy = np.arange(mask.shape[0])[:, None]
    return mask & (yy >= int(ys.max()) - (rows - 1))


def left_cols(mask, cols):
    xs = np.nonzero(mask)[1]
    if len(xs) == 0:
        return np.zeros_like(mask)
    xx = np.arange(mask.shape[1])[None, :]
    return mask & (xx <= int(xs.min()) + cols - 1)


def place_geometry(img):
    a = img[..., 3] == 255
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    return y0, y1, x0, x1, dst_y, dst_x


def mask_pivot(mask, how):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return (CELL_W / 2.0, CELL_H / 2.0)
    if how == "bottom":
        return (float(xs.mean()), float(ys.max()))
    if how == "left":
        root = xs <= int(xs.min()) + 1
        return (float(xs[root].mean()), float(ys[root].mean()))
    return (float(xs.mean()), float(ys.mean()))


def build_rig(img, masks, flip):
    y0, y1, x0, x1, dst_y, dst_x = place_geometry(img)
    sh, sw = y1 - y0, x1 - x0
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]

    placed = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm

    cushion = (
        top_rows(placed["fore"], 3)
        | top_rows(placed["hind_far"], 2)
        | top_rows(placed["hind_near"], 2)
        | left_cols(placed["tail"], 2)
        | bottom_rows(placed["ear_l"], 2)
        | bottom_rows(placed["ear_r"], 2)
    )
    layers = {}
    for name in PARTS:
        lyr = np.zeros_like(canvas)
        lyr[placed[name]] = canvas[placed[name]]
        layers[name] = lyr
    layers["body"][cushion] = canvas[cushion]

    bys, bxs = np.nonzero(canvas[..., 3] == 255)
    center = (float(bxs.mean()), float(bys.mean()))
    sole_x = bxs[bys >= bys.max() - 1]
    sole = (float(np.median(sole_x)), float(bys.max()))
    # Rings sit just in front of the left-facing snout, ahead of the eyes.
    eye = canvas[..., 0] == WHITE[0]
    eye &= canvas[..., 1] == WHITE[1]
    eye &= canvas[..., 3] == 255
    eys, exs = np.nonzero(eye)
    if len(exs):
        mouth = (float(exs.min()) - 6.0, float(np.median(eys)))
    else:
        mouth = (float(bxs.min()) + 2.0, float(np.median(bys)))
    pivots = {
        "tail": mask_pivot(placed["tail"], "left"),
        "ear_l": mask_pivot(placed["ear_l"], "bottom"),
        "ear_r": mask_pivot(placed["ear_r"], "bottom"),
        "center": center,
        "sole": sole,
        "mouth": mouth,
    }
    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        for name in ("tail", "ear_l", "ear_r", "center", "sole", "mouth"):
            px, py = pivots[name]
            pivots[name] = (CELL_W - 1 - px, py)
        canvas = canvas[:, ::-1].copy()
    facing = 1.0 if flip else -1.0
    return layers, canvas, pivots, facing


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        bdy=0,
        lift_f=0, lift_a=0, lift_b=0,
        step_f=0, step_a=0, step_b=0,
        tail=0.0, ear_l=0.0, ear_r=0.0,
        spin=0.0, hurt=0.0, knock=0.0,
        rings=0, ring=0, burst=0, sparks=0,
    )
    if action == "idle":
        breath = wave(f, n, 1.0)
        p["bdy"] = -1 if breath > 0.35 else 0
        p["tail"] = wave(f, n, 8.0)
        p["ear_l"] = wave(f, n, 5.0, phase=0.6)
        p["ear_r"] = wave(f, n, 4.0, phase=1.3)
    elif action == "walk":
        # Two-step diagonal trot. Near fore + near hind, far hind opposite.
        s = math.sin(2 * math.pi * 2 * f / n)
        p["bdy"] = -1 if abs(s) > 0.65 else 0
        p["lift_f"] = int(round(3 * max(0.0, s)))
        p["lift_b"] = int(round(3 * max(0.0, s)))
        p["lift_a"] = int(round(3 * max(0.0, -s)))
        p["step_f"] = p["lift_f"]
        p["step_b"] = p["lift_b"]
        p["step_a"] = p["lift_a"]
        p["tail"] = 9.0 * s
        p["ear_l"] = 4.0 * s
        p["ear_r"] = -3.0 * s
    elif action == "attack":
        # 0 still. 1-4 leaf roar. 5-10 full turn. 11-12 impact. 13-15 rest.
        flare = [0, 10, 16, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["ear_l"] = float(flare)
        p["ear_r"] = float(flare) * 0.85
        p["tail"] = float(flare)
        p["rings"] = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["ring"] = max(0, f - 1)
        if 5 <= f <= 10:
            p["spin"] = 60.0 * (f - 4)
        if f in (11, 12):
            p["burst"] = 1
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 38.0 * env
        p["knock"] = 2.0 * env
        p["sparks"] = 1 if 2 <= f <= 5 else 0
    return p


def assemble_pose(layers, pivots, p, facing):
    """Tail, torso, hind paws, forepaw, ears. Zero pose matches the still."""
    bdy = p["bdy"]
    body = shift(layers["body"], 0, bdy)
    tail = shift(
        rotate_nearest(layers["tail"], p["tail"] * facing, pivots["tail"]),
        0, bdy,
    )
    fore = shift(layers["fore"], facing * p["step_f"], -p["lift_f"])
    hind_far = shift(layers["hind_far"], facing * p["step_a"], -p["lift_a"])
    hind_near = shift(layers["hind_near"], facing * p["step_b"], -p["lift_b"])
    ear_l = shift(
        rotate_nearest(layers["ear_l"], p["ear_l"] * facing, pivots["ear_l"]),
        0, bdy,
    )
    ear_r = shift(
        rotate_nearest(layers["ear_r"], -p["ear_r"] * facing, pivots["ear_r"]),
        0, bdy,
    )
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for lyr in (tail, body, hind_near, hind_far, fore, ear_r, ear_l):
        paste(img, lyr)
    return img


def neutral_bitmap(layers):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in ("tail", "body", "hind_near", "hind_far", "fore", "ear_r", "ear_l"):
        paste(img, layers[name])
    return img


def roar_rings(out, mouth, facing, age, strength):
    mx, my = mouth
    cols = (LEAF, CREAM, WHITE, INK)
    for i in range(strength):
        rad = 6 + age + i * 4
        col = cols[i % 4]
        for deg in (-32, -8, 16):
            ang = math.radians(deg)
            put_empty(
                out,
                mx + facing * rad * math.cos(ang),
                my + rad * math.sin(ang) * 0.55,
                col,
            )
    return out


def impact_burst(out, facing):
    ys, xs = np.nonzero(out[..., 3] == 255)
    if len(xs) == 0:
        return out
    cy = float(np.median(ys))
    cx = float(xs.min() if facing < 0 else xs.max())
    cx += facing * 6
    for k in range(6):
        ang = k * math.pi / 3
        col = LEAF if k % 2 == 0 else CREAM
        put_empty(out, cx + 5 * math.cos(ang), cy + 4 * math.sin(ang), col)
        put_empty(out, cx + 9 * math.cos(ang), cy + 7 * math.sin(ang), INK if k % 2 else WHITE)
    return out


def sparks(out, facing):
    ys, xs = np.nonzero(out[..., 3] == 255)
    if len(xs) == 0:
        return out
    y = int(ys.min()) + 4
    x = int(xs.min() + 8 if facing < 0 else xs.max() - 8)
    put_empty(out, x, y, WHITE)
    put_empty(out, x + facing * 3, y + 2, CREAM)
    put_empty(out, x - facing * 2, y + 3, LEAF)
    return out


def synth(rig, action, f, palette):
    layers, _canvas, pivots, facing = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = 5 <= f <= 10 and action == "attack" and p["spin"] > 0.5
    # Frame 10 is 360 degrees: still one piece, then the same cleanup as the turn.
    full_turn = action == "attack" and 5 <= f <= 10
    recoiling = action == "hurt" and p["hurt"] > 0.45
    if full_turn or recoiling:
        base = neutral_bitmap(layers)
        if full_turn:
            img = rotate_nearest(base, p["spin"], pivots["center"])
        else:
            img = rotate_nearest(base, -facing * p["hurt"], pivots["sole"])
            img = shift(img, -facing * p["knock"], 0)
        img = despeckle(img)
        img = knit(img)
    else:
        img = assemble_pose(layers, pivots, p, facing)
    img = snap(img, palette)
    if p["rings"] and not full_turn:
        mouth = pivots["mouth"]
        img = roar_rings(img, mouth, facing, p["ring"], p["rings"])
    if p["burst"]:
        img = impact_burst(img, facing)
    if p["sparks"] and not full_turn:
        img = sparks(img, facing)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (SRC_DIR / "front.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def assemble(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png"))
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
    scale = 2
    x0, x1 = 48, 152
    y0, y1 = 96, 188
    cw, ch = (x1 - x0) * scale, (y1 - y0) * scale
    cols = 8
    blocks = []
    for act in ACTIONS:
        band = sheets[act][:CELL_H]
        frames = []
        for f in range(FRAMES[act]):
            tile = band[y0:y1, f * CELL_W + x0:f * CELL_W + x1]
            frames.append(np.repeat(np.repeat(tile, scale, 0), scale, 1))
        blocks.append(frames)
    rows = sum(math.ceil(len(b) / cols) for b in blocks)
    bg = np.full((rows * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    r = 0
    for frames in blocks:
        for i, up in enumerate(frames):
            rr, cc = divmod(i, cols)
            y = (r + rr) * ch
            x = cc * cw
            m = up[..., 3] == 255
            bg[y:y + ch, x:x + cw][m] = up[m]
        r += math.ceil(len(frames) / cols)
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 6
    crops = []
    for act in ACTIONS:
        row = []
        band = sheets[act][:CELL_H]
        for f in range(FRAMES[act]):
            tile = band[:, f * CELL_W:(f + 1) * CELL_W]
            a = tile[..., 3] == 255
            if not np.any(a):
                continue
            ys, xs = np.nonzero(a)
            yy0 = max(0, int(ys.min()) - pad)
            yy1 = min(CELL_H, int(ys.max()) + 1 + pad)
            xx0 = max(0, int(xs.min()) - pad)
            xx1 = min(CELL_W, int(xs.max()) + 1 + pad)
            crop = tile[yy0:yy1, xx0:xx1]
            row.append(np.repeat(np.repeat(crop, 3, 0), 3, 1))
        crops.append(row)
    cell_h = max(c.shape[0] for row in crops for c in row)
    cell_w = max(c.shape[1] for row in crops for c in row)
    cols = max(len(row) for row in crops)
    bg = np.full((len(crops) * cell_h, cols * cell_w, 4), (15, 20, 31, 255), np.uint8)
    for ri, row in enumerate(crops):
        for ci, c in enumerate(row):
            y = ri * cell_h + (cell_h - c.shape[0]) // 2
            x = ci * cell_w + (cell_w - c.shape[1]) // 2
            m = c[..., 3] == 255
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
        if name.startswith("front") or name.startswith("back"):
            raise SystemExit(f"refusing to copy {name}")
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    return n


def with_flock(lock_path, fn):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


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
        "dex": 470,
        "form": "normal",
        "variant": "leafeon",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0470_leafeon/normal",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir",
        "nf": 16,
        "cw": CELL_W,
        "ch": CELL_H,
        "w": CELL_W,
        "h": int(sprite_h),
        "fy": int(FOOT_Y),
        "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1,
        "src": "agent-sprite-forge",
        "v": VERSION,
    }

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        raw = text[len(prefix):]
        if raw.endswith("\n"):
            raw = raw[:-1]
        if not raw.endswith(";"):
            raise SystemExit("dir8.js missing trailing semicolon")
        raw = raw[:-1]
        data = json.loads(raw)
        if KEY not in data:
            raise SystemExit("dir8.js has no 0470_leafeon")
        frozen = {}
        for k in NEIGHBORS:
            s, e = _entry_span(text, k)
            frozen[k] = text[s:e]
        order = list(data)
        data[KEY] = entry
        if list(data) != order:
            raise SystemExit("dir8 key order changed")
        new_body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        new_text = prefix + new_body + ";\n"
        written = json.loads(new_body)
        if list(written) != order:
            raise SystemExit("dir8 dumps reordered keys")
        for k, old in data.items():
            if k == KEY:
                if written[k] != entry:
                    raise SystemExit("dir8 leafeon entry mismatch")
            elif written[k] != old:
                raise SystemExit(f"dir8 rewrote {k}")
        for k, old in frozen.items():
            s, e = _entry_span(new_text, k)
            if new_text[s:e] != old:
                raise SystemExit(f"dir8 neighbour bytes changed: {k}")
        if new_text.count(f'"v":"{VERSION}"') != 1:
            raise SystemExit("dir8 version stamp collided")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(new_text)
        tmp.replace(DIR8)
        check = DIR8.read_text()
        if check != new_text:
            raise SystemExit("dir8 reread mismatch")
        return entry["v"]

    return with_flock(DIR8_LOCK, edit)


def update_queue(status):
    def edit():
        text = QUEUE.read_text()
        marker = '"id": "0470_leafeon"'
        if text.count(marker) != 1:
            raise SystemExit("queue id count")
        idx = text.index(marker)
        start = text.rfind("{", 0, idx)
        depth = 0
        end = None
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end is None:
            raise SystemExit("queue block did not close")
        block = text[start:end]
        if '"status": "in_progress"' not in block or '"claim": "parallel-batch"' not in block:
            raise SystemExit("queue claim is not parallel-batch")
        block2 = block.replace('"status": "in_progress"', f'"status": "{status}"', 1)
        block2, n_claim = re.subn(r',\s*"claim": "parallel-batch"', "", block2, count=1)
        if n_claim != 1:
            raise SystemExit("queue claim line not removed")
        new = text[:start] + block2 + text[end:]
        if not new.endswith("\n"):
            new += "\n"
        old = json.loads(text)
        data = json.loads(new)
        if [item.get("id") for item in old["items"]] != [item.get("id") for item in data["items"]]:
            raise SystemExit("queue id order changed")
        for a, b in zip(old["items"], data["items"]):
            if a.get("id") == KEY:
                if b.get("status") != status or "claim" in b:
                    raise SystemExit("queue leafeon update did not stick")
                aa = dict(a)
                bb = dict(b)
                aa.pop("status", None)
                bb.pop("status", None)
                aa.pop("claim", None)
                if aa != bb:
                    raise SystemExit("queue leafeon fields other than status/claim changed")
            elif a != b:
                raise SystemExit(f"queue neighbour changed: {a.get('id')}")
        tmp = QUEUE.with_suffix(".json.tmp")
        tmp.write_text(new)
        tmp.replace(QUEUE)

    with_flock(QUEUE_LOCK, edit)


def assert_claim():
    import os as _rebuild_os
    if _rebuild_os.environ.get("GEN4_ATTACK_REBUILD") == "1":
        return
    data = json.loads(QUEUE.read_text())
    for item in data["items"]:
        if item.get("id") == KEY:
            if item.get("status") != "in_progress" or item.get("claim") != "parallel-batch":
                raise SystemExit(f"stop without drawing: {item.get('status')} {item.get('claim')}")
            return
    raise SystemExit("0470_leafeon missing from queue")


def opaque_stats(tile, base):
    c = tile[..., 3] == 255
    n = int(c.sum())
    ys, xs = np.nonzero(c)
    bw = int(xs.max() - xs.min() + 1) if len(xs) else 0
    bh = int(ys.max() - ys.min() + 1) if len(ys) else 0
    sizes = components(c)
    main = sizes[0] / n if sizes and n else 0.0
    edge = bool(c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()))
    return {
        "opaque": n,
        "ratio": (n / base) if base else 0.0,
        "w": bw,
        "h": bh,
        "parts": len(sizes),
        "main": main,
        "top": sizes[:4],
        "edge": edge,
    }


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), int(mask.sum()))


def main():
    assert_claim()
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    opaque = front[..., 3] == 255
    covered = np.zeros_like(opaque)
    overlap = 0
    for name in PARTS:
        overlap += int((covered & masks[name]).sum())
        covered |= masks[name]
    counts = {k: int(masks[k].sum()) for k in PARTS}
    counts["missing"] = int((opaque & ~covered).sum())
    counts["overlap"] = overlap
    print("parts", counts, "palette", len(palette), flush=True)
    print(
        "feet",
        "fore", bbox_of(masks["fore"]),
        "hind_far", bbox_of(masks["hind_far"]),
        "hind_near", bbox_of(masks["hind_near"]),
        "occluded_far_fore", 1,
        "drawn", 3,
        "total", 4,
        flush=True,
    )
    print("tail", bbox_of(masks["tail"]), "ear_l", bbox_of(masks["ear_l"]), "ear_r", bbox_of(masks["ear_r"]), flush=True)
    if counts["missing"] or overlap:
        raise SystemExit("segmentation does not partition the still")
    if counts["fore"] < 40 or counts["hind_far"] < 12 or counts["hind_near"] < 20:
        raise SystemExit("a paw mask collapsed")
    if counts["tail"] < 120 or counts["ear_l"] + counts["ear_r"] < 80:
        raise SystemExit("leaf mask collapsed")
    cream = colour_mask(front, CREAM)
    if int((cream & (masks["tail"] | masks["ear_l"] | masks["ear_r"])).sum()):
        raise SystemExit("leaf mask ate cream body")
    if components(masks["fore"]) != [counts["fore"]]:
        raise SystemExit("fore paw is not one piece")
    if components(masks["hind_far"]) != [counts["hind_far"]]:
        raise SystemExit("far hind paw is not one piece")
    # The hock leaf tip is one diagonal pixel. The paw itself is the rest.
    near_sizes = components(masks["hind_near"])
    if not near_sizes or near_sizes[0] < 20:
        raise SystemExit("near hind paw collapsed")

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    rest = synth(rigs["W"], "idle", 0, palette)
    canvas = rigs["W"][1]
    differ = np.any(rest != canvas, axis=2) & ((rest[..., 3] == 255) | (canvas[..., 3] == 255))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()):
        raise SystemExit("idle frame 0 does not match the still")
    for act in ("walk", "attack", "hurt"):
        fr = synth(rigs["W"], act, 0, palette)
        dlt = np.any(fr != canvas, axis=2) & ((fr[..., 3] == 255) | (canvas[..., 3] == 255))
        if int(dlt.sum()):
            raise SystemExit(f"{act} frame 0 does not match the still")
    south = synth(rigs["S"], "idle", 0, palette)
    mirror = canvas[:, ::-1].copy()
    ed = np.any(south != mirror, axis=2) & ((south[..., 3] == 255) | (mirror[..., 3] == 255))
    print("south mirror pixels changed", int(ed.sum()), flush=True)
    if int(ed.sum()):
        raise SystemExit("south idle 0 is not the mirror of the still")

    base = int((south[..., 3] == 255).sum())
    print("SOUTH_RATIOS base", base, flush=True)
    bad = []
    spin_parts = []
    for act in ACTIONS:
        ratios = []
        for f in range(FRAMES[act]):
            tile = synth(rigs["S"], act, f, palette)
            st = opaque_stats(tile, base)
            ratios.append(st["ratio"])
            flag = ""
            if st["ratio"] < 0.90 or st["ratio"] > 1.08 or st["edge"]:
                bad.append((act, f, round(st["ratio"], 4), st["edge"]))
                flag = " FAIL"
            if act == "attack" and 5 <= f <= 10:
                spin_parts.append((f, st["parts"], st["top"], round(st["main"], 4), st["w"], st["h"]))
                if st["parts"] != 1 or st["main"] < 0.90:
                    bad.append((act, f, "parts", st["parts"], st["top"]))
                if st["w"] < 20 or st["h"] < 20:
                    bad.append((act, f, "thin", st["w"], st["h"]))
            print(
                f"S {act} {f:02d} opaque {st['opaque']} ratio {st['ratio']:.4f} "
                f"parts {st['parts']} main {st['main']:.4f} {st['w']}x{st['h']}{flag}",
                flush=True,
            )
        print(f"S {act} min {min(ratios):.4f} max {max(ratios):.4f}", flush=True)
    print("SPIN_COMPONENTS", spin_parts, flush=True)
    if bad:
        print("early QA FAIL", bad, flush=True)
        raise SystemExit(1)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "body": (90, 140, 220),
        "fore": (220, 60, 60),
        "hind_far": (240, 200, 40),
        "hind_near": (40, 180, 90),
        "tail": (40, 160, 90),
        "ear_l": (255, 140, 40),
        "ear_r": (255, 220, 80),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    frames_dir = work / "frames"
    state_p = work / "state.json"
    digest = params_hash()
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    if state.get("hash") != digest:
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
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions = SRC_DIR / "actions"
    actions.mkdir(exist_ok=True)
    preview.save(actions / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] == 255
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] == 255][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = [int(v) for v in np.unique(master[..., 3])]
    bad_alpha = [v for v in alphas if v not in (0, 255)]
    clip = {}
    south_rows = {}
    base_n = int((sheets["idle"][:CELL_H, :CELL_W, 3] == 255).sum())
    for act in ACTIONS:
        sh = sheets[act]
        south_rows[act] = []
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] == 255
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if d == "S":
                    st = opaque_stats(tile, base_n)
                    south_rows[act].append({
                        "f": f,
                        "opaque": st["opaque"],
                        "ratio": round(st["ratio"], 4),
                        "parts": st["parts"],
                        "main": round(st["main"], 4),
                        "w": st["w"],
                        "h": st["h"],
                    })
    bad_ratio = []
    weak = []
    for act, rows in south_rows.items():
        vals = [row["ratio"] for row in rows]
        print(
            f"SHEET S {act} ratios",
            " ".join(f"{v:.4f}" for v in vals),
            f"min {min(vals):.4f} max {max(vals):.4f}",
            flush=True,
        )
        for row in rows:
            if row["ratio"] < 0.90 or row["ratio"] > 1.08:
                bad_ratio.append((act, row))
            if act == "attack" and 5 <= row["f"] <= 10 and (row["parts"] != 1 or row["main"] < 0.90):
                weak.append(row)
    print("sprite_h", sprite_h, "off", len(off), "alphas", alphas, "clip", clip, flush=True)
    report = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_h,
        "feet_drawn": 3,
        "feet_total": 4,
        "off_palette_colors": len(off),
        "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "south": south_rows,
    }
    if GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        qa = work / "qa"
        qa.mkdir(exist_ok=True)
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                continue
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (work / "qa").mkdir(exist_ok=True)
    (work / "qa" / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))

    meta = {
        "generator": "forge_leafeon_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 470,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED),
        "palette_colors": int(len(palette)),
        "off_palette_colors": len(off),
        "source": "front.png 96x96",
        "gifs_untouched": True,
        "facing": "left",
        "feet": 4,
        "feet_drawn": 3,
        "occluded_far_fore": 1,
        "archetype": "quadruped_beast",
        "mouth": "none_leaf_flare",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ok = not off and not bad_alpha and not clip and not bad_ratio and not weak and sprite_h > 0
    if not ok:
        print("QA FAIL", "off", off, "alpha", bad_alpha, "clip", clip, "bad_ratio", bad_ratio, "weak", weak, flush=True)
        raise SystemExit(1)
    print("synced files", sync_outputs(), flush=True)
    publish_dir8(sprite_h)
    update_queue("done")
    sheet = SYNC_OUT / "sheet.png"
    print(
        "published", KEY, VERSION, "h", sprite_h,
        "sheet_bytes", sheet.stat().st_size if sheet.is_file() else 0,
        flush=True,
    )


if __name__ == "__main__":
    main()
