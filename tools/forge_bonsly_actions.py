#!/usr/bin/env python3
"""Bonsly (0438) 8-direction action sheet from the normal 96x96 front.png.

Measured on this still only (opaque 843, bbox x34-62 y23-72, 12 colours).
Viewer-facing south: both eyes sit inside the pot, not on a profile edge.
S SE E NE keep the still. N NW W SW mirror it.

Parts, from this picture only:
  foliage_top / foliage_l / foliage_r
      three green balls (117, 93, 90) plus the ink that touches them.
      These are branches. They are not legs.
  pot
      stump body, face, and the brown trunk (y36-45, 51 px) the balls sit on.
  foot_l
      short left stub, sole at y68 (black). Does not reach the ground line.
  foot_r
      right stub, sole at y72. This is the foot that touches the ground.
  No third leg. The gap under the pot (y66 x42-46) stays on the pot.
  Mouth is the 3 red pixels at y56 x42-44. It is flared, not replaced.
  No separate tear colour. Peak roar recolors one brown pixel under each eye.

idle   : pot and both feet planted. Balls rustle. Frame 0 is the still.
walk   : two-step. Each stub lifts once per loop. Stance foot stays put.
attack : 0-4 branch flare plus rings, 5-10 one-piece in-plane 360 spin,
         11-12 impact, 13-15 return. No abs(cos) squash.
hurt   : one-piece recoil of at most 38 degrees. No belly mirror.

Palette-locked. Alpha is only 0 or 255. front/back png and gif are not written.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
STILL_DIR = ROOT / "monsterSprite/gen4/0438_bonsly/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0438_bonsly/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "bonsly_v2"
KEY = "0438_bonsly"
UNFLIPPED = {"S", "SE", "E", "NE"}

TAN = (172, 131, 90)
INK = (0, 0, 0)
MID = (131, 106, 65)
LEAF = (74, 197, 65)
MOSS = (90, 139, 90)
DARK = (82, 65, 49)
MINT = (156, 222, 123)
GOLD_BARK = (197, 164, 74)
DEEP = (49, 90, 24)
EYE = (255, 238, 123)
PUPIL = (246, 205, 57)
MOUTH = (164, 65, 74)
GREENS = (LEAF, MOSS, MINT, DEEP)
BROWNS = {TAN, MID, DARK, GOLD_BARK}

# Branch flare in degrees. Frames 5-10 are the spin, so the flare is 0 there.
# Frame 15 matches frame 0 so the 16-frame loop meets.
OPEN = [10, 14, 18, 16, 12, 0, 0, 0, 0, 0, 0, 0, 0, 8, 5, 10]
RING = [0, 1, 2, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]
FOLIAGE = ("foliage_l", "foliage_top", "foliage_r")
FEET = ("foot_l", "foot_r")


def flood_count(mask):
    return int(mask.sum())


def label_components(mask):
    """4-connected components. Returns label map starting at 1, and sizes."""
    h, w = mask.shape
    labels = np.zeros((h, w), np.int16)
    sizes = []
    ys, xs = np.nonzero(mask)
    nlab = 0
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if labels[y0, x0] or not mask[y0, x0]:
            continue
        nlab += 1
        stack = [(y0, x0)]
        labels[y0, x0] = nlab
        n = 0
        while stack:
            y, x = stack.pop()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = nlab
                    stack.append((ny, nx))
        sizes.append(n)
    return labels, sizes


def segment(img):
    """Three branch-balls, one pot (trunk included), two real feet."""
    op = img[..., 3] >= 128
    h, w = op.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = img[..., :3]
    green = np.zeros((h, w), bool)
    for col in GREENS:
        green |= op & np.all(rgb == np.array(col, np.uint8), axis=-1)
    labels, sizes = label_components(green)
    if len(sizes) != 3:
        raise SystemExit(f"expected 3 foliage balls, found {len(sizes)} sizes {sizes}")
    comps = []
    for lab, size in enumerate(sizes, start=1):
        ys, xs = np.nonzero(labels == lab)
        comps.append((int(ys.min()), float(xs.mean()), lab, size))
    comps.sort()
    top_lab = comps[0][2]
    sides = sorted(comps[1:], key=lambda t: t[1])
    name_of = {top_lab: "foliage_top", sides[0][2]: "foliage_l", sides[1][2]: "foliage_r"}

    ink = op & np.all(rgb == 0, axis=-1)
    touch = np.zeros((h, w), bool)
    gys, gxs = np.nonzero(green)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ys2 = gys + dy
            xs2 = gxs + dx
            ok = (ys2 >= 0) & (ys2 < h) & (xs2 >= 0) & (xs2 < w)
            touch[ys2[ok], xs2[ok]] = True
    outline = ink & touch & (yy <= 46)
    green_pts = {name: np.stack(np.nonzero(labels == lab), axis=1) for lab, name in name_of.items()}
    foliage = {name: (labels == lab) for lab, name in name_of.items()}
    oys, oxs = np.nonzero(outline)
    for y, x in zip(oys.tolist(), oxs.tolist()):
        best = None
        best_d = 10**9
        for name, pts in green_pts.items():
            d = (pts[:, 0] - y) ** 2 + (pts[:, 1] - x) ** 2
            dist = int(d.min())
            if dist < best_d:
                best_d = dist
                best = name
        foliage[best][y, x] = True

    foot_l = op & (yy >= 65) & (xx <= 40)
    foot_r = (op & (yy >= 67) & (xx >= 46)) | (op & (yy == 66) & (xx >= 47))
    if np.any(foot_l & foot_r):
        raise SystemExit("the two feet overlap")
    foliage_all = foliage["foliage_l"] | foliage["foliage_top"] | foliage["foliage_r"]
    if np.any(foliage_all & (foot_l | foot_r)):
        raise SystemExit("a branch ball was cut into a foot")
    pot = op & ~foliage_all & ~foot_l & ~foot_r
    parts = {
        "foliage_l": foliage["foliage_l"],
        "foliage_top": foliage["foliage_top"],
        "foliage_r": foliage["foliage_r"],
        "pot": pot,
        "foot_l": foot_l,
        "foot_r": foot_r,
    }
    own = np.zeros_like(op)
    overlap = 0
    for name in parts:
        overlap += int((own & parts[name]).sum())
        own |= parts[name]
    if int(own.sum()) != int(op.sum()) or overlap:
        raise SystemExit(
            f"masks do not partition the still: covered {int(own.sum())} opaque {int(op.sum())} overlap {overlap}"
        )
    if flood_count(parts["foot_l"]) < 12 or flood_count(parts["foot_r"]) < 20:
        raise SystemExit(f"a foot collapsed: L {flood_count(parts['foot_l'])} R {flood_count(parts['foot_r'])}")
    if min(flood_count(parts[n]) for n in FOLIAGE) < 70 or flood_count(pot) < 300:
        raise SystemExit("pot or a branch ball collapsed")
    # The right sole is the ground contact. The left sole stops higher.
    lys, lxs = np.nonzero(foot_l)
    rys, rxs = np.nonzero(foot_r)
    if int(rys.max()) != 72 or int(lys.max()) != 68:
        raise SystemExit(f"sole lines moved: L {int(lys.max())} R {int(rys.max())}")
    if int(lxs.min()) > 36 or int(rxs.max()) < 53:
        raise SystemExit("foot span does not match the still")
    return parts


def measure_face(img):
    """Both eyes and the 3px mouth are inside the pot. The still faces the viewer."""
    op = img[..., 3] >= 128
    rgb = img[..., :3]
    yellow = op & np.all(rgb == np.array(EYE, np.uint8), axis=-1)
    gold = op & np.all(rgb == np.array(PUPIL, np.uint8), axis=-1)
    red = op & np.all(rgb == np.array(MOUTH, np.uint8), axis=-1)
    eyes = yellow | gold
    n_eyes = int(eyes.sum())
    n_red = int(red.sum())
    if n_eyes < 16:
        raise SystemExit(f"expected both eyes, found {n_eyes} eye pixels")
    if n_red != 3:
        raise SystemExit(f"expected the 3px mouth, found {n_red}")
    ys, xs = np.nonzero(op)
    left, right = int(xs.min()), int(xs.max())
    eye_xs = np.nonzero(eyes)[1]
    if int(eye_xs.min()) <= left + 1 or int(eye_xs.max()) >= right - 1:
        raise SystemExit("an eye sits on the silhouette edge; facing must be remeasured")
    eys, exs = np.nonzero(eyes)
    # Two clusters: a single eye would sit in one tight span.
    if int(exs.max()) - int(exs.min()) < 6:
        raise SystemExit("only one eye cluster; facing must be remeasured")
    return {
        "eyes": n_eyes,
        "mouth": n_red,
        "centroid_x": float(xs.mean()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "eye_span": [int(exs.min()), int(eys.min()), int(exs.max()), int(eys.max())],
    }


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


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
    """4x nearest. A multiple of 360 stays exact. No abs(cos) squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    scale = 4
    up = np.repeat(np.repeat(layer, scale, axis=0), scale, axis=1)
    px, py = pivot[0] * scale + 1.5, pivot[1] * scale + 1.5
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
    return out[scale // 2::scale, scale // 2::scale].copy()


def despeckle(img):
    """Drop only pixels whose four neighbours are all empty. After rotation only."""
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
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    for i in range(0, len(ty), 400):
        sl = slice(i, i + 400)
        d = (ty[sl, None] - pts[None, :, 0]) ** 2 + (tx[sl, None] - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


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
    flat[sel] = mapped[inv]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    out[~a] = 0
    return out


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def edge_pivot(mask, mode):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0, 0.0
    if mode == "bottom":
        sel = ys >= ys.max() - 1
    elif mode == "top":
        sel = ys <= ys.min() + 1
    elif mode == "right":
        sel = xs >= xs.max() - 1
    elif mode == "left":
        sel = xs <= xs.min() + 1
    else:
        raise SystemExit(mode)
    return float(xs[sel].mean()), float(ys[sel].mean())


def side_of(mask, center_x):
    xs = np.nonzero(mask)[1]
    return -1.0 if float(xs.mean()) < center_x else 1.0


def color_coords(canvas, color):
    m = (canvas[..., 3] >= 128) & np.all(canvas[..., :3] == np.array(color, np.uint8), axis=-1)
    ys, xs = np.nonzero(m)
    return list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))


def build_rig(img, masks, flip):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]

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

    # A few trunk-coloured pixels under each socket. Hidden while the part is home.
    pivots = {
        "foliage_l": edge_pivot(placed["foliage_l"], "right"),
        "foliage_top": edge_pivot(placed["foliage_top"], "bottom"),
        "foliage_r": edge_pivot(placed["foliage_r"], "left"),
        "foot_l": edge_pivot(placed["foot_l"], "top"),
        "foot_r": edge_pivot(placed["foot_r"], "top"),
    }
    yy = np.arange(CELL_H)[:, None]
    xx = np.arange(CELL_W)[None, :]
    for name in FOLIAGE + FEET:
        px, py = pivots[name]
        zone = placed[name] & ((yy - py) ** 2 + (xx - px) ** 2 <= 2.25)
        fill = nearest_color(canvas, zone, placed["pot"])
        write = zone & (layers["pot"][..., 3] < 128)
        layers["pot"][write] = fill[write]

    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    eyes = color_coords(canvas, EYE) + color_coords(canvas, PUPIL)
    mouth = color_coords(canvas, MOUTH)
    if eyes:
        face = (float(sum(p[0] for p in eyes) / len(eyes)), float(sum(p[1] for p in eyes) / len(eyes)))
    else:
        face = center

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        placed = {k: v[:, ::-1].copy() for k, v in placed.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
        face = (CELL_W - 1 - face[0], face[1])
        eyes = [(CELL_W - 1 - x, y) for x, y in eyes]
        mouth = [(CELL_W - 1 - x, y) for x, y in mouth]
        canvas = canvas[:, ::-1].copy()

    sides = {name: side_of(placed[name], center[0]) for name in FOLIAGE + FEET}
    facing = 1.0 if not flip else -1.0
    return layers, pivots, sides, center, face, facing, canvas, eyes, mouth


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, sway_l=0.0, sway_top=0.0, sway_r=0.0,
        open=0.0, spin=0.0, recoil=0.0, ring=-1, burst=0, flash=0, sparks=0,
        step_l=0.0, step_r=0.0, hop_l=0.0, hop_r=0.0,
    )
    if action == "idle":
        p["sway_top"] = wave(f, n, 6.0, phase=0.4)
        p["sway_l"] = wave(f, n, 7.0, phase=1.1)
        p["sway_r"] = wave(f, n, 7.0, phase=2.2)
    elif action == "walk":
        # One stride: left stub, then right stub. Frame 0 is planted.
        lift_l = max(0.0, wave(f, n, 1.0, phase=0.0, cycles=1.0))
        lift_r = max(0.0, wave(f, n, 1.0, phase=math.pi, cycles=1.0))
        p["step_l"] = 46.0 * lift_l
        p["step_r"] = 46.0 * lift_r
        p["hop_l"] = lift_l
        p["hop_r"] = lift_r
        p["sway_top"] = wave(f, n, 8.0, phase=0.3, cycles=1.0)
        p["sway_l"] = wave(f, n, 9.0, phase=1.0, cycles=1.0)
        p["sway_r"] = wave(f, n, 9.0, phase=2.1, cycles=1.0)
    elif action == "attack":
        p["open"] = float(OPEN[f])
        p["ring"] = RING[f]
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
        if f in (11, 12):
            p["burst"] = 2 if f == 11 else 1
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.82, 0.55, 0.32, 0.16, 0.06, 0.0, 0.0, 0.0][f]
        p["recoil"] = 38.0 * env
        p["dx"] = -3.0 * env
        p["dy"] = -1.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def flare_face(pot, eyes, mouth, open_deg):
    """Widen the existing eyes and the 3px mouth. Recolor only. No new mouth."""
    if open_deg <= 0 or not eyes:
        return pot
    out = pot.copy()
    red = np.array(MOUTH, np.uint8)
    yel = np.array(EYE, np.uint8)
    for x, y in mouth:
        yy = y - 1
        if yy < 0:
            continue
        if out[yy, x, 3] < 128:
            continue
        col = tuple(int(v) for v in out[yy, x, :3])
        if col in BROWNS:
            out[yy, x, :3] = red
    if open_deg < 8:
        return out
    xs = [p[0] for p in eyes]
    mid = (min(xs) + max(xs)) / 2.0
    clusters = [[], []]
    for x, y in eyes:
        clusters[0 if x <= mid else 1].append((x, y))
    for cluster in clusters:
        if not cluster:
            continue
        x, y = max(cluster, key=lambda p: p[1])
        yy = y + 1
        if yy >= out.shape[0]:
            continue
        if out[yy, x, 3] < 128:
            continue
        col = tuple(int(v) for v in out[yy, x, :3])
        if col in BROWNS:
            out[yy, x, :3] = yel
    return out


def compose(layers, pivots, sides, facing, p, eyes, mouth):
    """Pot stays one stump. Balls pivot on the trunk. Only the two stubs step."""
    bdx, bdy = p["dx"] * facing, p["dy"]
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    pot = flare_face(layers["pot"], eyes, mouth, p["open"])
    paste(body, shift(pot, bdx, bdy))
    sway = {"foliage_l": p["sway_l"], "foliage_top": p["sway_top"], "foliage_r": p["sway_r"]}
    for name in ("foliage_top", "foliage_l", "foliage_r"):
        side = sides[name]
        if name == "foliage_top":
            ang = facing * (sway[name] + 0.55 * p["open"])
        else:
            # Positive angle swings a left-of-pivot mass up. -side opens outward.
            ang = facing * sway[name] + (-side) * p["open"]
        px, py = pivots[name]
        spun = rotate_nearest(layers[name], ang, (px, py))
        paste(body, shift(spun, bdx, bdy))
    for name, step_key, hop_key in (("foot_l", "step_l", "hop_l"), ("foot_r", "step_r", "hop_r")):
        side = sides[name]
        ang = (-side) * p[step_key]
        px, py = pivots[name]
        spun = rotate_nearest(layers[name], ang, (px, py))
        extra = -1.0 * p[hop_key]
        paste(body, shift(spun, bdx, bdy + extra))
    return body


def roar_rings(out, face, age):
    """Sparse rings outside the pot. Age 0 is one ring."""
    cx, cy = face
    cols = (MINT, EYE, PUPIL)
    radii = (22, 28, 34)
    steps = (24, 20, 18)
    for ri in range(age + 1):
        rad = radii[ri]
        step = steps[ri]
        col = cols[ri]
        deg = 0
        while deg < 360:
            ang = math.radians(deg + ri * 8)
            put(out, cx + rad * math.cos(ang), cy + rad * math.sin(ang) * 0.92, col)
            deg += step
    return out


def impact_burst(out, face, size):
    cx, cy = face
    ticks = [(20, -2), (-20, 2), (16, -12), (-16, -12), (18, 10), (-18, 8), (0, -22), (24, 4), (-24, -4)]
    col = EYE if size > 1 else PUPIL
    for dx, dy in ticks[: 6 + size]:
        put(out, cx + dx, cy + dy, col)
        if size > 1:
            put(out, cx + dx + (1 if dx >= 0 else -1), cy + dy, MINT)
    return out


def sparks(out, frame_idx, face):
    cx, cy = face
    d = frame_idx
    pts = [(cx - 10 - d, cy - 22), (cx + 10 + d, cy - 20), (cx + 2, cy - 26 - d)]
    for sx, sy in pts:
        put(out, sx, sy, EYE)
        put(out, sx, sy + 1, MINT)
    return out


def flash_eye(img):
    out = img.copy()
    a = out[..., 3] >= 128
    black = a & np.all(out[..., :3] == 0, axis=-1)
    out[a & ~black, :3] = EYE
    return out


def largest_fraction(img):
    a = img[..., 3] >= 128
    if not np.any(a):
        return 1.0, 0
    seen = np.zeros_like(a)
    h, w = a.shape
    best = 0
    total = int(a.sum())
    ys, xs = np.nonzero(a)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
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
                if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if n > best:
            best = n
    return best / total, total


def synth(rig, action, f, palette, effects=True):
    layers, pivots, sides, center, face, facing, _canvas, eyes, mouth = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        rest = pose(action, 0)
        rest = gen4_attack_strike.finish(rest, action, 0, __file__)

        rest["open"] = 0.0
        rest["sway_l"] = rest["sway_top"] = rest["sway_r"] = 0.0
        rest["dx"] = 0.0
        rest["dy"] = 0.0
        rest["step_l"] = rest["step_r"] = 0.0
        rest["hop_l"] = rest["hop_r"] = 0.0
        body = compose(layers, pivots, sides, facing, rest, eyes, mouth)
        if recoiling:
            body = shift(body, p["dx"] * facing, p["dy"])
            pivot = (center[0] + p["dx"] * facing, center[1] + p["dy"])
            body = rotate_nearest(body, -p["recoil"] * facing, pivot)
        else:
            body = rotate_nearest(body, p["spin"] * facing, center)
        body = despeckle(body)
    else:
        body = compose(layers, pivots, sides, facing, p, eyes, mouth)
        moving = p["open"] or abs(p["sway_l"]) > 0.45 or abs(p["sway_r"]) > 0.45 or abs(p["sway_top"]) > 0.45
        moving = moving or p["step_l"] > 0.45 or p["step_r"] > 0.45
        if moving:
            body = despeckle(body)

    img = snap(body, palette)
    if not effects:
        return img
    moved_face = (face[0] + p["dx"] * facing, face[1] + p["dy"])
    if p["flash"]:
        img = flash_eye(img)
    if p["ring"] >= 0 and not spinning:
        img = roar_rings(img, moved_face, p["ring"])
    if p["burst"]:
        img = impact_burst(img, face, p["burst"])
    if p["sparks"]:
        img = sparks(img, f, face)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (STILL_DIR / "front.png").read_bytes()
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
    samples = {"idle": (0, 4, 8, 12), "walk": (0, 4, 8, 12), "attack": (2, 6, 8, 11), "hurt": (1, 3, 6, 10)}
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            m = tile[..., 3] >= 128
            bg[ai * CELL_H:(ai + 1) * CELL_H, si * CELL_W:(si + 1) * CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    """South row, every frame, so the spin can be checked as one body."""
    pad = 4
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            a = tile[..., 3] >= 128
            if not np.any(a):
                continue
            ys, xs = np.nonzero(a)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(CELL_H, int(ys.max()) + 1 + pad)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(CELL_W, int(xs.max()) + 1 + pad)
            row.append(tile[y0:y1, x0:x1])
        crops.append(row)
    cell_h = max(c.shape[0] for row in crops for c in row)
    cell_w = max(c.shape[1] for row in crops for c in row)
    cols = max(len(row) for row in crops)
    bg = np.full((len(crops) * cell_h, cols * cell_w, 4), (15, 20, 31, 255), np.uint8)
    for ri, row in enumerate(crops):
        for ci, c in enumerate(row):
            y = ri * cell_h + (cell_h - c.shape[0]) // 2
            x = ci * cell_w + (cell_w - c.shape[1]) // 2
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


def south_attack_ratios(sheets):
    row = sheets["attack"][:CELL_H]
    counts = []
    for f in range(16):
        tile = row[:, f * CELL_W:(f + 1) * CELL_W]
        counts.append(int((tile[..., 3] == 255).sum()))
    base = counts[0]
    return counts, [c / base for c in counts]


def assert_claim():
    import os as _rebuild_os
    if _rebuild_os.environ.get("GEN4_ATTACK_REBUILD") == "1":
        return
    data = json.loads(QUEUE.read_text(encoding="utf-8"))
    item = next((i for i in data["items"] if i.get("id") == KEY), None)
    if item is None:
        raise SystemExit("queue has no 0438_bonsly")
    if item.get("status") != "in_progress" or item.get("claim") != "parallel-batch":
        raise SystemExit(f"stop: status {item.get('status')} claim {item.get('claim')}")


def with_flock(lock_path, fn):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def publish_dir8(sprite_h):
    entry = {
        "dex": 438,
        "form": "normal",
        "variant": "bonsly",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0438_bonsly/normal",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir",
        "nf": 16,
        "cw": CELL_W,
        "ch": CELL_H,
        "w": CELL_W,
        "h": int(sprite_h),
        "fy": FOOT_Y,
        "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1,
        "src": "agent-sprite-forge",
        "v": VERSION,
    }

    def edit():
        text = DIR8.read_text(encoding="utf-8")
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        raw = text[len(prefix):].strip()
        if raw.endswith(";"):
            raw = raw[:-1]
        data = json.loads(raw)
        before = {k: v for k, v in data.items() if k != KEY}
        data[KEY] = entry
        blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        out = prefix + blob + ";\n"
        parsed = json.loads(blob)
        if set(parsed) != set(before) | {KEY}:
            raise SystemExit("dir8 key set changed")
        for k, v in before.items():
            if parsed[k] != v:
                raise SystemExit(f"dir8 key {k} changed")
        if parsed[KEY] != entry:
            raise SystemExit("dir8 entry mismatch")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(out, encoding="utf-8")
        tmp.replace(DIR8)
        return len(parsed)

    return with_flock(DIR8_LOCK, edit)


def update_queue(status):
    def edit():
        fd = os.open(QUEUE, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with os.fdopen(os.dup(fd), "r", encoding="utf-8") as handle:
                data = json.loads(handle.read())
            found = False
            for item in data["items"]:
                if item.get("id") != KEY:
                    continue
                item["status"] = status
                item.pop("claim", None)
                found = True
                break
            if not found:
                raise SystemExit("queue has no 0438_bonsly")
            raw = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            tmp = QUEUE.with_suffix(".json.tmp")
            tmp.write_text(raw, encoding="utf-8")
            os.replace(tmp, QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    assert_claim()
    front = np.array(Image.open(STILL_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    face_info = measure_face(front)
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("facing viewer", face_info, "parts", counts, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}

    rest = synth(rigs["S"], "idle", 0, palette, effects=False)
    placed = rigs["S"][6]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")
    walk0 = synth(rigs["S"], "walk", 0, palette, effects=False)
    if np.any(walk0 != placed):
        raise SystemExit("walk frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "pot": (190, 160, 120),
        "foliage_top": (40, 200, 70),
        "foliage_l": (20, 140, 220),
        "foliage_r": (230, 200, 40),
        "foot_l": (220, 80, 60),
        "foot_r": (240, 120, 200),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "S", f) for f in (0, 4, 8, 12)]
        want += [("walk", "S", f) for f in (0, 2, 4, 8, 12)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in (0, 1, 3, 6, 9)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        base = int((synth(rigs["S"], "attack", 0, palette)[..., 3] == 255).sum())
        print("ATTACK_S_RATIOS", flush=True)
        ratios = []
        for f in range(16):
            tile = synth(rigs["S"], "attack", f, palette)
            n = int((tile[..., 3] == 255).sum())
            frac, _total = largest_fraction(tile)
            ratio = n / base
            ratios.append(ratio)
            print(f"attack S {f:02d} opaque {n} ratio {ratio:.3f} main {frac:.3f}", flush=True)
        print(f"min {min(ratios):.3f} max {max(ratios):.3f}", flush=True)
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
                Image.fromarray(synth(rigs[d], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    (STILL_DIR / "actions").mkdir(parents=True, exist_ok=True)
    preview.save(STILL_DIR / "actions" / "preview.png")
    contact = motion_contact(sheets)
    contact.save(OUT_DIR / "preview_motion.png")
    contact.save(work / "review_S.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    counts_s, ratios = south_attack_ratios(sheets)
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(counts_s, ratios)):
        print(f"attack S {f:02d} opaque {c} ratio {r:.3f}", flush=True)
    print(f"min {min(ratios):.3f} max {max(ratios):.3f}", flush=True)

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted({int(v) for v in np.unique(master[..., 3])})
    clip = {}
    spin_stats = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and d == "S" and 5 <= f <= 10:
                    frac, total = largest_fraction(tile)
                    spin_stats.append({"f": f, "opaque": total, "main": round(frac, 3)})
    bad_ratio = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
    report = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_h,
        "off_palette_colors": len(off),
        "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "attack_s_ratios": [round(r, 3) for r in ratios],
        "attack_s_min": round(min(ratios), 3),
        "attack_s_max": round(max(ratios), 3),
        "spin_body": spin_stats,
        "rest_diff": 0,
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind, extra in (
            ("idle", "idle", []),
            ("walk", "gait", ["--state", "walk"]),
        ):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            proc = subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids", *extra],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
            else:
                report[f"gait_{act}"] = f"no-report exit {proc.returncode}"
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1800], flush=True)

    bad_spin = [s for s in spin_stats if s["main"] < 0.90]
    if off or clip or alphas != [0, 255] or bad_ratio or bad_spin:
        raise SystemExit(f"qa failed off={len(off)} clip={clip} alphas={alphas} ratios={bad_ratio} spin={bad_spin}")

    meta = {
        "generator": "forge_bonsly_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 438,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED),
        "still_faces": "viewer",
        "archetype": "planted_bonsai_biped_stubs",
        "feet": {"left_sole_y": 68, "right_sole_y": 72, "count": 2},
        "branches": 3,
        "mouth_pixels": 3,
        "face": face_info,
        "parts": counts,
        "palette_colors": int(len(palette)),
        "off_palette_colors": len(off),
        "attack_s_ratios": [round(r, 3) for r in ratios],
        "attack_s_min": round(min(ratios), 3),
        "attack_s_max": round(max(ratios), 3),
        "source": "front.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        print("dir8 keys", publish_dir8(sprite_h), flush=True)
        update_queue("done")
        print("queue done", KEY, flush=True)


if __name__ == "__main__":
    main()
