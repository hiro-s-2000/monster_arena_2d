#!/usr/bin/env python3
"""Lickilicky (0463) normal 8-direction action sheet from this form's front.png.

Measured on normal/front.png: 96x96, 2232 opaque, bbox x17-78 y15-80 (62x66),
15 colours, alpha already 0 or 255. Large biped. Exactly two feet, both y>=75:
left sole x31-41 y75-78 (34px), right sole x44-64 y75-80 (66px).
Tongue is one 445px piece (orange spiral plus its cream/white curl), centroid
x36.6 against the body centroid x46.4, so the still faces left. It is not a leg.
Eyes are two white pixels (40,29) and (52,30). The mouth is the body rim where
the tongue meets the face. No extra limbs.

Unflipped dirs are N NW W SW. S SE E NE are mirrored. E faces right.

  idle   : feet planted. Tongue curls a few degrees around the mouth and stays
           one piece. Frame 0 is the still. No body drift.
  walk   : the two soles alternate. Tongue stays attached.
  attack : 0-4 open the existing mouth, stretch the existing tongue, plus rings.
           5-10 one-piece in-plane 360. 11-12 impact. 13-15 the still.
  hurt   : one-piece lean, at most 38 degrees. No |cos| squash.

Palette-locked to this still. front/back png and gif are not written.
"""

import argparse
import fcntl
import json
import math
import os
import shutil
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0463_lickilicky/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0463_lickilicky/normal/agent-sprite-forge"
)
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 192, 192, 168
VERSION = "lickilicky_v3"
KEY = "0463_lickilicky"

PINK = (230, 115, 148)
DPNK = (189, 82, 115)
ORNG = (246, 123, 90)
INK = (16, 16, 16)
SHAD = (148, 65, 98)
YELL = (255, 230, 123)
WHITE = (255, 255, 255)
DARK = (98, 41, 57)
RED = (197, 90, 90)
LIP = (255, 164, 180)
CREAM = (213, 205, 222)
GOLD = (205, 172, 74)
DRED = (148, 57, 65)
GRAY = (139, 139, 172)
LORG = (255, 180, 139)
PALETTE_COLS = (
    PINK, DPNK, ORNG, INK, SHAD, YELL, WHITE, DARK, RED, LIP, CREAM, GOLD, DRED, GRAY, LORG,
)
TONGUE_COLS = (ORNG, RED, DRED, LORG)
HILITE_COLS = (WHITE, CREAM, GRAY, LIP, INK)

DRAW = ("torso", "foot_l", "foot_r", "tongue")
RING = ((8, -7), (11, -3), (9, 4), (13, 8), (16, -5), (19, 1), (15, 6), (22, -2), (6, 1))
BURST = ((7, -8), (10, -4), (8, 5), (14, 7), (18, -2), (21, 3), (12, 2), (17, -6), (5, -3))


def flood(seeds, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow, dtype=bool)
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


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def colored(img, cols):
    rgb = img[..., :3]
    a = img[..., 3] == 255
    hit = np.zeros(a.shape, dtype=bool)
    for col in cols:
        hit |= np.all(rgb == np.array(col, np.uint8), axis=-1)
    return a & hit


def component_sizes(mask):
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    sizes = []
    ys, xs = np.nonzero(mask)
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
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes


def segment(img):
    """Tongue, two soles, torso. The tongue is not a foot and not a second limb."""
    h, w = img.shape[:2]
    a = img[..., 3] == 255
    if img.shape[0] != 96 or img.shape[1] != 96:
        raise SystemExit("front is not 96x96")
    if int(a.sum()) != 2232:
        raise SystemExit(f"still opaque changed: {int(a.sum())}")
    yy, xx = np.mgrid[0:h, 0:w]
    ys, xs = np.nonzero(a)
    if int(xs.min()) != 17 or int(xs.max()) != 78 or int(ys.min()) != 15 or int(ys.max()) != 80:
        raise SystemExit("still bbox changed; remeasure before forging")
    tongue_cols = colored(img, TONGUE_COLS)
    zone = (yy >= 32) & (yy <= 56) & (xx <= 56)
    hilite = colored(img, HILITE_COLS) & zone
    tongue = flood(tongue_cols, tongue_cols | hilite)
    if int(tongue.sum()) != 445 or component_sizes(tongue) != [445]:
        raise SystemExit(f"tongue mask changed: {int(tongue.sum())} {component_sizes(tongue)}")
    feet = a & (yy >= 75) & ~tongue
    sizes = component_sizes(feet)
    if sizes != [66, 34]:
        raise SystemExit(f"expected 2 feet, got {sizes}")
    foot_l = feet & (xx <= 42)
    foot_r = feet & (xx >= 43)
    if int(foot_l.sum()) != 34 or int(foot_r.sum()) != 66:
        raise SystemExit(f"foot split changed: {int(foot_l.sum())} {int(foot_r.sum())}")
    if component_sizes(foot_l) != [34] or component_sizes(foot_r) != [66]:
        raise SystemExit("a foot is not one piece")
    torso = a & ~tongue & ~foot_l & ~foot_r
    if int(torso.sum()) != 1687 or component_sizes(torso) != [1687]:
        raise SystemExit(f"torso changed: {int(torso.sum())} {component_sizes(torso)}")
    eyes = colored(img, (WHITE,)) & (yy < 32)
    if int(eyes.sum()) != 2:
        raise SystemExit(f"expected 2 eye pixels above the tongue, got {int(eyes.sum())}")
    tongue_x = float(np.nonzero(tongue)[1].mean())
    body_x = float(xs.mean())
    if not tongue_x < body_x - 5:
        raise SystemExit(f"still is not facing left: tongue {tongue_x:.1f} body {body_x:.1f}")
    tymin = int(np.nonzero(tongue)[0].min())
    rim = dilate(tongue, 1) & torso & (yy <= tymin + 8)
    if int(rim.sum()) < 8:
        raise SystemExit(f"mouth rim missing: {int(rim.sum())}")
    parts = {"tongue": tongue, "foot_l": foot_l, "foot_r": foot_r, "torso": torso, "rim": rim}
    print(
        f"still {w}x{h} opaque {int(a.sum())} bbox x={int(xs.min())}-{int(xs.max())} "
        f"y={int(ys.min())}-{int(ys.max())} bh={int(ys.max() - ys.min() + 1)} "
        f"facing left feet 2 tongue {int(tongue.sum())} mouth_rim {int(rim.sum())} "
        f"tongue_x {tongue_x:.1f} body_x {body_x:.1f}",
        flush=True,
    )
    return parts


def edge_pivot(mask, side):
    ys, xs = np.nonzero(mask)
    if side == "right":
        sel = xs >= xs.max() - 1
    elif side == "left":
        sel = xs <= xs.min() + 1
    elif side == "top":
        sel = ys <= ys.min() + 1
    else:
        sel = ys >= ys.max() - 1
    return float(xs[sel].mean()), float(ys[sel].mean())


def extract_palette(img):
    a = img[..., 3] == 255
    return np.unique(img[a][:, :3], axis=0)


def require_colors(palette):
    if len(palette) != 15:
        raise SystemExit(f"expected 15 colours, got {len(palette)}")
    for col in PALETTE_COLS:
        if not np.any(np.all(palette == np.array(col, np.uint8), axis=1)):
            raise SystemExit(f"still is missing measured colour {col}")


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    for i in range(0, len(ty), 300):
        sl = slice(i, i + 300)
        d = (ty[sl, None] - pts[None, :, 0]) ** 2 + (tx[sl, None] - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def build_rig(img, masks, flip):
    a = img[..., 3] == 255
    # v3: the 445px "tongue" flood also caught the white chest fur, the cream lip and
    # both hands' claws. Only the orange/red tongue and its own ink outline move.
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    core = (masks["tongue"] & (colored(img, TONGUE_COLS) | colored(img, (INK,)))
            & (yy <= 48) & ~((xx <= 21) & (yy >= 44)))
    masks = dict(masks)
    masks["torso"] = masks["torso"] | (masks["tongue"] & ~core)
    masks["tongue"] = core
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    crop = img[y0:y1, x0:x1].copy()
    crop[crop[..., 3] == 0] = 0
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = crop
    placed = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), dtype=bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm
    layers = {}
    for name in DRAW:
        lyr = np.zeros_like(canvas)
        lyr[placed[name]] = canvas[placed[name]]
        layers[name] = lyr
    # Chest colour under the tongue, covered exactly at rest, so a curl cannot open a hole.
    # v3: only where the body really is behind the tongue (torso on both sides of the
    # pixel in its row and its column). Outside that the bent tongue leaves air, not a
    # pink patch that made the tongue look detached.
    under = nearest_color(canvas, placed["tongue"], placed["torso"])
    tor = placed["torso"]
    inside = (np.maximum.accumulate(tor, axis=1) & np.maximum.accumulate(tor[:, ::-1], axis=1)[:, ::-1]
              & np.maximum.accumulate(tor, axis=0) & np.maximum.accumulate(tor[::-1], axis=0)[::-1])
    keep = placed["tongue"] & inside
    layers["torso"][keep] = under[keep]

    oys, oxs = np.nonzero(canvas[..., 3] == 255)
    center = (float(oxs.mean()), float(oys.mean()))
    yb = int(oys.max())
    sole = (float(oxs[oys >= yb - 1].mean()), float(yb))
    pivots = {"tongue": edge_pivot(placed["tongue"], "right")}
    # Tongue root = the end that sits in the mouth (right end in the left-facing still).
    tys, txs = np.nonzero(placed["tongue"])
    rx = int(txs.max())
    root = (float(rx), float(tys[txs >= rx - 3].mean()))
    tongue_len = float(rx - txs.min())
    soles = {}
    for name in ("foot_l", "foot_r"):
        sy, sx = np.nonzero(placed[name])
        yb = int(sy.max())
        soles[name] = [(int(x), yb) for x in sx[sy == yb]]
    rim = placed["rim"]
    facing = -1.0
    if flip:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        rim = np.ascontiguousarray(rim[:, ::-1])

        def mx(x):
            return CELL_W - 1 - x

        pivots = {k: (mx(x), y) for k, (x, y) in pivots.items()}
        root = (mx(root[0]), root[1])
        center = (mx(center[0]), center[1])
        sole = (mx(sole[0]), sole[1])
        soles = {k: [(mx(x), y) for x, y in pts] for k, pts in soles.items()}
        facing = 1.0
    rest = compose({k: layers[k] for k in DRAW})
    if np.any(rest != canvas):
        raise SystemExit("rest pose does not match the still")
    return {
        "layers": layers,
        "pivots": pivots,
        "center": center,
        "sole": sole,
        "facing": facing,
        "rest": canvas,
        "soles": soles,
        "rim": rim,
        "root": root,
        "tongue_len": tongue_len,
        "body_n": int((canvas[..., 3] == 255).sum()),
        "counts": {k: int(np.count_nonzero(layers[k][..., 3] == 255)) for k in DRAW},
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


def stretch(layer, dx):
    """Paint the same tongue along the shift so a thin curl cannot split."""
    dx_i = int(round(dx))
    if dx_i == 0:
        return layer.copy()
    step = 1 if dx_i > 0 else -1
    out = np.zeros_like(layer)
    for s in range(0, dx_i + step, step):
        moved = shift(layer, s, 0)
        m = moved[..., 3] == 255
        out[m] = moved[m]
    return out


def bend_tongue(layer, curl_deg, extend, root, length, facing):
    """v3: bend the tongue like a flexible strip. The root in the mouth never moves;
    the bend grows toward the tip, so the tongue cannot float off the face."""
    if abs(curl_deg) < 0.05 and abs(extend) < 0.05:
        return layer.copy()
    a = layer[..., 3] == 255
    ys, xs = np.nonzero(a)
    cols = layer[ys, xs]
    rx, ry = root
    out = np.zeros_like(layer)
    h, w = layer.shape[:2]
    subs = (-0.34, 0.0, 0.34)
    for oy in subs:
        for ox in subs:
            x = xs + ox
            y = ys + oy
            d = (x - rx) * facing
            t = np.clip(d / max(length, 1.0), 0.0, 1.0)
            wgt = np.clip((t - 0.12) / 0.88, 0.0, 1.0) ** 1.4
            th = np.radians(curl_deg) * wgt * -facing
            vx, vy = x - rx, y - ry
            nx = rx + vx * np.cos(th) - vy * np.sin(th) + facing * extend * wgt
            ny = ry + vx * np.sin(th) + vy * np.cos(th)
            ix = np.round(nx).astype(int)
            iy = np.round(ny).astype(int)
            ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            out[iy[ok], ix[ok]] = cols[ok]
    return out


def rotate_nearest(layer, angle_deg, pivot):
    """4x inverse rotate plus a forward splat. 0 and 360 stay exact. No |cos| squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.05 or not np.any(layer[..., 3] == 255):
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
    inv = out[scale // 2::scale, scale // 2::scale].copy()
    opaque = layer[..., 3] == 255
    ys, xs = np.nonzero(opaque)
    dx = xs.astype(np.float64) - pivot[0]
    dy = ys.astype(np.float64) - pivot[1]
    dest_x = np.rint(pivot[0] + c * dx - s * dy).astype(np.int32)
    dest_y = np.rint(pivot[1] + s * dx + c * dy).astype(np.int32)
    h2, w2 = layer.shape[:2]
    ok = (dest_x >= 0) & (dest_x < w2) & (dest_y >= 0) & (dest_y < h2)
    fwd = np.zeros_like(layer)
    fwd[dest_y[ok], dest_x[ok]] = layer[ys[ok], xs[ok]]
    missing = (fwd[..., 3] == 255) & (inv[..., 3] == 0)
    inv[missing] = fwd[missing]
    return inv


def despeckle(img):
    out = img.copy()
    a = out[..., 3] == 255
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img, target):
    out = img.copy()
    h, w = out.shape[:2]
    for need in (4, 3):
        a = out[..., 3] == 255
        have = int(a.sum())
        if have >= target:
            break
        p = np.pad(a, 1)
        n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        holes = (~a) & (n >= need)
        ys, xs = np.nonzero(holes)
        if len(ys) == 0:
            continue
        room = target - have
        if len(ys) > room:
            ys, xs = ys[:room], xs[:room]
        for y, x in zip(ys.tolist(), xs.tolist()):
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and out[ny, nx, 3] == 255:
                    out[y, x] = out[ny, nx]
                    break
    return out


def bridge(img, min_size=9, max_gap=4):
    """Join a rotation nick to the body. Do not draw a long new limb."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(6):
        a = out[..., 3] == 255
        seen = np.zeros_like(a)
        comps = []
        ys, xs = np.nonzero(a)
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
                    if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(pts)
        comps.sort(key=len, reverse=True)
        big = [c for c in comps if len(c) >= min_size]
        if len(big) <= 1:
            return out
        main = big[0]
        best = None
        main_xy = np.array(main, np.int32)
        for other in big[1:]:
            o_xy = np.array(other, np.int32)
            # subsample so a fat body does not dominate the join search
            step_m = max(1, len(main_xy) // 400)
            step_o = max(1, len(o_xy) // 200)
            aa = main_xy[::step_m]
            bb = o_xy[::step_o]
            d = (aa[:, None, 0] - bb[None, :, 0]) ** 2 + (aa[:, None, 1] - bb[None, :, 1]) ** 2
            flat = int(d.argmin())
            i, j = divmod(flat, bb.shape[0])
            dist = float(d[i, j]) ** 0.5
            if best is None or dist < best[0]:
                best = (dist, tuple(aa[i]), tuple(bb[j]))
        if best is None or best[0] > max_gap:
            return out
        y0, x0 = best[1]
        y1, x1 = best[2]
        n = int(max(abs(y1 - y0), abs(x1 - x0)))
        col = out[y0, x0].copy()
        for t in range(n + 1):
            y = int(round(y0 + (y1 - y0) * t / max(n, 1)))
            x = int(round(x0 + (x1 - x0) * t / max(n, 1)))
            if out[y, x, 3] == 0:
                out[y, x] = col
    return out


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] == 255
    if not np.any(a):
        out[...] = 0
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


def paste(dst, layer):
    m = layer[..., 3] == 255
    dst[m] = layer[m]


def compose(layers):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in DRAW:
        paste(img, layers[name])
    return img


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        curl=0.0, lift_l=0.0, lift_r=0.0, fwd_l=0.0, fwd_r=0.0,
        extend=0, spin=0.0, recoil=0.0, knock=0.0, lunge=0,
        rings=0, burst=0, flash=0, sparks=0, mouth=0,
    )
    if action == "idle":
        p["curl"] = wave(f, n, 8.0)
    elif action == "walk":
        step = wave(f, n, 1.0, cycles=2.0)
        p["lift_l"] = max(0.0, step) * 3.0
        p["lift_r"] = max(0.0, -step) * 3.0
        p["fwd_l"] = max(0.0, step) * 2.0
        p["fwd_r"] = max(0.0, -step) * 2.0
        p["curl"] = wave(f, n, 5.0, cycles=2.0)
    elif action == "attack":
        # v3 lick: roar with the mouth open and the tongue reared up, then lash it
        # forward and down (root stays in the mouth), impact, and roll it back.
        curl = (0, 6, 12, 16, 18, 10, -4, -14, -18, -16, -10, -6, -3, -1, 0, 0)
        ext = (0, 1, 1, 2, 2, 4, 7, 9, 9, 8, 6, 4, 2, 1, 0, 0)
        p["curl"] = float(curl[f])
        p["extend"] = ext[f]
        if f <= 4:
            p["mouth"] = 1
            p["rings"] = 1
        elif f in (11, 12):
            p["burst"] = 1
            p["lunge"] = 2
    else:
        env = math.sin(math.pi * f / 11.0)
        p["recoil"] = 38.0 * env
        p["knock"] = 3.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def tip_point(layer, facing):
    a = layer[..., 3] == 255
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return None
    if facing < 0:
        edge = int(xs.min())
        sel = xs <= edge + 1
    else:
        edge = int(xs.max())
        sel = xs >= edge - 1
    return float(xs[sel].mean()), float(ys[sel].mean())


def dots(out, origin, facing, table, age=0):
    if origin is None:
        return out
    ox, oy = origin
    bump = age % 2
    cols = (WHITE, YELL, ORNG)
    for i, (dx, dy) in enumerate(table):
        put(out, ox + facing * (dx + bump), oy + dy, cols[i % 3])
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] == 255
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def sparks(out, facing, frame_idx):
    a = out[..., 3] == 255
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return out
    x = int(xs.max() if facing < 0 else xs.min())
    y = int(np.median(ys))
    for i, col in enumerate((WHITE, YELL, ORNG)):
        put(out, x - facing * (5 + i * 3 + frame_idx), y - 6 + i * 4, col)
    return out


def open_mouth(img, rim):
    m = rim & (img[..., 3] == 255)
    img[m, :3] = DARK
    img[m, 3] = 255
    return img


def synth(rig, action, f, palette):
    layers = rig["layers"]
    facing = rig["facing"]
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = (p["spin"] % 360.0) > 0.5
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        if spinning:
            img = rotate_nearest(rig["rest"], p["spin"], rig["center"])
        else:
            img = rotate_nearest(rig["rest"], facing * p["recoil"], rig["sole"])
            img = shift(img, -facing * p["knock"], 0)
        img = despeckle(img)
        img = fill_pinholes(img, rig["body_n"])
        img = bridge(img)
        tongue_layer = None
    else:
        tongue = layers["tongue"]
        if abs(p["curl"]) > 0.05 or p["extend"]:
            tongue = bend_tongue(tongue, p["curl"], float(p["extend"]),
                                 rig["root"], rig["tongue_len"], facing)
        moved = {
            "torso": layers["torso"],
            "foot_l": shift(layers["foot_l"], p["fwd_l"] * facing, -p["lift_l"]),
            "foot_r": shift(layers["foot_r"], p["fwd_r"] * facing, -p["lift_r"]),
            "tongue": tongue,
        }
        img = compose(moved)
        img = bridge(img)
        if p["mouth"]:
            img = open_mouth(img, rig["rim"])
        if p["lunge"]:
            img = shift(img, facing * float(p["lunge"]), 0)
        tongue_layer = tongue
    img = snap(img, palette)
    if p["flash"]:
        img = flash_white(img)
    if p["rings"] and tongue_layer is not None:
        img = dots(img, tip_point(tongue_layer, facing), facing, RING, age=f)
    if p["burst"]:
        img = dots(img, tip_point(rig["layers"]["tongue"], facing), facing, BURST, age=0)
        if p["lunge"]:
            img = shift(img, 0, 0)
    if p["sparks"] and recoiling:
        img = sparks(img, facing, f)
    return snap(img, palette)


def soles_hold(img, pts):
    return all(img[y, x, 3] == 255 for x, y in pts)


def one_sheet(rigs, palette, action, only=None):
    width = max(FRAMES.values()) * CELL_W
    sheet = np.zeros((8 * CELL_H, width, 4), np.uint8)
    dirs = DIRS if only is None else [only]
    for d in dirs:
        di = DIRS.index(d)
        for f in range(FRAMES[action]):
            sheet[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = synth(
                rigs[d], action, f, palette
            )
    return sheet


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


def contact_sheet(sheets, row):
    probe = sheets["idle"][row * CELL_H:(row + 1) * CELL_H, :CELL_W]
    ys, xs = np.nonzero(probe[..., 3] == 255)
    pad = 28
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad)
    scale = 2
    cw, ch = (x1 - x0) * scale, (y1 - y0) * scale
    cols = 8
    blocks = []
    for act in ACTIONS:
        band = sheets[act][row * CELL_H:(row + 1) * CELL_H]
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
            y, x = (r + rr) * ch, cc * cw
            m = up[..., 3] == 255
            bg[y:y + ch, x:x + cw][m] = up[m]
        r += math.ceil(len(frames) / cols)
    return Image.fromarray(bg, "RGBA")


def south_attack_ratios(sheets):
    base_tile = sheets["attack"][:CELL_H, :CELL_W]
    base = int((base_tile[..., 3] == 255).sum())
    ratios = []
    counts = []
    for f in range(FRAMES["attack"]):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        n = int((tile[..., 3] == 255).sum())
        counts.append(n)
        ratios.append(n / base if base else 0.0)
    return counts, ratios


def large_parts(img, min_size=9):
    """Return (size, has_tongue_orange, has_body_pink) for 4-connected pieces."""
    a = img[..., 3] == 255
    seen = np.zeros_like(a)
    h, w = a.shape
    out = []
    ys, xs = np.nonzero(a)
    orng = np.all(img[..., :3] == np.array(ORNG, np.uint8), axis=-1)
    pink = np.all(img[..., :3] == np.array(PINK, np.uint8), axis=-1)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        n = 0
        has_o = False
        has_p = False
        while stack:
            y, x = stack.pop()
            n += 1
            has_o = has_o or bool(orng[y, x])
            has_p = has_p or bool(pink[y, x])
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if n >= min_size:
            out.append((n, has_o, has_p))
    out.sort(reverse=True)
    return out


def print_ratios(counts, ratios, sheets):
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(counts, ratios)):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        parts = large_parts(tile)
        print(
            f"attack S {f:02d} opaque {c} ratio {r:.4f} large {parts}",
            flush=True,
        )
    print(f"min {min(ratios):.4f} max {max(ratios):.4f}", flush=True)


def sprite_box(tile):
    ys, xs = np.nonzero(tile[..., 3] == 255)
    if len(xs) == 0:
        return 0, 0
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def qa_report(master, sheets, palette, ratios):
    opaque = master[..., 3] == 255
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    off = []
    if np.any(opaque):
        cols = np.unique(master[opaque][:, :3], axis=0)
        pal = {tuple(int(c) for c in row) for row in palette}
        off = [tuple(int(c) for c in row) for row in cols if tuple(int(c) for c in row) not in pal]
    clip = 0
    for act in ACTIONS:
        for di in range(8):
            for f in range(FRAMES[act]):
                tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = tile[..., 3] == 255
                if not np.any(a):
                    continue
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip += 1
    bad_ratio = [round(r, 4) for r in ratios if r < 0.90 or r > 1.08]
    split = []
    for f in (0, 5, 6, 7, 8, 9, 10):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        parts = large_parts(tile)
        if len(parts) != 1 or not parts[0][1] or not parts[0][2]:
            split.append({"f": f, "parts": parts})
    idle0 = sheets["idle"][:CELL_H, :CELL_W]
    bw, bh = sprite_box(idle0)
    narrow = []
    for f in range(5, 11):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        w, h = sprite_box(tile)
        if w < 45 or h < 40:
            narrow.append({"f": f, "w": w, "h": h})
    return {
        "alphas": alphas,
        "off_palette": len(off),
        "clip_frames": clip,
        "sprite_h": bh,
        "sprite_w": bw,
        "bad_ratio": bad_ratio,
        "split": split,
        "narrow": narrow,
    }


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
        "dex": 463,
        "form": "normal",
        "variant": "lickilicky",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0463_lickilicky/normal",
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
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        raw = text.strip()
        if not raw.startswith(prefix) or "\n" in raw:
            raise SystemExit("dir8.js is not a single assignment line")
        body = raw[len(prefix):]
        if not body.endswith(";"):
            raise SystemExit("dir8.js missing trailing semicolon")
        body = body[:-1]
        data = json.loads(body)
        if KEY not in data:
            raise SystemExit("dir8 has no 0463_lickilicky")
        before = {k: data[k] for k in data}
        data[KEY] = entry
        out = prefix + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
        if out.count("\n") != 1:
            raise SystemExit("dir8.js picked up a line break")
        parsed = json.loads(out[len(prefix):].rstrip().rstrip(";"))
        if parsed[KEY] != entry:
            raise SystemExit("dir8 entry did not stick")
        for k, old in before.items():
            if k == KEY:
                continue
            if parsed[k] != old:
                raise SystemExit(f"dir8 key changed: {k}")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(out)
        tmp.replace(DIR8)
        return True

    return with_flock(DIR8_LOCK, edit)


def update_queue(status):
    def edit():
        fd = os.open(QUEUE, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with os.fdopen(os.dup(fd), "r+", encoding="utf-8") as handle:
                raw = handle.read()
                data = json.loads(raw)
                found = False
                for item in data["items"]:
                    if item.get("id") != KEY:
                        continue
                    item["status"] = status
                    item.pop("claim", None)
                    found = True
                if not found:
                    raise SystemExit("queue has no 0463_lickilicky")
                before = json.loads(raw)
                for b, a in zip(before["items"], data["items"]):
                    if b.get("id") == KEY:
                        continue
                    if b != a:
                        raise SystemExit(f"queue neighbor changed: {b.get('id')}")
                text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
                after = json.loads(text)
                for b, a in zip(before["items"], after["items"]):
                    if b.get("id") == KEY:
                        if a.get("status") != status or "claim" in a:
                            raise SystemExit("queue status did not stick")
                        continue
                    if b != a:
                        raise SystemExit(f"queue neighbor changed: {b.get('id')}")
                tmp = QUEUE.with_suffix(".json.tmp")
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def sync_arena():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    copied = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            copied += 1
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if set(np.unique(front[..., 3]).tolist()) - {0, 255}:
        raise SystemExit("front alpha is not binary")
    palette = extract_palette(front)
    require_colors(palette)
    masks = segment(front)
    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    if int((rigs["W"]["rest"][..., 3] == 255).sum()) != 2232:
        raise SystemExit("placed still count changed")
    for d, rig in rigs.items():
        idle0 = synth(rig, "idle", 0, palette)
        if np.any(idle0 != rig["rest"]):
            raise SystemExit(f"idle frame 0 does not match the still ({d})")
        if np.any(synth(rig, "walk", 0, palette) != rig["rest"]):
            raise SystemExit(f"walk frame 0 does not match the still ({d})")
        if np.any(synth(rig, "hurt", 0, palette) != rig["rest"]):
            raise SystemExit(f"hurt frame 0 does not match the still ({d})")
    if max(pose("hurt", f)["recoil"] for f in range(12)) > 38.001:
        raise SystemExit("recoil exceeds 38")
    for f in range(16):
        step = pose("walk", f)
        if step["lift_l"] > 0.5 and step["lift_r"] > 0.5:
            raise SystemExit(f"both feet lifted on walk {f}")
        idle = synth(rigs["W"], "idle", f, palette)
        if not soles_hold(idle, rigs["W"]["soles"]["foot_l"]):
            raise SystemExit(f"left foot left the ground on idle {f}")
        if not soles_hold(idle, rigs["W"]["soles"]["foot_r"]):
            raise SystemExit(f"right foot left the ground on idle {f}")
        walked = synth(rigs["W"], "walk", f, palette)
        if step["lift_l"] < 0.5 and not soles_hold(walked, rigs["W"]["soles"]["foot_l"]):
            raise SystemExit(f"left sole slipped on walk {f}")
        if step["lift_r"] < 0.5 and not soles_hold(walked, rigs["W"]["soles"]["foot_r"]):
            raise SystemExit(f"right sole slipped on walk {f}")

    if args.samples:
        sheets = {act: one_sheet(rigs, palette, act, only="S") for act in ACTIONS}
    else:
        sheets = {act: one_sheet(rigs, palette, act) for act in ACTIONS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts, ratios = south_attack_ratios(sheets)
    print_ratios(counts, ratios, sheets)
    if not args.samples:
        master = write_outputs(sheets)
    else:
        master = np.concatenate([sheets[a][:CELL_H] for a in ACTIONS], axis=0)
    preview = contact_sheet(sheets, 0)
    preview.save(OUT_DIR / "preview_motion.png")
    preview.save(OUT_DIR / "preview.png")
    report = qa_report(master if not args.samples else np.concatenate(
        [sheets[a] for a in ACTIONS], axis=0
    ) if False else master, sheets, palette, ratios)
    # samples master is south-only stacked; full qa clip scan expects 8 dirs.
    if args.samples:
        report["clip_frames"] = 0
        for act in ACTIONS:
            for f in range(FRAMES[act]):
                tile = sheets[act][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = tile[..., 3] == 255
                if np.any(a) and (a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any()):
                    report["clip_frames"] += 1
        opaque = master[..., 3] == 255
        report["alphas"] = sorted(int(v) for v in np.unique(master[..., 3]))
        pal = {tuple(int(c) for c in row) for row in palette}
        cols = np.unique(master[opaque][:, :3], axis=0) if np.any(opaque) else []
        report["off_palette"] = sum(
            1 for row in cols if tuple(int(c) for c in row) not in pal
        )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    failed = (
        report["off_palette"] or report["clip_frames"] or report["alphas"] != [0, 255]
        or report["bad_ratio"] or report["split"] or report["narrow"]
    )
    if failed:
        if args.publish and not args.samples:
            update_queue("needs_fix")
        raise SystemExit(f"qa failed {report}")
    if args.samples:
        print("samples ok", flush=True)
        return
    meta = {
        "generator": "forge_lickilicky_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 463,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": report["sprite_h"],
        "frames": FRAMES,
        "directions": DIRS,
        "unflipped": sorted(UNFLIPPED),
        "still_faces": "left",
        "feet": 2,
        "tongue_pixels": 445,
        "tongue_is_leg": False,
        "palette_colors": 15,
        "off_palette_colors": 0,
        "attack_s_ratios": [round(r, 4) for r in ratios],
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.publish:
        copied = sync_arena()
        publish_dir8(report["sprite_h"])
        update_queue("done")
        arena_bytes = (SYNC_OUT / "sheet.png").stat().st_size
        print(f"synced {copied} arena_sheet_bytes {arena_bytes}", flush=True)
        print(f"dir8 v {VERSION}", flush=True)
    print(
        f"h {report['sprite_h']} cw {CELL_W} ch {CELL_H} fy {FOOT_Y} feet 2 facing left",
        flush=True,
    )


if __name__ == "__main__":
    main()
