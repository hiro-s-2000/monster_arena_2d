#!/usr/bin/env python3
"""Gastrodon (0423) west-sea normal, from this front.png only.

Measured on normal/front.png (96x96, alpha already 0 or 255, 14 colours):
  opaque bbox x23-71 y21-74 (49x54). Centroid about (43.5, 51.3).
  One connected component. Faces left: both eyestalks and the cheek sit
  left of the brown shell. Lowest opaque row is y74 x44-47, the belly
  contact. The scallops under the skirt are the mantle edge, not feet.
  No mouth pixels. Roar flares the two existing eyestalks and draws rings.
  The shell is the brown mass from about y42 and stays on the body.
  Walk is a row-wise crawl. The bottom three rows stay planted.
  Attack flares the eyestalks and draws speed lines. The body does not spin.
  front/back png and gif are not written.
"""

import argparse
import fcntl
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SPECIES = ROOT / "monsterSprite/gen4/0423_gastrodon/normal"
OUT_DIR = SPECIES / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0423_gastrodon/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left, so east-side directions are the mirror. E looks screen-right.
FACE_RIGHT = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 184, 176, 155
VERSION = "gastrodon_v2"
KEY = "0423_gastrodon"

BLACK = (16, 16, 16)
PINK = (255, 139, 197)
GOLD = (164, 139, 57)
YELLOW = (255, 222, 115)
WHITE = (255, 255, 255)

CFG = {"flare_max": 2, "rings": True, "lines": True, "burst": True}


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
    out = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        q = deque([(y, x)])
        seen[y, x] = True
        pts = []
        while q:
            cy, cx = q.popleft()
            pts.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        out.append(np.array(pts))
    return out


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
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


def put_empty(out, x, y, col, budget):
    if budget[0] <= 0:
        return False
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        budget[0] -= 1
        return True
    return False


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


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
    """4x nearest. Positive degrees turn clockwise on screen. No |cos| squash."""
    if abs(angle_deg) % 360 < 0.01 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    up = np.repeat(np.repeat(layer, 4, axis=0), 4, axis=1)
    px, py = pivot[0] * 4 + 1.5, pivot[1] * 4 + 1.5
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = xx - px, yy - py
    src_x = np.rint(px + c * rx + s * ry).astype(np.int32)
    src_y = np.rint(py - s * rx + c * ry).astype(np.int32)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(up)
    out[valid] = up[src_y[valid], src_x[valid]]
    return out[1::4, 1::4].copy()


def despeckle(img):
    """Drop only pixels whose 4-neighbourhood is empty."""
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
    step = 400
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None].astype(np.int32) - pts[None, :, 0]) ** 2
        d += (tx[sl, None].astype(np.int32) - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def fill_pinholes(img, max_size=8):
    a = img[..., 3] >= 128
    empty = ~a
    h, w = a.shape
    seen = np.zeros_like(empty)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if empty[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if empty[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and empty[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    holes = empty & ~seen
    if not np.any(holes):
        return img
    out = img.copy()
    for pts in components(holes):
        if len(pts) > max_size:
            continue
        mask = np.zeros_like(holes)
        mask[pts[:, 0], pts[:, 1]] = True
        fill = nearest_color(out, mask, a)
        out[mask] = fill[mask]
    return out


def bridge_parts(img, max_gap=2):
    a = img[..., 3] >= 128
    parts = components(a)
    if len(parts) <= 1:
        return img
    parts.sort(key=len, reverse=True)
    main = np.zeros_like(a)
    main[parts[0][:, 0], parts[0][:, 1]] = True
    out = img.copy()
    changed = False
    for pts in parts[1:]:
        gap_lim = 3 if len(pts) > 80 else max_gap
        frag = np.zeros_like(a)
        frag[pts[:, 0], pts[:, 1]] = True
        edge = frag.copy()
        grown = None
        for _ in range(gap_lim):
            p = np.pad(edge, 1)
            dil = edge | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
            if np.any(dil & main):
                grown = dil & ~a
                break
            edge = dil & ~main
        else:
            continue
        if grown is None or not np.any(grown):
            main = main | frag
            continue
        fill = nearest_color(out, grown, out[..., 3] >= 128)
        out[grown] = fill[grown]
        changed = True
        main = main | grown | frag
        a = out[..., 3] >= 128
    return out if changed else img


def repair(img):
    img = fill_pinholes(img)
    img = bridge_parts(img)
    img = despeckle(img)
    return img


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def segment(img):
    """Eyestalk crests only. The shell mass below y42 stays on the body. No legs."""
    opaque = img[..., 3] >= 128
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    # Left stalk includes the near eye (white at x27). Right stalk includes
    # the far eye (white at x37). The small right horn is the second stalk tip.
    left = opaque & (yy <= 40) & (xx <= 33)
    right = opaque & (yy <= 38) & (xx >= 34) & (xx <= 44) & ~left
    far = opaque & (yy <= 34) & (xx >= 45) & ~left & ~right
    shell = opaque & (yy >= 42) & (xx >= 40)
    if np.any(shell & (left | right | far)):
        raise SystemExit("crest mask overlaps the shell")
    body = opaque & ~(left | right | far)
    return {"body": body, "left": left, "right": right, "far": far, "shell": shell & body}


def place(img, masks):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - (sh - 1)
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    crests = []
    for name in ("far", "right", "left"):
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = masks[name][y0:y1, x0:x1]
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        crests.append((pm, lyr))
    cx, cy = float(xs.mean()), float(ys.mean())
    pivot_spin = (dst_x + (cx - x0), dst_y + (cy - y0))
    low = int(ys.max())
    cxs = xs[ys == low]
    # Belly contact pixel nearest the body centroid.
    contact_x = int(cxs[np.argmin(np.abs(cxs.astype(np.float64) - cx))])
    pivot_hurt = (float(dst_x + (contact_x - x0)), float(dst_y + (low - y0)))
    counts = {k: int(v.sum()) for k, v in masks.items()}
    return canvas, crests, pivot_spin, pivot_hurt, counts


def dilate(mask, n):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def flare_crests(canvas, crests, shifts):
    if all(dx == 0 and dy == 0 for dx, dy in shifts):
        return canvas.copy()
    out = canvas.copy()
    union = np.zeros(out.shape[:2], bool)
    for mask, _layer in crests:
        union |= mask
        out[mask] = 0
    unmoved = out[..., 3] >= 128
    for (mask, layer), (dx, dy) in zip(crests, shifts):
        moved = shift(layer, dx, dy)
        paste(out, moved)
    moved_px = (out[..., 3] >= 128) & ~unmoved
    empty = out[..., 3] < 128
    gap = empty & dilate(unmoved, 2) & dilate(moved_px, 2) & dilate(union, 2)
    if np.any(gap):
        ys, xs = np.nonzero(gap)
        if len(ys) > 48:
            # Keep the joint pixels nearest the unmoved neck, not a smear.
            neck = dilate(unmoved, 1)
            keep = neck[ys, xs]
            if int(keep.sum()) >= 8:
                gap[:] = False
                gap[ys[keep], xs[keep]] = True
            else:
                gap[:] = False
                gap[ys[:48], xs[:48]] = True
        fill = nearest_color(out, gap, out[..., 3] >= 128)
        out[gap] = fill[gap]
    return out


def crawl_field(f, n, y_top, y_bot):
    """Horizontal peristalsis. Bottom three rows stay at dx 0. No vertical hop."""
    dx = np.zeros(CELL_H, np.int32)
    span = max(1, y_bot - y_top)
    for y in range(y_top, y_bot + 1):
        t = (y - y_top) / span
        plant = 1.0 if t < 0.72 else max(0.0, (1.0 - t) / 0.28)
        phase = 2 * math.pi * 1.15 * t
        dx[y] = int(round(wave(f, n, 3.0 * plant, phase=phase, cycles=2)))
    dx[y_bot - 2:y_bot + 1] = 0
    for y in range(y_bot - 1, y_top - 1, -1):
        if dx[y] > dx[y + 1] + 1:
            dx[y] = dx[y + 1] + 1
        elif dx[y] < dx[y + 1] - 1:
            dx[y] = dx[y + 1] - 1
    return dx


def apply_dx(img, dx):
    if not np.any(dx):
        return img.copy()
    out = np.zeros_like(img)
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    nx = xs + dx[ys]
    valid = (nx >= 0) & (nx < CELL_W)
    out[ys[valid], nx[valid]] = img[ys[valid], xs[valid]]
    return out


def crest_shifts(action, f):
    n = FRAMES[action]
    if action == "idle":
        rise = int(round(max(0.0, wave(f, n, 1.0, cycles=1))))
        spread = int(round(wave(f, n, 1.0, phase=0.4, cycles=1)))
        return [(-spread, -rise), (0, -rise), (spread, -rise)]
    if action == "attack":
        amount = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0][f]
        amount = min(amount, int(CFG["flare_max"]))
        return [(-amount, -amount), (0, -amount), (amount, -amount)]
    return [(0, 0), (0, 0), (0, 0)]


def face_front(canvas):
    a = canvas[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max())
    target = int(y0 + (y1 - y0) * 0.42)
    band = (ys >= target - 3) & (ys <= target + 3)
    return int(xs[band].min()), target


def roar_rings(img, origin, f):
    if not CFG["rings"] or f < 1 or f > 4:
        return img
    age = {1: 0, 2: 1, 3: 2, 4: 1}[f]
    strength = 2 if f in (2, 3) else 1
    ox, oy = origin
    budget = [22]
    cols = (YELLOW, WHITE, PINK)
    for i in range(strength):
        rad = 6 + age * 2 + i * 3
        col = cols[i % 3]
        for deg in (-36, -18, 0, 18, 36):
            ang = math.radians(deg)
            put_empty(img, ox - rad * math.cos(ang), oy + rad * math.sin(ang) * 0.7, col, budget)
            put_empty(img, ox - (rad - 1) * math.cos(ang), oy + (rad - 1) * math.sin(ang) * 0.7, col, budget)
    return img


def speed_lines(img):
    if not CFG["lines"]:
        return img
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return img
    x1 = int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    budget = [9]
    cols = (WHITE, GOLD, YELLOW)
    for k, t in enumerate((0.32, 0.5, 0.68)):
        y = int(y0 + (y1 - y0) * t)
        col = cols[k]
        for i in range(3):
            put_empty(img, x1 + 3 + i + k, y, col, budget)
    return img


def impact_burst(img, origin, f):
    if not CFG["burst"] or f not in (11, 12):
        return img
    ox, oy = origin
    budget = [16 if f == 11 else 10]
    cols = (WHITE, YELLOW, PINK)
    for k in range(6):
        ang = math.radians(140 + k * 16)
        rad = 8 if f == 11 else 5
        put_empty(img, ox - rad * math.cos(ang), oy + rad * math.sin(ang) * 0.55, cols[k % 3], budget)
        put_empty(img, ox - (rad - 2) * math.cos(ang), oy + (rad - 2) * math.sin(ang) * 0.55, cols[k % 3], budget)
    return img


def synth(canvas, crests, pivot_spin, pivot_hurt, y_top, y_bot, origin, action, f, palette):
    if f == 0 and action in ("idle", "walk", "attack", "hurt"):
        return canvas.copy()
    if action == "walk":
        dx = crawl_field(f, FRAMES["walk"], y_top, y_bot)
        img = apply_dx(canvas, dx)
        img = bridge_parts(img, max_gap=2)
        return snap(img, palette)
    if action == "hurt":
        n = FRAMES["hurt"]
        env = math.sin(math.pi * f / (n - 1))
        ang = 38.0 * env
        img = canvas if ang < 0.01 else repair(rotate_nearest(canvas, ang, pivot_hurt))
        return snap(img, palette)
    if action == "attack" and 5 <= f <= 10:
        # Eyestalk strike. The measured flare is 2px; more than that
        # pulls the stalks off the head. No whole-body spin.
        slap = (0.5, 0.8, 1.0, 1.0, 0.75, 0.4)[f - 5]
        amount = max(1, int(round(CFG["flare_max"] * slap)))
        shifts = [(-amount, -amount), (0, -amount), (amount, -amount)]
        img = flare_crests(canvas, crests, shifts)
        img = speed_lines(img)
        return snap(img, palette)
    shifts = crest_shifts(action, f)
    img = flare_crests(canvas, crests, shifts)
    if action == "attack":
        img = roar_rings(img, origin, f)
        img = impact_burst(img, origin, f)
    return snap(img, palette)


def facing_flip(img, direction):
    if direction in FACE_RIGHT:
        return img[:, ::-1].copy()
    return img


def params_hash():
    src = Path(__file__).read_bytes() + (SPECIES / "front.png").read_bytes()
    src += json.dumps(CFG, sort_keys=True).encode()
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
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 2, 4, 7, 8, 10, 12, 14),
        "attack": (0, 3, 4, 5, 7, 9, 11, 13),
        "hurt": (0, 2, 4, 6, 8, 10),
    }
    cols = max(len(v) for v in samples.values())
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            x = si * CELL_W
            y = ai * CELL_H
            m = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 6
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            # West row is the unflipped still, so the head stays on the left.
            tile = sheets[act][6 * CELL_H:7 * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            a = tile[..., 3] >= 128
            if not np.any(a):
                continue
            ys, xs = np.nonzero(a)
            y0, y1 = max(0, int(ys.min()) - pad), min(CELL_H, int(ys.max()) + 1 + pad)
            x0, x1 = max(0, int(xs.min()) - pad), min(CELL_W, int(xs.max()) + 1 + pad)
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
        if src.is_file() and "_work" not in src.parts:
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    return n


def south_attack_ratios(sheets):
    row = sheets["attack"][:CELL_H]
    counts = []
    for f in range(16):
        tile = row[:, f * CELL_W:(f + 1) * CELL_W]
        counts.append(int((tile[..., 3] >= 128).sum()))
    base = max(1, counts[0])
    return counts, [c / base for c in counts]


def largest_fraction(tile):
    a = tile[..., 3] >= 128
    parts = components(a)
    if not parts:
        return 0.0, 0
    parts.sort(key=len, reverse=True)
    return len(parts[0]) / max(1, int(a.sum())), int(a.sum())


def qa_report(master, sheets, palette, ratios):
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = [int(v) for v in np.unique(master[..., 3]) if int(v) not in (0, 255)]
    clip = {}
    spin = []
    for act in ACTIONS:
        sheet = sheets[act]
        for di, direction in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sheet[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{direction}:{f}")
                if act == "attack" and direction == "S" and 5 <= f <= 10:
                    frac, total = largest_fraction(tile)
                    spin.append({"f": f, "opaque": total, "main": round(frac, 3)})
    bad = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
    return {
        "version": VERSION,
        "off_palette_colors": len(off),
        "bad_alpha": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "attack_s_ratios": [round(r, 4) for r in ratios],
        "attack_s_min": round(min(ratios), 4),
        "attack_s_max": round(max(ratios), 4),
        "spin_body": spin,
        "_off": off,
        "_bad_ratio": bad,
        "_clip": clip,
    }


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


def publish_dir8(sprite_h, foot_y):
    entry = {
        "dex": 423,
        "form": "normal",
        "variant": "gastrodon",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0423_gastrodon/normal",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir",
        "nf": 16,
        "cw": CELL_W,
        "ch": CELL_H,
        "w": CELL_W,
        "h": int(sprite_h),
        "fy": int(foot_y),
        "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1,
        "src": "agent-sprite-forge",
        "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    DIR8_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with DIR8_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            text = DIR8.read_text()
            prefix = "window.ARENA_DIR8 = "
            if not text.startswith(prefix):
                raise SystemExit("dir8.js header changed; not publishing")
            keys = re.findall(r'"(\d{4}_[^"]+)":\{', text)
            if keys.count(KEY) != 1:
                raise SystemExit("0423_gastrodon key missing or duplicated")
            idx = keys.index(KEY)
            neighbor = keys[idx + 1] if idx + 1 < len(keys) else keys[idx - 1]
            other = keys[idx - 1] if idx > 0 else neighbor
            nb_s, nb_e = _entry_span(text, neighbor)
            ot_s, ot_e = _entry_span(text, other)
            neighbor_blob = text[nb_s:nb_e]
            other_blob = text[ot_s:ot_e]
            s, e = _entry_span(text, KEY)
            text = text[:s] + f'"{KEY}":' + blob + text[e:]
            if neighbor_blob not in text or other_blob not in text:
                raise SystemExit("neighbor dir8 key changed")
            body = text[len(prefix):].strip()
            if body.endswith(";"):
                body = body[:-1].strip()
            json.loads(body)
            if text.count(f'"v":"{VERSION}"') != 1:
                raise SystemExit("dir8 version stamp collided")
            DIR8.write_text(text)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return neighbor, other


def update_queue(status):
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            text = QUEUE.read_text()
            data = json.loads(text)
            before = {item.get("id"): (item.get("status"), item.get("claim")) for item in data["items"]}
            start = text.index('"id": "0423_gastrodon"')
            nxt = text.find('"id":', start + 10)
            end = len(text) if nxt < 0 else nxt
            block = text[start:end]
            if '"status":' not in block:
                raise SystemExit("queue status missing")
            block2 = re.sub(r'"status": "[^"]*"', f'"status": "{status}"', block, count=1)
            mid, n_mid = re.subn(r'\n\s*"claim": "[^"]*",', "", block2, count=1)
            if n_mid:
                block2 = mid
            else:
                block2 = re.sub(r',\s*\n\s*"claim": "[^"]*"', "", block2, count=1)
            text = text[:start] + block2 + text[end:]
            data = json.loads(text)
            ids = [item.get("id") for item in data["items"]]
            if ids.count(KEY) != 1:
                raise SystemExit("queue id collision")
            for item in data["items"]:
                if item.get("id") == KEY:
                    if item.get("status") != status or "claim" in item:
                        raise SystemExit("queue update did not stick")
                else:
                    if (item.get("status"), item.get("claim")) != before.get(item.get("id")):
                        raise SystemExit(f"queue neighbor changed: {item.get('id')}")
            QUEUE.write_text(text if text.endswith("\n") else text + "\n")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def render_frames(canvas, crests, pivot_spin, pivot_hurt, y_top, y_bot, origin, palette, frames_dir, state, state_p, force):
    total = len(DIRS) * len(ACTIONS)
    for act in ACTIONS:
        for d in DIRS:
            key = f"{act}/{d}"
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") and not force:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                img = facing_flip(
                    synth(canvas, crests, pivot_spin, pivot_hurt, y_top, y_bot, origin, act, f, palette),
                    d,
                )
                Image.fromarray(img, "RGBA").save(tmp)
                tmp.replace(fp)
            if key not in state["done"]:
                state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)


def invalidate_attack(state, frames_dir):
    state["done"] = [k for k in state["done"] if not k.startswith("attack/")]
    ad = frames_dir / "attack"
    if ad.exists():
        shutil.rmtree(ad)


def finish(sheets, master, palette, frames_dir, digest, counts, publish, skip_qa):
    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    foot_y = int(nys.max()) if len(nys) else 0
    counts_a, ratios = south_attack_ratios(sheets)
    print("ATTACK_S_COUNTS", counts_a, flush=True)
    print("ATTACK_S_RATIOS", [round(r, 4) for r in ratios], flush=True)
    report = qa_report(master, sheets, palette, ratios)
    report["sprite_h"] = sprite_h
    report["foot_y"] = foot_y
    report["hash"] = digest
    report["parts"] = counts
    idle_bot = []
    walk_bot = []
    for act, bucket in (("idle", idle_bot), ("walk", walk_bot)):
        row = sheets[act][6 * CELL_H:7 * CELL_H]
        for f in range(16):
            tile = row[:, f * CELL_W:(f + 1) * CELL_W]
            ys = np.nonzero(tile[..., 3] >= 128)[0]
            bucket.append(int(ys.max()) if len(ys) else -1)
    report["idle_contact_y"] = idle_bot
    report["walk_contact_y"] = walk_bot
    planted = len(set(idle_bot)) == 1 and len(set(walk_bot)) == 1
    report["belly_planted"] = planted
    if not skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        qa = OUT_DIR / "_work" / "qa"
        qa.mkdir(parents=True, exist_ok=True)
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                capture_output=True, text=True, timeout=90,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    passed = (
        not report["_bad_ratio"]
        and report["off_palette_colors"] == 0
        and not report["bad_alpha"]
        and not report["_clip"]
        and planted
        and all(s["main"] >= 0.90 for s in report["spin_body"])
    )
    report["passed"] = passed
    qa_dir = OUT_DIR / "_work" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    public = {k: v for k, v in report.items() if not k.startswith("_")}
    (qa_dir / "qa.json").write_text(json.dumps(public, ensure_ascii=False, indent=1))
    print(json.dumps(public, ensure_ascii=False)[:1200], flush=True)
    meta = {
        "generator": "forge_gastrodon_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 423,
        "form": "normal",
        "variant": "gastrodon",
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": foot_y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "face_right": sorted(FACE_RIGHT),
        "palette_colors": int(len(palette)),
        "off_palette_colors": report["off_palette_colors"],
        "attack_s_ratios": public["attack_s_ratios"],
        "source": "front.png 96x96",
        "gifs_untouched": True,
        "anatomy": "slug, shell attached, eyestalk crest, no legs, no mouth",
        "passed": passed,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not passed:
        if publish:
            update_queue("needs_fix")
            print("queue needs_fix", flush=True)
        print("FAIL", flush=True)
        return False
    if publish:
        print("synced files", sync_outputs(), flush=True)
        neighbor, other = publish_dir8(sprite_h, foot_y)
        print("dir8 neighbors held", neighbor, other, flush=True)
        update_queue("done")
        print("queue done", flush=True)
    else:
        print("READY no publish", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SPECIES / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    opaque = front[..., 3] >= 128
    covered = masks["body"] | masks["left"] | masks["right"] | masks["far"]
    if int((opaque & ~covered).sum()) or int((covered & ~opaque).sum()):
        raise SystemExit("segmentation does not cover the still")
    eyes = np.nonzero(opaque & np.all(front[..., :3] == np.array(WHITE, np.uint8), axis=-1))
    shell_x = float(np.nonzero(masks["shell"])[1].mean()) if np.any(masks["shell"]) else -1
    eye_x = float(eyes[1].mean()) if len(eyes[1]) else -1
    print("facing eye_x", round(eye_x, 1), "shell_x", round(shell_x, 1), "palette", len(palette), flush=True)
    if not (eye_x < shell_x):
        raise SystemExit("expected the eyestalks left of the shell")

    canvas, crests, pivot_spin, pivot_hurt, counts = place(front, masks)
    print("parts", counts, "pivot_spin", [round(v, 2) for v in pivot_spin], "pivot_hurt", [round(v, 2) for v in pivot_hurt], flush=True)
    ys = np.nonzero(canvas[..., 3] >= 128)[0]
    y_top, y_bot = int(ys.min()), int(ys.max())
    if y_bot != FOOT_Y:
        raise SystemExit(f"foot row {y_bot} != {FOOT_Y}")
    origin = face_front(canvas)
    print("face_front", origin, "opaque_h", int(ys.max() - ys.min() + 1), flush=True)

    rest = synth(canvas, crests, pivot_spin, pivot_hurt, y_top, y_bot, origin, "idle", 0, palette)
    differ = np.any(rest != canvas, axis=2) & ((rest[..., 3] >= 128) | (canvas[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "body": (80, 180, 120), "left": (220, 60, 60), "right": (40, 90, 220),
        "far": (240, 200, 40), "shell": (255, 255, 255),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", f) for f in (0, 4, 8, 12)]
        want += [("walk", f) for f in (0, 4, 8, 12)]
        want += [("attack", f) for f in (0, 2, 4, 5, 7, 9, 11, 13)]
        want += [("hurt", f) for f in (0, 3, 6, 9)]
        tiles = [synth(canvas, crests, pivot_spin, pivot_hurt, y_top, y_bot, origin, act, f, palette) for act, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        base = int((canvas[..., 3] >= 128).sum())
        print("sample attack ratios", flush=True)
        for f in range(16):
            img = synth(canvas, crests, pivot_spin, pivot_hurt, y_top, y_bot, origin, "attack", f, palette)
            n = int((img[..., 3] >= 128).sum())
            print(f"  {f:02d} {n / base:.4f} {n}", flush=True)
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

    render_frames(canvas, crests, pivot_spin, pivot_hurt, y_top, y_bot, origin, palette, frames_dir, state, state_p, False)
    sheets = assemble(frames_dir)
    _counts, ratios = south_attack_ratios(sheets)
    if any(r < 0.90 or r > 1.08 for r in ratios):
        print("ratios outside 0.90-1.08, regenerating attack with quieter effects", flush=True)
        CFG["flare_max"] = 1
        CFG["rings"] = False
        CFG["lines"] = False
        CFG["burst"] = False
        invalidate_attack(state, frames_dir)
        state["hash"] = params_hash()
        render_frames(canvas, crests, pivot_spin, pivot_hurt, y_top, y_bot, origin, palette, frames_dir, state, state_p, True)
        sheets = assemble(frames_dir)

    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions_dir = SPECIES / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    preview.save(actions_dir / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    west = sheets["idle"][6 * CELL_H:7 * CELL_H, :CELL_W]
    differ = np.any(west != canvas, axis=2) & ((west[..., 3] >= 128) | (canvas[..., 3] >= 128))
    print("W idle0 pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("west idle frame 0 does not match the still")

    finish(sheets, master, palette, frames_dir, digest, counts, args.publish and not args.no_sync, args.skip_qa)


if __name__ == "__main__":
    main()
