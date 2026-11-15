#!/usr/bin/env python3
"""Yanmega (0469_yanmega, メガヤンマ) normal 8-dir sheet from front.png.

Measured on this still only (96x96, opaque 1844, one component, 15 colours,
bbox x8-85 y19-78). The red compound eye sits at x=17.6; the body centroid
is near x=49, so the still faces left. W keeps the still. S SE E NE are the
horizontal mirror so E points screen-right.

Wings, not feet: an upper blade (y<=28, x<68,  the raised pair), a far red
tip (x>=68), and the outer near membrane. The left membrane also bridges the
eye to the thorax, so that bridge stays on the body. No mouth. The lowest
pixels are a hanging abdomen tip (y78, x49-51), not planted feet. Feet are 0.
No invented legs.

  idle   : 1px hover and a small wing beat. Frame 0 is the still.
  walk   : two wing beats. Head, thorax, and abdomen stay one piece.
  attack : 0-4 wing flare plus rings, 5-10 one in-plane 360 of the whole
           dragonfly including the wings (no |cos| squash), 11-12 impact,
           13-15 the still.
  hurt   : lean at most 38 degrees, then recover. 12 frames.

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.
"""

import argparse
import fcntl
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import Counter, deque
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SPECIES = ROOT / "monsterSprite/gen4/0469_yanmega/normal"
OUT_DIR = SPECIES / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0469_yanmega/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"
KEY = "0469_yanmega"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left. These four are mirrored so E points screen-right.
MIRROR = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 200, 192, 160
VERSION = "yanmega_v2"
KEEP = 6
FEET = 0

RED = (246, 90, 82)
DRED = (180, 49, 24)
PINK = (255, 148, 139)
BLACK = (16, 16, 16)
GRAY = (74, 74, 82)
LGRAY = (123, 123, 131)
WHITE = (255, 255, 255)
PALE = (164, 172, 197)
PALE2 = (197, 205, 230)
GREEN = (82, 131, 74)
DGREEN = (32, 57, 24)
MGREEN = (57, 90, 41)
LGREEN = (115, 156, 98)
BROWN = (115, 49, 0)
WING_COLORS = (WHITE, PALE, PALE2, LGRAY, GRAY)
EYE_COLORS = (RED, DRED, PINK)
WING_PARTS = ("wing_up", "wing_far", "wing_near")
PARTS = ("body",) + WING_PARTS


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


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
        out.append(pts)
    return out


def label_mask(mask):
    labels = np.zeros(mask.shape, np.int32)
    nlab = 0
    for pts in components(mask):
        nlab += 1
        for y, x in pts:
            labels[y, x] = nlab
    return labels, nlab


def keep_large(mask, min_size):
    kept = np.zeros_like(mask)
    for pts in components(mask):
        if len(pts) >= min_size:
            for y, x in pts:
                kept[y, x] = True
    return kept


def shortest_bridge(cand, head, torso):
    """Pixels of cand that tie head to torso. None if no path."""
    h, w = cand.shape
    adj_head = dilate(head, 1) & cand
    adj_torso = dilate(torso, 1) & cand
    if not np.any(adj_head) or not np.any(adj_torso):
        return None
    parent = {}
    seen = np.zeros_like(cand)
    q = deque()
    ys, xs = np.nonzero(adj_head)
    for y, x in zip(ys.tolist(), xs.tolist()):
        seen[y, x] = True
        parent[(y, x)] = None
        q.append((y, x))
    found = None
    while q:
        y, x = q.popleft()
        if adj_torso[y, x]:
            found = (y, x)
            break
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and cand[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                parent[(ny, nx)] = (y, x)
                q.append((ny, nx))
    if found is None:
        return None
    bridge = np.zeros_like(cand)
    cur = found
    while cur is not None:
        bridge[cur] = True
        cur = parent[cur]
    return bridge


def is_colors(img, opaque, cols):
    rgb = img[..., :3]
    mask = np.zeros(opaque.shape, bool)
    for col in cols:
        mask |= np.all(rgb == np.array(col, np.uint8), axis=2)
    return opaque & mask


def segment(img):
    """Upper blade, far red tip, outer near membrane. Eye bridge stays body."""
    h, w = img.shape[:2]
    opaque = img[..., 3] == 255
    yy, xx = np.mgrid[0:h, 0:w]
    up = opaque & (yy <= 28) & (xx < 68)
    far = opaque & (xx >= 68)
    wingish = is_colors(img, opaque, WING_COLORS)
    cand = wingish & (xx <= 42) & (yy >= 38) & (yy <= 58) & ~up & ~far
    rest = opaque & ~up & ~far & ~cand
    labels, nlab = label_mask(rest)
    eye = is_colors(img, opaque, EYE_COLORS) & (xx <= 32) & (yy >= 30) & (yy <= 48)
    if nlab <= 1:
        near = cand
    else:
        eye_ids = labels[eye]
        eye_ids = eye_ids[eye_ids > 0]
        if len(eye_ids) == 0:
            raise SystemExit("compound eye is not on the body side of the near wing")
        head_id = int(Counter(eye_ids.tolist()).most_common(1)[0][0])
        sizes = [(int((labels == i).sum()), i) for i in range(1, nlab + 1) if i != head_id]
        if not sizes:
            raise SystemExit("thorax missing after the wing cut")
        torso_id = max(sizes)[1]
        bridge = shortest_bridge(cand, labels == head_id, labels == torso_id)
        if bridge is None:
            near = np.zeros_like(opaque)
        else:
            near = keep_large(cand & ~bridge, 20)
    body = opaque & ~up & ~far & ~near
    movable = up | far | near
    # One-pixel membrane specks stay tied to the thorax by a short path.
    for _ in range(8):
        parts = components(body)
        if len(parts) <= 1:
            break
        parts.sort(key=len)
        island = np.zeros_like(body)
        for y, x in parts[0]:
            island[y, x] = True
        main = body & ~island
        path = shortest_bridge(movable, island, main)
        if path is None or int(path.sum()) > 40:
            break
        body |= path
        movable &= ~path
        up &= ~path
        far &= ~path
        near &= ~path
    masks = {
        "body": body,
        "wing_up": up,
        "wing_far": far,
        "wing_near": near,
    }
    own = np.zeros_like(opaque)
    overlap = 0
    for name in PARTS:
        overlap += int((own & masks[name]).sum())
        own |= masks[name]
    if int(own.sum()) != int(opaque.sum()) or overlap:
        raise SystemExit(f"masks do not partition the still overlap={overlap}")
    if int(masks["wing_up"].sum()) < 40 or int(masks["wing_far"].sum()) < 40:
        raise SystemExit("a yanmega wing mask is missing")
    body_n = len(components(masks["body"]))
    if body_n != 1:
        raise SystemExit(f"body split into {body_n} pieces")
    if not np.all(eye <= masks["body"]):
        raise SystemExit("compound eye left the body")
    return masks


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
    """4x nearest, then a 6-of-16 mode. Multiples of 360 stay exact.

    Positive degrees are the script's in-plane turn. No |cos| squash.
    """
    ang = angle_deg % 360.0
    if ang < 0.05 or ang > 359.95 or not np.any(layer[..., 3] >= 128):
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
    outu = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    outu[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    height, width = layer.shape[:2]
    blk = outu.reshape(height, 4, width, 4, 4)
    opa = blk[..., 3] >= 128
    cnt = opa.sum(axis=(1, 3))
    out = np.zeros_like(layer)
    ys, xs = np.nonzero(cnt >= KEEP)
    for y, x in zip(ys.tolist(), xs.tolist()):
        cols = blk[y, :, x, :, :3][opa[y, :, x, :]]
        if len(cols) == 0:
            continue
        uniq, counts = np.unique(cols, axis=0, return_counts=True)
        out[y, x, :3] = uniq[int(counts.argmax())]
        out[y, x, 3] = 255
    return out


def forward_stamp(img, layer, angle_deg, pivot):
    """Put every source pixel back on. Collisions keep the first colour."""
    ang = math.radians(angle_deg)
    c, s = math.cos(ang), math.sin(ang)
    scale = 4
    px, py = pivot[0] * scale + 1.5, pivot[1] * scale + 1.5
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    if len(ys) == 0:
        return {}
    sx = xs.astype(np.float64) * scale + 1.5
    sy = ys.astype(np.float64) * scale + 1.5
    dx, dy = sx - px, sy - py
    ox = px + c * dx - s * dy
    oy = py + s * dx + c * dy
    xx = np.rint((ox - 1.5) / scale).astype(np.int32)
    yy = np.rint((oy - 1.5) / scale).astype(np.int32)
    h, w = img.shape[:2]
    dest = {}
    for y, x, dyi, dxi in zip(ys.tolist(), xs.tolist(), yy.tolist(), xx.tolist()):
        dest[(y, x)] = (dyi, dxi)
        if 0 <= dyi < h and 0 <= dxi < w and img[dyi, dxi, 3] < 128:
            img[dyi, dxi] = layer[y, x]
    return dest


def bridge_gaps(img, layer, dest, max_gap=3):
    """Fill holes where neighbours landed two or three cells apart."""
    if not dest:
        return
    h, w = img.shape[:2]
    opaque = layer[..., 3] >= 128
    ys, xs = np.nonzero(opaque)
    pix = {(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())}
    for y, x in pix:
        for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
            q = (y + dy, x + dx)
            if q not in pix:
                continue
            ya, xa = dest[(y, x)]
            yb, xb = dest[q]
            gap = max(abs(ya - yb), abs(xa - xb))
            if gap <= 1 or gap > max_gap:
                continue
            for step in range(1, gap):
                my = int(round(ya + (yb - ya) * step / gap))
                mx = int(round(xa + (xb - xa) * step / gap))
                if 0 <= my < h and 0 <= mx < w and img[my, mx, 3] < 128 and img[ya, xa, 3] >= 128:
                    img[my, mx] = img[ya, xa]


def rotate_piece(layer, angle_deg, pivot):
    """Forward in-plane rotate. One stamp per source pixel, no |cos| squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    img = np.zeros_like(layer)
    dest = forward_stamp(img, layer, angle_deg, pivot)
    bridge_gaps(img, layer, dest, max_gap=3)
    return img


def despeckle(img):
    """Drop only 8-isolated specks. A 4-neighbour test erases diagonal wings."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n8 = (
        p[:-2, :-2].astype(np.int16) + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    )
    out[a & (n8 == 0)] = 0
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
        dist = (ty[sl, None] - pts[None, :, 0]) ** 2 + (tx[sl, None] - pts[None, :, 1]) ** 2
        pick[sl] = dist.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def extract_palette(img):
    a = img[..., 3] == 255
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


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def hinge_pivot(part, body):
    """Rotate around the pixels that already touch the thorax, not a free tip."""
    touch = part & dilate(body, 2)
    if not np.any(touch):
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    return float(xs.mean()), float(ys.mean())


def build_rig(img, masks):
    opaque = img[..., 3] == 255
    ys, xs = np.nonzero(opaque)
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

    for part_name in WING_PARTS:
        if not np.any(placed[part_name]):
            continue
        zone = placed[part_name] & dilate(placed["body"], 1)
        if not np.any(zone):
            continue
        fill = nearest_color(canvas, zone, placed["body"])
        write = zone & (layers["body"][..., 3] < 128)
        layers["body"][write] = fill[write]

    oys, oxs = np.nonzero(canvas[..., 3] == 255)
    center = (float(oxs.mean()), float(oys.mean()))
    eye = placed["body"] & np.all(canvas[..., :3] == np.array(RED, np.uint8), axis=-1)
    ey, ex = np.nonzero(eye)
    face = (float(ex.mean()), float(ey.mean())) if len(ex) else center
    pivots = {}
    for name in WING_PARTS:
        if np.any(placed[name]):
            pivots[name] = hinge_pivot(placed[name], placed["body"])
    return layers, pivots, center, face, canvas


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def iwave(f, n, amp, phase=0.0, cycles=1.0):
    return int(round(wave(f, n, amp, phase, cycles)))


def pose(action, f):
    n = FRAMES[action]
    p = dict(dx=0, dy=0, wing=0.0, spin=0.0, hurt=0.0, rings=0, ring=0, burst=0, sparks=0)
    if action == "idle":
        p["dy"] = iwave(f, n, 1.0)
        p["wing"] = wave(f, n, 8.0)
    elif action == "walk":
        p["dy"] = iwave(f, n, 1.0, cycles=2.0)
        p["dx"] = iwave(f, n, 1.0, cycles=2.0)
        p["wing"] = wave(f, n, 14.0, cycles=2.0)
    elif action == "attack":
        wing = [6, 10, 14, 12, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        p["wing"] = float(wing[f])
        if 0 <= f <= 4:
            p["rings"] = 2 if f in (2, 3) else 1
            p["ring"] = f
        if 5 <= f <= 10:
            ang = 60.0 * (f - 4)
            p["spin"] = 0.0 if ang % 360 == 0 else ang
        if f in (11, 12):
            p["burst"] = 1
    else:
        env = [0.0, 0.35, 0.65, 0.88, 1.0, 0.92, 0.72, 0.48, 0.28, 0.12, 0.04, 0.0][f]
        p["hurt"] = -38.0 * env
        p["dx"] = int(round(3 * env))
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    if abs(p["hurt"]) > 38.01:
        raise SystemExit("hurt angle exceeds 38")
    return p


def turn(layer, angle, pivot):
    if abs(angle) < 0.05 or abs(angle) % 360 < 0.05:
        return layer
    return despeckle(rotate_piece(layer, angle, pivot))


def leading_edge(img, face):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    return float(xs.min()), float(face[1])


def roar_rings(out, face, age, strength):
    """No mouth on this still. Rings sit in front of the eye."""
    ox, oy = leading_edge(out, face)
    cols = (WHITE, PALE, RED)
    for r_i in range(strength):
        rad = 5 + age * 2 + r_i * 3
        col = cols[r_i % 3]
        for deg in range(-40, 41, 20):
            ang = math.radians(deg)
            put(out, ox - 2 - rad * math.cos(ang), oy + rad * math.sin(ang) * 0.65, col)
    return out


def impact_burst(out, face):
    ox, oy = leading_edge(out, face)
    cols = (WHITE, RED, LGREEN)
    for k in range(6):
        ang = math.radians(-50 + k * 20)
        put(out, ox - 3 - 6 * math.cos(ang), oy + 4 * math.sin(ang), cols[k % 3])
    return out


def sparks(out, frame_idx):
    a = out[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return out
    x0, y0 = int(xs.min()), int(ys.min())
    d = frame_idx
    for sx, sy, col in (
        (x0 - 2 - d, y0 + 6, WHITE),
        (x0 - 1, y0 - 2 - d, PALE),
        (x0 + 8, y0 - 3 - d, RED),
    ):
        put(out, sx, sy, col)
    return out


def link_pair(img, a_mask, b_mask, max_dist=4):
    if not np.any(a_mask) or not np.any(b_mask):
        return
    ay, ax = np.nonzero(a_mask)
    by, bx = np.nonzero(b_mask)
    if len(ay) > 2500:
        ay, ax = ay[::2], ax[::2]
    if len(by) > 2500:
        by, bx = by[::2], bx[::2]
    cheb = np.maximum(np.abs(ay[:, None] - by[None, :]), np.abs(ax[:, None] - bx[None, :]))
    flat = int(cheb.argmin())
    i, j = divmod(flat, cheb.shape[1])
    dist = int(cheb[i, j])
    if dist == 0 or dist > max_dist:
        return
    y0, x0 = int(ay[i]), int(ax[i])
    y1, x1 = int(by[j]), int(bx[j])
    if abs(y1 - y0) + abs(x1 - x0) <= 1:
        return
    col = img[y0, x0].copy()
    y, x = y0, x0
    while (y, x) != (y1, x1):
        if abs(y1 - y) >= abs(x1 - x) and y != y1:
            y += 1 if y1 > y else -1
        elif x != x1:
            x += 1 if x1 > x else -1
        else:
            break
        if (y, x) != (y1, x1) and img[y, x, 3] < 128:
            img[y, x] = col


def tie_wings(img):
    """Keep a lifted wing on the thorax. Rings are drawn later and stay separate."""
    opaque = img[..., 3] >= 128
    parts = components(opaque)
    if len(parts) <= 1:
        return
    parts.sort(key=len, reverse=True)
    main = np.zeros_like(opaque)
    for y, x in parts[0]:
        main[y, x] = True
    for pts in parts[1:]:
        island = np.zeros_like(opaque)
        for y, x in pts:
            island[y, x] = True
        link_pair(img, main, island, max_dist=8)
        newly = (img[..., 3] >= 128) & ~opaque & ~island
        main |= newly | island
        opaque = img[..., 3] >= 128


def synth(rig, action, f, palette):
    layers, pivots, center, face = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = abs(p["spin"]) > 0.05
    recoiling = abs(p["hurt"]) > 0.05
    wing = 0.0 if spinning or recoiling else p["wing"]
    img = np.zeros_like(layers["body"])
    paste(img, layers["body"])
    moved = []
    # Up blade and far tip beat; the near membrane beats opposite, like a dragonfly pair.
    signs = {"wing_up": -1.0, "wing_far": 1.0, "wing_near": -1.0}
    for name in WING_PARTS:
        if name not in pivots:
            continue
        part = turn(layers[name], signs[name] * wing, pivots[name])
        paste(img, part)
        moved.append(part)
    if not spinning and not recoiling and abs(wing) >= 0.45:
        tie_wings(img)
    if spinning:
        img = turn(img, p["spin"], center)
    elif recoiling:
        img = turn(img, p["hurt"], center)
    img = shift(img, p["dx"], p["dy"])
    img = snap(img, palette)
    if p["rings"] and not spinning:
        img = roar_rings(img, face, p["ring"], p["rings"])
    if p["burst"]:
        img = impact_burst(img, face)
    if p["sparks"]:
        img = sparks(img, f)
    return snap(img, palette)


def facing_flip(img, direction):
    if direction in MIRROR:
        return img[:, ::-1].copy()
    return img


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (SPECIES / "front.png").read_bytes()
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
        "walk": (0, 4, 8, 12),
        "attack": (0, 2, 4, 6, 8, 10, 11, 15),
        "hurt": (0, 2, 4, 6, 8, 11),
    }
    cols = max(len(v) for v in samples.values())
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            y = ai * CELL_H
            x = si * CELL_W
            m = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
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
            y0, y1 = max(0, int(ys.min()) - pad), min(CELL_H, int(ys.max()) + 1 + pad)
            x0, x1 = max(0, int(xs.min()) - pad), min(CELL_W, int(xs.max()) + 1 + pad)
            row.append(tile[y0:y1, x0:x1])
        crops.append(row)
    cell_h = max(c.shape[0] for row in crops for c in row)
    cell_w = max(c.shape[1] for row in crops for c in row)
    cols = max(len(row) for row in crops)
    bg = np.full((len(crops) * cell_h, cols * cell_w, 4), (15, 20, 31, 255), np.uint8)
    for ri, row in enumerate(crops):
        for ci, crop in enumerate(row):
            y = ri * cell_h + (cell_h - crop.shape[0]) // 2
            x = ci * cell_w + (cell_w - crop.shape[1]) // 2
            m = crop[..., 3] >= 128
            bg[y:y + crop.shape[0], x:x + crop.shape[1]][m] = crop[m]
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


def opaque_count(tile):
    return int((tile[..., 3] == 255).sum())


def component_count(tile):
    return len(components(tile[..., 3] >= 128))


def largest_fraction(tile):
    a = tile[..., 3] >= 128
    parts = components(a)
    if not parts:
        return 0.0
    parts.sort(key=len, reverse=True)
    return len(parts[0]) / max(1, int(a.sum()))


def south_ratios(sheets, still_n):
    out = {}
    for act in ACTIONS:
        row = sheets[act][:CELL_H]
        ratios = []
        for f in range(FRAMES[act]):
            n = opaque_count(row[:, f * CELL_W:(f + 1) * CELL_W])
            ratios.append(n / still_n if still_n else 0.0)
        out[act] = ratios
    return out


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
        "dex": 469,
        "form": "normal",
        "variant": "yanmega",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0469_yanmega/normal",
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
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    DIR8_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with DIR8_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        if text.count(f'"{KEY}":') != 1:
            raise SystemExit("0469_yanmega key is not unique")
        keys = re.findall(r'"(\d{4}_[^"]+)":\{', text)
        if KEY not in keys:
            raise SystemExit("0469_yanmega missing from dir8.js")
        if any(k.endswith("-mega") and k == KEY for k in keys):
            raise SystemExit("refusing a mega key")
        idx = keys.index(KEY)
        neighbor = keys[idx + 1] if idx + 1 < len(keys) else keys[idx - 1]
        nb_s, nb_e = _entry_span(text, neighbor)
        neighbor_blob = text[nb_s:nb_e]
        s, e = _entry_span(text, KEY)
        text = text[:s] + f'"{KEY}":' + blob + text[e:]
        if neighbor_blob not in text:
            raise SystemExit("neighbor dir8 key changed")
        if text.count(f'"{KEY}":') != 1:
            raise SystemExit("0469_yanmega key duplicated")
        body = text[len(prefix):].strip()
        if body.endswith(";"):
            body = body[:-1].strip()
        else:
            raise SystemExit("dir8.js missing trailing semicolon")
        json.loads(body)
        if text.count(f'"v":"{VERSION}"') != 1:
            raise SystemExit("dir8 version stamp collided")
        if not text.endswith(";\n"):
            if text.endswith(";"):
                text += "\n"
            else:
                raise SystemExit("dir8.js ending is not semicolon newline")
        DIR8.write_text(text)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return neighbor


def update_queue(status):
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = QUEUE.read_text()
        start = text.index('"id": "0469_yanmega"')
        nxt = text.find('"id":', start + 10)
        end = len(text) if nxt < 0 else nxt
        block = text[start:end]
        if '"status":' not in block or '"id": "0469_yanmega"' not in block:
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
        if not text.endswith("\n"):
            text += "\n"
        QUEUE.write_text(text)
        fcntl.flock(lock, fcntl.LOCK_UN)


def measure_face(front):
    opaque = front[..., 3] == 255
    yy, xx = np.mgrid[0:96, 0:96]
    eye = is_colors(front, opaque, EYE_COLORS) & (xx <= 32) & (yy >= 30) & (yy <= 48)
    ey, ex = np.nonzero(eye)
    by, bx = np.nonzero(opaque)
    return float(ex.mean()), float(bx.mean()), int(eye.sum()), int(by.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front_path = SPECIES / "front.png"
    front_hash = hashlib.sha1(front_path.read_bytes()).hexdigest()
    front = np.array(Image.open(front_path).convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    if np.any((front[..., 3] != 0) & (front[..., 3] != 255)):
        raise SystemExit("front.png has partial alpha")
    palette = extract_palette(front)
    face_x, body_x, eye_n, bottom = measure_face(front)
    print(
        f"facing eye_x {face_x:.1f} body_x {body_x:.1f} eye_px {eye_n} "
        f"bottom_y {bottom} feet {FEET} mouth 0",
        flush=True,
    )
    if not (face_x < body_x - 8):
        raise SystemExit("facing check failed; the eye is not left of the body")
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), "wings", 3, "feet", FEET, flush=True)

    layers, pivots, center, face, canvas = build_rig(front, masks)
    rig = (layers, pivots, center, face)
    print(
        "pivot", [round(v, 2) for v in center],
        "face", [round(v, 2) for v in face],
        "wing_pivots", {k: [round(v, 1) for v in piv] for k, piv in pivots.items()},
        flush=True,
    )
    rest = synth(rig, "idle", 0, palette)
    differ = np.any(rest != canvas, axis=2) & ((rest[..., 3] >= 128) | (canvas[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "body": (40, 70, 120),
        "wing_up": (230, 200, 60),
        "wing_far": (210, 60, 60),
        "wing_near": (80, 200, 200),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 3, 0), 3, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", f) for f in range(16)]
        want += [("walk", f) for f in range(16)]
        want += [("attack", f) for f in range(16)]
        want += [("hurt", f) for f in range(12)]
        tiles = [facing_flip(synth(rig, act, f, palette), "S") for act, f in want]
        base = opaque_count(canvas)
        print("south opaque ratios vs still", base, flush=True)
        cursor = 0
        for act in ACTIONS:
            print(act, flush=True)
            for f in range(FRAMES[act]):
                tile = tiles[cursor]
                cursor += 1
                n = opaque_count(tile)
                comps = component_count(tile)
                frac = largest_fraction(tile)
                print(f"  {f:02d} {n} {n / base:.3f} comp {comps} main {frac:.3f}", flush=True)
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
    cache = {}
    for act in ACTIONS:
        for d in DIRS:
            key = f"{act}/{d}"
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if key in state.get("done", []) and fp.exists() and fp.stat().st_size > 0 and not args.force:
                    continue
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest and not args.force:
                    continue
                ck = (act, f)
                if ck not in cache:
                    cache[ck] = synth(rig, act, f, palette)
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(facing_flip(cache[ck], d), "RGBA").save(tmp)
                tmp.replace(fp)
            if key not in state["done"]:
                state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    still_n = opaque_count(canvas)
    ratios = south_ratios(sheets, still_n)
    print("south opaque ratios vs still", still_n, flush=True)
    bad_ratio = []
    for act, seq in ratios.items():
        print(act, flush=True)
        for f, r in enumerate(seq):
            n = int(round(r * still_n))
            print(f"  {f:02d} {n} {r:.3f}", flush=True)
            if r < 0.90 or r > 1.08:
                bad_ratio.append(f"{act}:{f}:{r:.3f}")

    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = [int(v) for v in np.unique(master[..., 3])]
    bad_alpha = [v for v in alphas if v not in (0, 255)]
    clip = []
    spin_frac = []
    spin_comp = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                cell = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = cell[..., 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
                if act == "attack" and d == "S" and 5 <= f <= 10:
                    spin_frac.append(round(largest_fraction(cell), 3))
                    spin_comp.append(component_count(cell))
    west = sheets["idle"][6 * CELL_H:7 * CELL_H, :CELL_W]
    differ = np.any(west != canvas, axis=2) & ((west[..., 3] >= 128) | (canvas[..., 3] >= 128))
    rest_diff = int(differ.sum())
    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] == 255)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    print(
        f"min {min(min(v) for v in ratios.values()):.3f} "
        f"max {max(max(v) for v in ratios.values()):.3f} "
        f"h {sprite_h} feet {FEET} spin_comp {spin_comp} spin_frac {spin_frac}",
        flush=True,
    )

    qa = work / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    gait = {}
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                gait[act] = json.loads(rep.read_text()).get("recommendation")
    summary = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_h,
        "feet": FEET,
        "mouth": 0,
        "off_palette": len(off),
        "bad_alpha": bad_alpha,
        "clip": clip,
        "rest_diff": rest_diff,
        "south_ratios": {k: [round(r, 3) for r in v] for k, v in ratios.items()},
        "spin_main_frac": spin_frac,
        "spin_comp": spin_comp,
        "gait": gait,
    }
    (qa / "qa.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    meta = {
        "generator": "forge_yanmega_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 469,
        "form": "normal",
        "variant": "yanmega",
        "name_ja": "メガヤンマ",
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "feet": FEET,
        "mouth": 0,
        "wings": 3,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "mirror": sorted(MIRROR),
        "still_on": "W",
        "palette_colors": int(len(palette)),
        "off_palette_colors": len(off),
        "source": "front.png 96x96",
        "gifs_untouched": True,
        "face_x": round(face_x, 2),
        "body_x": round(body_x, 2),
        "archetype": "dragonfly_hover",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    failed = (
        off or bad_alpha or clip or rest_diff or bad_ratio
        or any(v < 0.99 for v in spin_frac)
        or any(v != 1 for v in spin_comp)
    )
    if failed:
        if args.publish:
            update_queue("needs_fix")
        raise SystemExit(
            f"qa failed off={len(off)} alpha={bad_alpha} clip={clip[:8]} "
            f"rest={rest_diff} ratio={bad_ratio[:8]} spin={spin_frac} comp={spin_comp}"
        )
    if hashlib.sha1(front_path.read_bytes()).hexdigest() != front_hash:
        raise SystemExit("front.png changed")
    if args.publish and not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
        neighbor = publish_dir8(sprite_h)
        update_queue("done")
        arena_sheet = SYNC_OUT / "sheet.png"
        print(
            f"published {neighbor} h {sprite_h} feet {FEET} "
            f"arena_sheet_bytes {arena_sheet.stat().st_size} v {VERSION}",
            flush=True,
        )


if __name__ == "__main__":
    main()
