#!/usr/bin/env python3
"""Froslass (0478) normal 8-direction action sheet from this form's front.png / back.png.

Measured on normal/front.png: 96x96, 1155 opaque, bbox x33-64 y20-74,
16 colours, alpha 0/255, one 4-connected piece.

Body plan (from the pixels): legless floating kimono ghost. No legs, no wings,
no mouth pixels (the face is two purple eye sockets; only the right one has a
yellow iris, x47-49 y36-39).
  horn_l   blue ice horn, top-left of the head, x35-40 y23-28
  crest    blue ice crest on the crown, x45-53 y20-28
  head     white dome x35-62 y20-42 (stays rigid; integer moves only)
  sleeve   big near sleeve, screen-left, x33-45 y45-64, blue pattern
  obi      red sash tail behind the body, screen-right, x55-64 y48-58
  hem      skirt y60-74 (sways as integer row shear, never torn)
The eye pair sits left of the head centre and the obi trails right, so the
still faces screen-left.

back.png (1118 opaque, subset of the front palette) is the back view facing
up-right: blue hair falls left, ice horn x55-60 y23-30, red obi bow on the
back, far sleeve lower-left x33-41 y57-68, near sleeve right x53-62 y45-63.

Directions: front still W SW unflipped, S SE E mirrored (faces right).
back still N NE unflipped, NW mirrored. E faces screen-right, W screen-left.

  idle   : hover bob (integer), sleeve and obi flutter, hem ripple, horns
           tremble. Frame 0 and the loop end are the still.
  walk   : glide (no legs): hover bob x2, sleeve rows, obi and hem trail,
           frost speed lines.
  attack : 0-4 roar: no mouth, so the ice horns and crest flare, the sleeve
           and obi rise, frost rings from the face (back rows: from the head).
           5-7 pirouette: the body turns around (front -> side -> back ->
           side) with a snow swirl, using the real back view, never an
           in-plane rotation. 8-10 ice sleeve sweep (<= 34 deg plus integer
           thrust) with an icicle trail. 11-12 ice burst. 13-15 recover.
  hurt   : 12 frames, white flash, backward lean <= 24 deg as an integer row
           shear (head rigid, face never resamples), knock back, recover.

Palette-locked to this still. front/back png and gif are not written.
Resumable via agent-sprite-forge/_work. --force clears only this form's frames.
"""

import argparse
import fcntl
import hashlib
import json
import math
import shutil
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0478_froslass/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0478_froslass/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
BOUNDS = ROOT / "data/sprite_bounds.js"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
VIEW = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False), "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "froslass_v2"
KEY = "0478_froslass"

C_K = (16, 16, 16)
C_D1 = (65, 65, 106)
C_D2 = (65, 82, 139)
C_P1 = (74, 49, 115)
C_B = (82, 123, 180)
C_R5 = (106, 49, 49)
C_PU = (115, 49, 164)
C_I = (115, 180, 213)
C_G = (139, 139, 172)
C_RED = (197, 41, 32)
C_A = (197, 205, 222)
C_O1 = (222, 148, 57)
C_O2 = (230, 123, 74)
C_Y = (255, 213, 90)
C_W = (255, 255, 255)
BLUES = (C_K, C_D1, C_D2, C_B, C_I)
REDS = (C_R5, C_RED, C_O1, C_O2)

FRONT_N = 1155
FRONT_BBOX = (33, 64, 20, 74)
BACK_N = 1118
BACK_BBOX = (33, 62, 22, 73)

# right edge (inclusive) of the front sleeve per row, read off the outline
SLEEVE_LIM = {45: 45, 46: 45, 47: 44, 48: 45, 49: 44, 50: 44, 51: 43, 52: 44, 53: 43,
              54: 43, 55: 43, 56: 43, 57: 42, 58: 42, 59: 42, 60: 42, 61: 42, 62: 41,
              63: 41, 64: 40}
# left edge (inclusive) of the back near sleeve per row
BSLEEVE_LIM = {45: 56, 46: 56, 47: 56, 48: 56, 49: 53, 50: 53, 51: 53, 52: 53, 53: 53,
               54: 53, 55: 53, 56: 53, 57: 53, 58: 53, 59: 53, 60: 54, 61: 55, 62: 55,
               63: 56}


# ---------------------------------------------------------------- helpers
def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def flood(seeds, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow)
    q = deque()
    ys, xs = np.nonzero(seeds & allow)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if not seen[y, x]:
            seen[y, x] = True
            q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and allow[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


def component_sizes(mask):
    seen = np.zeros_like(mask)
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


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
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
    flat[sel] = mapped[inv.reshape(-1)]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    out[~a] = 0
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
    """4x nearest rotation, positive = clockwise on screen. No squash."""
    if not np.any(layer[..., 3] >= 128):
        return layer.copy()
    norm = angle_deg % 360.0
    if min(norm, 360.0 - norm) < 0.45:
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
    spun = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    spun[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    inv = spun[2::4, 2::4].copy()
    for _ in range(2):
        al = inv[..., 3] >= 128
        pp = np.pad(al, 1)
        n = pp[:-2, 1:-1].astype(np.int16) + pp[2:, 1:-1] + pp[1:-1, :-2] + pp[1:-1, 2:]
        hole = (~al) & (n >= 3)
        if not np.any(hole):
            break
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            src = np.roll(np.roll(inv, -dy, 0), -dx, 1)
            take = hole & (src[..., 3] >= 128) & (inv[..., 3] < 128)
            inv[take] = src[take]
    return inv


def content_point(x, y, angle_deg, pivot):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    vx, vy = x - pivot[0], y - pivot[1]
    return pivot[0] + c * vx - s * vy, pivot[1] + s * vx + c * vy


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int64)
    for i in range(0, len(ty), 400):
        sl = slice(i, i + 400)
        d = (ty[sl, None].astype(np.int32) - pts[None, :, 0]) ** 2
        d += (tx[sl, None].astype(np.int32) - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def seal_cracks(img):
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        p = np.pad(a, 1)
        n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        hole = (~a) & (n >= 3)
        if not np.any(hole):
            break
        fill = nearest_color(out, hole, a)
        out[hole] = fill[hole]
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def touches_edge(tile):
    c = tile[..., 3] >= 128
    if not np.any(c):
        return False
    return bool(c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any())


def narrow_x(img, scale, cx):
    """Narrow horizontally around cx for a turn frame. Union sampling keeps it one piece."""
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    for x in range(w):
        s0 = cx + (x - 0.5 - cx) / scale
        s1 = cx + (x + 0.5 - cx) / scale
        a0, a1 = int(math.floor(min(s0, s1) + 0.5)), int(math.floor(max(s0, s1) - 0.5 + 1e-9))
        mid = int(round(cx + (x - cx) / scale))
        order = [mid] + [c for c in range(a0, a1 + 1) if c != mid]
        for sx in order:
            if not (0 <= sx < w):
                continue
            col = img[:, sx]
            use = (out[:, x, 3] < 128) & (col[:, 3] >= 128)
            out[use, x] = col[use]
    return out


def row_shear(layer, top_y, bot_y, amount):
    """Integer row shift growing from 0 at top_y to `amount` at bot_y (cloth ripple)."""
    if abs(amount) < 0.5:
        return layer
    out = layer.copy()
    span = max(1, bot_y - top_y)
    for y in range(top_y, layer.shape[0]):
        k = min(1.0, (y - top_y) / span)
        off = int(round(amount * k))
        if off:
            out[y] = np.roll(layer[y], off, axis=0)
    return out


# ---------------------------------------------------------------- measure
def check_still(img, n_exp, bbox_exp, name):
    a = img[..., 3] >= 128
    if img.shape[:2] != (96, 96):
        raise SystemExit(f"{name} is not 96x96")
    if int(a.sum()) != n_exp:
        raise SystemExit(f"{name} opaque changed: {int(a.sum())}")
    ys, xs = np.nonzero(a)
    if (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())) != bbox_exp:
        raise SystemExit(f"{name} bbox changed; remeasure before forging")
    if component_sizes(a) != [n_exp]:
        raise SystemExit(f"{name} is not one 4-connected piece")
    return a


def is_col(img, a, cols):
    rgb = img[..., :3]
    hit = np.zeros(a.shape, bool)
    for c in cols:
        hit |= np.all(rgb == np.array(c, np.uint8), axis=-1)
    return a & hit


def seed_at(shape, x, y):
    s = np.zeros(shape, bool)
    s[y, x] = True
    return s


def finish_parts(parts, n_exp, name):
    total = sum(int(m.sum()) for m in parts.values())
    if total != n_exp:
        raise SystemExit(f"{name} parts overlap or miss pixels: {total}")
    for pn, m in parts.items():
        sz = component_sizes(m)
        if pn != "torso" and len(sz) != 1:
            raise SystemExit(f"{name} {pn} is not one piece: {sz[:5]}")
    return {k: int(v.sum()) for k, v in parts.items()}


def segment_front(img):
    a = check_still(img, FRONT_N, FRONT_BBOX, "front")
    yy, xx = np.mgrid[0:96, 0:96]
    blue = is_col(img, a, BLUES)
    horn_l = flood(seed_at(a.shape, 37, 25), blue & (yy <= 28) & (xx <= 40))
    crest = flood(seed_at(a.shape, 48, 22), blue & (yy <= 28) & (xx >= 45) & (xx <= 53))
    allow = np.zeros_like(a)
    for y, lim in SLEEVE_LIM.items():
        allow[y, :lim + 1] = True
    sleeve = flood(seed_at(a.shape, 36, 55), a & allow)
    red = is_col(img, a, REDS + (C_K,))
    obi = flood(seed_at(a.shape, 60, 52),
                red & (xx >= 55) & (yy >= 48) & (yy <= 58) & ~((xx <= 56) & (yy >= 56)))
    torso = a & ~horn_l & ~crest & ~sleeve & ~obi
    parts = {"horn_l": horn_l, "crest": crest, "sleeve": sleeve, "obi": obi, "torso": torso}
    info = finish_parts(parts, FRONT_N, "front")
    eye = is_col(img, a, (C_Y,))
    sockets = is_col(img, a, (C_PU,)) & (yy <= 41)
    if int(eye.sum()) != 6:
        raise SystemExit("eye iris changed")
    ex = float(np.nonzero(sockets | eye)[1].mean())
    hx = np.nonzero(a & (yy <= 41))[1]
    head_cx = (float(hx.min()) + float(hx.max())) / 2
    if not ex < head_cx - 1:
        raise SystemExit(f"front no longer faces left: eyes {ex:.1f} head {head_cx:.1f}")
    info.update(eye_x=round(ex, 1), head_cx=round(head_cx, 1))
    pivots = {"horn_l": (39, 27), "crest": (48, 28), "sleeve": (44, 47), "obi": (55, 53),
              "hem_y": 60, "bot_y": 74, "hip_y": 62, "neck_y": 43}
    return {"parts": parts, "pivots": pivots, "eye": eye,
            "near": "sleeve", "far": None, "obi": "obi",
            "horns": ("horn_l", "crest"),
            # positive angle raises the near sleeve outward (forward/up)
            "near_sign": 1.0, "far_sign": 0.0, "horn_sign": {"horn_l": -1.0, "crest": 1.0},
            "draw": ("torso", "obi", "horn_l", "crest", "sleeve")}, info


def segment_back(img):
    a = check_still(img, BACK_N, BACK_BBOX, "back")
    yy, xx = np.mgrid[0:96, 0:96]
    blue = is_col(img, a, BLUES)
    horn = flood(seed_at(a.shape, 57, 25), blue & (yy <= 30) & (xx >= 55))
    red = is_col(img, a, REDS)
    far = flood(seed_at(a.shape, 35, 62), a & ~red & (yy >= 57) & (xx <= 41))
    allow = np.zeros_like(a)
    for y, lim in BSLEEVE_LIM.items():
        allow[y, lim:] = True
    near = flood(seed_at(a.shape, 58, 52), a & allow)
    torso = a & ~horn & ~far & ~near
    parts = {"horn_r": horn, "sleeve_far": far, "sleeve": near, "torso": torso}
    info = finish_parts(parts, BACK_N, "back")
    if int(is_col(img, a, (C_Y,)).sum()):
        raise SystemExit("back view shows an eye")
    pivots = {"horn_r": (55, 28), "sleeve": (55, 47), "sleeve_far": (40, 58),
              "hem_y": 60, "bot_y": 73, "hip_y": 60, "neck_y": 44}
    return {"parts": parts, "pivots": pivots, "eye": None,
            "near": "sleeve", "far": "sleeve_far", "obi": None,
            "horns": ("horn_r",),
            "near_sign": -1.0, "far_sign": 1.0, "horn_sign": {"horn_r": 1.0},
            "draw": ("torso", "sleeve_far", "horn_r", "sleeve")}, info


# ---------------------------------------------------------------- rig
def build_rig(img, seg, flip, kind):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    ox, oy = dst_x - x0, dst_y - y0
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    canvas[..., 3] = np.where(canvas[..., 3] >= 128, 255, 0).astype(np.uint8)
    canvas[canvas[..., 3] == 0] = 0

    def place(m):
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        return pm

    placed = {k: place(m) for k, m in seg["parts"].items()}
    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr
    # Joint cushions: body colour under each moving part's root, drawn only
    # where the torso layer is empty. Hidden at rest by the part itself.
    for part in placed:
        if part == "torso":
            continue
        rad = 2 if part in seg["horns"] else 3
        zone = placed[part] & dilate(placed["torso"], rad)
        fill = nearest_color(canvas, zone, placed["torso"])
        write = zone & (layers["torso"][..., 3] < 128)
        layers["torso"][write] = fill[write]

    pv = {}
    for k, v in seg["pivots"].items():
        pv[k] = (v[0] + ox, v[1] + oy) if isinstance(v, tuple) else v + oy
    near = placed[seg["near"]]
    nys, nxs = np.nonzero(near)
    head_rows = canvas[..., 3] >= 128
    head_rows[pv["neck_y"]:] = False
    hy, hx = np.nonzero(head_rows)
    head = (float(hx.mean()), float(hy.mean()))
    if kind == "front":
        # sleeve tip: lowest-left cuff point
        tip = (float(nxs.min()), float(nys[nxs <= nxs.min() + 1].mean()))
        eye = place(seg["eye"])
        ey, ex = np.nonzero(eye)
        eye_pt = (float(ex.mean()), float(ey.mean()))
        face_pt = (eye_pt[0] - 4, eye_pt[1])
    else:
        tip = (float(nxs.max()), float(nys[nxs >= nxs.max() - 1].mean()))
        eye_pt = None
        face_pt = (head[0] + 6, head[1] - 4)
    face = -1.0 if kind == "front" else 1.0
    mir = 1.0
    if flip:
        def mx(p):
            return (CELL_W - 1 - p[0], p[1])
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        pv = {k: (mx(v) if isinstance(v, tuple) else v) for k, v in pv.items()}
        head, tip, face_pt = mx(head), mx(tip), mx(face_pt)
        if eye_pt is not None:
            eye_pt = mx(eye_pt)
        face, mir = -face, -1.0
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    return {
        "kind": kind, "layers": layers, "pv": pv, "center": center, "head": head,
        "tip": tip, "eye": eye_pt, "face_pt": face_pt, "face": face, "mir": mir,
        "rest": canvas, "draw": seg["draw"], "near": seg["near"], "far": seg["far"],
        "obi": seg["obi"], "horns": seg["horns"], "near_sign": seg["near_sign"],
        "far_sign": seg["far_sign"], "horn_sign": seg["horn_sign"],
        "body_n": int((canvas[..., 3] >= 128).sum()),
    }


# ---------------------------------------------------------------- motion
def wave(f, n, amp, phase=0.0, cycles=1.0):
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def pose(action, f):
    """Angles are 'raise' amounts; each rig maps them to its own rotation sign.

    near: raises the near (facing-side) sleeve outward/forward.
    far : raises the far sleeve (back view only).
    obi : positive lifts the sash tail.
    horn: positive flares the ice horns outward.
    hem : integer cloth ripple at the hem bottom (+ = backward).
    bob : integer dy of the whole body (negative = up).
    """
    n = FRAMES[action]
    p = dict(
        near=0.0, far=0.0, near_dx=0, obi=0.0, horn=0.0, hem=0.0, bob=0, sway=0,
        view="own", narrow=1.0, recoil=0.0, knock=0, flash=0,
        rings=-1, swirl=-1, lines=0, trail=-1, burst=0, sparks=0, glint=0,
    )
    if action == "idle":
        p["bob"] = -int(round(wave(f, n, 1.0, phase=-math.pi / 2)))
        p["near"] = wave(f, n, 5.0, phase=0.0)
        p["far"] = wave(f, n, 5.0, phase=0.0)
        p["obi"] = wave(f, n, 7.0, phase=0.0, cycles=2.0)
        p["horn"] = wave(f, n, 3.0, phase=0.0, cycles=1.0)
        p["hem"] = wave(f, n, 1.4, phase=0.0, cycles=1.0)
    elif action == "walk":
        p["bob"] = -int(round(wave(f, n, 1.0, phase=-math.pi / 2, cycles=2.0)))
        p["near"] = wave(f, n, 12.0, cycles=1.0)
        p["far"] = wave(f, n, 12.0, cycles=1.0, phase=math.pi)
        p["obi"] = wave(f, n, 12.0, cycles=2.0)
        p["horn"] = wave(f, n, 3.0, cycles=2.0)
        p["hem"] = wave(f, n, 2.4, phase=0.0, cycles=2.0)
        p["sway"] = int(round(wave(f, n, 1.0, cycles=1.0)))
        p["lines"] = 1 if f % 4 in (1, 2) else 0
    elif action == "attack":
        near = [0, 8, 16, 22, 24, 28, 34, 34, 34, 34, 34, 30, 22, 12, 4, 0]
        far = [0, 6, 12, 16, 18, 6, 8, 8, 8, 8, 8, 6, 4, 2, 1, 0]
        obi = [0, 8, 14, 18, 20, -10, -14, -14, -14, -14, -14, -10, -6, -3, -1, 0]
        horn = [0, 4, 8, 10, 10, 4, 4, 4, 4, 4, 4, 6, 4, 2, 0, 0]
        bob = [0, 0, -1, -2, -2, -3, -3, -2, 0, 1, 1, 0, 0, 0, 0, 0]
        thrust = [0, 0, 0, 0, 0, 2, 3, 3, 3, 3, 3, 2, 1, 1, 0, 0]
        hem = [0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 2, 1, 1, 0, 0]
        p["near"], p["far"], p["obi"], p["horn"], p["bob"] = near[f], far[f], obi[f], horn[f], bob[f]
        p["near_dx"], p["hem"] = thrust[f], hem[f]
        if 1 <= f <= 4:
            p["rings"] = f - 1
            p["glint"] = 1
        if 5 <= f <= 10:
            p["trail"] = min(f - 5, 2)
        if f in (11, 12):
            p["burst"] = f - 10
    else:
        env = [0, 0.5, 0.85, 1.0, 0.9, 0.7, 0.48, 0.3, 0.16, 0.06, 0.0, 0][f]
        p["recoil"] = 24.0 * env
        p["knock"] = int(round(3 * env))
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 5 else 0
        p["near"] = 16.0 * env
        p["far"] = 12.0 * env
        p["obi"] = 14.0 * env
        p["horn"] = -4.0 * env
        p["hem"] = -2.0 * env
    return p


# ---------------------------------------------------------------- fx
def put(out, x, y, col, over=False):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and (over or out[y, x, 3] < 128):
        out[y, x, :3] = col
        out[y, x, 3] = 255


def frost_rings(out, origin, face, age):
    """Expanding frost shells from the face, wider on the facing side."""
    ox, oy = origin
    cols = (C_W, C_I, C_PU)
    for r_i in range(2):
        rad = 14 + age * 5 + r_i * 6
        col = cols[(age + r_i) % 3]
        for deg in range(-64, 65, 5):
            ang = math.radians(deg)
            for side, k in ((face, 1.0), (-face, 0.75)):
                put(out, ox + side * k * rad * math.cos(ang), oy + 0.75 * rad * math.sin(ang), col)


def glint(out, pt, f):
    """Small cold star next to the eye / horn tip (outside the face pixels)."""
    gx, gy = pt
    r = 2 + (f % 2)
    for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r)):
        put(out, gx + dx, gy + dy, C_I)
    put(out, gx, gy, C_W)


def snow_swirl(out, center, stage):
    cx, cy = center
    for arm in range(3):
        base = arm * 2 * math.pi / 3 + stage * 1.1
        for t in range(14):
            ang = base + t * 0.17
            r = 30 - t * 0.9
            col = C_W if t < 3 else (C_I if t < 8 else C_B)
            put(out, cx + r * math.cos(ang), cy + 0.6 * r * math.sin(ang), col)
    for k in range(6):
        ang = k * math.pi / 3 + stage * 0.7
        sx, sy = cx + 24 * math.cos(ang), cy + 15 * math.sin(ang)
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            put(out, sx + dx, sy + dy, C_W if (dx, dy) == (0, 0) else C_I)


def speed_lines(out, face, seed):
    bb = body_box(out)
    if not bb:
        return
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    base = x1 + 3 if face < 0 else x0 - 3
    side = 1 if face < 0 else -1
    for _ in range(3):
        y = int(rng.integers(y0 + (y1 - y0) // 4, y1 - 4))
        leng = int(rng.integers(4, 9))
        for i in range(leng):
            put(out, base + side * i, y, C_W if i < 2 else C_I)


def ice_shard(out, x, y, face, size):
    """Diamond icicle pointing toward face."""
    for i in range(size * 2 + 1):
        half = size - abs(i - size)
        hx = x + face * (i - size)
        for j in range(-half // 2, half // 2 + 1):
            col = C_W if (j == 0 and i > size) else (C_I if abs(j) < max(1, half // 2) else C_B)
            put(out, hx, y + j, col, over=True)
    put(out, x + face * (size + 1), y, C_W, over=True)


def icicle_trail(out, tip, face, stage):
    tx, ty = tip
    for row, leng in ((-6, 10), (-3, 14), (3, 14), (6, 10)):
        leng += stage * 3
        for i in range(leng):
            x, y = int(round(tx - face * (2 + i))), int(round(ty + row))
            if 0 <= x < out.shape[1] and 0 <= y < out.shape[0] and out[y, x, 3] >= 128:
                break  # stop at the body; never streak into the gaps by the neck
            col = C_W if i < 2 else (C_I if i < 6 else C_B)
            put(out, x, y, col)
    for k, (dx, dy) in enumerate(((6, -5), (10, 2), (7, 7))):
        ice_shard(out, tx + face * (dx + 3 * stage), ty + dy, face, 2 + (k + stage) % 2)


def ice_burst(out, center, size):
    cx, cy = center
    for k in range(6):
        ang = k * math.pi / 3 + math.pi / 6
        for r in range(1, size + 1):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang),
                C_W if r < size - 2 else C_I, over=True)
        # side barbs of the snowflake arms
        for side in (-1, 1):
            bx = cx + (size * 0.6) * math.cos(ang)
            by = cy + (size * 0.6) * math.sin(ang)
            for r in range(1, 3):
                put(out, bx + r * math.cos(ang + side * 1.0), by + r * math.sin(ang + side * 1.0), C_I, over=True)
    for k in range(12):
        ang = k * math.pi / 6
        put(out, cx + (size + 3) * math.cos(ang), cy + (size + 3) * math.sin(ang), C_B)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, C_W, over=True)


def sparks(out, f, back_dir):
    bb = body_box(out)
    if not bb:
        return
    x0, y0, x1, _ = bb
    cx = (x0 + x1) // 2
    for sx, sy in ((cx + back_dir * (8 + f), y0 + 4), (cx + back_dir * 15, y0 + 13),
                   (cx - back_dir * 9, y0 + 7)):
        put(out, sx, sy, C_W)
        put(out, sx + back_dir, sy + 1, C_I)


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 60)
    out[a & ~dark, :3] = C_W
    return out


# ---------------------------------------------------------------- synth
def shear_lean(layer, angle, hip_y, top_y, mir):
    """Lean the rows above hip_y backward; rows above top_y move rigidly."""
    tan = math.tan(math.radians(angle))
    out = np.zeros_like(layer)
    for y in range(layer.shape[0]):
        if y >= hip_y:
            out[y] = layer[y]
            continue
        dy = hip_y - max(y, top_y)
        off = int(round(dy * tan)) * int(mir)
        out[y] = np.roll(layer[y], off, axis=0) if off else layer[y]
    return out


def synth_body(rig, p):
    L = rig["layers"]
    pv = rig["pv"]
    m = rig["mir"]

    def rot(name, ang):
        return rotate_nearest(L[name], m * ang, pv[name])

    moved = {"torso": L["torso"]}
    nl = rot(rig["near"], rig["near_sign"] * p["near"])
    if p["near_dx"]:
        if rig["kind"] == "front":
            nl = shift(nl, rig["face"] * p["near_dx"], -int(round(p["near_dx"] * 0.5)))
        else:
            nl = shift(nl, rig["face"] * p["near_dx"], -int(round(p["near_dx"] * 0.6)))
    moved[rig["near"]] = nl
    if rig["far"]:
        moved[rig["far"]] = rot(rig["far"], rig["far_sign"] * p["far"])
    if rig["obi"]:
        # obi trails behind: lifting it = counter-clockwise in the left-facing still
        moved[rig["obi"]] = rot(rig["obi"], -p["obi"])
    for h in rig["horns"]:
        moved[h] = rot(h, rig["horn_sign"][h] * p["horn"])

    body = np.zeros_like(rig["rest"])
    near_l = None
    for name in rig["draw"]:
        if name == rig["near"]:
            near_l = moved[name]
            continue
        paste(body, moved[name])
    if p["hem"]:
        back = -rig["face"]
        body = row_shear(body, pv["hem_y"], pv["bot_y"], back * p["hem"])
    if p["recoil"] > 0.4:
        back = -rig["face"]
        hip, top = pv["hip_y"], pv["neck_y"]
        body = shear_lean(body, p["recoil"], hip, top, back)
        near_l = shear_lean(near_l, p["recoil"], hip, top, back)
    paste(body, near_l)
    if p["recoil"] > 0.4:
        body = seal_cracks(body)
        body = shift(body, -rig["face"] * p["knock"], 0)
    body = shift(body, p["sway"] * rig["face"], p["bob"])
    moving = (any(abs(float(p[k])) >= 0.45 for k in ("near", "far", "obi", "horn", "hem"))
              or p["near_dx"])
    if moving or p["recoil"] > 0.4:
        body = despeckle(body)
    return body


def synth(rigs, d, action, f, palette, with_fx=True):
    p = pose(action, f)
    own = rigs[d]
    kind, flip = VIEW[d]
    rig = own
    if p["view"] != "own":
        other = "back" if kind == "front" else "front"
        oflip = flip if p["view"] == "other" else (not flip)
        rig = rigs[(other, oflip)]
    body = synth_body(rig, p)
    if p["narrow"] < 0.99:
        body = narrow_x(body, p["narrow"], own["center"][0])
    img = snap(body, palette)
    if not with_fx:
        return img
    bob = p["bob"]
    face = own["face"]
    if p["glint"]:
        hx, hy = own["head"]
        if own["kind"] == "front":
            ex, ey = own["eye"]
            glint(img, (ex + face * 9, ey - 3 + bob), f)
        glint(img, (hx - face * 2, hy - 15 + bob), f + 1)
    if p["rings"] >= 0:
        fx, fy = own["face_pt"]
        frost_rings(img, (fx + face * 3, fy + bob + 2), face, p["rings"])
    if p["swirl"] >= 0:
        snow_swirl(img, (own["center"][0], own["center"][1] + bob), p["swirl"])
    if p["lines"]:
        speed_lines(img, face, seed=f * 13 + 5)
    near_pv = own["pv"]["sleeve"]
    tip = content_point(*own["tip"], own["mir"] * own["near_sign"] * p["near"], near_pv)
    lift = int(round(p["near_dx"] * (0.5 if own["kind"] == "front" else 0.6)))
    tip = (tip[0] + face * p["near_dx"], tip[1] + bob - lift)
    if p["trail"] >= 0:
        icicle_trail(img, (tip[0] + face * 2, tip[1]), face, p["trail"])
    if p["burst"]:
        ice_burst(img, (tip[0] + face * (10 + 3 * p["burst"]), tip[1] - 2), 6 + 3 * p["burst"])
    if p["sparks"]:
        sparks(img, f, -face)
    if p["flash"]:
        img = flash_white(img)
    return snap(img, palette)


# ---------------------------------------------------------------- outputs
def params_hash():
    src = (Path(__file__).read_bytes() + (SRC_DIR / "front.png").read_bytes()
           + (SRC_DIR / "back.png").read_bytes())
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
    Image.fromarray(master, "RGBA").save(OUT_DIR / "all_8dir_actions.png")
    return master


def preview_image(sheets, scale=2):
    """All 8 directions: idle 0, walk 4, attack 3/5/6/9/11, hurt 3."""
    picks = [("idle", 0), ("walk", 4), ("attack", 3), ("attack", 5), ("attack", 6),
             ("attack", 9), ("attack", 11), ("hurt", 3)]
    crop = (CELL_W // 2 - 50, CELL_H - 104, CELL_W // 2 + 50, CELL_H - 12)
    cw, ch = (crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale
    bg = np.full((8 * ch, len(picks) * cw, 4), (15, 20, 31, 255), np.uint8)
    for di in range(8):
        for pi, (act, f) in enumerate(picks):
            tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            t = tile[crop[1]:crop[3], crop[0]:crop[2]]
            t = np.repeat(np.repeat(t, scale, 0), scale, 1)
            m = t[..., 3] >= 128
            bg[di * ch:(di + 1) * ch, pi * cw:(pi + 1) * cw][m] = t[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets, row="S", scale=2):
    ri = DIRS.index(row)
    crop = (CELL_W // 2 - 50, CELL_H - 104, CELL_W // 2 + 50, CELL_H - 12)
    cw, ch = (crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale
    cols = 8
    rows = sum(math.ceil(FRAMES[a] / cols) for a in ACTIONS)
    bg = np.full((rows * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    r = 0
    for act in ACTIONS:
        for f in range(FRAMES[act]):
            rr, cc = divmod(f, cols)
            tile = sheets[act][ri * CELL_H:(ri + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            t = tile[crop[1]:crop[3], crop[0]:crop[2]]
            t = np.repeat(np.repeat(t, scale, 0), scale, 1)
            m = t[..., 3] >= 128
            bg[(r + rr) * ch:(r + rr + 1) * ch, cc * cw:(cc + 1) * cw][m] = t[m]
        r += math.ceil(FRAMES[act] / cols)
    return Image.fromarray(bg, "RGBA")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def with_flock(lock_path, fn):
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def replace_key_value(text, prefix_hint, key, new_val):
    token = '"' + key + '":'
    start = text.find(token)
    if start < 0 or text.find(token, start + len(token)) != -1:
        raise SystemExit(f"{prefix_hint}: key {key} missing or duplicated")
    val_at = start + len(token)
    _old, end = json.JSONDecoder().raw_decode(text, val_at)
    return text[:val_at] + new_val + text[end:]


def publish_dir8(sprite_h):
    entry = {
        "dex": 478, "form": "normal", "variant": "froslass", "gen": "gen4",
        "dir": "monsterSprite/gen4/0478_froslass/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    new_val = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js prefix changed")
        out = replace_key_value(text, "dir8", KEY, new_val)
        before = json.loads(text[len(prefix):].strip().rstrip(";"))
        after = json.loads(out[len(prefix):].strip().rstrip(";"))
        if after[KEY] != entry or list(after) != list(before):
            raise SystemExit("dir8 edit failed")
        for k in before:
            if k != KEY and before[k] != after[k]:
                raise SystemExit(f"dir8 neighbour changed: {k}")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(out)
        tmp.replace(DIR8)

    with_flock(DIR8_LOCK, edit)


def publish_bounds(sheets):
    vals = []
    for di in range(8):
        tile = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        x0, y0, x1, y1 = body_box(tile)
        vals += [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    new_val = json.dumps(vals, separators=(",", ":"))
    text = BOUNDS.read_text()
    s_at = text.find('"s":{')
    token = '"' + KEY + '":'
    pos = text.find(token, s_at)
    if s_at < 0 or pos < 0 or text.find(token, pos + len(token)) != -1:
        raise SystemExit("sprite_bounds s key not unique")
    val_at = pos + len(token)
    _old, end = json.JSONDecoder().raw_decode(text, val_at)
    out = text[:val_at] + new_val + text[end:]
    tmp = BOUNDS.with_suffix(".js.tmp")
    tmp.write_text(out)
    tmp.replace(BOUNDS)
    return vals


def update_queue(status, note=None):
    def edit():
        raw = QUEUE.read_text(encoding="utf-8")
        i = raw.find('"id": "' + KEY + '",')
        if i < 0:
            raise SystemExit("queue item missing")
        j = raw.find('"status": "', i)
        k = raw.find('"', j + len('"status": "'))
        out = raw[:j] + '"status": "' + status + raw[k:]
        if note:
            n0 = out.find('"notes": "', i)
            n1 = out.find('"', n0 + len('"notes": "'))
            while out[n1 - 1] == "\\":
                n1 = out.find('"', n1 + 1)
            out = out[:n1] + " " + json.dumps(note)[1:-1] + out[n1:]
        json.loads(out)
        tmp = QUEUE.with_suffix(".json.tmp")
        tmp.write_text(out, encoding="utf-8")
        tmp.replace(QUEUE)

    with_flock(QUEUE_LOCK, edit)


# ---------------------------------------------------------------- main
def load_rigs():
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    back = np.array(Image.open(SRC_DIR / "back.png").convert("RGBA"))
    for nm, im in (("front", front), ("back", back)):
        if set(np.unique(im[..., 3]).tolist()) - {0, 255}:
            raise SystemExit(f"{nm} alpha is not binary")
    palette = extract_palette(front)
    bp = {tuple(int(v) for v in c) for c in extract_palette(back)}
    fp = {tuple(int(v) for v in c) for c in palette}
    if not bp <= fp:
        raise SystemExit("back uses colours outside the front palette")
    fseg, finfo = segment_front(front)
    bseg, binfo = segment_back(back)
    rigs = {}
    for kind, img, seg in (("front", front, fseg), ("back", back, bseg)):
        for flip in (False, True):
            rigs[(kind, flip)] = build_rig(img, seg, flip, kind)
    for d in DIRS:
        rigs[d] = rigs[VIEW[d]]
    return front, back, palette, fseg, bseg, finfo, binfo, rigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true",
                    help="after visual approval: dir8 + sprite_bounds + queue done")
    args = ap.parse_args()

    front, back, palette, fseg, bseg, finfo, binfo, rigs = load_rigs()
    print("front parts", finfo, "back parts", binfo, "palette", len(palette), flush=True)

    for d in DIRS:
        for act in ACTIONS:
            r0 = synth(rigs, d, act, 0, palette, with_fx=False)
            if np.any(r0 != rigs[d]["rest"]):
                diff = int(np.any(r0 != rigs[d]["rest"], axis=2).sum())
                raise SystemExit(f"{act} frame 0 differs from the still ({d}): {diff}px")
        for act in ("idle", "walk"):
            n = FRAMES[act]
            end, rest = pose(act, n), pose("hurt", 0)
            if any(isinstance(end[k], (int, float)) and abs(float(end[k]) - float(rest[k])) > 1e-6
                   for k in end):
                raise SystemExit(f"{act} loop end is not the still")
    if not (rigs["E"]["face"] > 0 and rigs["W"]["face"] < 0 and rigs["NE"]["face"] > 0
            and rigs["NW"]["face"] < 0):
        raise SystemExit("E/NE must face screen-right and W/NW screen-left")
    print("still diff 0 for all dirs; E faces right, W faces left", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    pcols = {"torso": (180, 180, 190), "horn_l": (60, 220, 220), "crest": (250, 250, 60),
             "horn_r": (60, 220, 220), "sleeve": (40, 90, 220), "sleeve_far": (40, 180, 80),
             "obi": (240, 120, 20)}
    for nm, seg, src in (("front", fseg, front), ("back", bseg, back)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, mm in seg["parts"].items():
            part[mm, :3] = pcols[name]
            part[mm, 3] = 255
        both = np.concatenate([src, part], axis=1)
        Image.fromarray(np.repeat(np.repeat(both, 4, 0), 4, 1), "RGBA").save(work / f"parts_{nm}.png")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "W", f) for f in range(16)]
        want += [("attack", "N", f) for f in range(16)]
        want += [("hurt", "W", f) for f in (0, 1, 3, 5, 8, 11)]
        want += [("hurt", "N", f) for f in (3, 5)]
        want += [("idle", d, 0) for d in DIRS]
        want += [("walk", "N", f) for f in (0, 4, 8, 12)]
        want += [("walk", "E", f) for f in (0, 4, 8, 12)]
        tiles = []
        crop = (CELL_W // 2 - 50, CELL_H - 104, CELL_W // 2 + 50, CELL_H - 12)
        for act, d, f in want:
            bare = synth(rigs, d, act, f, palette, with_fx=False)
            tile = synth(rigs, d, act, f, palette)
            sizes = component_sizes(bare[..., 3] >= 128)
            print(f"sample {act} {d} {f:02d} comps={sizes[:4]} edge={touches_edge(tile)}", flush=True)
            tiles.append(tile[crop[1]:crop[3], crop[0]:crop[2]])
        cols = 8
        th, tw = tiles[0].shape[:2]
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * th, cols * tw, 4), (15, 20, 31, 255), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            mm = t[..., 3] >= 128
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw][mm] = t[mm]
        sheet = np.repeat(np.repeat(sheet, 3, 0), 3, 1)
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
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions_dir = SRC_DIR / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    preview.save(actions_dir / "preview.png")
    motion_contact(sheets, "W").save(OUT_DIR / "preview_motion.png")
    motion_contact(sheets, "N").save(work / "review_N.png")
    motion_contact(sheets, "E").save(work / "review_E.png")

    idle0 = sheets["idle"][:CELL_H, :CELL_W]
    bb = body_box(idle0)
    sprite_h = bb[3] - bb[1] + 1
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]).tolist())
    edge = []
    split = []
    still_diff = 0
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                if touches_edge(tile):
                    edge.append(f"{act}:{d}:{f}")
                bare = synth(rigs, d, act, f, palette, with_fx=False)
                sizes = component_sizes(bare[..., 3] >= 128)
                if len([s for s in sizes if s >= 20]) != 1:
                    split.append({"a": act, "d": d, "f": f, "sizes": sizes[:5]})
            t0 = synth(rigs, d, act, 0, palette, with_fx=False)
            still_diff += int(np.any(t0 != rigs[d]["rest"], axis=2).sum())
            if act != "attack":
                t0 = sheets[act][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
                still_diff += int(np.any(t0 != rigs[d]["rest"], axis=2).sum())
    report = {
        "version": VERSION, "hash": digest, "front_parts": finfo, "back_parts": binfo,
        "sprite_h": int(sprite_h), "off_palette": len(off), "alphas": alphas,
        "edge_touching": len(edge), "edge_list": edge[:20], "still_diff": still_diff,
        "split": split[:20], "split_n": len(split), "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_W"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "W"),
                 "--fps", "12", "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
            else:
                report[f"gait_{act}"] = "no report"
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False)[:2000], flush=True)
    meta = {
        "generator": "forge_froslass_actions", "version": VERSION, "key": KEY, "dex": 478,
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(sprite_h),
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "views": {d: {"still": VIEW[d][0], "mirrored": VIEW[d][1]} for d in DIRS},
        "still_faces": "front.png faces screen-left; back.png faces up-right",
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "source": "front.png + back.png 96x96", "gifs_untouched": True,
        "roar": "no mouth pixels: ice horns/crest flare, sleeve and obi rise, frost rings from the face",
        "attack": "pirouette via the real back view (no in-plane spin) + ice sleeve sweep + ice burst",
        "legs": 0, "layout": "actions_8dir", "sheetDir": "agent-sprite-forge", "raw": 1,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = (report["off_palette"] or report["edge_touching"] or report["still_diff"]
              or report["split_n"] or alphas != [0, 255])
    if failed:
        print("QA FAILED", flush=True)
        return
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        publish_dir8(sprite_h)
        vals = publish_bounds(sheets)
        update_queue("done")
        print("published dir8", VERSION, "h", sprite_h, "bounds", vals[:4], flush=True)


if __name__ == "__main__":
    main()
