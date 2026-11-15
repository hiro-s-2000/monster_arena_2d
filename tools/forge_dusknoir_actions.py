#!/usr/bin/env python3
"""Dusknoir (0477) normal 8-direction action sheet from this form's front.png / back.png.

Measured on normal/front.png: 96x96, 2268 opaque, bbox x11-83 y16-80, 12
colours, alpha 0/255, one 4-connected piece.

Body plan (from the pixels): legless hovering ghost. No legs, no face mouth.
  antenna    yellow ring on top of the hood, x48-61 y16-22
  head/hood  grey dome x42-58 y22-39, ONE red eye x45-48 y40-41 (6px)
  pads       pale shoulder pads either side of the hood (stay on the torso)
  arm_l      screen-left forearm (yellow bands) + big grey hand x11-38 y40-62
  arm_r      screen-right forearm + hand x64-84 y38-65
  belly      the yellow zigzag across the belly (x36-72 y52-66) is the belly
             mouth. The roar opens it: everything under the zigzag (lower
             belly + tail) drops 1-4px and the gap is mouth interior.
  tail       ghost wisp below the belly x51-70 y68-80
The eye sits left of the hood centre, so the still faces screen-left.

back.png (2176 opaque, subset of the front palette) is the back view turned
up-right: hood + antenna, one raised hand x58-82 y25-41 with its forearm,
wisp tail lower-left.

Directions: front still W SW unflipped, S SE E mirrored (faces right).
back still N NE unflipped, NW mirrored. E faces screen-right, W screen-left.

  idle   : hover bob (integer), arms sway, tail wisp sways. Frame 0 and the
           loop end are the still.
  walk   : glide (no legs): hover bob x2, arms row alternately, tail trails.
  attack : 0-4 roar: belly mouth opens + roar rings, arms spread, antenna
           wobble (back rows: antenna wobble + rings from the hood).
           5-7 pirouette: the body turns around (front -> side -> back ->
           side, using the real back view; never an in-plane rotation),
           ghost swirl. 8-10 shadow punch with the near arm (<= 34 deg plus an
           integer thrust) and a shadow trail. 11-12 impact. 13-15 recover.
  hurt   : 12 frames, white flash, backward lean <= 24 deg as an integer row
           shear (hood rigid, face never resamples), knock back, recover.

Palette-locked to this still. front/back png and gif are not written.
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
SRC_DIR = ROOT / "monsterSprite/gen4/0477_dusknoir/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0477_dusknoir/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
BOUNDS = ROOT / "data/sprite_bounds.js"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# which still and whether it is mirrored, per direction
VIEW = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False), "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "dusknoir_v2"
KEY = "0477_dusknoir"

C_K = (0, 0, 0)
C_D = (65, 57, 57)
C_P = (98, 90, 98)
C_M = (98, 82, 82)
C_G = (131, 115, 115)
C_L = (172, 164, 156)
C_W = (205, 197, 172)
C_O = (90, 74, 49)
C_Y1 = (156, 123, 41)
C_Y = (222, 180, 49)
C_R = (189, 49, 24)
YELLOWS = (C_O, C_Y1, C_Y)

FRONT_N = 2268
FRONT_BBOX = (11, 83, 16, 80)
BACK_N = 2176
BACK_BBOX = (16, 80, 14, 81)


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


def segment_front(img):
    a = check_still(img, FRONT_N, FRONT_BBOX, "front")
    yy, xx = np.mgrid[0:96, 0:96]
    yel = is_col(img, a, YELLOWS)
    antenna = (a & (yy <= 21)) | (yel & (yy == 22))
    head = a & ~antenna & (
        ((yy <= 39) & (xx >= 42) & (xx <= 58)) | ((yy >= 40) & (yy <= 42) & (xx >= 42) & (xx <= 54))
    )
    allow_l = a & (yy >= 40) & (xx <= 38) & ~((xx >= 34) & (yy <= 42))
    arm_l = flood(a & (yy == 52) & (xx <= 16), allow_l)
    allow_r = a & (yy >= 38) & (
        ((yy <= 51) & (xx >= 64)) | ((yy >= 52) & (yy <= 55) & (xx >= 66)) | ((yy >= 56) & (xx >= 70))
    ) & ~((yy <= 38) & (xx <= 65))
    arm_r = flood(a & (yy == 55) & (xx >= 80), allow_r)
    tail = a & ((yy >= 71) | ((yy >= 69) & (xx >= 59) & (xx <= 72)))
    torso = a & ~antenna & ~head & ~arm_l & ~arm_r & ~tail
    parts = {"antenna": antenna, "head": head, "arm_l": arm_l, "arm_r": arm_r,
             "tail": tail, "torso": torso}
    total = sum(int(m.sum()) for m in parts.values())
    if total != FRONT_N:
        raise SystemExit(f"front parts overlap or miss pixels: {total}")
    for name, m in parts.items():
        sz = component_sizes(m)
        if name not in ("torso",) and len(sz) != 1:
            raise SystemExit(f"front {name} is not one piece: {sz[:5]}")
    eye = is_col(img, a, (C_R,))
    if int(eye.sum()) != 6 or int((eye & ~head).sum()):
        raise SystemExit("eye changed or not on the hood")
    eye_x = float(np.nonzero(eye)[1].mean())
    hx = np.nonzero(head)[1]
    head_cx = (float(hx.min()) + float(hx.max())) / 2
    if not eye_x < head_cx - 1:
        raise SystemExit(f"front no longer faces left: eye {eye_x:.1f} head {head_cx:.1f}")
    # Belly mouth: lowest zigzag row per column; the lower jaw is the torso
    # under it plus the tail.
    zig = yel & (yy >= 52) & (yy <= 67) & (xx >= 36) & (xx <= 72) & ~arm_l & ~arm_r
    lower = np.zeros_like(a)
    lip = {}
    for x in range(36, 73):
        col = np.nonzero(zig[:, x])[0]
        if len(col) == 0:
            continue
        yb = int(col.max())
        lip[x] = yb
        lower[yb + 1:, x] = (torso[yb + 1:, x])
    lower |= tail
    if len(lip) < 30:
        raise SystemExit(f"belly zigzag too short: {len(lip)}")
    info = {k: int(v.sum()) for k, v in parts.items()}
    info.update(eye_x=round(eye_x, 1), head_cx=round(head_cx, 1), zig_cols=len(lip),
                jaw=int(lower.sum()))
    pivots = {"arm_l": (37, 45), "arm_r": (64, 44), "tail": (63, 69),
              "antenna": (55, 22), "hip_y": 67, "neck_y": 42}
    return {"parts": parts, "lower": lower, "lip": lip, "pivots": pivots,
            "eye": eye, "draw": ("tail", "torso", "head", "antenna", "arm_r", "arm_l"),
            "near": "arm_l", "far": "arm_r"}, info


def segment_back(img):
    a = check_still(img, BACK_N, BACK_BBOX, "back")
    yy, xx = np.mgrid[0:96, 0:96]
    antenna = a & (yy <= 18)
    head = a & ~antenna & (yy <= 33) & (xx >= 33) & (xx <= 52)
    allow = a & (((yy <= 41) & (xx >= 57)) | ((yy >= 41) & (yy <= 56) & (xx >= 53)))
    arm = flood(a & (yy == 30) & (xx >= 72), allow)
    tail = a & (yy >= 68)
    torso = a & ~antenna & ~head & ~arm & ~tail
    parts = {"antenna": antenna, "head": head, "arm_l": arm, "tail": tail, "torso": torso}
    total = sum(int(m.sum()) for m in parts.values())
    if total != BACK_N:
        raise SystemExit(f"back parts overlap or miss pixels: {total}")
    for name, m in parts.items():
        sz = component_sizes(m)
        if name != "torso" and len(sz) != 1:
            raise SystemExit(f"back {name} is not one piece: {sz[:5]}")
    if int(is_col(img, a, (C_R,)).sum()):
        raise SystemExit("back view shows an eye")
    info = {k: int(v.sum()) for k, v in parts.items()}
    pivots = {"arm_l": (54, 49), "tail": (31, 68), "antenna": (42, 19),
              "hip_y": 64, "neck_y": 33}
    return {"parts": parts, "lower": None, "lip": {}, "pivots": pivots, "eye": None,
            "draw": ("tail", "torso", "head", "antenna", "arm_l"),
            "near": "arm_l", "far": None}, info


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
    # Joint cushions: torso colour under each arm root / tail root / antenna
    # base, drawn only where the torso (or hood) layer is empty. Hidden at rest.
    for part, rad in (("arm_l", 3), ("arm_r", 3), ("tail", 2)):
        if part not in placed:
            continue
        zone = placed[part] & dilate(placed["torso"], rad)
        rgb = canvas[..., :3]
        plain = placed["torso"].copy()
        for c in YELLOWS:
            plain &= ~np.all(rgb == np.array(c, np.uint8), axis=-1)
        fill = nearest_color(canvas, zone, plain)
        write = zone & (layers["torso"][..., 3] < 128)
        layers["torso"][write] = fill[write]
    zone = placed["antenna"] & dilate(placed["head"], 2)
    fill = nearest_color(canvas, zone, placed["head"])
    write = zone & (layers["head"][..., 3] < 128)
    layers["head"][write] = fill[write]

    lower = place(seg["lower"]) if seg["lower"] is not None else None
    lip = {x + ox: y + oy for x, y in seg["lip"].items()}
    pv = {}
    for k, v in seg["pivots"].items():
        pv[k] = (v[0] + ox, v[1] + oy) if isinstance(v, tuple) else v + oy
    near = placed[seg["near"]]
    nys, nxs = np.nonzero(near)
    hy, hx = np.nonzero(placed["head"])
    hood = (float(hx.mean()), float(hy.mean()))
    if kind == "front":
        jaw_pts = np.nonzero(lower)
        mouth = (float(np.mean(list(lip.keys()))), float(np.mean(list(lip.values()))) + 1)
        fist = (float(nxs.min()), float(nys[nxs <= nxs.min() + 2].mean()))
        eye = place(seg["eye"])
        ey, ex = np.nonzero(eye)
        eye_pt = (float(ex.mean()), float(ey.mean()))
        del jaw_pts
    else:
        mouth = (hood[0], hood[1] - 4)
        fist = (float(nxs.max()), float(nys[nxs >= nxs.max() - 2].mean()))
        eye_pt = None
    ays, axs = np.nonzero(placed["antenna"])
    ant = (float(axs.mean()), float(ays.min()))
    # facing on screen in the unflipped still: front looks left, back up-right
    face = -1.0 if kind == "front" else 1.0
    mir = 1.0
    if flip:
        def mx(p):
            return (CELL_W - 1 - p[0], p[1])
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        if lower is not None:
            lower = np.ascontiguousarray(lower[:, ::-1])
        lip = {CELL_W - 1 - x: y for x, y in lip.items()}
        pv = {k: (mx(v) if isinstance(v, tuple) else v) for k, v in pv.items()}
        hood, mouth, fist, ant = mx(hood), mx(mouth), mx(fist), mx(ant)
        if eye_pt is not None:
            eye_pt = mx(eye_pt)
        face, mir = -face, -1.0
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    return {
        "kind": kind, "layers": layers, "pv": pv, "center": center, "hood": hood,
        "mouth": mouth, "fist": fist, "ant": ant, "eye": eye_pt, "lower": lower,
        "lip": lip, "face": face, "mir": mir, "rest": canvas, "draw": seg["draw"],
        "near": seg["near"], "far": seg["far"],
        "body_n": int((canvas[..., 3] >= 128).sum()),
    }


# ---------------------------------------------------------------- motion
def wave(f, n, amp, phase=0.0, cycles=1.0):
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def pose(action, f):
    """Angles are in the unflipped still's frame, positive = clockwise.

    near (arm_l): positive raises the near hand outward/up (front still).
    far (arm_r): negative raises the far hand outward/up.
    bob: integer dy of the whole body (negative = up).
    """
    n = FRAMES[action]
    p = dict(
        near=0.0, far=0.0, near_dx=0, near_dy=0, tail=0.0, ant=0.0, bob=0, sway=0,
        mouth=0, view="own", narrow=1.0, recoil=0.0, knock=0, flash=0,
        rings=-1, swirl=-1, lines=0, trail=-1, burst=0, sparks=0, glow=0,
    )
    if action == "idle":
        p["bob"] = int(round(wave(f, n, 1.0, phase=-math.pi / 2)))  # 0..2 rows up
        p["bob"] = -p["bob"]
        p["near"] = wave(f, n, 4.0, phase=0.0)
        p["far"] = wave(f, n, 4.0, phase=0.0)
        p["tail"] = wave(f, n, 6.0, phase=0.0, cycles=1.0)
        p["ant"] = wave(f, n, 3.0, phase=0.0, cycles=1.0)
    elif action == "walk":
        p["bob"] = -int(round(wave(f, n, 1.0, phase=-math.pi / 2, cycles=2.0)))
        p["near"] = wave(f, n, 10.0, cycles=1.0)
        p["far"] = wave(f, n, 10.0, cycles=1.0)
        p["tail"] = wave(f, n, 9.0, phase=0.0, cycles=2.0)
        p["ant"] = wave(f, n, 3.0, cycles=2.0)
        p["sway"] = int(round(wave(f, n, 1.0, cycles=1.0)))
        p["lines"] = 1 if f % 4 in (1, 2) else 0
    elif action == "attack":
        near = [0, 8, 16, 20, 22, 30, 34, 34, 34, 34, 34, 30, 22, 12, 4, 0]
        far = [0, -8, -16, -20, -22, 8, 10, 10, 10, 10, 10, 8, 5, 3, 1, 0]
        mouth = [0, 1, 2, 3, 4, 2, 2, 2, 2, 2, 2, 3, 2, 1, 0, 0]
        ant = [0, -6, 6, -8, 8, 6, -8, 8, 0, 0, 0, -4, 3, -1, 0, 0]
        bob = [0, 0, -1, -2, -2, -3, -3, -2, 0, 1, 1, 0, 0, 0, 0, 0]
        thrust = [0, 0, 0, 0, 0, 3, 4, 4, 4, 4, 4, 3, 2, 1, 0, 0]
        p["near"], p["far"], p["mouth"], p["ant"], p["bob"] = near[f], far[f], mouth[f], ant[f], bob[f]
        p["near_dx"] = thrust[f]
        p["tail"] = [0, 4, 8, 10, 10, -8, -10, -10, -10, -10, -10, -6, -3, -1, 0, 0][f]
        if 1 <= f <= 4:
            p["rings"] = f - 1
            p["glow"] = 1
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
        p["near"] = 14.0 * env
        p["far"] = -14.0 * env
        p["tail"] = 10.0 * env
        p["ant"] = -6.0 * env
    return p


# ---------------------------------------------------------------- fx
def put(out, x, y, col, over=False):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and (over or out[y, x, 3] < 128):
        out[y, x, :3] = col
        out[y, x, 3] = 255


def roar_rings(out, origin, face, age):
    """Expanding roar shells around the mouth, both sides, wider on the facing side."""
    ox, oy = origin
    cols = (C_Y, C_W, C_R)
    for r_i in range(2):
        rad = 30 + age * 5 + r_i * 6
        col = cols[(age + r_i) % 3]
        for deg in range(-60, 61, 4):
            ang = math.radians(deg)
            for side, k in ((face, 1.0), (-face, 0.8)):
                put(out, ox + side * k * rad * math.cos(ang), oy + 0.7 * rad * math.sin(ang), col)


def eye_glow(out, eye):
    if eye is None:
        return
    ex, ey = eye
    for dx, dy in ((-4, 0), (4, 0), (0, -3), (-3, -2), (3, -2), (-3, 2), (3, 2)):
        put(out, ex + dx, ey + dy, C_R if abs(dx) < 4 else C_Y)


def antenna_glow(out, ant, f):
    ax, ay = ant
    for k in range(6):
        ang = k * math.pi / 3 + f * 0.5
        put(out, ax + 8 * math.cos(ang), ay + 2 + 5 * math.sin(ang), C_Y if k % 2 else C_W)


def ghost_swirl(out, center, stage):
    cx, cy = center
    for arm in range(3):
        base = arm * 2 * math.pi / 3 + stage * 1.1
        for t in range(14):
            ang = base + t * 0.16
            r = 34 - t * 0.9
            col = C_W if t < 3 else (C_Y if t < 7 else C_P)
            put(out, cx + r * math.cos(ang), cy + 0.55 * r * math.sin(ang), col)


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
            put(out, base + side * i, y, C_W if i < 2 else C_P)


def shadow_trail(out, fist, face, stage):
    """Shadow-punch trail: dark streaks behind the fist plus a yellow rim."""
    fx, fy = fist
    # streaks above and below the forearm, trailing back toward the shoulder
    for row, leng in ((-9, 16), (-12, 10), (9, 16), (12, 10)):
        leng += stage * 3
        for i in range(leng):
            col = C_W if i < 2 else (C_Y if i < 5 else C_P)
            put(out, fx - face * (2 + i), fy + row, col)
    # shadow fist rim in front of the hand
    for rad, col in ((8, C_P), (9, C_Y), (11, C_R)):
        for deg in range(-70, 71, 8):
            ang = math.radians(deg)
            put(out, fx + face * (rad * math.cos(ang) - 2), fy + rad * math.sin(ang), col)


def burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4 + math.pi / 8
        for r in range(max(2, size // 2), size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang),
                C_W if r < size - 2 else C_Y, over=True)
    for k in range(12):
        ang = k * math.pi / 6
        put(out, cx + (size + 2) * math.cos(ang), cy + (size + 2) * math.sin(ang), C_R)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, C_W, over=True)


def sparks(out, f, back_dir):
    bb = body_box(out)
    if not bb:
        return
    x0, y0, x1, _ = bb
    cx = (x0 + x1) // 2
    for sx, sy in ((cx + back_dir * (8 + f), y0 + 4), (cx + back_dir * 16, y0 + 14),
                   (cx - back_dir * 10, y0 + 8)):
        put(out, sx, sy, C_W)
        put(out, sx + back_dir, sy + 1, C_Y)


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 40)
    out[a & ~dark, :3] = C_W
    return out


# ---------------------------------------------------------------- synth
def open_mouth(torso_tail, rig, d):
    """Drop the lower jaw (lower belly + tail) by d rows; fill the gap as mouth."""
    if d <= 0 or rig["lower"] is None:
        return torso_tail
    low = rig["lower"]
    lower_l = np.zeros_like(torso_tail)
    lower_l[low] = torso_tail[low]
    out = torso_tail.copy()
    out[low] = 0
    for x, yb in rig["lip"].items():
        if not low[yb + 1:, x].any():
            continue
        for k in range(1, d + 1):
            y = yb + k
            col = C_R if (d >= 3 and k == (d + 1) // 2) else C_K
            out[y, x, :3] = col
            out[y, x, 3] = 255
    paste(out, shift(lower_l, 0, d))
    return out


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

    moved = {}
    near, far = rig["near"], rig["far"]
    nl = rot(near, p["near"])
    # thrust toward the facing side for the front; up-right for the back view
    if p["near_dx"]:
        if rig["kind"] == "front":
            nl = shift(nl, rig["face"] * p["near_dx"], 0)
        else:
            nl = shift(nl, rig["face"] * p["near_dx"], -int(round(p["near_dx"] * 0.6)))
    moved[near] = nl
    if far:
        moved[far] = rot(far, p["far"])
    moved["tail"] = rot("tail", p["tail"])
    moved["antenna"] = rot("antenna", p["ant"])
    moved["torso"] = L["torso"]
    moved["head"] = L["head"]

    base = np.zeros_like(rig["rest"])
    # torso first: the tail-root cushion lives on the torso layer and must sit
    # under the tail
    paste(base, moved["torso"])
    paste(base, moved["tail"])
    if p["mouth"]:
        base = open_mouth(base, rig, int(p["mouth"]))
    hood = np.zeros_like(base)
    paste(hood, moved["head"])
    paste(hood, moved["antenna"])
    arms = np.zeros_like(base)
    for name in rig["draw"]:
        if name in (near, far) and name in moved:
            paste(arms, moved[name])

    if p["recoil"] > 0.4:
        # backward = away from the facing side
        back = -rig["face"]
        hip, top = rig["pv"]["hip_y"], rig["pv"]["neck_y"]
        base = shear_lean(base, p["recoil"], hip, top, back)
        arms = shear_lean(arms, p["recoil"], hip, top, back)
        off = int(round((hip - top) * math.tan(math.radians(p["recoil"])))) * int(back)
        hood = shift(hood, off, 1 if p["recoil"] > 12 else 0)
    body = base
    paste(body, hood)
    paste(body, arms)
    if p["recoil"] > 0.4:
        body = seal_cracks(body)
        body = shift(body, -rig["face"] * p["knock"], 0)
    body = shift(body, p["sway"] * rig["face"], p["bob"])
    moving = any(abs(float(p[k])) >= 0.45 for k in ("near", "far", "tail", "ant")) or p["near_dx"]
    if moving or p["mouth"] or p["recoil"] > 0.4:
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
    if p["glow"]:
        if own["kind"] == "front":
            eye_glow(img, (own["eye"][0], own["eye"][1] + bob))
        antenna_glow(img, (own["ant"][0], own["ant"][1] + bob), f)
    if p["rings"] >= 0:
        mo = own["mouth"]
        roar_rings(img, (mo[0] + face * 4, mo[1] + bob + p["mouth"] // 2), face, p["rings"])
    if p["swirl"] >= 0:
        ghost_swirl(img, (own["center"][0], own["center"][1] + bob), p["swirl"])
    if p["lines"]:
        speed_lines(img, face, seed=f * 13 + 5)
    fist = content_point(*own["fist"], own["mir"] * p["near"], own["pv"]["arm_l"])
    if own["kind"] == "front":
        fist = (fist[0] + face * p["near_dx"], fist[1] + bob)
    else:
        fist = (fist[0] + face * p["near_dx"], fist[1] + bob - int(round(p["near_dx"] * 0.6)))
    if p["trail"] >= 0:
        shadow_trail(img, (fist[0] + face * 2, fist[1]), face, p["trail"])
    if p["burst"]:
        burst(img, (fist[0] + face * (8 + 3 * p["burst"]), fist[1]), 6 + 3 * p["burst"])
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


def preview_image(sheets):
    """All 8 directions: idle 0, walk 4, attack 2/4/6/9/11, hurt 3."""
    picks = [("idle", 0), ("walk", 4), ("attack", 3), ("attack", 5), ("attack", 6),
             ("attack", 9), ("attack", 11), ("hurt", 3)]
    crop = (CELL_W // 2 - 70, CELL_H - 140, CELL_W // 2 + 70, CELL_H - 8)
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    bg = np.full((8 * ch, len(picks) * cw, 4), (15, 20, 31, 255), np.uint8)
    for di in range(8):
        for pi, (act, f) in enumerate(picks):
            tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            t = tile[crop[1]:crop[3], crop[0]:crop[2]]
            m = t[..., 3] >= 128
            bg[di * ch:(di + 1) * ch, pi * cw:(pi + 1) * cw][m] = t[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets, row="S", scale=2):
    ri = DIRS.index(row)
    crop = (CELL_W // 2 - 64, CELL_H - 136, CELL_W // 2 + 64, CELL_H - 12)
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
        "dex": 477, "form": "normal", "variant": "dusknoir", "gen": "gen4",
        "dir": "monsterSprite/gen4/0477_dusknoir/normal", "sheetDir": "agent-sprite-forge",
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
    if not (rigs["E"]["eye"][0] > rigs["E"]["hood"][0] and rigs["W"]["eye"][0] < rigs["W"]["hood"][0]):
        raise SystemExit("E must face screen-right and W screen-left")
    print("still diff 0 for all dirs; E faces right, W faces left", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    pcols = {"torso": (180, 180, 190), "head": (250, 250, 250), "antenna": (60, 220, 220),
             "arm_l": (40, 90, 220), "arm_r": (40, 180, 80), "tail": (240, 120, 20)}
    for nm, seg in (("front", fseg), ("back", bseg)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in seg["parts"].items():
            part[m, :3] = pcols[name]
            part[m, 3] = 255
        if seg["lower"] is not None:
            part[seg["lower"] & seg["parts"]["torso"], :3] = (200, 60, 200)
        Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / f"parts_{nm}.png")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "W", f) for f in range(16)]
        want += [("attack", "N", f) for f in range(16)]
        want += [("hurt", "W", f) for f in (0, 1, 3, 5, 8, 11)]
        want += [("idle", d, 0) for d in DIRS]
        want += [("hurt", "N", f) for f in (3, 5)]
        tiles = []
        crop = (CELL_W // 2 - 64, CELL_H - 136, CELL_W // 2 + 64, CELL_H - 12)
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
            m = t[..., 3] >= 128
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
        sheet = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
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
        "generator": "forge_dusknoir_actions", "version": VERSION, "key": KEY, "dex": 477,
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(sprite_h),
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "views": {d: {"still": VIEW[d][0], "mirrored": VIEW[d][1]} for d in DIRS},
        "still_faces": "front.png faces screen-left; back.png faces up-right",
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "source": "front.png + back.png 96x96", "gifs_untouched": True,
        "roar": "existing belly zigzag mouth opens + roar rings (back rows: antenna wobble + rings)",
        "attack": "pirouette via the real back view (no in-plane spin) + shadow punch with the near arm",
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
