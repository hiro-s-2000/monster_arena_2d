#!/usr/bin/env python3
"""Rotom (0479) normal 8-direction action sheet from this form's front.png / back.png.

Measured on normal/front.png: 96x96, 813 opaque, bbox x17-77 y23-72, 12
colours, alpha 0/255, one 4-connected piece.

Body plan (from the pixels): legless floating plasma orb. No limbs, no legs.
  spike    the cyan plasma tuft with an orange core on top, y23-36
  orb      orange body x36-58 y37-53 with a cyan aura rim. Two blue eyes
           (left eye x40-44, right eye x48-52, y42-49) and a tiny smirk mouth:
           black (46,50) (44,51) (45,51). The face sits left of the orb centre.
  bolt_l   screen-left lightning-bolt arm (cyan, white core) x17-40 y51-67
  bolt_r   screen-right lightning-bolt arm x54-77 y52-72
  tail     short plasma spike under the orb, x41-53 y54-65
The eyes + mouth sit left of the orb centre, so front.png faces screen-left.

back.png (810 opaque, same palette) is the back view looking up-right: the
same spike / orb / bolts / tail, one blue eye on the right rim x55-57.

Directions: front still W SW unflipped, S SE E mirrored (faces right).
back still N NE unflipped, NW mirrored. E faces screen-right, W screen-left.

  idle   : hover bob (integer), bolts sway, spike and tail wobble, a small
           crackle at the bolt tips. Frame 0 and the loop end are the still.
  walk   : glide (no legs): hover bob x2, the bolts row alternately like
           fins, tail trails, speed lines. Same loop rule.
  attack : 0-4 roar: the existing smirk opens (wedge on the orb) + roar rings,
           both bolts flare up, spike flares, plasma crackle (back rows: flare
           + rings from the orb top). 5-7 turn-around: the body turns (front ->
           side -> back -> side using the real back view, never an in-plane
           rotation) inside an electric swirl. 8-10 thunder strike: the near
           bolt swings forward (<= 34 deg) and fires a forked lightning bolt.
           11-12 impact. 13-15 recover.
  hurt   : 12 frames, white flash, knock back + lift (integer), bolts jolt up
           <= 24 deg, spike bends back <= 24 deg, electric sparks, recover.
The orb (eyes + mouth) only ever moves by integer pixels.

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
SRC_DIR = ROOT / "monsterSprite/gen4/0479_rotom/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0479_rotom/normal/agent-sprite-forge"
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
VERSION = "rotom_v2"
KEY = "0479_rotom"

C_K = (16, 16, 16)
C_BR = (123, 57, 32)
C_OD = (197, 74, 24)
C_O = (238, 115, 41)
C_OL = (255, 172, 131)
C_W = (255, 255, 255)
C_CL = (189, 246, 238)
C_C = (90, 197, 189)
C_T = (41, 172, 189)
C_B = (8, 65, 172)
C_BM = (49, 123, 238)
C_G = (189, 189, 189)
BLUES = (C_B, C_BM)

FRONT_N = 813
FRONT_BBOX = (17, 77, 23, 72)
BACK_N = 810
BACK_BBOX = (18, 78, 23, 71)
# the smirk in front.png (x, y)
SMIRK = ((46, 50), (44, 51), (45, 51))
# mouth wedge per opening level, front.png coords: (x, y, colour)
MOUTH = {
    1: ((46, 51, C_K), (44, 52, C_K), (45, 52, C_K)),
    2: ((46, 51, C_K), (43, 52, C_K), (44, 52, C_BR), (45, 52, C_BR), (46, 52, C_K),
        (44, 53, C_K), (45, 53, C_K)),
    3: ((46, 51, C_K), (47, 51, C_K), (43, 52, C_K), (44, 52, C_BR), (45, 52, C_OD),
        (46, 52, C_BR), (47, 52, C_K), (43, 53, C_K), (44, 53, C_BR), (45, 53, C_BR),
        (46, 53, C_K)),
}


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


def contact_pivot(part, body):
    """Mean of the part pixels that touch the body: the joint the part turns on."""
    touch = part & dilate(body, 1)
    ys, xs = np.nonzero(touch)
    if len(ys) == 0:
        raise SystemExit("part does not touch the body")
    return (float(xs.mean()), float(ys.mean()))


def segment(img, kind):
    if kind == "front":
        a = check_still(img, FRONT_N, FRONT_BBOX, "front")
    else:
        a = check_still(img, BACK_N, BACK_BBOX, "back")
    x0 = FRONT_BBOX[0] if kind == "front" else BACK_BBOX[0]
    yy, xx = np.mgrid[0:96, 0:96]
    spike = a & (yy <= 36)
    allow_l = a & (yy >= 51) & (xx <= x0 + 23) & ~((yy == 51) & (xx > x0 + 20))
    bolt_l = flood(a & (yy >= 64) & (xx <= x0 + 3), allow_l)
    allow_r = a & (yy >= 51) & (xx >= x0 + 37) & ~((yy <= 52) & (xx < x0 + 40))
    bolt_r = flood(a & (yy >= 66) & (xx >= x0 + 55), allow_r)
    tail = a & (yy >= 54) & ~bolt_l & ~bolt_r
    orb = a & ~spike & ~bolt_l & ~bolt_r & ~tail
    parts = {"spike": spike, "orb": orb, "bolt_l": bolt_l, "bolt_r": bolt_r, "tail": tail}
    total = sum(int(m.sum()) for m in parts.values())
    n_exp = FRONT_N if kind == "front" else BACK_N
    if total != n_exp:
        raise SystemExit(f"{kind} parts overlap or miss pixels: {total}")
    for name, m in parts.items():
        sz = component_sizes(m)
        if len(sz) != 1:
            raise SystemExit(f"{kind} {name} is not one piece: {sz[:5]}")
    bw = {k: (int(np.nonzero(m)[1].min()), int(np.nonzero(m)[1].max())) for k, m in parts.items()}
    if not (bw["bolt_l"][0] == x0 and bw["bolt_r"][1] >= x0 + 59):
        raise SystemExit(f"{kind} bolts not at the sprite ends: {bw}")
    eye = is_col(img, a, BLUES)
    if int((eye & ~orb).sum()):
        raise SystemExit(f"{kind} eye pixels outside the orb")
    oys, oxs = np.nonzero(orb)
    orb_c = (float(oxs.mean()), float(oys.mean()))
    eys, exs = np.nonzero(eye)
    eye_pt = (float(exs.mean()), float(eys.mean()))
    if kind == "front":
        for x, y in SMIRK:
            if tuple(int(v) for v in img[y, x, :3]) != C_K or not orb[y, x]:
                raise SystemExit("smirk moved; remeasure the mouth")
        for lvl in MOUTH.values():
            for x, y, _c in lvl:
                if not orb[y, x]:
                    raise SystemExit(f"mouth wedge leaves the orb at {(x, y)}")
        face_x = (eye_pt[0] * 2 + 45.0) / 3
        if not face_x < orb_c[0] - 0.5:
            raise SystemExit(f"front no longer faces left: face {face_x:.1f} orb {orb_c[0]:.1f}")
    else:
        if not eye_pt[0] > orb_c[0] + 4:
            raise SystemExit(f"back eye not on the right rim: {eye_pt[0]:.1f}")
    pivots = {
        "spike": contact_pivot(spike, orb),
        "bolt_l": contact_pivot(bolt_l, orb),
        "bolt_r": contact_pivot(bolt_r, orb),
        "tail": contact_pivot(tail, orb),
    }
    # bolt tips: farthest pixel from the joint
    tips = {}
    for nm in ("bolt_l", "bolt_r"):
        ys, xs = np.nonzero(parts[nm])
        px, py = pivots[nm]
        k = int(np.argmax((xs - px) ** 2 + (ys - py) ** 2))
        tips[nm] = (float(xs[k]), float(ys[k]))
    sys_, sxs = np.nonzero(spike)
    tips["spike"] = (float(sxs[sys_.argmin()]), float(sys_.min()))
    info = {k: int(v.sum()) for k, v in parts.items()}
    info.update(eye_x=round(eye_pt[0], 1), orb_cx=round(orb_c[0], 1),
                pivots={k: [round(v[0], 1), round(v[1], 1)] for k, v in pivots.items()})
    # facing in the unflipped still: front looks left, back looks up-right
    near, far = ("bolt_l", "bolt_r") if kind == "front" else ("bolt_r", "bolt_l")
    return {"parts": parts, "pivots": pivots, "tips": tips, "eye": eye_pt, "orb_c": orb_c,
            "near": near, "far": far}, info


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

    def pt(p):
        return (p[0] + ox, p[1] + oy)

    placed = {k: place(m) for k, m in seg["parts"].items()}
    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr
    # Joint cushions: aura/core colour from the orb at each joint, on their own
    # layer under every part. At rest the part itself covers them.
    under = np.zeros_like(canvas)
    for part, rad in (("bolt_l", 3), ("bolt_r", 3), ("tail", 2), ("spike", 2)):
        zone = placed[part] & dilate(placed["orb"], rad)
        fill = nearest_color(canvas, zone, placed["orb"])
        paste(under, fill)
    layers["under"] = under

    mouth = {}
    if kind == "front":
        for lvl, pts in MOUTH.items():
            mouth[lvl] = [(x + ox, y + oy, c) for x, y, c in pts]
    pv = {k: pt(v) for k, v in seg["pivots"].items()}
    tips = {k: pt(v) for k, v in seg["tips"].items()}
    eye = pt(seg["eye"])
    orb_c = pt(seg["orb_c"])
    face = -1.0 if kind == "front" else 1.0
    mir = 1.0
    if flip:
        def mx(p):
            return (CELL_W - 1 - p[0], p[1])
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        mouth = {k: [(CELL_W - 1 - x, y, c) for x, y, c in v] for k, v in mouth.items()}
        pv = {k: mx(v) for k, v in pv.items()}
        tips = {k: mx(v) for k, v in tips.items()}
        eye, orb_c = mx(eye), mx(orb_c)
        face, mir = -face, -1.0
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    # raising a bolt: the left-side bolt (tip left of its joint) raises
    # clockwise (+), the right-side bolt counter-clockwise (-), on screen.
    sign = {}
    for nm in ("bolt_l", "bolt_r"):
        sign[nm] = 1.0 if tips[nm][0] < pv[nm][0] else -1.0
    return {
        "kind": kind, "layers": layers, "pv": pv, "tips": tips, "center": center,
        "orb_c": orb_c, "eye": eye, "mouth": mouth, "face": face, "mir": mir,
        "rest": canvas, "near": seg["near"], "far": seg["far"], "sign": sign,
        "body_n": int((canvas[..., 3] >= 128).sum()),
    }


# ---------------------------------------------------------------- motion
def wave(f, n, amp, phase=0.0, cycles=1.0):
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def pose(action, f):
    """near/far: degrees a bolt is RAISED (sign applied per rig/side).

    spike / tail: degrees toward the BACK (away from the face) when positive.
    bob: integer dy of the whole body (negative = up). sway: integer dx toward
    the face. knock: integer dx away from the face.
    """
    n = FRAMES[action]
    p = dict(
        near=0.0, far=0.0, spike=0.0, tail=0.0, bob=0, sway=0, knock=0, mouth=0,
        view="own", narrow=1.0, flash=0, rings=-1, swirl=-1, lines=0, bolt=-1,
        burst=0, sparks=0, crackle=0, flare=0,
    )
    if action == "idle":
        p["bob"] = -int(round(wave(f, n, 1.0, phase=-math.pi / 2)))  # 0..2 up
        p["near"] = wave(f, n, 5.0)
        p["far"] = wave(f, n, 5.0, phase=0.6)
        p["spike"] = wave(f, n, 4.0, phase=0.3)
        p["tail"] = wave(f, n, 6.0, cycles=2.0)
        p["crackle"] = 1 if f in (5, 6, 12, 13) else 0
    elif action == "walk":
        p["bob"] = -int(round(wave(f, n, 1.0, phase=-math.pi / 2, cycles=2.0)))
        p["near"] = wave(f, n, 12.0)
        p["far"] = -wave(f, n, 12.0)
        p["spike"] = 4.0 * (1 - math.cos(2 * math.pi * f / n))  # leans back while gliding
        p["tail"] = wave(f, n, 10.0, cycles=2.0)
        p["sway"] = int(round(wave(f, n, 1.0)))
        p["lines"] = 1 if f % 4 in (1, 2) else 0
    elif action == "attack":
        near = [0, 8, 16, 20, 22, 30, 34, 34, 34, 34, 34, 30, 20, 10, 4, 0]
        far = [0, 8, 16, 20, 22, 14, 16, 16, 16, 16, 16, 14, 10, 5, 2, 0]
        mouth = [0, 1, 2, 3, 3, 2, 2, 2, 2, 2, 2, 3, 2, 1, 0, 0]
        spike = [0, 4, -6, 8, -8, -10, -12, -12, -12, -12, -12, -8, -4, -2, 0, 0]
        bob = [0, 0, -1, -2, -2, -3, -3, -2, 0, 1, 1, 0, 0, 0, 0, 0]
        sway = [0, 0, 0, -1, -1, 0, 0, 0, 2, 3, 3, 2, 1, 0, 0, 0]
        tail = [0, -4, -8, -10, -10, 8, 10, 10, 10, 10, 10, 6, 3, 1, 0, 0]
        p.update(near=near[f], far=far[f], mouth=mouth[f], spike=spike[f], bob=bob[f],
                 sway=sway[f], tail=tail[f])
        if 1 <= f <= 4:
            p["rings"] = f - 1
            p["crackle"] = 1
            p["flare"] = 1
        if 5 <= f <= 10:
            p["bolt"] = min(f - 5, 2)
        if f in (11, 12):
            p["burst"] = f - 10
    else:
        env = [0, 0.5, 0.85, 1.0, 0.9, 0.7, 0.48, 0.3, 0.16, 0.06, 0.0, 0][f]
        p["knock"] = int(round(4 * env))
        p["bob"] = -int(round(2 * env))
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
        p["near"] = 24.0 * env
        p["far"] = 24.0 * env
        p["spike"] = 24.0 * env
        p["tail"] = -14.0 * env
        p["mouth"] = 1 if 2 <= f <= 6 else 0
    return p


# ---------------------------------------------------------------- fx
def put(out, x, y, col, over=False):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and (over or out[y, x, 3] < 128):
        out[y, x, :3] = col
        out[y, x, 3] = 255


def roar_rings(out, origin, face, age):
    """Expanding roar shells, wider on the facing side."""
    ox, oy = origin
    cols = (C_CL, C_W, C_T)
    for r_i in range(2):
        rad = 24 + age * 5 + r_i * 6
        col = cols[(age + r_i) % 3]
        for deg in range(-60, 61, 4):
            ang = math.radians(deg)
            for side, k in ((face, 1.0), (-face, 0.75)):
                put(out, ox + side * k * rad * math.cos(ang), oy + 0.7 * rad * math.sin(ang), col)


def zigzag(out, p0, p1, seed, jitter=3.0, cols=(C_W, C_CL, C_C), width=1):
    """Lightning polyline from p0 to p1 with random sideways kinks."""
    rng = np.random.default_rng(seed)
    x0, y0 = p0
    x1, y1 = p1
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1:
        return []
    nx, ny = -(y1 - y0) / dist, (x1 - x0) / dist
    segs = max(3, int(dist // 6))
    pts = [(x0, y0)]
    for i in range(1, segs):
        t = i / segs
        off = float(rng.uniform(-jitter, jitter))
        pts.append((x0 + (x1 - x0) * t + nx * off, y0 + (y1 - y0) * t + ny * off))
    pts.append((x1, y1))
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        steps = int(max(abs(bx - ax), abs(by - ay))) + 1
        for s in range(steps + 1):
            t = s / max(1, steps)
            x, y = ax + (bx - ax) * t, ay + (by - ay) * t
            put(out, x, y, cols[0], over=True)
            if width > 1:
                put(out, x + nx, y + ny, cols[1])
                put(out, x - nx, y - ny, cols[2])
    return pts


def crackle(out, tips, f):
    for i, (tx, ty) in enumerate(tips):
        ang = (f * 1.7 + i * 2.3)
        for k in range(3):
            a2 = ang + k * 2.1
            ex, ey = tx + 6 * math.cos(a2), ty + 5 * math.sin(a2)
            zigzag(out, (tx, ty), (ex, ey), seed=f * 31 + i * 7 + k, jitter=1.5,
                   cols=(C_CL if k else C_W,))


def electric_swirl(out, center, stage):
    cx, cy = center
    for arm in range(3):
        base = arm * 2 * math.pi / 3 + stage * 1.1
        for t in range(16):
            ang = base + t * 0.15
            r = 36 - t * 0.9
            col = C_W if t < 3 else (C_CL if t < 8 else C_T)
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
            put(out, base + side * i, y, C_W if i < 2 else C_C)


def thunder(out, tip, face, stage, seed):
    """Forked lightning shot from the near bolt tip toward the facing side."""
    tx, ty = tip
    reach = 20 + stage * 7
    end = (tx + face * reach, ty - 2 + stage)
    pts = zigzag(out, (tx, ty), end, seed=seed, jitter=3.5, width=2)
    for k, idx in enumerate((len(pts) // 2, len(pts) - 2)):
        if 0 < idx < len(pts):
            bx, by = pts[idx]
            zigzag(out, (bx, by), (bx + face * 9, by + (6 if k else -6)), seed=seed + 5 + k,
                   jitter=2.0, cols=(C_CL,))
    # glow at the bolt tip
    for dx, dy in ((0, -2), (0, 2), (-2, 0), (2, 0), (-1, -1), (1, 1), (1, -1), (-1, 1)):
        put(out, tx + dx, ty + dy, C_W, over=True)
    return end


def burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4 + math.pi / 8
        for r in range(max(2, size // 2), size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang),
                C_W if r < size - 2 else C_CL, over=True)
    for k in range(12):
        ang = k * math.pi / 6
        put(out, cx + (size + 2) * math.cos(ang), cy + (size + 2) * math.sin(ang), C_O)
    for k in range(4):
        ang = k * math.pi / 2 + 0.4
        zigzag(out, (cx, cy), (cx + (size + 6) * math.cos(ang), cy + (size + 6) * math.sin(ang)),
               seed=size * 3 + k, jitter=1.5, cols=(C_C,))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, C_W, over=True)


def hurt_sparks(out, f, back_dir, seed):
    bb = body_box(out)
    if not bb:
        return
    x0, y0, x1, y1 = bb
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rng = np.random.default_rng(seed)
    for k in range(3):
        a = float(rng.uniform(0, 2 * math.pi))
        r0 = 6
        r1 = 16 + f
        zigzag(out, (cx + r0 * math.cos(a), cy + r0 * math.sin(a)),
               (cx + r1 * math.cos(a), cy + r1 * math.sin(a)), seed=seed + k, jitter=2.0,
               cols=(C_W,))
    for sx, sy in ((cx + back_dir * (14 + f), y0 + 4), (cx + back_dir * 20, y0 + 14)):
        put(out, sx, sy, C_W)
        put(out, sx + back_dir, sy + 1, C_CL)


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 60)
    out[a & ~dark, :3] = C_W
    return out


# ---------------------------------------------------------------- synth
def synth_body(rig, p):
    L = rig["layers"]
    pv = rig["pv"]
    m = rig["mir"]
    face = rig["face"]
    sg = rig["sign"]

    def rot(name, ang):
        return rotate_nearest(L[name], ang, pv[name])

    near, far = rig["near"], rig["far"]
    moved = {
        near: rot(near, sg[near] * p["near"]),
        far: rot(far, sg[far] * p["far"]),
        # positive spike/tail = toward the back: clockwise when the face is left
        "spike": rot("spike", -face * p["spike"]),
        "tail": rot("tail", -face * p["tail"]),
    }
    del m
    body = np.zeros_like(rig["rest"])
    paste(body, L["under"])
    paste(body, moved[far])
    paste(body, moved["tail"])
    paste(body, moved[near])
    orb = L["orb"].copy()
    if p["mouth"] and rig["mouth"]:
        for x, y, c in rig["mouth"][int(p["mouth"])]:
            orb[y, x, :3] = c
            orb[y, x, 3] = 255
    paste(body, orb)
    paste(body, moved["spike"])
    moving = any(abs(float(p[k])) >= 0.45 for k in ("near", "far", "tail", "spike"))
    if moving:
        body = despeckle(body)
    body = shift(body, p["sway"] * face - p["knock"] * face, p["bob"])
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
    dx = p["sway"] * face - p["knock"] * face

    def tip_of(name, ang):
        t = content_point(*own["tips"][name], own["sign"][name] * ang, own["pv"][name])
        return (t[0] + dx, t[1] + bob)

    tips = [tip_of(own["near"], p["near"]), tip_of(own["far"], p["far"])]
    if p["crackle"]:
        crackle(img, tips, f)
    if p["flare"]:
        sx, sy = own["tips"]["spike"]
        crackle(img, [(sx + dx, sy + bob)], f + 3)
    if p["rings"] >= 0:
        oc = own["orb_c"]
        roar_rings(img, (oc[0] + face * 3, oc[1] + bob + 3), face, p["rings"])
    if p["swirl"] >= 0:
        electric_swirl(img, (own["center"][0], own["center"][1] + bob), p["swirl"])
    if p["lines"]:
        speed_lines(img, face, seed=f * 13 + 5)
    if p["bolt"] >= 0:
        thunder(img, tips[0], face, p["bolt"], seed=40 + f)
    if p["burst"]:
        t0 = tips[0]
        reach = 20 + 2 * 7
        burst(img, (t0[0] + face * (reach - 4 + 2 * p["burst"]), t0[1] - 1), 6 + 3 * p["burst"])
    if p["sparks"]:
        hurt_sparks(img, f, -face, seed=90 + f)
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
    """All 8 directions: idle 0, walk 4, attack 3/5/6/9/11, hurt 3."""
    picks = [("idle", 0), ("walk", 4), ("attack", 3), ("attack", 5), ("attack", 6),
             ("attack", 9), ("attack", 11), ("hurt", 3)]
    crop = (CELL_W // 2 - 80, CELL_H - 120, CELL_W // 2 + 80, CELL_H - 8)
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
    crop = (CELL_W // 2 - 80, CELL_H - 116, CELL_W // 2 + 80, CELL_H - 12)
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
        "dex": 479, "form": "normal", "variant": "rotom", "gen": "gen4",
        "dir": "monsterSprite/gen4/0479_rotom/normal", "sheetDir": "agent-sprite-forge",
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


def update_queue(status):
    def edit():
        raw = QUEUE.read_text(encoding="utf-8")
        i = raw.find('"id": "' + KEY + '",')
        if i < 0:
            raise SystemExit("queue item missing")
        j = raw.find('"status": "', i)
        k = raw.find('"', j + len('"status": "'))
        out = raw[:j] + '"status": "' + status + raw[k:]
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
    fseg, finfo = segment(front, "front")
    bseg, binfo = segment(back, "back")
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
    if not (rigs["E"]["eye"][0] > rigs["E"]["orb_c"][0] - 0.01 and rigs["E"]["face"] > 0
            and rigs["W"]["eye"][0] < rigs["W"]["orb_c"][0] and rigs["W"]["face"] < 0):
        raise SystemExit("E must face screen-right and W screen-left")
    print("still diff 0 for all dirs; E faces right, W faces left", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    pcols = {"orb": (250, 250, 250), "spike": (60, 220, 220), "bolt_l": (40, 90, 220),
             "bolt_r": (40, 180, 80), "tail": (240, 120, 20)}
    for nm, seg in (("front", fseg), ("back", bseg)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, msk in seg["parts"].items():
            part[msk, :3] = pcols[name]
            part[msk, 3] = 255
        for name, (px, py) in seg["pivots"].items():
            part[int(round(py)), int(round(px))] = (255, 0, 0, 255)
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
        crop = (CELL_W // 2 - 80, CELL_H - 116, CELL_W // 2 + 80, CELL_H - 12)
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
        "generator": "forge_rotom_actions", "version": VERSION, "key": KEY, "dex": 479,
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(sprite_h),
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "views": {d: {"still": VIEW[d][0], "mirrored": VIEW[d][1]} for d in DIRS},
        "still_faces": "front.png faces screen-left; back.png faces up-right",
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "source": "front.png + back.png 96x96", "gifs_untouched": True,
        "roar": "existing smirk opens as a wedge on the orb + roar rings, bolts and spike flare",
        "attack": "turn-around via the real back view (no in-plane spin) + thunder strike from the near bolt",
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
