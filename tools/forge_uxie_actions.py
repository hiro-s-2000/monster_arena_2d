#!/usr/bin/env python3
"""Uxie (0480) normal 8-direction action sheet from this form's front.png / back.png.

Measured on normal/front.png: 96x96, 891 opaque, bbox x36-75 y29-66, 13
colours, alpha 0/255, one 4-connected piece.

Body plan (from the pixels): legless levitating psychic pixie. No legs, no
wings, and no mouth pixels (the face is a pale-blue patch with one closed-eye
line; nothing to open).
  head   big yellow helmet x36-63 y29-50 with the red forehead gem
         (x43-47 y35-40) and the pale face patch on its right (x49-59 y38-48)
  body   small grey torso under the head x40-57 y48-58 (near hand hangs at
         x52-57)
  arm    far arm reaching screen-left, x36-42 y51-56
  tail   the two thin tails: they loop down from the torso (y59-66) and curl
         up to the screen-right, where both three-lobed tips carry a red gem
         (x58-75 y46-63)
The face patch sits right of the helmet centre, so front.png faces screen-right.

back.png (937 opaque, same palette) is the back view facing up-right: plain
helmet, torso x47-62 with the near arm reaching right (x62-68 y53-57), and
the tails looping down-left to the two gem tips at x28-46.

Directions: front still S SE E unflipped, W SW mirrored. back still NE N
unflipped, NW mirrored. E faces screen-right, W screen-left.

  idle   : hover bob (integer), the tails sway, the arm floats, a twinkle on
           the tail gems. Frame 0 and the loop end are the still.
  walk   : glide (no legs): hover bob x2, the tails trail and ripple, the arm
           rows, psychic speed lines. Same loop rule.
  attack : 0-4 roar: no mouth, so the forehead gem flares white, the tails and
           arm rise and psychic rings burst from the gem (back rows: from the
           helmet top). 5-7 turn-around: the body turns (front -> side ->
           back -> side via the real back view, never an in-plane rotation)
           inside a psychic swirl. 8-10 strike: the tails lash forward and
           down (<= 34 deg), the arm thrusts, and a psychic wave fires from
           the gem. 11-12 impact burst. 13-15 recover.
  hurt   : 12 frames, white flash, knock back + lift (integer), tails and arm
           jolt up <= 24 deg, psychic sparks, recover.
The head (helmet + face) only ever moves by integer pixels.

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
SRC_DIR = ROOT / "monsterSprite/gen4/0480_uxie/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0480_uxie/normal/agent-sprite-forge"
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
    "S": ("front", False), "SE": ("front", False), "E": ("front", False),
    "NE": ("back", False), "N": ("back", False), "NW": ("back", True),
    "W": ("front", True), "SW": ("front", True),
}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "uxie_v2"
KEY = "0480_uxie"

C_K = (16, 16, 16)
C_D = (65, 65, 65)
C_O = (172, 139, 65)
C_N = (115, 90, 65)
C_Y = (246, 197, 115)
C_YL = (255, 222, 156)
C_B = (180, 205, 246)
C_M = (148, 156, 197)
C_P = (123, 115, 148)
C_R = (238, 65, 65)
C_RD = (172, 65, 65)
C_Q = (90, 57, 65)
C_W = (255, 255, 255)
REDS = (C_R, C_RD, C_Q, C_W)

FRONT_N = 891
FRONT_BBOX = (36, 75, 29, 66)
BACK_N = 937
BACK_BBOX = (28, 68, 28, 67)


# ---------------------------------------------------------------- helpers
def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def labels(mask):
    """4-connected components as a list of boolean masks, largest first."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        comp = np.zeros_like(mask)
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        while q:
            y, x = q.popleft()
            comp[y, x] = True
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        out.append(comp)
    out.sort(key=lambda m: -int(m.sum()))
    return out


def component_sizes(mask):
    return [int(m.sum()) for m in labels(mask)]


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


def absorb_strays(parts, order):
    """Keep each part's main piece; hand stray bits to the part they touch most."""
    for name in order:
        comps = labels(parts[name])
        for stray in comps[1:]:
            ring = dilate(stray, 1) & ~stray
            best, score = None, -1
            for other, m in parts.items():
                if other == name:
                    continue
                s = int((ring & m).sum())
                if s > score:
                    best, score = other, s
            parts[name] = parts[name] & ~stray
            parts[best] = parts[best] | stray
    return parts


def segment(img, kind):
    if kind == "front":
        a = check_still(img, FRONT_N, FRONT_BBOX, "front")
    else:
        a = check_still(img, BACK_N, BACK_BBOX, "back")
    yy, xx = np.mgrid[0:96, 0:96]
    rim = is_col(img, a, (C_K, C_O, C_N))
    if kind == "front":
        head = (a & (yy <= 45)) | (a & (yy <= 47) & (xx <= 62)) \
            | (rim & (yy >= 48) & (yy <= 50) & (xx >= 40) & (xx <= 60))
        tail = a & ~head & ((xx >= 58) | (yy >= 59))
        body = a & ~head & ~tail
        arm = body & (xx <= 42) & (yy <= 56)
    else:
        head = (a & (yy <= 48) & (xx >= 40)) | (rim & (yy >= 49) & (yy <= 51) & (xx >= 46))
        tail = a & ~head & ((xx <= 46) | (yy >= 60))
        body = a & ~head & ~tail
        arm = body & (xx >= 62)
    body = body & ~arm
    parts = {"head": head, "body": body, "arm": arm, "tail": tail}
    parts = absorb_strays(parts, ("tail", "body", "arm", "head"))
    total = sum(int(m.sum()) for m in parts.values())
    n_exp = FRONT_N if kind == "front" else BACK_N
    if total != n_exp:
        raise SystemExit(f"{kind} parts overlap or miss pixels: {total}")
    for name, m in parts.items():
        sz = component_sizes(m)
        if len(sz) != 1:
            raise SystemExit(f"{kind} {name} is not one piece: {sz[:5]}")
    face_m = is_col(img, a, (C_B, C_M)) & parts["head"]
    hys, hxs = np.nonzero(parts["head"])
    head_c = (float(hxs.mean()), float(hys.mean()))
    gem_t = is_col(img, a, (C_R, C_RD)) & parts["tail"]
    if int(gem_t.sum()) < 4:
        raise SystemExit(f"{kind} tail gems not found")
    gys, gxs = np.nonzero(gem_t)
    tail_tip = (float(gxs.mean()), float(gys.mean()))
    if kind == "front":
        fys, fxs = np.nonzero(face_m)
        face_x = float(fxs.mean())
        if not face_x > head_c[0] + 2:
            raise SystemExit(f"front no longer faces right: face {face_x:.1f} head {head_c[0]:.1f}")
        gem_h = is_col(img, a, REDS) & parts["head"]
        eys, exs = np.nonzero(gem_h)
        gem = (float(exs.mean()), float(eys.mean()))
        gem_px = [(int(x), int(y)) for x, y in zip(exs.tolist(), eys.tolist())]
        if not (tail_tip[0] > head_c[0]):
            raise SystemExit("front tail tips moved off the right")
    else:
        if int(face_m.sum()):
            raise SystemExit("back helmet shows face pixels; remeasure")
        gem = (head_c[0] + 3, float(hys.min()) + 3)
        gem_px = []
        if not (tail_tip[0] < head_c[0]):
            raise SystemExit("back tail tips moved off the left")
    pivots = {
        "arm": contact_pivot(parts["arm"], parts["body"]),
        "tail": contact_pivot(parts["tail"], parts["body"]),
    }
    ays, axs = np.nonzero(parts["arm"])
    px, py = pivots["arm"]
    k = int(np.argmax((axs - px) ** 2 + (ays - py) ** 2))
    tips = {"arm": (float(axs[k]), float(ays[k])), "tail": tail_tip}
    info = {k: int(v.sum()) for k, v in parts.items()}
    info.update(head_cx=round(head_c[0], 1), gem=[round(gem[0], 1), round(gem[1], 1)],
                tail_tip=[round(tail_tip[0], 1), round(tail_tip[1], 1)],
                pivots={k: [round(v[0], 1), round(v[1], 1)] for k, v in pivots.items()})
    return {"parts": parts, "pivots": pivots, "tips": tips, "gem": gem, "gem_px": gem_px,
            "head_c": head_c}, info


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
    # Joint cushions: torso colour where the arm and the tail roots sit, on
    # their own layer under every part. At rest the part covers them.
    under = np.zeros_like(canvas)
    for part, rad in (("arm", 2), ("tail", 3)):
        zone = placed[part] & dilate(placed["body"], rad)
        fill = nearest_color(canvas, zone, placed["body"])
        paste(under, fill)
    layers["under"] = under

    gem_px = [(x + ox, y + oy) for x, y in seg["gem_px"]]
    pv = {k: pt(v) for k, v in seg["pivots"].items()}
    tips = {k: pt(v) for k, v in seg["tips"].items()}
    gem = pt(seg["gem"])
    head_c = pt(seg["head_c"])
    face = 1.0
    if flip:
        def mx(p):
            return (CELL_W - 1 - p[0], p[1])
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        gem_px = [(CELL_W - 1 - x, y) for x, y in gem_px]
        pv = {k: mx(v) for k, v in pv.items()}
        tips = {k: mx(v) for k, v in tips.items()}
        gem, head_c = mx(gem), mx(head_c)
        face = -face
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    # raising a part: tip left of its joint raises clockwise (+), tip right
    # of its joint counter-clockwise (-), on screen.
    sign = {nm: (1.0 if tips[nm][0] < pv[nm][0] else -1.0) for nm in ("arm", "tail")}
    # paint order: front view has the tails behind the torso; back view in front
    order = ["under", "tail", "body", "arm", "head"] if kind == "front" else \
        ["under", "body", "arm", "tail", "head"]
    return {
        "kind": kind, "layers": layers, "pv": pv, "tips": tips, "center": center,
        "gem": gem, "gem_px": gem_px, "head_c": head_c, "face": face,
        "rest": canvas, "sign": sign, "order": order,
        "body_n": int((canvas[..., 3] >= 128).sum()),
    }


# ---------------------------------------------------------------- motion
def wave(f, n, amp, phase=0.0, cycles=1.0):
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def pose(action, f):
    """arm / tail: degrees the part is RAISED (sign applied per rig).

    bob: integer dy of the whole body (negative = up). sway: integer dx
    toward the face. knock: integer dx away from the face.
    """
    n = FRAMES[action]
    p = dict(
        arm=0.0, tail=0.0, bob=0, sway=0, knock=0, gem=0,
        view="own", narrow=1.0, flash=0, rings=-1, swirl=-1, lines=0, beam=-1,
        burst=0, sparks=0, twinkle=0,
    )
    if action == "idle":
        p["bob"] = -int(round(wave(f, n, 1.0, phase=-math.pi / 2)))  # 0..2 up
        p["tail"] = wave(f, n, 6.0, phase=0.4)
        p["arm"] = wave(f, n, 6.0, phase=1.2)
        p["twinkle"] = 1 if f in (5, 6, 12, 13) else 0
    elif action == "walk":
        p["bob"] = -int(round(wave(f, n, 1.0, phase=-math.pi / 2, cycles=2.0)))
        # the tails trail low behind the glide and ripple twice per loop
        p["tail"] = -5.0 * (1 - math.cos(2 * math.pi * f / n)) + wave(f, n, 5.0, cycles=2.0)
        p["arm"] = wave(f, n, 14.0)
        p["sway"] = int(round(wave(f, n, 1.0)))
        p["lines"] = 1 if f % 4 in (1, 2) else 0
    elif action == "attack":
        arm = [0, 8, 16, 22, 24, -26, -32, -32, -32, -32, -32, -26, -16, -8, -3, 0]
        tail = [0, 8, 16, 22, 24, -22, -28, -28, -28, -28, -28, -22, -14, -7, -3, 0]
        gem = [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0]
        bob = [0, 0, -1, -2, -2, -3, -3, -2, 0, 1, 1, 0, 0, 0, 0, 0]
        sway = [0, 0, 0, -1, -1, 0, 0, 0, 2, 3, 3, 2, 1, 0, 0, 0]
        p.update(arm=arm[f], tail=tail[f], gem=gem[f], bob=bob[f], sway=sway[f])
        if 1 <= f <= 4:
            p["rings"] = f - 1
            p["twinkle"] = 1
        if 5 <= f <= 10:
            p["beam"] = min(f - 5, 2)
        if f in (11, 12):
            p["burst"] = f - 10
    else:
        env = [0, 0.5, 0.85, 1.0, 0.9, 0.7, 0.48, 0.3, 0.16, 0.06, 0.0, 0][f]
        p["knock"] = int(round(4 * env))
        p["bob"] = -int(round(2 * env))
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
        p["arm"] = 24.0 * env
        p["tail"] = 20.0 * env
        p["gem"] = 1 if 2 <= f <= 4 else 0
    return p


# ---------------------------------------------------------------- fx
def put(out, x, y, col, over=False):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and (over or out[y, x, 3] < 128):
        out[y, x, :3] = col
        out[y, x, 3] = 255


def star(out, x, y, size, core=C_W, arm=C_B, over=True):
    put(out, x, y, core, over)
    for k in range(1, size + 1):
        c = core if k == 1 else arm
        for dx, dy in ((k, 0), (-k, 0), (0, k), (0, -k)):
            put(out, x + dx, y + dy, c, over)


def psychic_rings(out, origin, face, age):
    """Expanding psychic shells from the gem, wider on the facing side."""
    ox, oy = origin
    cols = (C_W, C_B, C_YL)
    for r_i in range(2):
        rad = 14 + age * 6 + r_i * 6
        col = cols[(age + r_i) % 3]
        for deg in range(0, 360, 5):
            ang = math.radians(deg)
            cx = math.cos(ang)
            k = 1.0 if cx * face >= 0 else 0.7
            put(out, ox + k * rad * cx, oy + 0.62 * rad * math.sin(ang), col)


def psychic_swirl(out, center, stage):
    cx, cy = center
    for arm in range(3):
        base = arm * 2 * math.pi / 3 + stage * 1.1
        for t in range(18):
            ang = base + t * 0.16
            r = 34 - t * 0.9
            col = C_W if t < 3 else (C_YL if t < 9 else C_B)
            put(out, cx + r * math.cos(ang), cy + 0.5 * r * math.sin(ang), col)


def speed_lines(out, face, seed):
    bb = body_box(out)
    if not bb:
        return
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    base = x0 - 3 if face > 0 else x1 + 3
    side = -1 if face > 0 else 1
    for _ in range(3):
        y = int(rng.integers(y0 + (y1 - y0) // 4, y1 - 4))
        leng = int(rng.integers(4, 9))
        for i in range(leng):
            put(out, base + side * i, y, C_W if i < 2 else C_B)


def psychic_wave(out, origin, face, stage):
    """A psychic beam from the gem with crescent ripples riding on it."""
    ox, oy = origin
    reach = 22 + stage * 9
    for i in range(0, reach):
        x = ox + face * i
        put(out, x, oy, C_W)
        put(out, x, oy - 1, C_W if i > 2 else C_B)
        put(out, x, oy + 1, C_W if i > 2 else C_B)
        if i % 3 != 1:
            put(out, x, oy - 2, C_B)
            put(out, x, oy + 2, C_B)
    for j in range(stage + 1):
        cx = ox + face * (8 + j * 9 + stage * 3)
        rad = 5 + j * 2
        for deg in range(-70, 71, 6):
            ang = math.radians(deg)
            put(out, cx + face * rad * math.cos(ang) * 0.6, oy + rad * math.sin(ang),
                C_YL if j % 2 == 0 else C_W)
    star(out, ox, oy, 2, over=False)
    return (ox + face * reach, oy)


def burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4 + math.pi / 8
        for r in range(max(2, size // 2), size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang),
                C_W if r < size - 2 else C_YL, over=True)
    for k in range(16):
        ang = k * math.pi / 8
        put(out, cx + (size + 2) * math.cos(ang), cy + 0.8 * (size + 2) * math.sin(ang), C_B)
    for k in range(4):
        ang = k * math.pi / 2 + 0.3
        put(out, cx + (size + 5) * math.cos(ang), cy + (size + 5) * math.sin(ang), C_R)
    star(out, cx, cy, 2)


def hurt_sparks(out, f, back_dir, seed):
    bb = body_box(out)
    if not bb:
        return
    x0, y0, x1, y1 = bb
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rng = np.random.default_rng(seed)
    for k in range(4):
        a = float(rng.uniform(0, 2 * math.pi))
        r = 18 + f * 1.5
        star(out, cx + r * math.cos(a), cy + 0.8 * r * math.sin(a), 1 + (k % 2), over=False)
    for sx, sy in ((cx + back_dir * (16 + f), y0 + 4), (cx + back_dir * 22, y0 + 14)):
        put(out, sx, sy, C_W)
        put(out, sx + back_dir, sy + 1, C_B)


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
    sg = rig["sign"]
    moved = {
        "arm": rotate_nearest(L["arm"], sg["arm"] * p["arm"], pv["arm"]),
        "tail": rotate_nearest(L["tail"], sg["tail"] * p["tail"], pv["tail"]),
        "under": L["under"], "body": L["body"],
    }
    head = L["head"].copy()
    if p["gem"] and rig["gem_px"]:
        # the forehead gem flares: red core goes white (palette colours only)
        for x, y in rig["gem_px"]:
            c = tuple(int(v) for v in head[y, x, :3])
            if p["gem"] >= 2 and c in (C_R, C_RD):
                head[y, x, :3] = C_W
            elif p["gem"] == 1 and c == C_RD:
                head[y, x, :3] = C_R
    moved["head"] = head
    body = np.zeros_like(rig["rest"])
    for name in rig["order"]:
        paste(body, moved[name])
    moving = any(abs(float(p[k])) >= 0.45 for k in ("arm", "tail"))
    if moving:
        body = despeckle(body)
    body = shift(body, p["sway"] * rig["face"] - p["knock"] * rig["face"], p["bob"])
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

    gem = (own["gem"][0] + dx, own["gem"][1] + bob)
    hb = body_box(own["layers"]["head"])
    front_x = (hb[2] + 2) if face > 0 else (hb[0] - 2)
    muzzle = (front_x + dx, own["gem"][1] + bob + 3)
    if p["twinkle"]:
        tx, ty = tip_of("tail", p["tail"])
        star(img, tx + face * 2, ty - 3, 2 if f % 2 else 1, over=False)
    if p["rings"] >= 0:
        psychic_rings(img, gem, face, p["rings"])
        star(img, gem[0], gem[1], 2 + (p["rings"] % 2))
    if p["swirl"] >= 0:
        psychic_swirl(img, (own["center"][0], own["center"][1] + bob), p["swirl"])
    if p["lines"]:
        speed_lines(img, face, seed=f * 13 + 5)
    if p["beam"] >= 0:
        psychic_wave(img, muzzle, face, p["beam"])
    if p["burst"]:
        reach = 22 + 2 * 9
        burst(img, (muzzle[0] + face * (reach - 4 + 2 * p["burst"]), muzzle[1]), 6 + 3 * p["burst"])
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
        "dex": 480, "form": "normal", "variant": "uxie", "gen": "gen4",
        "dir": "monsterSprite/gen4/0480_uxie/normal", "sheetDir": "agent-sprite-forge",
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
    fx = {C_W, C_B, C_YL, C_R}
    if not fx <= fp:
        raise SystemExit("fx colours are not in the front palette")
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
    if not (rigs["E"]["face"] > 0 and rigs["W"]["face"] < 0
            and rigs["E"]["gem"][0] < rigs["E"]["tips"]["tail"][0]
            and rigs["W"]["gem"][0] > rigs["W"]["tips"]["tail"][0]):
        raise SystemExit("E must face screen-right and W screen-left")
    print("still diff 0 for all dirs; E faces right, W faces left", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    pcols = {"head": (250, 250, 250), "body": (60, 200, 60), "arm": (40, 90, 220),
             "tail": (240, 120, 20)}
    for nm, seg in (("front", fseg), ("back", bseg)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, msk in seg["parts"].items():
            part[msk, :3] = pcols[name]
            part[msk, 3] = 255
        for name, (px, py) in seg["pivots"].items():
            part[int(round(py)), int(round(px))] = (255, 0, 0, 255)
        Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / f"parts_{nm}.png")

    if args.samples:
        want = [("idle", "E", f) for f in (0, 4, 8, 12)]
        want += [("walk", "E", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "E", f) for f in range(16)]
        want += [("attack", "N", f) for f in range(16)]
        want += [("hurt", "E", f) for f in (0, 1, 3, 5, 8, 11)]
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
    motion_contact(sheets, "E").save(OUT_DIR / "preview_motion.png")
    motion_contact(sheets, "N").save(work / "review_N.png")
    motion_contact(sheets, "W").save(work / "review_W.png")

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
            od = qa / f"gait_{act}_E"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "E"),
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
        "generator": "forge_uxie_actions", "version": VERSION, "key": KEY, "dex": 480,
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(sprite_h),
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "views": {d: {"still": VIEW[d][0], "mirrored": VIEW[d][1]} for d in DIRS},
        "still_faces": "front.png faces screen-right; back.png faces up-right",
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "source": "front.png + back.png 96x96", "gifs_untouched": True,
        "roar": "no mouth: the forehead gem flares white, tails and arm rise, psychic rings",
        "attack": "turn-around via the real back view (no in-plane spin) + tail lash and psychic wave from the gem",
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
