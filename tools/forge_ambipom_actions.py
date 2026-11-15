#!/usr/bin/env python3
"""Ambipom (0424) 8-direction action sheet from the 96x96 front still.

Measured on front.png: opaque 2043, bbox x16-78 y13-82 (63x70), 14 colors,
alpha already 0 or 255. Camera-facing biped. Both eyes are white and the tooth
band sits under them, so the still is not a side profile. Two tails end in
hands (cream palms, orange finger pads). Two purple feet. Five orange pixels
beside the muzzle are the nose, not a third hand. No third tail.

Attack: frames 0-4 roar, 5-10 a tail-hand slap inside the 7 degree clamp, 11-12 impact, 13-15 recover. Hurt lean peaks at 38
degrees. Idle frame 0 is the still. Feet stay on that sole line during idle.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
STILL_DIR = ROOT / "monsterSprite/gen4/0424_ambipom/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0424_ambipom/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

KEY = "0424_ambipom"
VERSION = "ambipom_v2"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces the camera. South matches the pixels. The far hemisphere is the mirror.
UNFLIPPED = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 192, 192, 172
HURT_DEG = [0, 16, 28, 36, 38, 34, 24, 14, 6, 2, 0, 0]
HURT_DX = [0, 1, 2, 3, 3, 2, 2, 1, 0, 0, 0, 0]
HURT_DY = [0, -1, -2, -2, -2, -1, -1, 0, 0, 0, 0, 0]

INK = (16, 16, 16)
ORANGE = (255, 115, 90)
WHITE = (255, 255, 255)
PURPLE = (131, 57, 172)

PART_DRAW = ("tail_l", "tail_r", "foot_l", "foot_r", "torso", "hand_l", "hand_r")
MOVING = ("tail_l", "tail_r", "foot_l", "foot_r", "hand_l", "hand_r")


def extract_palette(img):
    cols = img[img[..., 3] >= 128][:, :3]
    uniq = np.unique(cols, axis=0)
    return [tuple(int(v) for v in c) for c in uniq]


def snap(img, palette):
    pal = np.asarray(palette, np.int32)
    out = np.zeros_like(img)
    opaque = img[..., 3] >= 128
    if not np.any(opaque):
        return out
    pix = img[..., :3].astype(np.int32)
    flat = pix.reshape(-1, 3)
    idx = np.flatnonzero(opaque.reshape(-1))
    diff = flat[idx, None, :].astype(np.int32) - pal[None, :, :]
    dist = np.sum(diff * diff, axis=2)
    pick = pal[dist.argmin(axis=1)]
    dst = out.reshape(-1, 4)
    dst[idx, :3] = pick
    dst[idx, 3] = 255
    return out


def segment(img):
    """Two tail-hands, two tails, two feet, and the torso that holds the face."""
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    rules = [
        ("hand_l", a & (yy <= 36) & (xx <= 42) & ~((xx <= 24) & (yy >= 31))),
        ("hand_r", a & (xx >= 52) & (yy >= 22) & (yy <= 43)),
        ("tail_l", a & (xx <= 25) & (yy >= 28) & (yy <= 67)),
        ("tail_r", a & (xx >= 62) & (yy >= 40) & (yy <= 71)),
        ("foot_l", a & (yy >= 72) & (xx <= 42)),
        ("foot_r", a & (yy >= 72) & (xx >= 43)),
    ]
    owner = np.full(a.shape, -1, np.int16)
    names = []
    for i, (name, mask) in enumerate(rules):
        names.append(name)
        owner[(owner < 0) & mask] = i
    names.append("torso")
    owner[(owner < 0) & a] = len(names) - 1
    masks = {name: owner == i for i, name in enumerate(names)}
    chin = masks["torso"] & (yy >= 58) & (yy <= 64) & (xx >= 36) & (xx <= 46)
    masks["chin"] = chin
    counts = {name: int(masks[name].sum()) for name in names}
    counts["chin"] = int(chin.sum())
    if int(sum(counts[n] for n in names)) != int(a.sum()):
        raise SystemExit(f"segmentation dropped pixels: {counts}")
    if counts["chin"] < 40:
        raise SystemExit("mouth chin is missing from the torso")
    orange = a & np.all(img[..., :3] == ORANGE, axis=2)
    hands = masks["hand_l"] | masks["hand_r"]
    if int((orange & masks["hand_l"]).sum()) < 40 or int((orange & masks["hand_r"]).sum()) < 40:
        raise SystemExit("each tail-hand needs its finger pads")
    if int((orange & ~hands).sum()) > 12:
        raise SystemExit("orange outside the two hands looks like a third hand")
    return masks, counts


def socket_pivot(part, body):
    pad = np.pad(body, 1)
    near = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    ys, xs = np.nonzero(part & near)
    if len(ys) == 0:
        ys, xs = np.nonzero(part)
    by, bx = np.nonzero(body)
    cy, cx = float(by.mean()), float(bx.mean())
    d = (ys.astype(np.float64) - cy) ** 2 + (xs.astype(np.float64) - cx) ** 2
    i = int(d.argmin())
    return float(xs[i]), float(ys[i])


def shift(layer, dx, dy):
    dx_i, dy_i = int(dx), int(dy)
    if dx_i == 0 and dy_i == 0:
        return layer.copy()
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0, sx1 = max(0, -dx_i), min(w, w - dx_i)
    sy0, sy1 = max(0, -dy_i), min(h, h - dy_i)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy_i:sy1 + dy_i, sx0 + dx_i:sx1 + dx_i] = layer[sy0:sy1, sx0:sx1]
    return out


_GRID = {}


def _mgrid(h, w):
    key = (h, w)
    if key not in _GRID:
        _GRID[key] = np.mgrid[0:h, 0:w]
    return _GRID[key]


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest in-plane rotate. Angle 0 and 360 return the same pixels.

    Inverse sample, then forward-fill any source pixel that landed on empty
    so a thin tail is not dropped. No |cos| squash.
    """
    angle = float(angle_deg) % 360.0
    if angle > 180.0:
        angle -= 360.0
    if abs(angle) < 0.25 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    h, w = layer.shape[:2]
    up = np.repeat(np.repeat(layer, 4, axis=0), 4, axis=1)
    px, py = pivot[0] * 4 + 1.5, pivot[1] * 4 + 1.5
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    uh, uw = up.shape[:2]
    yy, xx = _mgrid(uh, uw)
    rx, ry = xx - px, yy - py
    src_x = np.rint(px + c * rx + s * ry).astype(np.int32)
    src_y = np.rint(py - s * rx + c * ry).astype(np.int32)
    valid = (src_x >= 0) & (src_x < uw) & (src_y >= 0) & (src_y < uh)
    spun = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    spun[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    out = spun[1::4, 1::4].copy()
    src = layer[..., 3] >= 128
    sy, sx = np.nonzero(src)
    ox = sx.astype(np.float64) - pivot[0]
    oy = sy.astype(np.float64) - pivot[1]
    dx = np.rint(c * ox - s * oy + pivot[0]).astype(np.int32)
    dy = np.rint(s * ox + c * oy + pivot[1]).astype(np.int32)
    ok = (dx >= 0) & (dx < w) & (dy >= 0) & (dy < h)
    sx, sy, dx, dy = sx[ok], sy[ok], dx[ok], dy[ok]
    empty = out[dy, dx, 3] < 128
    out[dy[empty], dx[empty]] = layer[sy[empty], sx[empty]]
    return out


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = (
        p[:-2, 1:-1].astype(np.int16)
        + p[2:, 1:-1]
        + p[1:-1, :-2]
        + p[1:-1, 2:]
    )
    out[a & (n == 0)] = 0
    return out


def component_sizes(img):
    m = img[..., 3] >= 128
    vis = np.zeros_like(m)
    h, w = m.shape
    sizes = []
    ys, xs = np.nonzero(m)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if vis[y, x]:
            continue
        stack = [(y, x)]
        vis[y, x] = True
        n = 0
        while stack:
            cy, cx = stack.pop()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not vis[ny, nx]:
                    vis[ny, nx] = True
                    stack.append((ny, nx))
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes


def wave(f, n, amp, phase=0.0, cycles=1.0):
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def place_still(front):
    a = front[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    placed = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    placed[dst_y:dst_y + sh, dst_x:dst_x + sw] = front[y0:y1, x0:x1]
    origin = (dst_x - x0, dst_y - y0)
    return placed, origin, (int(xs.mean()), int(ys.mean()))


def blit_mask(mask, src, origin):
    layer = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return layer
    ox, oy = origin
    layer[ys + oy, xs + ox] = src[ys, xs]
    return layer


def flip_layer(layer):
    return np.ascontiguousarray(layer[:, ::-1])


def flip_xy(pt):
    return (CELL_W - 1 - pt[0], pt[1])


def build_rig(front, masks, origin, flip):
    layers = {name: blit_mask(masks[name], front, origin) for name in list(PART_DRAW) + ["chin"]}
    torso_m = masks["torso"]
    body_m = torso_m | masks["chin"]
    pivots = {
        "hand_l": socket_pivot(masks["hand_l"], masks["tail_l"] | body_m),
        "hand_r": socket_pivot(masks["hand_r"], masks["tail_r"] | body_m),
        "tail_l": socket_pivot(masks["tail_l"], body_m),
        "tail_r": socket_pivot(masks["tail_r"], body_m),
        "foot_l": socket_pivot(masks["foot_l"], torso_m),
        "foot_r": socket_pivot(masks["foot_r"], torso_m),
    }
    pivots = {k: (v[0] + origin[0], v[1] + origin[1]) for k, v in pivots.items()}
    neutral = blit_mask(front[..., 3] >= 128, front, origin)
    cy, cx = np.nonzero(neutral[..., 3] >= 128)
    center = (float(cx.mean()), float(cy.mean()))
    soles = {}
    for name in ("foot_l", "foot_r"):
        ys = np.nonzero(layers[name][..., 3] >= 128)[0]
        soles[name] = int(ys.max()) if len(ys) else FOOT_Y - 1
    mouth_m = masks["chin"]
    my, mx = np.nonzero(mouth_m)
    mouth = (float(mx.mean()) + origin[0], float(my.min()) + origin[1])
    if flip:
        layers = {k: flip_layer(v) for k, v in layers.items()}
        neutral = flip_layer(neutral)
        pivots = {k: flip_xy(v) for k, v in pivots.items()}
        center = flip_xy(center)
        mouth = flip_xy(mouth)
    tips = {}
    for name in ("hand_l", "hand_r"):
        ys, xs = np.nonzero(layers[name][..., 3] >= 128)
        if len(ys) == 0:
            tips[name] = mouth
            continue
        i = int(ys.argmin())
        tips[name] = (int(xs[i]), int(ys[i]))
    return {
        "layers": layers,
        "pivots": pivots,
        "neutral": neutral,
        "center": center,
        "soles": soles,
        "mouth": mouth,
        "tips": tips,
    }


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def clamp_sole(layer, sole_y):
    ys = np.nonzero(layer[..., 3] >= 128)[0]
    if len(ys) == 0:
        return layer
    over = int(ys.max()) - sole_y
    if over > 0:
        return shift(layer, 0, -over)
    return layer


def cushion(img, pivot, color):
    x, y = int(round(pivot[0])), int(round(pivot[1]))
    h, w = img.shape[:2]
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w) or img[ny, nx, 3] >= 128:
                continue
            neigh = 0
            for oy, ox in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = ny + oy, nx + ox
                if 0 <= yy < h and 0 <= xx < w and img[yy, xx, 3] >= 128:
                    neigh += 1
            if neigh >= 3:
                img[ny, nx, :3] = color
                img[ny, nx, 3] = 255


def clamp_tails(tl, tr):
    # Past about 7 degrees a finger bridge on one tail-hand splits, and the
    # mirror facing splits on the opposite sign. Keep both swings inside that.
    return max(-7.0, min(7.0, tl)), max(-7.0, min(7.0, tr))


def pose_idle(f):
    # Each tail and its hand share one angle. Wrist-only rotation tore a finger off.
    tl, tr = clamp_tails(wave(f, 16, 5.5, 0.3), wave(f, 16, 5.5, 0.3 + math.pi))
    return {
        "tail_l": tl,
        "tail_r": tr,
        "foot_l": (0, 0, 0.0),
        "foot_r": (0, 0, 0.0),
    }


def pose_walk(f):
    ang = 2 * math.pi * f / 16.0
    s = math.sin(ang)
    sl = max(0.0, s)
    sr = max(0.0, -s)
    tl, tr = clamp_tails(7.0 * s, -7.0 * s)
    return {
        "tail_l": tl,
        "tail_r": tr,
        "foot_l": (int(round(-3 * sl)), int(round(-5 * sl)), -12.0 * sl),
        "foot_r": (int(round(3 * sr)), int(round(-5 * sr)), -12.0 * sr),
    }


def apply_parts(rig, angles):
    layers = rig["layers"]
    piv = rig["pivots"]
    out = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    # Hand and tail share one pivot and one angle so the wrist cannot shear open.
    moved = {}
    for tail, hand in (("tail_l", "hand_l"), ("tail_r", "hand_r")):
        ang, piv_t = angles[tail], piv[tail]
        moved[tail] = rotate_nearest(layers[tail], ang, piv_t)
        moved[hand] = rotate_nearest(layers[hand], ang, piv_t)
    for name in ("foot_l", "foot_r"):
        dx, dy, tilt = angles[name]
        foot = rotate_nearest(layers[name], tilt, piv[name])
        foot = shift(foot, dx, dy)
        moved[name] = clamp_sole(foot, rig["soles"][name])
    for name in PART_DRAW:
        if name == "torso":
            paste(out, layers["torso"])
        else:
            paste(out, moved[name])
    if any(abs(angles[n]) > 0.25 for n in ("tail_l", "tail_r")):
        cushion(out, piv["tail_l"], PURPLE)
        cushion(out, piv["tail_r"], PURPLE)
    return despeckle(out)


def paint_budget(img, cx, cy, n, color, base_r, y_hi=None):
    if n <= 0:
        return 0
    if y_hi is None:
        y_hi = CELL_H - 2
    placed = []
    seen = set()
    ring = 0
    while len(placed) < n and ring < 20:
        r = base_r + ring
        steps = max(n * 4, 24)
        for i in range(steps):
            ang = 2 * math.pi * (i + 0.37 * ring) / steps
            x = int(round(cx + r * math.cos(ang)))
            y = int(round(cy + r * math.sin(ang)))
            if 1 <= x < CELL_W - 1 and 1 <= y <= y_hi and img[y, x, 3] == 0:
                key = (y, x)
                if key not in seen:
                    seen.add(key)
                    placed.append(key)
                    if len(placed) == n:
                        break
        ring += 1
    for y, x in placed:
        img[y, x, :3] = color
        img[y, x, 3] = 255
    return len(placed)


def open_mouth(neutral, chin, drop):
    out = neutral.copy()
    if drop <= 0 or not np.any(chin[..., 3] >= 128):
        return out
    part = np.zeros_like(neutral)
    m = chin[..., 3] >= 128
    part[m] = neutral[m]
    out[m] = 0
    moved = shift(part, 0, int(drop))
    paste(out, moved)
    gap = m & (out[..., 3] < 128)
    ys, xs = np.nonzero(gap)
    if len(ys) == 0:
        return out
    mid = float(np.median(xs))
    for y, x in zip(ys.tolist(), xs.tolist()):
        if abs(x - mid) <= 1.2:
            out[y, x, :3] = ORANGE
        else:
            out[y, x, :3] = INK
        out[y, x, 3] = 255
    return out


def burst_at(img, tip, n, color):
    return paint_budget(img, tip[0], tip[1] - 2, n, color, 4)


def synth(rig, action, f, palette):
    neutral = rig["neutral"]
    if action == "idle" and f == 0:
        return neutral.copy()
    if action == "idle":
        img = apply_parts(rig, pose_idle(f))
    elif action == "walk":
        img = apply_parts(rig, pose_walk(f))
    elif action == "attack":
        if 5 <= f <= 10:
            # Tail-hand slap. clamp_tails keeps the swing at 7 degrees:
            # past that, a finger bridge on one tail-hand splits.
            slap = (0.55, 0.85, 1.0, 1.0, 0.72, 0.38)[f - 5]
            tl, tr = clamp_tails(-7.0 * slap, 7.0 * slap)
            img = apply_parts(rig, {
                "tail_l": tl,
                "tail_r": tr,
                "foot_l": (0, 0, 0.0),
                "foot_r": (int(round(3 * slap)), int(round(-4 * slap)), -12.0 * slap),
            })
            img = open_mouth(img, rig["layers"]["chin"], 3 if slap >= 0.7 else 2)
            if f >= 7:
                burst_at(img, rig["tips"]["hand_l"], 16, WHITE)
                burst_at(img, rig["tips"]["hand_r"], 16, ORANGE)
        elif f <= 4:
            drop = (2, 3, 3, 2, 2)[f]
            img = open_mouth(neutral, rig["layers"]["chin"], drop)
            radii = (16, 22, 28, 24, 18)[f]
            mx, my = rig["mouth"]
            # Rings stay around the muzzle and the tail-hands, not under the feet.
            y_hi = min(int(my) + 4, int(min(rig["soles"].values())) - 10)
            paint_budget(img, mx, my, 24, WHITE, radii, y_hi=y_hi)
            paint_budget(img, mx, my, 24, ORANGE, radii + 6, y_hi=y_hi)
        elif f in (11, 12):
            img = neutral.copy()
            burst_at(img, rig["tips"]["hand_l"], 18, WHITE)
            burst_at(img, rig["tips"]["hand_r"], 18, ORANGE)
            if f == 12:
                burst_at(img, rig["tips"]["hand_l"], 8, ORANGE)
        else:
            img = neutral.copy()
            extra = {13: 12, 14: 6}.get(f, 0)
            if extra:
                burst_at(img, rig["tips"]["hand_l"], extra // 2, WHITE)
                burst_at(img, rig["tips"]["hand_r"], extra - extra // 2, ORANGE)
    elif action == "hurt":
        if f == 0:
            img = neutral.copy()
            m = img[..., 3] >= 128
            img[m, :3] = WHITE
        else:
            img = rotate_nearest(neutral, HURT_DEG[f], rig["center"])
            img = shift(img, HURT_DX[f], HURT_DY[f])
            img = despeckle(img)
            if 3 <= f <= 6:
                cx, cy = rig["center"]
                paint_budget(img, cx + 18, cy - 16, 5, WHITE, 3)
    else:
        raise SystemExit(action)
    return snap(img, palette)


def differ_count(a_img, b_img):
    differ = np.any(a_img != b_img, axis=2) & (
        (a_img[..., 3] >= 128) | (b_img[..., 3] >= 128)
    )
    return int(differ.sum())


def opaque_count(img):
    return int((img[..., 3] >= 128).sum())


def params_hash():
    return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]


def facing_report(front):
    white = (front[..., 3] >= 128) & np.all(front[..., :3] == WHITE, axis=2)
    ys, xs = np.nonzero(white)
    eye = white & (ys.mean() * 0 + 1).astype(bool)  # keep shape logic local
    # Eyes are the white cluster above the mouth, y<=53.
    yy, xx = np.mgrid[0:96, 0:96]
    eyes = white & (yy <= 53)
    ey, ex = np.nonzero(eyes)
    mouth = (front[..., 3] >= 128) & (yy >= 54) & (yy <= 57) & (xx >= 32) & (xx <= 50)
    my, mx = np.nonzero(mouth)
    return {
        "eye_n": int(eyes.sum()),
        "eye_mean_x": round(float(ex.mean()), 2) if len(ex) else None,
        "eye_span": [int(ex.min()), int(ex.max())] if len(ex) else None,
        "mouth_mean_x": round(float(mx.mean()), 2) if len(mx) else None,
        "mouth_mean_y": round(float(my.mean()), 2) if len(my) else None,
    }


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
        "attack": (0, 4, 6, 7, 10, 11),
        "hurt": (0, 4, 5, 8),
    }
    cols = 6
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            y, x = ai * CELL_H, si * CELL_W
            m = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    """South row, every frame, cropped. Spin and the two feet are what to read."""
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
        for ci, c in enumerate(row):
            y = ri * cell_h + (cell_h - c.shape[0]) // 2
            x = ci * cell_w + (cell_w - c.shape[1]) // 2
            m = c[..., 3] >= 128
            bg[y:y + c.shape[0], x:x + c.shape[1]][m] = c[m]
    return Image.fromarray(bg, "RGBA")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in OUT_DIR.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(OUT_DIR)
        if rel.parts[0] == "_work":
            continue
        dest = SYNC_OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        n += 1
    return n


def with_flock(lock_path, fn):
    import fcntl

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
    while i < len(text) and text[i] != "{":
        i += 1
    depth = 0
    end = i
    begun = False
    while end < len(text):
        ch = text[end]
        if ch == "{":
            depth += 1
            begun = True
        elif ch == "}":
            depth -= 1
            if begun and depth == 0:
                end += 1
                break
        end += 1
    return text[start:end]


def publish_dir8(sprite_h):
    import fcntl

    entry = {
        "dex": 424,
        "form": "normal",
        "variant": "ambipom",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0424_ambipom/normal",
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
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        neighbors = []
        for nk in ("0380_latias-mega", "0001_bulbasaur", "0425_drifloon"):
            if f'"{nk}":' in text:
                neighbors.append((nk, _entry_span(text, nk)))
        if not neighbors:
            raise SystemExit("no neighbor key to assert")
        key = f'"{KEY}":'
        if key in text:
            start = text.index(key) + len(key)
            end = start
            depth = 0
            begun = False
            while end < len(text):
                ch = text[end]
                if ch == "{":
                    depth += 1
                    begun = True
                elif ch == "}":
                    depth -= 1
                    if begun and depth == 0:
                        end += 1
                        break
                end += 1
            text = text[:start] + blob + text[end:]
        else:
            text = text.replace(prefix + "{", prefix + "{" + key + blob + ",", 1)
        for nk, raw in neighbors:
            if _entry_span(text, nk) != raw:
                raise SystemExit(f"neighbor key changed: {nk}")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(text)
        json.loads(text[len(prefix):].rstrip().rstrip(";"))
        tmp.replace(DIR8)
        return [nk for nk, _ in neighbors]

    return with_flock(DIR8_LOCK, edit)


def update_queue(status):
    import fcntl

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
                raise SystemExit("queue has no 0424_ambipom")
            tmp = QUEUE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def save_parts(front, masks, path):
    colors = {
        "tail_l": (90, 200, 220),
        "tail_r": (40, 120, 200),
        "foot_l": (40, 200, 80),
        "foot_r": (20, 90, 40),
        "torso": (180, 140, 220),
        "hand_l": (240, 180, 40),
        "hand_r": (220, 80, 40),
        "chin": (255, 255, 255),
    }
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    for name, col in colors.items():
        if name == "chin":
            continue
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    part[masks["chin"], :3] = colors["chin"]
    part[masks["chin"], 3] = 255
    big = np.repeat(np.repeat(part, 4, 0), 4, 1)
    Image.fromarray(big, "RGBA").save(path)


def south_attack_ratios(rig, palette):
    frames = [synth(rig, "attack", f, palette) for f in range(16)]
    counts = [opaque_count(im) for im in frames]
    base = counts[0]
    ratios = [round(c / base, 4) for c in counts]
    return frames, counts, ratios


def qa_numbers(master, ratios, spin_cc, hurt_cc, rest_diff, sprite_h):
    pal = None
    reasons = []
    if any(r < 0.90 or r > 1.08 for r in ratios):
        reasons.append("attack ratio")
    if rest_diff:
        reasons.append("rest diff")
    opaque = master[..., 3] >= 128 if master is not None else None
    off = None
    alphas = None
    if master is not None:
        alphas = [int(v) for v in np.unique(master[..., 3])]
        if any(v not in (0, 255) for v in alphas):
            reasons.append("alpha")
    if spin_cc and (not spin_cc[0] or spin_cc[0] < 0.92):
        reasons.append("spin split")
    if any(abs(a) > 38 for a in HURT_DEG):
        reasons.append("hurt angle")
    return reasons, alphas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(STILL_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks, counts = segment(front)
    print("parts", counts, "palette", len(palette), flush=True)
    print("facing", json.dumps(facing_report(front)), flush=True)
    if max(abs(a) for a in HURT_DEG) > 38:
        raise SystemExit("hurt lean exceeds 38")

    placed, origin, _ = place_still(front)
    rigs = {d: build_rig(front, masks, origin, flip=d not in UNFLIPPED) for d in DIRS}
    rest_s = synth(rigs["S"], "idle", 0, palette)
    rest_n = synth(rigs["N"], "idle", 0, palette)
    diff_s = differ_count(rest_s, placed)
    diff_n = differ_count(rest_n, flip_layer(placed))
    print("rest-pose pixels changed S", diff_s, "N", diff_n, flush=True)
    ident = apply_parts(rigs["S"], pose_idle(0))
    diff_parts = differ_count(snap(ident, palette), placed)
    print("identity parts diff", diff_parts, flush=True)
    if diff_s or diff_n or diff_parts:
        raise SystemExit("idle frame 0 does not match the still")

    # Feet on every idle frame stay on the frame-0 sole.
    sole = int(np.nonzero(placed[..., 3] >= 128)[0].max())
    for f in (4, 8, 12):
        img = synth(rigs["S"], "idle", f, palette)
        foot = img[..., 3] >= 128
        # The original sole row must still contain the foot pixels' bottom.
        ys = np.nonzero(foot)[0]
        if int(ys.max()) != sole:
            raise SystemExit(f"idle {f} sole moved to {int(ys.max())} from {sole}")

    atk_frames, atk_counts, ratios = south_attack_ratios(rigs["S"], palette)
    print("south attack counts", atk_counts, flush=True)
    print("south attack ratios", ratios, flush=True)
    spin_cc = {}
    for f, img in ((f, atk_frames[f]) for f in range(5, 11)):
        sizes = component_sizes(img)
        spin_cc[f] = sizes[:4]
    hurt_cc = {}
    for f in (0, 4, 5, 11):
        hurt_cc[f] = component_sizes(synth(rigs["S"], "hurt", f, palette))[:4]
    print("spin cc", spin_cc, flush=True)
    print("hurt cc", hurt_cc, flush=True)
    ratio_bad = any(r < 0.90 or r > 1.08 for r in ratios)
    spin_bad = any((not sizes) or (sizes[0] < 0.92 * sum(sizes)) or (len(sizes) > 1 and sizes[1] > 24) for sizes in spin_cc.values())

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    save_parts(front, masks, work / "parts.png")

    if args.samples or ratio_bad or spin_bad:
        want = [("idle", "S", f) for f in (0, 4, 8, 12)]
        want += [("walk", "S", f) for f in (0, 4, 8, 12)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in (0, 2, 4, 8, 11)]
        want += [("walk", "E", f) for f in (4,)]
        tiles = []
        for act, d, f in want:
            if act == "attack" and d == "S":
                tiles.append(atk_frames[f])
            else:
                tiles.append(synth(rigs[d], act, f, palette))
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        print("samples", work / "samples.png", flush=True)
        if args.samples and not ratio_bad and not spin_bad:
            return
        if ratio_bad or spin_bad:
            if not args.no_publish:
                update_queue("needs_fix")
            raise SystemExit("south attack failed ratio or spin connectivity")

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
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions_dir = OUT_DIR / "actions"
    actions_dir.mkdir(exist_ok=True)
    preview.save(actions_dir / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    # Re-measure south attack from the sheet so the printed ratios match the files.
    sheet_counts = []
    for f in range(16):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        sheet_counts.append(opaque_count(tile))
    sheet_ratios = [round(c / sheet_counts[0], 4) for c in sheet_counts]
    print("sheet south attack ratios", sheet_ratios, flush=True)

    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    edge_n = sum(len(v) for v in clip.values())
    report = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_h,
        "cell": [CELL_W, CELL_H, FOOT_Y],
        "rest_diff_s": diff_s,
        "rest_diff_n": diff_n,
        "identity_parts_diff": diff_parts,
        "off_palette_colors": len(off),
        "alphas": alphas,
        "edge_touching": edge_n,
        "south_attack_counts": sheet_counts,
        "south_attack_ratios": sheet_ratios,
        "spin_cc_s": {str(k): v for k, v in spin_cc.items()},
        "hurt_cc_s": {str(k): v for k, v in hurt_cc.items()},
        "hurt_deg_max": max(abs(a) for a in HURT_DEG),
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        qa = work / "qa"
        qa.mkdir(exist_ok=True)
        for act, kind in (("idle", "hover"), ("walk", "gait")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run(
                [
                    py, "-I", str(GAIT_LOOP), "select",
                    "--frames-dir", str(frames_dir / act / "S"),
                    "--fps", "12", "--output-dir", str(od),
                    "--kind", kind, "--no-aids",
                ],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1600], flush=True)

    meta = {
        "generator": "forge_ambipom_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 424,
        "variant": "ambipom",
        "form": "normal",
        "archetype": "biped",
        "facing_still": "camera",
        "hands": 2,
        "feet": 2,
        "tails": 2,
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
        "south_attack_ratios": sheet_ratios,
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = []
    if sheet_ratios != ratios and any(r < 0.90 or r > 1.08 for r in sheet_ratios):
        failed.append("sheet ratio")
    if any(r < 0.90 or r > 1.08 for r in sheet_ratios):
        failed.append("ratio")
    if off:
        failed.append("palette")
    if alphas != [0, 255]:
        failed.append("alpha")
    if edge_n:
        failed.append("edge")
    if diff_s or diff_n:
        failed.append("rest")
    if sprite_h <= 0:
        failed.append("h")
    if failed:
        print("FAIL", failed, flush=True)
        if not args.no_publish:
            update_queue("needs_fix")
        raise SystemExit("qa failed: " + ",".join(failed))

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if not args.no_publish:
        neighbors = publish_dir8(sprite_h)
        update_queue("done")
        print("published", neighbors, "h", sprite_h, flush=True)


if __name__ == "__main__":
    main()
