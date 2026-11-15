#!/usr/bin/env python3
"""Infernape (0392_infernape) 8-direction action sheet from the 96x96 front still.

Measured on front.png: biped lunge, facing left. Snout tip x=19 is left of the
eyes (mean x=37.8). Unflipped dirs are N NW W SW. Flipped dirs are S SE E NE.

Parts are only what the still contains: head flame, raised arm, near arm, head,
torso, brown tail, and two legs with the existing gold guards and blue feet.
No invented mouth. The roar flares the head flame and draws rings. The spin and
the hurt lean rotate the whole body as one piece (hurt peaks at 38 degrees).

Does not write front/back png or gif, shiny, reverse, box-icons, or legends-arceus.
"""

import argparse
import hashlib
import json
import math
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
STILL_DIR = ROOT / "monsterSprite/gen4/0392_infernape/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0392_infernape/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left, so the left-facing compass keeps the drawing.
FACE_LEFT = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 176, 148
VERSION = "infernape_v2"

INK = (16, 16, 16)
RED = (230, 57, 65)
GOLD = (255, 222, 82)
ORANGE = (246, 139, 74)
WHITE = (255, 255, 255)
DARK_GOLD = (131, 90, 8)
GOLD2 = (197, 156, 32)

SEEDS = {
    "arm_up": [(16, 18), (15, 24), (18, 28), (20, 26)],
    "flame": [(16, 70), (21, 50), (24, 45)],
    "head": [(34, 32), (37, 28), (40, 41), (33, 24), (36, 36)],
    "torso": [(48, 40), (52, 46), (45, 38), (50, 42)],
    "tail": [(40, 68), (48, 60), (54, 52)],
    "leg_l": [(66, 33), (72, 31), (60, 30), (74, 32)],
    "leg_r": [(64, 54), (70, 60), (76, 59), (60, 52)],
    "arm_low": [(45, 29), (44, 26), (47, 32)],
}
EXPECT = {
    "arm_up": 148,
    "flame": 345,
    "head": 354,
    "torso": 209,
    "tail": 160,
    "leg_l": 291,
    "leg_r": 259,
    "arm_low": 107,
}
DRAW = ("tail", "leg_r", "torso", "leg_l", "arm_low", "flame", "head", "arm_up")
MOVING = ("arm_up", "flame", "tail", "leg_l", "leg_r", "arm_low")


def flood_owners(img):
    a = img[..., 3] >= 128
    h, w = a.shape
    names = list(SEEDS)
    owner = np.full((h, w), -1, np.int16)
    q = deque()
    for i, name in enumerate(names):
        for y, x in SEEDS[name]:
            if not a[y, x]:
                raise SystemExit(f"seed off the sprite: {name} {(y, x)}")
            if owner[y, x] < 0:
                owner[y, x] = i
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and owner[ny, nx] < 0:
                owner[ny, nx] = owner[y, x]
                q.append((ny, nx))
    # The one diagonal wisp above the crown belongs to the head flame.
    flame_i = names.index("flame")
    ys, xs = np.nonzero(a & (owner < 0))
    claimed = owner >= 0
    cys, cxs = np.nonzero(claimed)
    for y, x in zip(ys.tolist(), xs.tolist()):
        d = (cys.astype(np.int32) - y) ** 2 + (cxs.astype(np.int32) - x) ** 2
        j = int(d.argmin())
        owner[y, x] = owner[cys[j], cxs[j]]
        if owner[y, x] < 0:
            owner[y, x] = flame_i
    masks = {name: owner == i for i, name in enumerate(names)}
    counts = {name: int(m.sum()) for name, m in masks.items()}
    if counts != EXPECT:
        raise SystemExit(f"part counts drifted: {counts}")
    if int(sum(counts.values())) != int(a.sum()):
        raise SystemExit("segmentation dropped pixels")
    return masks, counts


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


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
    """4x nearest. A full turn lands on the same pixels as angle 0."""
    angle_deg = angle_deg % 360.0
    if angle_deg > 180.0:
        angle_deg -= 360.0
    if abs(angle_deg) < 0.45 or not np.any(layer[..., 3] >= 128):
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
    vy, vx = np.nonzero(valid)
    out[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    return out[1::4, 1::4].copy()


def despeckle(img):
    """Drop pixels whose four cardinal neighbours are empty. Eyes stay."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def close_cracks(img, passes=2):
    """Fill 1px rotation cracks. A hole needs two opaque cardinal neighbours."""
    out = img.copy()
    for _ in range(passes):
        a = out[..., 3] >= 128
        p = np.pad(a, 1, constant_values=False)
        up, dn = p[:-2, 1:-1], p[2:, 1:-1]
        lf, rt = p[1:-1, :-2], p[1:-1, 2:]
        n = up.astype(np.int16) + dn + lf + rt
        hole = (~a) & (n >= 2)
        if not np.any(hole):
            break
        col = np.zeros_like(out)
        # Prefer an opaque neighbour. Later writes do not cover an earlier one.
        for src, take in ((up, (0, -1)), (dn, (0, 1)), (lf, (-1, 0)), (rt, (1, 0))):
            use = hole & src & (col[..., 3] == 0)
            if not np.any(use):
                continue
            shifted = shift(out, take[0], take[1])
            col[use] = shifted[use]
        out[hole] = col[hole]
    return out


def component_sizes(img):
    a = img[..., 3] >= 128
    h, w = a.shape
    seen = np.zeros_like(a)
    sizes = []
    ys, xs = np.nonzero(a)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        n = 0
        while q:
            y, x = q.popleft()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    step = 400
    pts_y = ys.astype(np.int32)
    pts_x = xs.astype(np.int32)
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None].astype(np.int32) - pts_y[None, :]) ** 2
        d += (tx[sl, None].astype(np.int32) - pts_x[None, :]) ** 2
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


def socket_pivot(part, body):
    sel = part & dilate(body, 2)
    if not np.any(sel):
        sel = part & dilate(body, 6)
    if not np.any(sel):
        sel = part
    ys, xs = np.nonzero(sel)
    return float(xs.mean()), float(ys.mean())


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def build_rig(img, masks, flip):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
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

    torso = placed["torso"]
    head = placed["head"]
    parents = {
        "arm_up": head | torso,
        "flame": head,
        "arm_low": head | torso,
        "tail": torso,
        "leg_l": torso,
        "leg_r": torso,
    }
    # Cushion sits under every part. Parents drawn later must not cover the still.
    pivots = {}
    cushion = np.zeros_like(canvas)
    for name, body in parents.items():
        pivots[name] = socket_pivot(placed[name], body)
        zone = placed[name] & dilate(body, 3)
        if not np.any(zone):
            continue
        parent_name = "head" if name in ("arm_up", "flame") else "torso"
        fill = nearest_color(canvas, zone, placed[parent_name])
        write = zone & (layers[parent_name][..., 3] < 128)
        cushion[write] = fill[write]

    tys, txs = np.nonzero(torso)
    torso_pivot = (float(txs.mean()), float(tys.mean()))
    hip = (
        (pivots["leg_l"][0] + pivots["leg_r"][0]) * 0.5,
        (pivots["leg_l"][1] + pivots["leg_r"][1]) * 0.5,
    )

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        cushion = cushion[:, ::-1].copy()
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        torso_pivot = (CELL_W - 1 - torso_pivot[0], torso_pivot[1])
        hip = (CELL_W - 1 - hip[0], hip[1])

    facing = -1.0 if not flip else 1.0
    rest = np.zeros_like(canvas)
    paste(rest, cushion)
    for name in DRAW:
        paste(rest, layers[name])
    return {
        "layers": layers,
        "pivots": pivots,
        "cushion": cushion,
        "facing": facing,
        "rest": rest,
        "torso_pivot": torso_pivot,
        "hip": hip,
    }


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    """Frame 0 is the still. Walk and idle waves are 0 again at the loop point."""
    p = dict(
        dx=0.0, dy=0.0, rigid=0.0, pivot="hip",
        arm_up=0.0, arm_low=0.0, leg_l=0.0, leg_r=0.0, tail=0.0, flame=0.0,
        flare=0, ring=0, lines=0, burst=0, flash=0, sparks=0, spin=0,
    )
    if action == "idle":
        # Feet stay down. Flame, tail and the two arms breathe.
        p["flame"] = wave(f, 16, 6.0, phase=0.4)
        p["tail"] = wave(f, 16, 5.0, phase=-0.9)
        p["arm_up"] = wave(f, 16, 5.0, phase=0.2)
        p["arm_low"] = wave(f, 16, -4.0, phase=-0.6)
    elif action == "walk":
        # Alternating steps. Each leg only rotates on the side that lifts the foot.
        # The back foot is already on the ground line, so a downward angle would sink it.
        step = wave(f, 16, 26.0)
        p["leg_r"] = max(0.0, step)
        p["leg_l"] = min(0.0, step)
        p["dx"] = wave(f, 16, 1.0)
        p["arm_up"] = wave(f, 16, -16.0)
        p["arm_low"] = wave(f, 16, 18.0, phase=0.5)
        p["tail"] = wave(f, 16, 9.0, phase=-0.8)
        p["flame"] = wave(f, 16, 6.0, phase=0.5)
    elif action == "attack":
        # 0-4 roar, 5-10 one-piece spin, 11-12 impact, 13-15 recover.
        flame = [0, 8, 14, 18, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        flare = [0, 2, 3, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        arm_u = [0, -6, -10, -12, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        arm_l = [0, -4, -8, -10, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        tail = [0, 4, 7, 8, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        p["flame"] = flame[f]
        p["flare"] = flare[f]
        p["arm_up"] = arm_u[f]
        p["arm_low"] = arm_l[f]
        p["tail"] = tail[f]
        if 1 <= f <= 4:
            p["ring"] = 1 + (f - 1) % 3
        if 5 <= f <= 10:
            p["spin"] = 1
            p["lines"] = 4
            # 60 degrees per frame, full turn, upright again on frame 10.
            p["rigid"] = 60.0 * (f - 4)
            p["pivot"] = "torso"
        if f == 11:
            p["rigid"] = -12.0
            p["dx"] = 5
            p["burst"] = 11
        elif f == 12:
            p["rigid"] = -8.0
            p["dx"] = 3
            p["burst"] = 8
        elif f == 13:
            p["dx"] = 2
            p["rigid"] = -3.0
        elif f == 14:
            p["dx"] = 1
    else:
        ang = [0, 14, 26, 34, 38, 34, 26, 16, 10, 5, 2, 0]
        back = [0, 2, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0]
        p["rigid"] = ang[f]
        p["dx"] = -back[f]
        p["dy"] = [0, -1, -2, -2, -2, -1, -1, 0, 0, 0, 0, 0][f]
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
        p["pivot"] = "hip"
    return p


def put_empty(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def flare_tip(img, flame, pivot, amount):
    """Grow the existing crown flame outward. No new colour, no new limb."""
    if amount <= 0:
        return img
    a = flame[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return img
    d = (xs.astype(np.float32) - pivot[0]) ** 2 + (ys.astype(np.float32) - pivot[1]) ** 2
    i = int(d.argmax())
    tx, ty = float(xs[i]), float(ys[i])
    vx, vy = tx - pivot[0], ty - pivot[1]
    norm = math.hypot(vx, vy) or 1.0
    ux, uy = vx / norm, vy / norm
    cols = (ORANGE, GOLD, RED, DARK_GOLD)
    for k in range(1, amount + 1):
        x = tx + ux * k * 1.35
        y = ty + uy * k * 1.35
        put_empty(img, x, y, cols[min(k - 1, 3)])
        put_empty(img, x - uy, y + ux, GOLD2 if k > 1 else GOLD)
    return img


def snout_point(head, facing):
    a = head[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return None
    y_mid = float(np.median(ys))
    band = np.abs(ys - y_mid) <= 6
    if not np.any(band):
        band = np.ones(len(ys), dtype=bool)
    sel = np.flatnonzero(band)
    i = int(sel[np.argmin(xs[band]) if facing < 0 else np.argmax(xs[band])])
    return int(xs[i]), int(ys[i])


def roar_rings(out, mouth, facing, strength):
    mx, my = mouth
    cols = (GOLD, ORANGE, WHITE)
    for r_i in range(strength):
        rad = 6 + r_i * 5
        col = cols[r_i % 3]
        for deg in range(-50, 51, 8):
            ang = math.radians(deg)
            put_empty(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.72, col)
    return out


def speed_lines(out, facing, n, seed):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    # Streaks leave the trailing side. facing < 0 looks left, so the rear is +x.
    if facing < 0:
        rear = x1 - 1
        step = 1
    else:
        rear = x0 + 1
        step = -1
    y_lo = y0 + (y1 - y0) // 3
    y_hi = y1 - 4
    if y_hi <= y_lo:
        return out
    for k in range(n):
        y = int(rng.integers(y_lo, y_hi))
        length = int(rng.integers(5, 10))
        off = int(rng.integers(2, 6))
        for i in range(length):
            put_empty(out, rear + step * (off + i), y, GOLD if i < 2 else ORANGE)
    return out


def impact_burst(out, facing, size):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    cx = (x0 - 4) if facing < 0 else (x1 + 4)
    cy = y0 + (y1 - y0) // 2
    for k in range(8):
        ang = k * math.pi / 4 + 0.2
        for r in range(2, size):
            put_empty(out, cx + r * math.cos(ang), cy + r * math.sin(ang), WHITE if r < size - 3 else GOLD)
    put_empty(out, cx, cy, WHITE)
    return out


def sparks(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    cx = (x0 + x1) // 2
    back = -facing
    pts = (
        (cx + back * (8 + frame_idx), y0 + 4),
        (cx + back * (4 + frame_idx), y0 - 2),
        (cx + back * 6, y0 + 10),
    )
    for sx, sy in pts:
        put_empty(out, sx, sy, WHITE)
        put_empty(out, sx + back, sy + 1, ORANGE)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def rigid_transform(img, angle, pivot, dx, dy):
    out = img
    if abs(angle) >= 0.45:
        out = rotate_nearest(out, angle, pivot)
        out = close_cracks(out)
        out = despeckle(out)
    if dx or dy:
        out = shift(out, dx, dy)
    return out


def synth(rig, action, f, palette):
    layers = rig["layers"]
    pivots = rig["pivots"]
    facing = rig["facing"]
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    sgn = facing  # mirror limb angles when the sprite is flipped
    bdx, bdy = p["dx"] * facing, p["dy"]

    spinning = p["spin"] == 1
    rigid = abs(p["rigid"]) >= 0.45
    part_move = any(abs(p[k]) >= 0.45 for k in ("arm_up", "arm_low", "leg_l", "leg_r", "tail", "flame"))
    pivot = rig["torso_pivot"] if p["pivot"] == "torso" else rig["hip"]
    # Spin and the recoil/impact leans turn the still as one piece.
    if spinning or (rigid and not part_move):
        body = rigid_transform(rig["rest"], p["rigid"] * sgn, pivot, bdx, bdy)
    else:
        def moved(name, ang):
            lyr = rotate_nearest(layers[name], ang * sgn, pivots[name])
            return shift(lyr, bdx, bdy)

        body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        paste(body, shift(rig["cushion"], bdx, bdy))
        part_layers = {
            "tail": moved("tail", p["tail"]),
            "leg_r": moved("leg_r", p["leg_r"]),
            "torso": shift(layers["torso"], bdx, bdy),
            "leg_l": moved("leg_l", p["leg_l"]),
            "arm_low": moved("arm_low", p["arm_low"]),
            "flame": moved("flame", p["flame"]),
            "head": shift(layers["head"], bdx, bdy),
            "arm_up": moved("arm_up", p["arm_up"]),
        }
        for name in DRAW:
            paste(body, part_layers[name])
        if part_move:
            body = despeckle(body)
        if rigid:
            # Impact lean keeps the posed body one piece.
            bb = body_box(body)
            center = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0) if bb else pivot
            body = rigid_transform(body, p["rigid"] * sgn, center, 0, 0)
        if p["flare"]:
            flame = part_layers["flame"]
            body = flare_tip(body, flame, (pivots["flame"][0] + bdx, pivots["flame"][1] + bdy), p["flare"])

    body = snap(body, palette)
    if p["flash"]:
        body = flash_white(body)
    if p["ring"]:
        head = shift(layers["head"], bdx, bdy)
        mouth = snout_point(head, facing)
        if mouth:
            body = roar_rings(body, mouth, facing, p["ring"])
    if p["lines"]:
        body = speed_lines(body, facing, p["lines"], seed=f * 17 + int(facing > 0))
    if p["burst"]:
        body = impact_burst(body, facing, p["burst"])
    if p["sparks"]:
        body = sparks(body, f, facing)
    return snap(body, palette)


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
    """South row, several frames of each action, on a dark field."""
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 4, 8, 12),
        "attack": (0, 3, 6, 7, 10, 11),
        "hurt": (0, 4, 7, 10),
    }
    cols = 6
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
            # East row is index 2. East must face screen-right (flipped still).
            tile = sheets[act][2 * CELL_H:3 * CELL_H, f * CELL_W:(f + 1) * CELL_W]
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
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def placed_still(front):
    a = front[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    placed = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    placed[dst_y:dst_y + sh, dst_x:dst_x + sw] = front[y0:y1, x0:x1]
    return placed


def differ_count(a_img, b_img):
    differ = np.any(a_img != b_img, axis=2) & ((a_img[..., 3] >= 128) | (b_img[..., 3] >= 128))
    return int(differ.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(STILL_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks, counts = flood_owners(front)
    print("parts", counts, "palette", len(palette), flush=True)

    rigs = {}
    for d in DIRS:
        rigs[d] = build_rig(front, masks, flip=d not in FACE_LEFT)

    placed = placed_still(front)
    rest_w = synth(rigs["W"], "idle", 0, palette)
    rest_e = synth(rigs["E"], "idle", 0, palette)
    diff_w = differ_count(rest_w, placed)
    diff_e = differ_count(rest_e, placed[:, ::-1].copy())
    print("rest-pose pixels changed W", diff_w, "E", diff_e, flush=True)
    if diff_w or diff_e:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (180, 180, 190), "flame": (255, 80, 20), "tail": (160, 90, 40),
        "leg_l": (40, 200, 80), "leg_r": (20, 80, 220), "arm_up": (40, 180, 255),
        "head": (255, 220, 40), "arm_low": (220, 40, 180),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    # Spin and hurt must stay one body. Effects are not in this check.
    spin_cc = {}
    for f in (5, 6, 7, 8, 9, 10):
        img = synth(rigs["W"], "attack", f, palette)
        # Lines are separate components; measure the pre-effect body via sizes.
        sizes = component_sizes(img)
        spin_cc[f] = sizes[:4]
    hurt_cc = {}
    for f in (0, 4, 5, 11):
        hurt_cc[f] = component_sizes(synth(rigs["W"], "hurt", f, palette))[:4]
    print("spin cc", spin_cc, flush=True)
    print("hurt cc", hurt_cc, flush=True)

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 8, 12)]
        want += [("attack", "W", f) for f in (0, 2, 4, 5, 7, 10, 11, 13)]
        want += [("attack", "E", f) for f in (3, 7, 11)]
        want += [("hurt", "W", f) for f in (0, 1, 4, 8, 11)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        tw, th = CELL_W, CELL_H
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * th, cols * tw, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = tile[m]
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

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
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
    report = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_h,
        "rest_diff_w": diff_w,
        "rest_diff_e": diff_e,
        "off_palette_colors": len(off),
        "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "spin_cc_w": {str(k): v for k, v in spin_cc.items()},
        "hurt_cc_w": {str(k): v for k, v in hurt_cc.items()},
    }
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
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1200], flush=True)

    meta = {
        "generator": "forge_infernape_actions",
        "version": VERSION,
        "key": "0392_infernape",
        "dex": 392,
        "variant": "infernape",
        "form": "normal",
        "archetype": "biped",
        "facing_still": "left",
        "snout_x": 19,
        "eye_mean_x": 37.8,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "face_left": sorted(FACE_LEFT),
        "palette_colors": int(len(palette)),
        "off_palette_colors": len(off),
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
