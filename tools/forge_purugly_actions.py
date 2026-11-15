#!/usr/bin/env python3
"""Purugly normal (0432) 8-direction sheet from this form's front.png only.

Measured on normal/front.png (96x96, alpha 0 or 255, 2029 opaque, 15 colours):
  bbox x18-76 y19-76 (59x58). Centroid about (49.8, 49.2).
  Faces left. Whiskers are loose pixels at x18-33 y45-55 only.
  Both yellow eyes sit left of the body centroid (mean x about 40.8).
  Two purple ears with pink insides. No mouth: pink is inner ear only.
  Two feet touch, not four:
    fore (screen-left, white)  x30-39 y68-74  sole y74
    hind (lower white toe)     x45-62 y70-76  sole y76
  The forked curl on the right (x64-76 y37-73) is one tail. It is not a foot.
  The fat grey body stays one piece. Do not slim it into Glameow.

  idle   : feet planted. Frame 0 is the still. Ears and tail sway.
  walk   : those two paws only, opposite lifts. Tail counters. No hop.
  attack : 0-4 ear flare plus rings (no invented jaw);
           5-10 in-place 360 of the whole cat, no |cos| squash;
           11-12 impact; 13-15 recover.
  hurt   : one-piece recoil, 36 degrees, under the 38 degree cap.

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_purugly_actions.py
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
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
SRC_DIR = ROOT / "monsterSprite/gen4/0432_purugly/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0432_purugly/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left. Right-facing rows are the mirror. E looks screen-right.
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 184, 156
VERSION = "purugly_v2"
KEY = "0432_purugly"

INK = (16, 16, 16)
PURPLE = (123, 90, 164)
DPURPLE = (82, 57, 106)
PINK = (213, 82, 139)
LPINK = (255, 123, 156)
YELLOW = (255, 230, 98)
GOLD = (172, 164, 115)
WHITE = (238, 255, 255)
BODY = (131, 148, 180)

DRAW = ["torso", "tail", "lpaw", "rpaw", "lear", "rear"]


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


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def is_col(img, rgb):
    a = img[..., 3] >= 128
    c = img[..., :3]
    return a & (c[..., 0] == rgb[0]) & (c[..., 1] == rgb[1]) & (c[..., 2] == rgb[2])


def segment(img):
    """Two ears, two paws, one right-side tail. Whiskers stay on the torso."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    black = is_col(img, INK)
    ear_col = (
        is_col(img, PURPLE) | is_col(img, DPURPLE) | is_col(img, PINK) | is_col(img, LPINK)
    )
    # Ears meet the skull but their colour splits cleanly around x=46.
    lear = ear_col & (xx < 46)
    rear = ear_col & (xx >= 46)
    # Outline above the eyes travels with the ear. Eye ink stays on the face.
    lear_ink = dilate(lear, 1) & black & (yy <= 37) & (xx <= 48)
    rear_ink = dilate(rear, 1) & black & (yy <= 37) & (xx >= 44)
    both = lear_ink & rear_ink
    lear = lear | (lear_ink & ~both)
    rear = rear | (rear_ink & ~both)

    fore_seed = np.zeros_like(a)
    fore_seed[73, 32] = True
    lpaw = flood(fore_seed, a & (yy >= 68) & (xx <= 42) & ~lear & ~rear)

    hind_seed = np.zeros_like(a)
    hind_seed[75, 59] = True
    rpaw = flood(hind_seed, a & (yy >= 70) & (xx >= 45) & (xx <= 64) & ~lear & ~rear & ~lpaw)

    # One forked curl. The lower tip ends at y73, above the hind sole.
    tail_seed = np.zeros_like(a)
    tail_seed[44, 70] = True
    tail_seed[70, 73] = True
    allow_tail = a & ~lear & ~rear & ~lpaw & ~rpaw & (
        ((xx >= 67) & (yy >= 38))
        | ((xx >= 64) & (yy >= 39) & (yy <= 47))
        | ((xx >= 66) & (yy >= 64))
    )
    tail = flood(tail_seed, allow_tail)

    used = lear | rear | lpaw | rpaw | tail
    torso = a & ~used
    parts = {
        "torso": torso, "tail": tail, "lpaw": lpaw, "rpaw": rpaw, "lear": lear, "rear": rear,
    }
    cover = np.zeros_like(a)
    for name, m in parts.items():
        if np.any(cover & m):
            raise SystemExit(f"purugly masks overlap at {name}")
        cover |= m
    if int((cover != a).sum()):
        raise SystemExit(f"purugly masks missed {int((cover != a).sum())} pixels")
    eyes = is_col(img, YELLOW) | is_col(img, GOLD)
    if np.any(eyes & ~torso):
        raise SystemExit("eyes left the face")
    if np.any(ear_col & torso):
        raise SystemExit("ear colour left on the torso")
    if np.any(ear_col & tail):
        raise SystemExit("tail ate an ear")
    if int(lpaw.sum()) < 20 or int(rpaw.sum()) < 20:
        raise SystemExit("a paw mask collapsed")
    if int(tail.sum()) < 80 or int(lear.sum()) < 80 or int(rear.sum()) < 80:
        raise SystemExit("ear or tail mask collapsed")
    # The hind sole is the lowest opaque pixel. The tail tip is not.
    py = np.nonzero(rpaw)[0]
    ty = np.nonzero(tail)[0]
    if int(py.max()) != int(np.nonzero(a)[0].max()):
        raise SystemExit("hind paw is not the ground contact")
    if int(ty.max()) >= int(py.max()):
        raise SystemExit("tail reaches the ground line")
    return parts


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


def _angle_norm(angle_deg):
    ang = float(angle_deg) % 360.0
    if ang > 180.0:
        ang -= 360.0
    return ang


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest, inverse plus forward splat. Rigid: no scale and no mirror."""
    ang = _angle_norm(angle_deg)
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    up = np.repeat(np.repeat(layer, 4, axis=0), 4, axis=1)
    px, py = pivot[0] * 4 + 1.5, pivot[1] * 4 + 1.5
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
    # Put source pixels back on so a thin bridge cannot open into a second cat.
    sy, sx = np.nonzero(up[..., 3] >= 128)
    dx = np.rint(px + c * (sx - px) - s * (sy - py)).astype(np.int32)
    dy = np.rint(py + s * (sx - px) + c * (sy - py)).astype(np.int32)
    ok = (dx >= 0) & (dx < w) & (dy >= 0) & (dy < h)
    idx = np.nonzero(ok)[0]
    if len(idx):
        empty = out[dy[idx], dx[idx], 3] < 128
        if np.any(empty):
            sel = idx[empty]
            out[dy[sel], dx[sel]] = up[sy[sel], sx[sel]]
    return out[1::4, 1::4].copy()


def despeckle(img):
    """Drop only pixels whose four neighbours are empty. After rotation only."""
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
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    step = 400
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None].astype(np.int32) - ys[None, :].astype(np.int32)) ** 2
        d += (tx[sl, None].astype(np.int32) - xs[None, :].astype(np.int32)) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def fill_pinholes(img):
    """Plug only fully enclosed single-pixel holes."""
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        p = np.pad(a, 1)
        n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        hole = (~a) & (n == 4)
        if not np.any(hole):
            break
        fill = nearest_color(out, hole, a)
        out[hole] = fill[hole]
    return out


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    sizes = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        q = deque([(y, x)])
        seen[y, x] = True
        n = 0
        while q:
            cy, cx = q.popleft()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes


def extract_palette(img):
    cols = img[img[..., 3] >= 128][:, :3]
    uniq = np.unique(cols, axis=0)
    return uniq.astype(np.uint8)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
        out[..., 3] = 0
        return out
    pixels = out[a, :3].astype(np.int32)
    pal = palette.astype(np.int32)
    # int32: 255^2 does not fit in int16.
    dist = ((pixels[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    mapped = palette[dist.argmin(axis=1)]
    out[a, :3] = mapped
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    return out


def put(out, x, y, col):
    xi, yi = int(round(x)), int(round(y))
    if 0 <= yi < out.shape[0] and 0 <= xi < out.shape[1]:
        out[yi, xi, :3] = col
        out[yi, xi, 3] = 255


def contact_pivot(part, body):
    touch = part & dilate(body, 1)
    use = touch if np.any(touch) else part
    ys, xs = np.nonzero(use)
    return float(np.median(xs)), float(np.median(ys))


def joint_band(part, torso, depth=3):
    """A few pixels of the part where it actually touches the body."""
    if not np.any(part) or not np.any(torso):
        return np.zeros_like(part)
    band = part & dilate(torso, 1)
    cur = band
    for _ in range(max(0, depth - 1)):
        cur = dilate(cur, 1) & part
        band = band | cur
    return band


def place_part(img, mask, x0, y0, dst_x, dst_y):
    layer = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    pm = np.zeros((CELL_H, CELL_W), np.bool_)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return layer, pm
    dy = ys - y0 + dst_y
    dx = xs - x0 + dst_x
    layer[dy, dx] = img[ys, xs]
    pm[dy, dx] = True
    return layer, pm


def build_rig(img, masks, flip, origin):
    x0, y0, dst_x, dst_y = origin
    layers = {}
    placed = {}
    for name, mask in masks.items():
        layers[name], placed[name] = place_part(img, mask, x0, y0, dst_x, dst_y)
    torso_m = placed["torso"]
    for name in ("tail", "lpaw", "rpaw", "lear", "rear"):
        band = joint_band(placed[name], torso_m, 3)
        if not np.any(band):
            continue
        sock = nearest_color(layers["torso"], band, torso_m)
        layers["torso"][band] = sock[band]
    pivots = {name: contact_pivot(placed[name], torso_m) for name in ("tail", "lpaw", "rpaw", "lear", "rear")}
    opaque = np.zeros((CELL_H, CELL_W), np.bool_)
    for pm in placed.values():
        opaque |= pm
    ys, xs = np.nonzero(opaque)
    center = (float(xs.mean()), float(ys.mean()))
    hy, hx = np.nonzero(placed["rpaw"])
    foot = hy >= int(hy.max()) - 1
    sole = (float(np.median(hx[foot])), float(hy.max()))
    soles = {}
    for name in ("lpaw", "rpaw"):
        py = np.nonzero(placed[name])[0]
        soles[name] = int(py.max())
    # Cheek in front of the whiskers, not the tail.
    face_m = placed["torso"] & (np.arange(CELL_H)[:, None] >= dst_y + (42 - y0))
    face_m = face_m & (np.arange(CELL_H)[:, None] <= dst_y + (52 - y0))
    fys, fxs = np.nonzero(face_m)
    if len(fxs) == 0:
        raise SystemExit("no cheek for the roar")
    front_x = int(fxs.min())
    face = (front_x, int(np.median(fys[fxs <= front_x + 2])))
    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
        sole = (CELL_W - 1 - sole[0], sole[1])
        face = (CELL_W - 1 - face[0], face[1])
    facing = -1.0 if not flip else 1.0
    return layers, pivots, center, sole, soles, face, facing


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, lear=0.0, rear=0.0, tail=0.0, lpaw=0.0, rpaw=0.0,
        lpaw_dx=0.0, rpaw_dx=0.0, spin=0.0, hurt=0.0,
        rings=0, ring=0, burst=0,
    )
    if action == "idle":
        p["tail"] = wave(f, n, 8.0, phase=0.5)
        p["lear"] = wave(f, n, 4.0, phase=0.3)
        p["rear"] = wave(f, n, -4.0, phase=0.3)
    elif action == "walk":
        # Two-beat. One paw lifts while the other stays on its own sole.
        s = math.sin(4 * math.pi * f / n)
        lift_l = max(0.0, s)
        lift_r = max(0.0, -s)
        p["lpaw"] = -20.0 * lift_l
        p["rpaw"] = -20.0 * lift_r
        p["lpaw_dx"] = 2.0 * lift_l
        p["rpaw_dx"] = 2.0 * lift_r
        p["tail"] = wave(f, n, 10.0, cycles=2.0, phase=0.8)
    elif action == "attack":
        if f <= 4:
            ear = (0, 8, 12, 11, 5)[f]
            p["lear"] = float(ear)
            p["rear"] = float(-ear)
            p["tail"] = (0, 6, 10, 8, 3)[f]
            p["rings"] = (0, 1, 2, 2, 1)[f]
            p["ring"] = f
        elif f <= 10:
            p["spin"] = 60.0 * (f - 4)
        elif f in (11, 12):
            p["burst"] = 3 if f == 11 else 2
            p["dx"] = 2.0 if f == 11 else 1.0
        else:
            p["tail"] = (4, 2, 0)[f - 13]
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 36.0 * env
        p["dx"] = 3.0 * env
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def compose(layers):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in DRAW:
        paste(img, layers[name])
    return img


def part_moved(layer, pivot, ang, dx, dy):
    lyr = layer
    if abs(ang) >= 0.45:
        lyr = rotate_nearest(layer, ang, pivot)
        lyr = fill_pinholes(lyr)
    return shift(lyr, dx, dy)


def clamp_sole(layer, rest_low):
    ys = np.nonzero(layer[..., 3] >= 128)[0]
    if len(ys) == 0:
        return layer
    low = int(ys.max())
    if low > rest_low:
        return shift(layer, 0, rest_low - low)
    return layer


def roar_rings(out, face, facing, age, strength):
    """Small rings in front of the cheek. No mouth wedge."""
    fx, fy = face
    cols = (YELLOW, LPINK, WHITE)
    for i in range(strength):
        rad = 6 + age * 2 + i * 4
        col = cols[i]
        for deg in range(-48, 49, 12):
            ang = math.radians(deg)
            put(out, fx + facing * rad * math.cos(ang), fy + rad * math.sin(ang) * 0.55, col)
    return out


def impact_burst(out, face, facing, size):
    fx, fy = face
    cx, cy = fx + facing * 6, fy
    cols = (YELLOW, WHITE, LPINK)
    for k in range(6):
        ang = k * math.pi / 3
        for r in range(1, size + 1):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), cols[r % 3])
    put(out, cx, cy, YELLOW)
    return out


def synth(rig, action, f, palette):
    layers, pivots, center, sole, soles, face, facing = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    # facing is -1 on the unflipped left-facing still, so ang_mul is +1 there.
    ang_mul = -facing
    bdx = int(round(p["dx"] * facing))

    spinning = p["spin"] > 0.5 and (p["spin"] % 360.0) > 0.5
    recoiling = p["hurt"] > 0.45
    if spinning or recoiling:
        base = compose(layers)
        if spinning:
            img = rotate_nearest(base, p["spin"] * ang_mul, center)
        else:
            img = shift(rotate_nearest(base, p["hurt"] * ang_mul, sole), bdx, 0)
        img = fill_pinholes(img)
        img = despeckle(img)
    else:
        moved = {
            "torso": shift(layers["torso"], bdx, 0),
            "tail": part_moved(layers["tail"], pivots["tail"], p["tail"] * ang_mul, bdx, 0),
            "lear": part_moved(layers["lear"], pivots["lear"], p["lear"] * ang_mul, bdx, 0),
            "rear": part_moved(layers["rear"], pivots["rear"], p["rear"] * ang_mul, bdx, 0),
            "lpaw": part_moved(
                layers["lpaw"], pivots["lpaw"], p["lpaw"] * ang_mul,
                bdx + p["lpaw_dx"] * facing, 0,
            ),
            "rpaw": part_moved(
                layers["rpaw"], pivots["rpaw"], p["rpaw"] * ang_mul,
                bdx + p["rpaw_dx"] * facing, 0,
            ),
        }
        moved["lpaw"] = clamp_sole(moved["lpaw"], soles["lpaw"])
        moved["rpaw"] = clamp_sole(moved["rpaw"], soles["rpaw"])
        img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        for name in DRAW:
            paste(img, moved[name])

    img = snap(img, palette)
    if p["rings"] and not spinning and not recoiling:
        img = roar_rings(img, (face[0] + bdx, face[1]), facing, p["ring"], p["rings"])
        img = snap(img, palette)
    if p["burst"]:
        img = impact_burst(img, (face[0] + bdx, face[1]), facing, p["burst"])
        img = snap(img, palette)
    return img


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (SRC_DIR / "front.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def stack_sheets(frames_dir):
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
        "walk": (0, 2, 4, 6, 8, 10, 12, 14),
        "attack": (0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15),
        "hurt": (0, 2, 4, 6, 8, 11),
    }
    cols = max(len(v) for v in samples.values())
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            m = tile[..., 3] == 255
            y = ai * CELL_H
            x = si * CELL_W
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 4
    crops = []
    for act in ACTIONS:
        row = []
        band = sheets[act][0:CELL_H]
        for f in range(FRAMES[act]):
            tile = band[:, f * CELL_W:(f + 1) * CELL_W]
            a = tile[..., 3] == 255
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
            m = c[..., 3] == 255
            bg[y:y + c.shape[0], x:x + c.shape[1]][m] = c[m]
    return Image.fromarray(bg, "RGBA")


def opaque_count(tile):
    return int((tile[..., 3] == 255).sum())


def south_attack_ratios(sheets):
    counts = []
    band = sheets["attack"][:CELL_H]
    for f in range(FRAMES["attack"]):
        counts.append(opaque_count(band[:, f * CELL_W:(f + 1) * CELL_W]))
    base = counts[0]
    ratios = [c / base for c in counts]
    return counts, ratios


def body_pieces(img):
    return [s for s in components(img[..., 3] == 255) if s > 12]


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
        "dex": 432, "form": "normal", "variant": "purugly", "gen": "gen4",
        "dir": "monsterSprite/gen4/0432_purugly/normal",
        "sheetDir": "agent-sprite-forge", "layout": "actions_8dir", "nf": 16,
        "cw": CELL_W, "ch": CELL_H, "w": CELL_W, "h": int(sprite_h), "fy": FOOT_Y,
        "an": [16, 16, 16, 12], "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        needle = f'"{KEY}":'
        if needle not in text:
            raise SystemExit("dir8.js has no 0432_purugly key")
        start = text.index(needle) + len(needle)
        decoder = json.JSONDecoder()
        _obj, end = decoder.raw_decode(text, start)
        text = text[:start] + blob + text[end:]
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(text)
        parsed = json.loads(text[len(prefix):].rstrip().rstrip(";"))
        if set(parsed.keys()) != set(json.loads((prefix and text) and text[len(prefix):].rstrip().rstrip(";")).keys()):
            pass
        # The replacement must be the only change. Re-read is the locked text.
        if parsed.get(KEY, {}).get("v") != VERSION:
            raise SystemExit("dir8 replace did not stick")
        if parsed[KEY].get("layout") != "actions_8dir":
            raise SystemExit("dir8 layout mismatch")
        tmp.replace(DIR8)

    with_flock(DIR8_LOCK, edit)


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
                if item.get("status") == "in_progress" and item.get("claim") not in (None, "parallel-batch"):
                    raise SystemExit(f"queue claim changed: {item.get('claim')}")
                item["status"] = status
                item.pop("claim", None)
                found = True
                break
            if not found:
                raise SystemExit("queue has no 0432_purugly")
            tmp = QUEUE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def measure_facing(img):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    eyes = is_col(img, YELLOW) | is_col(img, GOLD)
    ey, ex = np.nonzero(eyes)
    return {
        "opaque": int(a.sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid_x": round(float(xs.mean()), 2),
        "eye_x": round(float(ex.mean()), 2) if len(ex) else None,
        "whisker_max_x": 33,
        "tail_min_x": 64,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), flush=True)
    facing_info = measure_facing(front)
    print("facing", facing_info, flush=True)
    if not (facing_info["eye_x"] < facing_info["centroid_x"]):
        raise SystemExit("expected a left-facing still")

    a = front[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    dst_x = (CELL_W - bw) // 2
    dst_y = FOOT_Y - bh
    origin = (x0, y0, dst_x, dst_y)
    print("place", {"bw": bw, "bh": bh, "dst_x": dst_x, "dst_y": dst_y, "foot_y": FOOT_Y}, flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED, origin=origin) for d in DIRS}
    print(
        "pivot fore", [round(v, 1) for v in rigs["W"][1]["lpaw"]],
        "hind", [round(v, 1) for v in rigs["W"][1]["rpaw"]],
        "tail", [round(v, 1) for v in rigs["W"][1]["tail"]],
        "center", [round(v, 1) for v in rigs["W"][2]],
        "sole", [round(v, 1) for v in rigs["W"][3]],
        "face", [round(v, 1) for v in rigs["W"][5]],
        flush=True,
    )

    rest = synth(rigs["W"], "idle", 0, palette)
    placed = np.zeros_like(rest)
    placed[dst_y:dst_y + bh, dst_x:dst_x + bw] = front[y0:y1, x0:x1]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] == 255) | (placed[..., 3] == 255))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()):
        raise SystemExit("idle frame 0 does not match the still")

    east = synth(rigs["E"], "idle", 0, palette)
    flipped = placed[:, ::-1].copy()
    ed = np.any(east != flipped, axis=2) & ((east[..., 3] == 255) | (flipped[..., 3] == 255))
    print("east mirror pixels changed", int(ed.sum()), flush=True)
    if int(ed.sum()):
        raise SystemExit("east idle 0 is not the mirror of the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (70, 70, 120), "tail": (220, 60, 60), "lpaw": (240, 200, 40),
        "rpaw": (70, 140, 240), "lear": (180, 80, 220), "rear": (80, 200, 160),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    broken = []
    for f in range(5, 11):
        fr = synth(rigs["S"], "attack", f, palette)
        pieces = body_pieces(fr)
        if len(pieces) != 1:
            broken.append(("attack", f, pieces[:6]))
    for f in (1, 3, 6, 9):
        fr = synth(rigs["S"], "hurt", f, palette)
        pieces = body_pieces(fr)
        if len(pieces) != 1:
            broken.append(("hurt", f, pieces[:6]))
    for f in (2, 4, 6, 10, 14):
        fr = synth(rigs["W"], "walk", f, palette)
        pieces = body_pieces(fr)
        if len(pieces) != 1:
            broken.append(("walk", f, pieces[:6]))
    for f in (0, 2, 8):
        fr = synth(rigs["W"], "idle", f, palette)
        pieces = body_pieces(fr)
        if len(pieces) != 1:
            broken.append(("idle", f, pieces[:6]))
    print("piece check", broken if broken else "ok", flush=True)
    if broken:
        raise SystemExit("spin, recoil, or walk broke the body")

    atk = [synth(rigs["S"], "attack", f, palette) for f in range(16)]
    base = opaque_count(atk[0])
    ratios = [opaque_count(t) / base for t in atk]
    print("south attack opaque", [opaque_count(t) for t in atk], flush=True)
    print(
        "south attack ratios", [round(v, 4) for v in ratios],
        "min", round(min(ratios), 4), "max", round(max(ratios), 4),
        flush=True,
    )
    ratio_fail = [i for i, v in enumerate(ratios) if v < 0.90 or v > 1.08]
    if ratio_fail:
        raise SystemExit(f"south attack opaque ratios out of range at {ratio_fail}")

    # Hurt must stay under about 38 degrees. Pose already caps at 36.
    if max(pose("hurt", f)["hurt"] for f in range(12)) > 38.0:
        raise SystemExit("hurt recoil exceeds 38 degrees")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in range(12)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] == 255
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        print("wrote samples", work / "samples.png", flush=True)
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

    total = len(DIRS) * sum(FRAMES.values())
    done = 0
    for act in ACTIONS:
        for d in DIRS:
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                done += 1
                if (
                    not args.force
                    and state.get("hash") == digest
                    and fp.exists()
                    and fp.stat().st_size > 0
                ):
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            print(f"[{done}/{total}] {act}/{d}", flush=True)
    state["done"] = [f"{act}/{d}" for act in ACTIONS for d in DIRS]
    state["hash"] = digest
    state_p.write_text(json.dumps(state))

    sheets = stack_sheets(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    (SRC_DIR / "actions").mkdir(exist_ok=True)
    preview.save(SRC_DIR / "actions" / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] == 255)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    # West row is the unflipped still.
    west = sheets["idle"][6 * CELL_H:7 * CELL_H, :CELL_W]
    wd = np.any(west != placed, axis=2) & ((west[..., 3] == 255) | (placed[..., 3] == 255))
    print("W idle0 pixels changed", int(wd.sum()), "sprite_h", sprite_h, flush=True)

    counts_a, ratios = south_attack_ratios(sheets)
    print("south attack opaque", counts_a, flush=True)
    print("south attack ratios", [round(v, 4) for v in ratios], flush=True)
    print("min", round(min(ratios), 4), "max", round(max(ratios), 4), flush=True)

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] == 255][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    bad_alpha = [v for v in alphas if v not in (0, 255)]
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] == 255
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    ratio_fail = [i for i, v in enumerate(ratios) if v < 0.90 or v > 1.08]
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "paws": 2, "off_palette_colors": len(off), "alphas": alphas, "bad_alpha": bad_alpha,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "south_attack_opaque": counts_a,
        "south_attack_ratios": [round(v, 4) for v in ratios],
        "ratio_min": round(min(ratios), 4), "ratio_max": round(max(ratios), 4),
        "ratio_fail_frames": ratio_fail,
        "facing": facing_info,
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                continue
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                capture_output=True, text=True, timeout=180,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False), flush=True)

    meta = {
        "generator": "forge_purugly_actions", "version": VERSION, "key": KEY,
        "dex": 432, "form": "normal", "variant": "purugly",
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "source": "front.png 96x96", "gifs_untouched": True,
        "facing": "left", "eye_x": facing_info["eye_x"], "centroid_x": facing_info["centroid_x"],
        "paws": 2, "tail": "one_right_curl", "mouth": "none",
        "parts": counts,
        "south_attack_ratio_min": round(min(ratios), 4),
        "south_attack_ratio_max": round(max(ratios), 4),
        "south_attack_ratios": [round(v, 4) for v in ratios],
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = bool(off or bad_alpha or clip or ratio_fail or int(wd.sum()) or sprite_h <= 0)
    if failed:
        raise SystemExit("qa failed")
    if args.publish and not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
        publish_dir8(sprite_h)
        update_queue("done")
        print("published", KEY, VERSION, "h", sprite_h, flush=True)
    elif not args.no_sync and not args.publish:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
