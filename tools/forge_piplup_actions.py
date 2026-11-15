#!/usr/bin/env python3
"""Piplup (0393) 8-direction action sheet from this form's 96x96 front.png.

Measured on front.png (bbox x33-59, y28-62, 27x35, 606 opaque, 15 colours):
beak tip x=35 is left of both pupils (44,40) and (46,40), so the still faces
left. Unflipped dirs are N NW W SW. Flipped dirs are S SE E NE (E faces right).

Archetype: biped penguin. One raised cheek flipper (viewer's left), one side
flipper (viewer's right), two webbed feet, one yellow beak. No wings, no tail,
no extra flippers. The beak opens; the face otherwise stays on integer shifts.

  idle   : 1px breath, flippers sway, a short beak chirp. Frame 0 is the still.
  walk   : waddle. Body rocks, feet lift in turn, flippers swing. Not a glide.
  attack : 0-4 beak roar plus rings; 5-10 in-place 360 of the whole body;
           11-12 impact; 13-15 rest.
  hurt   : one-piece recoil, 38 degrees max, no belly flip, no |cos| squash.

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_piplup_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen4/0393_piplup/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0393_piplup/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left (beak-tip x < eye x). Mirror the right-facing half.
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "piplup_v2"

# Colours measured on this front.png. Effects use only these.
BLACK = (16, 16, 16)
BROWN = (98, 74, 16)
YELLOW = (246, 205, 65)
WHITE = (255, 255, 255)
LIGHT = (156, 213, 246)
CREAM = (255, 238, 197)
GOLD = (172, 131, 57)
GOLD_SET = {GOLD, BROWN, YELLOW, CREAM}

DRAW = ["torso", "head", "flip_l", "flip_r", "foot_l", "foot_r", "jaw"]


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


def segment(img):
    """Partition this still. Cuts are the measured beak, feet, and two flippers."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    rgb = img[..., :3]
    yy, xx = np.mgrid[0:h, 0:w]
    gold = np.zeros_like(a)
    for c in GOLD_SET:
        gold |= a & (rgb[..., 0] == c[0]) & (rgb[..., 1] == c[1]) & (rgb[..., 2] == c[2])
    black = a & (rgb[..., 0] == 16) & (rgb[..., 1] == 16) & (rgb[..., 2] == 16)

    # Left webbed foot sits under the belly; right foot is the lower yellow toes.
    foot_l = a & (yy >= 58) & (xx <= 46)
    foot_r = ((gold & (yy >= 59) & (xx >= 47)) | (black & (yy >= 60) & (xx >= 47) & (xx <= 53)))

    # Yellow beak on the left of the face. Bottom two rows are the jaw.
    beak = (gold & (yy <= 50)) | (black & (xx == 36) & (yy == 45))
    jaw = beak & (yy >= 44)
    beak_upper = beak & ~jaw

    # Raised cheek flipper: the pale stub at x<=35, y36-44. Not the beak.
    flip_l = a & (xx <= 35) & (yy >= 36) & (yy <= 44) & ~beak
    # Side flipper: the right-pointing lobe whose tip is (59, 50).
    allow = a & (xx >= 54) & (yy >= 47) & (yy <= 55) & ~foot_r
    seed = np.zeros_like(a)
    if a[50, 59]:
        seed[50, 59] = True
    flip_r = flood(seed, allow)

    head = a & (yy <= 45) & ~flip_l & ~jaw & ~foot_l & ~foot_r & ~flip_r
    head = head | beak_upper
    used = foot_l | foot_r | jaw | flip_l | flip_r | head
    torso = a & ~used
    parts = {
        "torso": torso, "head": head, "flip_l": flip_l, "flip_r": flip_r,
        "foot_l": foot_l, "foot_r": foot_r, "jaw": jaw,
    }
    cover = np.zeros_like(a)
    for m in parts.values():
        if np.any(cover & m):
            raise SystemExit("piplup masks overlap")
        cover |= m
    if int((cover != a).sum()):
        raise SystemExit(f"piplup masks missed {(cover != a).sum()} pixels")
    return parts


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


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
    """4x nearest rotation about pivot. Exact colours. No scale, no mirror."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
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
    return out[1::4, 1::4].copy()


def despeckle(img):
    """Drop only pixels whose four neighbours are empty. Keeps eyes and outlines."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
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


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


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


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def contact_pivot(part, body):
    """Contact pixel of part closest to the torso."""
    ys, xs = np.nonzero(part)
    if len(ys) == 0:
        return 0.0, 0.0
    touch = part & dilate(body, 1)
    if np.any(touch):
        ys, xs = np.nonzero(touch)
    by, bx = np.nonzero(body)
    bcy, bcx = float(by.mean()), float(bx.mean())
    d = (ys.astype(np.float32) - bcy) ** 2 + (xs.astype(np.float32) - bcx) ** 2
    i = int(d.argmin())
    return float(xs[i]), float(ys[i])


def add_socket(host, host_mask, part_mask, canvas):
    zone = part_mask & dilate(host_mask, 2)
    if not np.any(zone):
        return
    fill = nearest_color(canvas, zone, host_mask)
    write = zone & (host[..., 3] < 128)
    host[write] = fill[write]


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

    # Joint colour lives under the limb, hidden until the limb moves.
    add_socket(layers["head"], placed["head"], placed["flip_l"], canvas)
    add_socket(layers["torso"], placed["torso"], placed["flip_r"], canvas)
    add_socket(layers["torso"], placed["torso"], placed["foot_l"], canvas)
    add_socket(layers["torso"], placed["torso"], placed["foot_r"], canvas)

    pivots = {
        "flip_l": contact_pivot(placed["flip_l"], placed["head"] | placed["torso"]),
        "flip_r": contact_pivot(placed["flip_r"], placed["torso"]),
        "foot_l": contact_pivot(placed["foot_l"], placed["torso"]),
        "foot_r": contact_pivot(placed["foot_r"], placed["torso"]),
    }
    fys, fxs = np.nonzero(placed["foot_l"] | placed["foot_r"])
    sole = (float(np.median(fxs)), float(fys.max()))
    opy, opx = np.nonzero(canvas[..., 3] >= 128)
    center = (float(opx.mean()), float(opy.mean()))

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        sole = (CELL_W - 1 - sole[0], sole[1])
        center = (CELL_W - 1 - center[0], center[1])
    facing = -1.0 if not flip else 1.0
    return layers, pivots, sole, center, facing


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(dx=0.0, dy=0.0, flip_l=0.0, flip_r=0.0, lift_l=0.0, lift_r=0.0,
             jaw=0, spin=0.0, recoil=0.0, rings=0, ring=0, burst=0, flash=0, sparks=0)
    if action == "idle":
        p["dy"] = wave(f, n, 1.0)
        p["flip_l"] = wave(f, n, 8.0)
        p["flip_r"] = wave(f, n, -7.0, phase=-0.5)
        if f in (8, 9, 10):
            p["jaw"] = 1
    elif action == "walk":
        step = wave(f, n, 1.0, cycles=2.0)
        p["dx"] = step
        p["dy"] = wave(f, n, 1.0, cycles=2.0)
        p["lift_l"] = max(0.0, step) * 2.0
        p["lift_r"] = max(0.0, -step) * 2.0
        p["flip_l"] = wave(f, n, 14.0, cycles=2.0)
        p["flip_r"] = wave(f, n, -14.0, cycles=2.0)
    elif action == "attack":
        # 0-4 roar, 5-10 in-place spin, 11-12 impact, 13-15 rest.
        if f <= 4:
            p["jaw"] = (2, 3, 3, 3, 2)[f]
            p["dy"] = (0, -1, -1, -1, 0)[f]
            p["flip_l"] = (8, 14, 16, 14, 8)[f]
            p["flip_r"] = (-10, -16, -18, -14, -8)[f]
            p["rings"] = 3
            p["ring"] = f
        elif f <= 10:
            p["spin"] = 60.0 * (f - 4)
        elif f in (11, 12):
            p["burst"] = 1
    else:
        env = math.sin(math.pi * f / 11.0)
        p["recoil"] = 38.0 * env
        p["dx"] = 3.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def gold_of(layer):
    a = layer[..., 3] >= 128
    rgb = layer[..., :3]
    g = np.zeros(a.shape, bool)
    for c in GOLD_SET:
        g |= a & (rgb[..., 0] == c[0]) & (rgb[..., 1] == c[1]) & (rgb[..., 2] == c[2])
    return g


def beak_tip(layers_moved, facing):
    g = np.zeros(layers_moved[0].shape[:2], bool)
    for lyr in layers_moved:
        g |= gold_of(lyr)
    ys, xs = np.nonzero(g)
    if len(xs) == 0:
        return None
    # Feet are gold too. The beak is the upper cluster.
    top = ys.min()
    keep = ys <= top + 10
    ys, xs = ys[keep], xs[keep]
    if facing > 0:
        x = int(xs.max())
        y = int(np.median(ys[xs >= x - 1]))
    else:
        x = int(xs.min())
        y = int(np.median(ys[xs <= x + 1]))
    return x, y


def open_beak(img, jaw_src, bdx, bdy, drop, reach, facing):
    """Drop the existing lower beak a pixel or two and grow a wedge off the tip."""
    if drop <= 0 and reach <= 0:
        return img
    home = shift(jaw_src, bdx, bdy)
    dropped = shift(jaw_src, bdx, bdy + max(0, drop))
    hole = (home[..., 3] >= 128) & (dropped[..., 3] < 128) & (img[..., 3] < 128)
    img[hole, :3] = BLACK
    img[hole, 3] = 255
    paste(img, dropped)
    g = gold_of(home) | gold_of(dropped)
    ys, xs = np.nonzero(g)
    if len(xs) == 0 or reach <= 0:
        return img
    if facing > 0:
        x = int(xs.max())
        y = int(np.median(ys[xs >= x - 1]))
        step = 1
    else:
        x = int(xs.min())
        y = int(np.median(ys[xs <= x + 1]))
        step = -1
    for i in range(1, reach + 1):
        put(img, x + step * i, y, BLACK)
        put(img, x + step * i, y + 1, BROWN)
    put(img, x + step * (reach + 1), y, YELLOW)
    return img


def roar_rings(out, mouth, facing, age, strength):
    mx, my = mouth
    cols = (WHITE, LIGHT, YELLOW)
    for r_i in range(strength):
        rad = 6 + age * 3 + r_i * 5
        col = cols[min(r_i, 2)]
        for deg in range(-50, 51, 8):
            ang = math.radians(deg)
            put(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.8, col)
    return out


def impact_burst(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(max(2, size // 2), size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), WHITE if r < size - 2 else YELLOW)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, WHITE)
    return out


def sparks(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    cx = (x0 + x1) // 2
    d = frame_idx
    pts = ((cx - facing * (4 + d), y0 - 2), (cx + facing * (3 + d), y0 - 4),
           (cx - facing * 2, y0 - 3 - d // 2))
    for sx, sy in pts:
        put(out, sx, sy, WHITE)
        put(out, sx + facing, sy + 1, YELLOW)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def moved_parts(layers, pivots, p, facing):
    bdx, bdy = p["dx"] * facing, p["dy"]
    # Facing -1 (left): recoil dx is already signed by the caller via facing.
    out = {}
    out["torso"] = shift(layers["torso"], bdx, bdy)
    out["head"] = shift(layers["head"], bdx, bdy)
    sgn = -1.0 if facing < 0 else 1.0
    out["flip_l"] = shift(rotate_nearest(layers["flip_l"], p["flip_l"] * sgn, pivots["flip_l"]), bdx, bdy)
    out["flip_r"] = shift(rotate_nearest(layers["flip_r"], p["flip_r"] * sgn, pivots["flip_r"]), bdx, bdy)
    # Feet lift straight up so the tiny webs keep their pixels.
    out["foot_l"] = shift(layers["foot_l"], bdx, bdy - p["lift_l"])
    out["foot_r"] = shift(layers["foot_r"], bdx, bdy - p["lift_r"])
    out["jaw"] = shift(layers["jaw"], bdx, bdy)
    return out, bdx, bdy


def assemble(moved):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in DRAW:
        paste(img, moved[name])
    return img


def synth(rig, action, f, palette):
    layers, pivots, sole, center, facing = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    # Hurt stores a backward magnitude. Walk sway is mirrored with the facing.
    if action == "hurt":
        knock = p["dx"]
        p["dx"] = 0.0
        bdx_hurt = -facing * knock
    else:
        bdx_hurt = 0.0
    moved, bdx, bdy = moved_parts(layers, pivots, p, facing)

    spinning = p["spin"] > 0.5 and (p["spin"] % 360.0) > 0.5
    recoiling = p["recoil"] > 0.45
    if spinning or recoiling:
        # One rigid bitmap. A squash roll would tear this 27px penguin.
        base = assemble({k: layers[k] for k in DRAW})
        if spinning:
            img = rotate_nearest(base, p["spin"], center)
        else:
            # Positive angle swings the head screen-right, so backward is -facing.
            img = shift(rotate_nearest(base, -facing * p["recoil"], sole), bdx_hurt, 0)
        img = despeckle(img)
    else:
        img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        for name in ("torso", "head", "flip_l", "flip_r", "foot_l", "foot_r"):
            paste(img, moved[name])
        if p["jaw"]:
            img = open_beak(img, layers["jaw"], bdx, bdy, min(2, int(p["jaw"])), int(p["jaw"]), facing)
        else:
            paste(img, moved["jaw"])

    img = snap(img, palette)
    if p["flash"]:
        img = flash_white(img)
    if p["rings"]:
        tip_layers = (moved["head"], moved["jaw"])
        tip = beak_tip(tip_layers, facing)
        if tip:
            img = roar_rings(img, tip, facing, p["ring"], p["rings"])
    if p["burst"]:
        # Beak of the settled body.
        tip = beak_tip((img,), facing)
        if tip is None:
            bb = body_box(img)
            tip = ((bb[2] if facing > 0 else bb[0]), (bb[1] + bb[3]) // 2) if bb else None
        if tip:
            reach = 8 if f >= 12 else 6
            img = impact_burst(img, (tip[0] + facing * 6, tip[1]), reach)
    if p["sparks"]:
        img = sparks(img, f, facing)
    return snap(img, palette)


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
    """South samples on a dark field, plus the unflipped west row so the beak side is readable."""
    samples = {"idle": (0, 4, 8, 11), "walk": (0, 2, 4, 6, 8, 10, 12, 14),
               "attack": (0, 2, 4, 5, 7, 9, 11, 13), "hurt": (0, 2, 4, 6, 8, 11)}
    cols = 8
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    west = 6  # W row
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][west * CELL_H:(west + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            m = tile[..., 3] >= 128
            bg[ai * CELL_H:(ai + 1) * CELL_H, si * CELL_W:(si + 1) * CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 6
    crops = []
    for act in ACTIONS:
        row = []
        band = sheets[act][6 * CELL_H:7 * CELL_H]
        for f in range(FRAMES[act]):
            tile = band[:, f * CELL_W:(f + 1) * CELL_W]
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


def zoom_sheet(sheets, scale=4):
    """All west frames, cropped tight and scaled, so the 27px penguin is readable."""
    pad = 4
    # Fixed window around the placed still, wide enough for rings and a 38 degree recoil.
    x0, x1 = 28, 150
    y0, y1 = 78, 166
    cw, ch = (x1 - x0) * scale, (y1 - y0) * scale
    cols = 8
    rows = 0
    blocks = []
    for act in ACTIONS:
        band = sheets[act][6 * CELL_H:7 * CELL_H]
        frames = []
        for f in range(FRAMES[act]):
            tile = band[y0:y1, f * CELL_W + x0:f * CELL_W + x1]
            up = np.repeat(np.repeat(tile, scale, 0), scale, 1)
            frames.append(up)
        blocks.append(frames)
        rows += math.ceil(len(frames) / cols)
    bg = np.full((rows * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    r = 0
    for frames in blocks:
        for i, up in enumerate(frames):
            rr, cc = divmod(i, cols)
            y = (r + rr) * ch
            x = cc * cw
            m = up[..., 3] >= 128
            bg[y:y + ch, x:x + cw][m] = up[m]
        r += math.ceil(len(frames) / cols)
    return Image.fromarray(bg, "RGBA")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), flush=True)
    for name in ("flip_l", "flip_r", "foot_l", "foot_r", "jaw", "head", "torso"):
        if counts[name] < 4:
            raise SystemExit(f"part {name} too small: {counts[name]}")

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}

    rest = synth(rigs["W"], "idle", 0, palette)
    a = front[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    placed = np.zeros_like(rest)
    dst_x = (CELL_W - (x1 - x0)) // 2
    dst_y = FOOT_Y - (y1 - y0)
    placed[dst_y:dst_y + (y1 - y0), dst_x:dst_x + (x1 - x0)] = front[y0:y1, x0:x1]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    n_diff = int(differ.sum())
    print("rest-pose pixels changed", n_diff, flush=True)
    if n_diff:
        raise SystemExit("idle frame 0 does not match the still")

    # East is the mirror. It must match a horizontal flip of the placed still.
    east = synth(rigs["E"], "idle", 0, palette)
    flipped = placed[:, ::-1].copy()
    ed = np.any(east != flipped, axis=2) & ((east[..., 3] >= 128) | (flipped[..., 3] >= 128))
    print("east mirror pixels changed", int(ed.sum()), flush=True)
    if int(ed.sum()):
        raise SystemExit("east idle 0 is not the mirror of the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {"torso": (80, 140, 200), "head": (240, 200, 40), "flip_l": (40, 200, 80),
                 "flip_r": (220, 60, 60), "foot_l": (255, 140, 40), "foot_r": (180, 80, 220),
                 "jaw": (40, 220, 220)}
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 6, 0), 6, 1), "RGBA").save(work / "parts.png")

    # One-piece check on the spin and the recoil before writing the sheet.
    broken = []
    for f in list(range(5, 11)) + [13, 15]:
        fr = synth(rigs["W"], "attack", f, palette)
        sizes = components(fr[..., 3] >= 128)
        body = [s for s in sizes if s > 8]
        if len(body) != 1:
            broken.append(("attack", f, sizes[:6]))
    for f in (0, 3, 6, 9, 11):
        fr = synth(rigs["W"], "hurt", f, palette)
        sizes = components(fr[..., 3] >= 128)
        body = [s for s in sizes if s > 8]
        if len(body) != 1:
            broken.append(("hurt", f, sizes[:6]))
    print("piece check", broken if broken else "ok", flush=True)
    if broken:
        raise SystemExit("spin or recoil broke the body")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 10)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 10, 14)]
        want += [("attack", "W", f) for f in (0, 2, 4, 5, 6, 7, 8, 9, 11, 13)]
        want += [("attack", "E", f) for f in (2, 7)]
        want += [("hurt", "W", f) for f in (0, 1, 3, 6, 9, 11)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        # Tight 5x crop of each sample so the beak and feet are readable.
        zooms = []
        for tile in tiles:
            crop = tile[90:162, 40:140]
            zooms.append(np.repeat(np.repeat(crop, 5, 0), 5, 1))
        zw, zh = zooms[0].shape[1], zooms[0].shape[0]
        zrows = math.ceil(len(zooms) / cols)
        zsh = np.full((zrows * zh, cols * zw, 4), (15, 20, 31, 255), np.uint8)
        for i, z in enumerate(zooms):
            r, c = divmod(i, cols)
            m = z[..., 3] >= 128
            zsh[r * zh:(r + 1) * zh, c * zw:(c + 1) * zw][m] = z[m]
        Image.fromarray(zsh, "RGBA").save(work / "samples_zoom.png")
        print("samples", work / "samples_zoom.png", flush=True)
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

    sheets = stack_sheets(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    (SRC_DIR / "actions").mkdir(exist_ok=True)
    preview.save(SRC_DIR / "actions" / "preview.png")
    contact = motion_contact(sheets)
    contact.save(OUT_DIR / "preview_motion.png")
    contact.save(work / "review_W.png")
    zoom_sheet(sheets, 4).save(work / "preview_zoom.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = set(int(v) for v in np.unique(master[..., 3]))
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    report = {"version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
              "off_palette_colors": len(off), "alphas": sorted(alphas),
              "edge_touching": {k: len(v) for k, v in clip.items()}}
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run([py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                            "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                           capture_output=True, text=True)
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False), flush=True)

    meta = {"generator": "forge_piplup_actions", "version": VERSION, "key": "0393_piplup",
            "dex": 393, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
            "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
            "unflipped": sorted(UNFLIPPED), "palette_colors": int(len(palette)),
            "off_palette_colors": len(off), "source": "front.png 96x96", "gifs_untouched": True,
            "facing": "left", "beak_tip_x": 35, "eye_x": [44, 46]}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if off or clip or alphas - {0, 255} or sprite_h <= 0:
        raise SystemExit("qa failed")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
