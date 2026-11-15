#!/usr/bin/env python3
"""Empoleon (0395) 8-direction action sheet from this form's 96x96 front.png.

Measured on normal/front.png (alpha already 0 or 255, 14 colours, 2521 opaque):
  bbox x10-78 y9-85 (69x77). Centroid (47.25, 52.72), max radius 43.8.
  Pale eye shines at (44, 26) and (51, 26). The beak is the central gold
  point between them, tip about (46-48, 31-33), pointing down. It is not a
  side-profile bill. The raised flipper reaches screen-left (to x=10) and the
  other flipper is tucked on screen-right, so the still presents left.
  Unflipped dirs are N NW W SW. Flipped dirs are S SE E NE (E faces right).

  Crown: three gold prongs only (left, tall centre, right). They stay on the
  head. No fourth prong is drawn.
  Flippers: the left wing (light edge, dark inner half, three gold claw marks)
  and the right fin. No third flipper.
  Feet: two, each with the three toes already in the still.
  The beak opens by dropping that existing tip. The crown does not stretch.

Archetype: biped emperor penguin. Walk is a waddle, not a glide.

  idle   : 1px breath, flippers sway, a short beak chirp. Frame 0 is the still.
  walk   : body rock, feet lift in turn, flippers counter-swing. Two steps.
  attack : 0-4 beak roar plus rings; 5-10 in-place 360 of the whole body;
           11-12 impact; 13-15 rest. No |cos| squash.
  hurt   : one-piece recoil, 38 degrees max, no belly flip.

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_empoleon_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen4/0395_empoleon/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0395_empoleon/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still presents left (open flipper to screen-left). East-side dirs are mirrored.
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 192, 184, 164
VERSION = "empoleon_v2"

NAVY = (16, 32, 65)
DBLUE = (16, 57, 115)
MBLUE = (82, 139, 230)
BLACK = (16, 16, 16)
WHITE = (255, 255, 255)
BLUE = (41, 90, 148)
TAN = (189, 139, 106)
LBLUE = (156, 205, 255)
GOLD = (238, 205, 98)
LGOLD = (255, 230, 131)
CBLUE = (115, 164, 238)
PALE = (222, 238, 255)
SLATE = (74, 115, 172)
DTAN = (123, 82, 65)

METAL = (GOLD, LGOLD, TAN, DTAN)
DRAW = ["torso", "flip_r", "flip_l", "foot_l", "foot_r", "jaw"]


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


def is_color(img, mask, color):
    rgb = img[..., :3]
    return mask & (rgb[..., 0] == color[0]) & (rgb[..., 1] == color[1]) & (rgb[..., 2] == color[2])


def segment(img):
    """Partition this still. Cuts follow the measured wing seam, fins, beak, and feet."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    black = is_color(img, a, BLACK)
    white = is_color(img, a, WHITE)
    dblue = is_color(img, a, DBLUE)
    navy = is_color(img, a, NAVY)
    metal = np.zeros_like(a)
    for c in METAL:
        metal |= is_color(img, a, c)

    # Two feet. Metal toes plus the black outline that touches them. Not the legs.
    foot_l = flood(metal & (yy >= 74) & (xx <= 46), (metal | black) & (yy >= 73) & (xx <= 46) & (xx >= 26))
    foot_r = flood(metal & (yy >= 74) & (xx >= 48), (metal | black) & (yy >= 75) & (xx >= 49) & (xx <= 66))

    # Lower tip of the centre crest only. Side prongs stay on the head.
    jaw = metal & (yy >= 29) & (yy <= 33) & (xx >= 44) & (xx <= 50)
    jaw = jaw | (black & dilate(jaw, 1) & (yy >= 31) & (yy <= 34) & (xx >= 45) & (xx <= 48))

    # Left wing: pixels left of the black belly-outline. The seam is the first
    # black walking left from the white chest through the dark torso margin.
    seam = {}
    for y in range(h):
        wxs = np.nonzero(white[y])[0]
        if len(wxs) == 0:
            continue
        x = int(wxs.min()) - 1
        seen_black = None
        while x >= 0 and a[y, x] and (black[y, x] or dblue[y, x] or navy[y, x] or white[y, x]):
            if black[y, x]:
                seen_black = x
                break
            x -= 1
        if seen_black is not None and seen_black >= 20:
            seam[y] = seen_black
    if seam:
        known = sorted(seam)
        for y in range(known[0], known[-1] + 1):
            if y in seam:
                continue
            prev = max(k for k in known if k < y)
            nxts = [k for k in known if k > y]
            if not nxts:
                seam[y] = seam[prev]
            else:
                nxt = nxts[0]
                t = (y - prev) / (nxt - prev)
                seam[y] = int(round(seam[prev] * (1 - t) + seam[nxt] * t))
        last = max(seam)
        sx = seam[last]
        for y in range(last + 1, 70):
            xs = np.nonzero(a[y])[0]
            if len(xs) == 0 or int(xs.min()) > sx - 8:
                break
            seam[y] = sx
    # Wing top edge, measured: y=40 flares to x=24 and the outline sits at x=30.
    seam[40] = 31

    flip_l = np.zeros_like(a)
    for y, sx in seam.items():
        xs = np.nonzero(a[y])[0]
        if len(xs) == 0 or int(xs.min()) > sx - 5:
            continue
        flip_l[y, :sx] = a[y, :sx]
    flip_l &= ~foot_l & ~foot_r & ~jaw

    # Right fin: the light lobe outside the dark torso, one connected piece.
    light = (
        is_color(img, a, LBLUE)
        | is_color(img, a, CBLUE)
        | is_color(img, a, PALE)
        | is_color(img, a, SLATE)
        | is_color(img, a, MBLUE)
        | is_color(img, a, BLUE)
    )
    allow_r = a & (yy >= 46) & (yy <= 77) & (xx >= 63) & ~foot_r & ~white & ~dblue
    allow_r = allow_r | (
        a & (yy >= 46) & (yy <= 77) & (xx >= 68) & ~foot_r & (black | navy | light | is_color(img, a, BLUE))
    )
    seed_r = np.zeros_like(a)
    for y, x in ((47, 74), (50, 74), (55, 74), (60, 74), (68, 78)):
        if 0 <= y < h and 0 <= x < w and a[y, x]:
            seed_r[y, x] = True
    if not np.any(seed_r):
        ys, xs = np.nonzero(a)
        i = int(np.argmax(xs))
        seed_r[int(ys[i]), int(xs[i])] = True
    flip_r = flood(seed_r, allow_r) & ~foot_r & ~flip_l & ~jaw

    torso = a & ~(flip_l | flip_r | foot_l | foot_r | jaw)
    parts = {
        "torso": torso,
        "flip_l": flip_l,
        "flip_r": flip_r,
        "foot_l": foot_l,
        "foot_r": foot_r,
        "jaw": jaw,
    }
    cover = np.zeros_like(a)
    for name, m in parts.items():
        if np.any(cover & m):
            raise SystemExit(f"empoleon masks overlap at {name}")
        cover |= m
    if int((cover != a).sum()):
        raise SystemExit(f"empoleon masks missed {int((cover != a).sum())} pixels")
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


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest rotation about pivot. Exact colours. No scale and no mirror."""
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
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def shoulder_pivot(part, body):
    """Top contact pixel, so a flipper swings from the shoulder rather than the hip."""
    ys, xs = np.nonzero(part)
    if len(ys) == 0:
        return 0.0, 0.0
    touch = part & dilate(body, 1)
    if np.any(touch):
        ys, xs = np.nonzero(touch)
    top = int(ys.min())
    keep = ys <= top + 1
    xs, ys = xs[keep], ys[keep]
    # Prefer the contact nearest the body horizontally: the inner end of the top edge.
    by, bx = np.nonzero(body)
    bcx = float(bx.mean())
    i = int(np.argmin(np.abs(xs.astype(np.float32) - bcx)))
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

    body = placed["torso"]
    add_socket(layers["torso"], body, placed["flip_l"], canvas)
    add_socket(layers["torso"], body, placed["flip_r"], canvas)
    add_socket(layers["torso"], body, placed["foot_l"], canvas)
    add_socket(layers["torso"], body, placed["foot_r"], canvas)
    add_socket(layers["torso"], body, placed["jaw"], canvas)

    pivots = {
        "flip_l": shoulder_pivot(placed["flip_l"], body),
        "flip_r": shoulder_pivot(placed["flip_r"], body),
    }
    soles = []
    for name in ("foot_l", "foot_r"):
        fys, fxs = np.nonzero(placed[name])
        if len(fys) == 0:
            continue
        yb = int(fys.max())
        xb = float(np.median(fxs[fys >= yb - 1]))
        soles.append((xb, float(yb)))
    if soles:
        sole = (sum(p[0] for p in soles) / len(soles), max(p[1] for p in soles))
    else:
        sole = (CELL_W / 2, float(FOOT_Y))
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
    p = dict(
        dx=0.0, dy=0.0, flip_l=0.0, flip_r=0.0, lift_l=0.0, lift_r=0.0,
        jaw=0, spin=0.0, recoil=0.0, rings=0, ring=0, burst=0, flash=0, sparks=0,
    )
    if action == "idle":
        p["dy"] = wave(f, n, 1.0)
        p["flip_l"] = wave(f, n, 8.0, phase=0.4)
        p["flip_r"] = wave(f, n, -7.0, phase=-0.6)
        if f in (8, 9, 10):
            p["jaw"] = 1
    elif action == "walk":
        step = wave(f, n, 1.0, cycles=2.0)
        p["dx"] = step
        p["dy"] = wave(f, n, 1.0, cycles=2.0)
        p["lift_l"] = max(0.0, step) * 3.0
        p["lift_r"] = max(0.0, -step) * 3.0
        p["flip_l"] = wave(f, n, 12.0, cycles=2.0)
        p["flip_r"] = wave(f, n, -11.0, cycles=2.0, phase=math.pi)
    elif action == "attack":
        # 0-4 roar, 5-10 in-place spin, 11-12 impact, 13-15 rest.
        if f <= 4:
            p["jaw"] = (1, 2, 2, 2, 1)[f]
            p["dy"] = (0, -1, -1, -1, 0)[f]
            p["flip_l"] = (6, 12, 14, 12, 6)[f]
            p["flip_r"] = (-6, -11, -13, -11, -6)[f]
            p["rings"] = 1
            p["ring"] = f
        elif f <= 10:
            p["spin"] = 60.0 * (f - 4)
        elif f in (11, 12):
            p["burst"] = 1
        elif f == 13:
            p["flip_l"] = 4
            p["flip_r"] = -4
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


def assemble(moved):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in DRAW:
        paste(img, moved[name])
    return img


def open_beak(img, jaw_src, bdx, bdy, drop):
    """Drop the existing lower beak. The gap is mouth black, not a new prong."""
    if drop <= 0:
        paste(img, shift(jaw_src, bdx, bdy))
        return img
    home = shift(jaw_src, bdx, bdy)
    dropped = shift(jaw_src, bdx, bdy + drop)
    hole = (home[..., 3] >= 128) & (dropped[..., 3] < 128) & (img[..., 3] < 128)
    img[hole, :3] = BLACK
    img[hole, 3] = 255
    paste(img, dropped)
    return img


def beak_tip_of(layer, facing):
    a = layer[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return None
    # The beak points down, so the mouth is the lowest tip, then one pixel forward.
    y = int(ys.max())
    tip_x = xs[ys == y]
    x = int(np.median(tip_x))
    return x + int(facing), y


def roar_rings(out, mouth, facing, age):
    """Sparse forward arc. Same dot count on every roar frame so the opaque ratio stays tight."""
    mx, my = mouth
    cols = (WHITE, LGOLD, LBLUE)
    for i, base in enumerate((8, 12, 16)):
        rad = base + (age % 2)
        col = cols[i]
        for deg in range(-70, 71, 20):
            ang = math.radians(deg)
            put(out, mx + facing * rad * math.cos(ang), my - 2 + rad * math.sin(ang) * 0.45, col)
    return out


def impact_burst(out, mouth, facing, wide):
    mx, my = mouth
    cols = (WHITE, LGOLD, PALE)
    rads = (7, 11, 15) if wide else (6, 10, 13)
    for i, rad in enumerate(rads):
        col = cols[i]
        for deg in range(-70, 71, 20):
            ang = math.radians(deg)
            put(out, mx + facing * (rad + 2) * math.cos(ang), my + rad * math.sin(ang) * 0.45, col)
    return out


def sparks(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    cx = (x0 + x1) // 2
    d = frame_idx
    pts = (
        (cx - facing * (5 + d), y0 + 2, WHITE),
        (cx + facing * (4 + d), y0 + 4, LGOLD),
        (cx - facing * 2, y0 + 1, PALE),
    )
    for sx, sy, col in pts:
        put(out, sx, sy, col)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def moved_parts(layers, pivots, p, facing):
    bdx, bdy = p["dx"] * facing, p["dy"]
    sgn = -1.0 if facing < 0 else 1.0
    out = {
        "torso": shift(layers["torso"], bdx, bdy),
        "jaw": shift(layers["jaw"], bdx, bdy),
        "foot_l": shift(layers["foot_l"], bdx, bdy - p["lift_l"]),
        "foot_r": shift(layers["foot_r"], bdx, bdy - p["lift_r"]),
    }
    out["flip_l"] = shift(rotate_nearest(layers["flip_l"], p["flip_l"] * sgn, pivots["flip_l"]), bdx, bdy)
    out["flip_r"] = shift(rotate_nearest(layers["flip_r"], p["flip_r"] * sgn, pivots["flip_r"]), bdx, bdy)
    return out, bdx, bdy


def synth(rig, action, f, palette):
    layers, pivots, sole, center, facing = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

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
        base = assemble({k: layers[k] for k in DRAW})
        if spinning:
            img = rotate_nearest(base, p["spin"], center)
        else:
            img = shift(rotate_nearest(base, -facing * p["recoil"], sole), bdx_hurt, 0)
        img = despeckle(img)
    else:
        img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        for name in ("torso", "flip_r", "flip_l", "foot_l", "foot_r"):
            paste(img, moved[name])
        img = open_beak(img, layers["jaw"], bdx, bdy, int(p["jaw"]))

    img = snap(img, palette)
    if p["flash"]:
        img = flash_white(img)
    if p["rings"]:
        tip = beak_tip_of(shift(layers["jaw"], bdx, bdy + int(p["jaw"])), facing)
        if tip:
            img = roar_rings(img, tip, facing, p["ring"])
    if p["burst"]:
        tip = beak_tip_of(layers["jaw"], facing)
        if tip is None:
            bb = body_box(img)
            tip = ((bb[2] if facing > 0 else bb[0]), (bb[1] + bb[3]) // 2) if bb else None
        if tip:
            img = impact_burst(img, tip, facing, wide=(f >= 12))
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


def south_opaque_ratios(sheets):
    """Attack row S, each frame over frame 0."""
    band = sheets["attack"][:CELL_H]
    counts = []
    for f in range(FRAMES["attack"]):
        tile = band[:, f * CELL_W:(f + 1) * CELL_W]
        counts.append(int((tile[..., 3] >= 128).sum()))
    base = counts[0] if counts[0] else 1
    ratios = [c / base for c in counts]
    return counts, ratios


def preview_image(sheets):
    """South row, every frame, tight and scaled so the crown and beak are readable."""
    scale = 2
    pad = 8
    crops = []
    for act in ACTIONS:
        band = sheets[act][:CELL_H]
        row = []
        for f in range(FRAMES[act]):
            tile = band[:, f * CELL_W:(f + 1) * CELL_W]
            a = tile[..., 3] >= 128
            if not np.any(a):
                crop = tile[:8, :8]
            else:
                ys, xs = np.nonzero(a)
                y0 = max(0, int(ys.min()) - pad)
                y1 = min(CELL_H, int(ys.max()) + 1 + pad)
                x0 = max(0, int(xs.min()) - pad)
                x1 = min(CELL_W, int(xs.max()) + 1 + pad)
                crop = tile[y0:y1, x0:x1]
            row.append(np.repeat(np.repeat(crop, scale, 0), scale, 1))
        crops.append(row)
    cell_h = max(c.shape[0] for row in crops for c in row)
    cell_w = max(c.shape[1] for row in crops for c in row)
    cols = 8
    rows = sum(math.ceil(len(row) / cols) for row in crops)
    bg = np.full((rows * cell_h, cols * cell_w, 4), (15, 20, 31, 255), np.uint8)
    r = 0
    for row in crops:
        for i, c in enumerate(row):
            rr, cc = divmod(i, cols)
            y = (r + rr) * cell_h + (cell_h - c.shape[0]) // 2
            x = cc * cell_w + (cell_w - c.shape[1]) // 2
            m = c[..., 3] >= 128
            bg[y:y + c.shape[0], x:x + c.shape[1]][m] = c[m]
        r += math.ceil(len(row) / cols)
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 6
    crops = []
    for act in ACTIONS:
        row = []
        band = sheets[act][:CELL_H]
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
    for name, need in (("flip_l", 200), ("flip_r", 40), ("foot_l", 30), ("foot_r", 30), ("jaw", 8), ("torso", 800)):
        if counts[name] < need:
            raise SystemExit(f"part {name} too small: {counts[name]}")

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}

    def placed_still(flip):
        a = front[..., 3] >= 128
        ys, xs = np.nonzero(a)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        placed = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        dst_x = (CELL_W - (x1 - x0)) // 2
        dst_y = FOOT_Y - (y1 - y0)
        sprite = front[y0:y1, x0:x1]
        if flip:
            sprite = sprite[:, ::-1].copy()
        placed[dst_y:dst_y + (y1 - y0), dst_x:dst_x + (x1 - x0)] = sprite
        return placed

    rest = synth(rigs["W"], "idle", 0, palette)
    placed = placed_still(False)
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    n_diff = int(differ.sum())
    print("rest-pose pixels changed", n_diff, flush=True)
    if n_diff:
        raise SystemExit("idle frame 0 does not match the still")

    south = synth(rigs["S"], "idle", 0, palette)
    # Full-canvas mirror. Crop-then-replace is 1px off when (cell - width) is odd.
    mirrored = placed[:, ::-1].copy()
    ed = np.any(south != mirrored, axis=2) & ((south[..., 3] >= 128) | (mirrored[..., 3] >= 128))
    print("south mirror pixels changed", int(ed.sum()), flush=True)
    if int(ed.sum()):
        raise SystemExit("south idle 0 is not the mirror of the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (80, 140, 200), "flip_l": (40, 200, 80), "flip_r": (220, 60, 60),
        "foot_l": (255, 140, 40), "foot_r": (180, 80, 220), "jaw": (40, 220, 220),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    # South attack opaque ratios, and a one-piece check, before the full sheet.
    atk = [synth(rigs["S"], "attack", f, palette) for f in range(16)]
    counts_a = [int((fr[..., 3] >= 128).sum()) for fr in atk]
    base_n = counts_a[0]
    ratios = [c / base_n for c in counts_a]
    print("south attack opaque", counts_a, flush=True)
    print("south attack ratios", " ".join(f"{r:.3f}" for r in ratios), flush=True)
    bad = [i for i, r in enumerate(ratios) if r < 0.90 or r > 1.08]
    if bad:
        raise SystemExit(f"attack opaque ratio out of range at frames {bad}")

    broken = []
    for f, fr in enumerate(atk):
        if f < 5 or f > 10:
            continue
        sizes = components(fr[..., 3] >= 128)
        body = [s for s in sizes if s > 12]
        if len(body) != 1:
            broken.append(("attack", f, sizes[:6]))
    for f in (0, 3, 6, 9, 11):
        fr = synth(rigs["S"], "hurt", f, palette)
        sizes = components(fr[..., 3] >= 128)
        body = [s for s in sizes if s > 12]
        if len(body) != 1:
            broken.append(("hurt", f, sizes[:6]))
    print("piece check", broken if broken else "ok", flush=True)
    if broken:
        raise SystemExit("spin or recoil broke the body")

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 10)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 10, 14)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in (0, 1, 3, 6, 9, 11)]
        tiles = []
        for act, d, f in want:
            if act == "attack" and d == "S":
                tiles.append(atk[f])
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
        zooms = []
        for tile in tiles:
            crop = tile[70:176, 30:170]
            zooms.append(np.repeat(np.repeat(crop, 2, 0), 2, 1))
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

    counts_a, ratios = south_opaque_ratios(sheets)
    print("south attack opaque", counts_a, flush=True)
    print("south attack ratios", " ".join(f"{r:.3f}" for r in ratios), flush=True)
    bad = [i for i, r in enumerate(ratios) if r < 0.90 or r > 1.08]
    if bad:
        raise SystemExit(f"attack opaque ratio out of range at frames {bad}")

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
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": sorted(alphas),
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "south_attack_ratios": [round(r, 3) for r in ratios],
        "south_attack_min": round(min(ratios), 3),
        "south_attack_max": round(max(ratios), 3),
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
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False), flush=True)

    meta = {
        "generator": "forge_empoleon_actions", "version": VERSION, "key": "0395_empoleon",
        "dex": 395, "form": "normal", "variant": "empoleon",
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "source": "front.png 96x96", "gifs_untouched": True,
        "facing": "left", "beak": "downward centre crest tip",
        "eyes": [[44, 26], [51, 26]], "crown_prongs": 3, "flippers": 2, "feet": 2,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if off or clip or alphas - {0, 255} or sprite_h <= 0:
        raise SystemExit("qa failed")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
