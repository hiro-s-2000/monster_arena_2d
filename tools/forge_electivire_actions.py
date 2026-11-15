#!/usr/bin/env python3
"""Electivire (0466) normal 8-direction action sheet from this form's front.png.

Measured on normal/front.png: 96x96, 2968 opaque, bbox x10-81 y13-83 (72x71),
13 colours, alpha already 0 or 255. One 4-connected piece.

Biped, camera-front. Two red eyes (left x33-34 y30-33, right x46-49 y33-36)
with the white/grey tooth row between them (x35-44 y38-40, 12px). That row is
the mouth. Red bulbs on the head are antenna tips, not a second mouth.

Exactly two feet, both free of plug metal:
  left sole  x22-31 y62-68 (61px)
  right sole x55-65 y65-70 (58px)
Cable tails hang past those soles and end in the plugs (white/grey metal).
The left plug is x17-34 y69-77. The right plug is x55-71 y71-83. They are not
legs and they are not lifted. A static ankle column keeps each cable on the
torso when a sole steps.

Unflipped dirs are S SE E NE (the still faces the camera). N NW W SW are mirrored.

  idle   : feet planted. Plug tips sway one pixel. Frame 0 is the still.
  walk   : the two soles alternate. Cables stay attached and are not feet.
  attack : 0-4 roar the existing mouth plus rings. 5-10 one-piece in-plane
           360 of the whole still, tails included, no |cos| squash.
           11-12 impact. 13-15 the still.
  hurt   : one-piece lean, at most 38 degrees.

Palette-locked to this still. front/back png and gif are not written.
"""

import fcntl
import json
import math
import os
import shutil
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0466_electivire/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0466_electivire/normal/agent-sprite-forge"
)
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
UNFLIPPED = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 192, 192, 168
VERSION = "electivire_v2"
KEY = "0466_electivire"

YEL = (255, 222, 32)
DBR = (49, 41, 41)
OLV = (197, 172, 65)
BLK = (0, 0, 0)
BRN = (90, 74, 65)
TAN = (156, 131, 74)
WHT = (255, 255, 255)
PYL = (255, 238, 148)
DRD = (115, 24, 0)
GRY = (180, 180, 197)
RBR = (180, 57, 24)
ORG = (246, 82, 57)
GG2 = (106, 106, 106)
PALETTE_COLS = (YEL, DBR, OLV, BLK, BRN, TAN, WHT, PYL, DRD, GRY, RBR, ORG, GG2)
METAL = (WHT, GRY, GG2)
RED = (DRD, RBR, ORG)
DRAW = ("torso", "foot_l", "foot_r")


def flood(seeds, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow, dtype=bool)
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


def colored(img, cols):
    rgb = img[..., :3]
    a = img[..., 3] == 255
    hit = np.zeros(a.shape, dtype=bool)
    for col in cols:
        hit |= np.all(rgb == np.array(col, np.uint8), axis=-1)
    return a & hit


def component_masks(mask):
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    parts = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        comp = np.zeros_like(mask)
        while stack:
            y, x = stack.pop()
            comp[y, x] = True
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        parts.append(comp)
    parts.sort(key=lambda m: int(m.sum()), reverse=True)
    return parts


def component_sizes(mask):
    return [int(m.sum()) for m in component_masks(mask)]


def extract_palette(img):
    a = img[..., 3] == 255
    return np.unique(img[a][:, :3], axis=0)


def segment(img):
    h, w = img.shape[:2]
    a = img[..., 3] == 255
    if (h, w) != (96, 96):
        raise SystemExit("front is not 96x96")
    if int(a.sum()) != 2968:
        raise SystemExit(f"still opaque changed: {int(a.sum())}")
    if component_sizes(a) != [2968]:
        raise SystemExit("still is not one 4-connected piece")
    yy, xx = np.mgrid[0:h, 0:w]
    ys, xs = np.nonzero(a)
    if (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())) != (10, 81, 13, 83):
        raise SystemExit("still bbox changed; remeasure before forging")
    palette = extract_palette(img)
    if len(palette) != 13:
        raise SystemExit(f"expected 13 colours, got {len(palette)}")
    for col in PALETTE_COLS:
        if not np.any(np.all(palette == np.array(col, np.uint8), axis=1)):
            raise SystemExit(f"still is missing measured colour {col}")

    foot_l = a & (xx >= 22) & (xx <= 31) & (yy >= 62) & (yy <= 68)
    foot_r = a & (xx >= 52) & (xx <= 65) & (yy >= 65) & (yy <= 70)
    if component_sizes(foot_l) != [61] or component_sizes(foot_r) != [58]:
        raise SystemExit(f"feet changed: {component_sizes(foot_l)} {component_sizes(foot_r)}")
    metal = colored(img, METAL)
    if int((foot_l & metal).sum()) or int((foot_r & metal).sum()):
        raise SystemExit("a foot contains plug metal")
    torso = a & ~foot_l & ~foot_r
    if component_sizes(torso) != [2849]:
        raise SystemExit(f"torso/cables split without the feet: {component_sizes(torso)[:6]}")
    if int(torso.sum()) + 61 + 58 != 2968:
        raise SystemExit("part counts do not add up")

    mouth = metal & (yy >= 37) & (yy <= 42) & (xx >= 33) & (xx <= 48)
    if component_sizes(mouth) != [12]:
        raise SystemExit(f"mouth changed: {component_sizes(mouth)}")
    red = colored(img, RED)
    left_eye = red & (xx < 42) & (yy >= 28) & (yy <= 36)
    right_eye = red & (xx >= 42) & (yy >= 28) & (yy <= 40)
    if int(left_eye.sum()) != 4 or int(right_eye.sum()) != 11:
        raise SystemExit(f"eyes changed: {int(left_eye.sum())} {int(right_eye.sum())}")
    mouth_x = float(np.nonzero(mouth)[1].mean())
    left_x = float(np.nonzero(left_eye)[1].mean())
    right_x = float(np.nonzero(right_eye)[1].mean())
    if not (left_x < mouth_x < right_x):
        raise SystemExit(f"mouth is not between the eyes: {left_x:.1f} {mouth_x:.1f} {right_x:.1f}")
    tip_l = a & (xx <= 40) & (yy >= 74)
    tip_r = a & (xx >= 55) & (yy >= 79)
    if int((tip_l & metal).sum()) < 8 or int((tip_r & metal).sum()) < 8:
        raise SystemExit("plug tips lost their metal")
    if int((tip_l & foot_l).sum()) or int((tip_r & foot_r).sum()):
        raise SystemExit("a plug tip overlaps a foot")
    print(
        f"still 96x96 opaque 2968 bbox x10-81 y13-83 bh=71 facing front "
        f"feet 2 ({int(foot_l.sum())}+{int(foot_r.sum())}) "
        f"tails 2 plugs L{int(tip_l.sum())} R{int(tip_r.sum())} "
        f"mouth {int(mouth.sum())} eyes 2",
        flush=True,
    )
    return {
        "torso": torso,
        "foot_l": foot_l,
        "foot_r": foot_r,
        "mouth": mouth,
        "tip_l": tip_l,
        "tip_r": tip_r,
    }


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


def shift_masked(layer, mask, dx, dy):
    dx_i, dy_i = int(round(dx)), int(round(dy))
    if dx_i == 0 and dy_i == 0:
        return layer
    part = np.zeros_like(layer)
    part[mask] = layer[mask]
    out = layer.copy()
    out[mask] = 0
    moved = shift(part, dx_i, dy_i)
    m = moved[..., 3] == 255
    out[m] = moved[m]
    return out


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest in-plane rotate. A multiple of 360 stays exact. No |cos| squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.05 or not np.any(layer[..., 3] == 255):
        return layer.copy()
    scale = 4
    up = np.repeat(np.repeat(layer, scale, axis=0), scale, axis=1)
    px, py = pivot[0] * scale + 1.5, pivot[1] * scale + 1.5
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
    inv = out[scale // 2::scale, scale // 2::scale].copy()
    opaque = layer[..., 3] == 255
    ys, xs = np.nonzero(opaque)
    dx = xs.astype(np.float64) - pivot[0]
    dy = ys.astype(np.float64) - pivot[1]
    dest_x = np.rint(pivot[0] + c * dx - s * dy).astype(np.int32)
    dest_y = np.rint(pivot[1] + s * dx + c * dy).astype(np.int32)
    h2, w2 = layer.shape[:2]
    ok = (dest_x >= 0) & (dest_x < w2) & (dest_y >= 0) & (dest_y < h2)
    fwd = np.zeros_like(layer)
    fwd[dest_y[ok], dest_x[ok]] = layer[ys[ok], xs[ok]]
    missing = (fwd[..., 3] == 255) & (inv[..., 3] == 0)
    inv[missing] = fwd[missing]
    return inv


def despeckle(img):
    out = img.copy()
    a = out[..., 3] == 255
    pad = np.pad(a, 1, constant_values=False)
    n = (
        pad[:-2, 1:-1].astype(np.int16)
        + pad[2:, 1:-1]
        + pad[1:-1, :-2]
        + pad[1:-1, 2:]
    )
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img, target):
    out = img.copy()
    h, w = out.shape[:2]
    for need in (4, 3):
        a = out[..., 3] == 255
        have = int(a.sum())
        if have >= target:
            break
        pad = np.pad(a, 1, constant_values=False)
        n = (
            pad[:-2, 1:-1].astype(np.int16)
            + pad[2:, 1:-1]
            + pad[1:-1, :-2]
            + pad[1:-1, 2:]
        )
        holes = (~a) & (n >= need)
        ys, xs = np.nonzero(holes)
        if len(ys) == 0:
            continue
        room = target - have
        if len(ys) > room:
            ys, xs = ys[:room], xs[:room]
        for y, x in zip(ys.tolist(), xs.tolist()):
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and out[ny, nx, 3] == 255:
                    out[y, x] = out[ny, nx]
                    break
    return out


def bridge(img, min_size=40, max_gap=4):
    """Join a nick between body-sized pieces. Do not draw a new limb."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(4):
        a = out[..., 3] == 255
        seen = np.zeros_like(a)
        comps = []
        ys, xs = np.nonzero(a)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if seen[y0, x0]:
                continue
            stack = [(y0, x0)]
            seen[y0, x0] = True
            pts = []
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(pts)
        comps.sort(key=len, reverse=True)
        big = [c for c in comps if len(c) >= min_size]
        if len(big) <= 1:
            return out
        main = np.array(big[0], np.int32)
        best = None
        step_m = max(1, len(main) // 400)
        aa = main[::step_m]
        for other in big[1:]:
            bb = np.array(other, np.int32)
            step_o = max(1, len(bb) // 200)
            bb = bb[::step_o]
            d = (aa[:, None, 0] - bb[None, :, 0]) ** 2 + (aa[:, None, 1] - bb[None, :, 1]) ** 2
            flat = int(d.argmin())
            i, j = divmod(flat, bb.shape[0])
            dist = float(d[i, j]) ** 0.5
            if best is None or dist < best[0]:
                best = (dist, tuple(int(v) for v in aa[i]), tuple(int(v) for v in bb[j]))
        if best is None or best[0] > max_gap:
            return out
        y0, x0 = best[1]
        y1, x1 = best[2]
        n = int(max(abs(y1 - y0), abs(x1 - x0)))
        col = out[y0, x0].copy()
        for t in range(n + 1):
            y = int(round(y0 + (y1 - y0) * t / max(n, 1)))
            x = int(round(x0 + (x1 - x0) * t / max(n, 1)))
            if out[y, x, 3] == 0:
                out[y, x] = col
    return out


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] == 255
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
    flat[sel] = mapped[inv]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    out[~a] = 0
    return out


def paste(dst, layer):
    m = layer[..., 3] == 255
    dst[m] = layer[m]


def compose(layers):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in DRAW:
        paste(img, layers[name])
    return img


def build_rig(img, masks, flip):
    a = img[..., 3] == 255
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    crop = img[y0:y1, x0:x1].copy()
    crop[crop[..., 3] == 0] = 0
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = crop
    placed = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), dtype=bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm
    layers = {}
    for name in DRAW:
        lyr = np.zeros_like(canvas)
        lyr[placed[name]] = canvas[placed[name]]
        layers[name] = lyr
    rest = compose(layers)
    if np.any(rest != canvas):
        raise SystemExit("rest pose does not match the still")
    oys, oxs = np.nonzero(canvas[..., 3] == 255)
    center = (float(oxs.mean()), float(oys.mean()))
    soles = {}
    foot_pts = []
    for name in ("foot_l", "foot_r"):
        sy, sx = np.nonzero(placed[name])
        yb = int(sy.max())
        pts = [(int(x), yb) for x in sx[sy == yb]]
        soles[name] = pts
        foot_pts.append((float(np.mean([p[0] for p in pts])), float(yb)))
    sole = (float(sum(p[0] for p in foot_pts) / 2.0), float(max(p[1] for p in foot_pts)))
    mouth = placed["mouth"]
    tip_l = placed["tip_l"]
    tip_r = placed["tip_r"]
    facing = 1.0
    if flip:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        mouth = np.ascontiguousarray(mouth[:, ::-1])
        tip_l = np.ascontiguousarray(tip_l[:, ::-1])
        tip_r = np.ascontiguousarray(tip_r[:, ::-1])

        def mx(x):
            return CELL_W - 1 - x

        center = (mx(center[0]), center[1])
        sole = (mx(sole[0]), sole[1])
        soles = {k: [(mx(x), y) for x, y in pts] for k, pts in soles.items()}
        facing = -1.0
        rest = np.ascontiguousarray(rest[:, ::-1])
    if np.any(compose(layers) != canvas):
        raise SystemExit("flipped rest does not match the still")
    return {
        "layers": layers,
        "center": center,
        "sole": sole,
        "soles": soles,
        "mouth": mouth,
        "tip_l": tip_l,
        "tip_r": tip_r,
        "facing": facing,
        "rest": canvas,
        "body_n": int((canvas[..., 3] == 255).sum()),
    }


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        lift_l=0.0, lift_r=0.0, fwd_l=0.0, fwd_r=0.0,
        tip_l=0.0, tip_r=0.0, spin=0.0, recoil=0.0, knock=0.0,
        mouth=0, rings=0, burst=0, sparks=0,
    )
    if action == "idle":
        p["tip_l"] = wave(f, n, 1.15)
        p["tip_r"] = wave(f, n, 1.15, phase=math.pi)
    elif action == "walk":
        step = wave(f, n, 1.0, cycles=2.0)
        p["lift_l"] = max(0.0, step) * 3.0
        p["lift_r"] = max(0.0, -step) * 3.0
        p["fwd_l"] = max(0.0, step) * 2.0
        p["fwd_r"] = max(0.0, -step) * 2.0
        p["tip_l"] = wave(f, n, 1.15, phase=math.pi, cycles=2.0)
        p["tip_r"] = wave(f, n, 1.15, cycles=2.0)
    elif action == "attack":
        if f <= 4:
            p["mouth"] = (2, 3, 4, 3, 2)[f]
            p["rings"] = 1
            p["tip_l"] = (1, -1, 1, -1, 0)[f]
            p["tip_r"] = (-1, 1, -1, 1, 0)[f]
        elif f <= 10:
            # Mouth stays at the roar's 4px. The tail plugs whip ±1, which stays
            # on the body. A ring pixel on these frames does not. No body spin.
            p["mouth"] = 4
            whip = (1, -1, 1, -1, 1, -1)[f - 5]
            p["tip_l"] = whip
            p["tip_r"] = -whip
        elif f <= 12:
            p["burst"] = 1
            p["mouth"] = 2
    else:
        env = math.sin(math.pi * f / 11.0)
        p["recoil"] = 38.0 * env
        p["knock"] = 3.0 * env
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def halo(out, origin, n, radius, spin):
    """Dots just outside the body, so a roar reads without gluing onto a tail."""
    if origin is None:
        return out
    ox, oy = origin
    cols = (YEL, WHT, PYL)
    for i in range(n):
        ang = spin + 2 * math.pi * i / n
        landed = False
        for rad in range(radius, radius + 22):
            x = ox + rad * math.cos(ang)
            y = oy + rad * math.sin(ang)
            if put(out, x, y, cols[i % 3]):
                landed = True
                break
            if not (2 <= int(round(x)) < out.shape[1] - 2 and 2 <= int(round(y)) < out.shape[0] - 2):
                break
        if not landed:
            continue
    return out


def open_mouth(img, mouth, drop):
    if drop <= 0 or not np.any(mouth):
        return img
    out = img.copy()
    part = np.zeros_like(img)
    part[mouth] = img[mouth]
    out[mouth, :3] = BLK
    out[mouth, 3] = 255
    paste(out, shift(part, 0, drop))
    return out


def mouth_origin(mouth, drop):
    ys, xs = np.nonzero(mouth)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean()) + drop


def sparks(out, facing, frame_idx):
    a = out[..., 3] == 255
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return out
    x = int(xs.max() if facing > 0 else xs.min())
    y = int(np.median(ys))
    for i, col in enumerate((YEL, WHT, PYL)):
        put(out, x - facing * (4 + i * 3 + frame_idx), y - 8 + i * 5, col)
    return out


def cleanup(img, target):
    img = despeckle(img)
    img = fill_pinholes(img, target)
    img = bridge(img)
    return img


def synth(rig, action, f, palette):
    layers = rig["layers"]
    facing = rig["facing"]
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = (p["spin"] % 360.0) > 0.5
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        if spinning:
            img = rotate_nearest(rig["rest"], p["spin"], rig["center"])
        else:
            img = rotate_nearest(rig["rest"], facing * p["recoil"], rig["sole"])
            img = shift(img, -facing * p["knock"], 0)
        img = cleanup(img, rig["body_n"])
    else:
        tip_l = int(np.clip(round(p["tip_l"]), -1, 1))
        tip_r = int(np.clip(round(p["tip_r"]), -1, 1))
        torso = layers["torso"]
        if tip_l or tip_r:
            torso = torso.copy()
            if tip_l:
                torso = shift_masked(torso, rig["tip_l"], facing * tip_l, 0)
            if tip_r:
                torso = shift_masked(torso, rig["tip_r"], facing * tip_r, 0)
        moved = {
            "torso": torso,
            "foot_l": shift(layers["foot_l"], facing * p["fwd_l"], -p["lift_l"]),
            "foot_r": shift(layers["foot_r"], facing * p["fwd_r"], -p["lift_r"]),
        }
        img = compose(moved)
        if tip_l or tip_r or p["lift_l"] or p["lift_r"] or p["fwd_l"] or p["fwd_r"]:
            img = bridge(img)
        if p["mouth"]:
            img = open_mouth(img, rig["mouth"], int(p["mouth"]))
    img = snap(img, palette)
    origin = mouth_origin(rig["mouth"], int(p["mouth"]))
    if p["rings"]:
        img = halo(img, origin, 10, 9, 0.2 * f)
    if p["burst"]:
        img = halo(img, origin, 12, 12, 0.4)
    if p["sparks"] and recoiling:
        img = sparks(img, facing, f)
    return snap(img, palette)


def soles_hold(img, pts):
    return all(img[y, x, 3] == 255 for x, y in pts)


def one_sheet(rigs, palette, action):
    width = max(FRAMES.values()) * CELL_W
    sheet = np.zeros((8 * CELL_H, width, 4), np.uint8)
    for d in DIRS:
        di = DIRS.index(d)
        for f in range(FRAMES[action]):
            sheet[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = synth(
                rigs[d], action, f, palette
            )
    return sheet


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


def contact_sheet(sheets):
    row = 0
    probe = sheets["idle"][:CELL_H, :CELL_W]
    ys, xs = np.nonzero(probe[..., 3] == 255)
    pad = 16
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad)
    scale = 2
    cw, ch = (x1 - x0) * scale, (y1 - y0) * scale
    cols = 8
    blocks = []
    for act in ACTIONS:
        band = sheets[act][row * CELL_H:(row + 1) * CELL_H]
        frames = []
        for f in range(FRAMES[act]):
            tile = band[y0:y1, f * CELL_W + x0:f * CELL_W + x1]
            frames.append(np.repeat(np.repeat(tile, scale, 0), scale, 1))
        blocks.append(frames)
    rows = sum(math.ceil(len(b) / cols) for b in blocks)
    bg = np.full((rows * ch, cols * cw, 4), (18, 22, 32, 255), np.uint8)
    r = 0
    for frames in blocks:
        for i, up in enumerate(frames):
            rr, cc = divmod(i, cols)
            y, x = (r + rr) * ch, cc * cw
            m = up[..., 3] == 255
            bg[y:y + ch, x:x + cw][m] = up[m]
        r += math.ceil(len(frames) / cols)
    return Image.fromarray(bg, "RGBA")


def south_attack_ratios(sheets):
    base_tile = sheets["attack"][:CELL_H, :CELL_W]
    base = int((base_tile[..., 3] == 255).sum())
    ratios = []
    counts = []
    for f in range(FRAMES["attack"]):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        n = int((tile[..., 3] == 255).sum())
        counts.append(n)
        ratios.append(n / base if base else 0.0)
    return counts, ratios


def sprite_box(tile):
    ys, xs = np.nonzero(tile[..., 3] == 255)
    if len(xs) == 0:
        return 0, 0
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def print_ratios(counts, ratios, sheets):
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(counts, ratios)):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        sizes = component_sizes(tile[..., 3] == 255)
        print(
            f"attack S {f:02d} opaque {c} ratio {r:.4f} components {sizes}",
            flush=True,
        )
    print(f"min {min(ratios):.4f} max {max(ratios):.4f}", flush=True)


def qa_report(master, sheets, palette, ratios):
    opaque = master[..., 3] == 255
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    off = []
    if np.any(opaque):
        cols = np.unique(master[opaque][:, :3], axis=0)
        pal = {tuple(int(c) for c in row) for row in palette}
        off = [tuple(int(c) for c in row) for row in cols if tuple(int(c) for c in row) not in pal]
    clip = 0
    split = []
    narrow = []
    for act in ACTIONS:
        for di in range(8):
            for f in range(FRAMES[act]):
                tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = tile[..., 3] == 255
                if not np.any(a):
                    clip += 1
                    continue
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip += 1
                sizes = component_sizes(a)
                big = [n for n in sizes if n >= 70]
                if len(big) != 1:
                    split.append({"act": act, "dir": DIRS[di], "f": f, "big": big, "sizes": sizes[:6]})
    for f in range(5, 11):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        sizes = component_sizes(tile[..., 3] == 255)
        if sizes != [sum(sizes)] or len(sizes) != 1:
            split.append({"act": "attack", "dir": "S", "f": f, "spin": sizes[:8]})
        w, h = sprite_box(tile)
        if w < 55 or h < 55:
            narrow.append({"f": f, "w": w, "h": h})
    bad_ratio = [round(r, 4) for r in ratios if r < 0.90 or r > 1.08]
    idle0 = sheets["idle"][:CELL_H, :CELL_W]
    bw, bh = sprite_box(idle0)
    return {
        "alphas": alphas,
        "off_palette": len(off),
        "clip_frames": clip,
        "sprite_h": bh,
        "sprite_w": bw,
        "bad_ratio": bad_ratio,
        "split": split[:12],
        "split_n": len(split),
        "narrow": narrow,
    }


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
        "dex": 466,
        "form": "normal",
        "variant": "electivire",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0466_electivire/normal",
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

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        raw = text.strip()
        if not raw.startswith(prefix) or "\n" in raw:
            raise SystemExit("dir8.js is not a single assignment line")
        body = raw[len(prefix):]
        if not body.endswith(";"):
            raise SystemExit("dir8.js missing trailing semicolon")
        body = body[:-1]
        data = json.loads(body)
        if KEY not in data:
            raise SystemExit("dir8 has no 0466_electivire")
        token = '"' + KEY + '":'
        start = body.find(token)
        if start < 0 or body.find(token, start + len(token)) != -1:
            raise SystemExit("dir8 key is missing or duplicated")
        val_at = start + len(token)
        _old, end = json.JSONDecoder().raw_decode(body, val_at)
        new_val = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        new_body = body[:val_at] + new_val + body[end:]
        out = prefix + new_body + ";\n"
        if out.count("\n") != 1:
            raise SystemExit("dir8.js picked up a line break")
        parsed = json.loads(out[len(prefix):].rstrip().rstrip(";"))
        if parsed[KEY] != entry:
            raise SystemExit("dir8 entry did not stick")
        if list(parsed) != list(data):
            raise SystemExit("dir8 key order changed")
        for k, old in data.items():
            if k == KEY:
                continue
            if parsed[k] != old:
                raise SystemExit(f"dir8 key changed: {k}")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(out)
        tmp.replace(DIR8)
        return True

    return with_flock(DIR8_LOCK, edit)


def update_queue(status):
    def edit():
        fd = os.open(QUEUE, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with os.fdopen(os.dup(fd), "r+", encoding="utf-8") as handle:
                raw = handle.read()
                data = json.loads(raw)
                found = False
                for item in data["items"]:
                    if item.get("id") != KEY:
                        continue
                    item["status"] = status
                    item.pop("claim", None)
                    found = True
                if not found:
                    raise SystemExit("queue has no 0466_electivire")
                before = json.loads(raw)
                for b, a in zip(before["items"], data["items"]):
                    if b.get("id") == KEY:
                        continue
                    if b != a:
                        raise SystemExit(f"queue neighbor changed: {b.get('id')}")
                text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
                after = json.loads(text)
                for b, a in zip(before["items"], after["items"]):
                    if b.get("id") == KEY:
                        if a.get("status") != status or "claim" in a:
                            raise SystemExit("queue status did not stick")
                        continue
                    if b != a:
                        raise SystemExit(f"queue neighbor changed: {b.get('id')}")
                tmp = QUEUE.with_suffix(".json.tmp")
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def sync_arena():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    copied = 0
    for fp in OUT_DIR.iterdir():
        if not fp.is_file():
            continue
        if "_work" in fp.name:
            continue
        shutil.copy2(fp, SYNC_OUT / fp.name)
        copied += 1
    return copied


def main():
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if set(np.unique(front[..., 3]).tolist()) - {0, 255}:
        raise SystemExit("front alpha is not binary")
    palette = extract_palette(front)
    masks = segment(front)
    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    if int((rigs["S"]["rest"][..., 3] == 255).sum()) != 2968:
        raise SystemExit("placed still count changed")
    for d, rig in rigs.items():
        for act in ("idle", "walk", "hurt"):
            if np.any(synth(rig, act, 0, palette) != rig["rest"]):
                raise SystemExit(f"{act} frame 0 does not match the still ({d})")
    if max(pose("hurt", f)["recoil"] for f in range(12)) > 38.001:
        raise SystemExit("recoil exceeds 38")
    for f in range(16):
        step = pose("walk", f)
        if step["lift_l"] > 0.5 and step["lift_r"] > 0.5:
            raise SystemExit(f"both feet lifted on walk {f}")
        idle = synth(rigs["S"], "idle", f, palette)
        if not soles_hold(idle, rigs["S"]["soles"]["foot_l"]):
            raise SystemExit(f"left foot left the ground on idle {f}")
        if not soles_hold(idle, rigs["S"]["soles"]["foot_r"]):
            raise SystemExit(f"right foot left the ground on idle {f}")
        walked = synth(rigs["S"], "walk", f, palette)
        if step["lift_l"] < 0.5 and not soles_hold(walked, rigs["S"]["soles"]["foot_l"]):
            raise SystemExit(f"left sole slipped on walk {f}")
        if step["lift_r"] < 0.5 and not soles_hold(walked, rigs["S"]["soles"]["foot_r"]):
            raise SystemExit(f"right sole slipped on walk {f}")

    sheets = {act: one_sheet(rigs, palette, act) for act in ACTIONS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts, ratios = south_attack_ratios(sheets)
    print_ratios(counts, ratios, sheets)
    master = write_outputs(sheets)
    preview = contact_sheet(sheets)
    preview.save(OUT_DIR / "preview.png")
    preview.save(OUT_DIR / "preview_motion.png")
    report = qa_report(master, sheets, palette, ratios)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    failed = (
        report["off_palette"] or report["clip_frames"] or report["alphas"] != [0, 255]
        or report["bad_ratio"] or report["split_n"] or report["narrow"]
        or report["sprite_h"] < 60
    )
    meta = {
        "generator": "forge_electivire_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 466,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": report["sprite_h"],
        "frames": FRAMES,
        "directions": DIRS,
        "unflipped": sorted(UNFLIPPED),
        "still_faces": "front",
        "feet": 2,
        "mouth_pixels": 12,
        "plugs_are_legs": False,
        "palette_colors": 13,
        "off_palette_colors": report["off_palette"],
        "attack_s_ratios": [round(r, 4) for r in ratios],
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failed:
        update_queue("needs_fix")
        raise SystemExit(f"qa failed {report}")
    copied = sync_arena()
    publish_dir8(report["sprite_h"])
    update_queue("done")
    arena_bytes = (SYNC_OUT / "sheet.png").stat().st_size
    print(f"height {report['sprite_h']} feet 2", flush=True)
    print(f"synced {copied} arena_sheet_bytes {arena_bytes}", flush=True)
    print(f"dir8 v {VERSION}", flush=True)


if __name__ == "__main__":
    main()
