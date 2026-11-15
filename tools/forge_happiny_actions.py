#!/usr/bin/env python3
"""Happiny (0440) 8-direction action sheet from this form's 96x96 front.png.

Measured on normal/front.png only (not Chansey, not Blissey):
  687 opaque pixels, alpha already 0 or 255, bbox x34-62 y27-68 (29x42), 9 colours.
  One hair curl trails up-right: y27-36 x49-60, 82 px, one piece.
  Eyes are four ink pixels (42,46) (42,47) (46,47) (46,48). No mouth.
  Eye mean x=44 is left of the body centroid (49.1); the curl mean is to the right.
  The still faces left. Unflipped dirs are N NW W SW. S SE E NE are mirrored.
  The white egg (74 px) and its gray shade sit in the pouch and stay on the torso.
  Two feet actually touch, and they are not the same row:
    left stub  y64-66 x41-44 (11 px), sole y66 x42-44
    right foot y65-68 x50-56 (24 px), sole y68 x51-54 (closer / lower)
  No extra legs. The egg is not a separate part, so the spin cannot drop it.

  idle   : curl hinges at its base. Frame 0 is the still. Both soles stay down.
  walk   : the two short feet alternate. Body and egg stay planted.
  attack : 0-4 curl flare plus rings (no invented mouth), 5-10 one-piece
           in-plane 360 (no |cos| squash), 11-12 impact, 13-15 the still.
  hurt   : one-piece lean about the sole, at most 38 degrees, head back.

Palette-locked to this still. front/back png and gif are not written.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
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
SRC_DIR = ROOT / "monsterSprite/gen4/0440_happiny/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0440_happiny/normal/agent-sprite-forge"
)
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 147
VERSION = "happiny_v2"
KEY = "0440_happiny"

# Colours measured on this still. Not another species' mask.
INK = (16, 16, 16)
BLUSH = (222, 106, 156)
PINK = (255, 197, 213)
WHITE = (255, 255, 255)
MAUVE = (164, 82, 123)
MID = (197, 139, 164)
ROSE = (164, 90, 123)
PLUM = (115, 65, 90)
SHADE = (205, 189, 197)
PALETTE_REQUIRED = (INK, BLUSH, PINK, WHITE, MAUVE, MID, ROSE, PLUM, SHADE)

DRAW = ("torso", "foot_l", "foot_r", "curl")
# dx > 0 is forward (times facing). dy > 0 is down. Fixed counts, palette only.
RING = (
    (-2, -6), (-4, -9), (-1, -11), (-6, -5),
    (-3, -13), (0, -8), (-7, -8), (-2, -3),
)
BURST = (
    (6, -4), (9, -1), (7, 3), (11, 0),
    (8, -7), (5, 5), (13, -3), (10, 4),
)
HURT_SPARK = ((-4, -6), (-6, -2), (-5, 2), (-8, -4))


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


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def components(mask):
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    out = []
    ys, xs = np.nonzero(mask)
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
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        xs2 = [p[1] for p in pts]
        ys2 = [p[0] for p in pts]
        out.append((len(pts), min(xs2), min(ys2), max(xs2), max(ys2)))
    return out


def segment(img):
    """Curl, two real feet, and one torso that keeps the egg in the pouch."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    if int(a.sum()) != 687:
        raise SystemExit(f"still opaque changed: {int(a.sum())}")
    yy, xx = np.mgrid[0:h, 0:w]
    ys, xs = np.nonzero(a)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    if bbox != (34, 27, 62, 68):
        raise SystemExit(f"still bbox changed: {bbox}")
    for x, y in ((42, 46), (42, 47), (46, 47), (46, 48)):
        if tuple(int(v) for v in img[y, x, :3]) != INK:
            raise SystemExit(f"eye pixel moved at {x},{y}")

    curl = flood(a & (yy <= 28) & (xx >= 55), a & (yy <= 36) & (xx >= 48))
    foot_l = a & (yy >= 64) & (xx <= 44)
    foot_r = a & (yy >= 65) & (xx >= 50)
    parts = {"curl": curl, "foot_l": foot_l, "foot_r": foot_r}
    used = np.zeros_like(a)
    for name, m in parts.items():
        if np.any(used & m):
            raise SystemExit(f"overlap at {name}")
        used |= m
    parts["torso"] = a & ~used
    if int(used.sum() + parts["torso"].sum()) != int(a.sum()):
        raise SystemExit("masks do not cover the still")

    expect = {"curl": 82, "foot_l": 11, "foot_r": 24, "torso": 570}
    info = {}
    for name, n in expect.items():
        got = components(parts[name])
        info[name] = got
        if len(got) != 1 or got[0][0] != n:
            raise SystemExit(f"{name} is not the measured piece: {got}")
    if info["curl"][0][2] != 27 or info["curl"][0][4] != 36:
        raise SystemExit(f"curl is not the top tuft: {info['curl']}")
    if info["foot_l"][0][4] != 66 or info["foot_r"][0][4] != 68:
        raise SystemExit(f"feet do not end on the measured soles: {info}")

    left_sole = {(int(y), int(x)) for y, x in zip(*np.nonzero(foot_l)) if y == 66}
    right_sole = {(int(y), int(x)) for y, x in zip(*np.nonzero(foot_r)) if y == 68}
    if left_sole != {(66, 42), (66, 43), (66, 44)}:
        raise SystemExit(f"left sole changed: {sorted(left_sole)}")
    if right_sole != {(68, 51), (68, 52), (68, 53), (68, 54)}:
        raise SystemExit(f"right sole changed: {sorted(right_sole)}")

    white = a & np.all(img[..., :3] == np.array(WHITE, np.uint8), axis=-1)
    shade = a & np.all(img[..., :3] == np.array(SHADE, np.uint8), axis=-1)
    if int(white.sum()) != 74 or int((white & parts["torso"]).sum()) != 74:
        raise SystemExit("egg white is not entirely inside the torso pouch")
    if int(shade.sum()) != 23 or int((shade & parts["torso"]).sum()) != 23:
        raise SystemExit("egg shade left the torso")
    eye_x = np.mean([42, 42, 46, 46])
    if eye_x >= float(xs.mean()):
        raise SystemExit("eyes are not left of centre; remeasure facing")
    cys, cxs = np.nonzero(curl)
    if float(cxs.mean()) <= float(xs.mean()):
        raise SystemExit("curl does not trail to the right of centre")
    return parts


def edge_pivot(mask, side):
    ys, xs = np.nonzero(mask)
    if side == "bottom":
        sel = ys >= ys.max() - 1
    elif side == "top":
        sel = ys <= ys.min() + 1
    elif side == "left":
        sel = xs <= xs.min() + 1
    else:
        sel = xs >= xs.max() - 1
    return float(xs[sel].mean()), float(ys[sel].mean())


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def require_colors(palette):
    for col in PALETTE_REQUIRED:
        if not np.any(np.all(palette == np.array(col, np.uint8), axis=1)):
            raise SystemExit(f"still is missing measured colour {col}")


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    for i in range(0, len(ty), 300):
        sl = slice(i, i + 300)
        d = (ty[sl, None] - pts[None, :, 0]) ** 2 + (tx[sl, None] - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def build_rig(img, masks, flip):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    if sh != 42 or sw != 29:
        raise SystemExit(f"cropped still is not 29x42: {sw}x{sh}")
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - (sh - 1)
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    if int(oys.max()) != FOOT_Y:
        raise SystemExit(f"sole is not on fy {FOOT_Y}: {int(oys.max())}")
    if CELL_H - 1 - FOOT_Y != 20:
        raise SystemExit("need 20 px under the foot line")

    placed = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), dtype=bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm
    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr
    for part, rad in (("curl", 2), ("foot_l", 1), ("foot_r", 1)):
        zone = placed[part] & dilate(placed["torso"], rad)
        fill = nearest_color(canvas, zone, placed["torso"])
        write = zone & (layers["torso"][..., 3] < 128)
        layers["torso"][write] = fill[write]

    pivots = {
        "curl": edge_pivot(placed["curl"], "bottom"),
        "foot_l": edge_pivot(placed["foot_l"], "top"),
        "foot_r": edge_pivot(placed["foot_r"], "top"),
    }
    center = (float(oxs.mean()), float(oys.mean()))
    sole_y = int(oys.max())
    sole = (float(oxs[oys == sole_y].mean()), float(sole_y))
    cys, cxs = np.nonzero(placed["curl"])
    tip = int(cys.min())
    crest = (float(cxs[cys == tip].mean()), float(tip))
    head = (oys >= int(oys.min()) + 12) & (oys <= int(oys.min()) + 22)
    hxs = oxs[head]
    hys = oys[head]
    front_x = int(hxs.min())
    face = (float(front_x), float(hys[hxs == front_x].mean()))
    soles = {}
    for name in ("foot_l", "foot_r"):
        sy, sx = np.nonzero(placed[name])
        yb = int(sy.max())
        soles[name] = [(int(x), yb) for x in sx[sy == yb]]

    if flip:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}

        def mx(x):
            return CELL_W - 1 - x

        pivots = {k: (mx(x), y) for k, (x, y) in pivots.items()}
        center = (mx(center[0]), center[1])
        sole = (mx(sole[0]), sole[1])
        crest = (mx(crest[0]), crest[1])
        face = (mx(face[0]), face[1])
        soles = {k: [(mx(x), y) for x, y in pts] for k, pts in soles.items()}
        facing = 1.0
    else:
        facing = -1.0
    rest = compose(layers)
    if np.any(rest != canvas):
        raise SystemExit("rest pose does not match the still")
    return {
        "layers": layers,
        "pivots": pivots,
        "center": center,
        "sole": sole,
        "crest": crest,
        "face": face,
        "facing": facing,
        "rest": canvas,
        "soles": soles,
        "counts": {k: int(np.count_nonzero(layers[k][..., 3] >= 128)) for k in DRAW},
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


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest in-plane rotate. 0 and 360 stay exact. No |cos| squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.05 or not np.any(layer[..., 3] >= 128):
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
    return out[scale // 2::scale, scale // 2::scale].copy()


def despeckle(img):
    """Drop a pixel only when all four neighbours are empty. After rotation only."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img, target):
    """Fill interior holes back toward the still count. Do not grow past it."""
    out = img.copy()
    for need in (4, 3, 3, 2):
        a = out[..., 3] >= 128
        have = int(a.sum())
        if have >= target:
            break
        p = np.pad(a, 1)
        n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        holes = (~a) & (n >= need)
        ys, xs = np.nonzero(holes)
        if len(ys) == 0:
            continue
        # Prefer pixels that sit between opaque neighbours, not a long outer fringe.
        order = np.argsort(-n[ys, xs], kind="stable")
        ys, xs = ys[order], xs[order]
        room = target - have
        if len(ys) > room:
            ys, xs = ys[:room], xs[:room]
        for y, x in zip(ys.tolist(), xs.tolist()):
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                ny, nx = y + dy, x + dx
                if out[ny, nx, 3] >= 128:
                    out[y, x] = out[ny, nx]
                    break
    return out


def rotate_part(layer, angle, pivot, target):
    spun = rotate_nearest(layer, angle, pivot)
    if abs(angle) < 0.05:
        return spun
    return despeckle(fill_pinholes(spun, target))


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
        out[..., 3] = 0
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
    # Snap maps onto the palette. Anything still off-palette is cleared.
    known = np.zeros(a.shape, dtype=bool)
    for row in palette:
        known |= np.all(out[..., :3] == row.astype(np.uint8), axis=-1)
    out[a & ~known] = 0
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def compose(layers):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in DRAW:
        paste(img, layers[name])
    return img


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero at frame 0 and at frame n, so the loop closes on the still."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        curl=0.0, lift_l=0.0, lift_r=0.0, fwd_l=0.0, fwd_r=0.0,
        spin=0.0, recoil=0.0, knock=0.0, rings=0, burst=0, sparks=0,
    )
    if action == "idle":
        p["curl"] = wave(f, n, 14.0)
    elif action == "walk":
        step = wave(f, n, 1.0, cycles=2.0)
        p["lift_l"] = max(0.0, step) * 2.0
        p["lift_r"] = max(0.0, -step) * 2.0
        p["fwd_l"] = max(0.0, step) * 1.0
        p["fwd_r"] = max(0.0, -step) * 1.0
        p["curl"] = wave(f, n, 10.0, cycles=2.0, phase=math.pi)
    elif action == "attack":
        if f <= 4:
            p["curl"] = (0.0, 10.0, 20.0, 14.0, 6.0)[f]
            p["rings"] = 1
        elif f <= 10:
            p["spin"] = 60.0 * (f - 4)
        elif f in (11, 12):
            p["burst"] = 1
    else:
        env = math.sin(math.pi * f / 11.0)
        p["recoil"] = min(38.0, 38.0 * env)
        p["knock"] = 3.0 * env
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def dots(out, origin, facing, table):
    ox, oy = origin
    cols = (WHITE, BLUSH, PINK, MID)
    for i, (dx, dy) in enumerate(table):
        put(out, ox + facing * dx, oy + dy, cols[i % 4])
    return out


def synth(rig, action, f, palette):
    layers, pivots = rig["layers"], rig["pivots"]
    facing = rig["facing"]
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = (p["spin"] % 360.0) > 0.5
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        if spinning:
            img = rotate_nearest(rig["rest"], p["spin"], rig["center"])
        else:
            # Facing left leans the head up and backward, pivoting on the sole.
            img = rotate_nearest(rig["rest"], -facing * p["recoil"], rig["sole"])
            img = shift(img, -facing * p["knock"], -0.35 * p["knock"])
        img = despeckle(fill_pinholes(img, rig["counts"]["body"]))
    else:
        moved = {
            "torso": layers["torso"],
            "foot_l": shift(layers["foot_l"], p["fwd_l"] * facing, -p["lift_l"]),
            "foot_r": shift(layers["foot_r"], p["fwd_r"] * facing, -p["lift_r"]),
            "curl": rotate_part(layers["curl"], -facing * p["curl"], pivots["curl"], rig["counts"]["curl"]),
        }
        img = compose(moved)
    img = snap(img, palette)
    if p["rings"] and not spinning and not recoiling:
        img = dots(img, rig["crest"], facing, RING)
    if p["burst"]:
        img = dots(img, rig["face"], facing, BURST)
    if p["sparks"] and recoiling:
        img = dots(img, rig["crest"], facing, HURT_SPARK)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (SRC_DIR / "front.png").read_bytes()
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
    return master


def contact_sheet(sheets, row):
    """Every frame of one direction, cropped so the body, egg, and feet read."""
    probe = sheets["idle"][row * CELL_H:(row + 1) * CELL_H, :CELL_W]
    ys, xs = np.nonzero(probe[..., 3] >= 128)
    pad = 18
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad)
    scale = 3
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
            m = up[..., 3] >= 128
            bg[y:y + ch, x:x + cw][m] = up[m]
        r += math.ceil(len(frames) / cols)
    return Image.fromarray(bg, "RGBA")


def south_attack_ratios(sheets):
    base = int((sheets["attack"][:CELL_H, :CELL_W, 3] == 255).sum())
    ratios = []
    counts = []
    for f in range(FRAMES["attack"]):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        n = int((tile[..., 3] == 255).sum())
        counts.append(n)
        ratios.append(n / base if base else 0.0)
    return counts, ratios


def largest_fraction(img):
    a = img[..., 3] >= 128
    if not np.any(a):
        return 1.0
    best = 0
    total = int(a.sum())
    seen = np.zeros_like(a)
    h, w = a.shape
    ys, xs = np.nonzero(a)
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
                if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        best = max(best, n)
    return best / total


def white_in_main(img):
    """Share of egg-white pixels that sit in the largest component."""
    a = img[..., 3] >= 128
    white = a & np.all(img[..., :3] == np.array(WHITE, np.uint8), axis=-1)
    total = int(white.sum())
    if total == 0 or not np.any(a):
        return 0.0
    seen = np.zeros_like(a)
    h, w = a.shape
    best_mask = None
    best_n = 0
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
        if len(pts) > best_n:
            best_n = len(pts)
            best_mask = pts
    inside = 0
    for y, x in best_mask:
        if white[y, x]:
            inside += 1
    return inside / total


def save_parts(img, masks, path):
    cols = {
        "curl": (255, 214, 90),
        "foot_l": (80, 170, 255),
        "foot_r": (80, 220, 120),
        "torso": (255, 170, 196),
    }
    part = np.zeros((*img.shape[:2], 4), np.uint8)
    for name, col in cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    big = np.repeat(np.repeat(part, 8, 0), 8, 1)
    still = np.repeat(np.repeat(img, 8, 0), 8, 1)
    gap = np.zeros((big.shape[0], 8, 4), np.uint8)
    Image.fromarray(np.concatenate([still, gap, big], axis=1), "RGBA").save(path)


def with_flock(lock_path, fn):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def object_span(text, key):
    token = f'"{key}":'
    start = text.index(token)
    i = start + len(token)
    while text[i].isspace():
        i += 1
    if text[i] != "{":
        raise SystemExit(f"{key} value is not an object")
    depth = 0
    end = i
    while end < len(text):
        ch = text[end]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, end + 1
        end += 1
    raise SystemExit(f"unclosed {key}")


def publish_dir8(sprite_h):
    entry = {
        "dex": 440,
        "form": "normal",
        "variant": "happiny",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0440_happiny/normal",
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
        if not text.startswith(prefix) or not text.rstrip().endswith(";"):
            raise SystemExit("dir8.js header changed; not publishing")
        key_at, val_end = object_span(text, KEY)
        val_start = text.index("{", key_at)
        updated = text[:val_start] + blob + text[val_end:]
        if updated[:val_start] != text[:val_start] or updated[val_start + len(blob):] != text[val_end:]:
            raise SystemExit("dir8 splice touched bytes outside 0440_happiny")
        data = json.loads(updated[len(prefix):].rstrip().rstrip(";"))
        got = data[KEY]
        if got["v"] != VERSION or got["layout"] != "actions_8dir" or got["an"] != [16, 16, 16, 12]:
            raise SystemExit("dir8 entry did not stick")
        if got["raw"] != 1 or got["sheetDir"] != "agent-sprite-forge" or got["h"] != int(sprite_h):
            raise SystemExit("dir8 entry fields mismatch")
        if got["cw"] != CELL_W or got["ch"] != CELL_H or got["fy"] != FOOT_Y:
            raise SystemExit("dir8 cell fields mismatch")
        # One line, same as the file we read, with only this value replaced.
        if "\n" in updated.strip():
            raise SystemExit("dir8.js would stop being one line")
        tmp = DIR8.with_suffix(f".js.tmp.{os.getpid()}")
        tmp.write_text(updated)
        tmp.replace(DIR8)
        return got

    return with_flock(DIR8_LOCK, edit)


def update_queue(status):
    def edit():
        fd = os.open(QUEUE, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with os.fdopen(os.dup(fd), "r+", encoding="utf-8") as handle:
                raw = handle.read()
                token = '"id": "0440_happiny"'
                at = raw.find(token)
                if at < 0:
                    raise SystemExit("queue has no 0440_happiny")
                obj_start = raw.rfind("{", 0, at)
                depth = 0
                obj_end = None
                for i in range(obj_start, len(raw)):
                    if raw[i] == "{":
                        depth += 1
                    elif raw[i] == "}":
                        depth -= 1
                        if depth == 0:
                            obj_end = i + 1
                            break
                if obj_end is None:
                    raise SystemExit("unclosed happiny queue object")
                block = raw[obj_start:obj_end]
                if f'"status": "{status}"' in block and '"claim"' not in block:
                    return
                if '"status": "in_progress"' not in block and f'"status": "{status}"' not in block:
                    raise SystemExit(f"0440_happiny status is not in_progress: {block[:180]}")
                block2 = block.replace('"status": "in_progress"', f'"status": "{status}"', 1)
                block2 = re.sub(r',\s*"claim":\s*"parallel-batch"', "", block2, count=1)
                if '"claim"' in block2:
                    raise SystemExit("claim was not removed")
                json.loads(block2)
                updated = raw[:obj_start] + block2 + raw[obj_end:]
                if updated[:obj_start] != raw[:obj_start] or updated[obj_start + len(block2):] != raw[obj_end:]:
                    raise SystemExit("queue splice touched another item")
                data = json.loads(updated)
                hit = [item for item in data["items"] if item.get("id") == KEY]
                if len(hit) != 1 or hit[0].get("status") != status or "claim" in hit[0]:
                    raise SystemExit("queue status did not stick")
                before = json.loads(raw)
                for b, a in zip(before["items"], data["items"]):
                    if b.get("id") == KEY:
                        continue
                    if b != a:
                        raise SystemExit(f"queue neighbor changed: {b.get('id')}")
                tmp = QUEUE.with_suffix(f".json.tmp.{os.getpid()}")
                tmp.write_text(updated, encoding="utf-8")
                tmp.replace(QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def qa_report(master, sheets, palette, ratios, sprite_h):
    opaque = master[..., 3] == 255
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    off = []
    if np.any(opaque):
        cols = np.unique(master[opaque][:, :3], axis=0)
        pal = {tuple(int(c) for c in row) for row in palette}
        off = [tuple(int(c) for c in row) for row in cols if tuple(int(c) for c in row) not in pal]
    partial = int(((master[..., 3] > 0) & (master[..., 3] < 255)).sum())
    clip = 0
    for act in ACTIONS:
        for di in range(8):
            for f in range(FRAMES[act]):
                tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = tile[..., 3] >= 128
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip += 1
    spin_body = []
    for f in range(5, 11):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        spin_body.append({
            "f": f,
            "main": round(largest_fraction(tile), 4),
            "egg": round(white_in_main(tile), 4),
            "ratio": round(ratios[f], 4),
        })
    rest = sheets["idle"][:CELL_H, :CELL_W]
    hurt = sheets["hurt"][:CELL_H, 5 * CELL_W:6 * CELL_W]

    def width(tile):
        xs = np.nonzero(tile[..., 3] >= 128)[1]
        return int(xs.max() - xs.min() + 1) if len(xs) else 0

    hurt_squash = width(hurt) < int(width(rest) * 0.75)
    bad_ratio = [round(r, 4) for r in ratios if r < 0.90 or r > 1.08]
    idle0 = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(idle0[..., 3] >= 128)[0]
    h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    return {
        "alphas": alphas,
        "partial_alpha": partial,
        "off_palette": len(off),
        "clip_frames": clip,
        "sprite_h": h,
        "hurt_squash": bool(hurt_squash),
        "hurt_w": width(hurt),
        "rest_w": width(rest),
        "bad_ratio": bad_ratio,
        "spin_body": spin_body,
        "h_matches": h == sprite_h,
    }


def soles_hold(img, pts):
    return all(img[y, x, 3] == 255 for x, y in pts)


def sheet_from_rig(rig, palette):
    sheets = {}
    for act in ACTIONS:
        sheet = np.zeros((CELL_H, FRAMES[act] * CELL_W, 4), np.uint8)
        for f in range(FRAMES[act]):
            sheet[0:CELL_H, f * CELL_W:(f + 1) * CELL_W] = synth(rig, act, f, palette)
        sheets[act] = sheet
    return sheets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front_path = SRC_DIR / "front.png"
    front_bytes = front_path.read_bytes()
    front = np.array(Image.open(front_path).convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    if sorted(int(v) for v in np.unique(front[..., 3])) != [0, 255]:
        raise SystemExit("still alpha is not already 0 or 255")
    palette = extract_palette(front)
    require_colors(palette)
    if len(palette) != 9:
        raise SystemExit(f"expected 9 colours, got {len(palette)}")
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("feet 2 facing left egg-in-torso parts", counts, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    body_n = int((rigs["W"]["rest"][..., 3] == 255).sum())
    if body_n != 687:
        raise SystemExit(f"placed still opaque {body_n}")
    for rig in rigs.values():
        rig["counts"]["body"] = body_n

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    save_parts(front, masks, work / "parts.png")

    rest = synth(rigs["W"], "idle", 0, palette)
    if np.any(rest != rigs["W"]["rest"]):
        raise SystemExit("idle frame 0 does not match the still")
    for d, rig in rigs.items():
        idle0 = synth(rig, "idle", 0, palette)
        if np.any(idle0 != rig["rest"]):
            raise SystemExit(f"idle 0 drifted on {d}")
        for f in range(16):
            idle = synth(rig, "idle", f, palette) if f else idle0
            for name in ("foot_l", "foot_r"):
                if not soles_hold(idle, rig["soles"][name]):
                    raise SystemExit(f"{name} left the ground on idle {d} {f}")
    if max(pose("hurt", f)["recoil"] for f in range(12)) > 38.001:
        raise SystemExit("recoil exceeds 38")

    if args.samples:
        sheets = sheet_from_rig(rigs["S"], palette)
        contact_sheet(sheets, 0).save(work / "samples_S.png")
        sheets_w = sheet_from_rig(rigs["W"], palette)
        contact_sheet(sheets_w, 0).save(work / "samples_W.png")
        counts_a, ratios = south_attack_ratios(sheets)
        print("ATTACK_S_RATIOS", flush=True)
        for f, (c, r) in enumerate(zip(counts_a, ratios)):
            tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            print(
                f"attack S {f:02d} opaque {c} ratio {r:.4f} main {largest_fraction(tile):.3f} egg {white_in_main(tile):.3f}",
                flush=True,
            )
        print(f"min {min(ratios):.4f} max {max(ratios):.4f}", flush=True)
        bad = [round(r, 4) for r in ratios if r < 0.90 or r > 1.08]
        if bad:
            raise SystemExit(f"attack opaque ratio out of range at {bad}")
        bad_spin = []
        for f in range(5, 11):
            tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            main_f = largest_fraction(tile)
            egg_f = white_in_main(tile)
            if main_f < 0.90 or egg_f < 0.90:
                bad_spin.append((f, round(main_f, 3), round(egg_f, 3)))
        if bad_spin:
            raise SystemExit(f"spin is not one body {bad_spin}")
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
        for direction in DIRS:
            key = f"{act}/{direction}"
            if key in state["done"]:
                continue
            od = frames_dir / act / direction
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[direction], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    write_outputs(sheets)
    preview = contact_sheet(sheets, 0)
    preview.save(OUT_DIR / "preview.png")
    (SRC_DIR / "actions").mkdir(parents=True, exist_ok=True)
    preview.save(SRC_DIR / "actions" / "preview.png")
    motion = contact_sheet(sheets, DIRS.index("W"))
    motion.save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    counts_a, ratios = south_attack_ratios(sheets)
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(counts_a, ratios)):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        print(
            f"attack S {f:02d} opaque {c} ratio {r:.4f} main {largest_fraction(tile):.3f} egg {white_in_main(tile):.3f}",
            flush=True,
        )
    print(f"min {min(ratios):.4f} max {max(ratios):.4f} h {sprite_h}", flush=True)

    master = np.concatenate([sheets[a] for a in ACTIONS], axis=0)
    report = qa_report(master, sheets, palette, ratios, sprite_h)
    (work / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)

    bad_spin = [s for s in report["spin_body"] if s["main"] < 0.90 or s["egg"] < 0.90]
    failed = (
        report["off_palette"] or report["clip_frames"] or report["partial_alpha"]
        or report["alphas"] != [0, 255] or report["bad_ratio"] or bad_spin
        or report["hurt_squash"] or not report["h_matches"] or sprite_h != 42
    )
    if front_path.read_bytes() != front_bytes:
        raise SystemExit("front.png was overwritten")

    meta = {
        "generator": "forge_happiny_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 440,
        "variant": "happiny",
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "unflipped": sorted(UNFLIPPED),
        "still_faces": "left",
        "feet": 2,
        "mouth": False,
        "egg": "inside torso pouch",
        "parts": counts,
        "palette_colors": int(len(palette)),
        "off_palette_colors": report["off_palette"],
        "attack_s_ratios": [round(r, 4) for r in ratios],
        "attack_s_opaque": counts_a,
        "spin_body": report["spin_body"],
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if failed:
        if args.publish:
            update_queue("needs_fix")
        raise SystemExit(f"qa failed {report} spin={bad_spin}")

    if args.publish and not args.no_sync:
        copied = 0
        SYNC_OUT.mkdir(parents=True, exist_ok=True)
        for fp in OUT_DIR.iterdir():
            if fp.is_file():
                shutil.copy2(fp, SYNC_OUT / fp.name)
                copied += 1
        publish_dir8(sprite_h)
        update_queue("done")
        arena_bytes = (SYNC_OUT / "sheet.png").stat().st_size
        print(f"synced {copied} arena_sheet_bytes {arena_bytes}", flush=True)
    print(f"h {sprite_h} cw {CELL_W} ch {CELL_H} fy {FOOT_Y}", flush=True)


if __name__ == "__main__":
    main()
