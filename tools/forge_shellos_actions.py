#!/usr/bin/env python3
"""Shellos (0422) west-sea normal. Sea slug from this front.png only.

Measured: 96x96, opaque 661, bbox x32-62 y28-64 (31x37), 14 colours, one
4-connected body. The mantle crest is y<=37 (137 px), split at x<=42 into
69 and 68. The single white eye is (42, 42), on the body, left of the
lower-body mass, so the still faces left. No mouth and no legs.

Unflipped (head screen-left): N NW W SW. Mirrored: S SE E NE, so E points
screen-right.

Idle: belly stays planted, crest sways. Frame 0 is the still.
Walk: crawl (slide + foot-pivot rock + crest sway). Not a biped stomp.
Attack: 0-4 crest flare plus rings, 5-10 one-piece in-plane 360 (no |cos|),
11-12 impact, 13-15 return.
Hurt: lean at most 38 degrees around the foot. No squash, no belly mirror.
"""

import argparse
import fcntl
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
STILL_DIR = ROOT / "monsterSprite/gen4/0422_shellos/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0422_shellos/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "shellos_v2"
KEY = "0422_shellos"
# Still faces left. These four keep the drawing. E is the mirror.
UNFLIPPED = {"N", "NW", "W", "SW"}

# Crest cut measured on this still. y<=37 is the mantle. x<=42 balances it.
CREST_Y = 37
CREST_SPLIT = 42
EYE = (42, 42)

PINK = (255, 139, 197)
YELLOW = (255, 222, 115)
HOT = (230, 82, 164)
CREAM = (238, 238, 238)

# Hinged mantle flare, degrees. Zero through the spin and at the loop end.
FLARE = [10, 16, 22, 22, 14, 0, 0, 0, 0, 0, 0, 0, 0, 8, 4, 0]
RING = [0, 1, 2, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
HURT_ENV = [0.0, 0.50, 0.82, 1.0, 0.88, 0.66, 0.42, 0.22, 0.08, 0.0, 0.0, 0.0]


def flood_count(mask):
    return int(mask.sum())


def segment(img):
    """Mantle above the face, split once. Everything else is the slug body."""
    op = img[..., 3] >= 128
    h, w = op.shape
    yy, xx = np.mgrid[0:h, 0:w]
    crest = op & (yy <= CREST_Y)
    crest_l = crest & (xx <= CREST_SPLIT)
    crest_r = crest & (xx > CREST_SPLIT)
    body = op & ~crest
    parts = {"crest_l": crest_l, "crest_r": crest_r, "body": body}
    own = np.zeros_like(op)
    overlap = 0
    for name in parts:
        overlap += int((own & parts[name]).sum())
        own |= parts[name]
    if int(own.sum()) != int(op.sum()) or overlap:
        raise SystemExit(
            f"masks do not partition the still: covered {int(own.sum())} opaque {int(op.sum())} overlap {overlap}"
        )
    if flood_count(crest_l) < 40 or flood_count(crest_r) < 40 or flood_count(body) < 400:
        raise SystemExit(f"a part collapsed: {[flood_count(parts[n]) for n in parts]}")
    ex, ey = EYE
    if not body[ey, ex]:
        raise SystemExit("the white eye is not on the body")
    if crest[ey, ex]:
        raise SystemExit("the crest cut swallowed the eye")
    return parts


def measure_face(img):
    """Eye left of the rear mass: the still faces left. No mouth is present."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    ex, ey = EYE
    if not op[ey, ex] or not np.all(img[ey, ex, :3] == 255):
        raise SystemExit(f"expected the white eye at {EYE}")
    lower = op & (yy >= 48)
    lys, lxs = np.nonzero(lower)
    rear_x = float(lxs.mean())
    if rear_x < ex + 2:
        raise SystemExit(f"rear mass is not right of the eye ({rear_x:.1f}); facing must be remeasured")
    ys, xs = np.nonzero(op)
    return {
        "eye": [ex, ey],
        "rear_x": round(rear_x, 2),
        "centroid_x": round(float(xs.mean()), 2),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "faces": "left",
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
    """4x nearest in-plane rotate. A multiple of 360 stays exact. No |cos| squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
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
    """Drop only pixels whose four neighbours are all empty. After rotation only."""
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
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    for i in range(0, len(ty), 400):
        sl = slice(i, i + 400)
        d = (ty[sl, None] - pts[None, :, 0]) ** 2 + (tx[sl, None] - pts[None, :, 1]) ** 2
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
    return out


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def socket_pivot(part, inward):
    ys, xs = np.nonzero(part)
    if len(ys) == 0:
        return 0.0, 0.0
    bottom = ys.max()
    sel = ys >= bottom - 2
    x = float(xs[sel].max() if inward > 0 else xs[sel].min())
    return x, float(ys[sel].mean())


def side_of(mask, center_x):
    xs = np.nonzero(mask)[1]
    return -1.0 if float(xs.mean()) < center_x else 1.0


def build_rig(img, masks, flip):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
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

    # One row of body colour under the mantle hinge, hidden while the crest is down.
    crest = placed["crest_l"] | placed["crest_r"]
    cys = np.nonzero(crest)[0]
    edge = int(cys.max())
    yy = np.arange(CELL_H)[:, None]
    zone = crest & (yy >= edge)
    fill = nearest_color(canvas, zone, placed["body"])
    write = zone & (layers["body"][..., 3] < 128)
    layers["body"][write] = fill[write]

    pivots = {
        "crest_l": socket_pivot(placed["crest_l"], inward=1),
        "crest_r": socket_pivot(placed["crest_r"], inward=-1),
    }
    both = crest
    bys, bxs = np.nonzero(both)
    bottom = int(bys.max())
    sel = bys >= bottom - 1
    shared = (float(bxs[sel].mean()), float(bottom))

    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    foot_y = int(oys.max())
    foot_sel = oys >= foot_y - 1
    foot = (float(oxs[foot_sel].mean()), float(foot_y))
    fys, fxs = np.nonzero(crest)
    face = (float(fxs.mean()), float(fys.mean()))

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        placed = {k: v[:, ::-1].copy() for k, v in placed.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        shared = (CELL_W - 1 - shared[0], shared[1])
        center = (CELL_W - 1 - center[0], center[1])
        foot = (CELL_W - 1 - foot[0], foot[1])
        face = (CELL_W - 1 - face[0], face[1])
        canvas = canvas[:, ::-1].copy()

    sides = {name: side_of(placed[name], center[0]) for name in ("crest_l", "crest_r")}
    # forward is the screen direction the head points. mirror negates angles on the flip.
    forward = 1.0 if flip else -1.0
    mirror = -1.0 if flip else 1.0
    return {
        "layers": layers,
        "pivots": pivots,
        "shared": shared,
        "sides": sides,
        "center": center,
        "foot": foot,
        "face": face,
        "forward": forward,
        "mirror": mirror,
        "canvas": canvas,
    }


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(dx=0.0, dy=0.0, sway=0.0, open=0.0, spin=0.0, rock=0.0, recoil=0.0, ring=-1, burst=0, sparks=0)
    if action == "idle":
        p["sway"] = wave(f, n, 16.0)
    elif action == "walk":
        p["dx"] = wave(f, n, 3.0, cycles=2.0)
        p["rock"] = wave(f, n, 4.0, cycles=2.0, phase=0.9)
        p["sway"] = wave(f, n, 12.0, cycles=2.0, phase=0.4)
    elif action == "attack":
        p["open"] = float(FLARE[f])
        p["ring"] = RING[f]
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
        if f in (11, 12):
            p["burst"] = 2 if f == 11 else 1
    else:
        env = HURT_ENV[f]
        p["recoil"] = 38.0 * env
        p["dx"] = 4.0 * env
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def compose(rig, p):
    """Body stays one piece. The mantle hinges at its base. The foot is the crawl pivot."""
    dx = p["dx"] * rig["forward"]
    dy = p["dy"]
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(body, shift(rig["layers"]["body"], dx, dy))
    for name in ("crest_l", "crest_r"):
        side = rig["sides"][name]
        ang = rig["mirror"] * p["sway"] + side * p["open"]
        pivot = rig["shared"] if abs(p["open"]) < 0.45 else rig["pivots"][name]
        spun = rotate_nearest(rig["layers"][name], ang, pivot)
        paste(body, shift(spun, dx, dy))
    if abs(p["rock"]) > 0.45:
        fpx = rig["foot"][0] + dx
        fpy = rig["foot"][1] + dy
        body = rotate_nearest(body, p["rock"] * rig["mirror"], (fpx, fpy))
        body = despeckle(body)
    elif abs(p["sway"]) > 0.45 or abs(p["open"]) > 0.45:
        body = despeckle(body)
    return body


def neutral_pose():
    return dict(dx=0.0, dy=0.0, sway=0.0, open=0.0, spin=0.0, rock=0.0, recoil=0.0, ring=-1, burst=0, sparks=0)


def roar_rings(out, face, age):
    """Rings outside the mantle. No mouth, so they leave the crest."""
    cx, cy = face
    cols = (YELLOW, PINK, HOT)
    radii = (14, 18, 22)
    steps = (30, 36, 45)
    for ri in range(age + 1):
        rad = radii[ri]
        step = steps[ri]
        col = cols[ri]
        deg = 0
        while deg < 360:
            ang = math.radians(deg)
            put(out, cx + rad * math.cos(ang), cy + rad * math.sin(ang), col)
            deg += step
    return out


def impact_burst(out, face, size):
    cx, cy = face
    ticks = [(16, -2), (-16, -2), (12, -10), (-12, -10), (0, -16), (18, 4), (-18, 4), (8, -14)]
    col = YELLOW if size > 1 else CREAM
    for dx, dy in ticks[: 5 + size]:
        put(out, cx + dx, cy + dy, col)
        if size > 1:
            put(out, cx + dx, cy + dy + 1, PINK)
    return out


def sparks(out, face, forward):
    """A few flecks behind the head. Palette colours only."""
    bx = face[0] - forward * 12
    by = face[1] - 6
    for dx, dy, col in ((0, 0, YELLOW), (1, -2, PINK), (-2, 1, CREAM)):
        put(out, bx + dx, by + dy, col)
    return out


def largest_fraction(img):
    a = img[..., 3] >= 128
    if not np.any(a):
        return 1.0, 0
    seen = np.zeros_like(a)
    h, w = a.shape
    best = 0
    total = int(a.sum())
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
        if n > best:
            best = n
    return best / total, total


def synth(rig, action, f, palette, effects=True):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        body = compose(rig, neutral_pose())
        if recoiling:
            screen_dx = -p["dx"] * rig["forward"]
            body = shift(body, screen_dx, p["dy"])
            pivot = (rig["foot"][0] + screen_dx, rig["foot"][1] + p["dy"])
            # Lean backward around the foot. Peak is 38 degrees. Not a mirror.
            body = rotate_nearest(body, -p["recoil"] * rig["forward"], pivot)
        else:
            body = rotate_nearest(body, p["spin"], rig["center"])
        body = despeckle(body)
    else:
        body = compose(rig, p)

    img = snap(body, palette)
    if not effects:
        return img
    if p["ring"] >= 0 and not spinning:
        img = roar_rings(img, rig["face"], p["ring"])
    if p["burst"]:
        img = impact_burst(img, rig["face"], p["burst"])
    if p["sparks"]:
        img = sparks(img, rig["face"], rig["forward"])
    return snap(img, palette)


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
    samples = {"idle": (0, 4, 8, 12), "walk": (0, 4, 8, 12), "attack": (2, 6, 8, 11), "hurt": (1, 3, 6, 10)}
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            m = tile[..., 3] >= 128
            bg[ai * CELL_H:(ai + 1) * CELL_H, si * CELL_W:(si + 1) * CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    """South row, every frame, cropped so the crawl and the spin are readable."""
    pad = 6
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            a = tile[..., 3] >= 128
            if not np.any(a):
                continue
            ys, xs = np.nonzero(a)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(CELL_H, int(ys.max()) + 1 + pad)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(CELL_W, int(xs.max()) + 1 + pad)
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


def south_attack_ratios(sheets):
    row = sheets["attack"][:CELL_H]
    counts = []
    for f in range(16):
        tile = row[:, f * CELL_W:(f + 1) * CELL_W]
        counts.append(int((tile[..., 3] >= 128).sum()))
    base = counts[0]
    ratios = [c / base for c in counts]
    return counts, ratios


def foot_bottom(img):
    ys = np.nonzero(img[..., 3] >= 128)[0]
    return int(ys.max()) if len(ys) else -1


def top_cx(img):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return 0.0
    ycut = ys.min() + 0.35 * (ys.max() - ys.min())
    top = ys <= ycut
    return float(xs[top].mean())


def mantle_width(img):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return 0
    top = ys <= ys.min() + 12
    if not np.any(top):
        return 0
    return int(xs[top].max() - xs[top].min() + 1)


def _entry_span(text, key):
    token = f'"{key}":'
    start = text.index(token)
    i = start + len(token)
    while text[i] != "{":
        i += 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return start, j + 1
    raise SystemExit(f"dir8 entry did not close: {key}")


def publish_dir8(sprite_h):
    entry = {
        "dex": 422,
        "form": "normal",
        "variant": "shellos",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0422_shellos/normal",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir",
        "nf": 16,
        "cw": CELL_W,
        "ch": CELL_H,
        "w": CELL_W,
        "h": int(sprite_h),
        "fy": int(FOOT_Y),
        "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1,
        "src": "agent-sprite-forge",
        "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    DIR8_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with DIR8_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise SystemExit("dir8.js header changed; not publishing")
        keys = re.findall(r'"(\d{4}_[^"]+)":\{', text)
        if keys.count(KEY) != 1:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise SystemExit("0422_shellos key missing or duplicated")
        idx = keys.index(KEY)
        neighbor = keys[idx + 1] if idx + 1 < len(keys) else keys[idx - 1]
        other = keys[idx - 1] if idx > 0 else neighbor
        nb_s, nb_e = _entry_span(text, neighbor)
        ot_s, ot_e = _entry_span(text, other)
        neighbor_blob = text[nb_s:nb_e]
        other_blob = text[ot_s:ot_e]
        s, e = _entry_span(text, KEY)
        text = text[:s] + f'"{KEY}":' + blob + text[e:]
        if neighbor_blob not in text or other_blob not in text:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise SystemExit("neighbor dir8 key changed")
        body = text[len(prefix):].strip()
        if body.endswith(";"):
            body = body[:-1].strip()
        json.loads(body)
        if text.count(f'"v":"{VERSION}"') != 1:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise SystemExit("dir8 version stamp collided")
        DIR8.write_text(text)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return neighbor


def update_queue(status):
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = QUEUE.read_text()
        data = json.loads(text)
        before = {item.get("id"): (item.get("status"), item.get("claim")) for item in data["items"]}
        start = text.index('"id": "0422_shellos"')
        nxt = text.find('"id":', start + 10)
        end = len(text) if nxt < 0 else nxt
        block = text[start:end]
        if '"status":' not in block:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise SystemExit("queue status missing")
        block2 = re.sub(r'"status": "[^"]*"', f'"status": "{status}"', block, count=1)
        mid, n_mid = re.subn(r'\n\s*"claim": "[^"]*",', "", block2, count=1)
        if n_mid:
            block2 = mid
        else:
            block2 = re.sub(r',\s*\n\s*"claim": "[^"]*"', "", block2, count=1)
        text = text[:start] + block2 + text[end:]
        data = json.loads(text)
        ids = [item.get("id") for item in data["items"]]
        if ids.count(KEY) != 1:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise SystemExit("queue id collision")
        for item in data["items"]:
            if item.get("id") == KEY:
                if item.get("status") != status or "claim" in item:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    raise SystemExit("queue update did not stick")
            else:
                if (item.get("status"), item.get("claim")) != before.get(item.get("id")):
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    raise SystemExit(f"queue neighbor changed: {item.get('id')}")
        QUEUE.write_text(text if text.endswith("\n") else text + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)


def qa_report(master, sheets, palette, ratios, sprite_h):
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted({int(v) for v in np.unique(master[..., 3])})
    clip = {}
    spin_stats = []
    hurt_widths = []
    rest_w = None
    for act in ACTIONS:
        sheet = sheets[act]
        for di, direction in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sheet[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{direction}:{f}")
                if act == "attack" and direction == "S" and 5 <= f <= 10:
                    frac, total = largest_fraction(tile)
                    spin_stats.append({"f": f, "opaque": total, "main": round(frac, 3)})
                if act == "hurt" and direction == "S":
                    xs = np.nonzero(c)[1]
                    width = int(xs.max() - xs.min() + 1) if len(xs) else 0
                    if f == 0:
                        rest_w = width
                    hurt_widths.append(width)
    bad_ratio = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
    squash = False
    if rest_w:
        squash = min(hurt_widths) < int(rest_w * 0.85)
    return {
        "version": VERSION,
        "sprite_h": sprite_h,
        "off_palette_colors": len(off),
        "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "attack_s_ratios": [round(r, 3) for r in ratios],
        "attack_s_min": round(min(ratios), 3),
        "attack_s_max": round(max(ratios), 3),
        "spin_body": spin_stats,
        "hurt_squash": squash,
        "hurt_peak_deg": 38,
        "rest_diff": 0,
        "_off": off,
        "_clip": clip,
        "_bad_ratio": bad_ratio,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(STILL_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    for col in (PINK, YELLOW, HOT, CREAM):
        if not np.any(np.all(palette == np.array(col, np.uint8), axis=1)):
            raise SystemExit(f"effect colour {col} is not in the still palette")
    masks = segment(front)
    face_info = measure_face(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("facing", face_info, "parts", counts, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    for d in ("W", "S"):
        rest = synth(rigs[d], "idle", 0, palette, effects=False)
        placed = rigs[d]["canvas"]
        differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
        print(f"rest-pose {d} pixels changed", int(differ.sum()), flush=True)
        if int(differ.sum()) != 0:
            raise SystemExit(f"idle frame 0 does not match the still for {d}")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {"body": (180, 180, 170), "crest_l": (40, 180, 90), "crest_r": (40, 90, 200)}
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    # Belly contact of every idle frame stays on the frame-0 foot line.
    base_foot = foot_bottom(synth(rigs["S"], "idle", 0, palette, effects=False))
    for f in range(16):
        got = foot_bottom(synth(rigs["S"], "idle", f, palette, effects=False))
        if got != base_foot:
            raise SystemExit(f"idle {f} lifted the belly: {got} vs {base_foot}")

    if args.samples:
        want = [("idle", "S", f) for f in (0, 4, 8, 12)]
        want += [("walk", "S", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in range(12)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        crops = []
        pad = 8
        for tile in tiles:
            a = tile[..., 3] >= 128
            ys, xs = np.nonzero(a)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(CELL_H, int(ys.max()) + 1 + pad)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(CELL_W, int(xs.max()) + 1 + pad)
            crops.append(tile[y0:y1, x0:x1])
        cell_h = max(c.shape[0] for c in crops)
        cell_w = max(c.shape[1] for c in crops)
        cols = 16
        rows = math.ceil(len(crops) / cols)
        sheet = np.full((rows * cell_h, cols * cell_w, 4), (15, 20, 31, 255), np.uint8)
        for i, c in enumerate(crops):
            r, col = divmod(i, cols)
            y = r * cell_h + (cell_h - c.shape[0]) // 2
            x = col * cell_w + (cell_w - c.shape[1]) // 2
            m = c[..., 3] >= 128
            sheet[y:y + c.shape[0], x:x + c.shape[1]][m] = c[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")

        base = int((synth(rigs["S"], "attack", 0, palette)[..., 3] >= 128).sum())
        print("ATTACK_S_RATIOS", flush=True)
        ratios = []
        for f in range(16):
            tile = synth(rigs["S"], "attack", f, palette)
            n = int((tile[..., 3] >= 128).sum())
            frac, _total = largest_fraction(tile)
            ratio = n / base
            ratios.append(ratio)
            print(f"attack S {f:02d} opaque {n} ratio {ratio:.3f} main {frac:.3f}", flush=True)
        print(f"min {min(ratios):.3f} max {max(ratios):.3f}", flush=True)
        w_rest = mantle_width(synth(rigs["W"], "attack", 0, palette))
        w_flare = mantle_width(synth(rigs["W"], "attack", 2, palette))
        print(f"mantle width rest {w_rest} flare {w_flare}", flush=True)
        rest_h = synth(rigs["W"], "hurt", 0, palette, effects=False)
        peak_h = synth(rigs["W"], "hurt", 3, palette, effects=False)
        print(f"hurt W top_cx rest {top_cx(rest_h):.1f} peak {top_cx(peak_h):.1f} forward {rigs['W']['forward']}", flush=True)
        rest_s = synth(rigs["S"], "hurt", 0, palette, effects=False)
        peak_s = synth(rigs["S"], "hurt", 3, palette, effects=False)
        print(f"hurt S top_cx rest {top_cx(rest_s):.1f} peak {top_cx(peak_s):.1f} forward {rigs['S']['forward']}", flush=True)
        print("samples", work / "samples.png", flush=True)
        bad = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
        if bad:
            raise SystemExit(f"attack opaque ratio out of range at {bad}")
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
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest:
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
    (STILL_DIR / "actions").mkdir(parents=True, exist_ok=True)
    preview.save(STILL_DIR / "actions" / "preview.png")
    contact = motion_contact(sheets)
    contact.save(OUT_DIR / "preview_motion.png")
    contact.save(work / "review_S.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    counts_s, ratios = south_attack_ratios(sheets)
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(counts_s, ratios)):
        print(f"attack S {f:02d} opaque {c} ratio {r:.3f}", flush=True)
    print(f"min {min(ratios):.3f} max {max(ratios):.3f} h {sprite_h}", flush=True)

    report = qa_report(master, sheets, palette, ratios, sprite_h)
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
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
    public = {k: v for k, v in report.items() if not k.startswith("_")}
    (qa / "qa.json").write_text(json.dumps(public, ensure_ascii=False, indent=1))
    print(json.dumps(public, ensure_ascii=False)[:1800], flush=True)

    bad_spin = [s for s in report["spin_body"] if s["main"] < 0.90]
    failed = (
        report["_off"] or report["_clip"] or report["alphas"] != [0, 255]
        or report["_bad_ratio"] or bad_spin or report["hurt_squash"]
    )
    if failed:
        if not args.samples:
            update_queue("needs_fix")
        raise SystemExit(
            f"qa failed off={len(report['_off'])} clip={report['_clip']} alphas={report['alphas']} "
            f"ratios={report['_bad_ratio']} spin={bad_spin} squash={report['hurt_squash']}"
        )

    meta = {
        "generator": "forge_shellos_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 422,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED),
        "still_faces": "left",
        "archetype": "sea_slug",
        "face": face_info,
        "parts": counts,
        "palette_colors": int(len(palette)),
        "off_palette_colors": 0,
        "attack_s_min": round(min(ratios), 3),
        "attack_s_max": round(max(ratios), 3),
        "source": "front.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.publish and not args.no_sync:
        copied = sync_outputs()
        neighbor = publish_dir8(sprite_h)
        update_queue("done")
        arena_bytes = (SYNC_OUT / "sheet.png").stat().st_size
        print(f"synced {copied} neighbor {neighbor} arena_sheet_bytes {arena_bytes}", flush=True)
    print(f"h {sprite_h} cw {CELL_W} ch {CELL_H} fy {FOOT_Y}", flush=True)


if __name__ == "__main__":
    main()
