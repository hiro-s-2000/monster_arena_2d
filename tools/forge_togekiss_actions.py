#!/usr/bin/env python3
"""Togekiss normal (0468) 8-direction sheet from this form's front.png only.

Measured on normal/front.png (96x96, alpha already 0 or 255, opaque 1636,
11 colours, one 4-connected body). bbox x10-86 y26-68 (77x43). Centroid
about (46.5, 43.6), max radius 41.4.

Faces left. The beak is the left point (x10-16, y29-33), ahead of the one
red eye (x24-33, y27-36). Wings are the rear white tip (x>=74, y<=50) and
the blue marks. The bottom point (y62-68) is a cloud tail, not a planted
foot. Nothing touches a ground line. Feet = 0. Do not invent legs.

  idle   : frame 0 is the still. Hover bob. Feet stay 0.
  walk   : wing beat in place (rear wing and tail). The body stays one piece.
  attack : 0-4 the existing beak opens plus rings; 5-10 one in-plane 360
           of the whole bird including both wings (no |cos| squash);
           11-12 impact; 13-15 the still.
  hurt   : one-piece recoil, at most 38 degrees.

Palette-locked. Alpha is only 0 or 255. front/back png and gif are not written.
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
STILL_DIR = ROOT / "monsterSprite/gen4/0468_togekiss/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0468_togekiss/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left. N NW W SW keep it. S SE E NE face screen-right.
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 192, 184, 160
HOVER_GAP = 16
VERSION = "togekiss_v2"
KEY = "0468_togekiss"
MAX_RECOIL = 38.0

INK = (16, 16, 16)
WHITE = (238, 238, 255)
PALE = (189, 197, 222)
SHADOW = (74, 90, 115)
MID = (131, 156, 180)
TIP = (65, 131, 238)
LTIP = (148, 180, 255)
DBLUE = (82, 90, 172)
RED = (205, 74, 49)
DRED = (115, 49, 41)
PINK = (189, 131, 131)
EYE_COLOURS = (RED, DRED)
PARTS = ("torso", "wing_lo", "wing_up", "beak", "jaw")

SHEET_FILES = (
    "sheet.png",
    "preview.png",
    "preview_motion.png",
    "meta.json",
    "all_8dir_actions.png",
    "idle.png",
    "walk.png",
    "attack.png",
    "hurt.png",
    "idle_8dir.png",
    "walk_8dir.png",
    "attack_8dir.png",
    "hurt_8dir.png",
    "idle_S_strip.png",
    "walk_S_strip.png",
    "attack_S_strip.png",
    "hurt_S_strip.png",
)


def components(mask):
    """4-connected opaque components. Returns sizes, largest first."""
    h, w = mask.shape
    seen = np.zeros(mask.shape, bool)
    sizes = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        q = deque([(y, x)])
        seen[y, x] = True
        n = 0
        while q:
            cy, cx = q.pop()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes


def is_color(img, color):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(color, np.uint8), axis=-1)


def segment(img):
    """Beak, rear wing, cloud tail. The red eye stays on the torso. No feet."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    beak_all = op & (xx <= 16) & (yy >= 29) & (yy <= 33)
    jaw = beak_all & (yy >= 31)
    beak = beak_all & ~jaw
    wing_up = op & (xx >= 74) & (yy <= 50) & ~beak_all
    wing_lo = op & (yy >= 64) & ~beak_all & ~wing_up
    torso = op & ~beak & ~jaw & ~wing_up & ~wing_lo
    masks = {
        "torso": torso,
        "wing_lo": wing_lo,
        "wing_up": wing_up,
        "beak": beak,
        "jaw": jaw,
    }
    own = np.zeros_like(op)
    overlap = 0
    for m in masks.values():
        overlap += int((own & m).sum())
        own |= m
    if overlap or int(own.sum()) != int(op.sum()):
        raise SystemExit(f"masks do not partition overlap={overlap} covered={int(own.sum())}")
    if len(components(op)) != 1:
        raise SystemExit("still is not one body")
    if int(beak.sum()) < 8 or int(jaw.sum()) < 8:
        raise SystemExit("beak mask collapsed")
    if int(wing_up.sum()) < 40 or int(wing_lo.sum()) < 20:
        raise SystemExit("wing mask collapsed")
    return masks


def measure(img, masks):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    eye = np.zeros_like(op)
    for c in EYE_COLOURS:
        eye |= is_color(img, c) & (np.mgrid[0:img.shape[0], 0:img.shape[1]][0] <= 40)
    ey, ex = np.nonzero(eye)
    by, bx = np.nonzero(masks["beak"] | masks["jaw"])
    beak_x = int(bx.min())
    eye_x = float(ex.mean()) if len(ex) else 999.0
    info = {
        "opaque": int(op.sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "sprite_wh": [int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        "beak_x": beak_x,
        "eye_x": round(eye_x, 2),
        "eye_px": int(eye.sum()),
        "facing": "left" if beak_x < eye_x else "right",
        "feet": 0,
        "wings": 2,
        "parts": {k: int(v.sum()) for k, v in masks.items()},
        "components": len(components(op)),
    }
    if info["facing"] != "left":
        raise SystemExit(f"expected a left-facing beak, got {info}")
    if int(ys.max()) < 60:
        raise SystemExit("tail missing; still is not togekiss")
    return info


def extract_palette(img):
    opaque = img[..., 3] >= 128
    return np.unique(img[opaque][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    opaque = out[..., 3] >= 128
    if not np.any(opaque):
        out[..., :] = 0
        return out
    flat = out[..., :3].reshape(-1, 3).copy()
    sel = opaque.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    mapped = palette[np.einsum("ijk,ijk->ij", diff, diff).argmin(axis=1)]
    flat[sel] = mapped[inv]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)
    out[~opaque] = 0
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
    """4x nearest in-plane rotate. A multiple of 360 stays exact. No squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    scale = 4
    up = np.repeat(np.repeat(layer, scale, axis=0), scale, axis=1)
    px, py = pivot[0] * scale + 1.5, pivot[1] * scale + 1.5
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rad = math.radians(ang)
    c, s = math.cos(rad), math.sin(rad)
    rx, ry = xx - px, yy - py
    src_x = np.rint(px + c * rx + s * ry).astype(np.int32)
    src_y = np.rint(py - s * rx + c * ry).astype(np.int32)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    outu = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    outu[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    return outu[scale // 2::scale, scale // 2::scale].copy()


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    pad = np.pad(a, 1, constant_values=False)
    n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img):
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        pad = np.pad(a, 1, constant_values=False)
        n = (
            pad[:-2, 1:-1].astype(np.int16)
            + pad[2:, 1:-1]
            + pad[1:-1, :-2]
            + pad[1:-1, 2:]
        )
        hole = (~a) & (n == 4)
        if not np.any(hole):
            break
        above = np.roll(out, 1, axis=0)
        out[hole] = above[hole]
        out[..., 3] = np.where(out[..., 3] >= 128, 255, 0).astype(np.uint8)
        out[out[..., 3] == 0] = 0
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def build_rig(img, masks, flip):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - HOVER_GAP - (sh - 1)
    if dst_y < 8 or dst_y + sh > CELL_H - 8:
        raise SystemExit(f"hover placement leaves the cell {dst_y}")
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    layers = {}
    for name, mask in masks.items():
        lyr = np.zeros_like(canvas)
        lyr[dst_y:dst_y + sh, dst_x:dst_x + sw][mask[y0:y1, x0:x1]] = img[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
        layers[name] = lyr
    if flip:
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        canvas = np.ascontiguousarray(canvas[:, ::-1])
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    facing = 1.0 if flip else -1.0
    return layers, center, facing, canvas


def pose(action, f):
    p = dict(
        dx=0.0, dy=0, wing=0, tail=0, jaw=0, spin=0.0, hurt=0.0,
        rings=0, ring=0, burst=0, sparks=0,
    )
    if action == "idle":
        bob = int(round(math.sin(2 * math.pi * f / 16.0)))
        p["dy"] = bob
        p["wing"] = -bob
    elif action == "walk":
        s = math.sin(2 * math.pi * 2.0 * f / 16.0)
        p["dy"] = int(round(s))
        p["wing"] = -int(round(2.0 * s))
        p["tail"] = -1 if s > 0.35 else 0
    elif action == "attack":
        if f <= 4:
            p["jaw"] = (1, 2, 2, 2, 1)[f]
            p["rings"] = (1, 2, 2, 2, 1)[f]
            p["ring"] = f
            p["wing"] = (0, -1, -2, -1, 0)[f]
        elif f <= 10:
            # Wing lift is the proven 2px. Two beats, beak open. The body stays upright.
            p["jaw"] = 2
            p["wing"] = (0, -2, -2, 0, -2, -1)[f - 5]
            p["tail"] = -1 if f in (7, 8) else 0
        elif f in (11, 12):
            p["burst"] = 1
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = MAX_RECOIL * env
        p["dx"] = 4.0 * env
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    if p["hurt"] > MAX_RECOIL + 0.01:
        raise SystemExit("recoil exceeds 38")
    return p


def open_beak(img, jaw_src, bdx, bdy, drop):
    """Drop the existing lower beak. The gap is palette ink. No new mouth."""
    home = shift(jaw_src, bdx, bdy)
    dropped = shift(jaw_src, bdx, bdy + max(0, drop))
    hole = (home[..., 3] >= 128) & (dropped[..., 3] < 128) & (img[..., 3] < 128)
    img[hole, :3] = INK
    img[hole, 3] = 255
    paste(img, dropped)
    return img


def beak_tip(img, facing):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(xs) == 0:
        return CELL_W // 2, CELL_H // 2
    # The open beak is the forward extreme of the whole bird, which faces `facing`.
    if facing > 0:
        x = int(xs.max())
        y = int(np.median(ys[xs >= x - 2]))
    else:
        x = int(xs.min())
        y = int(np.median(ys[xs <= x + 2]))
    return x, y


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def roar_rings(img, origin, facing, age, strength):
    """Sparse rings in front of the beak. Not part of the body."""
    ox, oy = origin
    cols = (WHITE, TIP, RED)
    drawn = 0
    for ri in range(strength):
        rad = 10 + age * 2 + ri * 5
        col = cols[ri % 3]
        for deg in range(-36, 37, 18):
            ang = math.radians(deg)
            drawn += put(
                img,
                ox + facing * rad * math.cos(ang),
                oy + rad * math.sin(ang) * 0.55,
                col,
            )
    return img, drawn


def impact_burst(img, origin, facing):
    ox, oy = origin
    cols = (WHITE, TIP, RED, PALE)
    for i, (dx, dy) in enumerate(((8, 0), (12, -3), (12, 3), (6, -6), (6, 6), (16, 0), (10, -8), (10, 8))):
        put(img, ox + facing * dx, oy + dy, cols[i % 4])
    return img


def sparks(img, frame_idx, facing):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(xs) == 0:
        return img
    cx = int(np.median(xs))
    top = int(ys.min())
    d = frame_idx
    for sx, sy, col in (
        (cx - facing * (6 + d), top + 4, WHITE),
        (cx + facing * (4 + d), top + 8, TIP),
        (cx, top + 2, RED),
    ):
        put(img, sx, sy, col)
    return img


def compose(layers, bdx, bdy, wing, tail, jaw_drop):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(img, shift(layers["torso"], bdx, bdy))
    paste(img, shift(layers["wing_lo"], bdx, bdy + tail))
    paste(img, shift(layers["wing_up"], bdx, bdy + wing))
    paste(img, shift(layers["beak"], bdx, bdy))
    if jaw_drop:
        img = open_beak(img, layers["jaw"], bdx, bdy, jaw_drop)
    else:
        paste(img, shift(layers["jaw"], bdx, bdy))
    return img


def synth(rig, action, f, palette, effects=True):
    layers, center, facing, _canvas = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = (p["spin"] % 360.0) > 0.5
    recoiling = p["hurt"] > 0.45
    # Screen x: unflipped faces left, so backward recoil is +x.
    bdx = int(round(-facing * p["dx"]))
    bdy = int(p["dy"])
    if spinning or recoiling:
        body = compose(layers, 0, 0, 0, 0, 0)
        ang = (p["spin"] if spinning else p["hurt"]) * (-facing)
        img = rotate_nearest(body, ang, center)
        if recoiling:
            img = shift(img, bdx, 0)
        img = fill_pinholes(img)
        img = despeckle(img)
    else:
        img = compose(layers, bdx, bdy, int(p["wing"]), int(p["tail"]), int(p["jaw"]))
    img = snap(img, palette)
    if not effects:
        return img
    if p["rings"] and not spinning and not recoiling:
        img, _n = roar_rings(img, beak_tip(img, facing), facing, p["ring"], p["rings"])
    if p["burst"]:
        img = impact_burst(img, beak_tip(img, facing), facing)
    if p["sparks"] and recoiling:
        img = sparks(img, f, facing)
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
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 2, 4, 6),
        "attack": (0, 2, 4, 6, 8, 10, 11, 15),
        "hurt": (0, 3, 5, 8, 11),
    }
    cols = max(len(v) for v in samples.values())
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            y, x = ai * CELL_H, si * CELL_W
            m = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
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


def assert_claim():
    import os as _rebuild_os
    if _rebuild_os.environ.get("GEN4_ATTACK_REBUILD") == "1":
        return
    data = json.loads(QUEUE.read_text())
    for item in data["items"]:
        if item.get("id") != KEY:
            continue
        if item.get("status") != "in_progress" or item.get("claim") != "parallel-batch":
            raise SystemExit(f"stop: {KEY} is {item.get('status')} claim={item.get('claim')}")
        return
    raise SystemExit(f"stop: queue has no {KEY}")


def publish_dir8(sprite_h):
    entry = {
        "dex": 468,
        "form": "normal",
        "variant": "togekiss",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0468_togekiss/normal",
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

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        stripped = text.strip()
        if not stripped.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        body = stripped[len(prefix):]
        if not body.endswith(";"):
            raise SystemExit("dir8.js missing trailing semicolon")
        body = body[:-1]
        data = json.loads(body)
        if KEY not in data:
            raise SystemExit(f"{KEY} missing from dir8")
        before = {k: data[k] for k in data if k != KEY}
        order = list(data)
        data[KEY] = entry
        blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        check = json.loads(blob)
        if list(check) != order:
            raise SystemExit("dir8 key order changed")
        for k, old in before.items():
            if check[k] != old:
                raise SystemExit(f"refusing to touch {k}")
        if check[KEY] != entry:
            raise SystemExit("dir8 entry did not round-trip")
        if blob.count(f'"v":"{VERSION}"') != 1:
            raise SystemExit("dir8 version stamp collided")
        out = prefix + blob + ";\n"
        tmp = DIR8.with_suffix(".js.togekiss.tmp")
        tmp.write_text(out)
        written = tmp.read_text()
        payload = written.strip()[len(prefix):]
        if payload.endswith(";"):
            payload = payload[:-1]
        again = json.loads(payload)
        if again[KEY] != entry:
            raise SystemExit("dir8 reread mismatch")
        for k, old in before.items():
            if again[k] != old:
                raise SystemExit(f"dir8 wrote over {k}")
        tmp.replace(DIR8)

    with_flock(DIR8_LOCK, edit)
    return entry["v"]


def update_queue(status):
    def edit():
        fd = os.open(QUEUE, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with os.fdopen(os.dup(fd), "r", encoding="utf-8") as handle:
                text = handle.read()
            data = json.loads(text)
            before = {
                item.get("id"): (item.get("status"), item.get("claim"))
                for item in data["items"]
            }
            found = False
            for item in data["items"]:
                if item.get("id") != KEY:
                    continue
                if item.get("status") != "in_progress" or item.get("claim") != "parallel-batch":
                    raise SystemExit(f"stop: lost claim before queue write {item}")
                item["status"] = status
                item.pop("claim", None)
                found = True
                break
            if not found:
                raise SystemExit(f"queue has no {KEY}")
            for item in data["items"]:
                if item.get("id") == KEY:
                    continue
                if (item.get("status"), item.get("claim")) != before.get(item.get("id")):
                    raise SystemExit(f"queue neighbor changed: {item.get('id')}")
            payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            tmp = QUEUE.with_suffix(".json.togekiss.tmp")
            tmp.write_text(payload, encoding="utf-8")
            check = json.loads(tmp.read_text())
            ids = [item.get("id") for item in check["items"]]
            if ids.count(KEY) != 1:
                raise SystemExit("queue id collision")
            for item in check["items"]:
                if item.get("id") == KEY:
                    if item.get("status") != status or "claim" in item:
                        raise SystemExit("queue update did not stick")
                elif (item.get("status"), item.get("claim")) != before.get(item.get("id")):
                    raise SystemExit(f"queue neighbor changed after write: {item.get('id')}")
            tmp.replace(QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def opaque_count(tile):
    return int((tile[..., 3] == 255).sum())


def qa_sheet(master, sheets, palette, frames_dir, digest, sprite_h, rest_diff):
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    partial = int(((master[..., 3] != 0) & (master[..., 3] != 255)).sum())
    clip = []
    ratios = {}
    comps = {}
    for act in ACTIONS:
        row = sheets[act][:CELL_H]
        counts = []
        comp_n = []
        for f in range(FRAMES[act]):
            tile = row[:, f * CELL_W:(f + 1) * CELL_W]
            counts.append(opaque_count(tile))
            sizes = components(tile[..., 3] == 255)
            comp_n.append(len(sizes))
        base = counts[0] or 1
        ratios[act] = [c / base for c in counts]
        comps[act] = comp_n
        print(f"SOUTH {act}", flush=True)
        for f, (c, r) in enumerate(zip(counts, ratios[act])):
            extra = ""
            if act == "attack" and 5 <= f <= 10:
                sizes = components(row[:, f * CELL_W:(f + 1) * CELL_W][..., 3] == 255)
                main = (sizes[0] / sum(sizes)) if sizes else 0
                extra = f" components {len(sizes)} main {main:.3f}"
            print(f"  {act} S {f:02d} opaque {c} ratio {r:.3f} parts {comp_n[f]}{extra}", flush=True)
        print(f"  min {min(ratios[act]):.3f} max {max(ratios[act]):.3f}", flush=True)
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                a = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] == 255
                if a.size and (a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any()):
                    clip.append(f"{act}:{d}:{f}")
    spin = []
    stray = []
    atk_row = sheets["attack"][:CELL_H]
    for f in range(16):
        tile = atk_row[:, f * CELL_W:(f + 1) * CELL_W]
        sizes = components(tile[..., 3] == 255)
        total = sum(sizes) or 1
        if 5 <= f <= 10:
            spin.append({"f": f, "components": len(sizes), "main": round(sizes[0] / total, 3) if sizes else 0, "opaque": total})
        elif len(sizes) > 1 and sizes[1] > 24:
            stray.append({"f": f, "second": sizes[1]})
    report = {
        "version": VERSION,
        "hash": digest,
        "sprite_h": sprite_h,
        "rest_diff": rest_diff,
        "off_palette_colors": len(off),
        "partial_alpha": partial,
        "edge_touching": clip,
        "attack_s_ratios": [round(r, 3) for r in ratios["attack"]],
        "south_ratios": {k: [round(r, 3) for r in v] for k, v in ratios.items()},
        "spin": spin,
        "stray": stray,
        "feet": 0,
        "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    if GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        qa = OUT_DIR / "_work" / "qa"
        qa.mkdir(parents=True, exist_ok=True)
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    ok = (
        rest_diff == 0
        and not off
        and partial == 0
        and not clip
        and 40 <= sprite_h <= 48
        and all(0.90 <= r <= 1.08 for r in ratios["attack"])
        and all(0.90 <= r <= 1.08 for act in ACTIONS for r in ratios[act])
        and all(s["components"] == 1 and s["main"] >= 0.99 for s in spin)
        and all(n == 1 for n in comps["walk"])
        and all(n == 1 for n in comps["idle"])
        and not stray
    )
    report["ok"] = ok
    report["south_components"] = comps
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()

    assert_claim()
    front = np.array(Image.open(STILL_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    info = measure(front, masks)
    print(
        "parts", info["parts"],
        "palette", len(palette),
        "facing", info["facing"],
        "wh", info["sprite_wh"],
        "feet", info["feet"],
        "beak_x", info["beak_x"],
        "eye_x", info["eye_x"],
        flush=True,
    )

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    rest = synth(rigs["W"], "idle", 0, palette, effects=False)
    rest_diff = int(np.any(rest != rigs["W"][3], axis=2).sum())
    rest_s = synth(rigs["S"], "idle", 0, palette, effects=False)
    rest_s_diff = int(np.any(rest_s != rigs["S"][3], axis=2).sum())
    print("rest-pose pixels changed west", rest_diff, "south", rest_s_diff, flush=True)
    if rest_diff or rest_s_diff:
        raise SystemExit("frame 0 does not match the still")

    # Connectivity of the wing beat and the roar before any publish.
    for f in range(16):
        tile = synth(rigs["W"], "walk", f, palette, effects=False)
        sizes = components(tile[..., 3] == 255)
        if len(sizes) != 1:
            raise SystemExit(f"walk frame {f} split into {len(sizes)}")
        atk = synth(rigs["W"], "attack", f, palette)
        asz = components(atk[..., 3] == 255)
        if 5 <= f <= 10 and len(asz) != 1:
            raise SystemExit(f"spin frame {f} split into {len(asz)}")
    peaks = [pose("hurt", f)["hurt"] for f in range(12)]
    print(f"hurt recoil max {max(peaks):.2f}", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
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
                if fp.exists() and fp.stat().st_size > 0 and not args.force:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview_image(sheets).save(OUT_DIR / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][6 * CELL_H:7 * CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    report = qa_sheet(master, sheets, palette, frames_dir, digest, sprite_h, rest_diff)
    (work / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(
        f"FEET {info['feet']} HEIGHT {sprite_h} cell {CELL_W}x{CELL_H} fy {FOOT_Y} "
        f"facing {info['facing']}",
        flush=True,
    )
    meta = {
        "generator": "forge_togekiss_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 468,
        "variant": "togekiss",
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED),
        "palette_colors": int(len(palette)),
        "off_palette_colors": report["off_palette_colors"],
        "attack_s_ratios": report["attack_s_ratios"],
        "parts": info["parts"],
        "facing": info["facing"],
        "beak_x": info["beak_x"],
        "eye_x": info["eye_x"],
        "feet": 0,
        "wings": 2,
        "archetype": "avian_hover",
        "source": "front.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not report["ok"]:
        print("qa failed", json.dumps({k: report[k] for k in ("off_palette_colors", "partial_alpha", "edge_touching", "attack_s_ratios", "spin", "sprite_h")}, ensure_ascii=False), flush=True)
        if not args.no_publish:
            update_queue("needs_fix")
        raise SystemExit("qa failed; dir8 not published")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    arena_sheet = SYNC_OUT / "sheet.png"
    if arena_sheet.is_file():
        print("ARENA_SHEET_BYTES", arena_sheet.stat().st_size, flush=True)
    if not args.no_publish:
        publish_dir8(sprite_h)
        update_queue("done")
        print("published", VERSION, flush=True)
    print("DIR8_V", VERSION, flush=True)
    print("COMPONENTS_SPIN", json.dumps(report["spin"]), flush=True)


if __name__ == "__main__":
    main()
