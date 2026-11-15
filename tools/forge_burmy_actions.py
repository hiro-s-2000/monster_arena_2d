#!/usr/bin/env python3
"""Burmy (0412) 8-direction action sheet from the normal 96x96 front.png.

Measured on this still only (opaque 562, bbox x35-58 y26-69, 12 colours, alpha 0 or 255).
Both yellow eyes sit inside the cloak hole: (41,48), (41,50) and (48,48), (49,49), (48,50).
Eye mean x is 45.4. Body mean x is 46.9. Neither eye is on the silhouette edge, so this
is the viewer-facing south still, not a left or right profile. S SE E NE stay. N NW W SW mirror.

Parts, from this picture only:
  twig    y<=41, the single gray stick (79 px). It joins the cloak on y41-42.
  cloak   the leaf bag, the gray face hole, and the eyes. It stays on. No mouth.
  foot_l  6 brown pixels, x39-41 y65-68.
  foot_r  7 brown pixels, x50-53 y65-69.
  Those nubs are the only feet. No legs are added.

idle   : cloak and feet planted. The twig sways. Frame 0 is the still.
walk   : the two existing feet step. The cloak waddles. Not a new gait.
attack : 0-4 the twig flares plus rings, 5-10 one-piece 360 spin,
         11-12 impact, 13-15 return. South opaque stays near frame 0.
hurt   : rigid recoil of at most 36 degrees. No squash, no belly mirror.

Palette-locked. Alpha is only 0 or 255. front/back png and gif are not written.
"""

import argparse
import hashlib
import json
import math
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
STILL_DIR = ROOT / "monsterSprite/gen4/0412_burmy/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0412_burmy/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "burmy_v2"
KEY = "0412_burmy"
# Viewer-facing still. South stays unflipped so both eyes match front.png.
UNFLIPPED = {"S", "SE", "E", "NE"}

YELLOW = (255, 213, 74)
LEAF = (123, 156, 74)
ASH = (131, 131, 123)
WOOD = (139, 106, 74)
WOOD_DARK = (98, 74, 57)

# Twig flare in degrees. Frames 5-10 are the spin, so the flare is ignored there.
FLARE = [0, 8, 14, 18, 12, 0, 0, 0, 0, 0, 0, 0, 6, 3, 0, 0]
# Ring age. -1 draws none. Frame 0 and frame 15 match the still.
RING = [-1, 0, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


def segment(img):
    """Stick above the leaves, two brown nubs, and the cloak those leave behind."""
    op = img[..., 3] >= 128
    h, w = op.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = img[..., :3]
    twig = op & (yy <= 41)
    brown = op & (
        np.all(rgb == np.array(WOOD, np.uint8), axis=-1)
        | np.all(rgb == np.array(WOOD_DARK, np.uint8), axis=-1)
    )
    foot_l = brown & (xx <= 46)
    foot_r = brown & (xx >= 47)
    cloak = op & ~twig & ~brown
    parts = {"twig": twig, "cloak": cloak, "foot_l": foot_l, "foot_r": foot_r}
    own = np.zeros_like(op)
    overlap = 0
    for name in parts:
        overlap += int((own & parts[name]).sum())
        own |= parts[name]
    if int(own.sum()) != int(op.sum()) or overlap:
        raise SystemExit(
            f"masks do not partition the still: covered {int(own.sum())} opaque {int(op.sum())} overlap {overlap}"
        )
    counts = {k: int(v.sum()) for k, v in parts.items()}
    if counts["twig"] < 60 or counts["cloak"] < 400 or counts["foot_l"] < 4 or counts["foot_r"] < 4:
        raise SystemExit(f"a part collapsed: {counts}")
    if int(brown.sum()) != counts["foot_l"] + counts["foot_r"]:
        raise SystemExit("brown pixels are not both feet")
    if int(yy[brown].min()) < 65:
        raise SystemExit("brown pixels are not the lower nubs")
    return parts


def measure_face(img):
    """Both eyes inside the hole. No mouth. The still faces the viewer."""
    op = img[..., 3] >= 128
    yel = op & np.all(img[..., :3] == np.array(YELLOW, np.uint8), axis=-1)
    ys, xs = np.nonzero(yel)
    if len(xs) < 4:
        raise SystemExit(f"expected both eyes, found {len(xs)} yellow pixels")
    bys, bxs = np.nonzero(op)
    left, right = int(bxs.min()), int(bxs.max())
    if int(xs.min()) <= left + 2 or int(xs.max()) >= right - 2:
        raise SystemExit("an eye sits on the silhouette edge; facing must be remeasured")
    if int(xs.max()) - int(xs.min()) < 4:
        raise SystemExit("eyes are not a pair; remeasure facing")
    return {
        "eyes": int(len(xs)),
        "face_x": float(xs.mean()),
        "body_x": float(bxs.mean()),
        "bbox": [int(bxs.min()), int(bys.min()), int(bxs.max()), int(bys.max())],
        "mouth": 0,
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
    """4x nearest. A multiple of 360 stays exact so the spin lands on the still."""
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
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def socket_pivot(part):
    ys, xs = np.nonzero(part)
    if len(ys) == 0:
        return 0.0, 0.0
    bottom = ys.max()
    sel = ys >= bottom - 1
    return float(xs[sel].mean()), float(ys[sel].mean())


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

    # Cloak colour under the stick base, hidden while the twig is at rest.
    pm = placed["twig"]
    edge = int(np.nonzero(pm)[0].max())
    yy = np.arange(CELL_H)[:, None]
    zone = pm & (yy >= edge - 1)
    fill = nearest_color(canvas, zone, placed["cloak"])
    write = zone & (layers["cloak"][..., 3] == 0)
    layers["cloak"][write] = fill[write]

    pivots = {"twig": socket_pivot(placed["twig"])}
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    yel = (canvas[..., 3] >= 128) & np.all(canvas[..., :3] == np.array(YELLOW, np.uint8), axis=-1)
    fy, fx = np.nonzero(yel)
    face = (float(fx.mean()), float(fy.mean()))
    fys, fxs = np.nonzero(placed["foot_l"] | placed["foot_r"])
    bottom = int(fys.max())
    ground = fys >= bottom - 1
    foot_pivot = (float(fxs[ground].mean()), float(bottom))

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        placed = {k: v[:, ::-1].copy() for k, v in placed.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        center = (CELL_W - 1 - center[0], center[1])
        face = (CELL_W - 1 - face[0], face[1])
        foot_pivot = (CELL_W - 1 - foot_pivot[0], foot_pivot[1])
        canvas = canvas[:, ::-1].copy()

    sides = {name: side_of(placed[name], center[0]) for name in ("foot_l", "foot_r")}
    facing = 1.0 if not flip else -1.0
    return layers, pivots, sides, center, face, facing, canvas, foot_pivot


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, sway=0.0, flare=0.0, spin=0.0, recoil=0.0,
        ring=-1, burst=0, sparks=0, foot_l_lift=0.0, foot_r_lift=0.0,
    )
    if action == "idle":
        p["sway"] = wave(f, n, 7.0)
    elif action == "walk":
        p["sway"] = wave(f, n, 8.0, cycles=2.0, phase=1.1)
        p["dx"] = wave(f, n, 1.0, cycles=2.0, phase=0.4)
        lift = wave(f, n, 1.0, cycles=2.0)
        p["foot_l_lift"] = 3.0 * max(0.0, lift)
        p["foot_r_lift"] = 3.0 * max(0.0, -lift)
    elif action == "attack":
        p["flare"] = float(FLARE[f])
        p["ring"] = RING[f]
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
        if f in (11, 12):
            p["burst"] = 2 if f == 11 else 1
    else:
        env = [0.0, 0.35, 0.7, 1.0, 0.85, 0.62, 0.4, 0.22, 0.1, 0.0, 0.0, 0.0][f]
        p["recoil"] = 36.0 * env
        p["dx"] = -3.0 * env
        p["dy"] = -1.0 * env
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def compose(layers, pivots, sides, facing, p):
    """Cloak stays one piece. Feet only step. The twig flares around its socket."""
    bdx, bdy = p["dx"] * facing, p["dy"]
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(body, shift(layers["cloak"], bdx, bdy))
    for name, key in (("foot_l", "foot_l_lift"), ("foot_r", "foot_r_lift")):
        lift = p[key]
        outward = sides[name] if lift > 0.5 else 0.0
        paste(body, shift(layers[name], bdx + outward, bdy - lift))
    ang = facing * (p["sway"] + p["flare"])
    spun = rotate_nearest(layers["twig"], ang, pivots["twig"])
    paste(body, shift(spun, bdx, bdy))
    return body


def roar_rings(out, face, age):
    """Sparse rings outside the cloak. Age 0 is one ring, age 1 is two."""
    cx, cy = face
    specs = ((22, 12, YELLOW), (28, 10, LEAF))
    for ri in range(age + 1):
        rad, steps, col = specs[ri]
        for k in range(steps):
            ang = math.radians(k * (360.0 / steps) + ri * 12)
            put(out, cx + rad * math.cos(ang), cy + rad * math.sin(ang), col)
    return out


def impact_burst(out, face, size):
    cx, cy = face
    ticks = [(20, -2), (-20, -2), (16, 10), (-16, 10), (0, -24), (24, 6), (-24, 6), (10, -18)]
    col = YELLOW if size > 1 else ASH
    for dx, dy in ticks[: 5 + size]:
        put(out, cx + dx, cy + dy, col)
        if size > 1:
            put(out, cx + dx, cy + dy + (1 if dy >= 0 else -1), LEAF)
    return out


def sparks(out, frame_idx, face):
    cx, cy = face
    d = frame_idx
    pts = [(cx - 14 - d, cy - 28), (cx + 14 + d, cy - 26), (cx, cy - 32)]
    for sx, sy in pts:
        put(out, sx, sy, YELLOW)
        put(out, sx, sy + 1, LEAF)
    return out


def largest_fraction(img):
    a = img[..., 3] == 255
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
    layers, pivots, sides, center, face, facing, _canvas, foot_pivot = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    spinning = p["spin"] > 1.0
    recoiling = p["recoil"] > 0.4
    if spinning or recoiling:
        rest = pose(action, 0)
        rest = gen4_attack_strike.finish(rest, action, 0, __file__)

        body = compose(layers, pivots, sides, facing, rest)
        if recoiling:
            body = shift(body, p["dx"] * facing, p["dy"])
            pivot = (foot_pivot[0] + p["dx"] * facing, foot_pivot[1] + p["dy"])
            body = rotate_nearest(body, -p["recoil"] * facing, pivot)
        else:
            body = rotate_nearest(body, p["spin"] * facing, center)
        body = despeckle(body)
    else:
        body = compose(layers, pivots, sides, facing, p)
        if p["flare"] or abs(p["sway"]) > 0.45 or p["foot_l_lift"] or p["foot_r_lift"]:
            body = despeckle(body)

    img = snap(body, palette)
    if not effects:
        return img
    moved_face = (face[0] + p["dx"] * facing, face[1] + p["dy"])
    if p["ring"] >= 0 and not spinning:
        img = roar_rings(img, moved_face, p["ring"])
    if p["burst"]:
        img = impact_burst(img, face, p["burst"])
    if p["sparks"]:
        img = sparks(img, f, moved_face)
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
    """South row, every frame, cropped so the twig, feet, and spin are visible."""
    pad = 6
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            a = tile[..., 3] == 255
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
        counts.append(int((tile[..., 3] == 255).sum()))
    base = counts[0]
    ratios = [c / base for c in counts]
    return counts, ratios


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
    face_info = measure_face(front)
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("facing viewer", face_info, "parts", counts, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}

    rest = synth(rigs["S"], "idle", 0, palette)
    placed = rigs["S"][6]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")
    for act in ("walk", "attack", "hurt"):
        other = synth(rigs["S"], act, 0, palette)
        if not np.array_equal(other, rest):
            raise SystemExit(f"{act} frame 0 does not match idle frame 0")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "cloak": (82, 115, 65),
        "twig": (180, 180, 170),
        "foot_l": (139, 106, 74),
        "foot_r": (98, 74, 57),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "S", f) for f in (0, 4, 8, 12)]
        want += [("walk", "S", f) for f in (0, 2, 4, 6, 8, 10)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in (0, 1, 3, 6, 9)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        base = int((synth(rigs["S"], "attack", 0, palette)[..., 3] == 255).sum())
        print("ATTACK_S_RATIOS", flush=True)
        ratios = []
        for f in range(16):
            tile = synth(rigs["S"], "attack", f, palette)
            n = int((tile[..., 3] == 255).sum())
            frac, _total = largest_fraction(tile)
            ratio = n / base
            ratios.append(ratio)
            print(f"attack S {f:02d} opaque {n} ratio {ratio:.3f} main {frac:.3f}", flush=True)
        print(f"min {min(ratios):.3f} max {max(ratios):.3f}", flush=True)
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
    na = neutral[..., 3] == 255
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    print(f"h {sprite_h} cw {CELL_W} ch {CELL_H} fy {FOOT_Y}", flush=True)

    counts_s, ratios = south_attack_ratios(sheets)
    print("ATTACK_S_RATIOS", flush=True)
    for f, (c, r) in enumerate(zip(counts_s, ratios)):
        print(f"attack S {f:02d} opaque {c} ratio {r:.3f}", flush=True)
    print(f"min {min(ratios):.3f} max {max(ratios):.3f}", flush=True)

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted({int(v) for v in np.unique(master[..., 3])})
    clip = {}
    spin_stats = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] == 255
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and d == "S" and 5 <= f <= 10:
                    frac, total = largest_fraction(tile)
                    spin_stats.append({"f": f, "opaque": total, "main": round(frac, 3)})
    bad_ratio = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "attack_s_ratios": [round(r, 3) for r in ratios],
        "attack_s_min": round(min(ratios), 3), "attack_s_max": round(max(ratios), 3),
        "spin_body": spin_stats, "rest_diff": 0,
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
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1600], flush=True)

    bad_spin = [s for s in spin_stats if s["main"] < 0.90]
    if off or clip or alphas != [0, 255] or bad_ratio or bad_spin:
        raise SystemExit(f"qa failed off={len(off)} clip={clip} alphas={alphas} ratios={bad_ratio} spin={bad_spin}")

    meta = {
        "generator": "forge_burmy_actions", "version": VERSION, "key": KEY,
        "dex": 412, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "still_faces": "viewer",
        "face": face_info, "parts": counts,
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "attack_s_min": round(min(ratios), 3), "attack_s_max": round(max(ratios), 3),
        "source": "front.png 96x96", "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
