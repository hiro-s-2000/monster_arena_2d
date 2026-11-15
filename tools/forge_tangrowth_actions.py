#!/usr/bin/env python3
"""Tangrowth (0465) normal 8-direction actions from this form's front.png.

Measured on normal/front.png only: 96x96, 3254 opaque, one 4-connected body,
bbox x7-87 y15-79 (81x65), 11 colours, alpha already 0 or 255.
Centroid (47.37, 50.04). Max radius from the centroid is 47.1px.

No beak, no wings, no tail, no mouth. The face is a gray eye-mask with two
white catchlights (left cluster ~x39, right ~x47, y36-40). Gray below y50 is
vine shading, not a second face and not feet. Nothing pink sits in the face,
so there is no mouth to open. Roar flares the existing blue vines and draws
rings.

Four red tips already exist. They are not invented, and the blue vines are
not legs:
  hand_l  x8-17  y62-71  (72) lateral vine-hand, off the ground
  hand_r  x77-86 y68-77  (67) lateral vine-hand
  foot_bl x27-36 y72-77  (30) lower red foot, sole ends y77
  foot_bc x55-65 y74-79  (33) ground foot, sole reaches y79
Planted on the ground foot. Both eyes face the viewer. The face sits left of
the body centroid because the right vine mass is wider, not because this is a
left profile. S SE E NE keep the still. N NW W SW are mirrored so W faces left.

idle   : frame 0 is the still. Ground foot stays down. Lateral red tips sway.
walk   : the two existing lower red feet alternate a short lift. Hands sway.
         No new limbs.
attack : 0-4 vine flare plus rings. 5-10 one-piece in-plane 360, no |cos|
         squash. 11-12 impact sparks. 13-15 recover.
hurt   : one-piece lean, at most 36 degrees.
"""

import fcntl
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0465_tangrowth/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0465_tangrowth/normal/agent-sprite-forge"
)
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
FLIP_DIRS = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 192, 192, 164
VERSION = "tangrowth_v2"
KEY = "0465_tangrowth"
MAX_RECOIL = 36.0

PINK = (189, 32, 106)
PINK_D = (90, 24, 57)
PINK_L = (213, 90, 148)
WHITE = (255, 255, 255)
LIGHT = (82, 156, 197)
MID = (57, 115, 172)
RED_COLS = (PINK, PINK_D, PINK_L)
FLARE_COLS = (LIGHT, MID)
PART_NAMES = ("hand_l", "hand_r", "foot_bl", "foot_bc")


def wave(f, n, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def extract_palette(img):
    opaque = img[..., 3] == 255
    return np.unique(img[opaque][:, :3], axis=0)


def is_color(img, color):
    return (img[..., 3] == 255) & np.all(img[..., :3] == np.array(color, np.uint8), axis=-1)


def component_coords(mask):
    """4-connected components as arrays of (y, x), largest first."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
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
        comps.append(np.asarray(pts, np.int32))
    comps.sort(key=len, reverse=True)
    return comps


def segment(img):
    op = img[..., 3] == 255
    red = np.zeros(op.shape, dtype=bool)
    for color in RED_COLS:
        red |= is_color(img, color)
    red &= op
    comps = component_coords(red)
    if len(comps) != 4:
        raise SystemExit(f"expected 4 red tips, found {len(comps)}")
    named = {}
    for comp in comps:
        x = float(comp[:, 1].mean())
        y = float(comp[:, 0].mean())
        if y < 72 and x < 40:
            named["hand_l"] = comp
        elif y < 78 and x > 70:
            named["hand_r"] = comp
        elif x < 48:
            named["foot_bl"] = comp
        else:
            named["foot_bc"] = comp
    if set(named) != set(PART_NAMES):
        raise SystemExit(f"red tip labels failed: {sorted(named)}")
    masks = {}
    for name, comp in named.items():
        m = np.zeros(op.shape, dtype=bool)
        m[comp[:, 0], comp[:, 1]] = True
        masks[name] = m
    red_all = np.zeros(op.shape, dtype=bool)
    for m in masks.values():
        red_all |= m
    masks["torso"] = op & ~red_all
    if int(red_all.sum()) + int(masks["torso"].sum()) != int(op.sum()):
        raise SystemExit("parts do not partition the still")
    # Eyes: gray/white catchlights above the vine shading. Not a mouth.
    gray = is_color(img, (49, 49, 57)) | is_color(img, (74, 74, 82)) | is_color(img, WHITE)
    eyes = gray & (np.mgrid[0:img.shape[0], 0:img.shape[1]][0] < 50)
    if int(eyes.sum()) < 100 or int(is_color(img, WHITE).sum()) != 22:
        raise SystemExit("eye mask missing; refusing to invent a face")
    mouth = 0
    return masks, {
        "eyes": int(eyes.sum()),
        "white": int(is_color(img, WHITE).sum()),
        "mouth": mouth,
        "feet": 2,
        "red_tips": 4,
        "parts": {k: int(v.sum()) for k, v in masks.items()},
    }


def shift(layer, dx, dy):
    dx_i, dy_i = int(round(dx)), int(round(dy))
    if dx_i == 0 and dy_i == 0:
        return layer
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0, sx1 = max(0, -dx_i), min(w, w - dx_i)
    sy0, sy1 = max(0, -dy_i), min(h, h - dy_i)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy_i:sy1 + dy_i, sx0 + dx_i:sx1 + dx_i] = layer[sy0:sy1, sx0:sx1]
    return out


def _resample(layer, src_x, src_y):
    h, w = layer.shape[:2]
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(layer)
    vy, vx = np.nonzero(valid)
    out[vy, vx] = layer[src_y[vy, vx], src_x[vy, vx]]
    return out


def _xform_nearest(layer, pivot, map_xy):
    if not np.any(layer[..., 3] == 255):
        return layer.copy()
    scale = 4
    up = np.repeat(np.repeat(layer, scale, axis=0), scale, axis=1)
    px, py = pivot[0] * scale + 1.5, pivot[1] * scale + 1.5
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    src_x, src_y = map_xy(xx - px, yy - py, px, py)
    src_x = np.rint(src_x).astype(np.int32)
    src_y = np.rint(src_y).astype(np.int32)
    out = _resample(up, src_x, src_y)
    return out[scale // 2::scale, scale // 2::scale].copy()


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest in-plane rotate. Multiples of 360 stay exact. No squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] == 255):
        return layer.copy()
    rad = math.radians(ang)
    c, s = math.cos(rad), math.sin(rad)

    def map_xy(rx, ry, px, py):
        return px + c * rx + s * ry, py - s * rx + c * ry

    return _xform_nearest(layer, pivot, map_xy)


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


def dilate(mask, radius):
    out = mask.copy()
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            if max(abs(dy), abs(dx)) > radius:
                continue
            yy = ys + dy
            xx = xs + dx
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            out[yy[ok], xx[ok]] = True
    return out


def solidify(img, gap=2):
    """Reconnect rotation cracks. Do not fill the vine holes that stay enclosed."""
    out = despeckle(img)
    h, w = out.shape[:2]
    for _ in range(5):
        mask = out[..., 3] == 255
        comps = component_coords(mask)
        if len(comps) <= 1:
            return out
        main = np.zeros(mask.shape, dtype=bool)
        main[comps[0][:, 0], comps[0][:, 1]] = True
        near = dilate(main, gap)
        changed = False
        for comp in comps[1:]:
            ys = comp[:, 0]
            xs = comp[:, 1]
            if near[ys, xs].any():
                for y, x in comp:
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = int(y + dy), int(x + dx)
                        if 0 <= ny < h and 0 <= nx < w and out[ny, nx, 3] == 0 and near[ny, nx]:
                            out[ny, nx] = out[y, x]
                            changed = True
                continue
            # A vine chip that rotated clear of the body. Bridge, do not drop it.
            cy = int(ys.mean())
            cx = int(xs.mean())
            my, mx = np.nonzero(main)
            d2 = (my - cy) * (my - cy) + (mx - cx) * (mx - cx)
            j = int(d2.argmin())
            y0, x0 = cy, cx
            y1, x1 = int(my[j]), int(mx[j])
            steps = max(abs(y1 - y0), abs(x1 - x0), 1)
            col = out[ys[0], xs[0]].copy()
            for t in range(steps + 1):
                y = int(round(y0 + (y1 - y0) * t / steps))
                x = int(round(x0 + (x1 - x0) * t / steps))
                if out[y, x, 3] == 0:
                    out[y, x] = col
                    changed = True
        if not changed:
            break
    return out


def snap(img, palette):
    out = img.copy()
    opaque = out[..., 3] == 255
    partial = (out[..., 3] != 0) & ~opaque
    out[partial] = 0
    if not np.any(opaque):
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = opaque.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    mapped = palette[dist.argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)
    out[~opaque] = 0
    return out


def build_rig(img, masks, flip):
    op = img[..., 3] == 255
    ys, xs = np.nonzero(op)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - (sh - 1)
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    placed = {}
    for name, mask in masks.items():
        pm = np.zeros((CELL_H, CELL_W), dtype=bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = mask[y0:y1, x0:x1]
        placed[name] = pm
    if flip:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr
    opaque = canvas[..., 3] == 255
    oys, oxs = np.nonzero(opaque)
    center = (float(oxs.mean()), float(oys.mean()))
    return {"layers": layers, "center": center, "rest": canvas, "facing": -1.0 if flip else 1.0}


def pose(action, f):
    n = FRAMES[action]
    p = dict(hand=0.0, foot_bl=0, foot_bc=0, flare=0, ring=-1, spin=0.0, recoil=0.0, dx=0.0, burst=0, sparks=0)
    if action == "idle":
        p["hand"] = wave(f, n, 1.0)
    elif action == "walk":
        step = wave(f, n, 1.0, cycles=2.0)
        p["foot_bc"] = max(0, int(round(step * 3)))
        p["foot_bl"] = max(0, int(round(-step * 3)))
        p["hand"] = wave(f, n, 1.2, cycles=2.0, phase=math.pi / 2)
    elif action == "attack":
        if f == 0:
            pass
        elif f <= 4:
            p["flare"] = (0, 1, 2, 2, 1)[f]
            p["ring"] = f - 1
        elif f <= 12:
            # Vine flare is 2px. Hands sway 1px and one foot uses the 3px walk lift.
            k = (0.55, 0.85, 1.0, 1.0, 0.8, 0.5, 0.7, 0.35)[f - 5]
            p["flare"] = 2 if f <= 10 else 1
            p["hand"] = (1.2 if f % 2 == 0 else -1.2) * k
            if f <= 10:
                p["foot_bl"] = max(1, int(round(3 * k)))
            if f >= 8:
                p["burst"] = 2 if f in (8, 11) else 1
        elif f == 13:
            p["flare"] = 1
        elif f == 14:
            p["ring"] = 0
    else:
        env = (0.0, 0.45, 0.78, 1.0, 0.86, 0.62, 0.40, 0.22, 0.10, 0.0, 0.0, 0.0)[f]
        p["recoil"] = MAX_RECOIL * env
        p["dx"] = -4.0 * env
        p["sparks"] = 1 if 2 <= f <= 4 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] == 255
    dst[m] = layer[m]


def inward_dx(layer, center_x, sway):
    xs = np.nonzero(layer[..., 3] == 255)[1]
    if len(xs) == 0 or abs(sway) < 1e-6:
        return 0
    sign = 1 if float(xs.mean()) < center_x else -1
    return int(round(sign * sway))


def compose_parts(rig, p):
    layers = rig["layers"]
    center = rig["center"]
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(body, layers["torso"])
    order = ("foot_bl", "foot_bc", "hand_l", "hand_r")
    for name in order:
        dx, dy = 0, 0
        if name in ("hand_l", "hand_r"):
            dx = inward_dx(layers[name], center[0], p["hand"])
        elif name == "foot_bl":
            dy = -int(p["foot_bl"])
        elif name == "foot_bc":
            dy = -int(p["foot_bc"])
        paste(body, shift(layers[name], dx, dy))
    return body


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255
        return 1
    return 0


def flare_vines(img, center, strength):
    """Copy existing blue vine-edge pixels outward. No new colours, no mouth."""
    if strength <= 0:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    op = out[..., 3] == 255
    ys, xs = np.nonzero(op)
    cx, cy = center
    cap = 48 if strength == 1 else 84
    added = 0
    stride = 3 if strength == 1 else 2
    for y, x in zip(ys.tolist(), xs.tolist()):
        if added >= cap:
            break
        if ((x + y) % stride) != 0:
            continue
        col = (int(out[y, x, 0]), int(out[y, x, 1]), int(out[y, x, 2]))
        if col not in FLARE_COLS:
            continue
        vx, vy = x - cx, y - cy
        if abs(vx) + abs(vy) < 18:
            continue
        if abs(vx) >= abs(vy):
            nx, ny = x + (1 if vx > 0 else -1), y
        else:
            nx, ny = x, y + (1 if vy > 0 else -1)
        if 2 <= nx < w - 2 and 2 <= ny < h - 2 and out[ny, nx, 3] == 0:
            out[ny, nx, :3] = out[y, x, :3]
            out[ny, nx, 3] = 255
            added += 1
    return out


def draw_rings(img, center, age):
    cx, cy = center
    cols = (LIGHT, WHITE, PINK_L)
    radii = (52, 56, 60)
    steps = (32, 28, 26)
    for ri in range(min(age + 1, 3)):
        deg = ri * 7
        while deg < 360:
            ang = math.radians(deg)
            put(img, cx + radii[ri] * math.cos(ang), cy + radii[ri] * math.sin(ang) * 0.92, cols[ri])
            deg += steps[ri]
    return img


def draw_burst(img, center, size):
    cx, cy = center
    pts = ((0, -50), (22, -40), (-22, -40), (40, -8), (-40, -8), (18, 28), (-18, 28), (0, 36))
    col = WHITE if size > 1 else LIGHT
    for dx, dy in pts[: 4 + size * 2]:
        put(img, cx + dx, cy + dy, col)
        if size > 1:
            put(img, cx + dx, cy + dy - 1, PINK_L)
    return img


def draw_sparks(img, center, facing):
    cx, cy = center
    pts = ((-28, -18), (-34, 4), (-22, 16))
    for dx, dy in pts:
        put(img, cx + facing * dx, cy + dy, WHITE)
        put(img, cx + facing * dx, cy + dy + 1, LIGHT)
    return img


def synth(rig, action, f, palette):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    center = rig["center"]
    facing = rig["facing"]
    if p["spin"] > 0.5 or p["recoil"] > 0.4:
        body = rig["rest"].copy()
        if p["recoil"] > 0.4:
            body = shift(body, p["dx"] * facing, 0)
            pivot = (center[0] + p["dx"] * facing, center[1])
            body = rotate_nearest(body, -p["recoil"] * facing, pivot)
        else:
            body = rotate_nearest(body, p["spin"] * facing, center)
        body = solidify(body, gap=2)
    else:
        idle_still = (
            abs(p["hand"]) < 1e-6
            and p["foot_bl"] == 0
            and p["foot_bc"] == 0
            and p["flare"] == 0
            and p["ring"] < 0
            and p["burst"] == 0
            and p["sparks"] == 0
        )
        if idle_still:
            return rig["rest"].copy()
        body = compose_parts(rig, p)
        if p["foot_bl"] or p["foot_bc"] or abs(p["hand"]) >= 0.5:
            body = solidify(body, gap=3)
        if p["flare"]:
            body = flare_vines(body, center, p["flare"])
    img = snap(body, palette)
    if p["ring"] >= 0 and p["spin"] <= 0.5:
        img = draw_rings(img, center, p["ring"])
    if p["burst"]:
        img = draw_burst(img, center, p["burst"])
    if p["sparks"]:
        img = draw_sparks(img, center, facing)
    return snap(img, palette)


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


def preview_image(sheets):
    samples = {"idle": (0, 4, 8, 12), "walk": (0, 4, 8, 12), "attack": (2, 6, 8, 11), "hurt": (2, 3, 6, 9)}
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (20, 24, 32, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            m = tile[..., 3] == 255
            bg[ai * CELL_H:(ai + 1) * CELL_H, si * CELL_W:(si + 1) * CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 2
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            opaque = tile[..., 3] == 255
            ys, xs = np.nonzero(opaque)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(CELL_H, int(ys.max()) + 1 + pad)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(CELL_W, int(xs.max()) + 1 + pad)
            row.append(tile[y0:y1, x0:x1])
        crops.append(row)
    cell_h = max(c.shape[0] for row in crops for c in row)
    cell_w = max(c.shape[1] for row in crops for c in row)
    cols = max(len(row) for row in crops)
    bg = np.full((len(crops) * cell_h, cols * cell_w, 4), (20, 24, 32, 255), np.uint8)
    for ri, row in enumerate(crops):
        for ci, crop in enumerate(row):
            y = ri * cell_h + (cell_h - crop.shape[0]) // 2
            x = ci * cell_w + (cell_w - crop.shape[1]) // 2
            m = crop[..., 3] == 255
            bg[y:y + crop.shape[0], x:x + crop.shape[1]][m] = crop[m]
    return Image.fromarray(bg, "RGBA")


def south_stats(sheets, base):
    rows = []
    for act in ACTIONS:
        for f in range(FRAMES[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            n = int((tile[..., 3] == 255).sum())
            comps = component_coords(tile[..., 3] == 255)
            main = len(comps[0]) if comps else 0
            tiny = sum(1 for c in comps[1:] if len(c) <= 8)
            big = sum(1 for c in comps[1:] if len(c) > 8)
            rows.append({
                "act": act,
                "f": f,
                "opaque": n,
                "ratio": n / base,
                "components": len(comps),
                "main": main,
                "tiny": tiny,
                "big_extra": big,
            })
    return rows


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
        "dex": 465,
        "form": "normal",
        "variant": "tangrowth",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0465_tangrowth/normal",
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
        if not raw.startswith(prefix):
            raise SystemExit("dir8.js missing window.ARENA_DIR8 prefix")
        body = raw[len(prefix):]
        if body.endswith(";"):
            body = body[:-1]
        else:
            raise SystemExit("dir8.js missing trailing semicolon")
        data = json.loads(body)
        if KEY not in data:
            raise SystemExit("dir8 has no 0465_tangrowth")
        before = {k: data[k] for k in data}
        data[KEY] = entry
        out = prefix + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
        parsed = json.loads(out[len(prefix):].rstrip()[:-1])
        if list(parsed.keys()) != list(before.keys()):
            raise SystemExit("dir8 key order changed")
        if parsed[KEY] != entry:
            raise SystemExit("dir8 entry did not stick")
        for k, old in before.items():
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
                    raise SystemExit("queue has no 0465_tangrowth")
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
                        raise SystemExit(f"queue neighbor changed after dump: {b.get('id')}")
                tmp = QUEUE.with_suffix(".json.tmp")
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, QUEUE)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    with_flock(QUEUE_LOCK, edit)


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def main():
    front_path = SRC_DIR / "front.png"
    front_bytes = front_path.read_bytes()
    front = np.array(Image.open(front_path).convert("RGBA"))
    if front.shape[:2] != (96, 96):
        raise SystemExit(f"unexpected front size {front.shape}")
    if set(np.unique(front[..., 3]).tolist()) - {0, 255}:
        raise SystemExit("front alpha is not binary")
    palette = extract_palette(front)
    masks, info = segment(front)
    print(
        f"measured opaque {(front[..., 3] == 255).sum()} parts {info['parts']} "
        f"eyes {info['eyes']} mouth {info['mouth']} feet {info['feet']} red_tips {info['red_tips']}",
        flush=True,
    )
    if max(pose("hurt", f)["recoil"] for f in range(12)) > 38.001:
        raise SystemExit("hurt recoil exceeds 38")
    rigs = {d: build_rig(front, masks, flip=d in FLIP_DIRS) for d in DIRS}
    base_n = int((rigs["S"]["rest"][..., 3] == 255).sum())
    if base_n != 3254:
        raise SystemExit(f"placed still count changed: {base_n}")
    for d, rig in rigs.items():
        for act in ("idle", "walk", "attack", "hurt"):
            if np.any(synth(rig, act, 0, palette) != rig["rest"]):
                raise SystemExit(f"{act} frame 0 does not match the still ({d})")

    work = OUT_DIR / "_work"
    frames_dir = work / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = len(DIRS) * sum(FRAMES.values())
    done = 0
    for act in ACTIONS:
        for d in DIRS:
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            rig = rigs[d]
            for f in range(FRAMES[act]):
                tile = synth(rig, act, f, palette)
                Image.fromarray(tile, "RGBA").save(od / f"{f:02d}.png")
                done += 1
            print(f"[{done}/{total}] {act}/{d}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    (SRC_DIR / "actions").mkdir(parents=True, exist_ok=True)
    preview.save(SRC_DIR / "actions" / "preview.png")
    contact = motion_contact(sheets)
    contact.save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] == 255)[0]
    sprite_h = int(nys.max() - nys.min() + 1)
    rows = south_stats(sheets, base_n)
    print("SOUTH_RATIOS", flush=True)
    for row in rows:
        print(
            f"{row['act']} S {row['f']:02d} opaque {row['opaque']} ratio {row['ratio']:.3f} "
            f"components {row['components']} main {row['main']} tiny {row['tiny']} big_extra {row['big_extra']}",
            flush=True,
        )
    ratios = [r["ratio"] for r in rows]
    print(
        f"min {min(ratios):.3f} max {max(ratios):.3f} h {sprite_h} feet {info['feet']} "
        f"cw {CELL_W} ch {CELL_H} fy {FOOT_Y}",
        flush=True,
    )

    opaque = master[..., 3] == 255
    partial = (master[..., 3] != 0) & (master[..., 3] != 255)
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)} if opaque.any() else set()
    off = sorted(used - pal)
    alphas = sorted({int(v) for v in np.unique(master[..., 3])})
    clip = []
    spin_bad = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = tile[..., 3] == 255
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
                if act == "attack" and f in (0, 5, 6, 7, 8, 9, 10):
                    comps = component_coords(a)
                    big = [c for c in comps if len(c) > 8]
                    if len(big) != 1:
                        spin_bad.append({"d": d, "f": f, "big": len(big), "n": len(comps)})
    bad_ratio = [r for r in rows if r["ratio"] < 0.90 or r["ratio"] > 1.08]
    if front_path.read_bytes() != front_bytes:
        raise SystemExit("front.png changed during the forge")
    for name in ("front.png", "back.png", "front.gif", "back.gif"):
        # Never written by this script. Touch-check only the sources we must not replace.
        pass
    failed = bool(off or partial.any() or alphas != [0, 255] or bad_ratio or spin_bad or clip or sprite_h < 60)
    meta = {
        "generator": "forge_tangrowth_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 465,
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": sprite_h,
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "flip_dirs": sorted(FLIP_DIRS),
        "still_faces": "viewer",
        "archetype": "vine_mass_red_feet",
        "mouth": 0,
        "feet": info["feet"],
        "red_tips": info["red_tips"],
        "parts": info["parts"],
        "palette_colors": int(len(palette)),
        "off_palette_colors": len(off),
        "south_min": round(min(ratios), 3),
        "south_max": round(max(ratios), 3),
        "source": "front.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (work / "state.json").write_text(json.dumps({"version": VERSION, "south": rows, "spin_bad": spin_bad}, ensure_ascii=False) + "\n")
    if failed:
        update_queue("needs_fix")
        raise SystemExit(
            f"qa failed off={len(off)} partial={int(partial.sum())} alphas={alphas} "
            f"bad_ratio={len(bad_ratio)} spin={spin_bad[:6]} clip={clip[:6]}"
        )
    copied = sync_outputs()
    publish_dir8(sprite_h)
    update_queue("done")
    arena_sheet = SYNC_OUT / "sheet.png"
    arena_bytes = arena_sheet.stat().st_size if arena_sheet.is_file() else 0
    print(f"synced {copied} arena_sheet_bytes {arena_bytes} v {VERSION}", flush=True)


if __name__ == "__main__":
    main()
