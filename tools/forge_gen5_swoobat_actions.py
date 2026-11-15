#!/usr/bin/env python3
"""Swoobat normal (0528) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x0-94 y19-71, 1994 px, alpha 0/255):
  Two white eyes. Left cluster mean x 35.8 (4 px), right cluster mean x 48.7
  (9 px). All white mean x 44.7.
  The heart nose is blush (246,164,156) plus the rose and pink locked inside
  it, bbox x36-48 y44-53, mean x 41.8. The bottom of the heart is x40-43.
  The nose is left of the eyes, so the still faces screen-left.
  The dark pixels inside the heart are the nose shading, not an open mouth.
  The roar does not cut a wedge and does not draw a ring.
  Two ear membranes and no feet. The sprite floats.
  Screen-left ear: grey membrane x0-32 y27-50.
  Screen-right ear: grey membrane x59-94 y19-47, plus the pink inner ear
  curled under it at x65-83 y56-67. That curl is part of the right ear.
normal/back.png (bbox x0-94 y20-74, 1978 px) has the dangling heart at
  x19-36 y62-73, mean x 26.6, left of the body centroid x 44.6.
  This back also faces screen-left. No eyes, no feet. The back sole is 3px
  lower, so the back is shifted up 3px and both forms share one float line.

Directions, both stills face screen-left when unmirrored:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NW back as drawn (heart to screen-left). NE back mirrored (heart to
  screen-right).
  Attack 5-8 shows the back at equal scale. A front that faces right turns
  into N. A front that faces left turns into N flipped. No |cos| squash,
  no in-plane body spin.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen5_attack_strike import finish, save_peaks

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen5/0528_swoobat/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0528_swoobat/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0528_swoobat"
VERSION = "swoobat_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 208, 208
OFF_X = (CELL_W - 96) // 2
FRONT_FOOT = 71
BACK_FOOT = 74
# The still is 95px wide and already touches the source edge. A 38 degree
# lean needs more than 16px, so the cell keeps the whole 96 image plus slack.
OFF_Y = CELL_H - 96
FOOT_Y = OFF_Y + FRONT_FOOT
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", True),
    "N": ("back", False), "NW": ("back", False),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Angles this script already draws. finish() must not be given a larger number.
PEAKS = {"ear_l": 22.0, "ear_r": 22.0, "curl": 16.0, "heart": 12.0}
LIMBS = ("ear_l", "ear_r", "curl", "heart")

INK = (16, 16, 16)
DK = (49, 49, 49)
GREY = (74, 74, 82)
NAVY = (65, 90, 164)
TEAL = (49, 123, 148)
SLATE = (106, 123, 131)
PLUM = (82, 82, 106)
ROSE = (148, 74, 74)
PINK = (222, 139, 131)
LAV = (189, 189, 205)
BLUSH = (246, 164, 156)
ICE = (205, 238, 255)
CYAN = (24, 164, 205)
WHITE = (255, 255, 255)
PINKISH = (PINK, ROSE, BLUSH)
OUTLINE = (GREY, DK, INK)

SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)


def is_color(img, color):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(color, np.uint8), axis=-1)


def any_color(img, cols):
    m = np.zeros(img.shape[:2], bool)
    for c in cols:
        m |= is_color(img, c)
    return m


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1, constant_values=False)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def flood(allowed, seeds):
    out = seeds & allowed
    if not out.any():
        return out
    while True:
        nxt = dilate(out, 1) & allowed
        if np.array_equal(nxt, out):
            return out
        out = nxt


def shift(layer, dx, dy):
    dx, dy = int(round(dx)), int(round(dy))
    if dx == 0 and dy == 0:
        return layer.copy()
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = layer[sy0:sy1, sx0:sx1]
    return out


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest in-plane rotation. Positive follows the forge formula. No squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    sc = 4
    up = np.repeat(np.repeat(layer, sc, axis=0), sc, axis=1)
    h, w = up.shape[:2]
    px, py = pivot[0] * sc + 1.5, pivot[1] * sc + 1.5
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = xx - px, yy - py
    r = math.radians(ang)
    c, s = math.cos(r), math.sin(r)
    sx = np.rint(px + c * rx + s * ry).astype(np.int32)
    sy = np.rint(py - s * rx + c * ry).astype(np.int32)
    ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    out = np.zeros_like(up)
    out[ok] = up[sy[ok], sx[ok]]
    return out[sc // 2::sc, sc // 2::sc].copy()


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1, constant_values=False)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img):
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        p = np.pad(a, 1, constant_values=False)
        n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        hole = (~a) & (n == 4)
        if not hole.any():
            break
        up = np.zeros_like(out)
        up[1:] = out[:-1]
        out[hole] = up[hole]
    return out


def snap(img, palette):
    out = img.copy()
    op = out[..., 3] >= 128
    out[~op] = 0
    if not op.any():
        return out
    px = out[op][:, :3].astype(np.int32)
    pal = palette.astype(np.int32)
    d = ((px[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    out[op, :3] = palette[d.argmin(axis=1)]
    out[op, 3] = 255
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def put(img, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 1 <= x < w - 1 and 1 <= y < h - 1 and img[y, x, 3] < 128:
        img[y, x, :3] = col
        img[y, x, 3] = 255


def paint_cushion(layer, allowed, steps):
    """Grow this layer's own colours into allowed pixels. Hidden at rest."""
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        grown_up = np.zeros_like(src)
        grown_dn = np.zeros_like(src)
        grown_lt = np.zeros_like(src)
        grown_rt = np.zeros_like(src)
        grown_up[:-1] = src[1:]
        grown_dn[1:] = src[:-1]
        grown_lt[:, :-1] = src[:, 1:]
        grown_rt[:, 1:] = src[:, :-1]
        for grown in (grown_up, grown_dn, grown_lt, grown_rt):
            take = (acc[..., 3] < 128) & allowed & (grown[..., 3] >= 128)
            acc[take] = grown[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def component_count(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    n = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        n += 1
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
    return n


def largest_component(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    best = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        n = 0
        while stack:
            y, x = stack.pop()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if n > best:
            best = n
    return best


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def check_partition(op, masks):
    acc = np.zeros_like(op)
    for name, m in masks.items():
        if (m & acc).any():
            raise SystemExit(f"overlap in {name}")
        if (m & ~op).any():
            raise SystemExit(f"{name} left the still")
        acc |= m
    if int(acc.sum()) != int(op.sum()):
        raise SystemExit(f"parts do not partition: {int(acc.sum())} vs {int(op.sum())}")


def segment(img, kind):
    """Two ear membranes, the heart nose, and on the front the pink inner ear.

    No feet are drawn, so none are invented. The eyes stay on the torso.
    The heart's dark centre stays with the nose. It is shading, not a mouth.
    """
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    outline = any_color(img, OUTLINE)
    pinkish = any_color(img, PINKISH)
    ink = any_color(img, (INK, DK))
    if kind == "front":
        if int(np.nonzero(op)[0].max()) != FRONT_FOOT:
            raise SystemExit("front float line moved")
        seed_r = is_color(img, GREY) & (xx >= 80) & (yy <= 32)
        ear_r = flood(outline & (xx >= 56) & (yy <= 52), seed_r)
        seed_l = is_color(img, GREY) & (xx <= 16) & (yy >= 26) & (yy <= 36)
        ear_l = flood(outline & (xx <= 36) & (yy <= 52) & (yy >= 24), seed_l)
        white = is_color(img, WHITE)
        if not white.any():
            raise SystemExit("front eyes missing")
        seed_h = is_color(img, BLUSH) & (xx < 55)
        dark_heart = ink & (xx >= 38) & (xx <= 46) & (yy >= 46) & (yy <= 53)
        allowed_h = (pinkish & (xx < 58) & (yy >= 42) & (yy <= 56)) | dark_heart
        allowed_h = allowed_h & ~dilate(white, 1)
        heart = flood(allowed_h, seed_h)
        curl_core = pinkish & (xx >= 60) & (yy >= 54)
        non_ink = op & ~ink & ~curl_core
        curl_ink = ink & dilate(curl_core, 1) & ~dilate(non_ink, 1)
        curl = curl_core | curl_ink
        ey, ex = np.nonzero(white)
        hy, hx = np.nonzero(heart)
        if float(hx.mean()) >= float(ex.mean()) - 0.5:
            raise SystemExit(
                f"front does not face left: nose {float(hx.mean()):.1f} eye {float(ex.mean()):.1f}"
            )
    else:
        if int(np.nonzero(op)[0].max()) != BACK_FOOT:
            raise SystemExit("back float line moved")
        heart_core = pinkish
        non_ink = op & ~ink & ~heart_core
        heart_ink = ink & dilate(heart_core, 1) & ~dilate(non_ink, 1)
        heart = heart_core | heart_ink
        seed_r = is_color(img, GREY) & (xx >= 78)
        ear_r = flood(outline & (xx >= 52) & (yy <= 55) & ~heart, seed_r)
        seed_l = (is_color(img, GREY) | is_color(img, DK)) & (xx <= 8) & (yy <= 26)
        ear_l = flood(outline & (xx <= 34) & (yy <= 48) & ~heart, seed_l)
        curl = np.zeros_like(op)
        body = op & ~heart
        by, bx = np.nonzero(body)
        hy, hx = np.nonzero(heart)
        if float(hx.mean()) >= float(bx.mean()) - 1.0:
            raise SystemExit(
                f"back nose is not left of the body: nose {float(hx.mean()):.1f} body {float(bx.mean()):.1f}"
            )
    for name, part in (("ear_l", ear_l), ("ear_r", ear_r), ("heart", heart)):
        if int(part.sum()) < 40 or component_count(part) != 1:
            raise SystemExit(f"{kind} {name} is not one piece ({int(part.sum())})")
    if kind == "front":
        if int(curl.sum()) < 40 or component_count(curl) != 1:
            raise SystemExit(f"front curl is not one piece ({int(curl.sum())})")
    masks = {"ear_l": ear_l, "ear_r": ear_r, "heart": heart, "curl": curl}
    torso = op.copy()
    for m in masks.values():
        torso &= ~m
    masks["torso"] = torso
    # Drop an empty curl on the back so the partition stays exact.
    if kind == "back":
        del masks["curl"]
    check_partition(op, masks)
    if kind == "front" and (is_color(img, WHITE) & ~torso).any():
        raise SystemExit("front eyes left the torso")
    for name, part in masks.items():
        if name == "torso":
            continue
        if not (part & dilate(torso, 2)).any():
            raise SystemExit(f"{kind} {name} does not touch the body")
    return masks


def place(arr, y_shift):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    y0 = OFF_Y + y_shift
    x0 = OFF_X
    h, w = arr.shape[:2]
    if y0 < 0 or x0 < 0 or y0 + h > CELL_H or x0 + w > CELL_W:
        raise SystemExit(f"place overflow y {y0} x {x0}")
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def joint_band(part, torso, depth):
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 3)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def pivot_of(layer, torso_layer):
    part = layer[..., 3] >= 128
    torso = torso_layer[..., 3] >= 128
    touch = part & dilate(torso, 2)
    if not touch.any():
        touch = part & dilate(torso, 6)
    if not touch.any():
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(torso)
    cx, cy = float(tx.mean()), float(ty.mean())
    d = (xs - cx) ** 2 + (ys - cy) ** 2
    near = d <= d.min() + 8.0
    return float(xs[near].mean()), float(ys[near].mean())


def pivot_top(layer):
    """Top edge. The heart hangs from here on both the face and the back."""
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    band = ys <= int(ys.min()) + 2
    return float(xs[band].mean()), float(ys[band].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs - pivot[0]) ** 2 + (ys - pivot[1]) ** 2
    i = int(d.argmax())
    return float(xs[i]), float(ys[i])


def signs_for(layer, pivot, facing):
    """Which rotation sign lifts the tip, and which sends it toward facing."""
    if int((layer[..., 3] >= 128).sum()) < 4:
        return {"up": 1.0, "fwd": 1.0}

    def tip_after(ang):
        return tip_xy(rotate_nearest(layer, ang, pivot), pivot)

    x_pos, y_pos = tip_after(12.0)
    x_neg, y_neg = tip_after(-12.0)
    up = -1.0 if y_neg < y_pos else 1.0
    if facing > 0:
        fwd = 1.0 if x_pos >= x_neg else -1.0
    else:
        fwd = 1.0 if x_pos <= x_neg else -1.0
    return {"up": up, "fwd": fwd}


def build_view(img, masks, kind, mirror):
    y_shift = 0 if kind == "front" else (FRONT_FOOT - BACK_FOOT)
    canvas = place(img, y_shift)
    placed = {k: place(v, y_shift) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    depths = {"ear_l": 5, "ear_r": 5, "heart": 3, "curl": 4}
    for name, depth in depths.items():
        if name not in placed or int(placed[name].sum()) < 4:
            continue
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth + 1,
        )
    # Both stills face screen-left when not mirrored.
    facing = 1.0 if mirror else -1.0
    pivots = {}
    signs = {}
    for name in ("ear_l", "ear_r", "curl"):
        if name not in layers or int((layers[name][..., 3] >= 128).sum()) < 4:
            continue
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    pivots["heart"] = pivot_top(layers["heart"])
    signs["heart"] = signs_for(layers["heart"], pivots["heart"], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    names = [n for n in LIMBS if n in layers and int((layers[n][..., 3] >= 128).sum()) >= 4]
    return {
        "kind": kind, "mirror": mirror, "facing": facing,
        "names": names,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, heart_dx=0,
        ear_l=0.0, ear_r=0.0, curl=0.0, heart=0.0,
        mode="up",
        lean=0.0, knock=0, flash=False,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Float. The ears breathe. The heart nose sways. No feet.
        p["ear_l"] = wave(f, n, 5.0, phase=0.4)
        p["ear_r"] = wave(f, n, 6.0, phase=0.9)
        p["curl"] = wave(f, n, 4.0, phase=0.9)
        p["heart"] = wave(f, n, 3.0, phase=0.2)
        p["bdy"] = int(round(wave(f, n, 1.2, phase=0.5)))
        p["mode"] = "up"
    elif action == "walk":
        # Glide. Two ear-flaps per loop. The inner ear follows the right ear.
        p["ear_l"] = wave(f, n, 12.0, phase=0.3, cycles=2.0)
        p["ear_r"] = wave(f, n, 14.0, phase=0.8, cycles=2.0)
        p["curl"] = wave(f, n, 10.0, phase=0.8, cycles=2.0)
        p["heart"] = wave(f, n, 4.0, phase=0.2, cycles=2.0)
        p["bdy"] = int(round(wave(f, n, 1.6, phase=0.4, cycles=2.0)))
        p["mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 ears and the heart nose rise. No mouth, so no wedge
        # and no ring. 5-8 the back, equal scale. 9-12 the ears slap and
        # the nose thrusts. 13-15 return.
        # Drawn peaks: ear 22, curl 16, heart 12.
        ear_l = [0, 6, 12, 16, 16, 0, 0, 0, 0, 18, 22, 22, 20, 8, 3, 0]
        ear_r = [0, 6, 14, 18, 18, 0, 0, 0, 0, 18, 22, 22, 20, 8, 3, 0]
        curl = [0, 4, 8, 12, 12, 0, 0, 0, 0, 12, 16, 16, 14, 6, 2, 0]
        heart = [0, 3, 6, 10, 10, 0, 0, 0, 0, 8, 12, 12, 10, 4, 2, 0]
        bdx = [0, -1, -1, -1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        bdy = [0, -1, -1, -2, -1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0]
        hdx = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
        p["ear_l"] = float(ear_l[f])
        p["ear_r"] = float(ear_r[f])
        p["curl"] = float(curl[f])
        p["heart"] = float(heart[f])
        p["bdx"] = int(bdx[f])
        p["bdy"] = int(bdy[f])
        p["heart_dx"] = int(hdx[f])
        p["mode"] = "up" if f <= 4 or f >= 13 else "fwd"
        p["turned"] = 5 <= f <= 8
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["ear_l"] = 8.0 * env
        p["ear_r"] = 10.0 * env
        p["curl"] = 6.0 * env
        p["heart"] = 4.0 * env
        p["mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["ear_l"] = 0.0
        p["ear_r"] = 0.0
        p["curl"] = 0.0
        p["heart"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["heart_dx"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["turned"] = True
        p["mode"] = "up"
    return p


def part_angle(rig, name, mode, amp):
    if abs(amp) < 0.45 or name not in rig["signs"]:
        return 0.0
    sign = rig["signs"][name]
    if mode == "up":
        return sign["up"] * amp
    return sign["fwd"] * amp


def moved(layer, ang, pivot, dx, dy):
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, pivot)
    return shift(layer, dx, dy)


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    for name in ("ear_l", "ear_r", "curl"):
        if name not in L or name not in rig["pivots"]:
            continue
        ang = part_angle(rig, name, p["mode"], p[name])
        paste(out, moved(L[name], ang, rig["pivots"][name], bdx, bdy))
    hdx = bdx + int(round(fc * p["heart_dx"]))
    ang = part_angle(rig, "heart", p["mode"], p["heart"])
    paste(out, moved(L["heart"], ang, rig["pivots"]["heart"], hdx, bdy))
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK)
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def limb_lost(rig, p):
    body = compose(rig, p)
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    lost = []
    named = {}
    for name in ("ear_l", "ear_r", "curl"):
        if name not in rig["pivots"]:
            continue
        ang = part_angle(rig, name, p["mode"], p[name])
        named[name] = moved(rig["layers"][name], ang, rig["pivots"][name], bdx, bdy)
    hdx = bdx + int(round(fc * p["heart_dx"]))
    ang = part_angle(rig, "heart", p["mode"], p["heart"])
    named["heart"] = moved(rig["layers"]["heart"], ang, rig["pivots"]["heart"], hdx, bdy)
    for name, layer in named.items():
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    # Both stills face left unmirrored, so a 180 turn flips the mirror.
    return rigs[("back", not mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    moved_part = (
        abs(p["ear_l"]) >= 0.45 or abs(p["ear_r"]) >= 0.45
        or abs(p["curl"]) >= 0.45 or abs(p["heart"]) >= 0.45
        or p["bdx"] or p["bdy"] or p["heart_dx"]
    )
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], piv)
    if moved_part or abs(p["lean"]) > 0.4:
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    return snap(img, palette)


def assemble(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                t = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png").convert("RGBA"))
                sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = t
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


def tile_grid(tiles, cols, bg=(18, 22, 32, 255)):
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    out = np.full((rows * th, cols * tw, 4), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        m = t[..., 3] >= 128
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
    return out


def preview_image(sheets):
    pick = {"idle": 4, "walk": 4, "attack": 11, "hurt": 3}
    tiles = []
    for act in ACTIONS:
        for di in range(8):
            f = pick[act]
            tiles.append(sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W])
    return Image.fromarray(tile_grid(tiles, 8), "RGBA")


def content_box(canvases, pad=16):
    acc = np.zeros(canvases[0].shape[:2], bool)
    for c in canvases:
        acc |= c[..., 3] >= 128
    ys, xs = np.nonzero(acc)
    y0 = max(0, int(ys.min()) - pad - 28)
    y1 = min(CELL_H, int(ys.max()) + pad + 20)
    x0 = max(0, int(xs.min()) - pad - 28)
    x1 = min(CELL_W, int(xs.max()) + pad + 28)
    return y0, y1, x0, x1


def motion_strip(sheets, di, box):
    y0, y1, x0, x1 = box
    rows = []
    for act in ACTIONS:
        row = []
        for f in range(16):
            if f < FRAMES[act]:
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                row.append(t[y0:y1, x0:x1])
            else:
                row.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def locked_palette(front_path, back_path, work):
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
    imgs = [np.array(Image.open(p).convert("RGBA")) for p in (front_path, back_path)]
    own = []
    for img in imgs:
        cols = np.unique(img[img[..., 3] >= 128][:, :3], axis=0)
        own.extend(tuple(int(v) for v in c) for c in cols)
    own = sorted(set(own))
    if not (lock_dir / "palette.json").exists():
        py = str(FORGE / ".venv/bin/python")
        hexes = ["#%02x%02x%02x" % c for c in own]
        for d in (pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        r = subprocess.run(
            [py, "-I", str(PALETTE_TOOL), "build", "--input", str(front_path),
             "--colors", str(len(hexes)), "--reserve", *hexes, "--output-dir", str(pal_dir)],
            capture_output=True, text=True,
        )
        if r.returncode:
            raise SystemExit("palette build failed: " + r.stdout + r.stderr)
        r = subprocess.run(
            [py, "-I", str(PALETTE_TOOL), "lock", "--palette", str(pal_dir / "palette.json"),
             "--source", str(front_path), "--strict", "--output-dir", str(lock_dir)],
            capture_output=True, text=True,
        )
        if r.returncode:
            raise SystemExit("palette lock failed: " + r.stdout + r.stderr)
    data = json.loads((lock_dir / "palette.json").read_text())
    cols = np.array([[int(c["hex"][i:i + 2], 16) for i in (1, 3, 5)] for c in data["colors"]], np.uint8)
    got = {tuple(int(v) for v in c) for c in cols}
    want = set(own)
    if got != want:
        raise SystemExit(f"locked palette differs from the stills: {sorted(got ^ want)}")
    return cols


def params_hash():
    src = Path(__file__).read_bytes() + (SRC_DIR / "front.png").read_bytes() + (SRC_DIR / "back.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def build_rigs():
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    back = np.array(Image.open(SRC_DIR / "back.png").convert("RGBA"))
    if front.shape[:2] != (96, 96) or back.shape[:2] != (96, 96):
        raise SystemExit("expected 96x96 stills")
    for im in (front, back):
        im[im[..., 3] < 128] = 0
        im[im[..., 3] >= 128, 3] = 255
    fm = segment(front, "front")
    bm = segment(back, "back")
    rigs = {
        ("front", False): build_view(front, fm, "front", False),
        ("front", True): build_view(front, fm, "front", True),
        ("back", False): build_view(back, bm, "back", False),
        ("back", True): build_view(back, bm, "back", True),
    }
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "torso": (70, 130, 210, 255),
        "ear_l": (240, 200, 60, 255),
        "ear_r": (80, 180, 90, 255),
        "curl": (230, 80, 140, 255),
        "heart": (220, 50, 50, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 3, 0), 3, 1))
    Image.fromarray(np.concatenate(tiles, axis=1), "RGBA").save(work / "parts.png")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    return n


def publish(bounds, sprite_h):
    import gen5_publish

    entry = {
        "dex": 528, "form": "normal", "variant": "swoobat", "gen": "gen5",
        "dir": "monsterSprite/gen5/0528_swoobat/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def frame_stats(img, palette):
    pal = {tuple(int(v) for v in c) for c in palette}
    op = img[..., 3] == 255
    partial = int(((img[..., 3] != 0) & (img[..., 3] != 255)).sum())
    used = set()
    if op.any():
        used = {tuple(int(v) for v in c) for c in np.unique(img[op][:, :3], axis=0)}
    edge = bool(op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any())
    return len(used - pal), partial, edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    save_peaks(Path(__file__).stem, PEAKS)
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    palette = locked_palette(SRC_DIR / "front.png", SRC_DIR / "back.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    print("names", {d: rigs[d]["names"] for d in ("S", "W", "N", "NE")}, flush=True)

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    turn_w = synth(rigs, "W", "attack", 6, palette, effects=False)
    turn_s = synth(rigs, "S", "attack", 6, palette, effects=False)
    if np.any(turn_s != rigs["N"]["canvas"]):
        raise SystemExit("S turn is not the N back at equal scale")
    if np.any(turn_w != rigs["NE"]["canvas"]):
        raise SystemExit("W turn is not N flipped at equal scale")
    flipped = np.ascontiguousarray(rigs["N"]["canvas"][:, ::-1])
    if np.any(rigs["NE"]["canvas"] != flipped):
        raise SystemExit("NE is not an equal-scale flip of N")

    if args.samples:
        lost = {}
        clip = []
        off_n = 0
        partial_n = 0
        bodies = []
        sample_fs = (
            ("idle", (0, 8)),
            ("walk", (4, 12)),
            ("attack", (0, 3, 6, 8, 11, 14)),
            ("hurt", (0, 3)),
        )
        for d in DIRS:
            for act, fs in sample_fs:
                for f in fs:
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost[f"{act}:{d}:{f}"] = bad
                    t = synth(rigs, d, act, f, palette)
                    off, partial, edge = frame_stats(t, palette)
                    off_n += off
                    partial_n += partial
                    if edge:
                        clip.append(f"{act}:{d}:{f}")
                    bodies.append(largest_fraction(t))
        print("limb lost", lost or "none", flush=True)
        print("sample clip", clip or "none", flush=True)
        print("rest_diff_sum", int(sum(rest_diff.values())), flush=True)
        print("off_palette", off_n, flush=True)
        print("partial_alpha", partial_n, flush=True)
        print("edge_n", len(clip), flush=True)
        print("body_main_min", round(min(bodies), 3) if bodies else 1, flush=True)
        box = content_box([rigs[d]["canvas"] for d in DIRS])
        y0, y1, x0, x1 = box
        tiles = []
        for d in ("S", "W", "N", "E"):
            for act, f in (
                ("idle", 0), ("idle", 8),
                ("walk", 4), ("walk", 12),
                ("attack", 0), ("attack", 3), ("attack", 6), ("attack", 8),
                ("attack", 11), ("attack", 14),
                ("hurt", 1), ("hurt", 3),
            ):
                t = synth(rigs, d, act, f, palette)
                tiles.append(t[y0:y1, x0:x1])
        sheet = np.repeat(np.repeat(tile_grid(tiles, 12), 2, 0), 2, 1)
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
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    box = content_box([rigs[d]["canvas"] for d in DIRS])
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.repeat(np.repeat(np.concatenate(strips, axis=0), 2, 0), 2, 1)
    Image.fromarray(motion, "RGBA").save(OUT_DIR / "preview_motion.png")
    preview_image(sheets).save(OUT_DIR / "preview.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", OUT_DIR / "actions" / "preview.png")

    pal = {tuple(int(v) for v in c) for c in palette}
    op = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[op][:, :3], axis=0)}
    off = sorted(used - pal)
    partial = int(((master[..., 3] != 0) & (master[..., 3] != 255)).sum())
    clip = []
    bodies = []
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = t[..., 3] == 255
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
                bodies.append(largest_fraction(t))
    import gen5_publish
    bounds = gen5_publish.measure_idle_bounds(OUT_DIR / "sheet.png", CELL_W, CELL_H, rows=8)
    sprite_h = max(bounds[i + 3] for i in range(0, 32, 4))
    idle0 = {}
    for di, d in enumerate(DIRS):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        idle0[d] = int(np.any(t != rigs[d]["canvas"], axis=2).sum())
    report = {
        "version": VERSION, "hash": digest, "rest_diff": rest_diff, "idle0": idle0,
        "off_palette_colors": len(off), "partial_alpha": partial, "edge_touching": clip[:12],
        "edge_n": len(clip), "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "bounds": bounds, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (
        not any(rest_diff.values()) and not any(idle0.values()) and not off
        and partial == 0 and not clip and min(bodies) >= 0.90
    )
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_swoobat_actions", "version": VERSION, "key": KEY,
        "dex": 528, "variant": "swoobat", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "hovering bat (two ear-wings, pink inner ear, heart nose, no feet, no mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (nose x41.8 < eye mean x44.7); back heart also screen-left",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    if args.publish:
        print("synced", sync_outputs(), flush=True)
        publish(bounds, sprite_h)
        print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
