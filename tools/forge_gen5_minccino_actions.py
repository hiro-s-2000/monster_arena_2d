#!/usr/bin/env python3
"""Minccino normal (0572) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x21-77 y26-74, 1551 px, alpha 0/255):
  Nine colours. The red, rose, and pink pixels are the two eyes (left cluster
  mean x 28.3, right cluster mean x 54.2, all 46 pixels mean x 46.30, y36-46).
  On those rows the head, excluding the tail past x61, starts at x21. That tip
  is left of the eyes, so the still faces screen-left. There is no mouth pixel.
  The roar does not invent one: it lifts the ears and the broom the still has.
  Left ear tip x23-35 y28-35. Right ear tip x47-59 y26-33. Skull and both eyes
  stay on the torso.
  The broom is two parts already in the still. The side curl is the right mass
  x55-77 y41-72 (white tip, beige shaft). The brush is the gray head under the
  belly, x33-60 y66-72. Two feet reach y74: snout-side x35-43, tail-side x44-52.

normal/back.png (bbox x13-70 y23-71, 1550 px, foot y71) has none of the eye
  colours. The tail is the beige mass on the left, x13-36 y38-70, so the snout
  side of this rear view is screen-right. Two ear tips. Hind foot x37-46 from
  y69, fore foot x47-55 reaching y71. The back is shifted down 3px. Both stills
  are 49px tall, so the tops and the ground line land together.

Directions, front faces left, back faces right:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (snout side screen-right). NW back mirrored.
  E is screen-right, W is screen-left, N is the back. Attack 5-8 swaps to the
  opposite still at equal scale. No width squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_minccino_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0572_minccino/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0572_minccino/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0572_minccino"
VERSION = "minccino_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 192
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 24
FRONT_FOOT = 74
BACK_FOOT = 71
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. Back as drawn looks screen-right.
# S (front mirrored) turns into N (back as drawn) by flipping the mirror flag.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Roar hold. The broom strike is drawn up to 34 deg, above these.
PEAKS = {"ear": 16.0, "tail": 16.0, "brush": 14.0}
LIMBS = ("ear_l", "ear_r", "tail", "brush", "foot_fore", "foot_hind")

INK = (16, 16, 16)
BROWN = (82, 74, 65)
GRAY = (139, 131, 123)
BEIGE = (189, 180, 164)
CREAM = (222, 205, 197)
WHITE = (255, 255, 255)
RED = (189, 41, 41)
ROSE = (213, 82, 82)
PINK = (238, 131, 131)
EYE_COLS = (RED, ROSE, PINK)

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
    """4x nearest in-plane rotation. Positive angle follows the shared forge formula. No squash."""
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


def component_list(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    groups = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
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
        groups.append(pts)
    return groups


def reconnect(img):
    """Fill a 1px gap if a turned ear or broom loses its stem. Effects stay loose."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(3):
        groups = component_list(out[..., 3] >= 128)
        if len(groups) <= 1:
            return out
        groups.sort(key=len, reverse=True)
        main = np.zeros((h, w), bool)
        for y, x in groups[0]:
            main[y, x] = True
        changed = False
        for g in groups[1:]:
            if len(g) < 4:
                continue
            best = None
            for y, x in g:
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and main[ny, nx]:
                            dist = max(abs(dy), abs(dx))
                            if best is None or dist < best[0]:
                                best = (dist, y, x, ny, nx)
            if best is None or best[0] > 2:
                continue
            _, y, x, ny, nx = best
            if abs(ny - y) == 1 and abs(nx - x) == 1:
                candidates = ((y, nx), (ny, x))
            else:
                sy = 0 if ny == y else (1 if ny > y else -1)
                sx = 0 if nx == x else (1 if nx > x else -1)
                candidates = ((y + sy, x + sx),)
            for fy, fx in candidates:
                if 0 <= fy < h and 0 <= fx < w and out[fy, fx, 3] < 128:
                    out[fy, fx, :3] = out[y, x, :3]
                    out[fy, fx, 3] = 255
                    changed = True
                    break
        if not changed:
            return out
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


def check_partition(op, masks):
    acc = np.zeros_like(op)
    for name, m in masks.items():
        if (m & acc).any():
            raise SystemExit(f"overlap in {name}")
        if (m & ~op).any():
            raise SystemExit(f"{name} left the still")
        acc |= m
    if int(acc.sum()) != int(op.sum()):
        raise SystemExit(f"parts do not partition the still: {int(acc.sum())} vs {int(op.sum())}")


def largest_component(mask):
    groups = component_list(mask)
    if not groups:
        return 0
    return max(len(g) for g in groups)


def touches(part, body):
    return bool((part & dilate(body, 1)).any())


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    white = is_color(img, WHITE)
    gray = is_color(img, GRAY)
    ink = is_color(img, INK)
    brown = is_color(img, BROWN)
    # Ear tips only. The skull and both eyes stay on the torso.
    ear_l = flood(op & (yy <= 35) & (xx <= 42), white & (xx < 40) & (yy <= 34))
    ear_r = flood(op & (yy <= 33) & (xx >= 47), white & (xx >= 49) & (yy <= 32))
    feet = op & (yy >= 72) & (xx <= 52)
    tail = flood(op & (xx >= 61) & (yy >= 40), op & (xx >= 70) & (yy >= 42) & (yy <= 50))
    brush_seed = gray & (yy >= 66) & (xx < 61)
    brush_allow = (
        (gray & (yy >= 66))
        | (ink & (yy >= 66) & (xx < 61) & (xx >= 33))
        | (brown & (yy >= 67) & (xx < 58))
    ) & ~feet
    brush = flood(brush_allow, brush_seed)
    # The curl's hip edge sits just left of the x61 cut. One beige speck at
    # x58 y70 touches the brush, not the curl.
    hip = op & (xx >= 55) & (xx <= 60) & (yy >= 64) & ~brush & ~feet
    joined = flood(hip | tail, tail) & ~tail
    tail = tail | joined
    speck = hip & ~joined
    if speck.any():
        brush = brush | speck
    # Snout is screen-left, so the left sole is the fore foot.
    foot_fore = op & (yy >= 72) & (xx <= 43)
    foot_hind = op & (yy >= 72) & (xx >= 44) & (xx <= 52)
    used = ear_l | ear_r | tail | brush | foot_fore | foot_hind
    torso = op & ~used
    masks = {
        "ear_l": ear_l, "ear_r": ear_r, "tail": tail, "brush": brush,
        "foot_fore": foot_fore, "foot_hind": foot_hind, "torso": torso,
    }
    check_partition(op, masks)
    eye = any_color(img, EYE_COLS)
    if int((eye & torso).sum()) != 46 or int(eye.sum()) != 46:
        raise SystemExit("front eyes left the face")
    if int((eye & (ear_l | ear_r | tail | brush | foot_fore | foot_hind)).sum()):
        raise SystemExit("front eyes were cut into a limb")
    ys, xs = np.nonzero(eye)
    eye_x = float(xs.mean())
    head = op[int(ys.min()):int(ys.max()) + 1, :62]
    tip = int(np.nonzero(head.any(axis=0))[0].min())
    if not tip < eye_x - 10:
        raise SystemExit(f"front does not face left: tip {tip} eye {eye_x:.1f}")
    counts = {
        "ear_l": 71, "ear_r": 77, "tail": 293, "brush": 97,
        "foot_fore": 19, "foot_hind": 18, "torso": 976,
    }
    for name, n in counts.items():
        if int(masks[name].sum()) != n:
            raise SystemExit(f"front {name} size {int(masks[name].sum())} != {n}")
        if name == "torso":
            continue
        # The curl meets the brush. The soles meet the brush, under the belly.
        if name == "tail":
            anchor = torso | brush
        elif name.startswith("foot"):
            anchor = torso | brush | tail
        else:
            anchor = torso
        if not touches(masks[name], anchor):
            raise SystemExit(f"front {name} does not touch the body")
        if largest_component(masks[name]) != n:
            raise SystemExit(f"front {name} fell apart")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("front torso fell apart")
    if int(np.nonzero(foot_fore | foot_hind)[0].max()) != FRONT_FOOT:
        raise SystemExit("front ground line moved")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    white = is_color(img, WHITE)
    ear_l = flood(op & (yy <= 32) & (xx <= 48), white & (xx >= 33) & (xx <= 44) & (yy <= 31))
    ear_r = flood(op & (yy <= 31) & (xx >= 55), white & (xx >= 58) & (yy <= 29))
    tail = flood(op & (xx <= 36) & (yy >= 37), op & (xx <= 22) & (yy >= 39) & (yy <= 48))
    # Snout side is screen-right, so the right sole is the fore foot.
    foot_hind = op & (yy >= 69) & (xx >= 37) & (xx <= 46) & ~tail
    foot_fore = op & (yy >= 69) & (xx >= 47) & ~tail
    brush = np.zeros_like(op)
    used = ear_l | ear_r | tail | foot_fore | foot_hind
    torso = op & ~used
    masks = {
        "ear_l": ear_l, "ear_r": ear_r, "tail": tail, "brush": brush,
        "foot_fore": foot_fore, "foot_hind": foot_hind, "torso": torso,
    }
    check_partition(op, masks)
    if any_color(img, EYE_COLS).any():
        raise SystemExit("back gained an eye")
    counts = {
        "ear_l": 78, "ear_r": 89, "tail": 323,
        "foot_fore": 18, "foot_hind": 17, "torso": 1025,
    }
    for name, n in counts.items():
        if int(masks[name].sum()) != n:
            raise SystemExit(f"back {name} size {int(masks[name].sum())} != {n}")
        if name != "torso" and not touches(masks[name], torso):
            raise SystemExit(f"back {name} does not touch the body")
        if name != "torso" and largest_component(masks[name]) != n:
            raise SystemExit(f"back {name} fell apart")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("back torso fell apart")
    if int(np.nonzero(foot_fore)[0].max()) != BACK_FOOT:
        raise SystemExit("back ground foot moved")
    tail_x = np.nonzero(tail)[1]
    head = op & (yy <= 36)
    if int(tail_x.max()) > 36 or int(np.nonzero(head)[1].max()) < 64:
        raise SystemExit("back tail is not left of the head")
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
    near = d <= d.min() + 6.0
    return float(xs[near].mean()), float(ys[near].mean())


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
    for name, depth in (
        ("ear_l", 3), ("ear_r", 3), ("brush", 5),
        ("foot_fore", 3), ("foot_hind", 3),
    ):
        if int(placed[name].sum()) == 0:
            continue
        layers["torso"] = paint_cushion(layers["torso"], joint_band(placed[name], torso_m, depth), depth)
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    brush_anchor = layers["torso"].copy()
    paste(brush_anchor, layers["brush"])
    for name in LIMBS:
        if int((layers[name][..., 3] >= 128).sum()) == 0:
            continue
        anchor = brush_anchor if name == "tail" else layers["torso"]
        pivots[name] = pivot_of(layers[name], anchor)
        signs[name] = signs_for(layers[name], pivots[name], facing)
    # The curl's stem is the brush, not the belly. Leave brush colour in that joint.
    # The soles sit under the broom, so their cushion lives on the brush too.
    if int(placed["brush"].sum()):
        for name, depth in (("tail", 5), ("foot_fore", 3), ("foot_hind", 3)):
            if int(placed[name].sum()) == 0 or not touches(placed[name], placed["brush"]):
                continue
            layers["brush"] = paint_cushion(
                layers["brush"], joint_band(placed[name], placed["brush"], depth), depth
            )
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        ear=0.0, ear_mode="up",
        tail=0.0, tail_mode="up",
        brush=0.0, brush_mode="up",
        step_fore=(0, 0), step_hind=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. Ear tips perk. The broom sways from its own joint.
        p["ear"] = wave(f, n, 7.0)
        p["tail"] = wave(f, n, 8.0, phase=0.5)
        p["brush"] = wave(f, n, 6.0, phase=1.1)
    elif action == "walk":
        # Two drawn feet. Fore and hind trot. The broom counters twice a loop.
        fore = wave(f, n, 1.0)
        hind = wave(f, n, 1.0, phase=math.pi)
        p["step_fore"] = (int(round(2 * fore)), -int(round(2 * max(0.0, fore))))
        p["step_hind"] = (int(round(2 * hind)), -int(round(2 * max(0.0, hind))))
        p["ear"] = wave(f, n, 6.0)
        p["tail"] = wave(f, n, 10.0, phase=0.4, cycles=2.0)
        p["brush"] = wave(f, n, 8.0, phase=0.9, cycles=2.0)
    elif action == "attack":
        # 0 rest. 1-4 lift the ears and the broom. No mouth is drawn.
        # 5-8 the opposite still, equal scale. 9-12 the curl and the brush sweep.
        # 11-12 dust. 13-15 recover. 15 is the still.
        ear = [0, 8, 14, 16, 16, 16, 16, 16, 14, 10, 8, 6, 4, 2, 0, 0]
        tail = [0, 6, 12, 16, 16, 16, 16, 16, 14, 22, 30, 34, 34, 16, 6, 0]
        brush = [0, 4, 10, 14, 14, 14, 14, 14, 12, 18, 26, 32, 32, 14, 4, 0]
        bdx = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        p["ear"] = float(ear[f])
        p["tail"] = float(tail[f])
        p["brush"] = float(brush[f])
        p["bdx"] = int(bdx[f])
        p["tail_mode"] = "fwd" if 9 <= f <= 12 else "up"
        p["brush_mode"] = "fwd" if 9 <= f <= 12 else "up"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        # Ears and the broom ride the lean. A separate turn tears their stems.
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["flash"] = f == 1
    return finish(p, action, f, __file__)


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


def leg_shift(p, name):
    if name == "foot_fore":
        return p["step_fore"]
    if name == "foot_hind":
        return p["step_hind"]
    return (0, 0)


def part_angle_name(p, name):
    if name == "tail":
        return p["tail"], p["tail_mode"]
    if name == "brush":
        return p["brush"], p["brush_mode"]
    if name.startswith("ear"):
        return p["ear"], p["ear_mode"]
    return 0.0, "up"


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])

    def one(name):
        if int((L[name][..., 3] >= 128).sum()) == 0:
            return np.zeros_like(L[name])
        if name.startswith("foot"):
            sx, sy = leg_shift(p, name)
            dx = bdx + int(round(fc * sx))
            dy = bdy + int(sy)
            ang = 0.0
        else:
            amp, mode = part_angle_name(p, name)
            ang = part_angle(rig, name, mode, amp)
            dx, dy = bdx, bdy
        return moved(L[name], ang, piv.get(name, (0.0, 0.0)), dx, dy)

    paste(out, shift(L["torso"], bdx, bdy))
    # Brush first, so the curl and the soles cover their cushions at rest.
    for name in ("brush", "foot_hind", "tail", "foot_fore", "ear_l", "ear_r"):
        paste(out, one(name))
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK) | is_color(out, BROWN)
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(rig):
    """Sole of the hind foot. The broom hangs beside it and is not the anchor."""
    fc = rig["facing"]
    layer = rig["layers"]["foot_hind"]
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    yb = int(ys.max())
    band = ys >= yb - 1
    xb = int(xs[band].min()) if fc > 0 else int(xs[band].max())
    return float(xb), float(yb)


def leading_tip(rig, p, name):
    fc = rig["facing"]
    layer = limb_layers(rig, p).get(name)
    if layer is None:
        return rig["pivots"]["center"]
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def dust(img, origin, fc, age):
    """A few palette pixels ahead of the broom. Not a mouth."""
    ox, oy = origin
    cols = (WHITE, CREAM, GRAY)
    specks = ((2, 0), (3, -1), (3, 1), (4, 0), (2, -2), (4, 2))
    for i, (dx, dy) in enumerate(specks[: 3 + age]):
        put(img, ox + fc * (dx + age), oy + dy, cols[i % 3])


def limb_layers(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    parts = {}
    for name in LIMBS:
        if int((L[name][..., 3] >= 128).sum()) == 0:
            continue
        if name.startswith("foot"):
            sx, sy = leg_shift(p, name)
            dx = bdx + int(round(fc * sx))
            dy = bdy + int(sy)
            ang = 0.0
        else:
            amp, mode = part_angle_name(p, name)
            ang = part_angle(rig, name, mode, amp)
            dx, dy = bdx, bdy
        parts[name] = moved(L[name], ang, piv[name], dx, dy)
    return parts


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name, layer in limb_layers(rig, p).items():
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    rig = rigs[d]
    if p["turned"]:
        kind, mirror = VIEW_OF[d]
        rig = rigs[("back" if kind == "front" else "front", not mirror)]
    return rig


def moving(p):
    if p["step_fore"] != (0, 0) or p["step_hind"] != (0, 0) or p["bdx"] or p["bdy"]:
        return True
    if abs(p["lean"]) > 0.4 or p["knock"]:
        return True
    for key in ("ear", "tail", "brush"):
        if abs(p[key]) >= 0.45:
            return True
    return False


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    body = compose(rig, p)
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        # Head moves opposite the facing. Positive rotation sends the top screen-left.
        body = rotate_nearest(body, fc * p["lean"], rear_pivot(rig))
    if moving(p):
        body = reconnect(fill_pinholes(despeckle(body)))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        name = "brush" if "brush" in rig["pivots"] else "tail"
        tx, ty = leading_tip(rig, p, name)
        dust(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def palette_set(palette):
    return {tuple(int(v) for v in c) for c in palette}


def audit(img, colors):
    a = img[..., 3]
    partial = int(((a != 0) & (a != 255)).sum())
    edge = 1 if (a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any()) else 0
    off = []
    op = a == 255
    if op.any():
        for c in np.unique(img[op][:, :3], axis=0):
            t = tuple(int(v) for v in c)
            if t not in colors:
                off.append(t)
    return off, partial, edge


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


def tile_grid(tiles, cols, bg=(18, 22, 34, 255)):
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    out = np.full((rows * th, cols * tw, 4), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        m = t[..., 3] >= 128
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
    return out


def crop_box(canvas, pad=12):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad - 28)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 20)
    x0 = max(0, int(xs.min()) - pad - 36)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 36)
    return y0, y1, x0, x1


def preview_image(sheets):
    pick = {"idle": 4, "walk": 4, "attack": 11, "hurt": 3}
    tiles = []
    for act in ACTIONS:
        for di in range(8):
            f = pick[act]
            tiles.append(sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W])
    return Image.fromarray(tile_grid(tiles, 8), "RGBA")


def motion_strip(sheets, di, box):
    y0, y1, x0, x1 = box
    tiles = []
    for act in ACTIONS:
        for f in range(16):
            if f < FRAMES[act]:
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                tiles.append(t[y0:y1, x0:x1])
            else:
                tiles.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
    return tile_grid(tiles, 16)


def locked_palette(front_path, back_path, work):
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
    img = np.array(Image.open(front_path).convert("RGBA"))
    own = np.unique(img[img[..., 3] >= 128][:, :3], axis=0)
    if not (lock_dir / "palette.json").exists():
        py = str(FORGE / ".venv/bin/python")
        hexes = ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in own]
        for d in (pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        r = subprocess.run(
            [py, "-I", str(PALETTE_TOOL), "build", "--input", str(front_path),
             "--colors", str(len(hexes)), "--reserve", *hexes,
             "--output-dir", str(pal_dir)],
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
    want = {tuple(int(v) for v in c) for c in own}
    if got != want:
        raise SystemExit(f"locked palette differs from the still: {sorted(got ^ want)}")
    back = np.array(Image.open(back_path).convert("RGBA"))
    extra = []
    for c in np.unique(back[back[..., 3] >= 128][:, :3], axis=0):
        t = tuple(int(v) for v in c)
        if t not in got:
            extra.append(c)
    if extra:
        cols = np.vstack([cols, np.array(extra, np.uint8)])
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
    if int(np.nonzero(front[..., 3])[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    if int(np.nonzero(back[..., 3])[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line changed")
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "ear_l": (220, 40, 40, 255),
        "ear_r": (255, 150, 40, 255),
        "tail": (240, 220, 40, 255),
        "brush": (80, 200, 220, 255),
        "foot_fore": (40, 180, 255, 255),
        "foot_hind": (40, 210, 80, 255),
        "torso": (170, 170, 170, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        a = img[..., 3] >= 128
        ys, xs = np.nonzero(a)
        y0, y1 = int(ys.min()) - 1, int(ys.max()) + 2
        x0, x1 = int(xs.min()) - 1, int(xs.max()) + 2
        tiles.append(np.repeat(np.repeat(part[y0:y1, x0:x1], 6, 0), 6, 1))
    h = max(t.shape[0] for t in tiles)
    padded = []
    for t in tiles:
        if t.shape[0] == h:
            padded.append(t)
            continue
        z = np.zeros((h, t.shape[1], 4), np.uint8)
        z[:t.shape[0]] = t
        padded.append(z)
    gap = np.zeros((h, 8, 4), np.uint8)
    Image.fromarray(np.concatenate([padded[0], gap, padded[1]], axis=1), "RGBA").save(work / "parts.png")


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
        "dex": 572, "form": "normal", "variant": "minccino", "gen": "gen5",
        "dir": "monsterSprite/gen5/0572_minccino/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


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
    colors = palette_set(palette)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, flush=True)
    print("parts", {k: int(v.sum()) for k, v in fm.items()}, {k: int(v.sum()) for k, v in bm.items()}, flush=True)

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
        r3 = synth(rigs, d, "attack", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r3 != rigs[d]["canvas"], axis=2).sum())
    rest_sum = int(sum(rest_diff.values()))
    print("rest diff", rest_diff, "sum", rest_sum, flush=True)
    if rest_sum:
        raise SystemExit("frame 0 does not match the still")

    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        y0, y1, x0, x1 = box
        off_colors = set()
        partial = 0
        edge = []
        lost = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    img = synth(rigs, d, act, f, palette)
                    off, part, ed = audit(img, colors)
                    off_colors.update(off)
                    partial += part
                    if ed:
                        edge.append(f"{act}:{d}:{f}")
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        for d in ("S", "W", "N", "E"):
            rows = []
            for act in ACTIONS:
                frames = []
                for f in range(16):
                    if f < FRAMES[act]:
                        frames.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
                    else:
                        frames.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
                rows.append(tile_grid(frames, 16))
            sheet = np.concatenate(rows, axis=0)
            z = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
            Image.fromarray(z, "RGBA").save(work / f"samples_{d}.png")
        atk = []
        for f in range(16):
            p = pose("attack", f)
            rig = rig_for(rigs, "S", p)
            atk.append((f, round(p["ear"], 1), round(p["tail"], 1), round(p["brush"], 1),
                        p["tail_mode"], p["turned"], int(p["spin"]), p["bdx"], rig["kind"]))
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("attack poses", atk, flush=True)
        print("hurt lean", leans, "max", max(leans), flush=True)
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        print("rest_diff", rest_sum, flush=True)
        print("off_palette", len(off_colors), flush=True)
        print("partial_alpha", partial, flush=True)
        print("edge_touch", len(edge), edge[:12], flush=True)
        print("samples", work, flush=True)
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
                if fp.exists() and fp.stat().st_size > 0 and not args.force:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    y0, y1, x0, x1 = box
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.concatenate(strips, axis=0)
    motion = np.repeat(np.repeat(motion, 2, 0), 2, 1)
    Image.fromarray(motion, "RGBA").save(OUT_DIR / "preview_motion.png")
    preview_image(sheets).save(OUT_DIR / "preview.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", OUT_DIR / "actions" / "preview.png")

    off_colors = set()
    partial = 0
    clip = []
    bodies = []
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                off, part, ed = audit(t, colors)
                off_colors.update(off)
                partial += part
                if ed:
                    clip.append(f"{act}:{d}:{f}")
                if act in ("attack", "hurt", "walk"):
                    bodies.append(largest_fraction(t))
    import gen5_publish
    bounds = gen5_publish.measure_idle_bounds(OUT_DIR / "sheet.png", CELL_W, CELL_H, rows=8)
    sprite_h = max(bounds[i + 3] for i in range(0, 32, 4))
    idle0 = {}
    for di, d in enumerate(DIRS):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        idle0[d] = int(np.any(t != rigs[d]["canvas"], axis=2).sum())
    idle0_sum = int(sum(idle0.values()))
    report = {
        "version": VERSION, "hash": digest, "rest_diff": rest_sum, "idle0": idle0_sum,
        "off_palette_colors": len(off_colors), "partial_alpha": partial,
        "edge_touch": len(clip), "edge_touching": clip[:12],
        "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = rest_sum == 0 and idle0_sum == 0 and not off_colors and partial == 0 and not clip
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("rest_diff", rest_sum, flush=True)
    print("idle0", idle0_sum, flush=True)
    print("off_palette", len(off_colors), flush=True)
    print("partial_alpha", partial, flush=True)
    print("edge_touch", len(clip), clip[:12], flush=True)
    print(json.dumps(report), flush=True)
    meta = {
        "generator": "forge_gen5_minccino_actions", "version": VERSION, "key": KEY,
        "dex": 572, "variant": "minccino", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off_colors),
        "archetype": "biped chinchilla (two ear tips, broom curl, broom brush, two feet, no mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (face tip x < eye x); back tail is screen-left so the snout side is screen-right",
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
