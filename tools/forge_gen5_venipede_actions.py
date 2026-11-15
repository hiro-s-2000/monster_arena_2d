#!/usr/bin/env python3
"""Venipede normal (0543) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x27-72 y30-63, 943 px, 14 colours,
alpha 0/255, sole y63):
  The eye is the pink cluster (230, 98, 123) and (255, 148, 164), 8 px,
  mean x 40.0, y 34.5. There is no mouth. Those pinks sit on the head,
  so the roar does not invent a wedge or a ring.
  The face tip is the antenna tip at x35 y30, left of the eye, so the
  still faces screen-left. The body runs back to the right. The green
  poison bulb (x53-64 y37-57) stays on the torso. The tail curl is the
  thin chain x65-72 y40-54 (52 px, no bulb pixels). Yellow specks along
  the belly stay on the torso; they are not a separate fang.
  Feet, gray plus their own ink, three components under the shell:
  head-side x29-35 y58-62, middle x36-40 y58-60, rear x41-62 y58-63.
normal/back.png (bbox x18-66 y28-62, 954 px, 13 colours, all in the front
  palette except the front-only brown speck, sole y62):
  The same eye, 8 px, mean x 52.5 y 36.0. The antenna tip is x59-60 y28,
  right of the eye, so this still faces screen-right. The tail is the
  left curl x18-22. Feet: head-side x53-59, middle x38-52 (sole y62),
  rear x32-36. The back sole is 1 px higher, so the back view is placed
  1 px lower and both soles share one ground line.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (screen-right). NW back mirrored.
  Attack 5-8 shows that direction's other side at equal scale: W turns
  into N, S turns into N flipped. No |cos| squash, no in-plane body spin.

  idle 16: antenna and tail sway, feet planted. Frame 0 is the still.
  walk 16: the three drawn foot rows step in a head-to-tail wave.
  attack 16: 0 still, 1-4 antenna up (no mouth), 5-8 the back,
  9 wind-up, 10-12 head-side feet and the tail strike, 13-15 return.
  hurt 12: lean at most 38 degrees, then recover.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_venipede_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0543_venipede/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0543_venipede/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0543_venipede"
VERSION = "venipede_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 192
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 32
FRONT_FOOT = 63
BACK_FOOT = 62
BACK_DY = FRONT_FOOT - BACK_FOOT
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. W keeps that picture. Its back is N.
# S is the mirrored front, so its back is N mirrored. Equal scale, no squash.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# finish() may raise these during the strike window. The hit is drawn
# higher (kick 34, tail 30), so the envelope does not stretch the hit.
PEAKS = {"kick": 16.0, "tail": 14.0}
LIMBS = ("ant", "tail", "leg_h", "leg_m", "leg_t")

INK = (16, 16, 16)
GRAY = (49, 49, 57)
MAG = (189, 49, 82)
CRIM = (148, 0, 65)
MAROON = (98, 32, 49)
ROSE = (115, 49, 65)
GREEN = (0, 148, 90)
GREEN_D = (8, 74, 57)
GREEN_M = (0, 115, 57)
GOLD = (189, 139, 8)
EYE = (230, 98, 123)
EYE_L = (255, 148, 164)
YELLOW = (238, 180, 8)
BROWN = (139, 90, 0)
GREEN_COLS = (GREEN, GREEN_D, GREEN_M)
EYE_COLS = (EYE, EYE_L)

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


def component_list(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    groups = []
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
        groups.append(pts)
    return groups


def mask_from_pts(shape, pts):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m


def largest_component(mask):
    groups = component_list(mask)
    if not groups:
        return 0
    return max(len(g) for g in groups)


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


def touches(part, body):
    return bool((part & dilate(body, 1)).any())


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
    """4x nearest rotation. Positive is clockwise on screen. No width squash."""
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
        grown = []
        up = np.zeros_like(src)
        dn = np.zeros_like(src)
        lt = np.zeros_like(src)
        rt = np.zeros_like(src)
        up[:-1] = src[1:]
        dn[1:] = src[:-1]
        lt[:, :-1] = src[:, 1:]
        rt[:, 1:] = src[:, :-1]
        grown.extend((up, dn, lt, rt))
        for g in grown:
            take = (acc[..., 3] < 128) & allowed & (g[..., 3] >= 128)
            acc[take] = g[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def hump(f, n, delay, width=5):
    """One step lobe. 0 at frame 0 for delays 0, 5 and 10 when width is 5."""
    x = (f - delay) % n
    if 0 <= x < width:
        return math.sin(math.pi * x / width)
    return 0.0


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
        if len(xs) == 0:
            return 0.0, 0.0
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(torso)
    cx, cy = float(tx.mean()), float(ty.mean())
    d = (xs - cx) ** 2 + (ys - cy) ** 2
    near = d <= d.min() + 8.0
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


def feet_masks(img, y0, head_on_left):
    """Three drawn foot rows. Head-side, middle, rear. Specks stay on the torso."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    seed = (is_color(img, GRAY) | is_color(img, INK)) & (yy >= y0)
    groups = [g for g in component_list(seed) if len(g) >= 7]
    if len(groups) != 3:
        raise SystemExit(f"expected 3 foot rows, got {len(groups)}")
    groups.sort(key=lambda g: float(np.mean([p[1] for p in g])))
    if not head_on_left:
        groups = list(reversed(groups))
    return [mask_from_pts(op.shape, g) for g in groups]


def segment(img, kind):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    green = any_color(img, GREEN_COLS)
    eye = any_color(img, EYE_COLS)
    if kind == "front":
        ant = op & (yy <= 33) & (xx >= 33) & (xx <= 37)
        tail = op & (xx >= 65) & ~green
        feet = feet_masks(img, 58, head_on_left=True)
    else:
        ant = op & (yy <= 33) & (xx >= 57)
        tail = op & (xx <= 22) & ~green
        feet = feet_masks(img, 57, head_on_left=False)
    leg_h, leg_m, leg_t = feet
    used = ant | tail | leg_h | leg_m | leg_t
    torso = op & ~used
    masks = {
        "ant": ant, "tail": tail, "leg_h": leg_h, "leg_m": leg_m, "leg_t": leg_t,
        "torso": torso,
    }
    check_partition(op, masks)
    if int(eye.sum()) != 8 or int((eye & torso).sum()) != 8:
        raise SystemExit(f"{kind} eye left the head")
    if (eye & (ant | tail | leg_h | leg_m | leg_t)).any():
        raise SystemExit(f"{kind} eye was cut off the head")
    for name in LIMBS:
        part = masks[name]
        if largest_component(part) != int(part.sum()):
            raise SystemExit(f"{kind} {name} fell apart")
        if not touches(part, torso):
            raise SystemExit(f"{kind} {name} does not touch the body")
    if int((green & torso).sum()) != int(green.sum()):
        raise SystemExit(f"{kind} poison bulb was cut")
    ys, xs = np.nonzero(ant)
    tip_i = int(ys.argmin())
    tip_x = int(xs[tip_i])
    eye_x = float(np.nonzero(eye)[1].mean())
    if kind == "front":
        if tip_x >= eye_x - 2:
            raise SystemExit("front antenna is not left of the eye")
        if int(ant.sum()) != 9 or int(tail.sum()) != 52:
            raise SystemExit(f"front feelers changed: {int(ant.sum())} {int(tail.sum())}")
        if int(np.nonzero(leg_t)[0].max()) != FRONT_FOOT:
            raise SystemExit("front ground foot moved")
    else:
        if tip_x <= eye_x + 2:
            raise SystemExit("back antenna is not right of the eye")
        if int(ant.sum()) != 18 or int(tail.sum()) != 17:
            raise SystemExit(f"back feelers changed: {int(ant.sum())} {int(tail.sum())}")
        if int(np.nonzero(leg_m)[0].max()) != BACK_FOOT:
            raise SystemExit("back ground foot moved")
    return masks


def place(arr, dy=0):
    y0 = OFF_Y + int(dy)
    x0 = OFF_X
    if arr.ndim == 2:
        out = np.zeros((CELL_H, CELL_W), bool)
        out[y0:y0 + 96, x0:x0 + 96] = arr
        return out
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    out[y0:y0 + 96, x0:x0 + 96] = arr
    return out


def build_view(img, masks, kind, mirror):
    dy = BACK_DY if kind == "back" else 0
    canvas = place(img, dy)
    placed = {k: place(v, dy) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    for name, depth in (("ant", 3), ("tail", 4), ("leg_h", 2), ("leg_m", 2), ("leg_t", 2)):
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth,
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in LIMBS:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "has_mouth": False,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        ant=0.0, ant_dy=0, ant_mode="up",
        tail=0.0, tail_mode="up",
        kick=0.0, kick_mode="fwd",
        sw_h=0.0, sw_m=0.0, sw_t=0.0,
        step_h=(0, 0), step_m=(0, 0), step_t=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The antenna and the tail sway. Frame 0 is the still.
        p["ant"] = wave(f, n, 8.0)
        p["tail"] = wave(f, n, 10.0, phase=math.pi)
        p["ant_dy"] = -1 if wave(f, n, 1.0) > 0.55 else 0
        p["ant_mode"] = "up"
        p["tail_mode"] = "up"
    elif action == "walk":
        # Three drawn foot rows, a head-to-tail wave. Planted at frame 0 and 15.
        hh = hump(f, n, 0)
        hm = hump(f, n, 5)
        ht = hump(f, n, 10)
        p["sw_h"] = 18.0 * hh
        p["sw_m"] = 16.0 * hm
        p["sw_t"] = 14.0 * ht
        p["step_h"] = (int(round(hh)), 0)
        p["step_m"] = (int(round(hm)), 0)
        p["step_t"] = (int(round(ht)), 0)
        p["ant"] = wave(f, n, 8.0, cycles=2.0)
        p["tail"] = wave(f, n, 10.0, phase=math.pi, cycles=2.0)
        p["ant_mode"] = "up"
        p["tail_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 raise the antenna. No mouth, so no wedge and no ring.
        # 5-8 the other side, equal scale. 9 wind-up. 10-12 feet and tail.
        # 13-15 return. 15 is the still.
        ant = [0, 8, 14, 18, 18, 0, 0, 0, 0, 8, 6, 4, 4, 2, 0, 0]
        ant_dy = [0, -1, -2, -2, -2, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]
        tail = [0, 8, 12, 16, 16, 0, 0, 0, 0, 12, 22, 30, 30, 10, 4, 0]
        kick = [0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 28, 34, 34, 12, 4, 0]
        bdx = [0, 0, -1, -1, -1, 0, 0, 0, 0, -1, 1, 2, 2, 1, 0, 0]
        bdy = [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        p["ant"] = float(ant[f])
        p["ant_dy"] = int(ant_dy[f])
        p["tail"] = float(tail[f])
        p["kick"] = float(kick[f])
        p["bdx"] = int(bdx[f])
        p["bdy"] = int(bdy[f])
        p["ant_mode"] = "up"
        p["tail_mode"] = "up" if f <= 4 else "fwd"
        p["kick_mode"] = "fwd"
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.64, 0.44, 0.26, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["ant"] = 8.0 * env
        p["tail"] = 10.0 * env
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        # The other side is the still, at equal scale. finish() must not swing it.
        p["ant"] = 0.0
        p["ant_dy"] = 0
        p["tail"] = 0.0
        p["kick"] = 0.0
        p["sw_h"] = 0.0
        p["sw_m"] = 0.0
        p["sw_t"] = 0.0
        p["step_h"] = (0, 0)
        p["step_m"] = (0, 0)
        p["step_t"] = (0, 0)
        p["bdx"] = 0
        p["bdy"] = 0
        p["spark"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["turned"] = True
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
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    legs = (
        ("leg_t", p["sw_t"], -0.35, p["step_t"]),
        ("leg_m", p["sw_m"], 0.55, p["step_m"]),
        ("leg_h", p["sw_h"], 1.0, p["step_h"]),
    )
    for name, swing, kick_scale, step in legs:
        sx, sy = step
        if abs(p["kick"]) >= 0.45:
            ang = part_angle(rig, name, p["kick_mode"], p["kick"] * kick_scale)
        else:
            ang = part_angle(rig, name, "up", swing)
        dx = bdx + int(round(fc * sx))
        dy = bdy + int(sy)
        paste(out, moved(L[name], ang, piv[name], dx, dy))
    tail_ang = part_angle(rig, "tail", p["tail_mode"], p["tail"])
    paste(out, moved(L["tail"], tail_ang, piv["tail"], bdx, bdy))
    ant_ang = part_angle(rig, "ant", p["ant_mode"], p["ant"])
    paste(out, moved(L["ant"], ant_ang, piv["ant"], bdx, bdy + int(p["ant_dy"])))
    return out


def flash_hit(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK)
    out[op & ~keep, :3] = np.array(EYE_L, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def leg_tip(rig, p):
    """Tip of the head-side foot after the strike, in cell pixels."""
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    ang = part_angle(rig, "leg_h", p["kick_mode"], p["kick"])
    layer = moved(L["leg_h"], ang, piv["leg_h"], bdx, bdy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (GOLD, EYE_L, CRIM)
    r = 3 + age
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (1, -r), (1, r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    other = "back" if kind == "front" else "front"
    return rigs[(other, mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    fc = rig["facing"]
    rotated = (
        abs(p["ant"]) >= 0.45 or abs(p["tail"]) >= 0.45 or abs(p["kick"]) >= 0.45
        or abs(p["sw_h"]) >= 0.45 or abs(p["sw_m"]) >= 0.45 or abs(p["sw_t"]) >= 0.45
        or abs(p["lean"]) > 0.4
    )
    steps = (p["step_h"], p["step_m"], p["step_t"])
    shifted = bool(p["bdx"] or p["bdy"] or p["ant_dy"] or p["knock"] or any(s != (0, 0) for s in steps))
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        body = rotate_nearest(body, -fc * p["lean"], rear_pivot(body, fc))
    if rotated:
        body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    elif shifted:
        img = body
    else:
        img = rig["canvas"].copy()
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_hit(img), palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = leg_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def limb_lost(rig, p):
    body = compose(rig, p)
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    lost = []
    specs = [
        ("leg_t", p["sw_t"], -0.35, p["step_t"], 0),
        ("leg_m", p["sw_m"], 0.55, p["step_m"], 0),
        ("leg_h", p["sw_h"], 1.0, p["step_h"], 0),
        ("tail", 0.0, 0.0, (0, 0), 0),
        ("ant", 0.0, 0.0, (0, 0), int(p["ant_dy"])),
    ]
    for name, swing, kick_scale, step, extra_dy in specs:
        if int((L[name][..., 3] >= 128).sum()) == 0:
            continue
        if name in ("leg_h", "leg_m", "leg_t"):
            sx, sy = step
            if abs(p["kick"]) >= 0.45:
                ang = part_angle(rig, name, p["kick_mode"], p["kick"] * kick_scale)
            else:
                ang = part_angle(rig, name, "up", swing)
            dx = bdx + int(round(fc * sx))
            dy = bdy + int(sy)
        elif name == "tail":
            ang = part_angle(rig, "tail", p["tail_mode"], p["tail"])
            dx, dy = bdx, bdy
        else:
            ang = part_angle(rig, "ant", p["ant_mode"], p["ant"])
            dx, dy = bdx, bdy + extra_dy
        layer = moved(L[name], ang, piv[name], dx, dy)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


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


def content_box(canvases, pad=8):
    acc = np.zeros_like(canvases[0][..., 3], bool)
    for c in canvases:
        acc |= c[..., 3] >= 128
    ys, xs = np.nonzero(acc)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(CELL_H, int(ys.max()) + pad + 8)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(CELL_W, int(xs.max()) + pad)
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
    if int(np.nonzero(front[..., 3])[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    if int(np.nonzero(back[..., 3])[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line changed")
    fm = segment(front, "front")
    bm = segment(back, "back")
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs[("front", False)]["facing"] != -1.0 or rigs[("back", False)]["facing"] != 1.0:
        raise SystemExit("facing locked to the wrong side")
    if rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("venipede has no drawn mouth")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "ant": (255, 220, 40, 255),
        "tail": (255, 80, 40, 255),
        "leg_h": (40, 200, 90, 255),
        "leg_m": (40, 160, 220, 255),
        "leg_t": (180, 80, 220, 255),
        "torso": (170, 170, 170, 255),
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
        "dex": 543, "form": "normal", "variant": "venipede", "gen": "gen5",
        "dir": "monsterSprite/gen5/0543_venipede/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def frame_metrics(rigs, palette):
    pal = {tuple(int(v) for v in c) for c in palette}
    off = set()
    partial = 0
    clip = []
    bodies = []
    for d in DIRS:
        for act in ACTIONS:
            for f in range(FRAMES[act]):
                t = synth(rigs, d, act, f, palette)
                a = t[..., 3]
                partial += int(((a != 0) & (a != 255)).sum())
                op = a == 255
                if op.any():
                    used = {tuple(int(v) for v in c) for c in np.unique(t[op][:, :3], axis=0)}
                    off |= used - pal
                if op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
                if act in ("attack", "hurt", "walk"):
                    bodies.append(largest_fraction(t))
    return {
        "off_palette": len(off),
        "partial_alpha": partial,
        "edge_touch": len(clip),
        "edge_touching": clip[:8],
        "body_main_min": round(min(bodies), 3) if bodies else 1,
    }


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
          "palette", len(palette), "foot", FOOT_Y, "mouth", 0, flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)

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
    if np.any(turn_w != rigs["N"]["canvas"]):
        raise SystemExit("W turn is not the N back at equal scale")
    if np.any(turn_s != rigs["NW"]["canvas"]):
        raise SystemExit("S turn is not N flipped at equal scale")

    if args.samples:
        lost = {}
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost[f"{act}:{d}:{f}"] = bad
        print("limb lost", lost or "none", flush=True)
        metrics = frame_metrics(rigs, palette)
        print("qa_samples", "rest_diff", int(sum(rest_diff.values())),
              "off_palette", metrics["off_palette"],
              "partial_alpha", metrics["partial_alpha"],
              "edge_touch", metrics["edge_touch"],
              "body_main_min", metrics["body_main_min"],
              "clip", metrics["edge_touching"], flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(float(p["kick"]), 1), round(float(p["tail"]), 1),
                        round(float(p["ant"]), 1), p["turned"], p["spin"], p["bdx"]))
        print("attack poses", atk, flush=True)
        leans = [round(float(pose("hurt", f)["lean"]), 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        box = content_box([rigs[d]["canvas"] for d in DIRS])
        y0, y1, x0, x1 = box
        tiles = []
        for d in ("S", "W", "N", "E"):
            for act, f in (
                ("idle", 0), ("idle", 8),
                ("walk", 4), ("walk", 12),
                ("attack", 0), ("attack", 3), ("attack", 6), ("attack", 11),
                ("hurt", 3),
            ):
                tiles.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
        sheet = np.repeat(np.repeat(tile_grid(tiles, 9), 2, 0), 2, 1)
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
                if act in ("attack", "hurt", "walk"):
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
    print("qa_full", "rest_diff", int(sum(rest_diff.values())),
          "off_palette", len(off),
          "partial_alpha", partial,
          "edge_touch", len(clip),
          "body_main_min", report["body_main_min"],
          "idle0", int(sum(idle0.values())),
          flush=True)
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_venipede_actions", "version": VERSION, "key": KEY,
        "dex": 543, "variant": "venipede", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "centipede (no mouth, one antenna, three foot rows, rear curl, poison bulb on the torso)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (antenna tip x35 < eye mean x 40.0); back faces screen-right (antenna tip x59-60 > eye mean x 52.5)",
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
