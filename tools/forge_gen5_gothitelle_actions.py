#!/usr/bin/env python3
"""Gothitelle normal (0576) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x22-73 y12-84, 2018 px, 14 colours,
alpha 0/255, sole y84):
  Left eye (8,115,205) and (57,164,255) at x42-43, mean x 42.33.
  Right eye at x50-52, mean x 51.25. Midpoint x 46.79.
  The mouth is the two lip pixels (255,106,74) at x46-47 y35, mean x 46.5,
  with the brown mouth (156,82,65) at x45-48 y34-36, mean x 46.71.
  The mouth sits between the eyes, so the still faces the viewer.
  The mouth is drawn, so the roar uses a wedge and a ring in these colours.
  Nothing is invented as a mouth on the back.
  Bow loops (the hair): left x22-39 y13-39 plus its shine x40-44 y<=22
  (353 px, one piece). Right x56-73 y12-40 plus its shine x51-55 y<=24
  (380 px). The face, the eyes, and the centre knot stay on the torso.
  Hands, the only striking limbs: left x27-36 y49-58 (60 px, pink palm and
  its own white cuff). Right x59-67 y51-60 (64 px). The white bows down the
  dress (x38-51) stay on the torso. They are not hands and not feet.
  The hem is one rounded skirt, sole y84. No feet are drawn.

normal/back.png (bbox x22-73 y12-84, 1990 px, 9 colours, sole y84, all colours
  are in the front palette). No eyes, no lips, no face pink. The back-of-bow
  white is one blob at x41-54 y16-24 and stays on the torso, so the loops are
  the dark sides only: left 314 px, right 324 px. Hands: left x28-36 y48-57
  (59 px), right x59-67 y50-59 (63 px). Opaque centroid x 48.00, with 969 px
  on x>48 and 948 on x<48, so this still's mass sits slightly screen-right.
  Both soles are y84, so the back is not shifted.

Directions, because the front faces the camera and the back mass is on the right:
  S SE E front mirrored (forward is screen-right). W SW front as drawn.
  N NE back as drawn. NW back mirrored.
  E is screen-right, W is screen-left, N is the back. Attack 5-8 shows that
  direction's back at equal scale (S/SE/E become the N still). No width squash,
  no in-plane body spin.

  idle 16: the bow loops and the hands breathe. The skirt floats one pixel.
  Frame 0 is the still. No feet are invented.
  walk 16: a glide. The hands swing, the loops sway, the body bobs. No legs.
  attack 16: 0 rest, 1-4 roar (mouth wedge and ring on the front only),
  5-8 the back still, 9 hand wind-up (18 deg), 10-12 both hands strike
  (28, 34, 34), 13-15 return.
  hurt 12: lean at most 38 degrees, then recover.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_gothitelle_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0576_gothitelle/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0576_gothitelle/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0576_gothitelle"
VERSION = "gothitelle_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 208, 208
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 28
FRONT_FOOT = 84
BACK_FOOT = 84
FOOT_Y = OFF_Y + FRONT_FOOT
# Front faces the viewer. Forward on the mirrored front is screen-right.
# Back as drawn has its extra mass on the screen-right, so N uses it unmirrored.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# finish() may raise these on the strike window. The hand hit is drawn at
# 34, above these peaks, so the envelope does not stretch the hit.
PEAKS = {"hand_l": 16.0, "hand_r": 16.0}
LIMBS = ("bow_l", "bow_r", "hand_l", "hand_r")

INK = (0, 0, 0)
GRAY = (82, 82, 82)
DGRAY = (57, 57, 57)
MGRAY = (49, 49, 49)
MID = (123, 123, 123)
WHITE = (255, 255, 255)
SILVER = (180, 180, 180)
FACE = (230, 172, 222)
PINK = (189, 131, 164)
PINK_D = (148, 90, 123)
BROWN = (156, 82, 65)
EYE = (8, 115, 205)
EYE_L = (57, 164, 255)
LIP = (255, 106, 74)

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
    """4x nearest rotation. Positive tips the top toward screen-right. No squash."""
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
        up = np.zeros_like(src)
        dn = np.zeros_like(src)
        lt = np.zeros_like(src)
        rt = np.zeros_like(src)
        up[:-1] = src[1:]
        dn[1:] = src[:-1]
        lt[:, :-1] = src[:, 1:]
        rt[:, 1:] = src[:, :-1]
        for g in (up, dn, lt, rt):
            take = (acc[..., 3] < 128) & allowed & (g[..., 3] >= 128)
            acc[take] = g[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def hump(f, n, delay, width=5):
    """One swing lobe. 0 at frame 0 for delays 0 and 8 when width is 5."""
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


def segment(img, kind):
    """Bows and hands only. The dress, the face, and the hem stay on the torso."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    bow_l = op & (yy <= 40) & (xx <= 39)
    bow_r = op & (yy <= 40) & (xx >= 56)
    if kind == "front":
        # Loop shines sit just inside the cut, above the face.
        bow_l = bow_l | (op & (yy <= 22) & (xx >= 40) & (xx <= 44))
        bow_r = bow_r | (op & (yy <= 24) & (xx >= 51) & (xx <= 55))
    hand_l = op & (yy >= 48) & (yy <= 62) & (xx <= 36)
    hand_r = op & (yy >= 48) & (yy <= 62) & (xx >= 59)
    used = bow_l | bow_r | hand_l | hand_r
    torso = op & ~used
    masks = {
        "bow_l": bow_l, "bow_r": bow_r, "hand_l": hand_l, "hand_r": hand_r,
        "torso": torso,
    }
    check_partition(op, masks)
    expect = {
        "front": {"bow_l": 353, "bow_r": 380, "hand_l": 60, "hand_r": 64, "torso": 1161},
        "back": {"bow_l": 314, "bow_r": 324, "hand_l": 59, "hand_r": 63, "torso": 1230},
    }[kind]
    counts = {k: int(v.sum()) for k, v in masks.items()}
    if counts != expect:
        raise SystemExit(f"{kind} parts changed {counts}")
    for name in LIMBS:
        part = masks[name]
        if largest_component(part) != int(part.sum()):
            raise SystemExit(f"{kind} {name} fell apart")
        if not touches(part, torso):
            raise SystemExit(f"{kind} {name} does not touch the body")
    eye = any_color(img, (EYE, EYE_L))
    lip = is_color(img, LIP)
    brown = is_color(img, BROWN)
    face = is_color(img, FACE)
    if kind == "front":
        if int(eye.sum()) != 7 or int((eye & torso).sum()) != 7:
            raise SystemExit("front eyes left the face")
        if int(lip.sum()) != 2 or int((lip & torso).sum()) != 2:
            raise SystemExit("front mouth left the face")
        if int(brown.sum()) != 7 or int((brown & torso).sum()) != 7:
            raise SystemExit("front mouth brown left the face")
        if int((face & (bow_l | bow_r)).sum()) != 0:
            raise SystemExit("face pink was cut into a bow")
        if int((face & (hand_l | hand_r)).sum()) != 19:
            raise SystemExit("hand palms changed")
        ys, xs = np.nonzero(eye)
        left = float(xs[xs < 47].mean())
        right = float(xs[xs >= 47].mean())
        mouth_x = float(np.nonzero(lip)[1].mean())
        mid = (left + right) / 2.0
        if not (left + 1 < mouth_x < right - 1):
            raise SystemExit(f"mouth is not between the eyes: {mouth_x} vs {left} {right}")
        if abs(mouth_x - mid) > 1.5:
            raise SystemExit(f"front facing changed: mouth {mouth_x:.2f} eyes {mid:.2f}")
        if int(np.nonzero(op)[0].max()) != FRONT_FOOT:
            raise SystemExit("front hem moved")
    else:
        if eye.any() or lip.any() or brown.any() or face.any():
            raise SystemExit("back gained a face")
        if int(np.nonzero(op)[0].max()) != BACK_FOOT:
            raise SystemExit("back hem moved")
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
    canvas = place(img, 0)
    placed = {k: place(v, 0) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    depth = {"bow_l": 5, "bow_r": 5, "hand_l": 6, "hand_r": 6}
    for name in LIMBS:
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth[name]), depth[name],
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
    mouth = is_color(canvas, LIP)
    has_mouth = bool(mouth.any())
    if has_mouth:
        my, mx = np.nonzero(mouth)
        pivots["mouth"] = (float(mx.mean()), float(my.mean()))
    else:
        pivots["mouth"] = pivots["center"]
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "has_mouth": has_mouth,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        bow_l=0.0, bow_r=0.0, hand_l=0.0, hand_r=0.0,
        bow_mode="up", hand_mode="up",
        lean=0.0, knock=0, flash=False, spark=0,
        wedge=0, ring=-1,
        turned=False, spin=0, head_dx=0, head_dy=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # No feet. The skirt floats. The bow loops and the hands breathe.
        p["bow_l"] = wave(f, n, 5.0)
        p["bow_r"] = wave(f, n, 5.0, phase=math.pi)
        p["hand_l"] = wave(f, n, 3.0, phase=0.4)
        p["hand_r"] = wave(f, n, 3.0, phase=0.4 + math.pi)
        p["bdy"] = -1 if wave(f, n, 1.0) > 0.55 else 0
        p["bow_mode"] = "up"
        p["hand_mode"] = "up"
    elif action == "walk":
        # Glide. The drawn hands swing. No legs are added.
        p["hand_l"] = 12.0 * hump(f, n, 0)
        p["hand_r"] = 12.0 * hump(f, n, 8)
        p["bow_l"] = wave(f, n, 6.0, cycles=2.0)
        p["bow_r"] = wave(f, n, 6.0, phase=math.pi, cycles=2.0)
        p["bdx"] = int(round(wave(f, n, 1.0, cycles=2.0)))
        p["bdy"] = -1 if abs(wave(f, n, 1.0, cycles=2.0)) > 0.55 else 0
        p["bow_mode"] = "up"
        p["hand_mode"] = "fwd"
    elif action == "attack":
        # 0 rest. 1-4 roar and lift the bows. 5-8 the back, cleared below.
        # 9 hand wind-up (18). 10-12 both hands strike. 13-15 return.
        bow = [0, 6, 10, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        hand = [0, 4, 8, 12, 14, 0, 0, 0, 0, 18, 28, 34, 34, 12, 4, 0]
        bdx = [0, -1, -1, -1, -1, 0, 0, 0, 0, -1, 1, 2, 2, 1, 0, 0]
        bdy = [0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        wedge = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ring = [-1, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        p["bow_l"] = float(bow[f])
        p["bow_r"] = float(bow[f])
        p["hand_l"] = float(hand[f])
        p["hand_r"] = float(hand[f])
        p["bdx"] = int(bdx[f])
        p["bdy"] = int(bdy[f])
        p["wedge"] = int(wedge[f])
        p["ring"] = int(ring[f])
        p["bow_mode"] = "up"
        p["hand_mode"] = "fwd" if f >= 9 else "up"
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.64, 0.44, 0.26, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["bow_l"] = 8.0 * env
        p["bow_r"] = 8.0 * env
        p["hand_l"] = 6.0 * env
        p["hand_r"] = 6.0 * env
        p["bow_mode"] = "up"
        p["hand_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        # The other side is the still, at equal scale. finish() must not swing it.
        for key in ("bow_l", "bow_r", "hand_l", "hand_r", "lean"):
            p[key] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["knock"] = 0
        p["wedge"] = 0
        p["ring"] = -1
        p["spark"] = 0
        p["spin"] = 0
        p["head_dx"] = 0
        p["head_dy"] = 0
        p["flash"] = False
        p["turned"] = True
    if abs(float(p["hand_l"])) > 34.05 or abs(float(p["hand_r"])) > 34.05:
        raise SystemExit(f"hand strike passed 34 deg on {action} {f}")
    if action == "hurt" and float(p["lean"]) > MAX_LEAN + 0.05:
        raise SystemExit("hurt lean passed 38 deg")
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
    for name in ("bow_l", "bow_r"):
        ang = part_angle(rig, name, p["bow_mode"], p[name])
        paste(out, moved(L[name], ang, piv[name], bdx, bdy))
    for name in ("hand_l", "hand_r"):
        ang = part_angle(rig, name, p["hand_mode"], p[name])
        paste(out, moved(L[name], ang, piv[name], bdx, bdy))
    return out


def flash_hit(img):
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


def hand_tip(rig, p):
    """Tip of the hand that reaches farthest along facing, after the strike."""
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    best = rig["pivots"]["center"]
    best_x = -1e9 if fc > 0 else 1e9
    for name in ("hand_l", "hand_r"):
        ang = part_angle(rig, name, p["hand_mode"], p[name])
        layer = moved(L[name], ang, piv[name], bdx, bdy)
        m = layer[..., 3] >= 128
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
        x = float(xs[i])
        if (fc > 0 and x > best_x) or (fc < 0 and x < best_x):
            best_x = x
            best = (x, float(ys[i]))
    return best


def mouth_wedge(img, origin, fc, age):
    """A few pixels in front of the drawn mouth. Empty pixels only."""
    ox, oy = origin
    cols = (LIP, BROWN, INK)
    for i in range(max(0, int(age))):
        x = ox + fc * (2 + i)
        span = 1 if i == 0 else 2
        for dy in range(-span, span + 1):
            put(img, x, oy + dy, cols[min(i, 2)])


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, FACE, SILVER)
    rad = 6 + int(age) * 3
    for deg in range(-48, 49, 8):
        a = math.radians(deg)
        x = ox + fc * rad * math.cos(a)
        y = oy + rad * math.sin(a) * 0.55
        put(img, x, y, cols[int(age) % 3])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, PINK, SILVER)
    r = 2 + int(age) * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (1, -r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def rig_for(rigs, d, p):
    """Attack 5-8 faces away. Front rows become that side's back, unscaled."""
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", not mirror)]


def moving(p):
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    if abs(p["lean"]) > 0.4:
        return True
    for key in ("bow_l", "bow_r", "hand_l", "hand_r"):
        if abs(p[key]) >= 0.45:
            return True
    return False


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    for key in ("bow_l", "bow_r", "hand_l", "hand_r"):
        if abs(p[key]) >= 0.45:
            return True
    return False


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    if not moving(p):
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            body = rotate_nearest(body, -fc * p["lean"], rear_pivot(body, fc))
        if rotated(p):
            body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_hit(img), palette)
    if rig["has_mouth"] and not p["turned"]:
        mx, my = rig["pivots"]["mouth"]
        origin = (mx + int(round(fc * p["bdx"])), my + int(round(p["bdy"])))
        drew = False
        if p["wedge"]:
            mouth_wedge(img, origin, fc, p["wedge"])
            drew = True
        if p["ring"] >= 0:
            roar_rings(img, origin, fc, p["ring"])
            drew = True
        if drew:
            img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = hand_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def limb_lost(rig, p):
    if not moving(p):
        return []
    body = compose(rig, p)
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    lost = []
    for name in LIMBS:
        mode = p["bow_mode"] if name.startswith("bow") else p["hand_mode"]
        ang = part_angle(rig, name, mode, p[name])
        layer = moved(rig["layers"][name], ang, rig["pivots"][name], bdx, bdy)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


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
    acc = np.zeros(canvases[0].shape[:2], bool)
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


def still_colors(*imgs):
    cols = []
    seen = set()
    for img in imgs:
        for c in np.unique(img[img[..., 3] >= 128][:, :3], axis=0):
            t = tuple(int(v) for v in c)
            if t not in seen:
                seen.add(t)
                cols.append(t)
    return cols


def locked_palette(front_path, back_path, work):
    front = np.array(Image.open(front_path).convert("RGBA"))
    back = np.array(Image.open(back_path).convert("RGBA"))
    own = still_colors(front, back)
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
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
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if not rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("mouth is on the wrong view")
    if rigs["W"]["facing"] != -1.0 or rigs["E"]["facing"] != 1.0:
        raise SystemExit("E/W facing is not screen-right / screen-left")
    if rigs["N"]["facing"] != 1.0 or rigs["NW"]["facing"] != -1.0:
        raise SystemExit("back facing changed")
    # Rest compose must equal the still, so a later motion does not pop.
    rest = base_pose()
    for key, rig in list(rigs.items()):
        if not isinstance(key, tuple):
            continue
        diff = int(np.any(compose(rig, rest) != rig["canvas"], axis=2).sum())
        if diff:
            raise SystemExit(f"rest compose differs for {key}: {diff}")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "bow_l": (255, 220, 40, 255),
        "bow_r": (255, 80, 40, 255),
        "hand_l": (40, 200, 90, 255),
        "hand_r": (40, 160, 220, 255),
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
        "dex": 576, "form": "normal", "variant": "gothitelle", "gen": "gen5",
        "dir": "monsterSprite/gen5/0576_gothitelle/normal", "sheetDir": "agent-sprite-forge",
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


def check_turns(rigs, palette):
    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
    if any(rest_diff.values()):
        raise SystemExit(f"frame 0 does not match the still: {rest_diff}")
    for d, back_d in (("S", "N"), ("E", "N"), ("SE", "N"), ("W", "NW"), ("SW", "NW")):
        got = synth(rigs, d, "attack", 6, palette, effects=False)
        if not np.array_equal(got, rigs[back_d]["canvas"]):
            raise SystemExit(f"attack turn {d} is not the {back_d} still")
    for d in ("N", "NE", "NW"):
        got = synth(rigs, d, "attack", 7, palette, effects=False)
        if not np.array_equal(got, rigs[d]["canvas"]):
            raise SystemExit(f"attack turn {d} changed the back still")
    for d in DIRS:
        a15 = synth(rigs, d, "attack", 15, palette, effects=False)
        if not np.array_equal(a15, rigs[d]["canvas"]):
            raise SystemExit(f"attack frame 15 does not match the still for {d}")
    return rest_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    args = ap.parse_args()

    save_peaks(Path(__file__).stem, PEAKS)
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    palette = locked_palette(SRC_DIR / "front.png", SRC_DIR / "back.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, "mouth", 1, flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    rest_diff = check_turns(rigs, palette)
    print("rest diff", rest_diff, flush=True)

    if args.samples:
        lost = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        print("limb lost", lost[:12] or "none", "count", len(lost), flush=True)
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
            atk.append((f, round(float(p["hand_l"]), 1), p["hand_mode"],
                        round(float(p["bow_l"]), 1), p["turned"], p["spin"],
                        p["wedge"], p["ring"]))
        print("attack poses", atk, flush=True)
        leans = [round(float(pose("hurt", f)["lean"]), 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        box = content_box([rigs[d]["canvas"] for d in DIRS])
        y0, y1, x0, x1 = box
        tiles = []
        for d in ("S", "W", "N", "E"):
            for act, f in (
                ("idle", 0), ("idle", 4),
                ("walk", 2), ("walk", 10),
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
            if key in state["done"] and not args.force:
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
    box = content_box([rigs[d]["canvas"] for d in DIRS])
    motion = np.concatenate([motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")], axis=0)
    Image.fromarray(np.repeat(np.repeat(motion, 2, 0), 2, 1), "RGBA").save(OUT_DIR / "preview_motion.png")
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
        "off_palette": len(off), "partial_alpha": partial,
        "edge_touch": len(clip), "edge_touching": clip[:8],
        "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    print("qa_full", "rest_diff", int(sum(rest_diff.values())),
          "off_palette", report["off_palette"],
          "partial_alpha", report["partial_alpha"],
          "edge_touch", report["edge_touch"],
          "body_main_min", report["body_main_min"],
          "clip", report["edge_touching"], flush=True)
    ok = (not any(rest_diff.values()) and not any(idle0.values()) and not off
          and partial == 0 and not clip)
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    meta = {
        "generator": "forge_gen5_gothitelle_actions", "version": VERSION, "key": KEY,
        "dex": 576, "variant": "gothitelle", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "dress humanoid (two bow loops, two hands, no feet, mouth on the front)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "mouth mean x 46.5 sits between the eyes (midpoint 46.79), so the front faces the viewer; back mass is slightly screen-right",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
