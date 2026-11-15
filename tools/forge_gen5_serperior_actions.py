#!/usr/bin/env python3
"""Serperior normal (0497) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x9-85 y14-80, 2533 px, alpha 0/255, 14 colours):
  Iris (brown + orange) sits at x53-59 y21-22, mean x 56.2. The snout tip is the
  opaque pixel at x40 y26-28, left of the eyes, so the still faces screen-left.
  No mouth is drawn: the white mass is the jaw and throat, and the brown/orange
  pixels are the eye. The roar does not invent a mouth, a wedge, or rings.
  It raises the two crown leaves that are already on the skull.
  Crown: left prong x61-69 y14-18, right prong x67-82 y16-22, rooted on the head.
  Side vine: the curled leaf at x72-85 y30-41, rooted on the neck.
  Tail: the raised left coil and its three-prong tuft, x10-37 y42-59, rooted
  where it meets the body. The bottom coil, belly gold, and chin leaf stay
  on the torso. There are no legs.
normal/back.png (bbox x10-85 y12-83, foot y83) shows the white snout to the
  right of the eyes (iris mean x 69.5, snout white reaches x79), so this back
  faces screen-right. Crown prongs are the green blades above the skull.
  The side vine is the right curl. The tail is the left raised coil.
  The back is shifted up 3px so both coils share a ground line.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (snout screen-right). NW back mirrored (snout screen-left).
  E is screen-right, W is screen-left, N is the back. Attack 5-8 swaps to the
  matching back at equal scale. No width squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_serperior_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0497_serperior/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0497_serperior/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0497_serperior"
VERSION = "serperior_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 240, 240
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 22
FRONT_FOOT = 80
BACK_FOOT = 83
FOOT_Y = OFF_Y + FRONT_FOOT
# kind, mirror. Front as drawn looks screen-left. Back as drawn looks screen-right.
# S (front mirrored) turns into N (back as drawn) by flipping the mirror flag.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Roar / turn hold. The whip itself is drawn at 34 deg, above these peaks.
PEAKS = {"crest": 22.0, "vine": 16.0, "tail": 16.0}
LIMBS = ("crest_l", "crest_r", "vine", "tail")

INK = (8, 16, 16)
BRIGHT = (24, 156, 74)
PALE = (156, 214, 156)
MID = (16, 115, 57)
LEAF = (115, 165, 115)
DEEP = (16, 82, 41)
YELLOW = (255, 206, 41)
OLIVE = (107, 115, 57)
WHITE = (255, 255, 255)
GOLD = (214, 156, 8)
LAV = (198, 198, 214)
DLAV = (140, 140, 173)
BROWN = (148, 57, 0)
ORANGE = (255, 132, 0)
FACE_COLS = (WHITE, LAV, DLAV, BROWN, ORANGE)
IRIS_COLS = (BROWN, ORANGE)

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


def split_lr(mask, what):
    groups = component_list(mask)
    if len(groups) != 2:
        raise SystemExit(f"{what} crown is not two leaves: {[len(g) for g in groups]}")
    scored = []
    for pts in groups:
        xs = [p[1] for p in pts]
        scored.append((float(sum(xs)) / len(xs), pts))
    scored.sort()
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)
    for y, x in scored[0][1]:
        left[y, x] = True
    for y, x in scored[1][1]:
        right[y, x] = True
    return left, right


def attach_specks(src, dst, limit=3):
    """Tiny src specks that 4-touch dst join dst. A diagonal speck stays put."""
    h, w = src.shape
    moved = np.zeros_like(src)
    for pts in component_list(src):
        if len(pts) > limit:
            continue
        touch = False
        for y, x in pts:
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and dst[ny, nx]:
                    touch = True
                    break
            if touch:
                break
        if touch:
            for y, x in pts:
                moved[y, x] = True
    return src & ~moved, dst | moved


def touches(part, body):
    return bool((part & dilate(body, 1)).any())


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    yel = is_color(img, YELLOW)
    gold = is_color(img, GOLD)
    olive = is_color(img, OLIVE)
    face = any_color(img, FACE_COLS)
    # Crown leaves only. The face column under them stays on the head.
    allowed = ((yel | gold | olive) & (yy <= 22) & (xx >= 60))
    allowed |= op & (yy <= 20) & (xx >= 66) & ~face
    allowed |= op & (yy <= 26) & (xx >= 74) & ~face
    allowed &= ~((yy >= 19) & (xx <= 66))
    crest = flood(allowed, yel & (yy <= 22) & (xx >= 60))
    vine = flood(
        op & (xx >= 72) & (yy >= 28) & (yy <= 44) & ~crest,
        op & (xx >= 80) & (yy >= 30) & (yy <= 42),
    )
    tail = flood(
        op & (xx <= 40) & (yy >= 42) & (yy <= 59) & ~crest & ~vine,
        op & (xx <= 32) & (yy >= 42) & (yy <= 50),
    )
    torso = op & ~crest & ~vine & ~tail
    crest_l, crest_r = split_lr(crest, "front")
    torso, crest_r = attach_specks(torso, crest_r)
    torso, crest_l = attach_specks(torso, crest_l)
    torso, vine = attach_specks(torso, vine)
    torso, tail = attach_specks(torso, tail)
    masks = {"crest_l": crest_l, "crest_r": crest_r, "vine": vine, "tail": tail, "torso": torso}
    check_partition(op, masks)
    iris = any_color(img, IRIS_COLS)
    if int((iris & torso).sum()) != int(iris.sum()) or iris.sum() < 8:
        raise SystemExit("front eyes left the head")
    if (face & (crest_l | crest_r)).any():
        raise SystemExit("front crown ate the face")
    head = op & (yy >= 14) & (yy <= 40)
    tip = int(np.nonzero(head)[1].min())
    ex = float(np.nonzero(iris)[1].mean())
    if not tip < ex - 8:
        raise SystemExit(f"front does not face left: snout {tip} eye {ex:.1f}")
    for name in LIMBS:
        if not touches(masks[name], torso):
            raise SystemExit(f"front {name} does not touch the body")
        if largest_component(masks[name]) != int(masks[name].sum()):
            raise SystemExit(f"front {name} fell apart")
    if largest_component(torso) < int(torso.sum()) - 2:
        raise SystemExit("front torso fell apart")
    if int(tail.sum()) < 180 or int(vine.sum()) < 40 or int((crest_l | crest_r).sum()) < 60:
        raise SystemExit("front limbs look short")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    white = is_color(img, WHITE)
    face = any_color(img, FACE_COLS)
    # Green crown blades above the skull. The white snout and the eyes stay.
    allowed = op & ~white & ~face & (((yy <= 17) & (xx <= 82)) | ((yy <= 23) & (xx <= 58)))
    crest = flood(allowed, op & (yy <= 16) & ~white)
    vine = flood(
        op & (xx >= 74) & (yy >= 29) & (yy <= 42) & ~crest,
        op & (xx >= 80) & (yy >= 30) & (yy <= 38),
    )
    tail = flood(
        op & (xx <= 44) & (yy >= 34) & (yy <= 52) & ~crest & ~vine,
        op & (xx <= 28) & (yy >= 34) & (yy <= 46),
    )
    torso = op & ~crest & ~vine & ~tail
    crest_l, crest_r = split_lr(crest, "back")
    torso, crest_r = attach_specks(torso, crest_r)
    torso, crest_l = attach_specks(torso, crest_l)
    torso, vine = attach_specks(torso, vine)
    torso, tail = attach_specks(torso, tail)
    masks = {"crest_l": crest_l, "crest_r": crest_r, "vine": vine, "tail": tail, "torso": torso}
    check_partition(op, masks)
    iris = any_color(img, IRIS_COLS)
    if int((iris & torso).sum()) != int(iris.sum()) or iris.sum() < 6:
        raise SystemExit("back eyes left the head")
    if (face & (crest_l | crest_r | vine | tail)).any():
        raise SystemExit("back limb ate the face")
    snout = white & (yy < 36)
    if not snout.any():
        raise SystemExit("back lost the snout")
    if float(np.nonzero(snout)[1].max()) <= float(np.nonzero(iris)[1].mean()) + 4:
        raise SystemExit("back snout is not right of the eyes")
    for name in LIMBS:
        if not touches(masks[name], torso):
            raise SystemExit(f"back {name} does not touch the body")
        if largest_component(masks[name]) != int(masks[name].sum()):
            raise SystemExit(f"back {name} fell apart")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("back torso fell apart")
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
    touch = part & dilate(torso, 2)
    if not touch.any():
        touch = part & dilate(torso, 5)
    band = touch.copy()
    for _ in range(depth):
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
    # A short root cushion only. A deep one would leave a ghost of the vine.
    for name, depth in (("crest_l", 4), ("crest_r", 4), ("vine", 3), ("tail", 5)):
        layers["torso"] = paint_cushion(layers["torso"], joint_band(placed[name], torso_m, depth), depth + 1)
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
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        crest=0.0, crest_mode="up",
        vine=0.0, vine_mode="up",
        tail=0.0, tail_mode="up",
        tail_dx=0,
        lean=0.0, knock=0, flash=False, spark=0, turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        breath = wave(f, n, 1.0)
        p["bdy"] = -1 if breath > 0.55 else 0
        p["crest"] = wave(f, n, 6.0, phase=0.4)
        p["vine"] = wave(f, n, 5.0, phase=1.5)
        p["tail"] = wave(f, n, 8.0, phase=0.8)
    elif action == "walk":
        # No legs. The raised tail, crown, and side vine undulate. The coil glides.
        p["bdy"] = -1 if wave(f, n, 1.0, cycles=2.0) > 0.35 else 0
        p["bdx"] = int(round(wave(f, n, 1.0, phase=0.3)))
        p["crest"] = wave(f, n, 8.0, phase=1.1)
        p["vine"] = wave(f, n, 9.0, phase=2.4)
        p["tail"] = wave(f, n, 14.0, phase=0.5)
        p["tail_dx"] = int(round(wave(f, n, 1.0, phase=0.5, cycles=2.0)))
    elif action == "attack":
        # 0 rest. 1-4 raise the crown (no mouth). 5-8 back view.
        # 9-12 tail and crown whip. 13-15 recover. 15 is the still.
        crest = [0, 8, 14, 20, 22, 22, 22, 22, 22, 18, 28, 34, 34, 16, 6, 0]
        tail = [0, 4, 8, 12, 16, 16, 16, 16, 16, 12, 24, 34, 30, 12, 4, 0]
        vine = [0, 6, 10, 14, 16, 16, 16, 16, 16, 10, 18, 22, 18, 8, 3, 0]
        p["crest"] = float(crest[f])
        p["tail"] = float(tail[f])
        p["vine"] = float(vine[f])
        strike = f >= 9
        p["crest_mode"] = "fwd" if strike else "up"
        p["tail_mode"] = "fwd" if strike else "up"
        p["vine_mode"] = "fwd" if strike else "up"
        if f in (3, 4):
            p["bdx"] = -1
            p["bdy"] = -1
        if f in (11, 12):
            p["bdx"] = 1
        if f in (11, 12):
            p["spark"] = f - 10
        p["turned"] = 5 <= f <= 8
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["crest"] = 12.0 * env
        p["tail"] = 14.0 * env
        p["vine"] = 8.0 * env
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


def limb_pose(p, name):
    if name.startswith("crest"):
        return p["crest"], p["crest_mode"], 0
    if name == "vine":
        return p["vine"], p["vine_mode"], 0
    return p["tail"], p["tail_mode"], p["tail_dx"]


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    for name in ("tail", "vine", "crest_l", "crest_r"):
        amp, mode, extra = limb_pose(p, name)
        ang = part_angle(rig, name, mode, amp)
        dx = bdx + int(round(fc * extra))
        paste(out, moved(L[name], ang, piv[name], dx, bdy))
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


def tail_tip(rig, p):
    ang = part_angle(rig, "tail", p["tail_mode"], p["tail"])
    dx = int(round(rig["facing"] * p["bdx"])) + int(round(rig["facing"] * p["tail_dx"]))
    layer = moved(rig["layers"]["tail"], ang, rig["pivots"]["tail"], dx, int(p["bdy"]))
    return tip_xy(layer, rig["pivots"]["tail"])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (YELLOW, WHITE, BRIGHT)
    r = 4 + age * 3
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (1, -r), (1, r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def limb_layers(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    parts = {}
    for name in ("tail", "vine", "crest_l", "crest_r"):
        amp, mode, extra = limb_pose(p, name)
        ang = part_angle(rig, name, mode, amp)
        dx = bdx + int(round(fc * extra))
        parts[name] = moved(L[name], ang, rig["pivots"][name], dx, bdy)
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


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rigs[d]
    if p["turned"]:
        kind, mirror = VIEW_OF[d]
        rig = rigs[("back" if kind == "front" else "front", not mirror)]
    body = compose(rig, p)
    moved_part = abs(p["crest"]) >= 0.45 or abs(p["vine"]) >= 0.45 or abs(p["tail"]) >= 0.45
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
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = tail_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
    return snap(img, palette)


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


def crop_box(canvas, pad=10):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad - 28)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 16)
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


def locked_palette(front_path, work):
    img = np.array(Image.open(front_path).convert("RGBA"))
    own = np.unique(img[img[..., 3] >= 128][:, :3], axis=0)
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
    if not (lock_dir / "palette.json").exists():
        py = str(FORGE / ".venv/bin/python")
        hexes = ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in own]
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
    want = {tuple(int(v) for v in c) for c in own}
    if got != want:
        raise SystemExit(f"locked palette differs from the still: {sorted(got ^ want)}")
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
        "crest_l": (220, 40, 40, 255),
        "crest_r": (255, 120, 40, 255),
        "vine": (40, 120, 255, 255),
        "tail": (240, 210, 30, 255),
        "torso": (170, 170, 170, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 4, 0), 4, 1))
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
        "dex": 497, "form": "normal", "variant": "serperior", "gen": "gen5",
        "dir": "monsterSprite/gen5/0497_serperior/normal", "sheetDir": "agent-sprite-forge",
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
    palette = locked_palette(SRC_DIR / "front.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, flush=True)

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        y0, y1, x0, x1 = box
        for d in ("S", "W", "N", "E"):
            frames = []
            for act, fs in (
                ("idle", (0, 4, 8, 12)),
                ("walk", (0, 2, 4, 6, 8, 10, 12, 14)),
                ("attack", (0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15)),
                ("hurt", (0, 1, 3, 6, 11)),
            ):
                for f in fs:
                    frames.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
            sheet = tile_grid(frames, len(frames))
            z = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
            Image.fromarray(z, "RGBA").save(work / f"samples_{d}.png")
        lost = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rigs[d]
                    if p["turned"]:
                        kind, mirror = VIEW_OF[d]
                        rig = rigs[("back" if kind == "front" else "front", not mirror)]
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["crest"], 1), round(p["tail"], 1), p["crest_mode"], p["turned"], p["spin"]))
        print("attack poses", atk, flush=True)
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
    ok = (not any(rest_diff.values()) and not any(idle0.values()) and not off
          and partial == 0 and not clip and min(bodies) >= 0.9)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_serperior_actions", "version": VERSION, "key": KEY,
        "dex": 497, "variant": "serperior", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "serpent coil (no legs, no mouth; crown, side vine, tail tuft)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (snout x < eye x); back snout is right of the eyes",
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
