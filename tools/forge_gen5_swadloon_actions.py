#!/usr/bin/env python3
"""Swadloon normal (0541) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x24-72 y25-65, 985 px, alpha 0/255,
12 colours, foot y65):
  Yellow face x35-51 y45-54, mean x 42.47. Leftmost yellow is x35 y50.
  Eyes (olive, olive-brown, white) mean x 43.43, so the face tip is left of
  the eyes. The still faces screen-left. No hands, fangs, wings, horns, or
  pincers. The cloak is a leaf.
  Two leaf prongs above y33: left x37-42 (37 px), right x45-51 (42 px, the
  apex at y25). They are the hood.
  Leaf hem corners, not feet: left x24-34 y58-64 (56 px), right x62-72 y58-64
  (58 px). A 7 px nub at y65 x49-55 is the bottom leaf point.
  Mouth is two ink pixels at x41-42 y54, under the yellow. The roar may use
  a wedge and a ring. Nothing else is a mouth.

normal/back.png (bbox x20-68 y25-71, 1068 px, 7 colours, foot y71) has no
yellow, white, or eye colour. The apex prong is the left one (42 px), so this
still is the front turned away to screen-right. The bottom leaf point hangs
to y71 (64 px). Side hems stay on the leaf. No face, so no roar on the back.

Directions, front faces left, back faces away-right:
  S SE E front mirrored (screen-right). SW W front as drawn (screen-left).
  NE back as drawn. N NW back mirrored.
  E is screen-right, W is screen-left, N is the back.
  Attack 5-8 swaps to the opposite still at equal scale. No width squash,
  no in-plane body spin.

  idle 16: no feet, a 1px float, hood and hem and the bottom point sway.
  walk 16: a 1px glide. The hood, hems, and bottom point ripple. No new legs.
  attack: 0 still, 1-4 hood rises and the mouth roars, 5-8 the other side,
          9-12 the hood tips and the leaf hems strike, 11-12 impact,
          13-15 recover.
  hurt 12: white flash, knock-back lean (<=38 deg) and recover.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_swadloon_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0541_swadloon/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0541_swadloon/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0541_swadloon"
VERSION = "swadloon_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 192
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 24
FRONT_FOOT = 65
BACK_FOOT = 71
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. Back as drawn looks away to the right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "SW": ("front", False), "W": ("front", False),
    "NE": ("back", False), "N": ("back", True), "NW": ("back", True),
}
MAX_LEAN = 38.0
# The strike already draws these. finish() will not raise a frame past them.
PEAKS = {"tip": 34.0, "hem": 22.0, "tail": 18.0}

LIME = (148, 205, 8)
INK = (16, 16, 16)
MID = (90, 156, 49)
YELLOW = (255, 238, 57)
TEAL = (24, 115, 74)
OLIVE = (57, 98, 24)
DARK = (16, 65, 57)
TEAL2 = (41, 156, 82)
GOLD = (180, 164, 41)
EYE = (90, 90, 57)
EYE_B = (123, 115, 57)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
EYE_COLS = (EYE, EYE_B, WHITE)
FACE_COLS = (YELLOW, GOLD, EYE, EYE_B, WHITE)

SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)
PARTS = ("tip_l", "tip_r", "hem_l", "hem_r", "tail")


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
    """4x nearest in-plane rotation. Positive follows the shared forge formula. No squash."""
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


def require_piece(name, mask):
    if not mask.any():
        raise SystemExit(f"{name} is empty")
    if largest_component(mask) != int(mask.sum()):
        raise SystemExit(f"{name} fell apart")


def mask_from(pts, shape):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m


def segment(img, kind):
    """Leaf hood prongs, hem corners, and the bottom leaf point. No invented limbs."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    prongs = sorted(component_list(op & (yy <= 33)), key=lambda g: float(np.mean([p[1] for p in g])))
    if len(prongs) != 2:
        raise SystemExit(f"{kind} hood is not two prongs")
    tip_l = mask_from(prongs[0], op.shape)
    tip_r = mask_from(prongs[1], op.shape)
    if kind == "front":
        tail = op & (yy >= 65)
    else:
        by = int(np.nonzero(op)[0].max())
        tail = flood(op & (yy >= 66), op & (yy == by))
    hem_l = op & (xx <= 34) & (yy >= 58) & ~tail & ~tip_l & ~tip_r
    hem_r = op & (xx >= 62) & (yy >= 58) & ~tail & ~tip_l & ~tip_r
    used = tip_l | tip_r | hem_l | hem_r | tail
    torso = op & ~used
    masks = {
        "tip_l": tip_l, "tip_r": tip_r,
        "hem_l": hem_l, "hem_r": hem_r,
        "tail": tail, "torso": torso,
    }
    check_partition(op, masks)
    expect = {
        "front": {"tip_l": 37, "tip_r": 42, "hem_l": 56, "hem_r": 58, "tail": 7, "torso": 785},
        "back": {"tip_l": 42, "tip_r": 37, "hem_l": 85, "hem_r": 39, "tail": 64, "torso": 801},
    }[kind]
    for name, n in expect.items():
        if int(masks[name].sum()) != n:
            raise SystemExit(f"{kind} {name} size {int(masks[name].sum())} != {n}")
    for name in PARTS:
        require_piece(name, masks[name])
        if not (masks[name] & dilate(torso, 1)).any():
            raise SystemExit(f"{kind} {name} does not touch the body")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit(f"{kind} torso fell apart")
    foot = FRONT_FOOT if kind == "front" else BACK_FOOT
    if int(np.nonzero(op)[0].max()) != foot:
        raise SystemExit(f"{kind} foot line changed")
    if kind == "front":
        eye = any_color(img, EYE_COLS)
        face = is_color(img, YELLOW)
        if not eye.any() or (eye & ~torso).any():
            raise SystemExit("front eye left the head")
        if int(np.nonzero(face)[1].min()) >= float(np.nonzero(eye)[1].mean()):
            raise SystemExit("front face tip is not left of the eyes")
        mouth = mouth_mask(img)
        if int(mouth.sum()) != 2 or (mouth & ~torso).any():
            raise SystemExit(f"front mouth is not the two ink pixels ({int(mouth.sum())})")
    else:
        if any_color(img, FACE_COLS).any() or is_color(img, INK).any():
            raise SystemExit("back grew a face")
        apex = tip_l if (tip_l & (yy == int(np.nonzero(op)[0].min()))).any() else tip_r
        if float(np.nonzero(apex)[1].mean()) >= float(np.nonzero(op)[1].mean()):
            raise SystemExit("back apex is not on the left")
    return masks


def place(arr):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    y0, x0 = OFF_Y, OFF_X
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


def pivot_of(part, torso):
    """Contact pixel of the part nearest the torso."""
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 4)
    if not touch.any():
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(torso)
    cx, cy = float(tx.mean()), float(ty.mean())
    d = (xs.astype(np.float64) - cx) ** 2 + (ys.astype(np.float64) - cy) ** 2
    near = d <= d.min() + 0.5
    return float(xs[near].mean()), float(ys[near].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs.astype(np.float64) - pivot[0]) ** 2 + (ys.astype(np.float64) - pivot[1]) ** 2
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


def mouth_mask(img):
    """Two ink pixels in a row, with yellow directly above. Empty on the back."""
    face = is_color(img, YELLOW)
    ink = is_color(img, INK)
    above = np.zeros(face.shape, bool)
    above[1:] = face[:-1]
    left = np.zeros_like(ink)
    right = np.zeros_like(ink)
    left[:, 1:] = ink[:, :-1]
    right[:, :-1] = ink[:, 1:]
    return ink & above & (left | right)


def build_view(img, masks, kind, mirror):
    canvas = place(img)
    placed = {k: place(v) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    if kind == "front":
        facing = -1.0 if not mirror else 1.0
    else:
        facing = 1.0 if not mirror else -1.0
    pivots = {}
    signs = {}
    for name in PARTS:
        depth = 4 if name.startswith("tip") else 3
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth
        )
        pivots[name] = pivot_of(placed[name], torso_m)
        signs[name] = signs_for(layers[name], pivots[name], facing)
    mouth = mouth_mask(canvas)
    has_mouth = bool(mouth.any())
    if has_mouth:
        my, mx = np.nonzero(mouth)
        pivots["mouth"] = (float(mx.mean()), float(my.mean()))
        if int(mouth.sum()) != 2:
            raise SystemExit(f"mouth count changed after place ({int(mouth.sum())})")
    else:
        pivots["mouth"] = None
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas, "has_mouth": has_mouth,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        tip=0.0, tip_mode="up",
        hem=0.0, hem_mode="up",
        tail=0.0, tail_mode="fwd",
        ripple=False,
        lean=0.0, knock=0, flash=False,
        wedge=0, ring=-1, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # No feet. The leaf hood, the hem, and the bottom point breathe.
        p["tip"] = wave(f, n, 7.0, phase=0.4)
        p["hem"] = wave(f, n, 5.0, phase=1.0)
        p["tail"] = wave(f, n, 8.0, phase=0.2)
        p["ripple"] = True
        p["bdy"] = -1 if wave(f, n, 1.0, phase=0.3) > 0.65 else 0
    elif action == "walk":
        # A glide. The leaf ripples. No legs are drawn on this still.
        p["bdx"] = int(round(wave(f, n, 1.0, cycles=2.0)))
        p["bdy"] = -1 if wave(f, n, 1.0, cycles=2.0, phase=0.4) > 0.55 else 0
        p["tip"] = wave(f, n, 9.0, phase=0.5)
        p["hem"] = wave(f, n, 8.0, phase=0.2)
        p["tail"] = wave(f, n, 11.0, phase=1.2)
        p["ripple"] = True
    elif action == "attack":
        # 0 rest. 1-4 raise the hood and roar (the mouth is two pixels).
        # 5-8 the other side, cleared again after finish().
        # 9-12 the hood and the hems strike. The bottom point counters.
        # 11-12 impact. 13-15 recover.
        tip = [0, 8, 14, 18, 22, 0, 0, 0, 0, 34, 34, 34, 34, 14, 6, 0]
        hem = [0, 6, 10, 14, 16, 0, 0, 0, 0, 22, 22, 22, 22, 10, 4, 0]
        tail = [0, -6, -10, -14, -16, 0, 0, 0, 0, -18, -18, -18, -18, -8, -3, 0]
        bdx = [0, 0, -1, -1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        bdy = [0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        p["tip"] = float(tip[f])
        p["hem"] = float(hem[f])
        p["tail"] = float(tail[f])
        p["bdx"] = int(bdx[f])
        p["bdy"] = int(bdy[f])
        p["tip_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["hem_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["tail_mode"] = "fwd"
        p["wedge"] = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["ring"] = [ -1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1][f]
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["tip"] = 12.0 * env
        p["tail"] = 10.0 * env
        p["tip_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    # 5-8 is the opposite still at equal scale. Drop the wind-up and the strike.
    if action == "attack" and 5 <= f <= 8:
        p["tip"] = 0.0
        p["hem"] = 0.0
        p["tail"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["wedge"] = 0
        p["ring"] = -1
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


def angle_of(rig, p, name):
    if name in ("tip_l", "tip_r"):
        amp = p["tip"]
        if name == "tip_r" and p["ripple"]:
            amp = -amp
        return part_angle(rig, name, p["tip_mode"], amp)
    if name in ("hem_l", "hem_r"):
        amp = p["hem"]
        if name == "hem_r" and p["ripple"]:
            amp = -amp
        return part_angle(rig, name, p["hem_mode"], amp)
    if name == "tail":
        return part_angle(rig, name, p["tail_mode"], p["tail"])
    return 0.0


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    for name in ("tail", "hem_l", "hem_r", "tip_l", "tip_r"):
        ang = angle_of(rig, p, name)
        paste(out, moved(L[name], ang, piv[name], bdx, bdy))
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = any_color(out, (INK, BLACK, DARK, OLIVE))
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def limb_names(rig):
    return [n for n in PARTS if n in rig["layers"]]


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    for name in limb_names(rig):
        ang = angle_of(rig, p, name)
        layer = moved(rig["layers"][name], ang, rig["pivots"][name], bdx, bdy)
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


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["tip"]) >= 0.45 or abs(p["hem"]) >= 0.45 or abs(p["tail"]) >= 0.45:
        return True
    return False


def mouth_wedge(img, origin, fc, age):
    """A few pixels in front of the leaf edge at the mouth row. Empty pixels only."""
    ox, oy = origin
    cols = (INK, GOLD, YELLOW)
    for i in range(max(0, int(age))):
        x = ox + fc * (2 + i)
        span = 1 if i == 0 else 2
        for dy in range(-span, span + 1):
            put(img, x, oy + dy, cols[min(i, 2)])


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, YELLOW, LIME)
    rad = 8 + int(age) * 3
    for deg in range(-60, 61, 10):
        a = math.radians(deg)
        x = ox + fc * rad * math.cos(a)
        y = oy + rad * math.sin(a) * 0.45
        put(img, x, y, cols[int(age) % 3])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, YELLOW, LIME)
    r = 3 + int(age) * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def forward_tip(rig, p):
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    best = None
    for name in ("tip_l", "tip_r"):
        layer = moved(
            rig["layers"][name], angle_of(rig, p, name), rig["pivots"][name], bdx, bdy
        )
        m = layer[..., 3] >= 128
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
        score = float(xs[i]) if fc > 0 else -float(xs[i])
        if best is None or score > best[0]:
            best = (score, float(xs[i]), float(ys[i]))
    if best is None:
        return rig["pivots"]["center"]
    return best[1], best[2]


def facing_edge(img, y, fc):
    y = int(round(y))
    h = img.shape[0]
    if y < 0 or y >= h:
        return None
    xs = np.nonzero(img[y, :, 3] >= 128)[0]
    if len(xs) == 0:
        return None
    return int(xs.max()) if fc > 0 else int(xs.min())


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    body = compose(rig, p)
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], piv)
    if rotated(p):
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if rig["has_mouth"] and not p["turned"] and rig["pivots"]["mouth"] is not None:
        mx, my = rig["pivots"]["mouth"]
        my = my + int(round(p["bdy"]))
        edge = facing_edge(img, my, fc)
        if p["wedge"] and edge is not None:
            mouth_wedge(img, (edge, my), fc, p["wedge"])
        if p["ring"] >= 0:
            roar_rings(img, (mx + fc * p["bdx"], my), fc, p["ring"])
        if p["wedge"] or p["ring"] >= 0:
            img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = forward_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


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


def crop_box(canvas, pad=8):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad - 10)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 18)
    x0 = max(0, int(xs.min()) - pad - 16)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 16)
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


def frame_metrics(img, palette):
    """Return off-palette count, partial-alpha count, and whether the frame touches the cell edge."""
    pal = {tuple(int(v) for v in c) for c in palette}
    op = img[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(img[op][:, :3], axis=0)} if op.any() else set()
    off = len(used - pal)
    partial = int(((img[..., 3] != 0) & (img[..., 3] != 255)).sum())
    a = img[..., 3] == 255
    edge = bool(a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any())
    return off, partial, edge


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
    fm = segment(front, "front")
    bm = segment(back, "back")
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs["W"]["facing"] != -1.0 or rigs["E"]["facing"] != 1.0:
        raise SystemExit("front facing is not left-as-drawn")
    if rigs["NE"]["facing"] != 1.0 or rigs["N"]["facing"] != -1.0:
        raise SystemExit("back facing is not away-right as drawn")
    if not rigs["W"]["has_mouth"] or rigs["N"]["has_mouth"]:
        raise SystemExit("mouth is not front-only")
    # Attack 5-8 for a screen-right row is the N row mirrored, same scale.
    flipped_n = np.ascontiguousarray(rigs["N"]["canvas"][:, ::-1])
    if not np.array_equal(flipped_n, rigs[("back", False)]["canvas"]):
        raise SystemExit("back turn is not the N row at equal scale")
    return front, back, fm, bm, rigs


def save_parts(work, fm, bm):
    cols = {
        "tip_l": (255, 80, 80, 255),
        "tip_r": (80, 180, 255, 255),
        "hem_l": (255, 200, 40, 255),
        "hem_r": (180, 80, 255, 255),
        "tail": (255, 140, 40, 255),
        "torso": (170, 190, 120, 255),
    }
    tiles = []
    for masks in (fm, bm):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 3, 0), 3, 1))
    gap = np.zeros((tiles[0].shape[0], 8, 4), np.uint8)
    Image.fromarray(np.concatenate([tiles[0], gap, tiles[1]], axis=1), "RGBA").save(work / "parts.png")


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
        "dex": 541, "form": "normal", "variant": "swadloon", "gen": "gen5",
        "dir": "monsterSprite/gen5/0541_swadloon/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def measure_all(rigs, palette):
    off = 0
    partial = 0
    clip = []
    bodies = []
    for d in DIRS:
        for act in ACTIONS:
            for f in range(FRAMES[act]):
                img = synth(rigs, d, act, f, palette, effects=True)
                o, p, edge = frame_metrics(img, palette)
                off += o
                partial += p
                if edge:
                    clip.append(f"{act}:{d}:{f}")
                if act in ("attack", "hurt", "walk"):
                    bodies.append(largest_fraction(img))
    return off, partial, clip, bodies


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
    save_parts(work, fm, bm)
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, flush=True)
    print("parts", {k: int(v.sum()) for k, v in fm.items()}, {k: int(v.sum()) for k, v in bm.items()}, flush=True)
    print("mouth", {d: rigs[d]["has_mouth"] for d in DIRS}, flush=True)

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
    still = int(sum(rest_diff.values()))
    print("rest diff", rest_diff, flush=True)
    print("STILL_DIFF", still, flush=True)
    if still:
        raise SystemExit("frame 0 does not match the still")

    turn = 0
    for d in DIRS:
        for f in range(5, 9):
            img = synth(rigs, d, "attack", f, palette, effects=False)
            rig = rig_for(rigs, d, pose("attack", f))
            turn += int(np.any(img != rig["canvas"], axis=2).sum())
    print("TURN_DIFF", turn, flush=True)
    if turn:
        raise SystemExit("attack 5-8 is not the opposite still")

    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        y0, y1, x0, x1 = box
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
        lost = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["tip"], 1), round(p["hem"], 1), round(p["tail"], 1),
                        p["tip_mode"], p["turned"], p["spin"], p["bdx"], p["wedge"], p["spark"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        off, partial, clip, bodies = measure_all(rigs, palette)
        print("OFF_PALETTE", off, flush=True)
        print("PARTIAL_ALPHA", partial, flush=True)
        print("EDGE_TOUCH", len(clip), clip[:12], flush=True)
        print("body_main_min", round(min(bodies), 3) if bodies else 1, flush=True)
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
            if key not in state["done"]:
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

    pal = {tuple(int(v) for v in c) for c in palette}
    op = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[op][:, :3], axis=0)}
    off_cols = sorted(used - pal)
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
    still = int(sum(rest_diff.values()) + sum(idle0.values()))
    report = {
        "version": VERSION, "hash": digest, "rest_diff": rest_diff, "idle0": idle0,
        "off_palette_colors": len(off_cols), "partial_alpha": partial, "edge_touching": clip[:12],
        "edge_n": len(clip), "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "bounds": bounds, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (still == 0 and not off_cols and partial == 0 and not clip)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("STILL_DIFF", still, flush=True)
    print("OFF_PALETTE", len(off_cols), flush=True)
    print("PARTIAL_ALPHA", partial, flush=True)
    print("EDGE_TOUCH", len(clip), clip[:12], flush=True)
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_swadloon_actions", "version": VERSION, "key": KEY,
        "dex": 541, "variant": "swadloon", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off_cols),
        "archetype": "leaf cloak (two hood prongs, two hem corners, bottom leaf point, two-pixel mouth, no feet)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (yellow xmin 35 < eye mean 43.4); back apex is screen-left so the back faces away-right",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
