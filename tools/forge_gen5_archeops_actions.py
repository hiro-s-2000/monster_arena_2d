#!/usr/bin/env python3
"""Archeops normal (0567) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x13-82 y6-89, 2508 px, 15 colours,
alpha 0/255, sole y89):
  The white eye is the cluster x23-27 y32-35, mean x 25.4. The face (teal
  and red) reaches x17, left of that eye, so the still faces screen-left.
  Six purple pixels (131, 65, 82) sit under the eye at x25-27 y38-40, ink
  along their upper-left edge, and that colour is absent on the back. That
  is the mouth. The roar uses a short wedge of that purple and a ring.
  Two blue wings, separate from the body: near wing x13-32 y6-29, far wing
  x51-82 y7-43. A lower far feather sits at x59-74 y52-64. The blue tail is
  the hanging fan x37-54 y75-89 plus the shaft just above it. One red foot
  is drawn, x46-61 y64-80. No second foot is invented.
normal/back.png (bbox x14-87 y2-92, 2670 px, 14 colours, sole y92) has the
  teal face mark at x47-58 y15-20, right of the tan body, so this back faces
  screen-right. No purple, so the back roar does not grow a mouth. Wings
  x14-37 y2-34 and x49-87 y2-42. Tail fan x33-50 y77-92 plus the shaft
  x37-50 y54-70. One red leg x27-41 y49-83. The back sole is 3 px lower, so
  the back is shifted up 3 px and both soles share one ground line.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (away, head toward screen-right). NW back mirrored.
  Attack 5-8 is that direction's opposite still at equal scale. W becomes N,
  S becomes NW. No |cos| squash and no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_archeops_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0567_archeops/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0567_archeops/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0567_archeops"
VERSION = "archeops_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 256, 256
OFF_X = (CELL_W - 96) // 2
FRONT_FOOT = 89
BACK_FOOT = 92
# 22 px under the shared sole. The back sole is 3 px lower in its own canvas.
OFF_Y = CELL_H - 22 - FRONT_FOOT
FOOT_Y = OFF_Y + FRONT_FOOT
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Angles this script draws. finish() may raise a weaker attack frame up to
# these, and the hit itself is already at the peak so it is not stretched.
PEAKS = {
    "wing_l": 34.0, "wing_r": 34.0, "tail": 28.0, "plume": 18.0, "foot": 20.0,
}
LIMB_NAMES = ("wing_l", "wing_r", "tail", "plume", "foot")

INK = (16, 16, 16)
BLUE = (49, 131, 246)
NAVY = (16, 49, 106)
DBLUE = (49, 82, 156)
GOLD = (246, 205, 98)
TAN = (189, 148, 82)
BROWN = (98, 82, 41)
RED = (222, 82, 74)
DRED = (156, 74, 74)
MAROON = (82, 49, 49)
TEAL = (16, 164, 148)
DTEAL = (8, 106, 90)
WHITE = (255, 255, 255)
GREY = (156, 156, 172)
PURPLE = (131, 65, 82)
BLUE_COLS = (BLUE, NAVY, DBLUE)
RED_COLS = (RED, DRED, MAROON)
TEAL_COLS = (TEAL, DTEAL)
GOLD_COLS = (GOLD, TAN, BROWN)
FACE_WHITE = (WHITE, GREY)

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


def stats(pts):
    ys = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    return {
        "pts": pts, "n": len(pts),
        "x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
        "mx": sum(xs) / len(xs), "my": sum(ys) / len(ys),
    }


def largest_component(mask):
    groups = component_list(mask)
    if not groups:
        return 0
    return max(len(g) for g in groups)


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
        raise SystemExit(f"parts do not partition the still: {int(acc.sum())} vs {int(op.sum())}")


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
    d = ((px[:, None, :] - palette.astype(np.int32)[None, :, :]) ** 2).sum(axis=2)
    out[op, :3] = palette[d.argmin(axis=1)]
    out[op, 3] = 255
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def put(img, x, y, col, over=False):
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 1 <= x < w - 1 and 1 <= y < h - 1 and (over or img[y, x, 3] < 128):
        img[y, x, :3] = col
        img[y, x, 3] = 255


def paint_cushion(layer, allowed, steps, dirs=("up", "down", "left", "right")):
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        for d in dirs:
            mv = np.zeros_like(src)
            if d == "up":
                mv[:-1] = src[1:]
            elif d == "down":
                mv[1:] = src[:-1]
            elif d == "left":
                mv[:, :-1] = src[:, 1:]
            else:
                mv[:, 1:] = src[:, :-1]
            take = (acc[..., 3] < 128) & allowed & (mv[..., 3] >= 128)
            acc[take] = mv[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def feather_name(st, kind):
    """Which moving feather a blue+ink blob is, from this form's own layout."""
    mx, my = st["mx"], st["my"]
    if kind == "front":
        if my < 36 and mx < 36:
            return "wing_l"
        if my < 46 and mx > 55:
            return "wing_r"
        if my > 72:
            return "tail"
        if 58 <= my <= 72 and 34 <= mx <= 56:
            return "tail"
        if my > 45 and mx > 56:
            return "plume"
    else:
        if my < 42 and mx < 40:
            return "wing_l"
        if my < 46 and mx > 55:
            return "wing_r"
        if my > 72:
            return "tail"
        if 50 <= my <= 72 and 32 <= mx <= 56:
            return "tail"
        if my > 45 and mx < 32:
            return "plume"
    return None


def exclusive_outline(core, op, ink):
    """Core plus outline ink that does not also touch another body colour."""
    other = op & ~core & ~ink
    near = ink & dilate(core, 1)
    shared = near & dilate(other, 1)
    return core | (near & ~shared)


def segment(img, kind):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    sole = int(np.nonzero(op)[0].max())
    expect = FRONT_FOOT if kind == "front" else BACK_FOOT
    if sole != expect:
        raise SystemExit(f"{kind} sole moved: {sole}")
    blue = any_color(img, BLUE_COLS)
    red = any_color(img, RED_COLS)
    ink = is_color(img, INK)
    teal = any_color(img, TEAL_COLS)
    purple = is_color(img, PURPLE)
    blobs = []
    for pts in component_list(blue | ink):
        st = stats(pts)
        if st["n"] < 30:
            continue
        name = feather_name(st, kind)
        if name is None:
            raise SystemExit(f"{kind} unexplained feather {st['n']} x{st['x0']}-{st['x1']} y{st['y0']}-{st['y1']}")
        blobs.append((name, st))
    masks = {name: np.zeros(op.shape, bool) for name in LIMB_NAMES}
    for name, st in blobs:
        masks[name] |= mask_from_pts(op.shape, st["pts"])
    if kind == "front":
        foot_groups = [stats(g) for g in component_list(red) if stats(g)["y0"] >= 62 and stats(g)["n"] >= 40]
    else:
        foot_groups = [stats(g) for g in component_list(red) if stats(g)["my"] > 48 and stats(g)["n"] >= 80]
    if len(foot_groups) != 1:
        raise SystemExit(f"{kind} expected one foot, got {len(foot_groups)}")
    foot_core = mask_from_pts(op.shape, foot_groups[0]["pts"])
    masks["foot"] = exclusive_outline(foot_core, op, ink)
    # Feathers already include their own ink. Do not let the foot claim it.
    masks["foot"] &= ~masks["wing_l"] & ~masks["wing_r"] & ~masks["tail"] & ~masks["plume"]
    for name in ("wing_l", "wing_r", "tail", "foot"):
        if int(masks[name].sum()) < 40:
            raise SystemExit(f"{kind} {name} is too small ({int(masks[name].sum())})")
    if not masks["plume"].any():
        del_plume = True
    else:
        del_plume = False
    mouth = purple & (yy >= 36) & (yy <= 42) & (xx >= 22) & (xx <= 30)
    if kind == "front":
        if int(mouth.sum()) < 4:
            raise SystemExit("front mouth mark is missing")
        # The glint is two diagonal clusters (5 px and 3 px), not one 4-connected blob.
        eye = is_color(img, WHITE) & (yy >= 30) & (yy <= 36) & (xx >= 22) & (xx <= 30)
        if int(eye.sum()) < 6:
            raise SystemExit(f"front eye is missing ({int(eye.sum())})")
        eye_x = float(np.nonzero(eye)[1].mean())
        face = (teal | red | purple) & (yy < 46)
        tip = int(np.nonzero(face)[1].min())
        if tip >= eye_x - 1:
            raise SystemExit(f"front does not face left: tip {tip} eye {eye_x:.1f}")
    else:
        if mouth.any():
            raise SystemExit("back grew a mouth colour")
        face = (teal & (yy < 32)) | (red & (yy >= 16) & (yy < 36))
        if not face.any():
            raise SystemExit("back face mark is missing")
        fy, fx = np.nonzero(face)
        if float(fx.mean()) <= 48:
            raise SystemExit(f"back face is not on the right: {float(fx.mean()):.1f}")
    used = np.zeros(op.shape, bool)
    for name in list(masks):
        if name == "plume" and del_plume:
            continue
        used |= masks[name]
    torso = op & ~used
    masks["torso"] = torso
    if del_plume:
        masks["plume"] = np.zeros(op.shape, bool)
    check_partition(op, masks)
    body = op & ~masks["wing_l"] & ~masks["wing_r"] & ~masks["tail"] & ~masks["plume"] & ~masks["foot"]
    for name in ("wing_l", "wing_r", "tail", "foot"):
        if not (masks[name] & dilate(body, 2)).any():
            raise SystemExit(f"{kind} {name} does not touch the body")
    if masks["plume"].any() and not (masks["plume"] & dilate(body, 2)).any():
        raise SystemExit(f"{kind} plume does not touch the body")
    info = {
        "mouth": mouth,
        "face": ((teal | red | purple) & (yy < 48) & ~masks["foot"]) if kind == "front"
        else ((teal & (yy < 36)) | (red & (yy < 40) & (yy > 14))),
        "has_mouth": bool(kind == "front"),
    }
    return masks, info


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


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs - pivot[0]) ** 2 + (ys - pivot[1]) ** 2
    i = int(d.argmax())
    return float(xs[i]), float(ys[i])


def signs_for(layer, pivot, facing):
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


def mean_xy(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def build_view(img, masks, info, kind, mirror):
    y_shift = 0 if kind == "front" else (FRONT_FOOT - BACK_FOOT)
    canvas = place(img, y_shift)
    placed = {k: place(v, y_shift) for k, v in masks.items()}
    mouth = place(info["mouth"], y_shift)
    face = place(info["face"], y_shift)
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
        mouth = np.ascontiguousarray(mouth[:, ::-1])
        face = np.ascontiguousarray(face[:, ::-1])
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    torso_m = placed["torso"]
    limbs = [name for name in LIMB_NAMES if int(placed[name].sum()) >= 4]
    for name in limbs:
        depth = 5 if name in ("foot", "tail") else 4
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth + 1,
        )
    pivots = {}
    signs = {}
    for name in limbs:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    mouth_xy = mean_xy(mouth)
    if mouth_xy is None:
        mouth_xy = pivots["center"]
    pivots["mouth"] = mouth_xy
    fys, fxs = np.nonzero(face)
    if len(fxs):
        score = fxs.astype(np.float64) * facing
        i = int(score.argmax())
        pivots["beak"] = (float(fxs[i]), float(fys[i]))
    else:
        pivots["beak"] = pivots["mouth"]
    centers = {}
    for name in ("wing_l", "wing_r"):
        if name in limbs:
            my, mx = np.nonzero(placed[name])
            centers[name] = float(mx.mean())
    if len(centers) == 2:
        ordered = sorted(centers, key=centers.get)
        near = ordered[1] if facing > 0 else ordered[0]
        far = ordered[0] if facing > 0 else ordered[1]
    else:
        near, far = "wing_l", "wing_r"
    return {
        "kind": kind, "mirror": mirror, "facing": facing,
        "has_mouth": bool(info["has_mouth"]),
        "layers": layers, "pivots": pivots, "signs": signs,
        "canvas": canvas, "limbs": limbs, "near": near, "far": far,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        wing_l=0.0, wing_r=0.0, tail=0.0, plume=0.0, foot=0.0,
        wing_mode="up", tail_mode="up", plume_mode="up", foot_mode="fwd",
        step=(0, 0),
        lean=0.0, knock=0, flash=False,
        wedge=0, ring=-1, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        p["wing_l"] = wave(f, n, 5.0, phase=0.4)
        p["wing_r"] = wave(f, n, 4.0, phase=0.4 + 0.6)
        p["plume"] = wave(f, n, 3.0, phase=0.9)
        p["tail"] = wave(f, n, 6.0, phase=1.1)
        p["wing_mode"] = "up"
        p["tail_mode"] = "up"
        p["plume_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        p["step"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["bdy"] = -1 if sa > 0.55 else 0
        p["wing_l"] = wave(f, n, 10.0, phase=0.2)
        p["wing_r"] = wave(f, n, 10.0, phase=0.2 + math.pi)
        p["plume"] = wave(f, n, 6.0, phase=0.5)
        p["tail"] = wave(f, n, 8.0, phase=1.2)
        p["wing_mode"] = "up"
        p["tail_mode"] = "fwd"
        p["plume_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 roar: wings and tail rise, mouth wedge and ring.
        # 5-8 the opposite still, cleared again after finish().
        # 9-12 wing strike, tail whip, the drawn foot kicks. 13-15 return.
        wing = [0, 8, 14, 18, 18, 0, 0, 0, 0, 20, 28, 34, 34, 14, 4, 0]
        tail = [0, 4, 8, 12, 14, 0, 0, 0, 0, 12, 20, 28, 28, 10, 3, 0]
        plume = [0, 4, 8, 10, 10, 0, 0, 0, 0, 8, 14, 18, 18, 6, 2, 0]
        foot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 14, 20, 20, 8, 2, 0]
        bdx = [0, -1, -1, -1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        bdy = [0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        p["wing_l"] = float(wing[f])
        p["wing_r"] = float(wing[f])
        p["tail"] = float(tail[f])
        p["plume"] = float(plume[f])
        p["foot"] = float(foot[f])
        p["bdx"] = int(bdx[f])
        p["bdy"] = int(bdy[f])
        p["wing_mode"] = "up" if f <= 4 else "fwd"
        p["tail_mode"] = "up" if f <= 4 else "fwd"
        p["plume_mode"] = "up" if f <= 4 else "fwd"
        p["foot_mode"] = "fwd"
        if f in (10, 11, 12):
            p["step"] = {10: (1, -1), 11: (2, 0), 12: (2, 0)}[f]
        if 1 <= f <= 4:
            p["wedge"] = f
            p["ring"] = f - 1
        p["turned"] = 5 <= f <= 8
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["wing_l"] = 10.0 * env
        p["wing_r"] = 8.0 * env
        p["tail"] = 12.0 * env
        p["plume"] = 6.0 * env
        p["wing_mode"] = "up"
        p["tail_mode"] = "up"
        p["plume_mode"] = "up"
        p["flash"] = f == 1
        p["spark"] = 1 if f in (1, 2) else 0
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        for key in ("wing_l", "wing_r", "tail", "plume", "foot"):
            p[key] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step"] = (0, 0)
        p["wedge"] = 0
        p["ring"] = -1
        p["spark"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["flash"] = False
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


def mode_of(p, name):
    if name in ("wing_l", "wing_r"):
        return p["wing_mode"]
    if name == "tail":
        return p["tail_mode"]
    if name == "plume":
        return p["plume_mode"]
    return p["foot_mode"]


def limb_shift(rig, p, name):
    fc = rig["facing"]
    dx = int(round(fc * p["bdx"]))
    dy = int(p["bdy"])
    if name == "foot":
        sx, sy = p["step"]
        dx += int(round(fc * sx))
        dy += int(sy)
    return dx, dy


def compose(rig, p):
    L = rig["layers"]
    out = np.zeros_like(rig["canvas"])
    dx, dy = limb_shift(rig, p, "torso")
    paste(out, shift(L["torso"], dx, dy))
    order = []
    for name in ("tail", "plume", rig["far"], "foot", rig["near"]):
        if name in rig["limbs"] and name not in order:
            order.append(name)
    for name in rig["limbs"]:
        if name not in order:
            order.append(name)
    for name in order:
        ang = part_angle(rig, name, mode_of(p, name), p[name])
        sx, sy = limb_shift(rig, p, name)
        paste(out, moved(L[name], ang, rig["pivots"][name], sx, sy))
    return out


def part_layer(rig, name, p):
    ang = part_angle(rig, name, mode_of(p, name), p[name])
    sx, sy = limb_shift(rig, p, name)
    return moved(rig["layers"][name], ang, rig["pivots"][name], sx, sy)


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name in rig["limbs"]:
        layer = part_layer(rig, name, p)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    kind, mirror = VIEW_OF[d]
    if p["turned"]:
        kind = "back" if kind == "front" else "front"
    return rigs[(kind, mirror)]


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK)
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def mouth_wedge(img, origin, fc, age):
    """A short wedge of the mouth colour that is already in the still."""
    ox, oy = origin
    for i in range(max(1, age)):
        put(img, ox + fc * (2 + i), oy, PURPLE)
        put(img, ox + fc * (2 + i), oy - 1, RED)
        put(img, ox + fc * (1 + i), oy + 1, INK)


def roar_ring(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, BLUE, GOLD)
    rad = 7 + age * 4
    for deg in range(-48, 49, 8):
        a = math.radians(deg)
        x = ox + fc * (3 + rad * math.cos(a))
        y = oy + rad * math.sin(a) * 0.55
        put(img, x, y, cols[(deg // 8 + age) % 3])


def spark(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, GOLD, BLUE)
    r = 4 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    moved_part = (
        abs(p["wing_l"]) >= 0.45 or abs(p["wing_r"]) >= 0.45
        or abs(p["tail"]) >= 0.45 or abs(p["plume"]) >= 0.45
        or abs(p["foot"]) >= 0.45
        or p["step"] != (0, 0) or p["bdx"] or p["bdy"]
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
    if not effects or p["turned"]:
        return img
    if p["flash"]:
        img = flash_white(img)
    if rig["has_mouth"] and p["wedge"]:
        mouth_wedge(img, rig["pivots"]["mouth"], fc, p["wedge"])
    if rig["has_mouth"] and p["ring"] >= 0:
        roar_ring(img, rig["pivots"]["beak"], fc, p["ring"])
    if p["spark"] and action == "hurt":
        cx, cy = rig["pivots"]["center"]
        spark(img, (cx - fc * 8, cy - 16), -fc, p["spark"])
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


def content_box(canvases, pad=12):
    acc = np.zeros(canvases[0].shape[:2], bool)
    for c in canvases:
        acc |= c[..., 3] >= 128
    ys, xs = np.nonzero(acc)
    y0 = max(0, int(ys.min()) - pad - 28)
    y1 = min(CELL_H, int(ys.max()) + pad + 16)
    x0 = max(0, int(xs.min()) - pad - 36)
    x1 = min(CELL_W, int(xs.max()) + pad + 36)
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
    fm, fi = segment(front, "front")
    bm, bi = segment(back, "back")
    rigs = {
        ("front", False): build_view(front, fm, fi, "front", False),
        ("front", True): build_view(front, fm, fi, "front", True),
        ("back", False): build_view(back, bm, bi, "back", False),
        ("back", True): build_view(back, bm, bi, "back", True),
    }
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "torso": (70, 130, 210, 255),
        "wing_l": (240, 180, 40, 255),
        "wing_r": (40, 180, 220, 255),
        "tail": (220, 60, 60, 255),
        "plume": (160, 80, 220, 255),
        "foot": (240, 240, 240, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            if name in cols and m.any():
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
        "dex": 567, "form": "normal", "variant": "archeops", "gen": "gen5",
        "dir": "monsterSprite/gen5/0567_archeops/normal", "sheetDir": "agent-sprite-forge",
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
    ap.add_argument("--no-publish", action="store_true")
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
    print("near", {d: rigs[d]["near"] for d in DIRS},
          "mouth", {d: rigs[d]["has_mouth"] for d in ("S", "W", "N")}, flush=True)

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
            atk.append((f, round(float(p["wing_l"]), 1), round(float(p["tail"]), 1),
                        round(float(p["foot"]), 1), p["turned"], p["spin"], p["wedge"]))
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
        and partial == 0 and not clip
    )
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print("qa_full", "rest_diff", int(sum(rest_diff.values())),
          "off_palette", len(off), "partial_alpha", partial,
          "edge_touch", len(clip), "idle0", int(sum(idle0.values())),
          "body_main_min", report["body_main_min"], flush=True)
    meta = {
        "generator": "forge_gen5_archeops_actions", "version": VERSION, "key": KEY,
        "dex": 567, "variant": "archeops", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "avian raptor (two blue wings, hanging tail, one red foot, small mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front faces screen-left (face tip x17 < eye mean x25.4); back teal is right of center",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    if args.no_publish:
        print("qa ok; publish skipped", flush=True)
        return
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
