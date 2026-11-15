#!/usr/bin/env python3
"""Drilbur normal (0529) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x20-72 y23-72, 1485 px, alpha 0/255,
14 colours):
  Faces screen-left. The pink nose (20 px, mean x 34.95 y 36.1) is the face
  tip. The one drawn eye is the ink pupil at x49 y40 inside the white snout
  (7 white neighbours). Nose x is left of the eye, so the still looks left.
  Open mouth: dark red (131,41,65) and (172,82,82), 18 px, mean x 44.5 y 47.3.
  Two claws already in the still. Left blade x20-33 y23-46 (nose side).
  Right blade x58-72 y23-43. Pale snout between them stays on the head.
  Feet split under y68. Left x34-43 ends y71. Right x50-59 ends y72.
  No tail, wing, horn, or second eye is drawn.
normal/back.png (bbox x23-71 y27-69, 1333 px, lowest y69, 12 colours) is the
  back. The nose peek is 4 pink px at x54-55 y34-36, on the screen-right of
  the head. No mouth. Left claw x24-32 y27-36, right claw x61-71 y28-47.
  Feet split under y66. The back is shifted down 3 px so both soles share y72.

Directions, because the front faces screen-left and the back's nose is on
the right:
  S SE E front mirrored (forward is screen-right). W SW front as drawn.
  N NE back as drawn. NW back mirrored.
  E is screen-right, W is screen-left, N is the back.
  Attack 5-8 is that back at equal scale: W and SW show N, S SE E show N
  flipped. Back rows stay the back. No |cos| squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_drilbur_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0529_drilbur/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0529_drilbur/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0529_drilbur"
VERSION = "drilbur_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 192
OFF_X = (CELL_W - 96) // 2
FRONT_FOOT = 72
BACK_FOOT = 69
OFF_Y = CELL_H - 96 - max(0, FRONT_FOOT - BACK_FOOT)
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. A 180 yaw keeps the mirror flag:
# W (front, False) shows N (back, False); S (front, True) shows NW.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Wind-up peaks finish() may raise frames 5-12 toward. The claw strike is
# drawn at 34 deg, above these, so finish does not pass it.
PEAKS = {"lead": 18.0, "off": 14.0}
LIMBS = ("claw_l", "claw_r", "foot_l", "foot_r")

INK = (16, 16, 16)
DK = (49, 49, 49)
MID = (65, 65, 65)
SLATE = (74, 74, 90)
SOLE = (90, 90, 90)
GREY = (98, 98, 98)
BELLY = (98, 106, 131)
MOUTH = (131, 41, 65)
SILVER = (164, 164, 164)
MOUTH2 = (172, 82, 82)
NOSE_D = (205, 123, 106)
CLAW = (213, 213, 213)
NOSE = (238, 156, 139)
WHITE = (255, 255, 255)
LIGHT_COLS = (WHITE, CLAW, SILVER, GREY)
NOSE_COLS = (NOSE, NOSE_D)
MOUTH_COLS = (MOUTH, MOUTH2)
BODY_COLS = (DK, MID, SLATE, SOLE, BELLY, MOUTH, MOUTH2, NOSE, NOSE_D)

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


def largest_component(mask):
    groups = component_list(mask)
    if not groups:
        return 0
    return max(len(g) for g in groups)


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
    """4x nearest in-plane rotation. Positive is clockwise on screen. No squash."""
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
    """Grow the layer's own colours into allowed pixels (hidden at rest)."""
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        for d in ("up", "down", "left", "right"):
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
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def check_partition(op, masks):
    acc = np.zeros_like(op)
    for name, m in masks.items():
        if (m & ~op).any():
            raise SystemExit(f"{name} left the still")
        if (m & acc).any():
            raise SystemExit(f"overlap in {name}")
        acc |= m
    if int(acc.sum()) != int(op.sum()):
        raise SystemExit(f"parts do not partition the still: {int(acc.sum())} vs {int(op.sum())}")


def require_piece(name, mask, torso):
    if not mask.any():
        raise SystemExit(f"{name} is missing")
    if largest_component(mask) != int(mask.sum()):
        raise SystemExit(f"{name} fell apart ({largest_component(mask)} vs {int(mask.sum())})")
    if not (mask & dilate(torso, 1)).any():
        raise SystemExit(f"{name} does not touch the body")


def absorb_islands(torso, limbs):
    """A torso speck that only touches one limb rides with that limb."""
    groups = component_list(torso)
    if len(groups) <= 1:
        return torso
    groups.sort(key=len, reverse=True)
    keep = np.zeros_like(torso)
    for y, x in groups[0]:
        keep[y, x] = True
    h, w = torso.shape
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    for pts in groups[1:]:
        target = None
        for name, mask in limbs.items():
            hit = False
            for y, x in pts:
                for dy, dx in neigh:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                target = name
                break
        if target is None:
            for y, x in pts:
                keep[y, x] = True
        else:
            for y, x in pts:
                limbs[target][y, x] = True
    return keep


def light_groups(img):
    light = any_color(img, LIGHT_COLS)
    groups = [g for g in component_list(light) if len(g) >= 8]
    groups.sort(key=lambda g: float(np.mean([p[1] for p in g])))
    return groups


def claw_blade(img, pts, body):
    """Silver blade plus the ink that touches only that blade, not the body."""
    core = np.zeros(img.shape[:2], bool)
    for y, x in pts:
        core[y, x] = True
    ink = is_color(img, INK)
    rim = ink & dilate(core, 1) & ~dilate(body, 1)
    return core | rim


def split_feet(op, claws, y_cut, x_split):
    yy, xx = np.mgrid[0:96, 0:96]
    feet = op & (yy >= y_cut)
    for c in claws:
        feet &= ~c
    foot_l = np.zeros_like(op)
    foot_r = np.zeros_like(op)
    for g in component_list(feet):
        xs = np.array([p[1] for p in g])
        dest = foot_l if float(xs.mean()) < x_split else foot_r
        for y, x in g:
            dest[y, x] = True
    return foot_l, foot_r


def segment_front(img):
    op = img[..., 3] >= 128
    groups = light_groups(img)
    if len(groups) != 3:
        raise SystemExit(f"front light groups changed: {len(groups)}")
    body = any_color(img, BODY_COLS)
    claw_l = claw_blade(img, groups[0], body)
    claw_r = claw_blade(img, groups[2], body)
    foot_l, foot_r = split_feet(op, (claw_l, claw_r), 68, 48)
    limbs = {"claw_l": claw_l, "claw_r": claw_r, "foot_l": foot_l, "foot_r": foot_r}
    torso = op & ~claw_l & ~claw_r & ~foot_l & ~foot_r
    torso = absorb_islands(torso, limbs)
    masks = dict(limbs)
    masks["torso"] = torso
    check_partition(op, masks)
    for name in LIMBS:
        require_piece(name, masks[name], torso)
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("front torso fell apart")
    eye = is_color(img, INK) & (np.arange(96)[:, None] == 40) & (np.arange(96)[None, :] == 49)
    if int(eye.sum()) != 1 or not (eye & torso).any():
        raise SystemExit("front eye left the head")
    nose = any_color(img, NOSE_COLS)
    mouth = any_color(img, MOUTH_COLS)
    if int((nose & torso).sum()) != int(nose.sum()) or int(nose.sum()) != 20:
        raise SystemExit("front nose left the head")
    if int((mouth & torso).sum()) != int(mouth.sum()) or int(mouth.sum()) != 18:
        raise SystemExit("front mouth left the head")
    nx = float(np.nonzero(nose)[1].mean())
    if not (nx < 49 - 5):
        raise SystemExit(f"front facing changed: nose {nx:.1f} eye 49")
    if int(np.nonzero(op)[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    counts = {k: int(v.sum()) for k, v in masks.items()}
    expect = {"claw_l": 196, "claw_r": 207, "foot_l": 33, "foot_r": 37, "torso": 1012}
    if counts != expect:
        raise SystemExit(f"front parts changed {counts}")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    if any_color(img, MOUTH_COLS).any():
        raise SystemExit("back gained a mouth")
    groups = light_groups(img)
    if len(groups) != 3:
        raise SystemExit(f"back light groups changed: {len(groups)}")
    body = any_color(img, BODY_COLS)
    claw_l = claw_blade(img, groups[0], body)
    claw_r = claw_blade(img, groups[2], body)
    foot_l, foot_r = split_feet(op, (claw_l, claw_r), 66, 46)
    limbs = {"claw_l": claw_l, "claw_r": claw_r, "foot_l": foot_l, "foot_r": foot_r}
    torso = op & ~claw_l & ~claw_r & ~foot_l & ~foot_r
    torso = absorb_islands(torso, limbs)
    masks = dict(limbs)
    masks["torso"] = torso
    check_partition(op, masks)
    for name in LIMBS:
        require_piece(name, masks[name], torso)
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("back torso fell apart")
    nose = any_color(img, NOSE_COLS)
    if int((nose & torso).sum()) != int(nose.sum()) or int(nose.sum()) != 4:
        raise SystemExit("back nose peek left the head")
    nx = float(np.nonzero(nose)[1].mean())
    tx = float(np.nonzero(torso)[1].mean())
    if nx <= tx:
        raise SystemExit(f"back nose is no longer on the right: {nx:.1f} vs torso {tx:.1f}")
    if int(np.nonzero(op)[0].max()) != BACK_FOOT:
        raise SystemExit("back lowest pixel changed")
    counts = {k: int(v.sum()) for k, v in masks.items()}
    expect = {"claw_l": 55, "claw_r": 114, "foot_l": 21, "foot_r": 38, "torso": 1105}
    if counts != expect:
        raise SystemExit(f"back parts changed {counts}")
    return masks


def place(arr, y_shift):
    if arr.ndim == 2:
        out = np.zeros((CELL_H, CELL_W), bool)
    else:
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
    depth = {"claw_l": 12, "claw_r": 12, "foot_l": 8, "foot_r": 8}
    for name in LIMBS:
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth[name]), depth[name] + 2
        )
    # Front as drawn faces left. Back as drawn has the nose peek on the right.
    if kind == "front":
        facing = 1.0 if mirror else -1.0
        lead = "claw_l"
    else:
        facing = -1.0 if mirror else 1.0
        lead = "claw_r"
    pivots = {}
    signs = {}
    for name in ("claw_l", "claw_r"):
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    mouth = any_color(canvas, MOUTH_COLS)
    nose = any_color(canvas, NOSE_COLS)
    has_mouth = bool(mouth.any())
    if has_mouth:
        my, mx = np.nonzero(mouth)
        pivots["muzzle"] = (float(mx.mean()), float(my.mean()))
    elif nose.any():
        ny, nx = np.nonzero(nose)
        pivots["muzzle"] = (float(nx.mean()), float(ny.mean()))
    else:
        pivots["muzzle"] = pivots["center"]
    if nose.any():
        ny, nx = np.nonzero(nose)
        i = int(nx.argmax()) if facing > 0 else int(nx.argmin())
        pivots["tip"] = (float(nx[i]), float(ny[i]))
    else:
        pivots["tip"] = pivots["muzzle"]
    foot_cx = {}
    for n in ("foot_l", "foot_r"):
        fxs = np.nonzero(placed[n])[1]
        foot_cx[n] = float(fxs.mean()) if len(fxs) else pivots["center"][0]
    if facing > 0:
        near = max(("foot_l", "foot_r"), key=lambda n: foot_cx[n])
    else:
        near = min(("foot_l", "foot_r"), key=lambda n: foot_cx[n])
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "lead": lead,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
        "has_mouth": has_mouth, "near_foot": near,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, lead=0.0, off=0.0, claw_mode="up",
        thrust=0, step_l=(0, 0), step_r=(0, 0), kick=(0, 0),
        lean=0.0, knock=0, flash=False, wedge=0, ring=-1, spark=0,
        turned=False, spin=0, head_dx=0, head_dy=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The two claws breathe. The body bobs one pixel.
        p["lead"] = wave(f, n, 4.0, phase=0.4)
        p["off"] = wave(f, n, 3.5, phase=1.2)
        p["bdy"] = -1 if wave(f, n, 1.0, phase=0.2) > 0.55 else 0
        p["claw_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        p["step_l"] = (int(round(2 * max(0.0, sa))), -int(round(2 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * max(0.0, -sa))), -int(round(2 * max(0.0, -sa))))
        p["lead"] = wave(f, n, 6.0, phase=0.3)
        p["off"] = wave(f, n, 6.0, phase=0.3 + math.pi)
        p["bdy"] = -1 if abs(math.sin(u)) > 0.55 else 0
        p["claw_mode"] = "fwd"
    elif action == "attack":
        # 0 rest. 1-4 roar from the mouth, claws rise.
        # 5-8 the back, at rest, equal scale. 9-12 the claws strike. 13-15 recover.
        lead = [0, 8, 14, 18, 18, 0, 0, 0, 0, 22, 30, 34, 34, 16, 6, 0]
        off = [0, 6, 10, 14, 14, 0, 0, 0, 0, 12, 16, 20, 18, 10, 4, 0]
        thrust = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0]
        wedge = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ring = [-1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        p["lead"] = float(lead[f])
        p["off"] = float(off[f])
        p["thrust"] = int(thrust[f])
        p["wedge"] = int(wedge[f])
        p["ring"] = int(ring[f])
        p["claw_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["kick"] = (1, -1)
            p["spark"] = f - 10
        p["bdx"] = 1 if f in (11, 12) else 0
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["lead"] = 8.0 * env
        p["off"] = 6.0 * env
        p["claw_mode"] = "up"
        p["flash"] = f == 1
        p["spark"] = 1 if f in (2, 3) else 0
    p = finish(p, action, f, __file__)
    # Frames 5-8 are the back still at equal scale. finish() would swing the
    # wind-up on these frames; drop it so the turned silhouette stays whole.
    if action == "attack" and 5 <= f <= 8:
        p["lead"] = 0.0
        p["off"] = 0.0
        p["thrust"] = 0
        p["bdx"] = 0
        p["bdy"] = 0
        p["kick"] = (0, 0)
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["wedge"] = 0
        p["ring"] = -1
        p["spark"] = 0
        p["spin"] = 0
        p["head_dx"] = 0
        p["head_dy"] = 0
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


def limb_layers(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    thrust = int(p["thrust"])
    lead = rig["lead"]
    off = "claw_r" if lead == "claw_l" else "claw_l"
    ang = {
        lead: part_angle(rig, lead, p["claw_mode"], p["lead"]),
        off: part_angle(rig, off, p["claw_mode"], p["off"]),
    }
    parts = {}
    for name in LIMBS:
        if name == "foot_l":
            sx, sy = p["step_l"]
            a = 0.0
            extra = 0
        elif name == "foot_r":
            sx, sy = p["step_r"]
            a = 0.0
            extra = 0
        else:
            sx, sy = 0, 0
            a = ang[name]
            extra = thrust
        if name == rig["near_foot"]:
            kx, ky = p["kick"]
            sx += kx
            sy += ky
        dx = bdx + int(round(fc * (sx + extra)))
        dy = bdy + int(sy)
        pivot = piv.get(name, piv["center"])
        parts[name] = moved(L[name], a, pivot, dx, dy)
    return parts


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    parts = limb_layers(rig, p)
    for name in ("foot_l", "foot_r", "claw_l", "claw_r"):
        paste(out, parts[name])
    return out


def pose_moved(p):
    return (
        abs(p["lead"]) >= 0.45 or abs(p["off"]) >= 0.45
        or p["step_l"] != (0, 0) or p["step_r"] != (0, 0)
        or p["kick"] != (0, 0) or p["thrust"] or p["bdx"] or p["bdy"]
        or p["head_dx"] or p["head_dy"]
    )


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


def mouth_wedge(img, origin, fc, age):
    ox, oy = origin
    cols = (MOUTH, MOUTH2, NOSE)
    for i in range(1, age + 1):
        put(img, ox + fc * (1 + i), oy, cols[i % 2])
        put(img, ox + fc * i, oy - 1, MOUTH)
        put(img, ox + fc * i, oy + 1, MOUTH2)


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, CLAW, NOSE)
    for ri in range(age + 1):
        rad = 6 + ri * 4
        for deg in range(-36, 37, 12):
            a = math.radians(deg)
            x = ox + fc * rad * math.cos(a)
            y = oy + rad * math.sin(a) * 0.5
            put(img, x, y, cols[ri % 3])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, SILVER, CLAW)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2), (2, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def weapon_tip(rig, p):
    parts = limb_layers(rig, p)
    layer = parts[rig["lead"]]
    fc = rig["facing"]
    if not np.any(layer[..., 3] >= 128):
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


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
    """Attack 5-8 shows this side's back at equal scale. Back rows stay."""
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    moved_part = pose_moved(p)
    fc = rig["facing"]
    if not moved_part and abs(p["lean"]) <= 0.4 and not p["knock"]:
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            piv = rear_pivot(body, fc)
            body = rotate_nearest(body, -fc * p["lean"], piv)
        body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    if rig["has_mouth"] and not p["turned"]:
        tx, ty = rig["pivots"]["tip"]
        tx += fc * p["bdx"]
        ty += p["bdy"]
        if p["wedge"]:
            mouth_wedge(img, (tx, ty), fc, p["wedge"])
        if p["ring"] >= 0:
            roar_rings(img, (tx + fc * 3, ty), fc, p["ring"])
    if p["spark"] and action == "attack" and not p["turned"]:
        wx, wy = weapon_tip(rig, p)
        impact(img, (wx + fc * 2, wy), fc, p["spark"])
    elif p["spark"] and action == "hurt":
        cx, cy = rig["pivots"]["center"]
        impact(img, (cx - fc * 6, cy - 8), -fc, p["spark"])
    if p["flash"] or (rig["has_mouth"] and (p["wedge"] or p["ring"] >= 0) and not p["turned"]) or p["spark"]:
        return snap(img, palette)
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


def crop_box(canvas, pad_x=28, pad_y=24):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad_y)
    y1 = min(CELL_H, int(ys.max()) + 1 + 16)
    x0 = max(0, int(xs.min()) - pad_x)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad_x)
    return y0, y1, x0, x1


def preview_image(sheets):
    pick = {"idle": 0, "walk": 4, "attack": 11, "hurt": 3}
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
        "claw_l": (220, 220, 230, 255),
        "claw_r": (180, 190, 210, 255),
        "foot_l": (250, 210, 60, 255),
        "foot_r": (80, 140, 230, 255),
        "torso": (90, 90, 110, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        pair = np.concatenate([img, part], axis=1)
        tiles.append(np.repeat(np.repeat(pair, 3, 0), 3, 1))
    Image.fromarray(np.concatenate(tiles, axis=0), "RGBA").save(work / "parts.png")


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
        "dex": 529, "form": "normal", "variant": "drilbur", "gen": "gen5",
        "dir": "monsterSprite/gen5/0529_drilbur/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def check_turns(rigs, palette):
    pairs = (("W", "N"), ("SW", "N"), ("S", "NW"), ("SE", "NW"), ("E", "NW"))
    for d, back_d in pairs:
        got = synth(rigs, d, "attack", 6, palette, effects=False)
        if not np.array_equal(got, rigs[back_d]["canvas"]):
            raise SystemExit(f"attack turn {d} is not the {back_d} still")
        got8 = synth(rigs, d, "attack", 8, palette, effects=False)
        if not np.array_equal(got8, rigs[back_d]["canvas"]):
            raise SystemExit(f"attack turn {d} frame 8 is not the {back_d} still")
    for d in ("N", "NE", "NW"):
        got = synth(rigs, d, "attack", 7, palette, effects=False)
        if not np.array_equal(got, rigs[d]["canvas"]):
            raise SystemExit(f"attack turn {d} changed the back still")
    for d in DIRS:
        a0 = synth(rigs, d, "attack", 0, palette, effects=False)
        a15 = synth(rigs, d, "attack", 15, palette, effects=False)
        if not np.array_equal(a0, rigs[d]["canvas"]) or not np.array_equal(a15, rigs[d]["canvas"]):
            raise SystemExit(f"attack ends do not match the still for {d}")


def frame_metrics(tiles, palette):
    pal = {tuple(int(v) for v in c) for c in palette}
    off = set()
    partial = 0
    edge = []
    bodies = []
    for name, t in tiles:
        a = t[..., 3]
        partial += int(((a != 0) & (a != 255)).sum())
        op = a == 255
        if op.any():
            used = {tuple(int(v) for v in c) for c in np.unique(t[op][:, :3], axis=0)}
            off |= used - pal
        if op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any():
            edge.append(name)
        bodies.append(largest_fraction(t))
    return off, partial, edge, bodies


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
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    print("mouth", {d: rigs[d]["has_mouth"] for d in DIRS}, flush=True)
    print("lead", {d: rigs[d]["lead"] for d in ("S", "W", "N", "E")}, flush=True)

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
    check_turns(rigs, palette)

    if args.samples:
        tiles = []
        lost = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    t = synth(rigs, d, act, f, palette)
                    tiles.append((f"{act}:{d}:{f}", t))
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        off, partial, edge, bodies = frame_metrics(tiles, palette)
        rest_sum = int(sum(rest_diff.values()))
        print(f"METRICS rest={rest_sum} off_palette={len(off)} partial_alpha={partial} edge={len(edge)}", flush=True)
        if off:
            print("off colors", sorted(off)[:8], flush=True)
        if edge:
            print("edge", edge[:12], flush=True)
        print("disconnected", lost[:12], "count", len(lost), flush=True)
        print("body_main_min", round(min(bodies), 3), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["lead"], 1), round(p["off"], 1), p["claw_mode"], p["turned"], p["spin"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        box = crop_box(rigs["S"]["canvas"])
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
            Image.fromarray(np.concatenate(rows, axis=0), "RGBA").save(work / f"samples_{d}.png")
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
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest and not args.force:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    box = crop_box(rigs["S"]["canvas"])
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    Image.fromarray(np.concatenate(strips, axis=0), "RGBA").save(OUT_DIR / "preview_motion.png")
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
    rest_sum = int(sum(rest_diff.values()))
    print(
        f"METRICS rest={rest_sum} off_palette={len(off)} partial_alpha={partial} edge={len(clip)}",
        flush=True,
    )
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
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_drilbur_actions", "version": VERSION, "key": KEY,
        "dex": 529, "variant": "drilbur", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped mole (two claws, open mouth, one eye, two feet)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front nose x34.95 is left of the eye x49; back nose peek is on the right",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    n = sync_outputs()
    print("synced", n, "files", flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, "h", sprite_h, flush=True)


if __name__ == "__main__":
    main()
