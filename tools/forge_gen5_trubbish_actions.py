#!/usr/bin/env python3
"""Trubbish normal (0568) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x26-68 y29-65, 733 px, 14 colours,
alpha 0/255, lowest pixel y65):
  Two eyes. Left eye white+pale x33-36 mean x 34.56, pupil (35, 51).
  Right eye x48-51 mean x 49.53, pupil (49, 52). Combined eye mean x 42.24.
  The jagged mouth is the white teeth between them, x38-44 y53-55, mean x
  41.08, left of the eyes, so the still faces screen-left. The roar uses
  that mouth. It does not invent a second one.
  No feet. The bag bottom is one hem (y65, x50-52). The walk glides.
  Two poison-gas hands, already separate colours. Screen-left hand
  (82,74,57)/(115,98,82)/(164,139,115)/(238,82,164) at x26-34 y55-61.
  Screen-right hand is the larger cloud x58-67 y55-62, plus the cyan tip
  (0,139,230) and (24,90,131). The tied top is the light-green ears,
  one piece at y<=40.
normal/back.png (bbox x21-74 y29-65, 816 px, 11 colours, all in the front
  palette, lowest y65): no white, no pale, no mouth. Two larger gas hands
  (x22-32 and x58-73). The upright tie sits at x49-52, the other side of
  the front's tie, so this drawing is the rear of the same left-facing bag.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn. NW back mirrored.
  Attack 5-8 is that direction's back at equal scale: W turns into N,
  S turns into N flipped. No |cos| squash, no in-plane spin.

  idle 16: the bag floats, the tie and both gas hands sway. Frame 0 is the still.
  walk 16: glide. No new feet. Hands swing, the bag bobs.
  attack: 0 still, 1-4 mouth wedge and ring (front only) plus a small hand lift,
          5-8 the back still, 9-12 both gas hands strike, 13-15 recover.
  hurt 12: white flash, knock-back lean (<=38 deg) and recover.

Head offsets stay 0. Python:
  agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_trubbish_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0568_trubbish/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0568_trubbish/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0568_trubbish"
VERSION = "trubbish_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 176, 176
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 20
FRONT_FOOT = 65
BACK_FOOT = 65
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
# finish() may lift a hand this far. The slap is drawn at 34, above this,
# so the hit is not stretched past the angle this script already draws.
PEAKS = {"arm_l": 16.0, "arm_r": 16.0, "knot": 8.0}
LIMBS = ("knot", "arm_l", "arm_r")

INK = (0, 0, 0)
DK = (16, 49, 32)
MID = (24, 74, 49)
GREEN = (41, 106, 74)
LIGHT = (74, 139, 106)
BROWN = (82, 74, 57)
TAN = (115, 98, 82)
BEIGE = (164, 139, 115)
PALE = (164, 197, 172)
MAGENTA = (189, 49, 123)
PINK = (238, 82, 164)
CYAN = (0, 139, 230)
BLUE = (24, 90, 131)
WHITE = (255, 255, 255)
BODY_COLS = (DK, MID, GREEN, LIGHT)
GAS_COLS = (BROWN, TAN, BEIGE, MAGENTA, PINK, CYAN, BLUE)
FACE_COLS = (WHITE, PALE)
OUTLINE_KEEP = (INK, DK, MID)

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


def dilate8(mask):
    p = np.pad(mask, 1, constant_values=False)
    h, w = mask.shape
    out = mask.copy()
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + h, dx:dx + w]
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


def mask_of(pts, shape):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m


def largest_component(mask):
    groups = component_list(mask)
    if not groups:
        return 0
    return max(len(g) for g in groups)


def largest_component8(mask):
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
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        best = max(best, n)
    return best


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
    """4x nearest in-plane rotation (positive = clockwise on screen). No squash."""
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
    """Grow the layer's own colours into allowed pixels (hidden at rest)."""
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
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def outer_outline(outline, fill, other, body):
    """Gas-cloud edge that does not touch the bag. The shared root stays on the torso."""
    taken = np.zeros_like(fill)
    pool = outline & dilate(fill, 1) & ~fill
    for pts in component_list(pool):
        blob = mask_of(pts, fill.shape)
        if (dilate(blob, 1) & body).any() or (dilate(blob, 1) & other).any():
            continue
        if (blob & dilate(fill, 1)).any():
            taken |= blob
    return taken


def gas_hands(img):
    """The two poison clouds already drawn on the lower sides. No new arms."""
    op = img[..., 3] >= 128
    gas = any_color(img, GAS_COLS)
    groups = component_list(gas)
    if len(groups) != 2:
        raise SystemExit(f"expected two gas hands, got {len(groups)}")
    groups.sort(key=lambda pts: sum(p[1] for p in pts) / len(pts))
    left = mask_of(groups[0], op.shape)
    right = mask_of(groups[1], op.shape)
    body = any_color(img, BODY_COLS)
    ink = is_color(img, INK)
    arm_l = left | outer_outline(ink, left, right, body)
    arm_r = right | outer_outline(ink, right, left, body)
    return arm_l, arm_r


def mouth_mask(img):
    """Teeth between the eyes. Empty on the back, which has no face."""
    face = any_color(img, FACE_COLS)
    if not face.any():
        return np.zeros(img.shape[:2], bool), np.zeros(img.shape[:2], bool)
    eyes = np.zeros(img.shape[:2], bool)
    mouth = np.zeros(img.shape[:2], bool)
    for pts in component_list(face):
        mean = sum(p[1] for p in pts) / len(pts)
        blob = mask_of(pts, img.shape[:2])
        if mean < 37.0 or mean > 46.0:
            eyes |= blob
        else:
            mouth |= blob
    return eyes, mouth


def segment(img, kind, foot):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    arm_l, arm_r = gas_hands(img)
    # The tied top is one piece through y40. y39 splits into the two ears.
    knot = op & (yy <= 40) & ~arm_l & ~arm_r
    eyes, mouth = mouth_mask(img)
    if kind == "back":
        if mouth.any() or eyes.any():
            raise SystemExit("back gained a face")
        mouth = np.zeros_like(op)
    torso = op & ~(arm_l | arm_r | knot | mouth)
    masks = {"knot": knot, "arm_l": arm_l, "arm_r": arm_r, "torso": torso}
    if mouth.any():
        masks = {"knot": knot, "arm_l": arm_l, "arm_r": arm_r, "mouth": mouth, "torso": torso}
    check_partition(op, masks)
    if int(np.nonzero(op)[0].max()) != foot:
        raise SystemExit(f"{kind} foot line changed")
    if not is_color(img, LIGHT).any() or (is_color(img, LIGHT) & ~knot).any():
        raise SystemExit(f"{kind} tie highlight left the knot")
    if (any_color(img, GAS_COLS) & ~(arm_l | arm_r)).any():
        raise SystemExit(f"{kind} gas left the hands")
    for name in ("knot", "arm_l", "arm_r"):
        part = masks[name]
        if largest_component8(part) != int(part.sum()):
            raise SystemExit(f"{kind} {name} fell apart")
        if not (part & dilate8(torso | mouth)).any():
            raise SystemExit(f"{kind} {name} does not touch the bag")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit(f"{kind} torso fell apart")
    if int(knot.sum()) < 80 or int(arm_l.sum()) < 30 or int(arm_r.sum()) < 40:
        raise SystemExit(f"{kind} part collapsed")
    if kind == "front":
        if int(mouth.sum()) < 8 or int(eyes.sum()) < 20:
            raise SystemExit("front face collapsed")
        eye_x = float(np.nonzero(eyes)[1].mean())
        mouth_x = float(np.nonzero(mouth)[1].mean())
        if mouth_x >= eye_x:
            raise SystemExit("front mouth is not left of the eyes")
        if int(np.nonzero(mouth)[1].min()) != 38:
            raise SystemExit("front mouth tip moved")
    return masks, mouth if kind == "front" else np.zeros_like(op)


def place(arr, src_foot):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    y0 = FOOT_Y - src_foot
    x0 = OFF_X
    h, w = arr.shape[:2]
    if y0 < 0 or x0 < 0 or y0 + h > CELL_H or x0 + w > CELL_W:
        raise SystemExit(f"place overflow y {y0} x {x0}")
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def joint_band(part, body, depth):
    touch = part & dilate(body, 1)
    if not touch.any():
        touch = part & dilate8(body)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def pivot_of(part, body):
    touch = part & dilate(body, 1)
    if not touch.any():
        touch = part & dilate8(body)
    if not touch.any():
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(body)
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


def build_view(img, masks, kind, mirror, src_foot, mouth):
    canvas = place(img, src_foot)
    placed = {k: place(v, src_foot) for k, v in masks.items()}
    mouth_p = place(mouth, src_foot)
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
        mouth_p = np.ascontiguousarray(mouth_p[:, ::-1])
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    # The teeth are a locator and a paste layer. They shift with the bag and do not rotate.
    torso_m = placed["torso"]
    facing = 1.0 if mirror else -1.0
    if kind == "back":
        facing = -facing
    pivots = {}
    signs = {}
    anchor = torso_m
    if "mouth" in placed:
        anchor = torso_m | placed["mouth"]
    for name in ("knot", "arm_l", "arm_r"):
        depth = 4 if name.startswith("arm") else 3
        layers["torso"] = paint_cushion(layers["torso"], joint_band(placed[name], anchor, depth), depth)
        pivots[name] = pivot_of(placed[name], anchor)
        signs[name] = signs_for(layers[name], pivots[name], facing)
        anchor = anchor | placed[name]
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
        "mouth": mouth_p, "has_mouth": bool(mouth_p.any()),
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, head_dx=0, head_dy=0,
        arm_l=0.0, arm_r=0.0, knot=0.0,
        arm_mode="up", knot_mode="up",
        lean=0.0, knock=0, flash=False, spark=0, ring=-1,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # A tied bag floats. The hands and the tie sway. No feet to plant.
        p["bdy"] = int(round(wave(f, n, 1.0, phase=0.4)))
        p["arm_l"] = wave(f, n, 7.0, phase=0.25)
        p["arm_r"] = wave(f, n, 7.0, phase=0.25 + math.pi)
        p["knot"] = wave(f, n, 6.0, phase=1.1)
        p["arm_mode"] = "up"
        p["knot_mode"] = "up"
    elif action == "walk":
        # Glide. The gas hands do the moving. The bag bobs. No legs are drawn.
        p["bdy"] = int(round(wave(f, n, 1.0, cycles=2.0)))
        p["bdx"] = int(round(wave(f, n, 1.0, cycles=2.0, phase=0.6)))
        p["arm_l"] = wave(f, n, 14.0, phase=0.15)
        p["arm_r"] = wave(f, n, 14.0, phase=0.15 + math.pi)
        p["knot"] = wave(f, n, 7.0, phase=0.9)
        p["arm_mode"] = "up"
        p["knot_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 the existing mouth roars; the hands lift a little.
        # 5-8 cleared after finish() so the back still stays at equal scale.
        # 9 windup (within 8-22). 10-11 the slap at 34. 12 follow. 13-15 recover.
        arm = [0, 8, 14, 18, 12, 0, 0, 0, 0, 18, 34, 34, 20, 8, 4, 0]
        knot = [0, 4, 8, 10, 6, 0, 0, 0, 0, 8, 6, 6, 4, 2, 0, 0]
        bdx = [0, -1, -1, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        p["arm_l"] = float(arm[f])
        p["arm_r"] = float(arm[f])
        p["knot"] = float(knot[f])
        p["bdx"] = int(bdx[f])
        p["arm_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["knot_mode"] = "up"
        p["turned"] = 5 <= f <= 8
        if 1 <= f <= 4:
            p["ring"] = f - 1
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.66, 0.46, 0.28, 0.14, 0.05, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["arm_l"] = 14.0 * env
        p["arm_r"] = 14.0 * env
        p["knot"] = 8.0 * env
        p["arm_mode"] = "up"
        p["knot_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["arm_l"] = 0.0
        p["arm_r"] = 0.0
        p["knot"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["head_dx"] = 0
        p["head_dy"] = 0
        p["spark"] = 0
        p["ring"] = -1
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["arm_mode"] = "up"
        p["knot_mode"] = "up"
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


def body_shift(rig, p):
    fc = rig["facing"]
    return int(round(fc * p["bdx"])), int(round(p["bdy"]))


def angle_of(rig, p, name):
    if name in ("arm_l", "arm_r"):
        return part_angle(rig, name, p["arm_mode"], p[name])
    if name == "knot":
        return part_angle(rig, name, p["knot_mode"], p["knot"])
    return 0.0


def part_layer(rig, name, p):
    if name not in rig["layers"]:
        return None
    bdx, bdy = body_shift(rig, p)
    return moved(rig["layers"][name], angle_of(rig, p, name), rig["pivots"][name], bdx, bdy)


def arm_order(rig):
    names = [n for n in ("arm_l", "arm_r") if n in rig["layers"]]

    def cx(name):
        m = rig["layers"][name][..., 3] >= 128
        return float(np.nonzero(m)[1].mean())

    # Far hand first, the hand toward the facing side on top.
    names.sort(key=cx, reverse=(rig["facing"] < 0))
    return names


def compose(rig, p):
    out = np.zeros_like(rig["canvas"])
    bdx, bdy = body_shift(rig, p)
    paste(out, shift(rig["layers"]["torso"], bdx, bdy))
    if "mouth" in rig["layers"]:
        paste(out, shift(rig["layers"]["mouth"], bdx, bdy))
    knot = part_layer(rig, "knot", p)
    if knot is not None:
        paste(out, knot)
    for name in arm_order(rig):
        layer = part_layer(rig, name, p)
        if layer is not None:
            paste(out, layer)
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = any_color(out, OUTLINE_KEEP)
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def facing_tip(mask, facing):
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    i = int(xs.argmax()) if facing > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def roar(img, origin, fc, age):
    """Mouth wedge and a ring, in this still's own colours. The teeth are already there."""
    ox, oy = origin
    cols = (INK, WHITE, PALE, PINK, MAGENTA)
    for i, (dx, dy) in enumerate(((1, 0), (2, -1), (2, 1), (3, 0), (1, -1))):
        put(img, ox + fc * dx, oy + dy, cols[i % 5])
    rad = 6 + age * 3
    prev = None
    for deg in range(-48, 49, 8):
        a = math.radians(deg)
        x = ox + fc * (4 + rad * math.cos(a))
        y = oy + rad * math.sin(a) * 0.62
        put(img, x, y, cols[(deg // 8) % 5])
        if prev is not None:
            put(img, (x + prev[0]) / 2, (y + prev[1]) / 2, cols[age % 5])
        prev = (x, y)


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, PINK, MAGENTA, TAN)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r), (2, r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def limb_names(rig):
    return [n for n in LIMBS if n in rig["layers"]]


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name in limb_names(rig):
        layer = part_layer(rig, name, p)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate8(other)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    kind, mirror = VIEW_OF[d]
    if p["turned"]:
        kind = "back" if kind == "front" else "front"
    return rigs[(kind, mirror)]


def moving(p):
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["arm_l"]) >= 0.45 or abs(p["arm_r"]) >= 0.45 or abs(p["knot"]) >= 0.45:
        return True
    return False


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["arm_l"]) >= 0.45 or abs(p["arm_r"]) >= 0.45 or abs(p["knot"]) >= 0.45:
        return True
    return False


def mouth_origin(rig, p):
    if not rig["has_mouth"]:
        return None
    bdx, bdy = body_shift(rig, p)
    shifted = shift(rig["mouth"].astype(np.uint8), bdx, bdy) > 0
    return facing_tip(shifted, rig["facing"])


def hand_tip(rig, name, p):
    layer = part_layer(rig, name, p)
    if layer is None:
        return None
    m = layer[..., 3] >= 128
    return facing_tip(m, rig["facing"])


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
        grounded = body[..., 3] >= 128
        if grounded.any():
            bot = int(np.nonzero(grounded)[0].max())
            body = shift(body, 0, FOOT_Y - bot)
    if rotated(p) or moving(p):
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["ring"] >= 0 and not p["turned"] and rig["has_mouth"]:
        tip = mouth_origin(rig, p)
        if tip is not None:
            roar(img, tip, fc, p["ring"])
            img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        for name in arm_order(rig):
            tip = hand_tip(rig, name, p)
            if tip is not None:
                impact(img, (tip[0] + fc * 2, tip[1]), fc, p["spark"])
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
    y0 = max(0, int(ys.min()) - pad - 18)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 10)
    x0 = max(0, int(xs.min()) - pad - 22)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 22)
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


def frame_faults(img, pal):
    a = img[..., 3]
    partial = int(((a != 0) & (a != 255)).sum())
    op = a >= 128
    edge = bool(op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any())
    off = set()
    solid = a == 255
    if solid.any():
        for c in np.unique(img[solid][:, :3], axis=0):
            t = tuple(int(v) for v in c)
            if t not in pal:
                off.add(t)
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
    fm, mouth_f = segment(front, "front", FRONT_FOOT)
    bm, mouth_b = segment(back, "back", BACK_FOOT)
    rigs = {}
    for kind, img, masks, foot, mouth in (
        ("front", front, fm, FRONT_FOOT, mouth_f),
        ("back", back, bm, BACK_FOOT, mouth_b),
    ):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror, foot, mouth)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs["W"]["facing"] != -1.0 or rigs["E"]["facing"] != 1.0:
        raise SystemExit("front facing is not left-as-drawn")
    if rigs["N"]["has_mouth"] or not rigs["W"]["has_mouth"]:
        raise SystemExit("mouth is on the wrong view")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "knot": (240, 210, 80, 255),
        "arm_l": (230, 80, 150, 255),
        "arm_r": (80, 180, 220, 255),
        "mouth": (255, 255, 255, 255),
        "torso": (70, 160, 90, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
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
        "dex": 568, "form": "normal", "variant": "trubbish", "gen": "gen5",
        "dir": "monsterSprite/gen5/0568_trubbish/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def tip_shift(rigs, d, name):
    rest = pose("attack", 0)
    hit = pose("attack", 11)
    rig = rig_for(rigs, d, rest)
    la = part_layer(rig, name, rest)
    lb = part_layer(rig, name, hit)
    if la is None or lb is None:
        return None
    fa = rig["facing"]
    a = facing_tip(la[..., 3] >= 128, fa)
    b = facing_tip(lb[..., 3] >= 128, fa)
    if a is None or b is None:
        return None
    return round((b[0] - a[0]) * fa, 2)


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
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, flush=True)
    print("parts", {k: int(v.sum()) for k, v in fm.items()}, {k: int(v.sum()) for k, v in bm.items()}, flush=True)
    print("signs W", rigs["W"]["signs"], "N", rigs["N"]["signs"], flush=True)

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
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    turn_w = synth(rigs, "W", "attack", 6, palette, effects=False)
    turn_s = synth(rigs, "S", "attack", 6, palette, effects=False)
    turn_diff = {
        "W": int(np.any(turn_w != rigs["N"]["canvas"], axis=2).sum()),
        "S": int(np.any(turn_s != rigs["NW"]["canvas"], axis=2).sum()),
    }
    print("turn diff", turn_diff, flush=True)
    if any(turn_diff.values()):
        raise SystemExit("attack 5-8 is not the N back at equal scale")

    strike = {name: tip_shift(rigs, "S", name) for name in ("arm_l", "arm_r")}
    print("strike dx S", strike, flush=True)

    pal = {tuple(int(v) for v in c) for c in palette}
    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        y0, y1, x0, x1 = box
        off = set()
        partial = 0
        edge = []
        lost = []
        bodies = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    img = synth(rigs, d, act, f, palette)
                    o, part, ed = frame_faults(img, pal)
                    off |= o
                    partial += part
                    if ed:
                        edge.append(f"{act}:{d}:{f}")
                    if act in ("attack", "hurt", "walk"):
                        bodies.append(largest_fraction(img))
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["arm_l"], 1), round(p["arm_r"], 1), round(p["knot"], 1),
                        p["arm_mode"], p["turned"], p["spin"], p["ring"], p["bdx"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        report = {
            "rest_diff_max": max(rest_diff.values()),
            "turn_diff_max": max(turn_diff.values()),
            "off_palette": len(off),
            "partial_alpha": partial,
            "edge_n": len(edge),
            "edge_touching": edge[:12],
            "disconnected": len(lost),
            "body_main_min": round(min(bodies), 3) if bodies else 1,
        }
        print("qa_samples", json.dumps(report), flush=True)
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
        "turn_diff": turn_diff,
        "off_palette_colors": len(off), "partial_alpha": partial, "edge_touching": clip[:12],
        "edge_n": len(clip), "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "bounds": bounds, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (not any(rest_diff.values()) and not any(idle0.values()) and not any(turn_diff.values())
          and not off and partial == 0 and not clip)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_trubbish_actions", "version": VERSION, "key": KEY,
        "dex": 568, "variant": "trubbish", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "trash bag (tied top, two poison-gas hands, jagged mouth, no feet)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (mouth mean x 41.08 < eye mean x 42.24); back has no face",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
