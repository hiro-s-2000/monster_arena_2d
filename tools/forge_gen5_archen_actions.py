#!/usr/bin/env python3
"""Archen normal (0566) 8-direction action sheet from this form's own stills.

Measured on normal/front.png (96x96, bbox x19-76 y29-67, 1121 opaque, foot y67,
13 colours, alpha 0/255):
  Faces screen-left. The white eye is x40-43 y33-38 (mean x 41.6). The snout
  edge is dark red at x29, left of the eye. A blue crest sits behind the eye
  at x40-48 y29-44. Mouth whites are (37,38) and (34,39), on the snout side
  of the eye, so the roar may use a wedge and a ring.
  Fore wing is the blue membrane x19-30 y50-58 plus the clawed hand x22-28
  y43-49 (they touch). Aft wing hangs at x60-72 y54-67 and is the lowest
  pixel. Tail diamond is blue at x61-75 y33-43. Two feet: near (face side)
  x34-42 y59-63, far x42-50 y60-66.

normal/back.png (bbox x18-78 y29-65, 1064 opaque, foot y65, same 13 colours)
  faces screen-right. Eye mean x 55.4. Red snout reaches x67, right of the
  eye. Crest is the blue left of the eye (x48-58 y29-44). Mouth whites are
  (59,39) and (63,39). Tail is the upper-left blue (x24-42 y38-46). Far wing
  is the left fan (x18-42 y47-53). Near wing is the right blue (x66-78
  y48-61). Near foot is the right one and reaches y65. Far foot is x44-49.

Directions: front still faces left, so S SE E mirror it (screen-right) and
W SW keep it. Back still faces right, so NE keeps it and N NW mirror it.
E is screen-right, W is screen-left, N is the back. Attack 5-8 swaps to the
opposite still at equal scale (same mirror flag). No |cos| squash, no
in-plane spin, the body is not laid on its side.

  idle   : feet planted, crest and wings settle, tail wags. Frame 0 is the still.
  walk   : the two drawn feet step. Wings flap. No new legs.
  attack : 0 still, 1-4 crest up and a mouth wedge plus ring,
           5-8 the opposite still, 9-12 the clawed wing strikes with the
           tail and a short kick, 13-15 recover.
  hurt   : 12 frames, lean at most 38 degrees around the rear foot.

Head offsets stay 0. Palette is this still's own colours. Alpha is 0 or 255.
front/back png and gif are never written.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen5_archen_actions.py [--samples] [--force]
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
SRC_DIR = ROOT / "monsterSprite/gen5/0566_archen/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0566_archen/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0566_archen"
VERSION = "archen_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 176, 168
FOOT_Y = 136
OFF_X = (CELL_W - 96) // 2
FRONT_FOOT = 67
BACK_FOOT = 65
# front.png looks screen-left, back.png looks screen-right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "SW": ("front", False), "W": ("front", False),
    "NE": ("back", False), "N": ("back", True), "NW": ("back", True),
}
MAX_LEAN = 38.0
LIMBS = ("crest", "wing", "wing_far", "tail", "foot_near", "foot_far")
ANGLES = ("crest", "wing", "wing_far", "tail")
# Max angles this script already draws. finish() may lift weaker attack
# frames up to these, and must not be given a higher target.
PEAKS = {"crest": 20.0, "wing": 34.0, "wing_far": 16.0, "tail": 20.0}

INK = (0, 0, 0)
GREY = (57, 57, 57)
BLUE = (49, 131, 246)
BLUE_M = (49, 82, 156)
BLUE_D = (16, 49, 106)
RED = (222, 82, 74)
RED_M = (148, 65, 65)
RED_D = (82, 49, 49)
YEL = (246, 205, 98)
TAN = (189, 148, 82)
BROWN = (82, 65, 41)
WHITE = (230, 230, 230)
SILVER = (156, 156, 172)
BLUE_COLS = (BLUE, BLUE_M, BLUE_D)
WHITE_COLS = (WHITE, SILVER)
OUTLINE_KEEP = (INK, GREY, RED_D)

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


def flood(mask, seeds):
    out = seeds & mask
    while True:
        nxt = dilate(out, 1) & mask
        if (nxt == out).all():
            return out
        out = nxt


def component_list(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    groups = []
    for y0, x0 in zip(*np.nonzero(mask)):
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


def require_piece(name, mask):
    if not mask.any():
        raise SystemExit(f"{name} is empty")
    if largest_component(mask) != int(mask.sum()):
        raise SystemExit(f"{name} fell apart")


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
    pal = palette.astype(np.int32)
    d = ((px[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
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


def blues_of(img):
    return any_color(img, BLUE_COLS)


def absorb_outline(torso, parts, limit=16):
    """Small outline islands left by a colour cut join the limb they touch."""
    changed = True
    while changed:
        changed = False
        groups = component_list(torso)
        if len(groups) <= 1:
            break
        for g in groups[1:]:
            if len(g) > limit:
                ys = [p[0] for p in g]
                xs = [p[1] for p in g]
                raise SystemExit(
                    f"torso island {len(g)} at x{min(xs)}-{max(xs)} y{min(ys)}-{max(ys)}"
                )
            island = np.zeros_like(torso)
            for y, x in g:
                island[y, x] = True
            best = None
            bestn = -1
            for name, m in parts.items():
                n = int((dilate(island, 2) & m).sum())
                if n > bestn:
                    bestn = n
                    best = name
            if best is None or bestn <= 0:
                ys = [p[0] for p in g]
                xs = [p[1] for p in g]
                raise SystemExit(f"orphan outline x{min(xs)} y{min(ys)} n={len(g)}")
            parts[best] = parts[best] | island
            torso = torso & ~island
            changed = True
    return torso, parts


def largest_mask(mask):
    groups = component_list(mask)
    if not groups:
        raise SystemExit("empty component")
    out = np.zeros_like(mask)
    for y, x in max(groups, key=len):
        out[y, x] = True
    return out


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    blue = blues_of(img)
    crest = blue & (yy <= 45) & (xx >= 36) & (xx <= 52)
    tail = blue & (yy <= 46) & (xx >= 58)
    fore = blue & (xx <= 36) & (yy >= 48)
    aft = blue & (yy >= 50) & (xx >= 54)
    claw = op & (xx <= 28) & (yy >= 43) & (yy <= 49)
    wing = fore | claw
    near_allow = op & (yy >= 59) & (xx <= 42) & ~aft
    near = flood(near_allow, near_allow & (yy >= 62) & (xx <= 40))
    far_allow = op & (yy >= 60) & (xx >= 42) & (xx <= 50) & ~blue
    far = flood(far_allow, far_allow & (yy >= 65))
    parts = {
        "crest": crest, "tail": tail, "wing": wing, "wing_far": aft,
        "foot_near": near, "foot_far": far,
    }
    torso = op.copy()
    for m in parts.values():
        torso &= ~m
    torso, parts = absorb_outline(torso, parts, 16)
    masks = dict(parts)
    masks["torso"] = torso
    check_partition(op, masks)
    eye = largest_mask(any_color(img, WHITE_COLS) & (yy <= 42))
    if int((eye & torso).sum()) != int(eye.sum()) or int(eye.sum()) < 12:
        raise SystemExit("front eye left the head")
    ex = float(np.nonzero(eye)[1].mean())
    head = op & (yy <= 42) & (yy >= 29)
    tip = int(np.nonzero(head)[1].min())
    if tip >= ex - 2:
        raise SystemExit(f"front does not face left: snout {tip} eye {ex:.1f}")
    mouth = any_color(img, WHITE_COLS) & (yy <= 42) & (xx < ex - 1) & ~eye
    if int(mouth.sum()) < 1:
        raise SystemExit("front mouth mark missing")
    for name in LIMBS:
        require_piece(name, masks[name])
        if not (masks[name] & dilate(torso, 1)).any():
            raise SystemExit(f"front {name} does not touch the body")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("front torso fell apart")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    blue = blues_of(img)
    crest = largest_mask(blue & (yy <= 44) & (xx >= 46) & (xx <= 60))
    tail = blue & (xx <= 42) & (yy >= 36) & (yy <= 46)
    wing_far = blue & (xx <= 42) & (yy >= 47)
    wing = blue & (xx >= 60)
    near_allow = op & (yy >= 60) & (xx >= 50) & (xx <= 62) & ~blue
    near = flood(near_allow, near_allow & (yy >= 64))
    far_allow = op & (yy >= 60) & (xx >= 42) & (xx <= 49) & ~blue
    far = flood(far_allow, far_allow & (yy >= 62))
    parts = {
        "crest": crest, "tail": tail, "wing": wing, "wing_far": wing_far,
        "foot_near": near, "foot_far": far,
    }
    torso = op.copy()
    for m in parts.values():
        torso &= ~m
    torso, parts = absorb_outline(torso, parts, 16)
    masks = dict(parts)
    masks["torso"] = torso
    check_partition(op, masks)
    eye = largest_mask(any_color(img, WHITE_COLS) & (yy <= 42) & (xx >= 48))
    if int((eye & torso).sum()) != int(eye.sum()) or int(eye.sum()) < 10:
        raise SystemExit("back eye left the head")
    ex = float(np.nonzero(eye)[1].mean())
    head = op & (yy <= 44) & (yy >= 29) & (xx >= 48)
    tip = int(np.nonzero(head)[1].max())
    if tip <= ex + 2:
        raise SystemExit(f"back does not face right: snout {tip} eye {ex:.1f}")
    mouth = any_color(img, WHITE_COLS) & (yy <= 42) & (xx > ex + 1)
    if int(mouth.sum()) < 1:
        raise SystemExit("back mouth mark missing")
    for name in LIMBS:
        require_piece(name, masks[name])
        if not (masks[name] & dilate(torso, 1)).any():
            raise SystemExit(f"back {name} does not touch the body")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("back torso fell apart")
    return masks


def place(arr, off_y):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    h, w = arr.shape[:2]
    if off_y < 0 or OFF_X < 0 or off_y + h > CELL_H or OFF_X + w > CELL_W:
        raise SystemExit(f"place overflow y {off_y}")
    out[off_y:off_y + h, OFF_X:OFF_X + w] = arr
    return out


def joint_band(part, body, depth):
    touch = part & dilate(body, 1)
    if not touch.any():
        touch = part & dilate(body, 3)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def pivot_of(part, body):
    touch = part & dilate(body, 1)
    if not touch.any():
        touch = part & dilate(body, 4)
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


def build_view(img, masks, kind, mirror, src_foot):
    off_y = FOOT_Y - src_foot
    canvas = place(img, off_y)
    placed = {k: place(v, off_y) for k, v in masks.items()}
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
        ("crest", 3), ("wing", 4), ("wing_far", 4), ("tail", 3),
        ("foot_near", 5), ("foot_far", 5),
    ):
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth
        )
    facing = 1.0 if mirror else -1.0
    if kind == "back":
        facing = -facing
    pivots = {}
    signs = {}
    for name in ANGLES:
        pivots[name] = pivot_of(placed[name], torso_m)
        signs[name] = signs_for(layers[name], pivots[name], facing)
    for name in ("foot_near", "foot_far"):
        pivots[name] = pivot_of(placed[name], torso_m)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    white = any_color(canvas, WHITE_COLS)
    eye = largest_mask(white & (np.mgrid[0:CELL_H, 0:CELL_W][0] < FOOT_Y - 18))
    ey, ex = np.nonzero(eye)
    eye_x = float(ex.mean())
    y0, y1 = int(ey.min()) - 1, int(ey.max()) + 4
    band = op & (np.mgrid[0:CELL_H, 0:CELL_W][0] >= y0) & (np.mgrid[0:CELL_H, 0:CELL_W][0] <= y1)
    xx = np.mgrid[0:CELL_H, 0:CELL_W][1]
    if facing > 0:
        band &= xx >= eye_x
        fx = int(np.nonzero(band)[1].max())
    else:
        band &= xx <= eye_x
        fx = int(np.nonzero(band)[1].min())
    fy = int(np.nonzero(band[:, fx])[0].mean()) if band[:, fx].any() else int(ey.mean())
    pivots["snout"] = (float(fx), float(fy))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, head_dx=0, head_dy=0,
        crest=0.0, wing=0.0, wing_far=0.0, tail=0.0,
        crest_mode="up", wing_mode="up", wing_far_mode="up", tail_mode="up",
        step_near=(0, 0), step_far=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0, ring=-1,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        p["crest"] = wave(f, n, 6.0, phase=0.4)
        p["wing"] = wave(f, n, 8.0, phase=0.9)
        p["wing_far"] = wave(f, n, 5.0, phase=1.4)
        p["tail"] = wave(f, n, 10.0, phase=0.2)
        p["bdy"] = -1 if wave(f, n, 1.0, phase=0.3) > 0.55 else 0
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_near"] = (int(round(2 * max(0.0, sa))), int(round(2 * max(0.0, sa))))
        p["step_far"] = (int(round(2 * max(0.0, sb))), int(round(2 * max(0.0, sb))))
        p["wing"] = wave(f, n, 14.0, phase=0.2)
        p["wing_far"] = wave(f, n, 8.0, phase=1.0)
        p["tail"] = wave(f, n, 12.0, phase=0.6, cycles=2.0)
        p["crest"] = wave(f, n, 4.0, phase=0.4)
        p["bdy"] = -1 if abs(math.sin(u)) > 0.65 else 0
    elif action == "attack":
        # 0 rest. 1-4 the crest lifts and the mouth roars. 5-8 the other side.
        # 9-12 the clawed wing strikes. 11 is the hit. 13-15 recover.
        # Values on 9-14 already sit at or above finish()'s envelope.
        crest = [0, 8, 14, 18, 20, 0, 0, 0, 0, 16, 18, 20, 12, 6, 2, 0]
        wing = [0, 8, 12, 16, 18, 0, 0, 0, 0, 30, 32, 34, 22, 10, 4, 0]
        wing_far = [0, 4, 8, 10, 12, 0, 0, 0, 0, 14, 16, 16, 10, 4, 2, 0]
        tail = [0, 6, 10, 14, 16, 0, 0, 0, 0, 16, 18, 20, 12, 6, 2, 0]
        bdx = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        kick = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        lift = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0]
        p["crest"] = float(crest[f])
        p["wing"] = float(wing[f])
        p["wing_far"] = float(wing_far[f])
        p["tail"] = float(tail[f])
        p["bdx"] = int(bdx[f])
        p["step_near"] = (int(kick[f]), int(lift[f]))
        p["wing_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["tail_mode"] = "back" if 9 <= f <= 14 else "up"
        p["turned"] = 5 <= f <= 8
        if 1 <= f <= 4:
            p["ring"] = f - 1
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.75, 1.0, 0.90, 0.70, 0.50, 0.32, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["crest"] = 10.0 * env
        p["wing"] = 14.0 * env
        p["wing_far"] = 8.0 * env
        p["tail"] = 12.0 * env
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    # 5-8 is the opposite still at equal scale. Drop the strike so the
    # silhouette is not swung.
    if action == "attack" and 5 <= f <= 8:
        for key in ANGLES:
            p[key] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["head_dx"] = 0
        p["head_dy"] = 0
        p["step_near"] = (0, 0)
        p["step_far"] = (0, 0)
        p["spark"] = 0
        p["ring"] = -1
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["turned"] = True
        p["flash"] = False
    return p


def part_angle(rig, name, mode, amp):
    if abs(amp) < 0.45 or name not in rig["signs"]:
        return 0.0
    sign = rig["signs"][name]
    if mode == "up":
        return sign["up"] * amp
    if mode == "back":
        return -sign["fwd"] * amp
    return sign["fwd"] * amp


def moved(layer, ang, pivot, dx, dy):
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, pivot)
    return shift(layer, dx, dy)


def body_shift(rig, p):
    fc = rig["facing"]
    return int(round(fc * p["bdx"])), int(round(p["bdy"]))


def part_layer(rig, name, p):
    L = rig["layers"]
    piv = rig["pivots"]
    bdx, bdy = body_shift(rig, p)
    fc = rig["facing"]
    if name in ("foot_near", "foot_far"):
        fwd, lift = p["step_near"] if name == "foot_near" else p["step_far"]
        # Feet ignore the body bob so the soles stay down. The lunge (bdx) is shared.
        return shift(L[name], bdx + int(round(fc * fwd)), -int(lift))
    ang = part_angle(rig, name, p[f"{name}_mode"], p[name])
    return moved(L[name], ang, piv[name], bdx, bdy)


def compose(rig, p):
    out = np.zeros_like(rig["canvas"])
    bdx, bdy = body_shift(rig, p)
    paste(out, shift(rig["layers"]["torso"], bdx, bdy))
    for name in ("wing_far", "tail", "foot_far", "foot_near", "wing", "crest"):
        paste(out, part_layer(rig, name, p))
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


def facing_tip(layer, facing):
    m = layer[..., 3] >= 128
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if facing > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def roar(img, origin, fc, age):
    """Mouth wedge and a ring, in this still's own colours."""
    ox, oy = origin
    cols = (INK, RED, WHITE, BLUE, YEL)
    for i, (dx, dy) in enumerate(((1, 0), (2, -1), (2, 1), (3, 0), (2, 0), (4, -1), (4, 1))):
        if i > 3 + age:
            break
        put(img, ox + fc * dx, oy + dy, cols[i % 5])
    rad = 6 + age * 3
    prev = None
    for deg in range(-48, 49, 8):
        a = math.radians(deg)
        x = ox + fc * (3 + rad * math.cos(a))
        y = oy + rad * math.sin(a) * 0.62
        put(img, x, y, cols[(deg // 8) % 5])
        if prev is not None:
            put(img, (x + prev[0]) / 2, (y + prev[1]) / 2, cols[age % 5])
        prev = (x, y)


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, YEL, RED, BLUE)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r), (2, r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name in LIMBS:
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


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    for key in ANGLES:
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
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], piv)
        grounded = body[..., 3] >= 128
        if grounded.any():
            bot = int(np.nonzero(grounded)[0].max())
            body = shift(body, 0, FOOT_Y - bot)
    if rotated(p):
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["ring"] >= 0 and not p["turned"]:
        sx, sy = rig["pivots"]["snout"]
        bdx, bdy = body_shift(rig, p)
        roar(img, (sx + bdx, sy + bdy), fc, p["ring"])
        img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tip = facing_tip(part_layer(rig, "wing", p), fc)
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
    y0 = max(0, int(ys.min()) - pad - 28)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 24)
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


def frame_metrics(images, palette):
    pal = {tuple(int(v) for v in c) for c in palette}
    off = set()
    partial = 0
    edge = 0
    for img in images:
        a = img[..., 3]
        partial += int(((a != 0) & (a != 255)).sum())
        op = a == 255
        if op.any():
            used = {tuple(int(v) for v in c) for c in np.unique(img[op][:, :3], axis=0)}
            off |= used - pal
        if op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any():
            edge += 1
    return len(off), partial, edge


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
    for kind, img, masks, foot in (
        ("front", front, fm, FRONT_FOOT),
        ("back", back, bm, BACK_FOOT),
    ):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror, foot)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "crest": (40, 140, 255, 255),
        "wing": (255, 80, 40, 255),
        "wing_far": (255, 180, 40, 255),
        "tail": (180, 40, 220, 255),
        "foot_near": (40, 200, 80, 255),
        "foot_far": (20, 180, 180, 255),
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
        tiles.append(np.repeat(np.repeat(part[y0:y1, x0:x1], 3, 0), 3, 1))
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
        "dex": 566, "form": "normal", "variant": "archen", "gen": "gen5",
        "dir": "monsterSprite/gen5/0566_archen/normal", "sheetDir": "agent-sprite-forge",
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
    ap.add_argument("--no-publish", action="store_true")
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
    print("snout S", rigs["S"]["pivots"]["snout"], "N", rigs["N"]["pivots"]["snout"], flush=True)
    print("signs S", rigs["S"]["signs"], flush=True)

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
    if still:
        raise SystemExit("frame 0 does not match the still")

    turn_diff = {}
    for d in DIRS:
        p = pose("attack", 6)
        rig = rig_for(rigs, d, p)
        img = synth(rigs, d, "attack", 6, palette, effects=True)
        turn_diff[d] = int(np.any(img != rig["canvas"], axis=2).sum())
    print("turn diff", turn_diff, flush=True)
    if any(turn_diff.values()):
        raise SystemExit("attack 5-8 is not the opposite still")

    if args.samples:
        images = []
        lost = []
        bodies = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    images.append(synth(rigs, d, act, f, palette))
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
                    if act in ("attack", "hurt", "walk") and f:
                        bodies.append(largest_fraction(images[-1]))
        off, partial, edge = frame_metrics(images, palette)
        print(
            f"METRICS still_diff={still} off_palette={off} partial_alpha={partial} edge_touch={edge}",
            flush=True,
        )
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        if bodies:
            print("body_main_min", round(min(bodies), 3), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["wing"], 1), round(p["tail"], 1), round(p["crest"], 1),
                        p["wing_mode"], p["turned"], p["spin"], p["bdx"]))
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
            sheet = np.concatenate(rows, axis=0)
            z = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
            Image.fromarray(z, "RGBA").save(work / f"samples_{d}.png")
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
    box = crop_box(rigs["S"]["canvas"])
    y0, y1, x0, x1 = box
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.concatenate(strips, axis=0)
    motion = np.repeat(np.repeat(motion, 2, 0), 2, 1)
    Image.fromarray(motion, "RGBA").save(OUT_DIR / "preview_motion.png")
    preview_image(sheets).save(OUT_DIR / "preview.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", OUT_DIR / "actions" / "preview.png")

    cells = []
    bodies = []
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                cells.append(t)
                if act in ("attack", "hurt", "walk"):
                    bodies.append(largest_fraction(t))
    off, partial, edge = frame_metrics(cells, palette)
    import gen5_publish
    bounds = gen5_publish.measure_idle_bounds(OUT_DIR / "sheet.png", CELL_W, CELL_H, rows=8)
    sprite_h = max(bounds[i + 3] for i in range(0, 32, 4))
    idle0 = {}
    for di, d in enumerate(DIRS):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        idle0[d] = int(np.any(t != rigs[d]["canvas"], axis=2).sum())
    still = int(sum(rest_diff.values()) + sum(idle0.values()))
    print(
        f"METRICS still_diff={still} off_palette={off} partial_alpha={partial} edge_touch={edge}",
        flush=True,
    )
    report = {
        "version": VERSION, "hash": digest, "rest_diff": rest_diff, "idle0": idle0,
        "off_palette_colors": off, "partial_alpha": partial, "edge_touch": edge,
        "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "bounds": bounds, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = still == 0 and off == 0 and partial == 0 and edge == 0
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_archen_actions", "version": VERSION, "key": KEY,
        "dex": 566, "variant": "archen", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": off,
        "archetype": "biped feathered raptor (crest, clawed wing, aft wing, tail diamond, two feet)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (snout x < eye x); back snout is screen-right",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    if not args.no_publish:
        publish(bounds, sprite_h)
        print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
