#!/usr/bin/env python3
"""Basculin red-striped (0550) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x21-70 y27-64, 958 px, alpha 0/255,
12 colours, lowest pixel y64):
  Eyes are the ten white pixels, mean x 43.1 (x40-45, y41-46).
  The mouth is the transparent gap at about x28-32 y41-42, between the
  teal snout and the silver lower jaw. The jaw tip is x21, left of the
  eyes, so the still faces screen-left. The roar uses that mouth. It does
  not draw a second one.
  Lower jaw: silver x21-31 y41-51. Dorsal spikes: silver x43-57 y27-37.
  Caudal fin: silver x50-70 y36-55 plus the coloured tip at x>=62.
  Anal fin: silver x52-61 y53-64. No feet, so the walk swims.

normal/back.png (bbox x25-65 y28-65, 925 px, lowest pixel y65, 11 colours,
all in the front palette) has no white eye and no mouth gap. The same
dorsal spikes and a hanging anal fin are there. The broad silver flank
stays on the body. Head teal is still on the left.

Directions, front as drawn looks screen-left:
  S SE E front mirrored (screen-right). W SW front as drawn.
  NE back mirrored. N NW back as drawn.
  Attack 5-8 swaps kind and flips the mirror, so S becomes the N still
  at equal scale. No width squash, no in-plane spin.

  idle 16: float. A 1px breath. Dorsal, tail, anal and jaw sway.
  walk 16: swim. The same fins undulate. No new feet.
  attack: 0 still, 1-4 the mouth opens with a wedge and a ring,
          5-8 the opposite still, 9-12 the tail, anal fin and jaw strike,
          11-12 impact, 13-15 recover.
  hurt 12: white flash, knock-back lean (<=38 deg) and recover.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_basculin-red-striped_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0550_basculin-red-striped/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0550_basculin-red-striped/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0550_basculin-red-striped"
VERSION = "basculin-red-striped_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 176, 176
OFF_X = (CELL_W - 96) // 2
FOOT_Y = 144
SRC_FOOT = {"front": 64, "back": 65}
# kind, mirror. Unmirrored art faces screen-left on both stills.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "W": ("front", False), "SW": ("front", False),
    "NE": ("back", True), "N": ("back", False), "NW": ("back", False),
}
MAX_LEAN = 38.0
# Ceilings this script already draws. finish() will not raise a frame past them.
PEAKS = {"jaw": 20.0, "dorsal": 12.0, "tail": 34.0, "anal": 22.0}
ALL_LIMBS = ("jaw", "dorsal", "tail", "anal")

INK = (16, 16, 16)
GREEN = (65, 164, 82)
GREEN_M = (41, 115, 49)
GREEN_D = (24, 74, 24)
SILVER_L = (230, 230, 230)
SILVER_M = (164, 164, 164)
SILVER_D = (98, 98, 98)
TEAL = (65, 82, 82)
TEAL_D = (32, 49, 57)
RED = (255, 90, 74)
RED_D = (156, 32, 24)
WHITE = (255, 255, 255)
SILVER_COLS = (SILVER_L, SILVER_M, SILVER_D)
GREEN_COLS = (GREEN, GREEN_M, GREEN_D)
TEAL_COLS = (TEAL, TEAL_D)
RED_COLS = (RED, RED_D)
BODY_COLS = GREEN_COLS + RED_COLS + TEAL_COLS + (WHITE,)
OUTLINE_KEEP = (INK,)

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
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + h, dx:dx + w]
    return out


def components(mask):
    """8-connected component masks, largest first."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    found = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        blob = np.zeros_like(mask)
        for y, x in pts:
            blob[y, x] = True
        found.append(blob)
    found.sort(key=lambda b: int(b.sum()), reverse=True)
    return found


def largest_component(mask):
    parts = components(mask)
    if not parts:
        return 0
    return int(parts[0].sum())


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
    pal = palette.astype(np.int32)
    d = ((px[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2, dtype=np.int32)
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


def paint_cushion(layer, allowed, steps):
    """Grow the layer's own colours into allowed pixels (hidden at rest)."""
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            mv = np.zeros_like(src)
            if dy < 0:
                mv[:dy] = src[-dy:]
            elif dy > 0:
                mv[dy:] = src[:-dy]
            elif dx < 0:
                mv[:, :dx] = src[:, -dx:]
            else:
                mv[:, dx:] = src[:, :-dx]
            take = (acc[..., 3] < 128) & allowed & (mv[..., 3] >= 128)
            acc[take] = mv[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def claim_rims(parts, outline, head):
    """Outline that touches a fin and does not touch the body fill."""
    free = outline & ~dilate(head, 1)
    taken = np.zeros_like(free)
    order = [name for name in ("jaw", "dorsal", "anal", "tail") if parts.get(name) is not None and parts[name].any()]
    for _ in range(4):
        for name in order:
            rim = free & dilate(parts[name], 1) & ~taken
            if not rim.any():
                continue
            parts[name] = parts[name] | rim
            taken |= rim
    return parts


def absorb_islands(parts, op):
    """A leftover outline speck joins the fin it already touches."""
    used = np.zeros_like(op)
    for m in parts.values():
        used |= m
    torso = op & ~used
    blobs = components(torso)
    if not blobs:
        raise SystemExit("empty torso")
    for blob in blobs[1:]:
        if int(blob.sum()) > 24:
            ys, xs = np.nonzero(blob)
            raise SystemExit(
                f"torso island {int(blob.sum())} x{int(xs.min())}-{int(xs.max())} y{int(ys.min())}-{int(ys.max())}"
            )
        best, bn = None, 0
        for name, m in parts.items():
            n = int((dilate8(blob) & m).sum())
            if n > bn:
                best, bn = name, n
        if best is None:
            raise SystemExit("island touches nothing")
        parts[best] = parts[best] | blob
    return parts


def mouth_hits(img, jaw):
    """Transparent gap with the lower jaw on the left and the head on the right."""
    op = img[..., 3] >= 128
    h, w = jaw.shape
    ys, xs = np.nonzero(jaw)
    hits = []
    if len(xs) == 0:
        return hits
    y0 = max(0, int(ys.min()) - 2)
    y1 = min(h, int(ys.max()) + 2)
    for y in range(y0, y1):
        for x in range(w):
            if op[y, x]:
                continue
            left = jaw[y, max(0, x - 5):x]
            right_op = op[y, x + 1:x + 6]
            right_jaw = jaw[y, x + 1:x + 6]
            if left.any() and right_op.any() and not right_jaw.any():
                hits.append((x, y))
    return hits


def segment(img, kind):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    silver = any_color(img, SILVER_COLS)
    dorsal = np.zeros_like(op)
    jaw = np.zeros_like(op)
    tail = np.zeros_like(op)
    anal = np.zeros_like(op)
    for blob in components(silver):
        n = int(blob.sum())
        if n < 8:
            continue
        ys, xs = np.nonzero(blob)
        my = float(ys.mean())
        mx = float(xs.mean())
        if my < 38.0:
            dorsal |= blob
        elif kind == "front" and mx < 36.0:
            jaw |= blob
        elif int(ys.min()) >= 54:
            anal |= blob
        elif kind == "front" and mx >= 50.0:
            tail |= blob
    if kind == "front":
        # Coloured caudal tip that is already on the still. Not a new fin.
        tail |= op & (xx >= 62) & ~dorsal & ~jaw & ~anal
    outline = any_color(img, (INK,))
    body_fill = any_color(img, BODY_COLS)
    head = body_fill & ~dorsal & ~jaw & ~tail & ~anal
    parts = {"dorsal": dorsal, "jaw": jaw, "tail": tail, "anal": anal}
    parts = claim_rims(parts, outline, head)
    parts = absorb_islands(parts, op)
    parts = {k: v for k, v in parts.items() if v.any()}
    used = np.zeros_like(op)
    for m in parts.values():
        used |= m
    masks = dict(parts)
    masks["torso"] = op & ~used
    check_partition(op, masks)
    for name, m in parts.items():
        if int(m.sum()) < 8:
            raise SystemExit(f"{kind} {name} collapsed: {int(m.sum())}")
        if largest_component(m) != int(m.sum()):
            raise SystemExit(f"{kind} {name} fell apart")
        if not (m & dilate8(masks["torso"])).any():
            raise SystemExit(f"{kind} {name} does not touch the body")
    if largest_component(masks["torso"]) != int(masks["torso"].sum()):
        raise SystemExit(f"{kind} torso fell apart")
    if int(np.nonzero(op)[0].max()) != SRC_FOOT[kind]:
        raise SystemExit(f"{kind} foot line changed")
    mouth = None
    if kind == "front":
        if "jaw" not in masks or "tail" not in masks or "anal" not in masks or "dorsal" not in masks:
            raise SystemExit(f"front is missing a fin: {sorted(masks)}")
        eye = is_color(img, WHITE)
        hits = mouth_hits(img, masks["jaw"])
        if int(eye.sum()) != 10 or len(hits) < 4:
            raise SystemExit(f"front face changed: eyes {int(eye.sum())} mouth {len(hits)}")
        eye_x = float(np.nonzero(eye)[1].mean())
        mouth_x = float(sum(x for x, _y in hits) / len(hits))
        mouth_y = float(sum(y for _x, y in hits) / len(hits))
        jaw_tip = int(np.nonzero(masks["jaw"])[1].min())
        if not (mouth_x < eye_x and jaw_tip < eye_x):
            raise SystemExit(f"face is not left of the eyes: mouth {mouth_x:.2f} jaw {jaw_tip} eyes {eye_x:.2f}")
        mouth = (int(round(mouth_x)), int(round(mouth_y)))
        if img[mouth[1], mouth[0], 3] >= 128:
            raise SystemExit(f"mouth gap is opaque at {mouth}")
    else:
        if any_color(img, (WHITE,)).any():
            raise SystemExit("back gained an eye")
        if "jaw" in masks or "tail" in masks:
            raise SystemExit("back grew a jaw or a caudal fin that is not separate")
        if "dorsal" not in masks or "anal" not in masks:
            raise SystemExit("back lost a fin")
    return masks, mouth


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
        return {"up": 1.0, "fwd": 1.0, "open": 1.0}

    def tip_after(ang):
        return tip_xy(rotate_nearest(layer, ang, pivot), pivot)

    x_pos, y_pos = tip_after(12.0)
    x_neg, y_neg = tip_after(-12.0)
    up = -1.0 if y_neg < y_pos else 1.0
    if facing > 0:
        fwd = 1.0 if x_pos >= x_neg else -1.0
    else:
        fwd = 1.0 if x_pos <= x_neg else -1.0
    opened = 1.0 if y_pos >= y_neg else -1.0
    return {"up": up, "fwd": fwd, "open": opened}


def build_view(img, masks, kind, mirror, src_foot, mouth_src):
    canvas = place(img, src_foot)
    placed = {k: place(v, src_foot) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    # Both stills face screen-left before the mirror.
    facing = -1.0 if not mirror else 1.0
    pivots = {}
    signs = {}
    depths = {"jaw": 4, "dorsal": 4, "tail": 6, "anal": 5}
    for name in ALL_LIMBS:
        if name not in placed:
            continue
        depth = depths[name]
        layers["torso"] = paint_cushion(layers["torso"], joint_band(placed[name], torso_m, depth), depth)
        pivots[name] = pivot_of(placed[name], torso_m)
        signs[name] = signs_for(layers[name], pivots[name], facing)
    mouth = None
    if mouth_src is not None:
        sx, sy = mouth_src
        x = OFF_X + sx
        y = (FOOT_Y - src_foot) + sy
        if mirror:
            x = CELL_W - 1 - x
        mouth = (int(x), int(y))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas, "mouth": mouth,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, head_dx=0, head_dy=0,
        jaw=0.0, dorsal=0.0, tail=0.0, anal=0.0,
        jaw_mode="open", dorsal_mode="up", tail_mode="up", anal_mode="up",
        lean=0.0, knock=0, flash=False, spark=0, ring=-1, wedge=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        p["bdy"] = -1 if wave(f, n, 1.0) > 0.55 else 0
        p["dorsal"] = wave(f, n, 5.0, phase=0.2)
        p["tail"] = wave(f, n, 8.0, phase=0.4)
        p["anal"] = wave(f, n, 6.0, phase=1.2)
        p["jaw"] = wave(f, n, 4.0, phase=2.0)
        p["dorsal_mode"] = "up"
        p["tail_mode"] = "up"
        p["anal_mode"] = "fwd"
        p["jaw_mode"] = "open"
    elif action == "walk":
        bob = wave(f, n, 1.0, cycles=2.0)
        p["bdy"] = -1 if bob > 0.45 else 0
        p["bdx"] = int(round(wave(f, n, 1.0, cycles=2.0, phase=0.6)))
        p["tail"] = wave(f, n, 18.0, cycles=2.0)
        p["anal"] = wave(f, n, 14.0, cycles=2.0, phase=0.8)
        p["dorsal"] = wave(f, n, 8.0, cycles=2.0, phase=1.5)
        p["jaw"] = wave(f, n, 6.0, cycles=2.0, phase=0.4)
        p["tail_mode"] = "fwd"
        p["anal_mode"] = "fwd"
        p["dorsal_mode"] = "up"
        p["jaw_mode"] = "open"
    elif action == "attack":
        # 0 rest. 1-4 the existing mouth roars. 5-8 are cleared after finish().
        # 9-12 the caudal fin, anal fin and jaw strike. 13-15 recover.
        jaw = [0, 10, 16, 20, 18, 0, 0, 0, 0, 12, 18, 16, 8, 4, 0, 0]
        dorsal = [0, 6, 10, 12, 10, 0, 0, 0, 0, 8, 12, 12, 8, 4, 2, 0]
        tail = [0, 6, 8, 10, 8, 0, 0, 0, 0, 34, 34, 34, 22, 10, 4, 0]
        anal = [0, 4, 6, 8, 6, 0, 0, 0, 0, 22, 22, 22, 14, 6, 2, 0]
        bdx = [0, -1, -1, -1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        p["jaw"] = float(jaw[f])
        p["dorsal"] = float(dorsal[f])
        p["tail"] = float(tail[f])
        p["anal"] = float(anal[f])
        p["bdx"] = int(bdx[f])
        p["jaw_mode"] = "open"
        p["dorsal_mode"] = "up"
        p["tail_mode"] = "fwd" if f >= 9 else "up"
        p["anal_mode"] = "fwd" if f >= 9 else "up"
        p["turned"] = 5 <= f <= 8
        if 1 <= f <= 4:
            p["ring"] = f - 1
            p["wedge"] = f
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.66, 0.46, 0.28, 0.14, 0.05, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["tail"] = 12.0 * env
        p["anal"] = 8.0 * env
        p["dorsal"] = 6.0 * env
        p["jaw"] = 8.0 * env
        p["tail_mode"] = "up"
        p["anal_mode"] = "up"
        p["dorsal_mode"] = "up"
        p["jaw_mode"] = "open"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        for key in ("jaw", "dorsal", "tail", "anal"):
            p[key] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["head_dx"] = 0
        p["head_dy"] = 0
        p["spark"] = 0
        p["ring"] = -1
        p["wedge"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["turned"] = True
    return p


def part_angle(rig, name, mode, amp):
    if abs(amp) < 0.45 or name not in rig["signs"]:
        return 0.0
    return rig["signs"][name][mode] * amp


def moved(layer, ang, pivot, dx, dy):
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, pivot)
    return shift(layer, dx, dy)


def body_shift(rig, p):
    fc = rig["facing"]
    return int(round(fc * p["bdx"])), int(round(p["bdy"]))


def angle_of(rig, p, name):
    if name not in rig["layers"]:
        return 0.0
    return part_angle(rig, name, p[name + "_mode"], p[name])


def part_layer(rig, name, p):
    if name not in rig["layers"]:
        return None
    bdx, bdy = body_shift(rig, p)
    return moved(rig["layers"][name], angle_of(rig, p, name), rig["pivots"][name], bdx, bdy)


def compose(rig, p):
    out = np.zeros_like(rig["canvas"])
    bdx, bdy = body_shift(rig, p)
    paste(out, shift(rig["layers"]["torso"], bdx, bdy))
    for name in ("anal", "tail", "dorsal", "jaw"):
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


def facing_tip(layer, facing):
    m = layer[..., 3] >= 128
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if facing > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def extremity(layer, pivot):
    m = layer[..., 3] >= 128
    if not m.any():
        return None
    return tip_xy(layer, pivot)


def roar(img, origin, fc, age, wedge):
    """Wedge and ring in this still's colours, in front of the mouth that is already there."""
    ox, oy = origin
    cols = (INK, RED, RED_D, SILVER_L, WHITE)
    if wedge:
        for i, (dx, dy) in enumerate(((1, 0), (2, -1), (2, 1), (3, 0), (4, -1), (4, 1))):
            put(img, ox + fc * (dx + wedge // 2), oy + dy, cols[i % 5])
    rad = 6 + age * 3
    prev = None
    for deg in range(-50, 51, 8):
        a = math.radians(deg)
        x = ox + fc * (3 + rad * math.cos(a))
        y = oy + rad * math.sin(a) * 0.62
        put(img, x, y, cols[(deg // 8) % 5])
        if prev is not None:
            put(img, (x + prev[0]) / 2, (y + prev[1]) / 2, cols[age % 5])
        prev = (x, y)


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, RED, SILVER_L, TEAL)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r), (2, r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name in ALL_LIMBS:
        layer = part_layer(rig, name, p)
        if layer is None:
            continue
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
        mirror = not mirror
    return rigs[(kind, mirror)]


def moving(p):
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    if abs(p["lean"]) > 0.4:
        return True
    for name in ALL_LIMBS:
        if abs(p[name]) >= 0.45:
            return True
    return False


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    for name in ALL_LIMBS:
        if abs(p[name]) >= 0.45:
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
    if rotated(p) or moving(p):
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["ring"] >= 0 and not p["turned"] and rig["mouth"] is not None:
        mx, my = rig["mouth"]
        mx += int(round(fc * p["bdx"]))
        my += int(round(p["bdy"]))
        roar(img, (mx, my), fc, p["ring"], p["wedge"])
        img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        strike_name = "tail" if "tail" in rig["layers"] else "anal"
        layer = part_layer(rig, strike_name, p)
        tip = extremity(layer, rig["pivots"][strike_name]) if layer is not None else None
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
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 14)
    x0 = max(0, int(xs.min()) - pad - 28)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 28)
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
    for kind, im in (("front", front), ("back", back)):
        if int(np.nonzero(im[..., 3])[0].max()) != SRC_FOOT[kind]:
            raise SystemExit(f"{kind} foot line changed")
    fm, mouth_f = segment(front, "front")
    bm, mouth_b = segment(back, "back")
    rigs = {}
    for kind, img, masks, mouth in (
        ("front", front, fm, mouth_f),
        ("back", back, bm, mouth_b),
    ):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror, SRC_FOOT[kind], mouth)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs["E"]["facing"] != 1.0 or rigs["W"]["facing"] != -1.0:
        raise SystemExit("front facing is not screen-right when mirrored")
    if rigs["NE"]["facing"] != 1.0 or rigs["N"]["facing"] != -1.0:
        raise SystemExit("back facing does not follow the mirror")
    if rigs["S"]["mouth"] is None or rigs["N"]["mouth"] is not None:
        raise SystemExit("mouth is on the wrong view")
    if rigs[("back", False)] is not rigs["N"]:
        raise SystemExit("turned south is not the north still")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "jaw": (220, 180, 60, 255),
        "dorsal": (80, 160, 220, 255),
        "tail": (20, 70, 200, 255),
        "anal": (40, 90, 160, 255),
        "torso": (70, 170, 90, 255),
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
        "dex": 550, "form": "normal", "variant": "basculin-red-striped", "gen": "gen5",
        "dir": "monsterSprite/gen5/0550_basculin-red-striped/normal", "sheetDir": "agent-sprite-forge",
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
    rig_a = rig_for(rigs, d, rest)
    rig_b = rig_for(rigs, d, hit)
    la = part_layer(rig_a, name, rest)
    lb = part_layer(rig_b, name, hit)
    if la is None or lb is None:
        return None
    fa = rig_b["facing"]
    a = facing_tip(la, fa)
    b = facing_tip(lb, fa)
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
    print("mouth S", rigs["S"]["mouth"], "W", rigs["W"]["mouth"], flush=True)
    print("signs S", rigs["S"]["signs"], "N", rigs["N"]["signs"], flush=True)

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

    turn_diff = {}
    for d in ("S", "N", "E", "W"):
        rig = rig_for(rigs, d, pose("attack", 6))
        img = synth(rigs, d, "attack", 6, palette, effects=True)
        turn_diff[d] = int(np.any(img != rig["canvas"], axis=2).sum())
    print("turn diff", turn_diff, flush=True)
    if any(turn_diff.values()):
        raise SystemExit("attack 5-8 is not the opposite still")

    strike = {name: tip_shift(rigs, "S", name) for name in ALL_LIMBS}
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
        stored = {d: {a: [] for a in ACTIONS} for d in ("S", "W", "N", "E")}
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
                    if d in stored:
                        stored[d][act].append(img[y0:y1, x0:x1])
        for d in ("S", "W", "N", "E"):
            rows = []
            for act in ACTIONS:
                frames = list(stored[d][act])
                blank = np.zeros_like(frames[0])
                while len(frames) < 16:
                    frames.append(blank)
                rows.append(tile_grid(frames, 16))
            sheet = np.concatenate(rows, axis=0)
            z = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
            Image.fromarray(z, "RGBA").save(work / f"samples_{d}.png")
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["tail"], 1), round(p["anal"], 1), round(p["jaw"], 1),
                        p["turned"], p["spin"], p["ring"], p["wedge"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", round(max(leans), 1), flush=True)
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
        "generator": "forge_gen5_basculin-red-striped_actions", "version": VERSION, "key": KEY,
        "dex": 550, "variant": "basculin-red-striped", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "fish (teal snout, silver lower jaw, mouth gap, dorsal spikes, caudal fin, anal fin, no feet)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (jaw tip x21 and mouth gap x~30 are left of eye mean x 43.1); back has no mouth, head teal on the left",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
