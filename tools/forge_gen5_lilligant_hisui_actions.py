#!/usr/bin/env python3
"""Hisuian Lilligant (0549 regional) 8-direction actions from this form's stills.

Measured on regional/lilligant-hisui/front.png (96x96, bbox x19-72 y11-82,
1275 px, 15 colours, alpha 0/255, sole y82):
  The white face (253,253,253) sits at x39-48 y25-32, mean x 44.03.
  Ink that touches that face in y24-36 is the eyes. Their mean x is right of
  the yellow/tan chin, whose leftmost pixel is x39 y36. The still faces
  screen-left. The pink crown trails to the right, the back of the head.
  The dark bar at y35 (x43-47) is also on back.png (y35 x42-46), so it is the
  collar, not a mouth. The roar does not invent a wedge or rings. It lifts
  the flower crown, the two leaf sleeves, and the hanging bloom.
  Two green leaf sleeves: screen-left x19-42 y35-48, screen-right x49-72
  y39-56. Hanging bloom x39-51 y54-70 (pink, white, cream). Two yellow shoes
  from y69, split at x45. The sole y82 is the screen-left shoe.
regional/lilligant-hisui/back.png (bbox x18-70 y11-82, 1219 px, same 15
  colours, sole y82): the crown is on the screen-left and the high leaf is
  on the screen-right, the rear of the left-facing front. No face, no mouth.
  Same ground line, so no vertical shift.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (away, screen-right). NW back mirrored.
  E is screen-right, W is screen-left, N is the back.
  Attack 5-8 shows that direction's back at equal scale. Back rows stay.
  No |cos| squash and no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_lilligant_hisui_actions.py
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen5_attack_strike import finish, save_peaks

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen5/0549_lilligant/regional/lilligant-hisui"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0549_lilligant/regional/lilligant-hisui/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0549_lilligant-hisui"
VERSION = "lilligant-hisui_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 256, 256
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 24
FRONT_FOOT = 82
BACK_FOOT = 82
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. W keeps that picture. Its back is N.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Wind-up ceiling. The leaf slash and the kick are drawn at 34, above these,
# so finish() does not push a frame past the angle this script already paints.
# Frames 5-8 are cleared after finish() so the back still is not swung.
PEAKS = {"arm": 18.0, "kick": 16.0}
LIMBS = ("arm_l", "arm_r", "crown", "bloom", "foot_l", "foot_r")

INK = (16, 16, 16)
GREEN_D = (49, 90, 49)
GREEN = (57, 172, 57)
GREEN_M = (74, 131, 74)
OLIVE = (82, 98, 41)
PINK_D = (112, 54, 90)
PALE = (156, 180, 98)
MAGENTA = (189, 89, 162)
TAN = (189, 164, 82)
LIME = (197, 238, 123)
CREAM = (205, 197, 189)
WHITE = (253, 253, 253)
PINK = (255, 132, 189)
PINK_L = (255, 187, 219)
YELLOW = (255, 222, 65)
GREEN_COLS = (GREEN_D, GREEN, GREEN_M)
PINK_COLS = (PINK_D, MAGENTA, PINK, PINK_L)
SHOE_COLS = (YELLOW, TAN, OLIVE)

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


def mask_from(pts, shape):
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


def edge_count(a, b):
    return int((a & dilate(b, 1)).sum())


def outer_ink(part, op, ink):
    """Outline that touches this part and does not also touch the rest of the body."""
    rest = op & ~part & ~ink
    touch_part = ink & dilate(part, 1)
    touch_rest = ink & dilate(rest, 1)
    return touch_part & ~touch_rest


def segment(img, kind):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    green = any_color(img, GREEN_COLS)
    pink = any_color(img, PINK_COLS)
    ink = is_color(img, INK)
    arms = []
    for g in component_list(green):
        if len(g) < 100:
            continue
        arms.append(mask_from(g, op.shape))
    if len(arms) != 2:
        raise SystemExit(f"{kind} expected two leaf sleeves, got {len(arms)}")
    arms.sort(key=lambda m: float(np.nonzero(m)[1].mean()))
    arm_l, arm_r = arms
    arm_l = arm_l | outer_ink(arm_l, op, ink)
    arm_r = arm_r | outer_ink(arm_r, op, ink)
    # Crown is the headdress above the face, plus pink petals beside it.
    crown = (op & (yy <= 23)) | (pink & (yy <= 32))
    crown = crown | outer_ink(crown & ~ink, op, ink)
    crown &= ~arm_l & ~arm_r
    # Hanging bloom in front of the skirt. The ankle below y70 stays off it.
    bloom = pink & (yy >= 54) & (yy <= 70)
    petals = (is_color(img, WHITE) | is_color(img, CREAM) | is_color(img, PINK_L))
    bloom = bloom | (petals & (yy >= 54) & (yy <= 68) & (xx >= 36) & (xx <= 54))
    bloom = bloom | outer_ink(bloom & ~ink, op, ink)
    bloom &= ~crown & ~arm_l & ~arm_r
    shoes = any_color(img, SHOE_COLS) | ink
    foot = shoes & (yy >= 74) & (xx >= 38) & (xx <= 54)
    foot &= ~bloom & ~crown & ~arm_l & ~arm_r
    if int(foot.sum()) < 40:
        raise SystemExit(f"{kind} shoes are missing")
    mid = float(np.median(np.nonzero(foot)[1]))
    foot_l = foot & (xx <= mid)
    foot_r = foot & (xx > mid)
    masks = {
        "arm_l": arm_l, "arm_r": arm_r, "crown": crown, "bloom": bloom,
        "foot_l": foot_l, "foot_r": foot_r,
    }
    # Eyes stay on the face. A collar line at y35 is on both stills, so it is
    # not an eye and not a mouth.
    if kind == "front":
        white = is_color(img, WHITE) & (yy >= 24) & (yy <= 36)
        groups = component_list(white)
        if not groups:
            raise SystemExit("front has no white face")
        groups.sort(key=len, reverse=True)
        face = mask_from(groups[0], op.shape)
        eyes = ink & dilate(face, 1) & (yy >= 24) & (yy <= 36)
        for name in list(masks):
            masks[name] &= ~eyes
    else:
        eyes = np.zeros_like(op)
        face_white = is_color(img, WHITE) & (yy >= 24) & (yy <= 36)
        if int(face_white.sum()) > 8:
            raise SystemExit("back grew a face")

    def torso_of():
        acc = np.zeros_like(op)
        for m in masks.values():
            acc |= m
        return op & ~acc

    torso = torso_of()
    groups = component_list(torso)
    groups.sort(key=len, reverse=True)
    for g in groups[1:]:
        speck = mask_from(g, op.shape)
        best, best_n = None, -1
        for name, m in masks.items():
            e = edge_count(speck, m)
            if e > best_n:
                best, best_n = name, e
        if best is None or best_n <= 0:
            raise SystemExit(f"{kind} orphan {g[:4]}")
        ymin = min(p[0] for p in g)
        if best in ("bloom", "foot_l", "foot_r") and ymin >= 68:
            masks["foot_l"] |= speck & (xx <= mid)
            masks["foot_r"] |= speck & (xx > mid)
        else:
            masks[best] |= speck
    torso = torso_of()
    masks["torso"] = torso
    check_partition(op, masks)
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit(f"{kind} torso fell apart")
    for name in ("arm_l", "arm_r", "bloom", "foot_l", "foot_r"):
        part = masks[name]
        if largest_component(part) != int(part.sum()):
            raise SystemExit(f"{kind} {name} fell apart ({int(part.sum())})")
        if int(part.sum()) < 8:
            raise SystemExit(f"{kind} {name} is too small")
    anchor = {"foot_l": torso | masks["bloom"], "foot_r": torso | masks["bloom"]}
    for name in LIMBS:
        body = anchor.get(name, torso)
        if not touches(masks[name], body):
            raise SystemExit(f"{kind} {name} does not touch the body")
    for g in component_list(masks["crown"]):
        bit = mask_from(g, op.shape)
        if not touches(bit, torso):
            raise SystemExit(f"{kind} crown petal does not touch the head")
    foot_line = op & (yy == (FRONT_FOOT if kind == "front" else BACK_FOOT))
    feet = masks["foot_l"] | masks["foot_r"]
    if int(foot_line.sum()) < 1 or int((foot_line & feet).sum()) != int(foot_line.sum()):
        raise SystemExit(f"{kind} ground line is not a shoe")
    if kind == "front":
        if int((eyes & torso).sum()) != int(eyes.sum()) or int(eyes.sum()) < 6:
            raise SystemExit(f"eyes left the face ({int(eyes.sum())})")
        ex = float(np.nonzero(eyes)[1].mean())
        muz = (is_color(img, YELLOW) | is_color(img, TAN)) & (yy >= 34) & (yy <= 40)
        mx = np.nonzero(muz)[1]
        chin = int(mx.min())
        if not (chin < ex):
            raise SystemExit(f"front does not face left: chin {chin} eye {ex}")
        masks["eyes"] = eyes
    return masks


def place(arr):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    out[OFF_Y:OFF_Y + 96, OFF_X:OFF_X + 96] = arr
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

    x_pos, y_pos = tip_after(18.0)
    x_neg, y_neg = tip_after(-18.0)
    up = -1.0 if y_neg < y_pos else 1.0
    if facing > 0:
        fwd = 1.0 if x_pos >= x_neg else -1.0
    else:
        fwd = 1.0 if x_pos <= x_neg else -1.0
    return {"up": up, "fwd": fwd}


def build_view(img, masks, kind, mirror):
    canvas = place(img)
    placed = {k: place(v) for k, v in masks.items() if k != "eyes"}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    bloom_m = placed["bloom"]
    body_m = torso_m | bloom_m
    for name, depth in (
        ("arm_l", 6), ("arm_r", 6), ("crown", 6), ("bloom", 5),
        ("foot_l", 5), ("foot_r", 5),
    ):
        anchor = body_m if name.startswith("foot") else torso_m
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], anchor, depth), depth,
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
        arm=0.0, arm_mode="up",
        bloom=0.0, crown=0,
        kick=0.0,
        foot_l_ang=0.0, foot_r_ang=0.0,
        step_l=(0, 0), step_r=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay. The crown breathes up by one pixel. Sleeves and the
        # hanging bloom sway. Frame 0 is the still.
        p["arm"] = wave(f, n, 5.0, phase=0.4)
        p["arm_mode"] = "up"
        p["bloom"] = wave(f, n, 7.0, phase=1.1)
        lift = wave(f, n, 1.0, phase=0.3)
        p["crown"] = int(round(max(0.0, lift)))
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        # The shoe only edge-touches the bloom. Sliding it opens a gap, so
        # the sole swings about the ankle and the joint stays put.
        p["foot_l_ang"] = 12.0 * max(0.0, sa)
        p["foot_r_ang"] = 12.0 * max(0.0, sb)
        p["arm"] = wave(f, n, 8.0, phase=0.2)
        p["arm_mode"] = "fwd"
        p["bloom"] = wave(f, n, 6.0, cycles=2.0, phase=0.8)
        p["crown"] = int(round(max(0.0, wave(f, n, 1.0, cycles=2.0, phase=0.5))))
    elif action == "attack":
        # 0 rest. 1-4 raise the crown, the sleeves, and the bloom. No mouth.
        # 5-8 the back, equal scale, cleared after finish().
        # 9 wind-up. 10-12 leaf slash and the lead-shoe kick. 13-15 return.
        arm = [0, 8, 14, 18, 20, 0, 0, 0, 0, 18, 28, 34, 34, 12, 4, 0]
        bloom = [0, 6, 10, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        crown = [0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        kick = [0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 26, 34, 34, 12, 4, 0]
        p["arm"] = float(arm[f])
        p["bloom"] = float(bloom[f])
        p["crown"] = int(crown[f])
        p["kick"] = float(kick[f])
        p["arm_mode"] = "up" if f <= 4 else "fwd"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.64, 0.44, 0.26, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["arm"] = 8.0 * env
        p["arm_mode"] = "up"
        p["bloom"] = 6.0 * env
        p["crown"] = int(round(2 * env))
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["arm"] = 0.0
        p["bloom"] = 0.0
        p["crown"] = 0
        p["kick"] = 0.0
        p["foot_l_ang"] = 0.0
        p["foot_r_ang"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["spark"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["turned"] = True
        p["arm_mode"] = "up"
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


def limb_order(fc):
    # Lead sleeve and lead shoe paste last, so the strike stays in front.
    if fc < 0:
        return ("foot_r", "foot_l", "arm_r", "bloom", "arm_l", "crown")
    return ("foot_l", "foot_r", "arm_l", "bloom", "arm_r", "crown")


def limb_delta(rig, p, name):
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    if name == "foot_l":
        sx, sy = p["step_l"]
        kick = float(p["kick"])
        ang = part_angle(rig, name, "fwd", p["foot_l_ang"]) + part_angle(rig, name, "up", kick)
        fwd = int(round(kick / 12.0))
        lift = int(round(kick / 10.0))
        return ang, bdx + int(round(fc * sx)) + int(round(fc * fwd)), bdy + int(sy) - lift
    if name == "foot_r":
        sx, sy = p["step_r"]
        ang = part_angle(rig, name, "fwd", p["foot_r_ang"])
        return ang, bdx + int(round(fc * sx)), bdy + int(sy)
    if name in ("arm_l", "arm_r"):
        return part_angle(rig, name, p["arm_mode"], p["arm"]), bdx, bdy
    if name == "bloom":
        return part_angle(rig, name, "up", p["bloom"]), bdx, bdy
    if name == "crown":
        return 0.0, bdx, bdy - int(p["crown"])
    return 0.0, bdx, bdy


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    for name in limb_order(fc):
        ang, dx, dy = limb_delta(rig, p, name)
        paste(out, moved(L[name], ang, piv[name], dx, dy))
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


def lean_angle(facing, deg):
    """Head moves opposite facing. Positive rotation is clockwise on screen."""
    return facing * deg


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (YELLOW, WHITE, PINK, LIME)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2), (2, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def forward_tip(rig, p):
    fc = rig["facing"]
    best = None
    for name in ("arm_l", "arm_r"):
        ang, dx, dy = limb_delta(rig, p, name)
        layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
        m = layer[..., 3] >= 128
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
        pt = (float(xs[i]), float(ys[i]))
        if best is None:
            best = pt
        elif fc > 0 and pt[0] > best[0]:
            best = pt
        elif fc < 0 and pt[0] < best[0]:
            best = pt
    if best is None:
        return rig["pivots"]["center"]
    return best


def rig_for(rigs, d, p):
    """Attack 5-8 shows this side's back at equal scale. Back rows stay."""
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", mirror)]


def moving(p):
    if p["step_l"] != (0, 0) or p["step_r"] != (0, 0):
        return True
    if p["bdx"] or p["bdy"] or p["knock"] or p["crown"]:
        return True
    if abs(p["arm"]) >= 0.45 or abs(p["bloom"]) >= 0.45 or abs(p["kick"]) >= 0.45:
        return True
    if abs(p["foot_l_ang"]) >= 0.45 or abs(p["foot_r_ang"]) >= 0.45:
        return True
    if abs(p["lean"]) > 0.4:
        return True
    return False


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        body = rotate_nearest(body, lean_angle(fc, p["lean"]), rear_pivot(body, fc))
    if moving(p) or abs(p["lean"]) > 0.4:
        body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    else:
        img = rig["canvas"].copy()
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
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


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name in LIMBS:
        ang, dx, dy = limb_delta(rig, p, name)
        layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
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

    def matches():
        if not (lock_dir / "palette.json").exists():
            return False
        data = json.loads((lock_dir / "palette.json").read_text())
        got = {tuple(int(c["hex"][i:i + 2], 16) for i in (1, 3, 5)) for c in data["colors"]}
        return got == set(own)

    if not matches():
        py = str(FORGE / ".venv/bin/python")
        hexes = ["#%02x%02x%02x" % c for c in own]
        for d in (pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        r = subprocess_run(
            [py, "-I", str(PALETTE_TOOL), "build", "--input", str(front_path),
             "--colors", str(len(hexes)), "--reserve", *hexes, "--output-dir", str(pal_dir)],
        )
        if r.returncode:
            raise SystemExit("palette build failed: " + r.stdout + r.stderr)
        r = subprocess_run(
            [py, "-I", str(PALETTE_TOOL), "lock", "--palette", str(pal_dir / "palette.json"),
             "--source", str(front_path), "--strict", "--output-dir", str(lock_dir)],
        )
        if r.returncode:
            raise SystemExit("palette lock failed: " + r.stdout + r.stderr)
    data = json.loads((lock_dir / "palette.json").read_text())
    cols = np.array([[int(c["hex"][i:i + 2], 16) for i in (1, 3, 5)] for c in data["colors"]], np.uint8)
    got = {tuple(int(v) for v in c) for c in cols}
    if got != set(own):
        raise SystemExit(f"locked palette differs from the stills: {sorted(got ^ set(own))}")
    return cols


def subprocess_run(cmd):
    import subprocess
    return subprocess.run(cmd, capture_output=True, text=True)


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
        raise SystemExit("this still has no mouth")
    for key in (("front", False), ("front", True), ("back", False)):
        rig = rigs[key]
        fc = rig["facing"]
        shift_x = crown_shift(rig["canvas"], fc, MAX_LEAN)
        if fc > 0 and not (shift_x < -5):
            raise SystemExit(f"hurt lean does not tip the head back ({shift_x})")
        if fc < 0 and not (shift_x > 5):
            raise SystemExit(f"hurt lean does not tip the head back ({shift_x})")
    return front, back, fm, bm, rigs


def crown_shift(img, facing, deg):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    i = int(ys.argmin())
    piv = rear_pivot(img, facing)
    rx = float(xs[i]) - piv[0]
    ry = float(ys[i]) - piv[1]
    ang = lean_angle(facing, deg)
    r = math.radians(ang)
    c, s = math.cos(r), math.sin(r)
    return (piv[0] + c * rx + s * ry) - float(xs[i])


def save_parts(work, front, fm, back, bm):
    cols = {
        "arm_l": (220, 60, 40, 255),
        "arm_r": (255, 180, 40, 255),
        "leg_l": (40, 200, 90, 255),
        "leg_r": (40, 160, 220, 255),
        "crown": (200, 80, 220, 255),
        "bloom": (255, 120, 180, 255),
        "foot_l": (40, 200, 90, 255),
        "foot_r": (40, 160, 220, 255),
        "torso": (170, 170, 170, 255),
    }
    tiles = []
    for masks in (fm, bm):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            if name == "eyes":
                continue
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
        "dex": 549, "form": "regional", "variant": "lilligant-hisui", "gen": "gen5",
        "dir": "monsterSprite/gen5/0549_lilligant/regional/lilligant-hisui",
        "sheetDir": "agent-sprite-forge",
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
    show = lambda masks: {k: int(v.sum()) for k, v in masks.items() if k != "eyes"}
    print("parts front", show(fm), "back", show(bm),
          "palette", len(palette), "foot", FOOT_Y, flush=True)
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
    turn_n = synth(rigs, "N", "attack", 6, palette, effects=False)
    if np.any(turn_w != rigs["N"]["canvas"]):
        raise SystemExit("W turn is not the N back at equal scale")
    if np.any(turn_s != rigs["NW"]["canvas"]):
        raise SystemExit("S turn is not the mirrored back at equal scale")
    if np.any(turn_n != rigs["N"]["canvas"]):
        raise SystemExit("N turn left the back still")

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
            atk.append((f, round(float(p["arm"]), 1), p["arm_mode"], p["turned"],
                        p["spin"], int(p["crown"]), round(float(p["kick"]), 1),
                        round(float(p["bloom"]), 1), p["spark"]))
        print("attack poses", atk, flush=True)
        leans = [round(float(pose("hurt", f)["lean"]), 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        rest = synth(rigs, "W", "attack", 0, palette, effects=False)
        hit = synth(rigs, "W", "attack", 11, palette, effects=False)
        rest_y = int(np.nonzero(rest[..., 3] >= 128)[0].max())
        hit_y = int(np.nonzero(hit[..., 3] >= 128)[0].max())
        print("sole y rest", rest_y, "kick", hit_y, "delta", hit_y - rest_y, flush=True)
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
    print("qa_sheet", "rest_diff", int(sum(rest_diff.values()) + sum(idle0.values())),
          "off_palette", len(off), "partial_alpha", partial, "edge_touch", len(clip),
          "body_main_min", report["body_main_min"], "clip", clip[:8], flush=True)
    meta = {
        "generator": "forge_gen5_lilligant_hisui_actions", "version": VERSION, "key": KEY,
        "dex": 549, "variant": "lilligant-hisui", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped leaf dancer (white face, no mouth, two leaf sleeves, flower crown, hanging bloom, two yellow shoes)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (yellow chin x 39 < eye mean x)",
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
