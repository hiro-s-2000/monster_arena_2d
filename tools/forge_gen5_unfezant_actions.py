#!/usr/bin/env python3
"""Unfezant normal (0521) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x22-78 y8-82, 1766 px, alpha 0/255):
  The two white eye pixels sit at (41,24) and (38,25), mean x 39.5.
  The closed beak is yellow (255,230,0) plus brown (98,74,49), x30-40 y23-28,
  mean x 34. The beak tip is left of the eye, so the still faces screen-left.
  There is no mouth cavity, so the roar does not cut a wedge or draw rings.
  It lifts the pink heart crest this sprite already has.
  The crest is the three pinks (255,65,123), (180,24,90), (123,32,82),
  from the top of the head down the near side of the neck.
  The barred tail is the rear fan, tip x78 y23, rooted near x60.
  Two grey feet split under y63 (screen-left toes end y78, the other toe
  reaches the sole at y82). Folded wings stay on the torso. No wing is drawn
  as its own silhouette, so none is invented.
normal/back.png (bbox x7-68 y8-81, 1780 px, sole y81) has the beak at x60-66
  and the one white eye pixel at x59 y24, so this back faces screen-right.
  The tail streams to the left (tip x7). The back sole is 1px higher, so the
  back is shifted down 1px and both forms share one ground line.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (beak to screen-right). NW back mirrored.
  Attack 5-8 shows that direction's back at equal scale: W turns into N,
  S turns into N flipped. No |cos| squash, no in-plane body spin.
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
SRC_DIR = ROOT / "monsterSprite/gen5/0521_unfezant/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0521_unfezant/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0521_unfezant"
VERSION = "unfezant_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 200, 200
OFF_X = (CELL_W - 96) // 2
FRONT_FOOT = 82
BACK_FOOT = 81
# Front sole is the lower one. Leave about 20px under it. Shift the back down 1px.
OFF_Y = CELL_H - 20 - FRONT_FOOT
FOOT_Y = OFF_Y + FRONT_FOOT
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# The heart is a loop around the skull (radius ~36px). Rotating it opens
# a hole at the crown, so the strike shifts it by whole pixels and swings
# the hanging lobe and the tail. Peaks are the angles this script draws.
PEAKS = {"lobe": 12.0, "tail": 24.0}
LIMBS = ("crest", "tail", "foot_l", "foot_r")

INK = (0, 0, 0)
DK = (41, 41, 41)
BODY = (65, 65, 82)
GREY = (98, 98, 98)
MGREY = (123, 123, 139)
LGREY = (180, 180, 180)
WHITE = (255, 255, 255)
PINK = (255, 65, 123)
DPINK = (180, 24, 90)
WINE = (123, 32, 82)
DGREEN = (49, 115, 74)
TGREEN = (49, 164, 123)
YELLOW = (255, 230, 0)
BROWN = (98, 74, 49)
PINK_COLS = (PINK, DPINK, WINE)
BEAK_COLS = (YELLOW, BROWN)
GREEN_COLS = (DGREEN, TGREEN)

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
    if not out.any():
        return out
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
    """4x nearest in-plane rotation. Positive follows the forge formula. No squash."""
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
        raise SystemExit(f"parts do not partition: {int(acc.sum())} vs {int(op.sum())}")


def mask_from(shape, pts):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m


def segment(img, kind):
    """Crest (pink heart, including its own fill), barred tail, two feet.

    The closed beak and the white eye stay on the torso. Interior pixels of
    the heart that pink fully separates from the belly move with the crest,
    so the heart does not leave its grey fill behind. The beak and the eye
    pocket stay, so the roar does not invent a mouth or slide the face.
    """
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    pink = any_color(img, PINK_COLS)
    beak = any_color(img, BEAK_COLS)
    green = any_color(img, GREEN_COLS)
    eye = is_color(img, WHITE)
    ink = is_color(img, INK)
    if kind == "front":
        if int(np.nonzero(op)[0].max()) != FRONT_FOOT:
            raise SystemExit("front sole moved")
        seed = op & (xx >= 74) & (yy <= 30)
        allowed = op & ~pink & ~beak & ~green & (yy <= 52) & (xx >= 60)
        foot_cut = 63
    else:
        if int(np.nonzero(op)[0].max()) != BACK_FOOT:
            raise SystemExit("back sole moved")
        seed = op & (xx <= 12) & (yy >= 36) & (yy <= 50)
        allowed = op & ~pink & ~green & ~beak & (yy <= 56) & (xx <= 36) & (yy >= 34)
        foot_cut = 64
    tail = flood(allowed, seed)
    if not tail.any() or len(component_list(tail)) != 1:
        raise SystemExit(f"{kind} tail did not come out in one piece")
    feet_all = op & (yy >= foot_cut) & ~tail
    fgroups = component_list(feet_all)
    if len(fgroups) != 2:
        raise SystemExit(f"{kind} expected 2 feet, got {len(fgroups)}")
    fgroups.sort(key=lambda g: sum(x for _y, x in g) / len(g))
    foot_l = mask_from(op.shape, fgroups[0])
    foot_r = mask_from(op.shape, fgroups[1])
    free = op & ~tail & ~feet_all
    ggroups = component_list(green & free)
    if not ggroups:
        raise SystemExit(f"{kind} belly is missing")
    reachable = flood(free & ~pink, mask_from(op.shape, ggroups[0]))
    if not eye.any():
        raise SystemExit(f"{kind} eye is missing")
    eye_blob = flood(free & ~pink, eye)
    holes = free & ~pink & ~reachable & ~eye_blob & ~beak
    crest_core = (pink & free) | holes
    # Outline that belongs to the heart, not the shared edge against the body.
    body_keep = reachable | eye_blob | (beak & free)
    crest_ink = ink & free & dilate(crest_core, 1) & ~dilate(body_keep, 1)
    crest = crest_core | crest_ink
    torso = op & ~crest & ~tail & ~foot_l & ~foot_r
    if (eye & ~torso).any() or (beak & ~torso).any():
        raise SystemExit(f"{kind} face left the torso")
    masks = {
        "crest": crest, "tail": tail, "foot_l": foot_l, "foot_r": foot_r, "torso": torso,
    }
    check_partition(op, masks)
    for name in LIMBS:
        part = masks[name]
        if int(part.sum()) < 8:
            raise SystemExit(f"{kind} {name} is too small ({int(part.sum())})")
        if not (part & dilate(torso, 2)).any():
            raise SystemExit(f"{kind} {name} does not touch the body")
    if kind == "front":
        ey, ex = np.nonzero(eye)
        by, bx = np.nonzero(beak)
        if float(bx.min()) >= float(ex.mean()) - 1.0:
            raise SystemExit(
                f"front does not face left: beak {int(bx.min())} eye {float(ex.mean()):.1f}"
            )
    else:
        ey, ex = np.nonzero(eye)
        by, bx = np.nonzero(beak)
        if float(bx.max()) <= float(ex.mean()) + 1.0:
            raise SystemExit(
                f"back beak is not right of the eye: beak {int(bx.max())} eye {float(ex.mean()):.1f}"
            )
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


def crest_pieces(layer, torso_layer, facing):
    """One pivot per heart blob, so the side drape does not orbit the skull."""
    mask = layer[..., 3] >= 128
    groups = component_list(mask)
    groups.sort(key=len, reverse=True)
    large = [g for g in groups if len(g) >= 12]
    small = [g for g in groups if len(g) < 12]
    if not large:
        large = groups[:1]
        small = groups[1:]
    labels = {}
    for i, g in enumerate(large):
        for y, x in g:
            labels[(y, x)] = i
    left = []
    for g in small:
        left.extend(g)
    guard = 0
    while left and guard < 48:
        guard += 1
        nxt = []
        progress = False
        for y, x in left:
            found = None
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                found = labels.get((y + dy, x + dx))
                if found is not None:
                    break
            if found is None:
                nxt.append((y, x))
            else:
                labels[(y, x)] = found
                progress = True
        if not progress:
            for y, x in nxt:
                labels[(y, x)] = 0
            left = []
            break
        left = nxt
    pieces = []
    for i in range(len(large)):
        sub = np.zeros_like(layer)
        for (y, x), lab in labels.items():
            if lab == i:
                sub[y, x] = layer[y, x]
        if int((sub[..., 3] >= 128).sum()) < 4:
            continue
        piv = pivot_of(sub, torso_layer)
        pieces.append({
            "layer": sub,
            "pivot": piv,
            "signs": signs_for(sub, piv, facing),
            "n": int((sub[..., 3] >= 128).sum()),
        })
    if not pieces:
        raise SystemExit("crest pieces empty")
    return pieces


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
    for name, depth in (("crest", 5), ("tail", 4), ("foot_l", 4), ("foot_r", 4)):
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth + 1,
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
        # Source-left foot is toward the beak. Mirroring keeps it toward facing.
        lead, rear = "foot_l", "foot_r"
    else:
        facing = -1.0 if mirror else 1.0
        # Back beak is on the source-right. That foot leads.
        lead, rear = "foot_r", "foot_l"
    pieces = crest_pieces(layers["crest"], layers["torso"], facing)
    pivots = {}
    signs = {}
    for name in ("tail", "foot_l", "foot_r"):
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    pivots["crest"] = pieces[0]["pivot"]
    signs["crest"] = pieces[0]["signs"]
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing,
        "lead": lead, "rear": rear,
        "layers": layers, "pieces": pieces,
        "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        crest=0.0, crest_mode="up",
        crest_dx=0, crest_dy=0,
        lobe=0.0, lobe_mode="up",
        tail=0.0, tail_mode="up",
        step_l=(0, 0), step_r=(0, 0),
        kick=(0, 0),
        lean=0.0, knock=0, flash=False,
        spark=0, turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The heart breathes; the hanging lobe and the tail sway.
        p["crest"] = wave(f, n, 3.0, phase=0.5)
        p["lobe"] = wave(f, n, 6.0, phase=0.8)
        p["tail"] = wave(f, n, 7.0, phase=1.0)
        p["crest_mode"] = "up"
        p["lobe_mode"] = "up"
        p["tail_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * sa)), -int(round(2 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * sb)), -int(round(2 * max(0.0, sb))))
        p["crest"] = wave(f, n, 2.5, phase=0.4)
        p["lobe"] = wave(f, n, 5.0, phase=0.3)
        p["tail"] = wave(f, n, 8.0, cycles=2.0, phase=0.6)
        p["crest_mode"] = "up"
        p["lobe_mode"] = "fwd"
        p["tail_mode"] = "fwd"
    elif action == "attack":
        # 0 rest. 1-4 the heart shifts up and the hanging lobe rises.
        # No mouth, so no wedge and no ring. 5-8 the back, equal scale.
        # 9-12 the lobe and the tail strike and the lead foot steps in.
        # 13-15 return. The heart is not spun. Drawn peaks: lobe 12, tail 24.
        lobe = [0, 4, 8, 10, 10, 0, 0, 0, 0, 6, 10, 12, 12, 4, 1, 0]
        tail = [0, 4, 8, 12, 12, 0, 0, 0, 0, 10, 18, 24, 24, 10, 3, 0]
        bdx = [0, -1, -1, -1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        crest_dy = [0, -1, -2, -2, -2, 0, 0, 0, 0, -1, -1, -2, -1, 0, 0, 0]
        crest_dx = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        p["crest"] = 0.0
        p["lobe"] = float(lobe[f])
        p["tail"] = float(tail[f])
        p["bdx"] = int(bdx[f])
        p["crest_dy"] = int(crest_dy[f])
        p["crest_dx"] = int(crest_dx[f])
        p["crest_mode"] = "up"
        p["lobe_mode"] = "up" if f <= 4 else "fwd"
        p["tail_mode"] = "up" if f <= 4 else "fwd"
        p["turned"] = 5 <= f <= 8
        if f in (10, 11, 12):
            p["kick"] = {10: (2, -2), 11: (3, -1), 12: (3, 0)}[f]
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["crest"] = 0.0
        p["lobe"] = 8.0 * env
        p["crest_dy"] = -1 if env > 0.5 else 0
        p["tail"] = 8.0 * env
        p["crest_mode"] = "up"
        p["lobe_mode"] = "up"
        p["tail_mode"] = "up"
        p["flash"] = f == 1
        p["spark"] = 1 if f in (1, 2) else 0
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["crest"] = 0.0
        p["lobe"] = 0.0
        p["crest_dx"] = 0
        p["crest_dy"] = 0
        p["tail"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["kick"] = (0, 0)
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


def piece_angle(signs, mode, amp):
    if abs(amp) < 0.45:
        return 0.0
    if mode == "up":
        return signs["up"] * amp
    return signs["fwd"] * amp


def moved(layer, ang, pivot, dx, dy):
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, pivot)
    return shift(layer, dx, dy)


def limb_delta(rig, p):
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    step = {"foot_l": p["step_l"], "foot_r": p["step_r"]}
    delta = {}
    for name in LIMBS:
        if name in step:
            sx, sy = step[name]
            kx, ky = p["kick"] if name == rig["lead"] else (0, 0)
            delta[name] = (bdx + int(round(fc * (sx + kx))), bdy + int(sy + ky))
        else:
            delta[name] = (bdx, bdy)
    return delta


def compose(rig, p):
    L = rig["layers"]
    delta = limb_delta(rig, p)
    out = np.zeros_like(rig["canvas"])
    bdx, bdy = delta["crest"]
    paste(out, shift(L["torso"], bdx, bdy))
    tdx, tdy = delta["tail"]
    paste(out, moved(L["tail"], part_angle(rig, "tail", p["tail_mode"], p["tail"]), rig["pivots"]["tail"], tdx, tdy))
    for name in (rig["rear"], rig["lead"]):
        dx, dy = delta[name]
        paste(out, shift(L[name], dx, dy))
    cdx = int(round(rig["facing"] * p["crest_dx"]))
    cdy = int(p["crest_dy"])
    for i, piece in enumerate(rig["pieces"]):
        if i == 0:
            ang = piece_angle(piece["signs"], p["crest_mode"], p["crest"])
        else:
            ang = piece_angle(piece["signs"], p["lobe_mode"], p["lobe"])
        paste(out, moved(piece["layer"], ang, piece["pivot"], bdx + cdx, bdy + cdy))
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


def spark(img, origin, fc, age):
    ox, oy = origin
    cols = (PINK, WHITE, YELLOW)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -1), (2, 1))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def limb_lost(rig, p):
    body = compose(rig, p)
    delta = limb_delta(rig, p)
    lost = []
    dx, dy = delta["crest"]
    cdx = int(round(rig["facing"] * p["crest_dx"]))
    cdy = int(p["crest_dy"])
    crest = np.zeros_like(body)
    for i, piece in enumerate(rig["pieces"]):
        if i == 0:
            ang = piece_angle(piece["signs"], p["crest_mode"], p["crest"])
        else:
            ang = piece_angle(piece["signs"], p["lobe_mode"], p["lobe"])
        paste(crest, moved(piece["layer"], ang, piece["pivot"], dx + cdx, dy + cdy))
    named = {"crest": crest}
    tdx, tdy = delta["tail"]
    named["tail"] = moved(
        rig["layers"]["tail"], part_angle(rig, "tail", p["tail_mode"], p["tail"]),
        rig["pivots"]["tail"], tdx, tdy,
    )
    for name in ("foot_l", "foot_r"):
        fx, fy = delta[name]
        named[name] = shift(rig["layers"][name], fx, fy)
    for name, layer in named.items():
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


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
    moved_part = (
        abs(p["crest"]) >= 0.45 or abs(p["lobe"]) >= 0.45 or abs(p["tail"]) >= 0.45
        or p["crest_dx"] or p["crest_dy"]
        or p["step_l"] != (0, 0) or p["step_r"] != (0, 0)
        or p["kick"] != (0, 0) or p["bdx"] or p["bdy"]
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
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    if p["spark"] and action == "hurt":
        cx, cy = rig["pivots"]["center"]
        spark(img, (cx - fc * 6, cy - 8), -fc, p["spark"])
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


def content_box(canvases, pad=16):
    acc = np.zeros(canvases[0].shape[:2], bool)
    for c in canvases:
        acc |= c[..., 3] >= 128
    ys, xs = np.nonzero(acc)
    y0 = max(0, int(ys.min()) - pad - 36)
    y1 = min(CELL_H, int(ys.max()) + pad + 24)
    x0 = max(0, int(xs.min()) - pad - 40)
    x1 = min(CELL_W, int(xs.max()) + pad + 40)
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
    fm = segment(front, "front")
    bm = segment(back, "back")
    rigs = {
        ("front", False): build_view(front, fm, "front", False),
        ("front", True): build_view(front, fm, "front", True),
        ("back", False): build_view(back, bm, "back", False),
        ("back", True): build_view(back, bm, "back", True),
    }
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "torso": (70, 130, 210, 255),
        "crest": (240, 70, 150, 255),
        "tail": (220, 50, 50, 255),
        "foot_l": (250, 250, 250, 255),
        "foot_r": (255, 210, 40, 255),
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
        "dex": 521, "form": "normal", "variant": "unfezant", "gen": "gen5",
        "dir": "monsterSprite/gen5/0521_unfezant/normal", "sheetDir": "agent-sprite-forge",
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
    palette = locked_palette(SRC_DIR / "front.png", SRC_DIR / "back.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    print("lead", {d: rigs[d]["lead"] for d in DIRS}, flush=True)
    print("crest pieces", {d: [p["n"] for p in rigs[d]["pieces"]] for d in ("S", "W", "N")}, flush=True)

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
        clip = []
        for d in ("S", "W", "N", "E"):
            for act, fs in (
                ("idle", (0, 8)),
                ("walk", (4, 12)),
                ("attack", (2, 4, 6, 8, 11)),
                ("hurt", (3,)),
            ):
                for f in fs:
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost[f"{act}:{d}:{f}"] = bad
                    t = synth(rigs, d, act, f, palette)
                    a = t[..., 3] == 255
                    if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                        clip.append(f"{act}:{d}:{f}")
        print("limb lost", lost or "none", flush=True)
        print("sample clip", clip or "none", flush=True)
        box = content_box([rigs[d]["canvas"] for d in DIRS])
        y0, y1, x0, x1 = box
        tiles = []
        for d in ("S", "W", "N", "E"):
            for act, f in (
                ("idle", 0), ("idle", 8),
                ("walk", 4), ("walk", 12),
                ("attack", 0), ("attack", 3), ("attack", 6), ("attack", 8),
                ("attack", 11), ("attack", 14),
                ("hurt", 1), ("hurt", 3),
            ):
                t = synth(rigs, d, act, f, palette)
                tiles.append(t[y0:y1, x0:x1])
        sheet = np.repeat(np.repeat(tile_grid(tiles, 12), 2, 0), 2, 1)
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
        and partial == 0 and not clip and min(bodies) >= 0.90
    )
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_unfezant_actions", "version": VERSION, "key": KEY,
        "dex": 521, "variant": "unfezant", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped bird (pink heart crest, closed yellow beak, barred tail, two feet)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (beak x30 < eye mean x39.5); tail trails screen-right",
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
