#!/usr/bin/env python3
"""Monferno (0391) 8-direction action sheet from the measured 96x96 front still.

Archetype: front-facing biped primate. Two arms, two legs, one tail whose
existing flame sits screen-right of the still. No extra limbs and no invented
mouth — the red mark is a nose. Roar flares that tail flame and draws rings.
The still is a camera-front view (nose inside the eye mask). S SE E NE keep
it; N NW W SW are the horizontal mirror.

  idle   : feet planted. Tail and flame sway. Frame 0 is the still.
  walk   : two-step biped. One foot stays planted while the other swings.
  attack : 0-4 flame roar + rings, 5-10 in-place 360 spin (one piece,
           no X squash), 11-12 impact, 13-15 recover.
  hurt   : rigid recoil, 38 degrees max, no belly-up flip. 12 frames.

Palette-locked. Alpha is only 0 or 255. front/back png and gif are not written.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SPECIES_DIR = ROOT / "monsterSprite/gen4/0391_monferno/normal"
OUT_DIR = SPECIES_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0391_monferno/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
FACE_RIGHT = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 176, 148
VERSION = "monferno_v2"

# Measured on front.png (alpha >= 128). Effects may use only these.
INK = (16, 16, 16)
WHITE = (255, 255, 255)
FLAME_Y = (255, 205, 65)
FLAME_R = (205, 65, 24)
DARK = (106, 57, 8)


def flood(seeds, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow)
    q = deque()
    ys, xs = np.nonzero(seeds & allow)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if not seen[y, x]:
            seen[y, x] = True
            q.append((int(y), int(x)))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and allow[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def segment(img):
    """Partition every opaque pixel. Cuts are from this still, not another species."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    rgb = img[..., :3]
    yy, xx = np.mgrid[0:h, 0:w]
    flame_col = (
        ((rgb[..., 0] == 255) & (rgb[..., 1] == 205) & (rgb[..., 2] == 65))
        | ((rgb[..., 0] == 205) & (rgb[..., 1] == 65) & (rgb[..., 2] == 24))
        | ((rgb[..., 0] == 238) & (rgb[..., 1] == 164) & (rgb[..., 2] == 65))
    )
    # Upper-right flame only. The red nose (x~40) and the arm's yellow band stay put.
    tail_flame = flood(a & flame_col & (xx >= 60) & (yy <= 42), a & flame_col & (xx >= 58))
    head = a & (yy <= 41) & (xx <= 58)
    # Thighs split at the crotch. Hips above y=58 stay on the torso.
    left_leg = a & (yy >= 58) & (xx < 47)
    right_leg = a & (yy >= 58) & (xx >= 47)
    pad = np.pad(tail_flame, 1)
    adj = (pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]) & a & ~tail_flame
    tail_allow = a & ~head & ~tail_flame & ~left_leg & ~right_leg & (yy <= 49) & (xx >= 56)
    tail = flood(adj, tail_allow)

    used = head | tail_flame | tail | left_leg | right_leg
    seeds = np.zeros_like(a)
    ys, xs = np.nonzero(a & (xx < 30) & (yy > 45) & (yy < 58))
    seeds[int(ys[np.argmin(xs)]), int(xs.min())] = True
    left_arm = flood(seeds, a & ~used & (xx <= 44) & (yy >= 44) & (yy <= 57))

    used = used | left_arm
    cream = a & (rgb[..., 0] == 238) & (rgb[..., 1] == 205) & (yy >= 49) & (xx >= 60)
    cys, cxs = np.nonzero(cream)
    seeds = np.zeros_like(a)
    seeds[int(cys[np.argmax(cxs)]), int(cxs.max())] = True
    right_arm = flood(seeds, a & ~used & (yy >= 48) & (xx >= 48))

    torso = a & ~(head | tail_flame | tail | left_leg | right_leg | left_arm | right_arm)
    parts = {
        "head": head, "torso": torso, "tail": tail, "flame": tail_flame,
        "left_arm": left_arm, "right_arm": right_arm,
        "left_leg": left_leg, "right_leg": right_leg,
    }
    cover = np.zeros_like(a)
    for m in parts.values():
        if np.any(cover & m):
            raise SystemExit("monferno parts overlap")
        cover |= m
    if int((a & ~cover).sum()) or int(cover.sum()) != int(a.sum()):
        raise SystemExit("monferno parts do not cover the still")
    return parts


def shift(layer, dx, dy):
    dx_i, dy_i = int(round(dx)), int(round(dy))
    if dx_i == 0 and dy_i == 0:
        return layer.copy()
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0, sx1 = max(0, -dx_i), min(w, w - dx_i)
    sy0, sy1 = max(0, -dy_i), min(h, h - dy_i)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy_i:sy1 + dy_i, sx0 + dx_i:sx1 + dx_i] = layer[sy0:sy1, sx0:sx1]
    return out


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest. Pivot is the joint. A full turn lands on the same pixels.

    Width is never scaled. 0 and 360 leave the bitmap untouched.
    """
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    up = np.repeat(np.repeat(layer, 4, axis=0), 4, axis=1)
    px, py = pivot[0] * 4 + 1.5, pivot[1] * 4 + 1.5
    rad = math.radians(ang)
    c, s = math.cos(rad), math.sin(rad)
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = xx - px, yy - py
    src_x = np.rint(px + c * rx + s * ry).astype(np.int32)
    src_y = np.rint(py - s * rx + c * ry).astype(np.int32)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    out[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    return out[1::4, 1::4].copy()


def content_point(x, y, angle_deg, pivot):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    vx, vy = x - pivot[0], y - pivot[1]
    return pivot[0] + c * vx - s * vy, pivot[1] + s * vx + c * vy


def seal_cracks(img):
    """Fill a 1px rotation crack that already has 3 opaque neighbours.

    Straight gaps (toes, the crotch) have only two neighbours and stay open.
    Colour is the nearest opaque pixel, then palette-snapped by the caller.
    """
    a = img[..., 3] >= 128
    if not np.any(a):
        return img
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    hole = (~a) & (n >= 3)
    if not np.any(hole):
        return img
    ys, xs = np.nonzero(a)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = img[ys, xs]
    ty, tx = np.nonzero(hole)
    pick = np.empty(len(ty), np.int32)
    for i in range(0, len(ty), 400):
        sl = slice(i, i + 400)
        d = (ty[sl, None].astype(np.int32) - pts[None, :, 0]) ** 2 + (tx[sl, None].astype(np.int32) - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    out = img.copy()
    out[ty, tx] = cols[pick]
    return out


def despeckle(img):
    """Drop only pixels whose 4-neighbours are all empty. Outline stays."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    step = 400
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None].astype(np.int32) - pts[None, :, 0]) ** 2 + (tx[sl, None].astype(np.int32) - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = a.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    mapped = palette[dist.argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    return out


def put_empty(out, x, y, col):
    """Effects land only on empty pixels, and only in palette colours."""
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def socket_pivot(part, body, rad=5):
    sel = part & dilate(body, rad)
    if not np.any(sel):
        sel = part
    ys, xs = np.nonzero(sel)
    return float(xs.mean()), float(ys.mean())


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def build_rig(img, masks, flip):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]

    placed = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm

    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr

    # Joint colour lives under the limb. At rest the limb covers it, so the still matches.
    torso_m = placed["torso"]
    for part_name, rad in (("left_arm", 6), ("right_arm", 6), ("left_leg", 5), ("right_leg", 5), ("tail", 5), ("flame", 3)):
        zone = placed[part_name] & dilate(torso_m | placed["tail"], rad)
        if part_name == "flame":
            zone = placed["flame"] & dilate(placed["tail"], rad)
            donor = placed["tail"]
        elif part_name == "tail":
            donor = torso_m
        else:
            donor = torso_m
        if np.any(zone) and np.any(donor):
            fill = nearest_color(canvas, zone, donor)
            empty = layers["torso"][..., 3] < 128
            write = zone & empty
            layers["torso"][write] = fill[write]

    pivots = {
        "left_arm": socket_pivot(placed["left_arm"], placed["torso"]),
        "right_arm": socket_pivot(placed["right_arm"], placed["torso"]),
        "left_leg": socket_pivot(placed["left_leg"], placed["torso"]),
        "right_leg": socket_pivot(placed["right_leg"], placed["torso"]),
        "tail": socket_pivot(placed["tail"], placed["torso"]),
        "flame": socket_pivot(placed["flame"], placed["tail"], rad=3),
    }
    nys, nxs = np.nonzero(placed["head"] & (img.shape[0] > 0))
    # Nose: the red mark inside the mask. Rings start here. It is not a jaw.
    rgb = canvas[..., :3]
    nose_m = placed["head"] & (rgb[..., 0] == FLAME_R[0]) & (rgb[..., 1] == FLAME_R[1]) & (rgb[..., 2] == FLAME_R[2])
    if np.any(nose_m):
        ny, nx = np.nonzero(nose_m)
        nose = (float(nx.mean()), float(ny.mean()))
    else:
        hy, hx = np.nonzero(placed["head"])
        nose = (float(np.median(hx)), float(np.median(hy)))

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        nose = (CELL_W - 1 - nose[0], nose[1])
    facing = -1.0 if flip else 1.0
    return layers, pivots, nose, facing


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0, tilt=0.0, spin=0.0,
        left_arm=0.0, right_arm=0.0, left_leg=0.0, right_leg=0.0,
        tail=0.0, flame=0.0, flare=0,
        ring=-1, rings=0, lines=0, burst=0, flash=0, sparks=0,
        leg_lift=0.0,
    )
    if action == "idle":
        # Feet stay down. The living part is the tail flame.
        p["tail"] = wave(f, n, 6.0, phase=0.4)
        p["flame"] = wave(f, n, 9.0, phase=-0.8, cycles=2.0)
        p["left_arm"] = wave(f, n, 3.0, phase=0.2)
        p["right_arm"] = wave(f, n, -3.0, phase=0.2)
    elif action == "walk":
        # Two steps. Positive sine lifts the left leg; the right leg plants, then they swap.
        s = math.sin(2 * math.pi * f / n)
        swing = 32.0
        p["left_leg"] = swing * max(0.0, s)
        p["right_leg"] = swing * max(0.0, -s)
        p["left_arm"] = 18.0 * max(0.0, -s)
        p["right_arm"] = -18.0 * max(0.0, s)
        p["tail"] = -8.0 * s
        p["flame"] = wave(f, n, 8.0, cycles=2.0)
        p["leg_lift"] = 3.0
        p["dx"] = wave(f, n, 1.0, cycles=2.0)
    elif action == "attack":
        # 0-4 roar, 5-10 spin, 11-12 impact, 13-15 recover. Frame 0 is the still.
        p["flame"] = [0, 12, 20, 26, 14, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0][f]
        p["flare"] = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["tail"] = [0, 4, 7, 8, 4, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0][f]
        raise_a = [0, 8, 12, 14, 8, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0][f]
        p["left_arm"] = raise_a
        p["right_arm"] = -raise_a
        if 1 <= f <= 4:
            p["ring"] = f - 1
            p["rings"] = 3
        if 5 <= f <= 10:
            # Rigid in-plane turn. Frame 10 is 360 and matches the still.
            # Do not scale X. A vertical-axis squash was slicing the body into strips.
            p["spin"] = 60.0 * (f - 4)
            p["lines"] = 4
        if f in (11, 12):
            p["burst"] = 1
            p["dx"] = 2.0
    else:
        # 0 at frame 0 and frame 11. Peak is sin(pi*f/11), under 38 degrees. No belly flip.
        env = math.sin(math.pi * f / 11.0)
        p["tilt"] = 38.0 * env
        p["dx"] = -6.0 * env
        p["dy"] = -2.0 * env
        p["tail"] = 10.0 * env
        p["flame"] = -8.0 * env
        p["left_arm"] = -10.0 * env
        p["right_arm"] = 8.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def moved_part(layer, ang, pivot, dx, dy):
    return shift(rotate_nearest(layer, ang, pivot), dx, dy)


def synth(rig, action, f, palette):
    layers, pivots, nose, facing = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    sgn = facing  # +1 keeps the still's screen-right tail
    bdx, bdy = p["dx"] * sgn, p["dy"]
    spinning = p["spin"] > 0.2

    # Positive angle raises a hand that sits to the left of its shoulder.
    # Mirror the sign with the facing so the flipped sheet swings the same limb.
    def ang(v):
        return v * sgn

    left_arm = moved_part(layers["left_arm"], ang(p["left_arm"]), pivots["left_arm"], bdx, bdy)
    right_arm = moved_part(layers["right_arm"], ang(p["right_arm"]), pivots["right_arm"], bdx, bdy)

    def leg(name, swing):
        lift = 0.0
        if p["leg_lift"] and swing > 0.5:
            lift = -p["leg_lift"] * (swing / 32.0)
        # Forward is the facing direction. Positive image rotation throws a foot left,
        # so the facing sign turns the swing into a forward step.
        return moved_part(layers[name], ang(-swing), pivots[name], bdx, bdy + lift)

    left_leg = leg("left_leg", p["left_leg"])
    right_leg = leg("right_leg", p["right_leg"])

    tail_ang = ang(p["tail"])
    tail = moved_part(layers["tail"], tail_ang, pivots["tail"], bdx, bdy)
    flame = rotate_nearest(layers["flame"], tail_ang, pivots["tail"])
    fx, fy = content_point(pivots["flame"][0], pivots["flame"][1], tail_ang, pivots["tail"])
    flame = rotate_nearest(flame, ang(p["flame"]), (fx, fy))
    flame = shift(flame, bdx, bdy)
    # Elongate the same flame along its axis. Not a second tail.
    flare_dx = p["flare"] * sgn
    flare_dy = -p["flare"]
    flame_tip = shift(flame, flare_dx, flare_dy) if p["flare"] else None

    torso = shift(layers["torso"], bdx, bdy)
    head = shift(layers["head"], bdx, bdy)

    # Torso first so the joint cushion stays under the limbs. At rest the limbs cover it.
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for lyr in (torso, tail, left_leg, right_leg, left_arm, right_arm, flame):
        paste(body, lyr)
    if flame_tip is not None:
        paste(body, flame_tip)
    paste(body, head)

    rotated = any(abs(p[k]) >= 0.45 for k in ("left_arm", "right_arm", "left_leg", "right_leg", "tail", "flame"))
    if rotated and not spinning:
        body = despeckle(body)

    bb = body_box(body)
    center = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0) if bb else (CELL_W / 2.0, FOOT_Y - 26.0)

    if spinning:
        # Whole bitmap, one piece. Pivot is the body centre so the turn stays in place.
        spun = rotate_nearest(body, p["spin"], center)
        if np.array_equal(spun, body):
            body = spun
        else:
            body = despeckle(seal_cracks(spun))
    elif abs(p["tilt"]) >= 0.45:
        # Whole still leans as one bitmap so the short neck is not cut.
        # Sign: facing right recoils the head backward (screen-left).
        body = rotate_nearest(body, -p["tilt"] * sgn, center)
        body = despeckle(body)

    img = snap(body, palette)
    if p["flash"]:
        img = flash_white(img)
    if p["rings"] and p["ring"] >= 0 and not spinning:
        nx = nose[0] + bdx
        ny = nose[1] + bdy
        img = roar_rings(img, (nx, ny), sgn, p["ring"], p["rings"])
    # Speed lines belonged to the in-plane spin. They are not part of the punch.
    if p["lines"] and spinning:
        img = speed_lines(img, sgn, p["lines"], seed=f * 17 + 3)
    if p["burst"]:
        bb2 = body_box(img)
        if bb2:
            tip_x = bb2[2] + 6 if sgn > 0 else bb2[0] - 6
            img = impact_burst(img, (tip_x, (bb2[1] + bb2[3]) // 2))
    if p["sparks"]:
        img = sparks(img, f, sgn)
    return snap(img, palette)


def roar_rings(out, mouth, facing, age, strength):
    """Small arcs in front of the nose. Kept off the tail flame."""
    mx, my = mouth
    cols = (WHITE, FLAME_Y, FLAME_R)
    for r_i in range(strength):
        rad = 4 + age * 2 + r_i * 3
        col = cols[min(r_i, 2)]
        for deg in range(-50, 51, 10):
            ang = math.radians(deg)
            put_empty(out, mx + facing * rad * math.cos(ang) * 0.40, my - 3 - rad * 0.25 + rad * math.sin(ang) * 0.35, col)
    return out


def speed_lines(out, facing, n, seed):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    rear = x0 - 2 if facing > 0 else x1 + 2
    for k in range(n):
        y = int(rng.integers(y0 + 4, max(y0 + 5, y1 - 4)))
        length = int(rng.integers(6, 11))
        for i in range(length):
            put_empty(out, rear - facing * i, y, WHITE if i < 2 else FLAME_Y)
    return out


def impact_burst(out, center):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4
        for r in range(4, 8):
            put_empty(out, cx + r * math.cos(ang), cy + r * math.sin(ang), WHITE if r < 9 else FLAME_Y)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put_empty(out, cx + dx, cy + dy, WHITE)
    return out


def sparks(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    cx = (x0 + x1) // 2
    d = frame_idx
    pts = (
        (cx - facing * (8 + d), y0 - d),
        (cx - facing * (3 + d), y0 - 2 - d),
        (cx + facing * (4 + d), y0 - d),
    )
    for sx, sy in pts:
        put_empty(out, sx, sy, WHITE)
        put_empty(out, sx - facing, sy + 1, FLAME_R)
    return out


def flash_white(img):
    """Hit flash. Ink and the brown outline stay so the silhouette does not blank out."""
    out = img.copy()
    a = out[..., 3] >= 128
    rgb = out[..., :3].astype(np.int16)
    ink = (rgb[..., 0] < 40) & (rgb[..., 1] < 40) & (rgb[..., 2] < 40)
    brown = (rgb[..., 0] < 180) & (rgb[..., 1] < 120) & (rgb[..., 2] < 50) & (rgb[..., 0] > rgb[..., 2] + 30)
    keep = a & (ink | brown)
    out[a & ~keep, :3] = WHITE
    return out


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (SPECIES_DIR / "front.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def assemble(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png"))
                sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = tile
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
    Image.fromarray(master, "RGBA").save(OUT_DIR / "all_8dir_actions.png")
    return master


def preview_image(sheets):
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (2, 4, 8, 12),
        "attack": (3, 6, 8, 11),
        "hurt": (1, 4, 6, 9),
    }
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            x = si * CELL_W
            y = ai * CELL_H
            m = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 6
    crops = []
    for act in ACTIONS:
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            a = tile[..., 3] >= 128
            if not np.any(a):
                continue
            ys, xs = np.nonzero(a)
            y0, y1 = max(0, int(ys.min()) - pad), min(CELL_H, int(ys.max()) + 1 + pad)
            x0, x1 = max(0, int(xs.min()) - pad), min(CELL_W, int(xs.max()) + 1 + pad)
            row.append(tile[y0:y1, x0:x1])
        crops.append(row)
    cell_h = max(c.shape[0] for row in crops for c in row)
    cell_w = max(c.shape[1] for row in crops for c in row)
    cols = max(len(row) for row in crops)
    bg = np.full((len(crops) * cell_h, cols * cell_w, 4), (15, 20, 31, 255), np.uint8)
    for ri, row in enumerate(crops):
        for ci, c in enumerate(row):
            y = ri * cell_h + (cell_h - c.shape[0]) // 2
            x = ci * cell_w + (cell_w - c.shape[1]) // 2
            m = c[..., 3] >= 128
            bg[y:y + c.shape[0], x:x + c.shape[1]][m] = c[m]
    return Image.fromarray(bg, "RGBA")


def main_component_fraction(tile):
    a = tile[..., 3] >= 128
    if not np.any(a):
        return 1.0
    seen = np.zeros_like(a)
    h, w = a.shape
    best = 0
    total = int(a.sum())
    ys, xs = np.nonzero(a)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        q = deque([(y, x)])
        seen[y, x] = True
        n = 0
        while q:
            cy, cx = q.popleft()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        best = max(best, n)
    return best / total


def sync_outputs():
    """Copy finished sheets only. _work stays on the revenge side."""
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SPECIES_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), flush=True)

    rigs = {d: build_rig(front, masks, flip=d not in FACE_RIGHT) for d in DIRS}

    rest = synth(rigs["S"], "idle", 0, palette)
    a = front[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    placed = np.zeros_like(rest)
    dst_x = (CELL_W - (x1 - x0)) // 2
    dst_y = FOOT_Y - (y1 - y0)
    placed[dst_y:dst_y + (y1 - y0), dst_x:dst_x + (x1 - x0)] = front[y0:y1, x0:x1]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (190, 190, 198), "head": (240, 200, 40), "flame": (255, 60, 40),
        "tail": (40, 90, 220), "left_arm": (40, 180, 80), "right_arm": (180, 40, 180),
        "left_leg": (40, 200, 200), "right_leg": (220, 140, 40),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", "S", f) for f in (0, 4, 8, 12)]
        want += [("walk", "S", f) for f in (0, 2, 4, 8, 12)]
        want += [("attack", "S", f) for f in (0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13)]
        want += [("hurt", "S", f) for f in (0, 1, 4, 6, 11)]
        want += [("walk", "W", 4), ("attack", "W", 3)]
        tiles = [synth(rigs[d], act, f, palette) for act, d, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
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
                if fp.exists() and fp.stat().st_size > 0:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview_image(sheets).save(OUT_DIR / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")
    actions = OUT_DIR / "actions"
    actions.mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", actions / "preview.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 52

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = np.unique(master[..., 3])
    clip = {}
    spin_frac = {}
    attack_s = []
    base_n = int((sheets["attack"][:CELL_H, :CELL_W, 3] >= 128).sum())
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                c = tile[..., 3] >= 128
                if c.size and (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any()):
                    clip.setdefault(act, []).append(f"{d}:{f}")
                if act == "attack" and d == "S":
                    n = int(c.sum())
                    ys, xs = np.nonzero(c)
                    bw = int(xs.max() - xs.min() + 1) if len(xs) else 0
                    attack_s.append({
                        "f": f,
                        "opaque": n,
                        "ratio": round(n / base_n, 4) if base_n else 0,
                        "w": bw,
                        "main": round(main_component_fraction(tile), 4),
                    })
                    if 5 <= f <= 10:
                        spin_frac[f] = attack_s[-1]["main"]
    ratios = [row["ratio"] for row in attack_s]
    print("south attack opaque ratios", " ".join(f"{r:.4f}" for r in ratios), flush=True)
    bad_ratio = [row for row in attack_s if row["ratio"] < 0.90 or row["ratio"] > 1.08]
    thin = [row for row in attack_s if 5 <= row["f"] <= 10 and row["w"] < 45]
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": [int(v) for v in alphas],
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "spin_main_frac_S": spin_frac,
        "attack_S_opaque": attack_s,
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:900], flush=True)

    meta = {
        "generator": "forge_monferno_actions", "version": VERSION, "key": "0391_monferno",
        "dex": 391, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "face_right": sorted(FACE_RIGHT), "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "source": "front.png 96x96", "gifs_untouched": True,
        "archetype": "biped_primate_tail_flame",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    ratio_ok = not bad_ratio and not thin and all(v >= 0.9 for v in spin_frac.values())
    if off or clip or not ratio_ok:
        print("QA FAIL", "off", len(off), "clip", clip, "bad_ratio", bad_ratio, "thin", thin, flush=True)
        raise SystemExit(1)
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
