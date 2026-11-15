#!/usr/bin/env python3
"""Vespiquen (0416_vespiquen) 8-direction action sheet from the normal front.png.

Measured on this still only (96x96, opaque 1834, one component, bbox x23-71
y19-76, 49x58, 14 colours). Lower red eyes (y>=32) sit at x=44.9. The opaque
centroid is x=47.4 because the large pale wing trails on the right, so the
still faces left. S SE E NE are mirrored so E points screen-right. W matches
the still.

Parts, from this still's gaps only: a small left wing (x<=42, y<=34), a larger
right wing (pale membrane plus the dark root from x>=53, y<=34), two arms
(left x<=36 y38-43, right x>=60 y40-45), and a crest (olive crown y<=24 plus
the red forehead gem y<=31). The eyes stay on the head. There is no mouth.
The honeycomb skirt (honey pixels, hem y62-76) stays on the body. No extra bees.

Roar flares the two wings and lifts the crest, plus rings. The spin turns the
whole queen as one piece. Idle frame 0 is the still, skirt planted.

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.
"""

import argparse
import fcntl
import hashlib
import json
import math
import re
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
SPECIES = ROOT / "monsterSprite/gen4/0416_vespiquen/normal"
OUT_DIR = SPECIES / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0416_vespiquen/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"
KEY = "0416_vespiquen"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left. These four are mirrored so E points screen-right.
MIRROR = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 200, 192, 168
VERSION = "vespiquen_v2"
KEEP = 6

BLACK = (16, 16, 16)
YELLOW = (255, 205, 8)
WING_GRAY = (65, 65, 74)
HONEY = (172, 82, 0)
GOLD = (180, 139, 0)
BROWN = (106, 41, 0)
DARK = (41, 41, 41)
WHITE = (238, 238, 255)
PALE = (189, 189, 213)
OLIVE = (106, 90, 0)
ORANGE = (238, 115, 0)
BLUE = (82, 90, 115)
DRED = (123, 49, 57)
RED = (238, 57, 57)
PARTS = ("body", "left_wing", "right_wing", "left_arm", "right_arm", "crest")


def is_color(img, opaque, *cols):
    rgb = img[..., :3]
    mask = np.zeros(opaque.shape, bool)
    for col in cols:
        mask |= np.all(rgb == np.array(col, np.uint8), axis=-1)
    return opaque & mask


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        pad = np.pad(out, 1)
        out = out | pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    return out


def segment(img):
    """Wings, arms, and crest. Skirt and abdomen stay on the body.

    Cuts are the gaps measured on this still. The left wing stops at x=42 so
    the yellow face (x>=38 on the upper rows) stays. The right wing takes the
    pale membrane and the dark root from x=53, and stops at y=34 above the
    arm. Arms are the side limbs under those gaps. The crest is only the olive
    crown and the red forehead gem; the lower red pair are the eyes.
    """
    opaque = img[..., 3] >= 128
    height, width = img.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    edge = is_color(img, opaque, WHITE, PALE, BLUE)
    dark = is_color(img, opaque, BLACK, WING_GRAY, DARK)
    left_wing = (edge & (xx <= 46) & (yy <= 34)) | (dark & (xx <= 42) & (yy <= 34))
    right_wing = (edge & (xx >= 56) & (yy <= 34)) | (
        opaque & (xx >= 53) & (yy <= 34) & is_color(img, opaque, BLACK, WING_GRAY, DARK, WHITE, PALE, BLUE)
    )
    right_wing &= ~left_wing
    left_arm = opaque & (yy >= 37) & (yy <= 44) & (xx <= 36)
    right_arm = opaque & (yy >= 39) & (yy <= 45) & (xx >= 60)
    left_arm &= ~(left_wing | right_wing)
    right_arm &= ~(left_wing | right_wing | left_arm)
    crest = (is_color(img, opaque, OLIVE) & (yy <= 24)) | (
        is_color(img, opaque, RED, DRED) & (yy <= 31)
    )
    crest &= ~(left_wing | right_wing | left_arm | right_arm)
    body = opaque & ~(left_wing | right_wing | left_arm | right_arm | crest)
    masks = {
        "body": body,
        "left_wing": left_wing,
        "right_wing": right_wing,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "crest": crest,
    }
    own = body | left_wing | right_wing | left_arm | right_arm | crest
    overlap = 0
    names = list(masks)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            overlap += int((masks[name] & masks[other]).sum())
    if int(own.sum()) != int(opaque.sum()) or overlap:
        raise SystemExit(f"masks do not partition the still overlap={overlap}")
    if min(int(masks[name].sum()) for name in PARTS) <= 0:
        raise SystemExit("a vespiquen part mask is empty")
    eyes = is_color(img, opaque, RED, DRED) & (yy >= 32)
    if int((eyes & body).sum()) != int(eyes.sum()) or int(eyes.sum()) < 8:
        raise SystemExit("eyes were cut off the head")
    skirt = is_color(img, opaque, HONEY) & (yy >= 62)
    if int((skirt & body).sum()) != int(skirt.sum()):
        raise SystemExit("honeycomb skirt left the body")
    return masks


def shift(layer, dx, dy):
    dx_i, dy_i = int(round(dx)), int(round(dy))
    if dx_i == 0 and dy_i == 0:
        return layer.copy()
    height, width = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0, sx1 = max(0, -dx_i), min(width, width - dx_i)
    sy0, sy1 = max(0, -dy_i), min(height, height - dy_i)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy_i:sy1 + dy_i, sx0 + dx_i:sx1 + dx_i] = layer[sy0:sy1, sx0:sx1]
    return out


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest, then a 6-of-16 mode. Multiples of 360 stay exact.

    Positive degrees are clockwise. No |cos| squash. The kept colour is the
    mode of the block, so the palette does not grow.
    """
    if abs(angle_deg) % 360 < 0.05 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    up = np.repeat(np.repeat(layer, 4, axis=0), 4, axis=1)
    px, py = pivot[0] * 4 + 1.5, pivot[1] * 4 + 1.5
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    height, width = up.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    rx, ry = xx - px, yy - py
    src_x = np.rint(px + cos_a * rx + sin_a * ry).astype(np.int32)
    src_y = np.rint(py - sin_a * rx + cos_a * ry).astype(np.int32)
    valid = (src_x >= 0) & (src_x < width) & (src_y >= 0) & (src_y < height)
    out_up = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    out_up[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    out_h, out_w = layer.shape[:2]
    blk = out_up.reshape(out_h, 4, out_w, 4, 4)
    opaque = blk[..., 3] >= 128
    count = opaque.sum(axis=(1, 3))
    out = np.zeros_like(layer)
    ys, xs = np.nonzero(count >= KEEP)
    for y, x in zip(ys.tolist(), xs.tolist()):
        cols = blk[y, :, x, :, :3][opaque[y, :, x, :]]
        if len(cols) == 0:
            continue
        uniq, counts = np.unique(cols, axis=0, return_counts=True)
        out[y, x, :3] = uniq[int(counts.argmax())]
        out[y, x, 3] = 255
    return out


def despeckle(img):
    """Drop pixels whose four neighbours are empty. After a rotation only."""
    out = img.copy()
    alpha = out[..., 3] >= 128
    pad = np.pad(alpha, 1)
    neighbors = (
        pad[:-2, 1:-1].astype(np.int16)
        + pad[2:, 1:-1]
        + pad[1:-1, :-2]
        + pad[1:-1, 2:]
    )
    out[alpha & (neighbors == 0)] = 0
    return out


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    for i in range(0, len(ty), 400):
        sl = slice(i, i + 400)
        dist = (ty[sl, None] - pts[None, :, 0]) ** 2 + (tx[sl, None] - pts[None, :, 1]) ** 2
        pick[sl] = dist.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def extract_palette(img):
    alpha = img[..., 3] >= 128
    return np.unique(img[alpha][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    alpha = out[..., 3] >= 128
    if not np.any(alpha):
        out[..., 3] = 0
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = alpha.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    mapped = palette[dist.argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(alpha, 255, 0).astype(np.uint8)
    out[~alpha] = 0
    return out


def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    height, width = out.shape[:2]
    if 2 <= x < width - 2 and 2 <= y < height - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def paste(dst, layer):
    mask = layer[..., 3] >= 128
    dst[mask] = layer[mask]


def components(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask)
    out = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        queue = deque([(y, x)])
        seen[y, x] = True
        pts = []
        while queue:
            cy, cx = queue.popleft()
            pts.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        out.append(pts)
    return out


def edge_pivot(part, side):
    ys, xs = np.nonzero(part)
    if side == "inner_right":
        edge = xs.max()
        sel = xs >= edge - 1
    else:
        edge = xs.min()
        sel = xs <= edge + 1
    return float(xs[sel].mean()), float(ys[sel].mean())


def base_pivot(part):
    ys, xs = np.nonzero(part)
    bottom = ys.max()
    sel = ys >= bottom - 1
    return float(xs[sel].mean()), float(bottom)


def build_rig(img, masks):
    opaque = img[..., 3] >= 128
    ys, xs = np.nonzero(opaque)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sprite_h, sprite_w = y1 - y0, x1 - x0
    dst_x = (CELL_W - sprite_w) // 2
    dst_y = FOOT_Y - sprite_h
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sprite_h, dst_x:dst_x + sprite_w] = img[y0:y1, x0:x1]

    placed = {}
    for name, mask in masks.items():
        placed_mask = np.zeros((CELL_H, CELL_W), bool)
        placed_mask[dst_y:dst_y + sprite_h, dst_x:dst_x + sprite_w] = mask[y0:y1, x0:x1]
        placed[name] = placed_mask

    layers = {}
    for name, placed_mask in placed.items():
        layer = np.zeros_like(canvas)
        layer[placed_mask] = canvas[placed_mask]
        layers[name] = layer

    # Joint cushion under wings and arms. Hidden while the part rests on it.
    for part_name in ("left_wing", "right_wing", "left_arm", "right_arm"):
        zone = placed[part_name] & dilate(placed["body"], 1)
        if not np.any(zone):
            continue
        fill = nearest_color(canvas, zone, placed["body"])
        write = zone & (layers["body"][..., 3] < 128)
        layers["body"][write] = fill[write]

    # The crest sits in the forehead. Paint the face under the whole gem so a
    # lift does not open a hole. At rest the crest covers this paint.
    if np.any(placed["crest"]):
        under = nearest_color(canvas, placed["crest"], placed["body"])
        layers["body"][placed["crest"]] = under[placed["crest"]]

    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    bottom = int(oys.max())
    foot_sel = oys >= bottom - 1
    foot_pivot = (float(oxs[foot_sel].mean()), float(bottom))
    pivots = {
        "left_wing": edge_pivot(placed["left_wing"], "inner_right"),
        "right_wing": edge_pivot(placed["right_wing"], "inner_left"),
        "left_arm": edge_pivot(placed["left_arm"], "inner_right"),
        "right_arm": edge_pivot(placed["right_arm"], "inner_left"),
        "crest": base_pivot(placed["crest"]),
    }
    return layers, pivots, center, foot_pivot, canvas


def wave(frame, count, amp, phase=0.0, cycles=1.0):
    ph = 2 * math.pi * cycles * (frame / count)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def iwave(frame, count, amp, phase=0.0, cycles=1.0):
    return int(round(wave(frame, count, amp, phase, cycles)))


def pose(action, frame):
    count = FRAMES[action]
    pose_now = dict(
        dx=0, dy=0, wing=0.0, arm_l=0.0, arm_r=0.0, crest_dy=0,
        tilt=0.0, spin=0.0, hurt=0.0, rings=0, ring=0, burst=0, sparks=0,
    )
    if action == "idle":
        # Frame 0 is the still, skirt planted. Later frames hover one pixel.
        pose_now["dy"] = iwave(frame, count, 1.0)
        pose_now["wing"] = wave(frame, count, 8.0)
        pose_now["arm_l"] = wave(frame, count, 5.0)
        pose_now["arm_r"] = wave(frame, count, 5.0, phase=math.pi)
        pose_now["crest_dy"] = iwave(frame, count, 1.0, phase=0.6)
    elif action == "walk":
        # No feet on this still. Hover-glide: wings flap, arms swing, skirt rides the body.
        pose_now["dx"] = iwave(frame, count, 1.0)
        pose_now["dy"] = iwave(frame, count, 1.0, cycles=2.0)
        pose_now["wing"] = wave(frame, count, 16.0, cycles=2.0)
        pose_now["arm_l"] = wave(frame, count, 11.0, cycles=2.0)
        pose_now["arm_r"] = wave(frame, count, 11.0, cycles=2.0, phase=math.pi)
        pose_now["crest_dy"] = iwave(frame, count, 1.0, cycles=2.0)
    elif action == "attack":
        # 0 is the still. 1-4 flare wings and lift the crest. 5-10 one full turn.
        wing = [0, 8, 16, 22, 14, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 0]
        arms = [0, -6, -12, -16, -10, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 0]
        crest = [0, -1, -2, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0]
        pose_now["wing"] = float(wing[frame])
        pose_now["arm_l"] = float(arms[frame])
        pose_now["arm_r"] = float(arms[frame])
        pose_now["crest_dy"] = int(crest[frame])
        if 1 <= frame <= 4:
            pose_now["rings"] = 2 if frame in (2, 3) else 1
            pose_now["ring"] = frame - 1
        if 5 <= frame <= 10:
            ang = 60.0 * (frame - 4)
            pose_now["spin"] = 0.0 if ang % 360 == 0 else ang
        if frame in (11, 12):
            pose_now["burst"] = 1
    else:
        env = [0.0, 0.35, 0.65, 0.88, 1.0, 0.92, 0.72, 0.48, 0.28, 0.12, 0.04, 0.0][frame]
        # Negative raises the left-hand face. Cap is 38 degrees.
        pose_now["hurt"] = -38.0 * env
        pose_now["dx"] = int(round(3 * env))
        pose_now["sparks"] = 1 if 2 <= frame <= 5 else 0
    if abs(pose_now["hurt"]) > 38.01:
        raise SystemExit("hurt angle exceeds 38")
    return pose_now


def turn(layer, angle, pivot):
    if abs(angle) < 0.05 or abs(angle) % 360 < 0.05:
        return layer
    return despeckle(rotate_nearest(layer, angle, pivot))


def leading_edge(img):
    """Still faces left, so the roar sits just past the left-hand wing tip."""
    alpha = img[..., 3] >= 128
    ys, xs = np.nonzero(alpha)
    return float(xs.min()), float(np.median(ys))


def roar_rings(out, age, strength):
    """Rings in front of the wings. There is no mouth on this still."""
    ox, oy = leading_edge(out)
    cols = (YELLOW, ORANGE, RED)
    for ring_i in range(strength):
        rad = 5 + age * 2 + ring_i * 3
        col = cols[ring_i % 3]
        for deg in range(-40, 41, 20):
            ang = math.radians(deg)
            put(out, ox - 2 - rad * math.cos(ang), oy + rad * math.sin(ang) * 0.55, col)
    return out


def impact_burst(out):
    ox, oy = leading_edge(out)
    cols = (YELLOW, ORANGE, WHITE)
    for k in range(6):
        ang = math.radians(-50 + k * 20)
        put(out, ox - 3 - 4 * math.cos(ang), oy + 3 * math.sin(ang), cols[k % 3])
        put(out, ox - 3 - 7 * math.cos(ang), oy + 5 * math.sin(ang), cols[(k + 1) % 3])
    return out


def sparks(out, frame_idx):
    alpha = out[..., 3] >= 128
    ys, xs = np.nonzero(alpha)
    if len(ys) == 0:
        return out
    x0, y0 = int(xs.min()), int(ys.min())
    step = frame_idx
    for sx, sy, col in (
        (x0 - 2 - step, y0 + 6, YELLOW),
        (x0 - 1, y0 - 2 - step, ORANGE),
        ((x0 + int(xs.max())) // 2, y0 - 3 - step, RED),
    ):
        put(out, sx, sy, col)
    return out


def synth(rig, action, frame, palette):
    layers, pivots, center, foot_pivot = rig
    pose_now = pose(action, frame)
    pose_now = gen4_attack_strike.finish(pose_now, action, frame, __file__)

    spinning = abs(pose_now["spin"]) > 0.05
    recoiling = abs(pose_now["hurt"]) > 0.05
    wing = 0.0 if spinning or recoiling else pose_now["wing"]
    arm_l = 0.0 if spinning or recoiling else pose_now["arm_l"]
    arm_r = 0.0 if spinning or recoiling else pose_now["arm_r"]
    crest_dy = 0 if spinning or recoiling else pose_now["crest_dy"]
    # Positive wing lifts both tips. Negative arm angles lift both hands.
    left = turn(layers["left_wing"], -wing, pivots["left_wing"])
    right = turn(layers["right_wing"], wing, pivots["right_wing"])
    arm_left = turn(layers["left_arm"], arm_l, pivots["left_arm"])
    arm_right = turn(layers["right_arm"], arm_r, pivots["right_arm"])
    crest = shift(layers["crest"], 0, crest_dy)
    img = np.zeros_like(layers["body"])
    paste(img, layers["body"])
    paste(img, left)
    paste(img, right)
    paste(img, arm_left)
    paste(img, arm_right)
    paste(img, crest)
    if spinning:
        img = turn(img, pose_now["spin"], center)
    elif recoiling:
        img = turn(img, pose_now["hurt"], foot_pivot)
    img = shift(img, pose_now["dx"], pose_now["dy"])
    img = snap(img, palette)
    if pose_now["rings"] and not spinning:
        img = roar_rings(img, pose_now["ring"], pose_now["rings"])
    if pose_now["burst"]:
        img = impact_burst(img)
    if pose_now["sparks"]:
        img = sparks(img, frame)
    return snap(img, palette)


def facing_flip(img, direction):
    if direction in MIRROR:
        return img[:, ::-1].copy()
    return img


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (SPECIES / "front.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def assemble(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sheet = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, direction in enumerate(DIRS):
            for frame in range(FRAMES[act]):
                tile = np.array(Image.open(frames_dir / act / direction / f"{frame:02d}.png"))
                sheet[di * CELL_H:(di + 1) * CELL_H, frame * CELL_W:(frame + 1) * CELL_W] = tile
        sheets[act] = sheet
    return sheets


def write_outputs(sheets):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for act, sheet in sheets.items():
        image = Image.fromarray(sheet, "RGBA")
        image.save(OUT_DIR / f"{act}.png")
        image.save(OUT_DIR / f"{act}_8dir.png")
        Image.fromarray(sheet[:CELL_H], "RGBA").save(OUT_DIR / f"{act}_S_strip.png")
    master = np.concatenate([sheets[act] for act in ACTIONS], axis=0)
    Image.fromarray(master, "RGBA").save(OUT_DIR / "sheet.png")
    Image.fromarray(master, "RGBA").save(OUT_DIR / "all_8dir_actions.png")
    return master


def preview_image(sheets):
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 4, 8, 12),
        "attack": (0, 2, 4, 6, 8, 10, 11, 15),
        "hurt": (0, 2, 4, 6, 8, 11),
    }
    cols = max(len(v) for v in samples.values())
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, frame in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, frame * CELL_W:(frame + 1) * CELL_W]
            y = ai * CELL_H
            x = si * CELL_W
            mask = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][mask] = tile[mask]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 4
    crops = []
    for act in ACTIONS:
        row = []
        for frame in range(FRAMES[act]):
            tile = sheets[act][0:CELL_H, frame * CELL_W:(frame + 1) * CELL_W]
            alpha = tile[..., 3] >= 128
            if not np.any(alpha):
                continue
            ys, xs = np.nonzero(alpha)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(CELL_H, int(ys.max()) + 1 + pad)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(CELL_W, int(xs.max()) + 1 + pad)
            row.append(tile[y0:y1, x0:x1])
        crops.append(row)
    cell_h = max(crop.shape[0] for row in crops for crop in row)
    cell_w = max(crop.shape[1] for row in crops for crop in row)
    cols = max(len(row) for row in crops)
    bg = np.full((len(crops) * cell_h, cols * cell_w, 4), (15, 20, 31, 255), np.uint8)
    for ri, row in enumerate(crops):
        for ci, crop in enumerate(row):
            y = ri * cell_h + (cell_h - crop.shape[0]) // 2
            x = ci * cell_w + (cell_w - crop.shape[1]) // 2
            mask = crop[..., 3] >= 128
            bg[y:y + crop.shape[0], x:x + crop.shape[1]][mask] = crop[mask]
    return Image.fromarray(bg, "RGBA")


SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "all_8dir_actions.png",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            copied += 1
    return copied


def opaque_count(tile):
    return int((tile[..., 3] == 255).sum())


def south_attack_ratios(sheets):
    row = sheets["attack"][:CELL_H]
    base = opaque_count(row[:, :CELL_W])
    ratios = []
    for frame in range(FRAMES["attack"]):
        count = opaque_count(row[:, frame * CELL_W:(frame + 1) * CELL_W])
        ratios.append(count / base if base else 0.0)
    return base, ratios


def largest_fraction(tile):
    alpha = tile[..., 3] >= 128
    parts = components(alpha)
    if not parts:
        return 0.0
    parts.sort(key=len, reverse=True)
    return len(parts[0]) / max(1, int(alpha.sum()))


def qa_numbers(master, sheets, palette, canvas):
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = [int(v) for v in np.unique(master[..., 3])]
    bad_alpha = [v for v in alphas if v not in (0, 255)]
    clip = []
    for act in ACTIONS:
        sheet = sheets[act]
        for di, direction in enumerate(DIRS):
            for frame in range(FRAMES[act]):
                cell = sheet[di * CELL_H:(di + 1) * CELL_H, frame * CELL_W:(frame + 1) * CELL_W, 3] >= 128
                if cell[0].any() or cell[-1].any() or cell[:, 0].any() or cell[:, -1].any():
                    clip.append(f"{act}:{direction}:{frame}")
    west = sheets["idle"][6 * CELL_H:7 * CELL_H, :CELL_W]
    differ = np.any(west != canvas, axis=2) & ((west[..., 3] >= 128) | (canvas[..., 3] >= 128))
    base, ratios = south_attack_ratios(sheets)
    bad_ratio = [round(r, 3) for r in ratios if r < 0.90 or r > 1.08]
    spin_frac = []
    for frame in range(5, 11):
        tile = sheets["attack"][:CELL_H, frame * CELL_W:(frame + 1) * CELL_W]
        spin_frac.append(round(largest_fraction(tile), 3))
    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] == 255)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0
    return {
        "off": off,
        "bad_alpha": bad_alpha,
        "clip": clip,
        "rest_diff": int(differ.sum()),
        "base": base,
        "ratios": ratios,
        "bad_ratio": bad_ratio,
        "spin_frac": spin_frac,
        "sprite_h": sprite_h,
    }


def _entry_span(text, key):
    token = f'"{key}":'
    start = text.index(token)
    i = start + len(token)
    while text[i] != "{":
        i += 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return start, j + 1
    raise SystemExit(f"dir8 entry did not close: {key}")


def publish_dir8(sprite_h):
    entry = {
        "dex": 416,
        "form": "normal",
        "variant": "vespiquen",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0416_vespiquen/normal",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir",
        "nf": 16,
        "cw": CELL_W,
        "ch": CELL_H,
        "w": CELL_W,
        "h": int(sprite_h),
        "fy": int(FOOT_Y),
        "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1,
        "src": "agent-sprite-forge",
        "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    DIR8_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with DIR8_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        keys = re.findall(r'"(\d{4}_[^"]+)":\{', text)
        if KEY not in keys:
            raise SystemExit("0416_vespiquen missing from dir8.js")
        idx = keys.index(KEY)
        neighbor = keys[idx + 1] if idx + 1 < len(keys) else keys[idx - 1]
        nb_s, nb_e = _entry_span(text, neighbor)
        neighbor_blob = text[nb_s:nb_e]
        start, end = _entry_span(text, KEY)
        text = text[:start] + f'"{KEY}":' + blob + text[end:]
        if neighbor_blob not in text:
            raise SystemExit("neighbor dir8 key changed")
        body = text[len(prefix):].strip()
        if body.endswith(";"):
            body = body[:-1].strip()
        json.loads(body)
        if text.count(f'"v":"{VERSION}"') != 1:
            raise SystemExit("dir8 version stamp collided")
        DIR8.write_text(text)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return neighbor


def update_queue(status):
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = QUEUE.read_text()
        start = text.index('"id": "0416_vespiquen"')
        nxt = text.find('"id":', start + 10)
        end = len(text) if nxt < 0 else nxt
        block = text[start:end]
        if '"status":' not in block:
            raise SystemExit("queue status missing")
        block2 = re.sub(r'"status": "[^"]*"', f'"status": "{status}"', block, count=1)
        mid, n_mid = re.subn(r'\n\s*"claim": "[^"]*",', "", block2, count=1)
        if n_mid:
            block2 = mid
        else:
            block2 = re.sub(r',\s*\n\s*"claim": "[^"]*"', "", block2, count=1)
        text = text[:start] + block2 + text[end:]
        data = json.loads(text)
        ids = [item.get("id") for item in data["items"]]
        if ids.count(KEY) != 1:
            raise SystemExit("queue id collision")
        for item in data["items"]:
            if item.get("id") == KEY:
                if item.get("status") != status or "claim" in item:
                    raise SystemExit("queue update did not stick")
        QUEUE.write_text(text if text.endswith("\n") else text + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)


def mark_needs_fix():
    try:
        update_queue("needs_fix")
    except Exception as exc:
        print("queue needs_fix failed", exc, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front_path = SPECIES / "front.png"
    front = np.array(Image.open(front_path).convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    print("parts", counts, "palette", len(palette), flush=True)

    opaque = front[..., 3] >= 128
    rgb = front[..., :3]
    eyes = opaque & (
        np.all(rgb == np.array(RED, np.uint8), axis=-1)
        | np.all(rgb == np.array(DRED, np.uint8), axis=-1)
    ) & (np.arange(96)[:, None] >= 32)
    ey, ex = np.nonzero(eyes)
    face_x = float(ex.mean())
    by, bx = np.nonzero(opaque)
    body_x = float(bx.mean())
    print(f"facing face_x {face_x:.1f} body_x {body_x:.1f}", flush=True)
    if not (face_x < body_x - 2):
        raise SystemExit("facing check failed; eyes are not left of the body")

    layers, pivots, center, foot_pivot, canvas = build_rig(front, masks)
    rig = (layers, pivots, center, foot_pivot)
    print(
        "pivot", [round(v, 2) for v in center],
        "foot", [round(v, 2) for v in foot_pivot],
        flush=True,
    )

    rest = synth(rig, "idle", 0, palette)
    differ = np.any(rest != canvas, axis=2) & ((rest[..., 3] >= 128) | (canvas[..., 3] >= 128))
    print("rest-pose pixels changed", int(differ.sum()), flush=True)
    if int(differ.sum()) != 0:
        raise SystemExit("idle frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "body": (40, 70, 150),
        "left_wing": (190, 210, 255),
        "right_wing": (245, 245, 255),
        "left_arm": (255, 150, 40),
        "right_arm": (255, 90, 30),
        "crest": (255, 40, 60),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", frame) for frame in (0, 4, 8, 12)]
        want += [("walk", frame) for frame in (0, 4, 8, 12)]
        want += [("attack", frame) for frame in range(16)]
        want += [("hurt", frame) for frame in (0, 2, 4, 6, 8, 11)]
        tiles = [facing_flip(synth(rig, act, frame, palette), "S") for act, frame in want]
        base = opaque_count(tiles[8])
        print("sample attack S ratios", flush=True)
        ratios = []
        for i, frame in enumerate(range(16)):
            count = opaque_count(tiles[8 + i])
            ratio = count / base
            ratios.append(ratio)
            print(f"  {frame:02d} {count} {ratio:.3f}", flush=True)
        print(f"min {min(ratios):.3f} max {max(ratios):.3f}", flush=True)
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            mask = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][mask] = tile[mask]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        print("samples", work / "samples.png", flush=True)
        return

    frames_dir = work / "frames"
    state_path = work / "state.json"
    digest = params_hash()
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if args.force or state.get("hash") != digest:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        state = {"hash": digest, "done": []}
    frames_dir.mkdir(parents=True, exist_ok=True)

    total = len(DIRS) * len(ACTIONS)
    for act in ACTIONS:
        for direction in DIRS:
            key = f"{act}/{direction}"
            out_dir = frames_dir / act / direction
            out_dir.mkdir(parents=True, exist_ok=True)
            for frame in range(FRAMES[act]):
                fp = out_dir / f"{frame:02d}.png"
                if key in state.get("done", []) and fp.exists() and fp.stat().st_size > 0 and not args.force:
                    continue
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest and not args.force:
                    continue
                tmp = out_dir / f"{frame:02d}.tmp.png"
                Image.fromarray(facing_flip(synth(rig, act, frame, palette), direction), "RGBA").save(tmp)
                tmp.replace(fp)
            if key not in state["done"]:
                state["done"].append(key)
            state_path.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions_dir = SPECIES / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    preview.save(actions_dir / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    report = qa_numbers(master, sheets, palette, canvas)
    base = report["base"]
    ratios = report["ratios"]
    print("attack S opaque ratios", flush=True)
    for frame, ratio in enumerate(ratios):
        count = int(round(ratio * base))
        print(f"  {frame:02d} {count} {ratio:.3f}", flush=True)
    print(f"min {min(ratios):.3f} max {max(ratios):.3f} h {report['sprite_h']}", flush=True)

    qa = work / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    gait = {}
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            out_dir = qa / f"gait_{act}_S"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(out_dir), "--kind", kind, "--no-aids"],
                capture_output=True, text=True,
            )
            rep = out_dir / "loop-report.json"
            if rep.exists():
                gait[act] = json.loads(rep.read_text()).get("recommendation")
    summary = {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": report["sprite_h"],
        "off_palette": len(report["off"]),
        "bad_alpha": report["bad_alpha"],
        "clip": report["clip"],
        "rest_diff": report["rest_diff"],
        "attack_S_ratios": [round(r, 3) for r in ratios],
        "spin_main_frac": report["spin_frac"],
        "gait": gait,
    }
    (qa / "qa.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "parts"}, ensure_ascii=False)[:1600], flush=True)

    meta = {
        "generator": "forge_vespiquen_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 416,
        "form": "normal",
        "variant": "vespiquen",
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": report["sprite_h"],
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "mirror": sorted(MIRROR),
        "palette_colors": int(len(palette)),
        "off_palette_colors": len(report["off"]),
        "source": "front.png 96x96",
        "gifs_untouched": True,
        "wings": 2,
        "arms": 2,
        "crest": 1,
        "mouth": 0,
        "skirt": "on-body",
        "bees_added": 0,
        "face_x": round(face_x, 2),
        "body_x": round(body_x, 2),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = (
        report["off"] or report["bad_alpha"] or report["clip"] or report["rest_diff"]
        or report["bad_ratio"] or any(v < 0.90 for v in report["spin_frac"])
    )
    if failed:
        mark_needs_fix()
        raise SystemExit(
            f"qa failed off={len(report['off'])} alpha={report['bad_alpha']} "
            f"clip={report['clip'][:8]} rest={report['rest_diff']} "
            f"ratio={report['bad_ratio']} spin={report['spin_frac']}"
        )
    if args.publish and not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
        neighbor = publish_dir8(report["sprite_h"])
        update_queue("done")
        sheet = SYNC_OUT / "sheet.png"
        print("published", neighbor, "arena_sheet_bytes", sheet.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
