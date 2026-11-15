#!/usr/bin/env python3
"""Mega Charizard Y (0006_charizard-mega-y): 8-direction x 4-action sheet, built FROM THE CURRENT SPRITE.

User request 2026-11-15: "the Mega Charizard Y actions are too plain; rewrite them at the level of Mega Charizard X,
without breaking the features / look of the current sprite". So unlike the other gen1 forge scripts there is NO new
key art: the stills are frame 0 of the existing hand-cleaned 8-direction sheet 0006_charizard-mega-y.png
(275x196 cells, columns S SE E NE N NW W SW). Flow (structure of tools/forge_gen1_0006_charizard_mega_x_actions.py;
rig, colours and attack are this form's own):
  1. Stills: S / SW / W / NE / N cells, cropped, area-downscaled x0.5 (S body = 96 px, the size the game showed the
     old sheet at). SE / E / NW are mirrors (the old sheet's were mirrors too).
  2. agent-sprite-forge palette_tool: build (30 colours + 3 reserved fire-FX colours) -> lock -> apply.
  3. Anatomical rig per view: two wings (pivot at the shoulder), head, arms, torso, two feet, tail with the flame.
     Posed at 4x (nearest), back to 1x.
     Actions (16 frames): idle (breath, slow wing beat, tail sway, tail-flame embers), walk (alternating stomps, arm
     swing, wing beat), attack (flamethrower: wings rise and the head rears back with a glow at the mouth -> lunge,
     wings sweep down, a growing stream of fire from the mouth -> embers, recover), hurt (white flash, recoil, wings
     flare, embers, recover).
  4. QC: off-palette, edge-touch, still diff (frame 0 of every action = the rest pose), gait_loop.py idle / walk.

The old sheet png and the official front / back png are never touched.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0006_charizard_mega_y_actions.py [--force]
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
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
KEY = "0006_charizard-mega-y"
REL = "monsterSprite/gen1/0006_charizard/mega/charizard-mega-y"
FORM_DIR = ROOT / REL
SRC_SHEET = FORM_DIR / "0006_charizard-mega-y.png"
SRC_CW, SRC_CH = 275, 196
SRC_COL = {"S": 0, "NE": 3, "N": 4, "W": 6, "SW": 7}
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/monster_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0006_charizard_mega_y_rig_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
SCALE = 0.5
CELL_W, CELL_H, FOOT_Y = 256, 148, 132
UP = 4
FX_HI, FX_MID, FX_LO = (255, 240, 150), (255, 172, 48), (236, 84, 24)
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each half-scale view)
# label: later polygons win. draw: back to front. sgn: rotation sign that RAISES a wing / tail tip.
# mouth = where the fire leaves (follows the head), fwd = screen direction of forward, tip = tail-flame top (embers).
RIG = {
    "S": dict(
        poly={"wing_l": [(0, 0), (48, 0), (50, 20), (56, 36), (63, 45), (60, 50), (52, 50), (46, 46), (41, 50), (39, 58), (30, 62), (0, 74)],
              "wing_r": [(88, 0), (136, 0), (136, 74), (112, 50), (104, 56), (97, 60), (93, 50), (84, 46), (74, 45), (80, 36), (86, 20)],
              "tail": [(97, 73), (106, 72), (110, 44), (128, 44), (128, 74), (120, 84), (100, 87), (95, 80)],
              "body": [(56, 40), (80, 40), (86, 50), (92, 74), (96, 86), (40, 86), (26, 80), (28, 62), (44, 56), (50, 48)],
              "foot_l": [(40, 78), (60, 80), (60, 96), (40, 96)],
              "foot_r": [(80, 80), (100, 78), (100, 96), (80, 96)],
              "head": [(50, 0), (86, 0), (86, 30), (78, 42), (60, 42), (52, 30)],
              "arm_l": [(41, 51), (56, 47), (60, 54), (56, 72), (41, 73)],
              "arm_r": [(78, 47), (93, 51), (99, 70), (84, 73), (78, 58)]},
        label=["wing_l", "wing_r", "tail", "body", "foot_l", "foot_r", "head", "arm_l", "arm_r"],
        draw=["wing_l", "wing_r", "tail", "foot_l", "foot_r", "body", "head", "arm_l", "arm_r"],
        pivot={"wing_l": (60, 46), "wing_r": (77, 46), "tail": (98, 80), "head": (68, 42), "arm_l": (56, 50),
               "arm_r": (80, 50), "body": (68, 86), "foot_l": (50, 94), "foot_r": (90, 94)},
        sgn={"wing_l": 1, "wing_r": -1, "tail": -1},
        body_x=68, fwd=(0.0, 1.0), mouth=(68, 38), tip=(116, 46), back=False),
    "SW": dict(
        poly={"wing_far": [(0, 0), (35, 0), (35, 28), (40, 36), (50, 40), (50, 52), (40, 53), (26, 54), (14, 80), (0, 80)],
              "wing_near": [(72, 0), (107, 0), (107, 80), (98, 77), (92, 60), (88, 56), (86, 48), (76, 48), (71, 40), (74, 28)],
              "tail": [(75, 67), (90, 63), (87, 44), (101, 44), (101, 70), (86, 77), (75, 75)],
              "body": [(52, 30), (66, 28), (74, 44), (76, 64), (75, 82), (52, 85), (40, 80), (42, 60), (50, 44)],
              "foot_a": [(20, 52), (44, 54), (46, 74), (44, 94), (28, 94), (20, 76)],
              "foot_b": [(64, 78), (84, 78), (84, 94), (64, 94)],
              "head": [(36, 0), (72, 0), (66, 16), (60, 30), (52, 35), (38, 33), (34, 28)],
              "arm_near": [(32, 40), (44, 40), (65, 62), (63, 71), (52, 71), (32, 50)]},
        label=["wing_far", "wing_near", "tail", "body", "foot_a", "foot_b", "head", "arm_near"],
        draw=["wing_far", "wing_near", "tail", "foot_b", "body", "foot_a", "head", "arm_near"],
        pivot={"wing_far": (42, 44), "wing_near": (76, 42), "tail": (78, 72), "head": (56, 32), "arm_near": (38, 42),
               "body": (58, 85), "foot_a": (36, 92), "foot_b": (74, 92)},
        sgn={"wing_far": 1, "wing_near": -1, "tail": -1},
        body_x=56, fwd=(-0.9, 0.45), mouth=(38, 27), tip=(94, 46), back=False),
    "W": dict(
        poly={"wing": [(30, 0), (62, 0), (110, 36), (112, 49), (96, 47), (78, 53), (64, 50), (52, 54), (44, 50), (36, 42), (30, 30)],
              "tail": [(62, 72), (96, 73), (108, 56), (108, 14), (128, 14), (128, 58), (113, 62), (100, 87), (58, 91), (57, 79)],
              "body": [(22, 40), (36, 40), (52, 54), (60, 60), (76, 56), (60, 72), (56, 81), (40, 85), (24, 81), (20, 60)],
              "foot_a": [(16, 75), (34, 77), (36, 96), (16, 96)],
              "foot_b": [(40, 73), (58, 71), (60, 96), (42, 96)],
              "head": [(0, 10), (16, 10), (24, 0), (40, 0), (40, 10), (30, 22), (28, 36), (22, 42), (4, 43), (0, 34)],
              "arm_far": [(8, 54), (22, 54), (24, 73), (8, 75)],
              "arm_near": [(30, 56), (46, 50), (52, 58), (44, 75), (30, 75)]},
        label=["wing", "tail", "body", "foot_a", "foot_b", "head", "arm_far", "arm_near"],
        draw=["wing", "tail", "foot_b", "arm_far", "body", "foot_a", "head", "arm_near"],
        pivot={"wing": (40, 40), "tail": (62, 80), "head": (26, 38), "arm_far": (20, 58), "arm_near": (40, 56),
               "body": (40, 85), "foot_a": (26, 94), "foot_b": (50, 94)},
        sgn={"wing": -1, "tail": -1},
        body_x=40, fwd=(-1.0, 0.0), mouth=(3, 37), tip=(120, 18), back=False),
    "NE": dict(
        poly={"wing_l": [(0, 8), (40, 8), (42, 30), (40, 40), (30, 46), (22, 54), (8, 68), (0, 68)],
              "wing_r": [(57, 45), (80, 48), (93, 62), (93, 75), (70, 77), (62, 64)],
              "tail": [(38, 0), (58, 0), (58, 28), (51, 40), (43, 40), (43, 28)],
              "body": [(42, 36), (62, 40), (70, 30), (80, 44), (94, 60), (90, 86), (40, 90), (20, 76), (18, 56), (26, 44)],
              "head": [(60, 0), (99, 0), (99, 44), (84, 44), (74, 36), (66, 22)],
              "foot_l": [(32, 84), (48, 84), (48, 94), (32, 94)],
              "foot_r": [(64, 85), (84, 85), (84, 96), (64, 96)]},
        label=["wing_l", "wing_r", "tail", "body", "head", "foot_l", "foot_r"],
        draw=["foot_l", "foot_r", "body", "head", "wing_r", "wing_l", "tail"],
        pivot={"wing_l": (40, 36), "wing_r": (62, 50), "head": (74, 40), "tail": (47, 40), "body": (55, 90),
               "foot_l": (40, 92), "foot_r": (74, 94)},
        sgn={"wing_l": 1, "wing_r": -1, "tail": 1},
        body_x=55, fwd=(0.9, -0.45), mouth=(96, 34), tip=(47, 2), back=True),
    "N": dict(
        poly={"wing_l": [(0, 0), (44, 0), (48, 40), (52, 48), (46, 54), (40, 52), (30, 62), (0, 74)],
              "wing_r": [(121, 0), (77, 0), (73, 40), (69, 48), (75, 54), (81, 52), (91, 62), (121, 74)],
              "body": [(46, 44), (76, 44), (92, 56), (94, 70), (90, 90), (32, 90), (28, 70), (30, 56)],
              "head": [(46, 0), (75, 0), (75, 26), (46, 26)],
              "foot_l": [(36, 85), (52, 85), (52, 96), (36, 96)],
              "foot_r": [(71, 85), (88, 85), (88, 96), (71, 96)],
              "tail": [(49, 22), (73, 22), (73, 46), (69, 60), (71, 84), (61, 93), (50, 90), (52, 60), (50, 46)]},
        label=["wing_l", "wing_r", "body", "head", "foot_l", "foot_r", "tail"],
        draw=["foot_l", "foot_r", "body", "head", "wing_l", "wing_r", "tail"],
        pivot={"wing_l": (50, 46), "wing_r": (71, 46), "head": (60, 26), "tail": (60, 86), "body": (60, 90),
               "foot_l": (44, 94), "foot_r": (80, 94)},
        sgn={"wing_l": 1, "wing_r": -1, "tail": -1},
        body_x=60, fwd=(0.0, -1.0), mouth=(60, 6), tip=(60, 22), back=True),
}


# ---------------------------------------------------------------- 1. stills from the current 8-direction sheet
def make_stills(dst: Path):
    sheet = Image.open(SRC_SHEET).convert("RGBA")
    dst.mkdir(parents=True, exist_ok=True)
    for v, col in SRC_COL.items():
        im = sheet.crop((col * SRC_CW, 0, (col + 1) * SRC_CW, SRC_CH))
        im = im.crop(im.getbbox())
        a = np.array(im).astype(float)
        a[..., :3] *= a[..., 3:4] / 255.0                 # premultiply so transparent texels do not darken edges
        pm = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")
        sm = np.array(pm.resize((round(im.width * SCALE), round(im.height * SCALE)), Image.BOX)).astype(float)
        al = sm[..., 3:4]
        rgb = np.where(al > 0, sm[..., :3] * 255.0 / np.maximum(al, 1), 0)
        out = np.dstack([np.clip(rgb, 0, 255), np.where(al[..., 0] >= 128, 255, 0)]).astype(np.uint8)
        out[out[..., 3] == 0] = 0
        Image.fromarray(out, "RGBA").save(dst / f"{v}.png")


def run_tool(args):
    r = subprocess.run([str(FPY), "-I", str(PALETTE_TOOL)] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"palette_tool {args[0]} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


# ---------------------------------------------------------------- 3. rig
def poly_mask(p, shape):
    m = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(m).polygon(p, fill=255)
    return np.array(m) > 0


def build_rig(view, still):
    R = RIG[view]
    op = still[..., 3] > 0
    lab = np.zeros(op.shape, int)
    names = R["label"]
    for i, n in enumerate(names, start=1):
        lab[poly_mask(R["poly"][n], op.shape) & op] = i
    un = op & (lab == 0)
    if un.any():
        _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab[un] = lab[iy[un], ix[un]]
    # every part must be one piece: stray islands go to the nearest other part
    for i, n in enumerate(names, start=1):
        m = lab == i
        cl, cn = ndi.label(m)
        if cn > 1:
            sz = ndi.sum(np.ones_like(cl), cl, range(1, cn + 1))
            big = int(np.argmax(sz)) + 1
            stray = m & (cl != big)
            other = op & (lab != i)
            if other.any():
                _, (iy, ix) = ndi.distance_transform_edt(~other, return_indices=True)
                lab[stray] = lab[iy[stray], ix[stray]]
    masks = {n: lab == (i + 1) for i, n in enumerate(names)}
    layers = {}
    order = R["draw"]
    r0, g0, b0 = [still[..., c].astype(int) for c in range(3)]
    bright = np.maximum(np.maximum(r0, g0), b0)
    for i, n in enumerate(order):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in order[i + 1:]:
            above |= masks[n2]
        cushion = 5 if (n.startswith("wing") or n in ("body", "tail")) else 3
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion)
        src_m = m
        if n.startswith("wing") and m.any():
            # the tail flame / arms sit in front of a wing: paint the membrane underneath so no hole opens
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(m, 10), iterations=10))[10:-10, 10:-10]
            seam |= above & ~m & inside & (ndi.distance_transform_edt(~m) <= 14)
        if n == "body" and m.any():
            # underlay for everything drawn above the body inside the body + head outline (arms over the belly),
            # so no hole opens when an arm moves
            bh = m | masks.get("head", np.zeros_like(m))
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(bh, 8), iterations=8))[8:-8, 8:-8]
            seam |= above & ~m & inside & ~masks.get("head", np.zeros_like(m))
            src_m = bh
        if seam.any() and m.any():
            fillable = src_m & (bright > 70)
            if n.startswith("wing"):
                # membrane colour only: not the outline and not the light rim next to another part
                core = ndi.binary_erosion(src_m, iterations=3) & (bright > 70)
                if core.any():
                    fillable = core
            if not fillable.any():
                fillable = src_m
            _, (iy, ix) = ndi.distance_transform_edt(~fillable, return_indices=True)
            sy, sx = np.nonzero(seam)
            lyr[sy, sx] = still[iy[sy, sx], ix[sy, sx]]
            lyr[sy, sx, 3] = 255
        layers[n] = lyr
    return layers, masks


def place_layer(layer, view):
    h, w = layer.shape[:2]
    cell = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    ox = CELL_W // 2 - RIG[view]["body_x"]
    oy = FOOT_Y - (h - 1)
    y0, x0 = max(0, -oy), max(0, -ox)
    y1, x1 = min(h, CELL_H - oy), min(w, CELL_W - ox)
    cell[y0 + oy:y1 + oy, x0 + ox:x1 + ox] = layer[y0:y1, x0:x1]
    return cell, (ox, oy)


# ---------------------------------------------------------------- 4. posing
def wave(f, amp, cycles=1, phase=0.0):
    """0 at frame 0 and at the loop end."""
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def pose(action, f):
    """Per-part [rot deg (+ raises a wing / tail tip), fwd px, up px]; global [fwd px, up px, lean deg]."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        breath = max(0.0, wave(f, 1.0, 1, -math.pi / 2) / 2.0)          # 0 -> 1 -> 0
        P["body"] = [0, 0, round(breath)]
        P["head"] = [0, 0, round(breath + wave(f, 1.0, 1, -1.2) / 1.6)]
        P["arms"] = [0, 0, round(breath)]
        P["wings"] = [wave(f, 5.0, 1, -0.4), 0, round(breath)]
        P["tail"] = [wave(f, 3.0, 1, -1.0), 0, 0]
        if f in (2, 3, 4, 5, 10, 11, 12, 13):
            fx.append(("ember", f))
    elif action == "walk":
        s1, c1 = math.sin(2 * math.pi * f / N), math.cos(2 * math.pi * f / N)
        up = 2.0 * abs(s1)
        P["body"] = [0, 0, up]
        P["head"] = [0, 0, up + (1 if math.sin(2 * 2 * math.pi * f / N - 0.6) > 0.3 else 0)]
        lift = 3.0 * abs(math.sin(2 * 2 * math.pi * f / N))
        P["foot_a"] = [0, 3.0 * s1, lift if c1 > 0 else 0.0]      # swings forward (lifted) while c1 > 0
        P["foot_b"] = [0, -3.0 * s1, lift if c1 < 0 else 0.0]
        P["arm_a"] = [0, -2.0 * s1, up]
        P["arm_b"] = [0, 2.0 * s1, up]
        P["wings"] = [wave(f, 8.0, 2, 0.0), 0, up]
        P["tail"] = [wave(f, 4.0, 1, 0.0), 0, 0]
        if f in (4, 12):
            fx.append(("dust", f))
    elif action == "attack":
        # flamethrower: 0-3 rear back + inhale, 4 snap forward, 5-10 fire, 11-15 recover
        g_f = [0, -2, -4, -5, 2, 6, 8, 9, 9, 8, 7, 5, 3, 2, 1, 0][f]
        g_u = [0, 1, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -3, -5, -7, 0, 5, 7, 8, 8, 7, 6, 4, 3, 1, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["head"] = [0, [0, -1, -2, -3, 1, 3, 4, 4, 4, 4, 3, 2, 1, 1, 0, 0][f], [0, 1, 2, 3, 1, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0][f]]
        P["arms"] = [0, [0, -1, -1, -2, 0, 2, 3, 3, 3, 3, 2, 2, 1, 0, 0, 0][f], [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["foot_a"] = [0, [0, 0, 0, 0, 2, 4, 5, 5, 5, 5, 4, 3, 2, 1, 0, 0][f], [0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["wings"] = [[0, 7, 14, 20, 10, -5, -11, -9, -11, -9, -7, -4, -2, -1, 0, 0][f], 0, [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["tail"] = [[0, 4, 8, 11, 4, -4, -7, -6, -7, -6, -5, -3, -2, -1, 0, 0][f], 0, 0]
        if 1 <= f <= 3:
            fx.append(("charge", f))
        if 4 <= f <= 11:
            fx.append(("fire", f))
        if 11 <= f <= 13:
            fx.append(("smoke", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0
        P["g"] = [-6.0 * env + jit, 1.0 * env, -6.0 * env]
        P["head"] = [0, -1.5 * env, 1.5 * env]
        P["arms"] = [0, -1.0 * env, 2.0 * env]
        P["wings"] = [15.0 * env, 0, 0]
        P["tail"] = [-7.0 * env, 0, 0]
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 7:
            fx.append(("embers", f))
    return P, fx


def up4(a):
    return np.repeat(np.repeat(a, UP, 0), UP, 1)


def rot4(img4, ang, piv):
    if abs(ang) < 0.3:
        return img4
    im = Image.fromarray(img4, "RGBA").rotate(-ang, resample=Image.NEAREST, center=(piv[0] * UP + UP / 2, piv[1] * UP + UP / 2))
    return np.array(im)


def shift4(img4, dx, dy):
    dx, dy = int(round(dx * UP)), int(round(dy * UP))
    if not dx and not dy:
        return img4
    out = np.zeros_like(img4)
    h, w = img4.shape[:2]
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = img4[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def compose(view, cells, piv, action, f):
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    zero = [0.0, 0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    feet_names = [n for n in R["draw"] if n.startswith("foot")]
    arm_names = [n for n in R["draw"] if n.startswith("arm")]
    offs = {}
    b = P.get("body", zero)
    for n in R["draw"]:
        rot, df, du = 0.0, 0.0, 0.0
        if n == "body":
            df, du = b[1], b[2]
        elif n == "head":
            h = P.get("head", zero); df, du = h[1], h[2] + (b[2] if action == "walk" else 0)
        elif n.startswith("arm"):
            key = ("arm_a" if n == arm_names[0] else "arm_b") if action == "walk" else "arms"
            a = P.get(key, zero)
            df, du = a[1], a[2]
        elif n.startswith("foot"):
            key = "foot_a" if n == feet_names[0] else "foot_b"
            ft = P.get(key, zero); df, du = ft[1], ft[2]
        elif n.startswith("wing"):
            w = P.get("wings", zero)
            rot = w[0] * R["sgn"][n]
            du = w[2]
        elif n == "tail":
            t = P.get("tail", zero)
            rot = t[0] * R["sgn"]["tail"]
            du = b[2] * 0.5
        dx, dy = df * fwx, df * fwy - du
        if n.startswith("foot") and abs(fwx) < 0.5:
            dx, dy = 0.0, -du                             # front / back: steps are vertical only
        offs[n] = (dx, dy, rot)
        l4 = rot4(up4(cells[n]), rot, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.4
    do_lean = abs(lean) > 0.3 and abs(fwx) > 0.3
    if do_lean:
        canvas = rot4(canvas, lean * side, (CELL_W // 2, FOOT_Y))
    canvas = shift4(canvas, gf * fwx, gf * fwy - gu)

    def track(pt, part):
        """cell position of a rest-pose point that rides on `part`."""
        dx, dy, rot = offs.get(part, (0, 0, 0))
        q = rot_pt(pt, rot, piv[part]) if part in piv and abs(rot) >= 0.3 else pt
        q = (q[0] + dx, q[1] + dy)
        if do_lean:
            q = rot_pt(q, lean * side, (CELL_W // 2, FOOT_Y))
        return (q[0] + gf * fwx, q[1] + gf * fwy - gu)

    out = canvas[1::UP, 1::UP].copy()
    return out, fx, track, (gf * fwx, gf * fwy - gu)


def rot_pt(p, ang, piv):
    """rotate a point (+ = clockwise on screen, y down) about piv."""
    t = math.radians(ang)
    dx, dy = p[0] - piv[0], p[1] - piv[1]
    return (piv[0] + dx * math.cos(t) - dy * math.sin(t), piv[1] + dx * math.sin(t) + dy * math.cos(t))


# ---------------------------------------------------------------- effects (fire, reserved colours only)
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 2 <= x < a.shape[1] - 2 and 2 <= y < a.shape[0] - 2:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def blob(a, cx, cy, r, col):
    ri = int(math.ceil(r))
    for dy in range(-ri, ri + 1):
        for dx in range(-ri, ri + 1):
            if math.hypot(dx, dy) <= r:
                put(a, cx + dx, cy + dy, col)


def fire_stream(a, mouth, fwd, length, k):
    """flame cone from the mouth along fwd: narrow at the mouth, billowing at the far end, flickers per frame.
    Drawn as three nested silhouettes (red-orange rim, orange body, pale-yellow core), never as ringed discs."""
    fwx, fwy = fwd
    L0 = math.hypot(fwx, fwy) or 1.0
    ux, uy = fwx / L0, fwy / L0
    px, py = -uy, ux
    rng = np.random.default_rng(600 + k)
    steps = max(3, int(length / 1.5))
    pts = []
    for j in range(steps + 1):
        s = length * j / steps
        r = 1.8 + s * 0.2 + rng.uniform(-0.5, 0.7)
        wob = math.sin(s * 0.42 + k * 1.9) * (0.5 + s * 0.05)
        pts.append((mouth[0] + ux * s + px * wob, mouth[1] + uy * s + py * wob, r, s / max(1.0, length)))
    for x, y, r, u in pts:
        blob(a, x, y, r, FX_LO)
    for x, y, r, u in pts:
        if u < 0.9:
            blob(a, x, y, r * 0.66 - 0.4, FX_MID)
    for x, y, r, u in pts:
        if u < 0.68:
            blob(a, x, y, r * 0.34 - 0.3, FX_HI)
    for j in range(6):                                  # sparks thrown off the sides
        s = rng.uniform(0.3, 1.08) * length
        o = rng.choice([-1, 1]) * (3.0 + s * 0.24 + rng.uniform(0, 3))
        put(a, mouth[0] + ux * s + px * o, mouth[1] + uy * s + py * o, FX_HI if j % 2 else FX_MID)


def draw_fx(a, fx, mouth, fwd, centre, tip):
    fwx, fwy = fwd
    foreshort = 1.0 - 0.38 * abs(fwy)                   # toward / away from the camera reads shorter
    for kind, k in fx:
        if kind == "ember" and tip:
            rise = {2: 1, 3: 3, 4: 6, 5: 9, 10: 1, 11: 3, 12: 6, 13: 9}[k]
            off = -3 if k < 8 else 3
            put(a, tip[0] + off, tip[1] - 2 - rise, FX_MID if rise < 9 else FX_LO)
            if rise < 6:
                put(a, tip[0] + off, tip[1] - 3 - rise, FX_HI)
        elif kind == "dust":
            for j in range(3):
                put(a, centre[0] - 16 - 3 * j, FOOT_Y - j, FX_HI)
                put(a, centre[0] + 16 + 3 * j, FOOT_Y - j, FX_HI)
        elif kind == "charge":
            for j in range(4 + k):                      # sparks drawn in toward the mouth
                ang = j * 2.3 + k * 0.8
                r = 11 - 2.5 * (k - 1)
                x0, y0 = mouth[0] + math.cos(ang) * (r + 3), mouth[1] + math.sin(ang) * (r + 3)
                put(a, x0, y0, FX_LO)
                put(a, mouth[0] + math.cos(ang) * r, mouth[1] + math.sin(ang) * r, FX_MID)
            blob(a, mouth[0] + fwx * 1.5, mouth[1] + fwy * 1.5, 1.0 + 0.7 * k, FX_MID)
            blob(a, mouth[0] + fwx * 1.5, mouth[1] + fwy * 1.5, 0.3 + 0.4 * k, FX_HI)
        elif kind == "fire":
            length = {4: 10, 5: 24, 6: 38, 7: 46, 8: 48, 9: 46, 10: 36, 11: 18}[k] * foreshort
            fire_stream(a, (mouth[0] + fwx * 2, mouth[1] + fwy * 2), fwd, length, k)
        elif kind == "smoke":
            rng = np.random.default_rng(900 + k)
            for j in range(7 - (k - 11) * 2):
                s = rng.uniform(8, 44) * foreshort
                o = rng.uniform(-7, 7)
                x, y = mouth[0] + fwx * s - fwy * o, mouth[1] + fwy * s + fwx * o - (k - 10) * 2
                put(a, x, y, FX_MID if j % 2 else FX_LO)
                if k == 11:
                    put(a, x, y - 1, FX_HI)
        elif kind == "embers":
            for j in range(5):
                ang = j * 1.7 + k * 0.9
                cx, cy = centre[0] + math.cos(ang) * 26, centre[1] - 8 + math.sin(ang) * 18 - (k - 2)
                put(a, cx, cy, FX_LO)
                put(a, cx, cy - 1, FX_HI)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = FX_HI
    return out


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    h.update(SRC_SHEET.read_bytes())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    work = OUT_DIR / "_work"
    h = state_hash()
    st_p = work / "state.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {}
    if args.force or st.get("hash") != h:
        if work.exists():
            shutil.rmtree(work)
        st = {"hash": h, "done": []}
    work.mkdir(parents=True, exist_ok=True)
    save_st = lambda: st_p.write_text(json.dumps(st))

    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    if "stills" not in st["done"]:
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        make_stills(raw)
        run_tool(["build", "--input", str(raw), "--colors", "30", "--reserve", "#%02x%02x%02x" % FX_HI, "#%02x%02x%02x" % FX_MID, "#%02x%02x%02x" % FX_LO,
                  "--balance", "--name", "charizard-mega-y", "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}

    rig = {}
    for v in RIG:
        layers, masks = build_rig(v, still[v])
        cells, piv = {}, {}
        for n, lyr in layers.items():
            cells[n], (ox, oy) = place_layer(lyr, v)
        for n, p in RIG[v]["pivot"].items():
            piv[n] = (p[0] + ox, p[1] + oy)
        rig[v] = (cells, piv, (ox, oy))

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        cells, piv, (ox, oy) = rig[v]
        R = RIG[v]
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, track, (gx, gy) = compose(v, cells, piv, act, f)
                centre = (CELL_W / 2 + gx, FOOT_Y - 42 + gy)
                tip = track((R["tip"][0] + ox, R["tip"][1] + oy), "tail")
                mouth = track((R["mouth"][0] + ox, R["mouth"][1] + oy), "head")
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, mouth, R["fwd"], centre, tip)
                Image.fromarray(a, "RGBA").save(od / f"{f:02d}.png")
            st["done"].append(key); save_st()
            print("frames", key, flush=True)

    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, N * CELL_W, 4), np.uint8)
        for di, d in enumerate(DIRS):
            v, mirror = DIR_SRC[d]
            for f in range(N):
                a = np.array(Image.open(frames_dir / v / act / f"{f:02d}.png").convert("RGBA"))
                if mirror:
                    a = a[:, ::-1]
                sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = a
        sheets[act] = sh
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for act in ACTIONS:
        im = Image.fromarray(sheets[act], "RGBA")
        im.save(OUT_DIR / f"{act}.png"); im.save(OUT_DIR / f"{act}_8dir.png")
        Image.fromarray(sheets[act][:CELL_H], "RGBA").save(OUT_DIR / f"{act}_S_strip.png")
    master = np.concatenate([sheets[a] for a in ACTIONS], axis=0)
    Image.fromarray(master, "RGBA").save(OUT_DIR / "sheet.png")
    Image.fromarray(master, "RGBA").save(OUT_DIR / "all_8dir_actions.png")
    pv = np.full((8 * CELL_H, 16 * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, fc in enumerate((0, 4, 8, 12) if act != "attack" else (2, 5, 8, 11)):
            src = sheets[act][:, fc * CELL_W:(fc + 1) * CELL_W]
            mm = src[..., 3] > 0
            pv[:, (ai * 4 + si) * CELL_W:(ai * 4 + si + 1) * CELL_W][mm] = src[mm]
    Image.fromarray(pv, "RGBA").save(OUT_DIR / "preview.png")

    idle0 = [sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W] for di in range(8)]
    Image.fromarray(np.concatenate(idle0, axis=1), "RGBA").save(FORM_DIR / "8dir_stills.png")

    # 5. QC
    qa = work / "qa"; qa.mkdir(exist_ok=True)
    pj = json.loads((lock_dir / "palette.json").read_text())
    cols = set()
    for c in pj.get("colors", []):
        hx = (c["hex"] if isinstance(c, dict) else c).lstrip("#")
        cols.add((int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)))
    aa = master[..., 3] > 0
    used = {tuple(int(v) for v in c) for c in np.unique(master[aa][:, :3], axis=0)}
    off = sorted(used - cols)
    alpha_bad = int(((master[..., 3] > 0) & (master[..., 3] < 255)).sum())
    clip = {}
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(N):
                c = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] > 0
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    # still diff: frame 0 (and the loop end of idle / walk = frame 0 again) of every action vs the placed still
    still_diff = {}
    for v in RIG:
        ref, _ = place_layer(still[v], v)
        for act in ACTIONS:
            a0 = np.array(Image.open(frames_dir / v / act / "00.png").convert("RGBA"))
            d = int((np.abs(a0.astype(int) - ref.astype(int)).sum(axis=2) > 0).sum())
            if d:
                still_diff[f"{v}/{act}"] = d
    s_h = int(np.ptp(np.nonzero(still["S"][..., 3].any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off), "alpha_partial": alpha_bad,
              "edge_touching_frames": clip, "still_diff": still_diff, "S_still_height": s_h}
    if not args.skip_qa and GAIT_LOOP.exists():
        for act, kind in (("idle", "idle"), ("walk", "gait")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run([str(FPY), "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / "S" / act), "--fps", "13",
                                "--output-dir", str(od), "--kind", kind, "--no-aids"], capture_output=True, text=True)
            rep = od / "loop-report.json"
            report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation") if rep.exists() else (r.stderr.strip().splitlines() or ["no report"])[-1]
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    bounds = []
    for di in range(8):
        c = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W, 3] > 0
        ys, xs = np.nonzero(c)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    meta = {"generator": "forge_gen1_0006_charizard_mega_y_actions", "version": VERSION, "key": KEY, "dex": 6,
            "source_art": ["0006_charizard-mega-y.png (frame 0 of S / SW / W / NE / N, x0.5)"], "made_from": ["0006_charizard-mega-y.png"],
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(bounds[3]), "frames": N, "frame_time": AD,
            "directions": DIRS, "direction_sources": DIR_SRC, "actions": ACTIONS, "palette_colors": len(cols),
            "off_palette_colors": len(off), "idle_bounds_S_to_SW": bounds}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))

    if not args.no_sync and SYNC_DIR.parent.exists():
        if SYNC_DIR.exists():
            shutil.rmtree(SYNC_DIR)
        SYNC_DIR.mkdir(parents=True)
        for fp in OUT_DIR.iterdir():
            if fp.is_file():
                shutil.copy2(fp, SYNC_DIR / fp.name)
        print("synced ->", SYNC_DIR)


if __name__ == "__main__":
    main()
