#!/usr/bin/env python3
"""Mega Charizard X (0006_charizard-mega-x): 8-direction x 4-action sheet (gen1 forge batch, md/gen1-forge.md).

Same flow as tools/forge_raichu_mega_x_actions.py (structure only; rig, colours and attack are this form's own):
  1. The official front.png / back.png (x8 nearest -> art/ref_front.png, art/ref_back.png) were given to the
     Antigravity CLI (agy, image generation) with art/agy_prompt.md (no proper nouns, exact part counts):
       art/out/keys_2x2.png  : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
       art/out/sw_single.png : SW (3/4 front, faces down-left)
     The other three are mirrors: SE = SW, E = W, NW = NE.
  2. Key the white backdrop (border-connected white + large enclosed gaps between wings / tail / body; small white
     areas = claws and teeth stay), area-downscale. S is scaled so its height = the official front.png opaque height
     (82 px); W / NE / N share the grid scale; SW is matched to the same body height.
  3. agent-sprite-forge palette_tool: build (30 colours + 2 reserved flame-FX colours) -> lock -> apply.
  4. Anatomical rig per view: two bat wings (pivot at the shoulder), head + neck + horns + mouth flames, two clawed
     arms, torso, two legs, one long tail with the flame tip (pivot at the rump). Posed at 4x (nearest), back to 1x.
     Actions (16 frames): idle (breath, slow wing beat, tail sway, embers off the tail flame), walk (alternating
     stomps, arm swing, wing beat, tail sway), attack (blue-flame dragon claw: wings rise and the claw draws back ->
     lunge, wings sweep down, claw rakes forward with three blue flame slashes -> flame burst -> recover),
     hurt (white flash, recoil, wings flare, embers, recover).
  5. QC: off-palette, edge-touch, still diff (frame 0 of every action = the rest pose), gait_loop.py idle / walk.

The official front / back png / gif are never touched. Writes 8dir_stills.png and agent-sprite-forge/ in the form
folder and syncs agent-sprite-forge/ (files only, not _work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0006_charizard_mega_x_actions.py [--force]
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
KEY = "0006_charizard-mega-x"
REL = "monsterSprite/gen1/0006_charizard/mega/charizard-mega-x"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0006_charizard_mega_x_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
OFFICIAL_H = 82                      # opaque height of the official front.png
GRID_H_S = 355                       # S body height in the generated grid (px)
SW_H = 691                           # SW body height in sw_single.png (px)
CELL_W, CELL_H, FOOT_Y = 232, 140, 128
UP = 4
FX_HI, FX_LO = (232, 250, 255), (40, 176, 236)
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# label: later polygons win. draw: back to front. sgn: rotation sign that RAISES a wing / tail tip.
# punch = the attacking arm, fwd = screen direction of forward, tip = tail-flame point (embers).
RIG = {
    "S": dict(
        poly={"wing_l": [(0, 0), (46, 0), (48, 24), (50, 32), (44, 38), (36, 38), (34, 54), (0, 54)],
              "wing_r": [(82, 0), (124, 0), (124, 26), (112, 26), (110, 40), (100, 54), (92, 54), (88, 40), (80, 32), (80, 22)],
              "tail": [(84, 56), (112, 26), (128, 26), (128, 72), (100, 72), (84, 66)],
              "body": [(50, 28), (80, 28), (84, 40), (84, 62), (76, 66), (54, 66), (48, 60), (48, 40)],
              "foot_l": [(38, 58), (62, 58), (62, 82), (36, 82)],
              "foot_r": [(68, 58), (96, 58), (96, 82), (68, 82)],
              "head": [(47, 0), (81, 0), (81, 26), (74, 31), (54, 31), (47, 26)],
              "arm_l": [(34, 36), (50, 32), (54, 40), (50, 50), (34, 50)],
              "arm_r": [(78, 32), (92, 36), (94, 52), (80, 52), (76, 42)]},
        label=["wing_l", "wing_r", "tail", "body", "foot_l", "foot_r", "head", "arm_l", "arm_r"],
        draw=["wing_l", "wing_r", "tail", "foot_l", "foot_r", "body", "head", "arm_l", "arm_r"],
        pivot={"wing_l": (50, 30), "wing_r": (80, 30), "tail": (84, 62), "head": (64, 30), "arm_l": (50, 36),
               "arm_r": (80, 36), "body": (64, 66), "foot_l": (50, 80), "foot_r": (81, 80)},
        sgn={"wing_l": 1, "wing_r": -1, "tail": -1},
        asgn=1, punch="arm_l", body_x=64, fwd=(0.0, 1.0), tip=(120, 30), back=False),
    "SW": dict(
        poly={"wing": [(38, 0), (92, 0), (92, 32), (82, 32), (78, 50), (66, 56), (50, 46), (40, 36), (38, 20)],
              "tail": [(46, 62), (70, 56), (80, 32), (92, 32), (92, 64), (66, 70), (52, 72)],
              "body": [(16, 22), (40, 20), (48, 34), (52, 52), (48, 66), (24, 68), (16, 52), (12, 36)],
              "foot_a": [(4, 56), (32, 58), (32, 82), (2, 82)],
              "foot_b": [(38, 62), (64, 62), (64, 82), (38, 82)],
              "head": [(0, 0), (38, 0), (38, 20), (28, 30), (14, 36), (0, 34)],
              "arm_far": [(0, 38), (20, 38), (22, 54), (0, 56)],
              "arm_near": [(28, 34), (44, 34), (44, 50), (28, 50)]},
        label=["wing", "tail", "body", "foot_a", "foot_b", "head", "arm_far", "arm_near"],
        draw=["wing", "tail", "foot_b", "body", "foot_a", "head", "arm_near", "arm_far"],
        pivot={"wing": (40, 28), "tail": (50, 64), "head": (22, 30), "arm_far": (20, 42), "arm_near": (36, 36),
               "body": (32, 68), "foot_a": (18, 80), "foot_b": (50, 80)},
        sgn={"wing": -1, "tail": -1},
        asgn=1, punch="arm_far", body_x=32, fwd=(-0.9, 0.45), tip=(86, 36), back=False),
    "W": dict(
        poly={"wing": [(42, 0), (96, 0), (96, 48), (70, 56), (52, 46), (42, 40), (38, 20)],
              "tail": [(36, 58), (60, 54), (100, 58), (110, 30), (129, 30), (129, 74), (100, 70), (56, 78), (40, 72)],
              "body": [(18, 28), (44, 28), (46, 50), (44, 66), (30, 70), (18, 62), (14, 46)],
              "foot_a": [(4, 56), (24, 56), (26, 70), (30, 82), (4, 82)],
              "foot_b": [(36, 64), (62, 62), (62, 82), (36, 82)],
              "head": [(0, 0), (42, 0), (40, 20), (30, 32), (16, 34), (0, 32)],
              "arm_far": [(2, 38), (22, 40), (24, 54), (2, 56)],
              "arm_near": [(26, 38), (42, 40), (44, 54), (26, 54)]},
        label=["wing", "tail", "body", "foot_a", "foot_b", "head", "arm_far", "arm_near"],
        draw=["wing", "tail", "foot_b", "body", "foot_a", "head", "arm_near", "arm_far"],
        pivot={"wing": (44, 30), "tail": (42, 64), "head": (24, 30), "arm_far": (22, 44), "arm_near": (34, 40),
               "body": (30, 70), "foot_a": (19, 80), "foot_b": (48, 80)},
        sgn={"wing": -1, "tail": -1},
        asgn=1, punch="arm_far", body_x=30, fwd=(-1.0, 0.0), tip=(122, 34), back=False),
    "NE": dict(
        poly={"wing_l": [(4, 0), (54, 0), (68, 24), (60, 34), (44, 50), (18, 48), (4, 30)],
              "wing_r": [(90, 0), (141, 0), (141, 48), (120, 50), (100, 40), (92, 30), (86, 24)],
              "tail": [(0, 28), (16, 28), (20, 54), (40, 62), (70, 66), (80, 74), (76, 80), (40, 80), (14, 70), (0, 58)],
              "body": [(42, 26), (92, 26), (100, 44), (96, 70), (90, 86), (52, 86), (40, 70), (36, 50)],
              "head": [(50, 0), (92, 0), (94, 22), (86, 30), (70, 32), (62, 26), (50, 22)],
              "foot_l": [(36, 77), (54, 77), (54, 84), (36, 84)],
              "foot_r": [(74, 79), (96, 79), (96, 86), (74, 86)],
              "arm_r": [(84, 48), (100, 48), (100, 64), (84, 64)]},
        label=["wing_l", "wing_r", "tail", "body", "head", "foot_l", "foot_r", "arm_r"],
        draw=["foot_l", "foot_r", "body", "head", "arm_r", "wing_l", "wing_r", "tail"],
        pivot={"wing_l": (56, 30), "wing_r": (90, 28), "head": (80, 30), "tail": (56, 70), "body": (68, 86),
               "foot_l": (45, 82), "foot_r": (84, 84), "arm_r": (88, 52)},
        sgn={"wing_l": 1, "wing_r": -1, "tail": 1},
        asgn=-1, punch="arm_r", body_x=68, fwd=(0.9, -0.45), tip=(6, 34), back=True),
    "N": dict(
        poly={"wing_l": [(0, 0), (50, 0), (56, 26), (52, 34), (40, 46), (30, 56), (0, 44)],
              "wing_r": [(82, 0), (132, 0), (132, 44), (104, 56), (92, 46), (80, 34), (76, 26)],
              "body": [(44, 26), (88, 26), (94, 50), (92, 70), (86, 84), (46, 84), (40, 70), (38, 50)],
              "head": [(50, 0), (82, 0), (82, 22), (74, 30), (58, 30), (50, 22)],
              "foot_l": [(38, 77), (58, 77), (58, 84), (38, 84)],
              "foot_r": [(74, 77), (94, 77), (94, 84), (74, 84)],
              "arm_l": [(34, 46), (48, 46), (48, 60), (34, 60)],
              "arm_r": [(84, 46), (98, 46), (98, 60), (84, 60)],
              "tail": [(60, 60), (90, 64), (100, 60), (100, 38), (118, 38), (116, 64), (104, 78), (84, 80), (62, 78), (56, 70)]},
        label=["wing_l", "wing_r", "body", "head", "foot_l", "foot_r", "arm_l", "arm_r", "tail"],
        draw=["foot_l", "foot_r", "body", "head", "arm_l", "arm_r", "wing_l", "wing_r", "tail"],
        pivot={"wing_l": (54, 30), "wing_r": (78, 30), "head": (66, 30), "tail": (78, 66), "body": (66, 84),
               "foot_l": (48, 82), "foot_r": (84, 82), "arm_l": (46, 48), "arm_r": (86, 48)},
        sgn={"wing_l": 1, "wing_r": -1, "tail": -1},
        asgn=-1, punch="arm_r", body_x=66, fwd=(0.0, -1.0), tip=(110, 42), back=True),
}


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
def key_white(rgb):
    a = rgb.astype(int)
    mn = a.min(axis=2)
    white = mn > 225
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    means = ndi.mean(mn, lab, range(1, n + 1))
    # enclosed gaps (between wings, tail and body) are big or pure white; claws / teeth are small light grey
    holes = [i + 1 for i in range(n) if (i + 1) not in border and (sizes[i] >= 2500 or (sizes[i] >= 150 and means[i] > 252))]
    bg |= np.isin(lab, holes)
    bg |= ndi.binary_dilation(bg, iterations=2) & (mn > 200)
    out = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    lab2, n2 = ndi.label(out[..., 3] > 0)
    if n2:
        sz = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 200])] = 0
    return out


def make_stills(dst: Path):
    k = np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], "W": k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": np.array(Image.open(ART / "out/sw_single.png").convert("RGB"))}
    scale = {v: OFFICIAL_H / GRID_H_S for v in src}
    scale["SW"] = OFFICIAL_H / SW_H
    dst.mkdir(parents=True, exist_ok=True)
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb), "RGBA")
        im = im.crop(im.getbbox())
        sm = im.resize((round(im.width * scale[v]), round(im.height * scale[v])), Image.BOX)
        a = np.array(sm)
        a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
        a[a[..., 3] == 0] = 0
        Image.fromarray(a, "RGBA").save(dst / f"{v}.png")


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
        if n == "body" and m.any():
            # underlay for everything drawn above the body inside the body + head outline (arms over the belly),
            # so no hole opens when an arm moves
            bh = m | masks.get("head", np.zeros_like(m))
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(bh, 8), iterations=8))[8:-8, 8:-8]
            seam |= above & ~m & inside & ~masks.get("head", np.zeros_like(m))
            src_m = bh
        if seam.any() and m.any():
            fillable = src_m & (bright > 70)
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
        P["head"] = [0, 0, round(wave(f, 1.0, 1, -1.2) / 1.6 + 0.0)]
        P["arms"] = [0, 0, round(breath)]
        P["wings"] = [wave(f, 4.0, 1, -0.4), 0, round(breath)]
        P["tail"] = [wave(f, 3.0, 1, -1.0), 0, 0]
        if f in (3, 4, 5, 11, 12, 13):
            fx.append(("ember", f))
    elif action == "walk":
        s1, c1 = math.sin(2 * math.pi * f / N), math.cos(2 * math.pi * f / N)
        up = 2.0 * abs(s1)
        P["body"] = [0, 0, up]
        P["head"] = [0, 0, up + (1 if math.sin(2 * 2 * math.pi * f / N - 0.6) > 0.3 else 0)]
        lift = 2.5 * abs(math.sin(2 * 2 * math.pi * f / N))
        P["foot_a"] = [0, 3.0 * s1, lift if c1 > 0 else 0.0]      # swings forward (lifted) while c1 > 0
        P["foot_b"] = [0, -3.0 * s1, lift if c1 < 0 else 0.0]
        P["arm_a"] = [0, -2.0 * s1, up]
        P["arm_b"] = [0, 2.0 * s1, up]
        P["wings"] = [wave(f, 7.0, 2, 0.0), 0, up]
        P["tail"] = [wave(f, 4.0, 1, 0.0), 0, 0]
        if f in (4, 12):
            fx.append(("dust", f))
    elif action == "attack":
        g_f = [0, -2, -3, -4, 3, 7, 10, 11, 11, 10, 8, 6, 4, 2, 1, 0][f]
        g_u = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -2, -4, -5, 2, 5, 7, 7, 6, 5, 4, 3, 2, 1, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["head"] = [0, [0, -1, -1, -2, 0, 1, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0][f], [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        # dragon claw: the arm swings up about the shoulder (wind-up), then rakes down and forward
        P["punch"] = [[0, 18, 32, 40, 24, 0, -14, -20, -20, -16, -11, -7, -4, -2, -1, 0][f],
                      [0, -1, -2, -2, 1, 3, 5, 6, 6, 5, 4, 3, 2, 1, 0, 0][f], 0]
        P["guard"] = [0, [0, 1, 1, 2, 0, -1, -2, -2, -2, -2, -1, -1, 0, 0, 0, 0][f], 0]
        P["foot_a"] = [0, [0, 0, 0, 0, 2, 4, 5, 5, 5, 5, 4, 3, 2, 1, 0, 0][f], [0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["wings"] = [[0, 6, 12, 16, 8, -4, -10, -10, -8, -6, -4, -2, -1, 0, 0, 0][f], 0, 0]
        P["tail"] = [[0, 4, 8, 10, 2, -4, -6, -6, -5, -4, -3, -2, -1, 0, 0, 0][f], 0, 0]
        if 1 <= f <= 3:
            fx.append(("charge", f))
        if 5 <= f <= 8:
            fx.append(("slash", f))
        if f in (7, 8, 9):
            fx.append(("burst", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0
        P["g"] = [-6.0 * env + jit, 1.0 * env, -6.0 * env]
        P["head"] = [0, -1.5 * env, 1.5 * env]
        P["arms"] = [0, -1.0 * env, 2.0 * env]
        P["wings"] = [14.0 * env, 0, 0]
        P["tail"] = [-6.0 * env, 0, 0]
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
    claw = None
    feet_names = [n for n in R["draw"] if n.startswith("foot")]
    arm_names = [n for n in R["draw"] if n.startswith("arm")]
    offs = {}
    b = P.get("body", zero)
    for n in R["draw"]:
        rot, df, du = 0.0, 0.0, 0.0
        if n == "body":
            df, du = b[1], b[2]
        elif n == "head":
            h = P.get("head", zero); df, du = h[1], h[2] + (b[2] if action != "idle" else 0)
        elif n.startswith("arm"):
            if action == "attack":
                key = "punch" if n == R["punch"] else "guard"
            elif action == "walk":
                key = "arm_a" if n == arm_names[0] else "arm_b"
            else:
                key = "arms"
            a = P.get(key, zero)
            df, du = a[1], a[2]
            if key == "punch":
                rot = a[0] * R["asgn"]
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
        offs[n] = (dx, dy)
        if n == R["punch"]:
            ys, xs = np.nonzero(cells[n][..., 3] > 0)
            pv = piv[n]
            i = int(np.argmax((xs - pv[0]) ** 2 + (ys - pv[1]) ** 2))
            claw = rot_pt((xs[i], ys[i]), rot, pv)
            claw = (claw[0] + dx, claw[1] + dy)
        l4 = rot4(up4(cells[n]), rot, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.4
    if abs(lean) > 0.3 and abs(fwx) > 0.3:
        canvas = rot4(canvas, lean * side, (CELL_W // 2, FOOT_Y))
        if claw:
            claw = rot_pt(claw, lean * side, (CELL_W // 2, FOOT_Y))
    canvas = shift4(canvas, gf * fwx, gf * fwy - gu)
    if claw:
        claw = (claw[0] + gf * fwx, claw[1] + gf * fwy - gu)
    out = canvas[1::UP, 1::UP].copy()
    return out, fx, offs, (gf * fwx, gf * fwy - gu), claw


def rot_pt(p, ang, piv):
    """rotate a point (+ = clockwise on screen, y down) about piv."""
    t = math.radians(ang)
    dx, dy = p[0] - piv[0], p[1] - piv[1]
    return (piv[0] + dx * math.cos(t) - dy * math.sin(t), piv[1] + dx * math.sin(t) + dy * math.cos(t))


# ---------------------------------------------------------------- effects (blue flame, reserved colours only)
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 2 <= x < a.shape[1] - 2 and 2 <= y < a.shape[0] - 2:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def stroke(a, x0, y0, x1, y1, col, col2=None):
    k = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for j in range(k + 1):
        u = j / k
        x, y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
        put(a, x, y, col)
        if col2 is not None:
            put(a, x, y + 1, col2)


def flame_tongue(a, x, y, h, lean, seed):
    """small upward flame: pale core, blue rim."""
    rng = np.random.default_rng(seed)
    for j in range(h):
        w = max(0, (h - j) // 3)
        cx = x + lean * j / max(1, h) + rng.uniform(-0.6, 0.6)
        for dx in range(-w - 1, w + 2):
            put(a, cx + dx, y - j, FX_HI if abs(dx) <= w - 1 else FX_LO)


def draw_fx(a, fx, claw, fwd, centre, tip):
    fwx, fwy = fwd
    px, py = -fwy, fwx                                  # perpendicular
    for kind, k in fx:
        if kind == "ember" and tip:
            rise = {3: 2, 4: 5, 5: 8, 11: 2, 12: 5, 13: 8}[k]
            off = -2 if k < 8 else 3
            put(a, tip[0] + off, tip[1] - 3 - rise, FX_LO)
            if rise < 8:
                put(a, tip[0] + off, tip[1] - 4 - rise, FX_HI)
        elif kind == "dust":
            for j in range(3):
                put(a, centre[0] - 14 - 3 * j, FOOT_Y - j, FX_HI)
                put(a, centre[0] + 14 + 3 * j, FOOT_Y - j, FX_HI)
        elif kind == "charge" and claw:
            for j in range(3 + k):
                ang = j * 2.3 + k * 0.8
                r = 9 - 2 * (k - 1)
                x0, y0 = claw[0] + math.cos(ang) * (r + 4), claw[1] + math.sin(ang) * (r + 4)
                stroke(a, x0, y0, claw[0] + math.cos(ang) * r, claw[1] + math.sin(ang) * r, FX_LO)
            flame_tongue(a, claw[0], claw[1], 3 + k, 0, 5 + k)
        elif kind == "slash" and claw:
            # three parallel blue-flame claw streaks, diagonal across the forward direction
            sx, sy = px + 0.7 * fwx, py + 0.7 * fwy
            L0 = math.hypot(sx, sy) or 1.0
            sx, sy = sx / L0, sy / L0
            ox_, oy_ = -sy, sx                         # spacing between streaks
            L = {5: 10, 6: 18, 7: 20, 8: 16}[k]
            cx, cy = claw[0] + fwx * 5, claw[1] + fwy * 5
            for o in (-4, 0, 4):
                bx, by = cx + ox_ * o, cy + oy_ * o
                for s_ in range(L + 1):
                    u = s_ / L - 0.5
                    w = 1 if abs(u) < 0.3 and k < 8 else 0
                    x, y = bx + sx * u * L, by + sy * u * L
                    put(a, x, y, FX_HI if k < 8 else FX_LO)
                    if w:
                        put(a, x + ox_, y + oy_, FX_LO)
        elif kind == "burst" and claw:
            r = {7: 8, 8: 12, 9: 10}[k]
            cx, cy = claw[0] + fwx * 8, claw[1] + fwy * 6
            for j in range(10):
                ang = j * math.pi / 5 + k * 0.4
                r0 = r * 0.45
                stroke(a, cx + math.cos(ang) * r0, cy + math.sin(ang) * r0, cx + math.cos(ang) * r, cy + math.sin(ang) * r,
                       FX_HI if j % 2 == 0 else FX_LO)
            if k == 8:
                flame_tongue(a, cx, cy + 2, 6, 0, 81)
        elif kind == "embers":
            for j in range(4):
                ang = j * 1.7 + k * 0.9
                cx, cy = centre[0] + math.cos(ang) * 22, centre[1] - 6 + math.sin(ang) * 16 - (k - 2)
                put(a, cx, cy, FX_LO)
                put(a, cx, cy - 1, FX_HI)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = FX_HI
    return out


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    for n in ("out/keys_2x2.png", "out/sw_single.png"):
        h.update((ART / n).read_bytes())
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
        run_tool(["build", "--input", str(raw), "--colors", "30", "--reserve", "#%02x%02x%02x" % FX_HI, "#%02x%02x%02x" % FX_LO,
                  "--balance", "--name", "charizard-mega-x", "--output-dir", str(pal_dir)])
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
                a, fx, offs, (gx, gy), claw = compose(v, cells, piv, act, f)
                centre = (CELL_W / 2 + gx, FOOT_Y - 36 + gy)
                tip = None
                if R.get("tip"):
                    toff = offs.get("tail", (0, 0))
                    tip = (R["tip"][0] + ox + gx + toff[0], R["tip"][1] + oy + gy + toff[1])
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, claw, R["fwd"], centre, tip)
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
        for si, fc in enumerate((0, 4, 8, 12) if act != "attack" else (2, 5, 7, 10)):
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
              "edge_touching_frames": clip, "still_diff": still_diff, "S_still_height": s_h, "official_height": OFFICIAL_H}
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
    meta = {"generator": "forge_gen1_0006_charizard_mega_x_actions", "version": VERSION, "key": KEY, "dex": 6,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["front.png", "back.png"],
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
