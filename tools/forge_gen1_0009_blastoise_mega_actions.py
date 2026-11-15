#!/usr/bin/env python3
"""Mega Blastoise (0009_blastoise-mega): 8-direction x 4-action sheet (gen1 forge batch, md/gen1-forge.md).

Same flow as tools/forge_raichu_mega_x_actions.py (structure only; rig, colours and attack are this form's own):
  1. The official front.png / back.png (x8 nearest -> art/ref_front.png, art/ref_back.png) were given to the
     Antigravity CLI (agy, image generation) with art/agy_prompt.md (no proper nouns, exact part counts: 3 cannons):
       art/out/keys_2x2.png  : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
       art/out/sw_single.png : SW (3/4 front, faces down-left)
     1st try (art/try1) drew S and W half-turned; the retry (art/try2 = art/out) is used.
     The other three are mirrors: SE = SW, E = W, NW = NE.
  2. Key the white backdrop (border-connected white only: the white shell rim, cannon highlights and claws stay),
     area-downscale. S is scaled so its height = the official front.png opaque height (87 px); W / NE / N share the
     grid scale; SW is matched to the same height. N: two small grey knobs the generator added on the shoulders
     (not on the official art) are erased and re-outlined.
  3. agent-sprite-forge palette_tool: build (30 colours + 2 reserved water-FX colours) -> lock -> apply.
  4. Anatomical rig per view: shell, the big back cannon (pivot at its base on the shell), head, two arms with the
     forearm cannons, belly, two legs, small tail. Posed at 4x (nearest), sampled back to 1x.
     Actions (16 frames): idle (breath, head bob, cannon sway, drip from the arm cannons), walk (heavy alternating
     stomps, arm swing, tail sway, dust), attack (water cannon volley: brace + the big cannon lowers to aim + water
     gathers at the three muzzles -> fire: recoil, three water jets from the big cannon and both arm cannons ->
     splash burst -> recover), hurt (white flash, recoil, cannon kicks up, water spray, recover).
  5. QC: off-palette, edge-touch, still diff (frame 0 of every action = the rest pose), gait_loop.py idle / walk.

The official front / back png / gif are never touched. Writes 8dir_stills.png and agent-sprite-forge/ in the form
folder and syncs agent-sprite-forge/ (files only, not _work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0009_blastoise_mega_actions.py [--force]
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
KEY = "0009_blastoise-mega"
REL = "monsterSprite/gen1/0009_blastoise/mega/blastoise-mega"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0009_blastoise_mega_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
OFFICIAL_H = 87                      # opaque height of the official front.png
GRID_H_S = 372                       # S height in the generated grid (px)
SW_H = 757                           # SW height in sw_single.png (px)
CELL_W, CELL_H, FOOT_Y = 232, 136, 124
UP = 4
FX_HI, FX_LO = (236, 248, 255), (72, 168, 248)
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# label: later polygons win. draw: back to front. csgn: rotation sign that LOWERS the big cannon toward forward
# (0 = it points at / away from the camera). muzzles: part -> (muzzle point, jet direction). fwd = screen forward.
RIG = {
    "S": dict(
        poly={"shell": [(14, 14), (80, 14), (82, 50), (12, 50)],
              "cannon": [(36, 0), (58, 0), (58, 24), (36, 24)],
              "body": [(24, 40), (70, 40), (74, 82), (20, 82)],
              "foot_l": [(14, 62), (30, 62), (34, 74), (40, 80), (40, 87), (12, 87)],
              "foot_r": [(80, 62), (64, 62), (60, 74), (54, 80), (54, 87), (82, 87)],
              "head": [(32, 24), (62, 24), (62, 44), (54, 51), (40, 51), (32, 44)],
              "arm_l": [(0, 43), (28, 43), (30, 50), (30, 68), (0, 68)],
              "arm_r": [(94, 43), (66, 43), (64, 50), (64, 68), (94, 68)]},
        label=["shell", "cannon", "body", "foot_l", "foot_r", "head", "arm_l", "arm_r"],
        draw=["shell", "cannon", "foot_l", "foot_r", "body", "head", "arm_l", "arm_r"],
        pivot={"shell": (47, 48), "cannon": (47, 24), "head": (47, 49), "arm_l": (28, 52), "arm_r": (66, 52),
               "body": (47, 80), "foot_l": (27, 85), "foot_r": (67, 85)},
        csgn=0, body_x=47, fwd=(0.0, 1.0),
        muzzles={"cannon": ((47, 5), (0.0, -1.0)), "arm_l": ((4, 54), (-0.3, 1.0)), "arm_r": ((90, 54), (0.3, 1.0))}),
    "SW": dict(
        poly={"shell": [(42, 6), (72, 6), (82, 40), (80, 66), (74, 74), (60, 66), (46, 42), (28, 22), (42, 18)],
              "cannon": [(8, 0), (44, 0), (50, 8), (48, 20), (30, 19), (8, 16)],
              "tail": [(76, 64), (87, 64), (87, 79), (76, 79)],
              "body": [(28, 40), (56, 40), (62, 62), (58, 75), (36, 75), (28, 60)],
              "foot_a": [(22, 60), (36, 62), (44, 72), (44, 87), (22, 87)],
              "foot_b": [(56, 62), (78, 62), (80, 87), (54, 87)],
              "head": [(20, 22), (44, 22), (47, 36), (40, 45), (24, 46), (19, 40)],
              "arm_far": [(0, 41), (30, 41), (31, 63), (0, 63)],
              "arm_near": [(41, 43), (63, 43), (63, 63), (41, 63)]},
        label=["shell", "cannon", "tail", "body", "foot_a", "foot_b", "head", "arm_far", "arm_near"],
        draw=["shell", "tail", "foot_b", "body", "foot_a", "head", "cannon", "arm_near", "arm_far"],
        pivot={"shell": (55, 40), "cannon": (45, 16), "head": (36, 43), "arm_far": (28, 50), "arm_near": (50, 48),
               "body": (45, 74), "foot_a": (34, 85), "foot_b": (67, 85), "tail": (77, 72)},
        csgn=-1, body_x=45, fwd=(-0.9, 0.45),
        muzzles={"cannon": ((14, 5), (-0.94, -0.33)), "arm_far": ((2, 48), (-0.95, 0.3)), "arm_near": ((52, 53), (-0.9, 0.42))}),
    "W": dict(
        poly={"shell": [(28, 9), (60, 9), (74, 30), (74, 72), (56, 72), (42, 58), (30, 42), (24, 20)],
              "cannon": [(0, 0), (34, 0), (42, 8), (40, 20), (28, 18), (0, 15)],
              "tail": [(62, 62), (78, 62), (78, 77), (62, 77)],
              "body": [(18, 40), (42, 40), (44, 62), (40, 73), (22, 73), (18, 60)],
              "foot_a": [(16, 60), (40, 60), (42, 83), (16, 83)],
              "foot_b": [(40, 60), (64, 60), (64, 83), (40, 83)],
              "head": [(2, 19), (30, 19), (33, 36), (27, 45), (8, 45), (2, 36)],
              "arm_near": [(26, 40), (48, 40), (48, 62), (26, 62)],
              "arm_far": [(0, 43), (27, 43), (27, 62), (0, 62)]},
        label=["shell", "cannon", "tail", "body", "foot_a", "foot_b", "head", "arm_near", "arm_far"],
        draw=["shell", "tail", "foot_b", "body", "foot_a", "head", "cannon", "arm_near", "arm_far"],
        pivot={"shell": (50, 40), "cannon": (35, 14), "head": (24, 43), "arm_far": (24, 50), "arm_near": (38, 48),
               "body": (32, 72), "foot_a": (30, 81), "foot_b": (52, 81), "tail": (63, 70)},
        csgn=-1, body_x=32, fwd=(-1.0, 0.0),
        muzzles={"cannon": ((4, 4), (-0.94, -0.34)), "arm_far": ((2, 51), (-1.0, 0.0)), "arm_near": ((29, 50), (-1.0, 0.1))}),
    "NE": dict(
        poly={"shell": [(4, 8), (52, 8), (60, 20), (62, 60), (58, 72), (4, 72)],
              "head": [(50, 13), (72, 13), (72, 42), (60, 44), (50, 34)],
              "cannon": [(36, 0), (70, 0), (70, 12), (52, 18), (38, 16)],
              "foot_l": [(0, 64), (30, 64), (30, 85), (0, 85)],
              "foot_r": [(32, 62), (64, 62), (64, 85), (32, 85)],
              "arm_r": [(58, 43), (87, 43), (87, 67), (58, 67)]},
        label=["shell", "head", "cannon", "foot_l", "foot_r", "arm_r"],
        draw=["head", "foot_l", "foot_r", "shell", "arm_r", "cannon"],
        pivot={"shell": (34, 72), "cannon": (46, 13), "head": (56, 40), "arm_r": (60, 53),
               "foot_l": (12, 82), "foot_r": (46, 83)},
        csgn=1, body_x=40, fwd=(0.9, -0.45),
        muzzles={"cannon": ((65, 5), (0.95, -0.32)), "arm_r": ((86, 50), (1.0, -0.1))}),
    "N": dict(
        poly={"shell": [(14, 17), (76, 17), (76, 72), (53, 72), (53, 87), (37, 87), (37, 72), (14, 72)],
              "cannon": [(37, 0), (53, 0), (53, 18), (37, 18)],
              "foot_l": [(10, 64), (37, 64), (37, 87), (10, 87)],
              "foot_r": [(53, 64), (80, 64), (80, 87), (53, 87)],
              "arm_l": [(0, 37), (19, 37), (19, 61), (0, 61)],
              "arm_r": [(71, 37), (90, 37), (90, 61), (71, 61)]},
        label=["shell", "cannon", "foot_l", "foot_r", "arm_l", "arm_r"],
        draw=["foot_l", "foot_r", "arm_l", "arm_r", "shell", "cannon"],
        pivot={"shell": (45, 72), "cannon": (45, 17), "arm_l": (18, 46), "arm_r": (72, 46),
               "foot_l": (24, 85), "foot_r": (66, 85)},
        csgn=0, body_x=45, fwd=(0.0, -1.0),
        muzzles={"cannon": ((45, 1), (0.0, -1.0)), "arm_l": ((4, 44), (-0.2, -1.0)), "arm_r": ((86, 44), (0.2, -1.0))}),
}
# N: the generator drew two grey knobs on the shoulders that the official art does not have -> erased
N_ERASE = [[(10, 16), (25, 16), (25, 23), (21, 27), (17, 31), (10, 31)],
           [(80, 16), (65, 16), (65, 23), (69, 27), (73, 31), (80, 31)]]


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
def key_white(rgb):
    a = rgb.astype(int)
    mn = a.min(axis=2)
    lab, n = ndi.label(mn > 225)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    bg |= ndi.binary_dilation(bg, iterations=2) & (mn > 200)
    out = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    lab2, n2 = ndi.label(out[..., 3] > 0)
    if n2:
        sz = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 200])] = 0
    return out


def poly_mask(p, shape):
    m = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(m).polygon(p, fill=255)
    return np.array(m) > 0


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
        if v == "N":
            er = np.zeros(a.shape[:2], bool)
            for p in N_ERASE:
                er |= poly_mask(p, a.shape)
            er &= a[..., 3] > 0
            a[er] = 0
            op = a[..., 3] > 0
            edge = op & ndi.binary_dilation(er, iterations=1)
            a[edge, :3] = (16, 16, 16)
            # drop any 1-2 px crumbs left by the erase
            lab, n = ndi.label(a[..., 3] > 0)
            if n > 1:
                sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
                a[lab != int(np.argmax(sz)) + 1] = 0
        Image.fromarray(a, "RGBA").save(dst / f"{v}.png")


def run_tool(args):
    r = subprocess.run([str(FPY), "-I", str(PALETTE_TOOL)] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"palette_tool {args[0]} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


# ---------------------------------------------------------------- 3. rig
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
        cushion = 5 if n in ("shell", "body") else 3
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion)
        src_m = m
        if n in ("body", "shell") and m.any():
            # underlay for everything drawn above inside the closed outline of this part (+ head / body), so no hole
            # opens when an arm or the head moves
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(m | masks.get("head", np.zeros_like(m)), 8), iterations=8))[8:-8, 8:-8]
            seam |= above & ~m & inside & ~masks.get("head", np.zeros_like(m)) & ~masks.get("cannon", np.zeros_like(m))
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
    """Per-part [rot deg, fwd px, up px]; 'cannon' rot = degrees the big cannon is LOWERED toward forward.
    Global [fwd px, up px, lean deg]."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        breath = max(0.0, wave(f, 1.0, 1, -math.pi / 2) / 2.0)          # 0 -> 1 -> 0
        P["body"] = [0, 0, round(breath)]
        P["head"] = [0, 0, round(wave(f, 1.0, 1, -1.2) / 1.6)]
        P["arms"] = [0, 0, round(breath)]
        P["cannon"] = [wave(f, 2.5, 1, -0.6), 0, 0]
        P["tail"] = [wave(f, 6.0, 1, 0.0), 0, 0]
        if 6 <= f <= 10:
            fx.append(("drip", f))
    elif action == "walk":
        s1, c1 = math.sin(2 * math.pi * f / N), math.cos(2 * math.pi * f / N)
        up = 1.5 * abs(s1)
        P["body"] = [0, 0, up]
        P["head"] = [0, 0, up + (1 if math.sin(2 * 2 * math.pi * f / N - 0.6) > 0.3 else 0)]
        lift = 3.0 * abs(math.sin(2 * 2 * math.pi * f / N))
        P["foot_a"] = [0, 3.0 * s1, lift if c1 > 0 else 0.0]
        P["foot_b"] = [0, -3.0 * s1, lift if c1 < 0 else 0.0]
        P["arm_a"] = [0, -2.0 * s1, up]
        P["arm_b"] = [0, 2.0 * s1, up]
        P["cannon"] = [wave(f, 2.0, 2, 0.0), 0, 0]
        P["tail"] = [wave(f, 8.0, 1, 0.0), 0, 0]
        if f in (4, 12):
            fx.append(("dust", f))
    elif action == "attack":
        # water cannon volley: brace + aim (1-3), fire with recoil (4-9), splash (8-10), recover (10-15)
        g_f = [0, 1, 2, 2, -3, -4, -4, -3, -3, -2, -2, -1, -1, 0, 0, 0][f]
        g_u = [0, -1, -2, -2, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0][f]
        lean = [0, 2, 3, 4, -3, -4, -4, -3, -3, -2, -2, -1, -1, 0, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["head"] = [0, [0, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0][f], [0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        arm_f = [0, 2, 3, 3, 0, -1, -1, -1, 0, 0, 1, 1, 0, 0, 0, 0][f]
        P["arms"] = [0, arm_f, [0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0][f]]
        P["cannon"] = [[0, 8, 16, 20, 14, 16, 18, 18, 18, 16, 12, 8, 5, 2, 1, 0][f], 0, 0]
        P["tail"] = [[0, -4, -6, -6, 6, 8, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0][f], 0, 0]
        P["foot_a"] = [0, [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0][f], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        if 1 <= f <= 3:
            fx.append(("charge", f))
        if 4 <= f <= 9:
            fx.append(("jet", f))
        if f in (8, 9, 10):
            fx.append(("splash", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0
        P["g"] = [-6.0 * env + jit, 1.0 * env, -7.0 * env]
        P["head"] = [0, -1.5 * env, 1.5 * env]
        P["arms"] = [0, -1.0 * env, 2.0 * env]
        P["cannon"] = [-8.0 * env, 0, 0]
        P["tail"] = [-8.0 * env, 0, 0]
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 7:
            fx.append(("spray", f))
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


def rot_pt(p, ang, piv):
    """rotate a point (+ = clockwise on screen, y down) about piv."""
    t = math.radians(ang)
    dx, dy = p[0] - piv[0], p[1] - piv[1]
    return (piv[0] + dx * math.cos(t) - dy * math.sin(t), piv[1] + dx * math.sin(t) + dy * math.cos(t))


def rot_vec(v, ang):
    t = math.radians(ang)
    return (v[0] * math.cos(t) - v[1] * math.sin(t), v[0] * math.sin(t) + v[1] * math.cos(t))


def compose(view, cells, piv, mz, action, f):
    """mz: part -> muzzle point placed on the cell. Returns 1x cell, effects, muzzles [(pt, dir)], global shift."""
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    zero = [0.0, 0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    feet_names = [n for n in R["draw"] if n.startswith("foot")]
    arm_names = [n for n in R["draw"] if n.startswith("arm")]
    b = P.get("body", zero)
    muz = {}
    for n in R["draw"]:
        rot, df, du = 0.0, 0.0, 0.0
        if n in ("body", "shell"):
            df, du = b[1], b[2]
        elif n == "cannon":
            c = P.get("cannon", zero)
            rot = c[0] * R["csgn"]
            df, du = b[1], b[2]
            if R["csgn"] == 0 and action == "attack":
                du += -0.12 * c[0]                       # front / back: the barrel cannot tilt, it sinks on recoil
            if R["csgn"] == 0 and action in ("idle", "walk"):
                du += round(c[0] / 2.5)
        elif n == "head":
            h = P.get("head", zero); df, du = h[1], h[2] + (b[2] if action != "idle" else 0)
        elif n.startswith("arm"):
            if action == "walk":
                key = "arm_a" if n == arm_names[0] else "arm_b"
            else:
                key = "arms"
            a = P.get(key, zero)
            df, du = a[1], a[2]
        elif n.startswith("foot"):
            key = "foot_a" if n == feet_names[0] else "foot_b"
            ft = P.get(key, zero); df, du = ft[1], ft[2]
        elif n == "tail":
            t = P.get("tail", zero)
            rot = t[0]
            du = b[2] * 0.5
        dx, dy = df * fwx, df * fwy - du
        if n.startswith("foot") and abs(fwx) < 0.5:
            dx, dy = 0.0, -du                             # front / back: steps are vertical only
        if n in mz:
            pt, d = mz[n]
            p2 = rot_pt(pt, rot, piv[n])
            muz[n] = ((p2[0] + dx, p2[1] + dy), rot_vec(d, rot))
        l4 = rot4(up4(cells[n]), rot, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.4
    if abs(lean) > 0.3 and abs(fwx) > 0.3:
        canvas = rot4(canvas, lean * side, (CELL_W // 2, FOOT_Y))
        for n in muz:
            muz[n] = (rot_pt(muz[n][0], lean * side, (CELL_W // 2, FOOT_Y)), rot_vec(muz[n][1], lean * side))
    canvas = shift4(canvas, gf * fwx, gf * fwy - gu)
    for n in muz:
        p, d = muz[n]
        muz[n] = ((p[0] + gf * fwx, p[1] + gf * fwy - gu), d)
    out = canvas[1::UP, 1::UP].copy()
    return out, fx, muz, (gf * fwx, gf * fwy - gu)


# ---------------------------------------------------------------- effects (water, reserved colours only)
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 2 <= x < a.shape[1] - 2 and 2 <= y < a.shape[0] - 2:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def disc(a, cx, cy, r, col):
    for yy in range(int(-r) - 1, int(r) + 2):
        for xx in range(int(-r) - 1, int(r) + 2):
            if xx * xx + yy * yy <= r * r:
                put(a, cx + xx, cy + yy, col)


def ring(a, cx, cy, r, col, gaps=0):
    n = max(8, int(2 * math.pi * r))
    for j in range(n):
        if gaps and (j * gaps // n) % 2:
            continue
        t = 2 * math.pi * j / n
        put(a, cx + math.cos(t) * r, cy + math.sin(t) * r * 0.8, col)


def jet(a, p, d, L, w):
    """water jet from p along d: pale core, blue rim, wobbling edge."""
    dx, dy = d
    Ld = math.hypot(dx, dy) or 1.0
    dx, dy = dx / Ld, dy / Ld
    px, py = -dy, dx
    for s in range(int(L) + 1):
        ww = w + (1 if (s // 3) % 2 else 0) * (1 if s > 4 else 0)
        cx, cy = p[0] + dx * s, p[1] + dy * s
        for o in range(-ww, ww + 1):
            put(a, cx + px * o, cy + py * o, FX_LO if abs(o) == ww else FX_HI)
    # rounded head of the jet
    disc(a, p[0] + dx * L, p[1] + dy * L, w + 1, FX_LO)
    disc(a, p[0] + dx * L, p[1] + dy * L, max(0, w - 1), FX_HI)


def draw_fx(a, fx, muz, fwd, centre, view):
    big_jet = RIG[view]["csgn"] != 0 or view == "N"
    for kind, k in fx:
        if kind == "drip":
            for n, (p, d) in muz.items():
                if n == "cannon":
                    continue
                y = p[1] + 1 + (k - 6) * 2
                put(a, p[0], y, FX_LO)
                if k < 9:
                    put(a, p[0], y - 1, FX_HI)
        elif kind == "dust":
            for j in range(3):
                put(a, centre[0] - 18 - 3 * j, FOOT_Y - j, FX_HI)
                put(a, centre[0] + 18 + 3 * j, FOOT_Y - j, FX_HI)
        elif kind == "charge":
            for n, (p, d) in muz.items():
                r = 7 - 2 * k
                for j in range(6):
                    t = j * math.pi / 3 + k * 0.5
                    put(a, p[0] + math.cos(t) * (r + 2), p[1] + math.sin(t) * (r + 2), FX_LO)
                disc(a, p[0] + d[0] * 2, p[1] + d[1] * 2, k - 1, FX_HI)
        elif kind == "jet":
            grow = {4: 8, 5: 18, 6: 28, 7: 34, 8: 36, 9: 30}[k]
            for n, (p, d) in muz.items():
                L = grow
                w = 2
                if n == "cannon":
                    if not big_jet:
                        # front view: the big barrel points at the camera -> a blast ring at its mouth instead
                        if k <= 7:
                            disc(a, p[0], p[1] + 3, 3 + (k - 4), FX_HI)
                            ring(a, p[0], p[1] + 3, 5 + 2 * (k - 4), FX_LO, gaps=6)
                        continue
                    w = 3
                    if view == "N":
                        L = min(L, 26)
                elif view in ("S", "N"):
                    L = min(L, 14 if view == "S" else 16)
                if k == 9:
                    jet(a, (p[0] + d[0] * 8, p[1] + d[1] * 8), d, max(4, L - 8), max(1, w - 1))
                else:
                    jet(a, p, d, L, w)
                if k == 4:
                    disc(a, p[0], p[1], 3, FX_HI)
        elif kind == "splash":
            for n, (p, d) in muz.items():
                if n == "cannon" and not big_jet:
                    continue
                L = 36 if k != 10 else 30
                if view in ("S", "N") and n != "cannon":
                    L = 14 if view == "S" else 16
                if n == "cannon" and view == "N":
                    L = 26
                cx, cy = p[0] + d[0] * L, p[1] + d[1] * L
                r = {8: 4, 9: 7, 10: 9}[k]
                for j in range(8):
                    t = j * math.pi / 4 + k * 0.4
                    put(a, cx + math.cos(t) * r, cy + math.sin(t) * r, FX_LO if j % 2 else FX_HI)
                    if k < 10:
                        put(a, cx + math.cos(t) * (r - 2), cy + math.sin(t) * (r - 2), FX_HI)
        elif kind == "spray":
            for j in range(5):
                t = j * 1.3 + k * 0.7
                cx = centre[0] + math.cos(t) * 26
                cy = centre[1] - 10 + math.sin(t) * 14 - (k - 2) * 1.5
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
                  "--balance", "--name", "blastoise-mega", "--output-dir", str(pal_dir)])
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
        mz = {n: ((p[0] + ox, p[1] + oy), d) for n, (p, d) in RIG[v]["muzzles"].items()}
        rig[v] = (cells, piv, mz, (ox, oy))

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        cells, piv, mz, (ox, oy) = rig[v]
        R = RIG[v]
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, muz, (gx, gy) = compose(v, cells, piv, mz, act, f)
                centre = (CELL_W / 2 + gx, FOOT_Y - 40 + gy)
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, muz, R["fwd"], centre, v)
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
        for si, fc in enumerate((0, 4, 8, 12) if act != "attack" else (2, 5, 7, 9)):
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
    meta = {"generator": "forge_gen1_0009_blastoise_mega_actions", "version": VERSION, "key": KEY, "dex": 9,
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
