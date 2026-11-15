#!/usr/bin/env python3
"""Charizard, normal form (0006_charizard): 8-direction x 4-action sheet (gen1 forge batch, md/gen1-forge.md).

Same flow as tools/forge_raichu_mega_x_actions.py (structure only; rig, colours and attack are this form's own):
  1. The official front.png / back.png (x8 nearest -> art/ref_front.png, art/ref_back.png) were given to the
     Antigravity CLI (agy, image generation) with art/agy_prompt.md (no proper nouns, exact part counts):
       art/out/keys_2x2.png  : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
       art/out/sw_single.png : SW (3/4 front, faces down-left)
     (1st try drew flame wings + a white belly -> art/try1/; the 2nd try of keys_2x2 is correct.)
     The other three are mirrors: SE = SW, E = W, NW = NE.
  2. Key the white backdrop (border-connected white + large enclosed gaps between wings / tail / body; small white
     areas = claws and teeth stay), area-downscale. S is scaled so its height = the official front.png opaque height
     (80 px); W / NE / N share the grid scale; SW is matched on horn-to-claw height.
  3. agent-sprite-forge palette_tool: build (30 colours + 3 reserved fire-FX colours) -> lock -> apply.
  4. Anatomical rig per view: two bat wings with teal membranes (pivot at the shoulder), head + neck + horns, two
     clawed arms, torso, two legs, one long tail with the flame tip (pivot at the rump). Posed at 4x (nearest), back
     to 1x. Actions (16 frames): idle (breath, slow wing beat, tail sway, embers off the tail flame), walk
     (alternating stomps, arm swing, wing beat, tail sway), attack (fire breath: head rears back and the wings rise
     -> head thrusts forward, small mouth opening via tools/mouth_open.py, wings sweep down, a cone of fire streams
     out of the mouth -> fire puffs -> recover), hurt (white flash, recoil, wings flare, embers, recover).
  5. QC: off-palette, edge-touch, still diff (frame 0 of every action = the rest pose), gait_loop.py idle / walk.

The official front / back png / gif are never touched. Writes 8dir_stills.png and agent-sprite-forge/ in the form
folder and syncs agent-sprite-forge/ (files only, not _work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0006_charizard_actions.py [--force]
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
sys.path.insert(0, str(ROOT / "tools"))
import mouth_open  # noqa: E402

KEY = "0006_charizard"
REL = "monsterSprite/gen1/0006_charizard/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0006_charizard_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
OFFICIAL_H = 80                      # opaque height of the official front.png
GRID_H_S = 379                       # S full height (wing tips -> claws) in the generated grid (px)
S_HEAD_FOOT = 375                    # S horn tip -> claws in the grid (px)
SW_HEAD_FOOT = 610                   # SW horn tip -> claws in sw_single.png (px)
CELL_W, CELL_H, FOOT_Y = 208, 136, 122
UP = 4
FX_HI, FX_MID, FX_LO = (255, 244, 176), (255, 176, 40), (240, 84, 24)
INK = None                           # darkest palette colour (set in main) for the mouth gap
INNER = None                         # dark red palette colour (set in main) for the throat
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# label: later polygons win. draw: back to front. sgn: rotation sign that RAISES a wing / tail tip.
# fwd = screen direction of forward, mouth = closed mouth line (hinge -> lip) or None, muzzle = where the fire
# leaves the head (for views without a visible mouth), tip = tail-flame point (embers), multi = parts that may be
# several islands (the tail passes behind a wing).
RIG = {
    "S": dict(
        poly={"wing_l": [(0, 0), (56, 0), (56, 28), (52, 35), (44, 37), (30, 41), (0, 48)],
              "wing_r": [(80, 0), (136, 0), (136, 48), (106, 41), (92, 37), (84, 35), (80, 28)],
              "tail": [(96, 56), (104, 56), (108, 48), (110, 33), (126, 33), (126, 62), (112, 72), (96, 74)],
              "body": [(26, 56), (46, 50), (48, 34), (56, 30), (80, 30), (88, 34), (90, 50), (96, 56), (96, 72),
                       (40, 74), (26, 72)],
              "foot_l": [(42, 60), (60, 62), (60, 80), (40, 80)],
              "foot_r": [(76, 62), (96, 60), (98, 80), (76, 80)],
              "head": [(55, 0), (78, 0), (78, 24), (73, 31), (60, 31), (55, 24)],
              "arm_l": [(34, 44), (44, 37), (55, 36), (56, 46), (48, 50), (46, 56), (34, 56)],
              "arm_r": [(81, 36), (92, 37), (102, 44), (102, 56), (90, 56), (88, 50), (80, 46)]},
        label=["wing_l", "wing_r", "tail", "body", "foot_l", "foot_r", "head", "arm_l", "arm_r"],
        draw=["wing_l", "wing_r", "tail", "foot_l", "foot_r", "body", "head", "arm_l", "arm_r"],
        pivot={"wing_l": (56, 33), "wing_r": (80, 33), "tail": (96, 66), "head": (67, 31), "arm_l": (54, 40),
               "arm_r": (82, 40), "body": (68, 76), "foot_l": (50, 78), "foot_r": (87, 78)},
        sgn={"wing_l": 1, "wing_r": -1, "tail": -1},
        body_x=68, fwd=(0.0, 1.0), mouth=[(62, 27), (72, 27)], peak=0.5, muzzle=(67, 30), tip=(117, 40),
        multi=[], back=False),
    "SW": dict(
        poly={"wing_l": [(0, 0), (30, 0), (30, 14), (22, 18), (12, 24), (10, 40), (14, 48), (4, 56), (0, 56)],
              "wing_r": [(48, 0), (97, 0), (97, 52), (90, 48), (80, 40), (68, 40), (60, 42), (50, 40), (46, 30)],
              "tail": [[(49, 12), (63, 12), (63, 32), (49, 32)],
                     [(62, 46), (82, 40), (92, 44), (94, 64), (86, 78), (70, 82), (66, 70)]],
              "body": [(26, 40), (46, 34), (60, 44), (68, 58), (68, 80), (50, 88), (34, 84), (28, 64)],
              "foot_l": [(16, 70), (42, 72), (42, 94), (16, 94)],
              "foot_r": [(54, 76), (78, 76), (78, 96), (54, 96)],
              "head": [(10, 24), (22, 15), (38, 15), (41, 30), (37, 40), (27, 47), (10, 47)],
              "arm_l": [(10, 52), (28, 44), (38, 46), (38, 56), (24, 60), (10, 66)],
              "arm_r": [(50, 50), (64, 50), (66, 68), (50, 68)]},
        label=["wing_l", "wing_r", "tail", "body", "foot_l", "foot_r", "head", "arm_l", "arm_r"],
        draw=["wing_l", "wing_r", "tail", "foot_r", "body", "foot_l", "head", "arm_r", "arm_l"],
        pivot={"wing_l": (32, 34), "wing_r": (52, 40), "tail": (66, 66), "head": (32, 44), "arm_l": (34, 48),
               "arm_r": (56, 52), "body": (48, 88), "foot_l": (30, 92), "foot_r": (64, 94)},
        sgn={"wing_l": 1, "wing_r": -1, "tail": -1},
        body_x=48, fwd=(-0.9, 0.45), mouth=[(24, 43), (13, 44)], peak=0.65, muzzle=(12, 42), tip=(55, 18),
        multi=["tail", "wing_l"], back=False),
    "W": dict(
        poly={"wing": [(28, 0), (90, 0), (90, 48), (62, 44), (55, 40), (42, 36), (36, 30), (28, 10)],
              "tail": [(50, 56), (70, 60), (94, 52), (93, 20), (109, 20), (109, 62), (90, 70), (60, 76), (50, 72)],
              "body": [(16, 22), (38, 22), (48, 36), (56, 50), (56, 68), (40, 74), (20, 72), (16, 52)],
              "foot_a": [(10, 56), (30, 58), (30, 80), (10, 80)],
              "foot_b": [(32, 60), (52, 60), (52, 80), (32, 80)],
              "head": [(0, 2), (30, 2), (30, 22), (24, 30), (12, 31), (0, 30)],
              "arm_far": [(4, 38), (24, 38), (26, 48), (4, 54)],
              "arm_near": [(24, 42), (42, 40), (44, 57), (26, 59)]},
        label=["wing", "tail", "body", "foot_a", "foot_b", "head", "arm_far", "arm_near"],
        draw=["wing", "tail", "foot_b", "body", "foot_a", "head", "arm_far", "arm_near"],
        pivot={"wing": (40, 30), "tail": (52, 64), "head": (22, 29), "arm_far": (24, 42), "arm_near": (34, 44),
               "body": (34, 74), "foot_a": (20, 78), "foot_b": (42, 78)},
        sgn={"wing": -1, "tail": -1},
        body_x=34, fwd=(-1.0, 0.0), mouth=[(15, 26), (4, 26)], peak=0.7, muzzle=(2, 25), tip=(101, 32),
        multi=["wing"], back=False),
    "NE": dict(
        poly={"wing_l": [(0, 0), (58, 0), (58, 28), (52, 34), (44, 38), (30, 36), (20, 34), (10, 36), (0, 46)],
              "wing_r": [(72, 26), (86, 18), (88, 0), (131, 0), (131, 48), (104, 44), (92, 37), (78, 35)],
              "body": [(50, 28), (80, 28), (86, 40), (88, 62), (84, 72), (64, 76), (48, 72), (44, 50)],
              "head": [(56, 0), (87, 0), (87, 28), (76, 31), (66, 33), (58, 28)],
              "foot_l": [(38, 70), (56, 70), (56, 79), (38, 79)],
              "foot_r": [(68, 62), (92, 62), (92, 79), (68, 79)],
              "arm_l": [(34, 38), (50, 38), (50, 54), (34, 54)],
              "arm_r": [(78, 38), (94, 40), (94, 56), (78, 56)],
              "tail": [(8, 32), (26, 32), (28, 56), (44, 62), (62, 60), (70, 64), (66, 70), (54, 73), (40, 74),
                       (28, 72), (16, 66), (8, 56)]},
        label=["wing_l", "wing_r", "body", "head", "foot_l", "foot_r", "arm_l", "arm_r", "tail"],
        draw=["foot_l", "foot_r", "body", "arm_l", "arm_r", "wing_l", "wing_r", "head", "tail"],
        pivot={"wing_l": (56, 32), "wing_r": (80, 34), "head": (68, 32), "tail": (60, 66), "body": (66, 78),
               "foot_l": (47, 78), "foot_r": (80, 78), "arm_l": (48, 42), "arm_r": (80, 42)},
        sgn={"wing_l": 1, "wing_r": -1, "tail": 1},
        body_x=66, fwd=(0.9, -0.45), mouth=None, muzzle=(86, 24), tip=(16, 40), multi=[], back=True),
    "N": dict(
        poly={"wing_l": [(0, 0), (58, 0), (58, 30), (54, 38), (40, 40), (28, 42), (14, 44), (0, 50)],
              "wing_r": [(76, 0), (133, 0), (133, 50), (110, 46), (96, 42), (80, 38), (76, 30)],
              "body": [(58, 30), (76, 30), (82, 44), (92, 58), (92, 70), (70, 74), (46, 72), (44, 58), (52, 44)],
              "head": [(56, 0), (78, 0), (78, 26), (72, 32), (62, 32), (56, 26)],
              "foot_l": [(40, 58), (58, 62), (58, 80), (40, 80)],
              "foot_r": [(72, 56), (92, 56), (92, 70), (72, 70)],
              "arm_l": [(38, 40), (56, 40), (56, 56), (38, 56)],
              "arm_r": [(78, 40), (96, 40), (96, 56), (78, 56)],
              "tail": [(58, 66), (66, 64), (80, 70), (100, 68), (106, 58), (106, 38), (122, 38), (122, 62),
                       (110, 74), (90, 78), (64, 80), (58, 74)]},
        label=["wing_l", "wing_r", "body", "head", "foot_l", "foot_r", "arm_l", "arm_r", "tail"],
        draw=["foot_l", "foot_r", "body", "head", "arm_l", "arm_r", "wing_l", "wing_r", "tail"],
        pivot={"wing_l": (58, 34), "wing_r": (76, 34), "head": (67, 32), "tail": (66, 70), "body": (66, 80),
               "foot_l": (49, 78), "foot_r": (82, 70), "arm_l": (54, 42), "arm_r": (80, 42)},
        sgn={"wing_l": 1, "wing_r": -1, "tail": -1},
        body_x=66, fwd=(0.0, -1.0), mouth=None, muzzle=(67, 2), tip=(115, 44), multi=[], back=True),
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
    scale["SW"] = OFFICIAL_H / GRID_H_S * S_HEAD_FOOT / SW_HEAD_FOOT
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
    """p: one polygon or a list of polygons (union)."""
    m = Image.new("L", (shape[1], shape[0]), 0)
    for q in (p if isinstance(p[0][0], (list, tuple)) else [p]):
        ImageDraw.Draw(m).polygon([tuple(v) for v in q], fill=255)
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
    # every part must be one piece (except `multi` parts): tiny stray islands go to the nearest other part
    for i, n in enumerate(names, start=1):
        m = lab == i
        cl, cn = ndi.label(m)
        if cn > 1:
            sz = ndi.sum(np.ones_like(cl), cl, range(1, cn + 1))
            big = int(np.argmax(sz)) + 1
            keep = {big}
            if n in R.get("multi", []):
                keep |= {j + 1 for j in range(cn) if sz[j] >= 12}
            stray = m & ~np.isin(cl, list(keep))
            other = op & (lab != i)
            if other.any() and stray.any():
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
            # underlay for everything drawn above the body inside the body + head outline (arms over the belly,
            # legs at the hips), so no hole opens when an arm or a leg moves
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
    """Per-part [rot deg (+ raises a wing / tail tip), fwd px, up px]; global [fwd px, up px, lean deg].
    P["mouth"] = mouth opening in px (0..2)."""
    P = {"g": [0.0, 0.0, 0.0], "mouth": 0}
    fx = []
    if action == "idle":
        breath = max(0.0, wave(f, 1.0, 1, -math.pi / 2) / 2.0)          # 0 -> 1 -> 0
        P["body"] = [0, 0, round(breath)]
        P["head"] = [0, 0, round(wave(f, 1.0, 1, -1.2) / 1.6)]
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
        # fire breath: rear back (1-3) -> thrust (4-5) -> fire stream (5-10) -> recover (11-15)
        g_f = [0, -1, -2, -3, 1, 4, 5, 5, 5, 5, 4, 3, 2, 1, 0, 0][f]
        g_u = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -2, -4, -5, 1, 4, 5, 5, 5, 4, 3, 2, 1, 1, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["head"] = [0, [0, -1, -2, -3, 1, 3, 4, 4, 4, 3, 2, 1, 1, 0, 0, 0][f], [0, 1, 2, 2, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0][f]]
        P["arms"] = [0, [0, 0, -1, -1, 0, 1, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0][f], [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["foot_a"] = [0, [0, 0, 0, 0, 2, 3, 3, 3, 3, 3, 2, 2, 1, 1, 0, 0][f], [0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["wings"] = [[0, 6, 12, 16, 8, -4, -8, -8, -7, -5, -4, -2, -1, 0, 0, 0][f], 0, 0]
        P["tail"] = [[0, 3, 6, 8, 2, -3, -5, -5, -4, -3, -2, -2, -1, 0, 0, 0][f], 0, 0]
        P["mouth"] = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0][f]
        if 1 <= f <= 3:
            fx.append(("charge", f))
        if 5 <= f <= 10:
            fx.append(("breath", f))
        if f in (9, 10, 11):
            fx.append(("puff", f))
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


def rot_pt(p, ang, piv):
    """rotate a point (+ = clockwise on screen, y down) about piv."""
    t = math.radians(ang)
    dx, dy = p[0] - piv[0], p[1] - piv[1]
    return (piv[0] + dx * math.cos(t) - dy * math.sin(t), piv[1] + dx * math.sin(t) + dy * math.cos(t))


def compose(view, cells, piv, action, f, origin):
    """Returns the 1x frame, effects, part offsets, global shift and the mouth point (cell coords)."""
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
            h = P.get("head", zero); df, du = h[1], h[2] + (b[2] if action != "idle" else 0)
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
        offs[n] = (dx, dy)
        l4 = rot4(up4(cells[n]), rot, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.4
    ox, oy = origin
    hd = offs.get("head", (0, 0))
    mz = (R["muzzle"][0] + ox + hd[0], R["muzzle"][1] + oy + hd[1])
    line = None
    if R["mouth"]:
        line = mouth_open.transform_points([(x + ox, y + oy) for x, y in R["mouth"]], dx=hd[0], dy=hd[1])
    ang = 0.0
    if abs(lean) > 0.3 and abs(fwx) > 0.3:
        ang = lean * side
        canvas = rot4(canvas, ang, (CELL_W // 2, FOOT_Y))
        mz = rot_pt(mz, ang, (CELL_W // 2, FOOT_Y))
        if line:
            line = mouth_open.transform_points(line, angle=ang, pivot=(CELL_W // 2, FOOT_Y))
    canvas = shift4(canvas, gf * fwx, gf * fwy - gu)
    gx, gy = gf * fwx, gf * fwy - gu
    mz = (mz[0] + gx, mz[1] + gy)
    out = canvas[1::UP, 1::UP].copy()
    if line and P["mouth"] > 0:
        line = mouth_open.transform_points(line, dx=gx, dy=gy)
        out = mouth_open.open_mouth(out, line, amount=P["mouth"], deep=INK, inner=INNER, peak=R.get("peak", 0.6))
        lip = line[-1]
        mz = (lip[0] + fwx * 1.5, lip[1] + 1 + fwy * 1.5)
    return out, fx, offs, (gx, gy), mz


# ---------------------------------------------------------------- effects (fire, reserved colours only)
def put(a, x, y, col, behind=False):
    x, y = int(round(x)), int(round(y))
    if 2 <= x < a.shape[1] - 2 and 2 <= y < a.shape[0] - 2:
        if behind and a[y, x, 3] > 0 and tuple(a[y, x, :3]) not in (FX_HI, FX_MID, FX_LO):
            return
        a[y, x, :3] = col
        a[y, x, 3] = 255


def flame_tongue(a, x, y, h, lean, seed, behind=False):
    """small upward flame: pale core, amber, red rim."""
    rng = np.random.default_rng(seed)
    for j in range(h):
        w = max(0, (h - j) // 3)
        cx = x + lean * j / max(1, h) + rng.uniform(-0.6, 0.6)
        for dx in range(-w - 1, w + 2):
            col = FX_HI if abs(dx) <= w - 2 else (FX_MID if abs(dx) <= w - 1 else FX_LO)
            put(a, cx + dx, y - j, col, behind)


def fire_cone(a, x0, y0, fwd, L, start, seed, behind=False):
    """fire stream along fwd from distance `start` to L: widening cone, pale core, amber body, red rim."""
    rng = np.random.default_rng(seed)
    fwx, fwy = fwd
    n = math.hypot(fwx, fwy) or 1.0
    fwx, fwy = fwx / n, fwy / n
    px, py = -fwy, fwx
    for d in range(int(start), int(L) + 1):
        w = 1.2 + d * 0.22
        wob = math.sin(d * 0.7 + seed) * 0.8
        for s in np.arange(-w - 1, w + 1.01, 0.5):
            r = abs(s) / (w + 1)
            if r > 0.92 and rng.random() < 0.5:
                continue
            col = FX_HI if r < 0.35 else (FX_MID if r < 0.7 else FX_LO)
            if d > L - 4:
                col = FX_LO if r > 0.3 else FX_MID
            put(a, x0 + fwx * d + px * (s + wob), y0 + fwy * d + py * (s + wob), col, behind)


def fireball(a, cx, cy, r, seed, behind=False):
    """ragged ball of fire facing the camera (front / back views): dark rim (the sprite's own outline colour),
    red, amber, pale core, with tongues licking upward."""
    rng = np.random.default_rng(seed)
    R = int(math.ceil(r)) + 3
    h, w = a.shape[:2]
    m = np.zeros((h, w), bool)
    yy, xx = np.mgrid[0:h, 0:w]
    ang = np.arctan2(yy - cy, xx - cx)
    rr = r * (1.0 + 0.18 * np.sin(5 * ang + seed) + 0.1 * np.sin(9 * ang + 2 * seed))
    d = np.hypot(xx - cx, (yy - cy) * 1.1) / np.maximum(rr, 0.5)
    m = d <= 1.0
    for j in range(3):                                   # upward tongues
        tx, ty, th = cx + (j - 1) * r * 0.5, cy - r * 0.7, max(2.0, r * 0.8)
        u = (ty - yy) / th
        m |= (u >= 0) & (u <= 1) & (np.abs(xx - tx - (j - 1) * u * 1.5) <= (1 - u) * r * 0.32 + 0.3)
    rim = m & ~ndi.binary_erosion(m)
    core = d < 0.4
    mid = (d < 0.75) & ~core
    for mk, col in ((m, FX_LO), (mid, FX_MID), (core, FX_HI)):
        ys, xs = np.nonzero(mk)
        for y, x in zip(ys, xs):
            put(a, x, y, col, behind)
    if INK is not None:
        ys, xs = np.nonzero(rim)
        for y, x in zip(ys, xs):
            put(a, x, y, INK, behind)


def draw_fx(a, fx, mouth, fwd, centre, tip, back):
    fwx, fwy = fwd
    for kind, k in fx:
        if kind == "ember" and tip:
            rise = {3: 2, 4: 5, 5: 8, 11: 2, 12: 5, 13: 8}[k]
            off = -2 if k < 8 else 3
            put(a, tip[0] + off, tip[1] - 3 - rise, FX_LO)
            if rise < 8:
                put(a, tip[0] + off, tip[1] - 4 - rise, FX_MID)
        elif kind == "dust":
            for j in range(3):
                put(a, centre[0] - 14 - 3 * j, FOOT_Y - j, FX_MID)
                put(a, centre[0] + 14 + 3 * j, FOOT_Y - j, FX_MID)
        elif kind == "charge" and mouth:
            # sparks gathering at the mouth
            for j in range(2 + k):
                ang = j * 2.3 + k * 0.8
                r = 8 - 2 * (k - 1)
                put(a, mouth[0] + math.cos(ang) * r, mouth[1] + math.sin(ang) * r, FX_MID if j % 2 else FX_LO, back)
            if k == 3:
                flame_tongue(a, mouth[0] + fwx * 2, mouth[1] + 1 + fwy * 2, 4, 0, 7, back)
        elif kind == "breath" and mouth:
            if abs(fwx) < 0.5:
                # toward / away from the camera: a ball of fire grows in front of (S) or beyond (N) the mouth
                r = {5: 3, 6: 6, 7: 9, 8: 11, 9: 10, 10: 7}[k]
                fireball(a, mouth[0] + fwx * r, mouth[1] + fwy * (r + 1), r, 11 + k, back)
            else:
                L = {5: 12, 6: 24, 7: 34, 8: 40, 9: 40, 10: 38}[k]
                start = {5: 0, 6: 0, 7: 0, 8: 2, 9: 10, 10: 22}[k]
                fire_cone(a, mouth[0], mouth[1], fwd, L, start, 11 + k, back)
        elif kind == "puff" and mouth and abs(fwx) < 0.5:
            if k == 11:                                       # the fire ball burns out
                fireball(a, mouth[0], mouth[1] + fwy * 5, 4, 30, back)
        elif kind == "puff" and mouth:
            d = {9: 40, 10: 42, 11: 44}[k] if abs(fwx) >= 0.5 else {9: 12, 10: 15, 11: 18}[k]
            cx, cy = mouth[0] + fwx * d, mouth[1] + fwy * d
            for j in range(5):
                ang = j * 1.3 + k
                r = 3 + (k - 9) * 2
                flame_tongue(a, cx + math.cos(ang) * r, cy + math.sin(ang) * r + 2, 4 - (k - 9), 0, 90 + j + k, back)
        elif kind == "embers":
            for j in range(4):
                ang = j * 1.7 + k * 0.9
                cx, cy = centre[0] + math.cos(ang) * 22, centre[1] - 6 + math.sin(ang) * 16 - (k - 2)
                put(a, cx, cy, FX_LO)
                put(a, cx, cy - 1, FX_MID)



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
        run_tool(["build", "--input", str(raw), "--colors", "30", "--reserve", "#%02x%02x%02x" % FX_HI, "#%02x%02x%02x" % FX_MID,
                  "#%02x%02x%02x" % FX_LO,
                  "--balance", "--name", "charizard", "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}
    # mouth colours from the sprite's own palette: the darkest colour (gap) and the darkest strong red (throat)
    global INK, INNER
    pal = set()
    for v in RIG:
        s = still[v]
        pal |= {tuple(int(c) for c in p) for p in np.unique(s[s[..., 3] > 0][:, :3], axis=0)}
    pal -= {FX_HI, FX_MID, FX_LO}
    INK = min(pal, key=lambda c: sum(c))
    reds = [c for c in pal if c[0] > 120 and c[0] - c[1] > 70 and c[0] - c[2] > 70]
    INNER = min(reds, key=lambda c: sum(c)) if reds else None

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
                a, fx, offs, (gx, gy), mouth = compose(v, cells, piv, act, f, (ox, oy))
                centre = (CELL_W / 2 + gx, FOOT_Y - 36 + gy)
                tip = None
                if R.get("tip"):
                    toff = offs.get("tail", (0, 0))
                    tip = (R["tip"][0] + ox + gx + toff[0], R["tip"][1] + oy + gy + toff[1])
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, mouth if act == "attack" else None, R["fwd"], centre, tip, v == "N")
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
    meta = {"generator": "forge_gen1_0006_charizard_actions", "version": VERSION, "key": KEY, "dex": 6,
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
