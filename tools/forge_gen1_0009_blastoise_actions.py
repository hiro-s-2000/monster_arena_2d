#!/usr/bin/env python3
"""Gen-1 shell turtle, 3rd stage (0009_blastoise): 8-direction stills + 8-direction x 4-action sheet.

Batch md/gen1-forge.md (2026-11-04). Same flow as tools/forge_raichu_mega_x_actions.py, rig / colours / attack
built from THIS form's own art:
  art/ref_front.png, art/ref_back.png : the official front.png / back.png at x8 (nearest)
  art/agy_prompt.md                   : Antigravity CLI prompt (no proper nouns, part counts and colours)
  art/out/keys_2x2.png                : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
  art/out/sw_single.png               : SW (3/4 front, faces down-left)
  SE = SW mirrored, E = W mirrored, NW = NE mirrored.

Pipeline:
  1. Key the white backdrop (border-connected only: the white shell rim, cannons and claws stay), downscale every
     view so the S still is exactly as tall as the official front.png (62 px); SW is matched to the same height.
  2. agent-sprite-forge palette_tool: build (the 14 official colours reserved + learned shades + 2 water FX colours)
     -> lock -> apply on the 5 stills.
  3. Anatomical rig per view: two shell cannons (pivot at their root on the shell), shell, torso with the chest
     plate, head, two arms (pivot at the shoulder), two legs. Hidden seams get cushions (nearest own colour).
  4. Poses at 4x (nearest), sampled back to 1x (no new colours).
     idle   : heavy breathing, head bob, cannons nod, arms sway
     walk   : heavy alternating stomp, body sway, arm swing, cannons bounce
     attack : twin cannon water blast (crouch + cannons tilt to aim -> both cannons recoil and fire two water
              jets forward -> splash burst at the jet tips -> recover)
     hurt   : white flash, recoil back, recover
  5. QC: off-palette colours, frames touching the cell edge, still diff, gait_loop.py on idle / walk.

Writes monsterSprite/gen1/0009_blastoise/normal/: 8dir_stills.png, agent-sprite-forge/ (sheet.png, preview.png,
meta.json, per-action sheets, _work/). The official front/back png/gif are never touched. Syncs agent-sprite-forge/
(without _work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0009_blastoise_actions.py [--force] [--no-sync]
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
KEY = "0009_blastoise"
REL = "monsterSprite/gen1/0009_blastoise/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0009_blastoise_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
TARGET_H = 62                      # opaque height of the official front.png
CELL_W, CELL_H, FOOT_Y = 168, 100, 90
UP = 4
OFFICIAL = ["#083962", "#101010", "#2062ac", "#4a4a4a", "#5a3918", "#5a8bcd", "#8b6241", "#949494", "#94ace6",
            "#cdcdd5", "#d5ac4a", "#e6c573", "#f6d59c", "#ffffff"]
WATER_CORE = (226, 246, 255)       # reserved FX colours (water jet core / mid)
WATER_MID = (128, 196, 248)
WATER_LT, WATER_DK = (148, 172, 230), (90, 139, 205)      # official light blues (jet rim)
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# label order: later polygons win. draw: back to front. body_x = body centre x (placed on the cell centre).
# fwd = screen direction of "forward"; muzzle = cannon hole per cannon part (still px); jet = screen direction the
# cannons fire in that view.
S_LABEL = ["shell", "cannon_l", "cannon_r", "body", "leg_l", "leg_r", "head", "arm_l", "arm_r"]
RIG = {
    "S": dict(
        poly={"shell": [(6, 5), (64, 5), (64, 28), (6, 28)],
              "cannon_l": [(0, 1), (16, 1), (22, 8), (22, 16), (8, 17), (0, 13)],
              "cannon_r": [(70, 1), (54, 1), (48, 8), (48, 16), (62, 17), (70, 13)],
              "body": [(15, 16), (55, 16), (57, 52), (13, 52)],
              "leg_l": [(6, 42), (19, 42), (24, 50), (27, 64), (6, 64)],
              "leg_r": [(51, 42), (64, 42), (65, 64), (43, 64), (46, 50)],
              "head": [(21, 0), (49, 0), (49, 17), (43, 22), (27, 22), (21, 17)],
              "arm_l": [(2, 18), (17, 18), (19, 26), (16, 41), (2, 41)],
              "arm_r": [(53, 18), (68, 18), (68, 41), (54, 41), (51, 26)]},
        label=S_LABEL,
        draw=["shell", "cannon_l", "cannon_r", "body", "leg_l", "leg_r", "head", "arm_l", "arm_r"],
        pivot={"cannon_l": (19, 13), "cannon_r": (51, 13), "head": (35, 22), "arm_l": (14, 22), "arm_r": (56, 22),
               "body": (35, 52), "leg_l": (17, 45), "leg_r": (53, 45), "shell": (35, 28)},
        aim={"cannon_l": -1, "cannon_r": 1}, muzzle={"cannon_l": (3, 7), "cannon_r": (67, 7)},
        jet={"cannon_l": (-0.8, 0.6), "cannon_r": (0.8, 0.6)}, jet_len=16,
        body_x=35, fwd=(0.0, 1.0), back=False),
    "SW": dict(
        poly={"shell": [(36, 4), (60, 6), (68, 30), (64, 46), (54, 46), (40, 20)],
              "cannon_l": [(0, 2), (15, 2), (17, 15), (5, 17), (0, 11)],
              "cannon_r": [(54, 3), (67, 4), (68, 14), (60, 21), (53, 18), (51, 10)],
              "body": [(17, 18), (50, 18), (48, 54), (17, 54)],
              "tail": [(59, 41), (69, 41), (69, 51), (59, 51)],
              "leg_l": [(13, 38), (30, 40), (34, 60), (13, 60)],
              "leg_r": [(42, 41), (56, 41), (60, 48), (62, 64), (42, 64)],
              "head": [(15, 0), (38, 0), (38, 18), (32, 24), (20, 24), (15, 18)],
              "arm_l": [(2, 19), (19, 18), (21, 26), (16, 37), (2, 37)],
              "arm_r": [(36, 18), (54, 18), (55, 37), (36, 39)]},
        label=["shell", "cannon_l", "cannon_r", "body", "tail", "leg_l", "leg_r", "head", "arm_l", "arm_r"],
        draw=["cannon_r", "shell", "tail", "body", "leg_l", "leg_r", "cannon_l", "head", "arm_l", "arm_r"],
        pivot={"cannon_l": (14, 12), "cannon_r": (55, 16), "head": (28, 22), "arm_l": (17, 22), "arm_r": (50, 22),
               "body": (33, 54), "leg_l": (22, 42), "leg_r": (51, 44), "shell": (50, 30), "tail": (60, 46)},
        aim={"cannon_l": -1, "cannon_r": 1}, muzzle={"cannon_l": (3, 8), "cannon_r": (64, 9)},
        jet={"cannon_l": (-0.86, 0.5), "cannon_r": (-0.96, 0.28)}, jet_len=46,
        body_x=33, fwd=(-0.9, 0.45), back=False),
    "W": dict(
        poly={"shell": [(38, 4), (62, 8), (62, 54), (50, 60), (38, 60)],
              "cannon_l": [(30, 3), (41, 2), (50, 7), (50, 17), (42, 19), (31, 13)],
              "body": [(17, 19), (42, 19), (44, 50), (17, 50)],
              "tail": [(50, 51), (63, 51), (63, 60), (50, 60)],
              "leg_l": [(13, 44), (29, 44), (29, 64), (13, 64)],
              "leg_r": [(29, 44), (42, 44), (46, 56), (50, 67), (28, 67)],
              "head": [(4, 0), (31, 0), (31, 23), (18, 25), (4, 19)],
              "arm_l": [(1, 25), (14, 21), (26, 18), (41, 18), (43, 30), (36, 37), (16, 38), (1, 36)]},
        label=["shell", "cannon_l", "body", "tail", "leg_l", "leg_r", "head", "arm_l"],
        draw=["shell", "tail", "leg_l", "body", "cannon_l", "leg_r", "head", "arm_l"],
        pivot={"cannon_l": (45, 15), "head": (24, 22), "arm_l": (34, 24), "body": (30, 50), "leg_l": (21, 47),
               "leg_r": (37, 47), "shell": (45, 40), "tail": (50, 55)},
        aim={"cannon_l": -1}, muzzle={"cannon_l": (32, 7)},
        jet={"cannon_l": (-1.0, 0.12)}, jet_len=56,
        body_x=30, fwd=(-1.0, 0.0), back=False),
    "NE": dict(
        poly={"body": [(38, 10), (53, 10), (53, 57), (38, 57)],
              "head": [(25, 0), (51, 0), (51, 12), (44, 15), (30, 13)],
              "shell": [(3, 9), (45, 9), (47, 52), (30, 58), (4, 53)],
              "arm_r": [(46, 20), (56, 22), (63, 32), (63, 43), (50, 43), (46, 34)],
              "leg_l": [(1, 49), (27, 49), (27, 66), (1, 66)],
              "leg_r": [(28, 46), (51, 46), (52, 67), (28, 67)],
              "tail": [(0, 49), (17, 49), (17, 57), (0, 59)],
              "cannon_l": [(5, 2), (16, 2), (17, 14), (6, 14)],
              "cannon_r": [(42, 12), (50, 7), (61, 6), (62, 15), (52, 19), (42, 19)]},
        label=["body", "head", "leg_l", "leg_r", "tail", "shell", "arm_r", "cannon_l", "cannon_r"],
        draw=["leg_l", "leg_r", "body", "head", "arm_r", "shell", "tail", "cannon_l", "cannon_r"],
        pivot={"cannon_l": (11, 12), "cannon_r": (46, 16), "head": (38, 13), "arm_r": (48, 24), "body": (45, 56),
               "leg_l": (14, 52), "leg_r": (40, 50), "shell": (25, 30), "tail": (14, 52)},
        aim={"cannon_l": 1, "cannon_r": -1}, muzzle={"cannon_l": (11, 3), "cannon_r": (60, 10)},
        jet={"cannon_l": (0.7, -0.72), "cannon_r": (0.7, -0.72)}, jet_len=26,
        body_x=30, fwd=(0.9, -0.45), back=True),
    "N": dict(
        poly={"body": [(9, 10), (55, 10), (55, 56), (9, 56)],
              "head": [(18, 0), (45, 0), (45, 12), (18, 12)],
              "arm_l": [(0, 21), (11, 21), (11, 39), (0, 39)],
              "arm_r": [(53, 21), (64, 21), (64, 39), (53, 39)],
              "shell": [(8, 10), (56, 10), (55, 46), (43, 55), (21, 55), (9, 46)],
              "leg_l": [(3, 46), (23, 46), (24, 67), (2, 67)],
              "leg_r": [(40, 46), (61, 46), (62, 67), (39, 67)],
              "tail": [(24, 51), (35, 51), (35, 64), (24, 64)],
              "cannon_l": [(1, 4), (13, 4), (15, 17), (4, 17), (1, 12)],
              "cannon_r": [(63, 4), (51, 4), (49, 17), (60, 17), (63, 12)]},
        label=["body", "head", "arm_l", "arm_r", "leg_l", "leg_r", "tail", "shell", "cannon_l", "cannon_r"],
        draw=["body", "leg_l", "leg_r", "arm_l", "arm_r", "head", "shell", "tail", "cannon_l", "cannon_r"],
        pivot={"cannon_l": (11, 14), "cannon_r": (53, 14), "head": (31, 12), "arm_l": (9, 24), "arm_r": (55, 24),
               "body": (32, 56), "leg_l": (13, 50), "leg_r": (50, 50), "shell": (32, 30), "tail": (29, 53)},
        aim={"cannon_l": 1, "cannon_r": -1}, muzzle={"cannon_l": (6, 7), "cannon_r": (58, 7)},
        jet={"cannon_l": (-0.12, -1.0), "cannon_r": (0.12, -1.0)}, jet_len=26,
        body_x=32, fwd=(0.0, -1.0), back=True),
}


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
def key_white(rgb):
    a = rgb.astype(int)
    white = a.min(axis=2) > 225
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    bg |= ndi.binary_dilation(bg, iterations=2) & (a.min(axis=2) > 200)        # light fringe
    out = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    lab2, n2 = ndi.label(out[..., 3] > 0)
    if n2:
        sz = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 1500])] = 0     # specks
    return out


def make_stills(dst: Path):
    k = np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], "W": k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": np.array(Image.open(ART / "out/sw_single.png").convert("RGB"))}
    keyed = {}
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb), "RGBA")
        keyed[v] = im.crop(im.getbbox())
    grid_scale = TARGET_H / keyed["S"].height
    dst.mkdir(parents=True, exist_ok=True)
    for v, im in keyed.items():
        sc = grid_scale if v != "SW" else TARGET_H / im.height
        sm = im.resize((round(im.width * sc), round(im.height * sc)), Image.BOX)
        a = np.array(sm)
        a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
        a[a[..., 3] == 0] = 0
        a = np.pad(a, ((1, 1), (1, 1), (0, 0)))       # 1 px transparent margin
        Image.fromarray(a, "RGBA").save(dst / f"{v}.png")


def run_tool(args):
    r = subprocess.run([str(FPY), "-I", str(PALETTE_TOOL)] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"palette_tool {args[0]} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def build_stills(work):
    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    for d in (raw, stills, pal_dir, lock_dir):
        if d.exists():
            shutil.rmtree(d)
    make_stills(raw)
    run_tool(["build", "--input", str(raw), "--colors", "28", "--reserve", *OFFICIAL,
              "#%02x%02x%02x" % WATER_CORE, "#%02x%02x%02x" % WATER_MID,
              "--balance", "--name", "blastoise", "--output-dir", str(pal_dir)])
    run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
    run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])


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
    # the dark seam line between an arm and the torso belongs to the arm (else it stays painted on the torso)
    r0, g0, b0 = [still[..., c].astype(int) for c in range(3)]
    dark = op & (np.maximum(np.maximum(r0, g0), b0) < 80)
    if "body" in names:
        for i, n in enumerate(names, start=1):
            if n.startswith("arm"):
                arm = lab == i
                near = dark & ~arm & (lab == names.index("body") + 1) & (ndi.distance_transform_edt(~arm) <= 2)
                lab[near] = i
    masks = {n: lab == (i + 1) for i, n in enumerate(names)}
    bright = np.maximum(np.maximum(r0, g0), b0) > 70
    layers = {}
    order = R["draw"]
    zero = np.zeros_like(op)
    for i, n in enumerate(order):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in order[i + 1:]:
            above |= masks[n2]
        cushion = 4 if n in ("shell", "body") else 2
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion)
        src_m = m
        if n == "body" and m.any():
            # underlay inside the closed body + head outline (arms over the chest, the neck): no hole opens
            bh = m | masks["head"]
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(bh, 6), iterations=6))[6:-6, 6:-6]
            seam |= above & ~m & inside & ~masks["head"] & ~masks.get("tail", zero) & ~masks.get("shell", zero)
            src_m = bh
        if seam.any() and m.any():
            fillable = src_m & bright
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
    oy = FOOT_Y - (h - 2)                  # stills carry a 1 px margin: the lowest opaque row lands on FOOT_Y
    y0, x0 = max(0, -oy), max(0, -ox)
    y1, x1 = min(h, CELL_H - oy), min(w, CELL_W - ox)
    cell[y0 + oy:y1 + oy, x0 + ox:x1 + ox] = layer[y0:y1, x0:x1]
    return cell, (ox, oy)


# ---------------------------------------------------------------- 4. posing
def wave(amp, cycles, f, phase=0.0):
    """Zero at f = 0 and at the loop end."""
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def pose(action, f):
    """Per-part [rot deg, fwd px, up px]; global [fwd px, up px, lean deg]; effects."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        P["body"] = [0, 0, round(wave(0.7, 1, f, 0.0))]
        P["head"] = [wave(2.0, 1, f, -0.5), 0, round(wave(1.0, 1, f, -0.6))]
        P["arms"] = [wave(4.0, 1, f, -0.3), 0, 0]
        P["cannon"] = [wave(3.0, 1, f, -0.9), 0, 0]
        P["tail"] = [wave(5.0, 1, f, -0.4), 0, 0]
    elif action == "walk":
        ph = 2 * math.pi * f / N
        s1 = math.sin(ph)
        up = round(1.0 * abs(math.sin(ph)))
        P["body"] = [0, 0, up]
        P["head"] = [wave(2.0, 2, f, 0.0), 0, up]
        P["leg_a"] = [0, 3.0 * s1, 3.0 * max(0.0, s1)]
        P["leg_b"] = [0, -3.0 * s1, 3.0 * max(0.0, -s1)]
        P["arm_a"] = [-12.0 * s1, 0, 0]
        P["arm_b"] = [12.0 * s1, 0, 0]
        P["cannon"] = [wave(3.0, 2, f, 0.0), 0, 0]
        P["tail"] = [wave(8.0, 1, f, 0.0), 0, 0]
        P["g"] = [0, 0, wave(3.0, 1, f, 0.0)]           # heavy side-to-side roll
    elif action == "attack":
        # crouch + aim (1-4) -> fire, recoil (5-10) -> recover
        g_f = [0, 0, -1, -1, -1, -2, -3, -3, -3, -3, -2, -2, -1, -1, 0, 0][f]
        g_u = [0, -1, -2, -2, -2, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0][f]
        lean = [0, 2, 4, 5, 5, -2, -4, -4, -4, -3, -2, -1, -1, 0, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["head"] = [[0, 2, 3, 4, 4, 0, -2, -2, -2, -1, -1, 0, 0, 0, 0, 0][f], 0, [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["cannon"] = [[0, 4, 8, 11, 12, 12, 10, 10, 10, 10, 8, 6, 4, 2, 1, 0][f],
                       [0, 0, 0, 0, 0, -2, -1, -2, -1, -2, -1, 0, 0, 0, 0, 0][f], 0]       # aim + recoil
        P["arms"] = [[0, -6, -12, -16, -16, -10, -8, -8, -8, -8, -6, -4, -3, -2, -1, 0][f], 0, 0]   # brace out
        P["legs"] = [0, 0, 0]
        P["tail"] = [[0, 4, 8, 10, 10, -6, -8, -8, -6, -4, -2, -1, 0, 0, 0, 0][f], 0, 0]
        if 2 <= f <= 4:
            fx.append(("charge", f))
        if 5 <= f <= 11:
            fx.append(("jet", f))
        if 8 <= f <= 11:
            fx.append(("splash", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 2 <= f <= 6 else 0
        P["g"] = [-5.0 * env + jit, 1.0 * env, -12.0 * env]
        P["head"] = [-10.0 * env, -1.0 * env, 1.0 * env]
        P["arms"] = [16.0 * env, 0, 1.0 * env]
        P["cannon"] = [-6.0 * env, 0, 0]
        P["tail"] = [-10.0 * env, 0, 0]
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 5:
            fx.append(("drops", f))
    return P, fx


def up4(a):
    return np.repeat(np.repeat(a, UP, 0), UP, 1)


def rot4(img4, ang, piv):
    if abs(ang) < 0.5:
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
    a = math.radians(ang)
    x, y = p[0] - piv[0], p[1] - piv[1]
    return (piv[0] + x * math.cos(a) - y * math.sin(a), piv[1] + x * math.sin(a) + y * math.cos(a))


def compose(view, cells, piv, action, f):
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    zero = [0.0, 0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    legs = sorted([n for n in R["draw"] if n.startswith("leg")], key=lambda n: R["label"].index(n))
    arms = sorted([n for n in R["draw"] if n.startswith("arm")], key=lambda n: R["label"].index(n))
    b = P.get("body", zero)
    cx = CELL_W // 2
    xf = {}
    for n in R["draw"]:
        rot, df, du = 0.0, 0.0, 0.0
        if n in ("body", "shell"):
            df, du = b[1], b[2]
        elif n == "head":
            h = P.get("head", zero)
            rot, df, du = h[0] * side, h[1], h[2] + b[2]
        elif n.startswith("arm"):
            sgn = 1.0 if piv[n][0] < cx else -1.0           # arms on either side mirror the swing
            if action == "walk":
                a = P["arm_a"] if n == arms[0] else P["arm_b"]
                rot = a[0] * (1.0 if abs(fwx) > 0.3 else 0.4) * side
            else:
                a = P.get("arms", zero)
                rot = a[0] * sgn
                if abs(fwx) > 0.3 and len(arms) == 1:
                    rot = -a[0] * side * 0.6
            df, du = a[1], a[2] + b[2]
        elif n.startswith("leg"):
            ft = P.get("leg_a" if n == legs[0] else "leg_b", zero)
            df, du = ft[1], ft[2]
        elif n.startswith("cannon"):
            c = P.get("cannon", zero)
            rot = c[0] * R["aim"][n]
            df, du = c[1], c[2] + b[2]
        elif n == "tail":
            t = P.get("tail", zero)
            rot, du = t[0] * (1.0 if piv[n][0] > cx else -1.0), b[2] * 0.5
        dx, dy = df * fwx, df * fwy - du
        if n.startswith("leg") and abs(fwx) < 0.5:
            dx, dy = 0.0, -du                                   # front / back: steps are vertical only
        xf[n] = (rot, dx, dy)
        l4 = rot4(up4(cells[n]), rot, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.4
    feet = (cx, FOOT_Y)
    if action == "walk":
        lean_s = lean                                       # roll: same in every view
    else:
        lean_s = lean * side if abs(fwx) > 0.3 else lean * 0.3
    if abs(lean_s) > 0.5:
        canvas = rot4(canvas, lean_s, feet)
    gx, gy = gf * fwx, gf * fwy - gu
    canvas = shift4(canvas, gx, gy)
    out = canvas[1::UP, 1::UP].copy()

    def track(pt, part):
        rot, dx, dy = xf[part]
        p = rot_pt(pt, rot, piv[part])
        p = (p[0] + dx, p[1] + dy)
        p = rot_pt(p, lean_s, feet) if abs(lean_s) > 0.5 else p
        return (p[0] + gx, p[1] + gy)
    return out, fx, track


# ---------------------------------------------------------------- effects
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def disc(a, x, y, r, col):
    for j in range(-int(r), int(r) + 1):
        for i in range(-int(r), int(r) + 1):
            if i * i + j * j <= r * r + 0.5:
                put(a, x + i, y + j, col)


def water_jet(a, p0, d, L, width, seed, broken=False):
    """Thick water stream from p0 along d: light-blue rim, mid body, white core, wobbling a little."""
    rng = np.random.default_rng(seed)
    px, py = -d[1], d[0]
    steps = int(L)
    for i in range(steps + 1):
        if broken and (i // 3) % 2 == 1:
            continue
        w = width * (0.75 + 0.25 * min(1.0, i / 6.0))
        wob = 0.6 * math.sin(i * 0.7 + seed)
        x, y = p0[0] + d[0] * i + px * wob, p0[1] + d[1] * i + py * wob
        for o in np.arange(-w, w + 0.01, 0.5):
            ao = abs(o)
            col = WATER_CORE if ao <= w * 0.3 else (WATER_MID if ao <= w * 0.7 else WATER_LT)
            put(a, x + px * o, y + py * o, col)
        if i % 4 == 0 and rng.random() < 0.6:
            s = rng.choice([-1, 1])
            put(a, x + px * (w + 1.5) * s, y + py * (w + 1.5) * s, WATER_DK)


def draw_fx(a, fx, R, track):
    for kind, k in fx:
        for cn, mz in R["muzzle"].items():
            m = track(mz, cn)
            d = R["jet"][cn]
            L0 = R["jet_len"]
            if kind == "charge":
                r = k - 1
                for j in range(6):
                    ang = j * math.pi / 3 + k * 0.5
                    put(a, m[0] + math.cos(ang) * (4 - r), m[1] + math.sin(ang) * (4 - r), WATER_MID if j % 2 else WATER_LT)
                put(a, m[0], m[1], WATER_CORE)
            elif kind == "jet":
                L = {5: 0.35, 6: 0.7, 7: 1.0, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.6}[k] * L0
                width = {5: 1.2, 6: 1.8, 7: 2.4, 8: 2.4, 9: 2.0, 10: 1.4, 11: 0.9}[k]
                water_jet(a, m, d, L, width, 7 + k + len(cn), broken=k >= 10)
                disc(a, m[0], m[1], 2 if k <= 8 else 1, WATER_CORE)
            elif kind == "splash":
                tip = (m[0] + d[0] * L0, m[1] + d[1] * L0)
                rad = {8: 3, 9: 5, 10: 7, 11: 8}[k]
                rng = np.random.default_rng(50 + k + len(cn))
                for j in range(14):
                    ang = j * 2 * math.pi / 14 + rng.uniform(-0.2, 0.2)
                    rr = rad * rng.uniform(0.7, 1.1)
                    col = WATER_CORE if k <= 9 else (WATER_MID if j % 2 else WATER_LT)
                    put(a, tip[0] + math.cos(ang) * rr, tip[1] + math.sin(ang) * rr * 0.8, col)
                    if k <= 10:
                        put(a, tip[0] + math.cos(ang) * (rr - 1), tip[1] + math.sin(ang) * (rr - 1) * 0.8, WATER_MID)
                if k == 8:
                    disc(a, tip[0], tip[1], 2, WATER_CORE)
        if kind == "drops":
            rng = np.random.default_rng(80 + k)
            c = (CELL_W / 2, FOOT_Y - 40)
            for _ in range(8):
                ang = rng.uniform(0, 2 * math.pi)
                rr = 30 + (k - 2) * 3 + rng.uniform(-3, 3)
                put(a, c[0] + math.cos(ang) * rr, c[1] + math.sin(ang) * rr * 0.7, WATER_MID)
                put(a, c[0] + math.cos(ang) * rr, c[1] + math.sin(ang) * rr * 0.7 - 1, WATER_CORE)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = (255, 255, 255)
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

    stills, lock_dir = work / "stills", work / "palette_lock"
    if "stills" not in st["done"]:
        build_stills(work)
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}

    rig = {}
    for v in RIG:
        layers, _ = build_rig(v, still[v])
        cells = {}
        for n, lyr in layers.items():
            cells[n], (ox, oy) = place_layer(lyr, v)
        piv = {n: (p[0] + ox, p[1] + oy) for n, p in RIG[v]["pivot"].items()}
        rig[v] = (cells, piv, place_layer(still[v], v)[0], (ox, oy))

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        cells, piv, _, (ox, oy) = rig[v]
        R = RIG[v]
        R_cell = dict(R, muzzle={k: (p[0] + ox, p[1] + oy) for k, p in R["muzzle"].items()})
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, track = compose(v, cells, piv, act, f)
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                # water is drawn on its own layer and goes BEHIND the sprite: the cannons sit behind the head, so
                # a jet passes behind the head / body and comes out in front of the snout (never across the face)
                fl = np.zeros_like(a)
                draw_fx(fl, fx, R_cell, track)
                m = (fl[..., 3] > 0) & (a[..., 3] == 0)
                a[m] = fl[m]
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
        for si, fc in enumerate((0, 3, 7, 10) if act == "attack" else (0, 4, 8, 12)):
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
        ref = rig[v][2]
        for act in ACTIONS:
            for fr in ("00", "15"):
                if fr == "15" and act not in ("idle", "walk"):
                    continue
                a0 = np.array(Image.open(frames_dir / v / act / f"{fr}.png").convert("RGBA"))
                dd = int((np.abs(a0.astype(int) - ref.astype(int)).sum(axis=2) > 0).sum())
                if dd and fr == "00":
                    still_diff[f"{v}/{act}"] = dd
    s_h = int(np.ptp(np.nonzero(rig["S"][2][..., 3].any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off),
              "partial_alpha_px": alpha_bad, "edge_touching_frames": clip, "still_diff": still_diff, "S_height": s_h}
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
    meta = {"generator": "forge_gen1_0009_blastoise_actions", "version": VERSION, "key": KEY, "dex": 9,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["front.png", "back.png"],
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": s_h, "frames": N, "frame_time": AD,
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
    import sys
    if sys.argv[1:2] == ["--stills-only"]:
        w = OUT_DIR / "_work"
        w.mkdir(parents=True, exist_ok=True)
        build_stills(w)
        raise SystemExit(0)
    main()
