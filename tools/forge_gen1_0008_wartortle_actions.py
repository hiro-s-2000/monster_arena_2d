#!/usr/bin/env python3
"""Gen-1 water turtle, 2nd stage (0008_wartortle): 8-direction stills + 8-direction x 4-action sheet.

Batch md/gen1-forge.md (2026-11-04). Same flow as tools/forge_raichu_mega_x_actions.py, rig / colours / attack
built from THIS form's own art:
  art/ref_front.png, art/ref_back.png : the official front.png / back.png at x8 (nearest)
  art/agy_prompt.md                   : Antigravity CLI prompt (no proper nouns, part counts and colours)
  art/out/keys_2x2.png                : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
  art/out/sw_single.png               : SW (3/4 front, faces down-left)
  SE = SW mirrored, E = W mirrored, NW = NE mirrored.

Pipeline:
  1. Key the white backdrop (border-connected only: the pale tail / ear fur and eye highlights stay), downscale
     every view so the S still is exactly as tall as the official front.png (57 px); SW is matched to the same
     height.
  2. agent-sprite-forge palette_tool: build (the 15 official colours reserved + learned shades) -> lock -> apply.
  3. Anatomical rig per view: one fluffy tail (pivot at the rump under the shell), head with the two feathery
     ears, two stubby arms (pivot at the shoulder), shell + belly torso, two legs. Hidden seams get cushions.
  4. Poses at 4x (nearest), sampled back to 1x (no new colours).
     idle   : breathing, tail sway, arms bob
     walk   : alternating legs, arm swing, tail swing, body bounce
     attack : water jet from the mouth (crouch + tail curls up + head pulls back -> mouth opens (mouth_open.py)
              and a water jet shoots forward -> splash -> recover). Water colours are the sprite's own blues/white.
     hurt   : white flash, recoil back, recover
  5. QC: off-palette colours, frames touching the cell edge, still diff, gait_loop.py on idle / walk.

Writes monsterSprite/gen1/0008_wartortle/normal/: 8dir_stills.png, agent-sprite-forge/ (sheet.png, preview.png,
meta.json, per-action sheets, _work/). The official front/back png/gif are never touched. Syncs agent-sprite-forge/
(without _work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0008_wartortle_actions.py [--force] [--no-sync]
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

KEY = "0008_wartortle"
REL = "monsterSprite/gen1/0008_wartortle/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0008_wartortle_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
TARGET_H = 57                      # opaque height of the official front.png
CELL_W, CELL_H, FOOT_Y = 144, 96, 86
UP = 4
OFFICIAL = ["#101010", "#29416a", "#31414a", "#627bc5", "#6a4a18", "#8b5a20", "#8bacf6", "#946262", "#ac8b62",
            "#acc5ff", "#bdc5e6", "#d5eef6", "#dec58b", "#ffffff", "#cd8320"]
INK = (16, 16, 16)
NAVY = (41, 65, 106)
W_DEEP, W_MID, W_LT, W_PALE, W_WHITE = (98, 123, 197), (139, 172, 246), (172, 197, 255), (213, 238, 246), (255, 255, 255)
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# label order: later polygons win. draw: back to front. body_x = body centre x (placed on the cell centre).
# fwd = screen direction of "forward"; mouth = closed mouth line (hinge -> lip) or None; nozzle = jet origin.
RIG = {
    "S": dict(
        poly={"tail": [(44, 15), (61, 15), (61, 51), (42, 51), (39, 43), (40, 24)],
              "body": [(14, 29), (38, 29), (40, 51), (14, 51)],
              "head": [(9, 0), (45, 0), (45, 19), (37, 24), (36, 30), (15, 30), (13, 22), (9, 18)],
              "leg_l": [(8, 45), (22, 45), (22, 59), (8, 59)],
              "leg_r": [(33, 44), (47, 44), (47, 59), (33, 59)],
              "arm_l": [(0, 27), (16, 27), (16, 40), (0, 40)],
              "arm_r": [(34, 27), (47, 27), (47, 40), (34, 40)]},
        label=["tail", "body", "head", "leg_l", "leg_r", "arm_l", "arm_r"],
        draw=["tail", "leg_l", "leg_r", "body", "head", "arm_l", "arm_r"],
        pivot={"tail": (41, 43), "head": (25, 30), "arm_l": (15, 32), "arm_r": (35, 32), "body": (25, 51),
               "leg_l": (15, 46), "leg_r": (39, 46)},
        mouth=[(20, 27), (25, 27), (30, 27)], mouth_peak=0.5, nozzle=(23, 30), jet_dir=(-0.6, 0.8),
        reach=[5, 10, 15, 19, 19, 15], jet_w=1.2, tail_side=1, body_x=25, fwd=(0.0, 1.0)),
    "SW": dict(
        poly={"tail": [(43, 15), (61, 15), (61, 51), (42, 51), (39, 41)],
              "body": [(14, 29), (37, 27), (39, 51), (14, 51)],
              "head": [(5, 0), (40, 0), (40, 19), (31, 22), (30, 30), (10, 30), (7, 25), (5, 18)],
              "leg_l": [(10, 45), (23, 45), (23, 59), (10, 59)],
              "leg_r": [(30, 44), (46, 44), (46, 59), (30, 59)],
              "arm_l": [(0, 29), (15, 29), (15, 40), (0, 40)],
              "arm_r": [(30, 25), (43, 25), (43, 41), (30, 41)]},
        label=["tail", "body", "head", "leg_l", "leg_r", "arm_l", "arm_r"],
        draw=["tail", "leg_l", "leg_r", "body", "head", "arm_l", "arm_r"],
        pivot={"tail": (41, 43), "head": (22, 30), "arm_l": (14, 33), "arm_r": (32, 30), "body": (25, 51),
               "leg_l": (16, 46), "leg_r": (37, 45)},
        mouth=[(25, 28), (19, 28), (13, 28)], mouth_peak=0.6, nozzle=(12, 29), jet_dir=(-0.894, 0.447),
        reach=[6, 12, 18, 22, 22, 18], jet_w=1.0, tail_side=1, body_x=24, fwd=(-0.9, 0.45)),
    "W": dict(
        poly={"tail": [(34, 15), (53, 15), (53, 54), (30, 54), (30, 47), (34, 40)],
              "body": [(8, 31), (23, 21), (36, 21), (36, 51), (9, 51)],
              "head": [(9, 0), (30, 0), (30, 20), (23, 24), (22, 31), (10, 31), (1, 29), (1, 15), (8, 10)],
              "leg_b": [(7, 44), (17, 44), (17, 58), (7, 58)],
              "leg_f": [(17, 42), (30, 42), (30, 59), (17, 59)],
              "arm_f": [(0, 29), (23, 29), (23, 39), (0, 39)]},
        label=["tail", "body", "head", "leg_b", "leg_f", "arm_f"],
        draw=["tail", "leg_b", "body", "leg_f", "head", "arm_f"],
        pivot={"tail": (33, 46), "head": (18, 30), "arm_f": (20, 33), "body": (22, 51), "leg_b": (13, 46),
               "leg_f": (23, 44)},
        mouth=[(16, 26), (10, 26), (4, 25)], mouth_peak=0.7, nozzle=(2, 26), jet_dir=(-1.0, 0.0),
        reach=[7, 14, 22, 28, 28, 24], jet_w=1.0, tail_side=1, body_x=20, fwd=(-1.0, 0.0)),
    "NE": dict(
        poly={"tail": [(0, 20), (21, 20), (21, 56), (14, 61), (0, 61)],
              "body": [(19, 23), (40, 23), (41, 56), (19, 56)],
              "head": [(14, 0), (47, 0), (47, 26), (20, 26), (14, 18)],
              "leg_l": [(17, 55), (29, 55), (29, 62), (17, 62)],
              "leg_r": [(36, 46), (49, 46), (49, 62), (36, 62)],
              "arm_r": [(39, 24), (54, 24), (54, 36), (39, 36)]},
        label=["tail", "body", "head", "leg_l", "leg_r", "arm_r"],
        draw=["leg_l", "leg_r", "arm_r", "head", "body", "tail"],
        pivot={"tail": (21, 50), "head": (31, 26), "arm_r": (40, 29), "body": (30, 56), "leg_l": (23, 56),
               "leg_r": (41, 48)},
        mouth=None, nozzle=(45, 20), jet_dir=(0.894, -0.447),
        reach=[5, 10, 15, 18, 18, 15], jet_w=1.0, tail_side=-1, body_x=30, fwd=(0.9, -0.45)),
    "N": dict(
        poly={"tail": [(0, 22), (24, 22), (26, 40), (30, 52), (22, 60), (0, 60)],
              "body": [(25, 22), (55, 22), (56, 56), (26, 56)],
              "head": [(22, 0), (57, 0), (57, 20), (50, 25), (28, 25), (22, 20)],
              "leg_l": [(20, 53), (33, 53), (33, 62), (20, 62)],
              "leg_r": [(47, 50), (60, 50), (60, 62), (47, 62)],
              "arm_l": [(17, 27), (27, 27), (27, 40), (17, 40)],
              "arm_r": [(53, 27), (64, 27), (64, 40), (53, 40)]},
        label=["tail", "body", "head", "leg_l", "leg_r", "arm_l", "arm_r"],
        draw=["leg_l", "leg_r", "arm_l", "arm_r", "head", "body", "tail"],
        pivot={"tail": (26, 50), "head": (39, 25), "arm_l": (26, 32), "arm_r": (54, 32), "body": (40, 56),
               "leg_l": (26, 54), "leg_r": (52, 52)},
        mouth=None, nozzle=(44, 12), jet_dir=(0.6, -0.8),
        reach=[4, 8, 12, 15, 15, 12], jet_w=1.2, tail_side=-1, body_x=40, fwd=(0.0, -1.0)),
}


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
def key_white(rgb):
    a = rgb.astype(int)
    white = a.min(axis=2) > 225
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    # no enclosed white is backdrop here: the pale fur of the ears / tail and the eye highlights stay
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
    keyed, boxes, eyes = {}, {}, {}
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb), "RGBA")
        boxes[v] = im.getbbox()
        keyed[v] = im.crop(boxes[v])
        # brown irises (they blur away in the area downscale): centroid of each brown blob, re-stamped below
        c = rgb.astype(int)
        brown = (c[..., 0] > 70) & (c[..., 0] < 170) & (c[..., 0] > c[..., 2] + 40) & (c[..., 1] < 110) & (c[..., 1] > 40)
        if v in ("S", "SW", "W"):
            lb, nb = ndi.label(brown)
            cands = []
            for i in range(1, nb + 1):
                m = lb == i
                if 20 <= m.sum() <= 1200:
                    ys, xs = np.nonzero(m)
                    if ys.mean() < boxes[v][1] + 0.4 * (boxes[v][3] - boxes[v][1]):       # upper part = face
                        cands.append((ys.mean(), xs.mean()))
            eyes[v] = cands
        else:
            eyes[v] = []
    grid_scale = TARGET_H / keyed["S"].height
    dst.mkdir(parents=True, exist_ok=True)
    for v, im in keyed.items():
        sc = grid_scale if v != "SW" else TARGET_H / im.height
        sm = im.resize((round(im.width * sc), round(im.height * sc)), Image.BOX)
        a = np.array(sm)
        a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
        a[a[..., 3] == 0] = 0
        x0, y0 = boxes[v][:2]
        for ey, ex in eyes[v]:
            px, py = int((ex - x0) * sc), int((ey - y0) * sc)
            if 0 <= py < a.shape[0] and 0 <= px < a.shape[1] and a[py, px, 3]:
                a[py, px, :3] = (106, 74, 24)
        a = np.pad(a, ((1, 1), (1, 1), (0, 0)))           # one transparent px margin keeps the outline intact
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
              "--balance", "--name", "wartortle", "--output-dir", str(pal_dir)])
    run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
    run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
    # the orphan pass can remove the 1 px irises again: put them back (#6a4a18 is an official, reserved colour)
    for fp in raw.glob("*.png"):
        r = np.array(Image.open(fp).convert("RGBA"))
        s = np.array(Image.open(stills / fp.name).convert("RGBA"))
        eye = (r[..., 3] > 0) & (r[..., 0] == 106) & (r[..., 1] == 74) & (r[..., 2] == 24)
        s[eye] = (106, 74, 24, 255)
        Image.fromarray(s, "RGBA").save(stills / fp.name)


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
    # the dark seam line between an arm and the torso belongs to the arm
    r0, g0, b0 = [still[..., c].astype(int) for c in range(3)]
    dark = op & (np.maximum(np.maximum(r0, g0), b0) < 80)
    for i, n in enumerate(names, start=1):
        if n.startswith("arm"):
            arm = lab == i
            near = dark & ~arm & np.isin(lab, [names.index("body") + 1]) & (ndi.distance_transform_edt(~arm) <= 1)
            lab[near] = i
    masks = {n: lab == (i + 1) for i, n in enumerate(names)}
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    bright = np.maximum(np.maximum(r, g), b) > 70
    layers = {}
    order = R["draw"]
    for i, n in enumerate(order):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in order[i + 1:]:
            above |= masks[n2]
        cushion = 4 if n in ("tail", "body") else 2
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion)
        src_m = m
        if n == "body" and m.any():
            bh = m | masks["head"]
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(bh, 6), iterations=6))[6:-6, 6:-6]
            seam |= above & ~m & inside & ~masks["head"] & ~masks["tail"]
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


MOUTH_AMT = [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0]


def pose(action, f):
    """Per-part [rot deg, fwd px, up px]; global [fwd px, up px, lean deg]; effects."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        P["body"] = [0, 0, round(wave(0.6, 1, f, 0.0))]
        P["head"] = [wave(2.0, 1, f, -0.4), 0, round(wave(1.0, 1, f, -0.5))]
        P["arms"] = [wave(5.0, 1, f, -0.3), 0, round(wave(0.8, 1, f, -0.3))]
        P["tail"] = [wave(5.0, 1, f, -0.9), 0, 0]
    elif action == "walk":
        ph = 2 * math.pi * f / N
        s1 = math.sin(ph)
        up = round(1.0 * abs(math.sin(ph)))
        P["body"] = [0, 0, up]
        P["head"] = [wave(2.0, 2, f, 0.0), 0, up]
        P["leg_a"] = [0, 2.5 * math.sin(ph), 2.5 * max(0.0, s1)]
        P["leg_b"] = [0, -2.5 * math.sin(ph), 2.5 * max(0.0, -s1)]
        P["arm_a"] = [-16.0 * s1, 0, up]
        P["arm_b"] = [16.0 * s1, 0, up]
        P["tail"] = [wave(7.0, 1, f, 0.0), 0, up * 0.5]
    elif action == "attack":
        g_f = [0, -1, -2, -3, -3, 2, 4, 5, 5, 5, 4, 3, 2, 1, 0, 0][f]
        g_u = [0, 0, -1, -1, -1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -3, -6, -8, -8, 3, 6, 6, 6, 5, 4, 3, 2, 1, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["head"] = [[0, -2, -4, -6, -6, 3, 5, 5, 5, 4, 3, 2, 1, 0, 0, 0][f], 0, [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        # both arms pull back while charging, then brace forward (recoil) while the jet fires
        P["arms_atk"] = [[0, 10, 20, 28, 30, -12, -20, -22, -22, -20, -16, -12, -8, -4, -2, 0][f], 0, 0]
        P["leg_a"] = [0, [0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 2, 2, 1, 1, 0, 0][f], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["tail"] = [[0, 8, 16, 22, 24, 6, -6, -10, -10, -8, -6, -4, -2, -1, 0, 0][f], 0, 0]
        P["mouth"] = MOUTH_AMT[f]
        if 2 <= f <= 4:
            fx.append(("gather", f))
        if 5 <= f <= 10:
            fx.append(("jet", f))
        if 8 <= f <= 12:
            fx.append(("splash", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 2 <= f <= 6 else 0
        P["g"] = [-5.0 * env + jit, 1.0 * env, -12.0 * env]
        P["head"] = [-8.0 * env, -1.0 * env, 1.0 * env]
        P["arms"] = [20.0 * env, 0, 1.0 * env]
        P["tail"] = [-12.0 * env, 0, 0]
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
    xf = {}
    head_canvas = None
    for n in R["draw"]:
        rot, df, du = 0.0, 0.0, 0.0
        if n == "body":
            df, du = b[1], b[2]
        elif n == "head":
            h = P.get("head", zero)
            rot, df, du = h[0] * side * (1.0 if abs(fwx) > 0.3 else 0.3), h[1], h[2]
        elif n.startswith("arm"):
            sgn = 1.0 if piv[n][0] < CELL_W // 2 else -1.0       # arms on either side mirror the swing
            if action == "attack":
                a = P["arms_atk"]
                rot = a[0] * (side if abs(fwx) > 0.3 else -sgn * 0.6)
            elif action == "walk":
                a = P["arm_a"] if n == arms[0] else P.get("arm_b", zero)
                rot = a[0] * (1.0 if abs(fwx) > 0.3 else 0.35) * (side if abs(fwx) > 0.3 else sgn)
            else:
                a = P.get("arms", zero)
                rot = a[0] * sgn
            df, du = a[1], a[2] + b[2]
        elif n.startswith("leg"):
            ft = P.get("leg_a" if n == legs[0] else "leg_b", zero)
            df, du = ft[1], ft[2]
        elif n == "tail":
            t = P.get("tail", zero)
            sgn = 1.0 if R["tail_side"] > 0 else -1.0     # + t lifts the tail tip away from the body
            rot, du = -t[0] * sgn, t[2] + b[2] * 0.5
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
    feet = (CELL_W // 2, FOOT_Y)
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

    amt = P.get("mouth", 0)
    if amt and R.get("mouth_cell"):
        line = [track(p, "head") for p in R["mouth_cell"]]
        out = mouth_open.open_mouth(out, line, amount=amt, deep=INK, inner=(148, 98, 98), peak=R.get("mouth_peak", 0.6))
    return out, fx, track


# ---------------------------------------------------------------- effects
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def disc(a, cx, cy, r, col):
    for yy in range(int(cy - r) - 1, int(cy + r) + 2):
        for xx in range(int(cx - r) - 1, int(cx + r) + 2):
            if (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r:
                put(a, xx, yy, col)


def jet(a, p0, p1, width, seed):
    """A water jet from p0 to p1: deep-blue rim, light blue body, white core, a few spray droplets."""
    L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if L < 1:
        return
    ux, uy = (p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L
    n = int(L) + 1
    for layer, extra, col in ((0, 1.0, W_DEEP), (1, 0.0, W_LT), (2, -1.0, W_WHITE)):
        for i in range(n + 1):
            u = i / n
            wdt = max(0.0, width * (0.55 + 0.45 * u) / 2 + extra)
            if layer == 2:
                wdt = max(0.0, wdt - 0.3)
            disc(a, p0[0] + ux * L * u, p0[1] + uy * L * u, wdt, col)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        u = rng.uniform(0.3, 1.0)
        o = rng.choice([-1, 1]) * rng.uniform(width / 2 + 2, width / 2 + 4)
        put(a, p0[0] + ux * L * u - uy * o, p0[1] + uy * L * u + ux * o, W_PALE)


def draw_fx(a, fx, R, track):
    fwx, fwy = R["fwd"]
    noz = track(R["nozzle_cell"], "head")
    rng = np.random.default_rng(7)
    for kind, k in fx:
        if kind == "gather":
            # water droplets drawn in toward the mouth
            r = {2: 9, 3: 6, 4: 3}[k]
            for j in range(6):
                ang = j * math.pi / 3 + k * 0.5
                put(a, noz[0] + math.cos(ang) * r, noz[1] + math.sin(ang) * r * 0.8, W_LT if j % 2 else W_PALE)
        elif kind == "jet":
            L = R["reach"][k - 5]
            wdt = [2, 3, 4, 4, 3, 2][k - 5] * R["jet_w"]
            d = R["jet_dir"]
            start = (noz[0] + d[0] * (2 if k < 9 else 2 + (k - 8) * 6), noz[1] + d[1] * (2 if k < 9 else 2 + (k - 8) * 6))
            end = (noz[0] + d[0] * L, noz[1] + d[1] * L)
            if k >= 9 and math.hypot(end[0] - start[0], end[1] - start[1]) < 2:
                continue
            jet(a, start, end, wdt, 30 + k)
        elif kind == "splash":
            d = R["jet_dir"]
            L = R["reach"][3]
            c = (noz[0] + d[0] * (L + 2), noz[1] + d[1] * (L + 2))
            r = {8: 3, 9: 5, 10: 7, 11: 8, 12: 9}[k]
            for j in range(10):
                ang = j * math.pi / 5 + 0.3
                x, y = c[0] + math.cos(ang) * r, c[1] + math.sin(ang) * r * 0.7
                col = W_WHITE if k <= 9 else (W_LT if k <= 11 else W_DEEP)
                put(a, x, y, col)
                if k <= 10:
                    put(a, x + math.cos(ang), y + math.sin(ang) * 0.7, W_MID)
                if 9 <= k <= 11 and j % 2 == 0:
                    ri = r - 3
                    put(a, c[0] + math.cos(ang + 0.3) * ri, c[1] + math.sin(ang + 0.3) * ri * 0.7, W_PALE)
                    put(a, c[0] + math.cos(ang) * (r + 2), c[1] - 1 + math.sin(ang) * (r + 2) * 0.7 - (k - 8), W_LT)
            if k <= 9:
                disc(a, c[0], c[1], 1.5, W_PALE)
        elif kind == "drops":
            c = track(R["nozzle_cell"], "head")
            for j in range(4):
                ang = -math.pi / 2 + (j - 1.5) * 0.7
                r = 4 + (k - 2) * 2
                put(a, c[0] + math.cos(ang) * r - fwx * 2, c[1] - 6 + math.sin(ang) * r, W_LT if j % 2 else W_PALE)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = (255, 255, 255)
    return out


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    h.update(Path(mouth_open.__file__).read_bytes())
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
        R = RIG[v]
        rig[v] = (cells, piv, place_layer(still[v], v)[0])
        R["_off"] = (ox, oy)
        R["nozzle_cell"] = (R["nozzle"][0] + ox, R["nozzle"][1] + oy)
        R["mouth_cell"] = [(x + ox, y + oy) for x, y in R["mouth"]] if R.get("mouth") else None

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        cells, piv, _ = rig[v]
        R = RIG[v]
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
                draw_fx(a, fx, R, track)
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
            for fr in ("00.png", "15.png") if act in ("idle", "walk") else ("00.png",):
                a0 = np.array(Image.open(frames_dir / v / act / fr).convert("RGBA"))
                dd = int((np.abs(a0.astype(int) - ref.astype(int)).sum(axis=2) > 0).sum())
                if dd and fr == "00.png":
                    still_diff[f"{v}/{act}"] = dd
    s_h = int(np.ptp(np.nonzero(rig["S"][2][..., 3].any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off),
              "off_list": ["#%02x%02x%02x" % c for c in off][:10],
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
    meta = {"generator": "forge_gen1_0008_wartortle_actions", "version": VERSION, "key": KEY, "dex": 8,
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
    if sys.argv[1:2] == ["--stills-only"]:
        w = OUT_DIR / "_work"
        w.mkdir(parents=True, exist_ok=True)
        build_stills(w)
        raise SystemExit(0)
    main()
