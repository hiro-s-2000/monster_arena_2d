#!/usr/bin/env python3
"""Gen-1 fire lizard, 2nd stage (0005_charmeleon): 8-direction stills + 8-direction x 4-action sheet.

Batch md/gen1-forge.md (2026-11-04). Same flow as tools/forge_raichu_mega_x_actions.py, rig / colours / attack
built from THIS form's own art:
  art/ref_front.png, art/ref_back.png : the official front.png / back.png at x8 (nearest)
  art/agy_prompt.md                   : Antigravity CLI prompt (no proper nouns, part counts and colours)
  art/out/keys_2x2.png                : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
  art/out/sw_single.png               : SW (3/4 front, faces down-left)
  SE = SW mirrored, E = W mirrored, NW = NE mirrored.

Pipeline:
  1. Key the white backdrop (border-connected only: the white claws and eye highlights stay), downscale every
     view so the S still is exactly as tall as the official front.png (56 px); SW is matched to the same height.
  2. agent-sprite-forge palette_tool: build (the 15 official colours reserved + learned shades + FX colours)
     -> lock -> apply on the 5 stills.
  3. Anatomical rig per view: one tail with the flame tip (pivot at the rump), head with the single horn,
     two arms with three claws each (pivot at the shoulder), torso, two legs. Hidden seams get cushions.
  4. Poses at 4x (nearest), sampled back to 1x (no new colours).
     idle   : breathing, tail sway, flame flicker
     walk   : alternating legs, arm swing, tail swing
     attack : fire claw slash (crouch + tail flame flares -> lunge, the front arm sweeps its claws down ->
              three burning claw streaks + embers -> recover)
     hurt   : white flash, recoil back, recover
  5. QC: off-palette colours, frames touching the cell edge, still diff, gait_loop.py on idle / walk.

Writes monsterSprite/gen1/0005_charmeleon/normal/: 8dir_stills.png, agent-sprite-forge/ (sheet.png, preview.png,
meta.json, per-action sheets, _work/). The official front/back png/gif are never touched. Syncs agent-sprite-forge/
(without _work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0005_charmeleon_actions.py [--force] [--no-sync]
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
KEY = "0005_charmeleon"
REL = "monsterSprite/gen1/0005_charmeleon/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0005_charmeleon_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
TARGET_H = 56                      # opaque height of the official front.png
CELL_W, CELL_H, FOOT_Y = 128, 84, 76
UP = 4
OFFICIAL = ["#101010", "#6a6a6a", "#942010", "#b48b5a", "#b4b4b4", "#cd3939", "#cdac7b", "#e6cdac", "#ff4100",
            "#ff524a", "#ff836a", "#ffa400", "#ffde29", "#ffffff", "#005aff"]
FIRE_CORE = (255, 246, 196)        # reserved FX colour (only for the attack streak core)
FIRE_Y, FIRE_O, FIRE_R = (255, 222, 41), (255, 164, 0), (255, 65, 0)     # official flame colours
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# label order: later polygons win. draw: back to front. body_x = body centre x (placed on the cell centre).
# fwd = screen direction of "forward"; slash = the arm that strikes; flame = tail flame centre (still px).
# raise = rotation sign that lifts the striking arm; claw = claw tip of the striking arm.
RIG = {
    "S": dict(
        poly={"tail": [(37, 16), (55, 16), (55, 42), (46, 45), (41, 49), (31, 49), (30, 44), (37, 40)],
              "body": [(11, 21), (31, 21), (33, 47), (12, 47)],
              "head": [(8, 0), (31, 0), (31, 19), (27, 23), (13, 23), (8, 18)],
              "leg_l": [(4, 40), (12, 40), (16, 46), (19, 47), (19, 58), (4, 58)],
              "leg_r": [(22, 45), (29, 40), (35, 44), (35, 58), (21, 58)],
              "arm_l": [(0, 25), (14, 25), (14, 44), (0, 44)],
              "arm_r": [(28, 26), (38, 26), (38, 43), (30, 43)]},
        label=["tail", "body", "head", "leg_l", "leg_r", "arm_l", "arm_r"],
        draw=["tail", "leg_l", "leg_r", "body", "head", "arm_l", "arm_r"],
        pivot={"tail": (31, 45), "head": (20, 22), "arm_l": (12, 29), "arm_r": (29, 29), "body": (21, 47),
               "leg_l": (12, 45), "leg_r": (27, 45)},
        slash="arm_l", raise_=1, claw=(3, 40), flame=(45, 19), body_x=21, fwd=(0.0, 1.0), back=False),
    "SW": dict(
        poly={"tail": [(39, 16), (60, 16), (60, 40), (45, 46), (38, 49), (31, 49), (31, 44), (38, 40)],
              "body": [(12, 21), (33, 21), (35, 47), (14, 47)],
              "head": [(8, 0), (30, 0), (30, 18), (25, 22), (10, 21), (8, 16)],
              "leg_l": [(4, 40), (13, 40), (17, 46), (20, 48), (20, 55), (4, 55)],
              "leg_r": [(21, 43), (31, 41), (33, 58), (20, 58)],
              "arm_l": [(0, 24), (13, 24), (13, 42), (0, 42)],
              "arm_r": [(28, 26), (39, 26), (39, 44), (29, 44)]},
        label=["tail", "body", "head", "leg_l", "leg_r", "arm_l", "arm_r"],
        draw=["tail", "leg_l", "leg_r", "body", "head", "arm_l", "arm_r"],
        pivot={"tail": (33, 45), "head": (20, 21), "arm_l": (12, 28), "arm_r": (29, 29), "body": (22, 47),
               "leg_l": (12, 45), "leg_r": (26, 45)},
        slash="arm_l", raise_=1, claw=(3, 36), flame=(50, 20), body_x=22, fwd=(-0.9, 0.45), back=False),
    "W": dict(
        poly={"tail": [(33, 9), (56, 9), (56, 40), (45, 46), (33, 48), (30, 41), (37, 36)],
              "body": [(13, 20), (31, 20), (33, 49), (13, 49)],
              "head": [(4, 0), (28, 0), (28, 20), (21, 23), (8, 22), (4, 18)],
              "leg_b": [(10, 42), (20, 42), (21, 55), (9, 55)],
              "leg_f": [(20, 39), (33, 39), (34, 56), (19, 56)],
              "arm_l": [(0, 25), (20, 25), (20, 33), (15, 34), (11, 40), (0, 40)]},
        label=["tail", "body", "head", "leg_b", "leg_f", "arm_l"],
        draw=["tail", "leg_b", "body", "leg_f", "head", "arm_l"],
        pivot={"tail": (31, 42), "head": (18, 21), "arm_l": (17, 29), "body": (22, 49), "leg_b": (15, 45),
               "leg_f": (26, 43)},
        slash="arm_l", raise_=1, claw=(2, 32), flame=(48, 12), body_x=22, fwd=(-1.0, 0.0), back=False),
    "NE": dict(
        poly={"tail": [(0, 9), (15, 9), (16, 34), (22, 40), (33, 41), (36, 50), (14, 50), (5, 41), (0, 31)],
              "body": [(21, 19), (42, 19), (44, 46), (20, 46)],
              "head": [(27, 0), (51, 0), (51, 20), (41, 23), (30, 21), (27, 13)],
              "leg_l": [(18, 47), (33, 47), (33, 55), (18, 55)],
              "leg_r": [(36, 42), (48, 39), (50, 58), (36, 58)],
              "arm_r": [(41, 27), (55, 27), (55, 41), (41, 41)]},
        label=["tail", "body", "head", "leg_l", "leg_r", "arm_r"],
        draw=["leg_l", "body", "leg_r", "arm_r", "head", "tail"],
        pivot={"tail": (31, 44), "head": (36, 21), "arm_r": (42, 31), "body": (32, 46), "leg_l": (26, 48),
               "leg_r": (42, 44)},
        slash="arm_r", raise_=-1, claw=(53, 37), flame=(6, 14), body_x=33, fwd=(0.9, -0.45), back=True),
    "N": dict(
        poly={"tail": [(15, 44), (37, 43), (37, 16), (50, 16), (50, 47), (37, 52), (15, 52)],
              "body": [(7, 21), (35, 21), (38, 48), (6, 48)],
              "head": [(9, 0), (30, 0), (30, 22), (9, 22)],
              "leg_l": [(2, 41), (15, 41), (16, 58), (2, 58)],
              "leg_r": [(23, 50), (36, 50), (36, 58), (23, 58)],
              "arm_l": [(0, 29), (10, 29), (10, 43), (0, 43)],
              "arm_r": [(30, 29), (38, 29), (38, 45), (30, 45)]},
        label=["tail", "body", "head", "leg_l", "leg_r", "arm_l", "arm_r"],
        draw=["leg_l", "leg_r", "body", "arm_l", "arm_r", "head", "tail"],
        pivot={"tail": (28, 47), "head": (19, 22), "arm_l": (9, 32), "arm_r": (31, 32), "body": (21, 48),
               "leg_l": (10, 45), "leg_r": (29, 50)},
        slash="arm_r", raise_=-1, claw=(36, 42), flame=(43, 19), body_x=22, fwd=(0.0, -1.0), back=True),
}


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
def key_white(rgb, hole_min):
    a = rgb.astype(int)
    white = a.min(axis=2) > 225
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    # enclosed white bigger than a claw / eye highlight = the backdrop seen between an arm and the body
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    bg |= np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= hole_min and (i + 1) not in border])
    bg |= ndi.binary_dilation(bg, iterations=2) & (a.min(axis=2) > 200)        # light fringe
    out = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    lab2, n2 = ndi.label(out[..., 3] > 0)
    if n2:
        sz = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 1500])] = 0     # loose embers / specks
    return out


def make_stills(dst: Path):
    k = np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], "W": k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": np.array(Image.open(ART / "out/sw_single.png").convert("RGB"))}
    hole_min = {"S": 200, "W": 200, "NE": 200, "N": 200, "SW": 600}
    keyed, boxes, eyes = {}, {}, {}
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb, hole_min[v]), "RGBA")
        boxes[v] = im.getbbox()
        keyed[v] = im.crop(boxes[v])
        # blue irises (they vanish in the area downscale): centroid of each blue blob, re-stamped below
        c = rgb.astype(int)
        blue = (c[..., 2] > 120) & (c[..., 2] > c[..., 0] + 70) & (c[..., 2] > c[..., 1] + 40)
        lb, nb = ndi.label(blue)
        eyes[v] = [ndi.center_of_mass(blue, lb, i) for i in range(1, nb + 1) if (lb == i).sum() >= 6]
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
                a[py, px, :3] = (0, 90, 255)
        # one transparent px margin keeps the outline intact
        a = np.pad(a, ((1, 1), (1, 1), (0, 0)))
        Image.fromarray(a, "RGBA").save(dst / f"{v}.png")


def run_tool(args):
    r = subprocess.run([str(FPY), "-I", str(PALETTE_TOOL)] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"palette_tool {args[0]} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def build_stills(work, force=False):
    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    for d in (raw, stills, pal_dir, lock_dir):
        if d.exists():
            shutil.rmtree(d)
    make_stills(raw)
    run_tool(["build", "--input", str(raw), "--colors", "28", "--reserve", *OFFICIAL, "#%02x%02x%02x" % FIRE_CORE,
              "--balance", "--name", "charmeleon", "--output-dir", str(pal_dir)])
    run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
    run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
    # the orphan pass removes the 1 px irises again: put them back (#005aff is an official, reserved colour)
    for fp in raw.glob("*.png"):
        r = np.array(Image.open(fp).convert("RGBA"))
        s = np.array(Image.open(stills / fp.name).convert("RGBA"))
        eye = (r[..., 3] > 0) & (r[..., 0] == 0) & (r[..., 1] == 90) & (r[..., 2] == 255)
        s[eye] = (0, 90, 255, 255)
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
    # the dark seam line between an arm and the torso belongs to the arm: otherwise it stays painted on the
    # torso when the arm swings away
    r0, g0, b0 = [still[..., c].astype(int) for c in range(3)]
    dark = op & (np.maximum(np.maximum(r0, g0), b0) < 80)
    for i, n in enumerate(names, start=1):
        if n.startswith("arm"):
            arm = lab == i
            near = dark & ~arm & np.isin(lab, [names.index("body") + 1]) & (ndi.distance_transform_edt(~arm) <= 2)
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
            # underlay for everything drawn above the body inside the closed body + head outline (arms over the
            # chest, the neck): filled from body / head colours, so no hole opens when an arm or the head moves
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
    ys = np.nonzero(layer[..., 3].any(axis=1))[0]
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
        P["body"] = [0, 0, round(wave(0.6, 1, f, 0.0))]
        P["head"] = [wave(2.0, 1, f, -0.4), 0, round(wave(1.0, 1, f, -0.5))]
        P["arms"] = [wave(4.0, 1, f, -0.3), 0, round(wave(0.8, 1, f, -0.3))]
        P["tail"] = [wave(4.0, 1, f, -0.9), 0, 0]
        if f in (3, 4, 11, 12):
            fx.append(("flicker", f))
    elif action == "walk":
        ph = 2 * math.pi * f / N
        s1, c1 = math.sin(ph), math.cos(ph) - 1.0      # c1 = 0 at f = 0
        up = round(1.0 * abs(math.sin(ph)))
        P["body"] = [0, 0, up]
        P["head"] = [wave(2.0, 2, f, 0.0), 0, up]
        P["leg_a"] = [0, 3.0 * math.sin(ph), 2.5 * max(0.0, s1)]
        P["leg_b"] = [0, -3.0 * math.sin(ph), 2.5 * max(0.0, -s1)]
        P["arm_a"] = [-14.0 * s1, 0, up]
        P["arm_b"] = [14.0 * s1, 0, up]
        P["tail"] = [wave(6.0, 1, f, 0.0), 0, up * 0.5]
    elif action == "attack":
        g_f = [0, -1, -2, -3, -3, 3, 7, 9, 9, 8, 6, 4, 3, 2, 1, 0][f]
        g_u = [0, -1, -1, -2, -2, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -3, -5, -7, -7, 4, 9, 10, 9, 7, 5, 3, 2, 1, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["head"] = [[0, -3, -5, -6, -6, 2, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0][f], 0, 0]
        # striking arm: wind up high behind (+), sweep down through the target (-)
        P["slash"] = [[0, 30, 60, 85, 95, 40, -20, -45, -50, -45, -35, -25, -15, -8, -3, 0][f],
                      [0, -1, -2, -2, -2, 2, 5, 6, 6, 5, 4, 3, 2, 1, 0, 0][f], 0]
        P["guard"] = [[0, -6, -10, -12, -12, 6, 12, 14, 12, 10, 8, 6, 4, 2, 1, 0][f], 0, 0]
        P["leg_a"] = [0, [0, 0, 0, 0, 0, 2, 4, 5, 5, 5, 4, 3, 2, 1, 0, 0][f], [0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["tail"] = [[0, 8, 14, 18, 20, 6, -8, -12, -10, -8, -6, -4, -2, -1, 0, 0][f], 0, 0]
        if 1 <= f <= 7:
            fx.append(("flare", f))
        if 5 <= f <= 9:
            fx.append(("streak", f))
        if 9 <= f <= 11:
            fx.append(("embers", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 2 <= f <= 6 else 0
        P["g"] = [-5.0 * env + jit, 1.0 * env, -14.0 * env]
        P["head"] = [-10.0 * env, -1.0 * env, 1.0 * env]
        P["arms"] = [18.0 * env, 0, 1.0 * env]     # arms fling up and out
        P["tail"] = [-12.0 * env, 0, 0]
        if f == 1:
            fx.append(("flash", 0))
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
    legs = [n for n in R["draw"] if n.startswith("leg")]
    # leg_a = the leg nearer the front of the screen order in the label list
    legs = sorted(legs, key=lambda n: R["label"].index(n))
    arms = sorted([n for n in R["draw"] if n.startswith("arm")], key=lambda n: R["label"].index(n))
    b = P.get("body", zero)
    xf = {}
    for n in R["draw"]:
        rot, df, du = 0.0, 0.0, 0.0
        if n == "body":
            df, du = b[1], b[2]
        elif n == "head":
            h = P.get("head", zero)
            rot, df, du = h[0] * side, h[1], h[2]
        elif n.startswith("arm"):
            if action == "attack":
                a = P["slash"] if n == R["slash"] else P["guard"]
                rot = a[0] * (R["raise_"] if n == R["slash"] else -R["raise_"])
            elif action == "walk":
                a = P["arm_a"] if n == arms[0] else P.get("arm_b", zero)
                rot = a[0] * (1.0 if abs(fwx) > 0.3 else 0.35) * side
            else:
                a = P.get("arms", zero)
                sgn = 1.0 if piv[n][0] < CELL_W // 2 else -1.0       # arms on either side mirror the swing
                rot = a[0] * sgn
            df, du = a[1], a[2] + b[2]
        elif n.startswith("leg"):
            ft = P.get("leg_a" if n == legs[0] else "leg_b", zero)
            df, du = ft[1], ft[2]
        elif n == "tail":
            t = P.get("tail", zero)
            sgn = 1.0 if R["flame"][0] > R["body_x"] else -1.0     # + t lifts the flame tip
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
    return out, fx, track


# ---------------------------------------------------------------- effects
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def line(a, p0, p1, col, every=1):
    k = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    for j in range(0, k + 1, every):
        u = j / k
        put(a, p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u, col)


def flame_tongue(a, x, y, hgt, seed):
    """A small flame licking upward from (x, y): orange-red rim, orange, yellow core."""
    rng = np.random.default_rng(seed)
    for j in range(hgt):
        w = max(0, round((hgt - j) * 0.45))
        sway = round(math.sin(j * 0.9 + seed) * 0.8)
        for i in range(-w, w + 1):
            col = FIRE_R if abs(i) == w else (FIRE_Y if abs(i) <= w // 3 and j < hgt * 0.6 else FIRE_O)
            put(a, x + i + sway, y - j, col)
    for _ in range(2):
        put(a, x + rng.integers(-3, 4), y - hgt - rng.integers(1, 4), FIRE_O)


def draw_fx(a, fx, R, track):
    fwx, fwy = R["fwd"]
    flame = track(R["flame"], "tail")
    for kind, k in fx:
        if kind == "flicker":
            put(a, flame[0] + (1 if k < 8 else -1), flame[1] - 2 - (k % 2), FIRE_O)
            put(a, flame[0] + (-2 if k < 8 else 2), flame[1] - 1 - (k % 2), FIRE_Y)
        elif kind == "flare":
            hgt = [0, 3, 5, 7, 8, 7, 5, 3][k]
            flame_tongue(a, flame[0], flame[1] + 2, hgt, 7 + k)
        elif kind == "streak":
            # three curved, tapered burning claw marks swept from high behind the claw to low ahead of it
            c = track(R["claw"], R["slash"])
            cx = CELL_W / 2
            if fwy > 0.5:               # toward the camera: across the front, upper strike side -> lower centre
                d0, d1 = (cx - 17, FOOT_Y - 48), (cx + 1, FOOT_Y - 18)
            elif fwy < -0.5:            # away from the camera: above the striking shoulder
                d0, d1 = (cx + 6, FOOT_Y - 62), (cx + 20, FOOT_Y - 40)
            else:
                base = (c[0] + fwx * 7, c[1])
                d0, d1 = (base[0] - fwx * 3, base[1] - 17), (base[0] + fwx * 8, base[1] + 9)
            L = math.hypot(d1[0] - d0[0], d1[1] - d0[1])
            px, py = -(d1[1] - d0[1]) / L, (d1[0] - d0[0]) / L
            if px * fwx + py * fwy < 0:
                px, py = -px, -py                  # bulge toward "forward"
            grow = {5: 0.5, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}[k]
            for j, o in enumerate((-3.5, 0, 3.5)):
                steps = int(L * grow) + 1
                for i in range(steps + 1):
                    u = i / max(1, int(L)) 
                    if u > grow:
                        break
                    bul = 3.0 * 4 * u * (1 - u)
                    x = d0[0] + (d1[0] - d0[0]) * u + px * (o + bul)
                    y = d0[1] + (d1[1] - d0[1]) * u + py * (o + bul)
                    mid = 0.2 < u < 0.8
                    if k <= 7:
                        if mid:
                            put(a, x + px, y + py, FIRE_R if k == 7 else FIRE_O)
                        put(a, x, y, FIRE_CORE if (k == 6 and mid) else (FIRE_Y if mid or k == 5 else FIRE_O))
                    elif k == 8 and i % 2 == 0:
                        put(a, x, y, FIRE_O)
                    elif k == 9 and i % 3 == 0:
                        put(a, x, y - 1, FIRE_R)
        elif kind == "embers":
            c = track(R["claw"], R["slash"])
            rng = np.random.default_rng(90 + k)
            for _ in range(7 - (k - 9) * 2):
                ex = c[0] + fwx * 8 + rng.uniform(-9, 9)
                ey = c[1] + rng.uniform(-12, 6) - (k - 9) * 2
                put(a, ex, ey, [FIRE_Y, FIRE_O, FIRE_R][(k - 9 + int(ex)) % 3])


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
        R = RIG[v]
        rig[v] = (cells, piv, place_layer(still[v], v)[0])
        # keep the fx anchor points in cell coordinates
        R["_off"] = (ox, oy)

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        cells, piv, _ = rig[v]
        R = RIG[v]
        ox, oy = R["_off"]
        R_cell = dict(R, claw=(R["claw"][0] + ox, R["claw"][1] + oy), flame=(R["flame"][0] + ox, R["flame"][1] + oy))
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
                draw_fx(a, fx, R_cell, track)
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
        for si, fc in enumerate((0, 4, 6, 12) if act == "attack" else (0, 4, 8, 12)):
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
            a0 = np.array(Image.open(frames_dir / v / act / "00.png").convert("RGBA"))
            dd = int((np.abs(a0.astype(int) - ref.astype(int)).sum(axis=2) > 0).sum())
            if dd:
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
    meta = {"generator": "forge_gen1_0005_charmeleon_actions", "version": VERSION, "key": KEY, "dex": 5,
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
