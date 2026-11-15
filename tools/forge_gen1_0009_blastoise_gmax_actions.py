#!/usr/bin/env python3
"""Gigantamax Blastoise (0009_blastoise-gmax): 8-direction stills + 8-direction x 4-action sheet.

Batch md/gen1-forge.md (2026-11-04), same flow as tools/forge_raichu_mega_x_actions.py:
  art/ref_front.png / ref_back.png  : official front.png / back.png x8 nearest (references for the Antigravity CLI)
  art/agy_prompt.md                 : morphological prompt (no proper nouns, exact part counts and colours)
  art/out/keys_2x2.png              : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
  art/out/sw_single.png             : SW (3/4 front, faces down-left)
  Mirrors: SE = SW, E = W, NW = NE.

Pipeline: key the white backdrop (border-connected only: the shell's white rim and the claws stay) -> one game
scale (S still height = opaque height of the official front.png; SW is matched to the same height) ->
agent-sprite-forge palette_tool build / lock / apply (+2 reserved water colours) -> part rig (giant cannon shell,
head, torso, arms, legs, tail) -> poses at 4x nearest, sampled back to 1x -> QC -> agent-sprite-forge/ -> sync.

Actions (16 frames each):
  idle   : heavy breathing; the shell settles a beat after the torso, arms and tail sway a little.
  walk   : heavy biped stomp (legs alternate), body bob, shell sways, arms and tail swing.
  attack : cannon volley - braces (crouch, shell tilts back, water gathers at the shell's own cannon muzzles)
           -> fires: recoil, muzzle bursts, water shells arc forward out of every cannon -> they land in splashes
           -> recovers.
  hurt   : white flash, recoil backwards (lean <= 6 deg), the shell jolts, recover.

The official front / back png / gif are never written.
run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0009_blastoise_gmax_actions.py [--force] [--no-sync]
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
KEY = "0009_blastoise-gmax"
REL = "monsterSprite/gen1/0009_blastoise/gigantamax/blastoise-gmax"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0009_blastoise_gmax_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 232, 140, 128
UP = 4
WATER_HI, WATER_LO = (200, 240, 255), (80, 176, 240)     # reserved: the cannon water only
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}


def official_height():
    a = np.array(Image.open(FORM_DIR / "front.png").convert("RGBA"))
    ys = np.nonzero((a[..., 3] > 0).any(axis=1))[0]
    return int(ys.max() - ys.min() + 1)


# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# Every pixel starts as "shell"; the polygons (label order, later wins) take the creature's own parts out of it.
# Blue / orange skin left in the shell goes to the nearest body part; shell-brown / rim-white inside the torso,
# arm and leg polygons goes back to the shell.
# draw: back to front.  legs: "a" / "b" alternate in the walk.  muzzles: cannon tips (attack water).
RIG = {
    "S": dict(
        poly={"body": [(26, 36), (74, 36), (78, 60), (72, 80), (28, 80), (22, 60)],
              "tail": [],
              "leg_a": [(8, 60), (42, 62), (44, 90), (6, 90)],
              "leg_b": [(58, 62), (92, 60), (94, 90), (56, 90)],
              "head": [(33, 12), (67, 12), (68, 30), (61, 38), (39, 38), (32, 30)],
              "arm_a": [(3, 32), (33, 32), (34, 50), (24, 58), (3, 58)],
              "arm_b": [(67, 32), (97, 32), (97, 58), (76, 58), (66, 50)]},
        label=["body", "leg_a", "leg_b", "head", "arm_a", "arm_b"],
        draw=["shell", "body", "leg_a", "leg_b", "head", "arm_a", "arm_b"],
        pivot={"shell": (50, 40), "body": (50, 88), "head": (50, 36), "arm_a": (28, 36), "arm_b": (72, 36),
               "leg_a": (26, 62), "leg_b": (74, 62)},
        muzzles=[(4, 10), (96, 10), (48, 2), (20, 8), (80, 8)],
        body_x=50, fwd=(0.0, 1.0), ground=4),
    "SW": dict(
        poly={"body": [(12, 18), (62, 18), (66, 58), (58, 80), (18, 80), (10, 56)],
              "tail": [(78, 63), (96, 63), (96, 79), (80, 79)],
              "leg_b": [(52, 52), (80, 50), (82, 90), (52, 90)],
              "leg_a": [(12, 62), (44, 62), (44, 90), (12, 90)],
              "head": [(0, 10), (30, 10), (35, 22), (32, 38), (16, 42), (0, 42)],
              "arm_a": [(0, 40), (18, 38), (20, 62), (0, 62)],
              "arm_b": [(38, 22), (60, 22), (60, 48), (38, 48)]},
        label=["body", "tail", "leg_b", "leg_a", "head", "arm_a", "arm_b"],
        draw=["tail", "shell", "leg_b", "body", "arm_b", "leg_a", "head", "arm_a"],
        pivot={"shell": (70, 40), "body": (40, 88), "head": (24, 36), "arm_a": (14, 42), "arm_b": (46, 26),
               "leg_a": (28, 64), "leg_b": (66, 54), "tail": (80, 70)},
        muzzles=[(30, 2), (66, 4), (98, 2), (128, 28), (112, 12)],
        body_x=46, fwd=(-0.9, 0.45), ground=3),
    "W": dict(
        poly={"body": [(18, 30), (46, 30), (50, 76), (26, 80), (20, 60)],
              "leg_b": [(72, 74), (102, 74), (104, 90), (72, 90)],
              "tail": [(90, 68), (113, 68), (113, 84), (98, 84)],
              "leg_a": [(36, 58), (68, 58), (68, 90), (34, 90)],
              "head": [(0, 10), (32, 10), (36, 28), (30, 42), (0, 42)],
              "arm_a": [(6, 38), (34, 38), (36, 56), (6, 56)]},
        label=["body", "leg_b", "tail", "leg_a", "head", "arm_a"],
        draw=["leg_b", "tail", "shell", "body", "leg_a", "head", "arm_a"],
        pivot={"shell": (72, 46), "body": (36, 88), "head": (30, 34), "arm_a": (30, 42),
               "leg_a": (50, 60), "leg_b": (86, 76), "tail": (94, 74)},
        muzzles=[(42, 8), (76, 2), (100, 6), (108, 28)],
        body_x=56, fwd=(-1.0, 0.0), ground=-2),
    "NE": dict(
        poly={"body": [(64, 30), (80, 30), (82, 72), (64, 74)],
              "leg_b": [(12, 76), (38, 76), (38, 90), (12, 90)],
              "tail": [(2, 62), (44, 62), (46, 80), (2, 80)],
              "leg_a": [(50, 56), (84, 56), (84, 90), (50, 90)],
              "head": [(68, 4), (94, 4), (94, 30), (80, 34), (68, 30)],
              "arm_a": [(74, 30), (96, 30), (96, 58), (74, 58)]},
        label=["body", "leg_b", "tail", "leg_a", "head", "arm_a"],
        draw=["leg_b", "body", "shell", "tail", "leg_a", "arm_a", "head"],
        pivot={"shell": (40, 40), "body": (72, 88), "head": (74, 30), "arm_a": (78, 34),
               "leg_a": (66, 58), "leg_b": (25, 78), "tail": (40, 70)},
        muzzles=[(4, 20), (30, 2), (66, 2), (6, 52), (14, 36)],
        body_x=55, fwd=(0.9, -0.45), ground=-8),
    "N": dict(
        poly={"body": [(18, 60), (68, 60), (68, 76), (18, 76)],
              "leg_a": [(2, 66), (34, 66), (34, 89), (2, 89)],
              "leg_b": [(52, 66), (84, 66), (84, 89), (52, 89)],
              "tail": [(30, 64), (54, 64), (52, 89), (30, 89)],
              "arm_a": [(0, 28), (14, 28), (14, 58), (0, 58)],
              "arm_b": [(72, 28), (86, 28), (86, 58), (72, 58)]},
        label=["body", "leg_a", "leg_b", "tail", "arm_a", "arm_b"],
        draw=["body", "leg_a", "leg_b", "arm_a", "arm_b", "shell", "tail"],
        pivot={"shell": (43, 40), "body": (43, 88), "arm_a": (12, 30), "arm_b": (74, 30),
               "leg_a": (18, 68), "leg_b": (68, 68), "tail": (42, 66)},
        muzzles=[(43, 2), (6, 12), (80, 12), (4, 24), (82, 24)],
        body_x=43, fwd=(0.0, -1.0), ground=-14),
}
for _v in RIG.values():
    _v["poly"] = {k: p for k, p in _v["poly"].items() if p}


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
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 200])] = 0
    return out


def make_stills(dst: Path, target_h):
    k = np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], "W": k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": np.array(Image.open(ART / "out/sw_single.png").convert("RGB"))}
    keyed = {v: key_white(r) for v, r in src.items()}
    bb = {v: Image.fromarray(a, "RGBA").getbbox() for v, a in keyed.items()}
    grid = target_h / (bb["S"][3] - bb["S"][1])          # S still height = official front height
    scale = {v: grid for v in src}
    scale["SW"] = target_h / (bb["SW"][3] - bb["SW"][1])
    dst.mkdir(parents=True, exist_ok=True)
    for v, a in keyed.items():
        im = Image.fromarray(a, "RGBA").crop(bb[v])
        sm = im.resize((round(im.width * scale[v]), round(im.height * scale[v])), Image.BOX)
        b = np.array(sm)
        b[..., 3] = np.where(b[..., 3] >= 128, 255, 0)
        b[b[..., 3] == 0] = 0
        Image.fromarray(b, "RGBA").save(dst / f"{v}.png")


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


def colour_classes(still):
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    op = still[..., 3] > 0
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    blue = op & (b > r + 40) & (b > 90)                              # skin, cyan markings
    orange = op & (r > 140) & (r - b > 60) & (g < 150)               # belly plates
    brown = op & (mx > 50) & (mx < 125) & (mx - mn < 30) & (r - b >= 5)  # shell plates
    light = op & (mn > 175) & (mx - mn < 30)                          # rim / metal highlights / claws
    return blue, orange, brown, light


def build_rig(view, still):
    R = RIG[view]
    op = still[..., 3] > 0
    shape = op.shape
    names = R["draw"]
    idx = {n: i + 1 for i, n in enumerate(names)}
    blue, orange, brown, light = colour_classes(still)
    lab = np.zeros(shape, int)
    lab[op] = idx["shell"]
    for n in R["label"]:
        lab[poly_mask(R["poly"][n], shape) & op] = idx[n]
    # shell plates / rim inside torso, arm and leg polygons stay on the shell
    for n in R["label"]:
        if n == "head":
            continue
        pm = lab == idx[n]
        back = pm & (brown | (light & (n in ("body", "tail"))))
        back = ndi.binary_opening(back, structure=np.ones((2, 2), bool))     # not single dark specks
        lab[back] = idx["shell"]
    # skin / belly left in the shell goes to the nearest body part (only near one)
    skin = (lab == idx["shell"]) & (blue | orange)
    if skin.any():
        other = op & (lab != idx["shell"])
        d, (iy, ix) = ndi.distance_transform_edt(~other, return_indices=True)
        mv = skin & (d <= 6)
        lab[mv] = lab[iy[mv], ix[mv]]
    # the dark outline next to a part follows that part when the shell pixel is only outline
    masks = {n: lab == idx[n] for n in names}

    layers = {}
    for i, n in enumerate(names):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in names[i + 1:]:
            above |= masks[n2]
        cushion = {"shell": 6, "body": 5, "head": 3}.get(n, 2)
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion) & op
        if seam.any() and m.any():
            r, g, b = [still[..., c].astype(int) for c in range(3)]
            fillable = m & (np.maximum(np.maximum(r, g), b) > 60)
            if n == "shell":
                fillable &= brown                    # plate colour, not rim or metal
            if not fillable.any():
                fillable = m
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
def wave(amp, cycles, f, phase=0.0):
    """0 at frame 0 and at the loop end."""
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def pose(action, f):
    """Per part: (rot deg, fwd px, up px). Global: (fwd px, up px, lean deg, + = forward). Effects list."""
    P = {}
    G = [0.0, 0.0, 0.0]
    fx = []
    ph = 2 * math.pi * f / N
    if action == "idle":
        P["body"] = (0, 0, wave(0.9, 1, f))
        P["head"] = (0, 0, wave(1.0, 1, f, -0.6))
        P["shell"] = (0, 0, wave(1.1, 1, f, -1.0))
        P["arms"] = (wave(2.0, 1, f, -0.8), 0, wave(1.0, 1, f, -0.4))
        P["tail"] = (wave(4.0, 1, f, -0.3), 0, 0)
    elif action == "walk":
        bob = 1.6 * abs(math.sin(ph))
        P["body"] = (0, 0, bob)
        P["head"] = (0, 0.6 * math.sin(2 * ph), bob)
        P["shell"] = (1.2 * math.sin(ph), 0, 1.4 * abs(math.sin(ph - 0.4)))
        P["leg_a"] = (0, 2.6 * math.sin(ph), 2.8 * max(0.0, math.cos(ph)))
        P["leg_b"] = (0, -2.6 * math.sin(ph), 2.8 * max(0.0, -math.cos(ph)))
        P["arm_a"] = (-6.0 * math.sin(ph), 0, bob)
        P["arm_b"] = (6.0 * math.sin(ph), 0, bob)
        P["tail"] = (7.0 * math.sin(ph - 0.6), 0, 0)
    elif action == "attack":
        g_f = [0, -1, -1, -2, -4, -5, -4, -3, -2, -2, -1, -1, 0, 0, 0, 0][f]
        g_u = [0, -1, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -1, -2, -3, -5, -6, -5, -4, -3, -2, -2, -1, -1, 0, 0, 0][f]
        G = [g_f, g_u, lean]
        sh = [0, -2, -4, -5, 3, 1, 3, 1, 2, 1, 0, 0, 0, 0, 0, 0][f]        # tilt back to aim, kicks on each shot
        P["shell"] = (sh, [0, 0, 0, 0, -2, -1, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0][f], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f])
        P["head"] = (0, [0, 0, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0][f], [0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f])
        arm = [0, 6, 12, 14, 8, 8, 8, 8, 6, 4, 3, 2, 1, 0, 0, 0][f]           # arms brace outward
        P["arm_a"] = (-arm, 0, [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f])
        P["arm_b"] = (arm, 0, P["arm_a"][2])
        P["leg_a"] = (0, [0, 0, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0][f], 0)   # braced stance
        P["tail"] = ([0, -3, -6, -8, 4, 6, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0][f], 0, 0)
        if 1 <= f <= 3:
            fx.append(("charge", f))
        if 4 <= f <= 6:
            fx.append(("muzzle", f))
        if 5 <= f <= 12:
            fx.append(("shells", f))
        if 10 <= f <= 14:
            fx.append(("splash", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0
        G = [-5.0 * env + jit, 1.0 * env, -6.0 * env]
        P["head"] = (0, -1.0 * env, 1.0 * env)
        lag = math.sin(math.pi * min(1.0, max(0.0, (f - 1) / 12.0))) if f < 13 else 0.0
        P["shell"] = (-3.0 * lag, 0, 1.0 * lag)
        P["arm_a"] = (8.0 * env, 0, 1.0 * env)
        P["arm_b"] = (-8.0 * env, 0, 1.0 * env)
        P["tail"] = (-6.0 * lag, 0, 0)
        if f == 1:
            fx.append(("flash", 0))
    return P, G, fx


def up4(a):
    return np.repeat(np.repeat(a, UP, 0), UP, 1)


def rot4(img4, ang, piv):
    """img4 at 4x; rotate (+ = clockwise on screen) about piv (1x coords)."""
    if abs(ang) < 0.3:
        return img4
    im = Image.fromarray(img4, "RGBA").rotate(-ang, resample=Image.NEAREST,
                                              center=(piv[0] * UP + UP / 2, piv[1] * UP + UP / 2))
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
    P, G, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    side = 0.0 if abs(fwx) < 0.5 else (-1.0 if fwx < 0 else 1.0)
    diag = 0.6 if (abs(fwx) > 0.5 and abs(fwy) > 0.3) else 1.0
    zero = (0.0, 0.0, 0.0)
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    offs = {}
    for n in R["draw"]:
        t = P.get(n, zero)
        if n.startswith("arm") and n not in P:
            t = P.get("arms", zero)
            if n == "arm_b" or (side != 0.0):
                t = (-t[0] if n == "arm_b" else t[0], t[1], t[2])
        if n.startswith("leg") and action == "attack" and n != "leg_a":
            t = zero
        rot, df, du = t
        if side == 0.0:
            dx, dy = 0.0, -du + (df * fwy * 0.3 if not n.startswith("leg") else 0.0)
            if action == "walk" and n == "shell":
                dx = rot * 0.8                                    # front / back: the shell sways sideways
                rot = 0.0
        else:
            dx, dy = df * fwx, df * fwy * 0.4 - du
        if n == "shell":
            ang = rot * side * diag if side != 0.0 else 0.0
        elif n == "tail":
            ang = rot * (side if side != 0.0 else 1.0)
        elif n.startswith("arm"):
            # front / back: arm_a (screen left) swings out with -rot, arm_b mirrored; side views: swing along fwd
            ang = rot if side == 0.0 else rot * 0.6
        else:
            ang = 0.0
        offs[n] = (dx, dy)
        l4 = rot4(up4(cells[n]), ang, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = G
    if side == 0.0:
        gf *= 0.4
    if abs(lean) > 0.3 and side != 0.0:
        pv = (CELL_W // 2, FOOT_Y)
        canvas = rot4(canvas, side * lean * (0.75 if diag < 1 else 1.0), pv)
    gdx, gdy = gf * fwx, gf * fwy - gu
    canvas = shift4(canvas, gdx, gdy)
    out = canvas[1::UP, 1::UP].copy()
    return out, fx, offs, (gdx, gdy)


# ---------------------------------------------------------------- effects (reserved water colours only)
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 2 <= x < a.shape[1] - 2 and 2 <= y < a.shape[0] - 2:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def blob(a, x, y, r):
    for u in range(-r, r + 1):
        for v in range(-r, r + 1):
            d = abs(u) + abs(v)
            if d <= r:
                put(a, x + u, y + v, WATER_HI if d <= r - 1 and (u <= 0 and v <= 0) else WATER_LO)


def ring(a, cx, cy, rx, ry, seed, n=32):
    rng = np.random.default_rng(seed)
    for j in range(n):
        t = 2 * math.pi * j / n
        if rng.random() < 0.15:
            continue
        put(a, cx + rx * math.cos(t), cy + ry * math.sin(t), WATER_HI if j % 2 else WATER_LO)


def bez(p0, p1, p2, u):
    return ((1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * p1[0] + u * u * p2[0],
            (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * p1[1] + u * u * p2[1])


def draw_fx(a, fx, ctx):
    for kind, k in fx:
        if kind == "charge":                       # droplets drawn into each muzzle
            for i, (mx, my) in enumerate(ctx["muzzles"]):
                r = 9 - 2.2 * k
                for j in range(5):
                    t = j * 2 * math.pi / 5 + i + k * 0.5
                    put(a, mx + math.cos(t) * r, my + math.sin(t) * r, WATER_LO if j % 2 else WATER_HI)
                if k == 3:
                    put(a, mx, my, WATER_HI)
        elif kind == "muzzle":                     # jet + burst at each cannon tip
            for i, ((mx, my), (p0, p1, p2)) in enumerate(zip(ctx["muzzles"], ctx["paths"])):
                tx, ty = p1[0] - p0[0], p1[1] - p0[1]
                L = math.hypot(tx, ty) or 1
                tx, ty = tx / L, ty / L
                nx, ny = -ty, tx
                ln = [7, 11, 6][k - 4]
                for q in range(ln):
                    w = 1 if q < ln - 2 else 0
                    for o in range(-w, w + 1):
                        put(a, mx + tx * q + nx * o, my + ty * q + ny * o, WATER_HI if o == 0 else WATER_LO)
                r = 2 + (k - 4) * 2
                ring(a, mx, my, r, r, 50 + i + k, n=12 + 4 * (k - 4))
        elif kind == "shells":                     # water shells arc from the muzzles to their landing points
            for i, ((p0, p1, p2), delay) in enumerate(zip(ctx["paths"], ctx["delays"])):
                u = (k - 5 - delay) / 5.0
                if u < 0 or u > 1.0:
                    continue
                for tr, col in ((0.24, WATER_LO), (0.18, WATER_LO), (0.12, WATER_HI), (0.06, WATER_HI)):
                    if u - tr >= 0:
                        x, y = bez(p0, p1, p2, u - tr)
                        put(a, x, y, col)
                        put(a, x + 1, y, WATER_LO)
                x, y = bez(p0, p1, p2, u)
                blob(a, x, y, 3 if u < 0.85 else 2)
        elif kind == "splash":
            for i, ((_, _, p2), delay) in enumerate(zip(ctx["paths"], ctx["delays"])):
                j = k - 10 - delay
                if j < 0 or j > 2:
                    continue
                cx, cy = p2
                ring(a, cx, cy, 4 + 4 * j, 1.5 + 1.2 * j, 70 + i + k, n=28)
                if j < 2:
                    ring(a, cx, cy, 2 + 2 * j, 1 + 0.6 * j, 80 + i + k, n=14)
                for q in range(-3, 4):
                    hgt = (4 - abs(q)) * (3 - j) + 2 * j
                    for z in range(0, 2 if j == 0 else 1):
                        put(a, cx + q * (2 + j), cy - hgt - z, WATER_HI if (q + z) % 2 == 0 else WATER_LO)


def flash(a, col):
    out = a.copy()
    out[a[..., 3] > 0, :3] = col
    return out


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    for n in ("out/keys_2x2.png", "out/sw_single.png"):
        h.update((ART / n).read_bytes())
    return h.hexdigest()[:12]


def nearest(cols, target):
    arr = np.array(sorted(cols), float)
    return tuple(int(v) for v in arr[np.argmin(((arr - np.array(target, float)) ** 2).sum(1))])


def volley(view, ox, oy):
    """Bezier paths (muzzle -> apex -> landing) for each cannon + launch delays (cell coords at rest)."""
    R = RIG[view]
    fwx, fwy = R["fwd"]
    cx = CELL_W / 2
    ground = FOOT_Y + R["ground"]
    paths, delays = [], []
    for i, (mx, my) in enumerate(R["muzzles"]):
        m = (mx + ox, my + oy)
        if abs(fwx) < 0.5:          # front / back: a fan out to both sides
            s = -1.0 if m[0] < cx - 3 else (1.0 if m[0] > cx + 3 else (1.0 if i % 2 else -1.0))
            land = (cx + s * (52 + 9 * (i % 3)), ground + (i % 2) * 3 * (1 if fwy > 0 else -1))
        else:                       # side / diagonal: forward
            land = (cx + fwx * (66 + 8 * (i % 3)) - fwx * 4 * (i // 3), ground + fwy * 6 + (i % 2) * 3)
        apex = ((m[0] + land[0]) / 2, min(m[1], land[1]) - 22 - 4 * (i % 2))
        paths.append((m, apex, land))
        delays.append(i % 3)
    return paths, delays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--stills-only", action="store_true")
    args = ap.parse_args()

    work = OUT_DIR / "_work"
    h = state_hash()
    st_p = work / "state.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {}
    if args.force or st.get("hash") != h:
        keep_stills = st.get("stills_src") == stills_src_hash() and (work / "stills").exists()
        if work.exists():
            for d in work.iterdir():
                if keep_stills and d.name in ("stills_raw", "stills", "palette", "palette_lock"):
                    continue
                shutil.rmtree(d) if d.is_dir() else d.unlink()
        st = {"hash": h, "stills_src": stills_src_hash(), "done": ["stills"] if keep_stills else []}
    work.mkdir(parents=True, exist_ok=True)
    save_st = lambda: st_p.write_text(json.dumps(st))
    target_h = official_height()

    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    if "stills" not in st["done"]:
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        make_stills(raw, target_h)
        run_tool(["build", "--input", str(raw), "--colors", "32", "--reserve", "#%02x%02x%02x" % WATER_HI, "#%02x%02x%02x" % WATER_LO,
                  "--balance", "--name", "blastoise-gmax", "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}

    pj = json.loads((lock_dir / "palette.json").read_text())
    cols = set()
    for c in pj.get("colors", []):
        hx = (c["hex"] if isinstance(c, dict) else c).lstrip("#")
        cols.add((int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)))
    body_cols = cols - {WATER_HI, WATER_LO}
    white = nearest(body_cols, (255, 255, 255))

    rig = {}
    dbg = work / "rig"
    dbg.mkdir(exist_ok=True)
    for v in RIG:
        layers, masks = build_rig(v, still[v])
        cells, piv = {}, {}
        for n, lyr in layers.items():
            cells[n], (ox, oy) = place_layer(lyr, v)
        for n, p in RIG[v]["pivot"].items():
            piv[n] = (p[0] + ox, p[1] + oy)
        rig[v] = (cells, piv, (ox, oy), masks)
        pal = [(200, 60, 60), (240, 200, 40), (60, 120, 220), (40, 180, 90), (180, 80, 200), (240, 140, 40), (90, 220, 220), (120, 120, 120)]
        lab = np.zeros(still[v].shape[:2] + (3,), np.uint8)
        for i, (n, m) in enumerate(masks.items()):
            lab[m] = pal[i % len(pal)]
        Image.fromarray(np.concatenate([still[v][..., :3], lab], 1)).resize(((still[v].shape[1] * 2) * 4, still[v].shape[0] * 4), Image.NEAREST).save(dbg / f"{v}.png")
    if args.stills_only:
        return

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        cells, piv, (ox, oy), masks = rig[v]
        paths, delays = volley(v, ox, oy)
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, offs, (gx, gy) = compose(v, cells, piv, act, f)
                so = offs.get("shell", (0, 0))
                mz = [(mx + ox + gx + so[0], my + oy + gy + so[1]) for mx, my in RIG[v]["muzzles"]]
                pth = [((p0[0] + gx, p0[1] + gy), p1, p2) for p0, p1, p2 in paths]
                ctx = {"muzzles": mz, "paths": pth, "delays": delays}
                if any(k == "flash" for k, _ in fx):
                    a = flash(a, white)
                draw_fx(a, fx, ctx)
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
        for si, fc in enumerate((0, 4, 8, 12) if act != "attack" else (3, 5, 7, 10)):
            src = sheets[act][:, fc * CELL_W:(fc + 1) * CELL_W]
            mm = src[..., 3] > 0
            pv[:, (ai * 4 + si) * CELL_W:(ai * 4 + si + 1) * CELL_W][mm] = src[mm]
    Image.fromarray(pv, "RGBA").save(OUT_DIR / "preview.png")

    idle0 = [sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W] for di in range(8)]
    Image.fromarray(np.concatenate(idle0, axis=1), "RGBA").save(FORM_DIR / "8dir_stills.png")

    # 5. QC
    qa = work / "qa"; qa.mkdir(exist_ok=True)
    aa = master[..., 3] > 0
    used = {tuple(int(v) for v in c) for c in np.unique(master[aa][:, :3], axis=0)}
    off = sorted(used - cols)
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
        fr = np.array(Image.open(frames_dir / v / "idle" / "00.png").convert("RGBA"))
        still_diff[v] = int((np.abs(ref.astype(int) - fr.astype(int)).sum(2) > 0).sum())
    alpha_ok = bool(np.isin(master[..., 3], [0, 255]).all())
    sprite_h_S = int(np.ptp(np.nonzero(idle0[0][..., 3].any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off),
              "edge_touching_frames": clip, "still_diff_px": still_diff, "alpha_binary": alpha_ok,
              "sprite_h_S": sprite_h_S, "official_front_h": target_h}
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
        c = idle0[di][..., 3] > 0
        ys, xs = np.nonzero(c)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    meta = {"generator": "forge_gen1_0009_blastoise_gmax_actions", "version": VERSION, "key": KEY, "dex": 9,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["front.png", "back.png"],
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h_S, "frames": N, "frame_time": AD,
            "directions": DIRS, "direction_sources": DIR_SRC, "actions": ACTIONS, "palette_colors": len(cols),
            "off_palette_colors": len(off), "idle_bounds_S_to_SW": bounds}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) or k in ("still_diff_px", "edge_touching_frames") else (v.get("policy") if "policy" in v else len(v))) for k, v in report.items()}, ensure_ascii=False))

    if not args.no_sync and SYNC_DIR.parent.exists():
        if SYNC_DIR.exists():
            shutil.rmtree(SYNC_DIR)
        SYNC_DIR.mkdir(parents=True)
        for fp in OUT_DIR.iterdir():
            if fp.is_file():
                shutil.copy2(fp, SYNC_DIR / fp.name)
        print("synced ->", SYNC_DIR)


def stills_src_hash():
    """Stills / palette only depend on the art and make_stills / key_white (cached across rig edits)."""
    import inspect
    h = hashlib.sha1((inspect.getsource(key_white) + inspect.getsource(make_stills)).encode())
    for n in ("out/keys_2x2.png", "out/sw_single.png"):
        h.update((ART / n).read_bytes())
    return h.hexdigest()[:12]


if __name__ == "__main__":
    main()
