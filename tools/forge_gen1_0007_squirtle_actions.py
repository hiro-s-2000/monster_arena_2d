#!/usr/bin/env python3
"""0007_squirtle (gen1 normal): 8-direction x 4-action sheet, Antigravity key art + agent-sprite-forge.

Batch procedure md/gen1-forge.md (2026-11-04); structure copied from tools/forge_raichu_mega_x_actions.py, rig /
colours / attack built from this form's own art (small bipedal turtle: big head, 2 arms, 2 legs, 1 shell,
1 curled tail).

Source:
  art/ref_front.png, art/ref_back.png : official front.png / back.png x8 nearest (references only)
  art/agy_prompt.md                   : morphological prompt (no proper nouns) for the Antigravity CLI (`agy -p`)
  art/out/keys_2x2.png                : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
  art/out/sw_single.png               : SW
  Mirrors: SE = SW, E = W, NW = NE.

Pipeline:
  1. key the white backdrop (border-connected only), crop, "mode" downscale (each sprite pixel takes the dominant
     colour of its block, thin dark lines win) + 1 px outer outline. S height = opaque height of the official
     front.png (39 px); W / NE / N share the grid scale, SW is matched on height.
  2. agent-sprite-forge palette_tool build (24 colours incl. reserved water / flash colours) -> lock -> apply.
  3. part rig per view: tail (pivot at the rump), body (shell + belly plate), head, two arms, two feet
     (W: near / far leg; NE: one visible arm). Hidden seams get cushions from the nearest own colour.
  4. poses at 4x nearest, sampled back to 1x.
     idle   : breathing (head bob), tail curl sway, small arm sway.
     walk   : alternating stubby steps, arm swing, tail wag.
     attack : water jet: rear back (head up, arms back, tail braced) -> lunge, arms thrust forward, mouth opens
              (mouth_open.py, max 2 px) -> water jet shoots forward from the mouth -> splash burst -> recover.
     hurt   : white flash, recoil back (lean <= 10 deg), droplets knocked off, recover.
  5. QC: off-palette, edge touch, still diff (idle f0 == still), gait_loop.py on idle / walk.

Writes <form>/8dir_stills.png and <form>/agent-sprite-forge/ (sheet.png = 4 actions x 8 rows, preview.png,
meta.json, idle.png ...). The official front/back png/gif are never touched. Syncs agent-sprite-forge/ (without
_work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0007_squirtle_actions.py [--force] [--no-sync]
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

KEY = "0007_squirtle"
REL = "monsterSprite/gen1/0007_squirtle/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0007_squirtle_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 128, 88, 78
UP = 4
WATER_HI = (226, 248, 255)         # reserved: water core / spray
WATER = (72, 160, 236)             # reserved: water body
FLASH = (255, 255, 255)            # reserved: hit flash
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each still)
# label: later polygons win. draw: back to front. body_x: placed on the cell centre. fwd: screen "forward".
# mouth: closed mouth line hinge -> lip (None = not visible).
RIG = {
    "S": dict(
        poly={"tail": [(28, 19), (38, 19), (38, 34), (27, 34), (27, 27)],
              "body": [(6, 17), (28, 17), (28, 36), (6, 36)],
              "head": [(5, 0), (26, 0), (26, 13), (22, 17), (8, 17), (5, 13)],
              "foot_l": [(3, 31), (11, 31), (13, 35), (14, 39), (3, 39)],
              "foot_r": [(20, 31), (28, 31), (30, 34), (30, 39), (20, 39)],
              "arm_l": [(0, 14), (8, 14), (10, 17), (10, 22), (0, 22)],
              "arm_r": [(21, 15), (31, 15), (31, 22), (27, 25), (21, 24)]},
        label=["tail", "body", "head", "foot_l", "foot_r", "arm_l", "arm_r"],
        draw=["tail", "body", "foot_l", "foot_r", "head", "arm_l", "arm_r"],
        pivot={"tail": (28, 28), "head": (15, 17), "body": (17, 36), "foot_l": (9, 37), "foot_r": (25, 37),
               "arm_l": (8, 19), "arm_r": (22, 19)},
        body_x=17, fwd=(0.0, 1.0), mouth=[(10, 15), (14, 16), (19, 15)], peak=0.5, back=False),
    "SW": dict(
        poly={"tail": [(31, 21), (40, 21), (40, 37), (30, 37), (30, 28)],
              "body": [(6, 17), (24, 12), (32, 12), (32, 34), (6, 34)],
              "head": [(5, 0), (25, 0), (26, 13), (22, 17), (8, 17), (5, 14)],
              "foot_l": [(5, 29), (14, 29), (16, 34), (16, 39), (5, 39)],
              "foot_r": [(20, 28), (29, 28), (30, 39), (20, 39)],
              "arm_l": [(0, 15), (10, 15), (11, 19), (10, 24), (0, 24)],
              "arm_r": [(17, 17), (25, 17), (25, 26), (19, 26), (16, 24)]},
        label=["tail", "body", "head", "foot_l", "foot_r", "arm_l", "arm_r"],
        draw=["tail", "body", "foot_l", "foot_r", "head", "arm_l", "arm_r"],
        pivot={"tail": (31, 30), "head": (14, 17), "body": (18, 34), "foot_l": (10, 37), "foot_r": (25, 37),
               "arm_l": (9, 19), "arm_r": (23, 19)},
        body_x=18, fwd=(-0.9, 0.45), mouth=[(17, 14), (11, 14), (7, 13)], peak=0.6, back=False),
    "W": dict(
        poly={"tail": [(28, 22), (37, 22), (37, 37), (25, 37), (25, 30)],
              "body": [(7, 14), (16, 11), (28, 13), (30, 32), (7, 32)],
              "head": [(0, 0), (19, 0), (21, 11), (16, 14), (15, 17), (4, 17), (0, 14)],
              "foot_b": [(17, 30), (25, 30), (25, 38), (17, 38)],
              "foot_a": [(6, 29), (17, 29), (17, 39), (6, 39)],
              "arm_l": [(1, 16), (14, 16), (16, 19), (14, 24), (1, 24)]},
        label=["tail", "body", "head", "foot_b", "foot_a", "arm_l"],
        draw=["tail", "foot_b", "body", "foot_a", "head", "arm_l"],
        pivot={"tail": (27, 30), "head": (12, 15), "body": (16, 32), "foot_a": (12, 37), "foot_b": (21, 36),
               "arm_l": (14, 20)},
        body_x=16, fwd=(-1.0, 0.0), mouth=[(9, 13), (1, 13)], peak=0.65, back=False),
    "NE": dict(
        poly={"tail": [(0, 22), (11, 22), (13, 28), (13, 35), (0, 35)],
              "body": [(8, 9), (30, 12), (31, 33), (11, 33), (10, 22), (7, 20)],
              "head": [(14, 0), (32, 0), (36, 10), (34, 15), (18, 12), (14, 8)],
              "foot_l": [(8, 31), (17, 31), (17, 37), (8, 37)],
              "foot_r": [(21, 29), (30, 29), (30, 37), (21, 37)],
              "arm_r": [(29, 15), (37, 15), (37, 23), (29, 23)]},
        label=["tail", "head", "body", "foot_l", "foot_r", "arm_r"],
        draw=["foot_l", "foot_r", "head", "arm_r", "body", "tail"],
        pivot={"tail": (11, 28), "head": (26, 14), "body": (20, 33), "foot_l": (12, 35), "foot_r": (25, 35),
               "arm_r": (29, 19)},
        body_x=20, fwd=(0.9, -0.45), mouth=None, peak=0.5, back=True),
    "N": dict(
        poly={"tail": [(13, 27), (23, 27), (23, 37), (13, 37)],
              "head": [(7, 0), (28, 0), (28, 12), (7, 12)],
              "body": [(6, 11), (29, 11), (29, 31), (6, 31)],
              "arm_l": [(0, 10), (9, 10), (10, 21), (0, 21)],
              "arm_r": [(26, 10), (35, 10), (35, 21), (25, 21)],
              "foot_l": [(4, 28), (13, 28), (13, 37), (4, 37)],
              "foot_r": [(23, 28), (32, 28), (32, 37), (23, 37)]},
        label=["head", "body", "tail", "arm_l", "arm_r", "foot_l", "foot_r"],
        draw=["foot_l", "foot_r", "head", "arm_l", "arm_r", "body", "tail"],
        pivot={"tail": (18, 29), "head": (17, 12), "body": (17, 31), "foot_l": (9, 35), "foot_r": (27, 35),
               "arm_l": (8, 16), "arm_r": (27, 16)},
        body_x=17, fwd=(0.0, -1.0), mouth=None, peak=0.5, back=True),
}


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
def key_white(rgb):
    a = rgb.astype(int)
    white = a.min(axis=2) > 225
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    bg |= ndi.binary_dilation(bg, iterations=2) & (a.min(axis=2) > 200)
    out = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    lab2, n2 = ndi.label(out[..., 3] > 0)
    if n2:
        sz = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 200])] = 0
    return out


def mode_down(a, s):
    """Each output pixel takes the dominant (4-bit quantised) colour of its source block; dark lines win at 30 %."""
    H, W = a.shape[:2]
    oh, ow = round(H * s), round(W * s)
    out = np.zeros((oh, ow, 4), np.uint8)
    q = (a[..., 0].astype(int) >> 4) * 256 + (a[..., 1].astype(int) >> 4) * 16 + (a[..., 2].astype(int) >> 4)
    for i in range(oh):
        y0 = int(i / s); y1 = max(y0 + 1, int((i + 1) / s))
        for j in range(ow):
            x0 = int(j / s); x1 = max(x0 + 1, int((j + 1) / s))
            blk = a[y0:y1, x0:x1].reshape(-1, 4)
            qq = q[y0:y1, x0:x1].reshape(-1)
            op = blk[:, 3] > 0
            if op.mean() < 0.5:
                continue
            px, qk = blk[op], qq[op]
            dark = px[:, :3].max(axis=1) < 70
            if dark.mean() >= 0.3:
                sel = px[dark]
            else:
                ks, cs = np.unique(qk, return_counts=True)
                sel = px[qk == ks[cs.argmax()]]
            out[i, j, :3] = sel[:, :3].mean(axis=0)
            out[i, j, 3] = 255
    return out


def outline(a, col=(16, 16, 16)):
    op = a[..., 3] > 0
    ring = ndi.binary_dilation(op) & ~op
    b = a.copy()
    b[ring, :3] = col
    b[ring, 3] = 255
    return b


def official_height():
    a = np.array(Image.open(FORM_DIR / "front.png").convert("RGBA"))
    ys = np.nonzero((a[..., 3] > 0).any(axis=1))[0]
    return int(ys.max() - ys.min() + 1)


def make_stills(dst: Path):
    k = np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], "W": k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": np.array(Image.open(ART / "out/sw_single.png").convert("RGB"))}
    crops = {}
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb), "RGBA")
        crops[v] = np.array(im.crop(im.getbbox()))
    target = official_height()

    def render(v, s):
        a = outline(mode_down(crops[v], s))
        ys, xs = np.nonzero(a[..., 3])
        return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    def fit(v):            # scale whose rendered height equals the official height
        base = target / crops[v].shape[0]
        for d in np.linspace(-0.06, 0.06, 25):
            s = base * (1 + d) if d else base
            if render(v, s).shape[0] == target:
                return s
        raise SystemExit(f"cannot fit {v} to {target}px")
    g = fit("S")
    scale = {v: g for v in src}
    scale["SW"] = fit("SW")
    dst.mkdir(parents=True, exist_ok=True)
    for v in src:
        Image.fromarray(render(v, scale[v]), "RGBA").save(dst / f"{v}.png")
    return {v: float(s) for v, s in scale.items()}


# ---------------------------------------------------------------- palette
def run_tool(args):
    r = subprocess.run([str(FPY), "-I", str(PALETTE_TOOL)] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"palette_tool {args[0]} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def load_palette(lock_dir):
    pj = json.loads((lock_dir / "palette.json").read_text())
    cols = []
    for c in pj.get("colors", []):
        hx = (c["hex"] if isinstance(c, dict) else c).lstrip("#")
        cols.append((int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)))
    return cols


def nearest(cols, rgb):
    a = np.array(cols, float)
    return tuple(int(v) for v in cols[int(((a - np.array(rgb, float)) ** 2).sum(1).argmin())])


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
    masks = {n: lab == (i + 1) for i, n in enumerate(names)}
    layers = {}
    order = R["draw"]
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    bright = np.maximum(np.maximum(r, g), b) > 70
    for i, n in enumerate(order):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in order[i + 1:]:
            above |= masks[n2]
        cushion = 3 if n in ("tail", "body") else 2
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion)
        src_m = m
        if n == "body" and m.any():
            # underlay inside the closed body + head outline, so arms / feet / head can move without holes
            bh = m | masks.get("head", np.zeros_like(m))
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(bh, 4), iterations=4))[4:-4, 4:-4]
            seam |= above & ~m & inside & ~masks.get("head", np.zeros_like(m))
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
    oy = FOOT_Y - (h - 1)
    cell[oy:oy + h, ox:ox + w] = layer
    return cell, (ox, oy)


# ---------------------------------------------------------------- 4. posing
def wave(amp, cycles, f, phase=0.0):
    """0 at f = 0 and at the loop end."""
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def pose(action, f):
    """part -> [rot deg (+ = raise), fwd px, up px]; 'g' = [forward px, up px, lean deg]; mouth px."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    mouth = 0
    if action == "idle":
        br = wave(0.6, 1, f, -math.pi / 2)            # 0 .. 1.2 breathing (1 - cos)
        P["head"] = [0, 0, round(br)]
        P["arms"] = [wave(5.0, 1, f, -math.pi / 2), 0, round(br)]
        P["tail"] = [wave(5.0, 1, f, 0.0), 0, 0]
    elif action == "walk":
        ph = 2 * math.pi * f / N
        up = round(1.0 * abs(math.sin(ph)))
        P["body"] = [0, 0, up]
        P["head"] = [0, 0, up]
        P["foot_a"] = [0, 2.0 * math.sin(ph), 2.0 * max(0.0, math.sin(ph))]       # swing forward lifted
        P["foot_b"] = [0, -2.0 * math.sin(ph), 2.0 * max(0.0, -math.sin(ph))]
        P["arm_a"] = [wave(12.0, 1, f, 0.0), 0, up]
        P["arm_b"] = [-wave(12.0, 1, f, 0.0), 0, up]
        P["tail"] = [wave(8.0, 2, f, 0.0), 0, up * 0.5]
    elif action == "attack":
        g_f = [0, -1, -2, -3, 1, 3, 4, 4, 4, 4, 3, 3, 2, 1, 1, 0][f]
        g_u = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -3, -5, -6, 2, 5, 7, 7, 7, 6, 4, 3, 2, 1, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["head"] = [0, [0, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0][f], [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["arms"] = [[0, 8, 14, 18, 4, -10, -14, -14, -12, -10, -8, -5, -3, -2, 0, 0][f],
                     [0, -1, -1, -1, 1, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0][f], 0]
        P["foot_a"] = [0, [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0][f], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["tail"] = [[0, 8, 16, 20, 10, -6, -10, -10, -8, -6, -4, -2, 0, 0, 0, 0][f], 0, 0]
        mouth = [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0][f]
        if 1 <= f <= 3:
            fx.append(("charge", f))
        if 4 <= f <= 11:
            fx.append(("jet", f))
        if 8 <= f <= 11:
            fx.append(("splash", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 5 else 0
        P["g"] = [-4.0 * env + jit, 1.0 * env, -10.0 * env]
        P["head"] = [0, -1.0 * env, 1.0 * env]
        P["arms"] = [22.0 * env, 0, 1.0 * env]
        P["tail"] = [-12.0 * env, 0, 0]
        if f == 1:
            fx.append(("flash", 0))
        if 1 <= f <= 4:
            fx.append(("hit", f))
        if 3 <= f <= 8:
            fx.append(("drops", f))
    return P, fx, mouth


def up4(a):
    return np.repeat(np.repeat(a, UP, 0), UP, 1)


def rot4(img4, ang, piv):
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


def drop_crumbs(a, max_px=2):
    """Remove tiny detached pixel groups left by nearest rotation (never part of the still)."""
    op = a[..., 3] > 0
    lab, n = ndi.label(op, structure=np.ones((3, 3)))
    if n <= 1:
        return a
    sz = ndi.sum(op, lab, range(1, n + 1))
    small = np.isin(lab, [i + 1 for i, s in enumerate(sz) if s <= max_px])
    if small.any():
        a = a.copy()
        a[small] = 0
    return a


def shear_lean(a, ang):
    """Lean the whole sprite by `ang` degrees (+ = top to the right) with integer row shifts about FOOT_Y."""
    if abs(ang) < 0.3:
        return a
    t = math.tan(math.radians(ang))
    out = np.zeros_like(a)
    w = a.shape[1]
    for y in range(a.shape[0]):
        s = int(round(t * (FOOT_Y - y)))
        if s >= 0:
            out[y, s:] = a[y, :w - s]
        else:
            out[y, :w + s] = a[y, -s:]
    return out


def shear_points(points, ang):
    t = math.tan(math.radians(ang))
    return [(x + round(t * (FOOT_Y - y)), y) for x, y in points]


def tail_side(view):
    """+1 when the tail sticks out to the right of its pivot (rotating it counter-clockwise raises it)."""
    xs = [p[0] for p in RIG[view]["poly"]["tail"]]
    return 1.0 if sum(xs) / len(xs) > RIG[view]["pivot"]["tail"][0] else -1.0


def compose(view, cells, piv, action, f):
    R = RIG[view]
    P, fx, mouth = pose(action, f)
    fwx, fwy = R["fwd"]
    zero = [0.0, 0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    feet = sorted(n for n in R["draw"] if n.startswith("foot"))
    arms = sorted(n for n in R["draw"] if n.startswith("arm"))
    b = P.get("body", zero)
    offs = {}
    for n in R["draw"]:
        rot, df, du = 0.0, 0.0, 0.0
        if n == "body":
            df, du = b[1], b[2]
        elif n == "head":
            h = P.get("head", zero); df, du = h[1], h[2]
        elif n.startswith("arm"):
            key = "arms"
            if action == "walk":
                key = "arm_a" if n == arms[0] else "arm_b"
            a = P.get(key, zero)
            sgn = 1.0 if piv[n][0] <= CELL_W // 2 else -1.0      # left-side arm: clockwise raises it
            if len(arms) == 1:
                sgn = 1.0 if fwx < 0 else -1.0
            rot, df, du = a[0] * sgn, a[1], a[2]
        elif n.startswith("foot"):
            key = "foot_a" if n == feet[0] else "foot_b"
            ft = P.get(key, zero); df, du = ft[1], ft[2]
        elif n == "tail":
            t = P.get("tail", zero)
            rot = t[0] * (-tail_side(view))
            du = t[2]
        dx, dy = df * fwx, df * fwy - du
        if action == "walk" and n.startswith("foot") and abs(fwx) < 0.5:
            dx, dy = 0.0, -du                                # front / back: steps are lifts only
        offs[n] = (dx, dy, rot)
        l4 = rot4(up4(cells[n]), rot, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.5
    ang = 0.0
    if abs(lean) > 0.3:
        ang = lean * side if abs(fwx) > 0.3 else lean * 0.5 * (1 if fwy < 0 else -1)
    gx, gy = gf * fwx, gf * fwy - gu
    canvas = shift4(canvas, gx, gy)
    out = canvas[1::UP, 1::UP].copy()
    out = shear_lean(out, ang)            # whole-body lean as a row shear about the feet: no rotation jaggies
    out = drop_crumbs(out)
    return out, fx, offs, (gx, gy), ang, mouth


# ---------------------------------------------------------------- effects (water)
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def disc(a, cx, cy, r, col):
    ri = int(math.ceil(r)) + 1
    for yy in range(-ri, ri + 1):
        for xx in range(-ri, ri + 1):
            if math.hypot(xx, yy) <= r + 0.25:
                put(a, cx + xx, cy + yy, col)


def drop(a, cx, cy, C, big=False):
    """A small water drop: dark rim below, water body, bright core."""
    put(a, cx, cy + 1, C["rim"])
    put(a, cx, cy, WATER)
    put(a, cx, cy - 1, WATER_HI)
    if big:
        put(a, cx + 1, cy, WATER)
        put(a, cx + 1, cy + 1, C["rim"])


def draw_fx(a, fx, mouth_pt, fwd, centre, C, front_hit):
    fwx, fwy = fwd
    L = math.hypot(fwx, fwy) or 1
    ux, uy = fwx / L, fwy / L
    px, py = -uy, ux
    toward = fwy > 0.5 and abs(fwx) < 0.5          # facing the camera
    away = fwy < -0.5 and abs(fwx) < 0.5           # facing away: the jet rises over the head
    sc = 0.6 if (toward or away) else 1.0
    for kind, k in fx:
        if kind == "charge" and mouth_pt:
            # water gathers in front of the mouth: drops spiral in
            for j in range(3):
                ang = j * 2.1 + k * 0.9
                rr = 8 - 2 * k
                drop(a, mouth_pt[0] + math.cos(ang) * rr, mouth_pt[1] + math.sin(ang) * rr * 0.7, C)
        elif kind == "jet" and mouth_pt:
            t = k - 4
            if toward:
                r = [1, 2, 3, 3, 3, 3, 2, 2][t]
                cx, cy = mouth_pt[0], mouth_pt[1] + 1
                for fr, col in ((1.0, C["rim"]), (0.75, WATER), (0.35, WATER_HI)):
                    disc(a, cx, cy, r * fr, col)
                for j in range(4 + t):          # spray streaks fanning out toward the viewer
                    ang = j * (2 * math.pi / (4 + t)) + t * 0.4
                    d = r + 3 + 2 * t
                    drop(a, cx + math.cos(ang) * d * 1.3, cy + math.sin(ang) * d * 0.7, C)
            else:
                head = [5, 11, 17, 22, 26, 26, 26, 26][t] * sc
                tail = [0, 0, 0, 0, 3, 9, 15, 20][t] * sc
                pts = []
                d = tail
                while d <= head:
                    w = math.sin(d * 0.45 - k * 1.1) * 0.5 * min(1.0, d / 8)
                    r = 1.6 + d * 0.06
                    pts.append((mouth_pt[0] + ux * d + px * w, mouth_pt[1] + uy * d + py * w, r))
                    d += 1.0
                for fr, col in ((1.0, C["rim"]), (0.7, WATER), (0.3, WATER_HI)):
                    for x, y, r in pts:
                        disc(a, x, y, r * fr, col)
                if pts and t >= 1:
                    x, y, _ = pts[-1]
                    for j in range(3):
                        drop(a, x + ux * (2 + j) + px * (j - 1) * 3, y + uy * (2 + j) + py * (j - 1) * 3, C)
        elif kind == "splash" and mouth_pt:
            if toward:
                bx, by, base = mouth_pt[0], mouth_pt[1] + 2, 8
            else:
                d = 26 * sc
                bx, by, base = mouth_pt[0] + ux * d, mouth_pt[1] + uy * d, 3
            rr = base + {8: 0, 9: 2, 10: 4, 11: 6}[k]
            for j in range(10):
                ang = j * math.pi / 5 + k * 0.35
                x, y = bx + math.cos(ang) * rr, by + math.sin(ang) * rr * 0.8
                drop(a, x, y, C, big=(k <= 9))
                if k <= 10:
                    put(a, bx + math.cos(ang) * (rr - 2), by + math.sin(ang) * (rr - 2) * 0.8, WATER_HI)
        elif kind == "hit" and front_hit:
            hx, hy = front_hit
            rr = 2 + k
            for j in range(6):
                ang = j * math.pi / 3 + 0.3
                put(a, hx + math.cos(ang) * rr, hy + math.sin(ang) * rr, FLASH if k < 3 else WATER_HI)
                put(a, hx + math.cos(ang) * (rr - 1), hy + math.sin(ang) * (rr - 1), WATER_HI)
        elif kind == "drops":
            for j in range(3):
                ang = -math.pi / 2 + (j - 1) * 0.9 + 0.2 * k
                rr = 10 + (k - 3) * 2
                drop(a, centre[0] + math.cos(ang) * rr, centre[1] - 6 + math.sin(ang) * rr * 0.8 + (k - 3) * 0.8, C)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = FLASH
    return out


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    for n in ("out/keys_2x2.png", "out/sw_single.png"):
        h.update((ART / n).read_bytes())
    h.update((ROOT / "tools/mouth_open.py").read_bytes())
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
        st["scale"] = make_stills(raw)
        run_tool(["build", "--input", str(raw), "--colors", "24", "--reserve", "#%02x%02x%02x" % WATER_HI,
                  "#%02x%02x%02x" % WATER, "#%02x%02x%02x" % FLASH, "--balance", "--name", KEY,
                  "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1",
                  "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", st["scale"], flush=True)
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}
    pal = load_palette(lock_dir)
    C = {"rim": nearest(pal, (41, 115, 131)), "ink": nearest(pal, (16, 16, 16)), "tongue": nearest(pal, (131, 41, 0))}

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
                a, fx, offs, (gx, gy), ang, mouth = compose(v, cells, piv, act, f)
                hdx, hdy, _ = offs.get("head", (0, 0, 0))
                if R["mouth"]:
                    line = [(x + ox, y + oy) for x, y in R["mouth"]]
                    line = mouth_open.transform_points(line, dx=hdx, dy=hdy)
                    line = mouth_open.transform_points(line, dx=round(gx), dy=round(gy))
                    line = shear_points(line, ang)
                    if mouth:
                        a = mouth_open.open_mouth(a, line, amount=mouth, deep=C["ink"], inner=C["tongue"], peak=R["peak"])
                    lx, ly = line[-1] if R["fwd"][0] != 0 else ((line[0][0] + line[-1][0]) / 2, line[0][1])
                    mouth_pt = (lx + R["fwd"][0] * 2, ly + 1 + R["fwd"][1] * 3)
                else:
                    hx = CELL_W / 2 + hdx + gx + R["fwd"][0] * 8
                    mouth_pt = (hx, FOOT_Y - 36 + hdy + gy + R["fwd"][1] * 6)
                centre = (CELL_W / 2 + gx, FOOT_Y - 22 + gy)
                front_hit = (CELL_W / 2 + gx + R["fwd"][0] * 10, FOOT_Y - 20 + gy + (6 if R["fwd"][1] > 0 else -6) * abs(R["fwd"][1]))
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, mouth_pt, R["fwd"], centre, C, front_hit)
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
        for si, fc in enumerate((0, 4, 8, 12) if act != "attack" else (2, 5, 8, 10)):
            src = sheets[act][:, fc * CELL_W:(fc + 1) * CELL_W]
            mm = src[..., 3] > 0
            pv[:, (ai * 4 + si) * CELL_W:(ai * 4 + si + 1) * CELL_W][mm] = src[mm]
    Image.fromarray(pv, "RGBA").save(OUT_DIR / "preview.png")

    idle0 = [sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W] for di in range(8)]
    Image.fromarray(np.concatenate(idle0, axis=1), "RGBA").save(FORM_DIR / "8dir_stills.png")

    # QC
    qa = work / "qa"; qa.mkdir(exist_ok=True)
    cols = set(pal)
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
    still_diff = 0
    loop_diff = 0
    for v in RIG:
        cells, piv, (ox, oy) = rig[v]
        ref = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        ref[oy:oy + still[v].shape[0], ox:ox + still[v].shape[1]] = still[v]
        for act in ACTIONS:
            f0 = np.array(Image.open(frames_dir / v / act / "00.png").convert("RGBA"))
            d = int((np.abs(f0.astype(int) - ref.astype(int)).sum(axis=2) > 0).sum())
            if act == "idle":
                still_diff += d
            else:
                loop_diff += d
    s_h = int(np.ptp(np.nonzero(idle0[0][..., 3].any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off),
              "alpha_not_binary": alpha_bad, "edge_touching_frames": clip, "still_diff_px": still_diff,
              "other_actions_f0_diff_px": loop_diff, "S_height": s_h, "official_height": official_height()}
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
    meta = {"generator": "forge_gen1_0007_squirtle_actions", "version": VERSION, "key": KEY, "dex": 7,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["front.png", "back.png"],
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(bounds[3]), "frames": N, "frame_time": AD,
            "directions": DIRS, "direction_sources": DIR_SRC, "actions": ACTIONS, "palette_colors": len(cols),
            "off_palette_colors": len(off), "idle_bounds_S_to_SW": bounds}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else (v.get("policy") if "policy" in v else v)) for k, v in report.items()}, ensure_ascii=False))

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
