#!/usr/bin/env python3
"""0019_rattata-alola (gen1 regional): 8-direction x 4-action sheet, Antigravity key art + agent-sprite-forge.

Batch procedure md/gen1-forge.md (2026-11-04); structure copied from tools/forge_raichu_mega_x_actions.py, rig /
colours / attack built from this form's own art (a slim dark grey rat standing on two hind legs: big head with two
pointed ears, red eyes, two white front teeth, a tan cheek ruff, tan belly, two thin arms, two long feet, ONE thin
tail ending in a curled tuft).

Source:
  art/ref_front.png, art/ref_back.png : official front.png / back.png x8 nearest (references only)
  art/agy_prompt.md                   : morphological prompt (no proper nouns) for the Antigravity CLI (`agy -p`)
  art/out/keys_2x2.png                : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
  art/out/sw_single.png               : SW. The generator drew FOUR feet here; the SW-only retry
                                        (art/agy_prompt_retry_sw.md) failed with a quota error, so fix_sw() paints out
                                        the two extra feet (the near and far foot of the official front stay).
  Mirrors: SE = SW, E = W, NW = NE.

Pipeline:
  1. key the white backdrop (border-connected only), crop, "mode" downscale (dominant colour per block, only the
     near-black outline wins as a line) + 1 px outer outline. S height = opaque height of the official front.png
     (44 px); W / NE / N share the grid scale, SW is matched on height.
  2. agent-sprite-forge palette_tool build (24 colours + reserved flash white) -> lock -> apply.
  3. part rig per view: tail (+ tuft box, pivot at the rump), body, head (ears + cheek ruff + teeth), two arms, two
     feet. Hidden seams get cushions from the nearest own colour.
  4. poses at 4x nearest, sampled back to 1x. The tail tuft sways by a row-wise shift (tip moves most).
     idle   : breathing, head bob, tail sway + tuft flick, small arm sway.
     walk   : alternating steps on the two hind feet, arm swing, tail swing.
     attack : fang bite: rear back (tail up, head back) -> lunge, claws forward, mouth opens (mouth_open.py, 2 px)
              -> two rows of white fangs open in front of the mouth and snap shut -> impact star -> recover.
     hurt   : white flash, recoil back (lean <= 8 deg), hit sparks, recover.
  5. QC: off-palette, edge touch, still diff (idle f0 == still), gait_loop.py on idle / walk.

Writes <form>/8dir_stills.png and <form>/agent-sprite-forge/ (sheet.png = 4 actions x 8 rows, preview.png,
meta.json, idle.png ...). The official front/back png/gif are never touched. Syncs agent-sprite-forge/ (without
_work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0019_rattata_alola_actions.py [--force] [--no-sync]
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

KEY = "0019_rattata-alola"
REL = "monsterSprite/gen1/0019_rattata/regional/rattata-alola"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0019_rattata_alola_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 128, 88, 78
UP = 4
FLASH = (255, 255, 255)
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each still)
# label: later polygons win. draw: back to front. body_x: placed on the cell centre. fwd: screen "forward".
# mouth: closed mouth line hinge -> lip (None = not visible). flame: box (x0, y0, x1, y1) of the tail tuft
# (the curled tip that flicks).
RIG = {
    "S": dict(
        poly={"tail": [(29, 8), (39, 8), (39, 35), (30, 35), (29, 28)],
              "body": [(5, 22), (27, 22), (28, 40), (4, 40)],
              "head": [(0, 0), (30, 0), (30, 21), (22, 24), (9, 24), (0, 21)],
              "foot_l": [(1, 38), (14, 38), (14, 44), (1, 44)],
              "foot_r": [(17, 38), (31, 38), (31, 44), (17, 44)],
              "arm_l": [(0, 25), (8, 25), (8, 38), (0, 38)],
              "arm_r": [(23, 25), (30, 25), (30, 37), (23, 37)]},
        label=["tail", "body", "head", "foot_l", "foot_r", "arm_l", "arm_r"],
        draw=["tail", "body", "foot_l", "foot_r", "head", "arm_l", "arm_r"],
        pivot={"tail": (29, 31), "head": (15, 23), "body": (15, 40), "foot_l": (8, 42), "foot_r": (24, 42),
               "arm_l": (7, 26), "arm_r": (24, 26)},
        body_x=15, fwd=(0.0, 1.0), mouth=[(13, 23), (18, 23)], peak=0.5, flame=(30, 9, 39, 20), back=False),
    "SW": dict(
        poly={"tail": [(33, 29), (41, 25), (41, 6), (54, 6), (54, 37), (34, 37)],
              "body": [(8, 20), (33, 20), (34, 40), (8, 40)],
              "head": [(1, 0), (30, 0), (30, 15), (24, 21), (13, 22), (1, 21)],
              "foot_l": [(3, 36), (14, 36), (14, 44), (3, 44)],
              "foot_r": [(17, 38), (28, 38), (28, 44), (17, 44)],
              "arm_l": [(0, 21), (13, 21), (13, 30), (0, 30)],
              "arm_r": [(18, 25), (26, 25), (26, 32), (18, 32)]},
        label=["tail", "body", "head", "foot_l", "foot_r", "arm_l", "arm_r"],
        draw=["tail", "foot_r", "body", "foot_l", "head", "arm_r", "arm_l"],
        pivot={"tail": (34, 32), "head": (16, 21), "body": (20, 40), "foot_l": (9, 41), "foot_r": (22, 41),
               "arm_l": (12, 24), "arm_r": (22, 26)},
        body_x=20, fwd=(-0.9, 0.45), mouth=[(14, 21), (5, 20)], peak=0.7, flame=(41, 7, 54, 22), back=False),
    "W": dict(
        poly={"tail": [(27, 29), (30, 25), (30, 8), (43, 8), (43, 37), (28, 37)],
              "body": [(9, 20), (28, 20), (28, 38), (9, 38)],
              "head": [(0, 0), (28, 0), (28, 17), (22, 22), (10, 23), (0, 22)],
              "foot_b": [(19, 38), (27, 38), (27, 43), (19, 43)],
              "foot_a": [(10, 38), (19, 38), (19, 43), (10, 43)],
              "arm_l": [(3, 23), (15, 23), (15, 29), (3, 29)]},
        label=["tail", "body", "head", "foot_b", "foot_a", "arm_l"],
        draw=["tail", "foot_b", "body", "foot_a", "head", "arm_l"],
        pivot={"tail": (27, 33), "head": (16, 21), "body": (18, 38), "foot_a": (14, 41), "foot_b": (22, 41),
               "arm_l": (14, 25)},
        body_x=18, fwd=(-1.0, 0.0), mouth=[(11, 21), (4, 21)], peak=0.65, flame=(31, 9, 43, 22), back=False),
    "NE": dict(
        poly={"tail": [(0, 10), (12, 10), (13, 30), (16, 33), (16, 38), (0, 38)],
              "body": [(14, 17), (34, 17), (34, 41), (14, 41)],
              "head": [(11, 0), (37, 0), (37, 20), (14, 20)],
              "foot_l": [(14, 37), (23, 37), (23, 44), (14, 44)],
              "foot_r": [(26, 38), (37, 38), (37, 44), (26, 44)],
              "arm_r": [(31, 21), (40, 21), (40, 32), (31, 32)]},
        label=["tail", "body", "head", "foot_l", "foot_r", "arm_r"],
        draw=["foot_l", "foot_r", "body", "arm_r", "head", "tail"],
        pivot={"tail": (15, 35), "head": (24, 19), "body": (24, 41), "foot_l": (18, 41), "foot_r": (31, 42),
               "arm_r": (32, 23)},
        body_x=24, fwd=(0.9, -0.45), mouth=None, peak=0.5, flame=(0, 11, 11, 24), back=True),
    "N": dict(
        poly={"tail": [(30, 12), (40, 12), (40, 40), (29, 40), (29, 32)],
              "body": [(6, 18), (28, 18), (28, 41), (6, 41)],
              "head": [(2, 0), (30, 0), (30, 20), (2, 20)],
              "foot_l": [(1, 38), (12, 38), (12, 44), (1, 44)],
              "foot_r": [(19, 38), (30, 38), (30, 44), (19, 44)],
              "arm_l": [(0, 24), (10, 24), (10, 31), (0, 31)],
              "arm_r": [(22, 24), (29, 24), (29, 31), (22, 31)]},
        label=["tail", "body", "head", "foot_l", "foot_r", "arm_l", "arm_r"],
        draw=["foot_l", "foot_r", "body", "head", "arm_l", "arm_r", "tail"],
        pivot={"tail": (29, 36), "head": (16, 19), "body": (16, 41), "foot_l": (6, 42), "foot_r": (24, 42),
               "arm_l": (9, 26), "arm_r": (23, 26)},
        body_x=16, fwd=(0.0, -1.0), mouth=None, peak=0.5, flame=(30, 12, 40, 24), back=True),
}

# ---------------------------------------------------------------- 1. stills from the Antigravity key art
# enclosed white areas at least this big are background seen through a gap (S: between the arm, the cheek ruff
# and the tail); smaller ones are eye whites and the front teeth and stay.
HOLE_MIN = {"S": 300}
HOLE_MIN_DEFAULT = 2000


def key_white(rgb, view):
    a = rgb.astype(int)
    white = a.min(axis=2) > 225
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    if n:
        sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        lim = HOLE_MIN.get(view, HOLE_MIN_DEFAULT)
        bg |= np.isin(lab, [i + 1 for i, sz in enumerate(sizes) if sz >= lim and (i + 1) not in border])
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
            dark = px[:, :3].max(axis=1) < 40      # outline only (the grey shadow #323832 is not a line)
            if dark.mean() >= 0.3:
                sel = px[dark]
            else:
                ks, cs = np.unique(qk, return_counts=True)
                sel = px[qk == ks[cs.argmax()]]
            out[i, j, :3] = sel[:, :3].mean(axis=0)
            out[i, j, 3] = 255
    return out


def outline(a, col=(13, 13, 13)):
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


def fix_sw(rgb):
    """The generator drew FOUR feet in the SW view (a second pair under the belly and under the rump), and the
    agy retry hit the quota. The two extra feet are painted out (white = keyed away); the near foot (lower left)
    and the far foot (centre) stay, as in the official front.png. The cut edges get the 1 px outline later."""
    out = rgb.copy()
    out[615:691, 448:537] = 255          # extra foot under the belly
    out[622:736, 668:816] = 255          # extra foot under the rump
    return out


def make_stills(dst: Path):
    k = np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], "W": k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": fix_sw(np.array(Image.open(ART / "out/sw_single.png").convert("RGB")))}
    crops = {}
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb, v), "RGBA")
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
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    bright = mx > 70
    # the grey fur (not the tan cheek ruff / hands, not the outline): cushions are filled from it, so a part that
    # moves never uncovers a tan band (the cheek ruff sits right at the neck / tail seams in the back views)
    fur = (mx > 45) & (mx < 150) & (mx - mn < 30)
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
            src_m = m          # fill only from the body's own fur (never copy the head's cheek ruff)
        if seam.any() and m.any():
            fillable = src_m & fur
            if not fillable.any():
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
    """part -> [rot deg (+ = raise), fwd px, up px]; 'g' = [forward px, up px, lean deg]; tuft flick amp; mouth px."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    lick = 0.0
    mouth = 0
    if action == "idle":
        br = wave(0.6, 1, f, -math.pi / 2)            # 0 .. 1.2 breathing (1 - cos)
        P["head"] = [0, 0, round(br)]
        P["arms"] = [wave(5.0, 1, f, -math.pi / 2), 0, round(br)]
        P["tail"] = [wave(4.0, 1, f, 0.0), 0, 0]
        lick = wave(1.0, 2, f, 0.0)
    elif action == "walk":
        ph = 2 * math.pi * f / N
        up = round(1.0 * abs(math.sin(ph)))
        P["body"] = [0, 0, up]
        P["head"] = [0, 0, up]
        P["foot_a"] = [0, 2.0 * math.sin(ph), 2.0 * max(0.0, math.sin(ph))]       # swing forward lifted
        P["foot_b"] = [0, -2.0 * math.sin(ph), 2.0 * max(0.0, -math.sin(ph))]
        P["arm_a"] = [wave(12.0, 1, f, 0.0), 0, up]
        P["arm_b"] = [-wave(12.0, 1, f, 0.0), 0, up]
        P["tail"] = [wave(6.0, 2, f, 0.0), 0, up * 0.5]
        lick = wave(1.5, 2, f, 0.3)
    elif action == "attack":
        # fang bite: rear back (1-3) -> lunge + bite (4-8) -> hold (9) -> recover (10-15)
        g_f = [0, -1, -2, -2, 2, 5, 7, 8, 8, 7, 5, 4, 3, 2, 1, 0][f]
        g_u = [0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -3, -5, -6, 2, 6, 9, 10, 9, 8, 6, 4, 3, 2, 1, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["head"] = [0, [0, -1, -1, -1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0][f], [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["arms"] = [[0, -8, -14, -16, 10, 26, 34, 34, 30, 24, 16, 10, 6, 3, 0, 0][f],
                     [0, -1, -1, -1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0][f], 0]
        P["foot_a"] = [0, [0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 2, 2, 1, 1, 0, 0][f], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["tail"] = [[0, 8, 16, 20, 10, -4, -10, -10, -8, -6, -4, -2, 0, 0, 0, 0][f], 0, 0]
        lick = [0, 1, 2, 2, 1, -1, -2, -2, -1, 0, 1, 1, 0, 0, 0, 0][f]
        mouth = [0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        if 4 <= f <= 7:
            fx.append(("fang", f))
        if 7 <= f <= 10:
            fx.append(("impact", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 5 else 0
        P["g"] = [-4.0 * env + jit, 1.0 * env, -8.0 * env]
        P["head"] = [0, -1.0 * env, 1.0 * env]
        P["arms"] = [18.0 * env, 0, 1.0 * env]
        P["tail"] = [-12.0 * env, 0, 0]
        lick = -1.5 * env
        if f == 1:
            fx.append(("flash", 0))
        if 1 <= f <= 4:
            fx.append(("hit", f))
        if 3 <= f <= 8:
            fx.append(("sparks", f))
    return P, fx, lick, mouth


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


def lick_flame(layer, box, amt):
    """Shift the flame rows sideways; the tip moves the most, the base not at all (no new colours)."""
    a = int(round(amt * 2)) / 2.0
    if abs(a) < 0.5:
        return layer
    x0, y0, x1, y1 = box
    out = layer.copy()
    region = layer[y0:y1 + 1, x0:x1 + 1].copy()
    out[y0:y1 + 1, x0:x1 + 1] = 0
    hgt = y1 - y0
    for r in range(region.shape[0]):
        k = 1.0 - r / max(1, hgt)             # 1 at the tip, 0 at the base
        dx = int(round(a * k * k))
        row = region[r]
        tgt = out[y0 + r]
        for c in range(row.shape[0]):
            if row[c, 3] and 0 <= x0 + c + dx < tgt.shape[0]:
                tgt[x0 + c + dx] = row[c]
        # keep the base row joined: fill gaps left of shifted pixels with the original
    # anything that was base-connected and is now empty stays empty only inside the box; pixels outside are intact
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


def compose(view, cells, piv, flame_box, action, f):
    R = RIG[view]
    P, fx, lick, mouth = pose(action, f)
    fwx, fwy = R["fwd"]
    zero = [0.0, 0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    feet = [n for n in R["draw"] if n.startswith("foot")]
    feet.sort()
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
            fb = flame_box
            tail_right = (fb[0] + fb[2]) / 2 > piv["tail"][0]
            rot = t[0] * (-1.0 if tail_right else 1.0)
            du = t[2]
        dx, dy = df * fwx, df * fwy - du
        if action == "walk" and n.startswith("foot") and abs(fwx) < 0.5:
            dx, dy = 0.0, -du                                # front / back: steps are lifts only
        offs[n] = (dx, dy, rot)
        lyr = cells[n]
        if n == "tail":
            lyr = lick_flame(lyr, flame_box, lick)
        l4 = rot4(up4(lyr), rot, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.5
    feetpt = (CELL_W // 2, FOOT_Y)
    ang = 0.0
    if abs(lean) > 0.3:
        ang = lean * side if abs(fwx) > 0.3 else lean * 0.5 * (1 if fwy < 0 else -1)
        canvas = rot4(canvas, ang, feetpt)
    gx, gy = gf * fwx, gf * fwy - gu
    canvas = shift4(canvas, gx, gy)
    out = canvas[1::UP, 1::UP].copy()
    out = drop_crumbs(out)
    return out, fx, offs, (gx, gy), ang, mouth


# ---------------------------------------------------------------- effects
BEHIND = None      # opaque mask of the creature: when set (back views), effects are drawn behind it


def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        if BEHIND is not None and BEHIND[y, x]:
            return
        a[y, x, :3] = col
        a[y, x, 3] = 255


def tooth_row(a, cx, cy, gap, down, C, n=3):
    """A curved row of n pointed white fangs (2 px wide, 4 px long) on a dark-red gum with an ink outline.
    The middle tip sits at cy -/+ gap, the outer tips 1 px further back (a jaw arc). down=True: upper jaw."""
    sgn = 1 if down else -1
    cx, cy = int(round(cx)), int(round(cy))
    pts, gum = [], []
    x_start = cx - (n * 3) // 2 + (0 if down else 1)          # lower teeth sit between the upper ones
    for t in range(n):
        x0 = x_start + t * 3
        arc = 0 if t == n // 2 else 1
        tip_y = cy - sgn * (gap + arc)
        for i in range(4):                      # i = 0 root .. 3 tip
            y = tip_y - sgn * (3 - i)
            for xx in ((x0, x0 + 1) if i < 3 else (x0 + (t % 2),)):
                pts.append((xx, y))
        for xx in (x0 - 1, x0, x0 + 1, x0 + 2):
            gum.append((xx, tip_y - sgn * 4))
    for x, y in pts + gum:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                put(a, x + dx, y + dy, C["ink"])
    for x, y in gum:
        put(a, x, y, C["dred"])
    for x, y in pts:
        put(a, x, y, C["white"])


def draw_fx(a, fx, mouth_pt, fwd, centre, C, front_hit):
    fwx, fwy = fwd
    for kind, k in fx:
        if kind == "fang" and mouth_pt:
            # two rows of fangs open in front of the mouth and snap shut on the target
            d = 7 if abs(fwx) > 0.3 else 5
            bx, by = mouth_pt[0] + fwx * d, mouth_pt[1] + fwy * d
            gap = {4: 4, 5: 3, 6: 2, 7: 0}[k]
            tooth_row(a, bx, by, gap, True, C)
            tooth_row(a, bx, by, gap, False, C)
        elif kind == "impact" and mouth_pt:
            d = 7 if abs(fwx) > 0.3 else 5
            bx, by = mouth_pt[0] + fwx * d, mouth_pt[1] + fwy * d
            rr = {7: 6, 8: 8, 9: 10, 10: 11}[k]
            for j in range(8):
                ang = j * math.pi / 4 + 0.2
                for s2 in range(2 if k < 10 else 1):
                    col = (FLASH if k == 7 else C["white"]) if s2 == 0 else C["red"]
                    put(a, bx + math.cos(ang) * (rr - s2), by + math.sin(ang) * (rr - s2), col)
                    if k <= 8:
                        put(a, bx + math.cos(ang) * (rr - 2 - s2), by + math.sin(ang) * (rr - 2 - s2), C["red"])
        elif kind == "hit" and front_hit:
            hx, hy = front_hit
            rr = 2 + k
            for j in range(6):
                ang = j * math.pi / 3 + 0.3
                put(a, hx + math.cos(ang) * rr, hy + math.sin(ang) * rr, FLASH if k < 3 else C["white"])
                put(a, hx + math.cos(ang) * (rr - 1), hy + math.sin(ang) * (rr - 1), C["red"])
        elif kind == "sparks":
            for j in range(3):
                ang = -math.pi / 2 + (j - 1) * 0.9 + 0.2 * k
                rr = 10 + (k - 3) * 2
                put(a, centre[0] + math.cos(ang) * rr, centre[1] - 6 + math.sin(ang) * rr * 0.8 - (k - 3), C["red"] if k % 2 else C["white"])


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
        run_tool(["build", "--input", str(raw), "--colors", "24", "--reserve", "#%02x%02x%02x" % FLASH, "--balance", "--name", KEY, "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1",
                  "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", st["scale"], flush=True)
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}
    pal = load_palette(lock_dir)
    C = {"white": nearest(pal, (243, 243, 243)), "red": nearest(pal, (231, 61, 61)), "dred": nearest(pal, (191, 13, 13)),
         "ink": nearest(pal, (13, 13, 13)), "tongue": nearest(pal, (191, 13, 13))}

    rig = {}
    for v in RIG:
        layers, masks = build_rig(v, still[v])
        cells, piv = {}, {}
        for n, lyr in layers.items():
            cells[n], (ox, oy) = place_layer(lyr, v)
        for n, p in RIG[v]["pivot"].items():
            piv[n] = (p[0] + ox, p[1] + oy)
        fb = RIG[v]["flame"]
        rig[v] = (cells, piv, (ox, oy), (fb[0] + ox, fb[1] + oy, fb[2] + ox, fb[3] + oy))

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        cells, piv, (ox, oy), fbox = rig[v]
        R = RIG[v]
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, offs, (gx, gy), ang, mouth = compose(v, cells, piv, fbox, act, f)
                hdx, hdy, _ = offs.get("head", (0, 0, 0))
                mouth_pt = None
                feetpt = (CELL_W // 2, FOOT_Y)
                if R["mouth"]:
                    line = [(x + ox, y + oy) for x, y in R["mouth"]]
                    line = mouth_open.transform_points(line, dx=hdx, dy=hdy)
                    line = mouth_open.transform_points(line, angle=ang, pivot=feetpt)
                    line = mouth_open.transform_points(line, dx=gx, dy=gy)
                    if mouth:
                        a = mouth_open.open_mouth(a, line, amount=mouth, deep=C["ink"], inner=C["tongue"], peak=R["peak"])
                    lx, ly = line[-1] if R["fwd"][0] != 0 else ((line[0][0] + line[-1][0]) / 2, line[0][1])
                    mouth_pt = (lx + R["fwd"][0] * 2, ly + 1 + R["fwd"][1] * 3)
                else:
                    # back views: the snout is hidden; the bite happens ahead of the head (above it for N,
                    # upper right for NE) and is drawn behind the creature
                    hx = CELL_W / 2 + hdx + gx + R["fwd"][0] * 13
                    mouth_pt = (hx, FOOT_Y - 36 + hdy + gy + R["fwd"][1] * 8)
                centre = (CELL_W / 2 + gx, FOOT_Y - 24 + gy)
                front_hit = (CELL_W / 2 + gx + R["fwd"][0] * 10, FOOT_Y - 22 + gy + (6 if R["fwd"][1] > 0 else -6) * abs(R["fwd"][1]))
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                global BEHIND
                BEHIND = (a[..., 3] > 0) if R["back"] else None
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
    for v in RIG:
        cells, piv, (ox, oy), _ = rig[v]
        ref = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        ref[oy:oy + still[v].shape[0], ox:ox + still[v].shape[1]] = still[v]
        f0 = np.array(Image.open(frames_dir / v / "idle" / "00.png").convert("RGBA"))
        still_diff += int((np.abs(f0.astype(int) - ref.astype(int)).sum(axis=2) > 0).sum())
    s_h = int(np.ptp(np.nonzero(idle0[0][..., 3].any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off),
              "alpha_not_binary": alpha_bad, "edge_touching_frames": clip, "still_diff_px": still_diff,
              "S_height": s_h, "official_height": official_height()}
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
    meta = {"generator": "forge_gen1_0019_rattata_alola_actions", "version": VERSION, "key": KEY, "dex": 19,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["front.png", "back.png"], "sw_fix": "two generator-added feet painted out (fix_sw)",
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
