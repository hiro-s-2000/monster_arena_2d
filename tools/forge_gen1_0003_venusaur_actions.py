#!/usr/bin/env python3
"""Venusaur (0003_venusaur): 8-direction stills + 8-direction x 4-action sheet (gen1 batch, md/gen1-forge.md).

Same flow as tools/forge_raichu_mega_x_actions.py, rig / colours / attack built from THIS creature's own art:
  art/ref_front.png, art/ref_back.png : the official front.png / back.png, x8 nearest (references for the CLI)
  art/agy_prompt.md                   : morphological prompt (no proper nouns)
  art/out/keys_2x2.png                : Antigravity CLI (agy): top-left S, top-right W, bottom-left NE, bottom-right N
  art/out/sw_single.png               : Antigravity CLI: SW (3/4 front, faces down-left)
  The other three rows are mirrors: SE = SW, E = W, NW = NE.

Pipeline:
  1. key the white backdrop (border-connected only; white claws stay), split the grid on its empty gutters,
     area-downscale: S's body height = opaque height of the official front.png (69 px), W / NE / N share S's scale,
     SW is matched to the same height.
  2. agent-sprite-forge palette_tool build (26 colours + 2 reserved sunlight colours) -> lock -> apply.
  3. Part rig per view (polygons in sprite px): flower (petals + crown + trunk), fronds (the four leaves), head,
     body, front legs, hind legs. Heavy quadruped with a flower on its back.
  4. Poses at 4x (nearest rotation), sampled back to 1x. Hidden seams get cushions from the part's own colours.
     idle  : slow heavy breathing, the flower and fronds sway and "breathe" a little.
     walk  : diagonal quadruped gait (front-left with hind-right), body sways, flower bobs behind the motion.
     attack: solar charge: crouch, the flower tilts back and gathers light at the crown -> heavy lunge forward with
             a stomp, the flower swings forward and fires a sunlight beam + leaf blades from the fronds' direction
             -> impact burst -> recover.
     hurt  : white flash, recoil back, flower whips back (max ~20 deg), return.
  5. QC: off-palette, edge-touching frames, still diff (idle frame 0 == still), gait_loop.py on idle / walk.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0003_venusaur_actions.py [--force] [--no-sync] [--stills-only]
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
KEY = "0003_venusaur"
REL = "monsterSprite/gen1/0003_venusaur/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0003_venusaur_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
TARGET_H = None                    # filled from the official front.png
CELL_W, CELL_H, FOOT_Y = 168, 104, 94
UP = 4
SUN_HI, SUN_LO = (255, 252, 214), (255, 222, 60)
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}
VIEWS = ["S", "W", "NE", "N", "SW"]




# ---------------------------------------------------------------- 1. stills
def key_white(rgb):
    a = rgb.astype(int)
    white = a.min(axis=2) > 225
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    # enclosed white gaps between petals / between the trunk and the petals (large, bounded by outline)
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    holes = [i + 1 for i, s in enumerate(sizes) if s >= 900 and (i + 1) not in border]
    bg |= np.isin(lab, holes)
    bg |= ndi.binary_dilation(bg, iterations=2) & (a.min(axis=2) > 205)
    out = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    lab2, n2 = ndi.label(out[..., 3] > 0)
    if n2:
        sz = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        big = sz.max()
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 0.02 * big])] = 0
    return out


def official_height():
    a = np.array(Image.open(FORM_DIR / "front.png").convert("RGBA"))
    ys = np.nonzero((a[..., 3] > 0).any(axis=1))[0]
    return int(ys.max() - ys.min() + 1)


def make_stills(dst: Path, target_h: int):
    k = np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB"))
    xs = 688
    src = {"S": k[:392, :xs], "W": k[:387, xs:], "NE": k[392:, :xs], "N": k[387:, xs:],
           "SW": np.array(Image.open(ART / "out/sw_single.png").convert("RGB"))}
    keyed = {v: key_white(r) for v, r in src.items()}
    crops = {}
    for v, a in keyed.items():
        im = Image.fromarray(a, "RGBA")
        crops[v] = im.crop(im.getbbox())
    dst.mkdir(parents=True, exist_ok=True)

    def shrink(im, sc):
        sm = im.resize((max(1, round(im.width * sc)), max(1, round(im.height * sc))), Image.BOX)
        a = np.array(sm)
        a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
        a[a[..., 3] == 0] = 0
        return a

    def height(a):
        ys = np.nonzero(a[..., 3].any(axis=1))[0]
        return ys.max() - ys.min() + 1

    def fit(im, h_target):
        sc = h_target / im.height
        for _ in range(8):
            a = shrink(im, sc)
            hh = height(a)
            if hh == h_target:
                return a, sc
            sc *= h_target / hh
        return a, sc

    s_arr, grid_sc = fit(crops["S"], target_h)
    out = {"S": s_arr}
    for v in ("W", "NE", "N"):
        out[v] = shrink(crops[v], grid_sc)
    out["SW"], _ = fit(crops["SW"], target_h)
    for v, a in out.items():
        im = Image.fromarray(a, "RGBA")
        im = im.crop(im.getbbox())
        im.save(dst / f"{v}.png")
    return grid_sc


def run_tool(args):
    r = subprocess.run([str(FPY), "-I", str(PALETTE_TOOL)] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"palette_tool {args[0]} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def stage_stills(work, force=False):
    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    if force or not (stills / "S.png").exists():
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        make_stills(raw, official_height())
        run_tool(["build", "--input", str(raw), "--colors", "28", "--reserve", "#%02x%02x%02x" % SUN_HI, "#%02x%02x%02x" % SUN_LO,
                  "--balance", "--name", "venusaur", "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
    return stills, lock_dir




# ---------------------------------------------------------------- 3. rig (sprite px of each downscaled still)
# Colour classes split flower / fronds / skin; polygons split the skin into head, legs and body.
# label: skin polygons in label order (later wins). draw: back to front. fwd: screen "forward".
# gait groups: A = leg_fl + leg_hr, B = leg_fr + leg_hl (diagonal quadruped gait).
RIG = {
    "S": dict(
        poly={"leg_hr": [(78, 38), (88, 38), (88, 64), (78, 64)],
              "leg_fl": [(0, 40), (21, 40), (23, 70), (0, 70)],
              "leg_fr": [(58, 47), (80, 47), (80, 70), (58, 70)],
              "head": [(17, 35), (62, 35), (62, 50), (57, 63), (22, 63), (18, 52)]},
        label=["leg_hr", "leg_fl", "leg_fr", "head"],
        draw=["leg_hr", "body", "leg_fl", "leg_fr", "head", "frond_l", "frond_r", "flower"],
        root=(44, 27), crown=(44, 4), body_x=44, fwd=(0.0, 1.0)),
    "SW": dict(
        poly={"leg_hr": [(66, 40), (78, 40), (78, 65), (68, 65)],
              "head": [(9, 35), (49, 35), (53, 48), (49, 61), (18, 61), (10, 54)],
              "leg_fl": [(0, 44), (9, 44), (10, 56), (19, 59), (19, 70), (0, 70)],
              "leg_fr": [(51, 55), (70, 55), (70, 70), (51, 70)]},
        label=["leg_hr", "head", "leg_fl", "leg_fr"],
        draw=["leg_hr", "body", "leg_fr", "leg_fl", "head", "frond_l", "frond_r", "flower"],
        root=(40, 27), crown=(40, 5), body_x=40, fwd=(-0.9, 0.45)),
    "W": dict(
        poly={"leg_fl": [(28, 51), (46, 51), (46, 68), (26, 68)],
              "leg_hl": [(62, 49), (86, 49), (86, 68), (62, 68)],
              "head": [(0, 29), (32, 29), (33, 50), (29, 57), (0, 57)]},
        label=["leg_fl", "leg_hl", "head"],
        draw=["body", "leg_hl", "leg_fl", "head", "frond_l", "frond_r", "flower"],
        root=(49, 25), crown=(49, 4), body_x=47, fwd=(-1.0, 0.0)),
    "NE": dict(
        poly={"leg_hl": [(2, 50), (28, 50), (28, 68), (2, 68)],
              "leg_hr": [(43, 55), (65, 55), (65, 68), (43, 68)],
              "leg_fr": [(68, 44), (86, 44), (86, 65), (68, 65)],
              "head": [(64, 18), (86, 18), (86, 42), (66, 42)]},
        label=["leg_hl", "leg_hr", "leg_fr", "head"],
        draw=["head", "body", "leg_fr", "leg_hl", "leg_hr", "frond_l", "frond_r", "flower"],
        root=(42, 28), crown=(42, 6), body_x=42, fwd=(0.9, -0.45)),
    "N": dict(
        poly={"leg_hl": [(4, 54), (33, 54), (33, 69), (4, 69)],
              "leg_hr": [(55, 54), (86, 54), (86, 69), (55, 69)]},
        label=["leg_hl", "leg_hr"],
        draw=["body", "leg_hl", "leg_hr", "frond_l", "frond_r", "flower"],
        root=(46, 28), crown=(46, 6), body_x=44, fwd=(0.0, -1.0)),
}
GROUP_A = ("leg_fl", "leg_hr")
FRONT_LEGS = ("leg_fl", "leg_fr")


def poly_mask(p, shape):
    m = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(m).polygon(p, fill=255)
    return np.array(m) > 0


def colour_classes(still):
    op = still[..., 3] > 0
    rgb = still[..., :3].astype(float) / 255.0
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    v = mx
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, (60 * (g - b) / d) % 360, np.where(mx == g, 60 * (b - r) / d + 120, 60 * (r - g) / d + 240))
    petal = op & ((h < 62) | (h > 340)) & (s > 0.25) & (v > 0.25)
    leaf = op & (h > 95) & (h < 140) & (v > 0.4)
    skin = op & (h > 145) & (h < 182) & (v > 0.36)
    lab, n = ndi.label(petal)
    if n:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        petal = np.isin(lab, [i + 1 for i, z in enumerate(sz) if z >= 30])
    cls = np.zeros(op.shape, int)
    cls[petal] = 1
    cls[leaf & ~petal] = 2
    cls[skin & ~petal & ~leaf] = 3
    un = op & (cls == 0)
    if un.any():
        _, (iy, ix) = ndi.distance_transform_edt(cls == 0, return_indices=True)
        cls[un] = cls[iy[un], ix[un]]
    return cls


def build_rig(view, still):
    R = RIG[view]
    op = still[..., 3] > 0
    cls = colour_classes(still)
    masks = {"flower": cls == 1}
    xx = np.mgrid[0:op.shape[0], 0:op.shape[1]][1]
    masks["frond_l"] = (cls == 2) & (xx < R["root"][0])
    masks["frond_r"] = (cls == 2) & (xx >= R["root"][0])
    skin = cls == 3
    lab = np.zeros(op.shape, int)          # 0 = body
    names = R["label"]
    for i, n in enumerate(names, start=1):
        lab[poly_mask(R["poly"][n], op.shape) & skin] = i
    masks["body"] = skin & (lab == 0)
    for i, n in enumerate(names, start=1):
        masks[n] = skin & (lab == i)
    layers = {}
    order = R["draw"]
    for i, n in enumerate(order):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in order[i + 1:]:
            above |= masks[n2]
        lower = m.copy()
        for n2 in order[:i]:
            lower |= masks[n2]
        if n == "body":
            lower = op.copy()          # the body underlays everything inside the silhouette (legs / head move over it)
        allowed = ndi.binary_fill_holes(ndi.binary_closing(np.pad(lower, 6), iterations=4))[6:-6, 6:-6]
        cushion = 7 if n in ("body", "frond_l", "frond_r", "head") else 3
        seam = above & ~m & allowed & (ndi.distance_transform_edt(~m) <= cushion)
        if seam.any() and m.any():
            src_m = m | (masks["head"] if n == "body" and "head" in masks else np.zeros_like(m))
            rr, gg, bb = [still[..., c].astype(int) for c in range(3)]
            fillable = src_m & (np.maximum(np.maximum(rr, gg), bb) > 90)
            if n != "flower":
                fillable &= gg > rr + 30          # teal skin / green leaf only (no eye, ear or claw colours)
            if not fillable.any():
                fillable = src_m
            _, (iy, ix) = ndi.distance_transform_edt(~fillable, return_indices=True)
            sy, sx = np.nonzero(seam)
            lyr[sy, sx] = still[iy[sy, sx], ix[sy, sx]]
            lyr[sy, sx, 3] = 255
        layers[n] = lyr
    return layers, masks


def place_layer(layer, view, h_still):
    h, w = layer.shape[:2]
    cell = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    ox = CELL_W // 2 - RIG[view]["body_x"]
    oy = FOOT_Y - (h_still - 1)
    y0, x0 = max(0, -oy), max(0, -ox)
    y1, x1 = min(h, CELL_H - oy), min(w, CELL_W - ox)
    cell[y0 + oy:y1 + oy, x0 + ox:x1 + ox] = layer[y0:y1, x0:x1]
    return cell, (ox, oy)


# ---------------------------------------------------------------- 4. posing
def wave(amp, cycles, f, phase=0.0):
    """0 at frame 0 and at the loop end."""
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def pose(action, f):
    """Per part: [rot (+ = toward 'forward' for the flower, + = droop for fronds), fwd px, up px].
    g = [forward px, up px, lean deg (+ forward / down, - rear up)]."""
    t = f / N
    ph = 2 * math.pi * t
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        br = wave(1.0, 1, f, -math.pi / 2) / 2.0          # 0 -> 1 -> 0 (slow breath)
        P["body"] = [0, 0, round(br)]
        P["head"] = [0, 0, round(wave(0.5, 1, f, -math.pi / 2 - 0.5))]
        P["flower"] = [wave(1.6, 1, f, 0.0), 0, round(br)]
        P["fronds"] = [wave(1.4, 1, f, -math.pi / 2) , 0, round(br)]
        if f in (6, 7, 8, 9):
            fx.append(("glint", f))
    elif action == "walk":
        up = 1.0 if abs(math.sin(ph)) > 0.5 else 0.0
        P["body"] = [0, 0, up]
        P["head"] = [0, 0.6 * math.sin(2 * ph), up]
        P["A"] = [0, -2.5 * math.cos(ph), 2.0 * max(0.0, math.sin(ph))]
        P["B"] = [0, 2.5 * math.cos(ph), 2.0 * max(0.0, -math.sin(ph))]
        P["flower"] = [2.5 * math.sin(2 * ph - 0.9), 0, up]
        P["fronds"] = [2.0 * math.sin(2 * ph - 1.3), 0, up]
    elif action == "attack":
        lean = [0, -2, -4, -6, -7, 3, 4, 3, 2, 2, 1, 1, 0, 0, 0, 0][f]
        g_f = [0, -1, -2, -2, -2, 4, 6, 6, 6, 5, 4, 3, 2, 1, 0, 0][f]
        g_u = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["front"] = [0, [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0][f], [0, 2, 4, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["head"] = [0, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0][f], [0, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["flower"] = [[0, -3, -6, -8, -9, 6, 9, 9, 8, 6, 4, 3, 2, 1, 0, 0][f], 0, [0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["fronds"] = [[0, -2, -4, -5, -5, 4, 5, 5, 4, 3, 2, 1, 1, 0, 0, 0][f], 0, 0]
        if 1 <= f <= 4:
            fx.append(("charge", f))
        if 5 <= f <= 9:
            fx.append(("beam", f))
        if 5 <= f <= 8:
            fx.append(("quake", f))
        if 6 <= f <= 9:
            fx.append(("leaves", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0
        P["g"] = [-5.0 * env + jit, 1.0 * env, -5.0 * env]
        P["head"] = [0, -1.0 * env, 1.0 * env]
        P["flower"] = [-9.0 * env, 0, 0]
        P["fronds"] = [-4.0 * env, 0, 0]
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 7:
            fx.append(("fall", f))
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


def rot_point(p, ang, piv):
    a = math.radians(ang)
    x, y = p[0] - piv[0], p[1] - piv[1]
    return (piv[0] + x * math.cos(a) - y * math.sin(a), piv[1] + x * math.sin(a) + y * math.cos(a))


def compose(view, cells, piv, action, f):
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    zero = [0.0, 0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0
    frontish = abs(fwx) < 0.5
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    crown = piv["crown"]
    b = P.get("body", zero)
    for n in R["draw"]:
        rot, df, du = 0.0, 0.0, 0.0
        if n == "body":
            df, du = b[1], b[2]
        elif n == "head":
            h = P.get("head", zero); df, du = h[1], h[2]
        elif n.startswith("leg"):
            if action == "walk":
                ft = P["A"] if n in GROUP_A else P["B"]
            elif action == "attack" and n in FRONT_LEGS:
                ft = P["front"]
            else:
                ft = zero
            df, du = ft[1], ft[2]
        elif n == "flower":
            fl = P.get("flower", zero)
            amp = 0.45 if frontish else 1.0
            rot = fl[0] * amp * side            # + tilts the flower's top toward "forward"
            du = fl[2]
        elif n.startswith("frond"):
            fr = P.get("fronds", zero)
            rot = fr[0] * (-1.0 if n == "frond_l" else 1.0)   # + = both tips droop
            du = fr[2]
        dx, dy = df * fwx, df * fwy - du
        if frontish and n.startswith("leg"):
            dx = 0.0
            dy = -du + (df * fwy * 0.3 if action == "walk" else 0.0)
        p = piv["root"] if (n == "flower" or n.startswith("frond")) else (0, 0)
        l4 = rot4(up4(cells[n]), rot, p)
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
        if n == "flower":
            crown = rot_point(piv["crown"], rot, piv["root"])
            crown = (crown[0] + dx, crown[1] + dy)
    gf, gu, lean = P["g"]
    if frontish:
        gf *= 0.4
    if abs(lean) > 0.3 and not frontish:
        lp = (CELL_W // 2 - fwx * 22, FOOT_Y)       # rear up / slam about the hind feet
        canvas = rot4(canvas, lean * side, lp)
        crown = rot_point(crown, lean * side, lp)
    gx, gy = gf * fwx, gf * fwy - gu
    canvas = shift4(canvas, gx, gy)
    out = canvas[1::UP, 1::UP].copy()
    return out, fx, (crown[0] + gx, crown[1] + gy), (gx, gy)


# ---------------------------------------------------------------- effects (reserved sunlight colours + own palette)
LEAF_HI, LEAF_LO, DUST = (122, 209, 110), (58, 116, 63), (163, 138, 98)


def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 2 <= x < a.shape[1] - 2 and 2 <= y < a.shape[0] - 2:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def sparkle(a, x, y, big):
    put(a, x, y, SUN_HI)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        put(a, x + dx, y + dy, SUN_LO)
    if big:
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            put(a, x + dx, y + dy, SUN_LO)


def leaf_blade(a, x, y, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    for i in range(-2, 3):
        put(a, x + ca * i, y + sa * i, LEAF_HI)
    put(a, x - sa, y + ca, LEAF_LO)
    put(a, x + sa * 0 + ca * 2, y + sa * 2 + 0, LEAF_LO)


def draw_fx(a, fx, crown, fwd, centre, view):
    fwx, fwy = fwd
    frontish = abs(fwx) < 0.5
    cx, cy = crown
    for kind, k in fx:
        if kind == "glint":
            sparkle(a, cx + (2 if k < 8 else -3), cy - 1 - (k % 2), k in (7, 8))
        elif kind == "charge":
            r = 16 - 3 * k
            for j in range(5):
                ang = j * 2 * math.pi / 5 + k * 0.5
                put(a, cx + math.cos(ang) * r, cy - 2 + math.sin(ang) * r * 0.7, SUN_HI if j % 2 else SUN_LO)
            sparkle(a, cx, cy - 1, k >= 3)
        elif kind == "beam":
            L = {5: 10, 6: 30, 7: 60, 8: 60, 9: 40}[k]
            wdt = {5: 1, 6: 2, 7: 3, 8: 2, 9: 1}[k]
            if frontish:
                dx, dy = 0.0, -1.0                   # front / back views: the beam goes up into the sky
                L = min(L, cy - 4)
            else:
                dx, dy = (1.0 if fwx > 0 else -1.0), (0.04 if abs(fwy) < 0.1 else fwy * 0.3)
            px, py = -dy, dx
            nrm = math.hypot(px, py)
            px, py = px / nrm, py / nrm
            for i in range(int(L) + 1):
                bx, by = cx + dx * i, cy - 1 + dy * i
                for o in range(-wdt, wdt + 1):
                    put(a, bx + px * o, by + py * o, SUN_HI if abs(o) < wdt or wdt == 1 and o == 0 else SUN_LO)
            sparkle(a, cx, cy - 1, True)
            if k in (7, 8):
                ex, ey = cx + dx * L, cy - 1 + dy * L
                for j in range(8):
                    ang = j * math.pi / 4 + k
                    rr = 3 + (k - 6) * 2
                    put(a, ex + math.cos(ang) * rr, ey + math.sin(ang) * rr, SUN_LO)
        elif kind == "quake":
            rx = 10 + (k - 5) * 9
            ry = 2 + (k - 5)
            ox = centre[0] + fwx * 14
            oy = FOOT_Y + fwy * 2
            for j in range(48):
                t = j * 2 * math.pi / 48
                y = oy + math.sin(t) * ry
                if y > CELL_H - 3:
                    continue
                if (j + k) % 3:
                    put(a, ox + math.cos(t) * rx, y, DUST)
        elif kind == "leaves":
            for j in range(4):
                spread = (j - 1.5) * 0.35
                if frontish:
                    ang = math.pi + spread if j % 2 else spread
                    d = 26 + (k - 6) * 8
                    x = centre[0] + math.cos(ang) * d
                    y = centre[1] - 14 + (j - 1.5) * 4 + (k - 6) * 2
                else:
                    ang = math.atan2(fwy * 0.5, fwx) + spread * 0.5
                    d = 30 + (k - 6) * 10 + j * 3
                    x = centre[0] + math.cos(ang) * d
                    y = centre[1] - 12 + math.sin(ang) * d + (j - 1.5) * 3
                leaf_blade(a, x, y, ang + k * 0.8)
        elif kind == "fall":
            for j in range(3):
                x = centre[0] + (j - 1) * 18 + (k % 2)
                y = centre[1] - 28 + (k - 2) * 4 + j * 3
                leaf_blade(a, x, y, k * 0.9 + j)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = SUN_HI
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

    target_h = official_height()
    stills_dir, lock_dir = stage_stills(work, force="stills" not in st["done"])
    if "stills" not in st["done"]:
        st["done"].append("stills"); save_st()
    still = {v: np.array(Image.open(stills_dir / f"{v}.png").convert("RGBA")) for v in VIEWS}

    rig = {}
    for v in VIEWS:
        layers, masks = build_rig(v, still[v])
        cells = {}
        for n, lyr in layers.items():
            cells[n], (ox, oy) = place_layer(lyr, v, still[v].shape[0])
        R = RIG[v]
        piv = {"root": (R["root"][0] + ox, R["root"][1] + oy), "crown": (R["crown"][0] + ox, R["crown"][1] + oy)}
        placed, _ = place_layer(still[v], v, still[v].shape[0])
        rig[v] = (cells, piv, placed)

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in VIEWS:
        cells, piv, _ = rig[v]
        R = RIG[v]
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, crown, (gx, gy) = compose(v, cells, piv, act, f)
                centre = (CELL_W / 2 + gx, FOOT_Y - 20 + gy)
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, crown, R["fwd"], centre, v)
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
        for si, fc in enumerate((0, 4, 8, 12)):
            src = sheets[act][:, fc * CELL_W:(fc + 1) * CELL_W]
            mm = src[..., 3] > 0
            pv[:, (ai * 4 + si) * CELL_W:(ai * 4 + si + 1) * CELL_W][mm] = src[mm]
    Image.fromarray(pv, "RGBA").save(OUT_DIR / "preview.png")

    idle0 = [sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W] for di in range(8)]
    Image.fromarray(np.concatenate(idle0, axis=1), "RGBA").save(FORM_DIR / "8dir_stills.png")

    # QC
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
    for v in VIEWS:
        a0 = np.array(Image.open(frames_dir / v / "idle" / "00.png").convert("RGBA"))
        aN = np.array(Image.open(frames_dir / v / "idle" / f"{N - 1:02d}.png").convert("RGBA"))
        still_diff[v] = int((a0 != rig[v][2]).any(axis=2).sum())
    s_h = int(np.ptp(np.nonzero((idle0[0][..., 3] > 0).any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off),
              "alpha_not_binary": alpha_bad, "edge_touching_frames": clip, "still_diff": still_diff,
              "S_height": s_h, "official_front_height": target_h}
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
    meta = {"generator": "forge_gen1_0003_venusaur_actions", "version": VERSION, "key": KEY, "dex": 3,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["front.png", "back.png"],
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": s_h, "frames": N, "frame_time": AD,
            "directions": DIRS, "direction_sources": DIR_SRC, "actions": ACTIONS, "palette_colors": len(cols),
            "off_palette_colors": len(off), "idle_bounds_S_to_SW": bounds}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) or k == "still_diff" else len(v)) for k, v in report.items()}, ensure_ascii=False, default=str))

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
