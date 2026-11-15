#!/usr/bin/env python3
"""0081_magnemite (normal): 8-direction x 4-action sheet, same flow as forge_raichu_mega_x_actions.py.

Source (2026-11-04, md/gen1-forge.md):
  art/ref_front.png / art/ref_back.png = official front.png / back.png x8 nearest.
  The Antigravity CLI (`agy -p`, prompt art/agy_prompt.md, no proper nouns) draws the key views:
    art/out/keys_2x2.png  : top-left S, top-right side view (SIDE), bottom-left NE, bottom-right N
    art/out/sw_single.png : SW (3/4 front, faces down-left)
  SE = SW mirrored, NW = NE mirrored, the other side view = the drawn one mirrored.

Creature (counted on the official sprite): ONE floating pale steel-green metal ball, ONE big round eye,
THREE screws (one big flat-headed screw sticking up from the top, two small cross screws on the lower front),
TWO dark-grey horseshoe magnets (left / right), each with one red and one blue tip. No legs, arms, mouth.

Pipeline: key white (border-connected) -> area downscale (S body height = official front.png opaque height;
all grid views share that scale, SW matched on height) -> palette_tool build / lock / apply -> outer light edge
pixels -> darkest outline colour -> part rig BY COLOUR (ball = steel green + everything inside its circle,
magnets = dark-grey / red / blue pieces sticking out of the ball (grown into the ball where a magnet lies in
front of it), top screw = what sticks out above the ball) -> hidden seams refilled from the ball's own colours
(outline ring kept) -> poses at 4x nearest (magnets rotate about their root on the ball) -> 1x -> QC -> sheets ->
sync agent-sprite-forge/ to pokemon_arena_2d.

Actions (16 frames each; floats: never below its rest height):
  idle   : hover bob, the magnets sway slowly (alternating), the top screw wobbles, a tiny spark between the prongs.
  walk   : no legs -> glides: deeper bob, leans into the direction of travel (side views), magnets swing.
  attack : THUNDER SHOCK with its own magnets: magnets swing up / forward while sparks gather on the red / blue
           tips -> short lunge -> zigzag bolts from both magnet tips to a point ahead -> electric burst -> recover.
  hurt   : white flash, tilts back (<= ~20 deg) and drifts back, static crackles around the ball, recovers.

Never touches the official front/back png/gif.
run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0081_magnemite_actions.py [--force] [--no-sync]
     [--art DIR --out DIR]  (smoke test on stand-in art: writes only to --out, never syncs)
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
KEY = "0081_magnemite"
REL = "monsterSprite/gen1/0081_magnemite/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_ROOT = FORM_DIR                       # 8dir_stills.png + agent-sprite-forge/ go here
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0081_magnemite_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 104, 76, 64
UP = 4
SPARK_HI, SPARK_LO, FLASH = (255, 252, 214), (255, 214, 40), (255, 255, 255)

# the side view drawn top-right in keys_2x2 (CHECK BY EYE on the real art): "E" (faces right) or "W" (faces left)
SIDE = "W"
OTHER = "W" if SIDE == "E" else "E"
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), SIDE: (SIDE, False), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), OTHER: (SIDE, True), "SW": ("SW", False)}
VIEWS = ["S", "SW", SIDE, "NE", "N"]
FWD = {"S": (0.0, 1.0), "SW": (-0.9, 0.45), "E": (1.0, 0.0), "W": (-1.0, 0.0), "NE": (0.9, -0.45), "N": (0.0, -1.0)}
BACK = {"NE", "N"}


# ---------------------------------------------------------------- 1. stills
def key_white(rgb):
    a = rgb.astype(int)
    white = a.min(axis=2) > 225
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    # the inside of a horseshoe magnet can be a closed white pocket (between the prongs) -> clear mid-size pockets
    # that are not round (the eye white is round and sits inside the green ball)
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    green = (a[..., 1] - a[..., 0] > 15) & (a.min(axis=2) <= 225)
    near_green = ndi.binary_dilation(green, iterations=6)
    for i, s in enumerate(sizes):
        li = i + 1
        if li in border or s < 60:
            continue
        m = lab == li
        if not (m & near_green).any():           # white pocket not touching the ball = gap between magnet prongs
            bg |= m
    bg |= ndi.binary_dilation(bg, iterations=2) & (a.min(axis=2) > 200)
    out = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    lab2, n2 = ndi.label(out[..., 3] > 0)
    if n2:
        sz = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 200])] = 0
    return out


def official_height():
    a = np.array(Image.open(FORM_DIR / "front.png").convert("RGBA"))
    ys = np.nonzero((a[..., 3] > 0).any(axis=1))[0]
    return int(ys.max() - ys.min() + 1)


def make_stills(art: Path, dst: Path, target_h):
    k = np.array(Image.open(art / "out/keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], SIDE: k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": np.array(Image.open(art / "out/sw_single.png").convert("RGB"))}
    crops = {}
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb), "RGBA")
        crops[v] = im.crop(im.getbbox())
    grid = target_h / crops["S"].height          # S body height = official front height
    scale = {v: grid for v in src}
    scale["SW"] = target_h / crops["SW"].height
    dst.mkdir(parents=True, exist_ok=True)
    for v, im in crops.items():
        sm = im.resize((max(1, round(im.width * scale[v])), max(1, round(im.height * scale[v]))), Image.BOX)
        a = np.array(sm)
        a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
        a[a[..., 3] == 0] = 0
        lab, n = ndi.label(a[..., 3] > 0, structure=np.ones((3, 3)))
        if n > 1:                                   # keep pieces >= 4 px (a magnet tip may detach by 1 px)
            sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            a[~np.isin(lab, [i + 1 for i, s in enumerate(sz) if s >= 4])] = 0
        Image.fromarray(a, "RGBA").save(dst / f"{v}.png")


def run_tool(args):
    r = subprocess.run([str(FPY), "-I", str(PALETTE_TOOL)] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"palette_tool {args[0]} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def palette_colors(pj_path):
    pj = json.loads(Path(pj_path).read_text())
    cols = []
    for c in pj.get("colors", []):
        hx = (c["hex"] if isinstance(c, dict) else c).lstrip("#")
        cols.append((int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)))
    return cols


def lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def crisp_outline(a, ink):
    """Light pixels on the outer silhouette edge -> outline colour (the official sprites have a dark outline)."""
    op = a[..., 3] > 0
    edge = op & ~ndi.binary_erosion(op, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]], border_value=0)
    L = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    m = edge & (L > 75)
    a[m, :3] = ink
    return a


# ---------------------------------------------------------------- 2. rig (by colour)
def classes(still):
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    op = still[..., 3] > 0
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    ball = op & (g - r > 12) & (mx > 70)                          # steel green + mint highlights + green shade
    red = op & (r > 150) & (r - g > 80) & (r - b > 60)
    blue = op & (b > 110) & (b - r > 50)
    grey = op & (mx >= 34) & (mx < 125) & (mx - mn < 34) & ~ball   # magnet metal / screw sockets
    ink = op & (mx < 34)
    silver = op & (mn >= 125) & (mx - mn < 30) & ~ball            # screw heads, eye white
    return op, ball, red, blue, grey, ink, silver


def build_rig(view, still, ink_col):
    op, ball, red, blue, grey, ink, silver = classes(still)
    h, w = op.shape
    # ball circle: the steel-green body plus everything it encloses (eye, cross screws)
    bf = ndi.binary_fill_holes(ndi.binary_closing(np.pad(ball, 4), iterations=2))[4:-4, 4:-4]
    lab, n = ndi.label(bf)
    if n > 1:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        bf = lab == (int(np.argmax(sz)) + 1)
    ys, xs = np.nonzero(bf)
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    rad = (xs.max() - xs.min() + ys.max() - ys.min() + 2) / 4.0 + 0.5
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - cx, yy - cy)
    inside = dist <= rad + 0.6
    ball_reg = op & (inside | ndi.binary_dilation(bf, iterations=1))
    outside = op & ~ball_reg

    # pieces sticking out of the ball
    lab, n = ndi.label(outside, structure=np.ones((3, 3)))
    metal = grey | red | blue
    mags, screw = [], np.zeros_like(op)
    pieces = []
    for i in range(1, n + 1):
        m = lab == i
        pieces.append((int(m.sum()), m))
    pieces.sort(key=lambda t: -t[0])
    for sz, m in pieces:
        my, mxx = np.nonzero(m)
        has_tip = (m & (red | blue)).any()
        above = my.mean() < cy - rad * 0.55 and abs(mxx.mean() - cx) < rad * 0.75
        if above and not has_tip:
            screw |= m
        elif has_tip and len(mags) < 2:
            mags.append(m)
        elif len(mags) < 2 and sz >= 6 and not above:
            mags.append(m)
        elif above:
            screw |= m
        # other specks stay with the ball
    # grow each magnet into the ball through magnet-coloured pixels (a magnet lying in front of the ball)
    grown = []
    for m in mags:
        cand = m | (metal & ball_reg & (dist > rad * 0.35))
        cl, _ = ndi.label(cand, structure=np.ones((3, 3)))
        keep = set(np.unique(cl[m])) - {0}
        g2 = np.isin(cl, list(keep))
        g2 |= ndi.binary_dilation(g2, iterations=1) & ink & ball_reg & ~bf   # its own outline over the ball
        grown.append(g2)
    # keep magnets disjoint
    if len(grown) == 2:
        grown[1] &= ~grown[0]
    # top screw: grow down through its dark shaft into the ball's top edge (not deeper than 2 px)
    if screw.any():
        shaft = (grey | silver | ink) & ball_reg & (dist > rad - 2.5) & (yy < cy)
        cl, _ = ndi.label(screw | shaft, structure=np.ones((3, 3)))
        keep = set(np.unique(cl[screw])) - {0}
        screw = np.isin(cl, list(keep)) & ~bf
        for g2 in grown:
            screw &= ~g2

    names = []
    masks = {}
    front = {}
    for i, g2 in enumerate(grown):
        nm = f"mag{i}"
        masks[nm] = g2
        frac = float((g2 & (dist < rad - 1.0)).sum()) / max(1, int(g2.sum()))
        front[nm] = (frac > 0.18) and view not in BACK
        names.append(nm)
    masks["screw"] = screw
    used = np.zeros_like(op)
    for m in masks.values():
        used |= m
    masks["ball"] = op & ~used

    # ball layer + refill of its own hidden seams (pixels inside the circle that belong to another part)
    lyr = np.zeros_like(still)
    lyr[masks["ball"]] = still[masks["ball"]]
    hole = used & inside
    if hole.any():
        r, g, b = [still[..., c].astype(int) for c in range(3)]
        fillable = masks["ball"] & ball & (np.maximum(np.maximum(r, g), b) > 70)
        _, (iy, ix) = ndi.distance_transform_edt(~fillable, return_indices=True)
        sy, sx = np.nonzero(hole)
        lyr[sy, sx] = still[iy[sy, sx], ix[sy, sx]]
        lyr[sy, sx, 3] = 255
        rim = hole & (dist >= rad - 0.9)
        lyr[rim, :3] = ink_col
        lyr[rim, 3] = 255
    layers = {"ball": lyr}
    for nm in names + ["screw"]:
        l2 = np.zeros_like(still)
        l2[masks[nm]] = still[masks[nm]]
        layers[nm] = l2
    # a magnet's root inside the ball circle lies over the ball: drawn again after the ball (same transform)
    for nm in names:
        m_in = masks[nm] & inside
        l2 = np.zeros_like(still)
        l2[m_in] = still[m_in]
        layers[nm + ":in"] = l2

    # pivots: magnet root = magnet pixel closest to the ball centre; tips = red / blue centroids
    info = {"centre": (cx, cy), "rad": rad, "pivot": {}, "tips": {}, "side": {}}
    for nm in names:
        my, mxx = np.nonzero(masks[nm])
        k = int(np.argmin(np.hypot(mxx - cx, my - cy)))
        info["pivot"][nm] = (float(mxx[k]), float(my[k]))
        tips = []
        for col in (red, blue):
            t = masks[nm] & col
            if t.any():
                ty, tx = np.nonzero(t)
                tips.append((float(tx.mean()), float(ty.mean())))
        if not tips:
            k2 = int(np.argmax(np.hypot(mxx - cx, my - cy)))
            tips.append((float(mxx[k2]), float(my[k2])))
        info["tips"][nm] = tips
        info["side"][nm] = -1.0 if mxx.mean() < cx else 1.0
    back_m = [nm for nm in names if not front[nm]]
    front_m = [nm for nm in names if front[nm]]
    order = back_m + ["ball", "screw"] + [nm + ":in" for nm in back_m] + front_m
    return layers, masks, order, info


def place_layer(layer, shape_ref):
    h, w = shape_ref.shape[:2]
    cell = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    ox = CELL_W // 2 - w // 2
    oy = FOOT_Y - (h - 1)
    cell[oy:oy + h, ox:ox + w] = layer
    return cell, (ox, oy)


# ---------------------------------------------------------------- 3. poses
def wave(f, amp, cycles=1.0, phase=0.0):
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def hump(f, amp, cycles=1.0):
    """0 at f = 0 and at the loop end, never negative (floats: never below the rest height)."""
    return amp * 0.5 * (1 - math.cos(2 * math.pi * cycles * f / N))


def pose(action, f):
    """g = [fwd px, up px, tilt deg (+ = leans forward)]; mag = raise deg for magnet 0 / 1 (+ = outer end up);
    screw = up px."""
    P = {"g": [0.0, 0.0, 0.0], "mag": [0.0, 0.0], "screw": 0.0}
    fx = []
    if action == "idle":
        P["g"] = [0.0, round(hump(f, 2.0, 1)), 0.0]
        P["mag"] = [wave(f, 5.0, 1, 0.0), wave(f, 5.0, 1, math.pi)]
        P["screw"] = round(hump(f, 1.0, 2))
        if f in (6, 7):
            fx.append(("prong", 0 if f == 6 else 1))
        if f in (12, 13):
            fx.append(("prong", 1 if f == 12 else 0))
    elif action == "walk":
        P["g"] = [0.0, round(hump(f, 3.0, 2)), 6.0 * math.sin(math.pi * f / N) ** 0.5 if f else 0.0]
        P["mag"] = [wave(f, 9.0, 2, 0.0), wave(f, 9.0, 2, math.pi)]
        P["screw"] = round(hump(f, 1.0, 2))
        if f % 4 == 2:
            fx.append(("trail", f))
    elif action == "attack":
        P["g"] = [[0, -1, -2, -2, -2, 2, 5, 7, 7, 6, 5, 4, 3, 2, 1, 0][f],
                  [0, 1, 2, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0][f],
                  [0, -2, -4, -5, -5, 3, 6, 6, 6, 5, 4, 3, 2, 1, 0, 0][f]]
        rz = [0, 8, 16, 22, 25, 18, 12, 12, 12, 12, 10, 7, 4, 2, 1, 0][f]
        P["mag"] = [rz, rz]
        P["screw"] = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        if 1 <= f <= 4:
            fx.append(("charge", f))
        if 5 <= f <= 9:
            fx.append(("bolt", f))
        if f in (7, 8, 9):
            fx.append(("burst", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 5 else 0
        P["g"] = [-5.0 * env + jit, round(2.0 * env), -20.0 * env]
        P["mag"] = [-14.0 * env, -14.0 * env]
        P["screw"] = round(1.0 * env)
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 8:
            fx.append(("static", f))
    return P, fx


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


def rot_pt(p, ang, piv):
    a = math.radians(ang)
    x, y = p[0] - piv[0], p[1] - piv[1]
    return (piv[0] + x * math.cos(a) - y * math.sin(a), piv[1] + x * math.sin(a) + y * math.cos(a))


def compose(view, cells, order, info, action, f):
    P, fx = pose(action, f)
    fwx, fwy = FWD[view]
    sideways = abs(fwx) > 0.3
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    tips = []
    for n in order:
        lyr = up4(cells[n])
        if n.startswith("mag"):
            inner = n.endswith(":in")
            n = n.split(":")[0]
            i = int(n[3:])
            sd = info["side"][n]
            ang = P["mag"][min(i, 1)] * (1.0 if sd < 0 else -1.0)      # + raise = outer end up
            piv = info["pivot"][n]
            lyr = rot4(lyr, ang, piv)
            if not inner:
                tips += [rot_pt(t, ang, piv) for t in info["tips"][n]]
        elif n == "screw":
            lyr = shift4(lyr, 0, -P["screw"])
        m = lyr[..., 3] > 0
        canvas[m] = lyr[m]
    gf, gu, tilt = P["g"]
    c = info["centre"]
    if not sideways:
        gf *= 0.5                                  # toward / away from the camera: a short drift
        tilt_s = tilt * 0.5 * (1.0 if action != "hurt" else 0.6)  # a slight sideways roll
    else:
        tilt_s = tilt * (1.0 if fwx > 0 else -1.0)
    canvas = rot4(canvas, tilt_s, c)
    tips = [rot_pt(t, tilt_s, c) for t in tips]
    gx, gy = gf * fwx, gf * fwy - gu
    canvas = shift4(canvas, gx, gy)
    out = canvas[1::UP, 1::UP].copy()
    tips = [(t[0] + gx, t[1] + gy) for t in tips]
    return out, fx, tips, (gx, gy)


# ---------------------------------------------------------------- effects
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def zigzag(a, x0, y0, x1, y1, seed, wob=1.6):
    rng = np.random.default_rng(seed)
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 3))
    pts = [(x0, y0)]
    for i in range(1, n):
        u = i / n
        nx, ny = -(y1 - y0), (x1 - x0)
        L = math.hypot(nx, ny) or 1
        o = rng.uniform(-wob, wob)
        pts.append((x0 + (x1 - x0) * u + nx / L * o, y0 + (y1 - y0) * u + ny / L * o))
    pts.append((x1, y1))
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        k = int(max(abs(bx - ax), abs(by - ay))) + 1
        for j in range(k + 1):
            u = j / k
            put(a, ax + (bx - ax) * u, ay + (by - ay) * u, SPARK_HI)
    for (ax, ay) in pts[1:-1]:
        put(a, ax + 1, ay, SPARK_LO)


def spark(a, x, y, r, seed):
    for j in range(3):
        ang = j * 2.1 + seed
        zigzag(a, x, y, x + math.cos(ang) * r, y + math.sin(ang) * r, seed * 7 + j, 0.8)


def draw_fx(a, fx, fwd, tips, centre, rad):
    fwx, fwy = fwd
    tx = centre[0] + fwx * (rad + 14)
    ty = centre[1] + fwy * (rad + 8)
    for kind, k in fx:
        if kind == "prong" and tips:
            t = tips[min(k * 2, len(tips) - 1)]
            spark(a, t[0], t[1], 3, 3 + k)
        elif kind == "trail":
            bx = centre[0] - fwx * (rad + 3)
            by = centre[1] - fwy * (rad * 0.5)
            for j in (-1, 1):
                put(a, bx + (-fwy) * j * 3, by + fwx * j * 3, SPARK_LO)
        elif kind == "charge":
            for i, t in enumerate(tips):
                spark(a, t[0], t[1], 1 + k, 10 + i + k)
            if k >= 3 and len(tips) >= 2:
                zigzag(a, tips[0][0], tips[0][1], tips[-1][0], tips[-1][1], 50 + k, 2.0)
        elif kind == "bolt":
            L = min(1.0, (k - 4) / 3.0)
            for i, t in enumerate(tips[:4]):
                ex, ey = t[0] + (tx - t[0]) * L, t[1] + (ty - t[1]) * L
                zigzag(a, t[0], t[1], ex, ey, 30 + i + k, 1.8)
        elif kind == "burst":
            r = {7: 4, 8: 7, 9: 5}[k]
            for j in range(8):
                ang = j * math.pi / 4 + k * 0.4
                zigzag(a, tx + math.cos(ang) * 1.5, ty + math.sin(ang) * 1.5,
                       tx + math.cos(ang) * r, ty + math.sin(ang) * r, 40 + j + k, 0.8)
        elif kind == "static":
            for j in range(3):
                ang = j * 2.2 + k * 0.9
                cx, cy = centre[0] + math.cos(ang) * (rad + 3), centre[1] + math.sin(ang) * (rad + 2)
                zigzag(a, cx, cy, cx + math.cos(ang + 1.3) * 4, cy + math.sin(ang + 1.3) * 4, 60 + j + k, 1.0)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = FLASH
    return out


# ---------------------------------------------------------------- main
def state_hash(art):
    h = hashlib.sha1(Path(__file__).read_bytes())
    for n in ("out/keys_2x2.png", "out/sw_single.png"):
        h.update((art / n).read_bytes())
    h.update((FORM_DIR / "front.png").read_bytes())
    return h.hexdigest()[:12]


def save_gif(im, path):
    pass  # official front/back are never rewritten by this script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--art", default=None, help="stand-in art folder (smoke test)")
    ap.add_argument("--out", default=None, help="output root for a smoke test (never synced)")
    args = ap.parse_args()
    art = Path(args.art) if args.art else ART
    out_root = Path(args.out) if args.out else OUT_ROOT
    out_dir = out_root / "agent-sprite-forge"
    smoke = args.out is not None

    work = out_dir / "_work"
    h = state_hash(art)
    st_p = work / "state.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {}
    if args.force or st.get("hash") != h:
        if work.exists():
            shutil.rmtree(work)
        st = {"hash": h, "done": []}
    work.mkdir(parents=True, exist_ok=True)
    save_st = lambda: st_p.write_text(json.dumps(st))

    target_h = official_height()
    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    if "stills" not in st["done"]:
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        make_stills(art, raw, target_h)
        run_tool(["build", "--input", str(raw), "--colors", "22", "--reserve", "#%02x%02x%02x" % SPARK_HI,
                  "#%02x%02x%02x" % SPARK_LO, "#%02x%02x%02x" % FLASH, "--balance", "--name", "magnet-ball",
                  "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
        pal = palette_colors(lock_dir / "palette.json")
        ink = min(pal, key=lum)
        for v in VIEWS:
            p = stills / f"{v}.png"
            a = crisp_outline(np.array(Image.open(p).convert("RGBA")), ink)
            Image.fromarray(a, "RGBA").save(p)
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    pal = palette_colors(lock_dir / "palette.json")
    ink = min(pal, key=lum)
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in VIEWS}

    rig = {}
    rig_report = {}
    for v in VIEWS:
        layers, masks, order, info = build_rig(v, still[v], ink)
        cells = {}
        for n, lyr in layers.items():
            cells[n], (ox, oy) = place_layer(lyr, still[v])
        info = dict(info)
        info["centre"] = (info["centre"][0] + ox, info["centre"][1] + oy)
        info["pivot"] = {k: (p[0] + ox, p[1] + oy) for k, p in info["pivot"].items()}
        info["tips"] = {k: [(t[0] + ox, t[1] + oy) for t in ts] for k, ts in info["tips"].items()}
        rig[v] = (cells, order, info)
        rig_report[v] = {"order": order, "px": {n: int(m.sum()) for n, m in masks.items()},
                         "rad": round(info["rad"], 1), "tips": {k: len(t) for k, t in info["tips"].items()}}
        # debug: colour-coded part map
        dbg = np.zeros(still[v].shape, np.uint8)
        colmap = {"ball": (120, 200, 160), "screw": (230, 230, 60), "mag0": (230, 80, 80), "mag1": (80, 120, 240)}
        for n, m in masks.items():
            dbg[m, :3] = colmap.get(n, (255, 0, 255)); dbg[m, 3] = 255
        (work / "rig").mkdir(exist_ok=True)
        Image.fromarray(dbg, "RGBA").resize((dbg.shape[1] * 6, dbg.shape[0] * 6), Image.NEAREST).save(work / "rig" / f"{v}.png")
    (work / "rig" / "rig.json").write_text(json.dumps(rig_report, indent=1))

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in VIEWS:
        cells, order, info = rig[v]
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, tips, (gx, gy) = compose(v, cells, order, info, act, f)
                centre = (info["centre"][0] + gx, info["centre"][1] + gy)
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, FWD[v], tips, centre, info["rad"])
                Image.fromarray(a, "RGBA").save(od / f"{f:02d}.png")
            st["done"].append(key); save_st()
        print("frames", v, flush=True)

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
    out_dir.mkdir(parents=True, exist_ok=True)
    for act in ACTIONS:
        im = Image.fromarray(sheets[act], "RGBA")
        im.save(out_dir / f"{act}.png"); im.save(out_dir / f"{act}_8dir.png")
        Image.fromarray(sheets[act][:CELL_H], "RGBA").save(out_dir / f"{act}_S_strip.png")
    master = np.concatenate([sheets[a] for a in ACTIONS], axis=0)
    Image.fromarray(master, "RGBA").save(out_dir / "sheet.png")
    Image.fromarray(master, "RGBA").save(out_dir / "all_8dir_actions.png")
    pv = np.full((8 * CELL_H, 16 * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, fc in enumerate((0, 4, 8, 12) if act != "attack" else (0, 3, 7, 11)):
            src = sheets[act][:, fc * CELL_W:(fc + 1) * CELL_W]
            mm = src[..., 3] > 0
            pv[:, (ai * 4 + si) * CELL_W:(ai * 4 + si + 1) * CELL_W][mm] = src[mm]
    Image.fromarray(pv, "RGBA").save(out_dir / "preview.png")
    idle0 = [sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W] for di in range(8)]
    Image.fromarray(np.concatenate(idle0, axis=1), "RGBA").save(out_root / "8dir_stills.png")

    # QC
    qa = work / "qa"; qa.mkdir(exist_ok=True)
    cols = set(pal)
    aa = master[..., 3] > 0
    used = {tuple(int(x) for x in c) for c in np.unique(master[aa][:, :3], axis=0)}
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
    for v in VIEWS:
        ref, _ = place_layer(still[v], still[v])
        for act in ACTIONS:
            a0 = np.array(Image.open(frames_dir / v / act / "00.png").convert("RGBA"))
            still_diff += int((a0 != ref).any(axis=2).sum())
    S_h = int(np.ptp(np.nonzero((still["S"][..., 3] > 0).any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off),
              "semi_alpha_px": alpha_bad, "edge_touching_frames": clip, "still_diff_px": still_diff,
              "S_height": S_h, "official_front_height": target_h, "rig": rig_report}
    if not args.skip_qa and GAIT_LOOP.exists():
        for act, kind in (("idle", "idle"), ("walk", "gait")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run([str(FPY), "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / "S" / act), "--fps", "13",
                                "--output-dir", str(od), "--kind", kind, "--no-aids"], capture_output=True, text=True)
            rep = od / "loop-report.json"
            rec = json.loads(rep.read_text()).get("recommendation") if rep.exists() else (r.stderr.strip().splitlines() or ["no report"])[-1]
            report[f"gait_{act}"] = rec
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    bounds = []
    for di in range(8):
        c = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W, 3] > 0
        ys, xs = np.nonzero(c)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    meta = {"generator": "forge_gen1_0081_magnemite_actions", "version": VERSION, "key": KEY, "dex": 81,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["front.png", "back.png"],
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(bounds[3]), "frames": N, "frame_time": AD,
            "directions": DIRS, "direction_sources": DIR_SRC, "actions": ACTIONS, "palette_colors": len(cols),
            "off_palette_colors": len(off), "idle_bounds_S_to_SW": bounds}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("rig",)}, ensure_ascii=False, default=str))

    if not smoke and not args.no_sync and SYNC_DIR.parent.exists():
        if SYNC_DIR.exists():
            shutil.rmtree(SYNC_DIR)
        SYNC_DIR.mkdir(parents=True)
        for fp in out_dir.iterdir():
            if fp.is_file():
                shutil.copy2(fp, SYNC_DIR / fp.name)
        print("synced ->", SYNC_DIR)


if __name__ == "__main__":
    main()
