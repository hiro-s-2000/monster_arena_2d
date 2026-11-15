#!/usr/bin/env python3
"""0071_victreebel-mega: 8-direction x 4-action sheet, same flow as forge_raichu_mega_x_actions.py.

Source (2026-11-04, md/gen1-forge.md):
  art/ref_front.png / art/ref_back.png = official front.png / back.png x8 nearest.
  The Antigravity CLI (`agy -p`, prompt art/agy_prompt.md, no proper nouns) drew the key views:
    art/out/keys_2x2.png  : top-left S, top-right side view, bottom-left NE, bottom-right N
    art/out/sw_single.png : SW (3/4 front, faces down-left)
  SE = SW mirrored, NW = NE mirrored, the other side view = the drawn one mirrored.

Creature (counted on the official sprite): legless pitcher plant. ONE huge yellow pear-shaped pitcher with green
spots resting on the ground, a short neck wrapped in brown vine coils, ONE small yellow bell head with a pink lid,
ONE big umbrella crown of leaves with orange flame edges, 2 small flame-tipped side leaves on the pitcher,
2 long thin brown vines. No legs, no arms.

Rig: "pitcher" (everything below the neck line, incl. side leaves and the vines below it) and "top"
(head + lid + leaf crown + the neck coils above the line), pivot = neck centre. Small pieces cut off above the
neck line (a vine-tip leaf beside the pitcher) stay with the pitcher. The pitcher layer carries a cushion under the
neck (nearest own colour) so no hole opens when the top tilts.

Pipeline: key white (border-connected) -> area downscale (S height = official front.png opaque height; grid views
share that scale, SW matched on height) -> palette_tool build / lock / apply -> outer light edge -> outline colour
-> rig -> poses (top rotated about the neck at 4x nearest, then a row shear = the pitcher rocks on its round
bottom; pixels only moved) -> QC -> sheets -> sync agent-sprite-forge/ to pokemon_arena_2d.

Actions (16 frames each):
  idle   : the pitcher rocks slowly, the leaf crown sways and bobs on the neck. Stays on the ground.
  walk   : no legs -> two heavy hops per loop, leaning forward on take-off, the crown lags behind.
  attack : the crown rears back (wind-up) -> the pitcher hops forward and the head + crown whip forward/down
           (bell-mouth bite / crown slam) -> razor leaves from its own crown fly forward + impact star -> recover.
  hurt   : white flash, rocks back and slides back, the crown snaps back, a few of its leaves flutter off.

Never touches the official front/back png/gif.
run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0071_victreebel_mega_actions.py [--force] [--no-sync]
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
KEY = "0071_victreebel-mega"
REL = "monsterSprite/gen1/0071_victreebel/mega/victreebel-mega"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0071_victreebel_mega_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 200, 132, 122
UP = 4
FLASH, GLOW = (255, 255, 255), (255, 250, 190)

# the side view drawn top-right in keys_2x2 (checked by eye): "E" (faces right) or "W" (faces left)
SIDE = "W"
OTHER = "W" if SIDE == "E" else "E"
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), SIDE: (SIDE, False), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), OTHER: (SIDE, True), "SW": ("SW", False)}

# neck = y of the cut between top and pitcher, as a fraction of the view's bbox height (measured on the stills)
RIG = {
    "S": dict(neck=0.50, fwd=(0.0, 1.0), back=False),
    "SW": dict(neck=0.50, fwd=(-0.9, 0.45), back=False),
    SIDE: dict(neck=0.50, fwd=((1.0, 0.0) if SIDE == "E" else (-1.0, 0.0)), back=False),
    "NE": dict(neck=0.50, fwd=(0.9, -0.45), back=True),
    "N": dict(neck=0.50, fwd=(0.0, -1.0), back=True),
}


# ---------------------------------------------------------------- 1. stills
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


def official_height():
    a = np.array(Image.open(FORM_DIR / "front.png").convert("RGBA"))
    ys = np.nonzero((a[..., 3] > 0).any(axis=1))[0]
    return int(ys.max() - ys.min() + 1)


def make_stills(dst: Path, target_h):
    k = np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], SIDE: k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": np.array(Image.open(ART / "out/sw_single.png").convert("RGB"))}
    crops = {}
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb), "RGBA")
        crops[v] = im.crop(im.getbbox())
    grid = target_h / crops["S"].height          # S height = official front height
    scale = {v: grid for v in src}
    scale["SW"] = target_h / crops["SW"].height
    dst.mkdir(parents=True, exist_ok=True)
    for v, im in crops.items():
        sm = im.resize((round(im.width * scale[v]), round(im.height * scale[v])), Image.BOX)
        a = np.array(sm)
        a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
        a[a[..., 3] == 0] = 0
        lab, n = ndi.label(a[..., 3] > 0)
        if n > 1:
            sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            keep = [i + 1 for i, s in enumerate(sz) if s >= max(12, 0.02 * sz.max())]
            a[~np.isin(lab, keep)] = 0
        bb = Image.fromarray(a, "RGBA").getbbox()
        Image.fromarray(a, "RGBA").crop(bb).save(dst / f"{v}.png")


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
    op = a[..., 3] > 0
    edge = op & ~ndi.binary_erosion(op, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]], border_value=0)
    L = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    a[edge & (L > 75), :3] = ink
    return a


# ---------------------------------------------------------------- 2. rig
def build_rig(view, still):
    R = RIG[view]
    op = still[..., 3] > 0
    h, w = op.shape
    ny = int(round(R["neck"] * (h - 1)))
    yy = np.mgrid[0:h, 0:w][0]
    top = op & (yy < ny)
    lab, n = ndi.label(top)
    if n > 1:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        main = int(np.argmax(sz)) + 1
        top = lab == main
    pit = op & ~top
    # neck centre: run of the neck row nearest the crown's centre of mass
    cx_top = float(np.nonzero(top)[1].mean())
    row = op[ny]
    xs = np.nonzero(row)[0]
    if len(xs):
        runs = np.split(xs, np.nonzero(np.diff(xs) > 1)[0] + 1)
        best = min(runs, key=lambda r: abs(r.mean() - cx_top))
        px = float(best.mean())
    else:
        px = cx_top
    pivot = (px, float(ny))
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    layers = {}
    # pitcher: own pixels + a cushion under the lowest rows of the top (near the neck)
    lyr = np.zeros_like(still)
    lyr[pit] = still[pit]
    cushion = top & (ndi.distance_transform_edt(~pit) <= 5) & (yy >= ny - 8)
    fillable = pit & (np.maximum(np.maximum(r, g), b) > 70)
    if not fillable.any():
        fillable = pit
    _, (iy, ix) = ndi.distance_transform_edt(~fillable, return_indices=True)
    sy, sx = np.nonzero(cushion)
    lyr[sy, sx] = still[iy[sy, sx], ix[sy, sx]]
    lyr[sy, sx, 3] = 255
    layers["pitcher"] = lyr
    lt = np.zeros_like(still)
    lt[top] = still[top]
    layers["top"] = lt
    return layers, {"pitcher": pit, "top": top}, ["pitcher", "top"], pivot


def place_layer(layer):
    h, w = layer.shape[:2]
    cell = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    ox = CELL_W // 2 - w // 2
    oy = FOOT_Y - (h - 1)
    cell[oy:oy + h, ox:ox + w] = layer
    return cell, (ox, oy)


# ---------------------------------------------------------------- 3. poses
def wave(f, amp, cycles=1.0, phase=0.0):
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def pose(action, f):
    """g = [fwd px, up px]; lean = px the top of the sprite moves forward (row shear about the ground);
    rot = top tilt in degrees (+ = forward / down toward the target); sway = sideways tilt (front / back views);
    bob = top up px."""
    P = {"g": [0, 0], "lean": 0.0, "rot": 0.0, "sway": 0.0, "bob": 0}
    fx = []
    if action == "idle":
        P["lean"] = wave(f, 1.2, 1, 0.0)
        P["rot"] = wave(f, 3.0, 1, -0.9)
        P["sway"] = wave(f, 3.0, 1, -0.9)
        P["bob"] = round(wave(f, 1.0, 2, 0.0))
    elif action == "walk":
        t = (f % 8) / 8.0
        hop = math.sin(math.pi * t)
        P["g"] = [0, round(5.0 * hop)]
        P["lean"] = 2.6 * math.sin(2 * math.pi * t) if t > 0 else 0.0
        P["rot"] = -5.0 * math.sin(2 * math.pi * t - 0.5) - (-5.0 * math.sin(-0.5)) if t > 0 else 0.0
        P["sway"] = 3.0 * math.sin(math.pi * f / 8.0) if f % 8 else 0.0
        P["bob"] = -round(1.0 * hop) if 0.3 < t < 0.7 else 0
    elif action == "attack":
        P["g"] = [[0, -1, -2, -2, -2, 0, 4, 8, 10, 10, 9, 7, 5, 3, 1, 0][f],
                  [0, 0, 0, 0, 0, 3, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["lean"] = [0, -1, -2, -3, -3, -1, 2, 4, 5, 4, 3, 2, 1, 1, 0, 0][f]
        P["rot"] = [0, -6, -12, -16, -17, -6, 10, 22, 26, 24, 18, 12, 7, 3, 1, 0][f]
        P["bob"] = [0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        if 2 <= f <= 4:
            fx.append(("charge", f))
        if 7 <= f <= 11:
            fx.append(("leaves", f))
        if f in (8, 9, 10):
            fx.append(("impact", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 5 else 0
        P["g"] = [round(-5.0 * env) + jit, round(1.0 * env)]
        P["lean"] = -4.0 * env
        P["rot"] = -16.0 * env
        P["sway"] = -8.0 * env
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 11:
            fx.append(("flutter", f))
    return P, fx


def shift(img, dx, dy):
    dx, dy = int(round(dx)), int(round(dy))
    if not dx and not dy:
        return img
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = img[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def up4(a):
    return np.repeat(np.repeat(a, UP, 0), UP, 1)


def rot4(img4, ang, piv):
    """img4 at 4x; rotate (+ = clockwise on screen) about piv (1x coords)."""
    if abs(ang) < 0.3:
        return img4
    im = Image.fromarray(img4, "RGBA").rotate(-ang, resample=Image.NEAREST,
                                              center=(piv[0] * UP + UP / 2, piv[1] * UP + UP / 2))
    return np.array(im)


def shear(img, lean_x, top_y):
    """Rock the pitcher on its round bottom: row y moves lean_x * (FOOT_Y - y) / (FOOT_Y - top_y) px."""
    if abs(lean_x) < 0.25:
        return img
    up = up4(img)
    out = np.zeros_like(up)
    H = max(1.0, float(FOOT_Y - top_y))
    for y4 in range(up.shape[0]):
        y = y4 / UP
        d = int(round(lean_x * (FOOT_Y + 0.5 - y) / H * UP))
        out[y4] = shift(up[y4:y4 + 1], d, 0)[0]
    return out[1::UP, 1::UP].copy()


def compose(view, cells, pivot, action, f, top_y):
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    sideways = abs(fwx) > 0.3
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    m = cells["pitcher"][..., 3] > 0
    canvas[up4(m)] = up4(cells["pitcher"])[up4(m)]
    if sideways:
        ang = P["rot"] * (1.0 if fwx > 0 else -1.0) * (1.0 if abs(fwy) < 0.3 else 0.8)
        tdy = -P["bob"]
    else:
        # toward / away from the camera: a forward tilt reads as a dip (toward camera) / rise; sideways sway rotates
        ang = P["sway"] * 0.8
        dip = max(0.0, P["rot"]) * 0.18
        tdy = -P["bob"] + (dip if fwy > 0 else -dip * 0.6) + min(0.0, P["rot"]) * 0.08
    t4 = rot4(up4(cells["top"]), ang, pivot)
    t4 = shift(t4, 0, int(round(tdy * UP)))
    mt = t4[..., 3] > 0
    canvas[mt] = t4[mt]
    canvas = canvas[1::UP, 1::UP].copy()
    gf, gu = P["g"]
    lean = P["lean"]
    if sideways:
        sx = lean * (1.0 if fwx > 0 else -1.0)
    else:
        sx = lean * 0.5
        gf *= 0.5
    canvas = shear(canvas, sx, top_y)
    gx, gy = round(gf * fwx), round(gf * fwy - gu)
    canvas = shift(canvas, gx, gy)
    return canvas, fx, ang, (gx, gy + tdy)


# ---------------------------------------------------------------- effects
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def star(a, x, y, r, c1, c2):
    for t in range(-r, r + 1):
        put(a, x + t, y, c1 if abs(t) < r else c2)
        put(a, x, y + t, c1 if abs(t) < r else c2)
    if r >= 3:
        for t in (-1, 1):
            put(a, x + t, y + t, c2)
            put(a, x + t, y - t, c2)


def leaf(a, x, y, d, cols):
    """A tiny 5x3 leaf of the crown (own palette: light green, mid green, orange tip, outline)."""
    light, mid, tip, ink = cols
    dx, dy = d
    pts = {(0, 0): light, (1, 0): light, (-1, 0): mid, (2, 0): tip, (-2, 0): ink,
           (0, -1): ink, (1, -1): ink, (-1, -1): ink, (0, 1): ink, (1, 1): ink, (-1, 1): ink, (3, 0): ink}
    for (u, v), c in pts.items():
        if abs(dy) > abs(dx):
            u, v = v, u * (1 if dy > 0 else -1)
        else:
            u = u * (1 if dx >= 0 else -1)
        put(a, x + u, y + v, c)


def draw_fx(a, fx, fwd, head, centre, bbox, lcols):
    fwx, fwy = fwd
    for kind, k in fx:
        if kind == "charge":
            r = k - 1
            for j in range(3):
                ang = j * 2.1 + k * 0.7
                star(a, head[0] + math.cos(ang) * (14 + r), head[1] - 6 + math.sin(ang) * 6, 1, FLASH, GLOW)
        elif kind == "leaves":
            u = k - 7
            for j in range(3):
                off = (j - 1) * 6
                d0 = 8 + u * 7 + (j % 2) * 3
                if abs(fwx) > 0.3:
                    x = head[0] + fwx * d0
                    y = head[1] + off + fwy * d0
                    leaf(a, x, y, (fwx, 0), lcols)
                else:
                    x = head[0] + off * 1.6
                    y = head[1] + (8 + u * 3) * (1 if fwy > 0 else -1) - j % 2 * 3
                    leaf(a, x, y, (1 if off >= 0 else -1, 0), lcols)
        elif kind == "impact":
            r = {8: 3, 9: 5, 10: 3}[k]
            if abs(fwx) > 0.3:
                hx, hy = (bbox[2] + 3 if fwx > 0 else bbox[0] - 3), head[1] + 4
            else:
                hx, hy = centre[0], (bbox[3] - 8 if fwy > 0 else bbox[1] + 8)
            star(a, hx, hy, r, FLASH, GLOW)
        elif kind == "flutter":
            for j in range(2):
                x = centre[0] + (-14 if j == 0 else 15) + round(2 * math.sin(k * 1.3 + j))
                y = bbox[1] + 4 + (k - 2) * 2 + j * 3
                leaf(a, x, y, (1 if (k + j) % 2 else -1, 0), lcols)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = FLASH
    return out


def nearest(pal, c):
    return min(pal, key=lambda p: sum((p[i] - c[i]) ** 2 for i in range(3)))


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    for n in ("out/keys_2x2.png", "out/sw_single.png"):
        h.update((ART / n).read_bytes())
    h.update((FORM_DIR / "front.png").read_bytes())
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
    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    if "stills" not in st["done"]:
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        make_stills(raw, target_h)
        run_tool(["build", "--input", str(raw), "--colors", "30", "--reserve", "#%02x%02x%02x" % FLASH, "#%02x%02x%02x" % GLOW,
                  "--balance", "--name", "pitcher-plant", "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
        pal = palette_colors(lock_dir / "palette.json")
        ink = min(pal, key=lum)
        for v in RIG:
            p = stills / f"{v}.png"
            a = crisp_outline(np.array(Image.open(p).convert("RGBA")), ink)
            Image.fromarray(a, "RGBA").save(p)
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    pal = palette_colors(lock_dir / "palette.json")
    ink = min(pal, key=lum)
    lcols = (nearest(pal, (175, 230, 33)), nearest(pal, (80, 119, 21)), nearest(pal, (252, 164, 31)), ink)
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}

    rig = {}
    for v in RIG:
        layers, masks, order, pv = build_rig(v, still[v])
        cells = {}
        for n, lyr in layers.items():
            cells[n], (ox, oy) = place_layer(lyr)
        rig[v] = (cells, (pv[0] + ox, pv[1] + oy), (ox, oy), masks)

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        cells, pivot, (ox, oy), masks = rig[v]
        R = RIG[v]
        allm = (cells["pitcher"][..., 3] > 0) | (cells["top"][..., 3] > 0)
        ys, xs = np.nonzero(allm)
        top_y = int(ys.min())
        bbox0 = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        ty, tx = np.nonzero(cells["top"][..., 3] > 0)
        head0 = (float(tx.mean()), float(pivot[1] - 6))
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, ang, (gx, gy) = compose(v, cells, pivot, act, f, top_y)
                # the head (bell mouth) follows the top's tilt about the neck
                hx = head0[0] + gx + R["fwd"][0] * 6 + math.sin(math.radians(ang)) * 8
                hy = head0[1] + gy
                centre = (CELL_W / 2 + gx, FOOT_Y - 30 + gy)
                fy_, fx_ = np.nonzero(a[..., 3] > 0)
                bbox = (int(fx_.min()), int(fy_.min()), int(fx_.max()), int(fy_.max())) if len(fy_) else bbox0
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, R["fwd"], (hx, hy), centre, bbox, lcols)
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
    for v in RIG:
        ref, _ = place_layer(still[v])
        for act in ACTIONS:
            a0 = np.array(Image.open(frames_dir / v / act / "00.png").convert("RGBA"))
            still_diff += int((a0 != ref).any(axis=2).sum())
    S_h = int(np.ptp(np.nonzero((still["S"][..., 3] > 0).any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off),
              "semi_alpha_px": alpha_bad, "edge_touching_frames": clip, "still_diff_px": still_diff,
              "S_height": S_h, "official_front_height": target_h}
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
    meta = {"generator": "forge_gen1_0071_victreebel_mega_actions", "version": VERSION, "key": KEY, "dex": 71,
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
