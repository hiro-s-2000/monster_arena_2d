#!/usr/bin/env python3
"""0011_metapod (normal): 8-direction x 4-action sheet, same flow as forge_raichu_mega_x_actions.py.

Source (2026-11-04, md/gen1-forge.md):
  art/ref_front.png / art/ref_back.png = official front.png / back.png x8 nearest.
  The Antigravity CLI (`agy -p`, prompt art/agy_prompt.md, no proper nouns) drew the key views:
    art/out/keys_2x2.png  : top-left S, top-right side view, bottom-left NE, bottom-right N
    art/out/sw_single.png : SW (3/4 front, faces down-left)
  SE = SW mirrored, NW = NE mirrored, the other side view = the drawn one mirrored.

Creature (counted on the official sprite): ONE rigid green crescent cocoon shell, pointed top, one sleepy eye per
side, sage-green inner belly, a short segmented bottom tip that curls forward and rests on the ground.
No legs, no arms, no wings, no mouth.

Pipeline: key white (border-connected) -> area downscale (S body height = official front.png opaque height;
all grid views share that scale, SW matched on height) -> palette_tool build / lock / apply -> outer light edge
pixels -> darkest outline colour -> part rig (curled bottom tip by polygon, shell = rest; cushion under the seam)
-> poses (part shifts at 4x nearest, then a row shear = the rigid shell rocks on its round bottom; pixels are only
moved, never resampled) -> QC -> sheets -> sync agent-sprite-forge/ to pokemon_arena_2d.

Actions (16 frames each):
  idle   : the shell rocks slowly back and forth on its round bottom, the tip twitches. Stays on the ground.
  walk   : no legs -> two small hops per loop, leaning forward on take-off, tip trails, lands back.
  attack : HARDEN: the shell glints (light sweeps over the plates, sparkles) while it rocks back -> hops forward and
           slams the pointed top into the target (shell tackle) -> impact star -> rocks back upright.
  hurt   : white flash, rocks back and slides back, dizzy stars over the tip, rocks upright.

Never touches the official front/back png/gif.
run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0011_metapod_actions.py [--force] [--no-sync]
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
KEY = "0011_metapod"
REL = "monsterSprite/gen1/0011_metapod/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0011_metapod_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 96, 72, 62
UP = 4
FLASH, GLOW = (255, 255, 255), (232, 255, 176)

# the side view drawn top-right in keys_2x2 (checked by eye): "E" (faces right) or "W" (faces left)
SIDE = "E"
OTHER = "W" if SIDE == "E" else "E"
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), SIDE: (SIDE, False), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), OTHER: (SIDE, True), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (fractions of each view's bbox, 0..1)
# tip: the curled segmented bottom tip (polygon in bbox fractions). body = everything else.
RIG = {
    "S": dict(tip=[(0.0, 0.78), (1.0, 0.78), (1.0, 1.0), (0.0, 1.0)], fwd=(0.0, 1.0), back=False),
    "SW": dict(tip=[(0.0, 0.70), (0.45, 0.70), (0.55, 1.0), (0.0, 1.0)], fwd=(-0.9, 0.45), back=False),
    SIDE: dict(tip=([(0.55, 0.72), (1.0, 0.72), (1.0, 1.0), (0.55, 1.0)] if SIDE == "E" else
                    [(0.0, 0.72), (0.45, 0.72), (0.45, 1.0), (0.0, 1.0)]),
               fwd=((1.0, 0.0) if SIDE == "E" else (-1.0, 0.0)), back=False),
    "NE": dict(tip=[(0.55, 0.75), (1.0, 0.75), (1.0, 1.0), (0.55, 1.0)], fwd=(0.9, -0.45), back=True),
    "N": dict(tip=None, fwd=(0.0, -1.0), back=True),
}
OFFICIAL_H = None      # measured from front.png


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
    grid = target_h / crops["S"].height          # S body height = official front height
    scale = {v: grid for v in src}
    scale["SW"] = target_h / crops["SW"].height
    dst.mkdir(parents=True, exist_ok=True)
    for v, im in crops.items():
        sm = im.resize((round(im.width * scale[v]), round(im.height * scale[v])), Image.BOX)
        a = np.array(sm)
        a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
        a[a[..., 3] == 0] = 0
        # keep only the biggest piece (no stray specks)
        lab, n = ndi.label(a[..., 3] > 0)
        if n > 1:
            sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            a[lab != (int(np.argmax(sz)) + 1)] = 0
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


# ---------------------------------------------------------------- 2. rig
def poly_mask(polys, shape):
    m = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(m)
    for p in polys:
        d.polygon(p, fill=255)
    return np.array(m) > 0


def build_rig(view, still):
    R = RIG[view]
    op = still[..., 3] > 0
    h, w = op.shape
    tipm = np.zeros_like(op)
    if R["tip"]:
        poly = [(x * (w - 1), y * (h - 1)) for x, y in R["tip"]]
        tipm = poly_mask([poly], op.shape) & op
        # the tip is only the part hanging off the shell: keep the piece(s) touching the ground row
        lab, n = ndi.label(tipm)
        keep = set(np.unique(lab[h - 2:][tipm[h - 2:]])) - {0}
        tipm = np.isin(lab, list(keep))
    masks = {"tip": tipm, "body": op & ~tipm}
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    layers = {}
    order = ["tip", "body"] if R["back"] else ["body", "tip"]
    for i, n in enumerate(order):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in order[i + 1:]:
            above |= masks[n2]
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= 2)
        if seam.any() and m.any():
            fillable = m & (np.maximum(np.maximum(r, g), b) > 70)
            if not fillable.any():
                fillable = m
            _, (iy, ix) = ndi.distance_transform_edt(~fillable, return_indices=True)
            sy, sx = np.nonzero(seam)
            lyr[sy, sx] = still[iy[sy, sx], ix[sy, sx]]
            lyr[sy, sx, 3] = 255
        layers[n] = lyr
    return layers, masks, order


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
    """g = [fwd px, up px]; lean = px the top of the shell moves forward (row shear around the ground point);
    tip = [fwd px, up px] of the curled tip relative to the shell."""
    P = {"g": [0, 0], "lean": 0.0, "tip": [0, 0]}
    fx = []
    if action == "idle":
        P["lean"] = wave(f, 2.0, 1, 0.0)
        P["tip"] = [round(wave(f, 0.6, 2, 0.0)), 0]
    elif action == "walk":
        # two hops per loop
        t = (f % 8) / 8.0
        hop = math.sin(math.pi * t)
        P["g"] = [0, round(4.0 * hop)]
        P["lean"] = 2.4 * math.sin(2 * math.pi * t) if t > 0 else 0.0
        P["tip"] = [round(-1.0 * hop), round(-1.0 * hop) if hop > 0.5 else 0]
    elif action == "attack":
        # harden (rock back + glint) -> hop-slam forward with the pointed top -> impact -> rock back upright
        P["g"] = [[0, 0, -1, -1, -1, 0, 3, 7, 9, 9, 8, 6, 4, 2, 1, 0][f],
                  [0, 0, 0, 0, 0, 3, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0][f]]
        P["lean"] = [0, -1, -2, -3, -3, -2, 2, 6, 8, 7, 5, 3, 2, 1, 0, 0][f]
        P["tip"] = [[0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0][f],
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        if 1 <= f <= 5:
            fx.append(("harden", f))
        if f in (7, 8, 9):
            fx.append(("impact", f))
        if 6 <= f <= 7:
            fx.append(("speed", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 5 else 0
        P["g"] = [round(-4.0 * env) + jit, round(1.0 * env)]
        P["lean"] = -5.0 * env
        P["tip"] = [round(1.0 * env), 0]
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 9:
            fx.append(("stars", f))
    return P, fx


def shift(img, dx, dy):
    dx, dy = int(round(dx)), int(round(dy))
    if not dx and not dy:
        return img
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = img[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def shear(img, lean_x, top_y):
    """Rock the rigid shell on its round bottom: row y moves lean_x * (FOOT_Y - y) / (FOOT_Y - top_y) px.
    Done at 4x so the steps stay even, then nearest back to 1x (pixels are moved, never blended)."""
    if abs(lean_x) < 0.25:
        return img
    up = np.repeat(np.repeat(img, UP, 0), UP, 1)
    out = np.zeros_like(up)
    H = max(1.0, float(FOOT_Y - top_y))
    for y4 in range(up.shape[0]):
        y = y4 / UP
        d = int(round(lean_x * (FOOT_Y + 0.5 - y) / H * UP))
        out[y4] = shift(up[y4:y4 + 1], d, 0)[0]
    return out[1::UP, 1::UP].copy()


def compose(view, cells, order, action, f, top_y):
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    sideways = abs(fwx) > 0.3
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    tdx, tdy = 0, 0
    for n in order:
        lyr = cells[n]
        if n == "tip":
            tf, tu = P["tip"]
            if sideways:
                tdx, tdy = round(tf * fwx), round(-tu)
            else:
                tdx, tdy = 0, round(-abs(tu))
            lyr = shift(lyr, tdx, tdy)
        m = lyr[..., 3] > 0
        canvas[m] = lyr[m]
    gf, gu = P["g"]
    lean = P["lean"]
    if sideways:
        sx = lean * (1.0 if fwx > 0 else -1.0)
    else:
        # toward / away from the camera the shell rocks sideways a little and nods via a 1 px drop
        sx = lean * 0.5
        gf *= 0.6
    canvas = shear(canvas, sx, top_y)
    gx, gy = round(gf * fwx), round(gf * fwy - gu)
    canvas = shift(canvas, gx, gy)
    return canvas, fx, sx, (gx, gy)


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


def harden_glint(a, k, bbox):
    """A diagonal band of light sweeps over the shell plates (only on lit shell pixels, outline kept)."""
    x0, y0, x1, y1 = bbox
    op = a[..., 3] > 0
    L = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    inner = op & ndi.binary_erosion(op, iterations=1) & (L > 70)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    span = (x1 - x0) + (y1 - y0)
    pos = x0 + y0 - 4 + (k - 1) / 4.0 * (span + 8)
    s = xx + yy
    band = inner & (np.abs(s - pos) <= 1.6)
    a[band, :3] = GLOW
    core = inner & (np.abs(s - pos) <= 0.4)
    a[core, :3] = FLASH


def draw_fx(a, fx, fwd, top, centre, bbox):
    fwx, fwy = fwd
    for kind, k in fx:
        if kind == "harden":
            harden_glint(a, k, bbox)
            if k in (3, 4, 5):
                r = {3: 1, 4: 2, 5: 1}[k]
                star(a, top[0] + 3, top[1] + 4, r, FLASH, GLOW)
                star(a, bbox[0] + 1, (bbox[1] + bbox[3]) // 2 + (k - 4) * 3, 1, FLASH, GLOW)
        elif kind == "impact":
            r = {7: 3, 8: 5, 9: 3}[k]
            if abs(fwx) > 0.3:
                hx, hy = (bbox[2] + 2 if fwx > 0 else bbox[0] - 2), top[1] + 7
            else:
                hx, hy = centre[0], (top[1] + 6 if fwy < 0 else bbox[3] - 6)
            star(a, hx, hy, r, FLASH, GLOW)
        elif kind == "speed":
            if abs(fwx) > 0.3:
                bx = bbox[0] - 2 if fwx > 0 else bbox[2] + 2
                for j, yy in enumerate(range(bbox[1] + 6, bbox[3] - 4, 6)):
                    ln = 4 - (j % 2)
                    for t in range(ln):
                        put(a, bx - fwx * (t + j % 2), yy, GLOW)
        elif kind == "stars":
            for j in range(3):
                ang = j * 2.1 + k * 0.8
                cx, cy = centre[0] + math.cos(ang) * 9, top[1] - 3 + math.sin(ang) * 2.5
                star(a, cx, cy, 1, FLASH, GLOW)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = FLASH
    return out


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
        run_tool(["build", "--input", str(raw), "--colors", "20", "--reserve", "#%02x%02x%02x" % FLASH, "#%02x%02x%02x" % GLOW,
                  "--balance", "--name", "cocoon", "--output-dir", str(pal_dir)])
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
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}

    rig = {}
    for v in RIG:
        layers, masks, order = build_rig(v, still[v])
        cells = {}
        for n, lyr in layers.items():
            cells[n], (ox, oy) = place_layer(lyr)
        rig[v] = (cells, order, (ox, oy), masks)

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        cells, order, (ox, oy), masks = rig[v]
        R = RIG[v]
        allm = np.zeros((CELL_H, CELL_W), bool)
        for c in cells.values():
            allm |= c[..., 3] > 0
        ys, xs = np.nonzero(allm)
        top_y = int(ys.min())
        top_x = float(xs[ys <= top_y + 1].mean())
        bbox0 = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, sx, (gx, gy) = compose(v, cells, order, act, f, top_y)
                top = (top_x + sx + gx, top_y + gy)
                centre = (CELL_W / 2 + gx, FOOT_Y - 12 + gy)
                fy_, fx_ = np.nonzero(a[..., 3] > 0)
                bbox = (int(fx_.min()), int(fy_.min()), int(fx_.max()), int(fy_.max())) if len(fy_) else bbox0
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, R["fwd"], top, centre, bbox)
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
    loop_diff = 0
    for v in RIG:
        for act in ("idle", "walk"):
            a0 = np.array(Image.open(frames_dir / v / act / "00.png").convert("RGBA"))
            a15 = np.array(Image.open(frames_dir / v / act / "15.png").convert("RGBA"))
            loop_diff += int((a0 != a15).any(axis=2).sum())
    S_h = int(np.ptp(np.nonzero((still["S"][..., 3] > 0).any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off),
              "semi_alpha_px": alpha_bad, "edge_touching_frames": clip, "still_diff_px": still_diff,
              "loop_end_diff_px(info)": loop_diff, "S_height": S_h, "official_front_height": target_h}
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
    meta = {"generator": "forge_gen1_0011_metapod_actions", "version": VERSION, "key": KEY, "dex": 11,
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
