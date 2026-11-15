#!/usr/bin/env python3
"""0070_weepinbell (normal): 8-direction x 4-action sheet, same flow as forge_raichu_mega_x_actions.py.

Source (2026-11-04, md/gen1-forge.md):
  art/ref_front.png / art/ref_back.png = official front.png / back.png x8 nearest.
  The Antigravity CLI (`agy -p`, prompt art/agy_prompt.md, no proper nouns) drew the key views:
    art/out/keys_2x2.png  : top-left S, top-right side view, bottom-left NE, bottom-right N
    art/out/sw_single.png : SW (3/4 front, faces down-left)
  SE = SW mirrored, NW = NE mirrored, the other side view = the drawn one mirrored.

Creature (counted on the official sprite): ONE legless pale-yellow bell (no legs, no arms) that floats, ONE big
round open mouth with thick pink lips at the wide lower end, two small eyes above it, TWO broad green leaves (one
each side, tips up and out), ONE short brown hooked stem on top.

Pipeline: key white (border-connected) -> area downscale (S height = official front.png opaque height, all grid
views share that scale, SW matched on height) -> palette_tool build / lock / apply -> outer light edge -> darkest
outline colour -> part rig found by colour (green components = the 2 leaves, brown on the top = the stem, the rest
= the bell; outline pixels go to the nearest part). The bell layer carries an underlay (nearest bell colour) for
every leaf / stem pixel inside its closed outline, so nothing opens when a leaf flaps. Leaves rotate about their
joint with the bell, the stem about its root; the whole creature rocks about the bell centre. 4x nearest, sampled
back to 1x (no new colours).

Actions (16 frames each):
  idle   : hovers with a slow bob, rocks a little, the leaves fan softly, the stem sways.
  walk   : no legs -> hover glide: two leaf beats per loop like wings, the bell bobs and tips forward, stem trails.
  attack : ACID SPRAY + SLAP: rears back with both leaves raised -> lunges mouth-first, the leaves slap forward ->
           a fan of acid droplets (reserved purple) bursts out of the open mouth + splash -> floats back.
  hurt   : white flash, knocked back and tipped back (max 16 deg), leaves droop, dizzy stars, recovers.

Never touches the official front/back png/gif.
run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0070_weepinbell_actions.py [--force] [--no-sync]
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
KEY = "0070_weepinbell"
REL = "monsterSprite/gen1/0070_weepinbell/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0070_weepinbell_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 128, 88, 74
UP = 4
FLASH, GLOW = (255, 255, 255), (255, 250, 208)
ACID, ACID_D = (200, 96, 232), (120, 40, 160)            # reserved: acid droplets of the attack only

SIDE = "W"            # side view drawn top-right in keys_2x2 (checked by eye)
OTHER = "W" if SIDE == "E" else "E"
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), SIDE: (SIDE, False), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), OTHER: (SIDE, True), "SW": ("SW", False)}
RIG = {
    "S": dict(fwd=(0.0, 1.0), back=False),
    "SW": dict(fwd=(-0.9, 0.45), back=False),
    SIDE: dict(fwd=((1.0, 0.0) if SIDE == "E" else (-1.0, 0.0)), back=False),
    "NE": dict(fwd=(0.9, -0.45), back=True),
    "N": dict(fwd=(0.0, -1.0), back=True),
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
    grid = target_h / crops["S"].height
    scale = {v: grid for v in src}
    scale["SW"] = target_h / crops["SW"].height
    dst.mkdir(parents=True, exist_ok=True)
    for v, im in crops.items():
        sm = im.resize((max(1, round(im.width * scale[v])), max(1, round(im.height * scale[v]))), Image.BOX)
        a = np.array(sm)
        a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
        a[a[..., 3] == 0] = 0
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
    op = a[..., 3] > 0
    edge = op & ~ndi.binary_erosion(op, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]], border_value=0)
    L = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    a[edge & (L > 110), :3] = ink
    return a


# ---------------------------------------------------------------- 2. rig (by colour)
def classify(still):
    op = still[..., 3] > 0
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    mx = np.maximum(np.maximum(r, g), b)
    dark = mx < 60
    green = op & ~dark & (g > r + 18) & (g > b + 8)
    brown = op & ~dark & (r > g + 12) & (g > b + 8) & (mx < 190)
    h, w = op.shape
    ys, xs = np.nonzero(op)
    top, bot = ys.min(), ys.max()
    # leaves: the two biggest green components (tiny spots on the bell stay bell)
    lab, n = ndi.label(green)
    leaves = []
    if n:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        order = np.argsort(sz)[::-1]
        for i in order[:2]:
            if sz[i] >= 0.03 * op.sum():
                leaves.append(lab == i + 1)
        for i in order[2:]:          # mid-sized green pieces (split by dark shading) join the nearest leaf
            if sz[i] >= 6 and leaves:
                m = lab == i + 1
                d = [ndi.distance_transform_edt(~L)[m].min() for L in leaves]
                if min(d) <= 3:
                    leaves[int(np.argmin(d))] |= m
    # stem: brown components that reach into the top 22 % of the sprite
    yy = np.mgrid[0:h, 0:w][0]
    lab, n = ndi.label(brown)
    stem = np.zeros_like(op)
    for i in range(1, n + 1):
        m = lab == i
        if yy[m].min() <= top + 0.22 * (bot - top) and m.sum() >= 3:
            stem |= m
    lbl = np.zeros(op.shape, int)            # 1 body, 2 stem, 3.. leaves
    seeds = op & ~dark
    lbl[seeds] = 1
    lbl[stem] = 2
    for i, L in enumerate(leaves):
        lbl[L] = 3 + i
    un = op & (lbl == 0) | (op & dark)
    lbl[op & dark] = 0
    known = lbl > 0
    if known.any():
        _, (iy, ix) = ndi.distance_transform_edt(~known, return_indices=True)
        lbl[un] = lbl[iy[un], ix[un]]
    # outline pixels assigned to the stem only when they hug the stem colour; otherwise bell outline stays bell
    names = ["body", "stem"] + [f"leaf{i}" for i in range(len(leaves))]
    masks = {nm: (lbl == i + 1) & op for i, nm in enumerate(names)}
    return masks


def build_rig(view, still):
    masks = classify(still)
    op = still[..., 3] > 0
    body = masks["body"]
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    mx = np.maximum(np.maximum(r, g), b)
    layers = {}
    # bell underlay: every non-bell pixel inside the closed bell outline gets the nearest lit bell colour
    inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(body, 6), iterations=6))[6:-6, 6:-6]
    inside &= ndi.binary_erosion(inside, iterations=1)
    lyr = np.zeros_like(still)
    lyr[body] = still[body]
    seam = inside & ~body & op
    fillable = body & (mx > 90)
    if seam.any() and fillable.any():
        _, (iy, ix) = ndi.distance_transform_edt(~fillable, return_indices=True)
        sy, sx = np.nonzero(seam)
        lyr[sy, sx] = still[iy[sy, sx], ix[sy, sx]]
        lyr[sy, sx, 3] = 255
    layers["body"] = lyr
    ys, xs = np.nonzero(body)
    bc = (float(xs.mean()), float(ys.mean()))
    pivots = {}
    for nm in masks:
        if nm == "body":
            continue
        m = masks[nm]
        L = np.zeros_like(still)
        L[m] = still[m]
        layers[nm] = L
        if not m.any():
            pivots[nm] = bc
            continue
        touch = m & ndi.binary_dilation(body, iterations=2)
        ty, tx = np.nonzero(touch if touch.any() else m)
        d = np.hypot(tx - bc[0], ty - bc[1])
        k = d <= np.percentile(d, 35)
        pivots[nm] = (float(tx[k].mean()), float(ty[k].mean()))
    # mouth = pink / red lip pixels
    pink = op & (r > g + 50) & (r > 110) & (b > g - 10) & ~masks.get("stem", np.zeros_like(op))
    my, mxp = np.nonzero(pink)
    mouth = (float(mxp.mean()), float(my.mean())) if len(my) > 4 else None
    order = ["body", "stem"] + sorted(n for n in masks if n.startswith("leaf"))
    return layers, masks, order, pivots, bc, mouth


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
    """g = [fwd px, up px]; lean deg (+ = tips toward forward / clockwise in front views);
    leaf = [flap deg leaf0, leaf1] (+ = tips raised), stem deg."""
    P = {"g": [0.0, 0.0], "lean": 0.0, "leaf": [0.0, 0.0], "stem": 0.0}
    fx = []
    if action == "idle":
        P["g"] = [0.0, wave(f, 1.6, 1, 0.0)]
        P["lean"] = wave(f, 2.5, 1, -0.6)
        P["leaf"] = [wave(f, 7.0, 1, 0.5), wave(f, 7.0, 1, 1.3)]
        P["stem"] = wave(f, 9.0, 1, 1.0)
    elif action == "walk":
        P["g"] = [0.0, wave(f, 2.0, 2, 0.0)]
        P["lean"] = wave(f, 4.0, 1, 0.0) * 0.5 + wave(f, 2.0, 2, 1.0)
        P["leaf"] = [wave(f, 16.0, 2, 0.0), wave(f, 16.0, 2, 0.0)]
        P["stem"] = wave(f, -12.0, 2, 0.8)
    elif action == "attack":
        P["g"] = [[0, -1, -3, -4, -4, 2, 7, 10, 10, 9, 7, 5, 3, 2, 1, 0][f],
                  [0, 1, 2, 3, 3, 1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0][f]]
        P["lean"] = [0, -3, -6, -9, -9, 0, 10, 14, 14, 12, 8, 5, 3, 1, 0, 0][f]
        fl = [0, 8, 16, 22, 24, 6, -18, -26, -26, -20, -12, -6, -3, -1, 0, 0][f]
        P["leaf"] = [fl, fl]
        P["stem"] = [0, 6, 12, 16, 16, 4, -14, -20, -18, -12, -6, -3, -1, 0, 0, 0][f]
        if 2 <= f <= 4:
            fx.append(("charge", f))
        if f in (6, 7, 8, 9, 10):
            fx.append(("acid", f))
        if f in (7, 8, 9):
            fx.append(("splash", f))
        if f in (5, 6):
            fx.append(("speed", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 5 else 0
        P["g"] = [-5.0 * env + jit, 2.0 * env]
        P["lean"] = -16.0 * env
        P["leaf"] = [-14.0 * env, -14.0 * env]
        P["stem"] = 14.0 * env
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


def up4(a):
    return np.repeat(np.repeat(a, UP, 0), UP, 1)


def rot4(img4, ang, piv):
    if abs(ang) < 0.8:
        return img4
    im = Image.fromarray(img4, "RGBA").rotate(-ang, resample=Image.NEAREST, center=(piv[0] * UP + UP / 2, piv[1] * UP + UP / 2))
    return np.array(im)


def rotpt(p, ang, c):
    a = math.radians(ang)
    dx, dy = p[0] - c[0], p[1] - c[1]
    return (c[0] + dx * math.cos(a) - dy * math.sin(a), c[1] + dx * math.sin(a) + dy * math.cos(a))


def compose(view, rg, action, f):
    R = RIG[view]
    cells, order, piv, bc, mouth = rg
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    sideways = abs(fwx) > 0.3
    side = 1.0 if fwx > 0 else -1.0
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    for n in order:
        lyr = up4(cells[n])
        if n == "stem":
            lyr = rot4(lyr, P["stem"] * (side if sideways else 1.0), piv[n])
        elif n.startswith("leaf"):
            i = int(n[4:])
            # + raises the tip: a leaf whose tip is left of its joint turns clockwise
            ys, xs = np.nonzero(cells[n][..., 3] > 0)
            sgn = 1.0 if xs.mean() < piv[n][0] else -1.0
            lyr = rot4(lyr, P["leaf"][min(i, 1)] * sgn, piv[n])
        m = lyr[..., 3] > 0
        canvas[m] = lyr[m]
    gf, gu = P["g"]
    lean = P["lean"]
    if sideways:
        ang = lean * side
    else:
        ang = lean * 0.3                      # toward / away from the camera: a small rock, never a flat spin
        gf *= 0.5
    canvas = rot4(canvas, ang, bc)
    gx, gy = round(gf * fwx), round(gf * fwy - gu)
    canvas = shift(canvas, gx * UP, gy * UP)
    out = canvas[1::UP, 1::UP].copy()
    return out, fx, ang, (gx, gy)


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


def droplet(a, x, y, big=False):
    put(a, x, y, ACID)
    put(a, x, y + 1, ACID_D)
    if big:
        put(a, x - 1, y + 1, ACID)
        put(a, x + 1, y + 1, ACID)
        put(a, x, y + 2, ACID_D)


def draw_fx(a, fx, fwd, mouth, centre, bbox):
    fwx, fwy = fwd
    px, py = -fwy, fwx
    for kind, k in fx:
        if kind == "charge":
            # acid gathers at the lips: droplets drawn into the mouth
            for j in range(k):
                ang = j * 2.3 + k
                rr = 9 - 2 * (k - 2)
                droplet(a, mouth[0] + math.cos(ang) * rr, mouth[1] + math.sin(ang) * rr * 0.6)
        elif kind == "acid":
            j = k - 6
            for s, sd in enumerate((0.0, 0.45, -0.45, 0.9, -0.9)):
                if s > 1 + j:
                    continue
                for t in range(2):
                    d = 5 + 4 * j + 5 * t
                    x = mouth[0] + fwx * d + px * sd * d * 0.6
                    y = mouth[1] + fwy * d * (0.6 if fwy > 0 else 1.0) + py * sd * d * 0.6 + (j * 0.5 if abs(fwx) > 0.3 else 0)
                    droplet(a, x, y, big=(t == 0))
        elif kind == "splash":
            r = {7: 3, 8: 5, 9: 3}[k]
            d = 16 + 3 * (k - 7)
            hx = mouth[0] + fwx * d
            hy = mouth[1] + fwy * d * (0.6 if fwy > 0 else 1.0)
            star(a, hx, hy, r, FLASH, GLOW)
            if k >= 8:
                for q in (-1, 1):
                    droplet(a, hx + px * q * (r + 2) + fwx * 2, hy + py * q * (r + 2) - 1)
        elif kind == "speed":
            if abs(fwx) > 0.3:
                bx = bbox[0] - 2 if fwx > 0 else bbox[2] + 2
                for j, yy in enumerate(range(bbox[1] + 6, bbox[3] - 4, 6)):
                    for t in range(4 - (j % 2)):
                        put(a, bx - fwx * (t + j % 2), yy, GLOW)
        elif kind == "stars":
            for j in range(3):
                ang = j * 2.1 + k * 0.8
                cx, cy = centre[0] + math.cos(ang) * 10, bbox[1] - 3 + math.sin(ang) * 2.5
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
    ap.add_argument("--stills-only", action="store_true")
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
        run_tool(["build", "--input", str(raw), "--colors", "24", "--reserve",
                  *["#%02x%02x%02x" % c for c in (FLASH, GLOW, ACID, ACID_D)],
                  "--balance", "--name", "bell-plant", "--output-dir", str(pal_dir)])
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
    dbg = work / "rig_debug"; dbg.mkdir(exist_ok=True)
    for v in RIG:
        layers, masks, order, pivots, bc, mouth = build_rig(v, still[v])
        cells = {}
        for n, lyr in layers.items():
            cells[n], (ox, oy) = place_layer(lyr)
        piv = {n: (p[0] + ox, p[1] + oy) for n, p in pivots.items()}
        bcc = (bc[0] + ox, bc[1] + oy)
        if mouth is None or RIG[v]["back"]:
            ys, xs = np.nonzero(cells["body"][..., 3] > 0)
            fwx, fwy = RIG[v]["fwd"]
            mouth_c = (bcc[0] + fwx * 8, (ys.min() + 6) if fwy < 0 else bcc[1])
        else:
            mouth_c = (mouth[0] + ox, mouth[1] + oy)
        rig[v] = (cells, order, piv, bcc, mouth_c)
        # rig debug: colour per part
        cm = np.zeros(still[v].shape[:2] + (3,), np.uint8)
        pal_dbg = {"body": (230, 210, 80), "stem": (140, 80, 30), "leaf0": (40, 200, 60), "leaf1": (40, 120, 220)}
        for n, m in masks.items():
            cm[m] = pal_dbg.get(n, (255, 0, 255))
        for n, p in pivots.items():
            x, y = int(p[0]), int(p[1])
            cm[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = (255, 0, 0)
        Image.fromarray(cm).resize((cm.shape[1] * 6, cm.shape[0] * 6), Image.NEAREST).save(dbg / f"{v}.png")
        print(v, "parts", {n: int(m.sum()) for n, m in masks.items()}, "mouth", mouth_c, flush=True)
    if args.stills_only:
        return

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        R = RIG[v]
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, ang, (gx, gy) = compose(v, rig[v], act, f)
                cells, order, piv, bcc, mouth_c = rig[v]
                mp = rotpt(mouth_c, ang, bcc)
                mp = (mp[0] + gx, mp[1] + gy)
                centre = (bcc[0] + gx, bcc[1] + gy)
                fy_, fx_ = np.nonzero(a[..., 3] > 0)
                bbox = (int(fx_.min()), int(fy_.min()), int(fx_.max()), int(fy_.max()))
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, R["fwd"], mp, centre, bbox)
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
    meta = {"generator": "forge_gen1_0070_weepinbell_actions", "version": VERSION, "key": KEY, "dex": 70,
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
