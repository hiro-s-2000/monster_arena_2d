#!/usr/bin/env python3
"""
Mega Latios (`0381_latios-mega` / メガラティオス) 8-direction x 4-action sheet, v2 ("cool" pass).

Builds on the Antigravity v1 pipeline (`forge_latios_mega_actions.py`): the same 7-layer anatomical
part-rig (tail fins / jet wings / torso / lower jaw / rigid head) cut from the native front.png / back.png,
the same palette lock and the same output layout (cell 120x96, foot_y 86, 16 frames, rows S..SW per action),
so the game loads it unchanged. v2 adds whole-body motion on top of the part rig:

  idle   : hover bob + phase-lagged wing/fin sway + slow bank tilt; a short growl (jaw +1/+2) with a small
           roar ring at frames 9..12.
  walk   : nose-down flight glide (tilt), strong wing beats (2 per loop), forward surge, speed lines behind.
  attack : ROAR -> CORKSCREW -> DIVE.  0..4 rear back, head up, jaw wide (+3), expanding roar rings;
           5..10 full 360-degree aileron roll (belly-up at the midpoint) while surging forward;
           11..12 nose-down dive strike with impact burst; 13..15 pull-up recovery.
  hurt   : white hit flash, knock-back with a half tumble (roll squash) and backward tilt, pained jaw,
           sparks, recovery.

Motion rules (monster-action-forge-architect / avian-style flyer): head stays a rigid integer shift (the face
and eye never deform), wings rotate about their shoulder sockets, whole-body rotation uses a 4x
nearest-neighbour RotSprite-style pass, and every pixel (including FX) is snapped to the sprite palette.

Resumable: each finished frame is checkpointed under <out>/_work/frames/<action>/<dir>/NN.png together
with a parameter hash; re-running skips finished frames (use --force to rebuild). QC (palette, canvas
clipping, gait_loop.py loop report) is written to <out>/_work/qa/.

usage: python forge_latios_mega_actions_v2.py [--force] [--no-sync] [--skip-qa]
"""

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("latios_v1", HERE / "forge_latios_mega_actions.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)
sys.path.insert(0, str(HERE))
import mouth_open  # noqa: E402  (small face-following mouth opening, 2026-11-04)
from scipy import ndimage as ndi  # noqa: E402

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen3/0381_latios/mega/latios-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"
SYNC_MEGA_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen3/0381_latios/mega/latios-mega")
SYNC_OUT_DIR = SYNC_MEGA_DIR / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = v1.DIRS
ACTIONS = v1.ACTIONS
N = 16
CELL_W, CELL_H, FOOT_Y = 120, 96, 86
VERSION = "latios_mega_v2m"

# Mouth (2026-11-04 fix, user: "rectangular mouth makes the chin jut"): the lower jaw no longer drops and the
# crimson/purple rectangular pocket the v1 rig paints behind the jaw is not used. Instead mouth_open.open_mouth
# carves a small lens-shaped gap just under the closed mouth line of the art (both corners closed).
# MOUTH_LINE_S: measured on the un-flipped front still (S/SW/W rigs share it), cell coords, hinge -> lip tip.
MOUTH_LINE_S = [(32, 38), (29, 38), (26, 38)]
MOUTH_DEEP = (16, 16, 16)       # outline colour (darkest palette entry)
MOUTH_INNER = (197, 57, 90)     # existing crimson (eye / mouth interior)


def mouth_amount(action: str, jaw: int) -> int:
    """Old jaw-drop value -> small opening: 1 for growl/walk/hurt, 2 only at the attack roar peak."""
    if jaw <= 0:
        return 0
    return 2 if (action == "attack" and jaw >= 3) else 1


WHITE = (250, 250, 250)
PINK = (209, 111, 136)
LAV = (152, 118, 198)


# ---------------------------------------------------------------- whole-body transforms
def rotate_frame(img: np.ndarray, ang: float, pivot: Tuple[float, float]) -> np.ndarray:
    """RotSprite-style: 4x nearest upscale, nearest rotation, centre-sample downscale (cleaner than 1x)."""
    if abs(ang) < 0.5:
        return img.copy()
    up = np.repeat(np.repeat(img, 4, axis=0), 4, axis=1)
    r = v1.rotate_layer_nearest(up, ang, (pivot[0] * 4 + 1.5, pivot[1] * 4 + 1.5))
    return r[1::4, 1::4].copy()


def roll_frame(img: np.ndarray, theta: float, cy: float) -> np.ndarray:
    """Aileron roll about the body's horizontal axis: rows squash by cos(theta); belly-up (vertical mirror) past 90 deg."""
    c = math.cos(theta)
    if abs(c - 1.0) < 1e-3:
        return img.copy()
    s = max(0.34, abs(c))
    h = img.shape[0]
    out = np.zeros_like(img)
    for y in range(h):
        sy = cy + (y - cy) / s
        if c < 0:
            sy = cy - (y - cy) / s
        sy = int(round(sy))
        if 0 <= sy < h:
            out[y] = img[sy]
    return out


def body_box(img: np.ndarray):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def shift(img, dx, dy):
    return v1.shift_layer_int(img, int(round(dx)), int(round(dy)))


def despeckle(img: np.ndarray, passes: int = 2) -> np.ndarray:
    """Remove 1px spurs left by nearest-neighbour rotation (opaque pixels with <=1 opaque 4-neighbour)."""
    out = img.copy()
    for _ in range(passes):
        a = out[..., 3] >= 128
        p = np.pad(a, 1)
        n = p[:-2, 1:-1].astype(int) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        spur = a & (n <= 1)
        if not spur.any():
            break
        out[spur] = 0
    return out


# ---------------------------------------------------------------- FX (all palette colours)
def put(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def roar_rings(out, mouth, facing, k, strength):
    """Expanding ')' arcs in front of the mouth. k = 0.. ring age, strength = number of rings."""
    mx, my = mouth
    for r_i in range(strength):
        rad = 5 + k * 3 + r_i * 5
        col = WHITE if r_i == 0 else (LAV if r_i == 1 else PINK)
        for deg in range(-55, 56, 6):
            a = math.radians(deg)
            x = mx + facing * rad * math.cos(a)
            y = my + rad * math.sin(a) * 0.9
            put(out, int(round(x)), int(round(y)), col)
    return out


def speed_lines(out, facing, n, seed, length=10):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    tail_x = x1 + 2 if facing < 0 else x0 - 2
    for _ in range(n):
        y = int(rng.integers(y0 + 6, y1 - 4))
        off = int(rng.integers(0, 6))
        L = int(rng.integers(length // 2, length + 1))
        for i in range(L):
            put(out, int(tail_x - facing * (off + i)), y, WHITE if i < 2 else LAV)
    return out


def impact_burst(out, center, size, phase):
    cx, cy = center
    for k in range(8):
        a = k * math.pi / 4 + phase * 0.4
        for r in range(size // 2, size):
            put(out, int(round(cx + r * math.cos(a))), int(round(cy + r * math.sin(a))), WHITE if r < size - 2 else PINK)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            put(out, cx + dx, cy + dy, WHITE)
    return out


def sparks(out, frame_idx, facing):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    d = frame_idx
    pts = [((x0 + x1) // 2 + facing * (6 + d * 2), y0 + 8 - d), ((x0 + x1) // 2 - facing * (10 + d), y0 + 4 - d // 2),
           ((x0 + x1) // 2 + facing * (2 + d), y0 - 2 - d)]
    for sx, sy in pts:
        put(out, sx, sy, WHITE); put(out, sx + 1, sy, PINK); put(out, sx, sy + 1, PINK)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


# ---------------------------------------------------------------- per-frame pose
def pose(action: str, f: int) -> Dict[str, float]:
    t = f / N
    ph = 2 * math.pi * t
    p = dict(dx=0.0, dy=0.0, tilt=0.0, roll=0.0, head_dx=0.0, head_dy=0.0, jaw=0, wn=0.0, wf=0.0, tn=0.0, tf=0.0,
             ring=-1, rings=0, lines=0, burst=0, flash=0, sparks=0)
    if action == "idle":
        p["dy"] = 2.2 * math.sin(ph)
        p["dx"] = 0.8 * math.cos(ph)
        p["tilt"] = 2.5 * math.sin(ph - 0.6)
        p["wn"] = 7.0 * math.sin(ph - 0.5)
        p["wf"] = -7.0 * math.sin(ph - 0.8)
        p["tn"] = 5.0 * math.sin(ph - 1.0)
        p["tf"] = -5.0 * math.sin(ph - 1.2)
        growl = {9: 1, 10: 2, 11: 2, 12: 1}
        p["jaw"] = growl.get(f, 0)
        p["head_dy"] = -1 if f in (10, 11) else 0
        if f in (10, 11, 12):
            p["ring"], p["rings"] = f - 10, 1
    elif action == "walk":
        p["dy"] = 1.6 * math.sin(2 * ph)
        p["dx"] = 2.0 * math.sin(ph)
        p["tilt"] = -7.0 + 2.0 * math.sin(2 * ph)        # nose down glide
        beat = math.sin(2 * ph)
        p["wn"] = 13.0 * beat
        p["wf"] = -13.0 * beat
        p["tn"] = 7.0 * math.sin(2 * ph - 0.8)
        p["tf"] = -7.0 * math.sin(2 * ph - 0.8)
        p["jaw"] = 1 if beat > 0.6 else 0
        p["lines"] = 3
    elif action == "attack":
        dx_t = [0, -2, -3, -4, -4, -2, 1, 3, 5, 7, 8, 9, 8, 5, 2, 0]
        dy_t = [0, -1, -3, -4, -4, -3, -2, -1, 0, 1, 1, 2, 1, 1, 0, 0]
        tilt_t = [0, 6, 12, 15, 15, 8, 0, -4, -8, -10, -13, -15, -12, -7, -3, 0]
        jaw_t = [0, 1, 3, 3, 3, 2, 1, 1, 1, 1, 2, 3, 3, 1, 0, 0]
        p["dx"], p["dy"], p["tilt"], p["jaw"] = dx_t[f], dy_t[f], tilt_t[f], jaw_t[f]
        p["head_dy"] = [0, -1, -2, -2, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["head_dx"] = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0][f]  # head thrown back (sign set by facing)
        wing_t = [0, 10, 18, 22, 22, 10, -8, -14, -14, -10, -16, -20, -14, -6, -2, 0]
        p["wn"], p["wf"] = wing_t[f], -wing_t[f]
        p["tn"], p["tf"] = wing_t[f] * 0.5, -wing_t[f] * 0.5
        if 5 <= f <= 10:                                   # full corkscrew: 0 -> 360 deg over 6 frames
            p["roll"] = 2 * math.pi * (f - 4) / 6
        if 2 <= f <= 5:
            p["ring"], p["rings"] = f - 2, 3
        if 6 <= f <= 11:
            p["lines"] = 6
        if f in (11, 12):
            p["burst"] = 1
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12))
        p["dx"] = 5.0 * env + ((1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0)
        p["dy"] = -2.0 * env
        p["tilt"] = 16.0 * env
        if 2 <= f <= 7:
            p["roll"] = math.pi * 0.3 * math.sin(math.pi * (f - 2) / 5)  # tumble squash, no flip
        p["jaw"] = 2 if 1 <= f <= 6 else (1 if f in (7, 8) else 0)
        p["wn"], p["wf"] = -14 * env, 14 * env
        p["tn"], p["tf"] = -8 * env, 8 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 7 else 0
    return p


def mouth_point(layers, jaw_dx, jaw_dy, facing):
    bb = body_box(layers["lower_jaw"])
    if not bb:
        bb = body_box(layers["head_upper"])
    x0, y0, x1, y1 = bb
    x = x0 if facing < 0 else x1
    return (x + jaw_dx + int(facing * 2), (y0 + y1) // 2 + jaw_dy)


def synth(rig, action, f, palette):
    layers, pivots, facing, is_back, mouth_line = rig
    p = pose(action, f)
    # facing: -1 = sprite looks left.  "forward" moves along facing; tilt>0 = nose up (rear back).
    fwd = -1.0 if facing < 0 else 1.0
    bdx, bdy = p["dx"] * fwd, p["dy"]
    hdx, hdy = bdx - p["head_dx"] * fwd, bdy + p["head_dy"]
    # front views: jaw stays on the head (no chin drop; mouth_open carves the opening below).
    # back views: no visible mouth; kept exactly as v2 (the cheek tip behind the torso) so back.gif is unchanged.
    jdx, jdy = hdx, hdy + (p["jaw"] if is_back else 0)

    def rot(name, ang, sgn):
        return v1.rotate_layer_nearest(layers[name], ang * sgn, pivots[name])

    s = -facing
    tf = shift(rot("tail_fin_far", p["tf"], facing), bdx, bdy)
    wf = shift(rot("jet_wing_far", p["wf"], facing), bdx, bdy)
    tn = shift(rot("tail_fin_near", p["tn"], s), bdx, bdy)
    wn = shift(rot("jet_wing_near", p["wn"], s), bdx, bdy)
    torso = shift(layers["torso"], bdx, bdy)
    jaw = shift(layers["lower_jaw"], jdx, jdy)
    head = shift(layers["head_upper"], hdx, hdy)
    order = [tf, wf, torso, tn, jaw, head, wn] if not is_back else [tf, wf, jaw, torso, tn, head, wn]
    img = v1.composite_layers((CELL_H, CELL_W, 4), order)
    amt = mouth_amount(action, p["jaw"])
    if amt and mouth_line and not is_back:
        line = mouth_open.transform_points(mouth_line, dx=int(round(hdx)), dy=int(round(hdy)))
        img = mouth_open.open_mouth(img, line, amount=amt, deep=MOUTH_DEEP, inner=MOUTH_INNER, peak=0.6)

    bb = body_box(img)
    cx, cy = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2) if bb else (CELL_W / 2, FOOT_Y - 35)
    if p["roll"]:
        img = roll_frame(img, p["roll"], cy)
    if p["tilt"]:
        # screen rotation sign: nose up for a left-facing sprite = clockwise on screen
        img = rotate_frame(img, -p["tilt"] * fwd, (cx, cy))
    img = despeckle(v1.snap_to_palette(img, palette))

    if p["flash"]:
        img = flash_white(img)
    if p["rings"] and p["ring"] >= 0 and not is_back:
        mx, my = mouth_point({"lower_jaw": jaw, "head_upper": head}, 0, 0, fwd)
        img = roar_rings(img, (mx, my), fwd, p["ring"], p["rings"])
    if p["rings"] and p["ring"] >= 0 and is_back:   # from behind: rings radiate past the head
        hb = body_box(head)
        if hb:
            img = roar_rings(img, ((hb[0] + hb[2]) // 2 + int(fwd * 6), hb[1] + 4), fwd, p["ring"], max(1, p["rings"] - 1))
    if p["lines"]:
        img = speed_lines(img, fwd, p["lines"], seed=f * 31 + len(action), length=12 if action == "attack" else 8)
    if p["burst"]:
        bb2 = body_box(img)
        if bb2:
            tip = (bb2[0] - 3, bb2[3] - 10) if fwd < 0 else (bb2[2] + 3, bb2[3] - 10)
            img = impact_burst(img, tip, 9 if f == 11 else 12, f)
    if p["sparks"]:
        img = sparks(img, f - 2, fwd)
    return v1.snap_to_palette(img, palette)


# ---------------------------------------------------------------- pipeline with checkpoints
def params_hash() -> str:
    src = Path(__file__).read_bytes() + (HERE / "forge_latios_mega_actions.py").read_bytes()
    src += (MEGA_DIR / "front.png").read_bytes() + (MEGA_DIR / "back.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def clean_front_layers(base: np.ndarray, flip: bool):
    """Rebuild the front-view `torso` and `lower_jaw` layers exactly like v1.build_rig_layers but WITHOUT the
    rectangular crimson/purple mouth pocket (it peeked out under the chin whenever the head moved)."""
    raw = v1.segment_front_native(base)
    src = base[:, ::-1, :].copy() if flip else base.copy()
    masks = {k: (m[:, ::-1].copy() if flip else m.copy()) for k, m in raw.items()}
    ys, xs = np.where(src[:, :, 3] >= 128)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dx0, dy0 = (CELL_W - sw) // 2, FOOT_Y - sh
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dy0:dy0 + sh, dx0:dx0 + sw] = src[y0:y1, x0:x1]
    cm = {}
    for k, m in masks.items():
        c = np.zeros((CELL_H, CELL_W), bool)
        c[dy0:dy0 + sh, dx0:dx0 + sw] = m[y0:y1, x0:x1]
        cm[k] = c
    fill = v1.is_latios_body_fill(canvas)
    fg_t = cm["tail_fin_near"] | cm["lower_jaw"] | cm["head_upper"] | cm["jet_wing_near"]
    torso = v1.build_part_layer(canvas, cm["torso"], ndi.binary_erosion(cm["torso"] | fg_t, iterations=1) & fg_t,
                                cushion_px=6, allowed_fill_mask=fill)
    fg_j = cm["head_upper"]
    jaw = v1.build_part_layer(canvas, cm["lower_jaw"], ndi.binary_erosion(cm["lower_jaw"] | fg_j, iterations=1) & fg_j,
                              cushion_px=3, allowed_fill_mask=fill)
    # mouth line follows the same placement (un-flipped canvas -> native -> this canvas)
    if not flip:
        line = list(MOUTH_LINE_S)
    else:
        ys0, xs0 = np.where(base[:, :, 3] >= 128)
        ux0 = (CELL_W - (int(xs0.max()) + 1 - int(xs0.min()))) // 2 - int(xs0.min())   # canvas = nx + ux0
        line = [(dx0 + x1 - 1 - (x - ux0), y) for x, y in MOUTH_LINE_S]
    return torso, jaw, line


def build_rigs():
    front = np.array(Image.open(MEGA_DIR / "front.png").convert("RGBA"))
    back = np.array(Image.open(MEGA_DIR / "back.png").convert("RGBA"))
    palette = v1.extract_palette(front, back)
    rigs = {}
    for d in DIRS:
        use_back = d in ("N", "NE", "NW")
        if not use_back:
            flip = d in ("SE", "E"); facing = 1.0 if flip else -1.0
        else:
            flip = d in ("NW",); facing = -1.0 if flip else 1.0
        layers, pivots = v1.build_rig_layers(back if use_back else front, is_back=use_back, flip_x=flip,
                                             cell_w=CELL_W, cell_h=CELL_H, foot_y=FOOT_Y)
        line = None
        if not use_back:
            layers["torso"], layers["lower_jaw"], line = clean_front_layers(front, flip)
        rigs[d] = (layers, pivots, facing, use_back, line)
    return rigs, palette


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    work = OUT_DIR / "_work"
    frames_dir = work / "frames"
    state_p = work / "state.json"
    h = params_hash()
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    if args.force or state.get("hash") != h:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        state = {"hash": h, "done": []}
    frames_dir.mkdir(parents=True, exist_ok=True)

    rigs, palette = build_rigs()
    total = len(ACTIONS) * len(DIRS)
    for act in ACTIONS:
        for d in DIRS:
            key = f"{act}/{d}"
            if key in state["done"]:
                continue
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                fp = od / f"{f:02d}.png"
                if fp.exists():
                    continue
                Image.fromarray(synth(rigs[d], act, f, palette), "RGBA").save(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state, indent=1))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    # assemble (same layout as v1: per action 8 rows x 16 cols; sheet = actions stacked)
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, N * CELL_W, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(N):
                sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png"))
        sheets[act] = sh
        im = Image.fromarray(sh, "RGBA")
        im.save(OUT_DIR / f"{act}.png"); im.save(OUT_DIR / f"{act}_8dir.png")
        Image.fromarray(sh[:CELL_H], "RGBA").save(OUT_DIR / f"{act}_S_strip.png")
    master = np.concatenate([sheets[a] for a in ACTIONS], axis=0)
    Image.fromarray(master, "RGBA").save(OUT_DIR / "sheet.png")
    Image.fromarray(master, "RGBA").save(OUT_DIR / "all_8dir_actions.png")
    v1.build_preview_strip(sheets, CELL_W, CELL_H).save(OUT_DIR / "preview.png")
    # full-motion review sheet: S row and E row of every action, all 16 frames
    rev = np.full((8 * CELL_H, N * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for ri, di in enumerate((0, 2)):
            src = sheets[act][di * CELL_H:(di + 1) * CELL_H]
            m = src[..., 3] >= 128
            dst = rev[(ai * 2 + ri) * CELL_H:(ai * 2 + ri + 1) * CELL_H]
            dst[m] = src[m]
    Image.fromarray(rev, "RGBA").save(work / "review_S_E_all_frames.png")

    idle = sheets["idle"]
    back_frames = [idle[4 * CELL_H:5 * CELL_H, i * CELL_W:(i + 1) * CELL_W] for i in range(N)]
    v1.save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
    v1.save_transparent_gif([b[:, ::-1].copy() for b in back_frames], MEGA_DIR / "back-reverse.gif", duration_ms=75)

    # QC
    qa = work / "qa"; qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c[:3]) for c in palette}
    a = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[a][:, :3], axis=0)}
    off = sorted(used - pal)
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(N):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    report = {"version": VERSION, "hash": h, "off_palette_colors": len(off), "edge_touching_frames": clip}
    if not args.skip_qa and GAIT_LOOP.exists():
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run([str(FORGE / ".venv/bin/python"), "-I", str(GAIT_LOOP), "select", "--frames-dir",
                                str(frames_dir / act / "S"), "--fps", "13", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                               capture_output=True, text=True)
            rep = od / "loop-report.json"
            report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation") if rep.exists() else (r.stderr[-400:] or "no report")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "edge_touching_frames"}, ensure_ascii=False)[:600])
    print("edge-touching:", {k: len(v) for k, v in clip.items()})

    meta = {"generator": "forge_latios_mega_actions_v2", "version": VERSION, "key": "0381_latios-mega", "dex": 381,
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": 70, "frames": N, "directions": DIRS,
            "actions": ACTIONS, "palette_colors": int(len(palette)), "off_palette_colors": len(off)}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_sync and SYNC_MEGA_DIR.exists():
        SYNC_OUT_DIR.mkdir(parents=True, exist_ok=True)
        for fp in OUT_DIR.iterdir():
            if fp.is_file():
                shutil.copy2(fp, SYNC_OUT_DIR / fp.name)
        for g in ("back.gif", "back-reverse.gif"):
            shutil.copy2(MEGA_DIR / g, SYNC_MEGA_DIR / g)
        print("synced ->", SYNC_OUT_DIR)


if __name__ == "__main__":
    main()
