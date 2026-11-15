#!/usr/bin/env python3
"""Phione (0489) normal 8-direction action sheet from this form's own
front.png / back.png (no pokeAnim, no image generation).

Measured on normal/front.png: 96x96, 841 opaque, bbox x23-71 y29-65,
14 colours, alpha 0/255, one 4-connected piece.

Body plan (from the pixels): legless little sea drifter. No legs, no wings,
no tail fin. A big round head, two flat flippers and a short pointed lower
body; a tiny existing mouth line between the eyes.
  head     round head x29-52 y29-54 with the top knob x37-48 y29-35,
           eyes x31-35 y44-50 (far, narrower) and x44-48 y45-50 (near),
           the mouth: a 2px dark line at x38-39 y51 with cheek corners
           at x37 / x40 y50
  antenna  the long curved feeler leaving the knob at x49 y32-35 and arcing
           right and down to its round tip x63-72 y48-57
  arm_l    flat left flipper x23-34 y55-61 (reaches far out, face side)
  arm_r    flat right flipper x44-56 y57-64
  torso    chest with the red gem x36-46 y55-61 and the pointed lower body
           down to y65
The eyes sit left of the head centre and the near eye is the right one, so
front.png faces screen-left; the antenna trails behind (screen-right).

back.png (802 opaque, same palette) is the back view facing up-right: plain
round head x37-60, knob x42-55 y30-36, the antenna arcing left to its tip
x18-26 y44-58, flippers left x36-45 y57-63 and right x57-66 y56-61, the
pointed lower body x46-55 to y66.

Directions: front still S SW W unflipped, SE E mirrored. back still NE N
unflipped, NW mirrored. E faces screen-right, W screen-left.

  idle   : hover bob in the water (integer 0..2 up), the antenna sways, the
           flippers paddle softly, a glint on the antenna tip. Frame 0 and
           the loop end are the still.
  walk   : swim glide (no legs): bob x2, both flippers stroke twice, the
           antenna streams back, water speed lines and bubbles behind.
  attack : 0-4 roar: the existing mouth opens (dark O with the pink tongue
           from the palette), the antenna flares up, the flippers rise and
           water rings burst from the mouth (back rows: from the knob).
           5-7 turn-around: front -> narrow -> real back view -> narrow
           mirrored (a facing change, never an in-plane rotation; the body
           stays one piece) inside a whirlpool. 8-10 strike: the face-side
           flipper thrusts up-forward, the far flipper power-strokes down,
           the antenna whips and a water pulse fires from the mouth.
           11-12 splash burst. 13-15 recover.
  hurt   : 12 frames, white flash, knock back + lift (integer), antenna and
           flippers jolt (<= 24 deg), the mouth winces open, droplets, recover.
The head (knob, eyes, mouth) only ever moves by integer pixels.

Palette-locked to this still. front/back png and gif are not written.
Resumable via agent-sprite-forge/_work. --force clears only this form's frames.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_phione_actions.py [--samples] [--publish]
"""

import argparse
import fcntl
import hashlib
import json
import math
import shutil
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0489_phione/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0489_phione/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
BOUNDS = ROOT / "data/sprite_bounds.js"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
VIEW = {
    "S": ("front", False), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False), "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "phione_v2"
KEY = "0489_phione"

C_K = (16, 16, 16)
C_O = (57, 82, 156)
C_A = (106, 197, 246)
C_B = (24, 156, 213)
C_D = (24, 106, 222)
C_M = (49, 123, 172)
C_L = (156, 230, 255)
C_WL = (205, 238, 255)
C_W = (255, 255, 255)
C_R = (172, 57, 41)
C_RR = (230, 65, 16)
C_P = (222, 131, 156)

FRONT_N = 841
FRONT_BBOX = (23, 71, 29, 65)
BACK_N = 802
BACK_BBOX = (18, 66, 30, 66)

ROT_PARTS = ("antenna", "arm_l", "arm_r")

# ---------------------------------------------------------------- helpers
def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def labels(mask):
    """4-connected components as a list of boolean masks, largest first."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        comp = np.zeros_like(mask)
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        while q:
            y, x = q.popleft()
            comp[y, x] = True
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        out.append(comp)
    out.sort(key=lambda m: -int(m.sum()))
    return out


def component_sizes(mask):
    return [int(m.sum()) for m in labels(mask)]


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
        out[...] = 0
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = a.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    mapped = palette[dist.argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv.reshape(-1)]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    out[~a] = 0
    return out


def shift(layer, dx, dy):
    dx_i, dy_i = int(round(dx)), int(round(dy))
    if dx_i == 0 and dy_i == 0:
        return layer.copy()
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0, sx1 = max(0, -dx_i), min(w, w - dx_i)
    sy0, sy1 = max(0, -dy_i), min(h, h - dy_i)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy_i:sy1 + dy_i, sx0 + dx_i:sx1 + dx_i] = layer[sy0:sy1, sx0:sx1]
    return out


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest rotation, positive = clockwise on screen. No squash."""
    if not np.any(layer[..., 3] >= 128):
        return layer.copy()
    norm = angle_deg % 360.0
    if min(norm, 360.0 - norm) < 0.45:
        return layer.copy()
    up = np.repeat(np.repeat(layer, 4, axis=0), 4, axis=1)
    px, py = pivot[0] * 4 + 1.5, pivot[1] * 4 + 1.5
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = xx - px, yy - py
    src_x = np.rint(px + c * rx + s * ry).astype(np.int32)
    src_y = np.rint(py - s * rx + c * ry).astype(np.int32)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    spun = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    spun[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    inv = spun[2::4, 2::4].copy()
    for _ in range(2):
        al = inv[..., 3] >= 128
        pp = np.pad(al, 1)
        n = pp[:-2, 1:-1].astype(np.int16) + pp[2:, 1:-1] + pp[1:-1, :-2] + pp[1:-1, 2:]
        hole = (~al) & (n >= 3)
        if not np.any(hole):
            break
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            src = np.roll(np.roll(inv, -dy, 0), -dx, 1)
            take = hole & (src[..., 3] >= 128) & (inv[..., 3] < 128)
            inv[take] = src[take]
    return inv


def content_point(x, y, angle_deg, pivot):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    vx, vy = x - pivot[0], y - pivot[1]
    return pivot[0] + c * vx - s * vy, pivot[1] + s * vx + c * vy


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int64)
    for i in range(0, len(ty), 400):
        sl = slice(i, i + 400)
        d = (ty[sl, None].astype(np.int32) - pts[None, :, 0]) ** 2
        d += (tx[sl, None].astype(np.int32) - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def touches_edge(tile):
    c = tile[..., 3] >= 128
    if not np.any(c):
        return False
    return bool(c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any())


def narrow_x(img, scale, cx):
    """Narrow horizontally around cx for a turn frame. Union sampling keeps it one piece."""
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    for x in range(w):
        s0 = cx + (x - 0.5 - cx) / scale
        s1 = cx + (x + 0.5 - cx) / scale
        a0, a1 = int(math.floor(min(s0, s1) + 0.5)), int(math.floor(max(s0, s1) - 0.5 + 1e-9))
        mid = int(round(cx + (x - cx) / scale))
        order = [mid] + [c for c in range(a0, a1 + 1) if c != mid]
        for sx in order:
            if not (0 <= sx < w):
                continue
            col = img[:, sx]
            use = (out[:, x, 3] < 128) & (col[:, 3] >= 128)
            out[use, x] = col[use]
    return out


# ---------------------------------------------------------------- measure
def check_still(img, n_exp, bbox_exp, name):
    a = img[..., 3] >= 128
    if img.shape[:2] != (96, 96):
        raise SystemExit(f"{name} is not 96x96")
    if int(a.sum()) != n_exp:
        raise SystemExit(f"{name} opaque changed: {int(a.sum())}")
    ys, xs = np.nonzero(a)
    if (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())) != bbox_exp:
        raise SystemExit(f"{name} bbox changed; remeasure before forging")
    if component_sizes(a) != [n_exp]:
        raise SystemExit(f"{name} is not one 4-connected piece")
    return a


def is_col(img, a, cols):
    rgb = img[..., :3]
    hit = np.zeros(a.shape, bool)
    for c in cols:
        hit |= np.all(rgb == np.array(c, np.uint8), axis=-1)
    return a & hit


def contact_pivot(part, body, top_rows=None, bottom_rows=None):
    """Mean of the part pixels that touch the body: the joint the part turns on.

    top_rows keeps only the highest rows of the contact (a lobe hangs from
    the top of its seam, not from the middle of it)."""
    touch = part & dilate(body, 1)
    ys, xs = np.nonzero(touch)
    if len(ys) == 0:
        raise SystemExit("part does not touch the body")
    if top_rows is not None:
        keep = ys <= ys.min() + top_rows - 1
        ys, xs = ys[keep], xs[keep]
    if bottom_rows is not None:
        keep = ys >= ys.max() - bottom_rows + 1
        ys, xs = ys[keep], xs[keep]
    return (float(xs.mean()), float(ys.mean()))


def absorb_strays(parts, order):
    """Keep each part's main piece; hand stray bits to the part they touch most."""
    for name in order:
        if not np.any(parts[name]):
            continue
        comps = labels(parts[name])
        for stray in comps[1:]:
            ring = dilate(stray, 1) & ~stray
            best, score = None, -1
            for other, m in parts.items():
                if other == name:
                    continue
                s = int((ring & m).sum())
                if s > score:
                    best, score = other, s
            parts[name] = parts[name] & ~stray
            parts[best] = parts[best] | stray
    return parts




def segment(img, kind):
    if kind == "front":
        a = check_still(img, FRONT_N, FRONT_BBOX, "front")
    else:
        a = check_still(img, BACK_N, BACK_BBOX, "back")
    yy, xx = np.mgrid[0:96, 0:96]
    if kind == "front":
        antenna = a & (((xx >= 49) & (yy <= 34)) | ((xx >= 50) & (yy == 35)) | ((xx >= 53) & (yy >= 36)))
        arm_l = a & ~antenna & (yy >= 55) & (xx <= 34)
        arm_r = a & ~antenna & (((yy >= 58) & (xx >= 44)) | ((yy == 57) & (xx >= 47)))
        head = a & ~antenna & (yy <= 54)
    else:
        antenna = a & (((yy <= 35) & (xx <= 41)) | ((yy == 36) & (xx <= 42))
                       | ((yy >= 37) & (yy <= 49) & (xx <= 36)) | ((yy >= 50) & (xx <= 27)))
        arm_l = a & ~antenna & (yy >= 56) & (xx >= 28) & (xx <= 45)
        arm_r = a & ~antenna & (yy >= 56) & (xx >= 57)
        head = a & ~antenna & (yy <= 54)
    torso = a & ~antenna & ~arm_l & ~arm_r & ~head
    parts = {"head": head, "antenna": antenna, "arm_l": arm_l, "arm_r": arm_r, "torso": torso}
    parts = absorb_strays(parts, ("antenna", "arm_l", "arm_r", "torso", "head"))
    total = sum(int(m.sum()) for m in parts.values())
    n_exp = FRONT_N if kind == "front" else BACK_N
    if total != n_exp:
        raise SystemExit(f"{kind} parts overlap or miss pixels: {total}")
    for name, m in parts.items():
        sz = component_sizes(m)
        if len(sz) != 1:
            raise SystemExit(f"{kind} {name} is not one piece: {sz[:5]}")
    hys, hxs = np.nonzero(parts["head"])
    head_c = (float(hxs.mean()), float(hys.mean()))
    hx0, hx1 = int(hxs.min()), int(hxs.max())
    mouth = {0: [], 1: [], 2: []}
    if kind == "front":
        wm = is_col(img, a, (C_W,)) & parts["head"]
        wys, wxs = np.nonzero(wm)
        if len(wxs) < 2:
            raise SystemExit("front eye glints not found")
        eye_cx = float(wxs.mean())
        if not eye_cx < (hx0 + hx1) / 2.0 - 0.5:
            raise SystemExit(f"front no longer faces left: eyes {eye_cx:.1f} head {(hx0 + hx1) / 2:.1f}")
        # the existing mouth: the 2px dark line between the eyes
        for (x, y) in ((38, 51), (39, 51)):
            if tuple(int(v) for v in img[y, x, :3]) != C_O:
                raise SystemExit("front mouth line moved; remeasure")
        mouth[1] = [(38, 51, C_K), (39, 51, C_K), (38, 52, C_K), (39, 52, C_K)]
        mouth[2] = [(37, 51, C_K), (38, 51, C_K), (39, 51, C_K), (40, 51, C_K),
                    (37, 52, C_K), (38, 52, C_P), (39, 52, C_P), (40, 52, C_K),
                    (38, 53, C_K), (39, 53, C_K)]
        mouth_c = (38.5, 52.0)
    else:
        if int((is_col(img, a, (C_W, C_D)) & parts["head"]).sum()):
            raise SystemExit("back head shows face pixels; remeasure")
        mouth_c = (head_c[0], float(hys.min()) + 2)
    pivots = {
        "antenna": contact_pivot(parts["antenna"], parts["head"]),
        "arm_l": contact_pivot(parts["arm_l"], parts["torso"]),
        "arm_r": contact_pivot(parts["arm_r"], parts["torso"]),
    }
    tips = {}
    for nm in ROT_PARTS:
        ys, xs = np.nonzero(parts[nm])
        px, py = pivots[nm]
        k = int(np.argmax((xs - px) ** 2 + (ys - py) ** 2))
        tips[nm] = (float(xs[k]), float(ys[k]))
    info = {k: int(v.sum()) for k, v in parts.items()}
    info.update(head_c=[round(head_c[0], 1), round(head_c[1], 1)],
                pivots={k: [round(v[0], 1), round(v[1], 1)] for k, v in pivots.items()},
                tips={k: [round(v[0], 1), round(v[1], 1)] for k, v in tips.items()})
    return {"parts": parts, "pivots": pivots, "tips": tips, "mouth": mouth,
            "mouth_c": mouth_c, "head_c": head_c}, info


# ---------------------------------------------------------------- rig
def build_rig(img, seg, flip, kind):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    ox, oy = dst_x - x0, dst_y - y0
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    canvas[..., 3] = np.where(canvas[..., 3] >= 128, 255, 0).astype(np.uint8)
    canvas[canvas[..., 3] == 0] = 0

    def place(m):
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        return pm

    def pt(p):
        return (p[0] + ox, p[1] + oy)

    placed = {k: place(m) for k, m in seg["parts"].items()}
    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr
    # the mouth stages live on head variants (palette colours only)
    heads = {0: layers["head"]}
    for st in (1, 2):
        h = layers["head"].copy()
        for x, y, col in seg["mouth"][st]:
            h[y + oy, x + ox, :3] = col
            h[y + oy, x + ox, 3] = 255
        heads[st] = h
    # Joint cushions under every moving part: at rest the part covers them;
    # when it turns, the seam shows its parent's colour.
    under = np.zeros_like(canvas)
    for part, parent, rad in (("antenna", "head", 2), ("arm_l", "torso", 2), ("arm_r", "torso", 2)):
        zone = placed[part] & dilate(placed[parent], rad)
        paste(under, nearest_color(canvas, zone, placed[parent]))
    layers["under"] = under

    pv = {k: pt(v) for k, v in seg["pivots"].items()}
    tips = {k: pt(v) for k, v in seg["tips"].items()}
    head_c = pt(seg["head_c"])
    mouth_c = pt(seg["mouth_c"])
    # front.png faces screen-left, back.png faces up-right
    face = -1.0 if kind == "front" else 1.0
    if flip:
        def mx(p):
            return (CELL_W - 1 - p[0], p[1])
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        heads = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in heads.items()}
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        pv = {k: mx(v) for k, v in pv.items()}
        tips = {k: mx(v) for k, v in tips.items()}
        head_c, mouth_c = mx(head_c), mx(mouth_c)
        face = -face
    oys, oxs = np.nonzero(canvas[..., 3] >= 128)
    center = (float(oxs.mean()), float(oys.mean()))
    # raising a part: tip left of its joint raises clockwise (+), tip right
    # of its joint counter-clockwise (-), on screen.
    sign = {nm: (1.0 if tips[nm][0] < pv[nm][0] else -1.0) for nm in ROT_PARTS}
    # the flipper reaching toward the face thrusts; the other strokes
    fwd = "arm_l" if face * (tips["arm_l"][0] - tips["arm_r"][0]) > 0 else "arm_r"
    hb = body_box(layers["head"])
    return {
        "kind": kind, "layers": layers, "heads": heads, "pv": pv, "tips": tips,
        "center": center, "head_c": head_c, "mouth_c": mouth_c, "face": face,
        "head_box": hb, "rest": canvas, "sign": sign, "fwd": fwd,
        "has_mouth": bool(seg["mouth"][1]),
        "order": ["under", "antenna", "torso", "arm_r", "arm_l", "head"],
        "body_n": int((canvas[..., 3] >= 128).sum()),
    }


# ---------------------------------------------------------------- motion
def wave(f, n, amp, phase=0.0, cycles=1.0):
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


ANGLE_KEYS = ("antenna", "arm_f", "arm_b")


def pose(action, f):
    """antenna / arm_f (face-side flipper) / arm_b (far flipper): degrees the
    part is RAISED (sign applied per rig). For the antenna, raised = streams
    back and up. bob: integer dy (negative = up). sway: integer dx toward the
    face. knock: integer dx away from the face. mouth: 0 shut, 1 open, 2 wide.
    """
    n = FRAMES[action]
    p = dict(
        antenna=0.0, arm_f=0.0, arm_b=0.0, bob=0, sway=0, knock=0, mouth=0,
        view="own", narrow=1.0, flash=0, rings=-1, swirl=-1, lines=0, pulse=-1,
        burst=0, drops=0, twinkle=0,
    )
    if action == "idle":
        p["bob"] = -int(round(wave(f, n, 1.0, phase=-math.pi / 2)))  # 0..2 up
        p["antenna"] = wave(f, n, 6.0, phase=0.4)
        p["arm_f"] = wave(f, n, 10.0, phase=1.0)
        p["arm_b"] = wave(f, n, 10.0, phase=1.6)
        p["twinkle"] = 1 if f in (5, 6, 12, 13) else 0
    elif action == "walk":
        p["bob"] = -int(round(wave(f, n, 1.0, phase=-math.pi / 2, cycles=2.0)))
        # breaststroke: both flippers sweep twice per loop, the far one lags
        p["arm_f"] = wave(f, n, 20.0, cycles=2.0)
        p["arm_b"] = wave(f, n, 20.0, phase=0.5, cycles=2.0)
        p["antenna"] = 5.0 * (1 - math.cos(2 * math.pi * f / n)) + wave(f, n, 4.0, phase=1.0, cycles=2.0)
        p["sway"] = int(round(wave(f, n, 1.0)))
        p["lines"] = 1 if f % 4 in (1, 2) else 0
    elif action == "attack":
        ant = [0, 8, 16, 22, 24, -12, -16, -16, -16, -16, -16, -12, -7, -3, -1, 0]
        arf = [0, 8, 16, 22, 24, 22, 28, 28, 28, 28, 28, 22, 13, 6, 2, 0]
        arb = [0, 8, 16, 22, 24, -18, -26, -26, -26, -26, -26, -20, -12, -6, -2, 0]
        mouth = [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0]
        bob = [0, 0, -1, -2, -2, -3, -3, -2, 0, 1, 1, 0, 0, 0, 0, 0]
        sway = [0, 0, 0, -1, -1, 0, 0, 0, 2, 3, 3, 2, 1, 0, 0, 0]
        p.update(antenna=ant[f], arm_f=arf[f], arm_b=arb[f], mouth=mouth[f], bob=bob[f], sway=sway[f])
        if 1 <= f <= 4:
            p["rings"] = f - 1
        if 5 <= f <= 10:
            p["pulse"] = min(f - 5, 2)
        if f in (11, 12):
            p["burst"] = f - 10
    else:
        env = [0, 0.5, 0.85, 1.0, 0.9, 0.7, 0.48, 0.3, 0.16, 0.06, 0.0, 0][f]
        p["knock"] = int(round(4 * env))
        p["bob"] = -int(round(2 * env))
        p["flash"] = 1 if f == 1 else 0
        p["drops"] = 1 if 2 <= f <= 6 else 0
        p["antenna"] = 22.0 * env
        p["arm_f"] = 24.0 * env
        p["arm_b"] = 24.0 * env
        p["mouth"] = 1 if 2 <= f <= 5 else 0
    return p


# ---------------------------------------------------------------- fx
def put(out, x, y, col, over=False):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and (over or out[y, x, 3] < 128):
        out[y, x, :3] = col
        out[y, x, 3] = 255


def star(out, x, y, size, core=C_W, arm=C_L, over=True):
    put(out, x, y, core, over)
    for k in range(1, size + 1):
        c = core if k == 1 else arm
        for dx, dy in ((k, 0), (-k, 0), (0, k), (0, -k)):
            put(out, x + dx, y + dy, c, over)


def bubble(out, x, y, r=1, col=C_L):
    if r <= 1:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            put(out, x + dx, y + dy, col)
        return
    for deg in range(0, 360, 30):
        ang = math.radians(deg)
        put(out, x + r * math.cos(ang), y + r * math.sin(ang), col)
    put(out, x - 1, y - 1, C_W)


def water_rings(out, origin, face, age):
    """Expanding water shells from the mouth, wider on the facing side."""
    ox, oy = origin
    cols = (C_W, C_WL, C_L)
    for r_i in range(2):
        rad = 12 + age * 6 + r_i * 6
        col = cols[(age + r_i) % 3]
        for deg in range(0, 360, 5):
            ang = math.radians(deg)
            cx = math.cos(ang)
            k = 1.0 if cx * face >= 0 else 0.7
            put(out, ox + k * rad * cx, oy + 0.62 * rad * math.sin(ang), col)
    for k in range(3):
        ang = math.radians(-50 + 50 * k + age * 17)
        r = 20 + age * 6
        bubble(out, ox + face * r * math.cos(ang), oy + r * math.sin(ang) * 0.7, 1 + (k % 2), C_WL)


def whirlpool(out, center, stage):
    cx, cy = center
    for arm in range(3):
        base = arm * 2 * math.pi / 3 + stage * 1.1
        for t in range(18):
            ang = base + t * 0.16
            r = 36 - t * 0.9
            col = C_W if t < 3 else (C_WL if t < 8 else (C_L if t < 13 else C_A))
            put(out, cx + r * math.cos(ang), cy + 0.5 * r * math.sin(ang), col)


def speed_lines(out, face, seed):
    bb = body_box(out)
    if not bb:
        return
    x0, y0, x1, y1 = bb
    rng = np.random.default_rng(seed)
    base = x0 - 3 if face > 0 else x1 + 3
    side = -1 if face > 0 else 1
    for _ in range(3):
        y = int(rng.integers(y0 + (y1 - y0) // 4, y1 - 4))
        leng = int(rng.integers(4, 9))
        for i in range(leng):
            put(out, base + side * i, y, C_W if i < 2 else C_L)
    for k in range(2):
        y = int(rng.integers(y0 + 4, y1 - 2))
        bubble(out, base + side * int(rng.integers(6, 14)), y, 1 + k, C_WL)


def water_pulse(out, origin, face, stage):
    """A water jet from the mouth with a pulse ring riding on its front."""
    ox, oy = origin
    reach = 20 + stage * 9
    for i in range(0, reach):
        x = ox + face * i
        put(out, x, oy, C_W)
        put(out, x, oy - 1, C_WL if i > 2 else C_W)
        put(out, x, oy + 1, C_WL if i > 2 else C_W)
        if i % 3 != 1 and i > 3:
            put(out, x, oy - 2, C_L)
            put(out, x, oy + 2, C_L)
        if i % 5 == 2 and i > 6:
            put(out, x, oy - 3, C_A)
            put(out, x, oy + 3, C_A)
    hx = ox + face * reach
    rad = 5 + stage * 2
    for deg in range(0, 360, 10):
        ang = math.radians(deg)
        put(out, hx + 0.55 * rad * math.cos(ang), oy + rad * math.sin(ang),
            C_W if (deg // 10) % 2 else C_L)
    for j in range(stage + 1):
        cx = ox + face * (8 + j * 9 + stage * 2)
        r2 = 4 + j * 2
        for deg in range(-70, 71, 8):
            ang = math.radians(deg)
            put(out, cx + face * r2 * math.cos(ang) * 0.6, oy + r2 * math.sin(ang),
                C_A if j % 2 == 0 else C_WL)
    star(out, ox, oy, 2, over=False)
    return (hx, oy)


def splash(out, center, size):
    cx, cy = center
    for k in range(8):
        ang = k * math.pi / 4 + math.pi / 8
        for r in range(max(2, size // 2), size):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang),
                C_W if r < size - 2 else C_WL, over=True)
    for k in range(16):
        ang = k * math.pi / 8
        put(out, cx + (size + 2) * math.cos(ang), cy + 0.8 * (size + 2) * math.sin(ang), C_L)
    for k in range(5):
        ang = k * 2 * math.pi / 5 + 0.3
        bubble(out, cx + (size + 6) * math.cos(ang), cy + (size + 5) * math.sin(ang), 1 + (k % 2), C_A)
    star(out, cx, cy, 2)


def hurt_drops(out, f, back_dir, seed):
    bb = body_box(out)
    if not bb:
        return
    x0, y0, x1, y1 = bb
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rng = np.random.default_rng(seed)
    for k in range(4):
        a = float(rng.uniform(0, 2 * math.pi))
        r = 22 + f * 1.5
        star(out, cx + r * math.cos(a), cy + 0.8 * r * math.sin(a), 1 + (k % 2), over=False)
    for sx, sy in ((cx + back_dir * (18 + f), y0 + 3), (cx + back_dir * 24, y0 + 12)):
        put(out, sx, sy, C_W)
        put(out, sx, sy + 1, C_L)
        put(out, sx + back_dir, sy + 2, C_A)


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].astype(np.int16).sum(axis=2) < 60)
    out[a & ~dark, :3] = C_W
    return out


# ---------------------------------------------------------------- synth
def turn(rig, name, deg):
    return rotate_nearest(rig["layers"][name], rig["sign"][name] * deg, rig["pv"][name])


def arm_angles(rig, p):
    fwd = rig["fwd"]
    back = "arm_r" if fwd == "arm_l" else "arm_l"
    return {fwd: p["arm_f"], back: p["arm_b"]}


def synth_body(rig, p):
    L = rig["layers"]
    moved = {"under": L["under"], "torso": L["torso"]}
    moved["antenna"] = turn(rig, "antenna", p["antenna"])
    for nm, ang in arm_angles(rig, p).items():
        moved[nm] = turn(rig, nm, ang)
    moved["head"] = rig["heads"][p["mouth"] if rig["has_mouth"] else 0]
    body = np.zeros_like(rig["rest"])
    for name in rig["order"]:
        paste(body, moved[name])
    moving = any(abs(float(p[k])) >= 0.45 for k in ANGLE_KEYS)
    if moving:
        body = despeckle(body)
    body = shift(body, p["sway"] * rig["face"] - p["knock"] * rig["face"], p["bob"])
    return body


def synth(rigs, d, action, f, palette, with_fx=True):
    p = pose(action, f)
    own = rigs[d]
    kind, flip = VIEW[d]
    rig = own
    if p["view"] != "own":
        other = "back" if kind == "front" else "front"
        oflip = flip if p["view"] == "other" else (not flip)
        rig = rigs[(other, oflip)]
    body = synth_body(rig, p)
    if rig is not own:
        # keep the head where it was: the other view is aligned on the head
        body = shift(body, own["head_c"][0] - rig["head_c"][0], 0)
    if p["narrow"] < 0.99:
        body = narrow_x(body, p["narrow"], own["head_c"][0])
    img = snap(body, palette)
    if not with_fx:
        return img
    bob = p["bob"]
    face = own["face"]
    dx = p["sway"] * face - p["knock"] * face
    hb = own["head_box"]
    if own["has_mouth"]:
        mouth = (own["mouth_c"][0] + dx, own["mouth_c"][1] + bob)
        muzzle = ((hb[2] + 2) if face > 0 else (hb[0] - 2), own["mouth_c"][1] + bob)
        muzzle = (muzzle[0] + dx, muzzle[1])
    else:
        mouth = (own["mouth_c"][0] + dx, own["mouth_c"][1] + bob)
        muzzle = (((hb[2] + 2) if face > 0 else (hb[0] - 2)) + dx, own["head_c"][1] + bob)
    if p["twinkle"]:
        t = content_point(*own["tips"]["antenna"], own["sign"]["antenna"] * p["antenna"], own["pv"]["antenna"])
        star(img, t[0] + dx, t[1] + bob - 3, 2 if f % 2 else 1, over=False)
    if p["rings"] >= 0:
        water_rings(img, mouth, face, p["rings"])
        star(img, mouth[0], mouth[1] - (0 if own["has_mouth"] else 2), 1 + (p["rings"] % 2), over=False)
    if p["swirl"] >= 0:
        whirlpool(img, (own["head_c"][0], own["center"][1] + bob), p["swirl"])
    if p["lines"]:
        speed_lines(img, face, seed=f * 13 + 5)
    if p["pulse"] >= 0:
        water_pulse(img, muzzle, face, p["pulse"])
    if p["burst"]:
        reach = 20 + 2 * 9
        splash(img, (muzzle[0] + face * (reach - 4 + 2 * p["burst"]), muzzle[1]), 6 + 3 * p["burst"])
    if p["drops"]:
        hurt_drops(img, f, -face, seed=90 + f)
    if p["flash"]:
        img = flash_white(img)
    return snap(img, palette)

# ---------------------------------------------------------------- outputs
def params_hash():
    src = (Path(__file__).read_bytes() + (SRC_DIR / "front.png").read_bytes()
           + (SRC_DIR / "back.png").read_bytes())
    return hashlib.sha1(src).hexdigest()[:12]


def assemble(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png").convert("RGBA"))
                sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = tile
        sheets[act] = sh
    return sheets


def write_outputs(sheets):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for act, sh in sheets.items():
        im = Image.fromarray(sh, "RGBA")
        im.save(OUT_DIR / f"{act}.png")
        im.save(OUT_DIR / f"{act}_8dir.png")
        Image.fromarray(sh[:CELL_H], "RGBA").save(OUT_DIR / f"{act}_S_strip.png")
    master = np.concatenate([sheets[a] for a in ACTIONS], axis=0)
    Image.fromarray(master, "RGBA").save(OUT_DIR / "sheet.png")
    Image.fromarray(master, "RGBA").save(OUT_DIR / "all_8dir_actions.png")
    return master


def preview_image(sheets):
    """All 8 directions: idle 0, walk 4, attack 3/5/6/9/11, hurt 3."""
    picks = [("idle", 0), ("walk", 4), ("attack", 3), ("attack", 5), ("attack", 6),
             ("attack", 9), ("attack", 11), ("hurt", 3)]
    crop = (CELL_W // 2 - 80, CELL_H - 120, CELL_W // 2 + 80, CELL_H - 8)
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    bg = np.full((8 * ch, len(picks) * cw, 4), (15, 20, 31, 255), np.uint8)
    for di in range(8):
        for pi, (act, f) in enumerate(picks):
            tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            t = tile[crop[1]:crop[3], crop[0]:crop[2]]
            m = t[..., 3] >= 128
            bg[di * ch:(di + 1) * ch, pi * cw:(pi + 1) * cw][m] = t[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets, row="S", scale=2):
    ri = DIRS.index(row)
    crop = (CELL_W // 2 - 80, CELL_H - 116, CELL_W // 2 + 80, CELL_H - 12)
    cw, ch = (crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale
    cols = 8
    rows = sum(math.ceil(FRAMES[a] / cols) for a in ACTIONS)
    bg = np.full((rows * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    r = 0
    for act in ACTIONS:
        for f in range(FRAMES[act]):
            rr, cc = divmod(f, cols)
            tile = sheets[act][ri * CELL_H:(ri + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            t = tile[crop[1]:crop[3], crop[0]:crop[2]]
            t = np.repeat(np.repeat(t, scale, 0), scale, 1)
            m = t[..., 3] >= 128
            bg[(r + rr) * ch:(r + rr + 1) * ch, cc * cw:(cc + 1) * cw][m] = t[m]
        r += math.ceil(FRAMES[act] / cols)
    return Image.fromarray(bg, "RGBA")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def with_flock(lock_path, fn):
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def replace_key_value(text, prefix_hint, key, new_val):
    token = '"' + key + '":'
    start = text.find(token)
    if start < 0 or text.find(token, start + len(token)) != -1:
        raise SystemExit(f"{prefix_hint}: key {key} missing or duplicated")
    val_at = start + len(token)
    _old, end = json.JSONDecoder().raw_decode(text, val_at)
    return text[:val_at] + new_val + text[end:]


def publish_dir8(sprite_h):
    entry = {
        "dex": 489, "form": "normal", "variant": "phione", "gen": "gen4",
        "dir": "monsterSprite/gen4/0489_phione/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    new_val = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js prefix changed")
        out = replace_key_value(text, "dir8", KEY, new_val)
        before = json.loads(text[len(prefix):].strip().rstrip(";"))
        after = json.loads(out[len(prefix):].strip().rstrip(";"))
        if after[KEY] != entry or list(after) != list(before):
            raise SystemExit("dir8 edit failed")
        for k in before:
            if k != KEY and before[k] != after[k]:
                raise SystemExit(f"dir8 neighbour changed: {k}")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(out)
        tmp.replace(DIR8)

    with_flock(DIR8_LOCK, edit)


def publish_bounds(sheets):
    vals = []
    for di in range(8):
        tile = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        x0, y0, x1, y1 = body_box(tile)
        vals += [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    new_val = json.dumps(vals, separators=(",", ":"))
    text = BOUNDS.read_text()
    s_at = text.find('"s":{')
    token = '"' + KEY + '":'
    pos = text.find(token, s_at)
    if s_at < 0 or pos < 0 or text.find(token, pos + len(token)) != -1:
        raise SystemExit("sprite_bounds s key not unique")
    val_at = pos + len(token)
    _old, end = json.JSONDecoder().raw_decode(text, val_at)
    out = text[:val_at] + new_val + text[end:]
    tmp = BOUNDS.with_suffix(".js.tmp")
    tmp.write_text(out)
    tmp.replace(BOUNDS)
    return vals


def update_queue(status):
    def edit():
        raw = QUEUE.read_text(encoding="utf-8")
        i = raw.find('"id": "' + KEY + '",')
        if i < 0:
            raise SystemExit("queue item missing")
        j = raw.find('"status": "', i)
        k = raw.find('"', j + len('"status": "'))
        out = raw[:j] + '"status": "' + status + raw[k:]
        json.loads(out)
        tmp = QUEUE.with_suffix(".json.tmp")
        tmp.write_text(out, encoding="utf-8")
        tmp.replace(QUEUE)

    with_flock(QUEUE_LOCK, edit)


# ---------------------------------------------------------------- main
def load_rigs():
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    back = np.array(Image.open(SRC_DIR / "back.png").convert("RGBA"))
    for nm, im in (("front", front), ("back", back)):
        if set(np.unique(im[..., 3]).tolist()) - {0, 255}:
            raise SystemExit(f"{nm} alpha is not binary")
    palette = extract_palette(front)
    bp = {tuple(int(v) for v in c) for c in extract_palette(back)}
    fp = {tuple(int(v) for v in c) for c in palette}
    if not bp <= fp:
        raise SystemExit("back uses colours outside the front palette")
    fx = {C_W, C_WL, C_L, C_A, C_B, C_P, C_K}
    if not fx <= fp:
        raise SystemExit("fx colours are not in the front palette")
    fseg, finfo = segment(front, "front")
    bseg, binfo = segment(back, "back")
    rigs = {}
    for kind, img, seg in (("front", front, fseg), ("back", back, bseg)):
        for flip in (False, True):
            rigs[(kind, flip)] = build_rig(img, seg, flip, kind)
    for d in DIRS:
        rigs[d] = rigs[VIEW[d]]
    return front, back, palette, fseg, bseg, finfo, binfo, rigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true",
                    help="after visual approval: dir8 + sprite_bounds + queue done")
    args = ap.parse_args()

    front, back, palette, fseg, bseg, finfo, binfo, rigs = load_rigs()
    print("front parts", finfo, "back parts", binfo, "palette", len(palette), flush=True)

    for d in DIRS:
        for act in ACTIONS:
            r0 = synth(rigs, d, act, 0, palette, with_fx=False)
            if np.any(r0 != rigs[d]["rest"]):
                diff = int(np.any(r0 != rigs[d]["rest"], axis=2).sum())
                raise SystemExit(f"{act} frame 0 differs from the still ({d}): {diff}px")
        for act in ("idle", "walk"):
            n = FRAMES[act]
            end, rest = pose(act, n), pose("hurt", 0)
            if any(isinstance(end[k], (int, float)) and abs(float(end[k]) - float(rest[k])) > 1e-6
                   for k in end):
                raise SystemExit(f"{act} loop end is not the still")
    if not (rigs["E"]["face"] > 0 and rigs["W"]["face"] < 0
            and rigs["E"]["mouth_c"][0] > rigs["E"]["head_c"][0] + 1
            and rigs["W"]["mouth_c"][0] < rigs["W"]["head_c"][0] - 1):
        raise SystemExit("E must face screen-right and W screen-left")
    print("still diff 0 for all dirs; E faces right, W faces left", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    pcols = {"head": (250, 250, 250), "antenna": (250, 120, 200), "arm_l": (40, 90, 220),
             "arm_r": (40, 200, 230), "torso": (60, 200, 60)}
    for nm, seg in (("front", fseg), ("back", bseg)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, msk in seg["parts"].items():
            part[msk, :3] = pcols[name]
            part[msk, 3] = 255
        for name, (px, py) in seg["pivots"].items():
            part[int(round(py)), int(round(px))] = (255, 0, 0, 255)
        Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / f"parts_{nm}.png")

    if args.samples:
        want = [("idle", "E", f) for f in (0, 4, 8, 12)]
        want += [("walk", "E", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "E", f) for f in range(16)]
        want += [("attack", "N", f) for f in range(16)]
        want += [("hurt", "E", f) for f in (0, 1, 3, 5, 8, 11)]
        want += [("idle", d, 0) for d in DIRS]
        want += [("hurt", "N", f) for f in (3, 5)]
        tiles = []
        crop = (CELL_W // 2 - 80, CELL_H - 116, CELL_W // 2 + 80, CELL_H - 12)
        for act, d, f in want:
            bare = synth(rigs, d, act, f, palette, with_fx=False)
            tile = synth(rigs, d, act, f, palette)
            sizes = component_sizes(bare[..., 3] >= 128)
            print(f"sample {act} {d} {f:02d} comps={sizes[:4]} edge={touches_edge(tile)}", flush=True)
            tiles.append(tile[crop[1]:crop[3], crop[0]:crop[2]])
        cols = 8
        th, tw = tiles[0].shape[:2]
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * th, cols * tw, 4), (15, 20, 31, 255), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            m = t[..., 3] >= 128
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
        sheet = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        print("samples", work / "samples.png", flush=True)
        return

    frames_dir = work / "frames"
    state_p = work / "state.json"
    digest = params_hash()
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    if args.force or state.get("hash") != digest:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        state = {"hash": digest, "done": []}
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = len(DIRS) * len(ACTIONS)
    for act in ACTIONS:
        for d in DIRS:
            key = f"{act}/{d}"
            if key in state["done"]:
                continue
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.exists() and fp.stat().st_size > 0:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions_dir = SRC_DIR / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    preview.save(actions_dir / "preview.png")
    motion_contact(sheets, "E").save(OUT_DIR / "preview_motion.png")
    motion_contact(sheets, "N").save(work / "review_N.png")
    motion_contact(sheets, "W").save(work / "review_W.png")

    idle0 = sheets["idle"][:CELL_H, :CELL_W]
    bb = body_box(idle0)
    sprite_h = bb[3] - bb[1] + 1
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]).tolist())
    edge = []
    split = []
    still_diff = 0
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                if touches_edge(tile):
                    edge.append(f"{act}:{d}:{f}")
                bare = synth(rigs, d, act, f, palette, with_fx=False)
                sizes = component_sizes(bare[..., 3] >= 128)
                if len([s for s in sizes if s >= 20]) != 1:
                    split.append({"a": act, "d": d, "f": f, "sizes": sizes[:5]})
            t0 = synth(rigs, d, act, 0, palette, with_fx=False)
            still_diff += int(np.any(t0 != rigs[d]["rest"], axis=2).sum())
            if act != "attack":
                t0 = sheets[act][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
                still_diff += int(np.any(t0 != rigs[d]["rest"], axis=2).sum())
    report = {
        "version": VERSION, "hash": digest, "front_parts": finfo, "back_parts": binfo,
        "sprite_h": int(sprite_h), "off_palette": len(off), "alphas": alphas,
        "edge_touching": len(edge), "edge_list": edge[:20], "still_diff": still_diff,
        "split": split[:20], "split_n": len(split), "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_E"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "E"),
                 "--fps", "12", "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
            else:
                report[f"gait_{act}"] = "no report"
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False)[:2000], flush=True)
    meta = {
        "generator": "forge_phione_actions", "version": VERSION, "key": KEY, "dex": 489,
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(sprite_h),
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "views": {d: {"still": VIEW[d][0], "mirrored": VIEW[d][1]} for d in DIRS},
        "still_faces": "front.png faces screen-left; back.png faces up-right",
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "source": "front.png + back.png 96x96", "gifs_untouched": True,
        "roar": "the existing 2px mouth opens (dark O + pink tongue), antenna flares up, flippers rise, water rings",
        "attack": "turn-around via the real back view (no in-plane spin) + flipper thrust/stroke, antenna whip and water pulse",
        "legs": 0, "layout": "actions_8dir", "sheetDir": "agent-sprite-forge", "raw": 1,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = (report["off_palette"] or report["edge_touching"] or report["still_diff"]
              or report["split_n"] or alphas != [0, 255])
    if failed:
        print("QA FAILED", flush=True)
        return
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        publish_dir8(sprite_h)
        vals = publish_bounds(sheets)
        update_queue("done")
        print("published dir8", VERSION, "h", sprite_h, "bounds", vals[:4], flush=True)


if __name__ == "__main__":
    main()
