#!/usr/bin/env python3
"""Porygon-Z normal (0474) 8-direction action sheet from this form's front.png only.

Measured on normal/front.png (96x96, alpha 0/255, 14 colours, 743 opaque):
  two 4-connected pieces, as drawn:
    head  x33-55 y22-43 (276 px): blue beak pointing LEFT (x33-44, y33-38),
          yellow ring eye x47-55 y33-40, red crest spike y22-28 (x47-52).
    body  x30-66 y36-73 (467 px): red/pink torso x44-58 y45-59,
          beak-side arm (blue ellipse) x30-45 y42-51,
          rear arm (blue ellipse) x54-66 y36-51,
          tail (blue ellipse) x44-56 y61-73 hanging below.
  No mouth or jaw is drawn, no legs, no wings. The head floats free of the body.
  Beak tip x33 < eye x~51, so the still faces screen-left.

Archetype: levitating virtual body (hover, legless). Parts: head (integer moves
only), crest (rotates on its base), torso, arm_fwd, arm_back, tail.

  idle   : hover bob, head floats with a lag, arms and tail sway. f0 = still.
  walk   : hover glide: two bob cycles, alternating arm strokes, tail rudder,
           head leads by 1px.
  attack : 0-4 crest flares back, arms raise, head lifts, roar rings from the
           beak. 5-10 head thrusts forward (integer px), arms strike forward,
           tail kicks back, a spinning Tri Attack triangle (red/yellow/cyan)
           flies out from the beak. 11-12 impact. 13-15 recover.
           The body never rotates in-plane (2026-11-04 rule).
  hurt   : 12 frames, white flash on f1, body tilts back (<=34 deg), head is
           knocked back separately by integer px, arms flail, sparks.

Palette-locked to front.png. Alpha is only 0 or 255. front/back png/gif untouched.
Resume: frames in agent-sprite-forge/_work/frames, hash in _work/state.json.
  --samples  write sample strips only    --force  rebuild frames
  --no-sync  do not copy to pokemon_arena_2d    --skip-qa  skip gait_loop
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen4_attack_strike  # noqa: E402

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
STILL_DIR = ROOT / "monsterSprite/gen4/0474_porygon-z/normal"
OUT_DIR = STILL_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0474_porygon-z/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces left. N NW W SW keep it; S SE E NE are mirrored to face screen-right.
UNFLIPPED = {"N", "NW", "W", "SW"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
HOVER_GAP = 12
VERSION = "porygon-z_v1"
KEY = "0474_porygon-z"
MAX_RECOIL = 34.0

INK = (16, 16, 16)
WHITE = (255, 255, 255)
CYAN = (139, 213, 255)
BLUE = (74, 156, 213)
RED = (238, 90, 98)
PINK = (255, 148, 180)
YELLOW = (255, 213, 98)
GOLD = (180, 148, 74)
MAGENTA = (148, 65, 106)

PARTS = ("torso", "arm_back", "tail", "arm_fwd", "head", "crest")

SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "all_8dir_actions.png",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)


# ---------------------------------------------------------------- utilities

def components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, bool)
    sizes = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        q = deque([(y, x)])
        seen[y, x] = True
        n = 0
        while q:
            cy, cx = q.pop()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes


def extract_palette(img):
    return np.unique(img[img[..., 3] >= 128][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    opaque = out[..., 3] >= 128
    if not np.any(opaque):
        out[...] = 0
        return out
    flat = out[..., :3].reshape(-1, 3).copy()
    sel = opaque.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    mapped = palette[np.einsum("ijk,ijk->ij", diff, diff).argmin(axis=1)]
    flat[sel] = mapped[inv.reshape(-1)]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)
    out[~opaque] = 0
    return out


def shift(layer, dx, dy):
    dx, dy = int(round(dx)), int(round(dy))
    if dx == 0 and dy == 0:
        return layer.copy()
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = layer[sy0:sy1, sx0:sx1]
    return out


def scale2x(img):
    """EPX / Scale2x on RGBA. Copies existing pixels only (no new colours)."""
    key = img.view(np.uint32).reshape(img.shape[:2]) if img.flags["C_CONTIGUOUS"] else \
        np.ascontiguousarray(img).view(np.uint32).reshape(img.shape[:2])
    p = np.pad(key, 1, mode="edge")
    B, D, F, H = p[:-2, 1:-1], p[1:-1, :-2], p[1:-1, 2:], p[2:, 1:-1]
    E = key
    e0 = np.where((D == B) & (B != F) & (D != H), D, E)
    e1 = np.where((B == F) & (B != D) & (F != H), F, E)
    e2 = np.where((D == H) & (D != B) & (H != F), D, E)
    e3 = np.where((H == F) & (D != H) & (B != F), F, E)
    h, w = key.shape
    out = np.empty((2 * h, 2 * w), np.uint32)
    out[0::2, 0::2], out[0::2, 1::2], out[1::2, 0::2], out[1::2, 1::2] = e0, e1, e2, e3
    return out.view(np.uint8).reshape(2 * h, 2 * w, 4)


def rotate_nearest(layer, angle_deg, pivot):
    """RotSprite-style rotate about pivot: Scale2x to 8x, nearest rotate, centre sample.
    Positive = clockwise on screen. Output colours are input colours only."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    scale = 8
    src = np.ascontiguousarray(layer.copy())
    src[src[..., 3] < 128] = 0
    up = scale2x(scale2x(scale2x(src)))
    px, py = pivot[0] * scale + (scale - 1) / 2.0, pivot[1] * scale + (scale - 1) / 2.0
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rad = math.radians(ang)
    c, s = math.cos(rad), math.sin(rad)
    rx, ry = xx - px, yy - py
    src_x = np.rint(px + c * rx + s * ry).astype(np.int32)
    src_y = np.rint(py - s * rx + c * ry).astype(np.int32)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    outu = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    outu[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    return outu[scale // 2::scale, scale // 2::scale].copy()


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    pad = np.pad(a, 1, constant_values=False)
    n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img):
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        pad = np.pad(a, 1, constant_values=False)
        n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        hole = (~a) & (n == 4)
        if not np.any(hole):
            break
        above = np.roll(out, 1, axis=0)
        out[hole] = above[hole]
        out[..., 3] = np.where(out[..., 3] >= 128, 255, 0).astype(np.uint8)
        out[out[..., 3] == 0] = 0
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def put(img, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        img[y, x, :3] = col
        img[y, x, 3] = 255
        return 1
    return 0


def put_free(img, x, y, col):
    """Effect pixel that never paints over the body."""
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and img[y, x, 3] < 128:
        img[y, x, :3] = col
        img[y, x, 3] = 255
        return 1
    return 0


# ---------------------------------------------------------------- rig

def segment(img):
    op = img[..., 3] >= 128
    from scipy import ndimage
    lab, n = ndimage.label(op)
    if n != 2:
        raise SystemExit(f"expected head + body pieces, got {n}")
    sizes = [(lab == i).sum() for i in (1, 2)]
    ymins = [np.nonzero(lab == i)[0].min() for i in (1, 2)]
    head_id = 1 + int(np.argmin(ymins))
    head_all = lab == head_id
    body = op & ~head_all
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    crest = head_all & (yy <= 28)
    head = head_all & ~crest
    arm_fwd = body & (xx <= 44) & (yy <= 51)
    arm_back = body & (((yy <= 50) & (xx >= 54)) | ((yy == 51) & (xx >= 54) & (xx <= 55)))
    tail = body & (yy >= 61)
    torso = body & ~arm_fwd & ~arm_back & ~tail
    masks = {"torso": torso, "arm_back": arm_back, "tail": tail,
             "arm_fwd": arm_fwd, "head": head, "crest": crest}
    cov = np.zeros_like(op)
    over = 0
    for m in masks.values():
        over += int((cov & m).sum())
        cov |= m
    if over or int(cov.sum()) != int(op.sum()):
        raise SystemExit("masks do not partition the still")
    for k, lo in (("crest", 15), ("arm_fwd", 80), ("arm_back", 70), ("tail", 60), ("torso", 120), ("head", 200)):
        if int(masks[k].sum()) < lo:
            raise SystemExit(f"{k} mask collapsed: {int(masks[k].sum())}")
    return masks, sizes


def contact_pivot(part, base):
    """Mean of part pixels 4-adjacent to base: the joint."""
    pad = np.pad(base, 1, constant_values=False)
    near = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    ys, xs = np.nonzero(part & near)
    if len(xs) == 0:
        raise SystemExit("part has no contact with its base")
    return float(xs.mean()), float(ys.mean())


def measure(img, masks):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    eye = np.all(img[..., :3] == np.array(YELLOW, np.uint8), axis=-1) & masks["head"]
    ey, ex = np.nonzero(eye)
    beak = np.zeros_like(op)
    for col in (BLUE, CYAN, WHITE):
        beak |= np.all(img[..., :3] == np.array(col, np.uint8), axis=-1)
    beak &= masks["head"] & (np.mgrid[0:96, 0:96][1] <= 45)
    by, bx = np.nonzero(beak)
    info = {
        "opaque": int(op.sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "sprite_wh": [int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        "beak_x": int(bx.min()),
        "eye_x": round(float(ex.mean()), 2),
        "parts": {k: int(v.sum()) for k, v in masks.items()},
        "feet": 0, "wings": 0, "mouth": 0,
    }
    info["facing"] = "left" if info["beak_x"] < info["eye_x"] else "right"
    if info["facing"] != "left":
        raise SystemExit(f"expected a left-facing beak: {info}")
    return info


def build_rig(img, masks, flip):
    op = img[..., 3] >= 128
    ys, xs = np.nonzero(op)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - HOVER_GAP - (sh - 1)
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    layers, bm = {}, {}
    for name, m in masks.items():
        lyr = np.zeros_like(canvas)
        sub = m[y0:y1, x0:x1]
        lyr[dst_y:dst_y + sh, dst_x:dst_x + sw][sub] = img[y0:y1, x0:x1][sub]
        layers[name] = lyr
    if flip:
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        canvas = np.ascontiguousarray(canvas[:, ::-1])
    for k, v in layers.items():
        bm[k] = v[..., 3] >= 128
    piv = {
        "arm_fwd": contact_pivot(bm["arm_fwd"], bm["torso"]),
        "arm_back": contact_pivot(bm["arm_back"], bm["torso"]),
        "tail": contact_pivot(bm["tail"], bm["torso"]),
        "crest": contact_pivot(bm["crest"], bm["head"]),
    }
    body = bm["torso"] | bm["arm_fwd"] | bm["arm_back"] | bm["tail"]
    bys, bxs = np.nonzero(bm["torso"])
    piv["body"] = (float(bxs.mean()), float(bys.mean()))
    # Joint cushion: the root pixels of each moving part (within 3px of its pivot),
    # drawn unrotated under the rotated part so a joint never opens.
    yy, xx = np.mgrid[0:CELL_H, 0:CELL_W]
    roots = {}
    for k in ("arm_fwd", "arm_back", "tail", "crest"):
        px, py = piv[k]
        r = (xx - px) ** 2 + (yy - py) ** 2 <= 2.6 ** 2
        roots[k] = np.where((bm[k] & r)[..., None], layers[k], 0).astype(np.uint8)
    facing = 1.0 if flip else -1.0  # +1: faces screen-right
    return {"layers": layers, "piv": piv, "roots": roots, "facing": facing,
            "canvas": canvas, "body_mask": body}


# ---------------------------------------------------------------- poses

def wave(amp, cycles, f, n, phase=0.0):
    """Zero at f=0 and at the loop end."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def pose(action, f):
    """Local angles: positive = raise / swing toward the rear (clockwise when facing left).
    dx positive = forward. dy positive = down. Head offsets are integers."""
    p = dict(dx=0, dy=0, hdx=0, hdy=0, crest=0.0, fwd=0.0, back=0.0, tail=0.0,
             tilt=0.0, spin=0, rings=0, ring=0, tri=-1, burst=0, sparks=0, flash=0)
    if action == "idle":
        n = 16
        p["dy"] = int(round(wave(-1.4, 1, f, n)))
        p["hdy"] = p["dy"] + int(round(wave(-1.2, 1, f, n, -math.pi / 2)))
        p["fwd"] = wave(5.0, 1, f, n, math.pi / 4)
        p["back"] = wave(-5.0, 1, f, n, math.pi / 4)
        p["tail"] = wave(6.0, 1, f, n, math.pi / 2)
        p["crest"] = wave(5.0, 2, f, n)
    elif action == "walk":
        n = 16
        p["dy"] = int(round(wave(-1.4, 2, f, n)))
        p["hdy"] = p["dy"] + int(round(wave(-1.0, 2, f, n, -math.pi / 2)))
        lead = 1 if math.sin(2 * math.pi * 2 * f / n) > 0.3 else 0
        p["hdx"] = lead
        p["fwd"] = wave(14.0, 1, f, n)
        p["back"] = wave(14.0, 1, f, n)
        p["tail"] = wave(-12.0, 1, f, n, math.pi / 2)
        p["crest"] = wave(-8.0, 2, f, n)
    elif action == "attack":
        # 0-4 charge + roar, 5-10 strike, 11-12 impact, 13-15 recover.
        charge = (0.0, 0.35, 0.7, 0.95, 1.0)
        strike = (0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 1.0, 1.0, 1.0, 1.0, 0.95, 0.85, 0.7, 0.4, 0.15, 0.0)
        if f <= 4:
            c = charge[f]
            p["crest"] = 30.0 * c
            p["fwd"] = 24.0 * c
            p["back"] = -18.0 * c
            p["tail"] = 14.0 * c
            p["hdy"] = -int(round(2 * c))
            p["hdx"] = -int(round(1 * c))
            p["dx"] = -int(round(1 * c))
            if f >= 1:
                p["rings"] = (0, 1, 2, 3, 3)[f]
                p["ring"] = f
        else:
            s = strike[f]
            p["crest"] = -10.0 * s
            p["fwd"] = -30.0 * s
            p["back"] = -32.0 * s
            p["tail"] = -24.0 * s
            p["hdx"] = int(round(4 * s))
            p["hdy"] = int(round(1 * s))
            p["dx"] = int(round(2 * s))
            if 5 <= f <= 10:
                p["tri"] = f - 5
            if f in (11, 12):
                p["burst"] = f - 10
    else:  # hurt, 12 frames
        env = math.sin(math.pi * f / 11.0)
        p["tilt"] = MAX_RECOIL * env
        p["dx"] = -int(round(3 * env))
        p["hdx"] = -int(round(5 * env))
        p["hdy"] = -int(round(3 * env))
        p["fwd"] = 22.0 * env
        p["back"] = 20.0 * env
        p["tail"] = -16.0 * env
        p["crest"] = -26.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
        if p["tilt"] > 38.0:
            raise SystemExit("recoil exceeds 38")
    return p


# ---------------------------------------------------------------- compose

def rot_part(rig, name, local):
    lyr = rig["layers"][name]
    ang = local * (-rig["facing"])  # facing left: local == screen clockwise
    out = rig["roots"][name].copy()
    paste(out, rotate_nearest(lyr, ang, rig["piv"][name]))
    return out


def compose(rig, p):
    fac = rig["facing"]
    L = rig["layers"]
    bdx = int(round(fac * p["dx"]))
    bdy = int(p["dy"])
    hdx = int(round(fac * p["hdx"]))
    hdy = int(p["hdy"])
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(body, L["torso"])
    paste(body, rot_part(rig, "arm_back", p["back"]) if abs(p["back"]) > 0.45 else L["arm_back"])
    paste(body, rot_part(rig, "tail", p["tail"]) if abs(p["tail"]) > 0.45 else L["tail"])
    paste(body, rot_part(rig, "arm_fwd", p["fwd"]) if abs(p["fwd"]) > 0.45 else L["arm_fwd"])
    if p["tilt"] > 0.45:
        body = rotate_nearest(body, p["tilt"] * (-fac), rig["piv"]["body"])
        body = fill_pinholes(body)
        body = despeckle(body)
    head = L["head"].copy()
    paste(head, rot_part(rig, "crest", p["crest"]) if abs(p["crest"]) > 0.45 else L["crest"])
    img = shift(body, bdx, bdy)
    paste(img, shift(head, hdx, hdy))
    return img, (hdx, hdy)


def beak_tip(rig, hoff):
    m = rig["layers"]["head"][..., 3] >= 128
    ys, xs = np.nonzero(m)
    if rig["facing"] > 0:
        x = int(xs.max())
    else:
        x = int(xs.min())
    y = int(np.median(ys[np.abs(xs - x) <= 1]))
    return x + hoff[0], y + hoff[1]


def roar_rings(img, origin, facing, age, strength):
    ox, oy = origin
    cols = (WHITE, CYAN, RED)
    for ri in range(strength):
        rad = 6 + age * 2 + ri * 5
        col = cols[ri % 3]
        for deg in range(-60, 61, 12):
            a = math.radians(deg)
            put_free(img, ox + facing * rad * math.cos(a), oy + rad * math.sin(a) * 0.8, col)
    return img


def line(img, a, b, col):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    for i in range(n + 1):
        t = i / max(1, n)
        put_free(img, a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, col)


def blob(img, cx, cy, col, core=WHITE):
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (2, 0), (0, 2), (1, -1), (-1, 1), (2, 1), (1, 2)):
        put(img, cx + dx, cy + dy, col)
    put(img, cx, cy, core)
    put(img, cx + 1, cy + 1, core)


def tri_attack(img, origin, facing, k):
    """Spinning triangle of three element orbs flying out of the beak."""
    ox, oy = origin
    cx = ox + facing * (9 + 6 * k)
    cy = oy + 1
    r = 6 + int(round(1.5 * k))
    rot = math.radians(50 * k)
    pts = []
    for i in range(3):
        a = rot + i * 2 * math.pi / 3
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    for i in range(3):
        line(img, pts[i], pts[(i + 1) % 3], WHITE)
    # beam trail from beak to triangle
    for t in range(0, int(abs(cx - ox)) - r, 3):
        put_free(img, ox + facing * (3 + t), oy + 1, CYAN if t % 6 else WHITE)
    for (x, y), col in zip(pts, (RED, YELLOW, CYAN)):
        blob(img, x - 1, y - 1, col)
    return img, (cx, cy)


def impact(img, centre, facing, stage):
    cx, cy = centre
    cols = (WHITE, YELLOW, RED, CYAN)
    rad = 7 + 6 * (stage - 1)
    for i, deg in enumerate(range(0, 360, 30)):
        a = math.radians(deg)
        for d in (rad, rad + 3, rad + 6):
            put_free(img, cx + d * math.cos(a), cy + d * math.sin(a), cols[(i + stage) % 4])
    blob(img, cx - 1, cy - 1, YELLOW if stage == 1 else RED)
    return img


def sparks(img, f, centre):
    cx, cy = centre
    d = f
    for sx, sy, col in ((cx - 10 - d, cy - 14, WHITE), (cx + 9 + d, cy - 10, CYAN),
                        (cx - 2, cy - 18 - d, YELLOW), (cx + 12, cy + 4 + d, WHITE)):
        put_free(img, sx, sy, col)
        put_free(img, sx + 1, sy, col)
        put_free(img, sx, sy + 1, col)
    return img


def flash(img):
    out = img.copy()
    m = (out[..., 3] >= 128) & ~np.all(out[..., :3] == np.array(INK, np.uint8), axis=-1)
    out[m, :3] = WHITE
    return out


def synth(rig, action, f, palette, effects=True):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)
    img, hoff = compose(rig, p)
    img = snap(img, palette)
    if p["flash"]:
        img = flash(img)
    if not effects:
        return img
    fac = rig["facing"]
    tip = beak_tip(rig, hoff)
    if p["rings"]:
        img = roar_rings(img, tip, fac, p["ring"], p["rings"])
    if p["tri"] >= 0:
        img, _c = tri_attack(img, tip, fac, p["tri"])
    if p["burst"]:
        _img2, c = tri_attack(img.copy(), tip, fac, 5)
        img = impact(img, (c[0] + fac * 4, c[1]), fac, p["burst"])
    if p["sparks"]:
        img = sparks(img, f, rig["piv"]["body"])
    return snap(img, palette)


# ---------------------------------------------------------------- outputs

def params_hash():
    src = (Path(__file__).read_bytes()
           + Path(__file__).with_name("gen4_attack_strike.py").read_bytes()
           + (STILL_DIR / "front.png").read_bytes())
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


def crop_tiles(tiles, pad=4):
    ys0, ys1, xs0, xs1 = CELL_H, 0, CELL_W, 0
    for t in tiles:
        ys, xs = np.nonzero(t[..., 3] >= 128)
        if len(xs):
            ys0, ys1 = min(ys0, ys.min()), max(ys1, ys.max())
            xs0, xs1 = min(xs0, xs.min()), max(xs1, xs.max())
    y0, y1 = max(0, ys0 - pad), min(CELL_H, ys1 + 1 + pad)
    x0, x1 = max(0, xs0 - pad), min(CELL_W, xs1 + 1 + pad)
    return [t[y0:y1, x0:x1] for t in tiles]


def grid(rows, scale=1, bg=(15, 20, 31, 255)):
    ch = max(c.shape[0] for r in rows for c in r)
    cw = max(c.shape[1] for r in rows for c in r)
    cols = max(len(r) for r in rows)
    out = np.full((len(rows) * ch, cols * cw, 4), bg, np.uint8)
    for ri, r in enumerate(rows):
        for ci, c in enumerate(r):
            y, x = ri * ch, ci * cw
            m = c[..., 3] >= 128
            out[y:y + c.shape[0], x:x + c.shape[1]][m] = c[m]
            out[y:y + ch, x + cw - 1] = (40, 48, 64, 255)
    im = Image.fromarray(out, "RGBA")
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    return im


def preview_image(sheets):
    rows = []
    for act in ACTIONS:
        for di in (0, 2, 4, 6):  # S E N W
            tiles = [sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                     for f in range(FRAMES[act])]
            rows.append(crop_tiles(tiles))
    return grid(rows, 2)


def motion_contact(sheets):
    rows = []
    for act in ACTIONS:
        tiles = [sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W] for f in range(FRAMES[act])]
        rows.append(crop_tiles(tiles))
    return grid(rows, 3)


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    return n


def body_pieces(tile):
    """Large components (>=12px). Expected: head and body, i.e. 1-2 (they may touch)."""
    return [s for s in components(tile[..., 3] == 255) if s >= 12]


def qa(master, sheets, rigs, palette, frames_dir, skip_gait):
    pal = {tuple(int(v) for v in c) for c in palette}
    op = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[op][:, :3], axis=0)}
    off = sorted(used - pal)
    partial = int(((master[..., 3] != 0) & (master[..., 3] != 255)).sum())
    clip = []
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                a = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] == 255
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
    # body-only checks (effects off)
    ratios, pieces = {}, {}
    base = int((rigs["S"]["canvas"][..., 3] >= 128).sum())
    for act in ACTIONS:
        ratios[act], pieces[act] = [], []
        for f in range(FRAMES[act]):
            t = synth(rigs["S"], act, f, palette, effects=False)
            ratios[act].append(round(int((t[..., 3] == 255).sum()) / base, 3))
            pieces[act].append(len(body_pieces(t)))
    rest = {d: int(np.any(synth(rigs[d], "idle", 0, palette, effects=False) != rigs[d]["canvas"], axis=2).sum())
            for d in DIRS}
    rep = {
        "version": VERSION,
        "off_palette_colors": len(off),
        "partial_alpha": partial,
        "edge_touching": clip,
        "rest_diff": rest,
        "body_ratios": ratios,
        "body_pieces": pieces,
        "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    if not skip_gait and GAIT_LOOP.exists():
        qd = OUT_DIR / "_work" / "qa"
        qd.mkdir(parents=True, exist_ok=True)
        for act in ("idle", "walk"):
            od = qd / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run([str(FORGE / ".venv/bin/python"), "-I", str(GAIT_LOOP), "select",
                                "--frames-dir", str(frames_dir / act / "S"), "--fps", "12",
                                "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                               capture_output=True, text=True)
            lr = od / "loop-report.json"
            rep[f"gait_{act}"] = (json.loads(lr.read_text()).get("recommendation") if lr.exists()
                                  else (r.stdout + r.stderr).strip()[-300:])
    ok = (not off and partial == 0 and not clip and all(v == 0 for v in rest.values())
          and all(0.88 <= r <= 1.12 for a in ACTIONS for r in ratios[a])
          and all(1 <= n <= 2 for a in ACTIONS for n in pieces[a]))
    rep["ok"] = ok
    return rep


def samples(rigs, palette):
    sd = OUT_DIR / "_work" / "samples"
    sd.mkdir(parents=True, exist_ok=True)
    for act in ACTIONS:
        rows = []
        for d in ("W", "E"):
            tiles = [synth(rigs[d], act, f, palette) for f in range(FRAMES[act])]
            c = crop_tiles(tiles)
            rows.append(c[:8])
            rows.append(c[8:])
        grid(rows, 3).save(sd / f"{act}.png")
    rows = []
    for d in DIRS:
        rows.append(crop_tiles([synth(rigs[d], "idle", 0, palette)] + [synth(rigs[d], "attack", f, palette) for f in (3, 6, 9, 12)]
                               + [synth(rigs[d], "hurt", f, palette) for f in (1, 5)]))
    grid(rows, 2).save(sd / "dirs.png")
    print("samples in", sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(STILL_DIR / "front.png").convert("RGBA"))
    if front.shape[:2] != (96, 96):
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks, _sizes = segment(front)
    info = measure(front, masks)
    print("parts", info["parts"], "palette", len(palette), "facing", info["facing"],
          "beak_x", info["beak_x"], "eye_x", info["eye_x"], flush=True)
    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}
    for d in ("W", "S"):
        diff = int(np.any(synth(rigs[d], "idle", 0, palette, effects=False) != rigs[d]["canvas"], axis=2).sum())
        if diff:
            raise SystemExit(f"frame 0 differs from the still in {d}: {diff}")
    print("rest diff 0; pivots", {k: tuple(round(v, 1) for v in val) for k, val in rigs["W"]["piv"].items()}, flush=True)
    print("hurt tilt max", round(max(pose("hurt", f)["tilt"] for f in range(12)), 2), flush=True)

    if args.samples:
        samples(rigs, palette)
        return

    work = OUT_DIR / "_work"
    frames_dir = work / "frames"
    state_p = work / "state.json"
    digest = params_hash()
    if args.force and frames_dir.exists():
        shutil.rmtree(frames_dir)
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    if state.get("hash") != digest:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        state = {"hash": digest, "done": []}
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = len(DIRS) * len(ACTIONS)
    for act in ACTIONS:
        for d in DIRS:
            key = f"{act}/{d}"
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            if key in state["done"] and all((od / f"{f:02d}.png").is_file() and (od / f"{f:02d}.png").stat().st_size > 0
                                            for f in range(FRAMES[act])):
                continue
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.is_file() and fp.stat().st_size > 0:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            if key not in state["done"]:
                state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    prev = preview_image(sheets)
    prev.save(OUT_DIR / "preview.png")
    (OUT_DIR.parent / "actions").mkdir(exist_ok=True)
    prev.save(OUT_DIR.parent / "actions" / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    s0 = sheets["idle"][:CELL_H, :CELL_W]
    ys = np.nonzero(s0[..., 3] >= 128)[0]
    sprite_h = int(ys.max() - ys.min() + 1)
    bounds = []
    for di in range(8):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        yy, xx = np.nonzero(t[..., 3] >= 128)
        bounds += [int(xx.min()), int(yy.min()), int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)]
    report = qa(master, sheets, rigs, palette, frames_dir, args.skip_qa)
    report["sprite_h"] = sprite_h
    report["bounds_s"] = bounds
    (work / "qa").mkdir(parents=True, exist_ok=True)
    (work / "qa" / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    meta = {
        "generator": "forge_porygon_z_actions", "version": VERSION, "key": KEY, "dex": 474,
        "variant": "porygon-z", "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y,
        "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "palette_colors": int(len(palette)),
        "off_palette_colors": report["off_palette_colors"], "parts": info["parts"],
        "facing": info["facing"], "beak_x": info["beak_x"], "eye_x": info["eye_x"],
        "feet": 0, "archetype": "levitating_virtual (head floats free)",
        "source": "front.png 96x96", "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k not in ("body_ratios",)}, ensure_ascii=False), flush=True)
    print("SPRITE_H", sprite_h, "BOUNDS", bounds, "OK", report["ok"], flush=True)
    if not report["ok"]:
        raise SystemExit("qa failed")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
