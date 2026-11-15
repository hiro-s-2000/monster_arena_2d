#!/usr/bin/env python3
"""Mega Blaziken (0257_blaziken-mega): adds two kick attacks to the agent-sprite-forge 8-direction sheet.

  attack1 = 回し蹴り (spinning roundhouse): chamber -> full 360 deg body turn through all 8 direction rigs
            -> high roundhouse with a fire arc -> follow-through -> stance.
  attack2 = 跳び蹴り (flying jump kick): crouch -> jump with tuck -> diving kick with the body leaning
            into it and a flame streak -> impact burst on landing -> hop back to stance.

Both run 24 frames (finer than the 16-frame actions). The existing idle / walk / attack / hurt are rebuilt
unchanged by the Antigravity rig (tools/forge_blaziken_mega_actions.py, imported, never edited) on a larger
cell so the jump and the extended kick fit.

Sheet layout (data/dir8.js "acts"): idle, walk, attack, hurt, attack1, attack2 — each 8 rows (S SE E NE N NW W SW),
frames left to right. Rules followed: md/AGENTS.md locked spec 2 (monster-action-forge-architect: palette lock,
anatomical part rig, gait_loop.py, visual check of preview.png).

Resumable: finished frames are kept in agent-sprite-forge/_work/frames/<act>/<dir>/NN.png and skipped on re-run
(state hash covers this script, the v1 rig and the source PNGs). --force rebuilds everything.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_blaziken_mega_kicks.py
"""
import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("blaz_v1", HERE / "forge_blaziken_mega_actions.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

MEGA_DIR = v1.MEGA_DIR
OUT_DIR = v1.OUT_DIR
SYNC_MEGA_DIR = v1.SYNC_MEGA_DIR
SYNC_OUT_DIR = v1.SYNC_OUT_DIR
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "blaziken_mega_kicks_v1m"
DIRS = v1.DIRS
ACTS = ["idle", "walk", "attack", "hurt", "attack1", "attack2"]
NF = {"idle": 16, "walk": 16, "attack": 16, "hurt": 16, "attack1": 24, "attack2": 24}
AD = {"idle": 0.08, "walk": 0.062, "attack": 0.052, "hurt": 0.055, "attack1": 0.042, "attack2": 0.042}
CELL_W, CELL_H, FOOT_Y = 176, 136, 128

# fire colours, all from Mega Blaziken's own palette
F_RED, F_ORG, F_YEL, F_SAND = (255, 110, 69), (240, 160, 16), (246, 213, 98), (222, 164, 82)
WHITE = (222, 222, 180)
PALETTE = None   # set by build_rigs (used for the mouth's darkest colour)


# ---------------------------------------------------------------- helpers
def rotate_frame(img, ang, pivot):
    """RotSprite-like whole-frame rotation (4x nearest, rotate, sample) — keeps 1px lines readable."""
    if abs(ang) < 0.5:
        return img.copy()
    up = np.repeat(np.repeat(img, 4, axis=0), 4, axis=1)
    r = v1.rotate_layer_nearest(up, ang, (pivot[0] * 4 + 1.5, pivot[1] * 4 + 1.5))
    return r[1::4, 1::4].copy()


def despeckle(img, passes=2):
    out = img.copy()
    for _ in range(passes):
        a = out[..., 3] >= 128
        p = np.pad(a, 1)
        nb = p[:-2, 1:-1].astype(int) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        lone = a & (nb <= 1)
        if not lone.any():
            break
        out[lone] = 0
    return out


def put(out, x, y, col, under=False):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        if under and out[y, x, 3] >= 128:
            return
        out[y, x, :3] = col
        out[y, x, 3] = 255


def lowest_point(layer):
    a = layer[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return None
    y = ys.max()
    return float(xs[ys >= y - 1].mean()), float(y)


def rot_pt(p, piv, ang):
    """Same convention as v1.rotate_layer_nearest: +ang turns clockwise on screen (y down)."""
    r = math.radians(ang)
    dx, dy = p[0] - piv[0], p[1] - piv[1]
    return piv[0] + math.cos(r) * dx - math.sin(r) * dy, piv[1] + math.sin(r) * dx + math.cos(r) * dy


def fire_arc(out, piv, foot, a0, a1, width, fade=1.0):
    """Flame trail swept by the foot from angle a0 to a1 around the hip (yellow core, orange, red rim)."""
    if abs(a1 - a0) < 1:
        return
    steps = int(abs(a1 - a0) * 1.2) + 4
    for i in range(steps + 1):
        u = i / steps                       # 0 = oldest end of the trail, 1 = at the foot
        if u < 1.0 - fade:
            continue
        a = a0 + (a1 - a0) * u
        fx, fy = rot_pt(foot, piv, a)
        dx, dy = fx - piv[0], fy - piv[1]
        L = math.hypot(dx, dy) or 1.0
        nx, ny = dx / L, dy / L
        wdt = max(1.0, width * (0.35 + 0.65 * u))
        for k in np.arange(-wdt, wdt + 0.01, 0.5):
            col = F_YEL if abs(k) < wdt * 0.35 else F_ORG if abs(k) < wdt * 0.7 else F_RED
            put(out, fx + nx * k, fy + ny * k, col, under=u < 0.85)


def burst(out, c, size, phase):
    cx, cy = c
    for i in range(10):
        a = i * math.pi / 5 + phase * 0.4
        r0, r1 = size * 0.35, size
        for t in np.linspace(r0, r1, int(r1 - r0) + 2):
            col = F_YEL if t < size * 0.6 else F_ORG if t < size * 0.85 else F_RED
            put(out, cx + math.cos(a) * t, cy + math.sin(a) * t * 0.8, col)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if abs(dx) + abs(dy) <= 2:
                put(out, cx + dx, cy + dy, F_YEL)


def streaks(out, facing, box, n, seed, length):
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = box
    for _ in range(n):
        y = rng.integers(y0 + 4, max(y0 + 5, y1 - 4))
        xs = x1 + 2 if facing < 0 else x0 - 2          # trail behind the body
        L = int(rng.integers(length // 2, length + 1))
        for k in range(L):
            col = F_YEL if k < L * 0.3 else F_ORG if k < L * 0.7 else F_RED
            put(out, xs - facing * k, y, col, under=True)


def swirl(out, center, rx, ry, phase, span=2.2):
    """Fire ring around the waist during the spin (front half drawn over the body, back half under it)."""
    cx, cy = center
    for i in range(60):
        a = phase + span * i / 59
        x, y = cx + math.cos(a) * rx, cy + math.sin(a) * ry
        col = F_YEL if i > 44 else F_ORG if i > 20 else F_RED
        put(out, x, y, col, under=math.sin(a) < 0)
        put(out, x, y + 1, F_RED if i < 30 else F_ORG, under=math.sin(a) < 0)


def body_box(img):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ---------------------------------------------------------------- rig
def build_rigs():
    front = np.array(Image.open(MEGA_DIR / "front.png").convert("RGBA"))
    back = np.array(Image.open(MEGA_DIR / "back.png").convert("RGBA"))
    palette = v1.extract_palette(front, back)
    global PALETTE
    PALETTE = palette
    rigs = {}
    for d in DIRS:
        use_back = d in ("N", "NE", "NW")
        if not use_back:
            flip = d in ("SE", "E"); facing = 1.0 if flip else -1.0
        else:
            flip = d in ("NW",); facing = -1.0 if flip else 1.0
        layers, pivots = v1.build_rig_layers(back if use_back else front, is_back=use_back, flip_x=flip,
                                             cell_w=CELL_W, cell_h=CELL_H, foot_y=FOOT_Y)
        kick = "leg_R" if use_back else "leg_L"
        pivot_leg = "leg_L" if use_back else "leg_R"
        rigs[d] = dict(layers=layers, pivots=pivots, facing=facing, back=use_back, kick=kick, pivot_leg=pivot_leg,
                       foot=lowest_point(layers[kick]))
    return rigs, palette


ORDER_FRONT = ["wrist_flame_far", "arm_far", "head_plumes", "KICK", "PIVOT", "torso", "lower_beak", "head_upper",
               "wrist_flame_near", "arm_near"]
ORDER_BACK = ["wrist_flame_far", "head_plumes", "KICK", "PIVOT", "torso", "lower_beak", "head_upper",
              "wrist_flame_near", "arm_near"]


def compose(rig, p):
    """p: body(dx,dy) + per-part offsets/angles. Head and hands only move by whole pixels (face stays intact)."""
    L, P, fc = rig["layers"], rig["pivots"], rig["facing"]
    bx, by = p.get("bx", 0), p.get("by", 0)
    kick_piv = P[rig["kick"]]
    parts = []
    order = list(ORDER_BACK if rig["back"] else ORDER_FRONT)
    if p.get("ktop"):          # extended kick passes in front of the arms
        order.remove("KICK"); order.append("KICK")
    for name in order:
        if name == "KICK":
            lyr = v1.rotate_layer_nearest(L[rig["kick"]], p.get("kang", 0.0) * -fc, kick_piv)
            parts.append(v1.shift_layer_int(lyr, bx + p.get("kdx", 0), by + p.get("kdy", 0)))
        elif name == "PIVOT":
            parts.append(v1.shift_layer_int(L[rig["pivot_leg"]], p.get("pdx", bx), p.get("pdy", 0)))
        elif name == "lower_beak":   # (2026-11-04) jaw stays with the head; mouth is carved below
            parts.append(v1.shift_layer_int(L[name], bx, by))
        elif name in ("arm_near", "arm_far"):
            lyr = v1.guard_claw_outline(L[name]) if name in L else None
            if lyr is None:
                continue
            parts.append(v1.shift_layer_int(lyr, bx + p.get("adx", 0) * fc, by + p.get("ady", 0)))
        elif name in ("wrist_flame_near", "wrist_flame_far"):
            piv = P.get(name)
            lyr = v1.rotate_layer_nearest(L[name], p.get("fang", 0.0) * fc, piv) if piv else L[name]
            parts.append(v1.shift_layer_int(lyr, bx + p.get("adx", 0) * fc, by + p.get("ady", 0)))
        elif name == "head_plumes":
            parts.append(v1.shift_layer_int(L[name], bx, by - p.get("plume", 0)))
        else:
            parts.append(v1.shift_layer_int(L[name], bx, by))
    img = v1.composite_layers((CELL_H, CELL_W, 4), parts)
    if not rig["back"]:
        img = v1.carve_mouth(img, P, p.get("beak", 0), bx, by, PALETTE)
    return img


def kick_foot(rig, p):
    """Canvas point of the kicking foot for the pose (for trails / bursts)."""
    piv = rig["pivots"][rig["kick"]]
    f = rot_pt(rig["foot"], piv, p.get("kang", 0.0) * -rig["facing"])
    bx, by = p.get("bx", 0), p.get("by", 0)
    return (f[0] + bx + p.get("kdx", 0), f[1] + by + p.get("kdy", 0)), (piv[0] + bx + p.get("kdx", 0), piv[1] + by + p.get("kdy", 0))


def lerp_table(tab, f):
    return tab[min(f, len(tab) - 1)]


# ---------------------------------------------------------------- attack1: 回し蹴り
A1 = dict(
    # 0-3 chamber | 4-10 spin (rig turns 45 deg / frame) | 11-14 roundhouse | 15-17 follow | 18-23 stance
    bx=[0, -1, -2, -2, -1, 0, 1, 2, 3, 3, 4, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 0],
    by=[0, 1, 2, 3, 1, -1, -2, -3, -3, -3, -2, -2, -2, -2, -1, -1, 0, 0, 1, 1, 0, 0, 0, 0],
    kang=[0, -6, -11, -14, 12, 18, 22, 22, 22, 22, 26, 48, 72, 84, 84, 78, 62, 40, 24, 12, 5, 2, 0, 0],
    kdx=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 6, 6, 5, 3, 2, 1, 0, 0, 0, 0, 0],
    kdy=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    tilt=[0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, -4, -8, -11, -10, -8, -6, -4, -2, -1, 0, 0, 0, 0],
    beak=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],   # 2 only at the kick peak
    ady=[0, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    adx=[0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0],
    fang=[0, -6, -10, -12, 10, 14, -10, 14, -10, 14, -8, 16, 18, 12, 6, 2, -2, -4, -2, 0, 0, 0, 0, 0],
    plume=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
)

# ---------------------------------------------------------------- attack2: 跳び蹴り
A2 = dict(
    # 0-3 crouch | 4-9 jump + tuck | 10-14 dive kick | 15-16 impact | 17-23 hop back to stance
    lift=[0, 0, 0, 0, -7, -15, -21, -25, -27, -27, -24, -19, -13, -7, -1, 0, 0, -3, -5, -5, -4, -2, -1, 0],
    kdx=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 6, 6, 6, 5, 4, 2, 1, 0, 0, 0, 0, 0],
    kdy=[0, 0, 0, 0, -1, -2, -2, -2, -1, 0, 1, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    bx=[0, -1, -2, -2, -1, 0, 1, 2, 3, 4, 6, 9, 12, 14, 16, 16, 16, 14, 11, 8, 5, 3, 1, 0],
    by=[0, 1, 3, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    kang=[0, -6, -10, -12, -16, -20, -22, -22, -18, 10, 48, 66, 72, 72, 70, 66, 58, 40, 24, 12, 6, 2, 0, 0],
    pdy=[0, 0, 0, 0, -2, -4, -5, -5, -5, -4, -3, -3, -3, -2, -1, 0, 0, -1, -2, -2, -1, 0, 0, 0],
    tilt=[0, 0, 2, 3, 0, -3, -5, -6, -6, 0, 10, 18, 22, 22, 18, 12, 8, 2, -4, -6, -4, -2, 0, 0],
    beak=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],   # 2 only at the dive peak
    ady=[0, 1, 2, 2, -2, -3, -3, -3, -2, -1, 0, 0, 0, 0, 0, 1, 1, 0, -1, -1, 0, 0, 0, 0],
    adx=[0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, -2, -2, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0],
    fang=[0, -4, -8, -10, 12, 16, 12, 8, 4, 0, -12, -16, -18, -18, -16, -10, -6, 0, 6, 4, 2, 0, 0, 0],
    plume=[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
)


def synth_kick(rigs, d, act, f, palette):
    rig = rigs[d]
    fc = rig["facing"]
    if act == "attack1":
        T = A1
        spin_off = f - 3 if 4 <= f <= 10 else 0        # 1..7 -> rig of the next directions (full turn)
        use = rigs[DIRS[(DIRS.index(d) + spin_off) % 8]] if spin_off else rig
        p = {k: T[k][f] for k in ("by", "kang", "kdy", "beak", "ady", "adx", "fang", "plume")}
        p["bx"] = int(round(T["bx"][f] * fc))
        p["kdx"] = int(round(T["kdx"][f] * fc))
        p["ktop"] = 11 <= f <= 17
        p["pdx"] = p["bx"] if f >= 11 else int(round(p["bx"] * 0.5))
        p["pdy"] = 0
        img = compose(use, p)
        (foot, hip) = kick_foot(use, p)
        tilt = T["tilt"][f] * -fc
        box = body_box(img)
        if 4 <= f <= 10:   # fire swirl that rotates with the body
            cx, cy = (box[0] + box[2]) / 2, hip[1] + 2
            swirl(img, (cx, cy), 30, 8, phase=(f - 4) * 0.9, span=1.6 + 0.25 * (f - 4))
        if tilt:
            img = rotate_frame(img, tilt, ((box[0] + box[2]) / 2, FOOT_Y))
            foot = rot_pt(foot, ((box[0] + box[2]) / 2, FOOT_Y), tilt)
            hip = rot_pt(hip, ((box[0] + box[2]) / 2, FOOT_Y), tilt)
        img = v1.snap_to_palette(img, palette)
        img = despeckle(img)
        if 11 <= f <= 17:    # fire arc swept by the roundhouse
            a_now = T["kang"][f]
            a_from = {11: 22, 12: 22, 13: 30, 14: 50, 15: 70, 16: 80, 17: 86}[f]
            fade = {11: 1.0, 12: 1.0, 13: 1.0, 14: 0.8, 15: 0.55, 16: 0.35, 17: 0.18}[f]
            piv = rig["pivots"][rig["kick"]]
            base_foot = rig["foot"]
            bxo, byo = p["bx"] + p["kdx"], p["by"] + p["kdy"]
            # trail in pre-tilt coordinates, then tilt the points the same way as the body
            def tp(pt):
                q = (pt[0] + bxo, pt[1] + byo)
                return rot_pt(q, ((box[0] + box[2]) / 2, FOOT_Y), tilt) if tilt else q
            for i in range(int(abs(a_now - a_from) * 1.2) + 4):
                u = i / (int(abs(a_now - a_from) * 1.2) + 3)
                if u < 1.0 - fade:
                    continue
                a = (a_from + (a_now - a_from) * u) * -fc
                q = tp(rot_pt(base_foot, piv, a))
                hp = tp(piv)
                dx, dy = q[0] - hp[0], q[1] - hp[1]
                Ln = math.hypot(dx, dy) or 1.0
                wdt = max(1.0, 4.5 * (0.35 + 0.65 * u))
                for k in np.arange(-wdt, wdt + 0.01, 0.5):
                    col = F_YEL if abs(k) < wdt * 0.35 else F_ORG if abs(k) < wdt * 0.7 else F_RED
                    put(img, q[0] + dx / Ln * k, q[1] + dy / Ln * k, col, under=u < 0.85)
        if f in (13, 14):
            burst(img, foot, 9 if f == 13 else 12, f)
        return img

    # attack2: 跳び蹴り
    T = A2
    p = {k: T[k][f] for k in ("by", "kang", "kdy", "beak", "ady", "adx", "fang", "plume")}
    p["kdx"] = int(round(T["kdx"][f] * fc))
    p["ktop"] = 9 <= f <= 16
    lift = T["lift"][f]
    p["bx"] = int(round(T["bx"][f] * fc))
    p["by"] = p["by"] + lift
    p["pdx"] = p["bx"]
    p["pdy"] = lift + T["pdy"][f]
    img = compose(rig, p)
    (foot, hip) = kick_foot(rig, p)
    box = body_box(img)
    tilt = T["tilt"][f] * -fc
    pc = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    if tilt:
        img = rotate_frame(img, tilt, pc)
        foot = rot_pt(foot, pc, tilt)
    img = v1.snap_to_palette(img, palette)
    img = despeckle(img)
    box = body_box(img)
    if f in (4, 5, 6):         # take-off: dust kicked up from the ground
        cx = CELL_W / 2 + T["bx"][3] * fc
        rr = {4: 10, 5: 15, 6: 19}[f]
        for a in np.linspace(math.pi, 2 * math.pi, 16):
            put(img, cx + math.cos(a) * rr, FOOT_Y + 1 + math.sin(a) * 3, F_SAND, under=True)
    if 10 <= f <= 14:          # diving kick: flame streaks trailing behind the body + burning foot
        cx = (box[0] + box[2]) / 2
        streaks(img, fc, (int(cx - 12), box[1] + 18, int(cx + 12), box[3] - 6), 5, f, 14 + 3 * (f - 10))
        for r in range(4):
            for a in np.linspace(0, 2 * math.pi, 10, endpoint=False):
                put(img, foot[0] + math.cos(a) * r, foot[1] + math.sin(a) * r, F_YEL if r < 2 else F_ORG, under=r >= 2)
    if f in (14, 15, 16):
        burst(img, (foot[0] + 2 * fc, min(FOOT_Y - 2, foot[1])), {14: 8, 15: 13, 16: 10}[f], f)
    if f in (15, 16, 17):      # dust ring on the ground
        cx = foot[0]
        for a in np.linspace(math.pi, 2 * math.pi, 18):
            rr = {15: 14, 16: 20, 17: 25}[f]
            put(img, cx + math.cos(a) * rr, FOOT_Y + 2 + math.sin(a) * 3, F_SAND, under=True)
    return img


def synth(rigs, d, act, f, palette):
    if act in ("attack1", "attack2"):
        return synth_kick(rigs, d, act, f, palette)
    r = rigs[d]
    return v1.synthesize_blaziken_mega_frame(r["layers"], r["pivots"], (CELL_H, CELL_W, 4), act, f, r["facing"],
                                             r["back"], palette)


# ---------------------------------------------------------------- main
def params_hash():
    src = Path(__file__).read_bytes() + (HERE / "forge_blaziken_mega_actions.py").read_bytes()
    src += (MEGA_DIR / "front.png").read_bytes() + (MEGA_DIR / "back.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


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
    total = len(ACTS) * len(DIRS)
    for act in ACTS:
        for d in DIRS:
            key = f"{act}/{d}"
            if key in state["done"]:
                continue
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(NF[act]):
                fp = od / f"{f:02d}.png"
                if not fp.exists():
                    Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state, indent=1))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    maxf = max(NF.values())
    sheets = {}
    for act in ACTS:
        sh = np.zeros((8 * CELL_H, NF[act] * CELL_W, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(NF[act]):
                sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png"))
        sheets[act] = sh
        im = Image.fromarray(sh, "RGBA")
        im.save(OUT_DIR / f"{act}.png"); im.save(OUT_DIR / f"{act}_8dir.png")
        Image.fromarray(sh[:CELL_H], "RGBA").save(OUT_DIR / f"{act}_S_strip.png")
    master = np.zeros((8 * CELL_H * len(ACTS), maxf * CELL_W, 4), np.uint8)
    for ai, act in enumerate(ACTS):
        master[ai * 8 * CELL_H:(ai + 1) * 8 * CELL_H, :sheets[act].shape[1]] = sheets[act]
    Image.fromarray(master, "RGBA").save(OUT_DIR / "sheet.png")
    Image.fromarray(master, "RGBA").save(OUT_DIR / "all_8dir_actions.png")

    # preview.png: 4 samples per action, all 8 directions (6 actions x 4 = 24 columns)
    pv = np.full((8 * CELL_H, len(ACTS) * 4 * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTS):
        n = NF[act]
        for si, fcol in enumerate([0, n // 4, n // 2, 3 * n // 4]):
            crop = sheets[act][:, fcol * CELL_W:(fcol + 1) * CELL_W]
            m = crop[..., 3] >= 128
            pv[:, (ai * 4 + si) * CELL_W:(ai * 4 + si + 1) * CELL_W][m] = crop[m]
    Image.fromarray(pv, "RGBA").save(OUT_DIR / "preview.png")
    # every frame of the two kicks, rows S / E / N
    rev = np.full((6 * CELL_H, 24 * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(("attack1", "attack2")):
        for ri, di in enumerate((0, 2, 4)):
            src = sheets[act][di * CELL_H:(di + 1) * CELL_H]
            m = src[..., 3] >= 128
            rev[(ai * 3 + ri) * CELL_H:(ai * 3 + ri + 1) * CELL_H][m] = src[m]
    Image.fromarray(rev, "RGBA").save(work / "review_kicks_all_frames.png")

    # QC
    qa = work / "qa"; qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c[:3]) for c in palette}
    a = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[a][:, :3], axis=0)}
    off = sorted(used - pal)
    clip = {}
    for act in ACTS:
        for di, d in enumerate(DIRS):
            for f in range(NF[act]):
                c = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    report = {"version": VERSION, "hash": h, "off_palette_colors": len(off), "edge_touching_frames": clip}
    if not args.skip_qa and GAIT_LOOP.exists():
        for act, kind in (("idle", "idle"), ("walk", "gait")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run([str(FORGE / ".venv/bin/python"), "-I", str(GAIT_LOOP), "select", "--frames-dir",
                                str(frames_dir / act / "S"), "--fps", "13", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                               capture_output=True, text=True)
            rep = od / "loop-report.json"
            report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation") if rep.exists() else (r.stderr[-300:] or "no report")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "edge_touching_frames"}, ensure_ascii=False)[:600])
    print("edge-touching:", {k: len(v) for k, v in clip.items()})

    # per-direction bounds of idle frame 0 (for data/sprite_bounds.js "s", used by debug.html on file://)
    bounds = []
    for di in range(8):
        c = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W, 3] >= 128
        ys, xs = np.nonzero(c)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    sprite_h = max(bounds[3::4])
    meta = {"generator": "forge_blaziken_mega_kicks", "version": VERSION, "key": "0257_blaziken-mega", "dex": 257,
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h, "directions": DIRS,
            "actions": ACTS, "frames": [NF[a] for a in ACTS], "frame_time": [AD[a] for a in ACTS],
            "palette_colors": int(len(palette)), "off_palette_colors": len(off), "idle_bounds_S_to_SW": bounds}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_sync and SYNC_MEGA_DIR.exists():
        SYNC_OUT_DIR.mkdir(parents=True, exist_ok=True)
        for fp in OUT_DIR.iterdir():
            if fp.is_file():
                shutil.copy2(fp, SYNC_OUT_DIR / fp.name)
        print("synced ->", SYNC_OUT_DIR)


if __name__ == "__main__":
    main()
