#!/usr/bin/env python3
"""Gen-2 mega action sheets (待機・歩き・攻撃・ダメージ) animated procedurally from each form's front.png.

Image/video generation (agent-sprite-forge via Codex / Grok) was refused by moderation for these
characters (2026-11-02), so motion is made by warping the still, the same way the gen-1 8-direction
idle loops were made:
  - breathing: the body above the feet rises and the chest widens a little (feet stay planted)
  - appendage sway: thin parts far from the body core (arms, wings, tails, manes, horns) rotate
    gently about the core, so they move without a hand-made rig
  - walk: steps in place (the two halves of the legs lift and swing in turn), bob and a forward lean
  - attack: wind-up, a lunge toward the facing side, a hold and the recovery
  - hurt: a jolt away from the facing side with a squash, then the recovery
Everything is nearest-neighbour at the sprite's own pixel size (no resample, no palette change).

Output per form (same contract as tools/forge_gen2_part_actions.py):
  <form>/actions/sheet.png (rows: idle, walk, attack, hurt; columns: frames), preview.png, meta.json
  and the form's entry in data/dir8.js (layout "actions", an = frames per action, src "procedural").
Usage: python tools/forge_gen2_mega_procedural.py [--only id,id] [--dry-run <dir>]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

GAME = Path(__file__).resolve().parent.parent
GEN2 = GAME / "monsterSprite" / "gen2"
DIR8 = GAME / "data" / "dir8.js"
PAD_X, PAD_TOP, PAD_BOT = 12, 8, 2

# frames and seconds per frame for each action row
ACTIONS = [("idle", 12, 0.1), ("walk", 8, 0.09), ("attack", 12, 0.065), ("hurt", 8, 0.06)]


# ---- displacement fields --------------------------------------------------------------------------
def core_mask(a, h):
    for f in (0.16, 0.12, 0.09, 0.06, 0.04):
        r = max(2, int(h * f))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        c = ndi.binary_opening(a, structure=(xx * xx + yy * yy <= r * r))
        if c.sum() > a.sum() * 0.25:
            lab, n = ndi.label(c)
            if n > 1:
                sizes = ndi.sum(c, lab, range(1, n + 1))
                c = lab == (np.argmax(sizes) + 1)
            return c
    return a.copy()


def fields(cell):
    a = cell[..., 3] > 16
    H, W = a.shape
    ys, xs = np.nonzero(a)
    top, bot = ys.min(), ys.max()
    h = bot - top + 1
    core = core_mask(a, h)
    cy, cx = np.nonzero(core)
    ccy, ccx = cy.mean(), cx.mean()
    d = ndi.distance_transform_edt(~core)
    w = np.clip(d / max(3, 0.22 * h), 0, 1) ** 0.8
    w[~a] = 0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    foot = np.clip((bot - yy) / max(1, 0.22 * h), 0, 1)
    w = ndi.gaussian_filter(w * np.where(yy > ccy, foot, 1), 1.2)
    _, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
    py = ndi.gaussian_filter(iy.astype(np.float32), 0.06 * h)
    px = ndi.gaussian_filter(ix.astype(np.float32), 0.06 * h)
    side = np.tanh((xx - ccx) / max(2, 0.12 * h))
    ramp = np.clip((bot - yy) / h, 0, 1)                      # 0 at the feet, 1 at the top
    chest = np.exp(-((yy - ccy) / (0.35 * h)) ** 2)
    legs = np.clip((yy - (bot - 0.28 * h)) / (0.12 * h), 0, 1)  # 1 in the lowest ~quarter (legs / base)
    return dict(a=a, w=w, py=py, px=px, side=side, ramp=ramp, chest=chest, legs=legs, h=h, bot=bot, ccx=ccx, ccy=ccy, xx=xx, yy=yy)


def render(cell, F, *, rise=0.0, widen=0.0, swing=0.0, lean=0.0, shift=0.0, sy=1.0, step=None):
    """Forward displacement → inverse nearest sample. rise/lean/shift in px, swing in radians,
    sy = vertical scale about the feet, step = (phase) for the walk's leg lift."""
    xx, yy, h = F["xx"], F["yy"], F["h"]
    up = F["ramp"] * (1 - F["legs"])
    dx = (xx - F["ccx"]) * widen * F["chest"] + lean * F["ramp"] ** 1.3 + shift
    dy = -rise * up + (yy - F["bot"]) * (sy - 1) * (1 - F["legs"])
    if step is not None:  # the two halves of the base lift and swing in turn
        s = np.sin(2 * np.pi * step)
        half = F["side"]  # -1 … 1 across the body
        lift = np.maximum(0, s * half) + np.maximum(0, -s * -half)
        dy = dy - 0.035 * h * lift * F["legs"] * np.abs(half)
        dx = dx + 0.03 * h * s * half * F["legs"]
    th = swing * F["w"] * F["side"]
    rx, ry = xx - F["px"], yy - F["py"]
    dx = dx + (np.cos(th) - 1) * rx - np.sin(th) * ry
    dy = dy + np.sin(th) * rx + (np.cos(th) - 1) * ry
    H, W = xx.shape
    qx, qy = xx - dx, yy - dy
    for _ in range(2):
        ix = np.clip(np.rint(qx), 0, W - 1).astype(int); iy = np.clip(np.rint(qy), 0, H - 1).astype(int)
        qx, qy = xx - dx[iy, ix], yy - dy[iy, ix]
    ix = np.rint(qx).astype(int); iy = np.rint(qy).astype(int)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    out = np.zeros_like(cell)
    out[ok] = cell[iy[ok], ix[ok]]
    return out


def keys(t, pts):
    """piecewise smooth interpolation through (t, value) keyframes"""
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t <= t1:
            u = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            u = u * u * (3 - 2 * u)
            return v0 + (v1 - v0) * u
    return pts[-1][1]


# ---- the four actions -----------------------------------------------------------------------------
def frames_for(cell, F, facing):
    h = F["h"]; fwd = -1 if facing == "left" else 1
    rows = []
    # idle: breath + sway, frame 0 = the still
    n = ACTIONS[0][1]
    rows.append([render(cell, F, rise=max(1.5, 0.028 * h) * (0.5 - 0.5 * np.cos(2 * np.pi * k / n)),
                        widen=0.02 * (0.5 - 0.5 * np.cos(2 * np.pi * k / n)),
                        swing=np.deg2rad(6) * np.sin(2 * np.pi * k / n)) for k in range(n)])
    # walk: two steps per loop, bob on each step, forward lean, faster sway
    n = ACTIONS[1][1]
    rows.append([render(cell, F, step=k / n, rise=max(1.0, 0.022 * h) * abs(np.sin(2 * np.pi * k / n)),
                        lean=fwd * max(1.0, 0.03 * h), swing=np.deg2rad(7) * np.sin(4 * np.pi * k / n)) for k in range(n)])
    # attack: wind-up (lean back, crouch) → lunge (lean + shift forward, stretch) → hold → recover
    n = ACTIONS[2][1]
    L, S = 0.09 * h, 0.07 * h
    out = []
    for k in range(n):
        t = k / (n - 1)
        lean = keys(t, [(0, 0), (0.25, -0.5 * L), (0.42, L), (0.62, L), (1, 0)]) * fwd
        shift = keys(t, [(0, 0), (0.25, -0.2 * S), (0.42, S), (0.62, S), (1, 0)]) * fwd
        sy = keys(t, [(0, 1), (0.25, 0.94), (0.42, 1.04), (0.62, 1.02), (1, 1)])
        sw = keys(t, [(0, 0), (0.25, -8), (0.42, 10), (0.62, 6), (1, 0)])
        out.append(render(cell, F, lean=lean, shift=shift, sy=sy, swing=np.deg2rad(sw)))
    rows.append(out)
    # hurt: jolt away from the facing side, squash, shake back
    n = ACTIONS[3][1]
    out = []
    for k in range(n):
        t = k / (n - 1)
        back = keys(t, [(0, 0), (0.15, 1), (0.4, 0.6), (0.7, 0.2), (1, 0)])
        shake = (1 if k % 2 else -1) * (1 if 0.1 < t < 0.6 else 0)
        out.append(render(cell, F, shift=-fwd * (0.07 * h * back) + shake, lean=-fwd * 0.08 * h * back,
                          sy=1 - 0.06 * back, swing=np.deg2rad(-9 * back)))
    rows.append(out)
    return rows


def facing_of(img):
    """which way the front sprite looks: the head (top third) sits toward the facing side"""
    a = img[..., 3] > 16
    ys, xs = np.nonzero(a)
    top = ys < ys.min() + (ys.max() - ys.min()) / 3
    return "left" if xs[top].mean() < xs.mean() else "right"


def build(form_dir: Path, out_dir: Path):
    src = np.array(Image.open(form_dir / "front.png").convert("RGBA"))
    ys, xs = np.nonzero(src[..., 3] > 16)
    crop = src[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    bh, bw = crop.shape[:2]
    cw, ch = bw + 2 * PAD_X, bh + PAD_TOP + PAD_BOT
    cell = np.zeros((ch, cw, 4), np.uint8)
    cell[PAD_TOP:PAD_TOP + bh, PAD_X:PAD_X + bw] = crop
    F = fields(cell)
    facing = "left"  # gen-2 front sprites look toward the viewer's left (head-position guess was unreliable)
    rows = frames_for(cell, F, facing)
    ncol = max(len(r) for r in rows)
    sheet = np.zeros((ch * len(rows), cw * ncol, 4), np.uint8)
    for r, fr in enumerate(rows):
        for k, f in enumerate(fr):
            sheet[r * ch:(r + 1) * ch, k * cw:(k + 1) * cw] = f
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out_dir / "sheet.png", optimize=True)
    pv = Image.fromarray(sheet).resize((sheet.shape[1] * 2, sheet.shape[0] * 2), Image.Resampling.NEAREST)
    bg = Image.new("RGB", pv.size, (16, 20, 32)); bg.paste(pv, mask=pv.split()[-1]); bg.save(out_dir / "preview.png")
    idle_n, idle_d = ACTIONS[0][1], ACTIONS[0][2]
    meta = {"layout": "actions", "cw": cw, "ch": ch, "h": int(bh), "n": [idle_n] * 8, "d": [[idle_d] * idle_n for _ in range(8)],
            "an": [a[1] for a in ACTIONS], "ad": [a[2] for a in ACTIONS], "raw": 1, "arow": 0, "src": "procedural", "facing": facing}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", default="", help="write sheets under this folder instead of the game")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    raw = DIR8.read_text()
    at = raw.index("window.ARENA_DIR8")
    head, body = raw[:at] + "window.ARENA_DIR8", raw[at:].split("=", 1)[1]
    man = json.loads(body.strip().rstrip(";"))
    done = []
    for form in sorted(GEN2.glob("*/mega/*")):
        fid = form.parent.parent.name.split("_")[0] + "_" + form.name
        if not (form / "front.png").exists() or (only and fid not in only):
            continue
        out = Path(args.dry_run) / fid if args.dry_run else form / "actions"
        meta = build(form, out)
        if not args.dry_run:
            man[fid] = meta
        done.append((fid, meta["cw"], meta["ch"], meta["facing"]))
    if not args.dry_run:
        DIR8.write_text(head.rstrip() + " = " + json.dumps(man, separators=(",", ":")) + ";\n")
    for d in done:
        print(*d)


if __name__ == "__main__":
    main()
