#!/usr/bin/env python3
"""8方向スプライト（待機アニメ）を front.gif / front-reverse.gif / back.gif / back-reverse.gif から作る。
（2026-11-15、ユーザー指示「8方向のSpriteのアニメーションを4つのGIFを基盤に作り直す。まずは gen1」）

向きの作り方はゲームの実行時ルール（game/js/dir8.js の view / frame）と同じ:
  SW = front.gif、SE = front-reverse.gif、NE = back.gif、NW = back-reverse.gif（そのまま）
  S  = front.gif の横幅 80%、N = back.gif の横幅 80%（最近傍）
  E  = front-reverse.gif を横幅 90% にして頭側へ前傾、W = front.gif を同様に逆向きへ前傾
シート: <フォーム>/<id>.png。列 = 8方向（S, SE, E, NE, N, NW, W, SW）、行 = コマ。
data/dir8.js の項目: {cw, ch, h, approx:[S,E,N,W], src:"gif", n[8], d[8][k], v}。色違いシートは作らない。

使い方（1フォームずつ独立に処理する）:
  <workspace>/agent-sprite-forge/.venv/bin/python tools/build_dir8_from_gifs.py --gen gen1            # 予定の一覧だけ
  <workspace>/agent-sprite-forge/.venv/bin/python tools/build_dir8_from_gifs.py --gen gen1 --apply    # 書き出す
  … --only 0025_pikachu  … --include-actions（actions_8dir のフォームも置き換える。既定では触らない）
"""
import argparse, json, re, shutil, sys
from pathlib import Path
from PIL import Image, ImageSequence
sys.path.insert(0, str(Path(__file__).parent))
import gen5_publish as pub

ROOT = Path(__file__).resolve().parent.parent
SPR = ROOT / "monsterSprite"
BACKUP = ROOT.parent / "_backup_dir8_before_gif_rebuild_20261115"
VIEWS = ["front", "front-reverse", "back", "back-reverse"]
MAX_FRAMES = 0  # 0 = 間引かない。[ユーザー指定 2026-11-15] 8方向は GIF と同じコマ数・同じ速さで動かす（32コマへの間引きは「鈍い」と却下）
VERSION = "bwgif2"

def load_js(path, name):
    raw = path.read_text(encoding="utf-8")
    i = raw.index("=", raw.index(name)) + 1
    return json.JSONDecoder().raw_decode(raw[i:].lstrip())[0]

def gif_frames(path):
    im = Image.open(path)
    out = []
    for f in ImageSequence.Iterator(im):
        rgba = f.convert("RGBA")
        a = rgba.getchannel("A").point(lambda v: 255 if v >= 128 else 0)  # alpha is 0 or 255
        rgba.putalpha(a)
        out.append([rgba, max(0.02, f.info.get("duration", 100) / 1000)])
    # hold frames: merge consecutive identical pictures
    merged = [out[0]]
    for img, d in out[1:]:
        if img.tobytes() == merged[-1][0].tobytes(): merged[-1][1] += d
        else: merged.append([img, d])
    # long loops: keep MAX_FRAMES evenly spaced pictures, each holding the time of the ones it stands for
    if MAX_FRAMES and len(merged) > MAX_FRAMES:
        n = len(merged); keep = []
        for k in range(MAX_FRAMES):
            a, b = round(k * n / MAX_FRAMES), round((k + 1) * n / MAX_FRAMES)
            keep.append([merged[a][0], sum(d for _, d in merged[a:b])])
        merged = keep
    return merged

def png_frames(path, flip):
    """静止画（GIF が無いフォーム用）: front.png / back.png を 1 コマとして読む。-reverse は左右反転。"""
    rgba = Image.open(path).convert("RGBA")
    rgba.putalpha(rgba.getchannel("A").point(lambda v: 255 if v >= 128 else 0))
    if flip: rgba = rgba.transpose(Image.FLIP_LEFT_RIGHT)
    return [[rgba, 1]]

def thin(frames, n):
    """n コマへ均等に間引く。残したコマは、代表するコマぶんの表示時間を持つ。"""
    if len(frames) <= n: return frames
    m = len(frames); keep = []
    for k in range(n):
        a, b = round(k * m / n), round((k + 1) * m / n)
        keep.append([frames[a][0], sum(d for _, d in frames[a:b])])
    return keep

def crop_union(frames):
    """全コマの不透明部分を合わせた範囲へ切りつめる（どのコマも同じ切り方なので動きはそのまま）。GIF の余白でシートが巨大になるのを防ぐ。"""
    box = None
    for img, _ in frames:
        b = img.getchannel("A").getbbox()
        if b: box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    if box is None: return frames
    return [[img.crop(box), d] for img, d in frames]

BUDGET_MB = 48  # 解いたときの 1 シートの上限。超えるぶんはコマを均等に間引く（大きい絵 × 数百コマの GIF で数百 MB になり、ブラウザが落ちるため）

def bbox(img):
    b = img.getchannel("A").getbbox()
    return b if b else (0, 0, img.width, img.height)

def narrow(img, k):
    return img.resize((max(1, round(img.width * k)), img.height), Image.NEAREST)

def lean(img, b0, direction):
    """Profile: 90% width, rows above the feet slide forward by whole pixels (same as Dir8.frame)."""
    nar = narrow(img, 0.9)
    bh = b0[3] - b0[1]; ay = b0[3]
    S = max(2, round(bh * 0.15))
    out = Image.new("RGBA", (nar.width + S, nar.height), (0, 0, 0, 0))
    for y in range(nar.height):
        sh = round(S * min(1, max(0, (ay - y) / max(1, bh))))
        out.paste(nar.crop((0, y, nar.width, y + 1)), (sh if direction > 0 else S - sh, y))
    return out

def build(form, apply, png=False):
    d = SPR / form["dir"]
    # png=True: GIF が無い向きだけ静止画（front.png / back.png、-reverse は反転）で埋める。GIF がある向きは GIF のまま動く。
    src = {v: gif_frames(d / f"{v}.gif") if (not png or (d / f"{v}.gif").exists()) else png_frames(d / (v.replace("-reverse", "") + ".png"), v.endswith("-reverse")) for v in VIEWS}
    src = {v: crop_union(f) for v, f in src.items()}
    cw0 = max(max(i.width for i, _ in f) for f in src.values()) + 2 + round(max(i.height for f in src.values() for i, _ in f) * 0.15)
    ch0 = max(i.height for f in src.values() for i, _ in f) + 2
    cap = max(4, int(BUDGET_MB * 1e6 / (cw0 * 8 * ch0 * 4)))
    src = {v: thin(f, cap) for v, f in src.items()}
    b0 = {v: bbox(src[v][0][0]) for v in VIEWS}
    cols = [  # S, SE, E, NE, N, NW, W, SW
        [(narrow(i, 0.8), t) for i, t in src["front"]],
        src["front-reverse"],
        [(lean(i, b0["front-reverse"], 1), t) for i, t in src["front-reverse"]],
        src["back"],
        [(narrow(i, 0.8), t) for i, t in src["back"]],
        src["back-reverse"],
        [(lean(i, b0["front"], -1), t) for i, t in src["front"]],
        src["front"],
    ]
    cw = max(i.width for c in cols for i, _ in c) + 2
    ch = max(i.height for c in cols for i, _ in c) + 2
    rows = max(len(c) for c in cols)
    sheet = Image.new("RGBA", (cw * 8, ch * rows), (0, 0, 0, 0))
    rects = []
    for ci, c in enumerate(cols):
        ox = (cw - c[0][0].width) // 2
        for k, (img, _) in enumerate(c):
            sheet.paste(img, (ci * cw + ox, k * ch + ch - 1 - img.height))
        b = bbox(sheet.crop((ci * cw, 0, ci * cw + cw, ch)))
        rects += [b[0], b[1], b[2] - b[0], b[3] - b[1]]
    fb = b0["front"]
    entry = {"cw": cw, "ch": ch, "h": fb[3] - fb[1], "approx": ["S", "E", "N", "W"], "src": "gif",
             "n": [len(c) for c in cols], "d": [[round(t, 3) for _, t in c] for c in cols], "v": VERSION}
    if png and max(entry["n"]) == 1: entry.update(src="png", raw=1, v="png8d1")  # 静止画だけから作った 8 方向（1 コマ）
    elif png: entry.update(raw=1, v="gifpng1", still=[v for v in VIEWS if not (d / f"{v}.gif").exists()])  # 一部の向きだけ静止画
    # checks: every frame inside its cell, alpha only 0/255, no empty frame
    for ci, c in enumerate(cols):
        for k in range(len(c)):
            cell = sheet.crop((ci * cw, k * ch, ci * cw + cw, k * ch + ch))
            bb = cell.getchannel("A").getbbox()
            assert bb, (form["id"], ci, k, "empty")
            assert bb[0] > 0 and bb[1] > 0 and bb[2] < cw and bb[3] < ch, (form["id"], ci, k, "touches cell edge")
    assert sum(sheet.getchannel("A").histogram()[1:255]) == 0
    # ブラウザが古いシートを使い回さないよう、版の文字に絵の中身の印を付ける（作り直すたびに URL が変わる）
    import hashlib
    entry["v"] = entry["v"].split("-")[0] + "-" + hashlib.md5(sheet.tobytes()).hexdigest()[:8]
    if apply:
        out = d / f"{form['id']}.png"
        if out.exists():
            bk = BACKUP / "monsterSprite" / form["dir"] / out.name
            if not bk.exists(): bk.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(out, bk)
        sheet.save(out)
        pub.publish_dir8(form["id"], entry)
        pub.publish_bounds(form["id"], rects)
    return entry, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True); ap.add_argument("--only", nargs="*"); ap.add_argument("--apply", action="store_true")
    ap.add_argument("--include-actions", action="store_true")
    ap.add_argument("--include-hand", action="store_true", help="手作りシートも GIF から作り直す（既定では触らない）")
    ap.add_argument("--png-fallback", action="store_true", help="GIF が欠けている向きを front.png / back.png の静止画で埋めて 8 方向を作る（dir8 の項目が無いか、静止画由来のフォームだけ）")
    a = ap.parse_args()
    dex = load_js(ROOT / "data/monsterdex.js", "window.ARENA_MONSTERDEX")
    dex = dex if isinstance(dex, list) else dex.get("monsters") or dex.get("pokemon") or list(dex.values())
    man = load_js(ROOT / "data/dir8.js", "window.ARENA_DIR8")
    forms = [p for p in dex if isinstance(p, dict) and p.get("dir", "").startswith(a.gen + "/") and (not a.only or p["id"] in a.only)]
    if a.apply:
        BACKUP.mkdir(parents=True, exist_ok=True)
        for f in ("data/dir8.js", "data/sprite_bounds.js"):
            if not (BACKUP / Path(f).name).exists(): shutil.copy2(ROOT / f, BACKUP / Path(f).name)
    log = []
    for p in forms:
        cur = man.get(p["id"]) or {}
        kind = "actions_8dir" if cur.get("layout") == "actions_8dir" else "gif" if cur.get("src") == "gif" else "hand" if cur else "none"
        miss = [v for v in VIEWS if not (SPR / p["dir"] / f"{v}.gif").exists()]
        r = {"id": p["id"], "dir": p["dir"], "before": kind}
        png = a.png_fallback and miss and (not cur or cur.get("src") == "png" or cur.get("still")) and all((SPR / p["dir"] / f"{v}.png").exists() for v in ("front", "back"))
        if png:
            try:
                e, rows = build(p, a.apply, png=True)
                r.update(result="built(png)" if a.apply else "would build(png)", cw=e["cw"], ch=e["ch"], h=e["h"], n=e["n"], rows=rows)
            except Exception as ex:
                r["result"] = "FAILED: " + repr(ex)
        elif miss: r["result"] = "skipped: no " + ", ".join(miss)
        elif kind == "actions_8dir" and not a.include_actions: r["result"] = "skipped: actions_8dir kept"
        elif kind == "hand" and cur.get("src") != "png" and not a.include_hand: r["result"] = "skipped: hand-made sheet kept"  # 手作りシート（メガゲンガーの挑発など）は置き換えない
        else:
            try:
                e, rows = build(p, a.apply)
                r.update(result="built" if a.apply else "would build", cw=e["cw"], ch=e["ch"], h=e["h"], n=e["n"], rows=rows)
            except Exception as ex:
                r["result"] = "FAILED: " + repr(ex)
        log.append(r)
    out = Path(a.only and "/dev/null" or (BACKUP / f"result_{a.gen}{'_png' if a.png_fallback else ''}.json" if a.apply else "/dev/null"))
    if a.apply and not a.only: out.write_text(json.dumps(log, ensure_ascii=False, indent=0))
    import collections
    c = collections.Counter((r["before"], r["result"].split(":")[0] if r["result"].startswith("skipped") else r["result"][:11]) for r in log)
    for k in sorted(c): print(k, c[k])
    for r in log:
        if r["result"].startswith("FAILED") or (r["result"].startswith("skipped: no") and a.only): print(r)
    if a.only:
        for r in log: print(r)

if __name__ == "__main__":
    main()
