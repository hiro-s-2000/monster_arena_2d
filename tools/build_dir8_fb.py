#!/usr/bin/env python3
"""8 方向スプライトを、front.gif / front-reverse.gif / back.gif / back-reverse.gif と「まったく同じ」絵・コマ・速さ・大きさで作る。
（2026-11-16、ユーザー指示「メガ進化やキョダイマックスの 8 方向は解像度・アニメーションの動き・大きさの質が悪過ぎる。
  4 つの GIF とまったく同じ解像度・動き・大きさになるように」）

これまでの tools/build_dir8_from_gifs.py との違い:
  - コマを間引かない（GIF の全コマ・同じ表示時間）。
  - 絵を変形しない（南・北の横幅 80%、東・西の前傾はしない）。8 方向は 4 つの GIF そのまま:
      南 S = front、南東 SE = front-reverse、東 E = front-reverse、北東 NE = back、
      北 N = back、北西 NW = back-reverse、西 W = front、南西 SW = front
  - シートは 8 列ではなく「正面のコマ」と「背面のコマ」の 2 枚だけ（<id>.png と <id>-b.png）。反転はゲームが描くときに左右反転する。
    8 列ぶん持つと 1 体で 1GB を超えるため。コマは左上から右へ、cols 個ごとに折り返して並べる（1 枚の縦横は 8192px 以内）。
  - コマの枠は全コマの絵が入る最小の箱（cw×ch）で切り出す。GIF の中での箱の位置（g）を持つので、
    debug.html の 8 方向欄は GIF の欄と同じ大きさ・同じ位置で表示できる。

data/dir8.js の項目:
  {cw, ch, h, approx: [], src: "gif", layout: "fb", n[8], d1[正面のコマごとの秒], d2[背面。d1 と同じなら省略], cols,
   g: [正面GIFの幅, 高さ, 箱の左, 箱の上, 背面GIFの幅, 高さ, 箱の左, 箱の上], v}
data/sprite_bounds.js の s[id]: 8 方向ぶんの [x, y, w, h]（箱の中での 0 コマ目の絵の範囲）。

使い方（1 体ずつ独立に実行する）:
  <agent-sprite-forge>/.venv/bin/python tools/build_dir8_fb.py --only 0006_charizard-mega-x            # 予定だけ
  <agent-sprite-forge>/.venv/bin/python tools/build_dir8_fb.py --only 0006_charizard-mega-x --apply
"""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageSequence

ROOT = Path(__file__).resolve().parent.parent
SPR = ROOT / "monsterSprite"
BACKUP = ROOT.parent / "_backup_dir8_before_fb_20261116"
MAXDIM = 8192

def load_js(path, name):
    raw = path.read_text(encoding="utf-8"); i = raw.index("=", raw.index(name)) + 1
    lead = len(raw[i:]) - len(raw[i:].lstrip())
    obj, ln = json.JSONDecoder().raw_decode(raw[i + lead:])
    return obj, raw, i + lead, i + lead + ln

def gif(path):
    im = Image.open(path); fr, du = [], []
    for f in ImageSequence.Iterator(im):
        a = np.array(f.convert("RGBA")); op = a[..., 3] >= 128
        a[..., 3] = np.where(op, 255, 0); a[~op] = 0
        fr.append(a); du.append(max(20, f.info.get("duration") or 100))
    return fr, du

def box_of(frames):
    op = np.any([f[..., 3] > 0 for f in frames], axis=0)
    ys, xs = np.where(op.any(1))[0], np.where(op.any(0))[0]
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1

def place(lo, hi, size, limit):
    """絵の範囲 [lo, hi) を含む、長さ size の箱の左上。GIF の枠 [0, limit) からはみ出さない（はみ出すなら 0 に寄せる）。"""
    o = lo - (size - (hi - lo)) // 2
    return max(0, min(o, max(0, limit - size)))

def build(form, apply):
    d = SPR / form["dir"]
    F, dF = gif(d / "front.gif"); B, dB = gif(d / "back.gif")
    # -reverse は左右反転のはず（ゲームは反転して描く）。違っていたらこの作り方は使えない
    for name, src, du in (("front-reverse.gif", F, dF), ("back-reverse.gif", B, dB)):
        R, dR = gif(d / name)
        assert len(R) == len(src) and dR == du, (form["id"], name, "frame count / durations differ")
        for k in range(len(src)): assert (R[k] == src[k][:, ::-1]).all(), (form["id"], name, k, "not a mirror")
    bF, bB = box_of(F), box_of(B)
    cw = max(bF[2] - bF[0], bB[2] - bB[0]) + 2; ch = max(bF[3] - bF[1], bB[3] - bB[1]) + 2
    gF, gB = (F[0].shape[1], F[0].shape[0]), (B[0].shape[1], B[0].shape[0])
    def crop(frames, bb, g):
        ox, oy = place(bb[0], bb[2], cw, g[0]), place(bb[1], bb[3], ch, g[1])
        out = []
        for f in frames:
            c = np.zeros((ch, cw, 4), np.uint8); part = f[oy:oy + ch, ox:ox + cw]; c[:part.shape[0], :part.shape[1]] = part
            out.append(c)
        assert all((c[..., 3] > 0).sum() == (f[..., 3] > 0).sum() for c, f in zip(out, frames)), (form["id"], "crop lost pixels")
        return out, ox, oy
    cF, oxF, oyF = crop(F, bF, gF); cB, oxB, oyB = crop(B, bB, gB)
    cols = max(1, min(max(len(cF), len(cB)), MAXDIM // cw))
    def atlas(cells):
        rows = -(-len(cells) // cols)
        assert rows * ch <= MAXDIM, (form["id"], "too many frames for one sheet", len(cells), cw, ch)
        a = np.zeros((rows * ch, cols * cw, 4), np.uint8)
        for k, c in enumerate(cells): a[(k // cols) * ch:(k // cols + 1) * ch, (k % cols) * cw:(k % cols + 1) * cw] = c
        return a
    aF, aB = atlas(cF), atlas(cB)
    def bb0(c):
        ys, xs = np.where((c[..., 3] > 0).any(1))[0], np.where((c[..., 3] > 0).any(0))[0]
        return [int(xs[0]), int(ys[0]), int(xs[-1] - xs[0] + 1), int(ys[-1] - ys[0] + 1)]
    rF, rB = bb0(cF[0]), bb0(cB[0])
    mir = lambda r: [cw - r[0] - r[2], r[1], r[2], r[3]]
    rects = rF + mir(rF) + mir(rF) + rB + rB + mir(rB) + rF + rF  # S, SE, E, NE, N, NW, W, SW
    nF, nB = len(cF), len(cB)
    entry = {"cw": cw, "ch": ch, "h": rF[3], "approx": [], "src": "gif", "layout": "fb", "n": [nF, nF, nF, nB, nB, nB, nF, nF],
             "d1": [round(x / 1000, 3) for x in dF], "cols": cols, "g": [gF[0], gF[1], oxF, oyF, gB[0], gB[1], oxB, oyB],
             "v": "fb1-" + hashlib.md5(aF.tobytes() + aB.tobytes()).hexdigest()[:8]}
    if dB != dF: entry["d2"] = [round(x / 1000, 3) for x in dB]
    info = {"id": form["id"], "cw": cw, "ch": ch, "frames": [nF, nB], "sec": [round(sum(dF) / 1000, 2), round(sum(dB) / 1000, 2)],
            "sheet": [list(aF.shape[1::-1]), list(aB.shape[1::-1])], "decodedMB": round((aF.size + aB.size) / 1e6, 1)}
    if not apply: return entry, info
    for name in (f"{form['id']}.png", f"{form['id']}-b.png"):
        src = d / name; bk = BACKUP / "monsterSprite" / form["dir"] / name
        if src.exists() and not bk.exists(): bk.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, bk)
    Image.fromarray(aF, "RGBA").save(d / f"{form['id']}.png", optimize=False, compress_level=6)
    Image.fromarray(aB, "RGBA").save(d / f"{form['id']}-b.png", optimize=False, compress_level=6)
    # 読み直して確かめる: 全コマが GIF のコマと同じ絵か
    for name, cells in ((f"{form['id']}.png", cF), (f"{form['id']}-b.png", cB)):
        a = np.array(Image.open(d / name).convert("RGBA"))
        for k, c in enumerate(cells): assert (a[(k // cols) * ch:(k // cols + 1) * ch, (k % cols) * cw:(k % cols + 1) * cw] == c).all(), (name, k)
    info["fileMB"] = round(((d / f"{form['id']}.png").stat().st_size + (d / f"{form['id']}-b.png").stat().st_size) / 1e6, 1)
    man, raw, i, j = load_js(ROOT / "data/dir8.js", "window.ARENA_DIR8")
    man[form["id"]] = entry
    (ROOT / "data/dir8.js").write_text(raw[:i] + json.dumps(man, separators=(",", ":"), ensure_ascii=False) + raw[j:], encoding="utf-8")
    bp = ROOT / "data/sprite_bounds.js"; braw = bp.read_text(encoding="utf-8"); i0, j0 = braw.index("{"), braw.rindex("}") + 1
    Bd = json.loads(braw[i0:j0]); Bd["s"][form["id"]] = rects
    bp.write_text(braw[:i0] + json.dumps(Bd, separators=(",", ":"), ensure_ascii=False) + braw[j0:], encoding="utf-8")
    return entry, info

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", required=True); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    dex, *_ = load_js(ROOT / "data/monsterdex.js", "window.ARENA_MONSTERDEX")
    form = next(p for p in dex if p["id"] == a.only)
    if a.apply:
        BACKUP.mkdir(parents=True, exist_ok=True)
        for f in ("data/dir8.js", "data/sprite_bounds.js"):
            if not (BACKUP / Path(f).name).exists(): shutil.copy2(ROOT / f, BACKUP / Path(f).name)
    try:
        entry, info = build(form, a.apply)
        info["result"] = "built" if a.apply else "would build"
    except Exception as ex:
        info = {"id": a.only, "result": "FAILED: " + repr(ex)}
    print(json.dumps(info, ensure_ascii=False))

if __name__ == "__main__":
    main()
