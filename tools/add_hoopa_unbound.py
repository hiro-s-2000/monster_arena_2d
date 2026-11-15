#!/usr/bin/env python3
"""フーパ（ときはなたれしすがた）を取り入れる。
（2026-11-15、ユーザー指示「https://pokestats.gg/sprites/hoopa-unbound が取り入れられてないので、front.gif / front-reverse.gif /
  back.gif / back-reverse.gif を Pixel, Black / White style で取り入れて 8 方向を用意」）

素材（pokestats.gg の Black / White スタイル。--src のフォルダに置く）:
  front_bw.gif  … black-white/animated/10086.gif（105x103・120 コマ。そのまま front.gif にする）
  front_bw.png  … black-white/10086.png（96x96 の静止画 → front.png）
  back_bw.png   … black-white/back/10086.png（96x96 の静止画 → back.png）
Black / White スタイルの背面には動く GIF が無い（配布元に無い）。back.gif は back_bw.png の絵だけから作る:
  体から離れて浮いている 5 本の腕を、絵を変えずに 1〜2 ドットずつ上下左右へ漂わせ、体は 1 ドット上下させる。
  拡大・回転・描き足しはしない（整数ドットの移動だけ）。0 コマ目は静止画と同じ。コマ数と表示時間は front.gif と同じ。

書き出し先: monsterSprite/gen6/0720_hoopa/form/hoopa-unbound/
  front.gif, front-reverse.gif, back.gif, back-reverse.gif, front.png, back.png
あわせて data/sprite_bounds.js の g（4 つの GIF）と data/monsterdex.js（0720_hoopa の次に 1 項目）を更新する。
8 方向シートはこのあと tools/build_dir8_from_gifs.py --gen gen6 --only 0720_hoopa-unbound --apply で作る。

使い方: <agent-sprite-forge>/.venv/bin/python tools/add_hoopa_unbound.py --src <素材フォルダ> [--apply]
注意: data/monsterdex.js は tools/build_monsterdex.js の生成物。作り直すとこの 1 項目は消えるので、このスクリプトを再実行する。
"""
import argparse, json, math, shutil, sys
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
FORM_REL = "gen6/0720_hoopa/form/hoopa-unbound"
OUT = ROOT / "monsterSprite" / FORM_REL
PAD = 3  # 腕が動くぶんの余白

ENTRY = {
    "id": "0720_hoopa-unbound", "dex": 720, "slug": "hoopa-unbound", "name": "Hoopa Unbound", "jp": "フーパ（ときはなたれしすがた）",
    "cat": "gen6", "gen": 6, "types": ["psychic", "dark"],
    "st": {"hp": 80, "atk": 160, "def": 60, "spa": 170, "spd": 130, "spe": 80},
    "h": 6.5, "w": 490, "leg": True, "dir": FORM_REL, "shiny": False,
    "moves": ["psychic", "dark-pulse", "drain-punch", "nasty-plot"],
    "pool": ["psychic", "dark-pulse", "hyperspace-hole", "knock-off", "gunk-shot", "zen-headbutt", "drain-punch", "fire-punch", "ice-punch", "thunder-punch", "throat-chop", "brick-break", "shadow-ball", "energy-ball", "thunderbolt"],
    "util": ["nasty-plot", "calm-mind", "light-screen", "protect"],
    "genus": "まじんポケモン", "shape": "arms",
    "ab": {"s": "magician", "jp": "マジシャン", "d": "技を 当てた 相手の 道具を 奪ってしまう。"},
}

def components(img):
    """8 近傍でつながった不透明ドットのかたまり。大きい順。"""
    W, H = img.size; a = img.getchannel("A").load(); seen = set(); out = []
    for y in range(H):
        for x in range(W):
            if a[x, y] and (x, y) not in seen:
                st = [(x, y)]; seen.add((x, y)); px = []
                while st:
                    cx, cy = st.pop(); px.append((cx, cy))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            q = (cx + dx, cy + dy)
                            if 0 <= q[0] < W and 0 <= q[1] < H and a[q] and q not in seen: seen.add(q); st.append(q)
                out.append(px)
    return sorted(out, key=len, reverse=True)

def layer(img, px):
    L = Image.new("RGBA", img.size, (0, 0, 0, 0)); s = img.load(); d = L.load()
    for p in px: d[p] = s[p]
    return L

def back_frames(still, n):
    comps = components(still)
    assert len(comps) == 6, len(comps)  # 体 + 浮いている腕 5 本
    body, arms = layer(still, comps[0]), [layer(still, c) for c in comps[1:]]
    # 腕ごとの漂い方: (上下の周期数, 上下の向き, 左右の周期数, 左右の向き)。周期数は整数なのでループがつながる。
    drift = [(5, 1, 2, 1), (6, -1, 3, -1), (4, 1, 2, -1), (5, -1, 3, 1), (6, 1, 2, 1)]
    W, H = still.size; out = []
    for i in range(n):
        t = 2 * math.pi * i / n
        f = Image.new("RGBA", (W + PAD * 2, H + PAD * 2), (0, 0, 0, 0))
        by = round(1.0 * math.sin(4 * t))
        f.alpha_composite(body, (PAD, PAD + by))
        for L, (cy, sy, cx, sx) in zip(arms, drift):
            f.alpha_composite(L, (PAD + sx * round(1.2 * math.sin(cx * t)), PAD + by + sy * round(2.0 * math.sin(cy * t))))
        out.append(f)
    return out

def save_gif(frames, durs, path):
    """共通パレット（透明 = 0 番）で書き出す。色は静止画の色だけ。"""
    cols = sorted({p for f in frames for p in set(f.get_flattened_data() if hasattr(f, "get_flattened_data") else f.getdata()) if p[3]})
    assert len(cols) < 255
    idx = {c: k + 1 for k, c in enumerate(cols)}
    pal = [255, 0, 255] + [v for c in cols for v in c[:3]]; pal += [0] * (768 - len(pal))
    ps = []
    for f in frames:
        p = Image.new("P", f.size, 0); p.putpalette(pal)
        p.putdata([idx[q] if q[3] else 0 for q in (f.get_flattened_data() if hasattr(f, "get_flattened_data") else f.getdata())])
        ps.append(p)
    ps[0].save(path, save_all=True, append_images=ps[1:], duration=durs, loop=0, transparency=0, disposal=2, optimize=False)

def gif_bounds(path):
    im = Image.open(path); im.seek(0); bb = im.convert("RGBA").getchannel("A").getbbox()
    return [im.size[0], im.size[1], bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(); src = Path(a.src)
    front = Image.open(src / "front_bw.gif"); n = front.n_frames
    durs = []
    for i in range(n): front.seek(i); durs.append(front.info.get("duration", 80))
    still = Image.open(src / "back_bw.png").convert("RGBA")
    still = Image.composite(still, Image.new("RGBA", still.size, (0, 0, 0, 0)), still.getchannel("A"))  # 透明ドットの色を 0 にそろえる
    frames = back_frames(still, n)
    assert frames[0].crop((PAD, PAD, PAD + still.width, PAD + still.height)).tobytes() == still.tobytes()  # 0 コマ目 = 静止画
    print("front", front.size, n, "frames,", sum(durs), "ms; back", frames[0].size)
    if not a.apply: print("dry run (use --apply)"); return
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "front_bw.gif", OUT / "front.gif")
    shutil.copyfile(src / "front_bw.png", OUT / "front.png")
    shutil.copyfile(src / "back_bw.png", OUT / "back.png")
    sys.path.insert(0, str(Path(__file__).parent)); sys.argv = sys.argv[:1]
    import update_gifs_pokestats_bw as pk  # 左右反転（パレット・表示時間を保つ）
    pk.flip_gif(str(OUT / "front.gif"), str(OUT / "front-reverse.gif")); assert pk.is_flip(str(OUT / "front.gif"), str(OUT / "front-reverse.gif"))
    save_gif(frames, durs, OUT / "back.gif")
    save_gif([ImageOps.mirror(f) for f in frames], durs, OUT / "back-reverse.gif")
    # data/sprite_bounds.js
    bp = ROOT / "data/sprite_bounds.js"; raw = bp.read_text(encoding="utf-8"); i, j = raw.index("{"), raw.rindex("}") + 1
    B = json.loads(raw[i:j])
    for v in ("front", "front-reverse", "back", "back-reverse"): B["g"][f"{FORM_REL}/{v}.gif"] = gif_bounds(OUT / f"{v}.gif")
    bp.write_text(raw[:i] + json.dumps(B, separators=(",", ":"), ensure_ascii=False) + raw[j:], encoding="utf-8")
    # data/monsterdex.js: 0720_hoopa の次（0721 の前）に入れる。すでにあれば置き換える。
    dp = ROOT / "data/monsterdex.js"; raw = dp.read_text(encoding="utf-8")
    js = json.dumps(ENTRY, separators=(",", ":"), ensure_ascii=False)
    k = raw.find('{"id":"0720_hoopa-unbound"')
    if k >= 0:
        _, ln = json.JSONDecoder().raw_decode(raw[k:]); raw = raw[:k] + js + raw[k + ln:]
    else:
        m = '{"id":"0721_volcanion"'; assert raw.count(m) == 1
        raw = raw.replace(m, js + "," + m)
    dp.write_text(raw, encoding="utf-8")
    print("written", OUT)

if __name__ == "__main__":
    main()
