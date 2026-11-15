#!/usr/bin/env python3
"""メガ進化の front.gif / front-reverse.gif / back.gif / back-reverse.gif を、pokestats.gg の Generation IX のアニメで上書きする。
（2026-11-16、ユーザー指示「メガ進化のモンスターの 4 つの GIF を全て、https://pokestats.gg の Generation IX から採用してきて上書き」）

素材（1 体ごとに、その体自身の PokeAPI id だけを使う。ほかの体の絵は使わない）:
  Generation IX・Legends: Z-A のアニメ（256px、動く WebP）
    https://assets.pokestats.gg/pokeapi/v1/anim/pokemon/versions/generation-ix/legends-za/<id>.webp
    https://assets.pokestats.gg/pokeapi/v1/anim/pokemon/versions/generation-ix/legends-za/back/<id>.webp
  （99 体すべてに正面・背面がある。無いときは同じ Generation IX の champions を使う。どちらも無ければその向きは変えない）
作り方（キョダイマックスの tools/update_gifs_gen8_gmax.py と同じ考え方）:
  - ゲームは .gif を読むので、WebP の全コマを GIF にする。大きさは 256x256 のまま、コマ数もそのまま。絵は描き足さない。
  - 透明は 0 か 255（半透明のふちは 50% で切る）。色は全コマ共通の 255 色以内（コマごとに色がゆれない）。
  - 表示時間は元の 33/33/34 ms を GIF の単位（10 ms）に丸める。合計の時間がずれないよう、累積で丸める（30/30/40 ms のくり返しになる）。
  - -reverse は左右反転（パレットと表示時間はそのまま）。
  - data/sprite_bounds.js の g（4 つの GIF）を更新する。
色違いの GIF、PNG、8 方向シートは変えない（シートは tools/build_dir8_from_gifs.py で別に作り直す）。
上書き前の GIF は ../_backup_monsterSprite_gifs_before_gen9_mega_20261116/ に残す（すでにあれば上書きしない）。

使い方（1 体ずつ独立に実行する）:
  <agent-sprite-forge>/.venv/bin/python tools/update_gifs_gen9_mega.py --only 0006_charizard-mega-x            # 予定だけ
  <agent-sprite-forge>/.venv/bin/python tools/update_gifs_gen9_mega.py --only 0006_charizard-mega-x --apply
  --list で対象（dir に /mega/ をふくむ体）の id を 1 行ずつ出す。
"""
import io, json, shutil, sys, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageSequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
SPRITE = ROOT / "monsterSprite"
BACKUP = ROOT.parent / "_backup_monsterSprite_gifs_before_gen9_mega_20261116"
GAMES = ("legends-za", "champions")
URL = "https://assets.pokestats.gg/pokeapi/v1/anim/pokemon/versions/generation-ix/{game}/{tail}{id}.webp"

def load_js(path, name):
    raw = path.read_text(encoding="utf-8"); i = raw.index("=", raw.index(name)) + 1
    return json.JSONDecoder().raw_decode(raw[i:].lstrip())[0]

def forms():
    dex = load_js(ROOT / "data/monsterdex.js", "window.ARENA_MONSTERDEX")
    ids = {}
    for path in (ROOT / "monsterData").rglob("*.json"):
        if path.name == "index.json": continue
        try: d = json.loads(path.read_text())
        except Exception: continue
        if isinstance(d.get("slug"), str) and isinstance(d.get("pokemonId"), int): ids.setdefault(d["slug"], d["pokemonId"])
    return [dict(id=p["id"], slug=p["slug"], dir=p["dir"], pid=ids.get(p["slug"])) for p in dex if "/mega/" in (p.get("dir") or "")]

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as res: return res.read()

def get_anim(pid, side):
    for game in GAMES:
        url = URL.format(game=game, tail="back/" if side == "back" else "", id=pid)
        try: body = fetch(url)
        except Exception: continue
        if body[:4] == b"RIFF" and b"ANIM" in body[:128]: return game, url, body
    return None, None, None

def to_gif(body):
    im = Image.open(io.BytesIO(body)); rgba, ms = [], []
    for f in ImageSequence.Iterator(im):
        a = np.array(f.convert("RGBA")); op = a[..., 3] >= 128
        a[..., 3] = np.where(op, 255, 0); a[~op] = 0
        rgba.append(a); ms.append(f.info.get("duration") or 33)
    # 表示時間: 累積で 10 ms に丸める（合計がずれない）。GIF で確実に動く下限の 20 ms は守る
    durs, t, done = [], 0, 0
    for d in ms:
        t += d; g = max(20, round(t / 10) * 10 - done); durs.append(g); done += g
    # 全コマ共通の色
    px = np.concatenate([a[a[..., 3] > 0][:, :3] for a in rgba])
    if len(px) > 4_000_000: px = px[:: len(px) // 4_000_000 + 1]
    q = Image.fromarray(px.reshape(1, -1, 3), "RGB").quantize(255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = q.getpalette()[:255 * 3]; n = len(pal) // 3
    pimg = Image.new("P", (1, 1)); pimg.putpalette(pal + pal[-3:] * (256 - n))  # 余りは最後の色で埋める（黒で埋めると透明の番号とぶつかる）
    full = [255, 0, 255] + pal; full += [0] * (768 - len(full))
    ps = []
    for a in rgba:
        qi = np.asarray(Image.fromarray(a[..., :3], "RGB").quantize(palette=pimg, dither=Image.Dither.NONE), np.uint8)
        idx = np.where(a[..., 3] > 0, np.minimum(qi, n - 1).astype(np.int32) + 1, 0).astype(np.uint8)
        p = Image.fromarray(idx, "P"); p.putpalette(full); ps.append(p)
    return ps, durs, rgba, sum(ms)

def save(ps, durs, path):
    ps[0].save(path, save_all=True, append_images=ps[1:], duration=durs, loop=0, transparency=0, disposal=2, optimize=False)

def timeline(path):
    r = Image.open(path); out = []
    for f in ImageSequence.Iterator(r): out.append((f.info.get("duration"), np.asarray(f.convert("RGBA"))[..., 3] > 0))
    return out

def bounds(path):
    im = Image.open(path); bb = im.convert("RGBA").getchannel("A").getbbox()
    return [im.size[0], im.size[1], bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]]

def main():
    apply = "--apply" in sys.argv
    fs = forms()
    if "--list" in sys.argv:
        for f in fs: print(f["id"])
        return
    only = sys.argv[sys.argv.index("--only") + 1]
    f = next(x for x in fs if x["id"] == only)
    rep = {"id": f["id"], "pid": f["pid"], "dir": f["dir"]}
    newb = {}
    for side in ("front", "back"):
        game, url, body = get_anim(f["pid"], side) if f["pid"] else (None, None, None)
        if not body: rep[side] = "missing: unchanged"; continue
        ps, durs, rgba, src_ms = to_gif(body)
        rep[side] = dict(game=game, frames=len(ps), size=list(ps[0].size), ms=sum(durs), src_ms=src_ms)
        if not apply: continue
        d = SPRITE / f["dir"]
        for name in (f"{side}.gif", f"{side}-reverse.gif"):
            src, dst = d / name, BACKUP / f["dir"] / name
            if src.exists() and not dst.exists(): dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        save(ps, durs, d / f"{side}.gif")
        save([p.transpose(Image.FLIP_LEFT_RIGHT) for p in ps], durs, d / f"{side}-reverse.gif")
        # 読み直して確かめる: 合計時間、各時点の輪郭（同じ絵が続くコマは 1 コマにまとまることがある）、左右反転
        A, B = timeline(d / f"{side}.gif"), timeline(d / f"{side}-reverse.gif")
        assert sum(x for x, _ in A) == sum(durs) == sum(x for x, _ in B)
        assert len(A) == len(B) and all(da == db and (ma == mb[:, ::-1]).all() for (da, ma), (db, mb) in zip(A, B))
        i = 0
        for dms, m in A:
            left = dms
            while left > 0: assert (m == (rgba[i][..., 3] > 0)).all(), (side, i); left -= durs[i]; i += 1
            assert left == 0
        assert i == len(rgba)
        for name in (f"{side}.gif", f"{side}-reverse.gif"): newb[f"{f['dir']}/{name}"] = bounds(d / name)
    if apply and newb:
        bp = ROOT / "data/sprite_bounds.js"; raw = bp.read_text(encoding="utf-8"); i0, j0 = raw.index("{"), raw.rindex("}") + 1
        data = json.loads(raw[i0:j0]); data["g"].update(newb)
        bp.write_text(raw[:i0] + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + raw[j0:], encoding="utf-8")
    rep["applied"] = apply
    print(json.dumps(rep, ensure_ascii=False))

if __name__ == "__main__":
    main()
