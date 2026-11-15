#!/usr/bin/env python3
"""Fill sides that have neither a pokestats Black/White animated GIF nor a
Pokémon Showdown front/back GIF.

Generation IX ANIMATED on pokestats is an animated WebP, not a GIF:
  https://assets.pokestats.gg/pokeapi/v1/anim/pokemon/versions/generation-ix/scarlet-violet/<id>.webp
  .../back/<id>.webp
Those frames are written as front.gif / back.gif. -reverse.gif is a lossless flip.
Shiny GIFs, PNGs, and 8-direction sheets are not touched.

Skips authored GIFs: latias-mega, charizard-mega-y, charizard-mega-x back.

  python3 tools/update_gifs_gen9_animated.py
  python3 tools/update_gifs_gen9_animated.py --apply
"""
import io, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image, ImageSequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import update_gifs_pokestats_bw as bw
import update_gifs_showdown_fallback as sd

ROOT = sd.ROOT
SPRITE = sd.SPRITE
BACKUP = Path('/Users/honmahiroshi/Desktop/ai_workspace/_backup_monsterSprite_gifs_before_gen9_20261115')
APPLY = '--apply' in sys.argv
GEN9 = (
    'https://assets.pokestats.gg/pokeapi/v1/anim/pokemon/versions/generation-ix/scarlet-violet/{tail}{id}.webp',
    'https://assets.pokestats.gg/pokeapi/v1/anim/pokemon/versions/generation-ix/legends-za/{tail}{id}.webp',
    'https://assets.pokestats.gg/pokeapi/v1/anim/pokemon/versions/generation-ix/champions/{tail}{id}.webp',
)

def showdown_exists(spriteid, side, poke_id):
    urls = [(sd.ANI if side == 'front' else sd.ANI_BACK).format(sid=spriteid)]
    mirror = 'https://assets.pokestats.gg/pokeapi/v1/pokemon/other/showdown/'
    urls.append(mirror + ('back/' if side == 'back' else '') + f'{poke_id}.gif')
    for url in urls:
        try:
            status, ctype, _ = sd.fetch(url, 'HEAD')
        except Exception:
            continue
        if sd.is_gif(status, ctype, b'', 'HEAD'):
            return True
    return False

def durations_ms(path, n):
    text = subprocess.check_output(['webpmux', '-info', str(path)], text=True, stderr=subprocess.DEVNULL)
    durs = []
    for line in text.splitlines():
        if ':' not in line:
            continue
        head, rest = line.split(':', 1)
        if not head.strip().isdigit():
            continue
        cols = rest.split()
        if len(cols) >= 6 and cols[5].isdigit():
            durs.append(int(cols[5]))
    if len(durs) != n:
        return [33] * n
    return durs

def webp_to_gif(data):
    tmp = Path('/tmp/gen9-frame.webp')
    tmp.write_bytes(data)
    im = Image.open(io.BytesIO(data))
    frames = [fr.convert('RGBA') for fr in ImageSequence.Iterator(im)]
    durs = durations_ms(tmp, len(frames))
    buf = io.BytesIO()
    frames[0].save(buf, format='GIF', save_all=True, append_images=frames[1:],
                   duration=durs, loop=0, disposal=2, optimize=False)
    return buf.getvalue(), len(frames)

def fetch_webp(poke_id, side):
    tail = 'back/' if side == 'back' else ''
    for template in GEN9:
        url = template.format(tail=tail, id=poke_id)
        try:
            status, ctype, body = sd.fetch(url, 'GET')
        except Exception as err:
            print('fetch error', url, err, flush=True)
            continue
        if status == 200 and 'webp' in ctype.lower() and body[:4] == b'RIFF' and b'ANIM' in body[:128]:
            return body, url
    return None, None

def backup(rel):
    src = SPRITE / rel
    dst = BACKUP / rel
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        bw.shutil.copy2(src, dst)

def write_side(form, side, gif):
    rel = f"{form['dir']}/{side}.gif"
    relr = f"{form['dir']}/{side}-reverse.gif"
    loc = SPRITE / rel
    locr = SPRITE / relr
    loc.parent.mkdir(parents=True, exist_ok=True)
    actions = []
    if loc.exists():
        backup(rel)
        actions.append('replaced')
    else:
        actions.append('created')
    loc.write_bytes(gif)
    if locr.exists():
        backup(relr)
        actions.append('reverse rewritten')
    else:
        actions.append('reverse created')
    bw.flip_gif(str(loc), str(locr))
    if not bw.is_flip(str(loc), str(locr)):
        raise RuntimeError('flip mismatch ' + relr)
    return actions, rel, relr

def main():
    ps = sd.load_ps()
    forms = sd.load_forms(ps)
    print('forms', len(forms), flush=True)
    have = sd.probe(forms)
    candidates = []
    for form in forms:
        for side in ('front', 'back'):
            if have[(form['slug'], side)]:
                continue
            if form['slug'] in sd.SKIP_ALL or (form['slug'], side) in sd.SKIP_SIDE:
                continue
            if not form['spriteid']:
                continue
            candidates.append((form, side))
    print('BW missing, not authored', len(candidates), flush=True)
    work = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(showdown_exists, form['spriteid'], side, form['id']): (form, side) for form, side in candidates}
        done = 0
        for fut in as_completed(futs):
            form, side = futs[fut]
            done += 1
            if done % 100 == 0:
                print(f'showdown checked {done}/{len(candidates)}', flush=True)
            if not fut.result():
                work.append((form, side))
    work.sort(key=lambda item: (item[0]['dir'], item[1]))
    print('sides with neither BW nor Showdown', len(work), flush=True)
    report = []
    newb = {}
    got = miss = 0
    for n, (form, side) in enumerate(work, 1):
        row = {'slug': form['slug'], 'side': side, 'id': form['id'], 'dir': form['dir']}
        body, url = fetch_webp(form['id'], side)
        if body is None:
            miss += 1
            row['result'] = 'gen9 missing'
            report.append(row)
            print(f'{n}/{len(work)} MISS {form["slug"]} {side}', flush=True)
            continue
        try:
            gif, frames = webp_to_gif(body)
            row['frames'] = frames
            row['source'] = url
            if APPLY:
                actions, rel, relr = write_side(form, side, gif)
                row['result'] = ', '.join(actions)
                newb[rel] = bw.bounds(str(SPRITE / rel))
                newb[relr] = bw.bounds(str(SPRITE / relr))
            else:
                row['result'] = f'would write {len(gif)} bytes {frames} frames'
            got += 1
        except Exception as err:
            miss += 1
            row['result'] = 'error ' + str(err)
            print(f'{n}/{len(work)} ERR {form["slug"]} {side} {err}', flush=True)
        report.append(row)
        print(f'{n}/{len(work)} {form["slug"]} {side} {row["result"]}', flush=True)
    out = Path('/tmp/gen9-animated-report.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=0))
    print('gen9 found', got, 'missing', miss, 'report', out, flush=True)
    if not APPLY:
        print('dry-run: no files written', flush=True)
        return
    bounds_path = ROOT / 'data/sprite_bounds.js'
    raw = bounds_path.read_text(encoding='utf-8')
    i = raw.index('{')
    j = raw.rindex('}') + 1
    data = json.loads(raw[i:j])
    changed = 0
    for key, value in newb.items():
        if data['g'].get(key) != value:
            data['g'][key] = value
            changed += 1
    bounds_path.write_text(raw[:i] + json.dumps(data, separators=(',', ':'), ensure_ascii=False) + raw[j:], encoding='utf-8')
    print('bounds entries updated', changed, 'of', len(newb), flush=True)

if __name__ == '__main__':
    main()
