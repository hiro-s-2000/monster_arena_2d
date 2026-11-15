#!/usr/bin/env python3
"""Overwrite every Gigantamax front/back GIF from pokestats Generation VIII.

Spec (do not copy another form's pixels; each slug uses only its own PokeAPI id):

  Displayed Sword / Shield, Animated (256px), Generation VIII:
    https://assets.pokestats.gg/pokeapi/v1/anim/pokemon/versions/generation-viii/sword-shield/<id>.webp
    .../back/<id>.webp
  The game loads .gif, so the animated WebP is converted. -reverse.gif is a
  lossless horizontal flip. Shiny GIFs, PNGs, and 8-direction sheets are not touched.
  A missing animated file leaves that side unchanged.

  python3 tools/update_gifs_gen8_gmax.py --apply
"""
import io, json, shutil, subprocess, sys, urllib.request
from pathlib import Path
from PIL import Image, ImageSequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import update_gifs_pokestats_bw as bw

ROOT = Path('/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude')
SPRITE = ROOT / 'monsterSprite'
BACKUP = Path('/Users/honmahiroshi/Desktop/ai_workspace/_backup_monsterSprite_gifs_before_gen8_gmax_20261115')
APPLY = '--apply' in sys.argv
URL = 'https://assets.pokestats.gg/pokeapi/v1/anim/pokemon/versions/generation-viii/sword-shield/{tail}{id}.webp'

def load_forms():
    index = json.loads((ROOT / 'monsterData/index.json').read_text())
    ids = {}
    for path in (ROOT / 'monsterData').rglob('*.json'):
        if path.name == 'index.json':
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        slug = data.get('slug')
        pid = data.get('pokemonId')
        if isinstance(slug, str) and slug.endswith('-gmax') and isinstance(pid, int):
            ids.setdefault(slug, pid)
    forms = []
    for entry in index['pokemon']:
        slug = entry.get('slug') or ''
        if not slug.endswith('-gmax'):
            continue
        sprite = entry['sprite']
        rel = sprite.replace('../pokeSprite/', '').replace('../monsterSprite/', '')
        directory = str(Path(rel).parent)
        forms.append({'slug': slug, 'id': ids[slug], 'dir': directory, 'jp': entry.get('jpName')})
    forms.sort(key=lambda f: f['dir'])
    return forms

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as res:
        ctype = res.headers.get('Content-Type', '')
        body = res.read()
    return ctype, body

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

def webp_to_gif(data, slug, side):
    tmp = Path(f'/tmp/gen8-gmax-{slug}-{side}.webp')
    tmp.write_bytes(data)
    im = Image.open(io.BytesIO(data))
    frames = [fr.convert('RGBA') for fr in ImageSequence.Iterator(im)]
    durs = durations_ms(tmp, len(frames))
    buf = io.BytesIO()
    frames[0].save(buf, format='GIF', save_all=True, append_images=frames[1:],
                   duration=durs, loop=0, disposal=2, optimize=False)
    tmp.unlink(missing_ok=True)
    return buf.getvalue(), len(frames), frames[0].size

def backup(path, rel):
    if path.exists():
        dst = BACKUP / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)

def write_side(form, side, gif):
    rel = f"{form['dir']}/{side}.gif"
    relr = f"{form['dir']}/{side}-reverse.gif"
    loc = SPRITE / rel
    locr = SPRITE / relr
    loc.parent.mkdir(parents=True, exist_ok=True)
    action = 'replaced' if loc.exists() else 'created'
    backup(loc, rel)
    backup(locr, relr)
    loc.write_bytes(gif)
    bw.flip_gif(str(loc), str(locr))
    if not bw.is_flip(str(loc), str(locr)):
        raise RuntimeError('flip mismatch ' + relr)
    return action, rel, relr

def main():
    forms = load_forms()
    print('gmax', len(forms), flush=True)
    report = []
    newb = {}
    got = miss = 0
    for n, form in enumerate(forms, 1):
        for side in ('front', 'back'):
            row = {'slug': form['slug'], 'side': side, 'id': form['id'], 'dir': form['dir']}
            url = URL.format(tail='back/' if side == 'back' else '', id=form['id'])
            try:
                ctype, body = fetch(url)
            except Exception as err:
                miss += 1
                row['result'] = 'fetch error ' + str(err)
                report.append(row)
                print(f'{n}/{len(forms)} ERR {form["slug"]} {side} {err}', flush=True)
                continue
            if 'webp' not in ctype.lower() or body[:4] != b'RIFF' or b'ANIM' not in body[:128]:
                miss += 1
                row['result'] = 'gen8 anim missing'
                report.append(row)
                print(f'{n}/{len(forms)} MISS {form["slug"]} {side}', flush=True)
                continue
            try:
                gif, frames, size = webp_to_gif(body, form['slug'], side)
                row['frames'] = frames
                row['size'] = list(size)
                row['source'] = url
                if APPLY:
                    action, rel, relr = write_side(form, side, gif)
                    row['result'] = action
                    newb[rel] = bw.bounds(str(SPRITE / rel))
                    newb[relr] = bw.bounds(str(SPRITE / relr))
                else:
                    row['result'] = f'would write {len(gif)} bytes {frames} frames'
                got += 1
            except Exception as err:
                miss += 1
                row['result'] = 'error ' + str(err)
                print(f'{n}/{len(forms)} ERR {form["slug"]} {side} {err}', flush=True)
            report.append(row)
            print(f'{n}/{len(forms)} {form["slug"]} {side} {row["result"]}', flush=True)
    out = Path('/tmp/gmax-gen8-report.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=0))
    print('written', got, 'missing', miss, 'report', out, flush=True)
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
