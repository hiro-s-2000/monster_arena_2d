#!/usr/bin/env python3
"""Fill sides whose pokestats Pixel / Black-White animated GIF is missing.

Source of the new bytes is Pokémon Showdown:
  front  https://play.pokemonshowdown.com/sprites/ani/<spriteid>.gif
  back   https://play.pokemonshowdown.com/sprites/ani-back/<spriteid>.gif

A side that already has the Black / White animated GIF is left as it is.
front-reverse.gif / back-reverse.gif are a lossless horizontal flip of the side
that was written. Shiny GIFs, PNGs, and 8-direction sheets are not touched.

Skipped on purpose:
  latias-mega and charizard-mega-y (authored GIFs)
  charizard-mega-x back (authored to match its front)

Usage:
  python3 tools/update_gifs_showdown_fallback.py            # count only
  python3 tools/update_gifs_showdown_fallback.py --apply
"""
import hashlib, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import update_gifs_pokestats_bw as bw

ROOT = Path('/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude')
SPRITE = ROOT / 'monsterSprite'
POKE = Path('/Users/honmahiroshi/Desktop/pokemon_arena_2d/pokeData')
BACKUP = Path('/Users/honmahiroshi/Desktop/ai_workspace/_backup_monsterSprite_gifs_before_showdown_20261115')
PS_DEX_URL = 'https://play.pokemonshowdown.com/data/pokedex.json'
BW = 'https://assets.pokestats.gg/pokeapi/v1/pokemon/versions/generation-v/black-white/animated/{id}.gif'
BW_BACK = 'https://assets.pokestats.gg/pokeapi/v1/pokemon/versions/generation-v/black-white/animated/back/{id}.gif'
ANI = 'https://play.pokemonshowdown.com/sprites/ani/{sid}.gif'
ANI_BACK = 'https://play.pokemonshowdown.com/sprites/ani-back/{sid}.gif'
UA = {'User-Agent': 'arena-sprite-sync/1.0'}
APPLY = '--apply' in sys.argv
SKIP_ALL = {'latias-mega', 'charizard-mega-y'}
SKIP_SIDE = {('charizard-mega-x', 'back')}
ALIAS = {
    'tauros-paldea-aqua-breed': 'taurospaldeaaqua',
    'tauros-paldea-blaze-breed': 'taurospaldeablaze',
    'tauros-paldea-combat-breed': 'taurospaldeacombat',
    'deoxys-normal': 'deoxys',
    'wormadam-plant': 'wormadam',
    'giratina-altered': 'giratina',
    'shaymin-land': 'shaymin',
    'basculin-red-striped': 'basculin',
    'darmanitan-galar-standard': 'darmanitangalar',
    'darmanitan-standard': 'darmanitan',
    'frillish-male': 'frillish',
    'jellicent-male': 'jellicent',
    'tornadus-incarnate': 'tornadus',
    'thundurus-incarnate': 'thundurus',
    'landorus-incarnate': 'landorus',
    'keldeo-ordinary': 'keldeo',
    'meloetta-aria': 'meloetta',
    'pyroar-male': 'pyroar',
    'meowstic-female-mega': 'meowsticfmega',
    'meowstic-male': 'meowstic',
    'meowstic-male-mega': 'meowsticmmega',
    'aegislash-shield': 'aegislash',
    'pumpkaboo-average': 'pumpkaboo',
    'gourgeist-average': 'gourgeist',
    'zygarde-50': 'zygarde',
    'oricorio-baile': 'oricorio',
    'lycanroc-midday': 'lycanroc',
    'wishiwashi-solo': 'wishiwashi',
    'minior-red-meteor': 'miniormeteor',
    'mimikyu-disguised': 'mimikyu',
    'toxtricity-amped': 'toxtricity',
    'toxtricity-amped-gmax': 'toxtricitygmax',
    'eiscue-ice': 'eiscue',
    'indeedee-male': 'indeedee',
    'morpeko-full-belly': 'morpeko',
    'urshifu-single-strike': 'urshifu',
    'urshifu-single-strike-gmax': 'urshifugmax',
    'basculegion-male': 'basculegion',
    'enamorus-incarnate': 'enamorus',
    'oinkologne-male': 'oinkologne',
    'maushold-family-of-four': 'mausholdfour',
    'squawkabilly-green-plumage': 'squawkabilly',
    'palafin-zero': 'palafin',
    'tatsugiri-curly': 'tatsugiri',
    'dudunsparce-two-segment': 'dudunsparce',
}

def to_id(text):
    return re.sub(r'[^a-z0-9]+', '', text.lower())

def sprite_id(entry):
    name = entry['name']
    base = entry.get('baseSpecies') or name
    forme = entry.get('forme') or ''
    base_id = to_id(base)
    sid = to_id(name)
    formeid = '' if base_id == sid else '-' + to_id(forme)
    sprite = base_id + formeid
    if sprite.endswith('totem'):
        sprite = sprite[:-5]
    if sprite == 'greninja-bond':
        sprite = 'greninja'
    if sprite == 'rockruff-dusk':
        sprite = 'rockruff'
    if sprite.endswith('-'):
        sprite = sprite[:-1]
    return sprite

def load_ps():
    path = Path('/tmp/showdown-fallback/pokedex.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        req = urllib.request.Request(PS_DEX_URL, headers=UA)
        path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return json.loads(path.read_text())

def load_forms(ps):
    ids = {}
    for dirpath, _, files in os.walk(POKE):
        for name in files:
            if not name.endswith('.json'):
                continue
            data = json.loads(Path(dirpath, name).read_text())
            if isinstance(data, dict) and 'slug' in data and 'pokemonId' in data:
                ids.setdefault(data['slug'], int(data['pokemonId']))
    forms = []
    index = json.loads((ROOT / 'monsterData/index.json').read_text())
    for mon in index['pokemon']:
        slug = mon['slug']
        rel = mon['sprite'].replace('../pokeSprite/', '')
        directory = rel.rsplit('/', 1)[0]
        key = ALIAS.get(slug, to_id(slug))
        entry = ps.get(key)
        forms.append({
            'slug': slug,
            'id': ids[slug],
            'dir': directory,
            'ps': key if entry else None,
            'spriteid': sprite_id(entry) if entry else None,
        })
    return forms

def fetch(url, method='GET'):
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, method=method, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as res:
                body = b'' if method == 'HEAD' else res.read()
                return res.status, res.headers.get('Content-Type', ''), body
        except Exception as err:
            last = err
            code = getattr(err, 'code', None)
            if code == 404:
                return 404, '', b''
            time.sleep(0.4 * (attempt + 1))
    raise last

def is_gif(status, ctype, body, method):
    if status != 200 or 'gif' not in ctype.lower():
        return False
    if method == 'GET':
        return len(body) > 100 and body[:6] in (b'GIF89a', b'GIF87a')
    return True

def probe(forms):
    jobs = []
    for form in forms:
        jobs.append((form['slug'], 'front', BW.format(id=form['id'])))
        jobs.append((form['slug'], 'back', BW_BACK.format(id=form['id'])))
    have = {}
    with ThreadPoolExecutor(max_workers=24) as pool:
        futs = {pool.submit(fetch, url, 'HEAD'): (slug, side) for slug, side, url in jobs}
        done = 0
        for fut in as_completed(futs):
            slug, side = futs[fut]
            done += 1
            if done % 400 == 0:
                print(f'bw probed {done}/{len(jobs)}', flush=True)
            try:
                status, ctype, _ = fut.result()
                have[(slug, side)] = is_gif(status, ctype, b'', 'HEAD')
            except Exception as err:
                have[(slug, side)] = False
                print('bw error', slug, side, err, flush=True)
    return have

def showdown_bytes(spriteid, side, poke_id):
    urls = [(ANI if side == 'front' else ANI_BACK).format(sid=spriteid)]
    mirror = 'https://assets.pokestats.gg/pokeapi/v1/pokemon/other/showdown/'
    urls.append(mirror + ('back/' if side == 'back' else '') + f'{poke_id}.gif')
    for url in urls:
        status, ctype, body = fetch(url, 'GET')
        if is_gif(status, ctype, body, 'GET'):
            return body
    return None

def backup(rel):
    src = SPRITE / rel
    dst = BACKUP / rel
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        bw.shutil.copy2(src, dst)

def write_side(form, side, body):
    rel = f"{form['dir']}/{side}.gif"
    relr = f"{form['dir']}/{side}-reverse.gif"
    loc = SPRITE / rel
    locr = SPRITE / relr
    loc.parent.mkdir(parents=True, exist_ok=True)
    changed = (not loc.exists()) or hashlib.md5(loc.read_bytes()).hexdigest() != hashlib.md5(body).hexdigest()
    actions = []
    if changed:
        if loc.exists():
            backup(rel)
            actions.append('replaced')
        else:
            actions.append('created')
        loc.write_bytes(body)
    if changed or (not locr.exists()) or (not bw.is_flip(str(loc), str(locr))):
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
    ps = load_ps()
    checks = {
        'charizardmegax': 'charizard-megax',
        'growlithehisui': 'growlithe-hisui',
        'indeedeef': 'indeedee-f',
        'nidoranf': 'nidoranf',
        'mrmime': 'mrmime',
        'zygarde10': 'zygarde-10',
        'urshifurapidstrike': 'urshifu-rapidstrike',
        'taurospaldeacombat': 'tauros-paldeacombat',
        'necrozmaduskmane': 'necrozma-duskmane',
        'toxtricitylowkey': 'toxtricity-lowkey',
        'mausholdfour': 'maushold-four',
        'charizardgmax': 'charizard-gmax',
    }
    for key, expect in checks.items():
        if sprite_id(ps[key]) != expect:
            raise SystemExit(f'sprite id {key} -> {sprite_id(ps[key])} expected {expect}')
    for target in ALIAS.values():
        if target not in ps:
            raise SystemExit('alias target missing from showdown dex: ' + target)
    forms = load_forms(ps)
    unmapped = [f['slug'] for f in forms if not f['spriteid']]
    print('forms', len(forms), 'unmapped', len(unmapped), unmapped, flush=True)
    have = probe(forms)
    need = []
    for form in forms:
        for side in ('front', 'back'):
            if have[(form['slug'], side)]:
                continue
            if form['slug'] in SKIP_ALL or (form['slug'], side) in SKIP_SIDE:
                need.append({**form, 'side': side, 'skip': 'authored'})
                continue
            if not form['spriteid']:
                need.append({**form, 'side': side, 'skip': 'no showdown species'})
                continue
            need.append({**form, 'side': side, 'skip': None})
    both = {}
    for item in need:
        both.setdefault(item['slug'], set()).add(item['side'])
    print('missing sides', len(need), 'forms', len(both),
          'both', sum(1 for v in both.values() if v == {'front', 'back'}),
          'front only', sum(1 for v in both.values() if v == {'front'}),
          'back only', sum(1 for v in both.values() if v == {'back'}), flush=True)
    authored = [i for i in need if i['skip'] == 'authored']
    print('authored skips', [(i['slug'], i['side']) for i in authored], flush=True)
    work = [i for i in need if not i['skip']]
    report = []
    newb = {}
    got = miss = 0
    for n, item in enumerate(work, 1):
        row = {'slug': item['slug'], 'side': item['side'], 'dir': item['dir'], 'spriteid': item['spriteid'], 'id': item['id']}
        if not APPLY:
            url = (ANI if item['side'] == 'front' else ANI_BACK).format(sid=item['spriteid'])
            status, ctype, _ = fetch(url, 'HEAD')
            if not is_gif(status, ctype, b'', 'HEAD'):
                miss += 1
                row['result'] = 'showdown missing'
                report.append(row)
                print(f'{n}/{len(work)} MISS {item["slug"]} {item["side"]} {item["spriteid"]}', flush=True)
                continue
            got += 1
            row['result'] = 'would write'
            report.append(row)
            if n % 25 == 0:
                print(f'{n}/{len(work)} {item["slug"]} {item["side"]} would write', flush=True)
            continue
        try:
            body = showdown_bytes(item['spriteid'], item['side'], item['id'])
        except Exception as err:
            miss += 1
            row['result'] = 'error ' + str(err)
            report.append(row)
            print(f'{n}/{len(work)} ERR {item["slug"]} {item["side"]} {err}', flush=True)
            continue
        if body is None:
            miss += 1
            row['result'] = 'showdown missing'
            report.append(row)
            print(f'{n}/{len(work)} MISS {item["slug"]} {item["side"]} {item["spriteid"]}', flush=True)
            continue
        got += 1
        try:
            actions, rel, relr = write_side(item, item['side'], body)
        except Exception as err:
            miss += 1
            got -= 1
            row['result'] = 'write error ' + str(err)
            report.append(row)
            print(f'{n}/{len(work)} WRITE {item["slug"]} {item["side"]} {err}', flush=True)
            continue
        row['result'] = ', '.join(actions) if actions else 'already identical'
        if actions:
            newb[rel] = bw.bounds(str(SPRITE / rel))
            newb[relr] = bw.bounds(str(SPRITE / relr))
        report.append(row)
        if n % 25 == 0 or body is None:
            print(f'{n}/{len(work)} {item["slug"]} {item["side"]} {row["result"]}', flush=True)
    out = Path('/tmp/showdown-fallback/report.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=0))
    print('showdown found', got, 'missing', miss, 'report', out, flush=True)
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
