"""Insert one gen5 actions_8dir entry. Never reformat the whole file."""

import json
from pathlib import Path

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
BOUNDS = ROOT / "data/sprite_bounds.js"
BOUNDS_LOCK = ROOT / "data/sprite_bounds.lock"
DIR8_PREFIX = "window.ARENA_DIR8 = "


def _flock(path, fn):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _split_js(text, prefix):
    if not text.startswith(prefix):
        raise SystemExit(f"header changed: {prefix!r}")
    rest = text[len(prefix):]
    semi = rest.rstrip().endswith(";")
    body = rest.strip()
    if body.endswith(";"):
        body = body[:-1].strip()
    return body, semi


def _join_js(prefix, body, semi):
    return prefix + body + (";" if semi else "") + "\n"


def _object_span(text, key):
    token = json.dumps(key) + ":"
    if text.count(token) != 1:
        raise SystemExit(f"{key} appears {text.count(token)} times")
    s = text.index(token)
    i = s + len(token)
    if text[i] != "{":
        raise SystemExit(f"{key} value is not an object")
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s, i, j + 1
    raise SystemExit(f"{key} object did not close")


def _array_span(text, key):
    token = json.dumps(key) + ":["
    if text.count(token) != 1:
        raise SystemExit(f"{key} array appears {text.count(token)} times")
    s = text.index(token)
    i = s + len(token) - 1
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s, j + 1
    raise SystemExit(f"{key} array did not close")


def _brace_end(text, start):
    """Index just past the object that starts at text[start] == '{'."""
    if text[start] != "{":
        raise SystemExit("brace start is not an object")
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise SystemExit("object did not close")


def publish_dir8(key, entry):
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        body, semi = _split_js(text, DIR8_PREFIX)
        old = json.loads(body)
        token = json.dumps(key) + ":"
        if token in body:
            _s, val, end = _object_span(body, key)
            new_body = body[:val] + blob + body[end:]
        else:
            if not body.endswith("}"):
                raise SystemExit("dir8 root is not a single object")
            new_body = body[:-1] + "," + token + blob + "}"
        new = json.loads(new_body)
        if new.get(key) != entry:
            raise SystemExit("dir8 entry did not parse back")
        for k, v in old.items():
            if k != key and new.get(k) != v:
                raise SystemExit(f"dir8 neighbour changed: {k}")
        if set(new) - set(old) - {key}:
            raise SystemExit("dir8 gained an unexpected key")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(_join_js(DIR8_PREFIX, new_body, semi))
        tmp.replace(DIR8)

    _flock(DIR8_LOCK, edit)


def _bounds_prefix_and_s(text):
    marker = "window.ARENA_BOUNDS = "
    i = text.find(marker)
    if i < 0:
        raise SystemExit("sprite_bounds header missing")
    prefix = text[: i + len(marker)]
    body, semi = _split_js(text[i:], marker)
    data = json.loads(body)
    if "s" not in data or not isinstance(data["s"], dict):
        raise SystemExit("sprite_bounds has no s map")
    token = '"s":{'
    s_key = body.find(token)
    if s_key < 0:
        raise SystemExit("s object text missing")
    s_start = s_key + len('"s":')
    s_end = _brace_end(body, s_start)
    return prefix, body, semi, s_start, s_end


def publish_bounds(key, rects):
    """rects is 32 ints: eight [x, y, w, h] for S SE E NE N NW W SW idle frame 0."""
    if len(rects) != 32:
        raise SystemExit(f"bounds need 32 ints, got {len(rects)}")
    blob = "[" + ",".join(str(int(v)) for v in rects) + "]"

    def edit():
        text = BOUNDS.read_text()
        prefix, body, semi, s_start, s_end = _bounds_prefix_and_s(text)
        s_text = body[s_start:s_end]
        old_s = json.loads(s_text)
        token = json.dumps(key) + ":"
        if token in s_text:
            rel_s, rel_end = _array_span(s_text, key)
            new_s = s_text[:rel_s] + token + blob + s_text[rel_end:]
        else:
            if not s_text.endswith("}"):
                raise SystemExit("s map is not an object")
            new_s = s_text[:-1] + "," + token + blob + "}"
        new_map = json.loads(new_s)
        if new_map.get(key) != [int(v) for v in rects]:
            raise SystemExit("bounds entry did not parse back")
        for k, v in old_s.items():
            if k != key and new_map.get(k) != v:
                raise SystemExit(f"bounds neighbour changed: {k}")
        new_body = body[:s_start] + new_s + body[s_end:]
        json.loads(new_body)
        tmp = BOUNDS.with_suffix(".js.tmp")
        tmp.write_text(prefix + new_body + (";" if semi else "") + "\n")
        tmp.replace(BOUNDS)

    _flock(BOUNDS_LOCK, edit)


def measure_idle_bounds(sheet_path, cell_w, cell_h, rows=8):
    import numpy as np
    from PIL import Image

    im = np.array(Image.open(sheet_path).convert("RGBA"))
    rects = []
    for i in range(rows):
        tile = im[i * cell_h:(i + 1) * cell_h, 0:cell_w]
        alpha = tile[..., 3] >= 128
        if not alpha.any():
            rects.extend([0, 0, 0, 0])
            continue
        ys, xs = np.nonzero(alpha)
        x = int(xs.min())
        y = int(ys.min())
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        rects.extend([x, y, w, h])
    return rects


def selftest():
    sample = DIR8_PREFIX + '{"0380_latias-mega":{"dex":380,"v":"a"}};\n'
    body, semi = _split_js(sample, DIR8_PREFIX)
    assert semi and json.loads(body)["0380_latias-mega"]["dex"] == 380
    entry = {"dex": 494, "v": "victini_v1"}
    blob = json.dumps(entry, separators=(",", ":"))
    token = '"0494_victini":'
    new_body = body[:-1] + "," + token + blob + "}"
    parsed = json.loads(new_body)
    assert parsed["0494_victini"] == entry
    assert parsed["0380_latias-mega"]["v"] == "a"
    bounds = (
        "// Generated\nwindow.ARENA_BOUNDS = "
        '{"g":{"x":[1]},"s":{"0001_bulbasaur":[1,2,3,4]}};\n'
    )
    prefix, body, semi, s_start, s_end = _bounds_prefix_and_s(bounds)
    assert json.loads(body[s_start:s_end])["0001_bulbasaur"] == [1, 2, 3, 4]
    real = DIR8.read_text()
    real_body, _semi = _split_js(real, DIR8_PREFIX)
    data = json.loads(real_body)
    assert "0494_victini" not in data
    btxt = BOUNDS.read_text()
    _p, bbody, _s, ss, se = _bounds_prefix_and_s(btxt)
    smap = json.loads(bbody[ss:se])
    assert "0494_victini" not in smap
    assert "0001_bulbasaur" in smap
    print("selftest ok", "dir8", len(data), "bounds_s", len(smap))


if __name__ == "__main__":
    selftest()
