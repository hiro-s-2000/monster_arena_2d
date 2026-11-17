#!/usr/bin/env python3
"""Build a self-contained static artifact for GitHub Pages under the 1 GB site limit.

Keeps local files on `main` completely untouched:
- Copies `index.html`, `debug.html`, `.nojekyll`, `game/`, `data/`, `trainers/`, and runtime stills/icons/GIFs
  (omitting `*-reverse.gif` which `assets.js` / `debug.js` mirror in memory, and omitting heavy `fb` normal GIFs).
- Losslessly converts sprite sheets (`agent-sprite-forge/sheet.png`, `actions/sheet.png`, 8-dir & `fb` sheets)
  to `.webp` using `cwebp -lossless -z 4` (with content-hashed `--cache` support for fast CI builds).
- Packs as many `fb` (Mega/Gigantamax) sheets as fit within `--max-mib`, routing any remaining overflow `fb`
  sheets and on-demand `fb` `#official` GIFs to `raw.githubusercontent.com` via `data/pages_assets.js`.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_NAMES = {".DS_Store", "__pycache__"}


def extract_json_var(path: Path, var_name: str):
    txt = path.read_text(encoding="utf-8")
    m = re.search(re.escape(var_name) + r"\s*=\s*", txt)
    if not m:
        raise ValueError(f"{var_name} not found in {path}")
    obj, _ = json.JSONDecoder().raw_decode(txt, m.end())
    return obj


def sha1_12(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:12]


def copy_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def copy_tree(src_dir: Path, dst_dir: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_NAMES]
        rel_root = Path(root).relative_to(src_dir)
        for fn in files:
            if fn in SKIP_NAMES or fn.endswith((".pyc", ".pyo")):
                continue
            total += copy_file(Path(root) / fn, dst_dir / rel_root / fn)
    return total


def convert_sheet(task):
    rel, src_path, cache_dir, z = task
    orig_size = src_path.stat().st_size
    sig = f"{orig_size}:{sha1_12(src_path)}"

    if cache_dir is not None:
        c_webp = cache_dir / (rel[:-4] + ".webp")
        c_meta = cache_dir / (rel + ".meta")
        if c_meta.exists():
            lines = c_meta.read_text(encoding="utf-8").splitlines()
            if len(lines) >= 2 and lines[0] == sig:
                mode = lines[1]
                if mode == "webp" and c_webp.exists():
                    return rel, "webp", c_webp, c_webp.stat().st_size
                if mode == "png":
                    return rel, "png", src_path, orig_size

    tmp_out = (cache_dir / (rel[:-4] + ".webp.tmp")) if cache_dir is not None else Path(str(src_path) + ".webp.tmp")
    tmp_out.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["cwebp", "-quiet", "-lossless", "-z", str(z), str(src_path), "-o", str(tmp_out)],
        capture_output=True,
    )
    use_webp = res.returncode == 0 and tmp_out.exists() and tmp_out.stat().st_size < orig_size
    if cache_dir is not None:
        c_webp = cache_dir / (rel[:-4] + ".webp")
        c_meta = cache_dir / (rel + ".meta")
        c_meta.parent.mkdir(parents=True, exist_ok=True)
        if use_webp:
            tmp_out.replace(c_webp)
            c_meta.write_text(f"{sig}\nwebp\n", encoding="utf-8")
            return rel, "webp", c_webp, c_webp.stat().st_size
        tmp_out.unlink(missing_ok=True)
        c_meta.write_text(f"{sig}\npng\n", encoding="utf-8")
        return rel, "png", src_path, orig_size

    if use_webp:
        return rel, "webp", tmp_out, tmp_out.stat().st_size
    tmp_out.unlink(missing_ok=True)
    return rel, "png", src_path, orig_size


def main():
    ap = argparse.ArgumentParser(description="Build GitHub Pages artifact under 1 GB.")
    ap.add_argument("--out", default="_site", help="Output directory")
    ap.add_argument("--cache", default=None, help="Optional WebP cache directory")
    ap.add_argument("--max-mib", type=float, default=820.0, help="Target max Pages size in MiB")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="cwebp worker threads")
    ap.add_argument("--z", type=int, default=4, help="cwebp -lossless compression effort (0-9)")
    ap.add_argument(
        "--raw-base",
        default="https://raw.githubusercontent.com/hiro-s-2000/monster_arena_2d/main/monsterSprite/",
        help="GitHub raw base URL for overflow assets",
    )
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    cache_dir = None
    if args.cache:
        cache_dir = Path(args.cache)
        if not cache_dir.is_absolute():
            cache_dir = ROOT / cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    if not shutil.which("cwebp"):
        raise SystemExit("cwebp not found on PATH. Install webp (brew install webp / apt install webp).")

    total_bytes = 0
    for fn in ("index.html", "debug.html", ".nojekyll"):
        p = ROOT / fn
        if p.exists():
            total_bytes += copy_file(p, out / fn)

    total_bytes += copy_tree(ROOT / "game", out / "game")
    for fn in ("monsterdex.js", "trainers.js", "dir8.js", "sprite_bounds.js", "flame_hues.js"):
        src = ROOT / "data" / fn
        if src.exists():
            total_bytes += copy_file(src, out / "data" / fn)

    trainers = extract_json_var(ROOT / "data/trainers.js", "window.ARENA_TRAINERS")
    seen_files = set()
    for t in trainers:
        src_rel = t["src"]
        if src_rel not in seen_files and (ROOT / src_rel).exists():
            seen_files.add(src_rel)
            total_bytes += copy_file(ROOT / src_rel, out / src_rel)

    spr_root = ROOT / "monsterSprite"
    out_spr = out / "monsterSprite"
    ball = spr_root / "items/poke-ball.png"
    if ball.exists():
        total_bytes += copy_file(ball, out_spr / "items/poke-ball.png")

    dex = extract_json_var(ROOT / "data/monsterdex.js", "window.ARENA_MONSTERDEX")
    dir8 = extract_json_var(ROOT / "data/dir8.js", "window.ARENA_DIR8")

    fb_dirs = {}
    non_fb_tasks = []
    fb_groups = {}  # pid -> list of sheet tasks
    seen_spr = set()

    def copy_spr(rel: str):
        nonlocal total_bytes
        if rel in seen_spr:
            return
        src = spr_root / rel
        if src.exists():
            seen_spr.add(rel)
            total_bytes += copy_file(src, out_spr / rel)

    for p in dex:
        d = p["dir"]
        pid = p["id"]
        m0 = dir8.get(pid)
        is_fb = bool(m0 and m0.get("layout") == "fb")
        if is_fb:
            fb_dirs[d] = 1

        ico = p.get("icon") or "front.png"
        copy_spr(f"{d}/{ico}")
        for fn in ("front.png", "back.png", "front-shiny.png", "back-shiny.png"):
            copy_spr(f"{d}/{fn}")
        if not is_fb:
            for fn in ("front.gif", "back.gif"):
                copy_spr(f"{d}/{fn}")
        if p.get("shiny"):
            for fn in ("front-shiny.gif", "back-shiny.gif"):
                copy_spr(f"{d}/{fn}")

        if not m0:
            continue
        for alt, m in (("normal", m0), ("shiny", m0.get("shiny"))):
            if not m:
                continue
            key = pid + ("-shiny" if alt == "shiny" else "")
            layout = m.get("layout")
            if layout == "fb":
                for fn in (f"{pid}.png", f"{pid}-b.png"):
                    rel = f"{d}/{fn}"
                    if rel not in seen_spr and (spr_root / rel).exists():
                        seen_spr.add(rel)
                        fb_groups.setdefault(pid, []).append((rel, spr_root / rel, cache_dir, args.z))
            elif layout == "actions_8dir" or m.get("sheetDir") == "agent-sprite-forge":
                rel = f"{d}/agent-sprite-forge/sheet.png"
                if rel not in seen_spr and (spr_root / rel).exists():
                    seen_spr.add(rel)
                    non_fb_tasks.append((rel, spr_root / rel, cache_dir, args.z))
            elif layout == "actions":
                rel = f"{d}/actions/sheet.png"
                if rel not in seen_spr and (spr_root / rel).exists():
                    seen_spr.add(rel)
                    non_fb_tasks.append((rel, spr_root / rel, cache_dir, args.z))
            else:
                rel = f"{d}/{key}.png"
                if rel not in seen_spr and (spr_root / rel).exists():
                    seen_spr.add(rel)
                    non_fb_tasks.append((rel, spr_root / rel, cache_dir, args.z))

    base_bytes = total_bytes
    print(f"Base files copied: {base_bytes / 1048576:.1f} MiB")

    no_webp = {}
    remote = {}

    fb_tasks = [t for tasks in fb_groups.values() for t in tasks]
    all_tasks = non_fb_tasks + fb_tasks
    print(f"Converting/resolving {len(all_tasks)} sheets with {args.workers} workers (z={args.z})...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(convert_sheet, all_tasks))

    res_map = {r[0]: r for r in results}
    for rel, _, _, _ in non_fb_tasks:
        _, mode, artifact_path, sz = res_map[rel]
        if mode == "webp":
            total_bytes += copy_file(artifact_path, out_spr / (rel[:-4] + ".webp"))
            if cache_dir is None:
                artifact_path.unlink(missing_ok=True)
        else:
            no_webp[rel] = 1
            total_bytes += copy_file(artifact_path, out_spr / rel)

    non_fb_bytes = total_bytes - base_bytes
    print(f"Non-fb sheets: {non_fb_bytes / 1048576:.1f} MiB (kept PNG: {len(no_webp)})")

    max_bytes = int(args.max_mib * 1048576)
    fb_ranked = []
    for pid, tasks in fb_groups.items():
        items = [res_map[t[0]] for t in tasks]
        pair_bytes = sum(it[3] for it in items)
        fb_ranked.append((pair_bytes, pid, items))
    fb_ranked.sort()

    local_fb = 0
    remote_fb = 0
    for pair_bytes, pid, items in fb_ranked:
        if total_bytes + pair_bytes <= max_bytes:
            local_fb += 1
            for rel, mode, artifact_path, _ in items:
                if mode == "webp":
                    total_bytes += copy_file(artifact_path, out_spr / (rel[:-4] + ".webp"))
                    if cache_dir is None:
                        artifact_path.unlink(missing_ok=True)
                else:
                    no_webp[rel] = 1
                    total_bytes += copy_file(artifact_path, out_spr / rel)
        else:
            remote_fb += 1
            for rel, mode, artifact_path, _ in items:
                remote[rel] = 1
                if cache_dir is None and mode == "webp":
                    artifact_path.unlink(missing_ok=True)

    pages_cfg = {
        "raw": args.raw_base,
        "mirrorRev": True,
        "omitFbGifs": True,
        "webpSheets": True,
        "noWebp": no_webp,
        "fbDirs": fb_dirs,
        "remote": remote,
    }
    cfg_path = out / "data/pages_assets.js"
    cfg_path.write_text(
        "// Auto-generated by tools/build_pages.py\nwindow.ARENA_PAGES = "
        + json.dumps(pages_cfg, separators=(",", ":"), ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    total_bytes += cfg_path.stat().st_size

    print(
        f"fb mons: {local_fb} local / {remote_fb} remote | "
        f"Total output: {total_bytes / 1048576:.1f} MiB ({total_bytes / 1e6:.1f} MB) in {time.time() - t0:.1f}s"
    )
    if total_bytes > 980_000_000:
        raise SystemExit(f"Output too large for GitHub Pages 1 GB limit: {total_bytes} bytes")


if __name__ == "__main__":
    main()
