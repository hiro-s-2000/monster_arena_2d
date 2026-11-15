// Sprite cache. GIFs are fetched, decoded and every frame is redrawn as pixel art (Pixel.pixelize) at a
// scale chosen so the sprite's body height matches `opts.artH` art px (size follows the Pokémon's real height).
// Record: { src, img, ready, failed, frames:[{canvas, delay}], total, w, h (art px), bounds (art px), portrait }.
// Must be served over http(s): file:// taints canvases (no pixel access, no WebGL upload).
"use strict";

const Assets = (() => {
  const cache = new Map();
  const canFetch = location.protocol.startsWith("http");
  const ROOT = "../monsterSprite/";

  let deferDepth = 0;
  function defer(fn) {
    deferDepth++;
    try { return fn(); } finally { deferDepth--; }
  }
  const isDeferred = () => deferDepth > 0;
  function start(recs) {
    if (!recs) return;
    if (typeof recs._start === "function") recs._start();
    else if (Array.isArray(recs)) recs.forEach((r) => { if (r && r._start) r._start(); });
  }

  function get(src, opts = {}) {
    const key = opts.raw ? src + "#raw" : opts.still ? src + "#still" : src;
    if (cache.has(key)) {
      const r = cache.get(key);
      if (opts.hd) r.hd = true;
      if (!deferDepth && r._start) r._start();
      return r;
    }
    const isGif = /\.gif$/i.test(src);
    const useFetchGif = canFetch && isGif && !opts.still;
    const rec = { src, img: new Image(), ready: false, failed: false, frames: null, total: 0, w: 0, h: 0, bounds: null, _tint: new Map(), raw: !!opts.raw, hd: !!opts.hd, artH: opts.artH || 0 };
    rec.img.decoding = "async";
    rec.img.onload = () => {
      if (rec.frames) return;
      try {
        if (rec.raw) { rec.frames = [{ canvas: rec.img, delay: 1 }]; rec.w = rec.img.naturalWidth; rec.h = rec.img.naturalHeight; }
        else {
          const b0 = Pixel.bounds(rec.img, 0, 0, rec.img.naturalWidth, rec.img.naturalHeight);
          setFrames(rec, [{ canvas: rec.img, delay: 1 }], rec.img.naturalWidth, rec.img.naturalHeight, b0);
        }
      } catch (e) { rec.frames = [{ canvas: rec.img, delay: 1 }]; rec.w = rec.img.naturalWidth; rec.h = rec.img.naturalHeight; }
      if (!isGif || !canFetch || opts.still) rec.ready = true;
    };
    rec.img.onerror = () => { if (isGif) pngFallback(rec); else rec.failed = true; };
    rec._start = () => {
      if (rec._started) return;
      rec._started = true;
      if (useFetchGif) {
        let fetchedOk = false;
        fetch(src).then((r) => { if (!r.ok) throw new Error("missing"); fetchedOk = true; return r.arrayBuffer(); }).then((buf) => {
          const g = Gif.decode(buf);
          setFrames(rec, g.frames, g.W, g.H, g.bounds0);
          rec.ready = true;
        }).catch(() => {
          if (fetchedOk) {
            rec.img.onload = () => {
              if (!rec.frames) {
                try { setFrames(rec, [{ canvas: rec.img, delay: 1 }], rec.img.naturalWidth, rec.img.naturalHeight, Pixel.bounds(rec.img)); }
                catch (e) { rec.frames = [{ canvas: rec.img, delay: 1 }]; rec.w = rec.img.naturalWidth; rec.h = rec.img.naturalHeight; }
              }
              rec.ready = true;
            };
            rec.img.src = src;
          } else pngFallback(rec);
        });
      } else {
        rec.img.src = src;
      }
    };
    cache.set(key, rec);
    if (!deferDepth) rec._start();
    return rec;
  }

  // Missing GIF (e.g. the Gigantamax GIFs deleted 2026-11-04 by user request): use the still front/back PNG of
  // the same name; "-reverse" is that PNG mirrored. No PNG either → failed.
  function pngFallback(rec) {
    if (rec._fb) return;
    rec._fb = true;
    const flip = /-reverse\.gif$/i.test(rec.src);
    const png = rec.src.replace(/-reverse\.gif$/i, ".gif").replace(/\.gif$/i, ".png");
    const im = new Image();
    im.onload = () => {
      const W = im.naturalWidth, H = im.naturalHeight, c = U.canvas(W, H), x = c.getContext("2d");
      if (flip) { x.translate(W, 0); x.scale(-1, 1); }
      x.drawImage(im, 0, 0);
      try { setFrames(rec, [{ canvas: c, delay: 1 }], W, H); } catch (e) { rec.frames = [{ canvas: c, delay: 1 }]; rec.total = 1; rec.w = W; rec.h = H; }
      rec.ready = true;
    };
    im.onerror = () => { rec.failed = true; rec.ready = true; };
    im.src = png;
  }

  function setFrames(rec, frames, W, H, bounds0) {
    // [ユーザー指定 2026-11-15] フィールド上も GIF の元のドットをそのまま使う（粗いドットへの作り直し＝pixelize は「荒すぎる」と却下）。
    // 大きさは Fighter.measure の fit が身長カーブへ合わせるので、ここでは拡大縮小しない。
    rec.scale = 1;
    rec.frames = frames.map((f) => ({ canvas: toCanvas(f.canvas, W, H), delay: f.delay }));
    rec.total = rec.frames.reduce((s, f) => s + f.delay, 0);
    rec.w = rec.frames[0].canvas.width; rec.h = rec.frames[0].canvas.height;
    rec.bounds = bounds0 || Pixel.bounds(rec.frames[0].canvas);
    rec.portrait = null; rec._tint.clear();
  }

  function toCanvas(img, W, H) {
    if (img && img.getContext && img.width === W && img.height === H) return img;
    const c = U.canvas(W, H); c.getContext("2d").drawImage(img, 0, 0); return c;
  }

  function frameIndex(rec, t) {
    if (!rec.frames || rec.frames.length < 2) return 0;
    let k = t % rec.total;
    for (let i = 0; i < rec.frames.length; i++) { if (k < rec.frames[i].delay) return i; k -= rec.frames[i].delay; }
    return rec.frames.length - 1;
  }
  const frame = (rec, t) => (rec.frames ? rec.frames[frameIndex(rec, t)].canvas : null);

  // Solid-colour silhouette of the current frame (hit flash, shadows, status tints). Cached.
  function tinted(rec, t, color) {
    const fi = frameIndex(rec, t), key = fi + color;
    let c = rec._tint.get(key);
    if (c) return c;
    const src = frame(rec, t);
    if (rec._tint.size > (src.width * src.height > 20000 ? 24 : 400)) rec._tint.clear(); // 大きいコマ（メガ・キョダイの 200px 級）はためこまない
    c = U.canvas(src.width, src.height);
    const x = c.getContext("2d");
    x.drawImage(src, 0, 0);
    x.globalCompositeOperation = "source-in"; x.fillStyle = color; x.fillRect(0, 0, c.width, c.height);
    rec._tint.set(key, c);
    return c;
  }

  // Head-and-shoulders crop of the first front frame for HUD portraits. Original-resolution (hd) records are
  // shown at a whole-number scale (crisp pixels); shrinking to tiny icons is smoothed instead of dropping pixels.
  function portrait(rec, size = 48) {
    if (!rec.frames) return null;
    rec.portraits = rec.portraits || {};
    if (rec.portraits[size]) return rec.portraits[size];
    const src = rec.frames[0].canvas, b = rec.bounds;
    let crop = Math.min(b.w, Math.max(16, Math.round(b.h * 0.72)));
    let k = size / crop;
    if (rec.hd && k >= 1) { k = Math.floor(k); crop = Math.round(size / k); }
    const cx = b.x + b.w / 2, sx = Math.round(U.clamp(cx - crop / 2, 0, src.width - crop)), sy = Math.max(0, Math.min(b.y, src.height - crop));
    const c = U.canvas(size, size), x = c.getContext("2d");
    x.imageSmoothingEnabled = k < 1; x.imageSmoothingQuality = "high";
    x.drawImage(src, sx, sy, crop, crop, 0, 0, crop * k, crop * k);
    rec.portraits[size] = c;
    return c;
  }

  // Height curve, scaled so a 1.7 m person / Charizard is the 48px ruler.
  // Below 1.7m scales linearly with height so smaller Pokémon stay genuinely small relative to Charizard.
  // Above 1.7m adds bulk for heavy monsters, while Gigantamax towers as a true stadium boss (195–268px).
  const TRAINER_M = 1.7;
  const TRAINER_PX = 48;
  function bodyArtPx(p) {
    if (!p) return TRAINER_PX;
    const gmax = p.kind === "gigantamax" || p.cat === "gigantamax";
    if (gmax) {
      let base = TRAINER_PX;
      if (p.base && typeof Data !== "undefined") {
        const root = Data.bySlug(p.base);
        // 元の体高だけで倍率を決める。1.9〜2.9m の縮小はキョダイマックスへ流さない。
        if (root && root !== p) base = bodyArtPx({ h: root.h, w: root.w, noBand: true });
      }
      const fromLore = +p.h > 5 ? 165 + (+p.h) * 2.6 : 0;
      return Math.round(U.clamp(Math.max(195, base * 4.2, fromLore), 195, 268));
    }
    // [ユーザー指定 2026-11-15] 1.7m（リザードン等）より小さいモンスターは実身長比（1.0乗）でしっかり小さくし、
    // 大きい側は体格・体重バルクを反映。キョダイマックスはさらに大きく（195〜268px → フィールド146〜201px）映す。
    const m = Math.max(0.1, +p.h || 1);
    const t = m / TRAINER_M;
    const fromH = TRAINER_PX * t;
    let bulk = 1;
    if (p.w > 100) bulk = Math.min(1.55, 1 + Math.log2(p.w / 100) * 0.24);
    // [ユーザー指定 2026-11-15] メガフシギバナ（2.4m）の ±0.5m、つまり 1.9〜2.9m は今の曲線より 15% 小さく。
    // 範囲のすぐ外 0.08m だけ、外側の曲線へなだらかにつなぐ。
    let band = 1;
    if (!p.noBand) {
      const d = Math.abs(m - 2.4);
      if (d <= 0.5) band = 0.85;
      else if (d < 0.58) band = 1 - 0.15 * ((0.58 - d) / 0.08);
    }
    return Math.round(U.clamp(fromH * bulk * band * bigBoost(p), BODY_MIN, BODY_MAX));
  }
  // [ユーザー指定 2026-11-15] リザードン（1.7m）以上のモンスターは全て少し大きく（×1.12）。1.4m から 1.7m にかけて
  // なだらかに上げ、1.6m が 1.7m を追い越すような逆転を作らない。キョダイマックス（とその元の体の計算 noBand）には掛けない。
  const BIG_BOOST = 1.25; // 1.12 → 1.25（ユーザー指定「もう少し大きく」2 回目）
  function bigBoost(p) {
    if (!p || p.noBand || p.kind === "gigantamax" || p.cat === "gigantamax") return 1;
    const m = Math.max(0.1, +p.h || 1);
    return 1 + (BIG_BOOST - 1) * U.clamp((m - 1.4) / 0.3, 0, 1);
  }
  const BODY_MIN = 12, BODY_MAX = 134;
  // 8方向シートを読むときの縮尺。GIF 由来・raw はそのまま。手作りシート（大きな原画）は GIF と同程度の細かさ（体高 56〜96px）へ。
  const sheetScale = (p, m) => (m.raw || m.src === "gif") ? 1 : Math.min(1, U.clamp(bodyArtPx(p) * 1.6, 56, 96) / Math.max(8, m.h));
  const artHeight = (m) => bodyArtPx({ h: m });
  const trainerArtPx = () => TRAINER_PX; // 1.7m の物差しそのもの（bigBoost は掛けない）
  // The pitch stays the same art size. FIELD_DRAW = 0.75 keeps 1.7m Charizard at 36px while tiny mons reach min 10px.
  // [ユーザー指定 2026-11-15] トレーナー（1.7m）が大きいので、トレーナーもモンスターも同じ比率で 2 割小さく（0.75 → 0.6）。
  // 1.7m = 29px。ドットが欠けないよう、スプライト用キャンバスの密度（render.js sprNeed）とカメラの下限で補う。
  const FIELD_DRAW = 0.6;
  const onFieldPx = (px) => Math.max(8, Math.round(px * FIELD_DRAW));

  // [2026-11-16] メガ進化・キョダイマックス（8 方向が layout "fb" のシート）の GIF は 256px × 数百コマあり、全コマを解くと 1 体で数百 MB〜1GB を
  // 使って固まる。フィールドの絵はシートから出すので、GIF は 1 コマ目だけ読む（still）。色違いは別の小さい GIF なので対象外。
  const heavyGif = (p, shiny) => { const m = (window.ARENA_DIR8 || {})[p.id]; return !!(m && m.layout === "fb" && !(shiny && p.shiny)); };
  function sprites(p, shiny) {
    const d = ROOT + p.dir + "/", s = shiny && p.shiny ? "-shiny" : "", o = { artH: bodyArtPx(p), still: heavyGif(p, shiny) };
    const frontRec = get(d + `front${s}.gif`, Object.assign({ hd: true }, o));
    if (typeof Dir8 !== "undefined" && Dir8.has && Dir8.has(p, shiny)) {
      let _fR = null, _b = null, _bR = null;
      return {
        front: frontRec,
        get frontR() { return _fR || (_fR = get(d + `front${s}-reverse.gif`, o)); },
        get back() { return _b || (_b = get(d + `back${s}.gif`, o)); },
        get backR() { return _bR || (_bR = get(d + `back${s}-reverse.gif`, o)); },
      };
    }
    return {
      front: frontRec, frontR: get(d + `front${s}-reverse.gif`, o),
      back: get(d + `back${s}.gif`, o), backR: get(d + `back${s}-reverse.gif`, o),
    };
  }
  const front = (p) => get(ROOT + p.dir + "/front.gif", { artH: bodyArtPx(p), still: heavyGif(p, false) });
  // Original-resolution animated sprites for everything outside the battle field (title, select, HUD).
  const hd = (p, shiny, view = "front") => get(ROOT + p.dir + `/${shiny && p.shiny ? view.replace(/^(front|back)/, "$1-shiny") : view}.gif`, { hd: true, still: heavyGif(p, shiny) });
  const iconUrl = (p) => ROOT + p.dir + "/" + (p.icon || "front.png");
  const stillUrl = (p) => ROOT + p.dir + "/front.png";
  const ball = () => get(ROOT + "items/poke-ball.png", { raw: true });

  function whenReady(recs, timeout = 10000) {
    start(recs);
    return new Promise((res) => {
      const t0 = performance.now();
      const tick = () => (recs.every((r) => r.ready || r.failed) || performance.now() - t0 > timeout ? res() : setTimeout(tick, 16));
      tick();
    });
  }

  return { get, defer, isDeferred, start, front, hd, artHeight, bodyArtPx, bigBoost, sheetScale, trainerArtPx, onFieldPx, TRAINER_M, frame, frameIndex, tinted, portrait, sprites, iconUrl, stillUrl, ball, whenReady, canFetch };
})();
