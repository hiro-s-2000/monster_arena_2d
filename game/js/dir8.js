// 8-way facing built from the four monsterSprite views (front = SW, front-reverse = SE, back = NE, back-reverse = NW).
// The cardinals are derived pixel-exactly from their neighbours: S / N narrow the 3/4 view (turned to face the
// camera / away), E / W lean it forward with whole-pixel row shifts (a profile in motion). Turning steps through
// the in-between rows one at a time, like the 8-direction sprites of the reference clip.
// Rows: 0 S, 1 SE, 2 E, 3 NE, 4 N, 5 NW, 6 W, 7 SW (screen space, y down).
"use strict";

const Dir8 = (() => {
  const SHEETS = window.ARENA_DIR8 || {}; // gen-1 hand-labelled 8-view stills (tools/build_dir8.js)
  const sheetCache = new Map();
  // [2026-11-16] layout "fb"（tools/build_dir8_fb.py）: メガ進化・キョダイマックス用。正面と背面の全コマを 2 枚のシート（<id>.png / <id>-b.png）に
  // 持ち、8 方向は 4 つの GIF とまったく同じ絵を使う（S・W・SW = front、SE・E = 反転、NE・N = back、NW = 反転）。変形も間引きもしない。
  // d1 / d2（正面・背面のコマごとの秒）を、ほかのシートと同じ d[8] の形にそろえる。
  const FB_VIEW = [0, 1, 1, 2, 2, 3, 0, 0]; // 行 S…SW → 0 front, 1 front-reverse, 2 back, 3 back-reverse
  for (const k in SHEETS) { const m = SHEETS[k]; if (m && m.layout === "fb" && !m.d) { const a = m.d1, b = m.d2 || m.d1; m.d = [a, a, a, b, b, b, a, a]; } }
  // コマの canvas は使うときに切り出す。1 コマ 20 万画素前後 × 数百コマ × 4 向きをためこむとメモリが尽きるので、新しい順に FB_KEEP 枚だけ持つ。
  // 外したコマは次に使うとき切り出し直す。canvas の中身を消すのは少し後（その瞬間に描いている所が持っていても壊れないように）。
  const FB_KEEP = 200, fbLive = [];
  function fbFrames(img, m, n, delays, flip) {
    const out = [];
    for (let k = 0; k < n; k++) {
      const sx = (k % m.cols) * m.cw, sy = Math.floor(k / m.cols) * m.ch;
      const fr = { delay: delays[k], _cv: null, get canvas() {
        if (!fr._cv) {
          const c = U.canvas(m.cw, m.ch), x = c.getContext("2d");
          if (flip) { x.translate(m.cw, 0); x.scale(-1, 1); }
          x.drawImage(img, sx, sy, m.cw, m.ch, 0, 0, m.cw, m.ch);
          fr._cv = c; fbLive.push(fr);
          while (fbLive.length > FB_KEEP) { const old = fbLive.shift(), oc = old._cv; old._cv = null; if (oc) setTimeout(() => { oc.width = oc.height = 0; }, 1500); }
        }
        return fr._cv;
      } };
      out.push(fr);
    }
    return out;
  }
  function sheetFb(p, m, key, bounds) {
    const recs = Array.from({ length: 8 }, (_, i) => ({ src: key + "#" + i, ready: false, failed: false, frames: null, total: 1, w: 0, h: 0, bounds: null, _tint: new Map() }));
    const imgs = [new Image(), new Image()]; recs.img = imgs[0]; recs.img2 = imgs[1];
    let loaded = 0, started = false;
    const done = () => {
      if (++loaded < 2) return;
      const sets = [fbFrames(imgs[0], m, m.n[0], m.d[0], false), fbFrames(imgs[0], m, m.n[0], m.d[0], true), fbFrames(imgs[1], m, m.n[3], m.d[3], false), fbFrames(imgs[1], m, m.n[3], m.d[3], true)];
      let bSets = null;
      if (!bounds) {
        const b0 = Pixel.bounds(imgs[0], 0, 0, m.cw, m.ch);
        const b1 = Pixel.bounds(imgs[1], 0, 0, m.cw, m.ch);
        bSets = [b0, { x: m.cw - b0.x - b0.w, y: b0.y, w: b0.w, h: b0.h }, b1, { x: m.cw - b1.x - b1.w, y: b1.y, w: b1.w, h: b1.h }];
      }
      recs.forEach((r, i) => {
        r.frames = sets[FB_VIEW[i]]; r.total = r.frames.reduce((a, f) => a + f.delay, 0); r.w = m.cw; r.h = m.ch;
        r.bounds = bounds ? bounds(i) : bSets[FB_VIEW[i]]; r.ready = true;
      });
    };
    const fail = () => recs.forEach((r) => { r.failed = true; });
    imgs.forEach((im) => { im.onload = done; im.onerror = fail; });
    const kick = () => {
      if (started || !recs.img || !recs.img2) return;
      started = true;
      const baseRel = p.dir + "/" + p.id, q = m.v ? "?v=" + m.v : "";
      if (typeof Assets !== "undefined" && Assets.loadImg) {
        Assets.loadImg(imgs[0], baseRel + ".png", q, { sheet: true });
        Assets.loadImg(imgs[1], baseRel + "-b.png", q, { sheet: true });
      } else {
        const base = "../monsterSprite/" + baseRel;
        imgs[0].src = base + ".png" + q; imgs[1].src = base + "-b.png" + q;
      }
    };
    recs._start = kick;
    recs.forEach((r) => { r._start = kick; });
    if (!(typeof Assets !== "undefined" && Assets.isDeferred && Assets.isDeferred())) kick();
    return recs;
  }
  const cache = new WeakMap(); // frame canvas → Map(transform → {img, dx, w})

  // Eight sprite records (rows S…SW) cut from monsterSprite/<dir>/<id>.png and pixelized like the GIFs, or null.
  // Hand-made sheets carry a baked idle loop (breath + limb sway, idle:1); GIF-built ones (src:"gif", tools/build_dir8_gif.js) carry
  // n[dir] animation frames stacked downwards with delays d[dir][k], and may have a "-shiny" twin.
  function has(p, shiny) {
    const m0 = SHEETS[p.id];
    if (!m0 || !p.dir) return false;
    return !!(shiny && p.shiny ? m0.shiny : m0);
  }

  function sheet(p, shiny) {
    const m0 = SHEETS[p.id];
    if (!m0 || !p.dir) return null;
    const alt = shiny && p.shiny;
    const m = alt ? m0.shiny : m0;
    if (!m) return null;
    const key = p.id + (alt ? "-shiny" : "");
    const deferred = typeof Assets !== "undefined" && Assets.isDeferred && Assets.isDeferred();
    if (sheetCache.has(key)) {
      const r = sheetCache.get(key);
      if (!deferred && r._start) r._start();
      return r;
    }
    if (m.layout === "fb") { const r = sheetFb(p, m, key, null); sheetCache.set(key, r); return r; }
    const actions8 = m.layout === "actions_8dir" || m.sheetDir === "agent-sprite-forge";
    const actions = m.layout === "actions";
    const acts = actions8 ? (m0.acts || DEF_ACTS) : null; // sheet blocks of 8 rows, top to bottom
    const totalRows = actions8 ? 8 * acts.length : 8;
    const recs = Array.from({ length: totalRows }, (_, i) => ({ src: key + "#" + i, ready: false, failed: false, frames: null, total: 1, w: 0, h: 0, bounds: null, _tint: new Map() }));
    if (acts) recs.acts = acts;
    const img = new Image(); recs.img = img;
    let started = false;
    img.onload = () => {
      const s = Assets.sheetScale(p, m), ctx = { colors: 32 };
      recs.forEach((r, idx) => {
        const actIdx = (idx / 8) | 0, i = idx % 8;
        const n = actions8 ? ((m.an && m.an[actIdx]) || 16) : (m.n ? m.n[i] : 1);
        const actDelay = actions8 && m.ad && m.ad[actIdx] ? m.ad[actIdx] : null;
        const frames = [];
        // each frame is cut and pixelized the first time it is shown (a long animation would stall the load)
        for (let k = 0; k < n; k++) {
          let cv = null;
          const sx = (actions8 || actions) ? k * m.cw : i * m.cw;
          const sy = actions8 ? idx * m.ch : actions ? (m.arow || 0) * m.ch : k * m.ch;
          frames.push({ delay: actDelay || (m.d && m.d[i] ? m.d[i][k] : 1), get canvas() {
            if (!cv) {
              const c = U.canvas(m.cw, m.ch); c.getContext("2d", s === 1 ? undefined : { willReadFrequently: true }).drawImage(img, sx, sy, m.cw, m.ch, 0, 0, m.cw, m.ch);
              try { cv = s === 1 ? c : Pixel.pixelize(c, m.cw, m.ch, s, ctx); } catch (e) { cv = c; }
            }
            return cv;
          } });
        }
        r.frames = frames; r.total = frames.reduce((a, f) => a + f.delay, 0);
        if (s === 1) {
          r.w = m.cw; r.h = m.ch;
          const sx0 = (actions8 || actions) ? 0 : i * m.cw;
          const sy0 = actions8 ? idx * m.ch : actions ? (m.arow || 0) * m.ch : 0;
          r.bounds = (actions8 && actIdx > 0 && recs[i].bounds) ? recs[i].bounds : Pixel.bounds(img, sx0, sy0, m.cw, m.ch);
        } else {
          r.w = frames[0].canvas.width; r.h = frames[0].canvas.height;
          r.bounds = (actions8 && actIdx > 0 && recs[i].bounds) ? recs[i].bounds : Pixel.bounds(frames[0].canvas);
        }
        r.ready = true;
      });
    };
    img.onerror = () => recs.forEach((r) => { r.failed = true; });
    const kick = () => {
      if (started || !recs.img) return;
      started = true;
      const rel = actions8 ? p.dir + "/agent-sprite-forge/sheet.png" : actions ? p.dir + "/actions/sheet.png" : p.dir + "/" + key + ".png";
      const q = actions8 ? "?v=" + (m.v || "asf8d4") : actions ? "?v=asf1" : (m.v ? "?v=" + m.v : m.idle ? "?v=idle1" : "");
      if (typeof Assets !== "undefined" && Assets.loadImg) Assets.loadImg(img, rel, q, { sheet: true });
      else img.src = "../monsterSprite/" + rel + q;
    };
    recs._start = kick;
    recs.forEach((r) => { r._start = kick; });
    if (!deferred) kick();
    sheetCache.set(key, recs);
    return recs;
  }
  // [2026-11-15] 使っていないシートを捨てる。シートは解いた状態で 1 枚あたり数十 MB あり、ためこむとブラウザのメモリが尽きて
  // 画面全体がちらつく（debug.html でモンスターを次々に切り替えたとき）。keep = 残す id（色違いは id + "-shiny"）の Set。
  function trim(keep, cacheMap = sheetCache) {
    for (const [key, recs] of cacheMap) {
      if (keep.has(key) || keep.has(key.replace(/-shiny$/, ""))) continue;
      if (recs.img) { recs.img.onload = recs.img.onerror = null; recs.img.src = ""; recs.img = null; }
      if (recs.img2) { recs.img2.onload = recs.img2.onerror = null; recs.img2.src = ""; recs.img2 = null; }
      recs.forEach((r) => { r.frames = null; r.ready = false; r.failed = true; r._tint.clear(); });
      cacheMap.delete(key);
    }
  }
  const tints = new WeakMap();
  const DEF_ACTS = ["idle", "walk", "attack", "hurt"];

  const rowOf = (ang) => ((Math.round((Math.PI / 2 - ang) / (Math.PI / 4)) % 8) + 8) % 8;
  const angOf = (row) => Math.PI / 2 - row * Math.PI / 4;

  // Advance the displayed row toward the facing (hysteresis + one row per step while turning).
  function update(f, dt) {
    if (f.row === undefined) { f.row = f.drow = rowOf(f.facing); f.lastLR = f.side ? -1 : 1; f.lastUD = 1; f.turnT = 0; }
    if (f.spinning) { f.turnT += dt; if (f.turnT > 0.045) { f.turnT = 0; f.drow = (f.drow + 1) % 8; } return; } // power-up twirl
    const off = Math.abs(U.angDiff(angOf(f.row), f.facing));
    if (off > Math.PI / 8 + 0.1) f.row = rowOf(f.facing);
    if (f.drow === f.row) { f.turnT = 0; return; }
    f.turnT += dt;
    if (f.turnT < 0.038) return;
    f.turnT = 0;
    const d = ((f.row - f.drow + 8) % 8);
    f.drow = (f.drow + (d <= 4 ? 1 : 7)) % 8;
    if (f.drow >= 1 && f.drow <= 3) f.lastLR = 1; else if (f.drow >= 5) f.lastLR = -1;
    if (f.drow === 7 || f.drow <= 1) f.lastUD = 1; else if (f.drow >= 3 && f.drow <= 5) f.lastUD = -1;
  }

  // Which action of an agent-sprite-forge sheet (待機・歩き・攻撃・ダメージ, + extra attacks listed in "acts") to show,
  // and the time inside it. Attack and hurt start from their first frame and play to the end even when the cast or
  // the hit-stun is shorter than the animation (a beam that is still firing keeps the attack looping); a new hit
  // interrupts an attack. A sheet with several attacks (Mega Blaziken: attack1 回し蹴り, attack2 跳び蹴り) takes the
  // next one on each cast / dash. Result: { idx, t } — t is undefined for idle / walk (they follow f.anim).
  function actionOf(f) {
    const acts = f.d8.acts || DEF_ACTS, A = f._act;
    const dur = (name) => { const i = acts.indexOf(name), rec = i >= 0 ? f.d8[i * 8] : null; return rec && rec.total > 0 ? rec.total : 0.8; };
    const hs = f.hitstun || 0, newHit = hs > 0 && hs > (f._hsPrev || 0) + 1e-3;
    f._hsPrev = hs;
    const tok = f.cast || f.dash;
    if (newHit && acts.includes("hurt")) f._act = { name: "hurt", t0: f.anim, tok: null };
    else if (tok && (!A || A.tok !== tok) && !(A && A.name === "hurt" && f.anim - A.t0 < dur("hurt") && hs > 0)) {
      const pool = acts.filter((n) => n.startsWith("attack")).sort((x, y) => (x === "attack") - (y === "attack")); // attack1, attack2, …, attack
      f._atkN = ((f._atkN === undefined ? -1 : f._atkN) + 1) % pool.length;
      f._act = { name: pool.length > 1 ? pool[f._atkN] : "attack", t0: f.anim, tok };
    }
    const C = f._act;
    if (C && !f.ko) {
      const el = f.anim - C.t0, d = dur(C.name);
      if (el < d || (C.tok && C.tok === tok)) return { idx: Math.max(0, acts.indexOf(C.name)), t: el };
      f._act = null;
    }
    if (hs > 0) return { idx: Math.max(0, acts.indexOf("hurt")), t: undefined };
    return { idx: Math.max(0, acts.indexOf(Math.hypot(f.vx, f.vy) > 25 ? "walk" : "idle")), t: undefined };
  }

  // Source GIF record + transform for the displayed row.
  function view(f) {
    const s = f.sprites, r = f.drow === undefined ? rowOf(f.facing) : f.drow;
    if (f.d8 && f.d8[r] && f.d8[r].ready) {
      if (f.d8.length >= 32) {
        const a = actionOf(f);
        const recAct = f.d8[a.idx * 8 + r];
        if (recAct && recAct.ready) return { rec: recAct, tf: "", t: a.t };
      }
      return { rec: f.d8[r], tf: "" }; // real 8-view art
    }
    switch (r) {
      case 0: return { rec: f.lastLR < 0 ? s.front : s.frontR, tf: "n" };
      case 1: return { rec: s.frontR, tf: "" };
      case 2: return { rec: f.lastUD >= 0 ? s.frontR : s.back, tf: "e+" };
      case 3: return { rec: s.back, tf: "" };
      case 4: return { rec: f.lastLR > 0 ? s.back : s.backR, tf: "n" };
      case 5: return { rec: s.backR, tf: "" };
      case 6: return { rec: f.lastUD >= 0 ? s.front : s.backR, tf: "e-" };
      default: return { rec: s.front, tf: "" };
    }
  }

  // Transformed frame: { img, ax, ay, w, h, bh } (anchor = feet centre, art px).
  function frame(rec, t, tf) {
    if (!rec.frames) return null;
    const src = Assets.frame(rec, t), b = rec.bounds;
    const ax = b.x + b.w / 2, ay = b.y + b.h;
    if (!tf) return { img: src, ax, ay, w: src.width, h: src.height, bh: b.h };
    let m = cache.get(src); if (!m) cache.set(src, (m = new Map()));
    let c = m.get(tf);
    if (!c) {
      if (tf === "n") { // facing the camera / away: 80 % width, nearest-neighbour column drop
        const w2 = Math.max(1, Math.round(src.width * 0.8)), cv = U.canvas(w2, src.height), x = cv.getContext("2d");
        x.imageSmoothingEnabled = false; x.drawImage(src, 0, 0, w2, src.height);
        c = { img: cv, dx: ax * (w2 / src.width) - ax };
      } else { // profile lean: rows above the feet slide forward by whole pixels, up to S at the head
        // (side-on bodies read slimmer: 90 % width first)
        const w2 = Math.max(1, Math.round(src.width * 0.9)), nar = U.canvas(w2, src.height), nx = nar.getContext("2d");
        nx.imageSmoothingEnabled = false; nx.drawImage(src, 0, 0, w2, src.height);
        const S = Math.max(2, Math.round(b.h * 0.15)), dir = tf === "e+" ? 1 : -1, cv = U.canvas(w2 + S, src.height), x = cv.getContext("2d");
        for (let y = 0; y < src.height; y++) {
          const sh = Math.round(S * U.clamp((ay - y) / b.h, 0, 1));
          x.drawImage(nar, 0, y, w2, 1, dir > 0 ? sh : S - sh, y, w2, 1);
        }
        c = { img: cv, dx: ax * 0.9 - ax + (dir > 0 ? 0 : S) };
      }
      m.set(tf, c);
    }
    return { img: c.img, ax: ax + c.dx, ay, w: c.img.width, h: c.img.height, bh: b.h };
  }

  // Solid-colour silhouette of any frame canvas (hit flash, status tints, afterimages). Cached.
  function tint(img, color) {
    let m = tints.get(img); if (!m) tints.set(img, (m = new Map()));
    let c = m.get(color);
    if (!c) {
      c = U.canvas(img.width, img.height);
      const x = c.getContext("2d"); x.drawImage(img, 0, 0); x.globalCompositeOperation = "source-in"; x.fillStyle = color; x.fillRect(0, 0, c.width, c.height);
      m.set(color, c);
    }
    return c;
  }

  return { has, sheet, sheetFb, FB_VIEW, trim, rowOf, angOf, update, view, frame, tint };
})();
