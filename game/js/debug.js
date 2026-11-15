// Sprite & move debug page (game/debug.html). Reuses the game's own engine: pick any Pokémon in
// monsterSprite, see its 8-direction sprites animate, and fire any move at a target on a real arena.
//
// Opened straight from the disk (file://), the browser forbids reading sprite pixels and fetching files,
// so GIFs cannot be decoded and sprites cannot be pixelized. In that case this file swaps in light
// loaders: sprites are scaled with nearest-neighbour instead of pixelized, sprite bounds come from
// data/sprite_bounds.js (tools/build_sprite_bounds.js), the 8-direction sheets (PNG) keep their full
// animation, and the WebGL post pass falls back to the plain 2D compositor. Served over http (the
// launcher), everything runs exactly as in the game.
"use strict";

(() => {
  const $ = (s) => document.querySelector(s);
  const FILE = !Assets.canFetch;
  const ROOT = "../monsterSprite/";
  const MAN = window.ARENA_DIR8 || {};
  const BOUNDS = window.ARENA_BOUNDS || { g: {}, s: {} };
  const DIRS = [["S", "南", "↓"], ["SE", "東南", "↘"], ["E", "東", "→"], ["NE", "北東", "↗"], ["N", "北", "↑"], ["NW", "北西", "↖"], ["W", "西", "←"], ["SW", "南西", "↙"]];
  // Dex meters for these bodies are 全長 (head to tail). The drawn pose is shorter:
  // span = opaque pixels ÷ median horizontal run ÷ bbox height, from each normal front.png.
  // Vertical size uses 全長 / span. Upright species are not listed. Gigantamax keeps its own scale.
  const LENGTH_SPAN = {
    ekans: 2.15, arbok: 1.42, dratini: 1.48, dragonair: 2.29, onix: 2.06, steelix: 2.31,
    gyarados: 3.63, seviper: 3.39, milotic: 4.42, rayquaza: 2.74, serperior: 2.91, dunsparce: 1.48,
  };
  function lengthSpan(mon) {
    if (!mon || mon.kind === "gigantamax" || mon.cat === "gigantamax") return 0;
    const slug = mon.slug || "";
    const keys = [slug, mon.base, slug.replace(/-(mega|gmax|hisui|alola|galar|paldea)$/, "")];
    for (const k of keys) if (k && LENGTH_SPAN[k]) return LENGTH_SPAN[k];
    return 0;
  }
  function posedArtPx(mon) {
    const span = lengthSpan(mon);
    if (!span) return 0;
    return Assets.bodyArtPx({ h: Math.max(0.1, (+mon.h || 1) / span), w: mon.w, kind: mon.kind, cat: mon.cat, base: mon.base });
  }
  const trimM = (n) => (Math.round(n * 10) / 10).toFixed(1);
  const store = { get(k, d) { try { const v = localStorage.getItem("dbg." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } }, set(k, v) { try { localStorage.setItem("dbg." + k, JSON.stringify(v)); } catch (e) { /* private mode */ } } };

  // ---- the engine sizes itself to the window: point it at the stage box instead ----------------------
  const stageEl = $("#view");
  Object.defineProperty(window, "innerWidth", { configurable: true, get: () => Math.max(1, stageEl.clientWidth) });
  Object.defineProperty(window, "innerHeight", { configurable: true, get: () => Math.max(1, stageEl.clientHeight) });

  // ---- file:// loaders ------------------------------------------------------------------------------
  function scaled(img, sx, sy, sw, sh, s) {
    const c = U.canvas(Math.max(1, Math.round(sw * s)), Math.max(1, Math.round(sh * s))), x = c.getContext("2d");
    x.imageSmoothingEnabled = false;
    x.drawImage(img, sx, sy, sw, sh, 0, 0, c.width, c.height);
    return c;
  }
  const scaleBox = (b, s) => ({ x: Math.floor(b.x * s), y: Math.floor(b.y * s), w: Math.max(1, Math.round(b.w * s)), h: Math.max(1, Math.round(b.h * s)) });
  const fileCache = new Map();
  function fileGet(rel, o) {
    const key = rel + (o.hd ? "#hd" : "@" + (o.artH || 0));
    if (fileCache.has(key)) return fileCache.get(key);
    const rec = { src: ROOT + rel, img: new Image(), ready: false, failed: false, frames: null, total: 1, w: 0, h: 0, bounds: null, _tint: new Map(), hd: !!o.hd, scale: 1 };
    rec.img.onload = () => {
      const W = rec.img.naturalWidth, H = rec.img.naturalHeight, t = BOUNDS.g[rel];
      const b = t ? { x: t[2], y: t[3], w: t[4], h: t[5] } : { x: 0, y: 0, w: W, h: H };
      if (o.hd) { rec.frames = [{ canvas: scaled(rec.img, 0, 0, W, H, 1), delay: 1 }]; rec.bounds = b; }
      else {
        const s = 1; // ゲームと同じ：GIF の元のドットのまま
        rec.scale = s; rec.frames = [{ canvas: scaled(rec.img, 0, 0, W, H, s), delay: 1 }]; rec.bounds = scaleBox(b, s);
      }
      rec.w = rec.frames[0].canvas.width; rec.h = rec.frames[0].canvas.height; rec.ready = true;
    };
    // GIF が無い（キョダイマックスは削除済み）→ 同名の front/back PNG。-reverse は PNG を左右反転（file:// なので CSS ではなく canvas で反転）
    rec.img.onerror = () => {
      if (rec._fb || !/\.gif$/i.test(rel)) { rec.failed = true; return; }
      rec._fb = true;
      const flip = /-reverse\.gif$/i.test(rel), png = rel.replace(/-reverse\.gif$/i, ".gif").replace(/\.gif$/i, ".png");
      const im = new Image();
      im.onload = () => {
        const W = im.naturalWidth, H = im.naturalHeight, s = 1;
        const c = document.createElement("canvas"); c.width = Math.max(1, Math.round(W * s)); c.height = Math.max(1, Math.round(H * s));
        const x = c.getContext("2d"); x.imageSmoothingEnabled = false;
        if (flip) { x.translate(c.width, 0); x.scale(-1, 1); }
        x.drawImage(im, 0, 0, c.width, c.height);
        rec.scale = s; rec.frames = [{ canvas: c, delay: 1 }]; rec.bounds = { x: 0, y: 0, w: c.width, h: c.height };
        rec.w = c.width; rec.h = c.height; rec.ready = true;
      };
      im.onerror = () => { rec.failed = true; };
      im.src = ROOT + png;
    };
    rec.img.src = ROOT + rel;
    fileCache.set(key, rec);
    return rec;
  }
  const sheetCache = new Map();
  function fileSheet(p, shiny) {
    const m0 = MAN[p.id];
    if (!m0 || !p.dir) return null;
    const alt = shiny && p.shiny, m = alt ? m0.shiny : m0;
    if (!m) return null;
    const key = p.id + (alt ? "-shiny" : "");
    if (sheetCache.has(key)) return sheetCache.get(key);
    if (m.layout === "fb") { // 正面・背面の 2 枚シート（メガ・キョダイ）。file:// ではドットを読めないので、絵の範囲は表（sprite_bounds）から取る
      const tb = BOUNDS.s[key];
      const r = Dir8.sheetFb(p, m, key, (i) => tb ? { x: tb[i * 4], y: tb[i * 4 + 1], w: tb[i * 4 + 2], h: tb[i * 4 + 3] } : { x: 0, y: 0, w: m.cw, h: m.ch });
      sheetCache.set(key, r); return r;
    }
    const actions8 = m.layout === "actions_8dir" || m.sheetDir === "agent-sprite-forge";
    const actions = m.layout === "actions";
    const acts = actions8 ? (m0.acts || ["idle", "walk", "attack", "hurt"]) : null;
    const totalRows = actions8 ? 8 * acts.length : 8;
    const recs = Array.from({ length: totalRows }, (_, i) => ({ src: key + "#" + i, ready: false, failed: false, frames: null, total: 1, w: 0, h: 0, bounds: null, _tint: new Map() }));
    if (acts) recs.acts = acts;
    const img = new Image(); recs.img = img;
    img.onload = () => {
      const s = Assets.sheetScale(p, m), tb0 = BOUNDS.s[key];
      const fy = m.fy || (m.ch - 1);
      const tbStale = tb0 && actions8 && ((tb0[0] + tb0[2] / 2) < m.cw * 0.25 || (m.fy && Math.abs((tb0[1] + tb0[3]) - m.fy) > 32));
      const tb = tbStale ? null : tb0;
      recs.forEach((r, idx) => {
        const actIdx = (idx / 8) | 0, i = idx % 8;
        const n = actions8 ? ((m.an && m.an[actIdx]) || 16) : (m.n ? m.n[i] : 1);
        const actDelay = actions8 && m.ad && m.ad[actIdx] ? m.ad[actIdx] : null;
        const frames = [];
        for (let k = 0; k < n; k++) {
          let cv = null;
          const sx = (actions8 || actions) ? k * m.cw : i * m.cw;
          const sy = actions8 ? idx * m.ch : actions ? (m.arow || 0) * m.ch : k * m.ch;
          frames.push({ delay: actDelay || (m.d && m.d[i] ? m.d[i][k] : 1), get canvas() { return cv || (cv = scaled(img, sx, sy, m.cw, m.ch, s)); } });
        }
        // without the table: feet on fy (or the cell's bottom line), centered horizontally, height from the manifest
        const b = tb ? { x: tb[i * 4], y: tb[i * 4 + 1], w: tb[i * 4 + 2], h: tb[i * 4 + 3] } : { x: Math.max(0, Math.round((m.cw - m.h) / 2)), y: Math.max(0, fy - m.h), w: Math.min(m.cw, m.h), h: Math.min(m.ch, m.h) };
        r.frames = frames; r.total = frames.reduce((a, f) => a + f.delay, 0);
        r.w = frames[0].canvas.width; r.h = frames[0].canvas.height; r.bounds = scaleBox(b, s); r.ready = true;
      });
    };
    img.onerror = () => recs.forEach((r) => { r.failed = true; });
    img.src = ROOT + (actions8 ? p.dir + "/agent-sprite-forge/sheet.png?v=" + (m.v || "asf8d4") : actions ? p.dir + "/actions/sheet.png" : p.dir + "/" + key + ".png");
    sheetCache.set(key, recs);
    return recs;
  }
  if (FILE) {
    Post.init = () => false; // sprite canvases are "tainted" on file://, WebGL may not upload them
    Assets.sprites = (p, shiny) => {
      const d = p.dir + "/", s = shiny && p.shiny ? "-shiny" : "", o = { artH: Assets.bodyArtPx(p) };
      return { front: fileGet(d + `front${s}.gif`, o), frontR: fileGet(d + `front${s}-reverse.gif`, o), back: fileGet(d + `back${s}.gif`, o), backR: fileGet(d + `back${s}-reverse.gif`, o) };
    };
    Assets.hd = (p, shiny, view = "front") => fileGet(p.dir + `/${shiny && p.shiny ? view.replace(/^(front|back)/, "$1-shiny") : view}.gif`, { hd: true });
    Assets.front = (p) => fileGet(p.dir + "/front.gif", { artH: Assets.bodyArtPx(p) });
    Dir8.sheet = fileSheet;
    const trim0 = Dir8.trim; Dir8.trim = (keep) => trim0(keep, sheetCache); // file:// 用のキャッシュを掃除する
  }

  // ---- state ----------------------------------------------------------------------------------------
  const QP = new URLSearchParams(location.search);
  const opt = {
    shiny: store.get("shiny", false), inf: store.get("inf", true), foeMove: store.get("foeMove", false), foeAtk: store.get("foeAtk", false),
    arena: QP.get("arena") || store.get("arena", "stadium"),
    weather: QP.has("weather") ? QP.get("weather") : store.get("weather", ""),
    foe: QP.get("foe") || store.get("foe", "snorlax"),
    speed: store.get("speed", 1), mute: store.get("mute", true),
    tr: [store.get("tr0", "trainers/hgss_0211.png"), store.get("tr1", "trainers/hgss_0000.png")], // [ユーザー指定 2026-11-15] トレーナーの画像パス（"" = ランダム）。0 = 自分側（既定レッド）、1 = 相手側（既定ヒビキ）
  };
  // えらんだトレーナー2人（side 順）。「ランダム」は、もう片方と重ならない1人をその場で引く。
  const TRAINERS = window.ARENA_TRAINERS || [];
  function pickTrainers() {
    const out = [0, 1].map((s) => TRAINERS.find((t) => t.src === opt.tr[s]) || null);
    for (let s = 0; s < 2; s++) {
      if (out[s] || !TRAINERS.length) continue;
      const pool = TRAINERS.filter((t) => t !== out[1 - s]);
      out[s] = U.pick(pool.length ? pool : TRAINERS);
    }
    return out.filter(Boolean);
  }
  let roster = [], shown = [], cur = null, battle = null, buildGen = 0;
  let paused = false, stepReq = 0, time = 0, endT = 0;
  let mode = QP.get("mode") || "walk", faceRow = 0, spinT = 0, camZ = 1; // auto: face the target · face: hold a row · spin · walk (default: roam and cast)
  let queue = [], gapT = 0, restore = null, playing = null, dodgeReq = false, slotReq = [false, false, false, false];
  let showHud = QP.get("hud") !== "0";
  let keyPane = "stage"; // 初期はパネル非表示のためステージ操作を優先する
  const held = new Set(), MOVE_KEYS = ["arrowup", "arrowdown", "arrowleft", "arrowright", "w", "a", "s", "d"];

  // the target stands still unless asked (the AI still decides, its act is replaced)
  // random waypoints over the whole pitch (keeps a walker from circling on its own half)
  const meWp = {}, foeWp = {};
  function roam(st, f, B, dt) {
    const F = B.arena.field, m = 50;
    st.t = (st.t || 0) - dt;
    if (!st.p || st.t <= 0 || U.dist(f.x, f.y, st.p.x, st.p.y) < 30) { st.p = { x: U.rand(F.x + m, F.x + F.w - m), y: U.rand(F.y + m, F.y + F.h - m) }; st.t = 6; }
    const a = Math.atan2(st.p.y - f.y, st.p.x - f.x);
    return { x: Math.cos(a) * 0.7, y: Math.sin(a) * 0.7 };
  }
  const aiAct = AI.act;
  AI.act = (f, B, dt, skill) => {
    if (B === battle && f.side === 1 && !opt.foeAtk) {
      const a = { mx: 0, my: 0, aim: null, skill: [false, false, false, false], dodge: false, swap: false, mega: false };
      if (opt.foeMove) { const v = roam(foeWp, f, B, dt); a.mx = v.x; a.my = v.y; } // wander the whole pitch
      return a;
    }
    const a = aiAct(f, B, dt, skill);
    if (B === battle && f.side === 1 && !opt.foeMove && a) { a.mx = a.my = 0; a.dodge = false; }
    return a;
  };

  // ---- battle ---------------------------------------------------------------------------------------
  async function build() {
    if (!cur) return;
    const gen = ++buildGen;
    const foe = Data.bySlug(opt.foe) || Data.bySlug("snorlax") || roster[0];
    const B = new Battle({ teams: [[{ species: cur, shiny: opt.shiny }], [{ species: foe, shiny: false }]], arenaId: opt.arena, trainers: pickTrainers(), mode: "play", cpuAttack: opt.foeAtk, skill: 0.8, duration: 300, label: "DEBUG" });
    const recs = B.teams.flat().flatMap((f) => [...f.loadRecs, Assets.hd(f.species, f.shiny)]);
    recs.push(Assets.ball(), ...B.trainerRecs);
    await Assets.whenReady(recs, 8000);
    if (gen !== buildGen) return;
    for (const f of B.teams.flat()) {
      f.measure();
      const span = lengthSpan(f.species);
      if (!span) continue;
      // measure() fitted the vertical box to the dex meters. Those meters are 全長, so
      // the posed height is shorter by span. Reach follows the same posed art size.
      const full = Assets.bodyArtPx(f.species);
      const pose = posedArtPx(f.species);
      const k = pose / Math.max(1, full);
      f.fit *= k;
      f.height = Math.max(20, f.height * k);
      f.r = U.clamp(f.r * k, 8, 110);
      f.reach = U.clamp(Math.pow(pose / Assets.trainerArtPx(), 0.5), 0.8, 1.35);
    }
    // no intro: both sides already out, as on the title exhibition (but with the normal battle camera)
    B.cfg.demo = true; B.beginExhibition(); B.cfg.demo = false;
    B.knockOut = function (f, by) { if (opt.inf) { f.hp = f.maxHp; return; } return Battle.prototype.knockOut.call(this, f, by); };
    if (opt.weather) B.setWeather(opt.weather, 99999);
    battle = B; window.BATTLE = B; queue = []; restore = null; playing = null; endT = 0; dodgeReq = false; slotReq.fill(false);
    frameCamera(B, 0, true);
    HUD.bind(B);
    if (showHud) HUD.show(); else HUD.hide();
    HUD.update(B, 0);
  }

  // Exactly the battle camera: follows the shown monster (side 0) with a small lead toward the target, 100% = GIF size by default.
  function frameCamera(B, dt, snap) {
    Renderer.updateCamera(B, dt, snap);
    camZ = Renderer.cam.zoom;
  }

  function tick(dt) {
    const B = battle;
    time += dt;
    const me = B.active[0];
    const act = { mx: 0, my: 0, aim: null, skill: [false, false, false, false], dodge: false, swap: false, mega: false };
    if (me && !me.ko) {
      me.energy = me.maxEnergy; me.cds.fill(0);
      if (mode === "spin") { spinT += dt; if (spinT > 0.7) { spinT = 0; faceRow = (faceRow + 1) % 8; } }
      // [ユーザー指定 2026-11-15] 方向キー（または WASD）を押している間は、その方向へ自分で動かす（自動の「あるく」より優先）
      const kx = (held.has("arrowright") || held.has("d") ? 1 : 0) - (held.has("arrowleft") || held.has("a") ? 1 : 0);
      const ky = (held.has("arrowdown") || held.has("s") ? 1 : 0) - (held.has("arrowup") || held.has("w") ? 1 : 0);
      if (kx || ky) { const l = Math.hypot(kx, ky); act.mx = kx / l; act.my = ky / l; meWp.p = null; }
      else if (mode === "walk") {
        const v = roam(meWp, me, B, dt); act.mx = v.x; act.my = v.y;
      }
      else if (mode === "face" || mode === "spin") act.aim = Dir8.angOf(faceRow);
      gapT -= dt;
      let autoMove = false;
      if (kx || ky) gapT = Math.max(gapT, 0.6); // 自分で動かしている間は、自動の技で足を止めない
      if (mode === "walk" && !queue.length && gapT <= 0 && me.canAct && !me.busy && !me.cast) {
        const pool = ((cur && cur.moves) || []).filter(Boolean);
        if (pool.length) { queue.push(pool[(Math.random() * pool.length) | 0]); autoMove = true; }
      }
      if (dodgeReq) { act.dodge = true; dodgeReq = false; }
      for (let i = 0; i < 4; i++) {
        if (slotReq[i]) {
          slotReq[i] = false;
          if (me.moves[i] && me.canAct && !me.busy) {
            act.skill[i] = true; playing = me.moves[i].slug; gapT = Math.max(gapT, 0.9);
            if (mode !== "walk") mode = "auto";
            syncMode();
          }
        }
      }
      if (queue.length && gapT <= 0 && me.canAct && !me.busy) {
        const slug = queue.shift();
        restore = me.moves[0];
        me.moves[0] = Ability.moveType(me, Moves.spec(slug));
        act.skill[0] = true; act.aim = null; playing = slug; gapT = autoMove ? 1.4 : 0.9;
        if (mode !== "walk") mode = "auto";
        syncMode();
      }
    }
    B.cpuAttack = opt.foeAtk;
    B.update(dt, act);
    if (opt.inf && B.phase === "fight" && B.time < 60) B.time = 300;
    if (restore && me) { me.moves[0] = restore; restore = null; }
    if (playing && me && !me.cast && !queue.length && gapT <= 0) playing = null;
    frameCamera(B, dt, false);
    if (B.phase === "end" && (endT += dt) > 2.5) build();
    markPlaying();
  }

  // ---- main loop --------------------------------------------------------------------------------------
  let last = performance.now();
  function loop(now) {
    const rdt = Math.min(0.05, (now - last) / 1000); last = now;
    if (battle) {
      if (!paused) tick(rdt * opt.speed);
      else while (stepReq > 0) { stepReq--; tick(1 / 60); }
      HUD.update(battle, paused ? 0 : rdt * opt.speed);
      Renderer.draw(battle, time);
    }
    Strip.update(rdt * opt.speed);
    requestAnimationFrame(loop);
  }

  // ---- list -------------------------------------------------------------------------------------------
  const genOf = (p) => (p.dir || "").split("/")[0];
  // Whole-word form search. "メガ" stays a name search (メガニウムなど) and is not this list.
  const FORM_WORDS = { "通常": "gen", "通常フォルム": "gen", "メガ進化": "mega", "メガシンカ": "mega", "キョダイマックス": "gigantamax", "キョダイ": "gigantamax", "リージョン": "regional", "リージョンフォーム": "regional" };
  function renderList() {
    const raw = $("#q").value.trim(), q = raw.toLowerCase(), g = $("#gen").value, only8 = $("#only8").checked, picked = $("#form").value, typed = FORM_WORDS[raw] || "";
    shown = roster.filter((p) => {
      if (g && genOf(p) !== g) return false;
      if (only8 && !(MAN[p.id] && MAN[p.id].layout !== "actions")) return false;
      if (picked && p.kind !== picked) return false;
      if (typed) return p.kind === typed;
      if (!q) return true;
      return p.search.includes(q) || String(p.dex).padStart(4, "0").includes(q) || p.id.includes(q);
    });
    const ul = $("#list"); ul.textContent = "";
    const frag = document.createDocumentFragment();
    for (const p of shown) {
      const li = document.createElement("li"); li.dataset.slug = p.slug;
      const kindTag = p.kind !== "gen" ? `<span class="tag">${{ mega: "メガ", regional: "リージョン", gigantamax: "キョダイ" }[p.kind] || p.kind}</span>` : "";
      const man = MAN[p.id];
      const sheetTag = man ? (man.layout === "actions_8dir" ? '<span class="tag d8">8方向アクション</span>' : man.layout === "actions" ? '<span class="tag d8">アクション</span>' : '<span class="tag d8">8方向</span>') : "";
      li.innerHTML = `<img loading="lazy" alt="" src="${ROOT + p.dir}/front.png"><span class="nm">${String(p.dex).padStart(4, "0")} ${p.jp}<small>${p.slug}</small></span><span>${kindTag}${sheetTag}</span>`;
      if (cur === p) li.classList.add("sel");
      frag.appendChild(li);
    }
    ul.appendChild(frag);
    $("#count").textContent = `${shown.length} / ${roster.length} 体`;
  }
  $("#list").addEventListener("click", (e) => { const li = e.target.closest("li"); if (li) select(Data.bySlug(li.dataset.slug)); });

  function select(p, keepHash) {
    if (!p) return;
    cur = p;
    document.querySelectorAll("#list li.sel").forEach((li) => li.classList.remove("sel"));
    const li = document.querySelector(`#list li[data-slug="${CSS.escape(p.slug)}"]`);
    if (li) { li.classList.add("sel"); li.scrollIntoView({ block: "nearest" }); }
    if (!keepHash) history.replaceState(null, "", "#" + p.slug);
    const m = MAN[p.id];
    const sheetNote = m && m.layout === "actions_8dir" ? ` · 8方向×${(m.acts || [0, 0, 0, 0]).length}アクション（agent-sprite-forge/sheet.png）` : m && m.layout === "actions" ? " · アクションシート（actions/sheet.png）" : m ? ` · 8方向シート（${m.src === "gif" ? "GIFから作成" : m.src === "png" ? "静止画から作成" : "手作り"}）` : " · 8方向シートなし（GIFを変形）";
    const span = lengthSpan(p);
    const meters = span ? `全長 ${trimM(p.h)}m · たて ${trimM(p.h / span)}m · ${p.w}kg` : `${p.h}m ${p.w}kg`;
    $("#info").innerHTML = `<b>${p.jp}</b> <span class="sub">${p.name} · No.${String(p.dex).padStart(4, "0")} · ${p.types.map(Data.typeName).join(" / ")} · ${meters}</span><div class="sub">${p.dir}${sheetNote}</div>`;
    Strip.show(p, opt.shiny);
    renderMoves();
    build();
  }

  // ---- moves ------------------------------------------------------------------------------------------
  const KIND_JP = { proj: "とびどうぐ", aoe: "まわりこうげき", erupt: "ふきだし", spread: "かくさん", leap: "とびかかり", uplift: "もちあげ", psybreak: "サイコブレイク", teleport: "テレポート", projectile: "とびどうぐ", beam: "ビーム", melee: "せっきん", dash: "とっしん", zone: "はんい", strike: "らっか", buff: "のうりょくアップ", heal: "かいふく", shield: "まもり", weather: "てんき", status: "じょうたい", smoke: "えんまく", dig: "あなをほる", grab: "つかみ", spiral: "らせん" };
  function moveButton(slug) {
    const s = Moves.spec(slug), col = Data.typeColor(s.type);
    const b = document.createElement("button"); b.className = "mv"; b.dataset.slug = slug;
    b.innerHTML = `<span>${s.jp || slug}</span><span class="t" style="background:${col[0]}">${Data.typeName(s.type)}</span><span class="meta">${KIND_JP[s.kind] || s.kind || "-"} · いりょく ${s.power || "-"} · CD ${s.cd != null ? s.cd.toFixed(1) : "-"}s · ${slug}</span>`;
    b.addEventListener("click", () => fire([slug]));
    return b;
  }
  function section(title, slugs, seq) {
    const box = $("#moves");
    const h = document.createElement("h2"); h.textContent = title; box.appendChild(h);
    if (seq && slugs.length > 1) { const row = document.createElement("div"); row.className = "row"; const b = document.createElement("button"); b.textContent = "▶ じゅんばんに全部"; b.addEventListener("click", () => fire(slugs)); row.appendChild(b); box.appendChild(row); row.style.marginBottom = "6px"; }
    for (const s of slugs) box.appendChild(moveButton(s));
  }
  function renderMoves() {
    const p = cur, box = $("#moves"); box.textContent = "";
    section("そうびちゅうの技（1〜4キー）", p.moves || [], true);
    const pool = (p.pool || []).filter((s) => !(p.moves || []).includes(s));
    if (pool.length) section("おぼえる技", pool, true);
    const util = (p.util || []).filter((s) => !(p.moves || []).includes(s));
    if (util.length) section("ほじょ技", util, true);
    // [ユーザー指定 2026-11-15] そのモンスターのタイプの技は、おぼえる・おぼえないに関係なく全部ためせるようにする。
    const allSlugs = [...new Set([...Object.keys(window.ARENA_MOVES || {}), "dig", "dive", "whirlpool"])];
    for (const ty of p.types || []) {
      const list = allSlugs.filter((k) => Moves.spec(k).type === ty).sort((a, b) => (Moves.spec(b).power || 0) - (Moves.spec(a).power || 0) || a.localeCompare(b));
      if (list.length) section(`${Data.typeName(ty)}タイプの技 ぜんぶ（${list.length}）`, list, true);
    }
    const h = document.createElement("h2"); h.textContent = "すべての技からさがす"; box.appendChild(h);
    const inp = document.createElement("input"); inp.type = "search"; inp.id = "mvsearch"; inp.placeholder = "技の名前 / slug / タイプ"; box.appendChild(inp);
    const res = document.createElement("div"); box.appendChild(res);
    const all = allSlugs;
    inp.addEventListener("input", () => {
      const q = inp.value.trim().toLowerCase(); res.textContent = "";
      if (!q) return;
      const hits = all.filter((k) => { const m = (window.ARENA_MOVES || {})[k] || {}; return k.includes(q) || (m.jp || "").includes(q) || (m.n || "").toLowerCase().includes(q) || Data.typeName(m.t || "").includes(q); }).slice(0, 60);
      for (const k of hits) res.appendChild(moveButton(k));
      if (!hits.length) res.textContent = "みつかりません";
    });
    markPlaying();
  }
  function fire(slugs) { Audio.unlock(); queue = slugs.slice(); gapT = 0; if (paused) setPaused(false); }
  let lastMark = null;
  function markPlaying() {
    const me = battle && battle.active[0];
    const now = (me && me.cast && me.cast.spec.slug) || playing;
    if (now === lastMark) return;
    lastMark = now;
    document.querySelectorAll(".mv.playing").forEach((b) => b.classList.remove("playing"));
    if (now) document.querySelectorAll(`.mv[data-slug="${CSS.escape(now)}"]`).forEach((b) => b.classList.add("playing"));
  }

  // ---- 8-direction strip (draws the sheet straight from the PNG: works on file:// too) ----------------
  const ACT_ROWS = [["idle", "待機"], ["walk", "歩き"], ["attack", "攻撃"], ["hurt", "ダメージ"]];
  // extra actions a sheet may list in its manifest "acts" (shown after the four, same cells, same 4-column grid)
  const ACT_JP = { idle: "待機", walk: "歩き", attack: "攻撃", hurt: "ダメージ", attack1: "攻撃1", attack2: "攻撃2" };
  const Strip = (() => {
    let img2 = null; // layout "fb" の背面シート
    let p = null, m = null, img = null, cells = [], t = [], fk = 0, playingS = true, zoom = 0, acts = false, rows = [], fit = null, fits = [];
    const strip = $("#strip");
    const official = $("#official");
    const dirsBody = $("#dirs-body");
    const setSrc = (s) => { const el = $("#dsrc"); el.textContent = s; el.title = s; };
    let gifGen = 0, gifFrame = null, gifFallback = null, view = { z: 1, w: 1, h: 1 };
    const OFFICIAL_GIFS = [["front.gif", "正面"], ["front-reverse.gif", "正面反転"], ["back.gif", "背面"], ["back-reverse.gif", "背面反転"]];
    function showOfficial() {
      const gen = ++gifGen;
      gifFrame = null;
      gifFallback = null;
      official.textContent = "";
      official.hidden = true;
      if (!p) return;
      const dir = p.dir;
      let settled = 0;
      const markSettled = () => {
        if (gen !== gifGen) return;
        settled++;
        if (settled >= OFFICIAL_GIFS.length && !gifFrame && !gifFallback) {
          gifFallback = { w: 96, h: 96 };
          layout();
        }
      };
      for (const [file, label] of OFFICIAL_GIFS) {
        const fig = document.createElement("figure");
        fig.hidden = true;
        official.appendChild(fig);
        const im = new Image();
        im.alt = label;
        im.onload = () => {
          if (gen !== gifGen || !im.naturalWidth) return;
          const z = im.naturalWidth < 80 ? 2 : 1;
          im.dataset.bw = String(im.naturalWidth * z);
          im.dataset.bh = String(im.naturalHeight * z);
          im.style.width = im.naturalWidth * z + "px";
          const cap = document.createElement("figcaption");
          cap.textContent = file;
          fig.appendChild(im);
          fig.appendChild(cap);
          fig.hidden = false;
          official.hidden = false;
          frameFrom(im.naturalWidth * z, im.naturalHeight * z);
        };
        // GIF が無いとき（例: キョダイマックスの GIF は 2026-11-04 にユーザー指示で削除）は No Image の枠を出す
        im.onerror = () => {
          if (gen !== gifGen) return;
          const cap = document.createElement("figcaption");
          cap.textContent = file;
          const ni = noImage();
          ni.dataset.bw = "96";
          ni.dataset.bh = "96";
          fig.appendChild(ni);
          fig.appendChild(cap);
          fig.hidden = false;
          official.hidden = false;
          layout();
          // [ユーザー指定 2026-11-15] GIF が無くても8方向・アクションの枠は GIF と同じ大きさにする（シートの原寸で巨大表示しない）。
          // 同名の .png の寸法を GIF の代わりに使い、4つとも無ければ No Image 枠と同じ 96px。
          const still = new Image();
          still.onload = () => { if (gen !== gifGen) return; const z = still.naturalWidth < 80 ? 2 : 1; frameFrom(still.naturalWidth * z, still.naturalHeight * z); };
          still.onerror = markSettled;
          still.src = ROOT + dir + "/" + file.replace(/\.gif$/, ".png");
        };
        im.src = ROOT + dir + "/" + file;
      }
    }
    function frameFrom(w, h) {
      gifFrame = gifFrame ? { w: Math.min(gifFrame.w, w), h: Math.min(gifFrame.h, h) } : { w, h };
      layout();
    }
    function noImage() {
      const d = document.createElement("div");
      d.className = "noimg";
      d.textContent = "No Image";
      return d;
    }
    // One scale for every facing, so a taller crop does not resize the cell.
    function largestCrop() {
      const nDir = m && (m.layout === "actions_8dir" || m.sheetDir === "agent-sprite-forge") ? 8 : 1;
      let w = 1, h = 1;
      for (let d = 0; d < nDir; d++) {
        const b = fitFor(d);
        if (b.w > w) w = b.w;
        if (b.h > h) h = b.h;
      }
      return { w, h };
    }
    // 8方向・アクションともに、表示枠を front.gif / front-reverse.gif / back.gif / back-reverse.gif に揃え、
    // 自動時は 30vh の固定枠内に2段（#strip + #official）が収まるよう等倍以下で縮小フィットする。
    // ×1/×2/×3 指定時のみシート／GIF のピクセル倍率で拡大表示する。
    function officialMaxH(base) {
      let maxH = 0;
      official.querySelectorAll("img").forEach((im) => {
        const bh = +(im.dataset.bh || 0);
        if (bh > maxH) maxH = bh;
      });
      official.querySelectorAll(".noimg").forEach((d) => {
        const bh = +(d.dataset.bh || 96);
        if (bh > maxH) maxH = bh;
      });
      return maxH || (base ? base.h : 96);
    }
    function autoRowScale(base) {
      if (zoom || !dirsBody || !base || !(base.h > 0)) return 1;
      const hostH = dirsBody.clientHeight;
      if (!(hostH > 0)) return 1;
      // 枠線・ラベル・段間ギャップの合計（#strip 20px + #official 32px + gap 6px + 余裕 10px = 68px）
      const availH = Math.max(48, hostH - 68);
      const needH = base.h + officialMaxH(base);
      const s = needH > availH ? availH / needH : 1;
      return Math.max(0.25, Math.min(1, s));
    }
    function applyOfficialScale(s) {
      const k = zoom ? 1 : s;
      official.querySelectorAll("img").forEach((im) => {
        const bw = +(im.dataset.bw || 0);
        if (bw > 0) im.style.width = Math.max(1, Math.round(bw * k)) + "px";
      });
      official.querySelectorAll(".noimg").forEach((d) => {
        const bw = +(d.dataset.bw || 96), bh = +(d.dataset.bh || 96);
        d.style.width = Math.max(1, Math.round(bw * k)) + "px";
        d.style.height = Math.max(1, Math.round(bh * k)) + "px";
      });
    }
    function gifSizedBox(srcW, srcH, rowScale = 1) {
      if (!(srcW > 0) || !(srcH > 0)) return null;
      if (zoom) return { z: zoom, w: Math.max(1, Math.round(srcW * zoom)), h: Math.max(1, Math.round(srcH * zoom)) };
      const base = gifFrame || gifFallback;
      if (!base) return null;
      const bw = Math.max(1, Math.round(base.w * rowScale)), bh = Math.max(1, Math.round(base.h * rowScale));
      const fit = Math.min(bw / srcW, bh / srcH);
      return { z: fit, w: bw, h: bh };
    }
    function applyCanvas(cv) {
      if (cv.width !== view.w || cv.height !== view.h) { cv.width = view.w; cv.height = view.h; }
      const w = view.w + "px", h = view.h + "px";
      if (cv.style.width !== w) cv.style.width = w;
      if (cv.style.height !== h) cv.style.height = h;
    }
    function show(sp, shiny) {
      p = sp; strip.textContent = ""; strip.classList.remove("acts"); strip.style.gridTemplateColumns = ""; cells = []; rows = []; acts = false; t = new Array(8).fill(0); fk = 0; img = null; fit = null; fits = [];
      showOfficial();
      const m0 = MAN[p.id], alt = shiny && p.shiny;
      const showActions = !!(m0 && (m0.layout === "actions" || m0.layout === "actions_8dir"));
      m = m0 ? (alt ? m0.shiny : m0) : null;
      if (showActions) {
        $("#dtitle").textContent = "アクションスプライト";
        const base = m0 && (m0.layout === "actions" || m0.layout === "actions_8dir") ? m0 : null;
        if (!base) { m = null; setSrc("gen2 のアクションシートがありません"); updateFrameLabel(); return; }
        m = base; acts = true; strip.classList.add("acts"); t = [0];
        const actList = m.acts ? m.acts.map((id) => [id, ACT_JP[id] || id]) : ACT_ROWS;
        const counts = (m.an && m.an.length === actList.length) ? m.an : [16, 16, 16, 12];
        const is8Dir = base.layout === "actions_8dir" || base.sheetDir === "agent-sprite-forge";
        actList.forEach(([id, jp], ri) => {
          const n = counts[ri];
          // [USER-LOCKED LAYOUT] 待機・歩き・攻撃・ダメージ（および追加アクションの 攻撃1・攻撃2 等）は縦並び・3段折り返し禁止。
          // 1動作 = 1セル（.cell.play）で常に1行横並びにし、下の #official 行と合わせて2段・枠高さ 30vh 固定とする。詳細: md/AGENTS.md
          const play = document.createElement("div"); play.className = "cell play";
          const cv = document.createElement("canvas"); play.appendChild(cv);
          play.insertAdjacentHTML("beforeend", `<span class="lb"><span>${jp}</span><i>${n}コマ</i></span>`);
          strip.appendChild(play); rows.push({ n, canvas: cv, ri });
        });
        img = new Image();
        img.onload = () => layout();
        img.src = ROOT + p.dir + (is8Dir ? "/agent-sprite-forge/sheet.png?v=" + (m.v || "asf8d4") : "/actions/sheet.png?v=asf1");
        const srcLabel = is8Dir ? "agent-sprite-forge/sheet.png（8方向）" : "actions/sheet.png";
        setSrc(`${srcLabel} · ${actList.map(([, jp], i) => `${jp} ${counts[i]}`).join(" / ")}コマ · ${base.cw}×${base.ch}px/コマ${alt ? " · 色違いシートは無いので通常色" : ""}`);
        updateFrameLabel();
        return;
      }
      $("#dtitle").textContent = "8方向スプライト";
      const s = alt ? "-shiny" : "";
      DIRS.forEach(([k, jp], i) => {
        const c = document.createElement("div"); c.className = "cell"; c.title = `${jp}（${k}）を向かせる`;
        c.innerHTML = `<span class="lb"><span>${jp}</span><i>${k}</i></span>`;
        c.addEventListener("click", () => { faceRow = i; mode = "face"; syncMode(); });
        strip.appendChild(c); cells.push(c);
      });
      if (m) {
        img = new Image();
        img.onload = () => {
          cells.forEach((c, i) => {
            const cv = document.createElement("canvas"); c.prepend(cv);
            if (m.approx && m.approx.includes(DIRS[i][0])) c.insertAdjacentHTML("beforeend", '<span class="ap" title="正面・背面のGIFを変形して作った向き">変形</span>');
          });
          layout();
        };
        img.src = ROOT + p.dir + "/" + p.id + s + ".png" + (m.v ? "?v=" + m.v : "");
        img2 = null;
        if (m.layout === "fb") { img2 = new Image(); img2.onload = () => layout(); img2.src = ROOT + p.dir + "/" + p.id + "-b.png" + (m.v ? "?v=" + m.v : ""); }
        const n = m.n ? Math.max(...m.n) : 1;
        setSrc(`${p.id}${s}.png · ${m.layout === "fb" ? "GIFと同じコマ・同じ大きさ（正面・背面の全コマ）" : m.src === "gif" ? "GIFから作ったシート" : m.src === "png" ? "静止画（front.png / back.png）から作ったシート" : m.idle ? "手作りシート（待機アニメ付き）" : "手作りシート"} · ${m.cw}×${m.ch}px/コマ · 最大${n}コマ`);
      } else {
        // no sheet: the same rules the game uses at runtime (Dir8.view / Dir8.frame), as CSS on the GIFs
        const PLAN = [["frontR", "n"], ["frontR", ""], ["frontR", "e+"], ["back", ""], ["back", "n"], ["backR", ""], ["front", "e-"], ["front", ""]];
        const file = { front: `front${s}.gif`, frontR: `front${s}-reverse.gif`, back: `back${s}.gif`, backR: `back${s}-reverse.gif` };
        cells.forEach((c, i) => {
          const [v, tf] = PLAN[i], im = document.createElement("img");
          im.src = ROOT + p.dir + "/" + file[v]; im.dataset.tf = tf; im.alt = "";
          im.onload = layout; im.onerror = () => { if (im.isConnected) { const ni = noImage(); c.prepend(ni); im.remove(); layout(); } }; c.prepend(im);
          if (tf) c.insertAdjacentHTML("beforeend", '<span class="ap" title="ゲーム内で変形して作る向き">変形</span>');
        });
        setSrc("8方向シートなし — ゲームと同じ規則で正面・背面GIFを変形して表示");
      }
      updateFrameLabel();
    }
    // One fixed crop for every action cell. Hug the monster body (rows and columns
    // that are part of the silhouette), plus 2px. A thin beam in a few frames does not
    // stretch the frame. Falls back to every opaque pixel, then to the full cell.
    function shownDir() {
      const is8 = m && (m.layout === "actions_8dir" || m.sheetDir === "agent-sprite-forge");
      if (!is8) return 0;
      const me = battle && battle.active[0];
      if (mode === "face" || mode === "spin") return faceRow;
      if (me && me.drow !== undefined) return me.drow;
      return 7;
    }
    function measureFit(dir) {
      const full = { x: 0, y: 0, w: m.cw, h: m.ch, dir };
      const saved = typeof ACTION_FIT !== "undefined" && ACTION_FIT[p.id] && ACTION_FIT[p.id][dir];
      if (saved) return { x: saved[0], y: saved[1], w: saved[2], h: saved[3], dir };
      try {
        const is8 = m.layout === "actions_8dir" || m.sheetDir === "agent-sprite-forge";
        const stride = is8 ? 8 : 1;
        const d0 = is8 ? dir : 0, d1 = d0 + 1;
        const c = document.createElement("canvas");
        c.width = img.naturalWidth; c.height = img.naturalHeight;
        const x = c.getContext("2d", { willReadFrequently: true });
        x.drawImage(img, 0, 0);
        const data = x.getImageData(0, 0, c.width, c.height).data;
        const W = c.width;
        const ch = m.ch, cw = m.cw;
        const rowPeak = new Uint16Array(ch), colPeak = new Uint16Array(cw);
        const rowFr = new Uint16Array(ch), colFr = new Uint16Array(cw);
        const rc = new Uint16Array(ch), cc = new Uint16Array(cw);
        let nframes = 0, minX = cw, minY = ch, maxX = -1, maxY = -1;
        rows.forEach((row) => {
          for (let d = d0; d < d1; d++) {
            const sy = (row.ri * stride + d) * ch;
            if (sy + ch > img.naturalHeight) continue;
            for (let k = 0; k < row.n; k++) {
              const sx = k * cw;
              if (sx + cw > W) continue;
              rc.fill(0); cc.fill(0);
              let any = false;
              for (let y = 0; y < ch; y++) {
                let i = ((sy + y) * W + sx) * 4 + 3;
                for (let px = 0; px < cw; px++, i += 4) {
                  if (data[i] > 0) {
                    any = true; rc[y]++; cc[px]++;
                    if (px < minX) minX = px;
                    if (px > maxX) maxX = px;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                  }
                }
              }
              if (!any) continue;
              nframes++;
              for (let y = 0; y < ch; y++) if (rc[y]) { rowFr[y]++; if (rc[y] > rowPeak[y]) rowPeak[y] = rc[y]; }
              for (let px = 0; px < cw; px++) if (cc[px]) { colFr[px]++; if (cc[px] > colPeak[px]) colPeak[px] = cc[px]; }
            }
          }
        });
        if (maxX < 0) return full;
        let rp = 1, cp = 1;
        for (let y = 0; y < ch; y++) if (rowPeak[y] > rp) rp = rowPeak[y];
        for (let px = 0; px < cw; px++) if (colPeak[px] > cp) cp = colPeak[px];
        const rthr = Math.max(6, Math.round(0.18 * rp));
        const cthr = Math.max(6, Math.round(0.18 * cp));
        const fthr = Math.max(6, Math.round(0.04 * nframes));
        let x0 = -1, y0 = -1, x1 = -1, y1 = -1;
        for (let y = 0; y < ch; y++) if (rowPeak[y] >= rthr || rowFr[y] >= fthr) { if (y0 < 0) y0 = y; y1 = y; }
        for (let px = 0; px < cw; px++) if (colPeak[px] >= cthr || colFr[px] >= fthr) { if (x0 < 0) x0 = px; x1 = px; }
        if (x0 < 0 || y0 < 0) { x0 = minX; y0 = minY; x1 = maxX; y1 = maxY; }
        const pad = 2;
        x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad);
        x1 = Math.min(cw - 1, x1 + pad); y1 = Math.min(ch - 1, y1 + pad);
        return { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1, dir };
      } catch (e) { return full; }
    }
    function fitFor(dir) {
      if (!fits[dir]) fits[dir] = measureFit(dir);
      return fits[dir];
    }
    function layout() {
      if (!p) return;
      const base = gifFrame || gifFallback;
      const rowScale = autoRowScale(base || { w: 96, h: 96 });
      applyOfficialScale(rowScale);
      if (acts && m && img && img.complete) {
        fit = fitFor(shownDir());
        const src = largestCrop();
        const box = gifSizedBox(src.w, src.h, rowScale);
        if (!box) return;
        view = box;
        rows.forEach((row) => applyCanvas(row.canvas));
        draw();
        return;
      }
      if (m && img && img.complete) {
        // layout "fb": 枠は GIF の枠そのもの（m.g）。GIF の欄とまったく同じ大きさ・同じ位置に出す
        const box = m.layout === "fb" ? gifSizedBox(Math.max(m.g[0], m.g[4]), Math.max(m.g[1], m.g[5]), rowScale) : gifSizedBox(m.cw, m.ch, rowScale);
        if (!box) return;
        view = box;
        strip.style.gridTemplateColumns = "";
        cells.forEach((c) => {
          const cv = c.querySelector("canvas"); if (!cv) return;
          applyCanvas(cv);
          // layout "fb": 絵は GIF の元の画素数のまま持ち、表示の大きさだけ CSS で合わせる（GIF の <img> と同じ扱い。高精細な画面で粗くならない）。
          // ×2・×3 のときは今までどおり拡大した画素数で描く。
          if (m.layout === "fb" && !zoom) { const gw = Math.max(m.g[0], m.g[4]), gh = Math.max(m.g[1], m.g[5]); if (cv.width !== gw || cv.height !== gh) { cv.width = gw; cv.height = gh; } }
          c.style.width = ""; c.style.height = "";
        });
        draw();
      }
      cells.forEach((c) => {
        const ni = c.querySelector(".noimg");
        if (ni) {
          const ref = base || { w: 96, h: 96 };
          const k = zoom ? 1 : rowScale;
          ni.style.width = Math.max(1, Math.round(ref.w * k)) + "px";
          ni.style.height = Math.max(1, Math.round(ref.h * k)) + "px";
        }
        const im = c.querySelector("img"); if (!im || !im.naturalWidth) return;
        const box = gifSizedBox(im.naturalWidth, im.naturalHeight, rowScale);
        if (!box) return;
        const tf = im.dataset.tf;
        im.style.width = Math.max(1, Math.round(im.naturalWidth * box.z)) + "px";
        c.style.width = zoom ? "" : Math.max(56, box.w + 2) + "px";
        c.style.height = zoom ? "" : (box.h + 20) + "px";
        im.style.transformOrigin = "50% 100%";
        im.style.transform = tf === "n" ? "scaleX(0.8)" : tf === "e+" ? "scaleX(0.9) skewX(-8.5deg)" : tf === "e-" ? "scaleX(0.9) skewX(8.5deg)" : "";
      });
    }
    function frameAt(i, tt) {
      const n = m.n ? m.n[i] : 1;
      if (n < 2) return 0;
      if (!playingS) return fk % n;
      const d = m.d[i], tot = d.reduce((a, b) => a + b, 0);
      let k = tt % tot;
      for (let j = 0; j < n; j++) { if (k < d[j]) return j; k -= d[j]; }
      return n - 1;
    }
    // Idle follows the sheet delays (m.d). Walk, attack and hurt each loop in
    // about 1.15s so a finer row still reads as motion. Layout stays 4 columns.
    function actFrame(row) {
      if (!playingS) return fk;
      const t0 = t[0] || 0;
      if (row.ri === 0 && m.d && m.d[0] && m.d[0].length === row.n) {
        const d = m.d[0], tot = d.reduce((a, b) => a + b, 0) || 0.1;
        let k = t0 % tot;
        for (let j = 0; j < row.n; j++) { if (k < d[j]) return j; k -= d[j]; }
        return row.n - 1;
      }
      if (m.ad && m.ad[row.ri]) return Math.floor(t0 / m.ad[row.ri]); // per-action frame time (procedural mega sheets)
      return Math.floor(t0 / (1.15 / Math.max(2, row.n)));
    }
    function draw() {
      if (!m || !img || !img.complete) return;
      if (acts) {
        const is8Dir = m.layout === "actions_8dir" || m.sheetDir === "agent-sprite-forge";
        const me = battle && battle.active[0];
        const dRow = is8Dir ? ((mode === "face" || mode === "spin") ? faceRow : (me && me.drow !== undefined ? me.drow : 7)) : 0;
        rows.forEach((row) => {
          const k = ((actFrame(row) % row.n) + row.n) % row.n;
          const cv = row.canvas, x = cv.getContext("2d");
          const sy = is8Dir ? (row.ri * 8 + dRow) * m.ch : row.ri * m.ch;
          const box = fit || { x: 0, y: 0, w: m.cw, h: m.ch };
          const dw = Math.max(1, Math.round(box.w * view.z)), dh = Math.max(1, Math.round(box.h * view.z));
          const dx = Math.round((cv.width - dw) / 2), dy = cv.height - dh;
          x.imageSmoothingEnabled = false;
          x.clearRect(0, 0, cv.width, cv.height);
          x.drawImage(img, k * m.cw + box.x, sy + box.y, box.w, box.h, dx, dy, dw, dh);
        });
        return;
      }
      cells.forEach((c, i) => {
        const cv = c.querySelector("canvas"); if (!cv) return;
        const x = cv.getContext("2d"), k = frameAt(i, t[i]);
        if (m.layout === "fb") {
          const v = Dir8.FB_VIEW[i], back = v >= 2, flip = v % 2 === 1, src = back ? img2 : img, z = zoom ? view.z : cv.width / Math.max(m.g[0], m.g[4]);
          x.setTransform(1, 0, 0, 1, 0, 0); x.clearRect(0, 0, cv.width, cv.height);
          if (!src || !src.complete || !src.naturalWidth) return;
          const gw = m.g[back ? 4 : 0], gh = m.g[back ? 5 : 1], ox = m.g[back ? 6 : 2], oy = m.g[back ? 7 : 3];
          // GIF の枠（gw×gh）を canvas の中央・下ぞろえに置き、その中の箱の位置（ox, oy）にコマを描く。反転は枠ごと左右を返す
          const fx0 = Math.round((cv.width - gw * z) / 2), fy0 = Math.round(cv.height - gh * z);
          x.imageSmoothingEnabled = false;
          if (flip) { x.translate(fx0 * 2 + Math.round(gw * z), 0); x.scale(-1, 1); }
          x.drawImage(src, (k % m.cols) * m.cw, Math.floor(k / m.cols) * m.ch, m.cw, m.ch, fx0 + Math.round(ox * z), fy0 + Math.round(oy * z), Math.round(m.cw * z), Math.round(m.ch * z));
          x.setTransform(1, 0, 0, 1, 0, 0);
          return;
        }
        const dw = Math.max(1, Math.round(m.cw * view.z)), dh = Math.max(1, Math.round(m.ch * view.z));
        const dx = Math.round((cv.width - dw) / 2), dy = cv.height - dh;
        x.imageSmoothingEnabled = false;
        x.clearRect(0, 0, cv.width, cv.height);
        x.drawImage(img, i * m.cw, k * m.ch, m.cw, m.ch, dx, dy, dw, dh);
      });
    }
    function update(dt) {
      if (acts) {
        if (playingS) t[0] = (t[0] || 0) + dt;
        const dir = shownDir();
        if (!fit || fit.dir !== dir) layout();
        else draw();
        return;
      }
      if (playingS) for (let i = 0; i < 8; i++) t[i] += dt;
      draw();
      const me = battle && battle.active[0];
      const row = me && me.drow !== undefined ? me.drow : -1;
      cells.forEach((c, i) => c.classList.toggle("cur", i === row));
    }
    function frameCount() {
      if (acts) return Math.max(1, ...rows.map((r) => r.n));
      return m && m.n ? Math.max(...m.n) : 1;
    }
    function updateFrameLabel() {
      const n = frameCount();
      $("#dframe").textContent = playingS ? `${n}コマ` : `${(fk % n) + 1} / ${n}`;
      $("#dplay").textContent = playingS ? "停止" : "再生";
    }
    $("#dplay").addEventListener("click", () => { playingS = !playingS; updateFrameLabel(); });
    $("#dprev").addEventListener("click", () => { const n = frameCount(); playingS = false; fk = (fk + n - 1) % n; updateFrameLabel(); draw(); });
    $("#dnext").addEventListener("click", () => { const n = frameCount(); playingS = false; fk = (fk + 1) % n; updateFrameLabel(); draw(); });
    $("#dzoom").addEventListener("change", (e) => { zoom = +e.target.value; layout(); });
    let layoutTick = 0, lastHostW = -1, lastHostH = -1;
    const ro = new ResizeObserver(() => {
      const w = dirsBody ? dirsBody.clientWidth : 0, h = dirsBody ? dirsBody.clientHeight : 0;
      if (w === lastHostW && h === lastHostH) return;
      lastHostW = w; lastHostH = h;
      cancelAnimationFrame(layoutTick);
      layoutTick = requestAnimationFrame(() => layout());
    });
    const dirsHost = document.getElementById("dirs");
    if (dirsHost) ro.observe(dirsHost);
    if (dirsBody) ro.observe(dirsBody);
    return { show, update };
  })();

  // ---- controls ---------------------------------------------------------------------------------------
  function syncMode() {
    document.querySelectorAll("#pad button").forEach((b) => b.classList.toggle("on", (mode === "face" || mode === "spin") && +b.dataset.row === faceRow || (b.dataset.row === "auto" && mode === "auto")));
    $("#spin").classList.toggle("on", mode === "spin");
    $("#walk").classList.toggle("on", mode === "walk");
  }
  function setPaused(p) { paused = p; $("#pause").classList.toggle("on", p); $("#pause").textContent = p ? "再生" : "一時停止"; }
  function initControls() {
    // バトルと同じカメラ（Renderer.updateCamera：モンスター追従、既定 100% = GIF 等倍）。倍率もバトルと同じ操作で変える。
    const zv = $("#camzoom-val"), zvh = $("#cam-zoom-val");
    const setZoom = (v) => {
      const z = Renderer.setUserZoom(v), txt = Math.round(z * 100) + "%";
      if (zv) zv.textContent = txt;
      if (zvh) zvh.textContent = txt;
    };
    const stepZoom = (d) => setZoom(d === 0 ? Renderer.cam.userDefault : Renderer.cam.user * (d > 0 ? 1.1 : 1 / 1.1));
    Renderer.zoomProfile("arena.camZoom.debug.v2", 1); // [ユーザー指定 2026-11-15] 100% = front.gif と同じ大きさ（バトルと同じ）
    setZoom(Renderer.cam.user);
    document.querySelectorAll("#camzoom button, #cam-zoom button").forEach((b) => b.addEventListener("click", (ev) => { ev.stopPropagation(); stepZoom(+b.dataset.z); b.blur(); }));
    $("#view").addEventListener("wheel", (e) => { e.preventDefault(); setZoom(Renderer.cam.user * Math.exp(-Math.max(-60, Math.min(60, e.deltaY)) * 0.0022)); }, { passive: false });
    addEventListener("keydown", (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey || (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName))) return;
      if (e.key === "-" || e.key === "_") stepZoom(-1); else if (e.key === "=" || e.key === "+") stepZoom(1); else if (e.key === "0") stepZoom(0);
    });
    const appEl = $("#app");
    const setPane = (p) => { keyPane = p; held.clear(); $("#side").classList.toggle("keys", p === "list"); $("#view").classList.toggle("keys", p === "stage"); };
    const toggleDrawer = (which, force) => {
      const cls = "show-" + which;
      const open = force === undefined ? !appEl.classList.contains(cls) : !!force;
      appEl.classList.toggle(cls, open);
      const btn = $("#tog-" + which);
      if (btn) { btn.classList.toggle("on", open); btn.setAttribute("aria-pressed", String(open)); }
      if (which === "side" && !open && keyPane === "list") setPane("stage");
      if (which === "dirs" && open && cur) Strip.show(cur, opt.shiny);
      dispatchEvent(new Event("resize"));
    };
    for (const which of ["side", "moves", "dirs"]) {
      const btn = $("#tog-" + which);
      if (btn) { btn.setAttribute("aria-pressed", "false"); btn.addEventListener("click", () => toggleDrawer(which)); }
    }
    const hudBtn = $("#tog-hud");
    const syncHudBtn = () => { if (hudBtn) { hudBtn.classList.toggle("on", showHud); hudBtn.setAttribute("aria-pressed", String(showHud)); } };
    syncHudBtn();
    if (hudBtn) hudBtn.addEventListener("click", () => { showHud = !showHud; if (showHud) HUD.show(); else HUD.hide(); syncHudBtn(); });
    document.querySelectorAll("#skills .card").forEach((c) => {
      c.addEventListener("pointerdown", (ev) => { ev.stopPropagation(); });
      c.addEventListener("click", (ev) => { ev.stopPropagation(); const i = +c.dataset.i; if (i >= 0 && i < 4) slotReq[i] = true; });
    });
    const uDodge = $("#u-dodge"), uMega = $("#u-mega");
    if (uDodge) { uDodge.addEventListener("pointerdown", (ev) => ev.stopPropagation()); uDodge.addEventListener("click", (ev) => { ev.stopPropagation(); dodgeReq = true; }); }
    if (uMega) { uMega.addEventListener("pointerdown", (ev) => ev.stopPropagation()); uMega.addEventListener("click", (ev) => { ev.stopPropagation(); $("#mega").click(); }); }
    const pad = $("#pad");
    const auto = document.createElement("button"); auto.textContent = "◎"; auto.title = "相手のほうを向く"; auto.dataset.row = "auto"; auto.addEventListener("click", () => { mode = "auto"; syncMode(); }); pad.appendChild(auto);
    DIRS.forEach(([k, jp, arrow], i) => { const b = document.createElement("button"); b.textContent = arrow; b.title = `${jp}（${k}）`; b.dataset.row = i; b.addEventListener("click", () => { faceRow = i; mode = "face"; syncMode(); }); pad.appendChild(b); });
    $("#spin").addEventListener("click", () => { mode = mode === "spin" ? "auto" : "spin"; spinT = 0; syncMode(); });
    $("#walk").addEventListener("click", () => { mode = mode === "walk" ? "auto" : "walk"; syncMode(); });
    $("#pause").addEventListener("click", () => setPaused(!paused));
    $("#stepf").addEventListener("click", () => { setPaused(true); stepReq++; });
    $("#speed").value = String(opt.speed);
    $("#speed").addEventListener("change", (e) => { opt.speed = +e.target.value; store.set("speed", opt.speed); });
    const bindCk = (id, k, rebuild) => { const el = $("#" + id); el.checked = !!opt[k]; el.addEventListener("change", () => { opt[k] = el.checked; store.set(k, opt[k]); if (k === "shiny" && cur) Strip.show(cur, opt.shiny); if (k === "mute") Audio.setMuted(opt.mute); if (rebuild) build(); }); };
    bindCk("shiny", "shiny", true); bindCk("inf", "inf"); bindCk("foeMove", "foeMove"); bindCk("foeAtk", "foeAtk"); bindCk("mute", "mute");
    const ar = $("#arena");
    for (const a of Arenas.list()) ar.insertAdjacentHTML("beforeend", `<option value="${a.id}">${a.jp}</option>`);
    ar.value = opt.arena; if (!ar.value) { opt.arena = "stadium"; ar.value = "stadium"; }
    ar.addEventListener("change", () => { opt.arena = ar.value; store.set("arena", opt.arena); build(); });
    const we = $("#weather"); we.value = opt.weather;
    we.addEventListener("change", () => { opt.weather = we.value; store.set("weather", opt.weather); build(); });
    const fo = $("#foe");
    fo.innerHTML = roster.map((p) => `<option value="${p.slug}">相手: ${String(p.dex).padStart(4, "0")} ${p.jp}${p.kind !== "gen" ? "（" + p.kind + "）" : ""}</option>`).join("");
    fo.value = opt.foe; if (!fo.value) { opt.foe = roster[0].slug; fo.value = opt.foe; }
    fo.addEventListener("change", () => { opt.foe = fo.value; store.set("foe", opt.foe); build(); });
    // [ユーザー指定 2026-11-15] トレーナーもえらべる。変えてもバトルは作り直さず、その場で立ち絵だけ差し替える。
    for (const s of [0, 1]) {
      const el = $("#tr" + s), who = s ? "相手" : "自分";
      el.innerHTML = `<option value="">${who}トレーナー: ランダム</option>` + TRAINERS.map((t) => `<option value="${t.src}">${who}トレーナー: ${t.n}（${String(t.g || "").toUpperCase()}）</option>`).join("");
      el.value = opt.tr[s]; if (el.value !== opt.tr[s]) { opt.tr[s] = ""; el.value = ""; }
      el.addEventListener("change", async () => {
        opt.tr[s] = el.value; store.set("tr" + s, opt.tr[s]);
        const B = battle, t = pickTrainers()[s];
        if (!B || !t) return;
        const rec = Assets.get("../" + t.src, { raw: true });
        await Assets.whenReady([rec], 8000);
        if (battle !== B || opt.tr[s] !== el.value) return;
        B.trainers[s] = t; B.trainerRecs[s] = rec;
      });
    }
    $("#mega").addEventListener("click", () => {
      const B = battle, me = B && B.active[0];
      if (!me) return;
      B.megaUsed[0] = false;
      if (B.canMega(me)) B.megaEvolve(me);
      else if (B.formRing && B.formRing(me) && B.canCycleForm(me)) B.cycleForm(me);
      else B.fx.label(me.x, me.y - me.height - 10, "フォルムチェンジなし", "#dddddd", { size: 11 });
    });
    $("#reset").addEventListener("click", () => build());
    const gs = $("#gen");
    for (const g of [...new Set(roster.map(genOf))].sort((a, b) => parseInt(a.slice(3)) - parseInt(b.slice(3)))) gs.insertAdjacentHTML("beforeend", `<option value="${g}">${g.replace("gen", "第")}世代</option>`);
    gs.value = store.get("gen", ""); $("#only8").checked = store.get("only8", false); $("#q").value = store.get("q", "");
    const form = $("#form"); form.value = store.get("form", ""); if (form.value !== store.get("form", "")) form.value = "";
    gs.addEventListener("change", () => { store.set("gen", gs.value); renderList(); });
    form.addEventListener("change", () => { store.set("form", form.value); renderList(); });
    $("#only8").addEventListener("change", (e) => { store.set("only8", e.target.checked); renderList(); });
    $("#q").addEventListener("input", (e) => { store.set("q", e.target.value); renderList(); });
    addEventListener("pointerdown", () => Audio.unlock());
    addEventListener("keydown", (e) => {
      const typing = e.target instanceof HTMLInputElement && e.target.type !== "checkbox";
      // 一覧の前後は、検索欄にいるときの ↑↓ と、どこでも使える PageUp / PageDown。方向キーはモンスターの移動に使う。
      // [ユーザー指定 2026-11-15] 左の一覧（SPRITE & MOVE DEBUG の枠）を触ったあとは ↑↓ で前後のモンスターへ。
      // バトル画面をタップしたあとは方向キーでモンスターを動かす。PageUp / PageDown と WASD はどちらの状態でも効く。
      const arrowUD = e.key === "ArrowUp" || e.key === "ArrowDown";
      const listKey = ((typing || keyPane === "list") && arrowUD) || e.key === "PageUp" || e.key === "PageDown";
      if (typing && !listKey) return;
      if (e.target instanceof HTMLSelectElement && !listKey) return; // プルダウン操作中の矢印キーはそのまま
      if (listKey) {
        if (!shown.length) return;
        e.preventDefault();
        const down = e.key === "ArrowDown" || e.key === "PageDown";
        const i = shown.indexOf(cur), j = i < 0 ? 0 : Math.max(0, Math.min(shown.length - 1, i + (down ? 1 : -1)));
        select(shown[j]);
      } else if (MOVE_KEYS.includes(e.key.toLowerCase()) && !e.ctrlKey && !e.metaKey && !e.altKey && (keyPane === "stage" || !e.key.startsWith("Arrow"))) { held.add(e.key.toLowerCase()); e.preventDefault(); }
      else if (/^[1-4]$/.test(e.key) && cur && cur.moves) { const s = cur.moves[+e.key - 1]; if (s) fire([s]); }
      else if (["z", "x", "c", "v"].includes(e.key.toLowerCase()) && !e.ctrlKey && !e.metaKey && !e.altKey) { slotReq[["z", "x", "c", "v"].indexOf(e.key.toLowerCase())] = true; }
      else if (e.key === "Shift") { dodgeReq = true; }
      else if (e.key.toLowerCase() === "m") { $("#mega").click(); }
      else if (e.key === " ") { e.preventDefault(); setPaused(!paused); }
      else if (e.key === ".") { setPaused(true); stepReq++; }
    });
    $("#side").addEventListener("pointerdown", () => setPane("list"));
    $("#view").addEventListener("pointerdown", () => { setPane("stage"); if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur(); });
    setPane("stage");
    addEventListener("keyup", (e) => held.delete(e.key.toLowerCase()));
    addEventListener("blur", () => held.clear());
    if (FILE) {
      const n = $("#notice");
      n.hidden = false;
      n.innerHTML = "ファイルを直接ひらいています。ブラウザの制限で、<b>バトルフィールドの3D表示</b>・GIFのアニメ・ドット化はつかえません（平面表示になります）。<br><b>3Dで見るには「デバッグ起動.command」をダブルクリックしてひらいてください。</b><br>8方向シートがあるモンスターはシートのアニメで動きます。ないモンスターはバトル画面では静止画です（下の8方向欄はアニメします）。<br><small>（クリックでとじる）</small>";
      n.style.pointerEvents = "auto"; n.style.cursor = "pointer";
      n.addEventListener("click", () => { n.hidden = true; });
    }
  }

  // ---- boot -------------------------------------------------------------------------------------------
  function boot() {
    if (!window.ARENA_MONSTERDEX) { $("#count").textContent = "data/monsterdex.js が みつかりません"; return; }
    roster = Data.build().filter((p) => p.dir);
    HUD.init();
    Renderer.init($("#screen"), $("#overlay"));
    new ResizeObserver(() => dispatchEvent(new Event("resize"))).observe(stageEl);
    Audio.setMuted(opt.mute);
    initControls();
    renderList();
    syncMode();
    dispatchEvent(new Event("resize"));
    const want = QP.get("mon") || decodeURIComponent(location.hash.slice(1));
    select(Data.bySlug(want) || Data.bySlug("charizard") || roster[0], !want);
    requestAnimationFrame(loop);
  }
  document.addEventListener("DOMContentLoaded", boot);
})();
