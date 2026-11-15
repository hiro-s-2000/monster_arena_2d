// Title menu + character select. Modes:
//   battle (1–6 vs 1–6, play or watch) · tournament (8-entry bracket) · mini (turn-based 3v3) · training (CPU never attacks)
"use strict";

const Menu = (() => {
  const $ = (s, r = document) => r.querySelector(s);
  const MODES = {
    battle: { title: "バトル", sub: "6たい6 リアルタイムバトル", sides: 2, max: [6, 6] },
    tournament: { title: "トーナメント", sub: "8たい の かちぬきせん", sides: 1, max: [8] },
    mini: { title: "ミニバトル", sub: "コマンドで たたかう ターンせい", sides: 2, max: [3, 3] },
    training: { title: "トレーニング", sub: "こうげきしてこない あいてで れんしゅう", sides: 2, max: [6, 1] },
  };
  const PRESETS = [
    { n: "トレーラーの たいせんカード", a: ["riolu", "torracat", "gible", "charizard", "flygon", "charmander"], b: ["mewtwo", "sandslash", "garchomp", "machamp", "tyranitar", "registeel"] },
    { n: "カントー ごさんけ", a: ["charizard", "blastoise", "venusaur"], b: ["pikachu", "gengar", "snorlax"] },
    { n: "でんせつ たいけつ", a: ["mewtwo", "rayquaza", "lugia"], b: ["kyogre", "groudon", "dialga"] },
    { n: "メガシンカ ぐんだん", a: ["charizard", "lucario", "gengar"], b: ["garchomp", "blaziken", "gyarados"] },
    { n: "みずの フィールド", a: ["gyarados", "seaking", "blastoise"], b: ["charizard", "kingler", "snorlax"] },
    { n: "さいしん せだい", a: ["skeledirge", "meowscarada", "quaquaval"], b: ["koraidon", "miraidon", "tinkaton"] },
  ];
  const T8 = ["greninja", "charizard", "gengar", "registeel", "mewtwo", "lucario", "snorlax", "pikachu"];
  // First battle select: FRLG レッド vs ヒビキ, with these six-on-six.
  const OPENING = [
    ["pikachu", "lapras", "snorlax", "venusaur", "blastoise-mega", "charizard"],
    ["ampharos", "scizor", "heracross", "meganium", "feraligatr-mega", "typhlosion"],
  ];
  let mode = "battle", teams = [[], []], side = 0, arenaId = "stadium", gen = 0, onStart = null, gridKey = "", titleCursor = 0;
  let trIdx = [0, 1];
  // Per-side name query. trIdx stays an index into the full list even when the strip hides that trainer.
  let trQuery = ["", ""];
  const trList = () => window.ARENA_TRAINERS || [];

  function init(startCb) {
    onStart = startCb;
    // title menu
    const tm = $("#title-menu");
    tm.innerHTML = Object.entries(MODES).map(([k, m], i) => `<button class="opt${i === 0 ? " sel" : ""}" data-mode="${k}"><span class="t">${m.title}</span><span class="sub">${m.sub}</span></button>`).join("");
    tm.querySelectorAll(".opt").forEach((b, i) => {
      b.addEventListener("mouseenter", () => setTitleCursor(i));
      b.addEventListener("click", () => openSelect(b.dataset.mode));
    });
    addEventListener("keydown", (e) => {
      if ($("#title").classList.contains("hidden")) return;
      const k = e.key.toLowerCase();
      if (!$("#press").classList.contains("hidden")) { pressStart(); e.preventDefault(); return; }
      if (k === "arrowdown" || k === "s") setTitleCursor((titleCursor + 1) % 4);
      else if (k === "arrowup" || k === "w") setTitleCursor((titleCursor + 3) % 4);
      else if (k === "enter" || k === "z" || k === " ") openSelect(Object.keys(MODES)[titleCursor]);
      else return;
      e.preventDefault();
    });
    addEventListener("keydown", (e) => {
      if ($("#select").classList.contains("hidden")) return;
      if (kbTyping(e)) {
        if (e.key === "Escape") { e.target.blur(); e.preventDefault(); }
        return;
      }
      const k = e.key.toLowerCase();
      if (k === "arrowleft") kbStep(-1, 0);
      else if (k === "arrowright") kbStep(1, 0);
      else if (k === "arrowup") kbStep(0, -1);
      else if (k === "arrowdown") kbStep(0, 1);
      else if (k === "enter" || k === "z") kbGo();
      else if (k === "escape" || k === "backspace") $("#back").click();
      else return;
      e.preventDefault();
    });
    $("#press").addEventListener("click", pressStart);
    // arenas
    const ar = $("#arenas");
    ar.innerHTML = `<button class="arena sel" data-id="random"><span>ランダム</span><small>RANDOM</small></button>` + Arenas.list().map((a) => `<button class="arena" data-id="${a.id}"><span>${a.jp}</span><small>${a.name.toUpperCase()}</small></button>`).join("");
    arenaId = "random";
    ar.querySelectorAll(".arena").forEach((d) => d.addEventListener("click", () => { arenaId = d.dataset.id; ar.querySelectorAll(".arena").forEach((x) => x.classList.toggle("sel", x === d)); Audio.sfx("cursor"); }));
    let qi = 0; const ids = Arenas.list().map((a) => a.id);
    let thumbTimer = 0, thumbPaused = false, gridBuilt = false;
    const thumbNext = () => {
      thumbTimer = 0;
      if (thumbPaused || qi >= ids.length) return;
      const id = ids[qi++];
      const d = ar.querySelector(`[data-id="${id}"]`);
      if (d) d.style.backgroundImage = `url(${Renderer.thumbnail(id)})`;
      if (qi < ids.length) thumbTimer = setTimeout(thumbNext, 30);
    };
    openSelect._prep = () => {
      thumbPaused = false;
      if (!gridBuilt) { gridBuilt = true; buildGrid(); }
      if (!thumbTimer && qi < ids.length) thumbNext();
    };
    openSelect._pause = () => {
      thumbPaused = true;
      if (thumbTimer) { clearTimeout(thumbTimer); thumbTimer = 0; }
    };
    // grid & filters
    const g = $("#gens");
    for (let i = 0; i <= 9; i++) { const b = document.createElement("button"); b.textContent = i === 0 ? "ALL" : String(i); b.className = i === 0 ? "on" : ""; b.addEventListener("click", () => { gen = i; g.querySelectorAll("button").forEach((x, k) => x.classList.toggle("on", k === i)); buildGrid(); }); g.appendChild(b); }
    $("#search").addEventListener("input", buildGrid);
    $("#filter-kind").addEventListener("change", buildGrid);
    const ps = $("#presets");
    PRESETS.forEach((p, i) => { const o = document.createElement("option"); o.value = i; o.textContent = p.n; ps.appendChild(o); });
    ps.addEventListener("change", () => { const p = PRESETS[+ps.value]; if (!p) return; const cap = MODES[mode].max; teams[0] = p.a.slice(0, cap[0]).map(Data.bySlug).filter(Boolean).map((s) => ({ species: s, shiny: false })); if (MODES[mode].sides > 1) teams[1] = p.b.slice(0, cap[1]).map(Data.bySlug).filter(Boolean).map((s) => ({ species: s, shiny: false })); ps.value = ""; renderSlots(); Audio.sfx("select"); });
    $("#grid").addEventListener("click", (e) => { const c = e.target.closest(".card"); if (c) { kbEl = c; add(Data.bySlug(c.dataset.slug)); } }); // the cursor follows the click, so the grid stays put
    $("#grid").addEventListener("mouseover", (e) => { const c = e.target.closest(".card"); if (c) showDex(Data.bySlug(c.dataset.slug)); });
    document.querySelectorAll(".teams .team").forEach((t) => t.addEventListener("click", (e) => { if (e.target.closest(".slot,button")) return; selectSide(+t.dataset.side); }));
    document.querySelectorAll("[data-random]").forEach((b) => b.addEventListener("click", () => {
      if (randomTeam(+b.dataset.random, visibleMons())) Audio.sfx("select");
      else Audio.sfx("back");
    }));
    document.querySelectorAll("[data-clear]").forEach((b) => b.addEventListener("click", () => { teams[+b.dataset.clear] = []; renderSlots(); Audio.sfx("back"); }));
    $("#start").addEventListener("click", start);
    $("#back").addEventListener("click", () => { Audio.sfx("back"); showTitle(true); });
    $("#opt-music").addEventListener("change", (e) => Audio.setMusic(e.target.checked));
    $("#opt-sfx").addEventListener("change", (e) => Audio.setMuted(!e.target.checked));
    if ($("#opt-hud")) $("#opt-hud").addEventListener("change", (e) => HUD.setSkin && HUD.setSkin(e.target.value));
    // 初期表示はかんせん＋つよい。モードを変えても、自分で選んだCPUはそのまま残す。
    $("#opt-mode").addEventListener("change", () => syncWatchOpts());
    // Grid and 3D arena thumbnails build lazily when openSelect() first opens.
    teams = OPENING.map(openingTeam);
    initTrainers();
  }
  function openingTeam(slugs) {
    const mons = slugs.map((slug) => Data.bySlug(slug));
    if (mons.some((p) => !p)) return [];
    return mons.map((species) => ({ species, shiny: false }));
  }

  function syncWatchOpts() {
    const watch = mode === "battle" && $("#opt-mode").value === "watch";
    $(".teams .team[data-side='0'] h3 span").textContent = mode === "tournament" ? "しゅつじょう モンスター（8たい）" : watch ? "あかコーナー" : "あなたの チーム";
    if (mode === "battle") $(".teams .team[data-side='1'] h3 span").textContent = watch ? "あおコーナー" : "あいての チーム";
    const help = $(".help");
    if (!help) return;
    const keys = " ・ このがめん <kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd> いどう ・ <kbd>Enter</kbd> / <kbd>Z</kbd> けってい";
    help.innerHTML = (watch
      ? "かんせん：あかコーナーに わざ <kbd>Z</kbd><kbd>X</kbd><kbd>C</kbd><kbd>V</kbd> ・ かいひ <kbd>SPACE</kbd> ・ こうたい <kbd>Q</kbd>かアイコン ・ 進化の きりかえ <kbd>M</kbd>（CPUは つよい）"
      : "いどう <kbd>WASD</kbd> / <kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd> ・ わざ <kbd>Z</kbd><kbd>X</kbd><kbd>C</kbd><kbd>V</kbd> / クリック ・ かいひ <kbd>SPACE</kbd> ・ こうたい <kbd>Q</kbd>かアイコン ・ メガ <kbd>M</kbd> ・ ゲームパッド たいおう") + keys;
  }
  function setTitleCursor(i) { if (i !== titleCursor) Audio.sfx("cursor"); titleCursor = i; document.querySelectorAll("#title-menu .opt").forEach((b, k) => b.classList.toggle("sel", k === i)); }
  function pressStart() { Audio.unlock(); Audio.sfx("go"); $("#press").classList.add("hidden"); $("#title-menu").classList.remove("hidden"); }
  function showTitle(skipPress) {
    $("#select").classList.add("hidden"); $("#title").classList.remove("hidden");
    $("#press").classList.toggle("hidden", !!skipPress); $("#title-menu").classList.toggle("hidden", !skipPress);
  }
  function hideAll() { if (openSelect._pause) openSelect._pause(); $("#title").classList.add("hidden"); $("#select").classList.add("hidden"); }

  function openSelect(m) {
    if (openSelect._prep) openSelect._prep();
    mode = m; Audio.sfx("select");
    const M = MODES[m];
    $("#title").classList.add("hidden");
    $("#select").classList.remove("hidden");
    $("#select-title").textContent = M.title; $("#select-sub").textContent = M.sub;
    $("#select").dataset.mode = m;
    $(".teams .team[data-side='1']").classList.toggle("hidden", M.sides < 2);
    $(".teams .team[data-side='1'] h3 span").textContent = m === "training" ? "トレーニング あいて" : "あいての チーム";
    syncWatchOpts();
    $("#opt-mode-wrap").classList.toggle("hidden", m !== "battle");
    $("#opt-time-wrap").classList.toggle("hidden", m === "mini");
    $("#opt-pilot-wrap").classList.toggle("hidden", m !== "tournament");
    for (let s = 0; s < 2; s++) if (!teams[s].length || teams[s].length > M.max[s]) randomTeam(s);
    if (m === "tournament" && teams[0].length < 8) teams[0] = T8.map(Data.bySlug).map((s) => ({ species: s, shiny: false }));
    for (let s = 0; s < 2; s++) teams[s] = teams[s].slice(0, M.max[s] || 0);
    selectSide(0); renderSlots();
    armKb();
    // The strip is display:none until now, so the first scroll has to wait for layout.
    requestAnimationFrame(() => markTrainers());
  }

  function selectSide(s) { side = s; document.querySelectorAll(".teams .team").forEach((t) => t.classList.toggle("sel", +t.dataset.side === s)); }
  function add(species) {
    if (!species) return;
    const M = MODES[mode];
    if (teams[side].length >= M.max[side]) { if (M.sides > 1 && teams[1 - side].length < M.max[1 - side]) selectSide(1 - side); else { Audio.sfx("back"); return; } }
    teams[side].push({ species, shiny: false }); Audio.sfx("select");
    if (teams[side].length >= M.max[side] && M.sides > 1 && teams[1 - side].length < M.max[1 - side]) selectSide(1 - side);
    renderSlots();
  }
  // The grid's search, form, and generation. Empty pool stays out so a pick can still fight.
  function visibleMons() {
    const q = $("#search").value, kind = $("#filter-kind").value;
    return Data.search(q, kind).filter((p) => (!gen || p.gen === gen) && p.pool && p.pool.length);
  }
  function randomTeam(s, list) {
    const r = list === undefined ? Data.roster().filter((p) => p.kind === "gen" && p.pool && p.pool.length) : list;
    const n = Math.min(MODES[mode].max[s] || 0, r.length), picks = [];
    if (!n) return false;
    while (picks.length < n) { const p = U.pick(r); if (!picks.includes(p)) picks.push(p); }
    teams[s] = picks.map((species) => ({ species, shiny: false }));
    renderSlots();
    return true;
  }
  // Card height follows the field curve. 21px is the short floor; about 105px and up, including every Gigantamax, fills the card.
  function iconPx(p, max, min) {
    const body = Assets.bodyArtPx(p);
    return Math.round(U.clamp(min + (max - min) * ((body - 21) / 84), min, max));
  }
  function cardHTML(p, cls = "card") {
    const [c1] = Data.typeColor(p.types[0]), c2 = Data.typeColor(p.types[1] || p.types[0])[2];
    const h = iconPx(p, 78, 34);
    return `<div class="${cls} ${p.kind}" data-slug="${p.slug}" style="--c1:${c1};--c2:${c2}"><img loading="lazy" decoding="async" src="${Assets.stillUrl(p)}" alt="" style="height:${h}px"><span class="nm">${p.jp}</span><span class="dx">${p.dex}</span></div>`;
  }
  // ---- ずかん panel: the arena-size pixel sprite next to a trainer, stats, ability and arena traits ----
  let dexSlug = null, dexRaf = 0, dexTr = null;
  function chosenTrainer(sideN) {
    const list = trList();
    return list[trIdx[sideN]] || list[0] || null;
  }
  function dexTrainer() {
    const t = chosenTrainer(0);
    if (!t) return null;
    if (!dexTr || dexTr.t !== t) dexTr = { t, rec: Assets.get("../" + t.src, { raw: true }) };
    return dexTr;
  }
  const trImg = new Map();
  function paintTrainer(cv, t) {
    const draw = (img) => {
      cv.width = t.w; cv.height = t.h;
      const x = cv.getContext("2d");
      x.imageSmoothingEnabled = false;
      x.clearRect(0, 0, t.w, t.h);
      if (t.ef) { x.translate(t.w, 0); x.scale(-1, 1); }
      x.drawImage(img, t.east[0] * t.w, 0, t.w, t.h, 0, 0, t.w, t.h);
    };
    let img = trImg.get(t.src);
    if (!img) { img = new Image(); img.src = "../" + t.src; trImg.set(t.src, img); }
    if (img.complete && img.naturalWidth) draw(img);
    else img.addEventListener("load", () => draw(img), { once: true });
  }
  function trainerHits(s) {
    const q = (trQuery[s] || "").trim();
    const list = trList();
    const hits = [];
    for (let i = 0; i < list.length; i++) if (!q || list[i].n.includes(q)) hits.push(i);
    return hits;
  }
  function markTrainers() {
    document.querySelectorAll(".trstrip").forEach((box) => {
      const s = +box.dataset.side, i = trIdx[s] || 0;
      box.querySelectorAll(".tr").forEach((b) => b.classList.toggle("on", +b.dataset.i === i));
      const name = box.parentElement && box.parentElement.querySelector(".trname");
      const t = trList()[i];
      if (name && t) name.textContent = t.n;
      const on = box.querySelector(".tr.on");
      if (on) box.scrollLeft = on.offsetLeft - box.clientWidth / 2 + on.offsetWidth / 2;
    });
  }
  function renderTrainerStrip(s) {
    const box = document.querySelector(`.trstrip[data-side="${s}"]`);
    if (!box) return;
    const list = trList();
    const hits = trainerHits(s);
    box.innerHTML = hits.length
      ? hits.map((i) => `<button type="button" class="tr" data-i="${i}" title="${list[i].n}"></button>`).join("")
      : `<span class="trnone">いない</span>`;
    box.querySelectorAll(".tr").forEach((btn) => {
      const i = +btn.dataset.i;
      const cv = document.createElement("canvas");
      btn.appendChild(cv);
      paintTrainer(cv, list[i]);
      btn.addEventListener("click", (ev) => { ev.stopPropagation(); pickTrainer(s, i); Audio.sfx("cursor"); });
    });
    markTrainers();
  }
  function stepTrainer(s, dir) {
    const hits = trainerHits(s);
    if (!hits.length) return;
    const cur = hits.indexOf(trIdx[s]);
    const step = dir > 0 ? 1 : -1;
    const next = cur < 0 ? (dir > 0 ? hits[0] : hits[hits.length - 1]) : hits[(cur + step + hits.length) % hits.length];
    pickTrainer(s, next);
  }
  function pickTrainer(s, i) {
    const n = trList().length;
    if (!n) return;
    trIdx[s] = (i + n) % n;
    markTrainers();
    if (dexSlug) { const p = Data.bySlug(dexSlug); dexSlug = null; showDex(p); }
  }
  function initTrainers() {
    const list = trList();
    const red = list.findIndex((t) => t.g === "frlg" && t.src.endsWith("frlg_000.png"));
    const hibiki = list.findIndex((t) => t.src.endsWith("hgss_0000.png"));
    trIdx = [red >= 0 ? red : 0, hibiki >= 0 ? hibiki : Math.min(1, list.length - 1)];
    trQuery = ["レッド", "ヒビキ"];
    document.querySelectorAll(".trfind").forEach((inp) => { inp.value = trQuery[+inp.dataset.side] || ""; });
    renderTrainerStrip(0);
    renderTrainerStrip(1);
    document.querySelectorAll(".trstep").forEach((b) => b.addEventListener("click", (ev) => {
      ev.stopPropagation();
      stepTrainer(+b.dataset.side, +b.dataset.dir || 1);
      Audio.sfx("cursor");
    }));
    document.querySelectorAll(".trfind").forEach((inp) => inp.addEventListener("input", () => {
      trQuery[+inp.dataset.side] = inp.value;
      renderTrainerStrip(+inp.dataset.side);
    }));
  }
  const STAT_JP = [["HP", "hp"], ["こうげき", "atk"], ["ぼうぎょ", "def"], ["とくこう", "spa"], ["とくぼう", "spd"], ["すばやさ", "spe"]];
  function showDex(p) {
    if (!p || dexSlug === p.slug) return;
    dexSlug = p.slug;
    const el = $("#dex"), st = p.st, pr = Fighter.profile(p);
    const size = p.h >= 3 ? "きょだい" : p.h >= 1.5 ? "おおがた" : p.h >= 0.7 ? "ちゅうがた" : "こがた";
    const traits = [Fighter.GAIT_JP[pr.gait], `からだ ${pr.bodyH}ドット（${p.h}m）`];
    if (pr.mass >= 2.2) traits.push("おもい：よわい わざでは ひるまない");
    else if (pr.mass < 0.75) traits.push("かるい：ふっとびやすい");
    if (pr.mightAtk >= 1.25 || pr.mightSpa >= 1.25) traits.push("わざが ひろい：こうげきが つよい");
    else if (pr.mightAtk <= 0.8 && pr.mightSpa <= 0.8) traits.push("わざが せまい：こうげきが ひくい");
    if (pr.tempo >= 1.2 || st.spe >= 100) traits.push("すばやい：いどうと かいひが はやい");
    else if (pr.tempo <= 0.75 || st.spe <= 40) traits.push("どんそく：じっくり たたかう");
    if (p.leg) traits.push("でんせつ：とうじょう えんしゅつ つき");
    const bars = STAT_JP.map(([n, k]) => `<div class="st"><span>${n}</span><b>${st[k]}</b><i style="width:${Math.min(100, st[k] / 1.6)}%;background:${st[k] >= 110 ? "#58d858" : st[k] >= 70 ? "#f8c830" : "#f07848"}"></i></div>`).join("");
    el.innerHTML = `<canvas width="192" height="160"></canvas><h4>${p.jp} <small>No.${p.dex}</small></h4><p class="genus">${p.genus || ""}</p>`
      + `<div class="types">${p.types.map((t) => `<span style="--c:${Data.typeColor(t)[0]}">${Data.typeName(t)}</span>`).join("")}</div>`
      + `<p class="hw">たかさ ${p.h}m ・ おもさ ${p.w}kg<em>${size}</em></p>`
      + (p.ab ? `<div class="ab"><b>とくせい：${p.ab.jp}</b><small>${p.ab.d}</small></div>` : "")
      + `<div class="stats">${bars}<div class="tot">ごうけい ${STAT_JP.reduce((a, [, k]) => a + st[k], 0)}</div></div><ul class="traits">${traits.map((t) => `<li>${t}</li>`).join("")}</ul>`;
    const cv = $("canvas", el), cx = cv.getContext("2d"), rec = Assets.hd(p);
    const tr = dexTrainer();
    cv.width = 192; cv.height = 160;
    cancelAnimationFrame(dexRaf);
    const t0 = performance.now();
    const tick = (now) => {
      if (dexSlug !== p.slug || !cv.isConnected) return;
      cx.clearRect(0, 0, 192, 160);
      // Fit up to 268px Gigantamax cleanly inside the 160px dex preview box while keeping <=200px scale exact.
      const gy = 148, ppm = 132 / Math.max(200, Assets.bodyArtPx(p) * 1.04);
      cx.fillStyle = "rgba(0,0,0,.35)";
      if (tr && tr.rec.frames) {
        const tH = Assets.trainerArtPx() * ppm, sc = tH / tr.t.h, dw = tr.t.w * sc;
        const sheet = tr.rec.frames[0].canvas, fi = tr.t.east[0];
        const tx = 36;
        cx.beginPath(); cx.ellipse(tx, gy + 1, Math.max(8, dw * 0.35), 4, 0, 0, U.TAU); cx.fill();
        cx.imageSmoothingEnabled = false;
        cx.drawImage(sheet, fi * tr.t.w, 0, tr.t.w, tr.t.h, Math.round(tx - dw / 2), Math.round(gy + 1 - tH), dw, tH);
        cx.fillStyle = "#ffe9a0"; cx.font = "11px sans-serif"; cx.textAlign = "center";
        cx.fillText("人 1.7m", tx, gy - tH - 4);
      }
      if (rec.frames) {
        const img = Assets.frame(rec, (now - t0) / 1000), b = rec.bounds;
        const monH = Assets.bodyArtPx(p) * ppm, k = monH / Math.max(1, b.h);
        const w = img.width * k, h = img.height * k, x = Math.round(128 - (b.x + b.w / 2) * k), y = Math.round(gy + 1 - (b.y + b.h) * k);
        cx.fillStyle = "rgba(0,0,0,.35)"; cx.beginPath(); cx.ellipse(128, gy + 1, Math.max(6, b.w * k * 0.4), 5, 0, 0, U.TAU); cx.fill();
        cx.imageSmoothingEnabled = false; cx.drawImage(img, x, y, w, h);
      }
      dexRaf = requestAnimationFrame(tick);
    };
    dexRaf = requestAnimationFrame(tick);
  }

  function renderSlots() {
    rememberKb();
    const M = MODES[mode];
    for (let s = 0; s < 2; s++) {
      const box = $(`.teams .team[data-side="${s}"] .slots`);
      box.innerHTML = "";
      for (let i = 0; i < (M.max[s] || 0); i++) {
        const e = teams[s][i], d = document.createElement("div");
        d.className = "slot" + (e ? " filled" : "");
        if (e) {
          const [c1] = Data.typeColor(e.species.types[0]);
          d.style.setProperty("--c1", c1);
          const ih = iconPx(e.species, 56, 26);
          d.innerHTML = `<img src="${Assets.stillUrl(e.species)}" alt="" style="height:${ih}px"><span class="nm">${e.species.jp}</span>${e.species.shiny ? `<b class="sh ${e.shiny ? "on" : ""}" title="いろちがい">★</b>` : ""}`;
          const sh = $(".sh", d); if (sh) sh.addEventListener("click", (ev) => { ev.stopPropagation(); e.shiny = !e.shiny; renderSlots(); Audio.sfx("cursor"); });
          d.addEventListener("click", () => { teams[s].splice(i, 1); renderSlots(); Audio.sfx("back"); });
        }
        box.appendChild(d);
      }
    }
    if (mode === "tournament") { const sel = $("#opt-pilot"), v = sel.value; sel.innerHTML = `<option value="-1">かんせん（ぜんぶ CPU）</option>` + teams[0].map((e, i) => `<option value="${i}">${e.species.jp} を そうさ</option>`).join(""); sel.value = [...sel.options].some((o) => o.value === v) ? v : "-1"; }
    const ok = mode === "tournament" ? teams[0].length === 8 : teams[0].length && teams[1].length;
    $("#start").disabled = !ok;
    $("#start").textContent = mode === "tournament" ? (teams[0].length === 8 ? "トーナメント かいし！" : `あと ${8 - teams[0].length}たい えらんでね`) : "バトル かいし！";
    stickKb();
  }
  function buildGrid() {
    const q = $("#search").value, kind = $("#filter-kind").value, key = q + "|" + kind + "|" + gen;
    if (key === gridKey) return;
    rememberKb();
    gridKey = key;
    const list = Data.search(q, kind).filter((p) => !gen || p.gen === gen);
    $("#grid").innerHTML = list.map((p) => cardHTML(p)).join("");
    $("#grid").scrollTop = 0;
    stickKb();
  }
  // Arrow cursor for the select screen. Search fields and <select> keep the browser's own arrows.
  let kbEl = null, kbHint = null, aimX = null;
  function kbTyping(e) {
    const t = e.target;
    if (!t || !t.tagName) return false;
    if (t.tagName === "TEXTAREA" || t.tagName === "SELECT") return true;
    return t.tagName === "INPUT" && t.type !== "checkbox";
  }
  function kbAlive(el) {
    if (!el || !el.isConnected || el.disabled || el.closest(".hidden")) return false;
    const r = el.getBoundingClientRect();
    return r.width > 2 && r.height > 2;
  }
  function kbItems() {
    const sel = "#select header button, #select header select, #select header input, #arenas .arena, .teams .team:not(.hidden) .trfind, .teams .team:not(.hidden) .trstep, .teams .team:not(.hidden) .tr.on, .teams .team:not(.hidden) .slot.filled, .teams .team:not(.hidden) .tbtn button, #search, #filter-kind, #gens button, #presets, #grid .card, #start";
    return [...document.querySelectorAll(sel)].filter(kbAlive);
  }
  function paintKb(silent) {
    document.querySelectorAll("#select .kb").forEach((el) => el.classList.remove("kb"));
    if (!kbAlive(kbEl)) return;
    kbEl.classList.add("kb");
    const grid = $("#grid");
    if (!silent && grid && grid.contains(kbEl)) { // only a keyboard move scrolls; a refresh never yanks the grid
      const er = kbEl.getBoundingClientRect(), gr = grid.getBoundingClientRect();
      if (er.top < gr.top + 4) grid.scrollTop -= gr.top + 4 - er.top;
      else if (er.bottom > gr.bottom - 4) grid.scrollTop += er.bottom - (gr.bottom - 4);
    }
    const strip = kbEl.closest(".trstrip");
    if (strip) strip.scrollLeft = kbEl.offsetLeft - strip.clientWidth / 2 + kbEl.offsetWidth / 2;
    if (!silent && kbEl.classList.contains("card")) showDex(Data.bySlug(kbEl.dataset.slug));
  }
  function rememberKb() {
    if (!kbAlive(kbEl)) return;
    const r = kbEl.getBoundingClientRect();
    kbHint = { x: r.left + r.width / 2, y: r.top + r.height / 2, slug: kbEl.dataset.slug || "", id: kbEl.id || "" };
  }
  function stickKb() {
    if (kbAlive(kbEl)) { paintKb(true); return; }
    if (kbHint && kbHint.slug) {
      const same = document.querySelector(`#grid .card[data-slug="${CSS.escape(kbHint.slug)}"]`);
      if (same) { kbEl = same; paintKb(true); return; }
    }
    if (kbHint && kbHint.id && kbAlive(document.getElementById(kbHint.id))) { kbEl = document.getElementById(kbHint.id); paintKb(true); return; }
    if (!kbHint) return;
    let best = null, bd = 1e9;
    for (const el of kbItems()) {
      const r = el.getBoundingClientRect();
      const d = Math.hypot(r.left + r.width / 2 - kbHint.x, r.top + r.height / 2 - kbHint.y);
      if (d < bd) { bd = d; best = el; }
    }
    kbEl = best;
    paintKb(true);
  }
  function armKb() {
    kbEl = document.querySelector("#grid .card") || document.querySelector("#arenas .arena.sel");
    kbHint = null;
    aimX = null;
    paintKb(false);
  }
  function kbStep(dx, dy) {
    if (!kbAlive(kbEl)) armKb();
    const cur = kbEl;
    if (!cur) return;
    if (cur.classList.contains("tr") && dx) {
      const strip = cur.closest(".trstrip");
      stepTrainer(+strip.dataset.side, dx);
      kbEl = strip.querySelector(".tr.on");
      Audio.sfx("cursor");
      paintKb(true);
      return;
    }
    const a = cur.getBoundingClientRect();
    const focusX = a.width > 220 && aimX != null ? aimX : a.left + a.width / 2;
    const gap = (a0, a1, b0, b1) => a1 < b0 ? b0 - a1 : b1 < a0 ? a0 - b1 : 0;
    let best = null, bestScore = Infinity;
    for (const el of kbItems()) {
      if (el === cur) continue;
      const b = el.getBoundingClientRect();
      const primary = dx ? (dx < 0 ? a.left - b.right : b.left - a.right) : (dy < 0 ? a.top - b.bottom : b.top - a.bottom);
      if (primary < 4) continue;
      const cross = dx ? gap(a.top, a.bottom, b.top, b.bottom) : gap(a.left, a.right, b.left, b.right);
      const aimDist = dx || a.width <= 220 ? 0 : Math.abs(b.left + b.width / 2 - focusX);
      const score = primary + cross * 4 + aimDist * 0.25;
      if (score < bestScore) { bestScore = score; best = el; }
    }
    if (!best) return;
    const br = best.getBoundingClientRect();
    if (a.width <= 220) aimX = a.left + a.width / 2;
    else if (br.width <= 220) aimX = br.left + br.width / 2;
    kbEl = best;
    Audio.sfx("cursor");
    paintKb(false);
  }
  function kbGo() {
    if (!kbAlive(kbEl)) return;
    if (kbEl.tagName === "SELECT") {
      kbEl.selectedIndex = (kbEl.selectedIndex + 1) % kbEl.options.length;
      kbEl.dispatchEvent(new Event("change", { bubbles: true }));
      Audio.sfx("cursor");
      return;
    }
    if (kbEl.tagName === "INPUT") { kbEl.focus(); return; }
    rememberKb();
    kbEl.click();
    stickKb();
  }
  function start() {
    Audio.unlock(); Audio.sfx("go");
    const odds = +$("#opt-shiny").value;
    const mk = (t) => t.map((e) => ({ species: e.species, shiny: e.shiny || (e.species.shiny && Math.random() < odds) }));
    const arena = arenaId === "random" ? U.pick(Arenas.list()).id : arenaId;
    const hudSkin = $("#opt-hud") ? $("#opt-hud").value : "esports";
    const cfg = { mode, teams: [mk(teams[0]), mk(teams[1])], arenaId, arena, skill: +$("#opt-skill").value, duration: +$("#opt-time").value, watch: $("#opt-mode").value === "watch", pilot: +$("#opt-pilot").value, trainers: [chosenTrainer(0), chosenTrainer(1)].filter(Boolean), hudSkin };
    hideAll();
    onStart(cfg);
  }
  return { init, showTitle, hideAll, openSelect, mode: () => mode };
})();
