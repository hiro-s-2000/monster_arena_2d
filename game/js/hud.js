// DOM HUD in a Dragon-Quest-style window skin. Reads battle state each frame and drains the event queue:
// side panels (pixel portrait, segmented HP, energy, team balls, move callout, combo), timer + league label,
// weather countdown, skill cards, typewriter narration window, big banners and the VS card.
"use strict";

const HUD = (() => {
  const $ = (s, r = document) => r.querySelector(s);
  let el, panels, cards, tcards, tUtils, fcards, foeBox, banner, koCutin, koMain, koSub, koKicker, clashEl, skinBtn, pauseBtn, vs, msg, msgText, timerEl, labelEl, weatherEl, B = null, onPause = null;
  let bannerTimer = null, koTimer = null, lines = [], typing = null, lineT = 0, vsT = 0, skin = "esports";

  function setSkin(next) {
    skin = next === "retro" ? "retro" : "esports";
    if (!el) return;
    el.classList.toggle("esports", skin === "esports");
    el.classList.toggle("retro", skin === "retro");
    if (skinBtn) {
      skinBtn.textContent = skin === "esports" ? "HUD: eSPORTS" : "HUD: RETRO DQ";
      skinBtn.setAttribute("aria-pressed", String(skin === "esports"));
    }
    const sel = $("#opt-hud");
    if (sel && sel.value !== skin) sel.value = skin;
  }

  function clearTransient() {
    clearTimeout(bannerTimer); bannerTimer = null;
    clearTimeout(koTimer); koTimer = null;
    if (banner) banner.className = "";
    if (koCutin) koCutin.className = "";
    if (clashEl) clashEl.classList.add("hidden");
    if (vs) vs.classList.add("hidden");
    if (msg) msg.classList.add("hidden");
    if (msgText) msgText.textContent = "";
    lines = []; typing = null; lineT = 0; vsT = 0;
  }

  function init(pauseCb) {
    onPause = pauseCb || null;
    el = $("#hud");
    panels = [$("#p0"), $("#p1")].map((p) => ({
      root: p, portrait: $(".portrait canvas", p), name: $(".name", p), types: $(".types", p), hp: $(".hp", p), hpbar: $(".hpbar", p),
      fill: $(".hpbar .fill", p), ghost: $(".hpbar .ghost", p), en: $(".enbar .fill", p), status: $(".status", p), team: $(".team", p),
      call: $(".callout", p), combo: $(".combo", p), abil: $(".abil", p), cur: null, sp: null, lastHp: -1, callT: 0, comboT: 0,
    }));
    cards = [...document.querySelectorAll("#skills .card")].map((c) => ({ root: c, cd: $(".cd", c), jp: $(".jp", c), meta: $(".meta", c), key: $(".key", c), spec: null }));
    tcards = [...document.querySelectorAll("#touch .tmoves [data-slot]")].map((c) => ({ root: c, name: $(".tname", c), cost: $(".tcost", c), cd: $(".tcd", c), spec: null }));
    tUtils = { dodge: $("#u-t-dodge"), swap: $("#u-t-swap"), mega: $("#u-t-mega"), megaLbl: $("#u-t-mega span") };
    foeBox = $("#skills-foe");
    fcards = [...document.querySelectorAll("#skills-foe .fcard")].map((c) => ({ root: c, cd: $(".cd", c), jp: $(".jp", c), meta: $(".meta", c), spec: null }));
    // かんせん：あいて側のコマンド欄・パネルをタップすると、そちらへ指示を出す
    const take = (side) => (ev) => {
      if (!B || B.commandSide < 0 || ev.button !== 0) return;
      const s = side === "foe" ? 1 - B.commandSide : side;
      ev.preventDefault(); ev.stopPropagation();
      B.setCommandSide(s);
    };
    foeBox.addEventListener("pointerdown", take("foe"));
    $("#p0").addEventListener("pointerdown", take(0));
    $("#p1").addEventListener("pointerdown", take(1));
    banner = $("#banner"); vs = $("#vs"); msg = $("#msg"); msgText = $("#msg p"); timerEl = $("#timer"); labelEl = $("#league"); weatherEl = $("#weather-tag");
    koCutin = $("#ko-cutin"); koMain = $("#ko-cutin .ko-main"); koSub = $("#ko-cutin .ko-sub"); koKicker = $("#ko-cutin .ko-kicker");
    clashEl = $("#clash-tag"); skinBtn = $("#hud-skin-btn"); pauseBtn = $("#hud-pause-btn");
    const urlSkin = new URLSearchParams(location.search).get("hud");
    setSkin(urlSkin === "retro" ? "retro" : "esports");
    if (skinBtn) {
      skinBtn.addEventListener("pointerdown", (ev) => ev.stopPropagation());
      skinBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        setSkin(skin === "esports" ? "retro" : "esports");
        Audio.sfx("cursor");
      });
    }
    if (pauseBtn) {
      pauseBtn.addEventListener("pointerdown", (ev) => ev.stopPropagation());
      pauseBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (onPause) onPause();
      });
    }
    addEventListener("keydown", (ev) => {
      if (ev.key.toLowerCase() === "h" && !el.classList.contains("hidden") && !(ev.target && /INPUT|SELECT|TEXTAREA/.test(ev.target.tagName))) {
        setSkin(skin === "esports" ? "retro" : "esports");
        Audio.sfx("cursor");
      }
    });
  }
  const show = () => el.classList.remove("hidden");
  const hide = () => { el.classList.add("hidden"); clearTransient(); };
  const badge = (t) => `<span class="badge" style="background:${Data.typeColor(t)[0]}">${Data.typeName(t)}</span>`;

  function bind(b) {
    B = b; clearTransient();
    labelEl.textContent = `${b.label} · ${b.arena.def.name}`;
    weatherEl.classList.add("hidden");
    el.classList.toggle("watch", b.playerSide < 0);
    el.classList.toggle("command", b.commandSide >= 0);
    el.classList.toggle("turn", !!b.turnMode);
    $("#skills").style.display = (b.playerSide >= 0 || b.commandSide >= 0) && !b.turnMode ? "" : "none";
    el.classList.toggle("duo", (b.playerSide >= 0 || b.commandSide >= 0) && !b.turnMode);
    fcards.forEach((c) => (c.spec = null)); cards.forEach((c) => (c.spec = null));
    if (tcards) tcards.forEach((c) => (c.spec = null));
    for (let s = 0; s < 2; s++) {
      const p = panels[s];
      p.cur = null; p.sp = null; p.lastHp = -1; p.team.innerHTML = ""; p.call.className = "callout"; p.combo.className = "combo";
      B.teams[s].forEach((f, i) => {
        const d = document.createElement("i");
        d.className = "ball";
        d.dataset.slot = String(i);
        d.appendChild(document.createElement("canvas"));
        d.addEventListener("pointerdown", (ev) => {
          if (ev.button !== 0) return;
          ev.preventDefault();
          ev.stopPropagation();
          const pilot = B.playerSide >= 0 ? B.playerSide : B.commandSide;
          if (s !== pilot) return;
          requestSwap(i);
        });
        p.team.appendChild(d);
      });
    }
    if (Input.touch()) cards.forEach((c, i) => (c.key.textContent = String(i + 1)));
    $("#u-swap").onpointerdown = (ev) => {
      if (ev.button !== 0) return;
      ev.preventDefault();
      ev.stopPropagation();
      requestSwap();
    };
  }

  // 操作側の控えアイコン、または「こうたい」ボタン。あいてのチームは選ばせない。
  // アイコンは選んだ個体を出す。ボタンはクールダウン中のチーム先頭だけ。
  function requestSwap(slot) {
    if (!B || B.turnMode) return;
    if (B.phase !== "fight" && B.phase !== "intro") return;
    const side = B.playerSide >= 0 ? B.playerSide : B.commandSide;
    if (side < 0 || !B.active[side]) return;
    const target = slot === undefined ? null : B.teams[side][slot];
    const focus = (B.chosen && B.chosen[side]) || B.active[side];
    if (target && (target.ko || target.side !== side || target === focus)) return;
    if (slot === undefined) {
      if (!B.canSwap(side)) return;
      B.swapNext(side);
      return;
    }
    if (!target || target === B.active[side]) return;
    B.swapNext(side, target);
  }

  // Head-and-shoulders crop of the ORIGINAL sprite (crisp, not the arena's coarse pixel art).
  function drawPortrait(canvas, f, size = 64) {
    const rec = Assets.hd(f.species, f.shiny);
    if (!rec.ready || !rec.frames) return false;
    const pc = Assets.portrait(rec, size);
    const x = canvas.getContext("2d");
    canvas.width = size; canvas.height = size; x.imageSmoothingEnabled = false;
    if (pc) x.drawImage(pc, 0, 0);
    return !!pc;
  }

  function say(text) {
    if (!text) return;
    if (lines.length > 2) lines.shift();
    if (lines[lines.length - 1] !== text && (!typing || typing.text !== text)) lines.push(text);
  }

  function update(b, dt) {
    if (!B) return;
    for (let s = 0; s < 2; s++) {
      const p = panels[s], f = B.active[s];
      if (!f) continue;
      if (p.cur !== f || p.sp !== f.species) {
        p.cur = f; p.sp = f.species; p.portraitOk = false;
        p.name.textContent = f.species.jp; p.types.innerHTML = f.species.types.map(badge).join(""); p.lastHp = -1;
        p.ghost.style.transition = "none"; p.ghost.style.width = f.hpRatio * 100 + "%"; requestAnimationFrame(() => (p.ghost.style.transition = ""));
      }
      if (!p.portraitOk) p.portraitOk = drawPortrait(p.portrait, f);
      const r = Math.max(0, f.hpRatio);
      if (p.lastHp !== f.hp) {
        p.fill.style.width = r * 100 + "%"; p.ghost.style.width = r * 100 + "%";
        p.hp.textContent = `${Math.max(0, Math.ceil(f.hp))}/${f.maxHp}`;
        p.hpbar.className = "hpbar" + (r <= 0.2 ? " low" : r <= 0.5 ? " mid" : "");
        p.root.classList.toggle("lowhp", r <= 0.2 && r > 0);
        if (p.lastHp > f.hp) { p.root.classList.remove("hit"); void p.root.offsetWidth; p.root.classList.add("hit"); }
        p.lastHp = f.hp;
      }
      p.en.style.width = f.energy + "%";
      const st = f.status ? STATUS_NAMES[f.status] : "";
      if (p.status.textContent !== st) { p.status.textContent = st; p.status.style.background = f.status ? STATUS_COLORS[f.status] : "transparent"; }
      const balls = p.team.children;
      for (let i = 0; i < balls.length; i++) {
        const t = B.teams[s][i], d = balls[i];
        const pilot = B.playerSide >= 0 ? B.playerSide : B.commandSide;
        const focus = (B.chosen && B.chosen[s]) || f;
        const pick = s === pilot && !B.turnMode && (B.phase === "fight" || B.phase === "intro") && !!f && !t.ko && t !== focus;
        const cls = "ball" + (t === focus ? " active" : "") + (t.ko ? " ko" : "") + (t.active || t.ko || s === B.playerSide || B.playerSide < 0 ? " seen" : "") + (pick ? " pick" : "");
        if (d.className !== cls) d.className = cls;
        if (!d.drawn && cls.includes("seen")) { d.drawn = drawPortrait(d.firstChild, t, 44); }
      }
      p.callT -= dt; p.comboT -= dt;
      if (p.callT <= 0 && p.call.classList.contains("on")) p.call.classList.remove("on");
      if (p.comboT <= 0 && p.combo.classList.contains("on")) p.combo.classList.remove("on");
    }
    const left = B.duration - B.clock, tt = B.turnMode ? `TURN ${B.turnNo || 1}` : U.fmtTime(left);
    if (timerEl.textContent !== tt) { timerEl.textContent = tt; timerEl.classList.toggle("hurry", !B.turnMode && left < 30 && B.phase === "fight"); }
    if (clashEl) clashEl.classList.toggle("hidden", !B.activeClash);
    if (B.weather.kind) { const n = { sand: "すなあらし", rain: "あめ", sun: "ひざしがつよい", hail: "あられ" }[B.weather.kind]; const w = `${n} ${B.weather.t > 900 ? "∞" : U.fmtTime(B.weather.t)}`; if (weatherEl.textContent !== w) weatherEl.textContent = w; weatherEl.classList.remove("hidden"); } else weatherEl.classList.add("hidden");
    // skills
    const pilotSide = B.playerSide >= 0 ? B.playerSide : B.commandSide;
    const me = pilotSide >= 0 && !B.turnMode ? B.active[pilotSide] : null;
    if (!el.classList.contains("pilot" + pilotSide)) { el.classList.remove("pilot0", "pilot1"); if (pilotSide >= 0) el.classList.add("pilot" + pilotSide); }
    // あいて側のコマンド（見るだけ。かんせんではタップで操作を切り替え）
    const foe = pilotSide >= 0 && !B.turnMode ? B.active[1 - pilotSide] : null;
    if (foe) {
      const who = B.playerSide >= 0 ? `あいて：${foe.species.jp}` : `${pilotSide === 0 ? "あおコーナー" : "あかコーナー"}：${foe.species.jp}`;
      const wEl = $(".who", foeBox); if (wEl.textContent !== who) wEl.textContent = who;
      foeBox.style.setProperty("--fc", pilotSide === 0 ? "#8ab4ff" : "#ff9aa8");
      fcards.forEach((c, i) => {
        const s = foe.moves[i];
        if (!s) { c.root.style.visibility = "hidden"; return; }
        c.root.style.visibility = "";
        if (c.spec !== s) { c.spec = s; c.root.style.setProperty("--tc", Data.typeColor(s.type)[0]); c.jp.textContent = s.jp; c.meta.innerHTML = `${badge(s.type)}<b>⚡${s.cost}</b>`; }
        const cd = Math.max(0, foe.cds[i]);
        c.cd.style.setProperty("--p", (cd > 0 ? (cd / (s.cd || 1)) * 100 : 0).toFixed(1));
        const txt = cd > 0 ? cd.toFixed(1) : ""; if (c.cd.textContent !== txt) c.cd.textContent = txt;
        c.root.classList.toggle("noen", foe.energy < s.cost && cd <= 0);
        c.root.classList.toggle("cast", !!foe.cast && foe.cast.slot === i);
      });
    }
    if (me) {
      cards.forEach((c, i) => {
        const s = me.moves[i];
        if (c.spec !== s) { c.spec = s; c.root.style.setProperty("--tc", Data.typeColor(s.type)[0]); c.jp.textContent = s.jp; c.meta.innerHTML = `${badge(s.type)}<b>⚡${s.cost}</b>${s.power ? `<em>${s.power}</em>` : ""}`; }
        const cd = Math.max(0, me.cds[i]);
        c.cd.style.setProperty("--p", (cd > 0 ? (cd / (s.cd || 1)) * 100 : 0).toFixed(1));
        const txt = cd > 0 ? cd.toFixed(1) : ""; if (c.cd.textContent !== txt) c.cd.textContent = txt;
        c.root.classList.toggle("ready", cd <= 0 && me.energy >= s.cost); c.root.classList.toggle("noen", me.energy < s.cost && cd <= 0);
        c.root.classList.toggle("cast", !!me.cast && me.cast.slot === i);
      });
      const dodgeOff = me.dodgeCd > 0 || me.energy < 15;
      const swapOff = !B.canSwap(pilotSide);
      $("#u-dodge").classList.toggle("off", dodgeOff);
      $("#u-swap").classList.toggle("off", swapOff);
      const cycling = B.commandSide === pilotSide;
      const can = cycling ? B.canCycleForm(me) : B.canMega(me);
      const megaLabel = $("#u-mega span");
      if (megaLabel) megaLabel.textContent = cycling ? "きりかえ" : "メガ";
      $("#u-mega").classList.toggle("off", !can); $("#u-mega").classList.toggle("pulse", can && B.phase === "fight");
      if (tcards) tcards.forEach((tc, i) => {
        const s = me.moves[i];
        if (!s) { tc.root.style.visibility = "hidden"; return; }
        tc.root.style.visibility = "";
        if (tc.spec !== s) {
          tc.spec = s;
          tc.root.style.setProperty("--tc", Data.typeColor(s.type)[0]);
          if (tc.name) tc.name.textContent = s.jp;
          if (tc.cost) tc.cost.textContent = `⚡${s.cost}`;
        }
        const cd = Math.max(0, me.cds[i]);
        if (tc.cd) {
          tc.cd.style.setProperty("--p", (cd > 0 ? (cd / (s.cd || 1)) * 100 : 0).toFixed(1));
          const txt = cd > 0 ? cd.toFixed(1) : "";
          if (tc.cd.textContent !== txt) tc.cd.textContent = txt;
        }
        tc.root.classList.toggle("ready", cd <= 0 && me.energy >= s.cost);
        tc.root.classList.toggle("noen", me.energy < s.cost && cd <= 0);
        tc.root.classList.toggle("cast", !!me.cast && me.cast.slot === i);
      });
      if (tUtils) {
        if (tUtils.dodge) tUtils.dodge.classList.toggle("off", dodgeOff);
        if (tUtils.swap) tUtils.swap.classList.toggle("off", swapOff);
        if (tUtils.megaLbl) tUtils.megaLbl.textContent = cycling ? "きりかえ" : "メガ";
        if (tUtils.mega) { tUtils.mega.classList.toggle("off", !can); tUtils.mega.classList.toggle("pulse", can && B.phase === "fight"); }
      }
    }
    if (vsT > 0) { vsT -= dt; if (vsT <= 0) vs.classList.add("hidden"); }
    for (const e of B.events) handle(e);
    B.events.length = 0;
    // typewriter narration
    if (!typing && lines.length) { typing = { text: lines.shift(), n: 0 }; lineT = 0; msg.classList.remove("hidden"); }
    if (typing) {
      lineT += dt;
      const want = Math.min(typing.text.length, Math.floor(lineT * 34));
      if (want > typing.n) { typing.n = want; msgText.textContent = typing.text.slice(0, want); if (want % 2) Audio.sfx("text"); }
      const hold = lines.length ? 0.55 : 2.2;
      if (typing.n >= typing.text.length && lineT > typing.text.length / 34 + hold) { typing = null; if (!lines.length) msg.classList.add("hidden"); }
    }
  }

  function showBanner(html, cls = "", ms = 950) {
    banner.innerHTML = html; banner.className = ""; void banner.offsetWidth; banner.className = "show " + cls;
    clearTimeout(bannerTimer);
    if (ms) bannerTimer = setTimeout(() => (banner.className = ""), ms);
  }

  function handle(e) {
    const pside = (f) => panels[f.side];
    switch (e.type) {
      case "intro": {
        const sides = [[".a", e.a], [".b", e.b]];
        for (const [sel, f] of sides) { const s = $(sel, vs); drawPortrait($("canvas", s), f, 120); $(".nm", s).textContent = f.species.name.toUpperCase(); $(".jp", s).textContent = f.species.jp; $(".ty", s).innerHTML = f.species.types.map(badge).join(""); }
        $(".lg", vs).textContent = B.label;
        vs.classList.remove("hidden"); vs.style.animation = "none"; void vs.offsetWidth; vs.style.animation = "";
        vsT = 2.5;
        say(B.turnMode ? `やせいの ${e.b.species.jp}が とびだしてきた！` : `${B.label}！ ${e.a.species.jp} たい ${e.b.species.jp}！`);
        break;
      }
      case "ready": say(`${B.active[0].species.jp}は どうする？`); break;
      case "count": showBanner(String(e.n), "num", 700); break;
      case "go": showBanner("GO!", "go"); break;
      case "ko": {
        if (koCutin) {
          koKicker.textContent = e.finish ? "FINAL KNOCKOUT!!" : "KNOCKOUT!!";
          koMain.textContent = "K.O.!";
          koSub.textContent = `${e.f.species.jp} (${e.f.species.name.toUpperCase()}) DOWN!`;
          koCutin.className = "";
          void koCutin.offsetWidth;
          koCutin.className = e.finish ? "show finish" : "show";
          clearTimeout(koTimer);
          koTimer = setTimeout(() => { if (koCutin) koCutin.className = ""; }, e.finish ? 1650 : 1300);
        } else {
          showBanner(`K.O.!<small>BOOM!</small>`, "ko", 1300);
        }
        say(`${e.f.species.jp}は たおれた！`);
        break;
      }
      case "sendout": say(e.side === B.playerSide ? `ゆけっ！ ${e.f.species.jp}！` : B.playerSide < 0 ? `${e.side === 0 ? "あかコーナー" : "あおコーナー"}から ${e.f.species.jp}！` : `あいては ${e.f.species.jp}を くりだした！`); break;
      case "swap": say(`もどれ ${e.from.species.jp}！`); break;
      case "command": {
        const name = B.trainers && B.trainers[e.side] ? B.trainers[e.side].n : "";
        say(`${e.side === 0 ? "あかコーナー" : "あおコーナー"}${name ? `の ${name}` : ""}に しじを だす！`);
        showBanner(e.side === 0 ? "RED CORNER" : "BLUE CORNER", "go", 900);
        const p = panels[e.side].root; p.classList.remove("hit"); void p.offsetWidth; p.classList.add("hit");
        break;
      }
      case "weather": if (e.kind) say({ sand: "すなあらしが ふきあれる！", rain: "あめが ふりはじめた！", sun: "ひざしが つよくなった！", hail: "あられが ふりはじめた！" }[e.kind]); break;
      case "mega": {
        const back = !!e.back;
        showBanner(back ? "もどった！" : e.form.kind === "gigantamax" ? "GIGANTAMAX!" : "MEGA EVOLUTION!", "go long", 1500);
        say(back ? `${e.f.species.jp}は ${e.form.jp}に もどった！` : `${e.f.species.jp}は ${e.form.jp}に ${e.form.kind === "gigantamax" ? "キョダイマックスした！" : "メガシンカした！"}`);
        Audio.sfx("levelup");
        break;
      }
      case "cast": {
        const p = pside(e.f); p.call.textContent = `${e.s.jp}！`; p.call.style.setProperty("--tc", Data.typeColor(e.s.type)[0]);
        p.call.classList.remove("on"); void p.call.offsetWidth; p.call.classList.add("on"); p.callT = 1.6;
        if (B.playerSide < 0 || B.turnMode || e.s.power >= 90) say(`${e.f.species.jp}の ${e.s.jp}！`);
        break;
      }
      case "combo": {
        const p = panels[e.side];
        const enName = e.move && e.move.name ? e.move.name : "Hit";
        const jpName = e.move && e.move.jp ? e.move.jp : "コンボ";
        p.combo.innerHTML = `<span class="cmv">${enName}</span><b>×${e.n}</b><small>${jpName} ×${e.n}</small>`;
        p.combo.classList.remove("on"); void p.combo.offsetWidth; p.combo.classList.add("on"); p.comboT = 1.9;
        break;
      }
      case "clash": {
        if (e.kind === "beam") say("わざと わざが はげしく ぶつかりあう！！");
        break;
      }
      case "say": say(e.text); break;
      case "ability": {
        const p = pside(e.f); if (!p) break;
        p.abil.innerHTML = `<small>${e.f.species.jp}の</small>${e.text}`; p.abil.style.setProperty("--tc", Data.typeColor(e.f.species.types[0])[0]);
        p.abil.classList.remove("on"); void p.abil.offsetWidth; p.abil.classList.add("on");
        if (e.extra) say(`${e.f.species.jp}の ${e.text}！ ${e.extra}`);
        break;
      }
      case "castfail": { const c = cards[e.i]; if (c) { c.root.classList.remove("fail"); void c.root.offsetWidth; c.root.classList.add("fail"); } break; }
      case "end": {
        const w = e.winner;
        if (B.cfg.roundName && w >= 0) { const f = B.teams[w][0]; showBanner(`<small>${B.cfg.roundName} WINNER</small>${f.species.name.toUpperCase()}`, "win round", 0); say(`しょうしゃ ${f.species.jp}！`); }
        else showBanner(w < 0 ? "DRAW" : B.playerSide < 0 ? `<small>WINNER</small>${w === 0 ? "RED CORNER" : "BLUE CORNER"}` : w === B.playerSide ? "<small>WINNER</small>YOU WIN!" : "<small>RESULT</small>YOU LOSE…", "win", 0);
        if (w === B.playerSide) say("バトルに しょうりした！");
        break;
      }
    }
  }

  return { init, show, hide, bind, update, say, setSkin, skin: () => skin };
})();
