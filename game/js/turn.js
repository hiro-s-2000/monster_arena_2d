// Turn-based mini battle (the website mini-game): a DQ command window drives a Battle in turnMode.
// Moves play their real arena animation with damage suppressed (visualOnly); the executor then applies
// one damage roll per move, resolves status per turn and narrates every step.
"use strict";

const TurnBattle = (() => {
  const $ = (s, r = document) => r.querySelector(s);
  let B = null, menu, state = "idle", tweens = [], waits = [], cursor = 0, items = [], view = "main", onExit = null;

  function init(exitCb) {
    onExit = exitCb;
    menu = $("#turnmenu");
    addEventListener("keydown", (e) => {
      if (!B || state !== "command") return;
      const k = e.key.toLowerCase();
      if (k === "arrowdown" || k === "s") move(view === "moves" ? 2 : 1);
      else if (k === "arrowup" || k === "w") move(view === "moves" ? -2 : -1);
      else if (k === "arrowright" || k === "d") move(1);
      else if (k === "arrowleft" || k === "a") move(-1);
      else if (k === "z" || k === "enter" || k === " ") choose(cursor);
      else if (k === "x" || k === "backspace" || k === "escape") { if (view !== "main") { Audio.sfx("back"); open("main"); } }
      else return;
      e.preventDefault(); e.stopPropagation();
    }, true);
  }
  function move(d) { if (!items.length) return; cursor = (cursor + d + items.length) % items.length; Audio.sfx("cursor"); paint(); }

  function attach(b) { B = b; B.turnNo = 1; state = "wait"; tweens = []; waits = []; menu.classList.add("hidden"); }
  function detach() { B = null; state = "idle"; menu.classList.add("hidden"); }

  // ---- menus --------------------------------------------------------------------------------------
  function open(v) {
    view = v; cursor = 0;
    const me = B.active[0], foe = B.active[1];
    if (v === "main") items = [{ t: "たたかう", go: () => open("moves") }, { t: "モンスター", go: () => open("team") }, { t: "どうぐ", dis: true }, { t: "にげる", go: run }];
    if (v === "moves") items = me.moves.map((s, i) => {
      const eff = s.power ? Data.effectiveness(s.type, foe.species.types) : 1;
      const tag = !s.power ? "へんか" : eff === 0 ? "こうかなし" : eff > 1 ? "ばつぐん▲" : eff < 1 ? "いまひとつ▼" : "";
      return { t: s.jp, sub: `${Data.typeName(s.type)}${s.power ? " いりょく" + s.power : ""}`, tag, eff, type: s.type, go: () => commit({ kind: "move", i }) };
    });
    if (v === "team") items = B.teams[0].map((f) => ({ t: f.species.jp, sub: `HP ${Math.max(0, Math.ceil(f.hp))}/${f.maxHp}`, dis: f.ko || f === me, go: () => commit({ kind: "swap", to: f }) }));
    paint();
  }
  function paint() {
    menu.classList.remove("hidden");
    menu.className = "dqwin turn-" + view;
    menu.innerHTML = (view !== "main" ? `<div class="hint">X: もどる</div>` : "") + items.map((it, i) =>
      `<button class="opt${i === cursor ? " sel" : ""}${it.dis ? " dis" : ""}" data-i="${i}" ${it.type ? `style="--tc:${Data.typeColor(it.type)[0]}"` : ""}><span class="t">${it.t}</span>${it.sub ? `<span class="sub">${it.sub}</span>` : ""}${it.tag ? `<span class="tag ${it.eff > 1 ? "up" : it.eff < 1 ? "dn" : ""}">${it.tag}</span>` : ""}</button>`).join("");
    menu.querySelectorAll(".opt").forEach((b) => {
      b.addEventListener("mouseenter", () => { const i = +b.dataset.i; if (i !== cursor) { cursor = i; Audio.sfx("cursor"); menu.querySelectorAll(".opt").forEach((x, k) => x.classList.toggle("sel", k === cursor)); } });
      b.addEventListener("click", () => choose(+b.dataset.i));
    });
  }
  function choose(i) { const it = items[i]; if (!it || it.dis) { Audio.sfx("back"); return; } Audio.sfx("select"); it.go(); }
  function run() { menu.classList.add("hidden"); state = "busy"; HUD.say("うまく にげきれた！"); wait(1.6).then(() => onExit && onExit()); }

  // ---- timing primitives (advance only while the game runs) ---------------------------------------
  const wait = (t) => new Promise((res) => waits.push({ t, res }));
  const tween = (f, x, y, dur) => new Promise((res) => tweens.push({ f, x0: f.x, y0: f.y, x, y, t: 0, dur, res }));
  function until(fn, max = 5) { return new Promise((res) => { const w = { t: max, res, fn }; waits.push(w); }); }

  function update(dt) {
    if (!B) return;
    for (let i = tweens.length - 1; i >= 0; i--) {
      const w = tweens[i]; w.t += dt; const k = Math.min(1, w.t / w.dur), e = U.easeInOut(k);
      w.f.x = U.lerp(w.x0, w.x, e); w.f.y = U.lerp(w.y0, w.y, e);
      const dx = w.x - w.x0, dy = w.y - w.y0; w.f.vx = k < 1 ? dx / w.dur : 0; w.f.vy = k < 1 ? dy / w.dur : 0;
      if (k < 1 && Math.hypot(dx, dy) > 4) {
        const o = B.active[1 - w.f.side];
        w.f.facing = o && !o.ko ? U.angle(w.f.x, w.f.y, o.x, o.y) : Math.atan2(dy, dx);
      }
      if (k >= 1) { tweens.splice(i, 1); w.res(); }
    }
    for (let i = waits.length - 1; i >= 0; i--) { const w = waits[i]; w.t -= dt; if (w.t <= 0 || (w.fn && w.fn())) { waits.splice(i, 1); w.res(); } }
    // face each other while idle
    for (const f of B.active) if (f && !f.cast && !tweens.some((w) => w.f === f)) { const o = B.active[1 - f.side]; if (o) f.facing = U.angle(f.x, f.y, o.x, o.y); }
    if (state === "wait" && B.phase === "fight" && B.active.every((f) => f && f.enter <= 0 && !f.ko)) { state = "command"; HUD.say(`${B.active[0].species.jp}は どうする？`); open("main"); }
  }

  // ---- turn execution -------------------------------------------------------------------------------
  function enemyChoice(foe, me) {
    let best = 0, bv = -1;
    foe.moves.forEach((s, i) => {
      let v = s.power ? s.power * Data.effectiveness(s.type, me.species.types) * (foe.species.types.includes(s.type) ? 1.5 : 1) * (s.cat === "p" ? foe.stats.atk : foe.stats.spa) / 100 : (me.status ? 0 : 25);
      if (s.kind === "heal") v = foe.hpRatio < 0.4 ? 90 : 0;
      v *= U.rand(0.75, 1.25);
      if (v > bv) { bv = v; best = i; }
    });
    return { kind: "move", i: best };
  }

  async function commit(act) {
    menu.classList.add("hidden"); state = "busy";
    const me = B.active[0], foe = B.active[1];
    const foeAct = enemyChoice(foe, me);
    const order = [];
    if (act.kind === "swap") { order.push([0, act]); order.push([1, foeAct]); }
    else {
      const pa = me.moves[act.i].pri, pb = foe.moves[foeAct.i].pri;
      const sa = me.stats.spe * Data.stageMul(me.stages.spe) * (me.status === "para" ? 0.5 : 1), sb = foe.stats.spe * Data.stageMul(foe.stages.spe) * (foe.status === "para" ? 0.5 : 1);
      const first = pa !== pb ? (pa > pb ? 0 : 1) : sa === sb ? (Math.random() < 0.5 ? 0 : 1) : sa > sb ? 0 : 1;
      order.push(first === 0 ? [0, act] : [1, foeAct]); order.push(first === 0 ? [1, foeAct] : [0, act]);
    }
    for (const [side, a] of order) {
      if (B.phase === "end") break;
      const user = B.active[side], target = B.active[1 - side];
      if (!user || user.ko || !target) continue;
      if (a.kind === "swap") { B.swapNext(0, a.to); await wait(0.6); await until(() => B.active[0] && B.active[0].enter <= 0, 3); continue; }
      if (target.ko || user !== (side === 0 ? B.active[0] : B.active[1])) continue;
      await doMove(user, target, user.moves[a.i]);
      await wait(0.35);
      if (B.active.some((f) => f && f.ko)) break;
    }
    // end of turn: residual damage
    for (const f of B.active) {
      if (!f || f.ko || B.phase === "end") continue;
      if (f.status === "burn" || f.status === "psn") { const n = Math.max(1, Math.round(f.maxHp * (f.status === "psn" ? 0.125 : 0.0625))); HUD.say(`${f.species.jp}は ${f.status === "psn" ? "どく" : "やけど"}の ダメージを うけた！`); B.fx.burst(f.status === "psn" ? "poison" : "fire", f.x, f.y - 30, 10, { size: 0.9 }); B.selfDamage(f, n, STATUS_COLORS[f.status]); await wait(0.9); }
      if (f.seedT > 0 && f.seedBy && !f.seedBy.ko && !f.ko) { const n = Math.max(1, Math.round(f.maxHp * 0.125)); HUD.say(`やどりぎが ${f.species.jp}の たいりょくを うばう！`); B.selfDamage(f, n, STATUS_COLORS.seed); f.seedBy.heal(n); await wait(0.9); }
    }
    // faint handling: Battle auto-sends the next Pokémon after a KO
    if (B.active.some((f) => f && f.ko) && B.phase !== "end") { await wait(2.4); await until(() => B.phase === "end" || B.active.every((f) => f && !f.ko && f.enter <= 0), 6); }
    if (B.phase === "end") { state = "done"; return; }
    B.turnNo++;
    state = "wait";
  }

  function approachKinds(s) { return ["melee", "dash", "aoe", "spiral"].includes(s.kind); }

  async function doMove(user, target, s) {
    // status gates
    if (user.status === "slp") { if ((user.slpTurns = (user.slpTurns === undefined ? U.irand(1, 3) : user.slpTurns) - 1) > 0) { HUD.say(`${user.species.jp}は ぐうぐう ねむっている…`); await wait(1.1); return; } user.cure(); user.slpTurns = undefined; HUD.say(`${user.species.jp}は めを さました！`); await wait(0.9); }
    if (user.status === "frz") { if (Math.random() > 0.2) { HUD.say(`${user.species.jp}は こおって しまって うごかない！`); await wait(1.1); return; } user.cure(); HUD.say(`${user.species.jp}の こおりが とけた！`); await wait(0.9); }
    if (user.status === "para" && Math.random() < 0.25) { HUD.say(`${user.species.jp}は からだが しびれて うごけない！`); B.fx.burst("electric", user.x, user.y - 30, 10); await wait(1.1); return; }
    if (user.confT > 0) { user.confTurns = (user.confTurns || U.irand(2, 4)) - 1; if (user.confTurns <= 0) { user.confT = 0; HUD.say(`${user.species.jp}の こんらんが とけた！`); await wait(0.8); } else if (Math.random() < 0.33) { HUD.say(`わけも わからず じぶんを こうげきした！`); B.selfDamage(user, Math.max(1, Math.round(user.maxHp * 0.1))); await wait(1.1); return; } }
    const home = { x: user.x, y: user.y };
    const ang = U.angle(user.x, user.y, target.x, target.y);
    B.event("cast", { f: user, s });
    if (s.power && Math.random() > (s.acc || 100) / 100 + 0.05) { await wait(0.8); HUD.say("しかし こうげきは はずれた！"); await wait(1); return; }
    if (approachKinds(s)) {
      const stop = s.kind === "melee" ? user.r + target.r + 30 : s.kind === "dash" ? 200 : 110;
      await tween(user, target.x - Math.cos(ang) * stop, target.y - Math.sin(ang) * stop, 0.38);
    }
    B.visualOnly = !!s.power; B.turnHits = 0;
    user.lockTarget = target; user.facing = user.aim = U.angle(user.x, user.y, target.x, target.y);
    user.cast = { slot: user.moves.indexOf(s), spec: s, t: 0, fired: false, aim: user.aim };
    await until(() => !user.cast && !user.leap && !B.attacks.some((a) => a.owner === user), 4.5);
    await wait(0.15);
    B.visualOnly = false;
    if (s.power && !target.ko) {
      const res = B.applyTurnDamage(target, user, s);
      await wait(0.5);
      if (res.eff > 1) { HUD.say("こうかは ばつぐんだ！"); await wait(0.7); }
      else if (res.eff === 0) { HUD.say(`${target.species.jp}には こうかが ないようだ…`); await wait(0.7); }
      else if (res.eff < 1) { HUD.say("こうかは いまひとつの ようだ…"); await wait(0.7); }
      if (res.crit) { HUD.say("きゅうしょに あたった！"); await wait(0.6); }
      if (res.status) { HUD.say(`${target.species.jp}は ${STATUS_NAMES[res.status]}状態に なった！`); await wait(0.7); }
    }
    if (approachKinds(s) || s.kind === "leap") await tween(user, home.x, home.y, 0.4);
  }

  return { init, attach, detach, update, active: () => !!B };
})();
