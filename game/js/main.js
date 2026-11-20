// Boot, screen flow and the main loop.
// Test URL: index.html?auto=charizard,blastoise;mewtwo,gengar&arena=stadium&mode=watch|play|turn&ff=12&freeze=1&seed=42&cam=x,y&weather=sand
"use strict";

(() => {
  const $ = (s) => document.querySelector(s);
  let battle = null, lastCfg = null, lastHooks = null, state = "loading", paused = false, frozen = false, endShown = false, stage = null;
  let demo = null, demoGen = 0, demoLoading = false, demoHold = 0, demoRound = 0;
  let last = performance.now(), time = 0, fpsAcc = 0, fpsN = 0;
  const q = new URLSearchParams(location.search);

  async function boot() {
    if (!window.ARENA_MONSTERDEX || !window.ARENA_MONSTERDEX.length) { $("#loading-text").textContent = "data/monsterdex.js が みつかりません。"; return; }
    if (!Assets.canFetch) { $("#loading-text").innerHTML = "このゲームは <b>ゲーム起動.command</b> から ひらいてください<br>（file:// では ドットえの しょりが できません）"; return; }
    Data.build();
    Input.init($("#overlay"));
    // かんせん：フィールドのトレーナー（または横の顔わく）をタップすると、そのコーナーへ指示を出す
    addEventListener("pointerdown", (e) => {
      if (!battle || battle.commandSide < 0 || e.button !== 0 || !(e.target instanceof HTMLCanvasElement)) return;
      const s = Renderer.trainerAt(battle, e.clientX, e.clientY);
      if (s < 0) return;
      e.preventDefault(); e.stopPropagation();
      battle.setCommandSide(s);
    }, true);
    Renderer.init($("#screen"), $("#overlay"));
    HUD.init(handleEscape);
    TurnBattle.init(toMenu);
    Menu.init(fromMenu);
    stage = makeStage();
    document.querySelectorAll("[data-act]").forEach((b) => b.addEventListener("click", () => {
      const act = b.dataset.act; Audio.unlock(); Audio.sfx("select");
      if (act === "resume") setPaused(false);
      if (act === "menu") toMenu();
      if (act === "mute") Audio.setMuted(!Audio.isMuted());
      if (act === "again" && lastCfg) startBattle(lastCfg, lastHooks);
    }));
    addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (e.defaultPrevented) return;
        e.preventDefault();
        handleEscape();
      } else {
        Audio.unlock();
      }
    });
    addEventListener("pointerdown", () => Audio.unlock());
    // カメラの引き／寄せ：ホイール、− / ＝ / 0 キー、HUD の − 100% ＋ ボタン。倍率は保存され、演出ズームはその上に乗る。
    const zoomVal = $("#cam-zoom-val");
    const setZoom = (v) => { const z = Renderer.setUserZoom(v); if (zoomVal) zoomVal.textContent = Math.round(z * 100) + "%"; };
    const stepZoom = (dir) => setZoom(dir === 0 ? Renderer.cam.userDefault : Renderer.cam.user * (dir > 0 ? 1.1 : 1 / 1.1));
    setZoom(q.get("zoom") ? +q.get("zoom") : Renderer.cam.user);
    $("#overlay").addEventListener("wheel", (e) => { if (state !== "battle") return; e.preventDefault(); setZoom(Renderer.cam.user * Math.exp(-Math.max(-60, Math.min(60, e.deltaY)) * 0.0022)); }, { passive: false });
    addEventListener("keydown", (e) => {
      if (state !== "battle" || e.ctrlKey || e.metaKey || e.altKey || (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName))) return;
      if (e.key === "-" || e.key === "_") stepZoom(-1); else if (e.key === "=" || e.key === "+") stepZoom(1); else if (e.key === "0") stepZoom(0);
    });
    document.querySelectorAll("#cam-zoom button").forEach((b) => {
      b.addEventListener("pointerdown", (e) => e.stopPropagation());
      b.addEventListener("click", (e) => { e.stopPropagation(); stepZoom(+b.dataset.z); b.blur(); });
    });
    if (!autostart()) {
      state = "menu";
      $("#loading").classList.add("hidden");
      Menu.showTitle(false);
      startDemo();
    } else $("#loading").classList.add("hidden");
    requestAnimationFrame(loop);
  }

  function autostart() {
    const a = q.get("auto");
    if (!a) return false;
    if (q.get("seed")) Math.random = U.seeded(+q.get("seed"));
    const sides = a.split(";").map((s) => s.split(",").map((slug) => { const shiny = slug.endsWith("*"); const sp = Data.bySlug(slug.replace("*", "")); return sp ? { species: sp, shiny } : null; }).filter(Boolean));
    if (sides.length < 2 || !sides[0].length || !sides[1].length) return false;
    const m = q.get("mode") || "watch";
    const rawSkill = q.get("skill");
    const skill = rawSkill == null || rawSkill === "" ? (m === "watch" ? 1 : 0.8) : +rawSkill;
    startBattle({ teams: sides, arenaId: q.get("arena") || "stadium", mode: m === "turn" ? "play" : m, turnMode: m === "turn", skill, duration: +(q.get("time") || 300), weather: q.get("weather"), label: q.get("label") || undefined, roundName: q.get("round") || undefined, hudSkin: q.get("hud") || undefined });
    return true;
  }

  function fromMenu(c) {
    const base = { teams: c.teams, arenaId: c.arena, skill: c.skill, duration: c.duration, trainers: c.trainers, hudSkin: c.hudSkin };
    if (c.mode === "battle") startBattle({ ...base, mode: c.watch ? "watch" : "play", label: "MONSTER LEAGUE" });
    if (c.mode === "training") startBattle({ ...base, mode: "play", cpuAttack: false, label: "ROUND 1 · TRAINING" });
    if (c.mode === "mini") startBattle({ ...base, mode: "play", turnMode: true, label: "MINI BATTLE", duration: 9999 });
    if (c.mode === "tournament") { state = "bracket"; Tournament.start(c, (cfg, onEnd) => startBattle(cfg, { onEnd }), toMenu); }
  }

  async function startBattle(cfg, hooks = null) {
    state = "loading";
    stopDemo();
    lastCfg = cfg; lastHooks = hooks; endShown = false;
    Menu.hideAll();
    $("#result").classList.add("hidden"); $("#pause").classList.add("hidden");
    $("#loading").classList.remove("hidden"); $("#loading-text").textContent = "スタジアムを じゅんびちゅう…";
    HUD.hide(); TurnBattle.detach();
    await new Promise((r) => setTimeout(r, 16));
    const b = new Battle(Object.assign({ deferBench: true }, cfg));
    battle = b;
    const lead = [b.teams[0][0], b.teams[1][0]].filter(Boolean);
    const recs = [...lead.flatMap((f) => f.loadRecs), Assets.ball(), ...b.trainerRecs];
    await Assets.whenReady(recs);
    for (const f of lead) f.measure();
    $("#loading").classList.add("hidden");
    const rest = b.teams.flat().filter((f) => !lead.includes(f));
    if (rest.length) {
      (async () => {
        for (let i = 0; i < rest.length; i += 2) {
          if (battle !== b || state !== "battle") break;
          const batch = rest.slice(i, i + 2);
          await Assets.whenReady(batch.flatMap((f) => f.loadRecs), 8000);
          if (battle !== b) break;
          for (const f of batch) f.measure();
        }
      })();
    }
    if (cfg.hudSkin && HUD.setSkin) HUD.setSkin(cfg.hudSkin);
    HUD.bind(battle); HUD.show();
    Renderer.updateCamera(battle, 0, true);
    battle.start();
    if (cfg.weather) battle.setWeather(cfg.weather, 9999);
    if (cfg.turnMode) TurnBattle.attach(battle);
    if (q.get("cam")) { const [x, y] = q.get("cam").split(",").map(Number); window.CAM_OVERRIDE = { x, y }; }
    state = "battle"; paused = false; frozen = false;
    syncTouchUi();
    const ff = +(q.get("ff") || 0);
    if (ff > 0 && q.get("auto")) {
      for (let t = 0; t < ff; t += 1 / 60) { battle.update(1 / 60, battle.playerSide >= 0 && !battle.turnMode ? AI.act(battle.active[0] || battle.teams[0][0], battle, 1 / 60, 0.8) : null); TurnBattle.update(1 / 60); Renderer.updateCamera(battle, 1 / 60, false); HUD.update(battle, 1 / 60); }
      if (q.get("freeze") === "1") { frozen = true; battle.fx.flash = 0; battle.fx.invert = battle.fx.invertN = 0; battle.fx.blur = 0; }
    }
    window.BATTLE = battle;
  }

  function syncTouchUi() {
    document.body.classList.toggle("in-battle", state === "battle" && !paused && !endShown && !(battle && battle.turnMode));
  }
  function handleEscape() {
    Audio.unlock();
    if (state === "battle") {
      if (!$("#result").classList.contains("hidden")) { Audio.sfx("back"); toMenu(); return; }
      if (paused) { Audio.sfx("select"); setPaused(false); return; }
      if (battle && battle.turnMode && TurnBattle.back && TurnBattle.back()) return;
      Audio.sfx("cursor");
      setPaused(true);
      return;
    }
    if (state === "bracket" && Tournament.exit && Tournament.exit()) { Audio.sfx("back"); return; }
    if (Menu.escape) Menu.escape();
  }
  function setPaused(p) {
    paused = p;
    $("#pause").classList.toggle("hidden", !p);
    if (Input.resetTouch) Input.resetTouch();
    syncTouchUi();
  }
  function toMenu() {
    battle = null; state = "menu"; paused = false; endShown = false; window.BATTLE = null;
    if (Input.resetTouch) Input.resetTouch();
    syncTouchUi();
    $("#pause").classList.add("hidden"); $("#result").classList.add("hidden"); $("#bracket").classList.add("hidden");
    HUD.hide(); TurnBattle.detach(); Audio.stopMusic(); Audio.stopCrowd();
    const c = $("#overlay").getContext("2d"); c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, 1e5, 1e5);
    Menu.showTitle(true);
    if (!demo) startDemo();
  }

  function titleOn() {
    const t = $("#title");
    return !!(t && !t.classList.contains("hidden"));
  }
  function demoTeams() {
    const pool = Data.roster().filter((p) => p.kind === "gen" && p.pool && p.pool.length && p.moves && p.moves.length);
    const picks = [];
    let guard = 0;
    while (picks.length < 6 && pool.length && guard++ < 40) {
      const p = U.pick(pool);
      if (p && !picks.includes(p)) picks.push(p);
    }
    const half = Math.max(1, Math.floor(picks.length / 2));
    const mk = (list) => list.map((species) => ({ species, shiny: false }));
    return [mk(picks.slice(0, half)), mk(picks.slice(half))];
  }
  function syncDemoAudio(on) {
    if (!on) { Audio.stopMusic(); Audio.stopCrowd(); return; }
    Audio.startCrowd();
    Audio.startMusic(demo.arena.night ? 150 : 140, demo.arena.night ? 110 : 130.8);
  }
  function stopDemo() {
    demoGen++;
    demo = null;
    demoHold = 0;
    demoLoading = false;
    Audio.stopMusic();
    Audio.stopCrowd();
    if (q.get("dbg") === "1") window.__exhibit = null;
  }
  async function startDemo() {
    if (state === "battle" || demoLoading) return;
    const gen = ++demoGen;
    demoLoading = true;
    const arenas = Arenas.list();
    const round = demoRound++;
    const cfg = {
      teams: demoTeams(),
      arenaId: round === 0 || !arenas.length ? "stadium" : U.pick(arenas).id,
      seed: (Math.random() * 1e9) | 0,
      mode: "watch",
      skill: 1,
      duration: 40,
      demo: true,
      label: "EXHIBITION",
    };
    try {
      const b = new Battle(Object.assign({ deferBench: true }, cfg));
      const lead = [b.teams[0][0], b.teams[1][0]].filter(Boolean);
      if (lead.length < 2) return;
      const recs = [...lead.flatMap((f) => f.loadRecs), Assets.ball(), ...b.trainerRecs];
      await Assets.whenReady(recs);
      if (gen !== demoGen || state === "battle") return;
      for (const f of lead) f.measure();
      b.start();
      demo = b;
      demo._snap = true;
      demoHold = 0;
      if (q.get("dbg") === "1") window.__exhibit = demo;
      const rest = b.teams.flat().filter((f) => !lead.includes(f));
      if (rest.length) {
        (async () => {
          for (let i = 0; i < rest.length; i += 2) {
            if (demo !== b || state === "battle") break;
            const batch = rest.slice(i, i + 2);
            await Assets.whenReady(batch.flatMap((f) => f.loadRecs), 8000);
            if (demo !== b || state === "battle") break;
            for (const f of batch) f.measure();
          }
        })();
      }
    } finally {
      if (gen === demoGen) demoLoading = false;
    }
  }

  function showResult(B) {
    if (Input.resetTouch) Input.resetTouch();
    syncTouchUi();
    const w = B.winner;
    $("#result-title").textContent = w < 0 ? "ひきわけ" : B.playerSide < 0 ? (w === 0 ? "あかコーナーの かち！" : "あおコーナーの かち！") : w === B.playerSide ? "YOU WIN!" : "YOU LOSE…";
    const alive = [0, 1].map((s) => B.teams[s].filter((f) => !f.ko).length);
    $("#result-sub").textContent = `${B.arena.def.jp} ・ ${B.turnMode ? `${B.turnNo || 1} ターン` : U.fmtTime(B.clock)} ・ のこり ${alive[0]} たい ${alive[1]}`;
    const mvp = B.teams.flat().sort((a, b) => b.dealt - a.dealt)[0];
    $("#result-stats").innerHTML = `<div><b>${B.stats.hits}</b><small>HITS</small></div><div><b>${B.stats.maxHit}</b><small>MAX DAMAGE</small></div><div><b>${mvp ? mvp.species.jp : "-"}</b><small>MVP ${mvp ? mvp.dealt : 0}</small></div>` +
      `<div class="log">${B.stats.koLog.map((k) => `${U.fmtTime(k.t)}　${k.f} ← ${k.by}`).join("<br>") || "KO なし"}</div>`;
    $("#result").classList.remove("hidden");
  }

  // Lit field behind the select screen, and behind Press Start until the exhibition is ready.
  function makeStage() {
    const arena = Arenas.create("cavern", 3);
    const fx = new FX(arena);
    const want = ["pikachu", "eevee", "charizard", "snorlax", "gyarados", "onix", "wailord"];
    const mons = want.map(Data.bySlug).filter(Boolean);
    mons.sort((a, b) => a.h - b.h);
    const picked = [];
    for (const p of mons) if (!picked.includes(p)) picked.push(p);
    const show = picked.length > 4 ? [picked[0], picked[Math.floor(picked.length / 3)], picked[Math.floor((picked.length * 2) / 3)], picked[picked.length - 1]] : picked;
    const field = arena.field;
    const makeShowcase = () => {
      const team = show.map((species, i) => {
        const mon = new Fighter(species, 0, i);
        const span = field.w * 0.62;
        mon.placeAt(field.cx - span / 2 + span * ((i + 0.5) / Math.max(1, show.length)), field.cy + field.h * 0.12, i % 2 ? Math.PI : 0);
        mon.active = true;
        return mon;
      });
      const TR = window.ARENA_TRAINERS || [];
      const trainers = [];
      for (let i = 0; i < TR.length && trainers.length < 2; i++) {
        const t = TR[(i * 19 + 4) % TR.length];
        if (!trainers.includes(t)) trainers.push(t);
      }
      return { team, trainers, trainerRecs: trainers.map((t) => Assets.get("../" + t.src, { raw: true })) };
    };
    const { team, trainers, trainerRecs } = Assets.defer ? Assets.defer(makeShowcase) : makeShowcase();
    const st = {
      arena, fx, teams: [team, []], active: [], clones: [], attacks: [],
      weather: { kind: null }, trainers, trainerRecs,
      _loadStage() {
        if (st._loaded) return;
        st._loaded = true;
        if (Assets.start) {
          Assets.start(team.flatMap((f) => f.loadRecs));
          Assets.start(trainerRecs);
        }
      },
      // Same posts as a fight. Feet sit 12 art px below the spot, so the spot is that far above the showcase line.
      trainerSpot: [
        { x: field.x - 22, y: field.cy + field.h * 0.12 - 12 * Arenas.K },
        { x: field.x + field.w + 22, y: field.cy + field.h * 0.12 - 12 * Arenas.K },
      ],
      trainerThrow: [-9, -9], hype: 0.2, lightsOn: 1, psy: null, time: 0, idlePan: true, playerSide: -1,
    };
    return st;
  }

  function loop(now) {
    requestAnimationFrame(loop);
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now; time += dt;
    if ((state === "menu" || state === "bracket") && !battle) {
      const open = titleOn();
      if (open && demo) {
        if (demo.phase !== "end") { demo.update(dt, null); demoHold = 0; }
        else {
          demo.fx.update(dt);
          demoHold += dt;
          if (demoHold > 1.8) startDemo();
        }
        demo.events.length = 0;
        Renderer.updateCamera(demo, dt, !!demo._snap);
        demo._snap = false;
        Renderer.draw(demo, time);
        syncDemoAudio(demo.phase === "fight" || demo.phase === "ko");
        return;
      }
      syncDemoAudio(false);
      if (open && !demo && !demoLoading) startDemo();
      if (!stage) return;
      if (!open && stage._loadStage) stage._loadStage();
      stage.time = time;
      for (const f of stage.teams.flat()) {
        if ((!f.measured && f.sprites.front.bounds) || (f.d8 && !f.measured8 && f.d8[0] && f.d8[0].bounds)) f.measure();
        Dir8.update(f, dt);
        f.anim += dt * (f.tempo || 1);
        f.bob = f.flies ? (f.hoverH || 4) + Math.sin(time * (1.3 + (f.tempo || 1) * 0.4) + f.slot) * 1.7 : 0;
        f.active = true;
      }
      stage.fx.update(dt);
      Renderer.updateCamera(stage, dt, false);
      Renderer.draw(stage, time);
      return;
    }
    if (state !== "battle" || !battle) return;
    if (!paused && !frozen) {
      const pilotSide = battle.playerSide >= 0 ? battle.playerSide : battle.commandSide;
      const me = pilotSide >= 0 && !battle.turnMode ? battle.active[pilotSide] : null;
      // ヒットストップ中に押した技・回避・進化は、止まっている間に捨てず次のフレームへ残す。
      const holdKeys = !battle.turnMode && battle.fx.hitstop > 0;
      const act = holdKeys ? null : Input.poll(dt, Renderer.toWorld, me);
      if (act && act.pause) setPaused(true);
      const steps = !act || dt <= 1 / 45 ? 1 : 2;
      for (let i = 0; i < steps; i++) battle.update((act ? dt / steps : dt), battle.turnMode || !act ? null : i === 0 ? act : { ...act, skill: [false, false, false, false], dodge: false, swap: false, mega: false });
      TurnBattle.update(dt);
      Renderer.updateCamera(battle, dt, false);
      HUD.update(battle, dt);
      if (battle.phase === "end" && !endShown) {
        endShown = true;
        const B = battle, hooks = lastHooks;
        setTimeout(() => { if (battle !== B) return; if (hooks && hooks.onEnd) { battle = null; state = "bracket"; syncTouchUi(); if (Input.resetTouch) Input.resetTouch(); HUD.hide(); Audio.stopMusic(); Audio.stopCrowd(); const c = $("#overlay").getContext("2d"); c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, 1e5, 1e5); hooks.onEnd(B.winner); } else showResult(B); }, hooks && hooks.onEnd ? 4200 : 3000);
      }
    } else if (frozen) { Input.poll(dt, Renderer.toWorld, null); HUD.update(battle, 0); }
    if (!battle) return;
    Renderer.draw(battle, time);
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 0.5) { window.__fps = Math.round(fpsN / fpsAcc); if (q.get("dbg") === "1") document.title = `${window.__fps} fps · ${battle.fx.parts.length} p`; fpsAcc = 0; fpsN = 0; }
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
