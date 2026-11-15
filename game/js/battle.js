// Match orchestration: teams, send-outs, the damage pipeline, KO flow, weather, timer and the event
// stream the HUD consumes. Both sides are Fighters; the player side (if any) gets the Input action frame.
"use strict";

class Battle {
  constructor(cfg) {
    this.cfg = cfg;
    // [ユーザー指定 2026-11-15] フィールドの広さは出場するモンスターの大きさで決める。いちばん大きい体（キョダイマックスは数えない）が
    // トレーナー（1.7m、常に同じ大きさ）と比べてどれだけ大きいかで、0.78〜1.75 倍。小さいモンスターだけなら狭いコートになる。
    this.fieldScale = Battle.fieldScaleFor(cfg.teams);
    this.arena = Arenas.create(cfg.arenaId || "stadium", cfg.seed || 7, this.fieldScale);
    this.fx = new FX(this.arena);
    this.arena.battle = this;
    this.attacks = [];
    this.timers = [];
    this.events = [];
    this.mode = cfg.mode || "play";
    this.playerSide = this.mode === "play" ? 0 : -1;
    // かんせんは観戦のまま、あかコーナー（side 0・左）へ技・回避・進化の指示だけ乗せる。
    // タイトルのエキシビションは入力を一切受けない。
    this.commandSide = this.mode === "watch" && !cfg.demo ? 0 : -1;
    this.skill = cfg.skill === undefined ? (this.mode === "watch" ? 1 : 0.8) : cfg.skill;
    this.cpuCdMul = 1 + (1 - this.skill) * 0.6;
    this.cpuAttack = cfg.cpuAttack !== false;
    this.duration = cfg.duration || 300;
    this.clock = 0; this.time = 0; this.phase = "intro"; this.introT = 0;
    this.weather = { kind: this.arena.weather, t: this.arena.weather ? 9999 : 0, tick: 0 };
    this.hype = 0;
    this.winner = -1;
    this.megaUsed = [false, false];
    this.swapCd = [0, 0];
    this.chosen = [null, null];
    this.swapToken = [0, 0];
    const makeF = (p, side, i) => new Fighter(p.species || p, side, i, { shiny: p.shiny });
    this.teams = cfg.teams.map((list, side) => list.map((p, i) => (cfg.deferBench && i > 0 && typeof Assets !== "undefined" && Assets.defer ? Assets.defer(() => makeF(p, side, i)) : makeF(p, side, i))));
    if (Dir8.trim) Dir8.trim(new Set(this.teams.flat().map((f) => f.species.id))); // この試合に出ないモンスターの 8 方向シートは捨てる
    this.active = [null, null];
    this.stats = { hits: 0, maxHit: 0, koLog: [] };
    this.label = cfg.label || "MONSTER LEAGUE";
    this.turnMode = !!cfg.turnMode; this.visualOnly = false;
    this.clashes = []; this.activeClash = null; this.camFocus = null; this.koCam = null;
    this.trainerThrow = [-9, -9];
    // two different random trainers (HGSS / FRLG overworld sprites, data/trainers.js)
    const TR = window.ARENA_TRAINERS || [];
    this.trainers = cfg.trainers || [];
    while (TR.length && this.trainers.length < 2) { const t = U.pick(TR); if (!this.trainers.includes(t)) this.trainers.push(t); }
    this.trainerRecs = this.trainers.map((t) => Assets.get("../" + t.src, { raw: true }));
    this.lightsOn = 0;
    this.clones = [];
    this.combo = [{ n: 0, t: -9 }, { n: 0, t: -9 }];
  }

  // ---- lifecycle ------------------------------------------------------------------------------------
  start() {
    this.phase = "intro";
    const a = this.teams[0][0], b = this.teams[1][0];
    this.arena.ticker = `${this.label} · ${a.species.name} VS ${b.species.name}`;
    if (this.cfg.demo) { this.beginExhibition(); return; }
    // lights come on one bank at a time (the dark-arena intro from the trailers)
    this.lightsOn = 0.05;
    const banks = this.arena.lights.filter((l) => l.flood).length || 1;
    for (let i = 0; i <= banks; i++) this.after(0.25 + i * 0.32, () => { this.lightsOn = Math.min(1, 0.2 + (i + 1) * (0.8 / (banks + 1)) + 0.15 * i); Audio.sfx("clunk"); });
    this.after(0.3 + banks * 0.32 + 0.3, () => { this.lightsOn = 1; this.event("intro", { a, b }); Audio.sfx("crowdHit"); Audio.setHype(0.8); this.hype = 0.8; });
    const t0 = 0.6 + banks * 0.32;
    if (this.turnMode) { this.after(t0 + 0.5, () => this.sendOut(0, 0)); this.after(t0 + 0.9, () => this.sendOut(1, 0)); this.after(t0 + 2.2, () => { this.phase = "fight"; this.event("ready"); Audio.startMusic(140, 110); }); return; }
    this.after(t0 + 0.5, () => this.sendOut(0, 0));
    this.after(t0 + 0.9, () => this.sendOut(1, 0));
    this.after(t0 + 2.4, () => { this.event("count", { n: 3 }); Audio.sfx("count"); });
    this.after(t0 + 3.2, () => { this.event("count", { n: 2 }); Audio.sfx("count"); });
    this.after(t0 + 4.0, () => { this.event("count", { n: 1 }); Audio.sfx("count"); });
    this.after(t0 + 4.8, () => { this.event("go"); Audio.sfx("go"); this.phase = "fight"; Audio.startCrowd(); Audio.startMusic(this.arena.night ? 150 : 140, this.arena.night ? 110 : 130.8); });
  }

  // Title card: the match is already going. No countdown, no ball arc, trainers already on the sidelines.
  beginExhibition() {
    this.lightsOn = 1;
    this.phase = "fight";
    const a = this.teams[0][0], b = this.teams[1][0];
    if (a && b) this.arena.ticker = `${this.label} · ${a.species.name} VS ${b.species.name}`;
    this.sendOut(0, 0);
    this.sendOut(1, 0);
    this.trainerThrow = [-9, -9];
    // Same sideline posts as a real match (drawTrainers uses field.x ± 22 when trainerSpot is absent).
    this.trainerSpot = null;
  }

  static ballArc(f, uOverride) {
    if (!f || !f.enterFrom) return null;
    const u = uOverride !== undefined ? U.clamp(uOverride, 0, 1) : U.clamp((1.05 - f.enter) / 0.53, 0, 1);
    const uh = 1 - Math.pow(1 - u, 1.35);
    const from = f.enterFrom, fromZ = from.z !== undefined ? from.z : 30;
    const x = U.lerp(from.x, f.x, uh), y = U.lerp(from.y, f.y, uh);
    const z = U.lerp(fromZ, 32, u) + Math.sin(u * Math.PI) * 185;
    return { x, y, z, u, uh };
  }
  trainerHandPos(side) {
    const fld = this.arena.field, K = Arenas.K || 2, spot = this.trainerSpot && this.trainerSpot[side];
    const wx = spot ? spot.x : (side === 0 ? fld.x - 22 : fld.x + fld.w + 22);
    const wy = spot ? spot.y : fld.cy;
    return { x: wx + (side === 0 ? 4 * K : -4 * K), y: wy + 12 * K, z: Math.round(6.5 * K) };
  }
  sendOut(side, idx) {
    const f = this.teams[side][idx];
    if (!f || f.ko) return;
    if (typeof Assets !== "undefined" && Assets.start) Assets.start(f.loadRecs);
    const fld = this.arena.field;
    const x = side === 0 ? fld.x + fld.w * 0.3 : fld.x + fld.w * 0.7;
    const y = fld.cy + (side ? -40 : 40);
    const prev = this.active[side];
    const hand = this.trainerHandPos(side);
    if (prev && prev !== f) {
      const wasOut = prev.active && !prev.ko;
      if (prev.active) Ability.onSwitchOut(this, prev);
      prev.active = false; prev.ai = null;
      if (wasOut && !(prev.out > 0)) { prev.out = 0.42; prev.outMax = 0.42; prev.recallTo = hand; }
    }
    f.placeAt(x, y, side === 0 ? 0 : Math.PI);
    if (this.cfg.demo) {
      f.enter = 0; f.enterFrom = null; f.active = true; f.z = 0;
    } else {
      f.startEnter(hand.x, hand.y, hand.z);
      this.fx.add({ x: hand.x, y: hand.y, z: hand.z, life: 0.18, size: 18, color: "#ffffff", shape: "orb", add: true, drag: 0 });
      this.fx.ring(hand.x, hand.y, 4, 28, 0.22, side === 0 ? "#ff6688" : "#66b2ff", 3, { ground: false, z: hand.z, squash: 0.8 });
      Audio.sfx("ballThrow", this.pan(hand.x));
      this.trainerThrow[side] = this.time;
    }
    f.controlled = side === this.playerSide;
    this.active[side] = f;
    this.chosen[side] = null;
    this.event("sendout", { f, side });
    this.arena.ticker = `${this.label} · ${(this.active[0] || f).species.name} VS ${(this.active[1] || f).species.name}`;
    for (const s of [0, 1]) if (this.active[s]) this.active[s].lockTarget = this.active[1 - s];
  }
  onBallTrail(f, dt) {
    f.enterTrailT = (f.enterTrailT || 0) - dt;
    if (f.enterTrailT > 0) return;
    const arc = Battle.ballArc(f);
    if (!arc) return;
    f.enterTrailT = 0.026;
    const th = this.fx.theme(f.species.types[0]);
    const color = arc.u > 0.72 ? "#ffffff" : th.c[1];
    this.fx.ballTrail(arc.x, arc.y, arc.z, (f.x - f.enterFrom.x) / 0.53, (f.y - f.enterFrom.y) / 0.53, Math.cos(arc.u * Math.PI) * 350, color);
  }
  onBallOpen(f) {
    this.fx.ballBurst(f.x, f.y, f.species.types[0], !!f.species.leg);
    Audio.sfx("ballOpen", this.pan(f.x));
  }
  onEntered(f) {
    if (f.species.leg) { // legendary entrance: roar + pillar of light
      this.fx.shakeCam(12, 0.5); this.fx.flashScreen(0.42, Data.typeColor(f.species.types[0])[1]);
      this.fx.ring(f.x, f.y, 20, 220, 0.8, Data.typeColor(f.species.types[0])[1], 8);
      this.fx.burst(f.species.types[0], f.x, f.y - f.height * 0.5, 28, { size: 1.4 });
      if (/^mewtwo/.test(f.species.slug) && !this.turnMode) this.psychicEntrance(f);
    } else if (f.mass >= 2) { this.fx.shakeCam(8, 0.3); this.fx.dust(f.x, f.y, 12); Audio.sfx("thud", this.pan(f.x)); }
    else { this.fx.dust(f.x, f.y, 6); }
    this.after(0.35, () => { if (f.active && !f.ko) Ability.onEnter(this, f); });
  }

  // Mewtwo's entrance: the arena drops into a violet gloom around it (the floor-lifting is its move, ワイドフォース).
  psychicEntrance(f) {
    this.psy = { f, t: 0, dur: 2.2 };
    Audio.sfx("rumble"); Audio.sfx("psychic", this.pan(f.x));
    this.fx.shakeCam(4, 1.6);
    this.fx.ring(f.x, f.y, 10, 260, 0.9, "#b48cff", 6);
  }

  after(t, fn) { this.timers.push({ t, fn }); }
  event(type, data = {}) { this.events.push(Object.assign({ type, time: this.time }, data)); }
  pan(x) { return (x - this.arena.field.cx) / this.arena.field.w; }

  // ---- queries --------------------------------------------------------------------------------------
  targetsOf(f) { const e = this.active[1 - f.side]; return e && !e.ko && e.enter <= 0 && e.active ? [e] : []; }
  nearestEnemy(f) { return this.targetsOf(f)[0] || null; }
  canSwap(side) { return this.swapCd[side] <= 0 && this.teams[side].some((f) => !f.ko && f !== this.active[side]); }
  bestSwap(side, enemy) {
    let best = null, bs = -1;
    for (const f of this.teams[side]) {
      if (f.ko || f === this.active[side]) continue;
      let s = f.hpRatio * 2;
      for (const m of f.moves) if (m.power) s += Data.effectiveness(m.type, enemy.species.types) * 0.5;
      for (const t of enemy.species.types) s -= Data.effectiveness(t, f.species.types) * 0.4;
      if (s > bs) { bs = s; best = f; }
    }
    return best;
  }
  canMega(f) { return !this.megaUsed[f.side] && !!(f.species.forms && f.species.forms.length) && f.active && !f.ko; }
  // ベース → メガ（複数あれば順に）→ キョダイ → ベース。進化の切り替え用。
  formRing(f) {
    if (!f) return null;
    const hinted = f.baseSpecies && f.baseSpecies.forms && f.baseSpecies.forms.length ? f.baseSpecies : null;
    const base = hinted || Data.bySlug(f.species.base || f.species.slug);
    if (!base || !base.forms || !base.forms.length) return null;
    const forms = base.forms.map(Data.bySlug).filter(Boolean);
    return [base, ...forms.filter((p) => p.kind === "mega"), ...forms.filter((p) => p.kind !== "mega")];
  }
  canCycleForm(f) { return !!(f && f.active && !f.ko && f.enter <= 0 && !(f.formLock > 0) && this.formRing(f)); }
  // Projectiles / dashes aimed at f within reach, or an imminent telegraph covering f.
  incoming(f) {
    for (const a of this.attacks) {
      if (a.owner === f || a.owner.side === f.side) continue;
      if (a.kind === "proj") { const d = U.dist(a.x, a.y, f.x, f.y); if (d < 220 && Math.abs(U.angDiff(Math.atan2(a.vy, a.vx), U.angle(a.x, a.y, f.x, f.y))) < 0.44) return a; }
      if (a.kind === "strike" && !a.fired && U.dist(a.x, a.y, f.x, f.y) < a.r + f.r + 16 && a.delay - a.t < 0.5) return a;
      if (a.kind === "dash" && U.dist(a.owner.x, a.owner.y, f.x, f.y) < 185) return a;
      if (a.kind === "leap" && !a.landed && U.dist(a.x, a.y, f.x, f.y) < a.r + f.r + 20 && a.dur - a.t < 0.42) return a;
      if (a.kind === "uplift" && a.phase === "tele" && U.dist(a.x, a.y, f.x, f.y) < a.r + f.r + 12 && a.spec.tele - a.t < 0.45) return a;
      if (a.kind === "grab" && a.phase === "shoot" && U.dist(a.x, a.y, f.x, f.y) < 180) return a;
      if (a.kind === "psybreak" && !a.fired && a.spec.tele - a.t < 0.4) { // standing in the aim line
        const c = Math.cos(a.aim), s = Math.sin(a.aim), dx = f.x - a.owner.x, dy = f.y - a.owner.y, along = dx * c + dy * s;
        if (along > 0 && along < (a.len || 700) && Math.abs(-dx * s + dy * c) < a.spec.w / 2 + f.r + 16) return a;
      }
    }
    const enemy = this.nearestEnemy(f);
    if (enemy && Moves.windupDanger) {
      const wd = Moves.windupDanger(enemy, f.x, f.y);
      if (wd && wd.u >= 0.52) return { kind: "windup", owner: enemy, spec: enemy.cast.spec, ux: wd.ux, uy: wd.uy, beam: wd.beam };
    }
    return null;
  }
  dangerAt(x, y, f) {
    let vx = 0, vy = 0, wMax = 0;
    const addVec = (ux, uy, w) => { vx += ux * w; vy += uy * w; if (w > wMax) wMax = w; };
    for (const p of this.arena.pools || []) {
      if (p.kind === "lava" && !f.flies && !f.species.types.includes("fire")) {
        const d = U.dist(p.x, p.y, x, y);
        if (d < p.r + f.r + 18) { const an = U.angle(p.x, p.y, x, y); addVec(Math.cos(an), Math.sin(an), 1.6); }
      }
    }
    for (const a of this.attacks) {
      if (a.owner === f || a.owner.side === f.side) continue;
      if ((a.kind === "zone" || (a.kind === "strike" && !a.fired) || (a.kind === "leap" && !a.landed) || (a.kind === "uplift" && a.phase === "tele")) && U.dist(a.x, a.y, x, y) < (a.r || 90) + f.r + 24) {
        const an = U.angle(a.x, a.y, x, y);
        addVec(Math.cos(an), Math.sin(an), a.kind === "zone" ? 1.8 : 2.8);
      } else if (a.kind === "spiral" && U.dist(a.x, a.y, x, y) < (a.R || a.spec.r1 || 150) + f.r + 22) {
        const an = U.angle(a.x, a.y, x, y);
        addVec(Math.cos(an), Math.sin(an), 2.5);
      } else if (a.kind === "beam" || a.kind === "psybreak") {
        const c = Math.cos(a.aim), s = Math.sin(a.aim), dx = x - a.owner.x, dy = y - a.owner.y;
        const along = dx * c + dy * s, perp = -dx * s + dy * c;
        const L = a.end ? a.end.d : (a.len || a.spec.len || 480), W = (a.spec.w || 30) * 0.5 + f.r + 22;
        if (along > -10 && along < L + 20 && Math.abs(perp) < W) {
          const side = perp >= 0 ? 1 : -1;
          addVec(-s * side, c * side, 3.2);
        }
      }
    }
    const enemy = this.nearestEnemy(f);
    if (enemy && Moves.windupDanger) {
      const wd = Moves.windupDanger(enemy, x, y);
      if (wd) addVec(wd.ux, wd.uy, wd.w);
    }
    const len = Math.hypot(vx, vy);
    return len > 0.05 ? { x: vx / len, y: vy / len, w: wMax } : null;
  }

  // ---- update ---------------------------------------------------------------------------------------
  update(dt, playerAct) {
    const fx = this.fx;
    this.time += dt;
    // hit-stop freezes the world but not the effects
    if (fx.hitstop > 0) { fx.update(dt); Audio.update(dt); return; }
    if (fx.slow > 0) dt *= fx.slowRate;
    fx.update(dt);
    Audio.update(dt);
    for (let i = this.timers.length - 1; i >= 0; i--) { const t = this.timers[i]; t.t -= dt; if (t.t <= 0) { this.timers.splice(i, 1); t.fn(); } }
    this.hype = Math.max(0, this.hype - dt * 0.12);
    this.arena.hype = this.hype;
    this.arena.time += dt;
    for (const p of this.arena.props) if (p.shake > 0) p.shake -= dt;
    for (const s of [0, 1]) if (this.swapCd[s] > 0) this.swapCd[s] -= dt;
    this.tickWeather(dt);
    this.clones.length = 0;
    for (const f of this.active) if (f && f.evade > 0 && !f.ko) for (let i = 0; i < 3; i++) { const a = this.time * 2.2 + i * (U.TAU / 3); this.clones.push({ owner: f, i, phase: i * 0.7, x: f.x + Math.cos(a) * 46, y: f.y + Math.sin(a) * 26 }); }
    if (this.focusT > 0) { this.focusT -= dt; if (this.focusT <= 0) this.focus = null; }
    if (this.camFocus && (this.camFocus.t -= dt) <= 0) this.camFocus = null;
    if (this.koCam && (this.koCam.t -= dt) <= 0) this.koCam = null;

    if (this.phase === "fight" && !this.turnMode) {
      this.clock += dt;
      if (this.clock >= this.duration) return this.timeUp();
    }
    // fighters
    for (let side = 0; side < 2; side++) {
      const f = this.active[side];
      if (!f) continue;
      let act = null;
      if (this.turnMode && f.enter <= 0) act = f.turnAct || { mx: 0, my: 0, aim: null, skill: [false, false, false, false], dodge: false };
      else if (this.phase === "fight" && f.enter <= 0) {
        if (side === this.playerSide) act = playerAct;
        else if (this.cpuAttack || side === 0) act = AI.act(f, this, dt, this.skill);
        else { act = AI.act(f, this, dt, this.skill); act.skill = [false, false, false, false]; }
        if (side === this.commandSide && act && playerAct) this.overlayCommand(act, playerAct);
        if (act) {
          if (act.swap && this.canSwap(side)) this.swapNext(side, act.swapTo);
          if (act.cycle) { if (side === this.commandSide) this.cycleForm(f); } // かんせんの M（手動のきりかえ）
          else if (act.mega && this.canMega(f)) this.megaEvolve(f);        // CPU / プレイヤーのメガシンカ
        }
      } else if (this.phase !== "fight" && f.enter <= 0) act = { mx: 0, my: 0, aim: null, skill: [false, false, false, false], dodge: false };
      f.update(dt, this, act);
    }
    for (const t of this.teams.flat()) { if (!t.active && t.out > 0) t.out = Math.max(0, t.out - dt); if (t !== this.active[t.side] && t.ko) t.koT += dt; }
    Moves.update(this, dt);
    if (this.psy && (this.psy.t += dt) >= this.psy.dur) this.psy = null;
    Slabs.update(this, dt);
    Ground.update(this.arena, dt, this.fx);
    Water.update(this.arena, dt);
    // Mewtwo's standing psychic aura
    for (const f of this.active) if (f && !f.ko && f.enter <= 0 && /^mewtwo/.test(f.species.slug)) {
      fx.light(f.x, f.y, 90, "#b48cff", 0.35, 0.05);
      if (Math.random() < dt * 8) { const an = U.rand(U.TAU); fx.add({ x: f.x + Math.cos(an) * f.r * 1.1, y: f.y + Math.sin(an) * f.r * 0.5, z: U.rand(0, f.height * 0.6), vz: U.rand(20, 60), life: U.rand(0.5, 0.9), size: U.rand(3, 6), color: U.pick(["#c9a8ff", "#ffffff", "#9a7aff"]), shape: "orb", add: true, drag: 0 }); }
    }
    // KO check
    for (let side = 0; side < 2; side++) {
      const f = this.active[side];
      if (f && f.hp <= 0 && !f.ko) this.knockOut(f);
    }
  }

  // ---- damage pipeline ------------------------------------------------------------------------------
  hit(target, attacker, spec, o = {}) {
    if (target.ko || target.enter > 0) return;
    const fx = this.fx;
    const pan = this.pan(target.x);
    if (target.invuln > 0) return;
    if (target.protect > 0) {
      fx.label(target.x, target.y - target.height - 8, "まもった！", "#9ad8ff", { size: 11 });
      fx.ring(target.x, target.y - target.height * 0.4, target.r + 10, target.r + 36, 0.3, "#9ad8ff", 5, { ground: false, h: target.height * 0.4, squash: 1 });
      Audio.sfx("shield", pan);
      return;
    }
    if (target.evade > 0 && Math.random() < 0.35) { fx.label(target.x, target.y - target.height - 8, "MISS", "#dddddd", { size: 11 }); fx.burst("normal", target.x + U.rand(-40, 40), target.y - 20, 6, { size: 0.6 }); return; }
    if (attacker.accDown > 0 && !Ability.ignoresAccDown(attacker) && Math.random() < 0.35 && !o.tickOf) { fx.label(target.x, target.y - target.height - 8, "MISS", "#dddddd", { size: 11 }); return; }
    if (Ability.absorb(this, target, attacker, spec)) return;
    if (this.visualOnly) { // turn-based mode: the executor applies damage; this only plays the impact
      target.flash = 1; target.squash = 1.25; fx.stop(0.05); fx.shakeCam(6, 0.2); Audio.sfx("hit", pan);
      if (o.angle !== undefined) { target.kx += Math.cos(o.angle) * 120; target.ky += Math.sin(o.angle) * 80; }
      fx.burst(spec.type, target.x, target.y - target.height * 0.45, 14, { h: target.height * 0.45, size: 1.1, dir: o.angle, spread: 1.1, owner: attacker });
      this.turnHits = (this.turnHits || 0) + 1;
      return;
    }
    if (spec.acc < 100 && Math.random() > spec.acc / 100 + 0.1 && !o.tickOf) { fx.label(target.x, target.y - target.height - 8, "MISS", "#dddddd", { size: 11 }); return; }
    const wm = this.weatherMul(spec.type);
    const screen = (spec.cat === "s" && target.screenS > 0) || (spec.cat === "p" && target.screenP > 0) ? 0.5 : 1;
    const res = Data.damage(attacker, target, spec, { mult: (o.mult || 1) * screen, weatherMul: wm, wk: this.weather.kind, critRate: attacker.critBoost ? 0.25 : Data.FX.crit(attacker) });
    const special = spec.cat === "s";
    let dmg = res.dmg;
    if (res.eff === 0) { fx.label(target.x, target.y - target.height - 8, "こうかがない", "#bbbbbb", { size: 11 }); return; }
    // substitute soaks first
    if (target.sub > 0) {
      target.sub -= dmg;
      fx.number(target.x, target.y - target.height - 10, dmg, { color: "#cccccc", size: 16 });
      if (target.sub <= 0) { target.sub = 0; fx.label(target.x, target.y - target.height, "みがわりが消えた", "#ffffff"); }
      fx.burst(spec.type, target.x, target.y - target.height * 0.4, 6, { h: target.height * 0.4, size: 0.7, owner: attacker });
      return;
    }
    const hpBefore = target.hp;
    target.hp -= dmg;
    Ability.endure(this, target, hpBefore);
    target.taken += dmg; attacker.dealt += dmg;
    const cb = this.combo[attacker.side];
    if (this.time - cb.t < 2.4) cb.n++; else cb.n = 1;
    cb.t = this.time;
    cb.move = spec;
    if (cb.n >= 2 && (!o.tickOf || o.tickOf === 3)) this.event("combo", { side: attacker.side, n: cb.n, move: spec });
    if (res.eff > 1 && (!o.tickOf || o.tickOf === 1)) this.event("say", { text: "こうかは ばつぐんだ！" });
    if (res.eff < 1 && (!o.tickOf || o.tickOf === 1)) this.event("say", { text: "こうかは いまひとつの ようだ…" });
    if (res.crit && (!o.tickOf || o.tickOf === 1)) this.event("say", { text: "きゅうしょに あたった！" });
    target.lastHitAt = this.time;
    this.stats.hits++; this.stats.maxHit = Math.max(this.stats.maxHit, dmg);
    const frac = dmg / target.maxHp;
    // numbers & labels
    const color = res.eff > 1 ? "#ffd54a" : res.eff < 1 ? "#c7c7c7" : "#ffffff";
    fx.number(target.x, target.y - target.height - 12, dmg, { color, crit: res.crit, size: 20 + Math.min(16, frac * 60) });
    if (!o.tickOf || o.tickOf === 1) {
      if (res.eff > 1) { fx.label(target.x, target.y - target.height + 8, "SUPER EFFECTIVE", "#ffd54a", { delay: 0.12 }); Audio.sfx("superEff", pan); }
      else if (res.eff < 1) { fx.label(target.x, target.y - target.height + 8, "NOT VERY EFFECTIVE", "#c7c7c7", { delay: 0.12 }); Audio.sfx("weak", pan); }
      if (res.crit) fx.label(target.x, target.y - target.height + 22, "CRITICAL!", "#ff8f5a", { delay: 0.2 });
      if (res.weight >= 1.15) fx.label(target.x, target.y - target.height + 36, "ずっしり おもい いちげき！", "#ffb070", { delay: 0.26, size: 11 });
      else if (res.weight <= 0.85) fx.label(target.x, target.y - target.height + 36, "かるくて ききめが うすい…", "#b8c4d0", { delay: 0.26, size: 11 });
    }
    // Stage-3 feel: 2–4F hitstop + pure white flash + knockback dust & zoom punch
    target.flash = 1.15; target.squash = 1.25;
    const hitAng = o.angle !== undefined ? o.angle : U.angle(attacker.x, attacker.y, target.x, target.y);
    if (!o.noStop) Motion.hurt(target, hitAng);
    const heavy = frac > 0.18 || res.crit || (spec.power || 0) >= 95;
    if (!o.noStop) {
      const stopDur = heavy ? 0.072 : (spec.power || 50) >= 55 ? 0.052 : 0.036;
      fx.stop(stopDur);
      fx.shakeCam(heavy ? 10 : 4.5, 0.2);
      if (heavy || res.eff > 1) {
        fx.zoomPunch = Math.max(fx.zoomPunch, res.crit ? 0.42 : 0.26);
        if (frac > 0.24 || res.crit) this.camFocus = { x: target.x, y: target.y, zoom: 1.22, t: 0.32 };
      }
    }
    Audio.sfx(heavy ? "hitHeavy" : "hit", pan);
    const kb = (o.knock || 100) / target.mass * Data.FX.knock(attacker, target, special) * (res.eff > 1 ? 1.3 : 1) * (this.arena.lowgrav ? 1.7 : 1);
    if (kb > 0) {
      target.kx += Math.cos(hitAng) * kb;
      target.ky += Math.sin(hitAng) * kb * 0.7;
      if (kb > 90 && !target.flies && !target.swimming) fx.dust(target.x, target.y, heavy ? 7 : 4, "rgba(175,155,120,0.75)");
    }
    // heavyweights shrug off light hits (super armour); featherweights get knocked around
    if (!o.noStop && (frac > 0.1 || o.contact) && !(target.mass >= 2.2 && frac < 0.12)) target.hitstun = Math.max(target.hitstun, U.clamp(frac * 1.2, 0.08, 0.35) * Ability.hitstunMul(target) * Data.FX.stun(attacker, target, special));
    if (o.contact) Ability.onContact(this, target, attacker);
    const th = fx.theme(spec.type, attacker);
    const big = heavy || res.eff > 1;
    if (!o.tickOf || o.tickOf === 1 || o.tickOf === 3) fx.impact(spec.type, target.x, target.y, target.height * 0.45, big, hitAng, attacker);
    fx.burst(spec.type, target.x, target.y - target.height * 0.45, big ? 26 : 12, { h: target.height * 0.45, size: big ? 1.4 : 1, dir: hitAng, spread: 1.1, owner: attacker });
    if (big && !o.tickOf) { fx.radialBlur(target.x, target.y - target.height * 0.45, res.crit ? 0.9 : 0.55); fx.flashScreen(0.22, th.c[1]); }
    if (res.crit || (res.eff > 1 && heavy)) fx.strobe(2);
    if (th.decal && Math.random() < 0.78) Arenas.decal(this.arena, th.decal, target.x + U.rand(-10, 10), target.y + U.rand(-6, 6), 18 + frac * 64);
    else if (heavy && !target.swimming) Arenas.decal(this.arena, "crater", target.x, target.y, 20 + frac * 45);
    if (o.contact && Math.random() < 0.55) Arenas.decal(this.arena, "slash", target.x, target.y, 16);
    // secondary
    const sec = spec.sec;
    if (sec && (!o.tickOf || o.tickOf === 3) && Math.random() < (sec.self ? sec.p : Math.min(0.95, sec.p * Data.FX.secChance(attacker, target, special))) && !(Ability.has(attacker, "sheer-force") && sec.st)) {
      if (sec.st) { if (target.setStatus(sec.st)) { this.stretchStatus(target, attacker); fx.label(target.x, target.y - target.height - 26, STATUS_NAMES[sec.st] + "！", STATUS_COLORS[sec.st], { delay: 0.3 }); Audio.sfx("status", pan); } }
      else if (sec.stage) {
        const who = sec.self ? attacker : target;
        if (!(Ability.has(attacker, "sheer-force") && !sec.self)) for (const k in sec.stage) if (who.stages[k] !== undefined) Ability.stage(this, who, k, sec.stage[k], sec.self ? who : attacker);
      }
    }
    if (spec.drain) { const h = attacker.heal(Math.round(dmg * spec.drain)); if (h > 0) { fx.number(attacker.x, attacker.y - attacker.height - 10, "+" + h, { color: "#8ee07a", size: 16 }); fx.aura(attacker.x, attacker.y, "#8ee07a", 6, attacker.r); } }
    this.hype = Math.min(1, this.hype + (heavy ? 0.35 : 0.12));
    Audio.setHype(heavy ? 0.9 : 0.4);
    this.event("hit", { target, attacker, dmg, eff: res.eff, crit: res.crit, spec });
    if (target.hp <= 0) this.knockOut(target, attacker);
  }

  selfDamage(f, n, color = "#ff8888") {
    if (f.ko || f.hp <= 0 || Ability.noIndirect(f)) return;
    f.hp -= n; f.taken += n;
    this.fx.number(f.x, f.y - f.height - 10, n, { color, size: 16 });
    if (f.hp <= 0) this.knockOut(f, null);
  }

  // とくこう vs とくぼう stretches or shortens a freshly applied status
  stretchStatus(target, attacker) {
    if (target.statusT > 0) target.statusT *= Data.FX.statusTime(attacker, target);
  }

  applyStatusMove(target, attacker, spec) {
    const fx = this.fx;
    if (target.protect > 0) { fx.label(target.x, target.y - target.height - 8, "まもった！", "#9ad8ff", { size: 11 }); return; }
    const st = spec.st || (spec.sec && spec.sec.st) || "para";
    if (st === "seed") { if (target.species.types.includes("grass")) { fx.label(target.x, target.y - target.height - 8, "こうかがない", "#bbbbbb", { size: 11 }); return; } target.seedBy = attacker; target.setStatus("seed"); fx.label(target.x, target.y - target.height - 8, "やどりぎを植えつけた！", "#8ee07a"); return; }
    if (target.setStatus(st)) { this.stretchStatus(target, attacker); fx.label(target.x, target.y - target.height - 8, STATUS_NAMES[st] + "状態になった！", STATUS_COLORS[st]); Audio.sfx("status", this.pan(target.x)); }
    else fx.label(target.x, target.y - target.height - 8, "こうかがない", "#bbbbbb", { size: 11 });
    fx.burst(spec.type, target.x, target.y - target.height * 0.4, 10, { h: target.height * 0.4, size: 0.8, owner: attacker });
  }

  onCastStart(f, s) {
    this.event("cast", { f, s });
    // wind-up: a spell circle for anything that isn't a quick physical swipe; bigger for strong moves
    const th = this.fx.theme(s.type, f);
    if (s.kind !== "melee" && s.kind !== "dash") this.fx.magicCircle(f.x, f.y, f.r + 26 + Math.min(40, s.power * 0.3), th.c[1], s.windup + 0.35, f);
    if (s.power >= 90) this.fx.charge(s.type, f.x, f.y - f.height * 0.5, 22, 110, f, f.height * 0.5);
  }
  // Turn-mode helper: damage computed once per move by the executor (TurnBattle).
  applyTurnDamage(target, attacker, spec) {
    if (Ability.absorb(this, target, attacker, spec)) return { dmg: 0, eff: 0, absorbed: true };
    const res = Data.damage(attacker, target, spec, { mult: 1.9, weatherMul: this.weatherMul(spec.type), wk: this.weather.kind });
    if (res.eff === 0) { this.fx.label(target.x, target.y - target.height - 8, "こうかがない", "#bbbbbb", { size: 11 }); return res; }
    const hpBefore = target.hp;
    target.hp -= res.dmg; attacker.dealt += res.dmg; target.taken += res.dmg;
    Ability.endure(this, target, hpBefore);
    this.stats.hits++; this.stats.maxHit = Math.max(this.stats.maxHit, res.dmg);
    this.fx.number(target.x, target.y - target.height - 12, res.dmg, { color: res.eff > 1 ? "#ffd54a" : res.eff < 1 ? "#c7c7c7" : "#ffffff", crit: res.crit, size: 24 });
    target.flash = 1; this.hype = Math.min(1, this.hype + 0.4); Audio.setHype(0.8);
    if (res.eff > 1) Audio.sfx("superEff", this.pan(target.x));
    const sec = spec.sec;
    if (sec && Math.random() < (sec.self ? sec.p : Math.min(0.95, sec.p * Data.FX.secChance(attacker, target, spec.cat === "s")))) {
      if (sec.st && target.setStatus(sec.st)) { this.stretchStatus(target, attacker); res.status = sec.st; }
      else if (sec.stage) { const who = sec.self ? attacker : target; for (const k in sec.stage) if (who.stages[k] !== undefined) Ability.stage(this, who, k, sec.stage[k], sec.self ? who : attacker); res.stage = sec; }
    }
    if (spec.drain) attacker.heal(Math.round(res.dmg * spec.drain));
    if (target.hp <= 0) this.knockOut(target, attacker);
    return res;
  }
  onCastFail(f, i) {
    const mine = this.playerSide >= 0 ? this.active[this.playerSide] : this.commandSide >= 0 ? this.active[this.commandSide] : null;
    if (f === mine) this.event("castfail", { i });
  }
  // 移動と技の判断はCPUのまま。押した技・回避・交代・進化だけを上書きする。
  overlayCommand(act, playerAct) {
    if (playerAct.skill && playerAct.skill.some(Boolean)) act.skill = playerAct.skill.slice();
    if (playerAct.dodge) act.dodge = true;
    if (playerAct.swap) act.swap = true;
    act.cycle = !!playerAct.mega; // CPU のメガシンカ判断（act.mega）はそのまま残す
    if (playerAct.mouseAim && playerAct.aim != null) act.aim = playerAct.aim;
  }

  // かんせん：トレーナーをタップした側へ指示を出す側を切り替える。
  setCommandSide(s) {
    if (this.commandSide < 0 || this.turnMode || (s !== 0 && s !== 1) || s === this.commandSide) return;
    this.commandSide = s;
    this.event("command", { side: s });
    Audio.sfx("select");
  }

  // ---- KO / swap / mega -----------------------------------------------------------------------------
  knockOut(f, by) {
    if (f.ko) return;
    f.ko = true; f.koT = 0; f.hp = 0; f.cast = null; f.dash = null; f.status = null; f.leap = false; f.burrow = false; f.digSink = 0; f.spinT = 0;
    if (by) { by.kos++; Ability.onKO(this, by); }
    const fx = this.fx;
    fx.stop(0.14); fx.slowmo(0.9, 0.28); fx.shakeCam(16, 0.4); fx.zoomPunch = 1; fx.flashScreen(0.4);
    fx.burst("normal", f.x, f.y - f.height * 0.4, 30, { h: f.height * 0.4, size: 1.4 });
    fx.ring(f.x, f.y, 10, 160, 0.6, "#ffffff", 8);
    fx.blast(by && by.lastMove ? by.lastMove.type : "normal", f.x, f.y, 1.2, by);
    fx.strobe(3); fx.radialBlur(f.x, f.y - f.height * 0.4, 1);
    Audio.sfx("ko"); Audio.setHype(1);
    this.focus = { x: f.x, y: f.y }; this.focusT = 1.3;
    this.koCam = { x: f.x, y: f.y, t: 1.3 };
    Arenas.decal(this.arena, "crater", f.x, f.y, 42);
    this.hype = 1;
    const side = f.side;
    const next = this.teams[side].find((t) => !t.ko);
    this.event("ko", { f, by, finish: !next });
    this.stats.koLog.push({ f: f.species.name, by: by ? by.species.name : "status", t: this.clock });
    if (!next) {
      this.after(1.75, () => this.finish(1 - side));
      return;
    }
    this.phase = "ko";
    this.after(2.2, () => { this.sendOut(side, this.teams[side].indexOf(next)); });
    this.after(3.3, () => { if (this.phase === "ko") this.phase = "fight"; });
  }

  swapNext(side, to) {
    const cur = this.active[side];
    if (!cur) return;
    const bench = this.teams[side].filter((f) => !f.ko && f !== cur);
    if (!bench.length) return;
    const next = to && bench.includes(to) ? to : bench[0];
    const token = (this.swapToken[side] = (this.swapToken[side] || 0) + 1);
    this.chosen[side] = next;
    this.swapCd[side] = 8;
    const hand = this.trainerHandPos(side);
    cur.active = false; cur.out = 0.42; cur.outMax = 0.42; cur.recallTo = hand;
    cur.cast = null; cur.dash = null; cur.leap = false; cur.burrow = false; cur.digSink = 0; cur.spinT = 0; cur.castHold = 0; cur.hop0 = 0; cur.z = 0;
    Ability.onSwitchOut(this, cur);
    this.trainerThrow[side] = this.time;
    this.fx.recallBeam(cur.x, cur.y, cur.height, hand.x, hand.y, hand.z, side === 0 ? "#ff4e78" : "#56a8ff");
    Audio.sfx("ballRecall", this.pan(cur.x));
    this.event("swap", { from: cur, to: next, side });
    this.after(0.42, () => {
      if (this.swapToken[side] !== token) return;
      const idx = this.teams[side].indexOf(next);
      if (idx >= 0) this.sendOut(side, idx);
    });
  }

  megaEvolve(f) {
    if (!this.canMega(f)) return;
    this.megaUsed[f.side] = true; f.megaDone = true;
    const forms = f.species.forms.map(Data.bySlug).filter(Boolean);
    const form = forms.find((p) => p.kind === "mega") || forms[0];
    const fx = this.fx;
    fx.slowmo(1.2, 0.35); fx.flashScreen(0.7, "#ffe9ff"); fx.shakeCam(8, 0.8);
    Audio.sfx("mega");
    f.invuln = 1.4; f.cast = null;
    this.event("mega", { f, form });
    for (let i = 0; i < 6; i++) this.after(i * 0.12, () => { fx.aura(f.x, f.y, i % 2 ? "#ff9bf0" : "#9bd8ff", 16, f.r + 10 + i * 6); fx.ring(f.x, f.y, 10, 120 + i * 20, 0.5, "#ffffff", 4); });
    this.after(0.8, () => {
      f.setForm(form, false);
      fx.burst("fairy", f.x, f.y - f.height * 0.5, 40, { h: f.height * 0.5, size: 1.6 });
      fx.light(f.x, f.y, 300, "#ffd8ff", 1, 0.8);
      fx.label(f.x, f.y - f.height - 20, form.kind === "gigantamax" ? "キョダイマックス！" : "メガシンカ！", "#ffd8ff", { size: 16 });
      if (form.kind === "gigantamax") { f.heal(f.maxHp * 0.4); }
    });
  }

  // かんせんの M。同じ演出で次の姿へ進み、最後は元の姿に戻る。
  cycleForm(f) {
    const ring = this.formRing(f);
    if (!ring || !this.canCycleForm(f)) return;
    const i = Math.max(0, ring.findIndex((p) => p.slug === f.species.slug));
    const form = ring[(i + 1) % ring.length];
    const back = form === ring[0];
    f.formLock = 1.05;
    f.megaDone = !back;
    const fx = this.fx;
    fx.slowmo(0.7, 0.3); fx.flashScreen(0.45, "#ffe9ff"); fx.shakeCam(6, 0.4);
    Audio.sfx("mega");
    f.invuln = 0.9; f.cast = null;
    this.event("mega", { f, form, back });
    this.after(0.45, () => {
      f.setForm(form, false);
      const src = (form.moves && form.moves.length ? form.moves : ring[0].moves).slice(0, 4);
      f.moves = src.map((s) => Moves.spec(s));
      while (f.moves.length < 4) f.moves.push(Moves.spec("tackle"));
      fx.burst("fairy", f.x, f.y - f.height * 0.5, 28, { h: f.height * 0.5, size: 1.3 });
      fx.light(f.x, f.y, 240, "#ffd8ff", 1, 0.55);
      fx.label(f.x, f.y - f.height - 20, back ? "もどった！" : form.kind === "gigantamax" ? "キョダイマックス！" : "メガシンカ！", "#ffd8ff", { size: 16 });
    });
  }

  // ---- weather ---------------------------------------------------------------------------------------
  setWeather(kind, dur, by) {
    this.weather = { kind, t: dur, tick: 0 };
    const names = { sand: "すなあらし", rain: "あめ", sun: "はれ", hail: "あられ" };
    this.event("weather", { kind, name: names[kind] });
    if (by) this.fx.label(by.x, by.y - by.height - 20, names[kind] + "！", "#ffffff", { size: 14 });
    this.fx.flashScreen(0.25, kind === "sun" ? "#fff0b0" : kind === "rain" ? "#b0c8ff" : "#e8d8b0");
  }
  tickWeather(dt) {
    const w = this.weather;
    if (!w.kind) return;
    w.t -= dt; w.tick += dt;
    if (w.t <= 0) { this.weather = { kind: null, t: 0, tick: 0 }; this.event("weather", { kind: null }); return; }
    if (w.tick >= 2.5 && (w.kind === "sand" || w.kind === "hail") && this.phase === "fight") {
      w.tick = 0;
      for (const f of this.active) {
        if (!f || f.ko || f.enter > 0) continue;
        const T = f.species.types;
        const immune = w.kind === "sand" ? T.includes("rock") || T.includes("ground") || T.includes("steel") : T.includes("ice");
        if (!immune) this.selfDamage(f, Math.max(2, Math.round(f.maxHp * 0.03)), "#e0d0b0");
      }
    }
  }
  weatherMul(type) {
    const k = this.weather.kind;
    if (k === "rain") return type === "water" ? 1.5 : type === "fire" ? 0.5 : 1;
    if (k === "sun") return type === "fire" ? 1.5 : type === "water" ? 0.5 : 1;
    return 1;
  }

  // ---- end -------------------------------------------------------------------------------------------
  timeUp() {
    const hp = [0, 1].map((s) => this.teams[s].reduce((a, f) => a + f.hpRatio, 0));
    this.finish(hp[0] === hp[1] ? -1 : hp[0] > hp[1] ? 0 : 1, true);
  }
  finish(winner, timeout) {
    if (this.phase === "end") return;
    this.phase = "end"; this.winner = winner;
    this.event("end", { winner, timeout });
    Audio.stopMusic();
    if (this.cfg.demo) { Audio.stopCrowd(); return; }
    if (winner === this.playerSide || this.playerSide < 0) Audio.sfx("win"); else Audio.sfx("lose");
    Audio.setHype(1);
  }
}
Battle.fieldScaleFor = (teams) => {
  const ref = Assets.onFieldPx(Assets.bodyArtPx({ h: Assets.TRAINER_M })); // 1.7m のモンスター（リザードン）= 標準のコート
  let big = 0, gmax = false;
  for (const e of (teams || []).flat()) {
    const p = e && (e.species || e);
    if (!p) continue;
    if (p.kind === "gigantamax" || p.cat === "gigantamax") { gmax = true; continue; } // キョダイマックスのためには広げない
    big = Math.max(big, Assets.onFieldPx(Assets.bodyArtPx(p)));
  }
  if (!big) return 1;
  const k = big / ref, sc = k >= 1 ? Math.min(1.75, Math.pow(k, 0.8)) : Math.max(0.78, Math.pow(k, 0.5));
  return gmax ? Math.max(1, sc) : sc; // …ただしキョダイマックスがいるのに標準より狭くはしない
};
