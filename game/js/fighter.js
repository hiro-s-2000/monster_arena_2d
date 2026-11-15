// A Pokémon in the arena: stats, movement, casting state machine, status conditions and the
// bookkeeping the renderer needs (facing, squash, flash, entrance / KO animation).
"use strict";

const STATUS_NAMES = { burn: "やけど", para: "まひ", psn: "どく", slp: "ねむり", frz: "こおり", conf: "こんらん", seed: "やどりぎ" };
const STATUS_COLORS = { burn: "#ff7a3a", para: "#ffd83a", psn: "#c35bd9", slp: "#9aa7c7", frz: "#8fdfff", conf: "#ff9ad6", seed: "#8ee07a" };

class Fighter {
  constructor(species, side, slot, opts = {}) {
    this.side = side; this.slot = slot;
    this.shiny = !!opts.shiny && species.shiny;
    this.setForm(species, true);
    this.energy = 100; this.maxEnergy = 100;
    this.stages = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
    this.status = null; this.statusT = 0; this.statusTick = 0;
    this.moves = (opts.moves || species.moves).slice(0, 4).map((s) => Moves.spec(s));
    while (this.moves.length < 4) this.moves.push(Moves.spec("tackle"));
    this.moves = this.moves.map((m) => Ability.moveType(this, m));
    this.cds = [0, 0, 0, 0];
    this.x = 0; this.y = 0; this.z = 0; this.vx = 0; this.vy = 0; this.kx = 0; this.ky = 0;
    this.facing = side ? Math.PI : 0; this.aim = this.facing;
    this.cast = null; this.castHold = 0; this.hitstun = 0; this.dash = null; this.dodgeT = 0; this.dodgeCd = 0; this.invuln = 0;
    this.protect = 0; this.sub = 0; this.evade = 0; this.screenS = 0; this.screenP = 0; this.regen = 0; this.critBoost = 0; this.slowT = 0; this.confT = 0; this.seedBy = null;
    this.ko = false; this.koT = 0; this.active = false; this.enter = 0; this.enterFrom = null; this.enterOpened = false; this.enterTrailT = 0; this.out = 0; this.outMax = 0.42; this.recallTo = null;
    this.anim = Math.random() * 3; this.flash = 0; this.squash = 1; this.squashV = 0; this.hop0 = 0; this.hopV = 0; this.tilt = 0;
    this.lastMove = null; this.lastHitAt = -9; this.dealt = 0; this.taken = 0; this.kos = 0; this.megaDone = false; this.formLock = 0;
    this.ghosts = []; this.ghostT = 0; this.recoverT = 0; this.skidT = 0; this.mashT = 0;
    this.ai = null; this.controlled = false; this.lockTarget = null; this.moveT = 0;
  }

  // Species / form swap (also used for mega evolution). Keeps the HP ratio.
  setForm(species, initial) {
    const ratio = initial ? 1 : this.hp / this.maxHp;
    if (initial) this.baseSpecies = species;
    this.species = species;
    this.sprites = Assets.sprites(species, this.shiny);
    this.d8 = Dir8.sheet(species, this.shiny); // gen-1: real 8-direction stills (null → derived from the GIFs)
    this.measured = false;
    this.stats = Data.stats(species);
    this.maxHp = this.stats.hp;
    this.hp = initial ? this.maxHp : Math.max(1, Math.round(this.maxHp * ratio));
    this.measure();
    // Personality from the dex entry: base Speed sets the pace, weight slows the run a little,
    // body shape picks the gait (see update / Renderer.fighter).
    const pr = Fighter.profile(species);
    this.speed = pr.speed; this.gait = pr.gait; this.tempo = pr.tempo;
    this.hoverH = pr.gait === "fly" ? Fighter.hoverPx(pr.bodyH) : 0;
    this.mightAtk = pr.mightAtk; this.mightSpa = pr.mightSpa; this.reach = pr.reach; this.push = pr.push;
  }
  measure() {
    const b = (this.d8 && this.d8[0].bounds) || this.sprites.front.bounds, K = Arenas.K; // bounds are in art px
    this.measured8 = !!(this.d8 && this.d8[0].bounds);
    const target = Assets.onFieldPx(Assets.bodyArtPx(this.species));
    // Draw and hitbox always land on the height curve, even if pixelize stopped short of it.
    this.fit = b ? target / Math.max(1, b.h) : 1;
    const w = b ? b.w * this.fit * K : target * K * 0.8, h = (b ? b.h * this.fit : target) * K;
    this.measured = !!b;
    this.r = U.clamp(w * 0.26, 8, 110);
    this.height = Math.max(20, h);
    // 1 kg ≈ 0.55, 6.9 kg (Bulbasaur) ≈ 0.85, 90 kg ≈ 1.8, 460 kg (Snorlax) ≈ 2.6, 999 kg ≈ 2.9
    this.mass = Fighter.massOf(this.species);
  }
  get loadRecs() { const s = this.sprites; return this.d8 ? [s.front, ...this.d8] : [s.front, s.frontR, s.back, s.backR]; }
  get flies() { return Fighter.canFly(this.species); }
  get alive() { return !this.ko; }
  get hpRatio() { return this.hp / this.maxHp; }
  get canAct() { return this.active && !this.ko && this.enter <= 0 && this.hitstun <= 0 && this.status !== "slp" && this.status !== "frz"; }
  get busy() { return !!this.cast || !!this.dash || this.hitstun > 0; }

  // Sprite record for the displayed 8-way facing (see Dir8.view for the transform that goes with it).
  spriteRec() { return Dir8.view(this).rec; }

  placeAt(x, y, facing) { this.x = x; this.y = y; this.facing = this.aim = facing; this.vx = this.vy = this.kx = this.ky = 0; }

  // ---- combat helpers -----------------------------------------------------------------------------
  heal(n) { const before = this.hp; this.hp = Math.min(this.maxHp, this.hp + n); return this.hp - before; }
  cure() { this.status = null; this.statusT = 0; this.confT = 0; }
  setStatus(st, dur, force) {
    if (this.ko) return false;
    if (st === "conf" && Ability.statusImmune(this, st)) return false;
    if (st === "conf") { this.confT = Math.max(this.confT, dur || 3); return true; }
    if (st === "seed") { this.seedT = 10; return true; }
    if (this.status && !force) return false;
    if (Ability.statusImmune(this, st)) return false;
    // type immunities
    const T = this.species.types;
    if ((st === "burn" && T.includes("fire")) || (st === "para" && T.includes("electric")) || (st === "frz" && T.includes("ice")) || (st === "psn" && (T.includes("poison") || T.includes("steel")))) return false;
    this.status = st; this.statusT = dur || { burn: 12, para: 9, psn: 12, slp: 2.6, frz: 2.2 }[st] || 8; this.statusTick = 0;
    if (st === "slp" || st === "frz") { this.cast = null; this.dash = null; }
    return true;
  }
  lunge(angle, px) { this.kx += Math.cos(angle) * px * 2.2; this.ky += Math.sin(angle) * px * 2.2; }
  hop(h) { this.hopV = Math.max(this.hopV, h * 4.5); }
  recoil(frac) { if (!Ability.noRecoil(this)) this.recoilPending = (this.recoilPending || 0) + frac * Data.FX.recoil(this); } // ぼうぎょ softens recoil
  startEnter(fromX, fromY, fromZ = 30) {
    this.enter = 1.05; this.enterFrom = { x: fromX, y: fromY, z: fromZ };
    this.enterOpened = false; this.enterTrailT = 0;
    this.out = 0; this.recallTo = null; this.active = true; this.z = 0;
  }

  // Begin a cast if legal; returns true when started.
  tryCast(i, aim, B) {
    const s = this.moves[i];
    if (!s || this.cast || this.dash || this.hitstun > 0 || this.cds[i] > 0 || this.energy < s.cost || !this.canAct) return false;
    if (this.status === "para" && Math.random() < 0.25) { B.fx.label(this.x, this.y - this.height, "しびれて動けない！", "#ffd83a"); this.cds[i] = 0.6; return false; }
    this.cast = { slot: i, spec: s, t: 0, fired: false, aim };
    this.aim = aim; this.facing = aim;
    this.energy -= s.cost;
    this.cds[i] = s.cd * (this.side === B.playerSide ? 1 : B.cpuCdMul) / (this.tempo || 1);
    this.lastMove = s;
    this.moveT = 0;
    B.onCastStart(this, s);
    return true;
  }

  // ---- per-frame ----------------------------------------------------------------------------------
  update(dt, B, act) {
    if ((!this.measured && this.sprites.front.bounds) || (this.d8 && !this.measured8 && this.d8[0].bounds)) this.measure(); // sprite / 8-view sheet arrived
    Dir8.update(this, dt); Motion.update(this, dt);
    this.anim += dt * (this.tempo || 1) * (this.ko ? 0 : this.status === "slp" || this.status === "frz" ? 0.15 : 1);
    this.flash = Math.max(0, this.flash - dt * 6);
    for (const k of ["protect", "evade", "screenS", "screenP", "regen", "slowT", "confT", "invuln", "dodgeCd", "castHold", "seedT", "dizzy", "megaGlow", "accDown", "formLock", "recoverT", "mashT"]) if (this[k] > 0) this[k] = Math.max(0, this[k] - dt);
    if (act && act.skill && act.skill.some(Boolean)) this.mashT = 0.11;
    // Stage-2 afterimages decay + high-speed sampling
    for (let i = this.ghosts.length - 1; i >= 0; i--) {
      this.ghosts[i].t += dt;
      if (this.ghosts[i].t >= this.ghosts[i].life) this.ghosts.splice(i, 1);
    }
    const fastLunge = this.cast && this.cast.fired && this.cast.spec && (this.cast.spec.kind === "melee" || this.cast.spec.kind === "cone") && this.cast.t < this.cast.spec.windup + 0.14;
    if (this.active && !this.ko && (this.dodgeT > 0 || this.leap || fastLunge)) {
      this.ghostT -= dt;
      if (this.ghostT <= 0) {
        this.ghostT = 0.028;
        const fr = typeof Renderer !== "undefined" && Renderer.frameOf ? Renderer.frameOf(this) : null;
        if (fr && fr.img) {
          const col = this.dodgeT > 0 ? "#7dd3fc" : this.lastMove ? B.fx.theme(this.lastMove.type).c[1] : "#bae6fd";
          this.ghosts.push({ x: this.x, y: this.y, z: this.z + (this.pose ? this.pose.up : 0), img: fr.img, ax: fr.ax, ay: fr.ay, w: fr.w, h: fr.h, bh: fr.bh, color: col, t: 0, life: 0.16 });
          if (this.ghosts.length > 5) this.ghosts.shift();
        }
      }
    }
    // Walk cycle. Fliers hover. Everyone else keeps their feet on the surface and only hops for a short step.
    const spdNow = Math.hypot(this.vx, this.vy), g = this.gait;
    const stepping = spdNow > 25 && !this.cast && g !== "fly";
    const freq = { walk: 0.045, gallop: 0.06, bounce: 0.03, slither: 0.035, stomp: 0.03, fly: 0.04 }[g] || 0.045;
    this.walkPhase = (this.walkPhase || 0) + (stepping ? spdNow * dt * freq : dt * 0.6);
    const ph = Math.abs(Math.sin(this.walkPhase));
    const amp = { walk: 1, gallop: 1.6, bounce: 3, slither: 0, stomp: 1, fly: 2 }[g];
    let bobT = 0;
    // [ユーザー指定 2026-11-15] 浮くのは鳥型と蛇型ドラゴンだけ。浮遊感を少し強める：体の大きさに応じた高さ（hoverH）を保ち、
    // ゆっくり大きく上下する（移動中は少し高く・揺れは小さく）。倒れたら地面に落ちる。
    if (g === "fly") bobT = this.ko ? 0 : this.hoverH + (spdNow > 25 ? 1 + Math.sin(this.anim * 3.2) * 0.8 : Math.sin(this.anim * 2.1) * 1.7);
    else if (!this.swimming && stepping && g !== "slither") bobT = ph > 0.78 ? ((ph - 0.78) / 0.22) * amp : 0;
    else if (!this.swimming && g === "bounce") bobT = Math.sin(this.anim * 2.2) > 0.75 ? 1 : 0;
    // in water: the body rides up and down against a waterline that stays put (stroke rhythm while moving, slow float at rest)
    else if (this.swimming) bobT = stepping ? 0.3 + 1.3 * (0.5 + 0.5 * Math.sin(this.walkPhase * 2)) : 0.6 + 0.6 * Math.sin(this.anim * 2.1 + this.side * 1.7);
    this.bob = U.damp(this.bob || 0, bobT, this.swimming ? 9 : g === "fly" ? 7 : 22, dt);
    // slither: a sideways S-sway instead of a bob
    this.sway = U.damp(this.sway || 0, g === "slither" ? Math.sin(this.walkPhase * (stepping ? 1 : 0.3) + this.anim * (stepping ? 0 : 1.5)) * (stepping ? 2.2 : 0.8) : 0, 12, dt);
    if (stepping && ph < 0.15 && !this.stepped) {
      this.stepped = true;
      if (g !== "slither") this.squash = Math.max(this.squash, g === "bounce" ? 1.16 : g === "stomp" ? 1.08 : 1.04);
      if (this.active) this.plantStep(B);
      if (g === "stomp" && this.active && this.mass >= 2.1) { B.fx.shakeCam(1.4 + this.mass * 0.35, 0.08); Audio.sfx("step", B.pan(this.x)); }
    } else if (ph > 0.45) this.stepped = false;
    // terrain
    const ter = this.active && !this.ko ? Arenas.terrainAt(B.arena, this.x, this.y) : null;
    const wasSwim = this.swimming;
    this.swimming = ter === "water" && !this.flies && this.z < 4 && !this.burrow;
    this.onIce = ter === "ice" && !this.flies && this.z < 4;
    this.fallV = Math.max(0, ((this.lastZ || 0) - this.z) / Math.max(dt, 1e-3)); this.lastZ = this.z;
    // standing in a crater / gouge: the feet go below the rim
    // The floor height is averaged over the footprint (not one point) and followed smoothly, so the body walks
    // down the slope and back up over the lip instead of snapping. Depth is capped by body size: small monsters
    // must not lose their legs in a dent. Positive = lower on screen (art px); slightly negative on a raised lip.
    let gh = 0;
    if (this.active && !this.flies && !this.swimming && this.z < 8) {
      const A = B.arena, rx = this.r * 0.45, ry = this.r * 0.2;
      gh = (Ground.heightAt(A, this.x, this.y) * 2 + Ground.heightAt(A, this.x - rx, this.y) + Ground.heightAt(A, this.x + rx, this.y) + Ground.heightAt(A, this.x, this.y - ry) + Ground.heightAt(A, this.x, this.y + ry)) / 6;
    }
    const sinkCap = Math.min(5, Math.max(1.5, this.height / Arenas.K * 0.14));
    this.sinkG = U.damp(this.sinkG || 0, gh < 0 ? Math.min(sinkCap, -gh * 0.6) : Math.max(-1.5, -gh * 0.4), 7, dt);
    // 0‥1 how far the water has closed around the body: stepping in / climbing out is a short slide, not a pop
    this.wade = U.damp(this.wade || 0, this.swimming ? 1 : 0, 8, dt);
    if (this.active && !this.ko) Water.fighter(B, this, dt, wasSwim);
    if (ter === "lava" && !this.flies && this.z < 4 && this.enter <= 0) { this.lavaT = (this.lavaT || 0) + dt; if (this.lavaT > 0.7) { this.lavaT = 0; if (!this.species.types.includes("fire")) { B.selfDamage(this, Math.max(2, Math.round(this.maxHp * 0.05)), "#ff8a3a"); if (Math.random() < 0.3) this.setStatus("burn"); } B.fx.burst("fire", this.x, this.y, 6, { size: 0.7 }); } }
    // squash & hop springs
    this.squashV += (1 - this.squash) * 260 * dt; this.squashV *= Math.exp(-14 * dt); this.squash += this.squashV * dt;
    this.hopV -= 900 * dt; this.hop0 += this.hopV * dt; if (this.hop0 < 0) { this.hop0 = 0; this.hopV = 0; }
    this.z = this.hop0;
    if (this.ko) { this.koT += dt; this.vx = this.vy = 0; return; }
    if (this.enter > 0) {
      if (this.enter > 0.52 && B && B.onBallTrail) B.onBallTrail(this, dt);
      this.enter -= dt;
      if (this.enter <= 0.52 && !this.enterOpened) {
        this.enterOpened = true;
        if (B && B.onBallOpen) B.onBallOpen(this);
      }
      if (this.enter <= 0) { this.enter = 0; B.onEntered(this); }
      return;
    }
    if (!this.active) return;
    for (let i = 0; i < 4; i++) if (this.cds[i] > 0) this.cds[i] -= dt;
    this.energy = Math.min(this.maxEnergy, this.energy + dt * (this.cast ? 6 : 13) * Ability.energyMul(this, B));
    Ability.update(B, this, dt);
    if (this.regen > 0) this.heal(this.maxHp * 0.06 * dt);
    this.moveT += dt;
    this.tickStatus(dt, B);
    if (this.recoilPending) { B.selfDamage(this, Math.round(this.maxHp * 0.06 * this.recoilPending)); this.recoilPending = 0; }

    // knockback decay + Stage-3 knockback dust & turf skid pebbles
    this.x += this.kx * dt; this.y += this.ky * dt;
    const kSpd = Math.hypot(this.kx, this.ky);
    if (kSpd > 90 && this.active && !this.flies && !this.swimming && this.z < 8) {
      this.skidT = (this.skidT || 0) - dt;
      if (this.skidT <= 0) {
        this.skidT = 0.04;
        B.fx.dust(this.x, this.y, kSpd > 200 ? 4 : 2, B.arena.def.floor === "sand" ? "rgba(235,205,145,0.75)" : "rgba(155,136,105,0.72)");
        Arenas.decal(B.arena, "skid", this.x, this.y, kSpd > 210 ? 12 : 8);
      }
    }
    const kd = Math.exp(-7 * dt); this.kx *= kd; this.ky *= kd;

    if (this.spinT > 0) { this.spinT -= dt; this.facing += dt * 16; }
    if (this.leap) { this.vx = this.vy = 0; }
    else if (this.burrow) { this.vx = this.vy = 0; } // tunnelling: the dig attack moves the body
    else if (this.hitstun > 0) { this.hitstun -= dt; if (this.hitstun > 0.22) this.dizzy = Math.max(this.dizzy || 0, 0.6); this.cast = null; this.dash = null; this.vx = this.vy = 0; }
    else if (this.dash) {
      const d = this.dash;
      d.t -= dt;
      const nx = this.x + d.ax * d.speed * dt, ny = this.y + d.ay * d.speed * dt;
      const p = Arenas.resolve(B.arena, nx, ny, this.r);
      const hitWall = U.dist(p.x, p.y, nx, ny) > 2;
      if (hitWall) d.t = 0; // hit a wall
      this.x = p.x; this.y = p.y;
      this.facing = Math.atan2(d.ay, d.ax);
      if (d.t <= 0) { const whiffed = !d.hitAny; this.dash = null; this.cast = null; this.recoverT = whiffed || hitWall ? 0.36 : 0.16; }
    } else if (this.cast) {
      const c = this.cast, s = c.spec;
      if (!c.mStart) { c.mStart = true; Motion.startCast(this, s, s.windup, s.kind === "beam" || s.kind === "spiral" ? s.dur : s.hold || 0); }
      c.t += dt * (c.fired ? 1 : Ability.castRate(this, s) * Data.FX.quick(this)); // すばやさ: quicker wind-up
      if (act && act.aim !== null && act.aim !== undefined && !c.fired) { c.aim = act.aim; }
      // auto-aim tracks the target during early wind-up; locks in final 35% so dodges can evade telegraphs!
      if (!c.fired && this.lockTarget && !act?.mouseAim && c.t < s.windup * 0.65) { const tg = this.lockTarget; if (!tg.ko) c.aim = U.angle(this.x, this.y, tg.x + tg.vx * 0.15, tg.y + tg.vy * 0.15); }
      this.aim = c.aim; this.facing = c.aim;
      if (!c.fired && c.t >= s.windup) {
        c.fired = true;
        if (s.power >= 95 && !B.turnMode) {
          const th = B.fx.theme(s.type, this);
          B.fx.flashScreen(0.2, th.c[1]);
          B.fx.ring(this.x, this.y, 10, 120, 0.3, th.c[0], 6);
          B.fx.shakeCam(5, 0.16);
          B.fx.zoomPunch = Math.max(B.fx.zoomPunch, 0.32);
          B.camFocus = { x: this.x, y: this.y, zoom: 1.22, t: 0.38 };
        }
        Moves.cast(B, this, s, c.aim, this.lockTarget);
        if (s.kind === "beam" || s.kind === "spiral") c.hold = s.dur;
        else if (s.hold) c.hold = s.hold;
      }
      if (c.fired) {
        const hold = c.hold || 0;
        if (c.t >= s.windup + hold + s.recover) {
          this.cast = null;
          this.recoverT = Math.max(this.recoverT || 0, U.clamp(0.15 + (s.power || 40) * 0.0022, 0.16, 0.46));
        }
      }
      if (!c.fired && c.t < s.windup && Math.random() < 0.42) B.fx.charge(s.type, this.x, this.y - this.height * 0.52, s.power >= 85 ? 3 : 1, 46, this, this.height * 0.52);
      // slight drift allowed while casting unlocked moves
      if (!s.lock && act) this.move(dt, act.mx * 0.35, act.my * 0.35, B);
      else this.vx = this.vy = 0;
    } else if (this.dodgeT > 0) {
      this.dodgeT -= dt;
      const p = Arenas.resolve(B.arena, this.x + this.vx * dt, this.y + this.vy * dt, this.r);
      this.x = p.x; this.y = p.y;
      if (Math.random() < 0.6) B.fx.dust(this.x, this.y, 1);
    } else if (act) {
      // movement
      let mx = act.mx, my = act.my;
      if (this.confT > 0) { const a = Math.atan2(my, mx) + Math.sin(this.moveT * 7) * 1.6; const l = Math.hypot(mx, my); mx = Math.cos(a) * l; my = Math.sin(a) * l; }
      if (this.canAct) this.move(dt, mx, my, B); else this.vx = this.vy = 0;
      // aim stays the shot direction (mouse, stick, or the foe). Facing while walking
      // looks at the foe, for both sides. The sprite row still steps one way at a time.
      if (act.aim !== null && act.aim !== undefined) this.aim = act.aim;
      else if (this.lockTarget && !this.lockTarget.ko) this.aim = U.angle(this.x, this.y, this.lockTarget.x, this.lockTarget.y);
      const foe = this.lockTarget;
      const stepping = Math.hypot(mx, my) > 0.1;
      if (stepping && foe && !foe.ko && U.dist(this.x, this.y, foe.x, foe.y) > 1) this.facing = U.angle(this.x, this.y, foe.x, foe.y);
      else if (stepping) this.facing = Math.atan2(my, mx);
      else this.facing = this.aim;
      if (this.canAct) {
        if (act.dodge && this.dodgeCd <= 0 && this.energy >= 15) {
          const l = Math.hypot(mx, my);
          const a = l > 0.1 ? Math.atan2(my, mx) : this.facing + Math.PI;
          this.dodgeT = 0.26; this.dodgeCd = 1.1 * Ability.dodgeCdMul(this) * U.clamp(1.25 - this.species.st.spe / 250, 0.75, 1.2); this.invuln = 0.3; this.energy -= 15;
          if (BLINKERS.test(this.species.slug)) { // psychics teleport a short hop instead of rolling
            Moves.warp(B.fx, this, true); Audio.sfx("psychic", B.pan(this.x));
            const p = Arenas.resolve(B.arena, this.x + Math.cos(a) * 170, this.y + Math.sin(a) * 120, this.r);
            this.x = p.x; this.y = p.y; this.dodgeT = 0.05; this.vx = this.vy = 0;
            Moves.warp(B.fx, this, false);
          } else {
            this.vx = Math.cos(a) * 640; this.vy = Math.sin(a) * 640;
            this.squash = 0.7; if (this.swimming) Water.splash(B, this.x, this.y, 1.1); else B.fx.dust(this.x, this.y, 5, this.onIce ? "rgba(240,250,255,0.7)" : undefined); Audio.sfx("dodge", B.pan(this.x));
            B.fx.ring(this.x, this.y, 8, 40, 0.3, "rgba(255,255,255,0.6)", 3);
          }
        }
        for (let i = 0; i < 4; i++) if (act.skill[i]) { if (this.tryCast(i, this.aim, B)) break; else if (this.cds[i] > 0 || this.energy < this.moves[i].cost) B.onCastFail(this, i); }
      }
    }
    const p = Arenas.resolve(B.arena, this.x, this.y, this.r);
    this.x = p.x; this.y = p.y;
    // lean into movement
    const spd = Math.hypot(this.vx, this.vy);
    this.tilt = U.damp(this.tilt, spd > 20 ? U.clamp(this.vx / this.speed, -1, 1) * 0.08 : 0, 10, dt);
  }

  move(dt, mx, my, B) {
    const l = Math.hypot(mx, my);
    // [ユーザー指定 2026-11-15] 移動を少し遅く（体が大きく映るようになったので、同じ速さだと速すぎる）。全員に掛かる倍率。
    let sp = this.speed * Fighter.moveRate(this.species) * Data.stageMul(this.stages.spe) ** 0.5 * Ability.speedMul(this, B);
    if (this.status === "para") sp *= 0.6;
    if (this.slowT > 0) sp *= 0.55;
    if (this.swimming) sp *= this.species.types.includes("water") ? 1.15 : 0.6;
    const grip = this.onIce && !this.species.types.includes("ice") ? 2.2 : 12;
    if (l > 0.05) {
      this.vx = U.damp(this.vx, mx * sp, grip, dt); this.vy = U.damp(this.vy, my * sp, grip, dt);
      if (!this.swimming && !this.flies && Math.random() < dt * 6) B.fx.dust(this.x - mx * 8, this.y, 1, B.arena.def.floor === "sand" ? "rgba(230,200,140,0.6)" : this.onIce ? "rgba(240,250,255,0.7)" : "rgba(120,110,90,0.45)");
      if (this.onIce && !this.flies && Math.random() < dt * 5) B.fx.add({ x: this.x + U.rand(-6, 6), y: this.y, z: 2, vx: -this.vx * 0.2 + U.rand(-30, 30), vy: U.rand(-10, 10), vz: U.rand(30, 70), grav: 500, life: 0.35, size: 2, color: "#ffffff", shape: "pixel", add: true }); // skate glints
    } else { this.vx = U.damp(this.vx, 0, grip === 12 ? 16 : 1.8, dt); this.vy = U.damp(this.vy, 0, grip === 12 ? 16 : 1.8, dt); }
    const p = Arenas.resolve(B.arena, this.x + this.vx * dt, this.y + this.vy * dt, this.r);
    this.x = p.x; this.y = p.y;
  }

  tickStatus(dt, B) {
    if (B.turnMode) return; // the turn executor resolves status per turn
    if (this.status) {
      this.statusT -= dt; this.statusTick += dt;
      if (this.statusTick >= 2 && (this.status === "burn" || this.status === "psn")) {
        this.statusTick = 0;
        B.selfDamage(this, Math.max(2, Math.round(this.maxHp * (this.status === "psn" ? 0.07 : 0.05))), STATUS_COLORS[this.status]);
        B.fx.burst(this.status === "burn" ? "fire" : "poison", this.x, this.y - this.height * 0.4, 6, { h: this.height * 0.4, size: 0.8, light: this.status === "burn" });
      }
      if (this.status === "frz" && Math.random() < dt * 1.5) B.fx.add({ x: this.x + U.rand(-this.r, this.r), y: this.y - U.rand(0, this.height), z: 0, vz: 20, life: 0.6, size: 4, color: "#dff6ff", shape: "shard", add: true });
      if (this.status === "slp" && Math.random() < dt * 2) B.fx.label(this.x + 14, this.y - this.height - 6, "Z", "#dfe6ff", { size: 10, life: 1 });
      if (this.statusT <= 0) { if (this.status === "slp" || this.status === "frz") B.fx.label(this.x, this.y - this.height, this.status === "slp" ? "目を覚ました！" : "こおりが とけた！", "#ffffff"); this.status = null; }
    }
    if (this.seedT > 0 && this.seedBy && !this.seedBy.ko) { this.seedT -= dt; this.seedTick = (this.seedTick || 0) + dt; if (this.seedTick > 1.5) { this.seedTick = 0; const n = Math.max(2, Math.round(this.maxHp * 0.04)); B.selfDamage(this, n, STATUS_COLORS.seed); this.seedBy.heal(n); B.fx.aura(this.seedBy.x, this.seedBy.y, "#8ee07a", 4, 10); } }
  }

  // Footfall. Grounded bodies kick the surface they are standing on; fliers never call this.
  plantStep(B) {
    const ter = Arenas.terrainAt(B.arena, this.x, this.y);
    if (this.swimming || ter === "water") {
      Water.ripple(B.arena, this.x + U.rand(-this.r, this.r) * 0.2, this.y, Math.max(8, this.r * 0.4), 26, 0.7, { foam: true });
      B.fx.add({ x: this.x + U.rand(-6, 6), y: this.y, z: 1, vz: U.rand(24, 60), grav: 480, life: 0.26, size: 2, color: "#f4fbff", shape: "drop", drag: 0.3 });
      return;
    }
    const floor = B.arena.def.floor;
    const col = floor === "sand" ? "rgba(230,200,140,0.75)" : floor === "stone" || floor === "pokeball" ? "rgba(200,200,210,0.6)" : floor === "ice" ? "rgba(240,250,255,0.8)" : floor === "dirt" ? "rgba(184,136,88,0.7)" : "rgba(120,150,80,0.6)";
    const back = Math.atan2(-(this.vy || 0), -(this.vx || 0.01));
    B.fx.dust(this.x, this.y, this.mass >= 2 ? 3 : 2, col);
    for (let i = 0; i < 2; i++) B.fx.add({ x: this.x, y: this.y, z: 0, vx: Math.cos(back) * U.rand(18, 46), vy: Math.sin(back) * U.rand(8, 24), vz: U.rand(12, 32), grav: 400, life: 0.2, size: 2, color: col, shape: "pixel", drag: 1 });
  }

  // ---- species profile (shared with the select-screen dex panel) -----------------------------------
  // [ユーザー指定 2026-11-15] 「モンスターが不自然に浮いて見える。全モンスターが地上を歩いている様に。鳥型や蛇の様なドラゴンは浮いたままでよい」
  // 空中にいられる種（ひこうタイプ・ふゆう・FLOATERS）のうち、実際に浮かせるのは体型が鳥型（wings）・蛇型（squiggle）・羽根のある虫型（bug-wings）だけ。
  // それ以外（リザードン・ミュウツー・ゲンガー等）は地面に足をつけ、接地影の上を歩く（水では泳ぎ、溶岩・氷の影響も受ける）。
  static airborne(p) { return p.types.includes("flying") || !!(p.ab && p.ab.s === "levitate") || FLOATERS.includes(p.slug.replace(/-mega.*|-gmax/, "")); }
  // 羽根のある虫型（bug-wings）は、ひこうタイプでなくても羽で飛ぶので浮かせる（スピアー・モルフォン・ウルガモス等）。歩く虫（BUG_WALKERS）だけ地上。
  // [ユーザー指定 2026-11-15 追加] 鳥でも地上を走る・歩く種（GROUND_BIRDS：ガラルサンダー、ドードー系など）は地上。
  // 脚のない浮遊体（HOVER_SLUGS：ゴース・コイル・ドガース系）は体型に関係なく浮かせる。
  static canFly(p) {
    if (HOVER_SLUGS.test(p.slug)) return true;
    if (GROUND_BIRDS.test(p.slug) || WINGS_FOLDED.has(p.slug)) return false;
    return p.shape === "bug-wings" ? !BUG_WALKERS.test(p.slug) : HOVER_SHAPES.includes(p.shape) && Fighter.airborne(p);
  }
  // 浮く高さ（アートpx）：体が大きいほど高い。以前は一律 1.5〜2px。
  static hoverPx(bodyH) { return U.clamp(Math.round(bodyH * 0.11), 4, 8); }
  // 1 kg ≈ 0.55, 6.9 kg (Bulbasaur) ≈ 0.85, 90 kg ≈ 1.8, 460 kg (Snorlax) ≈ 2.6, 999 kg ≈ 2.9
  static massOf(p) { return U.clamp(0.5 + 0.34 * Math.log2(1 + (p.w || 10) / 8), 0.5, 3); }
  static profile(p) {
    const mass = Fighter.massOf(p), sh = p.shape;
    const lv = Data.stats(p);
    const bodyH = Assets.bodyArtPx(p);
    // Lv.50 speed runs about 25–250. tempo is how fast the body and the cooldowns move.
    const tempo = U.clamp(Math.pow(lv.spe / 100, 0.5), 0.6, 1.5);
    const might = (stat) => U.clamp(Math.pow(stat / 105, 0.6), 0.65, 1.6);
    // base Speed sets the pace, weight slows the run a little, body shape picks the gait
    const speed = (130 + p.st.spe * 0.95) * U.clamp(1.06 - (mass - 1) * 0.07, 0.8, 1.08) * U.clamp(tempo, 0.85, 1.2);
    const gait = Fighter.canFly(p) ? "fly" : sh === "ball" || sh === "blob" ? "bounce" : sh === "squiggle" || sh === "fish" || sh === "tentacles" ? "slither"
      : mass >= 2.1 || sh === "armor" || sh === "heads" ? "stomp" : sh === "quadruped" ? "gallop" : "walk";
    return {
      mass, speed, gait, bodyH, tempo,
      mightAtk: might(lv.atk), mightSpa: might(lv.spa),
      reach: U.clamp(Math.pow(bodyH / Assets.trainerArtPx(), 0.5), 0.8, 1.35),
      push: U.clamp(Math.pow((p.w || 30) / 30, 0.15), 0.8, 1.4),
    };
  }
}
Fighter.GAIT_JP = { fly: "ふわふわ うかぶ", bounce: "ぽよんと はねる", slither: "くねくね すすむ", stomp: "ドスドス あるく", gallop: "かろやかに かける", walk: "てくてく あるく" };
const BLINKERS = /^(mewtwo|mew|abra|kadabra|alakazam)(-|$)/;
const HOVER_SLUGS = /^(gastly|haunter|magnemite|magneton|magnezone|koffing|weezing)(-|$)/; // ゴース・ゴースト／コイル系／ドガース系
const GROUND_BIRDS = /^(zapdos-galar|doduo|dodrio|farfetchd|sirfetchd|delibird|archen|oricorio|flamigo)(-|$)/; // ガラルサンダー、ドードー・ドードリオ、カモネギ系、デリバード、アーケン、オドリドリ、カラミンゴ
// [ユーザー指定 2026-11-15] 「鳥や虫ポケモンの中で、羽を閉じているモンスターは地上に足をつける」。front.gif の絵で、羽をたたんで足で立っている種。
// フォームごとに絵が違う（ピジョット＝たたんでいる／メガピジョット＝広げている 等）ので、接頭辞ではなく slug の完全一致で持つ。
const WINGS_FOLDED = new Set(["pidgey", "pidgeot", "spearow", "scyther", "hoothoot", "noctowl", "ledyba", "ledian", "natu", "xatu", "murkrow", "skarmory",
  "taillow", "swellow", "altaria", "starly", "staravia", "staraptor", "staraptor-mega", "honchkrow", "chatot", "pidove", "tranquill", "unfezant",
  "ducklett", "swanna", "rufflet", "vullaby", "mandibuzz", "fletchling", "rowlet", "dartrix", "pikipek", "trumbeak", "toucannon", "rookidee", "corviknight", "cramorant"]);
const BUG_WALKERS = /^(scizor|kricketune|kleavor)(-|$)/; // ハッサム・コロトック・バサギリは歩く
const HOVER_SHAPES = ["wings", "squiggle", "bug-wings"]; // 鳥型・蛇型（ギャラドス、レックウザ等）・羽根のある虫型（バタフリー等、2026-11-15 ユーザー追加）
const FLOATERS = ["gengar", "gastly", "haunter", "rotom", "bronzong", "magnezone", "magnemite", "magneton", "koffing", "weezing", "mismagius", "misdreavus", "latias", "latios", "mewtwo", "mew", "uxie", "mesprit", "azelf", "giratina-origin", "hydreigon", "vikavolt", "cresselia", "chandelure", "lunala"];
Fighter.MOVE_RATE = 0.78;
// [ユーザー指定 2026-11-15] 「もう少しだけ、weight に応じた按配で速く」：軽いほど上乗せが大きい（10kg 以下 +0.10、300kg 以上 +0.03、間は体重の対数で補間）。
// 例：ピカチュウ 6kg → 0.88、リザードン 90.5kg → 0.83、カビゴン 460kg → 0.81。
Fighter.moveRate = (p) => Fighter.MOVE_RATE + U.lerp(0.10, 0.03, U.clamp(Math.log10(Math.max(1, +(p && p.w) || 30) / 10) / Math.log10(30), 0, 1));
