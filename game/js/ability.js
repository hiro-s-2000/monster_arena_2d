// Abilities (とくせい) in the arena. Each fighter carries species.ab = {s: slug, jp, d}; the hooks below are
// called from Battle / Fighter / Data.damage. A proc shows a Gen-5 style ability bar via B.event("ability").
"use strict";

const Ability = (() => {
  const has = (f, ...s) => !!(f && f.species.ab && s.includes(f.species.ab.s));
  const name = (f) => (f && f.species.ab ? f.species.ab.jp : "");
  const STAGE_JP = { atk: "こうげき", def: "ぼうぎょ", spa: "とくこう", spd: "とくぼう", spe: "すばやさ" };
  const WEATHER = { drought: "sun", drizzle: "rain", "sand-stream": "sand", "snow-warning": "hail" };
  const RUSH = { "swift-swim": "rain", chlorophyll: "sun", "sand-rush": "sand", "slush-rush": "hail" };
  const PINCH = { blaze: "fire", torrent: "water", overgrow: "grass", swarm: "bug" };
  const ATE = { aerilate: "flying", pixilate: "fairy", refrigerate: "ice" };
  const PUNCH = /punch|hammer-arm|meteor-mash|sky-uppercut|dynamic-punch|drain-punch|bullet-punch|mach-punch|focus-punch|comet-punch|shadow-punch|power-up-punch|ice-hammer|plasma-fists|double-iron-bash/;
  const BITE = /bite|fang|crunch|jaw-lock|psychic-fangs|fishious-rend/;
  const ABSORB = { "water-absorb": ["water", "heal"], "dry-skin": ["water", "heal"], "volt-absorb": ["electric", "heal"], "flash-fire": ["fire", "fire"],
    "lightning-rod": ["electric", "spa"], "storm-drain": ["water", "spa"], "sap-sipper": ["grass", "atk"], "motor-drive": ["electric", "spe"], levitate: ["ground", "none"] };

  function show(B, f, extra) {
    if (!f || !f.species.ab) return;
    B.event("ability", { f, text: name(f), extra });
    B.fx.ring(f.x, f.y, f.r * 0.6, f.r + 34, 0.4, "#ffe9a0", 3);
    Audio.sfx("ability");
  }
  function stage(B, f, k, n, by) {
    if (n < 0 && has(f, "clear-body", "white-smoke") && by !== f) { show(B, f, "のうりょくは さがらない！"); return; }
    const before = f.stages[k];
    f.stages[k] = U.clamp(f.stages[k] + n, -6, 6);
    if (f.stages[k] === before) return;
    B.fx.label(f.x, f.y - f.height - 24, `${STAGE_JP[k]}${n > 0 ? "▲" : "▼"}`, n > 0 ? "#9bffb0" : "#ff9b9b", { delay: 0.3, size: 11 });
    if (n < 0 && by !== f && has(f, "defiant", "competitive")) { show(B, f); stage(B, f, has(f, "defiant") ? "atk" : "spa", 2, f); }
  }

  // Fighter just landed on the field.
  function onEnter(B, f) {
    const foe = B.active[1 - f.side];
    const ab = f.species.ab && f.species.ab.s;
    if (ab === "intimidate" && foe && !foe.ko && foe.active) { show(B, f); stage(B, foe, "atk", -1, f); B.fx.burst("dark", foe.x, foe.y - foe.height * 0.5, 10, { h: foe.height * 0.5, size: 0.8 }); }
    else if (WEATHER[ab]) { show(B, f); B.setWeather(WEATHER[ab], 14, f); }
    else if (ab === "pressure" || ab === "levitate" || ab === "multiscale" || ab === "shadow-shield" || ab === "wonder-guard") show(B, f);
  }
  function onSwitchOut(B, f) {
    if (has(f, "regenerator") && !f.ko) f.heal(Math.round(f.maxHp / 3));
    if (has(f, "natural-cure")) f.cure();
  }
  function onKO(B, by) {
    if (!by || by.ko) return;
    if (has(by, "moxie")) { show(B, by); stage(B, by, "atk", 1, by); }
    if (has(by, "beast-boost")) { const st = by.stats, best = ["atk", "def", "spa", "spd", "spe"].sort((a, b) => st[b] - st[a])[0]; show(B, by); stage(B, by, best, 1, by); }
  }
  // per-frame passive effects (realtime only)
  function update(B, f, dt) {
    if (!f.species.ab || B.turnMode) return;
    const ab = f.species.ab.s, w = B.weather.kind;
    f.abT = (f.abT || 0) + dt;
    if (f.abT < 6) return;
    f.abT = 0;
    if (ab === "speed-boost" && f.stages.spe < 3) { show(B, f); stage(B, f, "spe", 1, f); }
    if ((ab === "rain-dish" && w === "rain") || (ab === "ice-body" && w === "hail") || (ab === "dry-skin" && w === "rain") || (ab === "poison-heal" && f.status === "psn")) {
      const h = f.heal(Math.round(f.maxHp / 12)); if (h > 0) { show(B, f); B.fx.number(f.x, f.y - f.height - 10, "+" + h, { color: "#8ee07a", size: 16 }); }
    }
  }

  // ---- combat modifiers ---------------------------------------------------------------------------
  // Type change for -ate abilities (applied once to the move specs).
  function moveType(f, spec) { const t = ATE[f.species.ab && f.species.ab.s]; return t && spec.type === "normal" ? Object.assign({}, spec, { type: t, power: Math.round(spec.power * 1.2), ate: true }) : spec; }
  // Attack-side multiplier (Data.damage).
  function attackMul(att, def, spec, type, special, wk) {
    if (!att.species || !att.species.ab) return 1;
    const ab = att.species.ab.s, slug = spec.slug || "";
    let m = 1;
    if (PINCH[ab] === type && att.hp <= att.maxHp / 3) m *= 1.5;
    if (ab === "flash-fire" && att.flashFire && type === "fire") m *= 1.5;
    if ((ab === "huge-power" || ab === "pure-power") && !special) m *= 1.5;
    if (ab === "guts" && att.status && !special) m *= 1.5;
    if (ab === "hustle" && !special) m *= 1.3;
    if (ab === "technician" && (spec.power || spec.p || 0) <= 60) m *= 1.5;
    if (ab === "sheer-force" && spec.sec) m *= 1.3;
    if (ab === "iron-fist" && PUNCH.test(slug)) m *= 1.2;
    if (ab === "strong-jaw" && BITE.test(slug)) m *= 1.5;
    if (ab === "tough-claws" && (spec.kind === "melee" || spec.kind === "dash" || spec.kind === "leap")) m *= 1.3;
    if (ab === "reckless" && spec.recoil) m *= 1.2;
    if (ab === "solar-power" && special && wk === "sun") m *= 1.5;
    if (ab === "sand-force" && wk === "sand" && ["rock", "ground", "steel"].includes(type)) m *= 1.3;
    return m;
  }
  const stabMul = (att) => (has(att, "adaptability") ? 2 : 1.5);
  // Defence-side multiplier (Data.damage).
  function defendMul(att, def, type, eff, special) {
    if (!def.species || !def.species.ab) return 1;
    const ab = def.species.ab.s;
    let m = 1;
    if (ab === "thick-fat" && (type === "fire" || type === "ice")) m *= 0.5;
    if ((ab === "multiscale" || ab === "shadow-shield") && def.hp >= def.maxHp) m *= 0.5;
    if ((ab === "filter" || ab === "solid-rock" || ab === "prism-armor") && eff > 1) m *= 0.75;
    if (ab === "marvel-scale" && def.status && !special) m /= 1.5;
    if (ab === "dry-skin" && type === "fire") m *= 1.25;
    return m;
  }
  function critRate(att, def, base) {
    if (has(def, "shell-armor", "battle-armor")) return 0;
    return has(att, "super-luck") ? Math.max(base, 0.2) : base;
  }
  const critMul = (att) => (has(att, "sniper") ? 2.25 : 1.5);

  // Type immunities that heal / boost. Returns true when the hit was absorbed.
  function absorb(B, target, attacker, spec) {
    const ab = target.species.ab && target.species.ab.s, a = ABSORB[ab];
    if (has(target, "wonder-guard") && Data.effectiveness(spec.type, target.species.types) <= 1 && spec.power) { show(B, target, "こうかが ない！"); return true; }
    if (!a || a[0] !== spec.type || !spec.power) return false;
    if (target.absorbT > B.time) return true; // beam ticks: one bar per volley
    target.absorbT = B.time + 0.6;
    if (a[1] === "heal") { const h = target.heal(Math.round(target.maxHp / 4)); show(B, target, h > 0 ? `HPが かいふくした！` : "こうかが ない！"); if (h > 0) B.fx.number(target.x, target.y - target.height - 10, "+" + h, { color: "#8ee07a", size: 16 }); }
    else if (a[1] === "fire") { target.flashFire = true; show(B, target, "ほのおの いりょくが あがった！"); }
    else if (a[1] === "none") { show(B, target, "うかんでいる！"); }
    else { show(B, target); stage(B, target, a[1], 1, target); }
    B.fx.burst(spec.type, target.x, target.y - target.height * 0.5, 10, { h: target.height * 0.5, size: 0.8, owner: attacker });
    return true;
  }
  // Contact moves can trigger the defender's body ability.
  function onContact(B, target, attacker) {
    if (!target.species.ab || attacker.ko || Math.random() > 0.3) return;
    const st = { static: "para", "flame-body": "burn", "poison-point": "psn", "effect-spore": U.pick(["para", "psn", "slp"]) }[target.species.ab.s];
    if (st && attacker.setStatus(st)) { show(B, target); B.fx.label(attacker.x, attacker.y - attacker.height - 26, STATUS_NAMES[st] + "！", STATUS_COLORS[st], { delay: 0.3 }); }
  }
  // Sturdy: a full-HP fighter survives any single hit with 1 HP.
  function endure(B, f, hpBefore) {
    if (has(f, "sturdy") && hpBefore >= f.maxHp && f.hp <= 0) { f.hp = 1; show(B, f, "こらえた！"); return true; }
    return false;
  }
  function statusImmune(f, st) {
    return (st === "slp" && has(f, "insomnia", "vital-spirit")) || (st === "para" && has(f, "limber")) || (st === "psn" && has(f, "immunity")) || (st === "conf" && has(f, "own-tempo"));
  }
  function speedMul(f, B) { const w = RUSH[f.species.ab && f.species.ab.s]; return w && B && B.weather.kind === w ? 1.5 : 1; }
  const noRecoil = (f) => has(f, "rock-head", "magic-guard");
  const noIndirect = (f) => has(f, "magic-guard");
  const hitstunMul = (f) => (has(f, "inner-focus", "steadfast") ? 0.5 : 1);
  const castRate = (f, spec) => (has(f, "prankster") && !spec.power ? 2.2 : 1);
  const dodgeCdMul = (f) => (has(f, "run-away") ? 0.6 : 1);
  const ignoresAccDown = (f) => has(f, "keen-eye");
  const energyMul = (f, B) => { const foe = B.active[1 - f.side]; return foe && !foe.ko && has(foe, "pressure") ? 0.85 : 1; };

  return { has, name, show, stage, onEnter, onSwitchOut, onKO, update, moveType, attackMul, stabMul, defendMul, critRate, critMul, absorb, onContact, endure, statusImmune, speedMul, noRecoil, noIndirect, hitstunMul, castRate, dodgeCdMul, ignoresAccDown, energyMul };
})();
