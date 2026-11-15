// Roster, type chart and the damage formula. Reads the generated window.ARENA_* bundle (data/monsterdex.js).
"use strict";

const Data = (() => {
  // Canonical type palette (main, light, dark) — used by HUD badges and particle themes.
  const TYPE_COLORS = {
    normal: ["#a8a878", "#d8d8c0", "#6d6d4e"], fire: ["#f08030", "#ffc06a", "#9c2a0a"], water: ["#4a90e2", "#9cd4ff", "#1b3f8a"],
    electric: ["#f8d030", "#fff39a", "#a37f00"], grass: ["#78c850", "#b7f29a", "#2e6b1f"], ice: ["#98d8d8", "#e6ffff", "#3f8f9b"],
    fighting: ["#c03028", "#f28077", "#5e0d0b"], poison: ["#a040a0", "#e09be0", "#4b1a4b"], ground: ["#e0c068", "#f5e3a8", "#7a5a1c"],
    flying: ["#a890f0", "#dcd2ff", "#4e3f9c"], psychic: ["#f85888", "#ffb3c8", "#8d1c45"], bug: ["#a8b820", "#dfe97a", "#55600a"],
    rock: ["#b8a038", "#e2d08a", "#5f5016"], ghost: ["#705898", "#b8a4e0", "#2b1d47"], dragon: ["#7038f8", "#b79bff", "#2b0f7a"],
    dark: ["#705848", "#a88f7c", "#241a12"], steel: ["#b8b8d0", "#eaeaf5", "#5c5c7a"], fairy: ["#ee99ac", "#ffd3dd", "#9a3b56"],
  };
  const TYPE_JP = {};
  for (const t of window.ARENA_TYPES || []) TYPE_JP[t.slug] = t.jp;
  const EFF = window.ARENA_EFFICACY || {};
  const DEX = window.ARENA_MONSTERDEX || [];
  const MOVES = window.ARENA_MOVES || {};

  // Hand-picked movesets that override the generated ones (Mewtwo as in the 6v6 reference clip).
  const SETS = { mewtwo: ["psystrike", "shadow-ball", "psychic", "expanding-force"], "mewtwo-mega-y": ["psystrike", "shadow-ball", "psychic", "expanding-force"] };
  // Burrowers learn Dig in place of one of their moves (base slug → the move it replaces; forms included).
  const DIGGERS = { sandshrew: "slash", sandslash: "slash", diglett: "slash", dugtrio: "sucker-punch", onix: "bulldoze", steelix: "iron-head",
    drilbur: "crush-claw", excadrill: "crush-claw", garchomp: "crunch", gible: "slash", krookodile: "body-slam" };
  // [ユーザー指定 2026-11-15] うずしお（whirlpool）を使うモンスター（base slug → 置き換える技。フォームも含む）。技データに無かったので手で配る。
  const WHIRLERS = { gyarados: "rain-dance", horsea: "smokescreen", seadra: "smokescreen", kingdra: "smokescreen", tentacool: "acid-armor", tentacruel: "acid-armor", poliwhirl: "hypnosis", poliwrath: "hypnosis" };
  const CAT_LABEL = { gen: "通常", mega: "メガ", regional: "リージョン", gigantamax: "キョダイ" };
  let roster = null;
  const bySlug = new Map();

  function build() {
    if (roster) return roster;
    roster = DEX.map((p) => {
      const kind = p.cat.startsWith("gen") ? "gen" : p.cat;
      return Object.assign(p, {
        kind,
        iconUrl: Assets.iconUrl(p),
        search: `${p.name} ${p.jp} ${p.slug} ${p.dex} ${p.types.map((t) => t + " " + TYPE_JP[t]).join(" ")} ${CAT_LABEL[kind]}`.toLowerCase(),
      });
    });
    for (const p of roster) {
      bySlug.set(p.slug, p);
      if (SETS[p.slug]) p.moves = SETS[p.slug];
      const dig = DIGGERS[p.slug.split("-")[0]];
      const wh = WHIRLERS[p.slug.split("-")[0]];
      if (wh && p.moves && !p.moves.includes("whirlpool")) { const i = p.moves.indexOf(wh); if (i >= 0) p.moves = p.moves.map((m, j) => (j === i ? "whirlpool" : m)); }
      if (dig && p.moves && !p.moves.includes("dig")) { const i = p.moves.indexOf(dig); if (i >= 0) p.moves = p.moves.map((m, j) => (j === i ? "dig" : m)); }
    }
    return roster;
  }

  // Level-50 stats with 31 IV / 84 EV; HP is padded so an arena fight lasts ~5–8 hits.
  function stats(p) {
    const b = p.st;
    const base = (v) => Math.floor(((2 * v + 31 + 21) * 50) / 100) + 5;
    return { hp: Math.floor(((2 * b.hp + 31 + 21) * 50) / 100) + 60, atk: base(b.atk), def: base(b.def), spa: base(b.spa), spd: base(b.spd), spe: base(b.spe) };
  }

  function effectiveness(type, targetTypes) {
    let m = 1;
    const row = EFF[type];
    if (!row) return 1;
    for (const t of targetTypes) m *= row[t] === undefined ? 1 : row[t];
    return m;
  }

  const stageMul = (s) => (s >= 0 ? (2 + s) / 2 : 2 / (2 - s));

  // Gen-5 style damage. att/def are fighters (stats, types, stages, status, weather bonus).
  // Accepts either a catalog entry ({t, c, p}) or an arena spec ({type, cat, power}).
  // ---- weight & stat driven effects (real-time and turn modes share these) --------------------------
  // rel: (a/b)^e clamped — a stat matchup turned into a gentle multiplier.
  const rel = (a, b, e, lo, hi) => U.clamp(Math.pow(Math.max(1, a) / Math.max(1, b), e), lo, hi);
  const offStat = (f, special) => (special ? f.stats.spa * stageMul(f.stages.spa) : f.stats.atk * stageMul(f.stages.atk));
  const defStat = (f, special) => (special ? f.stats.spd * stageMul(f.stages.spd) : f.stats.def * stageMul(f.stages.def));
  const kg = (f) => (f.species && f.species.w) || 30;
  // Physical hits: a heavier attacker lands harder, a heavier target soaks a little (×0.8 … ×1.25).
  const weightMul = (att, def) => U.clamp(Math.pow(kg(att) / kg(def), 0.08), 0.8, 1.25);
  // Moves whose power is set by weight (as in the main series); they skip weightMul.
  function weightPower(slug, att, def) {
    if (slug === "heavy-slam" || slug === "heat-crash") { const r = kg(att) / kg(def); return r >= 5 ? 120 : r >= 4 ? 100 : r >= 3 ? 80 : r >= 2 ? 60 : 40; }
    if (slug === "low-kick" || slug === "grass-knot") { const w = kg(def); return w < 10 ? 20 : w < 25 ? 40 : w < 50 ? 60 : w < 100 ? 80 : w < 200 ? 100 : 120; }
    return 0;
  }
  const FX = {
    // こうげき/とくこう vs ぼうぎょ/とくぼう: how far and how long the target is knocked back / stunned
    knock: (att, def, special) => rel(offStat(att, special), defStat(def, special), 0.35, 0.7, 1.45),
    stun: (att, def, special) => rel(offStat(att, special), defStat(def, special), 0.3, 0.75, 1.35),
    // added-effect chance (burn, flinch, stat drops…): attacker's matching stat vs target's とくぼう
    secChance: (att, def, special) => rel(offStat(att, special), defStat(def, true), 0.5, 0.55, 1.6),
    // status length: とくこう vs とくぼう
    statusTime: (att, def) => rel(offStat(att, true), defStat(def, true), 0.4, 0.65, 1.5),
    // すばやさ: critical-hit rate (~3 % slow … 14 % fast), wind-up speed, projectile / dash speed
    crit: (att) => U.clamp((att.stats.spe * stageMul(att.stages.spe)) / 1600, 0.03, 0.14),
    quick: (f) => U.clamp(Math.pow((f.stats.spe * stageMul(f.stages.spe)) / 100, 0.25), 0.85, 1.2),
    // ぼうぎょ: recoil from your own reckless moves
    recoil: (f) => rel(100, f.stats.def * stageMul(f.stages.def), 0.35, 0.7, 1.3),
    // とくこう: area / zone size and how long zones last
    area: (f) => rel(f.stats.spa * stageMul(f.stages.spa), 100, 0.3, 0.85, 1.25),
  };

  const ARENA_SCALE = 0.5;
  function damage(att, def, move, opts = {}) {
    const cat = move.cat || move.c, type = move.type || move.t, slug = move.slug || "";
    const special = cat === "s";
    // Body Press hits with its user's ぼうぎょ, Foul Play with the target's こうげき
    let A = slug === "body-press" ? att.stats.def * stageMul(att.stages.def) : slug === "foul-play" ? def.stats.atk * stageMul(def.stages.atk) : offStat(att, special);
    if (att.status === "burn" && !special) A *= 0.5;
    const D = defStat(def, special);
    const wp = weightPower(slug, att, def);
    const power = (wp || move.power || move.p || 40) * (opts.powerMul || 1);
    let dmg = (((2 * 50) / 5 + 2) * power * (A / D) / 50 + 2) * ARENA_SCALE;
    const wt = !special && !wp ? weightMul(att, def) : 1;
    const stab = att.species.types.includes(type) ? Ability.stabMul(att) : 1;
    const eff = effectiveness(type, def.species.types);
    const crit = Math.random() < Ability.critRate(att, def, opts.critRate || FX.crit(att)) ? Ability.critMul(att) : 1;
    dmg *= Ability.attackMul(att, def, move, type, special, opts.wk) * Ability.defendMul(att, def, type, eff, special);
    const roll = U.rand(0.85, 1);
    const weather = opts.weatherMul || 1;
    dmg = dmg * wt * stab * eff * crit * roll * weather * (opts.mult || 1);
    return { dmg: Math.max(1, Math.round(dmg)), eff, crit: crit > 1, weight: wp ? (wp >= 100 ? 1.25 : wp <= 40 ? 0.8 : 1) : wt };
  }

  function search(q, kindFilter) {
    const r = build();
    q = (q || "").trim().toLowerCase();
    return r.filter((p) => (!kindFilter || p.kind === kindFilter) && (!q || p.search.includes(q)));
  }

  const typeName = (t) => TYPE_JP[t] || t;
  const typeColor = (t) => TYPE_COLORS[t] || TYPE_COLORS.normal;

  return { TYPE_COLORS, TYPE_JP, MOVES, CAT_LABEL, build, bySlug: (s) => (build(), bySlug.get(s)), stats, effectiveness, damage, stageMul, search, typeName, typeColor, roster: () => build(), FX, weightMul, weightPower };
})();
