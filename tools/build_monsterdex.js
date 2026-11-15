#!/usr/bin/env node
// Builds data/monsterdex.js (compact roster + move catalog) for Pokémon Arena REVENGE.
// Source: the full monsterData dump (default ./monsterData) + ./monsterSprite/manifest.json.
// Usage: node tools/build_monsterdex.js [path/to/monsterData]
"use strict";
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const SRC = path.resolve(process.argv[2] || path.join(ROOT, "monsterData"));
const SPRITES = path.join(ROOT, "monsterSprite");
const OUT = path.join(ROOT, "data", "monsterdex.js");

const index = JSON.parse(fs.readFileSync(path.join(SRC, "index.json"), "utf8"));
const manifest = JSON.parse(fs.readFileSync(path.join(SPRITES, "manifest.json"), "utf8"));
const chart = JSON.parse(fs.readFileSync(path.join(SRC, "type_matchups_chart.json"), "utf8"));
const movesCatalog = JSON.parse(fs.readFileSync(path.join(SRC, "moves_catalog.json"), "utf8"));

const ICON_PREF = ["champions.png", "swsh.png", "legends-za.png", "gen7.png", "gen6.png", "gen5.png", "lgpe.png", "sv.png", "legends-arceus.png"];

// Moves that read badly in a real-time arena (OHKO, counters, two-turn charge, etc.).
const BANNED = new Set([
  "fissure", "horn-drill", "guillotine", "sheer-cold", "counter", "mirror-coat", "metal-burst", "bide",
  "explosion", "self-destruct", "misty-explosion", "final-gambit", "memento", "healing-wish", "lunar-dance",
  "struggle", "fling", "natural-gift", "return", "frustration", "hidden-power", "dream-eater", "snore",
  "fake-out", "first-impression", "last-resort", "spit-up", "belch", "uproar", "thrash", "outrage", "petal-dance",
  "rollout", "ice-ball", "bind", "wrap", "fury-cutter", "echoed-voice", "round", "beat-up", "present",
  "magnitude", "endeavor", "super-fang", "nature-power", "me-first", "copycat", "mirror-move", "assist",
  "sleep-talk", "metronome", "sketch", "transform", "mimic", "foul-play", "psywave", "night-shade", "seismic-toss",
  "sonic-boom", "dragon-rage", "reversal", "flail", "gyro-ball", "electro-ball", "heavy-slam", "heat-crash",
  "grass-knot", "low-kick", "crush-grip", "wring-out", "power-trip", "stored-power", "punishment", "trump-card",
  "skull-bash", "sky-attack", "razor-wind", "solar-blade", "bounce", "dig", "dive", "fly", "phantom-force", "shadow-force",
  "freeze-shock", "ice-burn", "geomancy", "meteor-beam", "electro-shot", "skull-bash", "focus-punch", "doom-desire", "future-sight",
  "tera-blast", "tera-starstorm", "revelation-dance", "multi-attack", "judgment", "techno-blast", "weather-ball",
  "aura-wheel", "ivy-cudgel", "raging-bull", "steel-roller", "burn-up", "double-shock",
]);

// Status moves that have a real arena implementation (see game/js/moves.js). Everything else is skipped.
const STATUS_OK = new Set([
  "swords-dance", "dragon-dance", "nasty-plot", "calm-mind", "bulk-up", "agility", "iron-defense", "amnesia",
  "recover", "roost", "soft-boiled", "synthesis", "moonlight", "morning-sun", "rest", "slack-off", "milk-drink", "shore-up",
  "protect", "detect", "will-o-wisp", "toxic", "thunder-wave", "stun-spore", "sleep-powder", "spore", "hypnosis",
  "sandstorm", "rain-dance", "sunny-day", "hail", "snowscape", "quiver-dance", "shell-smash", "coil", "curse", "growth",
  "work-up", "howl", "harden", "withdraw", "barrier", "cotton-guard", "acid-armor", "leech-seed", "substitute",
  "wish", "aqua-ring", "ingrain", "poison-powder", "sing", "lovely-kiss", "glare", "nuzzle", "confuse-ray", "sweet-kiss",
  "double-team", "minimize", "light-screen", "reflect", "aromatherapy", "heal-bell", "life-dew", "jungle-healing",
  "focus-energy", "belly-drum", "tail-glow", "cosmic-power", "stockpile", "defend-order", "heal-order",
  "shift-gear", "autotomize", "smokescreen", "teleport", "rock-polish", "charge", "power-up-punch", "victory-dance", "clangorous-soul", "no-retreat", "tidy-up",
]);

// Recoil, recharge, stat-drop or very slow moves.
const DRAWBACK = new Set([
  "hyper-beam", "giga-impact", "double-edge", "take-down", "wild-charge", "flare-blitz", "brave-bird", "head-smash",
  "volt-tackle", "wood-hammer", "overheat", "draco-meteor", "leaf-storm", "psycho-boost", "superpower", "close-combat",
  "steel-beam", "wave-crash", "head-charge", "submission", "hammer-arm", "high-jump-kick", "jump-kick", "fleur-cannon",
  "prismatic-laser", "eternabeam", "blast-burn", "hydro-cannon", "frenzy-plant", "rock-wrecker", "roar-of-time",
  "mind-blown", "chloroblast", "light-of-ruin", "dragon-energy", "make-it-rain", "armor-cannon", "gigaton-hammer",
  "blood-moon", "glaive-rush", "axe-kick", "supercell-slam", "thunder", "blizzard", "fire-blast", "hydro-pump", "focus-blast",
  "hurricane", "zap-cannon", "dynamic-punch", "megahorn", "stone-edge", "sky-uppercut", "cross-chop", "power-whip",
  "hyper-voice", "self-destruct", "explosion", "uproar", "overdrive", "boomburst", "mega-kick",
]);

// Abilities with an arena effect (game/js/ability.js). A Pokémon gets the first of its abilities found here
// (regular slots before the hidden one); otherwise its first regular ability, shown but inert.
const ABILITY_OK = new Set([
  "intimidate", "levitate", "sturdy", "swift-swim", "chlorophyll", "sand-rush", "slush-rush", "blaze", "torrent", "overgrow", "swarm",
  "thick-fat", "flash-fire", "water-absorb", "volt-absorb", "dry-skin", "lightning-rod", "storm-drain", "sap-sipper", "motor-drive",
  "static", "flame-body", "poison-point", "effect-spore", "guts", "technician", "huge-power", "pure-power", "adaptability", "sheer-force",
  "iron-fist", "strong-jaw", "tough-claws", "multiscale", "shadow-shield", "regenerator", "natural-cure", "speed-boost", "moxie",
  "inner-focus", "steadfast", "rock-head", "hustle", "sniper", "super-luck", "keen-eye", "pressure", "shell-armor", "battle-armor",
  "clear-body", "white-smoke", "prankster", "drought", "drizzle", "sand-stream", "snow-warning", "magic-guard", "insomnia", "vital-spirit",
  "limber", "immunity", "own-tempo", "defiant", "competitive", "run-away", "beast-boost", "solar-power", "rain-dish", "ice-body",
  "poison-heal", "marvel-scale", "filter", "solid-rock", "prism-armor", "wonder-guard", "sand-force", "reckless", "aerilate", "pixilate", "refrigerate",
]);
const abil = (a) => a && { s: a.slug, jp: a.jpName || a.name, d: (a.jpDescription || a.description || "").replace(/\s+/g, " ").trim() };

const stats = { hp: "hp", attack: "atk", defense: "def", specialAttack: "spa", specialDefense: "spd", speed: "spe" };
const pokedex = [];
const usedMoves = new Set();
const skipped = [];

function pickIcon(dir) {
  const d = path.join(dir, "box-icons");
  if (!fs.existsSync(d)) return null;
  const have = new Set(fs.readdirSync(d));
  for (const n of ICON_PREF) if (have.has(n)) return `box-icons/${n}`;
  const any = [...have].find((n) => n.endsWith(".png"));
  return any ? `box-icons/${any}` : null;
}

function spriteFolder(entry) {
  // index.json sprite path: ../monsterSprite/gen1/0006_charizard/normal/front.gif
  const m = entry.sprite.match(/monsterSprite\/(.+)\/front\.gif$/);
  return m ? m[1] : null;
}

for (const e of index.pokemon) {
  const full = JSON.parse(fs.readFileSync(path.join(SRC, e.dataPath.replace(/^\.\.\/monsterData\//, "")), "utf8"));
  const folder = spriteFolder(e);
  if (!folder) { skipped.push(e.id + " (no sprite)"); continue; }
  const dir = path.join(SPRITES, folder);
  if (!fs.existsSync(path.join(dir, "front.gif"))) { skipped.push(e.id + " (missing gif)"); continue; }
  const hasShiny = fs.existsSync(path.join(dir, "front-shiny.gif"));
  const icon = pickIcon(dir);

  // Candidate moves: level-up (weighted by level), then machine/tutor/egg.
  const pool = new Map();
  const add = (m, src) => {
    if (!m || BANNED.has(m.slug)) return;
    const cat = movesCatalog[m.id] || m;
    if (cat.category === "status" && !STATUS_OK.has(m.slug)) return;
    if (cat.category !== "status" && (!cat.power || cat.power < 15)) return;
    if (!pool.has(m.slug)) pool.set(m.slug, { slug: m.slug, src, level: m.level || 0 });
  };
  for (const m of full.moves.levelUp || []) add(m, "level");
  for (const m of full.moves.tutor || []) add(m, "tutor");
  for (const m of full.moves.machine || []) add(m, "machine");
  for (const m of full.moves.egg || []) add(m, "egg");
  const types = full.typeMatchups.types;
  const bs = full.baseStats;
  const physical = bs.attack >= bs.specialAttack;
  // Score: STAB + power + matching attack stat + learned naturally.
  const scored = [...pool.values()].map((p) => {
    const m = Object.values(movesCatalog).find((c) => c.slug === p.slug);
    if (!m) return null;
    let s = 0;
    if (m.category === "status") s = 35;
    else {
      // Sweet spot is the reliable 70–95 power move; nukes with recoil / recharge / self-debuffs rank low so
      // default loadouts read like the anime (Flamethrower, not Hyper Beam).
      const pw = Math.min(m.power, 100) - Math.max(0, m.power - 100) * 0.8;
      s = pw * (m.accuracy ? Math.pow(m.accuracy / 100, 1.5) : 0.95);
      if (DRAWBACK.has(m.slug)) s *= 0.55;
      if (types.includes(m.type)) s *= 1.6;
      if ((m.category === "physical") === physical) s *= 1.2;
    }
    if (p.src === "level") s *= 1.3;
    if (p.src === "egg") s *= 0.85;
    return { slug: m.slug, type: m.type, cat: m.category, s };
  }).filter(Boolean).sort((a, b) => b.s - a.s);

  // Keep a diverse pool: max 3 per type for attacks, up to 14 attacks + 4 status.
  const perType = {};
  const attacks = [], status = [];
  for (const m of scored) {
    if (m.cat === "status") { if (status.length < 4) status.push(m.slug); continue; }
    perType[m.type] = (perType[m.type] || 0) + 1;
    if (perType[m.type] > 3) continue;
    if (attacks.length < 14) attacks.push(m.slug);
  }
  // Default loadout: best STAB, second STAB (other type if possible), best coverage, one status/utility.
  const loadout = [];
  const stabA = scored.find((m) => m.cat !== "status" && m.type === types[0]);
  const stabB = scored.find((m) => m.cat !== "status" && types[1] && m.type === types[1]);
  if (stabA) loadout.push(stabA.slug);
  if (stabB && !loadout.includes(stabB.slug)) loadout.push(stabB.slug);
  for (const m of scored) {
    if (loadout.length >= 3) break;
    if (m.cat === "status" || loadout.includes(m.slug) || types.includes(m.type)) continue;
    loadout.push(m.slug);
  }
  for (const m of scored) { if (loadout.length >= 3) break; if (m.cat !== "status" && !loadout.includes(m.slug)) loadout.push(m.slug); }
  if (status.length) loadout.push(status[0]);
  for (const m of scored) { if (loadout.length >= 4) break; if (!loadout.includes(m.slug)) loadout.push(m.slug); }
  if (!loadout.length) loadout.push("tackle", "body-slam");
  if (!attacks.length) attacks.push("tackle", "body-slam");

  for (const s of [...attacks, ...status, ...loadout]) usedMoves.add(s);

  const st = {};
  for (const [k, v] of Object.entries(stats)) st[v] = bs[k];
  pokedex.push({
    id: e.id, dex: e.dex, slug: e.slug, name: e.name, jp: e.jpName, cat: e.category, gen: e.generation,
    types, st, h: full.height.meters, w: full.weight.kilograms,
    leg: !!(full.isLegendary || full.isMythical),
    dir: folder, shiny: hasShiny, icon,
    moves: loadout, pool: attacks, util: status,
    genus: full.jpGenus || full.genus || "", shape: (full.shape && full.shape.slug) || "upright",
    ab: abil([...(full.abilities || [])].sort((a, b) => a.isHidden - b.isHidden).find((a) => ABILITY_OK.has(a.slug)) || (full.abilities || []).find((a) => !a.isHidden) || (full.abilities || [])[0]),
  });
}

// Link base forms to their mega / gigantamax forms.
const bySlug = new Map(pokedex.map((p) => [p.slug, p]));
for (const p of pokedex) {
  if (p.cat === "mega" || p.cat === "gigantamax") {
    const base = p.dir.split("/")[1].replace(/^\d+_/, "");
    const b = bySlug.get(base);
    if (b) { (b.forms = b.forms || []).push(p.slug); p.base = base; }
  }
}

const moveOut = {};
for (const m of Object.values(movesCatalog)) {
  if (!usedMoves.has(m.slug)) continue;
  moveOut[m.slug] = { n: m.name, jp: m.jpName, t: m.type, c: m.category[0], p: m.power || 0, a: m.accuracy || 0, pr: m.priority || 0 };
}
if (!moveOut.tackle) moveOut.tackle = { n: "Tackle", jp: "たいあたり", t: "normal", c: "p", p: 40, a: 100, pr: 0 };
if (!moveOut["body-slam"]) moveOut["body-slam"] = { n: "Body Slam", jp: "のしかかり", t: "normal", c: "p", p: 85, a: 100, pr: 0 };

const types = chart.types.map((t) => ({ slug: t.slug, name: t.name, jp: t.jpName }));
const out = `// Generated by tools/build_monsterdex.js — do not edit by hand.\n` +
  `window.ARENA_TYPES = ${JSON.stringify(types)};\n` +
  `window.ARENA_EFFICACY = ${JSON.stringify(chart.efficacy)};\n` +
  `window.ARENA_MOVES = ${JSON.stringify(moveOut)};\n` +
  `window.ARENA_MONSTERDEX = ${JSON.stringify(pokedex)};\n`;
fs.writeFileSync(OUT, out);
console.log(`pokedex: ${pokedex.length} forms, ${Object.keys(moveOut).length} moves, ${(out.length / 1024).toFixed(0)} KB -> ${path.relative(ROOT, OUT)}`);
if (skipped.length) console.log("skipped:", skipped.join(", "));
