// Move archetypes. A catalog move (type / category / power) is turned into an arena spec — projectile,
// beam, melee arc, dash, expanding ring, telegraphed strike, lingering zone, buff, heal, shield, status —
// with a handful of signature overrides so the famous moves look like themselves.
"use strict";

const Moves = (() => {
  const M = Data.MOVES;
  const cache = new Map();

  const set = (s) => new Set(s.split(/\s+/).filter(Boolean));
  const BEAM = set(`flamethrower hydro-pump thunderbolt thunder ice-beam psybeam solar-beam hyper-beam steel-beam dragon-breath bubble-beam
    charge-beam signal-beam hydro-cannon blast-burn frenzy-plant aurora-beam fire-blast-x flash-cannon-x dragon-pulse-x power-gem-x moongeist-beam
    prismatic-laser fleur-cannon eternabeam zap-cannon overheat psycho-boost dark-pulse-x techno-blast freezing-glare fiery-wrath dragon-energy
    water-pulse-x electro-beam scorching-sands-x bitter-blade-x mist-ball luster-purge-x fire-fang-x breaking-swipe-x aeroblast sacred-fire spacial-rend roar-of-time
    ice-burn-x electro-shot-x core-enforcer sunsteel-strike-x light-that-burns-the-sky`);
  const DASH = set(`quick-attack extreme-speed flare-blitz aqua-jet volt-tackle dragon-rush wild-charge brave-bird take-down double-edge giga-impact
    head-smash u-turn flame-charge aerial-ace bullet-punch mach-punch ice-shard shadow-sneak sucker-punch accelerock water-shuriken-x flame-wheel
    body-slam-x headbutt horn-attack high-horsepower wave-crash head-charge zen-headbutt iron-head skull-bash flip-turn volt-switch-x
    first-impression fake-out-x play-rough-x jet-punch collision-course electro-drift psyshield-bash wave-crash raging-fury-x triple-dive-x
    steamroller rollout-x bounce-x fly-x dive-x drill-run drill-peck megahorn-x poltergeist-x tackle slam`);
  const AOE = set(`earthquake bulldoze surf muddy-water discharge blizzard heat-wave lava-plume dazzling-gleam hyper-voice boomburst petal-blizzard
    sludge-wave icy-wind snarl struggle-bug electroweb parabolic-charge origin-pulse precipice-blades eruption water-spout sparkling-aria
    magnitude self-destruct explosion mind-blown earth-power-x powder-snow glaciate frost-breath-x silver-wind ominous-wind razor-leaf-x
    air-cutter twister swift-x round echoed-voice uproar disarming-voice bug-buzz-x clanging-scales clangorous-soulblaze overdrive
    dragon-energy-x fiery-dance-x lands-wrath thousand-arrows thousand-waves continental-crush steam-eruption mystical-fire-x
    chatter relic-song hyper-voice-x stomping-tantrum high-horsepower-x heavy-slam-x body-press-x expanding-force psychic-noise`);
  const STRIKE = set(`rock-slide stone-edge fire-blast earth-power psychic psystrike draco-meteor meteor-beam ancient-power power-gem
    rock-tomb rock-wrecker thunder-x hurricane sky-attack blizzard-x sheer-cold future-sight doom-desire meteor-mash-x
    magma-storm-x stealth-rock spikes rock-throw-x smack-down seed-bomb-x energy-ball-x sludge-bomb-x gunk-shot-x focus-blast-x
    extrasensory psyshock stored-power-x hex-x hyperspace-hole hyperspace-fury earth-power zen-headbutt-x blast-burn-x inferno-x
    searing-shot-x pyro-ball-x shadow-bone-x bone-rush-x diamond-storm head-smash-x ice-hammer-x icicle-crash astonish-x
    phantom-force-x shadow-force-x moonblast-x fleur-cannon-x thunder draining-kiss-x psychic-fangs-x mountain-gale glacial-lance
    scorching-sands lava-plume-x pollen-puff-x apple-acid-x grav-apple spirit-shackle-x anchor-shot-x spirit-break-x strange-steam
    springtide-storm wildbolt-storm sandsear-storm bleakwind-storm bitter-malice-x infernal-parade-x ceaseless-edge-x
    triple-arrows-x mystical-power-x barb-barrage-x esper-wing-x shelter-x power-shift-x stone-axe-x headlong-rush-x`);
  const ZONE = set(`fire-spin whirlpool sand-tomb magma-storm infestation snap-trap thunder-cage leech-seed-x toxic-spikes
    sticky-web wrap-x bind-x clamp-x g-max-x`);
  const SPREAD = set(`bullet-seed rock-blast icicle-spear pin-missile spike-cannon water-shuriken scale-shot barrage fury-attack
    fury-swipes double-kick bonemerang double-hit dual-chop dual-wingbeat tail-slap arm-thrust triple-axel surging-strikes population-bomb
    triple-kick twineedle comet-punch`);
  const HOMING = set(`aura-sphere swift magical-leaf shock-wave magnet-bomb feint-attack aerial-ace vital-throw shadow-punch smart-strike
    clear-smog disarming-voice-x`);
  const PROJ_PHYS = set(`rock-throw ice-shard-x seed-bomb bone-rush bone-club bonemerang-x pyro-ball poison-jab-x gunk-shot
    icicle-crash-x bullet-punch-x needle-arm-x mud-slap mud-shot egg-bomb shadow-bone spirit-shackle anchor-shot pollen-puff
    stone-axe-x ceaseless-edge-x barb-barrage triple-arrows kowtow-cleave-x`);

  // Secondary effects: { st: status, p: chance } or { stage: {stat: delta}, p, self }.
  const SEC = {
    flamethrower: { st: "burn", p: 0.15 }, "fire-blast": { st: "burn", p: 0.2 }, ember: { st: "burn", p: 0.15 }, "fire-punch": { st: "burn", p: 0.15 }, "sacred-fire": { st: "burn", p: 0.5 }, "lava-plume": { st: "burn", p: 0.3 }, "heat-wave": { st: "burn", p: 0.15 }, scald: { st: "burn", p: 0.3 }, "blue-flare": { st: "burn", p: 0.2 }, "fire-fang": { st: "burn", p: 0.15 }, "blaze-kick": { st: "burn", p: 0.15 }, "flare-blitz": { st: "burn", p: 0.15 }, "searing-shot": { st: "burn", p: 0.3 }, inferno: { st: "burn", p: 0.9 }, "will-o-wisp": { st: "burn", p: 1 }, "pyro-ball": { st: "burn", p: 0.15 }, "burning-jealousy": { st: "burn", p: 0.3 },
    thunderbolt: { st: "para", p: 0.15 }, thunder: { st: "para", p: 0.3 }, "thunder-shock": { st: "para", p: 0.15 }, "thunder-punch": { st: "para", p: 0.15 }, spark: { st: "para", p: 0.3 }, discharge: { st: "para", p: 0.3 }, "body-slam": { st: "para", p: 0.3 }, "thunder-fang": { st: "para", p: 0.15 }, nuzzle: { st: "para", p: 1 }, "thunder-wave": { st: "para", p: 1 }, "stun-spore": { st: "para", p: 1 }, glare: { st: "para", p: 1 }, "zap-cannon": { st: "para", p: 1 }, "volt-tackle": { st: "para", p: 0.15 }, "bolt-strike": { st: "para", p: 0.2 }, "force-palm": { st: "para", p: 0.3 }, lick: { st: "para", p: 0.3 }, "dragon-breath": { st: "para", p: 0.3 },
    "ice-beam": { st: "frz", p: 0.12 }, blizzard: { st: "frz", p: 0.12 }, "ice-punch": { st: "frz", p: 0.12 }, "ice-fang": { st: "frz", p: 0.12 }, "freeze-dry": { st: "frz", p: 0.12 }, "powder-snow": { st: "frz", p: 0.12 }, "frost-breath": { st: "frz", p: 0.0 }, "freezing-glare": { st: "frz", p: 0.12 },
    "sludge-bomb": { st: "psn", p: 0.3 }, sludge: { st: "psn", p: 0.3 }, "poison-jab": { st: "psn", p: 0.3 }, "poison-sting": { st: "psn", p: 0.3 }, "gunk-shot": { st: "psn", p: 0.3 }, "sludge-wave": { st: "psn", p: 0.15 }, "poison-fang": { st: "psn", p: 0.5 }, "cross-poison": { st: "psn", p: 0.15 }, toxic: { st: "psn", p: 1 }, "poison-powder": { st: "psn", p: 1 }, "toxic-spikes": { st: "psn", p: 1 }, smog: { st: "psn", p: 0.4 }, "shell-side-arm": { st: "psn", p: 0.2 }, "mortal-spin": { st: "psn", p: 1 }, "barb-barrage": { st: "psn", p: 0.5 }, "malignant-chain": { st: "psn", p: 0.5 },
    "sleep-powder": { st: "slp", p: 1 }, spore: { st: "slp", p: 1 }, hypnosis: { st: "slp", p: 1 }, sing: { st: "slp", p: 1 }, "lovely-kiss": { st: "slp", p: 1 }, "dark-void": { st: "slp", p: 1 }, "relic-song": { st: "slp", p: 0.15 }, yawn: { st: "slp", p: 1 },
    "confuse-ray": { st: "conf", p: 1 }, "sweet-kiss": { st: "conf", p: 1 }, supersonic: { st: "conf", p: 1 }, "dynamic-punch": { st: "conf", p: 1 }, "dizzy-punch": { st: "conf", p: 0.3 }, "water-pulse": { st: "conf", p: 0.3 }, "signal-beam": { st: "conf", p: 0.15 }, "rock-climb": { st: "conf", p: 0.3 }, "hurricane": { st: "conf", p: 0.3 }, "psybeam": { st: "conf", p: 0.15 }, chatter: { st: "conf", p: 1 }, "strange-steam": { st: "conf", p: 0.3 },
    crunch: { stage: { def: -1 }, p: 0.25 }, "shadow-ball": { stage: { spd: -1 }, p: 0.25 }, "psychic": { stage: { spd: -1 }, p: 0.15 }, "flash-cannon": { stage: { spd: -1 }, p: 0.15 }, "energy-ball": { stage: { spd: -1 }, p: 0.15 }, "earth-power": { stage: { spd: -1 }, p: 0.15 }, "focus-blast": { stage: { spd: -1 }, p: 0.15 }, "bug-buzz": { stage: { spd: -1 }, p: 0.15 }, "acid-spray": { stage: { spd: -2 }, p: 1 }, "iron-tail": { stage: { def: -1 }, p: 0.3 }, "rock-smash": { stage: { def: -1 }, p: 0.5 }, "crush-claw": { stage: { def: -1 }, p: 0.5 }, "breaking-swipe": { stage: { atk: -1 }, p: 1 }, "lunge": { stage: { atk: -1 }, p: 1 }, "play-rough": { stage: { atk: -1 }, p: 0.15 }, "icy-wind": { stage: { spe: -1 }, p: 1 }, "bulldoze": { stage: { spe: -1 }, p: 1 }, "mud-shot": { stage: { spe: -1 }, p: 1 }, "electroweb": { stage: { spe: -1 }, p: 1 }, "rock-tomb": { stage: { spe: -1 }, p: 1 }, "low-sweep": { stage: { spe: -1 }, p: 1 }, "mystical-fire": { stage: { spa: -1 }, p: 1 }, "moonblast": { stage: { spa: -1 }, p: 0.3 }, snarl: { stage: { spa: -1 }, p: 1 }, "struggle-bug": { stage: { spa: -1 }, p: 1 }, "skitter-smack": { stage: { spa: -1 }, p: 1 }, "spirit-break": { stage: { spa: -1 }, p: 1 }, "apple-acid": { stage: { spd: -1 }, p: 1 }, "seed-flare": { stage: { spd: -2 }, p: 0.4 }, "luster-purge": { stage: { spd: -1 }, p: 0.5 }, "aurora-beam": { stage: { atk: -1 }, p: 0.15 }, "trop-kick": { stage: { atk: -1 }, p: 1 }, "fire-lash": { stage: { def: -1 }, p: 1 }, "liquidation": { stage: { def: -1 }, p: 0.2 }, "thunderous-kick": { stage: { def: -1 }, p: 1 }, "grav-apple": { stage: { def: -1 }, p: 1 }, "bubble-beam": { stage: { spe: -1 }, p: 0.15 }, "octazooka": { stage: { acc: -1 }, p: 0.5 },
    // self stat changes (drawbacks / boosts)
    overheat: { stage: { spa: -2 }, p: 1, self: true }, "draco-meteor": { stage: { spa: -2 }, p: 1, self: true }, "leaf-storm": { stage: { spa: -2 }, p: 1, self: true }, "psycho-boost": { stage: { spa: -2 }, p: 1, self: true }, "fleur-cannon": { stage: { spa: -2 }, p: 1, self: true }, superpower: { stage: { atk: -1, def: -1 }, p: 1, self: true }, "close-combat": { stage: { def: -1, spd: -1 }, p: 1, self: true }, "hammer-arm": { stage: { spe: -1 }, p: 1, self: true }, "v-create": { stage: { def: -1, spd: -1, spe: -1 }, p: 1, self: true }, "flame-charge": { stage: { spe: 1 }, p: 1, self: true }, "power-up-punch": { stage: { atk: 1 }, p: 1, self: true }, "meteor-mash": { stage: { atk: 1 }, p: 0.2, self: true }, "charge-beam": { stage: { spa: 1 }, p: 0.7, self: true }, "ancient-power": { stage: { atk: 1, def: 1, spa: 1, spd: 1, spe: 1 }, p: 0.1, self: true }, "silver-wind": { stage: { atk: 1, def: 1, spa: 1, spd: 1, spe: 1 }, p: 0.1, self: true }, "fiery-dance": { stage: { spa: 1 }, p: 0.5, self: true }, "scale-shot": { stage: { spe: 1, def: -1 }, p: 1, self: true }, "rapid-spin": { stage: { spe: 1 }, p: 1, self: true }, "aura-wheel": { stage: { spe: 1 }, p: 1, self: true }, "torch-song": { stage: { spa: 1 }, p: 1, self: true }, "armor-cannon": { stage: { def: -1, spd: -1 }, p: 1, self: true }, "make-it-rain": { stage: { spa: -1 }, p: 1, self: true },
  };
  // Status moves.
  const BUFF = {
    "swords-dance": { atk: 2 }, "dragon-dance": { atk: 1, spe: 1 }, "nasty-plot": { spa: 2 }, "calm-mind": { spa: 1, spd: 1 }, "bulk-up": { atk: 1, def: 1 },
    agility: { spe: 2 }, "iron-defense": { def: 2 }, amnesia: { spd: 2 }, "quiver-dance": { spa: 1, spd: 1, spe: 1 }, "shell-smash": { atk: 2, spa: 2, spe: 2, def: -1, spd: -1 },
    coil: { atk: 1, def: 1 }, growth: { atk: 1, spa: 1 }, "work-up": { atk: 1, spa: 1 }, howl: { atk: 1 }, harden: { def: 1 }, withdraw: { def: 1 }, barrier: { def: 2 },
    "cotton-guard": { def: 3 }, "acid-armor": { def: 2 }, "belly-drum": { atk: 6, hpCost: 0.5 }, "tail-glow": { spa: 3 }, "cosmic-power": { def: 1, spd: 1 },
    "shift-gear": { spe: 2, atk: 1 }, autotomize: { spe: 2 }, "rock-polish": { spe: 2 }, charge: { spd: 1 }, curse: { atk: 1, def: 1, spe: -1 }, stockpile: { def: 1, spd: 1 },
    "victory-dance": { atk: 1, def: 1, spe: 1 }, "clangorous-soul": { atk: 1, def: 1, spa: 1, spd: 1, spe: 1, hpCost: 0.33 }, "no-retreat": { atk: 1, def: 1, spa: 1, spd: 1, spe: 1 },
    "tidy-up": { atk: 1, spe: 1 }, "defend-order": { def: 1, spd: 1 }, "focus-energy": { crit: 1 }, "double-team": { evade: 1 }, minimize: { evade: 1 },
    "light-screen": { screenS: 1 }, reflect: { screenP: 1 },
  };
  const HEAL = { recover: 0.4, roost: 0.4, "soft-boiled": 0.4, "slack-off": 0.4, "milk-drink": 0.4, "shore-up": 0.4, synthesis: 0.4, moonlight: 0.4, "morning-sun": 0.4, "heal-order": 0.4, "life-dew": 0.25, "jungle-healing": 0.25, wish: 0.4, rest: 1, "aqua-ring": 0.03, ingrain: 0.03, aromatherapy: 0.1, "heal-bell": 0.1, substitute: 0 };
  const SHIELD = set("protect detect");
  const WEATHER = { sandstorm: "sand", "rain-dance": "rain", "sunny-day": "sun", hail: "hail", snowscape: "hail" };
  const STATUS_PROJ = set("will-o-wisp toxic thunder-wave stun-spore sleep-powder spore hypnosis sing lovely-kiss glare poison-powder confuse-ray sweet-kiss leech-seed yawn nuzzle");

  // Signature tweaks (visual flavour and tuning) keyed by slug.
  const SIG = {
    "hydro-pump": { kind: "spiral", r1: 250, dur: 1.1, windup: 0.35 }, "water-spout": { kind: "spiral", r1: 230, dur: 1.0 },
    "body-slam": { kind: "leap", r: 120, h: 230, dur: 0.62 }, "heavy-slam": { kind: "leap", r: 130, h: 260, dur: 0.7 }, "bounce": { kind: "leap", r: 110, h: 300, dur: 0.75 }, "fly": { kind: "leap", r: 110, h: 320, dur: 0.75 }, "high-jump-kick": { kind: "leap", r: 90, h: 200, dur: 0.5 }, "stomping-tantrum": { kind: "leap", r: 140, h: 140, dur: 0.45 }, "sky-drop": { kind: "leap", r: 100, h: 300, dur: 0.7 },
    "stone-edge": { kind: "erupt", n: 7, gap: 46 }, "rock-tomb": { kind: "erupt", n: 5, gap: 50, big: true }, "shadow-claw": { kind: "erupt", n: 6, gap: 40 }, "icicle-crash": { kind: "erupt", n: 6, gap: 44 }, "spirit-shackle": { kind: "erupt", n: 6, gap: 44 }, "thousand-arrows": { kind: "erupt", n: 8, gap: 42 }, "diamond-storm": { kind: "erupt", n: 7, gap: 44 }, "headlong-rush": { kind: "erupt", n: 6, gap: 46 }, "ice-spinner": { kind: "erupt", n: 5, gap: 44 },
    "solar-beam": { kind: "strike", sky: "pillar", r: 90, delay: 0.8, windup: 0.5 }, "solar-blade": { kind: "strike", sky: "pillar", r: 80, delay: 0.6 },
    smokescreen: { kind: "smoke" }, teleport: { kind: "teleport" }, "double-team": { kind: "buff" }, flamethrower: { kind: "beam", cone: true, len: 400, w: 40 }, thunderbolt: { kind: "beam", jag: true, len: 420, w: 22 },
    "hyper-beam": { kind: "beam", len: 600, w: 50, windup: 0.6, recover: 0.9, straight: true }, "ice-beam": { kind: "beam", len: 440, w: 20, straight: true, helix: true },
    psybeam: { kind: "beam", helix: true, len: 400, w: 22 }, "dragon-breath": { kind: "beam", cone: true, len: 320, w: 36 }, "bubble-beam": { kind: "beam", len: 360, w: 28 },
    "steel-beam": { kind: "beam", len: 520, w: 40, straight: true, windup: 0.5 }, thunder: { kind: "strike", sky: "bolt", r: 90, delay: 0.55 }, "hurricane": { kind: "strike", sky: "wind", r: 120, delay: 0.5 },
    "shadow-ball": { kind: "proj", r: 24, speed: 430, size: 1.8, orb: "shadow", life: 1.3, homing: true, turn: 1.3 }, "aura-sphere": { kind: "proj", r: 18, speed: 460, homing: true, size: 1.3 }, "focus-blast": { kind: "proj", r: 24, speed: 420, size: 1.8 },
    "energy-ball": { kind: "proj", r: 18, speed: 470, size: 1.3 }, "sludge-bomb": { kind: "proj", r: 18, speed: 430, arc: true, size: 1.3 }, moonblast: { kind: "proj", r: 24, speed: 430, size: 1.8 },
    "dragon-pulse": { kind: "proj", r: 20, speed: 520, size: 1.5 }, "dark-pulse": { kind: "aoe", r1: 190, dur: 0.45 }, "flash-cannon": { kind: "beam", len: 420, w: 24, straight: true },
    ember: { kind: "proj", r: 12, speed: 520 }, "water-gun": { kind: "proj", r: 12, speed: 560 }, "thunder-shock": { kind: "proj", r: 12, speed: 600 },
    earthquake: { kind: "aoe", r1: 300, dur: 0.6, shake: 18, cracks: 6 }, bulldoze: { kind: "aoe", r1: 220, dur: 0.5, shake: 10, cracks: 3 }, surf: { kind: "aoe", r1: 280, dur: 1.9, wave: true },
    discharge: { kind: "aoe", r1: 240, dur: 0.5 }, blizzard: { kind: "aoe", r1: 260, dur: 0.8 }, "heat-wave": { kind: "aoe", r1: 260, dur: 0.7 },
    psystrike: { kind: "psybreak", w: 74, tele: 0.55, dur: 0.5, windup: 0.18, recover: 0.4, cd: 5.5, range: 640, hold: 1.05 },
    "expanding-force": { kind: "uplift", r: 150, tele: 0.55, windup: 0.25, recover: 0.35, cd: 7, range: 380, hold: 1.55 },
    dig: { kind: "dig", r: 70, windup: 0.18, recover: 0.3, cd: 5, range: 420, hold: 1.75 }, dive: { kind: "dig", r: 70, windup: 0.18, recover: 0.3, cd: 5, range: 420, hold: 1.75 },
    psychic: { kind: "grab", speed: 1150, range: 430, lift: 78, holdT: 0.55, windup: 0.2, recover: 0.25, cd: 4.5, hold: 1.0 }, "earth-power": { kind: "strike", r: 100, delay: 0.45, sky: "ground" },
    "rock-slide": { kind: "strike", r: 130, delay: 0.6, sky: "rocks", n: 7 }, "draco-meteor": { kind: "strike", r: 150, delay: 0.7, sky: "meteor", n: 5 },
    "fire-blast": { kind: "strike", r: 130, delay: 0.5, sky: "daimonji" }, "fire-spin": { kind: "zone", r: 80, dur: 3.5 }, whirlpool: { kind: "spiral", r1: 250, dur: 2.6, pull: 190 }, "sand-tomb": { kind: "zone", r: 85, dur: 3.5 },
    "quick-attack": { kind: "dash", dist: 300, dur: 0.18 }, "extreme-speed": { kind: "dash", dist: 360, dur: 0.15 }, "flare-blitz": { kind: "dash", dist: 320, dur: 0.3, recoil: 0.25 }, "volt-tackle": { kind: "dash", dist: 320, dur: 0.3, recoil: 0.25 },
    "brave-bird": { kind: "dash", dist: 340, dur: 0.28, recoil: 0.25 }, "double-edge": { kind: "dash", dist: 280, dur: 0.28, recoil: 0.25 }, "head-smash": { kind: "dash", dist: 280, dur: 0.3, recoil: 0.4 }, "wild-charge": { kind: "dash", dist: 300, dur: 0.28, recoil: 0.2 },
    "take-down": { kind: "dash", dist: 260, dur: 0.28, recoil: 0.2 }, "wave-crash": { kind: "dash", dist: 320, dur: 0.3, recoil: 0.25 }, "giga-impact": { kind: "dash", dist: 300, dur: 0.3, recover: 0.9 },
    "dragon-rush": { kind: "dash", dist: 300, dur: 0.3 }, "aqua-jet": { kind: "dash", dist: 300, dur: 0.16 }, "u-turn": { kind: "dash", dist: 260, dur: 0.25 },
    "close-combat": { kind: "melee", hits: 3, arc: 1.4, r: 80 }, "dragon-claw": { kind: "melee", r: 85, arc: 1.5 }, slash: { kind: "melee", r: 85, arc: 1.6 }, "night-slash": { kind: "melee", r: 85, arc: 1.6 },
    "leaf-blade": { kind: "melee", r: 90, arc: 1.6 }, "psycho-cut": { kind: "proj", r: 14, speed: 600, size: 1.2 }, "air-slash": { kind: "proj", r: 16, speed: 600, size: 1.2 }, "razor-leaf": { kind: "spread", n: 3, spread: 0.35, speed: 520 },
    "water-shuriken": { kind: "spread", n: 3, spread: 0.3, speed: 640 }, "icicle-spear": { kind: "spread", n: 4, spread: 0.25, speed: 560 }, "bullet-seed": { kind: "spread", n: 4, spread: 0.25, speed: 580 }, "rock-blast": { kind: "spread", n: 4, spread: 0.3, speed: 500 },
    "pin-missile": { kind: "spread", n: 4, spread: 0.3, speed: 600 }, swift: { kind: "spread", n: 3, spread: 0.4, speed: 520, homing: true }, "magical-leaf": { kind: "spread", n: 3, spread: 0.5, speed: 460, homing: true },
    "power-up-punch": { kind: "melee" }, "drain-punch": { kind: "melee", drain: 0.5 }, "giga-drain": { kind: "proj", drain: 0.5, speed: 440, r: 16 }, "draining-kiss": { kind: "proj", drain: 0.75, speed: 440, r: 16 }, "horn-leech": { kind: "melee", drain: 0.5 }, "leech-life": { kind: "melee", drain: 0.5 }, "oblivion-wing": { kind: "beam", drain: 0.75, len: 500, w: 34 }, "parabolic-charge": { kind: "aoe", drain: 0.5 }, "bitter-blade": { kind: "melee", drain: 0.5 }, "matcha-gotcha": { kind: "beam", drain: 0.5 }, "mega-drain": { kind: "proj", drain: 0.5, speed: 440 }, absorb: { kind: "proj", drain: 0.5, speed: 440 }, "dream-eater": { kind: "proj", drain: 0.5 },
    protect: { cd: 9, dur: 1.6 }, detect: { cd: 9, dur: 1.6 }, substitute: { cd: 12 }, rest: { cd: 16 },
    "iron-head": { kind: "dash", dist: 220, dur: 0.22 }, "iron-tail": { kind: "melee", r: 95, arc: 1.8, knock: 1.6 }, "dragon-tail": { kind: "melee", r: 95, arc: 1.8, knock: 2.4 }, "circle-throw": { kind: "melee", r: 70, knock: 2.4 }, "play-rough": { kind: "melee", hits: 2 }, "fury-swipes": { kind: "melee", hits: 4 }, "fury-attack": { kind: "melee", hits: 3 }, "double-kick": { kind: "melee", hits: 2 }, "dual-chop": { kind: "melee", hits: 2 }, "dual-wingbeat": { kind: "melee", hits: 2 }, "tail-slap": { kind: "melee", hits: 4 }, "arm-thrust": { kind: "melee", hits: 3 }, "triple-axel": { kind: "melee", hits: 3 }, "surging-strikes": { kind: "melee", hits: 3 }, "population-bomb": { kind: "melee", hits: 5 }, "bone-rush": { kind: "melee", hits: 3 }, "double-hit": { kind: "melee", hits: 2 }, "scale-shot": { kind: "spread", n: 4, spread: 0.3 },
    "leech-seed": { kind: "status", st: "seed" }, "rapid-spin": { kind: "aoe", r1: 110, dur: 0.3 }, "petal-dance": { kind: "aoe", r1: 160, dur: 0.6 }, "spin-out": { kind: "aoe", r1: 160, dur: 0.4 },
  };

  // Moves the generated catalog lacks.
  const EXTRA = { dig: { n: "Dig", jp: "あなをほる", t: "ground", c: "p", p: 80, a: 100, pr: 0 }, dive: { n: "Dive", jp: "ダイビング", t: "water", c: "p", p: 80, a: 100, pr: 0 },
    whirlpool: { n: "Whirlpool", jp: "うずしお", t: "water", c: "s", p: 35, a: 100, pr: 0 } }; // not in the generated move data

  const MULTI_NAMES = /(punch|kick|claw|slash|bite|fang|tail|chop|horn|headbutt|stomp|slam|jab|cut|blade|strike|smash|hammer|press|crunch|lick|scratch|peck|wing|tackle|cleave|edge|rush|drill|lash|axe|whip|thrust|knuckle|crash|bash|dive|trample|rend|chomp|fist|kowtow|pound|hit|throw)/;

  function spec(slug) {
    if (cache.has(slug)) return cache.get(slug);
    const m = M[slug] || EXTRA[slug] || { n: slug, jp: slug, t: "normal", c: "p", p: 40, a: 100, pr: 0 };
    const sig = SIG[slug] || {};
    const p = m.p || 0;
    let s = { slug, name: m.n, jp: m.jp, type: m.t, cat: m.c, power: p, acc: m.a || 100, pri: m.pr || 0, sec: SEC[slug] || null };
    if (m.c === "s" && !sig.kind) { // status
      if (BUFF[slug]) s.kind = "buff";
      else if (HEAL[slug] !== undefined) s.kind = "heal";
      else if (SHIELD.has(slug)) s.kind = "shield";
      else if (WEATHER[slug]) s.kind = "weather";
      else if (STATUS_PROJ.has(slug)) s.kind = "status";
      else if (slug === "smokescreen") s.kind = "smoke";
      else if (slug === "teleport") s.kind = "teleport";
      else s.kind = "buff";
    } else if (!sig.kind) {
      if (BEAM.has(slug)) s.kind = "beam";
      else if (DASH.has(slug)) s.kind = "dash";
      else if (AOE.has(slug)) s.kind = "aoe";
      else if (STRIKE.has(slug)) s.kind = "strike";
      else if (ZONE.has(slug)) s.kind = "zone";
      else if (SPREAD.has(slug)) s.kind = "spread";
      else if (m.c === "p") s.kind = PROJ_PHYS.has(slug) || !MULTI_NAMES.test(slug) && p >= 60 && /(bomb|shot|throw|ball|rock|shuriken|missile|arrow|spear|bullet)/.test(slug) ? "proj" : "melee";
      else s.kind = p >= 110 ? "beam" : "proj";
    }
    Object.assign(s, sig);
    if (s.kind === "melee" && s.hits && !sig.hits) s.hits = sig.hits;
    // Economy: cooldown & energy from power; timing from category.
    const heavy = Math.max(0, p - 80) / 80;
    const base = {
      cd: s.kind === "smoke" ? 12 : s.kind === "teleport" ? 7 : s.kind === "buff" ? 11 : s.kind === "heal" ? 14 : s.kind === "shield" ? 9 : s.kind === "weather" ? 20 : s.kind === "status" ? 7 : 1.2 + p / 45,
      cost: s.kind === "smoke" || s.kind === "teleport" ? 18 : s.kind === "buff" || s.kind === "heal" || s.kind === "weather" ? 30 : s.kind === "shield" ? 20 : s.kind === "status" ? 22 : Math.round(U.clamp(10 + p * 0.28, 10, 48)),
      windup: s.kind === "melee" || s.kind === "dash" ? 0.14 + heavy * 0.15 : s.kind === "buff" || s.kind === "heal" || s.kind === "shield" || s.kind === "weather" || s.kind === "status" ? 0.25 : 0.24 + heavy * 0.25,
      recover: s.kind === "beam" ? 0.3 : s.kind === "melee" ? 0.18 : 0.22,
      range: s.kind === "spiral" ? 220 : s.kind === "leap" ? 380 : s.kind === "erupt" ? 340 : s.kind === "melee" ? 80 : s.kind === "dash" ? 260 : s.kind === "aoe" ? 180 : s.kind === "zone" || s.kind === "strike" ? 320 : s.kind === "beam" ? 380 : 420,
      r: s.kind === "melee" ? 78 : 14, arc: 1.3, speed: 520, life: 0.95, len: 420, w: 28, dur: s.kind === "beam" ? 0.55 : s.kind === "aoe" ? 0.5 : s.kind === "zone" ? 3 : 0.3,
      r1: 200, delay: 0.5, n: 1, dist: 280, knock: 1, hits: s.kind === "spread" ? 3 : 1,
    };
    for (const k in base) if (s[k] === undefined) s[k] = base[k];
    if (s.kind === "spread") { s.hits = s.n || s.hits; }
    if (s.kind === "melee" && !s.r) s.r = 78;
    if (s.kind === "strike") s.range = 340;
    if (sig.range) s.range = sig.range;
    if (s.kind === "proj" || s.kind === "spread") s.range = s.speed * s.life * 0.85;
    if (s.kind === "erupt") s.range = (s.n || 6) * (s.gap || 44);
    s.lock = s.kind !== "melee" && s.kind !== "dash" && s.kind !== "buff" && s.kind !== "teleport";
    cache.set(slug, s);
    return s;
  }

  // ------------------------------------------------------------------------------------------------
  // Casting (called when the wind-up finishes). `aim` is an angle; `tgt` the soft-locked enemy or null.
  // Attack, special attack and body size change how wide, long and heavy a move looks.
  // The cached spec stays untouched. Damage still uses the move's own power.
  function visSpec(f, s) {
    const might = s.cat === "s" ? (f.mightSpa || 1) : (f.mightAtk || 1);
    const reach = f.reach || 1;
    const vis = U.clamp(might * 0.55 + reach * 0.5, 0.72, 1.65);
    const quick = f.stats ? Data.FX.quick(f) : 1, area = f.stats ? Data.FX.area(f) : 1;
    return Object.assign({}, s, {
      r: Math.max(6, s.r * vis),
      w: Math.max(8, (s.w || 28) * vis),
      range: s.range * U.clamp(0.75 + reach * 0.35, 0.85, 1.35),
      knock: (s.knock || 1) * (f.push || 1),
      len: (s.len || 420) * U.clamp(reach, 0.85, 1.3),
      // すばやさ: faster shots and longer dashes; とくこう: wider blasts and longer-lasting zones
      speed: (s.speed || 520) * quick,
      dist: (s.dist || 280) * quick,
      r1: (s.r1 || 200) * (s.cat === "s" ? area : 1),
      dur: s.kind === "zone" ? (s.dur || 3) * area : s.dur,
    });
  }

  function cast(B, f, s0, aim, tgt) {
    const s = visSpec(f, s0);
    const fx = B.fx, ox = f.x + Math.cos(aim) * 14, oy = f.y + Math.sin(aim) * 8, oz = f.height * 0.55;
    const pan = B.pan(f.x);
    switch (s.kind) {
      case "melee": {
        Audio.sfx("swing", pan);
        const th = fx.theme(s.type, f);
        const hits = s.hits || 1;
        for (let h = 0; h < hits; h++) {
          B.after(h * 0.09, () => {
            if (f.ko) return;
            let any = false;
            const a0 = aim + (h % 2 ? 0.5 : -0.5);
            fx.add({ x: f.x + Math.cos(aim) * 36, y: f.y + Math.sin(aim) * 20, z: oz, life: 0.16, size: s.r * 0.95, color: th.c[1], shape: "slash", add: true, rot: a0, arc: s.arc, dir: h % 2 ? -1 : 1 });
            for (const e of B.targetsOf(f)) {
              const d = U.dist(f.x, f.y, e.x, e.y);
              if (d > s.r + e.r || Math.abs(U.angDiff(aim, U.angle(f.x, f.y, e.x, e.y))) > s.arc / 2 + (d < 40 ? 2 : 0)) continue;
              B.hit(e, f, s, { mult: 1, knock: s.knock * 160 * (hits > 1 ? 0.5 : 1), angle: aim, contact: true, last: h === hits - 1 });
              any = true;
            }
            for (const p of B.arena.props) if (p.alive && U.dist(f.x, f.y, p.x, p.y) < s.r + p.r && Math.abs(U.angDiff(aim, U.angle(f.x, f.y, p.x, p.y))) < 1.2) Arenas.damageProp(B.arena, p, s.power, fx);
            if (!any && h === 0) Audio.sfx("swing", pan);
          });
        }
        f.lunge(aim, 120);
        break;
      }
      case "dash": {
        Audio.sfx(s.type === "fire" ? "fire" : s.type === "electric" ? "electric" : "swing", pan);
        B.attacks.push({ kind: "dash", owner: f, t: 0, dur: s.dur, dist: s.dist, ax: Math.cos(aim), ay: Math.sin(aim), spec: s, hit: new Set(), aim });
        f.dash = { t: s.dur, speed: s.dist / s.dur, ax: Math.cos(aim), ay: Math.sin(aim), type: s.type };
        break;
      }
      case "proj": case "spread": {
        Audio.sfx(s.type, pan);
        const n = s.kind === "spread" ? s.n || 3 : 1;
        const lob = s.arc ? 0 : (s.type === "rock" || s.type === "ground" || s.type === "poison" || s.type === "grass" || s.orb === "shadow" ? 28 : 14);
        for (let i = 0; i < n; i++) {
          const off = n > 1 ? (i - (n - 1) / 2) * (s.spread || 0.3) : 0;
          B.after(s.kind === "spread" && !s.spread ? i * 0.08 : 0, () => {
            if (f.ko) return;
            const a = aim + off;
            B.attacks.push({ kind: "proj", x: ox, y: oy, z: oz, z0: oz, life0: s.life, lob, vx: Math.cos(a) * s.speed, vy: Math.sin(a) * s.speed, vz: s.arc ? 220 : 0, r: s.r || 14, life: s.life, owner: f, spec: s, mult: 1, homing: s.homing || B.turnMode ? tgt : null, hit: new Set(), t: 0, rot: U.rand(U.TAU), size: s.size || 1 });
          });
        }
        f.lunge(aim, 30);
        break;
      }
      case "beam": {
        Audio.sfx(s.type, pan);
        f.aim = f.facing = aim;
        B.attacks.push({ kind: "beam", owner: f, aim, t: 0, dur: s.dur, spec: s, tick: 0, nextTick: 0.05, hits: 3, seed: Math.random() * 100, pushed: new Set() });
        break;
      }
      case "aoe": {
        Audio.sfx(s.type, pan);
        B.attacks.push({ kind: "aoe", x: f.x, y: f.y, owner: f, t: 0, dur: s.dur, r0: 20, r1: s.r1 || 200, spec: s, hit: new Set(), decaled: false });
        if (s.shake) fx.shakeCam(s.shake, s.dur + 0.2);
        f.hop(s.type === "ground" || s.type === "rock" ? 60 : 30);
        break;
      }
      case "strike": {
        Audio.sfx("swing", pan);
        const range = s.range;
        let tx, ty;
        if (tgt) { tx = tgt.x + tgt.vx * 0.2; ty = tgt.y + tgt.vy * 0.2; }
        else { tx = f.x + Math.cos(aim) * range * 0.7; ty = f.y + Math.sin(aim) * range * 0.7; }
        const d = U.dist(f.x, f.y, tx, ty);
        if (d > range) { tx = f.x + ((tx - f.x) / d) * range; ty = f.y + ((ty - f.y) / d) * range; }
        const p = Arenas.resolve(B.arena, tx, ty, 10);
        B.attacks.push({ kind: "strike", x: p.x, y: p.y, owner: f, t: 0, delay: s.delay, r: s.r || 100, spec: s, fired: false, n: s.n || 1, sky: s.sky || defaultSky(s.type), sub: [] });
        break;
      }
      case "spiral": {
        Audio.sfx("water", pan); Audio.sfx("water", pan);
        B.attacks.push({ kind: "spiral", owner: f, x: f.x, y: f.y, t: 0, dur: s.dur, spec: s, rot: aim, tick: 0, hitT: new Map() });
        f.castHold = s.dur; f.spinT = s.dur;
        break;
      }
      case "leap": {
        Audio.sfx("swing", pan);
        let tx = tgt ? tgt.x + tgt.vx * 0.3 : f.x + Math.cos(aim) * 260, ty = tgt ? tgt.y + tgt.vy * 0.3 : f.y + Math.sin(aim) * 260;
        const d = U.dist(f.x, f.y, tx, ty); if (d > s.range) { tx = f.x + ((tx - f.x) / d) * s.range; ty = f.y + ((ty - f.y) / d) * s.range; }
        const p = Arenas.resolve(B.arena, tx, ty, f.r);
        B.attacks.push({ kind: "leap", owner: f, x0: f.x, y0: f.y, x: p.x, y: p.y, t: 0, dur: s.dur, spec: s, landed: false, r: s.r });
        f.leap = true; f.squash = 0.7;
        break;
      }
      case "erupt": {
        Audio.sfx(s.type === "rock" || s.type === "ground" ? "rock" : s.type, pan);
        B.attacks.push({ kind: "erupt", owner: f, x: f.x, y: f.y, aim: tgt ? U.angle(f.x, f.y, tgt.x, tgt.y) : aim, t: 0, spec: s, spikes: [], n: s.n || 6, gap: s.gap || 44, hit: new Set() });
        break;
      }
      case "smoke": {
        Audio.sfx("dark", pan);
        B.attacks.push({ kind: "smoke", owner: f, x: f.x, y: f.y, t: 0, dur: 5, spec: s, r: 150 });
        for (let i = 0; i < 26; i++) fx.add({ x: f.x + U.rand(-60, 60), y: f.y + U.rand(-30, 30), z: U.rand(0, 60), vx: U.rand(-90, 90), vy: U.rand(-50, 50), vz: U.rand(10, 60), life: U.rand(1.5, 3), size: U.rand(22, 40), color: U.pick(["#1a1a1e", "#2a2a30", "#101014"]), shape: "smoke", alpha: 0.95, grow: 1.6, drag: 1.2 });
        for (const e of B.targetsOf(f)) if (U.dist(f.x, f.y, e.x, e.y) < 260) { e.accDown = 5; fx.label(e.x, e.y - e.height - 8, "めいちゅう▼", "#ff9b9b", { size: 11 }); }
        break;
      }
      case "teleport": {
        Audio.sfx("psychic", pan);
        const fl = B.arena.field;
        fx.burst("psychic", f.x, f.y - f.height * 0.4, 18, { size: 1.1 }); fx.ring(f.x, f.y, 10, 80, 0.35, "#ff9ad6", 4);
        warp(fx, f, true);
        let best = null, bd = -1;
        for (let i = 0; i < 8; i++) { const c = Arenas.resolve(B.arena, fl.x + U.rand(0.15, 0.85) * fl.w, fl.y + U.rand(0.15, 0.85) * fl.h, f.r); const d = tgt ? U.dist(c.x, c.y, tgt.x, tgt.y) : U.rand(1); if (d > bd) { bd = d; best = c; } }
        f.x = best.x; f.y = best.y; f.invuln = 0.4;
        fx.burst("psychic", f.x, f.y - f.height * 0.4, 18, { size: 1.1 }); fx.ring(f.x, f.y, 80, 10, 0.35, "#ff9ad6", 4);
        warp(fx, f, false);
        break;
      }
      case "psybreak": { // Psystrike: aim laser → screen-wide psychic beam that gouges a trench
        Audio.sfx("psychic", pan); Audio.sfx("charge", pan);
        B.attacks.push({ kind: "psybreak", owner: f, aim, t: 0, spec: s, fired: false, len: s.range, seed: Math.random() * 100 });
        break;
      }
      case "uplift": { // the field itself: a disc of floor tears loose under the target, hangs in the air, crashes down
        Audio.sfx("rumble"); Audio.sfx("psychic", pan);
        let tx = tgt ? tgt.x + tgt.vx * 0.3 : f.x + Math.cos(aim) * 220, ty = tgt ? tgt.y + tgt.vy * 0.3 : f.y + Math.sin(aim) * 220;
        const d = U.dist(f.x, f.y, tx, ty), R = B.turnMode ? 9999 : s.range;
        if (d > R) { tx = f.x + ((tx - f.x) / d) * R; ty = f.y + ((ty - f.y) / d) * R; }
        const p = Arenas.resolve(B.arena, tx, ty, s.r * 0.6);
        B.attacks.push({ kind: "uplift", owner: f, x: p.x, y: p.y, r: s.r, t: 0, spec: s, phase: "tele", caught: [] });
        break;
      }
      case "grab": { // Psychic grab: a telekinetic tether that latches on, lifts the foe and slams it down
        Audio.sfx("psychic", pan);
        B.attacks.push({ kind: "grab", owner: f, x: ox, y: oy, aim, t: 0, spec: s, phase: "shoot", ht: 0, tgt: null });
        f.lunge(aim, 20);
        break;
      }
      case "dig": { // burrow, tunnel under the floor toward the foe (a hump of earth shows where), burst out under it
        const wet = Arenas.terrainAt(B.arena, f.x, f.y) === "water";
        let tx = f.x + Math.cos(aim) * s.range * 0.8, ty = f.y + Math.sin(aim) * s.range * 0.8;
        if (tgt && U.dist(f.x, f.y, tgt.x, tgt.y) < s.range * 1.2) { tx = tgt.x; ty = tgt.y; }
        B.attacks.push({ kind: "dig", owner: f, t: 0, spec: s, phase: "down", tgt: tgt && U.dist(f.x, f.y, tgt.x, tgt.y) < s.range * 1.2 ? tgt : null, tx, ty, wet, seed: Math.random() * 1000 | 0 });
        break;
      }
      case "zone": {
        Audio.sfx(s.type, pan);
        let tx, ty;
        if (tgt) { tx = tgt.x; ty = tgt.y; } else { tx = f.x + Math.cos(aim) * 200; ty = f.y + Math.sin(aim) * 200; }
        const d = U.dist(f.x, f.y, tx, ty);
        if (d > s.range) { tx = f.x + ((tx - f.x) / d) * s.range; ty = f.y + ((ty - f.y) / d) * s.range; }
        B.attacks.push({ kind: "zone", x: tx, y: ty, owner: f, t: 0, dur: s.dur, r: s.r || 80, spec: s, tick: 0 });
        break;
      }
      case "buff": {
        const b = BUFF[s.slug] || { atk: 1 };
        Audio.sfx("buff", pan);
        const th = fx.theme(s.type, f);
        fx.aura(f.x, f.y, th.c[1], 18, f.r + 12);
        fx.ring(f.x, f.y, 10, 90, 0.5, th.c[1], 5);
        fx.light(f.x, f.y, 160, th.c[1], 0.8, 0.5);
        const labels = [];
        for (const k of ["atk", "def", "spa", "spd", "spe"]) if (b[k]) { f.stages[k] = U.clamp(f.stages[k] + b[k], -6, 6); labels.push(`${{ atk: "こうげき", def: "ぼうぎょ", spa: "とくこう", spd: "とくぼう", spe: "すばやさ" }[k]}${b[k] > 0 ? "▲".repeat(Math.min(3, b[k])) : "▼".repeat(Math.min(3, -b[k]))}`); }
        if (b.hpCost) f.hp = Math.max(1, f.hp - f.maxHp * b.hpCost);
        if (b.crit) { f.critBoost = 1; labels.push("きゅうしょ率UP"); }
        if (b.evade) { f.evade = 8; labels.push("回避UP"); }
        if (b.screenS) { f.screenS = 10; labels.push("ひかりのかべ"); }
        if (b.screenP) { f.screenP = 10; labels.push("リフレクター"); }
        labels.forEach((l, i) => fx.label(f.x, f.y - f.height - 10, l, b[Object.keys(b)[0]] < 0 ? "#ff9b9b" : "#9bffb0", { delay: i * 0.25, size: 11 }));
        break;
      }
      case "heal": {
        const amt = HEAL[s.slug] || 0.5;
        Audio.sfx("heal", pan);
        fx.aura(f.x, f.y, "#9bffb0", 16, f.r + 10);
        fx.light(f.x, f.y, 160, "#9bffb0", 0.8, 0.6);
        if (s.slug === "substitute") { f.sub = Math.round(f.maxHp * 0.25); f.hp = Math.max(1, f.hp - f.maxHp * 0.25); fx.label(f.x, f.y - f.height, "みがわり", "#ffffff"); break; }
        if (s.slug === "aqua-ring" || s.slug === "ingrain") { f.regen = 12; fx.label(f.x, f.y - f.height, "回復中…", "#9bffb0"); break; }
        if (s.slug === "aromatherapy" || s.slug === "heal-bell" || s.slug === "jungle-healing" || s.slug === "life-dew") { f.cure(); }
        if (s.slug === "wish") { B.after(1.5, () => f.heal(f.maxHp * 0.5)); fx.label(f.x, f.y - f.height, "ねがいごと…", "#ffe9a0"); break; }
        if (s.slug === "rest") { f.cure(); f.heal(f.maxHp); f.setStatus("slp", 2.5, true); break; }
        f.heal(Math.round(f.maxHp * amt));
        break;
      }
      case "shield": {
        Audio.sfx("shield", pan);
        f.protect = s.dur || 1.6;
        fx.ring(f.x, f.y - f.height * 0.4, 10, f.r + 30, 0.4, "#9ad8ff", 4, { ground: false, h: f.height * 0.4, squash: 1 });
        break;
      }
      case "weather": {
        Audio.sfx(s.type, pan);
        B.setWeather(WEATHER[s.slug], 18, f);
        break;
      }
      case "status": {
        Audio.sfx("status", pan);
        B.attacks.push({ kind: "proj", x: ox, y: oy, z: oz, vx: Math.cos(aim) * 380, vy: Math.sin(aim) * 380, vz: 0, r: 16, life: 1.1, owner: f, spec: s, mult: 0, homing: tgt, hit: new Set(), t: 0, rot: 0, size: 1.2, statusOnly: true });
        break;
      }
    }
  }

  // Teleport flourish: a thin light column, horizontal scan lines and a fading silhouette left behind.
  function warp(fx, f, leaving) {
    fx.add({ x: f.x, y: f.y, z: 0, life: 0.3, size: 26, color: "#c9a8ff", shape: "column", add: true, drag: 0 });
    for (let i = 0; i < 7; i++) fx.add({ x: f.x + U.rand(-30, 30), y: f.y, z: U.rand(4, f.height), vx: (i % 2 ? 1 : -1) * U.rand(260, 420), life: U.rand(0.18, 0.3), size: 3, color: i % 3 ? "#e3d2ff" : "#ffffff", shape: "streak", add: true, drag: 4 });
    if (leaving) fx.add({ x: f.x, y: f.y, z: 0, life: 0.45, size: 1, color: "#b48cff", shape: "ghostimg", add: true, drag: 0, fr: Renderer.frameOf(f), fit: f.fit || 1 });
    fx.light(f.x, f.y, 150, "#b48cff", 0.8, 0.3);
  }

  function defaultSky(type) {
    return { rock: "rocks", ground: "ground", electric: "bolt", fire: "blast", psychic: "psy", ice: "shards", flying: "wind", dragon: "meteor", steel: "spikes", water: "splash", grass: "leaves", poison: "stain", ghost: "psy", dark: "psy", fairy: "psy", bug: "leaves", fighting: "blast", normal: "blast" }[type] || "blast";
  }

  // Closest distance between two 2D segments AB and CD (handles collinear anti-parallel beams cleanly).
  function segSegDist(ax, ay, bx, by, cx, cy, dx, dy) {
    const ux = bx - ax, uy = by - ay, vx = dx - cx, vy = dy - cy, wx = ax - cx, wy = ay - cy;
    const a = ux * ux + uy * uy, b = ux * vx + uy * vy, c = vx * vx + vy * vy, d = ux * wx + uy * wy, e = vx * wx + vy * wy;
    const D = a * c - b * b;
    if (D <= 1e-4 * Math.max(1, a * c) && a > 1e-6) {
      const t0 = ((cx - ax) * ux + (cy - ay) * uy) / a;
      const t1 = ((dx - ax) * ux + (dy - ay) * uy) / a;
      const lo = Math.max(0, Math.min(t0, t1)), hi = Math.min(1, Math.max(t0, t1));
      const sc = lo <= hi ? (lo + hi) * 0.5 : (t0 < 0 && t1 < 0 ? 0 : 1);
      const px = ax + sc * ux, py = ay + sc * uy;
      const tc = c > 1e-6 ? U.clamp(((px - cx) * vx + (py - cy) * vy) / c, 0, 1) : 0;
      const qx = cx + tc * vx, qy = cy + tc * vy;
      return { d: Math.hypot(px - qx, py - qy), sa: sc, sb: tc, x: (px + qx) * 0.5, y: (py + qy) * 0.5 };
    }
    let sc, sN, sD = D, tc, tN, tD = D;
    if (D < 1e-6) { sN = 0; sD = 1; tN = e; tD = c; }
    else {
      sN = b * e - c * d; tN = a * e - b * d;
      if (sN < 0) { sN = 0; tN = e; tD = c; }
      else if (sN > sD) { sN = sD; tN = e + b; tD = c; }
    }
    if (tN < 0) {
      tN = 0;
      if (-d < 0) sN = 0; else if (-d > a) sN = sD; else { sN = -d; sD = a; }
    } else if (tN > tD) {
      tN = tD;
      if ((-d + b) < 0) sN = 0; else if ((-d + b) > a) sN = sD; else { sN = -d + b; sD = a; }
    }
    sc = Math.abs(sN) < 1e-6 ? 0 : sN / sD;
    tc = Math.abs(tN) < 1e-6 ? 0 : tN / tD;
    const px = ax + sc * ux, py = ay + sc * uy, qx = cx + tc * vx, qy = cy + tc * vy;
    return { d: Math.hypot(px - qx, py - qy), sa: sc, sb: tc, x: (px + qx) * 0.5, y: (py + qy) * 0.5 };
  }

  function isLiveBeam(a) {
    return !a.forceDead && a.owner && a.owner.active && !a.owner.ko && ((a.kind === "beam" && a.owner.hitstun <= 0.2) || (a.kind === "psybreak" && a.fired));
  }
  function beamSeg(B, a) {
    const ox = a.owner.x, oy = a.owner.y;
    const L = a.end ? a.end.d : (a.len || a.spec.len || a.spec.range || 460);
    return { ox, oy, ex: ox + Math.cos(a.aim) * L, ey: oy + Math.sin(a.aim) * L, L, w: a.spec.w || 28 };
  }

  // Beam Clash (押し合い・相殺) & Mid-Air Projectile Clash system.
  function resolveClashes(B, dt) {
    const A = B.attacks, fx = B.fx;
    if (!B.clashes) B.clashes = [];
    B.activeClash = null;
    for (const a of A) { a.clashing = false; a.clashEnd = null; }

    // 1. Update active beam-vs-beam struggles
    for (let i = B.clashes.length - 1; i >= 0; i--) {
      const cl = B.clashes[i];
      if (!A.includes(cl.a) || !A.includes(cl.b) || !isLiveBeam(cl.a) || !isLiveBeam(cl.b)) {
        B.clashes.splice(i, 1);
        continue;
      }
      cl.t += dt;
      if (cl.t < 1.25) {
        cl.a.t = Math.max(cl.a.kind === "psybreak" ? (cl.a.ft || 0) + 0.08 : 0.12, cl.a.t - dt * 0.68);
        cl.b.t = Math.max(cl.b.kind === "psybreak" ? (cl.b.ft || 0) + 0.08 : 0.12, cl.b.t - dt * 0.68);
        if (cl.a.owner.cast) cl.a.owner.cast.hold = Math.max(cl.a.owner.cast.hold || 0, cl.a.t + 0.22);
        if (cl.b.owner.cast) cl.b.owner.cast.hold = Math.max(cl.b.owner.cast.hold || 0, cl.b.t + 0.22);
      }
      const fa = (cl.a.spec.power || 80) * (cl.a.owner.mightSpa || 1) * (cl.a.owner.mashT > 0 ? 1.3 : 1) + (cl.a.clashPush || 0);
      const fb = (cl.b.spec.power || 80) * (cl.b.owner.mightSpa || 1) * (cl.b.owner.mashT > 0 ? 1.3 : 1) + (cl.b.clashPush || 0);
      cl.a.clashPush = (cl.a.clashPush || 0) * Math.exp(-4 * dt);
      cl.b.clashPush = (cl.b.clashPush || 0) * Math.exp(-4 * dt);
      cl.pos = U.clamp(cl.pos + ((fa - fb) / Math.max(90, fa + fb)) * 0.65 * dt, 0.12, 0.88);

      const ux = cl.b.owner.x - cl.a.owner.x, uy = cl.b.owner.y - cl.a.owner.y, distAB = Math.hypot(ux, uy) || 1;
      const jit = Math.sin(B.time * 48) * 5;
      const cx = U.lerp(cl.a.owner.x, cl.b.owner.x, cl.pos) + (ux / distAB) * jit;
      const cy = U.lerp(cl.a.owner.y, cl.b.owner.y, cl.pos) + (uy / distAB) * jit * 0.6;
      cl.x = cx; cl.y = cy;

      cl.a.aim = U.angle(cl.a.owner.x, cl.a.owner.y, cx, cy); cl.a.owner.facing = cl.a.owner.aim = cl.a.aim;
      cl.b.aim = U.angle(cl.b.owner.x, cl.b.owner.y, cx, cy); cl.b.owner.facing = cl.b.owner.aim = cl.b.aim;
      cl.a.clashing = true; cl.a.clashEnd = { x: cx, y: cy, d: Math.max(20, U.dist(cl.a.owner.x, cl.a.owner.y, cx, cy)) };
      cl.b.clashing = true; cl.b.clashEnd = { x: cx, y: cy, d: Math.max(20, U.dist(cl.b.owner.x, cl.b.owner.y, cx, cy)) };

      B.activeClash = cl;
      B.hype = Math.min(1, B.hype + dt * 0.55);
      fx.shakeCam(3.6, 0.08);
      cl.sparkT = (cl.sparkT || 0) - dt;
      if (cl.sparkT <= 0) {
        cl.sparkT = 0.055;
        fx.burst(cl.a.spec.type, cx, cy - 18, 4, { size: 1.15, owner: cl.a.owner });
        fx.burst(cl.b.spec.type, cx, cy - 18, 4, { size: 1.15, owner: cl.b.owner });
        Arenas.decal(B.arena, "beamburn", cx + U.rand(-8, 8), cy + U.rand(-5, 5), 18);
        if (Math.random() < 0.38) fx.ring(cx, cy, 8, 56, 0.22, "#fff8b8", 4);
      }

      if (cl.pos >= 0.84 || cl.pos <= 0.16) {
        const win = cl.pos >= 0.84 ? cl.a : cl.b, lose = win === cl.a ? cl.b : cl.a;
        lose.forceDead = true;
        win.clashing = false; win.clashEnd = null;
        if (win.kind === "psybreak") {
          win.t = (win.ft || 0) + (win.spec.dur || 0.55) * 0.35;
          win.clashBroken = true;
        } else {
          win.t = Math.min(win.t, (win.dur || win.spec.dur || 0.6) * 0.42);
        }
        fx.stop(0.075); fx.slowmo(0.32, 0.35); fx.zoomPunch = Math.max(fx.zoomPunch, 0.48);
        fx.shakeCam(14, 0.35); fx.flashScreen(0.3, "#fff8d0");
        fx.ring(cx, cy, 12, 135, 0.42, "#ffd54a", 8); fx.blast(win.spec.type, cx, cy, 1.15, win.owner);
        Arenas.decal(B.arena, "clash", cx, cy, 42);
        fx.label(cx, cy - 36, "CLASH BREAK!!", "#ffd54a", { size: 15 });
        Audio.sfx("hitHeavy", B.pan(cx));
        B.clashes.splice(i, 1);
      } else if (cl.t >= 1.22) {
        cl.a.forceDead = true; cl.b.forceDead = true;
        fx.stop(0.068); fx.zoomPunch = Math.max(fx.zoomPunch, 0.42);
        fx.shakeCam(12, 0.32); fx.flashScreen(0.26, "#ffffff");
        fx.ring(cx, cy, 12, 145, 0.45, "#ffffff", 8); fx.blast("electric", cx, cy, 1.1);
        Arenas.decal(B.arena, "clash", cx, cy, 46);
        for (const u of [cl.a.owner, cl.b.owner]) if (u && !u.ko) { const an = U.angle(cx, cy, u.x, u.y); u.kx += Math.cos(an) * 160; u.ky += Math.sin(an) * 110; }
        Audio.sfx("hitHeavy", B.pan(cx));
        B.clashes.splice(i, 1);
      }
    }

    // 2. Detect new beam-vs-beam clashes
    for (let i = 0; i < A.length; i++) {
      const a = A[i];
      if (!isLiveBeam(a) || a.clashing) continue;
      const sa = beamSeg(B, a);
      for (let j = i + 1; j < A.length; j++) {
        const b = A[j];
        if (!isLiveBeam(b) || b.clashing || a.owner.side === b.owner.side) continue;
        const sb = beamSeg(B, b);
        const ss = segSegDist(sa.ox, sa.oy, sa.ex, sa.ey, sb.ox, sb.oy, sb.ex, sb.ey);
        if (ss.d < sa.w * 0.6 + sb.w * 0.6 + 28 && (ss.sa > 0.03 || ss.sb > 0.03)) {
          const totalD = Math.max(60, U.dist(a.owner.x, a.owner.y, b.owner.x, b.owner.y));
          const pos = U.clamp(U.dist(a.owner.x, a.owner.y, ss.x, ss.y) / totalD, 0.28, 0.72);
          const cl = { a, b, pos, x: ss.x, y: ss.y, t: 0, sparkT: 0 };
          B.clashes.push(cl);
          B.activeClash = cl;
          a.clashing = b.clashing = true;
          a.clashEnd = { x: ss.x, y: ss.y, d: Math.max(20, U.dist(a.owner.x, a.owner.y, ss.x, ss.y)) };
          b.clashEnd = { x: ss.x, y: ss.y, d: Math.max(20, U.dist(b.owner.x, b.owner.y, ss.x, ss.y)) };
          fx.stop(0.065); fx.zoomPunch = Math.max(fx.zoomPunch, 0.38); fx.shakeCam(9, 0.25);
          fx.ring(ss.x, ss.y, 10, 96, 0.35, "#fff6b0", 7);
          Arenas.decal(B.arena, "clash", ss.x, ss.y, 34);
          fx.label(ss.x, ss.y - 34, "BEAM CLASH!!", "#7dd3fc", { size: 15 });
          B.event("clash", { x: ss.x, y: ss.y, kind: "beam" });
          Audio.sfx("superEff", B.pan(ss.x));
          break;
        }
      }
    }

    // 3. Mid-air projectile-vs-projectile & projectile-vs-beam clashes
    for (let i = 0; i < A.length; i++) {
      const a = A[i];
      if (a.kind !== "proj" || a.statusOnly || a.forceDead) continue;
      for (let j = i + 1; j < A.length; j++) {
        const b = A[j];
        if (b.kind !== "proj" || b.statusOnly || b.forceDead || a.owner.side === b.owner.side) continue;
        if (U.dist(a.x, a.y, b.x, b.y) < a.r + b.r + 10) {
          const mx = (a.x + b.x) * 0.5, my = (a.y + b.y) * 0.5, mz = ((a.z || 16) + (b.z || 16)) * 0.5;
          const pa = (a.spec.power || 50) * (a.mult || 1), pb = (b.spec.power || 50) * (b.mult || 1);
          fx.burst(a.spec.type, mx, my, 10, { size: 1.1, owner: a.owner });
          fx.burst(b.spec.type, mx, my, 10, { size: 1.1, owner: b.owner });
          fx.ring(mx, my - mz * 0.5, 6, 48, 0.24, "#ffffff", 4, { ground: false, squash: 0.85 });
          fx.stop(0.038); fx.shakeCam(4, 0.12);
          Audio.sfx("hit", B.pan(mx));
          Arenas.decal(B.arena, "scorch", mx, my, 18);
          if (B.time - (B._lastProjClashT || -9) > 0.35) {
            B._lastProjClashT = B.time;
            fx.label(mx, my - mz - 14, "CLASH!", "#fde68a", { size: 12 });
            B.event("clash", { x: mx, y: my, kind: "proj" });
          }
          if (pa > pb * 1.45 && !b.spec.pierce) { b.forceDead = true; a.mult = (a.mult || 1) * 0.65; }
          else if (pb > pa * 1.45 && !a.spec.pierce) { a.forceDead = true; b.mult = (b.mult || 1) * 0.65; }
          else { a.forceDead = true; b.forceDead = true; }
          break;
        }
      }
      if (a.forceDead) continue;
      for (let j = 0; j < A.length; j++) {
        const b = A[j];
        if (!isLiveBeam(b) || a.owner.side === b.owner.side) continue;
        const sb = beamSeg(B, b);
        if (segDist(sb.ox, sb.oy, sb.ex, sb.ey, a.x, a.y).d < sb.w * 0.55 + a.r) {
          a.forceDead = true;
          b.clashPush = (b.clashPush || 0) - 9;
          fx.burst(a.spec.type, a.x, a.y, 9, { size: 1.0, owner: a.owner });
          fx.ring(a.x, a.y - (a.z || 16) * 0.5, 4, 34, 0.2, "#ffffff", 3, { ground: false });
          Arenas.decal(B.arena, "scorch", a.x, a.y, 14);
          Audio.sfx("hit", B.pan(a.x));
          break;
        }
      }
    }
  }

  // ------------------------------------------------------------------------------------------------
  function update(B, dt) {
    resolveClashes(B, dt);
    const fx = B.fx, A = B.attacks;
    for (let i = A.length - 1; i >= 0; i--) {
      const a = A[i];
      if (a.forceDead) { A[i] = A[A.length - 1]; A.pop(); continue; }
      a.t += dt;
      const s = a.spec, f = a.owner;
      let dead = false;
      if (a.kind === "proj") {
        if (a.homing && !a.homing.ko) {
          const want = U.angle(a.x, a.y, a.homing.x, a.homing.y), cur = Math.atan2(a.vy, a.vx), sp = Math.hypot(a.vx, a.vy);
          const tr = (s.turn || 4.5) * dt, na = cur + U.clamp(U.angDiff(cur, want), -tr, tr);
          a.vx = Math.cos(na) * sp; a.vy = Math.sin(na) * sp;
        }
        a.x += a.vx * dt; a.y += a.vy * dt; a.rot += 9 * dt;
        if (s.arc) { a.vz -= 500 * dt; a.z += a.vz * dt; if (a.z < 10) { a.z = 10; dead = true; } }
        else if (a.life0 > 0) {
          const prog = U.clamp(a.t / a.life0, 0, 1);
          a.z = (a.z0 || 20) * (1 - prog * 0.22) + Math.sin(prog * Math.PI) * (a.lob || 14);
        }
        a.life -= dt;
        fx.trail(s.type, a.x, a.y, a.z, s.kind === "spread" ? 2 : 3, (a.size || 1) * 1.15, a.owner);
        if (s.orb === "shadow") { fx.add({ x: a.x, y: a.y, z: a.z, vx: U.rand(-30, 30), vy: U.rand(-20, 20), vz: U.rand(-10, 30), life: U.rand(0.3, 0.5), size: U.rand(12, 20), color: "#1c1036", shape: "smoke", alpha: 0.85, grow: 1.3, drag: 2 }); fx.light(a.x, a.y, 110, "#6a5cff", 0.5, 0.06); }
        if (Math.random() < 0.5) fx.add({ x: a.x, y: a.y, z: a.z, vx: U.rand(-60, 60), vy: U.rand(-40, 40), vz: U.rand(-30, 60), life: 0.25, size: 3, color: "#ffffff", shape: "pixel", add: true, drag: 3 });
        for (const e of B.targetsOf(f)) {
          if (a.hit.has(e)) continue;
          if (U.dist(a.x, a.y, e.x, e.y) < a.r + e.r) {
            a.hit.add(e);
            if (a.statusOnly) B.applyStatusMove(e, f, s); else B.hit(e, f, s, { mult: a.mult || 1, knock: 110 * s.knock, angle: Math.atan2(a.vy, a.vx), z: a.z });
            if (!s.pierce) dead = true;
          }
        }
        for (const p of B.arena.props) if (p.alive && U.dist(a.x, a.y, p.x, p.y) < a.r + p.r) { Arenas.damageProp(B.arena, p, s.power, fx); dead = true; }
        const fld = B.arena.field;
        if (a.x < fld.x || a.x > fld.x + fld.w || a.y < fld.y || a.y > fld.y + fld.h) dead = true;
        if (a.life <= 0) dead = true;
        if (dead && s.orb === "shadow") { // dark implosion: violet rings closing in, then a black burst
          fx.ring(a.x, a.y, 90, 8, 0.25, "#7a6cff", 6, { ground: false, z: a.z, squash: 0.8 }); fx.ring(a.x, a.y, 10, 140, 0.45, "#3a2a80", 9);
          for (let k = 0; k < 14; k++) { const an = U.rand(U.TAU), sp = U.rand(120, 300); fx.add({ x: a.x, y: a.y, z: a.z, vx: Math.cos(an) * sp, vy: Math.sin(an) * sp * 0.6, vz: U.rand(-40, 120), life: U.rand(0.4, 0.8), size: U.rand(14, 26), color: U.pick(["#1a0f30", "#2a1750", "#120a20"]), shape: "smoke", alpha: 0.9, grow: 1.4, drag: 3 }); }
          Arenas.decal(B.arena, "crater", a.x, a.y, 22);
        }
        if (dead) { fx.burst(s.type, a.x, a.y, a.statusOnly ? 6 : 14 + (a.size || 1) * 6, { size: (a.size || 1) * 1.2, owner: a.owner }); if (!a.statusOnly) fx.impact(s.type, a.x, a.y, a.z, (a.size || 1) > 1.2, undefined, a.owner); const th = fx.theme(s.type, a.owner); if (th.decal && !a.statusOnly && Math.random() < 0.5) Arenas.decal(B.arena, th.decal, a.x, a.y, 26 * (a.size || 1)); }
      } else if (a.kind === "beam") {
        if (f.ko || !f.active || f.hitstun > 0.2) dead = true;
        else {
          // follow the owner's aim with a little lag unless locked in a beam clash
          if (!a.clashing) a.aim += U.clamp(U.angDiff(a.aim, f.aim), -2.2 * dt, 2.2 * dt);
          f.facing = a.aim; f.castHold = 0.1;
          const ox = f.x, oy = f.y, L = s.len;
          const ex = ox + Math.cos(a.aim) * L, ey = oy + Math.sin(a.aim) * L;
          const hitEnd = a.clashEnd ? { x: a.clashEnd.x, y: a.clashEnd.y, d: a.clashEnd.d } : { x: ex, y: ey, d: L };
          // props block
          if (!a.clashing) {
            for (const p of B.arena.props) if (p.alive) { const d = segDist(ox, oy, ex, ey, p.x, p.y); if (d.d < p.r + 6 && d.t * L < hitEnd.d) { hitEnd.d = d.t * L; hitEnd.x = ox + Math.cos(a.aim) * hitEnd.d; hitEnd.y = oy + Math.sin(a.aim) * hitEnd.d; hitEnd.prop = p; } }
          }
          a.end = hitEnd;
          // ice freezes the floor along the beam (water too: a walkable floe); fire/energy scorches it
          a.frz = (a.frz || 0) - dt;
          if (a.frz <= 0) {
            a.frz = 0.055;
            const c = Math.cos(a.aim), sn = Math.sin(a.aim), reach = hitEnd.d * Math.min(1, a.t / 0.18);
            if (s.type === "ice" && reach > 34) {
              Ground.freezeLine(B.arena, ox + c * 26, oy + sn * 26, ox + c * reach, oy + sn * reach, s.w * 1.8, 13);
              if (Math.random() < 0.55) { const d = U.rand(40, reach), o = U.rand(-1, 1) * s.w * 0.7; Ground.addCrystal(B.arena, ox + c * d - sn * o, oy + sn * d + c * o * 0.62, U.rand(4, 10)); }
              if (Math.random() < 0.5) { const d = U.rand(30, reach); fx.add({ x: ox + c * d, y: oy + sn * d, z: 2, vx: U.rand(-20, 20), vy: U.rand(-10, 10), vz: U.rand(20, 50), life: 0.7, size: U.rand(8, 14), color: "#eef9ff", shape: "smoke", alpha: 0.35, grow: 1.6 }); } // cold mist
            } else if (s.type === "fire" && B.arena.iceN) {
              for (let d = 30; d < reach; d += 30) if (Ground.melt(B.arena, ox + c * d, oy + sn * d, s.w * 0.7) && Math.random() < 0.5) fx.add({ x: ox + c * d, y: oy + sn * d, z: 4, vx: U.rand(-20, 20), vz: U.rand(40, 90), life: 1, size: U.rand(12, 20), color: "#f4f8ff", shape: "smoke", alpha: 0.45, grow: 1.8 });
            } else if ((s.type === "fire" || s.type === "electric" || s.type === "dragon" || s.power >= 95) && reach > 50 && Math.random() < 0.5) {
              const d = U.rand(45, reach);
              Arenas.decal(B.arena, "beamburn", ox + c * d, oy + sn * d, Math.max(12, s.w * 0.55));
            }
          }
          a.nextTick -= dt;
          if (a.nextTick <= 0 && !a.clashing) {
            a.nextTick = a.dur / 3; a.tick++;
            for (const e of B.targetsOf(f)) {
              const d = segDist(ox, oy, hitEnd.x, hitEnd.y, e.x, e.y);
              if (d.d < s.w / 2 + e.r) B.hit(e, f, s, { mult: 0.45, knock: 60, angle: a.aim, tickOf: a.tick, noStop: a.tick !== 3 });
            }
            if (hitEnd.prop) Arenas.damageProp(B.arena, hitEnd.prop, s.power / 3, fx);
          }
          // continuous FX along the beam & at the end
          if (!VFX.beamTick(B, a, hitEnd, ox, oy, dt)) {
            for (let k = 0; k < 3; k++) { const d = U.rand(30, Math.max(36, hitEnd.d)); fx.trail(s.type, ox + Math.cos(a.aim) * d, oy + Math.sin(a.aim) * d, f.height * 0.55 + U.rand(-8, 8), 1, 1.2, f); }
            fx.burst(s.type, hitEnd.x, hitEnd.y, 2, { size: 1.2, light: false, dir: a.aim + Math.PI, spread: 1.2, owner: f });
            fx.light(hitEnd.x, hitEnd.y, 120, fx.theme(s.type, f).c[1], s.cone ? 0.3 : 0.45, 0.1);
            fx.light(ox, oy, 90, fx.theme(s.type, f).c[1], 0.3, 0.1);
          }
          if (a.t >= a.dur) { dead = true; const th = fx.theme(s.type, f); if (th.decal) Arenas.decal(B.arena, th.decal, hitEnd.x, hitEnd.y, 40); }
        }
      } else if (a.kind === "dash") {
        if (f.ko || !f.active) dead = true;
        else {
          for (const e of B.targetsOf(f)) if (!a.hit.has(e) && U.dist(f.x, f.y, e.x, e.y) < f.r + e.r + 14) { a.hit.add(e); if (f.dash) f.dash.hitAny = true; B.hit(e, f, s, { mult: 1, knock: 260 * s.knock, angle: a.aim, contact: true }); if (s.recoil) f.recoil(s.recoil); }
          for (const p of B.arena.props) if (p.alive && U.dist(f.x, f.y, p.x, p.y) < f.r + p.r + 6) { Arenas.damageProp(B.arena, p, s.power, fx); f.dash = null; }
          fx.trail(s.type, f.x + U.rand(-10, 10), f.y + U.rand(-6, 6), f.height * 0.5, 2, 1.4, f);
          if (!f.dash || a.t >= a.dur) dead = true;
        }
      } else if (a.kind === "aoe") {
        const k = U.easeOut(Math.min(1, a.t / a.dur));
        const r = s.wave ? VFX.waveHitR(a) : U.lerp(a.r0, a.r1, k); // Surf hits where its first wave has reached
        const th = fx.theme(s.type, f);
        for (const e of B.targetsOf(f)) if (!a.hit.has(e) && U.dist(a.x, a.y, e.x, e.y) < r + e.r) { a.hit.add(e); B.hit(e, f, s, { mult: 1, knock: 240 * s.knock, angle: U.angle(a.x, a.y, e.x, e.y) }); }
        for (const p of B.arena.props) if (p.alive && !a.hit.has(p) && U.dist(a.x, a.y, p.x, p.y) < r + p.r) { a.hit.add(p); Arenas.damageProp(B.arena, p, s.power, fx); }
        if (s.wave) VFX.waveTick(B, a, dt); // Surf: its own wall of water, spray and wash
        else if (a.t === dt || !a.started) { a.started = true; fx.ring(a.x, a.y, a.r0, a.r1 + 20, a.dur, th.c[1], 10); fx.ring(a.x, a.y, a.r0, a.r1, a.dur * 1.2, th.c[0], 3); fx.light(a.x, a.y, a.r1 * 1.2, th.c[1], 0.8, a.dur); }
        if (!s.wave) for (let i = 0; i < 6; i++) { const an = U.rand(U.TAU); fx.add({ x: a.x + Math.cos(an) * r, y: a.y + Math.sin(an) * r * 0.55, z: 2, vx: Math.cos(an) * 60, vy: Math.sin(an) * 30, vz: U.rand(40, 200), grav: th.grav > 0 ? th.grav : 0, life: U.rand(0.3, 0.7), size: U.rand(5, 12), color: U.pick(th.c), shape: th.shape, add: th.add, spin: U.rand(-5, 5) }); }
        if (s.cracks && !a.decaled && a.t > a.dur * 0.3) { a.decaled = true; for (let i = 0; i < s.cracks; i++) { const an = U.rand(U.TAU), d = U.rand(30, a.r1 * 0.8); Arenas.decal(B.arena, "crack", a.x + Math.cos(an) * d, a.y + Math.sin(an) * d * 0.6, 40); } Arenas.decal(B.arena, "crater", a.x, a.y, 50); }
        if (a.t >= a.dur) dead = true;
      } else if (a.kind === "strike") {
        const th = fx.theme(s.type, f);
        if (!a.fired && a.t >= a.delay) {
          a.fired = true;
          Audio.sfx(s.type, B.pan(a.x));
          for (const e of B.targetsOf(f)) if (U.dist(a.x, a.y, e.x, e.y) < a.r + e.r) B.hit(e, f, s, { mult: 1, knock: 200 * s.knock, angle: U.angle(a.x, a.y, e.x, e.y) });
          for (const p of B.arena.props) if (p.alive && U.dist(a.x, a.y, p.x, p.y) < a.r + p.r) Arenas.damageProp(B.arena, p, s.power, fx);
          if (a.sky !== "daimonji") { fx.blast(s.type, a.x, a.y, U.clamp(a.r / 100, 0.8, 1.4), f); fx.ring(a.x, a.y, 10, a.r * 1.3, 0.4, th.c[1], 8); } // だいもんじ has its own blast (VFX.fireBlast)
          fx.shakeCam(8 + s.power / 15, 0.3);
          if (th.decal) Arenas.decal(B.arena, th.decal === "crater" ? "crater" : th.decal, a.x, a.y, a.r * 0.6);
          if (a.sky === "rocks" || a.sky === "meteor") { for (let i = 0; i < a.n; i++) B.after(i * 0.07, () => { fx.burst(s.type, a.x + U.rand(-a.r, a.r) * 0.8, a.y + U.rand(-a.r, a.r) * 0.5, 10, { size: 1.2, owner: f }); fx.shakeCam(5, 0.1); }); }
          if (a.sky === "bolt") { fx.flashScreen(0.35, "#fff7c0"); a.boltSeed = Math.random(); }
          if (a.sky === "pillar") { fx.flashScreen(0.3, "#ffffff"); fx.shakeCam(16, 0.4); fx.ring(a.x, a.y, 10, a.r * 2, 0.5, "#ffffff", 10); fx.light(a.x, a.y, 260, "#fff8d0", 0.5, 0.5); }
          a.doneAt = a.t + (a.sky === "daimonji" ? 0.66 : 0.35); a.firedAt = a.t;
        }
        if (a.sky === "daimonji") VFX.fireBlastTick(B, a, dt);
        if (!a.fired) { // telegraph particles
          if (Math.random() < 0.6) fx.add({ x: a.x + U.rand(-a.r, a.r), y: a.y + U.rand(-a.r, a.r) * 0.6, z: 0, vz: 50, life: 0.4, size: 4, color: th.c[1], shape: "orb", add: true });
          // rocks / meteors visibly falling toward the circle
          if (a.sky === "rocks" || a.sky === "meteor" || a.sky === "shards") {
            if (a.sub.length < a.n * 2 && Math.random() < 0.5) a.sub.push({ x: a.x + U.rand(-a.r, a.r) * 0.7, y: a.y + U.rand(-a.r, a.r) * 0.5, z: 520, t0: a.t, rot: U.rand(U.TAU) });
            for (const sb of a.sub) sb.z = Math.max(0, 520 * (1 - (a.t - sb.t0) / Math.max(0.1, a.delay - sb.t0)));
          }
        }
        if (a.fired && a.t >= a.doneAt) dead = true;
      } else if (a.kind === "spiral") {
        if (f.ko || !f.active) dead = true;
        else {
          a.x = f.x; a.y = f.y; a.rot += dt * 7.5;
          const k = Math.min(1, a.t / 0.25), R = s.r1 * k * (a.t > a.dur - 0.2 ? Math.max(0, (a.dur - a.t) / 0.2) : 1);
          a.R = R;
          if (s.pull) { // Whirlpool: water spirals in to the caster; whoever is caught is hurt and dragged to the centre
            for (const e of B.targetsOf(f)) {
              const d = U.dist(a.x, a.y, e.x, e.y); if (d >= R + e.r || e.ko || e.flies && e.z > 40) continue;
              const an = U.angle(e.x, e.y, a.x, a.y), gap = d - (f.r + e.r + 4);
              if (gap > 0) { // inward, plus a little along the current so the body circles in instead of sliding straight
                const v = Math.min(gap / dt, s.pull * (0.55 + 0.45 * d / Math.max(1, R)));
                e.x += (Math.cos(an) * v + Math.sin(an) * v * 0.35) * dt; e.y += (Math.sin(an) * v - Math.cos(an) * v * 0.35) * dt;
              }
              const last = a.hitT.get(e) || -9;
              if (a.t - last > 0.42) { a.hitT.set(e, a.t); B.hit(e, f, s, { mult: 0.3, knock: 8, angle: an, noStop: true }); }
            }
            VFX.whirlTick(B, a, dt);
            if (a.t >= a.dur) { dead = true; Arenas.decal(B.arena, "puddle", a.x, a.y, s.r1 * 0.7); }
          } else {
          for (const e of B.targetsOf(f)) {
            const d = U.dist(a.x, a.y, e.x, e.y), last = a.hitT.get(e) || -9;
            if (d < R + e.r && a.t - last > 0.22) { a.hitT.set(e, a.t); B.hit(e, f, s, { mult: 0.32, knock: 140, angle: U.angle(a.x, a.y, e.x, e.y) + 0.8, noStop: true }); }
          }
          for (let i = 0; i < 4; i++) { const an = a.rot + i * Math.PI / 2 + R * 0.012, rr = R * U.rand(0.7, 1); fx.add({ x: a.x + Math.cos(an) * rr, y: a.y + Math.sin(an) * rr * 0.6, z: 20, vx: -Math.sin(an) * 200, vy: Math.cos(an) * 120, vz: U.rand(40, 160), grav: 520, life: U.rand(0.3, 0.6), size: U.rand(4, 9), color: U.pick(fx.theme("water").c), shape: "drop" }); }
          fx.light(a.x, a.y, R * 1.4 + 60, "#7fd0ff", 0.55, 0.1);
          if (a.t >= a.dur) { dead = true; Arenas.decal(B.arena, "puddle", a.x, a.y, s.r1 * 0.7); }
          }
        }
      } else if (a.kind === "leap") {
        const k = Math.min(1, a.t / a.dur);
        if (f.ko || !f.active) { dead = true; f.leap = false; f.hop0 = 0; f.z = 0; }
        else if (!a.landed) {
          f.x = U.lerp(a.x0, a.x, U.smooth(k)); f.y = U.lerp(a.y0, a.y, U.smooth(k));
          const h = Math.sin(k * Math.PI) * s.h; f.hop0 = h; f.hopV = 0; f.z = h;
          f.facing = U.angle(a.x0, a.y0, a.x, a.y);
          if (k >= 1) {
            a.landed = true; f.leap = false; f.hop0 = 0; f.z = 0; f.squash = 1.35;
            Audio.sfx("hitHeavy", B.pan(a.x)); Audio.sfx("rock", B.pan(a.x));
            fx.blast(s.type === "normal" ? "ground" : s.type, a.x, a.y, 1.1 * Math.sqrt(f.mass), f); fx.ring(a.x, a.y, 10, a.r * 1.3, 0.45, "#e8d0a0", 6);
            fx.dust(a.x, a.y, 14, "rgba(210,190,150,0.7)");
            Arenas.decal(B.arena, "crater", a.x, a.y, a.r * 0.6);
            for (const e of B.targetsOf(f)) if (U.dist(a.x, a.y, e.x, e.y) < a.r + e.r) B.hit(e, f, s, { mult: 1, knock: 280, angle: U.angle(a.x, a.y, e.x, e.y), contact: true });
            for (const p of B.arena.props) if (p.alive && U.dist(a.x, a.y, p.x, p.y) < a.r + p.r) Arenas.damageProp(B.arena, p, s.power, fx);
            a.endAt = a.t + 0.2;
          }
        } else if (a.t >= a.endAt) dead = true;
      } else if (a.kind === "erupt") {
        const step = 0.06, want = Math.min(a.n, Math.floor(a.t / step) + 1), th = fx.theme(s.type, f);
        while (a.spikes.length < want) {
          const i = a.spikes.length, d = 50 + i * a.gap, x = a.x + Math.cos(a.aim) * d + U.rand(-8, 8), y = a.y + Math.sin(a.aim) * d * 0.9 + U.rand(-6, 6);
          const fl = B.arena.field; if (x < fl.x || x > fl.x + fl.w || y < fl.y || y > fl.y + fl.h) { a.n = i; break; }
          a.spikes.push({ x, y, t: 0, h: (s.big ? 84 : 66) + U.rand(0, 22) });
          fx.burst(s.type, x, y, 6, { size: 0.8, light: false, owner: f }); fx.dust(x, y, 3, "rgba(120,100,80,0.6)");
          fx.light(x, y, 70, th.c[1], 0.25, 0.25);
          if (i % 2 === 0) { Audio.sfx("rock", B.pan(x)); fx.shakeCam(4, 0.1); }
          for (const e of B.targetsOf(f)) if (!a.hit.has(e) && U.dist(x, y, e.x, e.y) < 36 + e.r) { a.hit.add(e); B.hit(e, f, s, { mult: 1, knock: 200, angle: a.aim, z: 30 }); e.hop(70); }
          for (const p of B.arena.props) if (p.alive && U.dist(x, y, p.x, p.y) < 30 + p.r) Arenas.damageProp(B.arena, p, s.power, fx);
          if (Math.random() < 0.5) Arenas.decal(B.arena, s.type === "ghost" ? "fissure" : "crack", x, y, 20);
        }
        for (const sp of a.spikes) sp.t += dt;
        if (a.spikes.length >= a.n && a.t > a.n * step + 0.9) { dead = true; for (const sp of a.spikes) fx.debris(sp.x, sp.y - 10, 4, th.c[2], th.c[1]); }
      } else if (a.kind === "smoke") {
        if (Math.random() < 0.5) fx.add({ x: a.x + U.rand(-a.r, a.r), y: a.y + U.rand(-a.r, a.r) * 0.6, z: U.rand(0, 50), vx: U.rand(-20, 20), vy: U.rand(-10, 10), vz: U.rand(5, 20), life: U.rand(1, 2), size: U.rand(24, 40), color: U.pick(["#1a1a1e", "#26262c"]), shape: "smoke", alpha: 0.9, grow: 1.3, drag: 1 });
        if (a.t >= a.dur) dead = true;
      } else if (a.kind === "psybreak") {
        dead = updatePsybreak(B, a, dt);
      } else if (a.kind === "grab") {
        dead = updateGrab(B, a, dt);
      } else if (a.kind === "uplift") {
        dead = updateUplift(B, a, dt);
      } else if (a.kind === "dig") {
        dead = updateDig(B, a, dt);
      } else if (a.kind === "zone") {
        const th = fx.theme(s.type, f);
        a.tick -= dt;
        if (a.tick <= 0) {
          a.tick = 0.5;
          for (const e of B.targetsOf(f)) if (U.dist(a.x, a.y, e.x, e.y) < a.r + e.r * 0.5) { B.hit(e, f, s, { mult: 0.35, knock: 0, angle: U.angle(a.x, a.y, e.x, e.y), noStop: true }); e.slowT = Math.max(e.slowT, 0.6); }
        }
        for (let i = 0; i < 3; i++) { const an = a.t * 6 + i * 2.1 + U.rand(0.4), rr = a.r * U.rand(0.6, 1); fx.add({ x: a.x + Math.cos(an) * rr, y: a.y + Math.sin(an) * rr * 0.55, z: U.rand(0, 20), vx: -Math.sin(an) * 120, vy: Math.cos(an) * 60, vz: U.rand(40, 140), life: U.rand(0.3, 0.6), size: U.rand(6, 12), color: U.pick(th.c), shape: th.shape, add: th.add, drag: 0.5 }); }
        fx.light(a.x, a.y, a.r * 2, th.c[1], 0.5, 0.1);
        if (a.t >= a.dur) dead = true;
      }
      if (dead) { A[i] = A[A.length - 1]; A.pop(); }
    }
  }

  // Distance from (x, y) along `aim` to the field edge (capped at max).
  function beamLen(B, x, y, aim, max) {
    const fl = B.arena.field, c = Math.cos(aim), sn = Math.sin(aim);
    let L = max;
    if (c > 1e-3) L = Math.min(L, (fl.x + fl.w - x) / c); else if (c < -1e-3) L = Math.min(L, (fl.x - x) / c);
    if (sn > 1e-3) L = Math.min(L, (fl.y + fl.h - y) / sn); else if (sn < -1e-3) L = Math.min(L, (fl.y - y) / sn);
    return Math.max(60, L);
  }

  // Psystrike: `tele` seconds of red aim laser (slowly tracking, so it can be side-stepped), then one huge hit
  // along the whole line, a trench gouged into the floor and a crater where the beam meets the wall.
  function updatePsybreak(B, a, dt) {
    const f = a.owner, s = a.spec, fx = B.fx;
    if (f.ko || !f.active || (f.hitstun > 0 && !a.fired)) return true;
    f.castHold = 0.1;
    const strikeFullLine = () => {
      a.struck = true;
      a.len = beamLen(B, f.x, f.y, a.aim, B.turnMode ? 2000 : s.range);
      const c = Math.cos(a.aim), sn = Math.sin(a.aim), ex = f.x + c * a.len, ey = f.y + sn * a.len;
      for (const e of B.targetsOf(f)) if (segDist(f.x, f.y, ex, ey, e.x, e.y).d < s.w / 2 + e.r) { B.hit(e, f, s, { mult: 1, knock: 460, angle: a.aim }); e.hop(90); }
      for (const p of B.arena.props) if (p.alive && segDist(f.x, f.y, ex, ey, p.x, p.y).d < s.w / 2 + p.r) Arenas.damageProp(B.arena, p, s.power * 1.5, fx);
      Arenas.trench(B.arena, f.x + c * 26, f.y + sn * 26, ex, ey, s.w * 0.8);
      Arenas.decal(B.arena, "crater", ex, ey, 46);
      fx.blast("psychic", ex, ey, 1.3);
    };
    if (!a.fired) {
      const tg = f.lockTarget;
      if (tg && !tg.ko) a.aim += U.clamp(U.angDiff(a.aim, U.angle(f.x, f.y, tg.x, tg.y)), -1.4 * dt, 1.4 * dt);
      f.facing = f.aim = a.aim;
      a.len = beamLen(B, f.x, f.y, a.aim, B.turnMode ? 2000 : s.range);
      for (let k = 0; k < 2; k++) { // energy streaking in toward the caster
        const an = U.rand(U.TAU), r = U.rand(90, 180);
        fx.add({ x: f.x + Math.cos(an) * r, y: f.y + Math.sin(an) * r * 0.6, z: f.height * 0.5 + U.rand(-20, 20), vx: -Math.cos(an) * r * 4, vy: -Math.sin(an) * r * 2.4, life: 0.22, size: 3, color: k ? "#ffffff" : "#ff8ae0", shape: "streak", add: true, drag: 0 });
      }
      if (Math.random() < 0.5) fx.charge("psychic", f.x, f.y - f.height * 0.55, 1, 60, undefined, f.height * 0.55);
      fx.light(f.x, f.y, 110 + a.t * 180, "#ff5fd0", 0.6, 0.06);
      if (a.t < s.tele) return false;
      a.fired = true; a.ft = a.t;
      const c = Math.cos(a.aim), sn = Math.sin(a.aim), ex = f.x + c * a.len, ey = f.y + sn * a.len;
      Audio.sfx("psybeam", B.pan(f.x));
      fx.flashScreen(0.28, "#ff6ad8"); fx.shakeCam(18, 0.55); fx.radialBlur(f.x, f.y, 0.9);
      fx.ring(f.x, f.y, 10, 170, 0.4, "#ffffff", 8); fx.ring(f.x, f.y, 10, 240, 0.6, "#ff5fd0", 10);
      f.lunge(a.aim + Math.PI, 40); // recoil
      // Check if an opposing live beam blocks Psystrike into a Beam Clash instead of piercing straight through!
      let blockedByBeam = false;
      for (const b of B.attacks) {
        if (b === a || !isLiveBeam(b) || b.owner.side === f.side) continue;
        const sb = beamSeg(B, b);
        const ss = segSegDist(f.x, f.y, ex, ey, sb.ox, sb.oy, sb.ex, sb.ey);
        if (ss.d < (s.w || 46) * 0.6 + sb.w * 0.6 + 28 && (ss.sa > 0.03 || ss.sb > 0.03)) {
          blockedByBeam = true;
          a.clashEnd = { x: ss.x, y: ss.y, d: Math.max(24, U.dist(f.x, f.y, ss.x, ss.y)) };
          Arenas.trench(B.arena, f.x + c * 26, f.y + sn * 26, ss.x, ss.y, s.w * 0.72);
          break;
        }
      }
      if (!blockedByBeam) strikeFullLine();
      return false;
    }
    if (a.clashBroken && !a.struck) {
      a.clashBroken = false;
      strikeFullLine();
    }
    const k = Math.max(0, a.t - (a.ft || 0)), c = Math.cos(a.aim), sn = Math.sin(a.aim);
    const curLen = a.clashEnd ? a.clashEnd.d : a.len;
    for (let i = 0; i < 5; i++) { // sparks thrown up out of the beam
      const d = U.rand(30, Math.max(36, curLen)), o = U.rand(-0.5, 0.5) * s.w;
      fx.add({ x: f.x + c * d - sn * o, y: f.y + sn * d + c * o * 0.6, z: U.rand(10, f.height * 0.6), vx: c * U.rand(100, 400), vy: sn * U.rand(100, 400), vz: U.rand(80, 280), grav: 500, life: U.rand(0.25, 0.5), size: U.rand(3, 6), color: U.pick(["#ffffff", "#ffb3ec", "#ff5fd0"]), shape: "pixel", add: true, drag: 2 });
    }
    for (let i = 1; i <= 3; i++) fx.light(f.x + c * curLen * i / 3, f.y + sn * curLen * i / 3, 170, "#ff5fd0", 0.7, 0.06);
    return k >= s.dur;
  }

  // Psychic grab: tether shoots out (homing a little); on contact the foe is lifted, held, then slammed down.
  function updateGrab(B, a, dt) {
    const f = a.owner, s = a.spec, fx = B.fx;
    const release = () => { if (a.tgt) { a.tgt.held = null; a.tgt.hop0 = a.tgt.z; } if (a.slab) Slabs.land(B, a.slab); return true; };
    if (f.ko || !f.active || f.hitstun > 0) return release();
    f.castHold = 0.1;
    if (a.phase === "slam" && a.ht === 0) Motion.start(f, { keys: [[0, 0, 0, 1], [0.04, -4, 10, 0.88], [0.1, 18, 0, 1.18], [0.3, 0, 0, 1]] }, f.facing); // swing the arm down
    if (a.phase === "shoot") {
      const tg = f.lockTarget;
      if (tg && !tg.ko) a.aim += U.clamp(U.angDiff(a.aim, U.angle(a.x, a.y, tg.x, tg.y)), -2.5 * dt, 2.5 * dt);
      a.x += Math.cos(a.aim) * s.speed * dt; a.y += Math.sin(a.aim) * s.speed * dt;
      f.facing = U.angle(f.x, f.y, a.x, a.y);
      fx.add({ x: a.x, y: a.y, z: 30, vx: U.rand(-40, 40), vy: U.rand(-30, 30), vz: U.rand(0, 40), life: 0.25, size: 3, color: U.pick(["#ffffff", "#d6b8ff"]), shape: "pixel", add: true });
      for (const e of B.targetsOf(f)) {
        if (U.dist(a.x, a.y, e.x, e.y) > e.r + 18 || e.invuln > 0) continue;
        if (e.protect > 0) { fx.label(e.x, e.y - e.height - 8, "まもった！", "#9ad8ff", { size: 11 }); a.phase = "back"; break; }
        a.phase = "hold"; a.tgt = e; a.ht = 0; e.held = a; e.cast = null; e.dash = null;
        a.lift = s.lift / Math.sqrt(Math.max(1, e.mass));
        a.slab = Slabs.create(B, e.x, e.y, e.r * 1.2 + 14, { pieces: 1 });
        Audio.sfx("grab", B.pan(e.x));
        fx.ring(e.x, e.y, 6, 70, 0.3, "#d6b8ff", 5); fx.light(e.x, e.y, 140, "#b48cff", 0.9, 0.3);
      }
      const fl = B.arena.field;
      if (a.phase === "shoot" && (U.dist(f.x, f.y, a.x, a.y) > (B.turnMode ? 2000 : s.range) || a.x < fl.x || a.x > fl.x + fl.w || a.y < fl.y || a.y > fl.y + fl.h)) a.phase = "back";
    } else if (a.phase === "back") {
      const an = U.angle(a.x, a.y, f.x, f.y);
      a.x += Math.cos(an) * 1700 * dt; a.y += Math.sin(an) * 1700 * dt;
      if (U.dist(a.x, a.y, f.x, f.y) < 30) return true;
    } else if (a.phase === "hold" || a.phase === "slam") {
      const e = a.tgt;
      if (e.ko || !e.active) return release();
      a.ht += dt; a.x = e.x; a.y = e.y;
      f.facing = U.angle(f.x, f.y, e.x, e.y);
      e.hitstun = Math.max(e.hitstun, 0.2); e.vx = e.vy = 0; e.hopV = 0;
      if (a.phase === "hold") {
        const k = Math.min(1, a.ht / s.holdT);
        e.hop0 = e.z = a.lift * U.easeOut(k) + Math.sin(a.ht * 45) * 1.5;
        if (a.slab) a.slab.pieces[0].z = e.z / Arenas.K;
        if (Math.random() < 0.6) { const an = U.rand(U.TAU); fx.add({ x: e.x + Math.cos(an) * e.r, y: e.y + Math.sin(an) * e.r * 0.5, z: e.z + U.rand(0, e.height), vz: U.rand(30, 90), life: 0.4, size: U.rand(5, 9), color: "#b48cff", shape: "orb", add: true }); }
        fx.light(e.x, e.y, 120, "#b48cff", 0.6, 0.06);
        if (k >= 1) { a.phase = "slam"; a.ht = 0; a.z0 = e.z; }
      } else {
        const k = Math.min(1, a.ht / 0.1);
        e.hop0 = e.z = a.z0 * (1 - k * k);
        if (a.slab) a.slab.pieces[0].z = e.z / Arenas.K;
        if (k >= 1) {
          e.hop0 = e.z = 0; e.held = null;
          if (a.slab) Slabs.land(B, a.slab);
          B.hit(e, f, s, { mult: 1, knock: 60, angle: U.angle(f.x, f.y, e.x, e.y) });
          e.squash = 0.62;
          Audio.sfx("rock", B.pan(e.x));
          fx.impact("psychic", e.x, e.y, 6, true); fx.ring(e.x, e.y, 10, 130, 0.45, "#c9a8ff", 8);
          fx.dust(e.x, e.y, 12); fx.debris(e.x, e.y, 8, "#6a5a40", "#3e3424"); fx.shakeCam(12, 0.3);
          Arenas.decal(B.arena, "crater", e.x, e.y, Math.max(26, e.r * 1.2));
          return true;
        }
      }
    }
    return false;
  }

  // Expanding Force as "the field rises": `tele` s of glowing cracks (side-step to escape), then the disc of
  // floor splits into plates that rise with anyone standing on them, hover, and slam back down.
  function updateUplift(B, a, dt) {
    const f = a.owner, s = a.spec, fx = B.fx, K = Arenas.K;
    const T0 = s.tele, T1 = T0 + 0.45, T2 = T1 + 0.4, T3 = T2 + 0.13;
    const free = () => { for (const c of a.caught) { c.e.held = null; c.e.hop0 = c.e.z; } a.caught.length = 0; };
    if (f.ko || (!f.active && a.phase === "tele")) { free(); if (a.g) Slabs.land(B, a.g); return true; }
    if (a.t < T0 + 0.1) f.castHold = 0.1;
    if (a.phase === "tele") {
      if (Math.random() < 0.5) { const an = U.rand(U.TAU), d = a.r * U.rand(0.85, 1.05); fx.add({ x: a.x + Math.cos(an) * d, y: a.y + Math.sin(an) * d * 0.62, z: 2, vz: U.rand(30, 90), life: 0.4, size: U.rand(3, 5), color: U.pick(["#c9a8ff", "#ffffff"]), shape: "pixel", add: true }); }
      if (Math.random() < 0.25) fx.dust(a.x + U.rand(-a.r, a.r) * 0.8, a.y + U.rand(-a.r, a.r) * 0.45, 1, "rgba(150,130,100,0.5)");
      fx.shakeCam(2 + a.t * 6, 0.1);
      fx.light(a.x, a.y, a.r * 1.4, "#b48cff", 0.25 + a.t, 0.06);
      if (a.t < T0) return false;
      // tear the floor loose
      a.phase = "rise"; a.g = Slabs.create(B, a.x, a.y, a.r, { pieces: 6 });
      a.g.pieces.forEach((p, i) => { p.h = (i === 0 ? 52 : U.rand(26, 44)) / K * 2; p.ph = U.rand(U.TAU); p.delay = i === 0 ? 0 : U.rand(0, 0.12); });
      B.psy = { f, t: 0, dur: T3 - T0 + 0.6 };
      Audio.sfx("rock", B.pan(a.x)); fx.shakeCam(14, 0.3); fx.radialBlur(a.x, a.y, 0.5);
      fx.ring(a.x, a.y, 10, a.r * 1.4, 0.4, "#c9a8ff", 8);
      for (const e of B.targetsOf(f)) if (U.dist(a.x, a.y, e.x, e.y) < a.r + e.r * 0.4) {
        const pi = Slabs.pieceAt(a.g, e.x, e.y);
        a.caught.push({ e, pi }); e.held = a; e.cast = null; e.dash = null;
        B.hit(e, f, s, { mult: 0.35, knock: 0, angle: -Math.PI / 2, noStop: true });
      }
      for (let i = 0; i < 26; i++) { const an = U.rand(U.TAU), d = a.r * U.rand(0.95, 1.25); fx.add({ x: a.x + Math.cos(an) * d, y: a.y + Math.sin(an) * d * 0.62, z: 0, vx: Math.cos(an) * U.rand(20, 80), vy: Math.sin(an) * U.rand(10, 40), vz: U.rand(200, 420), grav: 700, life: U.rand(0.6, 1.1), size: U.rand(3, 8), color: U.pick(["#6a5a40", "#3e3424", Arenas.chunkColors(B.arena)[0]]), shape: "rock", spin: U.rand(-8, 8), bounce: true }); }
    }
    const g = a.g;
    if (a.phase === "rise" || a.phase === "hover") {
      for (const p of g.pieces) {
        const k = U.clamp((a.t - T0 - p.delay) / (T1 - T0), 0, 1);
        p.z = p.h * U.easeOut(k) + (a.t > T1 ? Math.sin((a.t - T1) * 5 + p.ph) * 1.5 : 0);
        p.dx = a.t > T1 ? Math.round(Math.sin((a.t - T1) * 3 + p.ph) * 0.6) : 0;
        if (Math.random() < 0.2) fx.add({ x: (p.cx + U.rand(-6, 6)) * K, y: p.cy * K, z: p.z * K - 2, vz: -20, grav: 400, life: 0.5, size: U.rand(2, 4), color: "#4a3424", shape: "rock" }); // soil raining off the underside
      }
      if (a.t >= T1) a.phase = "hover";
      if (a.t >= T2) { a.phase = "slam"; a.z0 = g.pieces.map((p) => p.z); Audio.sfx("dodge", B.pan(a.x)); }
    }
    if (a.phase === "slam") {
      const k = Math.min(1, (a.t - T2) / (T3 - T2));
      g.pieces.forEach((p, i) => { p.z = a.z0[i] * (1 - k * k); p.dx = 0; });
      if (k >= 1) {
        Slabs.land(B, g);
        Audio.sfx("hitHeavy", B.pan(a.x)); Audio.sfx("rock", B.pan(a.x)); Audio.sfx("thud");
        fx.shakeCam(20, 0.45); fx.flashScreen(0.18, "#c9a8ff");
        fx.ring(a.x, a.y, 20, a.r * 1.8, 0.55, "#ffffff", 9); fx.ring(a.x, a.y, 10, a.r * 2.3, 0.75, "#9a7aff", 12);
        fx.dust(a.x, a.y, 30, "rgba(160,140,110,0.6)");
        for (const p of g.pieces) fx.debris(p.cx * K, p.cy * K, 3, "#6a5a40", "#3e3424");
        Arenas.decal(B.arena, "crater", a.x, a.y, a.r * 0.45);
        for (let i = 0; i < 7; i++) { const an = U.rand(U.TAU); Arenas.decal(B.arena, "crack", a.x + Math.cos(an) * a.r, a.y + Math.sin(an) * a.r * 0.6, 30); }
        for (const c of a.caught) { const e = c.e; e.held = null; e.hop0 = e.z = 0; e.squash = 0.6; if (!e.ko) B.hit(e, f, s, { mult: 0.8, knock: 260, angle: U.angle(a.x, a.y, e.x, e.y) }); e.hop(60); }
        for (const e of B.targetsOf(f)) if (!a.caught.some((c) => c.e === e) && U.dist(a.x, a.y, e.x, e.y) < a.r * 1.35 + e.r) { B.hit(e, f, s, { mult: 0.35, knock: 220, angle: U.angle(a.x, a.y, e.x, e.y) }); e.hop(50); } // shockwave
        a.caught.length = 0;
        return true;
      }
    }
    for (const c of a.caught) { // riders move with their plate
      const e = c.e, p = g.pieces[c.pi];
      if (e.ko || !e.active) continue;
      e.hitstun = Math.max(e.hitstun, 0.2); e.vx = e.vy = 0; e.hopV = 0;
      e.hop0 = e.z = p.z * K;
    }
    return false;
  }

  // Dig / Dive: "down" (sinks into a hole) → "travel" (hidden, a hump of earth — or a bubbling bulge on water —
  // runs toward the target, heaping a ridge of broken soil behind it) → "up" (bursts out: crater, rocks, launch).
  function updateDig(B, a, dt) {
    const f = a.owner, s = a.spec, fx = B.fx, A = B.arena, pan = B.pan(f.x);
    const [, earth] = Arenas.chunkColors(A), soil = [earth, "#6a5a40", "#3e3424", Arenas.chunkColors(A)[0]];
    const done = () => { f.burrow = false; f.digSink = 0; if (B.mounds) B.mounds = B.mounds.filter((m) => m !== a.m); return true; };
    if (f.ko || !f.active) return done();
    f.castHold = 0.1;
    if (a.phase === "down") {
      const k = Math.min(1, a.t / 0.32);
      f.digSink = k * k; f.vx = f.vy = 0;
      if (!a.holed) {
        a.holed = true; fx.shakeCam(3, 0.15);
        if (a.wet) Water.splash(B, f.x, f.y, 1.4); else { Arenas.decal(A, "crater", f.x, f.y + 2, f.r * 1.05); Audio.sfx("rock", pan); }
      }
      for (let i = 0; i < 2; i++) { const an = U.rand(U.TAU); fx.add({ x: f.x + Math.cos(an) * f.r * 0.6, y: f.y + Math.sin(an) * f.r * 0.3, z: 2, vx: Math.cos(an) * U.rand(60, 140), vy: Math.sin(an) * U.rand(30, 70), vz: U.rand(120, 260), grav: 800, life: U.rand(0.4, 0.7), size: U.rand(2, 5), color: a.wet ? U.pick(["#ffffff", "#bfe6ff"]) : U.pick(soil), shape: a.wet ? "drop" : "rock", spin: U.rand(-8, 8), bounce: !a.wet }); }
      if (k >= 1) {
        a.phase = "travel"; a.t = 0; f.burrow = true; f.digSink = 1; a.lx = f.x; a.ly = f.y;
        if (!a.wet) { a.m = { x: f.x, y: f.y, r: Math.max(26, f.r * 1.35), h: 0, seed: a.seed }; (B.mounds || (B.mounds = [])).push(a.m); }
      }
      return false;
    }
    if (a.phase === "travel") {
      f.invuln = Math.max(f.invuln, 0.12);
      const tg = a.tgt && !a.tgt.ko && a.tgt.active ? a.tgt : null;
      if (tg) { a.tx = tg.x; a.ty = tg.y; }
      const d = U.dist(f.x, f.y, a.tx, a.ty), st = Math.min(d, 520 * dt);
      if (d > 1) { const p = Arenas.resolve(A, f.x + (a.tx - f.x) / d * st, f.y + (a.ty - f.y) / d * st, f.r); f.x = p.x; f.y = p.y; f.facing = U.angle(f.x, f.y, a.tx, a.ty); }
      if (a.m) { a.m.x = f.x; a.m.y = f.y; a.m.h = Math.min(7, a.t * 50); }
      if (Math.random() < 0.6) { const an = U.rand(U.TAU); fx.add({ x: f.x + Math.cos(an) * f.r * 0.8, y: f.y + Math.sin(an) * f.r * 0.4, z: a.wet ? 0 : 6, vx: Math.cos(an) * U.rand(20, 60), vy: Math.sin(an) * U.rand(10, 30), vz: U.rand(40, 120), grav: a.wet ? -60 : 700, life: U.rand(0.3, 0.6), size: U.rand(2, 4), color: a.wet ? "#dff4ff" : U.pick(soil), shape: a.wet ? "bubble" : "rock", spin: U.rand(-8, 8) }); }
      if (U.dist(f.x, f.y, a.lx, a.ly) > 14) { // leave the heaved-up tunnel behind
        a.lx = f.x; a.ly = f.y;
        if (a.wet) Water.ripple(A, f.x, f.y, f.r * 0.6, 40, 0.9, { foam: true });
        else { Ground.heap(A, f.x, f.y, f.r * 1.05, 3.6); Arenas.decal(A, "rubble", f.x + U.rand(-4, 4), f.y + U.rand(-3, 3), f.r * 0.9); }
        fx.shakeCam(1.4, 0.06);
      }
      if (d < 14 || a.t > 1.05) { a.phase = "up"; a.t = 0; }
      return false;
    }
    // up
    if (!a.erupted) {
      a.erupted = true; f.burrow = false; f.digSink = 1;
      if (B.mounds) B.mounds = B.mounds.filter((m) => m !== a.m);
      f.hop(90); f.squash = 1.3;
      for (const e of B.targetsOf(f)) if (U.dist(f.x, f.y, e.x, e.y) < s.r + e.r) { B.hit(e, f, s, { mult: 1, knock: 220 * s.knock, angle: U.angle(f.x, f.y, e.x, e.y), contact: true }); e.hop(70); }
      for (const p of A.props) if (p.alive && U.dist(f.x, f.y, p.x, p.y) < s.r + p.r) Arenas.damageProp(A, p, s.power, fx);
      fx.shakeCam(9, 0.3); fx.stop(0.05);
      if (a.wet) { Water.splash(B, f.x, f.y, 2.6); }
      else {
        Arenas.decal(A, "crater", f.x, f.y, s.r * 0.6);
        for (let i = 0; i < 4; i++) { const an = U.rand(U.TAU); Arenas.decal(A, "rubble", f.x + Math.cos(an) * s.r * 0.7, f.y + Math.sin(an) * s.r * 0.45, 18); }
        for (let i = 0; i < 22; i++) { const an = U.rand(U.TAU), dd = U.rand(4, s.r * 0.5); fx.add({ x: f.x + Math.cos(an) * dd, y: f.y + Math.sin(an) * dd * 0.62, z: 4, vx: Math.cos(an) * U.rand(60, 200), vy: Math.sin(an) * U.rand(30, 100), vz: U.rand(220, 460), grav: 800, life: U.rand(0.6, 1.1), size: U.rand(3, 8), color: U.pick(soil), shape: "rock", spin: U.rand(-10, 10), bounce: true }); }
        fx.dust(f.x, f.y, 10, "rgba(150,125,90,0.55)");
        Audio.sfx("rock", pan);
      }
      fx.ring(f.x, f.y, 10, s.r * 1.4, 0.35, a.wet ? "#dff4ff" : "#e8d2a0", 5);
    }
    f.digSink = Math.max(0, 1 - a.t / 0.14);
    return a.t > 0.3 ? done() : false;
  }

  function segDist(ax, ay, bx, by, px, py) {
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy || 1;
    const t = U.clamp(((px - ax) * dx + (py - ay) * dy) / l2, 0, 1);
    return { d: U.dist(px, py, ax + dx * t, ay + dy * t), t };
  }

  // ------------------------------------------------------------------------------------------------
  // Drawing of attack entities. layer "ground" = telegraphs, zone floors; "air" = projectiles, beams.
  function draw(ctx, B, layer, t) {
    const fx = B.fx;
    for (const a of B.attacks) {
      const s = a.spec, th = fx.theme(s.type, a.owner);
      if (layer === "ground") {
        if (a.kind === "strike" && !a.fired) {
          const k = a.t / a.delay;
          ctx.save(); ctx.globalAlpha = 0.55 + 0.3 * Math.sin(t * 18);
          ctx.strokeStyle = th.c[1]; ctx.lineWidth = 3; ctx.setLineDash([10, 8]);
          ctx.beginPath(); ctx.ellipse(a.x, a.y, a.r, a.r * FXV.gsq(0.55), 0, 0, U.TAU); ctx.stroke();
          ctx.setLineDash([]); ctx.globalAlpha = 0.25; ctx.fillStyle = th.c[1];
          ctx.beginPath(); ctx.ellipse(a.x, a.y, a.r * k, a.r * FXV.gsq(0.55) * k, 0, 0, U.TAU); ctx.fill();
          ctx.restore();
          for (const sb of a.sub) { ctx.save(); ctx.fillStyle = "rgba(0,0,0,0.3)"; const sr = 14 * (1 - sb.z / 600); ctx.beginPath(); ctx.ellipse(sb.x, sb.y, sr, sr * FXV.gsq(0.5), 0, 0, U.TAU); ctx.fill(); ctx.restore(); }
        } else if (a.kind === "zone") {
          ctx.save(); ctx.globalAlpha = 0.35 + 0.1 * Math.sin(t * 10);
          const g = ctx.createRadialGradient(a.x, a.y, a.r * 0.2, a.x, a.y, a.r);
          g.addColorStop(0, U.rgba(th.c[2], 0.1)); g.addColorStop(0.8, U.rgba(th.c[1], 0.5)); g.addColorStop(1, U.rgba(th.c[1], 0));
          ctx.fillStyle = g; ctx.beginPath(); ctx.ellipse(a.x, a.y, a.r, a.r * FXV.gsq(0.55), 0, 0, U.TAU); ctx.fill();
          ctx.restore();
        } else if (a.kind === "uplift" && a.phase === "tele") { // the disc that will lift: pulsing violet cracks
          const k = Math.min(1, a.t / s.tele), q62 = FXV.gsq(0.62), q25 = 0.4 * q62, q43 = 0.7 * q62;
          ctx.save(); ctx.globalCompositeOperation = "lighter";
          ctx.globalAlpha = 0.12 + 0.1 * k; ctx.fillStyle = "#9a6aff"; ctx.beginPath(); ctx.ellipse(a.x, a.y, a.r, a.r * q62, 0, 0, U.TAU); ctx.fill();
          ctx.globalAlpha = 0.6 + 0.35 * Math.sin(t * 22); ctx.strokeStyle = "#d6b8ff"; ctx.lineWidth = 3;
          const rng = U.seeded((a.x * 7 + a.y) | 0);
          ctx.beginPath(); for (let i = 0; i <= 40; i++) { const an = (i / 40) * U.TAU, rr = a.r * (0.94 + rng() * 0.12); i ? ctx.lineTo(a.x + Math.cos(an) * rr, a.y + Math.sin(an) * rr * q62) : ctx.moveTo(a.x + Math.cos(an) * rr, a.y + Math.sin(an) * rr * q62); } ctx.stroke();
          ctx.lineWidth = 2; ctx.globalAlpha *= k;
          for (let j = 0; j < 6; j++) { const an = j * 1.05 + rng(); ctx.beginPath(); ctx.moveTo(a.x + Math.cos(an) * a.r * 0.4, a.y + Math.sin(an) * a.r * q25); ctx.lineTo(a.x + Math.cos(an + 0.2) * a.r * 0.7, a.y + Math.sin(an + 0.2) * a.r * q43); ctx.lineTo(a.x + Math.cos(an) * a.r, a.y + Math.sin(an) * a.r * q62); ctx.stroke(); }
          ctx.restore();
        } else if (a.kind === "psybreak" && !a.fired) { // danger band on the floor
          ctx.save(); ctx.translate(a.owner.x, a.owner.y); ctx.rotate(a.aim);
          ctx.globalAlpha = 0.1 + 0.08 * Math.sin(t * 24); ctx.fillStyle = "#ff3aa0"; ctx.fillRect(0, -s.w / 2, a.len || s.range, s.w);
          ctx.globalAlpha = 0.5; ctx.strokeStyle = "#ff5fd0"; ctx.lineWidth = 2; ctx.setLineDash([12, 10]); ctx.lineDashOffset = -t * 60;
          ctx.beginPath(); ctx.moveTo(0, -s.w / 2); ctx.lineTo(a.len || s.range, -s.w / 2); ctx.moveTo(0, s.w / 2); ctx.lineTo(a.len || s.range, s.w / 2); ctx.stroke();
          ctx.restore();
        } else if (a.kind === "aoe" && s.wave) {
          VFX.waveGround(ctx, a, t);
        } else if (a.kind === "proj") {
          ctx.save();
          const zScale = U.clamp(1 - (a.z || 16) / 220, 0.42, 1);
          ctx.fillStyle = `rgba(0,0,0,${(0.32 * zScale).toFixed(2)})`;
          const sr = a.r * (a.size || 1) * 0.85 * zScale;
          ctx.beginPath(); ctx.ellipse(a.x, a.y, sr, sr * FXV.gsq(0.45), 0, 0, U.TAU); ctx.fill();
          ctx.restore();
        }
        continue;
      }
      // air layer
      airView(ctx, a);
      if (a.kind === "aoe" && s.wave) VFX.wave(ctx, a, t);
      if (a.kind === "spiral" && a.R) (s.pull ? VFX.whirlAir(ctx, a, t) : drawSpiral(ctx, a, th, t));
      else if (a.kind === "erupt") drawSpikes(ctx, a, th, t);
      else if (a.kind === "strike" && a.sky === "pillar") drawPillar(ctx, a, th, t);
      else if (a.kind === "strike" && a.sky === "daimonji") VFX.fireBlast(ctx, a, t);
      if (a.kind === "proj") (s.orb === "shadow" ? drawShadowBall : drawProjectile)(ctx, a, th, t);
      else if (a.kind === "beam") drawBeam(ctx, a, th, t);
      else if (a.kind === "psybreak") drawPsybreak(ctx, a, t);
      else if (a.kind === "grab") drawTether(ctx, a, t);
      else if (a.kind === "strike") {
        for (const sb of a.sub) {
          FXV.at(ctx, sb.x, sb.y);
          ctx.save(); ctx.translate(sb.x, sb.y - sb.z); ctx.rotate(sb.rot + t * 3);
          if (a.sky === "meteor") { ctx.globalCompositeOperation = "lighter"; const g = ctx.createRadialGradient(0, 0, 0, 0, 0, 22); g.addColorStop(0, "#fff"); g.addColorStop(0.4, th.c[1]); g.addColorStop(1, "rgba(0,0,0,0)"); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(0, 0, 22, 0, U.TAU); ctx.fill(); }
          else { ctx.fillStyle = th.c[2]; ctx.beginPath(); ctx.moveTo(-12, -6); ctx.lineTo(4, -13); ctx.lineTo(13, 2); ctx.lineTo(3, 12); ctx.lineTo(-10, 8); ctx.closePath(); ctx.fill(); ctx.fillStyle = th.c[1]; ctx.beginPath(); ctx.moveTo(-8, -4); ctx.lineTo(3, -9); ctx.lineTo(7, 0); ctx.closePath(); ctx.fill(); }
          ctx.restore();
        }
        FXV.at(ctx, a.x, a.y);
        if (a.fired && a.sky === "bolt" && a.t < a.doneAt - 0.15) { // lightning from the sky
          ctx.save(); ctx.globalCompositeOperation = "lighter"; ctx.lineCap = "round";
          for (const [w, c] of [[14, U.rgba(th.c[2], 0.5)], [6, th.c[1]], [2.5, "#ffffff"]]) {
            ctx.strokeStyle = c; ctx.lineWidth = w; ctx.beginPath();
            let x = a.x + U.rand(-30, 30), y = a.y - 700; ctx.moveTo(x, y);
            const rng = U.seeded((a.boltSeed * 1000 + Math.floor(t * 30)) | 0);
            while (y < a.y) { y += 40 + rng() * 40; x += (rng() - 0.5) * 70 * (y < a.y - 60 ? 1 : 0.2); ctx.lineTo(Math.min(x, a.x + 60), Math.min(y, a.y)); }
            ctx.stroke();
          }
          ctx.restore();
        }
        if (a.fired && a.sky === "psy") {
          const k = (a.t - a.delay) / 0.35;
          ctx.save(); ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = 1 - k;
          for (let i = 0; i < 3; i++) { ctx.strokeStyle = th.c[i]; ctx.lineWidth = 6 - i * 1.5; ctx.beginPath(); ctx.ellipse(a.x, a.y - 20, a.r * (0.3 + k * 1.2) * (1 - i * 0.2), a.r * (0.3 + k * 1.2) * (1 - i * 0.2) * 0.7, 0, 0, U.TAU); ctx.stroke(); }
          ctx.restore();
        }
      }
    }
    if (layer === "air" && B.clashes && B.clashes.length) {
      for (const cl of B.clashes) drawClashNexus(ctx, cl, fx, t);
    }
  }

  // [ユーザー指定 2026-11-16] 3D 表示で、この技の空中エフェクトをどこに立てるか（fx.js の FXV）。ビームは口から先端までの線、
  // それ以外は技の中心（無ければ使い手）の足もとに立つ板。平面表示では何もしない。
  function airView(ctx, a) {
    if (!FXV.on) return;
    const f = a.owner, s = a.spec;
    if ((a.kind === "beam" || a.kind === "psybreak") && f) {
      const c = Math.cos(a.aim), sn = Math.sin(a.aim), beam = a.kind === "beam";
      const L = Math.max(30, beam ? (a.end ? a.end.d : s.len) : (a.clashEnd ? a.clashEnd.d : (a.len || s.range)));
      const x0 = f.x + c * (beam ? 18 : 16), y0 = f.y + sn * (beam ? 10 : 8), h = f.height * (beam ? 0.55 : 0.5);
      FXV.seg(ctx, x0, y0, h, x0 + c * L, y0 + sn * L, h);
    } else if (a.kind === "grab" && f) {
      const e = a.tgt;
      FXV.seg(ctx, f.x + Math.cos(f.facing) * 14, f.y, f.height * 0.55, a.x, e ? e.y : a.y, e ? e.z + e.height * 0.5 : 30);
    } else if (a.x !== undefined && a.y !== undefined) FXV.at(ctx, a.x, a.y);
    else if (f) FXV.at(ctx, f.x, f.y);
  }

  function drawClashNexus(ctx, cl, fx, t) {
    const x = cl.x, y = cl.y - 22;
    FXV.at(ctx, cl.x, cl.y);
    const thA = fx.theme(cl.a.spec.type, cl.a.owner), thB = fx.theme(cl.b.spec.type, cl.b.owner);
    const R = (26 + 8 * Math.sin(t * 44)) * (1 + Math.min(0.4, cl.t * 0.35));
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const g = ctx.createRadialGradient(x, y, 0, x, y, R * 2.4);
    g.addColorStop(0, "rgba(255,255,255,0.98)");
    g.addColorStop(0.28, U.rgba(thA.c[0], 0.85));
    g.addColorStop(0.55, U.rgba(thB.c[1], 0.65));
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, R * 2.4, 0, U.TAU); ctx.fill();
    // Dual opposing energy rings + electric arcs
    for (let i = 0; i < 2; i++) {
      ctx.strokeStyle = i === 0 ? thA.c[1] : thB.c[1];
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.ellipse(x, y, R * (1.2 + i * 0.25), R * 0.55, t * (i ? -9 : 9) + i * 1.2, 0, U.TAU);
      ctx.stroke();
    }
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2;
    for (let i = 0; i < 6; i++) {
      const an = (i / 6) * U.TAU + t * 14;
      ctx.beginPath(); ctx.moveTo(x, y);
      ctx.lineTo(x + Math.cos(an) * R * 1.1, y + Math.sin(an) * R * 0.8);
      ctx.lineTo(x + Math.cos(an + 0.25) * R * 1.85, y + Math.sin(an + 0.25) * R * 1.45);
      ctx.stroke();
    }
    ctx.restore();
    flare(ctx, x, y, R * 3.1, t * 7, "#ffffff");
  }

  // Stage-1 Windup Ground Danger Telegraphs + shared AI threat query.
  function windupDanger(f, x, y) {
    if (!f || f.ko || !f.cast || f.cast.fired) return null;
    const c = f.cast, s0 = c.spec;
    if (!s0 || (!s0.power && s0.kind !== "grab" && s0.kind !== "status")) return null;
    const s = visSpec(f, s0);
    const u = U.clamp(c.t / Math.max(0.05, s0.windup || 0.2), 0, 1);
    const aim = c.aim;
    const dx = x - f.x, dy = y - f.y, d = Math.hypot(dx, dy);
    const pad = 28;
    if (s.kind === "melee") {
      if (d <= s.r + pad && Math.abs(U.angDiff(aim, Math.atan2(dy, dx))) <= (s.arc || 1.4) * 0.5 + 0.3) {
        const side = U.angDiff(aim, Math.atan2(dy, dx)) >= 0 ? 1 : -1;
        const esc = aim + side * (Math.PI * 0.58);
        return { w: 2.2 + u, ux: Math.cos(esc), uy: Math.sin(esc), u };
      }
    } else if (s.kind === "dash") {
      const ca = Math.cos(aim), sa = Math.sin(aim), along = dx * ca + dy * sa, perp = -dx * sa + dy * ca;
      if (along > -16 && along < s.dist + pad && Math.abs(perp) < f.r + pad + 14) {
        const side = perp >= 0 ? 1 : -1;
        return { w: 2.6 + u, ux: -sa * side, uy: ca * side, u };
      }
    } else if (s.kind === "beam" || s.kind === "psybreak") {
      const ca = Math.cos(aim), sa = Math.sin(aim), along = dx * ca + dy * sa, perp = -dx * sa + dy * ca;
      const L = s.len || s.range || 460, W = (s.w || 32) * 0.5 + pad;
      if (along > -10 && along < L && Math.abs(perp) < W) {
        const side = perp >= 0 ? 1 : -1;
        return { w: 3.0 + u, ux: -sa * side, uy: ca * side, u, beam: true };
      }
    } else if (s.kind === "aoe" || s.kind === "spiral") {
      const R = (s.r1 || 180) + pad;
      if (d < R) {
        const an = d > 1 ? Math.atan2(dy, dx) : aim + Math.PI;
        return { w: 2.5 + u, ux: Math.cos(an), uy: Math.sin(an), u };
      }
    } else if (s.kind === "leap" || s.kind === "strike" || s.kind === "uplift" || s.kind === "dig" || s.kind === "zone") {
      const tg = f.lockTarget;
      const maxR = s.range || 240;
      let tx = tg ? tg.x : f.x + Math.cos(aim) * Math.min(240, maxR);
      let ty = tg ? tg.y : f.y + Math.sin(aim) * Math.min(240, maxR);
      const distT = U.dist(f.x, f.y, tx, ty);
      if (distT > maxR) { tx = f.x + ((tx - f.x) / distT) * maxR; ty = f.y + ((ty - f.y) / distT) * maxR; }
      const R = (s.r || 95) + pad, td = U.dist(tx, ty, x, y);
      if (td < R) {
        const an = td > 1 ? Math.atan2(y - ty, x - tx) : aim + Math.PI * 0.5;
        return { w: 2.6 + u, ux: Math.cos(an), uy: Math.sin(an), u };
      }
    } else if (s.kind === "erupt") {
      const ca = Math.cos(aim), sa = Math.sin(aim), along = dx * ca + dy * sa, perp = -dx * sa + dy * ca;
      const L = 50 + (s.n || 6) * (s.gap || 44) + pad;
      if (along > 12 && along < L && Math.abs(perp) < 42 + pad) {
        const side = perp >= 0 ? 1 : -1;
        return { w: 2.3 + u, ux: -sa * side, uy: ca * side, u };
      }
    } else if (s.kind === "proj" || s.kind === "spread" || s.kind === "grab" || s.kind === "status") {
      const ca = Math.cos(aim), sa = Math.sin(aim), along = dx * ca + dy * sa, perp = -dx * sa + dy * ca;
      if (along > 0 && along < 340 && Math.abs(perp) < (s.kind === "spread" ? 75 : 44)) {
        const side = perp >= 0 ? 1 : -1;
        return { w: 1.6 + u, ux: -sa * side, uy: ca * side, u };
      }
    }
    return null;
  }

  function drawWindups(ctx, B, t) {
    for (const f of B.active) {
      if (!f || f.ko || !f.cast || f.cast.fired) continue;
      const c = f.cast, s0 = c.spec;
      if (!s0 || (!s0.power && s0.kind !== "grab" && s0.kind !== "status")) continue;
      const s = visSpec(f, s0);
      const u = U.clamp(c.t / Math.max(0.05, s0.windup || 0.2), 0, 1);
      const warn = f.side === 0 ? "#38bdf8" : "#fb7185";
      const aim = c.aim;
      ctx.save();
      if (s.kind === "melee") {
        const arc = s.arc || 1.4, R = s.r, ry = R * FXV.gsq(0.62);
        ctx.translate(f.x, f.y);
        ctx.scale(1, ry / R);
        ctx.fillStyle = U.rgba(warn, 0.1 + 0.14 * u);
        ctx.beginPath(); ctx.moveTo(0, 0); ctx.arc(0, 0, R, aim - arc / 2, aim + arc / 2); ctx.closePath(); ctx.fill();
        ctx.fillStyle = U.rgba(warn, 0.22);
        ctx.beginPath(); ctx.moveTo(0, 0); ctx.arc(0, 0, R * u, aim - arc / 2, aim + arc / 2); ctx.closePath(); ctx.fill();
        ctx.strokeStyle = U.rgba(warn, 0.65); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(0, 0, R, aim - arc / 2, aim + arc / 2); ctx.stroke();
      } else if (s.kind === "dash") {
        const W = f.r * 1.9 + 12, L = s.dist;
        ctx.translate(f.x, f.y); ctx.rotate(aim);
        ctx.fillStyle = U.rgba(warn, 0.1 + 0.12 * u); ctx.fillRect(0, -W / 2, L, W);
        ctx.fillStyle = U.rgba(warn, 0.24); ctx.fillRect(0, -W / 2, L * u, W);
        ctx.strokeStyle = U.rgba(warn, 0.65); ctx.lineWidth = 2; ctx.setLineDash([8, 6]); ctx.lineDashOffset = -t * 50;
        ctx.strokeRect(0, -W / 2, L, W);
      } else if (s.kind === "beam" || s.kind === "psybreak") {
        const W = s.w || 36, L = s.len || s.range || 460;
        ctx.translate(f.x, f.y); ctx.rotate(aim);
        ctx.fillStyle = U.rgba(warn, 0.09 + 0.14 * u); ctx.fillRect(0, -W / 2, L, W);
        ctx.strokeStyle = U.rgba(warn, 0.72); ctx.lineWidth = 1.8; ctx.setLineDash([10, 8]); ctx.lineDashOffset = -t * 70;
        ctx.beginPath(); ctx.moveTo(0, -W / 2); ctx.lineTo(L, -W / 2); ctx.moveTo(0, W / 2); ctx.lineTo(L, W / 2); ctx.stroke();
      } else if (s.kind === "erupt") {
        const W = 54, L = 50 + (s.n || 6) * (s.gap || 44);
        ctx.translate(f.x, f.y); ctx.rotate(aim);
        ctx.fillStyle = U.rgba(warn, 0.08 + 0.13 * u); ctx.fillRect(28, -W / 2, L - 28, W);
        ctx.fillStyle = U.rgba(warn, 0.22); ctx.fillRect(28, -W / 2, (L - 28) * u, W);
        ctx.strokeStyle = U.rgba(warn, 0.66); ctx.lineWidth = 1.8; ctx.setLineDash([7, 6]); ctx.lineDashOffset = -t * 55;
        ctx.strokeRect(28, -W / 2, L - 28, W);
      } else if (s.kind === "aoe" || s.kind === "spiral") {
        const R = s.r1 || 180;
        ctx.fillStyle = U.rgba(warn, 0.08 + 0.12 * u);
        ctx.beginPath(); ctx.ellipse(f.x, f.y, R, R * FXV.gsq(0.58), 0, 0, U.TAU); ctx.fill();
        ctx.fillStyle = U.rgba(warn, 0.2);
        ctx.beginPath(); ctx.ellipse(f.x, f.y, R * u, R * FXV.gsq(0.58) * u, 0, 0, U.TAU); ctx.fill();
        ctx.strokeStyle = U.rgba(warn, 0.68); ctx.lineWidth = 2; ctx.setLineDash([9, 7]);
        ctx.beginPath(); ctx.ellipse(f.x, f.y, R, R * FXV.gsq(0.58), 0, 0, U.TAU); ctx.stroke();
      } else if (s.kind === "leap" || s.kind === "strike" || s.kind === "uplift" || s.kind === "dig" || s.kind === "zone") {
        const tg = f.lockTarget;
        const maxR = s.range || 240;
        let tx = tg ? tg.x : f.x + Math.cos(aim) * Math.min(240, maxR);
        let ty = tg ? tg.y : f.y + Math.sin(aim) * Math.min(240, maxR);
        const distT = U.dist(f.x, f.y, tx, ty);
        if (distT > maxR) { tx = f.x + ((tx - f.x) / distT) * maxR; ty = f.y + ((ty - f.y) / distT) * maxR; }
        const R = s.r || 90;
        ctx.strokeStyle = U.rgba(warn, 0.7); ctx.lineWidth = 2; ctx.setLineDash([7, 6]); ctx.lineDashOffset = -t * 45;
        ctx.beginPath(); ctx.ellipse(tx, ty, R, R * FXV.gsq(0.58), 0, 0, U.TAU); ctx.stroke();
        ctx.setLineDash([]); ctx.fillStyle = U.rgba(warn, 0.18 + 0.1 * u);
        ctx.beginPath(); ctx.ellipse(tx, ty, R * u, R * FXV.gsq(0.58) * u, 0, 0, U.TAU); ctx.fill();
      } else if (s.kind === "proj" || s.kind === "spread" || s.kind === "grab" || s.kind === "status") {
        const span = s.kind === "spread" ? (s.spread || 0.3) * ((s.n || 3) - 1) * 0.5 : 0;
        ctx.translate(f.x, f.y); ctx.rotate(aim);
        ctx.strokeStyle = U.rgba(warn, 0.35 + 0.45 * u); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(20, -10 - span * 45); ctx.lineTo(48 + 30 * u, 0); ctx.lineTo(20, 10 + span * 45); ctx.stroke();
        ctx.fillStyle = U.rgba(warn, 0.14 + 0.14 * u);
        ctx.beginPath(); ctx.moveTo(20, -10 - span * 45); ctx.lineTo(48 + 30 * u, 0); ctx.lineTo(20, 10 + span * 45); ctx.closePath(); ctx.fill();
      }
      ctx.restore();
    }
  }

  function drawSpiral(ctx, a, th, t) {
    const R = a.R, x0 = a.x, y0 = a.y - 20;
    ctx.save(); ctx.lineCap = "round";
    for (const [w, col] of [[30, U.rgba(th.c[2], 0.55)], [18, th.c[1]], [8, th.c[0]], [3, "#ffffff"]]) {
      ctx.strokeStyle = col; ctx.lineWidth = w;
      for (let i = 0; i < 4; i++) {
        ctx.beginPath();
        for (let k = 0; k <= 16; k++) { const r = 10 + (R - 10) * (k / 16), an = a.rot + i * Math.PI / 2 - (k / 16) * 1.6; const px = x0 + Math.cos(an) * r, py = y0 + Math.sin(an) * r * 0.6; k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
        ctx.stroke();
      }
    }
    ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.arc(x0, y0, 12, 0, U.TAU); ctx.fill();
    ctx.restore();
  }
  function drawSpikes(ctx, a, th, t) {
    for (const sp of a.spikes) {
      FXV.at(ctx, sp.x, sp.y);
      const k = Math.min(1, sp.t / 0.1), sink = a.t > a.n * 0.06 + 0.6 ? Math.min(1, (a.t - a.n * 0.06 - 0.6) / 0.3) : 0;
      const h = sp.h * U.easeOutBack(k) * (1 - sink), w = sp.h * 0.42;
      if (h < 2) continue;
      ctx.fillStyle = th.c[3] || th.c[2]; ctx.beginPath(); ctx.moveTo(sp.x - w, sp.y); ctx.lineTo(sp.x - w * 0.15, sp.y - h); ctx.lineTo(sp.x + w, sp.y); ctx.fill();
      ctx.fillStyle = th.c[2]; ctx.beginPath(); ctx.moveTo(sp.x - w * 0.6, sp.y); ctx.lineTo(sp.x - w * 0.15, sp.y - h); ctx.lineTo(sp.x + w * 0.2, sp.y); ctx.fill();
      ctx.fillStyle = th.c[1]; ctx.beginPath(); ctx.moveTo(sp.x - w * 0.35, sp.y - 4); ctx.lineTo(sp.x - w * 0.15, sp.y - h); ctx.lineTo(sp.x - w * 0.02, sp.y - 6); ctx.fill();
      ctx.fillStyle = th.c[3] || th.c[2]; ctx.beginPath(); ctx.moveTo(sp.x + w * 0.2, sp.y); ctx.lineTo(sp.x + w * 0.55, sp.y - h * 0.55); ctx.lineTo(sp.x + w * 1.05, sp.y); ctx.fill();
    }
  }
  function drawPillar(ctx, a, th, t) {
    if (!a.fired) { // gathering light from the sky
      const k = a.t / a.delay;
      ctx.save(); ctx.globalAlpha = 0.35 + k * 0.4; ctx.fillStyle = "#fff8d0";
      ctx.fillRect(a.x - 3 - k * 6, a.y - 900, 6 + k * 12, 900);
      ctx.restore(); return;
    }
    const k = (a.t - a.delay) / 0.35;
    if (k > 1) return;
    const w = a.r * (1.1 - k * 0.5);
    ctx.save();
    for (const [m, col] of [[1.5, U.rgba("#fff3a0", 0.5)], [1, "#fffbe0"], [0.55, "#ffffff"]]) { ctx.fillStyle = col; ctx.fillRect(a.x - w * m * 0.5, a.y - 1000, w * m, 1000); ctx.beginPath(); ctx.ellipse(a.x, a.y, w * m * 0.6, w * m * 0.32, 0, 0, U.TAU); ctx.fill(); }
    ctx.restore();
  }
  // 4-point twinkle (the classic RPG spell sparkle): long thin cross + short diagonal cross.
  function flare(ctx, x, y, R, rot, col) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(rot); ctx.globalCompositeOperation = "lighter"; ctx.fillStyle = col;
    for (const [len, w, r0] of [[R, R * 0.12, 0], [R * 0.5, R * 0.09, Math.PI / 4]]) {
      ctx.save(); ctx.rotate(r0);
      ctx.beginPath(); ctx.moveTo(-len, 0); ctx.lineTo(0, -w); ctx.lineTo(len, 0); ctx.lineTo(0, w); ctx.closePath(); ctx.fill();
      ctx.beginPath(); ctx.moveTo(0, -len); ctx.lineTo(w, 0); ctx.lineTo(0, len); ctx.lineTo(-w, 0); ctx.closePath(); ctx.fill();
      ctx.restore();
    }
    ctx.restore();
  }
  function drawProjectile(ctx, a, th, t) {
    const x = a.x, y = a.y - a.z, R = a.r * (a.size || 1) * 1.1;
    // afterimages: three shrinking ghost orbs behind the shot
    const sp = Math.hypot(a.vx, a.vy) || 1, ux = a.vx / sp, uy = a.vy / sp;
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    for (let i = 3; i >= 1; i--) { ctx.globalAlpha = 0.45 - i * 0.1; ctx.fillStyle = th.c[i === 1 ? 1 : 2]; ctx.beginPath(); ctx.arc(x - ux * R * 1.3 * i, y - uy * R * 0.8 * i, R * (1 - i * 0.18), 0, U.TAU); ctx.fill(); }
    ctx.restore();
    ctx.save();
    ctx.globalCompositeOperation = th.add || a.statusOnly ? "lighter" : "source-over";
    const g = ctx.createRadialGradient(x, y, 0, x, y, R * 1.8);
    g.addColorStop(0, "rgba(255,255,255,0.9)"); g.addColorStop(0.25, th.c[1]); g.addColorStop(0.6, U.rgba(th.c[2], 0.5)); g.addColorStop(1, U.rgba(th.c[2], 0));
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, R * 1.8, 0, U.TAU); ctx.fill();
    ctx.globalCompositeOperation = "source-over";
    if (th.shape === "rock" || th.shape === "leaf" || th.shape === "shard") {
      ctx.translate(x, y); ctx.rotate(a.rot);
      ctx.fillStyle = th.c[2];
      if (th.shape === "leaf") { ctx.beginPath(); ctx.ellipse(0, 0, R * 1.3, R * 0.5, 0, 0, U.TAU); ctx.fill(); ctx.fillStyle = th.c[1]; ctx.beginPath(); ctx.ellipse(-R * 0.2, -R * 0.1, R * 0.8, R * 0.25, 0, 0, U.TAU); ctx.fill(); }
      else if (th.shape === "shard") { ctx.beginPath(); ctx.moveTo(R * 1.4, 0); ctx.lineTo(-R * 0.6, R * 0.6); ctx.lineTo(-R * 0.3, 0); ctx.lineTo(-R * 0.6, -R * 0.6); ctx.closePath(); ctx.fillStyle = th.c[1]; ctx.fill(); }
      else { ctx.beginPath(); ctx.moveTo(-R, -R * 0.4); ctx.lineTo(R * 0.3, -R); ctx.lineTo(R, R * 0.1); ctx.lineTo(R * 0.2, R); ctx.lineTo(-R * 0.8, R * 0.6); ctx.closePath(); ctx.fill(); ctx.fillStyle = th.c[1]; ctx.beginPath(); ctx.moveTo(-R * 0.6, -R * 0.3); ctx.lineTo(R * 0.2, -R * 0.7); ctx.lineTo(R * 0.4, 0); ctx.closePath(); ctx.fill(); }
    } else {
      ctx.fillStyle = th.c[3] || th.c[2]; ctx.beginPath(); ctx.arc(x, y, R + 3, 0, U.TAU); ctx.fill();
      const g2 = ctx.createRadialGradient(x - R * 0.3, y - R * 0.3, 0, x, y, R);
      g2.addColorStop(0, "#ffffff"); g2.addColorStop(0.3, th.c[0]); g2.addColorStop(0.8, th.c[1]); g2.addColorStop(1, th.c[2]);
      ctx.fillStyle = g2; ctx.beginPath(); ctx.arc(x, y, R, 0, U.TAU); ctx.fill();
      if (th.shape === "wisp" || th.shape === "smoke") { ctx.strokeStyle = th.c[0]; ctx.lineWidth = 2; ctx.globalAlpha = 0.7; ctx.beginPath(); ctx.ellipse(x, y, R * 1.5, R * 0.6, t * 5, 0, U.TAU); ctx.stroke(); }
    }
    ctx.restore();
    if (!a.statusOnly) flare(ctx, x, y, R * (2.6 + 0.6 * Math.sin(t * 30 + a.rot)), a.rot * 0.5, th.c[0]);
  }

  function drawFlameStream(ctx, a, th, t) {
    const f = a.owner, s = a.spec;
    const ox = f.x + Math.cos(a.aim) * 18, oy = f.y - f.height * 0.55 + Math.sin(a.aim) * 10;
    const L = a.end ? a.end.d : s.len;
    const k = Math.min(1, a.t / 0.15) * (a.t > a.dur - 0.12 ? Math.max(0, (a.dur - a.t) / 0.12) : 1);
    const n = 15;
    ctx.save(); ctx.translate(ox, oy); ctx.rotate(a.aim);
    const pal = s.type === "dragon" ? ["#ffffff", "#c8b8ff", "#8a5cff", "#4a1fb0"] : ["#fffbe0", "#ffd84a", "#ff8a1a", "#c8320a"];
    for (let i = n; i >= 0; i--) {
      const u = i / n, ph = (t * 9 + i * 0.37 + a.seed) % 1;
      const x = (u + ph / n) * L * k, wob = Math.sin(t * 23 + i * 1.7) * s.w * 0.25 * u;
      const r = s.w * (0.28 + u * 0.75) * (0.85 + 0.15 * Math.sin(t * 31 + i));
      ctx.fillStyle = pal[3]; ctx.beginPath(); ctx.arc(x, wob, r * 1.12, 0, U.TAU); ctx.fill();
      ctx.fillStyle = pal[2]; ctx.beginPath(); ctx.arc(x - r * 0.1, wob - r * 0.08, r, 0, U.TAU); ctx.fill();
      ctx.fillStyle = pal[1]; ctx.beginPath(); ctx.arc(x - r * 0.2, wob - r * 0.15, r * 0.66, 0, U.TAU); ctx.fill();
      if (u < 0.65) { ctx.fillStyle = pal[0]; ctx.beginPath(); ctx.arc(x - r * 0.25, wob - r * 0.2, r * 0.34, 0, U.TAU); ctx.fill(); }
    }
    ctx.restore();
  }
  function drawBeam(ctx, a, th, t) {
    const f = a.owner, s = a.spec;
    const styled = VFX.styleOf(s); // fire cones, ice lances, water jets have their own hand-built look (vfx.js)
    if (styled) return styled(ctx, a, t);
    const ox = f.x + Math.cos(a.aim) * 18, oy = f.y - f.height * 0.55 + Math.sin(a.aim) * 10;
    const L = a.end ? a.end.d : s.len;
    const k = Math.min(1, a.t / 0.12) * (a.t > a.dur - 0.12 ? Math.max(0, (a.dur - a.t) / 0.12) : 1); // grow / shrink
    const W = s.w * (0.6 + 0.4 * k) * (1 + 0.08 * Math.sin(t * 40));
    ctx.save();
    ctx.translate(ox, oy); ctx.rotate(a.aim);
    ctx.globalCompositeOperation = "lighter";
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    const path = (amp, freq, phase, jag) => {
      ctx.beginPath(); ctx.moveTo(0, 0);
      const n = 18;
      for (let i = 1; i <= n; i++) {
        const u = i / n, x = u * L * k;
        let y = 0;
        if (s.spiral) y = Math.sin(u * 7 - t * 22 + phase) * amp * u;
        else if (s.helix) y = Math.sin(u * 10 - t * 30 + phase) * amp;
        else if (s.jag || jag) y = (U.seeded((i * 31 + Math.floor(t * 24) * 7 + phase * 10) | 0)() - 0.5) * amp * 2;
        else if (s.cone) y = Math.sin(u * 4 - t * 14 + phase) * amp * u * 0.5;
        ctx.lineTo(x, y);
      }
    };
    const spread = s.cone ? 1 + 0.6 * k : 1;
    // outer glow → body → core → white hot line
    ctx.strokeStyle = U.rgba(th.c[2], 0.28); ctx.lineWidth = W * 1.5 * spread; path(W * 0.4, 1, 0); ctx.stroke();
    ctx.strokeStyle = U.rgba(th.c[1], 0.75); ctx.lineWidth = W * 0.85 * spread; path(W * 0.4, 1, 0); ctx.stroke();
    if (s.spiral || s.helix) { ctx.strokeStyle = U.rgba(th.c[1], 0.75); ctx.lineWidth = W * 0.6; path(W * 0.4, 1, Math.PI); ctx.stroke(); }
    ctx.strokeStyle = th.c[0]; ctx.lineWidth = W * 0.32; path(W * 0.3, 1, 0.5); ctx.stroke();
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = W * 0.1; path(W * 0.3, 1, 0.5); ctx.stroke();
    // energy pulses travelling down the beam (retro "segmented" look)
    for (let i = 0; i < 5; i++) {
      const u = ((t * 2.6 + i / 5) % 1), px = u * L * k, pr = W * (0.55 + 0.25 * Math.sin(u * Math.PI));
      ctx.strokeStyle = U.rgba(th.c[0], 0.9 * (1 - u * 0.5)); ctx.lineWidth = 3;
      ctx.beginPath(); ctx.ellipse(px, 0, pr * 0.35, pr, 0, 0, U.TAU); ctx.stroke();
    }
    // muzzle & impact flares
    for (const [px, r] of [[0, W * 0.9], [L * k, W * 1.3]]) {
      const g = ctx.createRadialGradient(px, 0, 0, px, 0, r);
      g.addColorStop(0, "rgba(255,255,255,0.85)"); g.addColorStop(0.4, U.rgba(th.c[1], 0.8)); g.addColorStop(1, U.rgba(th.c[2], 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(px, 0, r, 0, U.TAU); ctx.fill();
    }
    ctx.restore();
    const ex = ox + Math.cos(a.aim) * L * k, ey = oy + Math.sin(a.aim) * L * k;
    flare(ctx, ox, oy, W * 2.2 * (1 + 0.2 * Math.sin(t * 40)), t * 3, "#ffffff");
    flare(ctx, ex, ey, W * 3 * (1 + 0.25 * Math.sin(t * 33)), -t * 4, th.c[0]);
  }

  // Psystrike: red targeting laser while charging, then a fat magenta beam with travelling rings.
  function drawPsybreak(ctx, a, t) {
    const f = a.owner, s = a.spec, L = a.clashEnd ? a.clashEnd.d : (a.len || s.range);
    const ox = f.x + Math.cos(a.aim) * 16, oy = f.y - f.height * 0.5 + Math.sin(a.aim) * 8;
    ctx.save(); ctx.translate(ox, oy); ctx.rotate(a.aim); ctx.globalCompositeOperation = "lighter"; ctx.lineCap = "round";
    if (!a.fired) {
      const fl = 0.75 + 0.25 * Math.sin(t * 50);
      ctx.globalAlpha = fl; ctx.strokeStyle = "#ff2a40"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(L, 0); ctx.stroke();
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "#ff2a40"; ctx.beginPath(); ctx.arc(L, 0, 6 + 3 * Math.sin(t * 30), 0, U.TAU); ctx.fill();
      const ch = Math.min(1, a.t / s.tele), g = ctx.createRadialGradient(0, 0, 0, 0, 0, 20 + ch * 40);
      g.addColorStop(0, "rgba(255,255,255,0.95)"); g.addColorStop(0.4, "rgba(255,110,220,0.8)"); g.addColorStop(1, "rgba(255,60,200,0)");
      ctx.globalAlpha = 1; ctx.fillStyle = g; ctx.beginPath(); ctx.arc(0, 0, 20 + ch * 40, 0, U.TAU); ctx.fill();
      ctx.restore();
      flare(ctx, ox, oy, 30 + 50 * Math.min(1, a.t / s.tele), t * 6, "#ffd0f4");
      return;
    }
    const k = Math.max(0, a.t - (a.ft || 0)), g = Math.min(1, k / 0.07) * (k > s.dur - 0.16 ? Math.max(0, (s.dur - k) / 0.16) : 1);
    const W = Math.max(1, s.w * (0.35 + 0.65 * g) * (1 + 0.06 * Math.sin(t * 50))), Lk = Math.max(0, L * Math.min(1, k / 0.06));
    for (const [m, col] of [[1.7, "rgba(170,30,170,0.35)"], [1.15, "rgba(255,70,210,0.7)"], [0.72, "rgba(255,150,235,0.9)"], [0.36, "#fff0fb"], [0.14, "#ffffff"]]) {
      ctx.fillStyle = col; ctx.fillRect(0, -W * m / 2, Lk, W * m);
      ctx.beginPath(); ctx.arc(0, 0, W * m / 2, 0, U.TAU); ctx.fill();
    }
    // crackling filaments along the edges
    ctx.strokeStyle = "rgba(255,200,250,0.9)"; ctx.lineWidth = 1.5;
    for (const side of [-1, 1]) {
      const rng = U.seeded((a.seed * 100 + Math.floor(t * 30) + side * 7) | 0);
      ctx.beginPath(); ctx.moveTo(0, side * W * 0.55);
      for (let x = 20; x < Lk; x += 22) ctx.lineTo(x, side * W * (0.45 + rng() * 0.35));
      ctx.stroke();
    }
    // spinning rings travelling down the beam
    for (let i = 0; i < 7; i++) {
      const u = (k * 2.4 + i / 7) % 1, px = u * Lk, pr = W * (0.85 + 0.3 * Math.sin(u * Math.PI));
      ctx.strokeStyle = U.rgba("#ffffff", 0.85 * (1 - u * 0.6)); ctx.lineWidth = 3;
      ctx.beginPath(); ctx.ellipse(px, 0, pr * 0.22, pr, 0, 0, U.TAU); ctx.stroke();
    }
    const gr = ctx.createRadialGradient(Lk, 0, 0, Lk, 0, W * 1.8);
    gr.addColorStop(0, "rgba(255,255,255,0.95)"); gr.addColorStop(0.4, "rgba(255,120,230,0.8)"); gr.addColorStop(1, "rgba(255,60,200,0)");
    ctx.fillStyle = gr; ctx.beginPath(); ctx.arc(Lk, 0, W * 1.8, 0, U.TAU); ctx.fill();
    ctx.restore();
    flare(ctx, ox, oy, W * 2.4, t * 4, "#ffffff");
    flare(ctx, ox + Math.cos(a.aim) * Lk, oy + Math.sin(a.aim) * Lk, W * 3, -t * 5, "#ffc6f0");
  }

  // Shadow Ball: black core, violet swirl, a crisp electric-blue rim.
  function drawShadowBall(ctx, a, th, t) {
    const x = a.x, y = a.y - a.z, R = a.r * (a.size || 1) * 0.95;
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const g = ctx.createRadialGradient(x, y, R * 0.6, x, y, R * 2.1);
    g.addColorStop(0, "rgba(90,80,255,0.9)"); g.addColorStop(0.4, "rgba(70,50,200,0.45)"); g.addColorStop(1, "rgba(40,20,120,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, R * 2.1, 0, U.TAU); ctx.fill();
    ctx.restore();
    ctx.save();
    ctx.fillStyle = "#4a6cff"; ctx.beginPath(); ctx.arc(x, y, R + 3, 0, U.TAU); ctx.fill();
    const g2 = ctx.createRadialGradient(x - R * 0.25, y - R * 0.25, 0, x, y, R);
    g2.addColorStop(0, "#3a2470"); g2.addColorStop(0.55, "#170c30"); g2.addColorStop(1, "#06030c");
    ctx.fillStyle = g2; ctx.beginPath(); ctx.arc(x, y, R, 0, U.TAU); ctx.fill();
    ctx.globalCompositeOperation = "lighter"; ctx.lineCap = "round";
    for (let i = 0; i < 3; i++) { // swirl
      const r0 = a.rot * 1.4 + i * (U.TAU / 3);
      ctx.strokeStyle = i ? "rgba(150,110,255,0.75)" : "rgba(220,200,255,0.9)"; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(x, y, R * (0.45 + i * 0.15), r0, r0 + 1.6); ctx.stroke();
    }
    ctx.strokeStyle = "#b8c8ff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, R + 1.5, 0, U.TAU); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.8)"; ctx.beginPath(); ctx.arc(x - R * 0.35, y - R * 0.4, R * 0.14, 0, U.TAU); ctx.fill();
    ctx.restore();
    flare(ctx, x, y, R * (1.6 + 0.4 * Math.sin(t * 25)), a.rot * 0.3, "#8a9cff");
  }

  // Psychic grab tether: a sagging dotted thread of light from the caster's hand to the head / target.
  function drawTether(ctx, a, t) {
    const f = a.owner, e = a.tgt;
    const hx = f.x + Math.cos(f.facing) * 14, hy = f.y - f.height * 0.55;
    const tx = a.x, ty = e ? e.y - e.z - e.height * 0.5 : a.y - 30;
    const d = U.dist(hx, hy, tx, ty), n = Math.max(4, Math.floor(d / 9));
    const nx = -(ty - hy) / (d || 1), ny = (tx - hx) / (d || 1);
    const pt = (u) => { const w = Math.sin(u * Math.PI) * (6 + 4 * Math.sin(t * 9)) + Math.sin(u * 14 - t * 20) * 2.5; return [hx + (tx - hx) * u + nx * w, hy + (ty - hy) * u + ny * w + Math.sin(u * Math.PI) * d * 0.06]; };
    ctx.save(); ctx.globalCompositeOperation = "lighter"; ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(180,140,255,0.35)"; ctx.lineWidth = 7; ctx.beginPath();
    for (let i = 0; i <= n; i++) { const [px, py] = pt(i / n); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke();
    ctx.fillStyle = "#f2eaff";
    for (let i = 0; i <= n; i++) { const [px, py] = pt(i / n); const r = 1.6 + (((i + Math.floor(t * 20)) % 3) === 0 ? 1.2 : 0); ctx.fillRect(px - r, py - r, r * 2, r * 2); }
    const g = ctx.createRadialGradient(tx, ty, 0, tx, ty, e ? e.r * 1.4 : 16);
    g.addColorStop(0, "rgba(255,255,255,0.9)"); g.addColorStop(0.5, "rgba(190,150,255,0.6)"); g.addColorStop(1, "rgba(120,80,255,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(tx, ty, e ? e.r * 1.4 : 16, 0, U.TAU); ctx.fill();
    ctx.restore();
    if (e) { // telekinetic bubble around the held foe
      FXV.at(ctx, e.x, e.y);
      ctx.save(); ctx.globalCompositeOperation = "lighter"; ctx.strokeStyle = "rgba(200,170,255,0.8)"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.ellipse(e.x, e.y - e.z - e.height * 0.45, e.r * 1.5, e.height * 0.7, 0, 0, U.TAU); ctx.stroke(); ctx.restore();
    }
  }

  // Slash arc visual (drawn by FX particle system via shape "slash").
  return { spec, cast, update, draw, drawWindups, windupDanger, warp, BUFF, HEAL, SEC };
})();
