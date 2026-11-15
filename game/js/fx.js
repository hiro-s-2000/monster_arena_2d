// Visual effects: typed particle emitters, floating numbers, rings, dynamic lights, screen shake,
// hit-stop, slow-motion and flashes. Pure presentation — nothing here affects gameplay.
"use strict";

// Per-type particle theme: colours (light → dark), particle shape, additive blending, ground decal.
const THEMES = {
  fire: { c: ["#fff2a8", "#ffb347", "#ff6a1a", "#b3240a"], shape: "flame", add: true, decal: "scorch", grav: -260 },
  water: { c: ["#e6fbff", "#8fdcff", "#3aa0ff", "#1552b8"], shape: "drop", add: false, decal: "puddle", grav: 520 },
  electric: { c: ["#ffffff", "#fff7a0", "#ffe23a", "#c89b00"], shape: "bolt", add: true, decal: "scorch", grav: 0 },
  grass: { c: ["#e8ffb0", "#9cf26a", "#4caf3a", "#1f6b1e"], shape: "leaf", add: false, decal: null, grav: 140 },
  ice: { c: ["#ffffff", "#dcf8ff", "#8fd8ff", "#3f8fbf"], shape: "shard", add: true, decal: "frost", grav: 380 },
  fighting: { c: ["#fff1e0", "#ffb089", "#ff5a3c", "#8a1a10"], shape: "spark", add: true, decal: "crack", grav: 300 },
  poison: { c: ["#f7d5ff", "#d98bf0", "#9b3fc9", "#4a1a63"], shape: "bubble", add: false, decal: "stain", grav: -90 },
  ground: { c: ["#f5e0b0", "#c9a66b", "#8a6a3a", "#4a3416"], shape: "rock", add: false, decal: "crater", grav: 700 },
  flying: { c: ["#ffffff", "#e9e6ff", "#bdb2ff", "#7a68d6"], shape: "streak", add: true, decal: null, grav: -40 },
  psychic: { c: ["#ffffff", "#ffc6e4", "#ff5fa8", "#8a1e5a"], shape: "orb", add: true, decal: null, grav: -60 },
  bug: { c: ["#f4ffb0", "#cde86a", "#8aa32a", "#4a5a0f"], shape: "spark", add: true, decal: null, grav: 120 },
  rock: { c: ["#ece4d4", "#b4a892", "#7c6e5a", "#3e352a"], shape: "rock", add: false, decal: "crater", grav: 760 },
  ghost: { c: ["#e9dcff", "#b89aff", "#6d47b8", "#2a1750"], shape: "wisp", add: true, decal: "stain", grav: -120 },
  dragon: { c: ["#ffffff", "#b8a6ff", "#6a3cff", "#2c117a"], shape: "orb", add: true, decal: "scorch", grav: -40 },
  dark: { c: ["#8a7aa0", "#4a3d5e", "#221a2e", "#000000"], shape: "smoke", add: false, decal: "stain", grav: -80 },
  steel: { c: ["#ffffff", "#e8e8f5", "#aeb0c8", "#5c5e78"], shape: "spark", add: true, decal: null, grav: 500 },
  fairy: { c: ["#ffffff", "#ffe3ef", "#ff9ec4", "#d64f8a"], shape: "star", add: true, decal: null, grav: -60 },
  normal: { c: ["#ffffff", "#fff4d6", "#ffd98a", "#a67c2e"], shape: "star", add: true, decal: null, grav: 200 },
};

// [ユーザー指定 2026-11-16] 技のエフェクトの 3D 化（HD-2D）。3D 表示のとき、空中のエフェクト（ビーム・弾・爆発・粒子・空中の輪）は
// 地面に貼らず、「画面に正対して立つ板」として別のキャンバス（画面の座標）に描く。描く側のコードは今までどおり world 座標
// （x, y − 高さ）で描き、その直前に FXV.at / FXV.seg で「その world 座標が画面のどこに来るか」の変換を ctx に入れる。
//   FXV.at(ctx, x, y)                    足もとが地面の点 (x, y) にある板。上（y − 高さ）は画面の上へ、その奥行きの倍率で伸びる。
//   FXV.seg(ctx, x0, y0, h0, x1, y1, h1)  地面の 2 点（それぞれ高さ h）を結ぶ線（ビーム）。線の向きと長さは遠近どおり、太さは奥行きの倍率。
// FXV.p3 = いまのフレームが 3D か、FXV.on = いま空中の層を描いているか（平面表示では両方 false で、何もしない）。
// 地面のエフェクト（予告範囲・地面の輪・影）は地面の面にそのまま乗る。平面表示用の「たてつぶし」は FXV.gsq(v) で 3D のとき 1 に戻す。
const FXV = {
  p3: false, on: false, q: 1,
  gsq(v) { return this.p3 ? 1 : v; },
  pt(x, y, h = 0) {
    const K = Arenas.K, p = Scene3D.project(x / K, y / K, 0), m = p.s / K / this.q;
    return { x: p.x / this.q, y: p.y / this.q - h * m, m };
  },
  at(ctx, x, y) {
    if (!this.on) return;
    const p = this.pt(x, y);
    ctx.setTransform(p.m, 0, 0, p.m, p.x - x * p.m, p.y - y * p.m);
  },
  seg(ctx, x0, y0, h0, x1, y1, h1) {
    if (!this.on) return;
    const a = this.pt(x0, y0, h0), b = this.pt(x1, y1, h1);
    const ax = x0, ay = y0 - h0, ux = x1 - x0, uy = (y1 - h1) - ay, la = Math.hypot(ux, uy), vx = b.x - a.x, vy = b.y - a.y, ls = Math.hypot(vx, vy);
    if (la < 1e-3 || ls < 1e-3) { ctx.setTransform(a.m, 0, 0, a.m, a.x - ax * a.m, a.y - ay * a.m); return; }
    const m = (a.m + b.m) / 2, k = ls / la, ca = ux / la, sa = uy / la, cs = vx / ls, ss = vy / ls;
    const A = cs * k * ca + ss * m * sa, C = cs * k * sa - ss * m * ca, B = ss * k * ca - cs * m * sa, D = ss * k * sa + cs * m * ca;
    ctx.setTransform(A, B, C, D, a.x - (A * ax + C * ay), a.y - (B * ax + D * ay));
  },
};

class FX {
  constructor(arena) {
    this.arena = arena;
    this.parts = [];
    this.floaters = [];
    this.rings = [];
    this.lights = [];
    this.shake = 0; this.shakeAmp = 0;
    this.hitstop = 0;
    this.slow = 0; this.slowRate = 1;
    this.flash = 0; this.flashColor = "#ffffff";
    this.time = 0;
    this.zoomPunch = 0;
    this.circles = [];
    this.invert = 0; this.invertN = 0; // retro negative-flash strobe (frames)
    this.blur = 0; this.blurX = 0; this.blurY = 0; // radial impact blur centred on a world point
  }

  // [ユーザー指定 2026-11-16] 技の炎は使い手の正面スプライトの色。橙系の体・色の無い体は共有の炎のまま。
  // 使い手を渡さない呼び出し（やけど、溶岩、登場）も橙のまま。
  // debug.html は file:// で開くと絵の色を読めないので、data/flame_hues.js の色相を先に使う。
  static flameOf(owner) {
    const sp = owner && owner.species;
    if (!sp) return null;
    const key = owner.shiny ? "_flameSh" : "_flame";
    if (Object.prototype.hasOwnProperty.call(sp, key)) return sp[key];
    const table = window.ARENA_FLAME_HUES;
    if (table) {
      const id = owner.shiny ? sp.slug + "#s" : sp.slug;
      const hue = Object.prototype.hasOwnProperty.call(table, id) ? table[id] : table[sp.slug];
      const pal = hue == null || FX._keepOrange(hue) ? null : FX._flameRamp(hue);
      sp[key] = pal;
      return pal;
    }
    let rec = owner.sprites && owner.sprites.front;
    if (!rec && typeof Assets !== "undefined" && Assets.front) {
      try { rec = Assets.front(sp); } catch (e) { return null; }
    }
    if (!rec || !rec.ready || rec.failed || !rec.frames || !rec.frames[0]) return null;
    const canvas = rec.frames[0].canvas;
    if (!canvas || !canvas.getContext || !canvas.width) return null;
    let hue = null;
    try { hue = FX._hueOf(canvas); } catch (e) { return null; } // tainted (file://): do not cache the miss
    const pal = hue == null || FX._keepOrange(hue) ? null : FX._flameRamp(hue);
    sp[key] = pal;
    return pal;
  }
  // Dominant hue of saturated opaque pixels. null = grey body, keep the classic orange fire.
  static _hueOf(canvas) {
    const w = canvas.width, h = canvas.height, data = canvas.getContext("2d").getImageData(0, 0, w, h).data;
    const bins = new Uint32Array(36);
    const step = w * h > 90000 ? 2 : 1;
    let n = 0;
    for (let y = 0; y < h; y += step) {
      const row = y * w;
      for (let x = 0; x < w; x += step) {
        const i = (row + x) << 2;
        if (data[i + 3] < 128) continue;
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const max = r > g ? (r > b ? r : b) : (g > b ? g : b);
        const min = r < g ? (r < b ? r : b) : (g < b ? g : b);
        if (max < 40 || min > 230) continue;
        const d = max - min;
        if (d < 28 || d / max < 0.28) continue;
        let hue = max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
        if (hue < 0) hue += 6;
        bins[(hue * 6) | 0] += 1;
        n++;
      }
    }
    if (n < 24) return null;
    let best = 0;
    for (let i = 1; i < 36; i++) if (bins[i] > bins[best]) best = i;
    if (bins[best] < n * 0.18) return null;
    return (best * 10 + 5) % 360;
  }
  // Orange / red bodies stay on the shared FIRE hexes (Charmander, Charizard, Magmar).
  static _keepOrange(h) { return h <= 62 || h >= 348; }
  static _hsl(h, s, l) {
    h = ((h % 360) + 360) % 360;
    const c = (1 - Math.abs(2 * l - 1)) * s, hp = h / 60, x = c * (1 - Math.abs(hp % 2 - 1));
    let r = 0, g = 0, b = 0;
    if (hp < 1) { r = c; g = x; }
    else if (hp < 2) { r = x; g = c; }
    else if (hp < 3) { g = c; b = x; }
    else if (hp < 4) { g = x; b = c; }
    else if (hp < 5) { r = x; b = c; }
    else { r = c; b = x; }
    const m = l - c / 2;
    const hex = (v) => Math.round(Math.max(0, Math.min(1, v + m)) * 255).toString(16).padStart(2, "0");
    return "#" + hex(r) + hex(g) + hex(b);
  }
  static _rgb(hex) { const n = parseInt(hex.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  // Five-stop flame. Blue hues keep a light-blue core — shifting the hot stop toward yellow turns them green.
  static _flameRamp(h) {
    const blue = h >= 150 && h <= 270;
    const hotH = blue ? h : h + 16, edgeH = blue ? h : h - 16, darkH = blue ? h : h - 24, coreH = blue ? h : h + 12;
    const H = FX._hsl;
    const dark = H(darkH, 0.78, 0.16), edge = H(edgeH, 0.9, 0.42), mid = H(h, 0.96, 0.54);
    const hot = H(hotH, blue ? 0.62 : 0.9, blue ? 0.74 : 0.64);
    const core = H(coreH, blue ? 0.38 : 0.48, 0.93);
    const light = H(blue ? h : h + 6, 0.88, 0.66);
    const [sr, sg, sb] = FX._rgb(dark);
    const smoke = Math.round(sr * 0.7) + "," + Math.round(sg * 0.65) + "," + Math.round(sb * 0.7);
    return {
      dark, edge, mid, hot, core, smoke, light,
      c: [H(coreH, blue ? 0.32 : 0.62, 0.93), light, H(blue ? h : h - 8, 0.94, 0.5), H(darkH, 0.8, 0.3)],
      smokeHex: H(h, 0.22, 0.16), tipSmoke: H(h, 0.18, 0.22),
      flash: H(coreH, blue ? 0.28 : 0.45, 0.94), glow: H(hotH, blue ? 0.55 : 0.8, 0.7), mouth: core,
      smokeA: smoke, smokeB: Math.round(sr * 1.15) + "," + Math.round(sg * 1.05) + "," + Math.round(sb * 1.1),
    };
  }

  theme(t, owner) {
    const base = THEMES[t] || THEMES.normal;
    if (t !== "fire" || !owner) return base;
    const pal = FX.flameOf(owner);
    if (!pal) return base;
    return { c: pal.c, shape: base.shape, add: base.add, decal: base.decal, grav: base.grav };
  }

  // ---- particles ---------------------------------------------------------------------------------
  add(p) {
    if (this.parts.length > 1800) this.parts.splice(0, 200);
    p.life = p.life || 0.6; p.max = p.life; p.size = p.size || 6; p.vx = p.vx || 0; p.vy = p.vy || 0; p.vz = p.vz || 0; p.z = p.z || 0;
    p.drag = p.drag === undefined ? 1.5 : p.drag; p.grav = p.grav || 0; p.rot = p.rot || U.rand(U.TAU); p.spin = p.spin || 0;
    this.parts.push(p);
    return p;
  }

  // Impact burst at (x, y): n particles of the type's theme flying outward (dir biased if given).
  burst(type, x, y, n = 18, opts = {}) {
    const th = this.theme(type, opts.owner), sz = opts.size || 1, spd = opts.speed || 1;
    for (let i = 0; i < n; i++) {
      const a = opts.dir !== undefined ? opts.dir + U.rand(-(opts.spread || 0.9), opts.spread || 0.9) : U.rand(U.TAU);
      const s = U.rand(60, 260) * spd;
      this.add({ x, y, z: 20 + U.rand(30), vx: Math.cos(a) * s, vy: Math.sin(a) * s * 0.6, vz: U.rand(40, 200) * (th.grav < 0 ? 1 : 0.6),
        grav: th.grav, life: U.rand(0.35, 0.8), size: U.rand(4, 11) * sz, color: U.pick(th.c), shape: th.shape, add: th.add, spin: U.rand(-6, 6) });
    }
    if (th.shape === "flame" || th.shape === "smoke" || opts.smoke) for (let i = 0; i < n / 3; i++) {
      this.add({ x: x + U.rand(-10, 10), y: y + U.rand(-6, 6), z: 20, vx: U.rand(-30, 30), vy: U.rand(-20, 20), vz: U.rand(30, 90), life: U.rand(0.6, 1.1), size: U.rand(8, 14) * sz, color: this.arena && this.arena.water ? "#dff2ff" : "#7a7076", shape: "smoke", add: false, grow: 1.5, alpha: 0.28 });
    }
    if (opts.light !== false) this.light(x, y, 90 * sz + 40, th.c[1], 0.9, 0.25);
  }

  // Short-lived trailing puff behind a projectile.
  trail(type, x, y, z, n = 2, size = 1, owner) {
    const th = this.theme(type, owner);
    for (let i = 0; i < n; i++) this.add({ x: x + U.rand(-4, 4), y: y + U.rand(-4, 4), z: z + U.rand(-4, 4), vx: U.rand(-20, 20), vy: U.rand(-20, 20), vz: th.grav < 0 ? U.rand(20, 60) : U.rand(-10, 30),
      life: U.rand(0.2, 0.45), size: U.rand(4, 9) * size, color: U.pick(th.c), shape: th.shape === "rock" ? "smoke" : th.shape, add: th.add, drag: 3 });
  }

  // Charge-up: particles converge onto (x, y).
  charge(type, x, y, n = 10, radius = 60, owner) {
    const th = this.theme(type, owner);
    for (let i = 0; i < n; i++) {
      const a = U.rand(U.TAU), d = radius * U.rand(0.6, 1.2);
      const px = x + Math.cos(a) * d, py = y + Math.sin(a) * d * 0.6;
      this.add({ x: px, y: py, z: U.rand(10, 50), vx: (x - px) * 3.2, vy: (y - py) * 3.2, vz: 0, life: 0.32, size: U.rand(3, 6), color: U.pick(th.c), shape: th.shape === "rock" ? "spark" : th.shape, add: true, drag: 0 });
    }
  }

  // Aura ring around a fighter (buffs) — particles rising in a circle.
  aura(x, y, color, n = 14, r = 26) {
    for (let i = 0; i < n; i++) {
      const a = (i / n) * U.TAU + U.rand(0.3);
      this.add({ x: x + Math.cos(a) * r, y: y + Math.sin(a) * r * 0.5, z: 0, vx: 0, vy: 0, vz: U.rand(70, 140), life: U.rand(0.6, 1), size: U.rand(3, 7), color, shape: "orb", add: true, drag: 0.2 });
    }
  }

  debris(x, y, n, c1, c2) {
    for (let i = 0; i < n; i++) this.add({ x, y, z: 10, vx: U.rand(-220, 220), vy: U.rand(-120, 120), vz: U.rand(120, 360), grav: 800, life: U.rand(0.6, 1.2), size: U.rand(4, 9), color: U.chance(0.5) ? c1 : c2, shape: "rock", spin: U.rand(-10, 10), bounce: true });
  }

  dust(x, y, n = 4, color = "rgba(200,180,140,0.5)") {
    for (let i = 0; i < n; i++) this.add({ x: x + U.rand(-8, 8), y: y + U.rand(-3, 3), z: 0, vx: U.rand(-40, 40), vy: U.rand(-10, 10), vz: U.rand(10, 40), life: U.rand(0.3, 0.6), size: U.rand(5, 10), color, shape: "smoke", alpha: 0.5, grow: 1.5 });
  }

  ring(x, y, r0, r1, life, color, width = 4, opts = {}) {
    this.rings.push({ x, y, r0, r1, life, max: life, color, width, add: opts.add !== false, ground: opts.ground !== false, z: opts.z || 0, squash: opts.squash === undefined ? 0.55 : opts.squash });
  }

  light(x, y, r, color, i = 1, life = 0.3, opts = {}) {
    if (this.lights.length > 80) this.lights.shift();
    this.lights.push({ x, y, r, color, i, life, max: life, flicker: opts.flicker });
  }

  // ---- retro "spell" effects ----------------------------------------------------------------------
  // Hit spark: a spiky pixel starburst + speed lines + staggered shock rings. `big` for heavy / crit hits.
  impact(type, x, y, z, big = false, dir, owner) {
    const th = this.theme(type, owner), k = big ? 1.6 : 1;
    this.add({ x, y, z, life: big ? 0.2 : 0.14, size: 30 * k, color: th.c[1], shape: "impact", add: true, drag: 0, rot: U.rand(U.TAU), spikes: big ? 10 : 8 });
    this.add({ x, y, z, life: big ? 0.24 : 0.16, size: 60 * k, color: "#ffffff", shape: "speedlines", add: true, drag: 0, rot: U.rand(U.TAU), n: big ? 14 : 9 });
    this.ring(x, y, 6, 70 * k, 0.28, "#ffffff", 5, { ground: false, z, squash: 0.8 });
    this.ring(x, y, 4, 110 * k, 0.42, th.c[1], 7, { ground: false, z, squash: 0.8 });
    if (big) this.ring(x, y, 10, 170, 0.55, th.c[2], 9);
    // chunky square sparks that fall with gravity (8-bit debris)
    for (let i = 0; i < (big ? 18 : 10); i++) {
      const a = dir !== undefined ? dir + U.rand(-1.1, 1.1) : U.rand(U.TAU), sp = U.rand(160, 460) * k;
      this.add({ x, y, z, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp * 0.6, vz: U.rand(80, 300), grav: 900, life: U.rand(0.3, 0.6), size: U.rand(4, 8), color: U.pick([th.c[0], th.c[1], "#ffffff"]), shape: "pixel", add: true, drag: 2.5 });
    }
    this.light(x, y, big ? 260 : 160, th.c[1], 1.1, big ? 0.35 : 0.2);
  }
  // Wind-up aura under a caster: particles streaming radially outward in a ground-level circle.
  magicCircle(x, y, r, color, life = 0.5, owner) {
    this.circles.push({ x, y, r, color, life, max: life, owner, rot: U.rand(U.TAU) });
  }
  // Explosion column: impact + tall flash pillar + dust ring. For big landings / strikes.
  blast(type, x, y, scale = 1, owner) {
    const th = this.theme(type, owner);
    this.impact(type, x, y, 30, true, undefined, owner);
    this.add({ x, y, z: 0, life: 0.35, size: 70 * scale, color: th.c[0], shape: "column", add: true, drag: 0 });
    this.ring(x, y, 20, 220 * scale, 0.6, th.c[1], 10);
    this.burst(type, x, y, 26, { size: 1.4 * scale, speed: 1.4, owner });
    this.shakeCam(14 * scale, 0.35);
  }
  // One short negative-image flash (n frames at 30 Hz) — single pulse, never a repeating strobe.
  strobe(frames = 2) { if (this.invertCd > 0) return; this.invertN = Math.max(this.invertN, frames); this.invertCd = 0.6; }
  radialBlur(x, y, k = 0.6) { this.blur = Math.max(this.blur, k); this.blurX = x; this.blurY = y; }
  // Recall laser pulling a fighter back into its trainer's hand along a converging particle stream.
  recallBeam(x, y, h, tx, ty, tz = 30, color = "#ff5578") {
    const z0 = Math.max(24, h * 0.45);
    this.add({ x: tx, y: ty, z: tz, x2: x, y2: y, z2: z0, life: 0.34, size: 14, color, shape: "beamline", add: true, drag: 0 });
    this.ring(x, y, Math.max(20, h * 0.45), 6, 0.34, color, 5);
    this.ring(x, y, 8, Math.max(28, h * 0.6), 0.22, "#ffffff", 4, { ground: false, z: z0, squash: 0.75 });
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * U.TAU + U.rand(-0.2, 0.2), rr = U.rand(8, 34), life = U.rand(0.22, 0.36);
      const sx = x + Math.cos(a) * rr, sy = y + Math.sin(a) * rr * 0.55, sz = U.rand(10, Math.max(28, h * 0.8));
      this.add({ x: sx, y: sy, z: sz, vx: (tx - sx) / life, vy: (ty - sy) / life, vz: (tz - sz) / life, life, size: U.rand(6, 12), color: i % 3 ? color : "#ffffff", shape: i % 3 ? "orb" : "star", add: true, drag: 0 });
    }
    this.add({ x: tx, y: ty, z: tz, life: 0.28, size: 22, color: "#ffffff", shape: "orb", add: true, drag: 0 });
    this.light(x, y, 170, color, 0.9, 0.32);
    this.light(tx, ty, 130, color, 0.95, 0.28);
  }
  // Sparkling wake emitted along a thrown Poké Ball's flight arc.
  ballTrail(x, y, z, vx = 0, vy = 0, vz = 0, color = "#ff5a7a") {
    this.add({ x: x + U.rand(-4, 4), y: y + U.rand(-3, 3), z, vx: -vx * 0.14 + U.rand(-18, 18), vy: -vy * 0.14 + U.rand(-12, 12), vz: -vz * 0.12 + U.rand(10, 35), life: 0.22, size: U.rand(9, 14), color, shape: "orb", add: true, drag: 1.5 });
    this.add({ x, y, z, vx: -vx * 0.22, vy: -vy * 0.22, vz: -vz * 0.18, life: 0.16, size: 7, color: "#ffffff", shape: "streak", add: true, drag: 1 });
    if (Math.random() < 0.55) {
      this.add({ x: x + U.rand(-8, 8), y: y + U.rand(-4, 4), z: z + U.rand(-4, 8), vx: U.rand(-30, 30), vy: U.rand(-20, 20), vz: U.rand(20, 60), life: 0.26, size: U.rand(4, 7), color: "#ffffff", shape: "star", add: true, drag: 2, rot: U.rand(U.TAU), spin: U.rand(-7, 7) });
    }
  }
  // High-impact light pillar + shock rings + sparkles when a Poké Ball pops open mid-air.
  ballBurst(x, y, type = "normal", big = false) {
    const th = this.theme(type), k = big ? 1.35 : 1;
    this.add({ x, y, z: 0, life: 0.32 * k, size: 48 * k, color: th.c[1], shape: "column", add: true, drag: 0 });
    this.add({ x, y, z: 34, life: 0.2, size: 32 * k, color: "#ffffff", shape: "impact", add: true, drag: 0, rot: U.rand(U.TAU), spikes: big ? 10 : 8 });
    this.add({ x, y, z: 34, life: 0.24, size: 58 * k, color: "#ffffff", shape: "speedlines", add: true, drag: 0, rot: U.rand(U.TAU), n: big ? 14 : 10 });
    this.ring(x, y, 8, 115 * k, 0.42, "#ffffff", 6);
    this.ring(x, y, 6, 150 * k, 0.52, th.c[1], 7);
    this.ring(x, y, 6, 95 * k, 0.34, "#ffffff", 5, { ground: false, z: 36, squash: 0.82 });
    const n = big ? 20 : 12;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * U.TAU + U.rand(-0.2, 0.2), sp = U.rand(130, 340) * k;
      this.add({ x, y, z: 32, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp * 0.6, vz: U.rand(40, 210), grav: 480, life: U.rand(0.28, 0.48), size: U.rand(6, 12), color: i % 2 ? "#ffffff" : th.c[i % 3 ? 1 : 0], shape: i % 3 ? "orb" : "star", add: true, drag: 2.2, rot: U.rand(U.TAU), spin: U.rand(-8, 8) });
    }
    this.flashScreen(big ? 0.32 : 0.2, "#ffffff");
    this.light(x, y, big ? 270 : 195, th.c[1], 1.15, 0.42);
  }

  // ---- floaters ----------------------------------------------------------------------------------
  number(x, y, value, opts = {}) {
    // stack numbers that land on the same spot within a short window so beam ticks stay readable
    for (const o of this.floaters) if (o.kind === "num" && o.max - o.life < 0.35 && Math.abs(o.x - x) < 40 && Math.abs(o.y - y) < 30) y -= 18;
    const f = { x: x + U.rand(-10, 10), y, text: String(value), life: opts.life || 1.1, max: opts.life || 1.1, vy: -70, vx: U.rand(-15, 15), size: opts.size || 22, color: opts.color || "#ffffff", stroke: opts.stroke || "#1a1a1a", kind: "num", scalePop: 1 };
    if (opts.crit) { f.size *= 1.3; f.color = "#ffd54a"; }
    this.floaters.push(f);
    return f;
  }
  label(x, y, text, color = "#ffd54a", opts = {}) {
    this.floaters.push({ x, y, text, life: opts.life || 1.3, max: opts.life || 1.3, vy: -34, vx: 0, size: opts.size || 12, color, stroke: "#101010", kind: "label", delay: opts.delay || 0 });
  }

  // ---- camera feel -------------------------------------------------------------------------------
  shakeCam(amp, t = 0.25) { this.shakeAmp = Math.max(this.shakeAmp, amp); this.shake = Math.max(this.shake, t); }
  // ヒットストップの重さ。全呼び出しに掛かる倍率（1 = 旧来の 0.036 / 0.052 / 0.072 s、K.O. 0.14 s）。URL ?hitstop=倍率 でも変えられる。
  stop(t) { this.hitstop = Math.max(this.hitstop, Math.min(0.42, t * FX.HITSTOP_MUL)); }
  slowmo(t, rate = 0.25) { this.slow = Math.max(this.slow, t); this.slowRate = rate; }
  flashScreen(a = 0.5, color = "#ffffff") { this.flash = Math.max(this.flash, a); this.flashColor = color; }

  // ---- update ------------------------------------------------------------------------------------
  update(dt) {
    this.time += dt;
    const P = this.parts;
    for (let i = P.length - 1; i >= 0; i--) {
      const p = P[i];
      p.life -= dt;
      if (p.life <= 0) { P[i] = P[P.length - 1]; P.pop(); continue; }
      if (p.hover) { // telekinetically held: bob in place until `fallAt`, then drop
        if (p.life > p.fallAt) { p.z = U.damp(p.z, p.hz + Math.sin(this.time * 2.2 + p.rot * 3) * 4, 2.2, dt); p.rot += p.spin * dt; p.x += p.vx * dt; p.y += p.vy * dt; continue; }
        p.hover = false; p.grav = 1400; p.vz = 0;
      }
      p.vz -= p.grav * dt;
      const k = Math.max(0, 1 - p.drag * dt);
      p.vx *= k; p.vy *= k;
      p.x += p.vx * dt; p.y += p.vy * dt; p.z += p.vz * dt;
      p.rot += p.spin * dt;
      if (p.shape === "mote") { if (p.z < 0) { p.z = 0; p.vz = 0; p.life = Math.min(p.life, 0.4); } continue; }
      if (p.z < 0) {
        if (p.bounce && Math.abs(p.vz) > 40) { p.z = 0; p.vz = -p.vz * 0.45; p.vx *= 0.7; p.vy *= 0.7; }
        else if (p.shape === "drop" || p.shape === "rock" || p.shape === "shard" || p.shape === "chunk") { p.z = 0; p.vz = 0; p.vx *= 0.3; p.vy *= 0.3; p.life = Math.min(p.life, 0.12); }
        else p.z = 0;
      }
      if (p.flutter) { p.x += Math.sin(this.time * 9 + p.rot) * 40 * dt; }
    }
    for (let i = this.floaters.length - 1; i >= 0; i--) {
      const f = this.floaters[i];
      if (f.delay > 0) { f.delay -= dt; continue; }
      f.life -= dt;
      if (f.life <= 0) { this.floaters.splice(i, 1); continue; }
      const age = f.max - f.life;
      f.x += f.vx * dt; f.y += f.vy * dt * (age < 0.3 ? 1.6 : 0.5);
    }
    for (let i = this.rings.length - 1; i >= 0; i--) { const r = this.rings[i]; r.life -= dt; if (r.life <= 0) this.rings.splice(i, 1); }
    for (let i = this.lights.length - 1; i >= 0; i--) { const l = this.lights[i]; l.life -= dt; if (l.life <= 0) this.lights.splice(i, 1); }
    if (this.shake > 0) { this.shake -= dt; if (this.shake <= 0) this.shakeAmp = 0; }
    if (this.hitstop > 0) this.hitstop -= dt;
    if (this.slow > 0) this.slow -= dt;
    if (this.flash > 0) this.flash = Math.max(0, this.flash - dt * 2.2);
    if (this.zoomPunch > 0) this.zoomPunch = Math.max(0, this.zoomPunch - dt * 1.6);
    for (let i = this.circles.length - 1; i >= 0; i--) {
      const c = this.circles[i]; c.life -= dt;
      if (c.owner) { c.x = c.owner.x; c.y = c.owner.y; }
      if (c.life <= 0) { this.circles.splice(i, 1); continue; }
      let n = dt * 90;
      while (n > 0) {
        if (Math.random() < n) {
          const a = U.rand(U.TAU), r0 = c.r * U.rand(0.15, 0.35), sp = c.r * U.rand(1.8, 2.8);
          this.add({ x: c.x + Math.cos(a) * r0, y: c.y + Math.sin(a) * r0 * 0.55, z: U.rand(2, 10), vx: Math.cos(a) * sp, vy: Math.sin(a) * sp * 0.55, vz: U.rand(10, 50),
            life: U.rand(0.25, 0.4), size: U.rand(3, 6), color: Math.random() < 0.3 ? "#ffffff" : c.color, shape: "pixel", add: true, drag: 2.2 });
        }
        n -= 1;
      }
    }
    if (this.invertCd > 0) this.invertCd -= dt;
    if (this.invertN > 0) { this.invert = 1; this.invertTick = (this.invertTick || 0) + dt; if (this.invertTick > 1 / 30) { this.invertTick = 0; this.invertN--; } } else this.invert = 0;
    if (this.blur > 0) this.blur = Math.max(0, this.blur - dt * 3.2);
  }

  // ---- drawing -----------------------------------------------------------------------------------
  drawRings(ctx, ground) {
    for (const r of this.rings) {
      if (r.ground !== ground) continue;
      const t = 1 - r.life / r.max, rad = U.lerp(r.r0, r.r1, U.easeOut(t));
      ctx.save();
      ctx.globalAlpha = (1 - t) * 0.9;
      ctx.globalCompositeOperation = r.add ? "lighter" : "source-over";
      ctx.strokeStyle = r.color; ctx.lineWidth = r.width * (1 - t * 0.5);
      if (!ground) FXV.at(ctx, r.x, r.y);
      ctx.beginPath(); ctx.ellipse(r.x, r.y - r.z, rad, rad * (ground ? FXV.gsq(r.squash) : r.squash), 0, 0, U.TAU); ctx.stroke();
      ctx.restore();
    }
  }

  drawParticles(ctx, layer) { // layer: "ground" (z≈0 shadows/puddles) or "air"
    ctx.save();
    let lastAdd = null;
    for (const p of this.parts) {
      const t = p.life / p.max;
      const isGround = p.shape === "smoke" && p.z <= 0.5 && p.alpha !== undefined ? false : false;
      void isGround;
      if (layer === "ground") continue;
      const add = !!p.add;
      if (add !== lastAdd) { ctx.globalCompositeOperation = add ? "lighter" : "source-over"; lastAdd = add; }
      const sz = p.size * (p.grow ? 1 + (1 - t) * (p.grow - 1) : (p.shape === "flame" || p.shape === "orb" || p.shape === "wisp" ? 0.4 + t * 0.6 : 1));
      const x = p.x, y = p.y - p.z;
      if (FXV.on) { if (p.shape === "beamline") FXV.seg(ctx, p.x, p.y, p.z, p.x2 !== undefined ? p.x2 : p.x, p.y2 !== undefined ? p.y2 : p.y, p.z2 !== undefined ? p.z2 : 0); else FXV.at(ctx, p.x, p.y); }
      ctx.globalAlpha = (p.alpha !== undefined ? p.alpha : 1) * (t < 0.35 ? t / 0.35 : 1);
      ctx.fillStyle = p.color; ctx.strokeStyle = p.color;
      switch (p.shape) {
        case "flame": case "orb": case "wisp": {
          const g = ctx.createRadialGradient(x, y, 0, x, y, sz);
          g.addColorStop(0, p.shape === "wisp" ? p.color : "#ffffff"); g.addColorStop(0.35, p.color); g.addColorStop(1, "rgba(0,0,0,0)");
          ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, sz, 0, U.TAU); ctx.fill(); break;
        }
        case "smoke": {
          const g = ctx.createRadialGradient(x, y, 0, x, y, sz);
          g.addColorStop(0, p.color); g.addColorStop(1, "rgba(0,0,0,0)");
          ctx.globalAlpha *= 0.7; ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, sz, 0, U.TAU); ctx.fill(); break;
        }
        case "spark": case "streak": {
          const l = Math.hypot(p.vx, p.vy) * 0.05 + sz;
          const a = Math.atan2(p.vy - p.vz * 0.5, p.vx);
          ctx.lineWidth = p.shape === "streak" ? 2 : sz * 0.4; ctx.lineCap = "round";
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - Math.cos(a) * l, y - Math.sin(a) * l); ctx.stroke(); break;
        }
        case "drop": ctx.beginPath(); ctx.ellipse(x, y, sz * 0.4, sz * 0.7, Math.atan2(p.vy - p.vz, p.vx) + Math.PI / 2, 0, U.TAU); ctx.fill(); break;
        case "bubble": ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, sz * 0.6, 0, U.TAU); ctx.stroke(); ctx.globalAlpha *= 0.4; ctx.fill(); break;
        case "leaf": ctx.save(); ctx.translate(x, y); ctx.rotate(p.rot); ctx.beginPath(); ctx.ellipse(0, 0, sz, sz * 0.4, 0, 0, U.TAU); ctx.fill(); ctx.restore(); break;
        case "shard": ctx.save(); ctx.translate(x, y); ctx.rotate(p.rot); ctx.beginPath(); ctx.moveTo(0, -sz); ctx.lineTo(sz * 0.4, sz * 0.5); ctx.lineTo(-sz * 0.4, sz * 0.5); ctx.closePath(); ctx.fill(); ctx.restore(); break;
        case "rock": ctx.save(); ctx.translate(x, y); ctx.rotate(p.rot); ctx.beginPath(); ctx.moveTo(-sz * 0.6, -sz * 0.3); ctx.lineTo(sz * 0.2, -sz * 0.6); ctx.lineTo(sz * 0.6, sz * 0.1); ctx.lineTo(sz * 0.1, sz * 0.6); ctx.lineTo(-sz * 0.5, sz * 0.4); ctx.closePath(); ctx.fill(); ctx.restore(); break;
        case "bolt": {
          ctx.lineWidth = 2.2; ctx.lineCap = "round"; ctx.beginPath(); ctx.moveTo(x, y);
          let px = x, py = y, a = p.rot;
          for (let k = 0; k < 4; k++) { a += U.rand(-1.2, 1.2); px += Math.cos(a) * sz; py += Math.sin(a) * sz; ctx.lineTo(px, py); }
          ctx.stroke(); break;
        }
        case "star": {
          ctx.save(); ctx.translate(x, y); ctx.rotate(p.rot); ctx.beginPath();
          for (let k = 0; k < 8; k++) { const r = k % 2 ? sz * 0.35 : sz; const a = (k / 8) * U.TAU; ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r); }
          ctx.closePath(); ctx.fill(); ctx.restore(); break;
        }
        case "slash": { // melee arc: a crescent sweeping across p.arc radians around p.rot
          const prog = 1 - t, a0 = p.rot - (p.arc || 1.3) / 2 * p.dir, a1 = p.rot + (p.arc || 1.3) / 2 * p.dir;
          const sweep = U.lerp(a0, a1, Math.min(1, prog * 1.6));
          ctx.save(); ctx.translate(x, y); ctx.scale(1, 0.62);
          ctx.strokeStyle = p.color; ctx.lineWidth = sz * 0.22 * t; ctx.lineCap = "round";
          ctx.beginPath(); ctx.arc(0, 0, sz, Math.min(a0, sweep), Math.max(a0, sweep)); ctx.stroke();
          ctx.strokeStyle = "#ffffff"; ctx.lineWidth = sz * 0.08 * t; ctx.stroke();
          ctx.restore(); break;
        }
        case "mote": {
          if (p.blink) ctx.globalAlpha *= 0.4 + 0.6 * Math.abs(Math.sin(this.time * 3 + p.rot * 5));
          ctx.fillRect(x - sz / 2, y - sz / 2, sz, sz); break;
        }
        case "pixel": ctx.fillRect(x - sz / 2, y - sz / 2, sz, sz); break;
        case "impact": { // spiky starburst: outer type colour, inner white
          const pop = 1 - t, R = sz * (0.6 + U.easeOutBack(Math.min(1, pop * 3)) * 0.5), n = p.spikes || 8;
          for (const [m, col] of [[1, p.color], [0.55, "#ffffff"]]) {
            ctx.fillStyle = col; ctx.beginPath();
            for (let k = 0; k < n * 2; k++) { const a = p.rot + (k / (n * 2)) * U.TAU, r = (k % 2 ? R * 0.32 : R * (k % 4 ? 0.8 : 1.15)) * m; k ? ctx.lineTo(x + Math.cos(a) * r, y + Math.sin(a) * r) : ctx.moveTo(x + Math.cos(a) * r, y + Math.sin(a) * r); }
            ctx.closePath(); ctx.fill();
          }
          break;
        }
        case "speedlines": { // radial lines bursting outward from the hit point
          const pop = 1 - t, n = p.n || 10;
          ctx.lineCap = "butt";
          for (let k = 0; k < n; k++) {
            const a = p.rot + (k / n) * U.TAU + (k % 2) * 0.15, r0 = sz * (0.35 + pop * 0.9), r1 = r0 + sz * (k % 3 ? 0.6 : 1.1) * (1 - pop * 0.5);
            ctx.lineWidth = k % 3 ? 2 : 4;
            ctx.beginPath(); ctx.moveTo(x + Math.cos(a) * r0, y + Math.sin(a) * r0 * 0.8); ctx.lineTo(x + Math.cos(a) * r1, y + Math.sin(a) * r1 * 0.8); ctx.stroke();
          }
          break;
        }
        case "column": { // flash pillar from the ground up
          const w = sz * (0.3 + t * 0.7);
          ctx.globalAlpha *= t;
          ctx.fillRect(x - w / 2, y - 900, w, 900);
          ctx.fillStyle = "#ffffff"; ctx.fillRect(x - w / 5, y - 900, w / 2.5, 900);
          ctx.beginPath(); ctx.ellipse(x, y, w, w * 0.4, 0, 0, U.TAU); ctx.fill();
          break;
        }
        case "chunk": { // slab of floor: extruded quad, grass/stone on top, earth below
          const c = Math.cos(p.rot), s = Math.sin(p.rot), th = Math.max(2, sz * 0.45);
          const quad = (dy) => { ctx.beginPath(); [[-1, -0.8], [0.9, -1], [1, 0.75], [-0.85, 1]].forEach(([u, v], k) => { const px = x + (u * c - v * s) * sz, py = y + (u * s + v * c) * sz * 0.6 + dy; k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }); ctx.closePath(); ctx.fill(); };
          ctx.fillStyle = p.side; for (let d = th; d > 0; d -= 1.5) quad(d);
          ctx.fillStyle = p.color; quad(0);
          ctx.fillStyle = "rgba(255,255,255,0.25)"; ctx.fillRect(x - sz * 0.5, y - sz * 0.45, sz * 0.5, 1.5);
          break;
        }
        case "ghostimg": { // fading, vertically stretched silhouette of a fighter (teleport afterimage)
          const fr = p.fr;
          if (!fr) break;
          const K = Arenas.K, img = Dir8.tint(fr.img, p.color), fit = p.fit || 1, sx = t * fit, sy = (1 + (1 - t) * 0.8) * fit;
          ctx.drawImage(img, x - fr.ax * K * sx, y - fr.ay * K * sy, img.width * K * sx, img.height * K * sy);
          break;
        }
        case "beamline": { // tapered energy beam from (x, y) to (p.x2, p.y2 - p.z2)
          const x2 = p.x2 !== undefined ? p.x2 : x, y2 = (p.y2 !== undefined ? p.y2 : y) - (p.z2 !== undefined ? p.z2 : 0);
          ctx.lineCap = "round";
          ctx.strokeStyle = p.color;
          ctx.lineWidth = Math.max(2, sz * (0.45 + t * 0.75));
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x2, y2); ctx.stroke();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = Math.max(1, sz * 0.35 * t);
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x2, y2); ctx.stroke();
          break;
        }
        default: ctx.beginPath(); ctx.arc(x, y, sz * 0.5, 0, U.TAU); ctx.fill();
      }
    }
    ctx.restore();
  }

  // Ground shadows of airborne particles (rocks / drops) — cheap depth cue.
  drawParticleShadows(ctx) {
    ctx.save(); ctx.fillStyle = "rgba(0,0,0,0.25)";
    for (const p of this.parts) {
      if ((p.shape !== "rock" && p.shape !== "drop" && p.shape !== "shard" && p.shape !== "chunk") || p.z < 4) continue;
      const s = Math.max(1, p.size * 0.5 * (1 - Math.min(1, p.z / 300)));
      ctx.beginPath(); ctx.ellipse(p.x, p.y, s, s * 0.5, 0, 0, U.TAU); ctx.fill();
    }
    ctx.restore();
  }

  drawLights(lctx) { // into the light canvas (world space), additive
    lctx.save(); lctx.globalCompositeOperation = "lighter";
    for (const l of this.lights) {
      const t = l.life / l.max, a = l.i * t * (l.flicker ? 0.85 + Math.random() * 0.3 : 1);
      const g = lctx.createRadialGradient(l.x, l.y, 0, l.x, l.y, l.r);
      g.addColorStop(0, U.rgba(l.color, a)); g.addColorStop(1, U.rgba(l.color, 0));
      lctx.fillStyle = g; lctx.fillRect(l.x - l.r, l.y - l.r, l.r * 2, l.r * 2);
    }
    lctx.restore();
  }

  drawFloaters(ctx, zoom) { // world space; text sized against zoom so it stays legible
    ctx.save();
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (const f of this.floaters) {
      if (f.delay > 0) continue;
      const age = f.max - f.life, t = f.life / f.max;
      const pop = f.kind === "num" ? (age < 0.18 ? U.easeOutBack(age / 0.18) : 1) : (age < 0.2 ? U.easeOut(age / 0.2) : 1);
      const size = Math.max(f.kind === "num" ? 14 : 12, (f.size / Math.max(0.9, zoom)) * (f.kind === "num" ? 1.5 : 1.5)) * pop;
      ctx.globalAlpha = t < 0.3 ? t / 0.3 : 1;
      ctx.font = `${f.kind === "num" ? "900" : "800"} ${size}px "Segoe UI", "Hiragino Sans", "Noto Sans JP", Helvetica, Arial, sans-serif`;
      ctx.lineJoin = "round"; ctx.lineWidth = size * (f.kind === "num" ? 0.2 : 0.13); ctx.strokeStyle = f.stroke;
      ctx.strokeText(f.text, f.x, f.y); ctx.fillStyle = f.color; ctx.fillText(f.text, f.x, f.y);
    }
    ctx.restore();
  }
}
FX.HITSTOP_MUL = (() => { const v = parseFloat(new URLSearchParams(location.search).get("hitstop")); return v >= 0 ? Math.min(4, v) : 1.9; })();
