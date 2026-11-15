// Arenas, generated procedurally in ART pixels (1 art px = 2 world units). Each arena prerenders:
//   floor (or 4 animated water frames), crowd idle/cheer overlays, a decal layer, props, static lights,
//   hazards (lava pools, deep water with platforms, ice) and a post-processing grade.
// Logic (collision, positions) stays in world units.
"use strict";

const Arenas = (() => {
  const K = 2; // world units per art pixel
  const STANDS = { top: 96, bottom: 84, side: 120 };
  const OPEN = { top: 76, bottom: 76, side: 112 };

  const DEFS = {
    stadium: { name: "League Stadium", jp: "モンスターリーグ・スタジアム", floor: "grass", surround: "stands", field: [560, 340], ambient: [0.66, 0.7, 0.92], night: true, motes: "dust", props: ["crate", "crate"],
      grade: { tint: [1, 1, 1.04], sat: 1.12, con: 1.06, vig: 0.55, bloom: 0.75, tilt: 1, rays: 0.06 } },
    dirt: { name: "Clay Pitch", jp: "つちのグラウンド", floor: "dirt", surround: "stands", field: [560, 340], ambient: [1.0, 0.96, 0.88], motes: "dust", props: ["boulder", "boulder", "boulder", "boulder"],
      grade: { tint: [1.04, 1, 0.94], sat: 1.05, con: 1.05, vig: 0.45, bloom: 0.5, tilt: 1, rays: 0.1 } },
    stone: { name: "Stone Court", jp: "いしだたみコート", floor: "stone", surround: "stands", field: [540, 330], ambient: [0.92, 0.9, 0.96], motes: "dust", props: ["crate", "crate", "crate"],
      grade: { tint: [1, 0.99, 1.02], sat: 1.0, con: 1.08, vig: 0.5, bloom: 0.55, tilt: 1 } },
    colosseum: { name: "Sand Colosseum", jp: "サンド・コロシアム", floor: "sand", surround: "stands", track: true, field: [540, 340], ambient: [1.08, 1.0, 0.86], motes: "sand", props: ["crate", "crate"],
      grade: { tint: [1.06, 1.0, 0.9], sat: 1.08, con: 1.04, vig: 0.4, bloom: 0.55, tilt: 1, rays: 0.12 } },
    pokeball: { name: "Ball Dome", jp: "ボールドーム", floor: "pokeball", surround: "stands", field: [520, 330], ambient: [0.95, 0.94, 1.0], motes: "dust", props: [],
      grade: { tint: [1, 1, 1.02], sat: 1.1, con: 1.06, vig: 0.5, bloom: 0.6, tilt: 1 } },
    ice: { name: "Frost Rink", jp: "こおりのリンク", floor: "ice", surround: "stands", field: [540, 330], ambient: [0.9, 0.98, 1.1], motes: "snow", ice: true, props: ["iceblock", "iceblock", "iceblock"],
      grade: { tint: [0.96, 1.0, 1.08], sat: 0.95, con: 1.05, vig: 0.45, bloom: 0.85, tilt: 1 } },
    volcano: { name: "Magma Pit", jp: "マグマのふんか口", floor: "volcano", surround: "stands", field: [560, 340], ambient: [0.62, 0.48, 0.48], night: true, motes: "ember", props: ["boulder", "boulder"], lava: 4,
      grade: { tint: [1.08, 0.98, 0.92], sat: 1.15, con: 1.1, vig: 0.6, bloom: 1.0, tilt: 1 } },
    water: { name: "Water Field", jp: "ウォーターフィールド", floor: "water", surround: "stands", field: [560, 360], ambient: [0.92, 0.96, 1.05], motes: "bubble", water: true, props: [],
      grade: { tint: [0.96, 1.0, 1.06], sat: 1.12, con: 1.05, vig: 0.45, bloom: 0.8, tilt: 1, rays: 0.08 } },
    cavern: { name: "Crystal Cavern", jp: "クリスタル・ケイブ", floor: "cave", surround: "cave", field: [540, 340], ambient: [0.42, 0.38, 0.62], night: true, motes: "star", props: ["crystal", "crystal", "crystal", "crystal", "crystal"], starpool: true,
      grade: { tint: [0.98, 0.95, 1.08], sat: 1.15, con: 1.08, vig: 0.7, bloom: 1.05, tilt: 1 } },
    cosmic: { name: "Cosmic Stage", jp: "うちゅうステージ", floor: "cosmic", surround: "space", field: [520, 320], ambient: [0.6, 0.55, 0.85], night: true, motes: "star", lowgrav: true, props: [],
      grade: { tint: [1, 0.96, 1.08], sat: 1.15, con: 1.1, vig: 0.75, bloom: 1.1, tilt: 1 } },
    meadow: { name: "Night Meadow", jp: "よるのそうげん", floor: "meadow", surround: "meadow", field: [560, 340], ambient: [0.42, 0.5, 0.66], night: true, motes: "firefly", props: ["boulder"],
      grade: { tint: [0.98, 1.01, 1.05], sat: 1.08, con: 1.08, vig: 0.65, bloom: 0.88, tilt: 1, fog: [0.35, 0.45, 0.65, 0.08] } },
  };
  // [ユーザー指定 2026-11-16] 「HD-2D リメイクのような画質のバトルフィールドに」: ステージごとの光の設定（post.js / render.js が使う）。
  //   sun = 地面の日なた・日かげの強さ、sunCol / shCol = 日なた・日かげの色、rays = 左上からの光の筋、rayCol = 光の色、
  //   cast = モンスターと小物の落ち影の濃さ、mote = 光の中のちり、split = 暗部を青く・明部を黄色くする強さ。手前と奥のぼかし（tilt）は前のまま 1。
  //   ステージの形・床・観客・小物は変えない。夜のステージは夜のまま、照明や月の光として弱めに入れる。
  const DAY = { sun: 0.48, sunCol: [1.2, 1.13, 0.9], shCol: [0.68, 0.77, 0.94], rays: 0.56, rayCol: [1.0, 0.94, 0.76], cast: 0.5, mote: 0.55, split: 0.55 };
  const HD2D = {
    stadium: { ambient: [0.76, 0.8, 0.94], sun: 0.46, sunCol: [1.2, 1.15, 0.96], shCol: [0.68, 0.76, 0.95], rays: 0.54, rayCol: [1.0, 0.95, 0.82], cast: 0.48, mote: 0.48, split: 0.5, bloom: 0.88 },
    dirt: DAY,
    stone: Object.assign({}, DAY, { sun: 0.4, shCol: [0.7, 0.76, 0.94], rays: 0.48 }),
    colosseum: Object.assign({}, DAY, { sun: 0.42, sunCol: [1.16, 1.09, 0.9], shCol: [0.68, 0.72, 0.9], bloom: 0.55, rays: 0.52 }),
    pokeball: Object.assign({}, DAY, { sun: 0.24, shCol: [0.74, 0.8, 0.96], rays: 0.48 }), // 屋内ドーム: 天窓の光
    ice: Object.assign({}, DAY, { sun: 0.34, sunCol: [1.1, 1.12, 1.08], shCol: [0.66, 0.78, 0.98], rayCol: [0.92, 0.98, 1.0], split: 0.38, rays: 0.46 }),
    water: Object.assign({}, DAY, { sun: 0.44, shCol: [0.66, 0.78, 0.96], rayCol: [0.96, 1.0, 0.9], rays: 0.5 }),
    volcano: { sun: 0.36, sunCol: [1.3, 1.02, 0.74], shCol: [0.68, 0.62, 0.78], rays: 0.38, rayCol: [1.0, 0.62, 0.34], cast: 0.44, mote: 0.42, split: 0.25 },
    cavern: { sun: 0.38, sunCol: [1.04, 1.1, 1.38], shCol: [0.64, 0.64, 0.88], rays: 0.42, rayCol: [0.74, 0.86, 1.0], cast: 0.4, mote: 0.6, split: 0.12 },
    cosmic: { sun: 0.26, sunCol: [1.14, 1.06, 1.35], shCol: [0.72, 0.68, 0.94], rays: 0.28, rayCol: [0.82, 0.78, 1.0], cast: 0.34, mote: 0.7, split: 0.1 },
    meadow: { ambient: [0.72, 0.8, 0.92], sun: 0.48, sunCol: [1.14, 1.2, 1.28], shCol: [0.58, 0.68, 0.88], rays: 0.56, rayCol: [0.84, 0.94, 1.0], cast: 0.52, mote: 0.62, split: 0.35 },
  };
  // rays0 / bloom0 = 作り直す前の値（window.ARENA_HD_OFF = true のとき、この値に戻す）
  for (const k in HD2D) if (DEFS[k]) { const h = Object.assign({}, HD2D[k]), gr = DEFS[k].grade; gr.rays0 = gr.rays || 0; gr.bloom0 = gr.bloom; if (h.ambient) { DEFS[k].ambient0 = DEFS[k].ambient; DEFS[k].ambient = h.ambient; delete h.ambient; } Object.assign(gr, h); }
  const list = () => Object.entries(DEFS).map(([id, d]) => ({ id, name: d.name, jp: d.jp }));

  function create(id, seed = 7, scale = 1) {
    const def = DEFS[id] || DEFS.stadium;
    const rng = U.seeded(seed * 7919 + id.length * 31);
    const m = def.surround === "stands" ? STANDS : OPEN;
    // scale: the pitch grows / shrinks with the monsters that fight on it (Battle picks it; menus and the title use 1)
    const fw = Math.round(def.field[0] * scale / 2) * 2, fh = Math.round(def.field[1] * scale / 2) * 2;
    const aw = fw + m.side * 2, ah = fh + m.top + m.bottom;
    const af = { x: m.side, y: m.top, w: fw, h: fh, r: 22 };
    const A = {
      id, def, aw, ah, af, W: aw * K, H: ah * K, rng, props: [], lights: [], pools: [], platforms: [], time: 0, hype: 0, ticker: "MONSTER LEAGUE",
      field: { x: af.x * K, y: af.y * K, w: fw * K, h: fh * K, cx: (af.x + fw / 2) * K, cy: (af.y + fh / 2) * K, R: af.r * K },
      ice: !!def.ice, water: !!def.water, lowgrav: !!def.lowgrav, grade: def.grade, ambient: def.ambient, night: !!def.night,
    };
    // [ユーザー指定 2026-11-16] 3D の背景（Scene3D）があるステージは、まわりの絵を立体で作る。地面の絵（この canvas）には
    // 立体の足もとの地面だけを描く（scenes/<id>.js の paint）。観客・電光掲示・木などの平面の絵は描かない。
    A.scene3d = (typeof Scene3D !== "undefined" && Scene3D.get(id)) || null;
    A.base = U.canvas(aw, ah);
    const b = A.base.getContext("2d");
    if (A.scene3d && A.scene3d.paint) { b.save(); A.scene3d.paint(b, A, U.seeded(seed * 131 + 17)); b.restore(); }
    else surround(A, b, rng);
    if (!A.scene3d || A.scene3d.walls !== false) walls(A, b);
    if (def.water) {
      A.frames = [0, 1, 2, 3].map((k) => { const c = U.canvas(aw, ah), x = c.getContext("2d"); x.drawImage(A.base, 0, 0); clipField(A, x, () => FLOORS.water(A, x, U.seeded(seed + 3), k)); innerShade(A, x); return c; });
    } else {
      clipField(A, b, () => FLOORS[def.floor](A, b, rng));
      innerShade(A, b);
      A.frames = [A.base];
    }
    if (def.surround === "stands" && !A.scene3d) { A.crowd = [crowd(A, 0), crowd(A, 1)]; }
    A.decals = U.canvas(aw, ah);
    A.decalCount = 0;
    Ground.init(A);
    placeProps(A);
    placeLights(A);
    return A;
  }

  // Top / side colours of a slab of floor torn up by telekinesis.
  const CHUNKS = { grass: ["#5aa845", "#4a2e1a"], meadow: ["#3d6e45", "#2a1c12"], dirt: ["#b07a4a", "#5e3a20"], stone: ["#a4a2aa", "#5a5860"], sand: ["#e0c890", "#9a7848"],
    pokeball: ["#d8d8e0", "#808090"], ice: ["#e6f8ff", "#7fb8d8"], volcano: ["#4a3a36", "#24181a"], cave: ["#56508a", "#28224a"], cosmic: ["#4a3a7a", "#1e1838"], water: ["#5aa845", "#4a2e1a"] };
  function chunkColors(A) { return CHUNKS[A.def.floor] || CHUNKS.grass; }

  // ---- helpers -------------------------------------------------------------------------------------
  function rrPath(x, ctx, a, b, w, h, r) { U.rr(ctx, a, b, w, h, r); }
  function clipField(A, ctx, fn) { const f = A.af; ctx.save(); rrPath(0, ctx, f.x, f.y, f.w, f.h, f.r); ctx.clip(); fn(); ctx.restore(); }
  function speckle(ctx, rng, x, y, w, h, n, colors, sz = 1) {
    for (let i = 0; i < n; i++) { ctx.fillStyle = colors[(rng() * colors.length) | 0]; ctx.fillRect((x + rng() * w) | 0, (y + rng() * h) | 0, sz, sz); }
  }
  function blob(ctx, rng, cx, cy, r, color, dens = 0.9) { Pixel.dither(ctx, cx, cy, r, Math.round(r * 0.7), color, (k) => (1 - k * k) * dens, (rng() * 1e6) | 0); }
  // Jittered-grid Voronoi; returns per-pixel cell id + border mask for the field rect.
  function voronoi(A, rng, cell) {
    const f = A.af, gw = Math.ceil(f.w / cell) + 2, gh = Math.ceil(f.h / cell) + 2, sites = [];
    for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) sites.push([(i - 1 + 0.15 + rng() * 0.7) * cell, (j - 1 + 0.15 + rng() * 0.7) * cell]);
    const id = new Int32Array(f.w * f.h), edge = new Uint8Array(f.w * f.h);
    for (let y = 0; y < f.h; y++) for (let x = 0; x < f.w; x++) {
      const gi = Math.floor(x / cell) + 1, gj = Math.floor(y / cell) + 1;
      let best = 1e9, bi = 0, second = 1e9;
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        const k = (gj + dj) * gw + (gi + di), s = sites[k];
        if (!s) continue;
        const d = (s[0] - x) ** 2 + (s[1] - y) ** 2;
        if (d < best) { second = best; best = d; bi = k; } else if (d < second) second = d;
      }
      id[y * f.w + x] = bi;
      if (Math.sqrt(second) - Math.sqrt(best) < 1.2) edge[y * f.w + x] = 1;
    }
    return { id, edge, n: sites.length };
  }
  function lines(A, ctx, color, opts = {}) { // pitch markings on a scratch canvas, then weathered
    const f = A.af, c = U.canvas(A.aw, A.ah), x = c.getContext("2d"), t = opts.t || 2;
    const ins = 12, X0 = f.x + ins, Y0 = f.y + ins, X1 = f.x + f.w - ins, Y1 = f.y + f.h - ins, cx = f.x + f.w / 2, cy = f.y + f.h / 2;
    x.fillStyle = color;
    const dash = opts.dash || 0;
    const H = (a, b, y) => { for (let i = a; i < b; i++) if (!dash || Math.floor(i / dash) % 2 === 0) x.fillRect(i, y, 1, t); };
    const V = (xx, a, b) => { for (let i = a; i < b; i++) if (!dash || Math.floor(i / dash) % 2 === 0) x.fillRect(xx, i, t, 1); };
    H(X0, X1, Y0); H(X0, X1 + t, Y1); V(X0, Y0, Y1); V(X1, Y0, Y1); V(cx - 1, Y0, Y1);
    for (let k = 0; k < t; k++) Pixel.circle(x, cx, cy, 46 - k, color);
    Pixel.ellipse(x, cx, cy, 3, 3, color);
    const bh = 132, bw = 54;
    H(X0, X0 + bw, cy - bh / 2); H(X0, X0 + bw, cy + bh / 2); V(X0 + bw, cy - bh / 2, cy + bh / 2 + t);
    H(X1 - bw, X1, cy - bh / 2); H(X1 - bw, X1, cy + bh / 2); V(X1 - bw, cy - bh / 2, cy + bh / 2 + t);
    if (opts.double) { for (let k = 0; k < t; k++) { Pixel.circle(x, X0 + 70, cy, 60 - k, color); Pixel.circle(x, X1 - 70, cy, 60 - k, color); } }
    x.globalCompositeOperation = "destination-out";
    const rng = U.seeded(77);
    for (let i = 0; i < (opts.wear || 900); i++) x.fillRect((f.x + rng() * f.w) | 0, (f.y + rng() * f.h) | 0, 1 + (rng() * 3 | 0), 1 + (rng() * 2 | 0));
    ctx.drawImage(c, 0, 0);
  }

  // ---- floors ----------------------------------------------------------------------------------------
  const FLOORS = {
    grass(A, ctx, rng) {
      const f = A.af;
      for (let i = 0; i * 28 < f.w; i++) { ctx.fillStyle = i % 2 ? "#3f8a3a" : "#397f34"; ctx.fillRect(f.x + i * 28, f.y, 28, f.h); }
      speckle(ctx, rng, f.x, f.y, f.w, f.h, 9000, ["#33702e", "#2f6829", "#4a9843", "#55a34c"]);
      for (let i = 0; i < 900; i++) { const x = f.x + rng() * f.w | 0, y = f.y + rng() * f.h | 0; ctx.fillStyle = "#2a5e26"; ctx.fillRect(x, y, 1, 2); ctx.fillStyle = "#5fae52"; ctx.fillRect(x + 1, y - 1, 1, 2); }
      for (let i = 0; i < 14; i++) { const x = f.x + rng() * f.w, y = f.y + rng() * f.h; blob(ctx, rng, x, y, 8 + rng() * 22, "#6d5434", 0.75); blob(ctx, rng, x, y, 5 + rng() * 9, "#5a4329", 0.8); }
      lines(A, ctx, "#e6eee0");
    },
    meadow(A, ctx, rng) {
      const f = A.af;
      for (let i = 0; i * 32 < f.w; i++) { ctx.fillStyle = i % 2 ? "#2b5a2f" : "#27522b"; ctx.fillRect(f.x + i * 32, f.y, 32, f.h); }
      speckle(ctx, rng, f.x, f.y, f.w, f.h, 9000, ["#22472a", "#1e4025", "#356a38", "#3c7440"]);
      for (let i = 0; i < 1400; i++) { const x = f.x + rng() * f.w | 0, y = f.y + rng() * f.h | 0; ctx.fillStyle = "#1a3a20"; ctx.fillRect(x, y, 1, 3); ctx.fillStyle = "#3f7a44"; ctx.fillRect(x + 1, y - 1, 1, 3); }
      for (let i = 0; i < 160; i++) { ctx.fillStyle = ["#e8e4ff", "#ffd6f0", "#fff6a0", "#bfe0ff"][(rng() * 4) | 0]; const x = f.x + rng() * f.w | 0, y = f.y + rng() * f.h | 0; ctx.fillRect(x, y, 1, 1); ctx.fillRect(x - 1, y + 1, 3, 1); ctx.fillRect(x, y + 2, 1, 1); }
      lines(A, ctx, "#b7c8c0", { wear: 2600, double: true });
    },
    dirt(A, ctx, rng) {
      const f = A.af;
      ctx.fillStyle = "#7b5a3c"; ctx.fillRect(f.x, f.y, f.w, f.h);
      for (let i = 0; i < 40; i++) blob(ctx, rng, f.x + rng() * f.w, f.y + rng() * f.h, 14 + rng() * 40, rng() < 0.5 ? "#6c4e33" : "#8a6847", 0.7);
      speckle(ctx, rng, f.x, f.y, f.w, f.h, 7000, ["#5f442c", "#94704c", "#6f5136", "#a07c55"]);
      for (let i = 0; i < 260; i++) { const x = f.x + rng() * f.w | 0, y = f.y + rng() * f.h | 0; ctx.fillStyle = "#4b3522"; ctx.fillRect(x, y + 1, 2, 1); ctx.fillStyle = "#b2936b"; ctx.fillRect(x, y, 2, 1); }
      for (let i = 0; i < 24; i++) crackLine(ctx, rng, f.x + rng() * f.w, f.y + rng() * f.h, "#4a3220", 6 + rng() * 8);
      lines(A, ctx, "#e8dcc6", { dash: 6, wear: 1600 });
    },
    stone(A, ctx, rng) {
      const f = A.af, T = 24;
      for (let y = f.y; y < f.y + f.h; y += T) for (let x = f.x; x < f.x + f.w; x += T) {
        const split = rng();
        const tiles = split < 0.2 ? [[x, y, T, T / 2], [x, y + T / 2, T, T / 2]] : split < 0.4 ? [[x, y, T / 2, T], [x + T / 2, y, T / 2, T]] : [[x, y, T, T]];
        for (const [a, b, w, h] of tiles) {
          const v = 0.86 + rng() * 0.24, c = (n) => Math.round(n * v);
          ctx.fillStyle = `rgb(${c(112)},${c(110)},${c(118)})`; ctx.fillRect(a, b, w, h);
          ctx.fillStyle = "rgba(255,255,255,0.18)"; ctx.fillRect(a, b, w, 1); ctx.fillRect(a, b, 1, h);
          ctx.fillStyle = "rgba(0,0,0,0.35)"; ctx.fillRect(a, b + h - 1, w, 1); ctx.fillRect(a + w - 1, b, 1, h);
          speckle(ctx, rng, a + 1, b + 1, w - 2, h - 2, 6, ["#5f5d66", "#8b8994"]);
          if (rng() < 0.15) speckle(ctx, rng, a + 2, b + 2, w - 4, h - 4, 10, ["#4f7a43", "#5e8f4f"]);
        }
      }
      for (let i = 0; i < 18; i++) crackLine(ctx, rng, f.x + rng() * f.w, f.y + rng() * f.h, "#3a3842", 7);
      lines(A, ctx, "#d9d9e4", { dash: 5, wear: 1200 });
    },
    sand(A, ctx, rng) {
      const f = A.af;
      ctx.fillStyle = "#d8bd88"; ctx.fillRect(f.x, f.y, f.w, f.h);
      for (let i = 0; i < 26; i++) blob(ctx, rng, f.x + rng() * f.w, f.y + rng() * f.h, 20 + rng() * 50, rng() < 0.6 ? "#e6cd9a" : "#c9ab76", 0.6);
      speckle(ctx, rng, f.x, f.y, f.w, f.h, 12000, ["#c5a674", "#e9d2a4", "#b89764", "#f2dfb5"]);
      for (let i = 0; i < 10; i++) crackLine(ctx, rng, f.x + rng() * f.w, f.y + rng() * f.h, "#9c7c4c", 6);
      lines(A, ctx, "#f4ecdc", { dash: 7, wear: 1500 });
    },
    pokeball(A, ctx) {
      const f = A.af, cx = f.x + f.w / 2, cy = f.y + f.h / 2;
      ctx.fillStyle = "#c63e4c"; ctx.fillRect(f.x, f.y, f.w / 2, f.h);
      ctx.fillStyle = "#e8e6ef"; ctx.fillRect(cx, f.y, f.w / 2, f.h);
      ctx.fillStyle = "rgba(0,0,0,0.06)";
      for (let x = f.x; x < f.x + f.w; x += 32) ctx.fillRect(x, f.y, 1, f.h);
      for (let y = f.y; y < f.y + f.h; y += 32) ctx.fillRect(f.x, y, f.w, 1);
      ctx.fillStyle = "#e05866"; for (let y = f.y; y < f.y + f.h; y += 4) ctx.fillRect(f.x, y, f.w / 2, 1);
      ctx.fillStyle = "#2a2d4a"; ctx.fillRect(cx - 6, f.y, 12, f.h);
      Pixel.ellipse(ctx, cx, cy, 40, 40, "#2a2d4a"); Pixel.ellipse(ctx, cx, cy, 27, 27, "#f6f6fb"); Pixel.circle(ctx, cx, cy, 20, "#2a2d4a", 3);
      Pixel.ellipse(ctx, cx - 4, cy - 6, 6, 4, "#ffffff");
      lines(A, ctx, "rgba(42,45,74,0.6)", { wear: 400 });
    },
    ice(A, ctx, rng) {
      const f = A.af, T = 32;
      for (let y = f.y; y < f.y + f.h; y += T) for (let x = f.x; x < f.x + f.w; x += T) { ctx.fillStyle = ((x + y) / T) % 2 ? "#cfe6f2" : "#c3ddec"; ctx.fillRect(x, y, T, T); }
      speckle(ctx, rng, f.x, f.y, f.w, f.h, 4000, ["#e8f6ff", "#b1d1e6", "#ffffff"]);
      for (let i = 0; i < 40; i++) { const x = f.x + rng() * f.w, y = f.y + rng() * f.h, l = 10 + rng() * 30; Pixel.line(ctx, x, y, x + l, y - l * 0.6, "rgba(255,255,255,0.75)"); }
      for (let i = 0; i < 26; i++) crackLine(ctx, rng, f.x + rng() * f.w, f.y + rng() * f.h, "#93bcd6", 8);
      for (let i = 0; i < 10; i++) blob(ctx, rng, f.x + rng() * f.w, f.y + (rng() < 0.5 ? 6 : f.h - 6), 16 + rng() * 20, "#f6fbff", 0.85);
      lines(A, ctx, "#7fb6e0", { wear: 700 });
    },
    volcano(A, ctx, rng) {
      const f = A.af, vor = voronoi(A, rng, 34), shade = [];
      for (let i = 0; i < vor.n; i++) shade.push(0.75 + rng() * 0.4);
      const img = ctx.createImageData(f.w, f.h), d = img.data;
      const glowCell = new Uint8Array(vor.n); for (let i = 0; i < vor.n; i++) glowCell[i] = rng() < 0.45 ? 1 : 0;
      for (let i = 0; i < f.w * f.h; i++) {
        const o = i * 4, s = shade[vor.id[i]], x = i % f.w, y = (i / f.w) | 0;
        if (vor.edge[i]) {
          if (glowCell[vor.id[i]]) { const hot = Pixel.bayer(x, y) < 0.5; d[o] = 255; d[o + 1] = hot ? 140 : 96; d[o + 2] = hot ? 40 : 20; }
          else { d[o] = 26; d[o + 1] = 19; d[o + 2] = 18; }
        } else { const n = Pixel.bayer(x * 3 + y, y * 5) < 0.12 ? 0.85 : 1; d[o] = 62 * s * n; d[o + 1] = 50 * s * n; d[o + 2] = 47 * s * n; }
        d[o + 3] = 255;
      }
      ctx.putImageData(img, f.x, f.y);
      const spots = [[0.27, 0.3], [0.73, 0.72], [0.3, 0.76], [0.7, 0.26], [0.5, 0.5]].slice(0, A.def.lava || 4);
      for (const [u, v] of spots) {
        const x = f.x + f.w * u, y = f.y + f.h * v, r = 18 + rng() * 6;
        Pixel.ellipse(ctx, x, y, r + 4, Math.round((r + 4) * 0.72), "#241815");
        Pixel.ellipse(ctx, x, y, r, Math.round(r * 0.72), "#d8420c");
        Pixel.dither(ctx, x, y, r - 2, Math.round((r - 2) * 0.72), "#ff8a1c", (k) => 1 - k * 0.8);
        Pixel.dither(ctx, x, y, r - 7, Math.round((r - 7) * 0.72), "#ffd34a", (k) => 1 - k);
        for (let i = 0; i < 6; i++) Pixel.ellipse(ctx, x + (rng() - 0.5) * r, y + (rng() - 0.5) * r * 0.6, 2, 1, "#5a2410");
        A.pools.push({ kind: "lava", x: x * K, y: y * K, r: r * K * 0.85 });
      }
    },
    cave(A, ctx, rng) {
      const f = A.af, vor = voronoi(A, rng, 30), shade = [];
      for (let i = 0; i < vor.n; i++) shade.push(0.78 + rng() * 0.38);
      const img = ctx.createImageData(f.w, f.h), d = img.data;
      for (let i = 0; i < f.w * f.h; i++) {
        const o = i * 4, s = shade[vor.id[i]];
        if (vor.edge[i]) { d[o] = 30; d[o + 1] = 24; d[o + 2] = 44; } else { d[o] = 74 * s; d[o + 1] = 64 * s; d[o + 2] = 96 * s; }
        d[o + 3] = 255;
      }
      ctx.putImageData(img, f.x, f.y);
      speckle(ctx, rng, f.x, f.y, f.w, f.h, 3000, ["#5b5078", "#3a3150", "#8e7fb0"]);
      // starry pool
      const cx = f.x + f.w / 2, cy = f.y + f.h / 2;
      const lobes = [[0, 0, 70, 40], [-50, 6, 36, 26], [54, -4, 40, 26], [10, 18, 46, 24]];
      for (const [dx, dy, rx, ry] of lobes) Pixel.ellipse(ctx, cx + dx, cy + dy, rx + 3, ry + 3, "#5fd8ff");
      for (const [dx, dy, rx, ry] of lobes) Pixel.ellipse(ctx, cx + dx, cy + dy, rx, ry, "#0b1030");
      for (const [dx, dy, rx, ry] of lobes) Pixel.dither(ctx, cx + dx, cy + dy, rx - 4, ry - 4, "#162050", (k) => 1 - k);
      for (let i = 0; i < 140; i++) { const a = rng() * U.TAU, r = rng(); const x = cx + Math.cos(a) * r * 80, y = cy + Math.sin(a) * r * 36; ctx.fillStyle = rng() < 0.3 ? "#ffffff" : "#9fd8ff"; ctx.fillRect(x | 0, y | 0, 1, 1); }
      lines(A, ctx, "rgba(170,150,255,0.45)", { wear: 2400 });
    },
    cosmic(A, ctx, rng) {
      const f = A.af, T = 24;
      ctx.fillStyle = "#1d1533"; ctx.fillRect(f.x, f.y, f.w, f.h);
      for (let y = f.y; y < f.y + f.h; y += T) for (let x = f.x; x < f.x + f.w; x += T) { ctx.fillStyle = rng() < 0.5 ? "#231a3e" : "#1a1330"; ctx.fillRect(x + 1, y + 1, T - 1, T - 1); }
      ctx.fillStyle = "#5a2fc0";
      for (let x = f.x; x < f.x + f.w; x += T) ctx.fillRect(x, f.y, 1, f.h);
      for (let y = f.y; y < f.y + f.h; y += T) ctx.fillRect(f.x, y, f.w, 1);
      ctx.fillStyle = "#d24aff";
      for (let x = f.x; x < f.x + f.w; x += T * 4) ctx.fillRect(x, f.y, 1, f.h);
      for (let y = f.y; y < f.y + f.h; y += T * 4) ctx.fillRect(f.x, y, f.w, 1);
      speckle(ctx, rng, f.x, f.y, f.w, f.h, 400, ["#ffffff", "#c8b8ff"]);
      const cx = f.x + f.w / 2, cy = f.y + f.h / 2;
      Pixel.circle(ctx, cx, cy, 52, "#e070ff", 2); Pixel.circle(ctx, cx, cy, 12, "#e070ff", 2);
      Pixel.dither(ctx, cx, cy, 48, 48, "#3a1f66", (k) => (1 - k) * 0.5);
    },
    water(A, ctx, rng, frame = 0) {
      const f = A.af;
      ctx.fillStyle = "#1f5fae"; ctx.fillRect(f.x, f.y, f.w, f.h);
      for (let y = f.y; y < f.y + f.h; y += 2) for (let x = f.x; x < f.x + f.w; x += 2) {
        const v = Math.sin(x * 0.09 + y * 0.05 + frame * 1.57) + Math.sin(y * 0.13 - x * 0.04 + frame * 1.57 * 0.5);
        if (v > 1.3 && Pixel.bayer(x, y) < 0.6) { ctx.fillStyle = "#3a86d8"; ctx.fillRect(x, y, 2, 1); }
        else if (v < -1.2) { ctx.fillStyle = "#18509a"; ctx.fillRect(x, y, 2, 2); }
      }
      const r2 = U.seeded(901);
      for (let i = 0; i < 160; i++) { const x = f.x + r2() * f.w, y = f.y + r2() * f.h; if ((i + frame) % 4 === 0) { ctx.fillStyle = "#cfeaff"; ctx.fillRect(x | 0, y | 0, 3, 1); } }
      // platforms
      if (!A.platforms.length) {
        const spots = [[0.2, 0.3, 30, "wood"], [0.8, 0.7, 30, "wood"], [0.22, 0.75, 26, "wood"], [0.78, 0.28, 26, "wood"], [0.5, 0.5, 36, "ball"], [0.5, 0.14, 20, "wood"], [0.5, 0.86, 20, "wood"], [0.36, 0.5, 18, "wood"], [0.64, 0.5, 18, "wood"]];
        for (const [u, v, r, kind] of spots) A.platforms.push({ x: (f.x + f.w * u) * K, y: (f.y + f.h * v) * K, r: r * K, kind });
      }
      for (const p of A.platforms) {
        const x = p.x / K, y = p.y / K, r = p.r / K;
        Pixel.ellipse(ctx, x, y + 3, r + 2, Math.round(r * 0.62) + 2, "#123f76");
        if (p.kind === "ball") {
          Pixel.ellipse(ctx, x, y + 2, r, Math.round(r * 0.62), "#3a2a2a");
          Pixel.ellipse(ctx, x, y, r, Math.round(r * 0.62), "#f1eef4");
          ctx.save(); ctx.beginPath(); ctx.rect(x - r, y - r, r * 2, r); ctx.clip(); Pixel.ellipse(ctx, x, y, r, Math.round(r * 0.62), "#d8394a"); ctx.restore();
          ctx.fillStyle = "#2a2a35"; ctx.fillRect(x - r, y - 1, r * 2, 3);
          Pixel.ellipse(ctx, x, y, 7, 5, "#2a2a35"); Pixel.ellipse(ctx, x, y, 4, 3, "#ffffff");
        } else {
          Pixel.ellipse(ctx, x, y + 3, r, Math.round(r * 0.62), "#5a3a1e");
          Pixel.ellipse(ctx, x, y, r, Math.round(r * 0.62), "#b07840");
          for (let k = -r + 5; k < r; k += 6) { ctx.fillStyle = "#8c5c2e"; ctx.fillRect(x + k, y - Math.round(r * 0.55), 1, Math.round(r * 1.1)); }
          Pixel.ring(ctx, x, y, r, Math.round(r * 0.62), "#6e4724");
          Pixel.ring(ctx, x, y, r - 3, Math.round(r * 0.62) - 2, "#c99057");
        }
        if (frame % 2) Pixel.ring(ctx, x, y + 3, r + 4, Math.round(r * 0.62) + 4, "rgba(220,240,255,0.7)", 1, 3);
      }
    },
  };

  // A wandering crack; with A it is also cut into the height map so its lips catch the light.
  function crackLine(ctx, rng, x, y, color, n, A, depth = 1.1) {
    let a = rng() * U.TAU;
    for (let k = 0; k < n; k++) {
      a += (rng() - 0.5) * 1.3; const nx = x + Math.cos(a) * (3 + rng() * 5), ny = y + Math.sin(a) * (2 + rng() * 4);
      Pixel.line(ctx, x, y, nx, ny, color);
      if (A) Ground.groove(A, x * K, y * K, nx * K, ny * K, K * 2 * Math.max(0.6, 1.2 - k * 0.15), depth * Math.max(0.4, 1 - k * 0.12));
      x = nx; y = ny;
    }
  }

  // ---- surroundings ----------------------------------------------------------------------------------
  function surround(A, ctx, rng) {
    const { aw, ah, af, def } = A;
    if (def.surround === "stands") {
      ctx.fillStyle = A.night ? "#181a24" : "#3b2f2a"; ctx.fillRect(0, 0, aw, ah);
      const tier = A.night ? ["#262a38", "#20232f"] : ["#55443b", "#4a3a32"];
      for (let y = 0; y < af.y - 18; y += 9) { ctx.fillStyle = tier[(y / 9) % 2]; ctx.fillRect(0, y, aw, 9); ctx.fillStyle = "rgba(0,0,0,0.3)"; ctx.fillRect(0, y + 8, aw, 1); }
      for (let y = af.y + af.h + 26; y < ah; y += 9) { ctx.fillStyle = tier[(y / 9 | 0) % 2]; ctx.fillRect(0, y, aw, 9); ctx.fillStyle = "rgba(0,0,0,0.3)"; ctx.fillRect(0, y + 8, aw, 1); }
      for (let x = 30; x < aw; x += 110) { ctx.fillStyle = "#101219"; ctx.fillRect(x, 0, 8, af.y - 18); ctx.fillRect(x, af.y + af.h + 26, 8, ah); }
      // side aprons (concrete with grating around trainers) + outer side bleachers
      const apW = 36, sL = af.x - apW, sR = af.x + af.w + apW;
      ctx.fillStyle = A.night ? "#3b3e4a" : "#6d655e";
      ctx.fillRect(Math.max(0, sL), af.y - 18, apW - 6, af.h + 44);
      ctx.fillRect(af.x + af.w + 6, af.y - 18, apW - 6, af.h + 44);
      ctx.fillStyle = "rgba(0,0,0,0.25)";
      for (let y = af.y - 18; y < af.y + af.h + 26; y += 6) {
        ctx.fillRect(Math.max(0, sL), y, apW - 6, 1);
        ctx.fillRect(af.x + af.w + 6, y, apW - 6, 1);
      }
      if (sL > 8) {
        for (let x = 0; x < sL - 4; x += 9) {
          ctx.fillStyle = tier[(x / 9 | 0) % 2]; ctx.fillRect(x, af.y - 18, 8, af.h + 44);
          ctx.fillRect(aw - x - 8, af.y - 18, 8, af.h + 44);
          ctx.fillStyle = "rgba(0,0,0,0.32)"; ctx.fillRect(x + 8, af.y - 18, 1, af.h + 44);
          ctx.fillRect(aw - x - 9, af.y - 18, 1, af.h + 44);
        }
        ctx.fillStyle = "#101219";
        ctx.fillRect(0, af.y + af.h / 2 - 6, sL - 4, 12);
        ctx.fillRect(sR + 4, af.y + af.h / 2 - 6, aw - sR - 4, 12);
      }
      // tunnel
      ctx.fillStyle = "#07080c"; ctx.fillRect(aw / 2 - 26, af.y - 58, 52, 40);
      ctx.fillStyle = "#2c2f3b"; ctx.fillRect(aw / 2 - 30, af.y - 62, 60, 4);
      if (def.track) {
        ctx.fillStyle = "#b8423a"; U.rr(ctx, af.x - 16, af.y - 16, af.w + 32, af.h + 32, af.r + 14); ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.8)"; ctx.lineWidth = 1;
        for (const k of [4, 10]) { ctx.beginPath(); U.rr(ctx, af.x - k + 0.5, af.y - k + 0.5, af.w + k * 2 - 1, af.h + k * 2 - 1, af.r + k); ctx.stroke(); }
      }
      // ad boards along the bottom
      const ADS = [["#1d4ed8", "#ffffff", "ITEM MART"], ["#dc2626", "#fde68a", "ARENA REVENGE"], ["#0f766e", "#ecfeff", "SILPH CO"], ["#f59e0b", "#1f2937", "LEAGUE TV"], ["#7c3aed", "#f5f3ff", "HD-2D"], ["#111827", "#facc15", "BIG KID NATION"]];
      let i = 0;
      for (let x = af.x - 30; x < af.x + af.w + 30; x += 92) {
        const [bg, fg, t] = ADS[i++ % ADS.length];
        ctx.fillStyle = "#0b0c10"; ctx.fillRect(x - 1, af.y + af.h + 13, 90, 12);
        ctx.fillStyle = bg; ctx.fillRect(x, af.y + af.h + 14, 88, 10);
        ctx.fillStyle = "rgba(255,255,255,0.2)"; ctx.fillRect(x, af.y + af.h + 14, 88, 1);
        const tw = Pixel.textWidth(t, 1); if (tw < 86) Pixel.text(ctx, t, x + 44 - tw / 2, af.y + af.h + 16, 1, fg);
      }
      // LED board frame on the top (ticker text is drawn live)
      A.led = { x: af.x + 40, y: af.y - 16, w: af.w - 80, h: 11 };
      ctx.fillStyle = "#050507"; ctx.fillRect(A.led.x - 2, A.led.y - 2, A.led.w + 4, A.led.h + 4);
      ctx.fillStyle = "#14100c"; ctx.fillRect(A.led.x, A.led.y, A.led.w, A.led.h);
    } else if (def.surround === "cave") {
      ctx.fillStyle = "#0c0912"; ctx.fillRect(0, 0, aw, ah);
      for (let i = 0; i < 300; i++) { const x = rng() * aw, y = rng() * ah, r = 6 + rng() * 22; Pixel.ellipse(ctx, x, y, r, Math.round(r * 0.7), ["#1b1526", "#231c33", "#151020"][(rng() * 3) | 0]); }
      speckle(ctx, rng, 0, 0, aw, ah, 600, ["#3a2f55", "#6a52a0"]);
      ctx.fillStyle = "#2b2140"; U.rr(ctx, af.x - 8, af.y - 8, af.w + 16, af.h + 16, af.r + 8); ctx.fill();
    } else if (def.surround === "space") {
      ctx.fillStyle = "#07040f"; ctx.fillRect(0, 0, aw, ah);
      for (let i = 0; i < 16; i++) blob(ctx, rng, rng() * aw, rng() * ah, 30 + rng() * 70, ["#2a1248", "#3b1a5c", "#1a1040"][(rng() * 3) | 0], 0.55);
      speckle(ctx, rng, 0, 0, aw, ah, 900, ["#ffffff", "#bfb0ff", "#ffe9b0", "#8fd0ff"]);
      for (let i = 0; i < 30; i++) { const x = rng() * aw | 0, y = rng() * ah | 0; ctx.fillStyle = "#fff"; ctx.fillRect(x - 1, y, 3, 1); ctx.fillRect(x, y - 1, 1, 3); }
      // ringed planet in a corner
      const px = aw - 70, py = ah - 46;
      Pixel.ellipse(ctx, px, py, 34, 34, "#c9a86c");
      for (let k = -30; k < 34; k += 8) { ctx.fillStyle = "#a88752"; ctx.fillRect(px - 34, py + k, 68, 3); }
      ctx.save(); ctx.beginPath(); ctx.arc(px, py, 34, 0, U.TAU); ctx.clip(); Pixel.dither(ctx, px + 14, py + 14, 34, 34, "#5a4630", (k) => k); ctx.restore();
      Pixel.ring(ctx, px, py, 58, 12, "#e8d6a6", 3); Pixel.ring(ctx, px, py, 50, 10, "#b39966", 2);
      // field floats with an under-glow
      Pixel.dither(ctx, af.x + af.w / 2, af.y + af.h / 2, af.w / 2 + 30, af.h / 2 + 30, "#5a2fc0", (k) => (k > 0.8 ? (1 - k) * 2 : 0));
      ctx.fillStyle = "#120c22"; U.rr(ctx, af.x - 6, af.y - 6, af.w + 12, af.h + 20, af.r + 6); ctx.fill();
    } else { // meadow
      ctx.fillStyle = "#162a1a"; ctx.fillRect(0, 0, aw, ah);
      speckle(ctx, rng, 0, 0, aw, ah, 6000, ["#1d3822", "#244528", "#12241a"]);
      for (let i = 0; i < 60; i++) { const x = rng() * aw, y = rng() * ah; if (x > af.x - 10 && x < af.x + af.w + 10 && y > af.y - 10 && y < af.y + af.h + 10) continue; tree(ctx, x, y, rng); }
    }
  }
  function tree(ctx, x, y, rng) {
    const r = 10 + rng() * 8;
    Pixel.ellipse(ctx, x + 4, y + 6, r, Math.round(r * 0.6), "#0b160d");
    Pixel.ellipse(ctx, x, y, r, Math.round(r * 0.85), "#1c3d22");
    Pixel.dither(ctx, x - 3, y - 4, r - 3, Math.round((r - 3) * 0.8), "#2c5a30", (k) => 1 - k);
  }

  function walls(A, ctx) {
    const { af, def } = A;
    if (def.surround !== "stands") return;
    const t = 6;
    ctx.fillStyle = "#1a1b22"; U.rr(ctx, af.x - t - 2, af.y - t - 2, af.w + (t + 2) * 2, af.h + (t + 2) * 2, af.r + t + 2); ctx.fill();
    ctx.fillStyle = A.night ? "#4d5160" : "#8d8478"; U.rr(ctx, af.x - t, af.y - t, af.w + t * 2, af.h + t * 2, af.r + t); ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,0.15)"; ctx.fillRect(af.x, af.y - t, af.w, 1);
    // hazard stripe ring (yellow / black, 2px)
    const band = U.canvas(A.aw, A.ah), bx = band.getContext("2d");
    for (let i = -A.ah; i < A.aw; i += 6) { bx.fillStyle = "#e8c23a"; bx.beginPath(); bx.moveTo(i, 0); bx.lineTo(i + 3, 0); bx.lineTo(i + 3 + A.ah, A.ah); bx.lineTo(i + A.ah, A.ah); bx.fill(); }
    bx.globalCompositeOperation = "destination-in";
    bx.fillStyle = "#000"; U.rr(bx, af.x - 3, af.y - 3, af.w + 6, af.h + 6, af.r + 3); bx.fill();
    bx.globalCompositeOperation = "destination-out"; U.rr(bx, af.x - 1, af.y - 1, af.w + 2, af.h + 2, af.r + 1); bx.fill();
    ctx.fillStyle = "#151515"; U.rr(ctx, af.x - 3, af.y - 3, af.w + 6, af.h + 6, af.r + 3); ctx.fill();
    ctx.drawImage(band, 0, 0);
  }
  function innerShade(A, ctx) { // wall shadow falling onto the field (dithered)
    const f = A.af;
    ctx.save(); U.rr(ctx, f.x, f.y, f.w, f.h, f.r); ctx.clip();
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    for (let y = 0; y < 8; y++) for (let x = f.x; x < f.x + f.w; x++) if (Pixel.bayer(x, y) < (1 - y / 8) * 0.8) ctx.fillRect(x, f.y + y, 1, 1);
    for (let x = 0; x < 4; x++) for (let y = f.y; y < f.y + f.h; y++) if (Pixel.bayer(x, y) < (1 - x / 4) * 0.5) { ctx.fillRect(f.x + x, y, 1, 1); ctx.fillRect(f.x + f.w - 1 - x, y, 1, 1); }
    ctx.restore();
  }

  // Pixel crowd on transparent canvases: frame 0 idle, frame 1 cheering (arms up, hop, waving signs).
  function crowd(A, cheer) {
    const { aw, ah, af } = A, c = U.canvas(aw, ah), x = c.getContext("2d");
    const rng = U.seeded(4242);
    const SHIRT = ["#e04040", "#f0c030", "#3a7bd5", "#55b848", "#f58d2d", "#9b59b6", "#ecf0f1", "#2ecc71", "#e91e63", "#16a085", "#d35400", "#34495e"];
    const SKIN = ["#f2c9a0", "#d9a06c", "#a86f45", "#6f4a2c", "#ffd9b5"];
    const HAIR = ["#2a1a10", "#5a3b1e", "#e8c860", "#1a1a1a", "#b0482a", "#6a3a8a"];
    const SIGNS = ["#ffffff", "#ffe25a", "#ff7ab0"];
    const rows = [];
    for (let y = 0; y < af.y - 22; y += 9) rows.push(y);
    for (let y = af.y + af.h + 28; y < ah - 6; y += 9) rows.push(y);
    for (const y of rows) for (let px = 3; px < aw - 6; px += 7) {
      if (Math.abs(px - aw / 2) < 32 && y < af.y) continue; // tunnel
      if ((px - 30) % 110 < 10) continue; // aisles
      if (rng() < 0.08) { rng(); rng(); rng(); continue; }
      const shirt = SHIRT[(rng() * SHIRT.length) | 0], skin = SKIN[(rng() * SKIN.length) | 0], hair = HAIR[(rng() * HAIR.length) | 0];
      const sign = rng() < 0.05, up = cheer && rng() < 0.6, hop = up && rng() < 0.5 ? 1 : 0;
      const by = y + 1 - hop;
      x.fillStyle = "#0e0f14"; x.fillRect(px - 1, by + 1, 7, 8);
      x.fillStyle = shirt; x.fillRect(px, by + 4, 5, 4);
      x.fillStyle = skin; x.fillRect(px + 1, by + 1, 3, 3);
      x.fillStyle = hair; x.fillRect(px + 1, by, 3, 1);
      if (up) { x.fillStyle = skin; x.fillRect(px - 1, by + 1, 1, 3); x.fillRect(px + 5, by + 1, 1, 3); }
      if (sign && (cheer || rng() < 0.5)) { x.fillStyle = "#111"; x.fillRect(px - 1, by - 5 - hop, 8, 6); x.fillStyle = SIGNS[(px / 7 | 0) % 3]; x.fillRect(px, by - 4 - hop, 6, 4); x.fillStyle = "#c22"; x.fillRect(px + 2, by - 3 - hop, 2, 2); }
    }
    // outer side stands spectators
    const sL = af.x - 42, sR = af.x + af.w + 42;
    if (sL > 10) {
      for (let y = af.y - 12; y < af.y + af.h + 16; y += 8) {
        if (Math.abs(y - (af.y + af.h / 2)) < 9) continue;
        for (let col = 2; col < sL - 6; col += 9) {
          for (const side of [0, 1]) {
            if (rng() < 0.14) continue;
            const px = side === 0 ? col : aw - col - 6;
            const shirt = SHIRT[(rng() * SHIRT.length) | 0], skin = SKIN[(rng() * SKIN.length) | 0], hair = HAIR[(rng() * HAIR.length) | 0];
            const up = cheer && rng() < 0.55, hop = up && rng() < 0.5 ? 1 : 0, by = y - hop;
            x.fillStyle = "#0e0f14"; x.fillRect(px - 1, by + 1, 7, 7);
            x.fillStyle = shirt; x.fillRect(px, by + 4, 5, 3);
            x.fillStyle = skin; x.fillRect(px + 1, by + 1, 3, 3);
            x.fillStyle = hair; x.fillRect(px + 1, by, 3, 1);
            if (up) { x.fillStyle = skin; x.fillRect(px - 1, by + 1, 1, 2); x.fillRect(px + 5, by + 1, 1, 2); }
          }
        }
      }
    }
    return c;
  }

  // ---- props -------------------------------------------------------------------------------------------
  function propSprite(kind, rng) {
    if (kind === "crate") {
      const c = U.canvas(18, 18), x = c.getContext("2d");
      x.fillStyle = "#3a2410"; x.fillRect(0, 0, 18, 18);
      x.fillStyle = "#b47a3c"; x.fillRect(1, 1, 16, 16);
      x.fillStyle = "#8a5826"; x.fillRect(1, 1, 16, 2); x.fillRect(1, 15, 16, 2); x.fillRect(1, 1, 2, 16); x.fillRect(15, 1, 2, 16);
      Pixel.line(x, 3, 3, 14, 14, "#8a5826", 2); Pixel.line(x, 14, 3, 3, 14, "#8a5826", 1);
      x.fillStyle = "#d9a060"; x.fillRect(3, 3, 9, 1);
      return { c, r: 11, hp: 70, ax: 9, ay: 16 };
    }
    if (kind === "boulder" || kind === "iceblock") {
      const ice = kind === "iceblock";
      const c = U.canvas(26, 22), x = c.getContext("2d");
      Pixel.ellipse(x, 13, 12, 12, 9, ice ? "#3a6c8a" : "#2f2b28");
      Pixel.ellipse(x, 13, 11, 11, 8, ice ? "#9fd2ee" : "#7a746c");
      Pixel.dither(x, 10, 8, 8, 5, ice ? "#e6f7ff" : "#9c958b", (k) => 1 - k);
      if (!ice) speckle(x, rng, 4, 6, 18, 12, 10, ["#5d8a46", "#4f7a3c", "#5a554f"]);
      else { x.fillStyle = "#ffffff"; x.fillRect(7, 6, 1, 6); x.fillRect(9, 5, 1, 3); }
      return { c, r: 14, hp: ice ? 90 : 999, ax: 13, ay: 20 };
    }
    // crystal
    const c = U.canvas(20, 36), x = c.getContext("2d");
    const hue = rng() < 0.5 ? ["#5a2f9a", "#9b5cff", "#e2c8ff"] : ["#1f6a9a", "#5fc8ff", "#e0f6ff"];
    for (const [bx, h, w] of [[10, 34, 8], [4, 20, 5], [16, 24, 5]]) {
      x.fillStyle = hue[0]; x.beginPath(); x.moveTo(bx - w / 2, 35); x.lineTo(bx, 35 - h); x.lineTo(bx + w / 2, 35); x.fill();
      x.fillStyle = hue[1]; x.fillRect(bx - 1, 37 - h, 2, h - 4);
      x.fillStyle = hue[2]; x.fillRect(bx - 1, 38 - h, 1, h * 0.4);
    }
    return { c, r: 12, hp: 999, ax: 10, ay: 34, glow: hue[1] };
  }
  function placeProps(A) {
    const f = A.af, rng = A.rng;
    const spots = [[0.2, 0.22], [0.8, 0.78], [0.2, 0.78], [0.8, 0.22], [0.5, 0.12], [0.5, 0.88], [0.36, 0.5], [0.64, 0.5]];
    const kinds = A.def.props.slice();
    if (A.def.starpool) { spots.splice(0, 8, [0.12, 0.2], [0.88, 0.2], [0.12, 0.82], [0.88, 0.82], [0.5, 0.1]); }
    kinds.forEach((kind, i) => {
      const [u, v] = spots[i % spots.length];
      const p = propSprite(kind, rng);
      A.props.push({ kind, x: (f.x + f.w * u + (rng() - 0.5) * 24) * K, y: (f.y + f.h * v + (rng() - 0.5) * 16) * K, r: p.r * K, hp: p.hp, maxHp: p.hp, sprite: p.c, ax: p.ax, ay: p.ay, glow: p.glow, alive: true, shake: 0 });
    });
  }
  function placeLights(A) {
    const f = A.field, L = (x, y, r, color, i, extra = {}) => A.lights.push(Object.assign({ x, y, r, color, i }, extra));
    const d = A.def;
    if (d.surround === "stands" && d.night) for (const [u, v] of [[0.06, 0.06], [0.94, 0.06], [0.06, 0.94], [0.94, 0.94]]) L(f.x + f.w * u, f.y + f.h * v, 760, [255, 238, 205], 0.42, { flood: true });
    if (d.id === "stadium" || A.id === "stadium") L(f.cx, f.cy, 620, [210, 225, 255], 0.18);
    if (A.id === "volcano") for (const p of A.pools) L(p.x, p.y, p.r * 3.4, [255, 120, 30], 0.75, { flicker: true });
    if (A.id === "cavern") { for (const p of A.props) if (p.kind === "crystal") L(p.x, p.y - 30, 300, p.glow === "#9b5cff" ? [180, 110, 255] : [110, 210, 255], 0.65, { flicker: true }); L(f.cx, f.cy, 260, [90, 200, 255], 0.5, { flicker: true }); }
    if (A.id === "cosmic") { L(f.cx, f.cy, 520, [200, 80, 255], 0.35); L(f.x + f.w, f.y + f.h, 500, [255, 220, 160], 0.25); }
    if (A.id === "meadow") L(f.x + f.w * 0.8, f.y - 80, 900, [170, 190, 255], 0.35); // moonlight
  }

  // ---- runtime -------------------------------------------------------------------------------------------
  function floorFrame(A, t) { return A.frames[A.frames.length > 1 ? Math.floor(t * 3) % A.frames.length : 0]; }
  function drawTicker(ctx, A, t, ox, oy) {
    if (!A.led) return;
    const L = A.led, msg = (A.ticker + "   ·   ").toUpperCase(), tw = Pixel.textWidth(msg, 1);
    ctx.save(); ctx.beginPath(); ctx.rect(L.x - ox, L.y - oy, L.w, L.h); ctx.clip();
    let x = L.x - ox - ((t * 28) % tw);
    for (; x < L.x - ox + L.w; x += tw) Pixel.text(ctx, msg, x, L.y - oy + 2, 1, A.hype > 0.5 && Math.floor(t * 8) % 2 ? "#ffe680" : "#ff9a2a");
    ctx.restore();
  }

  // Psystrike scar: a long dark-violet gouge with ragged lips, glowing specks and cracks along the edges.
  function trench(A, wx0, wy0, wx1, wy1, ww) {
    const ctx = A.decals.getContext("2d"), f = A.af;
    const x0 = wx0 / K, y0 = wy0 / K, x1 = wx1 / K, y1 = wy1 / K, hw = ww / K / 2;
    const L = Math.hypot(x1 - x0, y1 - y0), ux = (x1 - x0) / (L || 1), uy = (y1 - y0) / (L || 1), nx = -uy, ny = ux;
    ctx.save(); U.rr(ctx, f.x, f.y, f.w, f.h, f.r); ctx.clip();
    if (A.water) { for (let d = 0; d < L; d += 6) Pixel.ring(ctx, Math.round(x0 + ux * d), Math.round(y0 + uy * d), Math.round(hw), Math.round(hw * 0.6), "rgba(230,190,255,0.45)", 1, 2); ctx.restore(); return; }
    const rng = U.seeded((Math.random() * 1e6) | 0);
    // edge profile: slow wobble + per-pixel jitter so the lips look torn rather than ruled
    const ph = rng() * 10, edge = (d, side) => hw * (0.8 + 0.18 * Math.sin(d * 0.21 + ph + side * 2) + 0.12 * Math.sin(d * 0.67 + side)) + (rng() - 0.5) * 1.6;
    for (let d = 0; d < L; d += 0.5) {
      const taper = Math.min(1, d / 6, (L - d) / 4 + 0.6), eL = edge(d, -1) * taper, eR = edge(d, 1) * taper;
      for (let o = -eL - 1.5; o <= eR + 1.5; o += 0.7) {
        const x = Math.round(x0 + ux * d + nx * o), y = Math.round(y0 + uy * d + ny * o * 0.85);
        const inner = o > -eL && o < eR, lip = o < -eL + 2 || o > eR - 2;
        ctx.fillStyle = !inner ? "rgba(30,14,22,0.55)" : lip ? "#3a1450" : rng() < 0.08 ? "#7a2aa8" : rng() < 0.5 ? "#2a0c3c" : "#1e0830";
        ctx.fillRect(x, y, 1, 1);
      }
    }
    // glowing specks and cracks spidering off the lips
    for (let d = 2; d < L; d += 3) if (rng() < 0.5) { const o = (rng() - 0.5) * hw * 1.2; ctx.fillStyle = rng() < 0.5 ? "#c060ff" : "#ff7ae0"; ctx.fillRect(Math.round(x0 + ux * d + nx * o), Math.round(y0 + uy * d + ny * o * 0.85), 1, 1); }
    for (let d = 4; d < L; d += 9 + rng() * 8) for (const side of [-1, 1]) if (rng() < 0.7) crackLine(ctx, rng, x0 + ux * d + nx * side * hw, y0 + uy * d + ny * side * hw * 0.85, "rgba(25,10,20,0.8)", 2 + (rng() * 3 | 0), A, 1.4);
    ctx.restore();
    Ground.groove(A, wx0, wy0, wx1, wy1, ww * 0.85, 7); // a real gouge: dark upper wall, lit lower wall
  }

  function scatterPebbles(ctx, A, x, y, rMin, rMax, count) {
    const [top, earth] = chunkColors(A);
    for (let i = 0; i < count; i++) {
      const a = Math.random() * U.TAU, d = U.rand(rMin, rMax);
      const px = Math.round(x + Math.cos(a) * d), py = Math.round(y + Math.sin(a) * d * 0.62);
      const w = Math.random() < 0.35 ? 2 : 1, h = Math.random() < 0.18 ? 2 : 1;
      ctx.fillStyle = "rgba(12,8,4,0.65)";
      ctx.fillRect(px, py + h, w, 1);
      ctx.fillStyle = Math.random() < 0.28 ? top : Math.random() < 0.55 ? earth : "#8b7a62";
      ctx.fillRect(px, py, w, h);
    }
  }

  function decal(A, kind, wx, wy, wr, color) {
    const ctx = A.decals.getContext("2d"), f = A.af;
    const x = Math.round(wx / K), y = Math.round(wy / K), r = Math.max(2, Math.round(wr / K));
    ctx.save(); U.rr(ctx, f.x, f.y, f.w, f.h, f.r); ctx.clip();
    const seed = (Math.random() * 1e6) | 0;
    if (kind === "frost") { ctx.restore(); Ground.freezeDisc(A, wx, wy, wr * 0.9, 11); ++A.decalCount; if (A.battle) for (let i = 0; i < 3; i++) Ground.addCrystal(A, wx + U.rand(-wr, wr) * 0.6, wy + U.rand(-wr, wr) * 0.35, U.rand(5, 11)); return; }
    if (A.iceN && Ground.iceAt(A, wx, wy) && kind !== "scorch" && kind !== "beamburn" && kind !== "clash") { ctx.restore(); if (A.battle) A.battle.fx.debris(wx, wy, 6, "#dff6ff", "#9fd2ee"); return; } // ice takes the hit instead of the ground
    if (A.water && kind !== "fissure") { // water swallows the blow: rings and spray, nothing permanent
      ctx.restore();
      if (kind === "scorch" || kind === "beamburn") { if (A.battle) A.battle.fx.add({ x: wx, y: wy, z: 4, vz: 40, life: 0.9, size: 18, color: "#f0f6ff", shape: "smoke", alpha: 0.5, grow: 1.8 }); Water.ripple(A, wx, wy, 4, 50, 0.9); return; }
      if (A.battle) Water.splash(A.battle, wx, wy, U.clamp(wr / 28, 0.5, 2.2)); else Water.ripple(A, wx, wy, 4, 60, 1);
      return;
    }
    if ((kind === "scorch" || kind === "beamburn" || kind === "clash") && A.iceN) { const n = Ground.melt(A, wx, wy, wr * 1.3); if (n && A.battle) for (let i = 0; i < Math.min(8, 2 + n / 6); i++) A.battle.fx.add({ x: wx + U.rand(-wr, wr) * 0.6, y: wy + U.rand(-wr, wr) * 0.3, z: 4, vx: U.rand(-20, 20), vz: U.rand(40, 90), life: U.rand(0.8, 1.4), size: U.rand(12, 20), color: "#f4f8ff", shape: "smoke", alpha: 0.45, grow: 1.8 }); }
    switch (kind) {
      case "scorch": {
        const [, earth] = chunkColors(A);
        Pixel.dither(ctx, x, y, Math.round(r * 1.15), Math.round(r * 0.72), U.rgba(earth, 0.5), (k) => (1 - k) * 0.72, seed);
        Pixel.dither(ctx, x, y, r, Math.round(r * 0.65), "rgba(28,16,9,0.58)", (k) => (1 - k) * 0.78, seed + 1);
        Pixel.dither(ctx, x, y, Math.round(r * 0.52), Math.round(r * 0.34), "rgba(12,6,3,0.82)", (k) => 1 - k, seed + 2);
        scatterPebbles(ctx, A, x, y, r * 0.55, r * 1.35, Math.max(3, Math.round(r * 0.65)));
        if (r >= 8) Ground.crater(A, wx, wy, wr * 0.75, U.clamp(wr / K * 0.14, 1.4, 4.2));
        break;
      }
      case "beamburn": {
        const [, earth] = chunkColors(A);
        Pixel.dither(ctx, x, y, r, Math.round(r * 0.62), U.rgba(earth, 0.52), (k) => (1 - k) * 0.75, seed);
        Pixel.dither(ctx, x, y, Math.round(r * 0.68), Math.round(r * 0.42), "rgba(16,8,4,0.78)", (k) => 1 - k * 0.7, seed + 3);
        scatterPebbles(ctx, A, x, y, r * 0.7, r * 1.3, 3);
        Ground.crater(A, wx, wy, wr * 0.8, U.clamp(wr / K * 0.16, 1.5, 4.5));
        break;
      }
      case "crater": { // exposed soil + thrown clods + pebbles + physical relief crater
        const [, earth] = chunkColors(A);
        Pixel.dither(ctx, x, y, Math.round(r * 1.28), Math.round(r * 0.8), "rgba(120,96,62,0.55)", (k) => (1 - k) * 0.9, seed);
        Pixel.ellipse(ctx, x, y, r, Math.round(r * 0.62), U.rgba(earth, 0.82));
        Pixel.dither(ctx, x, y + 1, Math.round(r * 0.7), Math.round(r * 0.42), "rgba(18,10,5,0.8)", (k) => 1 - k * 0.6, seed + 2);
        for (let i = 0; i < 5; i++) { const a = Math.random() * U.TAU; crackLine(ctx, Math.random, x + Math.cos(a) * r * 0.85, y + Math.sin(a) * r * 0.52, "rgba(36,20,8,0.68)", 2 + (r / 12 | 0), A, 0.95); }
        scatterPebbles(ctx, A, x, y, r * 0.85, r * 1.75, Math.max(6, Math.round(r * 1.35)));
        Ground.crater(A, wx, wy, wr, U.clamp(wr / K * 0.3, 3.2, 11));
        break;
      }
      case "clash": { // dramatic radial shock-crater at beam / projectile clash nexus
        const [, earth] = chunkColors(A);
        Pixel.dither(ctx, x, y, Math.round(r * 1.4), Math.round(r * 0.88), U.rgba(earth, 0.62), (k) => (1 - k) * 0.92, seed);
        Pixel.ring(ctx, x, y, Math.round(r * 1.08), Math.round(r * 0.68), "rgba(235,220,185,0.42)", 1, 2);
        Pixel.dither(ctx, x, y, Math.round(r * 0.78), Math.round(r * 0.48), "rgba(16,9,5,0.84)", (k) => 1 - k * 0.55, seed + 4);
        for (let i = 0; i < 7; i++) { const a = (i / 7) * U.TAU + U.rand(-0.25, 0.25); crackLine(ctx, Math.random, x + Math.cos(a) * r * 0.5, y + Math.sin(a) * r * 0.32, "rgba(28,15,6,0.78)", 3 + (r / 9 | 0), A, 1.25); }
        scatterPebbles(ctx, A, x, y, r * 0.8, r * 1.9, Math.max(10, Math.round(r * 1.7)));
        Ground.crater(A, wx, wy, wr * 1.15, U.clamp(wr / K * 0.34, 4, 12));
        break;
      }
      case "skid": { // knockback / dash turf gouge + kicked pebbles
        const [, earth] = chunkColors(A);
        Pixel.dither(ctx, x, y, r, Math.max(2, Math.round(r * 0.55)), U.rgba(earth, 0.62), (k) => (1 - k) * 0.85, seed);
        scatterPebbles(ctx, A, x, y, r * 0.4, r * 1.35, 3);
        Ground.crater(A, wx, wy, wr * 0.65, 1.5);
        break;
      }
      case "rubble": { // churned-up soil over a tunnel: loose clods, one short split
        const [top, earth] = chunkColors(A);
        Pixel.dither(ctx, x, y, r, Math.round(r * 0.6), U.rgba(earth, 0.55), (k) => (1 - k) * 0.8, seed);
        for (let i = 0; i < r * 1.5; i++) { const a = Math.random() * U.TAU, d = Math.random() * r; ctx.fillStyle = Math.random() < 0.3 ? top : Math.random() < 0.5 ? earth : "#3e3424"; ctx.fillRect(Math.round(x + Math.cos(a) * d), Math.round(y + Math.sin(a) * d * 0.6), 1 + (Math.random() < 0.25 ? 1 : 0), 1); }
        scatterPebbles(ctx, A, x, y, r * 0.5, r * 1.25, Math.max(3, Math.round(r * 0.6)));
        if (Math.random() < 0.5) crackLine(ctx, Math.random, x, y, "rgba(40,24,10,0.55)", 2, A, 0.7);
        break;
      }
      case "crack":
        for (let i = 0; i < 5; i++) crackLine(ctx, Math.random, x, y, color || "rgba(25,15,6,0.78)", 3 + (r / 6 | 0), A, 1.15);
        scatterPebbles(ctx, A, x, y, 2, r * 1.15, Math.max(4, Math.round(r * 0.5)));
        break;
      case "fissure":
        for (let i = 0; i < 4; i++) crackLine(ctx, Math.random, x, y, "#3a0f55", 4 + (r / 5 | 0), A, 1.35);
        Pixel.dither(ctx, x, y, r, Math.round(r * 0.6), "rgba(190,80,255,0.58)", (k) => (1 - k) * 0.5, seed);
        scatterPebbles(ctx, A, x, y, r * 0.5, r * 1.35, 5);
        break;
      case "puddle": Pixel.dither(ctx, x, y, r, Math.round(r * 0.6), color || "rgba(70,130,220,0.75)", (k) => (1 - k) * 0.95, seed); break;
      case "frost": Pixel.dither(ctx, x, y, r, Math.round(r * 0.6), "rgba(230,248,255,0.9)", (k) => (1 - k) * 0.8, seed); break;
      case "slash": for (let i = 0; i < 3; i++) { const a = Math.random() * U.TAU; Pixel.line(ctx, x + Math.cos(a) * r, y + Math.sin(a) * r * 0.6, x - Math.cos(a) * r, y - Math.sin(a) * r * 0.6, "rgba(20,14,8,0.65)"); } break;
      case "stain": Pixel.dither(ctx, x, y, r, Math.round(r * 0.6), color || "rgba(120,40,160,0.7)", (k) => (1 - k) * 0.8, seed); break;
      case "leaves": for (let i = 0; i < r; i++) { ctx.fillStyle = ["#3f8f2a", "#6bc046", "#2b6a1c"][i % 3]; ctx.fillRect(x + (Math.random() - 0.5) * r * 2 | 0, y + (Math.random() - 0.5) * r | 0, 2, 1); } break;
    }
    ctx.restore();
    // Permanent field destruction: scars, craters, trenches and pebbles persist until the match ends!
    ++A.decalCount;
  }

  // Keep a circle inside the rounded field and out of solid props.
  function resolve(A, x, y, r) {
    const f = A.field, R = f.R;
    x = U.clamp(x, f.x + r, f.x + f.w - r); y = U.clamp(y, f.y + r, f.y + f.h - r);
    for (const [cx, cy] of [[f.x + R, f.y + R], [f.x + f.w - R, f.y + R], [f.x + R, f.y + f.h - R], [f.x + f.w - R, f.y + f.h - R]]) {
      const inC = (cx === f.x + R ? x < cx : x > cx) && (cy === f.y + R ? y < cy : y > cy);
      if (!inC) continue;
      const d = U.dist(x, y, cx, cy), max = R - r;
      if (d > max) { x = cx + ((x - cx) / d) * max; y = cy + ((y - cy) / d) * max; }
    }
    for (const p of A.props) {
      if (!p.alive) continue;
      const d = U.dist(x, y, p.x, p.y), min = p.r + r * 0.8;
      if (d < min && d > 0.01) { x = p.x + ((x - p.x) / d) * min; y = p.y + ((y - p.y) / d) * min; }
    }
    return { x, y };
  }
  // What the ground is at a world point: "lava" | "water" | "platform" | "ice" | "star" | null.
  function terrainAt(A, x, y) {
    if (A.iceN && Ground.iceAt(A, x, y)) return "ice";
    for (const p of A.pools) if (((x - p.x) ** 2) / (p.r ** 2) + ((y - p.y) ** 2) / ((p.r * 0.72) ** 2) < 1) return p.kind;
    if (A.water) { for (const p of A.platforms) if (((x - p.x) ** 2) / (p.r ** 2) + ((y - p.y) ** 2) / ((p.r * 0.62) ** 2) < 1.05) return "platform"; return "water"; }
    if (A.ice) return "ice";
    if (A.def.starpool) { const f = A.field; if (((x - f.cx) ** 2) / (150 ** 2) + ((y - f.cy) ** 2) / (80 ** 2) < 1) return "star"; }
    return null;
  }
  function damageProp(A, p, dmg, fx) {
    if (!p.alive) return;
    p.shake = 0.25;
    if (p.maxHp < 900) { p.hp -= dmg; if (p.hp <= 0) { p.alive = false; if (fx) fx.debris(p.x, p.y - 16, 16, p.kind === "iceblock" ? "#bfe6ff" : "#b47a3c", p.kind === "iceblock" ? "#ffffff" : "#6a4420"); Audio.sfx("rock", 0); } }
  }

  return { K, DEFS, list, create, floorFrame, drawTicker, decal, trench, chunkColors, resolve, terrainAt, damageProp };
})();
