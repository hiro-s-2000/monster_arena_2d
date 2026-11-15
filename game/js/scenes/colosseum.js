// Sand Colosseum — 3D Monumental Roman Travertine Amphitheatre (arcade arches, Emperor box, cypresses)
"use strict";

Scene3D.define("colosseum", {
  walls: false,
  paint(ctx, A, rng) {
    const aw = A.aw, ah = A.ah, af = A.af;
    ctx.fillStyle = "#c8a876";
    ctx.fillRect(0, 0, aw, ah);
    // Roman stone flagstone concourse outside sand ring
    ctx.fillStyle = "#98866c";
    ctx.fillRect(af.x - 20, af.y - 20, af.w + 40, af.h + 40);
    ctx.strokeStyle = "#d8c4a2";
    ctx.lineWidth = 2;
    ctx.strokeRect(af.x - 4, af.y - 4, af.w + 8, af.h + 8);
    for (let i = 0; i < 900; i++) {
      ctx.fillStyle = rng() < 0.5 ? "#7e6e58" : "#b8a68a";
      ctx.fillRect(rng() * aw | 0, rng() * ah | 0, 2, 1);
    }
  },

  build(S, A) {
    const aw = S.aw, ah = S.ah, af = S.af, fy = S.frontY, cx = aw / 2;
    S.set({
      clear: "#3264a6",
      fog: { color: "#9ac2de", near: 620, far: 1750, max: 0.48 },
      light: [-0.62, -0.32, 0.72],
      keyTint: [1.16, 1.08, 0.88],
      shadowTint: [0.34, 0.36, 0.58],
      castCol: [0.05, 0.04, 0.14],
      pitch: 37,
      dof: 1.35,
    });

    // ---- 1. Warm Mediterranean Noon Sky & Distant Roman Aqueduct Horizon ----
    const skyT = S.tex(240, 110, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h);
      gr.addColorStop(0, "#2a5ca4"); gr.addColorStop(0.55, "#6ea8de"); gr.addColorStop(1, "#e6d8c0");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
      g.fillStyle = "rgba(255,248,228,0.85)";
      g.fillRect(42, 22, 46, 7); g.fillRect(48, 17, 28, 6);
      g.fillRect(160, 28, 42, 6);
      g.fillStyle = "#8a785c";
      for (let x = 0; x < w; x += 2) g.fillRect(x, h - 16 - Math.sin(x * 0.035) * 6, 2, 22);
    });
    S.backdrop(skyT, cx - 720, cx + 720, -320, -100, 220);

    const outSand = S.tex(48, 48, (g) => {
      g.fillStyle = "#b89c6c"; g.fillRect(0, 0, 48, 48);
      const r = U.seeded(84);
      for (let i = 0; i < 200; i++) { g.fillStyle = r() < 0.5 ? "#9c8258" : "#ceb280"; g.fillRect(r() * 48 | 0, r() * 48 | 0, 2, 1); }
    });
    S.ground(outSand, -460, -280, aw + 460, ah + 260, { texel: 1.5, fog: 0.9 });

    // ---- 2. Travertine Ashlar, Double Roman Archways & Fluted Columns ----
    const trav = S.tex(32, 24, (g) => {
      g.fillStyle = "#d4ba8e"; g.fillRect(0, 0, 32, 24);
      g.fillStyle = "#a28a64"; for (let y = 0; y < 24; y += 6) g.fillRect(0, y, 32, 1);
      for (let y = 0; y < 24; y += 6) for (let x = ((y / 6) % 2) * 8; x < 32; x += 16) g.fillRect(x, y, 1, 6);
      g.fillStyle = "#eed8b0"; g.fillRect(0, 1, 32, 1);
    });
    const arcade = S.tex(48, 32, (g) => {
      g.fillStyle = "#c8ac82"; g.fillRect(0, 0, 48, 32);
      // 2 recessed Roman arches per 48px tile!
      for (const ax0 of [4, 28]) {
        g.fillStyle = "#261a26";
        g.fillRect(ax0, 10, 16, 22);
        g.beginPath(); g.arc(ax0 + 8, 10, 8, Math.PI, 0); g.fill();
        g.fillStyle = "#eed8ae"; g.fillRect(ax0 - 2, 6, 20, 2);
      }
      g.fillStyle = "#987c54"; g.fillRect(0, 0, 48, 3);
    });
    const crowd = S.tex(80, 14, (g) => {
      const r = U.seeded(554), togas = ["#f2ead8", "#c84038", "#3a6eb8", "#d8a43c", "#6c3a8c"];
      g.fillStyle = "#8a7254"; g.fillRect(0, 9, 80, 5);
      for (let x = 2; x < 78; x += 5) {
        g.fillStyle = togas[(r() * togas.length) | 0]; g.fillRect(x, 6, 4, 6);
        g.fillStyle = "#e8bc8e"; g.fillRect(x + 1, 2, 2, 4);
      }
    });

    // Inner stone amphitheatre parapet + stepped seating bays + double arcade upper wall
    S.wall(trav, af.x - 14, af.y - 12, af.x + af.w + 14, af.y - 12, 0, 14, { texel: 1 });
    S.wall(trav, af.x - 14, af.y - 12, af.x - 14, fy + 12, 0, 14, { texel: 1 });
    S.wall(trav, af.x + af.w + 14, fy + 12, af.x + af.w + 14, af.y - 12, 0, 14, { texel: 1 });

    // North tiered stone seating & upper Roman arcade + protruding Corinthian pilasters!
    for (let t = 0; t < 3; t++) {
      const by = af.y - 26 - t * 14, bz = 12 + t * 9;
      S.box(af.x - 18, by, af.w + 36, 14, 0, bz, { all: trav, top: S.flat("#e6cf9f") }, { texel: 1 });
      S.wall(crowd, af.x - 10, by + 11, af.x + af.w + 10, by + 11, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.7 } });
    }
    S.box(af.x - 28, af.y - 76, af.w + 56, 22, 0, 62, { front: arcade, top: trav, left: trav, right: trav }, { texel: 1, shadow: 0.52 });
    // Sculpted Roman columns standing proud along north parapet casting sharp vertical-diagonal shadow stripes
    for (let px = af.x - 16; px <= af.x + af.w + 16; px += 48) {
      S.prism(S.flat("#d4bc90"), px, af.y - 54, 4, 3.4, 36, 30, 8, { top: S.flat("#dfcb9e") });
      S.box(px - 5, af.y - 59, 10, 10, 64, 4, { all: S.flat("#dac496") });
    }

    // West high Roman colosseum wall + arcade (casts dramatic cool shadow right across west sand apron!)
    for (let t = 0; t < 3; t++) {
      const bx = af.x - 28 - t * 14, bz = 12 + t * 9;
      S.box(bx, af.y - 12, 14, af.h + 24, 0, bz, { all: trav, top: S.flat("#cbb082") }, { texel: 1 });
      S.wall(crowd, bx + 9, af.y - 8, bx + 9, fy + 8, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.7 } });
    }
    S.box(af.x - 84, af.y - 68, 28, af.h + 84, 0, 62, { all: arcade, top: trav }, { texel: 1, shadow: 0.54 });
    for (let py = af.y - 40; py <= fy + 10; py += 52) {
      S.prism(S.flat("#d4bc90"), af.x - 54, py, 4.5, 3.8, 36, 30, 8, { top: S.flat("#dfcb9e") });
    }

    // East Roman amphitheatre stands & arcade
    for (let t = 0; t < 2; t++) {
      const bx = af.x + af.w + 16 + t * 14, bz = 12 + t * 9;
      S.box(bx, af.y - 12, 14, af.h + 24, 0, bz, { all: trav, top: S.flat("#cbb082") }, { texel: 1 });
      S.wall(crowd, bx + 4, fy + 8, bx + 4, af.y - 8, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.7 } });
    }
    S.box(af.x + af.w + 46, af.y - 68, 26, af.h + 84, 0, 56, { all: arcade, top: trav }, { texel: 1, shadow: 0.48 });

    // ---- 3. Grand Emperor's Silk-Canopy Imperial Box & Crimson Banners ----
    S.house3D(cx - 42, af.y - 64, 34, {
      w: 84, d: 34, h: 24, rh: 16, ov: 4, ridge: "ew",
      wall: "#eee0c2", trim: "#c8a048", roof: "#9c2832", roofHi: "#dc5246", roofSh: "#58161e",
      doorSide: "c", chimney: false, plinth: 2, shadow: false,
    });
    // Flaming Bronze Tripod Braziers on stone plinths
    for (const [bx, by, bz] of [[cx - 56, af.y - 36, 36], [cx + 56, af.y - 36, 36], [af.x - 16, af.y - 16, 14], [af.x + af.w + 16, af.y - 16, 14]]) {
      S.lamp3D(bx, by, bz, { h: 16, post: "#6a4834", color: "#ff9a3c", r: 46, i: 0.48, shadow: false });
    }

    // ---- 4. Sculpted Mediterranean Umbrella Pines & Cypresses around outer ruin ring ----
    const cyPal = ["#1c3e26", "#2f5f34", "#49843b", "#6ea84b"];
    const umPal = ["#244a2a", "#386c36", "#5a9a46", "#88c660"];
    for (const [k, tx, ty, th, tr] of [
      ["c", af.x - 94, 72, 62, 11], ["u", af.x - 94, 210, 50, 17],
      ["c", af.x - 102, 42, 66, 12], ["u", af.x - 106, 140, 54, 19], ["c", af.x - 102, 240, 66, 12], ["u", af.x - 106, 340, 52, 18],
      ["c", cx - 120, -96, 62, 12], ["c", cx + 120, -96, 62, 12],
      ["u", af.x + af.w + 92, 66, 54, 19], ["c", af.x + af.w + 88, 180, 66, 12], ["u", af.x + af.w + 92, 300, 54, 18],
    ]) {
      if (k === "c") S.tree3D(tx, ty, 0, { kind: "cypress", h: th, r: tr, pal: cyPal, anim: { type: "sway", amp: 0.45 }, shadow: 0.5 });
      else S.tree3D(tx, ty, 0, { kind: "oak", h: th, r: tr, pal: umPal, anim: { type: "sway", amp: 0.6 }, shadow: 0.5 });
    }

    // ---- 5. Foreground Carved Balustrade & Flaming Brazier (Front Bokeh) ----
    S.box(af.x - 24, fy + 14, af.w + 48, 12, 0, 11, { all: trav, top: S.flat("#ebd6ac") }, { layer: "front", shadow: 0.4 });
    S.tree3D(af.x - 36, fy + 24, 0, { kind: "cypress", h: 58, r: 11, pal: cyPal, layer: "front", anim: { type: "sway", amp: 0.5 }, shadow: 0.4 });
    S.tree3D(af.x + af.w + 36, fy + 24, 0, { kind: "cypress", h: 58, r: 11, pal: cyPal, layer: "front", anim: { type: "sway", amp: 0.5 }, shadow: 0.4 });
  },
});
