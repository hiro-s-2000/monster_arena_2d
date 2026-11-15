// Frost Rink — 3D Alpine Snow Village & Crystal Glacier Arena (snow chalets, ice monoliths, blue shadows)
"use strict";

Scene3D.define("ice", {
  walls: false,
  paint(ctx, A, rng) {
    const aw = A.aw, ah = A.ah, af = A.af;
    ctx.fillStyle = "#bddcf0";
    ctx.fillRect(0, 0, aw, ah);
    // Crisp winter drift pockets (cool icy shadows vs sparkling snow ridges)
    for (let i = 0; i < 160; i++) {
      ctx.fillStyle = rng() < 0.45 ? "#d8f0fc" : "#96bdd8";
      ctx.fillRect(rng() * aw | 0, rng() * ah | 0, 8 + (rng() * 18 | 0), 4 + (rng() * 10 | 0));
    }
    for (let i = 0; i < 1800; i++) {
      ctx.fillStyle = rng() < 0.4 ? "#ffffff" : "#80adc8";
      ctx.fillRect(rng() * aw | 0, rng() * ah | 0, 1 + (rng() < 0.3 ? 1 : 0), 1);
    }
    // Timber & stone snowy boardwalk around rink
    ctx.fillStyle = "#5c687c";
    ctx.fillRect(af.x - 16, af.y - 16, af.w + 32, af.h + 32);
    ctx.fillStyle = "#b4d2e8";
    ctx.fillRect(af.x - 14, af.y - 14, af.w + 28, 3);
  },

  build(S, A) {
    const aw = S.aw, ah = S.ah, af = S.af, fy = S.frontY, cx = aw / 2;
    S.set({
      clear: "#284c7c",
      fog: { color: "#8ac0ea", near: 620, far: 1750, max: 0.48 },
      light: [-0.62, -0.28, 0.72],
      keyTint: [1.08, 1.12, 1.18],
      shadowTint: [0.34, 0.48, 0.78],
      castCol: [0.07, 0.16, 0.34],
      pitch: 37,
      dof: 1.35,
    });

    // ---- 1. Alpine Jagged Peaks & Pastel Aurora Sky ----
    const skyT = S.tex(240, 110, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h);
      gr.addColorStop(0, "#1c3866"); gr.addColorStop(0.5, "#4c86ba"); gr.addColorStop(1, "#bde0f6");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
      // Shimmering faint winter aurora curve
      g.fillStyle = "rgba(130,248,215,0.22)";
      for (let x = 0; x < w; x += 2) {
        const y = 26 + Math.sin(x * 0.05) * 9;
        g.fillRect(x, y, 2, 12);
      }
      // Sharp snow-capped mountain peaks
      const peaks = [[0, 68], [32, 34], [62, 58], [98, 22], [134, 52], [172, 26], [208, 54], [240, 40]];
      g.fillStyle = "#5278a2";
      g.beginPath(); g.moveTo(0, h);
      for (const [px, py] of peaks) g.lineTo(px, py);
      g.lineTo(w, h); g.closePath(); g.fill();
      // Sunlit snow on left facets
      g.fillStyle = "#e6f5ff";
      for (let i = 1; i < peaks.length - 1; i += 2) {
        g.beginPath(); g.moveTo(peaks[i - 1][0], peaks[i - 1][1]); g.lineTo(peaks[i][0], peaks[i][1]); g.lineTo(peaks[i][0] + 6, peaks[i][1] + 22); g.closePath(); g.fill();
      }
    });
    S.backdrop(skyT, cx - 720, cx + 720, -320, -100, 220);

    const outSnow = S.tex(64, 64, (g) => {
      g.fillStyle = "#bddcf0"; g.fillRect(0, 0, 64, 64);
      const r = U.seeded(91);
      for (let i = 0; i < 320; i++) {
        g.fillStyle = r() < 0.5 ? "#d2ecfa" : "#98c0dc";
        g.fillRect(r() * 64 | 0, r() * 64 | 0, 2, 1);
      }
    });
    S.ground(outSnow, -460, -280, aw + 460, ah + 260, { texel: 1.5, fog: 0.9 });

    // ---- 2. Hockey Rink Dasher Boards + Cozy Alpine Snow Bleachers ----
    const boardT = S.tex(140, 12, (g) => {
      g.fillStyle = "#eaf6ff"; g.fillRect(0, 0, 140, 10);
      g.fillStyle = "#2b6eb8"; g.fillRect(0, 8, 140, 2);
      g.fillStyle = "#c83c40"; g.fillRect(0, 0, 140, 2);
      g.fillStyle = "#1f4e8c"; Pixel.text(g, "SNOW CUP   FROST RINK   HOT COCOA", 8, 3, 1);
    });
    const crT = S.tex(80, 14, (g) => {
      const r = U.seeded(332), coats = ["#c83840", "#2f6cb8", "#f0a038", "#3d946c", "#d8e8f4"];
      g.fillStyle = "#485872"; g.fillRect(0, 9, 80, 5);
      for (let x = 2; x < 78; x += 6) {
        g.fillStyle = coats[(r() * coats.length) | 0]; g.fillRect(x, 6, 4, 6);
        g.fillStyle = "#f2c8a2"; g.fillRect(x + 1, 3, 2, 3);
        g.fillStyle = r() < 0.5 ? "#e65258" : "#dceeff"; g.fillRect(x, 1, 4, 2);
      }
    });
    const timber = S.tex(24, 16, (g) => {
      g.fillStyle = "#5c4230"; g.fillRect(0, 0, 24, 16);
      g.fillStyle = "#dff2ff"; g.fillRect(0, 0, 24, 3);
    });

    // Rink boards enclosing combat zone cleanly
    S.wall(boardT, af.x - 10, af.y - 8, af.x + af.w + 10, af.y - 8, 0, 11, { texel: 1 });
    S.wall(boardT, af.x - 10, af.y - 8, af.x - 10, fy + 8, 0, 11, { texel: 1 });
    S.wall(boardT, af.x + af.w + 10, fy + 8, af.x + af.w + 10, af.y - 8, 0, 11, { texel: 1 });

    // Lower terraced wooden spectators north & west
    for (let t = 0; t < 2; t++) {
      const bz = 10 + t * 8, by = af.y - 22 - t * 14;
      S.box(af.x - 14, by, af.w + 28, 14, 0, bz, { all: timber, top: timber }, { texel: 1 });
      S.wall(crT, af.x - 8, by + 11, af.x + af.w + 8, by + 11, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.7 } });
    }
    for (let t = 0; t < 2; t++) {
      const bx = af.x - 26 - t * 14, bz = 10 + t * 8;
      S.box(bx, af.y - 8, 14, af.h + 16, 0, bz, { all: timber, top: timber }, { texel: 1 });
      S.wall(crT, bx + 9, af.y - 4, bx + 9, fy + 4, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.7 } });
    }

    // ---- 3. 3D Cozy Snowy Alpine Ski Chalets & Snowbanks ----
    S.house3D(af.x + 28, 4, 0, {
      w: 58, d: 34, h: 25, rh: 15, ov: 3.5, ridge: "ew",
      wall: "#7c5236", trim: "#d6e8f6", roof: "#c8e6fa", roofHi: "#ffffff", roofSh: "#6698c8",
      lit: true, chimney: true, doorSide: "l", shadow: 0.5,
    });
    S.house3D(af.x + af.w - 86, 4, 0, {
      w: 52, d: 32, h: 24, rh: 15, ov: 3, ridge: "ns",
      wall: "#6a4832", trim: "#dcecf8", roof: "#c8e6fa", roofHi: "#ffffff", roofSh: "#6698c8",
      lit: true, chimney: true, doorSide: "r", shadow: 0.5,
    });
    S.house3D(af.x - 134, 96, 0, {
      w: 44, d: 30, h: 22, rh: 13, ov: 3, ridge: "ns",
      wall: "#785036", trim: "#dceeff", roof: "#d0ebff", roofHi: "#ffffff", roofSh: "#6e9ece",
      lit: true, chimney: true, doorSide: "r", shadow: 0.5,
    });
    // ---- 4. Sculpted Faceted 3D Ice Crystal Monoliths + Snowy Conical Pines ----
    const crysT = S.tex(24, 24, (g) => {
      const gr = g.createLinearGradient(0, 0, 24, 24);
      gr.addColorStop(0, "#ffffff"); gr.addColorStop(0.4, "#9ee6ff"); gr.addColorStop(1, "#3884c8");
      g.fillStyle = gr; g.fillRect(0, 0, 24, 24);
      g.fillStyle = "rgba(255,255,255,0.65)"; g.fillRect(2, 0, 2, 24); g.fillRect(10, 4, 1, 14);
    });
    const crysCluster = (cx0, cy0, sc = 1) => {
      S.prism(crysT, cx0, cy0, 9 * sc, 2 * sc, 0, 38 * sc, 5, { rot: 0.4, top: S.flat("#eefaff"), emit: 0.25 });
      S.prism(crysT, cx0 - 9 * sc, cy0 + 4 * sc, 6 * sc, 1.5 * sc, 0, 24 * sc, 5, { rot: 0.8, top: S.flat("#ffffff"), emit: 0.22 });
      S.prism(crysT, cx0 + 8 * sc, cy0 + 5 * sc, 7 * sc, 1.8 * sc, 0, 28 * sc, 5, { rot: 0.1, top: S.flat("#ddf4ff"), emit: 0.22 });
      S.castConeShadow(cx0, cy0, 11 * sc, 36 * sc, 0.44);
      S.glow(cx0, cy0, 18 * sc, 26 * sc, "#88dcff", { alpha: 0.34, anim: { type: "pulse", amp: 0.16 } });
    };
    crysCluster(af.x - 78, 56, 1.15);
    crysCluster(cx, 12, 1.2);
    crysCluster(af.x - 102, 232, 1.1);
    crysCluster(af.x + af.w + 46, 70, 1.1);
    crysCluster(af.x + af.w + 48, 290, 0.95);
    const snowPine = ["#27495e", "#46768c", "#cce8fa", "#ffffff"];
    for (const [px, py, ph, pr] of [
      [-10, 44, 56, 17], [34, 34, 52, 15], [52, 148, 52, 16], [14, 174, 56, 17],
      [48, 224, 54, 16], [12, 266, 56, 17], [48, 308, 52, 16], [20, 356, 54, 16], [56, 392, 50, 15],
      [cx - 86, -24, 54, 16], [cx - 32, -28, 50, 15], [cx + 42, -26, 54, 16], [cx + 98, -20, 50, 15],
      [aw - 34, 32, 52, 15], [aw - 38, 156, 50, 15], [aw - 34, 232, 52, 15], [aw - 36, 348, 50, 15],
    ]) {
      S.pine3D(px, py, 0, { h: ph, r: pr, tiers: 5, pal: snowPine, anim: { type: "sway", amp: 0.5 }, shadow: 0.5 });
    }

    // Warm Amber Alpine Snow Lanterns (fire & ice palette harmony!)
    for (const [lx, ly] of [[af.x - 16, af.y - 12], [af.x + 150, af.y - 14], [af.x + af.w - 140, af.y - 14], [af.x + af.w + 16, af.y - 12]]) {
      S.lamp3D(lx, ly, 0, { h: 24, color: "#ffbf64", r: 58, i: 0.65 });
    }

    // ---- 5. Foreground Snowy Timber Rail & Pines (Front Bokeh Frame) ----
    S.fence3D(af.x - 14, fy + 14, af.x + af.w + 14, fy + 14, 0, { h: 9, step: 18, col: "#6c523e", cap: "#e2f4ff", layer: "front" });
    S.pine3D(af.x - 36, fy + 28, 0, { h: 48, r: 15, tiers: 4, pal: snowPine, layer: "front", anim: { type: "sway", amp: 0.6 }, shadow: 0.44 });
    S.pine3D(af.x + af.w + 36, fy + 28, 0, { h: 48, r: 15, tiers: 4, pal: snowPine, layer: "front", anim: { type: "sway", amp: 0.6 }, shadow: 0.44 });
    S.lamp3D(af.x + 80, fy + 16, 0, { h: 22, color: "#ffbf64", r: 50, i: 0.5, layer: "front" });
  },
});
