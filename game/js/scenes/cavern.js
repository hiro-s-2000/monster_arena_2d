// Crystal Cavern — 3D Amethyst Vault, Bioluminescent Waterfall Lagoon & Crystal Groves
"use strict";

Scene3D.define("cavern", {
  walls: false,
  paint(ctx, A, rng) {
    const aw = A.aw, ah = A.ah, af = A.af;
    ctx.fillStyle = "#161434";
    ctx.fillRect(0, 0, aw, ah);
    // Shimmering subterranean mirror lagoons bordering west & north
    ctx.fillStyle = "#123266";
    ctx.fillRect(0, 0, aw, af.y - 10);
    ctx.fillRect(0, 0, af.x - 16, ah);
    for (let i = 0; i < 520; i++) {
      const px = rng() * aw, py = rng() * ah;
      if (px > af.x - 6 && px < af.x + af.w + 6 && py > af.y - 6 && py < af.y + af.h + 6) continue;
      ctx.fillStyle = rng() < 0.35 ? "#5ea8ff" : rng() < 0.72 ? "#284c8e" : "#8854d8";
      ctx.fillRect(px | 0, py | 0, 3 + (rng() * 5 | 0), 1);
    }
    // Warm slate walkway edging & crystal trim around battle clearing
    ctx.fillStyle = "#3a346a";
    ctx.fillRect(af.x - 12, af.y - 12, af.w + 24, af.h + 24);
    ctx.strokeStyle = "#708cf2";
    ctx.lineWidth = 2;
    ctx.strokeRect(af.x - 9, af.y - 9, af.w + 18, af.h + 18);
  },

  build(S, A) {
    const aw = S.aw, ah = S.ah, af = S.af, fy = S.frontY, cx = aw / 2;
    S.set({
      clear: "#08061c",
      fog: { color: "#141038", near: 540, far: 1650, max: 0.5 },
      light: [-0.58, -0.32, 0.75],
      keyTint: [1.06, 1.12, 1.32],
      shadowTint: [0.28, 0.24, 0.52],
      castCol: [0.02, 0.01, 0.08],
      pitch: 37,
      dof: 1.35,
    });

    // ---- 1. Subterranean Grotto Dome & Bioluminescent Lagoon Floor ----
    const grottoBg = S.tex(240, 110, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h);
      gr.addColorStop(0, "#080618"); gr.addColorStop(0.55, "#1c1848"); gr.addColorStop(1, "#322870");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
      const r = U.seeded(844);
      for (let i = 0; i < 95; i++) {
        g.fillStyle = r() < 0.5 ? "#7ee4ff" : "#e884ff";
        g.fillRect(r() * w | 0, r() * 68 | 0, 1 + (r() < 0.35 ? 1 : 0), 1);
      }
      g.fillStyle = "#100c28";
      for (let x = 0; x < w; x += 6) {
        const sh = 12 + ((x * 13) % 24);
        g.beginPath(); g.moveTo(x, 0); g.lineTo(x + 6, 0); g.lineTo(x + 3, sh); g.closePath(); g.fill();
      }
    });
    S.backdrop(grottoBg, cx - 720, cx + 720, -320, -100, 220);

    const lakeT = S.tex(64, 64, (g) => {
      g.fillStyle = "#102854"; g.fillRect(0, 0, 64, 64);
      const r = U.seeded(319);
      for (let i = 0; i < 70; i++) {
        g.fillStyle = r() < 0.45 ? "#4898e8" : "#224684";
        g.fillRect(r() * 60 | 0, r() * 60 | 0, 5 + (r() * 6 | 0), 1);
      }
    });
    S.ground(lakeT, -460, -280, aw + 460, ah + 260, { z: -6, texel: 2, emit: 0.25, lit: 0, fog: 0.78 });

    // ---- 2. Stratified Slate Cavern Wall, Underground Waterfall & Warm Mine Bridges ----
    const slate = S.tex(32, 28, (g) => {
      g.fillStyle = "#2e2a58"; g.fillRect(0, 0, 32, 28);
      g.fillStyle = "#4d4886"; g.fillRect(0, 0, 32, 4); g.fillRect(0, 12, 32, 2);
      g.fillStyle = "#1a1838"; g.fillRect(0, 22, 32, 6);
      g.fillStyle = "#76dcff"; g.fillRect(6, 15, 10, 1);
    });
    const wfT = S.tex(24, 32, (g) => {
      const gr = g.createLinearGradient(0, 0, 24, 0);
      gr.addColorStop(0, "#245ca8"); gr.addColorStop(0.5, "#a4f4ff"); gr.addColorStop(1, "#245ca8");
      g.fillStyle = gr; g.fillRect(0, 0, 24, 32);
      g.fillStyle = "#ffffff"; for (let y = 0; y < 32; y += 6) g.fillRect(10, y, 4, 3);
    });

    // North cave cliff brought cleanly in frame + glowing bioluminescent waterfall!
    S.box(-120, -68, aw + 240, 52, -6, 60, { front: slate, top: S.flat("#3a366c"), left: slate, right: slate }, { texel: 1.5, shadow: 0.45 });
    S.wall(wfT, cx - 26, -15, cx + 26, -15, -6, 54, { emit: 0.92, lit: 0, anim: { type: "pulse", amp: 0.14 } });
    S.glow(cx, -12, 18, 46, "#78ecff", { alpha: 0.52 });
    S.lamp(cx, 16, 140, [110, 230, 255], 0.7);

    // Timber Miner's Suspension Walkways & Warm Amber Lanterns along West & North Rim (Octopath contrast!)
    S.box(af.x - 42, af.y - 12, 26, af.h + 24, 0, 5, { all: S.flat("#6a462a"), top: S.flat("#8c623a") }, { shadow: 0.4 });
    S.fence3D(af.x - 40, af.y - 10, af.x - 40, fy + 10, 5, { h: 7, step: 24, col: "#7e5634", cap: "#b08052" });
    for (const [lx, ly] of [[af.x - 36, 48], [af.x - 36, 172], [af.x - 36, 296], [af.x + 80, af.y - 14], [af.x + af.w - 80, af.y - 14]]) {
      S.lamp3D(lx, ly, 0, { h: 22, post: "#4a3220", color: "#ffc268", r: 68, i: 0.72 });
    }

    // Natural Vaulted Stone Pillars around corners
    for (const [px, py] of [[af.x - 62, 6], [af.x + 115, -8], [af.x + af.w - 115, -8], [af.x + af.w + 62, 6]]) {
      S.box(px - 11, py - 10, 22, 20, -6, 98, { all: slate, top: S.flat("#4d4886") }, { shadow: 0.5 });
    }

    // ---- 3. Faceted 3D Glowing Crystal Cluster Builders (Amethyst & Cerulean) ----
    const cryTex = (c0, c1, c2) => S.tex(24, 24, (g) => {
      const gr = g.createLinearGradient(0, 0, 24, 24);
      gr.addColorStop(0, c2); gr.addColorStop(0.45, c1); gr.addColorStop(1, c0);
      g.fillStyle = gr; g.fillRect(0, 0, 24, 24);
      g.fillStyle = "rgba(255,255,255,0.6)"; g.fillRect(2, 0, 2, 24); g.fillRect(12, 4, 1, 14);
    });
    const amT = cryTex("#4a1c88", "#a84ef6", "#f2b8ff");
    const cyT = cryTex("#184898", "#48b6ff", "#d4f8ff");
    const cluster = (x0, y0, sc, palT, glowCol, rgbCol, layer) => {
      S.prism(palT, x0, y0, 10 * sc, 2 * sc, 0, 44 * sc, 5, { rot: 0.35, top: S.flat("#ffffff"), emit: 0.72, layer });
      S.prism(palT, x0 - 10 * sc, y0 + 4 * sc, 7 * sc, 1.5 * sc, 0, 28 * sc, 5, { rot: 0.75, top: S.flat("#ffffff"), emit: 0.65, layer });
      S.prism(palT, x0 + 9 * sc, y0 + 5 * sc, 7 * sc, 1.8 * sc, 0, 32 * sc, 5, { rot: 0.1, top: S.flat("#ffffff"), emit: 0.65, layer });
      S.glow(x0, y0, 20 * sc, 32 * sc, glowCol, { alpha: 0.42, anim: { type: "pulse", amp: 0.2 }, layer });
      if (!layer) S.lamp(x0, y0, 86, rgbCol, 0.52);
      S.castConeShadow(x0, y0, 12 * sc, 42 * sc, 0.48, { layer });
    };

    cluster(af.x - 68, 44, 1.25, amT, "#d272ff", [210, 115, 255]);
    cluster(af.x - 74, 134, 1.15, cyT, "#68d8ff", [105, 215, 255]);
    cluster(af.x - 68, 224, 1.2, amT, "#d272ff", [210, 115, 255]);
    cluster(af.x - 74, 314, 1.1, cyT, "#68d8ff", [105, 215, 255]);

    cluster(af.x + 56, 20, 1.15, cyT, "#68d8ff", [105, 215, 255]);
    cluster(af.x + 178, 12, 1.3, amT, "#d272ff", [210, 115, 255]);
    cluster(af.x + af.w - 178, 12, 1.3, cyT, "#68d8ff", [105, 215, 255]);
    cluster(af.x + af.w - 56, 20, 1.15, amT, "#d272ff", [210, 115, 255]);

    cluster(af.x + af.w + 46, 56, 1.2, cyT, "#68d8ff", [105, 215, 255]);
    cluster(af.x + af.w + 52, 168, 1.25, amT, "#d272ff", [210, 115, 255]);
    cluster(af.x + af.w + 46, 286, 1.15, cyT, "#68d8ff", [105, 215, 255]);

    for (const [rx, ry, rr, rh] of [[af.x - 94, 86, 16, 30], [af.x - 94, 266, 17, 32], [af.x + af.w + 80, 110, 16, 30], [af.x + af.w + 80, 240, 16, 30]]) {
      S.rock3D(rx, ry, 0, { r: rr, h: rh, pal: ["#1a1638", "#2d285c", "#545096"], shadow: 0.48 });
    }

    // ---- 4. Foreground Stalagmites & Luminous Crystals (Front Bokeh) ----
    cluster(af.x - 28, fy + 20, 1.0, amT, "#d272ff", [210, 115, 255], "front");
    cluster(cx - 120, fy + 22, 0.85, cyT, "#68d8ff", [105, 215, 255], "front");
    cluster(cx + 120, fy + 22, 0.85, amT, "#d272ff", [210, 115, 255], "front");
    cluster(af.x + af.w + 28, fy + 20, 1.0, cyT, "#68d8ff", [105, 215, 255], "front");

    const rS = U.seeded(6120);
    for (let i = 0; i < 34; i++) {
      S.glow(af.x - 40 + rS() * (af.w + 80), 20 + rS() * (ah - 40), 10 + rS() * 44, 4 + rS() * 4, i % 2 ? "#7cf0ff" : "#ea8eff", { alpha: 0.65, anim: { type: "twinkle", amp: 0.5 } });
    }
  },
});
