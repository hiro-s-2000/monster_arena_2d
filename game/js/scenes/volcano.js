// Magma Pit — 3D Stratified Caldera Gorge, Hexagonal Basalt Spires & Glowing Molten Sea
"use strict";

Scene3D.define("volcano", {
  walls: false,
  paint(ctx, A, rng) {
    const aw = A.aw, ah = A.ah, af = A.af;
    ctx.fillStyle = "#160a0e";
    ctx.fillRect(0, 0, aw, ah);
    // Glowing swirling magma lake skirting battle island left & top
    ctx.fillStyle = "#c8320e";
    ctx.fillRect(0, 0, af.x - 10, ah);
    ctx.fillRect(0, 0, aw, af.y - 10);
    for (let i = 0; i < 650; i++) {
      const mx = rng() * aw, my = rng() * ah;
      if (mx > af.x - 8 && mx < af.x + af.w + 8 && my > af.y - 8 && my < af.y + af.h + 8) continue;
      ctx.fillStyle = rng() < 0.25 ? "#fff078" : rng() < 0.6 ? "#ff8822" : "#5c140c";
      ctx.fillRect(mx | 0, my | 0, 3 + (rng() * 10 | 0), 2 + (rng() * 5 | 0));
    }
    // Raised charred basalt rim skirting the fighting slab
    ctx.fillStyle = "#342426";
    ctx.fillRect(af.x - 10, af.y - 10, af.w + 20, af.h + 20);
    ctx.strokeStyle = "#ff7828";
    ctx.lineWidth = 2;
    ctx.strokeRect(af.x - 10, af.y - 10, af.w + 20, af.h + 20);
  },

  build(S, A) {
    const aw = S.aw, ah = S.ah, af = S.af, fy = S.frontY, cx = aw / 2;
    S.set({
      clear: "#140608",
      fog: { color: "#2d1010", near: 540, far: 1650, max: 0.5 },
      light: [-0.58, -0.3, 0.75],
      keyTint: [1.18, 0.96, 0.82],
      shadowTint: [0.34, 0.24, 0.4],
      castCol: [0.02, 0.01, 0.05],
      pitch: 37,
      dof: 1.35,
    });

    // ---- 1. Boiling Lava River Abyss Surrounding Caldera Island + Ash-Lit Sky ----
    const skyT = S.tex(240, 110, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h);
      gr.addColorStop(0, "#14080e"); gr.addColorStop(0.5, "#421614"); gr.addColorStop(1, "#b84014");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
      g.fillStyle = "#240c10";
      g.beginPath(); g.moveTo(76, h); g.lineTo(106, 46); g.lineTo(132, 46); g.lineTo(164, h); g.closePath(); g.fill();
      g.fillStyle = "#ff8a28"; g.fillRect(108, 46, 22, 4);
    });
    S.backdrop(skyT, cx - 720, cx + 720, -320, -100, 220);

    const magmaSeaT = S.tex(64, 64, (g) => {
      g.fillStyle = "#b82a0e"; g.fillRect(0, 0, 64, 64);
      const r = U.seeded(401);
      for (let i = 0; i < 90; i++) {
        const vx = r() * 64 | 0, vy = r() * 64 | 0, vw = 6 + (r() * 14 | 0);
        g.fillStyle = "#ff841e"; g.fillRect(vx, vy, vw, 3);
        g.fillStyle = "#ffe868"; g.fillRect(vx + 2, vy + 1, Math.max(2, vw - 4), 1);
      }
      for (let i = 0; i < 45; i++) {
        g.fillStyle = "#481414"; g.fillRect(r() * 60 | 0, r() * 60 | 0, 5 + (r() * 7 | 0), 4 + (r() * 5 | 0));
      }
    });
    S.ground(magmaSeaT, -460, -280, aw + 460, ah + 260, { z: -10, texel: 2, emit: 0.68, lit: 0, fog: 0.7 });

    // ---- 2. Basalt Texture, Stratified North Cliffs, Bridge Arch & Magma Falls ----
    const basalt = S.tex(24, 24, (g) => {
      g.fillStyle = "#362628"; g.fillRect(0, 0, 24, 24);
      g.fillStyle = "#564042"; g.fillRect(0, 0, 24, 3);
      g.fillStyle = "#1c1216"; g.fillRect(0, 20, 24, 4); g.fillRect(11, 0, 1, 24);
      g.fillStyle = "#ff681e"; g.fillRect(4, 18, 8, 1);
    });
    const cliff = S.tex(32, 32, (g) => {
      g.fillStyle = "#2c1c20"; g.fillRect(0, 0, 32, 32);
      g.fillStyle = "#4c3438"; g.fillRect(0, 0, 32, 4); g.fillRect(0, 14, 32, 2);
      g.fillStyle = "#ff7c28"; g.fillRect(6, 18, 10, 2); g.fillRect(20, 8, 6, 2);
    });
    const lavaFlow = S.tex(24, 32, (g) => {
      const gr = g.createLinearGradient(0, 0, 24, 0);
      gr.addColorStop(0, "#d83c10"); gr.addColorStop(0.5, "#ffe858"); gr.addColorStop(1, "#d83c10");
      g.fillStyle = gr; g.fillRect(0, 0, 24, 32);
      g.fillStyle = "#fff8b8"; for (let y = 0; y < 32; y += 6) g.fillRect(9, y, 6, 3);
    });

    S.box(af.x - 10, af.y - 10, af.w + 20, 14, -14, 14, { all: basalt, top: S.flat("#4a383a") });

    // Stratified Caldera Cliff Wall directly visible north of the platform + twin glowing magma falls!
    S.box(-120, -70, aw + 240, 56, -10, 58, { front: cliff, top: S.flat("#3c2628"), left: cliff, right: cliff }, { texel: 1.5, shadow: 0.48 });
    for (const fx of [cx - 110, cx + 100]) {
      S.wall(lavaFlow, fx - 14, -13, fx + 14, -13, -10, 48, { emit: 0.95, lit: 0, anim: { type: "pulse", amp: 0.14 } });
      S.glow(fx, -10, 16, 34, "#ff8828", { alpha: 0.55 });
      S.lamp(fx, 18, 110, [255, 115, 36], 0.65);
    }
    // Natural 3D Basalt Bridge Arch framing center-north view!
    S.box(cx - 72, 2, 16, 18, -10, 46, { all: basalt }, { shadow: 0.5 });
    S.box(cx + 56, 2, 16, 18, -10, 46, { all: basalt }, { shadow: 0.5 });
    S.box(cx - 78, 0, 156, 22, 34, 12, { all: cliff, top: S.flat("#583e42") }, { shadow: 0.54 });

    // ---- 3. Basalt Columns & Obsidian Spires inside Lava Moat & Perimeter Rim ----
    const rn = U.seeded(701);
    const hexCol = (hx, hy, hr, hh, z0 = 0, layer) => {
      S.prism(basalt, hx, hy, hr, hr * 0.94, z0, hh, 6, { top: S.flat("#5c4446"), layer });
      S.glow(hx, hy, z0 + 4, hr * 2.1, "#ff5c1a", { alpha: 0.25, layer });
      if (z0 >= -2) S.castConeShadow(hx, hy, hr, hh, 0.48, { layer });
    };
    for (let y = 18; y <= ah - 18; y += 22) {
      const x0 = af.x - 24 - (y % 44 ? 12 : 0), hh = 18 + ((y * 17) % 26);
      hexCol(x0, y, 9 + (y % 3), hh);
    }
    // Stepping basalt column islands rising directly from the west lava moat!
    for (const [hx, hy, hr, hh] of [[af.x - 66, 72, 14, 34], [af.x - 84, 154, 12, 26], [af.x - 66, 238, 15, 38], [af.x - 84, 326, 13, 30]]) {
      hexCol(hx, hy, hr, hh + 10, -10);
      hexCol(hx + 14, hy + 10, hr * 0.72, hh, -10);
    }
    for (let x = af.x - 16; x <= af.x + af.w + 16; x += 26) {
      hexCol(x, af.y - 22 - (x % 52 ? 8 : 0), 9, 18 + ((x * 13) % 22));
    }
    for (let y = 20; y <= ah - 20; y += 26) {
      hexCol(af.x + af.w + 24 + (y % 52 ? 12 : 0), y, 9, 18 + ((y * 11) % 24));
    }

    for (const [lx, ly] of [[af.x - 18, af.y - 18], [cx - 72, af.y - 20], [cx + 72, af.y - 20], [af.x + af.w + 18, af.y - 18]]) {
      S.lamp3D(lx, ly, 0, { h: 24, post: "#2a181c", color: "#ff9234", r: 74, i: 0.72 });
    }

    // ---- 4. Foreground Basalt Spires & Glowing Magma Cracks (Front Bokeh) ----
    for (let x = af.x - 16; x <= af.x + af.w + 16; x += 34) {
      hexCol(x, fy + 18 + (x % 68 ? 6 : 0), 9, 14 + (x % 12), 0, "front");
    }
    S.rock3D(af.x - 36, fy + 24, 0, { r: 15, h: 28, pal: ["#201418", "#3c282a", "#684644"], layer: "front", shadow: 0.42 });
    S.rock3D(af.x + af.w + 36, fy + 24, 0, { r: 15, h: 28, pal: ["#201418", "#3c282a", "#684644"], layer: "front", shadow: 0.42 });

    for (let i = 0; i < 30; i++) {
      S.glow(af.x - 50 + rn() * (af.w + 100), 20 + rn() * (ah - 40), 10 + rn() * 45, 4 + rn() * 4, "#ff9c38", { alpha: 0.65, anim: { type: "float", amp: 5 } });
    }
  },
});
