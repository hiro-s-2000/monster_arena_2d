// Water Field — 3D Sunny Mediterranean Resort Natatorium (3D palms, parasols, pool pavilion, open sea)
"use strict";

Scene3D.define("water", {
  walls: false,
  paint(ctx, A, rng) {
    const aw = A.aw, ah = A.ah, af = A.af;
    // Sparkling Mediterranean sandstone & white marble pool deck
    ctx.fillStyle = "#dfd8c8";
    ctx.fillRect(0, 0, aw, ah);
    for (let y = 0; y < ah; y += 10) for (let x = 0; x < aw; x += 14) {
      ctx.fillStyle = rng() < 0.4 ? "#ebe4d6" : rng() < 0.75 ? "#d2cab8" : "#c4baa6";
      ctx.fillRect(x, y, 13, 9);
    }
    // Bright white mosaic tile pool coping margin
    ctx.fillStyle = "#f4f8fc";
    ctx.fillRect(af.x - 14, af.y - 14, af.w + 28, af.h + 28);
    ctx.fillStyle = "#76bce0";
    ctx.fillRect(af.x - 5, af.y - 5, af.w + 10, af.h + 10);
  },

  build(S, A) {
    const aw = S.aw, ah = S.ah, af = S.af, fy = S.frontY, cx = aw / 2;
    S.set({
      clear: "#368ad4",
      fog: { color: "#90d2f4", near: 620, far: 1750, max: 0.48 },
      light: [-0.62, -0.3, 0.72],
      keyTint: [1.14, 1.1, 0.94],
      shadowTint: [0.34, 0.5, 0.78],
      castCol: [0.05, 0.14, 0.3],
      pitch: 37,
      dof: 1.35,
    });

    // ---- 1. Azure Resort Sea Horizon, Sailboats & Tropical Clouds Backdrop ----
    const skyT = S.tex(240, 110, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h);
      gr.addColorStop(0, "#2476c4"); gr.addColorStop(0.55, "#68bcf0"); gr.addColorStop(0.75, "#c6ecfc");
      gr.addColorStop(0.76, "#2278b8"); gr.addColorStop(1, "#3ba4dc");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
      g.fillStyle = "rgba(255,255,255,0.88)";
      g.fillRect(28, 22, 46, 7); g.fillRect(36, 17, 26, 6);
      g.fillRect(148, 26, 48, 7); g.fillRect(156, 21, 30, 6);
      // Distant white sailboats on sea band
      for (const sx of [52, 128, 198]) {
        g.fillStyle = "#ffffff"; g.beginPath(); g.moveTo(sx, 82); g.lineTo(sx + 6, 82); g.lineTo(sx + 2, 70); g.closePath(); g.fill();
        g.fillStyle = "#c84638"; g.fillRect(sx - 1, 83, 8, 2);
      }
    });
    S.backdrop(skyT, cx - 720, cx + 720, -320, -100, 220);

    // Sparkling turquoise sea outside resort peninsula (with white wooden harbor dock posts!)
    S.ground(S.tex(32, 32, (g) => {
      g.fillStyle = "#2c92cc"; g.fillRect(0, 0, 32, 32);
      g.fillStyle = "#58beee"; for (let i = 0; i < 16; i++) g.fillRect((i * 7) % 28, (i * 5) % 30, 4, 1);
      g.fillStyle = "#1d74ad"; for (let i = 0; i < 8; i++) g.fillRect((i * 11) % 28, (i * 9 + 3) % 30, 3, 1);
    }), -460, -280, aw + 460, ah + 260, { texel: 2, fog: 0.9 });
    for (let dy = 24; dy <= ah - 24; dy += 44) {
      S.box(2, dy, 5, 5, 0, 12, { all: S.flat("#8a6442"), top: S.flat("#d4aa72") }, { shadow: 0.4 });
    }

    // ---- 2. Poolside Resort Bleachers, Diving Platform & Mediterranean Poolhouse ----
    const deckWall = S.tex(48, 12, (g) => {
      g.fillStyle = "#edf4f8"; g.fillRect(0, 0, 48, 12);
      g.fillStyle = "#2b78be"; g.fillRect(0, 2, 48, 3);
      g.fillStyle = "#62b6ea"; g.fillRect(0, 8, 48, 2);
    });
    const crT = S.tex(80, 14, (g) => {
      const r = U.seeded(509), c0 = ["#ff685c", "#34a4ec", "#ffe054", "#4ad496", "#ffffff"];
      g.fillStyle = "#c6beb0"; g.fillRect(0, 9, 80, 5);
      for (let x = 2; x < 78; x += 5) {
        g.fillStyle = c0[(r() * c0.length) | 0]; g.fillRect(x, 6, 4, 6);
        g.fillStyle = "#f2c298"; g.fillRect(x + 1, 2, 2, 4);
      }
    });

    // Marble pool curb + starting blocks
    S.wall(deckWall, af.x - 10, af.y - 8, af.x + af.w + 10, af.y - 8, 0, 8, { texel: 1 });
    for (let sx = af.x + 60; sx <= af.x + af.w - 60; sx += 95) {
      S.box(sx, af.y - 12, 10, 10, 0, 10, { all: S.flat("#e8eff6"), top: S.flat("#3888d4") }, { shadow: 0.42 });
    }

    // North resort seating + Mediterranean Resort Villa/Clubhouse behind north pool!
    for (let t = 0; t < 2; t++) {
      const by = af.y - 26 - t * 14, bz = 8 + t * 8;
      S.box(af.x - 12, by, af.w + 24, 14, 0, bz, { all: S.flat("#dfd8c6"), top: S.flat("#f2ece0") });
      S.wall(crT, af.x - 6, by + 11, af.x + af.w + 6, by + 11, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.7 } });
    }
    // Mediterranean White Stucco Villa & Terracotta Roof on North Terrace
    S.house3D(af.x + 28, -2, 0, {
      w: 62, d: 34, h: 25, rh: 15, ov: 3.5, ridge: "ew",
      wall: "#f2eee6", trim: "#3888cc", roof: "#cc563c", roofHi: "#f08056", roofSh: "#842e24",
      doorSide: "c", chimney: false, shadow: 0.5,
    });
    // Championship Multi-Tier Diving Tower Center-North!
    S.box(cx - 14, af.y - 54, 28, 20, 0, 38, { all: S.flat("#e6eef6"), top: S.flat("#3688d6") }, { shadow: 0.48 });
    S.box(cx - 10, af.y - 38, 20, 16, 36, 4, { all: S.flat("#f6faff"), top: S.flat("#42a4f0") });
    S.box(cx - 8, af.y - 30, 16, 18, 20, 3, { all: S.flat("#f6faff"), top: S.flat("#42a4f0") });

    // Right Poolhouse / Cabana
    S.house3D(af.x + af.w - 92, -2, 0, {
      w: 54, d: 32, h: 23, rh: 14, ov: 3, ridge: "ns",
      wall: "#f0ece2", trim: "#2a7cc0", roof: "#2a7ec4", roofHi: "#56aee8", roofSh: "#184674",
      doorSide: "l", chimney: false, shadow: 0.5,
    });

    // West boardwalk seating + Striped 3D Octagonal Parasol Umbrellas!
    for (let t = 0; t < 2; t++) {
      const bx = af.x - 24 - t * 14, bz = 8 + t * 8;
      S.box(bx, af.y - 8, 14, af.h + 16, 0, bz, { all: S.flat("#dfd8c6"), top: S.flat("#f2ece0") });
      S.wall(crT, bx + 9, af.y - 4, bx + 9, fy + 4, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.7 } });
    }
    const umbrella = (ux, uy, c0, c1, layer) => {
      const ut = S.tex(24, 12, (g) => { for (let i = 0; i < 24; i += 3) { g.fillStyle = (i / 3) % 2 ? c1 : c0; g.fillRect(i, 0, 3, 12); } });
      S.prism(S.flat("#f0e8dc"), ux, uy, 1.3, 1, 0, 26, 5, { top: null, layer });
      S.prism(ut, ux, uy, 14, 1, 22, 9, 8, { top: null, layer });
      S.castConeShadow(ux, uy, 14, 29, 0.42, { round: true, layer });
    };
    for (const [ux, uy, c0] of [[af.x - 68, 70, "#e24844"], [af.x - 68, 175, "#368ae0"], [af.x - 68, 285, "#e8a434"], [af.x + af.w + 46, 86, "#e24844"], [af.x + af.w + 46, 260, "#368ae0"]]) {
      umbrella(ux, uy, c0, "#ffffff");
    }

    // ---- 3. 3D Curved Arched Palm Trees Casting Long Crisp Seaside Shadows ----
    for (const [px, py, ph, pr, lean, dir] of [
      [af.x - 94, 34, 62, 16, 1.4, 1], [af.x - 98, 128, 58, 15, 1.2, 1], [af.x - 94, 226, 62, 16, 1.4, 1], [af.x - 96, 330, 58, 15, 1.1, 1],
      [cx - 100, -18, 58, 15, 1.0, -1], [cx + 100, -18, 58, 15, 1.0, 1],
      [af.x + af.w + 78, 44, 60, 16, 1.3, -1], [af.x + af.w + 80, 174, 58, 15, 1.2, -1], [af.x + af.w + 78, 304, 60, 16, 1.3, -1],
    ]) {
      S.tree3D(px, py, 0, { kind: "palm", h: ph, r: pr, lean, dir, anim: { type: "sway", amp: 0.9 }, shadow: 0.46 });
    }

    // ---- 4. Foreground Pool Balustrade, Parasol & Palms (Front Bokeh) ----
    S.fence3D(af.x - 12, fy + 12, af.x + af.w + 12, fy + 12, 0, { h: 9, step: 18, col: "#e4e8ec", cap: "#ffffff", layer: "front" });
    S.tree3D(af.x - 36, fy + 26, 0, { kind: "palm", h: 56, r: 15, lean: 1.1, dir: 1, layer: "front", anim: { type: "sway", amp: 0.9 }, shadow: 0.42 });
    umbrella(af.x + af.w - 40, fy + 18, "#368ae0", "#ffffff", "front");
    S.tree3D(af.x + af.w + 36, fy + 26, 0, { kind: "palm", h: 56, r: 15, lean: 1.1, dir: -1, layer: "front", anim: { type: "sway", amp: 0.9 }, shadow: 0.42 });
  },
});
