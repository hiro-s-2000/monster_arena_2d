// Cosmic Stage — 3D Shattered Astral Temple Ruins, Orbital Astrolabe & Galactic Nebula Sea
"use strict";

Scene3D.define("cosmic", {
  walls: false,
  paint(ctx, A, rng) {
    const aw = A.aw, ah = A.ah, af = A.af;
    ctx.clearRect(0, 0, aw, ah);
    // Ancient star-marble platform slab with glowing astral runes trimming arena perimeter
    ctx.fillStyle = "#261c50";
    ctx.fillRect(af.x - 14, af.y - 14, af.w + 28, af.h + 28);
    ctx.fillStyle = "#3a2e72";
    ctx.fillRect(af.x - 10, af.y - 10, af.w + 20, af.h + 20);
    ctx.strokeStyle = "#7ae6ff";
    ctx.lineWidth = 2;
    ctx.strokeRect(af.x - 6, af.y - 6, af.w + 12, af.h + 12);
    for (const [gx, gy] of [[af.x + 8, af.y + 8], [af.x + af.w - 8, af.y + 8], [af.x + 8, af.y + af.h - 8], [af.x + af.w - 8, af.y + af.h - 8]]) {
      ctx.fillStyle = "#ee7eff"; ctx.fillRect(gx - 3, gy - 3, 6, 6);
    }
  },

  build(S, A) {
    const aw = S.aw, ah = S.ah, af = S.af, fy = S.frontY, cx = aw / 2;
    S.set({
      clear: "#050618",
      fog: { color: "#0b0e2c", near: 620, far: 1850, max: 0.48 },
      light: [-0.6, -0.3, 0.74],
      keyTint: [1.14, 1.06, 1.28],
      shadowTint: [0.28, 0.24, 0.54],
      castCol: [0.02, 0.02, 0.09],
      clip: { x: af.x - 14, y: af.y - 14, w: af.w + 28, h: af.h + 28, r: 16 },
      pitch: 36,
      dof: 1.35,
    });

    // ---- 1. Radiant Deep Galactic Nebula Plane & Ringed Planet Vista ----
    const cosBg = S.tex(260, 120, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h);
      gr.addColorStop(0, "#050618"); gr.addColorStop(0.5, "#1a144a"); gr.addColorStop(1, "#2f226c");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
      const nb = g.createRadialGradient(130, 56, 8, 130, 56, 115);
      nb.addColorStop(0, "rgba(195,110,255,0.48)"); nb.addColorStop(0.5, "rgba(68,185,255,0.26)"); nb.addColorStop(1, "rgba(10,12,40,0)");
      g.fillStyle = nb; g.fillRect(0, 0, w, h);
      const r = U.seeded(9701);
      for (let i = 0; i < 200; i++) {
        g.fillStyle = r() < 0.25 ? "#ffd88a" : r() < 0.6 ? "#9cf2ff" : "#ffffff";
        g.fillRect(r() * w | 0, r() * h | 0, 1 + (r() < 0.16 ? 1 : 0), 1);
      }
      g.save();
      const px = 64, py = 66;
      const pg = g.createLinearGradient(px - 22, py - 22, px + 22, py + 22);
      pg.addColorStop(0, "#86e4ff"); pg.addColorStop(0.55, "#425cb8"); pg.addColorStop(1, "#181846");
      g.fillStyle = pg; g.beginPath(); g.arc(px, py, 20, 0, 6.28); g.fill();
      g.strokeStyle = "#e0beff"; g.lineWidth = 3;
      g.beginPath(); g.ellipse(px, py, 38, 8, -0.28, 0, 6.28); g.stroke();
      g.restore();
    });
    S.backdrop(cosBg, cx - 760, cx + 760, -320, -100, 220);

    // Rich cosmic galaxy plane visible in abyss all around floating temple slab!
    const starVoid = S.tex(160, 120, (g, w, h) => {
      g.fillStyle = "#080922"; g.fillRect(0, 0, w, h);
      const n1 = g.createRadialGradient(48, 52, 4, 48, 52, 54);
      n1.addColorStop(0, "rgba(160,80,255,0.42)"); n1.addColorStop(0.6, "rgba(48,140,255,0.22)"); n1.addColorStop(1, "rgba(8,9,34,0)");
      g.fillStyle = n1; g.fillRect(0, 0, w, h);
      const n2 = g.createRadialGradient(120, 74, 4, 120, 74, 46);
      n2.addColorStop(0, "rgba(90,220,255,0.34)"); n2.addColorStop(1, "rgba(8,9,34,0)");
      g.fillStyle = n2; g.fillRect(0, 0, w, h);
      const r = U.seeded(4221);
      for (let i = 0; i < 160; i++) {
        g.fillStyle = r() < 0.2 ? "#ffdf88" : r() < 0.6 ? "#9cf0ff" : "#ffffff";
        g.fillRect(r() * w | 0, r() * h | 0, 1 + (r() < 0.14 ? 1 : 0), 1);
      }
    });
    S.floor(starVoid, cx - 640, -280, cx + 640, fy + 380, -42, { stretch: true, emit: 0.85, lit: 0, shade: false, fog: 0.4 });

    // ---- 2. Floating Platform Cliff Underside & Star-Marble Rim ----
    const rimT = S.tex(24, 24, (g) => {
      g.fillStyle = "#2b2256"; g.fillRect(0, 0, 24, 24);
      g.fillStyle = "#483a88"; g.fillRect(0, 0, 24, 4);
      g.fillStyle = "#62d4ff"; g.fillRect(0, 4, 24, 2);
      g.fillStyle = "#161032"; g.fillRect(0, 14, 24, 10);
    });
    S.box(af.x - 14, af.y - 14, af.w + 28, af.h + 28, -28, 28, { front: rimT, left: rimT, right: rimT, back: rimT, top: null }, { texel: 1.5 });

    // ---- 3. Ancient Fluted Ruin Temple Columns & Floating Astrolabe Monoliths ----
    const colT = S.tex(18, 32, (g) => {
      g.fillStyle = "#52468c"; g.fillRect(0, 0, 18, 32);
      g.fillStyle = "#7e70c4"; for (let x = 2; x < 18; x += 4) g.fillRect(x, 0, 2, 32);
      g.fillStyle = "#88f0ff"; g.fillRect(0, 4, 18, 2); g.fillRect(0, 26, 18, 2);
    });
    const runeCol = (px, py, z0, hh, broken, layer) => {
      S.box(px - 7, py - 7, 14, 14, z0, 5, { all: S.flat("#342c5c"), top: S.flat("#7668b4") }, { layer });
      S.prism(colT, px, py, 5, 4.5, z0 + 5, hh, 8, { top: S.flat("#9486d6"), anim: z0 > 2 ? { type: "float", amp: 2.2 } : undefined, layer });
      if (!broken) {
        S.box(px - 8, py - 8, 16, 16, z0 + 5 + hh, 4, { all: S.flat("#5c52a0"), front: S.flat("#88f2ff"), top: S.flat("#9486d6") }, { emit: 0.45, layer });
        S.lamp3D(px, py, z0 + 9 + hh, { h: 12, post: "#483a84", color: "#88f4ff", r: 74, i: 0.55, lamp: !layer, shadow: false, layer });
      }
      if (z0 === 0 && !layer) S.castConeShadow(px, py, 7, hh + 8, 0.42);
    };

    for (const [px, py, hh, brk] of [
      [af.x - 10, af.y - 10, 42, false], [af.x + 120, af.y - 12, 34, true], [af.x + af.w - 120, af.y - 12, 34, true], [af.x + af.w + 10, af.y - 10, 42, false],
      [af.x - 12, af.y + 110, 32, true], [af.x - 12, af.y + 230, 38, false], [af.x + af.w + 12, af.y + 110, 38, false], [af.x + af.w + 12, af.y + 230, 32, true],
    ]) {
      runeCol(px, py, 0, hh, brk);
    }

    const floatIsle = (ix, iy, iz, w, d, obH) => {
      S.box(ix - w * 0.5, iy - d * 0.5, w, d, iz - 14, 14, { front: rimT, left: rimT, right: rimT, top: S.flat("#4a3d8c") }, { anim: { type: "float", amp: 2.5 } });
      S.prism(S.flat("#2a2058"), ix, iy, w * 0.46, 2, iz - 32, 18, 6, { top: null, anim: { type: "float", amp: 2.5 } });
      if (obH > 0) {
        S.prism(S.tex(16, 32, (g) => {
          g.fillStyle = "#483c86"; g.fillRect(0, 0, 16, 32);
          g.fillStyle = "#88f0ff"; g.fillRect(6, 6, 4, 20);
        }), ix, iy, 6.5, 1.5, iz, obH, 4, { rot: 0.785, top: null, emit: 0.55, anim: { type: "float", amp: 2.5 } });
        S.glow(ix, iy, iz + obH * 0.55, 24, "#94e8ff", { alpha: 0.36, anim: { type: "float", amp: 2.5 } });
      }
    };
    floatIsle(af.x - 76, 54, 18, 46, 36, 48);
    floatIsle(af.x - 84, 210, 12, 42, 34, 42);
    floatIsle(af.x - 68, 342, 22, 38, 30, 36);
    floatIsle(cx - 110, -28, 22, 52, 38, 56);
    floatIsle(cx + 110, -28, 22, 52, 38, 56);
    floatIsle(af.x + af.w + 76, 70, 18, 46, 36, 48);
    floatIsle(af.x + af.w + 80, 240, 14, 42, 34, 44);

    // Center North Celestial Astrolabe Ring Arch
    S.prism(S.flat("#f2c868"), cx, -12, 48, 48, 30, 6, 12, { top: null, emit: 0.55, anim: { type: "pulse", amp: 0.15 } });
    S.prism(S.flat("#84f0ff"), cx, -12, 22, 2, 12, 42, 6, { top: S.flat("#ffffff"), emit: 0.8, anim: { type: "float", amp: 3 } });
    S.glow(cx, -12, 34, 48, "#a67eff", { alpha: 0.42 });
    S.lamp(cx, 14, 130, [155, 130, 255], 0.62);

    // ---- 4. Foreground Broken Marble Ruins & Hovering Crystals (Front Bokeh) ----
    runeCol(af.x - 10, fy + 16, 0, 22, true, "front");
    runeCol(af.x + af.w + 10, fy + 16, 0, 22, true, "front");
    S.rock3D(af.x + 80, fy + 22, 0, { r: 10, h: 14, pal: ["#201846", "#3a2d74", "#7264b8"], layer: "front", shadow: 0.35 });
    S.rock3D(af.x + af.w - 80, fy + 22, 0, { r: 10, h: 14, pal: ["#201846", "#3a2d74", "#7264b8"], layer: "front", shadow: 0.35 });

    const rF = U.seeded(9104);
    for (let i = 0; i < 42; i++) {
      S.glow(af.x - 80 + rF() * (af.w + 160), -40 + rF() * (ah + 60), 6 + rF() * 68, 4 + rF() * 5, i % 3 ? "#84f2ff" : "#f4b0ff", { alpha: 0.6, anim: { type: "twinkle", amp: 0.5 } });
    }
  },
});
