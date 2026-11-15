// League Stadium — 3D Cantilevered Arena Bowl & Illuminated Metropolis Night Skyline
"use strict";

Scene3D.define("stadium", {
  walls: false,
  paint(ctx, A, rng) {
    const aw = A.aw, ah = A.ah, af = A.af;
    ctx.fillStyle = "#263042";
    ctx.fillRect(0, 0, aw, ah);
    // Crisp terracotta clay running track oval outside turf
    ctx.fillStyle = "#7a362a";
    ctx.fillRect(af.x - 18, af.y - 18, af.w + 36, af.h + 36);
    ctx.fillStyle = "#924234";
    ctx.fillRect(af.x - 16, af.y - 16, af.w + 32, af.h + 32);
    // White lane stripes
    ctx.strokeStyle = "rgba(255,255,255,0.45)";
    ctx.lineWidth = 1;
    ctx.strokeRect(af.x - 12, af.y - 12, af.w + 24, af.h + 24);
    ctx.strokeRect(af.x - 6, af.y - 6, af.w + 12, af.h + 12);
    for (let i = 0; i < 650; i++) {
      ctx.fillStyle = rng() < 0.5 ? "#662c22" : "#a44f3e";
      ctx.fillRect((af.x - 15 + rng() * (af.w + 30)) | 0, (af.y - 15 + rng() * (af.h + 30)) | 0, 2, 1);
    }
  },

  build(S, A) {
    const aw = S.aw, ah = S.ah, af = S.af, fy = S.frontY, cx = aw / 2;
    S.set({
      clear: "#081226",
      fog: { color: "#122244", near: 600, far: 1700, max: 0.48 },
      light: [-0.62, -0.28, 0.72],
      keyTint: [1.08, 1.12, 1.18],
      shadowTint: [0.32, 0.38, 0.62],
      castCol: [0.02, 0.04, 0.12],
      pitch: 37,
      dof: 1.3,
    });

    // ---- 1. Evening Twilight City Sky Backdrop + Illuminated Towers (Zero Seam!) ----
    const skyT = S.tex(240, 110, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h);
      gr.addColorStop(0, "#070f24"); gr.addColorStop(0.55, "#152b54"); gr.addColorStop(1, "#2a4c84");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
      const r = U.seeded(411);
      for (let i = 0; i < 110; i++) { g.fillStyle = r() < 0.3 ? "#fff4c0" : "#9ec6ff"; g.fillRect(r() * w | 0, r() * 58 | 0, 1, 1); }
      // Distant silhouetted city towers + twinkling lit office windows
      for (let x = 0; x < w; x += 14) {
        const bw = 10 + ((x * 7) % 8), bh = 22 + ((x * 13) % 28);
        g.fillStyle = "#101e3c"; g.fillRect(x, h - bh, bw, bh);
        for (let wy = h - bh + 4; wy < h - 4; wy += 4) for (let wx = x + 2; wx < x + bw - 2; wx += 3) {
          if (r() < 0.45) { g.fillStyle = r() < 0.7 ? "#ffda78" : "#7ee0ff"; g.fillRect(wx, wy, 1, 2); }
        }
      }
    });
    S.backdrop(skyT, cx - 720, cx + 720, -320, -100, 220);

    // Exterior concourse deck
    S.ground(S.tex(32, 32, (g) => {
      g.fillStyle = "#202a3c"; g.fillRect(0, 0, 32, 32);
      g.fillStyle = "#2b384e"; g.fillRect(0, 0, 16, 16); g.fillRect(16, 16, 16, 16);
    }), -460, -280, aw + 460, ah + 260, { texel: 2, fog: 0.9 });

    // ---- 2. Illuminated LED Ribbon Wall & Structured Spectator Bays (with concrete stairs!) ----
    const adT = S.tex(192, 12, (g) => {
      const bands = [["#1c4894", "#ffe464", "STAR DRINK"], ["#b82842", "#ffffff", "VOLT GEAR"], ["#247a56", "#e6ffd0", "LEAGUE TV"], ["#662c98", "#ffe878", "KUMO AIR"], ["#d45226", "#ffffff", "NEON CUP"], ["#1e5e9a", "#9cf4ff", "GO! FIGHT!"]];
      for (let i = 0; i < 6; i++) {
        g.fillStyle = bands[i][0]; g.fillRect(i * 32, 0, 32, 12);
        g.fillStyle = bands[i][1]; Pixel.text(g, bands[i][2], i * 32 + 3, 4, 1);
      }
    });
    const crT = (s) => S.tex(80, 14, (g) => {
      const r = U.seeded(612 + s * 41), c0 = ["#e83c48", "#3a84f0", "#f4c842", "#42c872", "#e8eaf2", "#c858e8"];
      g.fillStyle = "#28344c"; g.fillRect(0, 9, 80, 5);
      for (let x = 2; x < 78; x += 5) {
        if ((x % 40) < 6) { g.fillStyle = "#5a667c"; g.fillRect(x, 6, 4, 8); continue; } // concrete aisle stair!
        g.fillStyle = c0[(r() * c0.length) | 0]; g.fillRect(x, 6, 4, 6);
        g.fillStyle = "#f2c69c"; g.fillRect(x + 1, 2, 2, 4);
      }
    });
    const conc = S.tex(24, 16, (g) => {
      g.fillStyle = "#3e4a62"; g.fillRect(0, 0, 24, 16);
      g.fillStyle = "#566480"; g.fillRect(0, 0, 24, 3);
      g.fillStyle = "#2a3448"; g.fillRect(0, 13, 24, 3);
    });
    const roofT = S.tex(32, 20, (g) => {
      g.fillStyle = "#2c384e"; g.fillRect(0, 0, 32, 20);
      g.fillStyle = "#485876"; for (let x = 0; x < 32; x += 8) g.fillRect(x, 0, 2, 20);
      g.fillStyle = "#5896e8"; g.fillRect(0, 17, 32, 2);
    });

    // LED perimeter ribbon
    S.wall(adT, af.x - 16, af.y - 15, af.x + af.w + 16, af.y - 15, 0, 11, { emit: 0.72, lit: 0, texel: 1 });
    S.wall(adT, af.x - 16, af.y - 15, af.x - 16, fy + 15, 0, 11, { emit: 0.72, lit: 0, texel: 1 });
    S.wall(adT, af.x + af.w + 16, fy + 15, af.x + af.w + 16, af.y - 15, 0, 11, { emit: 0.72, lit: 0, texel: 1 });

    const c0 = crT(0), c1 = crT(1);
    // North Tiers + Structural Piers + Dramatic Cantilevered Grandstand Roof
    for (let t = 0; t < 4; t++) {
      const by = af.y - 28 - t * 14, bz = 10 + t * 8;
      S.box(af.x - 20, by, af.w + 40, 14, 0, bz, { all: conc, top: conc }, { texel: 1 });
      S.wall(t % 2 ? c0 : c1, af.x - 14, by + 11, af.x + af.w + 14, by + 11, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.8 } });
    }
    // VIP Glass Skyboxes perched behind North stands
    S.house3D(cx - 88, af.y - 102, 42, {
      w: 176, d: 28, h: 24, rh: 10, ov: 4, ridge: "ew",
      wall: "#2c3852", trim: "#5888d8", roof: "#243046", roofHi: "#486898", roofSh: "#141c2c",
      lit: true, chimney: false, doorSide: "c", plinth: 0, shadow: false,
    });
    // Giant North & West Cantilevered Stadium Roof Canopies + Cast Shadow across Track!
    S.quad(roofT, [af.x - 26, af.y - 36, 70], [af.x + af.w + 26, af.y - 36, 70], [af.x + af.w + 26, af.y - 84, 84], [af.x - 26, af.y - 84, 84], { normal: [0, 0.3, 0.95], texel: 1 });
    for (let rx = af.x - 12; rx <= af.x + af.w + 12; rx += 76) {
      S.box(rx, af.y - 80, 5, 6, 36, 48, { all: S.flat("#52627c") });
    }
    S.castShadow(af.x - 24, af.y - 78, af.w + 48, 44, 58, 0.42);

    // West Tiers + Structural Bays + Sloped Cantilever Roof + Cast Shadow across West Apron!
    for (let t = 0; t < 5; t++) {
      const bx = af.x - 32 - t * 14, bz = 10 + t * 8;
      S.box(bx, af.y - 16, 14, af.h + 32, 0, bz, { all: conc, top: conc }, { texel: 1 });
      S.wall(t % 2 ? c1 : c0, bx + 9, af.y - 12, bx + 9, fy + 12, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.8 } });
    }
    S.quad(roofT, [af.x - 102, fy + 16, 84], [af.x - 42, fy + 16, 70], [af.x - 42, af.y - 20, 70], [af.x - 102, af.y - 20, 84], { normal: [0.3, 0, 0.95], texel: 1 });
    S.wall(S.flat("#263650"), af.x - 102, fy + 16, af.x - 42, fy + 16, 68, 84);
    for (let ry = af.y; ry <= fy; ry += 72) {
      S.box(af.x - 98, ry, 5, 5, 42, 42, { all: S.flat("#52627c") });
    }
    S.castShadow(af.x - 98, af.y - 18, 56, af.h + 34, 64, 0.44);

    // East Tiers
    for (let t = 0; t < 4; t++) {
      const bx = af.x + af.w + 18 + t * 14, bz = 10 + t * 8;
      S.box(bx, af.y - 16, 14, af.h + 32, 0, bz, { all: conc, top: conc }, { texel: 1 });
      S.wall(t % 2 ? c0 : c1, bx + 4, fy + 12, bx + 4, af.y - 12, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.8 } });
    }

    // ---- 3. Broadcast Jumbotron Screen + Towering 3D Floodlight Booms ----
    const jmb = S.tex(88, 34, (g) => {
      g.fillStyle = "#121c30"; g.fillRect(0, 0, 88, 34);
      g.fillStyle = "#3a84f0"; g.fillRect(2, 2, 84, 30);
      g.fillStyle = "#0e1a36"; g.fillRect(4, 4, 80, 26);
      g.fillStyle = "#ffe45e"; Pixel.text(g, "MONSTER LEAGUE", 11, 7, 1);
      g.fillStyle = "#7cf2ff"; Pixel.text(g, "CHAMPIONSHIP FINALS", 5, 19, 1);
    });
    S.box(cx - 46, af.y - 98, 92, 10, 74, 38, { all: S.flat("#1e283c"), front: jmb, top: S.flat("#485874") }, { emit: 0.85, shadow: 0.4 });
    S.glow(cx, af.y - 92, 92, 46, "#6cc8ff", { alpha: 0.35 });

    // Corner Stadium Floodlight Boom Towers + Crisp White Light Cones
    for (const [tx, ty] of [[af.x - 86, af.y - 74], [af.x + af.w + 68, af.y - 74]]) {
      S.box(tx, ty, 10, 10, 0, 104, { all: S.flat("#46546c"), top: S.flat("#6e7e9a") }, { shadow: 0.5 });
      S.box(tx - 10, ty + 1, 30, 8, 104, 16, { all: S.flat("#222c3c"), front: S.flat("#fffbe4"), top: S.flat("#6a7a96") }, { emit: 1 });
      S.glow(tx + 5, ty + 5, 112, 34, "#fff2c8", { alpha: 0.52 });
      S.lamp(tx + 5, ty + 20, 120, [255, 245, 210], 0.45);
    }

    // Exterior City Towers rising visibly around outer stadium rim
    const winB = S.tex(24, 48, (g) => {
      g.fillStyle = "#162442"; g.fillRect(0, 0, 24, 48);
      for (let y = 4; y < 44; y += 5) for (let x = 3; x < 20; x += 5) {
        g.fillStyle = ((x + y) % 3 === 0) ? "#ffda74" : "#6ed4ff";
        g.fillRect(x, y, 3, 2);
      }
    });
    for (const [bx, by, bw, bd, bh] of [[-120, -140, 44, 36, 130], [-60, -165, 52, 40, 165], [aw + 36, -150, 48, 38, 150], [aw + 96, -125, 42, 34, 118]]) {
      S.box(bx, by, bw, bd, 0, bh, { all: winB, top: S.flat("#2b3c5e") }, { emit: 0.35, texel: 1.5 });
    }

    // ---- 4. Foreground Rail & Stadium Bollards (Front Bokeh Frame) ----
    S.fence3D(af.x - 18, fy + 16, af.x + af.w + 18, fy + 16, 0, { h: 10, step: 20, col: "#3c4c64", cap: "#78a4e8", layer: "front" });
    for (const lx of [af.x - 4, af.x + af.w + 4]) {
      S.lamp3D(lx, fy + 16, 0, { h: 24, color: "#9ce4ff", r: 58, i: 0.55, layer: "front" });
    }
  },
});
