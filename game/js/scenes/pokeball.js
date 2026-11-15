// Ball Dome — 3D High-Tech Cyberpunk Geodesic League Dome & Hologram Scoreboard
"use strict";

Scene3D.define("pokeball", {
  walls: false,
  paint(ctx, A, rng) {
    const aw = A.aw, ah = A.ah, af = A.af;
    ctx.fillStyle = "#1c2438";
    ctx.fillRect(0, 0, aw, ah);
    // Gleaming dark titanium tech apron with cyan & red racing stripes
    ctx.fillStyle = "#26324c";
    ctx.fillRect(af.x - 16, af.y - 16, af.w + 32, af.h + 32);
    ctx.strokeStyle = "#48b6ff";
    ctx.lineWidth = 2;
    ctx.strokeRect(af.x - 8, af.y - 8, af.w + 16, af.h + 16);
    ctx.strokeStyle = "#ef4656";
    ctx.strokeRect(af.x - 13, af.y - 13, af.w + 26, af.h + 26);
  },

  build(S, A) {
    const aw = S.aw, ah = S.ah, af = S.af, fy = S.frontY, cx = aw / 2;
    S.set({
      clear: "#0c152b",
      fog: { color: "#142244", near: 580, far: 1680, max: 0.48 },
      light: [-0.58, -0.3, 0.75],
      keyTint: [1.08, 1.12, 1.22],
      shadowTint: [0.32, 0.38, 0.62],
      castCol: [0.02, 0.04, 0.12],
      pitch: 37,
      dof: 1.3,
    });

    // ---- 1. Interior Steel Dome Ribs & Upper Atrium Skylight ----
    const domeBg = S.tex(240, 110, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h);
      gr.addColorStop(0, "#101b36"); gr.addColorStop(0.5, "#1e3464"); gr.addColorStop(1, "#2a4678");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
      // Skylight glass truss diamond ribs
      g.strokeStyle = "rgba(110,210,255,0.22)"; g.lineWidth = 2;
      for (let x = -40; x < w + 40; x += 28) {
        g.beginPath(); g.moveTo(x, 0); g.lineTo(x + 60, h); g.stroke();
        g.beginPath(); g.moveTo(x + 60, 0); g.lineTo(x, h); g.stroke();
      }
      g.fillStyle = "rgba(140,225,255,0.16)";
      g.fillRect(60, 12, 120, 26);
    });
    S.backdrop(domeBg, cx - 720, cx + 720, -320, -100, 220);

    S.ground(S.tex(32, 32, (g) => {
      g.fillStyle = "#182034"; g.fillRect(0, 0, 32, 32);
      g.fillStyle = "#222e48"; g.strokeRect(0, 0, 32, 32);
    }), -460, -280, aw + 460, ah + 260, { texel: 2, fog: 0.9 });

    // ---- 2. Neon Crimson & White Tech Barriers + Bleachers ----
    const barT = S.tex(96, 12, (g) => {
      g.fillStyle = "#dce4f0"; g.fillRect(0, 0, 96, 12);
      g.fillStyle = "#e03e4e"; g.fillRect(0, 0, 96, 4);
      g.fillStyle = "#2a344c"; g.fillRect(0, 5, 96, 2);
      g.fillStyle = "#4ec6ff"; g.fillRect(0, 9, 96, 2);
    });
    const metalT = S.tex(24, 16, (g) => {
      g.fillStyle = "#323e58"; g.fillRect(0, 0, 24, 16);
      g.fillStyle = "#4c5c7e"; g.fillRect(0, 0, 24, 3);
      g.fillStyle = "#20283c"; g.fillRect(0, 12, 24, 4);
    });
    const crT = S.tex(80, 14, (g) => {
      const r = U.seeded(923), c0 = ["#e64852", "#ffffff", "#42a4f0", "#f4c842", "#52d492"];
      g.fillStyle = "#242e44"; g.fillRect(0, 9, 80, 5);
      for (let x = 2; x < 78; x += 5) {
        g.fillStyle = c0[(r() * c0.length) | 0]; g.fillRect(x, 6, 4, 6);
        g.fillStyle = "#f2c69c"; g.fillRect(x + 1, 2, 2, 4);
      }
    });

    S.wall(barT, af.x - 14, af.y - 12, af.x + af.w + 14, af.y - 12, 0, 11, { texel: 1 });
    S.wall(barT, af.x - 14, af.y - 12, af.x - 14, fy + 12, 0, 11, { texel: 1 });
    S.wall(barT, af.x + af.w + 14, fy + 12, af.x + af.w + 14, af.y - 12, 0, 11, { texel: 1 });

    // North & West futuristic tiered seating bays + steel arch rib vaults
    for (let t = 0; t < 4; t++) {
      const by = af.y - 26 - t * 14, bz = 10 + t * 8;
      S.box(af.x - 18, by, af.w + 36, 14, 0, bz, { all: metalT, top: S.flat("#42506e") }, { texel: 1 });
      S.wall(crT, af.x - 10, by + 11, af.x + af.w + 10, by + 11, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.7 } });
    }
    for (let t = 0; t < 4; t++) {
      const bx = af.x - 28 - t * 14, bz = 10 + t * 8;
      S.box(bx, af.y - 12, 14, af.h + 24, 0, bz, { all: metalT, top: S.flat("#42506e") }, { texel: 1 });
      S.wall(crT, bx + 9, af.y - 8, bx + 9, fy + 8, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.7 } });
    }
    for (let t = 0; t < 3; t++) {
      const bx = af.x + af.w + 16 + t * 14, bz = 10 + t * 8;
      S.box(bx, af.y - 12, 14, af.h + 24, 0, bz, { all: metalT, top: S.flat("#42506e") }, { texel: 1 });
      S.wall(crT, bx + 4, fy + 8, bx + 4, af.y - 8, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.7 } });
    }

    // ---- 3. Parabolic Steel Geodesic Dome Arches & Glowing Neon Energy Rib Trusses ----
    for (let ax = af.x + 24; ax <= af.x + af.w - 24; ax += 126) {
      S.box(ax - 5, af.y - 78, 10, 12, 0, 74, { all: metalT, top: S.flat("#687ca4") }, { shadow: 0.45 });
      S.box(ax - 3, af.y - 70, 6, 28, 72, 12, { all: S.flat("#364666"), front: S.flat("#4cc6ff") }, { emit: 0.45 });
      S.box(ax - 3, af.y - 46, 6, 20, 62, 10, { all: S.flat("#2d3c58"), front: S.flat("#4cc6ff") }, { emit: 0.45 });
    }
    for (let ay = af.y + 18; ay <= fy - 18; ay += 92) {
      S.box(af.x - 84, ay - 5, 14, 10, 0, 72, { all: metalT, top: S.flat("#687ca4") }, { shadow: 0.48 });
      S.box(af.x - 72, ay - 3, 28, 6, 70, 10, { all: S.flat("#364666"), front: S.flat("#ef4656") }, { emit: 0.45, shadow: 0.4 });
      S.box(af.x - 48, ay - 3, 20, 6, 60, 9, { all: S.flat("#2d3c58"), front: S.flat("#ef4656") }, { emit: 0.45, shadow: 0.4 });
    }

    // ---- 4. Four Corner Cyber Poké-Reactor Core Pylons + Central Holo Scoreboard ----
    const holoT = S.tex(88, 32, (g) => {
      g.fillStyle = "#162646"; g.fillRect(0, 0, 88, 32);
      g.fillStyle = "#e03c4e"; g.fillRect(2, 2, 84, 6);
      g.fillStyle = "#48ccff"; g.fillRect(2, 24, 84, 6);
      g.fillStyle = "#ffffff"; Pixel.text(g, "BALL DOME LEAGUE", 9, 13, 1);
    });
    S.box(cx - 46, af.y - 88, 92, 10, 52, 34, { all: S.flat("#1c2842"), front: holoT, top: S.flat("#5c7098") }, { emit: 0.9, shadow: 0.45 });

    for (const [px, py] of [[af.x - 42, af.y - 42], [af.x + af.w + 42, af.y - 42], [af.x - 42, fy + 26], [af.x + af.w + 42, fy + 26]]) {
      const lay = py > fy ? "front" : undefined;
      S.prism(S.flat("#2a3652"), px, py, 10, 8, 0, 44, 6, { top: S.flat("#58688c"), layer: lay });
      S.prism(S.flat("#ef4656"), px, py, 9, 9, 32, 6, 6, { emit: 0.8, lit: 0, layer: lay });
      S.prism(S.flat("#4ec6ff"), px, py, 6, 2, 44, 14, 6, { emit: 0.9, lit: 0, layer: lay });
      S.glow(px, py, 50, 26, "#4ec6ff", { alpha: 0.46, layer: lay });
      if (!lay) S.lamp(px, py, 88, [120, 210, 255], 0.52);
      S.castConeShadow(px, py, 10, 48, 0.4, { layer: lay });
    }

    // ---- 5. Foreground Futuristic Railing (Front Bokeh Frame) ----
    S.fence3D(af.x - 16, fy + 16, af.x + af.w + 16, fy + 16, 0, { h: 10, step: 22, col: "#3c4c68", cap: "#ef4656", layer: "front" });
  },
});
