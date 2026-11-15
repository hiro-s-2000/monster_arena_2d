// Night Meadow — 3D Village-Edge HD-2D Diorama (conical pines, 3D cottages, flower-bed fence, moonlight shadows)
"use strict";

Scene3D.define("meadow", {
  walls: false,
  paint(ctx, A, rng) {
    const aw = A.aw, ah = A.ah, af = A.af;
    ctx.fillStyle = "#224634";
    ctx.fillRect(0, 0, aw, ah);
    // Crisp moonlit glade patches vs deeper forest turf
    for (let i = 0; i < 200; i++) {
      const gx = rng() * aw, gy = rng() * ah;
      ctx.fillStyle = rng() < 0.48 ? "#2e6040" : "#1a382a";
      ctx.fillRect(gx | 0, gy | 0, 6 + (rng() * 14 | 0), 4 + (rng() * 10 | 0));
    }
    for (let i = 0; i < 2200; i++) {
      ctx.fillStyle = rng() < 0.5 ? "#3b754b" : rng() < 0.72 ? "#255038" : "#152d22";
      ctx.fillRect(rng() * aw | 0, rng() * ah | 0, 1 + (rng() < 0.35 ? 1 : 0), 1);
    }
    // Winding warm cobblestone/gravel village path skirting north & west side of the field
    const path = (px, py, pw, ph) => {
      ctx.fillStyle = "#485258";
      ctx.fillRect(px | 0, py | 0, pw | 0, ph | 0);
      for (let k = 0; k < (pw * ph * 0.18); k++) {
        ctx.fillStyle = rng() < 0.45 ? "#636d72" : rng() < 0.7 ? "#3a4248" : "#7b8482";
        ctx.fillRect((px + rng() * (pw - 2)) | 0, (py + rng() * (ph - 2)) | 0, 2, 1);
      }
    };
    path(af.x - 24, af.y - 22, af.w + 48, 16);
    path(af.x - 24, af.y - 10, 16, af.h + 18);
    // Cobble aprons in front of the cottages
    path(af.x + 28, 42, 62, 22);
    path(af.x + 245, 42, 62, 22);
    // Winding creek along NE corner
    ctx.fillStyle = "#173c5e";
    ctx.fillRect(af.x + 350, 24, 96, 36);
    for (let i = 0; i < 90; i++) {
      ctx.fillStyle = rng() < 0.5 ? "#3b7ca8" : "#70b8dc";
      ctx.fillRect((af.x + 352 + rng() * 90) | 0, (26 + rng() * 32) | 0, 3 + (rng() * 5 | 0), 1);
    }
    // Wildflower speckles around perimeter & foreground flowerbed zones
    const fl = ["#e65862", "#f6d464", "#8cd4ff", "#ffffff"];
    for (let i = 0; i < 260; i++) {
      const fx = rng() * aw, fy = rng() * ah;
      if (fx > af.x - 2 && fx < af.x + af.w + 2 && fy > af.y - 2 && fy < af.y + af.h + 2) continue;
      ctx.fillStyle = fl[i % fl.length];
      ctx.fillRect(fx | 0, fy | 0, 1, 1);
    }
    // Stone border edging around battle clearing
    ctx.strokeStyle = "#3a4c42";
    ctx.lineWidth = 3;
    ctx.strokeRect(af.x - 2, af.y - 2, af.w + 4, af.h + 4);
  },

  build(S, A) {
    const aw = S.aw, ah = S.ah, af = S.af, fy = S.frontY, cx = aw / 2;
    S.set({
      clear: "#0e1d38",
      fog: { color: "#193054", near: 540, far: 1650, max: 0.48 },
      light: [-0.62, -0.3, 0.72],
      keyTint: [1.08, 1.16, 1.24],
      shadowTint: [0.3, 0.36, 0.56],
      castCol: [0.02, 0.04, 0.12],
      pitch: 36,
      dof: 1.35,
    });

    // ---- 1. Moonlit Sky Backdrop & Layered Forest Horizon ----
    const skyT = S.tex(240, 110, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h);
      gr.addColorStop(0, "#091326"); gr.addColorStop(0.55, "#152b4c"); gr.addColorStop(1, "#26476a");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
      const r = U.seeded(771);
      for (let i = 0; i < 120; i++) {
        g.fillStyle = r() < 0.25 ? "#fff4c8" : "#a8d2ff";
        g.fillRect(r() * w | 0, r() * (h * 0.72) | 0, 1, 1);
      }
      // Silver crescent moon top-left
      g.fillStyle = "#eef6ff"; g.beginPath(); g.arc(46, 26, 9, 0, 6.28); g.fill();
      g.fillStyle = "#0e1e38"; g.beginPath(); g.arc(50, 23, 8, 0, 6.28); g.fill();
      // Distant wooded hills
      g.fillStyle = "#162d42";
      for (let x = 0; x < w; x += 2) {
        const mh = 18 + Math.sin(x * 0.04) * 9 + Math.sin(x * 0.11) * 5;
        g.fillRect(x, h - mh, 2, mh);
      }
      g.fillStyle = "#102334";
      for (let x = 0; x < w; x += 2) {
        const mh = 10 + Math.cos(x * 0.06) * 5;
        g.fillRect(x, h - mh, 2, mh);
      }
    });
    S.backdrop(skyT, cx - 720, cx + 720, -320, -100, 220);

    // Subtle extended grass horizon behind the village (texel=2, matched palette so zero seam!)
    const outerGrass = S.tex(64, 64, (g) => {
      g.fillStyle = "#224634"; g.fillRect(0, 0, 64, 64);
      const r = U.seeded(55);
      for (let i = 0; i < 420; i++) {
        g.fillStyle = r() < 0.5 ? "#2c5a3d" : "#1a382a";
        g.fillRect(r() * 64 | 0, r() * 64 | 0, 2, 1);
      }
    });
    S.ground(outerGrass, -460, -280, aw + 460, ah + 260, { texel: 2, fog: 0.9 });

    // ---- 2. North Terrace Cliffs & Pine Canopy Ridge (depth skyline) ----
    const cliffT = S.tex(32, 28, (g) => {
      g.fillStyle = "#2b3848"; g.fillRect(0, 0, 32, 28);
      g.fillStyle = "#486844"; g.fillRect(0, 0, 32, 5);
      g.fillStyle = "#628e4c"; g.fillRect(0, 0, 32, 2);
      g.fillStyle = "#1c2634"; for (let y = 8; y < 28; y += 6) g.fillRect(0, y, 32, 2);
    });
    S.box(-120, -145, aw + 240, 56, 0, 34, { front: cliffT, top: S.flat("#234430"), left: cliffT, right: cliffT }, { texel: 2 });

    const moonPine = ["#18482e", "#2b6e3e", "#4da84c", "#82df66"];
    const goldOak = ["#1d4e30", "#337c3b", "#59b24a", "#94e068"];

    // ---- 3. 3D Village Cottages on North Side (inspired by reference Pallet diorama) ----
    // Main terracotta/red-roofed cottage left-of-center (brought forward cleanly below top HUD bar!)
    S.house3D(af.x + 42, 18, 0, {
      w: 58, d: 34, h: 25, rh: 15, ov: 3.5, ridge: "ew",
      wall: "#e0e6ee", trim: "#586884", roof: "#c84836", roofHi: "#f07250", roofSh: "#6c2422",
      lit: true, chimney: true, doorSide: "l", shadow: 0.54,
    });
    // Mailbox + flower patch beside main cottage
    S.box(af.x + 28, 44, 5, 5, 0, 8, { all: S.flat("#526074"), top: S.flat("#9eb0c8") }, { shadow: 0.42 });
    S.box(af.x + 27, 43, 7, 7, 8, 4, { all: S.flat("#b8c8dc"), top: S.flat("#dce8f6") });

    // Blue-roofed 3D cabin right-of-center (north-south ridge like neighboring Pallet house!)
    S.house3D(af.x + 244, 18, 0, {
      w: 50, d: 34, h: 24, rh: 15, ov: 3, ridge: "ns",
      wall: "#ded6c4", trim: "#6e5c48", roof: "#3264aa", roofHi: "#5696e4", roofSh: "#1a3864",
      lit: true, chimney: true, doorSide: "r", shadow: 0.54,
    });

    // Small guardhouse / windmill tower in upper-right near creek bridge
    S.house3D(af.x + 478, 18, 0, {
      w: 38, d: 28, h: 28, rh: 16, ov: 2.5, ridge: "ns",
      wall: "#d4dae4", trim: "#485468", roof: "#3a7e64", roofHi: "#62be8e", roofSh: "#1c4034",
      lit: true, doorSide: "c", shadow: 0.5,
    });

    // Arched 3D wooden footbridge across the NE creek
    const brX = af.x + 354;
    S.box(brX, 34, 68, 18, 0, 5, { all: S.flat("#6c482a"), top: S.flat("#8c623a") }, { shadow: 0.4 });
    S.fence3D(brX, 34, brX + 68, 34, 5, { h: 7, step: 17 });
    S.fence3D(brX, 52, brX + 68, 52, 5, { h: 7, step: 17 });
    S.glow(brX + 34, 42, 2, 34, "#4aa4e8", { alpha: 0.28, anim: { type: "pulse", amp: 0.15 } });

    // Giant Glowing Spirit Oak perched centrally in upper woods
    S.tree3D(af.x + 172, 2, 0, { kind: "oak", h: 76, r: 28, pal: goldOak, blossom: "#b4ffb0", anim: { type: "sway", amp: 0.9 }, shadow: 0.48 });
    S.glow(af.x + 172, 6, 36, 36, "#7effa8", { alpha: 0.36, anim: { type: "pulse", amp: 0.2 } });
    S.lamp(af.x + 172, 26, 84, [120, 240, 175], 0.6);

    // Back row of tiered conical pines & broadleaf trees behind houses (upper bokeh horizon)
    const backTrees = [
      ["p", 40, -18, 58, 18], ["o", 96, -24, 54, 19], ["p", 148, -20, 56, 17], ["p", 214, -14, 54, 16],
      ["o", 316, -22, 56, 19], ["p", 386, -18, 58, 17], ["o", 456, -14, 54, 18], ["p", 550, -22, 56, 17],
      ["p", 630, -16, 54, 16], ["o", 700, -18, 52, 18], ["p", 760, -12, 54, 16],
    ];
    for (const [k, tx, ty, th, tr] of backTrees) {
      if (k === "p") S.pine3D(tx, ty, 0, { h: th, r: tr, tiers: 5, pal: moonPine, anim: { type: "sway", amp: 0.7 }, shadow: 0.45 });
      else S.tree3D(tx, ty, 0, { kind: "oak", h: th, r: tr, pal: goldOak, anim: { type: "sway", amp: 0.8 }, shadow: 0.45 });
    }
    // Second background ridge atop cliff
    for (let tx = 20; tx <= aw - 20; tx += 76) {
      S.pine3D(tx, -104, 34, { h: 46 + ((tx * 13) % 14), r: 15, tiers: 4, pal: moonPine, shadow: false });
    }

    // ---- 4. Dense Shoulder-to-Shoulder Left Forest (like reference Pallet forest border!) ----
    const leftPines = [
      [18, 22, 58, 18], [58, 36, 56, 18], [22, 68, 60, 19], [66, 84, 54, 17],
      [18, 114, 60, 19], [62, 134, 56, 18], [20, 164, 62, 19], [64, 184, 56, 18],
      [18, 214, 60, 19], [62, 234, 56, 18], [20, 264, 62, 19], [64, 286, 56, 18],
      [18, 314, 60, 19], [62, 336, 56, 18], [20, 364, 60, 19], [66, 392, 56, 18],
    ];
    for (const [px, py, ph, pr] of leftPines) {
      S.pine3D(px, py, 0, { h: ph, r: pr, tiers: 5, pal: moonPine, anim: { type: "sway", amp: 0.8 }, shadow: 0.54 });
    }
    // Right forest edge pines & oaks
    for (let py = 30; py <= ah - 30; py += 66) {
      const rx = aw - 44 + ((py * 7) % 18) - 9;
      if ((py & 1) === 0) S.pine3D(rx, py, 0, { h: 46, r: 14, tiers: 4, pal: moonPine, anim: { type: "sway", amp: 0.8 }, shadow: 0.46 });
      else S.tree3D(rx, py, 0, { kind: "oak", h: 46, r: 16, pal: goldOak, anim: { type: "sway", amp: 0.9 }, shadow: 0.46 });
    }

    // ---- 5. North Wooden Picket Fence + Glowing Village Lanterns ----
    S.fence3D(af.x - 4, af.y - 12, af.x + 150, af.y - 12, 0, { h: 9, step: 16, col: "#966b3e", cap: "#caa066" });
    S.fence3D(af.x + 210, af.y - 12, af.x + af.w + 4, af.y - 12, 0, { h: 9, step: 16, col: "#966b3e", cap: "#caa066" });

    for (const [lx, ly] of [[af.x - 10, af.y - 8], [af.x + 156, af.y - 10], [af.x + af.w - 140, af.y - 10], [af.x + af.w + 10, af.y - 8], [af.x - 10, fy - 4]]) {
      S.lamp3D(lx, ly, 0, { h: 24, color: "#ffd072", r: 62, i: 0.72 });
    }

    // Mossy 3D boulders around clearing edges
    for (const [rx, ry, rr, rh] of [[af.x - 18, af.y + 84, 9, 9], [af.x + 190, af.y - 16, 10, 10], [af.x + af.w + 18, af.y + 110, 9, 9]]) {
      S.rock3D(rx, ry, 0, { r: rr, h: rh, pal: ["#2c3a44", "#465c62", "#72947a"], shadow: 0.45 });
    }

    // ---- 6. Foreground Picket Fence & Raised Flower Bed (Front Bokeh Frame!) ----
    const fgY = fy + 2;
    // Raised timber flowerbed centered in lower foreground (like Pallet reference!)
    const bedX = af.x + 110, bedW = af.w - 220, bedY = fgY + 4;
    S.box(bedX, bedY, bedW, 24, 0, 5, { all: S.flat("#6a462c"), top: S.flat("#423020") }, { layer: "front", shadow: 0.36 });
    const flRowT = S.tex(36, 12, (g) => {
      g.fillStyle = "#2b6c34"; for (let x = 0; x < 36; x += 6) g.fillRect(x + 1, 5, 4, 7);
      const cols = ["#f85864", "#ffde58", "#fa78a2", "#fff2aa"];
      for (let i = 0; i < 6; i++) {
        g.fillStyle = cols[i % cols.length]; g.fillRect(i * 6 + 1, 1, 4, 4);
        g.fillStyle = "#fffbe4"; g.fillRect(i * 6 + 2, 2, 2, 2);
      }
    });
    for (let rOff = 4; rOff <= 18; rOff += 7) {
      S.wall(flRowT, bedX + 4, bedY + rOff, bedX + bedW - 4, bedY + rOff, 5, 13, { texel: 0.85, layer: "front", lit: 0.4, anim: { type: "sway", amp: 0.6, top: true } });
    }
    S.lamp(bedX + bedW * 0.5, bedY + 10, 140, [255, 220, 155], 0.42);
    S.fence3D(bedX - 8, bedY + 1, bedX + bedW - 34, bedY + 1, 0, { h: 10, step: 14, col: "#aa7a48", cap: "#deb274", layer: "front" });
    // Wooden town sign at right end of foreground fence
    const sgT = S.tex(16, 11, (g) => {
      g.fillStyle = "#dce2e8"; g.fillRect(0, 0, 16, 11);
      g.fillStyle = "#6c7684"; g.strokeRect(0, 0, 15, 10);
      g.fillStyle = "#c8523c"; g.fillRect(3, 3, 10, 2);
      g.fillStyle = "#485260"; g.fillRect(3, 6, 8, 1); g.fillRect(3, 8, 6, 1);
    });
    S.box(bedX + bedW - 26, bedY - 1, 16, 4, 5, 11, { all: S.flat("#765232"), front: sgT }, { layer: "front", shadow: 0.35 });
    // Flanking foreground pines & lanterns for lush foreground bokeh
    S.pine3D(af.x + 26, fy + 22, 0, { h: 54, r: 17, tiers: 5, pal: moonPine, layer: "front", anim: { type: "sway", amp: 0.9 }, shadow: 0.48 });
    S.pine3D(af.x + af.w - 26, fy + 22, 0, { h: 54, r: 17, tiers: 5, pal: moonPine, layer: "front", anim: { type: "sway", amp: 0.9 }, shadow: 0.48 });
    S.lamp3D(af.x + af.w - 76, fy + 12, 0, { h: 24, color: "#ffd072", r: 62, i: 0.65, layer: "front" });

    // ---- 7. Warm Fireflies & Night Mote Depth (Atmospheric rays rendered continuously in post.js) ----
    const rF = U.seeded(4302);
    for (let i = 0; i < 36; i++) {
      const fx = af.x - 30 + rF() * (af.w + 60), fz = 12 + rF() * 44, fyPos = 20 + rF() * (ah - 40);
      S.glow(fx, fyPos, fz, 5 + rF() * 5, i % 3 ? "#b8ff78" : "#ffe880", { alpha: 0.6, anim: { type: "float", amp: 4 } });
    }
  },
});
