// Stone Court — 3D Sunlit Citadel Courtyard Diorama (crenellated battlements, buttresses, aqueduct, cypresses)
"use strict";

Scene3D.define("stone", {
  walls: false,
  paint(ctx, A, rng) {
    const aw = A.aw, ah = A.ah, af = A.af;
    // Flagstone sunlit courtyard floor outside main stone battle ring
    ctx.fillStyle = "#767a84";
    ctx.fillRect(0, 0, aw, ah);
    for (let y = 0; y < ah; y += 8) for (let x = 0; x < aw; x += 12) {
      const v = rng();
      ctx.fillStyle = v < 0.3 ? "#848a94" : v < 0.7 ? "#6e747e" : "#5e6470";
      ctx.fillRect(x + ((y / 8) & 1 ? 6 : 0), y, 11, 7);
    }
    // Mossy stone joints & perimeter trim
    for (let i = 0; i < 450; i++) {
      ctx.fillStyle = rng() < 0.45 ? "#486a42" : "#525864";
      ctx.fillRect(rng() * aw | 0, rng() * ah | 0, 2, 1);
    }
    ctx.strokeStyle = "#9ca2ac";
    ctx.lineWidth = 3;
    ctx.strokeRect(af.x - 4, af.y - 4, af.w + 8, af.h + 8);
  },

  build(S, A) {
    const aw = S.aw, ah = S.ah, af = S.af, fy = S.frontY, cx = aw / 2;
    S.set({
      clear: "#3668a8",
      fog: { color: "#88b8de", near: 620, far: 1750, max: 0.48 },
      light: [-0.6, -0.32, 0.73],
      keyTint: [1.14, 1.07, 0.9],
      shadowTint: [0.35, 0.4, 0.62],
      castCol: [0.04, 0.05, 0.14],
      pitch: 37,
      dof: 1.35,
    });

    // ---- 1. Mediterranean Azure Sky & Distant Romanesque Aqueduct Ridge ----
    const skyT = S.tex(240, 110, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h);
      gr.addColorStop(0, "#2f64aa"); gr.addColorStop(0.55, "#6aa6dc"); gr.addColorStop(1, "#c8e0f4");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
      g.fillStyle = "rgba(255,250,236,0.8)";
      g.fillRect(28, 22, 42, 7); g.fillRect(34, 17, 26, 6);
      g.fillRect(156, 28, 48, 7); g.fillRect(166, 23, 28, 6);
      // Distant olive hills
      g.fillStyle = "#546c56";
      for (let x = 0; x < w; x += 2) {
        const mh = 18 + Math.sin(x * 0.03) * 8;
        g.fillRect(x, h - mh, 2, mh);
      }
    });
    S.backdrop(skyT, cx - 720, cx + 720, -320, -100, 220);

    const outTileT = S.tex(48, 48, (g) => {
      g.fillStyle = "#70747e"; g.fillRect(0, 0, 48, 48);
      g.fillStyle = "#5e626c"; for (let y = 0; y < 48; y += 8) g.fillRect(0, y, 48, 1);
      for (let y = 0; y < 48; y += 8) for (let x = (y & 8 ? 6 : 0); x < 48; x += 12) g.fillRect(x, y, 1, 8);
    });
    S.ground(outTileT, -460, -280, aw + 460, ah + 260, { texel: 1.2, fog: 0.9 });

    // ---- 2. Stone Blockwork Texture + Citadel Battlement Walls ----
    const ashlar = S.tex(32, 24, (g) => {
      g.fillStyle = "#888e9a"; g.fillRect(0, 0, 32, 24);
      g.fillStyle = "#626874"; for (let y = 0; y < 24; y += 6) g.fillRect(0, y, 32, 1);
      for (let y = 0; y < 24; y += 6) for (let x = ((y / 6) % 2) * 8; x < 32; x += 16) g.fillRect(x, y, 1, 6);
      g.fillStyle = "#aeb4c0"; g.fillRect(0, 1, 32, 1);
      // Moss streaks near bottom
      g.fillStyle = "#4c7244"; g.fillRect(4, 20, 6, 4); g.fillRect(20, 19, 5, 5);
    });
    const bannerT = S.tex(12, 22, (g) => {
      g.fillStyle = "#b83630"; g.fillRect(1, 0, 10, 18);
      g.fillStyle = "#e8c04a"; g.fillRect(5, 2, 2, 14); g.fillRect(3, 6, 6, 2);
      g.fillStyle = "#7a201c"; g.fillRect(1, 18, 4, 3); g.fillRect(7, 18, 4, 3);
    });

    // North Citadel Curtain Wall + Crenellations + Central Arched Cloister Gate
    const nwY = af.y - 36;
    S.box(af.x - 30, nwY, af.w + 60, 20, 0, 36, { all: ashlar, top: S.flat("#aab0bc") }, { texel: 1, shadow: 0.48 });
    // Crenels (merlons) along top of north wall
    for (let mx = af.x - 26; mx <= af.x + af.w + 18; mx += 22) {
      S.box(mx, nwY + 1, 12, 18, 36, 8, { all: ashlar, top: S.flat("#bec4d0") });
    }
    // Protruding Stone Buttresses & Banners along north curtain wall
    for (let bx = af.x + 12; bx <= af.x + af.w - 20; bx += 78) {
      S.box(bx, nwY + 18, 12, 10, 0, 34, { all: ashlar, top: S.flat("#aab0bc") }, { shadow: 0.5 });
      S.wall(bannerT, bx + 1, nwY + 28.5, bx + 11, nwY + 28.5, 10, 31, { anim: { type: "sway", amp: 0.4, top: false } });
    }

    // West Crenellated Battlement & Arched Arcade Buttresses casting long diagonal shadows across left stone apron!
    const wwX = af.x - 52;
    S.box(wwX, af.y - 36, 22, af.h + 52, 0, 36, { all: ashlar, top: S.flat("#aab0bc") }, { texel: 1, shadow: 0.52 });
    for (let my = af.y - 32; my <= fy + 6; my += 24) {
      S.box(wwX + 2, my, 18, 14, 36, 8, { all: ashlar, top: S.flat("#bec4d0") });
      S.box(wwX + 22, my + 2, 10, 10, 0, 32, { all: ashlar, top: S.flat("#aab0bc") }, { shadow: 0.5 });
    }
    // Torch braziers set on alternate west wall buttresses
    for (let ty = af.y + 12; ty <= fy - 12; ty += 96) {
      S.lamp3D(wwX + 27, ty + 7, 32, { h: 12, color: "#ff9a42", r: 44, i: 0.5, shadow: false });
    }

    // East Flanking Citadel Wall
    const ewX = af.x + af.w + 30;
    S.box(ewX, af.y - 36, 22, af.h + 52, 0, 36, { all: ashlar, top: S.flat("#aab0bc") }, { texel: 1, shadow: 0.46 });
    for (let my = af.y - 32; my <= fy + 6; my += 24) {
      S.box(ewX + 2, my, 18, 14, 36, 8, { all: ashlar, top: S.flat("#bec4d0") });
    }

    // ---- 3. Twin Corner Castle Watchtowers with Terracotta Conical Spires ----
    const tower = (tx, ty) => {
      S.prism(ashlar, tx, ty, 22, 21, 0, 56, 8, { top: S.flat("#9ea4b2") });
      S.prism(S.flat("#787e8a"), tx, ty, 24, 24, 54, 6, 8, { top: S.flat("#aab0bc") });
      S.prism(S.tex(24, 16, (g) => {
        g.fillStyle = "#b84632"; g.fillRect(0, 0, 24, 16);
        g.fillStyle = "#882c22"; for (let y = 3; y < 16; y += 4) g.fillRect(0, y, 24, 1);
      }), tx, ty, 25, 1, 60, 26, 8, { top: null });
      S.castConeShadow(tx, ty, 22, 80, 0.5);
    };
    tower(af.x - 42, af.y - 26);
    tower(af.x + af.w + 42, af.y - 26);

    // Rear Distant Multi-Arch Aqueduct Crossing Horizon behind courtyard
    S.box(af.x - 160, -108, af.w + 320, 18, 36, 16, { all: ashlar, top: S.flat("#b8beca") });
    for (let px = af.x - 140; px <= af.x + af.w + 140; px += 46) {
      S.box(px, -106, 14, 14, 0, 36, { all: ashlar }, { shadow: 0.45 });
    }

    // ---- 4. 3D Sculpted Castle Keep Gatehouse + Mediterranean Cypress & Olive Trees ----
    S.house3D(cx - 34, -22, 0, {
      w: 68, d: 34, h: 30, rh: 15, ov: 3, ridge: "ew",
      wall: "#b4b8c2", trim: "#6c727e", roof: "#b84a36", roofHi: "#dc7048", roofSh: "#782a24",
      masonry: true, doorSide: "c", chimney: false, shadow: 0.52,
    });

    const cyPal = ["#1c3e26", "#2f5f34", "#49843b", "#6ea84b"];
    const olPal = ["#284e2b", "#3e7038", "#64984b", "#92bf68"];
    for (const [k, tx, ty, th, tr] of [
      ["c", af.x - 84, 54, 62, 11], ["o", af.x - 94, 140, 48, 17], ["c", af.x - 84, 232, 60, 11], ["o", af.x - 92, 326, 48, 17],
      ["c", af.x + 90, -48, 60, 11], ["c", af.x + af.w - 90, -48, 60, 11],
      ["o", af.x + af.w + 84, 80, 50, 17], ["c", af.x + af.w + 78, 190, 62, 11], ["o", af.x + af.w + 84, 300, 48, 17],
    ]) {
      if (k === "c") S.tree3D(tx, ty, 0, { kind: "cypress", h: th, r: tr, pal: cyPal, anim: { type: "sway", amp: 0.5 }, shadow: 0.48 });
      else S.tree3D(tx, ty, 0, { kind: "oak", h: th, r: tr, pal: olPal, anim: { type: "sway", amp: 0.7 }, shadow: 0.48 });
    }

    // ---- 5. Foreground Low Crenellated Parapet & Torches (Front Bokeh Layer) ----
    const fgY = fy + 10;
    S.box(af.x - 28, fgY, af.w + 56, 12, 0, 10, { all: ashlar, top: S.flat("#aeb4c0") }, { layer: "front", shadow: 0.4 });
    for (let mx = af.x - 20; mx <= af.x + af.w + 10; mx += 28) {
      S.box(mx, fgY + 1, 14, 10, 10, 5, { all: ashlar, top: S.flat("#bec4d0") }, { layer: "front" });
    }
    S.tree3D(af.x - 42, fy + 24, 0, { kind: "cypress", h: 56, r: 10, pal: cyPal, layer: "front", anim: { type: "sway", amp: 0.5 }, shadow: 0.42 });
    S.tree3D(af.x + af.w + 42, fy + 24, 0, { kind: "cypress", h: 56, r: 10, pal: cyPal, layer: "front", anim: { type: "sway", amp: 0.5 }, shadow: 0.42 });
  },
});
