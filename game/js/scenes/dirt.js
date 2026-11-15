// Clay Pitch — Sunlit 3D Fairground Ballpark Diorama (covered dugouts, jumbotron, farmhouses, oaks & long shadows)
"use strict";

Scene3D.define("dirt", {
  walls: true,
  paint(ctx, A, rng) {
    const aw = A.aw, ah = A.ah, af = A.af;
    // Lush mowed country grass outside stands + packed clay warning track around af
    ctx.fillStyle = "#3f7e34";
    ctx.fillRect(0, 0, aw, ah);
    for (let i = 0; i * 22 < aw; i++) {
      if (i % 2) { ctx.fillStyle = "#488e3b"; ctx.fillRect(i * 22, 0, 22, ah); }
    }
    for (let i = 0; i < 2200; i++) {
      ctx.fillStyle = rng() < 0.5 ? "#336828" : "#5aa448";
      ctx.fillRect(rng() * aw | 0, rng() * ah | 0, 1 + (rng() < 0.3 ? 1 : 0), 1);
    }
    // Clay warning track apron + chalk perimeter
    ctx.fillStyle = "#94623a";
    ctx.fillRect(af.x - 16, af.y - 16, af.w + 32, af.h + 32);
    for (let i = 0; i < 900; i++) {
      ctx.fillStyle = rng() < 0.5 ? "#7a4e2c" : "#ad784a";
      ctx.fillRect((af.x - 15 + rng() * (af.w + 30)) | 0, (af.y - 15 + rng() * (af.h + 30)) | 0, 2, 1);
    }
    // Timber bench dirt aprons on far left/top outside bleachers
    ctx.fillStyle = "#8a7054";
    ctx.fillRect(0, 0, af.x - 86, ah);
    ctx.fillRect(0, 0, aw, 54);
    for (let i = 0; i < 650; i++) {
      ctx.fillStyle = rng() < 0.5 ? "#725b42" : "#a08666";
      ctx.fillRect(rng() * (af.x - 88) | 0, rng() * ah | 0, 2, 1);
    }
  },

  build(S, A) {
    const aw = S.aw, ah = S.ah, af = S.af, fy = S.frontY, cx = aw / 2;
    S.set({
      clear: "#4e92d8",
      fog: { color: "#90c2e6", near: 620, far: 1750, max: 0.48 },
      light: [-0.64, -0.28, 0.72],
      keyTint: [1.14, 1.07, 0.88],
      shadowTint: [0.36, 0.44, 0.66],
      castCol: [0.04, 0.06, 0.16],
      pitch: 37,
      dof: 1.35,
    });

    // ---- 1. Country Fair Sky & Rolling Sunlit Green Hills Backdrop ----
    const skyT = S.tex(240, 110, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h);
      gr.addColorStop(0, "#3a84cc"); gr.addColorStop(0.55, "#72b6ea"); gr.addColorStop(1, "#c8e6f8");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
      // Warm cumulus clouds
      g.fillStyle = "rgba(255,252,240,0.85)";
      for (const [cx0, cy0, cw] of [[34, 24, 36], [116, 18, 48], [192, 28, 34]]) {
        g.fillRect(cx0, cy0, cw, 7); g.fillRect(cx0 + 5, cy0 - 5, cw - 10, 6); g.fillRect(cx0 + 12, cy0 - 9, cw - 24, 5);
      }
      g.fillStyle = "#5d8e52";
      for (let x = 0; x < w; x += 2) {
        const mh = 20 + Math.sin(x * 0.035) * 9 + Math.cos(x * 0.09) * 4;
        g.fillRect(x, h - mh, 2, mh);
      }
      g.fillStyle = "#48743e";
      for (let x = 0; x < w; x += 2) {
        const mh = 11 + Math.sin(x * 0.07 + 1) * 4;
        g.fillRect(x, h - mh, 2, mh);
      }
    });
    S.backdrop(skyT, cx - 720, cx + 720, -320, -100, 220);

    const outGrassT = S.tex(64, 64, (g) => {
      g.fillStyle = "#3f7e34"; g.fillRect(0, 0, 64, 64);
      g.fillStyle = "#4c923e"; for (let y = 0; y < 64; y += 16) g.fillRect(0, y, 64, 8);
    });
    S.ground(outGrassT, -460, -280, aw + 460, ah + 260, { texel: 1.5, fog: 0.9 });

    // ---- 2. Stepped Bleachers + 3D Cantilevered Dugout Roof Canopies ----
    const crowdStrip = (k) => S.tex(80, 14, (g) => {
      const r = U.seeded(880 + k * 53);
      const shirts = ["#e64840", "#4284dc", "#f0c43c", "#4eb25c", "#f4f0e8", "#9454b8"];
      const hats = ["#c83830", "#2c64b0", "#f4c842", "#e8e4d8", "#48382c"];
      g.fillStyle = "#6e4c2e"; g.fillRect(0, 9, 80, 5);
      for (let x = 2; x < 78; x += 5) {
        g.fillStyle = shirts[(r() * shirts.length) | 0]; g.fillRect(x, 6, 4, 6);
        g.fillStyle = "#f2c89c"; g.fillRect(x + 1, 2, 2, 4);
        g.fillStyle = hats[(r() * hats.length) | 0]; g.fillRect(x, 1, 4, 2);
      }
    });
    const cr0 = crowdStrip(0), cr1 = crowdStrip(1);
    const adT = S.tex(160, 12, (g) => {
      g.fillStyle = "#6e4c2e"; g.fillRect(0, 0, 160, 12);
      const ads = [["#e6a842", "#523218", "SUNNY FARM"], ["#e8dfd0", "#2e649c", "FRESH MILK"], ["#5ea446", "#fff6d8", "HAY & SEED"], ["#d45040", "#fff4c0", "BAKE SHOP"]];
      for (let i = 0; i < 4; i++) {
        g.fillStyle = ads[i][0]; g.fillRect(i * 40 + 2, 1, 36, 10);
        g.fillStyle = ads[i][1]; Pixel.text(g, ads[i][2], i * 40 + 5, 4, 1);
      }
    });
    const timberWall = S.tex(24, 18, (g) => {
      g.fillStyle = "#8a603a"; g.fillRect(0, 0, 24, 18);
      g.fillStyle = "#684426"; for (let y = 0; y < 18; y += 4) g.fillRect(0, y, 24, 1);
      g.fillStyle = "#a87a4e"; for (let y = 1; y < 18; y += 4) g.fillRect(0, y, 24, 1);
    });
    const awningT = S.tex(24, 16, (g) => {
      for (let x = 0; x < 24; x++) {
        g.fillStyle = (Math.floor(x / 6) % 2) ? "#f2ece0" : "#c83c32";
        g.fillRect(x, 0, 1, 16);
      }
      g.fillStyle = "rgba(0,0,0,0.2)"; g.fillRect(0, 13, 24, 3);
    });

    // North tiered bleachers behind ad wall
    S.wall(adT, af.x - 14, af.y - 12, af.x + af.w + 14, af.y - 12, 0, 12, { texel: 1 });
    for (let t = 0; t < 3; t++) {
      const bz = 10 + t * 8, by = af.y - 22 - t * 14;
      S.box(af.x - 18, by, af.w + 36, 14, 0, bz, { top: timberWall, front: timberWall, left: timberWall, right: timberWall }, { texel: 1 });
      S.wall(t % 2 ? cr0 : cr1, af.x - 10, by + 11, af.x + af.w + 10, by + 11, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.7 } });
    }
    // Cast long diagonal shadow from north stands onto clay track
    S.castShadow(af.x - 18, af.y - 50, af.w + 36, 40, 26, 0.34);

    // West tiered bleachers + overhanging red-and-cream striped 3D ballpark roof canopy!
    S.wall(adT, af.x - 14, af.y - 12, af.x - 14, fy + 10, 0, 12, { texel: 1 });
    for (let t = 0; t < 4; t++) {
      const bx = af.x - 28 - t * 14, bz = 10 + t * 8;
      S.box(bx, af.y - 12, 14, af.h + 22, 0, bz, { top: timberWall, front: timberWall, left: timberWall, right: timberWall }, { texel: 1 });
      S.wall(t % 2 ? cr1 : cr0, bx + 9, af.y - 8, bx + 9, fy + 6, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.7 } });
    }
    // Overhanging 3D striped awning roof atop west stands + steel pillars + crisp long shadow across track!
    const roofX0 = af.x - 84, roofX1 = af.x - 30, roofY0 = af.y - 16, roofY1 = fy + 14;
    S.quad(awningT, [roofX0, roofY1, 56], [roofX1, roofY1, 46], [roofX1, roofY0, 46], [roofX0, roofY0, 56], { normal: [0.35, 0, 0.94], texel: 1 });
    S.wall(S.flat("#8a2620"), roofX0, roofY1, roofX1, roofY1, 44, 56, { normal: [0, 1, 0] });
    for (let py = roofY0 + 16; py <= roofY1 - 12; py += 68) {
      S.box(roofX0 + 4, py, 4, 4, 34, 22, { all: S.flat("#52463a"), top: S.flat("#82705e") });
      S.box(roofX1 - 6, py, 3, 3, 28, 18, { all: S.flat("#52463a") });
    }
    // Dramatic west stand shadow stretching across the clay warning track!
    S.castShadow(roofX0, roofY0, roofX1 - roofX0, roofY1 - roofY0, 46, 0.36);

    // East tiered stands (shorter sunlit opposite bleachers)
    S.wall(adT, af.x + af.w + 14, fy + 10, af.x + af.w + 14, af.y - 12, 0, 12, { texel: 1 });
    for (let t = 0; t < 2; t++) {
      const bx = af.x + af.w + 16 + t * 14, bz = 10 + t * 8;
      S.box(bx, af.y - 10, 14, af.h + 20, 0, bz, { top: timberWall, front: timberWall, left: timberWall, right: timberWall }, { texel: 1 });
      S.wall(t % 2 ? cr0 : cr1, bx + 4, af.y + af.h + 6, bx + 4, af.y - 6, bz, bz + 12, { texel: 1, anim: { type: "hop", amp: 0.7 } });
    }

    // ---- 3. Center Scoreboard Tower & Rustic Fairground Barn / Cottages ----
    const sbT = S.tex(88, 32, (g) => {
      g.fillStyle = "#263628"; g.fillRect(0, 0, 88, 32);
      g.fillStyle = "#6c4828"; g.strokeRect(1, 1, 86, 30);
      g.fillStyle = "#f2c852"; Pixel.text(g, "CLAY PITCH", 16, 6, 1.5);
      g.fillStyle = "#d8f2d8"; Pixel.text(g, "HOME 04   VIS 03", 10, 20, 1);
    });
    S.box(cx - 46, af.y - 62, 10, 8, 0, 36, { all: S.flat("#5e4228") }, { shadow: 0.45 });
    S.box(cx + 36, af.y - 62, 10, 8, 0, 36, { all: S.flat("#5e4228") }, { shadow: 0.45 });
    S.box(cx - 48, af.y - 66, 96, 10, 34, 34, { all: S.flat("#423020"), front: sbT, top: S.flat("#7a5838") }, { shadow: 0.5 });

    // Red-roofed country barn & farmhouse northwest of stadium (visible right through top-left view!)
    S.house3D(af.x - 148, 16, 0, {
      w: 54, d: 36, h: 24, rh: 15, ov: 3, ridge: "ew",
      wall: "#c84434", trim: "#f2ece0", roof: "#584c42", roofHi: "#847264", roofSh: "#342c26",
      masonry: false, chimney: false, doorSide: "c", shadow: 0.52,
    });
    // Cylindrical country grain silo beside barn
    S.prism(S.flat("#c8ced4"), af.x - 86, 26, 10, 10, 0, 34, 8, { top: null });
    S.prism(S.flat("#944038"), af.x - 86, 26, 11, 1, 34, 10, 8, { top: null });
    S.castConeShadow(af.x - 86, 26, 10, 42, 0.46);

    // Striped Fairground Concession Tents along outer west perimeter + parted tent flaps + nearby shade oaks!
    const tentFront = S.tex(32, 16, (g) => {
      g.fillStyle = "#efe6d4"; g.fillRect(0, 0, 32, 16);
      g.fillStyle = "#564234"; g.fillRect(10, 3, 12, 13);
      g.fillStyle = "#ffa848"; g.fillRect(13, 5, 6, 4);
      g.fillStyle = "#d8c8ae"; g.fillRect(0, 0, 32, 2);
    });
    for (const [tx, ty, c0, c1] of [[af.x - 148, 108, "#d44238", "#f6efe2"], [af.x - 148, 186, "#3672bc", "#f6efe2"], [af.x - 148, 266, "#e8a434", "#f6efe2"]]) {
      S.box(tx, ty, 32, 26, 0, 16, { all: S.flat("#ebe2d2"), front: tentFront }, { shadow: 0.48 });
      S.prism(S.tex(24, 12, (g) => { for (let i = 0; i < 24; i += 3) { g.fillStyle = (i / 3) % 2 ? c1 : c0; g.fillRect(i, 0, 3, 12); } }), tx + 16, ty + 13, 21, 1.5, 16, 13, 4, { rot: 0.785, top: null });
    }
    // Red country tractor on west apron
    S.box(af.x - 122, 340, 22, 13, 2, 10, { all: S.flat("#c8382c"), top: S.flat("#ea5442") }, { shadow: 0.45 });
    S.box(af.x - 110, 342, 10, 9, 12, 8, { all: S.flat("#dce4ec"), top: S.flat("#c8382c") });

    // Floodlight Pylon Towers in corners + diagonal sunny light shafts
    for (const [tx, ty] of [[af.x - 68, af.y - 68], [af.x + af.w + 48, af.y - 68]]) {
      S.box(tx, ty, 8, 8, 0, 74, { all: S.flat("#6a727e"), top: S.flat("#949eaa") }, { shadow: 0.5 });
      S.box(tx - 8, ty + 1, 24, 6, 74, 12, { all: S.flat("#3c4452"), front: S.flat("#fff4c4"), top: S.flat("#848e9c") }, { emit: 0.85 });
    }

    // ---- 4. 3D Volumetric Oaks & Conical Pines Ringing the Fairground ----
    const warmOak = ["#265a24", "#3d8630", "#64b644", "#98e05a"];
    const warmPine = ["#1b4a26", "#2f6e32", "#4e983c", "#7ec652"];
    for (const [k, tx, ty, th, tr] of [
      ["o", af.x - 108, 66, 48, 16], ["p", af.x - 106, 152, 50, 15], ["o", af.x - 110, 238, 48, 16],
      ["p", 18, 28, 50, 15], ["o", 48, 86, 48, 17], ["p", 18, 164, 52, 15], ["o", 38, 244, 50, 17], ["p", 20, 336, 50, 15],
      ["o", cx - 170, -18, 54, 19], ["p", cx - 95, -24, 52, 15], ["o", cx + 100, -22, 54, 19], ["p", cx + 180, -18, 52, 15],
      ["p", aw - 36, 46, 52, 15], ["o", aw - 32, 148, 52, 18], ["p", aw - 34, 248, 50, 15], ["o", aw - 32, 344, 50, 17],
    ]) {
      if (k === "p") S.pine3D(tx, ty, 0, { h: th, r: tr, tiers: 4, pal: warmPine, anim: { type: "sway", amp: 0.8 }, shadow: 0.48 });
      else S.tree3D(tx, ty, 0, { kind: "oak", h: th, r: tr, pal: warmOak, anim: { type: "sway", amp: 0.85 }, shadow: 0.48 });
    }

    // ---- 5. Foreground Timber Dugout Railing & Flowerboxes (Front Bokeh Layer) ----
    S.fence3D(af.x - 16, fy + 14, af.x + af.w + 16, fy + 14, 0, { h: 10, step: 18, col: "#8a603a", cap: "#b88a56", layer: "front" });
    S.pine3D(af.x - 42, fy + 26, 0, { h: 46, r: 14, tiers: 4, pal: warmPine, layer: "front", anim: { type: "sway", amp: 0.8 }, shadow: 0.44 });
    S.tree3D(af.x + af.w + 36, fy + 26, 0, { kind: "oak", h: 46, r: 16, pal: warmOak, layer: "front", anim: { type: "sway", amp: 0.8 }, shadow: 0.44 });
  },
});
