// Ground relief, ice sheets and dig mounds. All geometry is in art px (1 art px = Arenas.K world units).
//  - A.hm: a height map over the whole arena (0 = flat, < 0 = pit, > 0 = heaped soil). Craters, cracks and
//    dig trails are stamped into it; A.relief is a shading layer baked from it (slope light from the upper
//    left + short cast shadows), so a hole reads as a real dent: dark upper wall, lit lower wall, raised lip.
//  - A.iceC: frozen sheets laid by ice moves. A per-cell life grid melts them away patchily; the cells are
//    also what Arenas.terrainAt reports as "ice" (slippery; on the water field you can stand on them).
//  - drawMound: the travelling hump of earth over a burrowing Pokémon.
"use strict";

const Ground = (() => {
  const SQ = 0.62;                 // floor ellipse squash (matches decals / slabs)
  const CELL = 3;                  // ice cell size, art px
  const LX = -0.55, LY = -0.62, LZ = 0.56; // light from the upper left, fairly low so dents cast shadows
  const SH_STEPS = 9, SH_RISE = 0.55;      // cast-shadow march (art px per step / height gained per step)

  function init(A) {
    A.hm = new Float32Array(A.aw * A.ah);
    A.relief = U.canvas(A.aw, A.ah);
    A.iceC = U.canvas(A.aw, A.ah);
    A.icw = Math.ceil(A.aw / CELL); A.ich = Math.ceil(A.ah / CELL);
    A.iceLife = new Float32Array(A.icw * A.ich);
    A.iceN = 0; A.iceTick = 0;
    A.crystals = [];
  }

  // ---- height map --------------------------------------------------------------------------------------
  const inField = (A, x, y) => x >= A.af.x && x < A.af.x + A.af.w && y >= A.af.y && y < A.af.y + A.af.h;
  function H(A, x, y) {
    x = x | 0; y = y | 0;
    if (x < 0 || y < 0 || x >= A.aw || y >= A.ah) return 0;
    return A.hm[y * A.aw + x];
  }
  function noise(x, y, s) { const v = Math.sin(x * 12.9898 + y * 78.233 + s * 37.719) * 43758.5453; return v - Math.floor(v); }

  // Bowl crater (world units): depth in art px, a heaped lip just outside the rim.
  function crater(A, wx, wy, wr, depth = 6) {
    if (A.water) return;
    const K = Arenas.K, cx = wx / K, cy = wy / K, r = Math.max(3, wr / K), ry = r * SQ, s = Math.random() * 100;
    const x0 = Math.floor(cx - r * 1.5), x1 = Math.ceil(cx + r * 1.5), y0 = Math.floor(cy - ry * 1.5), y1 = Math.ceil(cy + ry * 1.5);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      if (!inField(A, x, y)) continue;
      const a = Math.atan2((y - cy) / SQ, x - cx), wob = 1 + 0.12 * Math.sin(a * 3 + s) + 0.07 * Math.sin(a * 7 + s * 2);
      const d = Math.hypot(x - cx, (y - cy) / SQ) / (r * wob);
      let h = 0;
      if (d < 1) h = -depth * (1 - d * d) * (0.85 + 0.15 * noise(x, y, s));
      h += depth * 0.32 * Math.exp(-(((d - 1.02) / 0.2) ** 2)); // the lip
      const i = y * A.aw + x;
      A.hm[i] = U.clamp(h < 0 ? Math.min(A.hm[i], A.hm[i] * 0.4 + h) : A.hm[i] + h, -16, 9);
    }
    shade(A, x0, y0, x1, y1);
  }

  // A narrow groove (crack / scar) between two world points.
  function groove(A, wx0, wy0, wx1, wy1, ww = 6, depth = 2.2) {
    if (A.water) return;
    const K = Arenas.K, x0 = wx0 / K, y0 = wy0 / K, x1 = wx1 / K, y1 = wy1 / K, hw = Math.max(0.8, ww / K / 2);
    const L = Math.hypot(x1 - x0, y1 - y0) || 1;
    let bx0 = 1e9, by0 = 1e9, bx1 = -1e9, by1 = -1e9;
    for (let d = 0; d <= L; d += 0.5) {
      const cx = x0 + (x1 - x0) * d / L, cy = y0 + (y1 - y0) * d / L, taper = Math.min(1, d / 3, (L - d) / 3 + 0.3);
      for (let oy = -Math.ceil(hw) - 1; oy <= Math.ceil(hw) + 1; oy++) for (let ox = -Math.ceil(hw) - 1; ox <= Math.ceil(hw) + 1; ox++) {
        const x = Math.round(cx + ox), y = Math.round(cy + oy);
        if (!inField(A, x, y)) continue;
        const q = Math.hypot(ox, oy) / hw;
        if (q > 1.6) continue;
        const i = y * A.aw + x, h = q < 1 ? -depth * taper * (1 - q * q) : depth * 0.25 * taper;
        A.hm[i] = U.clamp(h < 0 ? Math.min(A.hm[i], h) : Math.max(A.hm[i], A.hm[i] + h * 0.3), -16, 9);
        bx0 = Math.min(bx0, x); by0 = Math.min(by0, y); bx1 = Math.max(bx1, x); by1 = Math.max(by1, y);
      }
    }
    if (bx1 >= bx0) shade(A, bx0, by0, bx1, by1);
  }

  // Heaped, broken-up soil (dig trail): a low bump.
  function heap(A, wx, wy, wr, height = 2.5) {
    if (A.water) return;
    const K = Arenas.K, cx = wx / K, cy = wy / K, r = Math.max(2, wr / K), s = Math.random() * 100;
    const x0 = Math.floor(cx - r), x1 = Math.ceil(cx + r), y0 = Math.floor(cy - r * SQ), y1 = Math.ceil(cy + r * SQ);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      if (!inField(A, x, y)) continue;
      const d = Math.hypot(x - cx, (y - cy) / SQ) / r;
      if (d >= 1) continue;
      const i = y * A.aw + x;
      A.hm[i] = Math.min(9, Math.max(A.hm[i], height * (1 - d * d) * (0.6 + 0.4 * noise(x, y, s)) + Math.min(0, A.hm[i]) * 0.3));
    }
    shade(A, x0, y0, x1, y1);
  }

  // Bake the shading of a rectangle of the height map into A.relief (3 dithered tones each way).
  function shade(A, x0, y0, x1, y1) {
    const pad = SH_STEPS + 2;
    x0 = Math.max(A.af.x, Math.floor(x0) - 2); y0 = Math.max(A.af.y, Math.floor(y0) - 2);
    x1 = Math.min(A.af.x + A.af.w - 1, Math.ceil(x1) + pad); y1 = Math.min(A.af.y + A.af.h - 1, Math.ceil(y1) + pad);
    const w = x1 - x0 + 1, h = y1 - y0 + 1;
    if (w <= 0 || h <= 0) return;
    const bb = A.hmBox || (A.hmBox = { x0: 1e9, y0: 1e9, x1: -1e9, y1: -1e9 }); // where any relief exists
    bb.x0 = Math.min(bb.x0, x0); bb.y0 = Math.min(bb.y0, y0); bb.x1 = Math.max(bb.x1, x1); bb.y1 = Math.max(bb.y1, y1);
    const ctx = A.relief.getContext("2d"), img = ctx.createImageData(w, h), D = img.data;
    const ice = A.def.floor === "ice", dark = ice ? [30, 70, 110] : [24, 14, 10], lite = ice ? [240, 252, 255] : [255, 236, 196];
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const hc = H(A, x, y);
      const gx = (H(A, x + 1, y) - H(A, x - 1, y)) / 2, gy = (H(A, x, y + 1) - H(A, x, y - 1)) / 2 / SQ;
      if (hc === 0 && gx === 0 && gy === 0 && H(A, x - 3, y - 3) === 0 && H(A, x - 6, y - 6) === 0) continue;
      const nl = Math.hypot(gx, gy, 1), lam = (-gx * LX - gy * LY + LZ) / nl - LZ; // brighter (+) / darker (−) than flat ground
      // cast shadow: does the ground toward the light rise above the light ray?
      let sh = 0;
      for (let k = 1; k <= SH_STEPS; k++) { if (H(A, x + LX * k * 1.4, y + LY * k * 1.4 * SQ) > hc + k * SH_RISE) { sh = 1; break; } }
      let v = lam * 1.6 - sh * 0.45 + Math.min(0, hc) * 0.022; // deep floors are darker (less sky)
      const b = Pixel.bayer(x, y), i = ((y - y0) * w + (x - x0)) * 4;
      const lvl = Math.abs(v) < 0.05 ? 0 : Math.min(3, Math.floor(Math.abs(v) * 5 + b * 0.9));
      if (!lvl) continue;
      const c = v < 0 ? dark : lite, a = v < 0 ? [0, 0.2, 0.36, 0.55][lvl] : [0, 0.12, 0.22, 0.34][lvl];
      D[i] = c[0]; D[i + 1] = c[1]; D[i + 2] = c[2]; D[i + 3] = a * 255;
    }
    ctx.putImageData(img, x0, y0);
  }

  // Old damage slowly settles (called with the decal fade).
  function settle(A, k = 0.88) {
    const hm = A.hm;
    for (let i = 0; i < hm.length; i++) if (hm[i]) { hm[i] *= k; if (Math.abs(hm[i]) < 0.2) hm[i] = 0; }
    const bb = A.hmBox;
    if (!bb) return;
    A.relief.getContext("2d").clearRect(bb.x0, bb.y0, bb.x1 - bb.x0 + 1, bb.y1 - bb.y0 + 1);
    shade(A, bb.x0, bb.y0, bb.x1, bb.y1);
  }

  // Height (art px, negative = below ground) under a world point.
  function heightAt(A, wx, wy) { return A.hm ? H(A, wx / Arenas.K, wy / Arenas.K) : 0; }

  // ---- ice ----------------------------------------------------------------------------------------------
  const cellOf = (A, x, y) => { const cx = Math.floor(x / CELL), cy = Math.floor(y / CELL); return cx < 0 || cy < 0 || cx >= A.icw || cy >= A.ich ? -1 : cy * A.icw + cx; };
  function setCell(A, cx, cy, life) {
    if (cx < 0 || cy < 0 || cx >= A.icw || cy >= A.ich) return false;
    const px = cx * CELL + 1, py = cy * CELL + 1;
    if (!inField(A, px, py)) return false;
    const i = cy * A.icw + cx;
    if (A.iceLife[i] <= 0) A.iceN++;
    A.iceLife[i] = Math.max(A.iceLife[i], life);
    return true;
  }

  // Freeze a strip between two world points (ww = width in world units). Returns false when nothing new froze.
  function freezeLine(A, wx0, wy0, wx1, wy1, ww, life = 13) {
    const K = Arenas.K, x0 = wx0 / K, y0 = wy0 / K, x1 = wx1 / K, y1 = wy1 / K, hw = ww / K / 2;
    const L = Math.hypot(x1 - x0, y1 - y0) || 1, nx = -(y1 - y0) / L, ny = (x1 - x0) / L;
    let bx0 = 1e9, by0 = 1e9, bx1 = -1e9, by1 = -1e9;
    for (let d = 0; d <= L; d += CELL * 0.7) {
      const e = hw * (0.62 + 0.24 * Math.sin(d * 0.19 + x0) + 0.16 * Math.sin(d * 0.53 + y0) + 0.18 * noise(d, x0, 7)); // ragged edge
      for (let o = -e; o <= e; o += CELL * 0.7) {
        const x = x0 + (x1 - x0) * d / L + nx * o, y = y0 + (y1 - y0) * d / L + ny * o * SQ;
        const core = 1 - Math.abs(o) / (e + 0.01);
        if (setCell(A, Math.floor(x / CELL), Math.floor(y / CELL), life * (0.6 + 0.4 * core) + noise(x, y, 3) * 3)) {
          bx0 = Math.min(bx0, x); by0 = Math.min(by0, y); bx1 = Math.max(bx1, x); by1 = Math.max(by1, y);
        }
      }
    }
    if (bx1 >= bx0) paintIce(A, bx0, by0, bx1, by1);
  }
  function freezeDisc(A, wx, wy, wr, life = 12) {
    const K = Arenas.K, cx = wx / K, cy = wy / K, r = wr / K, s = Math.random() * 10;
    for (let y = cy - r * SQ * 1.2; y <= cy + r * SQ * 1.2; y += CELL) for (let x = cx - r * 1.2; x <= cx + r * 1.2; x += CELL) {
      const a = Math.atan2((y - cy) / SQ, x - cx), d = Math.hypot(x - cx, (y - cy) / SQ) / (r * (1 + 0.18 * Math.sin(a * 5 + s)));
      if (d < 1) setCell(A, Math.floor(x / CELL), Math.floor(y / CELL), life * (0.55 + 0.45 * (1 - d)) + noise(x, y, s) * 3);
    }
    paintIce(A, cx - r * 1.3, cy - r * 1.3, cx + r * 1.3, cy + r * 1.3);
  }
  // Fire on ice: melt cells inside a disc. Returns how many melted.
  function melt(A, wx, wy, wr) {
    if (!A.iceN) return 0;
    const K = Arenas.K, cx = wx / K, cy = wy / K, r = wr / K;
    let n = 0;
    for (let y = cy - r; y <= cy + r; y += CELL) for (let x = cx - r; x <= cx + r; x += CELL) {
      if (Math.hypot(x - cx, (y - cy) / SQ) > r) continue;
      const i = cellOf(A, x, y);
      if (i >= 0 && A.iceLife[i] > 0) { A.iceLife[i] = 0; A.iceN--; n++; }
    }
    if (n) paintIce(A, cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4);
    for (const c of A.crystals) if (Math.hypot(c.x / K - cx, (c.y / K - cy) / SQ) < r) c.life = Math.min(c.life, 0.2);
    return n;
  }
  function iceAt(A, wx, wy) {
    if (!A.iceN) return false;
    const i = cellOf(A, wx / Arenas.K, wy / Arenas.K);
    return i >= 0 && A.iceLife[i] > 0;
  }

  // Repaint the ice canvas inside a rectangle (art px) from the cell grid.
  function paintIce(A, x0, y0, x1, y1) {
    const c0 = Math.max(0, Math.floor(x0 / CELL) - 1), r0 = Math.max(0, Math.floor(y0 / CELL) - 1);
    const c1 = Math.min(A.icw - 1, Math.ceil(x1 / CELL) + 1), r1 = Math.min(A.ich - 1, Math.ceil(y1 / CELL) + 1);
    const ctx = A.iceC.getContext("2d");
    ctx.clearRect(c0 * CELL, r0 * CELL, (c1 - c0 + 1) * CELL, (r1 - r0 + 1) * CELL + 2);
    const L = A.iceLife, W = A.icw, wet = A.water;
    const on = (cx, cy) => cx >= 0 && cy >= 0 && cx < W && cy < A.ich && L[cy * W + cx] > 0;
    for (let cy = r0; cy <= r1; cy++) for (let cx = c0; cx <= c1; cx++) {
      const life = L[cy * W + cx];
      if (life <= 0) { // hoar frost dusting the ground right next to the sheet
        if (!wet && (on(cx - 1, cy) || on(cx + 1, cy) || on(cx, cy - 1) || on(cx, cy + 1))) for (let k = 0; k < 3; k++) { const x = cx * CELL + (k * 2 + cy) % CELL, y = cy * CELL + (k + cx) % CELL; if (noise(x, y, 5) < 0.45) { ctx.fillStyle = "rgba(236,248,255,0.7)"; ctx.fillRect(x, y, 1, 1); } }
        continue;
      }
      const thin = life < 2.5; // about to melt: glassy, see-through
      for (let py = 0; py < CELL; py++) for (let px = 0; px < CELL; px++) {
        const x = cx * CELL + px, y = cy * CELL + py, n = noise(x, y, 1);
        const top = py === 0 && !on(cx, cy - 1), bot = py === CELL - 1 && !on(cx, cy + 1), lef = px === 0 && !on(cx - 1, cy), rig = px === CELL - 1 && !on(cx + 1, cy);
        let col;
        if (top || lef) col = thin ? "rgba(235,250,255,0.55)" : "#f4fcff";              // lit frosty rim
        else if (bot || rig) col = thin ? "rgba(120,180,220,0.5)" : "#8cc4e4";           // rim in shade
        else if (((x + y) % 17 === 0 || (x + y) % 17 === 1) && noise(Math.floor((x - y) / 6), 0, 2) < 0.35) col = thin ? "rgba(255,255,255,0.45)" : "#ffffff"; // diagonal shine streaks
        else if (noise(Math.floor(x / 4), Math.floor(y / 3), 4) < 0.08 && n < 0.5) col = thin ? "rgba(120,190,235,0.5)" : "#9fd0f0"; // deeper, clearer ice
        else col = thin ? (wet ? "rgba(170,220,250,0.45)" : "rgba(205,238,255,0.45)") : wet ? (n < 0.55 ? "#cdeefe" : "#bde5fa") : n < 0.55 ? "#d6f2ff" : "#c4e9fb";
        ctx.fillStyle = col; ctx.fillRect(x, y, 1, 1);
        if (bot && wet && !thin) { ctx.fillStyle = "#3d7fae"; ctx.fillRect(x, y + 1, 1, 2); } // a floe has thickness above the water
      }
    }
  }

  function addCrystal(A, wx, wy, size) {
    if (A.crystals.length > 80) A.crystals.shift();
    A.crystals.push({ x: wx, y: wy, h: size, life: 9 + Math.random() * 5, t: 0, seed: Math.random() * 100 | 0, n: 1 + (Math.random() * 3 | 0) });
  }

  function update(A, dt, fx) {
    for (let i = A.crystals.length - 1; i >= 0; i--) {
      const c = A.crystals[i]; c.t += dt; c.life -= dt;
      if (c.life <= 0) { if (fx) for (let k = 0; k < 3; k++) fx.add({ x: c.x, y: c.y, z: 4, vx: U.rand(-40, 40), vy: U.rand(-20, 20), vz: U.rand(40, 120), grav: 600, life: 0.5, size: 3, color: "#dff6ff", shape: "shard", add: true }); A.crystals.splice(i, 1); }
    }
    if (!A.iceN) return;
    A.iceTick += dt;
    if (A.iceTick < 0.25) return;
    const step = A.iceTick; A.iceTick = 0;
    const L = A.iceLife, W = A.icw;
    let bx0 = 1e9, by0 = 1e9, bx1 = -1e9, by1 = -1e9;
    for (let i = 0; i < L.length; i++) {
      const v = L[i];
      if (v <= 0) continue;
      const nv = v - step;
      if ((v >= 2.5 && nv < 2.5) || nv <= 0) {
        const cx = i % W, cy = (i / W) | 0;
        bx0 = Math.min(bx0, cx); by0 = Math.min(by0, cy); bx1 = Math.max(bx1, cx); by1 = Math.max(by1, cy);
        if (nv <= 0) { A.iceN--; if (fx && Math.random() < 0.04) fx.add({ x: (cx + 0.5) * CELL * Arenas.K, y: (cy + 0.5) * CELL * Arenas.K, z: 2, vz: U.rand(10, 30), life: 0.8, size: 6, color: "#e8f6ff", shape: "smoke", alpha: 0.35, grow: 1.6 }); }
      }
      L[i] = nv > 0 ? nv : 0;
    }
    if (bx1 >= 0) paintIce(A, bx0 * CELL, by0 * CELL, (bx1 + 1) * CELL, (by1 + 1) * CELL);
  }

  // Ice sheet + moving glints (world canvas, art px).
  function drawIce(ctx, A, ix, iy, t) {
    if (!A.iceN) return;
    ctx.drawImage(A.iceC, -ix, -iy);
    // a slow diagonal shine sweeping across the frozen cells
    const band = (t * 60) % (A.aw + A.ah);
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    for (let k = 0; k < 40; k++) {
      const x = ((k * 97) % A.aw), y = band - x;
      if (y < 0 || y >= A.ah) continue;
      const i = cellOf(A, x, y);
      if (i >= 0 && A.iceLife[i] > 2.5) ctx.fillRect(x - ix, y - iy, 1, 1);
    }
  }

  // Standing ice crystals (sorted with the fighters).
  function entities(A, out) { for (const c of A.crystals) out.push({ y: c.y, crystal: c }); }
  function drawCrystal(ctx, c, ix, iy) {
    const K = Arenas.K, grow = Math.min(1, c.t / 0.12), fade = Math.min(1, c.life / 1.2);
    const h = Math.max(1, Math.round(c.h * grow * (0.4 + 0.6 * fade))), x = Math.round(c.x / K - ix), y = Math.round(c.y / K - iy);
    const rng = U.seeded(c.seed);
    Pixel.ellipse(ctx, x, y, Math.max(2, Math.round(h * 0.5)), 1, "rgba(40,90,140,0.35)");
    for (let k = 0; k < c.n; k++) {
      const ox = Math.round((rng() - 0.5) * h * 0.9), hh = Math.max(1, Math.round(h * (k ? 0.55 + rng() * 0.3 : 1))), lean = Math.round((rng() - 0.5) * 2), w = Math.max(1, Math.round(hh * 0.28));
      for (let j = 0; j < hh; j++) {
        const ww = Math.max(1, Math.round(w * (1 - j / hh) + 0.4)), sx = x + ox + Math.round(lean * j / hh);
        ctx.fillStyle = "#6fb2dc"; ctx.fillRect(sx - ww, y - j, ww, 1);          // shaded face
        ctx.fillStyle = j > hh * 0.6 ? "#ffffff" : "#d8f4ff"; ctx.fillRect(sx, y - j, ww, 1); // lit face
      }
      ctx.fillStyle = "#3f7fae"; ctx.fillRect(x + ox - w, y, w * 2 + 1, 1);
    }
  }

  // ---- dig mound -------------------------------------------------------------------------------------------
  // m = { x, y (world), r (world), h (art px), seed }. The floor itself bulges: each column of the patch is lifted
  // (most in the middle), the soil shows on the front face, crumbs and cracks on top.
  const tmp = U.canvas(80, 60);
  function drawMound(ctx, A, m, ix, iy, t, floor) {
    const K = Arenas.K, cx = m.x / K, cy = m.y / K, r = Math.max(4, Math.round(m.r / K)), ry = Math.max(3, Math.round(r * SQ));
    const [, earth] = Arenas.chunkColors(A);
    const W = r * 2 + 2, Hh = ry * 2 + 2, sx = Math.round(cx - r - 1), sy = Math.round(cy - ry - 1);
    if (tmp.width < W || tmp.height < Hh) { tmp.width = Math.max(tmp.width, W); tmp.height = Math.max(tmp.height, Hh); }
    const x = tmp.getContext("2d"); x.clearRect(0, 0, tmp.width, tmp.height);
    x.drawImage(floor, -sx, -sy); x.drawImage(A.decals, -sx, -sy);
    // ground shadow ring around the base
    Pixel.ellipse(ctx, cx - ix + 1, cy - iy + 1, r + 2, ry + 1, "rgba(20,12,6,0.4)");
    const rng = U.seeded(m.seed + Math.floor(t * 10));
    for (let i = -r; i <= r; i++) {
      const q = i / r, colH = Math.round(ry * Math.sqrt(Math.max(0, 1 - q * q))), lift = Math.round(m.h * Math.sqrt(Math.max(0, 1 - q * q)) * (0.85 + 0.15 * Math.sin(t * 30 + i)));
      if (colH <= 0) continue;
      const dx = Math.round(cx + i - ix), top = Math.round(cy - colH - iy), bot = Math.round(cy + colH - iy);
      // soil face from the lifted bottom edge down to the ground
      ctx.fillStyle = earth; ctx.fillRect(dx, bot - lift, 1, lift + 1);
      ctx.fillStyle = "rgba(0,0,0,0.25)"; ctx.fillRect(dx, bot - Math.max(1, lift >> 1), 1, Math.max(1, lift >> 1) + 1);
      // the lifted floor column
      ctx.drawImage(tmp, i + r + 1, 1 + ry - colH, 1, colH * 2, dx, top - lift, 1, colH * 2);
      // light on the upper-left of the dome, shade on the right
      const lit = q < -0.15 ? 0.22 : q > 0.45 ? -0.22 : 0.08;
      ctx.fillStyle = lit > 0 ? `rgba(255,240,210,${lit})` : `rgba(0,0,0,${-lit})`; ctx.fillRect(dx, top - lift, 1, colH * 2);
      if (rng() < 0.12) { ctx.fillStyle = earth; ctx.fillRect(dx, top - lift + Math.round(rng() * colH * 2), 1, 1); } // torn turf
    }
    // cracks across the top
    const cr = U.seeded(m.seed);
    for (let k = 0; k < 4; k++) {
      let px = cx - ix, py = cy - iy - m.h; const a = cr() * U.TAU;
      for (let j = 0; j < r * 0.8; j++) { px += Math.cos(a) + (cr() - 0.5); py += Math.sin(a) * SQ + (cr() - 0.5) * 0.5; ctx.fillStyle = "rgba(30,18,8,0.8)"; ctx.fillRect(Math.round(px), Math.round(py), 1, 1); }
    }
  }

  return { init, crater, groove, heap, shade, settle, heightAt, freezeLine, freezeDisc, melt, iceAt, update, drawIce, addCrystal, entities, drawCrystal, drawMound, SQ };
})();
