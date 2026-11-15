// Telekinetic floor slabs: a disc of the real arena floor (with its decals) cut into jagged plates that
// can be lifted into the air, leaving a dark pit, and slammed back down. Moves drive the heights
// (piece.z, art px); this module cuts, draws and lands them. All geometry is in art px.
"use strict";

const Slabs = (() => {
  // Cut a disc of radius R (world units) at (wx, wy) into `pieces` plates (1 = a single round plate).
  function create(B, wx, wy, R, o = {}) {
    const A = B.arena, K = Arenas.K, cx = wx / K, cy = wy / K, r = R / K, n = o.pieces || 6, SQ = 0.62;
    const jag = (k) => 1 + (Math.random() - 0.5) * k;
    const P = (a, rad) => [cx + Math.cos(a) * rad, cy + Math.sin(a) * rad * SQ];
    const polys = [];
    if (n <= 1) {
      const pts = []; for (let i = 0; i < 14; i++) pts.push(P((i / 14) * U.TAU, r * jag(0.3)));
      polys.push(pts);
    } else {
      // shared jagged rims so neighbouring plates meet along the same crack
      const M = 36, inner = [], outer = [];
      for (let i = 0; i < M; i++) { inner.push(0.42 * r * jag(0.35)); outer.push(r * jag(0.22)); }
      const at = (arr, a) => arr[((Math.round((a / U.TAU) * M) % M) + M) % M];
      polys.push(Array.from({ length: M }, (_, i) => P((i / M) * U.TAU, inner[i])));
      const N = n - 1, off = Math.random() * U.TAU, cuts = [];
      for (let k = 0; k < N; k++) cuts.push(off + ((k + 0.5 + (Math.random() - 0.5) * 0.5) / N) * U.TAU);
      const mids = cuts.map((a) => ({ a: a + (Math.random() - 0.5) * 0.25, t: 0.5 + (Math.random() - 0.5) * 0.3 }));
      for (let k = 0; k < N; k++) {
        const a0 = cuts[k], a1 = k + 1 < N ? cuts[k + 1] : cuts[0] + U.TAU, pts = [];
        const radial = (j) => { const a = cuts[j % N], m = mids[j % N]; return [P(a, at(inner, a)), P(m.a, U.lerp(at(inner, a), at(outer, a), m.t)), P(a, at(outer, a))]; };
        const r0 = radial(k), r1 = radial(k + 1);
        pts.push(r0[0], r0[1], r0[2]);
        for (let a = a0 + 0.15; a < a1 - 0.08; a += 0.17) pts.push(P(a, at(outer, a)));
        pts.push(r1[2], r1[1], r1[0]);
        for (let a = a1 - 0.15; a > a0 + 0.08; a -= 0.17) pts.push(P(a, at(inner, a)));
        polys.push(pts);
      }
    }
    const floor = Arenas.floorFrame(A, B.time);
    const [, earth] = Arenas.chunkColors(A);
    const g = { x: wx, y: wy, R, pieces: [], landed: false, t: 0, dead: false };
    for (const pts of polys) {
      let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
      for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
      x0 = Math.floor(x0); y0 = Math.floor(y0); x1 = Math.ceil(x1); y1 = Math.ceil(y1);
      const c = U.canvas(x1 - x0 + 1, y1 - y0 + 1), x = c.getContext("2d");
      path(x, pts, -x0, -y0); x.clip();
      x.drawImage(floor, -x0, -y0); x.drawImage(A.decals, -x0, -y0); x.drawImage(A.relief, -x0, -y0); if (A.iceN) x.drawImage(A.iceC, -x0, -y0);
      x.restore();
      x.strokeStyle = "rgba(255,255,255,0.35)"; x.lineWidth = 1; path(x, pts, -x0, -y0 + 0.5); x.stroke(); x.restore(); // lit rim
      const side = tint(c, earth), lit = tint(c, "#fff4e0");
      let mx = 0, my = 0; for (const [px, py] of pts) { mx += px; my += py; } mx /= pts.length; my /= pts.length;
      g.pieces.push({ pts, img: c, side, lit, x0, y0, cx: mx, cy: my, top: y0, z: 0, dx: 0, thick: Math.max(3, Math.round(r * 0.09)) });
    }
    (B.slabs || (B.slabs = [])).push(g);
    return g;
  }

  // Solid-colour silhouette of a plate (earthen sides, lit top).
  function tint(img, color) {
    const c = U.canvas(img.width, img.height), x = c.getContext("2d");
    x.drawImage(img, 0, 0); x.globalCompositeOperation = "source-in"; x.fillStyle = color; x.fillRect(0, 0, c.width, c.height);
    return c;
  }

  function path(ctx, pts, ox = 0, oy = 0) {
    ctx.save(); ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x + ox, y + oy) : ctx.moveTo(x + ox, y + oy)));
    ctx.closePath();
  }

  // Which plate (index) covers a world point, or the nearest one.
  function pieceAt(g, wx, wy) {
    const K = Arenas.K, x = wx / K, y = wy / K;
    let best = 0, bd = 1e9;
    g.pieces.forEach((p, i) => { const d = Math.hypot(p.cx - x, p.cy - y); if (d < bd) { bd = d; best = i; } });
    return best;
  }

  // Slam: plates back on the ground, a little out of place, with cracks along every seam.
  function land(B, g) {
    if (g.landed) return;
    g.landed = true;
    const A = B.arena, ctx = A.decals.getContext("2d"), f = A.af;
    ctx.save(); U.rr(ctx, f.x, f.y, f.w, f.h, f.r); ctx.clip();
    for (const p of g.pieces) {
      const ox = Math.round((Math.random() - 0.5) * 2), oy = Math.round((Math.random() - 0.5) * 2);
      ctx.drawImage(p.img, p.x0 + ox, p.y0 + oy);
      ctx.strokeStyle = "rgba(20,10,8,0.85)"; ctx.lineWidth = 1;
      path(ctx, p.pts, ox, oy); ctx.stroke(); ctx.restore();
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      path(ctx, p.pts, ox, oy - 1); ctx.stroke(); ctx.restore();
    }
    ctx.restore();
    for (const p of g.pieces) { p.z = 0; B.fx.dust(p.cx * Arenas.K, p.cy * Arenas.K, 3, "rgba(150,130,100,0.6)"); }
    g.dead = true;
  }

  function update(B, dt) {
    if (!B.slabs) return;
    for (let i = B.slabs.length - 1; i >= 0; i--) { const g = B.slabs[i]; g.t += dt; if (g.dead || g.t > 8) B.slabs.splice(i, 1); }
  }

  // Pits under lifted plates (drawn right after the floor + decals).
  function drawHoles(ctx, B, ix, iy) {
    if (!B.slabs) return;
    for (const g of B.slabs) for (const p of g.pieces) {
      if (p.z < 0.5) continue;
      const k = Math.min(1, p.z / 6);
      ctx.globalAlpha = k;
      ctx.fillStyle = "#120a16"; path(ctx, p.pts, -ix, -iy); ctx.fill(); ctx.restore();
      ctx.fillStyle = "#2a1630"; path(ctx, p.pts.map(([x, y]) => [x, y + 2]), -ix, -iy); ctx.clip(); ctx.fillRect(p.x0 - ix, p.y0 - iy, p.img.width, 3); ctx.restore();
      ctx.globalAlpha = 1;
    }
  }

  // Sortable entities for the world pass: one per lifted plate, keyed on its top edge.
  function entities(B, out) {
    if (!B.slabs) return;
    for (const g of B.slabs) for (const p of g.pieces) if (p.z >= 0.5) out.push({ y: p.top * Arenas.K, slab: p });
  }

  // A plate: the earthen side (stacked silhouette), then the top.
  function drawPiece(ctx, p, ix, iy) {
    const x = Math.round(p.x0 + p.dx - ix), y = Math.round(p.y0 - p.z - iy);
    for (let d = Math.min(p.thick, Math.round(p.z)); d >= 1; d--) ctx.drawImage(p.side, x, y + d);
    ctx.drawImage(p.img, x, y);
    ctx.globalAlpha = Math.min(0.22, p.z / 60); ctx.drawImage(p.lit, x, y); ctx.globalAlpha = 1; // the raised top catches the light
  }

  return { create, pieceAt, land, update, drawHoles, entities, drawPiece };
})();
