// Water surface life: ripples, V-shaped wakes behind swimmers, splash crowns and spray.
// Rings live in art px on the arena (A.ripples) and are drawn on the world layer right after the floor,
// so Pokémon and platforms sit on top of them.
"use strict";

const Water = (() => {
  const SQ = 0.62;

  function ripples(A) { return A.ripples || (A.ripples = []); }

  // Expanding ring (world units). a0/a1 limit it to an arc (radians, 0 = east); speed in world units / s.
  function ripple(A, wx, wy, r0, speed, life, o = {}) {
    const L = ripples(A);
    if (L.length > 260) L.splice(0, 40);
    L.push({ x: wx, y: wy, r: r0, v: speed, life, max: life, a0: o.a0, a1: o.a1, foam: !!o.foam, w: o.w || 1 });
  }

  // Big splash at a world point. power ≈ 0.5 (a step) … 2.5 (a heavy body slamming in).
  function splash(B, wx, wy, power = 1, color) {
    const A = B.arena, fx = B.fx, n = Math.round(8 + power * 10);
    for (let i = 0; i < n; i++) { // the crown: droplets thrown up and out
      const a = U.rand(U.TAU), sp = U.rand(40, 120) * (0.6 + power * 0.4);
      fx.add({ x: wx + Math.cos(a) * 6, y: wy + Math.sin(a) * 4, z: 2, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp * SQ, vz: U.rand(160, 300) * (0.7 + power * 0.3), grav: 900, life: U.rand(0.45, 0.8), size: U.rand(3, 6), color: color || U.pick(["#ffffff", "#dff4ff", "#9fd8ff"]), shape: "drop", drag: 0.6 });
    }
    if (power > 1.2) for (let i = 0; i < 5; i++) fx.add({ x: wx + U.rand(-6, 6), y: wy, z: 4, vx: U.rand(-15, 15), vy: 0, vz: U.rand(300, 460) * power * 0.6, grav: 1000, life: 0.55, size: U.rand(6, 9), color: "#eefaff", shape: "drop", drag: 0.4 }); // column
    for (let i = 0; i < 3 + power * 2; i++) fx.add({ x: wx + U.rand(-14, 14), y: wy + U.rand(-6, 6), z: 4, vx: U.rand(-30, 30), vy: U.rand(-10, 10), vz: U.rand(20, 60), life: U.rand(0.5, 0.9), size: U.rand(8, 14) * (0.6 + power * 0.3), color: "#f2fbff", shape: "smoke", alpha: 0.45, grow: 1.6 });
    ripple(A, wx, wy, 6, 70 + power * 30, 0.9 + power * 0.25, { foam: true, w: 2 });
    ripple(A, wx, wy, 2, 45 + power * 20, 1.3 + power * 0.3);
    if (power > 0.8) ripple(A, wx, wy, 0, 28 + power * 12, 1.6 + power * 0.3);
    if (power > 0.9) Audio.sfx("water", B.pan(wx));
  }

  // Per-frame surface behaviour of one fighter (called from Fighter.update).
  function fighter(B, f, dt, wasSwim) {
    const A = B.arena;
    const spd = Math.hypot(f.vx, f.vy) + Math.hypot(f.kx || 0, f.ky || 0);
    if (f.swimming && !wasSwim) splash(B, f.x, f.y, U.clamp(0.6 + f.mass * 0.4 + (f.fallV || 0) / 400, 0.6, 2.6)); // jumped / stepped / knocked in
    if (!f.swimming && wasSwim) f.dripT = 1.1; // climbed out: drip for a moment
    if (f.dripT > 0) { f.dripT -= dt; if (Math.random() < dt * 14) B.fx.add({ x: f.x + U.rand(-f.r, f.r) * 0.6, y: f.y, z: U.rand(4, f.height * 0.6), vz: -10, grav: 700, life: 0.35, size: 2, color: "#bfe6ff", shape: "drop", drag: 0 }); }
    if (!f.swimming) { f.wakeT = 0; return; }
    f.wakeT = (f.wakeT || 0) - dt;
    if (spd > 40) {
      // Kelvin wake: short arcs pushed out behind both flanks; their trail opens into a V
      if (f.wakeT <= 0) {
        f.wakeT = 0.055;
        const h = Math.atan2(f.vy + (f.ky || 0), f.vx + (f.kx || 0)), back = h + Math.PI, open = 1.0, half = 0.32;
        for (const s of [-1, 1]) ripple(A, f.x - Math.cos(h) * f.r * 0.3, f.y - Math.sin(h) * f.r * 0.2, f.r * 0.75, 38 + spd * 0.12, 0.85, { a0: back + s * open - half, a1: back + s * open + half, foam: true });
        ripple(A, f.x, f.y, f.r * 0.6, 20, 0.5, { a0: back - 0.5, a1: back + 0.5 }); // churned water right behind
      }
      // bow spray: water shoved aside in front
      if (Math.random() < dt * (6 + spd / 25)) {
        const h = Math.atan2(f.vy, f.vx), s = Math.random() < 0.5 ? -1 : 1, a = h + s * U.rand(0.9, 1.5);
        B.fx.add({ x: f.x + Math.cos(h) * f.r * 0.9, y: f.y + Math.sin(h) * f.r * 0.5, z: 3, vx: Math.cos(a) * U.rand(60, 120) + f.vx * 0.4, vy: Math.sin(a) * U.rand(30, 70) + f.vy * 0.4, vz: U.rand(80, 170), grav: 900, life: U.rand(0.3, 0.5), size: U.rand(2, 4), color: U.pick(["#ffffff", "#cfeeff"]), shape: "drop", drag: 0.5 });
      }
    } else if (f.wakeT <= 0) {
      f.wakeT = 0.75 + Math.random() * 0.4; // treading water: slow rings
      ripple(A, f.x, f.y, f.r * 0.7, 26, 1.4);
    }
  }

  function update(A, dt) {
    const L = A.ripples;
    if (!L) return;
    for (let i = L.length - 1; i >= 0; i--) { const p = L[i]; p.life -= dt; p.r += p.v * dt; p.v *= Math.exp(-0.9 * dt); if (p.life <= 0) { L[i] = L[L.length - 1]; L.pop(); } }
  }

  // Rings on the world canvas (art px). Older rings break up into dashes before they vanish.
  function draw(ctx, A, ix, iy) {
    const L = A.ripples;
    if (!L || !L.length) return;
    const K = Arenas.K;
    for (const p of L) {
      const k = p.life / p.max, rx = p.r / K, ry = rx * SQ;
      if (rx < 1) continue;
      const cx = p.x / K - ix, cy = p.y / K - iy;
      const n = Math.ceil((rx + ry) * 3.2), a0 = p.a0 === undefined ? 0 : p.a0, span = p.a0 === undefined ? U.TAU : p.a1 - p.a0;
      const m = Math.max(4, Math.ceil(n * span / U.TAU)), gap = k > 0.6 ? 0 : k > 0.3 ? 3 : 2;
      const col = p.foam && k > 0.5 ? `rgba(255,255,255,${0.85 * Math.min(1, k * 1.4)})` : `rgba(215,238,255,${0.7 * k})`;
      ctx.fillStyle = col;
      for (let i = 0; i <= m; i++) {
        if (gap && (i % (gap + 1)) === gap) continue;
        const a = a0 + span * (i / m);
        const x = Math.round(cx + Math.cos(a) * rx), y = Math.round(cy + Math.sin(a) * ry);
        ctx.fillRect(x, y, 1, 1);
        if (p.w > 1 && k > 0.5) ctx.fillRect(x, y + 1, 1, 1);
      }
      if (p.foam && k > 0.35) { // a darker trough just inside the crest gives the ring some body
        ctx.fillStyle = `rgba(16,60,120,${0.35 * k})`;
        for (let i = 0; i <= m; i += 2) { const a = a0 + span * (i / m); ctx.fillRect(Math.round(cx + Math.cos(a) * (rx - 1.5)), Math.round(cy + Math.sin(a) * (ry - 1)), 1, 1); }
      }
    }
  }

  return { ripple, splash, fighter, update, draw };
})();
