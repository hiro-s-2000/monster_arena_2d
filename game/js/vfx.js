// Hand-built move VFX in the retro HD-2D look (2026-11-15, user: 「かえんほうしゃ・なみのり・れいとうビームがダサい。
// 参考動画（BigKidNation / Ivandanke_Rivals）を参考に最高品質の VFX をレトロな HD-2D で」).
// Everything is drawn in world units on the low-res fx canvas, so shapes land on the art-pixel grid and Post
// dithers / posterises them: flat colour bands with a dark outline read as pixel art, lights and bloom do the "HD".
//   flame      – cone beams (Flamethrower, Dragon Breath): a rolling body of banded fire puffs, licking tongues, embers, smoke
//   fireBlast  – Fire Blast (だいもんじ): a fat stream to the target, white hit flash, a 大-shaped fire explosion, dark smoke
//   iceBeam    – ice beams: a hard crystalline lance with a serrated edge, orbiting shards, sparkles, a crystal burst at the tip
//   waterBeam  – water beams (Hydro Pump, Bubble Beam…): a pressurised jet with foam edges, streaks, spray and a splash crown
//   wave       – Surf: water floods the whole range (drawWorld), then waves are born at the caster and grow as they roll out
//   beamTick / waveTick – the particles, lights and floor marks that go with them (called from Moves.update)
"use strict";

const VFX = (() => {
  const TAU = Math.PI * 2;
  const hash = (n) => { const v = Math.sin(n * 127.1 + 311.7) * 43758.5453; return v - Math.floor(v); };
  const fract = (v) => v - Math.floor(v);
  const disc = (ctx, x, y, r) => { ctx.beginPath(); ctx.arc(x, y, Math.max(0.5, r), 0, TAU); ctx.fill(); };

  const FIRE = { dark: "#6e1206", edge: "#e2380a", mid: "#ff8a14", hot: "#ffd23a", core: "#fff8d0", smoke: "70,52,46", light: "#ffb347" };
  const DRAGON = { dark: "#2a0f6e", edge: "#5a2ad8", mid: "#8f6bff", hot: "#c8b8ff", core: "#ffffff", smoke: "52,44,80", light: "#a48cff" };
  // Fire moves use the caster's sprite colour when it is not the classic orange. Dragon cones stay purple.
  const firePal = (owner) => (typeof FX !== "undefined" && FX.flameOf(owner)) || FIRE;

  // Beam geometry shared by every beam: muzzle, direction, drawn length and the grow / shrink envelope.
  function geom(a) {
    const f = a.owner, s = a.spec, c = Math.cos(a.aim), sn = Math.sin(a.aim);
    const L = a.end ? a.end.d : s.len;
    return { f, s, c, sn, L, ox: f.x + c * 18, oy: f.y - f.height * 0.55 + sn * 10,
      grow: Math.min(1, a.t / 0.2), env: Math.min(1, a.t / 0.14) * (a.t > a.dur - 0.14 ? Math.max(0, (a.dur - a.t) / 0.14) : 1) };
  }

  // ---- Flamethrower -------------------------------------------------------------------------------------
  function flame(ctx, a, t) {
    const g = geom(a), { s, c, sn } = g, P = s.type === "dragon" ? DRAGON : firePal(g.f);
    const reach = g.L * g.grow, N = 34, puffs = [];
    for (let i = 0; i < N; i++) {
      const h1 = hash(i * 1.31 + a.seed), h2 = hash(i * 2.77 + a.seed * 3.1), h3 = hash(i * 5.13 + 1.7);
      const u = fract(t * (1.45 + h3 * 0.7) + h1);                    // 0 at the mouth → 1 at the tip
      const d = reach * Math.pow(u, 0.85), half = s.w * (0.16 + 0.95 * u);
      const lat = (h2 - 0.5) * 2 * half * 0.62 + Math.sin(t * 13 + i * 1.9) * half * 0.22;
      const lift = u * u * 20 + Math.sin(t * 9 + i) * 2;             // hot gas climbs as it travels
      const r = s.w * (0.2 + 0.6 * Math.pow(Math.sin(Math.PI * Math.min(1, u * 1.06)), 0.6)) * (0.75 + 0.5 * h3) * g.env;
      puffs.push({ x: g.ox + c * d - sn * lat, y: g.oy + sn * d + c * lat * 0.85 - lift, r, u, i });
    }
    // the stream billows where it stops
    if (g.grow >= 1) for (let i = 0; i < 7; i++) {
      const an = i / 7 * TAU + t * 2.2, rr = s.w * (0.5 + 0.25 * Math.sin(t * 14 + i * 2.3)) * g.env;
      puffs.push({ x: g.ox + c * reach + Math.cos(an) * s.w * 0.55, y: g.oy + sn * reach + Math.sin(an) * s.w * 0.38 - 16, r: rr, u: 0.72, i: 40 + i });
    }
    ctx.save();
    // smoke rolling off the far end (behind the fire)
    for (let i = 0; i < 6; i++) {
      const u = fract(t * 0.8 + hash(i * 7.7 + a.seed)), d = reach * (0.7 + 0.35 * hash(i * 3.1));
      ctx.fillStyle = `rgba(${P.smoke},${(0.42 * (1 - u) * g.env).toFixed(3)})`;
      disc(ctx, g.ox + c * d + (hash(i) - 0.5) * 30, g.oy + sn * d - 22 - u * 62, s.w * (0.32 + u * 0.5));
    }
    // colour bands, outline first: all puffs of one band are laid before the next, so they fuse into one flame body
    const band = (col, k, off, maxU) => {
      ctx.fillStyle = col;
      for (const p of puffs) if (p.u <= maxU) disc(ctx, p.x - c * p.r * off, p.y - sn * p.r * off + p.r * off * 0.25, p.r * k);
    };
    band(P.dark, 1.2, 0, 2); band(P.edge, 1.0, 0, 2);
    // tongues licking upward out of the body
    for (const p of puffs) {
      if (p.u < 0.2 || hash(p.i * 9.3) < 0.45) continue;
      const fl = 1.5 + 0.5 * Math.sin(t * 21 + p.i * 3.3), tip = p.x + Math.sin(t * 17 + p.i) * p.r * 0.35 - c * p.r * 0.3;
      ctx.fillStyle = P.dark; ctx.beginPath(); ctx.moveTo(p.x - p.r * 0.62, p.y - p.r * 0.45); ctx.lineTo(p.x + p.r * 0.62, p.y - p.r * 0.45); ctx.lineTo(tip, p.y - p.r * (fl + 0.22)); ctx.closePath(); ctx.fill();
      ctx.fillStyle = P.edge; ctx.beginPath(); ctx.moveTo(p.x - p.r * 0.5, p.y - p.r * 0.4); ctx.lineTo(p.x + p.r * 0.5, p.y - p.r * 0.4); ctx.lineTo(tip, p.y - p.r * fl); ctx.closePath(); ctx.fill();
      ctx.fillStyle = P.mid; ctx.beginPath(); ctx.moveTo(p.x - p.r * 0.28, p.y - p.r * 0.35); ctx.lineTo(p.x + p.r * 0.28, p.y - p.r * 0.35); ctx.lineTo(tip, p.y - p.r * fl * 0.72); ctx.closePath(); ctx.fill();
    }
    band(P.mid, 0.76, 0.12, 2); band(P.hot, 0.52, 0.2, 0.82); band(P.core, 0.28, 0.26, 0.5);
    // embers thrown past the stream
    for (let j = 0; j < 18; j++) {
      const e = fract(t * (0.9 + hash(j * 4.1) * 0.6) + hash(j * 9.1 + a.seed)), d = reach * (0.25 + 0.95 * e);
      const lat = (hash(j * 3.3) - 0.5) * s.w * 2.4 * e, sz = hash(j * 2.2) < 0.4 ? 4 : 2;
      if (Math.sin(t * 40 + j * 5) < -0.5) continue; // flicker
      ctx.fillStyle = e < 0.45 ? P.core : e < 0.75 ? P.hot : P.edge;
      ctx.fillRect(Math.round((g.ox + c * d - sn * lat) / 2) * 2, Math.round((g.oy + sn * d + c * lat * 0.8 - 10 - e * 54) / 2) * 2, sz, sz);
    }
    // white-hot muzzle
    ctx.fillStyle = P.hot; disc(ctx, g.ox, g.oy, s.w * 0.3 * g.env); ctx.fillStyle = P.core; disc(ctx, g.ox, g.oy, s.w * 0.18 * g.env);
    ctx.restore();
  }

  // ---- Fire Blast (だいもんじ) ---------------------------------------------------------------------------
  // [ユーザー指定 2026-11-15] 参考動画 sample6（Magmar の Fire Blast）のように：口から太い炎の帯が相手へ走り、白い閃光、
  // 火の爆発、黒い煙。そこへ技の名前どおり「大」の字の炎を一瞬立てる。strike（sky: "daimonji"）の空中レイヤーで描く。
  const BLAST_ARMS = [[0, -1, 0.72], [-1, 0.04, 0.88], [1, 0.04, 0.88], [-0.6, 1, 1.25], [0.6, 1, 1.25]]; // 大: 上（短い）・横棒の左右・左はらい・右はらい（長い）
  const easeOut = (k) => 1 - (1 - k) * (1 - k);
  function blastGeom(a) {
    const f = a.owner, ang = Math.atan2(a.y - f.y, a.x - f.x);
    const ox = f.x + Math.cos(ang) * 18, oy = f.y - f.height * 0.55 + Math.sin(ang) * 10, tx = a.x, ty = a.y - 24;
    const L = Math.hypot(tx - ox, ty - oy) || 1;
    const post = a.fired ? a.t - (a.firedAt || a.delay) : -1;
    return { ox, oy, tx, ty, L, c: (tx - ox) / L, sn: (ty - oy) / L, post,
      head: easeOut(Math.min(1, a.t / (a.delay * 0.86))), tail: post < 0 ? 0 : Math.min(1, post / 0.24) };
  }
  // fused fire body: every puff of one colour band is laid before the next band (dark outline first)
  function fireBody(ctx, puffs, P, c, sn) {
    const band = (col, k, off, maxHeat) => {
      ctx.fillStyle = col;
      for (const p of puffs) if (p.heat >= maxHeat) disc(ctx, p.x - c * p.r * off, p.y - sn * p.r * off + p.r * off * 0.25, p.r * k);
    };
    band(P.dark, 1.22, 0, 0); band(P.edge, 1.0, 0, 0); band(P.mid, 0.76, 0.12, 0.2); band(P.hot, 0.52, 0.2, 0.45); band(P.core, 0.28, 0.26, 0.7);
  }
  function fireBlast(ctx, a, t) {
    const g = blastGeom(a), P = firePal(a.owner), seed = a.seed || 0, { c, sn } = g;
    ctx.save();
    if (FXV.on) { // 3D: 炎の帯は口から着弾点までの線に沿わせ、爆発と「大」の字は着弾点に立てる
      const f = a.owner, ang = Math.atan2(a.y - f.y, a.x - f.x);
      FXV.seg(ctx, f.x + Math.cos(ang) * 18, f.y + Math.sin(ang) * 10, f.height * 0.55, a.x, a.y, 24);
    }
    // 1) the stream: a fat rolling body from the mouth, widening toward a fireball head; it burns off from the mouth's end after the hit
    if (g.tail < 1) {
      const puffs = [], span = g.head - g.tail;
      for (let i = 0; i < 30; i++) {
        const h1 = hash(i * 1.31 + seed), h2 = hash(i * 2.77 + seed * 3.1), h3 = hash(i * 5.13 + 1.7);
        const u = fract(t * (1.7 + h3 * 0.9) + h1), ua = g.tail + span * Math.pow(u, 0.8), d = g.L * ua;
        const half = 7 + 17 * ua, lat = (h2 - 0.5) * 2 * half * 0.6 + Math.sin(t * 14 + i * 1.9) * half * 0.22;
        const r = (6 + 12 * ua) * (0.75 + 0.5 * h3) * Math.pow(Math.sin(Math.PI * Math.min(1, u * 0.9 + 0.08)), 0.5) * (1 - g.tail * 0.5);
        puffs.push({ x: g.ox + c * d - sn * lat, y: g.oy + sn * d + c * lat * 0.85 - ua * 6, r, heat: 1 - u * 0.7, i });
      }
      // fireball head
      if (g.post < 0) for (let i = 0; i < 7; i++) {
        const an = i / 7 * TAU + t * 5, hr = 15 + 3 * Math.sin(t * 22 + i * 2.1);
        puffs.push({ x: g.ox + c * g.L * g.head + Math.cos(an) * 9, y: g.oy + sn * g.L * g.head + Math.sin(an) * 7 - g.head * 6, r: hr * (0.5 + 0.5 * g.head), heat: 1, i: 40 + i });
      }
      // tongues licking back along the stream
      for (const p of puffs) {
        if (hash(p.i * 9.3) < 0.5) continue;
        const fl = 1.5 + 0.5 * Math.sin(t * 21 + p.i * 3.3), tip = p.x - c * p.r * 0.9 + Math.sin(t * 17 + p.i) * p.r * 0.3;
        ctx.fillStyle = P.dark; ctx.beginPath(); ctx.moveTo(p.x - p.r * 0.62, p.y - p.r * 0.4); ctx.lineTo(p.x + p.r * 0.62, p.y - p.r * 0.4); ctx.lineTo(tip, p.y - p.r * (fl + 0.22)); ctx.closePath(); ctx.fill();
        ctx.fillStyle = P.edge; ctx.beginPath(); ctx.moveTo(p.x - p.r * 0.5, p.y - p.r * 0.35); ctx.lineTo(p.x + p.r * 0.5, p.y - p.r * 0.35); ctx.lineTo(tip, p.y - p.r * fl); ctx.closePath(); ctx.fill();
      }
      fireBody(ctx, puffs, P, c, sn);
      if (g.post < 0) { ctx.fillStyle = P.hot; disc(ctx, g.ox, g.oy, 9); ctx.fillStyle = P.core; disc(ctx, g.ox, g.oy, 5.5); } // white-hot muzzle
    }
    FXV.at(ctx, a.x, a.y);
    if (g.post >= 0) {
      const e = Math.min(1, g.post / 0.62), R = a.r * 0.5, cx = g.tx, cy = g.ty;
      // 2) dark smoke boiling up behind the fire
      for (let i = 0; i < 9; i++) {
        const h = hash(i * 3.7 + seed), an = h * TAU, k = easeOut(e), rr = R * (0.3 + 0.8 * hash(i * 6.1 + seed)) * k;
        const al = Math.min(1, e * 3) * (1 - Math.pow(e, 3)) * 0.78;
        ctx.fillStyle = `rgba(${P.smokeA || "52,42,44"},${al.toFixed(3)})`; disc(ctx, cx + Math.cos(an) * rr, cy + Math.sin(an) * rr * 0.55 - 14 - e * 46 * (0.5 + h), (10 + 16 * k) * (0.7 + 0.6 * h));
        ctx.fillStyle = `rgba(${P.smokeB || "96,84,84"},${(al * 0.7).toFixed(3)})`; disc(ctx, cx + Math.cos(an) * rr - 3, cy + Math.sin(an) * rr * 0.55 - 18 - e * 46 * (0.5 + h), (10 + 16 * k) * (0.7 + 0.6 * h) * 0.6);
      }
      // 3) the explosion: a rolling ball of fire that swells, breaks up and cools
      const puffs = [], cool = Math.max(0, 1 - e * 1.25);
      if (cool > 0) for (let i = 0; i < 20; i++) {
        const h1 = hash(i * 2.3 + seed), h2 = hash(i * 4.9 + seed * 1.7), an = h1 * TAU + e * (h2 - 0.5) * 2.4;
        const rr = R * 0.72 * Math.pow(h2, 0.6) * (0.25 + 0.95 * easeOut(Math.min(1, g.post / 0.3)));
        puffs.push({ x: cx + Math.cos(an) * rr, y: cy + Math.sin(an) * rr * 0.62 - e * 16, r: (8 + 13 * (1 - h2 * 0.5)) * Math.pow(cool, 0.55) * (0.8 + 0.4 * Math.sin(t * 19 + i * 2.7)), heat: cool * (1.1 - h2 * 0.6), i });
      }
      // 4) 大: five tongues of flame thrown out from the heart of the blast
      const ak = Math.min(1, g.post / 0.16), afade = Math.max(0, 1 - Math.max(0, g.post - 0.3) / 0.26);
      if (afade > 0) for (let j = 0; j < 5; j++) {
        const [ax, ay, al] = BLAST_ARMS[j], n = Math.hypot(ax, ay), len = a.r * 0.66 * al * easeOut(ak), bend = j > 2 ? (j === 3 ? -1 : 1) : 0; // the two legs sweep outward like brush strokes
        for (let q = 0; q < 12; q++) {
          const u = (q + 0.5) / 12, wob = Math.sin(t * 16 + j * 2.1 + q * 1.3) * 3.2 * u;
          puffs.push({ x: cx + ax / n * len * u - ay / n * wob + bend * u * u * a.r * 0.165, y: cy - a.r * 0.2 + ay / n * len * u * 0.86 + ax / n * wob, r: (19 - 5.5 * u) * afade * (0.85 + 0.15 * Math.sin(t * 24 + q + j)), heat: afade * (1.15 - u * 0.55), i: 80 + j * 12 + q });
        }
      }
      // little flames standing up off the blast
      for (const p of puffs) {
        if (p.r < 5 || hash(p.i * 7.7 + 2.1) < (p.i >= 80 ? 0.35 : 0.55)) continue; // 大 の腕は炎の舌を多く・高く
        const fl = (p.i >= 80 ? 2.1 : 1.45) + 0.5 * Math.sin(t * 20 + p.i * 2.9), tip = p.x + Math.sin(t * 15 + p.i) * p.r * 0.4;
        ctx.fillStyle = P.dark; ctx.beginPath(); ctx.moveTo(p.x - p.r * 0.62, p.y - p.r * 0.4); ctx.lineTo(p.x + p.r * 0.62, p.y - p.r * 0.4); ctx.lineTo(tip, p.y - p.r * (fl + 0.22)); ctx.closePath(); ctx.fill();
        ctx.fillStyle = P.edge; ctx.beginPath(); ctx.moveTo(p.x - p.r * 0.5, p.y - p.r * 0.35); ctx.lineTo(p.x + p.r * 0.5, p.y - p.r * 0.35); ctx.lineTo(tip, p.y - p.r * fl); ctx.closePath(); ctx.fill();
      }
      fireBody(ctx, puffs, P, 0, -0.6);
      // 5) the white hit flash: a jagged star for the first frames
      if (g.post < 0.085) {
        const k = 1 - g.post / 0.085, rr = 26 + 22 * (1 - k);
        ctx.fillStyle = "#ffffff"; ctx.beginPath();
        for (let i = 0; i < 16; i++) { const an = i / 16 * TAU + seed, r2 = rr * (i % 2 ? 0.5 : 1 + 0.35 * hash(i + seed)); ctx.lineTo(cx + Math.cos(an) * r2, cy + Math.sin(an) * r2 * 0.8); }
        ctx.closePath(); ctx.fill();
      }
      // 6) sparks: hard yellow crosses that twinkle as they fly out
      for (let j = 0; j < 14; j++) {
        const h = hash(j * 5.3 + seed), an = h * TAU, d = (0.25 + 0.95 * hash(j * 8.1 + seed)) * a.r * 0.9 * easeOut(e);
        if (e > 0.9 || Math.sin(t * 34 + j * 4.1) < -0.2) continue;
        const x = Math.round((cx + Math.cos(an) * d) / 2) * 2, y = Math.round((cy + Math.sin(an) * d * 0.62 - 10 - e * 26 * h) / 2) * 2, r = e < 0.5 ? 4 : 2;
        ctx.fillStyle = j % 3 ? P.hot : P.core; ctx.fillRect(x - r, y - 1, r * 2, 2); ctx.fillRect(x - 1, y - r, 2, r * 2);
      }
    }
    ctx.restore();
  }
  // lights, embers, smoke and the jolt that go with it (Moves.update, every frame of the strike)
  function fireBlastTick(B, a, dt) {
    const fx = B.fx, g = blastGeom(a), P = firePal(a.owner), fl = 0.8 + 0.4 * Math.random();
    if (!a.seed) a.seed = 1 + Math.random() * 97;
    if (g.tail < 1) {
      for (const u of [0.3, 0.65, 1]) { const ua = g.tail + (g.head - g.tail) * u; fx.light(g.ox + g.c * g.L * ua, g.oy + g.sn * g.L * ua + 24, 70 + 50 * ua, P.light, 0.26 * fl, 0.1); }
      if (Math.random() < dt * 46) { const ua = g.tail + (g.head - g.tail) * U.rand(0.2, 1); fx.add({ x: g.ox + g.c * g.L * ua + U.rand(-12, 12), y: g.oy + g.sn * g.L * ua + 24 + U.rand(-6, 6), z: 24 + U.rand(0, 16), vx: g.c * U.rand(40, 160), vy: g.sn * U.rand(40, 160), vz: U.rand(60, 190), grav: -60, life: U.rand(0.3, 0.6), size: U.pick([2, 2, 4]), color: U.pick([P.core, P.hot, P.mid]), shape: "pixel", add: true }); }
    }
    if (a.fired && !a.boomed) {
      a.boomed = true;
      fx.flashScreen(0.28, P.flash || "#fff0c0"); fx.shakeCam(7, 0.3);
      fx.light(a.x, a.y, 240, P.glow || "#ffd070", 0.85, 0.16); fx.light(a.x, a.y, 180, P.light, 0.6, 0.5);
      for (let i = 0; i < 26; i++) { const an = U.rand(TAU), sp = U.rand(90, 330); fx.add({ x: a.x, y: a.y, z: 22, vx: Math.cos(an) * sp, vy: Math.sin(an) * sp * 0.6, vz: U.rand(90, 340), grav: 520, life: U.rand(0.35, 0.8), size: U.pick([2, 2, 4]), color: U.pick([P.core, P.hot, P.mid, P.edge]), shape: "pixel", add: true, drag: 0.6 }); }
      for (let i = 0; i < 7; i++) { const an = U.rand(TAU); fx.add({ x: a.x + Math.cos(an) * U.rand(10, a.r * 0.5), y: a.y + Math.sin(an) * U.rand(6, a.r * 0.3), z: 26, vx: Math.cos(an) * 26, vy: Math.sin(an) * 12, vz: U.rand(40, 95), life: U.rand(0.8, 1.3), size: U.rand(14, 24), color: P.smokeHex || "#3a2e30", shape: "smoke", alpha: 0.5, grow: 1.7 }); }
    }
    if (g.post > 0 && g.post < 0.5) {
      fx.light(a.x, a.y, 150, P.light, 0.3 * fl * (1 - g.post / 0.5), 0.1);
      if (Math.random() < dt * 30) fx.add({ x: a.x + U.rand(-a.r, a.r) * 0.5, y: a.y + U.rand(-a.r, a.r) * 0.3, z: 10, vx: U.rand(-20, 20), vy: 0, vz: U.rand(80, 220), grav: -40, life: U.rand(0.4, 0.8), size: 2, color: U.pick([P.hot, P.mid]), shape: "pixel", add: true });
    }
  }

  // ---- Ice Beam ---------------------------------------------------------------------------------------
  const rhomb = (ctx, x, y, w, h, rot) => { ctx.save(); ctx.translate(x, y); ctx.rotate(rot); ctx.beginPath(); ctx.moveTo(0, -h); ctx.lineTo(w, 0); ctx.lineTo(0, h); ctx.lineTo(-w, 0); ctx.closePath(); ctx.fill(); ctx.restore(); };
  function iceBeam(ctx, a, t) {
    const g = geom(a), { s } = g, len = g.L * g.grow, W = s.w * 1.55 * (0.75 + 0.25 * g.env) * g.env; // drawn fatter than the hit line: it has to read as a lance of ice
    if (len < 4 || W < 1) return;
    ctx.save(); ctx.translate(g.ox, g.oy); ctx.rotate(a.aim);
    // cold aura
    ctx.globalCompositeOperation = "lighter"; ctx.fillStyle = "rgba(90,170,255,0.2)"; ctx.fillRect(0, -W * 1.5, len, W * 3);
    ctx.globalCompositeOperation = "source-over";
    // orbiting shards: the back half of the helix goes under the lance, the front half over it
    const shards = [];
    for (let st = 0; st < 2; st++) for (let j = 0; j < 9; j++) {
      const u = fract(j / 9 + t * 0.85 + st * 0.055), ph = u * 17 - t * 15 + st * Math.PI;
      shards.push({ x: u * len, y: Math.sin(ph) * W * 1.15, z: Math.cos(ph), rot: ph * 0.6, k: 0.7 + 0.6 * hash(j * 3.7 + st) });
    }
    const drawShard = (q) => { const sz = (7 + 3.5 * q.z) * q.k; ctx.fillStyle = "#1c58b0"; rhomb(ctx, q.x, q.y, sz * 0.62 + 2, sz + 2, q.rot); ctx.fillStyle = q.z > 0 ? "#ffffff" : "#8fd4ff"; rhomb(ctx, q.x, q.y, sz * 0.62, sz, q.rot); };
    for (const q of shards) if (q.z <= 0) drawShard(q);
    // the lance: hard bands with a serrated, scrolling crystal edge
    const band = (col, hw, amp, ph) => {
      const step = 12, off = (t * 300 + ph) % (step * 2);
      ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(0, 0);
      for (let x = 0; x <= len; x += step) ctx.lineTo(x, -hw - (Math.floor((x + off) / step) % 2 ? amp : 0));
      ctx.lineTo(len + hw * 1.6, 0);
      for (let x = len; x >= 0; x -= step) ctx.lineTo(x, hw + (Math.floor((x + off + step) / step) % 2 ? amp : 0));
      ctx.closePath(); ctx.fill();
    };
    band("#16479c", W * 0.66, W * 0.2, 0); band("#3f9cf0", W * 0.54, W * 0.16, 0); band("#a6e6ff", W * 0.34, W * 0.08, 6); band("#ffffff", W * 0.16, 0, 0);
    // bright chevrons racing down the core
    ctx.fillStyle = "#ffffff";
    for (let j = 0; j < 5; j++) { const x = fract(t * 2.4 + j / 5) * len, w = W * 0.42; ctx.beginPath(); ctx.moveTo(x, -w); ctx.lineTo(x + w * 1.1, 0); ctx.lineTo(x, w); ctx.lineTo(x + w * 0.5, 0); ctx.closePath(); ctx.fill(); }
    for (const q of shards) if (q.z > 0) drawShard(q);
    // twinkling frost
    ctx.globalCompositeOperation = "lighter"; ctx.fillStyle = "#e8fbff";
    for (let j = 0; j < 12; j++) {
      const tw = Math.sin(t * 16 + j * 2.4); if (tw < 0.1) continue;
      const x = hash(j * 5.3 + Math.floor(t * 3 + j * 0.37)) * len, y = (hash(j * 8.9 + Math.floor(t * 3 + j * 0.37)) - 0.5) * W * 3.4, r = 2 + tw * 5;
      ctx.fillRect(x - r, y - 1, r * 2, 2); ctx.fillRect(x - 1, y - r, 2, r * 2);
    }
    ctx.restore();
    // crystal burst where the beam ends
    const ex = g.ox + g.c * len, ey = g.oy + g.sn * len;
    ctx.save(); ctx.translate(ex, ey);
    for (const [col, kk] of [["#16479c", 1.12], ["#8fd4ff", 1], ["#ffffff", 0.55]]) {
      ctx.fillStyle = col;
      for (let j = 0; j < 8; j++) {
        const an = j / 8 * TAU + 0.2 + Math.sin(t * 9 + j) * 0.08, l = W * (1.5 + 1.1 * hash(Math.floor(t * 16) * 3.1 + j * 7.7)) * kk, b = W * 0.34 * kk;
        ctx.beginPath(); ctx.moveTo(Math.cos(an) * l, Math.sin(an) * l * 0.8); ctx.lineTo(Math.cos(an + 1.57) * b, Math.sin(an + 1.57) * b * 0.8); ctx.lineTo(Math.cos(an - 1.57) * b, Math.sin(an - 1.57) * b * 0.8); ctx.closePath(); ctx.fill();
      }
    }
    ctx.fillStyle = "#ffffff"; disc(ctx, 0, 0, W * 0.5);
    ctx.restore();
    // frost ring at the mouth
    ctx.save(); ctx.strokeStyle = "#dff6ff"; ctx.lineWidth = 3; ctx.beginPath(); ctx.ellipse(g.ox, g.oy, W * (0.9 + 0.3 * fract(t * 3)), W * (1.3 + 0.5 * fract(t * 3)), a.aim, 0, TAU); ctx.stroke();
    ctx.fillStyle = "#ffffff"; disc(ctx, g.ox, g.oy, W * 0.42); ctx.restore();
  }

  // ---- water jets -----------------------------------------------------------------------------------------
  function waterBeam(ctx, a, t) {
    const g = geom(a), { s } = g, len = g.L * g.grow, W = s.w * (0.8 + 0.2 * g.env) * g.env;
    if (len < 4 || W < 1) return;
    ctx.save(); ctx.translate(g.ox, g.oy); ctx.rotate(a.aim);
    const band = (col, hw, wob, ph, dy = 0) => {
      const step = 14; ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(0, dy - hw * 0.5);
      for (let x = 0; x <= len; x += step) ctx.lineTo(x, dy - hw * (1 + wob * Math.sin(x * 0.085 - t * 27 + ph)) * Math.min(1, 0.5 + x / 50));
      ctx.lineTo(len + hw, dy);
      for (let x = len; x >= 0; x -= step) ctx.lineTo(x, dy + hw * (1 + wob * Math.sin(x * 0.085 - t * 27 + ph + 2.1)) * Math.min(1, 0.5 + x / 50));
      ctx.closePath(); ctx.fill();
    };
    band("#0e3690", W * 0.7, 0.2, 0); band("#2478e0", W * 0.56, 0.18, 0.4); band("#66c2ff", W * 0.3, 0.3, 1.3, -W * 0.14); band("#d8f6ff", W * 0.1, 0.5, 2.2, -W * 0.22);
    // pressure streaks
    ctx.fillStyle = "#eefcff";
    for (let j = 0; j < 12; j++) { const x = fract(t * (2.6 + hash(j) * 1.2) + hash(j * 3.9)) * len, y = (hash(j * 7.1) - 0.5) * W * 0.8; ctx.fillRect(x, y - 1.5, 12 + 14 * hash(j * 2.3), 3); }
    // foam torn off both edges, and spray beyond it
    for (let j = 0; j < 18; j++) {
      const u = fract(t * (1.7 + hash(j * 1.9) * 0.8) + hash(j * 6.3)), x = u * len, side = j % 2 ? 1 : -1, y = side * W * (0.56 + 0.1 * Math.sin(t * 20 + j));
      ctx.fillStyle = "#16479c"; disc(ctx, x, y, W * (0.17 + 0.1 * hash(j * 4.4)) + 2);
      ctx.fillStyle = "#ffffff"; disc(ctx, x, y, W * (0.17 + 0.1 * hash(j * 4.4)));
      const e = fract(u * 3 + j * 0.31); ctx.fillStyle = e < 0.6 ? "#ffffff" : "#9fdcff";
      ctx.fillRect(Math.round((x - 10 * e) / 2) * 2, Math.round((side * W * (0.8 + e * 1.1)) / 2) * 2, 3, 3);
    }
    ctx.restore();
    // splash crown at the end
    const ex = g.ox + g.c * len, ey = g.oy + g.sn * len;
    for (const [col, kk] of [["#16479c", 1.14], ["#58b8ff", 1], ["#ffffff", 0.7]]) {
      ctx.fillStyle = col;
      for (let j = 0; j < 9; j++) { const an = j / 9 * TAU + t * 1.3, pul = 0.75 + 0.35 * Math.sin(t * 15 + j * 1.7); disc(ctx, ex + Math.cos(an) * W * 0.95 * pul, ey + Math.sin(an) * W * 0.62 * pul - 4, W * 0.42 * kk); }
    }
    ctx.fillStyle = "#ffffff"; disc(ctx, ex, ey - 3, W * 0.6); disc(ctx, g.ox, g.oy, W * 0.4);
  }

  // ---- water, shaded per pixel ------------------------------------------------------------------------------
  // [ユーザー指定 2026-11-15] 「うずしおがとてつもなくダサい。レトロだけど水の質感を綺麗に表現できている最高品質の HD-2D で。
  // なみのりも同じレベルに」。線や円を重ねる描き方をやめ、水面を 1 ドットずつ計算する:
  //   高さ場（うねり・波・渦の壁）を奥から手前へ描いて立体に見せ、深さの濃淡・光の網目（コースティクス）・流れの筋・泡・きらめきを
  //   8 色の水色パレットへベイヤーディザで落とす。world キャンバス（モンスターの下、ポストのライトとブルームが乗る）に置く。
  const SQ = 0.55;
  // (the arena light and bloom lift everything about a tone, so the ramp is authored a little deeper than it should read)
  const PAL = [[12, 44, 124], [22, 70, 160], [34, 98, 194], [50, 128, 220], [76, 160, 240], [112, 190, 250], [172, 222, 255], [255, 255, 255]];
  const PAL32 = PAL.map(([r, g, b]) => (255 << 24 | b << 16 | g << 8 | r) >>> 0);
  const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map((v) => v / 16);
  const sstep = (a, b, x) => { const u = Math.min(1, Math.max(0, (x - a) / (b - a))); return u * u * (3 - 2 * u); };
  const h2 = (x, y) => { const v = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return v - Math.floor(v); };
  function vnoise(x, y) { // smooth value noise, 0‥1
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi, u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    const a = h2(xi, yi), b = h2(xi + 1, yi), c = h2(xi, yi + 1), d = h2(xi + 1, yi + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  }
  let fieldCv = null, fieldCtx = null, oH = 0, oT = 0, oW = 0; // shader outputs: height (art px), tone (0‥7), wall tone
  // Renders a height field around (cx, cy) [art px on ctx]. fn(dx, dy, r, th) fills oH / oT / oW and returns false outside the water.
  // Columns are painted far → near with their vertical faces, so tall water hides what is behind it.
  function field(ctx, cx, cy, Rp, maxH, fade, fn) {
    const W = (Math.ceil(Rp * 1.1) * 2 + 4) | 0, GH = (Math.ceil(Rp * 1.1 * SQ) * 2 + 4) | 0, TM = Math.ceil(maxH) + 3, HT = GH + TM;
    if (!fieldCv) { fieldCv = document.createElement("canvas"); fieldCtx = fieldCv.getContext("2d"); }
    if (fieldCv.width !== W || fieldCv.height !== HT) { fieldCv.width = W; fieldCv.height = HT; }
    const img = fieldCtx.createImageData(W, HT), D = new Uint32Array(img.data.buffer);
    const put = (x, y, tone) => {
      if (y < 0 || y >= HT) return;
      const bz = BAYER[(x & 3) + ((y & 3) << 2)];
      if (fade < 1 && bz >= fade) return;
      const k = Math.round(tone + (bz - 0.47) * 1.15);
      D[y * W + x] = PAL32[k < 0 ? 0 : k > 7 ? 7 : k];
    };
    for (let x = 0; x < W; x++) {
      const dx = x - W / 2 + 0.5, streak = h2(x * 3.1, 7.7);
      for (let gy = 0; gy < GH; gy++) {
        const dy = (gy - GH / 2 + 0.5) / SQ, r = Math.sqrt(dx * dx + dy * dy);
        if (r > Rp * 1.09 || !fn(dx, dy, r, Math.atan2(dy, dx))) continue;
        const h = Math.max(0, Math.round(oH)), ys = gy + TM - h;
        for (let k = 1; k <= h; k++) { // the face toward the camera: light at the lip, deeper below, streaked, foam where it meets the pool
          const u = k / h;
          put(x, ys + k, h > 4 && k > h - 2 ? 6.6 : oW - 3.1 * u + (streak > 0.74 ? 1.7 * (1 - u * 0.6) : streak < 0.18 ? -0.8 : 0));
        }
        put(x, ys, oT);
      }
    }
    fieldCtx.putImageData(img, 0, 0);
    ctx.drawImage(fieldCv, Math.round(cx - W / 2), Math.round(cy - GH / 2 - TM));
  }
  // light net on the surface: thin bright lines where two drifting sine sheets cross, soft shadow between
  function caustic(dx, dy, t) {
    const c = Math.sin(dx * 0.33 + Math.sin(dy * 0.21 + t * 1.3) * 1.8 + t * 0.8) * Math.sin(dy * 0.47 + Math.sin(dx * 0.17 - t * 1.1) * 1.9 - t * 0.6);
    return Math.abs(c) < 0.075 ? 1.7 : c > 0.5 ? -0.55 : 0;
  }
  function bodyRings(ctx, B, x, y, R, ix, iy, t) { // foam where the water laps around every body standing in it
    const K = Arenas.K;
    for (const f of B.active) {
      if (!f || !f.active || f.ko || f.flies || U.dist(f.x, f.y, x, y) > R) continue;
      const fr = f.r / K * (1.1 + 0.12 * Math.sin(t * 6 + f.side * 2));
      ctx.save(); ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1; ctx.setLineDash([3, 2]); ctx.lineDashOffset = t * 8;
      ctx.beginPath(); ctx.ellipse(f.x / K - ix, f.y / K - iy, fr, fr * 0.5, 0, 0, TAU); ctx.stroke(); ctx.restore();
    }
  }

  // ---- Surf ---------------------------------------------------------------------------------------------
  //   1. 足元から水が噴き上がり、範囲いっぱいに広がる  2. 中心から波が次々に生まれ、外へ進むほど高くなる（後の波ほど大きい）
  //   3. 波は縁で砕けて水しぶきになり、水が引く
  const CRESTS = [{ at: 0.14, H: 64 }, { at: 0.27, H: 100 }, { at: 0.40, H: 146 }, { at: 0.53, H: 205 }], TRAVEL = 0.4;
  function waveState(a) {
    const life = Math.min(1, a.t / a.dur);
    const crests = CRESTS.map((c, i) => {
      const p = (life - c.at) / TRAVEL;
      if (p <= 0 || p >= 1.12) return null;
      const grow = 0.16 + 0.84 * Math.pow(Math.min(1, p), 1.25);                       // small at the centre, towering at the rim
      const crash = p > 0.9 ? Math.max(0, 1 - (p - 0.9) / 0.2) : 1;
      return { i, p, r: a.r0 + (a.r1 - a.r0) * Math.pow(Math.min(1, p), 0.92), h: c.H * grow * crash };
    }).filter(Boolean);
    return { life, crests, flood: a.r1 * 1.04 * U.easeOut(Math.min(1, life / 0.2)), drain: 1 - U.clamp((life - 0.86) / 0.14, 0, 1),
      spout: life < 0.3 ? Math.sin(Math.PI * life / 0.3) : 0 };
  }
  // Radius the damage has reached: the first wave. Nothing is hit before it leaves the caster.
  function waveHitR(a) { const p = (Math.min(1, a.t / a.dur) - CRESTS[0].at) / TRAVEL; return p <= 0 ? 0 : a.r0 + (a.r1 - a.r0) * Math.pow(Math.min(1, p), 0.92); }

  function surfWorld(ctx, B, a, ix, iy, t) {
    const K = Arenas.K, w = waveState(a), Rp = w.flood / K;
    if (Rp < 3 || w.drain <= 0) return;
    const cr = w.crests.map((c) => ({ r: c.r / K, A: c.h / K * 0.5, i: c.i })), nC = cr.length, spH = 26 * w.spout;
    let maxH = spH; for (const c of cr) maxH = Math.max(maxH, c.A);
    field(ctx, a.x / K - ix, a.y / K - iy, Rp, maxH + 2, w.drain, (dx, dy, r, th) => {
      const edge = Rp * (1 + 0.03 * Math.sin(th * 7 + t * 3.1) + 0.02 * Math.sin(th * 13 - t * 4.3));
      if (r > edge) return false;
      const d = edge - r, lit = -(Math.cos(th) * 0.5 + Math.sin(th) * 0.85); // > 0 where the outward face looks at the light (top-left)
      let tone = 4.9 - 1.3 * Math.min(1, d / 30) + caustic(dx, dy, t) * 0.9 + 0.45 * Math.sin(r * 0.34 - t * 5.5), h = 0;
      for (let j = 0; j < nC; j++) {
        const c = cr[j], s = r - c.r, wf = 3.5 + c.A * 0.1, wb = 9 + c.A * 0.32, g = Math.exp(-(s > 0 ? (s / wf) * (s / wf) : (s / wb) * (s / wb)));
        if (g < 0.01) continue;
        h += c.A * g;
        tone += g * Math.min(2.2, c.A / 9) * (s > 0 ? 1.5 * lit : -0.9 * lit - 0.3);       // lit face / shaded back
        const fz = vnoise(th * c.r * 0.22 + c.i * 9, t * 3 + c.i);
        if (c.A > 2 && s > -(2.6 + c.A * 0.07) && s < 1.8 + c.A * 0.03 && fz > 0.16) tone = 7;                           // the boiling lip
        else if (s < -2 && s > -wb * 1.5 && vnoise(th * c.r * 0.3 + 5, s * 0.35 + t * 7) > 0.66 - 0.22 * g) tone = Math.max(tone, 6.3); // foam dragged behind it
      }
      if (spH > 0.5) { const g = Math.exp(-(r / 11) * (r / 11)); h += spH * g * (0.8 + 0.3 * Math.sin(th * 6 + t * 20)); if (g > 0.22) tone = 6.2 + g; }
      const fn = vnoise(th * 26, t * 2.2);
      if (d < 1.4 + 1.9 * fn) tone = 7; else if (d < 5.5 && fn > 0.55) tone = 6.2;         // lapping foam at the shore
      else if (h2(Math.floor(dx * 0.5) + Math.floor(t * 4) * 3.1, Math.floor(dy * 0.5)) > 0.994) tone = 7; // glints
      oH = h; oT = tone; oW = 5.3;
      return true;
    });
    bodyRings(ctx, B, a.x, a.y, w.flood, ix, iy, t);
  }

  // ---- Whirlpool (うずしお) ----------------------------------------------------------------------------------
  // 水がキャラクターの中心へ、らせん状に渦を巻いて集まる。外側は遠心力で盛り上がり、中心は暗い目へ落ち込む。
  function whirlWorld(ctx, B, a, ix, iy, t) {
    const K = Arenas.K, Rp = a.R / K, grow = U.clamp(a.R / a.spec.r1, 0, 1);
    if (!(Rp > 3)) return;
    const Hm = 10 * grow;
    field(ctx, a.x / K - ix, a.y / K - iy, Rp, Hm + 6, 1, (dx, dy, r, th) => {
      const edge = 1 + 0.035 * Math.sin(th * 6 - t * 4) + 0.02 * Math.sin(th * 11 + t * 2.3), rho = r / Rp;
      if (rho > edge) return false;
      const lr = Math.log(Math.max(rho, 0.03));
      const phi = 3 * th + 7.2 * lr + t * 4.2 + 1.5 * t / (0.22 + rho);                  // log spirals; the centre turns faster
      const ridge = Math.max(0, Math.sin(phi)), r3 = ridge * ridge * ridge, mid = sstep(0.1, 0.42, rho);
      const flow = vnoise(phi * 1.25, lr * 9 - t * 3.2);                                   // streaks stretched along the current
      const bowl = Hm * sstep(0.08, 0.82, rho) * (1 - sstep(0.9, 1, rho / edge));        // piled up outside, sunk at the eye
      let tone = 0.4 + 3.3 * sstep(0.04, 0.96, rho) + 1.3 * (ridge - 0.5) + (flow - 0.5) * 1.5 + caustic(dx, dy, t) * 0.4;
      tone -= 0.6 * (Math.cos(th) * 0.5 + Math.sin(th) * 0.85) * sstep(0.2, 0.9, rho);    // the far-left wall catches the light
      if (r3 > 0.9 - 0.22 * rho && flow > 0.42) tone = 7;                                 // foam riding every spiral ridge
      else if (Math.sin(2 * phi + 1.3) > 0.975 && flow > 0.5) tone = Math.max(tone, 5.6); // finer threads between them
      if (rho < 0.075) tone = 0.1 + flow * 1.2;                                            // the eye
      else if (rho < 0.115 && flow > 0.3) tone = 6.6;                                      // froth ringing it
      const d = (edge - rho) * Rp, fz = vnoise(th * 24 - t * 2.5, t * 2);
      if (d < 1.4 + 1.8 * fz) tone = 7; else if (d < 5 && fz > 0.55) tone = 6.2;
      else if (h2(Math.floor(dx * 0.5) + Math.floor(t * 5) * 3.1, Math.floor(dy * 0.5)) > 0.995) tone = 7;
      oH = bowl + 3.4 * r3 * mid * grow; oT = tone; oW = 3.0;
      return true;
    });
    bodyRings(ctx, B, a.x, a.y, a.R, ix, iy, t);
  }
  // Both live on the world canvas, under the monsters.
  function drawWorld(ctx, B, ix, iy, t) {
    for (const a of B.attacks) {
      if (a.kind === "spiral" && a.spec.pull) whirlWorld(ctx, B, a, ix, iy, t);
      else if (a.kind === "aoe" && a.spec.wave) surfWorld(ctx, B, a, ix, iy, t);
    }
  }
  function waveGround() {} function wave() {} function whirlAir() {} // (nothing on the fx layer any more; kept for the callers)
  function whirlTick(B, a, dt) {
    const fx = B.fx, R = a.R;
    // no light of its own: the water is already bright, and any added light bleaches its blues to white
    for (let i = 0; i < Math.round(dt * 120); i++) { // drops torn from the rim, flung along the current and inward
      const an = U.rand(TAU), rr = R * U.rand(0.6, 1);
      fx.add({ x: a.x + Math.cos(an) * rr, y: a.y + Math.sin(an) * rr * SQ, z: U.rand(8, 26), vx: Math.sin(an) * 170 - Math.cos(an) * 150, vy: (-Math.cos(an) * 170 - Math.sin(an) * 150) * SQ, vz: U.rand(40, 170), grav: 560, life: U.rand(0.3, 0.6), size: U.rand(3, 6), color: U.pick(["#ffffff", "#e6f7ff", "#b4e4ff"]), shape: "drop", drag: 0.4 });
    }
    if (Math.random() < dt * 10) fx.add({ x: a.x + U.rand(-16, 16), y: a.y + U.rand(-8, 8), z: 6, vx: 0, vy: 0, vz: U.rand(20, 50), life: 0.8, size: U.rand(12, 20), color: "#f2fbff", shape: "smoke", alpha: 0.26, grow: 1.6 }); // mist over the eye
    if (B.arena.water && typeof Water !== "undefined" && Math.random() < dt * 10) { const an = U.rand(TAU); Water.ripple(B.arena, a.x + Math.cos(an) * R * 0.8, a.y + Math.sin(an) * R * 0.8 * SQ, 8, 50, 0.9, { foam: true }); }
  }

  // ---- particles / lights / floor marks ------------------------------------------------------------------------
  // Returns true when it handled the beam's continuous effects (the caller then skips the generic ones).
  function beamTick(B, a, end, ox, oy, dt) {
    const s = a.spec, fx = B.fx, f = a.owner, c = Math.cos(a.aim), sn = Math.sin(a.aim), hz = f.height * 0.55;
    const kind = s.cone ? "flame" : s.type === "ice" ? "ice" : s.type === "water" ? "water" : null;
    if (!kind) return false;
    const reach = end.d * Math.min(1, a.t / 0.2), fl = 0.8 + 0.4 * Math.random();
    if (kind === "flame") {
      const dragon = s.type === "dragon", P = dragon ? DRAGON : firePal(f);
      for (const u of [0.25, 0.6, 1]) fx.light(ox + c * reach * u, oy + sn * reach * u, 110 + 70 * u, P.light, 0.42 * fl, 0.1);
      fx.light(ox, oy, 80, P.mouth || "#fff2c0", 0.35, 0.1);
      if (Math.random() < dt * 40) { const d = U.rand(0.3, 1) * reach; fx.add({ x: ox + c * d + U.rand(-14, 14), y: oy + sn * d + U.rand(-8, 8), z: hz + U.rand(0, 20), vx: c * U.rand(40, 140), vy: sn * U.rand(40, 140), vz: U.rand(60, 180), grav: -80, life: U.rand(0.35, 0.7), size: U.pick([2, 2, 4]), color: dragon ? U.pick(["#fff8d0", "#ffd23a", "#ff8a14"]) : U.pick([P.core, P.hot, P.mid]), shape: "pixel", add: true, drag: 1.2 }); }
      if (Math.random() < dt * 9) fx.add({ x: ox + c * reach + U.rand(-20, 20), y: oy + sn * reach + U.rand(-10, 10), z: hz + 18, vx: U.rand(-20, 20), vy: U.rand(-8, 8), vz: U.rand(40, 80), life: U.rand(0.7, 1.1), size: U.rand(12, 20), color: P.tipSmoke || "#4a3a36", shape: "smoke", alpha: 0.3, grow: 1.8 });
    } else if (kind === "ice") {
      for (const u of [0.3, 0.7, 1]) fx.light(ox + c * reach * u, oy + sn * reach * u, 100 + 50 * u, "#8fd8ff", 0.4 * fl, 0.1);
      if (Math.random() < dt * 26) { const d = U.rand(0.1, 1) * reach, o = U.rand(-1, 1) * s.w; fx.add({ x: ox + c * d - sn * o, y: oy + sn * d + c * o * 0.6, z: hz + U.rand(-6, 10), vx: -sn * o * 2 + U.rand(-20, 20), vy: c * o * 1.2, vz: U.rand(20, 90), grav: 300, life: U.rand(0.35, 0.7), size: U.rand(4, 8), color: U.pick(["#ffffff", "#dcf8ff", "#8fd8ff"]), shape: "shard", add: true, spin: U.rand(-9, 9), rot: U.rand(TAU) }); }
      if (Math.random() < dt * 14) fx.add({ x: ox + c * reach + U.rand(-16, 16), y: oy + sn * reach + U.rand(-8, 8), z: hz, vx: U.rand(-40, 40), vy: U.rand(-20, 20), vz: U.rand(10, 50), life: 0.8, size: U.rand(12, 20), color: "#eef9ff", shape: "smoke", alpha: 0.32, grow: 1.7 });
    } else {
      for (const u of [0.4, 1]) fx.light(ox + c * reach * u, oy + sn * reach * u, 100 + 40 * u, "#6ab8ff", 0.32 * fl, 0.1);
      if (Math.random() < dt * 34) { const d = U.rand(0.1, 1) * reach, o = U.pick([-1, 1]); fx.add({ x: ox + c * d, y: oy + sn * d, z: hz, vx: -sn * o * U.rand(40, 110) + c * 60, vy: cn(c, o), vz: U.rand(40, 160), grav: 640, life: U.rand(0.3, 0.55), size: U.rand(3, 6), color: U.pick(["#ffffff", "#e6fbff", "#8fdcff"]), shape: "drop", drag: 0.4 }); }
      if (Math.random() < dt * 30) { const an = U.rand(TAU); fx.add({ x: ox + c * reach, y: oy + sn * reach, z: hz, vx: Math.cos(an) * U.rand(60, 160), vy: Math.sin(an) * U.rand(30, 90), vz: U.rand(120, 300), grav: 760, life: U.rand(0.35, 0.6), size: U.rand(3, 7), color: U.pick(["#ffffff", "#bfeaff"]), shape: "drop", drag: 0.3 }); }
      if (B.arena.water && Math.random() < dt * 8 && typeof Water !== "undefined") Water.ripple(B.arena, end.x, end.y, 6, 40, 0.8, { foam: true });
    }
    return true;
  }
  const cn = (c, o) => c * o * U.rand(25, 70);

  // Surf: called every frame of the attack instead of the generic ring + particle spray.
  function waveTick(B, a, dt) {
    const fx = B.fx, w = waveState(a);
    if (!a.started) { a.started = true; a.crashed = []; fx.shakeCam(5, 0.3); }
    const drop = (x, y, z, an, pow) => fx.add({ x, y, z, vx: Math.cos(an) * U.rand(40, 150) * pow, vy: Math.sin(an) * U.rand(20, 80) * pow, vz: U.rand(80, 260) * pow, grav: 720, life: U.rand(0.3, 0.65), size: U.rand(3, 7), color: U.pick(["#ffffff", "#ffffff", "#bfeaff", "#8fdcff"]), shape: "drop", drag: 0.3 });
    if (w.spout > 0.1) for (let i = 0; i < Math.round(dt * 160); i++) { const an = U.rand(TAU); drop(a.x + Math.cos(an) * 30, a.y + Math.sin(an) * 16, 60 * w.spout, an, 0.9); }
    for (const c of w.crests) {
      // spray whipped off the crest, more as it grows
      for (let i = 0; i < Math.round(dt * (60 + c.h * 1.6)); i++) { const an = U.rand(TAU); drop(a.x + Math.cos(an) * c.r, a.y + Math.sin(an) * c.r * SQ, c.h * U.rand(0.7, 1), an, 0.6 + c.h / 160); }
      // the wave breaks on the rim: a ring of splash, mist, a jolt — bigger for each wave
      if (c.p >= 0.93 && !a.crashed[c.i]) {
        a.crashed[c.i] = true;
        const big = 0.6 + c.i * 0.3;
        for (let i = 0; i < 26 + c.i * 14; i++) { const an = U.rand(TAU); drop(a.x + Math.cos(an) * a.r1, a.y + Math.sin(an) * a.r1 * SQ, U.rand(10, 60) * big, an, 1 + big * 0.5); }
        for (let i = 0; i < 6 + c.i * 3; i++) { const an = U.rand(TAU); fx.add({ x: a.x + Math.cos(an) * a.r1, y: a.y + Math.sin(an) * a.r1 * SQ, z: 10, vx: Math.cos(an) * 40, vy: Math.sin(an) * 20, vz: U.rand(30, 80), life: U.rand(0.6, 1), size: U.rand(16, 28) * big, color: "#f2fbff", shape: "smoke", alpha: 0.42, grow: 1.7 }); }
        fx.shakeCam(4 + c.i * 3.5, 0.22 + c.i * 0.05); 
        if (c.i === CRESTS.length - 1) fx.ring(a.x, a.y, a.r1 * 0.9, a.r1 * 1.25, 0.4, "#ffffff", 6);
      }
    }
    if (B.arena.water && typeof Water !== "undefined" && Math.random() < dt * 14) { const an = U.rand(TAU), r = U.rand(0.3, 1) * w.flood; Water.ripple(B.arena, a.x + Math.cos(an) * r, a.y + Math.sin(an) * r * SQ, 8, 50, 0.9, { foam: true }); }
  }

  const styleOf = (s) => (s.cone ? flame : s.type === "ice" ? iceBeam : s.type === "water" ? waterBeam : null);
  return { flame, fireBlast, fireBlastTick, iceBeam, waterBeam, wave, waveGround, drawWorld, waveHitR, whirlAir, whirlTick, beamTick, waveTick, styleOf };
})();
