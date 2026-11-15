// Title scene: a 320x180 pixel sunset (from the teaser), drawn 4x, with the ORIGINAL sprites (2x) wandering on the ridge.
"use strict";

const Title = (() => {
  const LW = 320, LH = 180, Z = 4; // backdrop art px → canvas px
  let cv, ctx, bg, running = false, mons = [], t = 0, last = 0;
  const CAST = ["pikachu", "charmander", "bulbasaur", "squirtle", "eevee", "ditto", "jigglypuff", "mimikyu-disguised", "gengar", "snorlax", "lucario", "greninja", "togepi", "psyduck", "piplup", "riolu"];

  function init(canvas) {
    cv = canvas; cv.width = LW * Z; cv.height = LH * Z; ctx = cv.getContext("2d"); ctx.imageSmoothingEnabled = false;
    bg = U.canvas(LW, LH); paintBackdrop(bg.getContext("2d"));
    ctx.setTransform(Z, 0, 0, Z, 0, 0); ctx.drawImage(bg, 0, 0); // visible at once, even before the loop runs
  }
  function paintBackdrop(x) {
    const bands = ["#231a40", "#2f2152", "#432a62", "#5d3270", "#7d3a78", "#a2477a", "#c75b74", "#e6766a", "#f3955e", "#f8b860", "#ffd679"];
    const H = 120;
    for (let y = 0; y < H; y++) {
      const v = (y / H) * (bands.length - 1), i = Math.floor(v), f = v - i;
      for (let xx = 0; xx < LW; xx++) { x.fillStyle = f > Pixel.bayer(xx, y) ? bands[Math.min(bands.length - 1, i + 1)] : bands[i]; x.fillRect(xx, y, 1, 1); }
    }
    x.fillStyle = bands[bands.length - 1]; x.fillRect(0, H, LW, LH - H); // no see-through gap where the ridges dip below the sky
    // sun with stepped glow
    for (const [r, c] of [[46, "rgba(255,214,121,0.35)"], [34, "rgba(255,226,150,0.5)"], [22, "#ffe6a0"], [17, "#fff3c8"]]) Pixel.dither(x, 160, 104, r, r, c, (k) => (r > 30 ? 1 - k : 1), 3);
    // cloud streaks
    const rng = U.seeded(12);
    for (let i = 0; i < 26; i++) { const y = 10 + rng() * 70, w = 20 + rng() * 70, xx = rng() * LW; x.fillStyle = y < 40 ? "#6a4a8a" : "#f6c08a"; x.fillRect(xx | 0, y | 0, w | 0, 1); x.fillStyle = y < 40 ? "#4a3470" : "#e0906e"; x.fillRect((xx + 4) | 0, (y + 1) | 0, (w - 8) | 0, 1); }
    // mountains
    const ridge = (base, amp, col, seed) => { const r = U.seeded(seed); let h = 0, v = 0; for (let xx = 0; xx < LW; xx++) { v += (r() - 0.5) * 0.9; v *= 0.92; h += v; h *= 0.985; const y = base + Math.sin(xx * 0.03 + seed) * amp + h * 2; x.fillStyle = col; x.fillRect(xx, y | 0, 1, LH); } };
    ridge(96, 10, "#6b4176", 3); ridge(108, 7, "#4e2f62", 7); ridge(118, 5, "#3a2550", 11);
    // grass ridge
    x.fillStyle = "#c9d45a"; x.fillRect(0, 132, LW, 2);
    x.fillStyle = "#7a9a34"; x.fillRect(0, 134, LW, 3);
    x.fillStyle = "#3e5a2a"; x.fillRect(0, 137, LW, 6);
    x.fillStyle = "#2a2f36"; x.fillRect(0, 143, LW, LH);
    for (let i = 0; i < 260; i++) { x.fillStyle = ["#e9f08a", "#a6c040", "#5f7f2c"][(rng() * 3) | 0]; x.fillRect((rng() * LW) | 0, 130 + (rng() * 5) | 0, 1, 1 + (rng() * 2 | 0)); }
    for (let i = 0; i < 40; i++) { x.fillStyle = ["#ffe8f0", "#fff6a0", "#d8e8ff"][(rng() * 3) | 0]; x.fillRect((rng() * LW) | 0, 131 + (rng() * 3) | 0, 1, 1); }
  }

  function start() {
    if (running) return;
    running = true;
    const picks = CAST.slice().sort(() => Math.random() - 0.5).slice(0, 4).map(Data.bySlug).filter(Boolean);
    try { mons = picks.map((p, i) => { const sh = Math.random() < 0.15; return { p, s: { front: Assets.hd(p, sh), frontR: Assets.hd(p, sh, "front-reverse") }, x: 30 + i * 75 + Math.random() * 20, dir: Math.random() < 0.5 ? -1 : 1, hop: 0, hv: 0, wait: Math.random() * 2, ph: Math.random() * 10 }; }); } catch (e) { console.error(e); mons = []; }
    last = performance.now();
    requestAnimationFrame(loop);
  }
  function stop() { running = false; }
  function loop(now) {
    if (!running) return;
    const dt = Math.min(0.05, (now - last) / 1000); last = now; t += dt;
    ctx.setTransform(Z, 0, 0, Z, 0, 0); ctx.imageSmoothingEnabled = false;
    ctx.drawImage(bg, 0, 0);
    for (let i = 0; i < 18; i++) { const x = (i * 53.3 + t * (3 + i % 4)) % LW, y = 40 + ((i * 37) % 70) + Math.sin(t + i) * 3; if (Math.sin(t * 2 + i * 1.7) > 0.6) { ctx.fillStyle = "#fff6d0"; ctx.fillRect(x | 0, y | 0, 1, 1); } }
    for (const m of mons) try {
      m.wait -= dt;
      if (m.wait <= 0) { m.x += m.dir * 18 * dt; if (Math.random() < dt * 0.4) { m.wait = 1 + Math.random() * 2; if (Math.random() < 0.4) m.dir *= -1; } if (Math.random() < dt * 0.5 && m.hop === 0) m.hv = 70; }
      if (m.x < 16) m.dir = 1; if (m.x > LW - 16) m.dir = -1;
      m.hv -= 260 * dt; m.hop = Math.max(0, m.hop + m.hv * dt); if (m.hop === 0) m.hv = Math.max(0, m.hv);
      const rec = m.dir > 0 ? m.s.frontR : m.s.front;
      if (!rec.frames || !rec.ready) continue;
      const img = Assets.frame(rec, t + m.ph), b = rec.bounds;
      const walk = m.wait <= 0 ? Math.abs(Math.sin((t + m.ph) * 8)) * 2 : 0;
      ctx.setTransform(Z, 0, 0, Z, 0, 0);
      Pixel.ellipse(ctx, m.x, 135, Math.round(b.w * 0.5 * 0.4), 2, "rgba(20,10,30,0.45)");
      // original sprite pixels at 2 canvas px each (the backdrop's art px are 4)
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      const k = 2, fx = Math.round(m.x * Z - (b.x + b.w / 2) * k), fy = Math.round((135 - m.hop - walk) * Z - (b.y + b.h) * k);
      ctx.drawImage(img, fx, fy, img.width * k, img.height * k);
    } catch (e) { console.error(e); } // one bad sprite must not stop the backdrop
    requestAnimationFrame(loop);
  }
  return { init, start, stop };
})();
