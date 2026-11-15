// Pixel-art toolkit: sprite pixelization (2:1 majority downsample + edge darkening), crisp primitives
// (rows / Bresenham / midpoint circles / ordered dither), a 5x7 bitmap font and a procedural trainer sprite.
"use strict";

const Pixel = (() => {
  const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map((v) => (v + 0.5) / 16);
  const bayer = (x, y) => BAYER[(y & 3) * 4 + (x & 3)];

  // ---- sprite pixelization -------------------------------------------------------------------------
  // Turns a smooth-shaded GIF frame into readable pixel art at `s` × its size (s ≤ 1):
  //  1. feature map — pixels clearly darker than their 5×5 neighbourhood (eyes, mouth, inner lines)
  //  2. block reduce — each output pixel keeps the darkest feature pixel of its block when one is there,
  //     otherwise the dominant colour bin (so eyes and contours survive the downscale)
  //  3. palette — every frame snaps to one shared ≤`colors` palette (built from the first frame)
  //  4. outline — a 1 px dark selective outline around the silhouette (canvas grows by 1 px each side)
  // `ctx` carries the shared palette between frames of one sprite.
  function pixelize(src, w, h, s = 1, ctx = {}) {
    const sc = U.canvas(w, h), sx = sc.getContext("2d", { willReadFrequently: true });
    sx.drawImage(src, 0, 0);
    const d0 = sx.getImageData(0, 0, w, h).data;
    const N = w * h, L = new Float32Array(N), A = new Uint8Array(N);
    for (let i = 0; i < N; i++) { const o = i * 4; A[i] = d0[o + 3] >= 128 ? 1 : 0; L[i] = 0.3 * d0[o] + 0.59 * d0[o + 1] + 0.11 * d0[o + 2]; }
    // integral images of opaque luminance / counts → local mean
    const W1 = w + 1, IS = new Float64Array(W1 * (h + 1)), IC = new Uint32Array(W1 * (h + 1));
    for (let y = 0; y < h; y++) { let rs = 0, rc = 0; for (let x = 0; x < w; x++) { const i = y * w + x; if (A[i]) { rs += L[i]; rc++; } IS[(y + 1) * W1 + x + 1] = IS[y * W1 + x + 1] + rs; IC[(y + 1) * W1 + x + 1] = IC[y * W1 + x + 1] + rc; } }
    const R = 2, F = new Uint8Array(N);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (!A[i]) continue;
      const x0 = Math.max(0, x - R), x1 = Math.min(w, x + R + 1), y0 = Math.max(0, y - R), y1 = Math.min(h, y + R + 1);
      const c = IC[y1 * W1 + x1] - IC[y0 * W1 + x1] - IC[y1 * W1 + x0] + IC[y0 * W1 + x0];
      const sum = IS[y1 * W1 + x1] - IS[y0 * W1 + x1] - IS[y1 * W1 + x0] + IS[y0 * W1 + x0];
      const mean = sum / Math.max(1, c);
      const dv = mean - L[i];
      if ((dv > 38 && L[i] < 100) || dv > 72 || L[i] < 42) F[i] = 1;
    }
    const OW = Math.max(1, Math.round(w * s)), OH = Math.max(1, Math.round(h * s)), P = 1;
    const CW = OW + P * 2, CH = OH + P * 2;
    const px = new Int32Array(CW * CH).fill(-1); // packed rgb or -1
    const bins = new Map();
    for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) {
      const bx0 = Math.floor(ox / s), by0 = Math.floor(oy / s);
      const bx1 = Math.min(w, Math.max(bx0 + 1, Math.floor((ox + 1) / s))), by1 = Math.min(h, Math.max(by0 + 1, Math.floor((oy + 1) / s)));
      let tot = 0, op = 0, fBest = -1, fL = 1e9, fN = 0;
      bins.clear();
      for (let y = by0; y < by1; y++) for (let x = bx0; x < bx1; x++) {
        tot++;
        const i = y * w + x; if (!A[i]) continue;
        op++;
        if (F[i]) { fN++; if (L[i] < fL) { fL = L[i]; fBest = i; } }
        const o = i * 4, k = ((d0[o] >> 4) << 8) | ((d0[o + 1] >> 4) << 4) | (d0[o + 2] >> 4);
        let b = bins.get(k); if (!b) { b = [0, 0, 0, 0]; bins.set(k, b); }
        b[0]++; b[1] += d0[o]; b[2] += d0[o + 1]; b[3] += d0[o + 2];
      }
      if (!op || op / tot < 0.42) continue;
      let rgb;
      if (fBest >= 0 && (fN / op >= 0.2 || fL < 60)) { const o = fBest * 4; rgb = [d0[o], d0[o + 1], d0[o + 2]]; }
      else { let best = null; for (const b of bins.values()) if (!best || b[0] > best[0]) best = b; rgb = [best[1] / best[0], best[2] / best[0], best[3] / best[0]]; }
      px[(oy + P) * CW + ox + P] = (Math.round(rgb[0]) << 16) | (Math.round(rgb[1]) << 8) | Math.round(rgb[2]);
    }
    // palette (shared across frames)
    if (!ctx.pal) ctx.pal = medianCut(px.filter((v) => v >= 0), ctx.colors || 22);
    const pal = ctx.pal, memo = new Map();
    const snap = (v) => { let r = memo.get(v); if (r !== undefined) return r; const R0 = v >> 16, G0 = (v >> 8) & 255, B0 = v & 255; let bd = 1e9; r = v; for (const p of pal) { const dr = R0 - (p >> 16), dg = G0 - ((p >> 8) & 255), db = B0 - (p & 255), dd = dr * dr * 3 + dg * dg * 4 + db * db * 2; if (dd < bd) { bd = dd; r = p; } } memo.set(v, r); return r; };
    for (let i = 0; i < px.length; i++) if (px[i] >= 0) px[i] = snap(px[i]);
    const out = U.canvas(CW, CH), ox2 = out.getContext("2d", { willReadFrequently: true });
    const img = ox2.createImageData(CW, CH), d = img.data;
    for (let y = 0; y < CH; y++) for (let x = 0; x < CW; x++) {
      const i = y * CW + x, o = i * 4;
      let v = px[i];
      if (v >= 0) { d[o] = v >> 16; d[o + 1] = (v >> 8) & 255; d[o + 2] = v & 255; d[o + 3] = 255; continue; }
      // outline: darkest-hue of the touching body pixels
      let n = -1;
      if (x > 0 && px[i - 1] >= 0) n = px[i - 1]; else if (x < CW - 1 && px[i + 1] >= 0) n = px[i + 1];
      else if (y > 0 && px[i - CW] >= 0) n = px[i - CW]; else if (y < CH - 1 && px[i + CW] >= 0) n = px[i + CW];
      if (n < 0) continue;
      d[o] = 10 + (n >> 16) * 0.18; d[o + 1] = 8 + ((n >> 8) & 255) * 0.16; d[o + 2] = 18 + (n & 255) * 0.2; d[o + 3] = 255;
    }
    ox2.putImageData(img, 0, 0);
    return out;
  }
  // Median-cut quantizer over packed rgb ints → ≤ k packed colours.
  function medianCut(list, k) {
    const uniq = [...new Set(list)];
    if (uniq.length <= k) return uniq;
    let boxes = [uniq];
    while (boxes.length < k) {
      let bi = -1, bestR = -1, ch = 0;
      boxes.forEach((b, i) => {
        if (b.length < 2) return;
        for (let c = 0; c < 3; c++) { let lo = 255, hi = 0; for (const v of b) { const q = (v >> (16 - c * 8)) & 255; if (q < lo) lo = q; if (q > hi) hi = q; } const r = (hi - lo) * (c === 1 ? 1.2 : 1); if (r > bestR) { bestR = r; bi = i; ch = c; } }
      });
      if (bi < 0) break;
      const b = boxes[bi].sort((p, q) => ((p >> (16 - ch * 8)) & 255) - ((q >> (16 - ch * 8)) & 255));
      boxes.splice(bi, 1, b.slice(0, b.length >> 1), b.slice(b.length >> 1));
    }
    return boxes.map((b) => { let r = 0, g = 0, bl = 0; for (const v of b) { r += v >> 16; g += (v >> 8) & 255; bl += v & 255; } const n = b.length; return (Math.round(r / n) << 16) | (Math.round(g / n) << 8) | Math.round(bl / n); });
  }
  const lum = (k) => 0.3 * ((k >> 16) & 255) + 0.59 * ((k >> 8) & 255) + 0.11 * (k & 255);

  let bc = null, bx = null;
  function boundsData(d, w, h) {
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (let y = 0; y < h; y++) for (let i = 0; i < w; i++) if (d[(y * w + i) * 4 + 3]) { if (i < x0) x0 = i; if (i > x1) x1 = i; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    return x1 < 0 ? { x: 0, y: 0, w, h } : { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
  }
  // Opaque bounds of a canvas or image region (CPU scratch avoids GPU readbacks).
  function bounds(c, sx, sy, sw, sh) {
    const w = sw || c.naturalWidth || c.width, h = sh || c.naturalHeight || c.height;
    if (!w || !h) return { x: 0, y: 0, w: 0, h: 0 };
    if (c.getContext && sx === undefined) {
      const ctx = c.getContext("2d", { willReadFrequently: true });
      if (ctx) return boundsData(ctx.getImageData(0, 0, w, h).data, w, h);
    }
    if (!bc) { bc = U.canvas(w, h); bx = bc.getContext("2d", { willReadFrequently: true }); }
    else {
      if (bc.width < w || bc.height < h) { bc.width = Math.max(bc.width, w); bc.height = Math.max(bc.height, h); }
      bx.clearRect(0, 0, w, h);
    }
    if (sx !== undefined) bx.drawImage(c, sx, sy, w, h, 0, 0, w, h);
    else bx.drawImage(c, 0, 0);
    return boundsData(bx.getImageData(0, 0, w, h).data, w, h);
  }

  // ---- primitives (integer art pixels, no anti-aliasing) -------------------------------------------
  function ellipse(ctx, cx, cy, rx, ry, color) {
    ctx.fillStyle = color;
    cx = Math.round(cx); cy = Math.round(cy);
    for (let y = -ry; y <= ry; y++) {
      const w = Math.round(rx * Math.sqrt(Math.max(0, 1 - (y * y) / (ry * ry + 0.01))));
      if (w > 0) ctx.fillRect(cx - w, cy + y, w * 2, 1);
    }
  }
  function circle(ctx, cx, cy, r, color, thick = 1) { // midpoint ring
    ctx.fillStyle = color;
    cx = Math.round(cx); cy = Math.round(cy);
    for (let t = 0; t < thick; t++) {
      let x = r - t, y = 0, err = 1 - x;
      while (x >= y) {
        for (const [a, b] of [[x, y], [y, x], [-y, x], [-x, y], [-x, -y], [-y, -x], [y, -x], [x, -y]]) ctx.fillRect(cx + a, cy + b, 1, 1);
        y++;
        if (err < 0) err += 2 * y + 1; else { x--; err += 2 * (y - x) + 1; }
      }
    }
  }
  function ring(ctx, cx, cy, rx, ry, color, thick = 1, dash = 0) { // ellipse outline via angle sampling
    ctx.fillStyle = color;
    const n = Math.ceil((rx + ry) * 3.4);
    for (let i = 0; i < n; i++) {
      if (dash && Math.floor(i / dash) % 2) continue;
      const a = (i / n) * U.TAU;
      for (let t = 0; t < thick; t++) ctx.fillRect(Math.round(cx + Math.cos(a) * (rx - t)), Math.round(cy + Math.sin(a) * (ry - t)), 1, 1);
    }
  }
  function line(ctx, x0, y0, x1, y1, color, w = 1) {
    ctx.fillStyle = color;
    x0 = Math.round(x0); y0 = Math.round(y0); x1 = Math.round(x1); y1 = Math.round(y1);
    const dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;
    for (let i = 0; i < 4000; i++) {
      ctx.fillRect(x0 - (w >> 1), y0 - (w >> 1), w, w);
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
  }
  // Ordered-dither blob: density(dx, dy) in 0..1 decides each pixel against the Bayer matrix.
  function dither(ctx, cx, cy, rx, ry, color, density = (k) => 1 - k, seed) {
    ctx.fillStyle = color;
    const rng = seed !== undefined ? U.seeded(seed) : null;
    for (let y = -ry; y <= ry; y++) for (let x = -rx; x <= rx; x++) {
      const k = Math.sqrt((x * x) / (rx * rx) + (y * y) / (ry * ry));
      if (k > 1) continue;
      let v = density(k);
      if (rng) v *= 0.75 + rng() * 0.5;
      const px = Math.round(cx + x), py = Math.round(cy + y);
      if (v > bayer(px, py)) ctx.fillRect(px, py, 1, 1);
    }
  }

  // ---- 5x7 bitmap font ----------------------------------------------------------------------------
  const G = {
    "0": ".###.#...##..###.#.###..##...#.###.", "1": "..#...##....#....#....#....#...###.", "2": ".###.#...#....#...#...#...#...#####",
    "3": "####.....#....#.###.....#....#####.", "4": "...#...##..#.#.#..#.#####...#....#.", "5": "######....####.....#....##...#.###.",
    "6": ".###.#....#....####.#...##...#.###.", "7": "#####....#...#...#...#....#....#...", "8": ".###.#...##...#.###.#...##...#.###.",
    "9": ".###.#...##...#.####....#....#.###.", A: ".###.#...##...#######...##...##...#", B: "####.#...##...#####.#...##...#####.",
    C: ".###.#...##....#....#....#...#.###.", D: "####.#...##...##...##...##...#####.", E: "######....#....####.#....#....#####",
    F: "######....#....####.#....#....#....", G: ".###.#...##....#.####...##...#.####", H: "#...##...##...#######...##...##...#",
    I: ".###...#....#....#....#....#...###.", J: "..###...#....#....#....##..#..##...", K: "#...##..#.#.#..##...#.#..#..#.#...#",
    L: "#....#....#....#....#....#....#####", M: "#...###.###.#.##.#.##...##...##...#", N: "#...##...###..##.#.##..###...##...#",
    O: ".###.#...##...##...##...##...#.###.", P: "####.#...##...#####.#....#....#....", Q: ".###.#...##...##...##.#.##..#..##.#",
    R: "####.#...##...#####.#.#..#..#.#...#", S: ".#####....#.....###.....#....#####.", T: "#####..#....#....#....#....#....#..",
    U: "#...##...##...##...##...##...#.###.", V: "#...##...##...##...##...#.#.#...#..", W: "#...##...##...##.#.##.#.##.#.#.#.#.",
    X: "#...##...#.#.#...#...#.#.#...##...#", Y: "#...##...#.#.#...#....#....#....#..", Z: "#####....#...#...#...#...#....#####",
    "!": "..#....#....#....#....#.........#..", ".": "................................#..", ":": ".......#...................#.......",
    "-": "................###................", "+": ".......#....#..#####..#....#.......", "/": "....#...#....#...#...#....#...#....",
    "·": "...............#...................", "'": "..#....#...........................", "?": ".###.#...#....#...#...#.........#..",
    "x": "..........#...#.#.#...#...#.#.#...#", "★": "..#....#..#####.###..#.#.#...#.....", " ": "...................................",
  };
  const glyphCache = new Map();
  function glyph(ch) {
    ch = ch === "É" ? "E" : ch === "é" ? "E" : ch;
    if (!G[ch]) ch = G[ch.toUpperCase()] ? ch.toUpperCase() : " ";
    let g = glyphCache.get(ch);
    if (!g) { g = []; const s = G[ch]; for (let i = 0; i < 35; i++) if (s[i] === "#") g.push([i % 5, (i / 5) | 0]); glyphCache.set(ch, g); }
    return g;
  }
  const textWidth = (str, scale = 1, gap = 1) => str.length * (5 + gap) * scale - gap * scale;
  // Draw bitmap text; (x, y) is the top-left. Outline drawn in `outline` colour one cell around.
  function text(ctx, str, x, y, scale, color, outline, gap = 1) {
    x = Math.round(x); y = Math.round(y);
    const pass = (col, o) => {
      ctx.fillStyle = col;
      let cx = x;
      for (const ch of str) {
        for (const [gx, gy] of glyph(ch)) {
          if (o) ctx.fillRect(cx + (gx - 1) * scale, y + (gy - 1) * scale, scale * 3, scale * 3);
          else ctx.fillRect(cx + gx * scale, y + gy * scale, scale, scale);
        }
        cx += (5 + gap) * scale;
      }
    };
    if (outline) pass(outline, true);
    pass(color, false);
  }

  // ---- procedural trainer sprite (14x22) — two frames: idle, throw ---------------------------------
  function trainer(cap, jacket, pants, skin = "#f2c79a", hair = "#3a2416") {
    const frames = [0, 1, 2].map((f) => {
      const c = U.canvas(16, 24), x = c.getContext("2d");
      const R = (col, a, b, w, h) => { x.fillStyle = col; x.fillRect(a, b, w, h); };
      const OL = "#1a1420";
      // legs
      R(OL, 4, 16, 8, 7); R(pants, 5, 16, 2, 6); R(pants, 9, 16, 2, 6); R("#222", 4, 21, 3, 2); R("#222", 9, 21, 3, 2);
      // body
      R(OL, 3, 9, 10, 8); R(jacket, 4, 10, 8, 6); R("#ffffff", 7, 10, 2, 6);
      // arms
      if (f === 1) { R(OL, 11, 3, 3, 8); R(jacket, 12, 4, 1, 6); R(skin, 12, 2, 2, 2); R("#e33", 12, 0, 3, 2); R("#fff", 12, 1, 3, 1); }
      else if (f === 2) { R(OL, 12, 9, 4, 3); R(jacket, 12, 10, 3, 1); R(skin, 15, 10, 1, 1); }
      else { R(OL, 12, 10, 3, 6); R(jacket, 12, 11, 2, 4); R(skin, 12, 15, 2, 1); }
      R(OL, 1, 10, 3, 6); R(jacket, 2, 11, 1, 4); R(skin, 2, 15, 1, 1);
      // head
      R(OL, 4, 2, 8, 8); R(skin, 5, 4, 6, 5); R(hair, 4, 4, 1, 3); R(hair, 11, 4, 1, 3); R("#111", 6, 6, 1, 1); R("#111", 9, 6, 1, 1);
      // cap
      R(OL, 3, 1, 10, 4); R(cap, 4, 1, 8, 3); R("#ffffff", 7, 2, 2, 1); R(cap, 10, 4, 4, 1);
      return c;
    });
    return frames;
  }

  return { BAYER, bayer, pixelize, bounds, boundsData, ellipse, circle, ring, line, dither, text, textWidth, trainer };
})();
