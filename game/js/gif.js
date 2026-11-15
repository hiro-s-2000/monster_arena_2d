// Minimal animated-GIF decoder. Canvas drawImage() of an <img> GIF only ever paints frame 0, so the
// sprite bytes are fetched and decoded to per-frame canvases (full composited frames, disposal honoured).
"use strict";

const Gif = (() => {
  function lzw(minCode, data, count) {
    const out = new Uint8Array(count);
    const clear = 1 << minCode, eoi = clear + 1;
    let codeSize = minCode + 1, mask = (1 << codeSize) - 1, next = eoi + 1;
    const prefix = new Uint16Array(4096), suffix = new Uint8Array(4096), stack = new Uint8Array(4097);
    for (let i = 0; i < clear; i++) suffix[i] = i;
    let acc = 0, bits = 0, ip = 0, op = 0, prev = -1, first = 0;
    while (op < count) {
      while (bits < codeSize) { if (ip >= data.length) return out; acc |= data[ip++] << bits; bits += 8; }
      let code = acc & mask; acc >>>= codeSize; bits -= codeSize;
      if (code === clear) { codeSize = minCode + 1; mask = (1 << codeSize) - 1; next = eoi + 1; prev = -1; continue; }
      if (code === eoi) break;
      if (prev === -1) { out[op++] = suffix[code]; prev = first = code; continue; }
      const incoming = code;
      let sp = 0;
      if (code >= next) { stack[sp++] = first; code = prev; }
      while (code >= clear) { stack[sp++] = suffix[code]; code = prefix[code]; }
      first = suffix[code];
      stack[sp++] = first;
      while (sp > 0 && op < count) out[op++] = stack[--sp];
      if (next < 4096) {
        prefix[next] = prev; suffix[next] = first; next++;
        if ((next & mask) === 0 && next < 4096) { codeSize++; mask = (1 << codeSize) - 1; }
      }
      prev = incoming;
    }
    return out;
  }

  function interlace(h) {
    const rows = [];
    for (const [s, step] of [[0, 8], [4, 8], [2, 4], [1, 2]]) for (let y = s; y < h; y += step) rows.push(y);
    return rows;
  }

  function decode(buffer) {
    const b = new Uint8Array(buffer);
    if (b[0] !== 0x47 || b[1] !== 0x49 || b[2] !== 0x46) throw new Error("not a GIF");
    const W = b[6] | (b[7] << 8), H = b[8] | (b[9] << 8);
    let p = 13, gct = null;
    if (b[10] & 0x80) { const n = 3 << ((b[10] & 7) + 1); gct = b.subarray(p, p + n); p += n; }
    const frames = [];
    let bounds0 = null;
    const comp = U.canvas(W, H), cctx = comp.getContext("2d", { willReadFrequently: true });
    let gce = { delay: 0.1, trans: -1, disposal: 0 }, saved = null;
    while (p < b.length) {
      const id = b[p++];
      if (id === 0x3b) break;
      if (id === 0x21) {
        const label = b[p++];
        if (label === 0xf9) {
          const f = b[p + 1];
          gce = { disposal: (f >> 2) & 7, delay: (b[p + 2] | (b[p + 3] << 8)) / 100, trans: f & 1 ? b[p + 4] : -1 };
        }
        while (b[p]) p += b[p] + 1;
        p++;
        continue;
      }
      if (id !== 0x2c) break;
      const fx = b[p] | (b[p + 1] << 8), fy = b[p + 2] | (b[p + 3] << 8), fw = b[p + 4] | (b[p + 5] << 8), fh = b[p + 6] | (b[p + 7] << 8);
      const flags = b[p + 8];
      p += 9;
      let ct = gct;
      if (flags & 0x80) { const n = 3 << ((flags & 7) + 1); ct = b.subarray(p, p + n); p += n; }
      const minCode = b[p++];
      let len = 0;
      const parts = [];
      while (b[p]) { parts.push(b.subarray(p + 1, p + 1 + b[p])); len += b[p]; p += b[p] + 1; }
      p++;
      const data = new Uint8Array(len);
      let o = 0;
      for (const part of parts) { data.set(part, o); o += part.length; }
      const idx = lzw(minCode, data, fw * fh);
      if (gce.disposal === 3) saved = cctx.getImageData(0, 0, W, H);
      const img = cctx.getImageData(fx, fy, fw, fh);
      const rows = flags & 0x40 ? interlace(fh) : null;
      const d = img.data;
      for (let y = 0; y < fh; y++) {
        const dy = rows ? rows[y] : y;
        for (let x = 0; x < fw; x++) {
          const ci = idx[y * fw + x];
          if (ci === gce.trans || !ct) continue;
          const di = (dy * fw + x) * 4;
          d[di] = ct[ci * 3]; d[di + 1] = ct[ci * 3 + 1]; d[di + 2] = ct[ci * 3 + 2]; d[di + 3] = 255;
        }
      }
      cctx.putImageData(img, fx, fy);
      if (!frames.length && typeof Pixel !== "undefined" && Pixel.boundsData) {
        bounds0 = Pixel.boundsData(fw === W && fh === H && fx === 0 && fy === 0 ? img.data : cctx.getImageData(0, 0, W, H).data, W, H);
      }
      const snap = U.canvas(W, H);
      snap.getContext("2d").drawImage(comp, 0, 0);
      frames.push({ canvas: snap, delay: Math.max(0.03, gce.delay || 0.1) });
      if (gce.disposal === 2) cctx.clearRect(fx, fy, fw, fh);
      else if (gce.disposal === 3 && saved) cctx.putImageData(saved, 0, 0);
      gce = { delay: 0.1, trans: -1, disposal: 0 };
    }
    return { W, H, frames, bounds0 };
  }

  return { decode };
})();
