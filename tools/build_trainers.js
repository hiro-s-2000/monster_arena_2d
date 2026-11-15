#!/usr/bin/env node
// Copies the human overworld sprites (HGSS + FRLG ROM rips) into ./trainers and writes data/trainers.js.
// People: HGSS attr_hint "ひと" (16-frame sheets), the walking protagonists that hint skipped,
// and FRLG 16×32 walkers without vehicle/pose variants.
// Extra HGSS walkers are resampled to a 16×32 cell so they draw at the same size as FRLG.
// Usage: node tools/build_trainers.js [path/to/rom_analytics]
"use strict";
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");

const ROOT = path.resolve(__dirname, "..");
const SRC = path.resolve(process.argv[2] || path.join(ROOT, "..", "あまり触らない。", "rom_analytics"));
const OUT_DIR = path.join(ROOT, "trainers");
const SKIP = /pikachu|ピカチュウ|すわって|ねている|おしえテレビ|うけつけ|_/;
// Overworld Pokémon the ROM still tags as ひと: red Gyarados, Sudowoodo, Lapras, Slowpoke.
const POKEMON_IDS = new Set([237, 381, 390, 414]);
// 32-frame protagonists, the other いたこ sheet, and the center nurse. Not bikes, poses, or objects.
const EXTRA_IDS = [0, 238, 258, 97, 239, 259, 295, 296, 250, 335];
const REF_H = 20; // modal opaque height of an imported FRLG west-stand frame
const out = [];

fs.mkdirSync(OUT_DIR, { recursive: true });
const copy = (from, name) => { fs.copyFileSync(from, path.join(OUT_DIR, name)); return "trainers/" + name; };

function crc32(buf) {
  let c = ~0;
  for (let i = 0; i < buf.length; i++) {
    c ^= buf[i];
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return (~c) >>> 0;
}
function pngChunk(type, data) {
  const body = Buffer.concat([Buffer.from(type), data]);
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const c = Buffer.alloc(4); c.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, c]);
}
function decodePNG(buf) {
  let p = 8, w = 0, h = 0, ct = 0, idat = [];
  while (p < buf.length) {
    const len = buf.readUInt32BE(p); p += 4;
    const type = buf.toString("ascii", p, p + 4); p += 4;
    const data = buf.subarray(p, p + len); p += len; p += 4;
    if (type === "IHDR") { w = data.readUInt32BE(0); h = data.readUInt32BE(4); ct = data[9]; }
    else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
  }
  if (ct !== 6 && ct !== 2) throw new Error("unsupported png color " + ct);
  const bpp = ct === 6 ? 4 : 3;
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const stride = w * bpp;
  const px = Buffer.alloc(w * h * 4);
  let s = 0, prev = Buffer.alloc(stride);
  const paeth = (a, b, cc) => {
    const q = a + b - cc, pa = Math.abs(q - a), pb = Math.abs(q - b), pc = Math.abs(q - cc);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : cc;
  };
  for (let y = 0; y < h; y++) {
    const filter = raw[s++];
    const row = Buffer.from(raw.subarray(s, s + stride)); s += stride;
    if (filter === 1) for (let i = 0; i < stride; i++) row[i] = (row[i] + (i >= bpp ? row[i - bpp] : 0)) & 255;
    else if (filter === 2) for (let i = 0; i < stride; i++) row[i] = (row[i] + prev[i]) & 255;
    else if (filter === 3) for (let i = 0; i < stride; i++) row[i] = (row[i] + (((i >= bpp ? row[i - bpp] : 0) + prev[i]) >> 1)) & 255;
    else if (filter === 4) for (let i = 0; i < stride; i++) row[i] = (row[i] + paeth(i >= bpp ? row[i - bpp] : 0, prev[i], i >= bpp ? prev[i - bpp] : 0)) & 255;
    else if (filter !== 0) throw new Error("png filter " + filter);
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4, i = x * bpp;
      px[o] = row[i]; px[o + 1] = row[i + 1]; px[o + 2] = row[i + 2]; px[o + 3] = bpp === 4 ? row[i + 3] : 255;
    }
    prev = row;
  }
  return { w, h, px };
}
function encodePNG(w, h, px) {
  const raw = Buffer.alloc((w * 4 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 4 + 1)] = 0;
    px.copy(raw, y * (w * 4 + 1) + 1, y * w * 4, (y + 1) * w * 4);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 6;
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", zlib.deflateSync(raw)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}
function opaqueBox(img, x0, fw, fh) {
  let minx = fw, miny = fh, maxx = -1, maxy = -1;
  for (let y = 0; y < fh; y++) for (let x = 0; x < fw; x++) {
    if (img.px[(y * img.w + x0 + x) * 4 + 3] > 16) {
      if (x < minx) minx = x; if (y < miny) miny = y; if (x > maxx) maxx = x; if (y > maxy) maxy = y;
    }
  }
  return maxx < 0 ? null : { w: maxx - minx + 1, h: maxy - miny + 1, y: miny };
}
function sameFrame(img, a, b, fw, fh) {
  for (let y = 0; y < fh; y++) {
    const row = y * img.w * 4;
    const sa = row + a * fw * 4, sb = row + b * fw * 4;
    if (Buffer.compare(img.px.subarray(sa, sa + fw * 4), img.px.subarray(sb, sb + fw * 4)) !== 0) return false;
  }
  return true;
}
// Bottom-center of each 32×32 cell stays the bottom-center of a 16×32 cell.
// Side-view bodies are scaled to the FRLG opaque height and kept inside 16px.
function fitWalker(img, name) {
  const fw = 32, fh = 32, dw = 16, dh = 32;
  const frames = img.w / fw;
  if (img.h !== fh || frames < 16 || img.w !== frames * fw) throw new Error(name + " is not a 32px walker sheet");
  if (sameFrame(img, 8, 9, fw, fh)) throw new Error(name + " west walk frames match");
  let maxW = 1, maxH = 1;
  for (let fi = 8; fi <= 15; fi++) {
    const b = opaqueBox(img, fi * fw, fw, fh);
    if (!b) continue;
    if (b.w > maxW) maxW = b.w;
    if (b.h > maxH) maxH = b.h;
  }
  let scale = REF_H / maxH;
  if (maxW * scale > dw) scale = dw / maxW;
  const destW = frames * dw;
  const px = Buffer.alloc(destW * dh * 4);
  for (let fi = 0; fi < frames; fi++) {
    for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) {
      const sx = fw / 2 + (x + 0.5 - dw / 2) / scale;
      const sy = fh + (y + 0.5 - dh) / scale;
      const ix = Math.floor(sx), iy = Math.floor(sy);
      if (ix < 0 || iy < 0 || ix >= fw || iy >= fh) continue;
      const s = (iy * img.w + fi * fw + ix) * 4;
      if (img.px[s + 3] <= 16) continue;
      const d = (y * destW + fi * dw + x) * 4;
      px[d] = img.px[s]; px[d + 1] = img.px[s + 1]; px[d + 2] = img.px[s + 2]; px[d + 3] = 255;
    }
  }
  const fitted = { w: destW, h: dh, px };
  const body = opaqueBox(fitted, 8 * dw, dw, dh);
  if (!body || body.h < 16 || body.h > 22 || body.w > 16) throw new Error(name + " fitted " + (body && (body.w + "x" + body.h)));
  return { png: encodePNG(destW, dh, px), body, scale };
}

// HGSS: 32×32 frames — 0-3 north, 4-7 south, 8-11 west, 12-15 east (stand, step, stand, step)
const hg = path.join(SRC, "charactor_sshg");
const hgChars = JSON.parse(fs.readFileSync(path.join(hg, "manifest.json"), "utf8")).characters;
for (const c of hgChars) {
  const d = JSON.parse(fs.readFileSync(path.join(hg, c.data_file), "utf8"));
  if (d.attr_hint !== "ひと" || d.frame_count !== 16 || POKEMON_IDS.has(c.graphics_id) || SKIP.test(d.name_ja + d.internal_name)) continue;
  const src = copy(path.join(hg, c.image_file), `hgss_${String(c.graphics_id).padStart(4, "0")}.png`);
  out.push({ n: d.name_ja, g: "hgss", src, w: 32, h: 32, west: [8, 9, 8, 11], east: [12, 13, 12, 15], ef: false });
}
const byId = new Map(hgChars.map((c) => [c.graphics_id, c]));
for (const id of EXTRA_IDS) {
  const c = byId.get(id);
  if (!c) throw new Error("missing graphics " + id);
  const d = JSON.parse(fs.readFileSync(path.join(hg, c.data_file), "utf8"));
  const img = decodePNG(fs.readFileSync(path.join(hg, c.image_file)));
  const fit = fitWalker(img, d.name_ja + " " + id);
  const file = `hgss_${String(id).padStart(4, "0")}.png`;
  fs.writeFileSync(path.join(OUT_DIR, file), fit.png);
  out.push({ n: d.name_ja, g: "hgss", src: "trainers/" + file, w: 16, h: 32, west: [8, 9, 8, 11], east: [12, 13, 12, 15], ef: false });
  console.log(`fit ${file} ${d.name_ja} scale ${fit.scale.toFixed(3)} west ${fit.body.w}x${fit.body.h}`);
}
// FRLG: 16×32 frames — 2 = face west, 7/8 = west steps; east is west mirrored
const fr = path.join(SRC, "charactor_frlg");
for (const c of JSON.parse(fs.readFileSync(path.join(fr, "manifest.json"), "utf8")).characters) {
  if (c.width !== 16 || c.height !== 32 || c.frame_count < 9 || SKIP.test(c.name_ja)) continue;
  const src = copy(path.join(fr, c.image_file), `frlg_${String(c.graphics_id).padStart(3, "0")}.png`);
  out.push({ n: c.name_ja, g: "frlg", src, w: 16, h: 32, west: [2, 7, 2, 8], east: [2, 7, 2, 8], ef: true });
}
fs.writeFileSync(path.join(ROOT, "data", "trainers.js"), `// Generated by tools/build_trainers.js — do not edit by hand.\nwindow.ARENA_TRAINERS = ${JSON.stringify(out)};\n`);
console.log(`trainers: ${out.length} (hgss ${out.filter((t) => t.g === "hgss").length}, frlg ${out.filter((t) => t.g === "frlg").length})`);
