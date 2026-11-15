// Opaque bounds of every sprite's first frame, for the debug page (game/debug.html) when it is opened
// straight from the disk (file://). There the browser forbids reading sprite pixels, so the page cannot
// measure where the body sits inside a GIF or a sheet cell. This table carries those numbers instead.
// Output: data/sprite_bounds.js
//   window.ARENA_BOUNDS = {
//     g: { "<dir>/<file>.gif": [W, H, x, y, w, h] },        // GIF size + frame-0 bounds
//     s: { "<id>" | "<id>-shiny": [x, y, w, h, ×8] }          // 8-direction sheet, row 0, per column (S…SW)
//   }
// Re-run after adding sprites or rebuilding sheets:
//   PLAYWRIGHT=/path/to/node_modules/playwright node tools/build_sprite_bounds.js
"use strict";
const fs = require("fs"), path = require("path"), http = require("http");
const ROOT = path.join(__dirname, "..");
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");

const load = (file, name) => { const raw = fs.readFileSync(path.join(ROOT, file), "utf8"); return JSON.parse(raw.slice(raw.indexOf("=", raw.indexOf(name)) + 1).trim().replace(/;\s*$/, "")); };
const dex = load("data/monsterdex.js", "window.ARENA_MONSTERDEX");
const man = load("data/dir8.js", "window.ARENA_DIR8");
let OLD = {}; try { OLD = load("data/sprite_bounds.js", "window.ARENA_BOUNDS"); } catch (e) {}
const VIEWS = ["front", "front-reverse", "back", "back-reverse", "front-shiny", "front-shiny-reverse", "back-shiny", "back-shiny-reverse"];

(async () => {
  const server = http.createServer((q, r) => fs.readFile(path.join(ROOT, decodeURIComponent(q.url.split("?")[0])), (e, d) => { if (e) { r.writeHead(404); r.end(); } else { r.writeHead(200); r.end(d); } }));
  await new Promise((res) => server.listen(0, "127.0.0.1", res));
  const base = `http://127.0.0.1:${server.address().port}/monsterSprite/`;
  const browser = await chromium.launch(); const page = await browser.newPage();
  await page.goto(base.replace("monsterSprite/", "data/dir8.js"));
  await page.evaluate(() => {
    window.measure = (url, cells) => new Promise((res) => {
      const img = new Image();
      img.onload = () => {
        const W = img.naturalWidth, H = img.naturalHeight;
        const c = document.createElement("canvas");
        c.width = cells ? (cells.mode === "actions_8dir" || cells.mode === "actions" ? cells.cw : W) : W;
        c.height = cells ? (cells.mode === "actions_8dir" ? cells.ch * 8 : cells.mode === "actions" ? cells.ch * ((cells.arow || 0) + 1) : cells.ch) : H;
        const x = c.getContext("2d", { willReadFrequently: true }); x.drawImage(img, 0, 0);
        const d = x.getImageData(0, 0, c.width, c.height).data;
        const box = (x0, x1, y0, y1) => {
          let a = x1, b = y1, e = -1, f = -1;
          for (let y = y0; y < y1; y++) for (let i = x0; i < x1; i++) if (d[(y * c.width + i) * 4 + 3]) { if (i < a) a = i; if (i > e) e = i; if (y < b) b = y; if (y > f) f = y; }
          return e < 0 ? [0, 0, x1 - x0, y1 - y0] : [a - x0, b - y0, e - a + 1, f - b + 1];
        };
        if (!cells) res([W, H, ...box(0, W, 0, H)]);
        else if (cells.mode === "actions_8dir") res(Array.from({ length: 8 }, (_, k) => box(0, cells.cw, k * cells.ch, (k + 1) * cells.ch)).flat());
        else if (cells.mode === "actions") { const r0 = cells.arow || 0, b0 = box(0, cells.cw, r0 * cells.ch, (r0 + 1) * cells.ch); res(Array.from({ length: 8 }, () => b0).flat()); }
        else res(Array.from({ length: 8 }, (_, k) => box(k * cells.cw, (k + 1) * cells.cw, 0, cells.ch)).flat());
      };
      img.onerror = () => res(null);
      img.src = url;
    });
  });
  const out = { g: {}, s: {} };
  const jobs = [];
  for (const p of dex) {
    if (!p.dir) continue;
    for (const v of VIEWS) jobs.push(["g", `${p.dir}/${v}.gif`, null]);
    const m = man[p.id];
    // layout "fb"（正面・背面の 2 枚シート。tools/build_dir8_fb.py）は、その道具が書いた値をそのまま残す（ここでは測り直さない）
    if (m && m.layout === "fb") { if (OLD.s && OLD.s[p.id]) out.s[p.id] = OLD.s[p.id]; continue; }
    if (m) {
      const act8 = m.layout === "actions_8dir" || m.sheetDir === "agent-sprite-forge", act = m.layout === "actions";
      const url = act8 ? `${p.dir}/agent-sprite-forge/sheet.png` : act ? `${p.dir}/actions/sheet.png` : `${p.dir}/${p.id}.png`;
      const mode = act8 ? "actions_8dir" : act ? "actions" : "dir8";
      jobs.push(["s", p.id, { url, cw: m.cw, ch: m.ch, mode, arow: m.arow || 0 }]);
      if (m.shiny && !act8 && !act) jobs.push(["s", p.id + "-shiny", { url: `${p.dir}/${p.id}-shiny.png`, cw: m.shiny.cw, ch: m.shiny.ch, mode: "dir8" }]);
    }
  }
  const N = 24;
  for (let i = 0; i < jobs.length; i += N) {
    const batch = jobs.slice(i, i + N);
    const res = await page.evaluate(({ batch, base }) => Promise.all(batch.map(([k, key, cells]) => window.measure(base + encodeURI(cells ? cells.url : key), cells))), { batch, base });
    batch.forEach(([k, key], j) => { if (res[j]) out[k][key] = res[j]; });
    if ((i / N) % 40 === 0) process.stdout.write(`\r${i}/${jobs.length}`);
  }
  fs.writeFileSync(path.join(ROOT, "data/sprite_bounds.js"), `// Generated by tools/build_sprite_bounds.js — do not edit by hand.\nwindow.ARENA_BOUNDS = ${JSON.stringify(out)};\n`);
  console.log(`\nsprite_bounds: ${Object.keys(out.g).length} gifs, ${Object.keys(out.s).length} sheets`);
  await browser.close(); server.close();
})();
