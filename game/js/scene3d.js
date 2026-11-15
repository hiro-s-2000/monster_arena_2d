// Scene3D — HD-2D の「3D の背景」。ステージごとの立体（観客席・壁・塔・木・岩・遠景）を本物の 3D メッシュで持ち、
// 遠近のあるカメラで描く。地面（world レイヤー）は z = 0 の面、モンスターは画面に正対する板（ビルボード）。
//   座標: art px。x = 右、y = 手前（2D マップの下）、z = 上。カメラは手前（+y）の上空から奥（−y）を見下ろす。回転はしない。
//   ステージの定義は game/js/scenes/<id>.js が Scene3D.define(id, { paint, build }) で登録する（書き方は md/scene3d.md）。
//   描画は post.js が同じ WebGL コンテキストで呼ぶ（Scene3D.draw）。render.js は project / unproject で画面との対応を取る。
// ?flat=1 で 3D を切る（作り直す前の平面表示）。?pitch=46&fov=38 でカメラの角度を試せる。
"use strict";

const Scene3D = (() => {
  const q = new URLSearchParams(location.search);
  let OFF = q.get("flat") === "1";
  const DEG = Math.PI / 180;
  const PITCH = (+q.get("pitch") || 38) * DEG; // 見下ろす角度（水平から）。小さいほど低いカメラ（HD-2D ジオラマ視点）
  const FOV = (+q.get("fov") || 38) * DEG;     // 縦の画角
  const NEAR = 10, FAR = 12000;
  const STRIDE = 17; // pos3 uv2 col4 fx4(kind, phase, amp, lift) mat4(emit, lit, fog, mode)
  const DEFS = {};
  const LAYERS = ["back", "front"], BLENDS = ["cut", "alpha", "add"], ATLAS = ["pix", "smooth"];
  const ANIM = { sway: 1, hop: 2, flicker: 3, pulse: 4, float: 5, twinkle: 6 };

  const define = (id, def) => { DEFS[id] = def; };
  const enabled = () => !OFF && typeof Post !== "undefined" && Post.ok();
  const get = (id) => (enabled() && DEFS[id]) || null;

  // ---- camera ------------------------------------------------------------------------------------------
  const V = { C: [0, 0, 0], u: [0, 0, 1], f: [0, -1, 0], F: 1, W: 1, H: 1, pitch: PITCH, vp: new Float32Array(16) };
  // (tx, ty) = 画面の中心に映す地面の点（art px）、ppa = その点での 1 art px の画素数。
  function setCamera(tx, ty, ppa, W, H, pitch) {
    const ph = pitch || PITCH, s = Math.sin(ph), c = Math.cos(ph);
    V.pitch = ph; V.W = W; V.H = H;
    V.f = [0, -c, -s]; V.u = [0, -s, c];
    V.F = (H / 2) / Math.tan(FOV / 2);
    const d = V.F / Math.max(1e-4, ppa);
    V.C = [tx, ty + c * d, s * d]; V.d = d;
    const m = V.vp, kx = 2 * V.F / W, ky = 2 * V.F / H, A = (FAR + NEAR) / (FAR - NEAR), B = -2 * FAR * NEAR / (FAR - NEAR);
    const [cx, cy, cz] = V.C, [, uy, uz] = V.u, [, fy, fz] = V.f;
    // column-major
    m[0] = kx; m[1] = 0; m[2] = 0; m[3] = 0;
    m[4] = 0; m[5] = uy * ky; m[6] = A * fy; m[7] = fy;
    m[8] = 0; m[9] = uz * ky; m[10] = A * fz; m[11] = fz;
    m[12] = -cx * kx; m[13] = -(uy * cy + uz * cz) * ky; m[14] = -A * (fy * cy + fz * cz) + B; m[15] = -(fy * cy + fz * cz);
  }
  // art (x, y, z) → 画面の画素（左上原点）と、その奥行きでの 1 art px の画素数 s。ok = カメラの前にある。
  function project(x, y, z = 0) {
    const dy = y - V.C[1], dz = z - V.C[2];
    const zc = V.f[1] * dy + V.f[2] * dz, yc = V.u[1] * dy + V.u[2] * dz, xc = x - V.C[0];
    const w = Math.max(NEAR, zc), s = V.F / w;
    return { x: V.W / 2 + xc * s, y: V.H / 2 - yc * s, s, ok: zc > NEAR };
  }
  // 画面の画素 → 地面（z = 0）の点。地平線より上を指したら、十分遠い地面の点を返す。
  function unproject(sx, sy) {
    const X = (sx - V.W / 2) / V.F; let Y = -(sy - V.H / 2) / V.F;
    let dz = V.u[2] * Y + V.f[2];
    if (dz > -0.03) { Y = (-0.03 - V.f[2]) / V.u[2]; dz = -0.03; }
    const dy = V.u[1] * Y + V.f[1], t = -V.C[2] / dz;
    return { x: V.C[0] + X * t, y: V.C[1] + dy * t };
  }

  // ---- builder (CPU) ---------------------------------------------------------------------------------------
  const norm = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
  function rgb(c, fb) {
    if (c === undefined || c === null) return fb || [1, 1, 1];
    if (typeof c === "number") return [c, c, c];
    if (typeof c === "string") { const n = parseInt(c.slice(1), 16); return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]; }
    return c;
  }

  function build(A) {
    const def = DEFS[A.id];
    const af = A.af, fcx = af.x + af.w / 2, frontY = af.y + af.h;
    const texs = [], batches = {};
    const cfg = {
      clear: [0.02, 0.03, 0.07],
      fog: { color: [0.06, 0.08, 0.16], near: 120, far: 1100, max: 0.9 }, // near / far = ピントの合う点（画面中央）より何 art px 奥から霞みはじめ、どこで最大になるか
      grade: null,                         // 後処理の上書き（bloom, rays, mote, sun, vig, tint, sat, con, fog[r,g,b,a] など。arena.js の grade と同じ名前）
      light: norm([-0.65, -0.23, 0.72]),   // 光の来る向き（モンスターの落ち影と同じ: 左奥の上から）
      keyTint: [1.08, 1.04, 0.92],         // 光が当たる面の暖色ハイライト
      shadowTint: [0.42, 0.49, 0.68],      // 光が当たらない面の寒色シャドウ
      castCol: [0.03, 0.05, 0.13],         // 落ち影の色
      pitch: 0,                            // ステージ固有のカメラ見下ろし角（deg、0 なら既定 PITCH）
      clip: null,                          // 地面（world）を描く範囲 { x, y, w, h, r }。無ければアリーナ全体
      dof: 1.25,
    };
    let quads = 0;

    function batch(layer, blend, atlas) {
      const k = layer + ":" + blend + ":" + atlas;
      return batches[k] || (batches[k] = { layer, blend, atlas, data: [], tex: [] });
    }
    // ---- textures ----
    function tex(w, h, paint, o = {}) {
      w = Math.max(1, Math.round(w)); h = Math.max(1, Math.round(h));
      const c = U.canvas(w, h), x = c.getContext("2d");
      x.imageSmoothingEnabled = false;
      if (paint) paint(x, w, h);
      const t = { id: texs.length, c, w, h, atlas: o.smooth ? "smooth" : "pix" };
      texs.push(t);
      return t;
    }
    const white = tex(4, 4, (x) => { x.fillStyle = "#fff"; x.fillRect(0, 0, 4, 4); });
    const flat = (color) => ({ flat: rgb(color) }); // 単色の「テクスチャ」（色だけ）

    function shadeOf(n) {
      const L = cfg.light, d = n[0] * L[0] + n[1] * L[1] + n[2] * L[2], k = Math.max(0, d) / Math.max(0.15, L[2]);
      const t = Math.min(1, k), s = cfg.shadowTint, ky = cfg.keyTint || [1, 1, 1];
      const backDark = d < 0 ? 1 + Math.max(-0.22, d * 0.24) : 1;
      const g = k > 1 ? 1 + (k - 1) * 0.38 : 1;
      const hi = [ky[0] * g, ky[1] * g, ky[2] * g];
      return [(s[0] * backDark + (hi[0] - s[0] * backDark) * t), (s[1] * backDark + (hi[1] - s[1] * backDark) * t), (s[2] * backDark + (hi[2] - s[2] * backDark) * t)];
    }
    function fxOf(o, top) {
      const a = o.anim;
      if (!a) return [0, 0, 0];
      const kind = ANIM[a.type] || 0;
      const amp = a.amp === undefined ? 1 : a.amp;
      return [kind, a.phase === undefined ? Math.random() * 6.283 : a.phase, (a.top && !top) ? 0 : amp];
    }
    // 4 頂点（左下, 右下, 右上, 左上）＋ uv ＋ lift を 2 枚の三角形として積む
    function push(t, P, UV, lifts, o, col, mode) {
      const tt = t && t.flat ? white : (t || white);
      const tint = t && t.flat ? t.flat : [1, 1, 1];
      const blend = o.blend || "cut";
      const minY = Math.min(P[0][1], P[1][1], P[2][1], P[3][1]);
      const layer = o.layer || (minY >= frontY - 0.01 ? "front" : "back");
      const b = batch(layer, BLENDS.includes(blend) ? blend : "cut", tt.atlas);
      const a = o.alpha === undefined ? 1 : o.alpha;
      const emit = o.emit || 0, lit = o.lit === undefined ? 1 : o.lit, fog = o.fog === undefined ? 1 : o.fog;
      const f0 = fxOf(o, false), f1 = fxOf(o, true);
      const idx = [0, 1, 2, 0, 2, 3];
      for (const i of idx) {
        const p = P[i], top = i >= 2, f = top ? f1 : f0;
        b.data.push(p[0], p[1], p[2], UV[i][0], UV[i][1], col[0] * tint[0], col[1] * tint[1], col[2] * tint[2], a, f[0], f[1], f[2], lifts ? lifts[i] : 0, emit, lit, fog, mode || 0);
        b.tex.push(tt.id);
      }
      quads++;
    }
    const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    // 任意の四角形。p0 左下, p1 右下, p2 右上, p3 左上（見える側から見て）。既定は 1 texel = 1 art px で敷きつめる。
    function quad(t, p0, p1, p2, p3, o = {}) {
      let col;
      if (o.shade === false) col = [1, 1, 1];
      else if (typeof o.shade === "number") col = [o.shade, o.shade, o.shade];
      else {
        const e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]], e2 = [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]];
        let n = norm([e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]);
        if (o.normal) n = norm(o.normal);
        else {
          // 見える側を向かせる: 上向き優先、次に手前（+y）、横向きの壁はフィールドの中心側
          if (Math.abs(n[2]) > 0.02 ? n[2] < 0 : Math.abs(n[1]) > 0.02 ? n[1] < 0 : (n[0] > 0) !== ((p0[0] + p1[0]) / 2 < fcx)) n = [-n[0], -n[1], -n[2]];
        }
        col = shadeOf(n);
      }
      if (o.color !== undefined) { const c = rgb(o.color); col = [col[0] * c[0], col[1] * c[1], col[2] * c[2]]; }
      const tt = t && !t.flat ? t : null;
      if (!tt || o.stretch) {
        const r = o.uv || [0, 0, 1, 1];
        let UV = [[r[0], r[3]], [r[2], r[3]], [r[2], r[1]], [r[0], r[1]]];
        if (o.flip) UV = [UV[1], UV[0], UV[3], UV[2]];
        push(t, [p0, p1, p2, p3], UV, null, o, col, 0);
        return;
      }
      const texel = o.texel || 1;
      const lenU = Math.hypot(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]), lenV = Math.hypot(p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]);
      const tu = o.tile ? o.tile[0] : lenU / (tt.w * texel), tv = o.tile ? o.tile[1] : lenV / (tt.h * texel);
      const nu = Math.max(1, Math.ceil(tu - 1e-4)), nv = Math.max(1, Math.ceil(tv - 1e-4));
      if (nu * nv > 24000) { console.warn("Scene3D: too many tiles", nu, nv); push(t, [p0, p1, p2, p3], [[0, 1], [1, 1], [1, 0], [0, 0]], null, o, col, 0); return; }
      const at = (s, r) => lerp3(lerp3(p0, p1, s), lerp3(p3, p2, s), r);
      for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
        const s0 = i / tu, s1 = Math.min(1, (i + 1) / tu), r0 = j / tv, r1 = Math.min(1, (j + 1) / tv);
        const fu = (s1 - s0) * tu, fv = (r1 - r0) * tv;
        push(t, [at(s0, r0), at(s1, r0), at(s1, r1), at(s0, r1)], [[0, 1], [fu, 1], [fu, 1 - fv], [0, 1 - fv]], null, o, col, 0);
      }
    }
    // 垂直の壁: 地面の線 (x0,y0)→(x1,y1)、高さ z0→z1
    const wall = (t, x0, y0, x1, y1, z0, z1, o = {}) => quad(t, [x0, y0, z0], [x1, y1, z0], [x1, y1, z1], [x0, y0, z1], o);
    // 水平の面: 高さ z の長方形（テクスチャの上 = 奥）
    const floor = (t, x0, y0, x1, y1, z, o = {}) => quad(t, [x0, y1, z], [x1, y1, z], [x1, y0, z], [x0, y0, z], o);
    // 直方体。x,y = 左奥の角、w = 幅(x)、d = 奥行き(y)、z0 = 底、h = 高さ。faces = { top, front, left, right, back, all }（テクスチャ or S.flat(色)）
    function box(x, y, w, d, z0, h, faces = {}, o = {}) {
      const g = (k) => (faces[k] !== undefined ? faces[k] : faces.side !== undefined && k !== "top" ? faces.side : faces.all);
      const z1 = z0 + h, x1 = x + w, y1 = y + d;
      if (g("top") !== null) floor(g("top"), x, y, x1, y1, z1, o);
      if (g("front") !== null) wall(g("front"), x, y1, x1, y1, z0, z1, o);
      if (g("left") !== null) wall(g("left"), x, y, x, y1, z0, z1, Object.assign({}, o, { normal: [-1, 0, 0] }));
      if (g("right") !== null) wall(g("right"), x1, y1, x1, y, z0, z1, Object.assign({}, o, { normal: [1, 0, 0] }));
      if (faces.back) wall(faces.back, x1, y, x, y, z0, z1, Object.assign({}, o, { normal: [0, -1, 0] }));
      if (o.shadow) castShadow(x, y, w, d, z1, typeof o.shadow === "number" ? o.shadow : 0.3, o.layer);
    }
    // 地面に落ちる影（直方体の足もと＋光の向きへ伸びた六角形）
    function castShadow(x, y, w, d, h, alpha = 0.38, o = {}) {
      const oo = typeof o === "string" ? { layer: o } : (o || {});
      const L = cfg.light, dx = -L[0] / L[2] * h, dy = -L[1] / L[2] * h;
      const P = dx >= 0 && dy >= 0
        ? [[x, y], [x + w, y], [x + w + dx, y + dy], [x + w + dx, y + d + dy], [x + dx, y + d + dy], [x, y + d]]
        : [[x, y], [x + w, y], [x + w, y + d], [x, y + d]];
      shadowPoly(P, alpha, oo);
    }
    function shadowPoly(pts, alpha = 0.42, o = {}) {
      if (!pts || pts.length < 3) return;
      const z = o.z === undefined ? 0.16 : o.z;
      const so = { blend: "alpha", alpha, shade: false, color: o.color || cfg.castCol || [0.03, 0.05, 0.13], lit: 0, fog: o.fog === undefined ? 1 : o.fog, layer: o.layer };
      const p0 = [pts[0][0], pts[0][1], z];
      for (let i = 1; i < pts.length - 1; i += 2) {
        const p1 = [pts[i][0], pts[i][1], z];
        const p2 = [pts[i + 1][0], pts[i + 1][1], z];
        const p3 = [pts[Math.min(pts.length - 1, i + 2)][0], pts[Math.min(pts.length - 1, i + 2)][1], z];
        push(white, [p0, p1, p2, p3], [[0, 0], [0, 0], [0, 0], [0, 0]], null, so, rgb(so.color), 0);
      }
    }
    function castConeShadow(cx, cy, rx, h, alpha = 0.44, o = {}) {
      const L = cfg.light, dx = -L[0] / L[2] * h, dy = -L[1] / L[2] * h, len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len, px = -uy * rx, py = ux * rx, round = o.round;
      if (round) {
        const shX = cx + dx * 0.72, shY = cy + dy * 0.72, rr = rx * (typeof round === "number" ? round : 0.95);
        shadowPoly([
          [cx - ux * rx * 0.65, cy - uy * rx * 0.65],
          [cx - px, cy - py],
          [shX - px * 0.85, shY - py * 0.85],
          [shX + ux * rr, shY + uy * rr],
          [shX + px * 0.85, shY + py * 0.85],
          [cx + px, cy + py],
        ], alpha, o);
      } else {
        shadowPoly([[cx - ux * rx * 0.55, cy - uy * rx * 0.55], [cx - px, cy - py], [cx + dx, cy + dy], [cx + px, cy + py]], alpha, o);
      }
    }
    // 角柱・円柱・円すい: 中心 (x,y)、下の半径 r0、上の半径 r1、底 z0、高さ h、sides 面
    function prism(t, x, y, r0, r1, z0, h, sides = 8, o = {}) {
      const top = o.top === undefined ? t : o.top, ry = o.ry === undefined ? 1 : o.ry, rot = o.rot || 0;
      for (let i = 0; i < sides; i++) {
        const a0 = rot + (i / sides) * 6.28318, a1 = rot + ((i + 1) / sides) * 6.28318, am = (a0 + a1) / 2;
        if (Math.sin(am) < -0.35 && !o.all) continue; // 奥を向いた面は見えない
        const p0 = [x + Math.cos(a1) * r0, y + Math.sin(a1) * r0 * ry, z0], p1 = [x + Math.cos(a0) * r0, y + Math.sin(a0) * r0 * ry, z0];
        const p2 = [x + Math.cos(a0) * r1, y + Math.sin(a0) * r1 * ry, z0 + h], p3 = [x + Math.cos(a1) * r1, y + Math.sin(a1) * r1 * ry, z0 + h];
        const sl = (r0 - r1) / Math.max(1e-3, h);
        quad(t, p0, p1, p2, p3, Object.assign({ stretch: true, uv: [i / sides, 0, (i + 1) / sides, 1] }, o, { normal: [Math.cos(am), Math.sin(am), sl] }));
      }
      if (r1 > 0.01 && top !== null) for (let i = 0; i < sides; i += 2) {
        const pt = (k) => { const a = rot + (k / sides) * 6.28318; return [x + Math.cos(a) * r1, y + Math.sin(a) * r1 * ry, z0 + h]; };
        quad(top, pt(i), pt(i + 1), pt(i + 2), [x, y, z0 + h], Object.assign({ stretch: true }, o, { normal: [0, 0, 1] }));
      }
    }
    // 画面に正対する板（木・人・看板など）。足もとの中心 (x, y, z)、大きさ w × h art px。モンスターと同じ縮尺で、高さは縮まない。
    function sprite(t, x, y, z, w, h, o = {}) {
      const col = o.color !== undefined ? rgb(o.color) : [1, 1, 1];
      const r = o.uv || [0, 0, 1, 1];
      let UV = [[r[0], r[3]], [r[2], r[3]], [r[2], r[1]], [r[0], r[1]]];
      if (o.flip) UV = [UV[1], UV[0], UV[3], UV[2]];
      const oo = o.anim && o.anim.type === "sway" && o.anim.top === undefined ? Object.assign({}, o, { anim: Object.assign({ top: true }, o.anim) }) : o;
      push(t, [[x - w / 2, y, z], [x + w / 2, y, z], [x + w / 2, y, z], [x - w / 2, y, z]], UV, [0, 0, h, h], oo, col, 0);
    }
    // 光のにじみ（加算。画面に正対する円）。中心 (x, y, z)、半径 r
    function glow(x, y, z, r, color, o = {}) {
      const oo = Object.assign({ blend: "add", lit: 0, fog: 0.5, alpha: 0.6 }, o);
      push(white, [[x - r, y, z], [x + r, y, z], [x + r, y, z], [x - r, y, z]], [[0, 1], [1, 1], [1, 0], [0, 0]], [-r, -r, r, r], oo, rgb(color), 1);
    }
    // 光の筋（加算）。上の点 (x0,y0,z0) 幅 w0 から、下の点 (x1,y1,z1) 幅 w1 へ。下ほど淡い
    function shaft(x0, y0, z0, x1, y1, z1, w0, w1, color, o = {}) {
      const oo = Object.assign({ blend: "add", lit: 0, fog: 0.3, alpha: 0.25 }, o);
      push(white, [[x1 - w1 / 2, y1, z1], [x1 + w1 / 2, y1, z1], [x0 + w0 / 2, y0, z0], [x0 - w0 / 2, y0, z0]], [[0, 1], [1, 1], [1, 0], [0, 0]], null, oo, rgb(color), 2);
    }
    // 地面のやわらかい丸い影（または色のにじみ）。中心 (x, y)、半径 rx, ry
    function shadow(x, y, rx, ry, alpha = 0.35, o = {}) {
      const z = o.z === undefined ? 0.2 : o.z;
      const oo = Object.assign({ blend: "alpha", lit: 0, alpha }, o);
      push(white, [[x - rx, y + ry, z], [x + rx, y + ry, z], [x + rx, y - ry, z], [x - rx, y - ry, z]], [[0, 1], [1, 1], [1, 0], [0, 0]], null, oo, rgb(o.color, [0.02, 0.03, 0.08]), 1);
    }
    // アリーナの絵（0..aw × 0..ah）の外側の地面。アリーナと重ならないよう 4 枚に分けて敷く
    function ground(t, x0, y0, x1, y1, o = {}) {
      const z = o.z || 0, aw = A.aw, ah = A.ah;
      if (y0 < 0) floor(t, x0, y0, x1, 0, z, o);
      if (y1 > ah) floor(t, x0, ah, x1, y1, z, Object.assign({ layer: "back" }, o));
      if (x0 < 0) floor(t, x0, 0, 0, ah, z, o);
      if (x1 > aw) floor(t, aw, 0, x1, ah, z, o);
    }
    // 遠景の絵（垂直の大きな板。光・影の計算なし）
    const backdrop = (t, x0, x1, y, z0, z1, o = {}) => wall(t, x0, y, x1, y, z0, z1, Object.assign({ stretch: true, shade: false, lit: 0, fog: 0 }, o));

    // 地面を照らす灯り（ライトマップに足す。モンスターと地面と立体が照らされる）。x, y = art px、r = 届く半径、i = 強さ 0..1
    function lamp(x, y, r, color, i = 0.5, o = {}) {
      if (A._s3lamps === "done") return;
      const c = rgb(color).map((v) => Math.round(v * 255));
      A.lights.push(Object.assign({ x: x * 2, y: y * 2, r: r * 2, color: c, i }, o));
    }
    // ---- reusable volumetric 3D diorama builders ------------------------------------------------------
    const _tc = {};
    const ctex = (k, w, h, fn, o) => _tc[k] || (_tc[k] = tex(w, h, fn, o));
    function pine3D(cx, cy, z0 = 0, o = {}) {
      const h = o.h || 42, r = o.r || 13, tiers = o.tiers || 4, pal = o.pal || ["#163c24", "#245d2d", "#458e36", "#74c244"];
      const tKey = "pineBough:" + pal.join(",");
      const boughT = ctex(tKey, 48, 16, (g) => {
        for (let y = 0; y < 16; y++) for (let x = 0; x < 48; x++) {
          const rib = ((x * 5 + y * 3) % 7 === 0), jag = (x % 3 === 0 ? 1 : x % 3 === 1 ? 0 : 2);
          if (y >= 13 + jag) continue;
          const v = y < 4 ? 3 : y < 9 ? (rib ? 3 : 2) : y < 12 ? (rib ? 2 : 1) : 0;
          g.fillStyle = pal[v]; g.fillRect(x, y, 1, 1);
        }
      });
      const trunkT = ctex("trunk:" + (o.trunk || "#5a3c24"), 12, 16, (g) => {
        g.fillStyle = o.trunk || "#5a3c24"; g.fillRect(0, 0, 12, 16);
        g.fillStyle = "#3a2414"; for (let i = 0; i < 12; i += 3) g.fillRect(i, 0, 1, 16);
      });
      const trH = Math.max(4, h * 0.24);
      prism(trunkT, cx, cy, Math.max(1.5, r * 0.2), Math.max(1.2, r * 0.15), z0, trH, 6, { top: null, layer: o.layer });
      for (let i = 0; i < tiers; i++) {
        const t0 = i / tiers, t1 = (i + 1) / tiers;
        const zb = z0 + h * (0.15 + t0 * 0.68), th = h * (0.28 + (1 - t0) * 0.12);
        const r0 = r * (1.0 - t0 * 0.62), r1 = i === tiers - 1 ? 0.4 : r * (0.34 - t1 * 0.24);
        const sh = 0.85 + 0.18 * t0;
        prism(boughT, cx, cy, r0, Math.max(0.2, r1), zb, th, 8, { ry: 0.9, rot: (i % 2) * 0.22, top: i === tiers - 1 ? null : flat(pal[2]), color: [sh, sh, sh], anim: o.anim, layer: o.layer });
      }
      if (o.shadow !== false) castConeShadow(cx, cy, r * 0.88, h * 0.96, typeof o.shadow === "number" ? o.shadow : 0.44, { z: z0 + 0.16, layer: o.layer });
    }
    function tree3D(cx, cy, z0 = 0, o = {}) {
      const kind = o.kind || "oak", h = o.h || 42, r = o.r || 15;
      if (kind === "cypress") { pine3D(cx, cy, z0, Object.assign({ r: r * 0.58, tiers: 5, pal: o.pal || ["#1b3f28", "#2d6034", "#49863d", "#6dae4b"] }, o)); return; }
      const trunkT = ctex("trunk:" + (o.trunk || "#5e3e26"), 12, 16, (g) => {
        g.fillStyle = o.trunk || "#5e3e26"; g.fillRect(0, 0, 12, 16);
        g.fillStyle = "#3c2614"; for (let i = 0; i < 12; i += 3) g.fillRect(i, 0, 1, 16);
        g.fillStyle = "#7a5436"; for (let i = 1; i < 12; i += 4) g.fillRect(i, 1, 1, 10);
      });
      if (kind === "palm") {
        const segs = 5, lean = o.lean || 1, swayDir = o.dir || 1, fPal = o.pal || ["#285a26", "#3f8432", "#67b240", "#9ad858"];
        const frondT = ctex("palmF:" + fPal.join(","), 24, 10, (g) => {
          for (let x = 0; x < 24; x++) {
            const w = Math.round(3.5 * Math.sin((x / 23) * Math.PI) + 1);
            for (let y = 5 - w; y <= 5 + w; y++) {
              if (x > 2 && x % 2 === 1 && (y <= 5 - w || y >= 5 + w)) continue;
              g.fillStyle = fPal[x < 6 ? 3 : y < 5 ? 2 : 1]; g.fillRect(x, y, 1, 1);
            }
          }
        });
        let px = cx, py = cy;
        for (let i = 0; i < segs; i++) {
          const z1 = z0 + (h * 0.88 * i) / segs, sh = (h * 0.88) / segs + 0.5;
          const r0 = r * (0.28 - i * 0.03), r1 = r * (0.25 - i * 0.03);
          prism(trunkT, px, py, r0, r1, z1, sh, 6, { top: null, layer: o.layer });
          px += swayDir * lean * (0.8 + i * 0.45); py += lean * 0.25;
        }
        const zTop = z0 + h * 0.88;
        prism(flat("#5c3a1e"), px, py, 2.2, 1.5, zTop - 2, 3, 6, { top: flat("#6b4423"), layer: o.layer });
        const frLen = r * 1.45, nFr = 6;
        for (let i = 0; i < nFr; i++) {
          const a = (i / nFr) * 6.28318 + (o.rot || 0.25), ca = Math.cos(a), sa = Math.sin(a), perp = [-sa * 3.5, ca * 3.5];
          const mid = [px + ca * frLen * 0.55, py + sa * frLen * 0.55, zTop + h * 0.16];
          const tip = [px + ca * frLen, py + sa * frLen, zTop - h * 0.06];
          const pA0 = [px - perp[0] * 0.5, py - perp[1] * 0.5, zTop], pA1 = [mid[0] - perp[0], mid[1] - perp[1], mid[2]], pA2 = [mid[0] + perp[0], mid[1] + perp[1], mid[2]], pA3 = [px + perp[0] * 0.5, py + perp[1] * 0.5, zTop];
          quad(frondT, pA0, pA1, pA2, pA3, { stretch: true, shade: false, color: shadeOf([ca * 0.3, sa * 0.3, 0.92]), anim: o.anim, layer: o.layer });
          const pB0 = [mid[0] - perp[0], mid[1] - perp[1], mid[2]], pB1 = [tip[0] - perp[0] * 0.65, tip[1] - perp[1] * 0.65, tip[2]], pB2 = [tip[0] + perp[0] * 0.65, tip[1] + perp[1] * 0.65, tip[2]], pB3 = [mid[0] + perp[0], mid[1] + perp[1], mid[2]];
          quad(frondT, pB0, pB1, pB2, pB3, { stretch: true, shade: false, color: shadeOf([ca * 0.65, sa * 0.65, 0.5]), anim: o.anim, layer: o.layer });
        }
        if (o.shadow !== false) castConeShadow(cx, cy, r * 0.8, h * 1.02, typeof o.shadow === "number" ? o.shadow : 0.4, { round: 1.05, z: z0 + 0.16, layer: o.layer });
        return;
      }
      const pal = o.pal || ["#1c4a26", "#2f6e30", "#54a23c", "#84cf50"];
      const leafT = ctex("oakLeaf:" + pal.join(",") + ":" + (o.blossom || ""), 36, 20, (g) => {
        const rn = U.seeded(4211);
        for (let y = 0; y < 20; y++) for (let x = 0; x < 36; x++) {
          const clump = Math.sin(x * 0.7) * Math.cos(y * 0.8);
          if (y > 16 && (x % 3 === 0 || rn() < 0.45)) continue;
          const v = y < 5 ? (clump > 0 ? 3 : 2) : y < 12 ? (clump > -0.2 ? 2 : 1) : (clump > 0.25 ? 1 : 0);
          g.fillStyle = pal[v]; g.fillRect(x, y, 1, 1);
          if (o.blossom && rn() < 0.08 && y < 14) { g.fillStyle = o.blossom; g.fillRect(x, y, 1, 1); }
        }
      });
      const trH = h * 0.42;
      prism(trunkT, cx, cy, Math.max(2, r * 0.22), Math.max(1.4, r * 0.14), z0, trH, 6, { top: null, layer: o.layer });
      const crownZ = z0 + trH * 0.8;
      // Volumetric multi-lobe 3D canopy dome (belly + sunlit crown + side clusters)
      prism(leafT, cx, cy, r * 0.72, r * 1.0, crownZ, h * 0.24, 8, { top: null, ry: 0.88, color: [0.82, 0.86, 0.92], anim: o.anim, layer: o.layer });
      prism(leafT, cx, cy, r * 1.0, r * 0.56, crownZ + h * 0.22, h * 0.24, 8, { top: flat(pal[3]), ry: 0.88, anim: o.anim, layer: o.layer });
      prism(leafT, cx, cy, r * 0.56, 0.5, crownZ + h * 0.44, h * 0.16, 8, { top: null, ry: 0.88, color: [1.06, 1.06, 1.0], anim: o.anim, layer: o.layer });
      prism(leafT, cx - r * 0.42, cy + r * 0.14, r * 0.48, r * 0.26, crownZ + h * 0.08, h * 0.24, 6, { top: flat(pal[2]), ry: 0.88, anim: o.anim, layer: o.layer });
      prism(leafT, cx + r * 0.42, cy - r * 0.08, r * 0.46, r * 0.24, crownZ + h * 0.1, h * 0.22, 6, { top: flat(pal[2]), ry: 0.88, color: [0.84, 0.86, 0.92], anim: o.anim, layer: o.layer });
      if (o.shadow !== false) castConeShadow(cx, cy, r * 0.96, h * 0.92, typeof o.shadow === "number" ? o.shadow : 0.44, { round: true, z: z0 + 0.16, layer: o.layer });
    }
    function house3D(x, y, z0 = 0, o = {}) {
      const w = o.w || 48, d = o.d || 34, h = o.h || 24, rh = o.rh || 14, ov = o.ov === undefined ? 3 : o.ov, plinth = o.plinth === undefined ? 3 : o.plinth;
      const ridge = o.ridge || "ew";
      const wallCol = o.wall || "#e4dfd2", trimCol = o.trim || "#7e8a9c", roofCol = o.roof || "#c84b34", roofHi = o.roofHi || "#ea794a", roofSh = o.roofSh || "#802b24";
      const winCol = o.win || "#4076b4", doorCol = o.door || "#7a5234", lit = !!o.lit, masonry = !!o.masonry;
      const wK = `hw:${w}:${h}:${wallCol}:${trimCol}:${winCol}:${doorCol}:${lit}:${masonry}:${o.doorSide || "l"}`;
      const frontT = ctex(wK + ":f", Math.max(24, w), Math.max(16, h), (g, W, H) => {
        g.fillStyle = wallCol; g.fillRect(0, 0, W, H);
        // Siding planks or brick courses
        g.fillStyle = masonry ? "rgba(40,42,52,0.16)" : "rgba(50,60,78,0.13)";
        for (let py = 3; py < H; py += 3) g.fillRect(0, py, W, 1);
        if (masonry) for (let py = 0; py < H; py += 6) for (let px = (py % 12 ? 3 : 0); px < W; px += 8) g.fillRect(px, py, 1, 3);
        // Framing corner trims
        g.fillStyle = trimCol; g.fillRect(0, 0, 2, H); g.fillRect(W - 2, 0, 2, H);
        if (H >= 20) { g.fillStyle = trimCol; g.fillRect(2, Math.floor(H * 0.46), W - 4, 2); }
        // Door
        const dw = Math.min(10, Math.floor(W * 0.22)), dh = Math.min(13, Math.floor(H * 0.5)), dx = o.doorSide === "c" ? Math.floor((W - dw) / 2) : o.doorSide === "r" ? W - dw - 7 : 6;
        g.fillStyle = "#3a281c"; g.fillRect(dx - 1, H - dh - 1, dw + 2, dh + 1);
        g.fillStyle = doorCol; g.fillRect(dx, H - dh, dw, dh);
        g.fillStyle = lit ? "#ffd688" : "#d8c49a"; g.fillRect(dx + dw - 3, H - Math.floor(dh * 0.45), 1, 2);
        // Framed windows (upper floor & ground floor beside door)
        const drawWin = (wx, wy, ww, wh) => {
          if (wx < 3 || wx + ww > W - 3 || wy < 3 || wy + wh > H - 2) return;
          g.fillStyle = trimCol; g.fillRect(wx - 1, wy - 1, ww + 2, wh + 2);
          g.fillStyle = lit ? "#ffb84c" : winCol; g.fillRect(wx, wy, ww, wh);
          if (lit) { g.fillStyle = "#fff2b2"; g.fillRect(wx + 1, wy + 1, Math.max(1, ww - 2), Math.max(1, wh - 2)); }
          else { g.fillStyle = "#8bc8f8"; g.fillRect(wx + 1, wy + 1, Math.max(1, ww >> 1), Math.max(1, wh - 2)); }
          g.fillStyle = trimCol; g.fillRect(wx + Math.floor(ww / 2), wy, 1, wh); g.fillRect(wx, wy + Math.floor(wh / 2), ww, 1);
          g.fillStyle = "#2b303c"; g.fillRect(wx - 1, wy + wh, ww + 2, 1);
        };
        if (H >= 20) {
          const uw = Math.min(10, Math.floor((W - 14) / 2)), uy = Math.floor(H * 0.18);
          drawWin(6, uy, uw, 5);
          if (W >= 34) drawWin(W - 6 - uw, uy, uw, 5);
        }
        const gwX = o.doorSide === "l" ? dx + dw + 5 : 5, gwW = Math.min(12, W - gwX - 5);
        if (gwW >= 6) drawWin(gwX, H - 10, gwW, 5);
        // Crisp dark overhang shadow under the eaves!
        g.fillStyle = "rgba(10,16,32,0.55)"; g.fillRect(0, 0, W, 3);
      });
      const sideT = ctex(wK + `:s:${d}`, Math.max(16, d), Math.max(16, h), (g, W, H) => {
        g.fillStyle = wallCol; g.fillRect(0, 0, W, H);
        g.fillStyle = "rgba(35,45,65,0.16)"; for (let py = 3; py < H; py += 3) g.fillRect(0, py, W, 1);
        g.fillStyle = trimCol; g.fillRect(0, 0, 2, H); g.fillRect(W - 2, 0, 2, H);
        const wx = Math.floor(W * 0.3), ww = Math.max(5, Math.floor(W * 0.4)), wy = Math.floor(H * 0.25);
        if (ww >= 4 && wy + 6 < H) {
          g.fillStyle = trimCol; g.fillRect(wx - 1, wy - 1, ww + 2, 7);
          g.fillStyle = lit ? "#ffc966" : winCol; g.fillRect(wx, wy, ww, 5);
          g.fillStyle = trimCol; g.fillRect(wx + (ww >> 1), wy, 1, 5);
        }
        g.fillStyle = "rgba(10,16,32,0.58)"; g.fillRect(0, 0, W, 3);
      });
      const roofT = ctex(`hr:${w}:${d}:${roofCol}:${roofHi}:${roofSh}`, 32, 24, (g, W, H) => {
        g.fillStyle = roofCol; g.fillRect(0, 0, W, H);
        for (let py = 0; py < H; py += 4) {
          g.fillStyle = roofSh; g.fillRect(0, py + 3, W, 1);
          g.fillStyle = roofHi; g.fillRect(0, py, W, 1);
          for (let px = (py & 4 ? 2 : 0); px < W; px += 4) { g.fillStyle = roofSh; g.fillRect(px, py, 1, 3); }
        }
        g.fillStyle = roofSh; g.fillRect(0, H - 2, W, 2);
      });
      let zb = z0;
      if (plinth > 0) { box(x - 1, y - 1, w + 2, d + 2, z0, plinth, { all: flat("#6c7486"), top: flat("#8e98aa") }, { layer: o.layer }); zb += plinth; }
      box(x, y, w, d, zb, h, { front: frontT, left: sideT, right: sideT, back: sideT, top: null }, { layer: o.layer });
      const zTop = zb + h, zPeak = zTop + rh;
      if (ridge === "ns") {
        const xm = x + w * 0.5;
        quad(roofT, [x - ov, y + d + ov, zTop - 1], [xm, y + d + ov, zPeak], [xm, y - ov, zPeak], [x - ov, y - ov, zTop - 1], { normal: [-rh, 0, w * 0.5 + ov], texel: 0.75, layer: o.layer });
        quad(roofT, [xm, y + d + ov, zPeak], [x + w + ov, y + d + ov, zTop - 1], [x + w + ov, y - ov, zTop - 1], [xm, y - ov, zPeak], { normal: [rh, 0, w * 0.5 + ov], texel: 0.75, layer: o.layer });
        box(xm - 1.5, y - ov, 3, d + ov * 2, zPeak - 0.8, 1.8, { all: flat(roofHi), right: flat(roofSh) }, { layer: o.layer });
        quad(flat(wallCol), [x, y + d, zTop], [x + w, y + d, zTop], [xm, y + d, zPeak], [xm, y + d, zPeak], { normal: [0, 1, 0], layer: o.layer });
      } else {
        const ym = y + d * 0.5;
        quad(roofT, [x - ov, y + d + ov, zTop - 1], [x + w + ov, y + d + ov, zTop - 1], [x + w + ov, ym, zPeak], [x - ov, ym, zPeak], { normal: [0, rh, d * 0.5 + ov], texel: 0.75, layer: o.layer });
        quad(roofT, [x - ov, ym, zPeak], [x + w + ov, ym, zPeak], [x + w + ov, y - ov, zTop - 1], [x - ov, y - ov, zTop - 1], { normal: [0, -rh, d * 0.5 + ov], texel: 0.75, layer: o.layer });
        box(x - ov, ym - 1.5, w + ov * 2, 3, zPeak - 0.8, 1.8, { all: flat(roofHi), front: flat(roofCol) }, { layer: o.layer });
        wall(flat(roofSh), x - ov, y + d + ov, x + w + ov, y + d + ov, zTop - 2.2, zTop - 0.8, { normal: [0, 1, 0], layer: o.layer });
        quad(flat(wallCol), [x, y, zTop], [x, y + d, zTop], [x, ym, zPeak], [x, ym, zPeak], { normal: [-1, 0, 0], layer: o.layer });
        quad(flat(wallCol), [x + w, y + d, zTop], [x + w, y, zTop], [x + w, ym, zPeak], [x + w, ym, zPeak], { normal: [1, 0, 0], layer: o.layer });
      }
      if (o.chimney) {
        const chx = x + w * 0.72, chy = y + d * 0.28, chZ = zPeak + 4;
        box(chx, chy, 6, 6, zTop, chZ - zTop, { all: flat("#884a42") }, { layer: o.layer });
        box(chx - 1, chy - 1, 8, 8, chZ, 2, { all: flat("#56302c"), top: flat("#a0584e") }, { layer: o.layer });
      }
      if (lit) {
        // Warm emissive glowing panes protruding 0.25px in front of front wall so lit cottage windows always shine!
        const wy2 = zb + h * 0.55;
        wall(ctex("litW", 8, 6, (g) => { g.fillStyle = "#ffc256"; g.fillRect(0, 0, 8, 6); g.fillStyle = "#fff4b8"; g.fillRect(1, 1, 6, 4); }), x + 6, y + d + 0.3, x + 6 + Math.min(10, Math.floor((w - 14) / 2)), y + d + 0.3, wy2, wy2 + 5, { emit: 0.95, lit: 0, shade: false, layer: o.layer });
        if (w >= 34) wall(ctex("litW"), x + w - 6 - Math.min(10, Math.floor((w - 14) / 2)), y + d + 0.3, x + w - 6, y + d + 0.3, wy2, wy2 + 5, { emit: 0.95, lit: 0, shade: false, layer: o.layer });
        glow(x + w * 0.5, y + d + 2, zb + h * 0.45, 20, "#ffb84c", { alpha: 0.24, layer: o.layer });
        lamp(x + w * 0.5, y + d + 8, 56, [255, 195, 100], 0.5);
      }
      if (o.shadow !== false) castShadow(x - 1, y - 1, w + 2, d + 2, plinth + h + rh * 0.72, typeof o.shadow === "number" ? o.shadow : 0.48, { z: z0 + 0.16, layer: o.layer });
    }
    function fence3D(x0, y0, x1, y1, z0 = 0, o = {}) {
      const h = o.h || 9, col = o.col || "#9a6c3e", postCol = o.postCol || "#7a522c", cap = o.cap || "#c4945c", step = o.step || 14;
      const len = Math.hypot(x1 - x0, y1 - y0), n = Math.max(1, Math.round(len / step));
      const railT = ctex(`rail:${col}:${cap}`, 16, Math.max(8, h), (g, W, H) => {
        g.fillStyle = col; g.fillRect(0, 2, W, 2); g.fillRect(0, Math.floor(H * 0.6), W, 2);
        g.fillStyle = cap; g.fillRect(0, 2, W, 1); g.fillRect(0, Math.floor(H * 0.6), W, 1);
        for (let px = 0; px < W; px += 4) { g.fillStyle = col; g.fillRect(px + 1, 1, 2, H - 1); g.fillStyle = cap; g.fillRect(px + 1, 1, 2, 1); }
      });
      wall(railT, x0, y0, x1, y1, z0, z0 + h, { layer: o.layer });
      for (let i = 0; i <= n; i++) {
        const t = i / n, px = x0 + (x1 - x0) * t, py = y0 + (y1 - y0) * t;
        box(px - 1.5, py - 1.5, 3, 3, z0, h + 1.5, { all: flat(postCol), top: flat(cap) }, { layer: o.layer, shadow: o.shadow === false ? false : 0.34 });
      }
    }
    function rock3D(cx, cy, z0 = 0, o = {}) {
      const r = o.r || 10, h = o.h || 11, sides = o.sides || 6, pal = o.pal || ["#4a5264", "#6e788c", "#98a4b8"];
      const rKey = "rock:" + pal.join(",");
      const rt = ctex(rKey, 24, 16, (g) => {
        g.fillStyle = pal[1]; g.fillRect(0, 0, 24, 16);
        g.fillStyle = pal[2]; g.fillRect(0, 0, 24, 3); g.fillRect(4, 5, 5, 3); g.fillRect(15, 7, 4, 2);
        g.fillStyle = pal[0]; g.fillRect(0, 12, 24, 4); g.fillRect(9, 4, 3, 6);
      });
      prism(rt, cx, cy, r, r * 0.68, z0, h * 0.65, sides, { rot: o.rot || 0.3, top: null, layer: o.layer });
      prism(rt, cx - r * 0.08, cy - r * 0.06, r * 0.68, r * 0.25, z0 + h * 0.62, h * 0.38, sides, { rot: (o.rot || 0.3) + 0.2, top: flat(pal[2]), layer: o.layer });
      if (o.shadow !== false) castConeShadow(cx, cy, r * 0.88, h, typeof o.shadow === "number" ? o.shadow : 0.42, { round: true, z: z0 + 0.16, layer: o.layer });
    }
    function lamp3D(x, y, z0 = 0, o = {}) {
      const h = o.h || 26, post = o.post || "#2a303e", col = o.color || "#ffcf6a", r = o.r || 48;
      box(x - 2, y - 2, 4, 4, z0, 4, { all: flat(post), top: flat("#4e586c") }, { layer: o.layer });
      prism(flat(post), x, y, 1.2, 0.9, z0 + 4, h - 7, 5, { top: null, layer: o.layer });
      box(x - 2.5, y - 2.5, 5, 5, z0 + h - 3, 4.5, { all: flat(col), top: flat("#566078") }, { emit: o.emit === undefined ? 0.95 : o.emit, lit: 0, layer: o.layer });
      glow(x, y, z0 + h - 1, o.glowR || 16, col, { alpha: 0.45, anim: { type: "flicker", amp: 0.25 }, layer: o.layer });
      if (o.lamp !== false) lamp(x, y, r, col, o.i === undefined ? 0.65 : o.i);
      if (o.shadow !== false) castConeShadow(x, y, 2, h, 0.36, { z: z0 + 0.16, layer: o.layer });
    }
    const S = {
      lamp,
      A, af, aw: A.aw, ah: A.ah, night: !!A.night, id: A.id, rng: U.seeded(9173 + A.id.length * 77), frontY, white, cfg,
      tex, flat, quad, wall, floor, box, prism, sprite, glow, shaft, shadow, castShadow, shadowPoly, castConeShadow, ground, backdrop,
      pine3D, tree3D, house3D, fence3D, rock3D, lamp3D,
      set(o) { for (const k in o) { if (k === "fog") Object.assign(cfg.fog, o.fog, o.fog.color !== undefined ? { color: rgb(o.fog.color) } : {}); else if (k === "clear" || k === "shadowTint" || k === "keyTint" || k === "castCol") cfg[k] = rgb(o[k]); else if (k === "light") cfg.light = norm(o.light); else if (k === "pitch") cfg.pitch = typeof o[k] === "number" && o[k] > 3 ? o[k] * DEG : o[k]; else cfg[k] = o[k]; } },
    };
    def.build(S, A);
    A._s3lamps = "done";

    // ---- atlas packing (shelf) ----
    const atl = {};
    for (const name of ATLAS) {
      const list = texs.filter((t) => t.atlas === name).sort((a, b) => b.h - a.h);
      if (!list.length) { atl[name] = null; continue; }
      let AW = 1024;
      const widest = Math.max(...list.map((t) => t.w + 4));
      while (AW < widest) AW *= 2;
      let AH, ok = false;
      for (; AW <= 8192 && !ok; AW *= 2) {
        let x = 0, y = 0, rowH = 0; ok = true;
        for (const t of list) {
          if (x + t.w + 4 > AW) { x = 0; y += rowH; rowH = 0; }
          t.ax = x + 2; t.ay = y + 2; x += t.w + 4; rowH = Math.max(rowH, t.h + 4);
        }
        AH = y + rowH;
        if (AH > AW) ok = false; else break;
      }
      if (!ok) console.warn("Scene3D: atlas overflow", A.id, name);
      const c = U.canvas(AW, Math.max(4, AH)), x = c.getContext("2d");
      x.imageSmoothingEnabled = false;
      for (const t of list) {
        x.drawImage(t.c, 0, 0, 1, t.h, t.ax - 1, t.ay, 1, t.h); x.drawImage(t.c, t.w - 1, 0, 1, t.h, t.ax + t.w, t.ay, 1, t.h);
        x.drawImage(t.c, 0, 0, t.w, 1, t.ax, t.ay - 1, t.w, 1); x.drawImage(t.c, 0, t.h - 1, t.w, 1, t.ax, t.ay + t.h, t.w, 1);
        x.drawImage(t.c, t.ax, t.ay);
        t.c = null;
      }
      atl[name] = c;
    }
    const out = { cfg, atl, batches: [], quads };
    for (const k in batches) {
      const b = batches[k], arr = new Float32Array(b.data), at = atl[b.atlas], n = b.tex.length;
      for (let i = 0; i < n; i++) {
        const t = texs[b.tex[i]], o = i * STRIDE + 3;
        arr[o] = (t.ax + arr[o] * t.w) / at.width; arr[o + 1] = (t.ay + arr[o + 1] * t.h) / at.height;
      }
      out.batches.push({ layer: b.layer, blend: b.blend, atlas: b.atlas, arr, n });
    }
    return out;
  }

  // ---- GL ------------------------------------------------------------------------------------------------
  let gl = null, prog = null, gpu = null, gpuFor = null;
  const VS = `
    attribute vec3 aPos;attribute vec2 aUV;attribute vec4 aCol;attribute vec4 aFx;attribute vec4 aMat;
    uniform mat4 uVP;uniform vec3 uUp;uniform float uTime,uHype;uniform vec3 uFog;uniform vec2 uLow;
    varying vec2 vUV;varying vec4 vCol;varying vec4 vMat;varying vec2 vG;varying float vFog;
    void main(){
      vec3 p=aPos;vec4 col=aCol;float k=aFx.x,ph=aFx.y,amp=aFx.z,lift=aFx.w;
      if(k>0.5&&k<1.5)p.x+=amp*(sin(uTime*1.6+ph)+0.4*sin(uTime*3.7+ph*1.7));
      else if(k>1.5&&k<2.5)lift+=amp*(max(0.0,sin(uTime*(7.0+uHype*5.0)+ph))*smoothstep(0.25,0.8,uHype)+0.12*sin(uTime*2.0+ph));
      else if(k>2.5&&k<3.5)col.rgb*=1.0-amp*(0.5+0.5*sin(uTime*11.0+ph)*sin(uTime*7.3+ph*2.1));
      else if(k>3.5&&k<4.5)col.rgb*=1.0-amp*(0.5+0.5*sin(uTime*1.4+ph));
      else if(k>4.5&&k<5.5)p.z+=amp*sin(uTime*0.9+ph);
      else if(k>5.5&&k<6.5)col.a*=1.0-amp*(0.5+0.5*sin(uTime*2.3+ph));
      p+=uUp*lift;
      gl_Position=uVP*vec4(p,1.0);
      vUV=aUV;vCol=col;vMat=aMat;vG=p.xy/uLow;
      vFog=clamp((gl_Position.w-uFog.x)/(uFog.y-uFog.x),0.0,1.0)*uFog.z*aMat.z;
    }`;
  const FSH = `
    precision mediump float;
    uniform sampler2D uAtlas,uLight;uniform vec3 uFogCol;uniform float uCut,uAdd,uOn;
    varying vec2 vUV;varying vec4 vCol;varying vec4 vMat;varying vec2 vG;varying float vFog;
    void main(){
      vec4 t;
      if(vMat.w<0.5)t=texture2D(uAtlas,vUV);
      else if(vMat.w<1.5){vec2 d=vUV*2.0-1.0;float r=clamp(1.0-length(d),0.0,1.0);t=vec4(1.0,1.0,1.0,r*r);}
      else{float e=1.0-abs(vUV.x*2.0-1.0);t=vec4(1.0,1.0,1.0,smoothstep(0.0,0.9,e)*(0.15+0.85*(1.0-vUV.y)*(1.0-vUV.y)));}
      float a=t.a*vCol.a;
      if(uCut>0.5){if(a<0.5)discard;a=1.0;}
      vec3 c=t.rgb*vCol.rgb;
      vec3 L=texture2D(uLight,vG).rgb*2.0;
      c*=mix(vec3(1.0),L,vMat.y);
      if(uAdd>0.5){c*=(0.2+0.8*uOn)*(1.0-vFog);gl_FragColor=vec4(c*a,1.0);return;}
      c=mix(c,uFogCol,vFog);
      gl_FragColor=vec4(c,uCut>0.5?vMat.x*max(c.r,max(c.g,c.b)):a);
    }`;
  function compile(g, vs, fs, attrs) {
    const sh = (type, src) => { const s = g.createShader(type); g.shaderSource(s, src); g.compileShader(s); if (!g.getShaderParameter(s, g.COMPILE_STATUS)) throw new Error(g.getShaderInfoLog(s)); return s; };
    const p = g.createProgram();
    g.attachShader(p, sh(g.VERTEX_SHADER, vs)); g.attachShader(p, sh(g.FRAGMENT_SHADER, fs));
    attrs.forEach((a, i) => g.bindAttribLocation(p, i, a));
    g.linkProgram(p);
    if (!g.getProgramParameter(p, g.LINK_STATUS)) throw new Error(g.getProgramInfoLog(p));
    const u = {}, n = g.getProgramParameter(p, g.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const a = g.getActiveUniform(p, i); u[a.name] = g.getUniformLocation(p, a.name); }
    return { p, u };
  }
  function initGL(g) { gl = g; prog = compile(g, VS, FSH, ["aPos", "aUV", "aCol", "aFx", "aMat"]); }
  // このアリーナの立体を GPU に用意する（アリーナが変わったら作り直す）
  function use(A) {
    if (gpuFor === A) return gpu;
    if (gpu) { for (const b of gpu.batches) gl.deleteBuffer(b.buf); for (const k in gpu.tex) if (gpu.tex[k]) gl.deleteTexture(gpu.tex[k]); }
    const t0 = performance.now();
    const sc = build(A);
    gpu = { cfg: sc.cfg, batches: [], tex: {}, quads: sc.quads };
    for (const name of ATLAS) {
      if (!sc.atl[name]) { gpu.tex[name] = null; continue; }
      const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, sc.atl[name]);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, name === "pix" ? gl.NEAREST : gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gpu.tex[name] = t;
    }
    for (const b of sc.batches) {
      const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, b.arr, gl.STATIC_DRAW);
      gpu.batches.push({ layer: b.layer, blend: b.blend, atlas: b.atlas, buf, n: b.n });
    }
    gpuFor = A;
    stats = { id: A.id, quads: sc.quads, ms: Math.round(performance.now() - t0), atlas: ATLAS.map((n) => (sc.atl[n] ? sc.atl[n].width + "x" + sc.atl[n].height : "-")).join(" ") };
    return gpu;
  }
  let stats = null;
  // layer = "back" | "front"。o = { lightTex, lowW, lowH, time, hype, on }。呼ぶ側が深度テストを有効にしておく。
  function draw(layer, o) {
    if (!gpu) return;
    const g = gl, u = prog.u, c = gpu.cfg, B = 4 * STRIDE;
    g.useProgram(prog.p);
    g.uniformMatrix4fv(u.uVP, false, V.vp); g.uniform3f(u.uUp, V.u[0], V.u[1], V.u[2]);
    g.uniform1f(u.uTime, o.time || 0); g.uniform1f(u.uHype, o.hype || 0); g.uniform1f(u.uOn, o.on === undefined ? 1 : o.on);
    g.uniform3f(u.uFog, V.d + c.fog.near, V.d + c.fog.far, c.fog.max); g.uniform3fv(u.uFogCol, o.fogCol || c.fog.color); g.uniform2f(u.uLow, o.lowW, o.lowH);
    g.activeTexture(g.TEXTURE1); g.bindTexture(g.TEXTURE_2D, o.lightTex); g.uniform1i(u.uLight, 1);
    g.uniform1i(u.uAtlas, 0);
    for (let i = 0; i < 5; i++) g.enableVertexAttribArray(i);
    for (const blend of BLENDS) for (const b of gpu.batches) {
      if (b.layer !== layer || b.blend !== blend || !b.n) continue;
      g.activeTexture(g.TEXTURE0); g.bindTexture(g.TEXTURE_2D, gpu.tex[b.atlas]);
      g.bindBuffer(g.ARRAY_BUFFER, b.buf);
      g.vertexAttribPointer(0, 3, g.FLOAT, false, B, 0); g.vertexAttribPointer(1, 2, g.FLOAT, false, B, 12); g.vertexAttribPointer(2, 4, g.FLOAT, false, B, 20);
      g.vertexAttribPointer(3, 4, g.FLOAT, false, B, 36); g.vertexAttribPointer(4, 4, g.FLOAT, false, B, 52);
      g.uniform1f(u.uCut, blend === "cut" ? 1 : 0); g.uniform1f(u.uAdd, blend === "add" ? 1 : 0);
      if (blend === "cut") { g.disable(g.BLEND); g.depthMask(true); }
      else { g.enable(g.BLEND); g.depthMask(false); if (blend === "add") g.blendFuncSeparate(g.ONE, g.ONE, g.ZERO, g.ONE); else g.blendFuncSeparate(g.SRC_ALPHA, g.ONE_MINUS_SRC_ALPHA, g.ZERO, g.ONE); }
      g.drawArrays(g.TRIANGLES, 0, b.n);
    }
    g.disable(g.BLEND); g.depthMask(true);
    for (let i = 1; i < 5; i++) g.disableVertexAttribArray(i);
  }

  return { define, get, enabled, disable: () => { OFF = true; }, has: (id) => !!DEFS[id], setCamera, project, unproject, view: V, initGL, use, draw, PITCH, FOV, stats: () => stats, cfg: () => (gpu ? gpu.cfg : null) };
})();
