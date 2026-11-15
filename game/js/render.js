// Renderer. Art-resolution buffers (1 art px = 2 world units = S device px):
//   world  – terrain, decals, crowd, trainers, shadows (the pitch stays on this coarse grid)
//   spr    – monsters at 2 sprite px per art px, so the pixelize detail is not thrown away
//   fx     – every effect, drawn in world coordinates at 0.5 scale; dithered + posterized by Post
//   light  – ambient × lights (smooth), encoded at half intensity so lights can over-brighten 2×
// Post (WebGL) composes and adds bloom / tilt-shift / fog / god rays / grading. A screen-resolution
// overlay canvas carries crisp pixel text: damage numbers, labels, mini HP bars, off-screen arrows.
"use strict";

const Renderer = (() => {
  let glc, ovc, octx, world, wctx, sprc, sctx, fxc, fctx, lightc, lctx, ctx2d = null;
  let airc, airx, airQ = 2; // 3D のときの空中エフェクトの層（画面の座標、1 ドット = airQ 画素）
  let W = 0, H = 0, S = 3, LW = 0, LH = 0, dpr = 1;
  // Monsters keep the same on-screen height and a hard pixel edge, with twice the texel density of the pitch.
  // [ユーザー指定 2026-11-15] モンスターは GIF の元のドットの細かさで描く。スプライト用キャンバスは「いま画面に映る範囲」だけを
  // SPR 倍（1 art px あたりの画素数、2〜4）で持つ。寄っているほど範囲が狭いので細かくでき、引くと自動で 3 → 2 に下がる。
  let SPR = 4, sox = 0, soy = 0, SWa = 2, SHa = 2; // window origin / size in art px (buffer-relative)
  let sprNeed = 4; // 欲しい密度（updateCamera が出場中の体の fit から決める）
  const SPR_BUDGET = 9e6; // sprite canvas pixels per frame (it is uploaded to the GPU every frame)
  // cam.user = プレイヤーが決めるズーム倍率（ホイール / - = 0 キー）。1 が標準、小さいほど引き、大きいほど寄り。
  // [ユーザー指定 2026-11-15] 100% = 追っているモンスターが front.gif と同じ大きさ（GIF の 1 ドット = 画面の 1 ドット）で映る倍率。既定は 100%。
  const USER_MIN = 0.4, USER_MAX = 3, USER_DEFAULT = 1;
  let ZOOM_KEY = "arena.camZoom.v3";
  const cam = { x: 0, y: 0, sx: 0, sy: 0, zoom: 1, zoomT: 1, user: USER_DEFAULT, userDefault: USER_DEFAULT };
  try { const v = +localStorage.getItem(ZOOM_KEY); if (v > 0) cam.user = Math.min(USER_MAX, Math.max(USER_MIN, v)); } catch (e) {}
  // BZ = 低解像度バッファが賄える最小ズーム。1 未満へ引くとき、その分だけバッファを広く取る（画面中心は W/(2·BZ)）。
  let BZ = 1;
  let ix = 0, iy = 0, fracX = 0, fracY = 0;
  const trainerSprites = [Pixel.trainer("#d9302c", "#2f62c8", "#2a2f4a"), Pixel.trainer("#222222", "#7a2a9a", "#3a3a3a", "#e8b890", "#1a1a1a")];
  const DEBUG = /\bdbg=1\b/.test(location.search);

  function init(glCanvas, overlayCanvas) {
    glc = glCanvas; ovc = overlayCanvas; octx = ovc.getContext("2d");
    world = U.canvas(2, 2); wctx = world.getContext("2d");
    sprc = U.canvas(2, 2); sctx = sprc.getContext("2d");
    fxc = U.canvas(2, 2); fctx = fxc.getContext("2d");
    airc = U.canvas(2, 2); airx = airc.getContext("2d");
    lightc = U.canvas(2, 2); lctx = lightc.getContext("2d");
    if (!Post.init(glc)) ctx2d = glc.getContext("2d");

    resize();
    addEventListener("resize", resize);
  }
  function resize() {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    const vw = innerWidth, vh = innerHeight;
    W = Math.round(vw * dpr); H = Math.round(vh * dpr);
    for (const c of [glc, ovc]) { c.width = W; c.height = H; c.style.width = vw + "px"; c.style.height = vh + "px"; }
    // Golden-ratio overhead stadium scale: pick whole-pixel low-res buffer factor S so
    // the full court + sidelines + upper/lower crowd stands fit with crisp integer pixels.
    S = Math.max(1, Math.min(4, Math.floor(Math.min(W / 680, H / 460))));
    alloc();
    Post.resize(W, H);
  }
  function alloc() {
    if (P3 && curArena) { LW = curArena.aw; LH = curArena.ah; } // 3D: 低解像度バッファはアリーナ全体
    else { LW = Math.ceil(W / S / BZ) + 2; LH = Math.ceil(H / S / BZ) + 2; }
    for (const c of [world, fxc, lightc]) { c.width = LW; c.height = LH; }
    wctx.imageSmoothingEnabled = false; fctx.imageSmoothingEnabled = false;
  }
  // Sprite window for this frame: the zoomed view (+ margin), snapped to 32 art px so the canvas is not resized every frame.
  function sprWindow() {
    const z = Math.max(1e-3, cam.zoom), q = 32;
    SWa = Math.min(LW, Math.ceil((W / S / z + 6) / q) * q); SHa = Math.min(LH, Math.ceil((H / S / z + 6) / q) * q);
    const area = SWa * SHa;
    SPR = 2;
    for (let k = sprNeed; k > 2; k--) if (area * k * k <= SPR_BUDGET) { SPR = k; break; }
    sox = U.clamp(Math.round(W / S / BZ / 2 + fracX - SWa / 2), 0, LW - SWa); soy = U.clamp(Math.round(H / S / BZ / 2 + fracY - SHa / 2), 0, LH - SHa);
    if (sprc.width !== SWa * SPR || sprc.height !== SHa * SPR) { sprc.width = SWa * SPR; sprc.height = SHa * SPR; }
  }
  // Base transform of the sprite canvas: scale s, shifted so buffer art px (sox, soy) is the canvas origin.
  // [ユーザー指定 2026-11-16] 3D（Scene3D）: 地面は遠近のある面、モンスターや小物は画面に正対する板。スプライト用キャンバスは画面と同じ大きさで、
  // 1 体ごとに「足もとの地面の点が画面のどこに来るか・その奥行きでの倍率」（E3）で描く。P3 = いま 3D で描いているか。
  let P3 = false, E3 = { gx: 0, gy: 0, sx: 0, sy: 0, sc: 1 }, pitch3 = 0;
  function anchor(wx, wy) { // world 座標の足もと
    if (!P3) return true;
    const K = Arenas.K, p = Scene3D.project(wx / K, wy / K, 0);
    E3 = { gx: wx / K, gy: wy / K, sx: Math.round(p.x), sy: Math.round(p.y), sc: p.s };
    return p.ok && p.x > -W && p.x < 2 * W && p.y > -H && p.y < 2.5 * H;
  }
  const sprT = (s) => (P3
    ? sctx.setTransform(s * E3.sc / SPR, 0, 0, s * E3.sc / SPR, E3.sx - E3.gx * E3.sc, E3.sy - E3.gy * E3.sc)
    : sctx.setTransform(s, 0, 0, s, -sox * SPR, -soy * SPR));
  // その地面の点での 1 art px の画素数（平面表示では S × ズーム）
  const sclAt = (wx, wy) => (P3 ? Scene3D.project(wx / Arenas.K, wy / Arenas.K, 0).s : S * cam.zoom);
  // 画面ごとの既定倍率と保存先（既定はどちらも 100% = GIF 等倍）。保存値があればそれを使う。
  function zoomProfile(key, def) {
    ZOOM_KEY = key; cam.userDefault = def; cam.user = def;
    try { const v = +localStorage.getItem(ZOOM_KEY); if (v > 0) cam.user = Math.min(USER_MAX, Math.max(USER_MIN, v)); } catch (e) {}
  }
  function setUserZoom(v) {
    cam.user = Math.min(USER_MAX, Math.max(USER_MIN, Math.round(v * 100) / 100));
    try { localStorage.setItem(ZOOM_KEY, String(cam.user)); } catch (e) {}
    return cam.user;
  }
  const scale = () => S;

  // ---- camera --------------------------------------------------------------------------------------
  function updateCamera(B, dt, snap) {
    const A = B.arena, f = A.field;
    const vwArt = W / S, vhArt = H / S;
    // Cover full arena canvas on landscape viewports; on portrait viewports fit the full pitch + sidelines horizontally.
    const portrait = vwArt < vhArt * 0.98;
    const coverZ = portrait ? vwArt / Math.min(A.aw, A.af.w + 96) : Math.max(vwArt / A.aw, vhArt / A.ah);
    const stadiumZ = Math.min(vwArt / (A.af.w + (portrait ? 96 : 216)), vhArt / (A.af.h + 168));
    const baseZ = Math.max(coverZ, stadiumZ);

    let cx = f.cx, cy = f.cy - 2, pairSpan = 360;
    const act = B.active.filter((p) => p && (p.active || p.ko || p.out > 0));
    // [ユーザー指定 2026-11-14] カメラはモンスターに追従する。操作中（指示中）のモンスターが主役で、
    // 相手の方向へ少しだけ先回りする。操作するモンスターがいない観戦では両者の中間を追う。
    const side = B.playerSide >= 0 ? B.playerSide : B.commandSide;
    const me = side >= 0 && B.active[side] && act.includes(B.active[side]) ? B.active[side] : null;
    let follow = false;
    if (act.length === 2) {
      pairSpan = U.dist(act[0].x, act[0].y, act[1].x, act[1].y);
      if (me) {
        const foe = act[0] === me ? act[1] : act[0], lead = Math.min(0.3, 110 / Math.max(1, pairSpan));
        cx = U.lerp(me.x, foe.x, lead); cy = U.lerp(me.y, foe.y, lead) - 10;
      } else { cx = (act[0].x + act[1].x) / 2; cy = (act[0].y + act[1].y) / 2 - 10; }
      follow = true;
    } else if (act.length === 1) {
      cx = act[0].x; cy = act[0].y - 10; follow = true;
    }
    if (B.idlePan) {
      const tt = B.time || 0;
      cx = f.cx + Math.sin(tt * 0.08) * 90;
      cy = f.cy + 16;
    }

    let zoomMul = 1.0;
    const inFight = B.cfg && !B.cfg.demo && B.phase !== "intro" && !B.idlePan;
    if (inFight) {
      // Wide golden-ratio court shot in neutral; subtle breathing when fighters cross blades.
      // 操作モンスターを GIF 等倍で映すときは、立ち回り中の倍率を動かさない（等倍を保つ）
      zoomMul = me ? 1 : U.lerp(1.06, 0.99, U.clamp((pairSpan - 160) / 380, 0, 1));
    }
    // Dynamic dramatic zoom-in ONLY on specials, beam clashes, intros, and K.O. blows!
    if (B.activeClash) {
      cx = U.lerp(cx, B.activeClash.x, 0.68);
      cy = U.lerp(cy, B.activeClash.y - 14, 0.65);
      zoomMul = Math.max(zoomMul, 1.28);
    }
    if (B.camFocus && B.camFocus.t > 0) {
      cx = U.lerp(cx, B.camFocus.x, 0.62);
      cy = U.lerp(cy, B.camFocus.y - 16, 0.6);
      zoomMul = Math.max(zoomMul, B.camFocus.zoom || 1.25);
    }
    if (B.koCam && B.koCam.t > 0) {
      cx = U.lerp(cx, B.koCam.x, 0.74);
      cy = U.lerp(cy, B.koCam.y - 18, 0.72);
      zoomMul = Math.max(zoomMul, 1.38);
    }
    if (B.focus) {
      cx = B.focus.x; cy = B.focus.y - 20;
      zoomMul = Math.max(zoomMul, 1.36);
    } else if (B.cfg && B.cfg.demo) {
      cy -= 64;
    }
    if (B.phase === "end") zoomMul = Math.max(zoomMul, 1.12);
    // [ユーザー指定 2026-11-15] 自分が操作しているモンスターを、いつも画面の中心に映す（体の中心。相手側への先回りや、
    // 強打・競り合い・登場・K.O. での寄り道はしない。演出はズームだけ残す）。フィールドの端でも中心を保つため、
    // このときはアリーナの外へはみ出さないための位置制限（computeView）も外す。操作モンスターがいない観戦は従来どおり。
    const lock = !!(me && (me.active || me.out > 0) && !me.ko && me.measured && B.cfg && !B.cfg.demo && !B.idlePan);
    const use3 = !!(A.scene3d && Post.ok());
    // 3D では地面が奥行き方向に縮んで映るので、体の中心（足もとから高さの半分）を画面の中心に置くための地面上のずらし量を、そのぶん大きくする
    if (lock) { cx = me.x; cy = me.y - (me.height * 0.5 + (me.z || 0) * 0.6) / (use3 ? Math.sin(pitch3 || Scene3D.PITCH) : 1); }
    cam.lock = lock && !window.CAM_OVERRIDE;
    if (window.CAM_OVERRIDE) { cx = window.CAM_OVERRIDE.x; cy = window.CAM_OVERRIDE.y; }

    const panRate = lock ? 45 : (B.koCam || B.activeClash || B.camFocus || B.focus) ? 6.5 : follow ? 5.2 : 3.4;
    if (snap) { cam.x = cx; cam.y = cy; } else { cam.x = U.damp(cam.x, cx, panRate, dt); cam.y = U.damp(cam.y, cy, panRate, dt); }
    // [ユーザー指定 2026-11-15] 基準のズームはモンスターの大きさで決める：追っているモンスターの元のドット（front.gif の解像度）が
    // 画面の 1 ドット（CSS px）になる倍率。小さいモンスターには寄り、大きいモンスターからは引く。観戦で主役がいないときは両者の中間。
    // fit = フィールド上の art px ÷ 元のドット数（Fighter.measure）なので、1 art px を dpr / (fit·S) 倍で映せば等倍になる。
    // bigBoost（1.7m 以上を少し大きく）のぶんはカメラで打ち消さない：倍率は上乗せ前の体で決め、上乗せぶんはそのまま画面で大きく映す。
    const nat = (p) => (p && p.measured && p.fit > 0 ? dpr / (p.fit * S) * Assets.bigBoost(p.species) : 0);
    let natZ = 0;
    if (B.cfg && !B.cfg.demo && !B.idlePan) {
      if (me) natZ = nat(me);
      else { const zs = act.map(nat).filter((v) => v > 0); if (zs.length) natZ = Math.exp(zs.reduce((a, v) => a + Math.log(v), 0) / zs.length); }
    }
    // フィールドが荒く見えないよう、等倍のための寄りは「1 art px = 3 CSS px」まで。アリーナより広く引くこともしない（大型は GIF より大きく映る）。
    const rawNat = natZ; // 上限をかける前の「GIF 等倍」の倍率
    if (natZ) natZ = U.clamp(natZ, baseZ, Math.max(baseZ, 3 * dpr / S));
    // [ユーザー指定] 巨大マックス以外は、等倍の下限のあとで 0.78 倍に引く。
    // フシギバナは今より少し小さく、トレーナーはフシギバナの体高まで少し大きくして、同じくらいで揃う。
    const leadGmax = me && me.species && (me.species.kind === "gigantamax" || me.species.cat === "gigantamax");
    // [ユーザー指定 2026-11-15] トレーナーとモンスターを同じ比率で小さく映す（FIELD_DRAW 0.75 → 0.6）。体が小さくなったぶん
    // カメラが寄って元に戻らないよう、カメラの倍率は縮小前の体（rawOld）で決める（フィールドの映り方は変えない）。ただしドットを欠けさせない：
    // GIF の 1 ドットが画面の 1 画素（device px）を下回らない倍率 oneDot を下限にする（これまで下回っていなかった場合だけ）。
    if (natZ && !leadGmax) {
      const rawOld = rawNat * 0.8; // 縮小前（FIELD_DRAW 0.75）の等倍倍率
      const prevDot = dpr * U.clamp(rawOld, baseZ, Math.max(baseZ, 3 * dpr / S)) * 0.78 / rawOld; // 縮小前、1 ドットが何画素だったか
      natZ = Math.max(U.clamp(rawOld, baseZ, Math.max(baseZ, 3 * dpr / S)) * 0.78, prevDot >= 1 ? rawNat / dpr : 0);
    }
    // スプライト用キャンバスの密度：いちばん細かく縮める体でも 1 ドット ≧ 1 画素になるだけ持つ（4〜6）。
    let minFit = 1;
    for (const p of act) if (p.measured && p.fit > 0) minFit = Math.min(minFit, p.fit);
    sprNeed = U.clamp(Math.ceil(1 / minFit - 0.02), 4, 6);
    cam.zoomT = U.clamp((natZ || baseZ) * zoomMul * cam.user * (1 + B.fx.zoomPunch * 0.24), 0.3, 14);
    const zRate = (B.koCam || B.fx.zoomPunch > 0.1 || B.activeClash) ? 8.5 : 4.8;
    cam.zoom = snap ? cam.zoomT : U.damp(cam.zoom, cam.zoomT, zRate, dt);
    // 引いたぶんだけ描画バッファを広げる（0.05 刻み。広げるのは即時、戻すのは余裕が出てから）。
    const need = Math.min(cam.zoom, cam.zoomT), bz = need >= 1 ? 1 : Math.max(0.3, Math.floor(need * 20) / 20);
    if (!use3 && (bz < BZ || bz > BZ + 0.2)) { BZ = bz; alloc(); }
    // 3D のカメラの見下ろし角: ふだんは Scene3D.PITCH（またはステージ設定 pitch）。タイトル・デモ・登場・K.O.・競り合いでは低く構えて奥の景色を見せる
    if (use3) {
      const cfg3 = Scene3D.cfg(), base = (cfg3 && cfg3.pitch) || Scene3D.PITCH, low = (B.cfg && B.cfg.demo) || B.idlePan ? 0.18 : (B.focus || (B.koCam && B.koCam.t > 0) || B.phase === "end") ? 0.12 : B.activeClash ? 0.06 : 0;
      const tgt = Math.max(0.34, base - low);
      pitch3 = snap || !pitch3 ? tgt : U.damp(pitch3, tgt, 2.6, dt);
    }
    const fx = B.fx;
    if (fx.shake > 0) { const k = fx.shakeAmp * Math.min(1, fx.shake / 0.25); cam.sx = U.rand(-k, k); cam.sy = U.rand(-k, k) * 0.7; } else cam.sx = cam.sy = 0;
    emitMotes(B, dt);
  }
  function computeView(A) {
    if (P3) { // 3D: バッファはアリーナ全体（原点 0,0）。カメラは地面の点 (cam.x, cam.y) を画面の中心に映す
      const K = Arenas.K, af = A.af;
      let ax = (cam.x + cam.sx) / K, ay = (cam.y + cam.sy) / K;
      if (!cam.lock) { ax = U.clamp(ax, af.x - 24, af.x + af.w + 24); ay = U.clamp(ay, af.y - 40, af.y + af.h + 16); }
      ix = iy = 0; fracX = fracY = 0;
      const basePitch = (Scene3D.cfg() && Scene3D.cfg().pitch) || Scene3D.PITCH;
      Scene3D.setCamera(ax, ay, S * Math.max(1e-3, cam.zoom), W, H, pitch3 || basePitch);
      return;
    }
    const vwArt = W / S, vhArt = H / S, K = Arenas.K, z = Math.max(1e-3, cam.zoom);
    // The shader zooms around this view center. Clamp the zoomed window, not the
    // full buffer, or a wide monitor locks the fight onto the stadium middle.
    const visW = vwArt / z, visH = vhArt / z;
    let ax = (cam.x + cam.sx) / K, ay = (cam.y + cam.sy) / K;
    if (!cam.lock) {
      ax = visW >= A.aw ? A.aw / 2 : U.clamp(ax, visW / 2, A.aw - visW / 2);
      ay = visH >= A.ah ? A.ah / 2 : U.clamp(ay, visH / 2, A.ah - visH / 2);
    }
    const ox = ax - vwArt / BZ / 2, oy = ay - vhArt / BZ / 2;
    ix = Math.floor(ox); iy = Math.floor(oy); fracX = ox - ix; fracY = oy - iy;
  }
  // art (absolute) → overlay device px
  function artToScreen(ax, ay) {
    if (P3) return Scene3D.project(ax, ay, 0);
    let x = (ax - ix - fracX) * S, y = (ay - iy - fracY) * S;
    x = (x - W / BZ / 2) * cam.zoom + W / 2; y = (y - H / BZ / 2) * cam.zoom + H / 2;
    return { x, y };
  }
  // z（world）= 地面からの高さ。モンスターと同じく、画面の上へそのままの倍率で持ち上げる
  function worldToScreen(x, y, z = 0) {
    if (!P3) return artToScreen(x / Arenas.K, (y - z) / Arenas.K);
    const p = Scene3D.project(x / Arenas.K, y / Arenas.K, 0);
    return { x: p.x, y: p.y - z / Arenas.K * p.s, s: p.s };
  }
  function toWorld(sx, sy) { // CSS px → world
    if (P3) { const g = Scene3D.unproject(sx * dpr, sy * dpr); return { x: g.x * Arenas.K, y: g.y * Arenas.K }; }
    let x = sx * dpr, y = sy * dpr;
    x = (x - W / 2) / cam.zoom + W / BZ / 2; y = (y - H / 2) / cam.zoom + H / BZ / 2;
    return { x: (x / S + ix + fracX) * Arenas.K, y: (y / S + iy + fracY) * Arenas.K };
  }

  // ---- ambient motes -------------------------------------------------------------------------------
  function emitMotes(B, dt) {
    const A = B.arena, f = A.field, kind = A.def.motes, fx = B.fx;
    if (!kind || fx.parts.length > 1400) return;
    const rate = { dust: 10, firefly: 6, snow: 30, ember: 16, star: 8, sand: 26, bubble: 8 }[kind] || 8;
    let n = rate * dt; while (n > 0) { if (Math.random() < n) spawn(); n -= 1; }
    function spawn() {
      const x = cam.x + U.rand(-700, 700), y = cam.y + U.rand(-420, 420);
      const P = { x, y, z: U.rand(10, 120), shape: "mote", add: true, drag: 0.2, life: U.rand(2, 5) };
      if (kind === "dust") Object.assign(P, { color: "#fff4d8", size: 2, vx: U.rand(-8, 8), vy: U.rand(-5, 5), vz: U.rand(-4, 4), alpha: 0.5 });
      if (kind === "firefly") Object.assign(P, { color: U.pick(["#d8ff6a", "#fff27a"]), size: 2.5, vx: U.rand(-20, 20), vy: U.rand(-15, 15), vz: U.rand(-8, 8), blink: true, light: true });
      if (kind === "snow") Object.assign(P, { color: "#ffffff", size: 2, vx: U.rand(-20, 10), vy: U.rand(-4, 4), vz: -U.rand(25, 45), z: U.rand(80, 260), add: false });
      if (kind === "ember") Object.assign(P, { color: U.pick(["#ffb040", "#ff6a1a", "#ffe080"]), size: 2, vx: U.rand(-10, 10), vy: U.rand(-6, 6), vz: U.rand(20, 60), z: 0, life: U.rand(1.5, 3) });
      if (kind === "star") Object.assign(P, { color: U.pick(["#ffffff", "#bfe6ff", "#e8c8ff"]), size: 2, vx: 0, vy: 0, vz: U.rand(2, 8), blink: true });
      if (kind === "sand") Object.assign(P, { color: U.pick(["#f0d8a0", "#d8b878"]), size: 2, vx: U.rand(60, 120), vy: U.rand(-6, 6), vz: U.rand(-5, 5), add: false, alpha: 0.7 });
      if (kind === "bubble") Object.assign(P, { color: "#d8f2ff", size: 2, vx: U.rand(-6, 6), vy: U.rand(-6, 6), vz: U.rand(8, 20), z: 0, alpha: 0.7 });
      fx.add(P);
    }
  }

  // ---- frame -----------------------------------------------------------------------------------------
  // [ユーザー指定 2026-11-15] ステージを実写寄りの HD-2D に加工する強さ（post.js の compose）。床の素材ごと：
  // [混ぜる色の近さ, 起伏の強さ, 照り返しの強さ, 肌の強さ, 肌の細かさ x, y, 照り返しの鋭さ]。先頭を 0 にすると加工なし（元のドット）。
  const HD_FLOOR = {
    grass: [0.3, 1.0, 0.05, 1.0, 1.5, 0.75, 20], meadow: [0.3, 1.0, 0.04, 1.0, 1.5, 0.75, 20],
    dirt: [0.28, 1.1, 0.04, 1.1, 1.1, 1.1, 18], sand: [0.28, 0.9, 0.1, 0.9, 1.6, 1.0, 40],
    stone: [0.2, 1.1, 0.16, 0.45, 0.9, 0.9, 30], pokeball: [0.2, 1.0, 0.22, 0.45, 0.9, 0.9, 36],
    ice: [0.26, 0.7, 0.5, 0.35, 0.7, 0.7, 60], water: [0.3, 0.55, 0.4, 0.1, 0.8, 0.5, 48],
    volcano: [0.22, 1.4, 0.1, 1.0, 1.0, 1.0, 24], cave: [0.22, 1.3, 0.18, 0.9, 1.0, 1.0, 30], cosmic: [0.24, 0.8, 0.25, 0.3, 0.8, 0.8, 40],
  };
  function draw(B, t) {
    const A = B.arena, fx = B.fx, K = Arenas.K;
    castK = window.ARENA_HD_OFF ? 0 : (A.grade && A.grade.cast) || 0;
    const p3 = !!(A.scene3d && Post.ok());
    if (p3 !== P3 || (p3 && (curArena !== A || LW !== A.aw || LH !== A.ah))) { P3 = p3; curArena = A; alloc(); }
    curArena = A;
    computeView(A);
    FXV.p3 = P3; // 地面に描く円のたてつぶしを 3D では戻す（FXV.gsq）
    // ---- world (art px, integer) ----
    wctx.setTransform(1, 0, 0, 1, 0, 0);
    wctx.fillStyle = "#080b18"; wctx.fillRect(0, 0, LW, LH);
    wctx.drawImage(Arenas.floorFrame(A, t), -ix, -iy);
    wctx.drawImage(A.decals, -ix, -iy);
    wctx.drawImage(A.relief, -ix, -iy);
    Water.draw(wctx, A, ix, iy);
    VFX.drawWorld(wctx, B, ix, iy, t); // Surf's flood: water lying on the floor, under the monsters
    Ground.drawIce(wctx, A, ix, iy, t);
    Slabs.drawHoles(wctx, B, ix, iy);
    if (A.crowd && !P3) {
      const cheer = A.hype > 0.25 && Math.floor(t * (5 + A.hype * 6)) % 2 === 1;
      wctx.drawImage(A.crowd[cheer ? 1 : 0], -ix, -iy - (cheer && A.hype > 0.7 ? 1 : 0));
    }
    if (!P3) Arenas.drawTicker(wctx, A, t, ix, iy);
    drawTrainers(B, t, true);
    // shadows
    const ents = [];
    for (const f of B.teams.flat()) if (f.active || f.out > 0 || (f.ko && f.koT < 2.6)) ents.push({ y: f.y, f });
    for (const p of A.props) if (p.alive) ents.push({ y: p.y, p });
    for (const c of B.clones || []) ents.push({ y: c.y, c });
    Slabs.entities(B, ents);
    Ground.entities(A, ents);
    for (const m of B.mounds || []) ents.push({ y: m.y, mound: m });
    for (const e of ents) if (e.f) shadow(e.f); else if (e.p) castProp(e.p);
    for (const e of ents) if (e.p) Pixel.ellipse(wctx, e.p.x / K - ix, e.p.y / K - iy, Math.round(e.p.r / K * 1.1), Math.max(2, Math.round(e.p.r / K * 0.45)), "rgba(0,0,0,0.35)");
    wctx.save(); wctx.setTransform(0.5, 0, 0, 0.5, -ix, -iy); fx.drawParticleShadows(wctx); wctx.restore();
    ents.sort((a, b) => a.y - b.y);
    // Monsters (and anything that has to cover them) on the finer grid. Pitch pixels stay on world.
    if (P3) { SPR = 4; sox = soy = 0; if (sprc.width !== W || sprc.height !== H) { sprc.width = W; sprc.height = H; } }
    else sprWindow();
    sctx.setTransform(1, 0, 0, 1, 0, 0);
    sctx.clearRect(0, 0, sprc.width, sprc.height);
    sctx.imageSmoothingEnabled = false;
    drawTrainers(B, t, false);
    for (const e of ents) {
      sctx.globalAlpha = 1;
      if (e.f) { const f = e.f, arc = f.enter > 0.52 && f.enterFrom ? Battle.ballArc(f) : null; if (anchor(arc ? arc.x : f.x, arc ? arc.y : f.y)) fighter(f, B, t); }
      else if (e.p) { if (anchor(e.p.x, e.p.y)) { sprT(SPR); prop(e.p, t); } }
      else if (e.slab) { if (anchor((e.slab.x0 + e.slab.img.width / 2) * K, e.slab.top * K)) { sprT(SPR); Slabs.drawPiece(sctx, e.slab, ix, iy); } }
      else if (e.crystal) { if (anchor(e.crystal.x, e.crystal.y)) { sprT(SPR); Ground.drawCrystal(sctx, e.crystal, ix, iy); } }
      else if (e.mound) { if (anchor(e.mound.x, e.mound.y)) { sprT(SPR); Ground.drawMound(sctx, A, e.mound, ix, iy, t, Arenas.floorFrame(A, t)); } }
      else if (anchor(e.c.x, e.c.y)) clone(e.c, t);
    }
    if (P3) weather3(B, t);
    // ---- fx ----
    fctx.setTransform(1, 0, 0, 1, 0, 0); fctx.clearRect(0, 0, LW, LH);
    fctx.setTransform(1 / K, 0, 0, 1 / K, -ix, -iy);
    fctx.imageSmoothingEnabled = true;
    if (Moves.drawWindups) Moves.drawWindups(fctx, B, t);
    Moves.draw(fctx, B, "ground", t);
    fx.drawRings(fctx, true);
    // [ユーザー指定 2026-11-16] 3D: 空中のエフェクトは地面に貼らず、画面に正対して立つ層（airc、画面の座標）に描く。
    // airQ = この層の 1 ドットの画素数（画面中央の地面の 1 art px と同じ大きさ。0.25 きざみ）。平面表示では今までどおり fx に描く。
    let actx = fctx;
    if (P3) {
      airQ = U.clamp(Math.round(S * cam.zoom * 4) / 4, 1, 8) || 2;
      const aw = Math.ceil(W / airQ), ah = Math.ceil(H / airQ);
      if (airc.width !== aw || airc.height !== ah) { airc.width = aw; airc.height = ah; }
      actx = airx; actx.setTransform(1, 0, 0, 1, 0, 0); actx.clearRect(0, 0, aw, ah); actx.imageSmoothingEnabled = true;
      FXV.q = airQ; FXV.on = true;
    }
    Moves.draw(actx, B, "air", t);
    fx.drawParticles(actx, "air");
    fx.drawRings(actx, false);
    FXV.on = false;
    actx.setTransform(1, 0, 0, 1, 0, 0);
    fctx.setTransform(1, 0, 0, 1, 0, 0);
    if (!P3) weather(B, t);
    // ---- light ----
    lighting(B, t);
    // ---- compose ----
    const g = Object.assign({}, A.grade);
    if (P3 && Scene3D.cfg() && Scene3D.cfg().grade) Object.assign(g, Scene3D.cfg().grade); // ステージの 3D 定義が後処理を上書きできる
    if (window.ARENA_HD_OFF) { g.sun = 0; g.mote = 0; g.split = 0; g.rays = g.rays0 || 0; if (g.bloom0 !== undefined) g.bloom = g.bloom0; } // 作り直す前の見た目に戻す
    else { const on = B.lightsOn === undefined ? 1 : B.lightsOn; g.rays = (g.rays || 0) * on; g.mote = (g.mote || 0) * on; g.sun = (g.sun || 0) * on; } // 照明が消えている間は光の筋もちりも出さない
    if (B.weather.kind === "sand") { g.tint = [g.tint[0] * 1.05, g.tint[1] * 0.98, g.tint[2] * 0.85]; g.fog = [0.82, 0.68, 0.45, 0.35]; }
    if (B.weather.kind === "rain") { g.tint = [g.tint[0] * 0.9, g.tint[1] * 0.95, g.tint[2] * 1.05]; g.fog = [0.35, 0.42, 0.55, 0.25]; }
    if (B.weather.kind === "sun") { g.rays = (g.rays || 0) + (window.ARENA_HD_OFF ? 0.18 : 0.1); g.tint = [g.tint[0] * 1.08, g.tint[1] * 1.03, g.tint[2] * 0.92]; }
    if (B.weather.kind === "hail") { g.fog = [0.85, 0.9, 1.0, 0.25]; }
    const pk = psyK(B);
    if (pk > 0) { g.tint = [g.tint[0] * (1 - 0.18 * pk), g.tint[1] * (1 - 0.3 * pk), g.tint[2] * (1 + 0.12 * pk)]; g.fog = [0.35, 0.22, 0.6, 0.35 * pk]; }
    const fl = fx.flash > 0 ? U.hexToRgb(fx.flashColor).map((v) => v / 255) : [1, 1, 1];
    if (Post.ok()) {
      const three = P3 ? { A, hype: A.hype || 0, on: B.lightsOn === undefined ? 1 : B.lightsOn } : null;
      if (P3) g.tilt = Math.max(g.tilt || 0, (Scene3D.cfg() && Scene3D.cfg().dof) || 1.25);
      Post.render({ three, air: P3 ? airc : null, airQ, world, fx: fxc, light: lightc, spr: sprc, sprW: sprc.width, sprH: sprc.height, lowW: LW, lowH: LH, sprX: sox, sprY: soy, sprK: SPR, S, fracX, fracY, zoom: cam.zoom, pad: 1 / BZ, time: t, hd: window.ARENA_HD_OFF ? null : HD_FLOOR[A.def.floor] || HD_FLOOR.grass, orgX: ix, orgY: iy, fld: [A.af.x, A.af.y, A.af.w, A.af.h], grade: g, flash: [fl[0], fl[1], fl[2], Math.min(0.85, fx.flash)], imp: impact(fx) });
    } else fallback(fl, fx.flash, g, t);
    overlay(B, t);
  }

  // 0‥1 strength of Mewtwo's violet entrance gloom (fast in, hold, slow out)
  function psyK(B) {
    const p = B.psy;
    if (!p) return 0;
    return Math.min(1, p.t / 0.35, (p.dur - p.t) / 0.7);
  }
  // invert strobe + radial blur centre in screen uv (y up, as the shader samples)
  function impact(fx) {
    if (!fx.invert && fx.blur <= 0) return [0, 0, 0.5, 0.5];
    const p = worldToScreen(fx.blurX, fx.blurY);
    return [fx.invert, fx.blur, p.x / W, 1 - p.y / H];
  }
  // WebGL が使えないとき（debug.html をファイルから直接ひらいたとき）の HD-2D：シェーダーと同じ考えを 2D で近似する。
  //  日なた・日かげ（ステージに固定した斜めの明暗）、左上からの光の筋、手前と奥のぼかし、四すみの暗がり。
  let sunTex = null;
  function sunTexture() {
    if (sunTex) return sunTex;
    const c = U.canvas(384, 288), x = c.getContext("2d");
    x.fillStyle = "rgb(128,128,128)"; x.fillRect(0, 0, c.width, c.height); // overlay で何も変えない灰色
    let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    x.translate(c.width / 2, c.height / 2); x.rotate(0.61); x.scale(1.5, 1); // 左上→右下に伸びた帯
    for (let i = 0; i < 22; i++) { // 大きくやわらかい面を少しだけ（細かい縞にしない）
      const lit = i % 2 === 0, px = (rnd() - 0.5) * 300, py = (rnd() - 0.5) * 420, r = 52 + rnd() * 48;
      const gr = x.createRadialGradient(px, py, 0, px, py, r);
      gr.addColorStop(0, lit ? "rgba(255,240,170,0.42)" : "rgba(20,40,90,0.4)"); gr.addColorStop(1, lit ? "rgba(255,240,170,0)" : "rgba(20,40,90,0)");
      x.fillStyle = gr; x.fillRect(px - r, py - r, r * 2, r * 2);
    }
    return (sunTex = c);
  }
  const rayTex = new Map();
  function rayTexture(col) {
    const key = col.join(",");
    if (rayTex.has(key)) return rayTex.get(key);
    const c = U.canvas(256, 256), x = c.getContext("2d"), rc = col.map((v) => Math.round(v * 255)).join(",");
    let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < 9; i++) { // 太さと明るさの違う帯。帯のふちは透明へなだらかに
      const y = 44 + rnd() * 168, wd = 8 + rnd() * 22, a = 0.25 + rnd() * 0.45; // 帯は絵のふち（上下 14px）にかからない
      const gr = x.createLinearGradient(0, y - wd, 0, y + wd);
      gr.addColorStop(0, `rgba(${rc},0)`); gr.addColorStop(0.5, `rgba(${rc},${a})`); gr.addColorStop(1, `rgba(${rc},0)`);
      x.fillStyle = gr; x.fillRect(0, y - wd, 256, wd * 2);
    }
    x.globalCompositeOperation = "destination-in"; // 光源（左）から離れるほど消える
    const fade = x.createLinearGradient(0, 0, 256, 0); fade.addColorStop(0, "rgba(0,0,0,1)"); fade.addColorStop(0.55, "rgba(0,0,0,0.35)"); fade.addColorStop(1, "rgba(0,0,0,0)");
    x.fillStyle = fade; x.fillRect(0, 0, 256, 256);
    rayTex.set(key, c);
    return c;
  }
  // 日なた・日かげ: 地面を描いた直後（モンスターや技を描く前）に、フィールドの中だけへ重ねる。ステージに固定、わずかにゆれる
  function fallbackSun(c, g, t) {
    const A = curArena;
    if (!g || !g.sun || !A) return;
    c.save();
    const af = A.af, x0 = (af.x - ix - fracX) * S, y0 = (af.y - iy - fracY) * S;
    c.beginPath(); c.rect(x0, y0, af.w * S, af.h * S); c.clip();
    c.globalCompositeOperation = "overlay"; c.globalAlpha = Math.min(1, g.sun * 1.7); c.imageSmoothingEnabled = true;
    const dx = Math.sin(t * 0.05) * 10;
    c.drawImage(sunTexture(), (-ix - fracX - 30 + dx) * S, (-iy - fracY - 30) * S, (A.aw + 60) * S, (A.ah + 60) * S);
    c.restore();
  }
  let dofc = null, dctx = null;
  function fallbackHd(c, g, t) {
    if (!g) return;
    if (g.tilt > 0) { // 手前と奥のぼかし: 半分の大きさでぼかした写しを、上下ほど濃く・中央は透明になるなだらかなマスクで重ねる（継ぎ目なし）
      const w2 = Math.max(1, W >> 1), h2 = Math.max(1, H >> 1);
      if (!dofc) { dofc = U.canvas(w2, h2); dctx = dofc.getContext("2d"); }
      if (dofc.width !== w2 || dofc.height !== h2) { dofc.width = w2; dofc.height = h2; }
      dctx.setTransform(1, 0, 0, 1, 0, 0); dctx.globalCompositeOperation = "copy";
      dctx.filter = `blur(${(0.9 * Math.max(1, dpr)).toFixed(2)}px)`; dctx.drawImage(glc, 0, 0, w2, h2); dctx.filter = "none";
      dctx.globalCompositeOperation = "destination-in";
      const m = dctx.createLinearGradient(0, 0, 0, h2), a = Math.min(1, g.tilt);
      m.addColorStop(0, `rgba(0,0,0,${a})`); m.addColorStop(0.27, "rgba(0,0,0,0)"); m.addColorStop(0.73, "rgba(0,0,0,0)"); m.addColorStop(1, `rgba(0,0,0,${a})`);
      dctx.fillStyle = m; dctx.fillRect(0, 0, w2, h2);
      c.save(); c.setTransform(1, 0, 0, 1, 0, 0); c.imageSmoothingEnabled = true; c.drawImage(dofc, 0, 0, W, H); c.restore();
    }
    if (g.rays > 0) { // 光の筋: 1 回だけ作ったやわらかい帯の絵を、左上から右下へ傾けて重ねる
      c.save(); c.setTransform(1, 0, 0, 1, 0, 0); c.globalCompositeOperation = "lighter"; c.imageSmoothingEnabled = true;
      c.globalAlpha = Math.min(0.4, g.rays * 0.5) * (0.9 + 0.1 * Math.sin(t * 0.4));
      const L = Math.hypot(W, H) * 1.6;
      c.translate(-W * 0.3, -H * 0.55); c.rotate(0.55); // 帯の始まりは画面の外（左上）
      c.drawImage(rayTexture(g.rayCol || [1, 0.95, 0.8]), 0, Math.sin(t * 0.07) * H * 0.02, L, H * 1.9);
      c.restore();
    }
    if (g.vig > 0) { // 四すみの暗がり
      c.save(); c.setTransform(1, 0, 0, 1, 0, 0);
      const gr = c.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.hypot(W, H) * 0.62);
      gr.addColorStop(0, "rgba(0,0,0,0)"); gr.addColorStop(1, `rgba(4,8,20,${Math.min(0.6, g.vig * 0.55)})`);
      c.fillStyle = gr; c.fillRect(0, 0, W, H); c.restore();
    }
  }
  let curArena = null;
  function fallback(fl, fa, g, t) {
    // Same zoom about the screen centre as the Post shader (and as artToScreen for the overlay).
    const c = ctx2d, z = cam.zoom; c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, W, H);
    c.setTransform(z, 0, 0, z, (W / 2) * (1 - z / BZ), (H / 2) * (1 - z / BZ)); c.imageSmoothingEnabled = false;
    c.drawImage(world, -fracX * S, -fracY * S, LW * S, LH * S);
    if (!window.ARENA_HD_OFF) fallbackSun(c, g, t);
    c.drawImage(sprc, (sox - fracX) * S, (soy - fracY) * S, SWa * S, SHa * S);
    c.globalCompositeOperation = "multiply"; c.imageSmoothingEnabled = true; c.drawImage(lightc, -fracX * S, -fracY * S, LW * S, LH * S);
    // 光の絵は「127 = そのまま」で作ってある（シェーダーは 2 倍して掛ける）。multiply だけだと半分の明るさになるので、2 倍に戻す
    if (g && !window.ARENA_HD_OFF) { c.globalCompositeOperation = "color-dodge"; c.fillStyle = "rgb(128,128,128)"; c.fillRect(-W, -H, W * 3, H * 3); }
    c.globalCompositeOperation = "source-over"; c.imageSmoothingEnabled = false; c.drawImage(fxc, -fracX * S, -fracY * S, LW * S, LH * S);
    c.setTransform(1, 0, 0, 1, 0, 0);
    if (g && !window.ARENA_HD_OFF) fallbackHd(c, g, t);
    if (fa > 0) { c.globalAlpha = Math.min(0.85, fa); c.fillStyle = `rgb(${fl.map((v) => v * 255).join(",")})`; c.fillRect(0, 0, W, H); c.globalAlpha = 1; }
  }

  // ---- entities -----------------------------------------------------------------------------------------
  // Current frame for the displayed 8-way facing: { img, ax, ay, w, h, bh } (feet anchor, art px).
  function frameOf(f, phase = 0) {
    const v = Dir8.view(f);
    return Dir8.frame(v.rec, (v.t !== undefined ? v.t : f.anim) + phase, v.tf); // v.t: time since this attack started
  }
  // 立ち絵の余白は除き、人の不透明な身長で合わせる。
  // [ユーザー指定 2026-11-15] トレーナーが大きすぎるので、1.7m の物差し（29px）より少し小さい 0.85 倍 = 24px で描く。
  const TRAINER_DRAW = 0.72; // 0.85 → 0.72（ユーザー指定「まだ大きすぎる」）= 21px
  function trainerBodyPx() {
    return Math.max(14, Math.round(Assets.onFieldPx(Assets.trainerArtPx()) * TRAINER_DRAW));
  }
  function trainerStandBox(tr, rec) {
    if (!tr) return null;
    if (tr._stand) return tr._stand;
    if (tr._standFail) return null; // [2026-11-15] file:// ではドットを読めない。毎フレーム canvas を作り直すと GPU のメモリが尽きて画面が点滅するので、失敗は 1 回で覚える
    const img = rec && rec.frames && rec.frames[0] && rec.frames[0].canvas;
    if (!img || !(tr.w > 0) || !(tr.h > 0)) return null;
    try {
      const fi = (tr.west && tr.west[0]) || 0;
      const b = Pixel.bounds(img, fi * tr.w, 0, tr.w, tr.h);
      if (!b || !(b.h > 0)) { tr._standFail = true; return null; }
      tr._stand = b;
      return b;
    } catch (e) { tr._standFail = true; return null; }
  }
  // [ユーザー指定 2026-11-15] トレーナーは顔（頭）だけが大きすぎるので、頭だけ TRAINER_HEAD 倍に縮めて描く（体はそのまま）。
  // 首の行 = 立ち絵で、頭のいちばん広い幅より十分せまくなる最初の行（身長の 45〜62% の範囲でさがす）。読めなければ 56%。
  const TRAINER_HEAD = 0.72;
  function trainerNeck(tr, rec) {
    if (tr._neck) return tr._neck;
    const b = trainerStandBox(tr, rec);
    if (!b) return Math.round(tr.h * 0.6);
    let row = Math.round(b.y + b.h * 0.56);
    try {
      const img = rec.frames[0].canvas, fi = (tr.west && tr.west[0]) || 0, c = U.canvas(tr.w, tr.h), x = c.getContext("2d", { willReadFrequently: true });
      x.drawImage(img, fi * tr.w, 0, tr.w, tr.h, 0, 0, tr.w, tr.h);
      const d = x.getImageData(0, 0, tr.w, tr.h).data, wid = [];
      for (let y = 0; y < tr.h; y++) { let lo = -1, hi = -1; for (let px = 0; px < tr.w; px++) if (d[(y * tr.w + px) * 4 + 3] > 40) { if (lo < 0) lo = px; hi = px; } wid.push(lo < 0 ? 0 : hi - lo + 1); }
      let top = 0;
      const y0 = Math.round(b.y + b.h * 0.45), y1 = Math.round(b.y + b.h * 0.62);
      for (let y = b.y; y < y0; y++) top = Math.max(top, wid[y]);
      for (let y = y0; y <= y1; y++) { if (wid[y] <= top * 0.82) { row = y; break; } top = Math.max(top, wid[y]); }
    } catch (e) {}
    tr._neck = row;
    return row;
  }
  function trainerSpriteSize(tr, rec) {
    const body = trainerBodyPx();
    if (!tr) return { dw: Math.round(16 * body / 24), dh: body, gap: 0, sc: 1 };
    const b = trainerStandBox(tr, rec);
    const opaque = b && b.h ? b.h : Math.max(1, Math.round(tr.h * 0.75));
    const sc = body / opaque;
    const gap = b ? Math.max(0, tr.h - (b.y + b.h)) : 0;
    return { dw: tr.w * sc, dh: tr.h * sc, gap, sc };
  }
  function trainerFieldPx() { return trainerBodyPx(); }
  function enterScale(f) {
    if (f.enter <= 0) return 1;
    const k = 1 - f.enter / 1.05;
    return k < 0.5 ? 0 : U.easeOutBack(Math.min(1, (k - 0.5) / 0.35));
  }
  // [ユーザー指定 2026-11-16] HD-2D の落ち影: 光は左上から。体の形そのままの影を、足もとから右下へ倒して地面に落とす
  // （体の高さ h の点は、足もとから右へ 0.9h・下へ 0.32h の所に落ちる）。跳んだり浮いたりしている間は、高さのぶん影が体から離れる。
  // 影の濃さはステージごと（grade.cast。0 なら出さない）。足もとの丸い接地影はこれまでどおり別に描く。
  const CAST_X = 0.9, CAST_Y = 0.32;
  let castK = 0;
  function castShadow(f, s, sx, sy) {
    const a = castK;
    if (!a || (f.ko && f.koT > 0.25)) return; // 倒れて横になったら出さない（立っている間は残す）
    const fr = frameOf(f), sc = enterScale(f) * (f.fit || 1);
    if (!fr || sc <= 0 || !fr.img.width) return;
    // 体と同じ動き（ため・突進・浮き上がり・ゆれ）に影も付いていく
    const K = Arenas.K, P = f.pose || { fwd: 0, up: 0, dir: 0 };
    const bx = sx + Math.cos(P.dir) * P.fwd / K + (f.sway || 0), by = sy + Math.sin(P.dir) * P.fwd * 0.6 / K + (f.z < 8 && !f.flies ? Math.round(f.sinkG || 0) : 0);
    const up = Math.min(90, ((f.z || 0) + (P.up || 0)) / K + (f.bob || 0));
    const fade = 1 - Math.min(0.6, up / 120);
    wctx.save();
    wctx.globalAlpha = a * s * fade;
    wctx.imageSmoothingEnabled = true;
    wctx.transform(1, 0, -CAST_X, -CAST_Y, bx + up * CAST_X, by + up * CAST_Y);
    wctx.drawImage(Dir8.tint(fr.img, "#07121c"), -fr.ax * sc, -fr.ay * sc, fr.w * sc, fr.h * sc);
    wctx.restore();
  }
  // 小物（木箱・岩・氷・水晶）も同じ向きに影を落とす
  function castProp(p) {
    if (!castK || !p.sprite || !p.sprite.width) return;
    const K = Arenas.K;
    wctx.save();
    wctx.globalAlpha = castK;
    wctx.imageSmoothingEnabled = true;
    wctx.transform(1, 0, -CAST_X, -CAST_Y, p.x / K - ix, p.y / K - iy);
    wctx.drawImage(Dir8.tint(p.sprite, "#07121c"), -p.ax, -p.ay);
    wctx.restore();
  }
  function shadow(f) {
    const K = Arenas.K, s = enterScale(f) * (f.ko ? Math.max(0, 1 - Math.max(0, f.koT - 1.6) / 0.8) : 1) * (f.out > 0 ? f.out / (f.outMax || 0.42) : 1);
    if (s <= 0 || f.swimming || f.burrow) return;
    const lift = Math.min(1, f.z / 200), r = f.r / K, planted = !f.flies && f.z < 8;
    const sx = f.x / K - ix, sy = f.y / K - iy;
    castShadow(f, s, sx, sy);
    // [ユーザー指定 2026-11-15] 浮いているモンスター（鳥型・蛇型ドラゴン）：影は地面に残り、体が高いほど小さく薄くなる（浮遊感）。
    if (f.flies) {
      const hk = 1 - Math.min(0.3, (f.bob || 0) * 0.028) - lift * 0.4;
      Pixel.ellipse(wctx, sx, sy, Math.max(2, Math.round(r * 0.95 * s * hk)), Math.max(1, Math.round(r * 0.3 * s * hk)), `rgba(0,0,0,${0.36 * hk * (1 - lift * 0.3)})`);
      return;
    }
    // 地上のモンスター：足元に濃い接地影（外側の影＋足の真下の濃い芯）。跳んでいる間は高さに応じて小さく薄くなる。
    const air = planted ? 0 : Math.min(1, 0.25 + lift), gy = sy + (planted ? Math.round(f.sinkG || 0) : 0);
    Pixel.ellipse(wctx, sx, gy, Math.max(2, Math.round(r * 1.2 * s * (1 - air * 0.4))), Math.max(1, Math.round(r * 0.34 * s * (1 - air * 0.4))), `rgba(0,0,0,${0.5 * (1 - air * 0.5)})`);
    if (planted) Pixel.ellipse(wctx, sx, gy, Math.max(1, Math.round(r * 0.8 * s)), Math.max(1, Math.round(r * 0.2 * s)), "rgba(0,0,0,0.3)");
  }
  function fighter(f, B, t) {
    const K = Arenas.K, ctx = sctx, k = SPR;
    sprT(1);
    if (f.enter > 0.52 && f.enterFrom) { // Poké Ball arc from trainer hand with smooth spin & afterimages
      const arc = Battle.ballArc(f), u = arc.u;
      const bx = (arc.x / K - ix) * k, gy = (arc.y / K - iy) * k, by = ((arc.y - arc.z) / K - iy) * k;
      const shR = Math.max(2, 4.2 - arc.z / 95);
      Pixel.ellipse(ctx, bx, gy, shR * k, shR * 0.48 * k, "rgba(0,0,0,0.34)");
      const ball = Assets.ball(), thColor = f.side === 0 ? "#ff688c" : "#6ab4ff";
      const drawBall = (x, y, ang, sz, alpha, tint) => {
        ctx.save(); ctx.globalAlpha = alpha; ctx.translate(Math.round(x), Math.round(y)); ctx.rotate(ang);
        const h = sz * 0.5;
        if (ball.frames) {
          const img = tint ? Dir8.tint(ball.frames[0].canvas, tint) : ball.frames[0].canvas;
          ctx.drawImage(img, 0, 0, 30, 30, -h, -h, sz, sz);
        } else {
          Pixel.ellipse(ctx, 0, 0, Math.max(2, h * 0.6), Math.max(2, h * 0.6), tint || "#e33");
        }
        ctx.restore();
      };
      for (let i = 3; i >= 1; i--) {
        const pu = u - i * 0.04;
        if (pu <= 0) continue;
        const pa = Battle.ballArc(f, pu);
        drawBall((pa.x / K - ix) * k, ((pa.y - pa.z) / K - iy) * k, pu * Math.PI * 6.5 * (f.side === 0 ? 1 : -1), (11 - i * 1.4) * k, 0.38 - i * 0.09, thColor);
      }
      const flare = u > 0.78 ? (u - 0.78) / 0.22 : 0;
      if (flare > 0) {
        ctx.save(); ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = flare * 0.75;
        const gr = (7 + flare * 8) * k, gg = ctx.createRadialGradient(bx, by, 0, bx, by, gr);
        gg.addColorStop(0, "#ffffff"); gg.addColorStop(0.45, thColor); gg.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = gg; ctx.beginPath(); ctx.arc(bx, by, gr, 0, U.TAU); ctx.fill();
        ctx.restore();
      }
      const bSz = (11 + flare * 2.5) * k, rot = u * Math.PI * 6.5 * (f.side === 0 ? 1 : -1);
      drawBall(bx, by, rot, bSz, 1, null);
      if (flare > 0) drawBall(bx, by, rot, bSz, flare * 0.85, "#ffffff");
      return;
    }
    if (f.burrow) return; // underground: only the mound shows
    const fr = frameOf(f);
    const sc = enterScale(f) * (f.fit || 1) * k;
    if (!fr || sc <= 0) return;
    // body motion (wind-up crouch / lunge / float / flinch) along its own direction, plus tremble
    const P = f.pose || { fwd: 0, up: 0, sq: 1, shake: 0, dir: 0 }, sh = P.shake ? Math.round((Math.random() - 0.5) * 2 * P.shake) : 0;
    const baseY = (f.y + Math.sin(P.dir) * P.fwd * 0.6 - f.z - P.up) / K - iy;
    let px = (Math.round((f.x + Math.cos(P.dir) * P.fwd) / K - ix + (f.sway || 0)) + sh) * k, py = Math.round(baseY - (f.bob || 0)) * k;
    const wade = f.swimming ? Math.max(0.15, f.wade || 0) : (f.wade || 0) > 0.06 && !f.flies && f.z < 8 ? f.wade : 0, inWater = wade > 0;
    if (!f.flies && !f.swimming && f.z < 8 && f.enter <= 0) py += k; // plant the outline into the contact shadow
    const outRatio = f.out > 0 ? U.clamp(f.out / (f.outMax || 0.42), 0, 1) : 1;
    const ou = f.out > 0 ? 1 - outRatio : 0;
    if (f.out > 0 && f.recallTo) {
      const pull = ou * ou * 0.55;
      const hx = (f.recallTo.x / K - ix) * k, hy = ((f.recallTo.y - (f.recallTo.z || 30)) / K - iy) * k;
      px = Math.round(U.lerp(px, hx, pull));
      py = Math.round(U.lerp(py, hy, pull));
    }
    let alpha = outRatio;
    if (f.ko && f.koT > 1.6) alpha = Math.max(0, 1 - (f.koT - 1.6) / 0.8);
    if (f.invuln > 0 && f.dodgeT <= 0 && Math.floor(t * 20) % 2) alpha *= 0.55;
    ctx.save();
    ctx.globalAlpha = alpha;
    if (f.ko && f.koT > 0.25) { // lying on its side, pixel-exact 90° rotation
      const fit = (f.fit || 1) * k;
      ctx.translate(px, py - Math.round(fr.w * 0.25 * fit)); ctx.rotate((f.side ? -1 : 1) * Math.PI / 2);
      ctx.drawImage(fr.img, -fr.ax * fit, -fr.ay * fit + Math.round(fr.w * 0.12 * fit), fr.w * fit, fr.h * fit);
      ctx.globalAlpha = alpha * 0.45; ctx.drawImage(Dir8.tint(fr.img, "#1a1020"), -fr.ax * fit, -fr.ay * fit + Math.round(fr.w * 0.12 * fit), fr.w * fit, fr.h * fit);
      ctx.restore(); return;
    }
    // agent-sprite-forge action sheets carry their own crouch / lunge frames: no extra squash on top of them
    const sq = f.d8 && f.d8.length >= 32 ? 1 : f.squash * P.sq;
    const recallSx = f.out > 0 ? Math.max(0.2, 1 - ou * 0.75) : 1, recallSy = f.out > 0 ? 1 + ou * 0.22 : 1;
    const dw = Math.max(1, Math.round(fr.w * sc * sq * recallSx)), dh = Math.max(1, Math.round(fr.h * sc / sq * recallSy));
    // sunk into a crater (sinkG, art px) or burrowing in / out (digSink 0‥1): shift down, cut at the ground line
    // A dent only lowers the body with the floor (no cut: the legs stay visible on the crater floor).
    // Burrowing (digSink) is the one case that is cut at the ground line.
    const gPx = f.swimming || f.z >= 8 ? 0 : Math.round((f.sinkG || 0) * k), sinkPx = Math.round((f.digSink || 0) * fr.bh * sc * 1.05);
    py += gPx;
    const foot = py;
    py += sinkPx;
    const dx = px - Math.round(fr.ax * dw / fr.w), dy = py - Math.round(fr.ay * dh / fr.h);
    // waterline: fixed to the water (not to the bobbing body) and rising with wade as the body steps in
    const wl = inWater ? Math.round(baseY) * k - Math.round(fr.bh * sc * 0.28 * wade) + k + (Math.sin(t * 3.1 + f.side * 2) > 0.6 ? k : 0) : 0;
    const wr = Math.round(f.r / K * 1.25) * k, wry = Math.max(2, Math.round(f.r / K * 0.5)) * k, moving = Math.hypot(f.vx, f.vy) > 40;
    if (inWater) { // far half of the waterline ring goes behind the body
      waterArc(ctx, px, wl, wr, Math.max(k, Math.round(wry * wade)), Math.PI, U.TAU, "rgba(225,244,255,0.9)", t, k);
      // cut along the near curve of the ring (a straight cut reads as a body sawn in half)
      ctx.beginPath(); ctx.rect(dx - 2 * k, dy - 2 * k, dw + 4 * k, wl - dy + 2 * k); ctx.ellipse(px, wl, wr, Math.max(k, Math.round(wry * wade)), 0, 0, Math.PI); ctx.clip();
    } else if (sinkPx > 0) { ctx.beginPath(); ctx.rect(dx - 4 * k, dy - 4 * k, dw + 8 * k, foot - dy + 4 * k); ctx.clip(); }
    // Stage-2 high-speed afterimages (dodge, leap, melee lunge, dash) — unwarped on actions_8dir!
    if (f.ghosts && f.ghosts.length && sinkPx <= 0 && !inWater) {
      for (const g of f.ghosts) {
        const ga = alpha * 0.44 * Math.max(0, 1 - g.t / g.life);
        if (ga <= 0.02) continue;
        const gsc = ((f.fit || 1) * k) * ((fr.bh && g.bh) ? fr.bh / g.bh : 1);
        const gw = g.w ? Math.max(k, Math.round(g.w * gsc)) : dw;
        const gh = g.h ? Math.max(k, Math.round(g.h * gsc)) : dh;
        const gax = g.ax !== undefined ? g.ax : fr.ax, gay = g.ay !== undefined ? g.ay : fr.ay;
        const gx = Math.round(g.x / K - ix) * k - Math.round(gax * gw / (g.w || fr.w));
        const gy = Math.round((g.y - g.z) / K - iy) * k - Math.round(gay * gh / (g.h || fr.h));
        ctx.globalAlpha = ga;
        ctx.drawImage(Dir8.tint(g.img || fr.img, g.color || "#7dd3fc"), gx, gy, gw, gh);
      }
      ctx.globalAlpha = alpha;
    }
    if (f.dash) { const th = B.fx.theme(f.dash.type); for (let i = 3; i >= 1; i--) { ctx.globalAlpha = alpha * (0.52 - i * 0.12); ctx.drawImage(Dir8.tint(fr.img, th.c[1]), dx - Math.round(f.dash.ax * i * 7 * k), dy - Math.round(f.dash.ay * i * 4 * k), dw, dh); } ctx.globalAlpha = alpha; }
    ctx.drawImage(fr.img, dx, dy, dw, dh);
    if (f.flash > 0) { ctx.globalAlpha = alpha * Math.min(1, f.flash * 1.45); ctx.drawImage(Dir8.tint(fr.img, "#ffffff"), dx, dy, dw, dh); }
    if (f.status && f.status !== "slp") { ctx.globalAlpha = alpha * (0.14 + 0.08 * Math.sin(t * (f.status === "para" ? 30 : 6))); /* [ユーザー指定 2026-11-16] 0.25±0.15 だと体の模様が単色で埋まって見えないので薄くした */ ctx.drawImage(Dir8.tint(fr.img, STATUS_COLORS[f.status]), dx, dy, dw, dh); }
    if (f.held) { ctx.globalAlpha = alpha * (0.45 + 0.2 * Math.sin(t * 30)); ctx.drawImage(Dir8.tint(fr.img, "#c9a8ff"), dx, dy, dw, dh); }
    if (f.megaGlow > 0) { ctx.globalAlpha = alpha * f.megaGlow; ctx.drawImage(Dir8.tint(fr.img, "#ffe8ff"), dx, dy, dw, dh); }
    if (f.enter > 0 && f.enter <= 0.52) {
      const mk = 1 - f.enter / 0.52;
      if (mk < 0.78) {
        ctx.globalAlpha = alpha * Math.min(1, (1 - mk / 0.78) * 1.15);
        ctx.drawImage(Dir8.tint(fr.img, mk < 0.28 ? "#ffffff" : B.fx.theme(f.species.types[0]).c[1]), dx, dy, dw, dh);
      }
    }
    if (f.out > 0) {
      ctx.globalAlpha = alpha * Math.min(1, 0.3 + ou * 0.85);
      ctx.drawImage(Dir8.tint(fr.img, ou > 0.55 ? "#ffffff" : (f.side === 0 ? "#ff688c" : "#70b8ff")), dx, dy, dw, dh);
    }
    ctx.restore();
    if (inWater) {
      // the body under the surface: blue, faint, wobbling in bands (refraction)
      const sub = Dir8.tint(fr.img, "#7fb4ee"), sy0 = wl + k, sy1 = dy + dh, band = 2 * k;
      ctx.save(); ctx.globalAlpha = alpha * 0.62;
      for (let y = sy0; y < sy1; y += band) {
        const hh = Math.min(band, sy1 - y), ox = Math.round(Math.sin(t * 6 + y * 0.35) * 1.2) * k;
        ctx.drawImage(sub, 0, (y - dy) * fr.h / dh, fr.w, hh * fr.h / dh, dx + ox, y, dw, hh);
      }
      ctx.restore();
      // near half of the ring in front; while moving a bow wave is pushed ahead and foam churns behind
      ctx.save(); ctx.globalAlpha = alpha;
      waterArc(ctx, px, wl, wr, Math.max(k, Math.round(wry * wade)), 0, Math.PI, "rgba(255,255,255,0.95)", t, k);
      if (moving) {
        const h = Math.atan2(f.vy, f.vx);
        waterArc(ctx, px + Math.round(Math.cos(h) * wr * 0.35), wl + Math.round(Math.sin(h) * wry * 0.5), wr + 2 * k, wry + k, h - 0.9, h + 0.9, "#ffffff", t, k, 2);
        for (let i = 0; i < 6; i++) { const a = h + Math.PI + U.rand(-1.2, 1.2); ctx.fillStyle = Math.random() < 0.5 ? "#ffffff" : "#cfeaff"; ctx.fillRect(Math.round(px + Math.cos(a) * wr * U.rand(0.6, 1.2)), Math.round(wl + Math.sin(a) * wry * U.rand(0.6, 1.2)), k, k); }
      }
      ctx.restore();
    }
    if (sinkPx > k && !inWater) { // burrowing: the rim of the hole laps over the body
      Pixel.ring(ctx, px, foot, Math.round(f.r / K * 1.1) * k, Math.max(1, Math.round(f.r / K * 0.4)) * k, "rgba(40,26,14,0.55)", k, 0);
    }
    if (f.protect > 0) { const r = Math.round((Math.max(f.r / K, fr.bh * (f.fit || 1) * 0.55) + 5) * k); Pixel.ring(ctx, px, py - Math.round(fr.bh * (f.fit || 1) * 0.45 * k), r, r, Math.floor(t * 12) % 2 ? "#bfe8ff" : "#7fc8ff", k, 0); }
  }
  // Part of the waterline ring, crest pixels shimmering. u = pixel size (SPR on the sprite canvas).
  function waterArc(ctx, cx, cy, rx, ry, a0, a1, color, t, u = 1, thick = 1) {
    const n = Math.max(6, Math.ceil((rx + ry) / u * 3.4 * (a1 - a0) / U.TAU));
    ctx.fillStyle = color;
    for (let i = 0; i <= n; i++) {
      const a = a0 + (a1 - a0) * (i / n), w = Math.sin(a * 5 + t * 7) > 0.85 ? u : 0;
      ctx.fillRect(Math.round((cx + Math.cos(a) * (rx + w)) / u) * u, Math.round((cy + Math.sin(a) * ry) / u) * u, u, u * thick);
    }
  }
  function clone(c, t) {
    const f = c.owner, K = Arenas.K; if (!f || f.ko) return;
    const fr = frameOf(f, c.phase); if (!fr) return;
    const img = fr.img, b = { x: fr.ax, y: fr.ay, w: 0, h: 0 };
    const px = Math.round(c.x / K - ix), py = Math.round(c.y / K - iy);
    Pixel.ellipse(wctx, px, py, Math.round(f.r / K), Math.max(1, Math.round(f.r / K * 0.4)), "rgba(0,0,0,0.25)");
    const k = SPR, fit = (f.fit || 1) * k, fpx = px * k, fpy = py * k;
    sprT(1);
    sctx.save(); sctx.globalAlpha = 0.55 + 0.25 * Math.sin(t * 9 + c.phase);
    sctx.drawImage(img, fpx - Math.round((b.x + b.w / 2) * fit), fpy - Math.round((b.y + b.h) * fit), Math.round(img.width * fit), Math.round(img.height * fit));
    sctx.globalAlpha = 0.35; sctx.drawImage(Dir8.tint(img, ["#9fe8ff", "#ffb0f0", "#c8ffb0"][c.i % 3]), fpx - Math.round((b.x + b.w / 2) * fit), fpy - Math.round((b.y + b.h) * fit), Math.round(img.width * fit), Math.round(img.height * fit));
    sctx.restore();
  }
  function prop(p, t) {
    const K = Arenas.K, sh = p.shake > 0 ? (Math.floor(t * 40) % 2 ? 1 : -1) : 0;
    sctx.drawImage(p.sprite, Math.round(p.x / K - ix - p.ax + sh), Math.round(p.y / K - iy - p.ay));
  }
  // Walk frames already on the sheet. lean > 0 steps toward the pitch, hop is 0..1, pulse is an evolve scale.
  function trainerMotion(B, s, t) {
    const th = B.trainerThrow ? (B.time || 0) - (B.trainerThrow[s] || -9) : 9;
    if (th >= 0 && th < 0.8) return { kind: "throw", frame: Math.floor(th / 0.12) % 4, hop: 0, lean: Math.sin(Math.min(1, th / 0.8) * Math.PI), pulse: 0 };
    const f = B.active && B.active[s];
    if (f && !f.ko && f.enter <= 0) {
      if (f.formLock > 0 || f.invuln > 0.55) return { kind: "evolve", frame: Math.floor(t * 12) % 4, hop: Math.abs(Math.sin(t * 9)), lean: 0, pulse: 0.5 + 0.5 * Math.sin(t * 14) };
      if (f.dodgeT > 0) return { kind: "dodge", frame: Math.floor(t * 18) % 2 ? 1 : 3, hop: 0.55, lean: -1, pulse: 0 };
      if (f.cast || f.dash) {
        const wind = !f.dash && f.cast && !f.cast.fired;
        return { kind: "move", frame: wind ? 1 : Math.floor(t * 14) % 4, hop: wind ? 0.15 : Math.abs(Math.sin(t * 16)), lean: wind ? -0.55 : 1, pulse: 0 };
      }
    }
    return { kind: "idle", frame: 0, hop: 0, lean: 0, pulse: 0 };
  }
  // Trainers beside the pitch: red corner (left) faces east, blue corner (right) faces west. Sending out a Pokémon plays the
  // walk frames with a short step toward the field; a big crowd moment makes them hop.
  function drawTrainers(B, t, shadowOnly) {
    const A = B.arena, K = Arenas.K, f = A.field, k = SPR;
    if (!shadowOnly) sprT(1);
    for (let s = 0; s < 2; s++) {
      if (!trainerMostlyOn(trainerBox(B, s))) continue;
      const tr = B.trainers && B.trainers[s], rec = B.trainerRecs && B.trainerRecs[s];
      const mot = trainerMotion(B, s, t), dir = s === 0 ? 1 : -1;
      const spot = B.trainerSpot && B.trainerSpot[s];
      const wx = spot ? spot.x : (s === 0 ? f.x - 22 : f.x + f.w + 22);
      const wy = spot ? spot.y : f.cy;
      const x = Math.round(wx / K - ix), y = Math.round(wy / K - iy);
      if (!shadowOnly) { anchor(wx, wy + 12 * K); sprT(1); } // 足もと（立ち位置の 12 art px 下）を基準にする
      if (!tr || !rec || !rec.frames) { // fallback while the sheet loads (16×24 procedural)
        const body = trainerBodyPx(), dh = body, dw = Math.round(16 * body / 24);
        if (shadowOnly) {
          Pixel.ellipse(wctx, x, y + 12, Math.max(3, Math.round(dw * 0.25)), Math.max(1, Math.round(dh * 0.08)), "rgba(0,0,0,0.35)");
        } else {
          sctx.save(); sctx.translate(x * k, (y + 12 - dh) * k); if (s === 1) sctx.scale(-1, 1);
          sctx.drawImage(trainerSprites[s][0], -Math.round(dw * k / 2), 0, Math.round(dw * k), Math.round(dh * k)); sctx.restore();
        }
        continue;
      }
      const sz = trainerSpriteSize(tr, rec), sc = sz.sc, dw = sz.dw, dh = sz.dh;
      if (shadowOnly) {
        Pixel.ellipse(wctx, x, y + 12, Math.max(3, Math.round(dw * 0.25)), Math.max(1, Math.round(dh * 0.06)), "rgba(0,0,0,0.35)");
        continue;
      }
      const seq = s === 0 ? tr.east : tr.west, flip = s === 0 && tr.ef;
      const fi = seq[mot.frame % seq.length];
      const step = Math.round(mot.lean * 4 * sc * k) * dir;
      const hop = mot.kind === "idle" && B.hype > 0.8
        ? Math.round(Math.abs(Math.sin(t * 9 + s * 1.3)) * Math.max(1, dh * 0.04) * k)
        : Math.round(mot.hop * Math.max(2, dh * 0.16) * k);
      const sw = Math.max(1, Math.round(dw * k)), sh = Math.max(1, Math.round(dh * k));
      const drop = Math.round((sz.gap || 0) * sc * k);
      sctx.save(); sctx.translate(x * k + step, (y + 12) * k - sh + drop - hop);
      if (flip) sctx.scale(-1, 1);
      sctx.imageSmoothingEnabled = false;
      // 体（首から下）はそのまま、頭（首から上）だけ小さく。頭は首の位置に下端をそろえ、左右は中央のまま。
      const nk = trainerNeck(tr, rec), neckY = Math.round(nk / tr.h * sh), img = rec.frames[0].canvas;
      sctx.drawImage(img, fi * tr.w, nk, tr.w, tr.h - nk, -Math.round(sw / 2), neckY, sw, sh - neckY);
      const hw = Math.max(1, Math.round(sw * TRAINER_HEAD)), hh = Math.max(1, Math.round(neckY * TRAINER_HEAD));
      sctx.drawImage(img, fi * tr.w, 0, tr.w, nk, -Math.round(hw / 2), neckY - hh, hw, hh);
      sctx.restore();
    }
  }

  // ---- lighting ---------------------------------------------------------------------------------------------
  function lighting(B, t) {
    const A = B.arena, fx = B.fx, K = Arenas.K;
    const on = B.lightsOn === undefined ? 1 : B.lightsOn;
    let [r, g, b] = (window.ARENA_HD_OFF && A.def.ambient0) || A.ambient;
    const dark = A.night ? U.lerp(0.12, 1, on) : U.lerp(0.35, 1, on);
    const w = B.weather.kind;
    if (w === "rain") { r *= 0.8; g *= 0.85; b *= 0.95; }
    if (w === "sand") { b *= 0.82; }
    if (w === "sun") { r *= 1.12; g *= 1.06; }
    const pk = psyK(B);
    if (pk > 0) { r *= 1 - 0.55 * pk; g *= 1 - 0.6 * pk; b *= 1 - 0.15 * pk; }
    lctx.setTransform(1, 0, 0, 1, 0, 0); lctx.globalCompositeOperation = "source-over"; lctx.globalAlpha = 1;
    lctx.fillStyle = `rgb(${Math.min(255, r * dark * 127.5) | 0},${Math.min(255, g * dark * 127.5) | 0},${Math.min(255, b * dark * 127.5) | 0})`;
    lctx.fillRect(0, 0, LW, LH);
    lctx.setTransform(1 / K, 0, 0, 1 / K, -ix, -iy);
    lctx.globalCompositeOperation = "lighter";
    A.lights.forEach((l, i) => {
      const lit = l.flood ? (on > 0.2 + i * 0.15 ? 1 : 0) : on;
      if (!lit) return;
      const k = l.i * 0.5 * (l.flicker ? 0.88 + 0.12 * Math.sin(t * 11 + i * 3) : 1);
      const gr = lctx.createRadialGradient(l.x, l.y, 0, l.x, l.y, l.r);
      gr.addColorStop(0, `rgba(${l.color[0]},${l.color[1]},${l.color[2]},${k})`); gr.addColorStop(1, `rgba(${l.color[0]},${l.color[1]},${l.color[2]},0)`);
      lctx.fillStyle = gr; lctx.fillRect(l.x - l.r, l.y - l.r, l.r * 2, l.r * 2);
    });
    lctx.globalAlpha = A.night ? 0.5 : 0.22; fx.drawLights(lctx); lctx.globalAlpha = 1;
    if (pk > 0 && B.psy.f) { // the caster glows through the gloom
      const f = B.psy.f, R = 260, gr = lctx.createRadialGradient(f.x, f.y, 0, f.x, f.y, R);
      gr.addColorStop(0, `rgba(190,140,255,${0.55 * pk})`); gr.addColorStop(1, "rgba(120,80,255,0)");
      lctx.fillStyle = gr; lctx.fillRect(f.x - R, f.y - R, R * 2, R * 2);
    }
    if (B.activeClash) {
      const c = B.activeClash, R = 210 + Math.sin(t * 36) * 24;
      const gr = lctx.createRadialGradient(c.x, c.y - 12, 0, c.x, c.y - 12, R);
      gr.addColorStop(0, "rgba(255,250,210,0.85)");
      gr.addColorStop(0.45, "rgba(120,210,255,0.42)");
      gr.addColorStop(1, "rgba(0,0,0,0)");
      lctx.fillStyle = gr; lctx.fillRect(c.x - R, c.y - 12 - R, R * 2, R * 2);
    }
    // fireflies / embers light their surroundings a little
    for (const p of fx.parts) if (p.light) { const gr = lctx.createRadialGradient(p.x, p.y - p.z, 0, p.x, p.y - p.z, 40); gr.addColorStop(0, "rgba(200,255,120,0.25)"); gr.addColorStop(1, "rgba(0,0,0,0)"); lctx.fillStyle = gr; lctx.fillRect(p.x - 40, p.y - p.z - 40, 80, 80); }
    lctx.globalCompositeOperation = "source-over";
  }

  // ---- weather (art space, on fx so it dithers & blooms) ------------------------------------------------------
  function weather(B, t) {
    const k = B.weather.kind;
    if (!k) return;
    const c = fctx, n = Math.round(LW * LH / 900);
    if (k === "rain") { c.fillStyle = "rgba(190,215,255,0.8)"; for (let i = 0; i < n * 2; i++) { const x = (i * 97.3 + t * 40) % LW, y = (i * 53.7 + t * 420 * (0.8 + (i % 5) * 0.1)) % LH; c.fillRect(x | 0, y | 0, 1, 4); } }
    if (k === "sand") { for (let i = 0; i < n * 3; i++) { c.fillStyle = i % 3 ? "rgba(232,205,150,0.85)" : "rgba(200,165,110,0.85)"; const x = (i * 131.7 + t * (200 + (i % 7) * 30)) % (LW + 20) - 10, y = (i * 71.3 + Math.sin(t + i) * 6 + t * 20) % LH; c.fillRect(x | 0, y | 0, 3, 1); } }
    if (k === "hail") { c.fillStyle = "rgba(245,252,255,0.95)"; for (let i = 0; i < n * 2; i++) { const x = (i * 131.7 + t * 30) % LW, y = (i * 71.3 + t * (160 + (i % 5) * 20)) % LH; c.fillRect(x | 0, y | 0, 2, 2); } }
    if (k === "sun") { c.fillStyle = "rgba(255,240,180,0.6)"; for (let i = 0; i < n / 2; i++) { const x = (i * 131.7 + t * 6) % LW, y = (i * 71.3 - t * 8 + 1000) % LH; c.fillRect(x | 0, y | 0, 1, 1); } }
  }

  // 3D のときの天気: 画面にそのまま降らせる（地面の面に描くと、雨や砂が地面をすべって見える）
  function weather3(B, t) {
    const k = B.weather.kind;
    if (!k) return;
    const c = sctx, u = Math.max(1, Math.round(S * cam.zoom * 0.5)), n = Math.round(W * H / (u * u) / 3600);
    c.setTransform(1, 0, 0, 1, 0, 0); c.globalAlpha = 1;
    if (k === "rain") { c.fillStyle = "rgba(190,215,255,0.8)"; for (let i = 0; i < n * 2; i++) { const x = (i * 97.3 * u + t * 60 * u) % W, y = (i * 53.7 * u + t * 620 * u * (0.8 + (i % 5) * 0.1)) % H; c.fillRect(x | 0, y | 0, u, 5 * u); } }
    if (k === "sand") { for (let i = 0; i < n * 3; i++) { c.fillStyle = i % 3 ? "rgba(232,205,150,0.85)" : "rgba(200,165,110,0.85)"; const x = (i * 131.7 * u + t * u * (300 + (i % 7) * 45)) % (W + 20 * u) - 10 * u, y = (i * 71.3 * u + Math.sin(t + i) * 6 * u + t * 30 * u) % H; c.fillRect(x | 0, y | 0, 4 * u, u); } }
    if (k === "hail") { c.fillStyle = "rgba(245,252,255,0.95)"; for (let i = 0; i < n * 2; i++) { const x = (i * 131.7 * u + t * 45 * u) % W, y = (i * 71.3 * u + t * u * (240 + (i % 5) * 30)) % H; c.fillRect(x | 0, y | 0, 2 * u, 2 * u); } }
    if (k === "sun") { c.fillStyle = "rgba(255,240,180,0.6)"; for (let i = 0; i < n / 2; i++) { const x = (i * 131.7 * u + t * 9 * u) % W, y = (i * 71.3 * u - t * 12 * u + 100000) % H; c.fillRect(x | 0, y | 0, u, u); } }
  }

  // ---- overlay (device px) ---------------------------------------------------------------------------------------
  function overlay(B, t) {
    const c = octx, K = Arenas.K, fx = B.fx, P = Math.max(2, Math.round(S * 0.75));
    c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, W, H);
    c.imageSmoothingEnabled = false;
    // mini bars
    for (const f of B.active) {
      if (!f || f.ko || f.enter > 0.3 || !f.active || f.burrow) continue;
      const fr = frameOf(f), hgt = fr ? fr.bh * (f.fit || 1) : f.height / K;
      const s = worldToScreen(f.x, f.y, f.z);
      const bw = 18 * P, bh = 2 * P, x = Math.round(s.x - bw / 2), y = Math.round(s.y - (hgt + 8 + (f.bob || 0)) * sclAt(f.x, f.y));
      const r = Math.max(0, f.hp / f.maxHp);
      c.fillStyle = "#0a0a12"; c.fillRect(x - P, y - P, bw + 2 * P, bh * 2 + 3 * P);
      c.fillStyle = "#3a3a48"; c.fillRect(x, y, bw, bh);
      c.fillStyle = r > 0.5 ? "#58d858" : r > 0.2 ? "#f8c830" : "#f04848"; c.fillRect(x, y, Math.round(bw * r / P) * P, bh);
      c.fillStyle = "#16324a"; c.fillRect(x, y + bh + P, bw, P);
      c.fillStyle = "#58c8ff"; c.fillRect(x, y + bh + P, Math.round(bw * f.energy / 100 / P) * P, P);
      if (f.side === 0 || f.side === 1) { c.fillStyle = f.side === 0 ? "#ff4848" : "#3a8bff"; c.fillRect(x - P, y - P, P, bh * 2 + 3 * P); }
      if (f.controlled && B.phase === "fight") { const ay = y - 5 * P + (Math.floor(t * 4) % 2) * P; c.fillStyle = "#fff"; for (let i = 0; i < 3; i++) c.fillRect(Math.round(s.x) - (3 - i) * P + P, ay + i * P, (3 - i) * 2 * P - P, P); }
      if (f.dizzy > 0 || f.confT > 0 || f.status === "slp") { // orbiting stars / Zs
        for (let i = 0; i < 3; i++) { const a = t * 5 + i * 2.1, sx = s.x + Math.cos(a) * 9 * P, sy = y + 3 * P + Math.sin(a) * 3 * P; Pixel.text(c, f.status === "slp" ? "Z" : "★", sx - 2.5 * P, sy, f.status === "slp" ? Math.max(1, P - 1) : Math.max(1, P - 1), f.status === "slp" ? "#c8d8ff" : "#ffe14a", "#2a1a00"); }
      }
    }
    // floaters
    for (const fl of fx.floaters) {
      if (fl.delay > 0) continue;
      const age = fl.max - fl.life, k = fl.life / fl.max;
      const s = worldToScreen(fl.x, fl.y);
      const pop = age < 0.15 ? 1 + (1 - age / 0.15) * 0.6 : 1;
      c.globalAlpha = k < 0.25 ? k / 0.25 : 1;
      if (fl.kind === "num") {
        const sc = Math.max(2, Math.round(S * (fl.size / 22) * pop));
        const tw = Pixel.textWidth(fl.text, sc);
        Pixel.text(c, fl.text, s.x - tw / 2, s.y - 3.5 * sc, sc, fl.color, "#14101c");
      } else {
        const px = Math.round(S * 5.2 * (fl.size / 12));
        c.font = `${px}px "DotGothic16", "Press Start 2P", monospace`;
        c.textAlign = "center"; c.textBaseline = "middle";
        c.lineJoin = "round"; c.lineWidth = Math.max(3, px * 0.28); c.strokeStyle = "#14101c";
        c.strokeText(fl.text, s.x, s.y); c.fillStyle = fl.color; c.fillText(fl.text, s.x, s.y);
      }
      c.globalAlpha = 1;
    }
    trainerWipes(B, t);
    monsterWipes(B, t);
  }
  // [ユーザー指定 2026-11-15] 画面の外にいるモンスターは、いる方向の画面端にワイプ（いまの姿・名前・HP・方向の矢印）で映す。
  // トレーナーのワイプと重なるときは上下によける。画面内に戻れば消える。
  function monsterWipes(B, t) {
    const c = octx, K = Arenas.K;
    for (const f of B.active) {
      if (!f || f.ko || !f.active || f.enter > 0.3 || f.burrow) continue;
      const fr = frameOf(f); if (!fr) continue;
      const zs = sclAt(f.x, f.y), bodyH = fr.bh * (f.fit || 1) * zs, s = worldToScreen(f.x, f.y, f.z);
      const cx0 = s.x, cy0 = s.y - bodyH / 2, hw = Math.max(6 * dpr, f.r / K * zs * 0.6);
      if (cx0 + hw > 0 && cx0 - hw < W && cy0 + bodyH * 0.3 > 0 && cy0 - bodyH * 0.3 < H) continue; // (partly) in the shot
      const narrow = W / dpr < 640;
      const faceH = Math.round(U.clamp(fr.bh * dpr, 40 * dpr, (narrow ? 56 : 84) * dpr)), k = Math.min(dpr, faceH / Math.max(1, fr.bh));
      const faceW = Math.round(U.clamp(fr.w * k, 40 * dpr, (narrow ? 64 : 96) * dpr));
      const pad = Math.round((narrow ? 4 : 6) * dpr), nameH = Math.round((narrow ? 12 : 16) * dpr), barH = Math.max(3, Math.round(4 * dpr));
      const boxW = faceW + pad * 2, boxH = faceH + nameH + barH + pad * 3, m = Math.round((narrow ? 8 : 12) * dpr);
      const top = Math.round((narrow ? 96 : 150) * dpr), bot = H - Math.round((narrow ? 150 : 112) * dpr);
      // edge point on the line from the screen centre to the monster
      const dx = cx0 - W / 2, dy = cy0 - H / 2, hx = W / 2 - m - boxW / 2, hy = Math.max(10, (bot - top) / 2 - boxH / 2), my = (top + bot) / 2;
      const kx = Math.abs(dx) > 1e-3 ? hx / Math.abs(dx) : 1e9, ky = Math.abs(dy) > 1e-3 ? (H / 2 - m - boxH / 2) / Math.abs(dy) : 1e9, kk = Math.min(kx, ky);
      let x = Math.round(W / 2 + dx * kk - boxW / 2), y = Math.round(U.clamp(H / 2 + dy * kk, my - hy, my + hy) - boxH / 2);
      for (const wb of wipeBox) { // step off a trainer inset drawn this frame
        if (!wb || wb.t !== t || x + boxW < wb.left - 4 || x > wb.left + wb.w + 4 || y + boxH < wb.top - 4 || y > wb.top + wb.h + 4) continue;
        y = dy < 0 ? wb.top - boxH - 8 * dpr : wb.top + wb.h + 8 * dpr;
      }
      y = Math.round(U.clamp(y, top, Math.max(top, bot - boxH)));
      const edge = f.side === 0 ? "#ff9aa8" : "#8ab4ff";
      c.fillStyle = "#000"; c.fillRect(x - 3, y - 3, boxW + 6, boxH + 6);
      c.fillStyle = edge; c.fillRect(x - 1, y - 1, boxW + 2, boxH + 2);
      c.fillStyle = "rgba(8,12,28,0.9)"; c.fillRect(x, y, boxW, boxH);
      c.save(); c.imageSmoothingEnabled = false;
      c.beginPath(); c.rect(x, y, boxW, faceH + pad * 2); c.clip();
      const feetX = x + boxW / 2, feetY = y + pad + faceH;
      c.drawImage(fr.img, Math.round(feetX - fr.ax * k), Math.round(feetY - fr.ay * k), Math.round(fr.w * k), Math.round(fr.h * k));
      if (f.flash > 0) { c.globalAlpha = Math.min(1, f.flash * 1.45); c.drawImage(Dir8.tint(fr.img, "#ffffff"), Math.round(feetX - fr.ax * k), Math.round(feetY - fr.ay * k), Math.round(fr.w * k), Math.round(fr.h * k)); }
      c.restore();
      const r = Math.max(0, f.hp / f.maxHp), by = y + pad * 2 + faceH;
      c.fillStyle = "#3a3a48"; c.fillRect(x + pad, by, faceW, barH);
      c.fillStyle = r > 0.5 ? "#58d858" : r > 0.2 ? "#f8c830" : "#f04848"; c.fillRect(x + pad, by, Math.round(faceW * r), barH);
      let px = Math.max(9, Math.round(11 * dpr)); const name = f.species.jp || f.species.name || "";
      c.font = `700 ${px}px "Hiragino Sans", "Noto Sans JP", sans-serif`; c.textAlign = "center"; c.textBaseline = "middle";
      while (px > 7 && c.measureText(name).width > boxW - 6) { px -= 1; c.font = `700 ${px}px "Hiragino Sans", "Noto Sans JP", sans-serif`; }
      const ny = by + barH + pad / 2 + nameH / 2;
      c.lineWidth = Math.max(2, dpr); c.strokeStyle = "#14101c"; c.strokeText(name, x + boxW / 2, ny); c.fillStyle = "#fff"; c.fillText(name, x + boxW / 2, ny);
      // arrow on the box edge, pointing at the monster
      const an = Math.atan2(cy0 - (y + boxH / 2), cx0 - (x + boxW / 2)), ar = Math.round(9 * dpr), pulse = 1 + 0.15 * Math.sin(t * 8);
      const axx = x + boxW / 2 + Math.cos(an) * (boxW / 2 + ar * 0.9), ayy = y + boxH / 2 + Math.sin(an) * (boxH / 2 + ar * 0.9);
      c.save(); c.translate(U.clamp(axx, ar, W - ar), U.clamp(ayy, ar, H - ar)); c.rotate(an); c.scale(pulse, pulse);
      c.beginPath(); c.moveTo(ar, 0); c.lineTo(-ar * 0.7, -ar * 0.8); c.lineTo(-ar * 0.7, ar * 0.8); c.closePath();
      c.fillStyle = edge; c.strokeStyle = "#000"; c.lineWidth = Math.max(2, dpr * 1.5); c.stroke(); c.fill(); c.restore();
    }
  }
  // Feet of a sideline trainer in device px, and the on-screen sprite box (zoom included).
  function trainerBox(B, s) {
    const A = B.arena, K = Arenas.K, f = A.field;
    const spot = B.trainerSpot && B.trainerSpot[s];
    const wx = spot ? spot.x : (s === 0 ? f.x - 22 : f.x + f.w + 22);
    const wy = spot ? spot.y : f.cy;
    const tr = B.trainers && B.trainers[s];
    const sz = trainerSpriteSize(tr, B.trainerRecs && B.trainerRecs[s]);
    const dh = sz.dh, dw = sz.dw;
    const zs = sclAt(wx, wy + 12 * K), feet = artToScreen(wx / K, wy / K + 12);
    const sw = dw * zs, sh = dh * zs;
    const drop = (sz.gap || 0) * (sz.sc || 0) * zs;
    return { left: feet.x - sw / 2, top: feet.y - sh + drop, w: sw, h: sh };
  }
  function trainerMostlyOn(box) {
    const visW = Math.min(box.left + box.w, W) - Math.max(box.left, 0);
    const visH = Math.min(box.top + box.h, H) - Math.max(box.top, 0);
    return visW > 0 && visH > 0 && visW * visH > box.w * box.h * 0.55;
  }
  // Which corner's trainer is under a client point (sideline sprite or side inset), or -1.
  const wipeBox = [null, null];
  let lastT = 0;
  function trainerAt(B, cx, cy) {
    if (!B || !B.trainers) return -1;
    const r = ovc ? ovc.getBoundingClientRect() : { left: 0, top: 0 };
    const x = (cx - r.left) * dpr, y = (cy - r.top) * dpr, pad = 10 * dpr;
    for (let s = 0; s < 2; s++) {
      const box = trainerBox(B, s);
      const inside = (b) => b && x >= b.left - pad && x <= b.left + b.w + pad && y >= b.top - pad && y <= b.top + b.h + pad;
      if (trainerMostlyOn(box) && inside(box)) return s;
      if (wipeBox[s] && lastT - wipeBox[s].t < 0.2 && inside(wipeBox[s])) return s;
    }
    return -1;
  }
  // Side inset when the sideline sprite is outside the shot. Hidden once they walk into frame.
  function trainerWipes(B, t) {
    lastT = t;
    if (!B.trainers || !B.trainers.length) return;
    const c = octx;
    for (let s = 0; s < 2; s++) {
      if (trainerMostlyOn(trainerBox(B, s))) continue;
      const tr = B.trainers[s], rec = B.trainerRecs && B.trainerRecs[s];
      const narrow = W / dpr < 640, portrait = W < H * 0.98;
      const faceH = narrow ? Math.round(30 * dpr) : Math.round(Math.min(58 * dpr, Math.max(38 * dpr, Assets.trainerArtPx() * S * 0.46)));
      const faceW = tr ? Math.max(8, Math.round(faceH * tr.w / tr.h)) : Math.round(faceH * 16 / 24);
      const pad = Math.round((narrow ? 4 : 5) * dpr), nameH = Math.round((narrow ? 13 : 14) * dpr);
      const fontName = '"Hiragino Sans", "Noto Sans JP", sans-serif';
      let fontPx = Math.max(8, Math.round((narrow ? 9 : 10) * dpr));
      let minTextW = 0;
      if (tr && tr.n) {
        c.font = `700 ${fontPx}px ${fontName}`;
        minTextW = c.measureText(tr.n).width;
      }
      const baseInner = Math.max(faceW, Math.round((narrow ? 38 : 56) * dpr));
      const maxInner = Math.round((narrow ? 76 : 102) * dpr);
      const boxW = Math.min(maxInner, Math.max(baseInner, Math.ceil(minTextW * 0.85))) + pad * 2;
      const boxH = faceH + nameH + pad * 2;
      const margin = Math.round((narrow ? 8 : 12) * dpr);
      const x = s === 0 ? margin : W - margin - boxW;
      const y = portrait ? Math.round(H * 0.25 - boxH / 2) : Math.round(H * 0.56 - boxH / 2);
      const edge = s === 0 ? "#ff9aa8" : "#8ab4ff";
      wipeBox[s] = { left: x, top: y, w: boxW, h: boxH, t };
      c.fillStyle = "#000"; c.fillRect(x - 3, y - 3, boxW + 6, boxH + 6);
      c.fillStyle = edge; c.fillRect(x - 1, y - 1, boxW + 2, boxH + 2);
      c.fillStyle = "rgba(8,12,28,0.88)"; c.fillRect(x, y, boxW, boxH);
      const mot = trainerMotion(B, s, t), dir = s === 0 ? 1 : -1;
      const dx = Math.round(x + (boxW - faceW) / 2), dy = y + pad;
      const leanPx = Math.round(mot.lean * Math.max(3 * dpr, faceW * 0.07)) * dir;
      const hopPx = Math.round(mot.hop * Math.max(2 * dpr, Math.min(pad * 0.85, faceH * 0.08)));
      const grow = 1 + mot.pulse * 0.06;
      c.save();
      c.imageSmoothingEnabled = false;
      c.beginPath(); c.rect(x, y, boxW, boxH - nameH); c.clip();
      const feetX = dx + faceW / 2 + leanPx, feetY = dy + faceH - hopPx;
      c.translate(feetX, feetY);
      if (!tr || !rec || !rec.frames) {
        if (s === 1) c.scale(-grow, grow); else c.scale(grow, grow);
        c.drawImage(trainerSprites[s][mot.kind === "idle" ? 0 : 1], -faceW / 2, -faceH, faceW, faceH);
      } else {
        const seq = s === 0 ? tr.east : tr.west, flip = s === 0 && tr.ef;
        const fi = seq[mot.frame % seq.length];
        if (flip) c.scale(-grow, grow); else c.scale(grow, grow);
        c.drawImage(rec.frames[0].canvas, fi * tr.w, 0, tr.w, tr.h, -faceW / 2, -faceH, faceW, faceH);
      }
      c.restore();
      if (tr && tr.n) {
        let px = fontPx;
        c.font = `700 ${px}px ${fontName}`;
        c.textAlign = "center"; c.textBaseline = "middle";
        while (px > 6 && c.measureText(tr.n).width > boxW - 6) { px -= 1; c.font = `700 ${px}px ${fontName}`; }
        const ny = y + boxH - nameH / 2;
        c.lineWidth = Math.max(2, dpr); c.strokeStyle = "#14101c";
        c.strokeText(tr.n, x + boxW / 2, ny);
        c.fillStyle = "#ffe9a0"; c.fillText(tr.n, x + boxW / 2, ny);
      }
    }
  }

  // Menu thumbnail of an arena (art px, scaled with nearest neighbour).
  const thumbs = new Map();
  function thumbnail(id) {
    if (thumbs.has(id)) return thumbs.get(id);
    const A = Arenas.create(id, 3), c = U.canvas(200, 112), x = c.getContext("2d");
    x.imageSmoothingEnabled = false;
    const sw = 300, sh = 168, sx = A.af.x + A.af.w / 2 - sw / 2, sy = A.af.y + A.af.h / 2 - sh / 2;
    x.drawImage(A.frames[0], sx, sy, sw, sh, 0, 0, 200, 112);
    for (const p of A.props) x.drawImage(p.sprite, (p.x / 2 - p.ax - sx) * 200 / sw, (p.y / 2 - p.ay - sy) * 112 / sh, p.sprite.width * 0.67, p.sprite.height * 0.67);
    const url = c.toDataURL("image/png");
    thumbs.set(id, url);
    return url;
  }

  return { frameOf, init, resize, cam, setUserZoom, zoomProfile, updateCamera, toWorld, worldToScreen, draw, thumbnail, scale, trainerAt };
})();
