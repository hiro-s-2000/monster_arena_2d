// Body motion tracks for wind-ups, releases and flinches — the "acting" between sprite frames.
// Timing follows the PMD step tables (crouch / float before the hit frame, a lunge of about a body length
// on the hit, 2-tick steps back), rebuilt as keyframes: [time s, forward, up, width] with forward / up in
// world units along the motion's direction and width as a squash factor (>1 = squat, <1 = stretch).
"use strict";

const Motion = (() => {
  const SLASH = /(claw|slash|cut|blade|edge|sword|chop|cross-poison|scissor|fury-cutter|aerial-ace|night|sacred-sword|psycho-cut|leaf-blade|x-scissor|dragon-tail|iron-tail|smart-strike)/;
  const BREATH = /(breath|flamethrower|ember|fire-blast|water-gun|hydro|bubble|spit|smog|sludge|mud|acid|powder|spore|sing|supersonic|voice|boomburst|scald|ice-beam|aurora|blizzard|icy-wind|heat-wave|inferno|dark-pulse|dragon-pulse)/;

  // Motion family for a move spec.
  function family(s) {
    const phys = s.cat === "p" && s.power > 0;
    switch (s.kind) {
      case "melee": return SLASH.test(s.slug) ? "strike" : "attack";
      case "dash": case "leap": return "crouch";
      case "proj": case "spread": return phys ? "throw" : "shoot";
      case "status": return "shoot";
      case "beam": return BREATH.test(s.slug) || s.cone ? "breath" : "charge";
      case "aoe": return phys ? "stomp" : "charge";
      case "strike": case "zone": case "erupt": case "spiral": return phys ? "strike" : "charge";
      case "psybreak": case "uplift": return "rise";
      case "grab": return "reach";
      case "teleport": return "charge";
      default: return "twirl"; // buff / heal / shield / weather / smoke
    }
  }

  // Keyframes for a cast: W = wind-up, H = how long the release pose is held (beams, grabs, lifts).
  function forCast(s, W, H) {
    const fam = family(s), F = U.clamp(6 + s.power / 10, 6, 20);
    const m = { fam, shake: null, spin: null, hopAt: -1 };
    switch (fam) {
      case "attack": m.keys = [[0, 0, 0, 1], [W * 0.5, -8, 2, 1.14], [W * 0.85, -10, 4, 1.18], [W, 30, 0, 0.88], [W + 0.08, 22, 0, 0.94], [W + 0.26, 0, 0, 1]]; break;
      case "strike": m.keys = [[0, 0, 0, 1], [W * 0.7, -12, 10, 0.88], [W * 0.9, -10, 12, 0.86], [W, 22, 0, 1.16], [W + 0.12, 14, 0, 1.04], [W + 0.34, 0, 0, 1]]; break;
      case "crouch": m.keys = [[0, 0, 0, 1], [W * 0.7, -6, 0, 1.2], [W, 6, 2, 0.86], [W + 0.2, 0, 0, 1]]; m.shake = [W * 0.3, W, 0.6]; break;
      case "throw": m.keys = [[0, 0, 0, 1], [W * 0.6, -10, 6, 0.9], [W, 14, 0, 1.12], [W + 0.1, 8, 0, 1.02], [W + 0.3, 0, 0, 1]]; break;
      case "shoot": m.keys = [[0, 0, 0, 1], [W * 0.55, -6, 0, 1.1], [W, 8, 0, 0.94], [W + 0.12, 5, 0, 0.98], [W + 0.3, 0, 0, 1]]; break;
      case "breath": m.keys = [[0, 0, 0, 1], [W * 0.6, -8, 3, 0.9], [W, 8, 0, 1.08], [W + H, 6, 0, 1.04], [W + H + 0.25, 0, 0, 1]]; break;
      case "charge": m.keys = [[0, 0, 0, 1], [W * 0.3, -2, 0, 1.1], [W, 0, F, 0.92], [W + 0.06, 8, F, 0.9], [W + Math.max(0.1, H), 5, F, 1], [W + Math.max(0.1, H) + 0.3, 0, 0, 1]]; m.shake = [W * 0.3, W, 0.7]; break;
      case "rise": m.keys = [[0, 0, 0, 1], [W, 0, 10, 0.92], [W + H * 0.4, 0, 22, 0.94], [W + H, 4, 18, 1], [W + H + 0.35, 0, 0, 1]]; m.shake = [W * 0.5, W + H * 0.5, 1.2]; break;
      case "reach": m.keys = [[0, 0, 0, 1], [W * 0.6, -6, 0, 1.08], [W, 12, 0, 0.92], [W + H, 10, 4, 0.95], [W + H + 0.25, 0, 0, 1]]; break;
      case "stomp": m.keys = [[0, 0, 0, 1], [W * 0.3, 0, 0, 1.22], [W * 0.38, 0, 0, 0.86], [W + 0.06, 0, 0, 1.25], [W + 0.25, 0, 0, 1]]; m.hopAt = W * 0.36; break;
      default: m.keys = [[0, 0, 0, 1], [W * 0.4, 0, 0, 1.16], [W, 0, 10, 0.88], [W + 0.25, 0, 6, 0.96], [W + 0.45, 0, 0, 1]]; m.spin = [W * 0.4, W + 0.3]; break;
    }
    return m;
  }

  // Knocked back along `ang`: jolt away, small rise, settle.
  const HURT = [[0, 0, 0, 1], [0.04, 12, 5, 0.86], [0.16, 6, 2, 1.05], [0.32, 0, 0, 1]];

  function start(f, m, dir) { f.mot = { keys: m.keys, t: 0, dir, shake: m.shake, spin: m.spin, hopAt: m.hopAt, hopped: false, end: m.keys[m.keys.length - 1][0] }; }
  function startCast(f, s, W, H) { start(f, forCast(s, W, H), f.aim); }
  function hurt(f, ang) { if (!f.mot || f.mot.hurt || !f.cast) { start(f, { keys: HURT }, ang); f.mot.hurt = true; } }

  // Advance and write f.pose = { fwd, up, sq, shake } (all zero / 1 when idle).
  function update(f, dt) {
    const M = f.mot, P = f.pose || (f.pose = { fwd: 0, up: 0, sq: 1, shake: 0, dir: 0 });
    f.spinning = false;
    if (!M) { P.fwd = U.damp(P.fwd, 0, 20, dt); P.up = U.damp(P.up, 0, 20, dt); P.sq = U.damp(P.sq, 1, 20, dt); P.shake = 0; return; }
    M.t += dt;
    const K = M.keys;
    let i = 0; while (i < K.length - 2 && M.t > K[i + 1][0]) i++;
    const a = K[i], b = K[i + 1], k = U.smooth(U.clamp((M.t - a[0]) / Math.max(1e-3, b[0] - a[0]), 0, 1));
    const reach = f.reach || 1;
    P.fwd = U.lerp(a[1], b[1], k) * reach; P.up = U.lerp(a[2], b[2], k); P.sq = U.lerp(a[3], b[3], k); P.dir = M.dir;
    P.shake = M.shake && M.t >= M.shake[0] && M.t <= M.shake[1] ? M.shake[2] : 0;
    f.spinning = !!(M.spin && M.t >= M.spin[0] && M.t <= M.spin[1]);
    if (M.hopAt >= 0 && !M.hopped && M.t >= M.hopAt) { M.hopped = true; f.hop(45); }
    if (M.t >= M.end) f.mot = null;
  }

  return { family, forCast, start, startCast, hurt, update };
})();
