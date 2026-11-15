// Competitive tactical CPU controller:
//   1. Archetype spacing (Zoner kites at 240–300, Rusher weaves in at 65–95 & steps out on cooldown, All-rounder adapts)
//   2. Telegraph & hazard evasion (steps out of windup cones/corridors/zones before firing, perpendicular dodge)
//   3. Whiff-punish / counter-attack ("差し返し" during enemy recovery frames) + Beam Clash counter-beams & mashing.
"use strict";

const AI = (() => {
  const RANGED_KINDS = new Set(["proj", "spread", "beam", "psybreak", "strike", "erupt", "zone"]);
  const CLOSE_KINDS = new Set(["melee", "dash", "leap", "spiral", "dig", "aoe", "grab"]);

  function classifyRole(f) {
    let ranged = 0, close = 0;
    for (const m of f.moves) {
      if (!m || !m.power) continue;
      if (RANGED_KINDS.has(m.kind)) ranged++;
      else if (CLOSE_KINDS.has(m.kind)) close++;
    }
    if (ranged >= 2 && (close === 0 || f.stats.spa >= f.stats.atk * 0.95)) return "zoner";
    if (close >= 2 && (ranged === 0 || f.stats.atk >= f.stats.spa * 0.95)) return "rusher";
    return "allrounder";
  }

  function state(f) {
    if (!f.ai || f.ai.speciesSlug !== f.species.slug) {
      f.ai = {
        speciesSlug: f.species.slug,
        role: classifyRole(f),
        think: 0,
        want: -1,
        strafe: U.chance(0.5) ? 1 : -1,
        strafeT: U.rand(0.7, 1.6),
        dodgeT: 0,
        aimNoise: 0,
        swapT: 4,
        age: 0,
        punishing: false,
      };
    }
    return f.ai;
  }

  function hasReadyAttack(f, kinds) {
    for (let i = 0; i < f.moves.length; i++) {
      const m = f.moves[i];
      if (m && m.power && (!kinds || kinds.has(m.kind)) && f.cds[i] <= 0.08 && f.energy >= m.cost) return true;
    }
    return false;
  }

  function findReadyBeam(f) {
    for (let i = 0; i < f.moves.length; i++) {
      const m = f.moves[i];
      if (m && (m.kind === "beam" || m.kind === "psybreak") && f.cds[i] <= 0 && f.energy >= m.cost) return i;
    }
    return -1;
  }

  function scoreMove(f, s, i, tgt, d, B, role, punish) {
    if (f.cds[i] > 0 || f.energy < s.cost) return -1;
    const hp = f.hpRatio;
    if (s.kind === "heal") return hp < 0.45 && d > 130 ? 85 + (0.45 - hp) * 210 : -1;
    if (s.kind === "shield") return (tgt.cast && !tgt.cast.fired && (tgt.cast.spec.power || 0) >= 70 && d < 380) || B.incoming(f) ? 92 : -1;
    if (s.kind === "buff") return d > 210 && !tgt.cast && !punish && hp > 0.55 && Object.values(f.stages).reduce((a, b) => a + b, 0) < 3 ? 52 : -1;
    if (s.kind === "weather") return B.weather.kind ? -1 : 42;
    if (s.kind === "status") return tgt.status ? -1 : (punish ? 62 : 48);
    const eff = Data.effectiveness(s.type, tgt.species.types);
    if (eff === 0) return -1;
    const stab = f.species.types.includes(s.type) ? 1.5 : 1;
    const atk = s.cat === "p" ? f.stats.atk : f.stats.spa;
    let v = ((Data.weightPower(s.slug, f, tgt) || s.power) * eff * stab * atk * (s.cat === "p" && !Data.weightPower(s.slug, f, tgt) ? Data.weightMul(f, tgt) : 1)) / 100;
    if (s.hits > 1) v *= 1.2;

    // Archetype & tactical distance weighting
    if (role === "zoner") {
      if (RANGED_KINDS.has(s.kind)) v *= d > 135 ? 1.28 : 1.05;
      if (s.kind === "dash" && tgt.hpRatio > 0.35 && d > 150) v *= 0.55; // zoners don't recklessly dive healthy brawlers
      if (s.kind === "melee" && d > 120) v *= 0.45;
    } else if (role === "rusher") {
      if (s.kind === "dash" || s.kind === "leap") v *= d > 110 && d < 320 ? 1.35 : 0.95;
      if (s.kind === "melee" && d < 140) v *= 1.3;
    }
    if ((s.kind === "proj" || s.kind === "spread") && d < 58) v *= 0.65;
    if (s.kind === "strike" && Math.hypot(tgt.vx, tgt.vy) > 130 && !punish) v *= 0.7;
    if (s.kind === "aoe" && d > (s.r1 || 180) * 1.05) v *= 0.28;

    // Whiff-punish ("差し返し") bonus: fast gap-closers, accurate snipes, and heavy punishes surge in priority!
    if (punish) {
      if (s.kind === "dash" && d > 55 && d < (s.dist || 260) * 1.05) v *= 1.65;
      else if (s.kind === "melee" && d < f.r + tgt.r + (s.r || 70) + 28) v *= 1.6;
      else if (s.kind === "proj" || s.kind === "beam" || s.kind === "leap") v *= 1.45;
    }
    return v;
  }

  // Pick a safe evasion vector by sampling candidate directions around `f`.
  function pickSafeVector(B, f, baseAngle) {
    const fl = B.arena.field, margin = f.r + 36;
    let bestAng = baseAngle, bestScore = -1e9;
    for (let k = 0; k < 10; k++) {
      const an = baseAngle + (k / 10) * U.TAU;
      const nx = f.x + Math.cos(an) * 78, ny = f.y + Math.sin(an) * 58;
      let sc = Math.cos(U.angDiff(an, baseAngle)) * 1.1;
      if (nx < fl.x + margin || nx > fl.x + fl.w - margin) sc -= 2.5;
      if (ny < fl.y + margin || ny > fl.y + fl.h - margin) sc -= 2.5;
      const dg = B.dangerAt(nx, ny, f);
      if (dg) sc -= dg.w * 2.4;
      if (sc > bestScore) { bestScore = sc; bestAng = an; }
    }
    return { x: Math.cos(bestAng), y: Math.sin(bestAng) };
  }

  function act(f, B, dt, skill) {
    const st = state(f);
    const a = { mx: 0, my: 0, aim: null, skill: [false, false, false, false], skillHeld: [false, false, false, false], dodge: false, swap: false, mega: false, pause: false };
    const tgt = B.nearestEnemy(f);
    if (!tgt) {
      const dg = B.dangerAt(f.x, f.y, f);
      if (dg) {
        const esc = pickSafeVector(B, f, Math.atan2(dg.y, dg.x));
        a.mx = esc.x; a.my = esc.y;
      }
      return a;
    }

    // Mash during an active Beam Clash involving `f`!
    if (B.activeClash && (B.activeClash.a.owner === f || B.activeClash.b.owner === f)) {
      a.aim = U.angle(f.x, f.y, B.activeClash.x, B.activeClash.y);
      if (Math.random() < (0.35 + skill * 0.55)) a.skill[0] = true;
      return a;
    }

    const d = U.dist(f.x, f.y, tgt.x, tgt.y);
    const toT = U.angle(f.x, f.y, tgt.x, tgt.y);
    st.think -= dt; st.strafeT -= dt; st.dodgeT -= dt; st.swapT -= dt;
    if (st.megaAt === undefined) st.megaAt = U.rand(1.2, 3.4) + (1 - skill) * 2;
    st.age += dt;
    if (!f.megaDone && B.canMega(f) && !f.cast && (st.age > st.megaAt || f.hpRatio < 0.6)) a.mega = true;
    if (st.strafeT <= 0) { st.strafeT = U.rand(0.65, 1.75); if (U.chance(0.48)) st.strafe *= -1; }

    // Detect enemy vulnerability / whiff recovery ("差し返し" window)
    const enemyWhiffed = (tgt.recoverT || 0) > 0.06 || tgt.hitstun > 0.05 || (tgt.cast && tgt.cast.fired && (tgt.cast.spec.kind === "melee" || tgt.cast.spec.kind === "aoe") && d > 110);
    if (enemyWhiffed && !st.punishing && skill >= 0.45) {
      st.punishing = true;
      st.think = 0; // snap-react to punish window!
    } else if (!enemyWhiffed) {
      st.punishing = false;
    }

    // Beam Clash counter-play: if enemy winds up or fires a beam at us, clash back with our own beam if ready!
    const enemyBeam = (tgt.cast && !tgt.cast.fired && (tgt.cast.spec.kind === "beam" || tgt.cast.spec.kind === "psybreak")) ||
      B.attacks.some((atk) => atk.owner === tgt && (atk.kind === "beam" || atk.kind === "psybreak"));
    if (enemyBeam && d > 130 && !f.cast && !f.dash) {
      const bi = findReadyBeam(f);
      if (bi >= 0 && U.chance(0.42 + skill * 0.38)) {
        a.aim = toT;
        a.skill[bi] = true;
        return a;
      }
    }

    // Re-score moves
    if (st.think <= 0) {
      st.think = (enemyWhiffed ? U.rand(0.04, 0.1) : U.rand(0.09, 0.22)) / (0.45 + skill * 0.55);
      let best = -1, bi = -1;
      f.moves.forEach((s, i) => {
        const v = scoreMove(f, s, i, tgt, d, B, st.role, enemyWhiffed) * U.rand(0.86, 1.14);
        if (v > best) { best = v; bi = i; }
      });
      st.want = bi;
      st.aimNoise = U.rand(-1, 1) * (1 - skill) * 0.24;
      if (st.swapT <= 0 && f.hpRatio < 0.28 && B.canSwap(f.side)) {
        const better = B.bestSwap(f.side, tgt);
        if (better && U.chance(0.35 + skill * 0.4)) { a.swap = true; st.swapT = 12; }
      }
    }

    const want = st.want >= 0 ? f.moves[st.want] : null;
    // Compute archetype-aware preferred spacing ("間合い管理")
    let prefMin = 130, prefMax = 210;
    if (st.role === "zoner") {
      prefMin = 215; prefMax = 295;
      if (want && CLOSE_KINDS.has(want.kind) && enemyWhiffed) { prefMin = 65; prefMax = 115; }
    } else if (st.role === "rusher") {
      const readyClose = hasReadyAttack(f, CLOSE_KINDS);
      if (readyClose || enemyWhiffed) {
        prefMin = 52; prefMax = want && want.kind === "dash" ? Math.min(210, want.dist * 0.8) : f.r + tgt.r + (want && want.r ? want.r * 0.72 : 54);
      } else {
        // Step back out to mid-range while melee cooldowns / energy recover instead of face-tanking!
        prefMin = 145; prefMax = 195;
      }
    } else {
      // All-rounder adapts to chosen move & readiness
      if (want && (want.kind === "melee" || want.kind === "aoe" || want.kind === "spiral")) {
        prefMin = 56; prefMax = f.r + tgt.r + (want.r || want.r1 || 75) * 0.72;
      } else if (want && want.kind === "dash") {
        prefMin = 95; prefMax = (want.dist || 250) * 0.82;
      } else {
        prefMin = 185; prefMax = 255;
      }
    }
    if (f.hpRatio < 0.28 && want && want.kind === "heal") { prefMin = 260; prefMax = 360; }

    let mv = 0;
    if (d > prefMax) mv = 1;
    else if (d < prefMin) mv = -1;

    // Diagonal weave approach so linear shots miss, crisp lateral orbit when inside sweet-spot band
    const sx = -Math.sin(toT) * st.strafe, sy = Math.cos(toT) * st.strafe;
    const strafeAmt = mv === 0 ? 0.95 : (mv > 0 && st.role === "rusher" ? 0.62 : 0.48);
    a.mx = Math.cos(toT) * mv + sx * strafeAmt;
    a.my = Math.sin(toT) * mv + sy * strafeAmt;

    // Keep out of arena corners so AI doesn't trap itself on walls
    const fl = B.arena.field, wallPad = f.r + 44;
    if (f.x < fl.x + wallPad) a.mx += 0.85; else if (f.x > fl.x + fl.w - wallPad) a.mx -= 0.85;
    if (f.y < fl.y + wallPad) a.my += 0.85; else if (f.y > fl.y + fl.h - wallPad) a.my -= 0.85;

    // Proactive hazard & telegraph evasion ("危険回避")
    const dangerHere = B.dangerAt(f.x, f.y, f);
    const dangerAhead = B.dangerAt(f.x + a.mx * 56, f.y + a.my * 42, f);
    if (dangerHere) {
      const esc = pickSafeVector(B, f, Math.atan2(dangerHere.y, dangerHere.x));
      a.mx = esc.x * 1.4 + sx * 0.25;
      a.my = esc.y * 1.4 + sy * 0.25;
    } else if (dangerAhead) {
      a.mx = sx * 1.1 - dangerAhead.x * 0.4;
      a.my = sy * 1.1 - dangerAhead.y * 0.4;
    }

    // Steer around solid props
    for (const p of B.arena.props) {
      if (!p.alive) continue;
      const pd = U.dist(f.x, f.y, p.x, p.y);
      if (pd < p.r + f.r + 38) {
        const an = U.angle(p.x, p.y, f.x, f.y);
        a.mx += Math.cos(an) * 0.85; a.my += Math.sin(an) * 0.85;
      }
    }
    const ml = Math.hypot(a.mx, a.my);
    if (ml > 1) { a.mx /= ml; a.my /= ml; }

    // Predictive lead aim (tighter on whiff-punish!)
    const lead = want && (want.kind === "proj" || want.kind === "spread") ? Math.min(0.45, d / Math.max(240, want.speed || 500)) : (want && want.kind === "dash" ? 0.12 : 0.06);
    const noise = enemyWhiffed ? st.aimNoise * 0.35 : st.aimNoise;
    a.aim = U.angle(f.x, f.y, tgt.x + tgt.vx * lead, tgt.y + tgt.vy * lead) + noise;

    // Timely i-frame dodge against incoming projectiles, dashes & late windup telegraphs
    const inc = B.incoming(f);
    if (inc && st.dodgeT <= 0 && f.dodgeCd <= 0 && f.energy >= 18) {
      if (U.chance(0.38 + skill * 0.56)) {
        a.dodge = true;
        st.dodgeT = U.lerp(1.3, 0.58, skill);
        if (inc.ux !== undefined) {
          const esc = pickSafeVector(B, f, Math.atan2(inc.uy, inc.ux));
          a.mx = esc.x; a.my = esc.y;
        } else if (inc.vx !== undefined) {
          const pa = Math.atan2(inc.vy, inc.vx) + st.strafe * Math.PI * 0.5;
          const esc = pickSafeVector(B, f, pa);
          a.mx = esc.x; a.my = esc.y;
        } else {
          a.mx = sx; a.my = sy;
        }
        return a;
      }
    }

    // Execute attack when in range (hold off slightly if dodging out of severe immediate danger unless punishing!)
    if (want && !f.cast && !f.dash && (!dangerHere || dangerHere.w < 2.6 || want.kind === "dash" || want.kind === "leap" || want.kind === "shield")) {
      const inRange =
        want.kind === "melee" ? d < f.r + tgt.r + want.r * 0.92 :
        want.kind === "dash" ? d < want.dist * 0.94 && d > 36 :
        want.kind === "aoe" || want.kind === "spiral" ? d < (want.r1 || 180) * 0.9 :
        want.kind === "buff" || want.kind === "heal" || want.kind === "shield" || want.kind === "weather" ? true :
        d < (want.range || want.len || 420) * 0.96;
      const aimOk =
        want.kind === "buff" || want.kind === "heal" || want.kind === "shield" || want.kind === "weather" ||
        want.kind === "aoe" || want.kind === "spiral" || want.homing || Math.abs(U.angDiff(a.aim, toT)) < 0.72;
      if (inRange && aimOk) a.skill[st.want] = true;
    }
    return a;
  }

  return { act, classifyRole };
})();
