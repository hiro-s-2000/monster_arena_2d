// Procedural audio: no sample files. SFX are short synth bursts; the crowd bed and a chip-tune loop are
// generated from oscillators and filtered noise. Everything hangs off one master gain with a mute toggle.
"use strict";

const Audio = (() => {
  let ac = null, master = null, sfxBus = null, musicBus = null, crowdBus = null;
  let muted = false, started = false, musicOn = true;
  let crowd = null, hype = 0, hypeTarget = 0;
  let noiseBuf = null;

  function ensure() {
    if (ac) return true;
    try {
      ac = new (window.AudioContext || window.webkitAudioContext)();
      master = ac.createGain(); master.gain.value = muted ? 0 : 0.8; master.connect(ac.destination);
      sfxBus = ac.createGain(); sfxBus.gain.value = 0.9; sfxBus.connect(master);
      musicBus = ac.createGain(); musicBus.gain.value = musicOn ? 0.32 : 0; musicBus.connect(master);
      crowdBus = ac.createGain(); crowdBus.gain.value = 0.35; crowdBus.connect(master);
      noiseBuf = ac.createBuffer(1, ac.sampleRate * 2, ac.sampleRate);
      const d = noiseBuf.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
      return true;
    } catch (e) { return false; }
  }
  function unlock() { if (ensure() && ac.state === "suspended") ac.resume(); started = true; }

  const now = () => ac.currentTime;
  function env(g, t, a, d, s = 0, r = 0.05, peak = 1) {
    g.gain.cancelScheduledValues(t);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(peak, t + a);
    g.gain.exponentialRampToValueAtTime(Math.max(0.0001, peak * s), t + a + d);
    g.gain.exponentialRampToValueAtTime(0.0001, t + a + d + r);
  }
  function tone({ type = "sine", f = 440, f2, dur = 0.2, a = 0.005, vol = 0.3, bus = sfxBus, detune = 0, pan = 0, curve }) {
    if (!ac) return;
    const o = ac.createOscillator(), g = ac.createGain(), p = ac.createStereoPanner();
    o.type = type; o.frequency.setValueAtTime(f, now()); o.detune.value = detune;
    if (f2) o.frequency.exponentialRampToValueAtTime(Math.max(20, f2), now() + dur * (curve || 1));
    env(g, now(), a, dur * 0.6, 0.2, dur * 0.4, vol);
    p.pan.value = pan;
    o.connect(g).connect(p).connect(bus);
    o.start(); o.stop(now() + dur + 0.1);
  }
  function noise({ dur = 0.2, f = 1000, q = 1, type = "bandpass", vol = 0.3, a = 0.005, f2, bus = sfxBus, pan = 0 }) {
    if (!ac) return;
    const s = ac.createBufferSource(); s.buffer = noiseBuf; s.loop = true;
    const fl = ac.createBiquadFilter(); fl.type = type; fl.frequency.setValueAtTime(f, now()); fl.Q.value = q;
    if (f2) fl.frequency.exponentialRampToValueAtTime(Math.max(40, f2), now() + dur);
    const g = ac.createGain(), p = ac.createStereoPanner(); p.pan.value = pan;
    env(g, now(), a, dur * 0.5, 0.3, dur * 0.5, vol);
    s.connect(fl).connect(g).connect(p).connect(bus);
    s.start(); s.stop(now() + dur + 0.1);
  }

  const SFX = {
    ui: () => tone({ type: "square", f: 880, f2: 1320, dur: 0.07, vol: 0.08 }),
    select: () => { tone({ type: "triangle", f: 660, dur: 0.08, vol: 0.12 }); tone({ type: "triangle", f: 990, dur: 0.12, vol: 0.1, a: 0.06 }); },
    back: () => tone({ type: "square", f: 440, f2: 220, dur: 0.1, vol: 0.08 }),
    count: () => tone({ type: "square", f: 740, dur: 0.12, vol: 0.14 }),
    go: () => { tone({ type: "square", f: 988, dur: 0.35, vol: 0.18 }); tone({ type: "square", f: 1480, dur: 0.35, vol: 0.1, detune: 6 }); },
    swing: (p) => noise({ dur: 0.12, f: 1800, f2: 500, q: 2, vol: 0.18, pan: p }),
    hit: (p) => { noise({ dur: 0.12, f: 900, f2: 200, q: 1, vol: 0.4, pan: p }); tone({ type: "square", f: 180, f2: 60, dur: 0.1, vol: 0.25, pan: p }); },
    hitHeavy: (p) => { noise({ dur: 0.25, f: 600, f2: 90, q: 1, vol: 0.55, pan: p }); tone({ type: "sawtooth", f: 120, f2: 35, dur: 0.22, vol: 0.4, pan: p }); },
    superEff: (p) => { tone({ type: "square", f: 520, f2: 1040, dur: 0.12, vol: 0.2, pan: p }); tone({ type: "square", f: 780, f2: 1560, dur: 0.14, vol: 0.15, a: 0.05, pan: p }); },
    weak: (p) => tone({ type: "triangle", f: 300, f2: 200, dur: 0.12, vol: 0.15, pan: p }),
    fire: (p) => { noise({ dur: 0.5, f: 1200, f2: 300, q: 0.8, vol: 0.3, type: "lowpass", pan: p }); noise({ dur: 0.3, f: 3000, q: 0.5, vol: 0.1, pan: p }); },
    water: (p) => { noise({ dur: 0.45, f: 2400, f2: 900, q: 1.5, vol: 0.28, pan: p }); tone({ type: "sine", f: 300, f2: 900, dur: 0.25, vol: 0.12, pan: p }); },
    electric: (p) => { noise({ dur: 0.3, f: 4000, q: 6, vol: 0.3, pan: p }); tone({ type: "sawtooth", f: 1800, f2: 900, dur: 0.18, vol: 0.12, pan: p }); },
    ice: (p) => { tone({ type: "sine", f: 1800, f2: 2600, dur: 0.3, vol: 0.14, pan: p }); noise({ dur: 0.3, f: 6000, q: 3, vol: 0.12, pan: p }); },
    grass: (p) => { noise({ dur: 0.3, f: 2600, f2: 1400, q: 2, vol: 0.18, pan: p }); tone({ type: "triangle", f: 520, f2: 780, dur: 0.2, vol: 0.1, pan: p }); },
    psychic: (p) => { tone({ type: "sine", f: 400, f2: 1600, dur: 0.5, vol: 0.18, pan: p }); tone({ type: "sine", f: 403, f2: 1610, dur: 0.5, vol: 0.12, pan: p }); },
    ghost: (p) => { tone({ type: "sawtooth", f: 220, f2: 90, dur: 0.6, vol: 0.14, pan: p }); noise({ dur: 0.5, f: 500, q: 4, vol: 0.12, type: "lowpass", pan: p }); },
    dark: (p) => { tone({ type: "square", f: 110, f2: 55, dur: 0.4, vol: 0.2, pan: p }); noise({ dur: 0.3, f: 300, q: 1, vol: 0.2, type: "lowpass", pan: p }); },
    rock: (p) => { noise({ dur: 0.35, f: 400, f2: 80, q: 1, vol: 0.5, type: "lowpass", pan: p }); tone({ type: "triangle", f: 90, f2: 40, dur: 0.3, vol: 0.3, pan: p }); },
    fighting: (p) => { noise({ dur: 0.1, f: 1200, f2: 300, q: 2, vol: 0.35, pan: p }); tone({ type: "square", f: 240, f2: 90, dur: 0.1, vol: 0.2, pan: p }); },
    flying: (p) => noise({ dur: 0.4, f: 900, f2: 2600, q: 1.5, vol: 0.2, pan: p }),
    dragon: (p) => { tone({ type: "sawtooth", f: 160, f2: 420, dur: 0.5, vol: 0.2, pan: p }); noise({ dur: 0.5, f: 1500, f2: 400, q: 1, vol: 0.2, pan: p }); },
    fairy: (p) => { tone({ type: "sine", f: 1200, f2: 2400, dur: 0.3, vol: 0.12, pan: p }); tone({ type: "sine", f: 1800, f2: 3000, dur: 0.3, vol: 0.08, a: 0.08, pan: p }); },
    steel: (p) => { tone({ type: "square", f: 2200, f2: 1100, dur: 0.18, vol: 0.12, pan: p }); noise({ dur: 0.2, f: 5000, q: 8, vol: 0.2, pan: p }); },
    poison: (p) => { noise({ dur: 0.4, f: 700, f2: 300, q: 3, vol: 0.2, pan: p }); tone({ type: "sine", f: 240, f2: 160, dur: 0.3, vol: 0.12, pan: p }); },
    bug: (p) => tone({ type: "sawtooth", f: 900, f2: 1300, dur: 0.25, vol: 0.1, pan: p }),
    ground: (p) => { noise({ dur: 0.6, f: 250, f2: 60, q: 1, vol: 0.55, type: "lowpass", pan: p }); tone({ type: "sine", f: 60, f2: 30, dur: 0.5, vol: 0.4, pan: p }); },
    normal: (p) => noise({ dur: 0.15, f: 1500, f2: 400, q: 1.5, vol: 0.25, pan: p }),
    buff: (p) => { tone({ type: "triangle", f: 440, f2: 880, dur: 0.3, vol: 0.14, pan: p }); tone({ type: "triangle", f: 660, f2: 1320, dur: 0.3, vol: 0.1, a: 0.1, pan: p }); },
    heal: (p) => { [523, 659, 784, 1047].forEach((f, i) => setTimeout(() => tone({ type: "sine", f, dur: 0.25, vol: 0.12, pan: p }), i * 70)); },
    shield: (p) => tone({ type: "sine", f: 300, f2: 1200, dur: 0.25, vol: 0.14, pan: p }),
    dodge: (p) => noise({ dur: 0.18, f: 600, f2: 3000, q: 2, vol: 0.14, pan: p }),
    ball: (p) => { tone({ type: "sine", f: 900, f2: 300, dur: 0.25, vol: 0.14, pan: p }); noise({ dur: 0.3, f: 3000, f2: 800, q: 1, vol: 0.16, pan: p }); },
    ballRecall: (p) => { tone({ type: "triangle", f: 1250, f2: 420, dur: 0.28, a: 0.01, vol: 0.16, pan: p }); tone({ type: "sine", f: 1850, f2: 680, dur: 0.22, vol: 0.09, pan: p }); noise({ dur: 0.24, f: 2400, f2: 850, q: 2, vol: 0.12, pan: p }); },
    ballThrow: (p) => { noise({ dur: 0.22, f: 680, f2: 2600, q: 2.2, vol: 0.16, pan: p }); tone({ type: "sine", f: 420, f2: 980, dur: 0.16, a: 0.005, vol: 0.11, pan: p }); },
    ballOpen: (p) => { noise({ dur: 0.24, f: 3600, f2: 900, q: 1.2, vol: 0.2, pan: p }); tone({ type: "triangle", f: 520, f2: 1480, dur: 0.18, a: 0.005, vol: 0.18, pan: p }); tone({ type: "sine", f: 1040, f2: 1760, dur: 0.26, a: 0.01, vol: 0.12, pan: p }); },
    ko: () => { tone({ type: "sawtooth", f: 220, f2: 40, dur: 0.9, vol: 0.35 }); noise({ dur: 0.8, f: 1200, f2: 100, q: 0.7, vol: 0.4 }); },
    win: () => SFX.fanfare(),
    lose: () => { [392, 349, 311, 262].forEach((f, i) => setTimeout(() => tone({ type: "square", f, dur: 0.4, vol: 0.14 }), i * 180)); },
    mega: () => { tone({ type: "sawtooth", f: 110, f2: 880, dur: 1.2, vol: 0.2 }); tone({ type: "sine", f: 440, f2: 1760, dur: 1.2, vol: 0.12 }); noise({ dur: 1, f: 800, f2: 4000, q: 1, vol: 0.12 }); },
    status: (p) => tone({ type: "square", f: 300, f2: 150, dur: 0.25, vol: 0.1, pan: p }),
    crowdHit: () => { hypeTarget = Math.min(1, hypeTarget + 0.5); noise({ dur: 1.2, f: 1100, q: 0.4, vol: 0.25, a: 0.15, bus: crowdBus || sfxBus }); },
    clunk: () => { tone({ type: "square", f: 70, f2: 40, dur: 0.18, vol: 0.35 }); noise({ dur: 0.12, f: 300, q: 1, vol: 0.3, type: "lowpass" }); tone({ type: "sine", f: 120, dur: 0.6, vol: 0.06 }); },
    thud: (p) => { tone({ type: "sine", f: 70, f2: 32, dur: 0.22, vol: 0.32, pan: p }); noise({ dur: 0.14, f: 220, q: 1, vol: 0.22, type: "lowpass", pan: p }); },
    step: (p) => { tone({ type: "sine", f: 55, f2: 30, dur: 0.12, vol: 0.18, pan: p }); noise({ dur: 0.08, f: 180, q: 1, vol: 0.1, type: "lowpass", pan: p }); },
    ability: () => { tone({ type: "square", f: 660, f2: 990, dur: 0.09, vol: 0.09 }); tone({ type: "square", f: 990, dur: 0.14, vol: 0.08, a: 0.08 }); },
    charge: (p) => { tone({ type: "sawtooth", f: 90, f2: 520, dur: 0.55, vol: 0.12, pan: p }); tone({ type: "square", f: 180, f2: 1040, dur: 0.55, vol: 0.06, pan: p }); },
    psybeam: (p) => { noise({ dur: 0.7, f: 2400, f2: 500, q: 0.7, vol: 0.4, pan: p }); tone({ type: "sawtooth", f: 140, f2: 45, dur: 0.7, vol: 0.35, pan: p }); tone({ type: "square", f: 880, f2: 220, dur: 0.4, vol: 0.1, pan: p }); },
    grab: (p) => { tone({ type: "sine", f: 1500, f2: 500, dur: 0.18, vol: 0.14, pan: p }); tone({ type: "triangle", f: 300, f2: 600, dur: 0.3, vol: 0.12, a: 0.05, pan: p }); },
    rumble: () => { noise({ dur: 1.6, f: 160, f2: 60, q: 0.8, vol: 0.45, type: "lowpass", a: 0.3 }); tone({ type: "sine", f: 48, f2: 36, dur: 1.6, vol: 0.3, a: 0.3 }); },
    cursor: () => tone({ type: "square", f: 1320, dur: 0.04, vol: 0.06 }),
    text: () => tone({ type: "square", f: 660 + Math.random() * 40, dur: 0.025, vol: 0.035 }),
    fanfare: () => { const N = [[523, 0], [523, 0.12], [523, 0.24], [659, 0.36], [784, 0.6], [659, 0.84], [784, 0.96], [1047, 1.2]]; N.forEach(([f, d]) => setTimeout(() => { tone({ type: "square", f, dur: 0.22, vol: 0.13 }); tone({ type: "triangle", f: f / 2, dur: 0.22, vol: 0.1 }); }, d * 1000)); },
    levelup: () => { [784, 988, 1175, 1568].forEach((f, i) => setTimeout(() => tone({ type: "square", f, dur: 0.12, vol: 0.12 }), i * 70)); },
  };

  function sfx(name, pan = 0) {
    if (!ac || muted) return;
    const f = SFX[name];
    if (f) f(U.clamp(pan, -0.8, 0.8));
  }

  // Crowd bed: two noise layers through low-pass filters whose gain follows the hype level.
  function startCrowd() {
    if (!ac || crowd) return;
    const mk = (f, vol) => {
      const s = ac.createBufferSource(); s.buffer = noiseBuf; s.loop = true;
      const fl = ac.createBiquadFilter(); fl.type = "lowpass"; fl.frequency.value = f; fl.Q.value = 0.5;
      const g = ac.createGain(); g.gain.value = vol;
      s.connect(fl).connect(g).connect(crowdBus); s.start();
      return { s, fl, g };
    };
    crowd = { base: mk(450, 0.12), roar: mk(1400, 0) };
  }
  function stopCrowd() { if (crowd) { for (const k of ["base", "roar"]) { try { crowd[k].s.stop(); } catch (e) {} } crowd = null; } }

  // Chip-tune loop: a bass line, an arpeggio and a hat pattern scheduled a bar ahead.
  let music = null;
  const SCALE = [0, 3, 5, 7, 10, 12, 15];
  const CHORDS = [[0, 3, 7], [-2, 2, 5], [-4, 0, 3], [-5, -2, 2]];
  function startMusic(bpm = 150, root = 110) {
    if (!ac || music) return;
    const stepDur = 60 / bpm / 2;
    music = { next: now() + 0.1, step: 0, bpm, root, timer: null, stepDur };
    const schedule = () => {
      if (!music) return;
      while (music.next < now() + 0.4) {
        const t = music.next, st = music.step;
        const bar = Math.floor(st / 16) % 4, chord = CHORDS[bar];
        // bass on 8ths
        if (st % 2 === 0) {
          const o = ac.createOscillator(), g = ac.createGain();
          o.type = "square"; o.frequency.value = root * Math.pow(2, chord[0] / 12) / 2;
          g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.18, t + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, t + stepDur * 1.6);
          o.connect(g).connect(musicBus); o.start(t); o.stop(t + stepDur * 2);
        }
        // arpeggio on 16ths
        {
          const n = chord[st % 3] + 12 + (st % 8 >= 4 ? 12 : 0);
          const o = ac.createOscillator(), g = ac.createGain();
          o.type = st % 4 === 0 ? "sawtooth" : "square"; o.frequency.value = root * Math.pow(2, n / 12);
          g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.07, t + 0.005); g.gain.exponentialRampToValueAtTime(0.0001, t + stepDur * 0.9);
          o.connect(g).connect(musicBus); o.start(t); o.stop(t + stepDur);
        }
        // kick / hat
        {
          const s = ac.createBufferSource(); s.buffer = noiseBuf;
          const fl = ac.createBiquadFilter(); fl.type = "highpass"; fl.frequency.value = 7000;
          const g = ac.createGain(); g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(st % 2 ? 0.05 : 0.09, t + 0.002); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.04);
          s.connect(fl).connect(g).connect(musicBus); s.start(t); s.stop(t + 0.06);
          if (st % 4 === 0 || st % 16 === 10) {
            const o = ac.createOscillator(), kg = ac.createGain();
            o.type = "sine"; o.frequency.setValueAtTime(150, t); o.frequency.exponentialRampToValueAtTime(40, t + 0.12);
            kg.gain.setValueAtTime(0.0001, t); kg.gain.exponentialRampToValueAtTime(0.5, t + 0.005); kg.gain.exponentialRampToValueAtTime(0.0001, t + 0.15);
            o.connect(kg).connect(musicBus); o.start(t); o.stop(t + 0.2);
          }
        }
        music.next += stepDur;
        music.step++;
      }
      music.timer = setTimeout(schedule, 120);
    };
    schedule();
  }
  function stopMusic() { if (music) { clearTimeout(music.timer); music = null; } }

  function update(dt) {
    if (!ac || !crowd) return;
    hypeTarget = Math.max(0, hypeTarget - dt * 0.25);
    hype = U.damp(hype, hypeTarget, 3, dt);
    crowd.roar.g.gain.value = hype * 0.3;
    crowd.base.fl.frequency.value = 450 + hype * 600;
  }
  function setHype(v) { hypeTarget = Math.max(hypeTarget, v); }
  function setMuted(m) { muted = m; if (master) master.gain.value = m ? 0 : 0.8; }
  function setMusic(on) { musicOn = on; if (musicBus) musicBus.gain.value = on ? 0.32 : 0; }

  return { unlock, sfx, update, startCrowd, stopCrowd, startMusic, stopMusic, setHype, setMuted, setMusic, isMuted: () => muted, musicOn: () => musicOn };
})();
