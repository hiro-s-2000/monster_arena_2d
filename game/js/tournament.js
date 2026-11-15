// 8-entry single-elimination tournament. Every match is a 1v1 arena battle on a random (or chosen) stage,
// ending with the "ROUND ONE WINNER — NAME" banner from the vote videos. The bracket screen sits between matches.
"use strict";

const Tournament = (() => {
  const $ = (s, r = document) => r.querySelector(s);
  const ROUND = ["ROUND ONE", "ROUND TWO", "FINAL ROUND"];
  let E = [], alive = [], round = 0, match = 0, results = [], cfg0 = null, play = null, done = null, autoT = null;

  function start(cfg, runMatch, onExit) {
    cfg0 = cfg; play = runMatch; done = onExit;
    E = cfg.teams[0].slice(0, 8);
    alive = [E.map((_, i) => i)]; round = 0; match = 0; results = [[], [], []];
    show();
  }

  function portrait(e) { return `<img src="${Assets.stillUrl(e.species)}" alt="">`; }
  function show(champion) {
    const el = $("#bracket");
    el.classList.remove("hidden");
    const col = (r) => {
      const ids = alive[r] || [], n = r === 0 ? 8 : r === 1 ? 4 : 2, out = [];
      for (let m = 0; m < n / 2; m++) {
        const a = ids[m * 2], b = ids[m * 2 + 1], w = results[r][m];
        const cell = (i) => i === undefined ? `<div class="ent tbd">？？？</div>` : `<div class="ent ${w === undefined ? "" : w === i ? "win" : "lose"}${cfg0.pilot === i ? " you" : ""}">${portrait(E[i])}<span>${E[i].species.jp}</span></div>`;
        const next = r === round && m === match && !champion;
        out.push(`<div class="match${next ? " next" : ""}">${cell(a)}${cell(b)}</div>`);
      }
      return `<div class="col"><h4>${ROUND[r]}</h4>${out.join("")}</div>`;
    };
    const champ = champion !== undefined ? E[champion] : null;
    el.innerHTML = `<div class="dqwin wide"><h2>トーナメント</h2><div class="tree">${col(0)}${col(1)}${col(2)}<div class="col champ"><h4>CHAMPION</h4>${champ ? `<div class="ent win big">${portrait(champ)}<span>${champ.species.jp}</span></div>` : `<div class="ent tbd big">？</div>`}</div></div>
      <div class="btns">${champ ? `<button class="btn-main" data-t="menu">タイトルへ</button>` : `<button class="btn-main" data-t="next">${ROUND[round]} だい${match + 1}しあい！</button><button class="btn-ghost" data-t="menu">やめる</button>`}</div></div>`;
    el.querySelectorAll("[data-t]").forEach((b) => b.addEventListener("click", () => { clearTimeout(autoT); Audio.sfx("select"); if (b.dataset.t === "next") next(); else { el.classList.add("hidden"); done(); } }));
    if (champ) { Audio.sfx("fanfare"); confetti(el); }
    else if (cfg0.pilot < 0) autoT = setTimeout(() => { if (!el.classList.contains("hidden")) next(); }, 3500);
  }
  function confetti(el) {
    for (let i = 0; i < 80; i++) { const c = document.createElement("i"); c.className = "confetti"; c.style.left = Math.random() * 100 + "%"; c.style.background = U.pick(["#ffd54a", "#ff5a7a", "#5ad8ff", "#8cff6a", "#ffffff"]); c.style.animationDelay = Math.random() * 1.5 + "s"; c.style.animationDuration = 2 + Math.random() * 2 + "s"; el.appendChild(c); }
  }

  function next() {
    $("#bracket").classList.add("hidden");
    const ids = alive[round], a = ids[match * 2], b = ids[match * 2 + 1];
    const swap = cfg0.pilot === b;
    const A = swap ? b : a, Bi = swap ? a : b;
    const pilotIn = cfg0.pilot === a || cfg0.pilot === b;
    const arena = cfg0.arenaId === "random" ? U.pick(Arenas.list()).id : cfg0.arenaId;
    play({ teams: [[E[A]], [E[Bi]]], arenaId: arena, mode: pilotIn ? "play" : "watch", skill: cfg0.skill, duration: Math.min(cfg0.duration || 180, 180), label: `TOURNAMENT · ${ROUND[round]}`, roundName: ROUND[round], trainers: cfg0.trainers }, (winnerSide) => {
      const w = winnerSide === 1 ? Bi : winnerSide === 0 ? A : (Math.random() < 0.5 ? A : Bi);
      results[round][match] = w;
      (alive[round + 1] = alive[round + 1] || [])[match] = w;
      match++;
      if (match * 2 >= ids.length) { round++; match = 0; }
      if (round >= 3) show(w); else show();
    });
  }
  return { start };
})();
