// Keyboard / mouse / gamepad / touch → one action frame per tick:
// { mx, my (move axis), aim (angle|null), skill[4] (edge), dodge, swap, mega, pause, anyKey }.
"use strict";

const Input = (() => {
  const keys = new Set(), edges = new Set();
  const mouse = { x: 0, y: 0, down: [false, false, false], moved: 0, worldX: 0, worldY: 0 };
  const touch = { active: false, stickId: null, sx: 0, sy: 0, dx: 0, dy: 0, buttons: {} };
  let lastGamepadButtons = [];
  let canvas = null;
  const SKILL_KEYS = [["z", "1", "j"], ["x", "2", "k"], ["c", "3", "l"], ["v", "4", ";"]];

  function init(cv) {
    canvas = cv;
    addEventListener("keydown", (e) => {
      const k = e.key.toLowerCase();
      if (!keys.has(k)) edges.add(k);
      keys.add(k);
      if (["arrowup", "arrowdown", "arrowleft", "arrowright", " "].includes(k) && e.target === document.body) e.preventDefault();
    });
    addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
    addEventListener("blur", () => keys.clear());
    cv.addEventListener("mousemove", (e) => {
      const r = cv.getBoundingClientRect();
      mouse.x = e.clientX - r.left;
      mouse.y = e.clientY - r.top;
      mouse.moved = 0.6;
    });
    cv.addEventListener("mousedown", (e) => { mouse.down[e.button] = true; edges.add("mouse" + e.button); e.preventDefault(); });
    addEventListener("mouseup", (e) => { mouse.down[e.button] = false; });
    cv.addEventListener("contextmenu", (e) => e.preventDefault());
    initTouch();
  }

  function initTouch() {
    const el = document.getElementById("touch");
    if (!el) return;
    const coarse = matchMedia("(pointer: coarse)").matches || "ontouchstart" in window;
    if (!coarse) return;
    touch.active = true;
    el.hidden = false;
    document.body.classList.add("touch");
    const stick = document.getElementById("stick"), knob = document.getElementById("stick-knob");
    const R = 46;
    const move = (t) => {
      const r = stick.getBoundingClientRect();
      let dx = t.clientX - (r.left + r.width / 2), dy = t.clientY - (r.top + r.height / 2);
      const d = Math.hypot(dx, dy);
      if (d > R) { dx *= R / d; dy *= R / d; }
      touch.dx = dx / R; touch.dy = dy / R;
      knob.style.transform = `translate(${dx}px,${dy}px)`;
    };
    stick.addEventListener("touchstart", (e) => { touch.stickId = e.changedTouches[0].identifier; move(e.changedTouches[0]); e.preventDefault(); }, { passive: false });
    stick.addEventListener("touchmove", (e) => { for (const t of e.changedTouches) if (t.identifier === touch.stickId) move(t); e.preventDefault(); }, { passive: false });
    const end = (e) => { for (const t of e.changedTouches) if (t.identifier === touch.stickId) { touch.stickId = null; touch.dx = touch.dy = 0; knob.style.transform = ""; } };
    stick.addEventListener("touchend", end); stick.addEventListener("touchcancel", end);
    el.querySelectorAll("[data-key]").forEach((b) => {
      const k = b.dataset.key;
      b.addEventListener("touchstart", (e) => { if (!keys.has(k)) edges.add(k); keys.add(k); b.classList.add("on"); e.preventDefault(); }, { passive: false });
      const up = (e) => { keys.delete(k); b.classList.remove("on"); e.preventDefault(); };
      b.addEventListener("touchend", up); b.addEventListener("touchcancel", up);
    });
  }

  function gamepad() {
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (const p of pads) if (p && p.connected) return p;
    return null;
  }

  // Build the action frame. `toWorld` maps screen px → world; `self` is the controlled fighter (for mouse aim).
  function poll(dt, toWorld, self) {
    const a = { mx: 0, my: 0, aim: null, skill: [false, false, false, false], skillHeld: [false, false, false, false], dodge: false, swap: false, mega: false, pause: false, any: edges.size > 0, mouseAim: false };
    const down = (k) => keys.has(k);
    const edge = (k) => edges.has(k);
    if (down("a") || down("arrowleft")) a.mx -= 1;
    if (down("d") || down("arrowright")) a.mx += 1;
    if (down("w") || down("arrowup")) a.my -= 1;
    if (down("s") || down("arrowdown")) a.my += 1;
    if (touch.active) { a.mx += touch.dx; a.my += touch.dy; }
    SKILL_KEYS.forEach((ks, i) => { a.skill[i] = ks.some(edge); a.skillHeld[i] = ks.some(down); });
    if (edge("mouse0")) a.skill[0] = true;
    if (edge("mouse2")) a.skill[1] = true;
    if (mouse.down[0]) a.skillHeld[0] = true;
    if (mouse.down[2]) a.skillHeld[1] = true;
    a.dodge = edge(" ") || edge("shift");
    a.swap = edge("q") || edge("tab");
    a.mega = edge("m") || edge("g");
    a.pause = edge("escape") || edge("p");
    const gp = gamepad();
    if (gp) {
      const dz = (v) => (Math.abs(v) < 0.18 ? 0 : v);
      a.mx += dz(gp.axes[0] || 0); a.my += dz(gp.axes[1] || 0);
      const rx = dz(gp.axes[2] || 0), ry = dz(gp.axes[3] || 0);
      if (rx || ry) a.aim = Math.atan2(ry, rx);
      const bt = gp.buttons.map((b) => b.pressed);
      const press = (i) => bt[i] && !lastGamepadButtons[i];
      const map = [2, 0, 3, 1]; // X A Y B → skills 1-4
      map.forEach((bi, i) => { if (press(bi)) a.skill[i] = true; if (bt[bi]) a.skillHeld[i] = true; });
      if (press(5) || press(7)) a.dodge = true;
      if (press(4) || press(6)) a.swap = true;
      if (press(3) && press(2)) a.mega = true;
      if (press(9)) a.pause = true;
      if (press(10)) a.mega = true;
      if (bt.some((b, i) => b && !lastGamepadButtons[i])) a.any = true;
      lastGamepadButtons = bt;
    }
    const l = Math.hypot(a.mx, a.my);
    if (l > 1) { a.mx /= l; a.my /= l; }
    mouse.moved = Math.max(0, mouse.moved - dt);
    if (mouse.moved > 0 && toWorld && self && !touch.active) {
      const w = toWorld(mouse.x, mouse.y);
      mouse.worldX = w.x; mouse.worldY = w.y;
      a.aim = Math.atan2(w.y - self.y, w.x - self.x);
      a.mouseAim = true;
    }
    edges.clear();
    return a;
  }

  const isDown = (k) => keys.has(k);
  return { init, poll, isDown, mouse, touch: () => touch.active };
})();
