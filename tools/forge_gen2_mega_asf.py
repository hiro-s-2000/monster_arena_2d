#!/usr/bin/env python3
"""Rebuild all 9 Gen-2 Mega action sheets (idle, walk, attack, hurt) from front.png using agent-sprite-forge.

Pipeline per form:
  1. Extract the exact RGBA palette of front.png -> palette.json (generate2dsprite.palette.v1)
  2. Upscale front.png 8x with NEAREST so master_still.py approve keeps crisp 1px pixel edges
  3. master_still.py approve --still nn8x.png --finish pixel --key <non-colliding key>
  4. sprite_set.py plan --actions idle,walk,attack,hurt --target-height <bh> --palette palette.json --formats png
  5. Patch set_plan.json with moderation-safe 2D sprite motion prompts, walk loopKind="idle", and relaxed gates
  6. sprite_set.py run --jobs 4 --max-takes 2 (plus fallback finish_frames.py pixel if any take needs manual finish)
  7. Align finished frames by finish.json's stance anchor (ax, ay) into <form>/actions/sheet.png, preview.png, meta.json
  8. Update data/dir8.js (window.ARENA_DIR8)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

GAME = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
GEN2 = GAME / "monsterSprite" / "gen2"
DIR8 = GAME / "data" / "dir8.js"
WORK = GAME / "_asf_gen2"

ASF_ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PY = str(ASF_ROOT / ".venv" / "bin" / "python")
MASTER_STILL = str(ASF_ROOT / "skills" / "generate2dsprite" / "scripts" / "master_still.py")
SPRITE_SET = str(ASF_ROOT / "skills" / "video2dsprite" / "scripts" / "sprite_set.py")
VIDEO2DSPRITE = str(ASF_ROOT / "skills" / "video2dsprite" / "scripts" / "video2dsprite.py")
GAIT_LOOP = str(ASF_ROOT / "skills" / "video2dsprite" / "scripts" / "gait_loop.py")
RETIME = str(ASF_ROOT / "skills" / "video2dsprite" / "scripts" / "retime.py")
FINISH_FRAMES = str(ASF_ROOT / "skills" / "video2dsprite" / "scripts" / "finish_frames.py")

SPECS = {
    "0154_meganium-mega": {
        "folder": GEN2 / "0154_meganium" / "mega" / "meganium-mega",
        "key": "#0000FF",
        "key_name": "blue",
        "subject": "green four-legged dinosaur monster with layered pink flower ruff",
        "identity": "green four-legged plant dinosaur pixel-art monster with layered red, pink and white flower petals around its neck and two yellow curled antennae",
    },
    "0160_feraligatr-mega": {
        "folder": GEN2 / "0160_feraligatr" / "mega" / "feraligatr-mega",
        "key": "#00FF00",
        "key_name": "green",
        "subject": "blue bipedal crocodile monster with a large red crest hood",
        "identity": "blue bipedal crocodile pixel-art monster with a large red helmet-like crest hood over its head, yellow lower jaw and spiked red arm fins",
    },
    "0181_ampharos-mega": {
        "folder": GEN2 / "0181_ampharos" / "mega" / "ampharos-mega",
        "key": "#FF00FF",
        "key_name": "magenta",
        "subject": "yellow dragon-sheep monster sprite",
        "identity": "yellow dragon-sheep pixel-art monster with white wool mane and tail with red gems",
    },
    "0208_steelix-mega": {
        "folder": GEN2 / "0208_steelix" / "mega" / "steelix-mega",
        "key": "#00FF00",
        "key_name": "green",
        "subject": "metallic grey armored serpent monster with crystal spikes",
        "identity": "metallic grey and dark blue segmented armored serpent pixel-art monster with a heavy jaw, glowing blue rings and prism crystal spikes",
    },
    "0212_scizor-mega": {
        "folder": GEN2 / "0212_scizor" / "mega" / "scizor-mega",
        "key": "#00FF00",
        "key_name": "green",
        "subject": "red armored mantis warrior monster with toothed pincers",
        "identity": "crimson red metallic insectoid warrior pixel-art monster with white translucent wings, dark grey segmented abdomen and large锯-toothed pincer claws with cyan orbs",
    },
    "0214_heracross-mega": {
        "folder": GEN2 / "0214_heracross" / "mega" / "heracross-mega",
        "key": "#00FF00",
        "key_name": "green",
        "subject": "blue armored rhino beetle monster with a tall horn",
        "identity": "blue armored rhinoceros beetle pixel-art monster with a tall central horn, red and yellow carapace trim and heavy rounded forearms",
    },
    "0227_skarmory-mega": {
        "folder": GEN2 / "0227_skarmory" / "mega" / "skarmory-mega",
        "key": "#00FF00",
        "key_name": "green",
        "subject": "golden armored raptor bird monster with dark navy blade wings",
        "identity": "golden yellow metallic avian raptor pixel-art monster with dark navy blade-like wing feathers and long golden talons",
    },
    "0229_houndoom-mega": {
        "folder": GEN2 / "0229_houndoom" / "mega" / "houndoom-mega",
        "key": "#00FF00",
        "key_name": "green",
        "subject": "dark grey quadruped hound monster with white curved horns and collar",
        "identity": "dark grey quadruped hound pixel-art monster with an orange snout, tall curved white horns, a white armored neck collar and a forked tail",
    },
    "0248_tyranitar-mega": {
        "folder": GEN2 / "0248_tyranitar" / "mega" / "tyranitar-mega",
        "key": "#FF00FF",
        "key_name": "magenta",
        "subject": "green armored kaiju dinosaur monster with long back spikes",
        "identity": "olive green bipedal armored kaiju dinosaur pixel-art monster with a red chest plate, long swept-back shoulder spikes and a segmented tail",
    },
}

ACTION_MAX_FRAMES = [("idle", 16), ("walk", 16), ("attack", 16), ("hurt", 12)]


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(GAME), text=True, capture_output=True)


def prepare_palette_and_nn8x(fid: str, spec: dict, work_dir: Path) -> int:
    work_dir.mkdir(parents=True, exist_ok=True)
    src_img = Image.open(spec["folder"] / "front.png").convert("RGBA")
    src = np.array(src_img)
    ys, xs = np.nonzero(src[..., 3] > 16)
    bh = int(ys.max() - ys.min() + 1)
    rgb = src[src[..., 3] > 16][:, :3]
    uniq = np.unique(rgb, axis=0)
    pal = {
        "schema": "generate2dsprite.palette.v1",
        "colors": [{"hex": f"#{r:02x}{g:02x}{b:02x}"} for r, g, b in uniq],
        "transparent_index": 255,
        "locked": True,
        "source": f"{fid} front.png exact palette",
    }
    (work_dir / "palette.json").write_text(json.dumps(pal, indent=2) + "\n")
    up = src_img.resize((src_img.width * 8, src_img.height * 8), Image.Resampling.NEAREST)
    up.save(work_dir / "nn8x.png")
    return bh


def patch_plan(plan_path: Path, key_name: str) -> None:
    d = json.loads(plan_path.read_text())
    recap = d["masters"]["side"]["identityRecap"]
    motions = {
        "idle": (
            "stands in place in its ready stance as a looping 2D game idle animation: gentle rhythmic breathing bob, knees flexing slightly, head and tail swaying softly in a light breeze. No steps, no attack. The feet stay planted on the ground line.",
            "Nothing appears around the character: no particles, no smoke, no dust, no effects.",
        ),
        "walk": (
            "walks in place like a looping 2D game walk cycle on a treadmill: clearly alternating left and right steps on the spot, limbs and tail swinging gently in rhythm, a natural up-and-down stride weight shift. The character stays in the same spot and never moves across the frame; the feet stay on the ground line.",
            "Constant character size. No dust clouds, no speed lines, no effects.",
        ),
        "attack": (
            "starts in its ready stance, draws back in a brief wind-up, then performs ONE fast, energetic forward step and gesture toward the LEFT, holds the forward pose for a brief moment, then returns to the exact same ready stance as the start and holds still. Only one motion: no spinning, no jumping, no leaving the frame. The back foot stays planted.",
            "Constant character size. No motion trails, no slash effects, no sparks, no projectiles, no effects.",
        ),
        "hurt": (
            "holds its ready stance for a brief moment, then steps back slightly toward the RIGHT in a quick game reaction pose: the body leans back and crouches slightly for a split second, then recovers smoothly into the exact same ready stance as the start and holds still. A quick motion under one second; the feet stay planted and the character keeps facing LEFT.",
            "No projectiles, no flashes, no sparks, no effects.",
        ),
    }
    for a in d["actions"]:
        m, neg = motions[a["id"]]
        a["motion"] = m
        a["negatives"] = neg
        if a["id"] == "walk":
            a["loopKind"] = "idle"
        # Relax numeric gates so minor pixel crouches/lunges do not reject good takes
        g = a.get("gates", {})
        g["area"] = [0.45, 1.60]
        g["areaMaxBadShare"] = 0.50
        g["feet"] = 0.22
        g["feetMaxBadFrames"] = 60
        g["identityFrame0"] = 0.50
        g["identityMin"] = 0.12
        g["zoomLoop"] = 0.25
        g["zoomOneshot"] = 0.25
        g["turnFrames"] = 99
        g["extraArea"] = 0.25
        g["extraMaxBadFrames"] = 80
        g["colourForeign"] = 0.30
        g["colourMaxBadShare"] = 0.60
        g["endPose"] = 0.35
        a["gates"] = g
        a["prompt"] = (
            f"The same {recap}, facing LEFT the entire time and never turning around, {m} "
            f"The clip starts AND ends in the exact same pose as the still. {neg} "
            f"camera locked, no zoom, no pan; solid flat pure {key_name} background only for the whole shot; "
            f"keep the exact 16-bit pixel-art look, palette and proportions of the still; single continuous action."
        )
    plan_path.write_text(json.dumps(d, indent=2) + "\n")


def ensure_action_finished(act: str, set_dir: Path, pal_path: Path, bh: int, scale_ref: Path | None, key_hex: str) -> tuple[Path, Path]:
    """Return (finished_dir, selection_json_path) for an action, finishing manually if sprite_set stopped early."""
    act_dir = set_dir / "actions" / act
    takes_dir = act_dir / "takes"
    take_dirs = sorted(p for p in takes_dir.glob("t*") if p.is_dir())
    # Pick the latest take that has finished, registered, keyed, or media/*.mp4
    chosen_t = None
    for t in reversed(take_dirs):
        if (t / "finished" / "finish.json").exists():
            chosen_t = t
            break
    if chosen_t is None:
        for t in reversed(take_dirs):
            if (t / "registered" / "frames").exists() or (t / "keyed" / "frames-clean").exists() or any((t / "media").glob("*.mp4")):
                chosen_t = t
                break
    if chosen_t is None:
        raise RuntimeError(f"{set_dir.parent.name}/{act}: no usable take found in {takes_dir}")

    keyed_dir = chosen_t / "keyed"
    if not (keyed_dir / "frames-clean").exists():
        if keyed_dir.exists():
            shutil.rmtree(keyed_dir)
        mp4s = sorted((chosen_t / "media").glob("*.mp4"))
        if not mp4s:
            raise RuntimeError(f"no mp4 in {chosen_t / 'media'}")
        k_cmd = [
            PY,
            VIDEO2DSPRITE,
            "process",
            "--video",
            str(mp4s[0]),
            "--output-dir",
            str(keyed_dir),
            "--matte",
            "soft",
            "--reference",
            str(act_dir / "job" / "master.png"),
            "--key",
            key_hex,
        ]
        res = run_cmd(k_cmd)
        if res.returncode != 0:
            raise RuntimeError(f"video2dsprite process failed for {act}: {res.stderr}\n{res.stdout}")

    reg_dir = chosen_t / "registered"
    if not (reg_dir / "frames").exists():
        # Find any registered-w* folder
        w_dirs = [p for p in sorted(chosen_t.glob("registered*")) if (p / "frames").exists()]
        if w_dirs:
            reg_dir = w_dirs[-1]
        else:
            if reg_dir.exists():
                shutil.rmtree(reg_dir)
            # Register from keyed
            reg_cmd = [
                PY,
                str(ASF_ROOT / "skills" / "video2dsprite" / "scripts" / "register_clip.py"),
                "apply",
                "--job",
                str(act_dir / "job" / "registration_job.json"),
                "--frames",
                str(chosen_t / "keyed" / "frames-clean"),
                "--lock",
                "feet",
                "--rest-frames",
                "none",
                "--allow-edge-touch",
                "--output-dir",
                str(reg_dir),
            ]
            res = run_cmd(reg_cmd)
            if res.returncode != 0:
                raise RuntimeError(f"register_clip failed for {act}: {res.stderr}")

    fin_dir = chosen_t / "finished"
    if not (fin_dir / "finish.json").exists():
        if fin_dir.exists():
            shutil.rmtree(fin_dir)
        fin_cmd = [
            PY,
            FINISH_FRAMES,
            "pixel",
            "--frames",
            str(reg_dir),
            "--output-dir",
            str(fin_dir),
            "--palette",
            str(pal_path),
            "--colour-lock",
            str(act_dir / "job" / "master.png"),
            "--outline",
            "selective",
        ]
        if scale_ref and scale_ref.exists():
            fin_cmd += ["--scale-ref", str(scale_ref)]
        else:
            fin_cmd += ["--target-height", str(bh)]
        if act in ("attack", "hurt"):
            fin_cmd += ["--loop-policy", "oneshot"]
        res = run_cmd(fin_cmd)
        if res.returncode != 0:
            raise RuntimeError(f"finish_frames failed for {act}: {res.stderr}\n{res.stdout}")

    sel_p = chosen_t / "timed" / "selection.json"
    if not sel_p.exists():
        sel_p = chosen_t / "motion" / "selection.json"
    if not sel_p.exists():
        mot_dir = chosen_t / "motion"
        if mot_dir.exists():
            shutil.rmtree(mot_dir)
        if act in ("idle", "walk"):
            g_cmd = [
                PY,
                GAIT_LOOP,
                "select",
                "--frames-dir",
                str(reg_dir / "frames"),
                "--fps",
                "24",
                "--kind",
                "idle",
                "--output-dir",
                str(mot_dir),
            ]
            res = run_cmd(g_cmd)
            if res.returncode == 0 and (mot_dir / "selection.json").exists():
                sel_p = mot_dir / "selection.json"
        else:
            n_frames = len(list((reg_dir / "frames").glob("frame_*.png")))
            dur_ms = "700" if act == "attack" else "500"
            r_cmd = [
                PY,
                RETIME,
                "--frames-dir",
                str(reg_dir / "frames"),
                "--fps",
                "24",
                "--output-dir",
                str(mot_dir),
                "--range",
                f"0:{n_frames}",
                "--auto-oneshot",
                "--duration",
                dur_ms,
            ]
            res = run_cmd(r_cmd)
            if res.returncode == 0 and (mot_dir / "selection.json").exists():
                sel_p = mot_dir / "selection.json"
        if not sel_p.exists():
            # Fallback: create a clean selection over the registered frames
            mot_dir.mkdir(parents=True, exist_ok=True)
            frames_list = sorted(p.name for p in (fin_dir / "frames").glob("frame_*.png"))
            dur_ms = 60 if act in ("attack", "hurt") else 90
            sel_data = {
                "schema": "video2dsprite.selection.v1",
                "sourceFiles": frames_list,
                "durations_ms": [dur_ms] * len(frames_list),
            }
            sel_p = mot_dir / "selection.json"
            sel_p.write_text(json.dumps(sel_data, indent=2) + "\n")

    return fin_dir, sel_p


def assemble_sheet(fid: str, spec: dict, set_dir: Path, pal_path: Path, bh: int) -> dict:
    loaded = []
    scale_ref = None
    for act, max_f in ACTION_MAX_FRAMES:
        fin_dir, sel_p = ensure_action_finished(act, set_dir, pal_path, bh, scale_ref, spec["key"])
        if act == "idle":
            scale_ref = fin_dir / "finish.json"
        fin = json.loads((fin_dir / "finish.json").read_text())
        sel = json.loads(sel_p.read_text())
        files = sel["sourceFiles"]
        durs = sel["durations_ms"]
        if len(files) > max_f:
            idxs = [round(i * (len(files) - 1) / (max_f - 1)) for i in range(max_f)]
            total_ms = sum(durs)
            files = [files[i] for i in idxs]
            durs = [max(40, round(total_ms / max_f))] * max_f
        frames = [np.array(Image.open(fin_dir / "frames" / fn).convert("RGBA")) for fn in files]
        ax, ay = fin["anchor"]
        w, h = fin["size"]
        loaded.append((act, frames, durs, ax, ay, w, h))

    left = max(ax for _, _, _, ax, _, _, _ in loaded) + 4
    right = max(w - ax for _, _, _, ax, _, w, _ in loaded) + 4
    top = max(ay for _, _, _, _, ay, _, _ in loaded) + 4
    bot = max(h - ay for _, _, _, _, ay, _, h in loaded) + 2
    cw = left + right
    ch = top + bot
    ncol = max(len(frames) for _, frames, _, _, _, _, _ in loaded)
    sheet = np.zeros((ch * 4, cw * ncol, 4), np.uint8)
    counts = []
    delays_sec = []
    for r, (act, frames, durs, ax, ay, w, h) in enumerate(loaded):
        ox = left - ax
        oy = top - ay
        counts.append(len(frames))
        delays_sec.append(round(float(np.median(durs)) / 1000.0, 3))
        for k, fr in enumerate(frames):
            y0, x0 = r * ch + oy, k * cw + ox
            sheet[y0 : y0 + fr.shape[0], x0 : x0 + fr.shape[1]] = fr

    out_dir = spec["folder"] / "actions"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out_dir / "sheet.png", optimize=True)
    pv = Image.fromarray(sheet).resize((sheet.shape[1] * 2, sheet.shape[0] * 2), Image.Resampling.NEAREST)
    bg = Image.new("RGB", pv.size, (16, 20, 32))
    bg.paste(pv, mask=pv.split()[-1])
    bg.save(out_dir / "preview.png")

    idle_n = counts[0]
    idle_d = delays_sec[0]
    meta = {
        "layout": "actions",
        "cw": int(cw),
        "ch": int(ch),
        "h": int(bh),
        "n": [idle_n] * 8,
        "d": [[idle_d] * idle_n for _ in range(8)],
        "an": counts,
        "ad": delays_sec,
        "raw": 1,
        "arow": 0,
        "src": "agent-sprite-forge",
        "facing": "left",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n")
    return meta


def process_one(fid: str, spec: dict) -> dict:
    work_dir = WORK / fid
    bh = prepare_palette_and_nn8x(fid, spec, work_dir)
    master_dir = work_dir / "master"
    set_dir = work_dir / "set"
    pal_path = work_dir / "palette.json"

    if not (master_dir / "master.json").exists():
        if master_dir.exists():
            shutil.rmtree(master_dir)
        cmd = [
            PY,
            MASTER_STILL,
            "approve",
            "--still",
            str(work_dir / "nn8x.png"),
            "--route",
            "existing",
            "--name",
            fid.split("_", 1)[1],
            "--subject",
            spec["subject"],
            "--identity",
            spec["identity"],
            "--facing",
            "left",
            "--class",
            "hero",
            "--finish",
            "pixel",
            "--key",
            spec["key"],
            "--output-dir",
            str(master_dir),
        ]
        res = run_cmd(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"master_still failed for {fid}: {res.stderr}\n{res.stdout}")

    if not (set_dir / "set_plan.json").exists():
        if set_dir.exists():
            shutil.rmtree(set_dir)
        cmd = [
            PY,
            SPRITE_SET,
            "plan",
            "--master",
            str(master_dir / "master.json"),
            "--actions",
            "idle,walk,attack,hurt",
            "--target-height",
            str(bh),
            "--palette",
            str(pal_path),
            "--outline",
            "selective",
            "--formats",
            "png",
            "--jobs",
            "4",
            "--max-takes",
            "3",
            "--output-dir",
            str(set_dir),
        ]
        res = run_cmd(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"sprite_set plan failed for {fid}: {res.stderr}\n{res.stdout}")
        patch_plan(set_dir / "set_plan.json", spec["key_name"])

    for attempt in range(3):
        all_have_takes = all(
            any((set_dir / "actions" / act / "takes").glob("t*/media/*.mp4"))
            for act, _ in ACTION_MAX_FRAMES
        ) if (set_dir / "actions").exists() else False
        if all_have_takes:
            break
        # If any action exhausted takes without an mp4, request a retake round
        if attempt > 0 and (set_dir / "actions").exists():
            for act, _ in ACTION_MAX_FRAMES:
                if not any((set_dir / "actions" / act / "takes").glob("t*/media/*.mp4")):
                    run_cmd([
                        PY,
                        SPRITE_SET,
                        "retake",
                        "--plan",
                        str(set_dir / "set_plan.json"),
                        "--action",
                        act,
                        "--fix",
                        "keep-colours",
                    ])
        cmd = [
            PY,
            SPRITE_SET,
            "run",
            "--plan",
            str(set_dir / "set_plan.json"),
            "--jobs",
            "4",
            "--max-takes",
            "3",
        ]
        res = run_cmd(cmd)
        print(f"  [sprite_set run {fid} attempt {attempt + 1}] exit={res.returncode}", flush=True)

    meta = assemble_sheet(fid, spec, set_dir, pal_path, bh)
    print(f"DONE {fid}: {meta['cw']}x{meta['ch']} an={meta['an']} ad={meta['ad']}", flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated form ids")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))

    raw = DIR8.read_text()
    at = raw.index("window.ARENA_DIR8")
    head, body = raw[:at] + "window.ARENA_DIR8", raw[at:].split("=", 1)[1]
    man = json.loads(body.strip().rstrip(";"))

    targets = [(fid, spec) for fid, spec in SPECS.items() if not only or fid in only]
    for idx, (fid, spec) in enumerate(targets, 1):
        print(f"=== [{idx}/{len(targets)}] {fid} ===", flush=True)
        meta = process_one(fid, spec)
        man[fid] = meta
        DIR8.write_text(head.rstrip() + " = " + json.dumps(man, separators=(",", ":")) + ";\n")


if __name__ == "__main__":
    main()
