#!/usr/bin/env python3
"""Keep up to 8 fresh grok processes, one gen5 form each."""

import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
PY = "/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge/.venv/bin/python"
EVENT = Path.home() / ".grok/long-running-background-tasks/gen5_forge_events.log"
LOGDIR = Path("/tmp/gen5-forge-logs")
SOCKDIR = Path("/tmp/gen5-forge-sockets")
PROMPTDIR = Path("/tmp/gen5-forge-prompts")
TEMPLATE = ROOT / "tools/gen5_monster_prompt.txt"
SLOTS = int(os.environ.get("GEN5_SLOTS", "8"))
PROMPT_RE = re.compile(r"gen5-forge-prompts/([A-Za-z0-9_.-]+)\.txt")


def event(text):
    EVENT.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H:%M:%S")
    with EVENT.open("a") as fh:
        fh.write(f"{stamp} {text}\n")


def queue(*args):
    return subprocess.run(
        [PY, "-I", "tools/gen5_queue.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def gate_ok():
    result = queue("gate")
    line = " ".join((result.stderr or "").split())
    event(f"GATE {result.returncode} {line}")
    if result.returncode == 0:
        return "ok"
    if "CLOCK_HOLD" in (result.stderr or ""):
        return "clock"
    return "weekly"


def live_jobs():
    out = subprocess.check_output(["ps", "aux"], text=True, errors="replace")
    jobs = {}
    for line in out.splitlines():
        if "gen5-forge-prompts/" not in line or "grok " not in line:
            continue
        match = PROMPT_RE.search(line)
        if not match:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        jobs[match.group(1)] = int(parts[1])
    return jobs


def claim():
    result = queue("claim")
    if result.returncode == 3:
        return None
    if result.returncode == 4:
        if "CLOCK_HOLD" in (result.stderr or ""):
            return "clock"
        return "hold"
    if result.returncode != 0:
        event(f"CLAIM_FAIL {result.returncode} {(result.stderr or '').strip()}")
        return False
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    event("CLAIM_FAIL bad json")
    return False


def spawn(item):
    item_id = item["id"]
    prompt = PROMPTDIR / f"{item_id}.txt"
    prompt.write_text(TEMPLATE.read_text().replace("__ITEM_JSON__", json.dumps(item, ensure_ascii=False)))
    sid = str(uuid.uuid4())
    sock = SOCKDIR / f"{item_id}.sock"
    sock.unlink(missing_ok=True)
    log = LOGDIR / f"{item_id}.log"
    argv = [
        "grok",
        "--cwd", str(ROOT),
        "--session-id", sid,
        "--leader-socket", str(sock),
        "--prompt-file", str(prompt),
        "--verbatim",
        "--no-subagents",
        "--disable-web-search",
        "--max-turns", "70",
        "--permission-mode", "bypassPermissions",
        "--always-approve",
        "--no-alt-screen",
        "--output-format", "plain",
    ]
    handle = log.open("w")
    proc = subprocess.Popen(
        argv,
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    event(f"START {item_id}")
    return proc, sid


def reap(item_id, grok_code):
    status = queue("status-one", item_id)
    state = (status.stdout or "").strip() or "missing"
    event(f"CHILD_EXIT {item_id} grok={grok_code} status={state}")
    if state == "done":
        event(f"MONSTER_DONE {item_id}")
        return
    if state == "needs_fix":
        event(f"MONSTER_NEEDS_FIX {item_id}")
        return
    released = queue("release", item_id)
    new_state = (released.stdout or "").strip() or "missing"
    event(f"RELEASE {item_id} {new_state}")
    if new_state == "needs_fix":
        event(f"MONSTER_NEEDS_FIX {item_id}")
    elif new_state == "pending":
        event(f"RETRY {item_id}")


def main():
    for path in (LOGDIR, SOCKDIR, PROMPTDIR):
        path.mkdir(parents=True, exist_ok=True)
    event(f"LOOP_PID {os.getpid()} SLOTS {SLOTS}")
    launched = {}
    adopted = set(live_jobs())
    hold = False
    while True:
        jobs = live_jobs()
        finished = [item_id for item_id in list(launched) if launched[item_id][0].poll() is not None]
        for item_id in finished:
            proc, _sid = launched.pop(item_id)
            reap(item_id, proc.returncode)
        for item_id in list(adopted):
            if item_id not in jobs and item_id not in launched:
                reap(item_id, "adopted")
                adopted.discard(item_id)
        jobs = live_jobs()
        active = set(launched) | set(jobs)
        if len(active) >= SLOTS:
            time.sleep(5)
            continue
        gate = gate_ok()
        if gate != "ok":
            if not hold:
                event("CLOCK_HOLD" if gate == "clock" else "WEEKLY_HOLD")
                hold = True
            if not active:
                return 4
            time.sleep(10)
            continue
        hold = False
        while len(active) < SLOTS:
            item = claim()
            if item is None or item in ("hold", "clock"):
                if item in ("hold", "clock") and not hold:
                    event("CLOCK_HOLD" if item == "clock" else "WEEKLY_HOLD")
                    hold = True
                break
            if item is False:
                event("LOOP_STOP claim failed")
                return 1
            if item["id"] in active:
                event(f"LOOP_STOP duplicate claim {item['id']}")
                return 1
            launched[item["id"]] = spawn(item)
            active.add(item["id"])
        if not active:
            event("QUEUE_EMPTY")
            return 0
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
