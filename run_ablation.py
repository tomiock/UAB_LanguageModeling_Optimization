"""
Runs all four padding strategies back-to-back on CUDA_VISIBLE_DEVICES=0.
Saves per-strategy JSONL logs to metrics/ablation_<strategy>.jsonl.
"""

import json
import os
import re
import subprocess
import sys

STRATEGIES = [
    ("fixed",   "finetune_padding.py",      {"PADDING_MODE": "PaddingMode.FIXED"}),
]

MAX_STEPS  = 300
LOG_STEPS  = 50
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(SCRIPT_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

# Regex to match Hugging Face Trainer log lines:  {'loss': ..., ...}
LOG_RE = re.compile(r"\{[^}]*'loss'[^}]*\}")


def patch_and_run(script: str, overrides: dict) -> list[dict]:
    """
    Reads <script>, monkey-patches the constants listed in overrides,
    injects MAX_STEPS / logging_steps overrides into COMMON_ARGS (or
    TrainingArguments for the varlen script), then executes in a subprocess.
    Returns parsed log dicts.
    """
    src_path = os.path.join(SCRIPT_DIR, script)
    with open(src_path) as f:
        src = f.read()

    # Patch top-level constants (e.g. PADDING_MODE)
    for key, val in overrides.items():
        src = re.sub(rf"^{key}\s*=.*$", f"{key} = {val}", src, flags=re.MULTILINE)

    # Patch max_steps / logging_steps in COMMON_ARGS dict (padding scripts)
    src = re.sub(r"(max_steps\s*=\s*)\d+",   rf"\g<1>{MAX_STEPS}", src)
    src = re.sub(r"(logging_steps\s*=\s*)\d+", rf"\g<1>{LOG_STEPS}", src)
    # Also patch eval_steps / save_steps to avoid extra work
    src = re.sub(r"(eval_steps\s*=\s*)\d+",  r"\g<1>9999", src)
    src = re.sub(r"(save_steps\s*=\s*)\d+",  r"\g<1>9999", src)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    result = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True, text=True, env=env,
        cwd=SCRIPT_DIR,
    )

    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed:\n{result.stderr[-3000:]}", flush=True)
        return []

    logs = []
    for line in result.stdout.splitlines():
        m = LOG_RE.search(line)
        if m:
            try:
                d = json.loads(m.group().replace("'", '"'))
                logs.append(d)
            except json.JSONDecodeError:
                pass
    return logs


def main():
    all_results = {}
    for name, script, overrides in STRATEGIES:
        print(f"\n{'='*60}", flush=True)
        print(f"  Running: {name}  ({script})", flush=True)
        print(f"{'='*60}", flush=True)

        logs = patch_and_run(script, overrides)

        out_path = os.path.join(METRICS_DIR, f"ablation_{name}.jsonl")
        with open(out_path, "w") as f:
            for d in logs:
                f.write(json.dumps(d) + "\n")

        # Print a summary
        tp  = [float(d["throughput/active_tokens_per_sec"]) for d in logs if "throughput/active_tokens_per_sec" in d]
        mem = [float(d["memory/peak_vram_gb"])              for d in logs if "memory/peak_vram_gb" in d]
        if tp:
            print(f"  active_tokens/s  : {sum(tp)/len(tp):.0f}  (avg of {len(tp)} logs)")
        if mem:
            print(f"  peak VRAM (GB)   : {sum(mem)/len(mem):.2f}")

        all_results[name] = logs
        print(f"  Saved → {out_path}", flush=True)

    print("\n\nAll strategies done.", flush=True)


if __name__ == "__main__":
    main()
