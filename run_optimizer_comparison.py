import json
import os
import re
import subprocess
import sys

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(SCRIPT_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

MAX_STEPS    = 600   # enough for stable throughput + one eval at step 500
LOG_STEPS    = 1
EVAL_STEPS   = 500
SAVE_STEPS   = 9999  # skip saving checkpoints during benchmark

RUNS = [
    ("standard", {"fused_adam": False}),
    ("fused",    {"fused_adam": True}),
]

LOG_RE = re.compile(r"\{[^{}]*'loss'[^{}]*\}")


def patch_src(overrides: dict) -> str:
    with open(os.path.join(SCRIPT_DIR, "finetune_summarization.py")) as f:
        src = f.read()

    for key, val in overrides.items():
        src = re.sub(
            rf"^\s*{key}\s*:.*$",
            f"    {key}: bool = {val}",
            src, flags=re.MULTILINE,
        )

    src = re.sub(r"(max_steps\s*=\s*)\d+",   rf"\g<1>{MAX_STEPS}", src)
    src = re.sub(r"(logging_steps\s*=\s*)\d+", rf"\g<1>{LOG_STEPS}", src)
    src = re.sub(r"(eval_steps\s*=\s*)\d+",   rf"\g<1>{EVAL_STEPS}", src)
    src = re.sub(r"(save_steps\s*=\s*)\d+",   rf"\g<1>{SAVE_STEPS}", src)
    return src


def run_and_capture(tag: str, overrides: dict) -> tuple[list[dict], dict]:
    print(f"\n{'='*60}")
    print(f"  Run: {tag}  |  overrides: {overrides}")
    print(f"{'='*60}", flush=True)

    src = patch_src(overrides)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "2",
           "WANDB_PROJECT": "qwen3-summarization"}

    result = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True, text=True, env=env, cwd=SCRIPT_DIR,
    )

    stdout = result.stdout + result.stderr   # HF Trainer logs go to stderr
    if result.returncode != 0:
        print(f"[ERROR] run failed:\n{stdout[-4000:]}", flush=True)
        return [], {}

    # --- parse training log lines ---
    logs = []
    for line in stdout.splitlines():
        m = LOG_RE.search(line)
        if m:
            try:
                d = json.loads(m.group().replace("'", '"'))
                logs.append(d)
            except json.JSONDecodeError:
                pass

    # --- parse the one-time optimizer breakdown ---
    opt_info = {}
    for line in stdout.splitlines():
        m = re.search(r"Optimizer\s*:\s*(.+)", line)
        if m:
            opt_info["optimizer"] = m.group(1).strip()
        m = re.search(r"Fused kernel\s*:\s*(.+)", line)
        if m:
            opt_info["fused_kernel"] = m.group(1).strip()
        m = re.search(r"State dtypes?\s*:\s*(.+)", line)
        if m:
            opt_info["state_dtypes"] = m.group(1).strip()
        m = re.search(r"State VRAM\s*:\s*([0-9.]+)\s*GB", line)
        if m:
            opt_info["opt_vram_gb"] = float(m.group(1))

    return logs, opt_info


def summarise(tag: str, logs: list[dict], opt_info: dict) -> None:
    train_logs = [d for d in logs if "loss" in d and "eval_loss" not in d]
    eval_logs  = [d for d in logs if "eval_loss" in d]
    tps   = [float(d["throughput/tokens_per_sec"]) for d in train_logs if "throughput/tokens_per_sec" in d]
    vram  = [float(d["memory/peak_vram_gb"])        for d in train_logs if "memory/peak_vram_gb" in d]
    losses = [float(d["loss"]) for d in train_logs]

    print(f"\n  [{tag}] optimizer: {opt_info.get('optimizer', '?')}")
    print(f"  [{tag}] fused kernel: {opt_info.get('fused_kernel', '?')}")
    print(f"  [{tag}] state dtypes: {opt_info.get('state_dtypes', '?')}")
    if "opt_vram_gb" in opt_info:
        print(f"  [{tag}] state VRAM: {opt_info['opt_vram_gb']:.2f} GB")
    if tps:
        print(f"  [{tag}] tok/s avg: {sum(tps)/len(tps):.0f}")
    if vram:
        print(f"  [{tag}] peak VRAM avg: {sum(vram)/len(vram):.2f} GB")
    if losses:
        print(f"  [{tag}] loss: {losses[0]:.3f} → {losses[-1]:.3f}")
    if eval_logs:
        print(f"  [{tag}] eval loss: {float(eval_logs[-1]['eval_loss']):.3f}")


def main():
    all_results = {}
    for tag, overrides in RUNS:
        logs, opt_info = run_and_capture(tag, overrides)

        out = os.path.join(METRICS_DIR, f"optimizer_{tag}.jsonl")
        with open(out, "w") as f:
            for d in logs:
                f.write(json.dumps(d) + "\n")
            if opt_info:
                f.write(json.dumps({"_optimizer_info": opt_info}) + "\n")

        summarise(tag, logs, opt_info)
        print(f"  Saved → {out}", flush=True)
        all_results[tag] = (logs, opt_info)

    # side-by-side VRAM comparison
    print("\n" + "="*60)
    print("  VRAM comparison (peak during training)")
    for tag, (logs, _) in all_results.items():
        train = [d for d in logs if "memory/peak_vram_gb" in d]
        if train:
            avg = sum(float(d["memory/peak_vram_gb"]) for d in train) / len(train)
            print(f"    {tag:8s}: {avg:.2f} GB")
    print("="*60)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
