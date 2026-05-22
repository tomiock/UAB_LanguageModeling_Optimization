"""
Evaluate final_model_epoch and final_model_half2e on the CNN/DailyMail test set.

Metrics computed:
  ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum   (n-gram overlap)
  BERTScore F1 / P / R                     (semantic similarity via RoBERTa-large)

Generated summaries are cached to disk so metric recomputation is free.

Usage:
    CUDA_VISIBLE_DEVICES=4 python evaluate_epoch_vs_repeat.py
    CUDA_VISIBLE_DEVICES=4 python evaluate_epoch_vs_repeat.py --max-samples 500  # quick smoke-test
    CUDA_VISIBLE_DEVICES=4 python evaluate_epoch_vs_repeat.py --batch-size 64
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import evaluate as hf_evaluate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = {
    "1 epoch  (full data)":    "final_model_epoch",
    "2 epochs (half data)":    "final_model_half2e",
}

DATASET_NAME    = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"
PROMPT_TEMPLATE = "Summarize the following article.\n\n### Article:\n{article}\n\n### Summary:\n"

MAX_INPUT_LEN   = 1024
MAX_NEW_TOKENS  = 128
BERTSCORE_MODEL = "roberta-large"

METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def load_model_and_tok(model_path: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"    # required for correct batched generation
    tok.pad_token    = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16
    ).to(device)
    model.eval()
    return model, tok


def generate_summaries(
    model_path: str,
    articles: list[str],
    batch_size: int,
    device: str,
) -> list[str]:
    cache_path = METRICS_DIR / f"{Path(model_path).name}_test_summaries_n{len(articles)}.json"

    if cache_path.exists():
        print(f"  [cache] loading generated summaries from {cache_path}")
        return json.loads(cache_path.read_text())

    print(f"  [gen]   loading model {model_path} ...")
    model, tok = load_model_and_tok(model_path, device)

    summaries: list[str] = []
    batches = [articles[i : i + batch_size] for i in range(0, len(articles), batch_size)]

    t0 = time.perf_counter()
    for batch_articles in tqdm(batches, desc=f"  generating ({Path(model_path).name})", unit="batch"):
        prompts = [PROMPT_TEMPLATE.format(article=a) for a in batch_articles]
        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
        ).to(device)

        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

        # With left-padding all sequences share the same prompt_len offset.
        new_tok = out[:, prompt_len:]
        summaries.extend(tok.batch_decode(new_tok, skip_special_tokens=True))

    elapsed = time.perf_counter() - t0
    rate    = len(articles) / elapsed
    print(f"  [gen]   done — {len(articles)} examples in {elapsed:.0f}s  ({rate:.1f} ex/s)")

    cache_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"  [cache] summaries saved to {cache_path}")

    del model
    torch.cuda.empty_cache()
    return summaries


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    rouge = hf_evaluate.load("rouge")
    result = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    return {k: round(v * 100, 3) for k, v in result.items()}   # percent


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    bscore = hf_evaluate.load("bertscore")
    result = bscore.compute(
        predictions=predictions,
        references=references,
        model_type=BERTSCORE_MODEL,
        lang="en",
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return {
        "bertscore_p":  round(float(np.mean(result["precision"])) * 100, 3),
        "bertscore_r":  round(float(np.mean(result["recall"]))    * 100, 3),
        "bertscore_f1": round(float(np.mean(result["f1"]))        * 100, 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--max-samples",  type=int, default=None,
                   help="Truncate test set (useful for a quick smoke-test)")
    p.add_argument("--no-bertscore", action="store_true",
                   help="Skip BERTScore (saves ~5 min)")
    p.add_argument("--output", default="metrics/eval_epoch_vs_repeat.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _bar(val_a, val_b, width=8):
    """Return '▲ +X.XXX' or '▼ -X.XXX' relative to val_a."""
    diff = val_b - val_a
    sym  = "▲" if diff >= 0 else "▼"
    return f"{sym} {diff:+.3f}"


def print_table(results: dict, include_bertscore: bool):
    labels = list(results.keys())
    metrics_rouge = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    metrics_bs    = ["bertscore_p", "bertscore_r", "bertscore_f1"] if include_bertscore else []
    all_metrics   = metrics_rouge + metrics_bs

    col_w   = 12
    name_w  = max(len(l) for l in labels) + 2

    header = f"{'Model':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in all_metrics)
    sep    = "─" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)
    for i, label in enumerate(labels):
        row = f"{label:<{name_w}}"
        for m in all_metrics:
            v = results[label].get(m, float("nan"))
            row += f"{v:>{col_w}.3f}"
        print(row)

    # Delta row (second minus first)
    if len(labels) == 2:
        print(sep)
        row = f"{'Δ (2nd − 1st)':<{name_w}}"
        for m in all_metrics:
            a = results[labels[0]].get(m, float("nan"))
            b = results[labels[1]].get(m, float("nan"))
            row += f"{_bar(a, b):>{col_w}}"
        print(row)

    print(sep + "\n")
    print("  All ROUGE scores are ×100 (percentage).  BERTScore is also ×100.")
    print("  use_stemmer=True for ROUGE;  roberta-large for BERTScore.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  CNN/DailyMail test-set evaluation")
    print(f"  device : {device}  |  batch_size : {args.batch_size}")
    print(f"{'='*60}\n")

    # Load test data
    test_ds = load_dataset(DATASET_NAME, DATASET_VERSION, split="test")
    if args.max_samples:
        test_ds = test_ds.select(range(args.max_samples))

    articles    = test_ds["article"]
    references  = test_ds["highlights"]
    n           = len(articles)
    print(f"  Test examples: {n:,}\n")

    results: dict = {}

    for label, model_path in MODELS.items():
        print(f"── {label} ──")

        predictions = generate_summaries(model_path, articles, args.batch_size, device)

        print(f"  [rouge]     computing ROUGE ...")
        rouge = compute_rouge(predictions, references)
        print(f"              R1={rouge['rouge1']:.2f}  R2={rouge['rouge2']:.2f}  RL={rouge['rougeL']:.2f}  RLsum={rouge['rougeLsum']:.2f}")

        bscore = {}
        if not args.no_bertscore:
            print(f"  [bertscore] computing BERTScore ({BERTSCORE_MODEL}) ...")
            bscore = compute_bertscore(predictions, references)
            print(f"              P={bscore['bertscore_p']:.2f}  R={bscore['bertscore_r']:.2f}  F1={bscore['bertscore_f1']:.2f}")

        results[label] = {
            "model_path":    model_path,
            "num_examples":  n,
            "max_new_tokens": MAX_NEW_TOKENS,
            "bertscore_model": BERTSCORE_MODEL if not args.no_bertscore else None,
            **rouge,
            **bscore,
        }
        print()

    # Save
    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Results saved to {out_path}\n")

    # Print table
    print_table(results, include_bertscore=not args.no_bertscore)


if __name__ == "__main__":
    main()
