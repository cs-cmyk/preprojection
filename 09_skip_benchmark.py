"""
Benchmark for pre-projection with content skip connection.

Tests whether adding a residual path from the pre-projection
output directly to after attention output improves results
beyond standard pre-projection.

Usage:
    python 09_skip_benchmark.py --model_name EleutherAI/pythia-160m
    python 09_skip_benchmark.py --model_name EleutherAI/pythia-410m --batch_size 2 --grad_accum 8
"""

import torch
import argparse
import os
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset
from preprojection_skip import inject_preprojection_with_skip, count_skip_params

import sys
sys.path.insert(0, os.path.dirname(__file__))
benchmarks = __import__("07_benchmarks")
run_all_evals = benchmarks.run_all_evals
prepare_training_data = benchmarks.prepare_training_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--expansion", type=float, default=1.25)
    parser.add_argument("--nonlinearity", type=str, default="silu")
    parser.add_argument("--skip_scale", type=float, default=0.01,
                        help="Init scale for skip projection weights")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results/skip")
    return parser.parse_args()


def train_frozen_probe(model, tokenizer, args):
    train_dataset = prepare_training_data(tokenizer, args.seq_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "tmp_train"),
        max_steps=args.train_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    return model


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # Baseline
    print("=" * 60)
    print(f"BASELINE: {args.model_name}")
    print("=" * 60)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).cuda()
    results["baseline"] = run_all_evals(model, tokenizer, args.eval_samples, "Baseline")
    del model
    torch.cuda.empty_cache()

    # Pre-projection with skip
    print("\n" + "=" * 60)
    print("PRE-PROJECTION WITH CONTENT SKIP")
    print("=" * 60)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).cuda()

    model = inject_preprojection_with_skip(
        model,
        expansion=args.expansion,
        nonlinearity=args.nonlinearity,
        skip_scale=args.skip_scale,
        freeze_base=True,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Training {trainable_params:,} params...")
    model = train_frozen_probe(model, tokenizer, args)
    results["skip"] = run_all_evals(model, tokenizer, args.eval_samples, "Pre-proj + skip")
    results["skip"]["trainable_params"] = trainable_params

    # Show skip projection norms per layer to see how much each layer uses it
    print("\n  Skip projection weight norms per layer:")
    for i, layer in enumerate(model.gpt_neox.layers):
        qkv = layer.attention.query_key_value
        if hasattr(qkv, 'preproj'):
            skip_norm = qkv.preproj.skip_proj.weight.data.float().norm().item()
            up_norm = qkv.preproj.up_proj.weight.data.float().norm().item()
            bar = "|" + "#" * min(20, int(skip_norm * 10)) + " " * max(0, 20 - int(skip_norm * 10)) + "|"
            print(f"    Layer {i:2d}: skip_norm={skip_norm:.4f}  up_norm={up_norm:.4f}  {bar}")

    del model
    torch.cuda.empty_cache()

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    methods = [k for k in ["baseline", "skip"] if k in results]
    header = f"{'Metric':<22}" + "".join(f"{m:>14}" for m in methods)
    print(header)
    print("-" * len(header))

    for metric, label in [
        ("lambada_accuracy", "LAMBADA acc"),
        ("hellaswag_accuracy", "HellaSwag acc"),
        ("arc_easy_accuracy", "ARC-Easy acc"),
        ("wikitext_ppl", "WikiText PPL"),
    ]:
        row = f"{label:<22}"
        for m in methods:
            val = results[m].get(metric, 0)
            if "ppl" in metric:
                row += f"{val:>14.2f}"
            else:
                row += f"{val:>14.4f}"
        print(row)

    print(f"\n  Skip params: {results['skip'].get('trainable_params', 'N/A'):,}")

    # Reference: standard pre-projection results for comparison
    print("\n  Reference (standard pre-proj, 160M): LAMBADA=0.154, PPL=73.0")
    print("  Reference (combined pre-proj+LoRA, 160M): LAMBADA=0.162, PPL=60.8")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = args.model_name.split("/")[-1]
    output_path = os.path.join(args.output_dir, f"skip_{model_tag}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
