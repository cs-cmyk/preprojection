"""
Additional benchmarks: HellaSwag, PIQA, and LoRA comparison.

Evaluates:
1. Pre-projection (frozen probe) on HellaSwag and PIQA
2. LoRA baseline with same parameter budget for comparison

Usage:
    # Run all benchmarks on Pythia-160M with pre-projection
    python 07_benchmarks.py --model_name EleutherAI/pythia-160m --method preprojection

    # Run LoRA comparison
    python 07_benchmarks.py --model_name EleutherAI/pythia-160m --method lora

    # Run both methods
    python 07_benchmarks.py --model_name EleutherAI/pythia-160m --method both

    # Pythia-410M
    python 07_benchmarks.py --model_name EleutherAI/pythia-410m --method both --batch_size 2 --grad_accum 8
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
from preprojection import inject_preprojection, count_preprojection_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--method", type=str, default="both",
                        choices=["preprojection", "lora", "both", "combined"],
                        help="'both' runs preproj and lora separately; 'combined' runs preproj+lora together")
    parser.add_argument("--expansion", type=float, default=1.25)
    parser.add_argument("--nonlinearity", type=str, default="silu")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="LoRA rank. 0 = auto-compute to match pre-proj param count")
    parser.add_argument("--eval_samples", type=int, default=500,
                        help="Samples per benchmark (0 = full)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results/benchmarks")
    return parser.parse_args()


# ============================================================
# Evaluation functions
# ============================================================

def evaluate_lambada(model, tokenizer, max_samples=500):
    """LAMBADA final-word prediction accuracy."""
    dataset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    num = len(dataset) if max_samples == 0 else min(max_samples, len(dataset))

    with torch.no_grad():
        for i in range(num):
            text = dataset[i]["text"]
            words = text.rsplit(" ", 1)
            if len(words) != 2:
                continue
            context, target_word = words
            context_ids = tokenizer(context, return_tensors="pt")["input_ids"].to(device)
            target_ids = tokenizer(" " + target_word, return_tensors="pt")["input_ids"][0].to(device)
            if context_ids.shape[1] + len(target_ids) > 512:
                continue

            all_correct = True
            current_ids = context_ids
            for target_token in target_ids:
                outputs = model(current_ids)
                if outputs.logits[0, -1].argmax() != target_token:
                    all_correct = False
                    break
                current_ids = torch.cat([current_ids, target_token.unsqueeze(0).unsqueeze(0)], dim=1)
            if all_correct:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0, correct, total


def evaluate_hellaswag(model, tokenizer, max_samples=500):
    """
    HellaSwag: pick the most likely continuation from 4 choices.
    Measures commonsense reasoning about physical situations.
    """
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    num = len(dataset) if max_samples == 0 else min(max_samples, len(dataset))

    with torch.no_grad():
        for i in range(num):
            sample = dataset[i]
            ctx = sample["ctx"]
            endings = sample["endings"]
            label = int(sample["label"])

            best_score = float("-inf")
            best_idx = -1

            for j, ending in enumerate(endings):
                text = ctx + " " + ending
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                input_ids = inputs["input_ids"]

                outputs = model(input_ids)
                logits = outputs.logits

                # Score = sum of log probs for the ending tokens
                # First, find where the ending starts
                ctx_ids = tokenizer(ctx, return_tensors="pt")["input_ids"]
                ctx_len = ctx_ids.shape[1]

                # Get log probs for ending tokens
                shift_logits = logits[0, ctx_len - 1:-1]
                shift_labels = input_ids[0, ctx_len:]
                log_probs = torch.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
                score = token_log_probs.sum().item()

                # Normalize by length to avoid bias toward short endings
                score = score / max(len(shift_labels), 1)

                if score > best_score:
                    best_score = score
                    best_idx = j

            if best_idx == label:
                correct += 1
            total += 1

            if (i + 1) % 100 == 0:
                print(f"    HellaSwag {i+1}/{num}: running acc={correct/total:.4f}")

    return correct / total if total > 0 else 0, correct, total


def evaluate_arc_easy(model, tokenizer, max_samples=500):
    """
    ARC-Easy: AI2 Reasoning Challenge (easy set).
    Multiple-choice science questions. Measures reasoning ability.
    """
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    num = len(dataset) if max_samples == 0 else min(max_samples, len(dataset))

    with torch.no_grad():
        for i in range(num):
            sample = dataset[i]
            question = sample["question"]
            choices = sample["choices"]
            answer_key = sample["answerKey"]

            labels = choices["label"]
            texts = choices["text"]

            # Find correct index
            if answer_key in labels:
                correct_idx = labels.index(answer_key)
            else:
                continue

            best_score = float("-inf")
            best_idx = -1

            for j, choice_text in enumerate(texts):
                text = question + " " + choice_text
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                input_ids = inputs["input_ids"]

                outputs = model(input_ids)
                logits = outputs.logits

                q_ids = tokenizer(question, return_tensors="pt")["input_ids"]
                q_len = q_ids.shape[1]

                shift_logits = logits[0, q_len - 1:-1]
                shift_labels = input_ids[0, q_len:]
                log_probs = torch.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
                score = token_log_probs.sum().item() / max(len(shift_labels), 1)

                if score > best_score:
                    best_score = score
                    best_idx = j

            if best_idx == correct_idx:
                correct += 1
            total += 1

            if (i + 1) % 100 == 0:
                print(f"    ARC-Easy {i+1}/{num}: running acc={correct/total:.4f}")

    return correct / total if total > 0 else 0, correct, total

    return correct / total if total > 0 else 0, correct, total


def evaluate_perplexity(model, tokenizer, max_samples=300):
    """WikiText-103 validation perplexity."""
    model.eval()
    device = next(model.parameters()).device
    wikitext = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation", streaming=True)
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i, sample in enumerate(wikitext):
            if i >= max_samples:
                break
            if len(sample["text"].strip()) < 50:
                continue
            inputs = tokenizer(sample["text"], return_tensors="pt", truncation=True, max_length=512).to(device)
            if inputs["input_ids"].shape[1] < 10:
                continue
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            count += 1
    avg_loss = total_loss / count
    return torch.exp(torch.tensor(avg_loss)).item()


def run_all_evals(model, tokenizer, max_samples, label=""):
    """Run all benchmarks and return results dict."""
    print(f"\n  Evaluating {label}...")

    print("    LAMBADA...")
    lambada_acc, lc, lt = evaluate_lambada(model, tokenizer, max_samples)
    print(f"    LAMBADA: {lambada_acc:.4f} ({lc}/{lt})")

    print("    HellaSwag...")
    hellaswag_acc, hc, ht = evaluate_hellaswag(model, tokenizer, max_samples)
    print(f"    HellaSwag: {hellaswag_acc:.4f} ({hc}/{ht})")

    print("    ARC-Easy...")
    arc_acc, ac, at = evaluate_arc_easy(model, tokenizer, max_samples)
    print(f"    ARC-Easy: {arc_acc:.4f} ({ac}/{at})")

    print("    WikiText perplexity...")
    ppl = evaluate_perplexity(model, tokenizer)
    print(f"    WikiText PPL: {ppl:.2f}")

    return {
        "lambada_accuracy": lambada_acc,
        "hellaswag_accuracy": hellaswag_acc,
        "arc_easy_accuracy": arc_acc,
        "wikitext_ppl": ppl,
    }


# ============================================================
# Training functions
# ============================================================

def prepare_training_data(tokenizer, seq_length, num_samples=2000):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    texts = []
    for sample in dataset:
        if len(sample["text"].strip()) > 50:
            texts.append(sample["text"])
        if len(texts) >= num_samples:
            break
    ds = Dataset.from_dict({"text": texts})
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=seq_length, padding="max_length")
    return ds.map(tokenize, batched=True, remove_columns=["text"])


def train_frozen_probe(model, tokenizer, args):
    """Train only the trainable params using HF Trainer."""
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


def setup_lora(model, target_param_count, args):
    """
    Apply LoRA to match the pre-projection parameter count.
    Targets the Q/K/V and output projections in attention.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("ERROR: peft not installed. Run: pip install peft")
        return None, 0

    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    # Compute rank to approximately match target param count
    # LoRA on qkv + dense: each adds 2 * rank * dim params per target module
    # For GPT-NeoX: query_key_value (3*dim x dim) and dense (dim x dim)
    # LoRA params per layer = 2 * rank * dim * num_target_modules
    # Total = layers * 2 * rank * dim * num_targets
    num_targets = 2  # query_key_value, dense
    if args.lora_rank > 0:
        rank = args.lora_rank
    else:
        # Solve: target = layers * 2 * rank * dim * num_targets
        rank = int(target_param_count / (num_layers * 2 * hidden_dim * num_targets))
        rank = max(rank, 4)  # minimum rank

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.0,
        target_modules=["query_key_value", "dense"],
        bias="none",
    )

    model = get_peft_model(model, config)
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA rank: {rank}")
    print(f"  LoRA trainable params: {lora_params:,}")
    print(f"  Target was: {target_param_count:,}")

    return model, lora_params


# ============================================================
# Main
# ============================================================

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

    # Pre-projection
    if args.method in ["preprojection", "both"]:
        print("\n" + "=" * 60)
        print("PRE-PROJECTION (frozen probe)")
        print("=" * 60)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).cuda()
        model, _ = inject_preprojection(
            model, expansion=args.expansion,
            nonlinearity=args.nonlinearity, variant="standard",
            freeze_base=True,
        )
        preproj_params = count_preprojection_params(model)

        print(f"  Training {preproj_params:,} pre-projection params...")
        model = train_frozen_probe(model, tokenizer, args)
        results["preprojection"] = run_all_evals(model, tokenizer, args.eval_samples, "Pre-projection")
        results["preprojection"]["trainable_params"] = preproj_params

        del model
        torch.cuda.empty_cache()

    # LoRA comparison
    if args.method in ["lora", "both"]:
        print("\n" + "=" * 60)
        print("LoRA (frozen probe, matched params)")
        print("=" * 60)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).cuda()

        # Match param count to pre-projection
        if "preprojection" in results:
            target_params = results["preprojection"]["trainable_params"]
        else:
            # Estimate
            hidden = model.config.hidden_size
            layers = model.config.num_hidden_layers
            target_params = int(2 * hidden * hidden * args.expansion * layers)

        model, lora_params = setup_lora(model, target_params, args)
        if model is not None:
            print(f"  Training {lora_params:,} LoRA params...")
            model = train_frozen_probe(model, tokenizer, args)
            results["lora"] = run_all_evals(model, tokenizer, args.eval_samples, "LoRA")
            results["lora"]["trainable_params"] = lora_params

        del model
        torch.cuda.empty_cache()

    # Combined pre-projection + LoRA
    if args.method in ["combined", "both"]:
        print("\n" + "=" * 60)
        print("COMBINED: Pre-projection + LoRA")
        print("=" * 60)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).cuda()

        # Step 1: inject pre-projection (freeze base)
        model, _ = inject_preprojection(
            model, expansion=args.expansion,
            nonlinearity=args.nonlinearity, variant="standard",
            freeze_base=True,
        )
        preproj_params = count_preprojection_params(model)

        # Step 2: apply LoRA to the linear inside the wrapper + dense
        # After injection, QKV linear is at query_key_value.linear
        # Use a moderate rank — LoRA supplements the pre-projection
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            hidden_dim = model.config.hidden_size
            num_layers = model.config.num_hidden_layers
            # Use rank 64 as a reasonable default, or match standalone LoRA
            lora_rank = args.lora_rank if args.lora_rank > 0 else 64

            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                lora_dropout=0.0,
                target_modules=["linear", "dense"],
                bias="none",
            )

            model = get_peft_model(model, config)

            # PEFT freezes all non-LoRA params — unfreeze pre-projection
            for name, param in model.named_parameters():
                if "preproj" in name:
                    param.requires_grad = True

            total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            lora_only = total_trainable - preproj_params
            print(f"  Pre-proj params: {preproj_params:,}")
            print(f"  LoRA params: {lora_only:,}")
            print(f"  Total trainable: {total_trainable:,}")

            print(f"  Training combined model...")
            model = train_frozen_probe(model, tokenizer, args)
            results["combined"] = run_all_evals(model, tokenizer, args.eval_samples, "Combined")
            results["combined"]["trainable_params"] = total_trainable
            results["combined"]["preproj_params"] = preproj_params
            results["combined"]["lora_params"] = lora_only

        except ImportError:
            print("ERROR: peft not installed. Run: pip install peft")

        del model
        torch.cuda.empty_cache()

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    methods = [k for k in ["baseline", "preprojection", "lora", "combined"] if k in results]
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

    if "preprojection" in results:
        print(f"\n  Pre-proj params: {results['preprojection'].get('trainable_params', 'N/A'):,}")
    if "lora" in results:
        print(f"  LoRA params:     {results['lora'].get('trainable_params', 'N/A'):,}")
    if "combined" in results:
        print(f"  Combined params: {results['combined'].get('trainable_params', 'N/A'):,} "
              f"(pre-proj: {results['combined'].get('preproj_params', 'N/A'):,} + "
              f"LoRA: {results['combined'].get('lora_params', 'N/A'):,})")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = args.model_name.split("/")[-1]
    output_path = os.path.join(args.output_dir, f"benchmarks_{model_tag}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
