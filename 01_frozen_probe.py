"""
Phase 1: Frozen Probe Test

Validates the pre-projection concept with minimal compute:
1. Load pretrained Pythia-160M
2. Freeze all original weights
3. Inject pre-projection layers (only new params are trainable)
4. Evaluate on LAMBADA (next-token prediction accuracy)
5. Fine-tune only pre-projection params for a few steps
6. Re-evaluate — if accuracy improves, the architecture has merit

This should run in under 30 minutes on an RTX 4080 12GB.
"""

import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from preprojection import inject_preprojection, count_preprojection_params


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen probe test for pre-projection")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-160m",
                        help="Pretrained model to use")
    parser.add_argument("--expansion", type=float, default=2.0,
                        help="Pre-projection MLP expansion ratio")
    parser.add_argument("--nonlinearity", type=str, default="silu",
                        choices=["silu", "gelu", "relu", "mish"],
                        help="Nonlinearity in pre-projection")
    parser.add_argument("--variant", type=str, default="standard",
                        choices=["standard", "gated"],
                        help="Pre-projection variant")
    parser.add_argument("--train_steps", type=int, default=500,
                        help="Number of fine-tuning steps")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for pre-projection params")
    parser.add_argument("--seq_length", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results/frozen_probe")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log to Weights & Biases")
    return parser.parse_args()


def evaluate_perplexity(model, tokenizer, dataset, max_samples=500, seq_length=512):
    """Evaluate perplexity on a dataset."""
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            
            inputs = tokenizer(
                sample["text"],
                return_tensors="pt",
                truncation=True,
                max_length=seq_length,
            ).to(device)
            
            if inputs["input_ids"].shape[1] < 10:
                continue
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return {"perplexity": perplexity, "avg_loss": avg_loss, "tokens_evaluated": total_tokens}


def evaluate_lambada(model, tokenizer, max_samples=500, seq_length=512):
    """
    Evaluate on LAMBADA — tests ability to predict the final word
    of a passage given context. Measures language understanding.
    """
    dataset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    model.eval()
    device = next(model.parameters()).device
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            
            text = sample["text"]
            # Split into context and target (last word)
            words = text.rsplit(" ", 1)
            if len(words) != 2:
                continue
            
            context, target_word = words
            
            # Tokenize context
            context_ids = tokenizer(context, return_tensors="pt")["input_ids"].to(device)
            # Tokenize target (with leading space)
            target_ids = tokenizer(" " + target_word, return_tensors="pt")["input_ids"][0].to(device)
            
            if context_ids.shape[1] + len(target_ids) > seq_length:
                continue
            
            # Get model predictions for each target token
            all_correct = True
            current_ids = context_ids
            
            for target_token in target_ids:
                outputs = model(current_ids)
                predicted = outputs.logits[0, -1].argmax()
                if predicted != target_token:
                    all_correct = False
                    break
                current_ids = torch.cat([
                    current_ids,
                    target_token.unsqueeze(0).unsqueeze(0)
                ], dim=1)
            
            if all_correct:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return {"lambada_accuracy": accuracy, "correct": correct, "total": total}


def prepare_training_data(tokenizer, seq_length=512, num_samples=2000):
    """Load a small training set for fine-tuning the pre-projection."""
    dataset = load_dataset(
        "wikitext", "wikitext-103-raw-v1",
        split="train",
        streaming=True,
    )
    
    texts = []
    for sample in dataset:
        if len(sample["text"].strip()) > 50:
            texts.append(sample["text"])
        if len(texts) >= num_samples:
            break
    
    from datasets import Dataset
    ds = Dataset.from_dict({"text": texts})
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_length,
            padding="max_length",
        )
    
    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("PHASE 1: FROZEN PROBE TEST")
    print("=" * 60)
    
    # Load model and tokenizer
    print(f"\nLoading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    ).cuda()
    
    # --- Baseline evaluation ---
    print("\n--- Baseline evaluation (original model) ---")
    
    lambada_baseline = evaluate_lambada(model, tokenizer, max_samples=500)
    print(f"LAMBADA accuracy: {lambada_baseline['lambada_accuracy']:.4f} "
          f"({lambada_baseline['correct']}/{lambada_baseline['total']})")
    
    wikitext_eval = load_dataset(
        "wikitext", "wikitext-103-raw-v1",
        split="validation",
        streaming=True,
    )
    ppl_baseline = evaluate_perplexity(
        model, tokenizer,
        wikitext_eval,
        max_samples=300,
        seq_length=args.seq_length,
    )
    print(f"WikiText perplexity: {ppl_baseline['perplexity']:.2f}")
    
    # --- Inject pre-projection ---
    print(f"\n--- Injecting pre-projection (variant={args.variant}, "
          f"expansion={args.expansion}, act={args.nonlinearity}) ---")
    
    model, new_params = inject_preprojection(
        model,
        expansion=args.expansion,
        nonlinearity=args.nonlinearity,
        variant=args.variant,
        freeze_base=True,  # Freeze everything except pre-projection
    )
    
    # --- Post-injection evaluation (should be near baseline due to near-identity init) ---
    print("\n--- Post-injection evaluation (before training, should match baseline) ---")
    
    lambada_injected = evaluate_lambada(model, tokenizer, max_samples=500)
    print(f"LAMBADA accuracy: {lambada_injected['lambada_accuracy']:.4f}")
    
    wikitext_eval = load_dataset(
        "wikitext", "wikitext-103-raw-v1",
        split="validation",
        streaming=True,
    )
    ppl_injected = evaluate_perplexity(
        model, tokenizer,
        wikitext_eval,
        max_samples=300,
        seq_length=args.seq_length,
    )
    print(f"WikiText perplexity: {ppl_injected['perplexity']:.2f}")
    
    # --- Fine-tune pre-projection only ---
    print(f"\n--- Fine-tuning pre-projection params only ({args.train_steps} steps) ---")
    
    train_dataset = prepare_training_data(tokenizer, args.seq_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    report_to = "wandb" if args.use_wandb else "none"
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=5,
        max_steps=args.train_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        logging_steps=25,
        save_steps=args.train_steps,
        report_to=report_to,
        run_name="frozen_probe_preprojection" if args.use_wandb else None,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        seed=args.seed,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # --- Post-training evaluation ---
    print("\n--- Post-training evaluation ---")
    
    lambada_trained = evaluate_lambada(model, tokenizer, max_samples=500)
    print(f"LAMBADA accuracy: {lambada_trained['lambada_accuracy']:.4f}")
    
    wikitext_eval = load_dataset(
        "wikitext", "wikitext-103-raw-v1",
        split="validation",
        streaming=True,
    )
    ppl_trained = evaluate_perplexity(
        model, tokenizer,
        wikitext_eval,
        max_samples=300,
        seq_length=args.seq_length,
    )
    print(f"WikiText perplexity: {ppl_trained['perplexity']:.2f}")
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'Baseline':>10} {'Injected':>10} {'Trained':>10}")
    print("-" * 60)
    print(f"{'LAMBADA accuracy':<25} "
          f"{lambada_baseline['lambada_accuracy']:>10.4f} "
          f"{lambada_injected['lambada_accuracy']:>10.4f} "
          f"{lambada_trained['lambada_accuracy']:>10.4f}")
    print(f"{'WikiText perplexity':<25} "
          f"{ppl_baseline['perplexity']:>10.2f} "
          f"{ppl_injected['perplexity']:>10.2f} "
          f"{ppl_trained['perplexity']:>10.2f}")
    print(f"\nPre-projection params: {count_preprojection_params(model):,}")
    print(f"Variant: {args.variant}, Expansion: {args.expansion}, "
          f"Nonlinearity: {args.nonlinearity}")
    
    # Interpretation guide
    print("\n--- Interpretation ---")
    delta_lambada = lambada_trained['lambada_accuracy'] - lambada_baseline['lambada_accuracy']
    delta_ppl = ppl_baseline['perplexity'] - ppl_trained['perplexity']
    
    if delta_lambada > 0.01:
        print(f"LAMBADA improved by {delta_lambada:.4f} — POSITIVE signal.")
        print("The pre-projection is extracting useful features the base model missed.")
    elif delta_lambada > -0.005:
        print(f"LAMBADA roughly unchanged ({delta_lambada:+.4f}) — NEUTRAL.")
        print("Pre-projection isn't hurting, but frozen probe may be too constrained.")
        print("Proceed to Phase 2 (continued pretraining) for a stronger test.")
    else:
        print(f"LAMBADA decreased by {abs(delta_lambada):.4f} — needs investigation.")
        print("Try: different expansion ratio, nonlinearity, or learning rate.")
    
    if delta_ppl > 0:
        print(f"Perplexity improved by {delta_ppl:.2f} — consistent with LAMBADA.")
    
    # Save results
    import json
    results = {
        "args": vars(args),
        "baseline": {
            "lambada": lambada_baseline,
            "perplexity": ppl_baseline,
        },
        "injected": {
            "lambada": lambada_injected,
            "perplexity": ppl_injected,
        },
        "trained": {
            "lambada": lambada_trained,
            "perplexity": ppl_trained,
        },
        "preprojection_params": count_preprojection_params(model),
    }
    
    results_path = f"{args.output_dir}/results.json"
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
