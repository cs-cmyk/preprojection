# Pre-Projection: Position-Agnostic Nonlinear Feature Construction for Transformer Attention

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19497474.svg)](https://doi.org/10.5281/zenodo.19497474)

This repository contains the code for the paper:

**"Position-Agnostic Pre-Projection for Transformer Attention: Nonlinear Feature Construction and Content Skip Before Q/K/V"**

by Chirag Shinde

📄 [Paper (PDF)](paper/preprojection_paper.pdf)  | 🔗 [Zenodo](https://zenodo.org/records/19497474)


## Key Idea

Standard transformer attention uses linear Q/K/V projections — they can only access features that are linearly separable in the post-normalization representation. We insert a small nonlinear MLP **before** Q/K/V that constructs richer features in a **position-agnostic** manner (before RoPE is applied).

Additionally, a **content skip** connection routes these enriched features around attention, letting the model learn where content information should bypass positional routing.

```
RMSNorm → Pre-projection → enriched ─┬→ Q/K/V → RoPE → Attention → attn_out
                                      │
                                      └→ skip_proj ──────────────→ content_skip

output = x + attn_out + content_skip
```

## Results

Frozen probe experiments on Pythia (base model completely frozen, only new parameters train):

| Method | Params | LAMBADA | HellaSwag | ARC-Easy | WikiText PPL |
|--------|--------|---------|-----------|----------|--------------|
| **Pythia-160M** | | | | | |
| Baseline | — | 0.128 | 0.326 | 0.358 | 78.4 |
| Pre-proj only | 17.7M | 0.154 | 0.318 | 0.342 | 73.0 |
| LoRA (r=480) | 26.5M | 0.126 | 0.326 | 0.346 | 63.8 |
| Pre-proj + LoRA | 21.2M | 0.162 | 0.324 | 0.352 | 60.8 |
| **Pre-proj + skip** | **24.8M** | **0.180** | **0.338** | 0.348 | **47.9** |
| **Pythia-410M** | | | | | |
| Baseline | — | 0.466 | 0.396 | 0.436 | 29.2 |
| Pre-proj only | 62.9M | 0.470 | 0.392 | 0.452 | 25.1 |
| LoRA (r=640) | 94.4M | 0.488 | 0.400 | 0.446 | 17.7 |
| Pre-proj + LoRA | 72.4M | 0.470 | 0.400 | 0.450 | 19.2 |
| **Pre-proj + skip** | **88.1M** | **0.484** | 0.398 | 0.432 | **17.0** |

### Layer-Wise Content Skip Analysis

The learned skip weights reveal that **later layers activate the content bypass more strongly** — a consistent pattern across model sizes:

```
Pythia-160M:
  Layers 0-7:  skip_norm ≈ 0.13 (minimal bypass)
  Layers 8-11: skip_norm → 0.30 (strong bypass)

Pythia-410M:
  Layers 0-15:  skip_norm ≈ 0.19 (minimal bypass)
  Layers 16-23: skip_norm → 0.28 (growing bypass)
```

Later layers have mature semantic representations where content information benefits from bypassing positional attention.

## Quick Start

### Installation

```bash
pip install torch transformers datasets accelerate evaluate peft
```

### 1. Frozen Probe (validates the concept in ~2 minutes)

```bash
python 01_frozen_probe.py --lr 1e-5 --expansion 1.25 --train_steps 500
```

### 2. Full Benchmarks with LoRA Comparison

```bash
# Pythia-160M
python 07_benchmarks.py \
    --model_name EleutherAI/pythia-160m \
    --method both \
    --lr 1e-5 \
    --train_steps 500

# Pythia-410M
python 07_benchmarks.py \
    --model_name EleutherAI/pythia-410m \
    --method both \
    --lr 1e-5 \
    --train_steps 500 \
    --batch_size 2 --grad_accum 8
```

### 3. Pre-Projection + Content Skip (best results)

```bash
# Pythia-160M
python 09_skip_benchmark.py \
    --model_name EleutherAI/pythia-160m \
    --lr 1e-5 --train_steps 500 --skip_scale 0.0001

# Pythia-410M
python 09_skip_benchmark.py \
    --model_name EleutherAI/pythia-410m \
    --lr 1e-5 --train_steps 500 --skip_scale 0.0001 \
    --batch_size 2 --grad_accum 8
```

## Repository Structure

```
preprojection/
├── preprojection.py          # Core pre-projection module + injection
├── preprojection_skip.py     # Pre-projection with content skip
├── 01_frozen_probe.py        # Quick validation (LAMBADA + perplexity)
├── 07_benchmarks.py          # Full benchmarks + LoRA comparison
├── 09_skip_benchmark.py      # Pre-proj + skip benchmarks
├── requirements.txt
└── paper/
    ├── preprojection_paper.pdf
    └── preprojection_paper.tex
```

## Hardware Requirements

All experiments were run on a single **RTX 4080 12GB**. The frozen probe setting (base model frozen, only new parameters train) makes this very memory-efficient:

- Pythia-160M: ~2GB VRAM
- Pythia-410M: ~4GB VRAM
- Pythia-1B: ~5GB VRAM
- Pythia-1.4B: ~6GB VRAM

## Key Design Decisions

- **Position-agnostic**: No positional encoding in the pre-projection or skip. Position enters only at RoPE on Q/K. This makes the approach modality-independent.
- **Near-identity initialization**: Pre-projection uses residual connection with small init (std=0.01). Skip uses near-zero init (std=0.0001). The model starts identical to baseline.
- **No K/V cache overhead**: Both modifications occur before/after attention, not within it.
- **Conservative learning rate**: 1e-5 works best. Higher rates (1e-3) cause catastrophic degradation in frozen probe setting.

## Citation

```bibtex
@article{shinde2025preprojection,
  title={Position-Agnostic Pre-Projection for Transformer Attention: 
         Nonlinear Feature Construction and Content Skip Before Q/K/V},
  author={Shinde, Chirag},
  year={2025}
}
```

## License

MIT
