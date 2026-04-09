"""
Pre-Projection with Content Skip

The pre-projection enriches features for Q/K/V (as proven).
Additionally, a small linear "skip projection" takes those same
enriched features and adds them directly to the attention output,
bypassing the attention mechanism entirely.

This gives the model two paths for content information:
  1. Through attention (position-aware) — proven to work
  2. Direct skip (position-agnostic) — new, adds content signal

The skip projection is just one small linear layer per block.
No alpha needed — the pre-projection already has strong gradients,
and the skip projection trains naturally as part of the same module.

Flow:
  RMSNorm(x) → PreProjection → enriched
                                  |
                                  ├→ Q/K/V → RoPE → Attention → attn_out
                                  |
                                  └→ skip_proj → content_skip
                                  
  output = x + attn_out + content_skip
"""

import torch
import torch.nn as nn
from typing import Optional
from preprojection import PreProjection, PreProjectionWithGate


class PreProjectionWithSkip(nn.Module):
    """
    Pre-projection that also produces a content skip signal.
    
    The skip projection is a small linear that maps enriched features
    to a residual signal added after attention. Initialized near-zero
    so the model starts close to baseline behavior.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        expansion: float = 1.25,
        nonlinearity: str = "silu",
        skip_scale: float = 0.01,
    ):
        super().__init__()
        inner_dim = int(hidden_dim * expansion)
        
        # Standard pre-projection (feeds into Q/K/V)
        self.up_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, hidden_dim, bias=False)
        
        nonlinearities = {
            "silu": nn.SiLU(), "gelu": nn.GELU(),
            "relu": nn.ReLU(), "mish": nn.Mish(),
        }
        self.act = nonlinearities.get(nonlinearity, nn.SiLU())
        
        # Skip projection: enriched features → content signal
        # This is the only new component vs standard pre-projection
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Near-identity init for pre-projection
        nn.init.normal_(self.up_proj.weight, std=0.01)
        nn.init.normal_(self.down_proj.weight, std=0.01)
        # Near-zero init for skip — starts with minimal effect
        nn.init.normal_(self.skip_proj.weight, std=skip_scale)
        
        # Store the skip signal for retrieval after attention
        self._skip_signal = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes enriched features for Q/K/V and stores skip signal.
        """
        enriched = x + self.down_proj(self.act(self.up_proj(x)))
        
        # Compute and store skip signal (retrieved after attention)
        self._skip_signal = self.skip_proj(enriched)
        
        return enriched
    
    def get_skip_signal(self) -> Optional[torch.Tensor]:
        """Retrieve the stored skip signal."""
        return self._skip_signal


def inject_preprojection_with_skip(
    model,
    expansion: float = 1.25,
    nonlinearity: str = "silu",
    skip_scale: float = 0.01,
    freeze_base: bool = True,
):
    """
    Inject pre-projection with content skip into a Pythia model.
    
    Two modifications per layer:
    1. PreProjectionWithSkip wraps the QKV linear (as before)
    2. Layer's forward is wrapped to add skip signal after attention
    """
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
    
    hidden_dim = model.config.hidden_size
    new_params = []
    
    for i, layer in enumerate(model.gpt_neox.layers):
        # Create pre-projection with skip
        preproj = PreProjectionWithSkip(
            hidden_dim=hidden_dim,
            expansion=expansion,
            nonlinearity=nonlinearity,
            skip_scale=skip_scale,
        )
        
        device = next(layer.parameters()).device
        dtype = next(layer.parameters()).dtype
        preproj = preproj.to(device=device, dtype=dtype)
        
        # Wrap Q/K/V (same as standard pre-projection)
        original_qkv = layer.attention.query_key_value
        
        class PreProjectedLinear(nn.Module):
            def __init__(self, preproj, original_linear):
                super().__init__()
                self.preproj = preproj
                self.linear = original_linear
            
            def forward(self, x):
                return self.linear(self.preproj(x))
            
            @property
            def weight(self):
                return self.linear.weight
            
            @property
            def bias(self):
                return self.linear.bias
        
        wrapped_qkv = PreProjectedLinear(preproj, original_qkv)
        layer.attention.query_key_value = wrapped_qkv
        
        # Wrap the LAYER's forward to add skip signal after attention
        original_layer_forward = layer.forward
        
        def make_skip_forward(orig_forward, preproj_ref):
            def wrapped_forward(*args, **kwargs):
                # Run normal layer forward (includes our pre-projection in QKV)
                output = orig_forward(*args, **kwargs)
                
                # Retrieve skip signal from pre-projection
                skip = preproj_ref.get_skip_signal()
                
                if skip is not None:
                    # output is typically a tuple (hidden_states, ...)
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        return (hidden_states + skip,) + output[1:]
                    else:
                        return output + skip
                
                return output
            return wrapped_forward
        
        layer.forward = make_skip_forward(original_layer_forward, preproj)
        
        new_params.append(f"layer_{i}")
    
    # Count parameters
    new_param_count = 0
    for layer in model.gpt_neox.layers:
        qkv = layer.attention.query_key_value
        if hasattr(qkv, 'preproj'):
            new_param_count += sum(p.numel() for p in qkv.preproj.parameters())
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # The skip_proj adds hidden_dim * hidden_dim per layer
    skip_params = hidden_dim * hidden_dim * len(model.gpt_neox.layers)
    preproj_only = new_param_count - skip_params
    
    print(f"Pre-projection with skip injected into {len(new_params)} layers")
    print(f"Pre-projection params: {preproj_only:,}")
    print(f"Skip projection params: {skip_params:,}")
    print(f"Total new params: {new_param_count:,}")
    print(f"Total model params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Overhead: {new_param_count/total_params*100:.1f}%")
    
    return model


def count_skip_params(model):
    """Count pre-projection + skip parameters."""
    count = 0
    for layer in model.gpt_neox.layers:
        qkv = layer.attention.query_key_value
        if hasattr(qkv, 'preproj'):
            count += sum(p.numel() for p in qkv.preproj.parameters())
    return count
