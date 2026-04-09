"""
Pre-Projection Module for Transformer Attention Layers

Inserts a learned nonlinear MLP between RMSNorm and Q/K/V projections.
- Position-agnostic: no positional encoding
- Cache-free: no K/V overhead
- Content-focused: purely feature construction
"""

import torch
import torch.nn as nn
from typing import Optional


class PreProjection(nn.Module):
    """
    Small nonlinear MLP inserted before Q/K/V projections.
    
    Projects hidden_dim -> expansion * hidden_dim -> hidden_dim
    with a nonlinearity in between.
    
    This gives the Q/K/V projections access to nonlinear feature
    combinations that a single linear projection cannot extract.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        expansion: float = 2.0,
        nonlinearity: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(hidden_dim * expansion)
        
        self.up_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Choose nonlinearity — keeping this configurable
        nonlinearities = {
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "mish": nn.Mish(),
        }
        self.act = nonlinearities.get(nonlinearity, nn.SiLU())
        
        # Initialize near-identity so the model starts close to
        # the original pretrained behavior. This is critical for
        # the frozen probe test and continued pretraining —
        # we don't want to destroy pretrained representations.
        self._init_near_identity(hidden_dim, inner_dim)
    
    def _init_near_identity(self, hidden_dim, inner_dim):
        """
        Initialize so the pre-projection is close to an identity
        function at the start of training. This preserves the
        pretrained model's behavior initially.
        
        Strategy: small random weights scaled so the output
        magnitude approximately matches the input magnitude.
        """
        nn.init.normal_(self.up_proj.weight, std=0.01)
        nn.init.normal_(self.down_proj.weight, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim) — post-RMSNorm hidden states
        Returns:
            (batch, seq_len, hidden_dim) — enriched features for Q/K/V
        """
        return x + self.dropout(self.down_proj(self.act(self.up_proj(x))))


class PreProjectionWithGate(nn.Module):
    """
    Gated variant — similar to SwiGLU in FFN.
    Uses a gate projection to control information flow.
    Slightly more parameters but potentially more expressive.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        expansion: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(hidden_dim * expansion)
        
        self.up_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        nn.init.normal_(self.up_proj.weight, std=0.01)
        nn.init.normal_(self.gate_proj.weight, std=0.01)
        nn.init.normal_(self.down_proj.weight, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        return x + self.dropout(self.down_proj(gate * up))


def inject_preprojection(
    model,
    expansion: float = 2.0,
    nonlinearity: str = "silu",
    variant: str = "standard",
    dropout: float = 0.0,
    freeze_base: bool = False,
):
    """
    Inject PreProjection modules into a Pythia/GPT-NeoX model.
    
    Modifies the model in-place by wrapping each attention layer's
    forward method to include pre-projection before Q/K/V computation.
    
    Args:
        model: A GPTNeoXForCausalLM model (Pythia)
        expansion: MLP expansion ratio (2.0 = double the hidden dim)
        nonlinearity: Activation function name
        variant: "standard" or "gated"
        dropout: Dropout rate in pre-projection
        freeze_base: If True, freeze all original model parameters
    
    Returns:
        model with pre-projection injected, list of new parameter names
    """
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
    
    hidden_dim = model.config.hidden_size
    new_params = []
    
    PreProjClass = PreProjectionWithGate if variant == "gated" else PreProjection
    
    for i, layer in enumerate(model.gpt_neox.layers):
        # Create pre-projection for this layer
        if variant == "gated":
            preproj = PreProjClass(
                hidden_dim=hidden_dim,
                expansion=expansion,
                dropout=dropout,
            )
        else:
            preproj = PreProjClass(
                hidden_dim=hidden_dim,
                expansion=expansion,
                nonlinearity=nonlinearity,
                dropout=dropout,
            )
        
        # Move to same device/dtype as the layer
        device = next(layer.parameters()).device
        dtype = next(layer.parameters()).dtype
        preproj = preproj.to(device=device, dtype=dtype)
        
        # Wrap the attention forward to include pre-projection
        # NOTE: do NOT also add_module on the layer — that creates
        # duplicate parameter paths and breaks save_pretrained.
        original_attention = layer.attention
        
        # We need to monkey-patch the attention's query_key_value projection
        # to apply pre-projection to its input
        original_qkv = original_attention.query_key_value
        
        class PreProjectedLinear(nn.Module):
            """Wraps the Q/K/V linear layer with a pre-projection."""
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
        
        # Replace the qkv projection with the wrapped version
        wrapped_qkv = PreProjectedLinear(preproj, original_qkv)
        original_attention.query_key_value = wrapped_qkv
        
        param_name = f"layer_{i}_pre_projection"
        new_params.append(param_name)
    
    # Count new parameters
    new_param_count = sum(
        p.numel() for layer in model.gpt_neox.layers
        if hasattr(layer.attention.query_key_value, 'preproj')
        for p in layer.attention.query_key_value.preproj.parameters()
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Pre-projection injected into {len(new_params)} layers")
    print(f"New parameters: {new_param_count:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Pre-projection overhead: {new_param_count/total_params*100:.1f}%")
    
    return model, new_params


def count_preprojection_params(model):
    """Utility to count only pre-projection parameters."""
    count = 0
    for layer in model.gpt_neox.layers:
        qkv = layer.attention.query_key_value
        if hasattr(qkv, 'preproj'):
            count += sum(p.numel() for p in qkv.preproj.parameters())
    return count
