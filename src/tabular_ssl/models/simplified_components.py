"""Simplified component implementations for tabular SSL.

This module provides streamlined component implementations that remove
unnecessary abstraction layers while maintaining all functionality.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple, Union, Any
from .simplified_base import create_mlp


# =============================================================================
# ENCODERS
# =============================================================================

class MLPEncoder(nn.Module):
    """Simple MLP encoder for tabular data.
    
    Combines functionality of MLPEventEncoder and handles both event and 
    sequence encoding depending on configuration.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = create_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.mlp(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequence processing."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 2048,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        
        # Input projection if needed
        self.input_projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_length, hidden_dim) * 0.02)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection if needed
        self.output_projection = nn.Linear(hidden_dim, self.output_dim) if hidden_dim != self.output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Output projection
        x = self.output_projection(x)
        
        return x


class RNNEncoder(nn.Module):
    """RNN/LSTM/GRU encoder for sequence processing."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.1,
        bidirectional: bool = False,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        # Calculate actual output dimension
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_dim = output_dim or rnn_output_dim
        
        # RNN layer
        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout, bidirectional=bidirectional
            )
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout, bidirectional=bidirectional
            )
        else:
            self.rnn = nn.RNN(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout, bidirectional=bidirectional
            )
        
        # Output projection if needed
        self.output_projection = nn.Linear(rnn_output_dim, self.output_dim) if rnn_output_dim != self.output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RNN."""
        output, _ = self.rnn(x)
        return self.output_projection(output)


# =============================================================================
# EMBEDDINGS & HEADS  
# =============================================================================

class TabularEmbedding(nn.Module):
    """Embedding layer for categorical features."""

    def __init__(self, vocab_sizes: Dict[str, int], embedding_dims: Dict[str, int]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, embedding_dims[name])
            for name, vocab_size in vocab_sizes.items()
        })

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through embeddings."""
        embedded = []
        for name, embedding_layer in self.embeddings.items():
            if name in x:
                embedded.append(embedding_layer(x[name]))
        
        if embedded:
            return torch.cat(embedded, dim=-1)
        else:
            # Return empty tensor if no categorical features
            return torch.empty(x[list(x.keys())[0]].shape[0], 0, device=next(iter(x.values())).device)


class MLPHead(nn.Module):
    """Generic MLP head for both projection and prediction tasks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False,
        final_activation: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_dims is None:
            # Simple linear layer
            self.mlp = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer MLP
            self.mlp = create_mlp(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                dropout=dropout,
                activation=activation,
                use_batch_norm=use_batch_norm,
                final_activation=final_activation,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP head."""
        return self.mlp(x)


# =============================================================================
# CORRUPTION STRATEGIES
# =============================================================================

class BaseCorruption(nn.Module):
    """Base class for corruption strategies."""

    def __init__(self, corruption_rate: float = 0.15, **kwargs: Any) -> None:
        super().__init__()
        self.corruption_rate = corruption_rate

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply corruption and return corrupted data with metadata."""
        raise NotImplementedError


class VIMECorruption(BaseCorruption):
    """VIME corruption strategy with masking and value imputation."""

    def __init__(
        self,
        corruption_rate: float = 0.3,
        categorical_indices: Optional[List[int]] = None,
        numerical_indices: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__(corruption_rate, **kwargs)
        self.categorical_indices = categorical_indices or []
        self.numerical_indices = numerical_indices or []

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply VIME corruption."""
        batch_size, seq_len, feature_dim = x.shape
        
        # Create random mask
        mask = torch.rand(batch_size, seq_len, feature_dim, device=x.device) < self.corruption_rate
        
        # Create corrupted version
        corrupted = x.clone()
        
        # For categorical features, replace with random values from the same feature
        for idx in self.categorical_indices:
            if idx < feature_dim:
                cat_mask = mask[:, :, idx]
                shuffled_idx = torch.randperm(batch_size, device=x.device)
                shuffled_values = corrupted[shuffled_idx, :, idx]
                corrupted[:, :, idx] = torch.where(
                    cat_mask,
                    shuffled_values,
                    corrupted[:, :, idx]
                )
        
        # For numerical features, replace with random values from normal distribution
        for idx in self.numerical_indices:
            if idx < feature_dim:
                num_mask = mask[:, :, idx]
                noise = torch.randn_like(corrupted[:, :, idx]) * corrupted[:, :, idx].std()
                corrupted[:, :, idx] = torch.where(num_mask, noise, corrupted[:, :, idx])
        
        return {
            'corrupted': corrupted,
            'mask': mask,
            'original': x
        }


class SCARFCorruption(BaseCorruption):
    """SCARF corruption strategy for contrastive learning."""

    def __init__(
        self,
        corruption_rate: float = 0.6,
        corruption_strategy: str = "random_swap",
        **kwargs
    ):
        super().__init__(corruption_rate, **kwargs)
        self.corruption_strategy = corruption_strategy

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply SCARF corruption."""
        batch_size, seq_len, feature_dim = x.shape
        
        # Create mask for corruption
        mask = torch.rand(batch_size, seq_len, feature_dim, device=x.device) < self.corruption_rate
        
        corrupted = x.clone()
        
        if self.corruption_strategy == "random_swap":
            # Random feature swapping within batch
            for i in range(feature_dim):
                feature_mask = mask[:, :, i]
                shuffled_indices = torch.randperm(batch_size, device=x.device)
                corrupted[:, :, i] = torch.where(
                    feature_mask,
                    corrupted[shuffled_indices, :, i],
                    corrupted[:, :, i]
                )
        
        # Create positive pair (different corruption of same input)
        positive_mask = torch.rand(batch_size, seq_len, feature_dim, device=x.device) < self.corruption_rate
        positive = x.clone()
        
        for i in range(feature_dim):
            feature_mask = positive_mask[:, :, i]
            shuffled_indices = torch.randperm(batch_size, device=x.device)
            positive[:, :, i] = torch.where(
                feature_mask,
                positive[shuffled_indices, :, i],
                positive[:, :, i]
            )
        
        return {
            'corrupted': corrupted,
            'positive': positive,
            'mask': mask,
            'original': x
        }


class ReConTabCorruption(BaseCorruption):
    """ReConTab corruption strategy with multiple reconstruction tasks."""

    def __init__(
        self,
        corruption_rate: float = 0.15,
        corruption_types: List[str] = ["masking", "noise", "swapping"],
        noise_std: float = 0.1,
        **kwargs
    ):
        super().__init__(corruption_rate, **kwargs)
        self.corruption_types = corruption_types
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply ReConTab corruption."""
        batch_size, seq_len, feature_dim = x.shape
        
        corrupted = x.clone()
        corruption_info = torch.zeros_like(x)  # Track type of corruption applied
        
        # Apply different types of corruption
        for corruption_type in self.corruption_types:
            mask = torch.rand(batch_size, seq_len, feature_dim, device=x.device) < (self.corruption_rate / len(self.corruption_types))
            
            if corruption_type == "masking":
                corrupted = torch.where(mask, torch.zeros_like(corrupted), corrupted)
                corruption_info = torch.where(mask, torch.ones_like(corruption_info), corruption_info)
            
            elif corruption_type == "noise":
                noise = torch.randn_like(corrupted) * self.noise_std
                corrupted = torch.where(mask, corrupted + noise, corrupted)
                corruption_info = torch.where(mask, torch.full_like(corruption_info, 2), corruption_info)
            
            elif corruption_type == "swapping":
                shuffled_indices = torch.randperm(batch_size, device=x.device)
                swapped = corrupted[shuffled_indices]
                corrupted = torch.where(mask, swapped, corrupted)
                corruption_info = torch.where(mask, torch.full_like(corruption_info, 3), corruption_info)
        
        return {
            'corrupted': corrupted,
            'corruption_info': corruption_info,
            'original': x
        }


# =============================================================================
# UNIFIED COMPONENT FACTORY
# =============================================================================

def create_encoder(encoder_type: str, **kwargs) -> nn.Module:
    """Factory function to create encoders."""
    encoder_map = {
        'mlp': MLPEncoder,
        'transformer': TransformerEncoder,
        'rnn': RNNEncoder,
        'lstm': lambda **k: RNNEncoder(rnn_type='lstm', **k),
        'gru': lambda **k: RNNEncoder(rnn_type='gru', **k),
    }
    
    if encoder_type not in encoder_map:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    return encoder_map[encoder_type](**kwargs)


def create_corruption(corruption_type: str, **kwargs) -> BaseCorruption:
    """Factory function to create corruption strategies."""
    corruption_map = {
        'vime': VIMECorruption,
        'scarf': SCARFCorruption,
        'recontab': ReConTabCorruption,
    }
    
    if corruption_type not in corruption_map:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
    
    return corruption_map[corruption_type](**kwargs) 