import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .base import SequenceEncoder, ComponentConfig, ComponentRegistry
from pydantic import Field

class S4Config(ComponentConfig):
    """Configuration for S4 model."""
    
    input_dim: int = Field(..., description="Input dimension")
    hidden_dim: int = Field(..., description="Hidden dimension (state dimension)")
    num_layers: int = Field(1, description="Number of S4 blocks")
    dropout: float = Field(0.1, description="Dropout rate")
    bidirectional: bool = Field(False, description="Whether to use bidirectional processing")
    max_sequence_length: int = Field(2048, description="Maximum sequence length")

def initialize_complex_parameters(d_state: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Utility function to initialize complex state space parameters."""
    A = torch.complex(-torch.rand(d_state) * 0.1, torch.randn(d_state) * 0.1)
    B = torch.randn(d_state, dtype=torch.cfloat) * 0.02
    C = torch.randn(d_state, dtype=torch.cfloat) * 0.02
    D = torch.randn(d_state, dtype=torch.cfloat) * 0.02
    return A, B, C, D


def compute_s4_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, L: int) -> torch.Tensor:
    """Utility function to compute the S4 kernel for sequence length L."""
    A_powers = torch.pow(A.unsqueeze(0), torch.arange(L, device=A.device))
    kernel = torch.einsum("l,dn->ldn", A_powers, B)
    kernel = torch.einsum("ldn,md->lmn", kernel, C)
    kernel = kernel + D.unsqueeze(0)
    return kernel


def apply_bidirectional_convolution(x: torch.Tensor, kernel: torch.Tensor, d_model: int) -> torch.Tensor:
    """Utility function to apply bidirectional convolution."""
    L = x.size(1)
    x_padded = F.pad(x, (0, 0, L-1, L-1))
    kernel_rev = kernel.flip(0)
    y_forward = F.conv1d(
        x_padded.transpose(1, 2),
        kernel.transpose(1, 2),
        groups=d_model
    ).transpose(1, 2)
    y_backward = F.conv1d(
        x_padded.transpose(1, 2),
        kernel_rev.transpose(1, 2),
        groups=d_model
    ).transpose(1, 2)
    return y_forward + y_backward


class S4Block(nn.Module):
    """S4 block with diagonal state space matrices."""

    def __init__(self, config: S4Config):
        super().__init__()
        self.d_model = config.input_dim
        self.d_state = config.hidden_dim
        self.bidirectional = config.bidirectional
        self.max_sequence_length = config.max_sequence_length

        # State space parameters
        self.A, self.B, self.C, self.D = initialize_complex_parameters(self.d_state)

        # Input projection
        self.input_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the S4 block.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, d_model)
        """
        # Input projection and normalization
        x = self.input_proj(x)
        x = self.norm(x)
        
        # Get sequence length
        L = x.size(1)
        
        # Compute kernel
        kernel = compute_s4_kernel(self.A, self.B, self.C, self.D, L)
        
        # Apply convolution
        if self.bidirectional:
            y = apply_bidirectional_convolution(x, kernel, self.d_model)
        else:
            # Pad for causal
            x_padded = F.pad(x, (0, 0, L-1, 0))
            # Compute convolution
            y = F.conv1d(
                x_padded.transpose(1, 2),
                kernel.transpose(1, 2),
                groups=self.d_model
            ).transpose(1, 2)
        
        # Apply dropout and residual connection
        y = self.dropout(y)
        y = y + x
        
        return y

@ComponentRegistry.register("s4")
class S4Model(SequenceEncoder):
    """S4 sequence model with multiple S4 blocks."""

    def __init__(self, config: S4Config):
        super().__init__(config)
        self.d_model = config.input_dim
        self.d_state = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.bidirectional = config.bidirectional
        self.max_sequence_length = config.max_sequence_length

        # Input projection if needed
        if hasattr(config, 'output_dim') and config.output_dim != self.d_model:
            self.output_dim = config.output_dim
            self.output_proj = nn.Linear(self.d_model, self.output_dim)
        else:
            self.output_dim = self.d_model
            self.output_proj = nn.Identity()

        # S4 blocks
        self.blocks = nn.ModuleList([
            S4Block(config)
            for _ in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the S4 model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, output_dim)
        """
        # Apply S4 blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x