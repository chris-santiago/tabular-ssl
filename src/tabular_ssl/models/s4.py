import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .base import SequenceEncoder, ComponentConfig, ComponentRegistry
from pydantic import Field

class S4Config(ComponentConfig):
    """Configuration for S4 model."""
    
    d_model: int = Field(..., description="Model dimension")
    d_state: int = Field(..., description="State dimension")
    dropout: float = Field(0.1, description="Dropout rate")
    bidirectional: bool = Field(False, description="Whether to use bidirectional processing")
    max_sequence_length: int = Field(2048, description="Maximum sequence length")

class S4Block(nn.Module):
    """S4 block with diagonal state space matrices."""

    def __init__(self, config: S4Config):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.bidirectional = config.bidirectional
        self.max_sequence_length = config.max_sequence_length

        # State space parameters
        self.A = nn.Parameter(torch.randn(d_state, dtype=torch.cfloat) * 0.02)
        self.B = nn.Parameter(torch.randn(d_state, dtype=torch.cfloat) * 0.02)
        self.C = nn.Parameter(torch.randn(d_model, d_state, dtype=torch.cfloat) * 0.02)
        self.D = nn.Parameter(torch.randn(d_model, dtype=torch.cfloat) * 0.02)

        # Input projection
        self.input_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(d_model)

        # Initialize state space parameters
        self._init_state_space()

    def _init_state_space(self):
        """Initialize state space parameters for stability."""
        # Initialize A with negative real parts for stability
        real = -torch.rand(self.d_state) * 0.1
        imag = torch.randn(self.d_state) * 0.1
        self.A.data = torch.complex(real, imag)

        # Initialize B and C with small values
        self.B.data *= 0.02
        self.C.data *= 0.02

    def _compute_kernel(self, L: int) -> torch.Tensor:
        """Compute the S4 kernel for sequence length L."""
        # Compute powers of A
        A_powers = torch.pow(self.A.unsqueeze(0), torch.arange(L, device=self.A.device))
        
        # Compute kernel
        kernel = torch.einsum("l,dn->ldn", A_powers, self.B)
        kernel = torch.einsum("ldn,md->lmn", kernel, self.C)
        
        # Add direct term
        kernel = kernel + self.D.unsqueeze(0)
        
        return kernel

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
        kernel = self._compute_kernel(L)
        
        # Apply convolution
        if self.bidirectional:
            # Pad for bidirectional
            x_padded = F.pad(x, (0, 0, L-1, L-1))
            # Compute both directions
            kernel_rev = kernel.flip(0)
            y_forward = F.conv1d(
                x_padded.transpose(1, 2),
                kernel.transpose(1, 2),
                groups=self.d_model
            ).transpose(1, 2)
            y_backward = F.conv1d(
                x_padded.transpose(1, 2),
                kernel_rev.transpose(1, 2),
                groups=self.d_model
            ).transpose(1, 2)
            # Combine directions
            y = y_forward + y_backward
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
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.num_layers = config.get("num_layers", 1)
        self.dropout = config.dropout
        self.bidirectional = config.bidirectional
        self.max_sequence_length = config.max_sequence_length

        # Input projection if needed
        if self.d_model != self.d_state:
            self.input_proj = nn.Linear(self.d_model, self.d_state)
        else:
            self.input_proj = nn.Identity()

        # S4 blocks
        self.blocks = nn.ModuleList([
            S4Block(config)
            for _ in range(self.num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(self.d_state, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the S4 model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, d_model)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Apply S4 blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x