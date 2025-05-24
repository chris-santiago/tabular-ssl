import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from .base import EventEncoder, SequenceEncoder, create_mlp


class MLPEventEncoder(EventEncoder):
    """Simple MLP-based event encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False
    ):
        super().__init__()
        self.mlp = create_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AutoEncoderEventEncoder(EventEncoder):
    """Autoencoder-based event encoder for self-supervised learning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.1,
        use_batch_norm: bool = False
    ):
        super().__init__()
        
        # Encoder
        self.encoder = create_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
        
        # Decoder (reverse architecture)
        decoder_dims = hidden_dims[::-1]  # Reverse hidden dims
        self.decoder = create_mlp(
            input_dim=latent_dim,
            hidden_dims=decoder_dims,
            output_dim=input_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns the latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to input space."""
        return self.decoder(z)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss."""
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return F.mse_loss(x_recon, x)


class ContrastiveEventEncoder(EventEncoder):
    """Contrastive learning-based event encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        temperature: float = 0.1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = create_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns normalized embeddings."""
        z = self.encoder(x)
        return F.normalize(z, dim=-1)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between two views."""
        # Compute similarity matrix
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # Labels for positive pairs (diagonal elements)
        batch_size = z1.size(0)
        labels = torch.arange(batch_size, device=z1.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class TransformerSequenceEncoder(SequenceEncoder):
    """Transformer-based sequence encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 2048
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Input projection if needed
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(max_seq_length, hidden_dim) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer."""
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        if seq_len <= self.max_seq_length:
            x = x + self.pos_encoding[:seq_len]
        
        # Apply dropout and layer norm
        x = self.dropout(x)
        x = self.layer_norm(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        return x


class RNNSequenceEncoder(SequenceEncoder):
    """RNN-based sequence encoder (LSTM/GRU)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        rnn_type: str = "lstm",  # "lstm", "gru", "rnn"
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Select RNN type
        rnn_classes = {
            "lstm": nn.LSTM,
            "gru": nn.GRU, 
            "rnn": nn.RNN
        }
        
        if rnn_type not in rnn_classes:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
            
        rnn_class = rnn_classes[rnn_type]
        
        self.rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RNN."""
        output, _ = self.rnn(x)
        return output


# S4 Utility Functions
def _initialize_s4_parameters(
    d_state: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize complex state space parameters for S4."""
    A = torch.complex(-torch.rand(d_state) * 0.1, torch.randn(d_state) * 0.1)
    B = torch.randn(d_state, dtype=torch.cfloat) * 0.02
    C = torch.randn(d_state, dtype=torch.cfloat) * 0.02
    D = torch.randn(d_state, dtype=torch.cfloat) * 0.02
    return A, B, C, D


def _compute_s4_kernel(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, L: int
) -> torch.Tensor:
    """Compute the S4 kernel for sequence length L."""
    A_powers = torch.pow(A.unsqueeze(0), torch.arange(L, device=A.device))
    kernel = torch.einsum("l,dn->ldn", A_powers, B)
    kernel = torch.einsum("ldn,md->lmn", kernel, C)
    kernel = kernel + D.unsqueeze(0)
    return kernel


def _apply_bidirectional_s4_conv(
    x: torch.Tensor, kernel: torch.Tensor, d_model: int
) -> torch.Tensor:
    """Apply bidirectional convolution for S4."""
    L = x.size(1)
    x_padded = F.pad(x, (0, 0, L - 1, L - 1))
    kernel_rev = kernel.flip(0)
    y_forward = F.conv1d(
        x_padded.transpose(1, 2), kernel.transpose(1, 2), groups=d_model
    ).transpose(1, 2)
    y_backward = F.conv1d(
        x_padded.transpose(1, 2), kernel_rev.transpose(1, 2), groups=d_model
    ).transpose(1, 2)
    return y_forward + y_backward


class _S4Block(nn.Module):
    """S4 block with diagonal state space matrices."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.1,
        bidirectional: bool = False,
        max_sequence_length: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional
        self.max_sequence_length = max_sequence_length

        # State space parameters
        self.A, self.B, self.C, self.D = _initialize_s4_parameters(self.d_state)

        # Input projection
        self.input_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the S4 block."""
        # Input projection and normalization
        x = self.input_proj(x)
        x = self.norm(x)

        # Get sequence length
        L = x.size(1)

        # Compute kernel
        kernel = _compute_s4_kernel(self.A, self.B, self.C, self.D, L)

        # Apply convolution
        if self.bidirectional:
            y = _apply_bidirectional_s4_conv(x, kernel, self.d_model)
        else:
            # Pad for causal
            x_padded = F.pad(x, (0, 0, L - 1, 0))
            # Compute convolution
            y = F.conv1d(
                x_padded.transpose(1, 2), kernel.transpose(1, 2), groups=self.d_model
            ).transpose(1, 2)

        # Apply dropout and residual connection
        y = self.dropout(y)
        y = y + x

        return y


class S4SequenceEncoder(SequenceEncoder):
    """S4 (Structured State Space) sequence encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = False,
        max_sequence_length: int = 2048,
        output_dim: Optional[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.max_sequence_length = max_sequence_length

        # Input projection to hidden dimension
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # S4 blocks
        self.blocks = nn.ModuleList([
            _S4Block(
                d_model=hidden_dim,
                d_state=hidden_dim,
                dropout=dropout,
                bidirectional=bidirectional,
                max_sequence_length=max_sequence_length
            )
            for _ in range(num_layers)
        ])

        # Output projection if specified
        if output_dim is not None and output_dim != hidden_dim:
            self.output_proj = nn.Linear(hidden_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.output_proj = nn.Identity()
            self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through S4 layers."""
        # Input projection
        x = self.input_proj(x)
        
        # Apply S4 blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        x = self.output_proj(x)

        return x


class CategoricalEmbedding(nn.Module):
    """Embedding layer for categorical variables with flexible dimensions."""

    def __init__(self, vocab_sizes: Dict[str, int], embedding_dims: Dict[str, int]):
        super().__init__()
        
        self.vocab_sizes = vocab_sizes
        self.embedding_dims = embedding_dims
        
        # Create embedding for each categorical column
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, embedding_dims[col])
            for col, vocab_size in vocab_sizes.items()
        })
        
        self.output_dim = sum(embedding_dims.values())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, sequence_length, num_categorical_features)
        Returns:
            Embedded features of shape (batch_size, sequence_length, total_embedding_dim)
        """
        embedded_features = []
        
        for i, (col, embedding) in enumerate(self.embeddings.items()):
            # Get indices for this column
            indices = x[:, :, i].long()
            # Embed and collect
            embedded = embedding(indices)
            embedded_features.append(embedded)
        
        # Concatenate all embeddings
        return torch.cat(embedded_features, dim=-1)


class MLPProjectionHead(nn.Module):
    """MLP-based projection head for downstream tasks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False
    ):
        super().__init__()
        
        self.projection = create_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class ClassificationHead(nn.Module):
    """Classification head for supervised learning."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = False
    ):
        super().__init__()
        
        if hidden_dims is None:
            # Simple linear classifier
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, num_classes)
            )
        else:
            # MLP classifier
            self.classifier = create_mlp(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=num_classes,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# Data corruption strategies for self-supervised learning
class RandomMasking(nn.Module):
    """Random masking corruption strategy."""

    def __init__(self, corruption_rate: float = 0.15):
        super().__init__()
        self.corruption_rate = corruption_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random masking to input tensor."""
        mask = torch.rand_like(x) > self.corruption_rate
        return x * mask.float()


class GaussianNoise(nn.Module):
    """Gaussian noise corruption strategy."""

    def __init__(self, noise_std: float = 0.1):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to input tensor."""
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x


class SwappingCorruption(nn.Module):
    """Feature swapping corruption strategy."""

    def __init__(self, swap_prob: float = 0.15):
        super().__init__()
        self.swap_prob = swap_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly swap features between samples."""
        if not self.training:
            return x
            
        batch_size, seq_len, num_features = x.shape
        x_corrupted = x.clone()
        
        # For each feature, randomly swap between samples
        for feat_idx in range(num_features):
            if torch.rand(1).item() < self.swap_prob:
                # Random permutation of batch indices
                perm_indices = torch.randperm(batch_size, device=x.device)
                x_corrupted[:, :, feat_idx] = x[perm_indices, :, feat_idx]
        
        return x_corrupted
