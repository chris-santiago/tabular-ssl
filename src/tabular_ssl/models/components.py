import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from .base import EventEncoder, SequenceEncoder, EmbeddingLayer, ProjectionHead, PredictionHead, create_mlp


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


class CategoricalEmbedding(EmbeddingLayer):
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


class MLPProjectionHead(ProjectionHead):
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


class ClassificationHead(PredictionHead):
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


class VIMECorruption(nn.Module):
    """
    VIME (Value Imputation and Mask Estimation) corruption strategy.
    
    Based on "VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain"
    https://arxiv.org/abs/2006.06775
    
    Creates corrupted views by:
    1. Generating binary mask vectors to indicate which features to corrupt
    2. For categorical features: replace with random category from vocabulary
    3. For numerical features: replace with random value from feature distribution
    4. Returns both corrupted data and mask for mask estimation pretext task
    """

    def __init__(
        self,
        corruption_rate: float = 0.3,
        categorical_indices: Optional[List[int]] = None,
        numerical_indices: Optional[List[int]] = None,
        categorical_vocab_sizes: Optional[Dict[int, int]] = None,
        numerical_distributions: Optional[Dict[int, Tuple[float, float]]] = None
    ):
        super().__init__()
        self.corruption_rate = corruption_rate
        self.categorical_indices = categorical_indices or []
        self.numerical_indices = numerical_indices or []
        self.categorical_vocab_sizes = categorical_vocab_sizes or {}
        self.numerical_distributions = numerical_distributions or {}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply VIME corruption.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            
        Returns:
            Tuple of (corrupted_x, mask) where:
            - corrupted_x: Corrupted input with same shape as x
            - mask: Binary mask indicating corrupted positions (1=corrupted, 0=original)
        """
        if not self.training:
            return x, torch.zeros_like(x)
            
        batch_size, seq_len, num_features = x.shape
        device = x.device
        
        # Generate corruption mask
        mask = torch.bernoulli(
            torch.full((batch_size, seq_len, num_features), self.corruption_rate, device=device)
        )
        
        x_corrupted = x.clone()
        
        # Corrupt categorical features
        for feat_idx in self.categorical_indices:
            if feat_idx < num_features:
                vocab_size = self.categorical_vocab_sizes.get(feat_idx, 10)  # Default vocab size
                mask_positions = mask[:, :, feat_idx].bool()
                
                if mask_positions.any():
                    # Replace with random categories
                    random_categories = torch.randint(
                        0, vocab_size, mask_positions.sum().shape, device=device
                    )
                    x_corrupted[:, :, feat_idx][mask_positions] = random_categories.float()
        
        # Corrupt numerical features  
        for feat_idx in self.numerical_indices:
            if feat_idx < num_features:
                mask_positions = mask[:, :, feat_idx].bool()
                
                if mask_positions.any():
                    # Get distribution parameters (mean, std)
                    mean, std = self.numerical_distributions.get(feat_idx, (0.0, 1.0))
                    
                    # Replace with random values from normal distribution
                    random_values = torch.normal(
                        mean, std, mask_positions.sum().shape, device=device
                    )
                    x_corrupted[:, :, feat_idx][mask_positions] = random_values
        
        return x_corrupted, mask

    def set_feature_distributions(self, data: torch.Tensor, categorical_indices: List[int], numerical_indices: List[int]):
        """Set feature distributions from training data."""
        self.categorical_indices = categorical_indices
        self.numerical_indices = numerical_indices
        
        # Compute categorical vocabulary sizes
        for feat_idx in categorical_indices:
            if feat_idx < data.shape[-1]:
                unique_values = torch.unique(data[:, :, feat_idx])
                self.categorical_vocab_sizes[feat_idx] = len(unique_values)
        
        # Compute numerical feature distributions
        for feat_idx in numerical_indices:
            if feat_idx < data.shape[-1]:
                feature_data = data[:, :, feat_idx].flatten()
                mean = feature_data.mean().item()
                std = feature_data.std().item()
                self.numerical_distributions[feat_idx] = (mean, std)


class SCARFCorruption(nn.Module):
    """
    SCARF (Self-Supervised Contrastive Learning using Random Feature Corruption) corruption strategy.
    
    Based on "SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption for Representation Learning"
    https://arxiv.org/abs/2106.15147
    
    Creates corrupted views by randomly replacing feature values with values from other samples
    in the batch, specifically designed for contrastive learning on tabular data.
    """

    def __init__(
        self,
        corruption_rate: float = 0.6,
        corruption_strategy: str = "random_swap"  # "random_swap", "marginal_sampling"
    ):
        super().__init__()
        self.corruption_rate = corruption_rate
        self.corruption_strategy = corruption_strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SCARF corruption by randomly replacing features with values from other samples.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            
        Returns:
            Corrupted tensor with same shape as input
        """
        if not self.training:
            return x
            
        batch_size, seq_len, num_features = x.shape
        device = x.device
        
        x_corrupted = x.clone()
        
        # Generate corruption mask for features
        corruption_mask = torch.rand(num_features, device=device) < self.corruption_rate
        
        for feat_idx in range(num_features):
            if corruption_mask[feat_idx]:
                if self.corruption_strategy == "random_swap":
                    # Randomly permute this feature across all samples and sequences
                    feature_values = x[:, :, feat_idx].flatten()
                    perm_indices = torch.randperm(len(feature_values), device=device)
                    shuffled_values = feature_values[perm_indices]
                    x_corrupted[:, :, feat_idx] = shuffled_values.view(batch_size, seq_len)
                    
                elif self.corruption_strategy == "marginal_sampling":
                    # Sample from marginal distribution (uniform sampling from feature values)
                    feature_values = x[:, :, feat_idx].flatten()
                    sample_indices = torch.randint(
                        0, len(feature_values), (batch_size, seq_len), device=device
                    )
                    x_corrupted[:, :, feat_idx] = feature_values[sample_indices]
        
        return x_corrupted

    def create_contrastive_pairs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create two different corrupted views for contrastive learning.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            
        Returns:
            Tuple of (view1, view2) - two differently corrupted versions
        """
        view1 = self.forward(x)
        view2 = self.forward(x)
        return view1, view2


class ReConTabCorruption(nn.Module):
    """
    ReConTab (Reconstruction-based Contrastive Learning for Tabular Data) corruption strategy.
    
    Based on reconstruction-based self-supervised learning approaches for tabular data.
    Combines multiple corruption techniques optimized for reconstruction tasks.
    """

    def __init__(
        self,
        corruption_rate: float = 0.15,
        categorical_indices: Optional[List[int]] = None,
        numerical_indices: Optional[List[int]] = None,
        corruption_types: List[str] = ["masking", "noise", "swapping"],
        masking_strategy: str = "random",  # "random", "column_wise", "block"
        noise_std: float = 0.1,
        swap_probability: float = 0.1
    ):
        super().__init__()
        self.corruption_rate = corruption_rate
        self.categorical_indices = categorical_indices or []
        self.numerical_indices = numerical_indices or []
        self.corruption_types = corruption_types
        self.masking_strategy = masking_strategy
        self.noise_std = noise_std
        self.swap_probability = swap_probability

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply ReConTab corruption with multiple strategies.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            
        Returns:
            Tuple of (corrupted_x, corruption_info) where:
            - corrupted_x: Corrupted input with same shape as x
            - corruption_info: Information about applied corruptions for reconstruction
        """
        if not self.training:
            return x, torch.zeros_like(x)
            
        batch_size, seq_len, num_features = x.shape
        device = x.device
        
        x_corrupted = x.clone()
        corruption_info = torch.zeros_like(x)  # 0=no corruption, 1=masked, 2=noise, 3=swapped
        
        # Apply each corruption type
        for corruption_idx, corruption_type in enumerate(self.corruption_types):
            
            if corruption_type == "masking":
                mask = self._generate_mask(x, strategy=self.masking_strategy)
                x_corrupted = x_corrupted * (1 - mask)  # Zero out masked positions
                corruption_info[mask.bool()] = 1
                
            elif corruption_type == "noise" and self.numerical_indices:
                noise_mask = torch.rand(batch_size, seq_len, num_features, device=device) < self.corruption_rate
                
                # Apply noise only to numerical features
                for feat_idx in self.numerical_indices:
                    if feat_idx < num_features:
                        feature_mask = noise_mask[:, :, feat_idx]
                        noise = torch.randn_like(x_corrupted[:, :, feat_idx]) * self.noise_std
                        x_corrupted[:, :, feat_idx] = torch.where(
                            feature_mask,
                            x_corrupted[:, :, feat_idx] + noise,
                            x_corrupted[:, :, feat_idx]
                        )
                        corruption_info[:, :, feat_idx][feature_mask] = 2
                        
            elif corruption_type == "swapping":
                swap_mask = torch.rand(num_features, device=device) < self.swap_probability
                
                for feat_idx in range(num_features):
                    if swap_mask[feat_idx]:
                        # Randomly permute this feature across batch
                        perm_indices = torch.randperm(batch_size, device=device)
                        x_corrupted[:, :, feat_idx] = x_corrupted[perm_indices, :, feat_idx]
                        corruption_info[:, :, feat_idx] = 3
        
        return x_corrupted, corruption_info

    def _generate_mask(self, x: torch.Tensor, strategy: str = "random") -> torch.Tensor:
        """Generate corruption mask based on strategy."""
        batch_size, seq_len, num_features = x.shape
        device = x.device
        
        if strategy == "random":
            # Random masking
            mask = torch.bernoulli(
                torch.full((batch_size, seq_len, num_features), self.corruption_rate, device=device)
            )
            
        elif strategy == "column_wise":
            # Randomly select columns to mask entirely
            num_cols_to_mask = max(1, int(num_features * self.corruption_rate))
            cols_to_mask = torch.randperm(num_features)[:num_cols_to_mask]
            
            mask = torch.zeros(batch_size, seq_len, num_features, device=device)
            mask[:, :, cols_to_mask] = 1
            
        elif strategy == "block":
            # Block-wise masking (mask contiguous regions)
            mask = torch.zeros(batch_size, seq_len, num_features, device=device)
            
            for batch_idx in range(batch_size):
                # Random block size
                block_size = max(1, int(seq_len * self.corruption_rate))
                start_pos = torch.randint(0, max(1, seq_len - block_size + 1), (1,)).item()
                
                # Random features to apply block mask
                num_feat_mask = max(1, int(num_features * 0.5))
                feat_indices = torch.randperm(num_features)[:num_feat_mask]
                
                mask[batch_idx, start_pos:start_pos + block_size, feat_indices] = 1
        
        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")
            
        return mask

    def reconstruction_targets(self, original: torch.Tensor, corrupted: torch.Tensor, corruption_info: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Create reconstruction targets for different corruption types.
        
        Args:
            original: Original uncorrupted data
            corrupted: Corrupted data  
            corruption_info: Information about applied corruptions
            
        Returns:
            Dictionary with reconstruction targets for each corruption type
        """
        targets = {}
        
        # Masking reconstruction: predict original values at masked positions
        mask_positions = (corruption_info == 1)
        if mask_positions.any():
            targets["masked_values"] = original[mask_positions]
            targets["mask_positions"] = mask_positions
        
        # Noise reconstruction: predict clean values at noisy positions  
        noise_positions = (corruption_info == 2)
        if noise_positions.any():
            targets["denoised_values"] = original[noise_positions]
            targets["noise_positions"] = noise_positions
            
        # Swap reconstruction: predict original values at swapped positions
        swap_positions = (corruption_info == 3)
        if swap_positions.any():
            targets["unswapped_values"] = original[swap_positions]
            targets["swap_positions"] = swap_positions
        
        return targets
