import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base import (
    EventEncoder,
    SequenceEncoder,
    EmbeddingLayer,
    ProjectionHead,
    PredictionHead,
)


class MLPEventEncoder(EventEncoder):
    """A simple MLP-based event encoder."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.output_dim = config["output_dim"]

        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class BaseSequenceModel(nn.Module):
    """Base class for sequence models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.input_dim = config["input_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config.get("num_layers", 1)
        self.dropout = config.get("dropout", 0.1)
        self.bidirectional = config.get("bidirectional", False)

        # Input projection if needed
        if self.input_dim != self.hidden_dim:
            self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        else:
            self.input_projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RNNSequenceModel(BaseSequenceModel):
    """RNN-based sequence model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.rnn = nn.RNN(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        output, _ = self.rnn(x)
        return output


class LSTMSequenceModel(BaseSequenceModel):
    """LSTM-based sequence model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        output, _ = self.lstm(x)
        return output


class GRUSequenceModel(BaseSequenceModel):
    """GRU-based sequence model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        output, _ = self.gru(x)
        return output


class TransformerSequenceModel(BaseSequenceModel):
    """Transformer-based sequence model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_heads = config.get("num_heads", 8)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        return self.transformer(x)


class SSMSequenceModel(BaseSequenceModel):
    """State Space Model (SSM) for sequences."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.state_dim = config.get("state_dim", self.hidden_dim)

        # State transition
        self.A = nn.Parameter(torch.randn(self.state_dim, self.state_dim))
        # Input projection
        self.B = nn.Parameter(torch.randn(self.hidden_dim, self.state_dim))
        # Output projection
        self.C = nn.Parameter(torch.randn(self.state_dim, self.hidden_dim))
        # Initial state
        self.h0 = nn.Parameter(torch.randn(self.state_dim))

        # Optional: Add learnable parameters for SSM variants
        self.use_gate = config.get("use_gate", False)
        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(self.hidden_dim, self.state_dim), nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        batch_size, seq_len, _ = x.shape

        # Initialize states
        h = self.h0.unsqueeze(0).repeat(batch_size, 1)
        outputs = []

        for t in range(seq_len):
            # State transition
            h = torch.matmul(h, self.A)

            # Input processing
            u = torch.matmul(x[:, t], self.B)

            # Optional gating
            if self.use_gate:
                gate = self.gate(x[:, t])
                h = gate * h + (1 - gate) * u
            else:
                h = h + u

            # Output projection
            y = torch.matmul(h, self.C)
            outputs.append(y)

        return torch.stack(outputs, dim=1)


class S4Model(BaseSequenceModel):
    """Diagonal S4 (Structured State Space Sequence) model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.state_dim = config.get("state_dim", self.hidden_dim)
        self.max_sequence_length = config.get("max_sequence_length", 1024)
        self.use_learnable_dt = config.get("use_learnable_dt", True)
        self.use_initial_state = config.get("use_initial_state", True)

        # Initialize diagonal state matrix (A)
        # Using complex eigenvalues for better expressivity
        real = torch.randn(self.state_dim) * 0.1
        imag = torch.randn(self.state_dim) * 0.1
        self.A = nn.Parameter(torch.complex(real, imag))

        # Input projection (B)
        self.B = nn.Parameter(torch.randn(self.hidden_dim, self.state_dim))

        # Output projection (C)
        self.C = nn.Parameter(torch.randn(self.state_dim, self.hidden_dim))

        # Optional learnable time step
        if self.use_learnable_dt:
            self.dt = nn.Parameter(torch.tensor(1.0))
        else:
            self.dt = 1.0

        # Optional initial state
        if self.use_initial_state:
            self.h0 = nn.Parameter(torch.randn(self.state_dim))

        # Pre-compute powers of A for efficient computation
        self.register_buffer("A_powers", self._compute_A_powers())

    def _compute_A_powers(self) -> torch.Tensor:
        """Pre-compute powers of A for efficient computation."""
        powers = []
        A = self.A
        for i in range(self.max_sequence_length):
            powers.append(A**i)
        return torch.stack(powers)

    def _compute_kernel(self, length: int) -> torch.Tensor:
        """Compute the S4 kernel for the given sequence length."""
        # Compute the kernel using the pre-computed powers of A
        kernel = torch.matmul(self.A_powers[:length], self.B)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using the diagonal S4 model."""
        x = self.input_projection(x)
        batch_size, seq_len, _ = x.shape

        # Compute the S4 kernel
        kernel = self._compute_kernel(seq_len)

        # Initialize output sequence
        outputs = []

        # Process each time step
        for t in range(seq_len):
            # Compute the current state using the kernel
            if t == 0 and self.use_initial_state:
                h = self.h0.unsqueeze(0).repeat(batch_size, 1)
            else:
                h = torch.zeros(batch_size, self.state_dim, device=x.device)

            # Apply the kernel to the input sequence up to the current time step
            for i in range(t + 1):
                h = h + kernel[i] * x[:, t - i]

            # Apply the output projection
            y = torch.matmul(h, self.C)
            outputs.append(y)

        return torch.stack(outputs, dim=1)

    def _compute_impulse_response(self, length: int) -> torch.Tensor:
        """Compute the impulse response of the system."""
        # This is useful for analysis and debugging
        impulse = torch.zeros(length, self.state_dim, device=self.A.device)
        impulse[0] = 1.0

        response = []
        h = torch.zeros(self.state_dim, device=self.A.device)

        for t in range(length):
            h = self.A * h + self.B * impulse[t]
            y = torch.matmul(h, self.C)
            response.append(y)

        return torch.stack(response)


class FlexibleSequenceEncoder(SequenceEncoder):
    """A flexible sequence encoder that can use any sequence model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = config["model_type"]
        self.output_dim = config.get("output_dim", config["hidden_dim"])

        # Initialize the appropriate sequence model
        if self.model_type == "rnn":
            self.model = RNNSequenceModel(config)
        elif self.model_type == "lstm":
            self.model = LSTMSequenceModel(config)
        elif self.model_type == "gru":
            self.model = GRUSequenceModel(config)
        elif self.model_type == "transformer":
            self.model = TransformerSequenceModel(config)
        elif self.model_type == "ssm":
            self.model = SSMSequenceModel(config)
        elif self.model_type == "s4":
            self.model = S4Model(config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Optional output projection
        if self.output_dim != config["hidden_dim"]:
            self.output_projection = nn.Linear(config["hidden_dim"], self.output_dim)
        else:
            self.output_projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return self.output_projection(x)


class CategoricalEmbedding(EmbeddingLayer):
    """Embedding layer for categorical variables."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embedding_dims = config[
            "embedding_dims"
        ]  # List of (num_categories, embedding_dim)
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories, embedding_dim)
                for num_categories, embedding_dim in self.embedding_dims
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_categorical_features)
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            embedded.append(embedding(x[:, i]))
        return torch.cat(embedded, dim=1)


class MLPProjectionHead(ProjectionHead):
    """A simple MLP projection head."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.output_dim = config["output_dim"]

        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class ClassificationHead(PredictionHead):
    """A classification head."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.num_classes = config["num_classes"]
        self.dropout = config.get("dropout", 0.1)

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout), nn.Linear(self.input_dim, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class AutoEncoderEventEncoder(EventEncoder):
    """An autoencoder-based event encoder that learns through reconstruction."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.output_dim = config["output_dim"]
        self.dropout = config.get("dropout", 0.1)

        # Encoder layers
        encoder_layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim
        
        # Final encoder layer
        encoder_layers.append(nn.Linear(prev_dim, self.output_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers (mirror of encoder)
        decoder_layers = []
        prev_dim = self.output_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim
        
        # Final decoder layer
        decoder_layers.append(nn.Linear(prev_dim, self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent representation."""
        return self.decoder(z)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss."""
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return F.mse_loss(x_recon, x)


class ContrastiveEventEncoder(EventEncoder):
    """An event encoder designed for contrastive learning."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.output_dim = config["output_dim"]
        self.dropout = config.get("dropout", 0.1)
        self.temperature = config.get("temperature", 0.07)

        # Encoder layers
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim
        
        # Final encoder layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.encoder = nn.Sequential(*layers)

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder."""
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)  # L2 normalize for contrastive learning

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between two views of the same input."""
        # Compute similarity matrix
        similarity = torch.matmul(z1, z2.t()) / self.temperature
        
        # Labels are the diagonal elements (positive pairs)
        labels = torch.arange(z1.size(0), device=z1.device)
        
        # Compute loss for both directions
        loss_1 = F.cross_entropy(similarity, labels)
        loss_2 = F.cross_entropy(similarity.t(), labels)
        
        return (loss_1 + loss_2) / 2


class InputCorruption(nn.Module):
    """Base class for input corruption strategies."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.corruption_rate = config.get("corruption_rate", 0.15)
        self.mask_value = config.get("mask_value", 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RandomMasking(InputCorruption):
    """Random feature masking (inspired by SCARF)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mask_distribution = config.get("mask_distribution", "uniform")
        self.feature_importance = config.get("feature_importance", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_features = x.shape
        device = x.device

        if self.mask_distribution == "uniform":
            mask = torch.rand(batch_size, n_features, device=device) > self.corruption_rate
        elif self.mask_distribution == "importance":
            if self.feature_importance is None:
                raise ValueError("Feature importance must be provided for importance-based masking")
            probs = torch.tensor(self.feature_importance, device=device)
            probs = probs / probs.sum()
            mask = torch.rand(batch_size, n_features, device=device) > (probs * self.corruption_rate)
        else:
            raise ValueError(f"Unknown mask distribution: {self.mask_distribution}")

        return x * mask + self.mask_value * (~mask)


class GaussianNoise(InputCorruption):
    """Additive Gaussian noise corruption."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.noise_scale = config.get("noise_scale", 0.1)
        self.feature_wise = config.get("feature_wise", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_wise:
            # Different noise scale per feature
            noise = torch.randn_like(x) * self.noise_scale
        else:
            # Same noise scale for all features
            noise = torch.randn_like(x) * self.noise_scale
        return x + noise


class SwappingCorruption(InputCorruption):
    """Feature swapping corruption (inspired by ReConTab)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.swap_prob = config.get("swap_prob", 0.1)
        self.feature_groups = config.get("feature_groups", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_features = x.shape
        device = x.device

        if self.feature_groups is None:
            # Random swapping between any features
            swap_mask = torch.rand(batch_size, n_features, device=device) < self.swap_prob
            for i in range(batch_size):
                if swap_mask[i].any():
                    swap_indices = torch.randperm(n_features, device=device)
                    x[i, swap_mask[i]] = x[i, swap_indices[swap_mask[i]]]
        else:
            # Swapping within feature groups
            for group in self.feature_groups:
                group_mask = torch.zeros(n_features, dtype=torch.bool, device=device)
                group_mask[group] = True
                swap_mask = (torch.rand(batch_size, device=device) < self.swap_prob).unsqueeze(1)
                for i in range(batch_size):
                    if swap_mask[i].any():
                        group_indices = torch.nonzero(group_mask).squeeze()
                        swap_indices = group_indices[torch.randperm(len(group_indices), device=device)]
                        x[i, group_mask] = x[i, swap_indices]

        return x


class VIMECorruption(InputCorruption):
    """VIME-style corruption with both masking and swapping."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mask_prob = config.get("mask_prob", 0.15)
        self.swap_prob = config.get("swap_prob", 0.1)
        self.noise_scale = config.get("noise_scale", 0.1)
        self.feature_importance = config.get("feature_importance", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_features = x.shape
        device = x.device

        # 1. Random masking
        if self.feature_importance is not None:
            probs = torch.tensor(self.feature_importance, device=device)
            probs = probs / probs.sum()
            mask = torch.rand(batch_size, n_features, device=device) > (probs * self.mask_prob)
        else:
            mask = torch.rand(batch_size, n_features, device=device) > self.mask_prob

        # 2. Feature swapping
        swap_mask = torch.rand(batch_size, n_features, device=device) < self.swap_prob
        for i in range(batch_size):
            if swap_mask[i].any():
                swap_indices = torch.randperm(n_features, device=device)
                x[i, swap_mask[i]] = x[i, swap_indices[swap_mask[i]]]

        # 3. Add noise to masked values
        noise = torch.randn_like(x) * self.noise_scale
        x = x * mask + (self.mask_value + noise) * (~mask)

        return x


class CorruptionPipeline(InputCorruption):
    """Pipeline of multiple corruption strategies."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.corruptions = nn.ModuleList()
        
        # Add corruption strategies based on config
        if config.get("use_masking", True):
            self.corruptions.append(RandomMasking(config))
        if config.get("use_noise", True):
            self.corruptions.append(GaussianNoise(config))
        if config.get("use_swapping", True):
            self.corruptions.append(SwappingCorruption(config))
        if config.get("use_vime", False):
            self.corruptions.append(VIMECorruption(config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for corruption in self.corruptions:
            x = corruption(x)
        return x


class TransformerSequenceEncoder(SequenceEncoder):
    """Transformer-based sequence encoder."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config.get("num_layers", 2)
        self.num_heads = config.get("num_heads", 4)
        self.dropout = config.get("dropout", 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Input projection if needed
        if self.input_dim != self.hidden_dim:
            self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        else:
            self.input_projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        return self.transformer(x)
