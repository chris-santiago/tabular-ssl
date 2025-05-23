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


class RNNSequenceModel(nn.Module):
    """Basic RNN sequence model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config.get("num_layers", 1)
        self.dropout = config.get("dropout", 0.0)
        
        self.rnn = nn.RNN(
            input_size=config["input_dim"],
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)[0]


class LSTMSequenceModel(nn.Module):
    """LSTM sequence model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config.get("num_layers", 1)
        self.dropout = config.get("dropout", 0.0)
        
        self.lstm = nn.LSTM(
            input_size=config["input_dim"],
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lstm(x)[0]


class GRUSequenceModel(nn.Module):
    """GRU sequence model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config.get("num_layers", 1)
        self.dropout = config.get("dropout", 0.0)
        
        self.gru = nn.GRU(
            input_size=config["input_dim"],
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]


class TransformerSequenceModel(nn.Module):
    """Transformer sequence model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config.get("num_layers", 1)
        self.num_heads = config.get("num_heads", 4)
        self.dropout = config.get("dropout", 0.0)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["input_dim"],
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)


class SSMSequenceModel(nn.Module):
    """State Space Model sequence model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config.get("num_layers", 1)
        self.dropout = config.get("dropout", 0.0)
        
        # Simple implementation using linear layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config["input_dim"] if i == 0 else self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ) for i in range(self.num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class S4Model(nn.Module):
    """S4 sequence model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config.get("num_layers", 1)
        self.dropout = config.get("dropout", 0.0)
        
        # Simple implementation using linear layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config["input_dim"] if i == 0 else self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ) for i in range(self.num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


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
        self.corruptions = {
            "masking": RandomMasking(config) if config.get("use_masking", True) else None,
            "noise": GaussianNoise(config) if config.get("use_noise", True) else None,
            "swapping": SwappingCorruption(config) if config.get("use_swapping", True) else None,
            "vime": VIMECorruption(config) if config.get("use_vime", False) else None,
        }
        # Remove None values
        self.corruptions = {k: v for k, v in self.corruptions.items() if v is not None}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for key in ["masking", "noise", "swapping", "vime"]:
            if key in self.corruptions:
                x = self.corruptions[key](x)
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
