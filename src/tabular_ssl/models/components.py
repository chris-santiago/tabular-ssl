import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from pydantic import Field
from .base import (
    EventEncoder,
    SequenceEncoder,
    EmbeddingLayer,
    ProjectionHead,
    PredictionHead,
    ComponentConfig,
    ComponentRegistry,
)

class MLPConfig(ComponentConfig):
    """Configuration for MLP-based components."""
    
    input_dim: int = Field(..., description="Input dimension")
    hidden_dims: List[int] = Field(..., description="List of hidden dimensions")
    output_dim: int = Field(..., description="Output dimension")
    dropout: float = Field(0.1, description="Dropout rate")
    use_batch_norm: bool = Field(True, description="Whether to use batch normalization")

@ComponentRegistry.register("mlp_event_encoder")
class MLPEventEncoder(EventEncoder):
    """A simple MLP-based event encoder."""

    def __init__(self, config: MLPConfig):
        super().__init__(config)
        self.input_dim = config.input_dim
        self.hidden_dims = config.hidden_dims
        self.output_dim = config.output_dim

        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim) if config.use_batch_norm else nn.Identity(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class SequenceModelConfig(ComponentConfig):
    """Base configuration for sequence models."""
    
    input_dim: int = Field(..., description="Input dimension")
    hidden_dim: int = Field(..., description="Hidden dimension")
    num_layers: int = Field(1, description="Number of layers")
    dropout: float = Field(0.1, description="Dropout rate")
    bidirectional: bool = Field(False, description="Whether to use bidirectional processing")

@ComponentRegistry.register("rnn")
class RNNSequenceModel(SequenceEncoder):
    """Basic RNN sequence model."""
    
    def __init__(self, config: SequenceModelConfig):
        super().__init__(config)
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        
        self.rnn = nn.RNN(
            input_size=config.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=config.bidirectional
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)[0]

@ComponentRegistry.register("lstm")
class LSTMSequenceModel(SequenceEncoder):
    """LSTM sequence model."""
    
    def __init__(self, config: SequenceModelConfig):
        super().__init__(config)
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=config.bidirectional
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lstm(x)[0]

@ComponentRegistry.register("gru")
class GRUSequenceModel(SequenceEncoder):
    """GRU sequence model."""
    
    def __init__(self, config: SequenceModelConfig):
        super().__init__(config)
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=config.bidirectional
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]

class TransformerConfig(SequenceModelConfig):
    """Configuration for Transformer models."""
    
    num_heads: int = Field(4, description="Number of attention heads")
    dim_feedforward: int = Field(..., description="Dimension of feedforward network")

@ComponentRegistry.register("transformer")
class TransformerSequenceModel(SequenceEncoder):
    """Transformer sequence model."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.input_dim,
            nhead=self.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)

class EmbeddingConfig(ComponentConfig):
    """Configuration for embedding layers."""
    
    embedding_dims: List[tuple[int, int]] = Field(..., description="List of (num_categories, embedding_dim) tuples")
    dropout: float = Field(0.1, description="Dropout rate")

@ComponentRegistry.register("categorical_embedding")
class CategoricalEmbedding(EmbeddingLayer):
    """Embedding layer for categorical variables."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.embedding_dims = config.embedding_dims
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim)
            for num_categories, embedding_dim in self.embedding_dims
        ])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_categorical_features)
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            embedded.append(embedding(x[:, i]))
        return self.dropout(torch.cat(embedded, dim=1))

class ProjectionHeadConfig(ComponentConfig):
    """Configuration for projection heads."""
    
    input_dim: int = Field(..., description="Input dimension")
    hidden_dims: List[int] = Field(..., description="List of hidden dimensions")
    output_dim: int = Field(..., description="Output dimension")
    dropout: float = Field(0.1, description="Dropout rate")
    use_batch_norm: bool = Field(True, description="Whether to use batch normalization")

@ComponentRegistry.register("mlp_projection")
class MLPProjectionHead(ProjectionHead):
    """MLP-based projection head."""

    def __init__(self, config: ProjectionHeadConfig):
        super().__init__(config)
        self.input_dim = config.input_dim
        self.hidden_dims = config.hidden_dims
        self.output_dim = config.output_dim

        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim) if config.use_batch_norm else nn.Identity(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class PredictionHeadConfig(ComponentConfig):
    """Configuration for prediction heads."""
    
    input_dim: int = Field(..., description="Input dimension")
    num_classes: int = Field(..., description="Number of output classes")
    hidden_dims: Optional[List[int]] = Field(None, description="Optional list of hidden dimensions")
    dropout: float = Field(0.1, description="Dropout rate")
    use_batch_norm: bool = Field(True, description="Whether to use batch normalization")

@ComponentRegistry.register("classification")
class ClassificationHead(PredictionHead):
    """Classification head."""

    def __init__(self, config: PredictionHeadConfig):
        super().__init__(config)
        self.input_dim = config.input_dim
        self.num_classes = config.num_classes

        if config.hidden_dims:
            layers = []
            prev_dim = self.input_dim
            for hidden_dim in config.hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim) if config.use_batch_norm else nn.Identity(),
                    nn.Dropout(config.dropout)
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, self.num_classes))
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class AutoEncoderConfig(MLPConfig):
    """Configuration for autoencoder."""
    
    use_reconstruction_loss: bool = Field(True, description="Whether to use reconstruction loss")

@ComponentRegistry.register("autoencoder")
class AutoEncoderEventEncoder(EventEncoder):
    """Autoencoder-based event encoder."""

    def __init__(self, config: AutoEncoderConfig):
        super().__init__(config)
        self.input_dim = config.input_dim
        self.hidden_dims = config.hidden_dims
        self.output_dim = config.output_dim
        self.use_reconstruction_loss = config.use_reconstruction_loss

        # Encoder
        encoder_layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim) if config.use_batch_norm else nn.Identity(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, self.output_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = self.output_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim) if config.use_batch_norm else nn.Identity(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return F.mse_loss(x_recon, x)

class ContrastiveConfig(MLPConfig):
    """Configuration for contrastive learning."""
    
    temperature: float = Field(0.07, description="Temperature for contrastive loss")

@ComponentRegistry.register("contrastive")
class ContrastiveEventEncoder(EventEncoder):
    """Contrastive learning-based event encoder."""

    def __init__(self, config: ContrastiveConfig):
        super().__init__(config)
        self.input_dim = config.input_dim
        self.hidden_dims = config.hidden_dims
        self.output_dim = config.output_dim
        self.temperature = config.temperature

        # Encoder
        encoder_layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim) if config.use_batch_norm else nn.Identity(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, self.output_dim))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(z1, z2.t()) / self.temperature

        # Labels are on the diagonal
        labels = torch.arange(similarity.size(0), device=similarity.device)

        # Compute loss
        loss = F.cross_entropy(similarity, labels)
        return loss

class CorruptionConfig(ComponentConfig):
    """Base configuration for corruption strategies."""
    
    corruption_rate: float = Field(0.15, description="Rate of corruption")

@ComponentRegistry.register("random_masking")
class RandomMasking(nn.Module):
    """Random masking corruption strategy."""

    def __init__(self, config: CorruptionConfig):
        super().__init__()
        self.corruption_rate = config.corruption_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(x) > self.corruption_rate
        return x * mask

@ComponentRegistry.register("gaussian_noise")
class GaussianNoise(nn.Module):
    """Gaussian noise corruption strategy."""

    def __init__(self, config: CorruptionConfig):
        super().__init__()
        self.corruption_rate = config.corruption_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.corruption_rate
        return x + noise

@ComponentRegistry.register("swapping")
class SwappingCorruption(nn.Module):
    """Feature swapping corruption strategy."""

    def __init__(self, config: CorruptionConfig):
        super().__init__()
        self.corruption_rate = config.corruption_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Randomly select features to swap
        mask = torch.rand_like(x) < self.corruption_rate
        # Create a copy of x with shuffled features
        x_shuffled = x[torch.randperm(x.size(0))]
        # Apply mask
        return torch.where(mask, x_shuffled, x)

@ComponentRegistry.register("vime")
class VIMECorruption(nn.Module):
    """VIME-style corruption strategy."""

    def __init__(self, config: CorruptionConfig):
        super().__init__()
        self.corruption_rate = config.corruption_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Random masking
        mask = torch.rand_like(x) > self.corruption_rate
        x_masked = x * mask

        # Add Gaussian noise
        noise = torch.randn_like(x) * self.corruption_rate
        x_noisy = x + noise

        # Combine strategies
        return torch.where(mask, x_masked, x_noisy)

class CorruptionPipelineConfig(ComponentConfig):
    """Configuration for corruption pipeline."""
    
    strategies: List[str] = Field(..., description="List of corruption strategies to apply")
    corruption_rates: List[float] = Field(..., description="Corruption rates for each strategy")

@ComponentRegistry.register("corruption_pipeline")
class CorruptionPipeline(nn.Module):
    """Pipeline of corruption strategies."""

    def __init__(self, config: CorruptionPipelineConfig):
        super().__init__()
        self.strategies = []
        for strategy_name, rate in zip(config.strategies, config.corruption_rates):
            strategy_config = CorruptionConfig(corruption_rate=rate)
            strategy = ComponentRegistry.get(strategy_name)(strategy_config)
            self.strategies.append(strategy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for strategy in self.strategies:
            x = strategy(x)
        return x
