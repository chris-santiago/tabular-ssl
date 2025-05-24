from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


class BaseComponent(nn.Module, ABC):
    """Base class for model components."""

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the component."""
        pass


# Simple base classes for backward compatibility
class EventEncoder(BaseComponent):
    """Base class for event encoders."""
    pass


class SequenceEncoder(BaseComponent):
    """Base class for sequence encoders."""
    pass


class EmbeddingLayer(BaseComponent):
    """Embedding layer for categorical features."""
    
    def __init__(self, vocab_sizes: Dict[str, int], embedding_dims: Dict[str, int]):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embedding_dims = embedding_dims
        
        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, embedding_dims[col])
            for col, vocab_size in vocab_sizes.items()
        })
        
        self.output_dim = sum(embedding_dims.values())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x shape is (batch_size, sequence_length, num_categorical_features)"""
        batch_size, seq_len, num_features = x.shape
        
        embedded_features = []
        for i, (col, embedding) in enumerate(self.embeddings.items()):
            # Get indices for this categorical feature
            indices = x[:, :, i].long()  # (batch_size, sequence_length)
            # Embed: (batch_size, sequence_length, embedding_dim)
            embedded = embedding(indices)
            embedded_features.append(embedded)
        
        # Concatenate all embeddings: (batch_size, sequence_length, total_embedding_dim)
        return torch.cat(embedded_features, dim=-1)


class ProjectionHead(BaseComponent):
    """Simple projection head for downstream tasks."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class PredictionHead(BaseComponent):
    """Prediction head for classification tasks."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def create_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    dropout: float = 0.1,
    activation: str = "relu",
    use_batch_norm: bool = False
) -> nn.Sequential:
    """Utility function to create MLP layers."""
    activation_fn = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU()
    }[activation]
    
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation_fn)
        layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class FeatureEncoder(BaseComponent):
    """Encodes tabular features (categorical + numerical)."""
    
    def __init__(
        self, 
        vocab_sizes: Dict[str, int], 
        embedding_dims: Dict[str, int], 
        numerical_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Categorical embeddings
        if vocab_sizes:
            self.categorical_encoder = EmbeddingLayer(vocab_sizes, embedding_dims)
            categorical_dim = self.categorical_encoder.output_dim
        else:
            self.categorical_encoder = None
            categorical_dim = 0
        
        # Numerical features projection
        if numerical_dim > 0:
            self.numerical_encoder = nn.Linear(numerical_dim, hidden_dim)
        else:
            self.numerical_encoder = None
            hidden_dim = 0
        
        self.output_dim = categorical_dim + hidden_dim
        
        # Optional feature fusion
        if categorical_dim > 0 and hidden_dim > 0:
            self.fusion = nn.Linear(self.output_dim, hidden_dim)
            self.output_dim = hidden_dim
        else:
            self.fusion = None
    
    def forward(self, categorical: Optional[torch.Tensor] = None, 
                numerical: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with separate categorical and numerical inputs."""
        features = []
        
        # Process categorical features
        if categorical is not None and self.categorical_encoder is not None:
            cat_features = self.categorical_encoder(categorical)
            features.append(cat_features)
        
        # Process numerical features  
        if numerical is not None and self.numerical_encoder is not None:
            num_features = self.numerical_encoder(numerical)
            features.append(num_features)
        
        if not features:
            raise ValueError("No features provided")
        
        # Concatenate features
        combined = torch.cat(features, dim=-1)
        
        # Optional fusion layer
        if self.fusion is not None:
            combined = self.fusion(combined)
        
        return combined


class BaseModel(pl.LightningModule):
    """Base model class for tabular self-supervised learning."""

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        embedding_dims: Dict[str, int], 
        numerical_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        **kwargs
    ):
        super().__init__()
        
        # Store hyperparameters
        self.save_hyperparameters()
        
        # Feature encoder
        self.feature_encoder = FeatureEncoder(
            vocab_sizes=vocab_sizes,
            embedding_dims=embedding_dims,
            numerical_dim=numerical_dim,
            hidden_dim=hidden_dim
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def encode_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode input features."""
        categorical = batch.get("categorical")
        numerical = batch.get("numerical")
        return self.feature_encoder(categorical=categorical, numerical=numerical)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the model."""
        return self.encode_features(batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step - to be implemented by subclasses."""
        raise NotImplementedError("Training step must be implemented")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step - to be implemented by subclasses."""
        raise NotImplementedError("Validation step must be implemented")

    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


# Backward compatibility placeholders (to be removed in future versions)
class ModelConfig:
    """Deprecated - use direct parameters instead."""
    pass

class TabularSSL(BaseModel):
    """Deprecated - use BaseModel instead."""
    pass

class TabularSSLConfig:
    """Deprecated - use direct parameters instead."""
    pass
