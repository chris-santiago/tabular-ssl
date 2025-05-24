from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Union
import torch
import torch.nn as nn
import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)


class BaseComponent(nn.Module, ABC):
    """Base class for all model components."""

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the component."""
        pass


class EventEncoder(BaseComponent):
    """Base class for event encoders - transforms raw events into representations."""

    pass


class SequenceEncoder(BaseComponent):
    """Base class for sequence encoders - processes sequences of events."""

    pass


class EmbeddingLayer(BaseComponent):
    """Base class for embedding layers - maps discrete tokens to dense vectors."""

    pass


class ProjectionHead(BaseComponent):
    """Base class for projection heads - projects representations to different spaces."""

    pass


class PredictionHead(BaseComponent):
    """Base class for prediction heads - makes final predictions from representations."""

    pass


def create_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    dropout: float = 0.1,
    activation: str = "relu",
    use_batch_norm: bool = False,
) -> nn.Sequential:
    """Utility function to create MLP layers."""
    activation_fn = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "silu": nn.SiLU(),
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


class BaseModel(pl.LightningModule):
    """Flexible base model class for self-supervised learning.

    Can be configured with any combination of:
    - Event encoder (required)
    - Sequence encoder (optional)
    - Projection head (optional)
    - Prediction head (optional)
    - Embedding layers (optional)
    """

    def __init__(
        self,
        event_encoder: EventEncoder,
        sequence_encoder: Optional[SequenceEncoder] = None,
        projection_head: Optional[ProjectionHead] = None,
        prediction_head: Optional[PredictionHead] = None,
        embedding_layer: Optional[EmbeddingLayer] = None,
        embedding: Optional[
            EmbeddingLayer
        ] = None,  # Alternative name for Hydra configs
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_type: str = "adamw",
        scheduler_type: Optional[str] = "cosine",
        **kwargs,
    ):
        super().__init__()

        # Store hyperparameters
        self.save_hyperparameters(
            ignore=[
                "event_encoder",
                "sequence_encoder",
                "projection_head",
                "prediction_head",
                "embedding_layer",
                "embedding",
            ]
        )

        # Required components
        self.event_encoder = event_encoder

        # Optional components
        self.sequence_encoder = sequence_encoder
        self.projection_head = projection_head
        self.prediction_head = prediction_head
        # Handle both embedding_layer and embedding parameter names
        self.embedding_layer = embedding_layer or embedding

        # Training configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass through the configured components.

        Args:
            x: Input tensor or dictionary of tensors

        Returns:
            Output tensor after passing through all configured components
        """
        # Handle embedding layer first if present
        if self.embedding_layer is not None:
            if isinstance(x, dict):
                # Assume embeddings are applied to specific keys
                for key in x:
                    if key in ["categorical", "tokens", "ids"]:
                        x[key] = self.embedding_layer(x[key])
            else:
                x = self.embedding_layer(x)

        # Event encoding (required)
        if isinstance(x, dict):
            # For dictionary inputs, pass the whole dict
            x = self.event_encoder(x)
        else:
            x = self.event_encoder(x)

        # Sequence encoding (optional)
        if self.sequence_encoder is not None:
            x = self.sequence_encoder(x)

        # Projection (optional)
        if self.projection_head is not None:
            x = self.projection_head(x)

        # Final prediction (optional)
        if self.prediction_head is not None:
            x = self.prediction_head(x)

        return x

    def encode(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Encode input without final prediction head."""
        # Apply all components except prediction head
        original_pred_head = self.prediction_head
        self.prediction_head = None

        encoded = self.forward(x)

        # Restore prediction head
        self.prediction_head = original_pred_head

        return encoded

    def training_step(
        self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int
    ):
        """Training step - to be implemented by subclasses."""
        raise NotImplementedError("Training step must be implemented")

    def validation_step(
        self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int
    ):
        """Validation step - to be implemented by subclasses."""
        raise NotImplementedError("Validation step must be implemented")

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Choose optimizer
        if self.optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        # Optional scheduler
        if self.scheduler_type is None:
            return optimizer

        if self.scheduler_type.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.learning_rate * 0.01,
            )
        elif self.scheduler_type.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif self.scheduler_type.lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# Concrete implementations for common use cases
class TabularEmbedding(EmbeddingLayer):
    """Embedding layer for categorical tabular features."""

    def __init__(self, vocab_sizes: Dict[str, int], embedding_dims: Dict[str, int]):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embedding_dims = embedding_dims

        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(vocab_size, embedding_dims[col])
                for col, vocab_size in vocab_sizes.items()
            }
        )

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


class TabularFeatureEncoder(EventEncoder):
    """Event encoder for tabular features (categorical + numerical)."""

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        embedding_dims: Dict[str, int],
        numerical_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Categorical embeddings
        if vocab_sizes:
            self.categorical_encoder = TabularEmbedding(vocab_sizes, embedding_dims)
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

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with dictionary input containing categorical and numerical features."""
        features = []

        # Process categorical features
        categorical = batch.get("categorical")
        if categorical is not None and self.categorical_encoder is not None:
            cat_features = self.categorical_encoder(categorical)
            features.append(cat_features)

        # Process numerical features
        numerical = batch.get("numerical")
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


class MLPProjectionHead(ProjectionHead):
    """MLP-based projection head."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class MLPPredictionHead(PredictionHead):
    """MLP-based prediction head for classification."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# Backward compatibility - maintain existing names
FeatureEncoder = TabularFeatureEncoder  # For imports that expect this name


# Backward compatibility placeholders (deprecated)
class ModelConfig:
    """Deprecated - use direct component instantiation instead."""

    pass


class TabularSSL(BaseModel):
    """Deprecated - use BaseModel with TabularFeatureEncoder instead."""

    pass


class TabularSSLConfig:
    """Deprecated - use direct parameters instead."""

    pass
