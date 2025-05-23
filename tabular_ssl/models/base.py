from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the base model."""

    event_encoder_config: Dict[str, Any]
    sequence_encoder_config: Optional[Dict[str, Any]] = None
    embedding_config: Optional[Dict[str, Any]] = None
    projection_head_config: Optional[Dict[str, Any]] = None
    prediction_head_config: Optional[Dict[str, Any]] = None


class BaseComponent(ABC, nn.Module):
    """Base class for all model components."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class EventEncoder(BaseComponent):
    """Base class for event encoders."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class SequenceEncoder(BaseComponent):
    """Base class for sequence encoders."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class EmbeddingLayer(BaseComponent):
    """Base class for embedding layers."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ProjectionHead(BaseComponent):
    """Base class for projection heads."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class PredictionHead(BaseComponent):
    """Base class for prediction heads."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BaseModel(pl.LightningModule):
    """Base model class for self-supervised sequence modeling."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Initialize components
        self.event_encoder = self._init_event_encoder()
        self.sequence_encoder = self._init_sequence_encoder()
        self.embedding_layer = self._init_embedding_layer()
        self.projection_head = self._init_projection_head()
        self.prediction_head = self._init_prediction_head()

        self.save_hyperparameters()

    def _init_event_encoder(self) -> EventEncoder:
        """Initialize the event encoder."""
        raise NotImplementedError("Event encoder must be implemented")

    def _init_sequence_encoder(self) -> Optional[SequenceEncoder]:
        """Initialize the sequence encoder if configured."""
        if self.config.sequence_encoder_config is None:
            return None
        raise NotImplementedError("Sequence encoder must be implemented")

    def _init_embedding_layer(self) -> Optional[EmbeddingLayer]:
        """Initialize the embedding layer if configured."""
        if self.config.embedding_config is None:
            return None
        raise NotImplementedError("Embedding layer must be implemented")

    def _init_projection_head(self) -> Optional[ProjectionHead]:
        """Initialize the projection head if configured."""
        if self.config.projection_head_config is None:
            return None
        raise NotImplementedError("Projection head must be implemented")

    def _init_prediction_head(self) -> Optional[PredictionHead]:
        """Initialize the prediction head if configured."""
        if self.config.prediction_head_config is None:
            return None
        raise NotImplementedError("Prediction head must be implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # Apply embeddings if present
        if self.embedding_layer is not None:
            x = self.embedding_layer(x)

        # Encode events
        x = self.event_encoder(x)

        # Apply sequence encoding if present
        if self.sequence_encoder is not None:
            x = self.sequence_encoder(x)

        # Apply projection head if present
        if self.projection_head is not None:
            x = self.projection_head(x)

        # Apply prediction head if present
        if self.prediction_head is not None:
            x = self.prediction_head(x)

        return x

    def training_step(self, batch, batch_idx):
        """Training step."""
        raise NotImplementedError("Training step must be implemented")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        raise NotImplementedError("Validation step must be implemented")

    def configure_optimizers(self):
        """Configure optimizers."""
        raise NotImplementedError("Optimizer configuration must be implemented")
