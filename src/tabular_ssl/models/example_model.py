import torch
import torch.nn.functional as F
from typing import Tuple
from .base import BaseModel
from .components import (
    MLPEventEncoder,
    TransformerSequenceEncoder,
    CategoricalEmbedding,
    MLPProjectionHead,
    ClassificationHead,
)


class ExampleModel(BaseModel):
    """An example model implementation using the modular components."""

    def _init_event_encoder(self) -> MLPEventEncoder:
        return MLPEventEncoder(self.config.event_encoder_config)

    def _init_sequence_encoder(self) -> TransformerSequenceEncoder:
        if self.config.sequence_encoder_config is None:
            return None
        return TransformerSequenceEncoder(self.config.sequence_encoder_config)

    def _init_embedding_layer(self) -> CategoricalEmbedding:
        if self.config.embedding_config is None:
            return None
        return CategoricalEmbedding(self.config.embedding_config)

    def _init_projection_head(self) -> MLPProjectionHead:
        if self.config.projection_head_config is None:
            return None
        return MLPProjectionHead(self.config.projection_head_config)

    def _init_prediction_head(self) -> ClassificationHead:
        if self.config.prediction_head_config is None:
            return None
        return ClassificationHead(self.config.prediction_head_config)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step with contrastive loss."""
        x, y = batch
        z = self(x)

        if self.prediction_head is not None:
            # Classification task
            loss = F.cross_entropy(z, y)
        else:
            # Self-supervised task (example: contrastive loss)
            # This is a simplified example - you would typically implement
            # a more sophisticated contrastive loss
            loss = self._contrastive_loss(z)

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Validation step."""
        x, y = batch
        z = self(x)

        if self.prediction_head is not None:
            # Classification task
            loss = F.cross_entropy(z, y)
            preds = torch.argmax(z, dim=1)
            acc = (preds == y).float().mean()
            self.log("val_loss", loss)
            self.log("val_acc", acc)
        else:
            # Self-supervised task
            loss = self._contrastive_loss(z)
            self.log("val_loss", loss)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _contrastive_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Simple contrastive loss implementation."""
        # This is a simplified example - you would typically implement
        # a more sophisticated contrastive loss
        z = F.normalize(z, dim=1)
        similarity = torch.matmul(z, z.t())
        labels = torch.arange(z.size(0), device=z.device)
        loss = F.cross_entropy(similarity, labels)
        return loss
