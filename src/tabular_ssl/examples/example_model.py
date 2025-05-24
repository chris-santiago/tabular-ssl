import torch
import torch.nn.functional as F
from typing import Tuple
from ..models.base import BaseModel
from ..models.components import (
    MLPEventEncoder,
    TransformerSequenceEncoder,
    CategoricalEmbedding,
    MLPProjectionHead,
    ClassificationHead,
)
from omegaconf import OmegaConf
from pytorch_lightning import Trainer


class ExampleModel(BaseModel):
    """An example model implementation using the modular components."""

    def __init__(self, config):
        super().__init__(config)
        # Directly initialize components
        self.event_encoder = MLPEventEncoder(config.event_encoder_config)
        self.sequence_encoder = (
            TransformerSequenceEncoder(config.sequence_encoder_config)
            if config.sequence_encoder_config
            else None
        )
        self.embedding_layer = (
            CategoricalEmbedding(config.embedding_config)
            if config.embedding_config
            else None
        )
        self.projection_head = (
            MLPProjectionHead(config.projection_head_config)
            if config.projection_head_config
            else None
        )
        self.prediction_head = (
            ClassificationHead(config.prediction_head_config)
            if config.prediction_head_config
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding_layer is not None:
            x = self.embedding_layer(x)
        x = self.event_encoder(x)
        if self.sequence_encoder is not None:
            x = self.sequence_encoder(x)
        if self.projection_head is not None:
            x = self.projection_head(x)
        if self.prediction_head is not None:
            x = self.prediction_head(x)
        return x

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


# Example configuration
config = OmegaConf.create(
    {
        "model": {
            "event_encoder": {
                "name": "mlp_event_encoder",
                "type": "mlp_event_encoder",
                "input_dim": 64,
                "hidden_dims": [128, 256],
                "output_dim": 512,
                "dropout": 0.1,
                "use_batch_norm": True,
            },
            "sequence_encoder": {
                "name": "s4",
                "type": "s4",
                "input_dim": 512,
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.1,
                "bidirectional": True,
                "max_sequence_length": 2048,
            },
            "projection_head": {
                "name": "mlp_projection",
                "type": "mlp_projection",
                "input_dim": 512,
                "hidden_dims": [256],
                "output_dim": 128,
                "dropout": 0.1,
                "use_batch_norm": True,
            },
        },
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
    }
)


# Create a random dataset
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int, input_dim: int):
        self.data = torch.randn(num_samples, input_dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # Dummy target


# Initialize the model
model = BaseModel(config)

# Create a data loader
train_loader = torch.utils.data.DataLoader(RandomDataset(1000, 64), batch_size=32)

# Initialize the trainer
trainer = Trainer(max_epochs=5)

# Train the model
trainer.fit(model, train_loader)

# Test the model
trainer.test(model, train_loader)
