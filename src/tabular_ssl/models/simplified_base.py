"""Simplified base classes for tabular SSL framework.

This module provides a streamlined base class hierarchy that removes unnecessary
abstraction layers while maintaining modularity and extensibility.
"""
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def create_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    dropout: float = 0.1,
    activation: str = "relu",
    use_batch_norm: bool = False,
    final_activation: bool = False
) -> nn.Sequential:
    """Create a multi-layer perceptron with configurable architecture.
    
    Simplified version that combines all MLP creation logic.
    """
    activation_fn = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "silu": nn.SiLU()
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
    if final_activation:
        layers.append(activation_fn)
    
    return nn.Sequential(*layers)


class TabularSSLModel(pl.LightningModule):
    """Unified model class for both standard and self-supervised learning.
    
    This simplified design merges BaseModel and SSLModel into a single class,
    removing unnecessary inheritance while maintaining all functionality.
    """

    def __init__(
        self,
        # Core components (required)
        event_encoder: nn.Module,
        
        # Optional components
        sequence_encoder: Optional[nn.Module] = None,
        projection_head: Optional[nn.Module] = None,
        prediction_head: Optional[nn.Module] = None,
        embedding_layer: Optional[nn.Module] = None,
        
        # SSL components (auto-creates heads based on corruption type)
        corruption: Optional[nn.Module] = None,
        
        # Training parameters
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_type: str = "adamw",
        scheduler_type: Optional[str] = "cosine",
        
        # SSL-specific parameters (only used if corruption is provided)
        ssl_loss_weights: Optional[Dict[str, float]] = None,
        contrastive_temperature: float = 0.1,
        
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        # Save hyperparameters for automatic logging
        self.save_hyperparameters(ignore=['event_encoder', 'sequence_encoder', 
                                         'projection_head', 'prediction_head', 
                                         'embedding_layer', 'corruption'])
        
        # Core components
        self.event_encoder = event_encoder
        self.sequence_encoder = sequence_encoder
        self.projection_head = projection_head
        self.prediction_head = prediction_head
        self.embedding_layer = embedding_layer
        
        # SSL components
        self.corruption = corruption
        self.is_ssl = corruption is not None
        self.ssl_loss_weights = ssl_loss_weights or {}
        self.contrastive_temperature = contrastive_temperature
        
        # Initialize SSL heads if needed
        if self.is_ssl:
            self._init_ssl_heads()

    def _init_ssl_heads(self) -> None:
        """Initialize SSL-specific heads based on corruption type."""
        if not self.corruption:
            return
            
        corruption_name = self.corruption.__class__.__name__.lower()
        repr_dim = self._get_representation_dim()
        
        # Simple SSL head initialization based on corruption type
        if "vime" in corruption_name:
            self.mask_head = nn.Linear(repr_dim, 1)
            self.value_head = nn.Linear(repr_dim, repr_dim)
        elif "recontab" in corruption_name:
            self.reconstruction_head = nn.Linear(repr_dim, repr_dim)
        # SCARF uses projection_head directly

    def _get_representation_dim(self) -> int:
        """Get the output dimension of the encoder pipeline."""
        # Start with event encoder output
        if hasattr(self.event_encoder, 'output_dim'):
            dim = self.event_encoder.output_dim
        elif hasattr(self.event_encoder, 'hidden_dim'):
            dim = self.event_encoder.hidden_dim
        else:
            # Fallback: inspect the last layer
            for layer in reversed(list(self.event_encoder.modules())):
                if isinstance(layer, nn.Linear):
                    dim = layer.out_features
                    break
            else:
                dim = 128  # Default fallback
        
        # Adjust for sequence encoder if present
        if self.sequence_encoder and hasattr(self.sequence_encoder, 'output_dim'):
            dim = self.sequence_encoder.output_dim
            
        return dim

    def encode(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Encode input through the encoder pipeline."""
        # Handle embedding if present
        if self.embedding_layer is not None:
            x = self.embedding_layer(x)
        
        # Event encoding
        x = self.event_encoder(x)
        
        # Sequence encoding if present
        if self.sequence_encoder is not None:
            x = self.sequence_encoder(x)
            
        return x

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass through the complete model."""
        # Get representations
        representations = self.encode(x)
        
        # Apply projection head if present
        if self.projection_head is not None:
            representations = self.projection_head(representations)
        
        # Apply prediction head if present (for downstream tasks)
        if self.prediction_head is not None:
            return self.prediction_head(representations)
            
        return representations

    def training_step(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """Training step with automatic SSL/standard loss selection."""
        if self.is_ssl:
            return self._ssl_training_step(batch)
        else:
            return self._standard_training_step(batch)

    def _ssl_training_step(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """SSL training step with corruption and appropriate loss."""
        x = batch['features'] if isinstance(batch, dict) else batch
        
        # Apply corruption
        corrupted_data = self.corruption(x)
        x_corrupted = corrupted_data['corrupted']
        
        # Get representations from corrupted data
        representations = self.encode(x_corrupted)
        
        # Compute SSL loss based on corruption type
        corruption_name = self.corruption.__class__.__name__.lower()
        
        if "vime" in corruption_name:
            loss = self._compute_vime_loss(representations, x, corrupted_data)
        elif "scarf" in corruption_name:
            loss = self._compute_scarf_loss(representations, x, corrupted_data)
        elif "recontab" in corruption_name:
            loss = self._compute_recontab_loss(representations, x, corrupted_data)
        else:
            # Generic SSL loss (reconstruction)
            reconstructed = self.reconstruction_head(representations)
            loss = F.mse_loss(reconstructed, x)
        
        self.log('train/ssl_loss', loss, on_step=True, on_epoch=True)
        return loss

    def _standard_training_step(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Standard training step for supervised learning."""
        if isinstance(batch, dict):
            x, y = batch['features'], batch['targets']
        else:
            x, y = batch  # Assume tuple/list
            
        y_pred = self.forward(x)
        loss = F.cross_entropy(y_pred, y)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        return loss

    def _compute_vime_loss(self, representations: torch.Tensor, 
                          original: torch.Tensor, corrupted_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute VIME loss (mask estimation + value imputation)."""
        mask_pred = torch.sigmoid(self.mask_head(representations))
        value_pred = self.value_head(representations)
        
        # Get weights from ssl_loss_weights or use defaults
        mask_weight = self.ssl_loss_weights.get('mask_estimation', 1.0)
        value_weight = self.ssl_loss_weights.get('value_imputation', 1.0)
        
        mask_loss = F.binary_cross_entropy(mask_pred.squeeze(), 
                                         corrupted_data['mask'].float())
        value_loss = F.mse_loss(value_pred, original)
        
        return mask_weight * mask_loss + value_weight * value_loss

    def _compute_scarf_loss(self, representations: torch.Tensor,
                           original: torch.Tensor, corrupted_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute SCARF contrastive loss."""
        if self.projection_head is None:
            raise ValueError("SCARF requires a projection head for contrastive learning")
            
        # Get positive and negative pairs from corruption
        anchor = self.projection_head(representations)
        positive = self.projection_head(self.encode(corrupted_data.get('positive', original)))
        
        # Simple InfoNCE loss implementation
        batch_size = anchor.size(0)
        
        # Compute similarities
        pos_sim = F.cosine_similarity(anchor, positive, dim=1) / self.contrastive_temperature
        neg_sim = torch.matmul(anchor, anchor.t()) / self.contrastive_temperature
        
        # Remove self-similarities from negatives
        mask = torch.eye(batch_size, device=anchor.device, dtype=torch.bool)
        neg_sim = neg_sim.masked_fill(mask, float('-inf'))
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, device=anchor.device, dtype=torch.long)
        
        return F.cross_entropy(logits, labels)

    def _compute_recontab_loss(self, representations: torch.Tensor,
                              original: torch.Tensor, corrupted_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute ReConTab reconstruction loss."""
        reconstructed = self.reconstruction_head(representations)
        
        # Multi-task reconstruction with optional weights
        reconstruction_loss = F.mse_loss(reconstructed, original)
        
        # Apply weights if specified
        total_weight = sum(self.ssl_loss_weights.get(k, 1.0) 
                          for k in ['masked', 'denoising', 'unswapping'])
        
        return reconstruction_loss * total_weight

    def validation_step(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        if self.is_ssl:
            loss = self._ssl_training_step(batch)  # Same as training for SSL
            self.log('val/ssl_loss', loss, on_step=False, on_epoch=True)
        else:
            loss = self._standard_training_step(batch)  # Same as training
            self.log('val/loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Tuple[List[torch.optim.Optimizer], List[Any]]]:
        """Configure optimizers and schedulers."""
        if self.hparams.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer_type}")

        if self.hparams.scheduler_type is None:
            return optimizer
        elif self.hparams.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100  # Should be set based on training epochs
            )
            return [optimizer], [scheduler]
        elif self.hparams.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
            return [optimizer], [scheduler]
        else:
            return optimizer 