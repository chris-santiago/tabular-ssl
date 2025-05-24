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
        custom_loss_fn: Optional[Callable] = None,
        
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
                                         'embedding_layer', 'corruption', 'custom_loss_fn'])
        
        # Core components
        self.event_encoder = event_encoder
        self.sequence_encoder = sequence_encoder
        self.projection_head = projection_head
        self.prediction_head = prediction_head
        self.embedding_layer = embedding_layer
        
        # SSL components
        self.corruption = corruption
        self.custom_loss_fn = custom_loss_fn
        self.is_ssl = corruption is not None
        self.ssl_loss_weights = ssl_loss_weights or {}
        self.contrastive_temperature = contrastive_temperature

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
        """SSL training step with corruption and loss computation."""
        x = batch['features'] if isinstance(batch, dict) else batch
        
        # Apply corruption and get representations
        corrupted_data = self.corruption(x)
        representations = self.encode(corrupted_data['corrupted'])
        
        # Compute loss using custom function or auto-detection
        if self.custom_loss_fn is not None:
            loss = self._compute_custom_loss(representations, x, corrupted_data)
        else:
            loss = self._compute_builtin_loss(representations, x, corrupted_data)
        
        self.log('train/ssl_loss', loss, on_step=True, on_epoch=True)
        return loss

    def _compute_custom_loss(self, representations: torch.Tensor, targets: torch.Tensor, 
                           corrupted_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss using custom loss function with unified interface."""
        # Try full signature first
        try:
            return self.custom_loss_fn(
                predictions=representations,
                targets=targets,
                model=self,
                corrupted_data=corrupted_data,
                ssl_loss_weights=self.ssl_loss_weights
            )
        except TypeError:
            # Fallback to simple (predictions, targets) signature
            predictions = self._create_predictions(representations, targets)
            return self.custom_loss_fn(predictions, targets)

    def _compute_builtin_loss(self, representations: torch.Tensor, targets: torch.Tensor,
                            corrupted_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss using built-in SSL methods."""
        # Auto-detection mapping
        builtin_losses = {
            'vime': vime_loss_fn,
            'scarf': scarf_loss_fn,
            'recontab': recontab_loss_fn
        }
        
        corruption_name = self.corruption.__class__.__name__.lower()
        
        # Find matching built-in loss function
        for name, loss_fn in builtin_losses.items():
            if name in corruption_name:
                return loss_fn(self, representations, targets, corrupted_data, self.ssl_loss_weights)
        
        # Generic reconstruction loss for unknown corruptions
        return self._generic_reconstruction_loss(representations, targets)

    def _create_predictions(self, representations: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Create predictions from representations for simple loss functions."""
        # Create reconstruction head if needed
        if not hasattr(self, 'simple_head'):
            repr_dim = representations.size(-1)
            target_dim = targets.size(-1)
            self.simple_head = nn.Linear(repr_dim, target_dim).to(representations.device)
        
        predictions = self.simple_head(representations)
        
        # Handle tensor dimension mismatches
        if predictions.dim() != targets.dim():
            if predictions.dim() == 3 and targets.dim() == 2:
                predictions = predictions.mean(dim=1)
            elif predictions.dim() == 2 and targets.dim() == 3:
                predictions = predictions.unsqueeze(1).expand(-1, targets.size(1), -1)
        
        return predictions

    def _generic_reconstruction_loss(self, representations: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Generic reconstruction loss for unknown corruption types."""
        if not hasattr(self, 'reconstruction_head'):
            repr_dim = representations.size(-1)
            target_dim = targets.size(-1)
            self.reconstruction_head = nn.Linear(repr_dim, target_dim).to(representations.device)
        
        reconstructed = self.reconstruction_head(representations)
        return F.mse_loss(reconstructed, targets)

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


# =============================================================================
# STANDALONE SSL LOSS FUNCTIONS
# =============================================================================

def vime_loss_fn(
    model: TabularSSLModel,
    representations: torch.Tensor,
    original: torch.Tensor,
    corrupted_data: Dict[str, torch.Tensor],
    ssl_loss_weights: Dict[str, float]
) -> torch.Tensor:
    """Standalone VIME loss function.
    
    Args:
        model: The TabularSSLModel instance
        representations: Encoded representations from corrupted data
        original: Original uncorrupted data
        corrupted_data: Dictionary containing corruption results
        ssl_loss_weights: Dictionary of loss component weights
        
    Returns:
        Combined VIME loss (mask estimation + value imputation)
    """
    # Ensure model has required heads - use same logic as _init_ssl_heads
    if not hasattr(model, 'mask_head'):
        repr_dim = representations.size(-1)
        model.mask_head = nn.Linear(repr_dim, 1).to(representations.device)
    if not hasattr(model, 'value_head'):
        repr_dim = representations.size(-1)
        # Use same logic as _init_ssl_heads for consistency
        input_dim = model.event_encoder.input_dim if hasattr(model.event_encoder, 'input_dim') else original.size(-1)
        model.value_head = nn.Linear(repr_dim, input_dim).to(representations.device)
    
    # Predictions
    mask_pred = torch.sigmoid(model.mask_head(representations))
    value_pred = model.value_head(representations)
    
    # Get weights
    mask_weight = ssl_loss_weights.get('mask_estimation', 1.0)
    value_weight = ssl_loss_weights.get('value_imputation', 1.0)
    
    # Losses - handle tensor dimensions properly
    mask_true = corrupted_data['mask'].float()
    if mask_true.dim() == 3:  # (batch, seq, features)
        mask_true = mask_true.mean(dim=-1)  # Average over features: (batch, seq)
    if mask_pred.dim() == 3:  # (batch, seq, 1)
        mask_pred = mask_pred.squeeze(-1)  # Remove last dim: (batch, seq)
    
    mask_loss = F.binary_cross_entropy(mask_pred, mask_true)
    value_loss = F.mse_loss(value_pred, original)
    
    return mask_weight * mask_loss + value_weight * value_loss


def scarf_loss_fn(
    model: TabularSSLModel,
    representations: torch.Tensor,
    original: torch.Tensor,
    corrupted_data: Dict[str, torch.Tensor],
    ssl_loss_weights: Dict[str, float]
) -> torch.Tensor:
    """Standalone SCARF contrastive loss function.
    
    Args:
        model: The TabularSSLModel instance
        representations: Encoded representations from corrupted data
        original: Original uncorrupted data
        corrupted_data: Dictionary containing corruption results
        ssl_loss_weights: Dictionary of loss component weights
        
    Returns:
        SCARF contrastive loss
    """
    if model.projection_head is None:
        raise ValueError("SCARF requires a projection head for contrastive learning")
        
    # Get positive and negative pairs from corruption
    anchor = model.projection_head(representations)
    positive = model.projection_head(model.encode(corrupted_data.get('positive', original)))
    
    # Flatten to 2D for matrix operations if needed
    if anchor.dim() == 3:  # (batch, seq, features)
        batch_size, seq_len, feature_dim = anchor.shape
        anchor = anchor.view(batch_size * seq_len, feature_dim)
        positive = positive.view(batch_size * seq_len, feature_dim)
    
    batch_size = anchor.size(0)
    
    # Compute similarities
    pos_sim = F.cosine_similarity(anchor, positive, dim=1) / model.contrastive_temperature
    neg_sim = torch.matmul(anchor, anchor.t()) / model.contrastive_temperature
    
    # Remove self-similarities from negatives
    mask = torch.eye(batch_size, device=anchor.device, dtype=torch.bool)
    neg_sim = neg_sim.masked_fill(mask, float('-inf'))
    
    # InfoNCE loss
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(batch_size, device=anchor.device, dtype=torch.long)
    
    contrastive_weight = ssl_loss_weights.get('contrastive', 1.0)
    return contrastive_weight * F.cross_entropy(logits, labels)


def recontab_loss_fn(
    model: TabularSSLModel,
    representations: torch.Tensor,
    original: torch.Tensor,
    corrupted_data: Dict[str, torch.Tensor],
    ssl_loss_weights: Dict[str, float]
) -> torch.Tensor:
    """Standalone ReConTab reconstruction loss function.
    
    Args:
        model: The TabularSSLModel instance
        representations: Encoded representations from corrupted data
        original: Original uncorrupted data
        corrupted_data: Dictionary containing corruption results
        ssl_loss_weights: Dictionary of loss component weights
        
    Returns:
        ReConTab reconstruction loss
    """
    # Ensure model has reconstruction head - use same logic as _init_ssl_heads
    if not hasattr(model, 'reconstruction_head'):
        repr_dim = representations.size(-1)
        # Use same logic as _init_ssl_heads for consistency
        input_dim = model.event_encoder.input_dim if hasattr(model.event_encoder, 'input_dim') else original.size(-1)
        model.reconstruction_head = nn.Linear(repr_dim, input_dim).to(representations.device)
    
    reconstructed = model.reconstruction_head(representations)
    
    # Multi-task reconstruction with optional weights
    reconstruction_loss = F.mse_loss(reconstructed, original)
    
    # Apply weights if specified
    total_weight = sum(ssl_loss_weights.get(k, 1.0) 
                      for k in ['masked', 'denoising', 'unswapping'])
    
    return reconstruction_loss * total_weight


def custom_mixup_loss_fn(
    model: TabularSSLModel,
    representations: torch.Tensor,
    original: torch.Tensor,
    corrupted_data: Dict[str, torch.Tensor],
    ssl_loss_weights: Dict[str, float]
) -> torch.Tensor:
    """Example custom loss function for Mixup-based SSL.
    
    Args:
        model: The TabularSSLModel instance
        representations: Encoded representations from corrupted data
        original: Original uncorrupted data
        corrupted_data: Dictionary containing corruption results
        ssl_loss_weights: Dictionary of loss component weights
        
    Returns:
        Custom Mixup SSL loss
    """
    # Create reconstruction head if needed
    if not hasattr(model, 'mixup_head'):
        repr_dim = representations.size(-1)
        model.mixup_head = nn.Linear(repr_dim, repr_dim).to(representations.device)
    
    # Predict mixed representations
    pred_mixed = model.mixup_head(representations)
    
    # Get mixup components from corruption
    mixup_targets = corrupted_data.get('mixup_targets', original)
    mixup_lambdas = corrupted_data.get('mixup_lambdas', torch.ones_like(original[:, :1, :1]))
    
    # Mixup target
    target_mixed = mixup_lambdas * original + (1 - mixup_lambdas) * mixup_targets
    
    # Main mixup loss
    mixup_loss = F.mse_loss(pred_mixed, target_mixed)
    
    # Optional lambda prediction loss
    lambda_weight = ssl_loss_weights.get('mixup_lambda', 0.1)
    if lambda_weight > 0:
        lambda_pred = torch.sigmoid(model.mixup_head(representations).mean(dim=-1, keepdim=True))
        lambda_loss = F.mse_loss(lambda_pred, mixup_lambdas)
        mixup_loss += lambda_weight * lambda_loss
    
    return mixup_loss


# =============================================================================
# LOSS FUNCTION REGISTRY
# =============================================================================

SSL_LOSS_FUNCTIONS = {
    'vime': vime_loss_fn,
    'scarf': scarf_loss_fn, 
    'recontab': recontab_loss_fn,
    'mixup': custom_mixup_loss_fn,
}


def get_ssl_loss_function(name: str) -> Callable:
    """Get a predefined SSL loss function by name.
    
    Args:
        name: Name of the loss function ('vime', 'scarf', 'recontab', 'mixup')
        
    Returns:
        The corresponding loss function
        
    Raises:
        ValueError: If the loss function name is not found
    """
    if name not in SSL_LOSS_FUNCTIONS:
        available = list(SSL_LOSS_FUNCTIONS.keys())
        raise ValueError(f"Unknown SSL loss function '{name}'. Available: {available}")
    
    return SSL_LOSS_FUNCTIONS[name] 