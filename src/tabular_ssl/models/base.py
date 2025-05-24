from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
import logging
import torch.nn.functional as F

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
    use_batch_norm: bool = False
) -> nn.Sequential:
    """Utility function to create MLP layers."""
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
        embedding: Optional[EmbeddingLayer] = None,  # Alternative name for Hydra configs
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_type: str = "adamw",
        scheduler_type: Optional[str] = "cosine",
        **kwargs
    ):
        super().__init__()
        
        # Store hyperparameters
        self.save_hyperparameters(ignore=["event_encoder", "sequence_encoder", 
                                         "projection_head", "prediction_head", 
                                         "embedding_layer", "embedding"])
        
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

    def training_step(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int):
        """Training step - to be implemented by subclasses."""
        raise NotImplementedError("Training step must be implemented")

    def validation_step(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int):
        """Validation step - to be implemented by subclasses."""
        raise NotImplementedError("Validation step must be implemented")

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Choose optimizer
        if self.optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
        
        # Configure scheduler if specified
        if self.scheduler_type is None:
            return optimizer
        elif self.scheduler_type.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            return [optimizer], [scheduler]
        elif self.scheduler_type.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
            return [optimizer], [scheduler]
        else:
            return optimizer


# Concrete implementations for common use cases
class TabularEmbedding(EmbeddingLayer):
    """Embedding layer for categorical tabular features."""
    
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


class TabularFeatureEncoder(EventEncoder):
    """Event encoder for tabular features (categorical + numerical)."""
    
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
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        output_dim: int, 
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False
    ):
        super().__init__()
        self.projection = create_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class MLPPredictionHead(PredictionHead):
    """MLP-based prediction head for classification."""
    
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int, 
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False
    ):
        super().__init__()
        
        if hidden_dims is None:
            # Simple linear classifier
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, num_classes)
            )
        else:
            # MLP classifier
            self.classifier = create_mlp(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=num_classes,
                dropout=dropout,
                activation=activation,
                use_batch_norm=use_batch_norm,
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class SSLModel(BaseModel):
    """Self-supervised learning model with corruption strategy support.
    
    Extends BaseModel to work with corruption strategies like VIME, SCARF, and ReConTab.
    Handles different corruption outputs and computes appropriate SSL losses.
    """
    
    def __init__(
        self,
        event_encoder: EventEncoder,
        corruption: Optional[nn.Module] = None,
        sequence_encoder: Optional[SequenceEncoder] = None,
        projection_head: Optional[ProjectionHead] = None,
        prediction_head: Optional[PredictionHead] = None,
        embedding_layer: Optional[EmbeddingLayer] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_type: str = "adamw",
        scheduler_type: Optional[str] = "cosine",
        # SSL-specific parameters (auto-detected from corruption module)
        mask_estimation_weight: float = 1.0,
        value_imputation_weight: float = 1.0,
        contrastive_temperature: float = 0.1,
        reconstruction_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__(
            event_encoder=event_encoder,
            sequence_encoder=sequence_encoder,
            projection_head=projection_head,
            prediction_head=prediction_head,
            embedding_layer=embedding_layer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            scheduler_type=scheduler_type,
            **kwargs
        )
        
        # SSL-specific components and parameters
        self.corruption = corruption
        self.corruption_type = self._detect_corruption_type(corruption)
        self.mask_estimation_weight = mask_estimation_weight
        self.value_imputation_weight = value_imputation_weight
        self.contrastive_temperature = contrastive_temperature
        self.reconstruction_weights = reconstruction_weights or {
            "masked": 1.0, "denoising": 1.0, "unswapping": 1.0
        }
        
        # Initialize SSL-specific heads based on corruption type
        self._init_ssl_heads()
    
    def _detect_corruption_type(self, corruption: Optional[nn.Module]) -> str:
        """Auto-detect corruption type from the corruption module."""
        if corruption is None:
            return "none"
        
        # Import here to avoid circular imports
        from .components import VIMECorruption, SCARFCorruption, ReConTabCorruption
        
        if isinstance(corruption, VIMECorruption):
            return "vime"
        elif isinstance(corruption, SCARFCorruption):
            return "scarf"
        elif isinstance(corruption, ReConTabCorruption):
            return "recontab"
        else:
            # For custom corruption strategies, try to infer from class name
            class_name = corruption.__class__.__name__.lower()
            if "vime" in class_name:
                return "vime"
            elif "scarf" in class_name:
                return "scarf"
            elif "recontab" in class_name or "recon" in class_name:
                return "recontab"
            else:
                logger.warning(f"Unknown corruption type: {corruption.__class__.__name__}")
                return "unknown"
    
    def _init_ssl_heads(self):
        """Initialize SSL-specific prediction heads based on corruption type."""
        if self.corruption_type == "vime":
            # VIME needs mask estimation and value imputation heads
            representation_dim = self._get_representation_dim()
            
            # Mask estimation head (binary classification per feature)
            self.mask_estimation_head = nn.Linear(representation_dim, 1)
            
            # Value imputation head (reconstruction)
            self.value_imputation_head = nn.Linear(representation_dim, 1)
            
        elif self.corruption_type == "scarf":
            # SCARF uses contrastive learning - no additional heads needed
            # Just ensure representations are normalized
            pass
            
        elif self.corruption_type == "recontab":
            # ReConTab needs heads for different reconstruction tasks
            representation_dim = self._get_representation_dim()
            
            self.masked_reconstruction_head = nn.Linear(representation_dim, 1)
            self.denoising_head = nn.Linear(representation_dim, 1)
            self.unswapping_head = nn.Linear(representation_dim, 1)
    
    def _get_representation_dim(self) -> int:
        """Get the dimension of representations from the encoder pipeline."""
        # This is a simplified approach - in practice you'd want to infer this
        # from the actual encoder dimensions
        if self.projection_head is not None:
            return getattr(self.projection_head, 'output_dim', 128)
        elif self.sequence_encoder is not None:
            return getattr(self.sequence_encoder, 'output_dim', 128)
        else:
            return getattr(self.event_encoder, 'output_dim', 128)
    
    def training_step(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int):
        """SSL training step with corruption strategy support."""
        # Handle both tensor and dict inputs
        if isinstance(batch, dict):
            x = batch.get('input', batch.get('x', None))
            if x is None:
                raise ValueError("Batch dict must contain 'input' or 'x' key")
        else:
            x = batch
        
        # Apply corruption strategy
        if self.corruption is None:
            # No corruption - just standard forward pass
            representations = self.encode(x)
            loss = self._compute_standard_loss(representations, x)
        else:
            loss = self._compute_ssl_loss(x)
        
        # Log loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def _compute_ssl_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SSL loss based on corruption type."""
        if self.corruption_type == "vime":
            return self._compute_vime_loss(x)
        elif self.corruption_type == "scarf":
            return self._compute_scarf_loss(x)
        elif self.corruption_type == "recontab":
            return self._compute_recontab_loss(x)
        elif self.corruption_type == "none":
            # No corruption - standard reconstruction loss
            representations = self.encode(x)
            return self._compute_standard_loss(representations, x)
        else:
            raise ValueError(
                f"Unknown corruption type: {self.corruption_type}. "
                f"Expected one of: 'vime', 'scarf', 'recontab', 'none'. "
                f"Corruption module: {self.corruption.__class__.__name__ if self.corruption else 'None'}"
            )
    
    def _compute_vime_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute VIME loss: mask estimation + value imputation."""
        # Apply VIME corruption
        corruption_output = self.corruption(x)
        x_corrupted = corruption_output['corrupted']
        mask = corruption_output['mask']
        
        # Get representations
        representations = self.encode(x_corrupted)
        
        # Flatten for per-element prediction
        batch_size, seq_len, num_features = x.shape
        repr_flat = representations.view(-1, representations.size(-1))
        mask_flat = mask.view(-1, 1)
        x_flat = x.view(-1, 1)
        
        # Mask estimation loss
        mask_pred = self.mask_estimation_head(repr_flat)
        mask_loss = F.binary_cross_entropy_with_logits(mask_pred, mask_flat)
        
        # Value imputation loss (only on masked positions)
        value_pred = self.value_imputation_head(repr_flat)
        masked_positions = mask_flat.bool().squeeze()
        if masked_positions.any():
            imputation_loss = F.mse_loss(
                value_pred[masked_positions], 
                x_flat[masked_positions]
            )
        else:
            imputation_loss = torch.tensor(0.0, device=x.device)
        
        # Combined loss
        total_loss = (
            self.mask_estimation_weight * mask_loss + 
            self.value_imputation_weight * imputation_loss
        )
        
        # Log individual losses
        self.log("train/mask_estimation_loss", mask_loss)
        self.log("train/value_imputation_loss", imputation_loss)
        
        return total_loss
    
    def _compute_scarf_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SCARF contrastive loss."""
        # Create contrastive pairs
        view1, view2 = self.corruption.create_contrastive_pairs(x)
        
        # Get normalized representations
        z1 = F.normalize(self.encode(view1), dim=-1)
        z2 = F.normalize(self.encode(view2), dim=-1)
        
        # Pool over sequence dimension (mean pooling)
        z1_pooled = z1.mean(dim=1)  # (batch_size, repr_dim)
        z2_pooled = z2.mean(dim=1)  # (batch_size, repr_dim)
        
        # Contrastive loss (InfoNCE)
        similarity_matrix = torch.matmul(z1_pooled, z2_pooled.T) / self.contrastive_temperature
        batch_size = z1_pooled.size(0)
        labels = torch.arange(batch_size, device=x.device)
        
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        
        # Log loss
        self.log("train/contrastive_loss", contrastive_loss)
        
        return contrastive_loss
    
    def _compute_recontab_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ReConTab multi-task reconstruction loss."""
        # Apply ReConTab corruption
        corruption_output = self.corruption(x)
        x_corrupted = corruption_output['corrupted']
        corruption_info = corruption_output['metadata']
        
        # Get representations
        representations = self.encode(x_corrupted)
        
        # Get reconstruction targets
        targets = self.corruption.reconstruction_targets(x, x_corrupted, corruption_info)
        
        total_loss = torch.tensor(0.0, device=x.device)
        num_tasks = 0
        
        # Flatten representations for per-element prediction
        repr_flat = representations.view(-1, representations.size(-1))
        
        # Masked reconstruction loss
        if "mask_positions" in targets:
            mask_positions = targets["mask_positions"].view(-1)
            if mask_positions.any():
                mask_pred = self.masked_reconstruction_head(repr_flat[mask_positions])
                mask_target = targets["masked_values"].view(-1, 1)
                mask_loss = F.mse_loss(mask_pred, mask_target)
                total_loss += self.reconstruction_weights["masked"] * mask_loss
                self.log("train/masked_reconstruction_loss", mask_loss)
                num_tasks += 1
        
        # Denoising loss
        if "noise_positions" in targets:
            noise_positions = targets["noise_positions"].view(-1)
            if noise_positions.any():
                denoise_pred = self.denoising_head(repr_flat[noise_positions])
                denoise_target = targets["denoised_values"].view(-1, 1)
                denoise_loss = F.mse_loss(denoise_pred, denoise_target)
                total_loss += self.reconstruction_weights["denoising"] * denoise_loss
                self.log("train/denoising_loss", denoise_loss)
                num_tasks += 1
        
        # Unswapping loss
        if "swap_positions" in targets:
            swap_positions = targets["swap_positions"].view(-1)
            if swap_positions.any():
                unswap_pred = self.unswapping_head(repr_flat[swap_positions])
                unswap_target = targets["unswapped_values"].view(-1, 1)
                unswap_loss = F.mse_loss(unswap_pred, unswap_target)
                total_loss += self.reconstruction_weights["unswapping"] * unswap_loss
                self.log("train/unswapping_loss", unswap_loss)
                num_tasks += 1
        
        # Average over active tasks
        if num_tasks > 0:
            total_loss = total_loss / num_tasks
        
        return total_loss
    
    def _compute_standard_loss(self, representations: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute standard reconstruction loss when no corruption is used."""
        # Simple reconstruction loss
        if representations.shape != x.shape:
            # If representations have different shape, need a reconstruction head
            if self.prediction_head is not None:
                reconstructed = self.prediction_head(representations)
            else:
                raise ValueError("Need prediction head for reconstruction when shapes don't match")
        else:
            reconstructed = representations
        
        loss = F.mse_loss(reconstructed, x)
        return loss
    
    def validation_step(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int):
        """Validation step."""
        # Handle both tensor and dict inputs
        if isinstance(batch, dict):
            x = batch.get('input', batch.get('x', None))
            if x is None:
                raise ValueError("Batch dict must contain 'input' or 'x' key")
        else:
            x = batch
        
        # Compute validation loss (same as training but no gradients)
        if self.corruption is None:
            representations = self.encode(x)
            loss = self._compute_standard_loss(representations, x)
        else:
            loss = self._compute_ssl_loss(x)
        
        # Log validation loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
