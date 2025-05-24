from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BaseComponent(nn.Module, ABC):
    """Base class for all model components.
    
    This abstract base class provides a common interface for all model components
    in the tabular SSL framework. All components must implement the forward method.
    
    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the component.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        pass


class EventEncoder(BaseComponent):
    """Base class for event encoders.
    
    Event encoders transform raw events into dense representations.
    This is typically the first processing step in the model pipeline.
    """
    pass


class SequenceEncoder(BaseComponent):
    """Base class for sequence encoders.
    
    Sequence encoders process sequences of events to capture temporal
    or sequential dependencies in the data.
    """
    pass


class EmbeddingLayer(BaseComponent):
    """Base class for embedding layers.
    
    Embedding layers map discrete tokens (e.g., categorical features)
    to dense vector representations.
    """
    pass


class ProjectionHead(BaseComponent):
    """Base class for projection heads.
    
    Projection heads project representations to different spaces,
    commonly used in self-supervised learning and multi-task scenarios.
    """
    pass


class PredictionHead(BaseComponent):
    """Base class for prediction heads.
    
    Prediction heads make final predictions from learned representations,
    such as classification or regression outputs.
    """
    pass


def create_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    dropout: float = 0.1,
    activation: str = "relu",
    use_batch_norm: bool = False
) -> nn.Sequential:
    """Create a multi-layer perceptron (MLP) with configurable architecture.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : List[int]
        List of hidden layer dimensions.
    output_dim : int
        Output feature dimension.
    dropout : float, default=0.1
        Dropout probability applied after each hidden layer.
    activation : str, default="relu"
        Activation function. One of {"relu", "gelu", "tanh", "leaky_relu", "silu"}.
    use_batch_norm : bool, default=False
        Whether to use batch normalization after each linear layer.
        
    Returns
    -------
    nn.Sequential
        The constructed MLP as a sequential module.
        
    Raises
    ------
    KeyError
        If the specified activation function is not supported.
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
    return nn.Sequential(*layers)


class BaseModel(pl.LightningModule):
    """Flexible base model class for self-supervised learning.
    
    This class provides a modular framework for building neural networks with
    configurable components. It can be used with any combination of encoders,
    projection heads, and prediction heads.
    
    Parameters
    ----------
    event_encoder : EventEncoder
        Required event encoder to transform input events.
    sequence_encoder : Optional[SequenceEncoder], default=None
        Optional sequence encoder for processing sequential data.
    projection_head : Optional[ProjectionHead], default=None
        Optional projection head for representation learning.
    prediction_head : Optional[PredictionHead], default=None
        Optional prediction head for final outputs.
    embedding_layer : Optional[EmbeddingLayer], default=None
        Optional embedding layer for discrete inputs.
    embedding : Optional[EmbeddingLayer], default=None
        Alternative parameter name for embedding_layer (for Hydra configs).
    learning_rate : float, default=1e-3
        Learning rate for optimization.
    weight_decay : float, default=1e-4
        Weight decay (L2 regularization) coefficient.
    optimizer_type : str, default="adamw"
        Optimizer type. One of {"adam", "adamw"}.
    scheduler_type : Optional[str], default="cosine"
        Learning rate scheduler type. One of {"cosine", "step", None}.
    **kwargs : Any
        Additional keyword arguments.
        
    Attributes
    ----------
    event_encoder : EventEncoder
        The event encoder component.
    sequence_encoder : Optional[SequenceEncoder]
        The sequence encoder component.
    projection_head : Optional[ProjectionHead]
        The projection head component.
    prediction_head : Optional[PredictionHead]
        The prediction head component.
    embedding_layer : Optional[EmbeddingLayer]
        The embedding layer component.
    learning_rate : float
        Learning rate for optimization.
    weight_decay : float
        Weight decay coefficient.
    optimizer_type : str
        Optimizer type.
    scheduler_type : Optional[str]
        Scheduler type.
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
        **kwargs: Any
    ) -> None:
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
        
        Parameters
        ----------
        x : Union[torch.Tensor, Dict[str, torch.Tensor]]
            Input tensor of shape (batch_size, ...) or dictionary of tensors
            with keys like "categorical", "numerical", "tokens", "ids".
            
        Returns
        -------
        torch.Tensor
            Output tensor after passing through all configured components.
            Shape depends on the final component in the pipeline.
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
        """Encode input without final prediction head.
        
        This method applies all components except the prediction head,
        useful for extracting representations for downstream tasks.
        
        Parameters
        ----------
        x : Union[torch.Tensor, Dict[str, torch.Tensor]]
            Input tensor or dictionary of tensors.
            
        Returns
        -------
        torch.Tensor
            Encoded representations without final prediction.
        """
        # Apply all components except prediction head
        original_pred_head = self.prediction_head
        self.prediction_head = None
        
        encoded = self.forward(x)
        
        # Restore prediction head
        self.prediction_head = original_pred_head
        
        return encoded

    def training_step(
        self, 
        batch: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        batch_idx: int
    ) -> torch.Tensor:
        """SSL training step with corruption strategy support.
        
        This method handles the training step for different SSL strategies.
        It automatically detects the input format (tensor vs dict), applies
        the appropriate corruption strategy if available, and computes the
        corresponding SSL loss.
        
        Parameters
        ----------
        batch : Union[torch.Tensor, Dict[str, torch.Tensor]]
            Training batch. Can be either:
            - torch.Tensor: Direct tensor input of shape (batch_size, seq_len, features)
            - Dict[str, torch.Tensor]: Dictionary with keys like 'input', 'x', etc.
        batch_idx : int
            Index of the current batch within the epoch.
            
        Returns
        -------
        torch.Tensor
            Computed training loss scalar. The specific loss depends on the
            corruption strategy:
            - No corruption: Standard reconstruction loss
            - VIME: Weighted combination of mask estimation and value imputation losses
            - SCARF: Contrastive loss (InfoNCE)
            - ReConTab: Multi-task reconstruction loss
            
        Raises
        ------
        ValueError
            If batch is a dictionary but doesn't contain 'input' or 'x' keys.
        """
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

    def validation_step(
        self, 
        batch: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        batch_idx: int
    ) -> torch.Tensor:
        """Validation step for SSL model evaluation.
        
        This method computes validation loss using the same SSL strategy as
        training but without gradient computation. The validation loss is
        logged automatically for monitoring training progress.
        
        Parameters
        ----------
        batch : Union[torch.Tensor, Dict[str, torch.Tensor]]
            Validation batch. Can be either:
            - torch.Tensor: Direct tensor input of shape (batch_size, seq_len, features)
            - Dict[str, torch.Tensor]: Dictionary with keys like 'input', 'x', etc.
        batch_idx : int
            Index of the current batch within the validation set.
            
        Returns
        -------
        torch.Tensor
            Computed validation loss scalar. Uses the same loss computation
            as the training step but without gradient updates.
            
        Raises
        ------
        ValueError
            If batch is a dictionary but doesn't contain 'input' or 'x' keys.
        """
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

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Tuple[List[torch.optim.Optimizer], List[Any]]]:
        """Configure optimizers and schedulers.
        
        Returns
        -------
        Union[torch.optim.Optimizer, Tuple[List[torch.optim.Optimizer], List[Any]]]
            Either a single optimizer or a tuple of (optimizers, schedulers).
            
        Raises
        ------
        ValueError
            If an unknown optimizer type is specified.
        """
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
    """Embedding layer for categorical tabular features.
    
    This class creates separate embedding layers for each categorical feature
    and concatenates their outputs to form a unified representation.
    
    Parameters
    ----------
    vocab_sizes : Dict[str, int]
        Dictionary mapping feature names to their vocabulary sizes.
    embedding_dims : Dict[str, int]
        Dictionary mapping feature names to their embedding dimensions.
        
    Attributes
    ----------
    vocab_sizes : Dict[str, int]
        Vocabulary sizes for each categorical feature.
    embedding_dims : Dict[str, int]
        Embedding dimensions for each categorical feature.
    embeddings : nn.ModuleDict
        Dictionary of embedding layers for each feature.
    output_dim : int
        Total output dimension (sum of all embedding dimensions).
    """
    
    def __init__(self, vocab_sizes: Dict[str, int], embedding_dims: Dict[str, int]) -> None:
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
        """Forward pass through categorical embeddings.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, num_categorical_features)
            containing integer indices for each categorical feature.
            
        Returns
        -------
        torch.Tensor
            Concatenated embeddings of shape (batch_size, sequence_length, total_embedding_dim).
        """
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
    """Event encoder for tabular features (categorical + numerical).
    
    This encoder handles mixed tabular data by separately processing categorical
    and numerical features, then optionally fusing them together.
    
    Parameters
    ----------
    vocab_sizes : Dict[str, int]
        Dictionary mapping categorical feature names to vocabulary sizes.
    embedding_dims : Dict[str, int]
        Dictionary mapping categorical feature names to embedding dimensions.
    numerical_dim : int
        Dimension of numerical features.
    hidden_dim : int, default=128
        Hidden dimension for numerical feature projection.
        
    Attributes
    ----------
    categorical_encoder : Optional[TabularEmbedding]
        Embedding encoder for categorical features.
    numerical_encoder : Optional[nn.Linear]
        Linear projection for numerical features.
    output_dim : int
        Output feature dimension.
    fusion : Optional[nn.Linear]
        Optional fusion layer when both categorical and numerical features are present.
    """
    
    def __init__(
        self, 
        vocab_sizes: Dict[str, int], 
        embedding_dims: Dict[str, int], 
        numerical_dim: int,
        hidden_dim: int = 128
    ) -> None:
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
        """Forward pass with dictionary input containing categorical and numerical features.
        
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dictionary containing:
            - "categorical": Tensor of shape (batch_size, seq_len, num_cat_features)
            - "numerical": Tensor of shape (batch_size, seq_len, num_num_features)
            
        Returns
        -------
        torch.Tensor
            Encoded features of shape (batch_size, seq_len, output_dim).
            
        Raises
        ------
        ValueError
            If no features are provided in the batch.
        """
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
    """MLP-based projection head.
    
    This projection head uses a multi-layer perceptron to project input
    representations to a different dimensional space, commonly used in
    self-supervised learning and representation learning tasks.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : List[int]
        List of hidden layer dimensions for the MLP.
    output_dim : int
        Output feature dimension after projection.
    dropout : float, default=0.1
        Dropout probability applied after each hidden layer.
    activation : str, default="relu"
        Activation function. One of {"relu", "gelu", "tanh", "leaky_relu", "silu"}.
    use_batch_norm : bool, default=False
        Whether to use batch normalization after each linear layer.
        
    Attributes
    ----------
    projection : nn.Sequential
        The MLP network used for projection.
        
    Examples
    --------
    >>> projection_head = MLPProjectionHead(
    ...     input_dim=512,
    ...     hidden_dims=[256, 128],
    ...     output_dim=64,
    ...     activation="gelu",
    ...     use_batch_norm=True
    ... )
    >>> x = torch.randn(32, 512)
    >>> projected = projection_head(x)
    >>> projected.shape
    torch.Size([32, 64])
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        output_dim: int, 
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False
    ) -> None:
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
        """Forward pass through the projection head.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ..., input_dim).
            
        Returns
        -------
        torch.Tensor
            Projected tensor of shape (batch_size, ..., output_dim).
        """
        return self.projection(x)


class MLPPredictionHead(PredictionHead):
    """MLP-based prediction head for classification.
    
    This prediction head can be configured as either a simple linear classifier
    or a multi-layer perceptron for more complex prediction tasks. It automatically
    chooses the architecture based on whether hidden dimensions are provided.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    num_classes : int
        Number of output classes for classification.
    hidden_dims : Optional[List[int]], default=None
        List of hidden layer dimensions for the MLP. If None, uses a simple 
        linear classifier with only dropout and a final linear layer.
    dropout : float, default=0.1
        Dropout probability applied before the final layer (and after each 
        hidden layer if using MLP architecture).
    activation : str, default="relu"
        Activation function for hidden layers. One of {"relu", "gelu", "tanh", 
        "leaky_relu", "silu"}. Only used when hidden_dims is not None.
    use_batch_norm : bool, default=False
        Whether to use batch normalization after each linear layer in the MLP.
        Only used when hidden_dims is not None.
        
    Attributes
    ----------
    classifier : nn.Module
        The classification network. Either a simple nn.Sequential with dropout
        and linear layer, or a more complex MLP created with create_mlp().
        
    Examples
    --------
    Simple linear classifier:
    
    >>> pred_head = MLPPredictionHead(input_dim=256, num_classes=10)
    >>> x = torch.randn(32, 256)
    >>> logits = pred_head(x)
    >>> logits.shape
    torch.Size([32, 10])
    
    MLP classifier:
    
    >>> pred_head = MLPPredictionHead(
    ...     input_dim=256,
    ...     num_classes=10,
    ...     hidden_dims=[128, 64],
    ...     activation="gelu",
    ...     use_batch_norm=True
    ... )
    >>> x = torch.randn(32, 256)
    >>> logits = pred_head(x)
    >>> logits.shape
    torch.Size([32, 10])
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int, 
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False
    ) -> None:
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
        """Forward pass through the prediction head.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ..., input_dim).
            
        Returns
        -------
        torch.Tensor
            Class logits of shape (batch_size, ..., num_classes).
            Raw logits are returned (no softmax applied).
        """
        return self.classifier(x)


class SSLModel(BaseModel):
    """Self-supervised learning model with corruption strategy support.
    
    This class extends BaseModel to provide specialized support for self-supervised
    learning (SSL) tasks on tabular data. It automatically detects and handles
    different corruption strategies (VIME, SCARF, ReConTab) and computes
    appropriate SSL losses based on the corruption type.
    
    The model can work with or without corruption strategies:
    - Without corruption: Uses standard reconstruction loss
    - With VIME: Mask estimation + value imputation losses
    - With SCARF: Contrastive learning loss
    - With ReConTab: Multi-task reconstruction losses
    
    Parameters
    ----------
    event_encoder : EventEncoder
        Required event encoder to transform input events into representations.
    corruption : Optional[nn.Module], default=None
        Corruption strategy module. If None, the model uses standard 
        reconstruction loss. Supports VIME, SCARF, and ReConTab corruption
        strategies.
    sequence_encoder : Optional[SequenceEncoder], default=None
        Optional sequence encoder for processing sequential data.
    projection_head : Optional[ProjectionHead], default=None
        Optional projection head for representation learning.
    prediction_head : Optional[PredictionHead], default=None
        Optional prediction head for final outputs.
    embedding_layer : Optional[EmbeddingLayer], default=None
        Optional embedding layer for discrete inputs.
    learning_rate : float, default=1e-3
        Learning rate for optimization.
    weight_decay : float, default=1e-4
        Weight decay (L2 regularization) coefficient.
    optimizer_type : str, default="adamw"
        Optimizer type. One of {"adam", "adamw"}.
    scheduler_type : Optional[str], default="cosine"
        Learning rate scheduler type. One of {"cosine", "step", None}.
    mask_estimation_weight : float, default=1.0
        Weight for mask estimation loss in VIME. Only used when corruption
        type is "vime".
    value_imputation_weight : float, default=1.0
        Weight for value imputation loss in VIME. Only used when corruption
        type is "vime".
    contrastive_temperature : float, default=0.1
        Temperature parameter for contrastive loss in SCARF. Only used when
        corruption type is "scarf".
    reconstruction_weights : Optional[Dict[str, float]], default=None
        Weights for different reconstruction tasks in ReConTab. Keys should be
        {"masked", "denoising", "unswapping"}. If None, uses equal weights of 1.0.
        Only used when corruption type is "recontab".
    **kwargs : Any
        Additional keyword arguments passed to BaseModel.
        
    Attributes
    ----------
    corruption : Optional[nn.Module]
        The corruption strategy module.
    corruption_type : str
        Auto-detected corruption type. One of {"vime", "scarf", "recontab", 
        "none", "unknown"}.
    mask_estimation_weight : float
        Weight for mask estimation loss.
    value_imputation_weight : float
        Weight for value imputation loss.
    contrastive_temperature : float
        Temperature for contrastive learning.
    reconstruction_weights : Dict[str, float]
        Weights for reconstruction tasks.
    mask_estimation_head : Optional[nn.Linear]
        VIME mask estimation head (created automatically for VIME corruption).
    value_imputation_head : Optional[nn.Linear]
        VIME value imputation head (created automatically for VIME corruption).
    masked_reconstruction_head : Optional[nn.Linear]
        ReConTab masked reconstruction head (created automatically for ReConTab).
    denoising_head : Optional[nn.Linear]
        ReConTab denoising head (created automatically for ReConTab).
    unswapping_head : Optional[nn.Linear]
        ReConTab unswapping head (created automatically for ReConTab).
        
    Examples
    --------
    Basic SSL model without corruption:
    
    >>> from tabular_ssl.models.components import MLPEventEncoder
    >>> event_encoder = MLPEventEncoder(input_dim=10, hidden_dims=[64, 32], output_dim=16)
    >>> model = SSLModel(event_encoder=event_encoder)
    
    VIME SSL model:
    
    >>> from tabular_ssl.models.components import VIMECorruption
    >>> corruption = VIMECorruption(mask_probability=0.3)
    >>> model = SSLModel(
    ...     event_encoder=event_encoder,
    ...     corruption=corruption,
    ...     mask_estimation_weight=1.0,
    ...     value_imputation_weight=2.0
    ... )
    
    SCARF contrastive model:
    
    >>> from tabular_ssl.models.components import SCARFCorruption, MLPProjectionHead
    >>> corruption = SCARFCorruption(corruption_probability=0.6)
    >>> projection_head = MLPProjectionHead(input_dim=16, hidden_dims=[32], output_dim=8)
    >>> model = SSLModel(
    ...     event_encoder=event_encoder,
    ...     corruption=corruption,
    ...     projection_head=projection_head,
    ...     contrastive_temperature=0.07
    ... )
    
    Notes
    -----
    The model automatically detects the corruption type from the corruption module
    and initializes appropriate SSL-specific heads. Loss computation is handled
    automatically based on the detected corruption type.
    
    For custom corruption strategies, the class name should contain keywords
    like "vime", "scarf", or "recontab" for automatic detection, or the
    corruption type will be marked as "unknown".
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
        **kwargs: Any
    ) -> None:
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
        """Auto-detect corruption type from the corruption module.
        
        This method inspects the corruption module to determine what type of
        SSL strategy it implements. Detection is based on isinstance checks
        for known corruption classes, falling back to class name inspection.
        
        Parameters
        ----------
        corruption : Optional[nn.Module]
            The corruption module to inspect. Can be None.
            
        Returns
        -------
        str
            The detected corruption type. One of:
            - "none": No corruption module provided
            - "vime": VIME (Variational Information Maximizing Exploration) corruption
            - "scarf": SCARF (Self-supervised Contrastive learning for Representation learning) corruption  
            - "recontab": ReConTab (Tabular data Reconstruction) corruption
            - "unknown": Unrecognized corruption module
            
        Notes
        -----
        For custom corruption strategies, the method attempts to infer the type
        from the class name. Class names containing "vime", "scarf", or "recontab"
        will be mapped to the corresponding types.
        """
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
    
    def _init_ssl_heads(self) -> None:
        """Initialize SSL-specific prediction heads based on corruption type.
        
        This method creates task-specific neural network heads required for
        different SSL strategies. The heads are created automatically based
        on the detected corruption type:
        
        - VIME: Creates mask estimation and value imputation heads
        - SCARF: No additional heads needed (uses projection head directly)
        - ReConTab: Creates masked reconstruction, denoising, and unswapping heads
        - None/Unknown: No additional heads created
        
        The representation dimension is automatically inferred from the
        encoder pipeline components.
        
        Notes
        -----
        All created heads are simple linear layers that map from the
        representation dimension to a single output value. For more complex
        heads, they should be created manually and assigned to the appropriate
        attributes after model initialization.
        """
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
        """Get the dimension of representations from the encoder pipeline.
        
        This method attempts to infer the output dimension of the encoder
        pipeline by checking the output_dim attribute of pipeline components
        in order of precedence: projection_head, sequence_encoder, event_encoder.
        
        Returns
        -------
        int
            The inferred representation dimension. Defaults to 128 if no
            output_dim attribute is found on any component.
            
        Notes
        -----
        This is a simplified approach that relies on components having an
        `output_dim` attribute. For more robust dimension inference, consider
        running a forward pass with dummy data or implementing explicit
        dimension tracking in component classes.
        """
        if self.projection_head is not None:
            return getattr(self.projection_head, 'output_dim', 128)
        elif self.sequence_encoder is not None:
            return getattr(self.sequence_encoder, 'output_dim', 128)
        else:
            return getattr(self.event_encoder, 'output_dim', 128)
    
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
