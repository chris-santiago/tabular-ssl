import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from typing import List, Optional, Dict, Tuple, Union, Any
from .base import (
    BaseComponent,
    EventEncoder,
    SequenceEncoder,
    EmbeddingLayer,
    ProjectionHead,
    PredictionHead,
    create_mlp,
)


class MLPEventEncoder(EventEncoder):
    """Simple MLP-based event encoder.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : List[int]
        List of hidden layer dimensions.
    output_dim : int
        Output feature dimension.
    dropout : float, default=0.1
        Dropout probability.
    activation : str, default="relu"
        Activation function. One of {"relu", "gelu", "tanh", "leaky_relu", "silu"}.
    use_batch_norm : bool, default=False
        Whether to use batch normalization.
        
    Attributes
    ----------
    input_dim : int
        Input feature dimension.
    output_dim : int
        Output feature dimension.
    mlp : nn.Sequential
        The MLP network.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = create_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ..., input_dim).
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, ..., output_dim).
        """
        return self.mlp(x)


class AutoEncoderEventEncoder(EventEncoder):
    """Autoencoder-based event encoder for self-supervised learning.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : List[int]
        List of hidden layer dimensions for the encoder.
    latent_dim : int
        Latent representation dimension.
    dropout : float, default=0.1
        Dropout probability.
    activation : str, default="relu"
        Activation function. One of {"relu", "gelu", "tanh", "leaky_relu", "silu"}.
    use_batch_norm : bool, default=False
        Whether to use batch normalization.
        
    Attributes
    ----------
    encoder : nn.Sequential
        The encoder network.
    decoder : nn.Sequential
        The decoder network with reversed architecture.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        # Encoder
        self.encoder = create_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

        # Decoder (reverse architecture)
        decoder_dims = hidden_dims[::-1]  # Reverse hidden dims
        self.decoder = create_mlp(
            input_dim=latent_dim,
            hidden_dims=decoder_dims,
            output_dim=input_dim,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns the latent representation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ..., input_dim).
            
        Returns
        -------
        torch.Tensor
            Latent representation of shape (batch_size, ..., latent_dim).
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to input space.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representation of shape (batch_size, ..., latent_dim).
            
        Returns
        -------
        torch.Tensor
            Reconstructed tensor of shape (batch_size, ..., input_dim).
        """
        return self.decoder(z)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ..., input_dim).
            
        Returns
        -------
        torch.Tensor
            Scalar reconstruction loss (MSE between input and reconstruction).
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return F.mse_loss(x_recon, x)


class ContrastiveEventEncoder(EventEncoder):
    """Contrastive learning-based event encoder.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : List[int]
        List of hidden layer dimensions.
    output_dim : int
        Output feature dimension.
    temperature : float, default=0.1
        Temperature parameter for contrastive learning.
    dropout : float, default=0.1
        Dropout probability.
    activation : str, default="relu"
        Activation function. One of {"relu", "gelu", "tanh", "leaky_relu", "silu"}.
    use_batch_norm : bool, default=False
        Whether to use batch normalization.
        
    Attributes
    ----------
    encoder : nn.Sequential
        The encoder network.
    temperature : float
        Temperature parameter for contrastive learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        temperature: float = 0.1,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = create_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns normalized embeddings.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ..., input_dim).
            
        Returns
        -------
        torch.Tensor
            L2-normalized embeddings of shape (batch_size, ..., output_dim).
        """
        z = self.encoder(x)
        return F.normalize(z, dim=-1)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between two views.
        
        Parameters
        ----------
        z1 : torch.Tensor
            First view embeddings of shape (batch_size, output_dim).
        z2 : torch.Tensor
            Second view embeddings of shape (batch_size, output_dim).
            
        Returns
        -------
        torch.Tensor
            Scalar contrastive loss value.
        """
        # Compute similarity matrix
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature

        # Labels for positive pairs (diagonal elements)
        batch_size = z1.size(0)
        labels = torch.arange(batch_size, device=z1.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class TransformerSequenceEncoder(SequenceEncoder):
    """Transformer-based sequence encoder.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden dimension and output dimension.
    num_heads : int, default=8
        Number of attention heads.
    num_layers : int, default=6
        Number of transformer layers.
    dim_feedforward : int, default=2048
        Dimension of feedforward network.
    dropout : float, default=0.1
        Dropout probability.
    max_seq_length : int, default=2048
        Maximum sequence length for positional encoding.
    use_positional_encoding : bool, default=True
        Whether to use positional encoding.
        
    Attributes
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden dimension.
    output_dim : int
        Output dimension (same as hidden_dim).
    max_seq_length : int
        Maximum sequence length.
    use_positional_encoding : bool
        Whether positional encoding is used.
    input_proj : nn.Module
        Input projection layer (Linear or Identity).
    pos_encoding : Optional[nn.Parameter]
        Positional encoding parameters.
    transformer : nn.TransformerEncoder
        The transformer encoder.
    dropout : nn.Dropout
        Dropout layer.
    layer_norm : nn.LayerNorm
        Layer normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 2048,
        use_positional_encoding: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim  # Output dimension is same as hidden dimension
        self.max_seq_length = max_seq_length
        self.use_positional_encoding = use_positional_encoding

        # Input projection if needed
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # Positional encoding (optional)
        if self.use_positional_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(max_seq_length, hidden_dim) * 0.02)
        else:
            self.pos_encoding = None

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)

        # Add positional encoding (if enabled)
        if self.use_positional_encoding and self.pos_encoding is not None:
            if seq_len <= self.max_seq_length:
                x = x + self.pos_encoding[:seq_len]
            else:
                # If sequence is longer than max_seq_length, truncate positional encoding
                x = x + self.pos_encoding[:self.max_seq_length].repeat(
                    (seq_len + self.max_seq_length - 1) // self.max_seq_length, 1
                )[:seq_len]

        # Apply dropout and layer norm
        x = self.dropout(x)
        x = self.layer_norm(x)

        # Transformer encoding
        x = self.transformer(x)

        return x


class RNNSequenceEncoder(SequenceEncoder):
    """RNN-based sequence encoder (LSTM/GRU/RNN).
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden dimension.
    num_layers : int, default=2
        Number of RNN layers.
    rnn_type : str, default="lstm"
        Type of RNN. One of {"lstm", "gru", "rnn"}.
    dropout : float, default=0.1
        Dropout probability (applied between layers if num_layers > 1).
    bidirectional : bool, default=False
        Whether to use bidirectional RNN.
        
    Attributes
    ----------
    hidden_dim : int
        Hidden dimension.
    output_dim : int
        Output dimension (hidden_dim * 2 if bidirectional, else hidden_dim).
    num_layers : int
        Number of RNN layers.
    bidirectional : bool
        Whether bidirectional RNN is used.
    rnn : nn.Module
        The RNN module (LSTM, GRU, or RNN).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        rnn_type: str = "lstm",  # "lstm", "gru", "rnn"
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Select RNN type
        rnn_classes = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}

        if rnn_type not in rnn_classes:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        rnn_class = rnn_classes[rnn_type]

        self.rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RNN.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, output_dim).
        """
        output, _ = self.rnn(x)
        return output


# S4 Utility Functions
def _initialize_s4_parameters(
    d_state: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize complex state space parameters for S4.
    
    Parameters
    ----------
    d_state : int
        State space dimension.
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple of (A, B, C, D) parameters as complex tensors.
    """
    A = torch.complex(-torch.rand(d_state) * 0.1, torch.randn(d_state) * 0.1)
    B = torch.randn(d_state, dtype=torch.cfloat) * 0.02
    C = torch.randn(d_state, dtype=torch.cfloat) * 0.02
    D = torch.randn(d_state, dtype=torch.cfloat) * 0.02
    return A, B, C, D


def _compute_s4_kernel(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, L: int
) -> torch.Tensor:
    """Compute the S4 kernel for sequence length L.
    
    Parameters
    ----------
    A : torch.Tensor
        State transition matrix of shape (d_state,).
    B : torch.Tensor
        Input matrix of shape (d_state,).
    C : torch.Tensor
        Output matrix of shape (d_state,).
    D : torch.Tensor
        Feedthrough matrix of shape (d_state,).
    L : int
        Sequence length.
        
    Returns
    -------
    torch.Tensor
        S4 kernel of shape (L, d_state, d_state).
    """
    A_powers = torch.pow(A.unsqueeze(0), torch.arange(L, device=A.device))
    kernel = torch.einsum("l,dn->ldn", A_powers, B)
    kernel = torch.einsum("ldn,md->lmn", kernel, C)
    kernel = kernel + D.unsqueeze(0)
    return kernel


def _apply_bidirectional_s4_conv(
    x: torch.Tensor, kernel: torch.Tensor, d_model: int
) -> torch.Tensor:
    """Apply bidirectional convolution for S4.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, d_model).
    kernel : torch.Tensor
        S4 kernel of shape (seq_len, d_model, d_model).
    d_model : int
        Model dimension.
        
    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, seq_len, d_model).
    """
    L = x.size(1)
    x_padded = F.pad(x, (0, 0, L - 1, L - 1))
    kernel_rev = kernel.flip(0)
    y_forward = F.conv1d(
        x_padded.transpose(1, 2), kernel.transpose(1, 2), groups=d_model
    ).transpose(1, 2)
    y_backward = F.conv1d(
        x_padded.transpose(1, 2), kernel_rev.transpose(1, 2), groups=d_model
    ).transpose(1, 2)
    return y_forward + y_backward


class _S4Block(nn.Module):
    """S4 block with diagonal state space matrices.
    
    Parameters
    ----------
    d_model : int
        Model dimension.
    d_state : int
        State space dimension.
    dropout : float, default=0.1
        Dropout probability.
    bidirectional : bool, default=False
        Whether to use bidirectional processing.
    max_sequence_length : int, default=2048
        Maximum sequence length.
        
    Attributes
    ----------
    d_model : int
        Model dimension.
    d_state : int
        State space dimension.
    bidirectional : bool
        Whether bidirectional processing is used.
    max_sequence_length : int
        Maximum sequence length.
    A, B, C, D : torch.Tensor
        State space parameters.
    input_proj : nn.Linear
        Input projection layer.
    dropout : nn.Dropout
        Dropout layer.
    norm : nn.LayerNorm
        Layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.1,
        bidirectional: bool = False,
        max_sequence_length: int = 2048,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional
        self.max_sequence_length = max_sequence_length

        # State space parameters
        self.A, self.B, self.C, self.D = _initialize_s4_parameters(self.d_state)

        # Input projection
        self.input_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the S4 block.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Input projection and normalization
        x = self.input_proj(x)
        x = self.norm(x)

        # Get sequence length
        L = x.size(1)

        # Compute kernel
        kernel = _compute_s4_kernel(self.A, self.B, self.C, self.D, L)

        # Apply convolution
        if self.bidirectional:
            y = _apply_bidirectional_s4_conv(x, kernel, self.d_model)
        else:
            # Pad for causal
            x_padded = F.pad(x, (0, 0, L - 1, 0))
            # Compute convolution
            y = F.conv1d(
                x_padded.transpose(1, 2), kernel.transpose(1, 2), groups=self.d_model
            ).transpose(1, 2)

        # Apply dropout and residual connection
        y = self.dropout(y)
        y = y + x

        return y


class S4SequenceEncoder(SequenceEncoder):
    """S4 (Structured State Space) sequence encoder.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden dimension.
    num_layers : int, default=4
        Number of S4 layers.
    dropout : float, default=0.1
        Dropout probability.
    bidirectional : bool, default=False
        Whether to use bidirectional processing.
    max_sequence_length : int, default=2048
        Maximum sequence length.
    output_dim : Optional[int], default=None
        Output dimension. If None, uses hidden_dim.
        
    Attributes
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden dimension.
    num_layers : int
        Number of S4 layers.
    dropout : float
        Dropout probability.
    bidirectional : bool
        Whether bidirectional processing is used.
    max_sequence_length : int
        Maximum sequence length.
    output_dim : int
        Output dimension.
    input_proj : nn.Module
        Input projection layer.
    blocks : nn.ModuleList
        List of S4 blocks.
    output_proj : nn.Module
        Output projection layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = False,
        max_sequence_length: int = 2048,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.max_sequence_length = max_sequence_length

        # Input projection to hidden dimension
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # S4 blocks
        self.blocks = nn.ModuleList(
            [
                _S4Block(
                    d_model=hidden_dim,
                    d_state=hidden_dim,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    max_sequence_length=max_sequence_length,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection if specified
        if output_dim is not None and output_dim != hidden_dim:
            self.output_proj = nn.Linear(hidden_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.output_proj = nn.Identity()
            self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through S4 layers.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, output_dim).
        """
        # Input projection
        x = self.input_proj(x)

        # Apply S4 blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        x = self.output_proj(x)

        return x


class CategoricalEmbedding(EmbeddingLayer):
    """Embedding layer for categorical variables with flexible dimensions.
    
    Parameters
    ----------
    vocab_sizes : Dict[str, int]
        Dictionary mapping column names to vocabulary sizes.
    embedding_dims : Dict[str, int]
        Dictionary mapping column names to embedding dimensions.
        
    Attributes
    ----------
    vocab_sizes : Dict[str, int]
        Vocabulary sizes for each categorical column.
    embedding_dims : Dict[str, int]
        Embedding dimensions for each categorical column.
    embeddings : nn.ModuleDict
        Dictionary of embedding layers for each column.
    output_dim : int
        Total output dimension (sum of all embedding dimensions).
    """

    def __init__(self, vocab_sizes: Dict[str, int], embedding_dims: Dict[str, int]) -> None:
        super().__init__()

        self.vocab_sizes = vocab_sizes
        self.embedding_dims = embedding_dims

        # Create embedding for each categorical column
        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(vocab_size, embedding_dims[col])
                for col, vocab_size in vocab_sizes.items()
            }
        )

        self.output_dim = sum(embedding_dims.values())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through categorical embeddings.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, num_categorical_features).
            Contains integer indices for each categorical feature.
            
        Returns
        -------
        torch.Tensor
            Embedded features of shape (batch_size, sequence_length, total_embedding_dim).
        """
        embedded_features = []

        for i, (col, embedding) in enumerate(self.embeddings.items()):
            # Get indices for this column
            indices = x[:, :, i].long()
            # Embed and collect
            embedded = embedding(indices)
            embedded_features.append(embedded)

        # Concatenate all embeddings
        return torch.cat(embedded_features, dim=-1)


class MLPProjectionHead(ProjectionHead):
    """MLP-based projection head for downstream tasks.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : List[int]
        List of hidden layer dimensions.
    output_dim : int
        Output feature dimension.
    dropout : float, default=0.1
        Dropout probability.
    activation : str, default="relu"
        Activation function. One of {"relu", "gelu", "tanh", "leaky_relu", "silu"}.
    use_batch_norm : bool, default=False
        Whether to use batch normalization.
        
    Attributes
    ----------
    projection : nn.Sequential
        The MLP projection network.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False,
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
        """Forward pass through projection head.
        
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


class ClassificationHead(PredictionHead):
    """Classification head for supervised learning.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    num_classes : int
        Number of output classes.
    hidden_dims : Optional[List[int]], default=None
        List of hidden layer dimensions. If None, uses a simple linear classifier.
    dropout : float, default=0.1
        Dropout probability.
    activation : str, default="relu"
        Activation function. One of {"relu", "gelu", "tanh", "leaky_relu", "silu"}.
    use_batch_norm : bool, default=False
        Whether to use batch normalization.
        
    Attributes
    ----------
    classifier : nn.Module
        The classification network (either linear or MLP).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            # Simple linear classifier
            self.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(input_dim, num_classes)
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
        """Forward pass through classification head.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ..., input_dim).
            
        Returns
        -------
        torch.Tensor
            Class logits of shape (batch_size, ..., num_classes).
        """
        return self.classifier(x)


# Base class for corruption strategies
class BaseCorruption(BaseComponent):
    """Base class for corruption strategies used in self-supervised learning.
    
    All corruption strategies should inherit from this class and implement
    the standardized interface for consistent integration with SSL models.
    
    Parameters
    ----------
    corruption_rate : float, default=0.15
        Base corruption rate (interpretation depends on specific strategy).
    **kwargs : Any
        Additional keyword arguments.
        
    Attributes
    ----------
    corruption_rate : float
        The corruption rate parameter.
    """
    
    def __init__(self, corruption_rate: float = 0.15, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.corruption_rate = corruption_rate
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply corruption to input tensor.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, num_features).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'corrupted': Corrupted input tensor
            - 'targets': Targets for reconstruction (optional)
            - 'mask': Corruption mask (optional)
            - 'metadata': Additional corruption metadata (optional)
        """
        pass
    
    def get_loss_components(
        self, output: Dict[str, torch.Tensor], original: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract loss components from corruption output.
        
        Parameters
        ----------
        output : Dict[str, torch.Tensor]
            Output from forward() method.
        original : torch.Tensor
            Original uncorrupted input.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of loss components for this corruption strategy.
        """
        return {}


# Simple corruption strategies
class RandomMasking(BaseCorruption):
    """Random masking corruption strategy.
    
    Parameters
    ----------
    corruption_rate : float, default=0.15
        Probability of masking each element.
    **kwargs : Any
        Additional keyword arguments passed to BaseCorruption.
    """

    def __init__(self, corruption_rate: float = 0.15, **kwargs: Any) -> None:
        super().__init__(corruption_rate=corruption_rate, **kwargs)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply random masking to input tensor.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, num_features).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'corrupted': Masked input tensor
            - 'mask': Binary mask (1 where corrupted, 0 where original)
            - 'targets': Original tensor for reconstruction
        """
        mask = torch.rand_like(x) > self.corruption_rate
        corrupted = x * mask.float()
        
        return {
            'corrupted': corrupted,
            'mask': (1 - mask.float()),  # 1 where corrupted, 0 where original
            'targets': x,  # Reconstruction target is original
        }


class GaussianNoise(BaseCorruption):
    """Gaussian noise corruption strategy.
    
    Parameters
    ----------
    noise_std : float, default=0.1
        Standard deviation of Gaussian noise.
    **kwargs : Any
        Additional keyword arguments passed to BaseCorruption.
        
    Attributes
    ----------
    noise_std : float
        Standard deviation of the noise.
    """

    def __init__(self, noise_std: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise to input tensor.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, num_features).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'corrupted': Noisy input tensor
            - 'targets': Clean version for denoising
            - 'metadata': Noise parameters
        """
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            corrupted = x + noise
        else:
            corrupted = x
            
        return {
            'corrupted': corrupted,
            'targets': x,  # Clean version for denoising
            'metadata': {'noise_std': self.noise_std}
        }


class SwappingCorruption(BaseCorruption):
    """Feature swapping corruption strategy."""

    def __init__(self, swap_prob: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.swap_prob = swap_prob

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Randomly swap features between samples."""
        if not self.training:
            return {
                'corrupted': x,
                'targets': x
            }

        batch_size, seq_len, num_features = x.shape
        x_corrupted = x.clone()
        swap_mask = torch.zeros_like(x)

        # For each feature, randomly swap between samples
        for feat_idx in range(num_features):
            if torch.rand(1).item() < self.swap_prob:
                # Random permutation of batch indices
                perm_indices = torch.randperm(batch_size, device=x.device)
                x_corrupted[:, :, feat_idx] = x[perm_indices, :, feat_idx]
                swap_mask[:, :, feat_idx] = 1.0

        return {
            'corrupted': x_corrupted,
            'targets': x,  # Original for unswapping
            'mask': swap_mask,  # Which features were swapped
        }


class VIMECorruption(BaseCorruption):
    """
    VIME (Value Imputation and Mask Estimation) corruption strategy.

    Based on "VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain"
    https://arxiv.org/abs/2006.06775

    Creates corrupted views by:
    1. Generating binary mask vectors to indicate which features to corrupt
    2. For categorical features: replace with random category from vocabulary
    3. For numerical features: replace with random value from feature distribution
    4. Returns both corrupted data and mask for mask estimation pretext task
    """

    def __init__(
        self,
        corruption_rate: float = 0.3,
        categorical_indices: Optional[List[int]] = None,
        numerical_indices: Optional[List[int]] = None,
        categorical_vocab_sizes: Optional[Dict[int, int]] = None,
        numerical_distributions: Optional[Dict[int, Tuple[float, float]]] = None,
        **kwargs
    ):
        super().__init__(corruption_rate=corruption_rate, **kwargs)
        self.categorical_indices = categorical_indices or []
        self.numerical_indices = numerical_indices or []
        self.categorical_vocab_sizes = categorical_vocab_sizes or {}
        self.numerical_distributions = numerical_distributions or {}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply VIME corruption.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)

        Returns:
            Dictionary containing:
            - 'corrupted': Corrupted input with same shape as x
            - 'mask': Binary mask indicating corrupted positions (1=corrupted, 0=original)
            - 'targets': Original tensor for value imputation
        """
        if not self.training:
            return {
                'corrupted': x,
                'mask': torch.zeros_like(x),
                'targets': x
            }

        batch_size, seq_len, num_features = x.shape
        device = x.device

        # Generate corruption mask
        mask = torch.bernoulli(
            torch.full(
                (batch_size, seq_len, num_features), self.corruption_rate, device=device
            )
        )

        x_corrupted = x.clone()

        # Corrupt categorical features
        for feat_idx in self.categorical_indices:
            if feat_idx < num_features:
                vocab_size = self.categorical_vocab_sizes.get(
                    feat_idx, 10
                )  # Default vocab size
                mask_positions = mask[:, :, feat_idx].bool()

                if mask_positions.any():
                    # Replace with random categories
                    random_categories = torch.randint(
                        0, vocab_size, mask_positions.sum().shape, device=device
                    )
                    x_corrupted[:, :, feat_idx][mask_positions] = (
                        random_categories.float()
                    )

        # Corrupt numerical features
        for feat_idx in self.numerical_indices:
            if feat_idx < num_features:
                mask_positions = mask[:, :, feat_idx].bool()

                if mask_positions.any():
                    # Get distribution parameters (mean, std)
                    mean, std = self.numerical_distributions.get(feat_idx, (0.0, 1.0))

                    # Replace with random values from normal distribution
                    random_values = torch.normal(
                        mean, std, mask_positions.sum().shape, device=device
                    )
                    x_corrupted[:, :, feat_idx][mask_positions] = random_values

        return {
            'corrupted': x_corrupted,
            'mask': mask,
            'targets': x,  # Original values for imputation
        }

    def set_feature_distributions(
        self,
        data: torch.Tensor,
        categorical_indices: List[int],
        numerical_indices: List[int],
    ):
        """Set feature distributions from training data."""
        self.categorical_indices = categorical_indices
        self.numerical_indices = numerical_indices

        # Compute categorical vocabulary sizes
        for feat_idx in categorical_indices:
            if feat_idx < data.shape[-1]:
                unique_values = torch.unique(data[:, :, feat_idx])
                self.categorical_vocab_sizes[feat_idx] = len(unique_values)

        # Compute numerical feature distributions
        for feat_idx in numerical_indices:
            if feat_idx < data.shape[-1]:
                feature_data = data[:, :, feat_idx].flatten()
                mean = feature_data.mean().item()
                std = feature_data.std().item()
                self.numerical_distributions[feat_idx] = (mean, std)


class SCARFCorruption(BaseCorruption):
    """
    SCARF (Self-Supervised Contrastive Learning using Random Feature Corruption) corruption strategy.

    Based on "SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption for Representation Learning"
    https://arxiv.org/abs/2106.15147

    Creates corrupted views by randomly replacing feature values with values from other samples
    in the batch, specifically designed for contrastive learning on tabular data.
    """

    def __init__(
        self,
        corruption_rate: float = 0.6,
        corruption_strategy: str = "random_swap",  # "random_swap", "marginal_sampling"
        **kwargs
    ):
        super().__init__(corruption_rate=corruption_rate, **kwargs)
        self.corruption_strategy = corruption_strategy

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply SCARF corruption by randomly replacing features with values from other samples.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)

        Returns:
            Dictionary containing:
            - 'corrupted': Corrupted tensor with same shape as input
            - 'targets': Original tensor (for contrastive learning)
        """
        if not self.training:
            return {
                'corrupted': x,
                'targets': x
            }

        batch_size, seq_len, num_features = x.shape
        device = x.device

        x_corrupted = x.clone()

        # Generate corruption mask for features
        corruption_mask = torch.rand(num_features, device=device) < self.corruption_rate

        for feat_idx in range(num_features):
            if corruption_mask[feat_idx]:
                if self.corruption_strategy == "random_swap":
                    # Randomly permute this feature across all samples and sequences
                    feature_values = x[:, :, feat_idx].flatten()
                    perm_indices = torch.randperm(len(feature_values), device=device)
                    shuffled_values = feature_values[perm_indices]
                    x_corrupted[:, :, feat_idx] = shuffled_values.view(
                        batch_size, seq_len
                    )

                elif self.corruption_strategy == "marginal_sampling":
                    # Sample from marginal distribution (uniform sampling from feature values)
                    feature_values = x[:, :, feat_idx].flatten()
                    sample_indices = torch.randint(
                        0, len(feature_values), (batch_size, seq_len), device=device
                    )
                    x_corrupted[:, :, feat_idx] = feature_values[sample_indices]

        return {
            'corrupted': x_corrupted,
            'targets': x,  # Original for contrastive learning
        }

    def create_contrastive_pairs(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create two different corrupted views for contrastive learning.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)

        Returns:
            Tuple of (view1, view2) - two differently corrupted versions
        """
        output1 = self.forward(x)
        output2 = self.forward(x)
        return output1['corrupted'], output2['corrupted']


class ReConTabCorruption(BaseCorruption):
    """
    ReConTab (Reconstruction-based Contrastive Learning for Tabular Data) corruption strategy.

    Based on reconstruction-based self-supervised learning approaches for tabular data.
    Combines multiple corruption techniques optimized for reconstruction tasks.
    """

    def __init__(
        self,
        corruption_rate: float = 0.15,
        categorical_indices: Optional[List[int]] = None,
        numerical_indices: Optional[List[int]] = None,
        corruption_types: List[str] = ["masking", "noise", "swapping"],
        masking_strategy: str = "random",  # "random", "column_wise", "block"
        noise_std: float = 0.1,
        swap_probability: float = 0.1,
        **kwargs
    ):
        super().__init__(corruption_rate=corruption_rate, **kwargs)
        self.categorical_indices = categorical_indices or []
        self.numerical_indices = numerical_indices or []
        self.corruption_types = corruption_types
        self.masking_strategy = masking_strategy
        self.noise_std = noise_std
        self.swap_probability = swap_probability

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply ReConTab corruption with multiple strategies.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)

        Returns:
            Dictionary containing:
            - 'corrupted': Corrupted input with same shape as x
            - 'metadata': Information about applied corruptions for reconstruction
            - 'targets': Original tensor for reconstruction
        """
        if not self.training:
            return {
                'corrupted': x,
                'metadata': torch.zeros_like(x),
                'targets': x
            }

        batch_size, seq_len, num_features = x.shape
        device = x.device

        x_corrupted = x.clone()
        corruption_info = torch.zeros_like(
            x
        )  # 0=no corruption, 1=masked, 2=noise, 3=swapped

        # Apply each corruption type
        for corruption_idx, corruption_type in enumerate(self.corruption_types):
            if corruption_type == "masking":
                mask = self._generate_mask(x, strategy=self.masking_strategy)
                x_corrupted = x_corrupted * (1 - mask)  # Zero out masked positions
                corruption_info[mask.bool()] = 1

            elif corruption_type == "noise" and self.numerical_indices:
                noise_mask = (
                    torch.rand(batch_size, seq_len, num_features, device=device)
                    < self.corruption_rate
                )

                # Apply noise only to numerical features
                for feat_idx in self.numerical_indices:
                    if feat_idx < num_features:
                        feature_mask = noise_mask[:, :, feat_idx]
                        noise = (
                            torch.randn_like(x_corrupted[:, :, feat_idx])
                            * self.noise_std
                        )
                        x_corrupted[:, :, feat_idx] = torch.where(
                            feature_mask,
                            x_corrupted[:, :, feat_idx] + noise,
                            x_corrupted[:, :, feat_idx],
                        )
                        corruption_info[:, :, feat_idx][feature_mask] = 2

            elif corruption_type == "swapping":
                swap_mask = (
                    torch.rand(num_features, device=device) < self.swap_probability
                )

                for feat_idx in range(num_features):
                    if swap_mask[feat_idx]:
                        # Randomly permute this feature across batch
                        perm_indices = torch.randperm(batch_size, device=device)
                        x_corrupted[:, :, feat_idx] = x_corrupted[
                            perm_indices, :, feat_idx
                        ]
                        corruption_info[:, :, feat_idx] = 3

        return {
            'corrupted': x_corrupted,
            'metadata': corruption_info,  # For reconstruction_targets method
            'targets': x,  # Original for reconstruction
        }

    def _generate_mask(self, x: torch.Tensor, strategy: str = "random") -> torch.Tensor:
        """Generate corruption mask based on strategy."""
        batch_size, seq_len, num_features = x.shape
        device = x.device

        if strategy == "random":
            # Random masking
            mask = torch.bernoulli(
                torch.full(
                    (batch_size, seq_len, num_features),
                    self.corruption_rate,
                    device=device,
                )
            )

        elif strategy == "column_wise":
            # Randomly select columns to mask entirely
            num_cols_to_mask = max(1, int(num_features * self.corruption_rate))
            cols_to_mask = torch.randperm(num_features)[:num_cols_to_mask]

            mask = torch.zeros(batch_size, seq_len, num_features, device=device)
            mask[:, :, cols_to_mask] = 1

        elif strategy == "block":
            # Block-wise masking (mask contiguous regions)
            mask = torch.zeros(batch_size, seq_len, num_features, device=device)

            for batch_idx in range(batch_size):
                # Random block size
                block_size = max(1, int(seq_len * self.corruption_rate))
                start_pos = torch.randint(
                    0, max(1, seq_len - block_size + 1), (1,)
                ).item()

                # Random features to apply block mask
                num_feat_mask = max(1, int(num_features * 0.5))
                feat_indices = torch.randperm(num_features)[:num_feat_mask]

                mask[batch_idx, start_pos : start_pos + block_size, feat_indices] = 1

        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")

        return mask

    def reconstruction_targets(
        self,
        original: torch.Tensor,
        corrupted: torch.Tensor,
        corruption_info: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Create reconstruction targets for different corruption types.

        Args:
            original: Original uncorrupted data
            corrupted: Corrupted data
            corruption_info: Information about applied corruptions

        Returns:
            Dictionary with reconstruction targets for each corruption type
        """
        targets = {}

        # Masking reconstruction: predict original values at masked positions
        mask_positions = corruption_info == 1
        if mask_positions.any():
            targets["masked_values"] = original[mask_positions]
            targets["mask_positions"] = mask_positions

        # Noise reconstruction: predict clean values at noisy positions
        noise_positions = corruption_info == 2
        if noise_positions.any():
            targets["denoised_values"] = original[noise_positions]
            targets["noise_positions"] = noise_positions

        # Swap reconstruction: predict original values at swapped positions
        swap_positions = corruption_info == 3
        if swap_positions.any():
            targets["unswapped_values"] = original[swap_positions]
            targets["swap_positions"] = swap_positions

        return targets
