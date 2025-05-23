from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Type, TypeVar, Generic, ClassVar
import torch
import torch.nn as nn
import pytorch_lightning as pl
from dataclasses import dataclass, field
from pydantic import BaseModel as PydanticBaseModel, Field, validator
import logging
from omegaconf import DictConfig, OmegaConf
import hydra

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseComponent')

class ComponentRegistry:
    """Registry for model components."""
    
    _components: ClassVar[Dict[str, Type['BaseComponent']]] = {}
    
    @classmethod
    def register(cls, name: str) -> Type[T]:
        """Register a component class."""
        def decorator(component_cls: Type[T]) -> Type[T]:
            cls._components[name] = component_cls
            return component_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type['BaseComponent']:
        """Get a component class by name."""
        if name not in cls._components:
            raise KeyError(f"Component {name} not found in registry")
        return cls._components[name]
    
    @classmethod
    def list_components(cls) -> Dict[str, Type['BaseComponent']]:
        """List all registered components."""
        return cls._components.copy()

class ComponentConfig(PydanticBaseModel):
    """Base configuration for components."""
    
    name: str = Field(..., description="Name of the component")
    type: str = Field(..., description="Type of the component")
    
    @validator('type')
    def validate_type(cls, v: str) -> str:
        """Validate that the component type exists in the registry."""
        if v not in ComponentRegistry._components:
            raise ValueError(f"Component type {v} not found in registry")
        return v

    @classmethod
    def from_hydra(cls, config: DictConfig) -> 'ComponentConfig':
        """Create a ComponentConfig from a Hydra config."""
        return cls(**OmegaConf.to_container(config, resolve=True))

class BaseComponent(ABC, nn.Module, Generic[T]):
    """Base class for all model components."""
    
    def __init__(self, config: ComponentConfig):
        super().__init__()
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the component configuration."""
        if not isinstance(self.config, ComponentConfig):
            raise ValueError("Config must be an instance of ComponentConfig")
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the component."""
        pass

@ComponentRegistry.register("event_encoder")
class EventEncoder(BaseComponent):
    """Base class for event encoders."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the event encoder."""
        raise NotImplementedError("Event encoder forward pass must be implemented")

@ComponentRegistry.register("sequence_encoder")
class SequenceEncoder(BaseComponent):
    """Base class for sequence encoders."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the sequence encoder."""
        raise NotImplementedError("Sequence encoder forward pass must be implemented")

@ComponentRegistry.register("embedding")
class EmbeddingLayer(BaseComponent):
    """Base class for embedding layers."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the embedding layer."""
        raise NotImplementedError("Embedding layer forward pass must be implemented")

@ComponentRegistry.register("projection_head")
class ProjectionHead(BaseComponent):
    """Base class for projection heads."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the projection head."""
        raise NotImplementedError("Projection head forward pass must be implemented")

@ComponentRegistry.register("prediction_head")
class PredictionHead(BaseComponent):
    """Base class for prediction heads."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the prediction head."""
        raise NotImplementedError("Prediction head forward pass must be implemented")

class BaseModel(pl.LightningModule):
    """Base model class for self-supervised sequence modeling."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        
        # Convert Hydra configs to ComponentConfigs
        self.event_encoder_config = ComponentConfig.from_hydra(config.model.event_encoder)
        self.sequence_encoder_config = ComponentConfig.from_hydra(config.model.sequence_encoder) if config.model.sequence_encoder else None
        self.embedding_config = ComponentConfig.from_hydra(config.model.embedding) if config.model.embedding else None
        self.projection_head_config = ComponentConfig.from_hydra(config.model.projection_head) if config.model.projection_head else None
        self.prediction_head_config = ComponentConfig.from_hydra(config.model.prediction_head) if config.model.prediction_head else None
        
        # Initialize components
        self.event_encoder = self._init_component(self.event_encoder_config)
        self.sequence_encoder = self._init_component(self.sequence_encoder_config) if self.sequence_encoder_config else None
        self.embedding_layer = self._init_component(self.embedding_config) if self.embedding_config else None
        self.projection_head = self._init_component(self.projection_head_config) if self.projection_head_config else None
        self.prediction_head = self._init_component(self.prediction_head_config) if self.prediction_head_config else None
        
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))
    
    def _init_component(self, config: ComponentConfig) -> BaseComponent:
        """Initialize a component from its configuration."""
        component_cls = ComponentRegistry.get(config.type)
        return component_cls(config)
    
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
        optimizer = hydra.utils.instantiate(self.config.optimizer, params=self.parameters())
        if "scheduler" in self.config:
            scheduler = hydra.utils.instantiate(self.config.scheduler, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                }
            }
        return optimizer


class TabularSSL:
    pass


class TabularSSLConfig:
    pass
