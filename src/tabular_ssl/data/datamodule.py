import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Type, ClassVar, Sequence
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass, field
import pandas as pd
import pytorch_lightning as lit

log = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration for a single processor in the pipeline."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureConfig:
    """Configuration for feature processing."""

    categorical_cols: List[str]
    numerical_cols: List[str]
    target_col: Optional[str] = None
    categorical_encoding: str = "embedding"
    normalize_numerical: bool = True
    processors: List[ProcessorConfig] = field(default_factory=list)


class FeatureProcessorRegistry:
    """Registry for feature processors."""

    _processors: ClassVar[Dict[str, Type["FeatureProcessor"]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Register a feature processor class."""

        def decorator(
            processor_class: Type["FeatureProcessor"],
        ) -> Type["FeatureProcessor"]:
            cls._processors[name] = processor_class
            return processor_class

        return decorator

    @classmethod
    def get_processor(cls, name: str) -> Type["FeatureProcessor"]:
        """Get a feature processor class by name."""
        if name not in cls._processors:
            raise ValueError(f"Unknown feature processor: {name}")
        return cls._processors[name]

    @classmethod
    def list_processors(cls) -> List[str]:
        """List all registered feature processors."""
        return list(cls._processors.keys())


class FeatureProcessor(ABC):
    """Abstract base class for feature processing."""

    @abstractmethod
    def fit(self, data: pl.DataFrame) -> None:
        """Fit the processor to the data."""
        pass

    @abstractmethod
    def transform(self, data: pl.DataFrame) -> Dict[str, torch.Tensor]:
        """Transform the data into tensors."""
        pass

    @abstractmethod
    def get_feature_dims(self) -> Dict[str, int]:
        """Get the dimensions of the processed features."""
        pass


class ProcessorPipeline(FeatureProcessor):
    """Pipeline of feature processors."""

    def __init__(self, processors: Sequence[FeatureProcessor]):
        self.processors = processors
        self._feature_dims: Optional[Dict[str, int]] = None

    def fit(self, data: pl.DataFrame) -> None:
        """Fit all processors in sequence."""
        current_data = data
        for processor in self.processors:
            processor.fit(current_data)
            # Update data for next processor if needed
            if hasattr(processor, "get_processed_data"):
                current_data = processor.get_processed_data()

    def transform(self, data: pl.DataFrame) -> Dict[str, torch.Tensor]:
        """Transform data through all processors."""
        current_data = data
        result = {}

        for processor in self.processors:
            # Transform current data
            processor_output = processor.transform(current_data)

            # Update result with processor output
            for key, value in processor_output.items():
                if key not in result:
                    result[key] = value
                else:
                    # Handle overlapping keys (e.g., concatenate features)
                    if isinstance(value, torch.Tensor) and isinstance(
                        result[key], torch.Tensor
                    ):
                        if value.dim() == result[key].dim():
                            result[key] = torch.cat([result[key], value], dim=-1)
                        else:
                            log.warning(f"Could not concatenate tensors for key {key}")

            # Update data for next processor if needed
            if hasattr(processor, "get_processed_data"):
                current_data = processor.get_processed_data()

        return result

    def get_feature_dims(self) -> Dict[str, int]:
        """Get combined feature dimensions from all processors."""
        if self._feature_dims is None:
            dims = {}
            for processor in self.processors:
                processor_dims = processor.get_feature_dims()
                for key, value in processor_dims.items():
                    if key not in dims:
                        dims[key] = value
                    else:
                        # Sum dimensions for overlapping keys
                        dims[key] += value
            self._feature_dims = dims
        return self._feature_dims


@FeatureProcessorRegistry.register("tabular")
class TabularFeatureProcessor(FeatureProcessor):
    """Processor for tabular features."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.categorical_mappings: Dict[str, Dict[Any, int]] = {}
        self.scaler: Optional[StandardScaler] = None
        self._feature_dims: Optional[Dict[str, int]] = None
        self._processed_data: Optional[pl.DataFrame] = None

    def fit(self, data: pl.DataFrame) -> None:
        """Fit categorical mappings and numerical scaler."""
        # Fit categorical mappings
        if self.config.categorical_encoding == "onehot":
            for col in self.config.categorical_cols:
                unique_values = data[col].unique()
                self.categorical_mappings[col] = {
                    val: idx for idx, val in enumerate(unique_values)
                }

        # Fit numerical scaler
        if self.config.normalize_numerical and self.config.numerical_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(data[self.config.numerical_cols])

        # Compute feature dimensions
        self._compute_feature_dims(data)

        # Store processed data
        self._processed_data = data

    def get_processed_data(self) -> pl.DataFrame:
        """Get the processed data for the next processor."""
        if self._processed_data is None:
            raise RuntimeError("Processor must be fitted before getting processed data")
        return self._processed_data

    def _compute_feature_dims(self, data: pl.DataFrame) -> None:
        """Compute the dimensions of processed features."""
        dims = {}

        # Categorical feature dimensions
        if self.config.categorical_encoding == "onehot":
            total_cat_dim = sum(
                len(mapping) for mapping in self.categorical_mappings.values()
            )
            dims["categorical"] = total_cat_dim
        else:
            dims["categorical"] = len(self.config.categorical_cols)

        # Numerical feature dimensions
        dims["numerical"] = len(self.config.numerical_cols)

        self._feature_dims = dims

    def get_feature_dims(self) -> Dict[str, int]:
        """Get the dimensions of the processed features."""
        if self._feature_dims is None:
            raise RuntimeError(
                "Feature processor must be fitted before getting dimensions"
            )
        return self._feature_dims

    def transform(self, data: pl.DataFrame) -> Dict[str, torch.Tensor]:
        """Transform data into tensors."""
        result = {}

        # Process categorical features
        if self.config.categorical_encoding == "onehot":
            categorical_data = []
            for col in self.config.categorical_cols:
                values = data[col].to_numpy()
                encoded = np.zeros((len(values), len(self.categorical_mappings[col])))
                for i, val in enumerate(values):
                    encoded[i, self.categorical_mappings[col][val]] = 1
                categorical_data.append(encoded)
            categorical_data = np.concatenate(categorical_data, axis=1)
            result["categorical"] = torch.FloatTensor(categorical_data)
        else:
            # For embedding encoding
            categorical_data = []
            for col in self.config.categorical_cols:
                values = data[col].to_numpy()
                encoded = np.array(
                    [self.categorical_mappings[col][val] for val in values]
                )
                categorical_data.append(encoded)
            result["categorical"] = torch.LongTensor(np.stack(categorical_data, axis=1))

        # Process numerical features
        if self.config.numerical_cols:
            if self.scaler is not None:
                numerical_data = self.scaler.transform(data[self.config.numerical_cols])
            else:
                numerical_data = data[self.config.numerical_cols].to_numpy()
            result["numerical"] = torch.FloatTensor(numerical_data)
        else:
            result["numerical"] = torch.zeros((len(data), 0))

        # Process target
        if self.config.target_col is not None:
            result["target"] = torch.LongTensor(data[self.config.target_col].to_numpy())
        else:
            result["target"] = torch.zeros(len(data), dtype=torch.long)

        # Update processed data
        self._processed_data = data

        return result


class BaseSequenceDataset(Dataset, ABC):
    """Base class for sequence datasets."""

    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pass


class EventSequenceDataset(BaseSequenceDataset):
    """Dataset for creating sequences of events from tabular data."""

    def __init__(
        self,
        data: pl.DataFrame,
        sequence_length: int,
        feature_processor: FeatureProcessor,
    ):
        super().__init__(sequence_length)
        self.data = data
        self.feature_processor = feature_processor
        self.n_samples = len(data) - sequence_length + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get sequence window
        sequence = self.data.slice(idx, self.sequence_length)
        return self.feature_processor.transform(sequence)


class TabularDataModule(lit.LightningDataModule):
    """DataModule for handling tabular data with event sequences."""

    def __init__(
        self,
        data_dir: str,
        train_file: str,
        val_file: str,
        test_file: str,
        feature_config: FeatureConfig,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        sequence_length: int = 100,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.feature_config = feature_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sequence_length = sequence_length

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.feature_processor = None

    def _create_processor_pipeline(self) -> FeatureProcessor:
        """Create a pipeline of feature processors."""
        processors = []

        # Add default tabular processor if no processors specified
        if not self.feature_config.processors:
            processors.append(TabularFeatureProcessor(self.feature_config))
        else:
            # Create processors in sequence
            for proc_config in self.feature_config.processors:
                processor_class = FeatureProcessorRegistry.get_processor(
                    proc_config.name
                )
                processor = processor_class(self.feature_config)
                processors.append(processor)

        return ProcessorPipeline(processors)

    def prepare_data(self) -> None:
        """Download data if needed."""
        # This is where you would download data if needed
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and prepare datasets."""
        if stage == "fit" or stage is None:
            # Load training data
            train_data = pl.read_parquet(os.path.join(self.data_dir, self.train_file))

            # Initialize and fit feature processor pipeline
            self.feature_processor = self._create_processor_pipeline()
            self.feature_processor.fit(train_data)

            # Create datasets
            self.train_dataset = EventSequenceDataset(
                data=train_data,
                sequence_length=self.sequence_length,
                feature_processor=self.feature_processor,
            )

            # Load validation data
            val_data = pl.read_parquet(os.path.join(self.data_dir, self.val_file))
            self.val_dataset = EventSequenceDataset(
                data=val_data,
                sequence_length=self.sequence_length,
                feature_processor=self.feature_processor,
            )

        if stage == "test":
            # Load test data
            test_data = pl.read_parquet(os.path.join(self.data_dir, self.test_file))
            self.test_dataset = EventSequenceDataset(
                data=test_data,
                sequence_length=self.sequence_length,
                feature_processor=self.feature_processor,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
