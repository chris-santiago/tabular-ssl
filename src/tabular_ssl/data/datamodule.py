import os
from typing import Optional, Dict, List, Any
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass, field
import pytorch_lightning as lit

log = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature processing."""
    
    categorical_cols: List[str]
    numerical_cols: List[str]
    target_col: Optional[str] = None
    categorical_encoding: str = "embedding"  # "embedding" or "onehot"
    normalize_numerical: bool = True
    categorical_embedding_dims: Dict[str, int] = field(default_factory=dict)
    default_embedding_dim: int = 16


class TabularFeatureProcessor:
    """Processor for tabular features with flexible embedding dimensions."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.categorical_mappings: Dict[str, Dict[Any, int]] = {}
        self.scaler: Optional[StandardScaler] = None
        self._embedding_dims: Dict[str, int] = {}
        self._feature_dims: Optional[Dict[str, int]] = None

    def fit(self, data: pl.DataFrame) -> None:
        """Fit categorical mappings and numerical scaler."""
        # Fit categorical mappings
        for col in self.config.categorical_cols:
            unique_values = data[col].unique()
            self.categorical_mappings[col] = {
                val: idx for idx, val in enumerate(unique_values)
            }
            
            # Set embedding dimensions for each categorical column
            if self.config.categorical_encoding == "embedding":
                self._embedding_dims[col] = self.config.categorical_embedding_dims.get(
                    col, self.config.default_embedding_dim
                )

        # Fit numerical scaler
        if self.config.normalize_numerical and self.config.numerical_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(data[self.config.numerical_cols])

        # Compute feature dimensions
        self._compute_feature_dims()

    def _compute_feature_dims(self) -> None:
        """Compute the dimensions of processed features."""
        dims = {}

        if self.config.categorical_encoding == "onehot":
            dims["categorical"] = sum(
                len(mapping) for mapping in self.categorical_mappings.values()
            )
        else:  # embedding
            dims["categorical"] = sum(self._embedding_dims.values())

        dims["numerical"] = len(self.config.numerical_cols)
        self._feature_dims = dims

    def get_feature_dims(self) -> Dict[str, int]:
        """Get the dimensions of the processed features."""
        if self._feature_dims is None:
            raise RuntimeError("Processor must be fitted before getting dimensions")
        return self._feature_dims

    def get_embedding_dims(self) -> Dict[str, int]:
        """Get embedding dimensions for each categorical column."""
        return self._embedding_dims.copy()

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for each categorical column."""
        return {col: len(mapping) for col, mapping in self.categorical_mappings.items()}

    def transform(self, data: pl.DataFrame) -> Dict[str, torch.Tensor]:
        """Transform data into tensors."""
        result = {}

        # Process categorical features
        if self.config.categorical_cols:
            if self.config.categorical_encoding == "onehot":
                categorical_data = []
                for col in self.config.categorical_cols:
                    values = data[col].to_list()
                    encoded = torch.zeros((len(values), len(self.categorical_mappings[col])))
                    for i, val in enumerate(values):
                        encoded[i, self.categorical_mappings[col][val]] = 1
                    categorical_data.append(encoded)
                result["categorical"] = torch.cat(categorical_data, dim=1)
            else:  # embedding
                categorical_data = []
                for col in self.config.categorical_cols:
                    values = data[col].to_list()
                    encoded = torch.tensor(
                        [self.categorical_mappings[col][val] for val in values],
                        dtype=torch.long
                    )
                    categorical_data.append(encoded)
                result["categorical"] = torch.stack(categorical_data, dim=1)

        # Process numerical features
        if self.config.numerical_cols:
            if self.scaler is not None:
                numerical_data = self.scaler.transform(data[self.config.numerical_cols])
                result["numerical"] = torch.FloatTensor(numerical_data)
            else:
                numerical_data = data[self.config.numerical_cols].to_numpy()
                result["numerical"] = torch.FloatTensor(numerical_data)
        else:
            result["numerical"] = torch.zeros((len(data), 0))

        # Process target
        if self.config.target_col is not None:
            result["target"] = torch.LongTensor(data[self.config.target_col].to_list())

        return result


class SequenceDataset(Dataset):
    """Dataset for creating sequences of events from tabular data."""

    def __init__(
        self,
        data: pl.DataFrame,
        sequence_length: int,
        feature_processor: TabularFeatureProcessor,
    ):
        self.data = data
        self.sequence_length = sequence_length
        self.feature_processor = feature_processor
        self.n_samples = max(0, len(data) - sequence_length + 1)

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

    def prepare_data(self) -> None:
        """Download data if needed."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and prepare datasets."""
        if stage == "fit" or stage is None:
            # Load training data
            train_data = pl.read_parquet(os.path.join(self.data_dir, self.train_file))

            # Initialize and fit feature processor
            self.feature_processor = TabularFeatureProcessor(self.feature_config)
            self.feature_processor.fit(train_data)

            # Create datasets
            self.train_dataset = SequenceDataset(
                data=train_data,
                sequence_length=self.sequence_length,
                feature_processor=self.feature_processor,
            )

            # Load validation data
            val_data = pl.read_parquet(os.path.join(self.data_dir, self.val_file))
            self.val_dataset = SequenceDataset(
                data=val_data,
                sequence_length=self.sequence_length,
                feature_processor=self.feature_processor,
            )

        if stage == "test":
            # Load test data
            test_data = pl.read_parquet(os.path.join(self.data_dir, self.test_file))
            self.test_dataset = SequenceDataset(
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
