import os
from typing import Optional, Dict, List, Any
import polars as pl
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from dataclasses import dataclass, field
import pytorch_lightning as lit

from .sample_data import setup_sample_data, load_credit_card_sample

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
        data_dir: str = "data",
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        feature_config: Optional[FeatureConfig] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        sequence_length: int = 100,
        # Sample data configuration
        use_sample_data: bool = False,
        sample_data_config: Optional[Dict[str, Any]] = None,
        train_val_test_split: List[float] = [0.7, 0.15, 0.15],
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sequence_length = sequence_length
        self.use_sample_data = use_sample_data
        self.sample_data_config = sample_data_config or {}
        self.train_val_test_split = train_val_test_split
        self.seed = seed

        # Will be set during setup
        self.feature_config = feature_config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.feature_processor = None

    def prepare_data(self) -> None:
        """Download data if needed."""
        if self.use_sample_data:
            # Download sample data (credit card transactions)
            if self.sample_data_config.get("data_source") == "credit_card":
                try:
                    load_credit_card_sample(data_dir=self.data_dir)
                    log.info("Sample credit card data downloaded successfully")
                except Exception as e:
                    log.error(f"Failed to download sample data: {e}")
                    raise

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and prepare datasets."""
        if stage == "fit" or stage is None:
            if self.use_sample_data:
                # Load sample data
                data = self._load_sample_data()
            else:
                # Load custom data from files
                data = self._load_custom_data()

            # Auto-detect features if needed
            if self.feature_config is None:
                self.feature_config = self._auto_detect_features(data)

            # Split data
            train_data, val_data, test_data = self._split_data(data)

            # Initialize and fit feature processor
            self.feature_processor = TabularFeatureProcessor(self.feature_config)
            self.feature_processor.fit(train_data)

            # Create datasets
            self.train_dataset = SequenceDataset(
                data=train_data,
                sequence_length=self.sequence_length,
                feature_processor=self.feature_processor,
            )

            self.val_dataset = SequenceDataset(
                data=val_data,
                sequence_length=self.sequence_length,
                feature_processor=self.feature_processor,
            )

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

    def _load_sample_data(self) -> pl.DataFrame:
        """Load sample data (credit card transactions)."""
        if self.sample_data_config.get("data_source") == "credit_card":
            # Load credit card sample data
            df, metadata = load_credit_card_sample(
                data_dir=self.data_dir,
                n_users=self.sample_data_config.get("n_users", 1000),
                min_transactions=self.sequence_length,
                max_transactions=200
            )
            
            log.info(f"Loaded sample data: {len(df)} transactions for {metadata['n_users']} users")
            
            # Convert pandas to polars
            return pl.from_pandas(df)
        else:
            raise ValueError(f"Unknown sample data source: {self.sample_data_config.get('data_source')}")

    def _load_custom_data(self) -> pl.DataFrame:
        """Load custom data from files."""
        if not self.train_file:
            raise ValueError("train_file must be specified when not using sample data")
        
        train_path = os.path.join(self.data_dir, self.train_file)
        if train_path.endswith('.parquet'):
            data = pl.read_parquet(train_path)
        elif train_path.endswith('.csv'):
            data = pl.read_csv(train_path)
        else:
            raise ValueError(f"Unsupported file format: {train_path}")
        
        return data

    def _auto_detect_features(self, data: pl.DataFrame) -> FeatureConfig:
        """Auto-detect categorical and numerical features."""
        categorical_cols = []
        numerical_cols = []
        
        for col in data.columns:
            # Skip common metadata columns
            if col.lower() in ['user_id', 'timestamp', 'sequence_id', 'position_in_sequence']:
                continue
                
            dtype = data[col].dtype
            
            if dtype == pl.String or dtype == pl.Categorical:
                # Consider categorical if not too many unique values
                n_unique = data[col].n_unique()
                if n_unique < len(data) * 0.5:  # Less than 50% unique values
                    categorical_cols.append(col)
            elif dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                numerical_cols.append(col)
        
        log.info(f"Auto-detected features: {len(categorical_cols)} categorical, {len(numerical_cols)} numerical")
        
        return FeatureConfig(
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            categorical_encoding="embedding",
            normalize_numerical=True
        )

    def _split_data(self, data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split data into train/val/test sets."""
        np.random.seed(self.seed)
        
        # Convert to pandas for splitting (easier with sklearn)
        df = data.to_pandas()
        
        # First split: train vs (val + test)
        train_size = self.train_val_test_split[0]
        val_test_size = 1 - train_size
        
        train_df, val_test_df = train_test_split(
            df, 
            train_size=train_size, 
            random_state=self.seed,
            shuffle=True
        )
        
        # Second split: val vs test
        val_size = self.train_val_test_split[1] / val_test_size
        
        val_df, test_df = train_test_split(
            val_test_df,
            train_size=val_size,
            random_state=self.seed,
            shuffle=True
        )
        
        log.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        # Convert back to polars
        return (
            pl.from_pandas(train_df),
            pl.from_pandas(val_df),
            pl.from_pandas(test_df)
        )
