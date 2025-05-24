import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import multiprocessing as mp
from functools import partial
import warnings

logger = logging.getLogger(__name__)


class TransactionDataGenerator:
    def __init__(
        self,
        n_entities: int = 1000,
        n_transactions: int = 100000,
        start_date: str = "2023-01-01",
        end_date: str = "2023-12-31",
        seed: int = 42,
        n_jobs: int = -1,
    ):
        """Initialize the transaction data generator.

        Args:
            n_entities: Number of unique entities (e.g., customers)
            n_transactions: Total number of transactions to generate
            start_date: Start date for transactions
            end_date: End date for transactions
            seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.n_entities = n_entities
        self.n_transactions = n_transactions
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.seed = seed
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        
        # Set random seed
        np.random.seed(seed)
        
        # Cache for random values
        self._cache: Dict[str, np.ndarray] = {}

        # Define transaction categories and their probabilities
        self.categories = {
            "groceries": 0.3,
            "dining": 0.2,
            "transportation": 0.15,
            "entertainment": 0.1,
            "shopping": 0.15,
            "utilities": 0.1,
        }

        # Define merchant types and their probabilities
        self.merchant_types = {
            "retail": 0.4,
            "service": 0.3,
            "online": 0.2,
            "subscription": 0.1,
        }

        # Define risk levels and their probabilities
        self.risk_levels = {"low": 0.7, "medium": 0.2, "high": 0.1}

    def _get_cached_random(self, key: str, generator: callable, *args, **kwargs) -> np.ndarray:
        """Get or generate cached random values."""
        if key not in self._cache:
            self._cache[key] = generator(*args, **kwargs)
        return self._cache[key]

    def generate_entity_data(self) -> pd.DataFrame:
        """Generate entity (customer) data with features using vectorized operations."""
        # Generate entity IDs
        entity_ids = np.array([f"ENT_{i:06d}" for i in range(self.n_entities)])

        # Generate entity features using vectorized operations
        data = {
            "entity_id": entity_ids,
            "age": self._get_cached_random("age", np.random.randint, 18, 80, size=self.n_entities),
            "income_level": self._get_cached_random(
                "income_level",
                np.random.choice,
                ["low", "medium", "high"],
                size=self.n_entities,
                p=[0.3, 0.5, 0.2]
            ),
            "credit_score": self._get_cached_random("credit_score", np.random.randint, 300, 850, size=self.n_entities),
            "account_age_days": self._get_cached_random("account_age", np.random.randint, 0, 3650, size=self.n_entities),
            "risk_level": self._get_cached_random(
                "risk_level",
                np.random.choice,
                list(self.risk_levels.keys()),
                size=self.n_entities,
                p=list(self.risk_levels.values())
            ),
        }

        return pd.DataFrame(data)

    def _generate_timestamps_chunk(self, chunk_size: int, start_date: datetime) -> List[datetime]:
        """Generate a chunk of timestamps."""
        timestamps = []
        current_date = start_date
        
        while len(timestamps) < chunk_size:
            if current_date > self.end_date:
                break

            # Generate more transactions during business hours
            n_transactions = np.random.poisson(100 if current_date.weekday() < 5 else 50)
            
            # Vectorized hour generation
            hours = np.clip(np.random.normal(14, 3, size=n_transactions), 0, 23).astype(int)
            minutes = np.random.randint(0, 60, size=n_transactions)
            
            for hour, minute in zip(hours, minutes):
                timestamp = current_date.replace(hour=hour, minute=minute)
                timestamps.append(timestamp)

            current_date += timedelta(days=1)

        return timestamps[:chunk_size]

    def _generate_amounts_vectorized(self, categories: np.ndarray) -> np.ndarray:
        """Generate transaction amounts using vectorized operations."""
        # Define distribution parameters for each category
        dist_params = {
            "groceries": (3.5, 0.5),
            "dining": (3.0, 0.6),
            "transportation": (2.5, 0.4),
            "entertainment": (3.2, 0.7),
            "shopping": (3.8, 0.8),
            "utilities": (4.0, 0.3),
        }
        
        # Generate amounts for each category
        amounts = np.zeros(len(categories))
        for category, (mu, sigma) in dist_params.items():
            mask = categories == category
            amounts[mask] = np.random.lognormal(mu, sigma, size=mask.sum())
        
        return np.round(amounts, 2)

    def generate_transaction_data(self) -> pd.DataFrame:
        """Generate transaction data with realistic patterns using vectorized operations."""
        # Generate transaction IDs
        transaction_ids = np.array([f"TXN_{i:08d}" for i in range(self.n_transactions)])

        # Generate entity IDs with some entities having more transactions
        entity_weights = self._get_cached_random("entity_weights", np.random.power, 2, size=self.n_entities)
        entity_weights = entity_weights / entity_weights.sum()
        entity_ids = np.random.choice(
            [f"ENT_{i:06d}" for i in range(self.n_entities)],
            size=self.n_transactions,
            p=entity_weights,
        )

        # Generate timestamps in parallel
        chunk_size = self.n_transactions // self.n_jobs
        with mp.Pool(self.n_jobs) as pool:
            timestamps_chunks = pool.map(
                partial(self._generate_timestamps_chunk, chunk_size=chunk_size),
                [self.start_date + timedelta(days=i) for i in range(self.n_jobs)]
            )
        timestamps = [ts for chunk in timestamps_chunks for ts in chunk]
        timestamps = timestamps[:self.n_transactions]
        timestamps.sort()

        # Generate categories
        categories = np.random.choice(
            list(self.categories.keys()),
            size=self.n_transactions,
            p=list(self.categories.values())
        )

        # Generate amounts using vectorized operations
        amounts = self._generate_amounts_vectorized(categories)

        # Generate other features using vectorized operations
        merchant_types = np.random.choice(
            list(self.merchant_types.keys()),
            size=self.n_transactions,
            p=list(self.merchant_types.values())
        )

        locations = np.random.choice(
            ["domestic", "international"],
            size=self.n_transactions,
            p=[0.9, 0.1]
        )

        statuses = np.random.choice(
            ["completed", "failed", "pending"],
            size=self.n_transactions,
            p=[0.95, 0.03, 0.02]
        )

        is_fraud = np.random.choice(
            [0, 1],
            size=self.n_transactions,
            p=[0.99, 0.01]
        )

        # Create transaction dataframe
        data = {
            "transaction_id": transaction_ids,
            "entity_id": entity_ids,
            "timestamp": timestamps,
            "amount": amounts,
            "category": categories,
            "merchant_type": merchant_types,
            "location": locations,
            "status": statuses,
            "is_fraud": is_fraud,
        }

        return pd.DataFrame(data)

    def generate_data(self, output_dir: str) -> None:
        """Generate and save sample data efficiently.

        Args:
            output_dir: Directory to save the generated data
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate entity data
        entity_df = self.generate_entity_data()
        entity_path = output_dir / "entities.parquet"
        entity_df.to_parquet(entity_path, index=False)
        logger.info(f"Generated entity data: {entity_path}")

        # Generate transaction data
        transaction_df = self.generate_transaction_data()
        transaction_path = output_dir / "transactions.parquet"
        transaction_df.to_parquet(transaction_path, index=False)
        logger.info(f"Generated transaction data: {transaction_path}")

        # Create train/val/test splits efficiently
        transaction_df = transaction_df.sort_values("timestamp")
        n = len(transaction_df)
        train_idx = int(0.7 * n)
        val_idx = int(0.85 * n)

        # Split data
        train_df = transaction_df[:train_idx]
        val_df = transaction_df[train_idx:val_idx]
        test_df = transaction_df[val_idx:]

        # Save splits
        train_path = output_dir / "train.parquet"
        val_path = output_dir / "val.parquet"
        test_path = output_dir / "test.parquet"

        # Save dataframes without index for better performance
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        logger.info("Created data splits:")
        logger.info(f"  Train: {train_path} ({len(train_df)} samples)")
        logger.info(f"  Val: {val_path} ({len(val_df)} samples)")
        logger.info(f"  Test: {test_path} ({len(test_df)} samples)")

        # Clear cache
        self._cache.clear()


def download_sample_data(output_dir: str = "data/sample", n_jobs: int = -1) -> None:
    """Download or generate sample data for the project.

    Args:
        output_dir: Directory to save the sample data
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    generator = TransactionDataGenerator(n_jobs=n_jobs)
    generator.generate_data(output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_sample_data()
