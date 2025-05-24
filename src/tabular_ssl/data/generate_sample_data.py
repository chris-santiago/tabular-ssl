import polars as pl
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List
import multiprocessing as mp
from functools import partial
import random

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
        random.seed(seed)
        pl.set_random_seed(seed)

        # Cache for random values
        self._cache: Dict[str, pl.Series] = {}

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

    def _get_cached_random(
        self, key: str, generator: callable, *args, **kwargs
    ) -> pl.Series:
        """Get or generate cached random values."""
        if key not in self._cache:
            self._cache[key] = generator(*args, **kwargs)
        return self._cache[key]

    def generate_entity_data(self) -> pl.DataFrame:
        """Generate entity (customer) data with features using vectorized operations."""
        # Generate entity IDs
        entity_ids = [f"ENT_{i:06d}" for i in range(self.n_entities)]

        # Generate entity features using vectorized operations
        data = {
            "entity_id": entity_ids,
            "age": pl.Series(random.randint(18, 80) for _ in range(self.n_entities)),
            "income_level": pl.Series(
                random.choices(
                    ["low", "medium", "high"],
                    weights=[0.3, 0.5, 0.2],
                    k=self.n_entities
                )
            ),
            "credit_score": pl.Series(
                random.randint(300, 850) for _ in range(self.n_entities)
            ),
            "account_age_days": pl.Series(
                random.randint(0, 3650) for _ in range(self.n_entities)
            ),
            "risk_level": pl.Series(
                random.choices(
                    list(self.risk_levels.keys()),
                    weights=list(self.risk_levels.values()),
                    k=self.n_entities
                )
            ),
        }

        return pl.DataFrame(data)

    def _generate_timestamps_chunk(
        self, chunk_size: int, start_date: datetime
    ) -> List[datetime]:
        """Generate a chunk of timestamps."""
        timestamps = []
        current_date = start_date

        while len(timestamps) < chunk_size:
            if current_date > self.end_date:
                break

            # Generate more transactions during business hours
            n_transactions = random.randint(50, 150) if current_date.weekday() < 5 else random.randint(20, 80)

            for _ in range(n_transactions):
                hour = min(max(int(random.gauss(14, 3)), 0), 23)
                minute = random.randint(0, 59)
                timestamp = current_date.replace(hour=hour, minute=minute)
                timestamps.append(timestamp)

            current_date += timedelta(days=1)

        return timestamps[:chunk_size]

    def _generate_amounts_vectorized(self, categories: List[str]) -> pl.Series:
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
        amounts = []
        for category in categories:
            mu, sigma = dist_params[category]
            # Generate lognormal distribution using random.gauss
            amount = random.lognormvariate(mu, sigma)
            amounts.append(round(amount, 2))

        return pl.Series(amounts)

    def generate_transaction_data(self) -> pl.DataFrame:
        """Generate transaction data with realistic patterns using vectorized operations."""
        # Generate transaction IDs
        transaction_ids = [f"TXN_{i:08d}" for i in range(self.n_transactions)]

        # Generate entity IDs with some entities having more transactions
        entity_weights = [random.paretovariate(2) for _ in range(self.n_entities)]
        total_weight = sum(entity_weights)
        entity_weights = [w / total_weight for w in entity_weights]
        entity_ids = random.choices(
            [f"ENT_{i:06d}" for i in range(self.n_entities)],
            weights=entity_weights,
            k=self.n_transactions
        )

        # Generate timestamps in parallel
        chunk_size = self.n_transactions // self.n_jobs
        with mp.Pool(self.n_jobs) as pool:
            timestamps_chunks = pool.map(
                partial(self._generate_timestamps_chunk, chunk_size=chunk_size),
                [self.start_date + timedelta(days=i) for i in range(self.n_jobs)],
            )
        timestamps = [ts for chunk in timestamps_chunks for ts in chunk]
        timestamps = timestamps[: self.n_transactions]
        timestamps.sort()

        # Generate categories
        categories = random.choices(
            list(self.categories.keys()),
            weights=list(self.categories.values()),
            k=self.n_transactions
        )

        # Generate amounts using vectorized operations
        amounts = self._generate_amounts_vectorized(categories)

        # Generate other features using vectorized operations
        merchant_types = random.choices(
            list(self.merchant_types.keys()),
            weights=list(self.merchant_types.values()),
            k=self.n_transactions
        )

        locations = random.choices(
            ["domestic", "international"],
            weights=[0.9, 0.1],
            k=self.n_transactions
        )

        statuses = random.choices(
            ["completed", "failed", "pending"],
            weights=[0.95, 0.03, 0.02],
            k=self.n_transactions
        )

        is_fraud = random.choices(
            [0, 1],
            weights=[0.99, 0.01],
            k=self.n_transactions
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

        return pl.DataFrame(data)

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
        entity_df.write_parquet(entity_path)
        logger.info(f"Generated entity data: {entity_path}")

        # Generate transaction data
        transaction_df = self.generate_transaction_data()
        transaction_path = output_dir / "transactions.parquet"
        transaction_df.write_parquet(transaction_path)
        logger.info(f"Generated transaction data: {transaction_path}")

        # Create train/val/test splits efficiently
        transaction_df = transaction_df.sort("timestamp")
        n = len(transaction_df)
        train_idx = int(0.7 * n)
        val_idx = int(0.85 * n)

        # Split data
        train_df = transaction_df.slice(0, train_idx)
        val_df = transaction_df.slice(train_idx, val_idx - train_idx)
        test_df = transaction_df.slice(val_idx, n - val_idx)

        # Save splits
        train_path = output_dir / "train.parquet"
        val_path = output_dir / "val.parquet"
        test_path = output_dir / "test.parquet"

        # Save dataframes without index for better performance
        train_df.write_parquet(train_path)
        val_df.write_parquet(val_path)
        test_df.write_parquet(test_path)

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
