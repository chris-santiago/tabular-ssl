"""
Sample data utilities for Tabular SSL.

This module provides functions to download and process sample datasets
for experimentation with the Tabular SSL library.
"""

import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm


class DownloadProgressHook:
    """Progress hook for urllib downloads with tqdm."""

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            )

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.update(total_size - self.pbar.n)
            self.pbar.close()


def download_credit_card_transactions(
    data_dir: str = "data", force_download: bool = False
) -> str:
    """
    Download IBM TabFormer credit card transaction dataset.

    Downloads and extracts the credit card transaction dataset from the
    IBM TabFormer repository. This dataset contains sequential transaction
    data that's perfect for experimenting with tabular SSL methods.

    Args:
        data_dir: Directory to store the downloaded data (default: "data")
        force_download: Whether to re-download if data already exists

    Returns:
        Path to the extracted data directory

    Raises:
        urllib.error.URLError: If download fails
        tarfile.TarError: If extraction fails
    """
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Paths
    tgz_path = data_path / "transactions.tgz"
    extracted_dir = data_path / "credit_card"

    # Check if already exists
    if extracted_dir.exists() and not force_download:
        print(f"✅ Credit card data already exists at {extracted_dir}")
        return str(extracted_dir)

    # Download URL
    url = "https://github.com/IBM/TabFormer/raw/main/data/credit_card/transactions.tgz"

    try:
        print("📥 Downloading IBM TabFormer credit card transaction data...")
        print(f"Source: {url}")

        # Download with progress bar
        progress_hook = DownloadProgressHook()
        urllib.request.urlretrieve(url, tgz_path, reporthook=progress_hook)

        print(f"💾 Downloaded to {tgz_path}")

        # Extract the tar.gz file
        print("📦 Extracting archive...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(data_path)

        # Clean up the downloaded archive
        tgz_path.unlink()

        print(f"✅ Data extracted to {extracted_dir}")

        # Show contents
        if extracted_dir.exists():
            files = list(extracted_dir.glob("*"))
            print(f"📂 Files available: {[f.name for f in files]}")

        return str(extracted_dir)

    except urllib.error.URLError as e:
        raise urllib.error.URLError(f"Failed to download data: {e}")
    except tarfile.TarError as e:
        raise tarfile.TarError(f"Failed to extract archive: {e}")


def load_credit_card_sample(
    data_dir: str = "data",
    n_users: Optional[int] = 1000,
    min_transactions: int = 10,
    max_transactions: int = 100,
) -> Tuple[pd.DataFrame, dict]:
    """
    Load and preprocess credit card transaction data for tabular SSL.

    Loads the IBM TabFormer credit card dataset and preprocesses it into
    a format suitable for sequence-based tabular SSL experiments.

    Args:
        data_dir: Directory containing the data
        n_users: Number of users to include (None for all)
        min_transactions: Minimum transactions per user
        max_transactions: Maximum transactions per user

    Returns:
        Tuple of (DataFrame, metadata_dict) where:
        - DataFrame contains user transaction sequences
        - metadata_dict contains feature information

    Raises:
        FileNotFoundError: If data files are not found
        ValueError: If data cannot be processed
    """
    # Ensure data is downloaded
    extracted_dir = Path(download_credit_card_transactions(data_dir))

    # Look for transaction files
    csv_files = list(extracted_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {extracted_dir}")

    # Load the main transactions file
    transactions_file = None
    for f in csv_files:
        if "transaction" in f.name.lower():
            transactions_file = f
            break

    if transactions_file is None:
        # Use the first CSV file as fallback
        transactions_file = csv_files[0]

    print(f"📊 Loading transactions from {transactions_file.name}")

    try:
        # Load data
        df = pd.read_csv(transactions_file)
        print(f"📈 Loaded {len(df)} transactions")

        # Basic preprocessing
        df = _preprocess_transactions(df, n_users, min_transactions, max_transactions)

        # Create metadata
        metadata = _create_metadata(df)

        print(
            f"✅ Prepared {len(df)} transaction sequences for {metadata['n_users']} users"
        )

        return df, metadata

    except Exception as e:
        raise ValueError(f"Failed to process transaction data: {e}")


def _preprocess_transactions(
    df: pd.DataFrame,
    n_users: Optional[int],
    min_transactions: int,
    max_transactions: int,
) -> pd.DataFrame:
    """Preprocess raw transaction data."""

    # Identify user and transaction columns
    user_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ["user", "customer", "card", "account"]):
            user_col = col
            break

    if user_col is None:
        # Create user IDs if not present
        print("⚠️  No user column found, creating user groups...")
        df["user_id"] = np.random.randint(0, 10000, len(df))
        user_col = "user_id"

    # Identify timestamp column
    time_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ["time", "date", "timestamp"]):
            time_col = col
            break

    if time_col is not None:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values([user_col, time_col])
        except:
            print("⚠️  Could not parse timestamp column")

    # Filter users by transaction count
    user_counts = df[user_col].value_counts()
    valid_users = user_counts[
        (user_counts >= min_transactions) & (user_counts <= max_transactions)
    ].index

    df = df[df[user_col].isin(valid_users)]

    # Sample users if requested
    if n_users is not None and len(valid_users) > n_users:
        sampled_users = np.random.choice(valid_users, n_users, replace=False)
        df = df[df[user_col].isin(sampled_users)]

    # Reset index
    df = df.reset_index(drop=True)

    return df


def _create_metadata(df: pd.DataFrame) -> dict:
    """Create metadata dictionary for the dataset."""

    # Identify categorical and numerical columns
    categorical_features = []
    numerical_features = []

    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            # Skip if too many unique values (likely not categorical)
            if df[col].nunique() < len(df) * 0.5:
                categorical_features.append(
                    {
                        "name": col,
                        "num_categories": df[col].nunique(),
                        "categories": df[col]
                        .unique()
                        .tolist()[:10],  # First 10 for display
                    }
                )
        elif np.issubdtype(df[col].dtype, np.number):
            numerical_features.append(
                {
                    "name": col,
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }
            )

    # Count users
    user_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ["user", "customer", "card", "account"]):
            user_col = col
            break

    n_users = df[user_col].nunique() if user_col else 1

    return {
        "dataset_name": "credit_card_transactions",
        "n_transactions": len(df),
        "n_users": n_users,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "columns": df.columns.tolist(),
    }


def generate_sample_sequences(
    df: pd.DataFrame, sequence_length: int = 32, target_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert transaction data into fixed-length sequences for training.

    Takes the preprocessed transaction data and creates fixed-length
    sequences suitable for sequence-based tabular SSL models.

    Args:
        df: Preprocessed transaction DataFrame
        sequence_length: Length of each sequence
        target_column: Column to use as prediction target

    Returns:
        DataFrame with sequences ready for training
    """
    # Identify user column
    user_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ["user", "customer", "card", "account"]):
            user_col = col
            break

    if user_col is None:
        raise ValueError("No user column found in the data")

    sequences = []

    for user_id in df[user_col].unique():
        user_data = df[df[user_col] == user_id].copy()

        # Create sequences for this user
        for i in range(0, len(user_data) - sequence_length + 1, sequence_length // 2):
            sequence = user_data.iloc[i : i + sequence_length].copy()

            if len(sequence) == sequence_length:
                # Add sequence metadata
                sequence["sequence_id"] = f"{user_id}_{i}"
                sequence["position_in_sequence"] = range(sequence_length)

                # Add target if specified
                if target_column and target_column in sequence.columns:
                    # Use last transaction's target as sequence target
                    sequence["sequence_target"] = sequence[target_column].iloc[-1]

                sequences.append(sequence)

    if sequences:
        result = pd.concat(sequences, ignore_index=True)
        print(f"📝 Created {len(sequences)} sequences of length {sequence_length}")
        return result
    else:
        raise ValueError("No valid sequences could be created")


# Convenience function for quick setup
def setup_sample_data(
    data_dir: str = "data", sequence_length: int = 32, n_users: int = 100
) -> Tuple[pd.DataFrame, dict]:
    """
    Quick setup function to download and prepare sample data.

    This is a convenience function that downloads the IBM TabFormer
    credit card data and prepares it for immediate use with Tabular SSL.

    Args:
        data_dir: Directory to store data
        sequence_length: Length of transaction sequences
        n_users: Number of users to include

    Returns:
        Tuple of (sequence_df, metadata)
    """
    print("🚀 Setting up sample transaction data...")

    # Download and load data
    df, metadata = load_credit_card_sample(
        data_dir=data_dir,
        n_users=n_users,
        min_transactions=sequence_length,
        max_transactions=200,
    )

    # Generate sequences
    sequence_df = generate_sample_sequences(df, sequence_length=sequence_length)

    # Update metadata
    metadata["sequence_length"] = sequence_length
    metadata["n_sequences"] = len(sequence_df) // sequence_length

    print("✅ Sample data ready for training!")
    return sequence_df, metadata


if __name__ == "__main__":
    # Demo usage
    print("📦 Tabular SSL Sample Data Demo")
    print("=" * 40)

    # Setup sample data
    df, metadata = setup_sample_data(n_users=50, sequence_length=16)

    print("\n📊 Dataset Overview:")
    print(f"  • Total transactions: {metadata['n_transactions']}")
    print(f"  • Number of users: {metadata['n_users']}")
    print(f"  • Sequence length: {metadata['sequence_length']}")
    print(f"  • Number of sequences: {metadata['n_sequences']}")

    print("\n🏷️  Features:")
    print(f"  • Categorical: {len(metadata['categorical_features'])}")
    print(f"  • Numerical: {len(metadata['numerical_features'])}")

    print(f"\n📋 Sample data shape: {df.shape}")
    print(f"📋 Sample columns: {df.columns.tolist()[:10]}...")
