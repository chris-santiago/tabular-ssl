import click
import logging

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Tabular SSL CLI tools."""
    pass


@cli.command()
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data/sample",
    help="Directory to save the sample data",
)
@click.option(
    "--n-entities", type=int, default=1000, help="Number of unique entities to generate"
)
@click.option(
    "--n-transactions",
    type=int,
    default=100000,
    help="Total number of transactions to generate",
)
@click.option(
    "--start-date",
    type=str,
    default="2023-01-01",
    help="Start date for transactions (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=str,
    default="2023-12-31",
    help="End date for transactions (YYYY-MM-DD)",
)
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
def download_data(
    output_dir: str,
    n_entities: int,
    n_transactions: int,
    start_date: str,
    end_date: str,
    seed: int,
):
    """Download or generate sample financial transaction data."""
    from tabular_ssl.data.generate_sample_data import TransactionDataGenerator

    logger.info(f"Generating sample data in {output_dir}")
    generator = TransactionDataGenerator(
        n_entities=n_entities,
        n_transactions=n_transactions,
        start_date=start_date,
        end_date=end_date,
        seed=seed,
    )
    generator.generate_data(output_dir)
    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
