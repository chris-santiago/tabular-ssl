import hydra
from omegaconf import DictConfig
from tabular_ssl.data.generate_sample_data import TransactionDataGenerator
from tabular_ssl.utils.utils import get_logger

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="data/generate")
def main(config: DictConfig) -> None:
    """Generate or preprocess data using Hydra configuration."""
    try:
        if config.mode == "generate":
            # Generate sample data
            log.info(f"Generating sample data in {config.paths.output_dir}")
            generator = TransactionDataGenerator(
                n_entities=config.n_entities,
                n_transactions=config.n_transactions,
                start_date=config.start_date,
                end_date=config.end_date,
                seed=config.seed,
                n_jobs=config.n_jobs,
            )
            generator.generate_data(config.paths.output_dir)
            log.info("Data generation completed successfully!")

        elif config.mode == "preprocess":
            # Preprocess existing data
            log.info(f"Preprocessing data from {config.preprocessing.input_dir}")
            # TODO: Implement preprocessing logic
            log.info("Preprocessing completed successfully!")

        else:
            raise ValueError(f"Unknown mode: {config.mode}")

    except Exception as e:
        log.error(f"Error in data generation/preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
