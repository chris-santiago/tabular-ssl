import hydra
import pytorch_lightning as lit
from omegaconf import DictConfig, OmegaConf
from tabular_ssl.utils.utils import get_logger, extras
import torch
import warnings
from typing import Optional, Any
import os

log = get_logger(__name__)


def setup_environment(config: DictConfig) -> None:
    """Setup environment variables and configurations."""
    # Set PyTorch to use deterministic algorithms if specified
    if config.get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    
    # Set number of threads for PyTorch
    if config.get("num_workers", None) is not None:
        torch.set_num_threads(config.num_workers)
    
    # Set memory growth for GPU if available
    if torch.cuda.is_available() and config.get("gpu_memory_fraction", None) is not None:
        torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction)
    
    # Set environment variables for better performance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["OMP_NUM_THREADS"] = str(config.get("num_workers", 1))


def instantiate_model(config: DictConfig) -> lit.LightningModule:
    """Instantiate the model with error handling."""
    try:
        log.info(f"Instantiating model <{config.model._target_}>")
        model = hydra.utils.instantiate(config.model)
        return model
    except Exception as e:
        log.error(f"Error instantiating model: {str(e)}")
        raise


def instantiate_datamodule(config: DictConfig) -> lit.LightningDataModule:
    """Instantiate the datamodule with error handling."""
    try:
        log.info(f"Instantiating datamodule <{config.data._target_}>")
        datamodule = hydra.utils.instantiate(config.data)
        return datamodule
    except Exception as e:
        log.error(f"Error instantiating datamodule: {str(e)}")
        raise


def instantiate_trainer(config: DictConfig, callbacks: list, logger: list) -> lit.Trainer:
    """Instantiate the trainer with error handling."""
    try:
        log.info(f"Instantiating trainer <{config.trainer._target_}>")
        trainer = hydra.utils.instantiate(
            config.trainer, callbacks=callbacks, logger=logger
        )
        return trainer
    except Exception as e:
        log.error(f"Error instantiating trainer: {str(e)}")
        raise


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> Optional[float]:
    """Main training pipeline with improved error handling and performance optimizations."""
    # Imports should be nested inside @hydra.main to optimize tab completion
    from tabular_ssl.utils.utils import instantiate_callbacks, instantiate_loggers
    from tabular_ssl.utils.utils import log_hyperparameters, get_metric_value
    from tabular_ssl.utils.utils import close_loggers

    try:
        # Setup environment
        setup_environment(config)

        # Apply optional utilities
        extras(config)

        # Pretty print config using Rich library
        if config.get("print_config"):
            log.info("Printing config with Rich! <cfg>")
            log.info(OmegaConf.to_yaml(config))

        # Set seed for random number generators
        if config.get("seed"):
            lit.seed_everything(config.seed, workers=True)

        # Initialize components with error handling
        model = instantiate_model(config)
        datamodule = instantiate_datamodule(config)
        callbacks = instantiate_callbacks(config.get("callbacks"))
        logger = instantiate_loggers(config.get("logger"))
        trainer = instantiate_trainer(config, callbacks, logger)

        # Log hyperparameters
        log.info("Logging hyperparameters!")
        log_hyperparameters(config=config, model=model, trainer=trainer)

        # Train the model
        if config.get("train"):
            log.info("Starting training!")
            trainer.fit(model=model, datamodule=datamodule)

        # Get metric value for hyperparameter optimization
        optimized_metric = config.get("optimized_metric", "val/loss")
        metric_value = get_metric_value(
            metric_dict=trainer.callback_metrics, metric_name=optimized_metric
        )

        # Test the model
        if config.get("test"):
            log.info("Starting testing!")
            trainer.test(model=model, datamodule=datamodule)

        # Make sure everything closed properly
        log.info("Finalizing!")
        close_loggers(logger)

        # Print path to best checkpoint
        if not config.trainer.get("fast_dev_run"):
            log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

        return metric_value

    except Exception as e:
        log.error(f"An error occurred during training: {str(e)}")
        raise

    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    main()
