import hydra
import pytorch_lightning as lit
from omegaconf import DictConfig, OmegaConf
from tabular_ssl.utils.utils import get_logger, extras

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training pipeline."""
    # Imports should be nested inside @hydra.main to optimize tab completion
    from tabular_ssl.utils.utils import instantiate_callbacks, instantiate_loggers
    from tabular_ssl.utils.utils import log_hyperparameters, get_metric_value
    from tabular_ssl.utils.utils import close_loggers

    # A couple of optional utilities:
    # - disabling python warnings if they annoy you
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        log.info("Printing config with Rich! <cfg>")
        log.info(OmegaConf.to_yaml(config))

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        lit.seed_everything(config.seed, workers=True)

    # Initialize lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)

    # Initialize lightning datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule = hydra.utils.instantiate(config.data)

    # Initialize lightning callbacks
    callbacks = instantiate_callbacks(config.get("callbacks"))

    # Initialize lightning loggers
    logger = instantiate_loggers(config.get("logger"))

    # Initialize lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger
    )

    # Send some parameters from config to all lightning loggers
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


if __name__ == "__main__":
    main()
