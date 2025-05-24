#!/usr/bin/env python

try:
    import dotenv

    # Load environment variables from `.env` file if it exists
    # Recursively searches for `.env` in all folders starting from work dir
    dotenv.load_dotenv(override=True)
except ImportError:
    # dotenv is optional
    pass

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import Logger
from tabular_ssl import utils

log = utils.get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training pipeline with error handling and performance optimizations."""

    # Set up environment variables and configurations
    utils.extras(config)

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    # Initialize the datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)

    # Initialize the model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Initialize callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Initialize loggers
    logger: list[Logger] = []
    if "logger" in config:
        for lg_conf in config.logger.values():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Initialize the trainer
    log.info("Instantiating trainer")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(config=config, loggers=logger)

    # Train the model
    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set
    if config.get("test"):
        log.info("Starting testing!")
        if not config.get("train"):
            log.warning(
                "No checkpoints found. Using model weights from the end of training."
            )
        trainer.test(model=model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")

    # Print path to best checkpoint
    if trainer.checkpoint_callback:
        log.info(f"Best checkpoint path: {trainer.checkpoint_callback.best_model_path}")

    # Return for possible use in optimizations like Optuna
    return trainer.callback_metrics.get("test/acc_best", None)


if __name__ == "__main__":
    main()
