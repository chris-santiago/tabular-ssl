import logging
import os
from typing import List

import hydra
import pytorch_lightning as lit
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

log = logging.getLogger(__name__)


def get_logger(name: str = __name__) -> logging.Logger:
    """Get logger with specified name."""
    return logging.getLogger(name)


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags."""
    if config.get("ignore_warnings"):
        import warnings

        warnings.filterwarnings("ignore")

    if config.get("enable_color_logging"):
        import coloredlogs

        coloredlogs.install(level="INFO", logger=log)

    if config.get("debug"):
        log.info("Running in debug mode")
        os.environ["PYTHONPATH"] = os.getcwd()
        config.trainer.fast_dev_run = True
        config.trainer.limit_train_batches = 2
        config.trainer.limit_val_batches = 2
        config.trainer.limit_test_batches = 2
        config.trainer.limit_predict_batches = 2
        config.trainer.log_every_n_steps = 1


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def log_hyperparameters(
    config: DictConfig,
    model: lit.LightningModule,
    trainer: lit.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers."""
    hparams = {}

    # choose which parts of hydra config are saved by loggers
    hparams["model"] = config["model"]
    hparams["data"] = config["data"]
    hparams["trainer"] = config["trainer"]
    hparams["callbacks"] = config["callbacks"]
    hparams["logger"] = config["logger"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers(logger: List[Logger]) -> None:
    """Makes sure everything closed properly."""
    for lg in logger:
        if isinstance(lg, lit.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()
