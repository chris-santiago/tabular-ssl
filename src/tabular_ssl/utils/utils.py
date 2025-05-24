import logging
import warnings
from typing import List, Dict, Any, Optional, Sequence
import polars as pl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pytorch_lightning as lit
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

import rich.syntax
import rich.tree

log = logging.getLogger(__name__)


def get_logger(name: str = __name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """Applies optional utilities before the task is run.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # return if no config
    if config is None:
        return

    # disable python warnings
    if config.get("ignore_warnings"):
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library
    if config.get("print_config"):
        print_config(config, resolve=True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "data",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration.
        fields (Sequence[str], optional): Determines which main fields from config to print.
        resolve (bool, optional): Whether to resolve reference fields. Defaults to True.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: lit.LightningModule,
    datamodule: lit.LightningDataModule,
    trainer: lit.Trainer,
    callbacks: List[Callback],
    logger: List[Logger],
) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Saves:
        - Parameters passed to model
        - Parameters passed to datamodule
        - Trainer parameters
        - Callback parameters
    """

    hparams = {}

    # get config as flat dictionary
    cfg_dict = OmegaConf.to_container(config, resolve=True)
    if isinstance(cfg_dict, dict):
        hparams = flatten(cfg_dict)

    # if loggers exist, log hyperparameters
    if logger:
        for lg in logger:
            lg.log_hyperparams(hparams)


def flatten(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flattens a nested dictionary, joining keys with separator."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compute_metrics(y_true: pl.Series, y_pred: pl.Series) -> Dict[str, float]:
    """Compute classification metrics."""
    y_true_list = y_true.to_list()
    y_pred_list = y_pred.to_list()
    return {
        "accuracy": accuracy_score(y_true_list, y_pred_list),
        "precision": precision_score(y_true_list, y_pred_list, average="weighted"),
        "recall": recall_score(y_true_list, y_pred_list, average="weighted"),
        "f1": f1_score(y_true_list, y_pred_list, average="weighted"),
    }


def plot_training_history(
    history: Dict[str, List[float]], save_path: Optional[str] = None
) -> None:
    """Plot training history."""
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    if "train_loss" in history:
        plt.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    if "train_accuracy" in history:
        plt.plot(history["train_accuracy"], label="Train Accuracy")
    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
