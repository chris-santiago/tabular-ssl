import logging
import warnings
from typing import List, Dict, Any, Optional, Sequence, Union
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

import rich.syntax
import rich.tree

log = logging.getLogger(__name__)


def get_logger(name: str = __name__) -> logging.Logger:
    """Initialize multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)

    # Mark all logging levels with rank zero decorator for multi-GPU compatibility
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """Apply optional utilities before task execution.
    
    Args:
        config: Hydra configuration object
        
    Features:
        - Disable Python warnings if configured
        - Pretty print config tree if configured
    """
    if config is None:
        return

    # Disable python warnings
    if config.get("ignore_warnings"):
        warnings.filterwarnings("ignore")

    # Pretty print config tree
    if config.get("print_config"):
        _print_config(config)


def log_hyperparameters(config: DictConfig, loggers: List[Logger]) -> None:
    """Log hyperparameters to all Lightning loggers.
    
    Args:
        config: Hydra configuration object 
        loggers: List of PyTorch Lightning loggers
    """
    if not loggers:
        return
        
    # Convert config to flat dictionary
    cfg_dict = OmegaConf.to_container(config, resolve=True)
    if isinstance(cfg_dict, dict):
        hparams = _flatten_dict(cfg_dict)
        
        # Log to all loggers
        for logger in loggers:
            logger.log_hyperparams(hparams)


def compute_metrics(
    y_true: Union[List, Any], y_pred: Union[List, Any]
) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        y_true: True labels (list, array, or polars Series)
        y_pred: Predicted labels (list, array, or polars Series)
        
    Returns:
        Dictionary with accuracy, precision, recall, and f1 scores
    """
    # Convert to lists if needed (handles polars Series, numpy arrays, etc.)
    if hasattr(y_true, 'to_list'):
        y_true = y_true.to_list()
    elif hasattr(y_true, 'tolist'):
        y_true = y_true.tolist()
    
    if hasattr(y_pred, 'to_list'):
        y_pred = y_pred.to_list()
    elif hasattr(y_pred, 'tolist'):
        y_pred = y_pred.tolist()
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def plot_training_history(
    history: Dict[str, List[float]], 
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4),
    show: bool = True
) -> None:
    """Plot training history curves.
    
    Args:
        history: Dictionary with training metrics over epochs
        save_path: Optional path to save the plot
        figsize: Figure size tuple (width, height)
        show: Whether to display the plot
        
    Supported keys:
        - loss metrics: train_loss, val_loss, test_loss
        - accuracy metrics: train_accuracy, val_accuracy, test_accuracy
        - Any other metrics will be plotted if available
    """
    # Find available metrics
    loss_keys = [k for k in history.keys() if 'loss' in k.lower()]
    acc_keys = [k for k in history.keys() if 'acc' in k.lower()]
    other_keys = [k for k in history.keys() if k not in loss_keys + acc_keys]
    
    # Determine subplot layout
    n_plots = min(3, sum([bool(loss_keys), bool(acc_keys), bool(other_keys)]))
    if n_plots == 0:
        log.warning("No recognized metrics found in history")
        return
        
    plt.figure(figsize=(figsize[0], figsize[1]))
    
    plot_idx = 1
    
    # Plot loss
    if loss_keys:
        plt.subplot(1, n_plots, plot_idx)
        for key in loss_keys:
            plt.plot(history[key], label=key.replace('_', ' ').title())
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plot_idx += 1
    
    # Plot accuracy
    if acc_keys:
        plt.subplot(1, n_plots, plot_idx)
        for key in acc_keys:
            plt.plot(history[key], label=key.replace('_', ' ').title())
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plot_idx += 1
    
    # Plot other metrics
    if other_keys and plot_idx <= n_plots:
        plt.subplot(1, n_plots, plot_idx)
        for key in other_keys[:3]:  # Limit to 3 metrics to avoid clutter
            plt.plot(history[key], label=key.replace('_', ' ').title())
        plt.title("Other Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()


# Private helper functions
@rank_zero_only
def _print_config(
    config: DictConfig,
    fields: Sequence[str] = ("trainer", "model", "data", "callbacks", "logger", "seed"),
    resolve: bool = True,
) -> None:
    """Print config tree using Rich library."""
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


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dictionary with dot notation."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
