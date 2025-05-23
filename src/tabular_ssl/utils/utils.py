import logging
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def get_feature_importance(model: torch.nn.Module, data: torch.Tensor, feature_names: List[str]) -> Dict[str, float]:
    """Compute feature importance scores."""
    model.eval()
    with torch.no_grad():
        # Get attention weights from the last layer
        _, attention_weights = model(data, return_attention=True)
        last_layer_weights = attention_weights[-1]
        
        # Average attention weights across heads and sequence length
        importance = last_layer_weights.mean(dim=(0, 1, 2))
        
        # Normalize importance scores
        importance = importance / importance.sum()
        
        return dict(zip(feature_names, importance.tolist()))


def compute_shap_values(model: torch.nn.Module, data: torch.Tensor, feature_names: List[str]) -> np.ndarray:
    """Compute SHAP values for model predictions."""
    model.eval()
    explainer = shap.DeepExplainer(model, data)
    shap_values = explainer.shap_values(data)
    return shap_values


def grid_search(objective: callable, param_grid: Dict[str, List[Any]]) -> Tuple[Dict[str, Any], float]:
    """Perform grid search for hyperparameter tuning."""
    best_score = float('-inf')
    best_params = None
    
    # Generate all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in np.array(np.meshgrid(*param_grid.values())).T.reshape(-1, len(param_grid))]
    
    for params in param_combinations:
        score = objective(params)
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score


def random_search(objective: callable, param_distributions: Dict[str, List[Any]], n_iter: int = 10) -> Tuple[Dict[str, Any], float]:
    """Perform random search for hyperparameter tuning."""
    best_score = float('-inf')
    best_params = None
    
    for _ in range(n_iter):
        # Sample random parameters
        params = {k: np.random.choice(v) for k, v in param_distributions.items()}
        score = objective(params)
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float, train_loss: float) -> bool:
        """Check if training should be stopped."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
