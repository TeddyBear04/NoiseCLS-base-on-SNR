import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early Stopping monitor to halt training when monitored performance metric stops improving.
    Complies with OOP design guidelines.
    """
    def __init__(self, patience: int = 30, delta: float = 0.0, verbose: bool = True) -> None:
        """
        Initialize EarlyStopping mechanism.

        Args:
            patience (int): Number of epochs to wait after the last improvement. Default: 30.
            delta (float): Minimum change to qualify as an improvement. Default: 0.0.
            verbose (bool): If True, logs a message for each epoch. Default: True.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        # Internal state counters
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def reset(self) -> None:
        """
        Reset internal state counters and flag.
        Useful when starting a new training run without re-instantiating.
        """
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        logger.info("Early Stopping state has been reset.")

    def step(self, score: float) -> bool:
        """
        Register a new epoch score and check if early stopping criteria are met.

        Args:
            score (float): Monitored score for the current epoch (e.g. accuracy or negative loss).

        Returns:
            bool: True if training should be stopped, otherwise False.
        """
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"Early Stopping: {self.counter}/{self.patience} epochs without improvement.")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.warning(
                    f"Early Stopping triggered: Metric did not improve by delta={self.delta} for {self.patience} consecutive epochs."
                )
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop
