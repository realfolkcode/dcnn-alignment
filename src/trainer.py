from typing import Optional, Callable
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


class Trainer:
    """Model trainer."""

    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 num_epochs: int,
                 optimizer: Optimizer,
                 device: str,
                 metrics_logger: Optional[Callable] = None,
                 images_logger: Optional[Callable] = None):
        """Initializes an instance of Trainer.

        Args:
            train_loader: Train set data loader.
            val_loader: Val set data loader.
            num_epochs: The number of epochs.
            optimizer: Optimizer.
            device: Device.
            metrics_logger: Metrics logger.
            images_logger: Images logger.
        """
        self.criterion = nn.MSELoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.device = device
        self.metrics_logger = metrics_logger
        self.images_logger = images_logger

    def _train_epoch(self, model: nn.Module) -> float:
        """One epoch pass."""
        epoch_loss = 0
        model.train()
        for batch in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            x = batch['image'].to(self.device)
            y_true = batch['target'].to(self.device)

            y_pred = model(x)
            loss = self.criterion(y_pred, y_true)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss

    @torch.no_grad
    def _validate(self, model: nn.Module) -> float:
        """Validation."""
        val_loss = 0
        model.eval()
        for i, batch in tqdm(enumerate(self.val_loader)):
            x = batch['image'].to(self.device)
            y_true = batch['target'].to(self.device)

            y_pred = model(x)
            loss = self.criterion(y_pred, y_true)
            val_loss += loss.item()

            if i == 0 and self.images_logger is not None:
                self.images_logger(x, y_pred)
        return val_loss

    def train(self, model: nn.Module) -> None:
        """Trains a DCNN model.

        Args:
            model: DCNN model.
        """
        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch(model)
            print(f"Epoch: {epoch}, Train loss: {train_loss}")
            val_loss = self._validate(model)
            print(f"Epoch: {epoch}, Val loss: {val_loss}")
            if self.metrics_logger is not None:
                self.metrics_logger({"train_loss": train_loss,
                                     "val_loss": val_loss})
