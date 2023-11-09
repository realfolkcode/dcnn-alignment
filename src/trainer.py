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
                 device: str):
        """Initializes an instance of Trainer.

        Args:
            train_loader: Train set data loader.
            val_loader: Val set data loader.
            num_epochs: The number of epochs.
            optimizer: Optimizer.
            device: Device.
        """
        self.criterion = nn.MSELoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.device = device

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
        for batch in tqdm(self.val_loader):
            x = batch['image'].to(self.device)
            y_true = batch['target'].to(self.device)

            y_pred = model(x)
            loss = self.criterion(y_pred, y_true)
            val_loss += loss.item()
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
