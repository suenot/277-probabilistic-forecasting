"""
Quantile Regression for Probabilistic Forecasting
==================================================

Implements quantile regression using neural networks to predict
multiple quantiles of the conditional distribution.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss) for quantile regression.

    L_tau(y, q) = (tau - I(y < q)) * (y - q)

    Where tau is the target quantile level.
    """

    def __init__(self, quantiles: List[float]):
        """
        Initialize quantile loss.

        Args:
            quantiles: List of quantile levels (e.g., [0.05, 0.5, 0.95])
        """
        super().__init__()
        self.quantiles = torch.tensor(quantiles)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.

        Args:
            predictions: Predicted quantiles [batch_size, num_quantiles]
            targets: Actual values [batch_size]

        Returns:
            Total loss across all quantiles
        """
        # Expand targets to match predictions shape
        targets = targets.unsqueeze(1).expand_as(predictions)

        # Move quantiles to same device
        quantiles = self.quantiles.to(predictions.device)

        # Compute errors
        errors = targets - predictions

        # Compute quantile loss for each quantile
        # L = max(tau * error, (tau - 1) * error)
        losses = torch.max(
            quantiles * errors,
            (quantiles - 1) * errors
        )

        return losses.mean()


class QuantileRegressor(nn.Module):
    """
    Neural Network for Quantile Regression.

    Predicts multiple quantiles of the conditional distribution.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    ):
        """
        Initialize quantile regressor.

        Args:
            input_dim: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of hidden layers
            dropout: Dropout probability
            quantiles: Quantile levels to predict
        """
        super().__init__()

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        # Build network
        layers = []
        prev_size = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, self.num_quantiles)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Predicted quantiles [batch_size, num_quantiles]
        """
        features = self.feature_extractor(x)
        quantiles = self.output_layer(features)

        # Ensure quantiles are monotonically increasing
        # Using cumulative softmax to enforce ordering
        sorted_quantiles, _ = torch.sort(quantiles, dim=1)

        return sorted_quantiles

    def predict_distribution(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict quantiles and return distribution statistics.

        Args:
            x: Input features

        Returns:
            Tuple of (quantiles, quantile_levels)
        """
        self.train(False)
        with torch.no_grad():
            quantiles = self.forward(x)
        return quantiles, torch.tensor(self.quantiles)

    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> torch.Tensor:
        """
        Generate samples from predicted distribution using quantile interpolation.

        Args:
            x: Input features [batch_size, input_dim]
            num_samples: Number of samples to generate

        Returns:
            Samples [batch_size, num_samples]
        """
        self.train(False)
        with torch.no_grad():
            quantiles = self.forward(x)

        batch_size = x.shape[0]
        samples = torch.zeros(batch_size, num_samples)

        # Generate uniform random numbers
        u = torch.rand(batch_size, num_samples)

        # Interpolate quantiles
        quantile_levels = torch.tensor(self.quantiles)

        for i in range(batch_size):
            samples[i] = torch.from_numpy(np.interp(
                u[i].numpy(),
                quantile_levels.numpy(),
                quantiles[i].cpu().numpy()
            ))

        return samples


class QuantileRegressorTrainer:
    """
    Trainer for Quantile Regression model.
    """

    def __init__(
        self,
        model: QuantileRegressor,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ):
        """
        Initialize trainer.

        Args:
            model: QuantileRegressor model
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        self.model = model
        self.criterion = QuantileLoss(model.quantiles)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def run_evaluation(self, dataloader: DataLoader) -> float:
        """
        Run evaluation on model.

        Args:
            dataloader: Validation data loader

        Returns:
            Average loss
        """
        self.model.train(False)
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ) -> dict:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)

            if val_loader is not None:
                val_loss = self.run_evaluation(val_loader)
                history['val_loss'].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted quantiles [n_samples, num_quantiles]
        """
        self.model.train(False)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.5

    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create and train model
    model = QuantileRegressor(
        input_dim=n_features,
        hidden_size=32,
        num_layers=2,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )

    trainer = QuantileRegressorTrainer(model)
    history = trainer.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=50,
        batch_size=32
    )

    # Make predictions
    predictions = trainer.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions (quantiles): {predictions[0]}")
    print(f"Actual value: {y_test[0]}")
