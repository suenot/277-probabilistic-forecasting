"""
Mixture Density Network for Probabilistic Forecasting
=====================================================

Implements Mixture Density Networks (MDN) that output
a mixture of Gaussian distributions for multi-modal predictions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MixtureDensityNetwork(nn.Module):
    """
    Mixture Density Network (MDN).

    Outputs a mixture of Gaussian distributions to model
    multi-modal conditional distributions.

    Based on: "Mixture Density Networks" (Bishop, 1994)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_components: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize MDN.

        Args:
            input_dim: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of hidden layers
            num_components: Number of mixture components
            dropout: Dropout probability
        """
        super().__init__()

        self.num_components = num_components

        # Build feature extractor
        layers = []
        prev_size = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.feature_extractor = nn.Sequential(*layers)

        # Output heads for mixture parameters
        # pi: mixing coefficients (K)
        # mu: means (K)
        # sigma: standard deviations (K)
        self.pi_layer = nn.Linear(hidden_size, num_components)
        self.mu_layer = nn.Linear(hidden_size, num_components)
        self.sigma_layer = nn.Linear(hidden_size, num_components)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Tuple of (pi, mu, sigma) where each has shape [batch_size, num_components]
        """
        features = self.feature_extractor(x)

        # Mixing coefficients (sum to 1)
        pi = F.softmax(self.pi_layer(features), dim=-1)

        # Means
        mu = self.mu_layer(features)

        # Standard deviations (positive)
        sigma = F.softplus(self.sigma_layer(features)) + 1e-6

        return pi, mu, sigma

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        Args:
            x: Input features [batch_size, input_dim]
            y: Target values [batch_size]

        Returns:
            Loss value
        """
        pi, mu, sigma = self.forward(x)

        # Expand y for broadcasting
        y = y.unsqueeze(-1)  # [batch_size, 1]

        # Compute log probability for each component
        # log N(y | mu_k, sigma_k)
        log_normal = -0.5 * (
            torch.log(2 * torch.tensor(np.pi)) +
            2 * torch.log(sigma) +
            ((y - mu) / sigma) ** 2
        )

        # Log-sum-exp for mixture
        # log(sum_k pi_k * N(y | mu_k, sigma_k))
        log_pi = torch.log(pi + 1e-10)
        log_prob = torch.logsumexp(log_pi + log_normal, dim=-1)

        return -log_prob.mean()

    def get_mixture_params(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get mixture parameters for inputs.

        Args:
            x: Input features

        Returns:
            Tuple of (pi, mu, sigma)
        """
        self.train(False)
        with torch.no_grad():
            return self.forward(x)

    def sample(self, x: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        """
        Sample from the predicted mixture distribution.

        Args:
            x: Input features [batch_size, input_dim]
            num_samples: Number of samples to generate

        Returns:
            Samples [batch_size, num_samples]
        """
        self.train(False)
        batch_size = x.shape[0]

        with torch.no_grad():
            pi, mu, sigma = self.forward(x)

        samples = torch.zeros(batch_size, num_samples)

        for i in range(batch_size):
            # Sample component indices
            component_idx = torch.multinomial(pi[i], num_samples, replacement=True)

            # Sample from selected components
            for j in range(num_samples):
                k = component_idx[j].item()
                samples[i, j] = torch.normal(mu[i, k], sigma[i, k])

        return samples

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mean of mixture distribution.

        Args:
            x: Input features

        Returns:
            Mean values [batch_size]
        """
        self.train(False)
        with torch.no_grad():
            pi, mu, _ = self.forward(x)
            return (pi * mu).sum(dim=-1)

    def variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute variance of mixture distribution.

        Uses law of total variance:
        Var(Y) = E[Var(Y|K)] + Var(E[Y|K])

        Args:
            x: Input features

        Returns:
            Variance values [batch_size]
        """
        self.train(False)
        with torch.no_grad():
            pi, mu, sigma = self.forward(x)

            # E[Y]
            mean = (pi * mu).sum(dim=-1, keepdim=True)

            # E[Var(Y|K)] = sum_k pi_k * sigma_k^2
            expected_variance = (pi * sigma ** 2).sum(dim=-1)

            # Var(E[Y|K]) = E[(mu_k - E[Y])^2] = sum_k pi_k * (mu_k - E[Y])^2
            variance_of_means = (pi * (mu - mean) ** 2).sum(dim=-1)

            return expected_variance + variance_of_means

    def quantile(
        self,
        x: torch.Tensor,
        q: float,
        num_samples: int = 1000
    ) -> torch.Tensor:
        """
        Estimate quantile via sampling.

        Args:
            x: Input features
            q: Quantile level (0-1)
            num_samples: Number of samples for estimation

        Returns:
            Quantile values [batch_size]
        """
        samples = self.sample(x, num_samples)
        return torch.quantile(samples, q, dim=1)

    def get_quantiles(
        self,
        x: torch.Tensor,
        quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
        num_samples: int = 1000
    ) -> Dict[float, torch.Tensor]:
        """
        Get multiple quantiles.

        Args:
            x: Input features
            quantiles: List of quantile levels
            num_samples: Number of samples for estimation

        Returns:
            Dictionary mapping quantile level to values
        """
        samples = self.sample(x, num_samples)
        return {q: torch.quantile(samples, q, dim=1) for q in quantiles}


class MDNTrainer:
    """Trainer for Mixture Density Network."""

    def __init__(
        self,
        model: MixtureDensityNetwork,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ):
        """
        Initialize trainer.

        Args:
            model: MDN model
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.loss(X_batch, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def run_validation(self, dataloader: DataLoader) -> float:
        """Run validation evaluation."""
        self.model.train(False)
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss = self.model.loss(X_batch, y_batch)
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
    ) -> Dict[str, List[float]]:
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
                val_loss = self.run_validation(val_loader)
                history['val_loss'].append(val_loss)

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

        return history

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Dictionary with mean, variance, and quantiles
        """
        self.model.train(False)
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            mean = self.model.mean(X_tensor).cpu().numpy()
            variance = self.model.variance(X_tensor).cpu().numpy()
            quantiles = self.model.get_quantiles(X_tensor)
            quantiles = {k: v.cpu().numpy() for k, v in quantiles.items()}

        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'quantiles': quantiles
        }


if __name__ == "__main__":
    # Example: Fit MDN to bimodal data
    np.random.seed(42)
    n_samples = 1000

    # Generate bimodal data
    X = np.random.randn(n_samples, 5)
    mode_indicator = (X[:, 0] > 0).astype(float)
    y = mode_indicator * (2 + np.random.randn(n_samples) * 0.3) + \
        (1 - mode_indicator) * (-2 + np.random.randn(n_samples) * 0.3)

    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create and train model
    model = MixtureDensityNetwork(
        input_dim=5,
        hidden_size=32,
        num_layers=2,
        num_components=2
    )

    trainer = MDNTrainer(model)
    history = trainer.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=50,
        batch_size=32
    )

    # Make predictions
    predictions = trainer.predict(X_test[:5])
    print("Predictions for first 5 samples:")
    print(f"Mean: {predictions['mean']}")
    print(f"Std: {predictions['std']}")
    print(f"Actual: {y_test[:5]}")
    print(f"5th percentile: {predictions['quantiles'][0.05]}")
    print(f"95th percentile: {predictions['quantiles'][0.95]}")
