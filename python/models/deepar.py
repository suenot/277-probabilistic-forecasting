"""
DeepAR Model for Probabilistic Forecasting
==========================================

Implements DeepAR (Deep Autoregressive) model for probabilistic
time series forecasting with various distribution options.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional, Dict
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaussianDistribution:
    """Gaussian distribution output head."""

    @staticmethod
    def num_params() -> int:
        return 2  # mu, sigma

    @staticmethod
    def get_params(output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = output[..., 0]
        sigma = F.softplus(output[..., 1]) + 1e-6
        return mu, sigma

    @staticmethod
    def log_prob(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu, sigma = GaussianDistribution.get_params(output)
        dist = torch.distributions.Normal(mu, sigma)
        return dist.log_prob(target)

    @staticmethod
    def sample(output: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        mu, sigma = GaussianDistribution.get_params(output)
        dist = torch.distributions.Normal(mu, sigma)
        return dist.sample((num_samples,))

    @staticmethod
    def mean(output: torch.Tensor) -> torch.Tensor:
        mu, _ = GaussianDistribution.get_params(output)
        return mu

    @staticmethod
    def quantile(output: torch.Tensor, q: float) -> torch.Tensor:
        mu, sigma = GaussianDistribution.get_params(output)
        dist = torch.distributions.Normal(mu, sigma)
        return dist.icdf(torch.tensor(q))


class StudentTDistribution:
    """Student-t distribution output head for heavy-tailed data."""

    @staticmethod
    def num_params() -> int:
        return 3  # mu, sigma, df

    @staticmethod
    def get_params(output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = output[..., 0]
        sigma = F.softplus(output[..., 1]) + 1e-6
        df = F.softplus(output[..., 2]) + 2.01  # df > 2 for finite variance
        return mu, sigma, df

    @staticmethod
    def log_prob(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu, sigma, df = StudentTDistribution.get_params(output)
        dist = torch.distributions.StudentT(df, mu, sigma)
        return dist.log_prob(target)

    @staticmethod
    def sample(output: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        mu, sigma, df = StudentTDistribution.get_params(output)
        dist = torch.distributions.StudentT(df, mu, sigma)
        return dist.sample((num_samples,))

    @staticmethod
    def mean(output: torch.Tensor) -> torch.Tensor:
        mu, _, _ = StudentTDistribution.get_params(output)
        return mu


class DeepARModel(nn.Module):
    """
    DeepAR Model for Probabilistic Time Series Forecasting.

    Based on: "DeepAR: Probabilistic Forecasting with Autoregressive
    Recurrent Networks" (Salinas et al., 2020)
    """

    DISTRIBUTIONS = {
        'gaussian': GaussianDistribution,
        'student_t': StudentTDistribution,
    }

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        distribution: str = 'gaussian'
    ):
        """
        Initialize DeepAR model.

        Args:
            input_dim: Number of input features (covariates)
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            distribution: Output distribution ('gaussian' or 'student_t')
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.distribution_name = distribution
        self.dist = self.DISTRIBUTIONS[distribution]

        # Input embedding for target values
        self.target_embedding = nn.Linear(1, hidden_size // 2)

        # Input embedding for covariates
        self.covariate_embedding = nn.Linear(input_dim, hidden_size // 2)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output head
        self.output_layer = nn.Linear(hidden_size, self.dist.num_params())

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        targets: torch.Tensor,
        covariates: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            targets: Target values [batch_size, seq_len]
            covariates: Covariate features [batch_size, seq_len, input_dim]
            hidden: Initial hidden state (optional)

        Returns:
            Tuple of (distribution_params, hidden_state)
        """
        batch_size, seq_len = targets.shape

        # Embed targets (shift by 1 for autoregressive)
        target_shifted = torch.cat([
            torch.zeros(batch_size, 1, device=targets.device),
            targets[:, :-1]
        ], dim=1).unsqueeze(-1)

        target_embed = self.target_embedding(target_shifted)
        covariate_embed = self.covariate_embedding(covariates)

        # Concatenate embeddings
        lstm_input = torch.cat([target_embed, covariate_embed], dim=-1)

        # LSTM forward pass
        if hidden is None:
            lstm_output, hidden = self.lstm(lstm_input)
        else:
            lstm_output, hidden = self.lstm(lstm_input, hidden)

        # Distribution parameters
        dist_params = self.output_layer(lstm_output)

        return dist_params, hidden

    def loss(
        self,
        targets: torch.Tensor,
        covariates: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        Args:
            targets: Target values [batch_size, seq_len]
            covariates: Covariate features [batch_size, seq_len, input_dim]

        Returns:
            Loss value
        """
        dist_params, _ = self.forward(targets, covariates)
        log_probs = self.dist.log_prob(dist_params, targets)
        return -log_probs.mean()

    def predict(
        self,
        targets_history: torch.Tensor,
        covariates: torch.Tensor,
        prediction_horizon: int,
        num_samples: int = 100
    ) -> torch.Tensor:
        """
        Generate probabilistic forecasts.

        Args:
            targets_history: Historical target values [batch_size, history_len]
            covariates: Covariates for full horizon [batch_size, history_len + pred_horizon, input_dim]
            prediction_horizon: Number of steps to predict
            num_samples: Number of Monte Carlo samples

        Returns:
            Samples [num_samples, batch_size, prediction_horizon]
        """
        self.train(False)
        batch_size = targets_history.shape[0]
        history_len = targets_history.shape[1]

        with torch.no_grad():
            # Encode history
            _, hidden = self.forward(
                targets_history,
                covariates[:, :history_len, :]
            )

            # Generate samples autoregressively
            samples = torch.zeros(num_samples, batch_size, prediction_horizon)
            current_target = targets_history[:, -1:]

            for t in range(prediction_horizon):
                # Get covariates for this step
                step_covariates = covariates[:, history_len + t: history_len + t + 1, :]

                # Forward one step
                dist_params, hidden = self.forward(
                    current_target,
                    step_covariates,
                    hidden
                )

                # Sample from distribution
                step_samples = self.dist.sample(dist_params[:, -1, :], num_samples)
                samples[:, :, t] = step_samples.squeeze(-1)

                # Update current target (use mean for autoregressive input)
                current_target = self.dist.mean(dist_params[:, -1:, :])

        return samples

    def get_quantiles(
        self,
        targets_history: torch.Tensor,
        covariates: torch.Tensor,
        prediction_horizon: int,
        quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
        num_samples: int = 200
    ) -> Dict[float, torch.Tensor]:
        """
        Get quantile forecasts.

        Args:
            targets_history: Historical target values
            covariates: Covariates for full horizon
            prediction_horizon: Number of steps to predict
            quantiles: Quantile levels to compute
            num_samples: Number of samples for quantile estimation

        Returns:
            Dictionary mapping quantile level to predictions
        """
        samples = self.predict(
            targets_history, covariates, prediction_horizon, num_samples
        )

        result = {}
        for q in quantiles:
            result[q] = torch.quantile(samples, q, dim=0)

        return result


class DeepARTrainer:
    """Trainer for DeepAR model."""

    def __init__(
        self,
        model: DeepARModel,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ):
        """
        Initialize trainer.

        Args:
            model: DeepAR model
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

        for targets, covariates in dataloader:
            targets = targets.to(self.device)
            covariates = covariates.to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.loss(targets, covariates)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
            for targets, covariates in dataloader:
                targets = targets.to(self.device)
                covariates = covariates.to(self.device)

                loss = self.model.loss(targets, covariates)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history
        """
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
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}")

        return history


if __name__ == "__main__":
    # Example usage
    batch_size = 32
    seq_len = 48
    input_dim = 10
    pred_horizon = 4

    # Generate synthetic data
    torch.manual_seed(42)
    targets = torch.randn(batch_size, seq_len)
    covariates = torch.randn(batch_size, seq_len + pred_horizon, input_dim)

    # Create model
    model = DeepARModel(
        input_dim=input_dim,
        hidden_size=64,
        num_layers=2,
        distribution='gaussian'
    )

    # Compute loss
    loss = model.loss(targets, covariates[:, :seq_len, :])
    print(f"Loss: {loss.item():.4f}")

    # Generate forecasts
    model.train(False)
    samples = model.predict(
        targets,
        covariates,
        prediction_horizon=pred_horizon,
        num_samples=100
    )
    print(f"Samples shape: {samples.shape}")

    # Get quantiles
    quantiles = model.get_quantiles(
        targets, covariates, pred_horizon,
        quantiles=[0.1, 0.5, 0.9]
    )
    print(f"Quantile shapes: {[q.shape for q in quantiles.values()]}")
