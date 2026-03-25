"""
LSTM Price Predictor.

Multi-horizon price prediction using LSTM neural network.
Predicts price movements at T+5min, T+10min, and T+15min.

Architecture:
- Input: 60 timesteps x 5 features (price, volume, volatility, momentum, order_flow)
- 2-layer LSTM (128 -> 64 units) with dropout
- Output: 3 price predictions (T+5, T+10, T+15 min)
"""

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Check for PyTorch availability
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    logger.info("PyTorch available for LSTM predictor")
except ImportError:
    logger.warning("PyTorch not available, LSTM predictor will use fallback")

# Robust default: keep torch-backed LSTM disabled unless explicitly enabled.
# Native torch backends can hard-crash the interpreter on some local macOS setups.
# On training-server (GPU), always enable since CUDA is stable.
TRAINING_SERVER = os.path.exists("/data/models")  # Heuristic for training server
if TORCH_AVAILABLE and os.getenv("COLDPATH_ENABLE_TORCH_LSTM", "0") != "1" and not TRAINING_SERVER:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch detected but disabled for LSTM robustness. "
        "Set COLDPATH_ENABLE_TORCH_LSTM=1 to enable torch LSTM training."
    )


# Input configuration
SEQUENCE_LENGTH = 60  # 60 timesteps (e.g., 60 seconds or 60 candles)
INPUT_FEATURES = 5  # price, volume, volatility, momentum, order_flow
OUTPUT_HORIZONS = 3  # T+5min, T+10min, T+15min

# Feature indices
FEATURE_PRICE = 0
FEATURE_VOLUME = 1
FEATURE_VOLATILITY = 2
FEATURE_MOMENTUM = 3
FEATURE_ORDER_FLOW = 4


@dataclass
class PricePrediction:
    """Price prediction result."""
    price_t5: float  # Predicted price at T+5 min
    price_t10: float  # Predicted price at T+10 min
    price_t15: float  # Predicted price at T+15 min
    return_t5: float  # Predicted return at T+5 min (%)
    return_t10: float  # Predicted return at T+10 min (%)
    return_t15: float  # Predicted return at T+15 min (%)
    confidence: float  # Model confidence (0-1)
    current_price: float  # Current price for reference

    @property
    def trend(self) -> str:
        """Get overall trend direction."""
        avg_return = (self.return_t5 + self.return_t10 + self.return_t15) / 3
        if avg_return > 3.0:
            return "strong_up"
        elif avg_return > 1.0:
            return "up"
        elif avg_return < -3.0:
            return "strong_down"
        elif avg_return < -1.0:
            return "down"
        else:
            return "neutral"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "price_t5": self.price_t5,
            "price_t10": self.price_t10,
            "price_t15": self.price_t15,
            "return_t5": self.return_t5,
            "return_t10": self.return_t10,
            "return_t15": self.return_t15,
            "confidence": self.confidence,
            "current_price": self.current_price,
            "trend": self.trend,
        }


if TORCH_AVAILABLE:
    class LSTMPricePredictorNet(nn.Module):
        """LSTM network for price prediction."""

        def __init__(
            self,
            input_size: int = INPUT_FEATURES,
            hidden_size_1: int = 128,
            hidden_size_2: int = 64,
            output_size: int = OUTPUT_HORIZONS,
            dropout: float = 0.2,
        ):
            super().__init__()

            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2

            # First LSTM layer
            self.lstm1 = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size_1,
                batch_first=True,
                bidirectional=False,
            )
            self.dropout1 = nn.Dropout(dropout)

            # Second LSTM layer
            self.lstm2 = nn.LSTM(
                input_size=hidden_size_1,
                hidden_size=hidden_size_2,
                batch_first=True,
                bidirectional=False,
            )
            self.dropout2 = nn.Dropout(dropout)

            # Fully connected output layer
            self.fc = nn.Linear(hidden_size_2, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor of shape (batch, seq_len, features)

            Returns:
                Output tensor of shape (batch, output_size)
            """
            # First LSTM layer
            lstm1_out, _ = self.lstm1(x)
            lstm1_out = self.dropout1(lstm1_out)

            # Second LSTM layer
            lstm2_out, _ = self.lstm2(lstm1_out)
            lstm2_out = self.dropout2(lstm2_out)

            # Take last timestep output
            last_output = lstm2_out[:, -1, :]

            # Fully connected layer
            output = self.fc(last_output)

            return output


class LSTMPricePredictor:
    """LSTM-based multi-horizon price predictor.

    Predicts price movements at T+5min, T+10min, and T+15min
    using historical sequence data.
    """

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        input_features: int = INPUT_FEATURES,
        hidden_size_1: int = 128,
        hidden_size_2: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        """Initialize the LSTM predictor.

        Args:
            sequence_length: Number of timesteps in input sequence
            input_features: Number of features per timestep
            hidden_size_1: Hidden size of first LSTM layer
            hidden_size_2: Hidden size of second LSTM layer
            dropout: Dropout rate between layers
            learning_rate: Learning rate for training
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.sequence_length = sequence_length
        self.input_features = input_features
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.dropout = dropout
        self.learning_rate = learning_rate

        self.is_fitted = False
        self._training_history: List[Dict[str, float]] = []

        # Normalization parameters
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._target_mean: Optional[float] = None
        self._target_std: Optional[float] = None

        if TORCH_AVAILABLE:
            # Set device
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            # Initialize model
            self.model = LSTMPricePredictorNet(
                input_size=input_features,
                hidden_size_1=hidden_size_1,
                hidden_size_2=hidden_size_2,
                output_size=OUTPUT_HORIZONS,
                dropout=dropout,
            ).to(self.device)

            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()

            logger.info(f"LSTM predictor initialized on {self.device}")
        else:
            self.device = None
            self.model = None
            self.optimizer = None
            self.criterion = None
            logger.warning("LSTM predictor using numpy fallback (no PyTorch)")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 5,
    ) -> Dict[str, Any]:
        """Train the LSTM model.

        Args:
            X: Input sequences of shape (n_samples, seq_len, features)
            y: Target values of shape (n_samples, 3) for T+5/10/15 min returns
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            early_stopping_patience: Epochs to wait before early stopping

        Returns:
            Training history dictionary
        """
        logger.info(f"Training LSTM on {X.shape[0]} samples, {epochs} epochs")

        # Normalize features
        self._feature_means = X.mean(axis=(0, 1))
        self._feature_stds = X.std(axis=(0, 1))
        self._feature_stds[self._feature_stds < 1e-8] = 1.0

        X_normalized = (X - self._feature_means) / self._feature_stds

        # Normalize targets
        self._target_mean = y.mean()
        self._target_std = y.std()
        if self._target_std < 1e-8:
            self._target_std = 1.0

        y_normalized = (y - self._target_mean) / self._target_std

        if not TORCH_AVAILABLE:
            # Fallback: just store mean predictions
            self.is_fitted = True
            return {"loss": 0.0, "val_loss": 0.0, "epochs": 0}

        # Split data
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        X_train = torch.FloatTensor(X_normalized[train_idx]).to(self.device)
        y_train = torch.FloatTensor(y_normalized[train_idx]).to(self.device)
        X_val = torch.FloatTensor(X_normalized[val_idx]).to(self.device)
        y_val = torch.FloatTensor(y_normalized[val_idx]).to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        self._training_history = []

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs, y_val).item()

            self._training_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        self.is_fitted = True

        # GPU memory cleanup: release training tensors
        del X_train, y_train, X_val, y_val
        if TORCH_AVAILABLE and self.device and str(self.device) != "cpu":
            torch.cuda.empty_cache()

        logger.info(f"Training complete. Final val_loss: {best_val_loss:.4f}")

        return {
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "history": self._training_history,
        }

    def predict(self, sequence: np.ndarray, current_price: float) -> PricePrediction:
        """Predict future prices from a sequence.

        Args:
            sequence: Input sequence of shape (seq_len, features) or (1, seq_len, features)
            current_price: Current price for return calculation

        Returns:
            PricePrediction with predictions at T+5/10/15 min
        """
        if not self.is_fitted:
            # Return neutral prediction
            return PricePrediction(
                price_t5=current_price,
                price_t10=current_price,
                price_t15=current_price,
                return_t5=0.0,
                return_t10=0.0,
                return_t15=0.0,
                confidence=0.0,
                current_price=current_price,
            )

        # Ensure correct shape
        if sequence.ndim == 2:
            sequence = sequence.reshape(1, *sequence.shape)

        # Handle sequence length mismatch
        if sequence.shape[1] != self.sequence_length:
            # Pad or truncate
            if sequence.shape[1] < self.sequence_length:
                padding = np.zeros((1, self.sequence_length - sequence.shape[1], self.input_features))
                sequence = np.concatenate([padding, sequence], axis=1)
            else:
                sequence = sequence[:, -self.sequence_length:, :]

        # Normalize
        if self._feature_means is not None:
            sequence = (sequence - self._feature_means) / self._feature_stds

        if not TORCH_AVAILABLE or self.model is None:
            # Fallback: return neutral prediction
            return PricePrediction(
                price_t5=current_price,
                price_t10=current_price,
                price_t15=current_price,
                return_t5=0.0,
                return_t10=0.0,
                return_t15=0.0,
                confidence=0.0,
                current_price=current_price,
            )

        # Predict
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(sequence).to(self.device)
            outputs = self.model(X).cpu().numpy()[0]

        # Denormalize predictions (predictions are returns in %)
        if self._target_mean is not None:
            returns = outputs * self._target_std + self._target_mean
        else:
            returns = outputs

        return_t5, return_t10, return_t15 = returns

        # Calculate predicted prices
        price_t5 = current_price * (1 + return_t5 / 100)
        price_t10 = current_price * (1 + return_t10 / 100)
        price_t15 = current_price * (1 + return_t15 / 100)

        # Estimate confidence from prediction variance
        confidence = min(1.0, 1.0 / (1.0 + abs(return_t15 - return_t5) / 10))

        return PricePrediction(
            price_t5=float(price_t5),
            price_t10=float(price_t10),
            price_t15=float(price_t15),
            return_t5=float(return_t5),
            return_t10=float(return_t10),
            return_t15=float(return_t15),
            confidence=float(confidence),
            current_price=float(current_price),
        )

    def predict_batch(
        self, sequences: np.ndarray, current_prices: np.ndarray
    ) -> List[PricePrediction]:
        """Predict future prices for multiple sequences.

        Args:
            sequences: Input sequences of shape (n_samples, seq_len, features)
            current_prices: Current prices for each sample

        Returns:
            List of PricePrediction objects
        """
        predictions = []
        for seq, price in zip(sequences, current_prices):
            pred = self.predict(seq, price)
            predictions.append(pred)
        return predictions

    def cleanup(self) -> None:
        """Release GPU resources by moving model to CPU and clearing CUDA cache."""
        if not TORCH_AVAILABLE:
            return
        if hasattr(self, "model") and self.model is not None:
            self.model.cpu()
        if hasattr(self, "device") and self.device and str(self.device) != "cpu":
            torch.cuda.empty_cache()
        logger.debug("LSTM predictor GPU resources released")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def save(self, path: str) -> None:
        """Save model to file."""
        data = {
            "sequence_length": self.sequence_length,
            "input_features": self.input_features,
            "hidden_size_1": self.hidden_size_1,
            "hidden_size_2": self.hidden_size_2,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "is_fitted": self.is_fitted,
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
            "target_mean": self._target_mean,
            "target_std": self._target_std,
            "training_history": self._training_history,
        }

        if TORCH_AVAILABLE and self.model is not None:
            data["model_state_dict"] = self.model.state_dict()
            data["optimizer_state_dict"] = self.optimizer.state_dict()

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved LSTM predictor to {path}")

    def load(self, path: str) -> "LSTMPricePredictor":
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.sequence_length = data["sequence_length"]
        self.input_features = data["input_features"]
        self.hidden_size_1 = data["hidden_size_1"]
        self.hidden_size_2 = data["hidden_size_2"]
        self.dropout = data["dropout"]
        self.learning_rate = data["learning_rate"]
        self.is_fitted = data["is_fitted"]
        self._feature_means = data["feature_means"]
        self._feature_stds = data["feature_stds"]
        self._target_mean = data["target_mean"]
        self._target_std = data["target_std"]
        self._training_history = data.get("training_history", [])

        if TORCH_AVAILABLE and "model_state_dict" in data:
            self.model = LSTMPricePredictorNet(
                input_size=self.input_features,
                hidden_size_1=self.hidden_size_1,
                hidden_size_2=self.hidden_size_2,
                output_size=OUTPUT_HORIZONS,
                dropout=self.dropout,
            ).to(self.device)

            self.model.load_state_dict(data["model_state_dict"])

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            if "optimizer_state_dict" in data:
                self.optimizer.load_state_dict(data["optimizer_state_dict"])

        logger.info(f"Loaded LSTM predictor from {path}")
        return self

    def get_training_history(self) -> List[Dict[str, float]]:
        """Get training history."""
        return self._training_history


def prepare_sequences(
    price_data: np.ndarray,
    volume_data: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
    prediction_horizons: Tuple[int, int, int] = (5, 10, 15),
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare training sequences from price and volume data.

    Args:
        price_data: 1D array of prices
        volume_data: 1D array of volumes
        sequence_length: Length of input sequences
        prediction_horizons: Minutes ahead to predict (default: 5, 10, 15)

    Returns:
        Tuple of (X sequences, y targets)
    """
    n_samples = len(price_data) - sequence_length - max(prediction_horizons)

    if n_samples <= 0:
        raise ValueError("Insufficient data for sequences")

    X = []
    y = []

    for i in range(n_samples):
        # Extract sequence
        seq_prices = price_data[i:i + sequence_length]
        seq_volumes = volume_data[i:i + sequence_length]

        # Calculate features
        features = np.zeros((sequence_length, INPUT_FEATURES))

        # Price (normalized by first price in sequence)
        features[:, FEATURE_PRICE] = seq_prices / seq_prices[0] - 1

        # Volume (log-normalized)
        features[:, FEATURE_VOLUME] = np.log1p(seq_volumes) / 10

        # Volatility (rolling std of returns)
        returns = np.diff(seq_prices) / seq_prices[:-1]
        volatility = np.zeros(sequence_length)
        for j in range(5, sequence_length):
            volatility[j] = np.std(returns[j - 5:j]) if j > 0 else 0
        features[:, FEATURE_VOLATILITY] = volatility

        # Momentum (price change over last 5 steps)
        features[:, FEATURE_MOMENTUM] = np.concatenate([
            [0] * 5,
            (seq_prices[5:] - seq_prices[:-5]) / seq_prices[:-5]
        ])

        # Order flow (volume-weighted price change)
        features[:, FEATURE_ORDER_FLOW] = np.concatenate([
            [0],
            returns * (seq_volumes[1:] / (seq_volumes[:-1] + 1e-8))
        ])

        X.append(features)

        # Calculate targets (future returns)
        current_price = seq_prices[-1]
        targets = []
        for horizon in prediction_horizons:
            future_idx = i + sequence_length + horizon
            if future_idx < len(price_data):
                future_price = price_data[future_idx]
                future_return = (future_price - current_price) / current_price * 100
                targets.append(future_return)
            else:
                targets.append(0.0)

        y.append(targets)

    return np.array(X), np.array(y)
