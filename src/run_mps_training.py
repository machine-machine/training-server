#!/usr/bin/env python3
"""
2DEXY MPS Training Runner
Native Apple Silicon GPU training using Metal Performance Shaders (MPS)

Usage:
    cd training-server
    source .venv/bin/activate
    python src/run_mps_training.py --model fraud --epochs 10
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np


def check_mps():
    """Check MPS availability and return device."""
    print("=" * 50)
    print("🍎 Apple Silicon MPS Check")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if not torch.backends.mps.is_available():
        print("\n❌ MPS not available. Falling back to CPU.")
        return torch.device("cpu")
    
    # Test MPS with a simple operation
    try:
        device = torch.device("mps")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        print(f"MPS test: Matrix multiplication on {z.device}")
        print("✅ MPS is working!")
        return device
    except Exception as e:
        print(f"❌ MPS test failed: {e}")
        return torch.device("cpu")


def benchmark_mps(device, size=5000):
    """Benchmark MPS vs CPU performance."""
    print("\n" + "=" * 50)
    print("📊 MPS vs CPU Benchmark")
    print("=" * 50)
    
    # CPU benchmark
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)
    
    start = time.time()
    for _ in range(5):
        z = torch.mm(x_cpu, y_cpu)
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.3f}s")
    
    if device.type == "mps":
        # MPS benchmark
        x_mps = x_cpu.to(device)
        y_mps = y_cpu.to(device)
        
        # Warm up
        torch.mm(x_mps, y_mps)
        
        start = time.time()
        for _ in range(5):
            z = torch.mm(x_mps, y_mps)
        mps_time = time.time() - start
        print(f"MPS time: {mps_time:.3f}s")
        print(f"Speedup: {cpu_time/mps_time:.2f}x")
    
    print()


def train_fraud_model(device, epochs=10):
    """Train fraud detection model using MPS."""
    print("\n" + "=" * 50)
    print("🔍 Training Fraud Detection Model")
    print("=" * 50)
    
    from training.synthetic_data import SyntheticDataGenerator
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import joblib
    
    print(f"Device: {device}")
    print(f"Epochs (n_estimators): {epochs}")
    
    # Generate synthetic training data
    print("\n📊 Generating synthetic training data...")
    generator = SyntheticDataGenerator(seed=42)
    dataset = generator.generate(n_samples_per_regime=2500)
    
    X = dataset.features_50
    y = dataset.rug_labels  # Use rug labels as fraud target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    
    # Convert to tensors for MPS (if available)
    if device.type == "mps":
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        print(f"Training data on: {X_train_t.device}")
    
    # Train model
    print("\n🚀 Training model...")
    start_time = time.time()
    
    model = GradientBoostingClassifier(
        n_estimators=epochs * 10,  # Scale epochs to n_estimators
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"\n✅ Training completed in {train_time:.2f}s")
    
    # Evaluate
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC: {auc:.4f}")
    
    # Save model
    model_path = Path(__file__).parent.parent / "data" / "models" / "fraud_model_v1.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model, auc


def train_xgboost_mps(device, epochs=10):
    """Train XGBoost model with MPS for tensor operations."""
    print("\n" + "=" * 50)
    print("🚀 Training XGBoost Model (with MPS tensors)")
    print("=" * 50)
    
    from training.synthetic_data import SyntheticDataGenerator
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    
    print(f"Device: {device}")
    
    # Generate data
    print("\n📊 Generating training data...")
    generator = SyntheticDataGenerator(seed=42)
    dataset = generator.generate(n_samples_per_regime=3000)
    
    X = dataset.features_50
    y = dataset.rug_labels
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X.shape[1]}")
    
    # Test MPS tensor creation
    if device.type == "mps":
        X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        print(f"MPS tensor shape: {X_tensor.shape}, device: {X_tensor.device}")
    
    # Train XGBoost
    print("\n🚀 Training XGBoost...")
    start_time = time.time()
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.1,
        'seed': 42,
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=epochs * 10,
        evals=[(dtest, 'test')],
        verbose_eval=20
    )
    
    train_time = time.time() - start_time
    print(f"\n✅ Training completed in {train_time:.2f}s")
    
    # Evaluate
    y_pred = model.predict(dtest)
    auc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC: {auc:.4f}")
    
    # Save model
    model_path = Path(__file__).parent.parent / "data" / "models" / "xgboost_fraud_v1.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    print(f"Model saved to: {model_path}")
    
    return model, auc


def train_neural_mps(device, epochs=10):
    """Train a simple neural network using MPS directly."""
    print("\n" + "=" * 50)
    print("🧠 Training Neural Network (Full MPS)")
    print("=" * 50)
    
    from training.synthetic_data import SyntheticDataGenerator
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    
    # Generate data
    print("\n📊 Generating training data...")
    generator = SyntheticDataGenerator(seed=42)
    dataset = generator.generate(n_samples_per_regime=2500)
    
    X = dataset.features_50.astype(np.float32)
    y = dataset.rug_labels.astype(np.float32)
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device).unsqueeze(1)
    X_test_t = torch.tensor(X_test, device=device)
    y_test_t = torch.tensor(y_test, device=device).unsqueeze(1)
    
    print(f"Training data on: {X_train_t.device}")
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X.shape[1]}")
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # Define model
    class FraudNet(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.net(x)
    
    model = FraudNet(X.shape[1]).to(device)
    print(f"\nModel architecture:\n{model}")
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print("\n🚀 Training on MPS...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t).item()
            
            # Calculate AUC
            y_pred_np = test_outputs.cpu().numpy()
            y_test_np = y_test_t.cpu().numpy()
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test_np, y_pred_np)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {test_loss:.4f} - AUC: {auc:.4f}")
    
    train_time = time.time() - start_time
    print(f"\n✅ Training completed in {train_time:.2f}s")
    print(f"Final AUC-ROC: {auc:.4f}")
    
    # Save model
    model_path = Path(__file__).parent.parent / "data" / "models" / "neural_fraud_v1.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    return model, auc


def main():
    parser = argparse.ArgumentParser(description="2DEXY MPS Training Runner")
    parser.add_argument(
        "--model", 
        choices=["fraud", "xgboost", "neural", "all"],
        default="fraud",
        help="Model to train"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run MPS benchmark"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("   2DEXY Native MPS Training (Apple Silicon GPU)")
    print("=" * 60)
    
    # Check MPS
    device = check_mps()
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_mps(device)
    
    # Train model(s)
    if args.model == "fraud" or args.model == "all":
        train_fraud_model(device, args.epochs)
    
    if args.model == "xgboost" or args.model == "all":
        train_xgboost_mps(device, args.epochs)
    
    if args.model == "neural" or args.model == "all":
        train_neural_mps(device, args.epochs)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
