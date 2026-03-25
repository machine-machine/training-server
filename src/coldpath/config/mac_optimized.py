"""
MacBook-Optimized ML Training Configuration for 2DEXY

Optimized for Apple Silicon (M4 Pro, 14 cores, 48GB RAM):
- MPS (Metal Performance Shaders) GPU acceleration
- Multi-core parallel processing
- Memory-efficient data loading
- ARM64 native optimizations
"""

import os
from dataclasses import dataclass, field


@dataclass
class MacOptimizedConfig:
    """Configuration optimized for MacBook M4 Pro training."""

    # Hardware specs (M4 Pro)
    cpu_cores: int = 14
    memory_gb: int = 48
    use_mps: bool = True  # Metal Performance Shaders

    # Parallel processing
    n_jobs: int = 12  # Leave 2 cores for system
    max_threads: int = 14

    # Memory management
    max_memory_fraction: float = 0.7  # Use max 70% of 48GB = ~33GB
    chunk_size: int = 100_000  # Process data in chunks
    preload_to_memory: bool = True  # 48GB allows full preload

    # XGBoost optimizations for M4
    xgboost_params: dict = field(
        default_factory=lambda: {
            "n_jobs": 12,
            "tree_method": "hist",  # Faster than exact on CPU
            "max_bin": 256,  # Faster histogram
            "enable_categorical": True,
            "device": "cpu",  # Will use MPS via PyTorch if available
        }
    )

    # LightGBM optimizations for M4
    lightgbm_params: dict = field(
        default_factory=lambda: {
            "n_jobs": 12,
            "device": "cpu",
            "force_col_wise": True,  # Better for ARM
            "max_bin": 255,
        }
    )

    # Optuna optimizations
    optuna_n_trials: int = 100
    optuna_n_jobs: int = 4  # Parallel trials
    optuna_timeout: int = 3600  # 1 hour max

    # Data settings
    cache_dir: str = "data/cache"
    output_dir: str = "models/mac_optimized"

    # Training settings
    batch_size: int = 10_000  # For neural networks
    early_stopping_rounds: int = 50
    cross_validation_folds: int = 5

    @classmethod
    def detect(cls) -> "MacOptimizedConfig":
        """Auto-detect MacBook specs."""
        import subprocess

        try:
            # Get CPU cores
            result = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True)
            cpu_cores = int(result.stdout.strip())

            # Get memory
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            mem_bytes = int(result.stdout.strip())
            memory_gb = mem_bytes // (1024**3)

            # Check for MPS
            use_mps = False
            try:
                import torch

                use_mps = torch.backends.mps.is_available()
            except Exception:
                pass

            config = cls(
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                n_jobs=max(1, cpu_cores - 2),
                use_mps=use_mps,
            )

            return config

        except Exception as e:
            print(f"Auto-detection failed, using defaults: {e}")
            return cls()

    def get_memory_limit_bytes(self) -> int:
        """Get max memory to use in bytes."""
        return int(self.memory_gb * 1024**3 * self.max_memory_fraction)

    def get_xgb_params(self, base_params: dict | None = None) -> dict:
        """Get XGBoost params optimized for M4."""
        params = self.xgboost_params.copy()
        if base_params:
            params.update(base_params)
        return params

    def get_lgb_params(self, base_params: dict | None = None) -> dict:
        """Get LightGBM params optimized for M4."""
        params = self.lightgbm_params.copy()
        if base_params:
            params.update(base_params)
        return params


# Pre-configured profiles for different use cases
TRAINING_PROFILES = {
    "fast": MacOptimizedConfig(
        n_jobs=10,
        optuna_n_trials=20,
        early_stopping_rounds=20,
    ),
    "balanced": MacOptimizedConfig(
        n_jobs=12,
        optuna_n_trials=50,
        early_stopping_rounds=50,
    ),
    "thorough": MacOptimizedConfig(
        n_jobs=12,
        optuna_n_trials=100,
        early_stopping_rounds=100,
        cross_validation_folds=10,
    ),
    "neural": MacOptimizedConfig(
        n_jobs=12,
        use_mps=True,
        batch_size=32_768,
        preload_to_memory=True,
    ),
}


def setup_mac_environment():
    """Set environment variables for optimal Mac performance."""

    config = MacOptimizedConfig.detect()

    # Set parallel processing
    os.environ["OMP_NUM_THREADS"] = str(config.n_jobs)
    os.environ["MKL_NUM_THREADS"] = str(config.n_jobs)
    os.environ["OPENBLAS_NUM_THREADS"] = str(config.n_jobs)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(config.n_jobs)
    os.environ["NUMEXPR_NUM_THREADS"] = str(config.n_jobs)

    # XGBoost
    os.environ["XGBOOST_BUILD_WITH_CCACHE"] = "1"

    # LightGBM
    os.environ["LIGHTGBM_BUILD_WITH_CCACHE"] = "1"

    # PyTorch
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Scikit-learn
    os.environ["SKLEARN_SKIP_OPENMP_TEST"] = "1"

    print("MacBook Optimized Environment:")
    print(f"  CPU Cores: {config.cpu_cores}")
    print(f"  Memory: {config.memory_gb} GB")
    print(f"  Workers: {config.n_jobs}")
    print(f"  MPS GPU: {'Yes' if config.use_mps else 'No'}")

    return config


def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage for Mac."""
    import numpy as np

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == "int64":
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                if c_min > -128 and c_max < 127:
                    df[col] = df[col].astype(np.int8)
                elif c_min > -32768 and c_max < 32767:
                    df[col] = df[col].astype(np.int16)
                elif c_min > -2147483648 and c_max < 2147483647:
                    df[col] = df[col].astype(np.int32)

        elif col_type == "float64":
            df[col] = df[col].astype(np.float32)

        elif col_type == "object":
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype("category")

    return df


if __name__ == "__main__":
    config = MacOptimizedConfig.detect()
    print("\nDetected Configuration:")
    print(f"  CPU Cores: {config.cpu_cores}")
    print(f"  Memory: {config.memory_gb} GB")
    print(f"  N Jobs: {config.n_jobs}")
    print(f"  MPS Available: {config.use_mps}")
    print(f"  Memory Limit: {config.get_memory_limit_bytes() / 1e9:.1f} GB")
