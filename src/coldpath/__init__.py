"""
Engine Cold Path

Training, optimization, and backtesting engine for SniperDesk.
"""

# ─────────────────────────────────────────────────────────────────────────
# CRITICAL: Set OMP/MKL/OpenBLAS thread limits BEFORE any imports
# This prevents "OMP: Error #179: pthread_mutex_init failed" on macOS
# when XGBoost/sklearn/numpy try to spawn threads.
# Must be at the VERY TOP before ANY imports.
# ─────────────────────────────────────────────────────────────────────────
import os as _os

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
del _os

# CRITICAL: Set multiprocessing start method BEFORE any other imports on macOS.
# PyTorch sets this to 'spawn' which causes XGBoost to deadlock.
# This must be at module level, before ANY imports that might pull in torch.
import multiprocessing as _mp  # noqa: E402

if _mp.get_start_method(allow_none=True) is None:
    try:
        _mp.set_start_method("fork")
    except RuntimeError:
        pass  # Already set by another module

__version__ = "0.1.0"

# Key exports for convenience
from coldpath.backtest import (  # noqa: E402
    BitqueryDataProvider,
    OHLCVBar,
    OptimizationConfig,
    OptunaOptimizer,
    VectorizedBacktester,
)
from coldpath.calibration import (  # noqa: E402
    BiasCalibrator,
    PaperFillCalibrator,
)
from coldpath.learning import (  # noqa: E402
    FeatureSet,
    ProfitabilityLearner,
    TrainingConfig,
    TrainingMetrics,
)
from coldpath.training import (  # noqa: E402
    FraudModel,
    RegimeDetector,
)

__all__ = [
    "__version__",
    # Learning
    "FeatureSet",
    "ProfitabilityLearner",
    "TrainingConfig",
    "TrainingMetrics",
    # Backtest
    "OHLCVBar",
    "BitqueryDataProvider",
    "VectorizedBacktester",
    "OptunaOptimizer",
    "OptimizationConfig",
    # Calibration
    "BiasCalibrator",
    "PaperFillCalibrator",
    # Training
    "FraudModel",
    "RegimeDetector",
]
