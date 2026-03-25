"""
Free ML Training Tools Integration for 2DEXY.

Integrates free/open-source ML tools for crypto trading:
- Scikit-learn (ensemble methods, preprocessing)
- XGBoost/LightGBM (gradient boosting)
- Optuna (hyperparameter optimization)
- MLflow (experiment tracking - local)
- Ray Tune (distributed tuning)
- PyTorch (deep learning)
- RLlib (reinforcement learning)
- Weights & Biases (free tier)

Based on resources from:
- https://github.com/grananqvist/Awesome-Quant-Machine-Learning-Trading
- QuantLib (quantitative finance library)
"""

import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ML model training."""

    model_type: str = "xgboost"
    objective: str = "binary:logistic"

    hyperparameters: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
    )

    optimization: str = "optuna"
    n_trials: int = 50
    cross_validation: int = 5

    early_stopping_rounds: int = 50
    eval_metric: str = "auc"

    feature_selection: bool = True
    feature_selection_method: str = "importance"
    n_features: int = 30

    ensemble: bool = False
    ensemble_method: str = "voting"
    ensemble_models: int = 5

    output_dir: str = "models"
    experiment_name: str = "default"

    use_mlflow: bool = False
    use_wandb: bool = False
    wandb_project: str = "2dexy-trading"


@dataclass
class TrainingResult:
    """Result of model training."""

    model: Any
    model_type: str
    feature_importance: dict[str, float]
    metrics: dict[str, float]
    cv_scores: list[float]
    best_params: dict[str, Any]
    training_time_seconds: float
    feature_count: int
    sample_count: int
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def save(self, path: str):
        """Save model and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        meta = {
            "model_type": self.model_type,
            "feature_importance": self.feature_importance,
            "metrics": self.metrics,
            "cv_scores": self.cv_scores,
            "best_params": self.best_params,
            "training_time_seconds": self.training_time_seconds,
            "feature_count": self.feature_count,
            "sample_count": self.sample_count,
            "timestamp": self.timestamp,
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingResult":
        """Load model and metadata."""
        path = Path(path)

        with open(path / "model.pkl", "rb") as f:
            model = pickle.load(f)

        with open(path / "metadata.json") as f:
            meta = json.load(f)

        return cls(
            model=model,
            model_type=meta["model_type"],
            feature_importance=meta["feature_importance"],
            metrics=meta["metrics"],
            cv_scores=meta["cv_scores"],
            best_params=meta["best_params"],
            training_time_seconds=meta["training_time_seconds"],
            feature_count=meta["feature_count"],
            sample_count=meta["sample_count"],
            timestamp=meta["timestamp"],
        )


class BaseTrainer(ABC):
    """Base class for ML trainers."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model = None
        self.feature_names: list[str] = []

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseTrainer":
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        pass

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance."""
        if not self.feature_names or self.model is None:
            return {}

        try:
            if hasattr(self.model, "feature_importances_"):
                importance = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                importance = np.abs(self.model.coef_).flatten()
            else:
                return {}

            return dict(zip(self.feature_names, importance.tolist(), strict=False))
        except Exception:
            return {}


class XGBoostTrainer(BaseTrainer):
    """
    XGBoost trainer with Optuna optimization.

    Best for tabular data, handles missing values,
    built-in regularization, feature importance.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__(config)
        self._xgb = None
        self._optuna = None

    def _ensure_xgboost(self):
        if self._xgb is None:
            try:
                import xgboost as xgb

                self._xgb = xgb
            except ImportError:
                raise ImportError("Install xgboost: pip install xgboost") from None
        return self._xgb

    def _ensure_optuna(self):
        if self._optuna is None:
            try:
                import optuna

                optuna.logging.set_verbosity(optuna.logging.WARNING)
                self._optuna = optuna
            except ImportError:
                raise ImportError("Install optuna: pip install optuna") from None
        return self._optuna

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        optimize: bool = True,
    ) -> "XGBoostTrainer":
        """
        Train XGBoost model with optional hyperparameter optimization.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Names of features
            optimize: Whether to run Optuna optimization
        """
        self._ensure_xgboost()

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        start_time = time.time()

        if optimize and self.config.optimization == "optuna":
            self._optimize_with_optuna(X, y, X_val, y_val)
        else:
            self._train_direct(X, y, X_val, y_val)

        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f}s")

        return self

    def _train_direct(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ):
        """Train with direct hyperparameters."""
        xgb = self._ensure_xgboost()

        params = {
            "objective": self.config.objective,
            "eval_metric": self.config.eval_metric,
            "random_state": 42,
            "n_jobs": -1,
            **self.config.hyperparameters,
        }

        dtrain = xgb.DMatrix(X, label=y)

        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "val"))

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.hyperparameters.get("n_estimators", 100),
            evals=evals,
            early_stopping_rounds=self.config.early_stopping_rounds if X_val is not None else None,
            verbose_eval=False,
        )

    def _optimize_with_optuna(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ):
        """Optimize hyperparameters with Optuna."""
        xgb = self._ensure_xgboost()
        optuna = self._ensure_optuna()

        from sklearn.model_selection import cross_val_score

        def objective(trial):
            params = {
                "objective": self.config.objective,
                "eval_metric": self.config.eval_metric,
                "random_state": 42,
                "n_jobs": -1,
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            model = (
                xgb.XGBClassifier(**params)
                if "binary" in self.config.objective
                else xgb.XGBRegressor(**params)
            )

            scores = cross_val_score(
                model,
                X,
                y,
                cv=self.config.cross_validation,
                scoring="roc_auc"
                if "binary" in self.config.objective
                else "neg_mean_squared_error",
                n_jobs=-1,
            )

            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)

        logger.info(f"Best trial: {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        self.config.hyperparameters.update(study.best_params)
        self._train_direct(X, y, X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        xgb = self._ensure_xgboost()
        dmatrix = xgb.DMatrix(X)
        if "binary" in self.config.objective:
            return (self.model.predict(dmatrix) > 0.5).astype(int)
        return self.model.predict(dmatrix)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        xgb = self._ensure_xgboost()
        dmatrix = xgb.DMatrix(X)
        probs = self.model.predict(dmatrix)
        return np.column_stack([1 - probs, probs])

    def get_feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}

        try:
            importance = self.model.get_score(importance_type="gain")
            return {self.feature_names[int(k[1:])]: v for k, v in importance.items()}
        except Exception:
            return {}


class LightGBMTrainer(BaseTrainer):
    """
    LightGBM trainer - faster than XGBoost for large datasets.

    Handles categorical features natively,
    leaf-wise tree growth for better accuracy.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__(config)
        self._lgb = None

    def _ensure_lightgbm(self):
        if self._lgb is None:
            try:
                import lightgbm as lgb

                self._lgb = lgb
            except ImportError:
                raise ImportError("Install lightgbm: pip install lightgbm") from None
        return self._lgb

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        categorical_features: list[int] | None = None,
    ) -> "LightGBMTrainer":
        lgb = self._ensure_lightgbm()

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        params = {
            "objective": self.config.objective,
            "metric": self.config.eval_metric,
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": self.config.hyperparameters.get("learning_rate", 0.1),
            "feature_fraction": self.config.hyperparameters.get("colsample_bytree", 0.8),
            "bagging_fraction": self.config.hyperparameters.get("subsample", 0.8),
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,
        }

        train_data = lgb.Dataset(
            X,
            label=y,
            feature_name=self.feature_names,
            categorical_feature=categorical_features,
        )

        valid_data = None
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [lgb.log_evaluation(period=0)]
        if valid_data and self.config.early_stopping_rounds:
            callbacks.append(lgb.early_stopping(self.config.early_stopping_rounds))

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.hyperparameters.get("n_estimators", 100),
            valid_sets=[valid_data] if valid_data else None,
            callbacks=callbacks,
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.model.predict(X)
        if "binary" in self.config.objective:
            return (probs > 0.5).astype(int)
        return probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.model.predict(X)
        return np.column_stack([1 - probs, probs])

    def get_feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importance().tolist(), strict=False))


class SklearnEnsembleTrainer(BaseTrainer):
    """
    Scikit-learn ensemble trainer.

    Combines multiple models:
    - RandomForest
    - GradientBoosting
    - AdaBoost
    - Voting/Stacking classifiers
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__(config)
        self._sklearn = None

    def _ensure_sklearn(self):
        if self._sklearn is None:
            try:
                from sklearn.ensemble import (
                    AdaBoostClassifier,
                    GradientBoostingClassifier,
                    RandomForestClassifier,
                    StackingClassifier,
                    VotingClassifier,
                )
                from sklearn.model_selection import cross_val_score

                self._sklearn = {
                    "RandomForest": RandomForestClassifier,
                    "GradientBoosting": GradientBoostingClassifier,
                    "AdaBoost": AdaBoostClassifier,
                    "Voting": VotingClassifier,
                    "Stacking": StackingClassifier,
                    "cross_val_score": cross_val_score,
                }
            except ImportError:
                raise ImportError("Install scikit-learn: pip install scikit-learn") from None
        return self._sklearn

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "SklearnEnsembleTrainer":
        sklearn = self._ensure_sklearn()

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        if self.config.ensemble:
            self._train_ensemble(X, y)
        else:
            model_class = sklearn.get(self.config.model_type, sklearn["RandomForest"])
            self.model = model_class(
                random_state=42,
                n_jobs=-1,
                **self.config.hyperparameters,
            )
            self.model.fit(X, y)

        return self

    def _train_ensemble(self, X: np.ndarray, y: np.ndarray):
        sklearn = self._ensure_sklearn()

        estimators = [
            ("rf", sklearn["RandomForest"](n_estimators=100, max_depth=10, random_state=42)),
            ("gb", sklearn["GradientBoosting"](n_estimators=100, max_depth=5, random_state=42)),
            ("ada", sklearn["AdaBoost"](n_estimators=50, random_state=42)),
        ]

        if self.config.ensemble_method == "voting":
            self.model = sklearn["Voting"](
                estimators=estimators,
                voting="soft",
                n_jobs=-1,
            )
        else:
            self.model = sklearn["Stacking"](
                estimators=estimators,
                final_estimator=sklearn["GradientBoosting"](n_estimators=50),
                n_jobs=-1,
            )

        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class ReinforcementLearningTrainer(BaseTrainer):
    """
    Reinforcement learning trainer for trading.

    Integrates:
    - RLlib (Ray) for distributed RL
    - Stable-Baselines3 for standard algorithms
    - Custom gym environments for trading

    Free and open-source.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__(config)
        self._sb3 = None
        self.env = None

    def _ensure_sb3(self):
        if self._sb3 is None:
            try:
                from stable_baselines3 import A2C, DQN, PPO
                from stable_baselines3.common.vec_env import DummyVecEnv

                self._sb3 = {
                    "PPO": PPO,
                    "A2C": A2C,
                    "DQN": DQN,
                    "DummyVecEnv": DummyVecEnv,
                }
            except ImportError:
                raise ImportError(
                    "Install stable-baselines3: pip install stable-baselines3"
                ) from None
        return self._sb3

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        env: Any | None = None,
        total_timesteps: int = 100000,
        algorithm: str = "PPO",
    ) -> "ReinforcementLearningTrainer":
        """
        Train RL agent.

        Args:
            X: Not used directly, for interface compatibility
            y: Not used directly
            env: Gym environment for trading
            total_timesteps: Number of training steps
            algorithm: RL algorithm ("PPO", "A2C", "DQN")
        """
        sb3 = self._ensure_sb3()

        if env is None:
            raise ValueError("RL training requires a gym environment")

        self.env = env

        algo_class = sb3.get(algorithm, sb3["PPO"])

        self.model = algo_class(
            "MlpPolicy",
            env,
            learning_rate=self.config.hyperparameters.get("learning_rate", 3e-4),
            n_steps=self.config.hyperparameters.get("n_steps", 2048),
            batch_size=self.config.hyperparameters.get("batch_size", 64),
            verbose=1,
        )

        self.model.learn(total_timesteps=total_timesteps)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")

        obs = self.env.reset() if hasattr(self.env, "reset") else X
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("RL agents don't support predict_proba")


class UnifiedMLPipeline:
    """
    Unified ML pipeline for crypto trading.

    Integrates all free ML tools:
    - Multiple model types (XGBoost, LightGBM, sklearn, RL)
    - Hyperparameter optimization (Optuna)
    - Feature selection
    - Cross-validation
    - Experiment tracking (MLflow, W&B free tier)

    Usage:
        pipeline = UnifiedMLPipeline(config)
        result = pipeline.train(X, y)
        predictions = pipeline.predict(X_new)
    """

    TRAINERS = {
        "xgboost": XGBoostTrainer,
        "lightgbm": LightGBMTrainer,
        "sklearn": SklearnEnsembleTrainer,
        "random_forest": SklearnEnsembleTrainer,
        "gradient_boosting": SklearnEnsembleTrainer,
        "rl": ReinforcementLearningTrainer,
    }

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.trainer: BaseTrainer | None = None
        self.result: TrainingResult | None = None
        self._mlflow = None
        self._wandb = None

    def _get_trainer(self) -> BaseTrainer:
        """Get appropriate trainer for model type."""
        model_type = self.config.model_type.lower()

        if model_type not in self.TRAINERS:
            raise ValueError(
                f"Unknown model type: {model_type}. Available: {list(self.TRAINERS.keys())}"
            )

        return self.TRAINERS[model_type](self.config)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        optimize: bool = True,
    ) -> TrainingResult:
        """
        Train model with full pipeline.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Feature names
            optimize: Run hyperparameter optimization

        Returns:
            TrainingResult with model and metrics
        """
        start_time = time.time()

        if self.config.use_mlflow:
            self._setup_mlflow()

        if self.config.use_wandb:
            self._setup_wandb()

        self.trainer = self._get_trainer()

        if self.config.feature_selection:
            X, feature_names = self._select_features(X, y, feature_names)

        if isinstance(self.trainer, ReinforcementLearningTrainer):
            self.trainer.fit(X, y)
        else:
            self.trainer.fit(X, y, X_val, y_val, feature_names, optimize=optimize)

        metrics = self._evaluate(X, y, X_val, y_val) if X_val is not None else {}

        self.result = TrainingResult(
            model=self.trainer.model,
            model_type=self.config.model_type,
            feature_importance=self.trainer.get_feature_importance(),
            metrics=metrics,
            cv_scores=[],
            best_params=self.config.hyperparameters,
            training_time_seconds=time.time() - start_time,
            feature_count=X.shape[1],
            sample_count=X.shape[0],
        )

        if self.config.output_dir:
            output_path = Path(self.config.output_dir) / self.config.experiment_name
            self.result.save(output_path)

        return self.result

    def _select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Select most important features."""
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        n_features = min(self.config.n_features, X.shape[1])

        selector = SelectKBest(
            score_func=mutual_info_classif
            if self.config.feature_selection_method == "mutual_info"
            else f_classif,
            k=n_features,
        )

        X_selected = selector.fit_transform(X, y)

        selected_mask = selector.get_support()
        selected_names = [n for n, m in zip(feature_names, selected_mask, strict=False) if m]

        logger.info(f"Selected {len(selected_names)} features from {len(feature_names)}")

        return X_selected, selected_names

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate model on validation set."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            log_loss,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_pred = self.trainer.predict(X_val)
        y_proba = self.trainer.predict_proba(X_val)

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
        }

        if y_proba.shape[1] == 2:
            metrics["auc"] = roc_auc_score(y_val, y_proba[:, 1])
            metrics["log_loss"] = log_loss(y_val, y_proba)

        return metrics

    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            import mlflow

            mlflow.set_tracking_uri(f"file://{Path(self.config.output_dir) / 'mlruns'}")
            mlflow.set_experiment(self.config.experiment_name)
            mlflow.start_run()
            self._mlflow = mlflow
        except ImportError:
            logger.warning("MLflow not installed, skipping tracking")

    def _setup_wandb(self):
        """Setup Weights & Biases tracking."""
        try:
            import wandb

            wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=self.config.__dict__,
            )
            self._wandb = wandb
        except ImportError:
            logger.warning("wandb not installed, skipping tracking")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model."""
        if self.trainer is None:
            raise ValueError("Model not trained")
        return self.trainer.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with trained model."""
        if self.trainer is None:
            raise ValueError("Model not trained")
        return self.trainer.predict_proba(X)

    def load(self, path: str) -> "UnifiedMLPipeline":
        """Load a trained model."""
        self.result = TrainingResult.load(path)
        self.trainer = self._get_trainer()
        self.trainer.model = self.result.model
        return self


FREE_ML_TOOLS = {
    "xgboost": {
        "description": "Gradient boosting framework, excellent for tabular data",
        "install": "pip install xgboost",
        "docs": "https://xgboost.readthedocs.io/",
    },
    "lightgbm": {
        "description": "Fast gradient boosting by Microsoft",
        "install": "pip install lightgbm",
        "docs": "https://lightgbm.readthedocs.io/",
    },
    "optuna": {
        "description": "Hyperparameter optimization framework",
        "install": "pip install optuna",
        "docs": "https://optuna.readthedocs.io/",
    },
    "scikit-learn": {
        "description": "Classic ML library with many algorithms",
        "install": "pip install scikit-learn",
        "docs": "https://scikit-learn.org/",
    },
    "stable-baselines3": {
        "description": "Reinforcement learning algorithms (PPO, A2C, DQN)",
        "install": "pip install stable-baselines3",
        "docs": "https://stable-baselines3.readthedocs.io/",
    },
    "ray[rllib]": {
        "description": "Distributed RL with many algorithms",
        "install": "pip install 'ray[rllib]'",
        "docs": "https://docs.ray.io/en/latest/rllib/",
    },
    "mlflow": {
        "description": "Experiment tracking (free local mode)",
        "install": "pip install mlflow",
        "docs": "https://mlflow.org/",
    },
    "wandb": {
        "description": "Experiment tracking (free tier for individuals)",
        "install": "pip install wandb",
        "docs": "https://docs.wandb.ai/",
    },
    "pytorch": {
        "description": "Deep learning framework",
        "install": "pip install torch",
        "docs": "https://pytorch.org/",
    },
    "tensorflow": {
        "description": "Deep learning framework by Google",
        "install": "pip install tensorflow",
        "docs": "https://www.tensorflow.org/",
    },
}


def list_free_ml_tools():
    """Print available free ML tools."""
    print("=== Free ML Tools for Crypto Trading ===\n")
    for name, info in FREE_ML_TOOLS.items():
        print(f"{name}:")
        print(f"  {info['description']}")
        print(f"  Install: {info['install']}")
        print(f"  Docs: {info['docs']}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    list_free_ml_tools()
