# Training Server Copy Manifest

This manifest defines the exact source files to copy from this repository into the new `training-server/` workspace to replicate the ML backend logic.

## 1) Backend ML Files (exact list)

```text
EngineColdPath/pyproject.toml
EngineColdPath/scripts/evaluate_profit_models.py
EngineColdPath/scripts/run_training_pipeline.py
EngineColdPath/scripts/test_training_api.py
EngineColdPath/scripts/train_synthetic.py
EngineColdPath/scripts/train_v3_model.py
EngineColdPath/scripts/validate_data_flow.py
EngineColdPath/src/coldpath/api/__init__.py
EngineColdPath/src/coldpath/api/routes/__init__.py
EngineColdPath/src/coldpath/api/routes/admin.py
EngineColdPath/src/coldpath/api/routes/advanced_optimizer.py
EngineColdPath/src/coldpath/api/routes/advanced_optimizers.py
EngineColdPath/src/coldpath/api/routes/ai.py
EngineColdPath/src/coldpath/api/routes/autotrader.py
EngineColdPath/src/coldpath/api/routes/backtest.py
EngineColdPath/src/coldpath/api/routes/backtest_guided.py
EngineColdPath/src/coldpath/api/routes/daily_optimizer.py
EngineColdPath/src/coldpath/api/routes/feedback_loop.py
EngineColdPath/src/coldpath/api/routes/learning.py
EngineColdPath/src/coldpath/api/routes/ml.py
EngineColdPath/src/coldpath/api/routes/optimization.py
EngineColdPath/src/coldpath/api/routes/settings.py
EngineColdPath/src/coldpath/api/routes/telemetry.py
EngineColdPath/src/coldpath/api/server.py
EngineColdPath/src/coldpath/autotrader/__init__.py
EngineColdPath/src/coldpath/autotrader/adaptive_limits.py
EngineColdPath/src/coldpath/autotrader/autonomous_mode.py
EngineColdPath/src/coldpath/autotrader/coordinator.py
EngineColdPath/src/coldpath/autotrader/monitoring.py
EngineColdPath/src/coldpath/autotrader/operation_modes.py
EngineColdPath/src/coldpath/autotrader/training_integration.py
EngineColdPath/src/coldpath/autotrader/validation_pipeline.py
EngineColdPath/src/coldpath/calibration/__init__.py
EngineColdPath/src/coldpath/calibration/bias_calibrator.py
EngineColdPath/src/coldpath/calibration/latency.py
EngineColdPath/src/coldpath/calibration/paper_fill.py
EngineColdPath/src/coldpath/calibration/regime_calibrator.py
EngineColdPath/src/coldpath/calibration/survivorship_model.py
EngineColdPath/src/coldpath/distillation/__init__.py
EngineColdPath/src/coldpath/distillation/artifact_exporter.py
EngineColdPath/src/coldpath/distillation/pipeline_orchestrator.py
EngineColdPath/src/coldpath/distillation/student_distiller.py
EngineColdPath/src/coldpath/distillation/teacher_trainer.py
EngineColdPath/src/coldpath/ipc/__init__.py
EngineColdPath/src/coldpath/ipc/coldpath_pb2.py
EngineColdPath/src/coldpath/ipc/coldpath_pb2_grpc.py
EngineColdPath/src/coldpath/ipc/hotpath_client.py
EngineColdPath/src/coldpath/ipc/hotpath_pb2.py
EngineColdPath/src/coldpath/ipc/hotpath_pb2_grpc.py
EngineColdPath/src/coldpath/learning/__init__.py
EngineColdPath/src/coldpath/learning/advanced_regime.py
EngineColdPath/src/coldpath/learning/announcement_learner.py
EngineColdPath/src/coldpath/learning/ensemble.py
EngineColdPath/src/coldpath/learning/feature_engineering.py
EngineColdPath/src/coldpath/learning/feedback_loop.py
EngineColdPath/src/coldpath/learning/lstm_predictor.py
EngineColdPath/src/coldpath/learning/model_updater.py
EngineColdPath/src/coldpath/learning/mutual_information.py
EngineColdPath/src/coldpath/learning/online_learner.py
EngineColdPath/src/coldpath/learning/outcome_tracker.py
EngineColdPath/src/coldpath/learning/profitability_learner.py
EngineColdPath/src/coldpath/learning/proposal_ranker.py
EngineColdPath/src/coldpath/learning/regime_detector.py
EngineColdPath/src/coldpath/learning/signal_generator.py
EngineColdPath/src/coldpath/learning/stacking_ensemble.py
EngineColdPath/src/coldpath/main.py
EngineColdPath/src/coldpath/publishing/__init__.py
EngineColdPath/src/coldpath/publishing/model_publisher.py
EngineColdPath/src/coldpath/publishing/param_pusher.py
EngineColdPath/src/coldpath/storage.py
EngineColdPath/src/coldpath/storage_backend.py
EngineColdPath/src/coldpath/storage_postgres.py
EngineColdPath/src/coldpath/training/__init__.py
EngineColdPath/src/coldpath/training/anti_overfit.py
EngineColdPath/src/coldpath/training/bandit.py
EngineColdPath/src/coldpath/training/contextual_bandit.py
EngineColdPath/src/coldpath/training/ensemble_trainer.py
EngineColdPath/src/coldpath/training/fraud_model.py
EngineColdPath/src/coldpath/training/hierarchical_rl.py
EngineColdPath/src/coldpath/training/isolation_forest_detector.py
EngineColdPath/src/coldpath/training/ppo_agent.py
EngineColdPath/src/coldpath/training/regime_detector.py
EngineColdPath/src/coldpath/training/rl_reward.py
EngineColdPath/src/coldpath/training/slippage_model.py
EngineColdPath/src/coldpath/training/synthetic_data.py
EngineColdPath/src/coldpath/training/synthetic_temporal.py
EngineColdPath/src/coldpath/training/synthetic_validator.py
EngineColdPath/src/coldpath/training/v3_profitability_trainer.py
EngineColdPath/src/coldpath/training/world_model.py
EngineColdPath/src/coldpath/validation/__init__.py
EngineColdPath/src/coldpath/validation/bounds.py
EngineColdPath/src/coldpath/validation/data_quality.py
EngineColdPath/src/coldpath/validation/drift_detector.py
EngineColdPath/src/coldpath/validation/feature_audit.py
EngineColdPath/src/coldpath/validation/numeric.py
```

## 2) App Integration Files (to update for backend interface)

```text
2DEXY/Core/Services/AutoTrainingService.swift
2DEXY/Core/Utilities/AIAPIClient.swift
2DEXY/Core/Services/EngineService.swift
2DEXY/Core/Models/EngineEvents.swift
2DEXY/Core/Models/SharedModels.swift
2DEXY/Features/ResourceMonitor/RLModels.swift
2DEXY/Features/Backtesting/BacktestingViewModel.swift
```

## 3) Copy Commands

```bash
# create target tree
mkdir -p training-server/backend training-server/app-interface

# copy backend python sources
rsync -av --files-from=<(awk 'f;/^```text$/{f=1;next}/^```$/{if(f){exit}}' training-server/COPY_MANIFEST.md | rg '^EngineColdPath/') ./ training-server/backend/

# copy selected app integration references
rsync -av \
  2DEXY/Core/Services/AutoTrainingService.swift \
  2DEXY/Core/Utilities/AIAPIClient.swift \
  2DEXY/Core/Services/EngineService.swift \
  2DEXY/Core/Models/EngineEvents.swift \
  2DEXY/Core/Models/SharedModels.swift \
  2DEXY/Features/ResourceMonitor/RLModels.swift \
  2DEXY/Features/Backtesting/BacktestingViewModel.swift \
  training-server/app-interface/
```
