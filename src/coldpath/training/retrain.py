"""
Retrain CLI - Train and deploy profitability model

Usage:
    python -m coldpath.training.retrain
    python -m coldpath.training.retrain --output-path custom/path.json
    python -m coldpath.training.retrain --data-path custom_data.jsonl

This will:
1. Load training samples from all available locations
2. Train XGBoost model with proper train/val/test split
3. Export to HotPath-compatible JSON format
4. Copy to both artifact locations (EngineHotPath and sniperdesk)
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_training_data() -> list[Path]:
    """Find all training data files."""
    base = Path(__file__).parent.parent.parent.parent
    data_locations = [
        base / "data" / "training" / "training_samples.jsonl",
        base / "data" / "training" / "collected_tokens.jsonl",
        base / "data" / "training_samples.jsonl",
        base / "data" / "collected_tokens.jsonl",
    ]
    return [p for p in data_locations if p.exists()]


def main():
    parser = argparse.ArgumentParser(description="Retrain profitability model")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Custom output path for trained model",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Custom training data path",
    )
    parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Train but do not deploy to artifact locations",
    )
    args = parser.parse_args()

    # Import trainer
    from coldpath.training.v3_profitability_trainer import V3ProfitabilityTrainer

    # Determine data source
    if args.data_path:
        data_path = Path(args.data_path)
        if not data_path.exists():
            logger.error(f"Data path does not exist: {data_path}")
            sys.exit(1)
    else:
        data_files = find_training_data()
        if not data_files:
            logger.error("No training data found. Run data collection first.")
            sys.exit(1)
        # Use first available file (prioritize training_samples.jsonl)
        data_path = sorted(data_files, key=lambda p: "training_samples" in str(p), reverse=True)[0]

    logger.info(f"Using training data from: {data_path}")

    # Train model
    trainer = V3ProfitabilityTrainer()

    try:
        result = trainer.train_from_file(str(data_path))
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    logger.info("Training complete. Metrics:")
    logger.info(f"  Accuracy: {result.metrics.get('accuracy', 0):.4f}")
    logger.info(f"  AUC-ROC:  {result.metrics.get('auc_roc', 0):.4f}")
    logger.info(f"  F1 Score: {result.metrics.get('f1', 0):.4f}")

    # Export
    base = Path(__file__).parent.parent.parent.parent
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = base / "EngineHotPath" / "artifacts" / "profitability" / "current.json"

    exported_path = trainer.export_to_hotpath(
        result, str(output_path), dataset_id=f"retrain_{data_path.stem}"
    )
    logger.info(f"Exported model to: {exported_path}")

    # Deploy to both locations
    if not args.no_deploy:
        sniperdesk_path = base / "sniperdesk" / "artifacts" / "profitability" / "current.json"
        sniperdesk_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(output_path, sniperdesk_path)
        logger.info(f"Deployed to sniperdesk: {sniperdesk_path}")

    logger.info("Retrain complete. Restart HotPath to load new model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
