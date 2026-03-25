"""
Training sample validation endpoints.

Provides REST API endpoints for validating training data quality.
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/learning", tags=["learning"])


class TrainingSampleValidationRequest(BaseModel):
    """Request to validate training samples."""

    samples: list[dict[str, Any]] = Field(..., description="List of training samples to validate")
    strict_mode: bool = Field(default=False, description="Enable strict validation mode")
    min_samples: int = Field(default=100, description="Minimum number of samples required")


class TrainingSampleValidationResponse(BaseModel):
    """Response from training sample validation."""

    valid: bool = Field(..., description="Whether validation passed")
    total_samples: int = Field(..., description="Total samples checked")
    valid_samples: int = Field(..., description="Number of valid samples")
    invalid_samples: int = Field(..., description="Number of invalid samples")
    feature_completeness: float = Field(..., description="Feature completeness score")
    label_consistency: float = Field(..., description="Label consistency score")
    issues: list[dict[str, Any]] = Field(
        default_factory=list, description="List of validation issues"
    )
    summary: str = Field(..., description="Human-readable summary")


class ValidationStatsResponse(BaseModel):
    """Validation statistics."""

    total_validations: int = Field(..., description="Total validations performed")
    pass_rate: float = Field(..., description="Validation pass rate")
    avg_completeness: float = Field(..., description="Average feature completeness")
    avg_label_consistency: float = Field(..., description="Average label consistency")
    common_issues: list[dict[str, Any]] = Field(
        default_factory=list, description="Most common validation issues"
    )


# Add this get_db dependency function:


def get_db():
    """Get database dependency."""
    # This would normally be injected from the app state
    # For now, return None to indicate unavailability
    return None


DbDep = Annotated[Any, Depends(get_db)]


# Add these endpoint functions at the end:


@router.post("/validate-samples", response_model=TrainingSampleValidationResponse)
async def validate_training_samples(
    request: TrainingSampleValidationRequest,
):
    """Validate a batch of training samples.

    Performs comprehensive validation including:
    - Feature vector completeness (all 50 features)
    - Feature value sanity (within bounds, no NaN/Inf)
    - Label correctness and consistency
    - Data distribution quality

    Returns detailed validation results with issues categorized by severity.
    """
    try:
        from coldpath.validation.training_sample_validator import TrainingSampleValidator

        validator = TrainingSampleValidator(
            strict_mode=request.strict_mode,
            min_samples=request.min_samples,
        )

        result = validator.validate_samples(request.samples)

        return TrainingSampleValidationResponse(
            valid=result.valid,
            total_samples=result.total_samples,
            valid_samples=result.valid_samples,
            invalid_samples=result.invalid_samples,
            feature_completeness=result.feature_completeness,
            label_consistency=result.label_consistency,
            issues=[issue.to_dict() for issue in result.issues],
            summary=result.summary(),
        )

    except Exception as e:
        logger.error(f"Failed to validate training samples: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}") from e


@router.get("/validation/stats", response_model=ValidationStatsResponse)
async def get_validation_stats():
    """Get validation statistics.

    Returns aggregated statistics from recent validations including:
    - Total validations performed
    - Pass rate
    - Average quality metrics
    - Most common issues
    """
    # For now, return mock data
    # In production, this would query a validation history store
    return ValidationStatsResponse(
        total_validations=42,
        pass_rate=0.85,
        avg_completeness=0.92,
        avg_label_consistency=0.95,
        common_issues=[
            {
                "code": "FEATURE_OUT_OF_BOUNDS",
                "count": 12,
                "severity": "warning",
            },
            {
                "code": "DUPLICATE_SAMPLES",
                "count": 5,
                "severity": "warning",
            },
        ],
    )


@router.post("/validate-from-db")
async def validate_samples_from_database(
    db: DbDep = None,
    since_hours: int = 168,
    limit: int = 1000,
):
    """Validate training samples directly from the database.

    Fetches recent training samples from the database and validates them.
    Useful for checking data quality before triggering model training.

    Args:
        since_hours: How many hours back to fetch samples (default: 168 = 7 days)
        limit: Maximum number of samples to validate
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        # Fetch samples from database
        samples = await db.get_training_outcomes(
            since_hours=since_hours,
            limit=limit,
        )

        if not samples:
            return {
                "valid": False,
                "message": "No training samples found in database",
                "total_samples": 0,
            }

        # Validate samples
        from coldpath.validation.training_sample_validator import TrainingSampleValidator

        validator = TrainingSampleValidator(
            strict_mode=False,
            min_samples=50,
        )

        result = validator.validate_samples(samples)

        # Get feature distribution stats
        feature_stats = validator.validate_feature_distribution(samples)

        return {
            "valid": result.valid,
            "total_samples": result.total_samples,
            "valid_samples": result.valid_samples,
            "invalid_samples": result.invalid_samples,
            "feature_completeness": result.feature_completeness,
            "label_consistency": result.label_consistency,
            "positive_ratio": result.positive_ratio,
            "uniqueness_score": result.uniqueness_score,
            "issues": [issue.to_dict() for issue in result.issues[:10]],  # Top 10 issues
            "feature_stats": {
                name: {
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "nan_ratio": stats["nan_ratio"],
                }
                for name, stats in list(feature_stats.items())[:10]  # Top 10 features
            },
            "summary": result.summary(),
        }

    except Exception as e:
        logger.error(f"Failed to validate samples from database: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}") from e
