"""
Model export module for ONNX format.

Provides utilities to export trained models (sklearn, XGBoost, LightGBM)
to ONNX format for deployment in HotPath's OnnxRunner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports for ONNX conversion
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    SKLEARN_ONNX_AVAILABLE = True
except ImportError:
    SKLEARN_ONNX_AVAILABLE = False
    logger.warning("skl2onnx not available - sklearn ONNX export disabled")

try:
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxmlFloatTensorType

    ONNXMLTOOLS_AVAILABLE = True
except ImportError:
    ONNXMLTOOLS_AVAILABLE = False
    logger.warning("onnxmltools not available - XGBoost ONNX export disabled")

try:
    import onnx
    from onnx import numpy_helper  # noqa: F401

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnx not available - ONNX validation disabled")


@dataclass
class OnnxExportResult:
    """Result of ONNX model export."""

    path: Path
    feature_count: int
    input_name: str
    output_name: str
    model_size_bytes: int
    validation_passed: bool
    validation_error: str | None = None


def validate_onnx_model(model_bytes: bytes, sample_input: np.ndarray) -> tuple[bool, str | None]:
    """
    Validate ONNX model by running inference and checking output.

    Args:
        model_bytes: Serialized ONNX model
        sample_input: Sample input for validation (1D or 2D array)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not ONNX_AVAILABLE:
        return False, "onnx package not available for validation"

    try:
        import onnxruntime as ort

        # Create session
        session = ort.InferenceSession(model_bytes)

        # Get input info
        input_info = session.get_inputs()[0]
        input_name = input_info.name

        # Prepare input
        if sample_input.ndim == 1:
            sample_input = sample_input.reshape(1, -1)

        # Run inference
        outputs = session.run(None, {input_name: sample_input.astype(np.float32)})

        # Check output
        if not outputs or len(outputs) == 0:
            return False, "Model produced no outputs"

        output = outputs[0]
        if not isinstance(output, np.ndarray):
            return False, f"Expected numpy array output, got {type(output)}"

        if output.size == 0:
            return False, "Model produced empty output"

        # Check for NaN/Inf
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            return False, "Model produced NaN or Inf values"

        return True, None

    except ImportError:
        return False, "onnxruntime not available for validation"
    except Exception as e:
        return False, str(e)


def export_sklearn_to_onnx(
    model: Any,
    feature_count: int,
    output_path: Path,
    input_name: str = "features",
    output_name: str = "score",
    validate: bool = True,
    sample_input: np.ndarray | None = None,
) -> OnnxExportResult:
    """
    Export sklearn model to ONNX format.

    Args:
        model: Trained sklearn model (LogisticRegression, RandomForest, etc.)
        feature_count: Number of input features
        output_path: Path to save the ONNX model
        input_name: Name of input tensor
        output_name: Name of output tensor
        validate: Whether to validate the exported model
        sample_input: Sample input for validation (optional, will create if not provided)

    Returns:
        OnnxExportResult with export details
    """
    if not SKLEARN_ONNX_AVAILABLE:
        raise ImportError(
            "skl2onnx is required for sklearn ONNX export. Install with: pip install skl2onnx"
        )

    logger.info(f"Exporting sklearn model to ONNX: {output_path}")

    # Define input type
    initial_type = [(input_name, FloatTensorType([None, feature_count]))]

    # Convert to ONNX
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12,  # Use opset 12 for compatibility
        options={id(model): {"zipmap": False}},  # Disable zipmap for cleaner output
    )

    # Set output name if different from default
    if output_name != "score":
        for output in onnx_model.graph.output:
            output.name = output_name

    # Serialize to bytes
    model_bytes = onnx_model.SerializeToString()

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(model_bytes)

    logger.info(f"ONNX model saved: {output_path} ({len(model_bytes)} bytes)")

    # Validate if requested
    validation_passed = True
    validation_error = None
    if validate:
        if sample_input is None:
            sample_input = np.random.randn(feature_count).astype(np.float32)
        validation_passed, validation_error = validate_onnx_model(model_bytes, sample_input)
        if validation_passed:
            logger.info("ONNX model validation passed")
        else:
            logger.warning(f"ONNX model validation failed: {validation_error}")

    return OnnxExportResult(
        path=output_path,
        feature_count=feature_count,
        input_name=input_name,
        output_name=output_name,
        model_size_bytes=len(model_bytes),
        validation_passed=validation_passed,
        validation_error=validation_error,
    )


def export_xgboost_to_onnx(
    model: Any,
    feature_count: int,
    output_path: Path,
    input_name: str = "features",
    output_name: str = "score",
    validate: bool = True,
    sample_input: np.ndarray | None = None,
) -> OnnxExportResult:
    """
    Export XGBoost model to ONNX format.

    Args:
        model: Trained XGBoost model (XGBClassifier or XGBRegressor)
        feature_count: Number of input features
        output_path: Path to save the ONNX model
        input_name: Name of input tensor
        output_name: Name of output tensor
        validate: Whether to validate the exported model
        sample_input: Sample input for validation (optional)

    Returns:
        OnnxExportResult with export details
    """
    if not ONNXMLTOOLS_AVAILABLE:
        raise ImportError(
            "onnxmltools is required for XGBoost ONNX export. Install with: pip install onnxmltools"
        )

    logger.info(f"Exporting XGBoost model to ONNX: {output_path}")

    # Define input type
    initial_type = [(input_name, OnnxmlFloatTensorType([None, feature_count]))]

    # Convert to ONNX
    onnx_model = convert_xgboost(
        model,
        initial_types=initial_type,
        target_opset=12,
    )

    # Set output name
    if output_name != "score":
        for output in onnx_model.graph.output:
            output.name = output_name

    # Serialize to bytes
    model_bytes = onnx_model.SerializeToString()

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(model_bytes)

    logger.info(f"XGBoost ONNX model saved: {output_path} ({len(model_bytes)} bytes)")

    # Validate if requested
    validation_passed = True
    validation_error = None
    if validate:
        if sample_input is None:
            sample_input = np.random.randn(feature_count).astype(np.float32)
        validation_passed, validation_error = validate_onnx_model(model_bytes, sample_input)
        if validation_passed:
            logger.info("ONNX model validation passed")
        else:
            logger.warning(f"ONNX model validation failed: {validation_error}")

    return OnnxExportResult(
        path=output_path,
        feature_count=feature_count,
        input_name=input_name,
        output_name=output_name,
        model_size_bytes=len(model_bytes),
        validation_passed=validation_passed,
        validation_error=validation_error,
    )


def export_lightgbm_to_onnx(
    model: Any,
    feature_count: int,
    output_path: Path,
    input_name: str = "features",
    output_name: str = "score",
    validate: bool = True,
    sample_input: np.ndarray | None = None,
) -> OnnxExportResult:
    """
    Export LightGBM model to ONNX format.

    Args:
        model: Trained LightGBM model
        feature_count: Number of input features
        output_path: Path to save the ONNX model
        input_name: Name of input tensor
        output_name: Name of output tensor
        validate: Whether to validate the exported model
        sample_input: Sample input for validation (optional)

    Returns:
        OnnxExportResult with export details
    """
    try:
        from onnxmltools import convert_lightgbm
    except ImportError:
        raise ImportError(
            "onnxmltools is required for LightGBM ONNX export. "
            "Install with: pip install onnxmltools"
        ) from None

    logger.info(f"Exporting LightGBM model to ONNX: {output_path}")

    # Define input type
    initial_type = [(input_name, OnnxmlFloatTensorType([None, feature_count]))]

    # Convert to ONNX
    onnx_model = convert_lightgbm(
        model,
        initial_types=initial_type,
        target_opset=12,
    )

    # Set output name
    if output_name != "score":
        for output in onnx_model.graph.output:
            output.name = output_name

    # Serialize to bytes
    model_bytes = onnx_model.SerializeToString()

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(model_bytes)

    logger.info(f"LightGBM ONNX model saved: {output_path} ({len(model_bytes)} bytes)")

    # Validate if requested
    validation_passed = True
    validation_error = None
    if validate:
        if sample_input is None:
            sample_input = np.random.randn(feature_count).astype(np.float32)
        validation_passed, validation_error = validate_onnx_model(model_bytes, sample_input)
        if validation_passed:
            logger.info("ONNX model validation passed")
        else:
            logger.warning(f"ONNX model validation failed: {validation_error}")

    return OnnxExportResult(
        path=output_path,
        feature_count=feature_count,
        input_name=input_name,
        output_name=output_name,
        model_size_bytes=len(model_bytes),
        validation_passed=validation_passed,
        validation_error=validation_error,
    )


def load_onnx_as_bytes(path: Path) -> bytes:
    """Load ONNX model file as bytes for embedding in artifact."""
    with open(path, "rb") as f:
        return f.read()


def get_onnx_model_info(model_bytes: bytes) -> dict[str, Any]:
    """Get information about an ONNX model."""
    if not ONNX_AVAILABLE:
        return {"error": "onnx package not available"}

    try:
        model = onnx.load_from_string(model_bytes)

        inputs = []
        for inp in model.graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append("dynamic")
            inputs.append(
                {
                    "name": inp.name,
                    "type": onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type),
                    "shape": shape,
                }
            )

        outputs = []
        for out in model.graph.output:
            shape = []
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append("dynamic")
            outputs.append(
                {
                    "name": out.name,
                    "type": onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type),
                    "shape": shape,
                }
            )

        return {
            "ir_version": model.ir_version,
            "opset_version": max(op.version for op in model.opset_import),
            "producer": model.producer_name,
            "inputs": inputs,
            "outputs": outputs,
        }
    except Exception as e:
        return {"error": str(e)}


def compare_sklearn_onnx_outputs(
    sklearn_model: Any,
    onnx_bytes: bytes,
    sample_input: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> tuple[bool, float, str | None]:
    """
    Compare outputs from sklearn model and ONNX model.

    Args:
        sklearn_model: Original sklearn model
        onnx_bytes: Serialized ONNX model
        sample_input: Test input
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        Tuple of (outputs_match, max_diff, error_message)
    """
    try:
        import onnxruntime as ort

        # Get sklearn prediction
        if hasattr(sklearn_model, "predict_proba"):
            sklearn_output = sklearn_model.predict_proba(sample_input)[:, 1]
        else:
            sklearn_output = sklearn_model.predict(sample_input)

        # Get ONNX prediction
        session = ort.InferenceSession(onnx_bytes)
        input_name = session.get_inputs()[0].name

        if sample_input.ndim == 1:
            sample_input = sample_input.reshape(1, -1)

        onnx_output = session.run(None, {input_name: sample_input.astype(np.float32)})[0]

        # Handle different output shapes
        if onnx_output.ndim == 2 and onnx_output.shape[1] == 1:
            onnx_output = onnx_output.flatten()
        elif onnx_output.ndim == 2 and onnx_output.shape[1] == 2:
            # Binary classification - get positive class probability
            onnx_output = onnx_output[:, 1]

        # Compare
        max_diff = np.max(np.abs(sklearn_output - onnx_output))
        outputs_match = np.allclose(sklearn_output, onnx_output, rtol=rtol, atol=atol)

        return outputs_match, float(max_diff), None

    except Exception as e:
        return False, float("inf"), str(e)
