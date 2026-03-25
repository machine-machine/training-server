"""Shared Plotly chart helpers for admin dashboard."""

import plotly.graph_objects as go


def training_loss_chart(epochs: list[int], train_loss: list[float], val_loss: list[float]) -> go.Figure:
    """Training/validation loss curves."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss", mode="lines"))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss", mode="lines", line=dict(dash="dash")))
    fig.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_dark")
    return fig


def gpu_memory_timeline(timestamps: list[str], memory_mb: list[float], total_mb: float) -> go.Figure:
    """GPU VRAM usage over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=memory_mb, fill="tozeroy", name="Used VRAM"))
    fig.add_hline(y=total_mb, line_dash="dash", line_color="red", annotation_text=f"Total: {total_mb}MB")
    fig.update_layout(title="GPU Memory", xaxis_title="Time", yaxis_title="MB", template="plotly_dark")
    return fig


def metric_radar(model_a: dict, model_b: dict, label_a: str = "Model A", label_b: str = "Model B") -> go.Figure:
    """Radar chart for comparing two model versions."""
    categories = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    keys = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[model_a.get(k, 0) for k in keys],
        theta=categories,
        fill="toself",
        name=label_a,
    ))
    fig.add_trace(go.Scatterpolar(
        r=[model_b.get(k, 0) for k in keys],
        theta=categories,
        fill="toself",
        name=label_b,
    ))
    fig.update_layout(title="Model Comparison", polar=dict(radialaxis=dict(range=[0, 1])), template="plotly_dark")
    return fig
