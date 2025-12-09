"""Prediction Viewer component for exploring model predictions."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Dict

from src.visual_apps.utils.model_loader import DemoModel, predict_single_sample


def plot_time_series(
    sample: np.ndarray,
    title: str = "ECG Heartbeat Signal"
) -> go.Figure:
    """Create interactive time series plot."""

    # Flatten sample if needed
    if sample.ndim > 1:
        sample = sample.flatten()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=sample,
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        name='Signal'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Timestep",
        yaxis_title="Normalized Value",
        template="plotly_white",
        height=300,
        hovermode='x'
    )

    return fig


def plot_probabilities(
    probabilities: np.ndarray,
    class_names: list,
    predicted_class: int,
    true_class: int
) -> go.Figure:
    """Create bar chart of class probabilities."""

    colors = ['#2ca02c' if i == true_class else '#d62728' if i == predicted_class else '#888888'
              for i in range(len(class_names))]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=class_names,
        x=probabilities * 100,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{p:.1f}%' for p in probabilities * 100],
        textposition='auto'
    ))

    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="Class",
        template="plotly_white",
        height=200,
        showlegend=False
    )

    return fig


def render_prediction_viewer(
    models: Dict[str, DemoModel],
    test_data: np.ndarray,
    test_labels: np.ndarray,
    class_names: list
):
    """
    Render the prediction viewer component.

    Args:
        models: Dictionary of loaded DemoModel instances
        test_data: Test samples, shape (n_samples, seq_length, n_features)
        test_labels: Test labels, shape (n_samples,)
        class_names: List of class names
    """

    st.title("🔍 Prediction Viewer")
    st.markdown("Explore model predictions on individual ECG heartbeat samples.")

    # Sidebar controls
    st.sidebar.markdown("### Controls")

    # Model selector
    available_models = list(models.keys())
    if not available_models:
        st.error("No models loaded! Please train models first using `python scripts/train_demo_models.py`")
        return

    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        available_models,
        help="Choose which model to use for predictions"
    )

    # Sample selector
    n_samples = len(test_data)

    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        sample_idx = st.number_input(
            "Sample Index",
            min_value=0,
            max_value=n_samples - 1,
            value=0,
            step=1,
            help=f"Select a sample (0 to {n_samples-1})"
        )

    with col2:
        if st.button("🎲", help="Random sample"):
            sample_idx = np.random.randint(0, n_samples)
            st.rerun()

    # Get selected sample and label
    sample = test_data[sample_idx]
    true_label = int(test_labels[sample_idx])
    true_class_name = class_names[true_label]

    # Get model
    demo_model = models[selected_model_name]

    # Run prediction
    with st.spinner("Running inference..."):
        prediction = predict_single_sample(demo_model, sample)

    predicted_class = prediction["predicted_class"]
    probabilities = prediction["probabilities"]
    confidence = prediction["confidence"]
    predicted_class_name = class_names[predicted_class]

    # Display results
    st.markdown("---")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Sample Index", f"#{sample_idx}")

    with col2:
        st.metric("Ground Truth", true_class_name)

    with col3:
        is_correct = predicted_class == true_label
        status_color = "🟢" if is_correct else "🔴"
        st.metric("Prediction", f"{status_color} {predicted_class_name}")

    with col4:
        st.metric("Confidence", f"{confidence:.1%}")

    # Show if prediction is correct
    if predicted_class == true_label:
        st.success(f"✅ Correct prediction! The model correctly identified this as **{predicted_class_name}**.")
    else:
        st.error(f"❌ Incorrect prediction! The model predicted **{predicted_class_name}** but the true class is **{true_class_name}**.")

    st.markdown("---")

    # Time series plot
    st.markdown("### ECG Signal")
    fig_ts = plot_time_series(sample, f"Sample #{sample_idx}: {true_class_name}")
    st.plotly_chart(fig_ts, use_container_width=True)

    # Probability plot
    st.markdown("### Prediction Probabilities")
    fig_probs = plot_probabilities(probabilities, class_names, predicted_class, true_label)
    st.plotly_chart(fig_probs, use_container_width=True)

    # Model info in expander
    with st.expander("ℹ️ Model Information"):
        st.markdown(f"""
        **Model:** {selected_model_name}

        **Hyperparameters:**
        """)
        for key, value in demo_model.hyperparams.items():
            st.markdown(f"- `{key}`: {value}")

        st.markdown(f"""
        **Performance Metrics:**
        - Accuracy: {demo_model.metrics['accuracy']:.2%}
        - F1 Score: {demo_model.metrics['f1_macro']:.4f}
        - Precision: {demo_model.metrics['precision']:.4f}
        - Recall: {demo_model.metrics['recall']:.4f}
        """)

    # Educational info
    with st.expander("📚 What am I looking at?"):
        st.markdown("""
        **Time Series Signal:** This shows the ECG heartbeat recording with 96 timesteps.
        Each point represents the electrical activity of the heart at that moment.

        **Prediction:** The model analyzes this signal and predicts whether it's a normal heartbeat
        or one associated with myocardial infarction (MI, heart attack).

        **Confidence:** This shows how certain the model is about its prediction. Higher confidence
        means the model is more certain.

        **Tips:**
        - Try different samples to see where the model succeeds and fails
        - Look for patterns in misclassified samples
        - Compare predictions across different models
        """)
