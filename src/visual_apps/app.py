"""
Time Series Classification - Interactive Demo Application

Run with: streamlit run src/visual_apps/app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.visual_apps.utils.model_loader import load_all_demo_models
from src.visual_apps.components.prediction_viewer import render_prediction_viewer
from src.data_loader import load_ucr_dataset


# Page configuration
st.set_page_config(
    page_title="Time Series Classification Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Fix metric text color to be visible */
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stMetric label {
        color: #31333F !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #0e1117 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #0e1117 !important;
    }
    .success {
        color: green;
        font-weight: bold;
    }
    .error {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_cached():
    """Load models once at startup."""
    checkpoint_dir = Path("checkpoints/ecg200_best")
    return load_all_demo_models(checkpoint_dir)


@st.cache_data
def load_data_cached():
    """Load ECG200 dataset once at startup."""
    dataset_name = "ECG200"
    data = load_ucr_dataset(dataset_name)

    return {
        "train_data": data["train_data"],
        "test_data": data["test_data"],
        "train_labels": data["train_labels"],
        "test_labels": data["test_labels"],
        "n_features": data["n_features"],
        "seq_length": data["seq_length"],
        "n_classes": data["n_classes"]
    }


def render_overview(models, data):
    """Render overview/landing page."""

    st.title("📈 Time Series Classification Demo")
    st.markdown("### Interactive Exploration of Deep Learning Models")

    # Display dataset info
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset", "ECG200")

    with col2:
        st.metric("Test Samples", len(data["test_labels"]))

    with col3:
        st.metric("Sequence Length", data["seq_length"])

    with col4:
        st.metric("Classes", data["n_classes"])

    st.markdown("---")

    # Introduction
    st.markdown("""
    ## Welcome!

    This interactive platform demonstrates neural network architectures for ECG heartbeat classification.
    The task is to classify heartbeats as either **Normal** or **Myocardial Infarction (MI)**.

    ### Available Models

    This demo currently supports the following models:
    """)

    # Model info
    if models:
        st.markdown(f"**{len(models)} model(s) loaded:**")

        for name, model in models.items():
            with st.expander(f"📊 {name} Classifier"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    **Performance:**
                    - Accuracy: {model.metrics['accuracy']:.2%}
                    - F1 Score: {model.metrics['f1_macro']:.4f}
                    - Precision: {model.metrics['precision']:.4f}
                    - Recall: {model.metrics['recall']:.4f}
                    """)

                with col2:
                    st.markdown(f"""
                    **Hyperparameters:**
                    """)
                    for key, value in model.hyperparams.items():
                        st.markdown(f"- `{key}`: {value}")
    else:
        st.warning("""
        ⚠️ **No models found!**

        Please train a model first by running:
        ```bash
        python scripts/train_demo_models.py
        ```

        This will train an LSTM model on the ECG200 dataset and save it to `checkpoints/ecg200_best/`.
        """)

    st.markdown("---")

    # Quick start guide
    st.markdown("""
    ### Getting Started

    1. **Prediction Viewer** (left sidebar): Explore predictions on individual ECG samples
       - Select different models to compare
       - View prediction confidence and probabilities
       - Visualize the ECG signal

    2. **Model Information**: Each page includes educational tooltips (📚) to help understand
       what you're looking at

    ### About the Dataset

    **ECG200** is a medical time series dataset containing:
    - 96-point univariate ECG recordings
    - 2 classes: Normal heartbeats and MI (myocardial infarction)
    - Real patient data for binary classification

    ### Tips

    - Use the random button (🎲) to quickly explore different samples
    - Check the "Model Information" expanders to understand model details
    - Look for patterns in correctly vs incorrectly classified samples
    """)


def main():
    """Main application entry point."""

    # Sidebar navigation
    st.sidebar.title("📈 ECG Classification")
    st.sidebar.markdown("---")

    # For MVP, we just have Overview and Prediction Viewer
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Overview", "🔍 Prediction Viewer"],
        help="Select a page to view"
    )

    # Load resources
    with st.spinner("Loading models and data..."):
        try:
            models = load_models_cached()
            data = load_data_cached()
        except Exception as e:
            st.error(f"Error loading resources: {str(e)}")
            st.stop()

    # Route to selected page
    if page == "🏠 Overview":
        render_overview(models, data)
    elif page == "🔍 Prediction Viewer":
        # Class names for ECG200
        class_names = ["Normal", "MI"]

        render_prediction_viewer(
            models=models,
            test_data=data["test_data"],
            test_labels=data["test_labels"],
            class_names=class_names
        )

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About**

    Interactive demo for time series classification benchmark.

    **Dataset:** ECG200 (96-point heartbeat signals)

    **Models:** LSTM, RNN, CNN, VAE classifiers
    """)

    # Additional info
    with st.sidebar.expander("ℹ️ Setup Instructions"):
        st.markdown("""
        **First time setup:**

        1. Install dependencies:
        ```bash
        pip install -r requirements_visual.txt
        ```

        2. Train demo model(s):
        ```bash
        python scripts/train_demo_models.py
        ```

        3. Launch demo:
        ```bash
        streamlit run src/visual_apps/app.py
        ```

        **Train all models:**
        ```bash
        python scripts/train_demo_models.py --models LSTM RNN CNN VAE
        ```
        """)


if __name__ == "__main__":
    main()
