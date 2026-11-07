
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing ML Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Term Deposit Prediction")
st.markdown(
    "Machine Learning Binary Classification Dashboard"
)

st.write(
    "Predict whether a customer will subscribe "
    "to a bank term deposit."
)

# ------------------------------------------------
# LOAD PRE-TRAINED MODELS
# ------------------------------------------------

@st.cache_resource
def load_models():

    bundle = joblib.load("model_bundle.joblib")

    return bundle

try:
    bundle = load_models()

except Exception as e:
    st.error(f"Unable to load model bundle: {e}")
    st.stop()

models = bundle["models"]
thresholds = bundle["thresholds"]
feature_columns = bundle["feature_columns"]

# ------------------------------------------------
# SIDEBAR - MODEL SELECTION
# ------------------------------------------------

st.sidebar.header("Model Configuration")

model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)

selected_model = models[model_name]

optimized_threshold = thresholds[model_name]

use_optimized = st.sidebar.checkbox(
    "Use optimized threshold",
    value=True
)

if use_optimized:
    threshold = optimized_threshold
else:
    threshold = (
        0.5
        if hasattr(selected_model, "predict_proba")
        else 0.0
    )

st.sidebar.info(
    f"Selected model: {model_name}"
)

st.sidebar.write(
    f"Classification threshold: {threshold:.4f}"
)

# ------------------------------------------------
# DATASET UPLOAD
# ------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Please upload a test dataset containing "
        "the input columns and target column y."
    )

    st.stop()

# Read CSV and automatically detect comma/semicolon
try:
    test_df = pd.read_csv(
        uploaded_file,
        sep=None,
        engine="python"
    )

except Exception as e:
    st.error(f"Dataset reading error: {e}")
    st.stop()

st.success("Dataset uploaded successfully!")

st.write("Dataset Shape:", test_df.shape)

with st.expander("Preview uploaded dataset"):
    st.dataframe(test_df.head(10))

# ------------------------------------------------
# DATA VALIDATION
# ------------------------------------------------

if "y" not in test_df.columns:

    st.error(
        "Target column 'y' is missing. "
        "Please upload labelled test data."
    )
    st.stop()

# Remove duration because this model predicts
# subscription before the marketing call.

if "duration" in test_df.columns:
    test_df = test_df.drop(columns=["duration"])

missing_columns = [
    col for col in feature_columns
    if col not in test_df.columns
]

if missing_columns:

    st.error(
        f"Missing required columns: {missing_columns}"
    )
    st.stop()

# Keep only the model's expected input columns
X_test = test_df[feature_columns].copy()

# Convert target into binary values
y_raw = test_df["y"]

y_test = (
    y_raw.astype(str)
    .str.strip()
    .str.lower()
    .map({
        "no": 0,
        "yes": 1,
        "0": 0,
        "1": 1
    })
)

if y_test.isna().any():

    st.error(
        "Target column y must contain only yes/no "
        "or 0/1 values."
    )
    st.stop()

y_test = y_test.astype(int)

if y_test.nunique() != 2:

    st.error(
        "Test data must contain both target classes "
        "to calculate ROC-AUC."
    )
    st.stop()

# ------------------------------------------------
# MODEL PREDICTION
# ------------------------------------------------

st.header("2. Model Evaluation")

if st.button("Run Model Evaluation", type="primary"):

    with st.spinner("Evaluating selected model..."):

        # Get continuous prediction scores

        if hasattr(selected_model, "predict_proba"):

            y_score = selected_model.predict_proba(
                X_test
            )[:, 1]

        else:

            y_score = selected_model.decision_function(
                X_test
            )

        # Apply selected threshold

        y_pred = (
            y_score >= threshold
        ).astype(int)

        # ------------------------------------------------
        # EVALUATION METRICS
        # ------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        auc = roc_auc_score(y_test, y_score)

        precision = precision_score(
            y_test, y_pred, zero_division=0
        )

        recall = recall_score(
            y_test, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test, y_pred
        )

        # ------------------------------------------------
        # DISPLAY METRICS
        # ------------------------------------------------

        st.subheader("3. Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        col2.metric(
            "AUC Score",
            f"{auc:.4f}"
        )

        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        col6.metric(
            "MCC Score",
            f"{mcc:.4f}"
        )

        # ------------------------------------------------
        # CONFUSION MATRIX
        # ------------------------------------------------

        st.subheader("4. Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No", "Yes"]
        )

        disp.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False
        )

        ax.set_title("Confusion Matrix")

        st.pyplot(fig)

        plt.close(fig)

        # ------------------------------------------------
        # CLASSIFICATION REPORT
        # ------------------------------------------------

        st.subheader("5. Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "No Subscription",
                "Subscription"
            ],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        # ------------------------------------------------
        # DOWNLOAD PREDICTIONS
        # ------------------------------------------------

        st.subheader("6. Download Predictions")

        output_df = test_df.copy()

        output_df["Actual"] = y_test.values

        output_df["Predicted"] = y_pred

        output_df["Prediction_Score"] = y_score

        output_df["Prediction_Label"] = np.where(
            y_pred == 1,
            "Yes",
            "No"
        )

        csv = output_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="bank_prediction_results.csv",
            mime="text/csv"
        )

        st.success(
            f"Evaluation completed successfully using {model_name}!"
        )
