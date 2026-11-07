# 1. Imports
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

RANDOM_STATE = 42

# 2. Load and validate the semicolon-delimited file
df = pd.read_csv("bankfull.csv", sep=";")

print("Dataset shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("\nTarget counts:\n", df["y"].value_counts())
print("\nTarget percentages:\n", (df["y"].value_counts(normalize=True) * 100).round(2))

assert "y" in df.columns

# 3. Prepare features and make train/validation/test splits
# duration is deliberately excluded for realistic pre-call prediction.
X = df.drop(columns=["y", "duration"])
y = df["y"].map({"no": 0, "yes": 1})

# First reserve 20% as the final test set.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# Split remaining 80% into 60% train and 20% validation.
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25,
    random_state=RANDOM_STATE, stratify=y_trainval
)

print("Train:", X_train.shape[0])
print("Validation:", X_val.shape[0])
print("Test:", X_test.shape[0])
print("Positive rate:", round(y_train.mean(), 4), round(y_val.mean(), 4), round(y_test.mean(), 4))

# 4. Preprocessing: scale numeric variables and one-hot encode categorical variables
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical_features)
    ]
)

print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)

# 5. Define compact, practical parameter grids for all five algorithms
model_grids = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {
            "classifier__C": [0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"]
        }
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_leaf": [5, 20],
            "classifier__class_weight": [None, "balanced"]
        }
    ),
    "KNN": (
        KNeighborsClassifier(n_jobs=1),
        {
            "classifier__n_neighbors": [5, 11, 21],
            "classifier__weights": ["uniform", "distance"]
        }
    ),
    "Gaussian Naive Bayes": (
        GaussianNB(),
        {
            "classifier__var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7]
        }
    ),
    "Linear SVM": (
        LinearSVC(random_state=RANDOM_STATE, max_iter=10000),
        {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"]
        }
    )
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# 6. Run GridSearchCV; ROC-AUC is the hyperparameter-selection score
best_models = {}
tuning_rows = []

for model_name, (estimator, param_grid) in model_grids.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    search.fit(X_train, y_train)
    best_models[model_name] = search.best_estimator_

    tuning_rows.append({
        "Model": model_name,
        "Best CV ROC-AUC": search.best_score_,
        "Best Parameters": search.best_params_
    })

tuning_results = pd.DataFrame(tuning_rows).sort_values(
    "Best CV ROC-AUC", ascending=False
)
display(tuning_results.round(4))

# 7. Obtain continuous scores and optimize each threshold on validation data
# For probabilistic classifiers use P(y=1); for LinearSVC use decision_function.
def get_positive_scores(fitted_pipeline, features):
    if hasattr(fitted_pipeline, "predict_proba"):
        return fitted_pipeline.predict_proba(features)[:, 1]
    return fitted_pipeline.decision_function(features)

def find_f1_threshold(y_true, validation_scores):
    precision, recall, thresholds = precision_recall_curve(
        y_true, validation_scores
    )
    # precision and recall have one more element than thresholds.
    f1_values = (
        2 * precision[:-1] * recall[:-1] /
        (precision[:-1] + recall[:-1] + 1e-12)
    )
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])

optimized_thresholds = {}
for model_name, fitted_model in best_models.items():
    val_scores = get_positive_scores(fitted_model, X_val)
    threshold, validation_f1 = find_f1_threshold(y_val, val_scores)
    optimized_thresholds[model_name] = threshold
    print(
        f"{model_name}: threshold={threshold:.4f}, "
        f"validation F1={validation_f1:.4f}"
    )

# 8. Evaluate on the untouched test set
# Report both default decision threshold and validation-optimized threshold.
def calculate_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "AUC": roc_auc_score(y_true, scores),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1 Score": f1_score(y_true, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predictions)
    }

evaluation_rows = []
test_scores = {}
test_predictions_optimized = {}

for model_name, fitted_model in best_models.items():
    scores = get_positive_scores(fitted_model, X_test)
    test_scores[model_name] = scores

    # Default cutoff: 0.5 for probability scores, 0 for decision_function.
    default_threshold = 0.5 if hasattr(fitted_model, "predict_proba") else 0.0
    optimized_threshold = optimized_thresholds[model_name]

    for label, threshold in [
        ("Default threshold", default_threshold),
        ("F1-optimized threshold", optimized_threshold)
    ]:
        row = {
            "Model": model_name,
            "Threshold type": label,
            "Threshold": threshold
        }
        row.update(calculate_metrics(y_test, scores, threshold))
        evaluation_rows.append(row)

    test_predictions_optimized[model_name] = (
        scores >= optimized_threshold
    ).astype(int)

final_results = pd.DataFrame(evaluation_rows)
display(final_results.round(4))

# 9. Confusion matrices using the validation-optimized thresholds
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, (model_name, predictions) in enumerate(test_predictions_optimized.items()):
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"]
    )
    axes[i].set_title(model_name)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

axes[-1].axis("off")
plt.tight_layout()
plt.show()

# 10. ROC curves for the tuned models
plt.figure(figsize=(9, 6))
for model_name, scores in test_scores.items():
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc_value = roc_auc_score(y_test, scores)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Tuned model ROC curves – test set")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 11. Save parameter search and evaluation results
tuning_results.to_csv("bank_gridsearch_best_parameters.csv", index=False)
final_results.to_csv("bank_tuned_threshold_metrics.csv", index=False)

print("Saved:")
print("- bank_gridsearch_best_parameters.csv")
print("- bank_tuned_threshold_metrics.csv")

# Best model by test-set F1 at its validation-selected threshold
optimized_only = final_results[
    final_results["Threshold type"] == "F1-optimized threshold"
]
best_f1_row = optimized_only.loc[optimized_only["F1 Score"].idxmax()]
print("\nHighest test F1 (threshold chosen on validation):")
print(best_f1_row.to_string())

"""## How to interpret the results

- GridSearchCV selects hyperparameters by mean stratified 3-fold ROC-AUC on the training partition.
- The classification threshold is selected separately using validation-set F1. This avoids choosing a threshold on the final test set.
- AUC is threshold-independent, so it is identical in the default and optimized-threshold rows for each model.
- Threshold optimization changes predicted class labels and therefore can change Accuracy, Precision, Recall, F1 and MCC.
- F1 optimization does not necessarily maximize MCC, profit, or recall. If the bank has explicit costs for missed subscribers and unnecessary calls, choose a threshold that maximizes expected business value instead.
- These are baseline grids. Broader grids, repeated cross-validation, calibration, and temporal validation may improve reliability.

"""
