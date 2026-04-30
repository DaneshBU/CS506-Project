import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

from xgboost import XGBClassifier


def main():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load data
    url = "https://raw.githubusercontent.com/DaneshBU/CS506-Project/main/data/credit_risk_dataset.csv"
    df = pd.read_csv(url)

    print("Original dataset shape:", df.shape)

    # 2. Clean data
    df = df.drop_duplicates().copy()

    target_col = "loan_status"

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print("After removing duplicates:", df.shape)
    print("\nTarget distribution:")
    print(y.value_counts(normalize=True))

    # 3. Detect features
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # 4. Preprocessing
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 5. Train / validation / test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )

    print("\nTrain shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    # 6. Build XGBoost model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    min_child_weight=1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    gamma=0,
                    reg_alpha=1,
                    reg_lambda=1,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )

    # 7. Train model
    print("\nTraining XGBoost model...")
    model.fit(X_train, y_train)

    # 8. Validation probabilities
    val_probs = model.predict_proba(X_val)[:, 1]

    # 9. Threshold tuning
    thresholds = np.arange(0.10, 0.91, 0.05)
    threshold_results = []

    for t in thresholds:
        val_preds_t = (val_probs >= t).astype(int)

        threshold_results.append(
            {
                "threshold": t,
                "accuracy": accuracy_score(y_val, val_preds_t),
                "precision": precision_score(y_val, val_preds_t, zero_division=0),
                "recall": recall_score(y_val, val_preds_t, zero_division=0),
                "f1": f1_score(y_val, val_preds_t, zero_division=0),
            }
        )

    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv("outputs/xgboost_threshold_results.csv", index=False)

    best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
    best_threshold = best_row["threshold"]

    print("\nBest threshold based on validation F1:")
    print(best_row)

    # 10. Validation evaluation
    val_preds = (val_probs >= best_threshold).astype(int)

    print("\n=== Validation Metrics ===")
    print(f"Threshold : {best_threshold:.2f}")
    print("Accuracy  :", accuracy_score(y_val, val_preds))
    print("Precision :", precision_score(y_val, val_preds, zero_division=0))
    print("Recall    :", recall_score(y_val, val_preds, zero_division=0))
    print("F1 Score  :", f1_score(y_val, val_preds, zero_division=0))
    print("ROC AUC   :", roc_auc_score(y_val, val_probs))

    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_preds, zero_division=0))

    # 11. Test evaluation
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    print("\n=== Test Metrics ===")
    print(f"Threshold : {best_threshold:.2f}")
    print("Accuracy  :", accuracy_score(y_test, test_preds))
    print("Precision :", precision_score(y_test, test_preds, zero_division=0))
    print("Recall    :", recall_score(y_test, test_preds, zero_division=0))
    print("F1 Score  :", f1_score(y_test, test_preds, zero_division=0))
    print("ROC AUC   :", roc_auc_score(y_test, test_probs))

    print("\nTest Classification Report:")
    print(classification_report(y_test, test_preds, zero_division=0))

    # 12. Save test predictions
    test_results = X_test.copy()
    test_results["actual_default"] = y_test.values
    test_results["predicted_probability"] = test_probs
    test_results["predicted_class"] = test_preds

    test_results.to_csv("outputs/XGBoost_test_predictions.csv", index=False)

    # 13. Save feature importance
    ohe = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    encoded_cat_names = ohe.get_feature_names_out(categorical_features)

    all_feature_names = numeric_features + list(encoded_cat_names)
    importances = model.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": all_feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv("outputs/xgboost_feature_importance.csv", index=False)

    print("\nTop 20 most important features:")
    print(importance_df.head(20))

    # 14. ROC curve
    fpr, tpr, _ = roc_curve(y_val, val_probs)
    val_auc = roc_auc_score(y_val, val_probs)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"XGBoost AUC = {val_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Validation Set")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/xgboost_roc_curve.png", bbox_inches="tight")
    plt.close()

    # 15. Precision-recall curve
    precisions, recalls, _ = precision_recall_curve(y_val, val_probs)

    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Validation Set")
    plt.grid(True)
    plt.savefig("outputs/xgboost_precision_recall_curve.png", bbox_inches="tight")
    plt.close()

    # 16. Validation confusion matrix
    cm_val = confusion_matrix(y_val, val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_val)
    disp.plot()
    plt.title(f"Validation Confusion Matrix Threshold = {best_threshold:.2f}")
    plt.savefig("outputs/xgboost_validation_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # 17. Test confusion matrix
    cm_test = confusion_matrix(y_test, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    disp.plot()
    plt.title(f"Test Confusion Matrix Threshold = {best_threshold:.2f}")
    plt.savefig("outputs/xgboost_test_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # 18. Feature importance plot
    top_features = importance_df.head(20).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Importance")
    plt.title("Top 20 XGBoost Feature Importances")
    plt.grid(True, axis="x")
    plt.savefig("outputs/xgboost_feature_importance.png", bbox_inches="tight")
    plt.close()

    print("\nSaved all outputs to the outputs/ folder.")


if __name__ == "__main__":
    main()