"""
Decision Tree Classification Module (J48-style equivalent)
Implements a scikit-learn Decision Tree classifier for the Customer_Segment target.
This is not Weka J48; it is a Python/scikit-learn C4.5-style equivalent used for the practical.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def prepare_classification_features(df_clean, target_col="Customer_Segment"):
    """
    Prepares X (features) and y (target) from cleaned dataset.
    Excludes Customer_ID and target column from predictors.
    Applies one-hot encoding to categorical features.
    """
    if target_col not in df_clean.columns:
        return None, None, None, None, f"Target column '{target_col}' not found."

    df_model = df_clean.copy()
    if "Customer_ID" in df_model.columns:
        df_model = df_model.drop(columns=["Customer_ID"])

    y = df_model[target_col].astype(str)
    X = df_model.drop(columns=[target_col])

    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    feature_names = list(X.columns)
    class_names = sorted(list(y.unique()))

    return X, y, feature_names, class_names, None


def train_j48_decision_tree(
    X,
    y,
    criterion="gini",
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2,
    test_size=0.25,
    random_state=42
):
    """
    Trains a Decision Tree Classifier (J48 equivalent in Python/scikit-learn).
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        classes = sorted(list(np.unique(y)))
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose().round(4)

        # Feature importances
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        results = {
            "model": model,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "classes": classes,
            "report_df": report_df,
            "feature_importance_df": feat_imp_df,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred
        }
        return results, None

    except Exception as e:
        return None, str(e)


def plot_confusion_matrix_heatmap(cm, classes, title="Confusion Matrix"):
    """
    Renders an interactive Plotly confusion matrix heatmap with modern dark styling.
    """
    x_labels = [f"Pred: {c}" for c in classes]
    y_labels = [f"Actual: {c}" for c in classes]

    fig = px.imshow(
        cm,
        x=x_labels,
        y=y_labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        title=title
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.7)",
        font=dict(color="#F3F4F6", family="sans-serif"),
        margin=dict(l=60, r=40, t=60, b=40)
    )
    return fig


def plot_feature_importance(feat_imp_df, top_n=10):
    """
    Renders a bar chart of top predictor features.
    """
    top_df = feat_imp_df.head(top_n).sort_values(by="Importance", ascending=True)

    fig = px.bar(
        top_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Teal",
        title=f"Top {top_n} Predictor Features (Information Gain / Importance)"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.7)",
        font=dict(color="#F3F4F6", family="sans-serif"),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)")
    )
    return fig


def render_decision_tree_plot(model, feature_names, class_names, max_depth=3):
    """
    Draws a clean Matplotlib Decision Tree diagram.
    """
    fig, ax = plt.subplots(figsize=(16, 8), facecolor="#111827")
    plot_tree(
        model,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax
    )
    ax.set_facecolor("#111827")
    fig.tight_layout()
    return fig
