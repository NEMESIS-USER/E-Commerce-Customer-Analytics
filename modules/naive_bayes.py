"""
Naive Bayes Classification Module
Implements Gaussian Naive Bayes classification for Customer_Segment prediction.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import plotly.express as px


def train_naive_bayes(
    X,
    y,
    test_size=0.25,
    random_state=42
):
    """
    Trains a Gaussian Naive Bayes Classifier on encoded features.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        model = GaussianNB()
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

        # Class priors
        priors_df = pd.DataFrame({
            "Class": model.classes_,
            "Prior Probability": np.round(model.class_prior_, 4)
        })

        results = {
            "model": model,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "classes": classes,
            "report_df": report_df,
            "priors_df": priors_df,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred
        }
        return results, None

    except Exception as e:
        return None, str(e)


def plot_class_priors(priors_df):
    """
    Plots the class prior probabilities learned by Naive Bayes.
    """
    fig = px.bar(
        priors_df,
        x="Class",
        y="Prior Probability",
        color="Class",
        color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B"],
        title="Naive Bayes Class Prior Probabilities P(Y)"
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
