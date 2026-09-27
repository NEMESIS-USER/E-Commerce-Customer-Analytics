"""
Regression Module
Implements Linear Regression and Tree-based Regression to predict Purchase_Amount.
Evaluates MAE, MSE, RMSE, R² and provides interactive Plotly actual vs predicted charts.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def prepare_regression_features(df_clean, target_col="Purchase_Amount"):
    """
    Prepares X and y for predicting Purchase_Amount.
    Drops Customer_ID and encodes categorical variables.
    """
    if target_col not in df_clean.columns:
        return None, None, f"Target column '{target_col}' not found."

    df_model = df_clean.copy()
    if "Customer_ID" in df_model.columns:
        df_model = df_model.drop(columns=["Customer_ID"])

    y = df_model[target_col].astype(float)
    X = df_model.drop(columns=[target_col])

    # One-hot encode categorical predictors
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y, None


def train_regression_model(
    X,
    y,
    model_type="Linear Regression",
    test_size=0.25,
    random_state=42
):
    """
    Trains selected regression algorithm on features X to predict y.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if model_type in ["Random Forest", "Random Forest Regression"]:
            model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=random_state)
        elif model_type == "Ridge Regression":
            model = Ridge(alpha=1.0)
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        pred_df = pd.DataFrame({
            "Actual_Amount": y_test.values,
            "Predicted_Amount": np.round(y_pred, 2),
            "Residual": np.round(y_test.values - y_pred, 2)
        })
        if len(pred_df) > 800:
            pred_df_plot = pred_df.sample(800, random_state=random_state).reset_index(drop=True)
        else:
            pred_df_plot = pred_df

        results = {
            "model": model,
            "model_type": model_type,
            "mae": round(mae, 2),
            "mse": round(mse, 2),
            "rmse": round(rmse, 2),
            "r2_score": round(r2, 4),
            "pred_df": pred_df_plot,
            "pred_df_full": pred_df,
            "X_train": X_train,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred
        }
        return results, None

    except Exception as e:
        return None, str(e)


def plot_actual_vs_predicted(pred_df, model_name="Regression"):
    """
    Generates an interactive Plotly scatter plot for Actual vs Predicted Purchase Amounts.
    """
    min_val = min(pred_df["Actual_Amount"].min(), pred_df["Predicted_Amount"].min())
    max_val = max(pred_df["Actual_Amount"].max(), pred_df["Predicted_Amount"].max())

    fig = px.scatter(
        pred_df,
        x="Actual_Amount",
        y="Predicted_Amount",
        color="Residual",
        color_continuous_scale="RdYlBu_r",
        title=f"Actual vs Predicted Purchase Amount ({model_name})",
        labels={"Actual_Amount": "Actual Purchase Amount (₹)", "Predicted_Amount": "Predicted Purchase Amount (₹)"},
        hover_data={"Actual_Amount": ":.2f", "Predicted_Amount": ":.2f", "Residual": ":.2f"}
    )

    # Add 45-degree ideal fit line
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal 1:1 Fit (y = x)",
            line=dict(color="#10B981", dash="dash", width=2)
        )
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


def plot_residual_distribution(pred_df):
    """
    Plots histogram of prediction residuals (errors).
    """
    fig = px.histogram(
        pred_df,
        x="Residual",
        nbins=35,
        marginal="box",
        color_discrete_sequence=["#06B6D4"],
        title="Residual Error Distribution (Actual - Predicted)"
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


def train_both_regression_models(X, y, test_size=0.25, random_state=42):
    """
    Trains Linear Regression and Random Forest Regression on the same train/test split.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model_specs = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest Regression", RandomForestRegressor(
                n_estimators=80, max_depth=8, random_state=random_state
            )),
        ]

        per_model = {}
        rows = []
        for name, model in model_specs:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            pred_df = pd.DataFrame({
                "Actual_Amount": y_test.values,
                "Predicted_Amount": np.round(y_pred, 2),
                "Residual": np.round(y_test.values - y_pred, 2)
            })
            if len(pred_df) > 800:
                pred_plot = pred_df.sample(800, random_state=random_state).reset_index(drop=True)
            else:
                pred_plot = pred_df
            packed = {
                "model_type": name,
                "mae": round(float(mae), 2),
                "mse": round(float(mse), 2),
                "rmse": round(float(np.sqrt(mse)), 2),
                "r2_score": round(float(r2), 4),
                "pred_df": pred_plot,
            }
            per_model[name] = packed
            rows.append({
                "Model": name,
                "MAE": packed["mae"],
                "MSE": packed["mse"],
                "R2": packed["r2_score"],
            })

        metrics_df = pd.DataFrame(rows)
        lr = per_model["Linear Regression"]
        rf = per_model["Random Forest Regression"]

        lr_points = 0
        rf_points = 0
        if rf["mae"] < lr["mae"]:
            rf_points += 1
        elif lr["mae"] < rf["mae"]:
            lr_points += 1
        if rf["mse"] < lr["mse"]:
            rf_points += 1
        elif lr["mse"] < rf["mse"]:
            lr_points += 1
        if rf["r2_score"] > lr["r2_score"]:
            rf_points += 1
        elif lr["r2_score"] > rf["r2_score"]:
            lr_points += 1

        if rf_points > lr_points:
            winner = "Random Forest Regression"
        elif lr_points > rf_points:
            winner = "Linear Regression"
        else:
            winner = "Random Forest Regression" if rf["r2_score"] >= lr["r2_score"] else "Linear Regression"

        winner_reason = (
            f"{winner} performed better on the selected dataset "
            f"(MAE {per_model[winner]['mae']}, MSE {per_model[winner]['mse']}, "
            f"R² {per_model[winner]['r2_score']}) using majority vote across lower error and higher R²."
        )

        return {
            "metrics_df": metrics_df,
            "models": per_model,
            "winner": winner,
            "winner_reason": winner_reason,
            "lr_points": lr_points,
            "rf_points": rf_points,
        }, None
    except Exception as e:
        return None, str(e)


def plot_regression_metrics_comparison(metrics_df):
    """Grouped comparison of R² and MAE for Linear vs Random Forest regression."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics_df["Model"],
        y=metrics_df["R2"],
        name="R² Score (higher is better)",
        marker_color="#34D399",
        text=[f"{v:.4f}" for v in metrics_df["R2"]],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=metrics_df["Model"],
        y=metrics_df["MAE"],
        name="MAE (lower is better)",
        marker_color="#38BDF8",
        text=[f"{v:,.2f}" for v in metrics_df["MAE"]],
        textposition="auto",
        yaxis="y2"
    ))
    fig.update_layout(
        title="Regression Model Metric Comparison",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.7)",
        font=dict(color="#F3F4F6", family="sans-serif"),
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis=dict(title="R²", gridcolor="rgba(255,255,255,0.08)"),
        yaxis2=dict(title="MAE (₹)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
