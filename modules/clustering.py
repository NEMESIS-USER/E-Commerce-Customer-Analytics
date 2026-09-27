"""
K-Means Clustering Module
Implements Customer Segmentation using K-Means clustering on behavioral attributes.
Calculates Silhouette score, Elbow curve, 2D/3D visualizations, and cluster personas.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go


def compute_elbow_curve(df_clean, feature_cols, max_k=8):
    """
    Computes sum of squared errors (Inertia) for K from 2 to max_k.
    """
    X = df_clean[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_values = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X_scaled, kmeans.labels_)
        silhouettes.append(sil)

    elbow_df = pd.DataFrame({
        "K": k_values,
        "Inertia": np.round(inertias, 2),
        "Silhouette": np.round(silhouettes, 4)
    })
    return elbow_df


def plot_elbow_curve(elbow_df):
    """
    Plots the Elbow Curve (Inertia vs Number of Clusters K) using Plotly.
    """
    fig = px.line(
        elbow_df,
        x="K",
        y="Inertia",
        markers=True,
        title="Elbow Method for Optimal K (Inertia vs Clusters)",
        labels={"K": "Number of Clusters (K)", "Inertia": "Within-Cluster Sum of Squares (Inertia)"}
    )
    fig.update_traces(line=dict(color="#06B6D4", width=3), marker=dict(size=8, color="#F59E0B"))
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


def perform_kmeans_clustering(
    df_clean,
    feature_cols=["Purchase_Amount", "Purchase_Frequency", "Customer_Rating"],
    k=3,
    random_state=42
):
    """
    Fits K-Means with k clusters on the selected behavioral features.
    """
    try:
        # Check feature existence
        missing_cols = [c for c in feature_cols if c not in df_clean.columns]
        if missing_cols:
            return None, f"Missing features for clustering: {missing_cols}"

        X = df_clean[feature_cols].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        sil_score = silhouette_score(X_scaled, labels)

        df_clustered = df_clean.copy()
        df_clustered["Cluster"] = [f"Cluster {i}" for i in labels]

        # Calculate cluster centers in original unscaled scale
        centers_scaled = kmeans.cluster_centers_
        centers_original = scaler.inverse_transform(centers_scaled)

        centers_df = pd.DataFrame(
            centers_original,
            columns=feature_cols,
            index=[f"Cluster {i}" for i in range(k)]
        ).round(2)

        # Cluster distribution
        dist_df = df_clustered["Cluster"].value_counts().reset_index()
        dist_df.columns = ["Cluster", "Count"]
        dist_df["Percentage"] = (dist_df["Count"] / len(df_clustered) * 100).round(2)
        dist_df = dist_df.sort_values(by="Cluster").reset_index(drop=True)

        results = {
            "model": kmeans,
            "scaler": scaler,
            "k": k,
            "silhouette_score": round(sil_score, 4),
            "inertia": round(kmeans.inertia_, 2),
            "df_clustered": df_clustered,
            "centers_df": centers_df,
            "distribution_df": dist_df,
            "feature_cols": feature_cols,
            "X_scaled": X_scaled,
            "labels": labels
        }
        return results, None

    except Exception as e:
        return None, str(e)


def plot_clusters_2d(df_clustered, x_col="Purchase_Frequency", y_col="Purchase_Amount"):
    """
    Renders 2D cluster scatter plot with Plotly.
    """
    plot_df = df_clustered
    if len(plot_df) > 1500:
        plot_df = plot_df.sample(1500, random_state=42)
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color="Cluster",
        hover_data=["Customer_ID", "City", "Customer_Segment", "Customer_Rating"] if "Customer_ID" in df_clustered.columns else None,
        title=f"Customer Clusters: {x_col} vs {y_col}",
        color_discrete_sequence=px.colors.qualitative.Bold
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


def plot_clusters_3d(df_clustered, feature_cols):
    """
    Renders interactive 3D cluster projection with Plotly.
    """
    if len(feature_cols) < 3:
        return None

    plot_df = df_clustered
    if len(plot_df) > 900:
        plot_df = plot_df.sample(900, random_state=42)

    fig = px.scatter_3d(
        plot_df,
        x=feature_cols[0],
        y=feature_cols[1],
        z=feature_cols[2],
        color="Cluster",
        hover_data=["Customer_ID", "Customer_Segment"] if "Customer_ID" in df_clustered.columns else None,
        title="3D Customer Behavioral Clustering Space",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(17,24,39,0.7)", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(backgroundcolor="rgba(17,24,39,0.7)", gridcolor="rgba(255,255,255,0.08)"),
            zaxis=dict(backgroundcolor="rgba(17,24,39,0.7)", gridcolor="rgba(255,255,255,0.08)")
        )
    )
    return fig
