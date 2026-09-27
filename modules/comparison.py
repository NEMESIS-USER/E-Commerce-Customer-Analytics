"""
Comparison Module
Computes live side-by-side results for association rules, classification,
regression, and clustering using the project's existing model implementations.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.association_rules import run_apriori
from modules.clustering import perform_kmeans_clustering
from modules.j48_classification import prepare_classification_features, train_j48_decision_tree
from modules.naive_bayes import train_naive_bayes
from modules.regression import prepare_regression_features, train_both_regression_models


def _metric_bar(labels, values, title, color="#38BDF8"):
    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=color,
        text=[f"{v:.4f}" if isinstance(v, float) and abs(v) <= 10 else f"{v}" for v in values],
        textposition="auto"
    ))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.7)",
        font=dict(color="#F3F4F6", family="sans-serif"),
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)")
    )
    return fig


def summarize_association(frequent_itemsets, rules, n_transactions):
    if frequent_itemsets is None or rules is None or rules.empty:
        return pd.DataFrame(), None, None

    summary_df = pd.DataFrame([
        {"Metric": "Number of transactions", "Value": int(n_transactions)},
        {"Metric": "Number of frequent itemsets", "Value": int(len(frequent_itemsets))},
        {"Metric": "Number of rules", "Value": int(len(rules))},
        {"Metric": "Average support", "Value": round(float(rules["support"].mean()), 4)},
        {"Metric": "Max support", "Value": round(float(rules["support"].max()), 4)},
        {"Metric": "Average confidence", "Value": round(float(rules["confidence"].mean()), 4)},
        {"Metric": "Max confidence", "Value": round(float(rules["confidence"].max()), 4)},
        {"Metric": "Average lift", "Value": round(float(rules["lift"].mean()), 4)},
        {"Metric": "Max lift", "Value": round(float(rules["lift"].max()), 4)},
    ])

    chart_df = pd.DataFrame({
        "Metric": ["Avg Support", "Max Support", "Avg Confidence", "Max Confidence", "Avg Lift", "Max Lift"],
        "Value": [
            float(rules["support"].mean()),
            float(rules["support"].max()),
            float(rules["confidence"].mean()),
            float(rules["confidence"].max()),
            float(rules["lift"].mean()),
            float(rules["lift"].max()),
        ]
    })
    fig = px.bar(
        chart_df,
        x="Metric",
        y="Value",
        color="Metric",
        title="Association Rule Metric Overview",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.7)",
        font=dict(color="#F3F4F6"),
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)")
    )
    return summary_df, fig, rules.iloc[0]


def classifier_table_and_chart(results, model_name, color):
    table = pd.DataFrame([
        {"Metric": "Accuracy", "Value": results["accuracy"]},
        {"Metric": "Precision", "Value": results["precision"]},
        {"Metric": "Recall", "Value": results["recall"]},
        {"Metric": "F1 Score", "Value": results["f1_score"]},
    ])
    fig = _metric_bar(
        table["Metric"].tolist(),
        table["Value"].tolist(),
        f"{model_name} Metrics",
        color=color
    )
    fig.update_yaxes(range=[0, 1.05])
    return table, fig


def compare_classifiers(X, y, test_size=0.25, tree_depth=4, random_state=42):
    """
    Trains Decision Tree and Naive Bayes on identical splits and computes comparison metrics.
    """
    dt_results, dt_err = train_j48_decision_tree(
        X, y, max_depth=tree_depth, test_size=test_size, random_state=random_state
    )
    nb_results, nb_err = train_naive_bayes(X, y, test_size=test_size, random_state=random_state)
    if dt_err:
        return None, dt_err
    if nb_err:
        return None, nb_err

    comparison_df = pd.DataFrame([
        {
            "Model": "Decision Tree (sklearn)",
            "Accuracy": dt_results["accuracy"],
            "Precision": dt_results["precision"],
            "Recall": dt_results["recall"],
            "F1_Score": dt_results["f1_score"],
        },
        {
            "Model": "Naive Bayes (Gaussian)",
            "Accuracy": nb_results["accuracy"],
            "Precision": nb_results["precision"],
            "Recall": nb_results["recall"],
            "F1_Score": nb_results["f1_score"],
        },
    ])
    return comparison_df, None


def plot_model_comparison_bar(comparison_df):
    """
    Creates a grouped bar chart comparing Decision Tree and Naive Bayes metrics.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
    fig = go.Figure()
    colors = ["#06B6D4", "#8B5CF6"]
    for idx, (_, row) in enumerate(comparison_df.iterrows()):
        fig.add_trace(go.Bar(
            x=metrics,
            y=[row[m] for m in metrics],
            name=row["Model"],
            marker_color=colors[idx % len(colors)],
            text=[f"{row[m]:.4f}" for m in metrics],
            textposition="auto"
        ))
    fig.update_layout(
        title="Classifier Comparison (Decision Tree vs Naive Bayes)",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.7)",
        font=dict(color="#F3F4F6", family="sans-serif"),
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_cluster_distribution(distribution_df):
    fig = px.bar(
        distribution_df,
        x="Cluster",
        y="Count",
        color="Cluster",
        title="K-Means Cluster Distribution",
        text="Count"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.7)",
        font=dict(color="#F3F4F6"),
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)")
    )
    return fig


def _best_classification(dt_results, nb_results):
    dt_acc, nb_acc = dt_results["accuracy"], nb_results["accuracy"]
    dt_f1, nb_f1 = dt_results["f1_score"], nb_results["f1_score"]
    dt_score = dt_acc + dt_f1
    nb_score = nb_acc + nb_f1
    if dt_score > nb_score:
        return (
            "Decision Tree",
            f"It achieved higher accuracy/F1 on the selected dataset "
            f"(Accuracy {dt_acc:.4f} vs {nb_acc:.4f}, F1 {dt_f1:.4f} vs {nb_f1:.4f})."
        )
    if nb_score > dt_score:
        return (
            "Naive Bayes",
            f"It achieved higher accuracy/F1 on the selected dataset "
            f"(Accuracy {nb_acc:.4f} vs {dt_acc:.4f}, F1 {nb_f1:.4f} vs {dt_f1:.4f})."
        )
    return (
        "Decision Tree and Naive Bayes (tie)",
        f"Both models obtained the same combined accuracy/F1 "
        f"(Accuracy {dt_acc:.4f}, F1 {dt_f1:.4f})."
    )


def compute_full_comparison(
    clean_df,
    min_support=0.05,
    min_confidence=0.30,
    max_len=3,
    tree_depth=4,
    test_size=0.25,
    k=3,
    cluster_features=None,
):
    """
    Runs all major techniques on the cleaned dataset and returns serializable result tables.
    """
    if cluster_features is None:
        cluster_features = ["Purchase_Amount", "Purchase_Frequency", "Customer_Rating"]

    bundle = {"errors": {}}

    frequent_itemsets, rules, assoc_err = run_apriori(
        clean_df, min_support=min_support, min_confidence=min_confidence, max_len=max_len
    )
    bundle["errors"]["association"] = assoc_err
    if assoc_err or rules is None or rules.empty:
        bundle["association_table"] = pd.DataFrame()
        bundle["association_fig"] = None
        bundle["strongest_rule"] = None
        bundle["rules"] = None
        bundle["frequent_itemsets"] = frequent_itemsets
    else:
        summary_df, fig, strongest = summarize_association(frequent_itemsets, rules, len(clean_df))
        bundle["association_table"] = summary_df
        bundle["association_fig"] = fig
        bundle["strongest_rule"] = strongest
        bundle["rules"] = rules
        bundle["frequent_itemsets"] = frequent_itemsets

    X, y, _, _, class_err = prepare_classification_features(clean_df)
    bundle["errors"]["classification_prep"] = class_err
    if class_err:
        bundle["dt_table"] = pd.DataFrame()
        bundle["nb_table"] = pd.DataFrame()
        bundle["classifier_comp_df"] = None
        bundle["dt_results"] = None
        bundle["nb_results"] = None
    else:
        dt_results, dt_err = train_j48_decision_tree(X, y, max_depth=tree_depth, test_size=test_size)
        nb_results, nb_err = train_naive_bayes(X, y, test_size=test_size)
        bundle["errors"]["decision_tree"] = dt_err
        bundle["errors"]["naive_bayes"] = nb_err
        bundle["dt_results"] = dt_results
        bundle["nb_results"] = nb_results
        if dt_results:
            bundle["dt_table"], bundle["dt_fig"] = classifier_table_and_chart(
                dt_results, "Decision Tree", "#06B6D4"
            )
        else:
            bundle["dt_table"], bundle["dt_fig"] = pd.DataFrame(), None
        if nb_results:
            bundle["nb_table"], bundle["nb_fig"] = classifier_table_and_chart(
                nb_results, "Naive Bayes", "#8B5CF6"
            )
        else:
            bundle["nb_table"], bundle["nb_fig"] = pd.DataFrame(), None
        if dt_results and nb_results:
            comp_df = pd.DataFrame([
                {
                    "Model": "Decision Tree (sklearn)",
                    "Accuracy": dt_results["accuracy"],
                    "Precision": dt_results["precision"],
                    "Recall": dt_results["recall"],
                    "F1_Score": dt_results["f1_score"],
                },
                {
                    "Model": "Naive Bayes (Gaussian)",
                    "Accuracy": nb_results["accuracy"],
                    "Precision": nb_results["precision"],
                    "Recall": nb_results["recall"],
                    "F1_Score": nb_results["f1_score"],
                },
            ])
            bundle["classifier_comp_df"] = comp_df
            bundle["classifier_comp_fig"] = plot_model_comparison_bar(comp_df)
            bundle["errors"]["classifier_compare"] = None
        else:
            bundle["classifier_comp_df"] = None
            bundle["classifier_comp_fig"] = None
            bundle["errors"]["classifier_compare"] = dt_err or nb_err

    Xr, yr, reg_prep_err = prepare_regression_features(clean_df)
    bundle["errors"]["regression_prep"] = reg_prep_err
    if reg_prep_err:
        bundle["regression"] = None
    else:
        reg_pack, reg_err = train_both_regression_models(Xr, yr, test_size=test_size)
        bundle["errors"]["regression"] = reg_err
        bundle["regression"] = reg_pack

    cluster_results, cluster_err = perform_kmeans_clustering(
        clean_df, feature_cols=cluster_features, k=k
    )
    bundle["errors"]["clustering"] = cluster_err
    bundle["clustering"] = cluster_results
    if cluster_results:
        k_table = pd.DataFrame([
            {"Metric": "Number of clusters (K)", "Value": cluster_results["k"]},
            {"Metric": "Silhouette Score", "Value": cluster_results["silhouette_score"]},
            {"Metric": "Inertia", "Value": cluster_results["inertia"]},
        ])
        for _, row in cluster_results["distribution_df"].iterrows():
            k_table = pd.concat([
                k_table,
                pd.DataFrame([{"Metric": f"{row['Cluster']} count", "Value": int(row["Count"])}])
            ], ignore_index=True)
        bundle["kmeans_table"] = k_table
        bundle["kmeans_fig"] = plot_cluster_distribution(cluster_results["distribution_df"])
        bundle["kmeans_centers"] = cluster_results["centers_df"]
    else:
        bundle["kmeans_table"] = pd.DataFrame()
        bundle["kmeans_fig"] = None
        bundle["kmeans_centers"] = None

    best = {}
    if bundle.get("dt_results") and bundle.get("nb_results"):
        name, reason = _best_classification(bundle["dt_results"], bundle["nb_results"])
        best["classification"] = {"model": name, "reason": reason}
    if bundle.get("regression"):
        best["regression"] = {
            "model": bundle["regression"]["winner"],
            "reason": bundle["regression"]["winner_reason"],
        }
    if bundle.get("clustering"):
        sil = bundle["clustering"]["silhouette_score"]
        if sil >= 0.5:
            quality = "reasonably well-separated clusters"
        elif sil >= 0.25:
            quality = "moderate cluster structure"
        else:
            quality = "weak cluster separation on the selected features"
        best["clustering"] = {
            "model": "K-Means",
            "reason": f"Silhouette Score: {sil:.4f} ({quality}). K = {bundle['clustering']['k']}.",
            "silhouette": sil,
        }
    if bundle.get("strongest_rule") is not None:
        rule = bundle["strongest_rule"]
        best["association"] = {
            "rule": f"{rule['antecedents_str']}  →  {rule['consequents_str']}",
            "support": float(rule["support"]),
            "confidence": float(rule["confidence"]),
            "lift": float(rule["lift"]),
        }
    bundle["best"] = best
    return bundle
