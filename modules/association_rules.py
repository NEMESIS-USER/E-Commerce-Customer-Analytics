"""
Association Rule Mining Module
Implements Apriori Algorithm using MLxtend on e-commerce transaction profiles.
Calculates Support, Confidence, and Lift with interactive visualizations.
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px


def _labeled_series(prefix, series):
    return np.where(series.notna(), prefix + series.astype(str), None)


def prepare_transaction_baskets(df):
    """
    Transforms tabular customer records into boolean transaction representation
    with descriptive item labels suitable for market basket / association mining.
    """
    n = len(df)
    cat = _labeled_series("Category_", df["Product_Category"]) if "Product_Category" in df.columns else np.array([None] * n)
    pay = _labeled_series("Payment_", df["Payment_Method"]) if "Payment_Method" in df.columns else np.array([None] * n)
    seg = _labeled_series("Segment_", df["Customer_Segment"]) if "Customer_Segment" in df.columns else np.array([None] * n)

    if "Discount_Used" in df.columns:
        disc_raw = df["Discount_Used"].astype(str).str.strip().str.lower()
        disc = np.where(
            df["Discount_Used"].notna(),
            np.where(disc_raw.isin(["yes", "1", "true"]), "Discount_Yes", "Discount_No"),
            None
        )
    else:
        disc = np.array([None] * n)

    if "Purchase_Frequency" in df.columns:
        freq = pd.to_numeric(df["Purchase_Frequency"], errors="coerce")
        freq_bin = np.where(freq.isna(), None, np.where(freq >= 15, "Freq_High", np.where(freq >= 5, "Freq_Medium", "Freq_Low")))
    else:
        freq_bin = np.array([None] * n)

    if "Purchase_Amount" in df.columns:
        amt = pd.to_numeric(df["Purchase_Amount"], errors="coerce")
        spend_bin = np.where(amt.isna(), None, np.where(amt >= 10000, "Spend_High", np.where(amt >= 3000, "Spend_Medium", "Spend_Low")))
    else:
        spend_bin = np.array([None] * n)

    if "Customer_Rating" in df.columns:
        rating = pd.to_numeric(df["Customer_Rating"], errors="coerce")
        rating_bin = np.where(rating.isna(), None, np.where(rating >= 4.0, "Rating_High", np.where(rating >= 3.0, "Rating_Medium", "Rating_Low")))
    else:
        rating_bin = np.array([None] * n)

    transactions = []
    for vals in zip(cat, pay, disc, seg, freq_bin, spend_bin, rating_bin):
        transactions.append([v for v in vals if v is not None])

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)


def run_apriori(df, min_support=0.05, min_confidence=0.3, metric="confidence", max_len=3):
    """
    Executes Apriori frequent itemset generation and association rules extraction.
    Returns:
        - frequent_itemsets: DataFrame of frequent itemsets with support
        - rules: DataFrame of association rules with support, confidence, lift
        - error_msg: String if an error or empty result occurs
    """
    try:
        df_basket = prepare_transaction_baskets(df)

        if df_basket.empty:
            return None, None, "Transaction basket is empty. Please verify dataset."
        n_transactions = len(df_basket)

        # 1. Frequent Itemsets
        frequent_itemsets = apriori(
            df_basket,
            min_support=min_support,
            use_colnames=True,
            max_len=max_len
        )

        if frequent_itemsets.empty:
            return None, None, f"No frequent itemsets found with minimum support {min_support}. Try lowering the threshold."
        frequent_itemsets.attrs["n_transactions"] = n_transactions

        # 2. Association Rules
        rules = association_rules(
            frequent_itemsets,
            metric=metric,
            min_threshold=min_confidence
        )

        if rules.empty:
            return frequent_itemsets, None, f"Frequent itemsets found, but no association rules meet minimum confidence {min_confidence}."

        # Format antecedent and consequent sets into clean readable strings
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        rules["rule_name"] = rules["antecedents_str"] + "  ➔  " + rules["consequents_str"]

        # Round numerical metric columns for clean presentation
        rules["support"] = rules["support"].round(4)
        rules["confidence"] = rules["confidence"].round(4)
        rules["lift"] = rules["lift"].round(4)
        if "leverage" in rules.columns:
            rules["leverage"] = rules["leverage"].round(4)
        if "conviction" in rules.columns:
            rules["conviction"] = rules["conviction"].round(4)

        # Sort by lift descending
        rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)

        return frequent_itemsets, rules, None

    except Exception as e:
        return None, None, str(e)


def plot_rules_scatter(rules):
    """
    Creates an interactive Plotly scatter plot for Association Rules (Support vs Confidence vs Lift).
    """
    if rules is None or rules.empty:
        return None

    plot_rules = rules.head(80) if len(rules) > 80 else rules

    fig = px.scatter(
        plot_rules,
        x="support",
        y="confidence",
        size="lift",
        color="lift",
        hover_name="rule_name",
        hover_data={
            "support": ":.4f",
            "confidence": ":.4f",
            "lift": ":.4f",
            "antecedents_str": False,
            "consequents_str": False,
            "rule_name": False
        },
        color_continuous_scale="Viridis",
        title="Association Rules: Support vs Confidence (Sized & Colored by Lift)",
        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"}
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
