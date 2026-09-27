"""
E-Commerce Customer Analytics & Data Mining System
Streamlit Multi-Page Application
A comprehensive Data Warehousing & Data Mining Mini-Project Dashboard.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure current directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_creation import generate_dataset
from modules.preprocessing import load_dataset, get_dataset_summary, preprocess_data, validate_dataset
from modules.association_rules import run_apriori, plot_rules_scatter
from modules.j48_classification import (
    prepare_classification_features,
    train_j48_decision_tree,
    plot_confusion_matrix_heatmap,
    plot_feature_importance,
    render_decision_tree_plot
)
from modules.naive_bayes import train_naive_bayes, plot_class_priors
from modules.regression import (
    prepare_regression_features,
    train_both_regression_models,
    plot_actual_vs_predicted,
    plot_residual_distribution,
    plot_regression_metrics_comparison
)
from modules.clustering import (
    perform_kmeans_clustering,
    compute_elbow_curve,
    plot_elbow_curve,
    plot_clusters_2d,
    plot_clusters_3d
)
from modules.comparison import compute_full_comparison, plot_model_comparison_bar

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="E-Commerce Customer Analytics & Data Mining System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS FOR DARK SAAS DASHBOARD THEME
# ==========================================
CUSTOM_CSS = """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
    }

    /* Top Header Bar */
    .app-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .app-title {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .app-subtitle {
        color: #94A3B8;
        font-size: 14px;
        font-weight: 500;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 18px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }

    .metric-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #F8FAFC;
    }

    .metric-sub {
        font-size: 12px;
        color: #10B981;
        margin-top: 4px;
    }

    /* Section Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-cyan { background: rgba(6, 182, 212, 0.15); color: #22D3EE; border: 1px solid rgba(6, 182, 212, 0.3); }
    .badge-indigo { background: rgba(99, 102, 241, 0.15); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.3); }
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }

    /* Pipeline Step Box */
    .pipeline-step {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 12px 14px;
        text-align: center;
        font-size: 12px;
        font-weight: 600;
        color: #E2E8F0;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Tables */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# SESSION STATE & DATA MANAGEMENT
# ==========================================
DATA_FILE_PATH = "data/ecommerce_customer_data.csv"
SEGMENT_COLORS = {"High": "#8B5CF6", "Medium": "#3B82F6", "Low": "#06B6D4"}


@st.cache_data(show_spinner=False)
def cached_load_csv(path, mtime):
    return load_dataset(path)


@st.cache_data(show_spinner=False)
def cached_preprocess(df, impute_strategy_num, impute_strategy_cat, drop_duplicates):
    return preprocess_data(
        df,
        impute_strategy_num=impute_strategy_num,
        impute_strategy_cat=impute_strategy_cat,
        drop_duplicates=drop_duplicates,
    )


@st.cache_data(show_spinner=False)
def cached_apriori(df, min_support, min_confidence, max_len):
    return run_apriori(df, min_support=min_support, min_confidence=min_confidence, max_len=max_len)


@st.cache_data(show_spinner=False)
def cached_class_features(df):
    return prepare_classification_features(df)


@st.cache_data(show_spinner=False)
def cached_j48(X, y, criterion, max_depth, min_samples_split, test_size):
    return train_j48_decision_tree(
        X, y,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        test_size=test_size,
    )


@st.cache_data(show_spinner=False)
def cached_nb(X, y, test_size):
    return train_naive_bayes(X, y, test_size=test_size)


@st.cache_data(show_spinner=False)
def cached_reg_features(df):
    return prepare_regression_features(df)


@st.cache_data(show_spinner=False)
def cached_both_regression(X, y, test_size):
    return train_both_regression_models(X, y, test_size=test_size)


@st.cache_data(show_spinner=False)
def cached_kmeans(df, feature_cols, k):
    return perform_kmeans_clustering(df, feature_cols=list(feature_cols), k=k)


@st.cache_data(show_spinner=False)
def cached_elbow(df, feature_cols, max_k=7):
    return compute_elbow_curve(df, feature_cols=list(feature_cols), max_k=max_k)


@st.cache_data(show_spinner=False)
def cached_full_comparison(df, min_support, min_confidence, max_len, tree_depth, test_size, k, cluster_features):
    return compute_full_comparison(
        df,
        min_support=min_support,
        min_confidence=min_confidence,
        max_len=max_len,
        tree_depth=tree_depth,
        test_size=test_size,
        k=k,
        cluster_features=list(cluster_features),
    )


def apply_preprocessed(df, num_strat="Median", cat_strat="Mode", drop_dups=True):
    clean_df, enc_df, encoders, scaler, stats = cached_preprocess(df, num_strat, cat_strat, drop_dups)
    st.session_state.cleaned_df = clean_df
    st.session_state.encoded_df = enc_df
    st.session_state.encoders = encoders
    st.session_state.scaler = scaler
    st.session_state.prep_stats = stats


def init_session_state():
    """Initializes and ensures default dataset is available."""
    if "raw_df" not in st.session_state:
        os.makedirs("data", exist_ok=True)
        need_generate = True
        if os.path.exists(DATA_FILE_PATH):
            mtime = os.path.getmtime(DATA_FILE_PATH)
            df, _ = cached_load_csv(DATA_FILE_PATH, mtime)
            if df is not None and len(df) >= 4000:
                need_generate = False
        else:
            df = None
        if need_generate:
            generate_dataset(output_path=DATA_FILE_PATH)
            mtime = os.path.getmtime(DATA_FILE_PATH)
            df, _ = cached_load_csv(DATA_FILE_PATH, mtime)
        st.session_state.raw_df = df
        st.session_state.dataset_source = "Default (data/ecommerce_customer_data.csv)"

    if "cleaned_df" not in st.session_state and st.session_state.raw_df is not None:
        apply_preprocessed(st.session_state.raw_df)

init_session_state()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0 20px 0; text-align: left;">
            <h2 style="color: #38BDF8; margin: 0; font-size: 17px; font-weight: 800; line-height: 1.3;">🛍️ E-Commerce Customer Analytics & Data Mining System</h2>
            <p style="color: #94A3B8; font-size: 12px; margin-top: 5px;">Data Warehousing & Mining Practical</p>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation Menu",
        [
            "Dashboard",
            "Dataset",
            "Preprocessing",
            "Association Rules",
            "J48 / Decision Tree",
            "Naive Bayes",
            "Regression",
            "K-Means Clustering",
            "Comparison & Best Model"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("""
        <div style="padding: 10px; background: rgba(30, 41, 59, 0.6); border-radius: 10px; font-size: 11px; color: #94A3B8;">
            <strong style="color: #F8FAFC;">Active Dataset:</strong><br>
            <span style="color: #38BDF8;">""" + st.session_state.get("dataset_source", "None") + """</span><br>
            <strong>Total Rows:</strong> """ + str(len(st.session_state.raw_df) if st.session_state.raw_df is not None else 0) + """
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Reset to Default Dataset", use_container_width=True):
        st.cache_data.clear()
        generate_dataset(output_path=DATA_FILE_PATH)
        mtime = os.path.getmtime(DATA_FILE_PATH)
        df, _ = cached_load_csv(DATA_FILE_PATH, mtime)
        st.session_state.raw_df = df
        st.session_state.dataset_source = "Default (data/ecommerce_customer_data.csv)"
        apply_preprocessed(df)
        st.success("Default dataset regenerated!")
        st.rerun()


# ==========================================
# PAGE 1: DASHBOARD
# ==========================================
if page == "Dashboard":
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">E-Commerce Customer Analytics</h1>
            <p class="app-subtitle">Data Mining & Predictive Analytics System</p>
        </div>
    """, unsafe_allow_html=True)

    raw_df = st.session_state.raw_df
    clean_df = st.session_state.cleaned_df

    if raw_df is not None and clean_df is not None:
        # Top KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Records</div>
                    <div class="metric-value">{len(raw_df):,}</div>
                    <div class="metric-sub">Cleaned: {len(clean_df):,} rows</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Features</div>
                    <div class="metric-value">{len(raw_df.columns)}</div>
                    <div class="metric-sub">Numerical & Categorical</div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            segments_count = clean_df["Customer_Segment"].nunique() if "Customer_Segment" in clean_df.columns else 0
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Customer Segments</div>
                    <div class="metric-value">{segments_count}</div>
                    <div class="metric-sub">Low, Medium, High</div>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            avg_amt = clean_df["Purchase_Amount"].mean() if "Purchase_Amount" in clean_df.columns else 0
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Average Purchase</div>
                    <div class="metric-value">₹{avg_amt:,.2f}</div>
                    <div class="metric-sub">Customer basket mean</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualizations Row
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown('<div class="glass-card"><div class="section-title">👥 Customer Segment Distribution</div>', unsafe_allow_html=True)
            if "Customer_Segment" in clean_df.columns:
                seg_counts = clean_df["Customer_Segment"].value_counts().reset_index()
                seg_counts.columns = ["Segment", "Count"]
                fig_seg = px.pie(
                    seg_counts,
                    names="Segment",
                    values="Count",
                    hole=0.45,
                    color="Segment",
                    color_discrete_map=SEGMENT_COLORS
                )
                fig_seg.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F3F4F6"),
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_seg, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card"><div class="section-title">📦 Product Category Distribution</div>', unsafe_allow_html=True)
            if "Product_Category" in clean_df.columns:
                cat_counts = clean_df["Product_Category"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                fig_cat = px.bar(
                    cat_counts,
                    x="Category",
                    y="Count",
                    color="Category",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_cat.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(17,24,39,0.7)",
                    font=dict(color="#F3F4F6"),
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=False
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Purchase Amount Histogram
        st.markdown('<div class="glass-card"><div class="section-title">💳 Purchase Amount Distribution by Segment</div>', unsafe_allow_html=True)
        if "Purchase_Amount" in clean_df.columns and "Customer_Segment" in clean_df.columns:
            fig_amt = px.histogram(
                clean_df,
                x="Purchase_Amount",
                color="Customer_Segment",
                barmode="overlay",
                nbins=35,
                color_discrete_map=SEGMENT_COLORS
            )
            fig_amt.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.7)",
                font=dict(color="#F3F4F6"),
                xaxis=dict(title="Purchase Amount (₹)", gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(title="Customer Count", gridcolor="rgba(255,255,255,0.08)"),
                margin=dict(l=40, r=20, t=30, b=40)
            )
            st.plotly_chart(fig_amt, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Project Pipeline Architecture
        st.markdown('<div class="glass-card"><div class="section-title">🔄 Integrated Data Mining Pipeline</div>', unsafe_allow_html=True)
        p_cols = st.columns(7)
        steps = [
            ("1. Dataset", "5,000+ Raw Records", "#38BDF8"),
            ("2. Preprocessing", "Clean & Impute", "#818CF8"),
            ("3. Association", "Apriori Basket", "#C084FC"),
            ("4. Classification", "Decision Tree", "#34D399"),
            ("5. Naive Bayes", "Probabilistic ML", "#F472B6"),
            ("6. Regression", "Predict Amount", "#FBBF24"),
            ("7. Clustering", "K-Means Personas", "#22D3EE")
        ]
        for idx, col in enumerate(p_cols):
            with col:
                st.markdown(f"""
                    <div class="pipeline-step" style="border-top: 3px solid {steps[idx][2]};">
                        <div style="font-weight: 700; color: #F8FAFC;">{steps[idx][0]}</div>
                        <div style="color: #94A3B8; font-size: 10px; margin-top: 4px;">{steps[idx][1]}</div>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 2: DATASET EXPLORER
# ==========================================
elif page == "Dataset":
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">Dataset Management & Exploration</h1>
            <p class="app-subtitle">Upload custom CSV datasets or explore the built-in synthetic e-commerce warehouse data.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="glass-card"><div class="section-title">📂 Upload Custom Dataset</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV file (e.g. customer_data.csv)", type=["csv"])
        if uploaded_file is not None:
            df_upload, err = load_dataset(uploaded_file)
            if err:
                st.error(f"Error loading file: {err}")
            elif df_upload is None or df_upload.empty:
                st.warning("Uploaded file is empty.")
            else:
                report = validate_dataset(df_upload)
                st.markdown("**Upload validation**")
                st.write(f"- Rows: {report['num_rows']:,} | Columns: {report['num_columns']}")
                st.write(f"- Numeric columns OK: {', '.join(report['numeric_ok']) or 'None'}")
                st.write(f"- Categorical columns present: {', '.join(report['categorical_present']) or 'None'}")
                st.write(f"- Missing value cells: {report['missing_cells']} ({report['missing_percentage']}%)")
                st.write(f"- Duplicate rows: {report['duplicate_rows']}")
                if report["extra_columns"]:
                    st.info("Extra columns kept: " + ", ".join(report["extra_columns"]))
                if report["numeric_failed"]:
                    st.warning("Numeric parse issues: " + ", ".join(report["numeric_failed"]))
                if not report["is_valid"]:
                    for msg in report["errors"]:
                        st.error(msg)
                    st.warning("Dataset was not activated because required columns are missing. The default/current dataset is unchanged.")
                else:
                    st.session_state.raw_df = report["coerced_df"]
                    st.session_state.dataset_source = f"Uploaded: {uploaded_file.name}"
                    apply_preprocessed(report["coerced_df"])
                    st.success(f"Successfully loaded '{uploaded_file.name}' ({len(report['coerced_df'])} records)")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card"><div class="section-title">ℹ️ Dataset Info & Active Status</div>', unsafe_allow_html=True)
        st.markdown(f"**Current Source:** <span class='badge badge-cyan'>{st.session_state.get('dataset_source')}</span>", unsafe_allow_html=True)
        summary = get_dataset_summary(st.session_state.raw_df)
        if summary:
            st.write(f"- **Total Rows:** {summary['num_rows']:,}")
            st.write(f"- **Total Columns:** {summary['num_columns']}")
            st.write(f"- **Numeric columns:** {', '.join(summary['numeric_cols'])}")
            st.write(f"- **Categorical columns:** {', '.join(summary['categorical_cols'])}")
            st.write(f"- **Missing Value Cells:** {summary['missing_cells']} ({summary['missing_percentage']}%)")
            st.write(f"- **Duplicate Rows:** {summary['duplicate_rows']}")
        default_report = validate_dataset(st.session_state.raw_df)
        if default_report["missing_required"]:
            st.warning("Required columns missing: " + ", ".join(default_report["missing_required"]))
        else:
            st.markdown("<span class='badge badge-green'>Required schema present</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    raw_df = st.session_state.raw_df
    if raw_df is not None:
        st.markdown('<div class="glass-card"><div class="section-title">🔍 Dataset Preview</div>', unsafe_allow_html=True)
        num_rows_preview = st.slider("Select rows to preview", min_value=5, max_value=50, value=10, step=5)
        st.dataframe(raw_df.head(num_rows_preview), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><div class="section-title">📊 Column Types & Missingness Matrix</div>', unsafe_allow_html=True)
        col_info = pd.DataFrame({
            "Column Name": list(raw_df.columns),
            "Data Type": [str(t) for t in raw_df.dtypes],
            "Missing Count": raw_df.isnull().sum().values,
            "Missing %": ((raw_df.isnull().sum().values / len(raw_df)) * 100).round(2),
            "Unique Values": [raw_df[c].nunique() for c in raw_df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 3: PREPROCESSING
# ==========================================
elif page == "Preprocessing":
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">Data Preprocessing & Cleaning</h1>
            <p class="app-subtitle">Handling missing values, removing duplicate records, categorical encoding, and feature scaling.</p>
        </div>
    """, unsafe_allow_html=True)

    raw_df = st.session_state.raw_df

    st.markdown('<div class="glass-card"><div class="section-title">⚙️ Preprocessing Configuration</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        num_strat = st.selectbox("Numerical Imputation Strategy", ["Median", "Mean", "Zero"])
    with c2:
        cat_strat = st.selectbox("Categorical Imputation Strategy", ["Mode", "Constant ('Missing')"])
    with c3:
        drop_dups = st.checkbox("Remove Duplicate Records", value=True)

    if st.button("🚀 Apply Preprocessing Pipeline", type="primary"):
        apply_preprocessed(
            raw_df,
            num_strat=num_strat,
            cat_strat="Mode" if "Mode" in cat_strat else "Missing",
            drop_dups=drop_dups
        )
        st.success("Preprocessing applied successfully!")

    st.markdown('</div>', unsafe_allow_html=True)

    stats = st.session_state.get("prep_stats", {})
    if stats and "before" in stats and "after" in stats:
        b = stats["before"]
        a = stats["after"]

        st.markdown('<div class="glass-card"><div class="section-title">📊 Before vs After Preprocessing Statistics</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{a['num_rows']:,}", delta=f"{a['num_rows'] - b['num_rows']} rows")
        with col2:
            st.metric("Missing Cells", f"{a['missing_cells']}", delta=f"{a['missing_cells'] - b['missing_cells']} missing", delta_color="inverse")
        with col3:
            st.metric("Duplicate Rows", f"{a['duplicate_rows']}", delta=f"{a['duplicate_rows'] - b['duplicate_rows']} duplicates", delta_color="inverse")
        with col4:
            st.metric("Columns", f"{a['num_columns']}", delta="0 changes")
        st.markdown('</div>', unsafe_allow_html=True)

    clean_df = st.session_state.cleaned_df
    if clean_df is not None:
        st.markdown('<div class="glass-card"><div class="section-title">✨ Preprocessed Cleaned Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(clean_df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 4: ASSOCIATION RULE MINING
# ==========================================
elif page == "Association Rules":
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">Association Rule Mining (Apriori)</h1>
            <p class="app-subtitle">Discover frequent itemsets and market basket affinities using the Apriori algorithm from MLxtend.</p>
        </div>
    """, unsafe_allow_html=True)

    clean_df = st.session_state.cleaned_df

    st.markdown('<div class="glass-card"><div class="section-title">⚙️ Apriori Hyperparameters</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        min_supp = st.slider("Minimum Support", min_value=0.01, max_value=0.30, value=0.05, step=0.01, help="Fraction of transactions containing the itemset.")
    with c2:
        min_conf = st.slider("Minimum Confidence", min_value=0.10, max_value=0.90, value=0.30, step=0.05, help="Conditional probability of consequent given antecedent.")
    with c3:
        max_len = st.slider("Max Itemset Length", min_value=2, max_value=4, value=3)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Mining frequent itemsets and rules..."):
        frequent_itemsets, rules, err = cached_apriori(clean_df, min_supp, min_conf, max_len)

    if err:
        st.warning(f"⚠️ {err}")
    elif rules is not None and not rules.empty:
        # Top Rule KPI Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Transactions", f"{len(clean_df):,}")
        with c2:
            st.metric("Frequent Itemsets", f"{len(frequent_itemsets):,}" if frequent_itemsets is not None else "0")
        with c3:
            st.metric("Total Rules Generated", f"{len(rules):,}")
        with c4:
            st.metric("Max Lift", f"{rules['lift'].max():.4f}")

        # Interactive Rules Scatter Plot
        st.markdown('<div class="glass-card"><div class="section-title">📈 Rules Map: Support vs Confidence (Sized & Colored by Lift)</div>', unsafe_allow_html=True)
        fig_scatter = plot_rules_scatter(rules)
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Clean Table Display
        st.markdown('<div class="glass-card"><div class="section-title">📋 Strongest Association Rules Table</div>', unsafe_allow_html=True)
        display_rules = rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].rename(
            columns={
                "antecedents_str": "Antecedent (IF)",
                "consequents_str": "Consequent (THEN)",
                "support": "Support",
                "confidence": "Confidence",
                "lift": "Lift"
            }
        )
        st.dataframe(display_rules.head(25), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No rules found for the selected thresholds. Try lowering minimum support or confidence.")


# ==========================================
# PAGE 5: J48 / DECISION TREE CLASSIFICATION
# ==========================================
elif page == "J48 / Decision Tree":
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">J48 / Decision Tree Classification</h1>
            <p class="app-subtitle">scikit-learn Decision Tree (C4.5 / J48-style equivalent) predicting Customer_Segment. This is not Weka J48.</p>
        </div>
    """, unsafe_allow_html=True)

    clean_df = st.session_state.cleaned_df

    st.markdown('<div class="glass-card"><div class="section-title">⚙️ Decision Tree Parameters (scikit-learn, J48-style equivalent)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        criterion = st.selectbox("Split Criterion", ["gini", "entropy"], index=0, help="Gini Impurity vs Information Gain / Entropy")
    with c2:
        max_depth = st.slider("Max Tree Depth", min_value=1, max_value=8, value=4)
    with c3:
        min_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=5)
    with c4:
        test_size = st.slider("Test Set Ratio", min_value=0.15, max_value=0.40, value=0.25, step=0.05)
    st.markdown('</div>', unsafe_allow_html=True)

    X, y, feat_names, class_names, err = cached_class_features(clean_df)

    if err:
        st.error(err)
    else:
        results, train_err = cached_j48(X, y, criterion, max_depth, min_split, test_size)

        if train_err:
            st.error(train_err)
        else:
            # Metrics Row
            st.markdown('<div class="glass-card"><div class="section-title">📊 Classification Performance</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")
            with m2:
                st.metric("Precision (Weighted)", f"{results['precision']:.4f}")
            with m3:
                st.metric("Recall (Weighted)", f"{results['recall']:.4f}")
            with m4:
                st.metric("F1-Score", f"{results['f1_score']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Confusion Matrix & Feature Importance
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="glass-card"><div class="section-title">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
                fig_cm = plot_confusion_matrix_heatmap(results["confusion_matrix"], results["classes"], title="Decision Tree Confusion Matrix")
                st.plotly_chart(fig_cm, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="glass-card"><div class="section-title">⭐ Feature Importance (Information Gain)</div>', unsafe_allow_html=True)
                fig_imp = plot_feature_importance(results["feature_importance_df"], top_n=8)
                st.plotly_chart(fig_imp, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Classification Report Table
            st.markdown('<div class="glass-card"><div class="section-title">📄 Detailed Classification Report</div>', unsafe_allow_html=True)
            st.dataframe(results["report_df"], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Tree Visualization
            st.markdown('<div class="glass-card"><div class="section-title">🌳 Decision Tree Hierarchy Visualization</div>', unsafe_allow_html=True)
            fig_tree = render_decision_tree_plot(results["model"], feat_names, class_names, max_depth=min(max_depth, 3))
            st.pyplot(fig_tree)
            st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 6: NAIVE BAYES CLASSIFICATION
# ==========================================
elif page == "Naive Bayes":
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">Naive Bayes Classification</h1>
            <p class="app-subtitle">Probabilistic classification of Customer_Segment using Bayes Theorem.</p>
        </div>
    """, unsafe_allow_html=True)

    clean_df = st.session_state.cleaned_df
    X, y, feat_names, class_names, err = cached_class_features(clean_df)

    if err:
        st.error(err)
    else:
        st.markdown('<div class="glass-card"><div class="section-title">⚙️ Naive Bayes Settings</div>', unsafe_allow_html=True)
        test_size = st.slider("Test Split Size", min_value=0.15, max_value=0.40, value=0.25, step=0.05, key="nb_split")
        st.markdown('</div>', unsafe_allow_html=True)

        results, train_err = cached_nb(X, y, test_size)

        if train_err:
            st.error(train_err)
        else:
            # Metrics
            st.markdown('<div class="glass-card"><div class="section-title">📊 Classification Performance</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")
            with m2:
                st.metric("Precision (Weighted)", f"{results['precision']:.4f}")
            with m3:
                st.metric("Recall (Weighted)", f"{results['recall']:.4f}")
            with m4:
                st.metric("F1-Score", f"{results['f1_score']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="glass-card"><div class="section-title">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
                fig_cm = plot_confusion_matrix_heatmap(results["confusion_matrix"], results["classes"], title="Naive Bayes Confusion Matrix")
                st.plotly_chart(fig_cm, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="glass-card"><div class="section-title">📊 Prior Probabilities P(Y)</div>', unsafe_allow_html=True)
                fig_priors = plot_class_priors(results["priors_df"])
                st.plotly_chart(fig_priors, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card"><div class="section-title">📄 Classification Report Table</div>', unsafe_allow_html=True)
            st.dataframe(results["report_df"], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 7: REGRESSION
# ==========================================
elif page == "Regression":
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">Regression Analysis</h1>
            <p class="app-subtitle">Predicting continuous Purchase_Amount (₹) from customer behavioral and demographic signals.</p>
        </div>
    """, unsafe_allow_html=True)

    clean_df = st.session_state.cleaned_df
    X, y, err = cached_reg_features(clean_df)

    if err:
        st.error(err)
    else:
        st.markdown('<div class="glass-card"><div class="section-title">⚙️ Regression Model Configuration</div>', unsafe_allow_html=True)
        test_size = st.slider("Test Split Size", min_value=0.15, max_value=0.40, value=0.25, step=0.05, key="reg_split")
        st.caption("Both Linear Regression and Random Forest Regression are trained on the same train/test split.")
        st.markdown('</div>', unsafe_allow_html=True)

        pack, train_err = cached_both_regression(X, y, test_size)

        if train_err:
            st.error(train_err)
        else:
            st.info(
                "**How to read the metrics:** Lower MAE = better. Lower MSE = better. Higher R² = better. "
                "The highlighted winner is chosen from these three metrics on the selected dataset (not hardcoded)."
            )
            st.markdown('<div class="glass-card"><div class="section-title">📊 Regression Evaluation Metrics</div>', unsafe_allow_html=True)
            st.dataframe(pack["metrics_df"], use_container_width=True)
            m1, m2 = st.columns(2)
            lr = pack["models"]["Linear Regression"]
            rf = pack["models"]["Random Forest Regression"]
            with m1:
                st.markdown("**Linear Regression**")
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"₹{lr['mae']:,.2f}")
                c2.metric("MSE", f"{lr['mse']:,.0f}")
                c3.metric("R²", f"{lr['r2_score']:.4f}")
            with m2:
                st.markdown("**Random Forest Regression**")
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"₹{rf['mae']:,.2f}")
                c2.metric("MSE", f"{rf['mse']:,.0f}")
                c3.metric("R²", f"{rf['r2_score']:.4f}")
            st.success(f"Better regression model for this dataset: **{pack['winner']}**. {pack['winner_reason']}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card"><div class="section-title">📉 Model Metric Comparison Chart</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_regression_metrics_comparison(pack["metrics_df"]), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="glass-card"><div class="section-title">📈 Actual vs Predicted — Linear Regression</div>', unsafe_allow_html=True)
                st.plotly_chart(plot_actual_vs_predicted(lr["pred_df"], model_name="Linear Regression"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="glass-card"><div class="section-title">📈 Actual vs Predicted — Random Forest</div>', unsafe_allow_html=True)
                st.plotly_chart(plot_actual_vs_predicted(rf["pred_df"], model_name="Random Forest Regression"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card"><div class="section-title">📉 Residuals — Linear vs Random Forest</div>', unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            with r1:
                st.plotly_chart(plot_residual_distribution(lr["pred_df"]), use_container_width=True)
            with r2:
                st.plotly_chart(plot_residual_distribution(rf["pred_df"]), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 8: K-MEANS CLUSTERING
# ==========================================
elif page == "K-Means Clustering":
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">K-Means Customer Clustering</h1>
            <p class="app-subtitle">Unsupervised customer segmentation on behavioral dimensions: Purchase Amount, Frequency, and Rating.</p>
        </div>
    """, unsafe_allow_html=True)

    clean_df = st.session_state.cleaned_df

    st.markdown('<div class="glass-card"><div class="section-title">⚙️ Clustering Parameters</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        k_clusters = st.slider("Select Number of Clusters (K)", min_value=2, max_value=6, value=3)
    with c2:
        feature_options = ["Purchase_Amount", "Purchase_Frequency", "Customer_Rating", "Delivery_Days", "Age"]
        selected_features = st.multiselect(
            "Select Clustering Features",
            feature_options,
            default=["Purchase_Amount", "Purchase_Frequency", "Customer_Rating"]
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if len(selected_features) < 2:
        st.warning("Please select at least 2 features for clustering.")
    else:
        results, cluster_err = cached_kmeans(clean_df, tuple(selected_features), k_clusters)

        if cluster_err:
            st.error(cluster_err)
        else:
            # Metrics
            st.markdown('<div class="glass-card"><div class="section-title">📊 Clustering Quality Metrics</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Silhouette Score", f"{results['silhouette_score']:.4f}", help="Measures cluster separation (-1 to 1)")
            with m2:
                st.metric("Inertia (Sum of Squared Errors)", f"{results['inertia']:,.2f}")
            with m3:
                st.metric("Number of Clusters (K)", f"{results['k']}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Elbow Curve & 2D Cluster Visual
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="glass-card"><div class="section-title">📐 Elbow Curve (Optimal K)</div>', unsafe_allow_html=True)
                elbow_df = cached_elbow(clean_df, tuple(selected_features), 7)
                fig_elbow = plot_elbow_curve(elbow_df)
                st.plotly_chart(fig_elbow, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="glass-card"><div class="section-title">📍 2D Cluster Scatter Plot</div>', unsafe_allow_html=True)
                x_axis = selected_features[1] if len(selected_features) > 1 else selected_features[0]
                y_axis = selected_features[0]
                fig_2d = plot_clusters_2d(results["df_clustered"], x_col=x_axis, y_col=y_axis)
                st.plotly_chart(fig_2d, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 3D Visualization if >= 3 features
            if len(selected_features) >= 3:
                st.markdown('<div class="glass-card"><div class="section-title">🌐 3D Interactive Behavioral Space</div>', unsafe_allow_html=True)
                fig_3d = plot_clusters_3d(results["df_clustered"], selected_features[:3])
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Cluster Centroids & Distribution Tables
            c1, c2 = st.columns([1.2, 0.8])
            with c1:
                st.markdown('<div class="glass-card"><div class="section-title">🎯 Cluster Centers (Original Units)</div>', unsafe_allow_html=True)
                st.dataframe(results["centers_df"], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="glass-card"><div class="section-title">👥 Cluster Population Distribution</div>', unsafe_allow_html=True)
                st.dataframe(results["distribution_df"], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 9: COMPARISON & BEST MODEL
# ==========================================
elif page == "Comparison & Best Model":
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">Comparison & Best Model</h1>
            <p class="app-subtitle">Live results for association rules, classification, regression, and clustering on the active dataset. Algorithms solve different tasks, so there is no single overall winner.</p>
        </div>
    """, unsafe_allow_html=True)

    clean_df = st.session_state.cleaned_df

    st.markdown('<div class="glass-card"><div class="section-title">⚙️ Shared Evaluation Settings</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_supp = st.slider("Apriori min support", 0.01, 0.30, 0.05, 0.01, key="cmp_supp")
    with c2:
        min_conf = st.slider("Apriori min confidence", 0.10, 0.90, 0.30, 0.05, key="cmp_conf")
    with c3:
        tree_depth = st.slider("Decision Tree depth", 2, 8, 4, key="cmp_depth")
    with c4:
        k_clusters = st.slider("K-Means K", 2, 6, 3, key="cmp_k")
    test_size = st.slider("Classification / regression test split", 0.15, 0.40, 0.25, 0.05, key="cmp_split")
    st.markdown('</div>', unsafe_allow_html=True)

    cluster_features = ("Purchase_Amount", "Purchase_Frequency", "Customer_Rating")
    with st.spinner("Computing live comparison results..."):
        bundle = cached_full_comparison(
            clean_df, min_supp, min_conf, 3, tree_depth, test_size, k_clusters, cluster_features
        )

    def _table_and_chart(title, table, fig, empty_msg):
        st.markdown(f'<div class="glass-card"><div class="section-title">{title}</div>', unsafe_allow_html=True)
        left, right = st.columns([1, 1])
        with left:
            if table is None or (hasattr(table, "empty") and table.empty):
                st.info(empty_msg)
            else:
                st.dataframe(table, use_container_width=True)
        with right:
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Chart unavailable until this technique produces results.")
        st.markdown('</div>', unsafe_allow_html=True)

    assoc_err = bundle["errors"].get("association")
    if assoc_err:
        st.warning(f"Association rules: {assoc_err}")
    _table_and_chart(
        "1. Association Rules",
        bundle.get("association_table"),
        bundle.get("association_fig"),
        "No association rules were generated for the current thresholds."
    )

    dt_err = bundle["errors"].get("decision_tree") or bundle["errors"].get("classification_prep")
    if dt_err:
        st.warning(f"Decision Tree: {dt_err}")
    _table_and_chart(
        "2. Decision Tree Classification (scikit-learn)",
        bundle.get("dt_table"),
        bundle.get("dt_fig"),
        "Decision Tree metrics are unavailable."
    )

    nb_err = bundle["errors"].get("naive_bayes")
    if nb_err:
        st.warning(f"Naive Bayes: {nb_err}")
    _table_and_chart(
        "3. Naive Bayes Classification",
        bundle.get("nb_table"),
        bundle.get("nb_fig"),
        "Naive Bayes metrics are unavailable."
    )

    st.markdown('<div class="glass-card"><div class="section-title">4. Regression (Linear vs Random Forest)</div>', unsafe_allow_html=True)
    reg_err = bundle["errors"].get("regression") or bundle["errors"].get("regression_prep")
    if reg_err:
        st.warning(f"Regression: {reg_err}")
    elif bundle.get("regression"):
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(bundle["regression"]["metrics_df"], use_container_width=True)
            st.caption("Lower MAE = better · Lower MSE = better · Higher R² = better")
        with right:
            st.plotly_chart(
                plot_regression_metrics_comparison(bundle["regression"]["metrics_df"]),
                use_container_width=True
            )
    else:
        st.info("Regression results are unavailable.")
    st.markdown('</div>', unsafe_allow_html=True)

    k_err = bundle["errors"].get("clustering")
    if k_err:
        st.warning(f"K-Means: {k_err}")
    _table_and_chart(
        "5. K-Means Clustering",
        bundle.get("kmeans_table"),
        bundle.get("kmeans_fig"),
        "K-Means results are unavailable."
    )
    if bundle.get("kmeans_centers") is not None:
        st.markdown('<div class="glass-card"><div class="section-title">K-Means Cluster Centers</div>', unsafe_allow_html=True)
        st.dataframe(bundle["kmeans_centers"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    _table_and_chart(
        "6. Model Comparison (Decision Tree vs Naive Bayes)",
        bundle.get("classifier_comp_df"),
        bundle.get("classifier_comp_fig") or (
            plot_model_comparison_bar(bundle["classifier_comp_df"]) if bundle.get("classifier_comp_df") is not None else None
        ),
        "Classifier comparison is unavailable."
    )

    st.markdown('<div class="glass-card"><div class="section-title">Best-Suited Model for Each Task</div>', unsafe_allow_html=True)
    best = bundle.get("best", {})
    if best.get("classification"):
        st.markdown(f"**Classification**  \nBest-suited model: **{best['classification']['model']}**  \nReason: {best['classification']['reason']}")
    if best.get("regression"):
        st.markdown(f"**Regression**  \nBest-suited model: **{best['regression']['model']}**  \nReason: {best['regression']['reason']}")
    if best.get("clustering"):
        st.markdown(f"**Clustering**  \nK-Means evaluation:  \n{best['clustering']['reason']}")
    if best.get("association"):
        a = best["association"]
        st.markdown(
            f"**Association Rules**  \nStrongest rule: **{a['rule']}**  \n"
            f"Support: {a['support']:.4f} · Confidence: {a['confidence']:.4f} · Lift: {a['lift']:.4f}"
        )
    if not best:
        st.info("Best-suited summaries appear after techniques produce live results.")
    st.caption("These techniques address different mining tasks, so no algorithm is claimed as the overall best model.")
    st.markdown('</div>', unsafe_allow_html=True)

