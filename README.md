# E-Commerce Customer Analytics & Data Mining System

A comprehensive, interactive Data Warehousing and Data Mining (DWM) mini-project built with Python, Streamlit, Scikit-Learn, MLxtend, and Plotly.

---

## 📌 Project Overview & Objectives

In modern e-commerce enterprises, extracting actionable insights from transaction logs and customer profiles is essential for targeted marketing, inventory planning, and churn mitigation. 

This mini-project combines theoretical and practical implementations across **10 DWM core topics**:
1. **Data Warehouse Architecture**: Centralized dimensional schema representation for customer behavioral analytics.
2. **WEKA / Data Mining Concepts**: Algorithmic equivalents implemented natively in Python.
3. **Dataset Creation**: Synthetic generator (~1,000+ records) with realistic cross-feature correlations and injected noise (missingness + duplicate records).
4. **Data Preprocessing**: Missing value imputation (Mean/Median/Mode), deduplication, categorical encoding (Label / One-Hot), and feature scaling.
5. **Association Rule Mining**: Market basket affinity mining using the Apriori algorithm (Support, Confidence, Lift).
6. **J48 / Decision Tree Classification**: C4.5 / J48 equivalent classifier predicting `Customer_Segment` with decision tree visualizer.
7. **Naive Bayes Classification**: Probabilistic Bayesian classification with prior probabilities and confusion matrix.
8. **Regression Analysis**: Continuous purchase amount prediction (₹) with MAE, MSE, RMSE, and $R^2$ goodness-of-fit metrics.
9. **K-Means Clustering**: Unsupervised segmentation on behavioral metrics with Elbow Method and Silhouette Score.
10. **Integrated Final System**: A unified SaaS-styled Streamlit dashboard combining all modules into an intuitive interactive workflow.

---

## 🛠️ Technology Stack

- **Frontend & Web App**: [Streamlit](https://streamlit.io/) (Dark SaaS Modern Dashboard)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/)
- **Association Mining**: [MLxtend](http://rasbt.github.io/mlxtend/)
- **Interactive Visualizations**: [Plotly](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/)

---

## 📂 Project Structure

```
E-Commerce Customer Analytics & Data Mining System/
│
├── data/
│   └── ecommerce_customer_data.csv       # Synthetic e-commerce dataset (1,025 rows)
│
├── outputs/                              # Directory for reports and exports
│
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py                  # Missing value imputation, deduplication, encoding
│   ├── association_rules.py              # Apriori frequent itemset & rule mining
│   ├── j48_classification.py             # J48 / Decision Tree classifier & tree diagram
│   ├── naive_bayes.py                    # Gaussian Naive Bayes classifier & priors
│   ├── regression.py                     # Linear / Ridge / Random Forest Regression
│   ├── clustering.py                     # K-Means clustering, Elbow method, 2D/3D space
│   └── comparison.py                     # Side-by-side classifier benchmark (DT vs NB)
│
├── dataset_creation.py                   # Script for generating synthetic customer records
├── app.py                                # Main Streamlit multi-page web application
├── requirements.txt                      # Project dependency definitions
└── README.md                             # Documentation and setup instructions
```

---

## 📊 Dataset Schema

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `Customer_ID` | String | Unique customer identifier (`CUST_1001`...) |
| `Age` | Integer | Customer age (18 to 70 years) |
| `Gender` | String | Male, Female, Other |
| `City` | String | Pune, Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata, Ahmedabad |
| `Product_Category` | String | Electronics, Clothing, Beauty, Grocery, Sports, Home |
| `Purchase_Amount` | Float | Transaction spend in ₹ |
| `Purchase_Frequency`| Integer | Annual order count (1 to 30) |
| `Discount_Used` | String | Whether a coupon/discount was redeemed (Yes / No) |
| `Customer_Rating` | Float | Product/service satisfaction rating (1.0 to 5.0) |
| `Delivery_Days` | Integer | Delivery turnaround time in days (1 to 10) |
| `Payment_Method` | String | UPI, Credit Card, Debit Card, Cash on Delivery, Net Banking |
| `Customer_Segment` | String | Target class: **Occasional**, **Regular**, **Premium** |

---

## 🚀 Setup & Execution Guide

### 1. Clone or Open Project
Open the project directory in **VS Code** or your preferred IDE:
```bash
cd "E-Commerce Customer Analytics & Data Mining System"
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset (Optional)
The application automatically generates the default dataset if not present. To manually regenerate it:
```bash
python dataset_creation.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the dashboard.

---

## 🌟 Key Application Pages

1. **📊 Dashboard**: High-level KPI metric cards, segment distributions, revenue histograms, and end-to-end pipeline diagram.
2. **📁 Dataset**: CSV file uploader, schema breakdown, missing value counters, and raw data explorer.
3. **🧹 Preprocessing**: Customizable imputation (Mean/Median/Mode/Zero), deduplication, and before-vs-after delta metrics.
4. **🔗 Association Rules**: Tunable Apriori support and confidence sliders, lift metrics, and scatter maps.
5. **🌳 J48 Classification**: Decision tree depth tuning, confusion matrix heatmap, feature importance, and graphical tree plot.
6. **🔮 Naive Bayes**: Probabilistic classification, class priors, and multi-class confusion matrix.
7. **📈 Regression**: Predicts `Purchase_Amount`, calculates MAE, MSE, RMSE, $R^2$, with actual vs. predicted 45° plots.
8. **🎯 K-Means Clustering**: Interactive Elbow curve, Silhouette score, 2D and 3D customer behavioral cluster projections.
9. **⚖️ Model Comparison**: Side-by-side grouped bar chart comparing Accuracy, Precision, Recall, and F1 score for academic evaluation.

---

## 👥 Authors & Academic Attribution
Developed as a Data Warehousing & Data Mining (DWM) College Mini-Project.
