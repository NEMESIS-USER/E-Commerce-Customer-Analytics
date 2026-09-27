"""
Dataset Creation Module
Generates a realistic synthetic dataset for E-Commerce Customer Analytics & Data Mining System.
Includes deliberate missing values and duplicate rows for preprocessing demonstration.
"""

import os
import random
import numpy as np
import pandas as pd


def generate_dataset(num_records=5000, seed=42, output_path="data/ecommerce_customer_data.csv"):
    """
    Generates synthetic e-commerce customer transaction data with realistic
    correlations, controlled missing values, and duplicate records.
    """
    np.random.seed(seed)
    random.seed(seed)

    cities = [
        "Pune", "Mumbai", "Delhi", "Bangalore",
        "Hyderabad", "Chennai", "Kolkata", "Ahmedabad"
    ]
    genders = ["Male", "Female", "Other"]
    gender_weights = [0.48, 0.48, 0.04]

    product_categories = [
        "Electronics", "Clothing", "Beauty", "Grocery", "Sports", "Home"
    ]
    payment_methods = [
        "UPI", "Credit Card", "Debit Card", "Cash on Delivery", "Net Banking"
    ]

    records = []

    for i in range(1, num_records + 1):
        cust_id = f"CUST_{10000 + i}"
        age = int(np.random.normal(35, 12))
        age = max(18, min(70, age))

        gender = random.choices(genders, weights=gender_weights)[0]
        city = random.choice(cities)
        category = random.choice(product_categories)

        segment_choice = random.choices(
            ["Low", "Medium", "High"],
            weights=[0.38, 0.42, 0.20]
        )[0]

        if segment_choice == "Low":
            freq = random.randint(1, 6)
            rating = round(random.uniform(2.0, 4.3), 1)
            discount_prob = 0.65
            if category == "Electronics":
                amount = round(random.uniform(2500, 12000), 2)
            elif category in ["Home", "Sports"]:
                amount = round(random.uniform(1000, 5000), 2)
            else:
                amount = round(random.uniform(300, 2500), 2)
            deliv_days = random.randint(3, 10)
            payment = random.choices(
                payment_methods,
                weights=[0.35, 0.15, 0.20, 0.25, 0.05]
            )[0]

        elif segment_choice == "Medium":
            freq = random.randint(4, 16)
            rating = round(random.uniform(3.0, 4.8), 1)
            discount_prob = 0.50
            if category == "Electronics":
                amount = round(random.uniform(6000, 24000), 2)
            elif category in ["Home", "Sports"]:
                amount = round(random.uniform(2500, 9000), 2)
            else:
                amount = round(random.uniform(800, 4500), 2)
            deliv_days = random.randint(2, 7)
            payment = random.choices(
                payment_methods,
                weights=[0.45, 0.25, 0.15, 0.10, 0.05]
            )[0]

        else:  # High
            freq = random.randint(12, 30)
            rating = round(random.uniform(3.6, 5.0), 1)
            discount_prob = 0.35
            if category == "Electronics":
                amount = round(random.uniform(15000, 45000), 2)
            elif category in ["Home", "Sports"]:
                amount = round(random.uniform(5000, 18000), 2)
            else:
                amount = round(random.uniform(1500, 8000), 2)
            deliv_days = random.randint(1, 5)
            payment = random.choices(
                payment_methods,
                weights=[0.35, 0.50, 0.08, 0.02, 0.05]
            )[0]

        # Keep realistic relationships but avoid a perfectly separable target.
        if random.random() < 0.12:
            segment_choice = random.choice(["Low", "Medium", "High"])
        if random.random() < 0.08:
            freq = max(1, min(30, freq + random.choice([-4, -3, 3, 5, 7])))
        if random.random() < 0.08:
            amount = round(max(200.0, amount * random.uniform(0.55, 1.55)), 2)

        discount_used = "Yes" if random.random() < discount_prob else "No"

        records.append({
            "Customer_ID": cust_id,
            "Age": age,
            "Gender": gender,
            "City": city,
            "Product_Category": category,
            "Purchase_Amount": amount,
            "Purchase_Frequency": freq,
            "Discount_Used": discount_used,
            "Customer_Rating": rating,
            "Delivery_Days": deliv_days,
            "Payment_Method": payment,
            "Customer_Segment": segment_choice
        })

    df = pd.DataFrame(records)

    # Intentional duplicate rows (~1.2%) so preprocessing can demonstrate removal.
    n_duplicates = max(40, int(num_records * 0.012))
    dup_indices = np.random.choice(len(df), size=n_duplicates, replace=False)
    duplicates = df.iloc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # Small missing-value rate across a few columns.
    n_rows = len(df)
    age_idx = np.random.choice(n_rows, size=max(20, int(n_rows * 0.016)), replace=False)
    df.loc[age_idx, "Age"] = np.nan

    rating_idx = np.random.choice(n_rows, size=max(20, int(n_rows * 0.018)), replace=False)
    df.loc[rating_idx, "Customer_Rating"] = np.nan

    deliv_idx = np.random.choice(n_rows, size=max(15, int(n_rows * 0.014)), replace=False)
    df.loc[deliv_idx, "Delivery_Days"] = np.nan

    disc_idx = np.random.choice(n_rows, size=max(15, int(n_rows * 0.012)), replace=False)
    df.loc[disc_idx, "Discount_Used"] = np.nan

    pay_idx = np.random.choice(n_rows, size=max(12, int(n_rows * 0.010)), replace=False)
    df.loc[pay_idx, "Payment_Method"] = np.nan

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at: {output_path}")
    print(f"Total Rows: {len(df)}, Total Columns: {len(df.columns)}")
    print(f"Total Missing Values: {df.isnull().sum().sum()}")
    print(f"Duplicate Rows Count: {df.duplicated().sum()}")
    print(f"Customer_Segment values: {sorted(df['Customer_Segment'].dropna().unique().tolist())}")
    return df


if __name__ == "__main__":
    generate_dataset()
