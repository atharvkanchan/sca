# models.py
# ML model utilities for the AI Supply Chain Dashboard

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import streamlit as st


def run_sales_forecast(df: pd.DataFrame, n_estimators: int = 400):
    """Train a RandomForest on historical sales and show a 6-month forecast."""
    df = df.copy()
    if df.empty:
        st.info("No data available for sales forecast.")
        return

    df["Date_Ordinal"] = df["Date"].map(lambda d: d.toordinal())
    le_p = LabelEncoder()
    le_c = LabelEncoder()
    try:
        df["Product_Code"] = le_p.fit_transform(df["Product"].astype(str))
        df["Category_Code"] = le_c.fit_transform(df["Category"].astype(str))
    except Exception:
        df["Product_Code"] = 0
        df["Category_Code"] = 0

    features = ["Product_Code", "Category_Code", "Inventory", "Lead_Time_Days", "Cost", "Date_Ordinal"]
    for f in features:
        if f not in df.columns:
            df[f] = 0

    X = df[features]
    y = df["Sales"]

    if len(X) < 10:
        st.info("Not enough rows to train RandomForest (need 10+).")
        return

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X, y)

    last_date = df["Date"].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 7)]
    future_df = pd.DataFrame({
        "Product_Code": [df["Product_Code"].mode()[0]] * 6,
        "Category_Code": [df["Category_Code"].mode()[0]] * 6,
        "Inventory": [df["Inventory"].mean()] * 6,
        "Lead_Time_Days": [df["Lead_Time_Days"].mean()] * 6,
        "Cost": [df["Cost"].mean()] * 6,
        "Date_Ordinal": [d.toordinal() for d in future_dates]
    })

    preds = rf.predict(future_df)
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted_Sales": preds})
    fig = px.line(pred_df, x="Date", y="Predicted_Sales", markers=True, title="6-Month Sales Forecast (RF)")
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Next month predicted sales: {int(preds[0])}")


def run_product_boom(df: pd.DataFrame):
    """Per-product linear regression to predict next-month growth."""
    df = df.copy()
    boom_list = []
    for prod, grp in df.groupby("Product"):
        grp = grp.sort_values("Date")
        if grp.shape[0] < 3:
            continue
        X = grp["Date"].map(lambda d: d.toordinal()).values.reshape(-1, 1)
        y = grp["Sales"].values
        lr = LinearRegression().fit(X, y)
        next_month = grp["Date"].max() + pd.DateOffset(months=1)
        pred = lr.predict([[next_month.toordinal()]])[0]
        last_val = grp.iloc[-1]["Sales"]
        growth = ((pred - last_val) / last_val * 100) if last_val != 0 else 0
        boom_list.append({"Product": prod, "Predicted_Next_Month": pred, "Growth_%": growth})

    boom_df = pd.DataFrame(boom_list).sort_values("Predicted_Next_Month", ascending=False)
    if boom_df.empty:
        st.info("Not enough product-level history for boom predictions.")
        return
    fig = px.bar(boom_df, x="Product", y="Growth_%", title="Projected Growth % Next Month (by Product)", text=boom_df["Growth_%"].apply(lambda x: f"{x:.1f}%"))
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    top = boom_df.iloc[0]
    st.success(f"Top expected boom: {top['Product']} → {int(top['Predicted_Next_Month'])} units (+{top['Growth_%']:.1f}%)")


def run_inventory_forecast(df: pd.DataFrame):
    """Simple linear model to estimate future inventory requirement."""
    df = df.copy()
    if df.shape[0] < 5:
        st.info("Not enough data to train inventory model (need 5+ rows).")
        return
    df["Date_Ordinal"] = df["Date"].map(lambda d: d.toordinal())
    X = df[["Sales", "Lead_Time_Days", "Date_Ordinal"]]
    y = df["Inventory"]
    lr = LinearRegression().fit(X, y)
    next_inv = lr.predict([[df["Sales"].mean(), df["Lead_Time_Days"].mean(), df["Date"].max().toordinal()]])[0]
    st.info(f"Recommended future inventory (model): {int(next_inv)} units")
