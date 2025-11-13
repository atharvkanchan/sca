# supplier_optimizer.py
# AI supplier optimizer and supplier mix optimizer module

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from utils import safe_series, safe_col, normalize


def train_supplier_models(df: pd.DataFrame, n_estimators: int = 400):
    """Train RandomForest models to predict delivery delay and total cost per shipment."""
    df = df.copy()
    df["delivery_delay"] = safe_series(df, "delay_days", 0).astype(float)
    df["total_cost_INR"] = safe_series(df, "total_cost_INR", df.get("Cost", 0) * df.get("Sales", 0))

    supplier_col = safe_col(df, ["supplier", "suppliers", "carrier", "vendor"], default=None)
    if supplier_col is None:
        df["Supplier_Fallback"] = "Supplier A"
        supplier_col = "Supplier_Fallback"

    le = LabelEncoder()
    df["Supplier_Code"] = le.fit_transform(df[supplier_col].astype(str))

    # feature selection with graceful fallbacks
    features = []
    for f in ["Supplier_Code", "distance_km", "Inventory", "Lead_Time_Days", "Cost", "Sales"]:
        if f not in df.columns:
            df[f] = safe_series(df, f, 0)
        features.append(f)

    X = df[features]
    y_delay = df["delivery_delay"]
    y_cost = df["total_cost_INR"]

    model_delay = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model_cost = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

    can_train = len(X) >= 30
    if can_train:
        model_delay.fit(X, y_delay)
        model_cost.fit(X, y_cost)
    else:
        model_delay = model_cost = None

    return {
        "model_delay": model_delay,
        "model_cost": model_cost,
        "le_supplier": le,
        "features": features,
        "trained": can_train
    }


def show_optimizer_ui(df: pd.DataFrame, supplier_col: str, n_estimators: int = 400, w_cost: float = 0.5):
    """Render the optimizer UI inside Streamlit app.
    Trains models (if sufficient data) and displays recommendations + mix allocation.
    """
    st.subheader("Supplier AI Optimizer")
    df = df.copy()

    # Train models
    with st.spinner("Training supplier models..."):
        models = train_supplier_models(df, n_estimators=n_estimators)

    model_delay = models["model_delay"]
    model_cost = models["model_cost"]
    le_supplier = models["le_supplier"]
    features = models["features"]
    trained = models["trained"]

    product_choice = st.selectbox("Choose product to optimize", df["Product"].unique())
    product_data = df[df["Product"] == product_choice]
    sup_list = product_data[supplier_col].unique().tolist()

    if not sup_list:
        st.warning("No suppliers found for this product.")
        return

    recs = []
    for s in sup_list:
        subset = product_data[product_data[supplier_col] == s]
        if subset.empty:
            # fallback to global mean
            pred_delay = float(df["delivery_delay"].mean()) if "delivery_delay" in df.columns else 0.0
            pred_cost = float(df["total_cost_INR"].mean()) if "total_cost_INR" in df.columns else 0.0
        else:
            row = subset.iloc[0]
            input_row = {f: row.get(f, df.get(f, df[f].mean() if f in df.columns else 0)) for f in features}
            X_in = pd.DataFrame([input_row])[features]
            if trained and model_delay is not None and model_cost is not None:
                pred_delay = float(model_delay.predict(X_in)[0])
                pred_cost = float(model_cost.predict(X_in)[0])
            else:
                pred_delay = float(subset.get("delay_days", subset.get("actual_lead_days", pd.Series([0]))).mean())
                pred_cost = float(subset.get("total_cost_INR", (subset.get("Cost", 0) * subset.get("Sales", 0))).mean())

        score = (1.0 / (pred_delay + 1e-6)) * 0.6 + (1.0 / (pred_cost + 1e-6)) * 0.4
        recs.append({"Supplier": s, "Predicted_Delay": pred_delay, "Predicted_Cost": pred_cost, "Opt_Score": score})

    rec_df = pd.DataFrame(recs).sort_values("Opt_Score", ascending=False)
    st.subheader("AI Supplier Recommendation")
    st.dataframe(rec_df.style.format({"Predicted_Delay": "{:.2f}", "Predicted_Cost": "₹{:.2f}", "Opt_Score": "{:.6f}"}))

    best = rec_df.iloc[0]
    st.success(f"Recommended supplier: {best['Supplier']} — Delay {best['Predicted_Delay']:.2f} d, Cost ₹{best['Predicted_Cost']:.2f}")

    # Mix optimizer
    st.subheader("Supplier Mix Optimizer (Hybrid)")
    mix_df = rec_df.copy()
    mix_df["inv_cost"] = 1.0 / (mix_df["Predicted_Cost"] + 1e-6)
    mix_df["inv_delay"] = 1.0 / (mix_df["Predicted_Delay"] + 1e-6)
    mix_df["inv_cost_n"] = normalize(mix_df["inv_cost"])
    mix_df["inv_delay_n"] = normalize(mix_df["inv_delay"])

    w_cost_local = w_cost
    w_delay_local = 1.0 - w_cost_local
    mix_df["Hybrid_Score"] = w_cost_local * mix_df["inv_cost_n"] + w_delay_local * mix_df["inv_delay_n"]
    mix_df["Proportion"] = mix_df["Hybrid_Score"] / mix_df["Hybrid_Score"].sum()

    st.dataframe(mix_df[["Supplier", "Predicted_Delay", "Predicted_Cost", "Hybrid_Score", "Proportion"]].style.format({"Predicted_Delay": "{:.2f}", "Predicted_Cost": "₹{:.2f}", "Proportion": "{:.2%}"}))
    fig = px.pie(mix_df, names="Supplier", values="Proportion", title=f"Supplier Mix for {product_choice}", hole=0.35)
    st.plotly_chart(fig, use_container_width=True)
