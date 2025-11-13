# =============================================================
# 📦 Fashion Supply Management System Dashboard (Dataset-Ready)
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime

st.set_page_config(page_title="Fashion Supply Chain Management Dashboard", layout="wide")

st.title("📊 Fashion Supply Chain Management Analytics")
st.markdown("""
Welcome to the **Fashion Supply Management System Dashboard**.  
This application provides data-driven insights into sales, inventory, suppliers and forecasts.
""")

# ---------------- LOAD YOUR DATASET ----------------
uploaded_path = "supply_chain_clean_deploy_ready.csv"

df = pd.read_csv(uploaded_path)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔍 Filters")

selected_categories = st.sidebar.multiselect(
    "Select Categories",
    df["Category"].dropna().unique(),
    default=list(df["Category"].dropna().unique())
)

selected_products = st.sidebar.multiselect(
    "Select Products",
    df["Product"].dropna().unique(),
    default=list(df["Product"].dropna().unique())
)

# If supplier column exists, allow filtering
supplier_column = None
for col in df.columns:
    if col.lower() in ["supplier", "suppliers", "vendor"]:
        supplier_column = col

if supplier_column:
    supplier_filter = st.sidebar.multiselect(
        "Select Suppliers",
        df[supplier_column].dropna().unique(),
        default=list(df[supplier_column].dropna().unique())
    )
else:
    supplier_filter = None

date_range = st.sidebar.date_input(
    "Select Date Range (start, end)",
    [df["Date"].min().date(), df["Date"].max().date()]
)

# Ensure valid date range
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# ---------------- APPLY FILTERS ----------------
filtered_df = df[
    (df["Category"].isin(selected_categories)) &
    (df["Product"].isin(selected_products)) &
    (df["Date"] >= start_date) &
    (df["Date"] <= end_date)
].copy()

if supplier_filter and supplier_column:
    filtered_df = filtered_df[filtered_df[supplier_column].isin(supplier_filter)]

# Early exit if empty
if filtered_df.empty:
    st.warning("No data matches your filters. Please adjust the filters.")
    st.dataframe(filtered_df)
    st.stop()

# ---------------- KPI SECTION ----------------
st.header("📈 Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

total_sales = int(filtered_df['Sales'].sum())
avg_inventory = filtered_df['Inventory'].mean()
avg_lead = filtered_df['Lead_Time_Days'].mean()
total_cost = float(filtered_df['Cost'].sum())

with col1:
    st.metric("Total Sales", f"{total_sales:,}")
with col2:
    st.metric("Average Inventory", f"{avg_inventory:.0f} units")
with col3:
    st.metric("Average Lead Time", f"{avg_lead:.1f} days")
with col4:
    st.metric("Total Cost", f"₹{total_cost:,.0f}")

# ---------------- VISUAL ANALYTICS ----------------
st.header("📊 Visual Analytics")

# Row 1
col1, col2 = st.columns(2)
with col1:
    st.subheader("Sales Trend")
    sales_trend = (
        filtered_df.groupby("Date")["Sales"].sum()
        .reset_index().sort_values("Date")
    )
    fig_sales = px.line(sales_trend, x="Date", y="Sales", markers=True)
    st.plotly_chart(fig_sales, use_container_width=True)

with col2:
    st.subheader("Inventory by Product")
    inventory_bar = px.bar(
        filtered_df.groupby("Product")["Inventory"].mean().reset_index(),
        x="Product", y="Inventory", color="Product"
    )
    st.plotly_chart(inventory_bar, use_container_width=True)

# Row 2
col3, col4 = st.columns(2)
with col3:
    st.subheader("Supplier Performance")
    if supplier_column:
        supplier_sales = filtered_df.groupby(supplier_column)["Sales"].sum().reset_index()
        fig_supplier = px.pie(
            supplier_sales,
            names=supplier_column, values="Sales",
            hole=0.4
        )
        st.plotly_chart(fig_supplier, use_container_width=True)
    else:
        st.info("Supplier column not available in dataset.")

with col4:
    st.subheader("Demand Forecast vs Sales")
    forecast_summary = (
        filtered_df.groupby("Date")["Sales"].sum()
        .reset_index().sort_values("Date")
    )
    forecast_summary["Demand_Forecast"] = (
        forecast_summary["Sales"].rolling(2, min_periods=1).mean()
        * np.random.uniform(0.9, 1.1)
    )
    melted = forecast_summary.melt(
        id_vars="Date",
        value_vars=["Sales", "Demand_Forecast"],
        var_name="Type", value_name="Value"
    )
    fig_forecast = px.bar(melted, x="Date", y="Value", color="Type", barmode="group")
    st.plotly_chart(fig_forecast, use_container_width=True)

# Scatter Plot
st.subheader("💰 Cost vs Sales")
fig_scatter = px.scatter(
    filtered_df,
    x="Cost", y="Sales",
    color="Category",
    size="Inventory",
    hover_data=["Product"]
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------- FUTURE ANALYTICS ----------------
st.markdown("---")
st.header("🔮 Future Analytics")
forecast_mode = st.radio(
    "Choose Forecast Mode:",
    ["📈 Total Demand Forecast", "🚀 Product Boom Forecast"],
    horizontal=True
)

# Forecast Mode 1
if forecast_mode == "📈 Total Demand Forecast":
    ts = filtered_df.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index()

    if len(ts) >= 3:
        X = ts["Date"].map(lambda d: d.toordinal()).values.reshape(1, -1).T
        y = ts["Sales"].values
        lr = LinearRegression().fit(X, y)

        future_dates = [ts["Date"].max() + pd.DateOffset(months=i) for i in range(1, 7)]
        preds = lr.predict(np.array([d.toordinal() for d in future_dates]).reshape(-1, 1))

        pred_df = pd.DataFrame({"Date": future_dates, "Predicted_Sales": preds})

        combined = pd.concat([
            ts.rename(columns={"Sales": "Value"}).assign(Type="Actual"),
            pred_df.rename(columns={"Predicted_Sales": "Value"}).assign(Type="Predicted")
        ])

        fig_future = px.line(combined, x="Date", y="Value", color="Type", markers=True)
        st.plotly_chart(fig_future, use_container_width=True)

        st.success(f"📦 Next Month Forecast: **{int(preds[0])} units**")

    else:
        st.warning("Not enough data for time-series forecasting.")

# Forecast Mode 2
elif forecast_mode == "🚀 Product Boom Forecast":
    boom_data = []

    for product, group in filtered_df.groupby("Product"):
        ts = group.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index()

        if len(ts) >= 3:
            X = ts["Date"].map(lambda d: d.toordinal()).values.reshape(-1, 1)
            y = ts["Sales"].values
            model = LinearRegression().fit(X, y)

            next_month = ts["Date"].max() + pd.DateOffset(months=1)
            pred_next = model.predict(np.array([[next_month.toordinal()]])).flatten()[0]

            last_val = ts.iloc[-1]["Sales"]
            growth = ((pred_next - last_val) / last_val * 100) if last_val > 0 else 0

            boom_data.append({"Product": product, "Predicted_Sales": pred_next, "Growth_%": growth})

    boom_df = pd.DataFrame(boom_data).sort_values("Predicted_Sales", ascending=False)

    if not boom_df.empty:
        fig_boom = px.bar(
            boom_df, x="Product", y="Predicted_Sales", color="Growth_%",
            text=boom_df["Growth_%"].apply(lambda v: f"{v:.1f}%")
        )
        fig_boom.update_traces(textposition="outside")
        st.plotly_chart(fig_boom, use_container_width=True)

        top = boom_df.iloc[0]
        st.success(
            f"🔥 **{top['Product']}** expected boom: **{int(top['Predicted_Sales'])} units** (+{top['Growth_%']:.1f}%)"
        )

# ---------------- INVENTORY OPTIMIZATION ----------------
st.markdown("---")
st.header("📦 Inventory Optimization & Reorder Alerts")

inventory_df = filtered_df.groupby("Product").agg({
    "Sales": "mean",
    "Inventory": "mean",
    "Lead_Time_Days": "mean"
}).reset_index()

inventory_df["Reorder_Level"] = (inventory_df["Sales"] * (inventory_df["Lead_Time_Days"] / 7)).round()
inventory_df["Status"] = np.where(
    inventory_df["Inventory"] < inventory_df["Reorder_Level"], "⚠️ Low Stock",
    "✅ Sufficient"
)

fig_inv = px.bar(
    inventory_df,
    x="Product",
    y=["Inventory", "Reorder_Level"],
    barmode="group"
)
st.plotly_chart(fig_inv, use_container_width=True)

low_stock = inventory_df[inventory_df["Status"] == "⚠️ Low Stock"]
if not low_stock.empty:
    st.warning("⚠️ Low stock detected:")
    low_stock["Suggested_Reorder_Qty"] = (
        low_stock["Reorder_Level"] * 1.5 - low_stock["Inventory"]
    ).clip(lower=0).astype(int)
    st.dataframe(low_stock)
else:
    st.success("All products are sufficiently stocked.")

# ---------------- DATA TABLE ----------------
st.header("📋 Filtered Dataset")
st.dataframe(filtered_df)

csv = filtered_df.to_csv(index=False)
st.download_button("⬇️ Download CSV", csv, "filtered_supply_data.csv", "text/csv")

st.markdown("---")
st.markdown("🧵 **Fashion Supply Management Dashboard — Powered by Streamlit**")
