# =============================================================
# 📦 Fashion Supply Management System Dashboard (Hardened)
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

# ---------------- DATA GENERATION (Replace with your CSV) ----------------
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="M")
products = ["T-Shirts", "Jeans", "Dresses", "Shoes", "Accessories"]
categories = ["Men", "Women", "Unisex"]
suppliers = ["Supplier A", "Supplier B", "Supplier C", "Supplier D"]

data = []
for _ in range(500):
    data.append({
        "Date": np.random.choice(dates),
        "Product": np.random.choice(products),
        "Category": np.random.choice(categories),
        "Supplier": np.random.choice(suppliers),
        "Sales": np.random.randint(100, 1000),
        "Inventory": np.random.randint(50, 500),
        "Lead_Time_Days": np.random.randint(5, 30),
        "Cost": np.random.uniform(10, 200)
    })

df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["Date"])

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔍 Filters")
selected_categories = st.sidebar.multiselect("Select Categories", df["Category"].unique(), default=list(df["Category"].unique()))
selected_products = st.sidebar.multiselect("Select Products", df["Product"].unique(), default=list(df["Product"].unique()))
supplier_filter = st.sidebar.multiselect("Select Suppliers", df["Supplier"].unique(), default=list(df["Supplier"].unique()))
date_range = st.sidebar.date_input("Select Date Range (start, end)", [df["Date"].min().date(), df["Date"].max().date()])

# ensure date_range is a pair (Streamlit can sometimes return a single date)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    # treat single selection as full-day range
    start_date = date_range
    end_date = date_range

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Apply filters
filtered_df = df[
    (df["Category"].isin(selected_categories)) &
    (df["Product"].isin(selected_products)) &
    (df["Supplier"].isin(supplier_filter)) &
    (df["Date"] >= start_date) &
    (df["Date"] <= end_date)
].copy()

# Early exit if no data after filters (prevents many crashes)
if filtered_df.empty:
    st.warning("No data matches your filters. Adjust filters or date range to see charts and KPIs.")
    st.header("📋 Filtered Dataset View")
    st.dataframe(filtered_df)  # empty view
    st.stop()  # stop further execution to avoid errors downstream

# ---------------- KPI SECTION ----------------
st.header("📈 Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

total_sales = int(filtered_df['Sales'].sum())
avg_inventory = filtered_df['Inventory'].mean()
avg_lead = filtered_df['Lead_Time_Days'].mean()
total_cost = float(filtered_df['Cost'].sum())

with col1:
    st.metric("Total Sales", f"${total_sales:,}")
with col2:
    st.metric("Average Inventory", f"{avg_inventory:.0f} units")
with col3:
    st.metric("Average Lead Time", f"{avg_lead:.1f} days")
with col4:
    st.metric("Total Cost", f"${total_cost:,.0f}")

# ---------------- VISUAL ANALYTICS ----------------
st.header("📊 Visual Analytics")

# Row 1: Sales Trend & Inventory
col1, col2 = st.columns(2)
with col1:
    st.subheader("Sales Trends Over Time")
    sales_trend = filtered_df.groupby("Date")["Sales"].sum().reset_index().sort_values("Date")
    fig_sales = px.line(sales_trend, x="Date", y="Sales", title="Monthly Sales Trend", markers=True)
    st.plotly_chart(fig_sales, use_container_width=True)

with col2:
    st.subheader("Inventory Levels by Product")
    inventory_bar = px.bar(filtered_df.groupby("Product")["Inventory"].mean().reset_index(),
                           x="Product", y="Inventory", title="Average Inventory per Product", color="Product")
    st.plotly_chart(inventory_bar, use_container_width=True)

# Row 2: Supplier & Forecast Comparison
col3, col4 = st.columns(2)
with col3:
    st.subheader("Supplier Performance")
    supplier_sales = filtered_df.groupby("Supplier")["Sales"].sum().reset_index()
    fig_supplier = px.pie(supplier_sales, names="Supplier", values="Sales",
                          title="Sales Distribution by Supplier", hole=0.4)
    st.plotly_chart(fig_supplier, use_container_width=True)

with col4:
    st.subheader("Demand Forecast vs Actual Sales")
    forecast_summary = filtered_df.groupby("Date")[["Sales"]].sum().reset_index().sort_values("Date")
    # rolling mean × scalar (scalar is OK)
    forecast_summary["Demand_Forecast"] = forecast_summary["Sales"].rolling(2, min_periods=1).mean() * np.random.uniform(0.9, 1.1)
    forecast_melted = forecast_summary.melt(id_vars="Date", value_vars=["Sales", "Demand_Forecast"],
                                            var_name="Type", value_name="Value")
    fig_forecast = px.bar(forecast_melted, x="Date", y="Value", color="Type", barmode="group",
                          title="Demand Forecast vs Actual Sales")
    st.plotly_chart(fig_forecast, use_container_width=True)

# Cost vs Sales Scatter
st.subheader("💰 Cost vs Sales Scatter Plot")
fig_scatter = px.scatter(filtered_df, x="Cost", y="Sales", color="Category", size="Inventory",
                         title="Cost Efficiency Analysis", hover_data=["Product", "Supplier"])
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------- FUTURE ANALYTICS SECTION ----------------
st.markdown("---")
st.header("🔮 Future Analytics")
forecast_mode = st.radio("Select Forecast Mode:", ["📈 Total Demand Forecast", "🚀 Product Boom Forecast"], horizontal=True)

# ---- Mode 1: Total Demand Forecast ----
if forecast_mode == "📈 Total Demand Forecast":
    ts = filtered_df.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index().sort_values("Date")
    if len(ts) >= 3:
        # safe ordinal conversion
        X = np.array(ts["Date"].map(lambda d: d.to_pydatetime().toordinal())).reshape(-1, 1)
        y = np.array(ts["Sales"])
        lr = LinearRegression().fit(X, y)
        future_dates = [ts["Date"].max() + pd.DateOffset(months=i) for i in range(1, 7)]
        preds = lr.predict(np.array([d.to_pydatetime().toordinal() for d in future_dates]).reshape(-1, 1))

        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Sales": preds})
        combined = pd.concat([
            ts.rename(columns={"Sales": "Value"}).assign(Type="Actual"),
            forecast_df.rename(columns={"Predicted_Sales": "Value"}).assign(Type="Predicted")
        ])
        fig_future = px.line(combined, x="Date", y="Value", color="Type",
                             title="6-Month Future Demand Forecast", markers=True)
        st.plotly_chart(fig_future, use_container_width=True)
        st.success(f"📦 Next Month Estimated Demand: **{int(preds[0])} units (approx.)**")
    else:
        st.warning("Not enough data for forecasting (need at least 3 monthly points).")

# ---- Mode 2: Product Boom Forecast ----
elif forecast_mode == "🚀 Product Boom Forecast":
    st.subheader("🚀 Product-wise Boom Forecast (Next Month Prediction)")
    boom_data = []
    last_date = filtered_df["Date"].max()
    for product, group in filtered_df.groupby("Product"):
        product_ts = group.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index().sort_values("Date")
        if len(product_ts) >= 3:
            X_p = np.array(product_ts["Date"].map(lambda d: d.to_pydatetime().toordinal())).reshape(-1, 1)
            y_p = np.array(product_ts["Sales"])
            model_p = LinearRegression().fit(X_p, y_p)
            next_month = last_date + pd.DateOffset(months=1)
            pred_next = model_p.predict(np.array([[next_month.to_pydatetime().toordinal()]]))[0]
            last_sales = product_ts.iloc[-1]["Sales"]
            growth = ((pred_next - last_sales) / last_sales) * 100 if last_sales > 0 else 0
            boom_data.append({"Product": product, "Predicted_Sales": pred_next, "Growth_%": growth})

    boom_df = pd.DataFrame(boom_data).sort_values("Predicted_Sales", ascending=False)
    if not boom_df.empty:
        fig_boom = px.bar(boom_df, x="Product", y="Predicted_Sales", color="Growth_%",
                          text=boom_df["Growth_%"].apply(lambda x: f"{x:.1f}%"),
                          title="Top Products Expected to Boom Next Month",
                          color_continuous_scale=px.colors.sequential.Viridis)
        fig_boom.update_traces(textposition="outside")
        st.plotly_chart(fig_boom, use_container_width=True)
        top_boom = boom_df.iloc[0]
        st.success(f"🔥 {top_boom['Product']} projected to boom: **{int(top_boom['Predicted_Sales'])} units** (+{top_boom['Growth_%']:.1f}% growth).")
    else:
        st.info("Not enough data for booming products.")

# ---------------- INVENTORY OPTIMIZATION SYSTEM ----------------
st.markdown("---")
st.header("📦 Smart Inventory Optimization & Reorder Alerts")

inventory_df = filtered_df.groupby("Product").agg({
    "Sales": "mean", "Inventory": "mean", "Lead_Time_Days": "mean"
}).reset_index()
inventory_df["Reorder_Level"] = (inventory_df["Sales"] * (inventory_df["Lead_Time_Days"] / 7)).round()
inventory_df["Status"] = np.where(inventory_df["Inventory"] < inventory_df["Reorder_Level"], "⚠️ Low Stock", "✅ Sufficient")

fig_inv = px.bar(inventory_df, x="Product", y=["Inventory", "Reorder_Level"],
                 barmode="group", title="Inventory vs Reorder Threshold")
st.plotly_chart(fig_inv, use_container_width=True)

low_stock = inventory_df[inventory_df["Status"] == "⚠️ Low Stock"].copy()
if not low_stock.empty:
    st.warning("⚠️ The following products are below safe stock levels:")
    low_stock["Suggested_Reorder_Qty"] = (low_stock["Reorder_Level"] * 1.5 - low_stock["Inventory"]).clip(lower=0).astype(int)
    st.dataframe(low_stock[["Product", "Inventory", "Reorder_Level", "Suggested_Reorder_Qty", "Status"]])
else:
    st.success("✅ All products are above their safe stock levels.")

# ---------------- DATA TABLE & DOWNLOAD ----------------
st.header("📋 Filtered Dataset View")
st.dataframe(filtered_df)
csv = filtered_df.to_csv(index=False)
st.download_button("⬇️ Download Filtered Data as CSV", csv, "fashion_supply_data.csv", "text/csv")

st.markdown("---")
st.markdown("🧵 **Fashion Supply Management Dashboard** — Built with ❤️ using Streamlit and Plotly")
