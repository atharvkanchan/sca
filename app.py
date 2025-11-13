# =============================================================
# 📦 Fashion Supply Management System Dashboard (Dataset-Ready)
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# load safely
try:
    df = pd.read_csv(uploaded_path)
except Exception as e:
    st.error(f"Unable to load dataset at `{uploaded_path}` — {e}")
    st.stop()

# ensure Date present and parse
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
else:
    st.error("Dataset must contain a 'Date' column.")
    st.stop()

# normalize column names (strip)
df.columns = [c.strip() for c in df.columns]

# helper existence flags
has_product = "Product" in df.columns
has_category = "Category" in df.columns
has_sales = "Sales" in df.columns
has_inventory = "Inventory" in df.columns
has_lead = "Lead_Time_Days" in df.columns
has_cost = "Cost" in df.columns

# If supplier column exists, detect it
supplier_column = None
for col in df.columns:
    if col.lower() in ["supplier", "suppliers", "vendor", "vendor_name"]:
        supplier_column = col
        break

# fill missing numeric cols with zeros to avoid plotting issues
for col in ["Sales", "Inventory", "Lead_Time_Days", "Cost"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔍 Filters")

if has_category:
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        df["Category"].dropna().unique(),
        default=list(df["Category"].dropna().unique())
    )
else:
    selected_categories = []

if has_product:
    selected_products = st.sidebar.multiselect(
        "Select Products",
        df["Product"].dropna().unique(),
        default=list(df["Product"].dropna().unique())
    )
else:
    selected_products = []

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
filtered_df = df.copy()

if has_category and selected_categories:
    filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]

if has_product and selected_products:
    filtered_df = filtered_df[filtered_df["Product"].isin(selected_products)]

if supplier_filter and supplier_column:
    filtered_df = filtered_df[filtered_df[supplier_column].isin(supplier_filter)]

filtered_df = filtered_df[(filtered_df["Date"] >= start_date) & (filtered_df["Date"] <= end_date)].copy()

# Early exit if empty
if filtered_df.empty:
    st.warning("No data matches your filters. Please adjust the filters.")
    st.dataframe(filtered_df)
    st.stop()

# ---------------- KPI SECTION ----------------
st.header("📈 Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

total_sales = int(filtered_df['Sales'].sum()) if has_sales else 0
avg_inventory = filtered_df['Inventory'].mean() if has_inventory else 0
avg_lead = filtered_df['Lead_Time_Days'].mean() if has_lead else 0
total_cost = float(filtered_df['Cost'].sum()) if has_cost else 0.0

with col1:
    st.metric("Total Sales", f"₹{total_sales:,}")
with col2:
    st.metric("Average Inventory", f"{avg_inventory:.0f} units")
with col3:
    st.metric("Average Lead Time", f"{avg_lead:.1f} days")
with col4:
    st.metric("Total Cost", f"₹{total_cost:,.0f}")

# ---------------- CLEAR WRITTEN INSIGHTS (Format A - simple bullets) ----------------
st.header("📝 Clear written insights")
insights = []

# Sales trend insight (MoM)
try:
    sales_ts = filtered_df.set_index("Date").resample("M")["Sales"].sum().sort_index()
    if len(sales_ts) >= 2:
        last = sales_ts.iloc[-1]
        prev = sales_ts.iloc[-2]
        pct = ((last - prev) / prev * 100) if prev != 0 else np.nan
        if not np.isnan(pct):
            insights.append(f"• Sales change (last month vs previous): {pct:.1f}% ({int(prev):,} → {int(last):,}).")
except Exception:
    pass

# Top category
if has_category and has_sales:
    try:
        top_cat = filtered_df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        if not top_cat.empty:
            name = top_cat.index[0]
            val = top_cat.iloc[0]
            pct_share = val / top_cat.sum() * 100 if top_cat.sum() != 0 else 0
            insights.append(f"• Top category: {name} (₹{int(val):,}, {pct_share:.1f}% of filtered sales).")
    except Exception:
        pass

# Top product
if has_product and has_sales:
    try:
        top_prod = filtered_df.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(3)
        if not top_prod.empty:
            items = ", ".join([f"{p} (₹{int(v):,})" for p, v in top_prod.items()])
            insights.append(f"• Top products: {items}.")
    except Exception:
        pass

# Low inventory count
if has_inventory:
    try:
        low_inv_count = (filtered_df.groupby("Product")["Inventory"].mean() < 20).sum()
        insights.append(f"• Products with average inventory <20 units: {int(low_inv_count)}.")
    except Exception:
        pass

# Supplier share
if supplier_column and has_sales:
    try:
        sup_sales = filtered_df.groupby(supplier_column)["Sales"].sum().sort_values(ascending=False)
        if not sup_sales.empty:
            sup_top = sup_sales.index[0]
            sup_pct = sup_sales.iloc[0] / sup_sales.sum() * 100 if sup_sales.sum() != 0 else 0
            insights.append(f"• Top supplier: {sup_top} contributing {sup_pct:.1f}% of filtered sales.")
    except Exception:
        pass

# Forecast quick insight (rolling mean)
try:
    if has_sales:
        rolling = filtered_df.set_index("Date").resample("W")["Sales"].sum().rolling(4, min_periods=1).mean()
        insights.append(f"• Recent weekly average sales (last point): ₹{int(rolling.dropna().iloc[-1]) if len(rolling.dropna()) else 0:,}.")
except Exception:
    pass

# Correlation quick insight
try:
    numeric_cols = [c for c in ["Sales", "Inventory", "Lead_Time_Days", "Cost"] if c in filtered_df.columns]
    if len(numeric_cols) >= 2:
        corr = filtered_df[numeric_cols].corr()
        # report one strong correlation if exists
        strong_pairs = []
        for a in corr.columns:
            for b in corr.columns:
                if a != b and abs(corr.loc[a, b]) >= 0.6:
                    strong_pairs.append((a, b, corr.loc[a, b]))
        if strong_pairs:
            a, b, val = strong_pairs[0]
            insights.append(f"• Strong correlation: {a} vs {b} ({val:.2f}).")
except Exception:
    pass

# show insights bullets
if len(insights):
    for s in insights:
        st.write(s)
else:
    st.write("• No clear insights could be generated for the selected filters.")

# ---------------- VISUAL ANALYTICS (Mixed pack) ----------------
st.markdown("---")
st.header("📊 Visual Analytics — Mixed Chart Pack")

# 1) Treemap: Category -> Product (Sales)
if has_category and has_product and has_sales:
    try:
        treemap_df = filtered_df.groupby(["Category", "Product"])["Sales"].sum().reset_index()
        fig_treemap = px.treemap(treemap_df, path=["Category", "Product"], values="Sales",
                                title="Treemap — Sales by Category and Product")
        st.plotly_chart(fig_treemap, use_container_width=True)
    except Exception:
        st.info("Treemap can't be generated for this filter selection.")

# 2) Sunburst: Supplier -> Category -> Sales (only if supplier exists)
if supplier_column and has_sales:
    try:
        sun_df = filtered_df.groupby([supplier_column, "Category"])["Sales"].sum().reset_index()
        fig_sun = px.sunburst(sun_df, path=[supplier_column, "Category"], values="Sales",
                              title="Sunburst — Supplier -> Category Sales")
        st.plotly_chart(fig_sun, use_container_width=True)
    except Exception:
        st.info("Sunburst can't be generated for this filter selection or supplier data missing.")

# 3) Area chart: cumulative sales over time
if has_sales:
    try:
        area_ts = filtered_df.groupby("Date")["Sales"].sum().reset_index().sort_values("Date")
        area_ts["Cumulative"] = area_ts["Sales"].cumsum()
        fig_area = px.area(area_ts, x="Date", y="Cumulative", title="Cumulative Sales Over Time (Area)")
        st.plotly_chart(fig_area, use_container_width=True)
    except Exception:
        pass

# 4) Heatmap of correlations between numeric columns
numeric_cols = [c for c in ["Sales", "Inventory", "Lead_Time_Days", "Cost"] if c in filtered_df.columns]
if len(numeric_cols) >= 2:
    try:
        corr = filtered_df[numeric_cols].corr()
        fig_heat = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0
        ))
        fig_heat.update_layout(title="Correlation Heatmap", template="plotly_white", height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
    except Exception:
        pass

# 5) Bubble chart: Cost vs Sales sized by Inventory
if has_cost and has_sales and has_inventory:
    try:
        bubble_df = filtered_df.groupby("Product").agg({"Sales": "sum", "Cost": "mean", "Inventory": "mean"}).reset_index()
        fig_bubble = px.scatter(bubble_df, x="Cost", y="Sales", size="Inventory", hover_name="Product",
                                title="Bubble Chart — Cost vs Sales (size = avg inventory)")
        st.plotly_chart(fig_bubble, use_container_width=True)
    except Exception:
        pass

# 6) Donut chart: category share (donut = pie with hole)
if has_category and has_sales:
    try:
        cat_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()
        fig_donut = px.pie(cat_sales, names="Category", values="Sales", hole=0.45, title="Donut — Category Share")
        st.plotly_chart(fig_donut, use_container_width=True)
    except Exception:
        pass

# 7) Stacked bar: monthly sales by category
if has_category and has_sales:
    try:
        monthly_cat = filtered_df.copy()
        monthly_cat["Month"] = monthly_cat["Date"].dt.to_period("M").dt.to_timestamp()
        monthly_group = monthly_cat.groupby(["Month", "Category"])["Sales"].sum().reset_index()
        fig_stacked = px.bar(monthly_group, x="Month", y="Sales", color="Category", title="Monthly Sales by Category (Stacked)")
        st.plotly_chart(fig_stacked, use_container_width=True)
    except Exception:
        pass

# 8) Simple sales trend line (kept as a small chart)
try:
    sales_trend = filtered_df.groupby("Date")["Sales"].sum().reset_index().sort_values("Date")
    fig_sales = px.line(sales_trend, x="Date", y="Sales", markers=True, title="Sales Trend (line)")
    st.plotly_chart(fig_sales, use_container_width=True)
except Exception:
    pass

# ---------------- FUTURE ANALYTICS (forecast) ----------------
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
        X = ts["Date"].map(lambda d: d.toordinal()).values.reshape(-1, 1)
        y = ts["Sales"].values
        lr = LinearRegression().fit(X, y)
        future_dates = [ts["Date"].max() + pd.DateOffset(months=i) for i in range(1, 7)]
        Xf = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        preds = lr.predict(Xf)
        pred_df = pd.DataFrame({"Date": future_dates, "Predicted_Sales": preds})
        combined = pd.concat([
            ts.rename(columns={"Sales": "Value"}).assign(Type="Actual"),
            pred_df.rename(columns={"Predicted_Sales": "Value"}).assign(Type="Predicted")
        ])
        fig_future = px.line(combined, x="Date", y="Value", color="Type", markers=True, title="Actual vs Predicted Sales (Linear)")
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
        fig_boom = px.bar(boom_df.head(15), x="Product", y="Predicted_Sales", color="Growth_%", title="Product Boom Predictions")
        st.plotly_chart(fig_boom, use_container_width=True)
        top = boom_df.iloc[0]
        st.success(f"🔥 **{top['Product']}** expected boom: **{int(top['Predicted_Sales'])} units** (+{top['Growth_%']:.1f}%)")
    else:
        st.info("No product-level forecast available (insufficient history).")

# ---------------- INVENTORY OPTIMIZATION ----------------
st.markdown("---")
st.header("📦 Inventory Optimization & Reorder Alerts")

inventory_df = filtered_df.groupby("Product").agg({
    "Sales": "mean" if "Sales" in filtered_df.columns else (lambda s: 0),
    "Inventory": "mean" if "Inventory" in filtered_df.columns else (lambda s: 0),
    "Lead_Time_Days": "mean" if "Lead_Time_Days" in filtered_df.columns else (lambda s: 0)
}).reset_index()

# If Sales or Lead_Time_Days are functions (from above fallback), coerce columns
if callable(inventory_df["Sales"].dtype):
    inventory_df["Sales"] = 0
if callable(inventory_df["Inventory"].dtype):
    inventory_df["Inventory"] = 0
if callable(inventory_df["Lead_Time_Days"].dtype):
    inventory_df["Lead_Time_Days"] = 0

inventory_df["Reorder_Level"] = (inventory_df["Sales"] * (inventory_df["Lead_Time_Days"] / 7)).round().fillna(0)
inventory_df["Status"] = np.where(
    inventory_df["Inventory"] < inventory_df["Reorder_Level"], "⚠️ Low Stock",
    "✅ Sufficient"
)

fig_inv = px.bar(
    inventory_df,
    x="Product",
    y=["Inventory", "Reorder_Level"],
    barmode="group",
    title="Inventory vs Reorder Level"
)
st.plotly_chart(fig_inv, use_container_width=True)

low_stock = inventory_df[inventory_df["Status"] == "⚠️ Low Stock"]
if not low_stock.empty:
    st.warning("⚠️ Low stock detected:")
    low_stock = low_stock.copy()
    low_stock["Suggested_Reorder_Qty"] = (low_stock["Reorder_Level"] * 1.5 - low_stock["Inventory"]).clip(lower=0).astype(int)
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
