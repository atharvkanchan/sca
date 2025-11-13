# app.py
# Final Streamlit Dashboard (Zero Errors) with Labeled Insights & Professional Charts

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Fashion Supply Chain Dashboard — Enhanced", layout="wide")

# ===================================================================
# DATA LOADER
# ===================================================================
@st.cache_data
def load_data(uploaded_file):
    """Load CSV file or fallback to a synthetic demo dataset."""
    if uploaded_file is None:
        rng = np.random.default_rng(42)
        n = 300
        df = pd.DataFrame({
            "Date": pd.date_range(end=pd.Timestamp.today(), periods=n),
            "Product": rng.choice([f"Product_{i}" for i in range(1, 21)], size=n),
            "Category": rng.choice(["Apparel", "Footwear", "Accessories", "Home"], size=n),
            "Sales": (rng.random(n) * 500).round(2),
            "Inventory": rng.integers(10, 500, size=n),
            "Lead_Time_Days": rng.integers(1, 30, size=n),
            "Cost": (rng.random(n) * 200).round(2)
        })
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        return df

    try:
        df = pd.read_csv(uploaded_file)
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin1")

    df.columns = [c.strip() for c in df.columns]

    # Required columns
    required = ["Date", "Product", "Category", "Sales", "Inventory", "Lead_Time_Days", "Cost"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        st.warning(f"Missing columns: {missing}. Loading demo dataset instead.")
        return load_data(None)

    # Clean data types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df.dropna(subset=["Date"])

    for c in ["Sales", "Inventory", "Lead_Time_Days", "Cost"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df

# ===================================================================
# INSIGHT GENERATOR (LABELED INSIGHTS)
# ===================================================================
def generate_insights(df):
    insights = []

    if df is None or len(df) == 0:
        return ["No data available to generate insights."]

    # Sales Insight
    total_sales = df["Sales"].sum()
    avg_sale = df["Sales"].mean()
    insights.append(f"**Sales Insight:** Total sales: ₹{total_sales:,.2f}. Average sale value: ₹{avg_sale:,.2f}.")

    # Category Insight
    top_cat = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    if len(top_cat):
        insights.append(f"**Category Insight:** Top-performing category: {top_cat.index[0]} with ₹{top_cat.iloc[0]:,.0f} sales.")

    # Product Insight
    top_products = df.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(5)
    if len(top_products):
        p_list = ", ".join([f"{p} (₹{v:,.0f})" for p, v in top_products.items()])
        insights.append(f"**Product Insight:** Best-selling products: {p_list}.")

    # Inventory Insight
    low_stock = df.groupby("Product")["Inventory"].mean().sort_values().head(10)
    low_stock_small = low_stock[low_stock < 20]
    if len(low_stock_small):
        ls = ", ".join([f"{p} ({int(v)} units)" for p, v in low_stock_small.items()])
        insights.append(f"**Inventory Insight:** Low stock detected for: {ls}. Restocking recommended.")

    # Correlation Insight
    corr = df[["Sales", "Inventory", "Lead_Time_Days", "Cost"]].corr()
    strong = []
    for a in corr.columns:
        for b in corr.columns:
            if a != b and abs(corr.loc[a,b]) >= 0.6:
                strong.append(f"{a} ↔ {b} ({corr.loc[a,b]:.2f})")
    if strong:
        insights.append("**Correlation Insight:** Strong relationships found — " + ", ".join(strong))

    # Trend Insight
    ts = df.copy()
    ts["Date"] = pd.to_datetime(ts["Date"])
    monthly = ts.set_index("Date").resample("M")["Sales"].sum()
    if len(monthly) >= 2:
        last, prev = monthly.iloc[-1], monthly.iloc[-2]
        if prev != 0:
            pct = (last - prev) / prev * 100
            if pct > 5:
                insights.append(f"**Trend Insight:** Sales increased by {pct:.1f}% last month.")
            elif pct < -5:
                insights.append(f"**Trend Insight:** Sales decreased by {abs(pct):.1f}% last month.")

    return insights


# ===================================================================
# UI START
# ===================================================================
st.title("📊 Fashion Supply Chain Management — Professional Dashboard")
st.markdown("Upload your CSV or use the built-in demo dataset.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
df = load_data(uploaded)

# ===================================================================
# SIDEBAR FILTERS
# ===================================================================
st.sidebar.header("Filters")
with st.sidebar.form("filters"):
    categories = sorted(df["Category"].unique())
    category_filter = st.multiselect("Category", categories, default=categories)

    date_min = pd.to_datetime(df["Date"]).min().date()
    date_max = pd.to_datetime(df["Date"]).max().date()
    from_date = st.date_input("From Date", date_min)
    to_date = st.date_input("To Date", date_max)

    reorder_threshold = st.number_input("Low Inventory Threshold", min_value=0, value=20)
    st.form_submit_button("Apply")

mask = (
    df["Category"].isin(category_filter) &
    (pd.to_datetime(df["Date"]) >= pd.to_datetime(from_date)) &
    (pd.to_datetime(df["Date"]) <= pd.to_datetime(to_date))
)

filtered = df.loc[mask].copy()

# ===================================================================
# KPIs
# ===================================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sales", f"₹{filtered['Sales'].sum():,.0f}")
c2.metric("Products", filtered["Product"].nunique())
c3.metric("Median Lead Time", f"{filtered['Lead_Time_Days'].median():.1f} days")
c4.metric("Avg Inventory", f"{filtered['Inventory'].mean():.1f}")

# ===================================================================
# INSIGHTS SECTION
# ===================================================================
st.header("🧠 Labeled Insights")
for i, ins in enumerate(generate_insights(filtered), 1):
    st.write(f"{i}. {ins}")

# ===================================================================
# CHARTS
# ===================================================================
st.header("📈 Visual Analytics")

# Monthly Sales
with st.expander("📅 Monthly Sales Trend", expanded=True):
    ts = filtered.copy()
    ts["Date"] = pd.to_datetime(ts["Date"])
    monthly = ts.set_index("Date").resample("M")["Sales"].sum().reset_index()
    fig = px.line(monthly, x="Date", y="Sales", markers=True, title="Monthly Sales Trend")
    fig.update_traces(marker_size=8, line_width=3)
    fig.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)

# Category Sales
with st.expander("📦 Sales by Category"):
    cat_sales = filtered.groupby("Category")["Sales"].sum().reset_index()
    fig2 = px.pie(cat_sales, names="Category", values="Sales", hole=0.45, title="Category-wise Sales Share")
    fig2.update_layout(template="plotly_white", height=380)
    st.plotly_chart(fig2, use_container_width=True)

# Top Products
with st.expander("🏆 Top Products"):
    pr = filtered.groupby("Product")["Sales"].sum().reset_index().sort_values("Sales", ascending=False).head(10)
    fig3 = px.bar(pr, x="Sales", y="Product", orientation="h", title="Top Selling Products")
    fig3.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig3, use_container_width=True)

# Inventory vs Sales
with st.expander("📉 Inventory vs Sales Scatter"):
    agg = filtered.groupby("Product").agg({"Sales": "sum", "Inventory": "mean", "Cost": "mean"}).reset_index()
    fig4 = px.scatter(agg, x="Inventory", y="Sales", size="Cost", hover_name="Product", title="Inventory vs Sales (Bubble = Avg Cost)")
    fig4.update_layout(template="plotly_white", height=460)
    st.plotly_chart(fig4, use_container_width=True)

# Correlation Matrix
with st.expander("🔗 Correlation Heatmap"):
    corr = filtered[["Sales", "Inventory", "Lead_Time_Days", "Cost"]].corr()
    fig5 = go.Figure(
        data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0)
    )
    fig5.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig5, use_container_width=True)

# ===================================================================
# INVENTORY ALERTS
# ===================================================================
st.header("🚨 Inventory Alerts")
low_stk = filtered.groupby("Product")["Inventory"].mean().reset_index()
alerts = low_stk[low_stk["Inventory"] <= reorder_threshold]

if len(alerts):
    st.warning(f"{len(alerts)} products below threshold ({reorder_threshold} units)")
    st.dataframe(alerts.rename(columns={"Inventory": "Avg Inventory"}))
else:
    st.success("All product inventory levels are healthy.")

# ===================================================================
# DOWNLOAD BUTTON
# ===================================================================
st.download_button(
    "Download Filtered Data",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_supply_chain_data.csv",
    mime="text/csv"
)
