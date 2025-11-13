# app.py
# Updated Streamlit app: adds clear written insights and improves chart quality (plotly)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Fashion Supply Chain Management — Enhanced", layout="wide")

# ---------- Helpers ----------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        rng = np.random.default_rng(42)
        n = 300
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n)
        products = [f"Product_{i}" for i in range(1, 21)]
        cats = ["Apparel", "Footwear", "Accessories", "Home"]

        df = pd.DataFrame({
            "Date": rng.choice(dates, size=n),
            "Product": rng.choice(products, size=n),
            "Category": rng.choice(cats, size=n),
            "Sales": (rng.random(n) * 500).round(2),
            "Inventory": rng.integers(0, 500, size=n),
            "Lead_Time_Days": rng.integers(1, 30, size=n),
            "Cost": (rng.random(n) * 200).round(2)
        })

        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        return df
    
    else:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin1')

        df.columns = [c.strip() for c in df.columns]

        required = ["Date", "Product", "Category", "Sales", "Inventory", "Lead_Time_Days", "Cost"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            st.warning(f"Missing columns in uploaded file: {missing}. Using demo dataset instead.")
            return load_data(None)

        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        for c in ["Sales", "Inventory", "Lead_Time_Days", "Cost"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        return df


def generate_insights(df):
    insights = []

    total_sales = df['Sales'].sum()
    avg_order_value = df['Sales'].mean()
    median_lead = df['Lead_Time_Days'].median()
    avg_inventory = df['Inventory'].mean()

    insights.append(f"Total sales: ₹{total_sales:,.2f} across {df['Product'].nunique()} products.")
    insights.append(f"Average sales per record: ₹{avg_order_value:,.2f}.")
    insights.append(f"Median lead time: {median_lead} days. Avg inventory: {avg_inventory:.1f} units.")

    # Top category
    top_cat = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
    if not top_cat.empty:
        tc = top_cat.index[0]
        pct = top_cat.iloc[0] / top_cat.sum() * 100
        insights.append(f"Top category: {tc} ({pct:.1f}% of total sales).")

    # Top products
    top_products = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(5)
    if not top_products.empty:
        prod_list = ", ".join([f"{p} (₹{v:,.0f})" for p, v in top_products.items()])
        insights.append(f"Top 5 products: {prod_list}.")

    # Low stock
    low_stock = df.groupby('Product')['Inventory'].mean().sort_values()
    low_stock = low_stock[low_stock < 20].head(5)
    if not low_stock.empty:
        ls = ", ".join([f"{p} ({int(v)} units)" for p, v in low_stock.items()])
        insights.append(f"Low stock products (<20 units): {ls}.")

    # Correlations
    corr = df[['Sales', 'Inventory', 'Lead_Time_Days', 'Cost']].corr()
    strong_corrs = []
    for a in corr.columns:
        for b in corr.columns:
            if a != b:
                if abs(corr.loc[a,b]) >= 0.6:
                    strong_corrs.append(f"{a} vs {b}: {corr.loc[a,b]:.2f}")

    if strong_corrs:
        insights.append("Strong correlations detected: " + "; ".join(strong_corrs) + ".")

    # Trend insight
    ts = df.copy()
    ts['Date'] = pd.to_datetime(ts['Date'])
    ts = ts.set_index('Date').resample('M').sum()

    if len(ts) >= 2:
        last, prev = ts['Sales'].iloc[-1], ts['Sales'].iloc[-2]
        if prev != 0:
            pct = (last - prev) / prev * 100
            if pct > 5:
                insights.append(f"Sales increased by {pct:.1f}% last month.")
            elif pct < -5:
                insights.append(f"Sales decreased by {abs(pct):.1f}% last month.")

    return insights


# ---------- UI ----------
st.title("📈 Fashion Supply Chain Management — Insights & Improved Charts")
st.markdown("Upload a CSV with: Date, Product, Category, Sales, Inventory, Lead_Time_Days, Cost")

uploaded = st.file_uploader("Upload CSV", type=['csv'])
df = load_data(uploaded)

# Sidebar filters
st.sidebar.header("Filters")
with st.sidebar.form("filters"):
    category_filter = st.multiselect("Category", df['Category'].unique(), default=list(df['Category'].unique()))
    date_min = st.date_input("Start Date", df['Date'].min())
    date_max = st.date_input("End Date", df['Date'].max())
    reorder_threshold = st.number_input("Reorder threshold", min_value=0, value=20)
    submitted = st.form_submit_button("Apply filters")

mask = (
    df['Category'].isin(category_filter) &
    (pd.to_datetime(df['Date']) >= pd.to_datetime(date_min)) &
    (pd.to_datetime(df['Date']) <= pd.to_datetime(date_max))
)

filtered = df.loc[mask]

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sales", f"₹{filtered['Sales'].sum():,.0f}")
c2.metric("Products", filtered['Product'].nunique())
c3.metric("Median Lead Time", f"{filtered['Lead_Time_Days'].median():.1f} days")
c4.metric("Avg Inventory", f"{filtered['Inventory'].mean():.1f}")

# Insights
st.header("📝 Clear Insights")
for i, x in enumerate(generate_insights(filtered), 1):
    st.write(f"**{i}.** {x}")

# Charts
st.header("📊 Visual Analysis")

# Monthly Sales
with st.expander("Monthly Sales Trend", expanded=True):
    tmp = filtered.copy()
    tmp['Date'] = pd.to_datetime(tmp['Date'])
    monthly = tmp.set_index('Date').resample('M')['Sales'].sum().reset_index()
    fig = px.line(monthly, x='Date', y='Sales', markers=True, title="Monthly Sales")
    fig.update_traces(marker_size=8, line_width=3)
    fig.update_layout(template='plotly_white', height=420)
    st.plotly_chart(fig, use_container_width=True)

# Category Pie Chart
with st.expander("Sales by Category"):
    cat_sales = filtered.groupby('Category')['Sales'].sum().reset_index()
    fig = px.pie(cat_sales, names='Category', values='Sales', hole=0.4)
    fig.update_layout(template='plotly_white', height=380)
    st.plotly_chart(fig, use_container_width=True)

# Top Products
with st.expander("Top Products"):
    prod = filtered.groupby('Product')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False).head(15)
    fig = px.bar(prod, x='Sales', y='Product', orientation='h', title="Top Products")
    fig.update_layout(template='plotly_white', height=520)
    st.plotly_chart(fig, use_container_width=True)

# Scatter
with st.expander("Inventory vs Sales"):
    agg = filtered.groupby('Product').agg({'Sales':'sum','Inventory':'mean','Cost':'mean'}).reset_index()
    fig = px.scatter(agg, x='Inventory', y='Sales', size='Cost', hover_name='Product')
    fig.update_layout(template='plotly_white', height=480)
    st.plotly_chart(fig, use_container_width=True)

# Heatmap
with st.expander("Correlation Matrix"):
    corr = filtered[['Sales','Inventory','Lead_Time_Days','Cost']].corr()
    fig = go.Figure(data=go.Heatmap(z=corr, x=corr.columns, y=corr.columns, colorscale='RdBu', zmid=0))
    fig.update_layout(height=420, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# Inventory Alerts
st.header("⚠ Inventory Alerts")
low = filtered.groupby('Product')['Inventory'].mean().reset_index()
alerts = low[low['Inventory'] <= reorder_threshold]

if len(alerts):
    st.warning(f"{len(alerts)} products below threshold
