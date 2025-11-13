# app.py
# Updated Streamlit app: adds clear written insights and improves chart quality (plotly)
# Replace your current app.py with this file and run `streamlit run app.py`

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
        # generate small synthetic example so app still works without upload
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
            "Inventory": (rng.integers(0, 500, size=n)).astype(int),
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
        # normalize column names
        df.columns = [c.strip() for c in df.columns]
        # basic required columns check
        required = ["Date", "Product", "Category", "Sales", "Inventory", "Lead_Time_Days", "Cost"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.warning(f"Missing columns in uploaded file: {missing}. The synthetic example will be used instead.")
            return load_data(None)
        # cast types
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        numeric_cols = ["Sales", "Inventory", "Lead_Time_Days", "Cost"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df


def generate_insights(df):
    insights = []
    # Overall KPIs
    total_sales = df['Sales'].sum()
    avg_order_value = df['Sales'].mean() if len(df) else 0
    median_lead = df['Lead_Time_Days'].median() if len(df) else 0
    avg_inventory = df['Inventory'].mean() if len(df) else 0
    insights.append(f"Total sales (sum): ₹{total_sales:,.2f} across {df['Product'].nunique()} unique products.")
    insights.append(f"Average sales per record (approx. AOV): ₹{avg_order_value:,.2f}.")
    insights.append(f"Median lead time: {median_lead} days. Average inventory level: {avg_inventory:,.1f} units.")

    # Top categories and products
    top_cat = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
    if not top_cat.empty:
        top_cat_name = top_cat.index[0]
        top_cat_pct = 100 * top_cat.iloc[0] / top_cat.sum()
        insights.append(f"Top category by sales: {top_cat_name} contributing {top_cat_pct:.1f}% of category sales.")

    top_products = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(5)
    if not top_products.empty:
        p_list = ", ".join([f"{p} (₹{s:,.0f})" for p, s in top_products.items()])
        insights.append(f"Top 5 products by sales: {p_list}.")

    # Inventory concerns
    low_stock = df.groupby('Product')['Inventory'].mean().sort_values().head(5)
    low_stock = low_stock[low_stock < 20]
    if not low_stock.empty:
        ls = ", ".join([f"{p} ({int(q)} units avg)" for p, q in low_stock.items()])
        insights.append(f"Products with low average inventory (<20 units): {ls}. Consider restocking soon.")

    # Correlations
    corr = df[['Sales', 'Inventory', 'Lead_Time_Days', 'Cost']].corr()
    strong_corrs = []
    for a in corr.columns:
        for b in corr.columns:
            if a == b: continue
            val = corr.loc[a,b]
            if abs(val) >= 0.6:
                strong_corrs.append(f"{a} vs {b}: {val:.2f}")
    if strong_corrs:
        insights.append("Strong correlations detected: " + "; ".join(strong_corrs) + \".\")

    # Trend insight (monthly)
    if 'Date' in df.columns:
        ts = df.copy()
        ts['Date'] = pd.to_datetime(ts['Date'])
        ts = ts.set_index('Date').resample('M').sum()
        if len(ts) >= 2:
            last = ts['Sales'].iloc[-1]
            prev = ts['Sales'].iloc[-2]
            pct = (last - prev) / prev * 100 if prev != 0 else np.nan
            if not np.isnan(pct):
                if pct > 5:
                    insights.append(f"Sales increased by {pct:.1f}% month-over-month (last month vs previous). Positive growth.")
                elif pct < -5:
                    insights.append(f"Sales decreased by {abs(pct):.1f}% month-over-month (last month vs previous). Investigate causes.")

    return insights


# ---------- UI ----------
st.title("📈 Fashion Supply Chain Management — Insights & Improved Charts")
st.markdown("Upload your dataset (CSV) with columns: Date, Product, Category, Sales, Inventory, Lead_Time_Days, Cost.")

uploaded = st.file_uploader("Upload CSV or leave empty to use a demo dataset", type=['csv'])

df = load_data(uploaded)

# Top-level KPIs
st.sidebar.header("Filters & Controls")
with st.sidebar.form("controls"):
    category_filter = st.multiselect("Category", options=sorted(df['Category'].unique()), default=sorted(df['Category'].unique()))
    date_min = st.date_input("From date", value=df['Date'].min())
    date_max = st.date_input("To date", value=df['Date'].max())
    reorder_threshold = st.number_input("Low inventory threshold (units)", min_value=0, value=20)
    submitted = st.form_submit_button("Apply")

# filter dataframe
mask = (df['Category'].isin(category_filter)) & (pd.to_datetime(df['Date']) >= pd.to_datetime(date_min)) & (pd.to_datetime(df['Date']) <= pd.to_datetime(date_max))
filtered = df.loc[mask].copy()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Sales (₹)", f"{filtered['Sales'].sum():,.0f}")
with col2:
    st.metric("Unique Products", f"{filtered['Product'].nunique()}")
with col3:
    st.metric("Avg Lead Time (days)", f"{filtered['Lead_Time_Days'].median():.1f}")
with col4:
    st.metric("Avg Inventory", f"{filtered['Inventory'].mean():.1f}")

# Insights box
st.header("Clear written insights")
insights = generate_insights(filtered)
for i, ins in enumerate(insights, 1):
    st.write(f"**{i}.** {ins}")

# Charts - improved quality using Plotly and layout tweaks
st.header("Visual analysis")

# Sales over time
with st.expander("Sales over time (monthly)", expanded=True):
    if 'Date' in filtered.columns:
        ts = filtered.copy()
        ts['Date'] = pd.to_datetime(ts['Date'])
        monthly = ts.set_index('Date').resample('M')['Sales'].sum().reset_index()
        fig = px.line(monthly, x='Date', y='Sales', markers=True, title='Monthly Sales')
        fig.update_traces(marker_size=8, line_width=3)
        fig.update_layout(template='plotly_white', height=420, margin=dict(l=40,r=20,t=60,b=40), font=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)
        # quick text summary
        st.markdown("**Trend summary:** The chart above shows monthly aggregated sales. Check for seasonality or sudden drops/spikes.")

# Top categories pie
with st.expander("Sales by Category", expanded=False):
    cat_sales = filtered.groupby('Category')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
    fig2 = px.pie(cat_sales, names='Category', values='Sales', title='Sales share by Category', hole=0.4)
    fig2.update_layout(template='plotly_white', height=380, margin=dict(t=50,l=20,r=20))
    st.plotly_chart(fig2, use_container_width=True)
    if not cat_sales.empty:
        top = cat_sales.iloc[0]
        st.write(f"Top category: **{top['Category']}** contributing **₹{top['Sales']:,.0f}** ({top['Sales']/cat_sales['Sales'].sum():.1%}).")

# Top products bar
with st.expander("Top products by sales", expanded=False):
    prod = filtered.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False).head(15)
    fig3 = px.bar(prod, x='Sales', y='Product', orientation='h', title='Top products by Sales')
    fig3.update_layout(template='plotly_white', height=520, margin=dict(l=200,t=40), yaxis={'categoryorder':'total ascending'}, font=dict(size=12))
    fig3.update_traces(marker_line_width=0.5)
    st.plotly_chart(fig3, use_container_width=True)

# Inventory vs Sales scatter with size by cost
with st.expander("Inventory vs Sales (product-level)", expanded=False):
    agg = filtered.groupby('Product').agg({'Sales':'sum','Inventory':'mean','Cost':'mean'}).reset_index()
    if not agg.empty:
        fig4 = px.scatter(agg, x='Inventory', y='Sales', size='Cost', hover_name='Product', title='Inventory vs Sales (bubble = avg cost)')
        fig4.update_layout(template='plotly_white', height=480, margin=dict(t=50))
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("**Insight tip:** Products far to the right with low sales are overstocked candidates; far left with high sales may need restock prioritization.")

# Correlation heatmap
with st.expander("Correlation matrix", expanded=False):
    corr = filtered[['Sales','Inventory','Lead_Time_Days','Cost']].corr()
    fig5 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmid=0))
    fig5.update_layout(title='Correlation matrix', template='plotly_white', height=420)
    st.plotly_chart(fig5, use_container_width=True)

# Inventory alerts
st.header("Inventory alerts & recommendations")
low_stock_df = filtered.groupby('Product')['Inventory'].mean().reset_index().sort_values('Inventory')
alerts = low_stock_df[low_stock_df['Inventory'] <= reorder_threshold]
if not alerts.empty:
    st.warning(f"{len(alerts)} products below the reorder threshold ({reorder_threshold} units).")
    st.dataframe(alerts.rename(columns={'Inventory':'Avg_Inventory'}).head(50))
else:
    st.success("No products below the reorder threshold in the selected filters.")

# Download filtered data option
st.markdown("---")
st.download_button("Download filtered data (CSV)", data=filtered.to_csv(index=False).encode('utf-8'), file_name='filtered_data.csv', mime='text/csv')

# Footer & tips
st.markdown("---")
st.markdown(
    "**What I changed / why this helps:**\n"
    "1. Added a concise written insights section generated from your data so non-technical stakeholders can read quick takeaways.\n"
    "2. Upgraded charts to Plotly for crisper, interactive visuals and tuned layout/font sizes for better readability.\n"
    "3. Added inventory alerts and a download button for easy operational use.\n"
    "4. Included a small demo dataset when no file is uploaded so the dashboard remains functional.\n"
)

st.markdown("If you'd like, I can further tailor the insights (for example: margin analysis, seasonality decomposition, or reorder point calculations). Tell me which analysis you want next.")
