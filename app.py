# app.py
# Main Streamlit entry file for the AI Supply Chain Dashboard
# This file assumes the following sibling modules exist:
#  - models.py
#  - supplier_optimizer.py
#  - supplier_ranking.py
#  - charts.py
#  - utils.py
# Place this file in the same folder as 'supply_chain_clean_deploy_ready.csv'

import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Supply Chain Dashboard — Modular", layout="wide")

st.title("📊 Fashion Supply Chain Management — Modular AI Dashboard")
st.markdown("""
A modular Streamlit app. This `app.py` orchestrates the UI and calls
functions defined in `models.py`, `supplier_optimizer.py`, `supplier_ranking.py`,
`charts.py` and `utils.py`.

If you haven't generated the modules yet, you'll see a helpful message below.
""")

# -------------------------------
# Import modular components (graceful fallback)
# -------------------------------
try:
    from utils import load_and_prepare_data, safe_col, normalize
except Exception as e:
    st.error("Module `utils.py` not found or failed to import. Please generate utils.py (choose option A).\n" + str(e))
    st.stop()

# try importing other modules but allow app to still show meaningful message
missing_modules = []
try:
    import models
except Exception:
    models = None
    missing_modules.append("models.py")

try:
    import supplier_optimizer
except Exception:
    supplier_optimizer = None
    missing_modules.append("supplier_optimizer.py")

try:
    import supplier_ranking
except Exception:
    supplier_ranking = None
    missing_modules.append("supplier_ranking.py")

try:
    import charts
except Exception:
    charts = None
    missing_modules.append("charts.py")

# -------------------------------
# Data loader
# -------------------------------
DATA_PATH = "supply_chain_clean_deploy_ready.csv"

@st.cache_data
def load_data(path=DATA_PATH):
    df = load_and_prepare_data(path)
    return df

with st.spinner("Loading dataset..."):
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"Dataset not found at '{DATA_PATH}'. Please upload or place the CSV in the app folder.")
        st.stop()
    except Exception as e:
        st.error("Failed to load dataset: " + str(e))
        st.stop()

# -------------------------------
# Sidebar filters (centralized)
# -------------------------------
st.sidebar.header("Filters & Settings")

# date range
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
start_date, end_date = st.sidebar.date_input("Date range", [min_date, max_date])
if isinstance(start_date, (list, tuple)) and len(start_date) == 2:
    start_date, end_date = start_date

# category/product/supplier multi-selects
categories = df["Category"].dropna().unique().tolist()
products = df["Product"].dropna().unique().tolist()

selected_categories = st.sidebar.multiselect("Category", categories, default=categories)
selected_products = st.sidebar.multiselect("Product", products, default=products)

supplier_col = safe_col(df, ["supplier", "suppliers", "carrier", "vendor"], default=None)
if supplier_col is None:
    supplier_col = "Supplier_Fallback"

suppliers = df[supplier_col].dropna().unique().tolist()
selected_suppliers = st.sidebar.multiselect("Supplier", suppliers, default=suppliers)

# ML settings (expose simple controls)
st.sidebar.markdown("---")
st.sidebar.markdown("### ML & Optimizer Settings")
rf_estimators = st.sidebar.slider("RF trees (estimators)", 100, 800, 400)
mix_weight_cost = st.sidebar.slider("Supplier Mix Weight: Cost", 0.0, 1.0, 0.5)

# -------------------------------
# Apply filters
# -------------------------------
mask = (
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date)) &
    (df["Category"].isin(selected_categories)) &
    (df["Product"].isin(selected_products)) &
    (df[supplier_col].isin(selected_suppliers))
)
filtered = df[mask].copy()

if filtered.empty:
    st.warning("No data matches the selected filters. Adjust filters to proceed.")
    st.stop()

# -------------------------------
# Top-level dashboard sections
# -------------------------------
st.header("Overview & KPIs")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"{int(filtered['Sales'].sum()):,}")
col2.metric("Avg Inventory", f"{filtered['Inventory'].mean():.0f}")
col3.metric("Avg Lead Time (days)", f"{filtered['Lead_Time_Days'].mean():.1f}")
col4.metric("Total Cost (₹)", f"{filtered['Cost'].sum():,.0f}")

st.markdown("---")

# Charts (charts module)
if charts is None:
    st.info("charts.py missing — basic visuals shown. Generate charts.py to see improved visuals.")
    st.subheader("Basic Sales Trend")
    st.line_chart(filtered.groupby(pd.Grouper(key='Date', freq='M'))['Sales'].sum())
else:
    charts.plot_sales_trend(filtered)
    charts.plot_inventory_by_product(filtered)
    charts.plot_supplier_pie(filtered, supplier_col)

# ML models (models module)
st.markdown("---")
st.header("Forecasting & ML Models")
if models is None:
    st.info("models.py missing — forecasting modules unavailable. Generate models.py to enable ML features.")
else:
    with st.expander("Sales Forecast (RandomForest)"):
        models.run_sales_forecast(filtered, n_estimators=rf_estimators)

    with st.expander("Product Boom Predictor"):
        models.run_product_boom(filtered)

    with st.expander("Inventory Forecast"):
        models.run_inventory_forecast(filtered)

# Supplier Ranking
st.markdown("---")
if supplier_ranking is None:
    st.info("supplier_ranking.py missing — supplier ranking tools are unavailable.")
else:
    supplier_ranking.show_supplier_ranking(filtered, supplier_col)

# Supplier AI Optimizer
st.markdown("---")
if supplier_optimizer is None:
    st.info("supplier_optimizer.py missing — supplier optimizer unavailable.")
else:
    supplier_optimizer.show_optimizer_ui(filtered, supplier_col, rf_estimators, mix_weight_cost)

# Data download
st.markdown("---")
st.header("Data")
st.dataframe(filtered)
csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Filtered Data", csv, "filtered_supply_data.csv", "text/csv")

st.markdown("---")
st.write("Tip: generate the other module files (models.py, charts.py, supplier_optimizer.py, supplier_ranking.py, utils.py) in the order you prefer. I will create them one-by-one.")
