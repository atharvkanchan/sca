# app.py
# Rebuilt Enterprise Supply Chain Dashboard (All-in-One)
# - Stable LR forecasting (month index), safer Prophet plotting
# - Form-based filters, caching, session_state for uploads
# - Precomputed aggregates to avoid repeated groupbys
# - Explainable insights under each chart (rule-based)
# - Defensive checks, logging

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("supply_chain_dashboard")

# -----------------------
# Streamlit page config
# -----------------------
st.set_page_config(page_title="Enterprise Fashion Supply Chain Dashboard", layout="wide")
st.title("🏬 Fashion Supply Chain Management Dashboard ")

# -----------------------
# Helper utilities
# -----------------------
COMMON_COLS = {
    "date": ["date", "order_date", "sale_date", "timestamp"],
    "product": ["product", "product_name", "sku", "item", "product_sku"],
    "category": ["category", "cat", "product_category"],
    "sales": ["sales", "revenue", "amount", "total_sales", "net_sales"],
    "inventory": ["inventory", "stock", "on_hand"],
    "lead": ["lead_time", "lead_time_days"],
    "cost": ["cost", "unit_cost", "cost_price"],
    "supplier": ["supplier", "vendor", "vendor_name", "supplier_name"]
}


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


@st.cache_data
def auto_clean_cached(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-clean wrapper cached to avoid repeated work."""
    df = df.copy()
    # Trim spaces
    df.columns = [str(c).strip() for c in df.columns]
    # Strip string columns
    for c in df.select_dtypes(include="object").columns:
        try:
            df[c] = df[c].astype(str).str.strip()
        except Exception:
            pass
    return df


@st.cache_data
def map_columns_cached(df: pd.DataFrame) -> dict:
    return map_columns(df)


def map_columns(df):
    """Return mapping from canonical name to actual column name in df (or None)"""
    cols = {}
    col_lookup = {str(c).lower().strip(): c for c in df.columns}
    for key, candidates in COMMON_COLS.items():
        found = None
        for cand in candidates:
            if cand in col_lookup:
                found = col_lookup[cand]
                break
        # fuzzy fallback: check substrings
        if not found:
            for orig in df.columns:
                low = str(orig).lower()
                for cand in candidates:
                    if cand in low:
                        found = orig
                        break
                if found:
                    break
        cols[key] = found
    return cols


def safe_date_parse(df, date_col):
    df = df.copy()
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    except Exception:
        df[date_col] = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    return df


@st.cache_data
def compute_supplier_risk_cached(df: pd.DataFrame, supplier_col: str, lead_col: str) -> pd.DataFrame:
    return compute_supplier_risk(df, supplier_col, lead_col)


def compute_supplier_risk(df, supplier_col, lead_col):
    """Simple supplier risk: high avg lead time and high variability -> higher risk score
    Defensive: handle missing Sales column gracefully.
    """
    # choose a sales-like column if possible
    sales_col = "Sales" if "Sales" in df.columns else None
    if sales_col is None:
        # fallback: try numeric column other than lead
        numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in [lead_col]]
        sales_col = numeric_cols[0] if numeric_cols else None

    agg_args = {
        'avg_lead': (lead_col, 'mean'),
        'std_lead': (lead_col, 'std')
    }
    if sales_col:
        agg_args['sales_share'] = (sales_col, 'sum')

    s = df.groupby(supplier_col).agg(**agg_args).reset_index()
    s['std_lead'] = s['std_lead'].fillna(0)

    # Normalize scores safely
    lead_ptp = s['avg_lead'].ptp() if s['avg_lead'].ptp() != 0 else 1
    var_ptp = s['std_lead'].ptp() if s['std_lead'].ptp() != 0 else 1
    s['lead_norm'] = (s['avg_lead'] - s['avg_lead'].min()) / lead_ptp
    s['var_norm'] = (s['std_lead'] - s['std_lead'].min()) / var_ptp
    s['risk_score'] = (s['lead_norm'] * 0.6 + s['var_norm'] * 0.4) * 100
    return s.sort_values('risk_score', ascending=False)


# --------------------------------
# Explainers toolkit (chart-level textual insights)
# --------------------------------
def pct_change_text(series):
    """Return last vs previous pct change string, guarding divide-by-zero."""
    if len(series) < 2:
        return "Insufficient history to compute change."
    last = float(series.iloc[-1])
    prev = float(series.iloc[-2]) if len(series) >= 2 else 0.0
    denom = prev if prev != 0 else 1.0
    pct = (last - prev) / denom * 100.0
    sign = "increased" if pct > 0 else "decreased" if pct < 0 else "no change"
    return f"{sign} by {abs(pct):.1f}% vs previous period."


def trend_slope_text(dates, values):
    """Compute simple linear slope on month index and return human text."""
    if len(values) < 3:
        return ""
    # convert dates to month index to avoid numeric instability
    months = (
        (pd.to_datetime(dates).dt.year - pd.to_datetime(dates).dt.year.min()) * 12
        + pd.to_datetime(dates).dt.month
    ).astype(int)
    coef = np.polyfit(months, values, 1)
    slope = coef[0]
    if abs(slope) < 1e-8:
        return "Flat trend overall."
    direction = "upward" if slope > 0 else "downward"
    return f"Longer-term trend is {direction} (slope ≈ {slope:.2f} sales / month)."


def detect_spike(series, z_thresh=2.8):
    """Simple z-score spike detection; returns list of (idx, value)."""
    arr = np.array(series)
    if len(arr) < 5:
        return []
    mu = np.nanmean(arr)
    sigma = np.nanstd(arr) if np.nanstd(arr) != 0 else 1.0
    z = (arr - mu) / sigma
    idxs = np.where(np.abs(z) > z_thresh)[0]
    return list(zip(idxs.tolist(), arr[idxs].tolist()))


def top_contributors_change(prev_df, last_df, top_n=3, value_col="Sales", key_col="Product"):
    """Given two groupby DataFrames (key_col, value_col) for previous and last period,
       returns top gaining and losing contributors."""
    prev = prev_df.set_index(key_col)[value_col] if not prev_df.empty else pd.Series(dtype=float)
    last = last_df.set_index(key_col)[value_col] if not last_df.empty else pd.Series(dtype=float)
    all_keys = sorted(set(prev.index).union(set(last.index)))
    diffs = []
    for k in all_keys:
        diffs.append((k, float(last.get(k, 0.0)) - float(prev.get(k, 0.0))))
    diffs = sorted(diffs, key=lambda x: x[1], reverse=True)
    gaining = diffs[:top_n]
    losing = diffs[-top_n:][::-1]
    return gaining, losing


def format_contribs(contribs):
    """Format list of (key, delta) into readable sentence fragment."""
    parts = []
    for k, d in contribs:
        parts.append(f"{k} ({'+' if d>=0 else ''}{d:,.0f})")
    return ", ".join(parts) if parts else "none notable"


def explain_timeseries(ts_df, date_col="Date", value_col="Sales"):
    """High-level text for a timeseries aggregated DataFrame (Date, Sales)."""
    # ensure sorted
    ts = ts_df.sort_values(date_col).reset_index(drop=True)
    if ts.empty:
        return "No data for this period."
    # pct change last vs prev
    pct_text = pct_change_text(ts[value_col])
    # slope
    slope_text = trend_slope_text(ts[date_col], ts[value_col])
    # spikes
    spikes = detect_spike(ts[value_col])
    spike_text = ""
    if spikes:
        spike_text = f" Notable spikes detected at {len(spikes)} point(s)."
    # seasonality hint: compare last 12 months average vs previous 12 months if available
    season_text = ""
    if len(ts) >= 24:
        last12 = ts[value_col].iloc[-12:].mean()
        prev12 = ts[value_col].iloc[-24:-12].mean()
        diff = (last12 - prev12) / (prev12 if prev12 != 0 else 1) * 100.0
        season_text = f" 12-month avg changed by {diff:.1f}% vs previous 12 months."
    return " ".join([pct_text, slope_text, spike_text, season_text]).strip()


def explain_treemap_or_donut(df, group_col="Category", value_col="Sales", prev_df=None):
    """
    Explain composition: top category and concentration and optionally change vs prev period.
    prev_df (optional) should be a df grouped the same way for previous period.
    """
    s = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
    if s.empty:
        return "No category sales available."
    top = s.index[0]
    pct = s.iloc[0] / s.sum() * 100 if s.sum() != 0 else 0
    text = f"Top: **{top}** ({pct:.1f}% of total)."
    # concentration check
    if pct > 60:
        text += " High concentration — consider diversifying top categories."
    # compare prev if available
    if prev_df is not None:
        prev_s = prev_df.groupby(group_col)[value_col].sum()
        prev_top_val = prev_s.get(top, 0.0)
        delta = s.iloc[0] - prev_top_val
        text += f" {top} changed by {delta:,.0f} vs previous period."
    return text


# -----------------------
# Upload / demo dataset handling (session_state)
# -----------------------
st.sidebar.header("Dataset & Settings")

if 'uploaded_dfs' not in st.session_state:
    st.session_state.uploaded_dfs = []

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSVs (optional)",
    accept_multiple_files=True,
    type=["csv"]
)

# When user uploads, read and append to session_state to avoid re-reading every rerun
if uploaded_files:
    for f in uploaded_files:
        try:
            # quick validate: read a small chunk first
            tmp = pd.read_csv(f, nrows=5)

            # IMPORTANT: reset pointer before reading full file
            f.seek(0)

            full = pd.read_csv(f)
            full = auto_clean_cached(full)
            st.session_state.uploaded_dfs.append(full)
            logger.info(f"Uploaded file appended: {getattr(f, 'name', 'uploaded')}")
        except Exception as e:
            st.sidebar.error(f"Failed reading {getattr(f, 'name', 'file')}: {e}")

dataset_option = st.sidebar.selectbox(
    "Dataset source",
    ["Use uploaded file(s)", "Load demo dataset"],
    index=0 if st.session_state.uploaded_dfs else 1
)

if dataset_option == "Use uploaded file(s)" and st.session_state.uploaded_dfs:
    try:
        df = pd.concat(st.session_state.uploaded_dfs, ignore_index=True, sort=False)
    except Exception as e:
        st.sidebar.error(f"Failed to concat uploaded datasets: {e}")
        st.session_state.uploaded_dfs = []
        dataset_option = "Load demo dataset"

if dataset_option == "Load demo dataset":
    rng = np.random.default_rng(42)
    n = 800
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n)
    df = pd.DataFrame({
        "Date": rng.choice(dates, size=n),
        "Product": rng.choice([f"P{str(i).zfill(3)}" for i in range(1, 61)], size=n),
        "Category": rng.choice(["Apparel", "Footwear", "Accessories", "Home"], size=n),
        "Sales": (rng.random(n) * 1000).round(2),
        "Inventory": rng.integers(0, 800, size=n),
        "Lead_Time_Days": rng.integers(1, 45, size=n),
        "Cost": (rng.random(n) * 200).round(2),
        "Supplier": rng.choice(["SupA", "SupB", "SupC", "SupD"], size=n)
    })

# auto-clean and column mapping
try:
    df = auto_clean_cached(df)
    cols = map_columns_cached(df)
except Exception as e:
    st.error(f"Data cleaning/map step failed: {e}")
    st.stop()

# canonical mapping
date_col = cols.get('date')
product_col = cols.get('product')
category_col = cols.get('category')
sales_col = cols.get('sales')
inventory_col = cols.get('inventory')
lead_col = cols.get('lead')
cost_col = cols.get('cost')
supplier_col = cols.get('supplier')

if date_col is None:
    st.error("No date-like column detected. Please upload a dataset with a Date column.")
    st.stop()

# parse date safely
try:
    df = safe_date_parse(df, date_col)
    df = df.dropna(subset=[date_col])
    df[date_col] = pd.to_datetime(df[date_col])
except Exception as e:
    st.error(f"Date parsing failed: {e}")
    st.stop()

# ensure numeric columns exist and are numeric
if sales_col and sales_col in df.columns:
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce').fillna(0)
else:
    df['Sales'] = 0
    sales_col = 'Sales'

if inventory_col and inventory_col in df.columns:
    df[inventory_col] = pd.to_numeric(df[inventory_col], errors='coerce').fillna(0)
else:
    df['Inventory'] = 0
    inventory_col = 'Inventory'

if lead_col and lead_col in df.columns:
    df[lead_col] = pd.to_numeric(df[lead_col], errors='coerce').fillna(0)
else:
    df['Lead_Time_Days'] = 0
    lead_col = 'Lead_Time_Days'

if cost_col and cost_col in df.columns:
    df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce').fillna(0)
else:
    df['Cost'] = 0
    cost_col = 'Cost'

if product_col is None:
    df['Product'] = df.index.astype(str)
    product_col = 'Product'
if category_col is None:
    df['Category'] = 'Unspecified'
    category_col = 'Category'
if supplier_col is None:
    df['Supplier'] = 'Unknown'
    supplier_col = 'Supplier'

# rename to canonical short names for internal use
rename_map = {
    date_col: 'Date',
    product_col: 'Product',
    category_col: 'Category',
    sales_col: 'Sales',
    inventory_col: 'Inventory',
    lead_col: 'Lead_Time_Days',
    cost_col: 'Cost',
    supplier_col: 'Supplier'
}
rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
try:
    df = df.rename(columns=rename_map)
except Exception as e:
    st.error(f"Failed to rename columns: {e}")
    logger.exception(e)

# Keep a shallow copy for debugging
_raw_df = df.copy()

# -----------------------
# Sidebar: Filters and settings (inside a form to avoid repeated reruns)
# -----------------------
st.sidebar.subheader("Filters")

min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

with st.sidebar.form('filters_form'):
    categories = sorted(df['Category'].dropna().unique())
    products = sorted(df['Product'].dropna().unique())
    suppliers = sorted(df['Supplier'].dropna().unique())

    selected_cats = st.multiselect('Categories', categories, default=categories)
    selected_prods = st.multiselect(
        'Products (top 50 shown)',
        products[:50],
        default=products[:min(50, len(products))]
    )
    selected_sups = st.multiselect('Suppliers', suppliers, default=suppliers)

    date_range = st.date_input(
        'Date range',
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    # theme
    theme_choice = st.selectbox('Theme', ['plotly_white', 'ggplot2', 'seaborn'])

    # EOQ params
    st.markdown('**EOQ / ROP Settings**')
    annual_holding_rate_pct = st.number_input(
        'Annual holding cost rate (%)',
        value=20.0,
        min_value=0.0
    )
    ordering_cost = st.number_input(
        'Ordering cost per order (₹)',
        value=500.0,
        min_value=0.0
    )
    lead_time_days_default = st.number_input(
        'Default lead time (days)',
        value=7,
        min_value=0
    )

    apply_filters = st.form_submit_button('Apply filters')

# If user hasn't applied filters yet, we continue using defaults but show info
if not apply_filters:
    st.info('Adjust filters on the left and click "Apply filters" to refresh analysis.')

# compute start/end
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

# Filter data
fdf = df[
    (df['Category'].isin(selected_cats)) &
    (df['Product'].isin(selected_prods)) &
    (df['Supplier'].isin(selected_sups)) &
    (df['Date'] >= start_date) &
    (df['Date'] <= end_date)
].copy()

if fdf.empty:
    st.warning('No data after applying filters. Adjust filters or broaden date range.')
    st.stop()

# show data freshness
st.caption(f"Rows: {len(fdf):,} • Date range: {fdf['Date'].min().date()} — {fdf['Date'].max().date()}")

# -----------------------
# Precompute commonly used aggregates (to avoid repeated groupbys)
# -----------------------
@st.cache_data
def precompute_aggregates(fdf: pd.DataFrame) -> dict:
    try:
        # monthly sales
        monthly_sales = (
            fdf.set_index('Date')
            .resample('M')['Sales']
            .sum()
            .reset_index()
            .sort_values('Date')
        )
        # product-level aggregates
        product_agg = (
            fdf.groupby('Product')
            .agg({
                'Sales': 'sum',
                'Inventory': 'mean',
                'Lead_Time_Days': 'mean',
                'Cost': 'mean'
            })
            .reset_index()
        )
        # category-product for treemap
        treemap_df = fdf.groupby(['Category', 'Product'])['Sales'].sum().reset_index()
        # supplier-category
        supplier_cat = fdf.groupby(['Supplier', 'Category'])['Sales'].sum().reset_index()
        return {
            'monthly_sales': monthly_sales,
            'product_agg': product_agg,
            'treemap': treemap_df,
            'supplier_cat': supplier_cat
        }
    except Exception as e:
        logger.exception('precompute failed')
        raise


aggs = precompute_aggregates(fdf)

# -----------------------
# KPIs & insights
# -----------------------
st.header('Key metrics & insights')

try:
    total_sales = fdf['Sales'].sum()
    unique_products = fdf['Product'].nunique()
    avg_lead = fdf['Lead_Time_Days'].mean()
    avg_inventory = fdf['Inventory'].mean()
    total_cost = fdf['Cost'].sum()

    k1, k2, k3, k4, k5 = st.columns([1.4, 1, 1, 1, 1])
    k1.metric('Total Sales (₹)', f"{total_sales:,.0f}")
    k2.metric('Unique Products', f"{unique_products}")
    k3.metric('Avg Lead Time (days)', f"{avg_lead:.1f}")
    k4.metric('Avg Inventory', f"{avg_inventory:.0f}")
    k5.metric('Total Cost (₹)', f"{total_cost:,.0f}")
except Exception as e:
    st.error(f"KPI calculation failed: {e}")
    logger.exception(e)

# Insight cards
ins_col1, ins_col2, ins_col3 = st.columns(3)

with ins_col1:
    try:
        recent_sales_ts = aggs['monthly_sales'].set_index('Date')['Sales'].sort_index()
        sales_trend_pct = None
        if len(recent_sales_ts) >= 2:
            sales_trend_pct = (
                (recent_sales_ts.iloc[-1] - recent_sales_ts.iloc[-2]) /
                (recent_sales_ts.iloc[-2] if recent_sales_ts.iloc[-2] != 0 else 1) * 100
            )
        st.markdown('#### 📈 Sales Insight')
        if sales_trend_pct is not None:
            st.write(f"Sales change MoM: **{sales_trend_pct:.1f}%**")
        else:
            st.write('Sales trend insufficient data')
    except Exception as e:
        st.info('Sales insight unavailable')
        logger.exception(e)

with ins_col2:
    try:
        low_products = aggs['product_agg'][['Product', 'Inventory']].rename(
            columns={'Inventory': 'AvgInventory'}
        )
        low_count = int((low_products['AvgInventory'] < 20).sum())
        st.markdown('#### 📦 Inventory Insight')
        st.write(f"Products with avg inventory < 20 units: **{low_count}**")
        if low_count > 0:
            sample = (
                low_products[low_products['AvgInventory'] < 20]
                .sort_values('AvgInventory')
                .head(5)
            )
            st.markdown("Low inventory products (sample):")
            st.table(sample)
    except Exception as e:
        st.info('Inventory insight unavailable')
        logger.exception(e)

with ins_col3:
    try:
        sup_stats = fdf.groupby('Supplier')['Sales'].sum().sort_values(ascending=False)
        st.markdown('#### 🏷️ Supplier Insight')
        if not sup_stats.empty:
            st.write(
                f"Top supplier by sales: **{sup_stats.index[0]}** "
                f"(₹{sup_stats.iloc[0]:,.0f})"
            )
        else:
            st.write('No supplier data')
    except Exception as e:
        st.info('Supplier insight unavailable')
        logger.exception(e)

# Executive summary (rule-based)
@st.cache_data
def executive_summary_cached(df_in: pd.DataFrame) -> str:
    return executive_summary(df_in)


def executive_summary(df_in):
    lines = []
    s = df_in.copy()
    total = s['Sales'].sum()
    lines.append(f"Total sales in selection: ₹{total:,.0f}.")
    # Growth
    try:
        monthly = s.set_index('Date').resample('M')['Sales'].sum().sort_index()
        if len(monthly) >= 2:
            last = monthly.iloc[-1]
            prev = monthly.iloc[-2]
            pct = (last - prev) / (prev if prev != 0 else 1) * 100
            if pct > 5:
                lines.append(f"Sales increased by {pct:.1f}% MoM — positive momentum.")
            elif pct < -5:
                lines.append(f"Sales decreased by {abs(pct):.1f}% MoM — investigate causes.")
    except Exception:
        pass
    # Category concentration
    try:
        cat = s.groupby('Category')['Sales'].sum().sort_values(ascending=False)
        if not cat.empty:
            top_cat = cat.index[0]
            pct = cat.iloc[0] / cat.sum() * 100 if cat.sum() != 0 else 0
            lines.append(f"Top category: {top_cat} — contributes {pct:.1f}% of sales.")
    except Exception:
        pass
    # Inventory warning
    try:
        prod_low = s.groupby('Product')['Inventory'].mean()
        low = prod_low[prod_low < 20].shape[0]
        if low > 0:
            lines.append(
                f"{low} products have average inventory below 20 units — consider restock."
            )
    except Exception:
        pass
    # Supplier risk
    try:
        if 'Supplier' in s.columns and 'Lead_Time_Days' in s.columns:
            risk = compute_supplier_risk_cached(s, 'Supplier', 'Lead_Time_Days')
            if not risk.empty:
                top_r = risk.iloc[0]
                if top_r['risk_score'] > 50:
                    lines.append(
                        f"Supplier risk flagged: {top_r['Supplier']} "
                        f"(risk score {top_r['risk_score']:.0f})."
                    )
    except Exception:
        pass
    # Add data window
    try:
        start = s['Date'].min().date()
        end = s['Date'].max().date()
        lines.insert(0, f"Data window: {start} — {end}.")
    except Exception:
        pass
    return ' '.join(lines)


st.markdown('### 🧾 Executive summary')
st.write(executive_summary_cached(fdf))

# -----------------------
# Visualizations (with explainers under each chart)
# -----------------------
st.markdown('---')
st.header('Visual analysis — Mixed chart pack')

# Treemap: Category -> Product
try:
    treemap_df = aggs['treemap']
    fig_treemap = px.treemap(
        treemap_df,
        path=['Category', 'Product'],
        values='Sales',
        title='Treemap: Sales by Category & Product',
        template=theme_choice
    )
    st.plotly_chart(fig_treemap, use_container_width=True)
    # explain treemap
    txt = explain_treemap_or_donut(treemap_df, group_col='Category', value_col='Sales')
    st.markdown(f"**Insight — Category composition:** {txt}")
except Exception as e:
    st.info('Treemap not available for this selection.')
    logger.exception(e)

# Sunburst if supplier present
if 'Supplier' in fdf.columns:
    try:
        sun = aggs['supplier_cat']
        fig_sun = px.sunburst(
            sun,
            path=['Supplier', 'Category'],
            values='Sales',
            title='Supplier -> Category Sales (Sunburst)',
            template=theme_choice
        )
        st.plotly_chart(fig_sun, use_container_width=True)
        # explain sunburst - top supplier highlight
        top_sup = fdf.groupby('Supplier')['Sales'].sum().sort_values(ascending=False)
        if not top_sup.empty:
            st.markdown(
                f"**Insight — Supplier:** Top supplier is **{top_sup.index[0]}** "
                f"contributing ₹{top_sup.iloc[0]:,.0f} of filtered sales."
            )
    except Exception:
        pass

# Area cumulative sales
try:
    area = aggs['monthly_sales'].copy()
    area['Cumulative'] = area['Sales'].cumsum()
    fig_area = px.area(
        area,
        x='Date',
        y='Cumulative',
        title='Cumulative Sales Over Time',
        template=theme_choice
    )
    st.plotly_chart(fig_area, use_container_width=True)
    # explain cumulative + drivers
    txt = explain_timeseries(aggs['monthly_sales'], date_col='Date', value_col='Sales')
    # top contributors last vs prev month
    last_month = fdf[fdf['Date'].dt.to_period('M') == fdf['Date'].dt.to_period('M').max()]
    prev_month = fdf[fdf['Date'].dt.to_period('M') == (fdf['Date'].dt.to_period('M').max() - 1)]
    gaining, losing = top_contributors_change(
        prev_month.groupby('Product')['Sales'].sum().reset_index(),
        last_month.groupby('Product')['Sales'].sum().reset_index(),
        top_n=3, value_col='Sales', key_col='Product'
    )
    st.markdown(
        f"**Insight — Cumulative:** {txt} "
        f"Top gainers: {format_contribs(gaining)}. Top losers: {format_contribs(losing)}."
    )
except Exception:
    pass

# Correlation heatmap
try:
    numeric_cols = [c for c in ['Sales', 'Inventory', 'Lead_Time_Days', 'Cost'] if c in fdf.columns]
    if len(numeric_cols) >= 2:
        corrm = fdf[numeric_cols].corr()
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=corrm.values,
                x=corrm.columns,
                y=corrm.columns,
                colorscale='RdBu',
                zmid=0
            )
        )
        fig_heat.update_layout(title='Correlation Heatmap', template=theme_choice, height=420)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown(
            "**Insight — Correlations:** Check pairs with correlation magnitude > 0.5 "
            "for potential relationships (e.g., inventory vs sales may indicate stocking impact)."
        )
except Exception:
    pass

# Bubble chart: Cost vs Sales, size Inventory
try:
    bubble = aggs['product_agg'].rename(columns={'Inventory': 'AvgInventory'})
    fig_bub = px.scatter(
        bubble,
        x='Cost',
        y='Sales',
        size='AvgInventory',
        hover_name='Product',
        title='Cost vs Sales (bubble=size inventory)',
        template=theme_choice
    )
    st.plotly_chart(fig_bub, use_container_width=True)
    # explain bubble
    high_cost_low_sales = bubble[
        (bubble['Cost'] > bubble['Cost'].median()) &
        (bubble['Sales'] < bubble['Sales'].median())
    ]
    txt = "Products to check: high cost but low sales — possible pricing or sourcing issues."
    if not high_cost_low_sales.empty:
        sample = ", ".join(
            high_cost_low_sales.sort_values('Cost', ascending=False).head(3)['Product'].tolist()
        )
        txt += f" Examples: {sample}."
    st.markdown(f"**Insight — Cost vs Sales:** {txt}")
except Exception:
    pass

# Donut chart category share
try:
    cat_sales = fdf.groupby('Category')['Sales'].sum().reset_index()
    fig_donut = px.pie(
        cat_sales,
        names='Category',
        values='Sales',
        hole=0.45,
        title='Category Share (Donut)',
        template=theme_choice
    )
    st.plotly_chart(fig_donut, use_container_width=True)
    txt = explain_treemap_or_donut(fdf, group_col='Category', value_col='Sales', prev_df=None)
    st.markdown(f"**Insight — Category share:** {txt}")
except Exception:
    pass

# Stacked monthly by category
try:
    monthly_cat = fdf.copy()
    monthly_cat['Month'] = monthly_cat['Date'].dt.to_period('M').dt.to_timestamp()
    monthly_group = monthly_cat.groupby(['Month', 'Category'])['Sales'].sum().reset_index()
    fig_stacked = px.bar(
        monthly_group,
        x='Month',
        y='Sales',
        color='Category',
        title='Monthly Sales by Category (Stacked)',
        template=theme_choice
    )
    st.plotly_chart(fig_stacked, use_container_width=True)
    # show which category gained/lost last month
    last_month = fdf[fdf['Date'].dt.to_period('M') == fdf['Date'].dt.to_period('M').max()]
    prev_month = fdf[fdf['Date'].dt.to_period('M') == (fdf['Date'].dt.to_period('M').max() - 1)]
    gains_cat, losses_cat = top_contributors_change(
        prev_month.groupby('Category')['Sales'].sum().reset_index(),
        last_month.groupby('Category')['Sales'].sum().reset_index(),
        top_n=2, value_col='Sales', key_col='Category'
    )
    st.markdown(
        f"**Insight — Monthly by category:** "
        f"Top gainers: {format_contribs(gains_cat)}. Top losers: {format_contribs(losses_cat)}."
    )
except Exception:
    pass

# Simple sales line
try:
    sales_ts = aggs['monthly_sales']
    fig_line = px.line(
        sales_ts,
        x='Date',
        y='Sales',
        title='Sales Trend (Line)',
        template=theme_choice
    )
    # Add annotation for latest point
    if not sales_ts.empty:
        latest_date = sales_ts['Date'].max()
        latest_val = float(sales_ts[sales_ts['Date'] == latest_date]['Sales'].iloc[0])
        fig_line.add_annotation(
            x=latest_date,
            y=latest_val,
            text=f"Latest: {latest_val:,.0f}",
            showarrow=True,
            arrowhead=1
        )
    st.plotly_chart(fig_line, use_container_width=True)
    # right-side compact insight card
    cols_line = st.columns([3, 1])
    with cols_line[0]:
        txt = explain_timeseries(sales_ts, date_col='Date', value_col='Sales')
        st.markdown(f"**Insight — Sales trend:** {txt}")
    with cols_line[1]:
        st.markdown("#### Quick actions")
        st.markdown("- Check top losing SKUs\n- Verify promotions or stockouts\n- Review supplier lead times")
except Exception:
    pass

# -----------------------
# Forecasting
# -----------------------
st.markdown('---')
st.header('Forecasting & what-if')

fc_method = st.selectbox('Forecasting method', ['Prophet (if available)', 'Linear Regression'], index=0)
horizon_months = st.number_input(
    'Forecast horizon (months)',
    min_value=1,
    max_value=24,
    value=3
)

if fc_method.startswith('Prophet'):
    # try to import prophet lazily
    try:
        try:
            from prophet import Prophet
        except Exception:
            from fbprophet import Prophet  # type: ignore
        HAS_PROPHET = True
    except Exception:
        HAS_PROPHET = False
        st.warning('Prophet not installed. Falling back to Linear Regression.')
        fc_method = 'Linear Regression'

# prepare timeseries
ts = aggs['monthly_sales'].copy()
if len(ts) < 3:
    st.info('Not enough history to run forecasting reliably (need >=3 monthly points).')
else:
    if fc_method == 'Prophet (if available)' and 'HAS_PROPHET' in globals() and HAS_PROPHET:
        try:
            m = Prophet()
            df_prop = ts.rename(columns={'Date': 'ds', 'Sales': 'y'})
            m.fit(df_prop)
            future = m.make_future_dataframe(periods=horizon_months, freq='M')
            forecast = m.predict(future)
            # safer plot: plot actuals and forecast separately
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=df_prop['ds'],
                y=df_prop['y'],
                mode='lines+markers',
                name='Actual'
            ))
            fig_f.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast'
            ))
            fig_f.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                name='Upper',
                line=dict(width=1),
                opacity=0.3
            ))
            fig_f.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                name='Lower',
                line=dict(width=1),
                opacity=0.3
            ))
            fig_f.update_layout(title='Prophet Forecast', template=theme_choice)
            st.plotly_chart(fig_f, use_container_width=True)
            st.markdown(
                "**Insight — Forecast:** Prophet forecasts shown with intervals. "
                "Check forecasted growth/decline against inventory planning."
            )
        except Exception as e:
            st.error(f'Prophet forecasting failed: {e}')
            logger.exception(e)
    else:
        # safer linear regression using month index
        try:
            ts_sorted = ts.sort_values('Date').reset_index(drop=True)
            # Month index since start
            start_year = ts_sorted['Date'].dt.year.min()
            ts_sorted['MonthIdx'] = (
                ts_sorted['Date'].dt.year - start_year
            ) * 12 + ts_sorted['Date'].dt.month
            X = ts_sorted['MonthIdx'].values
            y_vals = ts_sorted['Sales'].values
            coef = np.polyfit(X, y_vals, 1)
            poly = np.poly1d(coef)

            last_idx = X.max()
            future_idx = np.arange(last_idx + 1, last_idx + horizon_months + 1)
            preds = poly(future_idx)

            last_date = ts_sorted['Date'].max()
            future_dates = [
                last_date + pd.DateOffset(months=i) for i in range(1, horizon_months + 1)
            ]
            pred_df = pd.DataFrame({'Date': future_dates, 'Predicted': preds})

            fig_lr = go.Figure()
            fig_lr.add_trace(go.Scatter(
                x=ts_sorted['Date'],
                y=ts_sorted['Sales'],
                mode='lines+markers',
                name='Actual'
            ))
            fig_lr.add_trace(go.Scatter(
                x=pred_df['Date'],
                y=pred_df['Predicted'],
                mode='lines+markers',
                name='Predicted'
            ))
            fig_lr.update_layout(title='Linear Forecast (monthly)', template=theme_choice)
            st.plotly_chart(fig_lr, use_container_width=True)
            st.markdown(
                "**Insight — Forecast:** Linear projection shown. Use for quick planning; "
                "consider Prophet or more advanced models for seasonality."
            )
        except Exception as e:
            st.error(f'Linear regression forecasting error: {e}')
            logger.exception(e)

# -----------------------
# Inventory calculations: EOQ & ROP
# -----------------------
st.markdown('---')
st.header('Inventory calculations: EOQ & ROP')

try:
    prod_stats = (
        fdf.groupby('Product')
        .agg({
            'Sales': 'sum',
            'Inventory': 'mean',
            'Lead_Time_Days': 'mean'
        })
        .reset_index()
        .rename(columns={
            'Sales': 'TotalSales',
            'Inventory': 'AvgInventory',
            'Lead_Time_Days': 'AvgLeadDays'
        })
    )

    period_days = (fdf['Date'].max() - fdf['Date'].min()).days or 1
    prod_stats['AnnualDemand'] = prod_stats['TotalSales'] * (365.0 / period_days)

    # Unit cost
    cost_map = fdf.groupby('Product')['Cost'].mean().to_dict() if 'Cost' in fdf.columns else {}
    prod_stats['UnitCost'] = prod_stats['Product'].map(lambda p: cost_map.get(p, 0))
    prod_stats['HoldingCostPerUnit'] = prod_stats['UnitCost'] * (annual_holding_rate_pct / 100.0)

    prod_stats['EOQ'] = np.sqrt(
        (2 * prod_stats['AnnualDemand'] * ordering_cost) /
        (prod_stats['HoldingCostPerUnit'].replace(0, np.nan))
    )
    prod_stats['EOQ'] = prod_stats['EOQ'].fillna(0).round().astype(int)

    prod_stats['DailyDemand'] = prod_stats['AnnualDemand'] / 365.0
    prod_stats['ROP'] = (
        prod_stats['DailyDemand'] * (prod_stats['AvgLeadDays'].fillna(lead_time_days_default))
    ).round().astype(int)

    st.dataframe(
        prod_stats[['Product', 'AnnualDemand', 'EOQ', 'ROP', 'AvgInventory', 'AvgLeadDays']]
        .sort_values('EOQ', ascending=False)
        .head(50)
    )
    csv_eoq = prod_stats.to_csv(index=False).encode('utf-8')
    st.download_button('Download EOQ/ROP table', csv_eoq, 'eoq_rop.csv', 'text/csv')
except Exception as e:
    st.info('EOQ calculations unavailable')
    logger.exception(e)

# -----------------------
# Alerts panel
# -----------------------
st.markdown('---')
st.header('Operational Alerts')

try:
    prod_stats['AvgInventory'] = prod_stats['AvgInventory'].fillna(0)
    prod_stats['ROP'] = prod_stats['ROP'].fillna(0)

    critical = prod_stats[prod_stats['AvgInventory'] < prod_stats['ROP']]
    warning = prod_stats[
        (prod_stats['AvgInventory'] >= prod_stats['ROP']) &
        (prod_stats['AvgInventory'] < prod_stats['ROP'] * 1.5)
    ]

    if not critical.empty:
        st.error(f"CRITICAL: {len(critical)} products below ROP (immediate attention).")
        st.dataframe(critical[['Product', 'AvgInventory', 'ROP', 'EOQ']])
    else:
        st.success('No critical ROP breaches detected.')

    if not warning.empty:
        st.warning(f"Warning: {len(warning)} products near ROP threshold.")
        st.dataframe(warning[['Product', 'AvgInventory', 'ROP', 'EOQ']])
except Exception as e:
    st.info('Alerts unavailable')
    logger.exception(e)

# Supplier risk
if 'Supplier' in fdf.columns:
    try:
        sup_risk = compute_supplier_risk_cached(fdf, 'Supplier', 'Lead_Time_Days')
        st.subheader('Supplier Risk Scores')
        st.dataframe(sup_risk.head(20))
    except Exception:
        st.info('Supplier risk scoring not available for this dataset.')

# -----------------------
# Anomaly detection (simple z-score)
# -----------------------
st.subheader('Anomaly detection (sales)')
try:
    weekly = (
        fdf.set_index('Date')
        .groupby('Product')['Sales']
        .resample('W')
        .sum()
        .reset_index()
    )
    anomalies = []
    for prod, g in weekly.groupby('Product'):
        arr = g['Sales'].values
        if len(arr) < 6:
            continue
        z = (arr - np.nanmean(arr)) / (np.nanstd(arr) if np.nanstd(arr) != 0 else 1)
        idx = np.where(np.abs(z) > 3)[0]
        for i in idx:
            anomalies.append({
                'Product': prod,
                'Week': g.iloc[i]['Date'],
                'Sales': int(g.iloc[i]['Sales']),
                'Z': float(z[i])
            })
    if anomalies:
        st.warning(f'Detected {len(anomalies)} anomalies in weekly sales.')
        st.dataframe(pd.DataFrame(anomalies).head(50))
    else:
        st.success('No extreme anomalies detected via z-score.')
except Exception:
    st.info('Anomaly detection unavailable for this dataset.')

# -----------------------
# Optional ML explainability (lazy import)
# -----------------------
st.markdown('---')
st.header('Explainability / feature importance (optional)')

try:
    # import sklearn only when running this block
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

if HAS_SKLEARN:
    try:
        model_df = fdf.copy()
        model_df['Month'] = model_df['Date'].dt.to_period('M').dt.to_timestamp()
        agg = (
            model_df.groupby(['Product', 'Month'])
            .agg({
                'Sales': 'sum',
                'Inventory': 'mean',
                'Lead_Time_Days': 'mean',
                'Cost': 'mean'
            })
            .reset_index()
        )
        X = agg[['Inventory', 'Lead_Time_Days', 'Cost']].fillna(0)
        y_vals = agg['Sales'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_vals, test_size=0.2, random_state=42
        )
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        feats = X.columns.tolist()
        importances = rf.feature_importances_
        fi = (
            pd.DataFrame({'feature': feats, 'importance': importances})
            .sort_values('importance', ascending=False)
        )
        st.write('Feature importances (RandomForest):')
        st.dataframe(fi)
        fig_fi = px.bar(fi, x='feature', y='importance', title='Feature importances')
        st.plotly_chart(fig_fi, use_container_width=True)

        # SHAP (optional)
        try:
            import shap
            HAS_SHAP = True
        except Exception:
            HAS_SHAP = False
        if HAS_SHAP:
            try:
                explainer = shap.TreeExplainer(rf)
                shap_vals = explainer.shap_values(X_test.iloc[:100])
                st.write('SHAP summary plot (first 100 test rows):')
                try:
                    import matplotlib.pyplot as plt
                    shap.summary_plot(shap_vals, X_test.iloc[:100], show=False)
                    st.pyplot(plt.gcf())
                except Exception:
                    st.info('SHAP plotting failed in this environment.')
            except Exception:
                st.info('SHAP explanation failed.')
    except Exception as e:
        st.info(f'Explainability block skipped: {e}')
        logger.exception(e)
else:
    st.info('Install scikit-learn to enable explainability/feature importance (optional).')

# -----------------------
# Data table & download
# -----------------------
st.markdown('---')
st.header('Filtered dataset & download')

st.dataframe(fdf.head(200))
st.download_button(
    'Download filtered data (CSV)',
    fdf.to_csv(index=False).encode('utf-8'),
    'filtered_supply_data.csv',
    'text/csv'
)

st.markdown('---')
st.caption(
    'Enterprise Supply Chain Dashboard — rebuilt. '
    'If you want I can (1) add Prophet interval tuning, (2) export executive summary PDF, '
    '(3) add push alerts/email integration, (4) schedule auto-refresh.'
)
