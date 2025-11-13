# app.py
# Enterprise-level Supply Chain Dashboard (All-in-One)
# Features: labeled insight cards, AI-style summary, Prophet/Linear forecasting, EOQ/ROP,
# alerts, supplier risk, anomaly detection, multi-dataset upload, auto-clean, theme options.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Optional heavy imports — handled gracefully
HAS_SKLEARN = False
HAS_PROPHET = False
HAS_SHAP = False
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    # prophet package may be 'prophet' or 'fbprophet' depending on install
    try:
        from prophet import Prophet
        HAS_PROPHET = True
    except Exception:
        from fbprophet import Prophet  # type: ignore
        HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    import shap  # optional for explainability
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

st.set_page_config(page_title="Enterprise Fashion Supply Chain Dashboard", layout="wide")
st.title("🏬 Fashion Supply Chain Dashboard — Pro")

# =========================
# Helper utilities
# =========================
COMMON_COLS = {
    "date": ["date", "order_date", "sale_date", "timestamp"],
    "product": ["product", "product_name", "sku", "item"],
    "category": ["category", "cat", "product_category"],
    "sales": ["sales", "revenue", "amount", "total_sales"],
    "inventory": ["inventory", "stock", "on_hand"],
    "lead": ["lead_time", "lead_time_days", "lead_time_days"],
    "cost": ["cost", "unit_cost", "cost_price"],
    "supplier": ["supplier", "vendor", "vendor_name", "supplier_name"]
}

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def map_columns(df):
    """Return mapping from canonical name to actual column name in df (or None)"""
    cols = {}
    lc = [c.lower().strip() for c in df.columns]
    col_lookup = {c.lower().strip(): c for c in df.columns}
    for key, candidates in COMMON_COLS.items():
        found = None
        for cand in candidates:
            if cand in col_lookup:
                found = col_lookup[cand]
                break
        # fuzzy fallback: check substrings
        if not found:
            for orig in df.columns:
                low = orig.lower()
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
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df

def auto_clean(df):
    # Trim spaces
    df.columns = [c.strip() for c in df.columns]
    # Strip string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def compute_supplier_risk(df, supplier_col, lead_col):
    """Simple supplier risk: high avg lead time and high variability -> higher risk score"""
    s = df.groupby(supplier_col).agg(
        avg_lead=(lead_col, "mean"),
        std_lead=(lead_col, "std"),
        sales_share=( "Sales", "sum")
    ).reset_index()
    s["std_lead"] = s["std_lead"].fillna(0)
    # Normalize scores
    s["lead_norm"] = (s["avg_lead"] - s["avg_lead"].min()) / (s["avg_lead"].ptp() if s["avg_lead"].ptp() != 0 else 1)
    s["var_norm"] = (s["std_lead"] - s["std_lead"].min()) / (s["std_lead"].ptp() if s["std_lead"].ptp() != 0 else 1)
    s["risk_score"] = (s["lead_norm"] * 0.6 + s["var_norm"] * 0.4) * 100
    return s.sort_values("risk_score", ascending=False)

# =========================
# UI: Multi-file upload / dataset selection
# =========================
st.sidebar.header("Dataset & Settings")

uploaded_files = st.sidebar.file_uploader("Upload one or more CSVs (optional)", accept_multiple_files=True, type=["csv"])
dataset_option = st.sidebar.selectbox("Or choose sample/demo dataset", ["Use uploaded file(s)", "Load demo dataset"], index=0 if uploaded_files else 1)

if dataset_option == "Use uploaded file(s)" and uploaded_files:
    # If multiple, combine by concat with careful parsing
    df_list = []
    for f in uploaded_files:
        try:
            d = pd.read_csv(f)
            d = auto_clean(d)
            df_list.append(d)
        except Exception as e:
            st.sidebar.error(f"Failed reading {getattr(f, 'name', 'file')}: {e}")
    if not df_list:
        st.sidebar.warning("No valid uploaded files read; falling back to demo.")
        dataset_option = "Load demo dataset"
    else:
        df = pd.concat(df_list, ignore_index=True, sort=False)
else:
    # demo synthetic dataset
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
        "Supplier": rng.choice(["SupA","SupB","SupC","SupD"], size=n)
    })

# Auto-clean
df = auto_clean(df)

# Map columns
cols = map_columns(df)
date_col = cols.get("date")
product_col = cols.get("product")
category_col = cols.get("category")
sales_col = cols.get("sales")
inventory_col = cols.get("inventory")
lead_col = cols.get("lead")
cost_col = cols.get("cost")
supplier_col = cols.get("supplier")

# Quick required check
if date_col is None:
    st.error("No date-like column detected. Please upload a dataset with a Date column.")
    st.stop()

# Parse date
df = safe_date_parse(df, date_col)
df = df.dropna(subset=[date_col])
df[date_col] = pd.to_datetime(df[date_col])

# Ensure numeric sales etc.
if sales_col and sales_col in df.columns:
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0)
else:
    # create Sales column if missing (safe)
    df["Sales"] = 0
    sales_col = "Sales"

if inventory_col and inventory_col in df.columns:
    df[inventory_col] = pd.to_numeric(df[inventory_col], errors="coerce").fillna(0)
else:
    df["Inventory"] = 0
    inventory_col = "Inventory"

if lead_col and lead_col in df.columns:
    df[lead_col] = pd.to_numeric(df[lead_col], errors="coerce").fillna(0)
else:
    df["Lead_Time_Days"] = 0
    lead_col = "Lead_Time_Days"

if cost_col and cost_col in df.columns:
    df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0)
else:
    df["Cost"] = 0
    cost_col = "Cost"

if product_col is None:
    df["Product"] = df.index.astype(str)
    product_col = "Product"
if category_col is None:
    df["Category"] = "Unspecified"
    category_col = "Category"
if supplier_col is None:
    df["Supplier"] = "Unknown"
    supplier_col = "Supplier"

# Rename internal canonical columns for simplicity
df = df.rename(columns={
    date_col: "Date",
    product_col: "Product",
    category_col: "Category",
    sales_col: "Sales",
    inventory_col: "Inventory",
    lead_col: "Lead_Time_Days",
    cost_col: "Cost",
    supplier_col: "Supplier"
})
# keep only standard columns we will use plus any extras
# (we keep extras by not dropping columns)

# =========================
# Sidebar Filters (UI)
# =========================
st.sidebar.subheader("Filters")
categories = sorted(df["Category"].dropna().unique())
products = sorted(df["Product"].dropna().unique())
suppliers = sorted(df["Supplier"].dropna().unique())

selected_cats = st.sidebar.multiselect("Categories", categories, default=categories)
selected_prods = st.sidebar.multiselect("Products (top 50 shown)", products[:50], default=products[:min(50,len(products))])
selected_sups = st.sidebar.multiselect("Suppliers", suppliers, default=suppliers)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)

# theme
theme_choice = st.sidebar.selectbox("Theme", ["plotly_white", "ggplot2", "seaborn"])

# EOQ parameters
st.sidebar.subheader("EOQ / ROP Settings")
annual_holding_rate_pct = st.sidebar.number_input("Annual holding cost rate (%)", value=20.0, min_value=0.0)
ordering_cost = st.sidebar.number_input("Ordering cost per order (₹)", value=500.0, min_value=0.0)
lead_time_days_default = st.sidebar.number_input("Default lead time (days)", value=7, min_value=0)

# Filter data
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
fdf = df[
    (df["Category"].isin(selected_cats)) &
    (df["Product"].isin(selected_prods)) &
    (df["Supplier"].isin(selected_sups)) &
    (df["Date"] >= start_date) & (df["Date"] <= end_date)
].copy()

if fdf.empty:
    st.warning("No data after applying filters. Adjust filters.")
    st.stop()

# =========================
# KPI Row and Insight Cards
# =========================
st.header("Key metrics & insights")

total_sales = fdf["Sales"].sum()
unique_products = fdf["Product"].nunique()
avg_lead = fdf["Lead_Time_Days"].mean()
avg_inventory = fdf["Inventory"].mean()
total_cost = fdf["Cost"].sum()

k1, k2, k3, k4, k5 = st.columns([1.4,1,1,1,1])
k1.metric("Total Sales (₹)", f"{total_sales:,.0f}")
k2.metric("Unique Products", f"{unique_products}")
k3.metric("Avg Lead Time (days)", f"{avg_lead:.1f}")
k4.metric("Avg Inventory", f"{avg_inventory:.0f}")
k5.metric("Total Cost (₹)", f"{total_cost:,.0f}")

# Insight cards (visual)
ins_col1, ins_col2, ins_col3 = st.columns(3)
with ins_col1:
    # Sales insight card
    recent_sales_ts = fdf.set_index("Date").resample("M")["Sales"].sum().sort_index()
    sales_trend_pct = None
    if len(recent_sales_ts) >= 2:
        sales_trend_pct = (recent_sales_ts.iloc[-1] - recent_sales_ts.iloc[-2]) / (recent_sales_ts.iloc[-2] if recent_sales_ts.iloc[-2] != 0 else 1) * 100
    st.markdown("#### 📈 Sales Insight")
    if sales_trend_pct is not None:
        st.write(f"Sales change MoM: **{sales_trend_pct:.1f}%**")
    else:
        st.write("Sales trend insufficient data")

with ins_col2:
    # Inventory insight
    low_products = fdf.groupby("Product")["Inventory"].mean().reset_index()
    low_count = int((low_products["Inventory"] < 20).sum())
    st.markdown("#### 📦 Inventory Insight")
    st.write(f"Products with avg inventory < 20 units: **{low_count}**")
    if low_count > 0:
        sample = low_products[low_products["Inventory"] < 20].sort_values("Inventory").head(5)
        st.write(sample.to_dict(orient="records"))

with ins_col3:
    # Supplier insight
    sup_stats = fdf.groupby("Supplier")["Sales"].sum().sort_values(ascending=False)
    st.markdown("#### 🏷️ Supplier Insight")
    if not sup_stats.empty:
        st.write(f"Top supplier by sales: **{sup_stats.index[0]}** (₹{sup_stats.iloc[0]:,.0f})")
    else:
        st.write("No supplier data")

# AI-style short executive summary (rule-based)
def executive_summary(df_in):
    lines = []
    s = df_in.copy()
    total = s["Sales"].sum()
    lines.append(f"Total sales in selection: ₹{total:,.0f}.")
    # Growth
    try:
        monthly = s.set_index("Date").resample("M")["Sales"].sum().sort_index()
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
        cat = s.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        if not cat.empty:
            top_cat = cat.index[0]
            pct = cat.iloc[0] / cat.sum() * 100 if cat.sum() != 0 else 0
            lines.append(f"Top category: {top_cat} — contributes {pct:.1f}% of sales.")
    except Exception:
        pass
    # Inventory warning
    try:
        prod_low = s.groupby("Product")["Inventory"].mean()
        low = prod_low[prod_low < 20].shape[0]
        if low > 0:
            lines.append(f"{low} products have average inventory below 20 units — consider restock.")
    except Exception:
        pass
    # Supplier risk
    try:
        if "Supplier" in s.columns and "Lead_Time_Days" in s.columns:
            risk = compute_supplier_risk(s, "Supplier", "Lead_Time_Days")
            if not risk.empty:
                top_r = risk.iloc[0]
                if top_r["risk_score"] > 50:
                    lines.append(f"Supplier risk flagged: {top_r['Supplier']} (risk score {top_r['risk_score']:.0f}).")
    except Exception:
        pass
    return " ".join(lines)

st.markdown("### 🧾 Executive summary")
st.write(executive_summary(fdf))

# =========================
# Visualizations (mixed pack)
# =========================
st.markdown("---")
st.header("Visual analysis — Mixed chart pack")

# Treemap: Category -> Product
try:
    treemap_df = fdf.groupby(["Category", "Product"])["Sales"].sum().reset_index()
    fig_treemap = px.treemap(treemap_df, path=["Category", "Product"], values="Sales", title="Treemap: Sales by Category & Product", template=theme_choice)
    st.plotly_chart(fig_treemap, use_container_width=True)
except Exception:
    st.info("Treemap not available for this selection.")

# Sunburst if supplier present
if "Supplier" in fdf.columns:
    try:
        sun = fdf.groupby(["Supplier", "Category"])["Sales"].sum().reset_index()
        fig_sun = px.sunburst(sun, path=["Supplier", "Category"], values="Sales", title="Supplier -> Category Sales (Sunburst)", template=theme_choice)
        st.plotly_chart(fig_sun, use_container_width=True)
    except Exception:
        pass

# Area cumulative sales
try:
    area = fdf.groupby("Date")["Sales"].sum().reset_index().sort_values("Date")
    area["Cumulative"] = area["Sales"].cumsum()
    fig_area = px.area(area, x="Date", y="Cumulative", title="Cumulative Sales Over Time", template=theme_choice)
    st.plotly_chart(fig_area, use_container_width=True)
except Exception:
    pass

# Correlation heatmap
try:
    numeric_cols = [c for c in ["Sales", "Inventory", "Lead_Time_Days", "Cost"] if c in fdf.columns]
    if len(numeric_cols) >= 2:
        corrm = fdf[numeric_cols].corr()
        fig_heat = go.Figure(data=go.Heatmap(z=corrm.values, x=corrm.columns, y=corrm.columns, colorscale="RdBu", zmid=0))
        fig_heat.update_layout(title="Correlation Heatmap", template=theme_choice, height=420)
        st.plotly_chart(fig_heat, use_container_width=True)
except Exception:
    pass

# Bubble chart: Cost vs Sales, size Inventory
try:
    bubble = fdf.groupby("Product").agg({"Sales":"sum", "Cost":"mean", "Inventory":"mean"}).reset_index()
    fig_bub = px.scatter(bubble, x="Cost", y="Sales", size="Inventory", hover_name="Product", title="Cost vs Sales (bubble=size inventory)", template=theme_choice)
    st.plotly_chart(fig_bub, use_container_width=True)
except Exception:
    pass

# Donut chart category share
try:
    cat_sales = fdf.groupby("Category")["Sales"].sum().reset_index()
    fig_donut = px.pie(cat_sales, names="Category", values="Sales", hole=0.45, title="Category Share (Donut)", template=theme_choice)
    st.plotly_chart(fig_donut, use_container_width=True)
except Exception:
    pass

# Stacked monthly by category
try:
    monthly_cat = fdf.copy()
    monthly_cat["Month"] = monthly_cat["Date"].dt.to_period("M").dt.to_timestamp()
    monthly_group = monthly_cat.groupby(["Month", "Category"])["Sales"].sum().reset_index()
    fig_stacked = px.bar(monthly_group, x="Month", y="Sales", color="Category", title="Monthly Sales by Category (Stacked)", template=theme_choice)
    st.plotly_chart(fig_stacked, use_container_width=True)
except Exception:
    pass

# Simple sales line
try:
    sales_ts = fdf.groupby("Date")["Sales"].sum().reset_index().sort_values("Date")
    fig_line = px.line(sales_ts, x="Date", y="Sales", title="Sales Trend (Line)", template=theme_choice)
    st.plotly_chart(fig_line, use_container_width=True)
except Exception:
    pass

# =========================
# Forecasting (Prophet if available, else LR)
# =========================
st.markdown("---")
st.header("Forecasting & what-if")

fc_method = st.selectbox("Forecasting method", ["Prophet (if available)", "Linear Regression"], index=0)
horizon_months = st.number_input("Forecast horizon (months)", min_value=1, max_value=24, value=3)

if fc_method.startswith("Prophet") and not HAS_PROPHET:
    st.warning("Prophet not installed. Please install `prophet` to use this method. Falling back to Linear Regression.")
    fc_method = "Linear Regression"

# Prepare ts
ts = fdf.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index().sort_values("Date")
if len(ts) < 3:
    st.info("Not enough history to run forecasting reliably (need >=3 monthly points).")
else:
    if fc_method == "Prophet (if available)" and HAS_PROPHET:
        try:
            m = Prophet()
            df_prop = ts.rename(columns={"Date":"ds", "Sales":"y"})
            m.fit(df_prop)
            future = m.make_future_dataframe(periods=horizon_months, freq='M')
            forecast = m.predict(future)
            fc_plot = forecast[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={'ds':'Date'})
            # plot actual + forecast
            act = df_prop.rename(columns={'ds':'Date','y':'Actual'})
            merged = act.merge(fc_plot, on='Date', how='right').fillna(np.nan)
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=act['Date'], y=act['Actual'], mode='lines+markers', name='Actual'))
            fig_f.add_trace(go.Scatter(x=fc_plot['Date'], y=fc_plot['yhat'], mode='lines', name='Forecast'))
            fig_f.add_trace(go.Scatter(x=fc_plot['Date'], y=fc_plot['yhat_upper'], mode='lines', name='Upper', line=dict(width=1), opacity=0.3))
            fig_f.add_trace(go.Scatter(x=fc_plot['Date'], y=fc_plot['yhat_lower'], mode='lines', name='Lower', line=dict(width=1), opacity=0.3))
            fig_f.update_layout(title="Prophet Forecast", template=theme_choice)
            st.plotly_chart(fig_f, use_container_width=True)
        except Exception as e:
            st.error(f"Prophet forecasting failed: {e}")
    else:
        # Linear regression on ordinal
        try:
            X = np.array([d.toordinal() for d in ts["Date"]]).reshape(-1,1)
            y = ts["Sales"].values
            # simple linear regression
            coef = np.polyfit(X.flatten(), y, 1)
            poly = np.poly1d(coef)
            last_date = ts["Date"].max()
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon_months+1)]
            Xf = np.array([d.toordinal() for d in future_dates])
            preds = poly(Xf)
            pred_df = pd.DataFrame({"Date": future_dates, "Predicted": preds})
            # plot
            fig_lr = go.Figure()
            fig_lr.add_trace(go.Scatter(x=ts["Date"], y=ts["Sales"], mode='lines+markers', name='Actual'))
            fig_lr.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Predicted"], mode='lines+markers', name='Predicted'))
            fig_lr.update_layout(title="Linear Forecast (monthly)", template=theme_choice)
            st.plotly_chart(fig_lr, use_container_width=True)
        except Exception as e:
            st.error(f"Linear regression forecasting error: {e}")

# =========================
# EOQ & ROP (Inventory calculators)
# =========================
st.markdown("---")
st.header("Inventory calculations: EOQ & ROP")

# build per-product demand estimate (annualized)
prod_stats = fdf.groupby("Product").agg({
    "Sales":"sum",
    "Inventory":"mean",
    "Lead_Time_Days":"mean"
}).reset_index().rename(columns={"Sales":"TotalSales","Inventory":"AvgInventory","Lead_Time_Days":"AvgLeadDays"})

# Convert to annual demand (approx)
period_days = (fdf["Date"].max() - fdf["Date"].min()).days or 1
prod_stats["AnnualDemand"] = prod_stats["TotalSales"] * (365.0 / period_days)

# Holding cost per unit = cost * holding_rate%
prod_stats["UnitCost"] = prod_stats["Product"].map(lambda p: fdf[fdf["Product"]==p]["Cost"].mean() if "Cost" in fdf.columns else 0)
prod_stats["HoldingCostPerUnit"] = prod_stats["UnitCost"] * (annual_holding_rate_pct/100.0)

# EOQ formula: sqrt(2DS/H)
prod_stats["EOQ"] = np.sqrt((2 * prod_stats["AnnualDemand"] * ordering_cost) / (prod_stats["HoldingCostPerUnit"].replace(0, np.nan)))
prod_stats["EOQ"] = prod_stats["EOQ"].fillna(0).round().astype(int)

# ROP = (daily demand * lead time)
prod_stats["DailyDemand"] = prod_stats["AnnualDemand"] / 365.0
prod_stats["ROP"] = (prod_stats["DailyDemand"] * (prod_stats["AvgLeadDays"].fillna(lead_time_days_default))).round().astype(int)

# present table and allow download
st.dataframe(prod_stats[["Product","AnnualDemand","EOQ","ROP","AvgInventory","AvgLeadDays"]].sort_values("EOQ", ascending=False).head(50))
csv_eoq = prod_stats.to_csv(index=False).encode("utf-8")
st.download_button("Download EOQ/ROP table", csv_eoq, "eoq_rop.csv", "text/csv")

# =========================
# Alerts panel (operational)
# =========================
st.markdown("---")
st.header("Operational Alerts")

# critical: product inventory < safety stock (ROP) OR zero sales anomaly
critical = prod_stats[prod_stats["AvgInventory"] < prod_stats["ROP"]]
warning = prod_stats[(prod_stats["AvgInventory"] >= prod_stats["ROP"]) & (prod_stats["AvgInventory"] < prod_stats["ROP"]*1.5)]

if not critical.empty:
    st.error(f"CRITICAL: {len(critical)} products below ROP (immediate attention).")
    st.dataframe(critical[["Product","AvgInventory","ROP","EOQ"]])
else:
    st.success("No critical ROP breaches detected.")

# supplier risk panel
if "Supplier" in fdf.columns:
    try:
        sup_risk = compute_supplier_risk(fdf, "Supplier", "Lead_Time_Days")
        st.subheader("Supplier Risk Scores")
        st.dataframe(sup_risk.head(20))
    except Exception:
        st.info("Supplier risk scoring not available for this dataset.")

# anomaly detection on product-level sales (z-score)
st.subheader("Anomaly detection (sales)")
try:
    weekly = fdf.set_index("Date").groupby("Product")["Sales"].resample("W").sum().reset_index()
    anomalies = []
    for prod, g in weekly.groupby("Product"):
        arr = g["Sales"].values
        if len(arr) < 6:
            continue
        z = (arr - np.nanmean(arr)) / (np.nanstd(arr) if np.nanstd(arr)!=0 else 1)
        # mark any week where z>3 or z<-3 as anomaly
        idx = np.where(np.abs(z) > 3)[0]
        for i in idx:
            anomalies.append({"Product":prod, "Week": g.iloc[i]["Date"], "Sales": int(g.iloc[i]["Sales"]), "Z": float(z[i])})
    if anomalies:
        st.warning(f"Detected {len(anomalies)} anomalies in weekly sales.")
        st.dataframe(pd.DataFrame(anomalies).head(50))
    else:
        st.success("No extreme anomalies detected via z-score.")
except Exception:
    st.info("Anomaly detection unavailable for this dataset.")

# Optional: ML-based importance (RandomForest)
st.markdown("---")
st.header("Explainability / feature importance (optional)")

if HAS_SKLEARN:
    try:
        # Build a simple regression to predict Sales from Inventory, Lead_Time_Days, Cost (agg per product-month)
        model_df = fdf.copy()
        model_df["Month"] = model_df["Date"].dt.to_period("M").dt.to_timestamp()
        agg = model_df.groupby(["Product","Month"]).agg({
            "Sales":"sum","Inventory":"mean","Lead_Time_Days":"mean","Cost":"mean"
        }).reset_index()
        X = agg[["Inventory","Lead_Time_Days","Cost"]].fillna(0)
        y = agg["Sales"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        feats = X.columns.tolist()
        importances = rf.feature_importances_
        fi = pd.DataFrame({"feature":feats,"importance":importances}).sort_values("importance", ascending=False)
        st.write("Feature importances (RandomForest):")
        st.dataframe(fi)
        fig_fi = px.bar(fi, x="feature", y="importance", title="Feature importances")
        st.plotly_chart(fig_fi, use_container_width=True)
        # Optional shap
        if HAS_SHAP:
            explainer = shap.TreeExplainer(rf)
            shap_vals = explainer.shap_values(X_test.iloc[:100])
            st.write("SHAP summary plot (first 100 test rows):")
            # shap visualization in streamlit is tricky: use matplotlib and st.pyplot
            try:
                shap.summary_plot(shap_vals, X_test.iloc[:100], show=False)
                import matplotlib.pyplot as plt
                st.pyplot(plt.gcf())
            except Exception:
                st.info("SHAP plotting failed in this environment.")
    except Exception as e:
        st.info(f"Explainability block skipped: {e}")
else:
    st.info("Install scikit-learn to enable explainability/feature importance (optional).")

# =========================
# Data table and download
# =========================
st.markdown("---")
st.header("Filtered dataset & download")
st.dataframe(fdf.head(200))
st.download_button("Download filtered data (CSV)", fdf.to_csv(index=False).encode("utf-8"), "filtered_supply_data.csv", "text/csv")

st.markdown("---")
st.caption("Enterprise Supply Chain Dashboard — enhanced. If you'd like, I can: (1) add Prophet interval tuning, (2) export executive summary PDF, (3) add push alerts/email integration, (4) schedule auto-refresh. Tell me which one to implement next.")
