# app.py
# Enterprise-level Fashion Supply Chain Dashboard (Option D)
# - Labeled insight cards, executive summary
# - Mixed visualization pack
# - Forecasting (Prophet fallback to Linear Regression)
# - EOQ / ROP calculations (hidden defaults)
# - Supplier risk scoring, anomaly detection
# - Explainability (RandomForest) if scikit-learn available
# - Multi-file upload and auto-cleaning

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Optional heavy libs (handled gracefully)
HAS_SKLEARN = False
HAS_PROPHET = False
HAS_SHAP = False
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# Prophet may be installed under different names
try:
    try:
        from prophet import Prophet  # type: ignore
        HAS_PROPHET = True
    except Exception:
        from fbprophet import Prophet  # type: ignore
        HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

st.set_page_config(page_title="Enterprise Fashion Supply Chain Dashboard", layout="wide")
st.title("🏬 Enterprise Fashion Supply Chain Dashboard — Pro")

# ---------------------------
# Utilities / Column mapping
# ---------------------------
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

def auto_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def map_columns(df: pd.DataFrame):
    lc = [c.lower().strip() for c in df.columns]
    lookup = {c.lower().strip(): c for c in df.columns}
    cols = {}
    for key, candidates in COMMON_COLS.items():
        found = None
        for cand in candidates:
            if cand in lookup:
                found = lookup[cand]
                break
        if not found:
            # fuzzy fallback: substring match
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

def safe_date_parse(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df

def compute_supplier_risk(df: pd.DataFrame, supplier_col: str, lead_col: str) -> pd.DataFrame:
    s = df.groupby(supplier_col).agg(
        avg_lead=(lead_col, "mean"),
        std_lead=(lead_col, "std"),
        sales=( "Sales", "sum")
    ).reset_index()
    s["std_lead"] = s["std_lead"].fillna(0)
    if s["avg_lead"].ptp() == 0:
        s["lead_norm"] = 0
    else:
        s["lead_norm"] = (s["avg_lead"] - s["avg_lead"].min()) / (s["avg_lead"].ptp())
    if s["std_lead"].ptp() == 0:
        s["var_norm"] = 0
    else:
        s["var_norm"] = (s["std_lead"] - s["std_lead"].min()) / (s["std_lead"].ptp())
    s["risk_score"] = (s["lead_norm"] * 0.6 + s["var_norm"] * 0.4) * 100
    s = s.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return s

# ---------------------------
# Data ingestion & demo
# ---------------------------
st.sidebar.header("Dataset & settings")

uploaded_files = st.sidebar.file_uploader("Upload one or more CSV files (optional)", accept_multiple_files=True, type=["csv"])
use_demo = st.sidebar.checkbox("Use demo dataset instead of uploads", value=(not bool(uploaded_files)))

if uploaded_files and not use_demo:
    df_list = []
    for f in uploaded_files:
        try:
            d = pd.read_csv(f)
            d = auto_clean(d)
            df_list.append(d)
        except Exception as e:
            st.sidebar.error(f"Failed to read {getattr(f, 'name', 'file')}: {e}")
    if len(df_list) == 0:
        st.sidebar.warning("No valid uploaded files read — demo dataset will be used.")
        use_demo = True
    else:
        df = pd.concat(df_list, ignore_index=True, sort=False)
else:
    use_demo = True

if use_demo:
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

# Auto-clean and map
df = auto_clean(df)
cols = map_columns(df)

date_col = cols.get("date")
product_col = cols.get("product")
category_col = cols.get("category")
sales_col = cols.get("sales")
inventory_col = cols.get("inventory")
lead_col = cols.get("lead")
cost_col = cols.get("cost")
supplier_col = cols.get("supplier")

if date_col is None:
    st.error("No date-like column detected. Please upload a dataset with a Date column.")
    st.stop()

# Parse dates and coerce
df = safe_date_parse(df, date_col)
df = df.dropna(subset=[date_col]).copy()
df[date_col] = pd.to_datetime(df[date_col])

# Normalize/ensure columns exist
if sales_col and sales_col in df.columns:
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0)
else:
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

# Standardize column names for internal use
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

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.subheader("Filters")
categories = sorted(df["Category"].dropna().unique())
products = sorted(df["Product"].dropna().unique())
suppliers = sorted(df["Supplier"].dropna().unique())

selected_cats = st.sidebar.multiselect("Categories", categories, default=categories)
# show up to 100 products by default
selected_prods = st.sidebar.multiselect("Products (top 100 shown)", products[:100], default=products[:min(100,len(products))])
selected_sups = st.sidebar.multiselect("Suppliers", suppliers, default=suppliers)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)

theme_choice = st.sidebar.selectbox("Chart theme", ["plotly_white", "ggplot2", "seaborn"])

# ---------------------------
# Hidden EOQ/ROP defaults (HIDDEN in sidebar)
# ---------------------------
annual_holding_rate_pct = 20.0    # default 20% annual holding cost
ordering_cost = 500.0             # ₹500 per order default
lead_time_days_default = 7        # 7-day default lead time

# ---------------------------
# Filter dataset
# ---------------------------
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

# ---------------------------
# KPIs & Insight Cards
# ---------------------------
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

# Insight cards
ins_col1, ins_col2, ins_col3 = st.columns(3)
with ins_col1:
    st.markdown("#### 📈 Sales Insight")
    recent_sales_ts = fdf.set_index("Date").resample("M")["Sales"].sum().sort_index()
    if len(recent_sales_ts) >= 2:
        sales_trend_pct = (recent_sales_ts.iloc[-1] - recent_sales_ts.iloc[-2]) / (recent_sales_ts.iloc[-2] if recent_sales_ts.iloc[-2] != 0 else 1) * 100
        st.write(f"Sales change MoM: **{sales_trend_pct:.1f}%**")
    else:
        st.write("Insufficient monthly history for trend.")

with ins_col2:
    st.markdown("#### 📦 Inventory Insight")
    low_products = fdf.groupby("Product")["Inventory"].mean().reset_index()
    low_count = int((low_products["Inventory"] < 20).sum())
    st.write(f"Products with avg inventory < 20 units: **{low_count}**")
    if low_count > 0:
        st.write(low_products[low_products["Inventory"] < 20].sort_values("Inventory").head(5).to_dict(orient="records"))

with ins_col3:
    st.markdown("#### 🏷️ Supplier Insight")
    sup_stats = fdf.groupby("Supplier")["Sales"].sum().sort_values(ascending=False)
    if not sup_stats.empty:
        st.write(f"Top supplier by sales: **{sup_stats.index[0]}** (₹{sup_stats.iloc[0]:,.0f})")
    else:
        st.write("No supplier data available")

# Executive summary (rule-based)
def executive_summary(df_in):
    lines = []
    s = df_in.copy()
    total = s["Sales"].sum()
    lines.append(f"Total sales: ₹{total:,.0f}.")
    try:
        monthly = s.set_index("Date").resample("M")["Sales"].sum().sort_index()
        if len(monthly) >= 2:
            last = monthly.iloc[-1]
            prev = monthly.iloc[-2]
            pct = (last - prev) / (prev if prev != 0 else 1) * 100
            if pct > 5:
                lines.append(f"Sales increased by {pct:.1f}% MoM.")
            elif pct < -5:
                lines.append(f"Sales decreased by {abs(pct):.1f}% MoM — investigate.")
    except Exception:
        pass
    try:
        cat = s.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        if not cat.empty:
            top_cat = cat.index[0]
            pct = cat.iloc[0] / cat.sum() * 100 if cat.sum() != 0 else 0
            lines.append(f"Top category: {top_cat} ({pct:.1f}% of sales).")
    except Exception:
        pass
    try:
        prod_low = s.groupby("Product")["Inventory"].mean()
        low = int((prod_low < 20).sum())
        if low > 0:
            lines.append(f"{low} products have avg inventory <20 units.")
    except Exception:
        pass
    try:
        if "Supplier" in s.columns and "Lead_Time_Days" in s.columns:
            risk = compute_supplier_risk(s, "Supplier", "Lead_Time_Days")
            if not risk.empty and risk.iloc[0]["risk_score"] > 50:
                lines.append(f"Supplier risk flagged: {risk.iloc[0]['Supplier']} (score {risk.iloc[0]['risk_score']:.0f}).")
    except Exception:
        pass
    return " ".join(lines)

st.markdown("### 🧾 Executive summary")
st.write(executive_summary(fdf))

# ---------------------------
# Mixed visualization pack
# ---------------------------
st.markdown("---")
st.header("Visual analysis — Mixed chart pack")

# Treemap: Category -> Product
try:
    treemap_df = fdf.groupby(["Category", "Product"])["Sales"].sum().reset_index()
    fig_treemap = px.treemap(treemap_df, path=["Category", "Product"], values="Sales", title="Treemap — Sales by Category & Product", template=theme_choice)
    st.plotly_chart(fig_treemap, use_container_width=True)
except Exception:
    st.info("Treemap not available for selection.")

# Sunburst: Supplier -> Category
if "Supplier" in fdf.columns:
    try:
        sun_df = fdf.groupby(["Supplier", "Category"])["Sales"].sum().reset_index()
        fig_sun = px.sunburst(sun_df, path=["Supplier", "Category"], values="Sales", title="Sunburst — Supplier -> Category Sales", template=theme_choice)
        st.plotly_chart(fig_sun, use_container_width=True)
    except Exception:
        pass

# Area cumulative
try:
    area_ts = fdf.groupby("Date")["Sales"].sum().reset_index().sort_values("Date")
    area_ts["Cumulative"] = area_ts["Sales"].cumsum()
    fig_area = px.area(area_ts, x="Date", y="Cumulative", title="Cumulative Sales Over Time", template=theme_choice)
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

# Bubble: Cost vs Sales size Inventory
try:
    bubble = fdf.groupby("Product").agg({"Sales":"sum", "Cost":"mean", "Inventory":"mean"}).reset_index()
    fig_bubble = px.scatter(bubble, x="Cost", y="Sales", size="Inventory", hover_name="Product", title="Bubble Chart — Cost vs Sales (size=avg inventory)", template=theme_choice)
    st.plotly_chart(fig_bubble, use_container_width=True)
except Exception:
    pass

# Donut: category share
try:
    cat_sales = fdf.groupby("Category")["Sales"].sum().reset_index()
    fig_donut = px.pie(cat_sales, names="Category", values="Sales", hole=0.45, title="Donut — Category Share", template=theme_choice)
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
    sales_trend = fdf.groupby("Date")["Sales"].sum().reset_index().sort_values("Date")
    fig_line = px.line(sales_trend, x="Date", y="Sales", title="Sales Trend (Line)", template=theme_choice)
    st.plotly_chart(fig_line, use_container_width=True)
except Exception:
    pass

# ---------------------------
# Forecasting
# ---------------------------
st.markdown("---")
st.header("Forecasting & What-if")
fc_method = st.selectbox("Forecast method", ["Prophet (if available)", "Linear Regression"], index=0)
horizon_months = st.number_input("Forecast horizon (months)", min_value=1, max_value=24, value=3)

if fc_method.startswith("Prophet") and not HAS_PROPHET:
    st.warning("Prophet is not installed in this environment. Falling back to Linear Regression.")
    fc_method = "Linear Regression"

ts = fdf.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index().sort_values("Date")
if len(ts) < 3:
    st.info("Not enough monthly points for reliable forecasting (need >= 3).")
else:
    if fc_method == "Prophet (if available)" and HAS_PROPHET:
        try:
            m = Prophet()
            df_prop = ts.rename(columns={"Date":"ds", "Sales":"y"})
            m.fit(df_prop)
            future = m.make_future_dataframe(periods=horizon_months, freq='M')
            forecast = m.predict(future)
            fc_plot = forecast[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={'ds':'Date'})
            act = df_prop.rename(columns={'ds':'Date','y':'Actual'})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=act['Date'], y=act['Actual'], mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(x=fc_plot['Date'], y=fc_plot['yhat'], mode='lines', name='Forecast'))
            fig.add_trace(go.Scatter(x=fc_plot['Date'], y=fc_plot['yhat_upper'], mode='lines', name='Upper', line=dict(width=1), opacity=0.3))
            fig.add_trace(go.Scatter(x=fc_plot['Date'], y=fc_plot['yhat_lower'], mode='lines', name='Lower', line=dict(width=1), opacity=0.3))
            fig.update_layout(title="Prophet Forecast", template=theme_choice)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Prophet forecast error: {e}")
    else:
        try:
            X = np.array([d.toordinal() for d in ts["Date"]]).reshape(-1,1)
            y = ts["Sales"].values
            coef = np.polyfit(X.flatten(), y, 1)
            poly = np.poly1d(coef)
            last_date = ts["Date"].max()
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon_months+1)]
            Xf = np.array([d.toordinal() for d in future_dates])
            preds = poly(Xf)
            pred_df = pd.DataFrame({"Date": future_dates, "Predicted": preds})
            fig_lr = go.Figure()
            fig_lr.add_trace(go.Scatter(x=ts["Date"], y=ts["Sales"], mode='lines+markers', name='Actual'))
            fig_lr.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Predicted"], mode='lines+markers', name='Predicted'))
            fig_lr.update_layout(title="Linear Forecast (monthly)", template=theme_choice)
            st.plotly_chart(fig_lr, use_container_width=True)
        except Exception as e:
            st.error(f"Linear forecast error: {e}")

# ---------------------------
# EOQ & ROP calculations (hidden defaults used)
# ---------------------------
st.markdown("---")
st.header("Inventory calculations: EOQ & ROP (computed with hidden defaults)")

prod_stats = fdf.groupby("Product").agg({
    "Sales": "sum",
    "Inventory": "mean",
    "Lead_Time_Days": "mean",
    "Cost": "mean"
}).reset_index().rename(columns={"Sales":"TotalSales","Inventory":"AvgInventory","Lead_Time_Days":"AvgLeadDays","Cost":"UnitCost"})

# Period days in dataset
period_days = (fdf["Date"].max() - fdf["Date"].min()).days or 1
prod_stats["AnnualDemand"] = prod_stats["TotalSales"] * (365.0 / period_days)
prod_stats["HoldingCostPerUnit"] = prod_stats["UnitCost"] * (annual_holding_rate_pct/100.0)

# EOQ
prod_stats["EOQ"] = np.sqrt((2 * prod_stats["AnnualDemand"] * ordering_cost) / (prod_stats["HoldingCostPerUnit"].replace(0, np.nan)))
prod_stats["EOQ"] = prod_stats["EOQ"].fillna(0).round().astype(int)

# ROP = daily demand * lead time
prod_stats["DailyDemand"] = prod_stats["AnnualDemand"] / 365.0
prod_stats["ROP"] = (prod_stats["DailyDemand"] * (prod_stats["AvgLeadDays"].fillna(lead_time_days_default))).round().astype(int)

st.dataframe(prod_stats[["Product","AnnualDemand","EOQ","ROP","AvgInventory","AvgLeadDays"]].sort_values("EOQ", ascending=False).head(50))
st.download_button("Download EOQ/ROP table", prod_stats.to_csv(index=False).encode("utf-8"), "eoq_rop.csv", "text/csv")

# ---------------------------
# Operational alerts
# ---------------------------
st.markdown("---")
st.header("Operational alerts")

critical = prod_stats[prod_stats["AvgInventory"] < prod_stats["ROP"]]
if not critical.empty:
    st.error(f"CRITICAL: {len(critical)} products below ROP (immediate attention).")
    st.dataframe(critical[["Product","AvgInventory","ROP","EOQ"]])
else:
    st.success("No critical ROP breaches detected.")

# Supplier risk panel
if "Supplier" in fdf.columns:
    try:
        sup_risk = compute_supplier_risk(fdf, "Supplier", "Lead_Time_Days")
        st.subheader("Supplier risk scores")
        st.dataframe(sup_risk.head(20))
    except Exception:
        st.info("Supplier risk scoring not available for this dataset.")

# ---------------------------
# Anomaly detection
# ---------------------------
st.markdown("---")
st.header("Anomaly detection (weekly sales z-score)")

try:
    weekly = fdf.set_index("Date").groupby("Product")["Sales"].resample("W").sum().reset_index()
    anomalies = []
    for prod, g in weekly.groupby("Product"):
        arr = g["Sales"].values
        if len(arr) < 6:
            continue
        z = (arr - np.nanmean(arr)) / (np.nanstd(arr) if np.nanstd(arr) != 0 else 1)
        idx = np.where(np.abs(z) > 3)[0]
        for i in idx:
            anomalies.append({"Product": prod, "Week": g.iloc[i]["Date"], "Sales": int(g.iloc[i]["Sales"]), "Z": float(z[i])})
    if anomalies:
        st.warning(f"Detected {len(anomalies)} anomalies in weekly sales.")
        st.dataframe(pd.DataFrame(anomalies).head(200))
    else:
        st.success("No extreme anomalies detected via z-score.")
except Exception:
    st.info("Anomaly detection unavailable for this dataset.")

# ---------------------------
# Explainability (feature importances) if sklearn available
# ---------------------------
st.markdown("---")
st.header("Explainability / feature importance (optional)")

if HAS_SKLEARN:
    try:
        model_df = fdf.copy()
        model_df["Month"] = model_df["Date"].dt.to_period("M").dt.to_timestamp()
        agg = model_df.groupby(["Product","Month"]).agg({
            "Sales":"sum","Inventory":"mean","Lead_Time_Days":"mean","Cost":"mean"
        }).reset_index()
        X = agg[["Inventory","Lead_Time_Days","Cost"]].fillna(0)
        y = agg["Sales"].values
        if len(X) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            feats = X.columns.tolist()
            importances = rf.feature_importances_
            fi = pd.DataFrame({"feature":feats,"importance":importances}).sort_values("importance", ascending=False)
            st.write("Feature importances (RandomForest):")
            st.dataframe(fi)
            fig_fi = px.bar(fi, x="feature", y="importance", title="Feature importances", template=theme_choice)
            st.plotly_chart(fig_fi, use_container_width=True)
            if HAS_SHAP:
                try:
                    explainer = shap.TreeExplainer(rf)
                    shap_vals = explainer.shap_values(X_test.iloc[:100])
                    import matplotlib.pyplot as plt
                    shap.summary_plot(shap_vals, X_test.iloc[:100], show=False)
                    st.pyplot(plt.gcf())
                except Exception:
                    st.info("SHAP visualization failed in this environment.")
        else:
            st.info("Not enough aggregated rows to train a RandomForest for explainability.")
    except Exception as e:
        st.info(f"Explainability skipped: {e}")
else:
    st.info("Install scikit-learn to enable explainability/feature importance.")

# ---------------------------
# Data table & download
# ---------------------------
st.markdown("---")
st.header("Filtered dataset & download")
st.dataframe(fdf.head(200))
st.download_button("Download filtered data (CSV)", fdf.to_csv(index=False).encode("utf-8"), "filtered_supply_data.csv", "text/csv")

st.markdown("---")
st.caption("Enterprise Supply Chain Dashboard — enhanced. Next: PDF export, scheduled alerts, or interactive order simulation. Tell me which to implement next!")
