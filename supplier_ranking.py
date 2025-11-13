# supplier_ranking.py
# Supplier performance scoring, ranking table and radar chart

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import safe_series, normalize, safe_col


def show_supplier_ranking(df: pd.DataFrame, supplier_col: str):
    st.subheader("Supplier Performance Ranking")
    sup = df.copy()

    # ensure fields exist
    for f in ["on_time_delivery", "actual_lead_days", "damage_rate", "return_rate", "total_cost_INR"]:
        if f not in sup.columns:
            sup[f] = safe_series(sup, f, 0 if "rate" in f else sup.get("Cost", 0) * sup.get("Sales", 0))

    stats = sup.groupby(supplier_col).agg({
        "on_time_delivery": "mean",
        "actual_lead_days": "mean",
        "damage_rate": "mean",
        "return_rate": "mean",
        "total_cost_INR": "mean",
        "Sales": "sum"
    }).reset_index()

    stats["Cost_Efficiency"] = stats["total_cost_INR"] / (stats["Sales"] + 1e-6)

    stats["OnTimeScore"] = normalize(stats["on_time_delivery"])
    stats["LeadScore"] = 1 - normalize(stats["actual_lead_days"])
    stats["DamageScore"] = 1 - normalize(stats["damage_rate"])
    stats["ReturnScore"] = 1 - normalize(stats["return_rate"])
    stats["CostScore"] = 1 - normalize(stats["Cost_Efficiency"])

    stats["Performance_Score"] = (
        0.40 * stats["OnTimeScore"] +
        0.20 * stats["LeadScore"] +
        0.20 * stats["DamageScore"] +
        0.10 * stats["ReturnScore"] +
        0.10 * stats["CostScore"]
    )

    stats = stats.sort_values("Performance_Score", ascending=False)

    st.dataframe(stats[[supplier_col, "Performance_Score", "on_time_delivery", "actual_lead_days", "damage_rate", "return_rate", "Cost_Efficiency"]].style.format({"Performance_Score":"{:.4f}", "on_time_delivery":"{:.2f}", "actual_lead_days":"{:.1f}", "damage_rate":"{:.3f}", "return_rate":"{:.3f}", "Cost_Efficiency":"{:.2f}"}))

    st.subheader("Ranking Chart")
    fig = px.bar(stats, x=supplier_col, y="Performance_Score", color="Performance_Score", title="Supplier Performance Score")
    fig.update_traces(texttemplate="%{y:.3f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance Radar Chart")
    supplier_list = stats[supplier_col].tolist()
    selected = st.selectbox("Select Supplier", supplier_list)
    row = stats[stats[supplier_col] == selected].iloc[0]

    metrics = [row["OnTimeScore"], row["LeadScore"], row["DamageScore"], row["ReturnScore"], row["CostScore"]]
    labels = ["On-Time", "Lead Time", "Damage", "Return", "Cost Eff."]

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=metrics, theta=labels, fill='toself', name=selected))
    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=False)
    st.plotly_chart(fig_r, use_container_width=True)
