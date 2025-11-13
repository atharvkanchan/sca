# charts.py
# Charting utilities for the AI Supply Chain Dashboard

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def plot_sales_trend(df: pd.DataFrame):
    st.subheader("Sales Trend")
    ts = df.groupby(pd.Grouper(key="Date", freq="M"))["Sales"].sum().reset_index().sort_values("Date")
    if ts.empty:
        st.info("Not enough data for sales trend.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["Date"], y=ts["Sales"], mode="lines+markers", name="Sales", line=dict(width=3)))
    fig.update_layout(title="Monthly Sales Trend", xaxis_title="Date", yaxis_title="Sales", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def plot_inventory_by_product(df: pd.DataFrame):
    st.subheader("Inventory by Product")
    inv = df.groupby("Product")["Inventory"].mean().reset_index().sort_values("Inventory", ascending=False)
    if inv.empty:
        st.info("No inventory data to plot.")
        return
    fig = px.bar(inv, x="Product", y="Inventory", title="Average Inventory per Product", text="Inventory")
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def plot_supplier_pie(df: pd.DataFrame, supplier_col: str):
    st.subheader("Supplier Sales Distribution")
    sup = df.groupby(supplier_col)["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
    if sup.empty:
        st.info("No supplier sales data.")
        return
    fig = px.pie(sup, names=supplier_col, values="Sales", title="Sales by Supplier", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
