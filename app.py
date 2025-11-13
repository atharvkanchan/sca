# Streamlit Supply Chain Dashboard
# Auto-generated deployment package

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

CSV_PATH = "supply_chain_clean_deploy_ready.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    return df

df = load_data()
st.title("Supply Chain Dashboard - Deployment Ready")
st.dataframe(df)
