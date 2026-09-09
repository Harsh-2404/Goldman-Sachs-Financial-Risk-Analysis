import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Goldman Sachs Financial Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

# Title & Overview
st.title("📊 Goldman Sachs - Financial Risk & Anomaly Analysis")
st.markdown(
    "Interactive Financial Risk Dashboard analyzing customer transaction trends,"
    " account balances, volatility, and anomaly detection."
)


# Load Cleaned Data
@st.cache_data
def load_data():
  return pd.read_csv("clean_goldman_sachs.csv")


try:
  df = load_data()

  # Sidebar Controls
  st.sidebar.header("🔍 Filter Options")

  # Account Type Filter (if present, else fallback)
  if "account_type" in df.columns:
    acc_types = df["account_type"].unique().tolist()
    selected_acc = st.sidebar.multiselect(
        "Select Account Type", acc_types, default=acc_types
    )
    df_filtered = df[df["account_type"].isin(selected_acc)]
  else:
    df_filtered = df.copy()

  # High-Level KPIs
  st.subheader("📌 Key Financial Metrics")
  kpi1, kpi2, kpi3, kpi4 = st.columns(4)

  total_records = len(df_filtered)
  kpi1.metric("Total Transactions", f"{total_records:,}")

  if "account_balance" in df.columns:
    avg_balance = df_filtered["account_balance"].mean()
    kpi2.metric("Average Balance", f"${avg_balance:,.2f}")

  if "debit_amount" in df.columns:
    total_debit = df_filtered["debit_amount"].sum()
    kpi3.metric("Total Debits", f"${total_debit:,.2f}")

  if "credit_amount" in df.columns:
    total_credit = df_filtered["credit_amount"].sum()
    kpi4.metric("Total Credits", f"${total_credit:,.2f}")

  st.markdown("---")

  # Visualizations Layout
  col1, col2 = st.columns(2)

  with col1:
    st.subheader("📈 Balance Distribution")
    if "account_balance" in df.columns:
      fig, ax = plt.subplots(figsize=(6, 4))
      sns.histplot(
          df_filtered["account_balance"], kde=True, ax=ax, color="#1f77b4"
      )
      ax.set_title("Customer Account Balance Density")
      st.pyplot(fig)

  with col2:
    st.subheader("💳 Debit vs Credit Flow")
    if "debit_amount" in df.columns and "credit_amount" in df.columns:
      flow_data = pd.DataFrame({
          "Type": ["Total Debits", "Total Credits"],
          "Amount": [
              df_filtered["debit_amount"].sum(),
              df_filtered["credit_amount"].sum(),
          ],
      })
      fig2, ax2 = plt.subplots(figsize=(6, 4))
      sns.barplot(
          data=flow_data, x="Type", y="Amount", palette="Blues_d", ax=ax2
      )
      ax2.set_title("Total Transaction Volume Flow")
      st.pyplot(fig2)

  st.markdown("---")

  # Raw Data View
  st.subheader("📋 Dataset Preview")
  st.dataframe(df_filtered.head(50), use_container_width=True)

except Exception as e:
  st.error(
      f"Data file 'clean_goldman_sachs.csv' not found or error loading data:"
      f" {e}"
  )
