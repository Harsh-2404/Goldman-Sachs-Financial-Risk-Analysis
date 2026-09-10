import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Page Setup
st.set_page_config(
    page_title="Goldman Sachs Financial Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Goldman Sachs - Financial Risk & Anomaly Analysis")
st.markdown(
    "Interactive Financial Risk Dashboard analyzing customer transaction trends,"
    " regional distributions, and risk metrics."
)


# Load Data
@st.cache_data
def load_data():
  return pd.read_csv("clean_goldman_sachs.csv")


try:
  df = load_data()

  # Sidebar Filters
  st.sidebar.header("🔍 Filter Options")

  # Account Type Filter
  acc_types = (
      df["AccountType"].dropna().unique().tolist()
      if "AccountType" in df.columns
      else []
  )
  selected_acc = st.sidebar.multiselect(
      "Select Account Type", acc_types, default=acc_types
  )

  # Region Filter
  regions = (
      df["Region"].dropna().unique().tolist() if "Region" in df.columns else []
  )
  selected_region = st.sidebar.multiselect(
      "Select Region", regions, default=regions
  )

  # Filter Data
  df_filtered = df.copy()
  if selected_acc and "AccountType" in df.columns:
    df_filtered = df_filtered[df_filtered["AccountType"].isin(selected_acc)]
  if selected_region and "Region" in df.columns:
    df_filtered = df_filtered[df_filtered["Region"].isin(selected_region)]

  # High Level Metrics (KPIs)
  st.subheader("📌 Key Financial Metrics")
  k1, k2, k3, k4 = st.columns(4)

  k1.metric("Total Transactions", f"{len(df_filtered):,}")

  if "TransactionAmount" in df.columns:
    avg_val = df_filtered["TransactionAmount"].mean()
    total_val = df_filtered["TransactionAmount"].sum()
    k2.metric("Average Transaction", f"${avg_val:,.2f}")
    k3.metric("Total Volume", f"${total_val:,.2f}")

  if "CustomerID" in df.columns:
    unique_cust = df_filtered["CustomerID"].nunique()
    k4.metric("Unique Customers", f"{unique_cust:,}")

  st.markdown("---")

  # Visualizations
  col1, col2 = st.columns(2)

  with col1:
    st.subheader("📈 Transaction Amount Distribution")
    if "TransactionAmount" in df.columns:
      fig1, ax1 = plt.subplots(figsize=(6, 4))
      sns.histplot(
          df_filtered["TransactionAmount"], kde=True, ax=ax1, color="#1f77b4"
      )
      ax1.set_title("Distribution of Transaction Amounts")
      ax1.set_xlabel("Transaction Amount ($)")
      st.pyplot(fig1)

  with col2:
    st.subheader("💳 Volume by Transaction Type")
    if "TransactionType" in df.columns and "TransactionAmount" in df.columns:
      type_summary = (
          df_filtered.groupby("TransactionType")["TransactionAmount"]
          .sum()
          .reset_index()
      )
      fig2, ax2 = plt.subplots(figsize=(6, 4))
      sns.barplot(
          data=type_summary,
          x="TransactionType",
          y="TransactionAmount",
          palette="Blues_d",
          ax=ax2,
      )
      ax2.set_title("Total Amount by Transaction Type")
      ax2.set_ylabel("Total Amount ($)")
      st.pyplot(fig2)

  st.markdown("---")

  # Second Row Visualizations
  col3, col4 = st.columns(2)

  with col3:
    st.subheader("🌍 Regional Transaction Breakup")
    if "Region" in df.columns:
      fig3, ax3 = plt.subplots(figsize=(6, 4))
      sns.countplot(
          data=df_filtered,
          x="Region",
          palette="viridis",
          ax=ax3,
          order=df_filtered["Region"].value_counts().index,
      )
      ax3.set_title("Transaction Count by Region")
      st.pyplot(fig3)

  with col4:
    st.subheader("📦 Product Wise Distribution")
    if "Product" in df.columns:
      fig4, ax4 = plt.subplots(figsize=(6, 4))
      sns.countplot(
          data=df_filtered,
          y="Product",
          palette="mako",
          ax=ax4,
          order=df_filtered["Product"].value_counts().index,
      )
      ax4.set_title("Transaction Count by Product")
      st.pyplot(fig4)

  st.markdown("---")

  # Table
  st.subheader("📋 Dataset Preview")
  st.dataframe(df_filtered.head(100), use_container_width=True)

except Exception as e:
  st.error(f"Error loading dashboard: {e}")
