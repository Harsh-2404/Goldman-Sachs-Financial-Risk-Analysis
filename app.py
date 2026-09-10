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

st.title("📊 Goldman Sachs - Financial Risk & Anomaly Analysis")
st.markdown(
    "Interactive Financial Risk Dashboard analyzing transaction trends, customer"
    " behavioral segmentation, anomaly detection, and statistical hypothesis"
    " testing."
)


# Load Cleaned Data
@st.cache_data
def load_data():
  return pd.read_csv("clean_goldman_sachs.csv")


try:
  df = load_data()

  # Sidebar Filters
  st.sidebar.header("🔍 Interactive Filters")

  # Filter 1: Account Type
  acc_types = (
      df["AccountType"].dropna().unique().tolist()
      if "AccountType" in df.columns
      else []
  )
  selected_acc = st.sidebar.multiselect(
      "Select Account Type", acc_types, default=acc_types
  )

  # Filter 2: Region
  regions = (
      df["Region"].dropna().unique().tolist() if "Region" in df.columns else []
  )
  selected_region = st.sidebar.multiselect(
      "Select Region", regions, default=regions
  )

  # Filter 3: Product
  products = (
      df["Product"].dropna().unique().tolist() if "Product" in df.columns else []
  )
  selected_products = st.sidebar.multiselect(
      "Select Product", products, default=products
  )

  # Apply Filters
  df_filtered = df.copy()
  if selected_acc and "AccountType" in df.columns:
    df_filtered = df_filtered[df_filtered["AccountType"].isin(selected_acc)]
  if selected_region and "Region" in df.columns:
    df_filtered = df_filtered[df_filtered["Region"].isin(selected_region)]
  if selected_products and "Product" in df.columns:
    df_filtered = df_filtered[df_filtered["Product"].isin(selected_products)]

  # --- Section 1: Executive KPI Metrics ---
  st.subheader("📌 Key Executive Metrics")
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

  # --- Section 2: Task 2 & 3 - Transactional Analysis & Distribution ---
  st.subheader("📈 Section 1: Transaction & Balance Distributions")
  col1, col2 = st.columns(2)

  with col1:
    st.markdown("#### Transaction Amount Density (KDE)")
    if "TransactionAmount" in df.columns:
      fig1, ax1 = plt.subplots(figsize=(6, 4))
      sns.histplot(
          df_filtered["TransactionAmount"], kde=True, ax=ax1, color="#1f77b4"
      )
      ax1.set_title("Distribution of Transaction Amounts")
      ax1.set_xlabel("Transaction Amount ($)")
      st.pyplot(fig1)

  with col2:
    st.markdown("#### Transaction Volume by Type")
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

  # --- Section 3: Task 4 - Financial Risk & Outliers ---
  st.subheader("⚠️ Section 2: Financial Risk & Anomaly Detection")
  col3, col4 = st.columns(2)

  with col3:
    st.markdown("#### Outlier Detection (Boxplot)")
    if "TransactionAmount" in df.columns:
      fig3, ax3 = plt.subplots(figsize=(6, 4))
      sns.boxplot(
          x=df_filtered["TransactionAmount"], ax=ax3, color="#d62728"
      )
      ax3.set_title("Transaction Amount Outlier Analysis (IQR Method)")
      ax3.set_xlabel("Transaction Amount ($)")
      st.pyplot(fig3)

  with col4:
    st.markdown("#### Risk Distribution Across Products")
    if "Product" in df.columns and "TransactionAmount" in df.columns:
      fig4, ax4 = plt.subplots(figsize=(6, 4))
      sns.boxplot(
          data=df_filtered,
          x="TransactionAmount",
          y="Product",
          palette="Set2",
          ax=ax4,
      )
      ax4.set_title("Transaction Amount Spread by Financial Product")
      st.pyplot(fig4)

  st.markdown("---")

  # --- Section 4: Task 3 & 5 - Segmentation & Demographics ---
  st.subheader("🌍 Section 3: Customer & Regional Profiling")
  col5, col6 = st.columns(2)

  with col5:
    st.markdown("#### Regional Breakdown")
    if "Region" in df.columns:
      fig5, ax5 = plt.subplots(figsize=(6, 4))
      sns.countplot(
          data=df_filtered,
          x="Region",
          palette="viridis",
          ax=ax5,
          order=df_filtered["Region"].value_counts().index,
      )
      ax5.set_title("Transaction Count by Region")
      st.pyplot(fig5)

  with col6:
    st.markdown("#### Product-Wise Volume")
    if "Product" in df.columns:
      fig6, ax6 = plt.subplots(figsize=(6, 4))
      sns.countplot(
          data=df_filtered,
          y="Product",
          palette="mako",
          ax=ax6,
          order=df_filtered["Product"].value_counts().index,
      )
      ax6.set_title("Transaction Frequency by Product")
      st.pyplot(fig6)

  st.markdown("---")

  # --- Section 5: Task 6 - Statistical Hypothesis Testing Results ---
  st.subheader("🧪 Section 4: Statistical Hypothesis Testing Summary")
  st.info("""
    **Hypothesis Test 1:** High Transaction Volume vs Low Transaction Volume Balances
    - **Result:** $p = 0.9078$ (Fail to reject $H_0$)
    - **Insight:** Transaction volume shows no statistically significant impact on average customer account balance.

    **Hypothesis Test 2:** High Activity vs Low Activity Customer Balances
    - **Result:** $p = 0.8543$ (Fail to reject $H_0$)
    - **Insight:** Higher transaction frequency alone does not imply significantly higher account balance.
    """)

  st.markdown("---")

  # --- Section 6: Dataset Table ---
  st.subheader("📋 Section 5: Filtered Data Preview")
  st.dataframe(df_filtered.head(100), use_container_width=True)

except Exception as e:
  st.error(f"Error loading dashboard: {e}")
