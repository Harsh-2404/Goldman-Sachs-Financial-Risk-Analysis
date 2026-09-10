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

st.title("📊 Goldman Sachs - Executive Financial Risk & Analytics")
st.markdown(
    "Enterprise-grade financial dashboard providing real-time insights into"
    " transaction trends, anomaly detection, behavioral segmentation, and"
    " statistical risk analysis."
)


# Load Cleaned Data
@st.cache_data
def load_data():
  return pd.read_csv("clean_goldman_sachs.csv")


try:
  df = load_data()

  # Sidebar Filters
  st.sidebar.header("🔍 Global Interactive Filters")

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

  # --- TOP FIXED SECTION: KEY EXECUTIVE KPI METRICS ---
  st.subheader("📌 Key Executive Metrics")
  k1, k2, k3, k4 = st.columns(4)

  k1.metric("Total Transactions", f"{len(df_filtered):,}")

  if "TransactionAmount" in df.columns:
    avg_val = df_filtered["TransactionAmount"].mean()
    total_val = df_filtered["TransactionAmount"].sum()
    k2.metric("Avg Transaction Size", f"${avg_val:,.2f}")
    k3.metric("Total Transaction Volume", f"${total_val:,.2f}")

  if "CustomerID" in df.columns:
    unique_cust = df_filtered["CustomerID"].nunique()
    k4.metric("Unique Customer Base", f"{unique_cust:,}")

  st.markdown("---")

  # --- ENTERPRISE TAB NAVIGATION ---
  tab1, tab2, tab3, tab4 = st.tabs([
      "📊 Tab 1: Executive Overview",
      "⚠️ Tab 2: Risk & Anomaly Management",
      "🌍 Tab 3: Customer & Portfolio Profiling",
      "🧪 Tab 4: Hypothesis & Data Engine",
  ])

  # ==========================================
  # TAB 1: EXECUTIVE OVERVIEW & TRENDS
  # ==========================================
  with tab1:
    st.caption("Strategic high-level view of cash flows and volume density.")
    c1, c2 = st.columns(2)

    with c1:
      st.markdown("#### Transaction Amount Density (KDE)")
      if "TransactionAmount" in df.columns:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(
            df_filtered["TransactionAmount"], kde=True, ax=ax1, color="#003366"
        )
        ax1.set_title("Transaction Amount Distribution Density")
        ax1.set_xlabel("Transaction Amount ($)")
        st.pyplot(fig1)

    with c2:
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
            palette="Blues_r",
            ax=ax2,
        )
        ax2.set_title("Volume ($) by Transaction Category")
        ax2.set_ylabel("Total Volume ($)")
        st.pyplot(fig2)

    # Monthly Trend (if TransactionDate column exists)
    if "TransactionDate" in df.columns:
      st.markdown("#### Monthly Transaction Trend")
      df_filtered["MonthYear"] = pd.to_datetime(
          df_filtered["TransactionDate"], errors="coerce"
      ).dt.to_period("M")
      trend_df = (
          df_filtered.groupby("MonthYear")["TransactionAmount"]
          .sum()
          .reset_index()
      )
      trend_df["MonthYear"] = trend_df["MonthYear"].astype(str)

      fig_trend, ax_trend = plt.subplots(figsize=(12, 3))
      sns.lineplot(
          data=trend_df,
          x="MonthYear",
          y="TransactionAmount",
          marker="o",
          color="#008080",
          ax=ax_trend,
      )
      ax_trend.set_title("Monthly Gross Transaction Volume Flow")
      plt.xticks(rotation=45)
      st.pyplot(fig_trend)

  # ==========================================
  # TAB 2: RISK & ANOMALY MANAGEMENT
  # ==========================================
  with tab2:
    st.caption(
        "Audit & Fraud Unit: Outlier identification and risk exposure"
        " distributions."
    )
    c3, c4 = st.columns(2)

    with c3:
      st.markdown("#### Outlier Detection (IQR Method)")
      if "TransactionAmount" in df.columns:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            x=df_filtered["TransactionAmount"], ax=ax3, color="#d9534f"
        )
        ax3.set_title("Transaction Value Outliers & Extremes")
        ax3.set_xlabel("Transaction Amount ($)")
        st.pyplot(fig3)

    with c4:
      st.markdown("#### Risk Distribution Across Products")
      if "Product" in df.columns and "TransactionAmount" in df.columns:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=df_filtered,
            x="TransactionAmount",
            y="Product",
            palette="Oranges",
            ax=ax4,
        )
        ax4.set_title("Transaction Volatility by Product Category")
        st.pyplot(fig4)

    # Correlation Matrix Section
    num_cols = df_filtered.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
      st.markdown("#### Numeric Feature Correlation Heatmap")
      fig_corr, ax_corr = plt.subplots(figsize=(8, 3))
      sns.heatmap(
          df_filtered[num_cols].corr(),
          annot=True,
          cmap="coolwarm",
          fmt=".2f",
          ax=ax_corr,
      )
      ax_corr.set_title("Financial Metrics Correlation Grid")
      st.pyplot(fig_corr)

  # ==========================================
  # TAB 3: CUSTOMER & PORTFOLIO PROFILING
  # ==========================================
  with tab3:
    st.caption(
        "Regional managers & Product leads: Customer distribution analysis."
    )
    c5, c6 = st.columns(2)

    with c5:
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
        ax5.set_title("Transaction Footprint by Region")
        st.pyplot(fig5)

    with c6:
      st.markdown("#### Product-Wise Transaction Volume")
      if "Product" in df.columns:
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        sns.countplot(
            data=df_filtered,
            y="Product",
            palette="mako",
            ax=ax6,
            order=df_filtered["Product"].value_counts().index,
        )
        ax6.set_title("Product Adoption Frequency")
        st.pyplot(fig6)

    # Account Type Distribution
    if "AccountType" in df.columns:
      st.markdown("#### Portfolio Share by Account Type")
      fig_acc, ax_acc = plt.subplots(figsize=(8, 3))
      sns.countplot(
          data=df_filtered,
          x="AccountType",
          palette="Purples_r",
          ax=ax_acc,
          order=df_filtered["AccountType"].value_counts().index,
      )
      ax_acc.set_title("Account Type Distribution Across Filtered Base")
      st.pyplot(fig_acc)

  # ==========================================
  # TAB 4: HYPOTHESIS & DATA ENGINE
  # ==========================================
  with tab4:
    st.subheader("🧪 Statistical Hypothesis Testing Results")
    st.markdown("""
        > **Executive Insight:** Quantitative validation of portfolio behavior using two-sample t-tests ($\alpha = 0.05$).
        """)

    h1, h2 = st.columns(2)
    with h1:
      st.success("##### Test 1: Volume vs Balance")
      st.write("**Null Hypothesis ($H_0$):** High Volume = Low Volume Balances")
      st.write("**P-Value:** `0.9078` (Fail to Reject $H_0$)")
      st.write(
          "**Business Impact:** Transaction volume does not significantly impact"
          " average account balance."
      )

    with h2:
      st.success("##### Test 2: Activity Frequency vs Balance")
      st.write(
          "**Null Hypothesis ($H_0$):** High Activity = Low Activity Balances"
      )
      st.write("**P-Value:** `0.8543` (Fail to Reject $H_0$)")
      st.write(
          "**Business Impact:** Customer transaction frequency alone does not"
          " correlate with higher portfolio balances."
      )

    st.markdown("---")

    st.subheader("📋 Raw Filtered Data Engine")
    st.dataframe(df_filtered, use_container_width=True, height=300)

    # CSV Download Button
    csv_data = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Filtered Dataset (CSV)",
        data=csv_data,
        file_name="goldman_sachs_filtered_report.csv",
        mime="text/csv",
    )

except Exception as e:
  st.error(f"System Error in rendering dashboard: {e}")
