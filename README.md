# Goldman Sachs - Financial Risk & Transaction Analysis 📊💳

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Harsh-2404/Goldman-Sachs-Financial-Risk-Analysis/blob/main/Goldman_Financial_Risk_Analysis.ipynb)

## 📌 Project Overview
This project performs a detailed financial transaction and risk analysis on customer dataset for Goldman Sachs using Python. The primary objective is to evaluate transactional behavior, track account performance, segment customers, identify financial risks (overdrafts & transaction anomalies), and conduct hypothesis testing.

---

## 📊 Key Analysis & Technical Workflow
- **Task 1: Data Cleaning & Standardisation** — Date format alignment, data type validation, and quality checks across 800 transaction records.
- **Task 2: Descriptive Transactional Analysis** — Monthly/Yearly Credit vs Debit trend tracking and dormant account detection (inactivity > 60 days).
- **Task 3: Customer Profiling** — Transaction activity classification (High/Medium/Low), balance quartile analysis, and behavioral segmentation.
- **Task 4: Financial Risk Identification** — Overdraft frequency tracking, balance volatility evaluation, and Interquartile Range (IQR) anomaly detection.
- **Task 5: Exploratory Data Visualisation** — Custom plots generated using Matplotlib and Seaborn.
- **Task 6: Statistical Hypothesis Testing** — Two-sample independent T-tests evaluating balance variations across transaction volumes and activity levels.

---

## 🔍 Key Insights & Statistical Findings
1. **Net Transaction Flow:** Debit transaction volumes consistently exceeded credit transaction volumes, leading to an overall net negative flow across the evaluation period.
2. **Hypothesis Testing Results:**
   - **High Volume vs Low Volume Balances ($p = 0.9078$):** Failed to reject the null hypothesis ($H_0$). Transaction volume shows no statistically significant impact on average account balance.
   - **High Activity vs Low Activity Balances ($p = 0.8543$):** Failed to reject the null hypothesis ($H_0$). Transaction frequency alone does not imply significantly higher account balance.

---

## 🛠️ Tech Stack Used
- **Language:** Python 3.x
- **Environment:** Google Colab / Jupyter Notebook
- **Data Manipulation:** `pandas`, `numpy`
- **Visualisation:** `matplotlib`, `seaborn`
- **Statistical Inference:** `scipy.stats`

---

## 📁 Repository Structure
- `Goldman_Financial_Risk_Analysis.ipynb` — Full executed Jupyter Notebook containing code, outputs, and visual charts.
- `goldman_sachs.csv` — Raw financial transaction dataset.
- `clean_goldman_sachs.csv` — Cleaned and preprocessed financial dataset.
- `Goldman_Financial_Risk_Analysis_Summary_Report.docx` — Executive summary document with detailed business findings.
- `README.md` — Project documentation and executive summary.

---
*Author: Harsh Srivastav*
