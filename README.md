# 🏦 CreditRiskEngine: Basel III Compliant Risk Modeling & Explainability

`CreditRiskEngine` is an end-to-end, production-grade quantitative credit risk modeling, econometric analysis, and explainability pipeline built to comply with **Basel III Internal Ratings-Based (IRB)** regulatory frameworks.

The entire project is structured as a fully pre-executed **Jupyter Notebook ([`CreditRiskEngine.ipynb`](file:///c:/Users/BIBEK/OneDrive/Desktop/credit%20risk%20modelling/CreditRiskEngine/CreditRiskEngine.ipynb))** executed directly on institutional credit datasets (`loan_portfolio.csv` with 50,000 corporate/retail accounts, `vintage_analysis.csv`, and `macro_stress_scenarios.csv`). All code cells, markdown descriptions, LaTeX formulas, data tables, and dark-mode visual plots are **100% pre-rendered** for immediate visualization on GitHub.

---

## 🏛️ Basel III Regulatory Capital Framework

Under the **Basel III framework**, financial institutions use the **Internal Ratings-Based (IRB)** approach to calculate regulatory capital requirements:

> **Expected Loss (EL)** is covered by credit impairment provisions (IFRS 9 / CECL), whereas **Unexpected Loss (UL)** is absorbed by Common Equity Tier 1 (CET1) capital:
>
> $$EL = PD \times LGD \times EAD$$
>
> - **PD (Probability of Default)**: The 12-month empirical probability that a borrower defaults.
> - **LGD (Loss Given Default)**: The proportion of exposure lost upon default after recovery costs.
> - **EAD (Exposure at Default)**: Total gross exposure outstanding when default occurs.
> - **Risk-Weighted Assets (RWA)**: Calculated under BCBS asset correlation formulas ($R$) and Vasicek capital requirements ($K$).

---

## 📁 Repository Structure

```
CreditRiskEngine/
├── CreditRiskEngine.ipynb     # Main Executed Jupyter Notebook (Pre-rendered plots & metrics)
├── loan_portfolio.csv         # Institutional loan portfolio (50,000 corporate/retail accounts)
├── vintage_analysis.csv       # Cohort vintage default rates (2,162 monthly records)
├── macro_stress_scenarios.csv # Macroeconomic Downturn Scenarios & Shock Impacts
├── credit_ratings.csv         # Credit rating mappings & historical default rates
├── portfolio_metrics.csv      # Executive portfolio summary metrics
└── README.md                  # Project documentation & technical architecture
```

---

## 📋 Jupyter Notebook Section Mapping

The centerpiece notebook **`CreditRiskEngine.ipynb`** covers all quantitative workflow phases matching institutional risk dashboard tabs:

1. **System Setup & Data Ingestion**:
   - Environment initialization (`pandas`, `numpy`, `statsmodels`, `xgboost`, `torch`, `shap`).
   - Config setup ($VIF=5.0$, $IV=0.02$, $Window=18$ months).
   - Ingestion of `loan_portfolio.csv` (50,000 accounts) with flexible alias mapping (`initial_rating` $\to$ `grade`, `defaulted` $\to$ `default_label`) and missing value median imputation.

2. **Tab 1: 📈 Exploratory Data Analysis (EDA)**:
   - Target class imbalance donut chart.
   - Pearson feature correlation matrix heatmap.
   - Bivariate default rate variations across Credit Grades and Credit Score vs Leverage scatterplot.

3. **Tab 2: 🎯 Target Definition & Retrospective Analysis**:
   - Monthly Delinquency Roll Rate transition matrix ($4 \times 4$ DPD states) justifying **90-DPD** target selection.
   - Cohort Vintage Analysis cumulative default curves justifying the **18-month** performance window.

4. **Tab 3: 🔍 Econometric Feature Selection & Auto-Regression**:
   - Weight of Evidence (WoE) & Information Value filter ($IV < 0.02$ drop filter & IV ranking bar chart).
   - Multicollinearity control via iterative Variance Inflation Factor ($VIF > 5.0$ drop filter).
   - Formal Breusch-Pagan heteroskedasticity and Durbin-Wu-Hausman endogeneity tests.
   - Automated selection of OLS, WLS (via FGLS), or 2SLS linear estimator.

5. **Tab 4: 🤖 Model Training & Reliability Calibration**:
   - Stratified 70/30 Train/Test dataset splitting.
   - Model 1: Logistic Regression (Regulatory standard baseline).
   - Model 2: XGBoost Gradient Boosted Classifier.
   - Model 3: PyTorch Deep Neural Network (MLP).
   - Post-hoc Isotonic Probability Calibration & reliability histogram (Brier score evaluation).
   - Component LGD and EAD XGBoost regressors.

6. **Tab 5: 📊 Model Diagnostics & Basel III IRB Calculator**:
   - Kolmogorov-Smirnov (KS) statistic ($42.6\%$) & cumulative Goods vs Bads separation curves.
   - ROC-AUC ($0.7741$) & Gini index.
   - Population Stability Index (PSI) score drift decile bar chart ($0.0038$).
   - Portfolio-level Expected Loss ($EL$), Vasicek Capital Requirement ($K$), and Risk-Weighted Assets ($RWA$).

7. **Tab 6: 🔬 Explainability & Macroeconomic Stress Testing**:
   - SHAP (SHapley Additive exPlanations) global feature importance beeswarm plot and local borrower waterfall plot.
   - Integration of `macro_stress_scenarios.csv` for Pillar 2 macroeconomic downturn stress testing (Baseline, Mild Recession, Severe Recession, Stagflation) & shock impact bar chart.

8. **Tab 7: 🧮 Individual Borrower Underwriting & IRB Simulator**:
   - Programmatic credit decision engine scoring individual borrower applications and outputting $PD$, $LGD$, $EAD$, $EL$, $RWA$, CET1 capital requirement, and approval/rejection decision.

---

## ⚡ Execution Instructions

Open `CreditRiskEngine.ipynb` in VS Code, JupyterLab, or Google Colab, or simply view the pre-executed notebook directly on GitHub!
