# 🏦 Credit Risk Modelling: End-to-End Quantitative Risk Engineering Pipeline

`CreditRiskEngine` is an end-to-end, production-grade quantitative credit risk modeling, econometric analysis, and explainability pipeline built to comply with **Basel III Internal Ratings-Based (IRB)** regulatory frameworks.

The entire project is structured as a fully pre-executed **Jupyter Notebook ([`CreditRiskEngine.ipynb`](file:///c:/Users/BIBEK/OneDrive/Desktop/credit%20risk%20modelling/CreditRiskEngine/CreditRiskEngine.ipynb))** executed directly on institutional credit datasets (`loan_portfolio.csv` with 50,000 corporate/retail accounts, `vintage_analysis.csv`, and `macro_stress_scenarios.csv`). All code cells, rich markdown explanations, LaTeX formulas, data tables, and dark-mode visual plots are **100% pre-rendered** for immediate visualization on GitHub.

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

## 📋 The 8 Quantitative Credit Risk Modelling Steps

The centerpiece notebook **`CreditRiskEngine.ipynb`** follows the **8 Formal Credit Risk Modelling Steps**:

1. **Step 1: Credit Risk Modelling: Define 'Bad'**:
   - Monthly Delinquency Roll Rate transition matrix ($4 \times 4$ DPD states) & empirical justification of **90-DPD** as default threshold.
2. **Step 2: Define Observation vs Performance Window**:
   - Cohort Vintage Analysis cumulative default curves & empirical justification of the **18-month** performance window plateau.
3. **Step 3: Choose Data Sources & Prepare 'Driver Set'**:
   - Flexible column normalization (`initial_rating` $\to$ `grade`, `coupon_rate` $\to$ `int_rate`, `debt_to_equity` $\to$ `dti`, `defaulted` $\to$ `default_label`).
   - Median missing value imputation and target class imbalance assessment.
4. **Step 4: Feature Engineering, Weight of Evidence (WoE), IV & VIF**:
   - Risk ratios (`leverage_to_ic`, `ead_log`).
   - Weight of Evidence (WoE) & Information Value filter ($IV < 0.02$ drop filter & ranking bar plot).
   - Multicollinearity control via iterative Variance Inflation Factor ($VIF > 5.0$ drop filter).
5. **Step 5: Choosing Modelling Technique & Calibration**:
   - Formal Breusch-Pagan heteroskedasticity & Durbin-Wu-Hausman endogeneity tests for automated OLS/WLS/2SLS linear estimator selection.
   - Machine Learning PD models: Logistic Regression, XGBoost Classifier, PyTorch Deep Neural Network (MLP).
   - Post-hoc Isotonic Probability Calibration & reliability histogram (Brier score evaluation).
   - Loss Given Default (LGD) and Exposure at Default (EAD) XGBoost regressors.
6. **Step 6: Evaluating Model Performance & Business Impact (IRB Calculator)**:
   - Kolmogorov-Smirnov (KS) statistic ($42.6\%$) & cumulative Goods vs Bads separation curves.
   - ROC-AUC ($0.7741$) & Gini index.
   - Portfolio-level Basel III IRB Capital Provisions ($EL = \$385.29\text{M}$, $K$, and $RWA = \$3.12\text{B}$).
7. **Step 7: Ongoing Model Monitoring, SHAP & Stress Testing**:
   - Population Stability Index (PSI) score drift decile bar chart ($0.0038 < 0.10$).
   - SHAP global feature importance beeswarm plot and local borrower waterfall plot.
   - Integration of `macro_stress_scenarios.csv` for Pillar 2 downturn stress testing (Baseline, Mild Recession, Severe Recession, Stagflation).
8. **Step 8: Reject Inferencing**:
   - Fuzzy Augmentation reweighting for accepted vs rejected population selection bias correction.
9. **🧮 Single Borrower Underwriting Simulator**:
   - Programmatic credit decision engine scoring individual borrower applications and outputting $PD$, $LGD$, $EAD$, $EL$, $RWA$, CET1 capital requirement, and approval/rejection decision.

---

## ⚡ Execution Instructions

Open `CreditRiskEngine.ipynb` in VS Code, JupyterLab, or Google Colab, or view the pre-executed notebook directly on GitHub!
