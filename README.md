# CreditRiskEngine: Basel III Compliant Risk Modeling & Explainability

`CreditRiskEngine` is an end-to-end, production-grade credit risk modeling and explainability pipeline built to comply with Basel III regulatory frameworks. The system is provided as a pre-executed **Jupyter Notebook ([`CreditRiskEngine.ipynb`](file:///c:/Users/BIBEK/OneDrive/Desktop/credit%20risk%20modelling/CreditRiskEngine/CreditRiskEngine.ipynb))** trained on institutional credit datasets (`loan_portfolio.csv`, `vintage_analysis.csv`, `macro_stress_scenarios.csv`), featuring pre-rendered plots, charts, tables, and execution logs for immediate visualization on GitHub, alongside a modular 6-tab Streamlit dashboard (`app.py`).

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

---

## 🏗️ System Architecture & File Tree

```
CreditRiskEngine/
├── CreditRiskEngine.ipynb  # Main Executed Jupyter Notebook with pre-rendered plots & metrics
├── loan_portfolio.csv      # Institutional loan portfolio (50,000 corporate/retail accounts)
├── vintage_analysis.csv    # Cohort vintage default rates (2,162 monthly records)
├── macro_stress_scenarios.csv # Macroeconomic Downturn Scenarios & Shock Impact
├── config.py               # Dataclass for VIF=5.0, IV=0.02, performance window, etc.
├── requirements.txt        # Complete Python dependencies
├── README.md               # Technical documentation & setup guide
├── validate_pipeline.py    # Comprehensive 8-step backend test suite
├── src/
│   ├── __init__.py
│   ├── data_prep.py        # DataOrchestrator: CSV upload, Roll Rates, Vintage, Reject Inference
│   ├── econometrics.py      # EconometricEngine: WoE/IV, VIF=5.0 filter, OLS/WLS/2SLS auto-selection
│   ├── models.py            # RiskModels: Logistic Regression, XGBoost, PyTorch MLP, LGD, EAD
│   ├── evaluate.py          # RiskEvaluator: KS-Statistic, PSI data drift, Basel III EL/RWA
│   └── explain.py           # ShapExplainer: Unified SHAP local explanations across all 3 models
└── app.py                  # Streamlit Dashboard & Underwriting Simulator
```

---

## 🔬 Core Components & Logic

### 1. Data Ingestion & Target Definition (`src/data_prep.py`)
- **Custom CSV Upload**: Accepts any user-uploaded CSV via `st.file_uploader` or falls back gracefully to a synthetic Lending Club-style dataset.
- **Roll Rate Analysis**: Evaluates monthly delinquency transition matrices across DPD states (Current, 30 DPD, 60 DPD, 90 DPD). When transition rates from 60 to 90 DPD exceed 80%, rehabilitation drops to zero, justifying **90 DPD** as the objective default definition.
- **Vintage Analysis**: Identifies the performance window by plotting cumulative default rates across origination cohorts until the curve plateaus (marginal rate < 0.1%/month), setting the performance window to **18 months**.
- **Reject Inference**: Corrects selection bias via **Fuzzy Augmentation**. Rejections are scored by an accepted-population model and duplicated with complementary Good/Bad weights ($1-p$ and $p$).

### 2. Econometric & Feature Selection (`src/econometrics.py`)
- **Information Value (IV)**: Filters out weak features with $IV < 0.02$.
- **Multicollinearity Control (VIF)**: Iteratively eliminates features with $VIF > 5.0$ per Basel III guidelines. Returns a detailed drop report citing exact numeric VIF and IV values.
- **Automated Linear Regression Selection**:
  - **Breusch-Pagan Test**: Checks for heteroskedasticity. If detected ($p < 0.05$), fits **Weighted Least Squares (WLS)** via FGLS.
  - **Durbin-Wu-Hausman Test**: Checks for endogeneity. If detected ($p < 0.05$), fits **Two-Stage Least Squares (2SLS)** using instrumental variables.
  - **OLS Baseline**: Used when assumptions hold (BLUE).

### 3. Three Modeling Engines & Calibration (`src/models.py`)
- **Logistic Regression**: Regulatory benchmark model (L2 regularised).
- **XGBoost Classifier**: High-capacity gradient boosted decision tree.
- **PyTorch MLP**: Deep neural network wrapper with Dropout and Adam optimizer.
- **LGD & EAD Sub-models**: Separate regressors predicting Loss Given Default and Exposure at Default.
- **Isotonic Calibration**: Post-hoc probability calibration aligning ML outputs to empirical default frequencies, measured via Brier score.

### 4. Diagnostics & Explainability (`src/evaluate.py` & `src/explain.py`)
- **KS-Statistic Plot**: Cumulative Goods vs. Bads distribution curves measuring class separation.
- **Population Stability Index (PSI)**: Bucket-level score drift analysis between training and validation populations.
- **SHAP Explainer**: Unifies local borrower explainability across all 3 engines using `LinearExplainer`, `TreeExplainer`, or `KernelExplainer`.

---

## 🖥️ Streamlit Dashboard (5 Workspace Tabs)

1. **📊 IRB Calculator**: Displays predicted PD, LGD, EAD, Expected Loss, and Risk-Weighted Assets with an interactive gauge and automated econometric decision log.
2. **🔄 Target Definition**: Renders the **Roll Rate transition matrix heatmap** and **Vintage Analysis cumulative default curves** to justify the default target and performance window.
3. **🔍 Feature Selection Report**: Interactive table displaying exact features dropped, citing strict econometric reasons (e.g. `VIF = 12.5 > 5.0` or `IV = 0.01 < 0.02`), plus an IV bar chart.
4. **📈 Model Diagnostics**: Visualizes the **KS-Statistic cumulative curves** and **PSI bucket decomposition bar chart** alongside ROC-AUC and Gini metrics.
5. **💡 Explainability & Stress Testing**: Renders a dynamic **SHAP horizontal waterfall plot** for the simulated borrower and interactive macroeconomic sliders (income drop, interest rate spike, GDP slowdown) for Pillar 2 stress testing.

---

## 🚀 Installation & Running

### Installation
```powershell
pip install -r requirements.txt
```

### Run Pipeline Verification
```powershell
python validate_pipeline.py
```

### Launch Streamlit Dashboard
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
