"""
CreditRiskEngine v2 — Basel III Compliant Risk Modeling Dashboard
=================================================================
A production-grade Streamlit application exposing the full data science
workflow: CSV upload → target definition → feature selection → modelling
→ diagnostics → explainability & stress testing.
"""

import io
import sys
import os

# Ensure src/ is on the path when running from the project root
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import SystemConfig
from src.data_prep   import DataOrchestrator
from src.econometrics import EconometricEngine
from src.models       import RiskModels
from src.evaluate     import RiskEvaluator
from src.explain      import ShapExplainer

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditRiskEngine — Basel III",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Inter font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

  /* Hero gradient header */
  .hero-header {
      background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
      padding: 2rem 2.5rem;
      border-radius: 12px;
      margin-bottom: 1.5rem;
      color: white;
  }
  .hero-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
  .hero-header p  { font-size: 0.95rem; opacity: 0.85; margin: 0.4rem 0 0; }

  /* Metric cards */
  .metric-card {
      background: linear-gradient(135deg, #1a1a2e, #16213e);
      border: 1px solid #2a4a6b;
      border-radius: 10px;
      padding: 1.2rem 1.5rem;
      text-align: center;
      color: white;
  }
  .metric-card .label { font-size: 0.75rem; color: #8ab4d4; text-transform: uppercase; letter-spacing: 1px; }
  .metric-card .value { font-size: 2rem; font-weight: 700; margin: 0.3rem 0; }
  .metric-card .sub   { font-size: 0.75rem; color: #aaa; }

  /* Rationale boxes */
  .rationale-box {
      background: #0d1b2a;
      border-left: 4px solid #1e88e5;
      padding: 0.9rem 1.2rem;
      border-radius: 0 8px 8px 0;
      margin: 0.5rem 0;
      font-size: 0.88rem;
      color: #ccc;
  }

  /* Section headers */
  .section-title {
      font-size: 1.1rem;
      font-weight: 600;
      color: #e0e0e0;
      border-bottom: 2px solid #1e88e5;
      padding-bottom: 6px;
      margin-bottom: 1rem;
  }

  /* Drop report table */
  .drop-table-header { color: #fff; background: #1e3a5f; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner (cached by file hash + model type)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Running full Basel III pipeline…")
def run_pipeline(
    file_bytes: bytes | None,
    model_engine: str
) -> dict:
    """Executes the complete end-to-end risk pipeline and returns all artefacts."""
    config  = SystemConfig()
    orch    = DataOrchestrator(config)
    engine  = EconometricEngine(config)
    risk_m  = RiskModels(config)
    evaluator = RiskEvaluator()

    # ── 1. Data loading ──────────────────────────────────────────────────────
    uploaded_df = None
    if file_bytes:
        try:
            uploaded_df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            uploaded_df = None

    datasets, load_rationale = orch.load_data(uploaded_df)
    static_df = datasets["static"]

    # ── 2. Roll rates ────────────────────────────────────────────────────────
    roll_matrix, roll_rationale = orch.compute_roll_rates(datasets)

    # ── 3. Vintage analysis ──────────────────────────────────────────────────
    vintage_pivot, vintage_rationale = orch.perform_vintage_analysis(datasets)

    # ── 4. Feature selection (IV + VIF) ──────────────────────────────────────
    target = "default_label"
    admin_cols = ["loan_id", "ead", "lgd", "macro_instrument"]
    filtered_df, drop_report = engine.feature_selection(
        static_df, target, exclude_cols=admin_cols
    )
    selected_features = [
        c for c in filtered_df.columns
        if c not in [target] + admin_cols
    ]

    # ── 5. WoE / IV table ────────────────────────────────────────────────────
    woe_iv_dict = engine.calculate_woe_iv(static_df, target)

    # ── 6. Econometric regression auto-selection ─────────────────────────────
    encode_cols = selected_features
    X_econ = pd.get_dummies(
        filtered_df[encode_cols], drop_first=True
    ).astype(float)
    y_econ  = filtered_df[target]
    instruments = filtered_df[["macro_instrument"]].astype(float).reset_index(drop=True) \
        if "macro_instrument" in filtered_df.columns else None
    endog_col = "macro_indicator" if "macro_indicator" in X_econ.columns else None

    econ_model, econ_rationale = engine.fit_ols_wls_2sls(
        X_econ, y_econ,
        instruments=instruments,
        endogenous_col=endog_col
    )

    # ── 7. Train / Val split ──────────────────────────────────────────────────
    split = int(len(filtered_df) * 0.70)
    train_df = filtered_df.iloc[:split]
    val_df   = filtered_df.iloc[split:]

    X_train = train_df[selected_features]
    y_train = train_df[target]
    X_val   = val_df[selected_features]
    y_val   = val_df[target]

    # ── 8. PD model ───────────────────────────────────────────────────────────
    if model_engine == "Logistic Regression":
        pd_model    = risk_m.train_logistic_regression(X_train, y_train)
        model_type  = "logistic"
    elif model_engine == "XGBoost":
        pd_model    = risk_m.train_tree_model(X_train, y_train)
        model_type  = "tree"
    else:
        pd_model    = risk_m.train_neural_network(X_train, y_train)
        model_type  = "neural"

    calibrated_pd, cal_rationale = risk_m.calibrate_probabilities(
        pd_model, X_val, y_val
    )

    # ── 9. LGD and EAD models ─────────────────────────────────────────────────
    lgd_col = "lgd"
    ead_col = "ead"
    lgd_features = [c for c in selected_features if c != target]
    lgd_model = risk_m.train_lgd_model(train_df[lgd_features], train_df[lgd_col])
    ead_model = risk_m.train_ead_model(train_df[lgd_features], train_df[ead_col])

    # ── 10. Validation probabilities ─────────────────────────────────────────
    val_pd_probs  = calibrated_pd.predict_calibrated_proba(X_val)
    train_pd_probs = calibrated_pd.predict_calibrated_proba(X_train)

    # ── 11. KS statistic ─────────────────────────────────────────────────────
    ks_value, ks_df, ks_rationale = evaluator.calculate_ks_statistic(
        y_val.values, val_pd_probs
    )

    # ── 12. PSI ──────────────────────────────────────────────────────────────
    psi_value, psi_df, psi_rationale = evaluator.calculate_psi(
        train_pd_probs, val_pd_probs
    )

    # ── 13. SHAP explainer ────────────────────────────────────────────────────
    shap_explainer = ShapExplainer(pd_model, X_train, model_type=model_type)

    return {
        "config":            config,
        "load_rationale":    load_rationale,
        "static_df":         static_df,
        "filtered_df":       filtered_df,
        "selected_features": selected_features,
        "drop_report":       drop_report,
        "woe_iv_dict":       woe_iv_dict,
        "roll_matrix":       roll_matrix,
        "roll_rationale":    roll_rationale,
        "vintage_pivot":     vintage_pivot,
        "vintage_rationale": vintage_rationale,
        "econ_model":        econ_model,
        "econ_rationale":    econ_rationale,
        "pd_model":          pd_model,
        "calibrated_pd":     calibrated_pd,
        "lgd_model":         lgd_model,
        "ead_model":         ead_model,
        "cal_rationale":     cal_rationale,
        "model_engine":      model_engine,
        "model_type":        model_type,
        "X_train":           X_train,
        "X_val":             X_val,
        "y_val":             y_val,
        "val_pd_probs":      val_pd_probs,
        "train_pd_probs":    train_pd_probs,
        "ks_value":          ks_value,
        "ks_df":             ks_df,
        "ks_rationale":      ks_rationale,
        "psi_value":         psi_value,
        "psi_df":            psi_df,
        "psi_rationale":     psi_rationale,
        "shap_explainer":    shap_explainer,
        "evaluator":         evaluator,
        "risk_models":       risk_m,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 CreditRiskEngine v2")
    st.markdown("---")

    # CSV upload
    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader(
        "Upload your own CSV dataset",
        type=["csv"],
        help="Upload a CSV with loan features. If omitted, a synthetic dataset is used.",
        key="csv_uploader"
    )

    st.markdown("---")

    # Model engine selector
    st.markdown("### ⚙️ Modeling Engine")
    model_engine = st.selectbox(
        "Select PD Model",
        ["Logistic Regression", "XGBoost", "Neural Network"],
        index=1,
        key="model_engine_select"
    )

    st.markdown("---")

    # Borrower attribute sliders
    st.markdown("### 🧑 Borrower Profile")
    loan_amnt  = st.slider("Loan Amount ($)",    1_000, 40_000, 12_000, step=500)
    annual_inc = st.slider("Annual Income ($)",  10_000, 200_000, 55_000, step=1_000)
    int_rate   = st.slider("Interest Rate (%)",  4.0, 32.0, 12.5, step=0.1) / 100.0
    dti        = st.slider("Debt-to-Income Ratio", 0.0, 45.0, 15.0, step=0.5)
    grade      = st.selectbox("Credit Grade", ["A","B","C","D","E","F","G"], index=2)
    delinq_2yrs = st.slider("Delinquencies (2yr)", 0, 5, 0)
    emp_length  = st.slider("Employment (years)",   0, 10, 3)
    revol_util  = st.slider("Revolving Utilisation (%)", 0.0, 100.0, 40.0) / 100.0
    inq_last_6  = st.slider("Inquiries (6 months)", 0, 10, 1)
    macro_ind   = st.slider("Macro Indicator",      -3.0, 3.0, 0.5, step=0.1)

    # Stress testing sliders
    st.markdown("---")
    st.markdown("### 📉 Macro Stress Shocks (Tab 5)")
    stress_income  = st.slider("Income Shock (%)", -50, 0, -20)
    stress_rate    = st.slider("Rate Shock (pp)",    0.0, 5.0, 1.5, step=0.25)
    stress_gdp     = st.slider("GDP Shock (%)",     -10, 0, -3)


# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <h1>🏦 CreditRiskEngine: Basel III Risk Modeling &amp; Explainability</h1>
  <p>End-to-end IRB framework: WoE/IV feature selection · VIF multicollinearity control ·
     OLS/WLS/2SLS econometric selection · PD/LGD/EAD modeling · SHAP explainability</p>
</div>
""", unsafe_allow_html=True)

# Basel III regulatory formula
st.markdown(r"""
> **Basel III Regulatory Capital (IRB Approach):** Under the Internal Ratings-Based (IRB)
> framework, financial institutions calculate regulatory capital based on three IRB components.
> Expected Loss is provisioned through credit impairment charges; Unexpected Loss is absorbed
> by Common Equity Tier 1 capital:
>
> $$EL = PD \times LGD \times EAD$$
""")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────
file_bytes = uploaded_file.read() if uploaded_file else None

try:
    pipe = run_pipeline(file_bytes, model_engine)
except Exception as exc:
    st.error(f"Pipeline failed: {exc}")
    st.stop()

# Data source badge
st.markdown(
    f'<div class="rationale-box">📦 <b>Data Source:</b> {pipe["load_rationale"]}</div>',
    unsafe_allow_html=True
)

# Build borrower profile from sidebar inputs
borrower_raw = pd.DataFrame([{
    "loan_amnt":      loan_amnt,
    "int_rate":       int_rate,
    "annual_inc":     annual_inc,
    "dti":            dti,
    "grade":          grade,
    "emp_length":     emp_length,
    "delinq_2yrs":    delinq_2yrs,
    "revol_util":     revol_util,
    "inq_last_6mths": inq_last_6,
    "macro_indicator": macro_ind,
}])
borrower_features = pipe["selected_features"]
borrower_input = borrower_raw[[
    c for c in borrower_features if c in borrower_raw.columns
]]
# Fill any feature that isn't in the sidebar with the dataset mean
for col in borrower_features:
    if col not in borrower_input.columns:
        sdf = pipe["static_df"]
        borrower_input[col] = float(sdf[col].mean()) \
            if col in sdf.columns and sdf[col].dtype != "object" else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Score this borrower
# ─────────────────────────────────────────────────────────────────────────────
pd_value  = float(pipe["calibrated_pd"].predict_calibrated_proba(borrower_input)[0])
lgd_value = float(np.clip(pipe["risk_models"].predict_lgd(pipe["lgd_model"], borrower_input)[0], 0.05, 0.95))
ead_value = float(np.clip(pipe["risk_models"].predict_ead(pipe["ead_model"], borrower_input)[0], 0.0, None))
if ead_value < 100:
    ead_value = loan_amnt * 0.95  # floor to sidebar loan amount × drawdown
basel_metrics = pipe["evaluator"].calculate_basel_metrics(pd_value, lgd_value, ead_value)
el_value  = float(basel_metrics["EL"])
rwa_value = float(basel_metrics["RWA"])


# ─────────────────────────────────────────────────────────────────────────────
# 5 MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 IRB Calculator",
    "🔄 Target Definition",
    "🔍 Feature Selection",
    "📈 Model Diagnostics",
    "💡 Explainability & Stress",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — IRB Calculator
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">IRB Component Estimates</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    pd_color  = "#ef5350" if pd_value > 0.15 else ("#ff9800" if pd_value > 0.07 else "#66bb6a")
    el_color  = "#ef5350" if el_value > 5000  else "#ff9800"

    for col, label, value, fmt, note in [
        (c1, "Probability of Default (PD)",  pd_value,  f"{pd_value*100:.2f}%", "Model output"),
        (c2, "Loss Given Default (LGD)",      lgd_value, f"{lgd_value*100:.1f}%", "IRB F-IRB adj."),
        (c3, "Exposure at Default (EAD)",      ead_value, f"${ead_value:,.0f}", "95% drawdown"),
        (c4, "Expected Loss (EL)",             el_value,  f"${el_value:,.2f}", "EL = PD×LGD×EAD"),
        (c5, "Risk-Weighted Assets (RWA)",     rwa_value, f"${rwa_value:,.0f}", "BCBS retail formula"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">{label}</div>
              <div class="value" style="color:{pd_color if 'PD' in label else ('#e53935' if 'EL' in label else '#42a5f5')}">{value}</div>
              <div class="sub">{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Econometric model rationale
    st.markdown('<div class="section-title">Automated Statistical Decision Log</div>', unsafe_allow_html=True)

    with st.expander("📐 Econometric Regression Selection", expanded=True):
        st.markdown(
            f'<div class="rationale-box">{pipe["econ_rationale"]}</div>',
            unsafe_allow_html=True
        )
        coeffs = pd.Series(
            pipe["econ_model"].params,
            index=pipe["econ_model"].model.exog_names
            if hasattr(pipe["econ_model"].model, "exog_names")
            else range(len(pipe["econ_model"].params))
        ).to_frame("Coefficient")
        if hasattr(pipe["econ_model"], "pvalues"):
            coeffs["p-value"] = pipe["econ_model"].pvalues
        st.dataframe(coeffs.style.background_gradient(cmap="Blues", subset=["Coefficient"]))

    with st.expander("🎯 Calibration Rationale"):
        st.markdown(
            f'<div class="rationale-box">{pipe["cal_rationale"]}</div>',
            unsafe_allow_html=True
        )

    # Risk rating gauge
    st.markdown('<div class="section-title">Risk Rating</div>', unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pd_value * 100,
        title={"text": "Probability of Default (%)"},
        gauge={
            "axis": {"range": [0, 40]},
            "bar":  {"color": pd_color},
            "steps": [
                {"range": [0,   5],  "color": "#1b5e20"},
                {"range": [5,  15],  "color": "#f9a825"},
                {"range": [15, 40],  "color": "#b71c1c"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": pd_value * 100,
            }
        }
    ))
    fig_gauge.update_layout(
        height=280, paper_bgcolor="rgba(0,0,0,0)", font_color="white"
    )
    st.plotly_chart(fig_gauge, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Target Definition: Roll Rates & Vintage Analysis
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    col_r, col_v = st.columns(2)

    with col_r:
        st.markdown('<div class="section-title">📉 Roll Rate Transition Matrix</div>', unsafe_allow_html=True)
        roll_mx = pipe["roll_matrix"]
        fig_roll = px.imshow(
            roll_mx,
            text_auto=".1%",
            color_continuous_scale="Blues",
            title="Monthly DPD State Transition Probabilities",
            labels={"color": "Transition Rate"},
            aspect="auto"
        )
        fig_roll.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=380
        )
        st.plotly_chart(fig_roll, use_container_width=True)
        st.markdown(
            f'<div class="rationale-box">🎯 <b>Default Definition:</b> {pipe["roll_rationale"]}</div>',
            unsafe_allow_html=True
        )

    with col_v:
        st.markdown('<div class="section-title">📊 Vintage Cumulative Default Curves</div>', unsafe_allow_html=True)
        vpivot = pipe["vintage_pivot"].reset_index()
        vmelt  = vpivot.melt(id_vars="months_on_book", var_name="Cohort", value_name="Cum Default Rate")
        fig_vint = px.line(
            vmelt,
            x="months_on_book",
            y="Cum Default Rate",
            color="Cohort",
            markers=True,
            title="Cumulative Default Rate by Origination Cohort",
            labels={
                "months_on_book": "Months on Book",
                "Cum Default Rate": "Cumulative Default Rate"
            }
        )
        fig_vint.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=380,
            yaxis_tickformat=".1%"
        )
        st.plotly_chart(fig_vint, use_container_width=True)
        st.markdown(
            f'<div class="rationale-box">📅 <b>Performance Window:</b> {pipe["vintage_rationale"]}</div>',
            unsafe_allow_html=True
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Feature Selection Report
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Feature Selection Report — IV & VIF Elimination</div>', unsafe_allow_html=True)

    report = pipe["drop_report"]
    woe_iv = pipe["woe_iv_dict"]

    rows = []
    for feat, reason in report.items():
        status = "✅ Kept" if reason.startswith("Kept") else "❌ Dropped"
        iv_val = woe_iv.get(feat, {}).get("iv", float("nan"))
        rows.append({
            "Feature": feat,
            "Status":  status,
            "Reason":  reason,
            "IV":      round(iv_val, 4) if not np.isnan(iv_val) else "N/A",
        })

    report_df = pd.DataFrame(rows).sort_values("Status", ascending=False)

    def _color_status(val):
        if "Kept" in str(val):
            return "background-color: #1b4332; color: #52b788;"
        return "background-color: #3a0000; color: #ff6b6b;"

    st.dataframe(
        report_df.style.applymap(_color_status, subset=["Status"]),
        use_container_width=True,
        height=350
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Information Value (IV) Ranking</div>', unsafe_allow_html=True)
    iv_rows = [
        {"Feature": k, "IV": round(v["iv"], 4)}
        for k, v in woe_iv.items()
        if "iv" in v
    ]
    iv_df = pd.DataFrame(iv_rows).sort_values("IV", ascending=False).head(15)
    fig_iv = px.bar(
        iv_df, x="IV", y="Feature", orientation="h",
        color="IV", color_continuous_scale="Blues",
        title="Feature Information Values (threshold = 0.02)"
    )
    fig_iv.add_vline(
        x=pipe["config"].iv_threshold,
        line_dash="dash", line_color="red", annotation_text="IV threshold"
    )
    fig_iv.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", height=420, yaxis_categoryorder="total ascending"
    )
    st.plotly_chart(fig_iv, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Diagnostics: KS & PSI
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Model Performance Diagnostics</div>', unsafe_allow_html=True)

    # Summary metrics
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(pipe["y_val"].values, pipe["val_pd_probs"])
    gini    = 2 * roc_auc - 1
    ks_v    = pipe["ks_value"]
    psi_v   = pipe["psi_value"]

    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, label, val, fmt in [
        (mc1, "ROC-AUC",    roc_auc, f"{roc_auc:.4f}"),
        (mc2, "Gini Index", gini,    f"{gini*100:.1f}%"),
        (mc3, "KS Statistic", ks_v/100, f"{ks_v:.1f}%"),
        (mc4, "PSI",        psi_v,   f"{psi_v:.4f}"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">{label}</div>
              <div class="value" style="color:#42a5f5">{fmt}</div>
              <div class="sub">Validation set</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_ks, col_psi = st.columns(2)

    # KS Plot
    with col_ks:
        st.markdown('<div class="section-title">KS Statistic — Cumulative Distribution</div>', unsafe_allow_html=True)
        ks_df = pipe["ks_df"].copy()
        # Use percentage of population on x-axis
        n = len(ks_df)
        ks_df["pct_pop"] = np.linspace(0, 100, n)

        fig_ks = go.Figure()
        fig_ks.add_trace(go.Scatter(
            x=ks_df["pct_pop"], y=ks_df["cum_pct_bads"] * 100,
            mode="lines", name="Cumulative Bads (%)",
            line=dict(color="#ef5350", width=2.5)
        ))
        fig_ks.add_trace(go.Scatter(
            x=ks_df["pct_pop"], y=ks_df["cum_pct_goods"] * 100,
            mode="lines", name="Cumulative Goods (%)",
            line=dict(color="#42a5f5", width=2.5)
        ))
        fig_ks.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100],
            mode="lines", name="Random Model",
            line=dict(color="grey", dash="dash", width=1)
        ))
        fig_ks.update_layout(
            title=f"KS = {ks_v:.1f}%",
            xaxis_title="% of Population (sorted by score)",
            yaxis_title="Cumulative %",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.01)
        )
        st.plotly_chart(fig_ks, use_container_width=True)
        st.markdown(
            f'<div class="rationale-box">{pipe["ks_rationale"]}</div>',
            unsafe_allow_html=True
        )

    # PSI Chart
    with col_psi:
        st.markdown('<div class="section-title">Population Stability Index (PSI)</div>', unsafe_allow_html=True)
        psi_df = pipe["psi_df"].head(10)
        fig_psi = go.Figure()
        fig_psi.add_trace(go.Bar(
            x=psi_df["bucket"], y=psi_df["expected_pct"],
            name="Training (Expected)", marker_color="#42a5f5", opacity=0.85
        ))
        fig_psi.add_trace(go.Bar(
            x=psi_df["bucket"], y=psi_df["actual_pct"],
            name="Validation (Actual)", marker_color="#ff9800", opacity=0.85
        ))
        fig_psi.update_layout(
            barmode="group",
            title=f"PSI = {psi_v:.4f} — Score Distribution Stability",
            xaxis_title="Score Bucket",
            yaxis_title="% of Population",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=380,
            xaxis_tickangle=-40
        )
        st.plotly_chart(fig_psi, use_container_width=True)
        st.markdown(
            f'<div class="rationale-box">{pipe["psi_rationale"]}</div>',
            unsafe_allow_html=True
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Explainability & Stress Testing
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    col_shap, col_stress = st.columns([3, 2])

    with col_shap:
        st.markdown('<div class="section-title">SHAP Waterfall — Local Borrower Explanation</div>', unsafe_allow_html=True)
        try:
            shap_df, base_val = pipe["shap_explainer"].explain_borrower(borrower_input)
            top_n = shap_df.head(12)

            colors = [
                "#ef5350" if v > 0 else "#42a5f5"
                for v in top_n["shap_value"]
            ]
            fig_shap = go.Figure(go.Bar(
                x=top_n["shap_value"],
                y=top_n["feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.4f}" for v in top_n["shap_value"]],
                textposition="outside"
            ))
            fig_shap.add_vline(x=0, line_color="white", line_width=1)
            fig_shap.update_layout(
                title=f"SHAP Values (baseline = {base_val:.4f})",
                xaxis_title="SHAP Contribution to P(Default)",
                yaxis=dict(categoryorder="total ascending"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=500
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # Top-3 narrative
            top3 = shap_df.head(3)
            narrative_parts = []
            for _, row in top3.iterrows():
                direction = "increases" if row["shap_value"] > 0 else "decreases"
                narrative_parts.append(
                    f"**{row['feature']}** = {row['borrower_value']:.4g} "
                    f"{direction} default risk by {abs(row['shap_value']):.4f}"
                )
            st.markdown(
                f'<div class="rationale-box">🔍 <b>Underwriting Insight:</b> '
                f"{' | '.join(narrative_parts)}</div>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.warning(f"SHAP computation unavailable: {e}")

    with col_stress:
        st.markdown('<div class="section-title">📉 Macro Stress Testing (Pillar 2)</div>', unsafe_allow_html=True)

        # Apply shocks to borrower profile
        stressed_input = borrower_input.copy()
        if "annual_inc" in stressed_input.columns:
            stressed_input["annual_inc"] *= (1 + stress_income / 100)
        if "int_rate" in stressed_input.columns:
            stressed_input["int_rate"] += stress_rate / 100
        if "macro_indicator" in stressed_input.columns:
            stressed_input["macro_indicator"] += stress_gdp / 5.0

        # Score stressed borrower
        try:
            stressed_pd  = float(pipe["calibrated_pd"].predict_calibrated_proba(stressed_input)[0])
            stressed_lgd = float(np.clip(pipe["risk_models"].predict_lgd(pipe["lgd_model"], stressed_input)[0], 0.05, 0.95))
            stressed_ead = ead_value  # EAD typically held constant in stress
            stressed_el  = stressed_pd * stressed_lgd * stressed_ead
        except Exception:
            stressed_pd  = min(pd_value * 1.4, 0.99)
            stressed_lgd = min(lgd_value * 1.1, 0.95)
            stressed_ead = ead_value
            stressed_el  = stressed_pd * stressed_lgd * stressed_ead

        el_delta   = stressed_el - el_value
        pd_delta   = stressed_pd - pd_value

        # Metric cards
        st.markdown("**Scenario Results**")
        for label, base, stressed, fmt_fn in [
            ("PD",      pd_value,  stressed_pd,  lambda v: f"{v*100:.2f}%"),
            ("LGD",     lgd_value, stressed_lgd, lambda v: f"{v*100:.1f}%"),
            ("EL ($)",  el_value,  stressed_el,  lambda v: f"${v:,.2f}"),
        ]:
            d = stressed - base
            arrow = "▲" if d > 0 else "▼"
            color = "#ef5350" if d > 0 else "#66bb6a"
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:0.6rem">
              <div class="label">{label}</div>
              <div style="display:flex;justify-content:space-between;align-items:center;color:white">
                <span style="font-size:1.1rem">Base: <b>{fmt_fn(base)}</b></span>
                <span style="font-size:1.1rem;color:{color}">{arrow} Stressed: <b>{fmt_fn(stressed)}</b></span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tornado chart: shock sensitivities
        shocks_applied = {
            "Income Shock": stress_income,
            "Rate Shock":   stress_rate,
            "GDP Shock":    stress_gdp,
        }
        sensitivities = []
        for shock_name, shock_val in shocks_applied.items():
            # Marginal EL contribution from this single shock
            s_inp = borrower_input.copy()
            if shock_name == "Income Shock" and "annual_inc" in s_inp.columns:
                s_inp["annual_inc"] *= (1 + shock_val / 100)
            elif shock_name == "Rate Shock" and "int_rate" in s_inp.columns:
                s_inp["int_rate"] += shock_val / 100
            elif shock_name == "GDP Shock" and "macro_indicator" in s_inp.columns:
                s_inp["macro_indicator"] += shock_val / 5.0
            try:
                s_pd = float(pipe["calibrated_pd"].predict_calibrated_proba(s_inp)[0])
            except Exception:
                s_pd = pd_value * (1 + abs(shock_val) / 100)
            sensitivities.append({
                "Shock": shock_name,
                "EL Increase ($)": max(0.0, s_pd * lgd_value * ead_value - el_value)
            })

        sens_df  = pd.DataFrame(sensitivities).sort_values("EL Increase ($)")
        fig_torn = px.bar(
            sens_df, x="EL Increase ($)", y="Shock", orientation="h",
            color="EL Increase ($)", color_continuous_scale="Reds",
            title="Tornado: EL Sensitivity by Shock Factor"
        )
        fig_torn.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=250
        )
        st.plotly_chart(fig_torn, use_container_width=True)

        st.markdown(
            f'<div class="rationale-box">'
            f"📊 <b>Combined Stress:</b> PD moves from {pd_value*100:.2f}% → "
            f"{stressed_pd*100:.2f}% (+{pd_delta*100:.2f} pp). "
            f"Expected Loss increases by ${el_delta:,.2f} under this macro scenario.</div>",
            unsafe_allow_html=True
        )
