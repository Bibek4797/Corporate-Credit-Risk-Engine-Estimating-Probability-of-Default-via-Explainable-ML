"""
Validation script for CreditRiskEngine v2 pipeline.
Run this script to verify that all modules compile, execute, and return valid outputs.
"""

import os
import sys

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from config import SystemConfig
from src.data_prep import DataOrchestrator
from src.econometrics import EconometricEngine
from src.models import RiskModels
from src.evaluate import RiskEvaluator
from src.explain import ShapExplainer


def run_validation():
    print("=" * 70)
    print("      CreditRiskEngine v2 -- Pipeline Validation Suite")
    print("=" * 70)

    # 1. Config Test
    print("\n[Step 1/8] Validating SystemConfig...")
    config = SystemConfig()
    assert config.vif_threshold == 5.0, f"VIF threshold expected 5.0, got {config.vif_threshold}"
    assert config.iv_threshold == 0.02, f"IV threshold expected 0.02, got {config.iv_threshold}"
    assert config.performance_window == 18, f"Performance window expected 18, got {config.performance_window}"
    print("  [PASS] SystemConfig validated (VIF=5.0, IV=0.02, Window=18).")

    # 2. Data Preparation Test
    print("\n[Step 2/8] Validating DataOrchestrator...")
    orchestrator = DataOrchestrator(config)
    datasets, rationale = orchestrator.load_data(uploaded_file=None)
    assert "static" in datasets, "Missing 'static' dataset"
    assert "delinquency" in datasets, "Missing 'delinquency' dataset"
    assert "vintage" in datasets, "Missing 'vintage' dataset"
    print(f"  [PASS] Data loaded. Rationale: {rationale[:60]}...")

    matrix, roll_rat = orchestrator.compute_roll_rates(datasets)
    assert not matrix.empty, "Roll rate matrix is empty"
    print(f"  [PASS] Roll rate calculated. Shape: {matrix.shape}. Default target: 90-DPD.")

    vintage_pivot, vint_rat = orchestrator.perform_vintage_analysis(datasets)
    assert not vintage_pivot.empty, "Vintage pivot is empty"
    print(f"  [PASS] Vintage analysis calculated. Shape: {vintage_pivot.shape}.")

    # Test Reject Inference
    static_df = datasets["static"]
    accepted = static_df.iloc[:4000].copy()
    rejected = static_df.iloc[4000:].drop(columns=["default_label"]).copy()
    aug_df, rej_rat = orchestrator.reject_inference(accepted, rejected)
    assert len(aug_df) > len(accepted), "Reject inference failed to augment population"
    print(f"  [PASS] Reject inference completed. Augmented population: {len(aug_df):,} rows.")

    # 3. Econometrics & Feature Selection Test
    print("\n[Step 3/8] Validating EconometricEngine...")
    engine = EconometricEngine(config)
    
    woe_iv = engine.calculate_woe_iv(static_df, "default_label")
    assert len(woe_iv) > 0, "calculate_woe_iv returned empty dict"
    print(f"  [PASS] WoE/IV calculated for {len(woe_iv)} features.")

    filtered_df, drop_report = engine.feature_selection(
        static_df, "default_label",
        exclude_cols=["loan_id", "ead", "lgd", "macro_instrument"]
    )
    assert "default_label" in filtered_df.columns, "Target column dropped"
    print(f"  [PASS] Feature selection finished. Selected {filtered_df.shape[1] - 1} features.")
    print(f"  Drop Report Sample: {list(drop_report.items())[0]}")

    # Econometric model selection (OLS/WLS/2SLS)
    features = [c for c in filtered_df.columns if c not in ["default_label", "loan_id", "ead", "lgd"]]
    X_econ = pd.get_dummies(filtered_df[features], drop_first=True).astype(float)
    y_econ = filtered_df["default_label"].astype(float)
    inst = filtered_df[["macro_instrument"]].astype(float) if "macro_instrument" in filtered_df.columns else None
    endog = "macro_indicator" if "macro_indicator" in X_econ.columns else None
    
    model, reg_rat = engine.fit_ols_wls_2sls(X_econ, y_econ, instruments=inst, endogenous_col=endog)
    print(f"  [PASS] Regression auto-selection: {reg_rat}")

    # 4. Machine Learning Models Test
    print("\n[Step 4/8] Validating RiskModels (Logistic, XGBoost, PyTorch MLP, LGD, EAD)...")
    models = RiskModels(config)
    split = int(len(filtered_df) * 0.7)
    train_df = filtered_df.iloc[:split]
    val_df = filtered_df.iloc[split:]
    X_train = train_df[features]
    y_train = train_df["default_label"]
    X_val = val_df[features]
    y_val = val_df["default_label"]

    # Logistic Regression
    lr_model = models.train_logistic_regression(X_train, y_train)
    print("  [PASS] Logistic Regression trained.")

    # Tree model
    xgb_model = models.train_tree_model(X_train, y_train)
    print("  [PASS] XGBoost model trained.")

    # Neural network model
    nn_model = models.train_neural_network(X_train, y_train)
    print("  [PASS] PyTorch MLP trained.")

    # Calibrate
    calibrated_model, cal_rat = models.calibrate_probabilities(xgb_model, X_val, y_val)
    val_preds = calibrated_model.predict_calibrated_proba(X_val)
    print(f"  [PASS] Probability calibration complete. {cal_rat}")

    # LGD & EAD models
    lgd_model = models.train_lgd_model(X_train, train_df["lgd"])
    ead_model = models.train_ead_model(X_train, train_df["ead"])
    pred_lgd = models.predict_lgd(lgd_model, X_val)
    pred_ead = models.predict_ead(ead_model, X_val)
    assert len(pred_lgd) == len(X_val), "LGD prediction length mismatch"
    assert len(pred_ead) == len(X_val), "EAD prediction length mismatch"
    print("  [PASS] LGD and EAD models trained and validated.")

    # 5. Risk Evaluation Test
    print("\n[Step 5/8] Validating RiskEvaluator (KS, PSI, Basel III Metrics)...")
    evaluator = RiskEvaluator()
    ks_stat, ks_df, ks_rat = evaluator.calculate_ks_statistic(y_val.values, val_preds)
    assert 0 <= ks_stat <= 100, f"KS statistic out of bounds: {ks_stat}"
    print(f"  [PASS] KS Statistic: {ks_stat:.2f}%. {ks_rat}")

    train_preds = calibrated_model.predict_calibrated_proba(X_train)
    psi_stat, psi_df, psi_rat = evaluator.calculate_psi(train_preds, val_preds)
    print(f"  [PASS] PSI: {psi_stat:.4f}. {psi_rat}")

    basel_metrics = evaluator.calculate_basel_metrics(
        pd_val=float(val_preds.mean()),
        lgd_val=float(pred_lgd.mean()),
        ead_val=float(pred_ead.mean())
    )
    print(f"  [PASS] Basel III Metrics: EL=${basel_metrics['EL']:,.2f}, RWA=${basel_metrics['RWA']:,.2f}")

    # 6. SHAP Explainer Test
    print("\n[Step 6/8] Validating ShapExplainer for all 3 model types...")
    # Test Logistic
    lr_explainer = ShapExplainer(lr_model, X_train, model_type="logistic")
    shap_df_lr, base_lr = lr_explainer.explain_borrower(X_val.iloc[:1])
    assert not shap_df_lr.empty, "Logistic SHAP summary is empty"
    print("  [PASS] Logistic Regression SHAP explainer validated.")

    # Test Tree
    tree_explainer = ShapExplainer(xgb_model, X_train, model_type="tree")
    shap_df_tree, base_tree = tree_explainer.explain_borrower(X_val.iloc[:1])
    assert not shap_df_tree.empty, "Tree SHAP summary is empty"
    print("  [PASS] XGBoost SHAP explainer validated.")

    # Test Neural
    nn_explainer = ShapExplainer(nn_model, X_train, model_type="neural")
    shap_df_nn, base_nn = nn_explainer.explain_borrower(X_val.iloc[:1])
    assert not shap_df_nn.empty, "Neural Network SHAP summary is empty"
    print("  [PASS] PyTorch MLP SHAP explainer validated.")

    # 7. CSV Upload Test
    print("\n[Step 7/8] Validating custom CSV uploader logic...")
    sample_csv_data = static_df.head(100).to_csv(index=False)
    csv_bytes = sample_csv_data.encode("utf-8")
    up_datasets, up_rat = orchestrator.load_data(uploaded_file=csv_bytes)
    assert len(up_datasets["static"]) == 100, "Uploaded CSV row count mismatch"
    print(f"  [PASS] Custom CSV upload test passed. {up_rat[:60]}...")

    # 8. Complete Pipeline Verification
    print("\n[Step 8/8] Summary")
    print("=" * 70)
    print("  ALL 8 VALIDATION STEPS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_validation()
