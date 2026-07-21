import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any, Optional
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.sandbox.regression.gmm import IV2SLS
from config import SystemConfig


class EconometricEngine:
    """Performs econometric feature selection and automated regression model selection.

    Parameters
    ----------
    config : SystemConfig
        System-wide configuration.
    """

    def __init__(self, config: SystemConfig):
        self.config = config

    # ------------------------------------------------------------------
    # 1. Weight of Evidence & Information Value
    # ------------------------------------------------------------------
    def calculate_woe_iv(
        self,
        df: pd.DataFrame,
        target: str
    ) -> Dict[str, Dict[str, Any]]:
        """Computes Weight of Evidence (WoE) and Information Value (IV) for all features.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing features and target.
        target : str
            Binary target column name.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Maps feature name to {"iv": float, "woe_bins": pd.Series}.
        """
        exclude = [target, "loan_id", "ead", "lgd", "macro_instrument", "default_label"]
        results = {}
        for col in df.columns:
            if col in exclude:
                continue
            iv, woe_dict = self._calc_iv_woe(df, col, target)
            results[col] = {"iv": iv, "woe_bins": woe_dict}
        return results

    # ------------------------------------------------------------------
    # 2. Feature Selection
    # ------------------------------------------------------------------
    def feature_selection(
        self,
        df: pd.DataFrame,
        target: str,
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Filters features using IV and iterative VIF elimination.

        Parameters
        ----------
        df : pd.DataFrame
            Full training dataset.
        target : str
            Binary target column name.
        exclude_cols : List[str], optional
            Administrative columns to exclude from selection.

        Returns
        -------
        pd.DataFrame
            Dataset containing only selected features + admin cols + target.
        Dict[str, str]
            Drop report: maps each feature to its exact status string, e.g.
            ``"Dropped: IV = 0.01 < 0.02"`` or ``"Kept: IV = 0.35, VIF = 2.1"``.
        """
        if exclude_cols is None:
            exclude_cols = [
                "loan_id", "ead", "lgd", "macro_instrument", "default_label"
            ]

        report: Dict[str, str] = {}
        df_clean = df.copy()
        candidate_features = [
            c for c in df_clean.columns
            if c != target and c not in exclude_cols
        ]

        # Stage 1 — IV filter
        selected = []
        for col in candidate_features:
            iv, _ = self._calc_iv_woe(df_clean, col, target)
            if iv < self.config.iv_threshold:
                report[col] = (
                    f"Dropped: IV = {iv:.4f} < {self.config.iv_threshold} "
                    f"(insufficient predictive power)"
                )
            else:
                selected.append(col)
                report[col] = f"Stage 1 pass: IV = {iv:.4f}"

        # Stage 2 — VIF iterative elimination
        # Build WoE-encoded matrix for VIF (keeps numeric)
        df_vif = pd.DataFrame()
        for col in selected:
            iv, woe_dict = self._calc_iv_woe(df_clean, col, target)
            if df_clean[col].dtype == "object" or df_clean[col].dtype.name == "category":
                df_vif[col] = df_clean[col].map(woe_dict).fillna(0.0)
            else:
                try:
                    binned = pd.qcut(df_clean[col], q=5, labels=False, duplicates="drop")
                except Exception:
                    binned = pd.cut(df_clean[col], bins=5, labels=False)
                df_vif[col] = binned.map(woe_dict).fillna(0.0)

        vif_features = list(selected)
        while len(vif_features) > 1:
            X_vif      = sm.add_constant(df_vif[vif_features].copy(), has_constant="add")
            col_names  = list(X_vif.columns)
            vif_values = []
            for col in vif_features:
                pos = col_names.index(col) if col in col_names else -1
                val = variance_inflation_factor(X_vif.values, pos) if pos >= 0 else 1.0
                vif_values.append(val)

            max_vif = max(vif_values)
            worst   = vif_features[vif_values.index(max_vif)]
            if max_vif > self.config.vif_threshold:
                vif_features.remove(worst)
                report[worst] = (
                    f"Dropped: VIF = {max_vif:.2f} > {self.config.vif_threshold} "
                    f"(multicollinearity)"
                )
            else:
                break

        # Record final VIF for kept features
        if len(vif_features) > 1:
            X_final   = sm.add_constant(
                df_vif[vif_features].copy(), has_constant="add"
            )
            col_names = list(X_final.columns)
            for col in vif_features:
                pos      = col_names.index(col) if col in col_names else -1
                final_vif = variance_inflation_factor(X_final.values, pos) if pos >= 0 else np.nan
                prev = report.get(col, "")
                report[col] = f"Kept: {prev.replace('Stage 1 pass: ','')} | VIF = {final_vif:.2f}"

        final_cols = vif_features + [target] + [
            c for c in exclude_cols if c in df_clean.columns and c != target
        ]
        return df_clean[final_cols], report

    # ------------------------------------------------------------------
    # 3. OLS / WLS / 2SLS Auto-Selection
    # ------------------------------------------------------------------
    def fit_ols_wls_2sls(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        instruments: Optional[pd.DataFrame] = None,
        endogenous_col: Optional[str] = None
    ) -> Tuple[Any, str]:
        """Runs heteroskedasticity and endogeneity tests and fits the appropriate
        linear estimator (OLS, WLS, or 2SLS).

        Parameters
        ----------
        X : pd.DataFrame
            Exogenous feature matrix (already dummy-encoded, float64).
        y : pd.Series
            Binary or continuous target.
        instruments : pd.DataFrame, optional
            External instrumental variables.
        endogenous_col : str, optional
            Name of the potentially endogenous column in X.

        Returns
        -------
        Any
            Fitted statsmodels result object.
        str
            Rationale explaining the model choice.
        """
        # Enforce float64 and reset indices
        X = X.astype(float).reset_index(drop=True)
        y = y.astype(float).reset_index(drop=True)
        if instruments is not None:
            instruments = instruments.astype(float).reset_index(drop=True)

        X_const = sm.add_constant(X, has_constant="add")

        # Baseline OLS
        ols_fit   = sm.OLS(y.values, X_const).fit()
        residuals = ols_fit.resid

        # Breusch-Pagan heteroskedasticity test
        bp_pvalue = het_breuschpagan(residuals, X_const)[1]
        has_hetero = bp_pvalue < self.config.p_value_threshold

        # Durbin-Wu-Hausman endogeneity test
        has_endo   = False
        dwh_pvalue = 1.0
        stage1_fit = None

        if instruments is not None and endogenous_col and endogenous_col in X.columns:
            exog_cols = [c for c in X.columns if c != endogenous_col]
            X_s1      = pd.concat(
                [X[exog_cols].reset_index(drop=True), instruments], axis=1
            )
            X_s1_c    = sm.add_constant(X_s1, has_constant="add")
            stage1_fit = sm.OLS(X[endogenous_col].values, X_s1_c).fit()
            resid_s1   = pd.Series(stage1_fit.resid, name="stage1_resid")

            X_s2   = pd.concat([X.reset_index(drop=True), resid_s1], axis=1)
            X_s2_c = sm.add_constant(X_s2, has_constant="add")
            s2_fit = sm.OLS(y.values, X_s2_c).fit()

            if "stage1_resid" in s2_fit.pvalues.index:
                dwh_pvalue = s2_fit.pvalues["stage1_resid"]
                has_endo   = dwh_pvalue < self.config.p_value_threshold

        # --- Model Selection ---
        if has_endo and instruments is not None and endogenous_col:
            exog_cols  = [c for c in X.columns if c != endogenous_col]
            inst_mat   = pd.concat(
                [X[exog_cols].reset_index(drop=True), instruments], axis=1
            )
            inst_mat_c = sm.add_constant(inst_mat, has_constant="add")
            try:
                model = IV2SLS(y.values, X_const, inst_mat_c).fit()
                rationale = (
                    f"2SLS selected: DWH test detected endogeneity in "
                    f"'{endogenous_col}' (p = {dwh_pvalue:.4f} < {self.config.p_value_threshold}). "
                    f"BP heteroskedasticity p = {bp_pvalue:.4f}."
                )
            except Exception:
                if stage1_fit is not None:
                    X_hat = X.copy()
                    X_hat[endogenous_col] = stage1_fit.fittedvalues
                model     = sm.OLS(y.values, sm.add_constant(X_hat, has_constant="add")).fit()
                rationale = (
                    f"Manual 2SLS (fallback): DWH p = {dwh_pvalue:.4f}, "
                    f"IV-corrected Stage 1 predictions substituted."
                )
            return model, rationale

        if has_hetero:
            log_r2    = np.log(residuals**2 + 1e-8)
            var_model = sm.OLS(log_r2, X_const).fit()
            weights   = 1.0 / np.exp(var_model.fittedvalues)
            model     = sm.WLS(y.values, X_const, weights=weights).fit()
            rationale = (
                f"WLS selected: Breusch-Pagan detected heteroskedasticity "
                f"(p = {bp_pvalue:.4f} < {self.config.p_value_threshold}). "
                f"Variance weights estimated via FGLS."
            )
            return model, rationale

        rationale = (
            f"OLS selected: no heteroskedasticity (BP p = {bp_pvalue:.4f}) "
            f"and no endogeneity (DWH p = {dwh_pvalue:.4f}) detected. "
            f"OLS is BLUE under these classical assumptions."
        )
        return ols_fit, rationale

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _calc_iv_woe(
        self,
        df: pd.DataFrame,
        col: str,
        target: str
    ) -> Tuple[float, Dict[Any, float]]:
        """Computes IV and WoE bins for a single column."""
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            groups = df[col]
        else:
            try:
                groups = pd.qcut(df[col], q=5, duplicates="drop")
            except Exception:
                groups = pd.cut(df[col], bins=5)

        ct = pd.crosstab(groups, df[target])
        for v in [0, 1]:
            if v not in ct.columns:
                ct[v] = 0
        ct[0] += 0.5
        ct[1] += 0.5

        p_g = ct[0] / ct[0].sum()
        p_b = ct[1] / ct[1].sum()
        woe = np.log(p_g / p_b)
        iv  = float(((p_g - p_b) * woe).sum())
        return iv, woe.to_dict()
