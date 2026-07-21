import pandas as pd
import numpy as np
import shap
from typing import Tuple, Any, List
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


class ShapExplainer:
    """Computes SHAP local explanations for all three modeling engines.

    Selects the appropriate SHAP explainer type:
    - Logistic Regression  → shap.LinearExplainer
    - XGBoost              → shap.TreeExplainer
    - PyTorch MLP          → shap.KernelExplainer (on a small background sample)

    Parameters
    ----------
    model : Any
        Trained PD model (LogisticRegression, XGBClassifier, or PyTorchMLPWrapper).
    X_train : pd.DataFrame
        Training features (pre-encoding) used as background reference.
    model_type : str
        One of 'logistic', 'tree', 'neural'.
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        model_type: str = "tree"
    ):
        self.model      = model
        self.model_type = model_type.lower()

        # Encode training data to float64
        self.X_enc = pd.get_dummies(X_train).astype(np.float64)
        self.feature_columns: List[str] = list(self.X_enc.columns)

        bg = self.X_enc.sample(
            n=min(100, len(self.X_enc)), random_state=42
        ).values

        if self.model_type == "logistic":
            # Linear explainer for logistic regression
            # Align feature columns
            self.explainer = shap.LinearExplainer(
                model, bg, feature_perturbation="interventional"
            )
        elif self.model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        else:
            # Neural network: use KernelExplainer with small background
            def _predict_fn(X_arr: np.ndarray) -> np.ndarray:
                p = model.predict_proba(X_arr)
                return p[:, 1] if p.ndim == 2 else p
            self.explainer = shap.KernelExplainer(
                _predict_fn, shap.kmeans(bg, min(20, len(bg)))
            )

    def explain_borrower(
        self,
        borrower_profile: pd.DataFrame
    ) -> Tuple[pd.DataFrame, float]:
        """Generates SHAP explanation for a single borrower.

        Parameters
        ----------
        borrower_profile : pd.DataFrame
            Single-row DataFrame with raw (pre-encoded) borrower features.

        Returns
        -------
        pd.DataFrame
            Columns: feature, borrower_value, shap_value. Sorted by |shap|.
        float
            Model baseline value (expected output).
        """
        # Encode and align to training columns
        b_enc = pd.get_dummies(borrower_profile).astype(np.float64)
        for col in self.feature_columns:
            if col not in b_enc.columns:
                b_enc[col] = 0.0
        b_enc = b_enc[self.feature_columns]

        # Compute SHAP values
        if self.model_type in ("tree", "logistic"):
            raw = self.explainer(b_enc)
        else:
            raw = self.explainer.shap_values(b_enc.values)

        # Parse SHAP output uniformly
        if hasattr(raw, "values"):
            vals = raw.values
            base = raw.base_values
            if vals.ndim == 3:            # (samples, features, classes)
                contribs   = vals[0, :, 1]
                base_value = float(base[0, 1]) if base.ndim == 2 else float(base[0])
            else:                         # (samples, features)
                contribs   = vals[0, :]
                base_value = float(base[0]) if hasattr(base, "__len__") else float(base)
        else:
            # KernelExplainer returns ndarray or list
            if isinstance(raw, list):
                contribs   = raw[1][0] if len(raw) == 2 else np.array(raw).flatten()
                ev         = self.explainer.expected_value
                base_value = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
            else:
                contribs   = np.array(raw).flatten()
                ev         = self.explainer.expected_value
                base_value = float(ev[0]) if hasattr(ev, "__len__") else float(ev)

        summary_df = pd.DataFrame({
            "feature":        self.feature_columns,
            "borrower_value": b_enc.iloc[0].values,
            "shap_value":     contribs.flatten(),
        })
        summary_df["abs_shap"] = summary_df["shap_value"].abs()
        summary_df = (
            summary_df
            .sort_values("abs_shap", ascending=False)
            .drop(columns=["abs_shap"])
            .reset_index(drop=True)
        )
        return summary_df, base_value
