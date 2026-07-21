import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
from typing import Tuple, Any, Union, List
from config import SystemConfig


# ---------------------------------------------------------------------------
# PyTorch Architecture
# ---------------------------------------------------------------------------
class _MLP(nn.Module):
    """Feed-forward MLP for binary classification."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyTorchMLPWrapper:
    """Scikit-learn compatible wrapper for the PyTorch MLP."""

    def __init__(self, input_dim: int, config: SystemConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = _MLP(input_dim).to(self.device)
        self.scaler = StandardScaler()
        self._feature_cols: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PyTorchMLPWrapper":
        X_s = self.scaler.fit_transform(X)
        Xt  = torch.tensor(X_s, dtype=torch.float32)
        yt  = torch.tensor(y,   dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=self.config.pytorch_batch_size,
            shuffle=True
        )
        opt  = optim.Adam(self.model.parameters(), lr=self.config.pytorch_lr)
        loss_fn = nn.BCEWithLogitsLoss()
        self.model.train()
        for _ in range(self.config.pytorch_epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                opt.zero_grad()
                loss_fn(self.model(bx), by).backward()
                opt.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_s = self.scaler.transform(X)
        Xt  = torch.tensor(X_s, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            p = torch.sigmoid(self.model(Xt)).cpu().numpy().flatten()
        # Return 2D array matching sklearn convention [P(0), P(1)]
        return np.column_stack([1 - p, p])


# ---------------------------------------------------------------------------
# Calibration Wrapper
# ---------------------------------------------------------------------------
class IsotonicCalibrationWrapper:
    """Wraps any model and applies Isotonic Regression calibration.

    Stores training column names to enable auto-encoding of DataFrame inputs.
    """

    def __init__(self, base_model: Any):
        self.base_model = base_model
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self._training_columns: List[str] = []

    def _encode(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            Xe = pd.get_dummies(X).astype(np.float64)
            for c in self._training_columns:
                if c not in Xe.columns:
                    Xe[c] = 0.0
            if self._training_columns:
                Xe = Xe[self._training_columns]
            return Xe.values
        return np.asarray(X, dtype=np.float64)

    def _raw_probs(self, X_arr: np.ndarray) -> np.ndarray:
        p = self.base_model.predict_proba(X_arr)
        return p[:, 1] if p.ndim == 2 else p

    def fit_calibration(
        self, X_val: Any, y_val: np.ndarray
    ) -> Tuple["IsotonicCalibrationWrapper", float, float]:
        Xa  = self._encode(X_val)
        raw = self._raw_probs(Xa)
        self.calibrator.fit(raw, y_val)
        cal = self.calibrator.predict(raw)
        return self, brier_score_loss(y_val, raw), brier_score_loss(y_val, cal)

    def predict_calibrated_proba(self, X: Any) -> np.ndarray:
        return self.calibrator.predict(self._raw_probs(self._encode(X)))


# ---------------------------------------------------------------------------
# RiskModels
# ---------------------------------------------------------------------------
class RiskModels:
    """Trains PD, LGD, and EAD models and calibrates default probabilities.

    Parameters
    ----------
    config : SystemConfig
        System-wide configuration.
    """

    def __init__(self, config: SystemConfig):
        self.config = config

    # --- PD Models ---
    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> LogisticRegression:
        """Trains L2-regularised Logistic Regression for PD.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Binary default labels.

        Returns
        -------
        LogisticRegression
            Fitted sklearn LogisticRegression with predict_proba support.
        """
        X_enc = pd.get_dummies(X_train).astype(np.float64)
        clf   = LogisticRegression(
            C=1.0, max_iter=1000,
            solver="lbfgs", random_state=self.config.random_state
        )
        clf.fit(X_enc, y_train.values)
        # Store column names for later alignment
        clf._feature_cols = list(X_enc.columns)
        return clf

    def train_tree_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> xgb.XGBClassifier:
        """Trains an XGBoost gradient-boosted classifier for PD.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Binary default labels.

        Returns
        -------
        xgb.XGBClassifier
            Fitted XGBoost classifier.
        """
        X_enc = pd.get_dummies(X_train).astype(np.float64)
        clf   = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=self.config.random_state
        )
        clf.fit(X_enc, y_train.values)
        clf._feature_cols = list(X_enc.columns)
        return clf

    def train_neural_network(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> PyTorchMLPWrapper:
        """Trains a PyTorch MLP for PD.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Binary default labels.

        Returns
        -------
        PyTorchMLPWrapper
            Fitted PyTorch MLP wrapper.
        """
        X_enc = pd.get_dummies(X_train).astype(np.float64)
        wrapper = PyTorchMLPWrapper(
            input_dim=X_enc.shape[1], config=self.config
        )
        wrapper._feature_cols = list(X_enc.columns)
        wrapper.fit(X_enc.values, y_train.values)
        return wrapper

    def calibrate_probabilities(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Tuple[IsotonicCalibrationWrapper, str]:
        """Calibrates a trained PD model using Isotonic Regression.

        Parameters
        ----------
        model : Any
            Trained Logistic Regression, XGBoost, or PyTorch model.
        X_val : pd.DataFrame
            Validation feature matrix.
        y_val : pd.Series
            Validation labels.

        Returns
        -------
        IsotonicCalibrationWrapper
            Calibrated model wrapper.
        str
            Rationale citing Brier score improvement.
        """
        X_enc = pd.get_dummies(X_val).astype(np.float64)
        # Align to training columns if stored
        train_cols = getattr(model, "_feature_cols", [])
        if train_cols:
            for c in train_cols:
                if c not in X_enc.columns:
                    X_enc[c] = 0.0
            X_enc = X_enc[train_cols]

        wrap = IsotonicCalibrationWrapper(model)
        wrap._training_columns = list(X_enc.columns)
        _, bp, ba = wrap.fit_calibration(X_enc.values, y_val.values)
        pct = ((bp - ba) / bp) * 100 if bp > 0 else 0.0
        rationale = (
            f"Brier score: {bp:.5f} → {ba:.5f} ({pct:.1f}% improvement). "
            f"Isotonic calibration ensures PD values reflect true default "
            f"frequencies, satisfying Basel III calibration requirements."
        )
        return wrap, rationale

    # --- LGD Model ---
    def train_lgd_model(
        self,
        X_train: pd.DataFrame,
        lgd_train: pd.Series
    ) -> xgb.XGBRegressor:
        """Trains an XGBoost regressor for Loss Given Default (LGD).

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        lgd_train : pd.Series
            Continuous LGD values in [0, 1].

        Returns
        -------
        xgb.XGBRegressor
            Fitted XGBoost regressor for LGD.
        """
        X_enc = pd.get_dummies(X_train).astype(np.float64)
        reg   = xgb.XGBRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            random_state=self.config.random_state
        )
        reg.fit(X_enc, lgd_train.values)
        reg._feature_cols = list(X_enc.columns)
        return reg

    def predict_lgd(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Predicts LGD for a feature matrix, clipped to [0.05, 0.95]."""
        X_enc = pd.get_dummies(X).astype(np.float64)
        cols  = getattr(model, "_feature_cols", [])
        if cols:
            for c in cols:
                if c not in X_enc.columns:
                    X_enc[c] = 0.0
            X_enc = X_enc[cols]
        return np.clip(model.predict(X_enc.values), 0.05, 0.95)

    # --- EAD Model ---
    def train_ead_model(
        self,
        X_train: pd.DataFrame,
        ead_train: pd.Series
    ) -> xgb.XGBRegressor:
        """Trains an XGBoost regressor for Exposure at Default (EAD).

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        ead_train : pd.Series
            EAD monetary values.

        Returns
        -------
        xgb.XGBRegressor
            Fitted XGBoost regressor for EAD.
        """
        X_enc = pd.get_dummies(X_train).astype(np.float64)
        reg   = xgb.XGBRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            random_state=self.config.random_state
        )
        reg.fit(X_enc, ead_train.values)
        reg._feature_cols = list(X_enc.columns)
        return reg

    def predict_ead(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Predicts EAD for a feature matrix, clipped to be non-negative."""
        X_enc = pd.get_dummies(X).astype(np.float64)
        cols  = getattr(model, "_feature_cols", [])
        if cols:
            for c in cols:
                if c not in X_enc.columns:
                    X_enc[c] = 0.0
            X_enc = X_enc[cols]
        return np.clip(model.predict(X_enc.values), 0.0, None)
