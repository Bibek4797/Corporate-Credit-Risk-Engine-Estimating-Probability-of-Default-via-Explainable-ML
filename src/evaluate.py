import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple, Dict, Union


class RiskEvaluator:
    """Evaluates credit risk model performance and computes Basel III capital metrics."""

    # ------------------------------------------------------------------
    # 1. KS Statistic
    # ------------------------------------------------------------------
    def calculate_ks_statistic(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[float, pd.DataFrame, str]:
        """Computes the Kolmogorov-Smirnov statistic and returns plot data.

        Parameters
        ----------
        y_true : np.ndarray
            Actual binary labels.
        y_pred : np.ndarray
            Predicted default probabilities.

        Returns
        -------
        float
            KS statistic (0-100 scale).
        pd.DataFrame
            Columns: threshold, cum_pct_goods, cum_pct_bads for plotting.
        str
            Interpretive rationale string.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        ks_values = tpr - fpr
        ks_idx    = int(np.argmax(ks_values))
        ks_value  = float(ks_values[ks_idx]) * 100.0  # percentage

        plot_df = pd.DataFrame({
            "threshold":     thresholds[::-1],
            "cum_pct_goods": fpr[::-1],
            "cum_pct_bads":  tpr[::-1],
        })

        roc_auc = roc_auc_score(y_true, y_pred)
        gini    = 2.0 * roc_auc - 1.0

        if ks_value >= 40:
            label = "excellent"
        elif ks_value >= 30:
            label = "good"
        elif ks_value >= 20:
            label = "fair"
        else:
            label = "poor (consider model re-specification)"

        rationale = (
            f"KS-Statistic = {ks_value:.1f}% — {label} separation between Goods "
            f"and Bads. Gini = {gini*100:.1f}%, ROC-AUC = {roc_auc:.4f}."
        )
        return ks_value, plot_df, rationale

    # ------------------------------------------------------------------
    # 2. PSI
    # ------------------------------------------------------------------
    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        num_buckets: int = 10
    ) -> Tuple[float, pd.DataFrame, str]:
        """Computes Population Stability Index between two probability vectors.

        Parameters
        ----------
        expected : np.ndarray
            Reference population probabilities (e.g., training set).
        actual : np.ndarray
            Current population probabilities (e.g., validation set).
        num_buckets : int, default 10
            Number of equal-width quantile buckets.

        Returns
        -------
        float
            Aggregate PSI value.
        pd.DataFrame
            Bucket-level PSI decomposition for charting.
        str
            Interpretive rationale string.
        """
        # Build quantile bins on expected distribution
        pcts = np.linspace(0, 100, num_buckets + 1)
        bins = np.unique(np.percentile(expected, pcts))
        if len(bins) < 2:
            bins = np.linspace(expected.min() - 1e-5, expected.max() + 1e-5, num_buckets + 1)
        bins[0]  -= 1e-5
        bins[-1] += 1e-5

        exp_cnt, _ = np.histogram(expected, bins=bins)
        act_cnt, _ = np.histogram(actual,   bins=bins)

        eps = 0.5
        exp_pct = (exp_cnt + eps) / (len(expected) + eps * len(bins))
        act_pct = (act_cnt + eps) / (len(actual)   + eps * len(bins))

        bucket_psi = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
        total_psi  = float(bucket_psi.sum())

        labels = [
            f"{bins[i]:.3f}–{bins[i+1]:.3f}" for i in range(len(bins) - 1)
        ]
        bucket_df = pd.DataFrame({
            "bucket":      labels,
            "expected_pct": exp_pct * 100,
            "actual_pct":   act_pct * 100,
            "psi_contrib":  bucket_psi,
        })

        if total_psi < 0.10:
            verdict = "minimal shift — no action required"
        elif total_psi < 0.25:
            verdict = "moderate shift — monitor closely"
        else:
            verdict = "significant shift — model recalibration required"

        rationale = (
            f"PSI = {total_psi:.4f}: {verdict}. "
            f"A PSI < 0.10 is acceptable; > 0.25 indicates population drift "
            f"requiring model review under Basel III model risk management."
        )
        return total_psi, bucket_df, rationale

    # ------------------------------------------------------------------
    # 3. Basel III Capital Metrics
    # ------------------------------------------------------------------
    def calculate_basel_metrics(
        self,
        pd_val: Union[float, np.ndarray],
        lgd_val: Union[float, np.ndarray],
        ead_val: Union[float, np.ndarray]
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Computes Basel III Expected Loss and approximate Risk-Weighted Assets.

        Under the IRB approach:
          - EL  = PD × LGD × EAD
          - K   = LGD × N[(1-R)^{-0.5} × G(PD) + (R/(1-R))^{0.5} × G(0.999)] - PD × LGD
          - RWA = K × 12.5 × EAD

        For simplicity, the asset correlation R uses the BCBS retail formula.

        Parameters
        ----------
        pd_val : float or np.ndarray
            Probability of Default.
        lgd_val : float or np.ndarray
            Loss Given Default.
        ead_val : float or np.ndarray
            Exposure at Default.

        Returns
        -------
        Dict[str, float or np.ndarray]
            Keys: 'EL', 'K', 'RWA'.
        """
        from scipy.stats import norm

        pd_c  = np.clip(pd_val, 1e-6, 1 - 1e-6)
        lgd_c = np.clip(lgd_val, 0.0, 1.0)

        # Expected Loss
        EL = pd_c * lgd_c * ead_val

        # Asset correlation (BCBS retail formula)
        R = 0.03 * (1 - np.exp(-35 * pd_c)) / (1 - np.exp(-35)) + \
            0.16 * (1 - (1 - np.exp(-35 * pd_c)) / (1 - np.exp(-35)))

        # Capital requirement K
        G_pd  = norm.ppf(pd_c)
        G_999 = norm.ppf(0.999)
        try:
            K = lgd_c * norm.cdf(
                (1 - R)**(-0.5) * G_pd + (R / (1 - R))**0.5 * G_999
            ) - pd_c * lgd_c
        except Exception:
            K = EL / ead_val if np.any(ead_val > 0) else 0.0

        K   = np.clip(K, 0.0, None)
        RWA = K * 12.5 * ead_val

        return {"EL": EL, "K": K, "RWA": RWA}
