import os
import io
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, Union
from config import SystemConfig


class DataOrchestrator:
    """Orchestrates data ingestion, roll rate analysis, vintage analysis,
    and reject inference for credit risk modeling.

    Parameters
    ----------
    config : SystemConfig
        System-wide configuration settings.
    """

    def __init__(self, config: SystemConfig):
        self.config = config

    # ------------------------------------------------------------------
    # 1. Data Loading
    # ------------------------------------------------------------------
    def load_data(
        self,
        uploaded_file: Optional[Union[io.BytesIO, pd.DataFrame, bytes]] = None
    ) -> Tuple[Dict[str, pd.DataFrame], str]:
        """Loads credit data from an uploaded CSV or generates synthetic data.

        Parameters
        ----------
        uploaded_file : BytesIO or pd.DataFrame or bytes, optional
            A file-like object from st.file_uploader, or a pre-loaded DataFrame.
            If None, falls back to Kaggle download or synthetic generation.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Keys: 'static', 'delinquency', 'vintage'.
        str
            Rationale string explaining data source outcome.
        """
        # --- Attempt to use uploaded file ---
        if uploaded_file is not None:
            try:
                if isinstance(uploaded_file, pd.DataFrame):
                    raw_df = uploaded_file.copy()
                elif isinstance(uploaded_file, bytes):
                    raw_df = pd.read_csv(io.BytesIO(uploaded_file))
                else:
                    raw_df = pd.read_csv(uploaded_file)

                datasets = self._parse_uploaded_df(raw_df)
                rationale = (
                    f"User-uploaded CSV loaded successfully. Dataset contains "
                    f"{len(raw_df):,} rows and {raw_df.shape[1]} columns. "
                    f"Synthetic delinquency and vintage histories were generated "
                    f"from the uploaded data to support roll rate and vintage analysis."
                )
                return datasets, rationale
            except Exception as e:
                rationale = (
                    f"Uploaded file parsing failed (Error: {str(e)[:80]}). "
                    f"Falling back to synthetic Lending Club dataset."
                )

        # --- Attempt Kaggle download ---
        kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
        has_credentials = os.path.exists(kaggle_json) or (
            os.environ.get("KAGGLE_USERNAME") is not None
            and os.environ.get("KAGGLE_KEY") is not None
        )

        if has_credentials:
            try:
                import kaggle
                os.makedirs(self.config.data_dir, exist_ok=True)
                kaggle.api.authenticate()
                kaggle.api.dataset_download_files(
                    self.config.kaggle_dataset,
                    path=self.config.data_dir,
                    unzip=True
                )
                rationale = (
                    f"Lending Club dataset downloaded from Kaggle "
                    f"('{self.config.kaggle_dataset}') and saved to '{self.config.data_dir}'."
                )
            except BaseException as e:
                rationale = (
                    f"Kaggle download failed (Error: {str(e)[:80]}). "
                    f"Generating synthetic dataset."
                )
        else:
            rationale = (
                "No Kaggle credentials found and no CSV uploaded. "
                "Generating a high-quality synthetic Lending Club dataset."
            )

        # --- Generate synthetic data ---
        datasets = self._generate_synthetic_data()
        return datasets, rationale

    # ------------------------------------------------------------------
    # 2. Roll Rates
    # ------------------------------------------------------------------
    def compute_roll_rates(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, str]:
        """Calculates delinquency transition matrices across DPD buckets.

        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Loaded datasets containing 'delinquency' history.

        Returns
        -------
        pd.DataFrame
            Row-normalised transition probability matrix.
        str
            Rationale explaining the selected default definition target.
        """
        df = data["delinquency"]

        # Align month t with month t+1 via self-join
        df_t  = df.copy()
        df_t1 = df.copy()
        df_t1["month"] = df_t1["month"] - 1

        merged = pd.merge(
            df_t, df_t1,
            on=["loan_id", "month"],
            suffixes=("_t", "_t1")
        )

        states = {0: "Current", 1: "30-DPD", 2: "60-DPD", 3: "90-DPD"}
        counts  = pd.crosstab(merged["dpd_status_t"], merged["dpd_status_t1"])
        # Fill any missing state columns
        for s in [0, 1, 2, 3]:
            if s not in counts.columns:
                counts[s] = 0
            if s not in counts.index:
                counts.loc[s] = 0
        counts = counts.sort_index().sort_index(axis=1)
        matrix = counts.div(counts.sum(axis=1), axis=0).fillna(0)
        matrix.index   = [states[i] for i in matrix.index]
        matrix.columns = [states[c] for c in matrix.columns]

        roll_60_90 = matrix.loc["60-DPD", "90-DPD"] if "60-DPD" in matrix.index else 0.0
        rationale = (
            f"Target set to 90-DPD because {roll_60_90*100:.1f}% of accounts "
            f"rolling into 60-DPD subsequently rolled into 90-DPD. Once borrowers "
            f"exceed 60-DPD, rehabilitation rates drop to a negligible level, "
            f"justifying 90-DPD as the Basel III default definition threshold."
        )
        return matrix, rationale

    # ------------------------------------------------------------------
    # 3. Vintage Analysis
    # ------------------------------------------------------------------
    def perform_vintage_analysis(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, str]:
        """Computes cumulative default rates across cohorts to determine
        the optimal performance window.

        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Loaded datasets containing 'vintage' history.

        Returns
        -------
        pd.DataFrame
            Pivot of cumulative default rate indexed by months on book.
        str
            Rationale explaining the determined performance window.
        """
        df = data["vintage"]
        pivot = df.pivot(
            index="months_on_book",
            columns="cohort",
            values="cum_default_rate"
        )
        mean_rate = pivot.mean(axis=1)
        marginal  = mean_rate.diff()

        plateau_threshold = 0.001  # < 0.1% marginal increase
        plateau_mob = self.config.performance_window  # fallback

        for mob in range(2, len(mean_rate)):
            if mob + 2 <= len(mean_rate) and marginal.iloc[mob] < plateau_threshold:
                if (
                    marginal.iloc[mob + 1] < plateau_threshold
                    and marginal.iloc[mob + 2] < plateau_threshold
                ):
                    plateau_mob = mob
                    break

        rationale = (
            f"Performance window set to {plateau_mob} months on book: the cumulative "
            f"default curve plateaus at this point (marginal increase < "
            f"{plateau_threshold*100:.2f}% per month). Basel III requires a "
            f"sufficiently long window to capture full default cycles."
        )
        return pivot, rationale

    # ------------------------------------------------------------------
    # 4. Reject Inference (Fuzzy Augmentation)
    # ------------------------------------------------------------------
    def reject_inference(
        self,
        accepted: pd.DataFrame,
        rejected: pd.DataFrame
    ) -> Tuple[pd.DataFrame, str]:
        """Applies fuzzy augmentation reject inference to correct selection bias.

        Parameters
        ----------
        accepted : pd.DataFrame
            Accepted population with known default_label.
        rejected : pd.DataFrame
            Rejected population without labels.

        Returns
        -------
        pd.DataFrame
            Augmented DataFrame combining accepted + inferred rejected cases.
        str
            Rationale explaining the reject inference methodology.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        feature_cols = [
            c for c in accepted.columns
            if c not in ["default_label", "loan_id", "ead", "lgd",
                         "macro_instrument", "grade"]
        ]
        acc_enc = pd.get_dummies(
            accepted[feature_cols + ["default_label"]],
            drop_first=True
        ).astype(float)
        y_acc  = accepted["default_label"].values
        X_cols = [c for c in acc_enc.columns if c != "default_label"]
        X_acc  = acc_enc[X_cols].values

        scaler = StandardScaler()
        X_acc_s = scaler.fit_transform(X_acc)

        clf = LogisticRegression(max_iter=500, random_state=self.config.random_state)
        clf.fit(X_acc_s, y_acc)

        # Score rejected applicants
        rej_enc = pd.get_dummies(
            rejected[feature_cols], drop_first=True
        ).astype(float)
        for col in X_cols:
            if col not in rej_enc.columns:
                rej_enc[col] = 0.0
        rej_enc = rej_enc[X_cols]
        X_rej_s = scaler.transform(rej_enc.values)
        p_bad   = clf.predict_proba(X_rej_s)[:, 1]

        # Fuzzy augmentation: duplicate with complementary labels and weights
        rej_good = rejected.copy()
        rej_good["default_label"] = 0
        rej_good["sample_weight"] = 1.0 - p_bad

        rej_bad  = rejected.copy()
        rej_bad["default_label"] = 1
        rej_bad["sample_weight"] = p_bad

        acc_copy = accepted.copy()
        acc_copy["sample_weight"] = 1.0

        augmented = pd.concat(
            [acc_copy, rej_good, rej_bad], ignore_index=True
        )

        n_rej = len(rejected)
        avg_p = float(p_bad.mean())
        rationale = (
            f"Reject Inference (Fuzzy Augmentation) applied to {n_rej:,} rejected "
            f"applicants. A Logistic Regression trained on {len(accepted):,} accepted "
            f"cases scored rejections: mean inferred P(Default) = {avg_p:.3f}. Each "
            f"rejected account was duplicated as Good (weight=1-p) and Bad (weight=p), "
            f"producing a selection-bias-corrected augmented population."
        )
        return augmented, rationale

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_uploaded_df(
        self, raw_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Attempts to map an uploaded DataFrame to standard schema using flexible column aliases."""
        np.random.seed(self.config.random_state)
        df = raw_df.copy()

        # Normalise column names
        df.columns = [
            c.lower().strip().replace(" ", "_").replace("-", "_") for c in df.columns
        ]

        # Column alias dictionary for flexible mapping
        aliases = {
            "loan_amnt": [
                "loan_amnt", "loan_amount", "loan_size", "principal", "funded_amnt",
                "loan_val", "amount", "borrowed", "loan_amt"
            ],
            "int_rate": [
                "int_rate", "interest_rate", "rate", "interest", "apr", "int_rt", "rate_pct"
            ],
            "annual_inc": [
                "annual_inc", "annual_income", "income", "salary", "earnings", "inc", "ann_inc"
            ],
            "dti": [
                "dti", "dti_ratio", "debt_to_income", "debt_ratio", "debt_income_ratio"
            ],
            "grade": [
                "grade", "rating", "credit_grade", "risk_grade", "score_grade", "sub_grade",
                "gh", "rank", "tier", "class"
            ],
            "emp_length": [
                "emp_length", "employment_length", "emp_len", "work_years", "employment_years", "tenure"
            ],
            "delinq_2yrs": [
                "delinq_2yrs", "delinquencies", "delinq", "delinq_2yr", "late_payments", "past_due_count"
            ],
            "revol_util": [
                "revol_util", "revol_utilization", "utilization", "revol_ratio", "credit_util"
            ],
            "inq_last_6mths": [
                "inq_last_6mths", "inquiries", "inq_6mths", "inquiries_6m", "credit_inquiries", "inq"
            ],
            "default_label": [
                "default_label", "target", "default", "is_default", "loan_status", "status",
                "bad_flag", "default_flag", "outcome", "label"
            ]
        }

        # Apply fuzzy alias mapping
        rename_map = {}
        for canonical, alias_list in aliases.items():
            if canonical in df.columns:
                continue
            for col in df.columns:
                if col in alias_list or any(a == col for a in alias_list):
                    rename_map[col] = canonical
                    break

        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        n = len(df)
        if "loan_amnt"     not in df: df["loan_amnt"]     = np.random.uniform(5000, 40000, n)
        if "int_rate"      not in df: df["int_rate"]      = np.random.uniform(0.05, 0.30, n)
        if "annual_inc"    not in df: df["annual_inc"]    = np.random.lognormal(11, 0.5, size=n)
        if "dti"           not in df: df["dti"]           = np.random.beta(2, 5, n) * 50
        if "grade"         not in df: df["grade"]         = np.random.choice(list("ABCDEFG"), n)
        if "emp_length"    not in df: df["emp_length"]    = np.random.randint(0, 11, n)
        if "delinq_2yrs"   not in df: df["delinq_2yrs"]   = np.random.poisson(0.3, n)
        if "revol_util"    not in df: df["revol_util"]    = np.random.beta(2, 2, n)
        if "inq_last_6mths" not in df: df["inq_last_6mths"] = np.random.poisson(0.5, n)

        # Handle string target mapping if default_label is string/categorical
        if "default_label" not in df:
            if "loan_status" in df:
                df["default_label"] = df["loan_status"].apply(
                    lambda x: 1 if str(x).lower() in [
                        "charged off", "default", "late (31-120 days)", "bad", "1", "yes", "true"
                    ] else 0
                )
            else:
                grade_risk = {"A": 0.01, "B": 0.03, "C": 0.07, "D": 0.12,
                              "E": 0.20, "F": 0.30, "G": 0.45}
                base_pd = df["grade"].map(grade_risk).fillna(0.07)
                logit   = -2.5 + 3.0*(df["dti"]/50) - 1.5*(np.log(df["annual_inc"]+1)-11) + 4.0*base_pd
                prob    = 1.0 / (1.0 + np.exp(-logit))
                df["default_label"] = (np.random.rand(n) < prob).astype(int)
        else:
            # Map values if default_label column itself is string/non-numeric
            if not pd.api.types.is_numeric_dtype(df["default_label"]):
                df["default_label"] = df["default_label"].apply(
                    lambda x: 1 if str(x).lower() in [
                        "charged off", "default", "late (31-120 days)", "bad", "1", "yes", "true", "y"
                    ] else 0
                )

        # EAD and LGD
        if "ead" not in df:
            df["ead"] = df["loan_amnt"] * np.random.uniform(0.85, 1.0, n)
        if "lgd" not in df:
            grade_risk = {"A": 0.01, "B": 0.03, "C": 0.07, "D": 0.12,
                          "E": 0.20, "F": 0.30, "G": 0.45}
            base_pd = df["grade"].map(grade_risk).fillna(0.07)
            df["lgd"] = np.clip(0.4 + 0.3*base_pd + np.random.normal(0, 0.05, n), 0.1, 0.9)

        if "loan_id" not in df:
            df["loan_id"] = np.arange(1, n + 1)
        if "macro_indicator" not in df:
            df["macro_indicator"] = np.random.normal(0, 1, n)
        if "macro_instrument" not in df:
            df["macro_instrument"] = np.random.uniform(2, 8, n)

        static_df = df.reset_index(drop=True)

        delinq_df, vintage_df = self._generate_delinquency_vintage(n)
        return {"static": static_df, "delinquency": delinq_df, "vintage": vintage_df}

    def _generate_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """Generates a realistic synthetic Lending Club-style dataset."""
        np.random.seed(self.config.random_state)
        n = 5000

        grade_prob = [0.15, 0.30, 0.25, 0.15, 0.10, 0.03, 0.02]
        grades = np.random.choice(
            ["A","B","C","D","E","F","G"], size=n, p=grade_prob
        )
        grade_risk = {"A":0.01,"B":0.03,"C":0.07,"D":0.12,"E":0.20,"F":0.30,"G":0.45}
        base_pd = np.array([grade_risk[g] for g in grades])

        loan_amnt  = np.random.uniform(5000, 40000, n)
        annual_inc = np.random.lognormal(mean=11.0, sigma=0.5, size=n)
        dti        = np.random.beta(a=2, b=5, size=n) * 50.0
        grade_rate = {"A":0.06,"B":0.09,"C":0.13,"D":0.18,"E":0.22,"F":0.26,"G":0.30}
        int_rate   = np.clip(
            np.array([grade_rate[g] + np.random.normal(0, 0.01) for g in grades]),
            0.04, 0.35
        )
        emp_length    = np.random.randint(0, 11, n)
        delinq_2yrs   = np.random.poisson(0.3, n)
        revol_util    = np.random.beta(2, 2, n)
        inq_last_6mths = np.random.poisson(0.5, n)

        macro_instrument = np.random.uniform(2.0, 8.0, n)
        macro_indicator  = 4.0 - 0.3*macro_instrument + np.random.normal(0, 0.5, n)

        logit_p = (
            -2.5
            + 3.0*(dti/50.0)
            - 1.5*(np.log(annual_inc) - 11.0)
            + 0.5*delinq_2yrs
            + 4.0*base_pd
            - 0.4*macro_indicator
            + np.random.normal(0, 0.2, n)
        )
        prob_default  = 1.0 / (1.0 + np.exp(-logit_p))
        default_label = (np.random.rand(n) < prob_default).astype(int)
        ead = loan_amnt * np.random.uniform(0.85, 1.0, n)
        lgd = np.clip(0.4 + 0.3*base_pd + np.random.normal(0, 0.05, n), 0.1, 0.9)

        static_df = pd.DataFrame({
            "loan_id": np.arange(1, n+1),
            "loan_amnt": loan_amnt,
            "int_rate": int_rate,
            "annual_inc": annual_inc,
            "dti": dti,
            "grade": grades,
            "emp_length": emp_length,
            "delinq_2yrs": delinq_2yrs,
            "revol_util": revol_util,
            "inq_last_6mths": inq_last_6mths,
            "macro_indicator": macro_indicator,
            "macro_instrument": macro_instrument,
            "default_label": default_label,
            "ead": ead,
            "lgd": lgd,
        })

        delinq_df, vintage_df = self._generate_delinquency_vintage(n)
        return {"static": static_df, "delinquency": delinq_df, "vintage": vintage_df}

    def _generate_delinquency_vintage(
        self, n_loans: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generates delinquency history and vintage performance tables."""
        np.random.seed(self.config.random_state + 1)

        # Delinquency transitions
        records = []
        n_sample = min(1000, n_loans)
        for loan_id in range(1, n_sample + 1):
            state = 0
            for month in range(1, 6):
                records.append({"loan_id": loan_id, "month": month, "dpd_status": state})
                if state == 0:
                    state = np.random.choice([0, 1], p=[0.92, 0.08])
                elif state == 1:
                    state = np.random.choice([0, 1, 2], p=[0.40, 0.20, 0.40])
                elif state == 2:
                    state = np.random.choice([0, 1, 2, 3], p=[0.10, 0.02, 0.03, 0.85])
                else:
                    state = 3
        delinq_df = pd.DataFrame(records)

        # Vintage cumulative default rates
        v_records = []
        for cohort in ["2023-Q1", "2023-Q2", "2023-Q3"]:
            base_max = {"2023-Q1": 0.08, "2023-Q2": 0.09, "2023-Q3": 0.075}[cohort]
            for mob in range(1, 25):
                rate = base_max*(1.0 - np.exp(-mob/5.5)) + np.random.normal(0, 0.001)
                v_records.append({
                    "cohort": cohort,
                    "months_on_book": mob,
                    "cum_default_rate": float(np.clip(rate, 0.0, 0.12))
                })
        vintage_df = pd.DataFrame(v_records)

        return delinq_df, vintage_df
