from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SystemConfig:
    """System-wide configuration parameters for CreditRiskEngine v2.

    Attributes
    ----------
    data_dir : str
        Directory for storing raw and processed datasets.
    kaggle_dataset : str
        Kaggle dataset slug for Lending Club data.
    vif_threshold : float
        VIF limit above which collinear features are dropped (Basel III: 5.0).
    iv_threshold : float
        Information Value threshold below which weak features are dropped.
    p_value_threshold : float
        Statistical significance level (alpha) for hypothesis tests.
    performance_window : int
        Months on book used to define the observation/performance period.
    random_state : int
        Global random seed for reproducibility.
    pytorch_epochs : int
        Training epochs for the PyTorch MLP.
    pytorch_lr : float
        Learning rate for PyTorch Adam optimizer.
    pytorch_batch_size : int
        Mini-batch size for PyTorch DataLoader.
    stress_macro_shocks : Dict[str, float]
        Default macroeconomic shock magnitudes for Pillar 2 stress testing.
    """

    data_dir: str = "data"
    kaggle_dataset: str = "wordsforthewise/lending-club"

    # Feature selection thresholds (tightened per Basel III best practice)
    vif_threshold: float = 5.0
    iv_threshold: float = 0.02
    p_value_threshold: float = 0.05

    # Vintage / performance window
    performance_window: int = 18  # months on book

    # Reproducibility
    random_state: int = 42

    # Neural network hyper-parameters
    pytorch_epochs: int = 20
    pytorch_lr: float = 0.005
    pytorch_batch_size: int = 64

    # Stress testing shocks (Pillar 2 macroeconomic scenarios)
    stress_macro_shocks: Dict[str, float] = field(
        default_factory=lambda: {
            "income_shock_pct": -20.0,       # % drop in borrower income
            "unemployment_rate_shock": 2.0,   # +2 pp unemployment rise
            "gdp_growth_shock": -3.0,         # % GDP growth decline
            "interest_rate_shock": 1.5,       # +1.5 pp central bank rate rise
        }
    )
