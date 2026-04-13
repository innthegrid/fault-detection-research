# src/baselines/isolation_forest.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestBaseline:
    """
    Stronger, fairer baseline:
    - scale features
    - flatten windows (standard baseline approach)
    - normalized anomaly score output
    """

    def __init__(self, contamination=0.01):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def fit(self, train_x):
        """
        train_x: (N, T, F)
        """
        n = train_x.shape[0]
        x = train_x.reshape(n, -1)

        x = self.scaler.fit_transform(x)

        print(f"[IFOREST] Training on {x.shape}")
        self.model.fit(x)

    def score(self, test_x):
        """
        Returns: higher = more anomalous (0-1 normalized)
        """
        n = test_x.shape[0]
        x = test_x.reshape(n, -1)

        x = self.scaler.transform(x)

        raw = self.model.decision_function(x)

        # convert to anomaly score
        scores = -raw

        # normalize for fair comparison
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        return scores