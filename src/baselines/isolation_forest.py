# src/baselines/isolation_forest.py
import numpy as np
from sklearn.ensemble import IsolationForest

def train_isolation_forest(train_x, contamination=0.01):
    """
    Standard statistical baseline.
    Note: Isolation Forest expects 2D input (Samples x Features).
    We flatten the windowed temporal data for this baseline.
    """
    n_samples, n_window, n_features = train_x.shape
    train_flat = train_x.reshape(n_samples, -1)
    
    print(f"Training Isolation Forest on {n_samples} samples...")
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train_flat)
    return model

def get_iforest_scores(model, test_x):
    """
    Returns anomaly scores. Higher = more anomalous.
    """
    n_samples = test_x.shape[0]
    test_flat = test_x.reshape(n_samples, -1)
    
    raw_scores = model.decision_function(test_flat)
    return -raw_scores