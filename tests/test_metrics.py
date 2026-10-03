import numpy as np

from src.evaluation.metrics import compute_metrics


def test_compute_metrics_macro_f1():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 1])
    y_prob = np.array(
        [
            [0.9, 0.05, 0.03, 0.01, 0.01],
            [0.1, 0.8, 0.05, 0.03, 0.02],
            [0.05, 0.1, 0.8, 0.03, 0.02],
            [0.4, 0.4, 0.1, 0.05, 0.05],
        ]
    )
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=5)
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["accuracy"] == 0.75
