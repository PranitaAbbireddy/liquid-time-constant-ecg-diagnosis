import numpy as np
from sklearn.metrics import f1_score


def predict_from_logits(logits: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    if bias is not None:
        logits = logits + bias.reshape(1, -1)
    return logits.argmax(axis=1)


def tune_logit_biases(
    val_logits: np.ndarray,
    y_true: np.ndarray,  # validation labels only; must not include test split
    num_classes: int,
    seed: int = 42,
    n_restarts: int = 5,
    step_size: float = 0.25,
    max_iter: int = 80,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best_bias = np.zeros(num_classes, dtype=np.float64)
    best_score = f1_score(y_true, predict_from_logits(val_logits, best_bias), average="macro", zero_division=0)

    for _ in range(n_restarts):
        bias = rng.normal(0, 0.5, size=num_classes)
        score = f1_score(y_true, predict_from_logits(val_logits, bias), average="macro", zero_division=0)
        if score > best_score:
            best_score, best_bias = score, bias.copy()

        for _ in range(max_iter):
            improved = False
            for c in range(num_classes):
                for delta in (-step_size, step_size):
                    candidate = bias.copy()
                    candidate[c] += delta
                    cand_score = f1_score(
                        y_true,
                        predict_from_logits(val_logits, candidate),
                        average="macro",
                        zero_division=0,
                    )
                    if cand_score > score:
                        bias = candidate
                        score = cand_score
                        improved = True
            if not improved:
                break
        if score > best_score:
            best_score, best_bias = score, bias.copy()

    return best_bias.astype(np.float32)
