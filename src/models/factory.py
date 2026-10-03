from typing import Any

from src.models.liquid_network import LiquidECGClassifier


def build_model(config: dict[str, Any]) -> LiquidECGClassifier:
    data_cfg = config["data"]
    model_cfg = config["model"]
    stride = data_cfg.get("temporal_stride", 1)
    seq_len = data_cfg["num_features"] // stride
    return LiquidECGClassifier(
        seq_len=seq_len,
        input_size=1,
        hidden_size=model_cfg["hidden_size"],
        output_size=data_cfg["num_classes"],
        tau_min=model_cfg["tau_min"],
        pooling=model_cfg.get("pooling", "last"),
        temporal_stride=stride,
    )
