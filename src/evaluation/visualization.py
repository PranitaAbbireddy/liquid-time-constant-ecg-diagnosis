from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from src.training.trainer import TrainHistory
from src.utils.plotting import apply_publication_style, save_figure


def plot_class_distribution(labels: np.ndarray, class_names: dict[int, str], graphs_dir: str, dpi: int):
    apply_publication_style()
    counts = np.bincount(labels.astype(int), minlength=len(class_names))
    names = [class_names[i] for i in range(len(class_names))]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, counts, color="#4C72B0")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    return save_figure(fig, "class_distribution.png", "dataset", graphs_dir, dpi)


def plot_sample_beats(
    signals: np.ndarray,
    labels: np.ndarray,
    class_names: dict[int, str],
    graphs_dir: str,
    dpi: int,
):
    apply_publication_style()
    fig, axes = plt.subplots(len(class_names), 1, figsize=(10, 2.2 * len(class_names)), sharex=True)
    if len(class_names) == 1:
        axes = [axes]
    for class_id in sorted(class_names.keys()):
        idx = np.where(labels == class_id)[0][0]
        ax = axes[class_id]
        ax.plot(signals[idx], color="#55A868")
        ax.set_ylabel(class_names[class_id])
    axes[-1].set_xlabel("Time step")
    fig.suptitle("Representative ECG Beats per Class", y=1.01)
    fig.tight_layout()
    return save_figure(fig, "sample_beats.png", "dataset", graphs_dir, dpi)


def plot_training_curves(history: TrainHistory, graphs_dir: str, dpi: int):
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history.train_loss) + 1)
    ax.plot(epochs, history.train_loss, marker="o", label="Train loss")
    ax.plot(epochs, history.val_loss, marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    return save_figure(fig, "loss_curves.png", "training", graphs_dir, dpi)


def plot_confusion_matrix(
    cm: list[list[int]],
    class_names: dict[int, str],
    graphs_dir: str,
    dpi: int,
    filename: str = "confusion_matrix.png",
    title: str = "Confusion Matrix (Test)",
):
    apply_publication_style()
    labels = [class_names[i] for i in sorted(class_names.keys())]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return save_figure(fig, filename, "evaluation", graphs_dir, dpi)


def plot_confusion_matrix_normalized(
    cm_norm: list[list[float]],
    class_names: dict[int, str],
    graphs_dir: str,
    dpi: int,
    filename: str = "confusion_matrix_normalized.png",
):
    apply_publication_style()
    labels = [class_names[i] for i in sorted(class_names.keys())]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix (Test, row-wise)")
    return save_figure(fig, filename, "evaluation", graphs_dir, dpi)


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: dict[int, str],
    graphs_dir: str,
    dpi: int,
    filename: str = "roc_curves.png",
):
    apply_publication_style()
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=8)
    return save_figure(fig, filename, "evaluation", graphs_dir, dpi)


def plot_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: dict[int, str],
    graphs_dir: str,
    dpi: int,
    filename: str = "precision_recall_curves.png",
):
    apply_publication_style()
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(num_classes):
        if y_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        pr_auc = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(recall, precision, label=f"{class_names[i]} (AP={pr_auc:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (One-vs-Rest)")
    ax.legend(loc="lower left", fontsize=8)
    return save_figure(fig, filename, "evaluation", graphs_dir, dpi)


def export_evaluation_figures(
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: dict[int, str],
    graphs_dir: str,
    dpi: int,
    prefix: str = "",
) -> None:
    pfx = f"{prefix}_" if prefix else ""
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        graphs_dir,
        dpi,
        filename=f"{pfx}confusion_matrix.png",
    )
    plot_confusion_matrix_normalized(
        metrics["confusion_matrix_normalized"],
        class_names,
        graphs_dir,
        dpi,
        filename=f"{pfx}confusion_matrix_normalized.png",
    )
    plot_roc_curves(y_true, y_prob, class_names, graphs_dir, dpi, filename=f"{pfx}roc_curves.png")
    plot_pr_curves(y_true, y_prob, class_names, graphs_dir, dpi, filename=f"{pfx}precision_recall_curves.png")
