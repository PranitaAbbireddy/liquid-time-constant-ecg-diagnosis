from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients, Saliency
from torch.utils.data import DataLoader

from src.explainability.captum_wrappers import ECGCaptumWrapper
from src.utils.plotting import apply_publication_style, save_figure


def _select_class_samples(
    loader: DataLoader,
    samples_per_class: int,
    num_classes: int,
    seed: int,
) -> dict[int, torch.Tensor]:
    rng = np.random.default_rng(seed)
    collected: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_classes)}

    for signals, labels in loader:
        for signal, label in zip(signals, labels):
            label_int = int(label.item())
            if len(collected[label_int]) < samples_per_class:
                collected[label_int].append(signal.unsqueeze(0))
        if all(len(v) >= samples_per_class for v in collected.values()):
            break

    selected: dict[int, torch.Tensor] = {}
    for class_id in range(num_classes):
        if collected[class_id]:
            idx = int(rng.integers(0, len(collected[class_id])))
            selected[class_id] = collected[class_id][idx]
        else:
            selected[class_id] = torch.zeros(1, signals.size(1))
    return selected


def plot_integrated_gradients_per_class(
    model: torch.nn.Module,
    loader: DataLoader,
    class_names: dict[int, str],
    device: torch.device,
    graphs_dir: str,
    dpi: int,
    ig_steps: int,
    samples_per_class: int,
    seed: int,
    normalizer=None,
) -> Any:
    wrapped = ECGCaptumWrapper(model, normalizer).to(device)
    wrapped.eval()
    ig = IntegratedGradients(wrapped)
    samples = _select_class_samples(loader, samples_per_class, len(class_names), seed)

    apply_publication_style()
    fig, axes = plt.subplots(
        len(class_names),
        samples_per_class,
        figsize=(12, 2.2 * len(class_names)),
        squeeze=False,
    )

    for class_id, class_name in class_names.items():
        input_signal = samples[class_id].to(device)
        input_signal.requires_grad_(True)
        attributions, _ = ig.attribute(
            input_signal,
            target=class_id,
            n_steps=ig_steps,
            return_convergence_delta=True,
        )
        attr = attributions.squeeze().detach().cpu().numpy()
        signal = input_signal.squeeze().detach().cpu().numpy()
        for col in range(samples_per_class):
            ax = axes[class_id, col]
            ax.plot(signal, color="#4C72B0", alpha=0.6, label="ECG")
            ax.plot(attr, color="#C44E52", alpha=0.9, label="IG")
            ax.set_title(class_name)
            if class_id == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Integrated Gradients Attributions by Class", y=1.01)
    fig.tight_layout()
    return save_figure(fig, "integrated_gradients_per_class.png", "explainability", graphs_dir, dpi)


def plot_saliency_comparison(
    model: torch.nn.Module,
    loader: DataLoader,
    class_names: dict[int, str],
    device: torch.device,
    graphs_dir: str,
    dpi: int,
    ig_steps: int,
    seed: int,
    normalizer=None,
) -> Any:
    wrapped = ECGCaptumWrapper(model, normalizer).to(device)
    wrapped.eval()
    ig = IntegratedGradients(wrapped)
    sal = Saliency(wrapped)
    samples = _select_class_samples(loader, 1, len(class_names), seed)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    class_id = 2 if 2 in class_names else 0
    input_signal = samples[class_id].to(device)
    input_signal.requires_grad_(True)
    ig_attr, _ = ig.attribute(input_signal, target=class_id, n_steps=ig_steps, return_convergence_delta=True)
    sal_attr = sal.attribute(input_signal, target=class_id)
    signal = input_signal.squeeze().detach().cpu().numpy()
    ax.plot(signal, label="ECG", color="#4C72B0", alpha=0.7)
    ax.plot(ig_attr.squeeze().detach().cpu().numpy(), label="Integrated Gradients", color="#C44E52")
    ax.plot(sal_attr.squeeze().detach().cpu().numpy(), label="Saliency", color="#55A868")
    ax.set_title(f"IG vs Saliency ({class_names[class_id]})")
    ax.legend()
    return save_figure(fig, "ig_vs_saliency.png", "explainability", graphs_dir, dpi)
