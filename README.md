# Liquid Time-Constant Recurrent Modeling for ECG Arrhythmia Classification

A compact LTC-style liquid recurrent neural network for five-class ECG beat classification using MIT-BIH ECG data. The project implements a custom liquid recurrent cell with adaptive time constants, reproducible training and evaluation, class imbalance handling, model interpretability, and an interactive Streamlit demonstration.

---

# Technical Summary

This project explores a lightweight liquid recurrent architecture for automated classification of ECG beats into five arrhythmia-related categories. Each ECG beat contains 187 waveform samples that are processed as a temporal sequence using a custom liquid time-constant recurrent cell.

Unlike a standard RNN with fixed state transitions, the proposed recurrent cell learns an adaptive time constant (τ) that depends on both the current input and hidden state. This controls how quickly the hidden state changes over time.

The model uses:

- 64-dimensional liquid hidden state
- Temporal stride of 4
- Approximately 47 recurrent updates per ECG beat
- Approximately 8.7K trainable parameters

The training pipeline includes:

- Stratified 90/10 train-validation split
- Sqrt-inverse-frequency class-weighted cross-entropy loss
- Validation macro-F1 checkpoint selection
- Validation-only logit-bias calibration
- Final evaluation on a held-out test set

The final model achieved:

- Accuracy: **94.57%**
- Macro-F1: **74.95%**
- Macro ROC-AUC: **95.28%**
- Macro PR-AUC: **74.67%**

This architecture is inspired by Liquid Time-Constant Networks but is not a direct implementation of the canonical conductance-based LTC formulation.

---

# Project Overview

The goal of this project is to evaluate whether a compact liquid recurrent architecture can effectively model short ECG waveforms while maintaining a small parameter count.

The project includes:

- ECG data loading and preprocessing
- Stratified train-validation splitting
- Custom liquid recurrent model implementation
- Class imbalance handling
- Reproducible model training
- Validation-based checkpoint selection
- Validation-only logit-bias calibration
- Held-out test evaluation
- Confusion matrix analysis
- ROC analysis
- Precision-recall analysis
- Gradient-based explainability
- Experiment tracking
- Interactive Streamlit deployment

---

# Dataset

The project uses MIT-BIH-style ECG beat data containing 187 waveform samples per beat and five target classes.

| Class | Category |
|---------|---------|
| 0 | Normal |
| 1 | Supraventricular |
| 2 | Ventricular |
| 3 | Fusion |
| 4 | Unknown |

The official training file is split into:

- 90% Training
- 10% Validation
- Stratified sampling
- Random seed: 42

The official test file is reserved for final evaluation.

## Dataset Sizes

| Split | Samples |
|---------|---------:|
| Training | 78,798 |
| Validation | 8,756 |
| Test | 21,892 |

An exact duplicate check was performed between the training and test sets.

**Result:** Zero exact 187-sample waveform duplicates were found.

> **Note:** The dataset uses beat-level splitting rather than patient-level splitting. Therefore, the results do not represent patient-independent generalization.

---

# Model Architecture

The model processes each ECG beat as a temporal sequence.

```text
187-sample ECG beat
        │
        ▼
Temporal stride = 4
        │
        ▼
≈47 recurrent updates
        │
        ▼
Custom Liquid Recurrent Cell
        │
        ├── Input-dependent dynamics
        ├── Hidden-state-dependent dynamics
        ├── Adaptive time constant (τ)
        └── Nonlinear state update
        │
        ▼
64-dimensional hidden state
        │
        ▼
Last hidden state pooling
        │
        ▼
Linear classifier
        │
        ▼
5 ECG classes
```

## Configuration

| Parameter | Value |
|------------|--------|
| Input Length | 187 |
| Temporal Stride | 4 |
| Recurrent Steps | ~47 |
| Hidden Size | 64 |
| Pooling | Last Hidden State |
| Minimum τ | 0.5 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 128 |
| Epochs | 10 |
| Random Seed | 42 |
| Trainable Parameters | ~8.7K |

The small parameter count makes the model computationally efficient while preserving recurrent temporal dynamics.

---

# LTC-Style vs Canonical LTC

This project implements a custom LTC-style recurrent architecture rather than the canonical Liquid Time-Constant Network.

The model incorporates:

- Adaptive time constants
- Input-dependent dynamics
- Hidden-state-dependent dynamics

However, it does not implement the full conductance-based differential equation formulation of canonical LTC networks.

This distinction is important to avoid overstating the architecture.

Future work may include a direct comparison against a canonical LTC implementation under the same training and evaluation setup.

---

# Class Imbalance Handling

The dataset contains significant class imbalance, especially for the Fusion and Supraventricular classes.

To address this, the model uses:

**Sqrt-Inverse-Frequency Class-Weighted Cross-Entropy**

This balances learning by:

- Reducing majority-class dominance
- Avoiding extreme weighting of rare classes
- Improving macro-level performance

Validation macro-F1 is used as the primary checkpoint selection metric because it gives equal importance to all classes.

---

# Validation and Test Protocol

The evaluation pipeline separates model development from final testing.

## Training

Training data is used to optimize model parameters.

## Validation

Validation data is used for:

- Checkpoint selection
- Macro-F1 monitoring
- Logit-bias calibration
- Model development decisions

## Test

The test set is used only for final performance reporting.

The final model consists of:

- Frozen validation-selected checkpoint
- Validation-derived logit bias

No test labels are used to fit the logit bias.

### Development Transparency

During early experimentation, exploratory evaluations were performed on the test set.

Therefore, the project does not claim that the test set was never viewed during development.

The final reported model follows a validation-first selection protocol with frozen final evaluation.

---

# Final Test Results

Evaluation was performed on **21,892 held-out ECG beats**.

## Overall Performance

| Metric | Result |
|----------|---------|
| Accuracy | 94.57% |
| Weighted F1 | 94.43% |
| Macro Precision | 78.35% |
| Macro Recall | 72.93% |
| Macro F1 | 74.95% |
| Macro ROC-AUC | 95.28% |
| Macro PR-AUC | 74.67% |

---

# Per-Class Performance

| Class | Precision | Recall | F1 |
|---------|---------:|---------:|---------:|
| Normal | 96.10% | 98.20% | 97.14% |
| Supraventricular | 74.47% | 50.36% | 60.09% |
| Ventricular | 83.39% | 80.46% | 81.90% |
| Fusion | 38.65% | 49.38% | 43.36% |
| Unknown | 99.14% | 86.26% | 92.25% |

---

# Per-Class ROC-AUC

| Class | ROC-AUC |
|---------|---------:|
| Normal | 95.79% |
| Supraventricular | 89.01% |
| Ventricular | 97.02% |
| Fusion | 95.76% |
| Unknown | 98.83% |

---

# Per-Class PR-AUC

| Class | PR-AUC |
|---------|---------:|
| Normal | 98.79% |
| Supraventricular | 60.16% |
| Ventricular | 86.23% |
| Fusion | 33.09% |
| Unknown | 95.08% |

---

# Error Analysis

Performance varies significantly across classes.

- Normal class achieved the highest F1 score (97.14%).
- Ventricular beats achieved strong performance (81.90% F1).
- Supraventricular and Fusion classes remain challenging.

Fusion is the most difficult class:

- F1: 43.36%
- ROC-AUC: 95.76%
- PR-AUC: 33.09%

The high ROC-AUC but low PR-AUC and F1 suggest that the model can rank Fusion beats reasonably well but still struggles to make confident final predictions because of severe class imbalance.

The confusion matrix shows that rare classes are frequently misclassified as Normal beats.

These results demonstrate that accuracy alone is not sufficient for evaluating imbalanced ECG classification tasks.

---

# Model Interpretability

The project includes gradient-based explainability methods to identify which parts of the ECG waveform influence predictions.

Implemented methods:

- Integrated Gradients
- Saliency Maps

These methods provide waveform-level attribution visualizations that support model analysis and interpretation.

The explainability outputs are intended for research purposes only and should not be interpreted as clinical evidence.

---

# Reproducibility

The project follows a reproducible experimental setup.

Key settings include:

- Random seed: 42
- Fixed train-validation split
- Explicit model configuration
- Documented class weighting
- Validation-based checkpoint selection
- Validation-only calibration
- Frozen final checkpoint
- Machine-readable evaluation reports
- Automated testing
- Reproducible figure generation

All final results can be reproduced from the saved checkpoint and validation-derived calibration parameters without retraining.

---

# Project Structure

```text
LNN ECG Diagnosis/
│
├── app/
│   └── streamlit_app.py
│
├── checkpoints/
│   └── best_model.pt
│
├── graphs/
│   ├── dataset/
│   ├── training/
│   ├── evaluation/
│   └── explainability/
│
├── mitbih/
│   ├── mitbih_train.csv
│   └── mitbih_test.csv
│
├── notebooks/
│   └── ECG_Liquid_Network_Analysis.ipynb
│
├── reports/
│   ├── metrics/
│   ├── tables/
│   └── summaries/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── run_experiments.py
│   ├── build_experiment_table.py
│   ├── apply_val_bias_checkpoint.py
│   ├── verify_logit_bias_provenance.py
│   └── finalize_selected_model.py
│
├── src/
│   └── ...
│
├── tests/
│   └── ...
│
├── requirements.txt
└── README.md
```

Earlier notebooks and a Flask prototype are retained in `Previous code/` for historical reference.

---

# Installation

```bash
cd "LNN ECG Diagnosis"

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

pip install -e .
```

---

# Training

Run full training:

```bash
python scripts/train.py
```

Quick smoke test:

```bash
python scripts/train.py --epochs 2 --max-batches 40
```

Configuration file:

```text
src/config/default.yaml
```

---

# Evaluation

Evaluate a saved checkpoint:

```bash
python scripts/evaluate.py
```

Reproduce the final evaluation workflow:

```bash
python scripts/build_experiment_table.py

python scripts/finalize_selected_model.py
```

Verify logit-bias provenance:

```bash
python scripts/verify_logit_bias_provenance.py
```

Evaluation protocol:

```text
reports/summaries/evaluation_protocol.md
```

---

# Experiments

Run the experiment grid:

```bash
python scripts/run_experiments.py
```

Build experiment comparison tables:

```bash
python scripts/build_experiment_table.py
```

Generate validation-tuned logit biases:

```bash
python scripts/apply_val_bias_checkpoint.py
```

Historical experiments are retained for comparison and reproducibility.

---

# Streamlit Demo

Launch the application:

```bash
streamlit run app/streamlit_app.py
```

The application allows users to:

- Load ECG waveforms
- View model predictions
- Explore model analysis outputs

The demo is intended for research and education only.

It is not a clinical diagnostic system.

---

# Research Notebook

Main notebook:

```text
notebooks/ECG_Liquid_Network_Analysis.ipynb
```

Contents include:

- Dataset exploration
- Class distribution analysis
- ECG visualization
- Model training analysis
- Validation evaluation
- Confusion matrices
- ROC curves
- Precision-recall curves
- Per-class performance analysis
- Explainability visualizations
- Error analysis

The notebook clearly separates validation analysis from final test evaluation.

---

# Generated Artifacts

Generated outputs are stored in:

```text
checkpoints/
graphs/
reports/
```

Artifacts include:

- Model checkpoints
- Training curves
- Validation curves
- Dataset plots
- Confusion matrices
- ROC curves
- Precision-recall curves
- Performance tables
- Explainability visualizations
- Evaluation reports
- Experiment comparison tables
- Protocol documentation

---

# Research Contribution

The main contribution of this project is the design and evaluation of a compact adaptive liquid recurrent architecture for ECG beat classification.

Key contributions:

- Lightweight liquid recurrent architecture
- Adaptive input- and hidden-state-dependent time constants
- Severe class imbalance handling
- Validation-based model selection
- Validation-only calibration
- Held-out test evaluation
- Gradient-based interpretability
- Reproducible experiment workflow

The final model uses approximately **8.7K parameters** while maintaining strong classification performance.

---

# Limitations

The following limitations should be considered:

- The model is LTC-style, not a canonical LTC implementation.
- The dataset uses beat-level rather than patient-level splits.
- Results do not demonstrate patient-independent generalization.
- Fusion remains difficult to classify.
- Accuracy alone does not fully describe performance.
- Explainability methods do not establish clinical validity.
- The model has not been clinically validated.
- No prospective clinical evaluation has been performed.
- The system must not be used for medical diagnosis or treatment decisions.

---

# Future Work

Potential future directions include:

- Implementing a canonical LTC network
- Comparing against LSTM, GRU, and CNN baselines
- Performing patient-level evaluation
- Evaluating additional ECG datasets
- Investigating class-specific decision thresholds
- Improving minority-class performance
- Studying uncertainty estimation
- Optimizing CPU and edge-device deployment
- Expanding interpretability analysis
- Conducting statistical significance testing

---

# Disclaimer

This repository is intended for research and educational purposes only.

It is not a medical device, has not been clinically validated, and must not be used for clinical diagnosis, treatment, or medical decision-making.