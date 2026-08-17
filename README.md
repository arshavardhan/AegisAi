# 🛡️ AegisAI

### A Reliability & Trust Layer for Machine Learning Predictions

AegisAI is a lightweight, modular reliability layer that evaluates **how trustworthy a machine-learning prediction is** before it reaches a downstream application.

Instead of treating every model prediction as equally reliable, AegisAI analyzes multiple signals—including **confidence, predictive uncertainty, and out-of-distribution (OOD) risk**—and converts them into a structured trust assessment.

> **Your model makes the prediction. AegisAI evaluates whether that prediction should be trusted.**

---

## Why AegisAI?

Machine-learning models can produce predictions even when:

* The model is uncertain.
* The input is significantly different from its training data.
* The predicted probability is misleading.
* The prediction should be reviewed rather than automatically accepted.

Traditional ML pipelines often expose only:

```text
prediction → application
```

AegisAI introduces a reliability layer:

```text
                    ┌─────────────────────┐
                    │       Input         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     ML Model        │
                    └──────────┬──────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │   AegisAI Reliability   │
                  │        Engine           │
                  ├─────────────────────────┤
                  │ • Confidence            │
                  │ • Uncertainty           │
                  │ • OOD Risk              │
                  │ • Trust Scoring         │
                  └────────────┬────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Trust Score      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Risk + Recommendation│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Prediction Report   │
                    └─────────────────────┘
```

---

## Current Status

**Version: `0.1.0` — ML Reliability Core**

AegisAI currently focuses on classification models that expose probability estimates, with primary support for scikit-learn estimators.

### Implemented

* Scikit-learn model wrapper
* Prediction pipeline
* Confidence estimation
* Normalized predictive entropy
* Z-score based OOD detection
* Normalized OOD risk
* Configurable trust-score aggregation
* Risk classification
* Action recommendations
* Structured Pydantic prediction reports
* Batch prediction
* Feature/input validation
* Prediction and probability validation
* Error handling
* Automated tests
* GitHub Actions CI
* Python package configuration

---

## Core Reliability Signals

### 1. Confidence

AegisAI measures classification confidence using the model's maximum predicted class probability.

```text
confidence = max(class probabilities)
```

Higher confidence generally indicates that the model strongly favors one class.

Confidence alone is **not treated as sufficient evidence of reliability**.

---

### 2. Uncertainty

AegisAI uses normalized Shannon entropy to measure predictive uncertainty.

```text
Low entropy  → model is more decisive
High entropy → model is more uncertain
```

This allows AegisAI to distinguish between predictions that have similar confidence characteristics but different probability distributions.

---

### 3. Out-of-Distribution Risk

AegisAI currently provides a lightweight Z-score based OOD detector.

The detector learns feature statistics from reference data and evaluates how far new observations deviate from those statistics.

The raw detector distance is kept separate from the normalized OOD risk used by the trust-scoring system.

> The current OOD detector is intentionally a lightweight baseline. It is not presented as a universal OOD detection solution.

---

## Trust Score

AegisAI combines the reliability signals into a normalized score between `0` and `1`.

The current default configuration uses:

| Signal      | Weight |
| ----------- | -----: |
| Confidence  |   0.50 |
| Uncertainty |   0.30 |
| OOD Risk    |   0.20 |

Conceptually:

```text
Trust
  │
  ├── Model Confidence
  ├── Predictive Uncertainty
  └── OOD Risk
          │
          ▼
     Trust Scoring
          │
          ▼
    Risk Classification
          │
          ▼
    Recommendation
```

Drift is deliberately not included in the current trust score because live drift monitoring is not yet connected to the prediction pipeline.

---

## Quick Start

### Installation

Clone the repository:

```bash
git clone https://github.com/arshavardhan/AegisAi.git
cd AegisAi
```

Install AegisAI:

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

---

## Basic Usage

AegisAI can wrap a supported classification model and evaluate its predictions.

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from aegis import AegisModel


# Load data
X, y = load_iris(return_X_y=True)

# Train your ML model
classifier = LogisticRegression(max_iter=500)
classifier.fit(X, y)

# Create AegisAI reliability layer
model = AegisModel(classifier)

# Fit AegisAI's reference reliability components
model.fit(X)

# Evaluate a prediction
report = model.predict(X[:1])

print(report.summary())
```

Example output:

```text
Prediction=0,
Confidence=0.978,
Uncertainty=0.119,
OOD=False,
OOD Risk=0.102,
Trust=0.894,
Risk=low,
Recommendation=accept
```

The exact values depend on the model and input.

---

## Structured Prediction Reports

AegisAI returns a structured `PredictionReport` instead of an unstructured result.

A report contains:

```text
prediction
confidence
uncertainty
ood
ood_score
trust_score
risk_level
recommendation
metadata
```

Reports can be consumed as Python dictionaries:

```python
data = report.to_dict()
```

Or serialized as JSON:

```python
print(report.to_json())
```

This makes AegisAI suitable for integration with downstream applications, APIs, monitoring systems, and decision pipelines.

---

## Batch Prediction

AegisAI also supports evaluating multiple samples:

```python
reports = model.predict_batch(X[:10])

for report in reports:
    print(report.summary())
```

Each input receives an independent reliability assessment.

---

## Design Philosophy

AegisAI is built around five principles.

### 1. Model-Agnostic Reliability

Reliability logic should remain separate from the underlying ML estimator.

### 2. Explicit Signals

Confidence, uncertainty, and distribution risk should remain independently observable.

### 3. Deterministic Decisions

Given the same reliability signals and configuration, AegisAI should produce the same trust decision.

### 4. Small Public API

Users should be able to evaluate model reliability without needing to understand AegisAI's internal implementation.

### 5. Honest Capability Boundaries

AegisAI should clearly distinguish between:

```text
Implemented
    ↓
Experimental
    ↓
Planned
```

The project does not claim reliability capabilities that it has not actually implemented or evaluated.

---

## Architecture

```text
aegis/
│
├── config/
│   └── Configuration and scoring settings
│
├── core/
│   ├── Pipeline
│   ├── Prediction Reports
│   ├── Exceptions
│   └── Enums
│
├── drift/
│   └── Drift detection components
│
├── ood/
│   └── Out-of-distribution detection
│
├── recommendation/
│   └── Reliability-based recommendations
│
├── scoring/
│   └── Trust-score calculation
│
├── uncertainty/
│   ├── Confidence estimation
│   └── Entropy estimation
│
└── wrappers/
    ├── Base model interface
    └── Scikit-learn wrapper
```

The public API is intentionally kept small while the reliability components remain modular.

---

## Reliability Pipeline

The core pipeline follows:

```text
Model
  │
  ▼
Prediction
  │
  ├───────────────┐
  ▼               ▼
Confidence     Probability
                  │
                  ▼
              Uncertainty
                  │
Input ────────────┤
  │               ▼
  └──────────► OOD Detection
                  │
                  ▼
             Trust Scoring
                  │
                  ▼
          Risk Classification
                  │
                  ▼
            Recommendation
                  │
                  ▼
          PredictionReport
```

---

## Roadmap

### v0.1 — ML Reliability Core

* [x] Scikit-learn wrapper
* [x] Confidence estimation
* [x] Entropy-based uncertainty
* [x] Z-score OOD detection
* [x] Trust aggregation
* [x] Risk classification
* [x] Recommendation engine
* [x] Structured prediction reports
* [x] Batch prediction
* [x] Tests and CI

### v0.2 — Extended Reliability

* [ ] Probability calibration
* [ ] Additional OOD detection strategies
* [ ] Regression reliability
* [ ] Improved model capability detection
* [ ] Expanded reliability configuration
* [ ] Reliability evaluation benchmarks

### v0.3 — Monitoring & AI Reliability

* [ ] Prediction logging
* [ ] Production drift monitoring
* [ ] Trust-score trend analysis
* [ ] Reliability monitoring
* [ ] LLM reliability adapters
* [ ] Safety evaluation modules

### v1.0 — Production Reliability Platform

* [ ] FastAPI service
* [ ] Docker deployment
* [ ] REST API
* [ ] Plugin architecture
* [ ] Production deployment guidance
* [ ] Comprehensive reliability benchmarks
* [ ] Monitoring integrations

---

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=aegis
```

AegisAI also uses GitHub Actions for automated CI across supported Python versions.

---

## Python Support

AegisAI currently targets:

```text
Python 3.10
Python 3.11
Python 3.12
Python 3.13
```

The package currently requires:

```text
Python >= 3.10, < 3.14
```

---

## Project Goals

AegisAI is not intended to replace machine-learning models.

Its goal is to answer a different question:

> **"How much should we trust this prediction?"**

The long-term goal is to provide a common reliability interface that can sit between machine-learning systems and applications that depend on their predictions.

```text
              ML Systems
                  │
                  ▼
             ┌─────────┐
             │ AegisAI │
             └────┬────┘
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
   Applications  APIs    Monitoring
```

---

## Contributing

Contributions are welcome.

Before making significant architectural changes, please open an issue to discuss the proposed change and maintain a coherent public API.

For smaller improvements:

1. Fork the repository.
2. Create a feature branch.
3. Add or update tests.
4. Run the test suite.
5. Submit a pull request.

---

## License

AegisAI is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## Repository

**GitHub:** https://github.com/arshavardhan/AegisAi

**AegisAI — making ML predictions more observable, explainable, and trustworthy.**
