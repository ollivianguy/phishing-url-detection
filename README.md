# Phishing URL Detection — Baseline, Leakage Analysis & Adversarial Robustness

A machine-learning project that detects phishing URLs — and, more importantly,
interrogates *why* it works and how easily it can be fooled.

> **Status:** Baseline + data-leakage analysis **complete**. Adversarial attack
> and defence **in progress**.

---

## TL;DR

- Trained a logistic regression and a random forest on the **PhiUSIIL** dataset
  (235,795 URLs) to classify phishing vs. legitimate.
- Both models scored **~100%** — which turned out to be a **red flag, not a win**.
  The dataset is trivially separable because of **data leakage**.
- Diagnosed the cause: phishing pages were crawled nearly **empty** (0 images /
  0 scripts), and a single feature, `URLSimilarityIndex`, separates the classes
  almost perfectly on its own (AUC **0.996**).
- Rebuilt an **honest, URL-only baseline** using only features an attacker
  actually controls: **F1 0.997, ROC-AUC 0.999**.
- **Next:** attack the model (feature-space vs. problem-space evasion) and
  measure the robustness/accuracy trade-off of two defences.

---

## Why this project is about *honesty*, not leaderboard scores

A phishing detector that reports 99% accuracy sounds great until you realise the
data is imbalanced and the number is hollow. This project is built around three
ideas that matter more than a high score:

1. **The base-rate problem** — why accuracy is the wrong metric here.
2. **Data leakage** — why a "perfect" model can be learning the wrong thing.
3. **Adversarial robustness** — whether the model survives an attacker who edits
   the URL.

### 1. Why not accuracy?

The data is ~43% phishing, ~57% legitimate. A lazy model that *always* predicts
"legitimate" scores **57% accuracy while catching zero phishing** (recall = 0).
Accuracy just rewards guessing the majority class, so it can hide a useless
model. I report **precision, recall, F1, and ROC-AUC** instead, with phishing as
the positive class. (The effect is mild here but explodes on real traffic, where
phishing can be well under 1% of URLs.)

### 2. The leakage finding (the interesting part)

Ranking each feature by how well it separates the classes *on its own*
(single-feature ROC-AUC, where 1.00 = a perfect giveaway):

| feature | solo AUC |
|---|---|
| URLSimilarityIndex | 0.996 |
| LineOfCode | 0.990 |
| NoOfExternalRef | 0.988 |
| NoOfImage | 0.980 |
| NoOfJS | 0.971 |
| NoOfCSS | 0.958 |

Note `URLSimilarityIndex`: its linear *correlation* with the label is only 0.86,
so a correlation check waves it through — but its AUC is 0.996. **Single-feature
AUC is the right tool for spotting leakage.**

The page-content features are near-perfect separators because the phishing pages
were captured nearly empty:

| feature | mean (legit) | mean (phishing) | % of phishing rows = 0 |
|---|---|---|---|
| LineOfCode | 1947 | 66 | — |
| NoOfImage | 45 | 0.9 | 83% |
| NoOfJS | 18 | 0.9 | 77% |
| HasSocialNet | 0.79 | 0.01 | 99% |

So the model wasn't learning *"is this URL phishing"* — it was learning *"did the
crawler find a real web page."* That's an artifact of how the data was collected,
and it's leakage relative to the real task.

**Decision:** the honest baseline uses **URL-string-only features** — the part an
attacker actually controls — and excludes the leaky page-content signals.

---

## Results

Best model (random forest) on the held-out test set:

| baseline | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|
| All 50 features *(leaky)* | 1.000 | 1.000 | 1.000 | 1.000 |
| URL-string only *(honest)* | 0.998 | 0.995 | 0.997 | 0.999 |

The leaky baseline's perfect score is the trap; the URL-only row is the result I
actually trust.

## What the model relies on

![Random-forest feature importances (URL-only)](feature_importance_rf_url_only.png)

`IsHTTPS` alone carries **~37%** of the model's weight. That's a warning:
modern phishing routinely uses HTTPS (free certificates), and `IsHTTPS` is
trivial for an attacker to flip — making it the prime target for the adversarial
stage.

---

## How to run

Requires **Python 3.13**. From the project folder:

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe phishing_baseline.py

# macOS / Linux
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python phishing_baseline.py
```

The script downloads the dataset, trains both models for both feature sets,
prints the full report, and saves the confusion-matrix and feature-importance
PNGs next to itself. `diagnose.py` reproduces the leakage evidence tables above.

---

## Roadmap

- [x] Baseline detector (logistic regression + random forest)
- [x] Imbalance-aware metrics + base-rate writeup
- [x] Leakage detection and honest URL-only baseline
- [ ] **Failure-mode analysis** — characterise which phishing slip through and
      which legitimate URLs get flagged
- [ ] **Adversarial attack** — feature-space vs. problem-space evasion; measure
      how far recall drops when an attacker edits the URL
- [ ] **Defences** — adversarial training and feature hardening; quantify the
      robustness vs. clean-accuracy trade-off

## Limitations

- The dataset's collection bias (empty phishing pages; near-perfect `IsHTTPS`
  split) means even the URL-only baseline is likely optimistic vs. real traffic.
- Only two model families are compared.
- Feature importances use mean-decrease-in-impurity, which is biased toward
  high-cardinality features; permutation importance would be more trustworthy.

## Background reading

*(Read these before citing — don't cite what you haven't read.)*

- Pierazzi et al., *Intriguing Properties of Adversarial ML Attacks in the
  Problem Space* (2020) — the feature-space vs. problem-space distinction.
- Goodfellow et al., *Explaining and Harnessing Adversarial Examples* (2014) — FGSM.
- Szegedy et al., *Intriguing Properties of Neural Networks* (2013) — origin of
  adversarial examples.

## Repo contents

| File | Purpose |
|---|---|
| `phishing_baseline.py` | End-to-end pipeline: load → leakage check → train → evaluate → importances |
| `diagnose.py` | Standalone leakage-evidence script (the tables above) |
| `requirements.txt` | Pinned dependencies (Python 3.13 wheels) |
| `*.png` | Generated confusion matrices and feature-importance charts |
