r"""
phishing_baseline.py
=====================
Days 1-4: an HONEST baseline phishing-URL detector on PhiUSIIL (UCI id 967).

The same pipeline runs twice so the difference is visible:

  1. "all_features" -- every numeric column, as the original brief asked. It
     scores ~100%, which is a TRAP: diagnose.py shows several columns are
     answer keys (URLSimilarityIndex) or collection artifacts (empty phishing
     pages), not real signals.
  2. "url_only"     -- the 21 features computable from the URL string alone.
     The honest, attacker-relevant baseline: an attacker controls the URL, not
     how a crawler happened to capture the page.

For each one: stratified 70/15/15 split, logistic regression vs random forest
(chosen on VALIDATION), one final score on TEST, imbalance-aware metrics
(precision / recall / F1 / ROC-AUC + confusion matrix -- never accuracy as the
headline), and the forest's feature importances. For the url_only forest the
importances are double-checked with permutation importance, because the attack
stage is built on what this model relies on.

Run:
    .\.venv\Scripts\python.exe phishing_baseline.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (DEEMPHASIS, HIGHLIGHT, ID_LIKE_COLUMNS, INK, SEED, SERIES,
                    URL_ONLY_FEATURES, load_data, phish_proba, save_figure, save_table,
                    sequential_cmap, solo_auc, split, train_forest, use_chart_style)


def explain_accuracy(phish_rate):
    """The base-rate problem, in numbers from this dataset."""
    majority = max(phish_rate, 1 - phish_rate)
    print("\n" + "=" * 72)
    print("WHY NOT ACCURACY?  (the base-rate problem)")
    print("-" * 72)
    print(f"  Phishing is {phish_rate:.1%} of this data. A lazy model that ALWAYS says")
    print(f"  'legitimate' scores {majority:.1%} accuracy while catching ZERO phishing")
    print("  (recall = 0). Accuracy rewards guessing the common class, so a high number")
    print("  can hide a useless model. Precision / recall / F1 judge the phishing class")
    print("  itself, and ROC-AUC measures how well the model RANKS phishing above legit")
    print("  at every threshold. The imbalance here is mild; on real traffic, where")
    print("  phishing can be well under 1% of URLs, accuracy becomes almost meaningless.")
    print("=" * 72)


def evaluate(name, model, X, y, split_name, tag):
    """Imbalance-aware metrics + a saved confusion matrix."""
    pred = model.predict(X)
    metrics = {
        "precision": precision_score(y, pred),   # of URLs we flagged, how many were phishing
        "recall":    recall_score(y, pred),      # of real phishing, how many we caught
        "f1":        f1_score(y, pred),          # balance of the two
        "roc_auc":   roc_auc_score(y, phish_proba(model, X)),   # ranking, threshold-free
    }
    print(f"  [{name}] {split_name.upper()}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=["legitimate", "phishing"]).plot(
        ax=ax, cmap=sequential_cmap(), colorbar=False, values_format=",d")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"{name}, {tag.replace('_', ' ')} ({split_name})", fontsize=11)
    save_figure(fig, f"confusion_{name.lower()}_{tag}_{split_name}.png")
    return metrics


def run_baseline(X, y, tag, title):
    print("\n" + "#" * 72)
    print(f"# BASELINE: {title}   ({X.shape[1]} features)")
    print("#" * 72)
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
    print(f"Split -> train {len(X_train):,} | validation {len(X_val):,} | test {len(X_test):,}")

    # Logistic regression cares about feature SCALE, so it is standardised
    # INSIDE a pipeline: the scaler is fit on the training rows only, and the
    # validation/test rows never leak into it.
    logreg = Pipeline([("scale", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=1000, random_state=SEED))])
    logreg.fit(X_train, y_train)
    # A random forest splits on raw thresholds, so it needs no scaling.
    forest = train_forest(X_train, y_train)

    print("Validation (where we are allowed to compare and choose):")
    val = {"LogReg": evaluate("LogReg", logreg, X_val, y_val, "val", tag),
           "RandomForest": evaluate("RandomForest", forest, X_val, y_val, "val", tag)}
    best = max(val, key=lambda m: val[m]["roc_auc"])
    model = {"LogReg": logreg, "RandomForest": forest}[best]
    print(f"Chosen on validation (ROC-AUC): {best}. Opening TEST exactly once:")
    test = evaluate(best, model, X_test, y_test, "test", tag)

    rows = [{"baseline": tag, "model": m, "split": "val", **v} for m, v in val.items()]
    rows.append({"baseline": tag, "model": best, "split": "test", **test})
    mdi = pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False)
    return {"rows": rows, "forest": forest, "mdi": mdi, "X_val": X_val, "y_val": y_val,
            "best": best, "test": test}


def plot_all_features_importance(mdi):
    top = mdi.head(15)[::-1]                                   # largest at the top
    colours = [SERIES[0] if f in URL_ONLY_FEATURES else SERIES[1] for f in top.index]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.barh(top.index, top.to_numpy(), height=0.6, color=colours)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("importance (mean decrease in impurity)")
    ax.legend(handles=[Patch(color=SERIES[1], label="page content / similarity score"),
                       Patch(color=SERIES[0], label="URL string")], loc="lower right")
    ax.set_title("All-features model: the top signals are the leaky ones")
    save_figure(fig, "feature_importance_all_features.png")


def plot_url_only_importance(mdi, perm):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    panels = [(axes[0], mdi, "mean decrease in impurity", "Built-in importance (MDI)"),
              (axes[1], perm, "drop in F1 when the feature is shuffled", "Permutation importance")]
    for ax, values, xlabel, panel_title in panels:
        top = values.head(10)[::-1]
        colours = [HIGHLIGHT if f == "IsHTTPS" else DEEMPHASIS for f in top.index]
        ax.barh(top.index, top.to_numpy(), height=0.6, color=colours)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(xlabel)
        ax.set_title(panel_title, fontsize=11)
        ax.text(top.iloc[-1], len(top) - 1, f"  {top.iloc[-1]:.3f}", va="center",
                color=INK["secondary"], fontsize=9)
    fig.suptitle("What the URL-only model relies on  (IsHTTPS in blue)",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, "feature_importance_url_only.png")


def main():
    use_chart_style()
    X_raw, y = load_data()
    phish_rate = float(y.mean())
    print(f"Dataset: {len(X_raw):,} URLs -- {int((y == 0).sum()):,} legitimate, "
          f"{int(y.sum()):,} phishing ({phish_rate:.1%})")
    if phish_rate > 0.5:
        warnings.warn("Phishing is the majority -- check PHISHING_LABEL_IN_RAW in common.py.")

    # Drop identifiers / free text, keep numeric columns. (PhiUSIIL has nothing
    # left to encode after that, but we check so a change fails loudly.)
    X_all = X_raw.drop(columns=[c for c in ID_LIKE_COLUMNS if c in X_raw.columns])
    non_numeric = X_all.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"Dropping non-numeric columns: {non_numeric}")
        X_all = X_all.drop(columns=non_numeric)

    print("\nLEAKAGE CHECK -- columns that separate the classes on their own "
          "(AUC; full audit in diagnose.py):")
    print(solo_auc(X_all, y).head(5).round(3).to_string())
    explain_accuracy(phish_rate)

    leaky = run_baseline(X_all, y, "all_features",
                         "ALL features (the brief's baseline -- leaky, see diagnose.py)")
    honest = run_baseline(X_all[URL_ONLY_FEATURES], y, "url_only",
                          "URL-string only (honest, attacker-relevant)")

    # Double-check the url_only forest's importances with permutation importance
    # on validation data: shuffle one feature, see how much F1 drops.
    sample = honest["X_val"].sample(10_000, random_state=SEED)
    perm = permutation_importance(honest["forest"], sample, honest["y_val"].loc[sample.index],
                                  scoring="f1", n_repeats=5, random_state=SEED)
    perm = pd.Series(perm.importances_mean, index=sample.columns).sort_values(ascending=False)
    importances = pd.DataFrame({"mdi": honest["mdi"], "permutation_f1_drop": perm})
    importances = importances.sort_values("mdi", ascending=False)
    print("\nURL-only forest -- top features by both methods:")
    print(importances.head(8).round(4).to_string())

    plot_all_features_importance(leaky["mdi"])
    plot_url_only_importance(honest["mdi"], perm)
    save_table(pd.DataFrame(leaky["rows"] + honest["rows"]).round(4), "baseline_metrics.csv", index=False)
    save_table(importances.round(5), "feature_importance_url_only.csv", index_label="feature")
    save_table(leaky["mdi"].round(5).rename("mdi").to_frame(), "feature_importance_all_features.csv",
               index_label="feature")

    print("\n" + "=" * 72)
    print("SUMMARY -- the chosen model per baseline, on the held-out TEST set")
    print("-" * 72)
    for result, tag in [(leaky, "all_features"), (honest, "url_only")]:
        m = result["test"]
        print(f"  {tag:<13} {result['best']:<13} precision {m['precision']:.4f}  "
              f"recall {m['recall']:.4f}  F1 {m['f1']:.4f}  ROC-AUC {m['roc_auc']:.4f}")
    print("=" * 72)
    print("The all_features row is the trap; url_only is the baseline worth trusting.")
    print("Saved -> figures/ (confusion matrices, importances) and results/ (tables).")


if __name__ == "__main__":
    main()
