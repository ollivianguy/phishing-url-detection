"""
phishing_baseline.py
=====================
Baseline phishing-URL classifier on the PhiUSIIL dataset (UCI id=967).

This is the BASELINE stage of a larger project. The goal is an HONEST,
understandable baseline that the next (adversarial) stage can build on.

The script runs the SAME pipeline twice, so you can see the difference:

  1. "all_features"  -- every numeric feature, exactly as the assignment asks.
                        This scores ~100%, which is a TRAP, not a win (see the
                        leakage report it prints).
  2. "url_only"      -- only features derivable from the URL string. This is the
                        honest, attacker-relevant baseline: an attacker controls
                        the URL, not whether your crawler found page content.

Why the all-features version cheats (the leakage report shows this):
  * 'URLSimilarityIndex' separates the classes almost perfectly by itself
    (single-feature AUC ~0.996). Its linear *correlation* is only ~0.86, so a
    correlation check misses it -- we rank by single-feature AUC instead.
  * Page-content features (NoOfImage, NoOfJS, LineOfCode, ...) are ~0 for
    phishing rows: those pages were captured nearly EMPTY, while legit pages are
    full of content. So the model learns "did the crawler find a real page",
    not "is this URL phishing". That is a data-collection artifact -- leakage
    relative to the real task.

What the script does for EACH baseline:
  * stratified 70/15/15 split, train logistic regression + random forest,
  * evaluate with imbalance-aware metrics (precision/recall/F1/ROC-AUC +
    confusion matrix) -- NOT accuracy,
  * print the random forest's top-15 features (the signals for the next stage).

Run it with the Python 3.13 interpreter (see requirements.txt):

    python phishing_baseline.py

It prints a report and saves PNGs next to this file (one set per baseline).
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# Force matplotlib's "Agg" backend BEFORE importing pyplot. Agg draws to image
# files instead of opening a window, so this works on any machine (even one
# with no screen).
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Config -- every knob lives here, so the experiment is easy to reproduce.
# ---------------------------------------------------------------------------
SEED = 42                               # fixes split + models => identical result every run
HERE = Path(__file__).resolve().parent  # save PNGs next to this script
np.random.seed(SEED)

# PhiUSIIL encodes the label as 0 = phishing, 1 = legitimate (verified against
# the printed counts). We REMAP to 1 = phishing so phishing is the POSITIVE
# class everywhere -- it is what we want to catch, so precision/recall describe it.
PHISHING_LABEL_IN_RAW = 0

# Identifiers / free text: memorizing one specific URL or page title is not a
# generalizable signal, so we drop these up front.
ID_LIKE_COLUMNS = ["FILENAME", "URL", "Domain", "TLD", "Title"]

# Features computable from the raw URL string alone -- no page fetch, and the
# part an attacker actually controls. (URLSimilarityIndex is deliberately NOT
# here: it is a near-perfect separator, AUC ~0.996, and is computed against
# known-legitimate URLs, so we treat it as leakage.)
URL_ONLY_FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
    "CharContinuationRate", "TLDLegitimateProb", "URLCharProb",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
]

# A feature whose single-feature AUC is at/above this is flagged as a leakage
# or artifact suspect (it nearly determines the label by itself).
LEAKAGE_AUC_THRESHOLD = 0.98


# ---------------------------------------------------------------------------
# Helper: load the dataset
# ---------------------------------------------------------------------------
def load_data():
    print("Loading PhiUSIIL dataset from UCI (downloads a few MB the first time)...")
    data = fetch_ucirepo(id=967)
    X = data.data.features.copy()       # everything except the label
    y = data.data.targets.copy()        # a 1-column DataFrame holding 'label'
    y = pd.to_numeric(y.iloc[:, 0], errors="coerce")  # squeeze to a numeric Series
    return X, y


# ---------------------------------------------------------------------------
# Helper: leakage report by SINGLE-FEATURE AUC
# ---------------------------------------------------------------------------
def report_separability(X_numeric, y01):
    """Rank features by how well each ALONE separates the classes (ROC-AUC).

    We use single-feature ROC-AUC instead of correlation because a feature can
    split the classes almost perfectly while having only modest *linear*
    correlation. That is exactly the PhiUSIIL trap: 'URLSimilarityIndex' has
    correlation ~0.86 with the label but a single-feature AUC of ~0.996 -- it
    nearly *is* the answer, and a correlation check would let it through.
    """
    aucs = {}
    for col in X_numeric.columns:
        auc = roc_auc_score(y01, X_numeric[col])
        aucs[col] = max(auc, 1.0 - auc)   # direction-agnostic separability
    sep = pd.Series(aucs).sort_values(ascending=False)

    print("\nLEAKAGE CHECK -- top 10 features by SINGLE-FEATURE AUC "
          "(1.00 = perfect giveaway):")
    print(sep.head(10).to_string())

    flagged = sep[sep >= LEAKAGE_AUC_THRESHOLD].index.tolist()
    n_strong = int((sep >= 0.95).sum())
    if flagged:
        print(f"\n  !! WARNING (solo-AUC >= {LEAKAGE_AUC_THRESHOLD}): {flagged}")
        print(f"     {n_strong} features individually score >= 0.95. When a single")
        print( "     feature separates the classes this well, the model is reading an")
        print( "     answer key, not learning. In PhiUSIIL these are mostly PAGE-CONTENT")
        print( "     features: phishing pages were captured nearly empty (0 images / 0")
        print( "     JS / few lines of code), so 'has real content' ~ 'legitimate'.")
        print( "     => The 'all_features' baseline below inherits this leakage; the")
        print( "        'url_only' baseline avoids it.")
    return flagged


# ---------------------------------------------------------------------------
# Helper: why accuracy is the wrong headline (printed once)
# ---------------------------------------------------------------------------
def explain_accuracy(phish_rate):
    majority_acc = max(1 - phish_rate, phish_rate)
    print("\n" + "=" * 70)
    print("WHY NOT ACCURACY?  (the base-rate problem)")
    print("-" * 70)
    print(f"  Phishing is {phish_rate:.1%} of the data. A lazy model that ALWAYS")
    print(f"  predicts the majority class scores {majority_acc:.1%} accuracy while")
    print( "  catching ZERO phishing (recall = 0). Accuracy just rewards guessing")
    print( "  the common class, so a high number can hide a useless model.")
    print( "  Precision / recall / F1 judge the phishing class specifically, and")
    print( "  ROC-AUC measures how well the model RANKS phishing above legit across")
    print( "  every threshold. Here the imbalance is mild, but the effect explodes")
    print( "  on real traffic where phishing can be well under 1%.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Helper: evaluate one model on one split (metrics + confusion-matrix PNG)
# ---------------------------------------------------------------------------
def evaluate(name, model, X_split, y_split, split_name, tag, out_dir):
    """Compute imbalance-aware metrics and save a confusion-matrix image."""
    y_pred = model.predict(X_split)

    # Probability of the PHISHING class. We look up which probability column is
    # label 1 rather than assuming its position -- a safe habit.
    phish_col = list(model.classes_).index(1)
    y_proba = model.predict_proba(X_split)[:, phish_col]

    metrics = {
        "precision": precision_score(y_split, y_pred),  # of URLs we FLAGGED, how many were really phishing
        "recall":    recall_score(y_split, y_pred),     # of REAL phishing, how many we caught
        "f1":        f1_score(y_split, y_pred),          # balance of precision & recall
        "roc_auc":   roc_auc_score(y_split, y_proba),    # ranking quality, threshold-independent
    }

    print(f"  [{name}] on {split_name.upper()}: "
          f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  "
          f"f1={metrics['f1']:.4f}  roc_auc={metrics['roc_auc']:.4f}")

    # Confusion matrix: rows = truth, cols = prediction.
    cm = confusion_matrix(y_split, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["legit (0)", "phishing (1)"])
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} -- {tag} -- {split_name}")
    fig.tight_layout()
    fname = f"confusion_{name.lower().replace(' ', '_')}_{tag}_{split_name}.png"
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)
    return metrics


# ---------------------------------------------------------------------------
# One full baseline: split -> train 2 models -> compare on val -> test winner
#                     -> top-15 feature importances. Called once per feature set.
# ---------------------------------------------------------------------------
def run_baseline(X, y, tag, title, out_dir):
    print("\n" + "#" * 70)
    print(f"# BASELINE: {title}   ({X.shape[1]} features)")
    print("#" * 70)

    # Stratified 70/15/15. 'stratify=y' keeps the SAME phishing rate in every
    # split. We split 70/30, then halve the 30 -> 15/15.
    #
    # WHY a test set we never tune on: every time you look at a set and change
    # the model because of what you saw, the model starts fitting to THAT data.
    # The test set is opened exactly ONCE, at the very end, so its score honestly
    # estimates performance on brand-new URLs. (We pick the better model using
    # VALIDATION, never the test set.)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    print(f"Split -> train {len(X_train):,} | val {len(X_val):,} | test {len(X_test):,}")

    # Model 1: Logistic Regression (simple linear baseline). Linear models care
    # about feature SCALE, so we standardize INSIDE a Pipeline -- that way the
    # scaler is fit on the training data only and val/test never leak into it.
    logreg = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        # class_weight="balanced" is a knob to try later if recall is poor.
    ])
    logreg.fit(X_train, y_train)

    # Model 2: Random Forest (non-linear ensemble). Trees split on raw
    # thresholds, so no scaling is needed; the forest also gives us feature
    # importances for free.
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)

    # Compare BOTH models on validation (where we are allowed to make decisions).
    print("Validation results (both models):")
    val_lr = evaluate("LogReg", logreg, X_val, y_val, "val", tag, out_dir)
    val_rf = evaluate("RandomForest", rf, X_val, y_val, "val", tag, out_dir)

    # Pick the winner on validation (by ROC-AUC, which ignores the 0.5 cutoff),
    # then open the held-out TEST set exactly once for that model.
    if val_rf["roc_auc"] >= val_lr["roc_auc"]:
        best_name, best_model = "RandomForest", rf
    else:
        best_name, best_model = "LogReg", logreg
    print(f"Best on validation (ROC-AUC): {best_name}. Touching TEST once:")
    test_metrics = evaluate(best_name, best_model, X_test, y_test, "test", tag, out_dir)

    # Random-forest feature importance (MDI = mean decrease in impurity). Quick
    # and free, but biased toward features with many distinct values. For the
    # adversarial stage, treat this as a first look and consider sklearn's
    # permutation_importance for a more trustworthy ranking.
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top15 = importances.sort_values(ascending=False).head(15)
    print("Top 15 random-forest features (MDI importance):")
    print(top15.to_string())

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    top15.iloc[::-1].plot.barh(ax=ax, color="#3b7dd8")  # reverse -> #1 on top
    ax.set_title(f"RF top-15 importances -- {title}")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(out_dir / f"feature_importance_rf_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"Saved: confusion_*_{tag}_*.png  and  feature_importance_rf_{tag}.png")

    return {"tag": tag, "best": best_name, "test": test_metrics}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    # === Section 1: Load ===================================================
    X_raw, y_raw = load_data()
    print(f"\nDataset: {X_raw.shape[0]:,} rows x {X_raw.shape[1]} feature columns")

    # === Section 2a: Label -- remap so 1 = phishing (the positive class) ===
    y = (y_raw == PHISHING_LABEL_IN_RAW).astype(int)
    y.name = "is_phishing"
    counts = y.value_counts().sort_index()
    n_legit, n_phish = int(counts.get(0, 0)), int(counts.get(1, 0))
    phish_rate = n_phish / len(y)
    print(f"Class balance -> legit(0): {n_legit:,}   phishing(1): {n_phish:,}   "
          f"(phishing = {phish_rate:.1%})")
    if n_phish > n_legit:   # in PhiUSIIL phishing is the minority; warn if not
        warnings.warn("Phishing came out as the majority -- double-check "
                      "PHISHING_LABEL_IN_RAW against the counts above.")

    # === Section 2b: Drop identifier / free-text columns ===================
    present_ids = [c for c in ID_LIKE_COLUMNS if c in X_raw.columns]
    X_all = X_raw.drop(columns=present_ids)
    print(f"Dropped id/text columns: {present_ids}")

    # Keep only numeric columns. PhiUSIIL has nothing left to encode once the
    # text columns are gone, but we check so the script fails loudly if that
    # ever changes instead of crashing deep inside a model.
    non_numeric = X_all.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"Note: dropping remaining non-numeric columns: {non_numeric}")
        X_all = X_all.drop(columns=non_numeric)

    # === Section 2c: Leakage report + base-rate note (printed once) ========
    report_separability(X_all, y)
    explain_accuracy(phish_rate)

    # === Section 3-5: Run BOTH baselines ===================================
    results = []
    # (1) Exactly as the assignment asks: all numeric features, leakage WARNED
    #     about above but not removed. Expect a misleading ~100%.
    results.append(run_baseline(
        X_all, y, "all_features",
        "ALL features (assignment baseline -- leaky, see warning)", HERE))
    # (2) The honest, attacker-relevant baseline: URL-string features only.
    url_present = [c for c in URL_ONLY_FEATURES if c in X_all.columns]
    missing = [c for c in URL_ONLY_FEATURES if c not in X_all.columns]
    if missing:
        print(f"\n(note: expected URL features absent from data: {missing})")
    results.append(run_baseline(
        X_all[url_present].copy(), y, "url_only",
        "URL-string-only (honest, attacker-relevant baseline)", HERE))

    # === Section 6: Side-by-side summary ===================================
    print("\n" + "=" * 78)
    print("SUMMARY -- best model per baseline, on the held-out TEST set")
    print("-" * 78)
    print(f"{'baseline':<14}{'model':<14}{'precision':>10}{'recall':>9}{'f1':>9}{'roc_auc':>9}")
    for r in results:
        m = r["test"]
        print(f"{r['tag']:<14}{r['best']:<14}{m['precision']:>10.4f}{m['recall']:>9.4f}"
              f"{m['f1']:>9.4f}{m['roc_auc']:>9.4f}")
    print("=" * 78)
    print("Read the all_features row WITH the leakage warning in mind: its near-")
    print("perfect score is the trap. The url_only row is the honest baseline, and")
    print("its top-15 features are the signals to attack in the adversarial stage.")
    print("\nDone. PNGs saved next to this script (one set per baseline).")


if __name__ == "__main__":
    main()
