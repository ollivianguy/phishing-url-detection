r"""
defend.py
=========
Days 7-8: defend the detector -- and measure exactly what each defence costs.

attack.py showed a realistic, adaptive attacker gets roughly half of all
phishing past the URL-only model, mostly with two free moves: HTTPS and 'www.'.
Three defences, each a different idea:

  * ADVERSARIAL TRAINING -- disguise the TRAINING phishing with the moves the
    defender anticipates (HTTPS, a cleaner name), keep each URL's most evasive
    version, add those back as phishing, and retrain.
  * DELETE THE COLUMN -- drop IsHTTPS, the feature the attacker fakes for free.
  * REMOVE THE INFORMATION -- "prefix-blind" features: drop IsHTTPS AND measure
    every length and ratio without 'http(s)://www.'. Deleting the column is not
    enough, because the dataset counts characters after the scheme but measures
    URLLength with it, so the scheme can be rebuilt from the other features.

Evaluated honestly, which means three things:
  * every model is set to the SAME operating point (at most 0.1% false alarms
    on legitimate validation URLs), so no model can buy "robustness" just by
    flagging more of everything;
  * the attacker ADAPTS to each defended model -- tries all 32 move
    combinations against that model -- instead of replaying the attack that
    beat the original;
  * adversarial training only ever sees two of the five moves; the other three
    are held back to test whether it learned a general lesson or memorised
    specific tricks.

Run:
    .\.venv\Scripts\python.exe defend.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_auc_score

from common import (DEEMPHASIS, EDIT_LABELS, EDIT_ORDER, INK, SERIES, TARGET_FPR,
                    URL_ONLY_FEATURES, URLFeatureEngine, edit_combinations, load_data,
                    phish_proba, prefix_blind, raw_features, save_figure, save_table, split,
                    threshold_at_fpr, train_forest, use_chart_style)

ANTICIPATED = ("https", "clean_domain")      # the only moves adversarial training sees


def without_https_column(features, urls):
    return features[[c for c in URL_ONLY_FEATURES if c != "IsHTTPS"]]


def adversarial_examples(model, engine, train_phish, combos):
    """For each training phishing URL, its most evasive disguise (against the
    original model) using only the anticipated moves."""
    variants = engine.variants(train_phish, combos)
    scores = np.vstack([phish_proba(model, variants[c][1][URL_ONLY_FEATURES]) for c in combos])
    pick = scores.argmin(axis=0)
    stacked = np.stack([variants[c][1].to_numpy() for c in combos])
    rows = pd.DataFrame(stacked[pick, np.arange(len(train_phish))], columns=URL_ONLY_FEATURES)
    original = train_phish["URL"].astype(str).to_numpy()
    changed = np.array([variants[combos[p]][0][i] != original[i] for i, p in enumerate(pick)])
    return rows[changed].reset_index(drop=True)      # unchanged URLs would just be duplicates


def plot_defences(table):
    order = table.index[::-1]                          # baseline at the top
    groups = [("recall_no_attack", "no attack", DEEMPHASIS),
              ("recall_https_only", "HTTPS only", SERIES[2]),
              ("recall_adaptive_anticipated", "adaptive, anticipated moves only", SERIES[1]),
              ("recall_adaptive_all", "adaptive, all five moves", SERIES[0])]
    height = 0.19
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    for j, (col, _, colour) in enumerate(groups):          # top to bottom: escalating attacks
        ys = np.arange(len(order)) + (1.5 - j) * (height + 0.02)
        ax.barh(ys, table.loc[order, col], height=height, color=colour)
        if col == "recall_adaptive_all":               # label the honest number only
            for yv, v in zip(ys, table.loc[order, col]):
                ax.text(v, yv, f"  {v:.1%}", va="center", color=INK["primary"], fontsize=9)
    ax.set_yticks(np.arange(len(order)), order)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"phishing caught (recall), every model at {TARGET_FPR:.1%} false alarms")
    ax.legend(handles=[Patch(color=c, label=lab) for _, lab, c in groups],
              loc="upper left", bbox_to_anchor=(1.0, 1.0), title="attack", title_fontsize=9)
    ax.set_title("Each defence against an attacker who adapts to it")
    save_figure(fig, "defence_robustness.png")


def main():
    use_chart_style()
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
    engine = URLFeatureEngine(X_train)
    combos = edit_combinations()
    anticipated = [c for c in combos if c <= set(ANTICIPATED)]   # incl. "no edit"

    # --- Why deleting the IsHTTPS column can't be enough -------------------------
    counted = X[["NoOfLettersInURL", "NoOfDegitsInURL", "NoOfEqualsInURL", "NoOfQMarkInURL",
                 "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL"]].sum(axis=1)
    gap = X["URLLength"] - counted     # the prefix: 'http://'=7, 'https://'=8, +'www.'=11 / 12
    is_prefix = float(gap.isin([7, 8, 11, 12]).mean())
    rebuilt = float((gap.isin([8, 12]).astype(int) == X["IsHTTPS"]).mean())
    print(f"URLLength minus the counted characters is exactly the prefix length (7, 8, 11 or 12)\n"
          f"for {is_prefix:.1%} of URLs -- so 'is that gap 8 or 12?' recovers IsHTTPS for {rebuilt:.1%}\n"
          f"of them, even after the IsHTTPS column is deleted.\n")
    save_table(pd.Series({"gap_equals_prefix_length": is_prefix, "https_recovered_from_gap": rebuilt})
               .round(4).to_frame("value"), "prefix_leak.csv", index_label="metric")

    baseline = train_forest(X_train[URL_ONLY_FEATURES], y_train)

    # --- Adversarial training data (from TRAINING rows only) ------------------
    print("Generating adversarial training examples (anticipated moves only)...")
    train_phish = X_train[y_train.to_numpy() == 1]
    adv = adversarial_examples(baseline, engine, train_phish, [c for c in anticipated if c])
    X_aug = pd.concat([X_train[URL_ONLY_FEATURES].astype(float), adv], ignore_index=True)
    y_aug = pd.concat([y_train.reset_index(drop=True), pd.Series(np.ones(len(adv), dtype=int))],
                      ignore_index=True)
    print(f"  {len(adv):,} disguised phishing URLs added: {len(X_train):,} -> {len(X_aug):,} training rows")

    train_urls = X_train["URL"].astype(str).tolist()
    models = {"Baseline": (raw_features, baseline)}
    print("Training 'Adversarial training'...")
    models["Adversarial training"] = (raw_features, train_forest(X_aug, y_aug))
    for name, featurize in [("Delete IsHTTPS column", without_https_column),
                            ("Prefix-blind features", prefix_blind)]:
        print(f"Training '{name}'...")
        models[name] = (featurize, train_forest(featurize(X_train[URL_ONLY_FEATURES], train_urls), y_train))

    # --- The attacker adapts to each model --------------------------------------
    test_phish = X_test[y_test.to_numpy() == 1]
    test_legit = X_test[y_test.to_numpy() == 0]
    val_legit = X_val[y_val.to_numpy() == 0]
    variants = engine.variants(test_phish, combos)
    idx = {c: i for i, c in enumerate(combos)}

    def score(model, featurize, rows):
        return phish_proba(model, featurize(rows[URL_ONLY_FEATURES], rows["URL"].astype(str).tolist()))

    rows, moves_used = {}, {}
    for name, (featurize, model) in models.items():
        threshold = threshold_at_fpr(score(model, featurize, val_legit))
        scores = np.vstack([phish_proba(model, featurize(variants[c][1], variants[c][0]))
                            for c in combos])
        best_anticipated = scores[[idx[c] for c in anticipated]].min(axis=0)
        choice = scores.argmin(axis=0)
        best_all = scores[choice, np.arange(scores.shape[1])]
        evaded = best_all <= threshold
        moves_used[name] = {EDIT_LABELS[e]: float(np.mean([e in combos[c] for c in choice[evaded]]))
                            if evaded.any() else 0.0 for e in EDIT_ORDER}
        legit_scores = score(model, featurize, test_legit)
        rows[name] = {
            "features": featurize(test_phish[URL_ONLY_FEATURES].head(1), test_phish["URL"].head(1).tolist()).shape[1],
            "training_rows": len(X_aug) if name == "Adversarial training" else len(X_train),
            "threshold": threshold,
            "test_false_alarm_rate": float((legit_scores > threshold).mean()),
            "roc_auc_clean": roc_auc_score(y_test, score(model, featurize, X_test)),
            "recall_no_attack": float((scores[idx[frozenset()]] > threshold).mean()),
            "recall_https_only": float((scores[idx[frozenset({"https"})]] > threshold).mean()),
            "recall_adaptive_anticipated": float((best_anticipated > threshold).mean()),
            "recall_adaptive_all": float((best_all > threshold).mean()),
            # The same model at the naive default threshold (0.5), for comparison:
            "default_0.5_recall_adaptive_all": float((best_all > 0.5).mean()),
            "default_0.5_false_alarm_rate": float((legit_scores > 0.5).mean()),
        }
    table = pd.DataFrame(rows).T.astype(float)
    base = table.loc["Baseline"]
    table["cost_clean_recall"] = table["recall_no_attack"] - base["recall_no_attack"]
    table["gain_vs_adaptive_attacker"] = table["recall_adaptive_all"] - base["recall_adaptive_all"]

    print("\n" + "=" * 98)
    print(f"EVERY MODEL AT {TARGET_FPR:.1%} FALSE ALARMS -- recall on {len(test_phish):,} test phishing URLs")
    print("-" * 98)
    print(f"{'model':<24}{'no attack':>10}{'HTTPS only':>12}{'adaptive:':>12}{'adaptive:':>11}"
          f"{'clean cost':>12}{'false alarms':>14}")
    print(f"{'':<24}{'':>10}{'':>12}{'anticipated':>12}{'all moves':>11}{'(recall)':>12}{'(test)':>14}")
    for name, r in table.iterrows():
        print(f"{name:<24}{r['recall_no_attack']:>10.4f}{r['recall_https_only']:>12.4f}"
              f"{r['recall_adaptive_anticipated']:>12.4f}{r['recall_adaptive_all']:>11.4f}"
              f"{r['cost_clean_recall']:>+12.4f}{r['test_false_alarm_rate']:>14.3%}")
    print("=" * 98)
    print("\nThe same models at the naive default threshold of 0.5 (NOT a fair comparison):")
    for name, r in table.iterrows():
        print(f"  {name:<24} adaptive recall {r['default_0.5_recall_adaptive_all']:.4f}   "
              f"false alarms {r['default_0.5_false_alarm_rate']:.3%}")

    used = pd.DataFrame(moves_used).T
    print("\nWhat the adaptive attacker relied on against each model (share of its successes):")
    print(used.round(3).to_string())

    save_table(table.round(5), "defence_results.csv", index_label="model")
    save_table(used.round(4), "defence_moves_used.csv", index_label="model")
    plot_defences(table)
    print("\nSaved -> results/defence_results.csv, results/defence_moves_used.csv, "
          "figures/defence_robustness.png")


if __name__ == "__main__":
    main()
