r"""
attack.py
=========
Days 5-6: attack your own detector -- realistically.

The URL-only random forest catches ~99.5% of phishing. But an attacker controls
the URL. So: how easily can they slip past it -- and how much of the "evasion"
a naive analysis reports could actually happen?

The attacker gets five REAL moves (defined in common.py):
  * serve over HTTPS               free certificates make this cost nothing
  * add 'www.'                     just a subdomain of their own domain
  * a name without digits/hyphens  'pay-pal-24.top' -> 'paypal.top'
  * register as .com               instead of .top / .xyz / ...
  * page at the site root          no path, no query string
...but only on a domain they OWN. About half the phishing URLs here sit on
someone else's domain -- a hosting platform (web.app), an IPFS gateway or link
shortener (ipfs.io, bit.ly), a compromised site, a bare IP address -- and for
those only the HTTPS switch is allowed.

Each move edits the actual URL; the URL feature engine then recomputes all 21
features with the dataset's own rules (checked against all 235,795 rows), so the
model sees exactly what it would see for that real URL. That is a PROBLEM-SPACE
attack. We try it three ways, weakest to strongest:
  * one move at a time;
  * all five moves at once;
  * ADAPTIVE -- for each URL the attacker tries all 32 combinations and keeps
    whichever the model finds least suspicious, as a real attacker who can
    test against the detector would.

For contrast, a FEATURE-SPACE attack writes the same moves' intended effects
straight into the feature vector (HTTPS on, digits and symbols zeroed, length
and subdomain count set to typical legitimate values, TLD made .com) without
recomputing anything else. It is easy -- and it produces vectors no real URL
could have. We count how many break consistency rules that every real URL obeys.

All recall numbers use the shared operating point: the threshold that flags at
most 0.1% of legitimate VALIDATION URLs (common.TARGET_FPR).

Reference for the distinction: Pierazzi et al., "Intriguing Properties of
Adversarial ML Attacks in the Problem Space" (2020). Read it before citing it.

Run:
    .\.venv\Scripts\python.exe attack.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from common import (DEEMPHASIS, EDIT_LABELS, EDIT_ORDER, HIGHLIGHT, INK, SEED, TARGET_FPR,
                    URL_ONLY_FEATURES, URLFeatureEngine, adaptive_attack, combo_label,
                    constraint_violations, edit_combinations, load_data, phish_proba,
                    raw_features, save_figure, save_table, split, threshold_at_fpr,
                    train_forest, use_chart_style, wilson_interval)


def feature_space_perturb(rows, X_train, y_train, engine):
    """The same five moves, written straight into the features they target --
    without recomputing anything else they would change, and ignoring who owns
    the domain."""
    f = rows[URL_ONLY_FEATURES].astype(float).copy()
    legit = X_train[y_train.to_numpy() == 0]
    f["IsHTTPS"] = 1                                               # serve over HTTPS
    f["NoOfSubDomain"] = float(legit["NoOfSubDomain"].median())    # add 'www.'
    for col in ["NoOfDegitsInURL", "DegitRatioInURL",              # no digits or hyphens,
                "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL",
                "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL"]:   # no query string
        f[col] = 0.0
    f["URLLength"] = float(legit["URLLength"].median())            # short, like a site root
    f["TLDLength"] = 3.0                                           # register as .com
    f["TLDLegitimateProb"] = engine.tld_prob["com"]
    return f


def plot_attack(scenarios, feature_space_broken):
    labels = list(scenarios)[::-1]
    values = [scenarios[k] for k in labels]
    colours = [DEEMPHASIS if k.startswith("Feature-space") else HIGHLIGHT for k in labels]
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    ax.barh(labels, values, height=0.5, color=colours)
    for i, (label, v) in enumerate(zip(labels, values)):
        note = f"   {v:.1%}"
        if label.startswith("Feature-space"):
            note += f"    {feature_space_broken:.1%} of these vectors could not come from a real URL"
        ax.text(v, i, note, va="center", color=INK["secondary"], fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"phishing still caught (recall), at {TARGET_FPR:.1%} false alarms on legitimate URLs")
    ax.set_title("How much phishing the URL-only model still catches under attack")
    save_figure(fig, "attack_recall.png")


def main():
    use_chart_style()
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
    model = train_forest(X_train[URL_ONLY_FEATURES], y_train)

    # --- Operating point: at most 0.1% false alarms on legitimate VALIDATION URLs
    val_legit = X_val[y_val.to_numpy() == 0]
    threshold = threshold_at_fpr(phish_proba(model, val_legit[URL_ONLY_FEATURES]))
    test_legit = X_test[y_test.to_numpy() == 0]
    test_fpr = float((phish_proba(model, test_legit[URL_ONLY_FEATURES]) > threshold).mean())
    print(f"Operating point: flag when P(phishing) > {threshold:.3f} "
          f"(false alarms on TEST legitimate URLs: {test_fpr:.3%})")

    # --- The URL feature engine, and a check that its rules match the dataset
    engine = URLFeatureEngine(X_train)
    rules, ucp_corr = engine.check_rules(X)
    print(f"\nFeature rules reproduce the dataset's stored values (all {len(X):,} rows):")
    print(f"  {int((rules == 1).sum())} features on every row; all {len(rules)} on at least "
          f"{rules.min():.1%} of rows; URLCharProb approximated (r = {ucp_corr:.3f}).")
    save_table(rules.rename("share_of_rows_reproduced").round(5).to_frame(), "feature_rules_check.csv",
               index_label="feature")

    # --- Every combination of the five moves, for every phishing URL in TEST
    phish = X_test[y_test.to_numpy() == 1]
    n = len(phish)
    owned = engine.owned_share(phish)
    print(f"\nAttack set: {n:,} phishing URLs from the test set. The attacker owns the domain "
          f"for {owned:.1%} of them;\nthe rest sit on platforms, gateways, shorteners, compromised "
          f"sites or IP addresses\n({len(engine.shared):,} shared domains found), so only HTTPS applies there.")
    combos = edit_combinations()
    variants = engine.variants(phish, combos)

    def recall(frame):
        return float((phish_proba(model, frame[URL_ONLY_FEATURES]) > threshold).mean())

    clean = recall(variants[frozenset()][1])
    singles = {name: recall(variants[frozenset({name})][1]) for name in EDIT_ORDER}
    all_moves = recall(variants[frozenset(EDIT_ORDER)][1])
    caught, choice, best = adaptive_attack(model, raw_features, variants, combos, threshold)
    adaptive = float(caught.mean())
    fs_frame = feature_space_perturb(phish, X_train, y_train, engine)
    feature_space = recall(fs_frame)

    original_urls = variants[frozenset()][0]
    print("One move at a time (recall after the move):")
    for name in sorted(singles, key=singles.get):
        changed = np.mean([a != b for a, b in zip(original_urls, variants[frozenset({name})][0])])
        print(f"  {EDIT_LABELS[name]:<30} {singles[name]:.4f}   (applies to {changed:.0%} of URLs)")

    # --- Can these feature vectors exist? -----------------------------------
    stacked = np.stack([variants[c][1].to_numpy() for c in combos])
    adaptive_frame = pd.DataFrame(stacked[choice, np.arange(n)], index=phish.index,
                                  columns=URL_ONLY_FEATURES)
    checks = {
        "real phishing URLs": constraint_violations(phish[URL_ONLY_FEATURES]),
        "problem-space (all five moves)": constraint_violations(variants[frozenset(EDIT_ORDER)][1]),
        "problem-space (adaptive)": constraint_violations(adaptive_frame),
        "feature-space (same five moves)": constraint_violations(fs_frame),
    }
    print("\nShare of feature vectors breaking a rule that every real URL obeys:")
    for name, (_, rate) in checks.items():
        print(f"  {name:<34} {rate:.2%}")
    print("  Rules the feature-space vectors break most:")
    for rule, rate in checks["feature-space (same five moves)"][0].sort_values(ascending=False).head(3).items():
        print(f"    {rate:6.1%}  {rule}")
    save_table(pd.DataFrame({k: v[0] for k, v in checks.items()}).round(5),
               "attack_constraint_checks.csv", index_label="rule")

    # --- Headline --------------------------------------------------------------
    scenarios = {
        "No attack": clean,
        "HTTPS only": singles["https"],
        "All five moves at once": all_moves,
        "Adaptive (best of 32 per URL)": adaptive,
        "Feature-space (same five moves)": feature_space,
    }
    table = []
    for name, r in scenarios.items():
        lo, hi = wilson_interval(round(r * n), n)
        table.append({"scenario": name, "recall": r, "evasion": 1 - r, "ci95_low": lo, "ci95_high": hi})
    for name in EDIT_ORDER:
        lo, hi = wilson_interval(round(singles[name] * n), n)
        table.append({"scenario": f"single move: {EDIT_LABELS[name]}", "recall": singles[name],
                      "evasion": 1 - singles[name], "ci95_low": lo, "ci95_high": hi})
    table = pd.DataFrame(table)
    save_table(table.round(5), "attack_results.csv", index=False)
    widest = float((table["ci95_high"] - table["ci95_low"]).max() / 2)

    print("\n" + "=" * 70)
    print(f"HEADLINE -- phishing still caught at {TARGET_FPR:.1%} false alarms")
    print("-" * 70)
    for name, r in scenarios.items():
        print(f"  {name:<34} recall {r:.4f}   evasion {1 - r:.1%}")
    print(f"  (95% confidence intervals are within +/-{widest:.1%} -- {n:,} test URLs.)")
    print("=" * 70)

    # --- Which moves did successful adaptive evasions use? ----------------------
    evaded = ~caught
    used = pd.Series({EDIT_LABELS[e]: float(np.mean([e in combos[c] for c in choice[evaded]]))
                      for e in EDIT_ORDER}).sort_values(ascending=False)
    print(f"\nOf the {evaded.sum():,} URLs the adaptive attacker got through, the share using each move:")
    for label, share in used.items():
        print(f"  {label:<30} {share:.1%}")
    save_table(used.rename("share_of_successful_evasions").round(4).to_frame(), "attack_moves_used.csv",
               index_label="move")

    # --- Concrete examples: caught before, evading after --------------------------
    before = phish_proba(model, phish[URL_ONLY_FEATURES])
    flipped = np.flatnonzero((before > threshold) & evaded)
    pick = np.sort(np.random.default_rng(SEED).choice(flipped, size=min(15, len(flipped)), replace=False))
    examples = pd.DataFrame({
        "original_url": phish["URL"].to_numpy()[pick],
        "attacker_url": [variants[combos[choice[i]]][0][i] for i in pick],
        "moves": [combo_label(combos[choice[i]]) for i in pick],
        "p_phish_before": before[pick].round(3),
        "p_phish_after": best[pick].round(3),
    })
    save_table(examples, "attack_examples.csv", index=False)
    print(f"\n{len(flipped):,} URLs went from caught to evading. Examples -> results/attack_examples.csv")

    plot_attack(scenarios, checks["feature-space (same five moves)"][1])
    print("Plot -> figures/attack_recall.png")


if __name__ == "__main__":
    main()
