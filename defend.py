r"""
defend.py
=========
Days 7-8 deliverable: defend the detector, and measure what the defence COSTS.

attack.py showed a single free move -- serving phishing over HTTPS -- drops
recall from 0.995 to ~0.72. Here we try two defences and, crucially, measure
both sides of the ledger:

  * ADVERSARIAL TRAINING -- generate realistic (problem-space) disguised phishing
    from the TRAINING set, add them back in as phishing, and retrain. The model
    learns that a clean HTTPS URL can still be an attack.

  * FEATURE HARDENING -- delete the feature the attacker fakes for free
    (IsHTTPS) so the model physically cannot lean on it. Blunt, but it removes
    the lever entirely.

The point is NOT to "win". Robustness usually costs you accuracy on normal
traffic, and quantifying that trade-off is the mature result:
"making the model harder to fool made it slightly worse at its everyday job,
 and here is exactly how much."

Note: the perturbation helpers below mirror attack.py so each script reads
top-to-bottom on its own. In a larger project you would factor them into a
shared module.

Run:
    .\.venv\Scripts\python.exe defend.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

SEED = 42
PHISHING_LABEL_IN_RAW = 0
URL_ONLY_FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
    "CharContinuationRate", "TLDLegitimateProb", "URLCharProb",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
]

# The feature an attacker fakes for free. Hardening = refusing to use it.
# (Try adding more of the cheap ones here to see the trade-off move.)
HARDEN_DROP = ["IsHTTPS"]

# ---------------------------------------------------------------------------
# Load + same split as every other script (same SEED => same rows)
# ---------------------------------------------------------------------------
print("Loading data...")
data = fetch_ucirepo(id=967)
X_raw = data.data.features.copy()
urls = X_raw["URL"]
y = (pd.to_numeric(data.data.targets.iloc[:, 0], errors="coerce")
     == PHISHING_LABEL_IN_RAW).astype(int)
X = X_raw[URL_ONLY_FEATURES].copy()

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)


# ---------------------------------------------------------------------------
# Problem-space perturbation (same realistic attack as attack.py)
# ---------------------------------------------------------------------------
def string_counts(u):
    length = len(u)
    letters = sum(c.isalpha() for c in u)
    digits = sum(c.isdigit() for c in u)
    eq, qm, amp = u.count("="), u.count("?"), u.count("&")
    specials_all = sum(not c.isalnum() for c in u)
    return {
        "URLLength": length,
        "NoOfLettersInURL": letters,
        "NoOfDegitsInURL": digits,
        "NoOfEqualsInURL": eq,
        "NoOfQMarkInURL": qm,
        "NoOfAmpersandInURL": amp,
        "NoOfOtherSpecialCharsInURL": specials_all - eq - qm - amp,
        "IsHTTPS": int(u.lower().startswith("https")),
    }

def to_https(u):
    low = u.lower()
    if low.startswith("http://"):
        return "https://" + u[7:]
    if low.startswith("https://"):
        return u
    return "https://" + u

_KEEP = set(":/.?=&")

def drop_specials(u):
    return "".join(c for c in u if c.isalnum() or c in _KEEP)

def drop_digits(u):
    return "".join(c for c in u if not c.isdigit())

def realistic_bundle(u):
    return drop_specials(drop_digits(to_https(u)))


def perturb(Xsub, urls_sub):
    """Apply the realistic URL edit and move the features it really changes,
    keeping them mutually consistent (the problem-space constraint)."""
    Xnew = Xsub.copy()
    add = ["NoOfLettersInURL", "NoOfDegitsInURL", "NoOfEqualsInURL",
           "NoOfQMarkInURL", "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL"]
    for idx, u in urls_sub.items():
        o, n = string_counts(u), string_counts(realistic_bundle(u))
        new_len = max(1, Xsub.at[idx, "URLLength"] + (n["URLLength"] - o["URLLength"]))
        for f in add:
            Xnew.at[idx, f] = max(0, Xsub.at[idx, f] + (n[f] - o[f]))
        Xnew.at[idx, "URLLength"] = new_len
        Xnew.at[idx, "LetterRatioInURL"] = Xnew.at[idx, "NoOfLettersInURL"] / new_len
        Xnew.at[idx, "DegitRatioInURL"] = Xnew.at[idx, "NoOfDegitsInURL"] / new_len
        specials = (Xnew.at[idx, "NoOfEqualsInURL"] + Xnew.at[idx, "NoOfQMarkInURL"]
                    + Xnew.at[idx, "NoOfAmpersandInURL"]
                    + Xnew.at[idx, "NoOfOtherSpecialCharsInURL"])
        Xnew.at[idx, "SpacialCharRatioInURL"] = specials / new_len
        Xnew.at[idx, "IsHTTPS"] = n["IsHTTPS"]
    return Xnew


# ---------------------------------------------------------------------------
# Build the attacked TEST phishing (for measuring robustness) and the
# adversarial TRAINING examples (for the first defence).
# ---------------------------------------------------------------------------
print("Generating disguised phishing (problem-space)...")
test_phish = X_test[(y_test == 1).to_numpy()]
X_test_attacked = perturb(test_phish, urls.loc[test_phish.index])

train_phish = X_train[(y_train == 1).to_numpy()]
X_train_adv = perturb(train_phish, urls.loc[train_phish.index])

# Adversarially augmented training set: originals + disguised phishing.
X_train_aug = pd.concat([X_train, X_train_adv], ignore_index=True)
y_train_aug = pd.concat([y_train, pd.Series(1, index=X_train_adv.index)],
                        ignore_index=True)
print(f"Training set: {len(X_train):,} rows -> augmented {len(X_train_aug):,} rows")


# ---------------------------------------------------------------------------
# Train + evaluate each configuration on BOTH axes
# ---------------------------------------------------------------------------
ALL_FEATURES = URL_ONLY_FEATURES
HARDENED = [c for c in URL_ONLY_FEATURES if c not in HARDEN_DROP]

configs = {
    "baseline":              (ALL_FEATURES, X_train,     y_train),
    "adversarial training":  (ALL_FEATURES, X_train_aug, y_train_aug),
    "feature hardening":     (HARDENED,     X_train,     y_train),
    "both defences":         (HARDENED,     X_train_aug, y_train_aug),
}

results = {}
for name, (feats, Xtr, ytr) in configs.items():
    print(f"Training '{name}' ({len(feats)} features, {len(Xtr):,} rows)...")
    model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
    model.fit(Xtr[feats], ytr)

    # Axis 1: everyday performance on the clean test set.
    pred = model.predict(X_test[feats])
    # Axis 2: robustness -- recall on the disguised phishing.
    robust = model.predict(X_test_attacked[feats]).mean()

    results[name] = {
        "precision": precision_score(y_test, pred),
        "recall":    recall_score(y_test, pred),
        "f1":        f1_score(y_test, pred),
        "robust":    robust,
    }

# ---------------------------------------------------------------------------
# The trade-off table
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("THE TRADE-OFF -- everyday performance vs. resistance to the realistic attack")
print("-" * 78)
print(f"{'configuration':<24}{'clean P':>9}{'clean R':>9}{'clean F1':>10}"
      f"{'recall under attack':>21}")
for name, r in results.items():
    print(f"{name:<24}{r['precision']:>9.4f}{r['recall']:>9.4f}{r['f1']:>10.4f}"
          f"{r['robust']:>21.4f}")
print("=" * 78)

base = results["baseline"]
print("\nCost/benefit vs the undefended baseline:")
for name, r in results.items():
    if name == "baseline":
        continue
    d_f1 = r["f1"] - base["f1"]
    d_rob = r["robust"] - base["robust"]
    print(f"  {name:<22} clean F1 {d_f1:+.4f}   robustness {d_rob:+.4f}")
print("\n  (Negative F1 with positive robustness = you PAID accuracy for safety.")
print("   That sentence, with these numbers in it, is your Days 7-8 result.)")

# ---------------------------------------------------------------------------
# Trade-off plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.5))
colors = {"baseline": "#b03030", "adversarial training": "#e8a33d",
          "feature hardening": "#3b7dd8", "both defences": "#2e8b57"}
for name, r in results.items():
    ax.scatter(r["f1"], r["robust"], s=140, color=colors[name], zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.annotate(name, (r["f1"], r["robust"]), textcoords="offset points",
                xytext=(9, 6), fontsize=9)
ax.set_xlabel("clean-data F1  (everyday performance) →")
ax.set_ylabel("recall under realistic attack  (robustness) →")
ax.set_title("The robustness / accuracy trade-off")
ax.grid(alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig("defence_tradeoff.png", dpi=150)
plt.close(fig)
print("\nPlot saved -> defence_tradeoff.png")

pd.DataFrame(results).T.round(4).to_csv("defence_results.csv")
print("Numbers saved -> defence_results.csv")
