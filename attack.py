r"""
attack.py
=========
Days 5-6 deliverable: attack your own detector.

The url_only random forest gets ~99.5% recall on phishing. But an attacker
controls the URL. This script asks: how easily can they evade it, and how much
of that evasion is REAL vs. an artifact of attacking in the wrong space?

Two attacks, and the gap between them is the whole point:

  * FEATURE-SPACE attack (unconstrained): edit the feature VECTOR directly --
    set IsHTTPS=1, zero out the special-char and digit features, shrink the
    length. Easy, and it looks devastating. But it can produce IMPOSSIBLE
    vectors: e.g. a "length" that no longer matches the letters/digits it
    supposedly contains. No real URL could have these features.

  * PROBLEM-SPACE attack (realistic): edit the actual URL STRING, then
    recompute the features it changes -- so the features move together the way
    they must for a real URL. Only counts if the result is still a working,
    plausible phishing URL. This is the honest measure of evasion.

Reference for the distinction: Pierazzi et al., "Intriguing Properties of
Adversarial ML Attacks in the Problem Space" (2020). (Read before citing.)

Run:
    .\.venv\Scripts\python.exe attack.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

# ---------------------------------------------------------------------------
# Load data, keep URL strings, train the same url_only model as the baseline
# ---------------------------------------------------------------------------
print("Loading data and training the url_only model...")
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

rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
rf.fit(X_train, y_train)

# The attack set = the phishing URLs in the test set. We measure how many the
# model still catches (recall) after we disguise them.
phish_mask = (y_test == 1).to_numpy()
Xp = X_test[phish_mask].copy()
urls_p = urls.loc[Xp.index]

phish_col = list(rf.classes_).index(1)

def recall_of(frame):
    """All rows here are phishing, so mean(pred == phishing) is the recall."""
    return rf.predict(frame).mean()

baseline_recall = recall_of(Xp)
print(f"Attack set: {len(Xp):,} phishing URLs. Baseline recall = {baseline_recall:.4f}")


# ---------------------------------------------------------------------------
# A faithful URL-feature recomputation (only the string-derived features).
# We validate it against the dataset, then use it via DELTAS so any small,
# systematic offset between our extractor and the dataset's cancels out.
# ---------------------------------------------------------------------------
def string_counts(u):
    """Recompute the string-derived features from a URL. Returns a dict."""
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

# Sanity check: does our extractor track the dataset's stored values?
sample = urls_p.head(3000)
mine_len = pd.Series([string_counts(u)["URLLength"] for u in sample], index=sample.index)
corr_len = np.corrcoef(mine_len, Xp.loc[sample.index, "URLLength"])[0, 1]
print(f"Extractor check: our URLLength vs dataset -> correlation {corr_len:.3f} "
      f"(we use deltas, so a constant offset does not matter).")


# ---------------------------------------------------------------------------
# Realistic URL edits (problem-space "moves" an attacker can actually make)
# ---------------------------------------------------------------------------
def to_https(u):
    """Serve over HTTPS (any free certificate). Free and always realistic."""
    low = u.lower()
    if low.startswith("http://"):
        return "https://" + u[7:]
    if low.startswith("https://"):
        return u
    return "https://" + u

_KEEP = set(":/.?=&")   # keep URL-structure characters

def drop_specials(u):
    """Pick a cleaner-looking domain/path: no hyphens, underscores, etc."""
    return "".join(c for c in u if c.isalnum() or c in _KEEP)

def drop_digits(u):
    """Pick a digit-free lookalike domain/path."""
    return "".join(c for c in u if not c.isdigit())

def compose(*fns):
    def f(u):
        for fn in fns:
            u = fn(u)
        return u
    return f


def problem_space_perturb(edit):
    """Apply a URL-string edit, then move ONLY the features it really changes,
    keeping them mutually consistent. Features we can't faithfully recompute
    (e.g. TLDLegitimateProb) are held FIXED -- so this is a conservative,
    lower-bound estimate of real evasion. Returns the perturbed feature frame."""
    Xnew = Xp.copy()
    add = ["NoOfLettersInURL", "NoOfDegitsInURL", "NoOfEqualsInURL",
           "NoOfQMarkInURL", "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL"]
    for idx, u in urls_p.items():
        o, n = string_counts(u), string_counts(edit(u))
        new_len = max(1, Xp.at[idx, "URLLength"] + (n["URLLength"] - o["URLLength"]))
        for f in add:
            Xnew.at[idx, f] = max(0, Xp.at[idx, f] + (n[f] - o[f]))
        Xnew.at[idx, "URLLength"] = new_len
        # Re-derive ratios from the NEW counts -- this coupling is exactly what
        # a feature-space attack gets to ignore.
        Xnew.at[idx, "LetterRatioInURL"] = Xnew.at[idx, "NoOfLettersInURL"] / new_len
        Xnew.at[idx, "DegitRatioInURL"] = Xnew.at[idx, "NoOfDegitsInURL"] / new_len
        specials = (Xnew.at[idx, "NoOfEqualsInURL"] + Xnew.at[idx, "NoOfQMarkInURL"]
                    + Xnew.at[idx, "NoOfAmpersandInURL"]
                    + Xnew.at[idx, "NoOfOtherSpecialCharsInURL"])
        Xnew.at[idx, "SpacialCharRatioInURL"] = specials / new_len
        Xnew.at[idx, "IsHTTPS"] = n["IsHTTPS"]
    return Xnew


def feature_space_perturb():
    """Edit the feature VECTOR directly toward 'legit-looking' values with NO
    consistency constraint -- what an attacker CANNOT actually do. Returns the
    perturbed feature frame."""
    Xnew = Xp.copy()
    legit = X_train[y_train == 0]
    Xnew["IsHTTPS"] = 1
    for f in ["NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL",
              "NoOfDegitsInURL", "DegitRatioInURL",
              "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL"]:
        Xnew[f] = 0
    Xnew["URLLength"] = legit["URLLength"].median()
    Xnew["NoOfSubDomain"] = legit["NoOfSubDomain"].median()
    return Xnew


# ---------------------------------------------------------------------------
# Run the attacks
# ---------------------------------------------------------------------------
print("\nRunning attacks (the problem-space loops take a few seconds)...")
realistic_bundle = compose(to_https, drop_digits, drop_specials)

single_moves = {
    "https only":         recall_of(problem_space_perturb(to_https)),
    "drop digits only":   recall_of(problem_space_perturb(drop_digits)),
    "drop specials only": recall_of(problem_space_perturb(drop_specials)),
}
X_realistic = problem_space_perturb(realistic_bundle)
recall_realistic = recall_of(X_realistic)
recall_featurespace = recall_of(feature_space_perturb())

print("\nWhich single REALISTIC move hurts the model most? (recall after it)")
for name, r in sorted(single_moves.items(), key=lambda kv: kv[1]):
    print(f"    {name:<20}: recall {r:.4f}   (evasion {1 - r:.1%})")

# ---------------------------------------------------------------------------
# Headline comparison + plot
# ---------------------------------------------------------------------------
labels = ["no attack", "problem-space\n(https only)",
          "problem-space\n(realistic bundle)", "feature-space\n(unconstrained)"]
recalls = [baseline_recall, single_moves["https only"],
           recall_realistic, recall_featurespace]

print("\n" + "=" * 64)
print("HEADLINE -- phishing detection rate (recall) under each attack")
print("-" * 64)
for lab, r in zip([l.replace("\n", " ") for l in labels], recalls):
    print(f"    {lab:<34}: {r:.4f}   (evasion {1 - r:.1%})")
print("=" * 64)

fig, ax = plt.subplots(figsize=(8, 4.8))
colors = ["#3b7dd8", "#e8a33d", "#d9633b", "#b03030"]
bars = ax.bar(labels, recalls, color=colors)
ax.set_ylabel("phishing detection rate (recall)")
ax.set_ylim(0, 1.05)
ax.set_title("How far recall drops: realistic vs. unconstrained attacks")
for b, r in zip(bars, recalls):
    ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.2f}",
            ha="center", va="bottom", fontsize=10)
fig.tight_layout()
fig.savefig("attack_recall.png", dpi=150)
plt.close(fig)
print("\nPlot saved -> attack_recall.png")

# ---------------------------------------------------------------------------
# Concrete before/after examples (full realistic bundle) for the writeup
# ---------------------------------------------------------------------------
proba_before = rf.predict_proba(Xp)[:, phish_col]
proba_after = rf.predict_proba(X_realistic)[:, phish_col]
ex = pd.DataFrame({
    "url": urls_p.to_numpy(),
    "cleaned_url": [realistic_bundle(u) for u in urls_p.to_numpy()],
    "proba_before": proba_before.round(3),
    "proba_after": proba_after.round(3),
}, index=Xp.index)
# Keep the clean wins: caught before (>0.5), evades after (<0.5).
evaded = ex[(ex["proba_before"] > 0.5) & (ex["proba_after"] < 0.5)]
evaded.head(15).to_csv("attack_examples.csv", index=False)
print(f"{len(evaded):,} phishing URLs flipped from caught to evading under the "
      f"realistic bundle. Samples -> attack_examples.csv")

print("\nNow WRITE the key paragraph: the feature-space attack looks catastrophic,")
print("but much of it is unrealisable. The problem-space number is the honest one,")
print("and 'https only' shows a single free move an attacker really has.")
